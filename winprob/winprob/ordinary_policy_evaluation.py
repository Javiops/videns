"""Target-independent reranking and paired ordinary-decision evaluation.

The source owns candidate generation and both economic audits. This module
only reorders unchanged options. Main agreement preserves the frozen M2
ward-only convention; typed and complete-purchase metrics remain separate.
"""
from __future__ import annotations

import json
import math
from collections import Counter
import numpy as np

WARDS = {2055, 772043}
SAVE = 999999
PRIMARY_ARMS = ('m2', 'bc20_rerank', 'spend', 'geometric_pool')
METRICS = ('main', 'typed_main', 'complete')
RANK_FLAGS = {'bc_feature_fallback', 'geometric_zero_mass_fallback',
              'native19_candidates_verified', 'spend_control_verified'}


def multiset(counts, *, allow_empty=False):
    if (not isinstance(counts, dict) or (not counts and not allow_empty)
            or any(not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k or int(k) <= 0
                   or type(v) is not int or v <= 0 for k, v in counts.items())):
        raise ValueError('canonical positive item-count multiset required')
    return tuple(sorted((int(key), count) for key, count in counts.items()))


def target_description(label_counts):
    full = multiset(label_counts)
    if any(item == SAVE for item, count in full):
        if full != ((SAVE, 1),):
            raise ValueError('no-buy sentinel cannot be mixed with a purchase')
        return {'kind': 'no_buy', 'main': (), 'full': ()}
    main = tuple(pair for pair in full if pair[0] not in WARDS)
    kind = 'ward_only' if not main else 'mixed' if len(main) != len(full) else 'purchase'
    return {'kind': kind, 'main': main, 'full': full}


def option_multiset(option):
    if type(option.get('group')) is not int or option['group'] < 0:
        raise ValueError('invalid native group identity')
    if option.get('legal') is not True:
        raise ValueError('source option lacks its pinned-world legality claim')
    items = multiset(option.get('items'), allow_empty=True)
    if option.get('kind') == 'save':
        if option['group'] != 0 or items:
            raise ValueError('native SAVE must be the empty group0 option')
    elif option.get('kind') != 'buy' or option['group'] == 0 or not items:
        raise ValueError('native buy option must contain a positive main basket')
    if any(item in WARDS or item == SAVE for item, count in items):
        raise ValueError('native main card must not invent ward or sentinel purchases')
    return items


def option_identity(option):
    return json.dumps(option, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_candidates(options, group_main=None):
    if not isinstance(options, list) or not 1 <= len(options) <= 3:
        raise ValueError('one to three native M2 options required')
    for option in options:
        counts = option_multiset(option)
        if group_main is not None:
            group = option['group']
            if group >= len(group_main):
                raise ValueError('option group absent from canonical catalogue')
            entries = group_main[group]
            if (not isinstance(entries, list) or any(not isinstance(pair, list) or len(pair) != 2
                    or type(pair[0]) is not int or type(pair[1]) is not int or pair[0] <= 0 or pair[1] <= 0 for pair in entries)
                    or len({pair[0] for pair in entries}) != len(entries)
                    or tuple(sorted(map(tuple, entries))) != counts):
                raise ValueError('option group differs from its canonical main basket')
    if len({row['group'] for row in options}) != len(options):
        raise ValueError('duplicate native candidate group')


def assert_same_candidates(original, reordered):
    if sorted(map(option_identity, original)) != sorted(map(option_identity, reordered)):
        raise ValueError('candidate membership or complete option object changed')


def stable_order(options, scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(options),) or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('invalid candidate ranking scores')
    order = sorted(range(len(options)), key=lambda i: (-values[i], i))
    result = [options[i] for i in order]
    assert_same_candidates(options, result)
    return result


def rank_candidates(options, m2_probabilities, bc_group_log_mass, native19_audit, *, group_main, bc_available=True):
    """No target is accepted by this API. Audit entries key native group IDs."""
    validate_candidates(options, group_main)
    if type(bc_available) is not bool:
        raise ValueError('explicit boolean BC feature availability required')
    reference = np.asarray(m2_probabilities, dtype=np.float64)
    if (reference.ndim != 1 or not np.isfinite(reference).all() or np.any(reference < 0)
            or not math.isclose(float(reference.sum()), 1., abs_tol=2e-6, rel_tol=0)):
        raise ValueError('complete normalized M2 group masses required')
    if len(group_main) != len(reference):
        raise ValueError('canonical group map differs from reference probability axis')
    groups = [row['group'] for row in options]
    if max(groups) >= len(reference):
        raise ValueError('native option outside reference catalogue')
    ranked = {'m2': list(options)}
    flags = {'bc_feature_fallback': not bc_available, 'geometric_zero_mass_fallback': False}
    if bc_available:
        bc = np.asarray(bc_group_log_mass, dtype=np.float64)
        if bc.shape != reference.shape or not np.isfinite(bc).all():
            raise ValueError('complete finite grouped BC log masses required')
        ranked['bc20_rerank'] = stable_order(options, bc[groups])
        selected_mass = reference[groups]
        if not np.any(selected_mass > 0):
            ranked['geometric_pool'] = list(options)
            flags['geometric_zero_mass_fallback'] = True
        else:
            score = np.full(len(groups), -np.inf)
            supported = selected_mass > 0
            score[supported] = .5*np.log(selected_mass[supported])+.5*bc[np.asarray(groups)[supported]]
            ranked['geometric_pool'] = stable_order(options, score)
    else:
        ranked['bc20_rerank'] = list(options); ranked['geometric_pool'] = list(options)
    costs, native_valid = [], True
    for group in groups:
        audit = native19_audit.get(str(group), native19_audit.get(group, {}))
        cost = audit.get('incremental_cost')
        valid = (audit.get('feasible') is True and not isinstance(cost, bool)
                 and isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0)
        if group == 0:
            valid = valid and cost == 0
        native_valid = native_valid and valid
        costs.append(float(cost) if valid else 0.)
    ranked['spend'] = stable_order(options, costs) if native_valid else list(options)
    for value in ranked.values():
        assert_same_candidates(options, value)
    flags['native19_candidates_verified'] = native_valid
    flags['spend_control_verified'] = native_valid
    return ranked, flags


def recommendation_credit(target, option):
    if target is None or option is None:
        return {name: 0. for name in METRICS}
    predicted = option_multiset(option)
    main = float(predicted == target['main'])
    typed = main if target['kind'] != 'ward_only' else 0.
    complete = float(predicted == target['full'])
    return {'main': main, 'typed_main': typed, 'complete': complete}


def score_decision(match_id, target_counts, ranked, flags, *, available=True,
                   direct_options=None, catalogue_group=None, catalogue_groups=None,
                   catalogue_top_groups=None):
    """An unavailable selected source row stays as zero credit for every arm."""
    if not isinstance(match_id, str) or not match_id:
        raise ValueError('game identity required')
    if type(available) is not bool:
        raise ValueError('explicit source availability required')
    if not available:
        if any(value is not None for value in (target_counts, ranked, flags, direct_options,
                                              catalogue_group, catalogue_top_groups)):
            raise ValueError('unavailable source row cannot conceal existing predictions, targets or audit failures')
        return {'match_id': match_id, 'available': False, 'target_kind': None,
                'credits': {name: {key: 0. for key in METRICS} for name in PRIMARY_ARMS},
                'candidate_main_ceiling': 0., 'native19_candidates_verified': True,
                'spend_control_verified': True, 'bc_feature_fallback': True,
                'candidate_membership_unchanged': True}
    if set(ranked) != set(PRIMARY_ARMS):
        raise ValueError('all fixed primary/control option orders required')
    if not isinstance(flags, dict) or set(flags) != RANK_FLAGS or any(type(value) is not bool for value in flags.values()):
        raise ValueError('only the exact Boolean ranking flags are accepted')
    if catalogue_groups is None:
        raise ValueError('bound canonical main-group mapping required for scoring')
    validate_candidates(ranked['m2'], catalogue_groups)
    for options in ranked.values():
        assert_same_candidates(ranked['m2'], options)
    truth = target_description(target_counts)
    credits = {name: recommendation_credit(truth, options[0]) for name, options in ranked.items()}
    result = {'match_id': match_id, 'available': True, 'target_kind': truth['kind'], 'credits': credits,
              'candidate_main_ceiling': float(any(option_multiset(option) == truth['main'] for option in ranked['m2'])),
              'candidate_membership_unchanged': True, **flags,
              'first_groups': {name: options[0]['group'] for name, options in ranked.items()},
              'candidate_count': len(ranked['m2'])}
    if direct_options is not None:
        validate_candidates(direct_options, catalogue_groups)
        result['secondary_direct_credit'] = recommendation_credit(truth, direct_options[0])
        result['secondary_direct_main_ceiling'] = float(any(option_multiset(o) == truth['main'] for o in direct_options))
        result['secondary_direct_first_group'] = direct_options[0]['group']
    if catalogue_top_groups is not None:
        if (not isinstance(catalogue_top_groups, list) or not 1 <= len(catalogue_top_groups) <= 3
                or len(set(catalogue_top_groups)) != len(catalogue_top_groups)
                or any(type(group) is not int or not 0 <= group < len(catalogue_groups)
                       for group in catalogue_top_groups)):
            raise ValueError('ordered unique catalogue top-three groups required')
        if catalogue_group is not None and catalogue_group != catalogue_top_groups[0]:
            raise ValueError('catalogue argmax differs from first top-three group')
        catalogue_group = catalogue_top_groups[0]
        result['catalogue_diagnostic_top3_main_credit'] = float(any(
            tuple(sorted(map(tuple, catalogue_groups[group]))) == truth['main']
            for group in catalogue_top_groups))
    if catalogue_group is not None:
        if catalogue_groups is None or not 0 <= catalogue_group < len(catalogue_groups):
            raise ValueError('unfiltered catalogue diagnostic is not bound to group mapping')
        counts = {str(item): count for item, count in catalogue_groups[catalogue_group]}
        result['catalogue_diagnostic_main_credit'] = float(multiset(counts, allow_empty=True) == truth['main'])
        result['catalogue_diagnostic_first_group'] = int(catalogue_group)
    return result


def summarize_binary(matrix, names, indexes, candidate='bc20_rerank', controls=('m2', 'spend')):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or not np.isin(matrix, [0., 1.]).all():
        raise ValueError('aligned binary recommendation credits required')
    draw = matrix[indexes].mean(axis=1)
    arms = {name: {'accuracy': float(matrix[:, i].mean()),
                   'interval95': np.quantile(draw[:, i], [.025, .975]).tolist()} for i, name in enumerate(names)}
    contrasts = {}
    ci = names.index(candidate)
    for control in controls:
        difference = matrix[:, ci]-matrix[:, names.index(control)]
        interval = np.quantile(difference[indexes].mean(axis=1), [.025, .975]).tolist()
        contrasts[control] = {'gain': float(difference.mean()), 'interval95': interval,
                              'improved_games': int(np.count_nonzero(difference > 0)),
                              'regressed_games': int(np.count_nonzero(difference < 0))}
    return {'arms': arms, 'paired_contrasts': contrasts}


def optional_boolean_counts(rows, key):
    """Do not turn an omitted support/audit receipt into a positive claim."""
    available = [row for row in rows if row['available']]
    supplied = [row[key] for row in available if key in row]
    if any(type(value) is not bool for value in supplied):
        raise ValueError('explicit Boolean support/audit receipt required: '+key)
    return {'available_games': len(available), 'verified_true': sum(supplied),
            'verified_false': len(supplied)-sum(supplied), 'missing': len(available)-len(supplied),
            'status': 'complete' if len(supplied) == len(available) and available else
                      'partial' if supplied else 'not_measured'}


def first_choice_agreement(rows, arm):
    """Credit complementarity and action disagreement are different quantities."""
    comparable = [row for row in rows if row['available']]
    counts = Counter()
    for row in comparable:
        if arm == 'bc20_direct':
            group = row['secondary_direct_first_group']
            credit = row['secondary_direct_credit']['main']
        else:
            group = row['first_groups'][arm]
            credit = row['credits'][arm]['main']
        baseline = row['credits']['m2']['main']
        counts['same_first_group' if group == row['first_groups']['m2'] else 'different_first_group'] += 1
        counts['both_correct' if credit and baseline else 'candidate_only_correct' if credit else
               'm2_only_correct' if baseline else 'both_wrong'] += 1
    names = ('same_first_group', 'different_first_group', 'both_correct',
             'candidate_only_correct', 'm2_only_correct', 'both_wrong')
    return {'compared_available_games': len(comparable),
            'unavailable_selected_games': len(rows)-len(comparable),
            **{key: counts[key] for key in names}}


def evaluate_scored_decisions(rows, selected_game_ids, all_game_ids, *, samples=2000, seed=20260930):
    """One selected decision per game; all-model failures remain in denominator."""
    selected = set(selected_game_ids); universe = set(all_game_ids)
    if len(universe) != len(all_game_ids) or len(universe) != 859 or len(selected) != len(selected_game_ids) or not selected <= universe:
        raise ValueError('exact859-game ledger and distinct selected game subset required')
    mids = [row['match_id'] for row in rows]
    if len(set(mids)) != len(mids) or set(mids) != selected or len(rows) < 2:
        raise ValueError('one row per selected decision-bearing game required, including unavailable rows')
    if type(samples) is not int or samples < 1:
        raise ValueError('positive bootstrap sample count required')
    primary = np.asarray([[row['credits'][arm]['main'] for arm in PRIMARY_ARMS] for row in rows], dtype=float)
    if not np.isin(primary, [0., 1.]).all():
        raise ValueError('main-basket exactness must be binary')
    indexes = np.random.default_rng(seed).integers(0, len(rows), size=(samples, len(rows)), dtype=np.int32)
    draws = primary[indexes].mean(axis=1); point = primary.mean(axis=0)
    arms = {arm: {'accuracy': float(point[i]), 'interval95': np.quantile(draws[:, i], [.025, .975]).tolist()}
            for i, arm in enumerate(PRIMARY_ARMS)}
    contrasts = {}
    for arm in ('m2', 'spend'):
        j = PRIMARY_ARMS.index(arm); difference = primary[:, 1]-primary[:, j]
        difference_draws = difference[indexes].mean(axis=1)
        low, high = np.quantile(difference_draws, [.025, .975])
        contrasts[arm] = {'gain': float(difference.mean()), 'interval95': [float(low), float(high)],
                          'improved_games': int(np.count_nonzero(difference > 0)),
                          'regressed_games': int(np.count_nonzero(difference < 0)),
                          'discordant_games': int(np.count_nonzero(difference))}
    available = sum(row['available'] for row in rows)
    conditions = {
        'minimum_selected_games': len(rows) >= 700, 'prediction_coverage': available/len(rows) >= .95,
        'gain_at_least_three_pp': 100*(contrasts['m2']['improved_games']-contrasts['m2']['regressed_games']) >= 3*len(rows),
        'positive_paired_lower_bound_m2': contrasts['m2']['interval95'][0] > 0,
        'positive_paired_lower_bound_spend': contrasts['spend']['interval95'][0] > 0,
        'unchanged_candidates': all(row['candidate_membership_unchanged'] for row in rows),
        'native19_candidates_verified': all(row['native19_candidates_verified'] for row in rows),
        'spend_control_verified': all(row['spend_control_verified'] for row in rows),
    }
    report = {'games_in_selection_ledger': 859, 'decision_bearing_games': len(rows),
            'no_selected_decision_games': 859-len(rows), 'available_prediction_games': available,
            'arms': arms, 'primary_contrasts': contrasts, 'gates': conditions,
            'statistical_and_candidate_gate_passed': all(conditions.values()),
            'candidate_main_ceiling': float(np.mean([row['candidate_main_ceiling'] for row in rows])),
            'bootstrap': {'unit': 'whole selected game', 'samples': samples, 'seed': seed},
            'scope': 'exposed development main-purchase agreement; no win-effect or confirmation claim',
            'remaining_gate': 'Producer identity, label mutation and pinned decoder parity must independently pass.'}
    report['target_kind_counts_available_rows'] = dict(Counter(row['target_kind'] for row in rows if row['available']))
    report['candidate_cardinality_counts'] = dict(Counter(row.get('candidate_count', 0) for row in rows))
    report['feature_fallback_available_rows'] = sum(row['available'] and row['bc_feature_fallback'] for row in rows)
    report['geometric_zero_mass_fallbacks'] = sum(row.get('geometric_zero_mass_fallback', False) for row in rows)
    report['m2_to_ceiling_gap'] = report['candidate_main_ceiling']-arms['m2']['accuracy']
    report['additional_primary_metrics'] = {}
    for metric in ('typed_main', 'complete'):
        matrix = [[row['credits'][arm][metric] for arm in PRIMARY_ARMS] for row in rows]
        report['additional_primary_metrics'][metric] = summarize_binary(matrix, list(PRIMARY_ARMS), indexes)
    report['geometric_secondary'] = summarize_binary(primary, list(PRIMARY_ARMS), indexes,
                                                     candidate='geometric_pool', controls=('m2', 'spend'))
    report['geometric_secondary']['additional_metrics'] = {
        metric: summarize_binary([[row['credits'][arm][metric] for arm in PRIMARY_ARMS] for row in rows],
                                 list(PRIMARY_ARMS), indexes, candidate='geometric_pool', controls=('m2', 'spend'))
        for metric in ('typed_main', 'complete')}
    has_direct = any('secondary_direct_credit' in row for row in rows)
    if has_direct:
        if any(row['available'] and 'secondary_direct_credit' not in row for row in rows):
            raise ValueError('partial direct-BC secondary cannot silently omit available rows')
        direct = np.asarray([row.get('secondary_direct_credit', {}).get('main', 0.) for row in rows])
        report['direct_bc_secondary'] = summarize_binary(np.column_stack((primary[:, 0], direct)),
                                                         ['m2', 'bc20_direct'], indexes,
                                                         candidate='bc20_direct', controls=('m2',))
        report['direct_bc_secondary']['candidate_main_ceiling'] = float(np.mean([
            row.get('secondary_direct_main_ceiling', 0.) for row in rows]))
        report['direct_bc_secondary']['first_to_ceiling_gap'] = (
            report['direct_bc_secondary']['candidate_main_ceiling']-
            report['direct_bc_secondary']['arms']['bc20_direct']['accuracy'])
        report['direct_bc_secondary']['additional_metrics'] = {
            metric: summarize_binary([[row['credits']['m2'][metric],
                                       row.get('secondary_direct_credit', {}).get(metric, 0.)] for row in rows],
                                     ['m2', 'bc20_direct'], indexes, candidate='bc20_direct', controls=('m2',))
            for metric in ('typed_main', 'complete')}
        direct_audit = optional_boolean_counts(rows, 'direct_native19_verified')
        report['direct_bc_secondary']['native19_audit'] = direct_audit
        report['direct_bc_secondary']['candidate_legality_eligible'] = bool(
            direct_audit['status'] == 'complete' and direct_audit['verified_false'] == 0)
        report['direct_bc_secondary']['primary_gate_effect'] = 'none; a secondary cannot rescue the primary'
    else:
        report['direct_bc_secondary'] = {'status': 'not_measured'}
    known_catalogue = sum('catalogue_diagnostic_main_credit' in row for row in rows)
    known_top3 = sum('catalogue_diagnostic_top3_main_credit' in row for row in rows)
    report['catalogue_diagnostic'] = {
        'prediction_games': known_catalogue,
        'main_accuracy_all_selected_games': float(np.mean([row.get('catalogue_diagnostic_main_credit', 0.) for row in rows]))
            if known_catalogue else None,
        'top3_prediction_games': known_top3,
        'main_top3_recall_all_selected_games': float(np.mean([
            row.get('catalogue_diagnostic_top3_main_credit', 0.) for row in rows])) if known_top3 else None,
        'scope': 'unfiltered catalogue argmax/top3; absent predictions are zero credit, not a displayed policy'}
    report['target_catalogue_support'] = {
        'main': optional_boolean_counts(rows, 'catalogue_main_target_supported'),
        'full': optional_boolean_counts(rows, 'catalogue_full_target_supported'),
        'scope': 'receipt-verified support among available targets; unsupported targets stay in every metric'}
    compared_arms = ['bc20_rerank', 'spend', 'geometric_pool']+(['bc20_direct'] if has_direct else [])
    report['first_choice_complementarity_against_m2'] = {
        arm: first_choice_agreement(rows, arm) for arm in compared_arms}
    report['first_to_fixed_candidate_ceiling_gaps'] = {
        arm: report['candidate_main_ceiling']-arms[arm]['accuracy'] for arm in PRIMARY_ARMS}
    available_mask = np.asarray([row['available'] for row in rows])
    report['available_row_accuracy_diagnostic'] = {
        arm: float(primary[available_mask, i].mean()) if available else None for i, arm in enumerate(PRIMARY_ARMS)}
    report['slices'] = {}
    for label, kinds in (('purchase', {'purchase', 'mixed', 'ward_only'}), ('observed_no_buy', {'no_buy'})):
        chosen = np.flatnonzero([row['available'] and row['target_kind'] in kinds for row in rows])
        if len(chosen) < 30:
            report['slices'][label] = {'games': len(chosen), 'status': 'not_measured_below_30_games'}
        else:
            local = np.random.default_rng(seed).integers(0, len(chosen), size=(samples, len(chosen)), dtype=np.int32)
            slice_names = list(PRIMARY_ARMS)
            slice_matrix = primary[chosen]
            if has_direct:
                slice_names.append('bc20_direct')
                slice_matrix = np.column_stack((slice_matrix, direct[chosen]))
            report['slices'][label] = {'games': len(chosen), 'scope': 'exploratory conditional slice',
                                      'metrics': summarize_binary(slice_matrix, slice_names, local),
                                      'geometric_paired': summarize_binary(slice_matrix, slice_names, local,
                                                                          candidate='geometric_pool', controls=('m2', 'spend'))}
            if has_direct:
                report['slices'][label]['direct_paired'] = summarize_binary(
                    slice_matrix, slice_names, local, candidate='bc20_direct', controls=('m2',))
    return report
