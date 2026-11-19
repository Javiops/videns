"""Summarize a completed, hash-bound 16.17 paired validation run without scoring.

Reads only the displayed-policy report and its streamed paired outcome artifact.
Slices are exploratory and support a next controlled experiment, never promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.dataset_artifacts import publish_directory, sha256, write_json
from app.shop_econ import PINK, SAVE_ITEM

PRIMARY = ('first_action_exact', 'action_family_match')
MIN_VISITS, MIN_GAMES = 30, 5
ERROR_DESCRIPTIONS = {
    'overbuy': 'Lead basket contains items or copies absent from the recorded retained purchase basket.',
    'underbuy': 'Lead basket omits items or copies in the recorded retained purchase basket.',
    'extra_duplicate_copies': 'Lead basket includes too many copies of an item the player did purchase.',
    'save_false_positive': 'Lead option is save on a recorded purchase.',
    'buy_on_no_buy': 'Lead option purchases items on a recorded post-death no-buy.',
    'ward_miss': 'Lead basket omits at least one recorded Control Ward copy.',
    'ward_miss_while_buying': 'A lead buy option omits a recorded Control Ward; excludes misses caused by a save/no-option lead.',
    'unseen_label': 'Recorded target contains an item absent from the transformer TRAIN vocabulary.',
}
CONTROLLED_CHANGES = {
    'extra_duplicate_copies': ('count_decoder', 'Test one stricter rule for additional copies, holding first-copy ranking and the trained artifact fixed.',
                               'Track missing-copy errors as well as first_action_exact; extra-copy errors alone do not establish a better rule.'),
    'save_false_positive': ('save_gate', 'Test one predeclared save-trigger adjustment, holding the trained trunk and purchase ranking fixed.',
                           'Measure post-death no-buy recall alongside purchase exactness to expose the tradeoff.'),
    'ward_miss_while_buying': ('ward_options', 'Test one predeclared ward-option ranking adjustment, retaining the same inventory, wallet and slot checks.',
                  'Track first_action_exact on ward and non-ward visits on the identical frozen validation games.'),
}


def strict_json(text):
    def reject(value):
        raise ValueError(f'Nonstandard JSON numeric value: {value}')
    return json.loads(text, parse_constant=reject)


def counts(raw):
    if not isinstance(raw, dict):
        raise ValueError('Expected explicit item-count mapping')
    result = Counter()
    for item, quantity in raw.items():
        if not str(item).isdigit() or int(item) <= 0 or type(quantity) is not int or quantity <= 0:
            raise ValueError('Invalid item-count entry in paired evidence')
        result[int(item)] += quantity
    return result


def metric_cell():
    return {'A': {'sum': 0., 'n': 0}, 'B': {'sum': 0., 'n': 0},
            'paired_common': {'a_sum': 0., 'b_sum': 0., 'n': 0, 'games': set()}}


def group_cell():
    return {'visits': 0, 'games': set(), 'metrics': defaultdict(metric_cell)}


def add_visit(group, visit, metrics):
    group['visits'] += 1
    group['games'].add(visit['match_id'])
    for key in metrics:
        cell = group['metrics'][key]
        values = {}
        for tag in ('A', 'B'):
            policy = visit['policies'][tag]
            value = policy['metrics'][key]
            n = policy['metric_n'][key]
            if type(n) is not int or n != int(value is not None):
                raise ValueError('Per-visit metric denominator disagrees with definedness')
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Invalid per-visit metric value')
            values[tag] = value
            if n:
                cell[tag]['sum'] += value
                cell[tag]['n'] += 1
        if key in PRIMARY and (values['A'] is None) != (values['B'] is None):
            raise ValueError('Primary metrics have unpaired visit populations')
        if values['A'] is not None and values['B'] is not None:
            paired = cell['paired_common']
            paired['a_sum'] += values['A']
            paired['b_sum'] += values['B']
            paired['n'] += 1
            paired['games'].add(visit['match_id'])


def finish_group(group):
    result = {'visits': group['visits'], 'games': len(group['games']),
              'coverage': 'supported' if group['visits'] >= MIN_VISITS and len(group['games']) >= MIN_GAMES else 'sparse',
              'interpretation': 'exploratory; no slice confidence interval computed', 'metrics': {}}
    for key, cell in group['metrics'].items():
        metrics = {}
        for tag in ('A', 'B'):
            n = cell[tag]['n']
            metrics[tag] = {**cell[tag], 'value': cell[tag]['sum'] / n if n else None,
                            'status': 'measured' if n else 'not_measured'}
        paired = cell['paired_common']
        n = paired['n']
        metrics['paired_common_population'] = {
            'n': n, 'games': len(paired['games']), 'a_sum': paired['a_sum'], 'b_sum': paired['b_sum'],
            'a': paired['a_sum'] / n if n else None, 'b': paired['b_sum'] / n if n else None,
            'diff_a_minus_b': (paired['a_sum'] - paired['b_sum']) / n if n else None,
            'status': 'measured' if n else 'not_measured', 'ci95': None,
            'interpretation': 'Both policies evaluated on exactly the visits where this metric is defined for both.',
        }
        metrics['recommendation_support'] = n >= MIN_VISITS and len(paired['games']) >= MIN_GAMES
        result['metrics'][key] = metrics
    return result


def dimensions(visit, truth):
    minute = visit['minute']
    time_band = '0-9 minutes' if minute < 10 else '10-19 minutes' if minute < 20 else '20-29 minutes' if minute < 30 else '30+ minutes'
    budget = visit['budget']
    budget_band = '<500 gold' if budget < 500 else '500-999 gold' if budget < 1000 else '1000-1999 gold' if budget < 2000 else '2000+ gold'
    quantity = sum(n for item, n in truth.items() if item != SAVE_ITEM)
    return {'champion': visit.get('champion') or 'unknown', 'role': visit.get('role') or 'unknown',
            'save_kind': visit['save_kind'], 'time': time_band, 'budget': budget_band,
            'truth_basket_quantity': str(quantity)}


def errors_for(visit, tag, truth):
    policy = visit['policies'][tag]
    options = policy['displayed_options']
    first = options[0] if options else {'kind': 'no_option', 'item_counts': {}}
    predicted = counts(first['item_counts'])
    actual_buy = Counter({i: n for i, n in truth.items() if i != SAVE_ITEM})
    no_buy = visit['save_kind'] == 'no_buy_death'
    kinds = set()
    if no_buy:
        if first['kind'] == 'buy' and predicted:
            kinds.add('buy_on_no_buy')
    else:
        extra, missing = predicted - actual_buy, actual_buy - predicted
        if extra: kinds.add('overbuy')
        if missing: kinds.add('underbuy')
        if any(item in actual_buy for item in extra): kinds.add('extra_duplicate_copies')
        if first['kind'] == 'save': kinds.add('save_false_positive')
        if any(item in PINK for item in missing):
            kinds.add('ward_miss')
            if first['kind'] == 'buy': kinds.add('ward_miss_while_buying')
    if policy.get('unseen_truth_item_counts'):
        kinds.add('unseen_label')
    return kinds, first


def analyze(report_path, destination):
    report_path, destination = Path(report_path).resolve(), Path(destination).resolve()
    if (not report_path.name.startswith('policy_eval_val_') or report_path.suffix != '.json'
        or destination.exists() or destination.parent != report_path.parent):
        raise ValueError('Analysis requires a complete report and a new child directory of its isolated run')
    report_bytes = report_path.read_bytes()
    report = strict_json(report_bytes)
    if (report.get('purpose') != 'patch_development' or report.get('split') != 'val'
        or report.get('manifest', {}).get('patch') != '16.17'
        or report.get('final_test_evaluation') is not False or report.get('promotion_allowed') is not False
        or report.get('primary_metrics') != list(PRIMARY) or set(report.get('metrics', {})) != {'A', 'B'}):
        raise ValueError('Only isolated, nonpromotable 16.17 paired validation reports are accepted')
    binding = report['paired_outcomes']
    outcomes_path = Path(binding['path']).resolve()
    if (outcomes_path.parent != report_path.parent or outcomes_path.suffix != '.jsonl'
        or not outcomes_path.name.startswith('policy_outcomes_val_')):
        raise ValueError('Outcome capture must be an isolated run artifact, never a dataset payload')
    metrics = sorted(set(report['metrics']['A']) - {'unaffordable'})
    if set(report['metrics']['B']) - {'unaffordable'} != set(metrics) or not set(PRIMARY) <= set(metrics):
        raise ValueError('Report policy metric sets differ')
    groups = defaultdict(group_cell)
    global_group = group_cell()
    errors = {kind: {'A': {'n': 0, 'games': set(), 'examples': []},
                     'B': {'n': 0, 'games': set(), 'examples': []}} for kind in ERROR_DESCRIPTIONS}
    digest, byte_count, row_count = hashlib.sha256(), 0, 0
    unaffordable = Counter()
    with outcomes_path.open('rb') as source:
        for raw in source:
            digest.update(raw)
            byte_count += len(raw)
            if not raw.strip(): raise ValueError('Blank line in streamed paired evidence')
            visit = strict_json(raw)
            if (visit.get('schema_version') != 1 or visit.get('split') != 'val'
                or set(visit.get('policies', {})) != {'A', 'B'} or not visit.get('match_id')):
                raise ValueError('Unpaired or non-validation outcome row')
            truth = counts(visit['actual_item_counts'])
            add_visit(global_group, visit, metrics)
            for dimension, value in dimensions(visit, truth).items():
                add_visit(groups[(dimension, value)], visit, metrics)
            for tag in ('A', 'B'):
                unaffordable[tag] += visit['policies'][tag]['unaffordable']
                kinds, first = errors_for(visit, tag, truth)
                for kind in kinds:
                    error = errors[kind][tag]
                    error['n'] += 1
                    error['games'].add(visit['match_id'])
                    if len(error['examples']) < 3 and all(x['match_id'] != visit['match_id'] for x in error['examples']):
                        error['examples'].append({
                            'outcome_line': row_count + 1, 'match_id': visit['match_id'],
                            'participant_id': visit.get('participant_id'), 'ts': visit.get('ts'),
                            'champion': visit.get('champion'), 'role': visit.get('role'),
                            'budget': visit['budget'], 'inventory': visit['inventory'],
                            'actual_item_counts': visit['actual_item_counts'], 'first_option': first,
                        })
            row_count += 1
    if (digest.hexdigest() != binding['sha256'] or byte_count != binding['bytes']
        or row_count != binding['rows'] or row_count != report['visits']
        or len(global_group['games']) != report['games']):
        raise ValueError('Paired outcome digest or population differs from the report')
    global_summary = finish_group(global_group)
    for tag in ('A', 'B'):
        if unaffordable[tag] != report['metrics'][tag]['unaffordable']:
            raise ValueError('Unaffordable count differs from displayed-policy report')
        for key in metrics:
            recomputed, declared = global_summary['metrics'][key][tag], report['metrics'][tag][key]
            if (recomputed['n'] != declared['n'] or (recomputed['value'] is None) != (declared['value'] is None)
                or (recomputed['value'] is not None and not math.isclose(recomputed['value'], declared['value'], abs_tol=1e-12))):
                raise ValueError(f'Paired capture cannot reproduce {tag}/{key} in the report')
    for key in PRIMARY:
        common = global_summary['metrics'][key]['paired_common_population']
        declared = report['paired_diff_a_minus_b'][key]
        if declared['n_a'] != common['n'] or declared['n_b'] != common['n']:
            raise ValueError('Primary paired report denominators differ from captured visits')
        for reported_key, computed_key in (('a', 'a'), ('b', 'b'), ('diff', 'diff_a_minus_b')):
            observed, expected = declared[reported_key], common[computed_key]
            if ((observed is None) != (expected is None)
                or (expected is not None and not math.isclose(observed, expected, abs_tol=1e-12))):
                raise ValueError('Primary paired report estimate differs from captured visits')
    slices = [{'dimension': dimension, 'value': value, **finish_group(group)}
              for (dimension, value), group in sorted(groups.items())]
    error_summary = {}
    for kind, policies in errors.items():
        error_summary[kind] = {'description': ERROR_DESCRIPTIONS[kind], 'denominator_visits': row_count,
            'policies': {tag: {'n': data['n'], 'games': len(data['games']),
                'rate_all_visits': data['n'] / row_count if row_count else None,
                'recommendation_support': data['n'] >= MIN_VISITS and len(data['games']) >= MIN_GAMES,
                'examples': data['examples']} for tag, data in policies.items()}}
    recommendations = []
    for kind in sorted(CONTROLLED_CHANGES, key=lambda k: (-(errors[k]['A']['n'] - errors[k]['B']['n']), k)):
        data = error_summary[kind]['policies']
        if data['A']['recommendation_support'] and data['A']['n'] > data['B']['n']:
            code, change, guard = CONTROLLED_CHANGES[kind]
            recommendations.append({'hypothesis': code, 'candidate_error_class': kind,
                'candidate_n': data['A']['n'], 'baseline_n': data['B']['n'], 'population_n': row_count,
                'candidate_error_games': data['A']['games'], 'proposed_controlled_change': change,
                'guard_metric': guard, 'status': 'exploratory proposal only; no change executed',
                'source_report': str(report_path)})
        if len(recommendations) == 2: break
    supported_cells = [s for s in slices if s['dimension'] in ('champion', 'role')
                       and s['metrics']['first_action_exact']['recommendation_support']]
    weak_cells = sorted(supported_cells,
        key=lambda s: (s['metrics']['first_action_exact']['paired_common_population']['diff_a_minus_b'], s['dimension'], s['value']))[:8]
    result = {
        'schema_version': 1, 'purpose': 'exploratory_patch_development_analysis', 'patch': '16.17', 'split': 'val',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_report': str(report_path),
        'source_report_sha256': hashlib.sha256(report_bytes).hexdigest(), 'source_outcomes': binding,
        'analyzer_sha256': sha256(Path(__file__)), 'export_fingerprint': report['manifest']['export_fingerprint'],
        'run_plan_sha256': report.get('run_plan_sha256'), 'split_metadata': report['manifest'].get('split_metadata'),
        'artifacts': report['artifacts'], 'artifact_sha256': report['artifact_sha256'],
        'visits': row_count, 'games': len(global_group['games']),
        'global_primary_metrics': {key: {'metric': key, 'split': 'val', 'source_report': str(report_path),
            'A': report['metrics']['A'][key], 'B': report['metrics']['B'][key],
            'paired_a_minus_b': report['paired_diff_a_minus_b'][key]} for key in PRIMARY},
        'global_recomputed': global_summary, 'unaffordable': dict(unaffordable),
        'support_threshold': {'minimum_visits': MIN_VISITS, 'minimum_games': MIN_GAMES,
            'rule': 'Champion/role discussion requires this support on the identical primary-metric population; sparse cells remain coverage only.'},
        'slices': slices, 'supported_cells_with_smallest_candidate_minus_baseline': weak_cells,
        'error_classes': error_summary, 'next_controlled_changes': recommendations,
        'proposal_status': 'Supported recurring candidate errors suggest hypotheses, not proven causes.' if recommendations else 'No next change recommended: insufficient supported excess in the inspected concrete error classes.',
        'limits': ['All slices are exploratory on previously exposed development validation, not confirmatory tests.',
                   'No per-slice confidence intervals or significance claims are made.',
                   'Error classes overlap; never add them as independent failures.',
                   'Overbuy/underbuy mean disagreement with recorded purchases, not inferior or superior game decisions.',
                   'Metric-specific ratios are shown for each policy; paired slice differences use their identical common population.',
                   'The original final test remains sealed and unavailable here; no model was loaded or scored.'],
        'model_reevaluated': False, 'final_test_read': False, 'training_changes_executed': False, 'promotion_allowed': False,
    }
    json.dumps(result, allow_nan=False)
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir()
    write_json(pending / 'analysis.json', result)
    (pending / 'analysis.md').write_text(markdown(result), encoding='utf-8')
    publish_directory(pending, destination)
    return result


def number(value):
    return 'not measured' if value is None else f'{value:.4f}'


def markdown(result):
    lines = [f"Patch 16.17 exploratory validation analysis: {result['visits']} visits, {result['games']} games.",
             f"Source report: {result['source_report']}", '',
             'These are development findings. Champion, role and error slices are exploratory; no slice confidence intervals were computed.', '']
    for key, metric in result['global_primary_metrics'].items():
        for tag in ('A', 'B'):
            entry = metric[tag]
            lines.append(f"- {tag} {key}: {number(entry['value'])}; split=val; n={entry['n']}; CI95={entry.get('ci95')}; report={metric['source_report']}")
        paired = metric['paired_a_minus_b']
        lines.append(f"- Paired A-B {key}: {number(paired['diff'])}; n_A={paired['n_a']}, n_B={paired['n_b']}; CI95={paired.get('ci95')}; report={metric['source_report']}")
    lines.extend(['', 'Supported champion/role cells with the smallest candidate-minus-baseline differences (exploratory):', ''])
    for cell in result['supported_cells_with_smallest_candidate_minus_baseline']:
        metric = cell['metrics']['first_action_exact']['paired_common_population']
        lines.append(f"- {cell['dimension']}={cell['value']}: first_action_exact A-B={number(metric['diff_a_minus_b'])}; n={metric['n']}; games={metric['games']}; source={result['source_report']}")
    if not result['supported_cells_with_smallest_candidate_minus_baseline']:
        lines.append('No champion/role cell meets the 30-visit, five-game support threshold.')
    lines.extend(['', 'Recurring concrete error classes (overlapping):', ''])
    recurring = sorted(((kind, data) for kind, data in result['error_classes'].items()
                        if data['policies']['A']['n'] >= 2),
                       key=lambda entry: -entry[1]['policies']['A']['n'])[:3]
    for kind, data in recurring:
        a, b = data['policies']['A'], data['policies']['B']
        lines.append(f"- {kind}: candidate {a['n']} visits / {a['games']} games; baseline {b['n']} visits; denominator={data['denominator_visits']}. {data['description']}")
        for sample in a['examples'][:2]:
            lines.append(f"  Example: {sample['match_id']}, participant {sample['participant_id']}, ts={sample['ts']}, {sample['champion']} {sample['role']}, budget={sample['budget']}; actual={sample['actual_item_counts']}; shown={sample['first_option']}; outcomes line={sample['outcome_line']}.")
    lines.extend(['', 'Possible next controlled experiments:', ''])
    for recommendation in result['next_controlled_changes']:
        lines.append(f"- {recommendation['proposed_controlled_change']} {recommendation['guard_metric']} Evidence: {recommendation['candidate_error_class']} candidate n={recommendation['candidate_n']}, baseline n={recommendation['baseline_n']}, all-visits n={recommendation['population_n']}; source={recommendation['source_report']}.")
    if not result['next_controlled_changes']: lines.append(result['proposal_status'])
    lines.extend(['', *result['limits'], ''])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='New analysis directory directly under the isolated run (default: analysis)')
    args = parser.parse_args()
    output = args.out or args.report.resolve().parent / 'analysis'
    result = analyze(args.report, output)
    print(json.dumps({'analysis': str(output.resolve() / 'analysis.json'), 'visits': result['visits'],
                      'games': result['games'], 'proposals': len(result['next_controlled_changes'])}))
