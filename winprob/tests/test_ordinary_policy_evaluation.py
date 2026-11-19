from copy import deepcopy
import unittest
import numpy as np
from winprob.ordinary_policy_evaluation import (rank_candidates, score_decision,
    evaluate_scored_decisions, assert_same_candidates, target_description)

GROUPS = [[], [[1036, 1]], [[1029, 1]]]


def options():
    return [{'kind': 'save', 'items': {}, 'group': 0, 'legal': True},
            {'kind': 'buy', 'items': {'1036': 1}, 'group': 1, 'legal': True},
            {'kind': 'buy', 'items': {'1029': 1}, 'group': 2, 'legal': True}]


def audit():
    return {'0': {'feasible': True, 'incremental_cost': 0.},
            '1': {'feasible': True, 'incremental_cost': 350.},
            '2': {'feasible': True, 'incremental_cost': 300.}}


class OrdinaryPolicyTests(unittest.TestCase):
    def ranked(self, bc=None, cost=None, available=True):
        return rank_candidates(options(), [.5, .3, .2], bc if bc is not None else [-2, -1, -3],
                               audit() if cost is None else cost, group_main=GROUPS, bc_available=available)

    def score(self, *args, **kwargs):
        return score_decision(*args, catalogue_groups=GROUPS, **kwargs)

    def test_ward_only_baseline_credit_cannot_be_removed(self):
        ranked, flags = self.ranked()
        row = self.score('g1', {'2055': 2}, ranked, flags)
        self.assertEqual(row['credits']['m2']['main'], 1.)
        self.assertEqual(row['credits']['bc20_rerank']['main'], 0.)
        self.assertEqual(row['credits']['m2']['typed_main'], 0.)
        self.assertEqual(row['credits']['bc20_rerank']['typed_main'], 0.)
        self.assertEqual(row['credits']['m2']['complete'], 0.)

    def test_empty_no_buy_and_mixed_purchase_semantics(self):
        ranked, flags = self.ranked()
        row = self.score('g1', {'999999': 1}, ranked, flags)
        self.assertEqual(row['credits']['m2'], {'main': 1., 'typed_main': 1., 'complete': 1.})
        row = self.score('g2', {'1036': 1, '2055': 1}, ranked, flags)
        self.assertEqual(row['credits']['bc20_rerank'], {'main': 1., 'typed_main': 1., 'complete': 0.})

    def test_exact_ties_and_feature_failure_preserve_original_objects(self):
        native = options()
        ranked, flags = rank_candidates(native, [.5, .3, .2], [0., 0., 0.], audit(), group_main=GROUPS)
        self.assertTrue(all(old is new for old, new in zip(native, ranked['bc20_rerank'])))
        ranked, flags = rank_candidates(native, [.5, .3, .2], None, audit(), group_main=GROUPS, bc_available=False)
        self.assertTrue(flags['bc_feature_fallback'])
        self.assertTrue(all(old is new for old, new in zip(native, ranked['bc20_rerank'])))
        self.assertTrue(all(old is new for old, new in zip(native, ranked['geometric_pool'])))

    def test_unknown_native_cost_prevents_spend_verification(self):
        costs = audit(); costs['1']['feasible'] = False
        ranked, flags = self.ranked(cost=costs)
        self.assertFalse(flags['native19_candidates_verified'])
        self.assertFalse(flags['spend_control_verified'])
        self.assertEqual(ranked['spend'], options())

    def test_counts_membership_and_unmapped_target_are_preserved(self):
        native = options(); changed = deepcopy(native); changed[1]['items']['1036'] = 2
        with self.assertRaisesRegex(ValueError, 'changed'):
            assert_same_candidates(native, changed)
        ranked, flags = self.ranked()
        row = self.score('g1', {'999123': 1}, ranked, flags)
        self.assertTrue(all(credit['main'] == 0 for credit in row['credits'].values()))
        self.assertNotEqual(target_description({'1036': 1})['main'], target_description({'1036': 2})['main'])

    def test_all_zero_candidate_reference_mass_has_explicit_pool_fallback(self):
        ranked, flags = rank_candidates(options(), [0., 0., 0., 1.], [1., 3., 2., 0.], audit(), group_main=GROUPS+[[[1052, 1]]])
        self.assertTrue(flags['geometric_zero_mass_fallback'])
        self.assertEqual(ranked['geometric_pool'], options())

    def test_unavailable_selected_row_stays_in_paired_denominator(self):
        ranked, flags = self.ranked()
        first = self.score('g1', {'1036': 1}, ranked, flags)
        absent = score_decision('g2', None, None, None, available=False)
        universe = ['g'+str(i) for i in range(1, 860)]
        report = evaluate_scored_decisions([first, absent], ['g1', 'g2'], universe, samples=100)
        self.assertEqual(report['decision_bearing_games'], 2)
        self.assertEqual(report['available_prediction_games'], 1)
        self.assertEqual(report['arms']['bc20_rerank']['accuracy'], .5)
        self.assertFalse(report['statistical_and_candidate_gate_passed'])
        with self.assertRaisesRegex(ValueError, 'one row'):
            evaluate_scored_decisions([first, first], ['g1', 'g2'], universe)

    def test_reordering_does_not_change_candidate_ceiling(self):
        ranked, flags = self.ranked()
        self.assertEqual([ranked[arm][0]['group'] for arm in ('m2', 'bc20_rerank', 'spend')], [0, 1, 1])
        for candidate in ranked.values():
            assert_same_candidates(ranked['m2'], candidate)
        row = self.score('g1', {'1029': 1}, ranked, flags)
        self.assertEqual(row['candidate_main_ceiling'], 1.)

    def test_group_basket_identity_and_flag_override_fail(self):
        native = options(); native[1]['items'] = {'1029': 1}
        with self.assertRaisesRegex(ValueError, 'canonical main basket'):
            rank_candidates(native, [.5, .3, .2], [0., 0., 0.], audit(), group_main=GROUPS)
        ranked, flags = self.ranked()
        with self.assertRaisesRegex(ValueError, 'Boolean ranking flags'):
            self.score('g1', {'1036': 1}, ranked, {**flags, 'credits': {}})
        with self.assertRaisesRegex(ValueError, 'cannot conceal'):
            score_decision('g1', None, None, {**flags, 'native19_candidates_verified': False}, available=False)

    def test_exact_three_percentage_points_passes_integer_threshold(self):
        rows = []
        unchanged, flags = self.ranked(bc=[0., -1., -2.])
        changed, _ = self.ranked()
        for i in range(700):
            ranked = unchanged if i < 385 else changed
            truth = {'999999': 1} if i < 385 else {'1036': 1} if i < 406 else {'999123': 1}
            rows.append(self.score('g'+str(i), truth, ranked, flags))
        report = evaluate_scored_decisions(rows, [r['match_id'] for r in rows],
                                           ['g'+str(i) for i in range(859)], samples=100)
        self.assertEqual(report['primary_contrasts']['m2']['gain'], .03)
        self.assertTrue(report['gates']['gain_at_least_three_pp'])

    def test_catalogue_top_three_keeps_unmapped_targets_and_missing_predictions(self):
        ranked, flags = self.ranked()
        hit = self.score('g1', {'1029': 1}, ranked, flags,
                         catalogue_group=1, catalogue_top_groups=[1, 2, 0])
        miss = self.score('g2', {'999123': 1}, ranked, flags,
                          catalogue_top_groups=[1, 2, 0])
        absent = score_decision('g3', None, None, None, available=False)
        hit.update(catalogue_main_target_supported=True, catalogue_full_target_supported=True)
        miss.update(catalogue_main_target_supported=False, catalogue_full_target_supported=False)
        report = evaluate_scored_decisions([hit, miss, absent], ['g1', 'g2', 'g3'],
                                           ['g'+str(i) for i in range(859)], samples=100)
        self.assertEqual(report['catalogue_diagnostic']['prediction_games'], 2)
        self.assertEqual(report['catalogue_diagnostic']['main_accuracy_all_selected_games'], 0.)
        self.assertEqual(report['catalogue_diagnostic']['main_top3_recall_all_selected_games'], 1/3)
        self.assertEqual(report['target_catalogue_support']['main']['verified_false'], 1)
        self.assertEqual(report['decision_bearing_games'], 3)
        with self.assertRaisesRegex(ValueError, 'argmax differs'):
            self.score('g1', {'1029': 1}, ranked, flags, catalogue_group=0, catalogue_top_groups=[1, 2, 0])
        with self.assertRaisesRegex(ValueError, 'unique catalogue'):
            self.score('g1', {'1029': 1}, ranked, flags, catalogue_top_groups=[1, 1])

    def test_action_disagreement_and_credit_complementarity_exclude_unavailable(self):
        ranked, flags = self.ranked()
        rows = [self.score('g1', {'2055': 1}, ranked, flags),
                self.score('g2', {'1036': 1}, ranked, flags),
                self.score('g3', {'999123': 1}, ranked, flags),
                score_decision('g4', None, None, None, available=False)]
        report = evaluate_scored_decisions(rows, [row['match_id'] for row in rows],
                                           ['g'+str(i) for i in range(859)], samples=100)
        counts = report['first_choice_complementarity_against_m2']['bc20_rerank']
        self.assertEqual(counts['compared_available_games'], 3)
        self.assertEqual(counts['unavailable_selected_games'], 1)
        self.assertEqual(counts['different_first_group'], 3)
        self.assertEqual((counts['candidate_only_correct'], counts['m2_only_correct'], counts['both_wrong']), (1, 1, 1))

    def test_direct_metrics_and_separate_legality_cannot_change_primary_gate(self):
        ranked, flags = self.ranked()
        rows = [self.score('g1', {'1036': 1, '2055': 1}, ranked, flags, direct_options=ranked['bc20_rerank']),
                self.score('g2', {'999999': 1}, ranked, flags, direct_options=ranked['bc20_rerank'])]
        universe = ['g'+str(i) for i in range(859)]
        for row in rows:
            row['direct_native19_verified'] = True
        good = evaluate_scored_decisions(rows, ['g1', 'g2'], universe, samples=100)
        direct = good['direct_bc_secondary']
        self.assertTrue(direct['candidate_legality_eligible'])
        self.assertEqual(direct['additional_metrics']['typed_main']['arms']['bc20_direct']['accuracy'], .5)
        self.assertEqual(direct['additional_metrics']['complete']['arms']['bc20_direct']['accuracy'], 0.)
        self.assertEqual(direct['first_to_ceiling_gap'], .5)
        rows[0]['direct_native19_verified'] = False
        bad = evaluate_scored_decisions(rows, ['g1', 'g2'], universe, samples=100)
        self.assertFalse(bad['direct_bc_secondary']['candidate_legality_eligible'])
        self.assertEqual(good['gates'], bad['gates'])
        del rows[1]['direct_native19_verified']
        missing = evaluate_scored_decisions(rows, ['g1', 'g2'], universe, samples=100)
        self.assertEqual(missing['direct_bc_secondary']['native19_audit']['missing'], 1)
        self.assertFalse(missing['direct_bc_secondary']['candidate_legality_eligible'])

    def test_declared_slice_shows_all_scored_arms_and_support_receipts_are_boolean(self):
        ranked, flags = self.ranked()
        rows = [self.score('g'+str(i), {'1036': 1}, ranked, flags,
                           direct_options=ranked['bc20_rerank']) for i in range(30)]
        for row in rows:
            row['direct_native19_verified'] = True
        selected = [row['match_id'] for row in rows]
        universe = ['g'+str(i) for i in range(859)]
        report = evaluate_scored_decisions(rows, selected, universe, samples=100)
        self.assertIn('bc20_direct', report['slices']['purchase']['metrics']['arms'])
        self.assertIn('direct_paired', report['slices']['purchase'])
        self.assertEqual(report['slices']['observed_no_buy']['status'], 'not_measured_below_30_games')
        rows[0]['catalogue_main_target_supported'] = 'false'
        with self.assertRaisesRegex(ValueError, 'Boolean support'):
            evaluate_scored_decisions(rows, selected, universe, samples=100)


if __name__ == '__main__':
    unittest.main()
