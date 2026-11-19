"""Synthetic tests only; no raw data or W1 artifacts are opened."""
import sys
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from setback_experiment import (
    ALL_FEATURES, build_windows, choose_baseline, focal_sign, game_retrieval,
    orient_features, paired_retrieval, retrieval_point, risk_fit, risk_predict,
    select_windows, wp_rank_scores, reject_quarantined, verify_admitted_sources,
)


class SetbackTests(unittest.TestCase):
    def fixture(self, minutes=(6, 9, 12)):
        names = list(ALL_FEATURES)
        x = np.zeros((len(minutes), len(names)))
        timestamps = np.array([m*60000 + 11 + 4*j for j, m in enumerate(minutes)], dtype=np.int64)
        x[:, names.index('time_min')] = timestamps / 60000
        x[:, names.index('gold_diff_slope_available')] = 1
        sign = focal_sign('SYNTHETIC_1')
        x[:, names.index('gold_diff')] = np.array([3000] + [3000-sign*1500]*(len(minutes)-1))
        meta = [{'game_index': 0, 'match_id': 'SYNTHETIC_1', 'role': 'fit', 'patch': 'synthetic'}]
        return x, np.zeros(len(minutes), dtype=int), timestamps, np.array(['fit']*len(minutes)), names, meta

    def test_actual_endpoints_are_shared_without_interpolation(self):
        args = self.fixture()
        windows, coverage = build_windows(*args)
        np.testing.assert_array_equal(windows['label'], [1, 0])
        self.assertEqual(windows['end_index'][0], windows['start_index'][1])
        self.assertEqual(windows['end_ms'][0], windows['start_ms'][1])
        self.assertEqual(windows['start_ms'][0], 360011)
        self.assertEqual(coverage['fit']['complete_windows'], 2)
        self.assertAlmostEqual(coverage['fit']['actual_horizon_seconds']['min'], 180.004)

    def test_mutating_future_changes_label_but_not_that_window_start_features(self):
        args = list(self.fixture())
        before, _ = build_windows(*args)
        args[0] = args[0].copy()
        args[0][1, args[4].index('gold_diff')] = args[0][0, args[4].index('gold_diff')]
        after, _ = build_windows(*args)
        self.assertEqual(before['label'][0], 1)
        self.assertEqual(after['label'][0], 0)
        np.testing.assert_array_equal(before['x'][0], after['x'][0])

    def test_kill_threshold_is_inclusive_and_team_oriented(self):
        args = list(self.fixture())
        args[0][:, args[4].index('gold_diff')] = 0
        args[0][1:, args[4].index('kills_diff')] = -3*focal_sign('SYNTHETIC_1')
        windows, _ = build_windows(*args)
        self.assertEqual(windows['label'][0], 1)
        self.assertEqual(windows['kill_change'][0], -3)

    def test_missing_endpoints_are_counted_without_terminal_inference(self):
        windows, coverage = build_windows(*self.fixture((6, 9)))
        self.assertEqual(len(windows['label']), 1)
        self.assertEqual(coverage['fit']['missing_endpoint_clock_bin'], 1)
        self.assertEqual(coverage['fit']['missing_start_clock_bin'], 9)
        self.assertIn('no inferred terminal', coverage['fit']['exclusion_interpretation'])

    def test_role_mismatch_is_rejected(self):
        args = list(self.fixture())
        args[3] = np.array(['fit', 'calibration', 'fit'])
        with self.assertRaisesRegex(ValueError, 'role mismatch'):
            build_windows(*args)

    def test_orientation_retains_time_and_availability_and_encodes_side(self):
        names = list(ALL_FEATURES)
        x = np.ones((2, len(names)))
        oriented = orient_features(x, names, [1, -1])
        self.assertEqual(oriented[1, names.index('time_min')], 1)
        self.assertEqual(oriented[1, names.index('gold_diff_slope_available')], 1)
        self.assertEqual(oriented[1, names.index('gold_diff')], -1)
        self.assertEqual(oriented[1, names.index('gold_diff_slope_per_min')], -1)
        np.testing.assert_array_equal(oriented[:, -1], [1, 0])

    def test_wp_outputs_are_complemented_for_red_without_a_symmetry_assumption(self):
        scores = wp_rank_scores([.7, .7], [.6, .6], [1, -1])
        np.testing.assert_allclose(scores['wp_disagreement'], [.1, -.1])
        np.testing.assert_allclose(scores['low_current_wp'], [.3, .7])

    def test_fixed_budget_and_ties_are_deterministic(self):
        selected = select_windows([.5, .5, .5, .9], [0, 0, 0, 1], [9, 6, 12, 6])
        np.testing.assert_array_equal(selected, [True, True, False, True])

    def test_precision_includes_games_with_no_positive_windows(self):
        ids, rows, _ = game_retrieval(
            np.array([.9, .8, .1, .9, .8]), np.array([1, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1]), np.array([6, 9, 12, 6, 9]), np.array([9, 12, 15, 9, 12]))
        point = retrieval_point(rows)
        self.assertEqual(point[0], 1)
        self.assertEqual(point[1], .5)
        self.assertEqual(point[3], 1)
        self.assertEqual(point[4], .5)
        np.testing.assert_array_equal(ids, [0, 1])

    def test_paired_comparison_keeps_zero_difference_and_checks_population(self):
        rows = np.tile([2., 2., 1., 6.], (120, 1))
        comparison = paired_retrieval(rows, rows.copy(), 100)
        self.assertEqual(comparison['macro_recall']['estimate'], 0)
        self.assertEqual(comparison['macro_recall']['ci95'], [0, 0])
        self.assertFalse(comparison['useful_signal_gate'])
        mismatched = rows.copy()
        mismatched[0, 0] = 3
        with self.assertRaisesRegex(ValueError, 'identical games'):
            paired_retrieval(rows, mismatched, 100)

    def test_gate_requires_positive_game_count_and_precision_guard(self):
        reference = np.tile([2., 2., 1., 6.], (120, 1))
        candidate = np.tile([2., 2., 2., 6.], (120, 1))
        self.assertTrue(paired_retrieval(candidate, reference, 100)['useful_signal_gate'])
        self.assertFalse(paired_retrieval(candidate[:99], reference[:99], 100)['useful_signal_gate'])
        # Gain in macro recall can still lose precision: large-positive games lose many hits.
        reference = np.vstack((np.tile([1., 2., 0., 6.], (100, 1)), np.tile([10., 2., 2., 6.], (100, 1))))
        candidate = np.vstack((np.tile([1., 2., 1., 6.], (100, 1)), np.tile([10., 2., 0., 6.], (100, 1))))
        result = paired_retrieval(candidate, reference, 100)
        self.assertGreater(result['macro_recall']['estimate'], 0)
        self.assertLess(result['precision_all_selected']['estimate'], -.02)
        self.assertFalse(result['useful_signal_gate'])

    def test_baseline_selection_uses_supplied_calibration_counts_only(self):
        low = np.array([[1., 2., 0., 6.], [1., 2., 1., 6.]])
        slope = np.array([[1., 2., 1., 6.], [1., 2., 1., 6.]])
        self.assertEqual(choose_baseline({'low_current_wp': low, 'worst_gold_slope': slope}), 'worst_gold_slope')
        self.assertEqual(choose_baseline({'low_current_wp': low, 'worst_gold_slope': low}), 'low_current_wp')

    def test_direct_risk_fit_predicts_new_data_with_saved_calibrator(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(300, len(ALL_FEATURES)+1))
        labels = (x[:, 0] > 0).astype(int)
        artifact = risk_fit(x[:200], labels[:200], np.arange(200), x[200:250], labels[200:250],
                            np.arange(50), [0, 1], logistic=True)
        probability = risk_predict(artifact, x[250:])
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertGreater(probability[labels[250:] == 1].mean(), probability[labels[250:] == 0].mean())

    def test_quarantine_ancestor_is_rejected_before_file_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'cache').mkdir()
            (root/'QUARANTINED.md').write_text('synthetic quarantine', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'quarantined input ancestor'):
                reject_quarantined(root/'cache'/'unopened.npz')

    def test_original_roles_and_reservations_override_safe_cache_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root/'split_assignment.json'
            split.write_text(json.dumps({'SAFE_1': 'train', 'SEALED_1': 'test'}), encoding='utf-8')
            split_hash = hashlib.sha256(split.read_bytes()).hexdigest()
            manifest = {'schema': 'winprob-admission-v1', 'state': 'temporal_manifest_ready',
                        'source_data_root': str(root),
                        'split_metadata_sources': [{'path': split.name, 'sha256': split_hash}],
                        'games': [{'match_id': 'SAFE_1', 'role': 'train', 'historical_validation': False,
                                   'patch': 'synthetic', 'game_creation_ms': 1},
                                  {'match_id': 'SEALED_1', 'role': 'train', 'historical_validation': False,
                                   'patch': 'synthetic', 'game_creation_ms': 1}]}
            path = root/'admission.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix('.sha256').write_text(value+'  '+path.name+'\n', encoding='ascii')
            source = {'admission_manifest': str(path), 'admission_manifest_sha256': value,
                      'games': [{'match_id': 'SAFE_1', 'source_role': 'train', 'role': 'fit',
                                 'patch': 'synthetic', 'game_creation_ms': 1}]}
            metadata = {'admission_manifest_path': str(path), 'admission_manifest_sha256': value}
            verify_admitted_sources(source, metadata)
            source['games'][0]['source_role'] = 'development'
            with self.assertRaisesRegex(ValueError, 'source role'):
                verify_admitted_sources(source, metadata)
            source['games'][0]['source_role'] = 'train'
            source['games'][0]['match_id'] = 'SEALED_1'
            with self.assertRaisesRegex(ValueError, 'reserved'):
                verify_admitted_sources(source, metadata)
            metadata['admission_manifest_sha256'] = 'incorrect'
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                verify_admitted_sources(source, metadata)


if __name__ == '__main__':
    unittest.main()
