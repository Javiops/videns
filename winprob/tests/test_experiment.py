"""Synthetic experiment-contract tests; no archive reads and no model training."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from winprob import experiment
from winprob.features import ALL_FEATURES
from winprob.metrics import game_weights, paired_comparison, summarize


class RecordingModel:
    """Avoid expensive fitting while inspecting the exact arrays each arm sees."""

    records = []
    on_predict = None

    def __init__(self, kind, indices):
        self.kind = kind
        self.indices = list(indices)

    def fit(self, x, y, games):
        self.records.append(("fit", self.kind, games.copy(), y.copy(), x.copy()))
        return self

    def calibrate(self, x, y, games):
        self.records.append(("calibrate", self.kind, games.copy(), y.copy(), x.copy()))
        return self

    def raw(self, x):
        return np.full(len(x), .5)

    def predict(self, x):
        if type(self).on_predict is not None:
            callback = type(self).on_predict
            type(self).on_predict = None
            callback()
        return self.raw(x)


def synthetic_cache():
    """120 complete games per temporal role, five states per game."""
    games = np.repeat(np.arange(360), 5)
    minutes = np.tile(np.array([3, 4, 5, 6, 10]), 360)
    roles = np.repeat(np.array(["fit"] * 120 + ["calibration"] * 120 + ["development"] * 120), 5)
    x = np.zeros((len(games), len(ALL_FEATURES)))
    x[:, ALL_FEATURES.index("time_min")] = minutes
    x[:, ALL_FEATURES.index("gold_diff")] = games * 10
    return {
        "x": x, "y": (games % 2).astype(np.int8), "game": games,
        "role": roles, "timestamp_ms": minutes * 60000,
        "start_ms": 1700000000000 + games * 60000,
        "patch": np.full(len(games), "16.19"),
        "feature_names": np.array(ALL_FEATURES),
    }


class TrainingSamplingTests(unittest.TestCase):
    def test_one_state_per_game_interval_with_order_invariant_selection(self):
        games = np.array([1, 1, 1, 1, 1, 2, 2, 2])
        timestamps = np.array([180000, 240000, 300000, 540000, 600000, 180000, 240000, 300000])
        selected = experiment.sample_training_indices(games, timestamps)
        groups = [(int(games[i]), int(timestamps[i]) // 300000) for i in selected]
        expected = set(zip(games, timestamps // 300000))
        self.assertEqual(set(groups), expected)
        self.assertEqual(len(groups), len(expected))
        reverse = np.arange(len(games))[::-1]
        reordered = experiment.sample_training_indices(games[reverse], timestamps[reverse])
        self.assertEqual(set(zip(games[selected], timestamps[selected])),
                         set(zip(games[reverse][reordered], timestamps[reverse][reordered])))

    def test_label_permutation_preserves_games_and_patch_week_counts(self):
        game_ids = np.arange(40)
        counts = game_ids % 4 + 1
        games = np.repeat(game_ids, counts)
        y = (games % 2).astype(np.int8)
        patch = np.where(games < 20, "16.18", "16.19")
        starts = (games // 10) * 7 * 86400000
        original_y = y.copy()
        shuffled = experiment.permute_game_labels(y, games, starts, patch, 123)
        self.assertTrue(np.array_equal(y, original_y))
        self.assertTrue(np.array_equal(shuffled, experiment.permute_game_labels(y, games, starts, patch, 123)))
        self.assertTrue(np.any(shuffled != y))
        for gid in game_ids:
            self.assertEqual(len(np.unique(shuffled[games == gid])), 1)
        for first_gid in (0, 10, 20, 30):
            old = [int(y[np.flatnonzero(games == gid)[0]]) for gid in range(first_gid, first_gid + 10)]
            new = [int(shuffled[np.flatnonzero(games == gid)[0]]) for gid in range(first_gid, first_gid + 10)]
            self.assertEqual(sorted(old), sorted(new))

    def test_equal_game_metrics_are_invariant_to_one_games_frame_duplication(self):
        games = np.array([10, 20])
        y = np.array([0, 1])
        p = np.array([.1, .3])
        original = summarize(y, p, games, resamples=20)
        repeated = np.array([0] * 20 + [1])
        duplicate = summarize(y[repeated], p[repeated], games[repeated], resamples=20)
        weights = game_weights(games[repeated])
        self.assertAlmostEqual(float(weights[games[repeated] == 10].sum()), 1.0)
        self.assertAlmostEqual(float(weights[games[repeated] == 20].sum()), 1.0)
        for metric in ("log_loss", "brier", "accuracy"):
            self.assertAlmostEqual(original[metric]["estimate"], duplicate[metric]["estimate"])
            np.testing.assert_allclose(original[metric]["ci95"], duplicate[metric]["ci95"])
            self.assertEqual(duplicate[metric]["games"], 2)
        self.assertAlmostEqual(original["ece20"], duplicate["ece20"])

    def test_paired_metric_uses_same_game_weighted_population(self):
        games = np.array([1, 1, 1, 2])
        y = np.array([0, 0, 0, 1])
        reference = np.full(4, .5)
        candidate = np.array([.1, .1, .1, .9])
        report = paired_comparison(y, reference, candidate, games, resamples=20)
        self.assertAlmostEqual(report["brier"]["estimate"], .25 - .01)
        self.assertEqual(report["brier"]["games"], 2)
        self.assertGreater(report["log_loss"]["ci95"][0], 0)


class ExperimentPopulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="winprob-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "features.npz"
        self.output = self.root / "result"
        self.data = synthetic_cache()
        RecordingModel.records = []
        RecordingModel.on_predict = None
        self.addCleanup(setattr, RecordingModel, "on_predict", None)

    def run_tiny(self):
        np.savez_compressed(self.cache, **self.data)

        def dump_stub(model, path, **kwargs):
            Path(path).write_bytes(b"synthetic model fixture")

        with mock.patch.object(experiment, "CalibratedModel", RecordingModel), \
                mock.patch.object(experiment.joblib, "dump", side_effect=dump_stub):
            return experiment.run_experiment(self.cache, self.output, resamples=20)

    def test_fit_calibration_disjoint_and_report_matches_saved_development_rows(self):
        report = self.run_tiny()
        self.assertEqual(report["fit_sampled_frames"], 360)
        for phase, kind, games, y, x in RecordingModel.records:
            expected = set(range(120)) if phase == "fit" else set(range(120, 240))
            self.assertEqual(set(games.tolist()), expected)
            self.assertFalse(np.any(games >= 240))
            self.assertEqual(len(games), 360 if phase == "fit" else 600)
        for role in ("fit", "calibration", "development"):
            self.assertEqual(report["population"][role]["games"], 120)
            self.assertEqual(report["population"][role]["frames"], 600)
        with np.load(self.output / "development_predictions.npz", allow_pickle=False) as persisted:
            self.assertEqual(set(persisted["game"].tolist()), set(range(240, 360)))
            self.assertEqual(len(persisted["y"]), 600)
            self.assertTrue(np.array_equal(persisted["y"], self.data["y"][self.data["role"] == "development"]))
        for kind in ("constant", "logistic", "snapshot", "trajectory"):
            model = report["models"][kind]
            self.assertEqual((model["games"], model["frames"]), (120, 600))
            for metric in ("log_loss", "brier", "accuracy"):
                self.assertEqual(model[metric]["games"], 120)
        self.assertFalse(report["gates"]["practical_decision_benefit_established"])

    def test_split_collision_rejected_before_any_model_fit(self):
        self.data["game"][self.data["game"] == 120] = 0
        with self.assertRaisesRegex(ValueError, "split"):
            self.run_tiny()
        self.assertEqual(RecordingModel.records, [])

    def test_nonbinary_labels_rejected(self):
        self.data["y"][self.data["y"] == 1] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            self.run_tiny()

    def test_future_fit_role_rejected(self):
        roles = self.data["role"].copy()
        self.data["role"][roles == "fit"] = "development"
        self.data["role"][roles == "development"] = "fit"
        with self.assertRaisesRegex(ValueError, "chronolog|temporal"):
            self.run_tiny()
        self.assertEqual(RecordingModel.records, [])

    def test_unknown_feature_rejected_before_model_fit(self):
        self.data["feature_names"] = np.append(self.data["feature_names"], "winner")
        self.data["x"] = np.column_stack([self.data["x"], self.data["y"]])
        with self.assertRaisesRegex(ValueError, "feature"):
            self.run_tiny()
        self.assertEqual(RecordingModel.records, [])

    def test_duplicate_feature_rejected_before_model_fit(self):
        self.data["feature_names"][-1] = "gold_diff"
        with self.assertRaisesRegex(ValueError, "feature"):
            self.run_tiny()
        self.assertEqual(RecordingModel.records, [])

    def test_cache_change_during_fit_refuses_result(self):
        def change_source():
            with self.cache.open("ab") as handle:
                handle.write(b"source changed during run")

        RecordingModel.on_predict = change_source
        with self.assertRaisesRegex(RuntimeError, "cache|source"):
            self.run_tiny()
        self.assertFalse((self.output / "report.json").exists())

    def test_missing_time_bands_remain_visible_as_unmeasured(self):
        report = self.run_tiny()
        self.assertIn("20-30min", report["time_bands"])
        self.assertIn("30-40min", report["time_bands"])
        for key in ("20-30min", "30-40min"):
            self.assertEqual(report["time_bands"][key]["games"], 0)
            self.assertEqual(report["time_bands"][key]["frames"], 0)
            self.assertFalse(report["time_bands"][key]["measured"])


if __name__ == "__main__":
    unittest.main()
