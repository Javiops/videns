"""Synthetic feature-block and row-identity contracts; no run inputs are read."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablate_rich import COMPARISONS, feature_arms, validate_projection
from winprob.features import ALL_FEATURES
from winprob.features_rich import ALL_RICH_FEATURES


class AblationContracts(unittest.TestCase):
    def test_declared_blocks_and_combined_union(self):
        names = ALL_RICH_FEATURES
        arms = {name: {names[index] for index in indices} for name, indices in feature_arms(names).items()}
        base = set(ALL_FEATURES)
        self.assertEqual(arms["total"], base | {"rich_gold_total_sum"})
        self.assertEqual((len(arms["total"]), len(arms["allocation"]), len(arms["geometry"])), (20, 32, 48))
        self.assertEqual(arms["allocation"] | arms["geometry"], set(names))
        self.assertEqual(arms["allocation"] & arms["geometry"], base)
        self.assertIn("rich_gold_weighted_enemy_distance_mean_diff", arms["geometry"])
        self.assertIn("rich_local_gold_balance_min_sum", arms["geometry"])
        self.assertNotIn("rich_gold_total_sum", arms["geometry"])
        self.assertNotIn("rich_gold_std_diff", arms["geometry"])

    def test_unknown_duplicate_and_missing_names_rejected(self):
        for names in (ALL_RICH_FEATURES + ("winner",), ALL_RICH_FEATURES + (ALL_RICH_FEATURES[0],), ALL_RICH_FEATURES[:-1]):
            with self.assertRaisesRegex(ValueError, "schema"):
                feature_arms(names)

    def test_feature_selection_tracks_names_under_column_permutation(self):
        original = feature_arms(ALL_RICH_FEATURES)
        reversed_names = ALL_RICH_FEATURES[::-1]
        reordered = feature_arms(reversed_names)
        for name in original:
            self.assertEqual({ALL_RICH_FEATURES[i] for i in original[name]}, {reversed_names[i] for i in reordered[name]})

    def fixture(self):
        rich = {"feature_names": np.array(ALL_RICH_FEATURES), "x": np.arange(3 * len(ALL_RICH_FEATURES)).reshape(3, -1)}
        for key in ("y", "game", "role", "timestamp_ms", "start_ms", "split_time_ms", "patch"):
            rich[key] = np.array([0, 1, 2])
        original = copy.deepcopy(rich)
        original["feature_names"] = np.array(ALL_FEATURES)
        original["x"] = rich["x"][:, [ALL_RICH_FEATURES.index(name) for name in ALL_FEATURES]].copy()
        return rich, original

    def test_projection_requires_original_values_and_row_identity(self):
        rich, original = self.fixture()
        validate_projection(rich, original)
        for key in ("y", "game", "role", "timestamp_ms", "split_time_ms"):
            changed = copy.deepcopy(rich)
            changed[key][0] = 999
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_projection(changed, original)
        changed = copy.deepcopy(rich)
        changed["x"][0, ALL_RICH_FEATURES.index("gold_diff")] += 1
        with self.assertRaisesRegex(ValueError, "19 features"):
            validate_projection(changed, original)

    def test_all_fixed_contrasts_remain_declared(self):
        self.assertEqual(len(COMPARISONS), 9)
        self.assertEqual(len(set(COMPARISONS)), 9)
        for reference in ("total", "allocation", "geometry", "original_trajectory"):
            self.assertIn(("combined", reference), COMPARISONS)
        self.assertIn(("allocation", "total"), COMPARISONS)


if __name__ == "__main__":
    unittest.main()
