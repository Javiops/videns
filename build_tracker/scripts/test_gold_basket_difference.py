"""Property checks for the nominal-gold multiset distance diagnostic."""
from collections import Counter
import unittest

from report_gold_basket_difference import first_card_counts, gold_components


class GoldBasketDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.prices = {1036: 350, 2055: 75, 1001: 350}

    def test_missing_ward_is_gold_not_a_whole_basket_failure(self):
        result = gold_components(Counter({1036: 1, 2055: 1}), Counter({1036: 1}), self.prices)
        self.assertEqual((result["shared_gold"], result["missing_gold"], result["extra_gold"],
                          result["unmatched_gold"]), (350, 75, 0, 75))

    def test_equal_spend_with_wrong_item_does_not_cancel(self):
        result = gold_components(Counter({1036: 1}), Counter({1001: 1}), self.prices)
        self.assertEqual(result["abs_total_gold_gap"], 0)
        self.assertEqual(result["unmatched_gold"], 700)

    def test_duplicate_copies_and_extra_item_are_separate(self):
        result = gold_components(Counter({1036: 2}), Counter({1036: 1, 2055: 1}), self.prices)
        self.assertEqual((result["shared_gold"], result["missing_gold"], result["extra_gold"],
                          result["unmatched_gold"]), (350, 350, 75, 425))

    def test_no_buy_with_purchase_is_extra_gold(self):
        result = gold_components(Counter(), Counter({2055: 1}), self.prices)
        self.assertEqual((result["shared_gold"], result["missing_gold"], result["extra_gold"],
                          result["unmatched_gold"]), (0, 0, 75, 75))

    def test_displayed_counts_must_match_purchase_sequence(self):
        policy = {"displayed_options": [{"kind": "buy", "item_counts": {"1036": 2},
                                         "purchase_sequence": [{"item_id": 1036, "cost": 350}]}]}
        with self.assertRaisesRegex(ValueError, "purchase sequence"):
            first_card_counts(policy)


if __name__ == "__main__":
    unittest.main()
