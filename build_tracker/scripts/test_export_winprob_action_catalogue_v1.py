"""Synthetic full-action identity and SAVE/ward separation tests."""
import unittest
import export_winprob_action_catalogue_v1 as exporter


class CatalogueTests(unittest.TestCase):
    def test_save_and_ward_actions_remain_distinct_in_empty_main_group(self):
        data=exporter.construct({'baskets_A':['2055:1','1036:2,2055:1','1036:2'],'group_main':[[],[[1036,2]]]})
        self.assertEqual(data['full_actions'][0],[])
        self.assertEqual(data['full_action_kinds'][0],'observed_no_buy_death')
        self.assertEqual(data['group0_full_action_ids'],[0,1])
        self.assertEqual(data['full_to_group'],[0,0,1,1])
        self.assertEqual(data['full_actions'][2],[[1036,2],[2055,1]])
    def test_noncanonical_or_sentinel_physical_items_rejected(self):
        for key in ('999999:1','1036:0','01036:1','2055:1,1036:1','1036:1,1036:1'):
            with self.assertRaises(ValueError):exporter.parse_basket(key)
    def test_missing_main_support_or_duplicate_action_rejected(self):
        with self.assertRaises(ValueError):exporter.construct({'baskets_A':['1036:1'],'group_main':[[]]})
        with self.assertRaises(ValueError):exporter.construct({'baskets_A':['1036:1','1036:1'],'group_main':[[],[[1036,1]]]})


if __name__=='__main__':unittest.main()
