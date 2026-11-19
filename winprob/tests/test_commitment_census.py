import unittest
from winprob.commitment_census import analyze_episode,ancestors,completion_costs,money_status

ITEMS={str(item):{'gold':{'total':cost},'from':parents} for item,cost,parents in (
    (1001,300,[]),(1029,300,[]),(1033,400,[]),(3047,1200,['1001','1029']),
    (3111,1250,['1001','1033']),(9999,800,['1029']),(2055,75,[]))}
RECIPES={int(item):ancestors(ITEMS,int(item)) for item in ITEMS}

def node(index,inventory=None,basket=None,gold=500,end=None):
    return {'time':index*10000,'end':index*10000+100 if end is None else end,
            'inventory':inventory or {},'basket':basket or {},'gold':gold}

def analyze(nodes):return analyze_episode(nodes,{3047,3111},RECIPES,ITEMS)

class CommitmentCensusTests(unittest.TestCase):
    def test_neutral_component_trace_to_matching_boot(self):
        value=analyze([node(0,basket={1029:1}),node(1,{1029:1,1001:1},{3047:1}),node(2,{3047:1})])
        self.assertEqual(value['initiation']['status'],'neutral_single')
        self.assertTrue(value['initiation']['pure_main'])
        self.assertEqual(value['trace']['status'],'matching_defensive_boot')
        self.assertEqual(value['trace']['elapsed_ms'],10000)

    def test_other_recipe_use_not_relabelled_as_boot_intent(self):
        value=analyze([node(0,basket={1029:1}),node(1,{1029:1},{9999:1}),node(2,{9999:1},{3047:1}),node(3,{9999:1,3047:1})])
        self.assertEqual(value['trace']['status'],'other_item')
        self.assertEqual(value['trace']['consumer_item'],9999)

    def test_first_ambiguous_initiation_is_not_replaced(self):
        value=analyze([node(0,{1029:1},{1029:1}),node(1,basket={1033:1}),node(2,{1033:1})])
        self.assertEqual(value['initiation']['status'],'preowned_component')
        self.assertEqual(value['initiation']['index'],0)
        self.assertIsNone(value['trace'])

    def test_both_and_duplicate_components_distinguished(self):
        self.assertEqual(analyze([node(0,basket={1029:1,1033:1})])['initiation']['status'],'both_components')
        self.assertEqual(analyze([node(0,basket={1029:2})])['initiation']['status'],'duplicate_component')

    def test_censored_consumer_not_claimed_verified(self):
        value=analyze([node(0,basket={1033:1}),node(1,{1033:1},{3111:1})])
        self.assertEqual(value['trace']['status'],'consumer_without_following_state')

    def test_unexplained_disappearance_is_not_consumption(self):
        value=analyze([node(0,basket={1029:1}),node(1,{1029:1}),node(2)])
        self.assertEqual(value['trace']['status'],'disappeared_without_recipe_receipt')

    def test_mixed_consumer_is_ambiguous(self):
        value=analyze([node(0,basket={1029:1}),node(1,{1029:1},{3047:1,9999:1}),node(2,{3047:1,9999:1})])
        self.assertEqual(value['trace']['status'],'multiple_compatible_consumers')

    def test_closure_overlap_is_explicit(self):
        value=analyze([node(0,basket={1029:1},end=10000),node(1,{1029:1})])
        self.assertEqual(value['trace']['status'],'initiation_closure_overlap')

    def test_wards_preserve_pure_main_but_cost_money(self):
        value=analyze([node(0,basket={1033:1,2055:1},gold=450),node(1,{1033:1})])
        self.assertTrue(value['initiation']['pure_main'])
        self.assertEqual(value['initiation']['estimated_pair_money'],'estimated_insufficient')

    def test_exact_threshold_is_boundary_not_wallet_inflation(self):
        self.assertEqual(money_status(400.,400),'numerical_boundary')
        self.assertEqual(money_status(399.,400),'estimated_insufficient')
        self.assertEqual(money_status(401.,400),'estimated_sufficient')
        self.assertEqual(completion_costs({1001:1,1029:1},ITEMS),{'3047':600.,'3111':950.})
        self.assertIsNone(completion_costs({2422:1},ITEMS))

    def test_already_booted_episode_cannot_initiate(self):
        self.assertIsNone(analyze([node(0,{3047:1},{1033:1})])['initiation'])

    def test_future_labels_cannot_change_initiation_inputs(self):
        prefix=[node(0,basket={1029:1}),node(1,{1029:1})]
        left=analyze(prefix+[node(2,{1029:1},{3047:1}),node(3,{3047:1})])
        right=analyze(prefix+[node(2,{1029:1},{9999:1}),node(3,{9999:1})])
        self.assertEqual(left['initiation'],right['initiation'])
        self.assertNotEqual(left['trace']['status'],right['trace']['status'])

if __name__=='__main__':unittest.main()
