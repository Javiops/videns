from copy import deepcopy
import unittest
import numpy as np
from winprob.goal_census import goal_ids,goal_multiset,label_vocabulary,assign_episode_labels,summarize_mode


def item(parents=(),tags=(),depth=3,into=(),base=100):
    return {'from':list(map(str,parents)),'tags':list(tags),'depth':depth,'into':list(map(str,into)),
            'gold':{'base':base,'purchasable':True},'maps':{'11':True}}


class GoalCensusTests(unittest.TestCase):
    def test_exact_public_rule_excludes_ancestor_boots_free_and_tagged_items(self):
        items={'1':item(depth=1),'2':item(tags=['Boots'],depth=1),'3':item([1]),'4':item([2]),
               '5':item([1],tags=['GoldPer']),'6':item([1],base=0),'7':item([1],depth=2,into=[3]),
               '8':item([1],depth=2),'9':item([1])}
        catalogue={'full_actions':[[]]+[[[i,1]] for i in range(1,9)]}
        self.assertEqual(goal_ids({'items':items},catalogue),[3,8])

    def test_all_parents_and_entire_graph_are_validated(self):
        cat={'full_actions':[[],[[3,1]]]}
        with self.assertRaisesRegex(ValueError,'missing'):
            goal_ids({'items':{'1':item(tags=['Boots']),'3':item([1,999])}},cat)
        with self.assertRaisesRegex(ValueError,'cycle'):
            goal_ids({'items':{'1':item([2]),'2':item([1]),'3':item()}},cat)

    def test_multisets_preserve_duplicate_goal_copies_and_wards_do_not_become_goal(self):
        self.assertEqual(goal_multiset([[3,2],[2055,1],[4,1]],{3,4}),((3,2),(4,1)))
        cat={'full_actions':[[],[[3,2]],[[3,1],[4,1]],[[2055,1]]]}
        vocab=label_vocabulary(cat,{p:[3,4] for p in ('16.17','16.18','16.19')})
        self.assertEqual([v['type'] for v in vocab[2:]],['MULTIPLE','MULTIPLE'])
        self.assertIn({'type':'MULTIPLE','goal_counts':[[3,2]]},vocab)

    def test_two_modes_current_completion_and_strict_later_are_distinct(self):
        ts=np.array([0,100000,200000]);end=ts+100;codes=np.array([2,-1,3]);term=np.array([False,False,True]);elapsed=np.array([100000,100000,500000])
        result,T=assign_episode_labels(ts,end,codes,term,elapsed)
        self.assertEqual(result['including_current'][0].tolist(),[2,3,3])
        self.assertEqual(result['strictly_later'][0].tolist(),[3,3,0])
        self.assertEqual(result['strictly_later'][1].tolist(),[2,2,-1])

    def test_closure_at_horizon_is_inclusive_and_precedes_terminal_none(self):
        ts=np.array([0,300000]);end=np.array([1,360000]);codes=np.array([-1,2]);term=np.array([False,True]);elapsed=np.array([300000,60000])
        result,T=assign_episode_labels(ts,end,codes,term,elapsed)
        self.assertEqual(T,360000);self.assertEqual(result['strictly_later'][0].tolist(),[2,1])

    def test_late_closure_excluded_even_when_purchase_started_inside_window(self):
        ts=np.array([0,350000]);end=np.array([1,360001]);codes=np.array([-1,2]);term=np.array([False,True]);elapsed=np.array([350000,500000])
        result,T=assign_episode_labels(ts,end,codes,term,elapsed)
        self.assertEqual(result['including_current'][0][0],0)
        self.assertEqual(result['including_current'][1][0],-1)

    def test_none_terminal_versus_full_window_and_save_excluded(self):
        ts=np.array([0,500000]);end=ts+1;codes=np.array([-1,-1]);term=np.array([False,True]);elapsed=np.array([500000,100000])
        result,T=assign_episode_labels(ts,end,codes,term,elapsed)
        self.assertEqual(result['including_current'][0].tolist(),[0,1])

    def test_first_qualifying_visit_whole_multiset_never_merges_later_goal(self):
        ts=np.array([0,10,20]);end=ts+1;codes=np.array([-1,3,4]);term=np.array([False,False,True]);elapsed=np.array([10,10,100])
        result,_=assign_episode_labels(ts,end,codes,term,elapsed)
        self.assertEqual(result['strictly_later'][0][0],3);self.assertEqual(result['strictly_later'][1][0],1)

    def test_terminal_structure_and_crossing_intervals_fail(self):
        with self.assertRaisesRegex(ValueError,'terminal'):
            assign_episode_labels(np.array([0,10]),np.array([1,11]),np.array([-1,2]),np.array([True,True]),np.array([10,10]))
        with self.assertRaisesRegex(ValueError,'overlapping'):
            assign_episode_labels(np.array([0,10]),np.array([12,11]),np.array([-1,2]),np.array([False,True]),np.array([10,10]))

    def test_plugin_entropy_and_crossgame_variation_are_descriptive_counts(self):
        vocab=[{'type':'NONE_FULL_HORIZON','goal_counts':[]},{'type':'NONE_TERMINAL_HORIZON','goal_counts':[]},
               {'type':'SINGLE','goal_counts':[[3,1]]},{'type':'SINGLE','goal_counts':[[4,1]]}]
        result=summarize_mode(np.array([2,2,3,3]),np.array([1,1,3,3]),np.zeros(4,int),np.zeros(4,int),np.array([0,0,1,1]),vocab)
        summary=result['conditional_variation']
        self.assertAlmostEqual(summary['plugin_H_G_given_A_patch_nats'],np.log(2))
        self.assertEqual(summary['groups_with_variation_across_atleast2games'],1)
        self.assertEqual(summary['distinct_game_ordered_state_pairs'],8)
        self.assertEqual(summary['crossgame_goal_disagreement_fraction'],1)
        self.assertEqual(summary['rare_group_counts']['n2_to4'],1)

    def test_terminal_censoring_status_does_not_inflate_goal_identity_entropy(self):
        vocab=[{'type':'NONE_FULL_HORIZON','goal_counts':[]},{'type':'NONE_TERMINAL_HORIZON','goal_counts':[]}]
        result=summarize_mode(np.array([0,1]),np.array([-1,-1]),np.zeros(2,int),np.zeros(2,int),np.array([0,1]),vocab)
        self.assertEqual(result['state_counts']['NONE_FULL_HORIZON'],1)
        self.assertEqual(result['state_counts']['NONE_TERMINAL_HORIZON'],1)
        self.assertEqual(result['conditional_variation']['plugin_H_G_given_A_patch_nats'],0)
        self.assertEqual(result['conditional_variation']['groups_with_multiple_goal_labels'],0)

    def test_current_action_coarsening_and_distinct_later_goal_counts(self):
        vocab=[{'type':'NONE_FULL_HORIZON','goal_counts':[]},{'type':'NONE_TERMINAL_HORIZON','goal_counts':[]},
               {'type':'SINGLE','goal_counts':[[3,1]]},{'type':'SINGLE','goal_counts':[[4,1]]}]
        result=summarize_mode(np.array([2,3,3]),np.array([0,2,2]),np.array([1,1,2]),np.zeros(3,int),np.array([0,0,0]),vocab,np.array([2,2,3]))
        values=result['current_action_coarsening']
        self.assertEqual(values['labels_taken_from_current_completion'],2)
        self.assertEqual(values['later_goal_different_when_current_action_already_has_goal'],1)


if __name__=='__main__':unittest.main()
