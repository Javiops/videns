from copy import deepcopy
import unittest
import numpy as np
from export_winprob_scoreboard_v1 import ordered_scores,poison_non_features


class ScoreboardProjectionTests(unittest.TestCase):
    def setUp(self):
        self.board=[{'participant_id':i,'champion_id':100+i,'team_id':100 if i<=5 else 200,
            'is_self':i==3,'kills':i,'deaths':i+1,'assists':i+2,'team_position':'unused',
            'gold':99999,'items':[{'item_id':1001}]} for i in range(10,0,-1)]
        self.event={'board':self.board,'label_bought':[{'item_id':1036}],'win_team':100}
        self.index={str(100+i):i for i in range(1,11)}
        self.order=[3,1,2,4,5,6,7,8,9,10]

    def project(self,event):return ordered_scores(event,3,self.index,np.array(self.order))

    def test_buyer_and_team_order_exact(self):
        kda,pids=self.project(self.event)
        np.testing.assert_array_equal(pids,self.order)
        np.testing.assert_array_equal(kda,[[i,i+1,i+2] for i in self.order])
        self.assertEqual(kda.dtype,np.int32)

    def test_board_permutation_invariant(self):
        other=deepcopy(self.event);other['board'].reverse()
        for left,right in zip(self.project(self.event),self.project(other)):np.testing.assert_array_equal(left,right)

    def test_target_outcome_role_gold_inventory_mutations_ignored(self):
        original=deepcopy(self.event)
        for left,right in zip(self.project(self.event),self.project(poison_non_features(self.event))):np.testing.assert_array_equal(left,right)
        self.assertEqual(original,self.event)

    def test_missing_score_is_not_zero(self):
        event=deepcopy(self.event);del event['board'][0]['assists']
        with self.assertRaises(ValueError):self.project(event)

    def test_invalid_scores_rejected(self):
        for bad in (True,-1,1.5,None):
            event=deepcopy(self.event);event['board'][0]['kills']=bad
            with self.assertRaises(ValueError):self.project(event)

    def test_duplicate_roster_rejected(self):
        event=deepcopy(self.event);event['board'][0]['participant_id']=1
        with self.assertRaises(ValueError):self.project(event)

    def test_focal_and_cache_identity_rejected(self):
        with self.assertRaises(ValueError):ordered_scores(self.event,2,self.index,np.array(self.order))
        with self.assertRaises(ValueError):ordered_scores(self.event,3,self.index,np.arange(1,11))


if __name__=='__main__':unittest.main()
