"""Synthetic pre-A alignment tests; no production data access."""
from copy import deepcopy
import unittest

import align_winprob_revision_contexts_v1 as align
from test_export_winprob_trajectories_v1 import example


class Dragon:
    def gold_block(self,item):return {'total':350 if item==1036 else 400}


def fixture():
    pid=1;ta,tu,tb=200000,201000,203000
    pair={'match_id':'EUW1_1','participant_id':pid,'feature_cutoff_ms':ta,'undo_ms':tu,'decision_timestamp_ms':tb,
          'retention_end_ms':tb+10000,'action_a':1036,'action_b':1052,'strict_transaction_pair':True,'strict_context_pair':True,
          'pair_class':'different_item_revision','rejection_reasons':[],'certificate':{'full_static_price_a':350,'full_static_price_b':400},
          'feature_frame_ms':180000,'focal_pre_frame':{'currentGold':500,'level':1,'totalGold':1000},'champion_id':1,'team_id':100,
          'source_line_number':10,'game_candidate_index':0,'role':'fit','patch':'16.18'}
    ops=[{'type':'ITEM_PURCHASED','participantId':1,'timestamp':ta,'itemId':1036},
         {'type':'ITEM_UNDO','participantId':1,'timestamp':tu,'beforeId':1036,'afterId':0,'goldGain':350},
         {'type':'ITEM_PURCHASED','participantId':1,'timestamp':tb,'itemId':1052}]
    timeline={'info':{'frames':[{'timestamp':240000,'events':ops}]}}
    board=[{'is_self':i==1,'participant_id':i,'champion_id':i,'team_id':100 if i<=5 else 200,'items':[1001] if i==1 else [],'gold':1000} for i in range(1,11)]
    event={'type':'shop','event_index':1,'participant_id':1,'puuid':'p1','team_id':100,'ts':ta,'ts_end':tb,
           'opening_sales':[],'opening_sale_gold':0,'inventory_before':[1001],'decision_inventory':[1001],
           'inventory_after':[1001,1052],'label_bought':[1052],'training_eligible':True,'source_quality_issues':[],
           'gold_est':500,'gold_est_version':'prequential-v3','gold_gap':{'frame_ts':180000},'gold_left':500,'level':1,'board':board}
    mask={'admitted_decision_ids':[{'event_index':1,'participant_id':1,'ts':ta}]}
    players={1:{'puuid':'p1','win':True}}
    debug={'frame_series':{1:[(180000,500,1000),(240000,100,1100)]},'gold_events':{1:[(ta,350,'spend')]}}
    def projector(e):
        row=example(e);row['label_ids']=[e['label_bought'][0]['item_id'] if isinstance(e['label_bought'][0],dict) else e['label_bought'][0]]
        return row
    def gold(events,frames,spends):
        assert all(value[0]<ta for value in frames[1]+spends[1])
        for e in events:e.update(gold_est=500,gold_gap={'frame_ts':180000},gold_est_version='prequential-v3')
    return pair,[event],mask,players,projector,gold,debug,timeline,Dragon()


class AlignmentTests(unittest.TestCase):
    def test_exact_context_aligns_with_future_independence(self):
        result=align.align_pair(*fixture())
        self.assertEqual(result['status'],'aligned');self.assertTrue(result['checks']['wallet_strict_prefix_reproduced'])
        self.assertTrue(result['checks']['label_future_mutation_invariant'])
        self.assertEqual(result['context_timestamp_ms'],200000)
        self.assertEqual(result['certificate_b_timestamp_ms'],203000)
    def test_timestamp_match_alone_does_not_pass_inventory_or_sales(self):
        args=list(fixture());args[1][0]['opening_sale_gold']=50
        with self.assertRaisesRegex(ValueError,'Opening-sale'):align.align_pair(*args)
        args=list(fixture());args[1][0]['decision_inventory']=[]
        with self.assertRaisesRegex(ValueError,'Decision inventory'):align.align_pair(*args)
    def test_closed_b_timestamp_cannot_substitute_for_pre_a(self):
        args=list(fixture());args[1][0]['ts']=203000
        with self.assertRaisesRegex(ValueError,'unique qualified visit'):align.align_pair(*args)
    def test_extra_operation_fails(self):
        args=list(fixture());args[7]['info']['frames'][0]['events'].append({'type':'ITEM_SOLD','participantId':1,'timestamp':204000,'itemId':1001})
        with self.assertRaisesRegex(ValueError,'Extra or missing'):align.align_pair(*args)
    def test_future_sensitive_wallet_and_projection_fail(self):
        args=list(fixture())
        def bad_gold(events,frames,spends):
            for e in events:e.update(gold_est=999,gold_gap={'frame_ts':180000},gold_est_version='prequential-v3')
        args[5]=bad_gold
        with self.assertRaisesRegex(ValueError,'Wallet estimator'):align.align_pair(*args)
        args=list(fixture());original=args[4]
        def leaky(e):
            row=original(e);row['gold_est']=e.get('gold',500);return row
        args[4]=leaky
        with self.assertRaisesRegex(ValueError,'Reference context depends'):align.align_pair(*args)


if __name__=='__main__':unittest.main()
