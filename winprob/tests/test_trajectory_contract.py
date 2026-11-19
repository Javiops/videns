import copy
import unittest
from winprob.trajectory_contract import SCHEMA, compile_transitions


def episode(outcome=1):
    nodes=[]
    for i, timestamp in enumerate((100, 200, 300)):
        nodes.append({'node_id': f'EUW1_10:1:{i}', 'kind':'decision', 'timestamp_ms':timestamp,
                      'visit_end_ms':timestamp+1, 'reward':0, 'done':False,
                      'prestate':{'gold_est':1000,'inventory':[1001],'others':[]},
                      'action':{'encoding':'item-counts-v1','label_counts':{'1036':2,'2055':1}},
                      'next_ref': f'EUW1_10:1:{i+1}' if i<2 else 'EUW1_10:1:terminal'})
    return {'schema':SCHEMA,'match_id':'EUW1_10','participant_id':1,
            'whole_player_game_eligible':True,'ineligibility_reasons':[],'decisions':nodes,
            'terminal':{'node_id':'EUW1_10:1:terminal','kind':'terminal','timestamp_ms':400,
                        'prestate':None,'action':None,'reward':outcome,'done':True,'next_ref':None}}


class TrajectoryContractTests(unittest.TestCase):
    def test_terminal_arrival_reward_reaches_last_action_once(self):
        for outcome in (0, 1):
            rows=compile_transitions(episode(outcome))
            self.assertEqual([r['reward'] for r in rows],[0,0,outcome])
            self.assertEqual([r['bootstrap_multiplier'] for r in rows],[1,1,0])
            self.assertEqual(rows[-1]['next_state_ref'],None)

    def test_multisets_and_wards_are_preserved(self):
        self.assertEqual(compile_transitions(episode())[0]['action']['label_counts'],{'1036':2,'2055':1})

    def test_no_stitching_across_unmodeled_decisions(self):
        record=episode(); record['decisions'][1]['kind']='gap'
        with self.assertRaises(ValueError): compile_transitions(record)
        record=episode(); record['whole_player_game_eligible']=False
        with self.assertRaises(ValueError): compile_transitions(record)

    def test_no_reward_doubling(self):
        record=episode(); record['decisions'][-1]['reward']=1
        with self.assertRaises(ValueError): compile_transitions(record)

    def test_future_labels_and_legacy_roles_do_not_enter_state(self):
        for key in ('source_target_id','gold_arrival_true','win','role'):
            record=episode(); record['decisions'][0]['prestate'][key]=123
            with self.assertRaises(ValueError): compile_transitions(record)

    def test_identity_and_chain_fail_closed(self):
        record=episode(); record['decisions'][0]['next_ref']='EUW1_10:1:2'
        with self.assertRaises(ValueError): compile_transitions(record)
        record=episode(); record['terminal']['node_id']='EUW1_11:1:terminal'
        with self.assertRaises(ValueError): compile_transitions(record)


if __name__=='__main__': unittest.main()
