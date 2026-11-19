"""Synthetic FIT-only full-chain/catalogue/accounting tests."""
from copy import deepcopy
import gzip
import io
import unittest

import export_winprob_fit_trajectories_v2 as exporter
from test_export_winprob_trajectories_v1 import fixture,example


class FitTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.catalogue={'full_actions':[[],[[1036,2],[2055,1]],[[2055,1]]],
                        'full_action_kinds':['observed_no_buy_death','retained_net_purchase','retained_net_purchase'],
                        'full_to_group':[0,1,0]}
        self.index={exporter.producer.canonical(action):i for i,action in enumerate(self.catalogue['full_actions'])}
    def record(self):
        game,events,mask=fixture()
        return exporter.producer.trajectories_for_game(game,events,mask,example,600001,{'indexed_rows_sha256':'synthetic'})[0]
    def test_save_and_ward_only_are_not_collapsed(self):
        save={'encoding':'item-counts-v1','kind':'observed_no_buy_death','label_counts':{'999999':1}}
        ward={'encoding':'item-counts-v1','kind':'retained_net_purchase','label_counts':{'2055':1}}
        self.assertEqual(exporter.lookup_action(save,self.catalogue,self.index),0)
        self.assertEqual(exporter.lookup_action(ward,self.catalogue,self.index),2)
        with self.assertRaises(ValueError):exporter.lookup_action({**ward,'label_counts':{'999999':1}},self.catalogue,self.index)
    def test_complete_chain_keeps_counts_and_terminal_once_without_reference_roles(self):
        record,excluded=exporter.encode_record(self.record(),self.catalogue,'hash',self.index,3)
        self.assertIsNone(excluded);self.assertEqual(exporter.validate_chain(record),2)
        self.assertEqual(record['decisions'][0]['action']['full_action_id'],1)
        self.assertEqual(record['decisions'][0]['action']['label_counts'],{'1036':2,'2055':1})
        self.assertTrue(all('reference_compatibility' not in node for node in record['decisions']))
        self.assertEqual(record['source_quality']['game_binding_index'],3)
    def test_any_unknown_action_excludes_the_whole_player(self):
        record=self.record();record['decisions'][1]['action']['label_counts']={'1052':1}
        result,exclusion=exporter.encode_record(record,self.catalogue,'hash',self.index,0)
        self.assertIsNone(result);self.assertEqual(exclusion['reasons'],['complete_action_outside_frozen_catalogue'])
        self.assertEqual(len(exclusion['unsupported_node_ids']),1)
    def test_gap_and_terminal_timestamp_failure_cannot_be_stitched(self):
        record=self.record();record['whole_player_game_eligible']=False;record['ineligibility_reasons']=['gap']
        self.assertIsNone(exporter.encode_record(record,self.catalogue,'hash',self.index,0)[0])
        record=self.record();record['terminal']['timestamp_ms']=record['decisions'][-1]['timestamp_ms']
        self.assertIsNone(exporter.encode_record(record,self.catalogue,'hash',self.index,0)[0])
    def test_exact_fit_scope_excludes_calibration_and_denied_games(self):
        fits=[{'match_id':f'F_{i}','role':'fit','source_role':'train','patch':'16.17','game_creation_ms':1000+i} for i in range(4002)]
        base={'games':fits+[{'match_id':'CAL_1','role':'calibration'}]}
        manifest={'games':[{'match_id':row['match_id'],'role':'train','historical_validation':False,'patch':row['patch'],
                            'game_creation_ms':row['game_creation_ms']} for row in fits]}
        self.assertEqual(len(exporter.fit_scope(manifest,base,set())),4002)
        with self.assertRaisesRegex(ValueError,'reserved'):exporter.fit_scope(manifest,base,{'F_2'})
        altered=deepcopy(manifest);altered['games'][0]['role']='calibration'
        with self.assertRaisesRegex(ValueError,'role/date'):exporter.fit_scope(altered,base,set())
    def test_gzip_settings_are_deterministic(self):
        def encode():
            output=io.BytesIO()
            with gzip.GzipFile(filename='',mode='wb',fileobj=output,compresslevel=1,mtime=0) as stream:stream.write(b'row1\nrow2\n')
            return output.getvalue()
        first,second=encode(),encode()
        self.assertEqual(first,second);self.assertEqual(gzip.decompress(first),b'row1\nrow2\n')


if __name__=='__main__':unittest.main()
