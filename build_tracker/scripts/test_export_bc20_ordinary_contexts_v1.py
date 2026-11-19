"""Synthetic ordinary sampler/target-boundary tests; no production payloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_bc20_ordinary_contexts_v1 as exporter
from test_export_winprob_trajectories_v1 import example


def mask(rows):return {'admitted_decision_ids':rows,'admitted_rows':len(rows),'boundary_exclusions':[],'boundary_excluded_rows':0}


class OrdinarySelectionTests(unittest.TestCase):
    def test_focal_uses_exact_protocol_first_eight_bytes(self):
        mid='EUW1_123456789'
        digest=hashlib.sha256(b'bc20-ordinary-v1/focal/EUW1_123456789').digest()
        expected=1+int.from_bytes(digest[:8],'big')%10
        self.assertEqual(exporter.focal_player(mid),expected)
        self.assertNotEqual(expected,1+int.from_bytes(digest,'big')%10)
    def test_no_alternate_player_when_focal_has_no_decision(self):
        mid='EUW1_123456789';pid=exporter.focal_player(mid)
        other=pid%10+1
        result=exporter.choose_decision(mid,mask([{'participant_id':other,'event_index':1,'ts':999}]))
        self.assertEqual(result['status'],'no_admitted_decision');self.assertEqual(result['pid'],pid)
        self.assertIsNone(result['selected_decision']);self.assertEqual(result['sample_id'],f'{mid}|{pid}|none')
    def test_minimum_hash_all_timestamps_not_first_or_target_based(self):
        mid='EUW1_22';pid=exporter.focal_player(mid)
        rows=[{'participant_id':pid,'event_index':index,'ts':timestamp} for index,timestamp in ((9,0),(11,30000),(18,3600000))]
        expected=min(rows,key=lambda row:(hashlib.sha256(f'bc20-ordinary-v1/visit/{mid}|{pid}|{row["event_index"]}|{row["ts"]}'.encode()).digest(),row['ts'],row['event_index']))
        first=exporter.choose_decision(mid,mask(rows));second=exporter.choose_decision(mid,mask(list(reversed(rows))))
        self.assertEqual(first,second);self.assertEqual(first['selected_decision'],{'event_index':expected['event_index'],'pid':pid,'ts':expected['ts']})
        self.assertEqual(first['admitted_count'],3)
    def test_duplicate_or_boundary_overlap_fails(self):
        mid='EUW1_22';pid=exporter.focal_player(mid);row={'participant_id':pid,'event_index':1,'ts':1}
        with self.assertRaises(ValueError):exporter.choose_decision(mid,mask([row,row]))
        value=mask([row]);value.update(boundary_exclusions=[row],boundary_excluded_rows=1)
        with self.assertRaises(ValueError):exporter.choose_decision(mid,value)
    def test_target_kinds_preserve_wards_and_count_support_separately(self):
        full={exporter.producer.canonical([]):0,exporter.producer.canonical([[2055,1]]):2,
              exporter.producer.canonical([[1036,2]]):8}
        groups={exporter.producer.canonical([]):0,exporter.producer.canonical([[1036,2]]):7}
        save=exporter.target_record({'label_counts':{'999999':1}},full,groups)
        ward=exporter.target_record({'label_counts':{'2055':1}},full,groups)
        mixed=exporter.target_record({'label_counts':{'1036':2,'2055':2}},full,groups)
        unknown=exporter.target_record({'label_counts':{'999001':1}},full,groups)
        self.assertEqual(save['kind'],'observed_no_buy_death');self.assertEqual(save['full_action_id'],0)
        self.assertEqual(ward['kind'],'ward_only_purchase');self.assertEqual(ward['full_action_id'],2)
        self.assertEqual(save['main_group_id'],ward['main_group_id'])
        self.assertEqual(mixed['kind'],'mixed_purchase_with_wards');self.assertIsNone(mixed['full_action_id'])
        self.assertEqual(mixed['main_group_id'],7);self.assertEqual(mixed['main_label_counts'],{'1036':2})
        self.assertIsNone(unknown['main_group_id']);self.assertEqual(unknown['label_counts'],{'999001':1})

    def test_full_metadata_selection_never_opens_sql_raw_catalogue_or_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protocol=root/'protocol.md';protocol.write_text(exporter.FOCAL_PREFIX+' '+exporter.VISIT_PREFIX)
            admission=root/'admission.json';admission.write_text('{}')
            ids={f'GAME_{i}' for i in range(859)}
            manifest={'games':[{'match_id':mid,'role':'development','patch':'16.19'} for mid in sorted(ids)]}
            masks={mid:mask([{'participant_id':exporter.focal_player(mid),'event_index':1,'ts':i}]) for i,mid in enumerate(sorted(ids))}
            source={'approved':ids,'masks':masks,'bindings':{},'corpus':{'footwear_runtime':{'runtime_version':'synthetic'}}}
            binds={str(admission.resolve()):exporter.producer.file_hash(admission)}
            with patch.object(exporter.producer,'validate_admission',return_value=(manifest,set(),set(),set(),binds)), \
                 patch.object(exporter.native,'native_metadata',return_value=source),patch.object(exporter,'code_bindings',return_value={}), \
                 patch.object(exporter.sqlite3,'connect',side_effect=AssertionError('SQL forbidden in selection')), \
                 patch.object(exporter.producer,'read_raw_pair',side_effect=AssertionError('raw forbidden in selection')), \
                 patch.object(exporter,'target_maps',side_effect=AssertionError('catalogue forbidden in selection')):
                path,result=exporter.export_selection(admission,protocol,root/'selection')
            self.assertEqual(len(result['entries']),859);self.assertEqual(result['decision_bearing_games'],859)
            self.assertFalse(result['raw_payload_reads']);self.assertFalse(result['sqlite_opened'])
            self.assertEqual(path.with_suffix('.sha256').read_text().split(),[exporter.producer.file_hash(path),'selection.json'])

    def selected_fixture(self):
        event={'type':'shop','event_index':1,'participant_id':1,'puuid':'p1','team_id':100,'ts':10000,'ts_end':12000,
               'training_eligible':True,'source_quality_issues':[],'label_bought':[{'item_id':1036},{'item_id':1036},{'item_id':2055}],
               'gold_est':500,'gold_est_version':'prequential-v3','gold_gap':{'frame_ts':0}}
        entry={'match_id':'GAME_1','pid':1,'selected_decision':{'event_index':1,'pid':1,'ts':10000}}
        players={1:{'puuid':'p1','win':True}}
        def projection(e):
            row=example(e)
            row['label_ids']=[999999] if e.get('is_save') else [item['item_id'] for item in e['label_bought']]
            return row
        def attach(rows,frames,gold):
            assert all(value[0]<10000 for value in frames[1]+gold[1])
            for row in rows:row.update(gold_est=500,gold_est_version='prequential-v3',gold_gap={'frame_ts':0})
        debug={'frame_series':{1:[(0,500,500),(60000,900,1200)]},'gold_events':{1:[(10000,350,'spend')]}}
        full={exporter.producer.canonical([]):0,exporter.producer.canonical([[1036,2],[2055,1]]):1}
        groups={exporter.producer.canonical([]):0,exporter.producer.canonical([[1036,2]]):1}
        return entry,[event],players,projection,attach,debug,full,groups

    def test_selected_context_is_target_blind_and_target_is_separate(self):
        context,target,error=exporter.project_selected(*self.selected_fixture())
        self.assertIsNone(error);self.assertEqual(target['kind'],'mixed_purchase_with_wards')
        self.assertNotIn('label_counts',context);self.assertNotIn('target',context)
        self.assertNotIn('role',context['prestate']);self.assertIn('role',context['source_projection'])
        self.assertTrue(context['checks']['target_future_projection_mutation_equal'])

    def test_selected_identity_cannot_change_or_take_an_alternate(self):
        args=list(self.selected_fixture());args[1][0]['event_index']=2
        with self.assertRaisesRegex(ValueError,'identity did not reproduce'):exporter.project_selected(*args)

    def test_future_sensitive_projection_invalidates_instead_of_dropping_row(self):
        args=list(self.selected_fixture());original=args[3]
        def leaky(event):
            row=original(event);row['gold_est']=event.get('gold',500);return row
        args[3]=leaky
        with self.assertRaisesRegex(ValueError,'target/future'):exporter.project_selected(*args)


if __name__=='__main__':unittest.main()
