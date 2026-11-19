"""Synthetic trajectory-export regressions; never reads production data."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import export_winprob_trajectories_v1 as producer


def example(event):
    other={'champion':'Other','champion_id':2,'team_id':200,'role':'TOP','gold':500,'level':1,
           'items':[],'bot_quest_complete':False,'slot_state_version':'role-slots-v1',
           'footwear_state_version':'footwear-timer-slots-v1','magical_footwear_selected':False,'footwear_granted':False}
    row={key:0 for key in producer.STATE_FIELDS}
    row.update(ts=event['ts'],champion='Self',champion_id=1,role='TOP',team_id=event['team_id'],gold=500,
               gold_left=500,gold_est=500.,gold_est_version='prequential-v3',total_gold=500,level=1,inventory=[],
               bot_quest_complete=False,slot_state_version='role-slots-v1',footwear_state_version='footwear-timer-slots-v1',
               magical_footwear_selected=False,footwear_granted=False,ally_obj={},enemy_obj={},others=[dict(other) for _ in range(9)],
               label_ids=[1036,1036,2055],source_target_id=999,decision='LABEL_DERIVED',save_kind='build_spend',win=True)
    return row


def fixture():
    participants=[{'participant_id':pid,'puuid':f'p{pid}','team_id':100 if pid<=5 else 200,
                   'win':pid<=5,'keystone_id':8005,'sub_style':8100,'summ1':4,'summ2':12} for pid in range(1,11)]
    game={'match_id':'EUW1_1','game_creation':1000,'game_duration':600,'game_version':'16.18.1',
          'patch':'16.18','queue_id':420,'win_team':100,'participants_json':json.dumps(participants),'source':'unified'}
    events=[{'type':'shop','event_index':index,'participant_id':1,'puuid':'p1','team_id':100,
             'ts':timestamp,'ts_end':timestamp+1000,'training_eligible':True,'source_quality_issues':[]}
            for index,timestamp in enumerate((10000,20000),1)]
    mask={'admitted_decision_ids':[{key:event[key] for key in producer.DECISION_FIELDS} for event in events],
          'boundary_exclusions':[],'admitted_rows':2,'boundary_excluded_rows':0}
    return game,events,mask


class TrajectoryExportTests(unittest.TestCase):
    def test_selection_denies_before_data_access_and_keeps_fit_only(self):
        manifest={'games':[{'match_id':f'EUW1_{n}','role':'train' if n<=4 else 'development','historical_validation':False} for n in range(1,6)]}
        split={f'EUW1_{n}':'train' for n in range(1,6)}
        selected,count=producer.select_ids(manifest,split,set(split),{'EUW1_2'},{'EUW1_3'},{'EUW1_4'},10)
        self.assertEqual(selected,['EUW1_1']);self.assertEqual(count,1)
        with self.assertRaises(ValueError): producer.select_ids(manifest,split,set(split),set(),set(),set(),11)

    def test_denied_id_never_reaches_sql(self):
        with closing(sqlite3.connect(':memory:')) as connection:
            seen=[];connection.set_trace_callback(seen.append)
            reader=producer.ApprovedGameReader(connection,{'EUW1_1'},{'EUW1_2'})
            with self.assertRaises(PermissionError): reader.game('EUW1_2')
            self.assertEqual(seen,[])

    def test_indexed_reader_only_returns_requested_game(self):
        game,events,_=fixture()
        with closing(sqlite3.connect(':memory:')) as connection:
            connection.execute('CREATE TABLE games(match_id TEXT PRIMARY KEY,game_creation,game_duration,game_version,patch,queue_id,win_team,participants_json,source)')
            connection.execute('CREATE TABLE game_events(match_id,event_index,ts,ts_end,type,payload_json)')
            connection.execute('CREATE INDEX idx_events_match ON game_events(match_id,ts)')
            connection.execute('INSERT INTO games VALUES(?,?,?,?,?,?,?,?,?)',tuple(game.values()))
            for event in events:
                connection.execute('INSERT INTO game_events VALUES(?,?,?,?,?,?)',('EUW1_1',event['event_index'],event['ts'],event['ts_end'],'shop',json.dumps(event)))
            connection.execute('INSERT INTO game_events VALUES(?,?,?,?,?,?)',('EUW1_2',99,0,0,'shop','THIS MUST NOT BE PARSED'))
            result,rows=producer.ApprovedGameReader(connection,{'EUW1_1'},{'EUW1_2'}).game('EUW1_1')
            self.assertEqual(result,game);self.assertEqual(rows,events)

    def test_full_multiset_prestate_allowlist_and_terminal_reward_once(self):
        game,events,mask=fixture();joined=[]
        def projection(event): joined.append(event['perks']);return example(event)
        records=producer.trajectories_for_game(game,events,mask,projection,600001,{})
        row=records[0];self.assertTrue(row['whole_player_game_eligible'])
        self.assertEqual(len(records),10);self.assertEqual(len(joined),2)
        self.assertEqual(row['decisions'][0]['action']['label_counts'],{'1036':2,'2055':1})
        self.assertFalse({'source_target_id','decision','save_kind','win'}&row['decisions'][0]['prestate'].keys())
        self.assertNotIn('role',row['decisions'][0]['prestate'])
        self.assertTrue(all('role' not in player for player in row['decisions'][0]['prestate']['others']))
        self.assertEqual(row['decisions'][0]['reference_compatibility']['source_projection']['role'],'TOP')
        self.assertEqual(row['decisions'][0]['next_ref'],row['decisions'][1]['node_id'])
        self.assertEqual(row['decisions'][-1]['next_ref'],row['terminal']['node_id'])
        self.assertEqual(row['terminal']['timestamp_ms'],600001)
        self.assertEqual(sum(node['reward'] for node in row['decisions'])+row['terminal']['reward'],1)
        self.assertFalse(records[1]['whole_player_game_eligible'])
        self.assertEqual(records[5]['terminal']['reward'],0)

    def test_inferred_role_and_future_label_changes_do_not_change_prestate(self):
        _,events,_=fixture()
        first=example(events[0]);second=example(events[0])
        second.update(role='JUNGLE',win=False,source_target_id=123456,decision='DIFFERENT_LABEL',save_kind='other')
        for player in second['others']:
            player.update(role='MIDDLE',win=False,source_target_id=54321)
        first_state,second_state=producer.project_state(first),producer.project_state(second)
        self.assertEqual(first_state,second_state)
        compatibility=producer.reference_compatibility(second,second_state)
        self.assertEqual(compatibility['source_projection']['role'],'JUNGLE')
        self.assertEqual(compatibility['source_projection']['others'][0]['role'],'MIDDLE')
        self.assertNotIn('win',compatibility['source_projection'])
        self.assertNotIn('source_target_id',compatibility['source_projection'])

    def test_boundary_gap_invalidates_whole_player_without_bridging(self):
        game,events,mask=fixture()
        excluded=mask['admitted_decision_ids'].pop(0)
        mask.update(admitted_rows=1,boundary_excluded_rows=1,boundary_exclusions=[{**excluded,'reason':'grant_boundary'}])
        row=producer.trajectories_for_game(game,events,mask,example,600001,{})[0]
        self.assertFalse(row['whole_player_game_eligible']);self.assertEqual(row['decisions'][0]['kind'],'gap')
        self.assertIsNone(row['decisions'][0]['prestate']);self.assertIsNone(row['decisions'][0]['action'])
        self.assertEqual(row['decisions'][0]['next_ref'],row['decisions'][1]['node_id'])

    def test_missing_mask_entry_or_uncertain_state_fails_closed(self):
        game,events,mask=fixture()
        bad={**mask,'admitted_rows':1,'admitted_decision_ids':mask['admitted_decision_ids'][:1]}
        with self.assertRaisesRegex(ValueError,'outside exact audited mask'):
            producer.trajectories_for_game(game,events,bad,example,600001,{})
        row=example(events[0]);row['gold_est']=None
        with self.assertRaisesRegex(ValueError,'Unknown numeric prestate'):
            producer.project_state(row)
        row=example(events[0]);row['ally_obj']={'future_win':1}
        with self.assertRaisesRegex(ValueError,'Unknown objective'):
            producer.project_state(row)

    def test_terminal_requires_actual_game_end_and_replay_must_match(self):
        timeline={'info':{'frames':[{'events':[{'type':'GAME_END','timestamp':600123}]}]}}
        self.assertEqual(producer.terminal_from_timeline(timeline),600123)
        with self.assertRaises(ValueError):producer.terminal_from_timeline({'info':{'frames':[]}})
        game,events,_=fixture();producer.verify_replay(game,events,dict(game),list(events))
        changed=[{**events[0],'ts':999},events[1]]
        with self.assertRaisesRegex(ValueError,'events differ'):
            producer.verify_replay(game,events,game,changed)

    def test_streaming_mask_reads_selected_metadata_and_detects_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mask.json'
            payload={'schema':1,'padding':'x'*70000,'games':{'EUW1_1':{'ids':[1,2]},'EUW1_2':{'ids':[3]}},'tail':True}
            path.write_text(json.dumps(payload))
            self.assertEqual(producer.selected_object_entries(path,'games',{'EUW1_2'}),{'EUW1_2':{'ids':[3]}})
            with self.assertRaisesRegex(ValueError,'missing from decision mask'):
                producer.selected_object_entries(path,'games',{'EUW1_3'})

    def test_overwrite_refused_before_admission_or_queries(self):
        class Args:pass
        with tempfile.TemporaryDirectory() as folder:
            args=Args();args.output=Path(folder)
            with self.assertRaises(FileExistsError):producer.export(args)


if __name__=='__main__':
    unittest.main()
