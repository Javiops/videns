"""Synthetic consumer publication, denominator, identity and streaming tests."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from winprob import admission
from winprob import iql_dataset_contract as contract


def write_json(path,value,bound=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')
    if bound:path.with_suffix('.sha256').write_text(contract.file_hash(path)+'  '+path.name+'\n',encoding='ascii')


class DatasetFixture:
    def __init__(self,root):
        self.root=root;self.data=root/'data';self.data.mkdir()
        self.fit_rows=[]
        for patch_name,n in contract.EXPECTED_PATCH_COUNTS.items():
            for i in range(n):
                self.fit_rows.append({'match_id':f'F{patch_name.replace(".","")}_{i}','role':'fit','source_role':'train',
                                     'patch':patch_name,'game_creation_ms':100000+len(self.fit_rows)*1000})
        self.ids={row['match_id'] for row in self.fit_rows}
        split=self.data/'split_assignment.json'
        write_json(split,{mid:'train' for mid in self.ids})
        self.admission_path=root/'admission.json'
        self.admission={'schema':'winprob-admission-v1','state':'temporal_manifest_ready','source_data_root':str(self.data),
                        'split_metadata_sources':[{'path':split.name,'sha256':contract.file_hash(split)}],
                        'external_reservation_sources':[],'prior_manifest':None,
                        'games':[{**row,'role':'train','historical_validation':False} for row in self.fit_rows]}
        write_json(self.admission_path,self.admission,bound=True)
        self.source_path=root/'original'/'source_manifest.json'
        write_json(self.source_path,{'games':self.fit_rows,'test_payload_reads':False,'confirmatory_payload_reads':False,
                                    'historical_validation_payload_reads':False})
        self.code=root/'synthetic_producer.py';self.code.write_text('# synthetic\n',encoding='utf-8')
        self.catalogue={'full_actions':[[],[[1036,1]],[[2055,1]]],
                        'full_action_kinds':['observed_no_buy_death','retained_net_purchase','retained_net_purchase'],
                        'full_to_group':[0,1,0],'group_main':[[],[[1036,1]]]}
        self.lookup={('observed_no_buy_death',()):0,('retained_net_purchase',((1036,1),)):1,
                     ('retained_net_purchase',((2055,1),)):2}
        self.catalogue_path=root/'catalogue.json';write_json(self.catalogue_path,self.catalogue,bound=True)
        self.catalogue_hash=contract.file_hash(self.catalogue_path)

    def record(self,mid,pid,patch_name):
        entry=next(row for row in self.fit_rows if row['match_id']==mid)
        node_id=f'{mid}:{pid}:1';terminal_id=f'{mid}:{pid}:terminal'
        return {'schema':'winprob-retained-basket-trajectories-v1','match_id':mid,'participant_id':pid,
                'trajectory_id':f'{mid}:{pid}','patch':patch_name,'role':'fit','consumer_role':'fit','team_id':100,
                'source_role':'qualified_history_fit' if patch_name=='16.17' else 'v4_train',
                'game_creation_ms':entry['game_creation_ms'],'whole_player_game_eligible':True,'ineligibility_reasons':[],
                'full_action_catalogue_sha256':self.catalogue_hash,
                'source_quality':{'game_binding_index':0,'indexed_rows_sha256':'source-row-hash'},
                'decisions':[{'node_id':node_id,'kind':'decision','timestamp_ms':100,'visit_end_ms':110,'reward':0,
                              'done':False,'next_ref':terminal_id,'prestate':{'ts':100,'gold_est':500,'inventory':[1001]},
                              'action':{'encoding':'item-counts-v1','kind':'retained_net_purchase','label_counts':{'1036':1},
                                        'wards_preserved':True,'full_action_id':1,'main_group_id':1}}],
                'terminal':{'node_id':terminal_id,'kind':'terminal','timestamp_ms':200,'reward':1,'done':True,
                            'action':None,'prestate':None,'next_ref':None}}

    def shard(self,patch_name,full=False,expected_pids=(1,),written_pids=None,mutate=None,wrong_uncompressed=False):
        directory=self.root/('shard-'+patch_name);directory.mkdir()
        patch_ids=[row['match_id'] for row in self.fit_rows if row['patch']==patch_name]
        requested=patch_ids if full else patch_ids[:1];mid=requested[0]
        queried=[mid];qualification={missing:'native_qualification_exclusion' for missing in requested[1:]}
        excluded=[];remaining=[pid for pid in range(1,11) if pid not in expected_pids]
        for index,pid in enumerate(remaining):
            why=(['complete_action_outside_frozen_catalogue'] if index==0 else
                 ['source_chain_validation_failed'] if index==1 else ['unmodeled_or_excluded_decision_gap','overlapping_or_duplicate_decision_times'])
            excluded.append({'scope':'player_game','match_id':mid,'participant_id':pid,'trajectory_id':f'{mid}:{pid}',
                             'role':'fit','patch':patch_name,'reasons':why})
        excluded += [{'scope':'game','match_id':missing,'role':'fit','patch':patch_name,'reasons':[reason]}
                     for missing,reason in qualification.items()]
        (directory/'exclusions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in excluded),encoding='utf-8')
        records=[self.record(mid,pid,patch_name) for pid in (expected_pids if written_pids is None else written_pids)]
        if mutate is not None:records=mutate(records)
        raw=''.join(json.dumps(row,sort_keys=True)+'\n' for row in records).encode()
        (directory/'trajectories.jsonl.gz').write_bytes(gzip.compress(raw,compresslevel=1,mtime=0))
        meta={'schema':contract.SOURCE_SCHEMA,'state':'complete','patch':patch_name,'role':'fit','full_mode':full,
              'rows_schema':'winprob-retained-basket-trajectories-v1','rows_file':'trajectories.jsonl.gz','rows_compression':'gzip',
              'gzip_compresslevel':1,'gzip_mtime':0,'original_fit_games':4002,'original_fit_game_ids':sorted(self.ids),
              'original_fit_ids_sha256':admission.id_digest(self.ids),'original_fit_patch_counts':contract.EXPECTED_PATCH_COUNTS,
              'requested_game_ids':requested,'requested_patch_fit_games':len(patch_ids),'qualified_game_ids':queried,
              'queried_game_ids':queried,'qualification_exclusions':qualification,
              'catalogue_sha256':self.catalogue_hash,'full_action_catalogue_sha256':self.catalogue_hash,
              'source_code_sha256':{str(self.code):contract.file_hash(self.code)},
              'input_metadata_sha256':{str(self.admission_path):contract.file_hash(self.admission_path),
                                       str(self.source_path):contract.file_hash(self.source_path)},
              'eligible_player_games':len(expected_pids),'trajectory_count':len(expected_pids),'eligible_game_ids':[mid],'eligible_games':1,
              'counts':{'considered_player_games':10,'eligible_player_games':len(expected_pids),'excluded_player_games':10-len(expected_pids),
                        'terminal_transitions':len(expected_pids),'decision_nodes':len(expected_pids),'qualification_excluded_games':len(qualification)},
              'exclusion_reason_counts':dict(Counter(reason for row in excluded for reason in row['reasons'])),
              'game_bindings':[{'match_id':mid,'indexed_rows_sha256':'source-row-hash','raw_replay_matches_indexed_rows':True,
                                'admitted_mask_reproduced':True}],
              'uncompressed_rows_sha256':'bad' if wrong_uncompressed else hashlib.sha256(raw).hexdigest(),
              **dict.fromkeys(contract.FALSE_FLAGS,False),**dict.fromkeys(contract.TRUE_FLAGS,True)}
        selection_keys=('schema','full_mode','patch','role','original_fit_games','original_fit_ids_sha256','original_fit_game_ids',
                        'original_fit_patch_counts','requested_game_ids','requested_patch_fit_games','qualified_game_ids',
                        'qualification_exclusions','input_metadata_sha256','source_code_sha256','catalogue_sha256')
        write_json(directory/'selection.json',{key:meta[key] for key in selection_keys})
        meta['files']={name:contract.file_hash(directory/name) for name in contract.SOURCE_FILES}
        meta['rows_sha256']=meta['files']['trajectories.jsonl.gz']
        write_json(directory/'manifest.json',meta,bound=True)
        return directory


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.fixture=DatasetFixture(Path(self.tmp.name))
        self.mock=patch.object(contract.full_action_catalogue,'load_catalogue',return_value=(self.fixture.catalogue,self.fixture.lookup))
        self.mock.start();self.addCleanup(self.mock.stop)
    def validate(self,dirs,smoke=True):
        return contract.validate_sources(dirs,self.fixture.admission_path,self.fixture.catalogue_path,smoke=smoke)
    def test_smoke_denominator_distinguishes_catalogue_and_source_failures(self):
        directory=self.fixture.shard('16.17');bundle=self.validate([directory])
        mid=bundle['sources'][0]['requested_fit_ids'][0]
        self.assertEqual(bundle['source_complete_player_games'][mid],2)
        self.assertEqual(bundle['source_flag_complete_player_games'][mid],3)
        self.assertEqual(bundle['exported_player_games'][mid],1)
        self.assertEqual(len(bundle['fit_ids']),4002);self.assertFalse(bundle['training_scope_ready'])
        rows=list(contract.iter_episodes(bundle));self.assertTrue(bundle['verified_complete'])
        self.assertEqual(rows[0][2][0]['reward'],1);self.assertTrue(rows[0][2][0]['terminated'])
        self.assertEqual(rows[0][2][0]['full_action_id'],1)
    def test_full_sources_require_both_exact_disjoint_patch_requests(self):
        one=self.fixture.shard('16.17',full=True);two=self.fixture.shard('16.18',full=True)
        bundle=self.validate([one,two],smoke=False)
        self.assertEqual(len(bundle['requested_fit_ids']),4002)
        self.assertEqual(len(list(contract.iter_episodes(bundle))),2)
        with self.assertRaisesRegex(ValueError,'both disjoint'):self.validate([one],smoke=False)
        with self.assertRaisesRegex(ValueError,'at most 10'):self.validate([one],smoke=True)
    def test_partial_iteration_never_certifies_and_second_pass_resets_flag(self):
        directory=self.fixture.shard('16.17');bundle=self.validate([directory])
        iterator=contract.iter_episodes(bundle);next(iterator);iterator.close()
        self.assertFalse(bundle['verified_complete'])
        list(contract.iter_episodes(bundle));self.assertTrue(bundle['verified_complete'])
        iterator=contract.iter_episodes(bundle);self.assertFalse(bundle['verified_complete'])
        list(iterator);self.assertTrue(bundle['verified_complete']);self.assertEqual(bundle['completed_passes'],2)
    def test_independent_full_action_id_and_kind_verification(self):
        def corrupt(records):records[0]['decisions'][0]['action']['full_action_id']=2;return records
        directory=self.fixture.shard('16.17',mutate=corrupt);bundle=self.validate([directory])
        with self.assertRaisesRegex(ValueError,'canonical multiset'):list(contract.iter_episodes(bundle))
        self.assertFalse(bundle['verified_complete'])
    def test_duplicate_episode_rejected(self):
        directory=self.fixture.shard('16.17',mutate=lambda records:records+deepcopy(records));bundle=self.validate([directory])
        with self.assertRaisesRegex(ValueError,'Duplicate/cross-source'):list(contract.iter_episodes(bundle))
        self.assertFalse(bundle['verified_complete'])
    def test_uncompressed_hash_and_completion_counts_verified_after_yields(self):
        directory=self.fixture.shard('16.17',wrong_uncompressed=True);bundle=self.validate([directory])
        iterator=contract.iter_episodes(bundle);next(iterator)
        with self.assertRaisesRegex(ValueError,'Uncompressed'):next(iterator)
        self.assertFalse(bundle['verified_complete'])
    def test_missing_expected_player_is_not_silently_skipped(self):
        directory=self.fixture.shard('16.17',expected_pids=(1,2),written_pids=(1,));bundle=self.validate([directory])
        with self.assertRaisesRegex(ValueError,'counts did not reproduce'):list(contract.iter_episodes(bundle))
        self.assertFalse(bundle['verified_complete'])
    def test_new_reservation_rejected_before_compressed_payload_hashing(self):
        directory=self.fixture.shard('16.17');mid='F1617_0'
        real_hash=contract.file_hash;opened=[]
        def observe(path):opened.append(Path(path).name);return real_hash(path)
        with patch.object(contract,'current_reservations',return_value={mid}),patch.object(contract,'file_hash',side_effect=observe):
            with self.assertRaisesRegex(ValueError,'reserved game'):self.validate([directory])
        self.assertNotIn('trajectories.jsonl.gz',opened)
    def test_changed_files_and_unfinished_shards_fail_before_streaming(self):
        directory=self.fixture.shard('16.17');bundle=self.validate([directory])
        self.fixture.code.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'Bound source/code changed'):list(contract.iter_episodes(bundle))
        self.assertFalse(bundle['verified_complete'])
        unfinished=Path(self.tmp.name)/'unfinished';unfinished.mkdir()
        with self.assertRaisesRegex(ValueError,'unfinished'):self.validate([directory,unfinished])


if __name__=='__main__':unittest.main()
