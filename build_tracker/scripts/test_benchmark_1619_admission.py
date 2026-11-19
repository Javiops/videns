"""Holdouts, temporal boundaries and source evidence remain binding under new admission."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256, write_json
from app.dataset_generations import validate_1618_admission, validate_1618_split
from prepare_benchmark_1619 import freeze_split


def source_fixture():
    original = {'assignment':{'T':'train','V':'val','X':'test'},'game_creation_ms':{'T':100,'V':200,'X':300}}
    source = {'matches':[
        {'match_id':'T','game_creation':100,'prior_assignments':{'old':'train'}},
        {'match_id':'V','game_creation':200,'prior_assignments':{'old':'val'}},
        {'match_id':'N','game_creation':90,'prior_assignments':{}},
        {'match_id':'F','game_creation':201,'prior_assignments':{}}],
        'snapshot_utc':'2026-09-12T11:03:01Z','reserved_test_ids':['X'],
        'split_metadata_sha256':{},'training_or_export_authorized':False,'test_payload_reads':False}
    return source,original


def admission_fixture(directory):
    source,old_plan = source_fixture()
    source['matches'] = source['matches'][:2]
    assignment,_,metadata = freeze_split(source,{'T','V'},old_plan)
    write_json(directory/'original_split_assignment.json',old_plan['assignment'])
    write_json(directory/'original_split_plan.json',old_plan)
    source['split_metadata_sha256'] = {'old':sha256(directory/'original_split_assignment.json')}
    write_json(directory/'original_generation_manifest.json',{'files':{
        'split_assignment.json':{'sha256':sha256(directory/'original_split_assignment.json')},
        'split_plan.json':{'sha256':sha256(directory/'original_split_plan.json')}}})
    write_json(directory/'original_static_review.json',{'patch':'16.19','ddragon_version':'16.19.1',
                                                       'timer_economics_and_physical_rules_compatible':True})
    source['static_review_sha256'] = sha256(directory/'original_static_review.json')
    write_json(directory/'source_plan.json',source)
    mask = {'recovery_version':'footwear-viego-16.19-v1','timing_guard_ms':1000,'test_payload_reads':False,
        'games':{mid:{'admitted_decision_ids':[{'event_index':0,'participant_id':1,'ts':1000}],
                     'boundary_exclusions':[],'admitted_rows':1,'boundary_excluded_rows':0} for mid in ('T','V')}}
    write_json(directory/'decision_admission.json',mask)
    manifest = {'patch':'16.19','recovery_version':'footwear-viego-16.19-v1','db_sha256':'db',
        'source_plan_sha256':sha256(directory/'source_plan.json'),'games':2,
        'reconstruction_version':'decision-start-footwear-16.19-v1','inventory_version':'footwear-timer-inventory-16.19-v1',
        'slot_state_version':'role-slots-v1','static_data_sha256':'static',
        'footwear_runtime':{'patch':'16.19','runtime_version':'reconstructible-16.19-v1'},
        'files':{'static_review/static_review.json':{'sha256':source['static_review_sha256']}},
        'decision_admission_sha256':sha256(directory/'decision_admission.json')}
    write_json(directory/'corpus_manifest.json',manifest)
    (directory/'audit_code.py').write_text('# independent audit',encoding='utf-8')
    (directory/'qualification_code.py').write_text('# selected inventory checks',encoding='utf-8')
    audit = {'accepted_match_ids':['T','V'],'passed':True,'hard_failures':{},'quarantined_match_ids':{},
        'db_sha256':'db','corpus_manifest_sha256':sha256(directory/'corpus_manifest.json'),
        'audit_code_sha256':sha256(directory/'audit_code.py'),
        'counts':{'checks:reconstructible_whole_game_source_checks':2,
                  'checks:reconstructible_exact_admission_mask':2,'checks:reconstructible_causal_footwear_state':20}}
    write_json(directory/'semantic_audit.json',audit)
    original_q = {'status':'qualified_reconstruction_snapshot','training_clearance':False,
        'export_created':False,'test_payload_reads':False,'db_sha256':'db',
        'accepted_match_ids':['T','V'],'accepted_games':2,
        'semantic_audit_sha256':sha256(directory/'semantic_audit.json'),
        'corpus_manifest_sha256':sha256(directory/'corpus_manifest.json'),
        'decision_admission_sha256':sha256(directory/'decision_admission.json')}
    write_json(directory/'original_qualification.json',original_q)
    write_json(directory/'original_completion.json',{'state':'complete','completed_stages':['build','audit'],
        'qualification_sha256':sha256(directory/'original_qualification.json'),
        'protected_unchanged':True,'test_payload_reads':False})
    write_json(directory/'original_protocol.json',{'source_plan_sha256':sha256(directory/'source_plan.json')})
    write_json(directory/'preserved_split_assignment.json',assignment)
    write_json(directory/'split_metadata.json',metadata)
    admission = {'purpose':'patch_development_1619','patch':'16.19',
        'authorization':'owner job 2 (2026-09-29): qualify 16.19, fix a temporal cohort, measure frozen M2 and R1; no training','source_scope_unchanged':True,
        'final_test_payload_opened':False,
        'files':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in directory.iterdir()}}
    write_json(directory/'benchmark_admission.json',admission)
    q = {'benchmark_purpose':'patch_development_1619','status':'qualified_subset',
        'purpose':'isolated_experiment_only','training_clearance':'authorized benchmark IDs only',
        'corpus':str(directory),'semantic_audit':str(directory/'semantic_audit.json'),
        'promotion_allowed':False,'production_hold_cleared':False,'db_sha256':'db',
        'benchmark_admission_sha256':sha256(directory/'benchmark_admission.json'),
        'accepted_match_ids':['T','V'],'quality_accepted_games':2,'quarantined_match_ids':{},
        'inventory_constraint_states':['board_before','shop_before','shop_after'],'accepted_max_regular_slots':6,
        'source_qualification_sha256':sha256(directory/'original_qualification.json'),
        'semantic_audit_sha256':sha256(directory/'semantic_audit.json'),
        'corpus_manifest_sha256':sha256(directory/'corpus_manifest.json'),
        'qualification_code_sha256':sha256(directory/'qualification_code.py'),
        'decision_admission_sha256':sha256(directory/'decision_admission.json')}
    write_json(directory/'qualification.json',q)
    return q


class SplitTests(unittest.TestCase):
    def test_newer_sources_are_deferred_and_old_membership_is_preserved(self):
        source,original = source_fixture()
        assignment,excluded,metadata = freeze_split(source,{'T','V','N','F'},original)
        self.assertEqual(assignment,{'N':'train','T':'train','V':'val'})
        self.assertEqual(excluded,{'F':['after_original_validation_window']})
        self.assertFalse(metadata['final_test_payload_opened'])

    def test_prior_validation_cannot_become_training_even_before_cutoff(self):
        source,original = source_fixture()
        source['matches'][2]['prior_assignments'] = {'another_old_benchmark':'val'}
        assignment,excluded,_ = freeze_split(source,{'T','V','N'},original)
        self.assertNotIn('N',assignment)
        self.assertEqual(excluded['N'],['prior_validation_reserved_from_training'])

    def test_reserved_test_is_rejected_before_any_payload_api(self):
        source,original = source_fixture()
        source['matches'].append({'match_id':'X','game_creation':300,'prior_assignments':{'old':'test'}})
        with self.assertRaisesRegex(ValueError,'Reserved test'):
            freeze_split(source,{'T','V','X'},original)

    def test_timestamp_ties_cannot_cross_splits(self):
        source,original = source_fixture()
        original['game_creation_ms']['V'] = 100
        with self.assertRaises(ValueError):
            freeze_split(source,{'T','V'},original)

    def test_newer_training_cannot_hide_behind_disjoint_ids(self):
        source,original = source_fixture()
        assignment,_,metadata = freeze_split(source,{'T','V'},original)
        assignment['F'] = 'train'
        with self.assertRaisesRegex(ValueError,'temporal boundary'):
            validate_1618_split(assignment,metadata,source)


class AdmissionTests(unittest.TestCase):
    def test_new_authorization_preserves_original_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            admission_fixture(directory)
            verified = validate_1618_admission(directory)
            self.assertEqual(verified['assignment'],{'T':'train','V':'val'})
            self.assertFalse(json.loads((directory/'original_qualification.json').read_text())['training_clearance'])

    def test_changed_audit_mask_static_or_source_record_is_rejected(self):
        for name in ('semantic_audit.json','decision_admission.json','original_static_review.json',
                     'source_plan.json','original_qualification.json','qualification_code.py'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                admission_fixture(directory)
                with (directory/name).open('a',encoding='utf-8') as target:
                    target.write(' ')
                with self.assertRaisesRegex(ValueError,'evidence changed'):
                    validate_1618_admission(directory)

    def test_qualification_cannot_expand_into_unadmitted_game(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            q = admission_fixture(directory)
            q['accepted_match_ids'].append('X')
            q['quality_accepted_games'] += 1
            with self.assertRaisesRegex(ValueError,'selected qualification'):
                validate_1618_admission(directory,q)

    def test_scoped_generation_publication_and_verification(self):
        from app.dataset_generations import verify_generation
        import publish_patch_dataset as publisher
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source,coverage_dir,datasets = root/'source',root/'coverage',root/'datasets'
            source.mkdir();coverage_dir.mkdir();datasets.mkdir()
            q = admission_fixture(source)
            manifest = json.loads((source/'corpus_manifest.json').read_text())
            metadata = json.loads((source/'split_metadata.json').read_text())
            import shutil
            for name in ('preserved_split_assignment.json','split_metadata.json'):
                shutil.copyfile(source/name,coverage_dir/name)
            plan = {'purpose':'patch_development_1619','patch':'16.19','benchmark_id':'fixture','eval_version':6,
                'qualification_sha256':sha256(source/'qualification.json'),'row_builder_sha256':'builder',
                'assignment':{'T':'train','V':'val'},'game_creation_ms':{'T':100,'V':200},
                'game_counts':{'train':1,'val':1,'test':0},'row_counts':{'train':1,'val':1,'test':0},
                'policy':'fixture preserves source temporal windows',
                'decision_admission_sha256':sha256(source/'decision_admission.json'),
                'preserved_split_assignment_sha256':sha256(source/'preserved_split_assignment.json'),
                'split_metadata_sha256':sha256(source/'split_metadata.json'),'split_metadata':metadata}
            write_json(coverage_dir/'split_plan.json',plan)
            coverage = {'status':'coverage_complete','smoke_ready':True,'purpose':'patch_development_1619',
                'patch':'16.19','qualification':str(source/'qualification.json'),
                'qualification_sha256':sha256(source/'qualification.json'),
                'split_plan_sha256':sha256(coverage_dir/'split_plan.json'),'row_builder_sha256':'builder',
                'target_encoding':'item-counts-v1','max_modeled_inventory_items':2,'max_basket_items':2,
                'max_copies_of_one_item':2,'decision_admission_sha256':sha256(source/'decision_admission.json')}
            write_json(coverage_dir/'coverage.json',coverage)
            row = {'slot_state_version':'role-slots-v1','bot_quest_complete':False,'unmodeled_regular_slots':0,
                'footwear_state_version':'footwear-timer-slots-v1','magical_footwear_selected':True,
                'footwear_granted':False,'others':[],
                'source_target_id':3006,'source_target_version':'full-source-completions-v1',
                'gold_est_version':'prequential-v3','inventory_version':'footwear-timer-inventory-16.19-v1',
                'label_source_version':'purchase-net-v1','label_counts':{'1036':2},'target_encoding':'item-counts-v1'}
            context = SimpleNamespace(qualification=q,manifest=manifest,path=source,
                                      dragon=SimpleNamespace(version='16.19.1',signature='static'))
            def games(_context):
                for mid,stamp in (('T',100),('V',200)):
                    yield {'match_id':mid,'game_creation':stamp},[{**row,'match_id':mid}],None
            with patch.object(publisher,'load_qualified',return_value=context), patch.object(publisher,'iter_games',games), \
                 patch.object(publisher,'row_builder_digest',return_value='builder'):
                generation = publisher.publish(coverage_dir,datasets,'scoped')
            result = verify_generation(generation)
            self.assertEqual(result['purpose'],'patch_development_1619')
            self.assertEqual((generation/'visits_test.jsonl').stat().st_size,0)
            self.assertEqual(json.loads((generation/'visits_train.jsonl').read_text())['label_counts'],{'1036':2})
            original_hash = sha256(source/'original_qualification.json')
            self.assertEqual(original_hash,sha256(generation/'original_qualification.json'))

    def test_old_false_clearance_cannot_be_used_without_new_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            q = admission_fixture(directory)
            q['training_clearance'] = False
            with self.assertRaisesRegex(ValueError,'separately authorized'):
                validate_1618_admission(directory,q)


if __name__ == '__main__':
    unittest.main()
