"""Development accepts only declared train/validation membership and a fixed bounded recipe."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
import publish_patch_dataset as publisher
import report_corpus_coverage as coverage
import train_prefix as trainer
from app.dataset_artifacts import sha256,write_json
from app.dataset_generations import verify_generation,validate_development_run_plan
from app.reconstruct import SLOT_STATE_VERSION,RECONSTRUCTION_VERSION


def refused(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('An invalid development scope was accepted')


with tempfile.TemporaryDirectory(dir=ROOT / 'data/experiments') as temp:
    root = Path(temp)
    try:
        source = root / 'source'
        source.mkdir()
        for name in ('audit_code.py','qualification_code.py'):
            (source / name).write_text('# fixture',encoding='utf-8')
        corpus = {'db_sha256':'fixture-db','patch':'16.17','reconstruction_version':RECONSTRUCTION_VERSION,
                  'slot_state_version':SLOT_STATE_VERSION}
        write_json(source / 'corpus_manifest.json',corpus)
        audit = {'passed':True,'hard_failures':{},'db_sha256':'fixture-db',
                 'corpus_manifest_sha256':sha256(source / 'corpus_manifest.json'),
                 'audit_code_sha256':sha256(source / 'audit_code.py')}
        write_json(source / 'semantic_audit.json',audit)
        assignment = {'A':'train','B':'val'}
        q = {'status':'qualified_subset','purpose':'isolated_experiment_only','promotion_allowed':False,
             'production_hold_cleared':False,'semantic_audit':str(source / 'semantic_audit.json'),
             'db_sha256':'fixture-db','semantic_audit_sha256':sha256(source / 'semantic_audit.json'),
             'corpus_manifest_sha256':sha256(source / 'corpus_manifest.json'),
             'qualification_code_sha256':sha256(source / 'qualification_code.py'),
             'accepted_match_ids':list(assignment),
             'inventory_constraint_states':['board_before','shop_before','shop_after']}
        write_json(source / 'qualification.json',q)
        write_json(source / 'preserved_split_assignment.json',assignment)
        assert coverage.preserve_assignment({'A':1,'B':2},source / 'preserved_split_assignment.json') == assignment
        refused(lambda:coverage.preserve_assignment({'C':1},source / 'preserved_split_assignment.json'))
        metadata = {'test_status':'unavailable_original_holdout_not_materialized','reserved_test_games':2955,
                    'reserved_test_ids_sha256':'a'*64,'validation_exposure':'Previously used for probe validation',
                    'assignment_policy':'Prior training remains training; previously probe validation remains validation'}
        write_json(source / 'split_metadata.json',metadata)
        builder = publisher.row_builder_digest()
        plan = {'assignment':assignment,'game_creation_ms':{'A':1,'B':2},
                'game_counts':{'train':1,'val':1,'test':0},'row_counts':{'train':1,'val':1,'test':0},
                'row_builder_sha256':builder,'qualification_sha256':sha256(source / 'qualification.json'),
                'purpose':'patch_development','patch':'16.17','eval_version':1,'benchmark_id':'fixture-development',
                'policy':'Preserve assignments','preserved_split_assignment_sha256':sha256(source / 'preserved_split_assignment.json'),
                'split_metadata':metadata,'split_metadata_sha256':sha256(source / 'split_metadata.json')}
        write_json(source / 'split_plan.json',plan)
        report = {'status':'coverage_complete','smoke_ready':True,'qualification':str(source / 'qualification.json'),
                  'purpose':'patch_development','patch':'16.17',
                  'qualification_sha256':sha256(source / 'qualification.json'),
                  'split_plan_sha256':sha256(source / 'split_plan.json'),'row_builder_sha256':builder,
                  'target_encoding':'item-counts-v1','max_modeled_inventory_items':8,
                  'max_basket_items':6,'max_copies_of_one_item':5}
        write_json(source / 'coverage.json',report)
        context = SimpleNamespace(path=source,qualification=q,manifest=corpus,
                                  dragon=SimpleNamespace(version='16.17.1',signature='fixture-static'))
        def rows(_context):
            for mid in assignment:
                yield {'match_id':mid,'game_creation':plan['game_creation_ms'][mid]},[{
                    'match_id':mid,'gold_est_version':'prequential-v3','inventory_version':'causal-quests-v1',
                    'label_source_version':'purchase-net-v1','label_counts':{'1036':1},
                    'slot_state_version':SLOT_STATE_VERSION,'bot_quest_complete':False,
                    'unmodeled_regular_slots':0,'others':[]}],None
        with patch.object(publisher,'load_qualified',return_value=context),patch.object(publisher,'iter_games',side_effect=rows):
            dataset = publisher.publish(source,root / 'datasets','development')
        original_open = Path.open
        def sealed(path,*args,**kwargs):
            if path.name == 'visits_test.jsonl':
                raise AssertionError('Final test payload opened')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',sealed):
            generation = verify_generation(dataset)
        assert generation['purpose'] == 'patch_development'
        split = json.loads((dataset / 'split_manifest.json').read_text())
        assert split['probe'] is True and split['rows_per_split']['test'] == 0
        assert split['slot_state_versions'] == {SLOT_STATE_VERSION:2}
        run = root / 'run'
        run.mkdir()
        run_plan = {'schema':1,'purpose':'patch_development','patch':'16.17','dataset':str(dataset),
                    'run_directory':str(run),'export_fingerprint':generation['export_fingerprint'],
                    'generation_manifest_sha256':sha256(dataset / 'generation_manifest.json'),
                    'question':'Does the candidate improve on the conditional baseline?',
                    'primary_metrics':['first_action_exact','action_family_match'],'populations':['all scored visits'],
                    'stopping_rule':'One fixed trunk and one graft; then one paired validation evaluation',
                    'resource_budget':{'max_wall_seconds':14400,'max_train_processes':1,'max_ram_gb':12,'minimum_free_disk_gb':0.001},
                    'final_test_evaluation':False,'promotion_allowed':False,'production_hold_cleared':False,
                    'code_sha256':{'scripts/train_prefix.py':sha256(ROOT / 'scripts/train_prefix.py')},
                    'evaluation':{'split':'val','games':0,'compare_conditional':True},
                    'evaluation_seed':16,'evaluation_bootstrap':400}
        env = {'PREFIX_OUT':'trunk.pt','PREFIX_RUN_PLAN':str(run / 'run_plan.json'),'PREFIX_STAGE':'trunk'}
        with patch.multiple(trainer,ML_DIR=dataset,RUN_DIR=run,EPOCHS=16,USE_GOLDX=True,USE_GOLDEST=True,USE_RUNES=True),patch.dict(trainer.os.environ,env):
            fixed = trainer.current_training_environment()
            stage_keys = ('PREFIX_OUT','PREFIX_EPOCHS','PREFIX_SAVEW','PREFIX_TARGETW','PREFIX_INIT_FROM','PREFIX_FREEZE')
            run_plan['fixed_environment'] = {k:v for k,v in fixed.items() if k not in stage_keys}
            run_plan['training_stages'] = {'trunk':{k:fixed[k] for k in stage_keys},
                'graft':{'PREFIX_OUT':'candidate.pt','PREFIX_EPOCHS':'6','PREFIX_SAVEW':'0.5','PREFIX_TARGETW':'0.25',
                         'PREFIX_INIT_FROM':'trunk.pt','PREFIX_FREEZE':'1'}}
            write_json(run / 'run_plan.json',run_plan)
            bound = trainer.validate_generation_training(generation)
            assert bound['stage'] == 'trunk' and bound['sha256'] == sha256(run / 'run_plan.json')
            trainer.validate_development_warm_start({'provenance':{'run_plan_sha256':bound['sha256'],'training_stage':'trunk'}},bound)
            for parent in ({},{'run_plan_sha256':'other-plan','training_stage':'trunk'},
                           {'run_plan_sha256':bound['sha256'],'training_stage':'graft'}):
                refused(lambda:trainer.validate_development_warm_start({'provenance':parent},bound))
            with patch.object(trainer,'EPOCHS',15):
                refused(lambda:trainer.validate_generation_training(generation))
            with patch.dict(trainer.os.environ,{'PREFIX_STAGE':'sweep'}):
                refused(lambda:trainer.validate_generation_training(generation))
            for changes in ({'patch':'16.18'},{'final_test_evaluation':True},{'promotion_allowed':True},
                            {'export_fingerprint':'wrong'},{'evaluation':{'split':'test','games':0,'compare_conditional':True}}):
                write_json(run / 'run_plan.json',{**run_plan,**changes})
                refused(lambda:validate_development_run_plan(dataset,run,generation,run / 'run_plan.json'))
            for changes in ({'evaluation_seed':None},{'evaluation_seed':True},
                            {'evaluation_bootstrap':0},{'evaluation_bootstrap':True}):
                write_json(run / 'run_plan.json',{**run_plan,**changes})
                refused(lambda:validate_development_run_plan(dataset,run,generation,run / 'run_plan.json'))
            write_json(run / 'run_plan.json',run_plan)
        with patch.object(trainer,'EPOCHS',2):
            refused(lambda:trainer.validate_generation_training({'purpose':'pipeline_smoke_only'}))
        with patch.object(trainer,'EPOCHS',1):
            assert trainer.validate_generation_training({'purpose':'pipeline_smoke_only'}) is None
        # Artifact metadata tampering is caught before the trainer reads examples.
        target = dataset / 'split_metadata.json'
        target.chmod(0o666)
        write_json(target,{**metadata,'reserved_test_games':0})
        refused(lambda:verify_generation(dataset))
        print('ok patch development: bound train/val export, sealed unavailable test, unchanged smoke guard, fixed recipe and plan tamper rejection')
    finally:
        for path in root.rglob('*'):
            if path.is_file():
                path.chmod(0o666)
