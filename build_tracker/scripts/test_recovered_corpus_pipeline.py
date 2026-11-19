"""Bounded two-real-game integration; no model or reserved-test payload reads.

Requires the local frozen recovery evidence. Synthetic mask/gate regressions
remain in test_recovered_decision_admission.py for source-free testing.
"""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256,write_json
from build_recovered_patch_corpus import read_selection,build


def refused(fn):
    try:
        fn()
    except (ValueError,KeyError,TypeError):
        return
    raise AssertionError('Invalid recovered audit/qualification was accepted')


base = ROOT/'data/acceptance'
required = base/'footwear_repair_20260911/recovery_summary.json'
if not required.exists():
    print('SKIP local recovered corpus integration: frozen recovery sources are unavailable')
    raise SystemExit(0)
plan,prior,summary,recovered,records = read_selection(base/'16.17_development_20260911/source_plan.json',
    base/'16.17_development_20260911/qualification.json',required)
old_record = next(r for r in records if r['match_id'] not in recovered and r['split']=='val')
new_record = next(r for r in records if r['match_id'] in recovered and r['split']=='train')
with tempfile.TemporaryDirectory(prefix='recovered-pipeline-smoke-',dir=ROOT/'data') as temporary:
    target = Path(temporary).resolve()
    assert target.is_relative_to(ROOT/'data')
    try:
        write_json(target/'split_assignment.json',{r['match_id']:r['split'] for r in (old_record,new_record)})
        write_json(target/'split_metadata.json',plan['split_metadata'])
        small_plan = {**plan,'matches':[old_record,new_record],
            'split_assignment_sha256':sha256(target/'split_assignment.json'),
            'split_metadata_sha256':sha256(target/'split_metadata.json')}
        write_json(target/'source_plan.json',small_plan)
        small_prior = {**prior,'accepted_match_ids':[old_record['match_id']],'quality_accepted_games':1}
        write_json(target/'prior_qualification.json',small_prior)
        small_report = {'patch':'16.17','source_plan_sha256':sha256(target/'source_plan.json'),'timing_guard_ms':1000,
            'final_test_payload_opened':False,'original_test_explicitly_excluded':True,'games':[recovered[new_record['match_id']]]}
        write_json(target/'source_report.json',small_report)
        small_summary = {**summary,'source_report':str(target/'source_report.json'),
            'source_report_sha256':sha256(target/'source_report.json'),
            'previous_qualification_sha256':sha256(target/'prior_qualification.json'),
            'recovered_match_ids':[new_record['match_id']],'additional_source_consistent_games':1}
        write_json(target/'summary.json',small_summary)
        corpus = build(target/'source_plan.json',target/'prior_qualification.json',target/'summary.json',target/'corpus')
        from audit_recovered_patch_corpus import run,aggregate
        from qualify_patch_corpus import qualify
        from report_corpus_coverage import coverage
        from publish_patch_dataset import publish
        from app.dataset_generations import verify_generation
        evidence = target/'acceptance'
        evidence.mkdir()
        audit_path = evidence/'semantic_audit.json'
        assert run(corpus,audit_path,workers=2,max_seconds=90)
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        assert audit['counts']['games']==2 and audit['passed'] and not audit['hard_failures']
        assert audit['audit_scope']=='complete_recovered_corpus'
        first_shard = Path(audit['shard_reports'][0]['path'])
        refused(lambda:qualify(corpus,first_shard,evidence/'bad-shard-qualification.json'))
        receipt = evidence/'semantic_audit_workers/preflight.json'
        refused(lambda:aggregate(corpus,receipt,[first_shard,first_shard],evidence/'bad-aggregate.json',0))
        qualification = evidence/'qualification.json'
        qualify(corpus,audit_path,qualification)
        coverage(qualification,evidence/'coverage',purpose='patch_development',benchmark_id='bounded-footwear-pipeline-fixture',
            eval_version=2,split_assignment=target/'split_assignment.json',split_metadata=target/'split_metadata.json')
        dataset = publish(evidence/'coverage',target/'datasets','fixture-recovered-v2')
        result = verify_generation(dataset)
        assert result['purpose']=='patch_development' and result['source_target_version']=='full-source-completions-v1'
        split = json.loads((dataset/'split_manifest.json').read_text(encoding='utf-8'))
        assert split['counts']=={'train':1,'val':1,'test':0}
        assert (dataset/'visits_test.jsonl').stat().st_size==0
        assert split['footwear_state_versions']=={'footwear-timer-slots-v1':sum(split['rows_per_split'].values())}
        print(json.dumps({'bounded_pipeline':'passed','temporary_output':str(target),
            'games':split['counts'],'rows':split['rows_per_split'],'audit_elapsed_seconds':audit['elapsed_seconds'],
            'final_test_payload_opened':False,'cleanup':'temporary corpus and generation removed on context exit'},indent=2))
    finally:
        for file in target.rglob('*'):
            if file.is_file(): file.chmod(0o666)

print('ok recovered pipeline: complete disjoint audit, shard rejection, fresh qualification, preserved folds, export provenance')
