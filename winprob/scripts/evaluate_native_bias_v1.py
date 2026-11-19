"""Score fixed native calibration maps only after model and native audits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from winprob.native_bias_io_v1 import CompletedNativeBias
from winprob.native_bias_evaluation_v1 import evaluate, require
from winprob.iql_dataset_contract import bound_json, file_hash


def run(args):
    start = time.monotonic(); output = args.output.resolve(); require(not output.exists(),'new immutable score required')
    data = CompletedNativeBias(args.directory,args.decoded_manifest,args.native_manifest)
    audit,ah = bound_json(args.model_audit,{})
    require(audit['schema'] == 'native-bias-model-audit-v1' and audit['state'] == 'complete' and audit['passed'] is True
            and audit['fit_report_sha256'] == data.fit_sha and audit['distribution_manifest_sha256'] == data.dist_sha
            and audit['bias_queries_replayed_exactly'] == 5800 and audit['native_prior_queries_verified'] == 2900
            and audit['per_model_training_count_and_identity_verified'] is True
            and audit['current_labels_used_for_query_prediction'] is False, 'complete calibrated-model audit required')
    output.mkdir(parents=True); lock = output/'scoring_inputs_lock.json'
    lock.write_text(json.dumps(data.scoring_lock(),sort_keys=True,indent=2)+'\n',encoding='utf-8')
    labels,_ = data.load_targets(lock)
    metrics = evaluate(labels,data.scores(labels),data.base.game_ids[data.query].tolist(),data.folds,data.legality)
    require(metrics['games'] == 2900 and metrics['arms']['m2']['main_top1']['correct'] == 1483
            and metrics['arms']['m2']['main_top3']['correct'] == 1959, 'original matched M2 baseline did not reproduce')
    data.revalidate()
    report = {'schema':'native-bias-evaluation-v1','state':'complete',**metrics,
        'fit_directory':str(data.directory),'fit_report_sha256':data.fit_sha,'distribution_manifest_sha256':data.dist_sha,
        'model_audit_path':str(args.model_audit.resolve()),'model_audit_sha256':ah,'scoring_inputs_lock_sha256':file_hash(lock),
        'source_manifests':{world:{k:v[k] for k in ('manifest_path','manifest_sha256')} for world,v in data.source.items()},
        'costs':{'optimization_seconds':sum(v['elapsed_fit_s'] for v in data.fit['fit_receipts'].values()),
            'optimizer_and_recheck_evaluations':sum(v['evaluations'] for v in data.fit['fit_receipts'].values()),
            'fit_worker_elapsed_s':data.fit['elapsed_s'], 'model_audit_elapsed_s':audit['elapsed_s'],
            'source_stage_elapsed_s':{world:v['manifest']['elapsed_s'] for world,v in data.source.items()},
            'incremental_purchase_costs':data.costs},
        'test_payload_reads':False,'confirmation_payload_reads':False,'source_runtime_imported':False,
        'promotion_authorized':False,'practical_breakthrough_established':False,
        'bindings_sha256':data.bindings,'code_sha256':{str(p.resolve()):file_hash(p) for p in (
            Path(__file__),ROOT/'winprob/native_bias_io_v1.py',ROOT/'winprob/native_bias_evaluation_v1.py')},
        'elapsed_s':time.monotonic()-start,
        'scope':'Exposed retrospective16.17 internal comparison with native16.18 M2; not forward/live confirmation.'}
    p=output/'report.json';p.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    p.with_suffix('.sha256').write_text(file_hash(p)+' report.json\n',encoding='ascii')
    print(json.dumps({k:report[k] for k in ('state','metric_gate_passed','selected_candidate','elapsed_s')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--decoded-manifest',type=Path,required=True);p.add_argument('--native-manifest',type=Path,required=True)
    p.add_argument('--model-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
