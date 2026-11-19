"""Verify per-model TRAIN counts, convex residuals and all frozen predictions."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.native_bias_data_v1 import NativeBiasData, PROTOCOL_SHA, ARMS, opposite_fold_rows, array_hash
from winprob.native_bias_calibration_v1 import predict, normalized_prior, require
from winprob.native_bias_gpu_v1 import configure, CudaBiasObjective
from winprob.bounded_npy_rows_v1 import BoundedNpyRows


def run(directory, output):
    start = time.monotonic(); directory = directory.resolve(); output = output.resolve()
    require(directory.is_relative_to(ROOT/'data/runs') and not output.exists(), 'new own calibration audit required')
    bindings = {str(Path(__file__).resolve()):file_hash(__file__)}; report,rh = bound_json(directory/'report.json',bindings)
    dist,dh = bound_json(directory/'distribution/manifest.json',bindings)
    require(report['schema'] == 'native-bias-fit-v1' and report['state'] == 'complete' and report['actual_fits'] == 2
            and set(report['models']) == {'0','1'}
            and report['accuracy_computed'] is False and dh == report['distribution_manifest_sha256']
            and dist['schema'] == 'native-bias-distribution-v1' and dist['state'] == 'complete'
            and dist['selection_count'] == 2900 and dist['groups'] == 9333 and dist['arms'] == list(ARMS)
            and dist['protocol_sha256'] == PROTOCOL_SHA and dist['models'] == report['models'], 'completed frozen calibration required')
    for raw,sha in report['code_sha256'].items():
        p = Path(raw).resolve()
        require(p.is_relative_to(ROOT) and file_hash(p) == sha
                and file_hash(directory/'code_snapshot'/p.relative_to(ROOT)) == sha, 'fit code/snapshot changed')
        bindings[str(p)] = sha
    for name,sha in dist['files'].items():
        p = (directory/'distribution'/name).resolve()
        require(p.parent == directory/'distribution' and file_hash(p) == sha, 'prediction bytes changed')
        bindings[str(p)] = sha
    model_lock,mh = bound_json(directory/'models_lock.json',bindings)
    pred_lock,ph = bound_json(directory/'predictions_lock.json',bindings)
    counts_lock,ch = bound_json(directory/'counts_lock.json',bindings)
    control,coh = bound_json(directory/'control_lock.json',bindings)
    preflight,_ = bound_json(directory/'preflight.json',bindings)
    require(model_lock['state'] == 'both_calibrators_frozen_before_prediction' and model_lock['models'] == report['models']
            and model_lock['counts_lock_sha256'] == ch and model_lock['control_lock_sha256'] == coh
            and pred_lock['state'] == 'calibrated_distributions_frozen_before_native_scoring'
            and pred_lock['models_lock_sha256'] == mh and pred_lock['files'] == dist['files']
            and ph == dist['prediction_lock_sha256'] and counts_lock['actual_label_storage_released'] is True,
            'fit/prediction/count freeze changed')
    require(counts_lock['state'] == 'opposite_fold_counts_frozen_before_fits'
            and control['state'] == 'class_control_frozen_before_targets' and control['seed'] == 20261004
            and preflight == report['preflight'] and preflight['passed'] is True
            and preflight['real_purchase_targets_opened'] is False, 'pre-target resource/control boundary changed')
    def guard(): require(time.monotonic()-start < 3600,'fixed60minute model audit bound')
    data = NativeBiasData(); lock,ilh = bound_json(directory/'inputs_lock.json',bindings)
    require(ilh == dist['inputs_lock_sha256'] and all(lock.get(k) == v for k,v in data.input_lock().items()), 'original calibration inputs changed')
    expected_selection = [{**data.entries[int(row)], 'query_ordinal': i} for i,row in enumerate(data.focals)]
    require(json.loads((directory/'distribution/selection.json').read_text(encoding='utf-8')) == expected_selection
            and json.loads((directory/'distribution/group_main.json').read_text(encoding='utf-8')) == data.base.groups,
            'frozen prediction query/catalogue identity differs')
    permutation = np.load(directory/'class_permutation.npy',allow_pickle=False)
    require(file_hash(directory/'class_permutation.npy') == control['permutation_sha256']
            and file_hash(directory/'basket_sizes.npy') == control['sizes_sha256']
            and np.array_equal(permutation,data.permutation)
            and np.array_equal(np.load(directory/'basket_sizes.npy',allow_pickle=False),data.sizes), 'public class control differs')
    for name,sha in counts_lock['files'].items():
        p = (directory/name).resolve()
        require(p.parent == directory and file_hash(p) == sha, 'TRAIN count receipt changed'); bindings[str(p)] = sha
    # Target access here verifies TRAIN counts per recipient fold. It is not
    # a claim that no evaluation-game label exists anywhere across both fits.
    data.load_targets(directory/'inputs_lock.json')
    require(data.base.context_manifest['files']['targets.jsonl'] == counts_lock['target_ledger_sha256'], 'target ledger identity changed')
    counts = {}; identities = {}
    for fold in (0,1):
        counts[fold],identities[fold] = data.counts_for(fold,guard)
        direct = np.bincount(data.labels['main'][opposite_fold_rows(data.folds,fold)],minlength=9333)
        stored = np.load(directory/f'train_counts_fold{fold}.npy',allow_pickle=False)
        identity = json.loads((directory/f'training_identity_fold{fold}.json').read_text(encoding='utf-8'))
        require(np.array_equal(stored,counts[fold]) and np.array_equal(direct,counts[fold])
                and identity == identities[fold] and identity['heldout_targets_used'] is False,
                'per-fold actual TRAIN count or identity differs')
    data.labels = None
    runtime = configure(); proofs = {}; biases = {}
    for fold in (0,1):
        spec = report['models'][str(fold)]; path = (directory/spec['file']).resolve()
        require(path.parent == directory and file_hash(path) == spec['sha256'], 'bias model changed')
        bindings[str(path)] = spec['sha256']; b = np.load(path,allow_pickle=False)
        require(b.shape == (9333,) and b.dtype == np.float64 and np.isfinite(b).all(), 'invalid bias model')
        solver,sh = bound_json(directory/f'solver_fold{fold}.json',bindings)
        require(sh == spec['solver_sha256'] and solver == report['fit_receipts'][str(fold)]
                and file_hash(directory/f'training_identity_fold{fold}.json') == spec['training_identity_sha256']
                and file_hash(directory/f'train_counts_fold{fold}.npy') == spec['counts_sha256']
                and solver['success'] is True and solver['qualified'] is True and solver['ridge'] == 1.
                and solver['iterations'] <= 500 and solver['optimizer_evaluations'] <= 2000 and solver['gradient_rechecks'] == 1,
                'fixed optimizer receipt differs')
        calculation = CudaBiasObjective(data.priors,opposite_fold_rows(data.folds,fold),counts[fold],guard=guard)
        try: value,gradient = calculation(b)
        finally: calculation.close(); del calculation; gc.collect()
        residual = float(np.max(np.abs(gradient))); gap = float(gradient@gradient/2); distance = float(np.linalg.norm(gradient))
        require(residual < 1e-5 and np.isclose(value,solver['relative_penalized_nll_sum'],atol=1e-8,rtol=1e-12)
                and np.isclose(residual,solver['gradient_max_abs'],atol=1e-10,rtol=1e-8)
                and np.isclose(gap,solver['numerical_strong_convexity_gap_bound'],atol=1e-14,rtol=1e-8)
                and np.isclose(distance,solver['numerical_distance_to_unique_minimum_bound'],atol=1e-10,rtol=1e-8), 'convex stationarity proof did not reproduce')
        biases[fold] = b; proofs[str(fold)] = {'training_rows': len(opposite_fold_rows(data.folds,fold)),
            'gradient_max_abs': residual,'numerical_gap_bound':gap,'numerical_distance_bound':distance}
        print(json.dumps({'stage':'verified_convex_map','fold':fold,'gradient_max_abs':residual}),flush=True)
    matrices = {a:BoundedNpyRows(directory/'distribution'/f'{a}.npy',expected_shape=(2900,9333),expected_dtype=np.float64) for a in ARMS}
    native = BoundedNpyRows(directory/'distribution/m2.npy',expected_shape=(2900,9333),expected_dtype=np.float32)
    require(file_hash(directory/'distribution/m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'native M2 prior changed')
    neutral = np.load(directory/'distribution/neutral_flags.npy',allow_pickle=False)
    require(neutral.shape == (2,2900) and neutral.dtype == bool, 'neutral flags changed')
    replayed = baseline = 0; qfold = data.folds[data.focals]
    for fold in (0,1):
        positions = np.flatnonzero(qfold == fold); b = biases[fold]
        for lo in range(0,len(positions),128):
            guard(); local = positions[lo:lo+128]; p = data.priors[data.focals[local]]
            require(data.labels is None and np.array_equal(p,native[local]), 'baseline/current-label inference boundary changed')
            baseline += len(local)
            for j,arm in enumerate(ARMS):
                coefficients = b if arm == 'calibrated' else b[permutation]
                q = predict(p,coefficients); identity = bool(np.all(coefficients == 0))
                require(np.array_equal(q,matrices[arm][local]) and np.all(neutral[j,local] == identity), 'calibration forecast did not replay exactly')
                if identity: require(np.array_equal(q,normalized_prior(p)), 'zero correction changed native prior')
                replayed += len(local)
    require(replayed == 5800 and baseline == 2900, 'replay population incomplete')
    data.revalidate()
    for p,sha in bindings.items(): require(file_hash(p) == sha,'frozen calibration artifact changed during audit')
    output.mkdir(parents=True)
    result = {'schema':'native-bias-model-audit-v1','state':'complete','passed':True,
        'fit_report_sha256':rh,'distribution_manifest_sha256':dh,'bias_queries_replayed_exactly':replayed,
        'native_prior_queries_verified':baseline,'training_count_rows_support_verified':56288,
        'per_model_training_count_and_identity_verified':True,'stationarity':proofs,
        'training_ledger_opened_for_per_model_count_verification':True,'current_labels_used_for_query_prediction':False,
        'test_payload_reads':False,'confirmation_payload_reads':False,'source_runtime_imported':False,
        'runtime':runtime,'elapsed_s':time.monotonic()-start,'auditor_sha256':file_hash(__file__),
        'limits':['Numerical strong-convexity residual verification, not interval arithmetic or a second fitting sweep.',
                  'Exposed internal evidence; no forward gain, causal value, native decoder rerun or goal completion.']}
    p = output/'report.json'; p.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    p.with_suffix('.sha256').write_text(file_hash(p)+' report.json\n',encoding='ascii')
    print(json.dumps({k:result[k] for k in ('state','passed','bias_queries_replayed_exactly','elapsed_s')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.directory,args.output)
