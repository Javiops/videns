"""Reproduce only the first failed training-fold solve; no held-out predictions."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.recipe_transition_data_v1 import RecipeData, array_hash
from winprob.recipe_transition_probe_v1 import Standardizer, game_weights
from winprob.recipe_transition_v1 import fit, objective
from winprob.iql_dataset_contract import file_hash


def run(source, output):
    started = time.monotonic()
    if output.exists():
        raise ValueError('immutable new diagnostic required')
    data = RecipeData(source)
    labels = data.load_targets(ROOT/'data/runs/recipe-transition-fit-20261001-v3/experiment/inputs_lock.json')
    train = np.flatnonzero(data.folds != 0); raw = data.batch(train)
    scaler = Standardizer.fit(raw, data.game_ids[train])
    inputs = scaler.transform(raw, 'raw'); y = labels['coarsened'][train]; weights = game_weights(data.game_ids[train])
    theta, receipt = fit(inputs, y, weights, ridge=.01, max_iter=300)
    loss, gradient = objective(theta.ravel(), inputs, y, weights, ridge=.01)
    checks = []; rng = np.random.default_rng(20261001)
    for _ in range(3):
        direction = rng.normal(size=676); direction /= np.linalg.norm(direction); epsilon = 1e-4
        numerical = (objective(theta.ravel()+epsilon*direction, inputs, y, weights)[0]
                     - objective(theta.ravel()-epsilon*direction, inputs, y, weights)[0])/(2*epsilon)
        analytic = float(gradient @ direction)
        checks.append({'analytic': analytic, 'finite_difference': numerical, 'absolute_error': abs(analytic-numerical)})
    data.revalidate(); output.mkdir(parents=True)
    np.save(output/'unqualified_coefficients.npy', theta, allow_pickle=False)
    result = {'schema': 'recipe-solver-diagnostic-v1', 'state': 'complete', 'native_manifest_sha256': data.digest,
        'train_fold': 1, 'arm': 'raw', 'train_states': len(train), 'train_indices_sha256': array_hash(train),
        'fit_receipt': receipt, 'recomputed_objective': loss, 'gradient_max_abs': float(np.max(np.abs(gradient))),
        'gradient_l2': float(np.linalg.norm(gradient)), 'directional_checks': checks,
        'heldout_predictions_computed': False, 'purchase_accuracy_computed': False,
        'coefficients_qualified': bool(receipt['converged'] and receipt['gradient_max_abs'] < 1e-6),
        'test_payload_reads': False, 'confirmation_payload_reads': False,
        'coefficients_sha256': file_hash(output/'unqualified_coefficients.npy'), 'elapsed_s': time.monotonic()-started,
        'code_sha256': {str(p): file_hash(p) for p in (Path(__file__), ROOT/'winprob/recipe_transition_v1.py',
            ROOT/'winprob/recipe_transition_probe_v1.py', ROOT/'winprob/recipe_transition_data_v1.py')}}
    report = output/'report.json'
    report.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'report.sha256').write_text(file_hash(report)+' report.json\n', encoding='ascii')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); run(a.source, a.output)
