"""Replay frozen proxy banks/forecasts, label boundaries and anchored decisions."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import torch
from scipy.special import logsumexp
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.bayes_proxy_data_v2 import ProxyData
from winprob.bayes_proxy_state_v1 import StateOnlyStore, embedding, state_probabilities
from winprob.bayes_proxy_decision_v1 import ARMS, decisions, swap_evidence
from winprob.bayes_proxy_ratio_v1 import require
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.purchase_memory_v1 import normalized
from forecast_bayes_proxy_v3 import load_model, retrieve, array_sha


def run(directory, output):
    started = time.monotonic(); directory = directory.resolve(); output = output.resolve()
    require(directory.is_relative_to(ROOT/'data/runs') and not output.exists(), 'immutable own forecast audit required')
    bindings = {}; report, rh = bound_json(directory/'report.json', bindings)
    manifest, mh = bound_json(directory/'distribution/manifest.json', bindings)
    require(report['state'] == 'complete' and report['schema'] == 'bayes-proxy-forecast-v1'
            and report['training_performed'] is False and report['accuracy_computed'] is False
            and mh == report['distribution_manifest_sha256'], 'complete unscored forecast package required')
    for raw, sha in report['code_sha256'].items():
        p = Path(raw).resolve(); require(p.is_relative_to(ROOT) and file_hash(p) == sha
            and file_hash(directory/'code_snapshot'/p.relative_to(ROOT)) == sha, 'forecast code/snapshot changed')
        bindings[str(p)] = sha
    lock, lh = bound_json(directory/'forecast_lock.json', bindings)
    pred_lock, _ = bound_json(directory/'predictions_lock.json', bindings)
    bank_lock, _ = bound_json(directory/'banks_lock.json', bindings)
    require(lock['state'] == 'proxy_forecasts_and_closed_likelihoods_frozen_before_posterior'
            and pred_lock['state'] == 'all_distributions_frozen_before_scoring'
            and pred_lock['forecast_lock_sha256'] == lh and pred_lock['files'] == manifest['files']
            and bank_lock['banks'] == report['banks'], 'freeze lineage changed')
    for parent, files in ((directory, lock['files']), (directory/'proxy', lock['proxy_files']), (directory/'distribution', manifest['files'])):
        for name, sha in files.items():
            p = (parent/name).resolve(); require(p.parent == parent and file_hash(p) == sha, 'saved forecast member changed'); bindings[str(p)] = sha
    for name, sha in (('forecast_hashes.json', lock['forecast_hashes_sha256']), ('posterior_weights.npy', pred_lock['posterior_weights_sha256'])):
        p = directory/name; require(file_hash(p) == sha, 'evidence receipt changed'); bindings[str(p)] = sha
    def guard(): require(time.monotonic()-started < 7200, 'fixed120minute forecast replay deadline')
    def status(stage, **extra):
        guard(); print(json.dumps({'stage': stage, 'elapsed_s': time.monotonic()-started, **extra}), flush=True)
    status('validating_causal_inputs')
    data = ProxyData(); data.enable_labels(directory/'inputs_lock.json')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'original CUDA inference required for exact replay')
    store = StateOnlyStore(data.features, 'cuda')
    forecast_rows = np.load(directory/'forecast_state_indices.npy', allow_pickle=False)
    require(np.array_equal(forecast_rows, np.sort(np.r_[data.forecast_rows[0], data.forecast_rows[1]])), 'forecast batch order changed')
    positions = {int(row): i for i, row in enumerate(forecast_rows)}
    current_lookup = {int(row): i for i, row in enumerate(data.query_indices)}
    past_lookup = {int(row): i for i, row in enumerate(data.prefix)}
    hashes = json.loads((directory/'forecast_hashes.json').read_text(encoding='utf-8'))
    nr = np.load(directory/'neighbor_rows.npy', allow_pickle=False)
    nw = np.load(directory/'neighbor_weights.npy', allow_pickle=False)
    nd = np.load(directory/'neighbor_distances.npy', allow_pickle=False)
    saved_likelihood = np.load(directory/'past_log_likelihoods.npy', allow_pickle=False)
    saved_labels = np.load(directory/'past_main_labels.npy', allow_pickle=False)
    recomputed = np.full_like(saved_likelihood, np.nan)
    states = BoundedNpyRows(directory/'proxy/state.npy', expected_shape=(2900, 9333), expected_dtype=np.float64)
    memories = BoundedNpyRows(directory/'proxy/memory.npy', expected_shape=(2900, 9333), expected_dtype=np.float64)
    bank_rows_checked = forecast_checked = denied_current = denied_future = 0
    raw_label_calls = []; original_loader = data.labels.loader
    def counted_loader(name, rows):
        raw_label_calls.append((name, len(rows)))
        return original_loader(name, rows)
    data.labels.loader = counted_loader
    for i, q in enumerate(data.queries):
        requests = [(q['state_index'], 'current')]
        episode = data.base.episodes[q['episode_index']]; last = episode['first_state']+episode['states']-1
        if last > q['state_index']: requests.append((last, 'future'))
        for row, kind in requests:
            rejected = False
            try: data.labels.read(np.array([row]), q['fold'], 'past')
            except ValueError: rejected = True
            require(rejected, 'recipient current/future label access was accepted')
            denied_current += int(kind == 'current'); denied_future += int(kind == 'future')
    require(not raw_label_calls, 'denied label requests reached raw label storage')
    data.labels.loader = original_loader
    for fold in (0, 1):
        guard(); model = load_model(data, fold); train = np.flatnonzero(data.folds != fold)
        bank_path = directory/'banks'/f'embedding_fold{fold}.npy'; receipt = report['banks'][str(fold)]
        require(file_hash(bank_path) == receipt['embedding_sha256']
                and np.array_equal(train, np.load(directory/'banks'/f'rows_fold{fold}.npy', allow_pickle=False))
                and array_sha(train) == receipt['training_indices_sha256'], 'training bank identity changed')
        bindings[str(bank_path)] = receipt['embedding_sha256']
        bank_reader = BoundedNpyRows(bank_path, expected_shape=(len(train), 128), expected_dtype=np.float64)
        status('replaying_training_embeddings', fold=fold, rows=len(train))
        with torch.inference_mode():
            for lo in range(0, len(train), 256):
                guard(); batch = train[lo:lo+256]; expected = normalized(embedding(model, store, batch).cpu().numpy())
                require(np.array_equal(expected, bank_reader[lo:lo+len(batch)]), 'training embedding replay differs')
                bank_rows_checked += len(batch)
        training_labels = data.labels.read(train, fold, 'training')
        label_path = directory/'banks'/f'labels_fold{fold}.npy'
        require(file_hash(label_path) == receipt['labels_sha256']
                and np.array_equal(training_labels, np.load(label_path, allow_pickle=False)), 'bank labels do not match allowed TRAIN rows')
        bindings[str(label_path)] = receipt['labels_sha256']
        label_map = np.full(len(data.folds), -1, dtype=np.int32); label_map[train] = training_labels
        bank = np.load(bank_path, mmap_mode='r', allow_pickle=False)
        by_champion = {int(ch): np.flatnonzero(data.champions[train] == ch) for ch in np.unique(data.champions[data.forecast_rows[fold]])}
        rows = data.forecast_rows[fold]; status('replaying_all_proxy_forecasts', fold=fold, rows=len(rows))
        with torch.inference_mode():
            for lo in range(0, len(rows), 256):
                guard(); batch = rows[lo:lo+256]; h = embedding(model, store, batch)
                vectors = normalized(h.cpu().numpy()); probabilities = state_probabilities(model, h)
                for j, raw_row in enumerate(batch):
                    row = int(raw_row); pos = positions[row]; state_p = probabilities[j]
                    memory_p, ids, distances, weights = retrieve(data, fold, row, vectors[j], state_p, bank, train, label_map, by_champion)
                    count = len(ids); saved = hashes[pos]
                    require(saved == {'state_index': row, 'fold': fold, 'state_sha256': array_sha(state_p),
                        'memory_sha256': array_sha(memory_p), 'embedding_sha256': array_sha(vectors[j])}, 'proxy forecast/embedding hash replay differs')
                    require(np.array_equal(nr[pos, :count], ids) and np.all(nr[pos, count:] == -1)
                            and np.array_equal(nw[pos, :count], weights) and np.all(nw[pos, count:] == 0)
                            and np.array_equal(nd[pos, :count], distances) and np.all(np.isinf(nd[pos, count:])), 'retrieval receipt replay differs')
                    if count:
                        expected = np.exp(-(distances-distances[0])/max(float(distances[-1])/4, 1e-6)); expected /= expected.sum()
                        require(np.array_equal(expected, weights), 'independent distance-weight formula differs')
                    if row in current_lookup:
                        q = current_lookup[row]
                        require(np.array_equal(state_p, states[q]) and np.array_equal(memory_p, memories[q]), 'saved current proxy matrix differs')
                    else:
                        p = past_lookup[row]; y = int(data.labels.read(np.array([row]), fold, 'past')[0])
                        require(y == saved_labels[p], 'closed evidence label changed')
                        recomputed[p] = np.log([state_p[y], memory_p[y]])
                    forecast_checked += 1
        bank._mmap.close(); del bank, model, h, vectors, probabilities, label_map, training_labels, by_champion
        gc.collect(); torch.cuda.empty_cache()
    require(bank_rows_checked == 392448 and forecast_checked == 18473
            and np.array_equal(recomputed, saved_likelihood), 'incomplete bank/likelihood replay')
    saved_weights = np.load(directory/'posterior_weights.npy', allow_pickle=False)
    neutral = np.load(directory/'distribution/neutral_flags.npy', allow_pickle=False)
    matrices = {arm: BoundedNpyRows(directory/'distribution'/f'{arm}.npy', expected_shape=(2900, 9333), expected_dtype=np.float64) for arm in ARMS}
    require(file_hash(directory/'distribution/m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'original M2 focal matrix changed')
    max_posterior_error = max_formula_error = 0.; status('replaying_anchored_decisions')
    for i, q in enumerate(data.queries):
        guard(); lo, hi = map(int, data.offsets[i:i+2]); likelihood = recomputed[lo:hi]
        swaps = np.array([swap_evidence(q['match_id'], q['participant_id'], int(data.times[row])) for row in data.prefix[lo:hi]], dtype=bool)
        raw = data.native[i]; state_p = states[i]; memory_p = memories[i]
        predicted = decisions(raw, state_p, memory_p, likelihood, swaps)
        altered = likelihood.copy(); altered[swaps] = altered[swaps, ::-1]
        for arm_index, arm in enumerate(ARMS):
            result = predicted[arm]
            require(np.array_equal(result['probabilities'], matrices[arm][i])
                    and np.array_equal(result['weights'], saved_weights[arm_index, i])
                    and result['neutral'] == neutral[arm_index, i], 'exact posterior/anchored prediction replay differs')
            if arm == 'static_memory': independent = np.array([0., 1.]) if hi > lo else np.array([.5, .5])
            else:
                joint = np.log([.5, .5])+(likelihood if arm == 'history' else altered).sum(0)
                independent = np.exp(joint-logsumexp(joint)); independent /= independent.sum()
            error = float(np.max(np.abs(independent-result['weights']))); max_posterior_error = max(max_posterior_error, error)
            require(error <= 2e-12, 'independent likelihood-product posterior differs')
            p = raw.astype(np.float64); p /= p.sum()
            ratio = (independent[0]*state_p+independent[1]*memory_p)/(.5*state_p+.5*memory_p)
            expected = p*ratio; expected /= expected.sum()
            error = float(np.max(np.abs(expected-result['probabilities']))); max_formula_error = max(max_formula_error, error)
            require(error <= 2e-12, 'independent ratio transfer differs')
    require(np.all(neutral[:, np.diff(data.offsets) == 0]), 'zero-history parity failed')
    data.revalidate()
    for p, sha in bindings.items(): require(file_hash(p) == sha, 'forecast evidence changed during replay')
    output.mkdir(parents=True)
    result = {'schema': 'bayes-proxy-forecast-audit-v1', 'state': 'complete', 'passed': True,
        'forecast_report_sha256': rh, 'distribution_manifest_sha256': mh,
        'bank_embeddings_replayed_exactly': bank_rows_checked, 'proxy_queries_replayed_exactly': forecast_checked,
        'past_likelihood_pairs_replayed_exactly': 15573, 'anchored_distributions_replayed_exactly': 8700,
        'baseline_queries_verified': 2900, 'denied_current_label_requests': denied_current,
        'denied_future_label_requests': denied_future, 'maximum_independent_posterior_error': max_posterior_error,
        'maximum_independent_ratio_error': max_formula_error, 'zero_history_neutrality_verified': True,
        'query_accuracy_computed': False, 'recipient_fold_current_future_labels_used_for_its_predictions': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'elapsed_s': time.monotonic()-started, 'auditor_sha256': file_hash(__file__),
        'limits': ['Checks fixed causal proxies and ratio arithmetic; does not establish private intent, M2 regret or a purchase-accuracy gain.']}
    p = output/'report.json'; p.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    p.with_suffix('.sha256').write_text(file_hash(p)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'proxy_queries_replayed_exactly', 'anchored_distributions_replayed_exactly', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.directory, a.output)
