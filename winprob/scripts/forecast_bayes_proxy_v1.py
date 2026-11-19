"""Frozen causal proxy forecasts, closed evidence and M2 ratio outputs; no scoring."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from winprob.iql_dataset_contract import file_hash
from winprob.history_probe_models import HistoryProbeActor, set_history_validation
from winprob.bayes_proxy_data_v1 import ProxyData, PREFIX, OLD, PROTOCOL, CHECKPOINTS
from winprob.bayes_proxy_state_v1 import StateOnlyStore, embedding, state_probabilities
from winprob.bayes_proxy_decision_v1 import ARMS, decisions, swap_evidence
from winprob.bayes_proxy_ratio_v1 import require
from winprob.purchase_memory_v1 import normalized, neighbors, mix
from winprob.npy_row_writer_v1 import NpyRowWriter
from winprob.bounded_npy_rows_v1 import BoundedNpyRows

PROTOCOL_SHA = 'a60292830d167824f8d14a608020c4f75da3a023b65ef85d5c42ad7f2b91837b'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_model(data, fold):
    path = OLD/f'actor_state_fold{fold}.pt'; require(file_hash(path) == CHECKPOINTS[fold], 'frozen checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    training_ids = sorted(g['match_id'] for g in data.base.games if g['fold'] != fold)
    require(checkpoint['arm'] == 'state' and checkpoint['scored_fold'] == fold and checkpoint['epochs'] == 20
            and sorted(checkpoint['training_game_ids']) == training_ids, 'checkpoint training fold changed')
    model = HistoryProbeActor(data.num_champions, data.num_items).to('cuda')
    model.load_state_dict(checkpoint['actor'], strict=True); model.eval(); set_history_validation(model, False)
    del checkpoint
    return model


def retrieve(data, fold, query_row, vector, state, bank, training_rows, label_map, by_champion):
    eligible = by_champion.get(int(data.champions[query_row]), np.empty(0, dtype=np.int64))
    ids, distances, weights = neighbors(vector, bank[eligible], memory_rows=training_rows[eligible],
        memory_games=data.games[training_rows[eligible]], query_game=int(data.games[query_row]))
    require(np.all(data.folds[ids] != fold) and np.all(data.champions[ids] == data.champions[query_row])
            and len(np.unique(data.games[ids])) == len(ids) and np.all(label_map[ids] >= 0), 'retrieval escaped its training fold/champion/game')
    return mix(state, label_map[ids], weights), ids, distances, weights


def own_code():
    result = {str(PROTOCOL): file_hash(PROTOCOL)}
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if not value: continue
        p = Path(value).resolve()
        if p.is_relative_to(ROOT) and p.suffix == '.py' and '.venv' not in p.parts and 'data' not in p.relative_to(ROOT).parts:
            result[str(p)] = file_hash(p)
    return result


def run(output):
    start = time.monotonic(); output = output.resolve()
    require(not output.exists() and file_hash(PROTOCOL) == PROTOCOL_SHA, 'new run and unchanged predictive protocol required')
    require(shutil.disk_usage(ROOT).free >= 5*1024**3, '5GiB free artifact space required')
    output.mkdir(parents=True); writers = []
    def guard(): require(time.monotonic()-start < 10800, 'fixed180minute forecast deadline reached')
    def status(stage, **extra):
        guard(); value = {'stage': stage, 'elapsed_s': time.monotonic()-start, **extra}
        (output/'status.json').write_text(json.dumps(value)+'\n', encoding='utf-8'); print(json.dumps(value), flush=True)
    def writer(path, shape, dtype):
        value = NpyRowWriter(path, shape, dtype); writers.append(value); return value
    try:
        status('validating_public_inputs')
        data = ProxyData(); code = own_code()
        for raw, sha in code.items():
            p = Path(raw); destination = output/'code_snapshot'/p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, destination)
        write(output/'inputs_lock.json', {'state': 'inputs_frozen_before_label_access',
            'prefix_report_sha256': file_hash(PREFIX/'report.json'), 'bindings_sha256': data.bindings,
            'code_sha256': code, 'protocol_sha256': PROTOCOL_SHA, 'query_games': 2900, 'past_visits': 15573})
        data.enable_labels(output/'inputs_lock.json')
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available(), 'idle CUDA required for frozen proxy inference')
        store = StateOnlyStore(data.features, 'cuda')
        banks = output/'banks'; banks.mkdir(); bank_receipts = {}; timing = []
        for fold in (0, 1):
            train = np.flatnonzero(data.folds != fold); model = load_model(data, fold)
            bank_path = banks/f'embedding_fold{fold}.npy'; bank_out = writer(bank_path, (len(train), 128), np.float64)
            np.save(banks/f'rows_fold{fold}.npy', train, allow_pickle=False)
            status('encoding_training_memory', fold=fold, rows=len(train))
            benchmark_start = time.monotonic(); bank_seconds = None
            with torch.inference_mode():
                for lo in range(0, len(train), 256):
                    guard(); rows = train[lo:lo+256]
                    h = embedding(model, store, rows).cpu().numpy()
                    bank_out.write(np.arange(lo, lo+len(rows)), normalized(h))
                    if lo+len(rows) == 1024: bank_seconds = (time.monotonic()-benchmark_start)/1024
                    if (lo+len(rows)) % 32768 == 0: status('encoding_training_memory', fold=fold, completed=lo+len(rows), rows=len(train))
            bank_out.close(); require(bank_seconds is not None, 'training-memory pilot incomplete')
            training_labels = data.labels.read(train, fold, 'training')
            np.save(banks/f'labels_fold{fold}.npy', training_labels, allow_pickle=False)
            label_map = np.full(len(data.folds), -1, dtype=np.int32); label_map[train] = training_labels
            bank = np.load(bank_path, mmap_mode='r', allow_pickle=False)
            by_champion = {int(ch): np.flatnonzero(data.champions[train] == ch) for ch in np.unique(data.champions[data.forecast_rows[fold]])}
            # Input-only timing, with TRAIN labels but no recipient past label.
            pilot = data.forecast_rows[fold][:64]; t = time.monotonic()
            with torch.inference_mode():
                h = embedding(model, store, pilot); vectors = normalized(h.cpu().numpy()); probabilities = state_probabilities(model, h)
            for j, row in enumerate(pilot): retrieve(data, fold, int(row), vectors[j], probabilities[j], bank, train, label_map, by_champion)
            query_seconds = (time.monotonic()-t)/len(pilot)
            timing.append({'fold': fold, 'bank_seconds_per_row': bank_seconds, 'query_seconds_per_row': query_seconds,
                           'timed_bank_rows': 1024, 'timed_query_rows': len(pilot), 'recipient_past_labels_read': False})
            bank_receipts[str(fold)] = {'training_rows': len(train), 'training_games': len(set(data.games[train])),
                'checkpoint_sha256': CHECKPOINTS[fold], 'training_indices_sha256': array_sha(train),
                'embedding_sha256': file_hash(bank_path), 'labels_sha256': file_hash(banks/f'labels_fold{fold}.npy')}
            bank._mmap.close(); del bank, model, h, probabilities, vectors, label_map, training_labels, by_champion
            gc.collect(); torch.cuda.empty_cache()
        remaining = 10800-(time.monotonic()-start)
        projected = 2*max(t['bank_seconds_per_row'] for t in timing)*392448+2*max(t['query_seconds_per_row'] for t in timing)*18473+600
        preflight = {'passed': projected <= .8*remaining, 'projected_seconds': projected, 'remaining_seconds': remaining,
            'timings': timing, 'recipient_past_labels_read': False, 'posterior_updates': 0, 'training_performed': False}
        write(output/'preflight.json', preflight); require(preflight['passed'], 'fixed proxy computation exceeds cost envelope')
        write(output/'banks_lock.json', {'state': 'training_banks_frozen_before_past_evidence', 'banks': bank_receipts})
        forecast_rows = np.sort(np.concatenate([data.forecast_rows[f] for f in (0, 1)]))
        require(len(forecast_rows) == 18473 and len(np.unique(forecast_rows)) == len(forecast_rows), 'forecast union changed')
        np.save(output/'forecast_state_indices.npy', forecast_rows, allow_pickle=False)
        position = {int(row): i for i, row in enumerate(forecast_rows)}
        current_lookup = {int(row): i for i, row in enumerate(data.query_indices)}
        past_lookup = {int(row): i for i, row in enumerate(data.prefix)}
        proxy = output/'proxy'; proxy.mkdir()
        current_state = writer(proxy/'state.npy', (2900, 9333), np.float64)
        current_memory = writer(proxy/'memory.npy', (2900, 9333), np.float64)
        neighbor_rows = np.full((18473, 32), -1, dtype=np.int64)
        neighbor_weights = np.zeros((18473, 32), dtype=np.float64)
        neighbor_distances = np.full((18473, 32), np.inf, dtype=np.float64)
        past_likelihoods = np.full((15573, 2), np.nan, dtype=np.float64); past_labels = np.full(15573, -1, dtype=np.int32)
        forecast_hashes = [None]*18473; empty_counts = {}; queried = 0
        for fold in (0, 1):
            model = load_model(data, fold); train = np.load(banks/f'rows_fold{fold}.npy', allow_pickle=False)
            training_labels = np.load(banks/f'labels_fold{fold}.npy', allow_pickle=False)
            require(array_sha(train) == bank_receipts[str(fold)]['training_indices_sha256']
                    and file_hash(banks/f'labels_fold{fold}.npy') == bank_receipts[str(fold)]['labels_sha256'], 'frozen bank changed')
            label_map = np.full(len(data.folds), -1, dtype=np.int32); label_map[train] = training_labels
            bank = np.load(banks/f'embedding_fold{fold}.npy', mmap_mode='r', allow_pickle=False)
            by_champion = {int(ch): np.flatnonzero(data.champions[train] == ch) for ch in np.unique(data.champions[data.forecast_rows[fold]])}
            rows = data.forecast_rows[fold]; empty_counts[str(fold)] = 0
            with torch.inference_mode():
                for lo in range(0, len(rows), 256):
                    guard(); batch = rows[lo:lo+256]; h = embedding(model, store, batch)
                    vectors = normalized(h.cpu().numpy()); probabilities = state_probabilities(model, h)
                    for j, raw_row in enumerate(batch):
                        row = int(raw_row); pos = position[row]; state_p = probabilities[j]
                        memory_p, ids, distances, weights = retrieve(data, fold, row, vectors[j], state_p, bank, train, label_map, by_champion)
                        count = len(ids); empty_counts[str(fold)] += int(count == 0)
                        neighbor_rows[pos, :count] = ids; neighbor_weights[pos, :count] = weights; neighbor_distances[pos, :count] = distances
                        # Forecast hashes freeze BEFORE reading this past observation.
                        forecast_hashes[pos] = {'state_index': row, 'fold': fold, 'state_sha256': array_sha(state_p),
                            'memory_sha256': array_sha(memory_p), 'embedding_sha256': array_sha(vectors[j])}
                        if row in current_lookup:
                            q = current_lookup[row]; current_state.write(np.array([q]), state_p[None]); current_memory.write(np.array([q]), memory_p[None])
                        else:
                            p = past_lookup[row]; label = int(data.labels.read(np.array([row]), fold, 'past')[0])
                            require(state_p[label] > 0 and memory_p[label] > 0, 'closed observation impossible under proxies')
                            past_labels[p] = label; past_likelihoods[p] = np.log([state_p[label], memory_p[label]])
                        queried += 1
                    status('forecasting_closed_history_and_current_states', fold=fold, completed=queried, forecast_rows=18473)
            bank._mmap.close(); del bank, model, h, probabilities, vectors, label_map, training_labels, by_champion
            gc.collect(); torch.cuda.empty_cache()
        current_state.close(); current_memory.close()
        require(queried == 18473 and all(h is not None for h in forecast_hashes) and np.isfinite(past_likelihoods).all()
                and np.all(past_labels >= 0), 'incomplete target-deferred proxy forecasts')
        for name, values in (('past_log_likelihoods', past_likelihoods), ('past_main_labels', past_labels),
                             ('neighbor_rows', neighbor_rows), ('neighbor_weights', neighbor_weights), ('neighbor_distances', neighbor_distances)):
            np.save(output/(name+'.npy'), values, allow_pickle=False)
        write(output/'forecast_hashes.json', forecast_hashes)
        write(output/'forecast_lock.json', {'state': 'proxy_forecasts_and_closed_likelihoods_frozen_before_posterior',
            'files': {p.name: file_hash(p) for p in output.iterdir() if p.suffix == '.npy'},
            'proxy_files': {p.name: file_hash(p) for p in proxy.iterdir()}, 'forecast_hashes_sha256': file_hash(output/'forecast_hashes.json')})
        dist = output/'distribution'; dist.mkdir()
        out = {arm: writer(dist/f'{arm}.npy', (2900, 9333), np.float64) for arm in ARMS}
        native_out = writer(dist/'m2.npy', (2900, 9333), np.float32)
        srows = BoundedNpyRows(proxy/'state.npy', expected_shape=(2900, 9333), expected_dtype=np.float64)
        mrows = BoundedNpyRows(proxy/'memory.npy', expected_shape=(2900, 9333), expected_dtype=np.float64)
        posterior_weights = np.zeros((3, 2900, 2)); neutral = np.zeros((3, 2900), dtype=bool)
        for i, entry in enumerate(data.queries):
            guard(); lo, hi = map(int, data.offsets[i:i+2]); hist_rows = data.prefix[lo:hi]
            swaps = np.array([swap_evidence(entry['match_id'], entry['participant_id'], int(data.times[row])) for row in hist_rows], dtype=bool)
            native = data.native[i]; generated = decisions(native, srows[i], mrows[i], past_likelihoods[lo:hi], swaps)
            native_out.write(np.array([i]), native[None])
            for arm_index, arm in enumerate(ARMS):
                out[arm].write(np.array([i]), generated[arm]['probabilities'][None])
                posterior_weights[arm_index, i] = generated[arm]['weights']; neutral[arm_index, i] = generated[arm]['neutral']
        for value in out.values(): value.close()
        native_out.close(); np.save(output/'posterior_weights.npy', posterior_weights, allow_pickle=False)
        np.save(dist/'neutral_flags.npy', neutral, allow_pickle=False)
        write(dist/'selection.json', [{**e, 'query_ordinal': i} for i, e in enumerate(data.queries)])
        write(dist/'group_main.json', data.catalogue['group_main'])
        require(file_hash(dist/'m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'native float32 prior did not reproduce exactly')
        require(np.all(neutral[:, np.diff(data.offsets) == 0]), 'zero-history reference fallback failed')
        files = {p.name: file_hash(p) for p in dist.iterdir() if p.suffix in ('.npy', '.json')}
        write(output/'predictions_lock.json', {'state': 'all_distributions_frozen_before_scoring', 'files': files,
            'forecast_lock_sha256': file_hash(output/'forecast_lock.json'), 'posterior_weights_sha256': file_hash(output/'posterior_weights.npy')})
        data.revalidate()
        for path, sha in code.items(): require(file_hash(path) == sha, 'consumed proxy code changed')
        require(sum(r['rows'] for r in data.labels.receipts if r['purpose'] == 'past') == 15573
                and sum(r['rows'] for r in data.labels.receipts if r['purpose'] == 'training') == 392448, 'label-access accounting changed')
        manifest = {'schema': 'bayes-proxy-distribution-v1', 'state': 'complete', 'selection_count': 2900, 'groups': 9333,
            'arms': list(ARMS), 'files': files, 'protocol_path': str(PROTOCOL), 'protocol_sha256': PROTOCOL_SHA,
            'reference_manifest_path': data.native_manifest['reference_manifest_path'],
            'reference_manifest_sha256': data.native_manifest['reference_manifest_sha256'],
            'source_context_manifest_path': data.native_manifest['source_context_manifest_path'],
            'source_context_manifest_sha256': data.native_manifest['source_context_manifest_sha256'],
            'prefix_report_sha256': file_hash(PREFIX/'report.json'), 'experiment_directory': str(output),
            'inputs_lock_sha256': file_hash(output/'inputs_lock.json'), 'prediction_lock_sha256': file_hash(output/'predictions_lock.json'),
            'code_sha256': code, 'training_performed': False, 'optimizer_steps': 0,
            'query_fold_current_future_labels_used_for_its_predictions': False, 'allowed_past_evidence_used': True,
            'accuracy_computed': False, 'source_runtime_imported': False, 'test_payload_reads': False, 'confirmation_payload_reads': False}
        write(dist/'manifest.json', manifest)
        write(output/'report.json', {'schema': 'bayes-proxy-forecast-v1', 'state': 'complete', 'query_games': 2900,
            'past_visits': 15573, 'forecast_rows': 18473, 'zero_history_queries': 287, 'training_performed': False,
            'optimizer_steps': 0, 'preflight': preflight, 'banks': bank_receipts, 'empty_retrieval_counts': empty_counts,
            'neutral_counts': {a: int(neutral[i].sum()) for i, a in enumerate(ARMS)},
            'label_access_counts': {'opposite_fold_training': 392448, 'allowed_closed_past': 15573},
            'accuracy_computed': False, 'test_payload_reads': False, 'confirmation_payload_reads': False,
            'source_runtime_imported': False, 'distribution_manifest_sha256': file_hash(dist/'manifest.json'),
            'code_sha256': code, 'elapsed_s': time.monotonic()-start})
        status('complete_waiting_for_native_decoding', query_games=2900)
    finally:
        for value in writers:
            if not value.stream.closed: value.close(require_complete=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
