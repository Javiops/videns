"""Frozen OOF purchase-memory screen; no fitting, outcomes or source imports."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from winprob import admission
from winprob.history_probe_training import HistoryDataset
from winprob.history_probe_models import HistoryProbeActor, set_history_validation
from winprob.iql_dataset_contract import current_reservations
from winprob.iql_training import TensorStore, RuntimeGuard, file_hash, set_below_normal_priority
from winprob.history_probe_evaluation import summarize_logits
from winprob.purchase_memory_v1 import normalized, neighbors, mix, shuffled_labels, compact, evaluate, distribution

PROTOCOL = ROOT/'docs/purchase_memory_protocol_v1.md'
OLD = ROOT/'data/runs/history-probe-20260930-v3/experiment'
CACHE = ROOT/'data/runs/history-cache-20260930-v1/cache'
CHECKPOINTS = ['e050c340298e5495fb966378d33cd133b558cf6302f673db5d8a14188a84364a',
               '5381e72e1606f773817963c317652ecb2d6b568316ae189a63261df4565a1440']
OLD_PREDICTIONS = '27eb094eb1dcfc1551b375446684b0b423d86c03b79077b2ccb5772c8c0968f5'
BATCH = 256


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def embedding(model, store, rows):
    state = model.state(store.states(rows))
    padded = torch.cat((state, torch.zeros(len(rows), 114, device=state.device)), dim=1)
    return model.combined[1](model.combined[0](padded))


def checked_hash(path, want):
    if file_hash(path) != want: raise ValueError('frozen artifact changed: '+str(path))
    return {str(path.resolve()): want}


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    guard = RuntimeGuard(args.deadline_seconds)
    set_below_normal_priority(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    def status(stage, **extra):
        value = {'stage': stage, 'elapsed_s': time.monotonic()-guard.started, **extra}
        (args.output/'status.json').write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        print(json.dumps(value), flush=True)
    try:
        status('validating')
        paths = [Path(__file__), PROTOCOL]+sorted((ROOT/'winprob').glob('*.py'))
        bindings = {str(p.resolve()): file_hash(p) for p in paths}
        bindings.update(checked_hash(CACHE/'manifest.json', '1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b'))
        bindings.update(checked_hash(OLD/'predictions.npz', OLD_PREDICTIONS))
        for fold in (0, 1): bindings.update(checked_hash(OLD/f'actor_state_fold{fold}.pt', CHECKPOINTS[fold]))
        dataset = HistoryDataset(CACHE, ROOT/'docs/history_probe_protocol_v1.md')
        bindings.update(dataset.bindings); guard.bind(bindings); guard()
        a = dataset.arrays; selected = dataset.selected; n = len(selected)
        if n != 3350 or len(np.unique(a['game_index'][selected])) != n: raise ValueError('fixed3350-game selection changed')
        write(args.output/'plan.json', {'protocol_sha256': file_hash(PROTOCOL), 'bindings_sha256': bindings,
            'selection': dataset.selection, 'query_count': n, 'reference': 'OOF state-only, not M2',
            'training_performed': False, 'neighbors_per_query': 32, 'mixture_weight': .5})
        device = torch.device('cuda:0')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
        store = TensorStore(dataset, device, guard, preload_budget_gib=0.)
        if store.preloaded: raise ValueError('streaming recipe cannot preload full corpus')
        labels = np.asarray(a['main_group_id'][selected])
        champs = dataset.focal_champion_indices
        scores = {arm: {'top1': np.full(n, -1, dtype=np.int64), 'top3': np.full((n, 3), -1, dtype=np.int64),
                         'target_log_probability': np.full(n, np.nan)} for arm in ('state', 'memory', 'shuffled')}
        neighbor_rows = np.full((n, 32), -1, dtype=np.int64)
        neighbor_weights = np.zeros((n, 32), dtype=np.float64)
        neighbor_distances = np.full((n, 32), np.nan)
        memory_top3 = np.full((n, 3), -1, dtype=np.int64)
        fold_facts = {}; mutation_rows = 0
        with np.load(OLD/'predictions.npz', allow_pickle=False) as previous:
            if (not np.array_equal(previous['selected_state_indices'], selected)
                    or not np.array_equal(previous['labels'], labels)
                    or not np.array_equal(previous['prediction_fold'], a['fold'][selected])):
                raise ValueError('prior selected prediction identity changed')
            old = {key: np.array(previous['state_'+key]) for key in scores['state']}
        for fold in (0, 1):
            train = np.flatnonzero(a['fold'] != fold)
            positions = np.flatnonzero(a['fold'][selected] == fold); queries = selected[positions]
            allowed = sorted(g['match_id'] for g in dataset.games if g['fold'] != fold)
            checkpoint = torch.load(OLD/f'actor_state_fold{fold}.pt', map_location='cpu', weights_only=True)
            if (checkpoint['arm'] != 'state' or checkpoint['scored_fold'] != fold or checkpoint['epochs'] != 20
                    or sorted(checkpoint['training_game_ids']) != allowed): raise ValueError('encoder training game membership differs')
            model = HistoryProbeActor(dataset.num_champions, dataset.num_items).to(device)
            model.load_state_dict(checkpoint['actor'], strict=True); model.eval(); set_history_validation(model, False)
            del checkpoint
            if set(a['game_index'][train]) & set(a['game_index'][queries]): raise ValueError('OOF game overlap')
            features = np.empty((len(train), 128), dtype=np.float32)
            status('encoding_training_memory', fold=fold, rows=len(train))
            with torch.inference_mode():
                for lo in range(0, len(train), BATCH):
                    guard(); rows = train[lo:lo+BATCH]
                    features[lo:lo+len(rows)] = embedding(model, store, rows).cpu().numpy()
            memory = normalized(features); del features
            q_features = np.empty((len(queries), 128), dtype=np.float32)
            q_base = np.empty((len(queries), 9333), dtype=np.float64)
            status('predicting_queries', fold=fold, queries=len(queries))
            # Inference must be unchanged after ALL held-out action labels change.
            original_main, original_full = a['main_group_id'], a['full_action_id']
            altered_main = np.array(original_main); altered_full = np.array(original_full)
            altered_main[a['fold'] == fold] = (altered_main[a['fold'] == fold]+1)%9333
            altered_full[a['fold'] == fold] = (altered_full[a['fold'] == fold]+1)%13166
            with torch.inference_mode():
                for lo in range(0, len(queries), BATCH):
                    guard(); rows = queries[lo:lo+BATCH]
                    h = embedding(model, store, rows); logits = model.combined[2](h)
                    try:
                        a['main_group_id'], a['full_action_id'] = altered_main, altered_full
                        h_changed = embedding(model, store, rows)
                    finally:
                        a['main_group_id'], a['full_action_id'] = original_main, original_full
                    if not torch.equal(h, h_changed): raise ValueError('query labels changed representation')
                    mutation_rows += len(rows)
                    q_features[lo:lo+len(rows)] = h.cpu().numpy()
                    values = logits.cpu().numpy().astype(np.float64)
                    summary = summarize_logits(values, labels[positions[lo:lo+len(rows)]])
                    for key, value in summary.items(): scores['state'][key][positions[lo:lo+len(rows)]] = value
                    values -= values.max(axis=1, keepdims=True); np.exp(values, out=values); values /= values.sum(axis=1, keepdims=True)
                    q_base[lo:lo+len(rows)] = values
            del altered_main, altered_full
            query_vectors = normalized(q_features); del q_features
            shuffled, shuffle_info = shuffled_labels(a['main_group_id'], train, champs, a['patch_index'], a['timestamp_ms'], fold)
            by_champion = {int(ch): np.flatnonzero(champs[train] == ch) for ch in np.unique(champs[queries])}
            status('retrieving', fold=fold)
            for local, (position, row) in enumerate(zip(positions, queries)):
                if local%64 == 0: guard()
                eligible = by_champion[int(champs[row])]
                ids, distances, weights = neighbors(query_vectors[local], memory[eligible], memory_rows=train[eligible],
                    memory_games=a['game_index'][train[eligible]], query_game=int(a['game_index'][row]))
                if (np.any(a['fold'][ids] == fold) or np.any(champs[ids] != champs[row])
                        or len(np.unique(a['game_index'][ids])) != len(ids)):
                    raise ValueError('neighbor fold/champion/game contract failed')
                neighbor_rows[position, :len(ids)] = ids
                neighbor_weights[position, :len(ids)] = weights
                neighbor_distances[position, :len(ids)] = distances
                real_labels = np.asarray(a['main_group_id'][ids])
                for arm, neighbor_labels in (('memory', real_labels), ('shuffled', shuffled[ids])):
                    prediction = mix(q_base[local], neighbor_labels, weights)
                    summary = compact(prediction[None, :], labels[position:position+1])
                    for key, value in summary.items(): scores[arm][key][position] = value[0]
                if len(ids):
                    r = distribution(real_labels, weights, 9333)
                    support = np.flatnonzero(r > 0); order = support[np.argsort(-r[support], kind='stable')[:3]]
                    memory_top3[position, :len(order)] = order
            fold_facts[str(fold)] = {'training_rows': len(train), 'training_games': len(allowed), 'query_games': len(queries),
                                    'shuffle': shuffle_info, 'training_ids_verified_against_checkpoint': True}
            del model, memory, q_base, query_vectors, shuffled; gc.collect(); torch.cuda.empty_cache()
        parity = {'top1_mismatches': int(np.sum(scores['state']['top1'] != old['top1'])),
                  'top3_mismatches': int(np.sum(scores['state']['top3'] != old['top3'])),
                  'maximum_absolute_target_logp_error': float(np.max(np.abs(scores['state']['target_log_probability']-old['target_log_probability'])))}
        write(args.output/'reference_parity.json', parity)
        if parity['top1_mismatches'] or parity['top3_mismatches'] or parity['maximum_absolute_target_logp_error'] > 1e-5:
            raise ValueError('frozen state baseline reproduction failed')
        if mutation_rows != n: raise ValueError('mutation coverage incomplete')
        arrays = {'selected_state_indices': selected, 'labels': labels, 'fold': a['fold'][selected],
                  'game_index': a['game_index'][selected], 'neighbor_rows': neighbor_rows, 'neighbor_weights': neighbor_weights,
                  'neighbor_distances': neighbor_distances, 'memory_only_top3': memory_top3}
        arrays.update({arm+'_'+key: value for arm, summary in scores.items() for key, value in summary.items()})
        np.savez_compressed(args.output/'predictions.npz', **arrays)
        for path, want in bindings.items():
            guard()
            if file_hash(path) != want: raise ValueError('input/code changed before publication: '+path)
        admission.verify_manifest_sources(dataset.admission)
        if {g['match_id'] for g in dataset.games} & current_reservations(dataset.admission): raise ValueError('FIT game became reserved')
        status('evaluating_frozen_predictions')
        result = evaluate(labels, scores)
        result.update(schema='purchase-memory-probe-v1', state='complete', scope='exposed internal OOF mechanism screen; no legality decoder',
            reference='state-only history-probe actor; NOT M2', ledger_games=4001, unavailable_focal_games=651,
            reference_parity=parity, heldout_target_mutation_rows=mutation_rows, fold_facts=fold_facts,
            empty_memory_queries=int(np.sum(neighbor_rows[:, 0] < 0)),
            pure_memory_top1=float(np.mean(memory_top3[:, 0] == labels)),
            pure_memory_top3=float(np.mean((memory_top3 == labels[:, None]).any(axis=1))),
            predictions_sha256=file_hash(args.output/'predictions.npz'), bindings_sha256=bindings,
            training_performed=False, optimizer_steps=0, test_payload_reads=False, confirmation_payload_reads=False,
            outcomes_analyzed=False, source_code_imported=False, displayed_policy_evaluated=False,
            breakthrough_established=False, promotion_authorized=False, elapsed_s=time.monotonic()-guard.started,
            host_memory=guard.memory_report(), peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            runtime={'torch': str(torch.__version__), 'numpy': np.__version__, 'device': torch.cuda.get_device_name(device)})
        snapshot = args.output/'code_snapshot'; snapshot.mkdir()
        for path in paths:
            dest = snapshot/path.resolve().relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        write(args.output/'report.json', result)
        status('complete', report=str(args.output/'report.json'), metric_gate_passed=result['metric_gate_passed'])
    except Exception as error:
        status('failed', error=str(error), error_type=type(error).__name__); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline-seconds', default=1200, type=int)
    run(parser.parse_args())
