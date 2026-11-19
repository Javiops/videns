"""Fixed full-basket IQL/BC experiment. Real fitting requires --execute after review."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from winprob import admission
from winprob.iql_features import PublicFeatureCompiler
from winprob.iql_kernels import normalize_oof_actor_weights, assert_out_of_game_predictions
from winprob.iql_models import (PublicBoardBatch, FullActionBatch, FullCategoricalActor, TwinFullBasketCritic,
    StateValue, make_target, critic_value_step, categorical_actor_loss, collate_public_boards,
    set_input_content_validation)
from winprob.iql_training import (Dataset, TensorStore, RuntimeGuard, SEED, BATCH_SIZE, adam,
    clone_actor_optimizer, epoch_order, file_hash, tree_hash, write_json, main_group_log_odds, set_below_normal_priority)
from winprob.preference_reference import load_reference
from winprob.preference_evaluation import evaluate_preferences


def read_statics(path):
    path = Path(path).resolve(strict=True)
    if path.with_suffix('.sha256').read_text().split() != [file_hash(path), path.name]:
        raise ValueError('static export digest mismatch')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != 'winprob-boots-static-v1': raise ValueError('unknown static export')
    bindings = {str(path): file_hash(path), str(path.with_suffix('.sha256')): file_hash(path.with_suffix('.sha256'))}
    for entry in payload['patches'].values():
        for source in entry['source_files']:
            if file_hash(source['path']) != source['sha256']: raise ValueError('bound public static source changed')
            bindings[str(Path(source['path']).resolve())] = source['sha256']
    return payload['patches'], bindings


def checkpoint(path, payload):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, tmp); tmp.replace(path)
    return file_hash(path)


def enable_validated_cache_fastpath(models, store):
    """Skip repeated device-to-host content checks only after full cache proof."""
    dataset = store.dataset
    if (dataset.manifest.get('training_authorized') is not True or dataset.manifest.get('smoke') is not False
            or dataset.manifest.get('gates', {}).get('passed') is not True
            or dataset.audit.get('terminal_return_once') is not True or dataset.audit.get('states') != dataset.n):
        raise ValueError('trusted-input execution requires the complete verified training cache')
    for model in models:
        set_input_content_validation(model, False)


def prediction_weights(online_target, value, store, indices, guard):
    """Frozen observed-action predictions only; unsupported rows are omitted by caller."""
    q1, q2, v = [], [], []
    online_target.eval(); value.eval()
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            if start % (BATCH_SIZE*20) == 0: guard()
            rows = indices[start:start+BATCH_SIZE]
            states, actions = store.states(rows), store.actions(rows)
            a, b = online_target(states, actions)
            q1.append(a.cpu().numpy()); q2.append(b.cpu().numpy()); v.append(value(states).cpu().numpy())
    return tuple(np.concatenate(part) for part in (q1, q2, v))


def actor_epoch(actor, optimizer, store, order, advantage_weights, guard):
    actor.train(); total, maximum_gradient = 0., 0.
    for start in range(0, len(order), BATCH_SIZE):
        if start % (BATCH_SIZE*20) == 0: guard()
        rows = order[start:start+BATCH_SIZE]
        state = store.states(rows); labels = store.labels(rows)
        weight = (torch.ones(len(rows), device=store.device) if advantage_weights is None else
                  advantage_weights[torch.as_tensor(rows, dtype=torch.long, device=store.device)])
        optimizer.zero_grad(set_to_none=True)
        loss = categorical_actor_loss(actor(state), labels, weight)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach())*len(rows); maximum_gradient = max(maximum_gradient, float(norm))
    return {'mean_objective': total/len(order), 'transitions': len(order), 'maximum_gradient_norm_before_clip': maximum_gradient}


def fit_group_frequency(dataset):
    """Ordinary FIT retained-main-action frequency, not a revision-label frequency."""
    cells, global_counts = defaultdict(lambda: defaultdict(Counter)), Counter()
    a = dataset.arrays
    for champion, patch, group in zip(dataset.focal_champion_indices, a['patch_index'], a['main_group_id'], strict=True):
        cells[str(int(champion))][str(int(patch))][str(int(group))] += 1
        global_counts[str(int(group))] += 1
    return {'schema': 'iql-ordinary-fit-main-frequency-v1', 'smoothing': .5,
            'population': 'uniform complete eligible FIT transitions; full actions projected to exact main groups',
            'by_champion_patch': {c: {p: dict(v) for p, v in d.items()} for c, d in cells.items()},
            'global_counts': dict(global_counts)}


def group_frequency_odds(model, compiled, rows):
    result, fallbacks = [], Counter()
    for state, row in zip(compiled, rows, strict=True):
        cells = model['by_champion_patch'].get(str(int(state['champion_indices'][0])), {})
        requested = str({'16.17': 0, '16.18': 1, '16.19': 2}[row['patch']])
        if requested in cells: counts, mode = cells[requested], 'same_native_patch_champion'
        elif cells: counts, mode = cells[max(cells, key=int)], 'latest_fit_patch_champion'
        else: counts, mode = model['global_counts'], 'global_fit'
        result.append(np.log(counts.get(str(row['group_b']), 0)+.5)-np.log(counts.get(str(row['group_a']), 0)+.5))
        fallbacks[mode] += 1
    return np.asarray(result), dict(fallbacks)


def prepare_challenge(reference_path, dataset, compiler):
    rows, probabilities, groups, usable, source_audit = load_reference(reference_path, dataset.admission_path)
    if groups != dataset.catalogue['group_main']: raise ValueError('challenge and actor main-group order differ')
    selected, compiled, p_rows, a_full, b_full, excluded = [], [], [], [], [], []
    for index, row in enumerate(rows):
        if row['role'] != 'development': continue
        if not usable[index]: excluded.append({'pair_id': row['pair_id'], 'reason': 'reference_action_unsupported'}); continue
        if row['patch'] != '16.19': raise ValueError('challenge must retain the fixed untrained patch19 population')
        try:
            state = compiler.state(row['source_projection'], row['patch'])
            left = dataset.action_lookup[('retained_net_purchase', ((row['action_a'], 1),))]
            right = dataset.action_lookup[('retained_net_purchase', ((row['action_b'], 1),))]
            if not dataset.arrays['action_static_valid'][2, [left, right]].all():
                raise ValueError('challenge action lacks native attributes')
        except (ValueError, KeyError) as error:
            excluded.append({'pair_id': row['pair_id'], 'reason': str(error)}); continue
        selected.append(row); compiled.append(state); p_rows.append(probabilities[index]); a_full.append(left); b_full.append(right)
    if len(selected) < 100 or len({row['match_id'] for row in selected}) < 50:
        raise ValueError('fixed challenge under minimum after common feature/action exclusions')
    fit_ids = {row['match_id'] for row in dataset.games}
    if fit_ids & {row['match_id'] for row in selected}: raise ValueError('preference challenge overlaps IQL FIT')
    return {'rows': selected, 'compiled': compiled, 'reference': np.asarray(p_rows),
            'full_a': np.asarray(a_full), 'full_b': np.asarray(b_full), 'exclusions': excluded,
            'source_audit': source_audit}


@torch.no_grad()
def evaluate_actor(actor, challenge, full_to_group, device, guard):
    set_input_content_validation(actor, True)
    actor.eval(); main, full, entropy = [], [], []
    for start in range(0, len(challenge['rows']), BATCH_SIZE):
        guard(); end = min(start+BATCH_SIZE, len(challenge['rows']))
        rows = challenge['rows'][start:end]
        logits = actor(collate_public_boards(challenge['compiled'][start:end], device))
        main.append(main_group_log_odds(logits, [r['group_a'] for r in rows], [r['group_b'] for r in rows], full_to_group).cpu().numpy())
        a = torch.as_tensor(challenge['full_a'][start:end], device=device)
        b = torch.as_tensor(challenge['full_b'][start:end], device=device)
        local = torch.arange(end-start, device=device)
        full.append((logits[local, b]-logits[local, a]).cpu().numpy())
        logp = torch.log_softmax(logits, dim=1)
        entropy.append(-(logp.exp()*logp).sum(dim=1).cpu().numpy())
    return np.concatenate(main), np.concatenate(full), np.concatenate(entropy)


def synthetic_preflight(dataset, device, store, guard):
    """Measure real architecture on synthetic tensors, not on actual FIT labels."""
    torch.manual_seed(SEED+9000)
    guard.sample_memory('before_synthetic_models_and_optimizers')
    n = BATCH_SIZE
    champion = torch.arange(1, 11, device=device).repeat(n, 1)
    width = max(1, dataset.inventory_max)
    item = torch.arange(1, width+1, dtype=torch.long, device=device).view(1, 1, -1).expand(n, 10, -1).clone()
    count = torch.ones(n, 10, width, device=device)
    ally = torch.tensor([False]+[True]*4+[False]*5, device=device).expand(n, -1)
    enemy = torch.tensor([False]*5+[True]*5, device=device).expand(n, -1)
    state = PublicBoardBatch(champion, item, count, torch.zeros(n, 10, 22, device=device),
                            torch.full((n, 10, 1), .5, device=device), ally, enemy, torch.zeros(n, 37, device=device))
    action_lengths = np.diff(dataset.arrays['action_offsets'])
    valid = np.asarray(dataset.arrays['action_static_valid'][1])
    action_id = int(np.argmax(np.where(valid, action_lengths, -1)))
    action = FullActionBatch(store.action_indices[action_id:action_id+1].expand(n, -1),
                            store.action_counts[action_id:action_id+1].expand(n, -1),
                            store.action_static[1, action_id:action_id+1].expand(n, -1))
    done = torch.zeros(n, dtype=torch.bool, device=device); done[::10] = True
    rewards = torch.zeros(n, device=device); rewards[done] = (torch.arange(int(done.sum()), device=device)%2).float()
    next_states = state.select(torch.nonzero(~done, as_tuple=False).flatten())
    critic = TwinFullBasketCritic(dataset.num_champions, dataset.num_items).to(device)
    target = make_target(critic); value = StateValue(dataset.num_champions, dataset.num_items).to(device)
    actor = FullCategoricalActor(dataset.num_champions, dataset.num_items).to(device)
    qopt, vopt, aopt = adam(critic.parameters()), adam(value.parameters()), adam(actor.parameters())
    # The generated benchmark tensors are checked through the strict path
    # once. Real cached tensors have already passed the independent Dataset
    # proof. These forward calls change neither weights nor optimizer state.
    with torch.no_grad():
        critic(state, action); target(state, action); value(state); value(next_states); actor(state)
    enable_validated_cache_fastpath((critic, target, value, actor), store)
    guard.sample_memory('after_models_and_optimizers_before_warmup')
    q_times, a_times = [], []
    for iteration in range(30):
        guard()
        torch.cuda.synchronize(device); begin = time.monotonic()
        critic_value_step(critic, target, value, vopt, qopt, state, action, next_states, rewards, done)
        torch.cuda.synchronize(device); q_elapsed = time.monotonic()-begin
        begin = time.monotonic(); aopt.zero_grad(set_to_none=True)
        loss = categorical_actor_loss(actor(state), torch.full((n,), action_id, dtype=torch.long, device=device), torch.ones(n, device=device))
        loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 10., error_if_nonfinite=True); aopt.step()
        torch.cuda.synchronize(device)
        if iteration >= 10: q_times.append(q_elapsed); a_times.append(time.monotonic()-begin)
        if iteration == 9: guard.sample_memory('after_ten_optimizer_warmup_steps')
    # Measure the actual selected cache-gather path, including packed next states.
    # No real gathered state or label is fed to an optimizer during preflight.
    rng = np.random.default_rng(SEED+9001); gather_times = []
    for _ in range(20):
        rows = rng.choice(dataset.n, min(BATCH_SIZE, dataset.n), replace=False)
        torch.cuda.synchronize(device); begin = time.monotonic()
        gathered_state, gathered_action = store.states(rows), store.actions(rows)
        following = np.asarray(dataset.arrays['next_state_index'][rows]); following = following[following >= 0]
        gathered_next = store.states(following) if len(following) else None
        torch.cuda.synchronize(device); gather_times.append(time.monotonic()-begin)
        del gathered_state, gathered_action, gathered_next
    critic_steps = sum(20*int(np.ceil(row['train_transitions']/BATCH_SIZE)) for row in dataset.audit['folds'].values())
    actor_steps = 30*int(np.ceil(dataset.n/BATCH_SIZE))
    oof_batches = int(np.ceil(dataset.n/BATCH_SIZE))
    gather_seconds = float(np.mean(gather_times))*(critic_steps+actor_steps+oof_batches)
    compute_seconds = float(np.mean(q_times))*critic_steps+float(np.mean(a_times))*actor_steps
    oof_seconds = .6*float(np.mean(q_times))*oof_batches
    fixed_allowance = 300.
    projected = 1.5*(compute_seconds+gather_seconds+oof_seconds)+fixed_allowance
    result = {'synthetic_only': True, 'batch_size': n, 'critic_step_median_s': float(np.median(q_times)),
              'actor_step_median_s': float(np.median(a_times)), 'critic_step_mean_s': float(np.mean(q_times)),
              'actor_step_mean_s': float(np.mean(a_times)), 'cache_gather_mean_s': float(np.mean(gather_times)),
              'timed_steps': 20, 'warmup_steps': 10, 'inventory_width': width,
              'action_width': int(store.action_indices.shape[1]), 'nonterminal_rows': int((~done).sum()),
              'cache_preloaded': store.preloaded, 'critic_updates_fixed_recipe': critic_steps,
              'validated_input_content_fastpath': True,
              'actor_updates_fixed_recipe': actor_steps, 'projection_with_1_5_factor_s': float(projected),
              'projected_raw_compute_s': compute_seconds, 'projected_cache_gather_s': gather_seconds,
              'projected_oof_inference_s': oof_seconds, 'checkpoint_evaluation_integrity_allowance_s': fixed_allowance,
              'projection_limitations': 'measured synthetic optimization and real gather timing; 1.5factor plus300s allowance; not an ETA guarantee',
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device)}
    del actor, critic, target, value, qopt, vopt, aopt, loss
    del state, next_states, action, champion, item, count, ally, enemy, rewards, done
    gc.collect(); torch.cuda.empty_cache()
    guard.sample_memory('after_synthetic_models_released')
    result['host_memory'] = guard.memory_report()
    result['host_feature_release'] = store.host_release_report
    return result


def run(args):
    guard = RuntimeGuard(args.deadline_seconds)
    set_below_normal_priority(); torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    def status(stage, **extra):
        write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-guard.started, **extra})
    def log(entry):
        with (args.output/'epochs.jsonl').open('a', encoding='utf-8') as handle: handle.write(json.dumps(entry, allow_nan=False)+'\n')
    try:
        status('validating_fit_cache_and_challenge')
        if not torch.cuda.is_available(): raise RuntimeError('own-project CUDA runtime required; refusing CPU training')
        device = torch.device(args.device)
        if device.type != 'cuda': raise ValueError('real training device must be CUDA')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        dataset = Dataset(args.dataset, args.catalogue)
        if dataset.manifest.get('training_authorized') is not True or dataset.manifest.get('smoke') is not False:
            raise ValueError('cache is not authorized for the real fixed training recipe')
        static_path = args.static_export or Path(dataset.manifest['static_export_path'])
        if dataset.manifest.get('static_export_sha256') and file_hash(static_path) != dataset.manifest['static_export_sha256']:
            raise ValueError('challenge statics differ from dataset statics')
        static, static_bindings = read_statics(static_path)
        compiler = PublicFeatureCompiler(static)
        for name in ('champion_index', 'item_index'):
            if getattr(compiler, name) != {int(k): int(v) for k, v in dataset.feature_metadata[name].items()}:
                raise ValueError('challenge/cache fixed public vocabulary differs')
        challenge = prepare_challenge(args.reference, dataset, compiler)
        bindings = {**dataset.bindings, **static_bindings}
        reference_manifest = json.loads((args.reference/'manifest.json').read_text(encoding='utf-8'))
        for name, expected in reference_manifest['files'].items(): bindings[str((args.reference/name).resolve())] = expected
        for path in (args.reference/'manifest.json', args.reference/'manifest.sha256', Path(__file__).resolve(),
                     ROOT/'winprob/iql_training.py', ROOT/'winprob/iql_models.py', ROOT/'winprob/iql_kernels.py',
                     ROOT/'winprob/iql_features.py', ROOT/'winprob/full_action_catalogue.py', ROOT/'winprob/trajectory_contract.py',
                     ROOT/'winprob/iql_dataset_contract.py',
                     ROOT/'winprob/preference_reference.py', ROOT/'winprob/preference_evaluation.py', ROOT/'winprob/admission.py',
                     ROOT/'docs/iql_experiment_protocol_v1.md'):
            bindings[str(path.resolve())] = file_hash(path)
        guard.bind(bindings); guard()
        write_json(args.output/'input_audit.json', {'dataset': dataset.audit, 'challenge_source': challenge['source_audit'],
            'challenge_pairs': len(challenge['rows']), 'challenge_games': len({r['match_id'] for r in challenge['rows']}),
            'challenge_exclusions': challenge['exclusions'], 'bindings_sha256': bindings})
        free, _ = torch.cuda.mem_get_info(device)
        if free < 2*1024**3: raise RuntimeError('insufficient idle CUDA headroom, including one-GiB free reserve')
        status('loading_bounded_cuda_states')
        guard.sample_memory('before_cuda_state_preload')
        store = TensorStore(dataset, device, guard, preload_budget_gib=6.)
        guard.sample_memory('after_cuda_state_preload_and_host_release')
        status('synthetic_gpu_preflight')
        preflight = synthetic_preflight(dataset, device, store, guard)
        write_json(args.output/'preflight.json', preflight)
        if preflight['projection_with_1_5_factor_s'] > .85*(args.deadline_seconds-(time.monotonic()-guard.started)):
            raise RuntimeError('fixed complete recipe does not fit conservative remaining-time projection')
        if not args.execute:
            status('preflight_completed_no_real_training', preflight=str((args.output/'preflight.json').resolve()))
            return {'training_performed': False, 'preflight': preflight}
        a = dataset.arrays; n = dataset.n
        initial_hashes, order_hashes, checkpoints = {}, {}, {}
        q1_oof, q2_oof, v_oof = (np.full(n, np.nan, dtype=np.float32) for _ in range(3))
        supported = np.zeros(n, dtype=bool)
        for heldout in (0, 1):
            train = np.flatnonzero(a['fold'] != heldout); score = np.flatnonzero(a['fold'] == heldout)
            train_games = {dataset.games[int(i)]['match_id'] for i in np.unique(a['game_index'][train])}
            score_games = [dataset.games[int(i)]['match_id'] for i in np.unique(a['game_index'][score])]
            assert_out_of_game_predictions(score_games, [str(heldout)]*len(score_games), {str(heldout): train_games})
            seed = SEED+100+heldout; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            critic = TwinFullBasketCritic(dataset.num_champions, dataset.num_items).to(device)
            target = make_target(critic); value = StateValue(dataset.num_champions, dataset.num_items).to(device)
            enable_validated_cache_fastpath((critic, target, value), store)
            qopt, vopt = adam(critic.parameters()), adam(value.parameters())
            initial_hashes[f'critic_for_fold{heldout}'] = tree_hash(critic.state_dict())
            initial_hashes[f'value_for_fold{heldout}'] = tree_hash(value.state_dict())
            for epoch in range(20):
                guard(); status('training_critic_fold', scored_fold=heldout, epoch=epoch+1, required_epochs=20)
                order = epoch_order(train, 'critic', epoch, heldout)
                order_hashes[f'critic{heldout}:{epoch}'] = hashlib.sha256(order.tobytes()).hexdigest()
                losses = Counter(); begin = time.monotonic()
                for start in range(0, len(order), BATCH_SIZE):
                    if start % (BATCH_SIZE*20) == 0: guard()
                    rows = order[start:start+BATCH_SIZE]
                    done_np = np.asarray(a['terminated'][rows]); nxt = np.asarray(a['next_state_index'][rows])[~done_np]
                    stats = critic_value_step(critic, target, value, vopt, qopt, store.states(rows), store.actions(rows),
                        store.states(nxt) if len(nxt) else None,
                        torch.as_tensor(np.array(a['reward'][rows]), device=device),
                        torch.as_tensor(np.array(done_np), dtype=torch.bool, device=device))
                    losses['value_loss'] += stats['value_loss']*len(rows); losses['critic_loss'] += stats['critic_loss']*len(rows)
                log({'stage': 'critic', 'score_fold': heldout, 'epoch': epoch+1, 'transitions': len(train),
                     'value_loss': losses['value_loss']/len(train), 'critic_loss': losses['critic_loss']/len(train),
                     'elapsed_s': time.monotonic()-begin})
            path = args.output/f'critic_for_fold{heldout}.pt'
            checkpoints[path.name] = checkpoint(path, {'online': critic.state_dict(), 'target': target.state_dict(), 'value': value.state_dict(),
                'critic_optimizer': qopt.state_dict(), 'value_optimizer': vopt.state_dict(), 'epochs': 20, 'seed': seed,
                'training_game_ids': sorted(train_games), 'scored_game_ids': sorted(score_games)})
            permitted = np.isin(a['full_action_id'][score], np.unique(a['full_action_id'][train]))
            supported[score] = permitted
            rows = score[permitted]
            q1_oof[rows], q2_oof[rows], v_oof[rows] = prediction_weights(target, value, store, rows, guard)
            del critic, target, value, qopt, vopt
            gc.collect(); torch.cuda.empty_cache()
        weight_data = normalize_oof_actor_weights(q1_oof, q2_oof, v_oof, supported)
        if weight_data['effective_sample_size'] < .1*n: raise ValueError('OOF actor weights fail the 10% effective-count gate')
        weight_path = args.output/'oof_weights.npz'
        np.savez_compressed(weight_path, q1=q1_oof, q2=q2_oof, value=v_oof, supported=supported,
                            raw=weight_data['raw_weights'], weights=weight_data['weights'], game_index=np.asarray(a['game_index']),
                            fold=np.asarray(a['fold']), full_action_id=np.asarray(a['full_action_id']))
        weight_report = {k: v for k, v in weight_data.items() if k not in ('raw_weights', 'weights')}
        weight_report['normalized_weight_quantiles'] = np.quantile(weight_data['weights'], [0, .01, .5, .99, 1]).tolist()
        weight_report['artifact_sha256'] = file_hash(weight_path)
        write_json(args.output/'oof_weight_report.json', weight_report)
        actor_seed = SEED+1000; torch.manual_seed(actor_seed); torch.cuda.manual_seed_all(actor_seed)
        actor = FullCategoricalActor(dataset.num_champions, dataset.num_items).to(device)
        enable_validated_cache_fastpath((actor,), store)
        optimizer = adam(actor.parameters()); initial_hashes['actor'] = tree_hash(actor.state_dict())
        all_rows = np.arange(n)
        for epoch in range(10):
            guard(); status('warming_uniform_actor', epoch=epoch+1, required_epochs=10)
            order = epoch_order(all_rows, 'actor_warm', epoch)
            order_hashes[f'warm:{epoch}'] = hashlib.sha256(order.tobytes()).hexdigest()
            stats = actor_epoch(actor, optimizer, store, order, None, guard)
            log({'stage': 'warm_bc', 'epoch': epoch+1, **stats})
        warm_path = args.output/'actor_warm10.pt'
        checkpoints[warm_path.name] = checkpoint(warm_path, {'actor': actor.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': 10, 'seed': actor_seed})
        warm_model_hash, warm_optimizer_hash = tree_hash(actor.state_dict()), tree_hash(optimizer.state_dict())
        bc20, bc_opt = clone_actor_optimizer(actor, optimizer)
        iql, iql_opt = clone_actor_optimizer(actor, optimizer)
        del actor, optimizer
        gpu_weights = torch.as_tensor(weight_data['weights'].astype(np.float32), device=device)
        continuation_starts = {'bc20_model': tree_hash(bc20.state_dict()), 'iql_model': tree_hash(iql.state_dict()),
                               'bc20_optimizer': tree_hash(bc_opt.state_dict()), 'iql_optimizer': tree_hash(iql_opt.state_dict())}
        if continuation_starts['bc20_model'] != warm_model_hash or continuation_starts['iql_model'] != warm_model_hash:
            raise ValueError('actor warmstart mismatch')
        if continuation_starts['bc20_optimizer'] != warm_optimizer_hash or continuation_starts['iql_optimizer'] != warm_optimizer_hash:
            raise ValueError('actor optimizer warmstart mismatch')
        for epoch in range(10):
            guard(); status('matched_actor_continuations', epoch=epoch+1, required_epochs=10)
            order = epoch_order(all_rows, 'actor_continuation', epoch)
            order_hashes[f'continuation_both:{epoch}'] = hashlib.sha256(order.tobytes()).hexdigest()
            # Both arms consume the exact same persisted permutation for this epoch.
            bc_stats = actor_epoch(bc20, bc_opt, store, order, None, guard)
            iql_stats = actor_epoch(iql, iql_opt, store, order, gpu_weights, guard)
            log({'stage': 'matched_continuation', 'epoch': epoch+1, 'bc20': bc_stats, 'iql': iql_stats})
        for name, model, optim in (('bc20', bc20, bc_opt), ('iql', iql, iql_opt)):
            checkpoints['actor_'+name+'.pt'] = checkpoint(args.output/('actor_'+name+'.pt'),
                {'actor': model.state_dict(), 'optimizer': optim.state_dict(), 'epochs': 20, 'continuation_epochs': 10, 'seed': actor_seed})
        write_json(args.output/'training_lock.json', {'checkpoints': checkpoints, 'initial_state_hashes': initial_hashes,
            'continuation_initial_hashes': continuation_starts, 'warm_model_hash': warm_model_hash,
            'warm_optimizer_hash': warm_optimizer_hash, 'batch_order_hashes': order_hashes,
            'actor_epochs_complete': {'warm10': 10, 'bc20': 20, 'iql': 20}, 'critic_epochs_complete': [20, 20],
            'preference_metrics_inspected': False, 'bindings_sha256': bindings})
        guard(); status('evaluating_frozen_cross_signal_challenge')
        warm = FullCategoricalActor(dataset.num_champions, dataset.num_items).to(device)
        warm.load_state_dict(torch.load(warm_path, map_location=device, weights_only=True)['actor'])
        main_odds, full_odds, entropies = {}, {}, {}
        for name, model in (('warm10', warm), ('bc20', bc20), ('iql', iql)):
            main_odds[name], full_odds[name], entropies[name] = evaluate_actor(model, challenge, store.full_to_group, device, guard)
        rows = challenge['rows']; weights = np.asarray([row['random_focal_pair_weight'] for row in rows])
        ga = np.asarray([row['group_a'] for row in rows]); gb = np.asarray([row['group_b'] for row in rows]); pp = challenge['reference']
        main_odds['m2'] = np.log(pp[np.arange(len(rows)), gb].astype(float))-np.log(pp[np.arange(len(rows)), ga].astype(float))
        price_delta = np.asarray(a['action_static'][2, challenge['full_b'], 0]-a['action_static'][2, challenge['full_a'], 0])
        main_odds['price_cheaper'], main_odds['price_dearer'] = -price_delta, price_delta
        frequency = fit_group_frequency(dataset)
        main_odds['ordinary_fit_frequency'], frequency_fallback = group_frequency_odds(frequency, challenge['compiled'], rows)
        write_json(args.output/'ordinary_fit_frequency.json', frequency)
        inputs = ([row['match_id'] for row in rows], weights, [row['action_a'] for row in rows], [row['action_b'] for row in rows])
        primary = evaluate_preferences(*inputs, main_odds, candidate='iql', baselines=('bc20', 'm2'))
        extra = evaluate_preferences(*inputs, main_odds, candidate='iql', baselines=('warm10', 'price_cheaper', 'price_dearer', 'ordinary_fit_frequency'), minimum_gain=0.)
        full = evaluate_preferences(*inputs, full_odds, candidate='iql', baselines=('bc20', 'warm10'))
        diagnostics = {}
        norm_weight = weights/weights.sum()
        for name in ('warm10', 'bc20', 'iql'):
            diagnostics[name] = {'weighted_actor_full_entropy': float(norm_weight@entropies[name]),
                'weighted_main_vs_exact_w0_ranking_disagreement': float(norm_weight@(np.sign(main_odds[name]) != np.sign(full_odds[name])))}
        slices = {}
        for name, mask in (('B_cheaper', price_delta < 0), ('B_dearer', price_delta > 0), ('equal_price', price_delta == 0)):
            ids = np.asarray(inputs[0])[mask]; counts = {'pairs': int(mask.sum()), 'games': len(set(ids))}
            if counts['games'] < 30: slices[name] = {**counts, 'status': 'not_measured_below_30_games'}
            else: slices[name] = {**counts, 'scope': 'exploratory diagnostic', 'metrics': evaluate_preferences(ids, weights[mask],
                np.asarray(inputs[2])[mask], np.asarray(inputs[3])[mask], {arm: value[mask] for arm, value in main_odds.items()},
                candidate='iql', baselines=('bc20', 'm2'))}
        np.savez_compressed(args.output/'challenge_predictions.npz', pair_id=np.asarray([row['pair_id'] for row in rows]),
            game=np.asarray(inputs[0]), weights=weights, action_a=np.asarray(inputs[2]), action_b=np.asarray(inputs[3]),
            **{name+'_main_log_odds': value for name, value in main_odds.items()},
            **{name+'_full_w0_log_odds': value for name, value in full_odds.items()})
        guard()
        for path, expected in bindings.items():
            if file_hash(path) != expected: raise RuntimeError('bound artifact changed during training')
        admission.verify_manifest_sources(dataset.admission)
        report = {'schema': 'iql-full-basket-experiment-v1', 'stage': 'already_exposed_development_cross_signal',
            'training_performed': True, 'dataset_audit': dataset.audit, 'bindings_sha256': bindings,
            'preflight': preflight, 'cuda_state_preloaded': store.preloaded, 'estimated_state_bytes': store.estimated_state_bytes,
            'validated_cache_input_content_fastpath': True, 'challenge_input_content_validation': True,
            'oof_weights': weight_report, 'checkpoints': checkpoints, 'primary_main_comparison': primary,
            'warm_and_cheap_controls': extra, 'full_w0_vs_matched_actors_only': full,
            'full_reference_comparison': 'not_measured_no_joint_reference_supplied; M2 group masses not substituted',
            'actor_diagnostics': diagnostics, 'price_direction_slices': slices, 'frequency_fallback_counts': frequency_fallback,
            'frequency_control_scope': frequency['population'], 'challenge_exclusions': challenge['exclusions'],
            'cross_signal_gate_passed': bool(primary['metric_gate_passed'] and extra['metric_gate_passed']),
            'causal_claim': False, 'win_benefit_established': False, 'serving_promotion': False,
            'runtime_versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device)},
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'host_memory': guard.memory_report(), 'host_feature_release': store.host_release_report,
            'elapsed_s': time.monotonic()-guard.started, 'report_path': str((args.output/'report.json').resolve())}
        write_json(args.output/'report.json', report)
        status('completed', report=report['report_path'])
        return report
    except Exception as exc:
        guard.sample_memory('failure')
        status('failed_no_valid_complete_comparison', error_type=type(exc).__name__, error=str(exc), host_memory=guard.memory_report())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--static-export', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--deadline-seconds', type=int, default=5400)
    parser.add_argument('--execute', action='store_true', help='Execute the reviewed complete fixed recipe; otherwise synthetic preflight only')
    args = parser.parse_args()
    if not 0 < args.deadline_seconds <= 5400: parser.error('deadline must be at most the fixed90minutes')
    for name in ('dataset', 'catalogue', 'reference', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.static_export: args.static_export = args.static_export.resolve()
    report = run(args)
    print(json.dumps({'training_performed': report['training_performed'], 'report': report.get('report_path'),
                      'cross_signal_gate_passed': report.get('cross_signal_gate_passed', False)}, indent=2))


if __name__ == '__main__': main()
