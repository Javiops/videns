"""Bounded synthetic M2/recipe-receipt probe. No match/target/SQLite reads.

New source-owned adapter only; no consumer runtime imports or frozen edits.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import export_bc20_ordinary_reference_v2 as reference
producer = reference.producer
require = reference.require
PROTOCOL = ROOT.parent/'winprob/docs/recipe_transition_design_v1.md'


def replay_candidate(row, action, dragon):
    """Run the unchanged native search, then retain every native inventory step."""
    from app.conditional_policy import footwear_arguments, purchase_sequence
    from app.shop_econ import try_purchase
    before = list(row['inventory']); wallet = float(row['gold_est'])
    seq = purchase_sequence(tuple(action), row, wallet, dragon) if action else []
    require(seq is not None, 'candidate must be source-native feasible')
    target = Counter(action); pending = target.copy(); inv = Counter(before); steps = []
    footwear = footwear_arguments(row, dragon, required_version=dragon.footwear_state_version)
    for native in seq:
        iid = native['item_id']
        result = try_purchase(iid, list(inv.elements()), wallet, dragon,
            role=row.get('role') or '', bot_quest_complete=row['bot_quest_complete'],
            unmodeled_regular_slots=row['unmodeled_regular_slots'],
            role_slot_inventory_unknown=row['role_slot_inventory_unknown'],
            protected=target-pending, **footwear)
        require(result is not None, 'native replay diverged from search')
        cost, after = result
        require(abs(cost-native['cost']) < 1e-8, 'native replay cost changed')
        steps.append({'item_id': iid, 'cost': float(cost), 'inventory_after': sorted(after.elements())})
        inv = after; wallet -= cost; pending[iid] -= 1
        if not pending[iid]:
            del pending[iid]
    require(not pending, 'native replay dropped a retained purchase')
    return {'before': before, 'action': list(action), 'wallet': float(row['gold_est']), 'steps': steps}


def synthetic_rows(policy):
    champions = sorted(policy.champ_index)[:10]
    require(len(champions) == 10, 'ten synthetic champions required')
    flags = {'slot_state_version': 'role-slots-v1', 'bot_quest_complete': False,
             'unmodeled_regular_slots': 0, 'role_slot_inventory_unknown': False,
             'footwear_state_version': 'footwear-timer-slots-v1',
             'magical_footwear_selected': False, 'footwear_granted': False}
    inventories = ([], [1001, 1029], [1036, 1036], [1001, 1033])
    result = []
    for i in range(16):
        cid = champions[i % 10]; rest = [c for c in champions if c != cid]
        others = [{**flags, 'champion_id': c, 'champion': policy.dragon.champion_id_name(c),
                   'team_id': 100 if j < 4 else 200, 'role': 'MIDDLE' if j == 4 else '',
                   'items': [1036] if j % 2 else [1052], 'level': 10, 'gold': 0.}
                  for j, c in enumerate(rest)]
        inv = inventories[i % 4]; wallet = (500., 1000., 1800., 3000.)[i // 4]
        result.append({**flags, 'match_id': 'SYNTHETIC-RECIPE-'+str(i),
            'champion_id': cid, 'champion': policy.dragon.champion_id_name(cid), 'role': 'MIDDLE',
            'team_id': 100, 'inventory': list(inv), 'n_inventory': len(inv), 'others': others,
            'gold': wallet, 'gold_est': wallet, 'gold_est_version': 'prequential-v3',
            'level': 10, 'ts': 600000+i*10000, 'cs': 75, 'kills': 1, 'deaths': 1, 'assists': 2,
            'total_gold': 0., 'ally_obj': {}, 'enemy_obj': {}, 'keystone_id': 0, 'sub_style': 0,
            'summ1': 4, 'summ2': 14, 'label_counts': {'999999': 1}, 'label_id': 999999,
            'target_encoding': 'item-counts-v1', 'save_kind': 'no_buy_death'})
    return result


def run(output):
    started = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'immutable new synthetic output required')
    output.mkdir(parents=True)
    entry = reference.runtime_code_bindings()
    protocol_hash = producer.file_hash(PROTOCOL)

    # Missing local statics must fail, not silently trigger network collection.
    def boundary(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'sqlite3.connect'):
            raise RuntimeError('synthetic probe cannot use network or databases')
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            raw = os.fsdecode(args[0]).replace('\\', '/').lower()
            if ('raw_match_v5/' in raw or '/visits_' in raw or raw.endswith(('.db', '.sqlite'))
                    or '/trajectories.jsonl' in raw or '/targets.jsonl' in raw):
                raise RuntimeError('synthetic probe cannot read game or target payloads')
    sys.addaudithook(boundary)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), 'declared idle CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import app.mixture_policy as mp
    active = mp.activate_runtime(producer.PACKAGE, reference.REVIEW)
    policy = mp.load_package('mixture-v4-m2-v1', device='cuda')
    require(policy.digest == reference.PACKAGE_DIGEST, 'frozen M2 identity mismatch')
    rows = synthetic_rows(policy); receipts = []; measurements = []
    torch.cuda.reset_peak_memory_stats()

    # Two warm batches, then three timed same-shape batches. All rows synthetic.
    for iteration in range(5):
        require(time.monotonic()-started < 600, 'synthetic probe deadline reached')
        torch.cuda.synchronize(); t0 = time.monotonic()
        probabilities, options = reference.capture_native(policy, rows)
        torch.cuda.synchronize(); seconds = time.monotonic()-t0
        values = probabilities.cpu().numpy()
        reference.validate_probabilities(values, 16)
        if iteration >= 2:
            measurements.append(seconds)
        if iteration == 4:
            original = values.copy(); original_options = options
        print(json.dumps({'stage': 'synthetic_m2', 'iteration': iteration,
                          'warmup': iteration < 2, 'seconds': seconds}), flush=True)

    t0 = time.monotonic()
    for row, options in zip(rows, original_options, strict=True):
        reference.validate_options(options, policy.group_main)
        for option in options:
            receipt = replay_candidate(row, policy.action_of(option['group']), policy.dragon)
            receipt.update(synthetic_id=row['match_id'], group=option['group'])
            receipts.append(receipt)
    replay_seconds = time.monotonic()-t0

    changed = [reference.mutate_ignored(row) for row in rows]
    altered, altered_options = reference.capture_native(policy, changed)
    require(torch.equal(torch.from_numpy(original).to('cuda'), altered), 'forbidden fields changed native probabilities')
    require(original_options == altered_options, 'forbidden fields changed native options')

    fixtures = []
    for inv, action, wallet, free in [([1001, 1029], [3047], 600., False),
        ([1001, 1029, 1029], [3047, 1029], 1000., False),
        ([], [1036, 1036], 1000., False), ([1058], [3089], 10000., False),
        ([2422, 1029], [3047], 1000., True), ([1001, 1029], [], 10., False)]:
        row = deepcopy(rows[0]); row.update(inventory=inv, gold_est=wallet,
            magical_footwear_selected=free, footwear_granted=free)
        fixtures.append(replay_candidate(row, action, policy.dragon))
    all_items = set()
    for receipt in [*receipts, *fixtures]:
        all_items.update(receipt['before']); all_items.update(receipt['action'])
        for step in receipt['steps']:
            all_items.update(step['inventory_after'])
    payload = {'schema': 'synthetic-native-recipe-receipts-v1', 'world': '16.18',
               'receipts': receipts, 'fixtures': fixtures,
               'native_items': {str(i): policy.dragon.item(i) for i in sorted(all_items)}}
    reference.write_json(output/'receipts.json', payload)
    reference.revalidate(entry)
    require(producer.file_hash(PROTOCOL) == protocol_hash, 'consumed design changed')
    report = {'schema': 'recipe-transition-native-probe-v1', 'state': 'complete', 'passed': True,
        'package_digest': policy.digest, 'runtime': active, 'static_signature': policy.dragon.signature,
        'synthetic_distinct_states': 16, 'synthetic_scored_rows_including_repeats': 96,
        'timed_batch_size': 16, 'timed_batch_seconds': measurements,
        'median_seconds_per_synthetic_row': sorted(measurements)[1]/16,
        'receipt_count': len(receipts), 'fixture_count': len(fixtures),
        'native_receipt_replay_seconds': replay_seconds, 'forbidden_field_mutation_rows': 16,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_s': time.monotonic()-started, 'protocol_sha256': protocol_hash,
        'files': {'receipts.json': producer.file_hash(output/'receipts.json')},
        'code_sha256': reference.runtime_code_bindings(),
        'game_payload_reads': False, 'target_file_opened': False, 'network_reads': False,
        'training_performed': False, 'consumer_runtime_imported': False,
        'limits': ['Synthetic latency is not representative real-corpus throughput or a purchase-accuracy result.']}
    reference.write_json(output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('state', 'passed', 'elapsed_s', 'median_seconds_per_synthetic_row', 'receipt_count')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
