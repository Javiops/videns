"""Retain full native M2 priors with exact replay of every previously hashed row.

Reads only the completed selected-context release and metadata; no target file,
trajectory archive, database, raw archive or consumer runtime is opened.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import shutil
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import probe_recipe_transition_native_v2 as native
import export_recipe_roles_v1 as scope
import recipe_native_receipts_v1 as receipts_adapter
reference = native.reference; producer = native.producer; require = native.require
CONTEXT = ROOT/'data/exports/winprob-recipe-transition/recipe-contexts-20261001-v1/manifest.json'
SECONDS = 10800
PRIOR = ROOT/'data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2/manifest.json'
PRIOR_SHA = 'f0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe'
NEW_PROTOCOL = ROOT.parent/'winprob/docs/route_prior_export_protocol_v1.md'
NEW_PROTOCOL_SHA = 'f76ffa31a92ea3d3d8d0a563366dee03478a1184050cda703f83cf326502d2cb'


def verify_probability_row(prob, expected_sha):
    require(prob.dtype == np.float32 and prob.shape == (9333,)
            and reference.array_hash(prob) == expected_sha,
            'complete native probability row does not replay exactly')



def coarsen(full, groups):
    """Normalize the frozen float32 mass once; tail is summed, never subtracted."""
    p = np.asarray(full, dtype=np.float64)
    require(p.shape == (9333,) and np.isfinite(p).all() and np.all(p >= 0) and abs(p.sum()-1) <= 2e-6, 'invalid frozen distribution')
    require(1 <= len(groups) <= 3 and len(set(groups)) == len(groups)
            and all(type(g) is int and 0 <= g < 9333 for g in groups), 'invalid native candidate identities')
    p /= p.sum(); keep = np.ones(len(p), dtype=bool); keep[groups] = False
    selected = p[groups].tolist(); require(all(v > 0 for v in selected), 'native option has zero prior support')
    values = selected + [0.]*(3-len(groups)) + [float(p[keep].sum())]
    require(abs(sum(values)-1) < 1e-12, 'coarsened mass changed')
    return values


def run(output):
    started = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable M2 release required')
    def guard():
        require(time.monotonic()-started < SECONDS, 'fixed180minute native worker bound reached')
    require(producer.file_hash(NEW_PROTOCOL) == NEW_PROTOCOL_SHA, 'route-prior protocol changed')
    old_manifest, old_sha = producer.load_bound(PRIOR)
    require(old_sha == PRIOR_SHA and old_manifest['state'] == 'complete'
            and old_manifest['states'] == 56288 and old_manifest['games'] == 3518
            and old_manifest['target_file_opened'] is False, 'frozen prior release changed')
    manifest, ch = producer.load_bound(CONTEXT)
    require(manifest['schema'] == 'recipe-transition-source-contexts-v1' and manifest['state'] == 'complete'
            and manifest['games'] == 3518 and manifest['states'] == 56288
            and manifest['selection_manifest_sha256'] == scope.SELECTION_SHA
            and manifest['protocol_sha256'] == scope.PROTOCOL_SHA, 'context release changed')
    require(producer.file_hash(scope.PROTOCOL) == scope.PROTOCOL_SHA, 'consumed protocol changed')
    selection_meta, smh = producer.load_bound(scope.SELECTION); require(smh == scope.SELECTION_SHA, 'selection manifest changed')
    selection_path = scope.SELECTION.parent/'selection.json'
    require(producer.file_hash(selection_path) == selection_meta['files']['selection.json'], 'selection data changed')
    selection = producer.load_json(selection_path); entries = selection['entries']; n = len(entries)
    require(n == 56288, 'selection size changed')
    path = CONTEXT.parent/'contexts.jsonl.gz'
    bindings = {str(CONTEXT): ch, str(scope.SELECTION): smh, str(selection_path): selection_meta['files']['selection.json'],
                str(scope.PROTOCOL): scope.PROTOCOL_SHA, str(path): manifest['files'][path.name]}
    for raw, want in manifest['code_sha256'].items():
        require(producer.file_hash(raw) == want, 'context producer changed'); bindings[raw] = want
    # Do not follow inherited trajectory or target provenance paths.
    base, bh = producer.load_bound(scope.BASE); require(bh == scope.BASE_SHA, 'base metadata changed'); bindings[str(scope.BASE)] = bh
    admission = Path(base['source_admission_path'])
    admitted, tests, validation, confirm, ab = producer.validate_admission(admission); bindings.update(ab)
    ids = {e['match_id'] for e in entries}; lookup = {e['match_id']: e for e in admitted['games']}
    require(not ids & (tests | validation | confirm) and all(lookup[mid]['role'] == 'train' and lookup[mid]['patch'] == '16.17' for mid in ids), 'non-FIT/reserved query')
    require(producer.file_hash(path) == manifest['files'][path.name], 'context payload changed')
    # Admission above precedes opening the historical prediction payload.
    old_rows_path = PRIOR.parent/'rows.jsonl.gz'
    require(producer.file_hash(old_rows_path) == old_manifest['files']['rows.jsonl.gz'], 'prior rows changed')
    bindings.update({str(PRIOR): old_sha, str(old_rows_path): old_manifest['files']['rows.jsonl.gz'],
                     str(NEW_PROTOCOL): NEW_PROTOCOL_SHA})
    for raw, want in old_manifest['code_sha256'].items():
        require(producer.file_hash(raw) == want, 'consumed native release code changed'); bindings[raw] = want
    old_rows = {}
    with gzip.open(old_rows_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            old = json.loads(line); ordinal = old['ordinal']
            require(type(ordinal) is int and 0 <= ordinal < n and ordinal not in old_rows
                    and all(old[k] == entries[ordinal][k] for k in
                            ('sample_id', 'match_id', 'participant_id', 'source_event_index', 'fold', 'is_original_focal')),
                    'prior row identity changed')
            old_rows[ordinal] = {k: old[k] for k in ('full_probability_sha256', 'source_projection_sha256',
                                                    'group_ids', 'reference_mass', 'm2_options')}
    require(len(old_rows) == n, 'prior probability hash ledger incomplete')
    pilot_ids = set(selection['pilot_sample_ids'])
    require(len(pilot_ids) == 200 and sorted(pilot_ids) == sorted(e['sample_id'] for e in entries)[:200], 'label-blind pilot changed')
    def contexts():
        seen = set()
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                guard(); c = json.loads(line); i = c['ordinal']
                require(type(i) is int and 0 <= i < n and i not in seen, 'duplicate/out-of-range context')
                e = entries[i]
                require(all(c[k] == v for k, v in e.items()) and 'action' not in c and 'target' not in c
                        and c['reference_roles_not_certified_predecision'] is True, 'context identity or target boundary changed')
                require(set(c['source_projection']) == set(producer.STATE_FIELDS) | {'role'}, 'M2 prestate allowlist changed')
                seen.add(i); yield c
        require(len(seen) == n, 'context file is incomplete')
    pilot = sorted((c for c in contexts() if c['sample_id'] in pilot_ids), key=lambda c: c['sample_id'])
    require(len(pilot) == 200, 'pilot contexts missing')

    def boundary(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'sqlite3.connect'):
            raise RuntimeError('M2 release cannot collect or query databases')
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            raw = os.fsdecode(args[0]).replace('\\', '/').lower()
            if ('raw_match_v5/' in raw or raw.endswith(('/targets.jsonl', '/trajectories.jsonl.gz', '.db', '.sqlite'))):
                raise RuntimeError('M2 release cannot read targets or source trajectories')
    sys.addaudithook(boundary)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'declared idle CUDA required')
    import app.mixture_policy as mp
    active = mp.activate_runtime(producer.PACKAGE, reference.REVIEW)
    policy = mp.load_package('mixture-v4-m2-v1', device='cuda')
    require(policy.digest == reference.PACKAGE_DIGEST and len(policy.group_main) == 9333
            and policy.shortlist == 128 and policy.n_options == 3, 'frozen M2 contract changed')
    fixture = native.synthetic_rows(policy)[0]
    for inventory, action, wallet in (([1001, 1029], [3047], 600.), ([], [1036, 1036], 1000.), ([1001], [], 0.)):
        row = {**fixture, 'inventory': inventory, 'gold_est': wallet}
        old_receipt = native.replay_candidate(row, action, policy.dragon)
        new_receipt = receipts_adapter.replay_candidate(row, action, policy.dragon)
        require(old_receipt == new_receipt, 'receipt fix changed explicit-field native semantics')
        omitted = dict(row); omitted.pop('role_slot_inventory_unknown')
        require(new_receipt == receipts_adapter.replay_candidate(omitted, action, policy.dragon), 'absent optional flag differs from native false default')
    entry_code = reference.runtime_code_bindings()
    require(shutil.disk_usage(ROOT).free >= 5*1024**3, '5GiB free disk required')
    output.mkdir(parents=True)
    full_path = output/'full_probability.npy'
    full_probability = np.lib.format.open_memmap(full_path, mode='w+', dtype=np.float32, shape=(n, 9333))
    def status(stage, **extra):
        value = {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra}
        (output/'status.json').write_text(json.dumps(value, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(value), flush=True)
    reference.write_json(output/'selection_receipt.json', {'selection_manifest_sha256': smh, 'source_context_manifest_sha256': ch,
        'states': n, 'pilot_states': 200, 'target_file_opened': False, 'protocol_sha256': scope.PROTOCOL_SHA})
    tensor_fields = ('champs', 'sides', 'items', 'query', 'inv', 'hist')
    def score(batch, mutate=False):
        guard(); rows = [reference.model_row(c) for c in batch]
        torch.cuda.synchronize(); t = time.monotonic()
        probabilities, options = reference.capture_native(policy, rows)
        values = probabilities.cpu().numpy(); reference.validate_probabilities(values, len(batch))
        results = []
        for c, row, prob, opts in zip(batch, rows, values, options, strict=True):
            verify_probability_row(prob, old_rows[c['ordinal']]['full_probability_sha256'])
            full_probability[c['ordinal']] = prob
            reference.validate_options(opts, policy.group_main)
            groups = [o['group'] for o in opts]
            receipts = [receipts_adapter.replay_candidate(row, policy.action_of(g), policy.dragon) for g in groups]
            results.append({'ordinal': c['ordinal'], 'sample_id': c['sample_id'], 'match_id': c['match_id'],
                'participant_id': c['participant_id'], 'source_event_index': c['source_event_index'],
                'fold': c['fold'], 'is_original_focal': c['is_original_focal'], 'group_ids': groups,
                'reference_mass': coarsen(prob, groups), 'm2_options': opts, 'receipts': receipts,
                'residual_public': c['residual_public'], 'full_probability_sha256': reference.array_hash(prob),
                'source_projection_sha256': producer.value_hash(c['source_projection'])})
        seconds = time.monotonic()-t
        if mutate:
            changed = [reference.mutate_ignored(row) for row in rows]
            left = policy._tp.ShopDataset(rows, policy.champ_index, policy.item_index, policy.label_index, policy.dragon)
            right = policy._tp.ShopDataset(changed, policy.champ_index, policy.item_index, policy.label_index, policy.dragon)
            require(all(torch.equal(getattr(left, key), getattr(right, key)) for key in tensor_fields), 'ignored fields changed actual M2 inputs')
            repeated, repeated_options = reference.capture_native(policy, changed)
            require(torch.equal(probabilities, repeated) and options == repeated_options, 'ignored fields changed full native prediction/options')
        return results, seconds

    # Two fixed full-batch warmups, no purchase targets opened or performance inspected.
    for _ in range(2):
        score(pilot[:16])
    torch.cuda.reset_peak_memory_stats(); pilot_outputs = {}; timings = []
    for lo in range(0, 200, 16):
        records, elapsed = score(pilot[lo:lo+16], mutate=True)
        pilot_outputs.update({r['ordinal']: r for r in records}); timings.append({'rows': len(records), 'seconds': elapsed})
        status('real_cost_pilot', completed=len(pilot_outputs))
    per_row = sum(t['seconds'] for t in timings)/200
    remaining = SECONDS-(time.monotonic()-started); projected = 1.5*per_row*n+900
    preflight = {'states': 200, 'batch_size': 16, 'batch_timings': timings, 'seconds_per_row': per_row,
        'projected_seconds': projected, 'remaining_seconds': remaining, 'allowed_fraction': .8,
        'passed': projected <= .8*remaining, 'mutation_checked_states': 200, 'model_input_fields_verified': list(tensor_fields),
        'target_file_opened': False, 'real_accuracy_computed': False}
    reference.write_json(output/'preflight.json', preflight)
    require(preflight['passed'], 'complete fixed export fails resource projection; do not shrink sample')
    status('full_native_export', completed=0, projected_seconds=projected)
    count = 0; seen = set(); buffer = []
    rows_path = output/'rows.jsonl.gz'
    with rows_path.open('xb') as file, gzip.GzipFile(filename='', fileobj=file, mode='wb', compresslevel=1, mtime=0) as stream:
        def emit(records):
            nonlocal count
            for record in records:
                require(record['ordinal'] not in seen, 'duplicate native prediction')
                require(all(record[k] == v for k, v in old_rows[record['ordinal']].items()),
                        'native probability/options/projection replay differs')
                stream.write(producer.canonical(record)+b'\n'); seen.add(record['ordinal']); count += 1
            if count//1024 != (count-len(records))//1024:
                full_probability.flush()
                status('full_native_export', completed=count, states=n)
                producer.validate_admission(admission)
        for context in contexts():
            if context['ordinal'] in pilot_outputs:
                emit([pilot_outputs.pop(context['ordinal'])])
            else:
                buffer.append(context)
                if len(buffer) == 16:
                    records, _ = score(buffer); emit(records); buffer = []
        if buffer:
            records, _ = score(buffer); emit(records)
    require(count == n and len(seen) == n and not pilot_outputs, 'native release dropped selected states')
    full_probability.flush(); del full_probability
    saved = np.load(full_path, mmap_mode='r', allow_pickle=False)
    require(saved.dtype == np.float32 and saved.shape == (n, 9333), 'saved full-prior shape changed')
    for ordinal in range(n):
        guard()
        verify_probability_row(saved[ordinal], old_rows[ordinal]['full_probability_sha256'])
    del saved
    groups = [[[int(i), int(c)] for i, c in group] for group in policy.group_main]
    reference.write_json(output/'group_main.json', groups)
    statics = {'world': '16.18', 'static_signature': policy.dragon.signature,
        'items': {str(i): {'stats': value.get('stats', {})} for i, value in policy.dragon._items['data'].items()},
        'champion_tags': {str(i): value.get('tags', []) for i, value in policy.dragon._champ_by_id.items()}}
    reference.write_json(output/'native_descriptors.json', statics)
    reference.revalidate({**bindings, **entry_code}); producer.validate_admission(admission)
    result = {'schema': 'route-prior-native-m2-v1', 'state': 'complete', 'games': 3518, 'states': n,
        'package_digest': policy.digest, 'source_context_manifest_path': str(CONTEXT), 'source_context_manifest_sha256': ch,
        'selection_manifest_path': str(scope.SELECTION), 'selection_manifest_sha256': smh, 'protocol_sha256': scope.PROTOCOL_SHA,
        'route_protocol_path': str(NEW_PROTOCOL), 'route_protocol_sha256': NEW_PROTOCOL_SHA,
        'prior_release_manifest_path': str(PRIOR), 'prior_release_manifest_sha256': old_sha,
        'complete_probability_rows_replayed_exactly': n, 'saved_rows_verified_exactly': n,
        'full_probability_shape': [n, 9333], 'full_probability_dtype': 'float32',
        'full_probability_payload_bytes': n*9333*4,
        'bindings_sha256': bindings, 'code_sha256': reference.runtime_code_bindings(), 'native_world': '16.18', 'native_runtime': active,
        'files': {name: producer.file_hash(output/name) for name in ('selection_receipt.json', 'preflight.json', 'rows.jsonl.gz', 'group_main.json', 'native_descriptors.json', 'full_probability.npy')},
        'receipt_default_equivalence_fixtures': 3, 'real_pilot': preflight, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'target_file_opened': False, 'source_trajectory_payload_reads': False, 'source_database_reads': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'consumer_runtime_imported': False,
        'training_performed': False, 'outcomes_analyzed': False, 'accuracy_computed': False, 'elapsed_s': time.monotonic()-started,
        'limits': ['Frozen native16.18 M2 retrospectively scored on16.17 FIT states; not chronological transfer evidence.',
                   'Legacy role inputs retained only for exact frozen-reference compatibility.']}
    reference.write_json(output/'manifest.json', result); status('complete', states=n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
