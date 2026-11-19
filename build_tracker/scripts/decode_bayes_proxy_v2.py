"""Unchanged native route-distribution decoding and isolated16.17 legality.

Source-owned adapter. No target purchases, consumer runtime or model fitting.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import export_bc20_ordinary_reference_v2 as reference
import export_recipe_roles_v1 as scope
from source_probability_rows_v1 import ProbabilityRows
producer = reference.producer; require = reference.require
ARMS = ('m2', 'history', 'scrambled', 'static_memory')
PROTOCOL = ROOT.parent/'winprob/docs/bayes_proxy_probe_protocol_v1.md'
PROTOCOL_SHA = 'a60292830d167824f8d14a608020c4f75da3a023b65ef85d5c42ad7f2b91837b'
AMENDMENT = ROOT.parent/'winprob/docs/bayes_proxy_decode_memory_amendment_v2.md'
AMENDMENT_SHA = 'de770b42d94453f1c2cb55b302e41459fde23c25ecc4a25f08bd3a669f0b2ad7'
OLD = ROOT/'data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2/manifest.json'
CONTEXT = ROOT/'data/exports/winprob-recipe-transition/recipe-contexts-20261001-v1/manifest.json'


def supplied_probability(original, corrected, neutral):
    require(original.dtype == np.float32 and original.shape == (1, 9333), 'original native precision/shape changed')
    require(corrected.dtype == np.float64 and corrected.shape == original.shape, 'correction precision/shape changed')
    if neutral:
        p = original.astype(np.float64); p /= p.sum(1, keepdims=True)
        require(np.array_equal(corrected, p), 'neutral flag is not an exact unchanged normalized prior')
        return original
    return corrected


def load_bundle(path):
    manifest, sha = producer.load_bound(path)
    require(path.is_relative_to(ROOT.parent/'winprob/data/runs')
            and manifest['schema'] == 'bayes-proxy-distribution-v1' and manifest['state'] == 'complete'
            and manifest['selection_count'] == 2900 and manifest['groups'] == 9333
            and manifest['arms'] == ['history', 'scrambled', 'static_memory'], 'wrong route distribution')
    require(file_sha(PROTOCOL) == PROTOCOL_SHA == manifest['protocol_sha256']
            and Path(manifest['protocol_path']).resolve() == PROTOCOL.resolve(), 'route protocol changed')
    for flag in ('query_fold_current_future_labels_used_for_its_predictions', 'training_performed', 'accuracy_computed', 'source_runtime_imported', 'test_payload_reads', 'confirmation_payload_reads'):
        require(manifest[flag] is False, 'distribution boundary failed: '+flag)
    sm, sh = producer.load_bound(scope.SELECTION); require(sh == scope.SELECTION_SHA, 'fixed selection changed')
    selected_path = scope.SELECTION.parent/'selection.json'
    require(file_sha(selected_path) == sm['files']['selection.json'], 'selection metadata changed')
    selected = producer.load_json(selected_path)['entries']; expected = [e for e in selected if e['is_original_focal']]
    base, bh = producer.load_bound(scope.BASE); require(bh == scope.BASE_SHA, 'original FIT metadata changed')
    admitted, tests, validation, confirm, admission_bindings = producer.validate_admission(base['source_admission_path'])
    ids = {e['match_id'] for e in selected}; lookup = {e['match_id']: e for e in admitted['games']}
    require(len(ids) == 3518 and not ids & (tests | validation | confirm)
            and all(lookup[i]['role'] == 'train' and lookup[i]['patch'] == '16.17' for i in ids), 'non-FIT/reserved contexts')
    bindings = {str(path): sha, str(PROTOCOL): PROTOCOL_SHA, str(scope.SELECTION): sh,
        str(selected_path): sm['files']['selection.json'], str(scope.BASE): bh, **admission_bindings}
    require(file_sha(AMENDMENT) == AMENDMENT_SHA, 'native reader amendment changed')
    bindings[str(AMENDMENT)] = AMENDMENT_SHA
    # All prediction/context payload access starts AFTER admission above.
    required_files = {'selection.json', 'group_main.json', 'm2.npy', 'history.npy', 'scrambled.npy', 'static_memory.npy', 'neutral_flags.npy'}
    require(set(manifest['files']) == required_files, 'distribution member contract changed')
    for name, want in manifest['files'].items():
        p = (path.parent/name).resolve(); require(p.parent == path.parent and file_sha(p) == want, 'prediction artifact changed')
        bindings[str(p)] = want
    ledger = producer.load_json(path.parent/'selection.json'); groups = producer.load_json(path.parent/'group_main.json')
    require(len(ledger) == len(expected) == 2900 and len(groups) == 9333 and groups[0] == [], 'query/catalogue axis changed')
    for i, (record, expected_record) in enumerate(zip(ledger, expected, strict=True)):
        require(record == {**expected_record, 'query_ordinal': i}, 'focal query identity or order changed')
    require(len({e['match_id'] for e in ledger}) == 2900, 'more than one focal per game')
    cm, ch = producer.load_bound(CONTEXT)
    require(ch == manifest['source_context_manifest_sha256'] and cm['state'] == 'complete'
            and Path(manifest['source_context_manifest_path']).resolve() == CONTEXT
            and cm['selection_manifest_sha256'] == scope.SELECTION_SHA, 'context release changed')
    rp, rh = producer.load_bound(OLD)
    require(rh == 'f0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe'
            == manifest['reference_manifest_sha256'] and Path(manifest['reference_manifest_path']).resolve() == OLD,
            'frozen native reference changed')
    bindings.update({str(CONTEXT): ch, str(OLD): rh})
    query_map = {e['ordinal']: i for i, e in enumerate(ledger)}; contexts = [None]*2900; old = [None]*2900
    for p, owner, target in ((CONTEXT.parent/'contexts.jsonl.gz', cm, contexts), (OLD.parent/'rows.jsonl.gz', rp, old)):
        require(file_sha(p) == owner['files'][p.name], 'source payload changed'); bindings[str(p)] = owner['files'][p.name]
        with gzip.open(p, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); ordinal = row['ordinal']
                require(type(ordinal) is int and 0 <= ordinal < len(selected)
                        and row['sample_id'] == selected[ordinal]['sample_id'] and row['match_id'] in ids,
                        'source context outside fixed selection')
                if ordinal in query_map:
                    i = query_map[ordinal]; require(target[i] is None, 'duplicate query'); target[i] = row
    require(all(c is not None for c in contexts+old), 'missing focal inputs')
    for c, r in zip(contexts, old, strict=True):
        require(producer.value_hash(c['source_projection']) == r['source_projection_sha256']
                and c['sample_id'] == r['sample_id'], 'native predecision projection changed')
    return manifest, sha, ledger, groups, contexts, old, bindings, base['source_admission_path']


file_sha = producer.file_hash


def run(args):
    start = time.monotonic(); output = args.output.resolve(); require(not output.exists(), 'immutable new source artifact required')
    entry_code = reference.runtime_code_bindings()
    def boundary(event, values):
        if event in ('socket.connect', 'socket.getaddrinfo', 'sqlite3.connect'):
            raise RuntimeError('route decoder may not collect or query databases')
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            name = os.fsdecode(values[0]).replace('\\', '/').lower()
            if ('raw_match_v5/' in name or name.endswith(('/targets.jsonl', '/trajectories.jsonl.gz', '.db', '.sqlite'))):
                raise RuntimeError('route decoder may not read targets/raw/trajectory payloads')
    sys.addaudithook(boundary)
    path = args.distribution_manifest.resolve()
    manifest, sha, ledger, groups, contexts, old, bindings, admission = load_bundle(path)
    output.mkdir(parents=True)
    def guard(): require(time.monotonic()-start < 1800, 'fixed30minute source stage bound reached')
    def status(stage, **extra):
        guard(); record = {'stage': stage, 'elapsed_s': time.monotonic()-start, **extra}
        (output/'status.json').write_text(json.dumps(record)+'\n', encoding='utf-8'); print(json.dumps(record), flush=True)
    status('activating', mode=args.mode); counts = Counter(); records = []
    if args.mode == 'decode':
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        require(torch.cuda.is_available(), 'original CUDA topk required for exact native replay')
        import app.mixture_policy as mp
        active = mp.activate_runtime(producer.PACKAGE, reference.REVIEW)
        policy = mp.load_package('mixture-v4-m2-v1', device='cuda')
        actual = [[[int(i), int(n)] for i, n in g] for g in policy.group_main]
        require(policy.digest == reference.PACKAGE_DIGEST and actual == groups
                and policy.shortlist == 128 and policy.n_options == 3, 'frozen native decoder changed')
        for p in (producer.PACKAGE/'manifest.json', producer.PACKAGE/'package_id.json', reference.REVIEW): bindings[str(p)] = file_sha(p)
        for weight in policy.manifest['weights'].values(): bindings[str(producer.PACKAGE/weight['file'])] = weight['sha256']
        from app.conditional_policy import purchase_sequence
        matrices = {arm: ProbabilityRows(path.parent/f'{arm}.npy', (2900, 9333), np.float32 if arm == 'm2' else np.float64) for arm in ARMS}
        neutral = np.load(path.parent/'neutral_flags.npy', allow_pickle=False)
        require(neutral.shape == (3, 2900) and neutral.dtype == bool, 'neutral flag axis/type changed')
        for arm, p in matrices.items():
            require(p.shape == (2900, 9333) and p.dtype == (np.float32 if arm == 'm2' else np.float64), 'probability shape/type changed')
        for i, context in enumerate(contexts):
            guard(); row = reference.model_row(context); rows = [row]
            require(reference.array_hash(matrices['m2'][i]) == old[i]['full_probability_sha256'], 'original float32 prior mismatch')
            record = {'sample_id': context['sample_id'], 'match_id': context['match_id'], 'ordinal': i,
                      'source_ordinal': context['ordinal'], 'options': {}, 'audits': {}}
            for arm in ARMS:
                array = matrices[arm][i:i+1]
                require(np.isfinite(array).all() and np.all(array >= 0)
                        and abs(array.sum(dtype=np.float64)-1) <= (2e-6 if arm == 'm2' else 2e-12), 'invalid supplied probabilities')
                if arm != 'm2':
                    array = supplied_probability(matrices['m2'][i:i+1], array, bool(neutral[ARMS.index(arm)-1, i]))
                p = torch.from_numpy(np.array(array, copy=True)).to('cuda')
                options = reference.options_override(policy, rows, p)[0]
                mutated = reference.options_override(policy, [reference.mutate_ignored(row)], p)[0]
                require(options == mutated, 'forbidden fields changed route decoding')
                if arm == 'm2': require(options == old[i]['m2_options'], 'M2 native option replay mismatch')
                audits = reference.audit_options(options, row, groups, purchase_sequence, policy.dragon)
                require(all(a['feasible'] for a in audits), 'native18 decoder produced an infeasible action')
                record['options'][arm] = options; record['audits'][arm] = audits
                counts[arm+'_options'] += len(options); counts[arm+'_infeasible'] += sum(not a['feasible'] for a in audits)
            records.append(record)
            if (i+1) % 128 == 0: status('decoding', completed=i+1, queries=2900)
        world = '16.18'; dragon = policy.dragon
    else:
        require(args.decoded_manifest is not None, 'native17 audit requires completed decoder')
        decoded, ds = producer.load_bound(args.decoded_manifest)
        require(decoded['schema'] == 'source-bayes-proxy-decoder-v1' and decoded['state'] == 'complete'
                and decoded['distribution_manifest_sha256'] == sha and decoded['m2_replay_mismatches'] == 0, 'decoder identity changed')
        bindings[str(args.decoded_manifest.resolve())] = ds
        rp = args.decoded_manifest.parent/'rows.jsonl'; require(file_sha(rp) == decoded['files'][rp.name], 'decoded rows changed')
        bindings[str(rp)] = file_sha(rp); decoded_rows = reference.read_lines(rp)
        require(len(decoded_rows) == 2900, 'decoded query count changed')
        trajectory, th = producer.load_bound(scope.SOURCE); bindings[str(scope.SOURCE)] = th
        review = ROOT/'data/acceptance/16.17_history_20260927/inputs/static_review.json'
        require(file_sha(review) == trajectory['input_metadata_sha256'][str(review)]
                == 'afe5bb33d4922bf0395737f598d191640e215e889d99469868d45f88a37fb279', 'native17 reviewed statics changed')
        bindings[str(review)] = file_sha(review)
        import reconstructible_1617_runtime as runtime
        active = runtime.activate(review); require(active == trajectory['runtime'], 'native17 runtime differs from admitted reconstruction')
        from app.ddragon import dragon_for_patch
        from app.conditional_policy import purchase_sequence
        dragon = dragon_for_patch('16.17'); require(dragon.version == '16.17.1', 'wrong audit world')
        for p in (ROOT/'data/cache/item-16.17.1.json', ROOT/'data/cache/champion-16.17.1.json'): bindings[str(p)] = file_sha(p)
        for i, (context, previous) in enumerate(zip(contexts, decoded_rows, strict=True)):
            guard(); require(previous['ordinal'] == i and previous['sample_id'] == context['sample_id'], 'decoded/audit query mismatch')
            row = reference.model_row(context); record = {'sample_id': context['sample_id'], 'ordinal': i, 'audits': {}}
            for arm in ARMS:
                checks = reference.audit_options(previous['options'][arm], row, groups, purchase_sequence, dragon)
                record['audits'][arm] = checks; counts[arm+'_options'] += len(checks)
                counts[arm+'_infeasible'] += sum(not a['feasible'] for a in checks)
            records.append(record)
        world = '16.17'
    reference.revalidate(entry_code); code = reference.runtime_code_bindings()
    reference.revalidate({**bindings, **code}); producer.validate_admission(admission)
    with (output/'rows.jsonl').open('x', encoding='utf-8') as stream:
        for record in records: stream.write(producer.canonical(record).decode()+'\n')
    files = {'rows.jsonl': file_sha(output/'rows.jsonl')}
    result = {'schema': 'source-bayes-proxy-decoder-v1' if args.mode == 'decode' else 'source-bayes-proxy-native17-v1',
        'state': 'complete', 'selection_count': 2900, 'files': files, 'counts': dict(counts), 'world': world,
        'distribution_manifest_path': str(path), 'distribution_manifest_sha256': sha,
        'protocol_sha256': PROTOCOL_SHA, 'reader_amendment_sha256': AMENDMENT_SHA, 'native_runtime': active, 'static_signature': dragon.signature,
        'bindings_sha256': bindings, 'code_sha256': code, 'm2_replay_mismatches': 0 if args.mode == 'decode' else None,
        'mutation_checked': 11600 if args.mode == 'decode' else None, 'target_file_opened': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'training_performed': False,
        'outcomes_analyzed': False, 'accuracy_computed': False, 'consumer_runtime_imported': False,
        'elapsed_s': time.monotonic()-start}
    reference.write_json(output/'manifest.json', result); status('complete', counts=dict(counts))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('decode', 'audit17'), required=True)
    p.add_argument('--distribution-manifest', type=Path, required=True); p.add_argument('--decoded-manifest', type=Path)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
