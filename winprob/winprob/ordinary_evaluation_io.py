"""Artifact-only evaluation of the predeclared ordinary recommendation trial.

No source-project imports, model inference, fitting, or raw corpus access.
Prediction artifacts must validate before the separate purchase ledger opens.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from collections import Counter
import numpy as np

from winprob import admission
from winprob.full_action_catalogue import load_catalogue, counts_key
from winprob.ordinary_policy_evaluation import (rank_candidates, score_decision,
    evaluate_scored_decisions, target_description, multiset, option_identity,
    validate_candidates)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/bc20_ordinary_eval_protocol_v1.md'
SELECTION_SHA = '611c1f7d3943074325c88659dcae59693518745bc5abcceebf52c5e67ce27f85'
CHECKPOINT_SHA = '57fca2f787aaff6dce43bfd729e34e90464de2f536abdf4ba534d5d01e5b1f0d'
CATALOGUE_SHA = '14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8'
M2_IDENTITY = 'ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def bind(path, expected, bindings):
    path = Path(path).resolve(strict=True)
    require(not any((parent/'QUARANTINED.md').exists() for parent in path.parents), 'Quarantined artifact cannot be consumed')
    require(digest(path) == expected, 'Artifact digest mismatch: '+str(path))
    require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting artifact bindings')
    bindings[str(path)] = expected
    return path


def sidecar(path, bindings):
    path = Path(path).resolve(strict=True)
    require(not any((parent/'QUARANTINED.md').exists() for parent in path.parents), 'Quarantined artifact cannot be consumed')
    saved = path.with_suffix('.sha256')
    require(saved.read_text(encoding='ascii').split() == [digest(path), path.name], 'Invalid artifact sidecar: '+str(path))
    bind(path, digest(path), bindings)
    bind(saved, digest(saved), bindings)


def load_manifest(path, schema, bindings, *, skip_files=()):
    path = Path(path).resolve(strict=True)
    sidecar(path, bindings)
    result = read_json(path)
    require(result.get('schema') == schema and result.get('state') == 'complete', 'Incomplete/wrong artifact schema')
    required_files = {
        'winprob-ordinary-contexts-v1': {'selection.json', 'contexts.jsonl', 'targets.jsonl'},
        'winprob-ordinary-bc20-export-v1': {'distribution.npz', 'rows.jsonl', 'group_main.json', 'full_to_group.json'},
        'winprob-ordinary-reference-v1': {'rows.jsonl', 'probabilities.npz', 'group_main.json'},
        'winprob-ordinary-native19-audit-v1': {'rows.jsonl'},
    }
    require(schema in required_files and required_files[schema] == set(result.get('files', {})),
            'Manifest must contain exactly the mandatory payload hashes')
    for name, expected in result.get('files', {}).items():
        require(Path(name).name == name and name not in ('.', '..'), 'Artifact file must be a plain filename')
        if name not in skip_files:
            bind(path.parent/name, expected, bindings)
    for field in ('bindings', 'bindings_sha256', 'code_sha256'):
        for source, expected in result.get(field, {}).items():
            require(Path(source).name != 'targets.jsonl', 'Target binding cannot be opened before prediction validation')
            require('raw_match_v5' not in Path(source).parts and Path(source).suffix not in ('.gz', '.db', '.sqlite'),
                    'Evaluator must not reopen raw corpus files')
            bind(source, expected, bindings)
    require(Path(result['protocol_path']).resolve() == PROTOCOL.resolve(), 'Wrong evaluation protocol')
    bind(PROTOCOL, result['protocol_sha256'], bindings)
    return result


def validate_source(path, bindings):
    manifest = load_manifest(path, 'winprob-ordinary-contexts-v1', bindings, skip_files=('targets.jsonl',))
    require(manifest.get('smoke') is False and manifest.get('full_cohort_complete') is True,
            'No smoke or partial cohort may be evaluated')
    for field in ('test_payload_reads', 'confirmation_payload_reads', 'outcomes_exported', 'outcomes_analyzed'):
        require(manifest.get(field) is False, 'Source boundary failure: '+field)
    directory = Path(path).resolve().parent
    require(manifest['files']['selection.json'] == SELECTION_SHA, 'Frozen selection changed')
    admitted_path = bind(manifest['admission_path'], manifest['admission_sha256'], bindings)
    admitted = read_json(admitted_path)
    admission.verify_manifest_sources(admitted)
    selection = read_json(directory/'selection.json')
    contexts = read_rows(directory/'contexts.jsonl')
    expected = {r['match_id'] for r in admitted['games'] if r['role'] == 'development' and r['patch'] == '16.19'}
    require(len(expected) == 859 and len(contexts) == len(selection['entries']) == 859,
            'Exact859-game source population required')
    require({r['match_id'] for r in contexts} == expected, 'Context population differs from admission')
    require(len({r['sample_id'] for r in contexts}) == 859, 'Duplicate context identity')
    next_tensor = 0
    for ordinal, (entry, row) in enumerate(zip(selection['entries'], contexts, strict=True)):
        mid = row['match_id']
        focal = 1+int.from_bytes(hashlib.sha256(('bc20-ordinary-v1/focal/'+mid).encode()).digest()[:8], 'big')%10
        choice = entry['selected_decision']
        require(entry['ordinal'] == ordinal and entry['match_id'] == mid and entry['pid'] == focal
                and row['participant_id'] == focal and row['selected_decision'] == choice,
                'Selected focal player/visit identity changed')
        sample = f'{mid}|{focal}|none' if choice is None else f"{mid}|{focal}|{choice['event_index']}|{choice['ts']}"
        require(row['sample_id'] == entry['sample_id'] == sample and row['role'] == 'development' and row['patch'] == '16.19',
                'Source role or selected sample changed')
        require(row['status'] in ('available', 'unavailable', 'no_admitted_decision'), 'Unknown source status')
        require((choice is None) == (row['status'] == 'no_admitted_decision'), 'Selected decision silently replaced/removed')
        if row['status'] == 'available':
            require(row['tensor_row'] == next_tensor, 'Source tensor index changed')
            require(row['context_timestamp_ms'] == choice['ts'] == row['prestate']['ts'] == row['source_projection']['ts'],
                    'Context is not the selected decision prestate')
            require(hashlib.sha256(canonical(row['source_projection'])).hexdigest() == row['projection_sha256'], 'Projection mismatch')
            for flag in ('selected_identity_matches', 'prefix_wallet_reproduced', 'target_future_projection_mutation_equal'):
                require(row['checks'].get(flag) is True, 'Source replay/mutation proof failed: '+flag)
            for flag in ('raw_replay_matches_indexed_rows', 'admitted_mask_reproduced'):
                require(row.get('source_receipts', {}).get(flag) is True, 'Source replay receipt failed: '+flag)
            next_tensor += 1
        else:
            require(row.get('tensor_row') is None, 'Unavailable source has a tensor index')
    require(next_tensor == manifest['available_contexts'], 'Source coverage count mismatch')
    return manifest, selection, contexts, admitted


def validate_actor(path, source_path, source, contexts, catalogue, bindings):
    manifest = load_manifest(path, 'winprob-ordinary-bc20-export-v1', bindings)
    require(manifest['checkpoint_sha256'] == CHECKPOINT_SHA, 'Wrong BC20 checkpoint')
    bind(manifest['checkpoint_path'], CHECKPOINT_SHA, bindings)
    require(Path(manifest['source_manifest_path']).resolve() == Path(source_path).resolve()
            and manifest['source_manifest_sha256'] == digest(source_path)
            and manifest['source_selection_sha256'] == SELECTION_SHA
            and manifest['source_contexts_sha256'] == source['files']['contexts.jsonl']
            and manifest['opaque_targets_sha256_not_opened'] == source['files']['targets.jsonl'], 'Actor source lineage mismatch')
    for flag in ('target_file_opened', 'targets_analyzed', 'outcomes_analyzed', 'training_performed',
                 'mixture_or_accuracy_computed', 'raw_payload_reads', 'test_payload_reads', 'confirmation_payload_reads'):
        require(manifest.get(flag) is False, 'Actor boundary flag failed: '+flag)
    require(manifest.get('rank_field') == 'group_logits' and manifest.get('decoder_field') == 'group_probabilities',
            'Wrong BC rank/decoder fields')
    directory = Path(path).resolve().parent
    rows = read_rows(directory/'rows.jsonl')
    require(len(rows) == 859 and read_json(directory/'group_main.json') == catalogue['group_main']
            and read_json(directory/'full_to_group.json') == catalogue['full_to_group'], 'Actor group map/ledger changed')
    with np.load(directory/'distribution.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    require(set(arrays) == {'full_probabilities', 'group_probabilities', 'group_logits', 'group_log_probabilities'},
            'Wrong BC probability arrays')
    n = manifest['scored_rows']
    for key, values in arrays.items():
        width = 13166 if key == 'full_probabilities' else 9333
        require(values.dtype == np.float32 and values.shape == (n, width) and np.isfinite(values).all(), 'Malformed BC array: '+key)
        require(manifest['arrays'][key] == {'dtype': 'float32', 'shape': [n, width]}, 'BC array metadata mismatch')
        if 'log' not in key:
            require(np.all(values >= 0) and np.all(abs(values.sum(axis=1, dtype=np.float64)-1) <= 2e-6), 'Unnormalized BC masses')
    next_bc = 0
    for context, row in zip(contexts, rows, strict=True):
        for key in ('sample_id', 'match_id', 'participant_id', 'role', 'patch', 'status'):
            require(row[key] == context[key], 'BC row identity mismatch: '+key)
        require(row['source_tensor_row'] == context.get('tensor_row'), 'BC source index changed')
        if row['bc_status'] == 'scored':
            require(context['status'] == 'available' and row['bc_tensor_row'] == next_bc
                    and row.get('ignored_field_mutation_equal') is True, 'BC index/mutation failure')
            for key, values in arrays.items():
                require(array_digest(values[next_bc]) == row['row_sha256'][key], 'BC probability row digest mismatch')
            projected = np.bincount(catalogue['full_to_group'], weights=arrays['full_probabilities'][next_bc].astype(np.float64), minlength=9333).astype(np.float32)
            require(np.array_equal(projected, arrays['group_probabilities'][next_bc]), 'BC grouped mass omits/changes full-action variants')
            next_bc += 1
        else:
            expected_status = {'available': 'bc_feature_fallback', 'unavailable': 'source_unavailable',
                               'no_admitted_decision': 'no_admitted_decision'}[context['status']]
            require(row['bc_status'] == expected_status and row['bc_tensor_row'] is None, 'Invalid BC fallback status')
    require(next_bc == n == manifest['mutation_checked_rows'] and manifest['mutation_failures'] == 0, 'BC mutation coverage mismatch')
    require(hashlib.sha256(canonical([r['sample_id'] for r in rows])).hexdigest() == manifest['row_order_sha256'], 'BC row order hash mismatch')
    require(hashlib.sha256(canonical([r['sample_id'] for r in rows if r['bc_status'] == 'scored'])).hexdigest()
            == manifest['probability_row_order_sha256'], 'BC matrix order hash mismatch')
    return manifest, rows, arrays


def validate_targets(path, source, contexts, catalogue, full_lookup, bindings):
    """Call only after all fixed predictions, decoding and audits are validated."""
    bind(path, source['files']['targets.jsonl'], bindings)
    rows = read_rows(path)
    require(len(rows) == len(contexts), 'Target ledger coverage mismatch')
    group_lookup = {tuple(map(tuple, basket)): group for group, basket in enumerate(catalogue['group_main'])}
    kinds = {'no_buy': 'observed_no_buy_death', 'ward_only': 'ward_only_purchase',
             'mixed': 'mixed_purchase_with_wards', 'purchase': 'other_purchase'}
    for context, row in zip(contexts, rows, strict=True):
        for key in ('sample_id', 'match_id', 'participant_id', 'role', 'patch', 'selected_decision'):
            require(row[key] == context[key], 'Target identity/status differs from selected prestate: '+key)
        if row['status'] != 'available':
            require(row['status'] == context['status'] and context['status'] != 'available', 'Target status differs from source')
            require(row.get('target') is None, 'Unavailable source cannot have an observed target')
            continue
        require(context['status'] in ('available', 'unavailable'), 'Observed target without a selected decision')
        target = row['target']; truth = target_description(target['label_counts'])
        require(target['kind'] == kinds[truth['kind']] and target['encoding'] == 'item-counts-v1', 'Target kind/count mismatch')
        require(multiset(target['physical_label_counts'], allow_empty=True) == truth['full']
                and multiset(target['main_label_counts'], allow_empty=True) == truth['main'], 'Target multiset projection mismatch')
        require(multiset(target['ward_label_counts'], allow_empty=True) == tuple(x for x in truth['full'] if x[0] in (2055, 772043)),
                'Target ward count mismatch')
        full_id = full_lookup.get(counts_key(target['label_counts']))
        group_id = group_lookup.get(truth['main'])
        require(target['full_action_id'] == full_id and target['main_group_id'] == group_id
                and target['full_catalogue_supported'] is (full_id is not None)
                and target['main_catalogue_supported'] is (group_id is not None), 'Target catalogue support was altered')
    return rows


def validate_audit_entries(entries, options, context, group_main):
    """Verify the exported source-native receipt; never reimplement legality."""
    validate_candidates(options, group_main)
    require(isinstance(entries, list) and len(entries) == len(options), 'Economic audit omitted a candidate')
    result = {}
    wallet = context['source_projection']['gold_est']
    require(not isinstance(wallet, bool) and isinstance(wallet, (int, float)) and math.isfinite(wallet), 'Invalid bound estimated wallet')
    for entry, option in zip(entries, options, strict=True):
        group = option['group']
        require(entry.get('group') == group and entry.get('option_sha256') == hashlib.sha256(canonical(option)).hexdigest(),
                'Economic audit belongs to a different complete option object/order')
        feasible = entry.get('feasible')
        require(type(feasible) is bool, 'Economic audit lacks explicit feasibility')
        cost, sequence = entry.get('incremental_cost'), entry.get('sequence')
        if feasible:
            require(not isinstance(cost, bool) and isinstance(cost, (int, float)) and math.isfinite(cost)
                    and 0 <= cost <= wallet+1e-6 and isinstance(sequence, list), 'Invalid native incremental cost/sequence')
            counts = Counter(); costs = []
            for step in sequence:
                require(type(step.get('item_id')) is int and step['item_id'] > 0, 'Invalid native purchase sequence item')
                value = step.get('cost')
                require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                        'Invalid native purchase sequence incremental cost')
                counts[step['item_id']] += 1; costs.append(value)
            require(counts == Counter({int(k): v for k, v in option['items'].items()})
                    and math.isclose(sum(costs), cost, rel_tol=0, abs_tol=1e-6), 'Native sequence does not reproduce candidate counts/cost')
            require(group != 0 or (cost == 0 and sequence == []), 'SAVE economic audit must have zero cost')
        else:
            require(cost is None and sequence is None, 'Infeasible candidate cannot claim a purchase sequence/cost')
        result[str(group)] = entry
    return result


def validate_reference(path, source_path, actor_path, source, contexts, actor_rows, catalogue, bindings):
    manifest = load_manifest(path, 'winprob-ordinary-reference-v1', bindings)
    require(manifest.get('package_identity_sha256') == M2_IDENTITY and manifest.get('reference_world_patch') == '16.18'
            and manifest.get('native_context_patch') == '16.19', 'Wrong M2 package/economic worlds')
    for name, expected_path in (('source_manifest', source_path), ('actor_manifest', actor_path)):
        require(Path(manifest[name+'_path']).resolve() == Path(expected_path).resolve()
                and manifest[name+'_sha256'] == digest(expected_path), 'Reference upstream artifact differs: '+name)
    require(manifest['source_selection_sha256'] == SELECTION_SHA
            and manifest['source_contexts_sha256'] == source['files']['contexts.jsonl']
            and manifest['opaque_targets_sha256_not_opened'] == source['files']['targets.jsonl'], 'Reference source lineage mismatch')
    for flag in ('target_file_opened', 'targets_analyzed', 'outcomes_analyzed', 'raw_payload_reads',
                 'test_payload_reads', 'confirmation_payload_reads', 'training_performed', 'accuracy_computed'):
        require(manifest.get(flag) is False, 'Reference boundary flag failed: '+flag)
    require(manifest['decoder_contract'] == {'shortlist': 128, 'options': 3, 'fallback': 'native SAVE',
                                           'direct_bc': 'same native options with exported BC group masses'}, 'Native decoder contract changed')
    directory = Path(path).resolve().parent
    require(read_json(directory/'group_main.json') == catalogue['group_main'], 'Reference group map differs from full catalogue')
    rows = read_rows(directory/'rows.jsonl')
    require(len(rows) == 859, 'Reference ledger lost games')
    with np.load(directory/'probabilities.npz', allow_pickle=False) as saved:
        require(saved.files == ['probabilities'], 'Unexpected reference arrays')
        probabilities = saved['probabilities']
    require(probabilities.dtype == np.float32 and probabilities.shape == (manifest['scored_rows'], 9333)
            and np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            and np.all(abs(probabilities.sum(axis=1, dtype=np.float64)-1) <= 2e-6), 'Malformed reference probabilities')
    next_m2 = 0
    for context, actor, row in zip(contexts, actor_rows, rows, strict=True):
        for key in ('sample_id', 'match_id', 'participant_id', 'role', 'patch', 'status'):
            require(row[key] == context[key], 'Reference identity mismatch: '+key)
        require(row['source_tensor_row'] == context.get('tensor_row') and row['projection_sha256'] == context.get('projection_sha256')
                and row['context_timestamp_ms'] == context.get('context_timestamp_ms')
                and row['bc_status'] == actor['bc_status'] and row['bc_tensor_row'] == actor['bc_tensor_row'], 'Reference context/actor join mismatch')
        if context['status'] == 'available':
            require(row['reference_status'] == 'scored' and type(row['m2_tensor_row']) is int and row['m2_tensor_row'] == next_m2,
                    'Available reference row/index missing')
            require(row['probability_row_sha256'] == array_digest(probabilities[next_m2]), 'M2 probability row mismatch')
            for flag in ('zero_residual_native_options_equal', 'whole_distribution_label_mutation_equal',
                         'target_future_native_options_equal', 'target_future_direct_bc_options_equal'):
                require(row.get(flag) is True, 'Reference parity/mutation failure: '+flag)
            for arm, key in (('m2', 'm2_options'), ('direct_bc', 'direct_bc_options')):
                receipt = validate_audit_entries(row['native18_audit'][arm], row[key], context, catalogue['group_main'])
                require(all(x['feasible'] for x in receipt.values()), 'Pinned decoder returned an infeasible candidate')
            if actor['bc_status'] != 'scored':
                require(row['direct_bc_fallback'] == 'original_m2' and row['direct_bc_options'] == row['m2_options'],
                        'Direct BC fallback changed original native options')
            else:
                require(row['direct_bc_fallback'] is None, 'Scored BC row silently fell back')
            next_m2 += 1
        else:
            expected = 'no_admitted_decision' if context['status'] == 'no_admitted_decision' else 'source_unavailable'
            require(row['reference_status'] == expected and row['m2_tensor_row'] is None
                    and row['m2_options'] == row['direct_bc_options'] == [], 'Unavailable reference row has predictions')
    require(next_m2 == manifest['scored_rows'] == manifest['parity_checked_rows'] == manifest['mutation_checked_rows']
            == manifest['native_options_mutation_checked_rows'] == manifest['direct_bc_options_mutation_checked_rows']
            and manifest['zero_residual_native_options_mismatches'] == 0
            and manifest['whole_distribution_label_mutation_equal'] is True
            and manifest['target_future_native_options_equal'] is True
            and manifest['target_future_direct_bc_options_equal'] is True, 'Reference proof coverage incomplete')
    require(hashlib.sha256(canonical([r['sample_id'] for r in rows])).hexdigest() == manifest['row_order_sha256'], 'M2 ledger order digest mismatch')
    return manifest, rows, probabilities


def validate_native19(path, source_path, reference_path, actor_path, source, contexts, reference_rows, catalogue, bindings):
    manifest = load_manifest(path, 'winprob-ordinary-native19-audit-v1', bindings)
    require(manifest.get('native_world_patch') == '16.19' and manifest.get('ddragon_version') == '16.19.1'
            and manifest.get('package_identity_sha256') == M2_IDENTITY, 'Wrong native economic audit world')
    for name, expected_path in (('source_manifest', source_path), ('reference_manifest', reference_path)):
        require(Path(manifest[name+'_path']).resolve() == Path(expected_path).resolve()
                and manifest[name+'_sha256'] == digest(expected_path), 'Native audit upstream artifact differs: '+name)
    require(manifest['actor_manifest_sha256'] == digest(actor_path) and manifest['source_selection_sha256'] == SELECTION_SHA
            and manifest['source_contexts_sha256'] == source['files']['contexts.jsonl']
            and manifest['opaque_targets_sha256_not_opened'] == source['files']['targets.jsonl'], 'Native audit source/actor lineage mismatch')
    for flag in ('target_file_opened', 'targets_analyzed', 'outcomes_analyzed', 'raw_payload_reads',
                 'test_payload_reads', 'confirmation_payload_reads', 'training_performed', 'accuracy_computed',
                 'policy_model_loaded', 'candidates_filtered_or_replaced'):
        require(manifest.get(flag) is False, 'Native audit boundary flag failed: '+flag)
    require(manifest.get('candidate_objects_unchanged') is True and manifest.get('unknown_candidates') == 0,
            'Incomplete native candidate audit')
    rows = read_rows(Path(path).resolve().parent/'rows.jsonl')
    require(len(rows) == 859, 'Native audit lost a selected game')
    counts = Counter(); unique_count = 0; audited = 0; receipts = []
    for context, reference, row in zip(contexts, reference_rows, rows, strict=True):
        for key in ('sample_id', 'match_id', 'participant_id', 'role', 'patch', 'status'):
            require(row[key] == context[key], 'Native audit identity mismatch: '+key)
        require(row['projection_sha256'] == context.get('projection_sha256')
                and row['context_timestamp_ms'] == context.get('context_timestamp_ms')
                and row['reference_status'] == reference['reference_status'], 'Native audit context/reference join mismatch')
        expected = {}
        if context['status'] == 'available':
            audited += 1
            for arm, key in (('m2', 'm2_options'), ('direct_bc', 'direct_bc_options')):
                for rank, option in enumerate(reference[key]):
                    identity = (option['group'], hashlib.sha256(canonical(option)).hexdigest())
                    if identity not in expected:
                        expected[identity] = {'option': option, 'arms': {}}
                    expected[identity]['arms'][arm] = rank
        candidates = row['candidates']
        actual = [(entry['group'], entry['option_sha256']) for entry in candidates]
        require(actual == list(expected), 'Native audit changed candidate union membership/order')
        for entry in candidates:
            identity = (entry['group'], entry['option_sha256'])
            require(entry['option'] == expected[identity]['option'] and entry['arms'] == expected[identity]['arms'],
                    'Native audit changed candidate object or arm rank')
        result = {}
        if context['status'] == 'available':
            for arm, key in (('m2', 'm2_options'), ('direct_bc', 'direct_bc_options')):
                ordered = sorted((e for e in candidates if arm in e['arms']), key=lambda e: e['arms'][arm])
                result[arm] = validate_audit_entries(ordered, reference[key], context, catalogue['group_main'])
                for entry in ordered:
                    counts[arm+'_candidates'] += 1
                    counts[arm+('_feasible' if entry['feasible'] else '_infeasible')] += 1
        unique_count += len(candidates); receipts.append(result)
    require(dict(counts) == manifest['candidate_counts'] and unique_count == manifest['candidate_union_count']
            and audited == manifest['audited_contexts'], 'Native audit counts differ from candidate receipts')
    require(hashlib.sha256(canonical([r['sample_id'] for r in rows])).hexdigest() == manifest['row_order_sha256'], 'Native audit row order mismatch')
    return manifest, rows, receipts


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    path.with_suffix('.sha256').write_text(digest(path)+' '+path.name+'\n', encoding='ascii')


def run_evaluation(source_path, actor_path, reference_path, audit_path, output):
    started = time.monotonic(); output = Path(output).resolve()
    require(not output.exists(), 'Refuse to overwrite ordinary evaluation')
    bindings = {}
    for code_path in (Path(__file__).resolve(), ROOT/'winprob/ordinary_policy_evaluation.py', ROOT/'scripts/evaluate_ordinary_bc20.py'):
        bind(code_path, digest(code_path), bindings)
    source, selection, contexts, admitted = validate_source(source_path, bindings)
    catalogue_path = bind(source['catalogue_path'], CATALOGUE_SHA, bindings)
    catalogue, full_lookup = load_catalogue(catalogue_path)
    actor, actor_rows, arrays = validate_actor(actor_path, source_path, source, contexts, catalogue, bindings)
    reference, reference_rows, probabilities = validate_reference(reference_path, source_path, actor_path, source,
                                                                  contexts, actor_rows, catalogue, bindings)
    audit, audit_rows, audit_receipts = validate_native19(audit_path, source_path, reference_path, actor_path, source,
                                                        contexts, reference_rows, catalogue, bindings)
    elapsed_stages = {name: float(manifest['elapsed_s']) for name, manifest in
                     (('contexts', source), ('actor', actor), ('reference', reference), ('native19', audit))}
    require(all(math.isfinite(x) and x >= 0 for x in elapsed_stages.values()) and sum(elapsed_stages.values()) < 3600,
            'Combined durable computation exceeds the fixed60-minute bound')
    # Complete prediction orders are fixed before the separate purchase file opens.
    predictions = []
    for context, actor_row, ref, native in zip(contexts, actor_rows, reference_rows, audit_receipts, strict=True):
        row = {'sample_id': context['sample_id'], 'match_id': context['match_id'], 'status': context['status']}
        if context['status'] == 'available':
            bc_available = actor_row['bc_status'] == 'scored'
            bc_logits = arrays['group_logits'][actor_row['bc_tensor_row']] if bc_available else None
            ranked, flags = rank_candidates(ref['m2_options'], probabilities[ref['m2_tensor_row']], bc_logits,
                                             native['m2'], group_main=catalogue['group_main'], bc_available=bc_available)
            row.update(ranked=ranked, flags=flags, direct_options=ref['direct_bc_options'],
                       catalogue_top_groups=np.argsort(-bc_logits, kind='stable')[:3].tolist() if bc_available else None,
                       direct_native19_verified=all(x['feasible'] for x in native['direct_bc'].values()))
        predictions.append(row)
    prediction_sha = hashlib.sha256(canonical(predictions)).hexdigest()
    targets = validate_targets(Path(source_path).resolve().parent/'targets.jsonl', source, contexts, catalogue, full_lookup, bindings)
    scored = []
    for context, predicted, observed in zip(contexts, predictions, targets, strict=True):
        if context['status'] == 'no_admitted_decision':
            continue
        if context['status'] == 'available':
            target = observed['target']
            row = score_decision(context['match_id'], target['label_counts'], predicted['ranked'], predicted['flags'],
                                 direct_options=predicted['direct_options'], catalogue_groups=catalogue['group_main'],
                                 catalogue_top_groups=predicted['catalogue_top_groups'])
            row.update(catalogue_main_target_supported=target['main_catalogue_supported'],
                       catalogue_full_target_supported=target['full_catalogue_supported'],
                       direct_native19_verified=predicted['direct_native19_verified'])
        else:
            row = score_decision(context['match_id'], None, None, None, available=False)
        row.update(sample_id=context['sample_id'], source_status=context['status'])
        scored.append(row)
    report = evaluate_scored_decisions(scored, [r['match_id'] for r in contexts if r['selected_decision'] is not None],
                                       [r['match_id'] for r in contexts])
    report['known_target_kind_counts_all_ledger_rows'] = dict(Counter(
        row['target']['kind'] for row in targets if row['status'] == 'available'))
    report['source_unavailable_reasons'] = dict(Counter(
        row.get('unavailable_reason') for row in contexts if row['status'] == 'unavailable'))
    report['known_targets_with_unavailable_contexts'] = sum(
        c['status'] == 'unavailable' and t['status'] == 'available' for c, t in zip(contexts, targets, strict=True))
    require(hashlib.sha256(canonical(predictions)).hexdigest() == prediction_sha, 'Prediction orders changed after target access')
    elapsed_stages['evaluation'] = time.monotonic()-started
    require(sum(elapsed_stages.values()) < 3600, 'Combined durable computation exceeded60minutes')
    producer_gates = {'selection_identity_and_admission_verified': True, 'source_reconstruction_verified': True,
                      'actor_forbidden_field_mutation_verified': True, 'reference_full_distribution_mutation_verified': True,
                      'native_options_parity_and_mutation_verified': True, 'pinned_world_legality_verified': True,
                      'prediction_order_fixed_before_targets': True, 'combined_runtime_under_60_minutes': True}
    report.update(schema='winprob-ordinary-evaluation-v1', state='complete',
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  producer_gates=producer_gates,
                  useful_development_reranking_gate_passed=report['statistical_and_candidate_gate_passed'] and all(producer_gates.values()),
                  prediction_orders_sha256_before_targets=prediction_sha, stage_elapsed_s=elapsed_stages,
                  combined_elapsed_s=sum(elapsed_stages.values()),
                  input_manifests={name: {'path': str(Path(path).resolve()), 'sha256': digest(path)} for name, path in
                                   (('source', source_path), ('actor', actor_path), ('reference', reference_path), ('native19', audit_path))},
                  checkpoint_sha256=CHECKPOINT_SHA, package_identity_sha256=M2_IDENTITY, protocol_sha256=digest(PROTOCOL),
                  source_reference_worlds={'context': '16.19', 'm2_decoder': '16.18', 'separate_candidate_audit': '16.19'},
                  test_payload_reads=False, confirmation_payload_reads=False, training_performed=False, outcomes_analyzed=False,
                  promotion_authorized=False, breakthrough_established=False,
                  limitations=['All859 games are exposed development; BC20 was selected from an earlier revision challenge.',
                      'Purchase agreement is not causal purchase value, increased win rate, or improved human decisions.',
                      'Legality/cost depend on reconstructed inventory, estimated wallet, and legacy role/quest assumptions.',
                      'The primary main-basket metric preserves ward-only group0 credit; typed and complete metrics are separate.',
                      'Secondary/direct-policy results cannot rescue a failed primary reranking gate.'])
    admission.verify_manifest_sources(admitted)
    for path, expected in bindings.items():
        require(digest(path) == expected, 'Bound artifact changed before publication: '+path)
    report['bindings_sha256'] = bindings
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'predictions.json', predictions)
    write_json(output/'scored_rows.json', scored)
    report['files'] = {name: digest(output/name) for name in ('predictions.json', 'scored_rows.json')}
    write_json(output/'report.json', report)
    return report
