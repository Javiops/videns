"""Frozen BC20 ordinary-context inference v2: float64 normalization.

Full-action probabilities retain ward variants. Group ranking uses finite
logsumexp masses, including when float32 probability exports underflow.
Model logits/weights/features and the fixed 2e-6 export guard are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from winprob import admission
from winprob.full_action_catalogue import load_catalogue
from winprob.iql_features import PublicFeatureCompiler
from winprob.iql_models import FullCategoricalActor, collate_public_boards

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = '57fca2f787aaff6dce43bfd729e34e90464de2f536abdf4ba534d5d01e5b1f0d'
TRAINING_REPORT_SHA256 = 'c812ef7d19909f9366354e82f79e936e3b1127b95a4b045f446c07ff7a146a4d'
CATALOGUE_SHA256 = '14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8'
SCHEMA = 'winprob-ordinary-bc20-export-v1'
SOURCE_SCHEMA = 'winprob-ordinary-contexts-v1'
PROTOCOL = ROOT/'docs/bc20_ordinary_eval_protocol_v1.md'


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): value.update(part)
    return value.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_sidecar(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text(encoding='ascii').split() != [digest(path), path.name]:
        raise ValueError('artifact sidecar mismatch: '+str(path))


def revalidate(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected: raise ValueError('bound artifact changed: '+str(path))


class GroupProjection:
    """Fixed complete-action partition, with stable log masses and actual sums."""
    def __init__(self, full_to_group, group_count):
        mapping = np.asarray(full_to_group)
        if (mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer)
                or type(group_count) is not int or group_count < 1
                or not len(mapping) or np.any(mapping < 0) or np.any(mapping >= group_count)
                or set(mapping.tolist()) != set(range(group_count))):
            raise ValueError('every fixed main group needs complete-action support')
        self.mapping = torch.as_tensor(mapping.astype(np.int64))
        members = [np.flatnonzero(mapping == group) for group in range(group_count)]
        width = max(map(len, members))
        indices = np.zeros((group_count, width), dtype=np.int64)
        valid = np.zeros((group_count, width), dtype=bool)
        for group, ids in enumerate(members):
            indices[group, :len(ids)] = ids; valid[group, :len(ids)] = True
        self.indices, self.valid = torch.from_numpy(indices), torch.from_numpy(valid)
        self.group_count = group_count

    def __call__(self, logits):
        if (logits.device.type != 'cpu' or logits.dtype != torch.float32 or logits.ndim != 2
                or logits.shape[1] != len(self.mapping) or not bool(torch.isfinite(logits).all())):
            raise ValueError('finite CPU float32 full-catalogue logits required')
        # The actor still emits the same float32 logits. Across 13,166 classes,
        # the float32 softmax kernel can itself accumulate >2e-6 mass error.
        # Normalize in float64, then round only the exported probabilities.
        logits64 = logits.to(torch.float64)
        full64 = torch.softmax(logits64, dim=1)
        full = full64.to(torch.float32)
        # Group the actual exported full masses, using float64 accumulation;
        # the final float32 cast is the only grouping-rounding operation.
        group64 = torch.zeros((len(logits), self.group_count), dtype=torch.float64)
        group64.scatter_add_(1, self.mapping[None, :].expand(len(logits), -1), full.to(torch.float64))
        group = group64.to(torch.float32)
        grouped_logits64 = torch.logsumexp(logits64[:, self.indices].masked_fill(~self.valid[None, :, :], -torch.inf), dim=2)
        log_group64 = grouped_logits64-torch.logsumexp(logits64, dim=1, keepdim=True)
        grouped_logits = grouped_logits64.to(torch.float32)
        log_group = log_group64.to(torch.float32)
        if (not bool(torch.isfinite(grouped_logits).all()) or not bool(torch.isfinite(log_group).all())
                or not torch.allclose(full64.sum(1), torch.ones(len(logits),dtype=torch.float64), atol=1e-12, rtol=0)
                or not torch.allclose(full.to(torch.float64).sum(1), torch.ones(len(logits),dtype=torch.float64), atol=2e-6, rtol=0)
                or not torch.allclose(group.to(torch.float64).sum(1), torch.ones(len(logits),dtype=torch.float64), atol=2e-6, rtol=0)
                or not torch.allclose(full.sum(1), torch.ones(len(logits)), atol=2e-6, rtol=0)
                or not torch.allclose(group.sum(1), torch.ones(len(logits)), atol=2e-6, rtol=0)):
            raise ValueError('invalid full or grouped distribution')
        return {'full_probabilities': full.numpy(), 'group_probabilities': group.numpy(),
                'group_log_probabilities': log_group.numpy(), 'group_logits': grouped_logits.numpy()}


def mutate_ignored_fields(state):
    """Preserve all allowed current features; poison only forbidden metadata."""
    changed = deepcopy(state)
    changed.update({'target': {'label_counts': {'999999': 1}}, 'label_counts': {'2055': 77},
        'action': 999999, 'outcome': 'changed', 'win': 'changed', 'role': 'changed',
        'final_role': 'changed', 'gold': -987654321, 'total_gold': 987654321,
        'gold_after_complete': -987654321, 'can_complete': 'changed',
        'next_state': {'win': True, 'total_gold': 987654321}, 'future_gold': -987654321})
    for other in changed.get('others', []):
        other.update({'gold': -987654321, 'total_gold': 987654321, 'role': 'changed',
                      'win': 'changed', 'target': {'label_counts': {'999999': 1}}})
    return changed


def compiled_equal(first, second):
    if first.keys() != second.keys(): return False
    return all(first[key] == second[key] if key == 'inventory_bags'
               else np.array_equal(first[key], second[key]) for key in first)


def validate_context_ledger(manifest, selection, rows, admitted):
    """Validate identity/selection metadata before compiling any state."""
    expected = {r['match_id'] for r in admitted['games'] if r['role'] == 'development' and r['patch'] == '16.19'}
    entries = selection.get('entries', [])
    if (manifest.get('schema') != SOURCE_SCHEMA or manifest.get('state') != 'complete'
            or manifest.get('smoke') is not False or manifest.get('full_cohort_complete') is not True
            or manifest.get('selection_count') != 859 or len(expected) != 859
            or len(entries) != 859 or len(rows) != 859
            or {r['match_id'] for r in rows} != expected
            or len({r['sample_id'] for r in rows}) != 859):
        raise ValueError('ordinary ledger must contain the exact admitted 859-game population')
    next_tensor = 0
    for ordinal, (entry, row) in enumerate(zip(entries, rows, strict=True)):
        if any(key in row for key in ('target','label_counts','win','outcome','full_action_id','main_group_id')):
            raise ValueError('target or outcome metadata is not permitted in the context-only ledger')
        mid = row['match_id']
        focal = 1+int.from_bytes(hashlib.sha256(('bc20-ordinary-v1/focal/'+mid).encode()).digest()[:8], 'big')%10
        if (type(entry.get('ordinal')) is not int or entry.get('ordinal') != ordinal or entry.get('match_id') != mid
                or type(entry.get('pid')) is not int or entry.get('pid') != focal
                or type(row.get('participant_id')) is not int or row.get('participant_id') != focal
                or row.get('role') != 'development' or row.get('patch') != '16.19'):
            raise ValueError('ordinary selection identity or fixed focal player changed')
        choice = entry.get('selected_decision')
        if choice is None:
            sample = f'{mid}|{focal}|none'
            if entry.get('status') != 'no_admitted_decision' or row.get('status') != 'no_admitted_decision':
                raise ValueError('empty focal selection was replaced')
        else:
            if (not isinstance(choice, dict) or any(type(choice.get(k)) is not int for k in ('event_index','pid','ts'))
                    or choice['pid'] != focal or choice['event_index'] < 0 or choice['ts'] < 0
                    or entry.get('status') != 'selected' or row.get('status') not in ('available','unavailable')):
                raise ValueError('malformed selected visit identity')
            sample = f"{mid}|{focal}|{choice['event_index']}|{choice['ts']}"
        if row.get('sample_id') != sample or entry.get('sample_id') != sample:
            raise ValueError('ordinary sample identity changed')
        if canonical_bytes(row.get('selected_decision')) != canonical_bytes(choice):
            raise ValueError('context selected decision differs from the frozen selection ledger')
        if row['status'] == 'available':
            if (type(row.get('tensor_row')) is not int or row.get('tensor_row') != next_tensor or not isinstance(row.get('prestate'), dict)
                    or not isinstance(row.get('source_projection'), dict)
                    or hashlib.sha256(canonical_bytes(row['source_projection'])).hexdigest() != row.get('projection_sha256')):
                raise ValueError('available state projection/row binding failed')
            if (type(row.get('context_timestamp_ms')) is not int
                    or row['context_timestamp_ms'] != choice['ts']
                    or row['prestate'].get('ts') != choice['ts']
                    or row['source_projection'].get('ts') != choice['ts']):
                raise ValueError('actor prestate does not represent the selected decision timestamp')
            next_tensor += 1
        elif row.get('tensor_row') is not None:
            raise ValueError('unavailable state has a tensor row')
    return next_tensor


def load_contexts(path):
    path = Path(path).resolve(strict=True); check_sidecar(path)
    manifest = json_read(path); directory = path.parent
    if (manifest.get('schema') != SOURCE_SCHEMA or manifest.get('state') != 'complete'
            or manifest.get('smoke') is not False or manifest.get('full_cohort_complete') is not True):
        raise ValueError('full-cohort completed ordinary context artifact required; smoke scoring is forbidden')
    for flag in ('test_payload_reads','confirmation_payload_reads','outcomes_exported','outcomes_analyzed'):
        if manifest.get(flag) is not False: raise ValueError('source boundary flag failed: '+flag)
    bindings = {str(path): digest(path), str(path.with_suffix('.sha256')): digest(path.with_suffix('.sha256'))}
    for name in ('selection.json','contexts.jsonl'):
        if manifest.get('files', {}).get(name) != digest(directory/name): raise ValueError('ordinary source file digest mismatch')
        bindings[str(directory/name)] = manifest['files'][name]
    target_digest = manifest.get('files', {}).get('targets.jsonl')
    if not isinstance(target_digest, str) or len(target_digest) != 64:
        raise ValueError('separate opaque target identity required')
    # Deliberately do not open or hash targets.jsonl. The evaluator verifies it.
    for key in ('admission','protocol','static_export'):
        source = Path(manifest[key+'_path']).resolve(strict=True)
        expected = manifest[key+'_sha256']
        if digest(source) != expected: raise ValueError('ordinary source metadata changed: '+key)
        bindings[str(source)] = expected
    if Path(manifest['protocol_path']).resolve() != PROTOCOL.resolve():
        raise ValueError('ordinary source refers to a different protocol')
    extra = manifest.get('bindings', {})
    if not isinstance(extra, dict) or not extra:
        raise ValueError('source sampler/reconstruction metadata/code bindings required')
    target_path = (directory/'targets.jsonl').resolve()
    for source, expected in extra.items():
        source = Path(source).resolve(strict=True)
        if source == target_path or 'raw_match_v5' in source.parts or source.suffix in ('.gz','.db','.sqlite','.npz','.npy'):
            raise ValueError('only source metadata/code bindings may be opened by the actor')
        bindings[str(source)] = expected
    revalidate(bindings)
    admitted = json_read(manifest['admission_path']); admission.verify_manifest_sources(admitted)
    rows = [json.loads(line) for line in (directory/'contexts.jsonl').read_text(encoding='utf-8').splitlines()]
    selection = json_read(directory/'selection.json')
    available = validate_context_ledger(manifest, selection, rows, admitted)
    return manifest, rows, admitted, bindings, available


def load_frozen_actor(training_run, dataset_manifest):
    run = Path(training_run).resolve(strict=True); dataset_path = Path(dataset_manifest).resolve(strict=True)
    checkpoint = run/'actor_bc20.pt'; report_path = run/'report.json'
    if digest(checkpoint) != CHECKPOINT_SHA256 or digest(report_path) != TRAINING_REPORT_SHA256:
        raise ValueError('only the frozen completed BC20 checkpoint/report is permitted')
    report = json_read(report_path); dataset = json_read(dataset_path)
    if str(torch.__version__) != report['runtime_versions']['torch']:
        raise ValueError('actor Torch runtime differs from the frozen training runtime')
    if report['bindings_sha256'].get(str(dataset_path)) != digest(dataset_path):
        raise ValueError('dataset lineage is not the frozen actor training world')
    bindings = {str(checkpoint): CHECKPOINT_SHA256, str(report_path): TRAINING_REPORT_SHA256,
                str(dataset_path): digest(dataset_path)}
    snapshot_path = run/'code_snapshot/manifest.json'; snapshot = json_read(snapshot_path)
    bindings[str(snapshot_path)] = digest(snapshot_path)
    for relative, expected in snapshot['files'].items():
        saved = run/'code_snapshot'/relative
        if digest(saved) != expected: raise ValueError('frozen actor source snapshot changed')
        bindings[str(saved)] = expected
    for relative in ('winprob/iql_models.py','winprob/iql_features.py','winprob/ipo_residual.py'):
        current = ROOT/relative
        if digest(current) != snapshot['files'][relative]: raise ValueError('actor implementation differs from frozen snapshot')
        bindings[str(current)] = snapshot['files'][relative]
    feature_path = dataset_path.parent/dataset['feature_metadata_file']
    if digest(feature_path) != dataset['files'][feature_path.name]: raise ValueError('actor feature vocabulary changed')
    static_path = Path(dataset['static_export_path']).resolve(strict=True)
    if digest(static_path) != dataset['static_export_sha256']: raise ValueError('actor static export changed')
    static = json_read(static_path); check_sidecar(static_path)
    compiler = PublicFeatureCompiler(static['patches'])
    if json.loads(json.dumps(compiler.metadata())) != json_read(feature_path):
        raise ValueError('actor compiler vocabulary/units differ from training metadata')
    catalogue_path = Path(dataset['catalogue_path']).resolve(strict=True)
    if digest(catalogue_path) != CATALOGUE_SHA256: raise ValueError('actor catalogue changed')
    catalogue, _ = load_catalogue(catalogue_path)
    for source in (feature_path, static_path, static_path.with_suffix('.sha256'), catalogue_path, catalogue_path.with_suffix('.sha256')):
        bindings[str(source)] = digest(source)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('epochs') != 20: raise ValueError('frozen actor is not the 20-epoch checkpoint')
    actor = FullCategoricalActor(len(compiler.champion_index), len(compiler.item_index))
    actor.load_state_dict(saved['actor'], strict=True); actor.eval(); actor.requires_grad_(False)
    del saved
    return actor, compiler, catalogue, bindings, static_path


def score_contexts(actor, compiler, rows, full_to_group, group_count, *, batch_size=32, guard=lambda: None):
    """All-row ledger; every successful feature row receives a full mutation test."""
    if batch_size not in (32,64): raise ValueError('fixed small CPU batch size required')
    if any(parameter.device.type != 'cpu' for parameter in actor.parameters()): raise ValueError('CPU-only scorer')
    actor.eval(); projection = GroupProjection(full_to_group, group_count)
    ledger, features, changed_features, index = [], [], [], []
    for ordinal,row in enumerate(rows):
        if ordinal%64 == 0: guard()
        result = {key: row[key] for key in ('sample_id','match_id','participant_id','role','patch','status')}
        result.update({'source_tensor_row':row.get('tensor_row'),'bc_tensor_row':None,
                       'bc_status':'source_unavailable','fallback':'original_m2'})
        if row['status'] == 'no_admitted_decision':
            result.update(bc_status='no_admitted_decision',fallback='no_decision')
        elif row['status'] == 'available':
            try:
                state = compiler.state(row['prestate'],row['patch'])
            except (ValueError,KeyError,TypeError) as error:
                result.update(bc_status='bc_feature_fallback',reason=str(error))
            else:
                compatible = compiler.state(row['source_projection'],row['patch'])
                changed = compiler.state(mutate_ignored_fields(row['prestate']),row['patch'])
                if not compiled_equal(state,compatible): raise ValueError('source prestate/M2 projection public features differ')
                if not compiled_equal(state,changed): raise ValueError('forbidden label/future field affects actor features')
                result.update(bc_status='scored',fallback=None,bc_tensor_row=len(features))
                features.append(state);changed_features.append(changed);index.append(ordinal)
        ledger.append(result)
    pieces = {key:[] for key in ('full_probabilities','group_probabilities','group_log_probabilities','group_logits')}
    with torch.inference_mode():
        for start in range(0,len(features),batch_size):
            guard(); end=min(start+batch_size,len(features))
            logits=actor(collate_public_boards(features[start:end])); mutated=actor(collate_public_boards(changed_features[start:end]))
            if not torch.equal(logits,mutated): raise ValueError('full actor distribution changed under forbidden-field mutation')
            arrays=projection(logits)
            mutated_arrays=projection(mutated)
            if any(not np.array_equal(values,mutated_arrays[key]) for key,values in arrays.items()):
                raise ValueError('exported full/group masses changed under forbidden-field mutation')
            for key,values in arrays.items(): pieces[key].append(values)
            for local,ordinal in enumerate(index[start:end]):
                ledger[ordinal]['ignored_field_mutation_equal']=True
                ledger[ordinal]['row_sha256']={key:array_digest(values[local]) for key,values in arrays.items()}
    arrays={key:(np.concatenate(value,axis=0) if value else np.empty((0,len(full_to_group) if key=='full_probabilities' else group_count),dtype=np.float32))
            for key,value in pieces.items()}
    return ledger,arrays,{'ledger_rows':len(rows),'scored_rows':len(features),
        'mutation_checked_rows':len(features),'mutation_failures':0,
        'feature_fallback_rows':sum(row['bc_status']=='bc_feature_fallback' for row in ledger),
        'source_unavailable_rows':sum(row['bc_status']=='source_unavailable' for row in ledger),
        'no_decision_rows':sum(row['bc_status']=='no_admitted_decision' for row in ledger)}
