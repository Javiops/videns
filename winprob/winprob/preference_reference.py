"""Independent consumer of source-owned M2 preference reference artifacts.

Only versioned files cross the project boundary; source inference is never
imported. No function here reads a raw match/timeline or fits a policy.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from winprob import admission

M2_IDENTITY = 'ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9'
ROLE_MAP = {'fit': 'train', 'calibration': 'calibration', 'development': 'development'}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024*1024):
            result.update(chunk)
    return result.hexdigest()


def value_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def validate_header(manifest):
    if (manifest.get('schema') != 'winprob-m2-preference-v1' or manifest.get('state') != 'complete'
            or manifest.get('package_identity_sha256') != M2_IDENTITY
            or manifest.get('group_count') != 9333 or type(manifest.get('pairs')) is not int
            or manifest['pairs'] < 1):
        raise ValueError('wrong or incomplete frozen M2 reference')
    if any(manifest.get(key) is not False for key in
           ('test_payload_reads', 'confirmation_payload_reads', 'training_performed')):
        raise ValueError('forbidden or undeclared source access')
    if (manifest.get('zero_residual_native_options_mismatches') != 0
            or manifest.get('parity_checked_pairs') != manifest['pairs']
            or manifest.get('mutation_checked_pairs') != manifest['pairs']
            or manifest.get('whole_distribution_label_mutation_equal') is not True):
        raise ValueError('real reference identity/mutation checks lack full coverage')


def validate_rows(rows, admitted):
    seen_pairs, seen_contexts = set(), set()
    for row in rows:
        entry = admitted.get(row['match_id'])
        if (entry is None or row.get('role') not in ROLE_MAP
                or entry.get('role') != ROLE_MAP[row['role']]
                or entry.get('historical_validation') is not False
                or row.get('patch') != entry.get('patch')
                or (row['role'] == 'development' and row['patch'] != '16.19')):
            raise ValueError('reference row not in the permitted role/patch')
        ta, tu, tb = (row.get(key) for key in ('context_timestamp_ms', 'undo_timestamp_ms', 'decision_timestamp_ms'))
        if (any(type(t) is not int for t in (ta, tu, tb)) or not ta < tu < tb
                or row['source_projection'].get('ts') != ta):
            raise ValueError('canonical pre-A/undo/B boundary changed')
        identity = {'match_id': row['match_id'], 'participant_id': row['participant_id'],
                    'feature_cutoff_ms': ta, 'undo_ms': tu, 'decision_timestamp_ms': tb,
                    'action_a': row['action_a'], 'action_b': row['action_b']}
        expected_pair = '|'.join(str(identity[key]) for key in identity)
        if row.get('pair_id') != expected_pair or row.get('context_id') != value_hash(identity):
            raise ValueError('canonical pair/context identity changed')
        if row.get('pair_id') in seen_pairs or row.get('context_id') in seen_contexts:
            raise ValueError('duplicate pair or context')
        if not row.get('pair_id') or not row.get('context_id'):
            raise ValueError('missing stable pair/context identity')
        seen_pairs.add(row['pair_id']); seen_contexts.add(row['context_id'])
        if value_hash(row['source_projection']) != row['projection_sha256']:
            raise ValueError('canonical pre-A projection changed')
        if not row['context_timestamp_ms'] < row['decision_timestamp_ms']:
            raise ValueError('context is not earlier than the B certificate')
        weight = row.get('random_focal_pair_weight')
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 < weight <= .1:
            raise ValueError('missing positive focal-player pair weight')
        if row['action_a'] == row['action_b']:
            raise ValueError('same-item roundtrip cannot enter preference reference')


def validate_arrays(rows, probabilities, groups):
    if (probabilities.dtype != np.float32 or probabilities.shape != (len(rows), 9333)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1, dtype=np.float64), 1, atol=2e-6, rtol=0)):
        raise ValueError('invalid complete float32 M2 probability matrix')
    if len(groups) != 9333 or groups[0] != []:
        raise ValueError('wrong canonical full action catalogue')
    usable = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        if (row.get('tensor_row') != i or row.get('probability_row_sha256') !=
                hashlib.sha256(probabilities[i].tobytes(order='C')).hexdigest()):
            raise ValueError('full probability tensor row identity changed')
        indexes = [row.get('group_a'), row.get('group_b')]
        if any(index is None for index in indexes):
            continue
        for label, group in zip(('a', 'b'), indexes):
            if type(group) is not int or not 0 <= group < 9333:
                raise ValueError('group index outside frozen catalogue')
            if groups[group] != [[row['action_'+label], 1]]:
                raise ValueError('singleton pair action differs from full-catalogue mapping')
            declared = row.get('probability_'+label)
            if declared is None or float(probabilities[i, group]) != declared:
                raise ValueError('pair mass differs from its full-distribution tensor row')
        usable[i] = bool(probabilities[i, indexes[0]] > 0 and probabilities[i, indexes[1]] > 0)
    return usable


def load_reference(directory, admission_path):
    directory, admission_path = Path(directory).resolve(strict=True), Path(admission_path).resolve(strict=True)
    for root in (directory, *directory.parents):
        if (root/'QUARANTINED.md').exists():
            raise ValueError('quarantined reference')
    manifest_path = directory/'manifest.json'
    if file_hash(manifest_path) != (directory/'manifest.sha256').read_text().split()[0]:
        raise ValueError('reference manifest digest mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_header(manifest)
    needed = {'contexts.jsonl', 'probabilities.npz', 'group_main.json', 'exclusions.jsonl'}
    if not needed <= set(manifest['files']):
        raise ValueError('reference is missing required bound files')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_hash(directory/name) != expected:
            raise ValueError('reference file name or digest mismatch')
    if file_hash(admission_path) != admission_path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('admission digest mismatch')
    admitted_manifest = json.loads(admission_path.read_text(encoding='utf-8'))
    admission.verify_manifest_sources(admitted_manifest)
    admitted = {row['match_id']: row for row in admitted_manifest['games']}
    rows = [json.loads(line) for line in (directory/'contexts.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != manifest['pairs']:
        raise ValueError('reference tensor row count not bound to manifest')
    validate_rows(rows, admitted)
    order_hash = hashlib.sha256(('\n'.join(row['pair_id'] for row in rows)+'\n').encode()).hexdigest()
    if order_hash != manifest.get('pair_order_sha256'):
        raise ValueError('ordered pair-to-tensor identity changed')
    groups = json.loads((directory/'group_main.json').read_text(encoding='utf-8'))
    with np.load(directory/'probabilities.npz', allow_pickle=False) as arrays:
        if arrays.files != ['probabilities']:
            raise ValueError('unexpected reference arrays')
        probabilities = arrays['probabilities']
    usable = validate_arrays(rows, probabilities, groups)
    counts = {}
    for role in ROLE_MAP:
        indexes = [i for i, row in enumerate(rows) if usable[i] and row['role'] == role]
        counts[role] = {'pairs': len(indexes), 'games': len({rows[i]['match_id'] for i in indexes})}
    gate = counts['fit']['games'] >= 100 and counts['development']['games'] >= 50 and counts['development']['pairs'] >= 100
    report = {'schema': 'winprob-reference-consumer-audit-v1', 'manifest_sha256': file_hash(manifest_path),
              'rows': len(rows), 'usable_rows': int(usable.sum()), 'role_counts_after_mapping_support': counts,
              'minimum_population_gate': gate, 'role_counts_before_mapping_support': dict(Counter(row['role'] for row in rows)),
              'training_performed': False, 'raw_sources_reopened': False}
    return rows, probabilities, groups, usable, report
