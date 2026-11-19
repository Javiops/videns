"""Public state/rune/catalogue input preparation, no target/teacher payloads."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.history_probe_data import BaseCache, write_json, require
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.full_action_catalogue import load_catalogue
from winprob.ipo_residual import action_features

# Public fixed vocabularies, matching causal_student_v1; no runtime/weights import.
KEYSTONES = (8005, 8008, 8010, 8021, 8112, 8128, 9923, 8214, 8229, 8230, 8437, 8439, 8465, 8351, 8360, 8369, 8992)
STYLES = (8000, 8100, 8200, 8300, 8400)
PROTOCOL = ROOT/'docs/causal_student_input_protocol_v1.md'
SELECTION = ROOT/'data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json'
CONTEXT = ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-contexts-20261001-v1/manifest.json'
STATIC = ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'


def run(output):
    start = time.monotonic(); output = output.resolve(); require(not output.exists(), 'new immutable student inputs required')
    code = {str(p.resolve()): file_hash(p) for p in (Path(__file__), PROTOCOL, ROOT/'docs/causal_student_design_v1.md',
        ROOT/'winprob/causal_student_v1.py', ROOT/'winprob/history_probe_data.py', ROOT/'winprob/ipo_residual.py')}
    module = ast.parse((ROOT/'winprob/causal_student_v1.py').read_text(encoding='utf-8'))
    literals = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('KEYSTONES', 'STYLES')}
    require(literals == {'KEYSTONES': KEYSTONES, 'STYLES': STYLES}, 'prototype/public rune vocabularies differ')
    base = BaseCache(ROOT/'data/runs/iql-cache-20260930-v1/cache', ROOT/'docs/history_probe_protocol_v1.md')
    bindings = dict(base.bindings)
    sm, sh = bound_json(SELECTION, bindings); require(sh == '979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6', 'fixed selection changed')
    sp = SELECTION.parent/'selection.json'; require(file_hash(sp) == sm['files'][sp.name], 'selection entries changed')
    bindings[str(sp)] = sm['files'][sp.name]; selection = json.loads(sp.read_text(encoding='utf-8')); entries = selection['entries']
    require(len(entries) == 56288 and len({e['match_id'] for e in entries}) == 3518
            and all(e['patch'] == '16.17' for e in entries) and sum(e['is_original_focal'] for e in entries) == 2900, 'fixed learner population changed')
    for e in entries:
        episode = base.episodes[e['episode_index']]; index = e['state_index']
        require(episode['match_id'] == e['match_id'] and episode['participant_id'] == e['participant_id']
                and episode['first_state']+e['source_node_ordinal'] == index
                and base.array('timestamp_ms')[index] == e['timestamp_ms'] and base.array('fold')[index] == e['fold'], 'selected state identity changed')
    cm, ch = bound_json(CONTEXT, bindings)
    require(ch == 'f45e22753294b0d349c1c58bf6db4a23575352d5a9fc2ae5af1bf4d6b65bc25a'
            and cm['state'] == 'complete' and cm['selection_manifest_sha256'] == sh, 'versioned context changed')
    cp = CONTEXT.parent/'contexts.jsonl.gz'; require(file_hash(cp) == cm['files'][cp.name], 'context payload changed')
    bindings[str(cp)] = cm['files'][cp.name]
    encoded = np.zeros((56288, 2), dtype=np.int16); raw_runes = np.zeros((56288, 2), dtype=np.int32)
    seen = np.zeros(56288, dtype=bool); player_values = {}; missing = Counter(); unknown = Counter()
    vocabulary = ({v: i+1 for i, v in enumerate(KEYSTONES)}, {v: i+1 for i, v in enumerate(STYLES)})
    with gzip.open(cp, 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(time.monotonic()-start < 900, 'fixed15minute preparation bound')
            row = json.loads(line); i = row['ordinal']; e = entries[i]
            require(not seen[i] and all(row[k] == v for k, v in e.items()), 'source context identity changed')
            values = []
            for col, key in enumerate(('keystone_id', 'sub_style')):
                value = row['source_projection'].get(key)
                value = 0 if value is None else value
                require(type(value) is int and value >= 0, 'invalid selected rune ID')
                values.append(value); raw_runes[i, col] = value; encoded[i, col] = vocabulary[col].get(value, 0)
                missing[key] += int(value == 0)
                if value and value not in vocabulary[col]: unknown[key+'|'+str(value)] += 1
            player = (e['match_id'], e['participant_id'])
            require(player not in player_values or player_values[player] == tuple(values), 'selected rune configuration changes within player-game')
            player_values[player] = tuple(values); seen[i] = True
    require(seen.all(), 'incomplete rune state coverage')
    meta = base.json_member(base.manifest['feature_metadata_file']); item_index = meta['item_index']
    catalogue, _ = load_catalogue(base.manifest['catalogue_path'])
    static, st = bound_json(STATIC, bindings)
    require(st == '5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0', 'public static definitions changed')
    groups = catalogue['group_main']; require(len(groups) == 9333 and groups[0] == [], 'main catalogue changed')
    width = max(map(len, groups)); require(width <= 6, 'main-basket width exceeds fixed six-copy domain')
    ids = np.zeros((9333, width), dtype=np.int32); counts = np.zeros_like(ids, dtype=np.float32)
    for row, bag in enumerate(groups):
        require(sum(c for _, c in bag) <= 6, 'main-basket copy overflow')
        for col, (item, count) in enumerate(bag):
            require(str(item) in item_index, 'catalogue item absent from fixed public vocabulary')
            ids[row, col] = item_index[str(item)]; counts[row, col] = count
    attributes = action_features(groups, static['patches']['16.18']['items']).astype(np.float32)
    empty = np.array([not bag for bag in groups], dtype=bool)
    require(attributes.shape == (9333, 23), 'public action schema changed'); attributes[:, -1] = empty.astype(np.float32)
    output.mkdir(parents=True)
    arrays = {'rune_indices': encoded, 'raw_rune_ids': raw_runes,
        'state_indices': np.array([e['state_index'] for e in entries], dtype=np.int64),
        'folds': np.array([e['fold'] for e in entries], dtype=np.int8),
        'focal_ordinals': np.flatnonzero([e['is_original_focal'] for e in entries]),
        'action_item_indices': ids, 'action_item_counts': counts, 'action_attributes': attributes, 'empty_main_mask': empty}
    for name, value in arrays.items(): np.save(output/(name+'.npy'), value, allow_pickle=False)
    write_json(output/'selection.json', selection)
    base.revalidate()
    for path, sha in {**bindings, **code}.items(): require(file_hash(path) == sha, 'input preparation binding changed')
    report = {'schema': 'causal-student-inputs-v1', 'state': 'complete', 'states': 56288, 'games': 3518, 'focals': 2900,
        'selected_player_games': len(player_values), 'runes_constant_within_selected_player_game': True,
        'missing_rune_values': dict(missing), 'unknown_nonzero_rune_values': dict(unknown),
        'rune_vocabularies': {'keystone': list(KEYSTONES), 'secondary_style': list(STYLES), 'unknown_index': 0},
        'action_count': 9333, 'action_width': width, 'empty_main_count': int(empty.sum()),
        'files': {p.name: file_hash(p) for p in output.iterdir() if p.suffix in ('.npy', '.json')},
        'bindings_sha256': {**bindings, **base.bindings}, 'code_sha256': code,
        'purchase_targets_opened': False, 'teacher_probability_payload_opened': False,
        'legacy_roles_indexed': False, 'source_context_json_deserialized': True,
        'raw_or_database_reads': False, 'test_payload_reads': False, 'confirmation_payload_reads': False,
        'training_performed': False, 'accuracy_computed': False, 'elapsed_s': time.monotonic()-start}
    write_json(output/'manifest.json', report)
    print(json.dumps({k: report[k] for k in ('state', 'states', 'focals', 'missing_rune_values', 'unknown_nonzero_rune_values', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
