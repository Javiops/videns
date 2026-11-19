"""Artifact-only inputs for the fixed native-M2 correction experiment.

Admission precedes payload reads. Inputs are validated and frozen before the
separate selected-purchase ledger can open. No source runtime is imported.
"""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from winprob.full_action_catalogue import load_catalogue, counts_key, WARDS
from winprob.iql_dataset_contract import bound_json, current_reservations, file_hash, json_file
from winprob.recipe_transition_v1 import ChoiceInputs, context_features, transition_features
from winprob.recipe_transition_probe_v1 import shuffle_for_entry

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent/'build_tracker'
PROTOCOL = ROOT/'docs/recipe_transition_probe_protocol_v1.md'
PROTOCOL_SHA = 'e87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e'
SELECTION = ROOT/'data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json'
SELECTION_SHA = '979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6'
CATALOGUE = SOURCE_ROOT/'data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1/catalogue.json'
CATALOGUE_SHA = '14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8'
STATIC = ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'
STATIC_SHA = '5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0'
BASE = ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_options(row, groups):
    ids = row['group_ids']; options = row['m2_options']; receipts = row['receipts']
    require(1 <= len(ids) <= 3 and len(set(ids)) == len(ids) and len(options) == len(receipts) == len(ids), 'incomplete/repeated native options')
    for gid, option, receipt in zip(ids, options, receipts, strict=True):
        require(type(gid) is int and 0 <= gid < len(groups), 'invalid group ID')
        expected = {str(i): n for i, n in groups[gid]}
        require(option == {'group': gid, 'items': expected, 'kind': 'save' if gid == 0 else 'buy', 'legal': True}, 'option differs from the native main multiset')
        require(Counter(receipt['action']) == Counter({i: n for i, n in groups[gid]}), 'receipt action differs from its native option')
        require(receipt['before'] == row['residual_public']['inventory']
                and receipt['wallet'] == row['residual_public']['gold_est'], 'receipt predecision state changed')
    return ids


def labels_for_action(action, catalogue, lookup, candidate_ids):
    require(action['encoding'] == 'item-counts-v1' and action['wards_preserved'] is True, 'observed purchase encoding changed')
    key = counts_key(action['label_counts']); full = lookup.get(key)
    require(full is not None and full == action['full_action_id'] and key[0] == action['kind'], 'observed full multiset identity changed')
    main = catalogue['full_to_group'][full]
    require(main == action['main_group_id'], 'observed main projection changed')
    where = np.flatnonzero(np.asarray(candidate_ids) == main)
    require(len(where) <= 1, 'duplicated target candidate')
    return {'main': main, 'full': full, 'coarsened': int(where[0]) if len(where) else 3,
            'contains_wards': any(i in WARDS for i, _ in key[1]), 'is_no_buy': key[0] == 'observed_no_buy_death',
            'main_size': sum(c for i, c in key[1] if i not in WARDS)}


class RecipeData:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        require(self.directory.is_relative_to(SOURCE_ROOT/'data/exports/winprob-recipe-transition'), 'wrong artifact scope')
        self.bindings = {}; self.target_file_opened = False
        self.manifest, self.digest = bound_json(self.directory/'manifest.json', self.bindings)
        m = self.manifest
        require(m['schema'] == 'recipe-transition-native-m2-v2' and m['state'] == 'complete'
                and m['states'] == 56288 and m['games'] == 3518 and m['native_world'] == '16.18'
                and m['protocol_sha256'] == PROTOCOL_SHA and m['selection_manifest_sha256'] == SELECTION_SHA
                and m['real_pilot']['passed'] is True and m['real_pilot']['mutation_checked_states'] == 200
                and m['receipt_default_equivalence_fixtures'] == 3, 'incomplete or different native comparison')
        for flag in ('target_file_opened', 'source_trajectory_payload_reads', 'source_database_reads', 'test_payload_reads',
                     'confirmation_payload_reads', 'consumer_runtime_imported', 'training_performed', 'outcomes_analyzed', 'accuracy_computed'):
            require(m[flag] is False, 'source boundary failed: '+flag)
        require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'consumed protocol changed'); self.bindings[str(PROTOCOL)] = PROTOCOL_SHA
        sm, sh = bound_json(SELECTION, self.bindings); require(sh == SELECTION_SHA, 'selection manifest changed')
        self.selection = self.bound_member(SELECTION.parent, 'selection.json', sm['files']['selection.json'], read_json=True)
        self.entries = self.selection['entries']; n = len(self.entries)
        require(n == 56288 and all(e['ordinal'] == i for i, e in enumerate(self.entries)), 'selection axis changed')
        base, bs = bound_json(BASE, self.bindings)
        require(bs == '1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea', 'base metadata changed')
        self.admitted, ad = bound_json(base['source_admission_path'], self.bindings)
        require(ad == base['source_admission_sha256'], 'admission changed')
        denied = current_reservations(self.admitted); lookup = {r['match_id']: r for r in self.admitted['games']}
        ids = {e['match_id'] for e in self.entries}
        require(len(ids) == 3518 and not ids & denied and all(lookup[mid]['role'] == 'train'
                and lookup[mid]['patch'] == '16.17' and lookup[mid]['historical_validation'] is False for mid in ids), 'non-FIT or sealed input')
        for raw, want in {**sm['bindings_sha256'], **sm['code_sha256'], **m['code_sha256']}.items():
            path = Path(raw).resolve(strict=True)
            require(path.name != 'targets.jsonl' and path.suffix.lower() not in ('.db', '.gz', '.sqlite'), 'lineage attempts forbidden payload read')
            require(file_hash(path) == want, 'selection/source code or metadata changed'); self.bindings[str(path)] = want
        require(file_hash(CATALOGUE) == CATALOGUE_SHA, 'frozen catalogue changed')
        self.catalogue, self.action_lookup = load_catalogue(CATALOGUE); self.bindings[str(CATALOGUE)] = CATALOGUE_SHA
        require(m['package_digest'] == self.catalogue['package_identity_sha256'], 'different M2 package')
        self.groups = self.bound_member(self.directory, 'group_main.json', m['files']['group_main.json'], read_json=True)
        require(self.groups == self.catalogue['group_main'], 'native action catalogue order changed')
        static = self.bound_member(self.directory, 'native_descriptors.json', m['files']['native_descriptors.json'], read_json=True)
        require(static['world'] == '16.18' and file_hash(STATIC) == STATIC_SHA, 'native/own static world changed')
        original_static = json_file(STATIC)['patches']['16.18']; self.bindings[str(STATIC)] = STATIC_SHA
        for iid, value in static['items'].items():
            require(iid in original_static['items'] and value['stats'] == original_static['items'][iid].get('stats', {}), 'native item-stat mismatch')
        require(static['champion_tags'] == original_static['champion_tags'], 'native champion tags differ from pinned statics')
        context_path = Path(m['source_context_manifest_path']).resolve(strict=True)
        require(context_path.is_relative_to(SOURCE_ROOT/'data/exports/winprob-recipe-transition'), 'context artifact scope mismatch')
        cm, ch = bound_json(context_path, self.bindings)
        require(ch == m['source_context_manifest_sha256'] and cm['selection_manifest_sha256'] == SELECTION_SHA
                and cm['states'] == n and cm['residual_mutation_checked_rows'] == n, 'context lineage mismatch')
        self.context_manifest, self.context_directory = cm, context_path.parent
        cp = self.bound_member(context_path.parent, 'contexts.jsonl.gz', cm['files']['contexts.jsonl.gz'])
        public_digests = [None]*n; source_digests = [None]*n
        with gzip.open(cp, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); i = row['ordinal']; e = self.entries[i]
                require(public_digests[i] is None and all(row[k] == v for k, v in e.items()), 'source context identity mismatch')
                require(set(row['residual_public']) == {'inventory', 'champion_id', 'team_id', 'level', 'ts', 'gold_est', 'others'}, 'residual allowlist changed')
                public_digests[i] = value_hash(row['residual_public']); source_digests[i] = value_hash(row['source_projection'])
        require(all(p is not None for p in public_digests), 'source context coverage incomplete')
        reference = np.zeros((n, 4), dtype=np.float64); context = np.zeros((n, 26), dtype=np.float64)
        candidates = np.zeros((n, 3, 26), dtype=np.float64)
        self.group_ids = np.full((n, 3), -1, dtype=np.int32); self.permutations = np.zeros((n, 3), dtype=np.int64)
        seen = np.zeros(n, dtype=bool); receipts = 0
        rp = self.bound_member(self.directory, 'rows.jsonl.gz', m['files']['rows.jsonl.gz'])
        with gzip.open(rp, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); i = row['ordinal']; e = self.entries[i]
                require(not seen[i] and all(row[k] == e[k] for k in ('sample_id', 'match_id', 'participant_id', 'source_event_index', 'fold', 'is_original_focal')),
                        'native prediction identity differs')
                require(value_hash(row['residual_public']) == public_digests[i] and row['source_projection_sha256'] == source_digests[i], 'native query differs from certified context')
                groups = validate_options(row, self.groups)
                require(len(row['reference_mass']) == 4 and all(v > 0 for v in row['reference_mass'][:len(groups)])
                        and all(v == 0 for v in row['reference_mass'][len(groups):3]), 'candidate/prior support mismatch')
                reference[i] = row['reference_mass']; self.group_ids[i, :len(groups)] = groups
                context[i] = context_features(row['residual_public'], static['items'], static['champion_tags'])
                for j, r in enumerate(row['receipts']):
                    candidates[i, j] = transition_features(r['before'], r['action'], r['wallet'], r['steps'], static['items']); receipts += 1
                self.permutations[i] = shuffle_for_entry(e, len(groups)); seen[i] = True
        require(bool(seen.all()), 'native release dropped selected states')
        self.inputs = ChoiceInputs(reference, context, candidates).validate()
        self.folds = np.array([e['fold'] for e in self.entries], dtype=np.int8)
        self.focal = np.flatnonzero([e['is_original_focal'] for e in self.entries])
        self.game_ids = np.array([e['match_id'] for e in self.entries])
        require(len(self.focal) == 2900 and len(set(self.game_ids[self.focal])) == 2900
                and [int(np.sum(self.folds == f)) for f in (0, 1)] == [28176, 28112]
                and [int(np.sum(self.folds[self.focal] == f)) for f in (0, 1)] == [1449, 1451], 'whole-game fold/focal counts changed')
        self.receipts_verified = receipts
        self.feature_digests = {key: array_hash(value) for key, value in {
            'reference': reference, 'context': context, 'candidates': candidates, 'group_ids': self.group_ids,
            'permutations': self.permutations, 'folds': self.folds, 'focal': self.focal}.items()}
        self.revalidate(); self.validated = True

    def bound_member(self, directory, name, expected, read_json=False):
        path = (Path(directory)/name).resolve(strict=True)
        require(path.parent == Path(directory).resolve() and path.name != 'targets.jsonl', 'input member escaped or is a target')
        require(file_hash(path) == expected, 'artifact member changed: '+name); self.bindings[str(path)] = expected
        return json_file(path) if read_json else path

    def batch(self, indices):
        idx = np.asarray(indices)
        require(idx.ndim == 1 and np.issubdtype(idx.dtype, np.integer) and np.all((idx >= 0) & (idx < len(self.entries))), 'invalid state indices')
        return ChoiceInputs(self.inputs.reference[idx], self.inputs.context[idx], self.inputs.candidates[idx]).validate()

    def revalidate(self):
        require(not set(e['match_id'] for e in self.entries) & current_reservations(self.admitted), 'selected game became reserved')
        for raw, want in self.bindings.items():
            require(file_hash(raw) == want, 'bound input changed')
        if hasattr(self, 'feature_digests'):
            current = {key: array_hash(value) for key, value in {
                'reference': self.inputs.reference, 'context': self.inputs.context, 'candidates': self.inputs.candidates,
                'group_ids': self.group_ids, 'permutations': self.permutations, 'folds': self.folds, 'focal': self.focal}.items()}
            require(current == self.feature_digests, 'frozen in-memory inputs changed')

    def load_targets(self, input_lock_path):
        lock = json_file(input_lock_path)
        require(self.validated and lock['state'] == 'inputs_frozen' and lock['native_manifest_sha256'] == self.digest
                and lock['feature_digests'] == self.feature_digests, 'validated inputs must be durably frozen before targets')
        self.revalidate(); path = self.context_directory/'targets.jsonl'
        require(file_hash(path) == self.context_manifest['files']['targets.jsonl'], 'separate purchase ledger changed')
        self.bindings[str(path)] = self.context_manifest['files']['targets.jsonl']; self.target_file_opened = True
        labels = {key: np.zeros(len(self.entries), dtype=dtype) for key, dtype in (
            ('main', np.int32), ('full', np.int32), ('coarsened', np.int8), ('contains_wards', bool), ('is_no_buy', bool), ('main_size', np.int16))}
        seen = np.zeros(len(self.entries), dtype=bool)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); i = row['ordinal']; e = self.entries[i]
                require(not seen[i] and all(row[k] == e[k] for k in ('sample_id', 'match_id', 'participant_id')), 'purchase identity differs from frozen inputs')
                values = labels_for_action(row['action'], self.catalogue, self.action_lookup, self.group_ids[i])
                for key, value in values.items():
                    labels[key][i] = value
                seen[i] = True
        require(bool(seen.all()) and np.all(self.inputs.reference[np.arange(len(seen)), labels['coarsened']] > 0), 'purchase coverage/prior support failed')
        return labels
