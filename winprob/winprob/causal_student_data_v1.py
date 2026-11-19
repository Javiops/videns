"""Admitted selected public inputs with separate, fold-guarded supervision."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from winprob.history_probe_data import BaseCache
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.iql_features import GLOBAL_NAMES, INVENTORY_NAMES
from winprob.iql_models import PublicBoardBatch
from winprob.iql_training import padded_bags
from winprob.bounded_npy_array_v1 import BoundedNpyArray
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.full_action_catalogue import load_catalogue

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT/'data/runs/causal-student-inputs-20261004-v1/inputs/manifest.json'
TEACHER = ROOT.parent/'build_tracker/data/exports/winprob-route-prior/route-prior-20261003-v1/manifest.json'
PROTOCOL = ROOT/'docs/causal_student_probe_protocol_v1.md'
PROTOCOL_SHA = 'fc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e'
ARMS = ('hard', 'distilled', 'shuffled_teacher')


def require(ok, message):
    if not ok: raise ValueError(message)


def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def value_hash(a): return hashlib.sha256(json.dumps(a, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def epoch_order(train_ordinals, fold, epoch):
    require(fold in (0, 1) and type(epoch) is int and 1 <= epoch <= 20, 'fixed training fold/epoch required')
    seed = int.from_bytes(hashlib.sha256(f'causal-student-v1/order/{fold}|{epoch}'.encode()).digest()[:8], 'big')
    return np.random.default_rng(seed).permutation(np.asarray(train_ordinals, dtype=np.int64))


def teacher_permutation(folds, champions, times, heldout):
    f = np.asarray(folds); c = np.asarray(champions); t = np.asarray(times)
    require(heldout in (0, 1) and f.shape == c.shape == t.shape and f.ndim == 1, 'aligned teacher-control metadata required')
    result = np.full(len(f), -1, dtype=np.int64); bins = defaultdict(list)
    for i in np.flatnonzero(f != heldout): bins[(int(c[i]), int(t[i]//180000))].append(int(i))
    rng = np.random.default_rng(20261004+heldout)
    for key in sorted(bins):
        rows = np.array(sorted(bins[key]), dtype=np.int64); result[rows] = rng.permutation(rows)
    train = np.flatnonzero(f != heldout)
    require(np.array_equal(np.sort(result[train]), train) and np.all(f[result[train]] != heldout)
            and np.all(c[result[train]] == c[train]) and np.all(t[result[train]]//180000 == t[train]//180000), 'teacher permutation crosses its training stratum')
    return result, {'bins': len(bins), 'singleton_bins': sum(len(v) == 1 for v in bins.values()),
                    'self_mappings': int(np.sum(result[train] == train)), 'training_rows': len(train)}


def permitted_supervision_indices(folds, ordinals, heldout, permutation=None):
    f = np.asarray(folds); rows = np.asarray(ordinals)
    require(heldout in (0, 1) and rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer)
            and np.all((rows >= 0) & (rows < len(f))) and np.all(f[rows] != heldout), 'held-out row requested for training supervision')
    teacher_rows = rows if permutation is None else np.asarray(permutation)[rows]
    require(teacher_rows.shape == rows.shape and np.all((teacher_rows >= 0) & (teacher_rows < len(f)))
            and np.all(f[teacher_rows] != heldout), 'teacher donor escaped training fold')
    return rows, teacher_rows


class SelectedPublicStore:
    """Only causal public tensors, no teacher or observed target attributes."""
    def __init__(self, tensors, runes):
        self.tensors = tensors; self.runes = runes
        self.ally = torch.tensor([False]+[True]*4+[False]*5, device=runes.device)
        self.enemy = torch.tensor([False]*5+[True]*5, device=runes.device)

    def batch(self, ordinals):
        rows = torch.as_tensor(np.asarray(ordinals, dtype=np.int64), device=self.runes.device)
        require(rows.ndim == 1 and 1 <= len(rows) <= 128, 'bounded public minibatch required')
        v = {k: x.index_select(0, rows) for k, x in self.tensors.items()}
        board = PublicBoardBatch(v['champion_indices'], v['inventory_indices'], v['inventory_counts'],
            v['inventory_static'], v['visible_level'], self.ally.expand(len(rows), -1), self.enemy.expand(len(rows), -1), v['global_public'])
        return board, self.runes.index_select(0, rows)


class StudentData:
    def __init__(self):
        require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'fixed student protocol changed')
        self.base = BaseCache(ROOT/'data/runs/iql-cache-20260930-v1/cache', ROOT/'docs/history_probe_protocol_v1.md')
        self.bindings = dict(self.base.bindings)
        self.manifest, self.input_sha = bound_json(INPUTS, self.bindings)
        m = self.manifest
        require(self.input_sha == '4b45bb0f8e68d68645d1ac40c3f1a6c06780fe8e09667f9366146e966d5b8c2a'
                and m['state'] == 'complete' and m['states'] == 56288 and m['games'] == 3518 and m['focals'] == 2900
                and m['purchase_targets_opened'] is False and m['legacy_roles_indexed'] is False, 'qualified public inputs required')
        self.arrays = {}
        for name, sha in m['files'].items():
            path = INPUTS.parent/name; require(file_hash(path) == sha, 'public input member changed'); self.bindings[str(path)] = sha
            if name.endswith('.npy'): self.arrays[name[:-4]] = np.load(path, allow_pickle=False)
        self.entries = json.loads((INPUTS.parent/'selection.json').read_text(encoding='utf-8'))['entries']
        self.indices = self.arrays['state_indices']; self.folds = self.arrays['folds']; self.focals = self.arrays['focal_ordinals']
        require(self.indices.shape == self.folds.shape == (56288,) and self.focals.shape == (2900,), 'student selected axes changed')
        self.game_ids = np.array([e['match_id'] for e in self.entries])
        require(len(set(self.game_ids)) == 3518 and len(set(self.game_ids[self.focals])) == 2900, 'game identity/count changed')
        for i, e in enumerate(self.entries):
            ep = self.base.episodes[e['episode_index']]
            require(e['ordinal'] == i and e['state_index'] == int(self.indices[i]) and e['fold'] == int(self.folds[i])
                    and ep['match_id'] == e['match_id'] and ep['participant_id'] == e['participant_id']
                    and ep['first_state']+e['source_node_ordinal'] == e['state_index'] and e['patch'] == '16.17', 'selected state/episode mismatch')
        for fold in (0, 1):
            require(len(set(self.game_ids[self.folds != fold]) & set(self.game_ids[self.folds == fold])) == 0, 'whole-game fold overlap')
        require(np.array_equal(np.flatnonzero([e['is_original_focal'] for e in self.entries]), self.focals), 'fixed focal selection changed')
        teacher, th = bound_json(TEACHER, self.bindings); self.teacher_manifest = teacher; self.teacher_sha = th
        require(th == '35113430e6997ec1ef993da5d9dc3f9325941138e21db24fab300ea3b5f470ac'
                and teacher['state'] == 'complete' and teacher['states'] == 56288
                and teacher['complete_probability_rows_replayed_exactly'] == 56288, 'complete native teacher required')
        path = TEACHER.parent/'full_probability.npy'; require(file_hash(path) == teacher['files'][path.name], 'teacher bytes changed')
        self.bindings[str(path)] = teacher['files'][path.name]
        self.teacher_reader = BoundedNpyRows(path, expected_shape=(56288, 9333), expected_dtype=np.float32)
        meta = self.base.json_member(self.base.manifest['feature_metadata_file'])
        require(meta['global_names'] == list(GLOBAL_NAMES) and meta['inventory_names'] == list(INVENTORY_NAMES)
                and meta['padding_index'] == 0 and meta['inventory_truncation'] is False, 'public feature schema changed')
        self.num_champions = len(meta['champion_index']); self.num_items = len(meta['item_index'])
        self.features = {}
        for name in ('champion_indices', 'inventory_static', 'visible_level', 'global_public'):
            desc = self.base.manifest['arrays'][name]; path = self.base.member(desc['file'])
            self.features[name] = BoundedNpyArray(path, desc['shape'], desc['dtype'], max_rows=256)
        for name in ('inventory_offsets', 'inventory_item_indices', 'inventory_item_counts'): self.features[name] = self.base.array(name)
        off = self.features['inventory_offsets']; bags = (self.indices[:, None]*10+np.arange(10)).ravel()
        self.inventory_width = int(np.max(off[bags+1]-off[bags])); require(self.inventory_width > 0, 'empty global inventory domain')
        self.champions = np.empty(56288, dtype=np.int32)
        for lo in range(0, 56288, 256):
            hi = min(lo+256, 56288); self.champions[lo:hi] = self.features['champion_indices'][self.indices[lo:hi]][:, 0]
        self.times = np.array([e['timestamp_ms'] for e in self.entries], dtype=np.int64)
        self.permutations = {}; self.shuffle_facts = {}
        for fold in (0, 1): self.permutations[fold], self.shuffle_facts[fold] = teacher_permutation(self.folds, self.champions, self.times, fold)
        self.catalogue, _ = load_catalogue(self.base.manifest['catalogue_path'])
        self.bindings.update(self.base.bindings); self.bindings[str(PROTOCOL)] = PROTOCOL_SHA
        for path, sha in m['code_sha256'].items(): require(file_hash(path) == sha, 'consumed public input code changed'); self.bindings[path] = sha
        self.labels_gpu = None; self.labels_cpu = None; self.teacher_gpu = None; self.public_store = None
        self.revalidate()

    def input_lock(self):
        return {'state': 'student_inputs_frozen_before_targets', 'public_manifest_sha256': self.input_sha,
            'teacher_manifest_sha256': self.teacher_sha, 'protocol_sha256': PROTOCOL_SHA,
            'bindings_sha256': self.bindings, 'folds_sha256': array_hash(self.folds), 'state_indices_sha256': array_hash(self.indices),
            'teacher_permutation_sha256': {str(f): array_hash(v) for f, v in self.permutations.items()}, 'purchase_targets_opened': False}

    def preload(self, guard, status):
        device = torch.device('cuda'); n = len(self.indices); width = self.inventory_width
        specs = {'champion_indices': ((n, 10), torch.long), 'inventory_indices': ((n, 10, width), torch.long),
            'inventory_counts': ((n, 10, width), torch.float32), 'inventory_static': ((n, 10, 22), torch.float32),
            'visible_level': ((n, 10, 1), torch.float32), 'global_public': ((n, 37), torch.float32)}
        tensors = {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in specs.items()}
        self.teacher_gpu = torch.empty((n, 9333), dtype=torch.float32, device=device)
        a = self.features
        for lo in range(0, n, 256):
            guard(); hi = min(lo+256, n); selected = self.indices[lo:hi]; bags = (selected[:, None]*10+np.arange(10)).ravel()
            ids, counts = padded_bags(a['inventory_offsets'], a['inventory_item_indices'], a['inventory_item_counts'], bags, width)
            packed = {'champion_indices': a['champion_indices'][selected].astype(np.int64),
                'inventory_indices': ids.reshape(hi-lo, 10, width), 'inventory_counts': counts.reshape(hi-lo, 10, width),
                **{name: a[name][selected] for name in ('inventory_static', 'visible_level', 'global_public')}}
            require(all(np.isfinite(v).all() for v in packed.values()), 'nonfinite selected public tensor')
            for name, values in packed.items(): tensors[name][lo:hi].copy_(torch.as_tensor(values, device=device))
            prior = self.teacher_reader[lo:hi].astype(np.float64)
            require(np.isfinite(prior).all() and np.all(prior >= 0) and np.all(np.abs(prior.sum(1)-1) < 2e-6), 'invalid teacher probabilities')
            prior /= prior.sum(1, keepdims=True)
            self.teacher_gpu[lo:hi].copy_(torch.as_tensor(prior.astype(np.float32), device=device))
            if hi % 8192 == 0 or hi == n: status('preloading_selected_inputs', completed=hi, states=n)
        del packed, prior, ids, counts, bags, selected
        torch.cuda.synchronize()
        self.public_store = SelectedPublicStore(tensors, torch.as_tensor(self.arrays['rune_indices'].astype(np.int64), device=device))
        self.actions = (torch.as_tensor(self.arrays['action_item_indices'].astype(np.int64), device=device),
            torch.as_tensor(self.arrays['action_item_counts'], device=device), torch.as_tensor(self.arrays['action_attributes'], device=device),
            torch.as_tensor(self.arrays['empty_main_mask'], device=device))
        for name in ('inventory_offsets', 'inventory_item_indices', 'inventory_item_counts'):
            mapping = getattr(self.features[name], '_mmap', None)
            if mapping is not None: mapping.close()
            self.features.pop(name); self.base.arrays.pop(name, None)

    def load_targets(self, lock_path):
        lock = json.loads(Path(lock_path).read_text(encoding='utf-8'))
        require(lock['state'] == 'student_inputs_frozen_before_targets' and lock['public_manifest_sha256'] == self.input_sha
                and lock['teacher_manifest_sha256'] == self.teacher_sha and lock['bindings_sha256'] == self.bindings, 'student input lock changed')
        self.revalidate(); self.base.selection_frozen = True
        full = np.array(self.base.array('full_action_id')[self.indices], dtype=np.int64)
        main = np.array(self.base.array('main_group_id')[self.indices], dtype=np.int64)
        mapping = np.asarray(self.catalogue['full_to_group'])
        require(np.all((full >= 0) & (full < len(mapping))) and np.array_equal(main, mapping[full]), 'actual counted target mapping changed')
        self.labels_cpu = main; self.labels_gpu = torch.as_tensor(main, dtype=torch.long, device='cuda')

    def teacher_training(self, ordinals, fold, arm):
        require(arm in ARMS, 'declared student arm required')
        _, donor = permitted_supervision_indices(self.folds, ordinals, fold,
            self.permutations[fold] if arm == 'shuffled_teacher' else None)
        if arm == 'hard': return None
        return self.teacher_gpu.index_select(0, torch.as_tensor(donor, device='cuda'))

    def targets_training(self, ordinals, fold):
        rows, _ = permitted_supervision_indices(self.folds, ordinals, fold)
        require(self.labels_gpu is not None, 'target stage not enabled')
        return self.labels_gpu.index_select(0, torch.as_tensor(rows, device='cuda'))

    def revalidate(self):
        self.base.revalidate()
        for p, sha in self.bindings.items(): require(file_hash(p) == sha, 'student input binding changed')
