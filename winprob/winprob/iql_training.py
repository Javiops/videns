"""Bounded IQL dataset/runtime helpers. No source-project imports or raw reads."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from winprob import admission
from winprob.iql_dataset_contract import current_reservations, original_fit_from_metadata
from winprob.full_action_catalogue import load_catalogue
from winprob.iql_features import GLOBAL_NAMES, INVENTORY_NAMES
from winprob.iql_models import PublicBoardBatch, FullActionBatch

SEED = 20260930
BATCH_SIZE = 256
HOST_FEATURE_ARRAYS = frozenset({'champion_indices', 'inventory_static', 'visible_level', 'global_public',
    'inventory_offsets', 'inventory_item_indices', 'inventory_item_counts'})
EXPECTED_DTYPES = {
    'champion_indices': 'int32', 'inventory_static': 'float32', 'visible_level': 'float32',
    'global_public': 'float32', 'inventory_offsets': 'int64', 'inventory_item_indices': 'int32',
    'inventory_item_counts': 'int32', 'game_index': 'int32', 'episode_index': 'int32',
    'full_action_id': 'int32', 'main_group_id': 'int32', 'patch_index': 'uint8', 'fold': 'uint8',
    'next_state_index': 'int64', 'reward': 'float32', 'terminated': 'bool', 'timestamp_ms': 'int64',
    'elapsed_ms': 'int64', 'action_static': 'float32', 'action_static_valid': 'bool',
    'action_offsets': 'int64', 'action_item_indices': 'int32', 'action_item_counts': 'int32',
    'full_to_group': 'int32',
    'fold_action_counts': 'int64',
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def tree_hash(value):
    """Value hash independent of tensor device or checkpoint archive timestamps."""
    result = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            result.update(str(tensor.dtype).encode()); result.update(str(tuple(tensor.shape)).encode())
            result.update(tensor.numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=lambda k: str(k)):
                result.update(str(key).encode()); visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            result.update(str(type(obj)).encode())
            for item in obj: visit(item)
        else:
            result.update(repr(obj).encode())
    visit(value)
    return result.hexdigest()


def fold_for_game(match_id):
    return int.from_bytes(hashlib.sha256(('iql-v1-fold:'+match_id).encode()).digest()[:8], 'big') % 2


def refuse_quarantine(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    if any((parent/'QUARANTINED.md').exists() for parent in (directory, *directory.parents)):
        raise ValueError('quarantined IQL input')


def _safe_file(root, name):
    path = (root/name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('dataset member escapes bound directory')
    return path


def validate_transition_graph(arrays, game_metadata, episode_count, episode_metadata=None):
    """Independent linear/sort-time checks; never repair a partial episode."""
    n = len(arrays['reward'])
    g = np.asarray(arrays['game_index']); ep = np.asarray(arrays['episode_index'])
    nxt = np.asarray(arrays['next_state_index']); done = np.asarray(arrays['terminated'])
    reward = np.asarray(arrays['reward']); ts = np.asarray(arrays['timestamp_ms']); elapsed = np.asarray(arrays['elapsed_ms'])
    fold = np.asarray(arrays['fold']); patch = np.asarray(arrays['patch_index'])
    if (not n or any(np.shape(arrays[key]) != (n,) for key in ('game_index', 'episode_index', 'next_state_index',
        'terminated', 'reward', 'timestamp_ms', 'elapsed_ms', 'fold', 'patch_index', 'full_action_id', 'main_group_id'))):
        raise ValueError('unaligned transition arrays')
    if (np.any(g < 0) or np.any(g >= len(game_metadata)) or np.any(ep < 0) or np.any(ep >= episode_count)
            or not np.array_equal(np.unique(g), np.arange(len(game_metadata)))
            or not np.array_equal(np.unique(ep), np.arange(episode_count))):
        raise ValueError('missing or invalid game/episode index')
    if done.dtype != np.bool_ or not np.isfinite(reward).all() or not np.isin(reward, [0, 1]).all() or np.any(elapsed <= 0):
        raise ValueError('invalid terminal/reward/time arrays')
    if np.any(nxt[done] != -1) or np.any(nxt[~done] < 0) or np.any(nxt[~done] >= n) or np.any(reward[~done] != 0):
        raise ValueError('terminal or bootstrap boundary is invalid')
    active = np.flatnonzero(~done)
    if (np.any(g[active] != g[nxt[active]]) or np.any(ep[active] != ep[nxt[active]])
            or np.any(ts[nxt[active]]-ts[active] != elapsed[active])):
        raise ValueError('next state crosses game/episode or disagrees with elapsed time')
    expected_fold = np.array([fold_for_game(row['match_id']) for row in game_metadata], dtype=np.uint8)
    expected_patch = np.array([{'16.17': 0, '16.18': 1, '16.19': 2}[row['patch']] for row in game_metadata], dtype=np.uint8)
    if any(row['fold'] != int(expected_fold[i]) for i, row in enumerate(game_metadata)):
        raise ValueError('metadata fold differs from fixed whole-game hash')
    if not np.array_equal(fold, expected_fold[g]) or not np.array_equal(patch, expected_patch[g]):
        raise ValueError('state fold/patch differs from its complete game')
    order = np.argsort(ep, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(ep[order]))+1, n]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        rows = order[lo:hi]
        rows = rows[np.argsort(ts[rows], kind='stable')]
        if (np.any(np.diff(ts[rows]) <= 0) or int(done[rows].sum()) != 1 or not done[rows[-1]]
                or not np.array_equal(nxt[rows[:-1]], rows[1:])):
            raise ValueError('episode has a gap, duplicate timestamp or misplaced terminal')
        if episode_metadata is not None:
            metadata = episode_metadata[int(ep[rows[0]])]
            game = game_metadata[int(g[rows[0]])]
            if (metadata.get('first_state') != int(rows[0]) or metadata.get('states') != len(rows)
                    or metadata.get('game_index') != int(g[rows[0]]) or metadata.get('fold') != int(fold[rows[0]])
                    or metadata.get('match_id') != game['match_id'] or metadata.get('patch') != game['patch']
                    or not np.array_equal(rows, np.arange(int(rows[0]), int(rows[0])+len(rows)))):
                raise ValueError('episode metadata range/game/fold disagrees with transition arrays')
    counts = {}
    for heldout in (0, 1):
        train = np.flatnonzero(fold != heldout); score = np.flatnonzero(fold == heldout)
        supported = np.isin(arrays['full_action_id'][score], np.unique(arrays['full_action_id'][train]))
        counts[str(heldout)] = {'train_games': len(np.unique(g[train])), 'train_transitions': len(train),
                                'score_games': len(np.unique(g[score])), 'score_transitions': len(score),
                                'observed_full_action_support_fraction': float(supported.mean()) if len(score) else 0.,
                                'training_terminal_classes': sorted(map(int, np.unique(reward[train][done[train]])))}
    return {'states': n, 'games': len(game_metadata), 'episodes': episode_count, 'folds': counts,
            'terminal_transitions': int(done.sum()), 'terminal_return_once': True}


class Dataset:
    def __init__(self, directory, catalogue_path, *, allow_smoke=False):
        self.directory = Path(directory).resolve(strict=True)
        refuse_quarantine(self.directory); refuse_quarantine(catalogue_path)
        manifest_path = self.directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema') != 'iql-dataset-v1' or manifest.get('state') != 'complete':
            raise ValueError('dataset is incomplete or did not pass its source/feature gates')
        if not allow_smoke and (manifest.get('training_authorized') is not True or manifest.get('smoke') is not False
                                or manifest.get('gates', {}).get('passed') is not True):
            raise ValueError('smoke or failed-gate cache cannot authorize real training')
        sidecar = manifest_path.with_suffix('.sha256')
        if sidecar.read_text().split() != [file_hash(manifest_path), manifest_path.name]:
            raise ValueError('dataset manifest sidecar mismatch')
        self.admission_path = Path(manifest['source_admission_path']).resolve(strict=True)
        if file_hash(self.admission_path) != manifest['source_admission_sha256']:
            raise ValueError('source admission hash mismatch')
        if self.admission_path.with_suffix('.sha256').read_text().split()[0] != manifest['source_admission_sha256']:
            raise ValueError('source admission sidecar mismatch')
        self.admission = json.loads(self.admission_path.read_text(encoding='utf-8'))
        admission.verify_manifest_sources(self.admission)
        self.games = manifest['game_metadata']
        admitted = {row['match_id']: row for row in self.admission['games']}
        forbidden = current_reservations(self.admission)
        original_ids, eligible_source_ids, requested_ids = None, set(), set()
        normalized_bindings = {str(Path(path).resolve()): value for path, value in manifest.get('bindings', {}).items()}
        if not manifest.get('source_directories'): raise ValueError('no bound original FIT source directories')
        for directory in manifest['source_directories']:
            source_path = Path(directory).resolve(strict=True)/'manifest.json'
            source_hash = file_hash(source_path)
            if normalized_bindings.get(str(source_path)) != source_hash or source_path.with_suffix('.sha256').read_text().split() != [source_hash, source_path.name]:
                raise ValueError('source trajectory manifest is not bound')
            source = json.loads(source_path.read_text(encoding='utf-8'))
            if source.get('schema') != 'winprob-fit-trajectories-v2' or source.get('state') != 'complete':
                raise ValueError('invalid source trajectory scope')
            for path, expected in source['input_metadata_sha256'].items():
                if Path(path).name == 'source_manifest.json' and file_hash(path) != expected:
                    raise ValueError('original FIT membership metadata changed')
            expected_ids, _ = original_fit_from_metadata(source, admitted, forbidden)
            if set(source.get('original_fit_game_ids', [])) != expected_ids or source.get('original_fit_ids_sha256') != admission.id_digest(expected_ids):
                raise ValueError('source original 4002-FIT universe disagrees with bound metadata')
            if original_ids is not None and original_ids != expected_ids: raise ValueError('source shards use different original FIT universes')
            original_ids = expected_ids
            eligible_source_ids.update(source['eligible_game_ids']); requested_ids.update(source['requested_game_ids'])
        if not allow_smoke and requested_ids != original_ids:
            raise ValueError('full cache source requests do not cover the original fixed FIT scope')
        if len({row['match_id'] for row in self.games}) != len(self.games):
            raise ValueError('duplicate game metadata')
        # Membership precedes every state/reward array read.
        for row in self.games:
            entry = admitted.get(row['match_id'])
            if (entry is None or entry['role'] != 'train' or entry['historical_validation'] is not False or entry['patch'] != row['patch']
                    or row['match_id'] not in original_ids or row['match_id'] not in eligible_source_ids or row['match_id'] in forbidden):
                raise ValueError('IQL may read only admitted FIT game outcomes')
        self.catalogue, self.action_lookup = load_catalogue(catalogue_path)
        if file_hash(catalogue_path) != manifest['catalogue_sha256']:
            raise ValueError('dataset uses a different full-action catalogue')
        self.bindings = {str(manifest_path): file_hash(manifest_path), str(self.admission_path): file_hash(self.admission_path),
                         str(Path(catalogue_path).resolve()): file_hash(catalogue_path)}
        for name, expected in manifest['files'].items():
            path = _safe_file(self.directory, name)
            if file_hash(path) != expected:
                raise ValueError('dataset file hash mismatch: '+name)
            self.bindings[str(path)] = expected
        for path, expected in manifest.get('bindings', {}).items():
            refuse_quarantine(path)
            if file_hash(path) != expected:
                raise ValueError('dataset source binding changed')
            self.bindings[str(Path(path).resolve())] = expected
        self.feature_metadata = json.loads(_safe_file(self.directory, manifest['feature_metadata_file']).read_text(encoding='utf-8'))
        episodes = json.loads(_safe_file(self.directory, manifest['episode_metadata_file']).read_text(encoding='utf-8'))
        if not isinstance(episodes, list):
            raise ValueError('episode metadata must be a complete bound list')
        self.episode_metadata = episodes
        if (self.feature_metadata.get('global_names') != list(GLOBAL_NAMES)
                or self.feature_metadata.get('inventory_names') != list(INVENTORY_NAMES)
                or self.feature_metadata.get('padding_index') != 0 or self.feature_metadata.get('inventory_truncation') is not False):
            raise ValueError('public feature names/order/vocabulary semantics changed')
        self.num_champions = len(self.feature_metadata['champion_index'])
        self.num_items = len(self.feature_metadata['item_index'])
        self.arrays = {}
        if set(manifest['arrays']) != set(EXPECTED_DTYPES):
            raise ValueError('missing or unexpected cache arrays')
        for name, dtype in EXPECTED_DTYPES.items():
            description = manifest['arrays'][name]
            if description['file'] not in manifest['files']:
                raise ValueError('unbound array file')
            array = np.load(_safe_file(self.directory, description['file']), mmap_mode='r', allow_pickle=False)
            if array.dtype != np.dtype(dtype) or array.dtype != np.dtype(description['dtype']) or list(array.shape) != description['shape']:
                raise ValueError('cache array dtype/shape mismatch: '+name)
            self.arrays[name] = array
        self.n = int(manifest['state_count'])
        shapes = {'champion_indices': (self.n, 10), 'inventory_static': (self.n, 10, 22),
                  'visible_level': (self.n, 10, 1), 'global_public': (self.n, 37),
                  'inventory_offsets': (self.n*10+1,), 'action_offsets': (13167,),
                  'action_static': (3, 13166, 23), 'action_static_valid': (3, 13166), 'full_to_group': (13166,),
                  'fold_action_counts': (2, 13166)}
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape: raise ValueError('fixed cache tensor shape mismatch: '+name)
        for name in ('inventory_static', 'visible_level', 'global_public', 'action_static'):
            for lo in range(0, len(self.arrays[name]), 16384):
                if not np.isfinite(self.arrays[name][lo:lo+16384]).all(): raise ValueError('nonfinite public features')
        champions = self.arrays['champion_indices']
        if np.any(champions < 1) or np.any(champions > self.num_champions): raise ValueError('unknown champion index')
        for prefix, entries in (('inventory', self.n*10), ('action', 13166)):
            offset = self.arrays[prefix+'_offsets']; ids = self.arrays[prefix+'_item_indices']; counts = self.arrays[prefix+'_item_counts']
            if (offset.shape != (entries+1,) or offset[0] != 0 or offset[-1] != len(ids) or ids.shape != counts.shape
                    or np.any(np.diff(offset) < 0) or np.any(ids < 1) or np.any(ids > self.num_items) or np.any(counts < 1)):
                raise ValueError('invalid full ragged inventory/action representation')
        actions, patches = self.arrays['full_action_id'], self.arrays['patch_index']
        if np.any(actions < 0) or np.any(actions >= 13166) or np.any(patches > 2): raise ValueError('invalid action or patch index')
        if not np.array_equal(self.arrays['full_to_group'], self.catalogue['full_to_group']): raise ValueError('main-group projection changed')
        if not np.array_equal(self.arrays['main_group_id'], self.arrays['full_to_group'][actions]): raise ValueError('transition action/group mismatch')
        if not self.arrays['action_static_valid'][patches, actions].all(): raise ValueError('observed action has no native-patch attributes')
        actual_counts = np.stack([np.bincount(actions[self.arrays['fold'] == fold], minlength=13166) for fold in (0, 1)])
        if not np.array_equal(actual_counts, self.arrays['fold_action_counts']): raise ValueError('fold action counts do not match actual transitions')
        self.audit = validate_transition_graph(self.arrays, self.games, len(episodes), episodes)
        if not allow_smoke and (self.audit['games'] < 1000 or self.audit['episodes'] < 5000 or self.n < 100000):
            raise ValueError('insufficient complete FIT population for fixed recipe')
        if not allow_smoke and any(row['train_games'] < 450 or row['train_transitions'] < 40000 or row['observed_full_action_support_fraction'] < .7
                                  or row['training_terminal_classes'] != [0, 1] for row in self.audit['folds'].values()):
            raise ValueError('critic fold count/action-support gate failed')
        self.manifest = manifest
        self.inventory_max = int(np.max(np.diff(self.arrays['inventory_offsets'])))
        # This small owned copy is the only host feature needed by later
        # ordinary-frequency diagnostics after the CUDA state cache is ready.
        self.focal_champion_indices = np.array(self.arrays['champion_indices'][:, 0], dtype=np.int32, copy=True)
        self.host_features_released = False

    def release_host_features(self, *, synchronized_cuda_copy):
        """Close only feature mmaps after a verified/synchronized CUDA copy.

        Do not retain any ndarray view of these mappings. Transition arrays,
        action tables, metadata, source files and their bindings stay intact.
        """
        if synchronized_cuda_copy is not True:
            raise ValueError('host maps cannot close before the CUDA copy is synchronized')
        if self.host_features_released:
            raise ValueError('host feature mappings were already released')
        if self.focal_champion_indices.base is not None:
            raise ValueError('focal champion diagnostic vector must own its storage')
        before = memory_snapshot()
        removed = []
        for name in sorted(HOST_FEATURE_ARRAYS):
            array = self.arrays.pop(name)
            mapping = getattr(array, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()
            removed.append(name)
            del array, mapping
        self.host_features_released = True
        gc.collect()
        return {'removed_host_feature_arrays': removed, 'transition_and_action_arrays_retained': True,
                'focal_champion_owned_copy_bytes': int(self.focal_champion_indices.nbytes),
                'memory_before_release': before, 'memory_after_release': memory_snapshot()}


def padded_bags(offsets, item_indices, item_counts, bag_indices, width=None):
    bags = np.asarray(bag_indices, dtype=np.int64)
    starts = np.asarray(offsets[bags], dtype=np.int64)
    lengths = np.asarray(offsets[bags+1]-offsets[bags], dtype=np.int64)
    actual = int(lengths.max()) if len(lengths) else 0
    width = actual if width is None else int(width)
    if width < actual: raise ValueError('inventory truncation is forbidden')
    ids = np.zeros((len(bags), width), dtype=np.int64); counts = np.zeros((len(bags), width), dtype=np.float32)
    row = np.repeat(np.arange(len(bags)), lengths)
    total = int(lengths.sum())
    if total:
        columns = np.arange(total)-np.repeat(np.cumsum(lengths)-lengths, lengths)
        positions = starts[row]+columns
        ids[row, columns], counts[row, columns] = item_indices[positions], item_counts[positions]
    return ids, counts


class TensorStore:
    """Bounded resident CUDA cache or exact memmap-to-CUDA minibatch gather."""
    def __init__(self, dataset, device, guard, preload_budget_gib=6.):
        if torch.device(device).type != 'cuda': raise ValueError('real IQL uses CUDA; no CPU training fallback')
        self.dataset, self.device, self.resident = dataset, torch.device(device), {}
        self.guard = guard
        n, width = dataset.n, dataset.inventory_max
        estimate = n*(1148+120*width)
        free, _ = torch.cuda.mem_get_info(self.device)
        self.preloaded = estimate <= preload_budget_gib*1024**3 and estimate+2*1024**3 <= free
        self.estimated_state_bytes = estimate
        self.host_release_report = None
        self.ally_mask = torch.tensor([False]+[True]*4+[False]*5, dtype=torch.bool, device=self.device)
        self.enemy_mask = ~self.ally_mask; self.enemy_mask[0] = False
        if self.preloaded:
            shapes = {'champion_indices': ((n, 10), torch.long), 'inventory_static': ((n, 10, 22), torch.float32),
                      'visible_level': ((n, 10, 1), torch.float32), 'global_public': ((n, 37), torch.float32),
                      'inventory_indices': ((n, 10, width), torch.long), 'inventory_counts': ((n, 10, width), torch.float32)}
            for name, (shape, dtype) in shapes.items(): self.resident[name] = torch.empty(shape, dtype=dtype, device=self.device)
            for lo in range(0, n, 8192):
                guard(); hi = min(lo+8192, n)
                rows = np.arange(lo, hi)
                packed = self._cpu_state(rows, width)
                for name, value in packed.items(): self.resident[name][lo:hi].copy_(torch.as_tensor(value, device=self.device))
            # The last temporary batch must not retain any host arrays.
            del packed, value, rows
        ids, counts = padded_bags(dataset.arrays['action_offsets'], dataset.arrays['action_item_indices'],
                                  dataset.arrays['action_item_counts'], np.arange(13166))
        self.action_indices = torch.as_tensor(ids, device=self.device)
        self.action_counts = torch.as_tensor(counts, device=self.device)
        self.action_static = torch.tensor(np.asarray(dataset.arrays['action_static']), dtype=torch.float32, device=self.device)
        self.full_to_group = torch.tensor(np.asarray(dataset.arrays['full_to_group']), dtype=torch.long, device=self.device)
        del ids, counts
        if self.preloaded:
            if not self.resident or any(tensor.device.type != 'cuda' for tensor in self.resident.values()):
                raise RuntimeError('feature maps may only be released for a complete CUDA resident cache')
            torch.cuda.synchronize(self.device)
            self.host_release_report = dataset.release_host_features(synchronized_cuda_copy=True)

    def _cpu_state(self, rows, width=None):
        if self.dataset.host_features_released:
            raise RuntimeError('host feature maps were released; use the resident CUDA state path')
        a = self.dataset.arrays
        bags = (rows[:, None]*10+np.arange(10)).ravel()
        ids, counts = padded_bags(a['inventory_offsets'], a['inventory_item_indices'], a['inventory_item_counts'], bags, width)
        return {'champion_indices': np.array(a['champion_indices'][rows], dtype=np.int64),
                'inventory_indices': ids.reshape(len(rows), 10, -1), 'inventory_counts': counts.reshape(len(rows), 10, -1),
                **{name: np.array(a[name][rows], copy=True) for name in ('inventory_static', 'visible_level', 'global_public')}}

    def states(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if not len(rows): raise ValueError('no terminal placeholder state allowed')
        if self.preloaded:
            index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
            values = {name: tensor.index_select(0, index) for name, tensor in self.resident.items()}
        else:
            values = {name: torch.as_tensor(value, device=self.device) for name, value in self._cpu_state(rows).items()}
        return PublicBoardBatch(**values, ally_mask=self.ally_mask.expand(len(rows), -1), enemy_mask=self.enemy_mask.expand(len(rows), -1))

    def actions(self, rows):
        a = self.dataset.arrays
        ids = torch.as_tensor(np.array(a['full_action_id'][rows], dtype=np.int64), device=self.device)
        patch = torch.as_tensor(np.array(a['patch_index'][rows], dtype=np.int64), device=self.device)
        return FullActionBatch(self.action_indices[ids], self.action_counts[ids], self.action_static[patch, ids])

    def labels(self, rows):
        return torch.as_tensor(np.array(self.dataset.arrays['full_action_id'][rows], dtype=np.int64), device=self.device)


def adam(parameters):
    return torch.optim.Adam(parameters, lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)


def clone_actor_optimizer(actor, optimizer):
    cloned = deepcopy(actor)
    copied_optimizer = adam(cloned.parameters())
    copied_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    if tree_hash(actor.state_dict()) != tree_hash(cloned.state_dict()) or tree_hash(optimizer.state_dict()) != tree_hash(copied_optimizer.state_dict()):
        raise ValueError('warmstart parameters/Adam state were not copied exactly')
    return cloned, copied_optimizer


def epoch_order(indices, phase, epoch, fold=None):
    key = f'{SEED}|{phase}|{fold}|{epoch}'
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
    return np.random.default_rng(seed).permutation(np.asarray(indices, dtype=np.int64))


def main_group_log_odds(logits, group_a, group_b, full_to_group):
    if logits.ndim != 2 or full_to_group.shape != (logits.shape[1],): raise ValueError('full-action mapping mismatch')
    a = torch.as_tensor(group_a, dtype=torch.long, device=logits.device)
    b = torch.as_tensor(group_b, dtype=torch.long, device=logits.device)
    if a.shape != (len(logits),) or b.shape != a.shape: raise ValueError('pair groups are not aligned')
    left = torch.logsumexp(logits.masked_fill(full_to_group[None, :] != a[:, None], -torch.inf), dim=1)
    right = torch.logsumexp(logits.masked_fill(full_to_group[None, :] != b[:, None], -torch.inf), dim=1)
    result = right-left
    if not bool(torch.isfinite(result).all()): raise ValueError('missing full variants for a main group')
    return result


def available_ram_bytes():
    if __import__('os').name != 'nt': return None
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)]+[(name, ctypes.c_ulonglong) for name in
            ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
    state = Memory(); state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)): raise OSError('memory status unavailable')
    return state.available


def memory_snapshot():
    result = {'available_host_bytes': available_ram_bytes()}
    if __import__('os').name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in (
                'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
                'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile', 'private_usage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.argtypes=[]; kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        counter = Counters(); counter.cb = ctypes.sizeof(counter)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            result.update(process_working_set_bytes=int(counter.working_set),
                          peak_process_working_set_bytes=int(counter.peak_working_set),
                          process_private_bytes=int(counter.private_usage),
                          peak_process_commit_bytes=int(counter.peak_pagefile))
    return result


class RuntimeGuard:
    def __init__(self, deadline_seconds=5400, host_floor_gib=.75):
        self.started = time.monotonic(); self.deadline = deadline_seconds; self.floor = host_floor_gib*1024**3
        self.snapshots = {}
        self.minimum_available = None; self.peak_working_set = 0; self.peak_private = 0; self.memory_marks = []

    def sample_memory(self, stage=None):
        sample = memory_snapshot()
        available = sample['available_host_bytes']
        if available is not None:
            self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)
        self.peak_working_set = max(self.peak_working_set, sample.get('peak_process_working_set_bytes', 0))
        self.peak_private = max(self.peak_private, sample.get('process_private_bytes', 0))
        if stage is not None:
            self.memory_marks.append({'stage': stage, 'elapsed_s': time.monotonic()-self.started, **sample})
        return sample

    def memory_report(self):
        return {'minimum_observed_available_host_bytes': self.minimum_available,
                'peak_process_working_set_bytes': self.peak_working_set,
                'maximum_sampled_process_private_bytes': self.peak_private,
                'live_host_floor_bytes': int(self.floor), 'marks': self.memory_marks}

    def bind(self, paths):
        self.snapshots = {str(Path(path)): (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths}

    def __call__(self):
        if time.monotonic()-self.started > self.deadline: raise RuntimeError('fixed IQL wall deadline exceeded')
        available = self.sample_memory()['available_host_bytes']
        if available is not None and available < self.floor:
            raise RuntimeError(f'available host RAM below .75 GiB live floor: {available/1024**3:.3f} GiB')
        for path, expected in self.snapshots.items():
            actual = Path(path).stat()
            if (actual.st_size, actual.st_mtime_ns) != expected: raise RuntimeError('bound input/code changed during training')


def set_below_normal_priority():
    if __import__('os').name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.argtypes=[]; kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.SetPriorityClass.restype=wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000): raise ctypes.WinError(ctypes.get_last_error())
