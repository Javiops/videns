"""Admitted state-only inputs and recipient-fold-specific label permissions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from winprob.history_probe_data import BaseCache, STATE_ARRAYS
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.iql_features import GLOBAL_NAMES, INVENTORY_NAMES
from winprob.full_action_catalogue import load_catalogue
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.bayes_proxy_ratio_v1 import require

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT/'data/runs/bayes-proxy-prefix-20261003-v1/census'
NATIVE = ROOT/'data/runs/route-probe-fit-20261003-v3/experiment/distribution'
OLD = ROOT/'data/runs/history-probe-20260930-v3/experiment'
PROTOCOL = ROOT/'docs/bayes_proxy_probe_protocol_v1.md'
CHECKPOINTS = ('e050c340298e5495fb966378d33cd133b558cf6302f673db5d8a14188a84364a',
               '5381e72e1606f773817963c317652ecb2d6b568316ae189a63261df4565a1440')


class LabelGate:
    def __init__(self, folds, past_by_fold, full_to_group, loader):
        self.folds = np.asarray(folds); self.mapping = np.asarray(full_to_group); self.loader = loader
        self.past = {}
        for fold in (0, 1):
            mask = np.zeros(len(self.folds), dtype=bool); indices = np.asarray(past_by_fold[fold], dtype=np.int64)
            require(np.all((indices >= 0) & (indices < len(mask))) and np.all(self.folds[indices] == fold), 'past permission crosses fold')
            mask[indices] = True; self.past[fold] = mask
        self.receipts = []

    def read(self, indices, fold, purpose):
        rows = np.asarray(indices)
        require(fold in (0, 1) and purpose in ('training', 'past') and rows.ndim == 1
                and np.issubdtype(rows.dtype, np.integer) and np.all((rows >= 0) & (rows < len(self.folds))), 'explicit label purpose/fold/indices required')
        allowed = self.folds[rows] != fold if purpose == 'training' else self.past[fold][rows]
        require(bool(allowed.all()), 'label request includes recipient-fold current/future or other unapproved rows')
        if not len(rows): return np.empty(0, dtype=np.int32)
        # Validation above intentionally happens before either file is opened.
        full = np.asarray(self.loader('full_action_id', rows)); main = np.asarray(self.loader('main_group_id', rows))
        require(np.issubdtype(full.dtype, np.integer) and full.shape == main.shape == rows.shape
                and np.all((full >= 0) & (full < len(self.mapping))) and np.array_equal(main, self.mapping[full]), 'observed full/main counted action mapping changed')
        if purpose == 'past': require(np.all(full != 0), 'synthetic no-buy entered closed evidence')
        self.receipts.append({'fold': fold, 'purpose': purpose, 'rows': len(rows)})
        return main.astype(np.int32, copy=True)


class ProxyData:
    def __init__(self):
        self.base = BaseCache(ROOT/'data/runs/iql-cache-20260930-v1/cache', ROOT/'docs/history_probe_protocol_v1.md')
        self.bindings = dict(self.base.bindings)
        census, sha = bound_json(PREFIX/'report.json', self.bindings)
        require(sha == '89ff1f75c4031bd88c22ea31050e157eed2f50cb6adfed9a37b0f0d309aa1b35'
                and census['state'] == 'complete' and census['query_games'] == 2900
                and census['past_real_closed_visits'] == 15573, 'fixed prefix census changed')
        for name, digest in census['files'].items():
            require(file_hash(PREFIX/name) == digest, 'prefix member changed'); self.bindings[str(PREFIX/name)] = digest
        self.queries = json.loads((PREFIX/'query_selection.json').read_text(encoding='utf-8'))['queries']
        self.prefix = np.load(PREFIX/'prefix_state_indices.npy', allow_pickle=False)
        self.offsets = np.load(PREFIX/'prefix_offsets.npy', allow_pickle=False)
        self.query_indices = np.array([e['state_index'] for e in self.queries], dtype=np.int64)
        self.folds = self.base.array('fold'); self.games = self.base.array('game_index'); self.times = self.base.array('timestamp_ms')
        self.query_folds = self.folds[self.query_indices]
        require(len(self.queries) == 2900 and len(self.prefix) == 15573 and self.offsets.shape == (2901,)
                and self.offsets[0] == 0 and self.offsets[-1] == len(self.prefix) and np.all(np.diff(self.offsets) >= 0)
                and len(np.unique(self.prefix)) == len(self.prefix)
                and not np.intersect1d(self.prefix, self.query_indices).size, 'prefix/current axes overlap or differ')
        history_path = ROOT/'data/runs/history-cache-20260930-v1/cache/manifest.json'
        history, hh = bound_json(history_path, self.bindings)
        require(hh == '1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b', 'closure lineage changed')
        ep = history_path.parent/'visit_end_ms.npy'; require(file_hash(ep) == history['files'][ep.name], 'closure sidecar changed')
        self.bindings[str(ep)] = history['files'][ep.name]; self.ends = np.load(ep, mmap_mode='r', allow_pickle=False)
        past_by_fold = {0: [], 1: []}
        self.prefix_owner = np.empty(len(self.prefix), dtype=np.int32)
        for i, q in enumerate(self.queries):
            current = q['state_index']; episode = self.base.episodes[q['episode_index']]
            lo, hi = map(int, self.offsets[i:i+2]); past = self.prefix[lo:hi]
            require(episode['first_state']+q['source_node_ordinal'] == current
                    and episode['match_id'] == q['match_id'] and episode['participant_id'] == q['participant_id']
                    and q['timestamp_ms'] == self.times[current] and q['fold'] == self.folds[current]
                    and np.all((past >= episode['first_state']) & (past < current))
                    and np.all(self.ends[past] < self.times[current]) and np.all(self.folds[past] == q['fold'])
                    and np.all(np.diff(past) > 0), 'query/prefix identity or strict cutoff changed')
            self.prefix_owner[lo:hi] = i; past_by_fold[q['fold']].extend(past.tolist())
        self.forecast_rows = {f: np.sort(np.concatenate((self.query_indices[self.query_folds == f], np.asarray(past_by_fold[f], dtype=np.int64)))) for f in (0, 1)}
        self.past_by_fold = {f: np.asarray(past_by_fold[f], dtype=np.int64) for f in (0, 1)}
        self.features = {name: self.base.array(name) for name in sorted(STATE_ARRAYS)}
        meta = self.base.json_member(self.base.manifest['feature_metadata_file'])
        require(meta['global_names'] == list(GLOBAL_NAMES) and meta['inventory_names'] == list(INVENTORY_NAMES)
                and meta['padding_index'] == 0 and meta['inventory_truncation'] is False, 'public feature schema changed')
        self.num_champions = len(meta['champion_index']); self.num_items = len(meta['item_index'])
        self.champions = np.array(self.features['champion_indices'][:, 0], dtype=np.int32)
        off = self.features['inventory_offsets']; ids = self.features['inventory_item_indices']; cnt = self.features['inventory_item_counts']
        require(off.shape == (3924480+1,) and off[0] == 0 and off[-1] == len(ids) and ids.shape == cnt.shape
                and np.all(np.diff(off) >= 0) and np.all((ids >= 1) & (ids <= self.num_items)) and np.all(cnt > 0), 'ragged inventory counts invalid')
        for key in ('inventory_static', 'visible_level', 'global_public'):
            a = self.features[key]
            for lo in range(0, len(a), 4096): require(np.isfinite(a[lo:lo+4096]).all(), 'nonfinite public feature')
        require(np.all((self.features['champion_indices'] >= 1) & (self.features['champion_indices'] <= self.num_champions)), 'champion vocabulary changed')
        self.catalogue, _ = load_catalogue(self.base.manifest['catalogue_path'])
        self.mapping = np.asarray(self.catalogue['full_to_group'], dtype=np.int32)
        native, nh = bound_json(NATIVE/'manifest.json', self.bindings)
        require(nh == 'b69a407f794bbebebaf0be7da8c52d50a61629b2f755df3f92c7d95df3173e2e', 'frozen native focal prior changed')
        for name in ('m2.npy', 'selection.json', 'group_main.json'):
            require(file_hash(NATIVE/name) == native['files'][name], 'native focal member changed')
            self.bindings[str(NATIVE/name)] = native['files'][name]
        native_selection = json.loads((NATIVE/'selection.json').read_text(encoding='utf-8'))
        require(native_selection == [{**e, 'query_ordinal': i} for i, e in enumerate(self.queries)]
                and json.loads((NATIVE/'group_main.json').read_text()) == self.catalogue['group_main'], 'native/prefix query or catalogue mismatch')
        self.native_manifest = native
        self.native = BoundedNpyRows(NATIVE/'m2.npy', expected_shape=(2900, 9333), expected_dtype=np.float32)
        for fold, digest in enumerate(CHECKPOINTS):
            path = OLD/f'actor_state_fold{fold}.pt'; require(file_hash(path) == digest, 'frozen causal model changed'); self.bindings[str(path)] = digest
        self.labels = None; self.bindings.update(self.base.bindings); self.revalidate()

    def enable_labels(self, input_lock_path):
        lock = json.loads(Path(input_lock_path).read_text(encoding='utf-8'))
        require(lock['state'] == 'inputs_frozen_before_label_access' and lock['prefix_report_sha256'] == file_hash(PREFIX/'report.json')
                and lock['bindings_sha256'] == self.bindings, 'durable input lock required before labels')
        self.base.selection_frozen = True
        self.labels = LabelGate(self.folds, self.past_by_fold, self.mapping,
            lambda name, rows: self.base.array(name)[rows])

    def revalidate(self):
        self.base.revalidate()
        for p, sha in self.bindings.items(): require(file_hash(p) == sha, 'proxy input binding changed')
