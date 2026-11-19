"""Bind verified kill graphs to unchanged native M2 inputs; targets stay deferred."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash, json_file
from winprob.recipe_transition_data_v1 import RecipeData, array_hash, require, value_hash
from winprob.recipe_transition_v1 import ChoiceInputs
from winprob.personal_kill_graph_v2 import degree_control, control_seed, excess_profile
from winprob.personal_graph_models_v1 import ARMS, public_profiles

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/personal_kill_graph_probe_protocol_v1.md'
PROTOCOL_SHA = 'cdd3c2f163f10b6862eae9737a130a96258768629d9a31eefca070d93e663d91'
NATIVE = ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2'
GRAPH = ROOT.parent/'build_tracker/data/exports/winprob-personal-kill-graph/personal-kill-graph-20261003-v2'
GRAPH_SHA = '771d1888d270285a4651ff0dbdec8f375acdd54c4ad6828d1dfd2b910b52eed8'
AUDIT = ROOT/'data/runs/personal-kill-graph-audit-20261003-v2/audit'
AUDIT_SHA = '490ba260928feec51cc79596e087dd552edf446452513e1befd14101e3c0a6bd'


class GraphData:
    def __init__(self):
        require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'fixed graph prediction protocol changed')
        # This is an input validator only: no old net-recipe model is fitted or loaded.
        self.base = RecipeData(NATIVE)
        self.bindings = dict(self.base.bindings); self.bindings[str(PROTOCOL)] = PROTOCOL_SHA
        self.entries = self.base.entries; self.folds = self.base.folds; self.focal = self.base.focal
        self.game_ids = self.base.game_ids; self.group_ids = self.base.group_ids
        self.digest = self.base.digest; self.target_file_opened = False
        gm, gh = bound_json(GRAPH/'manifest.json', self.bindings)
        am, ah = bound_json(AUDIT/'report.json', self.bindings)
        require(gh == GRAPH_SHA and ah == AUDIT_SHA and am['passed'] and gm['state'] == am['state'] == 'complete'
                and am['source_manifest_sha256'] == gh and am['states'] == gm['states'] == 56288
                and am['exact_KDA_graph_roster_temporal_replay'] and am['all_null_row_column_degrees_preserved']
                and am['necessary_three_pp_capacity_gate_passed'], 'verified information artifacts changed')
        for record in (gm, am):
            for raw, expected in record['code_sha256'].items():
                require(file_hash(raw) == expected, 'consumed graph code changed'); self.bindings[raw] = expected
        arrays = {}
        for key in ('counts', 'teams', 'champions'):
            p = GRAPH/(key+'.npy'); require(file_hash(p) == gm['files'][p.name], 'source graph member changed')
            self.bindings[str(p)] = gm['files'][p.name]; arrays[key] = np.load(p, mmap_mode='r', allow_pickle=False)
        p = AUDIT/'null_counts.npy'; require(file_hash(p) == am['files'][p.name], 'fixed null changed')
        self.bindings[str(p)] = am['files'][p.name]; null = np.load(p, mmap_mode='r', allow_pickle=False)
        require(arrays['counts'].shape == null.shape == (56288, 10, 10)
                and arrays['teams'].shape == arrays['champions'].shape == (56288, 10), 'graph axes changed')
        static_path = NATIVE/'native_descriptors.json'; static = json_file(static_path)
        require(file_hash(static_path) == self.base.manifest['files'][static_path.name], 'native static descriptors changed')
        self.bindings[str(static_path)] = self.base.manifest['files'][static_path.name]
        self.graph_contexts = {arm: np.zeros((56288, 17), dtype=np.float64) for arm in ARMS}
        context_path = self.base.context_directory/'contexts.jsonl.gz'
        require(file_hash(context_path) == self.base.context_manifest['files'][context_path.name], 'certified public contexts changed')
        seen = np.zeros(56288, dtype=bool)
        with gzip.open(context_path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); i = row['ordinal']; entry = self.entries[i]
                require(not seen[i] and all(row[k] == v for k, v in entry.items()), 'graph/context query identity differs')
                teams = arrays['teams'][i]; champions = arrays['champions'][i]; graph = arrays['counts'][i]
                control = degree_control(graph, teams, control_seed(entry))
                require(np.array_equal(control, null[i]), 'control does not reproduce fixed identity seed')
                profiles = public_profiles(row['source_projection'], teams, champions, entry['participant_id'],
                                           static['items'], static['champion_tags'])
                self.graph_contexts['personal'][i] = excess_profile(graph, teams, entry['participant_id'], profiles)
                self.graph_contexts['degree_null'][i] = excess_profile(control, teams, entry['participant_id'], profiles)
                seen[i] = True
        require(bool(seen.all()), 'missing new profile row')
        self.candidates = np.array(self.base.inputs.candidates[:, :, :16], copy=True)
        self.reference = self.base.inputs.reference
        self.zero_profiles = {arm: np.all(value == 0, axis=1) for arm, value in self.graph_contexts.items()}
        self.active_focals = int((~self.zero_profiles['personal'][self.focal]).sum())
        require(self.active_focals >= 87, 'actual projected profiles fail necessary3pp capacity bound')
        self.graph_feature_digests = {**{arm: array_hash(v) for arm, v in self.graph_contexts.items()},
                                     'candidate_raw16': array_hash(self.candidates), 'reference': array_hash(self.reference)}
        for arm in ARMS:
            ChoiceInputs(self.reference, self.graph_contexts[arm], self.candidates).validate()
        self.revalidate()

    def batch(self, rows, arm):
        require(arm in ARMS, 'fixed graph arm required')
        idx = np.asarray(rows)
        require(idx.ndim == 1 and np.issubdtype(idx.dtype, np.integer) and np.all((idx >= 0) & (idx < len(self.entries))), 'invalid graph state indices')
        return ChoiceInputs(self.reference[idx], self.graph_contexts[arm][idx], self.candidates[idx]).validate()

    def revalidate(self):
        self.base.revalidate()
        for raw, expected in self.bindings.items():
            require(file_hash(raw) == expected, 'graph input binding changed')
        if hasattr(self, 'graph_feature_digests'):
            actual = {**{arm: array_hash(v) for arm, v in self.graph_contexts.items()},
                      'candidate_raw16': array_hash(self.candidates), 'reference': array_hash(self.reference)}
            require(actual == self.graph_feature_digests, 'new graph input arrays changed in memory')

    def input_lock(self):
        require(not self.target_file_opened, 'input lock must precede target access')
        return {'state': 'inputs_frozen', 'native_manifest_sha256': self.digest,
                'feature_digests': self.base.feature_digests, 'graph_feature_digests': self.graph_feature_digests,
                'source_graph_sha256': GRAPH_SHA, 'information_audit_sha256': AUDIT_SHA,
                'target_file_opened': False, 'active_personal_focals': self.active_focals,
                'zero_profile_counts': {arm: int(z.sum()) for arm, z in self.zero_profiles.items()},
                'zero_profile_focals': {arm: int(z[self.focal].sum()) for arm, z in self.zero_profiles.items()}}

    def load_targets(self, lock_path):
        lock = json_file(lock_path)
        require(lock['graph_feature_digests'] == self.graph_feature_digests and lock['source_graph_sha256'] == GRAPH_SHA
                and lock['information_audit_sha256'] == AUDIT_SHA, 'graph inputs not frozen before targets')
        self.revalidate(); labels = self.base.load_targets(lock_path)
        self.target_file_opened = True; self.bindings.update(self.base.bindings)
        return labels
