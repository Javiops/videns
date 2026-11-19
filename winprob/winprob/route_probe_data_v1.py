"""Admitted full M2 priors and target-separated inputs for the route trial."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.recipe_transition_data_v1 import RecipeData, require, array_hash
from winprob.route_probe_v1 import ARMS, kernels, normalized_prior

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2'
PRIOR = ROOT.parent/'build_tracker/data/exports/winprob-route-prior/route-prior-20261003-v1/manifest.json'
PROTOCOL = ROOT/'docs/route_probe_protocol_v1.md'
PROTOCOL_SHA = 'f6e0a8de1fdaffa76dc7775457cb1f618058c384858eafe71997cb4d5d23ec18'
CENSUS = ROOT/'data/runs/route-static-census-20261003-v1/census/report.json'
AUDIT = ROOT/'data/runs/route-static-audit-20261003-v1/audit/report.json'


class RouteData:
    def __init__(self, cache_directory, guard, status):
        require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'route protocol changed')
        # Metadata-only completion checks before the old loader can read inputs.
        prior, prior_sha = bound_json(PRIOR, {})
        require(prior['schema'] == 'route-prior-native-m2-v1' and prior['state'] == 'complete'
                and prior['states'] == 56288 and prior['games'] == 3518
                and prior['complete_probability_rows_replayed_exactly'] == 56288
                and prior['saved_rows_verified_exactly'] == 56288
                and prior['full_probability_shape'] == [56288, 9333]
                and prior['full_probability_dtype'] == 'float32'
                and prior['route_protocol_sha256'] == 'f76ffa31a92ea3d3d8d0a563366dee03478a1184050cda703f83cf326502d2cb'
                and prior['prior_release_manifest_sha256'] == 'f0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe',
                'incomplete/different full-prior release')
        for flag in ('target_file_opened', 'source_trajectory_payload_reads', 'source_database_reads',
                     'test_payload_reads', 'confirmation_payload_reads', 'consumer_runtime_imported',
                     'training_performed', 'outcomes_analyzed', 'accuracy_computed'):
            require(prior[flag] is False, 'prior boundary failed: '+flag)
        self.base = RecipeData(OLD)  # Performs admission/reservation checks first.
        self.bindings = dict(self.base.bindings)
        self.bindings.update({str(PRIOR): prior_sha, str(PROTOCOL): PROTOCOL_SHA})
        for path, sha in {**prior['bindings_sha256'], **prior['code_sha256']}.items():
            require(Path(path).name != 'targets.jsonl' and Path(path).suffix.lower() not in ('.db', '.sqlite'),
                    'prior lineage contains forbidden payload')
            require(file_hash(path) == sha, 'prior input/code changed'); self.bindings[path] = sha
        require(file_hash(CENSUS) == '08b426d97260521b52c9271c235f705a60c3b899ac9d449fddf67a1a8c951def'
                and file_hash(AUDIT) == 'f1e6d16d642cede7238d0a3b142de42a2a29b7d1b1732dab212325e539cc014f',
                'public geometry evidence changed')
        census = json.loads(CENSUS.read_text()); audit = json.loads(AUDIT.read_text())
        require(audit['passed'] is True and census['state'] == 'complete', 'public audit incomplete')
        for path in (CENSUS, AUDIT): self.bindings[str(path)] = file_hash(path)
        matrix_path = CENSUS.parent/'fraction_units.npy'
        require(file_hash(matrix_path) == 'd5157ae14c1b7267306686dc3bdcca32fd2f40b1442375cd4736db6d402cf572', 'public matrix changed')
        self.bindings[str(matrix_path)] = file_hash(matrix_path)
        sizes = np.array([sum(c for _, c in b) for b in self.base.groups], dtype=np.int32)
        self.kernels, self.permutation = kernels(np.load(matrix_path, mmap_mode='r', allow_pickle=False), sizes)
        require(all(r.shape == (9333, 258) for r in self.kernels.values()) and self.permutation[0] == 0, 'route axes changed')
        full_path = PRIOR.parent/'full_probability.npy'
        require(file_hash(full_path) == prior['files'][full_path.name], 'native full prior changed')
        self.bindings[str(full_path)] = prior['files'][full_path.name]
        self.full = np.load(full_path, mmap_mode='r', allow_pickle=False)
        require(self.full.dtype == np.float32 and self.full.shape == (56288, 9333), 'native full prior schema changed')
        self.entries = self.base.entries; self.folds = self.base.folds; self.focal = self.base.focal
        self.game_ids = self.base.game_ids; self.context = self.base.inputs.context
        self.prior_manifest, self.prior_sha = prior, prior_sha
        self.cache = Path(cache_directory).resolve(); self.cache.mkdir(parents=True, exist_ok=False)
        self.marginals = {}; self.cache_files = {}
        np.save(self.cache/'row_permutation.npy', self.permutation, allow_pickle=False)
        for arm in ARMS:
            np.save(self.cache/f'kernel_{arm}.npy', self.kernels[arm], allow_pickle=False)
            self.marginals[arm] = np.lib.format.open_memmap(self.cache/f'marginal_{arm}.npy',
                mode='w+', shape=(len(self.entries), 258), dtype=np.float64)
        for lo in range(0, len(self.entries), 128):
            guard(); hi = min(lo+128, len(self.entries)); p = normalized_prior(self.full[lo:hi])
            for local, absolute in enumerate(range(lo, hi)):
                ids = self.base.group_ids[absolute]; ids = ids[ids >= 0]
                tail = np.ones(9333, dtype=bool); tail[ids] = False
                coarse = [*p[local, ids], *([0.]*(3-len(ids))), float(p[local, tail].sum())]
                require(np.allclose(coarse, self.base.inputs.reference[absolute], atol=2e-15, rtol=0),
                        'full prior does not reproduce original coarsened baseline')
            for arm in ARMS:
                marginal = p @ self.kernels[arm]
                require(np.isfinite(marginal).all() and np.all(marginal > 0)
                        and np.all(np.abs(marginal.sum(1)-1) < 1e-12), 'route marginal conservation failed')
                self.marginals[arm][lo:hi] = marginal
            if hi % 2048 == 0 or hi == len(self.entries):
                for matrix in self.marginals.values(): matrix.flush()
                status('building_full_route_marginals', completed=hi, states=len(self.entries))
        for arm in ARMS:
            self.marginals[arm].flush()
            self.marginals[arm] = np.load(self.cache/f'marginal_{arm}.npy', mmap_mode='r', allow_pickle=False)
        self.cache_files = {p.name: file_hash(p) for p in self.cache.glob('*.npy')}
        self.feature_digests = {'context': array_hash(self.context), 'folds': array_hash(self.folds),
            'focal': array_hash(self.focal), 'kernels': {arm: array_hash(r) for arm, r in self.kernels.items()},
            'cache_files': self.cache_files}
        self.revalidate()

    def input_lock(self):
        return {'state': 'inputs_frozen', 'native_manifest_sha256': self.base.digest,
            'feature_digests': self.base.feature_digests, 'route_feature_digests': self.feature_digests,
            'route_prior_manifest_sha256': self.prior_sha, 'route_protocol_sha256': PROTOCOL_SHA,
            'bindings_sha256': self.bindings, 'targets_opened': False}

    def load_targets(self, lock_path):
        lock = json.loads(Path(lock_path).read_text())
        require(lock['route_feature_digests'] == self.feature_digests and lock['route_prior_manifest_sha256'] == self.prior_sha,
                'route inputs must freeze before targets')
        self.revalidate(); labels = self.base.load_targets(lock_path)
        for lo in range(0, len(self.entries), 128):
            hi = min(lo+128, len(self.entries)); values = self.full[np.arange(lo, hi), labels['main'][lo:hi]]
            require(np.isfinite(values).all() and np.all(values > 0), 'observed full target has zero native prior; no epsilon/drop allowed')
        return labels

    def revalidate(self):
        self.base.revalidate()
        for path, sha in self.bindings.items(): require(file_hash(path) == sha, 'route input binding changed')
        for name, sha in self.cache_files.items(): require(file_hash(self.cache/name) == sha, 'route cache changed')
        if hasattr(self, 'feature_digests'):
            require(array_hash(self.context) == self.feature_digests['context']
                    and array_hash(self.folds) == self.feature_digests['folds']
                    and array_hash(self.focal) == self.feature_digests['focal']
                    and {arm: array_hash(r) for arm, r in self.kernels.items()} == self.feature_digests['kernels'],
                    'in-memory route features changed')

