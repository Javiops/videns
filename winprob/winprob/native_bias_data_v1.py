"""Admitted native probabilities and separately enabled opposite-fold counts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.recipe_transition_data_v1 import RecipeData
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.native_bias_calibration_v1 import require, training_counts, control_permutation

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/native_bias_calibration_protocol_v1.md'
PROTOCOL_SHA = '8e91013fc3f637684605a84e43ed755e2bf3edf68bb5c57a5b83e56efdb6ce85'
REFERENCE = ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2'
PRIOR = ROOT.parent/'build_tracker/data/exports/winprob-route-prior/route-prior-20261003-v1/manifest.json'
ARMS = ('calibrated', 'permuted')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def opposite_fold_rows(folds, heldout):
    f = np.asarray(folds)
    require(type(heldout) is int and heldout in (0, 1) and f.ndim == 1 and np.all(np.isin(f, [0, 1])),
            'declared whole-game fold required')
    rows = np.flatnonzero(f != heldout)
    require(len(rows) > 0 and np.all(f[rows] != heldout), 'empty or held-out training request')
    return rows


class NativeBiasData:
    def __init__(self):
        require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'fixed native calibration protocol changed')
        self.base = RecipeData(REFERENCE)  # Performs metadata admission before any source payload.
        self.bindings = dict(self.base.bindings); self.bindings[str(PROTOCOL)] = PROTOCOL_SHA
        self.prior_manifest, self.prior_sha = bound_json(PRIOR, self.bindings)
        m = self.prior_manifest
        require(self.prior_sha == '35113430e6997ec1ef993da5d9dc3f9325941138e21db24fab300ea3b5f470ac'
                and m['state'] == 'complete' and m['states'] == 56288
                and m['complete_probability_rows_replayed_exactly'] == 56288,
                'exact complete native priors required')
        p = PRIOR.parent/'full_probability.npy'
        require(file_hash(p) == m['files'][p.name]
                == 'a2c4b14975027a7aafafe5a4a879322c2ea74f2116f9fd021d3128be4641b678', 'full native prior changed')
        self.bindings[str(p)] = m['files'][p.name]
        self.priors = BoundedNpyRows(p, expected_shape=(56288, 9333), expected_dtype=np.float32, max_rows=256)
        self.entries = self.base.entries; self.folds = self.base.folds
        self.focals = self.base.focal; self.game_ids = self.base.game_ids
        self.sizes = np.array([sum(count for _, count in group) for group in self.base.groups], dtype=np.int16)
        self.permutation = control_permutation(self.sizes)
        require(self.sizes.shape == (9333,) and self.sizes[0] == 0
                and len(self.focals) == 2900 and len(set(self.game_ids)) == 3518, 'frozen catalogue/population changed')
        for fold in (0, 1):
            train = opposite_fold_rows(self.folds, fold)
            require(not set(self.game_ids[train]) & set(self.game_ids[self.folds == fold]), 'whole-game fold overlap')
            _, counts = np.unique(self.game_ids[train], return_counts=True)
            require(np.all(counts == 16), 'unequal game contribution to summed likelihood')
        self.labels = None
        self.revalidate()

    def input_lock(self):
        require(self.labels is None, 'input lock must precede the target stage')
        return {'state': 'inputs_frozen', 'native_manifest_sha256': self.base.digest,
            'feature_digests': self.base.feature_digests, 'prior_manifest_sha256': self.prior_sha,
            'protocol_sha256': PROTOCOL_SHA, 'folds_sha256': array_hash(self.folds),
            'class_permutation_sha256': array_hash(self.permutation), 'bindings_sha256': self.bindings,
            'purchase_targets_opened': False}

    def load_targets(self, lock_path):
        require(self.labels is None, 'targets already enabled')
        lock = json.loads(Path(lock_path).read_text(encoding='utf-8'))
        expected = self.input_lock()
        require(all(lock.get(k) == v for k, v in expected.items()), 'calibration input lock changed')
        self.labels = self.base.load_targets(lock_path)
        self.bindings.update(self.base.bindings)

    def counts_for(self, heldout, guard=lambda: None):
        require(self.labels is not None, 'counts require the explicit post-preflight target stage')
        rows = opposite_fold_rows(self.folds, heldout); result = np.zeros(9333, dtype=np.int64)
        for lo in range(0, len(rows), 256):
            guard(); selected = rows[lo:lo+256]
            require(np.all(self.folds[selected] != heldout), 'held-out purchase requested for calibration')
            result += training_counts(self.priors[selected], self.labels['main'][selected])
        require(result.sum() == len(rows), 'training count accounting failed')
        ids = sorted(str(v) for v in set(self.game_ids[rows]))
        return result, {'scored_fold': heldout, 'training_rows': len(rows), 'training_games': len(ids),
            'training_game_ids': ids, 'training_game_ids_sha256': value_hash(ids),
            'training_ordinals_sha256': array_hash(rows), 'counts_sha256': array_hash(result),
            'positive_target_support_checked_rows': len(rows), 'heldout_targets_used': False}

    def revalidate(self):
        self.base.revalidate()
        for p, sha in self.bindings.items():
            require(file_hash(p) == sha, 'calibration input binding changed')
