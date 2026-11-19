"""Train-only standardization and fixed controls for the native-M2 screen."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from winprob.recipe_transition_v1 import ChoiceInputs, CONTEXT_NAMES, CANDIDATE_NAMES

ARMS = ('raw', 'net', 'shuffled_net')
NET_START = 16


def game_weights(game_ids):
    ids = np.asarray(game_ids)
    if ids.ndim != 1 or not len(ids):
        raise ValueError('nonempty game identities required')
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1. / counts[inverse]
    return weight / weight.sum()


def shuffle_for_entry(entry, present_count):
    if type(present_count) is not int or not 1 <= present_count <= 3:
        raise ValueError('one to three native candidates required')
    salt = f"recipe-transition-v1/shuffle/{entry['match_id']}|{entry['participant_id']}|{entry['source_event_index']}"
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], 'big')
    value = np.arange(3, dtype=np.int64)
    value[:present_count] = np.random.default_rng(seed).permutation(present_count)
    return value


@dataclass(frozen=True)
class Standardizer:
    context_mean: np.ndarray
    context_scale: np.ndarray
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray

    @classmethod
    def fit(cls, train, train_game_ids):
        """Caller supplies ONLY one training fold, never held-out states/targets."""
        train.validate()
        if train.parameter_shape != (len(CONTEXT_NAMES), len(CANDIDATE_NAMES)):
            raise ValueError('fixed context/candidate schema required')
        h, f = train.context, train.candidates
        if not np.array_equal(h[:, 0], np.ones(len(h))) or len(train_game_ids) != len(h):
            raise ValueError('intercept or training identity mismatch')
        weight = game_weights(train_game_ids)
        hmean = weight @ h
        hvar = weight @ (h-hmean)**2
        hscale = np.sqrt(np.maximum(hvar, 0)); hscale[hvar < 1e-12] = 1.
        hmean[0], hscale[0] = 0., 1.
        present = train.reference[:, :3] > 0
        cw = weight[:, None] * present / present.sum(1, keepdims=True)
        fmean = np.einsum('nk,nkf->f', cw, f)
        fvar = np.einsum('nk,nkf->f', cw, (f-fmean)**2)
        fscale = np.sqrt(np.maximum(fvar, 0)); fscale[fvar < 1e-12] = 1.
        return cls(hmean, hscale, fmean, fscale)

    def validate(self):
        expected = (len(CONTEXT_NAMES), len(CANDIDATE_NAMES))
        for mean, scale, n in ((self.context_mean, self.context_scale, expected[0]),
                               (self.candidate_mean, self.candidate_scale, expected[1])):
            if mean.shape != (n,) or scale.shape != (n,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('finite positive standardization geometry required')
        if self.context_mean[0] != 0 or self.context_scale[0] != 1:
            raise ValueError('intercept standardization changed')

    def transform(self, inputs, arm, permutations=None):
        inputs.validate(); self.validate()
        if inputs.parameter_shape != (len(CONTEXT_NAMES), len(CANDIDATE_NAMES)) or arm not in ARMS:
            raise ValueError('fixed arm/input schema required')
        if not np.array_equal(inputs.context[:, 0], np.ones(len(inputs.context))):
            raise ValueError('inference intercept must remain one')
        present = inputs.reference[:, :3] > 0
        f = inputs.candidates.copy()
        if arm == 'shuffled_net':
            perm = np.asarray(permutations)
            if perm.shape != present.shape or not np.issubdtype(perm.dtype, np.integer):
                raise ValueError('fixed per-state permutation required')
            for row, mask in zip(perm, present, strict=True):
                count = int(mask.sum())
                if not np.array_equal(np.sort(row), np.arange(3)) or not np.array_equal(np.sort(row[:count]), np.arange(count)) or not np.array_equal(row[count:], np.arange(count, 3)):
                    raise ValueError('control permutes padding or changes candidate set')
            f[:, :, NET_START:] = np.take_along_axis(f[:, :, NET_START:], perm[:, :, None], axis=1)
        elif permutations is not None:
            raise ValueError('permutations belong only to shuffled_net')
        f = (f-self.candidate_mean) / self.candidate_scale
        if arm == 'raw':
            f[:, :, NET_START:] = 0.
        f[~present] = 0.
        h = (inputs.context-self.context_mean) / self.context_scale
        return ChoiceInputs(inputs.reference.copy(), h, f).validate()

    def as_dict(self):
        return {name: getattr(self, name).tolist() for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values):
        if set(values) != set(cls.__dataclass_fields__):
            raise ValueError('complete scaler state required')
        obj = cls(**{k: np.asarray(v, dtype=np.float64) for k, v in values.items()})
        obj.validate()
        return obj
