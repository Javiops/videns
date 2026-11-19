"""Public enemy profiles and edge-only, exactly baseline-preserving correction."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from winprob.ipo_residual import ACTION_STATS, CHAMPION_TAGS
from winprob.recipe_transition_v1 import ChoiceInputs, predict, _logits
from winprob.recipe_transition_probe_v1 import game_weights

ARMS = ('personal', 'degree_null')
PROFILE_NAMES = tuple('tag_'+t for t in CHAMPION_TAGS) + tuple(k for k, _ in ACTION_STATS) + ('level_over_18',)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def public_profiles(projection, teams, champions, buyer_pid, items, champion_tags):
    """Only public champion/item/level fields; no roles, gold, ages or targets."""
    teams = np.asarray(teams); champions = np.asarray(champions)
    require(teams.shape == champions.shape == (10,) and type(buyer_pid) is int and 1 <= buyer_pid <= 10, 'complete public roster required')
    me = {'champion_id': projection['champion_id'], 'team_id': projection['team_id'],
          'items': projection['inventory'], 'level': projection['level']}
    players = [me, *projection['others']]
    require(len(players) == 10, 'complete context roster required')
    lookup = {(p['team_id'], p['champion_id']): p for p in players}
    require(len(lookup) == 10 and me['champion_id'] == champions[buyer_pid-1]
            and me['team_id'] == teams[buyer_pid-1], 'buyer or duplicate public identity mismatch')
    result = np.zeros((10, 17), dtype=np.float64)
    for i, (team, cid) in enumerate(zip(teams, champions, strict=True)):
        require((int(team), int(cid)) in lookup and str(cid) in champion_tags, 'unmatched public champion')
        p = lookup[(int(team), int(cid))]; inventory = p['items']; level = p['level']
        require(isinstance(inventory, list) and all(type(v) is int and v > 0 for v in inventory), 'observed inventory multiset required')
        require(not isinstance(level, bool) and isinstance(level, (int, float)) and np.isfinite(level) and 1 <= level <= 30, 'observed level required')
        result[i, :6] = [float(t in champion_tags[str(cid)]) for t in CHAMPION_TAGS]
        for iid, count in Counter(inventory).items():
            require(str(iid) in items, 'item absent from pinned native static world')
            stats = items[str(iid)].get('stats', {})
            for j, (key, scale) in enumerate(ACTION_STATS, start=6):
                value = stats.get(key, 0)
                require(not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value), 'finite numeric item descriptor required')
                result[i, j] += count*value/scale
        result[i, -1] = level/18.
    return result


@dataclass(frozen=True)
class RMSScaler:
    context_scale: np.ndarray
    candidate_scale: np.ndarray

    @classmethod
    def fit(cls, inputs, game_ids):
        inputs.validate()
        require(inputs.parameter_shape == (17, 16) and len(game_ids) == len(inputs.reference), 'fixed graph geometry/identities required')
        weights = game_weights(game_ids)
        hscale = np.sqrt(np.maximum(weights @ (inputs.context**2), 0))
        present = inputs.reference[:, :3] > 0
        candidate_weights = weights[:, None]*present/present.sum(1, keepdims=True)
        fscale = np.sqrt(np.maximum(np.einsum('nk,nkf->f', candidate_weights, inputs.candidates**2), 0))
        hscale[hscale < 1e-12] = 1.; fscale[fscale < 1e-12] = 1.
        return cls(hscale, fscale)

    def validate(self):
        for value, shape in ((self.context_scale, (17,)), (self.candidate_scale, (16,))):
            require(value.shape == shape and np.isfinite(value).all() and np.all(value > 0), 'positive finite RMS scales required')

    def transform(self, inputs):
        inputs.validate(); self.validate(); require(inputs.parameter_shape == (17, 16), 'fixed graph input geometry')
        return ChoiceInputs(inputs.reference.copy(), inputs.context/self.context_scale,
                            inputs.candidates/self.candidate_scale).validate()

    def as_dict(self):
        return {'context_scale': self.context_scale.tolist(), 'candidate_scale': self.candidate_scale.tolist()}

    @classmethod
    def from_dict(cls, value):
        require(set(value) == {'context_scale', 'candidate_scale'}, 'uncentered RMS state required')
        result = cls(np.asarray(value['context_scale'], dtype=np.float64), np.asarray(value['candidate_scale'], dtype=np.float64))
        result.validate(); return result


def predict_graph(inputs, scaler, theta):
    """Target-free; zero graph information is bit-for-bit unchanged M2."""
    transformed = scaler.transform(inputs); theta = np.asarray(theta, dtype=np.float64)
    p, order = predict(transformed, theta)
    logits = _logits(transformed, theta, slice(None)); logp = logits-logsumexp(logits, axis=1, keepdims=True)
    unchanged = np.all(inputs.context == 0, axis=1)
    if not np.any(theta):
        unchanged[:] = True
    if unchanged.any():
        reference = inputs.reference[unchanged]
        lp = np.full_like(reference, -np.inf); np.log(reference, out=lp, where=reference > 0)
        p[unchanged] = reference; logp[unchanged] = lp; order[unchanged] = [0, 1, 2]
    require(np.allclose(np.exp(logp), p, atol=1e-14, rtol=1e-12), 'stable graph log probabilities disagree')
    return {'probabilities': p, 'log_probabilities': logp, 'candidate_order': order}
