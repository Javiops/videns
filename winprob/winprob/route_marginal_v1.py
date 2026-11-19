"""Exact route-marginal update and finite recipe-frontier matching.

No datasets, model packages, labels, source runtime or file/network side effects.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def simplex(value, name, ndim):
    a = np.asarray(value, dtype=np.float64)
    require(a.ndim == ndim and all(a.shape) and np.isfinite(a).all() and np.all(a >= 0), name+' must be finite nonnegative probabilities')
    require(np.all(np.abs(a.sum(axis=-1)-1) <= 1e-12), name+' must already be normalized')
    return a


def bridge(prior, routing, desired_routes):
    """Batched exact bridge; no true action/goal label parameter exists."""
    p = simplex(prior, 'prior', 2); r = simplex(routing, 'routing', 2)
    alpha = simplex(desired_routes, 'desired route marginal', 2)
    require(p.shape[1] == r.shape[0] and alpha.shape == (len(p), r.shape[1]), 'action/route axes disagree')
    marginal = p @ r
    require(not np.any((marginal == 0) & (alpha > 0)), 'new route mass outside prior support')
    ratio = np.zeros_like(alpha); np.divide(alpha, marginal, out=ratio, where=marginal > 0)
    updated = p*(ratio @ r.T)
    neutral = np.all(alpha == marginal, axis=1)
    updated[neutral] = p[neutral]  # exact neutral probabilities, including ties
    require(np.isfinite(updated).all() and np.all(updated >= 0)
            and np.all(np.abs(updated.sum(1)-1) <= 2e-12)
            and not np.any((p == 0) & (updated != 0)), 'probability/support conservation failed')
    return updated, marginal


def joint_bridge(prior, routing, desired_routes):
    p = simplex(prior, 'prior', 1); r = simplex(routing, 'routing', 2)
    alpha = simplex(desired_routes, 'desired route marginal', 1)
    updated, marginal = bridge(p[None, :], r, alpha[None, :])
    original = p[:, None]*r; ratio = np.zeros_like(alpha)
    np.divide(alpha, marginal[0], out=ratio, where=marginal[0] > 0)
    joint = original*ratio[None, :]
    require(np.allclose(joint.sum(0), alpha, atol=2e-12, rtol=0)
            and np.allclose(joint.sum(1), updated[0], atol=2e-12, rtol=0), 'joint marginals disagree')
    return original, joint, updated[0], marginal[0]


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    require(p.shape == q.shape and np.isfinite(p).all() and np.isfinite(q).all()
            and np.all(p >= 0) and np.all(q >= 0)
            and abs(p.sum()-1) <= 2e-12 and abs(q.sum()-1) <= 2e-12, 'aligned probability distributions required')
    positive = p > 0
    if np.any(q[positive] == 0):
        return float('inf')
    return float(np.sum(p[positive]*(np.log(p[positive])-np.log(q[positive]))))


def canonical_bag(value):
    require(isinstance(value, (tuple, list)), 'canonical counted multiset required')
    pairs = tuple(tuple(pair) for pair in value)
    require(all(len(pair) == 2 and type(pair[0]) is int and pair[0] > 0
                and type(pair[1]) is int and pair[1] > 0 for pair in pairs)
            and pairs == tuple(sorted(pairs)) and len({i for i, _ in pairs}) == len(pairs), 'invalid counted multiset')
    return pairs


class RecipeFrontiers:
    """A descriptive recipe forest, not a purchase/legality simulator."""
    def __init__(self, recipes):
        require(isinstance(recipes, dict) and recipes, 'explicit public recipe mapping required')
        self.recipes = {}
        for node, children in recipes.items():
            require(type(node) is int and node > 0 and isinstance(children, (tuple, list))
                    and all(type(child) is int and child > 0 for child in children), 'invalid recipe node/children')
            self.recipes[node] = tuple(children)
        require(all(child in self.recipes for children in self.recipes.values() for child in children), 'recipe child missing from native world')
        active = set(); finished = set()
        def visit(node):
            require(node not in active, 'cyclic recipe graph')
            if node in finished:
                return
            active.add(node)
            for child in self.recipes[node]:
                visit(child)
            active.remove(node); finished.add(node)
        for node in self.recipes:
            visit(node)

    def _matcher(self, basket):
        basket = canonical_bag(basket)
        require(all(node in self.recipes for node, _ in basket), 'unknown basket item')
        require(sum(n for _, n in basket) <= 6, 'bounded six-copy compatibility prototype')
        names = tuple(i for i, _ in basket); caps = tuple(n for _, n in basket)
        zero = (0,)*len(names); positions = {node: i for i, node in enumerate(names)}
        def product(left, right):
            result = set()
            for a in left:
                for b in right:
                    v = tuple(x+y for x, y in zip(a, b))
                    if all(x <= cap for x, cap in zip(v, caps)):
                        result.add(v)
            return result
        @lru_cache(None)
        def tree(node):
            result = {zero}
            if node in positions:
                one = list(zero); one[positions[node]] = 1; result.add(tuple(one))
            children = self.recipes[node]
            if children:
                combined = {zero}
                for child in children:  # duplicate children represent separate capacity
                    combined = product(combined, tree(child))
                result.update(combined)
            return frozenset(result)
        def forest(goals):
            goals = canonical_bag(goals)
            require(all(node in self.recipes for node, _ in goals) and sum(n for _, n in goals) <= 6, 'unknown or unbounded goal forest')
            result = {zero}
            for goal, copies in goals:
                for _ in range(copies):
                    result = product(result, tree(goal))
            return tuple(sorted(result))
        return forest

    def matched_frontiers(self, basket, goals):
        return self._matcher(basket)(goals)

    def maximum_matched_copies(self, basket, goals):
        return max(map(sum, self.matched_frontiers(basket, goals)), default=0)

    def match_many(self, basket, goal_forests):
        """Share the exact subtree cache across public goals for one basket."""
        matcher = self._matcher(basket)
        return np.asarray([max(map(sum, matcher(goal)), default=0) for goal in goal_forests], dtype=np.uint8)
