"""Fixed-recipe ensemble, metrics and game-level paired bootstrap."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


class Ensemble:
    """Mean of standardized logistic regression and LightGBM (protocol v1 recipe)."""

    def __init__(self, seed: int = 7):
        self.lr = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=3000))
        self.gb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=200, colsample_bytree=0.8, random_state=seed,
                                     n_jobs=1, verbose=-1)

    def fit(self, X, y):
        self.lr.fit(X, y)
        self.gb.fit(X, y)
        return self

    def parts(self, X):
        return self.lr.predict_proba(X)[:, 1], self.gb.predict_proba(X)[:, 1]

    def predict(self, X):
        a, b = self.parts(X)
        return 0.5 * (a + b)


class Coherent:
    """Antisymmetric wrapper: trained on both orientations, p = (f(x) + 1 - f(x_swap)) / 2."""

    def __init__(self, seed: int = 7):
        self.m = Ensemble(seed)

    def fit(self, x, xs, y):
        self.m.fit(np.vstack([x, xs]), np.concatenate([y, 1 - y]))
        return self

    def predict(self, x, xs):
        return 0.5 * (self.m.predict(x) + 1.0 - self.m.predict(xs))

    def predict_parts(self, x, xs):
        (a1, b1), (a2, b2) = self.m.parts(x), self.m.parts(xs)
        return 0.5 * (a1 + 1 - a2), 0.5 * (b1 + 1 - b2)


def game_logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration(y, p):
    z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    a, s = calibration(y, p)
    return {"n": int(len(y)), "logloss": float(game_logloss(y, p).mean()),
            "brier": float(np.mean((p - y) ** 2)), "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
            "auc": float(roc_auc_score(y, p)), "cal_intercept": a, "cal_slope": s, "ece": ece(y, p),
            "mean_pred": float(p.mean()), "observed": float(y.mean())}


def paired_bootstrap(loss_a, loss_b, n_boot=2000, seed=11) -> dict:
    """Mean per-game loss difference a - b with a 95% percentile interval (games resampled)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(0, n_boot, 200):
        idx = rng.integers(0, len(d), size=(min(200, n_boot - i), len(d)))
        stats[i:i + len(idx)] = d[idx].mean(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "resolved": bool(lo > 0 or hi < 0)}
