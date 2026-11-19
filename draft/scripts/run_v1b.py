"""Exploratory addendum v1b (declared after v1 results; development only, no confirmation).

(a) structure without experience: I_noexp vs C_noexp;
(b) shrinkage priors frozen from the full fit and reused in every fold;
(c) light iTero-style shrinkage k=50 on pair tables (matchup, cross, synergy, matchup gold).
Gate: the 'v1' variant must reproduce run_v1's C and I_top log loss exactly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import draft.tables as tables  # noqa: E402
from draft.data import load_games  # noqa: E402
from draft.features import exp_features  # noqa: E402
from draft.models import Coherent, Ensemble, game_logloss, metrics, paired_bootstrap  # noqa: E402
from run_v1 import EXTRACT, FOLDS, PATCHES, SPLIT, feature_sets  # noqa: E402

OUT = ROOT / "data/runs/v1b"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_sets(Gf, Gd, fold, exp_mean, fixed_k):
    tables.FIXED_K = fixed_k
    fit_sets = {}
    for k in range(FOLDS):
        fs = feature_sets(tables.build_tables(Gf.subset(np.where(fold != k)[0])),
                          Gf.subset(np.where(fold == k)[0]), exp_mean)
        for name in ("persp", "C", "C_noexp"):
            v = fs[name]
            if name not in fit_sets:
                fit_sets[name] = (np.zeros((len(Gf),) + v.shape[1:], np.float32) if name == "persp" else
                                  tuple(np.zeros((len(Gf), a.shape[1]), np.float32) for a in v))
            if name == "persp":
                fit_sets[name][fold == k] = v
            else:
                for dst, src in zip(fit_sets[name], v):
                    dst[fold == k] = src
    T = tables.build_tables(Gf)
    dev_sets = feature_sets(T, Gd, exp_mean)
    tables.FIXED_K = None
    return fit_sets, dev_sets, T


def fit_variant(fit_sets, dev_sets, yf, nd):
    preds = {}
    P, Pd = fit_sets["persp"], dev_sets["persp"]
    yp = np.stack([yf, 1 - yf], 1)[:, :, None].repeat(5, 2).ravel()
    for tag, cols in (("", slice(None)), ("_noexp", slice(0, -2))):
        m = Ensemble().fit(P[..., cols].reshape(len(yp), -1), yp)
        p = m.predict(Pd[..., cols].reshape(nd * 10, -1)).reshape(nd, 2, 5)
        preds[f"I{tag}_top"] = p[:, 0, 0]
        preds[f"I{tag}_avg10"] = 0.5 * (p[:, 0].mean(1) + 1 - p[:, 1].mean(1))
    for name in ("C", "C_noexp"):
        (a, b), (c, d) = fit_sets[name], dev_sets[name]
        preds[name] = Coherent().fit(a, b, yf).predict(c, d)
    return preds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_games(EXTRACT, SPLIT)
    in_scope = np.isin(G.patch, list(PATCHES))
    Gf = G.subset(np.where((G.role == "fit") & in_scope)[0])
    Gd = G.subset(np.where((G.role == "dev") & in_scope)[0])
    known = Gf.exp_total > 0
    ecf, etf = exp_features(Gf.exp_champ, Gf.exp_total)
    exp_mean = (float(ecf[known].mean()), float(etf[known].mean()))
    fold = np.random.default_rng(11).integers(0, FOLDS, size=len(Gf))
    yf, yd = Gf.blue_win.astype(int), Gd.blue_win.astype(float)

    full_k = tables.build_tables(Gf).prior_k
    variants = {"v1": None, "frozen": dict(full_k),
                "k50": {**full_k, "matchup": 50.0, "cross": 50.0, "syn": 50.0, "mgold": 50.0}}
    all_preds, report = {}, {"full_fit_prior_k": full_k, "variants": {}}
    for vname, fk in variants.items():
        fs, ds, T = build_sets(Gf, Gd, fold, exp_mean, fk)
        preds = fit_variant(fs, ds, yf, len(Gd))
        for k, v in preds.items():
            all_preds[f"{vname}:{k}"] = v
        report["variants"][vname] = {"fixed_k": fk, "metrics": {k: metrics(yd, v) for k, v in preds.items()}}
        log(f"{vname}: " + " ".join(f"{k}={metrics(yd, v)['logloss']:.5f}" for k, v in preds.items()))

    v1 = json.loads((ROOT / "data/runs/v1/report.json").read_text(encoding="utf-8"))["metrics"]
    gate = {k: [v1[k]["logloss"], report["variants"]["v1"]["metrics"][k]["logloss"]] for k in ("C", "I_top")}
    report["reproduction_gate"] = {k: {"run_v1": a, "v1b": b, "equal": abs(a - b) < 1e-12} for k, (a, b) in gate.items()}
    loss = {k: game_logloss(yd, v) for k, v in all_preds.items()}
    pairs = [("v1:C_noexp", "v1:I_noexp_top"), ("v1:C_noexp", "v1:I_noexp_avg10"),
             ("v1:I_top", "v1:I_noexp_top"), ("v1:C", "v1:C_noexp"),
             ("frozen:C", "v1:C"), ("frozen:I_top", "v1:I_top"), ("frozen:C_noexp", "frozen:I_noexp_top"),
             ("k50:C", "frozen:C"), ("k50:I_top", "frozen:I_top"), ("k50:C_noexp", "frozen:C_noexp")]
    report["paired_logloss"] = {f"{a} - {b}": paired_bootstrap(loss[a], loss[b]) for a, b in pairs}
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(json.dumps(report["reproduction_gate"]))
    for k, v in report["paired_logloss"].items():
        log(f"{k:36s} {v['diff']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")


if __name__ == "__main__":
    main()
