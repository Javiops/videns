"""Protocol v1, tests T1/T2/T4: fit on 'fit', evaluate on 'dev'. Confirmation stays sealed."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.data import load_games, make_split, sha256_file  # noqa: E402
from draft.features import (MATCH_NAMES, PERSP_NAMES, exp_features, match_pair,  # noqa: E402
                            perspective_features)
from draft.models import Coherent, Ensemble, game_logloss, metrics, paired_bootstrap  # noqa: E402
from draft.tables import build_tables  # noqa: E402

EXTRACT = ROOT / "data/extract/games_v1.jsonl.gz"
SPLIT = ROOT / "data/split_v1.json"
OUT = ROOT / "data/runs/v1"
FOLDS = 5
PATCHES = ("16.17", "16.18", "16.19")  # declared population


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ctx_cols(G, flip=False):
    auto = (G.asg == 2).sum(2).astype(np.float32)
    sec = (G.asg == 1).sum(2).astype(np.float32)
    a, b = (1, 0) if flip else (0, 1)
    return np.column_stack([auto[:, a], auto[:, b], sec[:, a], sec[:, b]])


def unknown_cols(G, flip=False):
    unk = (G.exp_total == 0).sum(2).astype(np.float32)
    a, b = (1, 0) if flip else (0, 1)
    return np.column_stack([unk[:, a], unk[:, b]])


def feature_sets(T, G, exp_mean=None):
    ec, et = exp_features(G.exp_champ, G.exp_total)
    x, xs = match_pair(T, G.champ, ec, et)
    ec0, et0 = np.zeros_like(ec), np.zeros_like(et)
    x0, xs0 = match_pair(T, G.champ, ec0, et0)          # experience removed (constant zero)
    out = {"persp": perspective_features(T, G.champ, ec, et), "C": (x, xs), "C_noexp": (x0, xs0)}
    if exp_mean is not None:  # unknown history -> population mean + count of unknown players
        unk = G.exp_total == 0
        ecu, etu = np.where(unk, exp_mean[0], ec), np.where(unk, exp_mean[1], et)
        xu, xsu = match_pair(T, G.champ, ecu, etu)
        out["C_unknown"] = (np.hstack([xu, unknown_cols(G)]), np.hstack([xsu, unknown_cols(G, True)]))
    out["C_ctx"] = (np.hstack([x, ctx_cols(G)]), np.hstack([xs, ctx_cols(G, True)]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SPLIT.exists():
        log(f"split: {make_split(EXTRACT, SPLIT)}")
    G = load_games(EXTRACT, SPLIT)
    in_scope = np.isin(G.patch, list(PATCHES))
    fit = np.where((G.role == "fit") & in_scope)[0]
    dev = np.where((G.role == "dev") & in_scope)[0]
    out_of_scope = {r: int(((G.role == r) & ~in_scope).sum()) for r in ("fit", "dev")}
    log(f"loaded fit={len(fit)} dev={len(dev)} players={G.player.max() + 1}")
    Gf, Gd = G.subset(fit), G.subset(dev)
    known = Gf.exp_total > 0
    ecf, etf = exp_features(Gf.exp_champ, Gf.exp_total)
    exp_mean = (float(ecf[known].mean()), float(etf[known].mean()))

    rng = np.random.default_rng(11)
    fold = rng.integers(0, FOLDS, size=len(fit))
    fit_sets = None
    for k in range(FOLDS):
        Tk = build_tables(Gf.subset(np.where(fold != k)[0]))
        fs = feature_sets(Tk, Gf.subset(np.where(fold == k)[0]), exp_mean)
        if fit_sets is None:
            fit_sets = {name: (np.zeros((len(fit),) + v.shape[1:], np.float32) if not isinstance(v, tuple) else
                               tuple(np.zeros((len(fit), a.shape[1]), np.float32) for a in v))
                        for name, v in fs.items()}
        for name, v in fs.items():
            if isinstance(v, tuple):
                for dst, src in zip(fit_sets[name], v):
                    dst[fold == k] = src
            else:
                fit_sets[name][fold == k] = v
        log(f"fold {k}: tables {Tk.prior_k}")
    T = build_tables(Gf)
    dev_sets = feature_sets(T, Gd, exp_mean)
    log("features built")

    yf = Gf.blue_win.astype(int)
    yd = Gd.blue_win.astype(float)
    preds, extra = {}, {}

    preds["R0_side"] = np.full(len(dev), T.p_side)
    # R1: champion-role deltas only, coherent logistic on (base_a - base_b, side)
    xf, xsf = fit_sets["C"]
    xd, xsd = dev_sets["C"]
    cols = [MATCH_NAMES.index("side"), MATCH_NAMES.index("base_a"), MATCH_NAMES.index("base_b")]
    r1 = lambda x: np.column_stack([x[:, cols[0]], x[:, cols[1]] - x[:, cols[2]]])
    m1 = LogisticRegression(C=1.0).fit(np.vstack([r1(xf), r1(xsf)]), np.concatenate([yf, 1 - yf]))
    preds["R1_champ_wr"] = 0.5 * (m1.predict_proba(r1(xd))[:, 1] + 1 - m1.predict_proba(r1(xsd))[:, 1])

    # I: iTero-style perspective model
    P = fit_sets["persp"]
    yp = np.stack([yf, 1 - yf], 1)[:, :, None].repeat(5, 2)
    mI = Ensemble().fit(P.reshape(-1, P.shape[-1]), yp.ravel())
    Pd = dev_sets["persp"]
    pI = mI.predict(Pd.reshape(-1, Pd.shape[-1])).reshape(len(dev), 2, 5)
    preds["I_top"] = pI[:, 0, 0]
    preds["I_avg10"] = 0.5 * (pI[:, 0, :].mean(1) + 1 - pI[:, 1, :].mean(1))
    log("I fitted")

    sums = pI[:, 0, :] + pI[:, 1, :] - 1.0                      # per game, per lane
    spread = pI[:, 0, :].max(1) - pI[:, 0, :].min(1)
    extra["T2_I_coherence"] = {
        "sum_minus_1_mean": float(sums.mean()), "sum_minus_1_sd": float(sums.std()),
        "abs_sum_minus_1_mean": float(np.abs(sums).mean()),
        "abs_sum_minus_1_p50_p90_p99": [float(v) for v in np.percentile(np.abs(sums), [50, 90, 99])],
        "share_abs_gt_2pp": float((np.abs(sums) > 0.02).mean()),
        "blue_spread_across_lanes_mean": float(spread.mean()),
        "blue_spread_p50_p90_p99": [float(v) for v in np.percentile(spread, [50, 90, 99])],
    }

    # C and ablations
    models = {}
    for name in ("C", "C_noexp", "C_unknown", "C_ctx"):
        (a, b), (c, d) = fit_sets[name], dev_sets[name]
        models[name] = Coherent().fit(a, b, yf)
        preds[name] = models[name].predict(c, d)
        if name == "C":
            lr, gb = models[name].predict_parts(c, d)
            preds["C_lr_only"], preds["C_gbm_only"] = lr, gb
            # exact coherence check: swapping the teams gives 1 - p
            assert np.allclose(models[name].predict(d, c), 1 - preds[name], atol=1e-9)
        log(f"{name} fitted")

    report = {"protocol": "docs/protocol_v1.md", "protocol_sha256": sha256_file(ROOT / "docs/protocol_v1.md"),
              "split_sha256": sha256_file(SPLIT), "extract_sha256": sha256_file(EXTRACT),
              "games": {"fit": int(len(fit)), "dev": int(len(dev))}, "excluded_out_of_patch_scope": out_of_scope,
              "dev_patches": {p: int((Gd.patch == p).sum()) for p in np.unique(Gd.patch)},
              "fit_patches": {p: int((Gf.patch == p).sum()) for p in np.unique(Gf.patch)},
              "prior_k_full": T.prior_k, "exp_mean_known": exp_mean,
              "share_unknown_history_dev": float((Gd.exp_total == 0).mean()),
              "metrics": {k: metrics(yd, v) for k, v in preds.items()}}
    loss = {k: game_logloss(yd, v) for k, v in preds.items()}
    pairs = [("I_top", "R0_side"), ("I_top", "R1_champ_wr"), ("C", "R0_side"), ("C", "R1_champ_wr"),
             ("R1_champ_wr", "R0_side"), ("C", "I_top"), ("C", "I_avg10"), ("I_avg10", "I_top"),
             ("C", "C_noexp"), ("C_unknown", "C"), ("C_ctx", "C"), ("C_lr_only", "C_gbm_only")]
    report["paired_logloss"] = {f"{a} - {b}": paired_bootstrap(loss[a], loss[b]) for a, b in pairs}
    report.update(extra)
    gb = models["C"].m.gb
    report["C_gbm_gain_importance"] = dict(sorted(
        zip(MATCH_NAMES, gb.booster_.feature_importance("gain").round(1).tolist()), key=lambda kv: -kv[1]))
    lrc = models["C"].m.lr[-1].coef_[0]
    report["C_lr_std_coef"] = dict(zip(MATCH_NAMES, np.round(lrc, 4).tolist()))
    report["I_lr_std_coef"] = dict(zip(PERSP_NAMES, np.round(mI.lr[-1].coef_[0], 4).tolist()))
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "dev_predictions.npz", ids=Gd.ids, y=yd, pI=pI, **preds)
    with open(OUT / "models.pkl", "wb") as f:
        pickle.dump({"T": T, "I": mI, "C": models["C"], "C_noexp": models["C_noexp"],
                     "exp_mean": exp_mean}, f)
    log("report written")
    for k, v in report["metrics"].items():
        log(f"{k:14s} ll={v['logloss']:.5f} acc={v['accuracy']:.4f} auc={v['auc']:.4f} "
            f"slope={v['cal_slope']:.3f} mean={v['mean_pred']:.4f} obs={v['observed']:.4f}")
    for k, v in report["paired_logloss"].items():
        log(f"{k:26s} {v['diff']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")
    log(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main()
