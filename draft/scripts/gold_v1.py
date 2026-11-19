"""Protocol gold v1: lane gold difference at 12:00 from the finished draft (development only)."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.data import load_games, sha256_file  # noqa: E402
from draft.features import exp_features  # noqa: E402
from draft.gold import LaneGoldModel, build_gold_tables, gold_pair, lane_diff  # noqa: E402
from draft.models import paired_bootstrap  # noqa: E402
from draft.tables import R, build_tables  # noqa: E402

EXTRACT = ROOT / "data/extract/games_v1.jsonl.gz"
SPLIT = ROOT / "data/split_v1.json"
OUT = ROOT / "data/runs/gold_v1"
PATCHES = ("16.17", "16.18", "16.19")
FOLDS = 5
LANES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def exp_extra(G, L):
    ec, et = exp_features(G.exp_champ, G.exp_total)
    a = np.column_stack([ec[:, 0, L], et[:, 0, L], ec[:, 0].sum(1)])
    b = np.column_stack([ec[:, 1, L], et[:, 1, L], ec[:, 1].sum(1)])
    return np.hstack([a, b]).astype(np.float32), np.hstack([b, a]).astype(np.float32)


def scores(y, p):
    e = p - y
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_games(EXTRACT, SPLIT)
    scope = np.isin(G.patch, list(PATCHES)) & ~np.isnan(G.g12).any((1, 2))
    Gf = G.subset(np.where((G.role == "fit") & scope)[0])
    Gd = G.subset(np.where((G.role == "dev") & scope)[0])
    log(f"games with a 12:00 frame: fit={len(Gf)} dev={len(Gd)}")
    # same folds as v1 are defined over the v1 fit population; re-derive them there and map by id
    v1_fit = G.subset(np.where((G.role == "fit") & np.isin(G.patch, list(PATCHES)))[0])
    v1_fold = dict(zip(v1_fit.ids, np.random.default_rng(11).integers(0, FOLDS, size=len(v1_fit))))
    fold = np.array([v1_fold[i] for i in Gf.ids])

    yf, yd = lane_diff(Gf), lane_diff(Gd)
    xf = {(L, v): None for L in range(R) for v in ("champ", "exp")}
    for k in range(FOLDS):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        # tables from the v1 fold games (incl. games without a 12:00 frame for win-rate tables)
        tr_ids = set(Gf.ids[tr]) | {i for i, f in v1_fold.items() if f != k}
        Tk = build_tables(v1_fit.subset(np.where(np.isin(v1_fit.ids, list(tr_ids)))[0]))
        GTk = build_gold_tables(Gf.subset(tr))
        Gk = Gf.subset(te)
        for L in range(R):
            for v, extra in (("champ", None), ("exp", exp_extra(Gk, L))):
                x, xs = gold_pair(GTk, Tk, Gk.champ, L, extra)
                if xf[(L, v)] is None:
                    xf[(L, v)] = (np.zeros((len(Gf), x.shape[1]), np.float32), np.zeros((len(Gf), x.shape[1]), np.float32))
                xf[(L, v)][0][te], xf[(L, v)][1][te] = x, xs
        log(f"fold {k} done")

    with open(ROOT / "data/runs/v1/models.pkl", "rb") as f:
        T = pickle.load(f)["T"]                        # frozen v1 tables (same fit games)
    GT = build_gold_tables(Gf)
    models, preds, report = {}, {}, {"protocol": "docs/protocol_gold_v1.md",
                                     "protocol_sha256": sha256_file(ROOT / "docs/protocol_gold_v1.md"),
                                     "games": {"fit": len(Gf), "dev": len(Gd)}, "lanes": {}}
    side_mean = np.nanmean(yf, 0)
    for L in range(R):
        lane = {}
        for v in ("champ", "exp"):
            extra = exp_extra(Gd, L) if v == "exp" else None
            m = LaneGoldModel().fit(xf[(L, v)][0], xf[(L, v)][1], yf[:, L])
            xd, xsd = gold_pair(GT, T, Gd.champ, L, extra)
            preds[(L, v)] = m.predict(xd, xsd)
            assert np.allclose(m.predict(xsd, xd), -preds[(L, v)])
            models[(L, v)] = m
        y = yd[:, L]
        b2 = T.cgold[L, Gd.champ[:, 0, L]] - T.cgold[L, Gd.champ[:, 1, L]]
        cands = {"B0_zero": np.zeros_like(y), "B1_side": np.full_like(y, side_mean[L]), "B2_champ_gold": b2,
                 "model_champ": preds[(L, "champ")], "model_exp": preds[(L, "exp")]}
        lane["scores"] = {k: scores(y, p) for k, p in cands.items()}
        se = {k: (p - y) ** 2 / 1e6 for k, p in cands.items()}
        lane["paired_sq_error_k2"] = {
            "model_champ - B2_champ_gold": paired_bootstrap(se["model_champ"], se["B2_champ_gold"]),
            "model_champ - B0_zero": paired_bootstrap(se["model_champ"], se["B0_zero"]),
            "model_exp - model_champ": paired_bootstrap(se["model_exp"], se["model_champ"])}
        lane["dev_sd_target"] = float(y.std())
        report["lanes"][LANES[L]] = lane
        s = lane["scores"]
        log(f"{LANES[L]:8s} sd={y.std():6.0f} | B2 mae={s['B2_champ_gold']['mae']:.0f} r2={s['B2_champ_gold']['r2']:+.4f} | "
            f"champ mae={s['model_champ']['mae']:.0f} r2={s['model_champ']['r2']:+.4f} r={s['model_champ']['pearson']:.3f} | "
            f"exp r2={s['model_exp']['r2']:+.4f}")
    tot_y = yd.sum(1)
    tot = {v: sum(preds[(L, v)] for L in range(R)) for v in ("champ", "exp")}
    report["team_total"] = {"champ": scores(tot_y, tot["champ"]), "exp": scores(tot_y, tot["exp"]),
                            "B0_zero": scores(tot_y, np.zeros_like(tot_y)), "dev_sd_target": float(tot_y.std())}
    log(f"TOTAL sd={tot_y.std():.0f} champ r2={report['team_total']['champ']['r2']:+.4f} "
        f"exp r2={report['team_total']['exp']['r2']:+.4f}")
    report["gold_prior_k"] = GT.prior_k
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    rmse = {L: report["lanes"][LANES[L]]["scores"]["model_champ"]["rmse"] for L in range(R)}
    with open(OUT / "model.pkl", "wb") as f:
        pickle.dump({"GT": GT, "models": {L: models[(L, "champ")] for L in range(R)}, "dev_rmse": rmse}, f)
    log("gold report written")


if __name__ == "__main__":
    main()
