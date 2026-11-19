"""Protocol gold v2: lane gold difference at 12 / 24 / 36 minutes (development only)."""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.data import load_games, sha256_file  # noqa: E402
from draft.frames import MINUTES  # noqa: E402
from draft.gold import LaneGoldModel, build_gold_tables, gold_pair  # noqa: E402
from draft.models import paired_bootstrap  # noqa: E402
from draft.tables import R, build_tables  # noqa: E402

EXTRACT = ROOT / "data/extract/games_v1.jsonl.gz"
FRAMES = ROOT / "data/extract/gold_frames_v1.jsonl.gz"
SPLIT = ROOT / "data/split_v1.json"
OUT = ROOT / "data/runs/gold_v2"
PATCHES = ("16.17", "16.18", "16.19")
FOLDS = 5
FIT_MINUTES = (12, 24, 30, 36)
LANES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scores(y, p):
    e = p - y
    return {"rmse": float(np.sqrt((e ** 2).mean())), "rmse_zero": float(np.sqrt((y ** 2).mean())),
            "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "pearson": float(np.corrcoef(y, p)[0, 1]), "sd": float(y.std())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_games(EXTRACT, SPLIT)
    G = G.subset(np.where(np.isin(G.patch, list(PATCHES)))[0])
    pos = {mid: i for i, mid in enumerate(G.ids)}
    gm = np.full((len(G), 2, 5, len(MINUTES)), np.nan)
    with gzip.open(FRAMES, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["id"])
            if i is not None:
                gm[i] = np.array(r["g"], dtype=float)
    # timelines downloaded after the v1 extraction are left out, so the population matches v1 exactly
    added_later = int((~G.tl & ~np.isnan(gm[..., MINUTES.index(12)]).any((1, 2))).sum())
    gm[~G.tl] = np.nan
    log(f"games whose timeline appeared after the v1 extraction (excluded): {added_later}")
    # gate 1: the sidecar's 12:00 gold equals the frozen extract's g12
    k12 = MINUTES.index(12)
    same_nan = np.array_equal(np.isnan(gm[..., k12]), np.isnan(G.g12))
    ok = ~np.isnan(G.g12)
    assert same_nan and np.array_equal(gm[..., k12][ok], G.g12[ok].astype(float)), "sidecar 12:00 != extract g12"
    log("gate 1 passed: sidecar 12:00 gold identical to the v1 extract")

    with open(ROOT / "data/runs/v1/models.pkl", "rb") as f:
        T = pickle.load(f)["T"]
    v1_fit = G.subset(np.where(G.role == "fit")[0])
    fold_of = dict(zip(v1_fit.ids, np.random.default_rng(11).integers(0, FOLDS, size=len(v1_fit))))
    T_fold = [build_tables(v1_fit.subset(np.where(np.array([fold_of[i] for i in v1_fit.ids]) != k)[0]))
              for k in range(FOLDS)]
    n_dev_scope = int((G.role == "dev").sum())
    report = {"protocol": "docs/protocol_gold_v2.md", "protocol_sha256": sha256_file(ROOT / "docs/protocol_gold_v2.md"),
              "frames_sha256": sha256_file(FRAMES), "excluded_timelines_added_after_v1": added_later, "minutes": {}}
    saved = {}
    for m in FIT_MINUTES:
        k = MINUTES.index(m)
        d = gm[:, 0, :, k] - gm[:, 1, :, k]
        reach = ~np.isnan(d).any(1)
        Gf, Gd = G.subset(np.where((G.role == "fit") & reach)[0]), G.subset(np.where((G.role == "dev") & reach)[0])
        df, dd = d[(G.role == "fit") & reach], d[(G.role == "dev") & reach]
        fold = np.array([fold_of[i] for i in Gf.ids])
        xf = {L: (np.zeros((len(Gf), 7), np.float32), np.zeros((len(Gf), 7), np.float32)) for L in range(R)}
        for f in range(FOLDS):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            GTk = build_gold_tables(Gf.subset(tr), df[tr])
            for L in range(R):
                x, xs = gold_pair(GTk, T_fold[f], Gf.champ[te], L)
                xf[L][0][te], xf[L][1][te] = x, xs
        GT = build_gold_tables(Gf, df)
        models, preds = {}, np.zeros_like(dd)
        lanes = {}
        for L in range(R):
            mdl = LaneGoldModel().fit(xf[L][0], xf[L][1], df[:, L])
            preds[:, L] = mdl.predict(*gold_pair(GT, T, Gd.champ, L))
            models[L] = mdl
            sq = lambda p: (p - dd[:, L]) ** 2 / 1e6
            lanes[LANES[L]] = {**scores(dd[:, L], preds[:, L]),
                               "model_minus_zero_sq_k2": paired_bootstrap(sq(preds[:, L]), sq(np.zeros(len(dd))))}
        team = scores(dd.sum(1), preds.sum(1))
        report["minutes"][str(m)] = {"games_fit": len(Gf), "games_dev": len(Gd),
                                     "dev_share_reaching_minute": len(Gd) / n_dev_scope, "lanes": lanes, "team": team}
        saved[m] = {"GT": GT, "models": models, "dev_rmse": {L: lanes[LANES[L]]["rmse"] for L in range(R)},
                    "team_rmse": team["rmse"], "dev_share": len(Gd) / n_dev_scope}
        log(f"{m:2d}' reach {len(Gd) / n_dev_scope:6.1%} dev={len(Gd)} | " +
            " ".join(f"{LANES[L][:3]} r2={lanes[LANES[L]]['r2']:+.3f} rmse={lanes[LANES[L]]['rmse']:.0f}" for L in range(R)) +
            f" | TEAM sd={team['sd']:.0f} rmse={team['rmse']:.0f} r2={team['r2']:+.3f}")
        if m == 12:  # gate 2: identical to the frozen gold v1 predictions
            with open(ROOT / "data/runs/gold_v1/model.pkl", "rb") as f:
                g1 = pickle.load(f)
            ref = np.column_stack([g1["models"][L].predict(*gold_pair(g1["GT"], T, Gd.champ, L)) for L in range(R)])
            report["gate_reproduces_gold_v1"] = bool(np.allclose(ref, preds, atol=1e-6))
            assert report["gate_reproduces_gold_v1"], float(np.abs(ref - preds).max())
            log("gate 2 passed: 12:00 model reproduces gold v1 exactly")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(OUT / "model.pkl", "wb") as f:
        pickle.dump(saved, f)
    log("gold v2 written")


if __name__ == "__main__":
    main()
