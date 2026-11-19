"""Protocol profile v3: kill participation and team dragons at 12 / 24 / 30 minutes."""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from draft.data import ROLES, load_games, sha256_file  # noqa: E402
from draft.models import paired_bootstrap  # noqa: E402
from draft.profile import role_means  # noqa: E402
from profile_v2 import dragon_features, dragon_tables  # noqa: E402

EXTRACT = ROOT / "data/extract/games_v1.jsonl.gz"
EVENTS = ROOT / "data/extract/event_frames_v1.jsonl.gz"
STATS = ROOT / "data/extract/match_stats_v1.jsonl.gz"
SPLIT = ROOT / "data/split_v1.json"
OUT = ROOT / "data/runs/profile_v3"
PATCHES = ("16.17", "16.18", "16.19")
FOLDS = 5
PHASES = (12, 24, 30)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_games(EXTRACT, SPLIT)
    G = G.subset(np.where(np.isin(G.patch, list(PATCHES)))[0])
    pos = {mid: i for i, mid in enumerate(G.ids)}
    keys = [str(m) for m in PHASES] + ["final"]
    K = {k: np.full((len(G), 2, 5), np.nan) for k in keys}
    A = {k: np.full((len(G), 2, 5), np.nan) for k in keys}
    D = {k: np.full((len(G), 2), np.nan) for k in keys}
    with gzip.open(EVENTS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["id"])
            if i is None:
                continue
            for k in keys:
                if k in r["at"]:
                    K[k][i], A[k][i], D[k][i] = r["at"][k]["k"], r["at"][k]["a"], r["at"][k]["d"]
    # gate: whole-game event totals == match payload totals
    mk, ma, md = np.full((len(G), 2, 5), np.nan), np.full((len(G), 2, 5), np.nan), np.full((len(G), 2), np.nan)
    with gzip.open(STATS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["id"])
            if i is not None:
                kda = np.array(r["kda"], dtype=float)
                mk[i], ma[i] = kda[..., 0], kda[..., 2]
                md[i] = [r["teams"][s]["dragons"] for s in (0, 1)]
    have = ~np.isnan(K["final"]).any((1, 2))
    agree = ((K["final"] == mk).all((1, 2)) & (A["final"] == ma).all((1, 2)) & (D["final"] == md).all(1))[have]
    report = {"protocol": "docs/protocol_profile_v3.md", "protocol_sha256": sha256_file(ROOT / "docs/protocol_profile_v3.md"),
              "events_sha256": sha256_file(EVENTS), "games_with_events": int(have.sum()),
              "event_totals_match_payload_share": float(agree.mean()), "phases": {}}
    log(f"gate: event totals equal payload in {agree.mean():.4%} of {have.sum()} games")
    assert agree.mean() >= 0.995

    fit_all = (G.role == "fit")
    fold_all = np.full(len(G), -1)
    fold_all[fit_all] = np.random.default_rng(11).integers(0, FOLDS, size=int(fit_all.sum()))
    saved = {"kp": {}, "dragons": {}}
    for m in PHASES:
        k = str(m)
        reach = ~np.isnan(D[k]).any(1)
        fit, dev = fit_all & reach, (G.role == "dev") & reach
        team = K[k].sum(2, keepdims=True)
        kp = np.where(team > 0, (K[k] + A[k]) / np.maximum(team, 1), np.nan)
        prof = role_means(kp[fit], G.champ[fit])
        lanes = {}
        for r in range(5):
            y = kp[dev][:, :, r]
            ok = ~np.isnan(y)
            yy, cc = y[ok], G.champ[dev][:, :, r][ok]
            se_m, se_b = (prof[r, cc] - yy) ** 2, (prof[r, 0] - yy) ** 2
            lanes[ROLES[r]] = {"n": int(ok.sum()), "rmse_model": float(np.sqrt(se_m.mean())),
                               "rmse_role_mean": float(np.sqrt(se_b.mean())),
                               "r2": float(1 - se_m.sum() / ((yy - yy.mean()) ** 2).sum()),
                               "model_minus_role_mean_sq": paired_bootstrap(se_m, se_b)}
        saved["kp"][m] = prof
        # dragons by m
        champ_f, df, dd = G.champ[fit], D[k][fit], D[k][dev]
        mu = float(df.mean())
        fold = fold_all[fit]
        Xf = np.zeros((len(df), 2, 3))
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            own, enemy = dragon_tables(champ_f[tr], df[tr], mu)
            Xf[te] = dragon_features(champ_f[te], own, enemy)
        reg = LinearRegression().fit(Xf.reshape(-1, 3), (df - mu).reshape(-1))
        own, enemy = dragon_tables(champ_f, df, mu)
        pred = mu + reg.predict(dragon_features(G.champ[dev], own, enemy).reshape(-1, 3)).reshape(-1, 2)
        base = np.broadcast_to(df.mean(0), dd.shape)
        se_m, se_b = ((pred - dd) ** 2).mean(1), ((base - dd) ** 2).mean(1)
        drag = {"mu": mu, "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_), "games_dev": int(dev.sum()),
                "rmse_model": float(np.sqrt(se_m.mean())), "rmse_side_mean": float(np.sqrt(se_b.mean())),
                "r2": float(1 - ((pred - dd) ** 2).sum() / ((dd - dd.mean()) ** 2).sum()),
                "model_minus_side_mean_sq": paired_bootstrap(se_m, se_b)}
        saved["dragons"][m] = {"mu": mu, "own": own, "enemy": enemy, "coef": reg.coef_, "intercept": float(reg.intercept_)}
        report["phases"][k] = {"kp": lanes, "dragons": drag}
        log(f"{m}' KP " + " ".join(f"{L[:3]} r2={v['r2']:+.3f}" for L, v in lanes.items()) +
            f" | dragons mean {dd.mean():.2f} rmse {drag['rmse_model']:.3f} vs {drag['rmse_side_mean']:.3f} r2={drag['r2']:+.3f} "
            f"ci={[round(x, 4) for x in drag['model_minus_side_mean_sq']['ci95']]}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(OUT / "profile.pkl", "wb") as f:
        pickle.dump(saved, f)
    log("profile v3 written")


if __name__ == "__main__":
    main()
