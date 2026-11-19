"""Protocol profile v2: expected kills, kill participation, team dragons (development only)."""
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
from draft.data import NC, ROLES, load_games, sha256_file  # noqa: E402
from draft.models import paired_bootstrap  # noqa: E402
from draft.profile import role_means  # noqa: E402

EXTRACT = ROOT / "data/extract/games_v1.jsonl.gz"
STATS = ROOT / "data/extract/match_stats_v1.jsonl.gz"
SPLIT = ROOT / "data/split_v1.json"
OUT = ROOT / "data/runs/profile_v2"
PATCHES = ("16.17", "16.18", "16.19")
FOLDS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dragon_tables(champ, drag, mu):
    """own/enemy shrunk mean (team dragons - mu) per (role, champion)."""
    dev = (drag - mu)[:, :, None].repeat(5, 2)                   # (n, 2, 5) from each team's view
    own = role_means(dev, champ) - 0.0
    enemy = role_means(dev, champ[:, ::-1, :])
    # role_means shrinks toward the role mean of the deviations (~0); champion 0 gets that mean
    return own, enemy


def dragon_features(champ, own, enemy):
    ridx = np.arange(5)
    rows = []
    for s in (0, 1):
        a, b = champ[:, s], champ[:, 1 - s]
        rows.append(np.column_stack([own[ridx, a].sum(1), enemy[ridx, b].sum(1), np.full(len(a), 1.0 - 2 * s)]))
    return np.stack(rows, 1)                                     # (n, 2, 3)


def per_player(name, y_fit, y_dev, Gf, Gd):
    prof = role_means(y_fit, Gf.champ)
    out = {}
    for r in range(5):
        ok = ~np.isnan(y_dev[:, :, r])
        yy, cc = y_dev[:, :, r][ok], Gd.champ[:, :, r][ok]
        pm, pb = prof[r, cc], np.full_like(yy, prof[r, 0])
        se_m, se_b = (pm - yy) ** 2, (pb - yy) ** 2
        out[ROLES[r]] = {"n": int(ok.sum()), "rmse_model": float(np.sqrt(se_m.mean())), "rmse_role_mean": float(np.sqrt(se_b.mean())),
                         "r2": float(1 - se_m.sum() / ((yy - yy.mean()) ** 2).sum()),
                         "model_minus_role_mean_sq": paired_bootstrap(se_m, se_b)}
    log(f"{name}: " + " ".join(f"{L[:3]} r2={v['r2']:+.3f} rmse={v['rmse_model']:.3f}/{v['rmse_role_mean']:.3f}" for L, v in out.items()))
    return prof, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_games(EXTRACT, SPLIT)
    G = G.subset(np.where(np.isin(G.patch, list(PATCHES)))[0])
    pos = {mid: i for i, mid in enumerate(G.ids)}
    kills = np.full((len(G), 2, 5), np.nan)
    assists = np.full((len(G), 2, 5), np.nan)
    team_kills = np.full((len(G), 2), np.nan)
    drag = np.full((len(G), 2), np.nan)
    with gzip.open(STATS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["id"])
            if i is None:
                continue
            kda = np.array(r["kda"], dtype=float)
            kills[i], assists[i] = kda[..., 0], kda[..., 2]
            for s in (0, 1):
                team_kills[i, s] = r["teams"][s]["kills"]
                drag[i, s] = r["teams"][s]["dragons"]
    assert not np.isnan(kills).any() and not np.isnan(drag).any()
    # sanity: team kills recorded == sum of player kills
    report = {"protocol": "docs/protocol_profile_v2.md", "protocol_sha256": sha256_file(ROOT / "docs/protocol_profile_v2.md"),
              "stats_sha256": sha256_file(STATS),
              "team_kills_equal_sum_of_player_kills": float((team_kills == kills.sum(2)).mean())}
    team_sum = kills.sum(2, keepdims=True)                     # scoreboard team kills (standard KP)
    kp = np.where(team_sum > 0, (kills + assists) / np.maximum(team_sum, 1), np.nan)
    fit, dev = G.role == "fit", G.role == "dev"
    Gf, Gd = G.subset(np.where(fit)[0]), G.subset(np.where(dev)[0])
    kills_prof, report["kills"] = per_player("kills", kills[fit], kills[dev], Gf, Gd)
    kp_prof, report["kp"] = per_player("kill participation", kp[fit], kp[dev], Gf, Gd)

    # dragons: OOF tables on fit (v1 folds), linear combination, dev with full tables
    df, dd = drag[fit], drag[dev]
    mu = float(df.mean())
    fold = np.random.default_rng(11).integers(0, FOLDS, size=len(Gf))
    Xf = np.zeros((len(Gf), 2, 3))
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        own, enemy = dragon_tables(Gf.champ[tr], df[tr], mu)
        Xf[te] = dragon_features(Gf.champ[te], own, enemy)
    reg = LinearRegression().fit(Xf.reshape(-1, 3), (df - mu).reshape(-1))
    own, enemy = dragon_tables(Gf.champ, df, mu)
    pred = mu + reg.predict(dragon_features(Gd.champ, own, enemy).reshape(-1, 3)).reshape(-1, 2)
    side_mu = df.mean(0)
    base = np.broadcast_to(side_mu, dd.shape)
    se_m, se_b = ((pred - dd) ** 2).mean(1), ((base - dd) ** 2).mean(1)
    report["dragons"] = {"mu_fit": mu, "side_means_fit": side_mu.tolist(), "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_),
                         "rmse_model": float(np.sqrt(se_m.mean())), "rmse_side_mean": float(np.sqrt(se_b.mean())),
                         "r2": float(1 - ((pred - dd) ** 2).sum() / ((dd - dd.mean()) ** 2).sum()),
                         "model_minus_side_mean_sq": paired_bootstrap(se_m, se_b)}
    log(f"dragons: rmse {report['dragons']['rmse_model']:.3f} vs side mean {report['dragons']['rmse_side_mean']:.3f}, "
        f"r2={report['dragons']['r2']:+.3f}, coef={np.round(reg.coef_, 3).tolist()}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(OUT / "profile.pkl", "wb") as f:
        pickle.dump({"kills": kills_prof, "kp": kp_prof, "dragons": {"mu": mu, "own": own, "enemy": enemy,
                                                                     "coef": reg.coef_, "intercept": float(reg.intercept_)}}, f)
    log("profile v2 written")


if __name__ == "__main__":
    main()
