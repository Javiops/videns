"""Protocol v1, test T3: partial drafts. 'Average champion' imputation vs Monte Carlo
marginalisation over fit pick rates, scored on real development outcomes."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.data import load_games  # noqa: E402
from draft.features import exp_features, match_pair, perspective_features  # noqa: E402
from draft.models import game_logloss, metrics, paired_bootstrap  # noqa: E402

RUN = ROOT / "data/runs/v1"
PATCHES = ("16.17", "16.18", "16.19")
M = 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Sampler:
    """Draw champions for hidden slots from fit pick rates, excluding bans, visible picks and
    champions already drawn for the same simulated draft."""

    def __init__(self, pick: np.ndarray, seed: int = 5):
        self.prob = pick / pick.sum(1, keepdims=True)
        self.rng = np.random.default_rng(seed)

    def draw(self, champ, bans, hidden, m):
        n = len(champ)
        out = np.repeat(champ[None], m, 0).copy()          # (m, n, 2, 5)
        for s, r in hidden:
            out[:, :, s, r] = 0
        banned = np.broadcast_to(bans.reshape(1, n, -1), (m, n, 10))
        for s, r in hidden:
            d = self.rng.choice(self.prob.shape[1], size=(m, n), p=self.prob[r])
            for _ in range(30):
                taken = np.concatenate([banned, out.reshape(m, n, 10)], 2)
                bad = (d[:, :, None] == taken).any(2)
                if not bad.any():
                    break
                d[bad] = self.rng.choice(self.prob.shape[1], size=int(bad.sum()), p=self.prob[r])
            out[:, :, s, r] = d
        return out


def predict_C(model, T, champ, ec, et):
    x, xs = match_pair(T, champ, ec, et)
    return model.predict(x, xs)


def run_scenario(name, hidden_sets, G, T, C, ec, et, exp_mean, sampler):
    """hidden_sets: list of lists of (side, role). Returns per-game mean loss for avg and mc."""
    y = G.blue_win.astype(float)
    loss_avg, loss_mc, p_avg_all, p_mc_all = [], [], [], []
    for hidden in hidden_sets:
        c_avg = G.champ.copy()
        e1, e2 = ec.copy(), et.copy()
        for s, r in hidden:
            c_avg[:, s, r] = 0
            e1[:, s, r], e2[:, s, r] = exp_mean
        p_avg = predict_C(C, T, c_avg, e1, e2)
        draws = sampler.draw(G.champ, G.bans, hidden, M)
        p_mc = np.mean([predict_C(C, T, draws[j], e1, e2) for j in range(M)], 0)
        loss_avg.append(game_logloss(y, p_avg))
        loss_mc.append(game_logloss(y, p_mc))
        p_avg_all.append(p_avg)
        p_mc_all.append(p_mc)
    la, lm = np.mean(loss_avg, 0), np.mean(loss_mc, 0)
    pa, pm = np.concatenate(p_avg_all), np.concatenate(p_mc_all)
    yy = np.tile(y, len(hidden_sets))
    res = {"variants": len(hidden_sets), "avg": metrics(yy, pa), "mc": metrics(yy, pm),
           "mc_minus_avg_logloss": paired_bootstrap(lm, la),
           "mean_abs_prob_gap": float(np.mean(np.abs(pa - pm)))}
    log(f"{name}: avg ll={res['avg']['logloss']:.5f} mean={res['avg']['mean_pred']:.4f} | "
        f"mc ll={res['mc']['logloss']:.5f} mean={res['mc']['mean_pred']:.4f} | obs={res['avg']['observed']:.4f} "
        f"| d={res['mc_minus_avg_logloss']['diff']:+.5f} {res['mc_minus_avg_logloss']['ci95']}")
    return res


def main():
    with open(RUN / "models.pkl", "rb") as f:
        pk = pickle.load(f)
    T, C, I, exp_mean = pk["T"], pk["C"], pk["I"], pk["exp_mean"]
    G = load_games(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/split_v1.json")
    G = G.subset(np.where((G.role == "dev") & np.isin(G.patch, list(PATCHES)))[0])
    ec, et = exp_features(G.exp_champ, G.exp_total)
    sampler = Sampler(T.pick[:, :].copy())
    sampler.prob[:, 0] = 0
    sampler.prob /= sampler.prob.sum(1, keepdims=True)
    y = G.blue_win.astype(float)
    report = {"M": M, "games": len(G)}
    full = predict_C(C, T, G.champ, ec, et)
    report["full_draft_C"] = metrics(y, full)
    report["enemy_team_hidden"] = run_scenario(
        "enemy team hidden", [[(1, r) for r in range(5)], [(0, r) for r in range(5)]], G, T, C, ec, et, exp_mean,
        sampler)
    report["lane_opponent_hidden"] = run_scenario(
        "lane opponent hidden", [[(s, r)] for s in (0, 1) for r in range(5)], G, T, C, ec, et, exp_mean, sampler)
    everyone_but_blue_top = [(s, r) for s in (0, 1) for r in range(5) if (s, r) != (0, 0)]
    report["only_blue_top_known_C"] = run_scenario(
        "only blue top known (C)", [everyone_but_blue_top], G, T, C, ec, et, exp_mean, sampler)

    # iTero-style reading of the same probe: perspective (blue, TOP) of the perspective model
    c_avg, e1, e2 = G.champ.copy(), ec.copy(), et.copy()
    for s, r in everyone_but_blue_top:
        c_avg[:, s, r] = 0
        e1[:, s, r], e2[:, s, r] = exp_mean
    P = perspective_features(T, c_avg, e1, e2)
    pI_avg = I.predict(P[:, 0, 0])
    draws = sampler.draw(G.champ, G.bans, everyone_but_blue_top, M)
    pI_mc = np.mean([I.predict(perspective_features(T, draws[j], e1, e2)[:, 0, 0]) for j in range(M)], 0)
    report["only_blue_top_known_I"] = {"avg": metrics(y, pI_avg), "mc": metrics(y, pI_mc),
                                       "mc_minus_avg_logloss": paired_bootstrap(game_logloss(y, pI_mc),
                                                                                game_logloss(y, pI_avg))}
    log(f"I probe: avg mean={pI_avg.mean():.4f} mc mean={pI_mc.mean():.4f} obs={y.mean():.4f}")
    (RUN / "blind_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log("blind report written")


if __name__ == "__main__":
    main()
