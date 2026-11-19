"""Turn-order analysis (MODEL-BASED, not outcome-validated; see protocol v1 'Turn order').

Pick order is not stored in match-v5. We replay development drafts through the solo-queue
sequence B1 R1 R2 B2 B3 R3 R4 B4 B5 R5 with a random role order per team, and use the frozen
v1 coherent model without experience (C_noexp, champion information only) to measure:
  1. counter exposure of each pick whose lane opponent picks later;
  2. value of picking after the lane opponent (best own reply vs pick-rate mean);
  3. recommendation rules at every pick with a later lane opponent:
     greedy-vs-average (iTero), expected over the opponent's pick-rate reply, and maximin.
Unknown slots are the 'average champion' except where stated.
"""
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
from draft.features import match_pair  # noqa: E402

RUN = ROOT / "data/runs/v1"
PATCHES = ("16.17", "16.18", "16.19")
SEQ = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2), (1, 3), (0, 3), (0, 4), (1, 4)]  # (team, k-th pick)
K = 30
N_DRAFTS = 500


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    with open(RUN / "models.pkl", "rb") as f:
        pk = pickle.load(f)
    T, C = pk["T"], pk["C_noexp"]
    G = load_games(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/split_v1.json")
    G = G.subset(np.where((G.role == "dev") & np.isin(G.patch, list(PATCHES)))[0])
    rng = np.random.default_rng(23)
    pick_rate = T.pick / T.pick.sum(1, keepdims=True)
    top = np.argsort(-T.pick, 1)[:, :K]                               # (5, K) candidate pools

    def p_blue(champ):
        z = np.zeros(champ.shape, np.float32)
        x, xs = match_pair(T, champ, z, z)
        return C.predict(x, xs)

    rows = []
    games = rng.choice(len(G), size=min(N_DRAFTS, len(G)), replace=False)
    for gi, g in enumerate(games):
        order = [rng.permutation(5), rng.permutation(5)]
        slot_at = [(s, int(order[s][k])) for s, k in SEQ]               # position -> (team, role)
        pos_of = {sr: t for t, sr in enumerate(slot_at)}
        final = G.champ[g]
        bans = set(G.bans[g].ravel().tolist()) - {0}
        for t, (s, r) in enumerate(slot_at):
            known = np.zeros((2, 5), dtype=np.int32)
            for u in range(t):
                ss, rr = slot_at[u]
                known[ss, rr] = final[ss, rr]
            taken = bans | set(known.ravel().tolist())
            sign = 1.0 if s == 0 else -1.0
            val = lambda pb: pb if s == 0 else 1.0 - pb                  # P(picker's team wins)
            own = [c for c in top[r] if c not in taken]
            opp_later = pos_of[(1 - s, r)] > t
            if opp_later:
                resp = [c for c in top[r] if c not in taken]
                # candidate x response grid, response slot last column = average champion
                grid = np.repeat(known[None], len(own) * (len(resp) + 1), 0)
                ci = np.repeat(own, len(resp) + 1)
                di = np.tile(resp + [0], len(own))
                grid[:, s, r] = ci
                grid[:, 1 - s, r] = di
                same = ci == di
                v = val(p_blue(grid)).reshape(len(own), len(resp) + 1)
                v_avg = v[:, -1]
                V = np.where(same.reshape(len(own), -1)[:, :-1], np.nan, v[:, :-1])
                w = pick_rate[r, resp]
                W = np.where(np.isnan(V), 0, w[None, :])
                v_exp = np.nansum(V * W, 1) / W.sum(1)
                v_min = np.nanmin(V, 1)
                choice = {"greedy_avg": int(np.argmax(v_avg)), "expected": int(np.argmax(v_exp)),
                          "maximin": int(np.argmax(v_min))}
                actual = final[s, r]
                a_idx = own.index(actual) if actual in own else None
                rows.append({"pos": t + 1, "team": s, "role": r, "opp_later": True,
                             "rules": {k: {"vs_avg": float(v_avg[i]), "vs_expected": float(v_exp[i]),
                                           "vs_best_reply": float(v_min[i])} for k, i in choice.items()},
                             "same_choice_greedy_expected": choice["greedy_avg"] == choice["expected"],
                             "same_choice_greedy_maximin": choice["greedy_avg"] == choice["maximin"],
                             "actual_exposure": None if a_idx is None else
                             {"vs_avg": float(v_avg[a_idx]), "vs_expected": float(v_exp[a_idx]),
                              "vs_best_reply": float(v_min[a_idx])},
                             "pool_mean_vs_avg": float(np.average(v_avg, weights=pick_rate[r, own]))})
            else:
                # lane opponent already known: value of the best own reply vs pick-rate mean
                grid = np.repeat(known[None], len(own), 0)
                grid[:, s, r] = own
                v = val(p_blue(grid))
                rows.append({"pos": t + 1, "team": s, "role": r, "opp_later": False,
                             "best_reply": float(v.max()),
                             "pool_mean": float(np.average(v, weights=pick_rate[r, own])),
                             "actual": float(v[own.index(final[s, r])]) if final[s, r] in own else None})
        if gi % 50 == 0:
            log(f"{gi}/{len(games)} drafts")

    summary = {"model": "v1 C_noexp (champion information only)", "drafts": int(len(games)), "K": K,
               "by_position": {}}
    for pos in range(1, 11):
        rs = [r for r in rows if r["pos"] == pos]
        exp_rows = [r for r in rs if r["opp_later"]]
        rep_rows = [r for r in rs if not r["opp_later"]]
        d = {"n": len(rs), "share_lane_opponent_later": len(exp_rows) / max(len(rs), 1)}
        if exp_rows:
            act = [r["actual_exposure"] for r in exp_rows if r["actual_exposure"]]
            d["actual_pick_exposure_pp"] = {
                "avg_minus_best_reply": 100 * float(np.mean([a["vs_avg"] - a["vs_best_reply"] for a in act])),
                "avg_minus_expected": 100 * float(np.mean([a["vs_avg"] - a["vs_expected"] for a in act]))}
            for rule in ("greedy_avg", "expected", "maximin"):
                d[f"rule_{rule}_pp"] = {k: 100 * float(np.mean([r["rules"][rule][k] for r in exp_rows]))
                                        for k in ("vs_avg", "vs_expected", "vs_best_reply")}
            d["greedy_equals_expected"] = float(np.mean([r["same_choice_greedy_expected"] for r in exp_rows]))
            d["greedy_equals_maximin"] = float(np.mean([r["same_choice_greedy_maximin"] for r in exp_rows]))
        if rep_rows:
            d["counter_reply_gain_pp"] = 100 * float(np.mean([r["best_reply"] - r["pool_mean"] for r in rep_rows]))
        summary["by_position"][pos] = d
    exp_all = [r for r in rows if r["opp_later"]]
    summary["all_exposed_picks"] = {rule: {k: 100 * float(np.mean([r["rules"][rule][k] for r in exp_all]))
                                           for k in ("vs_avg", "vs_expected", "vs_best_reply")}
                                    for rule in ("greedy_avg", "expected", "maximin")}
    summary["all_exposed_picks"]["greedy_equals_expected"] = float(np.mean(
        [r["same_choice_greedy_expected"] for r in exp_all]))
    summary["all_exposed_picks"]["greedy_equals_maximin"] = float(np.mean(
        [r["same_choice_greedy_maximin"] for r in exp_all]))
    (RUN / "turns_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(json.dumps(summary["all_exposed_picks"], indent=1))
    for pos, d in summary["by_position"].items():
        log(f"pos {pos}: {json.dumps(d)}")


if __name__ == "__main__":
    main()
