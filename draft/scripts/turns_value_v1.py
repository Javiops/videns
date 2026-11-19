"""MODEL-BASED value of seeing the lane opponent before picking (not outcome-validated).

For development drafts and each lane/team: choose the best of the top-K pool with the lane
opponent unknown ('average champion') vs with the actual opponent visible; both choices are
scored against the actual opponent with every other slot as actually drafted. The difference
is the counter-pick information value under the frozen v1 C_noexp model. It is an in-model
quantity: estimation noise in the tables inflates it (max over noisy candidates).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.data import load_games  # noqa: E402
from draft.features import match_pair  # noqa: E402

RUN = ROOT / "data/runs/v1"
PATCHES = ("16.17", "16.18", "16.19")
K = 30
N = 800


def main():
    with open(RUN / "models.pkl", "rb") as f:
        pk = pickle.load(f)
    T, C = pk["T"], pk["C_noexp"]
    G = load_games(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/split_v1.json")
    G = G.subset(np.where((G.role == "dev") & np.isin(G.patch, list(PATCHES)))[0])
    rng = np.random.default_rng(29)
    games = rng.choice(len(G), size=N, replace=False)
    top = np.argsort(-T.pick, 1)[:, :K]

    def p_blue(champ):
        z = np.zeros(champ.shape, np.float32)
        x, xs = match_pair(T, champ, z, z)
        return C.predict(x, xs)

    gains, by_role = [], {r: [] for r in range(5)}
    for g in games:
        final = G.champ[g]
        bans = set(G.bans[g].ravel().tolist()) - {0}
        for s in (0, 1):
            for r in range(5):
                others = set(final.ravel().tolist()) - {final[s, r]}
                own = [c for c in top[r] if c not in bans and c not in others]
                grid = np.repeat(final[None], 2 * len(own), 0)
                grid[:, s, r] = np.tile(own, 2)
                grid[:len(own), 1 - s, r] = 0          # opponent unknown
                pb = p_blue(grid)
                v = (pb if s == 0 else 1 - pb).reshape(2, len(own))
                blind, seen = v[0].argmax(), v[1].argmax()
                gains.append(v[1, seen] - v[1, blind])
                by_role[r].append(v[1, seen] - v[1, blind])
    gains = np.array(gains) * 100
    boot = lambda x: [float(v) for v in np.percentile(
        [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], [2.5, 97.5])]
    out = {"model": "v1 C_noexp", "decisions": int(len(gains)), "K": K,
           "counter_info_value_pp": {"mean": float(gains.mean()), "ci95": boot(gains),
                                     "share_choice_changes": float((gains > 0).mean())},
           "by_role_pp": {["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][r]: float(np.mean(v) * 100)
                          for r, v in by_role.items()}}
    (RUN / "turns_value_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
