"""Protocol profile v1: expected CS at 12/24/30 and team damage composition (development only)."""
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
from draft.data import ROLES, load_games, sha256_file  # noqa: E402
from draft.frames import MINUTES  # noqa: E402
from draft.models import paired_bootstrap  # noqa: E402
from draft.profile import actual_team_shares, build_cs_profile, build_damage_profile  # noqa: E402

EXTRACT = ROOT / "data/extract/games_v1.jsonl.gz"
CS = ROOT / "data/extract/cs_frames_v1.jsonl.gz"
SPLIT = ROOT / "data/split_v1.json"
OUT = ROOT / "data/runs/profile_v1"
PATCHES = ("16.17", "16.18", "16.19")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_games(EXTRACT, SPLIT)
    G = G.subset(np.where(np.isin(G.patch, list(PATCHES)))[0])
    Gf, Gd = G.subset(np.where(G.role == "fit")[0]), G.subset(np.where(G.role == "dev")[0])
    report = {"protocol": "docs/protocol_profile_v1.md", "protocol_sha256": sha256_file(ROOT / "docs/protocol_profile_v1.md")}

    # damage composition
    DP = build_damage_profile(Gf)
    pred, act = DP.team(Gd.champ), actual_team_shares(Gd)
    base = DP.team(np.zeros_like(Gd.champ))
    e_m, e_b = np.abs(pred[..., 0] - act[..., 0]).mean(1), np.abs(base[..., 0] - act[..., 0]).mean(1)
    report["damage"] = {"games_dev": len(Gd), "physical_share_mae_model": float(e_m.mean()),
                        "physical_share_mae_role_mean": float(e_b.mean()),
                        "pearson_physical_share": float(np.corrcoef(pred[..., 0].ravel(), act[..., 0].ravel())[0, 1]),
                        "model_minus_role_mean_abs_error": paired_bootstrap(e_m, e_b)}
    log(f"damage: phys-share MAE model {e_m.mean():.4f} vs role mean {e_b.mean():.4f}, r={report['damage']['pearson_physical_share']:.3f}")

    # expected CS
    pos = {mid: i for i, mid in enumerate(G.ids)}
    csm = np.full((len(G), 2, 5, len(MINUTES)), np.nan)
    with gzip.open(CS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["id"])
            if i is not None:
                csm[i] = np.array(r["cs"], dtype=float)
    # gate: the sidecar's 12:00 CS equals the cs12 written in the frozen v1 extract
    cs12 = np.full((len(G), 2, 5), np.nan)
    with gzip.open(EXTRACT, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["id"])
            if i is not None:
                for p in r["p"]:
                    if p["cs12"] is not None:
                        cs12[i, p["side"], ROLES.index(p["role"])] = p["cs12"]
    side12 = csm[..., MINUTES.index(12)]
    assert np.array_equal(np.isnan(side12), np.isnan(cs12)) and np.array_equal(side12[~np.isnan(cs12)], cs12[~np.isnan(cs12)])
    log("gate passed: sidecar 12:00 CS identical to the v1 extract")
    fit_m, dev_m = G.role == "fit", G.role == "dev"
    profiles, report["cs"] = {}, {}
    for m in (12, 24, 30):
        k = MINUTES.index(m)
        prof = build_cs_profile(Gf, csm[fit_m][..., k])
        profiles[m] = prof
        y = csm[dev_m][..., k]
        champ = Gd.champ
        lanes = {}
        for r in range(5):
            ok = ~np.isnan(y[:, :, r])
            yy, cc = y[:, :, r][ok], champ[:, :, r][ok]
            pm, pb = prof[r, cc], np.full_like(yy, prof[r, 0])
            se_m, se_b = (pm - yy) ** 2, (pb - yy) ** 2
            lanes[ROLES[r]] = {"n_players": int(ok.sum()), "rmse_model": float(np.sqrt(se_m.mean())),
                               "rmse_role_mean": float(np.sqrt(se_b.mean())),
                               "r2": float(1 - se_m.sum() / ((yy - yy.mean()) ** 2).sum()),
                               "model_minus_role_mean_sq": paired_bootstrap(se_m / 100, se_b / 100)}
        report["cs"][str(m)] = lanes
        log(f"cs {m}': " + " ".join(f"{L[:3]} r2={v['r2']:+.3f} rmse={v['rmse_model']:.1f}/{v['rmse_role_mean']:.1f}" for L, v in lanes.items()))
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(OUT / "profile.pkl", "wb") as f:
        pickle.dump({"damage": DP, "cs": profiles}, f)
    log("profile v1 written")


if __name__ == "__main__":
    main()
