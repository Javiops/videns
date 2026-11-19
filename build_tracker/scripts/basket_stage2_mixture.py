"""Declared mixture M2 = 0.5 * P(econ_ft_new, grouped) + 0.5 * P(wardfree_new) (segment D).

Protocol: docs/experiment_stage2_econ_wardfree_v1.md. The old package's rule
on the new generation: the econ-tuned arm's corrected full-catalogue scores
are ward-marginalised into the 9,333 groups and softmaxed; the ward-free
partner scores the same groups directly; both live in the SAME group index
(no permutation needed) and are averaged with the declared 50/50 weights —
never tuned. Integrity gate: both parents must reproduce their own persisted
segment rows with 0 mismatches. Scored on the cohort against R1 (new Bf),
paired by game; M2 is adopted only if it beats R1 with the CI excluding 0.
"""
import argparse
import json
import sys
import time
from collections import Counter

import torch

from stage2_newgen import (OUT_DIR, EXIT_GATE, setup, range_batch, group_logits, decode_batch, reference_rows,
                           compare, beats, write_report, now_utc, v1, p2, sha256_file)
from basket_newbf_mixture import load_rows, mismatches
from basket_vocabulary_probe import basket_key
from basket_bf_newgen import basket_tensors
from basket_econ_finetune import TunedNet
from basket_residual_econ import CorrectionNet

WEIGHT_A, WEIGHT_B = 0.5, 0.5          # declared in the protocol, never tuned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    S = setup(args.smoke, load_fit=False, load_model=False)
    stamp, device = S.stamp, S.device
    ep_ft, ep_wf = (1, 1) if args.smoke else (8, 16)
    ck_a = OUT_DIR / f"econ_ft_new_ep{ep_ft}_{stamp}.pt"
    ck_b = OUT_DIR / f"wardfree_new_ep{ep_wf}_{stamp}.pt"
    model_a = TunedNet(v1.BasketNet(*S.arch).to(device), CorrectionNet().to(device))
    model_a.load_state_dict(torch.load(ck_a, map_location="cpu", weights_only=False)["state_dict"])
    model_a.eval()
    baskets_wf = [basket_key(sorted(S.group_main[g].items())) for g in range(1, S.n_groups)]
    cat_idx_wf, cat_cnt_wf, n_items_wf = basket_tensors(baskets_wf, S.n_groups, device)
    model_b = v1.BasketNet(len(S.champ_index), len(S.item_index), len(S.label_index), len(S.item_index) + 1, n_items_wf).to(device)
    model_b.load_state_dict(torch.load(ck_b, map_location="cpu", weights_only=False)["state_dict"])
    model_b.eval()
    dom_val = torch.load(OUT_DIR / f"domains_cohort_{stamp}.pt", weights_only=True)
    econ_val = torch.load(OUT_DIR / f"econ_cohort_{stamp}.pt", weights_only=True)
    stats = torch.load(OUT_DIR / f"econ_stats_{stamp}.pt", weights_only=True)
    assert int(dom_val["n_rows"]) == S.n_val == econ_val.shape[0]
    val_domain = dom_val["top_idx"].long()
    sm, ss = stats["mean"].to(device), stats["std"].to(device)
    print(f"setup listo ({S.setup_seconds}s): cohorte {S.n_val} / {len(S.cohort_games)} partidas; grupos {S.n_groups}", flush=True)

    rec = {"econ_ft_new": [], "wardfree_new": [], "M2": []}
    agree = Counter()
    with torch.no_grad():
        vecs_b = model_b.baskets.vectors(cat_idx_wf, cat_cnt_wf, torch.arange(S.n_groups, device=device))
        for s in range(0, S.n_val, 512):
            sl = slice(s, min(s + 512, S.n_val))
            batch = range_batch(S.val_ds, sl)
            dom = val_domain[sl].to(device)
            sA, _ = model_a.corrected_scores(batch, device, S.cat_idx, S.cat_cnt, S.n_cat, dom,
                                             (econ_val[sl].float().to(device) - sm) / ss)
            lA = group_logits(sA, S.gid, S.n_groups)
            P_A = torch.softmax(lA, dim=1)
            sB = model_b.scores(model_b.state(batch, device), vecs_b)
            P_B = torch.softmax(sB, dim=1)
            P_M = WEIGHT_A * P_A + WEIGHT_B * P_B
            k = min(v1.SHORTLIST, S.n_groups)
            orders = {"econ_ft_new": torch.topk(lA.cpu(), k, dim=1).indices,
                      "wardfree_new": torch.topk(sB.cpu(), k, dim=1).indices,
                      "M2": torch.topk(P_M.cpu(), k, dim=1).indices}
            for name, order in orders.items():
                rec[name].extend(decode_batch(order, S.val_rows[sl], S.group_main, S))
            for j in range(len(S.val_rows[sl])):
                f = {name: json.dumps(rec[name][s + j]["pred_first"], sort_keys=True) for name in rec}
                agree["A_eq_B"] += f["econ_ft_new"] == f["wardfree_new"]
                agree["M2_eq_A"] += f["M2"] == f["econ_ft_new"]
                agree["M2_eq_B"] += f["M2"] == f["wardfree_new"]
                agree["M2_novel"] += f["M2"] != f["econ_ft_new"] and f["M2"] != f["wardfree_new"]
    for name, recs in rec.items():
        p2.dump(OUT_DIR / f"rows_{name}_mix_{stamp}.jsonl", recs)
    # ---- integrity: both parents reproduce their own segment rows ----
    repro = {"econ_ft_new_vs_segment_rows": mismatches(rec["econ_ft_new"], load_rows(OUT_DIR / f"rows_econ_ft_new_{stamp}.jsonl", S.n_val)),
             "wardfree_new_vs_segment_rows": mismatches(rec["wardfree_new"], load_rows(OUT_DIR / f"rows_wardfree_new_{stamp}.jsonl", S.n_val))}
    gate = all(v == 0 for d in repro.values() for v in d.values())
    print("reproduccion:", json.dumps(repro), "| gate", "OK" if gate else "FAILED", flush=True)
    arms = {**rec, "new_bf": reference_rows(S.n_val)}
    report = {"schema": 1, "segment": "D_mixture", "protocol": "docs/experiment_stage2_econ_wardfree_v1.md", "mode": stamp,
              "created_utc": now_utc(), "cohort_rows": S.n_val, "cohort_games": len(S.cohort_games),
              "weights": {"econ_ft_new": WEIGHT_A, "wardfree_new": WEIGHT_B}, "weights_tuned": False,
              "parents": {"econ_ft_new": {"path": str(ck_a.relative_to(OUT_DIR.parents[1])), "sha256": sha256_file(ck_a)},
                          "wardfree_new": {"path": str(ck_b.relative_to(OUT_DIR.parents[1])), "sha256": sha256_file(ck_b)}},
              "reference_sha256": S.ckpt_sha, "reproduction_mismatches": repro, "integrity_gate_passed": gate,
              "first_recommendation_agreement": dict(agree)}
    compare(arms, [("M2", "new_bf"), ("M2", "econ_ft_new"), ("M2", "wardfree_new"), ("econ_ft_new", "wardfree_new")],
            report, draw_pairs=[("M2", "new_bf")])
    fe = {k: [r["first_action_exact"] for r in v] for k, v in arms.items()}
    report["movement"]["both_parents_wrong_M2_right"] = sum(1 for m, a, b in zip(fe["M2"], fe["econ_ft_new"], fe["wardfree_new"]) if m and not a and not b)
    report["movement"]["both_parents_right_M2_wrong"] = sum(1 for m, a, b in zip(fe["M2"], fe["econ_ft_new"], fe["wardfree_new"]) if not m and a and b)
    report["movement"]["oracle_union_parents"] = sum(1 for a, b in zip(fe["econ_ft_new"], fe["wardfree_new"]) if a or b) / max(S.n_val, 1)
    pv = report["paired"]["M2_vs_new_bf"]
    report["decision"] = {"rule": "M2 adopted as the practical reference only if it beats R1 (new Bf) with the 95% CI excluding 0; integrity gate must pass",
                          "adopt_M2": bool(gate and beats(report, "M2", "new_bf")),
                          "M2_vs_R1_points": round(pv["diff"] * 100, 3), "ci95_points": [round(x * 100, 3) for x in pv["ci95"]]}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_report(OUT_DIR / f"mixture_report_{stamp}.json", report)
    print(json.dumps({"aggregates": {k: v["first_action_exact"]["value"] for k, v in report["aggregates"].items()},
                      "decision": report["decision"], "agreement": dict(agree)}, indent=1), flush=True)
    if not gate:
        sys.exit(EXIT_GATE)


if __name__ == "__main__":
    main()
