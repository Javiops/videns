"""Frozen mixtures of M2 with the transfer arms, inference only (owner, 2026-09-29).

  M2H = 0.5 * P(M2) + 0.5 * P(H)        M2T = 0.5 * P(M2) + 0.5 * P(T)
P(M2) is computed exactly as stage 2 did (0.5 * softmax(grouped econ_ft_new corrected scores) + 0.5 *
softmax(wardfree_new group scores)) with the stage-1 vocabulary; P(H) and P(T) are the softmax of the
grouped full-catalogue scores of the transfer checkpoints with the common vocabulary. All live in the same
9,333-group space (asserted: identical group catalogue). Weights declared, never tuned.

Integrity gates: M2 reproduces rows_M2_mix_full (0/0), H reproduces rows_H_full (0/0), T reproduces
rows_T_full (0/0) in THIS process; references digest-bound. Readings (paired by game, 95 % CI):
M2H vs M2 and M2T vs M2 (does adding another model help?), M2H vs M2T (is the history specifically
worth anything?), plus each mixture vs R1. Decision rule, declared: nothing is adopted unless a mixture
beats M2 with the CI excluding 0; a history-specific advantage needs M2H − M2T with the CI excluding 0.
Run via footwear_runtime with PREFIX_ML_DIR = the v4 generation (--review = the v4 static review).
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from stage2_newgen import (OUT_DIR as STAGE2_DIR, STAGE1, EXIT_GATE, setup, range_batch, group_logits, decode_batch,
                           reference_rows, compare, beats, write_report, now_utc, v1, p2, tp, sha256_file)
from basket_newbf_mixture import load_rows, mismatches
from basket_vocabulary_probe import basket_key
from basket_bf_newgen import basket_tensors, group_map
from basket_econ_finetune import TunedNet
from basket_residual_econ import CorrectionNet
from basket_transfer_ab import basket_tensors_shared, FROZEN_R1, FROZEN_R2
from prepare_transfer_caches import OUT_DIR as TRANSFER_DIR, GEN_1618

W_M2, W_ARM = 0.5, 0.5            # declared, never tuned
ARMS = ("H", "T")


def main():
    global TRANSFER_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--transfer-dir", type=Path, default=TRANSFER_DIR, help="transfer experiment directory (the smoke chain has its own)")
    args = ap.parse_args()
    TRANSFER_DIR = args.transfer_dir.resolve()
    started = time.monotonic()
    S = setup(args.smoke, load_fit=False, load_model=False)
    stamp, device = S.stamp, S.device
    tstamp = "smoke" if args.smoke else "full"

    # ---- references bound before anything is scored ----
    r1_ref = json.loads(FROZEN_R1.read_text(encoding="utf-8")); r2_ref = json.loads(FROZEN_R2.read_text(encoding="utf-8"))
    bound = {"R1_rows": (STAGE1 / "rows_new_bf_full.jsonl", r1_ref["rows"]["rows_new_bf_full.jsonl"]),
             "M2_rows": (STAGE2_DIR / "rows_M2_mix_full.jsonl", r2_ref["rows"]["M2_mix"])}
    for name, (path, expected) in bound.items():
        if sha256_file(path) != expected:
            raise SystemExit(f"{name} digest differs from the frozen reference; no run")
    transfer_report = json.loads((TRANSFER_DIR / f"transfer_report_{tstamp}.json").read_text(encoding="utf-8"))

    # ---- M2's parents, exactly as stage 2 (stage-1 vocabulary) ----
    ep_ft, ep_wf = (1, 1) if args.smoke else (8, 16)
    ck_a = STAGE2_DIR / f"econ_ft_new_ep{ep_ft}_{stamp}.pt"
    ck_b = STAGE2_DIR / f"wardfree_new_ep{ep_wf}_{stamp}.pt"
    for name, ck in (("econ_ft_new_ep8", ck_a), ("wardfree_new_ep16", ck_b)):
        if not args.smoke and sha256_file(ck) != r2_ref["components"][name]["sha256"]:
            raise SystemExit(f"{name} checkpoint digest differs from the frozen reference R2")
    model_a = TunedNet(v1.BasketNet(*S.arch).to(device), CorrectionNet().to(device))
    model_a.load_state_dict(torch.load(ck_a, map_location="cpu", weights_only=False)["state_dict"]); model_a.eval()
    baskets_wf = [basket_key(sorted(S.group_main[g].items())) for g in range(1, S.n_groups)]
    cat_idx_wf, cat_cnt_wf, n_items_wf = basket_tensors(baskets_wf, S.n_groups, device)
    model_b = v1.BasketNet(len(S.champ_index), len(S.item_index), len(S.label_index), len(S.item_index) + 1, n_items_wf).to(device)
    model_b.load_state_dict(torch.load(ck_b, map_location="cpu", weights_only=False)["state_dict"]); model_b.eval()
    dom_val = torch.load(STAGE2_DIR / f"domains_cohort_{stamp}.pt", weights_only=True)
    econ_val = torch.load(STAGE2_DIR / f"econ_cohort_{stamp}.pt", weights_only=True)
    stats = torch.load(STAGE2_DIR / f"econ_stats_{stamp}.pt", weights_only=True)
    assert int(dom_val["n_rows"]) == S.n_val == econ_val.shape[0]
    val_domain = dom_val["top_idx"].long()
    sm, ss = stats["mean"].to(device), stats["std"].to(device)

    # ---- the transfer arms (common vocabulary, own cohort tensors, same group space) ----
    vocab = json.loads((TRANSFER_DIR / "vocab.json").read_text(encoding="utf-8"))
    champ_index = {int(k): v for k, v in vocab["champ_index"].items()}
    item_index = {int(k): v for k, v in vocab["item_index"].items()}
    label_index = {int(k): v for k, v in vocab["label_index"].items()}
    item_map = {it: k + 1 for k, it in enumerate(vocab["basket_items"])}
    cat18 = json.loads((TRANSFER_DIR / f"catalogue_1618_{tstamp}.json").read_text(encoding="utf-8"))
    assert cat18["vocab_sha256"] == vocab["digest"]
    baskets18 = cat18["baskets"]
    assert baskets18 == S.baskets, "the transfer catalogue is not the stage-1 catalogue"
    gids18, group_main18 = group_map(baskets18)
    assert group_main18 == S.group_main and torch.equal(gids18, S.gids), "group space differs from stage 1"
    cat_idx18, cat_cnt18 = (t.to(device) for t in basket_tensors_shared(baskets18, len(baskets18) + 1, item_map))
    n_cat18 = len(baskets18) + 1
    expect = {"history": False, "extras": True, "gold_est": True, "gold_x": True, "runes": True, "pov_gold": True,
              "vocab_sha256": vocab["digest"], "patch": "16.18", "split": "cohort"}
    cds = tp.ShopDataset.from_cache(TRANSFER_DIR / f"cache_cohort_{tstamp}.pt", champ_index, item_index, label_index, S.dragon, expect=expect)
    assert cds.champs.shape[0] == S.n_val, f"transfer cohort {cds.champs.shape[0]} rows vs stage-1 cohort {S.n_val}"
    if device.type == "cuda":
        cds.to_device(device)
    arch = (len(champ_index), len(item_index), len(label_index), len(item_index) + 1, len(vocab["basket_items"]))
    models = {}
    for arm in ARMS:
        ck = TRANSFER_DIR / f"{arm}_ep{1 if args.smoke else 16}_{tstamp}.pt"
        if not args.smoke and sha256_file(ck) != transfer_report["checkpoints"][arm]["sha256"]:
            raise SystemExit(f"{arm} checkpoint digest differs from the transfer report")
        m = v1.BasketNet(*arch).to(device)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]); m.eval()
        models[arm] = m
    print(f"setup listo ({time.monotonic() - started:.0f}s): cohorte {S.n_val} / {len(S.cohort_games)} partidas; grupos {S.n_groups}; "
          f"catalogo comun asegurado", flush=True)

    rec = {"M2": [], "H": [], "T": [], "M2H": [], "M2T": []}
    agree = Counter()
    k = min(v1.SHORTLIST, S.n_groups)
    with torch.no_grad():
        vecs_b = model_b.baskets.vectors(cat_idx_wf, cat_cnt_wf, torch.arange(S.n_groups, device=device))
        vecs_arm = {arm: models[arm].baskets.vectors(cat_idx18, cat_cnt18, torch.arange(n_cat18, device=device)) for arm in ARMS}
        for s in range(0, S.n_val, 512):
            sl = slice(s, min(s + 512, S.n_val))
            batch = range_batch(S.val_ds, sl)
            dom = val_domain[sl].to(device)
            sA, _ = model_a.corrected_scores(batch, device, S.cat_idx, S.cat_cnt, S.n_cat, dom, (econ_val[sl].float().to(device) - sm) / ss)
            P_M2 = W_M2 * torch.softmax(group_logits(sA, S.gid, S.n_groups), dim=1) + (1 - W_M2) * torch.softmax(model_b.scores(model_b.state(batch, device), vecs_b), dim=1)
            batch_c = range_batch(cds, sl)
            P_arm = {arm: torch.softmax(group_logits(models[arm].scores(models[arm].state(batch_c, device), vecs_arm[arm]), S.gid, S.n_groups), dim=1) for arm in ARMS}
            orders = {"M2": torch.topk(P_M2.cpu(), k, dim=1).indices,
                      "H": torch.topk(P_arm["H"].cpu(), k, dim=1).indices, "T": torch.topk(P_arm["T"].cpu(), k, dim=1).indices,
                      "M2H": torch.topk((W_M2 * P_M2 + W_ARM * P_arm["H"]).cpu(), k, dim=1).indices,
                      "M2T": torch.topk((W_M2 * P_M2 + W_ARM * P_arm["T"]).cpu(), k, dim=1).indices}
            for name, order in orders.items():
                rec[name].extend(decode_batch(order, S.val_rows[sl], S.group_main, S))
            for j in range(len(S.val_rows[sl])):
                f = {name: json.dumps(rec[name][s + j]["pred_first"], sort_keys=True) for name in rec}
                agree["M2H_eq_M2"] += f["M2H"] == f["M2"]; agree["M2H_eq_H"] += f["M2H"] == f["H"]
                agree["M2T_eq_M2"] += f["M2T"] == f["M2"]; agree["M2T_eq_T"] += f["M2T"] == f["T"]
                agree["M2H_eq_M2T"] += f["M2H"] == f["M2T"]
            if (s // 512) % 40 == 0:
                print(f"  {s + len(S.val_rows[sl])}/{S.n_val} ({time.monotonic() - started:.0f}s)", flush=True)
    out_dir = TRANSFER_DIR
    for name, recs in rec.items():
        p2.dump(out_dir / f"rows_mix_{name}_{tstamp}.jsonl", recs)
    # ---- integrity: every parent reproduces its persisted rows in this process ----
    repro = {"M2_vs_stage2_rows": mismatches(rec["M2"], load_rows(STAGE2_DIR / f"rows_M2_mix_{stamp}.jsonl", S.n_val)),
             "H_vs_transfer_rows": mismatches(rec["H"], load_rows(TRANSFER_DIR / f"rows_H_{tstamp}.jsonl", S.n_val)),
             "T_vs_transfer_rows": mismatches(rec["T"], load_rows(TRANSFER_DIR / f"rows_T_{tstamp}.jsonl", S.n_val))}
    gate = all(v == 0 for d in repro.values() for v in d.values())
    print("reproduccion:", json.dumps(repro), "| gate", "OK" if gate else "FAILED", flush=True)
    arms = {**rec, "R1": reference_rows(S.n_val)}
    report = {"schema": 1, "purpose": "frozen_mixtures_M2_with_transfer_arms", "protocol": "docs/experiment_prior_patch_transfer_v1.md section 10",
              "mode": tstamp, "created_utc": now_utc(), "cohort_rows": S.n_val, "cohort_games": len(S.cohort_games),
              "weights": {"M2": W_M2, "arm": W_ARM}, "weights_tuned": False,
              "components": {"econ_ft_new": sha256_file(ck_a), "wardfree_new": sha256_file(ck_b),
                             "H": sha256_file(TRANSFER_DIR / f"H_ep{1 if args.smoke else 16}_{tstamp}.pt"),
                             "T": sha256_file(TRANSFER_DIR / f"T_ep{1 if args.smoke else 16}_{tstamp}.pt")},
              "references": {"R1_rows_sha256": bound["R1_rows"][1], "M2_rows_sha256": bound["M2_rows"][1]},
              "reproduction_mismatches": repro, "integrity_gate_passed": gate, "first_recommendation_agreement": dict(agree)}
    pairs = [("M2H", "M2"), ("M2T", "M2"), ("M2H", "M2T"), ("M2H", "R1"), ("M2T", "R1"), ("M2H", "H"), ("M2T", "T")]
    compare(arms, pairs, report, draw_pairs=[("M2H", "M2"), ("M2T", "M2"), ("M2H", "M2T")])
    fe = {k: [float(r["first_action_exact"]) for r in v] for k, v in arms.items()}
    report["movement"]["M2_wrong_H_right"] = sum(1 for m, h in zip(fe["M2"], fe["H"]) if h and not m)
    report["movement"]["M2_wrong_T_right"] = sum(1 for m, t in zip(fe["M2"], fe["T"]) if t and not m)
    report["movement"]["oracle_union_M2_H"] = sum(1 for m, h in zip(fe["M2"], fe["H"]) if m or h) / S.n_val
    report["movement"]["oracle_union_M2_T"] = sum(1 for m, t in zip(fe["M2"], fe["T"]) if m or t) / S.n_val
    report["decision"] = {
        "rule": "adopt nothing unless a mixture beats M2 with the 95% CI excluding 0; a history-specific advantage needs M2H - M2T with the CI excluding 0; integrity gate must pass",
        "M2H_beats_M2": bool(gate and beats(report, "M2H", "M2")), "M2T_beats_M2": bool(gate and beats(report, "M2T", "M2")),
        "M2H_beats_M2T": bool(gate and beats(report, "M2H", "M2T"))}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_report(out_dir / f"mixture_report_{tstamp}.json", report)
    print(json.dumps({"aggregates": {k: round(v["first_action_exact"]["value"], 5) for k, v in report["aggregates"].items()},
                      "paired": {k: (round(v["diff"] * 100, 3), [round(x * 100, 3) for x in v["ci95"]]) for k, v in report["paired"].items() if not k.endswith("_any3")},
                      "decision": report["decision"], "agreement": dict(agree)}, indent=1), flush=True)
    if not gate:
        sys.exit(EXIT_GATE)


if __name__ == "__main__":
    main()
