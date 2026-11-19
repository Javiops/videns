"""Stage-2 budget pilot on the v4 generation: measures, never trains.

Protocol: docs/experiment_stage2_econ_wardfree_v1.md. Three timings, each
preceded by untimed warm-up (the cold-start lesson of 2026-09-24):
  (A) econ features, ms/decision on 200 real FIT rows with the new Bf's
      top-128 domain -> hours for FIT + cohort;
  (B) s/effective batch of the twin econ fine-tune step from new Bf_ep16
      (TunedNet + zero-init CorrectionNet, full-catalogue CE, backward) ->
      hours for 8 epochs x 2 arms;
  (C) s/effective batch of the ward-free recipe (clean BasketNet on the
      9,333-group catalogue, full CE) -> hours for 16 epochs.
Run via footwear_runtime with PREFIX_ML_DIR = the v4 generation and
--review = the v4 snapshot's static review.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import torch.nn as nn

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import basket_key
from basket_bf_newgen import group_map, basket_tensors, sha256_file
from basket_econ_cache import N_ECON, econ_features, to_action
from basket_econ_finetune import TunedNet, LR as FT_LR, EPOCHS as FT_EPOCHS
from basket_residual_econ import CorrectionNet
from basket_wardfree import EPOCHS as WF_EPOCHS

OUT_DIR = ROOT / "data" / "experiments" / "16.18-stage2-budget-v1"
STAGE1 = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v4-bf-stage1"
WARMUP, TIMED = 2, 12
BENCH_ROWS, BENCH_WARMUP = 200, 5
BOUNDS_H = {"A_econ_cache": 6.0, "B_twin_finetune": 12.0, "C_wardfree": 8.0}   # declared in the protocol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(encoding="utf-8"))["input_schema"]
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    config = torch.load(ref_artifact, map_location="cpu", weights_only=False)["config"]
    config = {**config, "max_items": schema["max_modeled_inventory_items"],
              "max_copies": schema["max_copies_of_one_item"],
              "max_basket_items": schema["max_basket_items"]}
    tp.apply_config(config)
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    tp.D_MODEL, tp.N_LAYERS, tp.FF_DIM, tp.N_HEADS = (int(config[k]) for k in ("d_model", "layers", "ff_dim", "heads"))
    assert not tp.USE_HISTORY and tp.D_MODEL == 64

    assignment = json.loads((generation / "split_assignment.json").read_text(encoding="utf-8"))
    train_set = {m for m, f in assignment.items() if f == "train"}
    val_set = {m for m, f in assignment.items() if f == "val"}
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, _ = v1.first_pass(train_path, train_set, None, None)
    champ_index, item_index = tp._index_map(champ_ids), tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(), key=lambda kv: (-kv[1], kv[0]))]
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gids, group_main = group_map(baskets)
    n_groups = len(group_main)
    cat_idx, cat_cnt, n_items = basket_tensors(baskets, n_cat, device)
    report1 = json.loads((STAGE1 / "bf_stage1_report_full.json").read_text(encoding="utf-8"))
    assert len(baskets) == report1["catalogue"] and n_groups == report1["groups"]

    expect = {"history": False, "extras": True, "gold_est": True, "gold_x": True, "runes": True, "pov_gold": True}
    fit_ds = tp.ShopDataset.from_cache(STAGE1 / "cache_fit_full.pt", champ_index, item_index, label_index, dragon, expect=expect)
    fit_targets = torch.load(STAGE1 / "targets_fit_full.pt", weights_only=True)
    n_fit = fit_ds.champs.shape[0]
    assert n_fit == report1["fit_rows"] and fit_targets.shape[0] == n_fit
    if device.type == "cuda":
        fit_ds.to_device(device)
    fit_targets = fit_targets.to(device)
    n_cohort = report1["cohort_rows"]

    ckpt = STAGE1 / "bf_newgen_ep16_full.pt"
    assert sha256_file(ckpt) == report1["checkpoint"]["sha256"]
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = (len(champ_index), len(item_index), len(label_index), len(item_index) + 1, n_items)
    base = v1.BasketNet(*arch).to(device)
    base.load_state_dict(blob["state_dict"])
    base.eval()
    print(f"setup listo ({time.monotonic() - started:.0f}s): fit {n_fit} filas, cohorte {n_cohort}, "
          f"catalogo {len(baskets)}, grupos {n_groups}", flush=True)

    def slice_batch(ds, idxs):
        return {k: v[idxs.to(v.device)] for k, v in {"champs": ds.champs, "sides": ds.sides, "items": ds.items,
                                                       "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}

    nll = nn.CrossEntropyLoss()
    out = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
           "protocol": "docs/experiment_stage2_econ_wardfree_v1.md", "generation": generation.name,
           "fit_rows": n_fit, "cohort_rows": n_cohort, "catalogue": len(baskets), "groups": n_groups,
           "device": str(device), "warmup_batches_untimed": WARMUP, "timed_batches": TIMED, "bounds_h": BOUNDS_H}

    # ---- (A) econ features on real FIT rows, new Bf's top-128 domain ----
    rows = list(v1.stream_split(train_path, train_set, True, champ_index, dragon, BENCH_WARMUP + BENCH_ROWS))
    with torch.no_grad():
        vecs = base.baskets.vectors(cat_idx, cat_cnt, torch.arange(n_cat, device=device))
        idx = torch.arange(len(rows))
        scores = base.scores(base.state(slice_batch(fit_ds, idx), device), vecs)
        dom = torch.topk(scores, v1.SHORTLIST, dim=1).indices.cpu()
    counters = Counter()

    def features_for(j):
        row = rows[j]
        budget = float(row["gold_est"])
        for k in range(v1.SHORTLIST):
            cand = int(dom[j, k])
            if cand:
                econ_features(to_action(baskets[cand - 1]), row, budget, dragon, counters)

    for j in range(BENCH_WARMUP):
        features_for(j)
    t0 = time.monotonic()
    for j in range(BENCH_WARMUP, BENCH_WARMUP + BENCH_ROWS):
        features_for(j)
    per_row = (time.monotonic() - t0) / BENCH_ROWS
    hours_a = per_row * (n_fit + n_cohort) / 3600
    out["A_econ_cache"] = {"bench_rows": BENCH_ROWS, "ms_per_decision": round(per_row * 1e3, 2),
                           "ms_per_candidate": round(per_row * 1e3 / v1.SHORTLIST, 4),
                           "decisions": n_fit + n_cohort, "projected_hours_one_core": round(hours_a, 2),
                           "v3_anchor_ms_per_decision": 15.2, "within_bound": hours_a <= BOUNDS_H["A_econ_cache"]}
    print(f"A econ: {per_row * 1e3:.1f} ms/decision -> {hours_a:.2f} h para {n_fit + n_cohort} decisiones (tope {BOUNDS_H['A_econ_cache']})", flush=True)

    # ---- (B) twin econ fine-tune step from new Bf (backward included) ----
    torch.manual_seed(16)
    tuned = TunedNet(v1.BasketNet(*arch).to(device), CorrectionNet().to(device))
    tuned.base.load_state_dict(blob["state_dict"])
    tuned.train()
    opt = torch.optim.Adam(tuned.parameters(), lr=FT_LR)
    g = torch.Generator().manual_seed(99)
    batches = [torch.randperm(n_fit, generator=g)[:p2.EFFECTIVE] for _ in range(WARMUP + TIMED)]
    with torch.no_grad():                       # the real trainer reads the domain from the cache
        doms = []
        for idxs_all in batches:
            sc = base.scores(base.state(slice_batch(fit_ds, idxs_all), device), vecs)
            doms.append(torch.topk(sc, v1.SHORTLIST, dim=1).indices)
    econ_zero = torch.zeros((p2.MICRO, v1.SHORTLIST, N_ECON), device=device)
    t0 = None
    for i, idxs_all in enumerate(batches):
        if i == WARMUP:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.monotonic()
        opt.zero_grad()
        for m0 in range(0, len(idxs_all), p2.MICRO):
            idxs = idxs_all[m0:m0 + p2.MICRO]
            sc, _ = tuned.corrected_scores(slice_batch(fit_ds, idxs), device, cat_idx, cat_cnt, n_cat,
                                           doms[i][m0:m0 + p2.MICRO], econ_zero[:len(idxs)])
            loss = nll(sc, fit_targets[idxs.to(device)])
            (loss * len(idxs) / len(idxs_all)).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_b = (time.monotonic() - t0) / TIMED
    hours_b = per_b * (n_fit / p2.EFFECTIVE) * FT_EPOCHS * 2 / 3600
    out["B_twin_finetune"] = {"sec_per_effective_batch": round(per_b, 3), "epochs": FT_EPOCHS, "arms": 2,
                              "projected_hours": round(hours_b, 2), "v3_anchor_hours": 2.12,
                              "within_bound": hours_b <= BOUNDS_H["B_twin_finetune"]}
    print(f"B twin FT: {per_b:.2f} s/lote -> {hours_b:.2f} h ({FT_EPOCHS} epocas x 2 brazos; tope {BOUNDS_H['B_twin_finetune']})", flush=True)
    del tuned, opt, doms
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- (C) ward-free recipe on the group catalogue (clean init) ----
    baskets_wf = [basket_key(sorted(group_main[g].items())) for g in range(1, n_groups)]
    cat_idx_wf, cat_cnt_wf, n_items_wf = basket_tensors(baskets_wf, n_groups, device)
    targets_wf = gids.to(device)[fit_targets]
    torch.manual_seed(16)
    wf = v1.BasketNet(len(champ_index), len(item_index), len(label_index), len(item_index) + 1, n_items_wf).to(device)
    wf.train()
    opt = torch.optim.Adam(wf.parameters(), lr=1e-3)
    t0 = None
    for i, idxs_all in enumerate(batches):
        if i == WARMUP:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.monotonic()
        opt.zero_grad()
        for m0 in range(0, len(idxs_all), p2.MICRO):
            idxs = idxs_all[m0:m0 + p2.MICRO]
            state = wf.state(slice_batch(fit_ds, idxs), device)
            vw = wf.baskets.vectors(cat_idx_wf, cat_cnt_wf, torch.arange(n_groups, device=device))
            loss = nll(wf.scores(state, vw), targets_wf[idxs.to(device)])
            (loss * len(idxs) / len(idxs_all)).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_c = (time.monotonic() - t0) / TIMED
    hours_c = per_c * (n_fit / p2.EFFECTIVE) * WF_EPOCHS / 3600
    out["C_wardfree"] = {"sec_per_effective_batch": round(per_c, 3), "epochs": WF_EPOCHS, "groups": n_groups,
                         "projected_hours": round(hours_c, 2), "v3_anchor_hours": 1.29,
                         "stage1_bf_measured_hours": 7.71, "within_bound": hours_c <= BOUNDS_H["C_wardfree"]}
    print(f"C wardfree: {per_c:.2f} s/lote -> {hours_c:.2f} h ({WF_EPOCHS} epocas; tope {BOUNDS_H['C_wardfree']})", flush=True)

    out["totals"] = {"gpu_hours_B_plus_C": round(hours_b + hours_c, 2), "cpu_hours_A_one_core": round(hours_a, 2),
                     "wall_hours_if_A_overlaps_C": round(max(hours_a, hours_c) + hours_b, 2),
                     "note": "evaluation of every configuration on the cohort is minutes once A caches the cohort features; "
                             "34 rows/s (~72 min per configuration) without it"}
    out["elapsed_seconds"] = round(time.monotonic() - started, 1)
    path = OUT_DIR / "budget_pilot.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["totals"], indent=1), flush=True)
    print(f"report: {path}", flush=True)


if __name__ == "__main__":
    main()
