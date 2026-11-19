"""Prior-patch transfer, two symmetric arms (protocol v2, docs/experiment_prior_patch_transfer_v1.md).

  T: U updates on the 16.18 FIT  -> reset Adam + scheduler -> 16 epochs on the 16.18 FIT
  H: U updates on the 16.17 FIT  -> reset Adam + scheduler -> 16 epochs on the 16.18 FIT
Same recipe in both phases (full-catalogue CE, micro 32 / effective 256, Adam 1e-3, cosine),
same seeds, the SAME phase-2 batch order (one generator, seed 16). One common vocabulary
(prepare_transfer_caches.py) for every indexed structure; each phase its own action
catalogue; the basket encoder's item table is the union of both catalogues, so nothing is
re-indexed between phases. U = 11,844 updates (four epochs of the 16.18 FIT), declared.

Scored on the v4 forward cohort against the persisted references: R1 (new Bf, stage 1) and
R2 (M2, stage 2). Readings kept: H vs T (history), H vs R1, H vs R2, T vs R1. The history
corpus is re-validated here before any of its rows are read (history_1617.validate_history).

--pilot-only measures phase-1 and phase-2 steady-state costs (2 untimed warm-up + 12 timed
effective batches) and exits: the budget is fixed before any result. --smoke caps everything.
Run via footwear_runtime with PREFIX_ML_DIR = the v4 generation and --review = the v4 review.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from stage2_newgen import (STAGE1, OUT_DIR as STAGE2_DIR, group_logits, decode_batch, reference_rows, compare, beats, EXIT_BOUND, EXIT_GATE,
                           pilot, progress_line, now_utc, write_report, sync, v1, p2, tp, sha256_file, load_rows)
from basket_bf_newgen import group_map
from history_1617 import HISTORY_DIR, validate_history
from history_rows_check import validate_rows
from prepare_transfer_caches import OUT_DIR, GEN_1618, ROOT

# Persisted reference rows are bound to the frozen reference files before a single row is read (protocol section 4.4).
FROZEN_R1 = STAGE1 / "frozen_reference_v4.json"
FROZEN_R2 = STAGE2_DIR / "frozen_reference_stage2.json"

UPDATES = 11_844                 # 4 x ceil(757,975 / 256): declared in the protocol before any result
EPOCHS = 16
LR = 1e-3
BOUNDS_H = {"phase1": 3.0, "phase2": 10.0}  # per arm; phase 2 raised 9 -> 10 h by the owner (2026-09-27 21:20, before launch, protocol section 8.9)


def bind_references(report):
    """Protocol section 4.4: the persisted R1/R2 rows are bound to the frozen reference files BEFORE any training."""
    r1_ref, r2_ref = (json.loads(p.read_text(encoding="utf-8")) for p in (FROZEN_R1, FROZEN_R2))
    ref_files = {"R1_new_bf": (STAGE1 / "rows_new_bf_full.jsonl", r1_ref["rows"]["rows_new_bf_full.jsonl"]),
                 "R2_M2": (STAGE2_DIR / "rows_M2_mix_full.jsonl", r2_ref["rows"]["M2_mix"])}
    for name, (path, expected) in ref_files.items():
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"{name} rows {path} digest {actual[:12]} differ from the frozen reference {expected[:12]}; no run")
    report["references"] = {name: {"rows": str(path), "sha256": expected} for name, (path, expected) in ref_files.items()}
    report["references"]["R1_new_bf"]["first_action_exact_wardfree"] = r1_ref["metrics_cohort"]["first_action_exact_wardfree"]
    report["references"]["R2_M2"]["first_action_exact_wardfree"] = r2_ref["metrics_cohort"]["first_action_exact_wardfree"]
    return ref_files


def r1_gate(smoke, n_val, report):
    """Protocol section 4.4: R1 must reproduce its persisted stage-1 rows (0/0) in THIS process, with the live
    decoder, legality and static data. Uses the stage-1 vocabulary (R1's own), then frees the GPU."""
    from stage2_newgen import setup as stage2_setup
    from basket_newbf_mixture import mismatches
    t0 = time.monotonic()
    S = stage2_setup(smoke, need_gpu=True, load_fit=False, load_cohort=True, cohort_tensors=True, load_model=True)
    assert S.n_val == n_val, f"stage-1 cohort has {S.n_val} rows, this run's cohort {n_val}"
    records = []
    with torch.no_grad():
        vecs = S.model.baskets.vectors(S.cat_idx, S.cat_cnt, torch.arange(S.n_cat, device=S.device))
        for s in range(0, S.n_val, 512):
            sl = slice(s, min(s + 512, S.n_val))
            batch = {k: v[sl] for k, v in {"champs": S.val_ds.champs, "sides": S.val_ds.sides, "items": S.val_ds.items,
                                            "query": S.val_ds.query, "inv": S.val_ds.inv, "hist": S.val_ds.hist}.items()}
            sc = S.model.scores(S.model.state(batch, S.device), vecs)
            order = torch.topk(group_logits(sc, S.gid, S.n_groups).cpu(), min(v1.SHORTLIST, S.n_groups), dim=1).indices
            records.extend(decode_batch(order, S.val_rows[sl], S.group_main, S))
    repro = mismatches(records, reference_rows(S.n_val))
    report["r1_reproduction"] = {"checkpoint_sha256": S.ckpt_sha, "rows": S.n_val, "mismatches": repro,
                                 "seconds": round(time.monotonic() - t0, 1)}
    print(f"gate R1: {repro} en {S.n_val} filas ({time.monotonic() - t0:.0f}s)", flush=True)
    del S, records
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if any(repro.values()):
        report["aborted"] = f"R1 does not reproduce its stage-1 rows in this process: {repro}"
        print("ABORTADO: " + report["aborted"], flush=True)
        return False
    return True


def basket_tensors_shared(baskets, n, item_map):
    ci = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
    cc = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, p in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, c = p.split(":")
            ci[row_i, slot] = item_map[int(i)]
            cc[row_i, slot] = min(int(c), v1.MAX_BASKET)
    return ci, cc


def slice_batch(ds, idxs):
    return {k: v[idxs.to(v.device)] for k, v in {"champs": ds.champs, "sides": ds.sides, "items": ds.items,
                                                   "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("T", "H", "both"), default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--compare-only", action="store_true", help="no training: compare the persisted rows_{arm} files with the references")
    ap.add_argument("--history", type=Path, default=HISTORY_DIR, help="validated history export (smoke chains use their own)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="cache/report directory (smoke chains use their own)")
    args = ap.parse_args()
    OUT_DIR = args.out.resolve()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    U = 40 if args.smoke else args.updates
    epochs = 1 if args.smoke else args.epochs
    arms = ["T", "H"] if args.arm == "both" else [args.arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- the history corpus is validated before any of its rows are touched ----
    hist = validate_history(args.history, scan_rows=not args.smoke, hash_db=not args.smoke)
    validate_rows(args.history, hist)                 # rows vs the frozen corpus + reviewed row-builder digests
    if hist["acceptance"].get("selection_limit") and not args.smoke:
        raise SystemExit(f"{args.history} comes from a --limit snapshot ({hist['acceptance']['selection_limit']} sources); smoke only")
    vocab = json.loads((OUT_DIR / "vocab.json").read_text(encoding="utf-8"))
    champ_index = {int(k): v for k, v in vocab["champ_index"].items()}
    item_index = {int(k): v for k, v in vocab["item_index"].items()}
    label_index = {int(k): v for k, v in vocab["label_index"].items()}
    item_map = {it: k + 1 for k, it in enumerate(vocab["basket_items"])}
    n_items_union = len(vocab["basket_items"])
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(GEN_1618)
    schema = json.loads((GEN_1618 / "split_manifest.json").read_text(encoding="utf-8"))["input_schema"]
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    config = torch.load(ref_artifact, map_location="cpu", weights_only=False)["config"]
    config = {**config, "max_items": schema["max_modeled_inventory_items"], "max_copies": schema["max_copies_of_one_item"],
              "max_basket_items": schema["max_basket_items"]}
    tp.apply_config(config)
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    tp.D_MODEL, tp.N_LAYERS, tp.FF_DIM, tp.N_HEADS = (int(config[k]) for k in ("d_model", "layers", "ff_dim", "heads"))
    from app.ddragon import dragon_for_patch
    from app.conditional_policy import purchase_sequence
    dragon = dragon_for_patch(manifest_gen["patch"])
    expect = {"history": False, "extras": True, "gold_est": True, "gold_x": True, "runes": True, "pov_gold": True,
              "vocab_sha256": vocab["digest"]}
    cat18 = json.loads((OUT_DIR / f"catalogue_1618_{stamp}.json").read_text(encoding="utf-8"))
    cat17 = json.loads((OUT_DIR / f"catalogue_1617_{stamp}.json").read_text(encoding="utf-8"))
    assert cat18["vocab_sha256"] == vocab["digest"] == cat17["vocab_sha256"]
    baskets18, baskets17 = cat18["baskets"], cat17["baskets"]
    cat_sha = {p: hashlib.sha256("|".join(b).encode()).hexdigest() for p, b in (("1618", baskets18), ("1617", baskets17))}
    stage1 = json.loads((STAGE1 / "bf_stage1_report_full.json").read_text(encoding="utf-8"))
    gids, group_main = group_map(baskets18)
    n_groups = len(group_main)
    assert len(baskets18) == stage1["catalogue"] and n_groups == stage1["groups"]
    gid = gids.to(device)
    cat_idx18, cat_cnt18 = (t.to(device) for t in basket_tensors_shared(baskets18, len(baskets18) + 1, item_map))
    cat_idx17, cat_cnt17 = (t.to(device) for t in basket_tensors_shared(baskets17, len(baskets17) + 1, item_map))
    n_cat18, n_cat17 = len(baskets18) + 1, len(baskets17) + 1

    def load_fit(patch):
        ds = tp.ShopDataset.from_cache(OUT_DIR / f"cache_fit_{patch}_{stamp}.pt", champ_index, item_index, label_index, dragon,
                                       expect={**expect, "patch": "16.17" if patch == "1617" else "16.18", "split": "fit",
                                               "catalogue_sha256": cat_sha[patch],
                                               **({"history_rows_sha256": hist["rows"]["sha256"]} if patch == "1617" else {})})
        tg = torch.load(OUT_DIR / f"targets_fit_{patch}_{stamp}.pt", weights_only=True)
        assert tg.shape[0] == ds.champs.shape[0]
        if device.type == "cuda":
            ds.to_device(device)
        return ds, tg.to(device)

    # cohort: cached tensors + the raw rows (for legality and truth), digest-bound
    cds = tp.ShopDataset.from_cache(OUT_DIR / f"cache_cohort_{stamp}.pt", champ_index, item_index, label_index, dragon,
                                    expect={**expect, "patch": "16.18", "split": "cohort"})
    if device.type == "cuda":
        cds.to_device(device)
    cap_val = 1500 if args.smoke else None
    val_rows = list(v1.stream_split(GEN_1618 / "visits_val.jsonl", set(), False, champ_index, dragon, cap_val))
    n_val = len(val_rows)
    digest = hashlib.sha256()
    for r in val_rows:
        digest.update(f"{r['match_id']}|{r['ts']};".encode())
    cohort_meta = json.loads((OUT_DIR / f"cohort_rows_{stamp}.json").read_text(encoding="utf-8"))
    assert cohort_meta["n"] == n_val == cds.champs.shape[0] and cohort_meta["rows_sha256"] == digest.hexdigest(), "cohort rows differ from the cached tensors"
    nll = nn.CrossEntropyLoss()
    arch = (len(champ_index), len(item_index), len(label_index), len(item_index) + 1, n_items_union)
    print(f"setup listo ({time.monotonic() - started:.0f}s): vocab {vocab['digest'][:12]}, item table {n_items_union}, "
          f"catalogos 16.18 {len(baskets18)} / 16.17 {len(baskets17)}, grupos {n_groups}, cohorte {n_val}, U {U}, epochs {epochs}", flush=True)

    def make_step(model, opt, ds, targets, cat_idx, cat_cnt, n_cat):
        def step(idxs_all):
            opt.zero_grad()
            vecs = None
            for m0 in range(0, len(idxs_all), p2.MICRO):
                idxs = idxs_all[m0:m0 + p2.MICRO]
                state = model.state(slice_batch(ds, idxs), device)
                vecs = model.baskets.vectors(cat_idx, cat_cnt, torch.arange(n_cat, device=device))
                loss = nll(model.scores(state, vecs), targets[idxs.to(device)])
                (loss * len(idxs) / len(idxs_all)).backward()
            opt.step()
        return step

    report = {"schema": 1, "protocol": "docs/experiment_prior_patch_transfer_v1.md", "mode": stamp, "created_utc": now_utc(),
              "updates_phase1": U, "epochs_phase2": epochs, "lr": LR, "micro": p2.MICRO, "effective": p2.EFFECTIVE,
              "vocab_sha256": vocab["digest"], "vocab_counts": vocab["counts"], "item_table": n_items_union,
              "history": {"manifest_rows_sha256": hist["rows"]["sha256"], "games": hist["games"]["exported_count"], "rows": hist["rows"]["count"],
                          "corpus_manifest_sha256": hist["corpus"]["manifest_sha256"]},
              "catalogues": {"16.18": len(baskets18), "16.17": len(baskets17), "groups": n_groups},
              "cohort_rows": n_val, "pilot": {}, "curves": {}, "checkpoints": {}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"transfer_report_{stamp}.json"

    # ---- references bound and R1 reproduced BEFORE any GPU hour is spent (protocol section 4.4) ----
    ref_files = bind_references(report)
    if not args.pilot_only:
        if not r1_gate(args.smoke, n_val, report):
            write_report(report_path, report)
            sys.exit(EXIT_GATE)
    if args.compare_only:
        rows_by_arm = {arm: load_rows(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", n_val)
                       for arm in arms if (OUT_DIR / f"rows_{arm}_{stamp}.jsonl").is_file()}
        if not rows_by_arm:
            raise SystemExit("no persisted rows_{arm} files to compare")
        report["compare_only"] = sorted(rows_by_arm)
        finish(report, report_path, rows_by_arm, ref_files, n_val, started)
        return

    # ---- pilots (steady state), per arm and phase ----
    fit18, tg18 = load_fit("1618")
    n18 = fit18.champs.shape[0]
    for arm in arms:
        torch.manual_seed(16)
        model = v1.BasketNet(*arch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        if arm == "H":
            fit17, tg17 = load_fit("1617")
            per1 = pilot(make_step(model, opt, fit17, tg17, cat_idx17, cat_cnt17, n_cat17), fit17.champs.shape[0], device)
            del fit17, tg17
        else:
            per1 = pilot(make_step(model, opt, fit18, tg18, cat_idx18, cat_cnt18, n_cat18), n18, device)
        per2 = pilot(make_step(model, opt, fit18, tg18, cat_idx18, cat_cnt18, n_cat18), n18, device)
        # projection always with the DECLARED budget (U updates, 16 epochs over the full 16.18 FIT), whatever the cache size
        U_decl, ep_decl, n18_decl = args.updates, args.epochs, int(cat18["n_fit_rows"])
        h1, h2 = per1 * U_decl / 3600, per2 * math.ceil(n18_decl / p2.EFFECTIVE) * ep_decl / 3600
        report["pilot"][arm] = {"phase1_sec_per_batch": round(per1, 3), "phase1_hours": round(h1, 2),
                                "phase2_sec_per_batch": round(per2, 3), "phase2_hours": round(h2, 2), "total_hours": round(h1 + h2, 2),
                                "projection": {"updates": U_decl, "epochs": ep_decl, "fit_rows_1618": n18_decl, "cache_rows_1618": n18,
                                               "catalogue_1617_from": hist["games"]["exported_count"]}}
        print(f"pilot {arm}: fase1 {per1:.3f} s/lote -> {h1:.2f} h ({U_decl} updates); fase2 {per2:.3f} s/lote -> {h2:.2f} h "
              f"({ep_decl} epocas x {math.ceil(n18_decl / p2.EFFECTIVE)} lotes); total {h1 + h2:.2f} h", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (OUT_DIR / f"pilot_{stamp}.json").write_text(json.dumps(report["pilot"], indent=2), encoding="utf-8")
    if args.pilot_only:
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_report(OUT_DIR / f"pilot_report_{stamp}.json", report)
        return
    # ---- declared bounds (protocol section 8.2): a pilot above its bound stops the run before it starts ----
    over = [(arm, ph, report["pilot"][arm][f"{ph}_hours"]) for arm in arms for ph in ("phase1", "phase2")
            if report["pilot"][arm][f"{ph}_hours"] > BOUNDS_H[ph]]
    report["bounds_hours"] = BOUNDS_H
    if over and not args.smoke:
        report["aborted"] = "pilot above its declared bound: " + "; ".join(f"{a} {ph} {h:.2f} h > {BOUNDS_H[ph]} h" for a, ph, h in over)
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_report(report_path, report)
        print("ABORTADO: " + report["aborted"], flush=True)
        sys.exit(EXIT_BOUND)

    # ---- train both arms ----
    rows_by_arm = {}
    for arm in arms:
        torch.manual_seed(16)
        model = v1.BasketNet(*arch).to(device)
        progress = OUT_DIR / f"transfer_progress_{arm}_{stamp}.jsonl"
        # phase 1: U updates on the arm's data, cosine over U, one cycling order (seed 16)
        if arm == "H":
            ds1, tg1, ci1, cc1, nc1 = (*load_fit("1617"), cat_idx17, cat_cnt17, n_cat17)
        else:
            ds1, tg1, ci1, cc1, nc1 = fit18, tg18, cat_idx18, cat_cnt18, n_cat18
        n1 = ds1.champs.shape[0]
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=U)
        step = make_step(model, opt, ds1, tg1, ci1, cc1, nc1)
        order_gen = torch.Generator().manual_seed(16)
        order, pos, done, t_ph = torch.randperm(n1, generator=order_gen), 0, 0, time.monotonic()
        model.train()
        while done < U:
            if pos >= n1:
                order, pos = torch.randperm(n1, generator=order_gen), 0
            idxs_all = order[pos:pos + p2.EFFECTIVE]
            pos += p2.EFFECTIVE
            step(idxs_all)
            sched.step()
            done += 1
            if done % 500 == 0 or done == U:
                entry = {"arm": arm, "phase": 1, "updates": done, "seconds": round(time.monotonic() - t_ph, 1), "utc": now_utc()}
                progress_line(progress, entry)
                print(f"  {arm} fase1 {done}/{U} ({entry['seconds']}s)", flush=True)
        p2.save_checkpoint(OUT_DIR / f"{arm}_phase1_{stamp}.pt", model, opt, sched, 0)
        report["curves"][f"{arm}_phase1_seconds"] = round(time.monotonic() - t_ph, 1)
        if arm == "H":
            del ds1, tg1
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # reset, then the identical adaptation: optimiser, scheduler, batch order AND the global RNG (dropout masks)
        torch.manual_seed(16)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        step = make_step(model, opt, fit18, tg18, cat_idx18, cat_cnt18, n_cat18)
        order_gen = torch.Generator().manual_seed(16)            # SAME batch order in both arms
        curve = []
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            order = torch.randperm(n18, generator=order_gen)
            for s in range(0, n18, p2.EFFECTIVE):
                step(order[s:s + p2.EFFECTIVE])
            sched.step()
            entry = {"arm": arm, "phase": 2, "epoch": epoch, "seconds": round(time.monotonic() - t_ep, 1), "utc": now_utc()}
            curve.append(entry)
            progress_line(progress, entry)
            print(f"  {arm} fase2 ep{epoch}/{epochs} ({entry['seconds']}s)", flush=True)
            if epoch in (4, 8, epochs):
                p2.save_checkpoint(OUT_DIR / f"{arm}_ep{epoch}_{stamp}.pt", model, opt, sched, epoch)
        report["curves"][arm] = curve
        ckpt = OUT_DIR / f"{arm}_ep{epochs}_{stamp}.pt"
        report["checkpoints"][arm] = {"path": str(ckpt), "sha256": sha256_file(ckpt)}
        # score on the cohort with the common vocabulary tensors
        model.eval()
        records = []
        with torch.no_grad():
            vecs = model.baskets.vectors(cat_idx18, cat_cnt18, torch.arange(n_cat18, device=device))
            for s in range(0, n_val, 512):
                sl = slice(s, min(s + 512, n_val))
                batch = {k: v[sl] for k, v in {"champs": cds.champs, "sides": cds.sides, "items": cds.items,
                                                "query": cds.query, "inv": cds.inv, "hist": cds.hist}.items()}
                sc = model.scores(model.state(batch, device), vecs)
                order = torch.topk(group_logits(sc, gid, n_groups).cpu(), min(v1.SHORTLIST, n_groups), dim=1).indices
                S = type("S", (), {})()
                S.dragon, S.purchase_sequence = dragon, purchase_sequence
                records.extend(decode_batch(order, val_rows[sl], group_main, S))
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", records)
        rows_by_arm[arm] = records
        print(f"eval {arm}: exact {p2.aggregate(records)['first_action_exact']['value']:.4f}", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    finish(report, report_path, rows_by_arm, ref_files, n_val, started)


def finish(report, report_path, rows_by_arm, ref_files, n_val, started):
    """Paired comparisons against the digest-bound references; every pair needs BOTH sides present."""
    refs = {name: load_rows(path, n_val) for name, (path, _) in ref_files.items()}
    all_arms = {**rows_by_arm, **refs}
    pairs = [(a, b) for a, b in (("H", "T"), ("H", "R1_new_bf"), ("H", "R2_M2"), ("T", "R1_new_bf"), ("T", "R2_M2"))
             if a in all_arms and b in all_arms]
    compare(all_arms, pairs, report, draw_pairs=[p for p in (("H", "T"), ("H", "R1_new_bf"), ("H", "R2_M2")) if p in pairs])
    report["decision"] = {
        "rule": "history adds value only if H beats T with the 95% CI excluding 0; practical progress only if H beats R1 AND R2 likewise",
        "H_beats_T": beats(report, "H", "T") if ("H", "T") in pairs else None,
        "H_beats_R1": beats(report, "H", "R1_new_bf") if ("H", "R1_new_bf") in pairs else None,
        "H_beats_R2": beats(report, "H", "R2_M2") if ("H", "R2_M2") in pairs else None,
        "T_beats_R1": beats(report, "T", "R1_new_bf") if ("T", "R1_new_bf") in pairs else None}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_report(report_path, report)
    print(json.dumps({"aggregates": {k: v["first_action_exact"]["value"] for k, v in report["aggregates"].items()},
                      "decision": report["decision"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
