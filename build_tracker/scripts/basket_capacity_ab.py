"""Capacity contrast d64 vs d128 (16.18-capacity-d128-v1).

Protocol: docs/experiment_capacity_d128_v1.md. Two clean-init arms, ONLY
d_model differs (passed explicitly; BasketNet.state_proj adapts via the
module global during construction only). Same frozen caches (POV +
corrected wallet contract), full catalogue WITH wards, exact full-CE
(micro 32 -> effective 256), Adam 1e-3 cosine-16, seed 16, batch order
from an independent generator shared by both arms. Grouped ward-
marginalized decoder for evaluation; ward-free metric; identity
(match_id, ts, champion) with uniqueness verification.

Gates before training: real forward+backward pilot (memory + projected
duration; abort > 7.0 h) and >= 40 GB free disk.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] tiny caps.
"""
import argparse
import hashlib
import json
import os
import shutil
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
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_wardfree import WARDS, strip_counter, wardfree_record

OUT_DIR = ROOT / "data" / "experiments" / "16.18-capacity-d128-v1"
EPOCHS = 16
CHECKPOINTS = (4, 8, 16)
PILOT_BATCHES = 3
MAX_PROJECTED_H = 7.0
MIN_FREE_GB = 40


def make_model(arch_args, d_model, device):
    """BasketNet whose trunk uses an explicit d_model.

    PrefixModel takes d_model as a constructor kwarg; BasketNet reads
    tp.D_MODEL only inside __init__ (state_proj input = 2*d_model), so the
    global is swapped during construction ONLY and restored after. Runtime
    forward paths read no architecture globals (verified in protocol).
    """
    orig_global, orig_cls = tp.D_MODEL, tp.PrefixModel

    def pinned(*a, **k):
        return orig_cls(*a, d_model=d_model, **k)

    tp.D_MODEL = d_model
    tp.PrefixModel = pinned
    try:
        model = v1.BasketNet(*arch_args).to(device)
    finally:
        tp.D_MODEL = orig_global
        tp.PrefixModel = orig_cls
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    epochs = 1 if args.smoke else EPOCHS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = OUT_DIR / f"progress_{stamp}.jsonl"

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "epochs": epochs, "free_disk_gb": round(free_gb, 1),
              "arms": {"A": {"d_model": 64}, "B": {"d_model": 128}},
              "pilot": {}, "curves": {}, "aggregates": {}, "paired": {},
              "net_vs": {}, "counters": {}, "latency": {}}

    def bail(reason):
        report["aborted"] = reason
        (OUT_DIR / f"capacity_report_{stamp}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"ABORTADO: {reason}", flush=True)
        sys.exit(0)

    if free_gb < MIN_FREE_GB and not args.smoke:
        bail(f"free disk {free_gb:.1f} GB < declared {MIN_FREE_GB} GB reserve")

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(
        encoding="utf-8"))["input_schema"]
    tp.MAX_ITEMS = schema["max_modeled_inventory_items"]
    tp.MAX_COPIES = schema["max_copies_of_one_item"]
    tp.MAX_BASKET_ITEMS = schema["max_basket_items"]
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    plan = json.loads((generation / "split_plan.json").read_text(encoding="utf-8"))
    fit_games, probe_games, _ = split_fit_probe(plan)
    assert membership_digest(fit_games) == v1.FROZEN_FIT_SHA
    assert membership_digest(probe_games) == v1.FROZEN_PROBE_SHA
    fit_set = set(fit_games)
    report["fit_sha"] = v1.FROZEN_FIT_SHA
    report["probe_sha"] = v1.FROZEN_PROBE_SHA

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    from app.shop_econ import components_of
    dragon = dragon_for_patch(manifest_gen["patch"])
    fam_cache = {}

    def family_of(t):
        if t not in fam_cache:
            fam_cache[t] = set(components_of(t, dragon)) | {t}
        return fam_cache[t]

    train_path = generation / "visits_train.jsonl"
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, n_probe_all = \
        v1.first_pass(train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fit = min(800, n_fit_all) if args.smoke else n_fit_all
    n_probe = min(300, n_probe_all) if args.smoke else n_probe_all

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index,
                                       dragon)
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    fit_targets = torch.load(v1_dir / "cache_targets_full_v2.pt",
                             weights_only=True)
    if args.smoke:
        for ds, n in ((fit_ds, n_fit), (probe_ds, n_probe)):
            for name in ("champs", "sides", "items", "query", "inv", "hist"):
                setattr(ds, name, getattr(ds, name)[:n])
        fit_targets = fit_targets[:n_fit]
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    fit_targets = fit_targets.to(device)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, n_probe if args.smoke else None))
    assert len(probe_rows) == n_probe
    keys = [(r["match_id"], r["ts"], r.get("champion")) for r in probe_rows]
    assert len(set(keys)) == len(keys), "probe identity not unique"

    cat_items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    cat_idx = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    cat_cnt = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, n_ = part.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(n_), v1.MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)
    arch_args = (len(champ_index), len(item_index), len(label_index),
                 len(item_index) + 1, len(cat_items))

    # ward-group map (identical rule to the frozen dev baseline decoder)
    group_of_key = {None: 0}
    group_main = [None]
    group_ids = torch.zeros(n_cat, dtype=torch.long)
    for cid, bkey in enumerate(baskets, start=1):
        stripped = strip_counter(Counter(
            {int(part.split(":")[0]): int(part.split(":")[1])
             for part in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(stripped)
        group_ids[cid] = group_of_key[skey]
    n_groups = len(group_main)
    gid = group_ids.to(device)
    nll = nn.CrossEntropyLoss()
    print(f"setup listo: fit {n_fit} probe {n_probe} catalogo {len(baskets)} "
          f"grupos {n_groups} disco libre {free_gb:.0f}GB "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    # ---------------- pilot: real forward+backward, both arms ----------------
    def micro_step(model, idxs):
        batch = {k: v[idxs.to(v.device)] for k, v in {
            "champs": fit_ds.champs, "sides": fit_ds.sides,
            "items": fit_ds.items, "query": fit_ds.query,
            "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
        state = model.state(batch, device)
        vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                     torch.arange(n_cat, device=device))
        return nll(model.scores(state, vecs), fit_targets[idxs.to(device)])

    n_eff = max(1, n_fit // p2.EFFECTIVE)
    projected_total_h = 0.0
    for arm, dm in (("A", 64), ("B", 128)):
        torch.manual_seed(16)
        model = make_model(arch_args, dm, device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        g_pilot = torch.Generator().manual_seed(99)
        t0 = time.monotonic()
        for _ in range(PILOT_BATCHES):
            idxs_all = torch.randperm(n_fit, generator=g_pilot)[:p2.EFFECTIVE]
            opt.zero_grad()
            for m0 in range(0, len(idxs_all), p2.MICRO):
                loss = micro_step(model, idxs_all[m0:m0 + p2.MICRO])
                (loss * p2.MICRO / len(idxs_all)).backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_eff = (time.monotonic() - t0) / PILOT_BATCHES
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda" else 0.0)
        arm_h = per_eff * n_eff * epochs / 3600
        projected_total_h += arm_h
        report["pilot"][arm] = {"sec_per_effective_batch": round(per_eff, 2),
                                "peak_gpu_gb": round(peak, 2),
                                "projected_train_hours": round(arm_h, 2)}
        n_params = sum(p_.numel() for p_ in model.parameters())
        report["arms"][arm]["n_params"] = n_params
        print(f"pilot {arm} (d{dm}): {per_eff:.2f}s/eff-batch, "
              f"peak {peak:.2f}GB, proyeccion {arm_h:.2f}h, "
              f"params {n_params:,}", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["pilot"]["projected_total_hours"] = round(projected_total_h, 2)
    if projected_total_h > MAX_PROJECTED_H and not args.smoke:
        bail(f"pilot projects {projected_total_h:.2f} h > {MAX_PROJECTED_H} h")

    # ---------------- train + eval both arms ----------------
    rows_by_arm = {}
    for arm, dm in (("A", 64), ("B", 128)):
        torch.manual_seed(16)                       # model randomness
        model = make_model(arch_args, dm, device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        order_gen = torch.Generator().manual_seed(16)   # independent, shared
        curve = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            order = torch.randperm(n_fit, generator=order_gen)
            total, n = 0.0, 0
            for s in range(0, n_fit, p2.EFFECTIVE):
                idxs_all = order[s:s + p2.EFFECTIVE]
                opt.zero_grad()
                for m0 in range(0, len(idxs_all), p2.MICRO):
                    idxs = idxs_all[m0:m0 + p2.MICRO]
                    loss = micro_step(model, idxs)
                    (loss * len(idxs) / len(idxs_all)).backward()
                    total += float(loss.detach()) * len(idxs)
                opt.step()
                n += len(idxs_all)
            sched.step()
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if device.type == "cuda" else 0.0)
            entry = {"arm": arm, "epoch": epoch,
                     "train_loss": total / max(n, 1),
                     "seconds": round(time.monotonic() - t_ep, 1),
                     "peak_gpu_gb": round(peak, 2)}
            curve.append(entry)
            with open(progress, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"  {arm}(d{dm}) epoch {epoch}/{epochs} "
                  f"loss {entry['train_loss']:.3f} ({entry['seconds']}s, "
                  f"peak {entry['peak_gpu_gb']}GB)", flush=True)
            if epoch in CHECKPOINTS or epoch == epochs:
                p2.save_checkpoint(OUT_DIR / f"{arm}_ep{epoch}.pt", model,
                                   opt, sched, epoch)
        report["curves"][arm] = curve

        # eval: grouped decoder, ward-free records (+family), counters
        model.eval()
        records = []
        illegal_walked = 0
        t_eval = time.monotonic()
        with torch.no_grad():
            vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                         torch.arange(n_cat, device=device))
            for s in range(0, n_probe, 512):
                sl = slice(s, min(s + 512, n_probe))
                batch = {k: v[sl] for k, v in {
                    "champs": probe_ds.champs, "sides": probe_ds.sides,
                    "items": probe_ds.items, "query": probe_ds.query,
                    "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
                state = model.state(batch, device)
                scores = model.scores(state, vecs)
                b = scores.shape[0]
                gidx = gid.unsqueeze(0).expand(b, -1)
                gmax = torch.full((b, n_groups), float("-inf"), device=device)
                gmax.scatter_reduce_(1, gidx, scores, reduce="amax",
                                     include_self=True)
                gsum = torch.zeros(b, n_groups, device=device)
                gsum.scatter_add_(1, gidx,
                                  (scores - gmax.gather(1, gidx)).exp())
                grouped = (gmax + gsum.log()).cpu()
                top = torch.topk(grouped, min(v1.SHORTLIST, n_groups),
                                 dim=1).indices
                for j in range(b):
                    row = probe_rows[s + j]
                    budget = float(row["gold_est"])
                    options = []
                    for grp in top[j].tolist():
                        if grp == 0:
                            options.append(("save", None))
                        else:
                            main = group_main[grp]
                            action = tuple(sorted(main.elements()))
                            if purchase_sequence(action, row, budget,
                                                 dragon) is not None:
                                options.append(("buy", Counter(main)))
                            else:
                                illegal_walked += 1
                        if len(options) >= 3:
                            break
                    fallback = not options
                    if fallback:
                        options = [("save", None)]
                    rec = wardfree_record(row, options)
                    rec["fallback_save"] = fallback
                    truth = rec["true_size"] > 0
                    target = int(row.get("target_id") or 0)
                    if truth and target and target not in WARDS:
                        pred = (rec["pred_first"]
                                if isinstance(rec["pred_first"], dict) else {})
                        fam = family_of(target)
                        rec["action_family_match"] = float(
                            bool(pred) and any(int(i) in fam for i in pred))
                    records.append(rec)
        eval_s = time.monotonic() - t_eval
        rkeys = [(r["match_id"], r["ts"], r["champion"]) for r in records]
        assert len(set(rkeys)) == len(rkeys) == n_probe
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", records)
        rows_by_arm[arm] = records
        report["aggregates"][arm] = p2.aggregate(records)
        report["counters"][arm] = {"illegal_walked": illegal_walked}
        report["latency"][arm] = {
            "eval_seconds": round(eval_s, 1),
            "ms_per_decision_decode": round(eval_s / n_probe * 1e3, 3)}
        print(f"eval {arm}: exact "
              f"{report['aggregates'][arm]['first_action_exact']['value']:.4f}"
              f" ({eval_s:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------- comparisons ----------------
    def key_of(r):
        return (r["match_id"], r["ts"], r["champion"])

    report["paired"]["B_vs_A"] = p2.paired(rows_by_arm["B"], rows_by_arm["A"],
                                           "first_action_exact")
    report["paired"]["B_vs_A_any3"] = p2.paired(
        rows_by_arm["B"], rows_by_arm["A"], "any_of_three_exact")
    report["paired"]["B_vs_A_family"] = p2.paired(
        rows_by_arm["B"], rows_by_arm["A"], "action_family_match")
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    for label, pred in subsets.items():
        report["paired"][f"B_vs_A_{label}"] = p2.paired(
            [r for r in rows_by_arm["B"] if pred(r)],
            [r for r in rows_by_arm["A"] if pred(r)],
            "first_action_exact")
    a_by_key = {key_of(r): r for r in rows_by_arm["A"]}
    cor = bro = 0
    for r in rows_by_arm["B"]:
        p_ = a_by_key.get(key_of(r))
        if p_ is None:
            continue
        a, b = p_["first_action_exact"], r["first_action_exact"]
        cor += a < b
        bro += a > b
    report["net_vs"]["B_vs_A"] = {"corrected": cor, "broken": bro,
                                  "net": cor - bro}
    baseline_path = (ROOT / "data" / "experiments" / "16.18-wardfree-v1"
                     / "rows_grouped_econft_full.jsonl")
    if baseline_path.exists() and not args.smoke:
        base_rows = [json.loads(l) for l in open(baseline_path,
                                                 encoding="utf-8")]
        for arm in ("A", "B"):
            report["paired"][f"{arm}_vs_econ_baseline"] = p2.paired(
                rows_by_arm[arm], base_rows, "first_action_exact")

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"capacity_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired"], indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
