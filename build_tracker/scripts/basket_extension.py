"""Pre-registered training extension (16.18-whole-basket-ext1).

Motivation (measured): every generalization curve was still rising at the
epoch-16 cap for BOTH branches. This extends each branch by a CLOSED budget
of 8 epochs from its full-state ep16 checkpoint, keeping its own objective
(Bs sampled / Bf full-CE microbatched). Pre-declared before launch:

- New LR segment stated explicitly (the parent cosine ended at ~0):
  fresh Adam lr=5e-4, cosine T_max=8. Optimizers re-initialized identically.
- Checkpoints at +4/+8; THE result is the final (+8) checkpoint — no
  retrospective best-picking. No further auto-extension.
- Evaluation: common evaluator on the probe; paired (by game, 400) vs the
  ep16 parent rows and vs arm A rows persisted in phase 2.
- The phase-2 verdict stands on its own; this is a new bounded phase.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps probe rows
and runs 1 epoch (plumbing only).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import torch.nn as nn

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe

V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
OUT_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-ext1"
LR = 5e-4
EXT_EPOCHS = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(16)

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

    from app.ddragon import dragon_for_patch
    from app.shop_econ import components_of
    dragon = dragon_for_patch(manifest_gen["patch"])
    fam_cache = {}

    def family_of(t):
        if t not in fam_cache:
            fam_cache[t] = set(components_of(t, dragon)) | {t}
        return fam_cache[t]

    train_path = generation / "visits_train.jsonl"
    cap_probe = 2000 if args.smoke else None
    champ_ids, item_ids, label_counts, catalogue, n_fit, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index,
                                       dragon)
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    fit_targets = torch.load(v1_dir / "cache_targets_full_v2.pt",
                             weights_only=True).to(device)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, cap_probe))
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    cat_items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    n_cat = len(baskets) + 1
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
    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(n_fit, generator=g)[:10000].to(device)
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    epochs = 1 if args.smoke else EXT_EPOCHS
    nll = nn.CrossEntropyLoss()
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "lr": LR, "epochs": epochs, "curves": {},
              "aggregates": {}, "paired": {}}

    for branch, kind in (("Bs", "sampled"), ("Bf", "full")):
        blob = torch.load(V2_DIR / f"{branch}_ep16.pt", map_location="cpu",
                          weights_only=False)
        model = v1.BasketNet(*arch_args).to(device)
        model.load_state_dict(blob["state_dict"])
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        curve = []
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            total, n = 0.0, 0
            order = torch.randperm(n_fit, device=device)
            for s in range(0, n_fit, p2.EFFECTIVE):
                idxs_all = order[s:s + p2.EFFECTIVE]
                opt.zero_grad()
                if kind == "sampled":
                    batch = {k: v[idxs_all.to(v.device)] for k, v in {
                        "champs": fit_ds.champs, "sides": fit_ds.sides,
                        "items": fit_ds.items, "query": fit_ds.query,
                        "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                    state = model.state(batch, device)
                    targets = fit_targets[idxs_all]
                    negs = torch.randint(1, n_cat, (2048,), device=device)
                    cand = torch.cat([targets, torch.zeros(
                        1, dtype=torch.long, device=device), negs])
                    uniq, inv_pos = torch.unique(cand, return_inverse=True)
                    vecs = model.baskets.vectors(cat_idx, cat_cnt, uniq)
                    loss = nll(model.scores(state, vecs),
                               inv_pos[:len(targets)])
                    loss.backward()
                    total += float(loss.detach()) * len(idxs_all)
                else:
                    for m0 in range(0, len(idxs_all), p2.MICRO):
                        idxs = idxs_all[m0:m0 + p2.MICRO]
                        batch = {k: v[idxs.to(v.device)] for k, v in {
                            "champs": fit_ds.champs, "sides": fit_ds.sides,
                            "items": fit_ds.items, "query": fit_ds.query,
                            "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                        state = model.state(batch, device)
                        vecs = model.baskets.vectors(
                            cat_idx, cat_cnt,
                            torch.arange(n_cat, device=device))
                        loss = nll(model.scores(state, vecs),
                                   fit_targets[idxs])
                        (loss * len(idxs) / len(idxs_all)).backward()
                        total += float(loss.detach()) * len(idxs)
                opt.step()
                n += len(idxs_all)
            sched.step()
            fixed = p2.full_ce_on_subset(model, fit_ds, fit_targets,
                                         subset_idx, cat_idx, cat_cnt, device)
            curve.append({"epoch": 16 + epoch, "train_loss": total / max(n, 1),
                          "fixed_subset_full_ce": fixed,
                          "seconds": round(time.monotonic() - t_ep, 1)})
            print(f"  {branch}+{epoch}/{epochs} loss {total / max(n, 1):.3f} "
                  f"fixedCE {fixed:.3f} ({curve[-1]['seconds']}s)", flush=True)
            if epoch in (4, epochs):
                p2.save_checkpoint(OUT_DIR / f"{branch}_ext_ep{16 + epoch}.pt",
                                   model, opt, sched, 16 + epoch)
        report["curves"][branch] = curve
        recs = p2.eval_basket_model(f"{branch}_ext", model, probe_ds,
                                    probe_rows, baskets, basket_pos, cat_idx,
                                    cat_cnt, dragon, device, family_of)
        p2.dump(OUT_DIR / f"rows_{branch}ext_{stamp}.jsonl", recs)
        report["aggregates"][branch] = p2.aggregate(recs)
        if not args.smoke:
            parent = [json.loads(l) for l in open(
                V2_DIR / f"rows_{branch}_full.jsonl", encoding="utf-8")]
            arm_a = [json.loads(l) for l in open(
                V2_DIR / "rows_A_full.jsonl", encoding="utf-8")]
            report["paired"][f"{branch}ext_vs_{branch}16"] = p2.paired(
                recs, parent, "first_action_exact")
            report["paired"][f"{branch}ext_vs_A"] = p2.paired(
                recs, arm_a, "first_action_exact")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    (OUT_DIR / f"extension_report_{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("paired", {}), indent=1), flush=True)
    print(f"report: {OUT_DIR / f'extension_report_{stamp}.json'}", flush=True)


if __name__ == "__main__":
    main()
