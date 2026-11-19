"""Ward-free basket vocabulary experiment (16.18-wardfree-v1).

Protocol: docs/experiment_wardfree_baskets_v1.md. One change only: control
wards (2055, 772043) are stripped from every basket key and target; a
basket left empty maps to SAVE. Training is the unchanged phase-2 Bf
recipe (clean init, exact full-catalogue CE, 16 epochs). Baselines
(parent, control-ft, econ-ft) are re-scored under the SAME ward-free
metric from their persisted enriched rows and compared paired by game.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows and
runs 1 epoch.
"""
import argparse
import hashlib
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
from basket_vocabulary_probe import (basket_entries, basket_key,
                                     membership_digest, split_fit_probe)

OUT_DIR = ROOT / "data" / "experiments" / "16.18-wardfree-v1"
FT_DIR = ROOT / "data" / "experiments" / "16.18-basket-econ-finetune-v1"
PW_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v3-prework"
WARDS = {2055, 772043}
EPOCHS = 16
CHECKPOINTS = (4, 8, 16)


def strip_counter(c):
    return Counter({k: n for k, n in c.items() if k not in WARDS})


def stripped_truth(row):
    """Ward-free truth: None-like empty Counter for no-buy AND ward-only."""
    entries = basket_entries(row.get("label_counts"))
    if not entries or any(i == v1.SAVE_SENTINEL for i, _ in entries):
        return Counter()
    return strip_counter(Counter(dict(entries)))


def wardfree_record(row, options):
    """(kind, Counter) options -> ward-free metric record."""
    truth = stripped_truth(row)
    first_kind, first_basket = options[0]
    pred = (Counter() if first_kind == "save"
            else strip_counter(first_basket or Counter()))
    stripped_opts = [Counter() if k == "save" else strip_counter(c or Counter())
                     for k, c in options]
    rec = {"match_id": row["match_id"], "ts": row["ts"],
           "champion": row.get("champion"), "save_kind": row.get("save_kind"),
           "true_size": sum(truth.values()), "pred_size": sum(pred.values()),
           "pred_first": "SAVE" if not pred else dict(pred),
           "first_action_exact": float(pred == truth),
           "any_of_three_exact": (float(any(o == truth for o in stripped_opts))
                                  if truth else None)}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    epochs = 1 if args.smoke else EPOCHS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
    champ_ids, item_ids, label_counts, _, n_fit_all, n_probe_all = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fit = min(3000, n_fit_all) if args.smoke else n_fit_all
    n_probe = min(1000, n_probe_all) if args.smoke else n_probe_all

    # ---- ward-free catalogue + targets from the FIT stream (stripped) ----
    catalogue = Counter()
    fit_truths = []
    for i, row in enumerate(v1.stream_split(train_path, fit_set, True,
                                            champ_index, dragon,
                                            n_fit if args.smoke else None)):
        t = stripped_truth(row)
        key = basket_key(sorted(t.items())) if t else None
        fit_truths.append(key)
        if key:
            catalogue[key] += 1
    assert len(fit_truths) == n_fit
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
    n_cat = len(baskets) + 1
    fit_targets = torch.tensor(
        [0 if k is None else basket_pos[k] for k in fit_truths],
        dtype=torch.long).to(device)
    print(f"catalogo ward-free: {len(baskets)} cestas (antes ~7937); "
          f"targets SAVE {int((fit_targets == 0).sum())}/{n_fit}", flush=True)

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index,
                                       dragon)
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    if args.smoke:
        for ds, n in ((fit_ds, n_fit), (probe_ds, n_probe)):
            for name in ("champs", "sides", "items", "query", "inv", "hist"):
                setattr(ds, name, getattr(ds, name)[:n])
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, n_probe if args.smoke else None))
    assert len(probe_rows) == n_probe

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

    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(n_fit, generator=g)[:10000].to(device)
    torch.manual_seed(16)
    model = v1.BasketNet(len(champ_index), len(item_index), len(label_index),
                         len(item_index) + 1, len(cat_items)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    nll = nn.CrossEntropyLoss()
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "recipe": "phase2-Bf-full-CE", "epochs": epochs,
              "n_catalogue": len(baskets), "curves": [], "aggregates": {},
              "paired": {}, "slices": {}}

    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.monotonic()
        total, n = 0.0, 0
        order = torch.randperm(n_fit, device=device)
        for s in range(0, n_fit, p2.EFFECTIVE):
            idxs_all = order[s:s + p2.EFFECTIVE]
            opt.zero_grad()
            for m0 in range(0, len(idxs_all), p2.MICRO):
                idxs = idxs_all[m0:m0 + p2.MICRO]
                batch = {k: v[idxs.to(v.device)] for k, v in {
                    "champs": fit_ds.champs, "sides": fit_ds.sides,
                    "items": fit_ds.items, "query": fit_ds.query,
                    "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                state = model.state(batch, device)
                vecs = model.baskets.vectors(
                    cat_idx, cat_cnt, torch.arange(n_cat, device=device))
                loss = nll(model.scores(state, vecs), fit_targets[idxs])
                (loss * len(idxs) / len(idxs_all)).backward()
                total += float(loss.detach()) * len(idxs)
            opt.step()
            n += len(idxs_all)
        sched.step()
        fixed = p2.full_ce_on_subset(model, fit_ds, fit_targets, subset_idx,
                                     cat_idx, cat_cnt, device)
        report["curves"].append({"epoch": epoch, "train_loss": total / max(n, 1),
                                 "fixed_subset_full_ce": fixed,
                                 "seconds": round(time.monotonic() - t_ep, 1)})
        print(f"  wardfree epoch {epoch}/{epochs} loss {total / max(n, 1):.3f} "
              f"fixedCE {fixed:.3f} ({report['curves'][-1]['seconds']}s)",
              flush=True)
        if epoch in CHECKPOINTS or epoch == epochs:
            p2.save_checkpoint(OUT_DIR / f"wardfree_ep{epoch}.pt", model, opt,
                               sched, epoch)

    # ------------------------------ eval ------------------------------
    model.eval()
    records = []
    with torch.no_grad():
        vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                     torch.arange(n_cat, device=device))
        for s in range(0, n_probe, tp.BATCH):
            sl = slice(s, min(s + tp.BATCH, n_probe))
            batch = {k: v[sl] for k, v in {
                "champs": probe_ds.champs, "sides": probe_ds.sides,
                "items": probe_ds.items, "query": probe_ds.query,
                "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
            state = model.state(batch, device)
            scores = model.scores(state, vecs)
            top = torch.topk(scores, min(v1.SHORTLIST, n_cat),
                             dim=1).indices.cpu()
            for j in range(top.shape[0]):
                row = probe_rows[s + j]
                budget = float(row["gold_est"])
                options = []
                for cand in top[j].tolist():
                    if cand == 0:
                        options.append(("save", None))
                    else:
                        action = p2.to_action(baskets[cand - 1])
                        if purchase_sequence(action, row, budget,
                                             dragon) is not None:
                            options.append(("buy", Counter(action)))
                    if len(options) >= 3:
                        break
                if not options:
                    options = [("save", None)]
                records.append(wardfree_record(row, options))
    p2.dump(OUT_DIR / f"rows_wardfree_{stamp}.jsonl", records)
    report["aggregates"]["wardfree"] = p2.aggregate(records)

    # ---- baselines re-scored under the SAME ward-free metric ----
    def enriched_to_records(path):
        recs = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if args.smoke and i >= n_probe:
                    break
                r = json.loads(line)
                row = probe_rows[i]
                assert (row["match_id"], row["ts"]) == (r["match_id"], r["ts"])
                options = []
                for o in r["options"]:
                    if o["kind"] == "save":
                        options.append(("save", None))
                    else:
                        options.append(("buy", Counter(
                            {int(k): int(v_) for k, v_ in
                             o["item_counts"].items()})))
                recs.append(wardfree_record(row, options or [("save", None)]))
        return recs

    baselines = {
        "parent": PW_DIR / "rows_Bf_ep16_full_enriched.jsonl",
        "control_ft": FT_DIR / "rows_control_ft_full_enriched.jsonl",
        "econ_ft": FT_DIR / "rows_econ_ft_full_enriched.jsonl",
    }
    rows_all = {"wardfree": records}
    for name, path in baselines.items():
        if not path.exists():
            print(f"baseline {name}: {path.name} ausente, salto", flush=True)
            continue
        rows_all[name] = enriched_to_records(path)
        report["aggregates"][name] = p2.aggregate(rows_all[name])

    for name in ("econ_ft", "parent", "control_ft"):
        if name in rows_all:
            report["paired"][f"wardfree_vs_{name}"] = p2.paired(
                records, rows_all[name], "first_action_exact")
            report["paired"][f"wardfree_vs_{name}_any3"] = p2.paired(
                records, rows_all[name], "any_of_three_exact")
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    if "econ_ft" in rows_all:
        for label, pred in subsets.items():
            report["paired"][f"wardfree_vs_econ_ft_{label}"] = p2.paired(
                [r for r in records if pred(r)],
                [r for r in rows_all["econ_ft"] if pred(r)],
                "first_action_exact")

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"wardfree_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report["paired"].items()
                      if not k.endswith("any3")}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
