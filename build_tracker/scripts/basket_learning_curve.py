"""Learning curve on the existing FIT games: 500 / 1,000 / (2,000 reused).

Protocol: docs/experiment_learning_curve_v1.md. Same phase-2 Bf recipe as
the capacity contrast (clean d64, full-catalogue CE, 16 epochs); the only
thing that changes between arms is HOW MANY games the rows come from.
Subsets are nested by game from one seeded shuffle. The 2,000-game point is
arm A of 16.18-capacity-d128-v1 (identical recipe and script family) and
its persisted probe rows are reused, digest recorded.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows and
runs 1 epoch.
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
from basket_wardfree import strip_counter, wardfree_record

OUT_DIR = ROOT / "data" / "experiments" / "16.18-learning-curve-v1"
REF_ROWS = ROOT / "data" / "experiments" / "16.18-capacity-d128-v1" / "rows_A_full.jsonl"
ARMS = (("g500", 500), ("g1000", 1000))
EPOCHS = 16
MAX_PROJECTED_H = 3.0
MIN_FREE_DISK_GB = 40
PILOT_BATCHES = 6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    epochs = 1 if args.smoke else EPOCHS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = OUT_DIR / f"progress_{stamp}.jsonl"
    assert tp.D_MODEL == 64, "la receta es d64"

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
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, n_probe_all = \
        v1.first_pass(train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group_of_key = {None: 0}
    group_main = [None]
    group_ids = torch.zeros(n_cat, dtype=torch.long)
    for cid, bkey in enumerate(baskets, start=1):
        stripped = strip_counter(Counter(
            {int(p.split(":")[0]): int(p.split(":")[1]) for p in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(stripped)
        group_ids[cid] = group_of_key[skey]
    n_groups = len(group_main)
    gid = group_ids.to(device)

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index, dragon)
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index, dragon)
    fit_targets = torch.load(v1_dir / "cache_targets_full_v2.pt", weights_only=True)
    n_fit, n_probe = fit_ds.champs.shape[0], probe_ds.champs.shape[0]
    assert n_fit == n_fit_all and n_probe == n_probe_all
    # row -> game, in cache order (both come from the same stream)
    fit_match = [r["match_id"] for r in
                 v1.stream_split(train_path, fit_set, True, champ_index, dragon, None)]
    assert len(fit_match) == n_fit
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, None))
    assert len(probe_rows) == n_probe
    if args.smoke:
        n_probe = 1000
        probe_rows = probe_rows[:n_probe]
        for name in ("champs", "sides", "items", "query", "inv", "hist"):
            setattr(probe_ds, name, getattr(probe_ds, name)[:n_probe])
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    fit_targets = fit_targets.to(device)

    # nested subsets by game from ONE seeded shuffle
    games = sorted(fit_set)
    perm = torch.randperm(len(games), generator=torch.Generator().manual_seed(16)).tolist()
    subsets = {}
    for arm, n_games in ARMS:
        chosen = {games[i] for i in perm[:n_games]}
        idx = torch.tensor([i for i, m in enumerate(fit_match) if m in chosen],
                           dtype=torch.long)
        if args.smoke:
            idx = idx[:600]
        subsets[arm] = {"games": sorted(chosen), "rows": idx}
    assert subsets["g500"]["games"] == sorted(set(subsets["g500"]["games"])
                                              & set(subsets["g1000"]["games"]))

    cat_items = sorted({i for b in baskets for p in b.split(",")
                        for i in [int(p.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    cat_idx = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    cat_cnt = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, p in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, n_ = p.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(n_), v1.MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)
    arch_args = (len(champ_index), len(item_index), len(label_index),
                 len(item_index) + 1, len(cat_items))
    nll = nn.CrossEntropyLoss()
    print(f"setup listo ({time.monotonic() - started:.0f}s): "
          + ", ".join(f"{a}={len(s['rows'])} filas/{len(s['games'])} partidas"
                      for a, s in subsets.items()), flush=True)

    def micro_step(model, idxs):
        batch = {k: v[idxs.to(v.device)] for k, v in {
            "champs": fit_ds.champs, "sides": fit_ds.sides,
            "items": fit_ds.items, "query": fit_ds.query,
            "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
        state = model.state(batch, device)
        vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                     torch.arange(n_cat, device=device))
        return nll(model.scores(state, vecs), fit_targets[idxs.to(device)])

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/experiment_learning_curve_v1.md",
              "scope": "internal chronological TRAIN probe; no VAL, no test",
              "mode": stamp, "epochs": epochs, "recipe": "phase2-Bf d64",
              "subsets": {a: {"games": len(s["games"]), "rows": int(len(s["rows"])),
                              "games_sha256": membership_digest(s["games"])}
                          for a, s in subsets.items()},
              "pilot": {}, "curves": {}, "aggregates": {}, "paired": {}}

    # pilot gate: project the whole run from a few real steps of the largest arm
    torch.manual_seed(16)
    model = v1.BasketNet(*arch_args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    big = subsets["g1000"]["rows"]
    g_pilot = torch.Generator().manual_seed(99)
    t0 = time.monotonic()
    for _ in range(PILOT_BATCHES):
        idxs_all = big[torch.randperm(len(big), generator=g_pilot)[:p2.EFFECTIVE]]
        opt.zero_grad()
        for m0 in range(0, len(idxs_all), p2.MICRO):
            loss = micro_step(model, idxs_all[m0:m0 + p2.MICRO])
            (loss * p2.MICRO / len(idxs_all)).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_eff = (time.monotonic() - t0) / PILOT_BATCHES
    projected = sum(per_eff * (len(s["rows"]) / p2.EFFECTIVE) * epochs
                    for s in subsets.values()) / 3600
    report["pilot"] = {"sec_per_effective_batch": round(per_eff, 2),
                       "projected_train_hours": round(projected, 2)}
    print(f"pilot: {per_eff:.2f}s/eff-batch, proyeccion {projected:.2f}h", flush=True)
    if projected > MAX_PROJECTED_H and not args.smoke:
        raise RuntimeError(f"pilot projects {projected:.2f}h > {MAX_PROJECTED_H}h")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows_by_arm = {}
    for arm, _n in ARMS:
        idx = subsets[arm]["rows"]
        torch.manual_seed(16)
        model = v1.BasketNet(*arch_args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        order_gen = torch.Generator().manual_seed(16)
        curve = []
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            order = idx[torch.randperm(len(idx), generator=order_gen)]
            total, n = 0.0, 0
            for s in range(0, len(order), p2.EFFECTIVE):
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
            entry = {"arm": arm, "epoch": epoch, "train_loss": total / max(n, 1),
                     "seconds": round(time.monotonic() - t_ep, 1),
                     "free_disk_gb": round(shutil.disk_usage(OUT_DIR).free / 1e9, 1),
                     "utc": datetime.now(timezone.utc).isoformat()}
            curve.append(entry)
            with open(progress, "a", encoding="utf-8", newline=chr(10)) as fh:
                fh.write(json.dumps(entry) + chr(10))
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  {arm} ep{epoch}/{epochs} loss {entry['train_loss']:.4f} "
                  f"({entry['seconds']}s)", flush=True)
            if not args.smoke and entry["free_disk_gb"] < MIN_FREE_DISK_GB:
                raise RuntimeError("disco libre insuficiente")
        report["curves"][arm] = curve
        p2.save_checkpoint(OUT_DIR / f"{arm}_ep{epochs}.pt", model, opt, sched, epochs)

        model.eval()
        records = []
        with torch.no_grad():
            vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                         torch.arange(n_cat, device=device))
            for s in range(0, n_probe, 512):
                sl = slice(s, min(s + 512, n_probe))
                batch = {k: v[sl] for k, v in {
                    "champs": probe_ds.champs, "sides": probe_ds.sides,
                    "items": probe_ds.items, "query": probe_ds.query,
                    "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
                scores = model.scores(model.state(batch, device), vecs)
                b = scores.shape[0]
                gidx = gid.unsqueeze(0).expand(b, -1)
                gmax = torch.full((b, n_groups), float("-inf"), device=device)
                gmax.scatter_reduce_(1, gidx, scores, reduce="amax", include_self=True)
                gsum = torch.zeros(b, n_groups, device=device)
                gsum.scatter_add_(1, gidx, (scores - gmax.gather(1, gidx)).exp())
                top = torch.topk((gmax + gsum.log()).cpu(),
                                 min(v1.SHORTLIST, n_groups), dim=1).indices
                for j in range(b):
                    row = probe_rows[s + j]
                    budget = float(row["gold_est"])
                    options = []
                    for grp in top[j].tolist():
                        if grp == 0:
                            options.append(("save", None))
                        else:
                            main = group_main[grp]
                            if purchase_sequence(tuple(sorted(main.elements())),
                                                 row, budget, dragon) is not None:
                                options.append(("buy", Counter(main)))
                        if len(options) >= 3:
                            break
                    records.append(wardfree_record(row, options or [("save", None)]))
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", records)
        rows_by_arm[arm] = records
        report["aggregates"][arm] = p2.aggregate(records)
        print(f"eval {arm}: exact "
              f"{report['aggregates'][arm]['first_action_exact']['value']:.4f}", flush=True)
        del model, opt, sched
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # the 2,000-game point: reused, digest recorded, identity asserted
    ref = [json.loads(l) for l in open(REF_ROWS, encoding="utf-8")][:n_probe]
    ident = [(r["match_id"], r["ts"], r["champion"]) for r in rows_by_arm["g500"]]
    assert [(r["match_id"], r["ts"], r["champion"]) for r in ref] == ident
    rows_by_arm["g2000_reused"] = ref
    report["aggregates"]["g2000_reused"] = p2.aggregate(ref)
    report["reused_2000_point"] = {
        "rows": str(REF_ROWS.relative_to(ROOT)),
        "sha256": hashlib.sha256(REF_ROWS.read_bytes()).hexdigest(),
        "source": "16.18-capacity-d128-v1 arm A (clean d64, same recipe)"}

    subsets_m = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
    }
    for x, y in (("g1000", "g500"), ("g2000_reused", "g1000"), ("g2000_reused", "g500")):
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(rows_by_arm[x], rows_by_arm[y], metric)
        for label, pred in subsets_m.items():
            report["paired"][f"{x}_vs_{y}_{label}"] = p2.paired(
                [r for r in rows_by_arm[x] if pred(r)],
                [r for r in rows_by_arm[y] if pred(r)], "first_action_exact")
    g1 = report["paired"]["g1000_vs_g500"]["diff"]
    g2 = report["paired"]["g2000_reused_vs_g1000"]["diff"]
    report["reading"] = {
        "gain_500_to_1000": g1, "gain_1000_to_2000": g2,
        "ratio": (g2 / g1) if g1 else None,
        "rule": "ratio >= 0.5 steep (data is the lever); <= 0.25 flat; between: undecided",
        "verdict": ("steep" if g1 and g2 / g1 >= 0.5 else
                    "flat" if g1 and g2 / g1 <= 0.25 else "undecided"),
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = OUT_DIR / f"learning_curve_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": {k: v["first_action_exact"]["value"]
                                     for k, v in report["aggregates"].items()},
                      "reading": report["reading"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
