"""Stage 1 of the expansion: Bf d64 on the new generation, scored on its cohort.

Protocol: docs/expansion_plan_v1.md (owner GO 2026-09-23, design B).
Recipe fixed in advance: clean d64 BasketNet, full-catalogue CE, micro 32 /
effective 256, Adam 1e-3 cosine T_max=16, 16 epochs, seed 16. The result is
the epoch-16 checkpoint; no other epoch is scored.

On EXACTLY the same cohort games three arms are scored with the ward-free
metric, paired by game: the new Bf (own catalogue), the old Bf_ep16 (the
package's domain parent, v3 catalogue) and the frozen 50/50 mixture (the
package adapter, features computed fresh). The continuation criterion for
the economic + ward-free chain is evaluated and written, never decided by
hand.

Run via footwear_runtime with PREFIX_ML_DIR = the new generation and
--review = the new snapshot's static review; [--smoke] caps rows/epochs.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
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
from basket_wardfree import strip_counter, wardfree_record

EPOCHS = 16
# Pilot: the first effective batches on a fresh CUDA context pay kernel
# loading, cuBLAS/cuDNN set-up and allocator growth. Timing them (6 batches,
# nothing untimed) projected 10.60 h on 2026-09-24 and aborted the chain;
# two untimed warm-up batches precede the timed ones so the projection
# measures steady-state cost. The training itself re-seeds afterwards.
PILOT_WARMUP = 2
PILOT_BATCHES = 12
MIN_FREE_DISK_GB = 40
CONTINUE_MIN_POINTS = 1.0            # declared in the plan before any result


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_map(baskets):
    n_cat = len(baskets) + 1
    group_of_key, group_main = {None: 0}, [Counter()]
    gids = torch.zeros(n_cat, dtype=torch.long)
    for cid, bkey in enumerate(baskets, start=1):
        stripped = strip_counter(Counter(
            {int(p.split(":")[0]): int(p.split(":")[1]) for p in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(stripped)
        gids[cid] = group_of_key[skey]
    return gids, group_main


def basket_tensors(baskets, n, device):
    items = sorted({i for b in baskets for p in b.split(",") for i in [int(p.split(":")[0])]})
    idx = {it: k + 1 for k, it in enumerate(items)}
    ci = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
    cc = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, p in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, c = p.split(":")
            ci[row_i, slot] = idx[int(i)]
            cc[row_i, slot] = min(int(c), v1.MAX_BASKET)
    return ci.to(device), cc.to(device), len(items)


def decode(order, group_main, row, dragon, purchase_sequence):
    budget = float(row["gold_est"])
    options = []
    for act in order:
        if act == 0:
            options.append(("save", None))
        else:
            main = group_main[act]
            main = main if isinstance(main, Counter) else Counter(dict(main))
            if purchase_sequence(tuple(sorted(main.elements())), row, budget, dragon) is not None:
                options.append(("buy", Counter(main)))
        if len(options) >= 3:
            break
    return options or [("save", None)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="mixture-50-50-v1")
    ap.add_argument("--max-projected-hours", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    epochs = 1 if args.smoke else EPOCHS

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    OUT_DIR = ROOT / "data" / "experiments" / f"{generation.name}-bf-stage1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = OUT_DIR / f"progress_{stamp}.jsonl"
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
    qual = json.loads((generation / "qualification.json").read_text(encoding="utf-8"))
    admitted_counts = {
        "source_games": qual.get("source_games"), "quality_accepted_games": qual.get("quality_accepted_games"),
        "quarantined_games": qual.get("quarantined_games"),
        "exclusion_reason_counts": qual.get("exclusion_reason_counts"),
        "source_exclusion_reason_counts": qual.get("source_exclusion_reason_counts"),
        "split_games": {"train": len(train_set), "val_cohort": len(val_set)}}
    print("admitidos:", json.dumps(admitted_counts), flush=True)

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path, val_path = generation / "visits_train.jsonl", generation / "visits_val.jsonl"
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, _ = v1.first_pass(train_path, train_set, None, None)
    champ_index, item_index = tp._index_map(champ_ids), tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(), key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gids, group_main = group_map(baskets)
    gid = gids.to(device)
    n_groups = len(group_main)
    cat_idx, cat_cnt, n_items = basket_tensors(baskets, n_cat, device)
    cap_fit = 3000 if args.smoke else None

    # ---- FIT cache (built once, fingerprinted) + targets ----
    cache_fit = OUT_DIR / f"cache_fit_{stamp}.pt"
    expect = {"history": False, "extras": True, "gold_est": True, "gold_x": True, "runes": True, "pov_gold": True}
    if cache_fit.exists():
        fit_ds = tp.ShopDataset.from_cache(cache_fit, champ_index, item_index, label_index, dragon, expect=expect)
        fit_targets = torch.load(OUT_DIR / f"targets_fit_{stamp}.pt", weights_only=True)
    else:
        n_fit = min(n_fit_all, cap_fit) if cap_fit else n_fit_all
        fit_ds = tp.ShopDataset.from_stream(
            v1.stream_split(train_path, train_set, True, champ_index, dragon, cap_fit),
            n_fit, champ_index, item_index, label_index, dragon)
        fit_targets = torch.zeros(n_fit, dtype=torch.long)
        for i, row in enumerate(v1.stream_split(train_path, train_set, True, champ_index, dragon, cap_fit)):
            t = v1.truth_of(row)
            fit_targets[i] = 0 if t is None else basket_pos[basket_key(sorted(t.items()))]
        fit_ds.save_cache(cache_fit, meta={"generation": generation.name, "split": "fit",
                                           "catalogue_sha256": hashlib.sha256("|".join(baskets).encode()).hexdigest()})
        torch.save(fit_targets, OUT_DIR / f"targets_fit_{stamp}.pt")
    n_fit = fit_ds.champs.shape[0]
    assert fit_targets.shape[0] == n_fit
    if device.type == "cuda":
        fit_ds.to_device(device)
    fit_targets = fit_targets.to(device)

    # ---- cohort rows + tensors (new catalogue featurization) ----
    cap_val = 1500 if args.smoke else None
    val_rows = list(v1.stream_split(val_path, set(), False, champ_index, dragon, cap_val))
    n_val = len(val_rows)
    val_ds = tp.ShopDataset.from_stream(iter(val_rows), n_val, champ_index, item_index, label_index, dragon)
    if device.type == "cuda":
        val_ds.to_device(device)
    cohort_games = sorted({r["match_id"] for r in val_rows})
    assert set(cohort_games) <= val_set
    print(f"setup listo ({time.monotonic() - started:.0f}s): fit {n_fit} filas / {len(train_set)} partidas; "
          f"cohorte {n_val} decisiones / {len(cohort_games)} partidas; catalogo {len(baskets)}, grupos {n_groups}",
          flush=True)

    nll = nn.CrossEntropyLoss()
    arch = (len(champ_index), len(item_index), len(label_index), len(item_index) + 1, n_items)

    def micro_step(model, idxs):
        batch = {k: v[idxs.to(v.device)] for k, v in {
            "champs": fit_ds.champs, "sides": fit_ds.sides, "items": fit_ds.items,
            "query": fit_ds.query, "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
        state = model.state(batch, device)
        vecs = model.baskets.vectors(cat_idx, cat_cnt, torch.arange(n_cat, device=device))
        return nll(model.scores(state, vecs), fit_targets[idxs.to(device)])

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/expansion_plan_v1.md", "generation": generation.name,
              "generation_manifest_sha256": sha256_file(generation / "generation_manifest.json"),
              "mode": stamp, "recipe": "phase2-Bf d64, 16 epochs, ep16 fixed", "epochs": epochs,
              "admitted": admitted_counts, "fit_rows": n_fit, "cohort_rows": n_val,
              "cohort_games": len(cohort_games), "catalogue": len(baskets), "groups": n_groups,
              "pilot": {}, "curve": [], "aggregates": {}, "paired": {}, "slices": {}, "movement": {}}

    # ---- pilot gate ----
    torch.manual_seed(16)
    model = v1.BasketNet(*arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g_pilot = torch.Generator().manual_seed(99)
    t0 = time.monotonic()
    for i in range(PILOT_WARMUP + PILOT_BATCHES):
        if i == PILOT_WARMUP:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.monotonic()
        idxs_all = torch.randperm(n_fit, generator=g_pilot)[:p2.EFFECTIVE]
        opt.zero_grad()
        for m0 in range(0, len(idxs_all), p2.MICRO):
            loss = micro_step(model, idxs_all[m0:m0 + p2.MICRO])
            (loss * p2.MICRO / len(idxs_all)).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_eff = (time.monotonic() - t0) / PILOT_BATCHES
    projected_h = per_eff * (n_fit / p2.EFFECTIVE) * epochs / 3600
    report["pilot"] = {"sec_per_effective_batch": round(per_eff, 3), "projected_train_hours": round(projected_h, 2),
                       "max_projected_hours": args.max_projected_hours,
                       "warmup_batches_untimed": PILOT_WARMUP, "timed_batches": PILOT_BATCHES}
    (OUT_DIR / f"pilot_{stamp}.json").write_text(json.dumps(report["pilot"], indent=2), encoding="utf-8")
    print(f"pilot: {per_eff:.2f}s/eff-batch -> {projected_h:.2f} h proyectadas (tope {args.max_projected_hours})", flush=True)
    if projected_h > args.max_projected_hours and not args.smoke:
        raise RuntimeError(f"pilot projects {projected_h:.2f} h > {args.max_projected_hours} h: abort")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- train (recipe fixed) ----
    torch.manual_seed(16)
    model = v1.BasketNet(*arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    order_gen = torch.Generator().manual_seed(16)
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
        entry = {"epoch": epoch, "train_loss": total / max(n, 1), "seconds": round(time.monotonic() - t_ep, 1),
                 "free_disk_gb": round(shutil.disk_usage(OUT_DIR).free / 1e9, 1),
                 "utc": datetime.now(timezone.utc).isoformat()}
        report["curve"].append(entry)
        with open(progress, "a", encoding="utf-8", newline=chr(10)) as fh:
            fh.write(json.dumps(entry) + chr(10))
            fh.flush()
            os.fsync(fh.fileno())
        print(f"  ep{epoch}/{epochs} loss {entry['train_loss']:.4f} ({entry['seconds']}s)", flush=True)
        if not args.smoke and entry["free_disk_gb"] < MIN_FREE_DISK_GB:
            raise RuntimeError("disco libre insuficiente")
    ckpt = OUT_DIR / f"bf_newgen_ep{epochs}_{stamp}.pt"
    p2.save_checkpoint(ckpt, model, opt, sched, epochs)
    report["checkpoint"] = {"path": str(ckpt.relative_to(ROOT)), "sha256": sha256_file(ckpt)}

    # ---- score NEW Bf on the cohort ----
    model.eval()
    rec = {"new_bf": [], "old_bf": [], "mixture": []}
    with torch.no_grad():
        vecs = model.baskets.vectors(cat_idx, cat_cnt, torch.arange(n_cat, device=device))
        for s in range(0, n_val, 512):
            sl = slice(s, min(s + 512, n_val))
            batch = {k: v[sl] for k, v in {"champs": val_ds.champs, "sides": val_ds.sides, "items": val_ds.items,
                                            "query": val_ds.query, "inv": val_ds.inv, "hist": val_ds.hist}.items()}
            scores = model.scores(model.state(batch, device), vecs)
            b = scores.shape[0]
            gidx = gid.unsqueeze(0).expand(b, -1)
            gmax = torch.full((b, n_groups), float("-inf"), device=device)
            gmax.scatter_reduce_(1, gidx, scores, reduce="amax", include_self=True)
            gsum = torch.zeros(b, n_groups, device=device)
            gsum.scatter_add_(1, gidx, (scores - gmax.gather(1, gidx)).exp())
            top = torch.topk((gmax + gsum.log()).cpu(), min(v1.SHORTLIST, n_groups), dim=1).indices
            for j in range(b):
                rec["new_bf"].append(wardfree_record(val_rows[s + j], decode(top[j].tolist(), group_main, val_rows[s + j], dragon, purchase_sequence)))
    print(f"new Bf en cohorte: exact {p2.aggregate(rec['new_bf'])['first_action_exact']['value']:.4f}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- score OLD Bf and the frozen MIXTURE on the SAME rows via the package ----
    import app.mixture_policy as mp
    policy = mp.load_package(args.package, dragon=dragon, device=str(device))
    pkg_groups = [Counter(dict(g)) for g in policy.group_main]
    t_mix = time.monotonic()
    with torch.no_grad():
        for s in range(0, n_val, 256):
            rows = val_rows[s:s + 256]
            # old Bf alone: the package's domain parent, its own catalogue, grouped decode
            policy._apply_config()
            ds3 = tp.ShopDataset(list(rows), policy.champ_index, policy.item_index, policy.label_index, policy.dragon)
            b3 = next(ds3.batches(len(rows), shuffle=False))
            b3 = {k: b3[k] for k in ("champs", "sides", "items", "query", "inv", "hist")}
            ps = policy.parent.scores(policy.parent.state(b3, policy.device), policy._pvecs)
            b = ps.shape[0]
            gi = policy.group_ids.unsqueeze(0).expand(b, -1)
            gm = torch.full((b, policy.n_act), float("-inf"), device=policy.device)
            gm.scatter_reduce_(1, gi, ps, reduce="amax", include_self=True)
            gs = torch.zeros(b, policy.n_act, device=policy.device)
            gs.scatter_add_(1, gi, (ps - gm.gather(1, gi)).exp())
            top_old = torch.topk((gm + gs.log()).cpu(), min(v1.SHORTLIST, policy.n_act), dim=1).indices
            for j, row in enumerate(rows):
                rec["old_bf"].append(wardfree_record(row, decode(top_old[j].tolist(), pkg_groups, row, dragon, purchase_sequence)))
            # mixture: the adapter, features fresh, legality inside
            for row, opts in zip(rows, policy.options(rows)):
                conv = [("save", None) if o["kind"] == "save" else ("buy", Counter({int(k): int(v) for k, v in o["items"].items()})) for o in opts]
                rec["mixture"].append(wardfree_record(row, conv))
            if (s // 256) % 40 == 0:
                rate = (s + len(rows)) / max(time.monotonic() - t_mix, 1e-9)
                print(f"  paquete {s + len(rows)}/{n_val} ({rate:.0f} filas/s, ETA {(n_val - s - len(rows)) / max(rate, 1e-9) / 60:.0f} min)", flush=True)
    # tp globals were re-pointed by the package; restore ours for anything after
    tp.apply_config(config)

    ident = [(r["match_id"], r["ts"], r.get("champion")) for r in rec["new_bf"]]
    for k, recs in rec.items():
        assert [(r["match_id"], r["ts"], r.get("champion")) for r in recs] == ident, f"identidad desalineada: {k}"
        p2.dump(OUT_DIR / f"rows_{k}_{stamp}.jsonl", recs)
        report["aggregates"][k] = p2.aggregate(recs)
    subsets = {"single": lambda r: r["true_size"] == 1, "multi": lambda r: r["true_size"] > 1,
               "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
               "ward_only": lambda r: r.get("save_kind") == "ward_only"}
    for x, y in (("new_bf", "old_bf"), ("new_bf", "mixture"), ("mixture", "old_bf")):
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(rec[x], rec[y], metric)
        for label, pred in subsets.items():
            report["slices"][f"{x}_vs_{y}_{label}"] = p2.paired([r for r in rec[x] if pred(r)], [r for r in rec[y] if pred(r)], "first_action_exact")
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for t, recs in (("x", rec[x]), ("y", rec[y])):
            for r in recs:
                per[t][r["match_id"]][0] += r["first_action_exact"]
                per[t][r["match_id"]][1] += 1
        report["paired"][f"{x}_vs_{y}"]["ci95_2000draws"] = list(v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000))
        c = bk = 0
        for a, o in zip(rec[x], rec[y]):
            c += a["first_action_exact"] > o["first_action_exact"]
            bk += a["first_action_exact"] < o["first_action_exact"]
        report["movement"][f"{x}_vs_{y}"] = {"corrected": c, "broken": bk, "net": c - bk}

    prim = report["paired"]["new_bf_vs_old_bf"]
    ok = prim["diff"] * 100 >= CONTINUE_MIN_POINTS and prim["ci95"][0] > 0
    report["continuation"] = {
        "criterion": f"new_bf vs old_bf >= +{CONTINUE_MIN_POINTS} pts ward-free first-exact AND 95% CI excludes 0",
        "new_vs_old_points": round(prim["diff"] * 100, 3), "ci95_points": [round(x * 100, 3) for x in prim["ci95"]],
        "run_econ_wardfree_chain": bool(ok)}
    vm = report["paired"]["new_bf_vs_mixture"]
    report["product_summary"] = {
        "cohort": f"{n_val} decisions / {len(cohort_games)} games, played 2026-09-20..21, later than every training game",
        "new_bf_first_exact": report["aggregates"]["new_bf"]["first_action_exact"]["value"],
        "frozen_mixture_first_exact": report["aggregates"]["mixture"]["first_action_exact"]["value"],
        "old_bf_first_exact": report["aggregates"]["old_bf"]["first_action_exact"]["value"],
        "new_bf_vs_mixture_points": round(vm["diff"] * 100, 3), "new_bf_vs_mixture_ci95": [round(x * 100, 3) for x in vm["ci95"]],
        "new_bf_vs_mixture_net_decisions": report["movement"]["new_bf_vs_mixture"]["net"],
        "new_bf_vs_old_bf_points": round(prim["diff"] * 100, 3),
        "new_bf_vs_old_bf_net_decisions": report["movement"]["new_bf_vs_old_bf"]["net"],
        "continue_chain": bool(ok)}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = OUT_DIR / f"bf_stage1_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"product_summary": report["product_summary"], "continuation": report["continuation"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
