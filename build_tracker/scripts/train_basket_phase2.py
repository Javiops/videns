"""Phase 2 of the whole-basket experiment: bounded loss contrast (B-s vs B-f).

Protocol: docs/experiment_whole_basket_v2_loss_contrast.md. Implements the
external review's program:

- Two branches from IDENTICAL clean state (same seed, same initial weights,
  same optimizer config): B-s = v1's sampled-softmax loss; B-f = exact
  full-catalogue cross-entropy via microbatch (32) gradient accumulation to
  an effective batch of 256 (chunking alone does not bound backward memory).
- Max 16 epochs, full-state checkpoints (model/optimizer/scheduler/RNG) at
  epochs 4/8/16, per-epoch full-catalogue CE on a FIXED 10k FIT subset for
  BOTH branches, wall time and peak GPU memory logged. No auto-extension:
  not converged in budget => "inconclusive".
- A COMMON evaluator producing per-decision records for arm A (real
  predictor path, evaluated once and persisted) and both branches:
  first_action_exact (is_no_buy = save_kind == no_buy_death, the canonical
  definition), any_of_three, action_family_match (family = next completed
  final + its components), SAVE-fallback counter, true/predicted basket
  size, over/under-fill and exactly-one-extra rates, and a decomposition
  whose legality bucket is SPLIT into budget_blocked vs structurally_blocked
  (executor retried at 10M gold). Full per-decision JSONL persisted for
  independent recomputation of every interval.

Run through the activated 16.18 runtime with PREFIX_ML_DIR set:
  footwear_runtime.py --entry scripts/train_basket_phase2.py --review ... -- [--smoke]
"""
import argparse
import copy
import json
import os
import random
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
import train_prefix as tp
from basket_vocabulary_probe import (basket_entries, basket_key,
                                     membership_digest, split_fit_probe)

RUN_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
MICRO = 32          # microbatch for full-CE; accumulate to EFFECTIVE
EFFECTIVE = 256
FIXED_SUBSET = 10000
CHECKPOINTS = (4, 8, 16)


def to_action(bkey):
    return tuple(sorted(i for part in bkey.split(",")
                        for i in [int(part.split(":")[0])] * int(part.split(":")[1])))


def full_ce_on_subset(model, fit_ds, fit_targets, subset_idx, cat_idx, cat_cnt,
                      device):
    """Exact full-catalogue CE on the fixed FIT subset (comparable scale)."""
    model.eval()
    nll = nn.CrossEntropyLoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                     torch.arange(cat_idx.shape[0], device=device))
        for s in range(0, len(subset_idx), 512):
            idxs = subset_idx[s:s + 512]
            batch = {k: v[idxs.to(v.device)] for k, v in {
                "champs": fit_ds.champs, "sides": fit_ds.sides,
                "items": fit_ds.items, "query": fit_ds.query,
                "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
            state = model.state(batch, device)
            scores = model.scores(state, vecs)
            total += float(nll(scores, fit_targets[idxs]))
            n += len(idxs)
    model.train()
    return total / max(n, 1)


def save_checkpoint(path, model, opt, sched, epoch):
    torch.save({"state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "py_random": random.getstate()}, path)


def train_branch(name, loss_kind, init_state, fit_ds, fit_targets, cat_idx,
                 cat_cnt, n_fit, baskets, subset_idx, device, epochs, out_dir,
                 arch_args):
    model = v1.BasketNet(*arch_args).to(device)
    model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    nll = nn.CrossEntropyLoss()
    torch.manual_seed(16)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    curve = []
    n_cat = len(baskets) + 1
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.monotonic()
        total, n = 0.0, 0
        order = torch.randperm(n_fit, device=device)
        for s in range(0, n_fit, EFFECTIVE):
            idxs_all = order[s:s + EFFECTIVE]
            opt.zero_grad()
            if loss_kind == "sampled":
                batch = {k: v[idxs_all.to(v.device)] for k, v in {
                    "champs": fit_ds.champs, "sides": fit_ds.sides,
                    "items": fit_ds.items, "query": fit_ds.query,
                    "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                state = model.state(batch, device)
                targets = fit_targets[idxs_all]
                negs = torch.randint(1, n_cat, (2048,), device=device)
                cand = torch.cat([targets, torch.zeros(1, dtype=torch.long,
                                                       device=device), negs])
                uniq, inv_pos = torch.unique(cand, return_inverse=True)
                vecs = model.baskets.vectors(cat_idx, cat_cnt, uniq)
                loss = nll(model.scores(state, vecs), inv_pos[:len(targets)])
                loss.backward()
                total += float(loss.detach()) * len(idxs_all)
            else:  # full-catalogue CE, microbatched with grad accumulation
                loss_sum = 0.0
                for m0 in range(0, len(idxs_all), MICRO):
                    idxs = idxs_all[m0:m0 + MICRO]
                    batch = {k: v[idxs.to(v.device)] for k, v in {
                        "champs": fit_ds.champs, "sides": fit_ds.sides,
                        "items": fit_ds.items, "query": fit_ds.query,
                        "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                    state = model.state(batch, device)
                    vecs = model.baskets.vectors(
                        cat_idx, cat_cnt, torch.arange(n_cat, device=device))
                    loss = nll(model.scores(state, vecs), fit_targets[idxs])
                    (loss * len(idxs) / len(idxs_all)).backward()
                    loss_sum += float(loss.detach()) * len(idxs)
                total += loss_sum
            opt.step()
            n += len(idxs_all)
        sched.step()
        fixed_ce = full_ce_on_subset(model, fit_ds, fit_targets, subset_idx,
                                     cat_idx, cat_cnt, device)
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0)
        curve.append({"epoch": epoch, "train_loss": total / max(n, 1),
                      "fixed_subset_full_ce": fixed_ce,
                      "seconds": round(time.monotonic() - t_ep, 1),
                      "peak_gpu_gb": round(peak, 2)})
        print(f"  {name} epoch {epoch}/{epochs} loss {total / max(n, 1):.3f} "
              f"fixedCE {fixed_ce:.3f} ({curve[-1]['seconds']}s, "
              f"peak {curve[-1]['peak_gpu_gb']}GB)", flush=True)
        if epoch in CHECKPOINTS:
            save_checkpoint(out_dir / f"{name}_ep{epoch}.pt", model, opt,
                            sched, epoch)
    return model, curve


def eval_basket_model(name, model, probe_ds, probe_rows, baskets, basket_pos,
                      cat_idx, cat_cnt, dragon, device, family_of):
    """Common-evaluator records for a catalogue-scorer policy."""
    from app.conditional_policy import purchase_sequence
    model.eval()
    records = []
    n_cat = len(baskets) + 1
    with torch.no_grad():
        vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                     torch.arange(n_cat, device=device))
        for s in range(0, len(probe_rows), tp.BATCH):
            sl = slice(s, min(s + tp.BATCH, len(probe_rows)))
            batch = {k: v[sl] for k, v in {
                "champs": probe_ds.champs, "sides": probe_ds.sides,
                "items": probe_ds.items, "query": probe_ds.query,
                "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
            state = model.state(batch, device)
            scores = model.scores(state, vecs)
            top = torch.topk(scores, min(v1.SHORTLIST, n_cat), dim=1).indices.cpu()
            for j in range(top.shape[0]):
                row = probe_rows[s + j]
                budget = float(row["gold_est"])
                options = []          # list of (kind, Counter|None)
                fallback_save = False
                for cand in top[j].tolist():
                    if cand == 0:
                        options.append(("save", None))
                    else:
                        action = to_action(baskets[cand - 1])
                        if purchase_sequence(action, row, budget, dragon) is not None:
                            options.append(("buy", Counter(action)))
                    if len(options) >= 3:
                        break
                if not options:
                    options = [("save", None)]
                    fallback_save = True
                rec = base_record(row, options, family_of, dragon)
                rec["fallback_save"] = fallback_save
                truth = v1.truth_of(row)
                if truth is not None:
                    tkey = basket_key(sorted(truth.items()))
                    tpos = basket_pos.get(tkey)
                    if tpos is None:
                        rec["bucket"] = "catalogue_miss"
                    elif tpos not in set(top[j].tolist()):
                        rec["bucket"] = "shortlist_miss"
                    else:
                        action = to_action(tkey)
                        if purchase_sequence(action, row, budget, dragon) is not None:
                            rec["bucket"] = ("hit" if rec["first_action_exact"]
                                             else "ranking_miss")
                        elif purchase_sequence(action, row, 10_000_000.0,
                                               dragon) is not None:
                            rec["bucket"] = "budget_blocked"
                        else:
                            rec["bucket"] = "structurally_blocked"
                records.append(rec)
    return records


def base_record(row, options, family_of, dragon):
    """Metric record from (kind, Counter) options — canonical definitions."""
    truth_all = Counter({int(k): int(n) for k, n in
                         (row.get("label_counts") or {}).items() if int(k)})
    truth = Counter({k: n for k, n in truth_all.items() if k != 999999})
    is_no_buy = row.get("save_kind") == "no_buy_death"
    first_kind, first_basket = options[0]
    predicted_save = first_kind == "save"
    pred = first_basket or Counter()
    rec = {"match_id": row["match_id"], "ts": row["ts"],
           "champion": row.get("champion"), "save_kind": row.get("save_kind"),
           "true_size": sum(truth.values()),
           "pred_size": 0 if predicted_save else sum(pred.values()),
           "pred_first": "SAVE" if predicted_save else dict(pred),
           "first_action_exact": (float(predicted_save) if is_no_buy
                                  else float(not predicted_save and pred == truth)),
           "any_of_three_exact": None, "action_family_match": None}
    if not is_no_buy and truth:
        rec["any_of_three_exact"] = float(any(
            k == "buy" and c == truth for k, c in options))
        target = int(row.get("target_id") or 0)
        if target:
            fam = family_of(target)
            rec["action_family_match"] = float(bool(pred)
                                               and any(i in fam for i in pred))
    return rec


def eval_arm_a(probe_rows, artifact, family_of, dragon):
    from app.predictor import Predictor
    from eval_policy import serving_row
    predictor = Predictor(artifact)
    records = []
    for row in probe_rows:
        budget = float(row["gold_est"])
        opts = predictor.predict_options(serving_row(row, budget),
                                         budget_slack=0, allow_save=True)
        options = []
        for o in (opts or [{"kind": "buy", "items": []}]):
            if o.get("kind") == "save":
                options.append(("save", None))
            else:
                options.append(("buy", Counter(int(i["item_id"])
                                               for i in o.get("items") or [])))
        records.append(base_record(row, options[:3] or [("buy", Counter())],
                                   family_of, dragon))
    return records


def aggregate(records):
    out = {}
    for metric in ("first_action_exact", "any_of_three_exact",
                   "action_family_match"):
        vals = [r[metric] for r in records if r.get(metric) is not None]
        out[metric] = {"value": sum(vals) / max(len(vals), 1), "n": len(vals)}
    purch = [r for r in records if r["true_size"] > 0]
    buys = [r for r in purch if r["pred_size"] > 0]
    out["size_stats"] = {
        "mean_true_size": sum(r["true_size"] for r in purch) / max(len(purch), 1),
        "mean_pred_size_when_buying": (sum(r["pred_size"] for r in buys)
                                       / max(len(buys), 1)),
        "overfill_rate": sum(r["pred_size"] > r["true_size"] for r in buys)
                         / max(len(buys), 1),
        "underfill_rate": sum(0 < r["pred_size"] < r["true_size"] for r in buys)
                          / max(len(buys), 1),
        "exactly_one_extra_rate": sum(r["pred_size"] == r["true_size"] + 1
                                      for r in buys) / max(len(buys), 1),
    }
    per_size = {}
    for r in purch:
        b = per_size.setdefault(min(r["true_size"], 5), {"n": 0, "hit": 0})
        b["n"] += 1
        b["hit"] += int(r["first_action_exact"])
    out["per_size"] = {str(k): per_size[k] for k in sorted(per_size)}
    dup = [r for r in purch if any(
        isinstance(r["pred_first"], dict) for _ in [0]) and r["true_size"] > 0]
    out["fallback_save_n"] = sum(int(r.get("fallback_save") or 0)
                                 for r in records)
    if "bucket" in (records[0] if records else {}):
        out["decomposition"] = dict(Counter(r["bucket"] for r in records
                                            if "bucket" in r))
    return out


def paired(records_x, records_y, metric):
    per_game = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
    for tag, recs in (("x", records_x), ("y", records_y)):
        for r in recs:
            if r.get(metric) is not None:
                per_game[tag][r["match_id"]][0] += r[metric]
                per_game[tag][r["match_id"]][1] += 1
    ci = v1.paired_bootstrap(per_game["x"], per_game["y"])
    vx = [r[metric] for r in records_x if r.get(metric) is not None]
    vy = [r[metric] for r in records_y if r.get(metric) is not None]
    return {"diff": sum(vx) / max(len(vx), 1) - sum(vy) / max(len(vy), 1),
            "ci95": list(ci), "n": len(vx)}


def dump(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False,
                                separators=(",", ":")) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    torch.manual_seed(16)
    random.seed(16)

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = "smoke" if args.smoke else "full"
    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"

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
    if (membership_digest(fit_games) != v1.FROZEN_FIT_SHA
            or membership_digest(probe_games) != v1.FROZEN_PROBE_SHA):
        raise SystemExit("split membership mismatch; refusing to run")
    fit_set = set(fit_games)

    import footwear_runtime as fr
    fr.declaration_for_manifest(manifest_gen)
    from app.ddragon import dragon_for_patch
    from app.shop_econ import components_of
    dragon = dragon_for_patch(manifest_gen["patch"])
    fam_cache = {}

    def family_of(target_id: int):
        if target_id not in fam_cache:
            fam_cache[target_id] = set(components_of(target_id, dragon)) | {target_id}
        return fam_cache[target_id]

    train_path = generation / "visits_train.jsonl"
    cap_fit = 20000 if args.smoke else None
    cap_probe = 4000 if args.smoke else None
    champ_ids, item_ids, label_counts, catalogue, n_fit, n_probe = v1.first_pass(
        train_path, fit_set, cap_fit, cap_probe)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_ids = sorted(label_counts)
    label_index = {i: k for k, i in enumerate(label_ids)}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reuse v1's tensor caches (identical indexes, deterministic)
    cache_fit = v1_dir / f"cache_fit_{stamp}_v2.pt"
    cache_probe = v1_dir / f"cache_probe_{stamp}_v2.pt"
    cache_targets = v1_dir / f"cache_targets_{stamp}_v2.pt"
    if not (cache_fit.exists() and cache_probe.exists()
            and cache_targets.exists()):
        raise SystemExit("v1 tensor caches missing; run the v1 harness first")
    fit_ds = tp.ShopDataset.from_cache(cache_fit, champ_index, item_index,
                                       label_index, dragon)
    probe_ds = tp.ShopDataset.from_cache(cache_probe, champ_index, item_index,
                                         label_index, dragon)
    fit_targets = torch.load(cache_targets, weights_only=True).to(device)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, cap_probe))
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    print(f"setup listo ({time.monotonic() - started:.0f}s): fit {n_fit} "
          f"probe {n_probe} catalogo {len(baskets)}", flush=True)

    cat_items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    n_cat = len(baskets) + 1
    cat_idx = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    cat_cnt = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, nn_ = part.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(nn_), v1.MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)

    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(n_fit, generator=g)[
        :min(FIXED_SUBSET, n_fit)].to(device)

    # identical clean init for both branches
    arch_args = (len(champ_index), len(item_index), len(label_ids),
                 len(item_index) + 1, len(cat_items))
    torch.manual_seed(16)
    init_model = v1.BasketNet(*arch_args)
    init_state = copy.deepcopy(init_model.state_dict())
    del init_model

    epochs = 2 if args.smoke else 16
    curves = {}
    finals = {}
    for name, kind in (("Bs", "sampled"), ("Bf", "full")):
        print(f"=== branch {name} ({kind}) ===", flush=True)
        model, curve = train_branch(name, kind, init_state, fit_ds,
                                    fit_targets, cat_idx, cat_cnt, n_fit,
                                    baskets, subset_idx, device, epochs,
                                    RUN_DIR, arch_args)
        curves[name] = curve
        finals[name] = model

    # ---- common evaluation ----
    results = {}
    arm_a_path = RUN_DIR / f"rows_A_{stamp}.jsonl"
    if arm_a_path.exists():
        results["A"] = [json.loads(line) for line in
                        open(arm_a_path, encoding="utf-8")]
        print("arm A: filas persistidas reutilizadas", flush=True)
    else:
        print("=== eval arm A (real predictor path) ===", flush=True)
        results["A"] = eval_arm_a(probe_rows,
                                  v1_dir / f"arm_a_{stamp}.pt",
                                  family_of, dragon)
        dump(arm_a_path, results["A"])
    for name in ("Bs", "Bf"):
        print(f"=== eval {name} ===", flush=True)
        results[name] = eval_basket_model(
            name, finals[name], probe_ds, probe_rows, baskets, basket_pos,
            cat_idx, cat_cnt, dragon, device, family_of)
        dump(RUN_DIR / f"rows_{name}_{stamp}.jsonl", results[name])

    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_whole_basket_v2_loss_contrast.md",
        "mode": stamp, "epochs": epochs,
        "architecture": {"d_model": tp.D_MODEL, "layers": tp.N_LAYERS,
                         "ff_dim": tp.FF_DIM, "heads": tp.N_HEADS,
                         "note": "v1 defaults kept deliberately — one "
                                 "variable (the loss) changes in this phase"},
        "curves": curves,
        "aggregates": {k: aggregate(v) for k, v in results.items()},
        "paired": {
            "Bs_vs_A_first_action": paired(results["Bs"], results["A"],
                                           "first_action_exact"),
            "Bf_vs_A_first_action": paired(results["Bf"], results["A"],
                                           "first_action_exact"),
            "Bf_vs_Bs_first_action": paired(results["Bf"], results["Bs"],
                                            "first_action_exact"),
            "Bs_vs_A_any3": paired(results["Bs"], results["A"],
                                   "any_of_three_exact"),
            "Bf_vs_Bs_any3": paired(results["Bf"], results["Bs"],
                                    "any_of_three_exact"),
        },
        "per_decision_rows": {k: f"rows_{k}_{stamp}.jsonl" for k in results},
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (RUN_DIR / f"phase2_report_{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("aggregates", "paired")},
                     indent=1), flush=True)
    print(f"\nreport: {RUN_DIR / f'phase2_report_{stamp}.json'}", flush=True)


if __name__ == "__main__":
    main()
