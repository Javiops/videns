"""Frozen dev baseline on the independent temporal window (VAL split).

Protocol: docs/experiment_val_window_check_v1.md. ONE frozen configuration
(econ_ft_ep8 + frozen-parent domain + FIT stats + ward-grouped decoder),
evaluated once on visits_val.jsonl under the ward-free metric. Everything
is computed fresh here (no caches cover VAL): parent forward -> D(x),
economic features through the pinned engine, tuned corrected scores,
grouped logsumexp decode. No comparisons, no selection, no tuning.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows.
"""
import argparse
import json
import hashlib
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR, econ_features, to_action
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import strip_counter, wardfree_record

OUT_DIR = ROOT / "data" / "experiments" / "16.18-val-window-check-v1"
V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"


def game_bootstrap_ci(records, metric):
    per_game = {}
    zeros = {}
    for r in records:
        if r.get(metric) is None:
            continue
        e = per_game.setdefault(r["match_id"], [0.0, 0])
        e[0] += r[metric]
        e[1] += 1
        zeros[r["match_id"]] = [0.0, 0]
    return list(v1.paired_bootstrap(per_game, zeros))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
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
    val_path = generation / "visits_val.jsonl"
    champ_ids, item_ids, label_counts, catalogue, _, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets).encode()).hexdigest()
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frozen_cfg = json.loads((FT_DIR / "frozen_dev_baseline_v1.json").read_text(
        encoding="utf-8"))

    # group map (identical rule to basket_grouped_decoder)
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

    cache_report = json.loads(
        (CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas_fit = econ_fit[:, :, 1] > 0
    stats_mean = torch.zeros(N_ECON)
    stats_std = torch.ones(N_ECON)
    flat = econ_fit[feas_fit].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)
    del econ_fit, flat, feas_fit

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

    parent = v1.BasketNet(*arch_args).to(device)
    parent.load_state_dict(torch.load(V2_DIR / "Bf_ep16.pt",
                                      map_location="cpu",
                                      weights_only=False)["state_dict"])
    parent.eval()
    tuned = TunedNet(v1.BasketNet(*arch_args).to(device),
                     CorrectionNet().to(device))
    tuned.load_state_dict(torch.load(FT_DIR / "econ_ft_ep8.pt",
                                     map_location="cpu",
                                     weights_only=False)["state_dict"])
    tuned.eval()

    cap = 2000 if args.smoke else None
    val_rows = list(v1.stream_split(val_path, set(), False, champ_index,
                                    dragon, cap))
    n_val = len(val_rows)
    val_ds = tp.ShopDataset.from_stream(iter(val_rows), n_val, champ_index,
                                        item_index, label_index, dragon)
    if device.type == "cuda":
        val_ds.to_device(device)
    gid = group_ids.to(device)
    games = len({r["match_id"] for r in val_rows})
    print(f"val: {n_val} decisiones / {games} partidas "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    records = []
    counters = Counter()
    with torch.no_grad():
        for s in range(0, n_val, 512):
            sl = slice(s, min(s + 512, n_val))
            batch = {k: v[sl] for k, v in {
                "champs": val_ds.champs, "sides": val_ds.sides,
                "items": val_ds.items, "query": val_ds.query,
                "inv": val_ds.inv, "hist": val_ds.hist}.items()}
            # frozen parent -> D(x)
            pstate = parent.state(batch, device)
            pvecs = parent.baskets.vectors(cat_idx, cat_cnt,
                                           torch.arange(n_cat, device=device))
            pscores = parent.scores(pstate, pvecs)
            dom = torch.topk(pscores, v1.SHORTLIST, dim=1).indices
            # economic features through the pinned engine (fresh, no cache)
            b = dom.shape[0]
            econ = torch.zeros((b, v1.SHORTLIST, N_ECON))
            dom_cpu = dom.cpu()
            for j in range(b):
                row = val_rows[s + j]
                budget = float(row["gold_est"])
                for k in range(v1.SHORTLIST):
                    cand = int(dom_cpu[j, k])
                    if cand == 0:
                        econ[j, k, 0] = 1.0
                    else:
                        econ[j, k] = torch.tensor(econ_features(
                            to_action(baskets[cand - 1]), row, budget,
                            dragon, counters))
            econ_n = (econ.to(device) - sm) / ss
            scores, _ = tuned.corrected_scores(batch, device, cat_idx,
                                               cat_cnt, n_cat, dom, econ_n)
            gidx = gid.unsqueeze(0).expand(b, -1)
            gmax = torch.full((b, n_groups), float("-inf"), device=device)
            gmax.scatter_reduce_(1, gidx, scores, reduce="amax",
                                 include_self=True)
            gsum = torch.zeros(b, n_groups, device=device)
            gsum.scatter_add_(1, gidx, (scores - gmax.gather(1, gidx)).exp())
            grouped = (gmax + gsum.log()).cpu()
            top = torch.topk(grouped, min(v1.SHORTLIST, n_groups),
                             dim=1).indices
            for j in range(b):
                row = val_rows[s + j]
                budget = float(row["gold_est"])
                feas_map = {int(dom_cpu[j, k]): bool(econ[j, k, 1] > 0)
                            for k in range(v1.SHORTLIST)}
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
                    if len(options) >= 3:
                        break
                if not options:
                    options = [("save", None)]
                records.append(wardfree_record(row, options))
            if (s // 512) % 20 == 0:
                rate = (s + b) / max(time.monotonic() - started, 1)
                print(f"  {s + b}/{n_val} ({rate:.0f} filas/s, "
                      f"ETA {(n_val - s - b) / max(rate, 1) / 60:.0f} min)",
                      flush=True)
    p2.dump(OUT_DIR / f"rows_val_baseline_{stamp}.jsonl", records)

    agg = p2.aggregate(records)
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "config": frozen_cfg["name"],
              "config_checkpoint_sha256":
                  frozen_cfg["components"]["checkpoint"]["sha256"],
              "split": "val (frozen temporal, later window)",
              "n_decisions": n_val, "n_games": games,
              "aggregates": agg, "ci": {}, "slices": {},
              "feature_counters": dict(counters)}
    report["ci"]["first_action_exact"] = game_bootstrap_ci(
        records, "first_action_exact")
    report["ci"]["any_of_three_exact"] = game_bootstrap_ci(
        records, "any_of_three_exact")
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    for label, pred in subsets.items():
        sub = [r for r in records if pred(r)]
        vals = [r["first_action_exact"] for r in sub]
        report["slices"][label] = {
            "n": len(sub),
            "first_action_exact": sum(vals) / max(len(vals), 1),
            "ci95": game_bootstrap_ci(sub, "first_action_exact")}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"val_window_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"first": agg["first_action_exact"],
                      "any3": agg["any_of_three_exact"],
                      "ci": report["ci"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
