"""Size diagnostics on the frozen finals, per the external review's precisions.

Evidence, not automatic cause identification. Three measurements per model
(Bs_ep16, Bf_ep16), one inference pass over the probe:

1. CONTEXT calibration of the multi-item event, conditioned on buying:
   p_multi(x) = sum_{k>=2} mass_k / (1 - mass_SAVE). Reported as (a) 10
   confidence bins (reliability + ECE) and (b) PRE-DECLARED observable
   strata — role, game phase (<10 / 10-20 / >=20 min), wallet staleness
   (ts mod 60s: <15 / 15-30 / >=30 s) — each with n, predicted-vs-observed
   gap and a per-game bootstrap CI. Confidence bins alone measure
   confidence-calibration, not context (opposite errors can cancel inside
   a bin); the strata are the context part.
2. ORACLE-SIZE diagnostic (privileged information, labelled as such): argmax
   restricted to catalogue baskets of the TRUE size, legality at the
   ORIGINAL budget, truth NOT inserted as a candidate. Shows whether knowing
   the size WOULD help — not that size is predictable from the inputs.
   Reported globally and on the independent feasible population.
3. PRE- vs POST-economic-filter masses by size (post = renormalized over the
   legal members of the top-128; partial, declared), so a wallet error is
   not misread as a cardinality error.

No training, no validation/test reads. Run via footwear_runtime after the
prework frees the GPU:
  footwear_runtime.py --entry scripts/basket_size_diagnostics.py --review ... -- [--smoke]
"""
import argparse
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

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe

RUN_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
OUT_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v3-prework"


def phase_of(ts):
    m = ts / 60000
    return "early(<10m)" if m < 10 else ("mid(10-20m)" if m < 20 else "late(>=20m)")


def staleness_of(ts):
    s = (ts % 60000) / 1000
    return "fresh(<15s)" if s < 15 else ("mid(15-30s)" if s < 30 else "stale(>=30s)")


def game_ci(pairs_by_game, n=400, seed=16):
    """CI95 of mean(observed - predicted) resampling games."""
    games = sorted(pairs_by_game)
    if not games:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        pick = [games[rng.randrange(len(games))] for _ in games]
        num = sum(pairs_by_game[g][0] for g in pick)
        den = sum(pairs_by_game[g][1] for g in pick)
        if den:
            diffs.append(num / den)
    diffs.sort()
    return [diffs[int(0.025 * (len(diffs) - 1))],
            diffs[int(0.975 * (len(diffs) - 1))]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"

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
    cap_probe = 2000 if args.smoke else None
    champ_ids, item_ids, label_counts, catalogue, n_fit, _n = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_ds = tp.ShopDataset.from_cache(
        ROOT / "data/experiments/16.18-whole-basket-v1/cache_probe_full_v2.pt",
        champ_index, item_index, label_index, dragon)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, cap_probe))
    if device.type == "cuda":
        probe_ds.to_device(device)
    cat_items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    n_cat = len(baskets) + 1
    cat_idx = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    cat_cnt = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    sizes = [0]
    for row_i, b in enumerate(baskets, start=1):
        parts = b.split(",")
        sizes.append(sum(int(p.split(":")[1]) for p in parts))
        for slot, part in enumerate(parts[:v1.MAX_BASKET]):
            i, n_ = part.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(n_), v1.MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)
    cat_sizes_t = torch.tensor(sizes, device=device)
    by_size_slices = {k: torch.nonzero(cat_sizes_t == k).squeeze(1)
                      for k in range(1, 7)}
    arch_args = (len(champ_index), len(item_index), len(label_index),
                 len(item_index) + 1, len(cat_items))
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "models": {},
              "reading_rules": [
                  "Evidence, not automatic cause identification.",
                  "Oracle-size uses privileged information (true size).",
                  "Post-filter masses renormalize over legal top-128 only "
                  "(partial).",
              ]}

    for branch in ("Bs", "Bf"):
        blob = torch.load(RUN_DIR / f"{branch}_ep16.pt", map_location="cpu",
                          weights_only=False)
        model = v1.BasketNet(*arch_args).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        bins = [{"p": 0.0, "o": 0.0, "n": 0} for _ in range(10)]
        strata = defaultdict(lambda: {"p": 0.0, "o": 0.0, "n": 0,
                                      "games": defaultdict(lambda: [0.0, 0])})
        oracle = {"global": [0, 0], "feasible": [0, 0]}
        pre_mass = torch.zeros(7, dtype=torch.float64)
        post_mass = torch.zeros(7, dtype=torch.float64)
        n_buy = 0
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
                probs = torch.softmax(scores, dim=1)
                top = torch.topk(scores, 128, dim=1).indices.cpu()
                for j in range(probs.shape[0]):
                    row = probe_rows[s + j]
                    truth = v1.truth_of(row)
                    if truth is None:
                        continue
                    n_buy += 1
                    budget = float(row["gold_est"])
                    pr = probs[j]
                    m0 = float(pr[0])
                    cond = pr[1:] / max(1.0 - m0, 1e-9)
                    mass_k = [0.0] * 7
                    for k in range(1, 7):
                        mass_k[k] = float(cond[cat_sizes_t[1:] == k].sum())
                        pre_mass[k] += mass_k[k]
                    p_multi = sum(mass_k[2:])
                    true_multi = float(sum(truth.values()) >= 2)
                    b = min(int(p_multi * 10), 9)
                    bins[b]["p"] += p_multi
                    bins[b]["o"] += true_multi
                    bins[b]["n"] += 1
                    for key in (f"role:{row.get('role')}",
                                f"phase:{phase_of(row['ts'])}",
                                f"wallet:{staleness_of(row['ts'])}"):
                        st = strata[key]
                        st["p"] += p_multi
                        st["o"] += true_multi
                        st["n"] += 1
                        g = st["games"][row["match_id"]]
                        g[0] += true_multi - p_multi
                        g[1] += 1
                    # post-filter masses over legal top-128
                    legal_probs = []
                    for cand in top[j].tolist():
                        if cand == 0:
                            continue
                        action = p2.to_action(baskets[cand - 1])
                        if purchase_sequence(action, row, budget,
                                             dragon) is not None:
                            legal_probs.append((int(cat_sizes_t[cand]),
                                                float(pr[cand])))
                    tot = sum(p for _k, p in legal_probs) or 1e-9
                    for k, p in legal_probs:
                        post_mass[min(k, 6)] += p / tot
                    # oracle-size: argmax within TRUE size slice, legal at
                    # original budget, truth not inserted
                    tk = min(sum(truth.values()), 6)
                    slice_idx = by_size_slices.get(tk)
                    hit = 0
                    if slice_idx is not None and len(slice_idx):
                        sl_scores = scores[j, slice_idx]
                        order = torch.argsort(sl_scores, descending=True)[:128]
                        for oi in order.tolist():
                            cand = int(slice_idx[oi])
                            action = p2.to_action(baskets[cand - 1])
                            if purchase_sequence(action, row, budget,
                                                 dragon) is not None:
                                hit = int(Counter(action) == truth)
                                break
                    oracle["global"][0] += hit
                    oracle["global"][1] += 1
                    if purchase_sequence(p2.to_action(
                            v1.basket_key(sorted(truth.items()))), row,
                            budget, dragon) is not None:
                        oracle["feasible"][0] += hit
                        oracle["feasible"][1] += 1
        ece = sum(abs(b["p"] - b["o"]) for b in bins if b["n"]) / max(n_buy, 1)
        report["models"][branch] = {
            "n_purchases": n_buy,
            "reliability_bins": [
                {"bin": i, "n": b["n"],
                 "mean_pred": b["p"] / b["n"] if b["n"] else None,
                 "observed": b["o"] / b["n"] if b["n"] else None}
                for i, b in enumerate(bins)],
            "ece_multi": ece,
            "strata": {k: {"n": v["n"],
                           "mean_pred_multi": v["p"] / v["n"],
                           "observed_multi": v["o"] / v["n"],
                           "gap_obs_minus_pred": (v["o"] - v["p"]) / v["n"],
                           "gap_ci95_per_game": game_ci(v["games"])}
                       for k, v in sorted(strata.items()) if v["n"] >= 200},
            "oracle_size": {
                "global": {"hits": oracle["global"][0],
                           "n": oracle["global"][1],
                           "value": oracle["global"][0] / max(oracle["global"][1], 1)},
                "feasible_independent": {
                    "hits": oracle["feasible"][0], "n": oracle["feasible"][1],
                    "value": oracle["feasible"][0] / max(oracle["feasible"][1], 1)}},
            "pre_filter_mean_mass_by_size": {
                str(k): pre_mass[k].item() / max(n_buy, 1) for k in range(1, 7)},
            "post_filter_mean_mass_by_size": {
                str(k): post_mass[k].item() / max(n_buy, 1) for k in range(1, 7)},
        }
        o = report["models"][branch]["oracle_size"]
        print(f"{branch}: ECE {ece:.4f}  oracle global "
              f"{o['global']['value']:.4f}  feasible "
              f"{o['feasible_independent']['value']:.4f} "
              f"({time.monotonic() - started:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"size_diagnostics_{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {OUT_DIR / f'size_diagnostics_{stamp}.json'}", flush=True)


if __name__ == "__main__":
    main()
