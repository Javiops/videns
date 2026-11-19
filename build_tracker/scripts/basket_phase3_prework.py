"""Phase-3 prework ordered by the external review (no new training).

1. GENERALIZATION CURVES: evaluate the saved full-state checkpoints
   Bs_ep{4,8,16} and Bf_ep{4,8,16} on the probe with the common evaluator —
   FIT loss descending does not establish that generalization is still
   improving; this measures it.
2. SIZE-MASS CALIBRATION on the frozen final models: P(K=k|x) = sum of
   catalogue softmax mass over baskets of size k (SAVE = size 0). Compares
   the learned size distribution against the empirical one, separating
   "size probabilities miscalibrated" from "argmax choice favours small
   baskets" (the reviewer's 60/40 point: argmax of a perfectly calibrated
   distribution can still under-predict the mean).
3. ENRICHED per-decision rows for the final models: all three displayed
   options (kind + item counts) alongside the truth, so duplicates and
   alternatives can be audited externally.

Reads fit/probe caches from phase 2; no training, no validation/test reads.
Run through footwear_runtime with PREFIX_ML_DIR set:
  footwear_runtime.py --entry scripts/basket_phase3_prework.py --review ... -- [--smoke]
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

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe

RUN_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
OUT_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v3-prework"


def size_of_catalogue(baskets):
    sizes = [0]                                   # SAVE
    for b in baskets:
        sizes.append(sum(int(part.split(":")[1]) for part in b.split(",")))
    return torch.tensor(sizes)


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

    from app.ddragon import dragon_for_patch
    from app.shop_econ import components_of
    dragon = dragon_for_patch(manifest_gen["patch"])
    fam_cache = {}

    def family_of(t):
        if t not in fam_cache:
            fam_cache[t] = set(components_of(t, dragon)) | {t}
        return fam_cache[t]

    train_path = generation / "visits_train.jsonl"
    # smoke still needs FULL indexes/catalogue (the checkpoints were trained
    # on them); it only caps how many probe rows are evaluated.
    cap_fit = None
    cap_probe = 2000 if args.smoke else None
    champ_ids, item_ids, label_counts, catalogue, n_fit, n_probe = v1.first_pass(
        train_path, fit_set, cap_fit, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_ids = sorted(label_counts)
    label_index = {i: k for k, i in enumerate(label_ids)}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    probe_ds = tp.ShopDataset.from_cache(
        v1_dir / "cache_probe_full_v2.pt", champ_index, item_index,
        label_index, dragon)
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
    for row_i, b in enumerate(baskets, start=1):
        for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, n_ = part.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(n_), v1.MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)
    cat_sizes = size_of_catalogue(baskets).to(device)
    arch_args = (len(champ_index), len(item_index), len(label_ids),
                 len(item_index) + 1, len(cat_items))
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    def load_model(ckpt_path):
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = v1.BasketNet(*arch_args).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model

    report = {"schema": 1,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "checkpoint_curves": {},
              "size_mass_calibration": {}, "elapsed_seconds": None}

    # ---- 1. generalization curves over saved checkpoints ----
    epochs_avail = [4, 8, 16] if not args.smoke else [16]
    for branch in ("Bs", "Bf"):
        report["checkpoint_curves"][branch] = []
        for ep in epochs_avail:
            path = RUN_DIR / f"{branch}_ep{ep}.pt"
            if not path.exists():
                continue
            model = load_model(path)
            recs = p2.eval_basket_model(f"{branch}_ep{ep}", model, probe_ds,
                                        probe_rows, baskets, basket_pos,
                                        cat_idx, cat_cnt, dragon, device,
                                        family_of)
            agg = p2.aggregate(recs)
            entry = {"epoch": ep,
                     "first_action_exact": agg["first_action_exact"]["value"],
                     "any_of_three_exact": agg["any_of_three_exact"]["value"],
                     "action_family_match": agg["action_family_match"]["value"],
                     "mean_pred_size": agg["size_stats"]["mean_pred_size_when_buying"],
                     "multi_item_first_exact": None}
            multi = [r for r in recs if r["true_size"] >= 2]
            entry["multi_item_first_exact"] = (
                sum(r["first_action_exact"] for r in multi) / max(len(multi), 1))
            report["checkpoint_curves"][branch].append(entry)
            print(f"{branch}_ep{ep}: exact {entry['first_action_exact']:.4f} "
                  f"any3 {entry['any_of_three_exact']:.4f} "
                  f"multi {entry['multi_item_first_exact']:.4f} "
                  f"size {entry['mean_pred_size']:.3f} "
                  f"({time.monotonic() - started:.0f}s)", flush=True)
            # enriched rows only for the finals
            if ep == epochs_avail[-1]:
                enrich_and_dump(f"{branch}_ep{ep}", model, probe_ds,
                                probe_rows, baskets, cat_idx, cat_cnt,
                                dragon, device, stamp)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- 2. size-mass calibration (frozen finals) ----
    for branch in ("Bs", "Bf"):
        path = RUN_DIR / f"{branch}_ep{epochs_avail[-1]}.pt"
        if not path.exists():
            continue
        model = load_model(path)
        mass_by_k = torch.zeros(7, dtype=torch.float64)
        argmax_size = Counter()
        true_size = Counter()
        n_rows = 0
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
                probs = torch.softmax(model.scores(state, vecs), dim=1)
                for k in range(7):
                    mask = (cat_sizes == k)
                    mass_by_k[k] += float(probs[:, mask].sum())
                am = probs.argmax(dim=1)
                for a in am.tolist():
                    argmax_size[int(cat_sizes[a])] += 1
                for j in range(probs.shape[0]):
                    t = v1.truth_of(probe_rows[s + j])
                    true_size[0 if t is None else min(sum(t.values()), 6)] += 1
                n_rows += probs.shape[0]
        report["size_mass_calibration"][branch] = {
            "n": n_rows,
            "mean_mass_by_size": {str(k): mass_by_k[k].item() / n_rows
                                  for k in range(7)},
            "argmax_basket_size_freq": {str(k): argmax_size.get(k, 0) / n_rows
                                        for k in range(7)},
            "true_size_freq": {str(k): true_size.get(k, 0) / n_rows
                               for k in range(7)},
        }
        print(f"{branch} calibracion masas: "
              f"{report['size_mass_calibration'][branch]['mean_mass_by_size']}",
              flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    (OUT_DIR / f"prework_report_{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {OUT_DIR / f'prework_report_{stamp}.json'}", flush=True)


def enrich_and_dump(name, model, probe_ds, probe_rows, baskets, cat_idx,
                    cat_cnt, dragon, device, stamp):
    """Rows with all three displayed options (kind + counts) + full truth."""
    from app.conditional_policy import purchase_sequence
    out = OUT_DIR / f"rows_{name}_{stamp}_enriched.jsonl"
    n_cat = len(baskets) + 1
    with torch.no_grad(), open(out, "w", encoding="utf-8",
                               newline="\n") as fh:
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
                options = []
                for cand in top[j].tolist():
                    if cand == 0:
                        options.append({"kind": "save"})
                    else:
                        action = p2.to_action(baskets[cand - 1])
                        if purchase_sequence(action, row, budget,
                                             dragon) is not None:
                            options.append({"kind": "buy",
                                            "item_counts": dict(Counter(action))})
                    if len(options) >= 3:
                        break
                truth = v1.truth_of(row)
                fh.write(json.dumps({
                    "match_id": row["match_id"], "ts": row["ts"],
                    "champion": row.get("champion"),
                    "save_kind": row.get("save_kind"),
                    "gold_est": row["gold_est"],
                    "truth_item_counts": (dict(truth) if truth else None),
                    "options": options}, ensure_ascii=False,
                    separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
