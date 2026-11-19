"""TRAIN-only basket vocabulary feasibility probe (no model prediction).

Python port of the frozen diagnostic
data/reports/strategy_analysis_20260920/basket_vocabulary_probe.js (its JSON
output and sample are the frozen evidence; this port exists so the diagnostic
is versioned, parametrized and reproducible). Reproduction is verified by
comparing this port's output against the frozen report — same counts, same
fit/probe membership digests, same sampled rows.

Definitions (identical to the original):
- fit/probe: TRAIN games sorted by (game_creation_ms, id); boundary is the
  creation of the game at index floor(0.8*N); fit = creation < boundary
  (timestamp ties therefore stay together on the probe side).
- basket key: "item:count,..." sorted by item id ascending; SAVE rows
  (empty or containing 999999) are excluded from purchases.
- retrieval shortlists: union of top-k fit baskets by champion+role,
  champion+role+inventory multiset, and champion+role+250g band+5min band;
  ties broken by count desc then key ascending. No truth used in retrieval.
- legality sample: sha256(match_id|participant_id|ts) first 4 bytes
  big-endian % 20 == 0, probe purchases only.

Reads ONLY visits_train.jsonl of the given generation. No validation or test
payloads, no model inference.

Usage:
  python scripts/basket_vocabulary_probe.py
      [--generation data/dataset_generations/16.18-reconstructible-development-v3]
      [--out data/reports/<dir>]
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVE_SENTINEL = 999999


def js_num(x) -> str:
    """Number -> string the way JS string concatenation renders it."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def basket_entries(label_counts: dict) -> list:
    entries = sorted((int(i), int(n)) for i, n in (label_counts or {}).items()
                     if int(i) and int(n) > 0)
    return entries


def basket_key(entries: list) -> str:
    return ",".join(f"{i}:{n}" for i, n in entries)


def inv_key(inventory: list) -> str:
    c = Counter(int(x) for x in (inventory or []))
    return ",".join(f"{i}:{c[i]}" for i in sorted(c))


def sampled(match_id: str, participant_id, ts) -> bool:
    key = f"{match_id}|{js_num(participant_id)}|{js_num(ts)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 20 == 0


def top_k(counter: Counter | None, k: int) -> list:
    if not counter:
        return []
    return [key for key, _n in sorted(counter.items(),
                                      key=lambda kv: (-kv[1], kv[0]))[:k]]


def quantile(values: list, p: float):
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * p)]


def membership_digest(ids: list) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def split_fit_probe(plan: dict) -> tuple:
    creation = plan["game_creation_ms"]
    train_games = sorted((g for g, s in plan["assignment"].items()
                          if s == "train"), key=lambda g: (creation[g], g))
    boundary = creation[train_games[math.floor(len(train_games) * 0.8)]]
    fit = [g for g in train_games if creation[g] < boundary]
    probe = [g for g in train_games if creation[g] >= boundary]
    return fit, probe, boundary


def run(generation: Path, out_dir: Path) -> dict:
    started = time.monotonic()
    plan = json.loads((generation / "split_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((generation / "generation_manifest.json").read_text(encoding="utf-8"))
    fit_games, probe_games, boundary = split_fit_probe(plan)
    fit_set = set(fit_games)
    creation = plan["game_creation_ms"]
    assignment = plan["assignment"]

    vocabulary, items, sizes = Counter(), Counter(), Counter()
    all_cr = defaultdict(Counter)
    fit_vocab = Counter()
    fit_cr, fit_cri, fit_context = (defaultdict(Counter) for _ in range(3))
    probe_rows_list = []
    rows = purchases = save = fit_purchases = fit_rows = probe_rows = 0
    duplicate_rows = ordered_labels = label_count_rows = 0
    observed = set()

    with open(generation / "visits_train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if assignment.get(r["match_id"]) != "train":
                raise SystemExit(f"Non-TRAIN row: {r['match_id']}")
            rows += 1
            observed.add(r["match_id"])
            in_fit = r["match_id"] in fit_set
            if in_fit:
                fit_rows += 1
            else:
                probe_rows += 1
            if r.get("label_counts"):
                label_count_rows += 1
            if r.get("purchase_sequence") or r.get("purchase_events") or r.get("ordered_label_ids"):
                ordered_labels += 1
            entries = basket_entries(r.get("label_counts"))
            if not entries or any(i == SAVE_SENTINEL for i, _n in entries):
                save += 1
                continue
            b = basket_key(entries)
            qty = sum(n for _i, n in entries)
            cr = f"{r['champion_id']}|{r['role']}"
            cri = f"{cr}|{inv_key(r.get('inventory'))}"
            context = (f"{cr}|{math.floor(float(r['gold_est']) / 250)}"
                       f"|{math.floor(r['ts'] / 300000)}")
            purchases += 1
            vocabulary[b] += 1
            sizes[qty] += 1
            all_cr[cr][b] += 1
            if any(n > 1 for _i, n in entries):
                duplicate_rows += 1
            for i, n in entries:
                items[i] += n
            if in_fit:
                fit_purchases += 1
                fit_vocab[b] += 1
                fit_cr[cr][b] += 1
                fit_cri[cri][b] += 1
                fit_context[context][b] += 1
            else:
                entry = {"b": b, "cr": cr, "cri": cri, "context": context,
                         "qty": qty}
                if sampled(r["match_id"], r["participant_id"], r["ts"]):
                    entry["sample_row"] = r
                    entry["sample_membership"] = {}
                probe_rows_list.append(entry)

    support = {"global_seen": 0, "champion_role_seen": 0,
               "champion_role_inventory_seen": 0, "global_seen_min5": 0,
               "champion_role_seen_min5": 0}
    by_size = {}
    configs = [{"k": k, "hits": 0, "total_candidates": 0,
                "max_candidates": 0, "by_size": {}} for k in (16, 32, 64, 128)]
    caches = [dict() for _ in configs]
    for r in probe_rows_list:
        if r["b"] in fit_vocab:
            support["global_seen"] += 1
        if fit_vocab.get(r["b"], 0) >= 5:
            support["global_seen_min5"] += 1
        if r["b"] in fit_cr.get(r["cr"], ()):
            support["champion_role_seen"] += 1
        if fit_cr.get(r["cr"], Counter()).get(r["b"], 0) >= 5:
            support["champion_role_seen_min5"] += 1
        if r["b"] in fit_cri.get(r["cri"], ()):
            support["champion_role_inventory_seen"] += 1
        bs = by_size.setdefault(r["qty"], {"n": 0, "global_seen": 0,
                                           "champion_role_seen": 0})
        bs["n"] += 1
        bs["global_seen"] += r["b"] in fit_vocab
        bs["champion_role_seen"] += r["b"] in fit_cr.get(r["cr"], ())
        for cfg, cache in zip(configs, caches):
            key = r["cri"] + "#" + r["context"]
            candidates = cache.get(key)
            if candidates is None:
                candidates = set(top_k(fit_cr.get(r["cr"]), cfg["k"]))
                candidates.update(top_k(fit_cri.get(r["cri"]), cfg["k"]))
                candidates.update(top_k(fit_context.get(r["context"]), cfg["k"]))
                cache[key] = candidates
            hit = int(r["b"] in candidates)
            if "sample_row" in r:
                r["sample_membership"][str(cfg["k"])] = bool(hit)
            cfg["hits"] += hit
            cfg["total_candidates"] += len(candidates)
            cfg["max_candidates"] = max(cfg["max_candidates"], len(candidates))
            cs = cfg["by_size"].setdefault(r["qty"], {"n": 0, "hits": 0})
            cs["n"] += 1
            cs["hits"] += hit

    counts_desc = sorted(vocabulary.values(), reverse=True)
    thresholds = []
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        total = k = 0
        while total / purchases < threshold and k < len(counts_desc):
            total += counts_desc[k]
            k += 1
        thresholds.append({"coverage": threshold, "baskets": k, "rows": total})
    cr_sizes = [len(c) for c in all_cr.values()]
    singleton = sum(1 for n in counts_desc if n == 1)

    report = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "TRAIN-only basket vocabulary feasibility; no model prediction",
        "dataset": str(generation),
        "split": "train",
        "export_fingerprint": manifest["export_fingerprint"],
        "restrictions": {
            "validation_or_test_payloads_read": False,
            "model_evaluated": False,
            "observed_support_is_not_prediction_accuracy": True,
            "candidate_legality_not_checked": True,
            "own_train_holdout_not_unseen_to_existing_models": True,
        },
        "population": {
            "rows": rows, "purchases": purchases, "save_rows": save,
            "games": len(observed), "unique_whole_baskets": len(vocabulary),
            "unique_items": len(items),
            "duplicate_item_purchase_rows": duplicate_rows,
            "basket_size_counts": {str(k): sizes[k] for k in sorted(sizes)},
            "label_count_rows": label_count_rows,
            "rows_with_sequence_label_keys": ordered_labels,
        },
        "whole_basket_concentration": thresholds,
        "singleton_baskets": {"types": singleton, "rows": singleton},
        "champion_role": {
            "strata": len(all_cr),
            "basket_vocabulary_p50": quantile(cr_sizes, 0.5),
            "basket_vocabulary_p90": quantile(cr_sizes, 0.9),
            "basket_vocabulary_max": max(cr_sizes),
        },
        "internal_temporal_probe": {
            "definition": ("Earliest 80% TRAIN games fit; remaining TRAIN "
                           "games probe; timestamp ties kept together"),
            "cutoff_ms": boundary,
            "fit_games": len(fit_games), "probe_games": len(probe_games),
            "fit_rows": fit_rows, "probe_rows": probe_rows,
            "fit_purchases": fit_purchases,
            "probe_purchases": len(probe_rows_list),
            "fit_max_creation_ms": max(creation[g] for g in fit_games),
            "probe_min_creation_ms": min(creation[g] for g in probe_games),
            "fit_unique_baskets": len(fit_vocab),
            "support_counts": support,
            "by_basket_size": {str(k): by_size[k] for k in sorted(by_size)},
            "candidate_shortlists": [
                {**cfg,
                 "by_size": {str(k): cfg["by_size"][k]
                             for k in sorted(cfg["by_size"])},
                 "coverage": cfg["hits"] / len(probe_rows_list),
                 "mean_candidates": cfg["total_candidates"] / len(probe_rows_list),
                 "definition": ("Union of top k fit baskets in champion+role, "
                                "champion+role+inventory multiset, "
                                "champion+role+250g band+5min band; "
                                "no truth used in retrieval")}
                for cfg in configs],
        },
        "item_copy_support": {
            "min": min(items.values()),
            "items_below_10": [{"item_id": i, "copies": n}
                               for i, n in items.items() if n < 10],
            "items_below_100": sum(1 for n in items.values() if n < 100),
        },
        "bindings": {
            "fit_games_n": len(fit_games),
            "fit_games_sha256": membership_digest(fit_games),
            "probe_games_n": len(probe_games),
            "probe_games_sha256": membership_digest(probe_games),
            "game_digest_encoding": ("creation-ascending then ID-ascending; "
                                     "one ID per LF-terminated line; UTF-8"),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    sample = [{"row": r["sample_row"], "global_seen": r["b"] in fit_vocab,
               "shortlist_membership": r["sample_membership"]}
              for r in probe_rows_list if "sample_row" in r]
    with open(out_dir / "basket_probe_legality_sample_train.jsonl", "w",
              encoding="utf-8", newline="\n") as fh:
        for entry in sample:
            fh.write(json.dumps(entry, separators=(",", ":"),
                                ensure_ascii=False) + "\n")
    report["legality_probe_sample"] = {
        "n": len(sample),
        "path": "basket_probe_legality_sample_train.jsonl",
        "rule": ("sha256(match_id|participant_id|ts).readUInt32BE(0) % 20 "
                 "== 0; internal chronological probe purchases only"),
    }
    (out_dir / "basket_vocabulary_probe.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default=str(
        ROOT / "data/dataset_generations/16.18-reconstructible-development-v3"))
    ap.add_argument("--out", required=True,
                    help="output directory for report + legality sample")
    args = ap.parse_args()
    report = run(Path(args.generation), Path(args.out))
    p = report["internal_temporal_probe"]
    print(f"purchases={report['population']['purchases']} "
          f"baskets={report['population']['unique_whole_baskets']} "
          f"probe global_seen={p['support_counts']['global_seen']}"
          f"/{p['probe_purchases']}")


if __name__ == "__main__":
    main()
