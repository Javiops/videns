"""Common vocabulary + featurisation caches for the prior-patch transfer (protocol v2, section 2).

Both arms, both phases, ONE vocabulary built exclusively from the two FITs
(16.17 history rows and the v4 16.18 FIT): champion index, item index, label
index and the basket encoder's item table (the union of both catalogues).
Each phase keeps its own ACTION catalogue. Nothing is re-indexed later; the
tables simply contain both patches' IDs.

Two runs, each under its own isolated runtime so every row is featurised
and labelled with the rules of its patch:
  --patch 16.17  (activates reconstructible_1617_runtime FIRST; validates the
                  history corpus with history_1617.validate_history; writes
                  cache_fit_1617.pt + targets + catalogue_1617.json)
  --patch 16.18  (run via footwear_runtime with PREFIX_ML_DIR = v4 generation
                  and the v4 static review; writes cache_fit_1618.pt, targets,
                  cache_cohort.pt, catalogue_1618.json)
Both write vocab.json with the same digest (asserted on the second run) and
each cache carries that digest in its meta, so the trainer refuses a cache
built with another vocabulary.
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

OUT_DIR = ROOT / "data" / "experiments" / "16.18-prior-patch-transfer-v1"
GEN_1618 = ROOT / "data" / "dataset_generations" / "16.18-reconstructible-development-v4"
STAGE1 = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v4-bf-stage1"
SMOKE_FIT, SMOKE_COHORT = 3000, 1500


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def common_vocabulary(hist_rows_path, hist_games, gen_train_path, train_set, v1, tp):
    """One pass over each FIT (first_pass reads only raw ids/labels; no dragon involved)."""
    c17, i17, l17, cat17, n17, _ = v1.first_pass(hist_rows_path, hist_games, None, None)
    c18, i18, l18, cat18, n18, _ = v1.first_pass(gen_train_path, train_set, None, None)
    champ_ids = sorted(set(c17) | set(c18))
    item_ids = sorted(set(i17) | set(i18))
    labels = sorted(set(l17) | set(l18))
    champ_index, item_index = tp._index_map(champ_ids), tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(labels)}
    baskets17 = [k for k, _ in sorted(cat17.items(), key=lambda kv: (-kv[1], kv[0]))]
    baskets18 = [k for k, _ in sorted(cat18.items(), key=lambda kv: (-kv[1], kv[0]))]
    basket_items = sorted({int(p.split(":")[0]) for b in baskets17 + baskets18 for p in b.split(",")})
    vocab = {"champ_index": {str(k): v for k, v in champ_index.items()}, "item_index": {str(k): v for k, v in item_index.items()},
             "label_index": {str(k): v for k, v in label_index.items()}, "basket_items": basket_items,
             "sources": {"history_rows": str(hist_rows_path), "history_games": len(hist_games), "history_fit_rows": n17,
                         "generation_train": str(gen_train_path), "generation_games": len(train_set), "generation_fit_rows": n18},
             "counts": {"champions": len(champ_ids), "items": len(item_ids), "labels": len(labels), "basket_items": len(basket_items),
                        "champions_1617_only": len(set(c17) - set(c18)), "champions_1618_only": len(set(c18) - set(c17)),
                        "items_1617_only": len(set(i17) - set(i18)), "items_1618_only": len(set(i18) - set(i17)),
                        "labels_1617_only": len(set(l17) - set(l18)), "labels_1618_only": len(set(l18) - set(l17))}}
    digest = hashlib.sha256(json.dumps({k: vocab[k] for k in ("champ_index", "item_index", "label_index", "basket_items")},
                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    vocab["digest"] = digest
    return vocab, champ_index, item_index, label_index, baskets17, baskets18, n17, n18


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", choices=("16.17", "16.18"), required=True)
    ap.add_argument("--smoke", action="store_true")
    from history_1617 import HISTORY_DIR, validate_history
    ap.add_argument("--history", type=Path, default=HISTORY_DIR, help="validated history export (smoke chains use their own)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="cache directory (smoke chains use their own)")
    args = ap.parse_args()
    OUT_DIR = args.out.resolve()
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = "smoke" if args.smoke else "full"
    hist = validate_history(args.history, scan_rows=not args.smoke, hash_db=not args.smoke)
    from history_rows_check import validate_rows
    validate_rows(args.history, hist)                 # rows vs the frozen corpus + reviewed row-builder digests
    if hist["acceptance"].get("selection_limit") and not args.smoke:
        raise SystemExit(f"{args.history} comes from a --limit snapshot ({hist['acceptance']['selection_limit']} sources); smoke only")
    hist_rows = Path(args.history).resolve() / hist["rows"]["file"]
    hist_games = set(hist["games"]["exported"])
    print(f"history corpus validated: {hist['games']['exported_count']} games, {hist['rows']['count']} rows ({time.monotonic() - started:.0f}s)", flush=True)

    if args.patch == "16.17":
        import reconstructible_1617_runtime as runtime      # FIRST, clean process
        plan = json.loads((Path(hist["acceptance"]["path"]) / "source_plan.json").read_text(encoding="utf-8"))
        decl = runtime.activate(Path(plan["static_review"]))   # the review the validated history binds
        assert decl == hist["footwear_runtime"], "history runtime differs from the active 16.17 runtime"
    # (16.18: footwear_runtime.py already activated the benchmark runtime before this entry ran)
    import torch
    import train_basket_ab as v1
    import train_prefix as tp
    from basket_vocabulary_probe import basket_key
    from app.ddragon import dragon_for_patch
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(GEN_1618)
    schema = json.loads((GEN_1618 / "split_manifest.json").read_text(encoding="utf-8"))["input_schema"]
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    config = torch.load(ref_artifact, map_location="cpu", weights_only=False)["config"]
    config = {**config, "max_items": schema["max_modeled_inventory_items"], "max_copies": schema["max_copies_of_one_item"],
              "max_basket_items": schema["max_basket_items"]}
    tp.apply_config(config)
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    assert not tp.USE_HISTORY and tp.D_MODEL == 64
    assignment = json.loads((GEN_1618 / "split_assignment.json").read_text(encoding="utf-8"))
    train_set = {m for m, f in assignment.items() if f == "train"}
    vocab, champ_index, item_index, label_index, baskets17, baskets18, n17, n18 = common_vocabulary(
        hist_rows, hist_games, GEN_1618 / "visits_train.jsonl", train_set, v1, tp)
    vocab_path = OUT_DIR / "vocab.json"
    if vocab_path.exists():
        prior = json.loads(vocab_path.read_text(encoding="utf-8"))
        assert prior["digest"] == vocab["digest"], "the common vocabulary differs between the two runs"
    else:
        vocab_path.write_text(json.dumps(vocab, indent=1, sort_keys=True), encoding="utf-8")
    print(f"vocabulario comun {vocab['digest'][:12]}: {vocab['counts']} ({time.monotonic() - started:.0f}s)", flush=True)
    expect = {"history": False, "extras": True, "gold_est": True, "gold_x": True, "runes": True, "pov_gold": True,
              "vocab_sha256": vocab["digest"]}

    if args.patch == "16.17":
        dragon = dragon_for_patch("16.17")
        baskets = baskets17
        basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
        (OUT_DIR / f"catalogue_1617_{stamp}.json").write_text(json.dumps({"baskets": baskets, "n_fit_rows": n17, "patch": "16.17",
                                                                            "vocab_sha256": vocab["digest"]}), encoding="utf-8")
        cap = SMOKE_FIT if args.smoke else None
        n = min(n17, cap) if cap else n17
        ds = tp.ShopDataset.from_stream(v1.stream_split(hist_rows, hist_games, True, champ_index, dragon, cap), n,
                                        champ_index, item_index, label_index, dragon)
        targets = torch.zeros(n, dtype=torch.long)
        for i, row in enumerate(v1.stream_split(hist_rows, hist_games, True, champ_index, dragon, cap)):
            t = v1.truth_of(row)
            targets[i] = 0 if t is None else basket_pos[basket_key(sorted(t.items()))]
        ds.save_cache(OUT_DIR / f"cache_fit_1617_{stamp}.pt", meta={**expect, "patch": "16.17", "split": "fit",
                                                                     "history_rows_sha256": hist["rows"]["sha256"],
                                                                     "catalogue_sha256": hashlib.sha256("|".join(baskets).encode()).hexdigest()})
        torch.save(targets, OUT_DIR / f"targets_fit_1617_{stamp}.pt")
        print(f"cache 16.17: {n} filas, catalogo {len(baskets)} cestas ({time.monotonic() - started:.0f}s)", flush=True)
    else:
        dragon = dragon_for_patch(manifest_gen["patch"])
        baskets = baskets18
        stage1 = json.loads((STAGE1 / "bf_stage1_report_full.json").read_text(encoding="utf-8"))
        assert len(baskets) == stage1["catalogue"], "the 16.18 catalogue differs from stage 1"
        basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
        (OUT_DIR / f"catalogue_1618_{stamp}.json").write_text(json.dumps({"baskets": baskets, "n_fit_rows": n18, "patch": "16.18",
                                                                            "vocab_sha256": vocab["digest"]}), encoding="utf-8")
        cap = SMOKE_FIT if args.smoke else None
        n = min(n18, cap) if cap else n18
        train_path = GEN_1618 / "visits_train.jsonl"
        ds = tp.ShopDataset.from_stream(v1.stream_split(train_path, train_set, True, champ_index, dragon, cap), n,
                                        champ_index, item_index, label_index, dragon)
        targets = torch.zeros(n, dtype=torch.long)
        for i, row in enumerate(v1.stream_split(train_path, train_set, True, champ_index, dragon, cap)):
            t = v1.truth_of(row)
            targets[i] = 0 if t is None else basket_pos[basket_key(sorted(t.items()))]
        ds.save_cache(OUT_DIR / f"cache_fit_1618_{stamp}.pt", meta={**expect, "patch": "16.18", "split": "fit",
                                                                     "catalogue_sha256": hashlib.sha256("|".join(baskets).encode()).hexdigest()})
        torch.save(targets, OUT_DIR / f"targets_fit_1618_{stamp}.pt")
        cap_val = SMOKE_COHORT if args.smoke else None
        val_rows = list(v1.stream_split(GEN_1618 / "visits_val.jsonl", set(), False, champ_index, dragon, cap_val))
        vds = tp.ShopDataset.from_stream(iter(val_rows), len(val_rows), champ_index, item_index, label_index, dragon)
        vds.save_cache(OUT_DIR / f"cache_cohort_{stamp}.pt", meta={**expect, "patch": "16.18", "split": "cohort"})
        digest = hashlib.sha256()
        for r in val_rows:
            digest.update(f"{r['match_id']}|{r['ts']};".encode())
        (OUT_DIR / f"cohort_rows_{stamp}.json").write_text(json.dumps({"n": len(val_rows), "rows_sha256": digest.hexdigest()}), encoding="utf-8")
        print(f"cache 16.18: fit {n} filas, cohorte {len(val_rows)}, catalogo {len(baskets)} ({time.monotonic() - started:.0f}s)", flush=True)
    (OUT_DIR / f"prepare_{args.patch.replace('.', '')}_{stamp}.json").write_text(json.dumps({
        "patch": args.patch, "mode": stamp, "created_utc": datetime.now(timezone.utc).isoformat(), "vocab_sha256": vocab["digest"],
        "rows": n, "catalogue": len(baskets), "elapsed_seconds": round(time.monotonic() - started, 1)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
