"""Build FIT/probe tensor caches that ACTUALLY contain purchase history.

Protocol: docs/experiment_history_contrast_v1.md, blocker 1.

The basket line loads caches by fixed filename, and `from_cache` used to
validate nothing: setting PREFIX_HISTORY=1 and reloading
`cache_fit_full_v2.pt` yields an all-zero `hist` and two identical arms,
silently. Measured on that file: 238,459 x 12 = 2,861,508 entries, all
zero. So history needs its own cache, written with a fingerprint.

This script turns `train_prefix.USE_HISTORY` on for the build, writes
`cache_fit_hist_v1.pt` / `cache_probe_hist_v1.pt` with a `__meta__` block,
and refuses to finish unless the history tensor is genuinely populated. It
also reports what the history actually looks like, because "non-zero" is
not the same as "useful": the share of decisions with any history at all,
the mean filled slots, and the share saturating HIST_LEN.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

import train_basket_ab as v1
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe

OUT_DIR = ROOT / "data" / "experiments" / "16.18-history-contrast-v1"


def describe(hist):
    """What the history actually contains -- not just 'non-zero'."""
    filled = (hist != 0).sum(dim=1)
    n = hist.shape[0]
    return {
        "rows": int(n),
        "slots": int(hist.shape[1]),
        "nonzero_entries": int((hist != 0).sum()),
        "rows_with_any_history": int((filled > 0).sum()),
        "share_with_any_history": float((filled > 0).float().mean()),
        "mean_filled_slots": float(filled.float().mean()),
        "share_saturating_hist_len": float((filled == hist.shape[1])
                                           .float().mean()),
        "distinct_items_seen": int(torch.unique(hist[hist != 0]).numel()),
    }


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

    # THE point of this script -- and the place a previous draft of it went
    # wrong. Setting tp.USE_HISTORY alone leaves extras/gold_est/gold_x/runes/
    # pov_gold at module defaults, which changes the QUERY tensor: the cache
    # then belongs to a regime the frozen backbone was never trained in, and
    # nothing crashes (apply_config's own docstring warns about exactly this).
    # So take the backbone's OWN config and flip the single intended flag.
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    base_config = torch.load(ref_artifact, map_location="cpu",
                             weights_only=False)["config"]
    config = dict(base_config)
    assert config["history"] is False, "la referencia ya traia historial"
    config["history"] = True                      # the one intended change
    tp.apply_config(config)
    assert tp.USE_HISTORY and tp.QUERY_DIM == base_config["query_dim"]
    featurization = {k: config[k] for k in
                     ("extras", "history", "gold_est", "gold_x", "runes",
                      "pov_gold", "query_dim", "max_items", "max_copies",
                      "max_basket_items", "inventory_encoding")}
    print(f"featurizacion tomada de {ref_artifact.name}: "
          f"{json.dumps(featurization)}", flush=True)

    plan = json.loads((generation / "split_plan.json").read_text(
        encoding="utf-8"))
    fit_games, probe_games, _ = split_fit_probe(plan)
    assert membership_digest(fit_games) == v1.FROZEN_FIT_SHA
    assert membership_digest(probe_games) == v1.FROZEN_PROBE_SHA
    fit_set = set(fit_games)

    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
    champ_ids, item_ids, label_counts, catalogue, n_fit, n_probe = \
        v1.first_pass(train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets).encode()).hexdigest()
    cap_fit = 3000 if args.smoke else None
    cap_probe = 1000 if args.smoke else None
    if args.smoke:
        n_fit, n_probe = min(n_fit, cap_fit), min(n_probe, cap_probe)
    print(f"setup listo ({time.monotonic() - started:.0f}s); "
          f"USE_HISTORY={tp.USE_HISTORY} HIST_LEN={tp.HIST_LEN}; "
          f"fit {n_fit} probe {n_probe}", flush=True)

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/experiment_history_contrast_v1.md",
              "mode": stamp, "hist_len": tp.HIST_LEN,
              "baskets_sha256": baskets_sha, "splits": {},
              "featurization": featurization,
              "featurization_source": str(ref_artifact.relative_to(ROOT))}
    meta_extra = {"built_by": "scripts/build_history_cache.py",
                  "baskets_sha256": baskets_sha,
                  "generation": generation.name,
                  "featurization_source": str(ref_artifact.relative_to(ROOT))}
    # Always the FULL reference, compared by prefix: it is the one built with
    # the full vocabulary and the backbone's own featurization. The smoke
    # reference is useless here -- train_basket_ab's smoke caps first_pass at
    # 20,000 rows, so its champ/item/label indices are a different vocabulary.
    reference_cache = {
        "fit": ROOT / "data/experiments/16.18-whole-basket-v1/cache_fit_full_v2.pt",
        "probe": ROOT / "data/experiments/16.18-whole-basket-v1/cache_probe_full_v2.pt",
    }

    for split, want_fit, n, cap in (("fit", True, n_fit, cap_fit),
                                    ("probe", False, n_probe, cap_probe)):
        t0 = time.monotonic()
        ds = tp.ShopDataset.from_stream(
            v1.stream_split(train_path, fit_set, want_fit, champ_index,
                            dragon, cap),
            n, champ_index, item_index, label_index, dragon)
        stats = describe(ds.hist)
        # The gate this whole script exists for.
        if stats["nonzero_entries"] == 0:
            raise RuntimeError(
                f"{split}: la cache saldria con hist TODO A CEROS "
                "-- USE_HISTORY no llego a la featurizacion")
        path = OUT_DIR / f"cache_{split}_hist_v1{'_smoke' if args.smoke else ''}.pt"
        ds.save_cache(path, meta={**meta_extra, "split": split})
        # read it back through the validating path, as a consumer would
        back = tp.ShopDataset.from_cache(path, champ_index, item_index,
                                         label_index, dragon,
                                         expect={"history": True,
                                                 "hist_len": tp.HIST_LEN})
        assert int((back.hist != 0).sum()) == stats["nonzero_entries"]
        assert back.cache_meta["hist_nonzero"] == stats["nonzero_entries"]
        # The invariant that actually closes this: the history cache must be
        # IDENTICAL to the standard one in every tensor except `hist`. If any
        # other tensor moved, the featurization drifted and the frozen
        # backbone would be scored in a regime it never saw.
        same_except_hist = None
        ref_path = reference_cache[split]
        if ref_path.exists():
            ref_blob = torch.load(ref_path, map_location="cpu",
                                  weights_only=False)
            # the reference may hold MORE rows (different smoke caps); both
            # come from the same stream in the same order, so compare the
            # prefix. Row count itself is not the invariant -- featurization is.
            per_tensor = {}
            for name in tp.ShopDataset.TENSOR_ATTRS:
                a, b = getattr(ds, name), ref_blob[name]
                if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                    # non-tensor attrs (e.g. `inventories` is a list): compare
                    # the same prefix, or a shorter cache always "differs"
                    try:
                        per_tensor[name] = list(a) == list(b)[:len(a)]
                    except TypeError:
                        per_tensor[name] = a == b
                    continue
                if b.shape[0] < a.shape[0] or b.shape[1:] != a.shape[1:]:
                    per_tensor[name] = False
                    continue
                per_tensor[name] = bool(torch.equal(a, b[:a.shape[0]]))
            differing = [k for k, v in per_tensor.items() if not v]
            same_except_hist = (differing == ["hist"])
            if not same_except_hist:
                # In --smoke the REFERENCE is the weak side: train_basket_ab's
                # smoke builds its vocabulary from first_pass capped at 20,000
                # FIT rows, so its champ/item/label indices differ from the
                # full ones used here and every index tensor moves. That is a
                # property of the reference, not of this cache. In full mode
                # both use the full vocabulary and the invariant is exact, so
                # it is enforced there and only reported here.
                msg = ("la cache de historial difiere de la estandar en mas "
                       f"que `hist`: {differing}")
                raise RuntimeError(msg + " -- la featurizacion se ha desviado")
            else:
                print(f"  {split}: identica a la cache estandar salvo `hist` "
                      f"OK (prefijo de {ds.champs.shape[0]} filas)", flush=True)
            del ref_blob
        report["splits"][split] = {
            "path": str(path.relative_to(ROOT)),
            "identical_to_standard_cache_except_hist": same_except_hist,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seconds": round(time.monotonic() - t0, 1),
            "history": stats,
            "meta": back.cache_meta,
        }
        print(f"  {split}: {stats['rows_with_any_history']}/{stats['rows']} "
              f"filas con historial ({100 * stats['share_with_any_history']:.1f}%), "
              f"media {stats['mean_filled_slots']:.2f} huecos de {tp.HIST_LEN}, "
              f"{100 * stats['share_saturating_hist_len']:.1f}% saturan "
              f"({report['splits'][split]['seconds']}s)", flush=True)
        del ds, back

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"history_cache_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["history"] for k, v in report["splits"].items()},
                     indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
