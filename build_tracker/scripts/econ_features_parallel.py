"""Parallel econ_features across cores, with a bit-identity equivalence gate.

`econ_features` is a pure function of (action, row, budget, dragon): no
randomness, no global mutable state (the `counters` argument only carries
two order-independent diagnostic keys, and the lru_caches inside
shop_econ / conditional_policy are per-process speed, not semantics). It is
therefore safe to shard by row -- PROVIDED every worker is configured
exactly like the parent.

That proviso is the whole risk. Windows spawns workers as fresh
interpreters, so each one must redo the footwear activation with the
explicit static review, the train_prefix schema fields and its own
DataDragon. A worker that skipped any of it would not crash: it would
return plausible, different features. This repo's failure mode exactly.

So nothing here is usable until `--verify` passes, and `--verify` demands
BOTH:
  1. every worker reports the same runtime declaration, Data Dragon
     version and schema as the parent, and
  2. parallel output is BIT-IDENTICAL to serial output on a fixed sample
     (torch.equal, not allclose).

Modes:
  --verify    the equivalence gate (default sample 512 FIT rows)
  --bench     serial vs parallel throughput at several worker counts
  --profile   cProfile of the SERIAL path, to see where the time goes
              and test the duplicated-purchase_sequence hypothesis

Run via footwear_runtime with PREFIX_ML_DIR set.
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
if str(ROOT) not in sys.path:
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

OUT_DIR = ROOT / "data" / "reports" / "econ_features_parallel"

# ---------------------------------------------------------------- worker ---
_W = {}


def _init_worker(generation_str, review_str, schema, baskets):
    """Spawned workers start blank: rebuild the parent's whole configuration.

    Anything missed here yields plausible-but-different features, so the
    worker also reports what it ended up with and the parent compares.
    """
    import train_prefix as tp
    from app.dataset_generations import verify_generation
    import footwear_runtime as fr

    generation = Path(generation_str)
    manifest = verify_generation(generation)
    active = fr.activate_for_manifest(manifest, review_path=Path(review_str))
    tp.MAX_ITEMS = schema["max_modeled_inventory_items"]
    tp.MAX_COPIES = schema["max_copies_of_one_item"]
    tp.MAX_BASKET_ITEMS = schema["max_basket_items"]
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    from app.ddragon import dragon_for_patch
    _W["dragon"] = dragon_for_patch(manifest["patch"])
    _W["baskets"] = baskets
    _W["counters"] = Counter()
    _W["fingerprint"] = {
        "pid": os.getpid(),
        "active_runtime": active,
        "patch": manifest["patch"],
        "ddragon_version": getattr(_W["dragon"], "version", None),
        "max_items": tp.MAX_ITEMS, "max_copies": tp.MAX_COPIES,
        "max_basket_items": tp.MAX_BASKET_ITEMS,
        "inventory_encoding": tp.INVENTORY_ENCODING,
        "baskets_sha256": hashlib.sha256(
            "|".join(baskets).encode()).hexdigest(),
    }


def _fingerprint(_ignored):
    return _W["fingerprint"]


def _chunk(job):
    """job = list of (row, domain_ids). Returns list of flat feature lists."""
    from basket_econ_cache import N_ECON, econ_features, to_action
    dragon = _W["dragon"]
    baskets = _W["baskets"]
    counters = _W["counters"]
    out = []
    for row, domain in job:
        budget = float(row["gold_est"])
        per_row = []
        for cand in domain:
            if cand == 0:
                feats = [0.0] * N_ECON
                feats[0] = 1.0
            else:
                feats = econ_features(to_action(baskets[cand - 1]), row,
                                      budget, dragon, counters)
            per_row.append(feats)
        out.append(per_row)
    return out, dict(counters)


# ---------------------------------------------------------------- parent ---
class EconFeaturePool:
    """Warm pool of configured workers. Verify before trusting it."""

    def __init__(self, workers, generation, review, schema, baskets):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.workers = workers
        self.pool = ctx.Pool(
            processes=workers, initializer=_init_worker,
            initargs=(str(generation), str(review), schema, baskets))

    def fingerprints(self):
        return self.pool.map(_fingerprint, range(self.workers * 4))

    def features(self, rows, domains, n_econ, shortlist, chunk=16):
        jobs = [list(zip(rows[i:i + chunk], domains[i:i + chunk]))
                for i in range(0, len(rows), chunk)]
        out = torch.zeros((len(rows), shortlist, n_econ))
        counters = Counter()
        pos = 0
        for block, ctr in self.pool.imap(_chunk, jobs):
            counters.update(ctr)
            for per_row in block:
                out[pos] = torch.tensor(per_row)
                pos += 1
        assert pos == len(rows)
        return out, counters

    def close(self):
        self.pool.close()
        self.pool.join()


def serial_features(rows, domains, baskets, dragon, n_econ, shortlist):
    from basket_econ_cache import econ_features, to_action
    out = torch.zeros((len(rows), shortlist, n_econ))
    counters = Counter()
    for j, (row, domain) in enumerate(zip(rows, domains)):
        budget = float(row["gold_est"])
        for k, cand in enumerate(domain):
            if cand == 0:
                out[j, k, 0] = 1.0
            else:
                out[j, k] = torch.tensor(econ_features(
                    to_action(baskets[cand - 1]), row, budget, dragon,
                    counters))
    return out, counters


def build_context(args):
    """Everything the parent needs; mirrors the sibling experiment scripts."""
    import train_basket_ab as v1
    import train_prefix as tp
    from basket_vocabulary_probe import membership_digest, split_fit_probe

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(
        encoding="utf-8"))["input_schema"]
    tp.MAX_ITEMS = schema["max_modeled_inventory_items"]
    tp.MAX_COPIES = schema["max_copies_of_one_item"]
    tp.MAX_BASKET_ITEMS = schema["max_basket_items"]
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    plan = json.loads((generation / "split_plan.json").read_text(
        encoding="utf-8"))
    fit_games, probe_games, _ = split_fit_probe(plan)
    assert membership_digest(fit_games) == v1.FROZEN_FIT_SHA
    assert membership_digest(probe_games) == v1.FROZEN_PROBE_SHA
    fit_set = set(fit_games)
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
    champ_ids, _, _, catalogue, _, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    rows = []
    for row in v1.stream_split(train_path, fit_set, True, champ_index,
                               dragon, args.sample):
        rows.append(row)
    # Domains must be the REAL frozen top-128, not a random draw: a random
    # basket is infeasible far more often than a model's shortlist, and
    # econ_features returns early when it is. Random ids would leave the
    # equivalence gate valid but make the benchmark optimistic.
    from basket_econ_cache import OUT_DIR as CACHE_DIR
    cache = CACHE_DIR / "frozen_fit_full.pt"
    if cache.exists():
        blob = torch.load(cache, weights_only=True)
        domains = blob["top_idx"][:len(rows)].long().tolist()
        del blob
        domain_source = "frozen_fit_full.pt top_idx (real parent shortlist)"
    else:
        g = torch.Generator().manual_seed(16)
        n_cat = len(baskets) + 1
        domains = [torch.randperm(n_cat, generator=g)[:v1.SHORTLIST].tolist()
                   for _ in rows]
        domain_source = ("SEEDED RANDOM fallback - the equivalence gate is "
                         "still valid but any timing here is OPTIMISTIC")
    assert len(domains) == len(rows)
    review = Path(os.environ.get(
        "PREFIX_RUNTIME_REVIEW",
        ROOT / "data/corpora/16.18-reconstructible-v3/static_review/"
               "static_review.json"))
    parent_fp = {
        "patch": manifest_gen["patch"],
        "ddragon_version": getattr(dragon, "version", None),
        "max_items": tp.MAX_ITEMS, "max_copies": tp.MAX_COPIES,
        "max_basket_items": tp.MAX_BASKET_ITEMS,
        "inventory_encoding": tp.INVENTORY_ENCODING,
        "baskets_sha256": hashlib.sha256(
            "|".join(baskets).encode()).hexdigest(),
    }
    return {"generation": generation, "review": review, "schema": schema,
            "baskets": baskets, "rows": rows, "domains": domains,
            "dragon": dragon, "parent_fingerprint": parent_fp,
            "shortlist": v1.SHORTLIST, "domain_source": domain_source}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sample", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bench-workers", default="2,4,6,8")
    args = ap.parse_args()
    if not (args.verify or args.bench or args.profile):
        args.verify = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from basket_econ_cache import N_ECON
    started = time.monotonic()
    ctx = build_context(args)
    rows, domains, baskets = ctx["rows"], ctx["domains"], ctx["baskets"]
    print(f"contexto listo ({time.monotonic() - started:.0f}s); "
          f"muestra {len(rows)} filas x {ctx['shortlist']} candidatos",
          flush=True)
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "sample_rows": len(rows), "shortlist": ctx["shortlist"],
              "n_econ": N_ECON, "parent_fingerprint": ctx["parent_fingerprint"],
              "domain_source": ctx["domain_source"],
              "machine": {"cpu_count": os.cpu_count()}}

    if args.profile:
        import cProfile
        import pstats
        import io as _io
        pr = cProfile.Profile()
        pr.enable()
        serial_features(rows, domains, baskets, ctx["dragon"], N_ECON,
                        ctx["shortlist"])
        pr.disable()
        buf = _io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
        text = buf.getvalue()
        (OUT_DIR / "serial_profile.txt").write_text(text, encoding="utf-8")
        report["profile"] = "data/reports/econ_features_parallel/serial_profile.txt"
        print(text[:2500], flush=True)

    if args.verify or args.bench:
        t0 = time.monotonic()
        ser, ser_ctr = serial_features(rows, domains, baskets, ctx["dragon"],
                                       N_ECON, ctx["shortlist"])
        serial_s = time.monotonic() - t0
        report["serial"] = {"seconds": round(serial_s, 2),
                            "rows_per_s": round(len(rows) / serial_s, 2),
                            "counters": dict(ser_ctr),
                            "tensor_sha256": hashlib.sha256(
                                ser.numpy().tobytes()).hexdigest()}
        print(f"serie: {serial_s:.1f}s ({len(rows) / serial_s:.1f} filas/s)",
              flush=True)

    if args.verify:
        pool = EconFeaturePool(args.workers, ctx["generation"], ctx["review"],
                               ctx["schema"], baskets)
        try:
            fps = pool.fingerprints()
            keys = [k for k in ctx["parent_fingerprint"]]
            mismatched = [fp for fp in fps
                          if any(fp.get(k) != ctx["parent_fingerprint"][k]
                                 for k in keys)]
            report["workers_seen"] = len({fp["pid"] for fp in fps})
            report["worker_config_matches_parent"] = not mismatched
            if mismatched:
                report["mismatched_worker_fingerprints"] = mismatched[:4]
            t0 = time.monotonic()
            par, par_ctr = pool.features(rows, domains, N_ECON,
                                         ctx["shortlist"])
            par_s = time.monotonic() - t0
        finally:
            pool.close()
        identical = torch.equal(ser, par)
        report["parallel"] = {
            "workers": args.workers, "seconds": round(par_s, 2),
            "rows_per_s": round(len(rows) / par_s, 2),
            "speedup": round(serial_s / max(par_s, 1e-9), 2),
            "counters": dict(par_ctr),
            "tensor_sha256": hashlib.sha256(
                par.numpy().tobytes()).hexdigest()}
        report["equivalence_gate"] = {
            "bit_identical": bool(identical),
            "max_abs_diff": float((ser - par).abs().max()),
            "counters_match": dict(ser_ctr) == dict(par_ctr),
            "worker_config_matches_parent": report["worker_config_matches_parent"],
            "verdict": ("PASS" if (identical and not mismatched
                                   and dict(ser_ctr) == dict(par_ctr))
                        else "FAIL"),
            "rule": ("bit-identical required (torch.equal), not allclose; "
                     "every worker must report the parent's runtime "
                     "declaration, Data Dragon version, schema and catalogue "
                     "digest; diagnostic counters must agree"),
        }
        print("PUERTA DE EQUIVALENCIA: "
              + json.dumps(report["equivalence_gate"], indent=1), flush=True)

    if args.bench:
        bench = {}
        for w in [int(x) for x in args.bench_workers.split(",") if x.strip()]:
            pool = EconFeaturePool(w, ctx["generation"], ctx["review"],
                                   ctx["schema"], baskets)
            try:
                pool.fingerprints()          # pay warm-up before timing
                t0 = time.monotonic()
                par, _ = pool.features(rows, domains, N_ECON, ctx["shortlist"])
                dt = time.monotonic() - t0
            finally:
                pool.close()
            ok = torch.equal(ser, par)
            bench[str(w)] = {"seconds": round(dt, 2),
                             "speedup": round(serial_s / max(dt, 1e-9), 2),
                             "bit_identical": bool(ok)}
            print(f"  {w} trabajadores: {dt:.1f}s "
                  f"(x{serial_s / max(dt, 1e-9):.2f}) identico={ok}", flush=True)
        report["bench"] = bench

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / "equivalence_and_bench.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
