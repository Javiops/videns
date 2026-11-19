"""Latency of the PACKAGE ADAPTER, separating cold load from warm requests.

`scripts/basket_latency_bench.py` measures the experimental circuits
assembled inline (parent / tuned / ward-free). That answers "what does the
mixture add over a single model". It does NOT answer "what does serving this
package cost", because serving goes through `app/mixture_policy.py` with a
package identified by a digest.

So this measures the adapter, and separates the two things a caller
experiences differently:

  COLD  - constructing MixturePolicy: digest verification of every
          component, vocabularies, catalogues, tensors, three state_dicts,
          architecture re-inspection, basket vector precomputation. Paid
          once per process, never per request.
  WARM  - policy.options([row]) with the package already loaded: the actual
          per-request cost.

Per-decision records are written with timestamps, re-read, counted, and the
aggregates are RECOMPUTED from them before the report is published — the
previous benchmark reported aggregates whose backing records were never
written at all.

Run DIRECTLY (the runtime must be activated before train_prefix is
imported):

    .venv\\Scripts\\python.exe scripts\\mixture_package_latency.py \\
        --package mixture-50-50-v1 [--requests 400] [--cold 5]
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

REVIEW = ROOT / "data/corpora/16.18-reconstructible-v3/static_review/static_review.json"
OUT_DIR = ROOT / "data" / "reports" / "mixture_displayed_policy"


def stats_ms(xs):
    xs = sorted(xs)
    return {"n": len(xs),
            "median_ms": round(statistics.median(xs), 3),
            "mean_ms": round(statistics.fmean(xs), 3),
            "p90_ms": round(xs[int(0.9 * (len(xs) - 1))], 3),
            "p99_ms": round(xs[int(0.99 * (len(xs) - 1))], 3),
            "min_ms": round(xs[0], 3), "max_ms": round(xs[-1], 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="mixture-50-50-v1")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cold", type=int, default=5)
    ap.add_argument("--review", default=None, help="static review to activate the runtime with (default: v3 corpus review)")
    ap.add_argument("--generation", default=None, help="generation whose FIT rows feed the warm requests (default: v3)")
    args = ap.parse_args()
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkg_dir = ROOT / "data" / "packages" / args.package

    import app.mixture_policy as mp
    active = mp.activate_runtime(pkg_dir, Path(args.review) if args.review else REVIEW)

    import torch
    import train_basket_ab as v1
    from basket_vocabulary_probe import split_fit_probe

    # ---- COLD: constructing the policy, repeated ----
    cold = []
    from app.ddragon import dragon_for_patch
    manifest = json.loads((pkg_dir / "manifest.json").read_text(encoding="utf-8"))
    dragon = dragon_for_patch(manifest["generation"]["patch"])
    for i in range(args.cold):
        t0 = time.perf_counter()
        policy = mp.MixturePolicy(pkg_dir, dragon, verify=True)
        if policy.device.type == "cuda":
            torch.cuda.synchronize()
        cold.append((time.perf_counter() - t0) * 1000.0)
        print(f"  carga en frio {i + 1}/{args.cold}: {cold[-1]:.0f} ms",
              flush=True)

    # ---- rows for the warm requests (evaluation-time source) ----
    generation = Path(args.generation) if args.generation else ROOT / "data/dataset_generations/16.18-reconstructible-development-v3"
    if (generation / "split_assignment.json").exists():          # v4-style generation: explicit assignment
        assignment = json.loads((generation / "split_assignment.json").read_text(encoding="utf-8"))
        fit_games = sorted(m for m, f in assignment.items() if f == "train")
    else:
        plan = json.loads((generation / "split_plan.json").read_text(encoding="utf-8"))
        fit_games, _, _ = split_fit_probe(plan)
    g = torch.Generator().manual_seed(16)
    pool = list(v1.stream_split(generation / "visits_train.jsonl",
                                set(fit_games), True, policy.champ_index,
                                dragon, 4000))
    pick = torch.randperm(len(pool), generator=g)[
        :args.requests + args.warmup].tolist()
    rows = [pool[i] for i in pick]
    print(f"muestra FIT fija: {len(rows)} filas "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    sync = (torch.cuda.synchronize if policy.device.type == "cuda"
            else (lambda: None))

    def clock():
        sync()
        return time.perf_counter()

    raw = []
    for i, row in enumerate(rows):
        t0 = clock()
        opts = policy.options([row])
        t1 = clock()
        raw.append({"i": i, "is_warmup": i < args.warmup,
                    "ms": (t1 - t0) * 1000.0,
                    "n_options": len(opts[0]),
                    "first_kind": opts[0][0]["kind"],
                    "t_start_perf": t0, "t_end_perf": t1,
                    "utc": datetime.now(timezone.utc).isoformat()})
    warm = [r["ms"] for r in raw if not r["is_warmup"]]

    rawpath = OUT_DIR / f"package_latency_raw_{args.package}.jsonl"
    with open(rawpath, "w", encoding="utf-8", newline=chr(10)) as fh:
        for r in raw:
            fh.write(json.dumps(r) + chr(10))
        fh.flush()
        os.fsync(fh.fileno())

    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "what": "latency of the PACKAGE ADAPTER (app/mixture_policy.py)",
        "package": args.package,
        "package_identity_sha256": policy.digest,
        "runtime_version": active["runtime_version"],
        "device": (torch.cuda.get_device_name(0)
                   if policy.device.type == "cuda" else "cpu"),
        "split": "fixed FIT sample (not VAL, not the selection probe, not test)",
        "cold_load_ms": stats_ms(cold),
        "cold_note": ("constructing MixturePolicy: digest verification of "
                      "every component, vocab, catalogues, tensors, three "
                      "state_dicts, architecture re-inspection, basket vector "
                      "precomputation. Paid ONCE per process, not per request."),
        "warm_request_ms": stats_ms(warm),
        "warm_note": ("policy.options([row]) with the package already loaded: "
                      "parent forward -> top-128 domain -> economic features "
                      "-> tuned scores -> ward-free forward -> permutation + "
                      "blend -> decode with legality. Batch size 1."),
        "requests": len(warm), "warmup_requests": args.warmup,
    }

    # ---- VERIFY THE RECORDS BEFORE CLOSING ----
    reread = [json.loads(l) for l in
              rawpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(reread) != len(raw):
        raise RuntimeError(f"registros incompletos: {len(reread)}/{len(raw)}")
    back = stats_ms([r["ms"] for r in reread if not r["is_warmup"]])
    for field in ("n", "median_ms", "mean_ms", "p90_ms", "p99_ms"):
        if back[field] != report["warm_request_ms"][field]:
            raise RuntimeError(
                f"los registros no reproducen warm_request_ms.{field}: "
                f"{back[field]} vs {report['warm_request_ms'][field]}")
    report["raw_measurements"] = {
        "file": str(rawpath.relative_to(ROOT)),
        "sha256": hashlib.sha256(rawpath.read_bytes()).hexdigest(),
        "rows": len(reread),
        "warmup_rows": sum(1 for r in reread if r["is_warmup"]),
        "measured_rows": sum(1 for r in reread if not r["is_warmup"]),
        "aggregates_reproduced_from_records": True,
        "fields_checked": ["n", "median_ms", "mean_ms", "p90_ms", "p99_ms"],
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = OUT_DIR / f"package_latency_{args.package}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"registros verificados: {len(reread)} filas, agregados "
          f"reproducidos desde disco", flush=True)
    print(json.dumps({"cold_load_ms": report["cold_load_ms"],
                      "warm_request_ms": report["warm_request_ms"]}, indent=1),
          flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
