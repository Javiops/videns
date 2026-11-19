"""Economic-feature cache on the v4 generation, parent = the new Bf (R1).

Protocol: docs/experiment_stage2_econ_wardfree_v1.md, segment A. Phases:
  --phase domains  [--cpu]: the frozen parent's top-128 per decision for FIT
      and for the cohort, persisted with row digests and the parent's digest.
  --phase gate     (GPU): re-decodes R1 on the cohort and must reproduce its
      stage-1 rows with 0 mismatches (integrity gate; exit 2 otherwise).
  --phase features (CPU only): the 10 economic features per candidate with
      the activated engine (purchase_sequence at gold_est), aligned to the
      domains by row digest, written in shards of --shard-rows rows.
      Sequential by default (bench-gated on 200 real rows against the
      declared 6 h bound, re-measured ten minutes apart before an abort).
      Parallel form (owner, 2026-09-26, "use more resources"): several
      workers, each `--shards i,j` computing only those shards (every worker
      still streams the whole FIT file and checks the full row digest), one
      `--cohort-only` worker, then `--merge`, which verifies every shard and
      writes the manifest + report. Bit-identical to the sequential build by
      construction (same function, same rows); the smoke asserts it.
"""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import Counter

import torch

from stage2_newgen import (OUT_DIR, EXIT_GATE, BOUNDS_H, setup, range_batch, group_logits, decode_batch,
                           reference_rows, rows_digest, now_utc, write_report, v1)
from basket_newbf_mixture import mismatches
from basket_econ_cache import N_ECON, econ_features, to_action
from winmem import available_ram_bytes

BENCH_ROWS, BENCH_WARMUP = 200, 5
SHARD_ROWS_DEFAULT = 100_000


def phase_domains(args):
    started = time.monotonic()
    S = setup(args.smoke, need_gpu=not args.cpu)
    stamp = S.stamp
    print(f"setup listo ({S.setup_seconds}s, {S.device}): fit {S.n_fit}, cohorte {S.n_val} / {len(S.cohort_games)} partidas, "
          f"catalogo {len(S.baskets)}, grupos {S.n_groups}", flush=True)
    top_fit = torch.empty((S.n_fit, v1.SHORTLIST), dtype=torch.int32)
    top_val = torch.empty((S.n_val, v1.SHORTLIST), dtype=torch.int32)
    t0 = time.monotonic()
    with torch.no_grad():
        vecs = S.model.baskets.vectors(S.cat_idx, S.cat_cnt, torch.arange(S.n_cat, device=S.device))
        for s in range(0, S.n_fit, 512):
            sl = slice(s, min(s + 512, S.n_fit))
            sc = S.model.scores(S.model.state(range_batch(S.fit_ds, sl), S.device), vecs)
            top_fit[sl] = torch.topk(sc, v1.SHORTLIST, dim=1).indices.int().cpu()
            if (s // 512) % 200 == 0 and s:
                print(f"  dominios fit {s}/{S.n_fit} ({time.monotonic() - t0:.0f}s)", flush=True)
        for s in range(0, S.n_val, 512):
            sl = slice(s, min(s + 512, S.n_val))
            sc = S.model.scores(S.model.state(range_batch(S.val_ds, sl), S.device), vecs)
            top_val[sl] = torch.topk(sc, v1.SHORTLIST, dim=1).indices.int().cpu()
    fit_digest, n_stream = rows_digest(v1.stream_split(S.train_path, S.train_set, True, S.champ_index, S.dragon,
                                                       S.n_fit if args.smoke else None))
    assert n_stream == S.n_fit, f"FIT stream {n_stream} vs cache {S.n_fit}"
    val_digest, _ = rows_digest(iter(S.val_rows))
    torch.save({"top_idx": top_fit, "n_rows": S.n_fit, "rows_sha256": fit_digest, "parent_sha256": S.ckpt_sha},
               OUT_DIR / f"domains_fit_{stamp}.pt")
    torch.save({"top_idx": top_val, "n_rows": S.n_val, "rows_sha256": val_digest, "parent_sha256": S.ckpt_sha},
               OUT_DIR / f"domains_cohort_{stamp}.pt")
    report = {"schema": 1, "phase": "domains", "mode": stamp, "created_utc": now_utc(), "device": str(S.device),
              "parent_sha256": S.ckpt_sha, "fit_rows": S.n_fit, "cohort_rows": S.n_val, "cohort_games": len(S.cohort_games),
              "shortlist": v1.SHORTLIST, "integrity_gate": "deferred to --phase gate (exact, on the GPU)",
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    write_report(OUT_DIR / f"domains_report_{stamp}.json", report)


def phase_gate(args):
    started = time.monotonic()
    S = setup(args.smoke, load_fit=False)
    stamp = S.stamp
    records = []
    with torch.no_grad():
        vecs = S.model.baskets.vectors(S.cat_idx, S.cat_cnt, torch.arange(S.n_cat, device=S.device))
        for s in range(0, S.n_val, 512):
            sl = slice(s, min(s + 512, S.n_val))
            sc = S.model.scores(S.model.state(range_batch(S.val_ds, sl), S.device), vecs)
            order = torch.topk(group_logits(sc, S.gid, S.n_groups).cpu(), min(v1.SHORTLIST, S.n_groups), dim=1).indices
            records.extend(decode_batch(order, S.val_rows[sl], S.group_main, S))
    bad = mismatches(records, reference_rows(S.n_val))
    gate = all(v == 0 for v in bad.values())
    print("reproduccion R1:", json.dumps(bad), "| gate", "OK" if gate else "FAILED", flush=True)
    report = {"schema": 1, "phase": "gate", "mode": stamp, "created_utc": now_utc(), "device": str(S.device),
              "parent_sha256": S.ckpt_sha, "cohort_rows": S.n_val, "r1_reproduction_mismatches": bad,
              "integrity_gate_passed": gate, "elapsed_seconds": round(time.monotonic() - started, 1)}
    write_report(OUT_DIR / f"gate_report_{stamp}.json", report)
    if not gate:
        sys.exit(EXIT_GATE)


# ----------------------------------------------------------------------------
def _features_context(args, need_cohort):
    S = setup(args.smoke, need_gpu=False, load_fit=False, load_cohort=need_cohort, cohort_tensors=False, load_model=False)
    assert S.device.type == "cpu"
    dom_fit = torch.load(OUT_DIR / f"domains_fit_{S.stamp}.pt", weights_only=True)
    dom_val = torch.load(OUT_DIR / f"domains_cohort_{S.stamp}.pt", weights_only=True)
    assert dom_fit["parent_sha256"] == S.ckpt_sha and dom_val["parent_sha256"] == S.ckpt_sha
    if need_cohort:
        val_digest, _ = rows_digest(iter(S.val_rows))
        assert dom_val["n_rows"] == S.n_val and dom_val["rows_sha256"] == val_digest, "cohort domains misaligned"
    counters = Counter()
    # The cohort rows (146,597 dicts, ~4 M objects) stay resident for the whole
    # build; Python's cyclic collector then taxes every allocation the feature
    # engine makes. Measured 2026-09-26: 18.4 ms/decision with them resident vs
    # 13.9 after gc.freeze() (pure runtime hygiene; nothing computed changes).
    gc.freeze()

    def features_for(row, cand_ids):
        budget = float(row["gold_est"])
        feats = torch.zeros((len(cand_ids), N_ECON))
        for k, cand in enumerate(cand_ids):
            if cand == 0:
                feats[k, 0] = 1.0
                continue
            feats[k].copy_(torch.tensor(econ_features(to_action(S.baskets[cand - 1]), row, budget, S.dragon, counters)))
        return feats

    def stream():
        return v1.stream_split(S.train_path, S.train_set, True, S.champ_index, S.dragon,
                               int(dom_fit["n_rows"]) if args.smoke else None)
    return S, dom_fit, dom_val, counters, features_for, stream


def _bench(args, dom_fit, features_for, stream, n_total, report):
    bench_rows = []
    for row in stream():
        bench_rows.append(row)
        if len(bench_rows) >= BENCH_WARMUP + BENCH_ROWS:
            break
    for j in range(min(BENCH_WARMUP, len(bench_rows))):
        features_for(bench_rows[j], dom_fit["top_idx"][j].tolist())
    timed = bench_rows[BENCH_WARMUP:]
    # The bound is on the steady-state cost of this machine. A foreground
    # game or a maintenance window can double single-core throughput cost, so
    # a projection above the bound is re-measured a few times, ten minutes
    # apart, before the segment is stopped for it.
    attempts = []
    for attempt in range(6):
        t0 = time.monotonic()
        for j, row in enumerate(timed, start=BENCH_WARMUP):
            features_for(row, dom_fit["top_idx"][j].tolist())
        per_row = (time.monotonic() - t0) / max(len(timed), 1)
        projected_h = per_row * n_total / 3600
        attempts.append({"ms_per_decision": round(per_row * 1e3, 2), "projected_build_hours": round(projected_h, 2)})
        print(f"BENCH #{attempt + 1}: {per_row * 1e3:.1f} ms/decision -> {projected_h:.2f} h para {n_total} "
              f"decisiones (tope {BOUNDS_H['econ_cache']})", flush=True)
        if projected_h <= BOUNDS_H["econ_cache"] or args.smoke:
            break
        time.sleep(600)
    report["bench"] = {"rows": len(timed), "ms_per_decision": round(per_row * 1e3, 2),
                       "ms_per_candidate": round(per_row * 1e3 / v1.SHORTLIST, 4),
                       "projected_build_hours": round(projected_h, 2), "bound_hours": BOUNDS_H["econ_cache"],
                       "attempts": attempts}
    return projected_h


def _fit_shards(args, S, dom_fit, features_for, stream, shard_rows, wanted):
    """Compute the FIT features of the wanted shards (all if None); every row of the stream feeds the digest."""
    n_fit = int(dom_fit["n_rows"])
    n_shards = math.ceil(n_fit / shard_rows)
    wanted = set(range(n_shards)) if wanted is None else set(wanted)
    assert wanted <= set(range(n_shards)), f"shards {sorted(wanted)} outside 0..{n_shards - 1}"
    shards, buf, cur, t0 = [], None, None, time.monotonic()
    digest = hashlib.sha256()
    i = -1
    done_rows = 0
    for i, row in enumerate(stream()):
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
        k = i // shard_rows
        if k not in wanted:
            continue
        if k != cur:
            if buf is not None:
                shards.append(_save_shard(buf, cur, cur * shard_rows, S.stamp))
            cur = k
            buf = torch.zeros((min(shard_rows, n_fit - k * shard_rows), v1.SHORTLIST, N_ECON), dtype=torch.float16)
        buf[i - k * shard_rows] = features_for(row, dom_fit["top_idx"][i].tolist()).half()
        done_rows += 1
        if done_rows % 5000 == 0:
            rate = done_rows / (time.monotonic() - t0)
            todo = sum(min(shard_rows, n_fit - s * shard_rows) for s in wanted) - done_rows
            print(f"  econ fit shards {sorted(wanted)}: {done_rows} filas ({rate:.1f} filas/s, ETA {todo / rate / 60:.0f} min)", flush=True)
    assert i + 1 == n_fit, f"FIT stream {i + 1} vs domains {n_fit}"
    assert digest.hexdigest() == dom_fit["rows_sha256"], "FIT rows misaligned with the domains"
    if buf is not None:
        shards.append(_save_shard(buf, cur, cur * shard_rows, S.stamp))
    return shards, time.monotonic() - t0


def _save_shard(buf, k, start, stamp):
    path = OUT_DIR / f"econ_fit_{stamp}_part{k:02d}.pt"
    torch.save(buf, path)
    meta = {"path": path.name, "rows": int(buf.shape[0]), "start": int(start), "shard": int(k)}
    (OUT_DIR / f"econ_fit_{stamp}_part{k:02d}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _cohort_features(S, dom_val, features_for):
    feats_v = torch.zeros((S.n_val, v1.SHORTLIST, N_ECON), dtype=torch.float16)
    t1 = time.monotonic()
    for i, row in enumerate(S.val_rows):
        feats_v[i] = features_for(row, dom_val["top_idx"][i].tolist()).half()
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.monotonic() - t1)
            print(f"  econ cohorte {i + 1}/{S.n_val} ({rate:.1f} filas/s, ETA {(S.n_val - i - 1) / rate / 60:.0f} min)", flush=True)
    torch.save(feats_v, OUT_DIR / f"econ_cohort_{S.stamp}.pt")
    return time.monotonic() - t1


def _merge(args, S, dom_fit, shard_rows, report, extra):
    n_fit = int(dom_fit["n_rows"])
    n_shards = math.ceil(n_fit / shard_rows)
    shards = []
    for k in range(n_shards):
        meta_path = OUT_DIR / f"econ_fit_{S.stamp}_part{k:02d}.json"
        assert meta_path.exists(), f"missing shard {k}"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        t = torch.load(OUT_DIR / meta["path"], weights_only=True)
        expected = min(shard_rows, n_fit - k * shard_rows)
        assert meta["start"] == k * shard_rows and meta["rows"] == expected == t.shape[0], f"shard {k} shape mismatch"
        assert t.shape[1:] == (v1.SHORTLIST, N_ECON) and t.dtype == torch.float16
        del t
        shards.append(meta)
    assert (OUT_DIR / f"econ_cohort_{S.stamp}.pt").exists(), "cohort features missing"
    manifest = {"n_rows": n_fit, "shards": shards, "rows_sha256": dom_fit["rows_sha256"], "parent_sha256": S.ckpt_sha,
                "shard_rows": shard_rows}
    (OUT_DIR / f"econ_fit_{S.stamp}_shards.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total_bytes = n_fit * v1.SHORTLIST * N_ECON * 2
    concatenated = False
    if available_ram_bytes() > 3 * total_bytes + 2_000_000_000:
        full = torch.cat([torch.load(OUT_DIR / sh["path"], weights_only=True) for sh in shards])
        assert full.shape[0] == n_fit
        torch.save(full, OUT_DIR / f"econ_fit_{S.stamp}.pt")
        del full
        for sh in shards:
            (OUT_DIR / sh["path"]).unlink()
        concatenated = True
    report.update({"shards": n_shards, "shard_rows": shard_rows, "concatenated": concatenated, **extra})
    return manifest


def phase_features(args):
    started = time.monotonic()
    shard_rows = args.shard_rows or SHARD_ROWS_DEFAULT
    wanted = [int(x) for x in args.shards.split(",")] if args.shards else None
    parallel_worker = wanted is not None or args.cohort_only or args.merge
    S, dom_fit, dom_val, counters, features_for, stream = _features_context(args, need_cohort=(not parallel_worker) or args.cohort_only or args.merge)
    stamp = S.stamp
    n_fit = int(dom_fit["n_rows"])
    n_val = int(dom_val["n_rows"])
    report = {"schema": 2, "phase": "features", "mode": stamp, "created_utc": now_utc(), "parent_sha256": S.ckpt_sha,
              "fit_rows": n_fit, "cohort_rows": n_val, "n_econ": N_ECON, "shortlist": v1.SHORTLIST}
    if args.merge:
        extra = {"build": "parallel workers (bit-identical to sequential by construction)",
                 "bench_ms_per_decision_sequential": args.bench_ms, "rows_per_second": args.rows_per_second,
                 "counters": "per worker (see worker logs)", "elapsed_seconds": round(time.monotonic() - started, 1)}
        _merge(args, S, dom_fit, shard_rows, report, extra)
        write_report(OUT_DIR / f"cache_report_{stamp}.json", report)
        print("CACHE COMPLETA (merge)", flush=True)
        return
    if args.cohort_only:
        secs = _cohort_features(S, dom_val, features_for)
        (OUT_DIR / f"econ_cohort_{stamp}.json").write_text(json.dumps({"rows": n_val, "seconds": round(secs, 1),
                                                                        "counters": dict(counters)}), encoding="utf-8")
        print(f"cohorte lista ({secs / 60:.1f} min)", flush=True)
        return
    if wanted is not None:
        shards, secs = _fit_shards(args, S, dom_fit, features_for, stream, shard_rows, wanted)
        print(f"shards {sorted(wanted)} listos ({secs / 60:.1f} min, {sum(s['rows'] for s in shards) / max(secs, 1e-9):.1f} filas/s)", flush=True)
        return
    # ---- sequential build: bench-gated, then shards, then cohort ----
    projected_h = _bench(args, dom_fit, features_for, stream, n_fit + n_val, report)
    if projected_h > BOUNDS_H["econ_cache"] and not args.smoke:
        report["aborted"] = f"projected build {projected_h:.2f} h > declared bound {BOUNDS_H['econ_cache']} h"
        write_report(OUT_DIR / f"cache_report_{stamp}.json", report)
        print("ABORTADO: " + report["aborted"], flush=True)
        sys.exit(3)
    shards, fit_seconds = _fit_shards(args, S, dom_fit, features_for, stream, shard_rows, None)
    cohort_seconds = _cohort_features(S, dom_val, features_for)
    _merge(args, S, dom_fit, shard_rows, report,
           {"build": "sequential", "counters": dict(counters), "fit_seconds": round(fit_seconds, 1),
            "cohort_seconds": round(cohort_seconds, 1),
            "rows_per_second": round((n_fit + n_val) / max(fit_seconds + cohort_seconds, 1e-9), 2),
            "elapsed_seconds": round(time.monotonic() - started, 1)})
    write_report(OUT_DIR / f"cache_report_{stamp}.json", report)
    print("CACHE COMPLETA", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("domains", "gate", "features"), required=True)
    ap.add_argument("--cpu", action="store_true", help="domains: never touch CUDA")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shards", default=None, help="features worker: comma-separated shard indices to compute")
    ap.add_argument("--cohort-only", action="store_true", help="features worker: cohort features only")
    ap.add_argument("--merge", action="store_true", help="features: verify shards, write manifest + report")
    ap.add_argument("--shard-rows", type=int, default=None)
    ap.add_argument("--bench-ms", type=float, default=None, help="merge: the sequential bench to record")
    ap.add_argument("--rows-per-second", type=float, default=None, help="merge: aggregate throughput to record")
    args = ap.parse_args()
    {"domains": phase_domains, "gate": phase_gate, "features": phase_features}[args.phase](args)


if __name__ == "__main__":
    main()
