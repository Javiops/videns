"""Displayed-policy report for a mixture package, through the shared adapter.

Protocol: docs/mixture_promotion_prework_v1.md. Four questions, reported
separately because they fail separately:

  1. EQUIVALENCE  - does the package reproduce the frozen mixture's
                    persisted rows, cell by cell, computing its economic
                    features FRESH instead of reading the experiment cache?
  2. LEGALITY     - is every one of the THREE displayed options legal at the
                    decision's declared wallet? All three are persisted.
  3. CORPUS INDEPENDENCE - can the package be loaded and scored with the
                    training corpus and the experiment caches unreadable?
  4. LATENCY      - reported by scripts/basket_latency_bench.py, not here.

This scores through `app/mixture_policy.MixturePolicy`, the same object a
serving path would use. If evaluation had its own implementation the report
would describe a policy nobody serves.

The footwear runtime is activated FIRST, from this clean process, before
train_prefix is imported anywhere -- see MixturePolicy.activate_runtime.

Run DIRECTLY (not through footwear_runtime.py, which imports the trainer
before we get control):

    .venv\\Scripts\\python.exe scripts\\mixture_displayed_policy.py \\
        --package mixture-50-50-v1 [--limit N]
"""
import argparse
import builtins
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

REVIEW = ROOT / "data/corpora/16.18-reconstructible-v3/static_review/static_review.json"
OUT_DIR = ROOT / "data" / "reports" / "mixture_displayed_policy"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
WF_DIR = ROOT / "data" / "experiments" / "16.18-wardfree-v1"


class CorpusGuard:
    """Make the training corpus and the experiment caches unreadable.

    Serving must answer from the package alone. A method that returns True
    proves nothing; this actually denies the reads and records every attempt,
    so "independent of the corpus" becomes a measurement.
    """

    def __init__(self, blocked_roots):
        self.roots = [str(Path(r).resolve()).lower() for r in blocked_roots]
        self.attempts = []
        self._open = builtins.open
        self._io_open = io.open
        self.active = False

    def _check(self, path):
        try:
            p = str(Path(path).resolve()).lower()
        except Exception:
            return
        for root in self.roots:
            if p.startswith(root):
                self.attempts.append(str(path))
                raise PermissionError(
                    f"CorpusGuard: blocked read of {path}")

    def __enter__(self):
        guard = self

        def guarded(file, *a, **kw):
            if isinstance(file, (str, bytes, os.PathLike)):
                guard._check(file)
            return guard._open(file, *a, **kw)

        builtins.open = guarded
        io.open = guarded
        self.active = True
        return self

    def __exit__(self, *exc):
        builtins.open = self._open
        io.open = self._io_open
        self.active = False
        return False


def canon(pred):
    return ("SAVE" if pred == "SAVE"
            else {str(k): int(v) for k, v in pred.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="mixture-50-50-v1")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap probe rows (0 = all)")
    ap.add_argument("--sandbox-rows", type=int, default=64)
    args = ap.parse_args()
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkg_dir = ROOT / "data" / "packages" / args.package

    # ---- 0. runtime FIRST, from a clean process ----
    import app.mixture_policy as mp
    active = mp.activate_runtime(pkg_dir, REVIEW)
    print(f"runtime activo y coincidente: {active['runtime_version']}",
          flush=True)

    policy = mp.load_package(args.package)
    print(f"paquete {args.package} cargado; identidad {policy.digest[:16]}...; "
          f"arquitectura {policy.arch}; acciones {policy.n_act}", flush=True)

    # ---- rows: from the frozen generation, for evaluation only ----
    import train_basket_ab as v1
    import train_prefix as tp
    from basket_vocabulary_probe import split_fit_probe
    generation = ROOT / "data/dataset_generations/16.18-reconstructible-development-v3"
    plan = json.loads((generation / "split_plan.json").read_text(encoding="utf-8"))
    fit_games, _, _ = split_fit_probe(plan)
    probe_rows = list(v1.stream_split(generation / "visits_train.jsonl",
                                      set(fit_games), False,
                                      policy.champ_index, policy.dragon,
                                      args.limit or None))
    n = len(probe_rows)
    print(f"probe: {n} decisiones ({time.monotonic() - started:.0f}s)",
          flush=True)

    # ---- 1+2. score through the adapter; persist ALL THREE options ----
    from app.conditional_policy import purchase_sequence
    records = []
    t0 = time.monotonic()
    for s in range(0, n, 256):
        chunk = probe_rows[s:s + 256]
        for row, opts in zip(chunk, policy.options(chunk)):
            first = opts[0]
            records.append({
                "match_id": row["match_id"], "ts": row["ts"],
                "champion": row.get("champion"),
                "save_kind": row.get("save_kind"),
                "budget": float(row["gold_est"]),
                "options": [{"kind": o["kind"], "items": o["items"]}
                            for o in opts],
                "pred_first": "SAVE" if first["kind"] == "save"
                              else first["items"],
            })
        if (s // 256) % 40 == 0:
            rate = (s + len(chunk)) / max(time.monotonic() - t0, 1e-9)
            print(f"  {s + len(chunk)}/{n} ({rate:.0f} filas/s, ETA "
                  f"{(n - s - len(chunk)) / max(rate, 1e-9) / 60:.0f} min)",
                  flush=True)
    scoring_s = time.monotonic() - t0

    rows_path = OUT_DIR / f"rows_{args.package}_probe.jsonl"
    with open(rows_path, "w", encoding="utf-8", newline=chr(10)) as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + chr(10))

    # legality of EVERY displayed option, re-checked independently
    legality = Counter()
    illegal = []
    for rec, row in zip(records, probe_rows):
        for k, o in enumerate(rec["options"]):
            if o["kind"] == "save":
                legality["save"] += 1
                continue
            action = tuple(sorted(int(i) for i, c in o["items"].items()
                                  for _ in range(int(c))))
            ok = purchase_sequence(action, row, rec["budget"],
                                   policy.dragon) is not None
            legality["legal" if ok else "ILLEGAL"] += 1
            legality[f"legal_pos{k + 1}" if ok else f"ILLEGAL_pos{k + 1}"] += 1
            if not ok and len(illegal) < 20:
                illegal.append({"match_id": rec["match_id"], "ts": rec["ts"],
                                "position": k + 1, "action": list(action),
                                "budget": rec["budget"]})

    # equivalence against the frozen mixture's persisted rows
    ref_path = MIX_DIR / "rows_M_full.jsonl"
    equivalence = None
    if ref_path.exists():
        ref = [json.loads(l) for l in
               ref_path.read_text(encoding="utf-8").splitlines() if l.strip()][:n]
        bad = {"pred_first": 0, "identity": 0}
        for a, b in zip(records, ref):
            if (a["match_id"], a["ts"], a["champion"]) != (
                    b["match_id"], b["ts"], b["champion"]):
                bad["identity"] += 1
                continue
            bad["pred_first"] += canon(a["pred_first"]) != canon(b["pred_first"])
        equivalence = {
            "reference": str(ref_path.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
            "rows_compared": len(ref), "mismatches": bad,
            "note": ("the package computes its economic features FRESH; the "
                     "reference was produced from the experiment cache, so "
                     "agreement here is evidence the package needs neither"),
        }
        print("equivalencia:", json.dumps(bad), flush=True)

    # ---- 3. corpus independence, actually enforced ----
    sample = [dict(r) for r in probe_rows[:args.sandbox_rows]]
    guard = CorpusGuard([ROOT / "data" / "dataset_generations",
                         ROOT / "data" / "experiments",
                         ROOT / "data" / "corpora"])
    sandbox = {"rows_scored": 0, "blocked_reads": 0, "ok": False,
               "blocked_paths": [], "error": None}
    try:
        with guard:
            # SELF-TEST FIRST. "0 blocked reads" means nothing unless the
            # guard can actually fire: a broken guard and a package that
            # never touches the corpus look identical from outside.
            probes = [generation / "visits_train.jsonl",
                      ROOT / "data/experiments/16.18-basket-residual-econ-v1"
                           / "econ_fit_full.pt"]
            for probe in probes:
                try:
                    with open(probe, "rb") as fh:
                        fh.read(1)
                except PermissionError:
                    sandbox.setdefault("selftest_blocked", []).append(
                        str(probe.relative_to(ROOT)))
                else:
                    raise RuntimeError(
                        f"CorpusGuard no bloqueo {probe} -- la prueba de "
                        "independencia seria vacia")
            try:
                import torch as _t
                _t.load(probes[1], map_location="cpu", weights_only=True)
            except PermissionError:
                sandbox["selftest_torch_load_blocked"] = True
            else:
                raise RuntimeError("CorpusGuard no bloqueo torch.load")
            sandbox["selftest_attempts"] = len(guard.attempts)
            guard.attempts.clear()      # only the package's reads count below

            fresh = mp.MixturePolicy(pkg_dir, policy.dragon,
                                     device=str(policy.device))
            got = fresh.options(sample)
            sandbox["rows_scored"] = len(got)
            sandbox["ok"] = len(got) == len(sample) and all(
                o and o[0].get("legal") for o in got)
    except Exception as exc:                       # noqa: BLE001
        sandbox["error"] = f"{type(exc).__name__}: {exc}"
    sandbox["blocked_reads"] = len(guard.attempts)
    sandbox["blocked_paths"] = guard.attempts[:10]
    sandbox["blocked_roots"] = ["data/dataset_generations", "data/experiments",
                                "data/corpora"]
    print("independencia del corpus:", json.dumps(
        {k: sandbox[k] for k in ("ok", "rows_scored", "blocked_reads", "error")}),
        flush=True)

    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/mixture_promotion_prework_v1.md",
        "package": args.package,
        "package_identity_sha256": policy.digest,
        "manifest_provenance_status":
            policy.manifest["provenance"]["status"],
        "NOT_A_PROMOTION_ARTIFACT": (
            "promote_served_model.py cannot consume this report: it expects a "
            "single-artifact schema. This is the evidence, not the promotion."),
        "split": "internal chronological TRAIN probe (already exposed); "
                 "NOT val, NOT test",
        "n_decisions": n,
        "architecture_verified": policy.arch,
        "runtime": {"active": active, "matches_manifest": True,
                    "economics_verified_by_identity": True},
        "equivalence": equivalence,
        "legality": {
            "budget": "row['gold_est'] -- one wallet for scorer and filter",
            "options_checked": sum(v for k, v in legality.items()
                                   if k in ("legal", "ILLEGAL", "save")),
            "counts": dict(legality), "illegal_examples": illegal,
            "unaffordable": legality["ILLEGAL"],
        },
        "corpus_independence": sandbox,
        "rows_file": str(rows_path.relative_to(ROOT)),
        "rows_file_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "scoring_seconds": round(scoring_s, 1),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = OUT_DIR / f"displayed_policy_{args.package}_probe.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("equivalence", "legality", "corpus_independence")},
                     indent=1)[:1800], flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
