"""Displayed-policy report for an M2-style package on the v4 forward cohort.

Same four questions as scripts/mixture_displayed_policy.py (equivalence,
legality of every displayed option, corpus independence, latency elsewhere),
scored through `app/mixture_policy.MixturePolicy` — the object a serving path
would use. Differences from v1: the rows are the v4 development cohort
(visits_val.jsonl of the v4 generation, the population every stage-2 arm was
scored on) and the equivalence reference is the stage-2 experiment's M2 rows
(`rows_M2_mix_full.jsonl`), compared on pred_first AND the two exactness
flags via the ward-free record.

Run DIRECTLY (not through footwear_runtime.py):

    .venv\\Scripts\\python.exe scripts\\mixture_displayed_policy_v2.py --package mixture-v4-m2-v1 [--limit N]
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from mixture_displayed_policy import CorpusGuard, canon      # noqa: E402  (no runtime side effects at import)

REVIEW = ROOT / "data/acceptance/16.18_reconstructible_20260923/inputs/static_review.json"
GENERATION = ROOT / "data/dataset_generations/16.18-reconstructible-development-v4"
STAGE2 = ROOT / "data/experiments/16.18-stage2-v4"
OUT_DIR = ROOT / "data" / "reports" / "mixture_displayed_policy"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="mixture-v4-m2-v1")
    ap.add_argument("--limit", type=int, default=0, help="cap cohort rows (0 = all)")
    ap.add_argument("--sandbox-rows", type=int, default=64)
    args = ap.parse_args()
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkg_dir = ROOT / "data" / "packages" / args.package

    # ---- 0. runtime FIRST, from a clean process ----
    import app.mixture_policy as mp
    active = mp.activate_runtime(pkg_dir, REVIEW)
    print(f"runtime activo y coincidente: {active['runtime_version']}", flush=True)
    policy = mp.load_package(args.package)
    print(f"paquete {args.package} cargado; identidad {policy.digest[:16]}...; arquitectura {policy.arch}; "
          f"acciones {policy.n_act}", flush=True)

    # ---- rows: the v4 cohort, for evaluation only ----
    import train_basket_ab as v1
    from basket_wardfree import wardfree_record
    rows = list(v1.stream_split(GENERATION / "visits_val.jsonl", set(), False, policy.champ_index, policy.dragon,
                                args.limit or None))
    n = len(rows)
    print(f"cohorte: {n} decisiones ({time.monotonic() - started:.0f}s)", flush=True)

    # ---- 1+2. score through the adapter; persist ALL THREE options ----
    from app.conditional_policy import purchase_sequence
    records, wf = [], []
    t0 = time.monotonic()
    for s in range(0, n, 256):
        chunk = rows[s:s + 256]
        for row, opts in zip(chunk, policy.options(chunk)):
            first = opts[0]
            records.append({"match_id": row["match_id"], "ts": row["ts"], "champion": row.get("champion"),
                            "save_kind": row.get("save_kind"), "budget": float(row["gold_est"]),
                            "options": [{"kind": o["kind"], "items": o["items"]} for o in opts],
                            "pred_first": "SAVE" if first["kind"] == "save" else first["items"]})
            conv = [("save", None) if o["kind"] == "save" else ("buy", Counter({int(k): int(v) for k, v in o["items"].items()}))
                    for o in opts]
            wf.append(wardfree_record(row, conv))
        if (s // 256) % 40 == 0:
            rate = (s + len(chunk)) / max(time.monotonic() - t0, 1e-9)
            print(f"  {s + len(chunk)}/{n} ({rate:.0f} filas/s, ETA {(n - s - len(chunk)) / max(rate, 1e-9) / 60:.0f} min)", flush=True)
    scoring_s = time.monotonic() - t0
    rows_path = OUT_DIR / f"rows_{args.package}_cohort.jsonl"
    with open(rows_path, "w", encoding="utf-8", newline=chr(10)) as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + chr(10))

    legality = Counter()
    illegal = []
    for rec, row in zip(records, rows):
        for k, o in enumerate(rec["options"]):
            if o["kind"] == "save":
                legality["save"] += 1
                continue
            action = tuple(sorted(int(i) for i, c in o["items"].items() for _ in range(int(c))))
            ok = purchase_sequence(action, row, rec["budget"], policy.dragon) is not None
            legality["legal" if ok else "ILLEGAL"] += 1
            legality[f"legal_pos{k + 1}" if ok else f"ILLEGAL_pos{k + 1}"] += 1
            if not ok and len(illegal) < 20:
                illegal.append({"match_id": rec["match_id"], "ts": rec["ts"], "position": k + 1,
                                "action": list(action), "budget": rec["budget"]})

    # ---- equivalence against the stage-2 experiment's M2 rows ----
    ref_path = STAGE2 / "rows_M2_mix_full.jsonl"
    ref = [json.loads(l) for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    bad = {"identity": 0, "pred_first": 0, "first_action_exact": 0, "any_of_three_exact": 0}
    for a, w, b in zip(records, wf, ref):
        if (a["match_id"], a["ts"], a["champion"]) != (b["match_id"], b["ts"], b["champion"]):
            bad["identity"] += 1
            continue
        bad["pred_first"] += canon(a["pred_first"]) != canon(b["pred_first"])
        for key in ("first_action_exact", "any_of_three_exact"):
            bad[key] += w[key] != b[key]
    equivalence = {"reference": str(ref_path.relative_to(ROOT)).replace("\\", "/"),
                   "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                   "rows_compared": len(ref), "mismatches": bad,
                   "first_action_exact_through_adapter": round(sum(w["first_action_exact"] for w in wf) / max(n, 1), 5),
                   "note": ("the package computes its economic features FRESH and the ward-free arm's scores itself; "
                            "the reference came from the experiment's cached features and separate checkpoints")}
    print("equivalencia:", json.dumps(bad), flush=True)

    # ---- 3. corpus independence, actually enforced ----
    sample = [dict(r) for r in rows[:args.sandbox_rows]]
    guard = CorpusGuard([ROOT / "data" / "dataset_generations", ROOT / "data" / "experiments", ROOT / "data" / "corpora"])
    sandbox = {"rows_scored": 0, "blocked_reads": 0, "ok": False, "blocked_paths": [], "error": None}
    try:
        with guard:
            probes = [GENERATION / "visits_val.jsonl", STAGE2 / "econ_cohort_full.pt"]
            for probe in probes:
                try:
                    with open(probe, "rb") as fh:
                        fh.read(1)
                except PermissionError:
                    sandbox.setdefault("selftest_blocked", []).append(str(probe.relative_to(ROOT)).replace("\\", "/"))
                else:
                    raise RuntimeError(f"CorpusGuard no bloqueo {probe} -- la prueba de independencia seria vacia")
            try:
                import torch as _t
                _t.load(probes[1], map_location="cpu", weights_only=True)
            except PermissionError:
                sandbox["selftest_torch_load_blocked"] = True
            else:
                raise RuntimeError("CorpusGuard no bloqueo torch.load")
            sandbox["selftest_attempts"] = len(guard.attempts)
            guard.attempts.clear()
            fresh = mp.MixturePolicy(pkg_dir, policy.dragon, device=str(policy.device))
            got = fresh.options(sample)
            sandbox["rows_scored"] = len(got)
            sandbox["ok"] = len(got) == len(sample) and all(o and o[0].get("legal") for o in got)
    except Exception as exc:                        # noqa: BLE001
        sandbox["error"] = f"{type(exc).__name__}: {exc}"
    sandbox["blocked_reads"] = len(guard.attempts)
    sandbox["blocked_paths"] = guard.attempts[:10]
    sandbox["blocked_roots"] = ["data/dataset_generations", "data/experiments", "data/corpora"]
    print("independencia del corpus:", json.dumps({k: sandbox[k] for k in ("ok", "rows_scored", "blocked_reads", "error")}), flush=True)

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/mixture_promotion_prework_v1.md (questions 1-3) on the v4 cohort", "package": args.package,
              "package_identity_sha256": policy.digest, "manifest_provenance_status": policy.manifest["provenance"]["status"],
              "NOT_A_PROMOTION_ARTIFACT": ("promote_served_model.py cannot consume this report: it expects a single-artifact "
                                           "schema; and this cohort is exposed (selection), not a frozen validation."),
              "split": "v4 forward development cohort (exposed by stage 1 and stage 2); NOT the sealed test",
              "n_decisions": n, "architecture_verified": policy.arch,
              "runtime": {"active": active, "matches_manifest": True, "economics_verified_by_identity": True},
              "equivalence": equivalence,
              "legality": {"budget": "row['gold_est'] -- one wallet for scorer and filter",
                           "options_checked": sum(v for k, v in legality.items() if k in ("legal", "ILLEGAL", "save")),
                           "counts": dict(legality), "illegal_examples": illegal, "unaffordable": legality["ILLEGAL"]},
              "corpus_independence": sandbox,
              "rows_file": str(rows_path.relative_to(ROOT)).replace("\\", "/"),
              "rows_file_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
              "scoring_seconds": round(scoring_s, 1), "elapsed_seconds": round(time.monotonic() - started, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = OUT_DIR / f"displayed_policy_{args.package}_cohort.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("equivalence", "legality", "corpus_independence")}, indent=1)[:1800], flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
