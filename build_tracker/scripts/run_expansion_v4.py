"""Design-B expansion chain, v4: snapshot -> reconstruction -> verify ->
admission (explicit forward boundary) -> coverage -> export -> tests ->
Bf d64 stage-1 training + cohort scoring.

Owner GO recorded in docs/expansion_plan_v1.md (2026-09-23). Runs every
stage as a supervised subprocess with its own log, freezes the digests of
the pipeline code before starting and re-checks them between stages,
enforces a 40 GB disk reserve and a wall budget, and writes status.json
after every step. Each stage refuses to overwrite existing outputs, so a
resume (--from-stage) can only continue, never silently redo.

    python scripts/run_expansion_v4.py --dry-run     # preflight + commands
    python scripts/run_expansion_v4.py               # the whole chain
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from app.dataset_artifacts import sha256, write_json          # noqa: E402
from run_patch_development import WindowsMemoryJob             # noqa: E402

NAME = "16.18-reconstructible-development-v4"
EVAL_VERSION = 7
ACCEPT = ROOT / "data/acceptance/16.18_reconstructible_20260923"
CORPUS = ROOT / "data/corpora/16.18-reconstructible-v4"
PRIOR_BASE = ROOT / "data/acceptance/16.18_reconstructible_20260914"
PRIOR_GEN = ROOT / "data/dataset_generations/16.18-reconstructible-development-v3"
PREPARED = ROOT / "data/acceptance/16.18_development_v4_admission"
COVERAGE = ROOT / "data/acceptance/16.18_development_v4_admission_coverage"
DATASET = ROOT / "data/dataset_generations" / NAME
RUN = ROOT / "data/experiments/16.18-expansion-v4"
PLAN_DOC = ROOT / "docs/expansion_plan_v1.md"
BOUNDARY_UTC = "2026-09-20T00:00:00+00:00"
AUTHORIZATION = "owner authorized design B expansion (snapshot, qualification, export, Bf d64) on 2026-09-23"
WALL_SECONDS = 20 * 3600
RECON_WALL = 6 * 3600
MIN_FREE_GB_START = 60.0
RESERVE_GB = 40.0
FROZEN_CODE = [
    "scripts/prepare_reconstructible_1618.py", "scripts/run_reconstructible_1618.py",
    "scripts/process_reconstructible_1618.py", "scripts/reconstructible_1618_runtime.py",
    "scripts/verify_reconstructible_1618_evidence.py", "scripts/prepare_development_1618_v2.py",
    "scripts/benchmark_1618_entry.py", "scripts/report_corpus_coverage.py", "scripts/publish_patch_dataset.py",
    "scripts/footwear_runtime.py", "scripts/basket_bf_newgen.py", "scripts/train_basket_ab.py",
    "scripts/train_basket_phase2.py", "scripts/train_prefix.py", "scripts/basket_wardfree.py",
    "app/mixture_policy.py", "app/dataset_generations.py", "app/conditional_policy.py", "app/shop_econ.py",
]
PY = sys.executable
# The ONLY tolerated test failures: these two files, and only when their error
# is a FileNotFoundError on a path under the repository's OLD root (the repo
# moved to videns.gg\build_tracker; some frozen acceptance JSONs still carry
# absolute paths to the old location). Measured 2026-09-23: 55/57 pass.
KNOWN_STALE_PATH_TESTS = {"test_reconstructible_1618_pipeline.py", "test_recovered_corpus_pipeline.py"}
OLD_ROOT_MARKER = "Users\\Javier\\build_tracker\\"        # note: no 'videns.gg'


def evaluate_tests_log(text):
    """Return (ok, verdict). ok only if every failure is the identified one."""
    failed = [ln.split()[1] for ln in text.splitlines() if ln.startswith(("FAIL", "TIMEOUT"))]
    verdict = {"failed_files": failed, "tolerated": [], "not_tolerated": []}
    if not failed:
        return True, verdict
    blocks = {}
    current = None
    for ln in text.splitlines():
        if ln.startswith("--- ") and ln.endswith(" ---"):
            current = ln[4:-4]
            blocks[current] = []
        elif current:
            blocks[current].append(ln)
    for name in failed:
        detail = "\n".join(blocks.get(name, []))
        last_err = next((ln for ln in reversed(detail.splitlines()) if "Error" in ln), "")
        # the traceback prints the path as a repr: backslashes come doubled
        # ('C:\\\\Users\\\\...'); normalise before matching either root
        norm = last_err.strip().replace("\\\\", "\\").replace("/", "\\")
        stale = (name in KNOWN_STALE_PATH_TESTS and norm.startswith("FileNotFoundError")
                 and OLD_ROOT_MARKER in norm and "videns.gg" not in norm)
        (verdict["tolerated"] if stale else verdict["not_tolerated"]).append({"file": name, "last_error": last_err[:200]})
    return not verdict["not_tolerated"], verdict


def stages():
    review = ACCEPT / "inputs/static_review.json"
    env_train = {"PREFIX_ML_DIR": str(DATASET)}
    return [
        ("snapshot", [PY, "scripts/prepare_reconstructible_1618.py", "--base", str(ACCEPT)], {}, True),
        ("recon", [PY, "scripts/run_reconstructible_1618.py", "--base", str(ACCEPT), "--corpus", str(CORPUS),
                   "--max-wall-seconds", str(RECON_WALL)], {}, True),
        ("verify", [PY, "scripts/verify_reconstructible_1618_evidence.py", "--source-root", str(ROOT),
                    "--base", str(ACCEPT), "--corpus", str(CORPUS),
                    "--out", str(RUN / "prerequisite_verification.json")], {}, True),
        ("admission", [PY, "scripts/prepare_development_1618_v2.py", "--source-root", str(ROOT),
                       "--base", str(ACCEPT), "--corpus", str(CORPUS), "--out", str(PREPARED),
                       "--benchmark-id", NAME, "--eval-version", str(EVAL_VERSION),
                       "--prior-base", str(PRIOR_BASE), "--prior-generation", str(PRIOR_GEN),
                       "--forward-boundary-utc", BOUNDARY_UTC, "--boundary-document", str(PLAN_DOC),
                       "--authorization", AUTHORIZATION,
                       "--authorization-document", "docs/expansion_plan_v1.md"], {}, True),
        ("coverage", [PY, "scripts/benchmark_1618_entry.py", "--corpus", str(CORPUS),
                      "--entry", "report_corpus_coverage", "--",
                      "--qualification", str(PREPARED / "qualification.json"), "--out", str(COVERAGE),
                      "--benchmark-id", NAME, "--eval-version", str(EVAL_VERSION),
                      "--purpose", "patch_development_1618",
                      "--split-assignment", str(PREPARED / "preserved_split_assignment.json"),
                      "--split-metadata", str(PREPARED / "split_metadata.json")], {}, True),
        ("export", [PY, "scripts/benchmark_1618_entry.py", "--corpus", str(CORPUS),
                    "--entry", "publish_patch_dataset", "--", "--coverage", str(COVERAGE), "--name", NAME], {}, True),
        # tests: FATAL on any failure except the one identified in advance --
        # two files fail with FileNotFoundError on absolute paths under the
        # repo's OLD root (before the move), frozen inside acceptance JSONs.
        # Any other failing file, or any other error in those two, aborts the
        # chain BEFORE the multi-hour training (reviewer's condition).
        ("tests", [PY, "scripts/run_tests.py"], {}, True),
        ("train_score", [PY, "scripts/footwear_runtime.py", "--entry", "scripts/basket_bf_newgen.py",
                         "--review", str(review), "--", "--max-projected-hours", "10"], env_train, True),
    ]


def preflight(from_stage):
    problems = []
    outputs = {"snapshot": [ACCEPT / "source_plan.json"], "recon": [ACCEPT / "run", CORPUS],
               "admission": [PREPARED], "coverage": [COVERAGE], "export": [DATASET]}
    order = [s[0] for s in stages()]
    start = order.index(from_stage)
    for name in order[start:]:
        for p in outputs.get(name, []):
            if p.exists():
                problems.append(f"output already exists for stage {name}: {p}")
    for f in ("item-16.17.1.json", "item-16.18.1.json", "runesReforged-16.18.1.json"):
        if not (ACCEPT / "inputs" / f).is_file():
            problems.append(f"missing seeded static input: {f}")
    for rel in FROZEN_CODE:
        if not (ROOT / rel).is_file():
            problems.append(f"frozen code file missing: {rel}")
    if not PLAN_DOC.is_file():
        problems.append("plan document missing")
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < MIN_FREE_GB_START:
        problems.append(f"free disk {free:.1f} GB < {MIN_FREE_GB_START} GB")
    for rel in ("app/dataset_generations.py",):
        if AUTHORIZATION not in (ROOT / rel).read_text(encoding="utf-8"):
            problems.append("the authorization literal is not accepted by validate_1618_admission")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-stage", default="snapshot")
    args = ap.parse_args()
    problems = preflight(args.from_stage)
    if args.dry_run:
        print(json.dumps({"preflight_problems": problems, "free_disk_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
                          "stages": [(n, " ".join(c[1:]), fatal) for n, c, _e, fatal in stages()]}, indent=1))
        return 0 if not problems else 2
    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))
    RUN.mkdir(parents=True, exist_ok=True)
    code = {rel: sha256(ROOT / rel) for rel in FROZEN_CODE}
    write_json(RUN / "run_plan.json", {
        "schema": 1, "name": NAME, "design": "B forward cohort", "boundary_utc": BOUNDARY_UTC,
        "authorization": AUTHORIZATION, "plan_document": str(PLAN_DOC.relative_to(ROOT)),
        "plan_document_sha256": sha256(PLAN_DOC), "code_sha256": code,
        "budget": {"wall_seconds": WALL_SECONDS, "recon_wall_seconds": RECON_WALL, "reserve_gb": RESERVE_GB},
        "started_utc": datetime.now(timezone.utc).isoformat(), "from_stage": args.from_stage})
    start = time.monotonic()
    completed, current = [], "preflight"
    job = WindowsMemoryJob(12_000_000_000)

    def status(state, **extra):
        write_json(RUN / "status.json", {"state": state, "stage": current, "completed_stages": completed,
                                          "time_utc": datetime.now(timezone.utc).isoformat(),
                                          "elapsed_seconds": round(time.monotonic() - start, 1), **extra})

    def unchanged():
        for rel, digest in code.items():
            if sha256(ROOT / rel) != digest:
                raise RuntimeError("frozen code changed during the chain: " + rel)

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError("wall budget reached")
        if shutil.disk_usage(ROOT).free / 1e9 < RESERVE_GB:
            raise RuntimeError("disk reserve reached")

    base_env = {k: v for k, v in os.environ.items() if not k.startswith("PREFIX_")}
    base_env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", HTTP_PROXY="http://127.0.0.1:9",
                    HTTPS_PROXY="http://127.0.0.1:9", ALL_PROXY="http://127.0.0.1:9", NO_PROXY="")
    order = [s[0] for s in stages()]
    try:
        for name, cmd, extra_env, fatal in stages():
            if order.index(name) < order.index(args.from_stage):
                completed.append(name)
                continue
            current = name
            unchanged()
            budget()
            status("running")
            with (RUN / f"{name}.log").open("a", encoding="utf-8") as log:
                log.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {name}: {' '.join(cmd)}\n")
                log.flush()
                child = subprocess.Popen(cmd, cwd=ROOT, env={**base_env, **extra_env},
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=4)
                job.attach(child)
                while True:
                    try:
                        rc = child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        budget()
                        status("running")
            unchanged()
            if name == "tests":
                text = (RUN / "tests.log").read_text(encoding="utf-8", errors="replace")
                text = text[text.rfind("=== "):]          # this attempt's run only
                ok, verdict = evaluate_tests_log(text)
                write_json(RUN / "tests_gate.json", {"exit_code": rc, "ok": ok, **verdict,
                                                     "rule": "only the identified stale-old-root "
                                                             "FileNotFoundError in the two known files is tolerated"})
                if not ok:
                    raise RuntimeError(f"tests gate: unexpected failures {verdict['not_tolerated']}; "
                                       "aborting BEFORE training")
                completed.append("tests" + ("" if not rc else "(known stale-path failures tolerated)"))
                status("running", tests_gate=verdict)
                continue
            if rc and fatal:
                raise RuntimeError(f"{name} failed with exit {rc}; see {RUN / (name + '.log')}")
            completed.append(name + ("" if not rc else f"(exit {rc}, non-fatal)"))
            status("running")
        status("completed")
        print(json.dumps({"completed": completed, "elapsed_h": round((time.monotonic() - start) / 3600, 2)}))
        return 0
    except BaseException as exc:                       # noqa: BLE001
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
