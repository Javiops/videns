"""Stage-2 chain on the v4 generation (owner GO 2026-09-25 23:30; "use the free cores and RAM" 2026-09-26 00:05).

Protocol: docs/experiment_stage2_econ_wardfree_v1.md. Segments and inputs:
  domains  (CPU only)  R1's top-128 for FIT + cohort, persisted with digests
  features (CPU only)  economic features, bench-gated (6 h), sharded   [needs domains]
  gate     (GPU, minutes) R1 must reproduce its stage-1 rows exactly
  wardfree (GPU) ward-free partner, pilot-gated (8 h)                  [needs gate]
  finetune (GPU) twin econ fine-tune, pilot-gated (12 h)               [needs gate, domains, features]
  mixture  (GPU, minutes) M2 = 0.5 econ_ft_new + 0.5 wardfree_new      [needs finetune, wardfree]
  final report (always).
Scheduling only, never weights or criteria:
  * CPU segments run at any time; while the owner's game ("League of
    Legends.exe") runs they are re-prioritised to BelowNormal on all cores
    (the game keeps the P-cores), otherwise AboveNormal on the P-cores.
  * GPU segments do not START while a game runs; if a game starts while
    one runs, its processes are suspended until the game ends. Pilot-gated
    segments also wait for a quiet GPU before their pilot. Paused time is
    excluded from the 30 h wall budget and recorded per segment.
  * A CPU segment may run beside a GPU segment only if the smoke's measured
    peaks say both fit in RAM (else series). At most one GPU segment at a
    time. Each child is a Windows job with a memory ceiling.
  * exit 3 = the segment's pilot exceeded its declared bound: the segment is
    stopped and the reason published; dependants are skipped; the rest run.
`--smoke` runs everything on capped inputs in series and records peak
memory; `--resume` skips segments whose report exists without an abort.
"""
import argparse
import ctypes
from ctypes import wintypes
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
from app.dataset_artifacts import sha256, write_json
from run_patch_development import WindowsMemoryJob
from winmem import (available_ram_bytes, set_priority_affinity, BELOW_NORMAL_PRIORITY_CLASS,
                    NORMAL_PRIORITY_CLASS, ABOVE_NORMAL_PRIORITY_CLASS)

OUT_DIR = ROOT / "data/experiments/16.18-stage2-v4"
RUN = OUT_DIR / "run"
GEN = ROOT / "data/dataset_generations/16.18-reconstructible-development-v4"
REVIEW = ROOT / "data/acceptance/16.18_reconstructible_20260923/inputs/static_review.json"
PY = sys.executable
WALL_SECONDS = 30 * 3600
RESERVE_GB = 40.0
# Commit ceiling per segment tree. 12 GB was hit by the full twin fine-tune on
# 2026-09-26 (job peak 12.04 GB: FIT cache + 1.9 GB of features + CUDA's
# host-side commit) and surfaced as a CUDA "out of memory" with 4.8 GB of
# VRAM free -- the job, not the card, refused the allocation. 20 GB keeps the
# runaway protection (commit limit of the machine is 37 GB) without that trap.
JOB_LIMIT = 20_000_000_000
PARALLEL_MARGIN_GB = 2.0
GPU_QUIET_UTIL = 15
CREATE_SUSPENDED = 0x4
P_CORES_MASK = 0xFFF          # i5-14400F: 12 logical P-core CPUs; 0xFFFF = all 16 incl. the 4 E-cores
ALL_CORES_MASK = 0xFFFF
SHARD_ROWS = 100_000
EXIT_BOUND = 3
GAME_IMAGE = "League of Legends.exe"
ORDER = ["domains", "features", "gate", "wardfree", "finetune", "mixture"]
PILOT_GATED = {"wardfree", "finetune"}
FROZEN_CODE = ["scripts/stage2_newgen.py", "scripts/basket_econ_cache_newgen.py", "scripts/basket_wardfree_newgen.py",
               "scripts/basket_econ_finetune_newgen.py", "scripts/basket_stage2_mixture.py", "scripts/stage2_final_report.py",
               "scripts/basket_bf_newgen.py", "scripts/basket_newbf_mixture.py", "scripts/footwear_runtime.py",
               "scripts/basket_econ_cache.py", "scripts/basket_econ_finetune.py", "scripts/basket_residual_econ.py",
               "scripts/basket_wardfree.py", "scripts/train_basket_ab.py", "scripts/train_basket_phase2.py",
               "scripts/train_prefix.py", "scripts/winmem.py", "app/conditional_policy.py", "app/shop_econ.py"]

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll")
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
_ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
PROCESS_SUSPEND_RESUME = 0x0800


class _PIDLIST(ctypes.Structure):
    _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD), ("NumberOfProcessIdsInList", wintypes.DWORD),
                ("ProcessIdList", ctypes.c_size_t * 512)]


def utc():
    return datetime.now(timezone.utc).isoformat()


def job_pids(job):
    """Every process currently in the segment's job (the venv launcher and the real interpreter)."""
    info = _PIDLIST()
    info.NumberOfAssignedProcesses = 512
    if not job.kernel.QueryInformationJobObject(job.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None):
        return []
    return [int(info.ProcessIdList[i]) for i in range(int(info.NumberOfProcessIdsInList))]


def set_suspended(pids, suspend):
    for pid in pids:
        h = _kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if h:
            (_ntdll.NtSuspendProcess if suspend else _ntdll.NtResumeProcess)(h)
            _kernel32.CloseHandle(h)


def game_running():
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {GAME_IMAGE}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
        return GAME_IMAGE in out
    except Exception:                                   # noqa: BLE001
        return False


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:                                   # noqa: BLE001
        return None


def stages(stamp, smoke):
    sm = ["--smoke"] if smoke else []

    def ent(script, *extra):
        return [PY, "scripts/footwear_runtime.py", "--entry", f"scripts/{script}", "--review", str(REVIEW), "--", *extra, *sm]
    return {
        "domains": {"cmd": ent("basket_econ_cache_newgen.py", "--phase", "domains", "--cpu"), "gpu": False, "cpu_only": True,
                    "deps": [], "report": OUT_DIR / f"domains_report_{stamp}.json"},
        "features": {"cmd": ent("basket_econ_cache_newgen.py", "--phase", "features"), "gpu": False, "cpu_only": True,
                     "deps": ["domains"], "report": OUT_DIR / f"cache_report_{stamp}.json"},
        "gate": {"cmd": ent("basket_econ_cache_newgen.py", "--phase", "gate"), "gpu": True, "cpu_only": False,
                 "deps": [], "report": OUT_DIR / f"gate_report_{stamp}.json"},
        "wardfree": {"cmd": ent("basket_wardfree_newgen.py"), "gpu": True, "cpu_only": False,
                     "deps": ["gate"], "report": OUT_DIR / f"wardfree_report_{stamp}.json"},
        "finetune": {"cmd": ent("basket_econ_finetune_newgen.py"), "gpu": True, "cpu_only": False,
                     "deps": ["gate", "domains", "features"], "report": OUT_DIR / f"finetune_report_{stamp}.json"},
        "mixture": {"cmd": ent("basket_stage2_mixture.py"), "gpu": True, "cpu_only": False,
                    "deps": ["finetune", "wardfree"], "report": OUT_DIR / f"mixture_report_{stamp}.json"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip segments whose report exists without an abort")
    args = ap.parse_args()
    stamp = "smoke" if args.smoke else "full"
    RUN.mkdir(parents=True, exist_ok=True)
    status_path = RUN / ("smoke_status.json" if args.smoke else "status.json")
    specs = stages(stamp, args.smoke)
    code = {rel: sha256(ROOT / rel) for rel in FROZEN_CODE}
    state = {"schema": 2, "mode": stamp, "started_utc": utc(), "code_sha256": code,
             "stages": {n: {"state": "pending"} for n in ORDER}, "parallel": None, "paused": False,
             "budget": {"wall_seconds": WALL_SECONDS, "reserve_gb": RESERVE_GB}, "state": "running"}
    start = time.monotonic()
    chain_log = open(RUN / f"chain_{stamp}.log", "a", encoding="utf-8")

    def log(msg):
        chain_log.write(f"[{utc()}] {msg}\n")
        chain_log.flush()
        print(msg, flush=True)

    def save():
        state["elapsed_seconds"] = round(time.monotonic() - start, 1)
        state["time_utc"] = utc()
        write_json(status_path, state)

    def unchanged():
        for rel, digest in code.items():
            if sha256(ROOT / rel) != digest:
                raise RuntimeError("frozen code changed during the chain: " + rel)

    pause = {"paused": False, "since": None, "total": 0.0}

    def paused_now():
        return pause["total"] + ((time.monotonic() - pause["since"]) if pause["paused"] else 0.0)

    def budget():
        if time.monotonic() - start - paused_now() > WALL_SECONDS:      # paused time is not machine time
            raise RuntimeError("wall budget reached")
        if shutil.disk_usage(ROOT).free / 1e9 < RESERVE_GB:
            raise RuntimeError("disk reserve reached")

    base_env = {k: v for k, v in os.environ.items() if not k.startswith("PREFIX_")}
    base_env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PREFIX_ML_DIR=str(GEN), HTTP_PROXY="http://127.0.0.1:9",
                    HTTPS_PROXY="http://127.0.0.1:9", ALL_PROXY="http://127.0.0.1:9", NO_PROXY="")

    def cpu_policy(game):
        return (BELOW_NORMAL_PRIORITY_CLASS, ALL_CORES_MASK) if game else (ABOVE_NORMAL_PRIORITY_CLASS, P_CORES_MASK)

    def launch(name, spec, game):
        env = dict(base_env)
        if spec["cpu_only"]:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        prio = cpu_policy(game)[0] if spec["cpu_only"] else NORMAL_PRIORITY_CLASS
        lf = open(RUN / f"{name}_{stamp}.log", "a", encoding="utf-8")
        lf.write(f"\n=== {utc()} {name}: {' '.join(spec['cmd'])}\n")
        lf.flush()
        child = subprocess.Popen(spec["cmd"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 creationflags=CREATE_SUSPENDED | prio)
        job = WindowsMemoryJob(JOB_LIMIT)
        job.attach(child)                                # assigns the job and resumes the suspended child
        proc = {"name": name, "child": child, "job": job, "log": lf, "started": time.monotonic(), "started_utc": utc(),
                "cpu_only": spec["cpu_only"], "paused": 0.0, "game_overlap": 0.0, "policy_game": game}
        if spec["cpu_only"]:
            set_priority_affinity(job_pids(job), *cpu_policy(game))
        state["stages"][name] = {"state": "running", "started_utc": proc["started_utc"],
                                 "cpu_policy": "below-normal/all-cores (game running)" if (spec["cpu_only"] and game)
                                 else ("above-normal/P-cores" if spec["cpu_only"] else "normal/GPU")}
        log(f"{name}: started" + (" (game running: below-normal on all cores)" if spec["cpu_only"] and game else ""))
        save()
        return proc

    def finish(proc, rc):
        name = proc["name"]
        st = state["stages"][name]
        wall = time.monotonic() - proc["started"]
        st.update({"exit_code": rc, "elapsed_seconds": round(wall, 1), "paused_seconds": round(proc["paused"], 1),
                   "active_seconds": round(wall - proc["paused"], 1), "game_overlap_seconds": round(proc["game_overlap"], 1),
                   "started_utc": proc["started_utc"], "ended_utc": utc(), "peak_job_memory_bytes": proc["job"].peak_bytes()})
        rep = specs[name]["report"]
        if rc == 0:
            st["state"] = "completed"
        elif rc == EXIT_BOUND:
            st["state"] = "stopped_by_bound"
            try:
                st["reason"] = json.loads(rep.read_text(encoding="utf-8")).get("aborted")
            except Exception:                            # noqa: BLE001
                st["reason"] = "pilot exceeded its declared bound (report unreadable)"
        else:
            st["state"] = "failed"
            st["reason"] = f"exit {rc}; see {RUN / (name + '_' + stamp + '.log')}"
        proc["log"].close()
        proc["job"].close()
        log(f"{name}: {st['state']} ({st['active_seconds']} s active, {st['paused_seconds']} s paused, "
            f"peak {st['peak_job_memory_bytes'] / 1e9:.2f} GB)")
        save()

    def parallel_decision():
        if args.smoke:
            return False, "smoke: series, measuring peaks"
        smoke_path = RUN / "smoke_status.json"
        if not smoke_path.exists():
            return False, "no smoke peaks available -> series"
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        peaks = {n: smoke["stages"].get(n, {}).get("peak_job_memory_bytes") for n in ("features", "wardfree")}
        if not all(peaks.values()):
            return False, "smoke peaks incomplete -> series"
        est_feat = peaks["features"] + SHARD_ROWS * 128 * 10 * 2 + 1.0e9
        est_wf = peaks["wardfree"] + 0.5e9
        avail = available_ram_bytes()
        ok = est_feat + est_wf + PARALLEL_MARGIN_GB * 1e9 <= avail
        return ok, (f"features est {est_feat / 1e9:.2f} GB + wardfree est {est_wf / 1e9:.2f} GB + margin {PARALLEL_MARGIN_GB} GB "
                    f"{'<=' if ok else '>'} available {avail / 1e9:.2f} GB")

    def schedule(parallel):
        pending = [n for n in ORDER]
        running = []
        quiet_samples = 0
        last_game = None
        while pending or running:
            game = (not args.smoke) and game_running()
            if last_game is not None and game != last_game:
                log("game " + ("started" if game else "ended"))
            last_game = game
            # ---- launch what can run now ----
            for name in list(pending):
                spec = specs[name]
                if args.resume and spec["report"].exists() and "aborted" not in spec["report"].read_text(encoding="utf-8"):
                    state["stages"][name] = {"state": "completed", "resumed": True}
                    log(f"{name}: report exists, skipped (--resume)")
                    pending.remove(name)
                    continue
                dep_states = {d: state["stages"][d]["state"] for d in spec["deps"]}
                if any(s in ("failed", "stopped_by_bound", "skipped") for s in dep_states.values()):
                    state["stages"][name] = {"state": "skipped", "reason": "a dependency did not complete: " +
                                             ", ".join(f"{d}={s}" for d, s in dep_states.items())}
                    log(f"{name}: skipped ({state['stages'][name]['reason']})")
                    pending.remove(name)
                    continue
                if not all(s == "completed" for s in dep_states.values()):
                    continue
                gpu_busy = any(specs[p["name"]]["gpu"] for p in running)
                cpu_busy = any(p["cpu_only"] for p in running)
                if spec["gpu"] and (gpu_busy or game):
                    continue                              # one GPU segment at a time; never start one during a game
                if spec["cpu_only"] and cpu_busy:
                    continue
                if running and not parallel:
                    continue                              # series
                if name in PILOT_GATED and not args.smoke:
                    u = gpu_util()
                    quiet_samples = quiet_samples + 1 if (u is None or u < GPU_QUIET_UTIL) else 0
                    if quiet_samples < 3:
                        continue                          # wait for a quiet GPU before a pilot
                    quiet_samples = 0
                unchanged()
                budget()
                running.append(launch(name, spec, game))
                pending.remove(name)
            if not running:
                if pending:
                    time.sleep(15)
                continue
            # ---- poll the running segments ----
            for proc in list(running):
                try:
                    rc = proc["child"].wait(timeout=15 / len(running))
                except subprocess.TimeoutExpired:
                    continue
                finish(proc, rc)
                running.remove(proc)
            if args.smoke:
                budget()
                save()
                continue
            # ---- game policy: suspend GPU work, re-prioritise CPU work ----
            gpu_procs = [p for p in running if not p["cpu_only"]]
            if game and not pause["paused"] and gpu_procs:
                for p in gpu_procs:
                    set_suspended(job_pids(p["job"]), True)
                pause.update(paused=True, since=time.monotonic())
                state["paused"] = True
                log("game running: GPU segment(s) PAUSED")
            elif not game and pause["paused"]:
                span = time.monotonic() - pause["since"]
                for p in gpu_procs:
                    set_suspended(job_pids(p["job"]), False)
                    p["paused"] += span
                pause["total"] += span
                pause.update(paused=False, since=None)
                state["paused"] = False
                state["paused_total_seconds"] = round(pause["total"], 1)
                log(f"game gone: GPU segment(s) RESUMED after {span:.0f} s paused")
            for p in running:
                if p["cpu_only"] and p["policy_game"] != game:
                    n = set_priority_affinity(job_pids(p["job"]), *cpu_policy(game))
                    p["policy_game"] = game
                    state["stages"][p["name"]]["cpu_policy"] = "below-normal/all-cores (game running)" if game else "above-normal/P-cores"
                    log(f"{p['name']}: re-prioritised for {'game' if game else 'idle machine'} ({n} processes)")
                if game:
                    p["game_overlap"] += 15
            budget()
            save()

    try:
        parallel, why = parallel_decision()
        state["parallel"] = {"cpu_with_gpu": parallel, "reason": why}
        log(f"CPU segment beside GPU segment: {parallel} ({why})")
        save()
        schedule(parallel)
        state["state"] = "completed"
    except BaseException as exc:                           # noqa: BLE001
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        log(state["error"])
        raise
    finally:
        save()
        final = subprocess.run([PY, "scripts/stage2_final_report.py", "--stamp", stamp, "--status", str(status_path)],
                               cwd=ROOT, capture_output=True, text=True)
        chain_log.write(final.stdout + final.stderr)
        chain_log.flush()
        print(final.stdout, flush=True)
        chain_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
