"""Pause / resume the running transfer training so the owner can use the GPU (2026-09-28).

Suspends every python process whose command line runs basket_transfer_ab.py (NtSuspendProcess, the same
mechanism run_stage2_v4.py used around games): compute stops, the CUDA context and its ~4 GB of VRAM stay
allocated, nothing is lost. The epoch in progress keeps running from where it was on resume; only its
wall-clock `seconds` in transfer_progress_H_full.jsonl is inflated by the pause (descriptive only).

    .venv\\Scripts\\python.exe scripts\\pause_transfer_train.py --suspend
    .venv\\Scripts\\python.exe scripts\\pause_transfer_train.py --resume
"""
import argparse
import ctypes
import subprocess
from ctypes import wintypes

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll")
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
_ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
PROCESS_SUSPEND_RESUME = 0x0800


def trainer_pids():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'basket_transfer_ab' } | ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--suspend", action="store_true")
    g.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    pids = trainer_pids()
    if not pids:
        raise SystemExit("no basket_transfer_ab.py process found")
    fn = _ntdll.NtSuspendProcess if args.suspend else _ntdll.NtResumeProcess
    done = []
    for pid in pids:
        h = _kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not h:
            raise SystemExit(f"cannot open pid {pid}: error {ctypes.get_last_error()}")
        status = fn(h)
        _kernel32.CloseHandle(h)
        if status != 0:
            raise SystemExit(f"NtSuspend/ResumeProcess failed for pid {pid}: NTSTATUS {status:#x}")
        done.append(pid)
    print(("suspended" if args.suspend else "resumed"), "pids", done)


if __name__ == "__main__":
    main()
