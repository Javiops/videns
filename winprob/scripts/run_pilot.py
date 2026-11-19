"""One bounded offline pilot, called by the durable resource-gated launcher."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                ('avail_extended', ctypes.c_ulonglong)]


def free_gib():
    if sys.platform != 'win32':
        return float('inf')
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise RuntimeError('Cannot read memory pressure')
    return state.avail_phys/(1024**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-games', type=int, default=6000)
    args = parser.parse_args()
    if sys.platform == 'win32':
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise RuntimeError('Cannot set BelowNormal priority')
    started = time.monotonic()
    def check():
        if time.monotonic()-started > 85*60:
            raise RuntimeError('Pilot deadline reached')
        if free_gib() < 1.0:
            raise RuntimeError('Pilot stopped: less than 1 GiB physical memory available')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        check()
        from winprob.cache import build_cache
        print('Extracting admitted development states', flush=True)
        metadata = build_cache(args.manifest, args.output/'cache', args.max_games, resource_check=check)
        check()
        from winprob.experiment import run_experiment, write_json, sha256
        meta_path = Path(metadata['cache_metadata_path'])
        expected = meta_path.with_suffix('.sha256')
        if expected.read_text().split()[0] != sha256(meta_path):
            raise RuntimeError('Cache metadata digest mismatch')
        cache_path = Path(metadata['npz_path'])
        if sha256(cache_path) != metadata['npz_sha256']:
            raise RuntimeError('NPZ digest mismatch before model fitting')
        print('Fitting fixed models and negative control', flush=True)
        report = run_experiment(cache_path, args.output/'experiment', resource_check=check)
        report['cache_metadata_path'] = str(meta_path.resolve())
        report['cache_metadata_sha256'] = sha256(meta_path)
        write_json(args.output/'experiment'/'report.json', report)
        print(json.dumps({'status': 'completed', 'gates': report['gates'],
                          'population': report['population'], 'elapsed_s': time.monotonic()-started}), flush=True)
    except Exception as error:
        payload = {'status': 'failed', 'type': type(error).__name__, 'reason': str(error),
                   'elapsed_s': time.monotonic()-started}
        (args.output/'failure.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
