"""Run the fixed outcome-blind component-initiation census."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.commitment_census import run

def guard():
    if os.name!='nt':return
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
             ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) or value.available<.75*1024**3:
        raise RuntimeError('Census RAM floor0.75GiB')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();guard()
    report=run(args.output,ROOT/'docs/commitment_census_protocol_v1.md',guard)
    print(json.dumps({key:report[key] for key in ('primary_focal','gates','source_validation_readiness','elapsed_s')},indent=2))
