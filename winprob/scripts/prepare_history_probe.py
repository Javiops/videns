"""Prepare the frozen closed-visit history probe; no fitting or policy scoring."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.history_probe_data import prepare


def memory_guard():
    if os.name!='nt':return
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) or memory.available<.75*1024**3:
        raise RuntimeError('History adapter memory guard: available RAM below0.75GiB')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest',type=Path,default=ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/history_probe_protocol_v1.md')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deadline-seconds',type=int,default=900)
    args=parser.parse_args();memory_guard()
    if args.base_manifest.name!='manifest.json':raise ValueError('Expected the immutable base manifest.json path')
    result=prepare(args.base_manifest.parent,args.output,args.protocol,deadline_seconds=args.deadline_seconds,guard=memory_guard)
    print(json.dumps({key:result[key] for key in ('schema','state_count','selected_count','selection_status_counts','donor_counts','elapsed_s')},indent=2))


if __name__=='__main__':main()
