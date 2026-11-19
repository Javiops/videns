"""Publish static goal definition first; run the reviewed descriptive census later."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.goal_census_v2 import prepare_definition,census


def memory_guard():
    if os.name!='nt':return
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) or memory.available<.75*1024**3:
        raise RuntimeError('Goal census memory floor0.75GiB')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    define=sub.add_parser('define');define.add_argument('--base-manifest',type=Path,default=ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json')
    define.add_argument('--history-manifest',type=Path,default=ROOT/'data/runs/history-cache-20260930-v1/cache/manifest.json')
    define.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('census');run.add_argument('--definition',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--deadline-seconds',type=int,default=300)
    args=parser.parse_args();memory_guard()
    if args.mode=='define':
        result=prepare_definition(args.base_manifest,args.history_manifest,args.output)
        print(json.dumps({'definition':str((args.output/'definition.json').resolve()),'goals_per_patch':result['goal_item_count_per_patch'],
                          'goal_ids_sha256':result['goal_item_ids_sha256'],'static_goal_multiset_classes':len(result['label_vocabulary'])}))
    else:
        result=census(args.definition,args.output,deadline_seconds=args.deadline_seconds,guard=memory_guard)
        print(json.dumps({'report':str((args.output/'report.json').resolve()),'population':result['population'],'elapsed_s':result['elapsed_s']}))


if __name__=='__main__':main()
