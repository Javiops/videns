"""Build the reviewed auxiliary future-goal dataset without fitting a model."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.future_goal_data import prepare


def memory_guard():
    if os.name!='nt':return
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) or value.available<.75*1024**3:
        raise RuntimeError('Auxiliary dataset RAM floor0.75GiB')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults={'base-manifest':ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json',
              'history-manifest':ROOT/'data/runs/history-cache-20260930-v1/cache/manifest.json',
              'census-report':ROOT/'data/runs/goal-census-20260930-v1/census/report.json',
              'definition':ROOT/'artifacts/goal-definition/20260930-v2/definition.json',
              'census-audit':ROOT/'data/reports/goal_census_audit_v1.json',
              'protocol':ROOT/'docs/future_goal_protocol_v1.md'}
    for name,path in defaults.items():parser.add_argument('--'+name,type=Path,default=path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--deadline-seconds',type=int,default=900)
    args=parser.parse_args();memory_guard();result=prepare(args,guard=memory_guard)
    print(json.dumps({key:result[key] for key in ('state_count','selected_count','goal_class_count','shuffle_counts','elapsed_s')},indent=2))


if __name__=='__main__':main()
