"""Evaluate the fixed ordinary-shop recommendation trial from versioned artifacts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winprob.ordinary_evaluation_io import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'actor-manifest', 'reference-manifest', 'audit-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = run_evaluation(args.source_manifest, args.actor_manifest, args.reference_manifest, args.audit_manifest, args.output)
    print(json.dumps({key: report[key] for key in ('decision_bearing_games', 'available_prediction_games', 'arms',
                      'primary_contrasts', 'gates', 'useful_development_reranking_gate_passed', 'combined_elapsed_s')}, indent=2))


if __name__ == '__main__': main()
