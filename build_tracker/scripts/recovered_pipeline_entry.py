"""Activate the isolated 16.17 runtime before an offline data-gate entry."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True, choices=(
        'qualify_patch_corpus', 'report_corpus_coverage', 'publish_patch_dataset'))
    args, remaining = parser.parse_known_args()
    from footwear_runtime import activate
    activate()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    path = ROOT / 'scripts' / (args.entry + '.py')
    sys.argv = [str(path), *remaining]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
