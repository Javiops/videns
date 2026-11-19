"""Activate the reviewed 16.19 runtime before isolated export consumers."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True, choices=('report_corpus_coverage', 'publish_patch_dataset'))
    parser.add_argument('--corpus', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    manifest = json.loads((args.corpus / 'corpus_manifest.json').read_text(encoding='utf-8'))
    from footwear_runtime import activate_for_manifest
    activate_for_manifest(manifest, corpus_path=args.corpus)
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    path = ROOT / 'scripts' / (args.entry + '.py')
    sys.argv = [str(path), *remaining]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
