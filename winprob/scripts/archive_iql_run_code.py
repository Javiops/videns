"""Snapshot this project's bound code/protocols for a running immutable experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(experiment):
    audit_path = experiment/'input_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    destination = experiment/'code_snapshot'
    destination.mkdir(exist_ok=False)
    files = {}
    for raw, expected in audit['bindings_sha256'].items():
        source = Path(raw).resolve(strict=True)
        if not source.is_relative_to(ROOT) or source.suffix not in ('.py', '.md'):
            continue
        if digest(source) != expected:
            raise ValueError('Bound source changed before snapshot: '+str(source))
        relative = source.relative_to(ROOT)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError('Copied code digest differs')
        files[relative.as_posix()] = expected
    runtime = ROOT/'data/runs/torch-runtime-20260930-v1/requirements.txt'
    shutil.copyfile(runtime, destination/'requirements.txt')
    files['requirements.txt'] = digest(runtime)
    result = {'schema': 'iql-code-snapshot-v1', 'input_audit_sha256': digest(audit_path),
              'files': files, 'source_code_modified': False, 'raw_data_copied': False,
              'scope': 'Own-project bound Python and protocol files plus recorded environment versions; data artifacts stay versioned separately.'}
    (destination/'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'code_and_protocol_files': len(files)-1, 'directory': str(destination)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment.resolve(strict=True))
