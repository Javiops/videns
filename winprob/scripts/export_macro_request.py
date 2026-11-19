"""Export a diagnostic-only macro request using split metadata, never payloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winprob import admission

SCHEMA = 'winprob-macro-request-v1'
DEFAULT_SPLIT = ROOT.parent/'macro_coach/data/ml/split.json'
DEFAULT_ADMISSION = ROOT/'artifacts/admission/w1-admission-20260929-v2/admission_manifest.json'
DEFAULT_OUTPUT = ROOT/'artifacts/macro-request/20260930-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound_json(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    value = hashlib.sha256(raw).hexdigest()
    if path.with_suffix('.sha256').read_text(encoding='ascii').split() != [value, path.name]:
        raise ValueError('SHA256 sidecar mismatch: ' + str(path))
    return json.loads(raw.decode('utf-8-sig')), value


def split_metadata(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('provisional'):
        raise ValueError('Macro split must be frozen metadata')
    seen, folds = set(), {}
    for role in ('train', 'val', 'test'):
        ids = data.get(role)
        if (not isinstance(ids, list) or any(not isinstance(mid, str) or not admission.MATCH_ID.fullmatch(mid) for mid in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError('Invalid or overlapping macro split IDs')
        seen.update(ids)
        folds[role] = sorted(ids)
    if data.get('n_games') != len(seen):
        raise ValueError('Macro split count mismatch')
    if len(folds['train'])+len(folds['val']) > 34:
        raise ValueError('This bounded diagnostic request allows at most 34 train/validation games')
    return folds


def current_denials(manifest):
    admission.verify_manifest_sources(manifest)
    tests = set()
    root = Path(manifest['source_data_root'])
    for binding in manifest['split_metadata_sources']:
        mapping, actual = admission.read_assignment(root/binding['path'])
        if actual != binding['sha256']:
            raise ValueError('Split metadata changed during diagnostic admission')
        tests.update(mid for mid, role in mapping.items() if role == 'test')
    for binding in manifest.get('external_reservation_sources', []):
        ids, actual = admission.read_external_reservations(Path(binding['path']))
        if actual != binding:
            raise ValueError('External reservation binding changed')
        tests.update(ids)
    confirmatory = {row['match_id'] for row in manifest['games'] if row['role']=='confirmatory_reserved'}
    prior = manifest.get('prior_manifest')
    visited = set()
    while prior:
        if prior['path'] in visited:
            raise ValueError('Cyclic prior manifest ancestry')
        visited.add(prior['path'])
        old, binding = admission.read_prior_manifest(Path(prior['path']))
        if binding != prior:
            raise ValueError('Prior manifest binding changed')
        confirmatory.update(row['match_id'] for row in old['games'] if row['role']=='confirmatory_reserved')
        prior = old.get('prior_manifest')
    return tests, confirmatory


def request_population(folds, tests, confirmatory):
    games, excluded = [], []
    tests = set(tests) | set(folds['test'])
    for role in ('train', 'val'):
        for mid in folds[role]:
            reasons = []
            if mid in tests:
                reasons.append('sealed_test_reservation')
            if mid in confirmatory:
                reasons.append('confirmatory_reservation')
            if reasons:
                excluded.append({'match_id': mid, 'macro_role': role, 'reasons': reasons})
            else:
                games.append({'match_id': mid, 'macro_role': role})
    return games, excluded


def export_request(source_split=DEFAULT_SPLIT, admission_path=DEFAULT_ADMISSION, output_dir=DEFAULT_OUTPUT):
    source_split, admission_path = Path(source_split).resolve(strict=True), Path(admission_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Diagnostic request version already exists')
    source_hash = digest(source_split)
    folds = split_metadata(source_split)
    manifest, manifest_hash = read_bound_json(admission_path)
    tests, confirmatory = current_denials(manifest)
    games, excluded = request_population(folds, tests, confirmatory)
    payload = {'schema': SCHEMA, 'version': output_dir.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'source_project': 'macro_coach', 'source_split_path': str(source_split),
               'macro_split_sha256': source_hash, 'source_fold_counts': {role: len(ids) for role, ids in folds.items()},
               'admission_manifest_path': str(admission_path), 'admission_manifest_sha256': manifest_hash,
               'global_reserved_test_count': len(tests), 'global_reserved_test_ids_sha256': admission.id_digest(tests),
               'confirmatory_count': len(confirmatory), 'confirmatory_ids_sha256': admission.id_digest(confirmatory),
               'games': games, 'requested_game_ids': [row['match_id'] for row in games], 'excluded': excluded,
               'purpose': 'retrospective_review_diagnostic_only', 'fitting_authorized': False,
               'raw_payload_reads': False, 'source_macro_payload_reads': False,
               'exporter_sha256': digest(__file__)}
    if digest(source_split) != source_hash:
        raise ValueError('Macro split changed during request export')
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir/'request.json'
    raw = admission.canonical_bytes(payload)
    with path.open('xb') as handle:
        handle.write(raw)
    path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest()+'  request.json\n', encoding='ascii')
    return path, payload


def load_request(path):
    request, request_hash = read_bound_json(path)
    if (request.get('schema') != SCHEMA or request.get('purpose') != 'retrospective_review_diagnostic_only'
            or request.get('fitting_authorized') is not False):
        raise ValueError('Unknown or non-diagnostic macro request')
    source = Path(request['source_split_path']).resolve(strict=True)
    if digest(source) != request['macro_split_sha256']:
        raise ValueError('Macro split changed since request export')
    folds = split_metadata(source)
    manifest, manifest_hash = read_bound_json(request['admission_manifest_path'])
    if manifest_hash != request['admission_manifest_sha256']:
        raise ValueError('Admission changed since request export')
    tests, confirmatory = current_denials(manifest)
    games, excluded = request_population(folds, tests, confirmatory)
    if (request['games'] != games or request['excluded'] != excluded
            or request['requested_game_ids'] != [row['match_id'] for row in games]):
        raise ValueError('Requested identities differ from current allowed macro train/validation population')
    return request, request_hash, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--admission', type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path, request = export_request(args.source_split, args.admission, args.output)
    print(json.dumps({'path': str(path), 'sha256': digest(path), 'requested': len(request['games']),
                      'excluded': len(request['excluded']), 'raw_payload_reads': False}, indent=2))


if __name__ == '__main__':
    main()
