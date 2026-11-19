"""Verify the completed source-bound 16.17 HISTORY audit before any export (derived from the 16.18 verifier, 2026-09-27).

Reads frozen non-test sources and metadata only. Never edits earlier evidence.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json


def verify_prerequisite(source_root, base=None, corpus=None):
    source_root = Path(source_root).resolve()
    base = Path(base).resolve() if base else source_root / 'data/acceptance/16.17_history_20260927'
    corpus = Path(corpus).resolve() if corpus else source_root / 'data/corpora/16.17-history-v1'
    paths = {'source_plan': base / 'source_plan.json',
             'original_qualification': base / 'qualification.json',
             'audit': base / 'semantic_audit.json',
             'manifest': corpus / 'corpus_manifest.json',
             'mask': corpus / 'decision_admission.json',
             'completion': base / 'run/completion.json',
             'protocol': base / 'run/protocol.json'}
    data = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in paths.items()}
    bindings = {}

    def check(path, digest=None, size=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError('Missing prerequisite evidence: ' + str(path))
        stat = path.stat()
        if size is not None and stat.st_size != size:
            raise ValueError('Evidence size changed: ' + str(path))
        actual = bindings.get(str(path)) or sha256(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Evidence changed while hashing: ' + str(path))
        if digest is not None and actual != digest:
            raise ValueError('Evidence digest changed: ' + str(path))
        bindings[str(path)] = actual
        return actual

    completion, protocol = data['completion'], data['protocol']
    status = json.loads((base / 'run/status.json').read_text(encoding='utf-8'))
    for record in (status, completion):
        if (record.get('state') != 'complete' or record.get('stage') != 'complete'
                or record.get('completed_stages') != ['build', 'audit']
                or record.get('protected_unchanged') is not True
                or record.get('test_payload_reads') is not False
                or record.get('training_or_export') is not False):
            raise ValueError('The complete independent build/audit prerequisite has not passed')
    if status != completion:
        raise ValueError('Completion and current prerequisite status disagree')
    plan, q, audit, manifest, mask = (data[k] for k in
        ('source_plan', 'original_qualification', 'audit', 'manifest', 'mask'))
    if (plan.get('patch') != '16.17' or plan.get('test_payload_reads') is not False
            or plan.get('training_or_export_authorized') is not False
            or plan.get('purpose') != 'isolated_reconstruction_qualification'):
        raise ValueError('Wrong immutable source snapshot or changed original scope')
    selected = {r['match_id']: r for r in plan['matches']}
    if len(selected) != len(plan['matches']) or len(selected) != plan['selected_sources']:
        raise ValueError('Source plan membership is inconsistent')
    reserved = set(plan['reserved_test_ids'])
    if reserved & selected.keys():
        raise ValueError('Reserved test ID in source plan; no source payload may be read')
    from history_1617 import split_files, holdout_memberships, split_assignment
    for name, digest in plan['split_metadata_sha256'].items():
        check(name, digest)
        if any(fold in ('test', 'val') and mid in selected for mid, fold in split_assignment(name).items()):
            raise ValueError('Prior test/validation membership conflicts with the history source selection')
    if selected.keys() & set(plan.get('historical_validation_holdout_ids', [])):
        raise ValueError('Historical validation holdout in source plan; no source payload may be read')
    _, live_reserved, live_holdouts = holdout_memberships(split_files(source_root, exclude_dirs=[base]))
    if selected.keys() & (live_reserved | live_holdouts):
        raise ValueError('A split file of the repository holds a selected source as test or val')
    if (q.get('status') != 'qualified_reconstruction_snapshot' or q.get('patch') != '16.17'
            or q.get('training_clearance') is not False or q.get('export_created') is not False
            or q.get('promotion_allowed') is not False or q.get('test_payload_reads') is not False):
        raise ValueError('Wrong original reconstruction qualification or changed scope')
    check(paths['original_qualification'], completion['qualification_sha256'])
    if Path(completion['qualification']).resolve() != paths['original_qualification'].resolve():
        raise ValueError('Completion references another qualification')
    check(paths['source_plan'], protocol['source_plan_sha256'])
    if manifest.get('source_plan_sha256') != protocol['source_plan_sha256']:
        raise ValueError('Corpus and completed run bind different sources')
    check(paths['manifest'], q['corpus_manifest_sha256'])
    check(paths['audit'], q['semantic_audit_sha256'])
    if (audit.get('passed') is not True or audit.get('hard_failures')
            or audit.get('quarantined_match_ids')
            or audit.get('corpus_manifest_sha256') != q['corpus_manifest_sha256']):
        raise ValueError('Full audit does not certify this reconstructed corpus')
    accepted = set(q['accepted_match_ids'])
    if (len(accepted) != q['accepted_games'] or len(accepted) != completion['accepted_games']
            or len(accepted) != manifest['games'] or not accepted <= selected.keys()
            or accepted != set(audit['accepted_match_ids']) or accepted != set(mask['games'])):
        raise ValueError('Accepted source, audit and mask populations differ')
    if (manifest.get('patch') != '16.17' or manifest.get('recovery_version') != 'footwear-viego-16.17-v1'
            or manifest['db_sha256'] != q['db_sha256'] or audit['db_sha256'] != q['db_sha256']
            or manifest['decision_admission_sha256'] != q['decision_admission_sha256']):
        raise ValueError('Wrong reconstructed population or database/mask binding')
    if Path(q['corpus']).resolve() != corpus.resolve() or Path(q['semantic_audit']).resolve() != paths['audit'].resolve():
        raise ValueError('Qualification references different evidence')
    check(corpus / 'tracker.db', q['db_sha256'])
    check(paths['mask'], q['decision_admission_sha256'])
    for name, record in manifest['files'].items():
        path = (corpus / name).resolve()
        if not path.is_relative_to(corpus.resolve()):
            raise ValueError('Corpus file escaped its frozen directory')
        check(path, record['sha256'], record.get('bytes'))
    for name, digest in protocol['code_sha256'].items():
        path = (base / 'run/code' / name).resolve()
        if not path.is_relative_to((base / 'run/code').resolve()):
            raise ValueError('Code snapshot escaped its frozen directory')
        check(path, digest)
    for name, digest in manifest['reconstruction_code_sha256'].items():
        if protocol['code_sha256'].get(name) != digest:
            raise ValueError('Reconstruction code is not the completed supervisor code: ' + name)
    check(base / 'audit_code.py', audit['audit_code_sha256'])
    for name, digest in protocol['protected_sha256'].items():
        path = Path(name)
        if digest is None:
            if path.exists():
                raise ValueError('Previously absent protected artifact appeared: ' + name)
        else:
            check(path, digest)
    review_path = Path(plan['static_review']).resolve()
    paths['static_review'] = review_path
    check(review_path, plan['static_review_sha256'])
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if (review.get('patch') != '16.17' or review.get('ddragon_version') != '16.17.1'
            or review.get('timer_economics_and_physical_rules_compatible') is not True):
        raise ValueError('Pinned static mechanics did not pass review')
    for name, digest in review['files'].items():
        check(review_path.parent / name, digest)
    check(source_root / 'data/cache/item-16.17.1.json', review['files']['item-16.17.1.json'])
    # Recompute the exact audited generated variants without activating them.
    from reconstructible_1617_runtime import declaration
    if declaration() != manifest['footwear_runtime']:
        raise ValueError('Execution checkout would change the audited 16.17 semantics')
    for mid, record in selected.items():
        if record.get('patch') != '16.17' or record.get('queue_id') != 420:
            raise ValueError('Source selection has mixed patch or queue')
        for kind in ('match', 'timeline'):
            raw = record['files'][kind]
            path = Path(raw['path']).resolve()
            expected = (source_root / 'data/raw_match_v5' / kind / (mid + '.json.gz')).resolve()
            if path != expected:
                raise ValueError('Unexpected source path for ' + mid)
            check(path, raw['sha256'], raw['bytes'])
    for name, path in paths.items():
        check(path)
    return {**data, 'source_root': source_root, 'corpus': corpus,
            'paths': paths, 'bindings': bindings, 'verified_utc': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path, help='acceptance directory (default: the 16.17 history snapshot)')
    parser.add_argument('--corpus', type=Path, help='reconstructed corpus (default: 16.17-history-v1)')
    args = parser.parse_args()
    result = verify_prerequisite(args.source_root, base=args.base, corpus=args.corpus)
    report = {'schema': 1, 'passed': True, 'verified_utc': result['verified_utc'],
              'accepted_games': result['original_qualification']['accepted_games'],
              'admitted_rows': result['original_qualification']['admitted_rows'],
              'test_payload_reads': False, 'source_root': str(result['source_root']),
              'bindings': result['bindings'], 'verifier_sha256': sha256(Path(__file__))}
    if args.out.exists():
        raise ValueError('Use a fresh prerequisite verification report')
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'bindings'}, indent=2))


if __name__ == '__main__':
    main()
