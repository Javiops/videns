"""Admit the second 16.18 development benchmark with a fresh forward validation window.

Owner instruction (2026-09-13): run the acceptance gates on the new collection,
then publish a fresh temporal development generation whose validation window is
unshopped — no prior selection round has ever scored a game in it. Policy
`forward-validation-v1`:

- boundary = the FIRST reconstructible snapshot instant (2026-09-12T11:03 UTC,
  read from the v1 source plan, never retyped). No decision taken during the
  September 12 development day could have observed any game created after it.
- train:  game_creation <= boundary, and never previously assigned to any
  validation or test fold. Games the v1 benchmark deferred as
  `after_original_validation_window` but that predate the boundary enter
  training here, with that provenance recorded.
- val:    game_creation > boundary. Such games can hold no prior assignment;
  this is asserted, not assumed.
- excluded entirely: every game with a prior `val` assignment (development-
  exposed by the v1 selection rounds) and, as always, every reserved test ID
  before any payload access.

This is development evidence only: no final-test materialization, no
promotion, no serving change. The prior 151-game validation window is retired
from selection use, not deleted.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json, publish_directory
from app.dataset_generations import validate_1618_split, validate_1618_admission
from prepare_benchmark_1618 import inventory_qualification
from verify_reconstructible_1618_evidence import verify_prerequisite

POLICY = 'forward-validation-v1'
AUTHORIZATION = 'owner instructed steps 1-3 on 2026-09-13'
AUTHORIZATION_DOCUMENT = 'HANDOFF.md'
V1_BASE = ROOT / 'data/acceptance/16.18_reconstructible_20260912'
V1_GENERATION = ROOT / 'data/dataset_generations/16.18-reconstructible-development-v1'


EXPLICIT_BOUNDARY_UTC = None          # set from --forward-boundary-utc; None keeps the original rule
EXPLICIT_BOUNDARY_SOURCE = None       # document that declares it, digest-bound in split_metadata


def forward_boundary_ms():
    """The v1 snapshot instant, read from its frozen plan; never retyped.

    Design-B expansion (owner GO 2026-09-23): an EXPLICIT boundary may be
    declared on the command line, bound to the plan document that declares
    it. The prior-snapshot rule stays the default and is untouched.
    """
    if EXPLICIT_BOUNDARY_UTC is not None:
        stamp = datetime.fromisoformat(EXPLICIT_BOUNDARY_UTC)
        if stamp.tzinfo is None:
            raise ValueError('explicit forward boundary must carry a timezone')
        return int(stamp.timestamp() * 1000), EXPLICIT_BOUNDARY_UTC
    plan = json.loads((V1_BASE / 'source_plan.json').read_text(encoding='utf-8'))
    stamp = datetime.fromisoformat(plan['snapshot_utc'])
    return int(stamp.timestamp() * 1000), plan['snapshot_utc']


def freeze_forward_split(source_plan, accepted_ids):
    boundary_ms, boundary_utc = forward_boundary_ms()
    rows = {row['match_id']: row for row in source_plan['matches']}
    if len(rows) != len(source_plan['matches']) or not set(accepted_ids) <= rows.keys():
        raise ValueError('Qualified source membership changed')
    assignment, excluded = {}, {}
    for mid in sorted(accepted_ids):
        row = rows[mid]
        prior = set(row.get('prior_assignments', {}).values())
        if mid in source_plan['reserved_test_ids'] or 'test' in prior:
            raise ValueError('Reserved test source reached benchmark preparation')
        if row['game_creation'] <= boundary_ms:
            if 'val' in prior:
                excluded[mid] = ['prior_validation_exposed']
            elif prior - {'train'}:
                excluded[mid] = ['conflicting_prior_assignment']
            else:
                assignment[mid] = 'train'
        else:
            if prior:
                raise ValueError(f'A post-boundary game carries prior assignments: {mid}')
            assignment[mid] = 'val'
    folds = Counter(assignment.values())
    if not folds['train'] or not folds['val']:
        raise ValueError('Forward split needs non-empty training and validation windows')
    metadata = {'schema': 1, 'purpose': 'patch_development_1618',
        'policy': POLICY, 'forward_boundary_ms': boundary_ms, 'forward_boundary_utc': boundary_utc,
        'boundary_source': (str(EXPLICIT_BOUNDARY_SOURCE) if EXPLICIT_BOUNDARY_UTC is not None
                            else str(V1_BASE / 'source_plan.json')),
        'boundary_source_sha256': sha256(Path(EXPLICIT_BOUNDARY_SOURCE) if EXPLICIT_BOUNDARY_UTC is not None
                                         else V1_BASE / 'source_plan.json'),
        'boundary_rule': ('explicit forward boundary declared by the owner plan'
                          if EXPLICIT_BOUNDARY_UTC is not None else 'prior snapshot instant'),
        'reserved_test_ids': sorted(source_plan['reserved_test_ids']),
        'reserved_test_games': len(source_plan['reserved_test_ids']),
        'validation_exposure': 'fresh forward development validation; unshopped at admission',
        'retired_validation': (f'the prior development-exposed validation window '
                               f'({V1_GENERATION.name}) is excluded from training and validation'),
        'prior_benchmark': V1_GENERATION.name,
        'prior_generation_manifest_sha256': sha256(V1_GENERATION / 'generation_manifest.json'),
        'final_test_payload_opened': False, 'final_test_materialized': False,
        'assignment_policy': ('Training ends at the frozen prior snapshot instant; validation is '
                              'every qualified newer game; prior validation/test never enter any fold.'),
        'source_snapshot_utc': source_plan['snapshot_utc'],
        'split_metadata_sha256': source_plan['split_metadata_sha256']}
    validate_1618_split(assignment, metadata, source_plan)
    return assignment, excluded, metadata


def prepare(source_root, base, corpus_path, destination, benchmark_id, eval_version):
    if destination.exists() or not benchmark_id or type(eval_version) is not int or eval_version < 1:
        raise ValueError('Use a new benchmark admission directory and explicit identity/version')
    verified = verify_prerequisite(source_root, base=base, corpus=corpus_path)
    source, original, manifest, corpus = (verified[k] for k in
        ('source_plan', 'original_qualification', 'manifest', 'corpus'))
    assignment, excluded, metadata = freeze_forward_split(source, original['accepted_match_ids'])
    # Freeze identity before reading any selected payload; reserved tests and
    # excluded games cannot be reached by the inventory qualification below.
    inventory_excluded, inventory = inventory_qualification(corpus, manifest, set(assignment))
    excluded.update(inventory_excluded)
    assignment = {mid: fold for mid, fold in assignment.items() if mid not in inventory_excluded}
    validate_1618_split(assignment, metadata, source)
    metadata.update(benchmark_id=benchmark_id, eval_version=eval_version,
                    temporal_and_inventory_exclusions=excluded)
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir(parents=True)
    paths = verified['paths']
    copies = {'source_plan.json': paths['source_plan'],
        'original_qualification.json': paths['original_qualification'],
        'original_completion.json': paths['completion'], 'original_protocol.json': paths['protocol'],
        'original_static_review.json': base / 'inputs/static_review.json',
        'corpus_manifest.json': paths['manifest'], 'semantic_audit.json': paths['audit'],
        'decision_admission.json': paths['mask'],
        'audit_code.py': paths['audit'].parent / 'audit_code.py',
        'qualification_code.py': Path(__file__),
        'original_split_plan.json': V1_GENERATION / 'split_plan.json',
        'original_split_assignment.json': V1_GENERATION / 'split_assignment.json',
        'original_generation_manifest.json': V1_GENERATION / 'generation_manifest.json'}
    for name, path in copies.items():
        shutil.copyfile(path, pending / name)
    write_json(pending / 'preserved_split_assignment.json', assignment)
    write_json(pending / 'split_metadata.json', metadata)
    files = {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in pending.iterdir()}
    admission = {'schema': 1, 'purpose': 'patch_development_1618', 'patch': '16.18',
        'benchmark_id': benchmark_id, 'eval_version': eval_version,
        'authorization': AUTHORIZATION,
        'authorization_document': AUTHORIZATION_DOCUMENT,
        'authorization_document_sha256': sha256(ROOT / AUTHORIZATION_DOCUMENT),
        'source_root': str(source_root), 'source_scope_unchanged': True,
        'final_test_payload_opened': False,
        'source_binding_verification': ('Complete v2 corpus/source/static/code hashes verified before '
                                        'selected payload reads.'),
        'files': files, 'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(pending / 'benchmark_admission.json', admission)
    q = {'schema': 3, 'status': 'qualified_subset', 'purpose': 'isolated_experiment_only',
        'benchmark_purpose': 'patch_development_1618', 'benchmark_id': benchmark_id,
        'eval_version': eval_version, 'patch': '16.18', 'corpus': str(corpus),
        'corpus_manifest_sha256': files['corpus_manifest.json']['sha256'],
        'db_sha256': manifest['db_sha256'],
        'semantic_audit': str(destination / 'semantic_audit.json'),
        'semantic_audit_sha256': files['semantic_audit.json']['sha256'],
        'source_qualification_sha256': files['original_qualification.json']['sha256'],
        'qualification_code_sha256': files['qualification_code.py']['sha256'],
        'benchmark_admission_sha256': sha256(pending / 'benchmark_admission.json'),
        'decision_admission_sha256': files['decision_admission.json']['sha256'],
        'source_games': manifest['games'], 'quality_accepted_games': len(assignment),
        'accepted_match_ids': sorted(assignment), 'quarantined_match_ids': excluded,
        'quarantined_games': len(excluded),
        'source_exclusion_reason_counts': original['source_exclusions'],
        'exclusion_reason_counts': dict(Counter(r for rs in excluded.values() for r in rs)),
        'reconstruction_version': manifest['reconstruction_version'],
        'inventory_version': manifest['inventory_version'],
        'recovery_version': manifest['recovery_version'],
        'training_clearance': 'authorized benchmark IDs only',
        'production_hold_cleared': False, 'promotion_allowed': False, **inventory,
        'limits': ['Validation is unshopped at admission; each later selection round spends it.',
                   'Footwear delivery is inferred with the fixed 1000 ms boundary guard.',
                   'Reviewed Viego ownership admission is preserved; no blanket champion exclusion is applied.',
                   'Reconstructed wallet legality is not exact live-wallet evidence.',
                   'Never compare headline metrics against the v1 export as model improvement.']}
    write_json(pending / 'qualification.json', q)
    validate_1618_admission(pending, q)
    for path in pending.iterdir():
        path.chmod(0o444)
    publish_directory(pending, destination)
    print(json.dumps({'qualification': str(destination / 'qualification.json'),
                      'games_by_split': dict(Counter(assignment.values())),
                      'excluded_games': len(excluded), 'promotion_allowed': False}), flush=True)
    return destination / 'qualification.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True,
                        help='THIS snapshot acceptance directory (e.g. the 20260913 one for v2)')
    parser.add_argument('--corpus', type=Path, required=True,
                        help='the reconstructed corpus this admission scopes')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--benchmark-id', required=True)
    parser.add_argument('--eval-version', type=int, required=True)
    parser.add_argument('--prior-base', type=Path, default=V1_BASE,
                        help='PRIOR snapshot acceptance directory; its instant is the forward '
                             'boundary (training ends there, newer qualified games become the '
                             'fresh validation)')
    parser.add_argument('--forward-boundary-utc', default=None,
        help='ISO-8601 instant with timezone; train <= boundary < val. Default: the prior snapshot instant')
    parser.add_argument('--boundary-document', type=Path, default=ROOT / 'docs/expansion_plan_v1.md',
        help='document that declares the explicit boundary; its sha256 is bound into split_metadata')
    parser.add_argument('--authorization', default=None,
        help='owner authorization literal for this admission (must be accepted by validate_1618_admission)')
    parser.add_argument('--authorization-document', default='HANDOFF.md',
        help='repo-relative document recording the authorization; digest-bound')
    parser.add_argument('--prior-generation', type=Path, default=V1_GENERATION,
                        help='PRIOR published generation; its validation fold is development-'
                             'exposed and retired from every fold, and its documents are bound '
                             'as the original_* evidence')
    args = parser.parse_args()
    V1_BASE = args.prior_base.resolve()
    V1_GENERATION = args.prior_generation.resolve()
    if args.forward_boundary_utc:
        EXPLICIT_BOUNDARY_UTC = args.forward_boundary_utc
        EXPLICIT_BOUNDARY_SOURCE = args.boundary_document.resolve()
        if not EXPLICIT_BOUNDARY_SOURCE.is_file():
            raise ValueError('the document declaring the explicit boundary must exist')
    if args.authorization:
        AUTHORIZATION = args.authorization
        AUTHORIZATION_DOCUMENT = args.authorization_document
    prepare(args.source_root.resolve(), args.base.resolve(), args.corpus.resolve(),
            args.out.resolve(), args.benchmark_id, args.eval_version)
