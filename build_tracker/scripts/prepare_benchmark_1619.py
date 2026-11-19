"""16.19 twin of prepare_benchmark_1618.py (2026-09-29): only inventory_qualification is used by prepare_development_1619.py."""
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
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256, write_json, publish_directory
from app.dataset_generations import validate_1618_split, validate_1618_admission


def freeze_split(source_plan, accepted_ids, original_plan):
    """Only IDs, timestamps and old fold assignments are consulted here."""
    original, stamps = original_plan['assignment'], original_plan['game_creation_ms']
    train_end = max(stamps[mid] for mid,fold in original.items() if fold == 'train')
    val_end = max(stamps[mid] for mid,fold in original.items() if fold == 'val')
    rows = {row['match_id']:row for row in source_plan['matches']}
    if len(rows) != len(source_plan['matches']) or not set(accepted_ids) <= rows.keys():
        raise ValueError('Qualified source membership changed')
    assignment, excluded = {}, {}
    for mid in sorted(accepted_ids):
        row = rows[mid]
        prior = set(row.get('prior_assignments',{}).values())
        if mid in source_plan['reserved_test_ids'] or 'test' in prior:
            raise ValueError('Reserved test source reached benchmark preparation')
        expected = 'train' if row['game_creation'] <= train_end else 'val' if row['game_creation'] <= val_end else None
        if expected is None:
            excluded[mid] = ['after_original_validation_window']
        elif expected == 'train' and 'val' in prior:
            excluded[mid] = ['prior_validation_reserved_from_training']
        elif mid in original and original[mid] != expected:
            raise ValueError('Original assignment conflicts with source chronology')
        elif prior - {expected}:
            excluded[mid] = ['conflicting_prior_assignment']
        else:
            assignment[mid] = expected
    metadata = {'schema':1,'purpose':'patch_development_1619',
        'policy':'preserve-original-temporal-windows-v1','train_end_ms':train_end,'val_end_ms':val_end,
        'original_assignment':original,'original_game_creation_ms':stamps,
        'reserved_test_ids':sorted(source_plan['reserved_test_ids']),
        'reserved_test_games':len(source_plan['reserved_test_ids']),
        'validation_exposure':'reused development validation; not confirmatory',
        'final_test_payload_opened':False,'final_test_materialized':False,
        'assignment_policy':'Preserve old train/validation temporal windows; prior validation/test never enter training; later sources deferred.',
        'source_snapshot_utc':source_plan['snapshot_utc'],'split_metadata_sha256':source_plan['split_metadata_sha256']}
    validate_1618_split(assignment,metadata,source_plan)
    return assignment,excluded,metadata


def inventory_qualification(corpus, manifest, selected):
    """Check physical occupancy independently for every admitted game's three states."""
    from footwear_runtime import activate_for_manifest
    activate_for_manifest(manifest,corpus_path=corpus)
    from app.ddragon import dragon_for_patch
    from app.riot import RawMatchArchive
    from app.shop_econ import payload_ids
    from qualify_patch_corpus import inventory_shape, inventory_issues, first_bot_quests
    from baseline import _unmodeled_regular_slots
    dragon = dragon_for_patch('16.19')
    archive = RawMatchArchive(corpus/'raw')
    excluded, checked, max_copies, max_slots = {},0,0,0
    conn = sqlite3.connect((corpus/'tracker.db').as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        for index,mid in enumerate(sorted(selected),1):
            quests = first_bot_quests(archive.read('timeline',mid))
            reasons, game_copies, game_slots = set(),0,0
            events = 0
            for (payload,) in conn.execute('SELECT payload_json FROM game_events WHERE match_id=?',(mid,)):
                event = json.loads(payload)
                events += 1
                anchor = event['ts']-15000 if event.get('is_save') else event['ts']
                shopper = next(member for member in event['board'] if member['participant_id']==event['participant_id'])
                states = [('board_before',member,member['items'],anchor,False) for member in event['board']]
                states += [('shop_before',shopper,event['inventory_before'],anchor,False),
                           ('shop_after',shopper,event['inventory_after'],event['ts_end'],True)]
                for _state,member,items,when,inclusive in states:
                    quest = quests.get(member['participant_id'],float('inf'))
                    completed = quest <= when if inclusive else quest < when
                    shape = inventory_shape(items,member['team_position'],completed,dragon)
                    issues = inventory_issues(items,member['team_position'],completed,dragon)
                    occupied = shape['regular_slots'] + _unmodeled_regular_slots(items,dragon)
                    if occupied > 6:
                        issues.add('total_regular_slot_overflow')
                    reasons.update(issues)
                    game_copies = max(game_copies,shape['modeled_item_copies'])
                    game_slots = max(game_slots,occupied)
                if Counter(payload_ids(event.get('label_bought',[]))) - Counter(payload_ids(event['inventory_after'])):
                    reasons.add('purchase_label_absent_from_inventory_after')
            if not events:
                reasons.add('missing_game_events')
            checked += events
            if reasons:
                excluded[mid] = sorted(reasons)
            else:
                max_copies,max_slots = max(max_copies,game_copies),max(max_slots,game_slots)
            if index % 100 == 0:
                print(f'16.19 benchmark inventory qualification: {index}/{len(selected)} games',flush=True)
    finally:
        conn.close()
    return excluded,{'inventory_constraint_checked_events':checked,
        'inventory_constraint_states':['board_before','shop_before','shop_after'],
        'accepted_max_modeled_item_copies':max_copies,'accepted_max_regular_slots':max_slots}


def prepare(source_root, destination, benchmark_id, eval_version):
    if destination.exists() or not benchmark_id or type(eval_version) is not int or eval_version < 1:
        raise ValueError('Use a new benchmark admission directory and explicit identity/version')
    from verify_reconstructible_1619_evidence import verify_prerequisite
    verified = verify_prerequisite(source_root)
    source,original,manifest,corpus = (verified[k] for k in ('source_plan','original_qualification','manifest','corpus'))
    original_dir = source_root/'data/dataset_generations/16.19-initial-smoke-v2'
    original_plan_path = original_dir/'split_plan.json'
    original_plan = json.loads(original_plan_path.read_text(encoding='utf-8'))
    assignment,excluded,metadata = freeze_split(source,original['accepted_match_ids'],original_plan)
    # Freeze identity before reading any selected game's payload. Later sources
    # and all old test IDs cannot be reached by inventory qualification below.
    inventory_excluded,inventory = inventory_qualification(corpus,manifest,set(assignment))
    excluded.update(inventory_excluded)
    assignment = {mid:fold for mid,fold in assignment.items() if mid not in inventory_excluded}
    validate_1618_split(assignment,metadata,source)
    metadata.update(benchmark_id=benchmark_id,eval_version=eval_version,
                    temporal_and_inventory_exclusions=excluded)
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir(parents=True)
    paths = verified['paths']
    copies = {'source_plan.json':paths['source_plan'],'original_qualification.json':paths['original_qualification'],
        'original_completion.json':paths['completion'],'original_protocol.json':paths['protocol'],
        'original_static_review.json':paths['static_review'],'corpus_manifest.json':paths['manifest'],
        'semantic_audit.json':paths['audit'],'decision_admission.json':paths['mask'],
        'audit_code.py':paths['audit'].parent/'audit_code.py','qualification_code.py':Path(__file__),
        'original_split_plan.json':original_plan_path,
        'original_split_assignment.json':original_dir/'split_assignment.json',
        'original_generation_manifest.json':original_dir/'generation_manifest.json'}
    for name,path in copies.items():
        shutil.copyfile(path,pending/name)
    write_json(pending/'preserved_split_assignment.json',assignment)
    write_json(pending/'split_metadata.json',metadata)
    files = {path.name:{'sha256':sha256(path),'bytes':path.stat().st_size} for path in pending.iterdir()}
    admission = {'schema':1,'purpose':'patch_development_1619','patch':'16.19',
        'benchmark_id':benchmark_id,'eval_version':eval_version,
        'authorization':'owner authorized steps 2-4 on 2026-09-12',
        'authorization_document':'docs/experiment_16_18_continuation.md',
        'authorization_document_sha256':sha256(ROOT/'docs/experiment_16_18_continuation.md'),
        'source_root':str(source_root),'source_scope_unchanged':True,'final_test_payload_opened':False,
        'source_binding_verification':'Complete original corpus/source/static/code hashes verified before selected payload reads.',
        'files':files,'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(pending/'benchmark_admission.json',admission)
    q = {'schema':3,'status':'qualified_subset','purpose':'isolated_experiment_only',
        'benchmark_purpose':'patch_development_1619','benchmark_id':benchmark_id,'eval_version':eval_version,
        'patch':'16.19','corpus':str(corpus),'corpus_manifest_sha256':files['corpus_manifest.json']['sha256'],
        'db_sha256':manifest['db_sha256'],'semantic_audit':str(destination/'semantic_audit.json'),
        'semantic_audit_sha256':files['semantic_audit.json']['sha256'],
        'source_qualification_sha256':files['original_qualification.json']['sha256'],
        'qualification_code_sha256':files['qualification_code.py']['sha256'],
        'benchmark_admission_sha256':sha256(pending/'benchmark_admission.json'),
        'decision_admission_sha256':files['decision_admission.json']['sha256'],
        'source_games':manifest['games'],'quality_accepted_games':len(assignment),
        'accepted_match_ids':sorted(assignment),'quarantined_match_ids':excluded,'quarantined_games':len(excluded),
        'source_exclusion_reason_counts':original['source_exclusions'],
        'exclusion_reason_counts':dict(Counter(reason for reasons in excluded.values() for reason in reasons)),
        'reconstruction_version':manifest['reconstruction_version'],'inventory_version':manifest['inventory_version'],
        'recovery_version':manifest['recovery_version'],'training_clearance':'authorized benchmark IDs only',
        'production_hold_cleared':False,'promotion_allowed':False,**inventory,
        'limits':['Development validation reuses prior windows; later games remain deferred.',
                  'Footwear delivery is inferred with the fixed 1000 ms boundary guard.',
                  'Reviewed Viego ownership admission is preserved; no blanket champion exclusion is applied.',
                  'Reconstructed wallet legality is not exact live-wallet evidence.']}
    write_json(pending/'qualification.json',q)
    validate_1618_admission(pending,q)
    for path in pending.iterdir():
        path.chmod(0o444)
    publish_directory(pending,destination)
    print(json.dumps({'qualification':str(destination/'qualification.json'),'games_by_split':dict(Counter(assignment.values())),
                      'deferred_or_excluded_games':len(excluded),'promotion_allowed':False}),flush=True)
    return destination/'qualification.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--benchmark-id',required=True)
    parser.add_argument('--eval-version',type=int,required=True)
    args = parser.parse_args()
    prepare(args.source_root.resolve(),args.out.resolve(),args.benchmark_id,args.eval_version)
