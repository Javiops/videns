"""Freeze a 16.17 HISTORY snapshot: metadata + non-test, non-holdout raw sources.

Derived from prepare_reconstructible_1618.py (2026-09-27, prior-patch transfer, owner GO).
Besides every reserved test ID (fold `test` in any split file), every HISTORICAL HOLDOUT
(fold `val` in any split file, e.g. the 16.17 development validations) is excluded HERE,
before any payload is opened, as the owner required. The split files are found by the scan
shared with the build's plan validation, the evidence verifier and the history validator
(history_1617.split_files: every split_assignment / preserved_split_assignment / split_plan
under data/), after the first snapshot (2026-09-27 10:59) missed the 16.17 development
validations kept only as preserved_split_assignment.json / split_plan.json (153 games).
--limit N freezes only the first N eligible sources (smoke chains; recorded in the plan).
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256,write_json
from history_1617 import split_files,holdout_memberships

BASE=ROOT/'data/acceptance/16.17_history_20260927'
LIMIT=None


def static_evidence():
    """The reviewed base runtime IS patch 16.17: no cross-patch port is reviewed here.

    The pinned 16.17.1 static data is copied and digest-bound so the corpus, its
    audit and its export are tied to exactly these definitions; the review records
    the same checked item set and the rune 8304 payload as the 16.18 review did,
    against the single reviewed version (no differences to compute)."""
    folder=BASE/'inputs';folder.mkdir(parents=True,exist_ok=True)
    for name in ('item-16.17.1.json','champion-16.17.1.json','runesReforged-16.17.1.json'):
        shutil.copy2(ROOT/'data/cache'/name,folder/name)
    rune_path=folder/'runesReforged-16.17.1.json'
    rune=next(r for tree in json.loads(rune_path.read_text(encoding='utf-8'))
              for slot in tree['slots'] for r in slot['runes'] if r['id']==8304)
    payload=json.loads((folder/'item-16.17.1.json').read_text(encoding='utf-8'))
    if payload.get('version')!='16.17.1': raise ValueError('Wrong item payload version')
    data=payload['data']
    item_ids={1001,2422,2138,2139,2140,*range(1200,1210)}
    item_ids.update(int(k) for k,v in data.items() if 'Boots' in v.get('tags',[]) or int(k) in range(3865,3878))
    keys=('gold','from','into','specialRecipe','inStore','hideFromAll','consumed','consumeOnFull','stacks','maps','tags')
    report={'patch':'16.17','ddragon_version':'16.17.1','rune_source':'data/cache/runesReforged-16.17.1.json (pinned, reviewed base)',
        'review_basis':'the reviewed footwear runtime is patch 16.17 itself; sources are loaded verbatim (reconstructible_1617_runtime)',
        'rune_8304_unchanged':True,'checked_item_ids':sorted(item_ids),
        'checked_fields':list(keys),'mechanical_field_differences':{},
        'timer_economics_and_physical_rules_compatible':True,
        'rune':rune,'files':{p.name:sha256(p) for p in folder.glob('*.json')},
        'limits':['Matching static definitions do not independently verify actual server grant timing.',
            'Every game still needs source QA, prefix checks, intermediate inventory checks and decision boundary masking.']}
    write_json(folder/'static_review.json',report)
    return folder/'static_review.json'


def main():
    if (BASE/'source_plan.json').exists(): raise ValueError('Frozen source plan already exists')
    review=static_evidence()
    assignments=split_files(ROOT,exclude_dirs=[BASE])     # every split file under data/ (shared scan, history_1617)
    bindings={str(path):sha256(path) for path in assignments}
    memberships,reserved,holdouts=holdout_memberships(assignments)
    if not reserved: raise ValueError('No reserved test ids found in any split file; refuse to snapshot')
    database=ROOT/'data/tracker.db'
    conn=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=5)
    conn.row_factory=sqlite3.Row
    try:
        conn.execute('BEGIN')
        rows=[dict(r) for r in conn.execute('SELECT match_id,patch,queue_id,game_creation,game_duration FROM games WHERE patch=? ORDER BY match_id',('16.17',))]
        observed=datetime.now(timezone.utc).isoformat()
    finally: conn.close()
    selected=[];exclusions={};counts=Counter()
    for row in rows:
        mid=row['match_id']
        if mid in reserved: reason='reserved_test_metadata_only'
        elif mid in holdouts: reason='historical_validation_holdout'
        elif row['queue_id']!=420: reason='not_ranked_solo'
        elif not all((ROOT/'data/raw_match_v5'/kind/(mid+'.json.gz')).is_file() for kind in ('match','timeline')):
            reason='missing_raw_pair'
        else: reason=None
        if reason:
            exclusions[mid]=reason;counts[reason]+=1;continue
        row['prior_assignments']=memberships.get(mid,{})
        row['files']={}
        for kind in ('match','timeline'):
            path=ROOT/'data/raw_match_v5'/kind/(mid+'.json.gz')
            before=path.stat();digest=sha256(path);after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Source changed during freeze: '+mid+'/'+kind)
            row['files'][kind]={'path':str(path),'sha256':digest,'bytes':after.st_size}
        selected.append(row)
        if LIMIT and len(selected)>=LIMIT: break
    if any(sha256(Path(p))!=digest for p,digest in bindings.items()): raise ValueError('Split reservation metadata changed')
    plan={'schema':1,'purpose':'isolated_reconstruction_qualification','patch':'16.17','queue_id':420,
        'snapshot_utc':observed,'database_metadata_only':str(database),'database_games_at_snapshot':len(rows),
        'selected_sources':len(selected),'source_selection_exclusions':exclusions,'exclusion_counts':dict(counts),
        'reserved_test_ids':sorted(reserved),'historical_validation_holdout_ids':sorted(holdouts),
        'holdout_policy':'fold test or val in ANY split_assignment/preserved_split_assignment/split_plan file under data/ is excluded before any payload is opened (owner condition 1, 2026-09-27; shared scan history_1617.split_files)',
        'split_files_scanned':len(assignments),'selection_limit':LIMIT,
        'split_metadata_sha256':bindings,'matches':selected,
        'static_review':str(review),'static_review_sha256':sha256(review),
        'test_payload_reads':False,'training_or_export_authorized':False,
        'future_split_policy':'Preserve recorded prior assignments; do not silently move held-out IDs into future training.',
        'collection_continues':True,'processing_scope':'Fixed metadata snapshot; later collected games require a new snapshot.'}
    write_json(BASE/'source_plan.json',plan)
    (BASE/'source_plan.json').chmod(0o444)
    print(json.dumps({k:plan[k] for k in ('snapshot_utc','database_games_at_snapshot','selected_sources','exclusion_counts')},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE,
        help='Dated acceptance directory for this frozen snapshot (a new snapshot needs a new directory)')
    parser.add_argument('--limit',type=int,default=None,help='freeze only the first N eligible sources (smoke chains)')
    args=parser.parse_args()
    BASE=args.base.resolve();LIMIT=args.limit
    main()
