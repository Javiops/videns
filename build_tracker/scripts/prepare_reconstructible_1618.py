"""Freeze a metadata snapshot and non-test raw sources while collection continues."""
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

BASE=ROOT/'data/acceptance/16.18_reconstructible_20260912'


def static_evidence():
    folder=BASE/'inputs';folder.mkdir(parents=True,exist_ok=True)
    import httpx
    rune_path=folder/'runesReforged-16.18.1.json'
    url='https://ddragon.leagueoflegends.com/cdn/16.18.1/data/en_US/runesReforged.json'
    if not rune_path.exists():
        response=httpx.get(url,timeout=30)
        response.raise_for_status()
        payload=response.json()
        if not isinstance(payload,list): raise ValueError('Invalid pinned rune definitions')
        rune_path.write_bytes(response.content)
    old_runes=ROOT/'data/acceptance/footwear_repair_20260911/sources/runesReforged-16.17.1.json'
    def rune(path):
        return next(r for tree in json.loads(path.read_text(encoding='utf-8'))
                    for slot in tree['slots'] for r in slot['runes'] if r['id']==8304)
    before,after=rune(old_runes),rune(rune_path)
    versions={}
    for patch in ('16.17.1','16.18.1'):
        path=ROOT/'data/cache'/('item-'+patch+'.json')
        payload=json.loads(path.read_text(encoding='utf-8'))
        if payload.get('version')!=patch: raise ValueError('Wrong item payload version')
        versions[patch]=payload['data']
        shutil.copy2(path,folder/path.name)
    item_ids={1001,2422,2138,2139,2140,*range(1200,1210)}
    for data in versions.values():
        item_ids.update(int(k) for k,v in data.items() if 'Boots' in v.get('tags',[]) or int(k) in range(3865,3878))
    keys=('gold','from','into','specialRecipe','inStore','hideFromAll','consumed','consumeOnFull','stacks','maps','tags')
    differences={}
    for item in sorted(item_ids):
        left=versions['16.17.1'].get(str(item),{});right=versions['16.18.1'].get(str(item),{})
        changed={k:{'before':left.get(k),'after':right.get(k)} for k in keys if left.get(k)!=right.get(k)}
        if changed: differences[str(item)]=changed
    reviewed=before==after and not differences
    report={'patch':'16.18','ddragon_version':'16.18.1','rune_source_url':url,
        'patch_notes_url':'https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-18-notes/',
        'rune_8304_unchanged':before==after,'checked_item_ids':sorted(item_ids),
        'checked_fields':list(keys),'mechanical_field_differences':differences,
        'timer_economics_and_physical_rules_compatible':reviewed,
        'rune':after,'files':{p.name:sha256(p) for p in folder.glob('*.json')},
        'limits':['Matching static definitions do not independently verify actual server grant timing.',
            'Every game still needs source QA, prefix checks, intermediate inventory checks and decision boundary masking.']}
    write_json(folder/'static_review.json',report)
    if not reviewed: raise ValueError('Pinned static mechanics changed; inspect static_review.json before porting')
    return folder/'static_review.json'


def main():
    if (BASE/'source_plan.json').exists(): raise ValueError('Frozen source plan already exists')
    review=static_evidence()
    assignments=[ROOT/'data/ml/split_assignment.json',*(ROOT/'data/dataset_generations').glob('*/split_assignment.json')]
    memberships={};bindings={};reserved=set()
    for path in assignments:
        if not path.is_file(): continue
        mapping=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(mapping,dict): raise ValueError('Invalid existing split metadata')
        bindings[str(path)]=sha256(path)
        for mid,fold in mapping.items():
            if not isinstance(fold,str): raise ValueError('Invalid split assignment')
            memberships.setdefault(mid,{})[str(path)]=fold
            if fold=='test': reserved.add(mid)
    database=ROOT/'data/tracker.db'
    conn=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=5)
    conn.row_factory=sqlite3.Row
    try:
        conn.execute('BEGIN')
        rows=[dict(r) for r in conn.execute('SELECT match_id,patch,queue_id,game_creation,game_duration FROM games WHERE patch=? ORDER BY match_id',('16.18',))]
        observed=datetime.now(timezone.utc).isoformat()
    finally: conn.close()
    selected=[];exclusions={};counts=Counter()
    for row in rows:
        mid=row['match_id']
        if mid in reserved: reason='reserved_test_metadata_only'
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
    if any(sha256(Path(p))!=digest for p,digest in bindings.items()): raise ValueError('Split reservation metadata changed')
    plan={'schema':1,'purpose':'isolated_reconstruction_qualification','patch':'16.18','queue_id':420,
        'snapshot_utc':observed,'database_metadata_only':str(database),'database_games_at_snapshot':len(rows),
        'selected_sources':len(selected),'source_selection_exclusions':exclusions,'exclusion_counts':dict(counts),
        'reserved_test_ids':sorted(reserved),'split_metadata_sha256':bindings,'matches':selected,
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
    BASE=parser.parse_args().base.resolve()
    main()
