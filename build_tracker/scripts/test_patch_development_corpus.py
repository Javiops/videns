"""A development corpus contains only reconciled games with frozen fold evidence."""
from collections import Counter
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import build_patch_development_corpus as builder
from app.dataset_artifacts import canonical_sha, sha256, write_json
from app.riot import RawMatchArchive


def refused(fn, message):
    try:
        fn()
    except (ValueError,KeyError,TypeError):
        return
    raise AssertionError(message)


def pair(mid,created=1000000):
    participants=[{
        'participantId':pid,'puuid':f'{mid}-player-{pid}',
        'championId':103,'championName':'Ahri','teamId':100 if pid<=5 else 200,
        'teamPosition':'MIDDLE','win':pid<=5,'kills':0,'deaths':0,'assists':0,
        'perks':{'styles':[{'selections':[{'perk':8112}]}]},
        'item0':1001,'item1':0,'item2':0,'item3':0,'item4':0,'item5':0,'item6':0,
    } for pid in range(1,11)]
    match={'metadata':{'matchId':mid},'info':{
        'gameVersion':'16.17.1','queueId':420,'gameDuration':600,
        'gameCreation':created,'participants':participants,
    }}
    timeline={'metadata':{'matchId':mid},'info':{
        'participants':[{'participantId':p['participantId'],'puuid':p['puuid']} for p in participants],
        'frames':[{
            'timestamp':ts,'participantFrames':{str(pid):{
                'participantId':pid,'currentGold':500,'totalGold':1000,'level':5,
                'minionsKilled':20,'jungleMinionsKilled':0,
            } for pid in range(1,11)},
            'events':[{'type':'ITEM_PURCHASED','participantId':pid,'itemId':1001,'timestamp':120000}
                      for pid in range(1,11)] if ts==120000 else [],
        } for ts in range(0,600001,60000)],
    }}
    return match,timeline


base_match,base_timeline=pair('KR_FILTER')
assert builder.source_exclusions(base_match,base_timeline)==set()
for pid in (1,6,10):
    for rule in ('footwear','viego','undo'):
        match,timeline=deepcopy(base_match),deepcopy(base_timeline)
        if rule=='footwear':
            match['info']['participants'][pid-1]['perks']['styles'][-1]['selections'].append({'perk':8304})
            expected='unobserved_magical_footwear_grant'
        elif rule=='viego':
            match['info']['participants'][pid-1].update(championId=234,championName='Viego')
            expected='unmodeled_viego_inventory'
        else:
            timeline['info']['frames'][3]['events'].append({'type':'ITEM_UNDO','participantId':pid,'timestamp':170000,'beforeId':0,'afterId':0})
            expected='undo_without_item_ids'
        assert expected in builder.source_exclusions(match,timeline), (pid,rule)
known_undo=deepcopy(base_timeline)
known_undo['info']['frames'][3]['events'].append({'type':'ITEM_UNDO','participantId':10,'timestamp':170000,'beforeId':1001,'afterId':0})
assert 'undo_without_item_ids' not in builder.source_exclusions(base_match,known_undo)

with tempfile.TemporaryDirectory(prefix='development-corpus-',dir=ROOT/'data') as temp:
    root=Path(temp)
    original_db=builder.db.DB_PATH
    try:
        for name in builder.CODE:
            path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,path)
        archive=RawMatchArchive(root/'data/raw_match_v5')
        metadata=root/'selection'
        metadata.mkdir()
        entries=[]
        cases={
            'KR_TRAIN':('train',None),'KR_VAL':('val',None),
            'KR_FOOTWEAR':('train','footwear'),'KR_VIEGO':('train','viego'),
            'KR_UNDO':('train','undo'),'KR_TERMINAL':('train','terminal'),
            'KR_OVERFLOW':('train','overflow'),'KR_WRONGPATCH':('train','wrong_patch'),
        }
        for mid,(split,kind) in cases.items():
            match,timeline=pair(mid,2000000 if split=='val' else 1000000)
            if kind=='footwear':
                match['info']['participants'][9]['perks']['styles'][0]['selections'].append({'perk':8304})
            elif kind=='viego':
                match['info']['participants'][8].update(championId=234,championName='Viego')
            elif kind=='undo':
                timeline['info']['frames'][3]['events'].append({'type':'ITEM_UNDO','participantId':8,'timestamp':170000,'beforeId':0,'afterId':0})
            elif kind=='terminal':
                match['info']['participants'][7]['item0']=1036
            elif kind=='overflow':
                timeline['info']['frames'][2]['events'].extend(
                    {'type':'ITEM_PURCHASED','participantId':7,'timestamp':120000,'itemId':1036} for _ in range(6))
            elif kind=='wrong_patch':
                match['info']['gameVersion']='16.18.1'
            files={}
            for endpoint,payload in (('match',match),('timeline',timeline)):
                archive.write(endpoint,mid,payload)
                path=root/'data/raw_match_v5'/endpoint/f'{mid}.json.gz'
                files[endpoint]={'path':path.relative_to(root).as_posix(),'sha256':sha256(path),'bytes':path.stat().st_size}
            entries.append({'match_id':mid,'split':split,'patch':'16.17','queue_id':420,
                            'game_creation':match['info']['gameCreation'],'game_duration':600,'files':files})
        assignment={r['match_id']:r['split'] for r in entries}
        split_metadata={'test_status':'unavailable_original_holdout_not_materialized',
                        'reserved_test_ids_sha256':canonical_sha(['KR_RESERVED']),
                        'reserved_test_games':1,'validation_exposure':'synthetic development fixture'}
        write_json(metadata/'split_assignment.json',assignment)
        write_json(metadata/'split_metadata.json',split_metadata)
        plan={'patch':'16.17','queue_id':420,'matches':entries,'split_metadata':split_metadata,
              'split_assignment_sha256':sha256(metadata/'split_assignment.json'),
              'split_metadata_sha256':sha256(metadata/'split_metadata.json')}
        path=metadata/'source_plan.json'
        write_json(path,plan)
        with patch.object(builder,'ROOT',root):
            # Changed split bytes, a self-consistent but reassigned fold, and
            # another hash-valid raw file must all fail before publication.
            write_json(metadata/'split_assignment.json',{**assignment,'KR_TRAIN':'val'})
            refused(lambda:builder.build(path,root/'bad-split-hash'),'Changed fold file accepted')
            mismatched=deepcopy(plan)
            mismatched['split_assignment_sha256']=sha256(metadata/'split_assignment.json')
            write_json(path,mismatched)
            refused(lambda:builder.build(path,root/'bad-split-role'),'Fold role mismatch accepted')
            write_json(metadata/'split_assignment.json',assignment)
            write_json(path,plan)
            mismatched=deepcopy(plan)
            mismatched['matches'][0]['files']['match']=deepcopy(entries[1]['files']['match'])
            write_json(path,mismatched)
            refused(lambda:builder.build(path,root/'bad-archive-path'),'Hash-valid different archive accepted')
            write_json(path,plan)
            with patch.object(builder,'reconstruct_game',wraps=builder.reconstruct_game) as replay:
                builder.build(path,root/'corpus')
                # Footwear, Viego, opaque undo and wrong patch never replay.
                assert replay.call_count==4, replay.call_count
            target=root/'corpus'
            report=json.loads((target/'screening_report.json').read_text(encoding='utf-8'))
            assert set(report['accepted_match_ids'])=={'KR_TRAIN','KR_VAL'}
            assert set(report['accepted_match_ids'])|set(report['excluded_match_ids'])==set(cases)
            assert report['selected_sources']==8 and report['stored_games']==2 and report['excluded_games']==6
            assert report['stored_games_by_split']=={'train':1,'val':1}
            assert report['training_clearance'] is False
            for mid,reason in (('KR_FOOTWEAR','unobserved_magical_footwear_grant'),
                               ('KR_VIEGO','unmodeled_viego_inventory'),('KR_UNDO','undo_without_item_ids'),
                               ('KR_TERMINAL','unreconciled_terminal_inventory'),
                               ('KR_OVERFLOW','modeled_inventory_slot_overflow')):
                assert reason in report['excluded_match_ids'][mid], (mid,report['excluded_match_ids'][mid])
            assert report['excluded_match_ids']['KR_WRONGPATCH'][0].startswith('invalid_raw_source:')
            with closing(sqlite3.connect((target/'tracker.db').as_uri()+'?mode=ro&immutable=1',uri=True)) as conn:
                assert {r[0] for r in conn.execute('SELECT match_id FROM games')}=={'KR_TRAIN','KR_VAL'}
                assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0]==20
                assert conn.execute('SELECT COUNT(*) FROM game_events').fetchone()[0]==20
                assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            inventory=[json.loads(line) for line in (target/'source_inventory.jsonl').read_text(encoding='utf-8').splitlines()]
            assert {r['match_id'] for r in inventory}=={'KR_TRAIN','KR_VAL'}
            frozen_archive=RawMatchArchive(target/'raw')
            for record in inventory:
                for endpoint,binding in record['files'].items():
                    assert sha256(target/binding['path'])==binding['sha256']
                    assert canonical_sha(frozen_archive.read(endpoint,record['match_id']))==binding['payload_sha256']
            manifest=json.loads((target/'corpus_manifest.json').read_text(encoding='utf-8'))
            assert manifest['db_sha256']==sha256(target/'tracker.db')
            assert manifest['games']==2 and manifest['production_hold_cleared'] is False
            for relative,binding in manifest['files'].items():
                assert sha256(target/relative)==binding['sha256']
            refused(lambda:builder.build(path,target),'Existing immutable corpus was replaced')
            assert not (root/'data/tracker.db').exists(), 'Main database path was written'
            assert not list(target.rglob('*test*.jsonl')), 'A test payload was materialized'
    finally:
        builder.db.DB_PATH=original_db
        for path in root.rglob('*'):
            if path.is_file():
                path.chmod(0o666)

print('ok development corpus: whole-lobby exclusions, frozen source/fold checks, real replay, two-fold immutable database, full exclusion ledger, no test payloads')
