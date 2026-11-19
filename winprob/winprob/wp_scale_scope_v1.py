"""Explicit larger source scope; original pilot extraction cap remains frozen."""
from collections import Counter
from pathlib import Path
import json
from . import admission
from .iql_dataset_contract import current_reservations,file_hash,reject_quarantined

ROLE_MAP={'train':'fit','calibration':'calibration','development':'development'}
EXPECTED={'train':11468,'calibration':2866,'development':2861}


def need(value,message):
    if not value:raise ValueError(message)


def plan_rows(manifest,pilot,forbidden):
    entries=manifest['games'];lookup={row['match_id']:row for row in entries}
    need(len(lookup)==len(entries),'Duplicate admitted identity')
    eligible=[row for row in entries if row['role'] in ROLE_MAP]
    need(Counter(row['role'] for row in eligible)==EXPECTED,'Frozen full-pool role counts changed')
    for row in eligible:
        need(row.get('historical_validation') is False and row['queue_id']==420 and admission.MATCH_ID.fullmatch(row['match_id'])
             and row['match_id'] not in forbidden,'Unsafe full-pool source identity')
    old=pilot['games'];need(len(old)==6000 and [r['game_index'] for r in old]==list(range(6000)),'Pilot game order changed')
    old_ids={row['match_id'] for row in old};need(len(old_ids)==6000,'Duplicate pilot identity')
    for row in old:
        entry=lookup.get(row['match_id'])
        need(entry is not None and entry['role'] in ROLE_MAP and row['source_role']==entry['role'] and row['role']==ROLE_MAP[entry['role']]
             and row['patch']==entry['patch'] and row['game_creation_ms']==entry['game_creation_ms'],'Pilot is not an exact admitted subset')
    ordered=[lookup[row['match_id']] for row in old]+sorted((row for row in eligible if row['match_id'] not in old_ids),
                                                         key=lambda row:(row['game_creation_ms'],row['match_id']))
    rows=[]
    for index,row in enumerate(ordered):
        rows.append({'scope_index':index,'match_id':row['match_id'],'source_role':row['role'],'role':ROLE_MAP[row['role']],
            'patch':row['patch'],'game_creation_ms':row['game_creation_ms'],'queue_id':420,'historical_validation':False,
            'cohort':'pilot' if index<6000 else 'complement','pilot_game_index':index if index<6000 else None})
    need(len(rows)==17195 and len({r['match_id'] for r in rows})==17195,'Full source scope differs')
    return rows


def checked_json(path):
    path=Path(path).resolve(strict=True);reject_quarantined(path);digest=file_hash(path)
    tokens=path.with_suffix('.sha256').read_text().split()
    need(tokens==[digest,path.name],'Metadata sidecar mismatch')
    return json.loads(path.read_text(encoding='utf-8')),digest


def verify_scope(scope):
    manifest,digest=checked_json(scope['admission_path'])
    need(digest==scope['admission_sha256'],'Parent admission changed')
    forbidden=current_reservations(manifest)
    pilot_path=Path(scope['pilot_source_manifest_path']).resolve(strict=True)
    need(file_hash(pilot_path)==scope['pilot_source_manifest_sha256'],'Pilot membership changed')
    pilot=json.loads(pilot_path.read_text());rows=plan_rows(manifest,pilot,forbidden)
    need(rows==scope['games'],'Scope identities/roles/order changed')
    for path,want in scope['bindings_sha256'].items():
        reject_quarantined(path);need(file_hash(path)==want,'Scope binding changed')
    return manifest,pilot,rows
