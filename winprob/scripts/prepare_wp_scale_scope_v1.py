"""Publish safe identities, then bind only new archive bytes without decoding."""
from collections import Counter
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.wp_scale_scope_v1 import plan_rows,checked_json,verify_scope,need
from winprob.iql_dataset_contract import current_reservations,file_hash

ADMISSION=ROOT/'artifacts/admission/w1-admission-20260929-v2/admission_manifest.json'
PILOT=ROOT/'data/runs/w1-pilot-20260929-v3/cache/source_manifest.json'
PROTOCOL=ROOT/'docs/wp_scale_protocol_v1.md'


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')


def plan(output):
    need(not output.exists(),'Refuse scope overwrite')
    manifest,asha=checked_json(ADMISSION)
    need(asha=='fd13d31892465d676d201bc069a1e099f852ddb8b24c940c80340601e9ab5881','Wrong source admission')
    pilot=json.loads(PILOT.read_text());forbidden=current_reservations(manifest)
    rows=plan_rows(manifest,pilot,forbidden)
    paths=[Path(__file__),PROTOCOL,ADMISSION,ADMISSION.with_suffix('.sha256'),PILOT,ROOT/'winprob/wp_scale_scope_v1.py',
           ROOT/'winprob/admission.py',ROOT/'winprob/iql_dataset_contract.py']
    payload={'schema':'wp-full-scale-source-scope-v1','state':'metadata_scope_frozen','games':rows,'requested_games':len(rows),
        'requested_roles':dict(Counter(row['role'] for row in rows)),
        'complement_roles':dict(Counter(row['role'] for row in rows if row['cohort']=='complement')),
        'admission_path':str(ADMISSION),'admission_sha256':asha,'pilot_source_manifest_path':str(PILOT),
        'pilot_source_manifest_sha256':file_hash(PILOT),'protocol_path':str(PROTOCOL),'protocol_sha256':file_hash(PROTOCOL),
        'source_data_root':manifest['source_data_root'],'bindings_sha256':{str(path.resolve()):file_hash(path) for path in paths},
        'initial_pilot_cap_left_unchanged':6000,'new_study_requested_cap':17195,
        'raw_source_bytes_read':False,'raw_payloads_deserialized':False,'outcomes_read':False,'training_authorized':False,
        'reserved_intersections':0,'confirmatory_games_still_closed':1911,
        'limits':['Identity/role scope only; raw availability and feature qualification have not yet been established.',
                  'Development complement means outside this project pilot, not globally untouched or confirmatory.']}
    output.mkdir(parents=True,exist_ok=False);write(output/'scope.json',payload)
    print(json.dumps({key:payload[key] for key in ('requested_games','requested_roles','complement_roles','reserved_intersections')}))


def bind_sources(scope_path,output):
    started=time.monotonic();need(not output.exists(),'Refuse archive-binding overwrite')
    scope,scope_sha=checked_json(scope_path);manifest,pilot,rows=verify_scope(scope)
    archive=(Path(manifest['source_data_root'])/'raw_match_v5').resolve(strict=True)
    old={row['match_id']:row for row in pilot['compressed_sources']}
    need(len(old)==6000,'Pilot compressed-source mapping differs')
    output.mkdir(parents=True,exist_ok=False);bindings=[];missing=[];bytes_read=0
    def guard():
        need(time.monotonic()-started<=900,'Archive binding900second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,'Binding RAM floor0.75GiB')
    with (output/'byte_access_ledger.jsonl').open('x',encoding='utf-8') as ledger:
        for count,row in enumerate(rows):
            if count%100==0:guard()
            mid=row['match_id']
            if row['cohort']=='pilot':bindings.append({**old[mid],'reused_pilot_binding_without_reopening':True});continue
            result={'match_id':mid,'reused_pilot_binding_without_reopening':False};absent=[]
            for endpoint in ('match','timeline'):
                path=archive/endpoint/(mid+'.json.gz')
                if not path.is_file():absent.append(endpoint);continue
                resolved=path.resolve(strict=True);need(resolved.is_relative_to(archive),'Archive path escaped allowed root')
                ledger.write(json.dumps({'match_id':mid,'role':row['role'],'endpoint':endpoint,'path':str(resolved),'operation':'sha256_compressed_bytes_only'})+'\n');ledger.flush()
                before=resolved.stat();digest=file_hash(resolved);after=resolved.stat()
                need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed while binding bytes')
                result[endpoint]={'path':str(resolved),'bytes':after.st_size,'sha256':digest};bytes_read+=after.st_size
            if absent:missing.append({'match_id':mid,'role':row['role'],'reason':'missing_archive','missing_endpoints':absent})
            else:bindings.append(result)
    verify_scope(scope);guard()
    result={'schema':'wp-full-scale-compressed-bindings-v1','state':'complete','scope_path':str(scope_path.resolve()),'scope_sha256':scope_sha,
        'compressed_sources':bindings,'missing_archives':missing,'requested_games':len(rows),'complete_source_games':len(bindings),
        'complement_compressed_bytes_hashed':bytes_read,'byte_access_ledger_sha256':file_hash(output/'byte_access_ledger.jsonl'),
        'raw_payloads_deserialized':False,'outcome_fields_indexed':False,'sealed_or_confirmation_source_reads':False,
        'script_sha256':file_hash(__file__),'elapsed_s':time.monotonic()-started,'training_authorized':False}
    write(output/'sources.json',result)
    print(json.dumps({key:result[key] for key in ('requested_games','complete_source_games','missing_archives','complement_compressed_bytes_hashed','elapsed_s')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['plan','bind'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--scope',type=Path)
    args=parser.parse_args()
    if args.stage=='plan':plan(args.output)
    else:
        need(args.scope is not None,'Binding requires frozen scope');bind_sources(args.scope,args.output)
