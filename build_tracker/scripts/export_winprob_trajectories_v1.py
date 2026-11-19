"""Source-owned, indexed ten-game retained-basket trajectory smoke exporter.

Only new artifacts are written. Never streams a mixed exported training file,
never imports winprob code, and never loads or scores a policy. Admission is
metadata-only and completes before any game query or raw payload read.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
GENERATION = ROOT/'data/dataset_generations/16.18-reconstructible-development-v4'
ADMISSION = ROOT.parent/'winprob/artifacts/admission/w1-admission-20260929-v2/admission_manifest.json'
PACKAGE = ROOT/'data/packages/mixture-v4-m2-v1'
SCHEMA = 'winprob-retained-basket-trajectories-v1'
MATCH_ID = re.compile(r'[A-Za-z0-9_\-]+\Z')
SPLIT_NAMES = {'split_assignment.json','preserved_split_assignment.json','split_plan.json',
               'original_split_assignment.json','original_split_plan.json'}
STATE_FIELDS = ('ts','champion','champion_id','team_id','gold','gold_left','gold_est','gold_est_version',
                'total_gold','level','cs','kills','deaths','assists','inventory','opening_sale_gold',
                'unmodeled_regular_slots','bot_quest_complete','slot_state_version','footwear_state_version',
                'magical_footwear_selected','footwear_granted','can_complete','n_completable','cheapest_complete',
                'gold_after_complete','n_inventory','ally_obj','enemy_obj','keystone_id','sub_style','summ1','summ2','others')
OTHER_FIELDS = ('champion','champion_id','team_id','gold','level','items','bot_quest_complete',
                'slot_state_version','footwear_state_version','magical_footwear_selected','footwear_granted')
DECISION_FIELDS = ('event_index','participant_id','ts')
OBJECTIVE_FIELDS = {'towers','dragons','barons','heralds','voidgrubs'}
SAVE_SENTINEL = 999999


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_bound(path):
    path=Path(path).resolve(strict=True)
    digest=file_hash(path)
    if path.with_suffix('.sha256').read_text(encoding='ascii').split()!=[digest,path.name]:
        raise ValueError('Sidecar mismatch: '+str(path))
    return load_json(path),digest


def id_hash(ids):
    return hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest()


def assignment(path):
    data=load_json(path)
    if isinstance(data,dict) and isinstance(data.get('assignment'),dict):
        data=data['assignment']
    if not isinstance(data,dict) or any(not isinstance(mid,str) or not MATCH_ID.fullmatch(mid)
        or fold not in ('train','val','test','post_cutoff','excluded') for mid,fold in data.items()):
        raise ValueError('Unknown split metadata layout: '+str(path))
    return data


def validate_admission(path, source_root=ROOT):
    """Independent artifact consumer; imports no consumer-project Python code."""
    manifest,digest=load_bound(path)
    if manifest.get('schema')!='winprob-admission-v1' or manifest.get('state')!='temporal_manifest_ready':
        raise ValueError('Expected frozen winprob temporal admission metadata')
    data_root=(Path(source_root).resolve()/'data').resolve(strict=True)
    if Path(manifest['source_data_root']).resolve(strict=True)!=data_root:
        raise ValueError('Admission does not name this producer data root')
    discovered={}
    def traversal_error(error):
        raise error
    for folder,dirs,files in os.walk(data_root,followlinks=False,onerror=traversal_error):
        here=Path(folder)
        if here==data_root:
            dirs[:]=[name for name in dirs if name not in ('raw_match_v5','cache')]
        for name in dirs:
            p=here/name
            if p.is_symlink() or not p.resolve().is_relative_to(data_root):
                raise ValueError('Linked directory in split metadata scan')
        for name in files:
            if name in SPLIT_NAMES:
                p=here/name
                if not p.resolve(strict=True).is_relative_to(data_root):
                    raise ValueError('Split file escaped source root')
                discovered[p.relative_to(data_root).as_posix()]=file_hash(p)
    expected={row['path']:row['sha256'] for row in manifest['split_metadata_sources']}
    if not discovered or discovered!=expected:
        raise ValueError('Current split metadata differs from admission snapshot')
    tests,validation=set(),set()
    for relative in discovered:
        for mid,fold in assignment(data_root/relative).items():
            if fold=='test': tests.add(mid)
            elif fold=='val': validation.add(mid)
    bound_files={str(Path(path).resolve()):digest}
    bound_files.update({str(data_root/relative):value for relative,value in discovered.items()})
    for binding in manifest.get('external_reservation_sources',[]):
        artifact,observed=load_bound(binding['path'])
        if observed!=binding['sha256'] or artifact.get('schema')!='winprob-external-reservations-v1':
            raise ValueError('External reservation artifact changed')
        source=Path(artifact['source_metadata_path']).resolve(strict=True)
        if (file_hash(source)!=artifact['source_metadata_sha256']
                or artifact['source_metadata_sha256']!=binding['source_metadata_sha256']
                or str(source)!=str(Path(binding['source_metadata_path']).resolve(strict=True))):
            raise ValueError('External reservation source metadata changed')
        ids=artifact.get('reserved_test_ids')
        if not isinstance(ids,list) or any(not isinstance(mid,str) or not MATCH_ID.fullmatch(mid) for mid in ids):
            raise ValueError('Invalid external reservation identities')
        tests.update(ids)
        bound_files[str(Path(binding['path']).resolve())]=observed
        bound_files[str(source)]=artifact['source_metadata_sha256']
    by_id={row['match_id']:row for row in manifest['games']}
    if len(by_id)!=len(manifest['games']):
        raise ValueError('Duplicate admitted identity')
    confirm={mid for mid,row in by_id.items() if row['role']=='confirmatory_reserved'}
    prior=manifest.get('prior_manifest');visited=set()
    while prior:
        prior_path=str(Path(prior['path']).resolve(strict=True))
        if prior_path in visited: raise ValueError('Cyclic prior admission')
        visited.add(prior_path)
        old,old_hash=load_bound(prior_path)
        if old_hash!=prior['sha256']: raise ValueError('Prior admission changed')
        old_roles={row['match_id']:row['role'] for row in old['games']}
        if any(old_roles.get(mid)!=row['role'] for mid,row in by_id.items()):
            raise ValueError('Previously frozen role was changed')
        confirm.update(mid for mid,role in old_roles.items() if role=='confirmatory_reserved')
        bound_files[prior_path]=old_hash
        prior=old.get('prior_manifest')
    if set(by_id)&tests:
        raise ValueError('Admission contains a globally sealed test')
    return manifest,tests,validation,confirm,bound_files


def select_ids(manifest, source_assignment, accepted, tests, validation, confirm, limit):
    if type(limit) is not int or not 1<=limit<=10:
        raise ValueError('First smoke is bounded to one through ten games')
    rows={row['match_id']:row for row in manifest['games']}
    candidates={mid for mid,fold in source_assignment.items() if fold=='train' and mid in accepted
                and mid in rows and rows[mid]['role']=='train' and rows[mid].get('historical_validation') is False}
    candidates-=set(tests)|set(validation)|set(confirm)
    if not candidates:
        raise ValueError('No permitted source TRAIN intersect winprob-fit games')
    selected=sorted(candidates,key=lambda mid:hashlib.sha256(('trajectory-smoke-v1:'+mid).encode()).digest())[:limit]
    return selected,len(candidates)


class JSONStream:
    """Bounded JSON value parser, used only for metadata object entries."""
    def __init__(self,handle):
        self.handle=handle;self.buffer='';self.position=0;self.eof=False;self.decoder=json.JSONDecoder()
    def more(self):
        self.buffer=self.buffer[self.position:];self.position=0
        chunk=self.handle.read(65536)
        self.eof=not chunk
        self.buffer+=chunk
    def peek(self):
        while True:
            while self.position<len(self.buffer) and self.buffer[self.position].isspace(): self.position+=1
            if self.position<len(self.buffer): return self.buffer[self.position]
            if self.eof: return ''
            self.more()
    def expect(self,char):
        if self.peek()!=char: raise ValueError('Invalid metadata JSON delimiter')
        self.position+=1
    def value(self):
        self.peek()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buffer,self.position)
                self.position=end
                return value
            except json.JSONDecodeError:
                if self.eof: raise
                self.more()


def selected_object_entries(path, object_key, selected):
    """Scan identity metadata without retaining the entire decision-mask object."""
    found={};seen=set();top_found=False
    with Path(path).open('r',encoding='utf-8-sig') as handle:
        stream=JSONStream(handle);stream.expect('{')
        while stream.peek()!='}':
            key=stream.value();stream.expect(':')
            if key==object_key:
                if top_found: raise ValueError('Duplicate metadata object key')
                top_found=True;stream.expect('{')
                while stream.peek()!='}':
                    mid=stream.value();stream.expect(':');entry=stream.value()
                    if mid in seen: raise ValueError('Duplicate decision-mask identity')
                    seen.add(mid)
                    if mid in selected: found[mid]=entry
                    if stream.peek()==',': stream.expect(',')
                    elif stream.peek()!='}': raise ValueError('Invalid mask metadata delimiter')
                stream.expect('}')
            else:
                stream.value()
            if stream.peek()==',': stream.expect(',')
            elif stream.peek()!='}': raise ValueError('Invalid outer metadata delimiter')
        stream.expect('}')
        if stream.peek(): raise ValueError('Trailing metadata JSON bytes')
    if not top_found or set(found)!=set(selected):
        raise ValueError('Requested games missing from decision mask')
    return found


def source_metadata(generation, manifest, tests, validation, confirm, limit):
    generation=Path(generation).resolve(strict=True)
    if generation.name!='16.18-reconstructible-development-v4':
        raise ValueError('This initial producer is fixed to reviewed 16.18 v4')
    package=load_json(PACKAGE/'manifest.json');identity=load_json(PACKAGE/'package_id.json')
    if value_hash(package)!=identity['manifest_sha256']:
        raise ValueError('Reference package metadata identity changed')
    generation_path=generation/'generation_manifest.json'
    if file_hash(generation_path)!=package['generation']['generation_manifest_sha256']:
        raise ValueError('Source generation differs from frozen M2 source generation')
    g=load_json(generation_path)
    names=('qualification.json','corpus_manifest.json','benchmark_admission.json','source_plan.json',
           'semantic_audit.json','decision_admission.json','split_assignment.json','original_static_review.json')
    bindings={str(generation_path):file_hash(generation_path),str(PACKAGE/'manifest.json'):file_hash(PACKAGE/'manifest.json'),
              str(PACKAGE/'package_id.json'):file_hash(PACKAGE/'package_id.json')}
    for name in names:
        p=generation/name;expected=g['files'].get(name)
        if not expected or p.stat().st_size!=expected['bytes'] or file_hash(p)!=expected['sha256']:
            raise ValueError('Frozen generation evidence changed: '+name)
        bindings[str(p)]=expected['sha256']
    q=load_json(generation/'qualification.json');corpus=load_json(generation/'corpus_manifest.json')
    audit=load_json(generation/'semantic_audit.json');source=load_json(generation/'source_plan.json')
    if (q.get('status')!='qualified_subset' or q.get('purpose')!='isolated_experiment_only'
        or q.get('benchmark_purpose')!='patch_development_1618' or q.get('patch')!='16.18'
        or q.get('promotion_allowed') is not False or not audit.get('passed') or audit.get('hard_failures')
        or q['corpus_manifest_sha256']!=bindings[str(generation/'corpus_manifest.json')]
        or q['semantic_audit_sha256']!=bindings[str(generation/'semantic_audit.json')]
        or q['decision_admission_sha256']!=bindings[str(generation/'decision_admission.json')]
        or q['db_sha256']!=corpus['db_sha256'] or audit['db_sha256']!=corpus['db_sha256']
        or source.get('test_payload_reads') is not False or corpus.get('test_payload_reads') is not False):
        raise ValueError('Unqualified or inconsistent source evidence')
    selected,n_candidates=select_ids(manifest,assignment(generation/'split_assignment.json'),set(q['accepted_match_ids']),
                                     tests,validation,confirm,limit)
    plans={row['match_id']:row for row in source['matches']}
    if not set(selected)<=plans.keys(): raise ValueError('Approved IDs absent from source plan')
    masks=selected_object_entries(generation/'decision_admission.json','games',set(selected))
    path=Path(q['corpus']).resolve(strict=True)
    if path!=ROOT/'data/corpora/16.18-reconstructible-v4': raise ValueError('Unexpected qualified source path')
    if file_hash(path/'corpus_manifest.json')!=q['corpus_manifest_sha256']:
        raise ValueError('Corpus metadata differs from qualified generation')
    for name in ('_example','_label','_item_ids','_others','_unmodeled_regular_slots'):
        def node_at(p):
            return next(node for node in ast.parse(p.read_text(encoding='utf-8-sig')).body
                        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name)
        frozen=generation/'export_code/baseline.py'
        if ast.dump(node_at(ROOT/'scripts/baseline.py'),include_attributes=False)!=ast.dump(node_at(frozen),include_attributes=False):
            raise ValueError('Source row projection changed since frozen export: '+name)
    return {'generation':generation,'corpus_path':path,'corpus':corpus,'qualification':q,'selected':selected,
            'candidates':n_candidates,'plans':{mid:plans[mid] for mid in selected},'masks':masks,'bindings':bindings}


class ApprovedGameReader:
    def __init__(self,connection,allowed,denied):
        self.connection=connection;self.allowed=frozenset(allowed);self.denied=frozenset(denied)
        if not self.allowed or self.allowed&self.denied: raise ValueError('Invalid SQL admission set')
        self.connection.row_factory=sqlite3.Row
    def game(self,mid):
        if mid not in self.allowed or mid in self.denied: raise PermissionError('Game rejected before SQL')
        row=self.connection.execute('SELECT match_id,game_creation,game_duration,game_version,patch,queue_id,win_team,participants_json,source FROM games WHERE match_id=?',(mid,)).fetchone()
        if row is None: raise ValueError('Approved game missing from qualified corpus')
        records=self.connection.execute('SELECT event_index,ts,ts_end,type,payload_json FROM game_events WHERE match_id=? ORDER BY event_index',(mid,))
        events=[]
        for record in records:
            event=json.loads(record['payload_json'])
            if any(event.get(key)!=record[key] for key in ('event_index','ts','ts_end','type')):
                raise ValueError('Event identity columns differ from payload')
            events.append(event)
        return dict(row),events


def decision_key(event):
    values=tuple(event.get(key) for key in DECISION_FIELDS)
    if any(type(value) is not int for value in values) or values[0]<0 or not 1<=values[1]<=10 or values[2]<0:
        raise ValueError('Invalid decision identity')
    return values


def counts_of(row):
    if 'label_counts' in row:
        counts=dict(row['label_counts'])
    else:
        counts={str(item):count for item,count in Counter(row.get('label_ids',[])).items()}
    if (not counts or any(not str(key).isdecimal() or str(int(key))!=str(key) or int(key)<=0
                         or type(value) is not int or value<=0 for key,value in counts.items())):
        raise ValueError('Invalid retained purchase multiset')
    if str(SAVE_SENTINEL) in counts and counts!={str(SAVE_SENTINEL):1}:
        raise ValueError('No-buy sentinel cannot share an action')
    return {str(key):counts[key] for key in sorted(counts,key=int)}


def project_state(row):
    missing=set(STATE_FIELDS)-row.keys()
    if missing: raise ValueError('Uncertain prestate fields: '+','.join(sorted(missing)))
    state={key:row[key] for key in STATE_FIELDS}
    for key in ('ts','champion_id','team_id','gold','gold_left','gold_est','total_gold','level','cs','kills','deaths','assists',
                'opening_sale_gold','unmodeled_regular_slots','n_completable','cheapest_complete','gold_after_complete',
                'n_inventory','keystone_id','sub_style','summ1','summ2'):
        value=state[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Unknown numeric prestate field: '+key)
    if state['team_id'] not in (100,200) or state['gold_est_version']!='prequential-v3':
        raise ValueError('Invalid team or unreviewed wallet estimator')
    for key in ('bot_quest_complete','magical_footwear_selected','footwear_granted'):
        if type(state[key]) is not bool: raise ValueError('Unknown prestate flag: '+key)
    if state['footwear_granted'] and not state['magical_footwear_selected']:
        raise ValueError('Inconsistent footwear prestate')
    if not isinstance(state['inventory'],list) or any(type(item) is not int or item<=0 for item in state['inventory']):
        raise ValueError('Invalid prestate inventory')
    if not isinstance(state['others'],list) or len(state['others'])!=9: raise ValueError('Incomplete prestate board')
    for key in ('ally_obj','enemy_obj'):
        objective=state[key]
        if (not isinstance(objective,dict) or set(objective)-OBJECTIVE_FIELDS
                or any(type(value) is not int or value<0 for value in objective.values())):
            raise ValueError('Unknown objective state fields')
    clean=[]
    for other in state['others']:
        if set(OTHER_FIELDS)-other.keys(): raise ValueError('Unknown other-player prestate fields')
        player={key:other[key] for key in OTHER_FIELDS}
        for key in ('champion_id','team_id','gold','level'):
            value=player[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Unknown other-player numeric state field')
        if (player['team_id'] not in (100,200) or not isinstance(player['items'],list)
                or any(type(item) is not int or item<=0 for item in player['items'])):
            raise ValueError('Invalid other-player identity/inventory')
        for key in ('bot_quest_complete','magical_footwear_selected','footwear_granted'):
            if type(player[key]) is not bool: raise ValueError('Unknown other-player prestate flag')
        clean.append(player)
    state['others']=clean
    canonical(state)  # Refuse NaN/infinity anywhere nested.
    return state


def reference_compatibility(row,prestate):
    """Legacy role fields belong to frozen-reference replay, never IQL state."""
    if not isinstance(row.get('role'),str) or any(not isinstance(player.get('role'),str) for player in row['others']):
        raise ValueError('Missing legacy inferred role for reference compatibility')
    projection={**prestate,'role':row['role'],
                'others':[{**clean,'role':raw['role']} for clean,raw in zip(prestate['others'],row['others'],strict=True)]}
    return {'schema':'m2-source-context-compatibility-v1','not_iql_prestate':True,
            'role_source':'Riot final match teamPosition; not certified pre-action',
            'source_projection':projection,
            'use':'Stateless reconstruction of the frozen M2 reference input only; labels and outcomes omitted'}


def terminal_from_timeline(timeline):
    terminals=[event for frame in timeline.get('info',{}).get('frames',[]) for event in frame.get('events',[])
               if event.get('type')=='GAME_END']
    if len(terminals)!=1 or type(terminals[0].get('timestamp')) is not int or terminals[0]['timestamp']<=0:
        raise ValueError('Require one actual positive GAME_END timestamp')
    return terminals[0]['timestamp']


def verify_replay(game,events,replayed_game,replayed_events):
    for key in ('match_id','game_creation','game_duration','game_version','patch','queue_id','win_team','source'):
        if game.get(key)!=replayed_game.get(key): raise ValueError('Qualified game differs from certified raw replay: '+key)
    if load_participants(game)!=load_participants(replayed_game):
        raise ValueError('Qualified participants differ from certified raw replay')
    replayed=[event for event in replayed_events if event.get('type')=='shop']
    if canonical(events)!=canonical(replayed):
        raise ValueError('Indexed qualified events differ from certified raw replay')


def load_participants(game):
    value=game['participants_json']
    return json.loads(value) if isinstance(value,str) else value


def trajectories_for_game(game,events,mask,projector,terminal_ms,quality):
    mid=game['match_id'];participants=load_participants(game)
    if len(participants)!=10: raise ValueError('Qualified game lacks ten participants')
    players={p['participant_id']:p for p in participants}
    if set(players)!=set(range(1,11)) or game['win_team'] not in (100,200):
        raise ValueError('Invalid participant mapping or terminal winner')
    for p in participants:
        if p.get('team_id') not in (100,200) or type(p.get('win')) is not bool or p['win']!=(p['team_id']==game['win_team']):
            raise ValueError('Participant result disagrees with terminal winner')
    admitted={decision_key(row) for row in mask['admitted_decision_ids']}
    excluded={decision_key(row):row.get('reason','boundary_exclusion') for row in mask['boundary_exclusions']}
    if admitted&excluded.keys() or len(admitted)!=mask['admitted_rows'] or len(excluded)!=mask['boundary_excluded_rows']:
        raise ValueError('Invalid audited decision mask')
    seen=set();nodes=defaultdict(list)
    for event in events:
        key=decision_key(event);pid=event['participant_id']
        if event.get('type')!='shop' or event.get('training_eligible') is not True or event.get('source_quality_issues'):
            raise ValueError('Unqualified source event')
        if event.get('puuid')!=players[pid]['puuid'] or event.get('team_id')!=players[pid]['team_id']:
            raise ValueError('Event participant binding mismatch')
        bound=dict(event);bound['match_id']=mid;bound['perks']=players[pid]
        row=projector(bound)
        base={'node_id':f'{mid}:{pid}:{event["event_index"]}','event_index':event['event_index'],
              'timestamp_ms':event['ts'],'visit_end_ms':event['ts_end'],'reward':0,'done':False,
              'source_event_sha256':value_hash(event),'next_ref':None}
        if row is None:
            if key in admitted or key in excluded: raise ValueError('Audited modeled decision disappeared during projection')
            node={**base,'kind':'gap','reason':'source_projection_has_no_retained_target','prestate':None,'action':None}
        else:
            if key in seen or key not in admitted|excluded.keys():
                raise ValueError('Projected decision outside exact audited mask')
            seen.add(key)
            if key in excluded:
                node={**base,'kind':'gap','reason':excluded[key],'prestate':None,'action':None}
            else:
                counts=counts_of(row)
                if row.get('team_id')!=players[pid]['team_id'] or row.get('ts')!=event['ts']:
                    raise ValueError('Projection changed the decision identity')
                prestate=project_state(row)
                node={**base,'kind':'decision','prestate':prestate,
                      'reference_compatibility':reference_compatibility(row,prestate),
                      'action':{'encoding':'item-counts-v1','label_counts':counts,
                                'kind':'observed_no_buy_death' if counts=={str(SAVE_SENTINEL):1} else 'retained_net_purchase',
                                'wards_preserved':True}}
        nodes[pid].append(node)
    if seen!=admitted|excluded.keys(): raise ValueError('Missing audited decisions in indexed game')
    result=[]
    for pid,player in sorted(players.items()):
        sequence=sorted(nodes[pid],key=lambda node:(node['timestamp_ms'],node['event_index']))
        reasons=[]
        if not sequence: reasons.append('no_observed_shop_decisions')
        if any(node['kind']=='gap' for node in sequence): reasons.append('unmodeled_or_excluded_decision_gap')
        if any(a['timestamp_ms']>=b['timestamp_ms'] or a['visit_end_ms']>b['timestamp_ms'] for a,b in zip(sequence,sequence[1:])):
            reasons.append('overlapping_or_duplicate_decision_times')
        if any(type(node['visit_end_ms']) is not int or node['visit_end_ms']<node['timestamp_ms']
               or node['visit_end_ms']>terminal_ms for node in sequence):
            reasons.append('invalid_or_postterminal_visit_interval')
        terminal={'node_id':f'{mid}:{pid}:terminal','kind':'terminal','timestamp_ms':terminal_ms,
                  'prestate':None,'action':None,'reward':int(player['win']),'done':True,'next_ref':None}
        for current,nxt in zip(sequence,sequence[1:]+[terminal]): current['next_ref']=nxt['node_id']
        result.append({'schema':SCHEMA,'trajectory_id':f'{mid}:{pid}','match_id':mid,'participant_id':pid,
                       'team_id':player['team_id'],'patch':game['patch'],'source_role':'v4_train','consumer_role':'fit',
                       'game_creation_ms':game['game_creation'],'whole_player_game_eligible':not reasons,
                       'ineligibility_reasons':reasons,'decisions':sequence,'terminal':terminal,
                       'source_quality':quality,
                       'action_scope':'retained net basket including wards; observed no-buy-death is explicit; not primitive transaction control',
                       'iql_readiness':'complete_observed_modeled_decision_chain' if not reasons else 'excluded_whole_player_game',
                       'causal_action_identification_established':False})
    return result


def read_raw_pair(mid,plan,allowed,denied):
    if mid not in allowed or mid in denied: raise PermissionError('Game rejected before raw access')
    payloads={};bindings={}
    for endpoint in ('match','timeline'):
        binding=plan['files'][endpoint];path=Path(binding['path']).resolve(strict=True)
        expected=(ROOT/'data/raw_match_v5'/endpoint/(mid+'.json.gz')).resolve(strict=True)
        if path!=expected: raise ValueError('Raw source path differs from bound archive')
        compressed=path.read_bytes();observed=hashlib.sha256(compressed).hexdigest()
        if len(compressed)!=binding['bytes'] or observed!=binding['sha256']:
            raise ValueError('Approved raw source changed: '+mid+'/'+endpoint)
        envelope=json.loads(gzip.decompress(compressed).decode('utf-8'))
        if (envelope.get('schema')!=1 or envelope.get('endpoint')!=endpoint or envelope.get('match_id')!=mid
                or envelope.get('payload',{}).get('metadata',{}).get('matchId')!=mid):
            raise ValueError('Invalid raw envelope identity')
        payloads[endpoint]=envelope['payload'];bindings[endpoint]={**binding,'path':str(path)}
    return payloads,bindings


def export(args):
    started=time.monotonic();output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Immutable trajectory export already exists')
    manifest,tests,validation,confirm,bound=validate_admission(args.admission)
    source=source_metadata(args.generation,manifest,tests,validation,confirm,args.limit)
    selected=source['selected'];denied=tests|validation|confirm
    bound.update(source['bindings'])
    # Runtime activation is source-owned and must precede baseline consumers.
    from footwear_runtime import activate_for_manifest
    active=activate_for_manifest(source['corpus'],corpus_path=source['corpus_path'])
    if active!=source['corpus']['footwear_runtime']: raise ValueError('Reviewed source runtime declaration changed')
    import baseline
    from app.ddragon import dragon_for_patch
    from app.reconstruct import reconstruct_game
    from process_reconstructible_1618 import decision_mask
    dragon=dragon_for_patch('16.18')
    if dragon.signature!=source['corpus']['static_data_sha256']: raise ValueError('Source static data changed')
    code_names=('scripts/export_winprob_trajectories_v1.py','scripts/baseline.py','scripts/footwear_runtime.py',
                'scripts/reconstructible_1618_runtime.py','scripts/process_reconstructible_1618.py',
                'scripts/footwear_economics.py','scripts/footwear_timer.py','scripts/footwear_timer_inventory.py',
                'app/footwear_reconstruction.py','app/ddragon.py','app/shop_econ.py','app/purchase_multiset.py')
    code={name:file_hash(ROOT/name) for name in code_names}
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n',encoding='utf-8')
    status('ready_for_indexed_reads',selected_game_ids=selected)
    db_path=source['corpus_path']/'tracker.db';db_stat=db_path.stat()
    all_records=[];game_bindings=[]
    try:
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=2)) as conn:
            conn.execute('PRAGMA query_only=ON');conn.execute('PRAGMA cache_size=-2048')
            reader=ApprovedGameReader(conn,selected,denied)
            for index,mid in enumerate(selected,1):
                if time.monotonic()-started>150: raise RuntimeError('Ten-game smoke deadline exceeded')
                game,events=reader.game(mid)
                if game['patch']!='16.18' or game['queue_id']!=420: raise ValueError('Wrong source patch/queue')
                raw,raw_bindings=read_raw_pair(mid,source['plans'][mid],set(selected),denied)
                debug={};replayed_game,replayed_events=reconstruct_game(raw['match'],raw['timeline'],dragon,debug)
                verify_replay(game,events,replayed_game,replayed_events)
                fresh_mask=decision_mask(replayed_events,debug,dragon)
                frozen=source['masks'][mid]
                if any(fresh_mask[key]!=frozen[key] for key in ('admitted_decision_ids','boundary_exclusions','admitted_rows','boundary_excluded_rows')):
                    raise ValueError('Per-game decision mask does not reproduce')
                terminal_ms=terminal_from_timeline(raw['timeline'])
                quality={'qualified_corpus_manifest_sha256':source['qualification']['corpus_manifest_sha256'],
                         'qualification_sha256':source['bindings'][str(source['generation']/'qualification.json')],
                         'audited_decision_mask_sha256':source['qualification']['decision_admission_sha256'],
                         'selected_mask_sha256':value_hash(frozen),'raw_sources':raw_bindings,
                         'indexed_rows_sha256':value_hash({'game':game,'events':events}),
                         'raw_replay_matches_indexed_rows':True,'admitted_mask_reproduced':True}
                projected=trajectories_for_game(game,events,frozen,lambda event:baseline._example(event,dragon),terminal_ms,quality)
                all_records.extend(projected);game_bindings.append({'match_id':mid,**quality})
                status('exporting',processed_games=index,trajectories=len(all_records))
        if db_path.stat().st_mtime_ns!=db_stat.st_mtime_ns or db_path.stat().st_size!=db_stat.st_size:
            raise ValueError('Qualified corpus changed during bounded read')
        validate_admission(args.admission)
        if any(file_hash(path)!=expected for path,expected in bound.items()): raise ValueError('Input metadata changed during export')
        if any(file_hash(ROOT/name)!=expected for name,expected in code.items()): raise ValueError('Producer implementation changed during export')
        rows_path=output/'trajectories.jsonl'
        with rows_path.open('x',encoding='utf-8',newline='\n') as handle:
            for record in all_records: handle.write(canonical(record).decode('utf-8')+'\n')
        rows_hash=file_hash(rows_path)
        rows_path.with_suffix('.sha256').write_text(rows_hash+'  trajectories.jsonl\n',encoding='ascii')
        meta={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
              'producer':'build_tracker','version':output.name,'selection':'hash-smoke of source16.18v4TRAIN intersect frozen winprob fit',
              'candidate_games':source['candidates'],'selected_game_ids':selected,'selected_ids_sha256':id_hash(selected),
              'global_tests_denied':len(tests),'historical_validation_denied':len(validation),'confirmation_denied':len(confirm),
              'rows_file':'trajectories.jsonl','rows_sha256':rows_hash,'trajectory_count':len(all_records),
              'eligible_player_games':sum(row['whole_player_game_eligible'] for row in all_records),
              'decision_nodes':sum(sum(node['kind']=='decision' for node in row['decisions']) for row in all_records),
              'gap_nodes':sum(sum(node['kind']=='gap' for node in row['decisions']) for row in all_records),
              'ineligibility_counts':dict(Counter(reason for row in all_records for reason in row['ineligibility_reasons'])),
              'input_metadata_sha256':bound,'producer_code_sha256':code,'runtime':active,'game_bindings':game_bindings,
              'database':{'path':str(db_path),'qualified_full_file_sha256_reference':source['corpus']['db_sha256'],
                          'full_file_digest_recomputed':False,'validation':'selected indexed rows exactly reproduce hash-bound raw sources',
                          'indexed_access_only':True,'bytes':db_stat.st_size},
              'prestate_whitelist':list(STATE_FIELDS),'other_player_whitelist':list(OTHER_FIELDS),
              'postmatch_inferred_fields_excluded_from_prestate':['role','others[].role'],
              'reference_compatibility_contract':'Legacy role-bearing source_projection is preserved separately for frozen M2 input replay, never IQL features.',
              'privileged_prestate_fields':['total_gold','others[].gold'],
              'pov_actor_contract':'Do not consume privileged total-gold fields; M2 uses its existing POV gold projection.',
              'timestamp_contract':'Source visit timestamps are preserved. Qualified decision inventory/wallet include only opening sales before the first purchase; no primitive-action timestamp is inferred.',
              'prestate_cutoff_semantics':'Buy rows: after opening sales, before first purchase; source ts is visit start. No-buy-death rows retain the audited decision-start snapshot. This is not certification of an exact primitive-action cutoff.',
              'reward_contract':'zero on every decision/gap node; focal terminal Y exactly once at actual GAME_END; terminal potential zero',
              'action_contract':'full retained purchase multiset INCLUDING wards/copies; SAVE sentinel only for observed no-buy-death',
              'limits':['No primitive transaction policy or unobserved manual no-buy coverage is claimed.',
                        'Any modeled-decision gap makes the entire player-game ineligible for initial IQL.',
                        'No M2 lift, propensity, policy scoring, neural fitting or causal identification performed.'],
              'test_payload_reads':False,'confirmation_payload_reads':False,'mixed_export_reads':False,'elapsed_s':time.monotonic()-started}
        meta_path=output/'manifest.json';meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
        meta_path.with_suffix('.sha256').write_text(file_hash(meta_path)+'  manifest.json\n',encoding='ascii')
        status('complete',eligible_player_games=meta['eligible_player_games'],decision_nodes=meta['decision_nodes'],gap_nodes=meta['gap_nodes'])
        return meta
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,default=GENERATION)
    parser.add_argument('--admission',type=Path,default=ADMISSION)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.limit<=10: parser.error('This first smoke permits at most ten games')
    result=export(args)
    print(json.dumps({key:result[key] for key in ('state','selected_game_ids','eligible_player_games','decision_nodes','gap_nodes','rows_sha256','elapsed_s')},indent=2))


if __name__=='__main__':
    main()
