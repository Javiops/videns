"""Closed-visit history controls over the immutable own-project FIT cache.

Selection is published before current action arrays or source trajectories
are opened. Source JSON necessarily deserializes terminal fields; this module
never indexes those fields, opens reward arrays, or imports source code.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import admission
from .full_action_catalogue import load_catalogue,counts_key
from .iql_dataset_contract import (bound_json,current_reservations,file_hash,json_file,
                                  reject_quarantined,validate_sources)
from .iql_features import PublicFeatureCompiler

SCHEMA='history-probe-dataset-v1'
BASE_MANIFEST_SHA256='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea'
FOCAL_PREFIX='history-probe-v1/focal/'
VISIT_PREFIX='history-probe-v1/visit/'
DONOR_PREFIX='history-probe-v1/donor/'
AGE_SCALE_MS=360000
PATCHES=('16.17','16.18','16.19')
NEVER_OPEN=frozenset({'reward','terminated','elapsed_ms','next_state_index','fold_action_counts'})
TARGET_ARRAYS=frozenset({'full_action_id','main_group_id'})
STATE_ARRAYS=frozenset({'champion_indices','inventory_static','visible_level','global_public',
                       'inventory_offsets','inventory_item_indices','inventory_item_counts'})


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def initial_code_bindings():
    """Freeze this preparation and every own called helper before any input."""
    root=Path(__file__).resolve().parents[1]
    names=('winprob/history_probe_data.py','scripts/prepare_history_probe.py','winprob/admission.py',
           'winprob/full_action_catalogue.py','winprob/iql_dataset_contract.py','winprob/iql_features.py',
           'winprob/ipo_residual.py','winprob/preference_reference.py','winprob/trajectory_contract.py')
    return {str((root/name).resolve(strict=True)):file_hash(root/name) for name in names}


def write_json(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')


def fold_for_game(mid):return int.from_bytes(hashlib.sha256(('iql-v1-fold:'+mid).encode()).digest()[:8],'big')%2
def focal_pid(mid):return 1+int.from_bytes(hashlib.sha256((FOCAL_PREFIX+mid).encode()).digest()[:8],'big')%10


def make_selection(games,episodes,timestamps):
    """Reads only immutable identity/time metadata, never purchase targets."""
    by_player={}
    for episode_index,episode in enumerate(episodes):
        key=(episode['match_id'],episode['participant_id'])
        require(key not in by_player,'Duplicate complete player episode')
        by_player[key]=(episode_index,episode)
    entries=[]
    for ordinal,game in enumerate(sorted(games,key=lambda row:row['match_id'])):
        mid=game['match_id'];pid=focal_pid(mid)
        row={'ordinal':ordinal,'match_id':mid,'game_index':game['game_index'],'fold':game['fold'],'patch':game['patch'],
             'participant_id':pid,'status':'no_eligible_focal_episode','selected_state_index':None,
             'episode_index':None,'trajectory_id':None,'source_node_ordinal':None,'timestamp_ms':None,'visit_digest':None}
        found=by_player.get((mid,pid))
        if found is not None:
            ei,ep=found;lo=ep['first_state'];n=ep['states']
            def rank(j):
                ts=int(timestamps[lo+j]);raw=f'{VISIT_PREFIX}{mid}|{pid}|{j}|{ts}'
                return hashlib.sha256(raw.encode()).digest(),ts,j
            local=min(range(n),key=rank);digest,ts,_=rank(local)
            row.update(status='selected',selected_state_index=lo+local,episode_index=ei,
                       trajectory_id=ep['trajectory_id'],source_node_ordinal=local,timestamp_ms=ts,visit_digest=digest.hex())
        entries.append(row)
    return entries


def history_slots(episodes,timestamps,visit_end_ms,full_action_ids):
    n=len(timestamps);require(np.shape(visit_end_ms)==np.shape(full_action_ids)==(n,),'Unaligned closure/action sidecar')
    previous=np.full((n,2),-1,dtype=np.int64);present=np.zeros((n,2),dtype=bool);age=np.zeros((n,2),dtype=np.float32)
    ties=overlap=excluded_save=0
    for ep in episodes:
        lo=ep['first_state'];hi=lo+ep['states']
        require(np.all(np.diff(timestamps[lo:hi])>0) and np.all(visit_end_ms[lo:hi]>=timestamps[lo:hi]),'Invalid visit start/end chronology')
        prior_purchases=[]
        for current in range(lo,hi):
            for slot in range(2):
                if len(prior_purchases)<=slot:continue
                prior=prior_purchases[-slot-1]
                end=int(visit_end_ms[prior]);now=int(timestamps[current])
                if end==now:ties+=1;continue
                if end>now:overlap+=1;continue
                previous[current,slot]=prior;present[current,slot]=True
                age[current,slot]=(now-end)/AGE_SCALE_MS
            # Synthetic no-buy nodes use future qualification windows. Remove
            # them BEFORE adjacency; never let their existence displace a
            # causally closed purchase (ward-only real purchases remain).
            if full_action_ids[current]!=0:prior_purchases.append(current)
            else:excluded_save+=1
    require(overlap==0,'Source complete episodes unexpectedly overlap')
    return previous,present,age,{'closure_tied_slots_masked':ties,'overlapping_slots':overlap,
                                 'synthetic_save_nodes_excluded_before_adjacency':excluded_save}


def donor_indices(games,episodes,timestamps,champions,patch_index,fold,game_index,present,guard=lambda:None):
    """Donors are other TRAIN-fold games; basket content is never inspected."""
    n=len(timestamps);result=np.full((2,n),-1,dtype=np.int64)
    local=np.zeros(n,dtype=np.int32);pid=np.zeros(n,dtype=np.uint8)
    for ep in episodes:
        lo=ep['first_state'];hi=lo+ep['states'];local[lo:hi]=np.arange(ep['states']);pid[lo:hi]=ep['participant_id']
    mid=[game['match_id'] for game in games]
    patterns=present[:,0].astype(np.uint8)+2*present[:,1].astype(np.uint8)
    keys=list(zip(map(int,champions[:,0]),map(int,patch_index),map(int,timestamps//180000),map(int,patterns)))
    summaries={}
    for heldout in (0,1):
        guard();buckets=defaultdict(list)
        for row in np.flatnonzero(fold!=heldout):
            buckets[keys[row]].append(int(row))
        bucket_games={key:set(map(int,game_index[rows])) for key,rows in buckets.items()}
        eligible_cache={}
        for row,key in enumerate(keys):
            if row%25000==0:guard()
            members=buckets.get(key);recipient_game=int(game_index[row])
            if members is None or bucket_games[key]<={recipient_game}:continue
            if recipient_game not in bucket_games[key]:eligible=members
            else:
                cache_key=(key,recipient_game)
                if cache_key not in eligible_cache:
                    eligible_cache[cache_key]=[index for index in members if game_index[index]!=recipient_game]
                eligible=eligible_cache[cache_key]
            raw=f'{DONOR_PREFIX}{heldout}|{mid[recipient_game]}|{int(pid[row])}|{int(local[row])}|{int(timestamps[row])}'
            candidate=eligible[int.from_bytes(hashlib.sha256(raw.encode()).digest(),'big')%len(eligible)]
            require(fold[candidate]!=heldout and game_index[candidate]!=recipient_game
                    and np.array_equal(present[candidate],present[row]),'Donor escaped TRAIN or matching mask pattern')
            result[heldout,row]=candidate
        valid=result[heldout]>=0
        summaries[str(heldout)]={'recipient_rows':n,'donor_rows_found':int(valid.sum()),'missing_donor_rows':int((~valid).sum()),
            'missing_with_present_history':int(np.sum((~valid)&present.any(1))),'unique_donor_states':int(len(np.unique(result[heldout,valid]))),
            'unique_donor_games':int(len(np.unique(game_index[result[heldout,valid]]))),
            'training_fold':1-heldout}
    return result,summaries


def history_content_indices(previous,donors,rows,arm,heldout):
    """Output indexes identify ONLY earlier action content, never current labels."""
    rows=np.asarray(rows,dtype=np.int64)
    if arm=='state':return np.full((len(rows),2),-1,dtype=np.int64)
    if arm=='history':return np.array(previous[rows],copy=True)
    require(arm=='donor' and heldout in (0,1),'Unknown fixed history arm/fold')
    indexes=donors[heldout,rows];out=np.full((len(rows),2),-1,dtype=np.int64);valid=indexes>=0
    out[valid]=previous[indexes[valid]]
    return out


def history_timing(present,age,rows,arm):
    rows=np.asarray(rows,dtype=np.int64)
    require(arm in ('state','history','donor'),'Unknown history timing arm')
    if arm=='state':return np.zeros((len(rows),2),dtype=bool),np.zeros((len(rows),2),dtype=np.float32)
    return np.array(present[rows],copy=True),np.array(age[rows],copy=True)


class BaseCache:
    """Target-deferred, outcome-free reader of the frozen existing cache."""
    def __init__(self,directory,protocol):
        self.directory=Path(directory).resolve(strict=True);reject_quarantined(self.directory)
        self.bindings={};self.manifest,self.digest=bound_json(self.directory/'manifest.json',self.bindings)
        require(self.digest==BASE_MANIFEST_SHA256,'Only the original immutable IQL FIT cache is admitted')
        m=self.manifest
        require(m.get('schema')=='iql-dataset-v1' and m.get('state')=='complete' and m.get('smoke') is False
                and m.get('training_authorized') is True and m.get('gates',{}).get('passed') is True
                and (m['state_count'],m['episode_count'],m['game_count'])==(392448,33212,4001),'Wrong frozen FIT feature cache')
        self.admission_path=Path(m['source_admission_path']).resolve(strict=True)
        self.admitted,_=bound_json(self.admission_path,self.bindings)
        require(file_hash(self.admission_path)==m['source_admission_sha256'],'Cache admission changed')
        forbidden=current_reservations(self.admitted);admitted={r['match_id']:r for r in self.admitted['games']}
        self.games=[{**row,'game_index':i} for i,row in enumerate(m['game_metadata'])]
        require(len({row['match_id'] for row in self.games})==4001,'Duplicate base game identity')
        for row in self.games:
            entry=admitted.get(row['match_id'])
            require(row['match_id'] not in forbidden and entry is not None and entry['role']=='train'
                    and entry.get('historical_validation') is False and row['patch']==entry['patch']
                    and row['patch'] in ('16.17','16.18') and row['fold']==fold_for_game(row['match_id']),
                    'Base cache game is not original nonreserved FIT/fold')
        # Only metadata/code bindings are followed before selection. Source
        # compressed trajectories, and current action arrays, remain unopened.
        self.deferred_bindings={}
        for raw,want in m['bindings'].items():
            path=Path(raw).resolve(strict=True)
            if path.suffix.lower() in ('.json','.sha256','.py'):
                require(file_hash(path)==want,'Bound base metadata/code changed: '+str(path));self.bindings[str(path)]=want
            else:self.deferred_bindings[str(path)]=want
        protocol=Path(protocol).resolve(strict=True)
        require(protocol.name=='history_probe_protocol_v1.md','Wrong protocol')
        text=protocol.read_text(encoding='utf-8')
        require(all(salt in text for salt in (FOCAL_PREFIX,VISIT_PREFIX,DONOR_PREFIX)),'Protocol does not bind sampler/donor salts')
        self.bindings[str(protocol)]=file_hash(protocol);self.protocol=protocol
        self.arrays={};self.selection_frozen=False
        self.episodes=self.json_member(m['episode_metadata_file'])
        require(len(self.episodes)==33212,'Base episode count changed')
        ts=self.array('timestamp_ms');fold=self.array('fold');gi=self.array('game_index');pi=self.array('patch_index');ei=self.array('episode_index')
        cursor=0;identities=set()
        for index,ep in enumerate(self.episodes):
            lo=ep['first_state'];hi=lo+ep['states'];game=self.games[ep['game_index']]
            require(ep['trajectory_id']==f"{ep['match_id']}:{ep['participant_id']}" and ep['trajectory_id'] not in identities
                    and ep['first_state']==cursor and ep['states']>0 and 1<=ep['participant_id']<=10
                    and ep['match_id']==game['match_id'] and ep['fold']==game['fold'] and ep['patch']==game['patch']
                    and np.all(ei[lo:hi]==index) and np.all(gi[lo:hi]==ep['game_index']) and np.all(fold[lo:hi]==ep['fold'])
                    and np.all(pi[lo:hi]==PATCHES.index(ep['patch'])) and np.all(np.diff(ts[lo:hi])>0),
                    'Base complete episode identity/time/fold graph changed')
            identities.add(ep['trajectory_id']);cursor=hi
        require(cursor==392448,'Base episode ranges are incomplete')

    def member(self,name):
        path=(self.directory/name).resolve(strict=True)
        require(path.is_relative_to(self.directory) and path.is_file(),'Cache member escaped bound directory')
        require(file_hash(path)==self.manifest['files'][name],'Cache member digest changed: '+name)
        self.bindings[str(path)]=self.manifest['files'][name]
        return path

    def json_member(self,name):return json_file(self.member(name))

    def array(self,name):
        require(name not in NEVER_OPEN,'Outcome/terminal/support array is outside history-probe scope')
        require(name not in TARGET_ARRAYS or self.selection_frozen,'Current action targets cannot be opened before selection publication')
        if name not in self.arrays:
            spec=self.manifest['arrays'][name];path=self.member(spec['file']);value=np.load(path,mmap_mode='r',allow_pickle=False)
            require(str(value.dtype)==spec['dtype'] and list(value.shape)==spec['shape'],'Base array schema changed: '+name)
            self.arrays[name]=value
        return self.arrays[name]

    def revalidate(self):
        for path,want in self.bindings.items():require(file_hash(path)==want,'Input changed while preparing history: '+path)
        require(not {r['match_id'] for r in self.games}&current_reservations(self.admitted),'Base FIT game became reserved')


def source_closures(cache,bundle,guard=lambda:None):
    """Stream exported nodes for closure/action parity without inspecting rewards."""
    require(cache.selection_frozen,'Selection must precede trajectory deserialization')
    actions=cache.array('full_action_id');groups=cache.array('main_group_id');timestamps=cache.array('timestamp_ms')
    ends=np.full(len(timestamps),-1,dtype=np.int64);source_events=np.full(len(timestamps),-1,dtype=np.int64)
    by_id={ep['trajectory_id']:(i,ep) for i,ep in enumerate(cache.episodes)};seen=set();reports=[]
    for source_index,shard in enumerate(bundle['sources']):
        digest=hashlib.sha256();records=nodes=0
        with gzip.open(shard['rows_path'],'rb') as stream:
            for line in stream:
                if records%250==0:guard()
                digest.update(line);record=json.loads(line)
                identity=record['trajectory_id'];require(identity in by_id and identity not in seen,'Unexpected/duplicate source episode')
                _,ep=by_id[identity];decisions=record['decisions']
                require(record['match_id']==ep['match_id'] and record['participant_id']==ep['participant_id']
                        and record['patch']==ep['patch']==shard['manifest']['patch'] and record['role']==record['consumer_role']=='fit'
                        and record['whole_player_game_eligible'] is True and not record['ineligibility_reasons']
                        and ep['source_index']==source_index and len(decisions)==ep['states'],
                        'Source complete episode differs from fixed cache')
                for ordinal,node in enumerate(decisions):
                    row=ep['first_state']+ordinal;t=node['timestamp_ms'];end=node.get('visit_end_ms');event=node['event_index']
                    require(type(t) is int and type(end) is int and type(event) is int and end>=t
                            and t==timestamps[row] and node['node_id']==f"{record['match_id']}:{record['participant_id']}:{event}"
                            and node['kind']=='decision','Source visit identity/closure changed')
                    if ordinal+1<len(decisions):require(end<=decisions[ordinal+1]['timestamp_ms'],'Source visit overlaps next node')
                    action=node['action'];key=counts_key(action['label_counts']);expected=bundle['lookup'].get(key)
                    require(expected is not None and action['kind']==key[0] and action['wards_preserved'] is True
                            and action['full_action_id']==expected==actions[row]
                            and action['main_group_id']==bundle['catalogue']['full_to_group'][expected]==groups[row],
                            'Source full retained action differs from fixed cache')
                    ends[row]=end;source_events[row]=event;nodes+=1
                seen.add(identity);records+=1
        require(digest.hexdigest()==shard['manifest']['uncompressed_rows_sha256'] and records==shard['manifest']['trajectory_count']
                and nodes==shard['manifest']['counts']['decision_nodes'],'Source closure scan count/digest mismatch')
        reports.append({'source_index':source_index,'patch':shard['manifest']['patch'],'episodes':records,'nodes':nodes,
                        'uncompressed_sha256':digest.hexdigest()})
    require(seen==set(by_id) and np.all(ends>=timestamps),'Closure sidecar did not cover every fixed state')
    return ends,source_events,reports


def prepare(cache_directory,output,protocol,*,deadline_seconds=900,guard=lambda:None):
    started=time.monotonic();output=Path(output).resolve();require(not output.exists(),'Refuse to overwrite history dataset')
    require(1<=deadline_seconds<=900,'History preparation ceiling is900seconds')
    initial_code=initial_code_bindings()
    def check_budget():
        guard();require(time.monotonic()-started<=deadline_seconds,'History preparation deadline exceeded')
    cache=BaseCache(cache_directory,protocol);m=cache.manifest;check_budget()
    for path,expected in initial_code.items():
        require(path not in cache.bindings or cache.bindings[path]==expected,'Initial helper differs from frozen base provenance')
        require(file_hash(path)==expected,'Preparation code changed before selection publication')
        cache.bindings[path]=expected
    entries=make_selection(cache.games,cache.episodes,cache.array('timestamp_ms'))
    output.mkdir(parents=True,exist_ok=False)
    selection={'schema':'history-probe-selection-v1','state':'complete','entries':entries,'game_count':4001,
               'selected_count':sum(row['status']=='selected' for row in entries),
               'selection_before_action_payloads':True,'current_action_arrays_opened':False,'source_trajectories_opened':False,
               'cache_manifest_path':str(cache.directory/'manifest.json'),'cache_manifest_sha256':cache.digest,
               'bindings_sha256':dict(cache.bindings),'protocol_path':str(cache.protocol),'protocol_sha256':file_hash(cache.protocol),
               'salts':{'focal':FOCAL_PREFIX,'visit':VISIT_PREFIX},'node_ordinal':'zero-based complete source decisions list / cache episode offset'}
    write_json(output/'selection.json',selection);cache.selection_frozen=True
    def status(stage,**extra):
        (output/'status.json').write_text(json.dumps({'stage':stage,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n',encoding='utf-8')
    try:
        status('selection_frozen_before_targets',selected=selection['selected_count'])
        bundle=validate_sources(m['source_directories'],cache.admission_path,Path(m['catalogue_path']))
        require(set(bundle['eligible_fit_ids'])=={game['match_id'] for game in cache.games},'Source eligible game set differs from cache')
        cache.bindings.update(bundle['bindings']);check_budget()
        source_root=Path(cache.admitted['source_data_root']).resolve().parent
        closure_proof={}
        for relative in ('app/footwear_reconstruction.py','app/reconstruct.py','app/inventory.py',
                         'scripts/footwear_timer_inventory.py','scripts/footwear_economics.py',
                         'scripts/reconstructible_1617_runtime.py','scripts/reconstructible_1618_runtime.py'):
            path=(source_root/relative).resolve(strict=True);observed=file_hash(path)
            require(str(path) not in cache.bindings or cache.bindings[str(path)]==observed,'Reviewed closed-visit source code changed')
            cache.bindings[str(path)]=observed;closure_proof[str(path)]=observed
        ends,events,closure_report=source_closures(cache,bundle,check_budget)
        previous,present,age,closure_counts=history_slots(cache.episodes,cache.array('timestamp_ms'),ends,cache.array('full_action_id'))
        # Bind all public state inputs and frozen complete-action tables. No
        # terminal/reward/next-state array is opened, even just for hashing.
        for name in STATE_ARRAYS|{'full_action_id','main_group_id','action_static','action_static_valid','action_offsets',
                                  'action_item_indices','action_item_counts','full_to_group'}:cache.array(name)
        static,static_digest=bound_json(Path(m['static_export_path']),cache.bindings)
        require(static_digest==m['static_export_sha256'],'Native static artifact changed')
        compiler=PublicFeatureCompiler(static['patches']);feature_metadata=cache.json_member(m['feature_metadata_file'])
        require(json.loads(json.dumps(compiler.metadata()))==feature_metadata,'Fixed public vocabulary/feature schema changed')
        catalogue=bundle['catalogue'];table=cache.array('action_static');valid=cache.array('action_static_valid')
        for pi,patch in enumerate(PATCHES):
            for aid,(basket,kind) in enumerate(zip(catalogue['full_actions'],catalogue['full_action_kinds'],strict=True)):
                try:_,vector=compiler.full_action(basket,kind,patch)
                except (ValueError,KeyError):require(not valid[pi,aid],'Cache claims unavailable native static features');continue
                require(valid[pi,aid] and np.array_equal(vector,table[pi,aid]),'Frozen native23 action attributes changed')
            check_budget()
        status('closed_visit_history_ready',states=len(ends),**closure_counts)
        donors,donor_counts=donor_indices(cache.games,cache.episodes,cache.array('timestamp_ms'),cache.array('champion_indices'),
                         cache.array('patch_index'),cache.array('fold'),cache.array('game_index'),present,check_budget)
        arrays={'visit_end_ms':ends,'source_event_index':events,'prev_state_index':previous,'history_present':present,'history_age':age,
                'donor_history_state_index':donors,'selected_state_indices':np.asarray([r['selected_state_index'] for r in entries if r['status']=='selected'],dtype=np.int64)}
        for heldout in (0,1):
            selected=arrays['selected_state_indices'][cache.array('fold')[arrays['selected_state_indices']]==heldout]
            found=donors[heldout,selected]>=0
            donor_counts[str(heldout)].update(selected_heldout_rows=len(selected),selected_with_donor=int(found.sum()),
                selected_missing_donor=int((~found).sum()),
                selected_missing_with_present_history=int(np.sum((~found)&present[selected].any(1))))
        specs={}
        for name,value in arrays.items():
            path=output/(name+'.npy');np.save(path,value,allow_pickle=False);specs[name]={'file':path.name,'dtype':str(value.dtype),'shape':list(value.shape)}
        cache.bindings[str(output/'selection.json')]=file_hash(output/'selection.json')
        cache.revalidate();check_budget()
        files={path.name:file_hash(path) for path in output.iterdir() if path.name.endswith('.npy') or path.name=='selection.json'}
        require(selection['selected_count']==3350,'Frozen3350/651 metadata selection count changed')
        manifest={'schema':SCHEMA,'state':'complete','training_authorized':True,'smoke':False,
                  'gates':{'passed':True,'minimum_selected_games':selection['selected_count']>=3000,
                           'selection_counts_match_frozen_census':selection['selected_count']==3350,
                           'strict_history_closure_verified':True,'source_and_cache_action_parity_verified':True},
                  'selection_file':'selection.json','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  'arrays':specs,'files':files,'cache_manifest_path':str(cache.directory/'manifest.json'),'cache_manifest_sha256':cache.digest,
                  'cache_directory':str(cache.directory),'base_state_count':392448,'state_count':392448,'episode_count':33212,'game_count':4001,
                  'selected_count':selection['selected_count'],'selection_status_counts':dict(Counter(row['status'] for row in entries)),
                  'game_metadata':cache.games,'source_closure_reports':closure_report,'closure_counts':closure_counts,
                  'history_mask_pattern_counts':dict(Counter(str(tuple(map(int,row))) for row in present)),
                  'history_mask_pattern_counts_by_fold':{str(f):dict(Counter(str(tuple(map(int,row))) for row in present[cache.array('fold')==f])) for f in (0,1)},
                  'history_age_formula':'(current_start_ms-prior_visit_end_ms)/360000; float32; no cap; absent0',
                  'history_slots':'previous1and2 real purchase nodes, excluding full_action_id0 BEFORE adjacency; closure<currentstart; no older purchase replacement for ties',
                  'synthetic_save_history':'Excluded before adjacency because its existence uses future no-purchase/endgame windows; current SAVE targets retained.',
                  'purchase_label_closure_proof':'Source footwear_reconstruction.close_visit pops the current open visit and computes labels from copied inventory_before/inventory_after; maybe_close runs before the current item batch; emitted purchase label objects are not rewritten by later undo.',
                  'purchase_label_closure_code_sha256':closure_proof,
                  'donor_axis':'heldout fold0or1; donor only opposite TRAIN fold and another game',
                  'donor_rule':DONOR_PREFIX+'heldout|recipient_mid|recipient_pid|recipient_nodeordinal|recipient_ts; full-big-endian SHA256 modulo sorted eligible TRAIN donor state indices, excluding recipient game',
                  'donor_counts':donor_counts,'donor_fallback':'zero basket content, preserve recipient age/presence',
                  'static_export_path':m['static_export_path'],'static_export_sha256':m['static_export_sha256'],
                  'catalogue_path':m['catalogue_path'],'catalogue_sha256':m['catalogue_sha256'],
                  'admission_path':str(cache.admission_path),'admission_sha256':m['source_admission_sha256'],
                  'protocol_path':str(cache.protocol),'protocol_sha256':file_hash(cache.protocol),'bindings':cache.bindings,
                  'opaque_outcome_cache_members_not_opened':{name:m['files'][m['arrays'][name]['file']] for name in NEVER_OPEN},
                  'selection_precedes_current_action_access':True,'raw_sources_reopened':False,'source_code_imported':False,
                  'test_payload_reads':False,'confirmation_payload_reads':False,'reward_arrays_opened':False,
                  'source_json_terminal_fields_deserialized':True,'terminal_outcomes_used_or_analyzed':False,
                  'training_performed':False,'elapsed_s':time.monotonic()-started,
                  'limits':['Conditional complete catalogue-supported FIT episodes only; missing focal episodes never replaced.',
                            'Source JSON contains terminal fields during deserialization; no terminal field is indexed or used.',
                            'History retains complete prior real purchase multisets including wards; synthetic SAVE nodes are excluded before adjacency.',
                            'Current target SAVE remains in the unchanged endpoint; no wallet override or displayed policy claim.']}
        write_json(output/'manifest.json',manifest);status('complete',selected=selection['selected_count']);return manifest
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


class HistoryProbeData:
    """Trainer-facing verified arrays; no reward loading and no source imports."""
    def __init__(self,directory):
        self.directory=Path(directory).resolve(strict=True);self.bindings={}
        self.manifest,_=bound_json(self.directory/'manifest.json',self.bindings);m=self.manifest
        require(m.get('schema')==SCHEMA and m.get('state')=='complete' and m.get('reward_arrays_opened') is False
                and m.get('terminal_outcomes_used_or_analyzed') is False and m.get('training_authorized') is True
                and m.get('smoke') is False and m.get('gates',{}).get('passed') is True,
                'Unfinished, failed-gate, smoke or outcome-using history artifact')
        expected_specs={
            'visit_end_ms':('int64',[392448]),'source_event_index':('int64',[392448]),
            'prev_state_index':('int64',[392448,2]),'history_present':('bool',[392448,2]),
            'history_age':('float32',[392448,2]),'donor_history_state_index':('int64',[2,392448]),
            'selected_state_indices':('int64',[3350])}
        require(set(m['arrays'])==set(expected_specs) and m.get('state_count')==392448 and m.get('selected_count')==3350,
                'History sidecar axes or fixed selection count changed')
        for name,(dtype,shape) in expected_specs.items():
            require(m['arrays'][name]=={'file':name+'.npy','dtype':dtype,'shape':shape},'Unexpected fixed history array specification')
        for path,want in m['bindings'].items():require(file_hash(path)==want,'Prepared history binding changed');self.bindings[path]=want
        self.cache=BaseCache(m['cache_directory'],m['protocol_path']);require(self.cache.digest==m['cache_manifest_sha256'],'History base cache changed')
        self.selection,_=bound_json(self.directory/'selection.json',self.bindings)
        expected=make_selection(self.cache.games,self.cache.episodes,self.cache.array('timestamp_ms'))
        require(expected==self.selection['entries'] and self.selection.get('selection_before_action_payloads') is True,'History selection does not reproduce')
        self.cache.selection_frozen=True;self.arrays={}
        for name,spec in m['arrays'].items():
            path=(self.directory/spec['file']).resolve(strict=True)
            require(path.is_relative_to(self.directory) and file_hash(path)==m['files'][spec['file']],'History array changed')
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            require(str(value.dtype)==spec['dtype'] and list(value.shape)==spec['shape'],'History array schema changed')
            self.arrays[name]=value;self.bindings[str(path)]=m['files'][spec['file']]
        self.base_manifest=self.cache.manifest;self.games=self.cache.games;self.episodes=self.cache.episodes
        self.bindings.update(self.cache.bindings)

    def base_array(self,name):return self.cache.array(name)

    def revalidate(self):
        self.cache.revalidate()
        for path,want in self.bindings.items():require(file_hash(path)==want,'History training input changed: '+str(path))
