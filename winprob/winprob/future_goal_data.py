"""Training-only future-goal labels and matched control; no inference inputs.

Consumes frozen own artifacts, never imports sibling source code. Parent
terminal timing is an opaque provenance dependency, not reopened or decoded.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from . import admission
from .full_action_catalogue import load_catalogue
from .history_probe_data import HistoryProbeData,BASE_MANIFEST_SHA256
from .iql_dataset_contract import current_reservations,file_hash,reject_quarantined
from .iql_features import PublicFeatureCompiler

SCHEMA='future-goal-dataset-v1'
DONOR_PREFIX='future-goal-v1/donor/'
PATCHES=('16.17','16.18','16.19')
DEFINITION_SHA='4457a015b1d08a0893acfc19b8016b7a7104d98f220f2c893949d055144a9072'
CENSUS_SHA='367293d55cdeae9aba107055be2eca8af0b4bd03c95de75fcc19dd424624d566'
PROTOCOL_SHA='545dd8121281159516fb14661edf663d5e1d65f51ee17a7383e6175ae573b6bb'
TIMING_NAMES={'terminated.npy','elapsed_ms.npy','next_state_index.npy'}


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_input_path(path):
    path=Path(path).resolve(strict=True);reject_quarantined(path)
    require(path.name!='reward.npy' and path.name not in TIMING_NAMES and 'raw_match_v5' not in path.parts
            and path.suffix.lower() not in ('.db','.sqlite'),'Forbidden outcome/raw/timing payload dependency')
    return path


def bind(path,want,bindings):
    path=safe_input_path(path)
    key=str(path)
    require(key not in bindings or bindings[key]==want,'Conflicting input binding')
    require(file_hash(path)==want,'Input/code digest changed: '+key);bindings[key]=want


def bound(path,bindings):
    path=safe_input_path(path);digest=file_hash(path)
    require(path.with_suffix('.sha256').read_text().split()==[digest,path.name],'Artifact sidecar differs')
    bind(path,digest,bindings);return read(path)


def revalidate(bindings):
    for path,want in bindings.items():
        reject_quarantined(path);require(file_hash(path)==want,'Artifact changed during preparation: '+path)


def write_json(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')


def initial_bindings():
    root=Path(__file__).resolve().parents[1]
    names=('winprob/future_goal_data.py','scripts/prepare_future_goal.py','winprob/history_probe_data.py',
           'winprob/admission.py','winprob/full_action_catalogue.py','winprob/iql_dataset_contract.py','winprob/iql_features.py',
           'winprob/ipo_residual.py','winprob/preference_reference.py','winprob/trajectory_contract.py')
    return {str((root/name).resolve()):file_hash(root/name) for name in names}


def collapse_goals(diagnostic,vocabulary):
    require(len(vocabulary)==259 and vocabulary[0]['goal_counts']==vocabulary[1]['goal_counts']==[],
            'Expected259diagnostic codes with two EMPTY statuses')
    raw=np.asarray(diagnostic)
    require(np.issubdtype(raw.dtype,np.integer) and np.all((raw>=0)&(raw<259)),'Invalid diagnostic goal codes')
    goals=[{'goal_id':0,'kind':'EMPTY','goal_counts':[]}]
    seen={()}
    for code,entry in enumerate(vocabulary[2:],2):
        key=tuple(tuple(pair) for pair in entry['goal_counts'])
        require(key and key==tuple(sorted(key)) and key not in seen
                and len({item for item,count in key})==len(key)
                and all(type(item) is int and type(count) is int and item>0 and count>0 for item,count in key),
                'Goal class loses canonical multiplicity or duplicates another class')
        seen.add(key);kind='SINGLE' if sum(count for item,count in key)==1 else 'MULTIPLE'
        require(entry['type']==kind,'Goal MULTIPLE semantic tag changed')
        goals.append({'goal_id':code-1,'kind':kind,'goal_counts':[list(pair) for pair in key]})
    return np.where(raw<2,0,raw-1).astype(np.int64),goals


def padded_baskets(baskets,item_index):
    width=max(1,max(len(basket) for basket in baskets));ids=np.zeros((len(baskets),width),dtype=np.int64)
    counts=np.zeros((len(baskets),width),dtype=np.float32)
    for row,basket in enumerate(baskets):
        seen=set()
        for column,(item,count) in enumerate(basket):
            require(type(item) is int and item in item_index and item not in seen and type(count) is int and 0<count<=2**24,
                    'Unknown item, duplicate row, or inexact multiplicity')
            seen.add(item);ids[row,column]=item_index[item];counts[row,column]=count
    return ids,counts


def public_tables(compiler,main_baskets,goal_vocabulary,guard=lambda:None):
    require(main_baskets and main_baskets[0]==[] and goal_vocabulary[0]['goal_counts']==[],'EMPTY basket identity changed')
    main_ids,main_counts=padded_baskets(main_baskets,compiler.item_index)
    goals=[entry['goal_counts'] for entry in goal_vocabulary];goal_ids,goal_counts=padded_baskets(goals,compiler.item_index)
    main_static=np.zeros((3,len(main_baskets),23),dtype=np.float32);main_valid=np.zeros((3,len(main_baskets)),dtype=bool)
    goal_static=np.zeros((3,len(goals),23),dtype=np.float32);goal_valid=np.ones((3,len(goals)),dtype=bool)
    for pi,patch in enumerate(PATCHES):
        for group,basket in enumerate(main_baskets):
            # Only genuine absence from the native item table permits the
            # declared neutral-composition fallback. Corrupt metadata or a
            # compiler error must fail, never silently mask a known class.
            if any(str(item) not in compiler.static[patch]['items'] for item,count in basket):continue
            _,features=compiler.full_action(basket,'retained_net_purchase' if group else 'observed_no_buy_death',patch)
            require(np.shape(features)==(23,) and np.isfinite(features).all(),'Invalid main native static attributes')
            main_static[pi,group]=features;main_valid[pi,group]=True
            if group%1024==0:guard()
        for goal,basket in enumerate(goals[1:],1):
            try:_,features=compiler.full_action(basket,'retained_net_purchase',patch)
            except (KeyError,ValueError) as error:raise ValueError('Every fixed future goal needs native attributes') from error
            require(np.shape(features)==(23,) and np.isfinite(features).all(),'Invalid goal native static attributes')
            goal_static[pi,goal]=features
        guard()
    require(np.all(main_static[:,0,:22]==0) and np.all(main_static[:,0,22]==1) and main_valid[:,0].all(),
            'Main group0 SAVE flag changed')
    require(np.all(goal_static[:,0]==0),'Goal EMPTY must have all23 attributes zero')
    return {'main_item_indices':main_ids,'main_item_counts':main_counts,'main_static':main_static,'main_static_valid':main_valid,
            'goal_item_indices':goal_ids,'goal_item_counts':goal_counts,'goal_static':goal_static,'goal_static_valid':goal_valid}


def kth_unexcluded(excluded,rank,total):
    """Index of rank-th allowed member without materializing filtered buckets."""
    excluded=np.asarray(excluded,dtype=np.int64)
    require(0<=rank<total-len(excluded),'No eligible donor at requested rank')
    lo=rank;hi=rank+len(excluded)
    while lo<hi:
        mid=(lo+hi)//2
        allowed_through=mid+1-int(np.searchsorted(excluded,mid,side='right'))
        if allowed_through>rank:hi=mid
        else:lo=mid+1
    require(lo<total and not np.any(excluded==lo),'Excluded donor rank leaked')
    return lo


def shuffle_training_goals(real,main,patch,fold,game,timestamps,games,episodes,guard=lambda:None):
    n=len(real);require(all(np.shape(value)==(n,) for value in (main,patch,fold,game,timestamps)),'Unaligned auxiliary labels')
    shuffled=np.full((2,n),-1,dtype=np.int64);donor=np.full((2,n),-1,dtype=np.int64)
    pid=np.zeros(n,dtype=np.uint8);ordinal=np.zeros(n,dtype=np.int32)
    for ep in episodes:
        lo=ep['first_state'];hi=lo+ep['states'];pid[lo:hi]=ep['participant_id'];ordinal[lo:hi]=np.arange(ep['states'])
    summaries={}
    for heldout in (0,1):
        guard();train=np.flatnonzero(fold!=heldout);buckets={}
        for row in train:
            key=(int(main[row]),int(patch[row]),bool(real[row]==0))
            bucket=buckets.setdefault(key,{'rows':[],'excluded':defaultdict(list)})
            bucket['excluded'][int(game[row])].append(len(bucket['rows']));bucket['rows'].append(int(row))
        for bucket in buckets.values():
            bucket['rows']=np.asarray(bucket['rows'],dtype=np.int64)
            bucket['excluded']={gi:np.asarray(indexes,dtype=np.int64) for gi,indexes in bucket['excluded'].items()}
        for step,row in enumerate(train):
            if step%10000==0:guard()
            gi=int(game[row]);key=(int(main[row]),int(patch[row]),bool(real[row]==0));bucket=buckets[key]
            excluded=bucket['excluded'][gi];available=len(bucket['rows'])-len(excluded)
            if not available:shuffled[heldout,row]=real[row];continue
            raw=f'{DONOR_PREFIX}{heldout}|{games[gi]["match_id"]}|{int(pid[row])}|{int(ordinal[row])}|{int(timestamps[row])}'
            rank=int.from_bytes(hashlib.sha256(raw.encode()).digest(),'big')%available
            chosen=int(bucket['rows'][kth_unexcluded(excluded,rank,len(bucket['rows']))])
            require(fold[chosen]!=heldout and game[chosen]!=game[row] and main[chosen]==main[row] and patch[chosen]==patch[row]
                    and (real[chosen]==0)==(real[row]==0),'Auxiliary donor escaped fixed training stratum')
            donor[heldout,row]=chosen;shuffled[heldout,row]=real[chosen]
        missing=donor[heldout,train]<0;changed=shuffled[heldout,train]!=real[train];nonempty=real[train]!=0
        require(np.all(shuffled[heldout,fold==heldout]==-1) and np.all(donor[heldout,fold==heldout]==-1),'Held-out row received a training target/donor')
        summaries[str(heldout)]={'train_states':len(train),'heldout_states':int((fold==heldout).sum()),
            'matching_buckets':len(buckets),'missing_donor_states':int(missing.sum()),'missing_donor_fraction':float(missing.mean()),
            'actually_changed_states':int(changed.sum()),'actually_changed_fraction':float(changed.mean()),
            'nonempty_train_states':int(nonempty.sum()),'nonempty_missing_donor_states':int((missing&nonempty).sum()),
            'nonempty_actually_changed_states':int((changed&nonempty).sum()),
            'nonempty_actually_changed_fraction':float(changed[nonempty].mean()) if nonempty.any() else 0.,
            'distinct_donor_states':len(np.unique(donor[heldout,train][~missing])),
            'distinct_donor_games':len(np.unique(game[donor[heldout,train][~missing]])),
            'weighting':'each original training state once; no row/goal filtering'}
        del buckets
    return shuffled,donor,summaries


def prepare(args,guard=lambda:None):
    started=time.monotonic();output=Path(args.output).resolve();require(not output.exists(),'Refuse future goal dataset overwrite')
    require(1<=args.deadline_seconds<=900,'Preparation ceiling900seconds')
    bindings=initial_bindings();opaque={}
    def budget():guard();require(time.monotonic()-started<=args.deadline_seconds,'Future goal preparation deadline exceeded')
    protocol=Path(args.protocol).resolve(strict=True);reject_quarantined(protocol)
    require(protocol.name=='future_goal_protocol_v1.md' and file_hash(protocol)==PROTOCOL_SHA
            and DONOR_PREFIX in protocol.read_text(encoding='utf-8'),'Wrong frozen future supervision protocol/salt')
    bind(protocol,file_hash(protocol),bindings);budget()
    base=bound(args.base_manifest,bindings);history=bound(args.history_manifest,bindings)
    census=bound(args.census_report,bindings);definition=bound(args.definition,bindings);audit=bound(args.census_audit,bindings)
    require(file_hash(args.base_manifest)==BASE_MANIFEST_SHA256 and file_hash(args.definition)==DEFINITION_SHA
            and file_hash(args.census_report)==CENSUS_SHA,'Wrong fixed base/definition/census')
    require(census['state']=='complete' and census['schema']=='goal-completion-census-v1' and census['horizon_ms']==360000
            and Path(census['definition_path']).resolve()==Path(args.definition).resolve() and census['definition_sha256']==DEFINITION_SHA,
            'Census definition/horizon lineage changed')
    require(audit['state']=='passed' and audit['all_label_ids_and_selected_visit_indices_equal'] is True
            and audit['report_sha256']==CENSUS_SHA and audit['definition_sha256']==DEFINITION_SHA,'Independent census audit missing')
    require(Path(definition['base_manifest_path']).resolve()==Path(args.base_manifest).resolve()
            and Path(definition['history_manifest_path']).resolve()==Path(args.history_manifest).resolve()
            and definition['base_manifest_sha256']==BASE_MANIFEST_SHA256
            and definition['history_manifest_sha256']==file_hash(args.history_manifest)
            and history['cache_manifest_sha256']==BASE_MANIFEST_SHA256,'History/census/base world changed')
    require(census['reward_array_read'] is False and census['test_or_confirmation_payload_reads'] is False,'Census boundary failed')
    history_data=HistoryProbeData(Path(args.history_manifest).parent);base_reader=history_data.cache
    require(base_reader.digest==BASE_MANIFEST_SHA256,'Wrong old cache reader')
    for path,want in history_data.bindings.items():bind(path,want,bindings)
    # Parent timing identities are cryptographically recorded but NOT opened.
    for path,want in census['bindings_sha256'].items():
        p=Path(path).resolve(strict=True)
        require(p.name!='reward.npy','Unexpected win-reward dependency')
        if p.name in TIMING_NAMES:
            require(p.parent==Path(args.base_manifest).resolve().parent and base['files'].get(p.name)==want,'Opaque timing lineage differs')
            opaque[str(p)]=want
        else:bind(p,want,bindings)
    for path,want in definition['bindings_sha256'].items():bind(path,want,bindings)
    budget()
    census_dir=Path(args.census_report).resolve().parent
    for name,want in census['files'].items():bind(census_dir/name,want,bindings)
    vocabulary=read(census_dir/'label_vocabulary.json')
    require(vocabulary==definition['label_vocabulary'],'Goal static vocabulary differs from definition')
    with np.load(census_dir/'labels.npz',allow_pickle=False) as archive:
        codes=archive['strictly_later_label_id'];future=archive['strictly_later_qualifying_state_index']
    real,goal_vocabulary=collapse_goals(codes,vocabulary)
    n=len(real);require(n==392448 and future.shape==(n,) and np.issubdtype(future.dtype,np.integer),'Future labels do not cover frozen states')
    names=('full_action_id','main_group_id','patch_index','fold','game_index','episode_index','timestamp_ms')
    a={name:history_data.base_array(name) for name in names};ends=history_data.arrays['visit_end_ms']
    available=real!=0;rows=np.flatnonzero(available);chosen=future[rows]
    require(np.all(future[~available]==-1) and np.all(chosen>rows) and np.all(chosen<n)
            and np.all(a['episode_index'][chosen]==a['episode_index'][rows])
            and np.all(ends[chosen]<=a['timestamp_ms'][rows]+360000) and np.all(a['full_action_id'][chosen]!=0),
            'Goal label is not a strictly later closed real purchase within six minutes')
    catalogue,lookup=load_catalogue(base['catalogue_path']);static=bound(base['static_export_path'],bindings)
    require(file_hash(base['catalogue_path'])==base['catalogue_sha256']==definition['catalogue_sha256']
            and file_hash(base['static_export_path'])==base['static_export_sha256']==definition['static_export_sha256'],
            'Static/catalogue lineage changed')
    for patch in static['patches'].values():
        for file in patch['source_files']:bind(file['path'],file['sha256'],bindings)
    compiler=PublicFeatureCompiler(static['patches'])
    metadata=base_reader.json_member(base['feature_metadata_file'])
    require(json.loads(json.dumps(compiler.metadata()))==metadata,'Shared immutable public vocabulary differs from base')
    qualified={patch:set(definition['goal_item_ids_by_patch'][patch]) for patch in PATCHES}
    full_goal={}
    for pi,patch in enumerate(PATCHES):
        for aid,basket in enumerate(catalogue['full_actions']):full_goal[pi,aid]=[[item,count] for item,count in basket if item in qualified[patch]]
    for row,chosen in zip(rows,chosen,strict=True):
        require(goal_vocabulary[int(real[row])]['goal_counts']==full_goal[int(a['patch_index'][chosen]),int(a['full_action_id'][chosen])],
                'Future goal multiset does not match its bound later purchase')
    tables=public_tables(compiler,catalogue['group_main'],goal_vocabulary,budget)
    require(tables['main_static_valid'][a['patch_index'],a['main_group_id']].all(),'Observed main target lacks native public attributes; no row may be dropped')
    require(tables['goal_static_valid'].all(),'A fixed future goal lacks native attributes')
    shuffled,donors,counts=shuffle_training_goals(real,a['main_group_id'],a['patch_index'],a['fold'],a['game_index'],
                                                 a['timestamp_ms'],history_data.games,history_data.episodes,budget)
    selected=np.array(history_data.arrays['selected_state_indices'],copy=True)
    require(len(selected)==3350,'Original selected population changed')
    arrays={'real_goal_id':real,'shuffled_goal_id':shuffled,'goal_donor_state_index':donors,
            'selected_state_indices':selected,**tables}
    for path,want in base_reader.bindings.items():bind(path,want,bindings)
    revalidate(bindings);history_data.revalidate();budget()
    output.mkdir(parents=True,exist_ok=False);specs={}
    for name,value in arrays.items():
        path=output/(name+'.npy');np.save(path,value,allow_pickle=False);specs[name]={'file':path.name,'dtype':str(value.dtype),'shape':list(value.shape)}
    write_json(output/'goal_vocabulary.json',goal_vocabulary)
    files={spec['file']:file_hash(output/spec['file']) for spec in specs.values()};files['goal_vocabulary.json']=file_hash(output/'goal_vocabulary.json')
    revalidate(bindings);history_data.revalidate();budget()
    manifest={'schema':SCHEMA,'state':'complete','training_authorized':True,'smoke':False,
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'arrays':specs,'files':files,
        'state_count':392448,'game_count':4001,'episode_count':33212,'selected_count':3350,'main_class_count':9333,'goal_class_count':258,
        'goal_vocabulary_file':'goal_vocabulary.json','goal_scope':'strictly_later','horizon_ms':360000,
        'diagnostic_to_goal_mapping':'diagnostic0/1->EMPTY0; diagnostic2..258->goal1..257; exact nonempty countedmultisets unchanged',
        'shuffle_scope':['main_group_id','patch_index','goal_is_EMPTY'],
        'shuffle_scope_description':'TRAIN rows only; exact(current MAIN action id, patch, EMPTY/nonempty), another TRAIN game; whole goal label copied',
        'shuffle_salt':DONOR_PREFIX,'shuffle_rank_rule':'fullSHA bigendian modulo eligible sorted global state indices; recipient identity heldout|mid|pid|ordinal|ts',
        'missing_donor_rule':'keep original auxiliary goal on TRAIN rows; heldout shuffled/donor sentinel-1',
        'shuffle_counts':counts,'main_native_invalid_class_counts':{patch:int((~tables['main_static_valid'][pi]).sum()) for pi,patch in enumerate(PATCHES)},
        'native_invalid_main_rule':'compositional embedding zero only; free class score and full9333 softmax retained; observed native labels valid required',
        'auxiliary_labels_are_inference_inputs':False,'feature_vocabulary':'exact immutable base public-static union; no learned or goal-dependent vocabulary',
        'gates':{'passed':True,'same_original_population':True,'same_original_selection':True,'strict_later_goal_receipts':True,
                 'goal_multisets_preserved':True,'train_only_different_game_shuffle':True,'heldout_shuffle_sentinel':True,
                 'all_observed_main_native_valid':True,'all_goal_native_valid':True,'no_reward_or_outcome_inputs':True},
        'history_manifest_path':str(Path(args.history_manifest).resolve()),'history_manifest_sha256':file_hash(args.history_manifest),
        'base_manifest_path':str(Path(args.base_manifest).resolve()),'base_manifest_sha256':BASE_MANIFEST_SHA256,
        'census_report_path':str(Path(args.census_report).resolve()),'census_report_sha256':CENSUS_SHA,
        'census_audit_path':str(Path(args.census_audit).resolve()),'census_audit_sha256':file_hash(args.census_audit),
        'definition_path':str(Path(args.definition).resolve()),'definition_sha256':DEFINITION_SHA,
        'admission_path':str(base_reader.admission_path),'admission_sha256':history['admission_sha256'],
        'static_export_path':base['static_export_path'],'static_export_sha256':base['static_export_sha256'],
        'catalogue_path':base['catalogue_path'],'catalogue_sha256':base['catalogue_sha256'],
        'protocol_path':str(protocol),'protocol_sha256':file_hash(protocol),'bindings':bindings,'opaque_parent_bindings':opaque,
        'reward_arrays_opened':False,'terminal_timing_arrays_opened':False,'outcomes_analyzed':False,'raw_payload_reads':False,
        'source_code_imported':False,'source_payloads_parsed':False,'test_payload_reads':False,'confirmation_payload_reads':False,
        'training_performed':False,'elapsed_s':time.monotonic()-started,
        'limits':['Existing exposed, complete catalogue-supported FIT population only; no new observation or private-intention claim.',
                  'Future completion and censoring statuses are supervision only; main inference receives current state and public static tables.',
                  'GoalEMPTY collapses both parent NONE statuses; future terminal horizon reason is not a model input.',
                  'Missing donor keeps original goal, so report actual changed-label fraction when interpreting the shuffle control.']}
    write_json(output/'manifest.json',manifest);return manifest
