"""Frozen public goal definition and descriptive future-completion census.

No model, policy scoring, raw/source imports or win rewards. Definition
publication reads only metadata/public static objects. A separate reviewed
census reads existing FIT action IDs and terminal timing, never reward.npy.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from . import admission
from .full_action_catalogue import validate_catalogue
from .iql_dataset_contract import current_reservations,file_hash,reject_quarantined

BASE_SHA='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea'
STATIC_SHA='5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0'
CATALOGUE_SHA='14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8'
GOAL_IDS_SHA='75616f9251474eeb712026289dbf151cb8e7e1936662409ad42e668e1f459e15'
HORIZON_MS=360000
PATCHES=('16.17','16.18','16.19')
MODES=('including_current','strictly_later')


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')


def bound(path,bindings):
    path=Path(path).resolve(strict=True);reject_quarantined(path);digest=file_hash(path)
    require(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting bound artifact digest: '+str(path))
    require(path.with_suffix('.sha256').read_text().split()==[digest,path.name],'Artifact sidecar mismatch: '+str(path))
    bindings[str(path)]=digest;return read(path)


def revalidate(bindings):
    for path,digest in bindings.items():
        reject_quarantined(path)
        require(file_hash(path)==digest,'Bound input/code changed: '+str(path))


def code_bindings():
    root=Path(__file__).resolve().parents[1]
    names=('winprob/goal_census_v2.py','scripts/prepare_goal_census_v2.py','winprob/admission.py',
           'winprob/full_action_catalogue.py','winprob/iql_dataset_contract.py','winprob/trajectory_contract.py','winprob/preference_reference.py')
    return {str((root/name).resolve()):file_hash(root/name) for name in names}


def goal_ids(data,catalogue):
    """Exact paid nonboot completed-goal rule; validate the entire graph."""
    items=data['items'];supported={i for basket in catalogue['full_actions'] for i,count in basket};memo={}
    def boots(i,stack=()):
        if i in stack:raise ValueError('recipe cycle')
        if i in memo:return memo[i]
        if str(i) not in items:raise ValueError('missing recipe ancestor: '+str(i))
        item=items[str(i)];parents=[int(j) for j in item.get('from',[])]
        inherited=[boots(j,stack+(i,)) for j in parents]
        memo[i]='Boots' in item.get('tags',[]) or any(inherited)
        return memo[i]
    for i in map(int,items):boots(i)
    result=[]
    for sid,item in items.items():
        i=int(sid);gold=item.get('gold',{});parents=item.get('from',[])
        if (i in supported and item.get('maps',{}).get('11') is True and gold.get('purchasable') is True
                and gold.get('base',0)>0 and parents and (item.get('depth',0)>=3 or not item.get('into'))
                and not set(item.get('tags',[]))&{'Consumable','Trinket','GoldPer'} and not memo[i]):result.append(i)
    return sorted(result)


def goal_multiset(basket,qualified):
    result=[];seen=set()
    for item,count in basket:
        require(type(item) is int and type(count) is int and item>0 and count>0 and item not in seen,'Invalid complete basket')
        seen.add(item)
        if item in qualified:result.append((item,count))
    return tuple(sorted(result))


def label_vocabulary(catalogue,by_patch):
    multisets=set()
    for patch in PATCHES:
        qualified=set(by_patch[patch])
        for basket in catalogue['full_actions']:
            key=goal_multiset(basket,qualified)
            if key:multisets.add(key)
    labels=[{'type':'NONE_FULL_HORIZON','goal_counts':[]},{'type':'NONE_TERMINAL_HORIZON','goal_counts':[]}]
    labels.extend({'type':'SINGLE' if sum(count for item,count in key)==1 else 'MULTIPLE',
                   'goal_counts':[list(pair) for pair in key]} for key in sorted(multisets))
    return labels


def input_metadata(base_manifest,history_manifest,bindings):
    base_path=Path(base_manifest).resolve(strict=True);history_path=Path(history_manifest).resolve(strict=True)
    base=bound(base_path,bindings);history=bound(history_path,bindings)
    require(file_hash(base_path)==BASE_SHA and base['state']=='complete' and base['schema']=='iql-dataset-v1'
            and (base['state_count'],base['episode_count'],base['game_count'])==(392448,33212,4001),'Wrong frozen FIT population')
    require(history['schema']=='history-probe-dataset-v1' and history['state']=='complete'
            and history['cache_manifest_sha256']==BASE_SHA and Path(history['cache_manifest_path']).resolve()==base_path,
            'History closure sidecar belongs to another base')
    ad=bound(base['source_admission_path'],bindings)
    require(file_hash(base['source_admission_path'])==base['source_admission_sha256']==history['admission_sha256'],'Admission identity changed')
    forbidden=current_reservations(ad);admitted={row['match_id']:row for row in ad['games']}
    games=base['game_metadata'];require(len({g['match_id'] for g in games})==4001,'Duplicate FIT games')
    for game in games:
        entry=admitted.get(game['match_id'])
        require(game['match_id'] not in forbidden and entry is not None and entry['role']=='train'
                and entry['historical_validation'] is False and entry['patch']==game['patch']
                and game['patch'] in ('16.17','16.18'),'Census contains non-FIT/reserved identity')
    # Definition-time bindings are metadata/code only. In particular no
    # action, history-closure, terminated, elapsed or reward array is opened.
    for path,digest in {**base['bindings'],**history['bindings']}.items():
        path=Path(path).resolve(strict=True)
        if path.suffix.lower() in ('.json','.py','.sha256'):
            reject_quarantined(path)
            require(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting source metadata digest: '+str(path))
            require(file_hash(path)==digest,'Bound source metadata/code changed');bindings[str(path)]=digest
    for key in ('catalogue','static_export'):
        require(base[key+'_sha256']==history[key+'_sha256'],'Static/catalogue lineage differs')
    return base,history,ad


def prepare_definition(base_manifest,history_manifest,output):
    bindings=code_bindings();output=Path(output).resolve();require(not output.exists(),'Refuse definition overwrite')
    base,history,ad=input_metadata(base_manifest,history_manifest,bindings)
    static=bound(base['static_export_path'],bindings);catalogue=bound(base['catalogue_path'],bindings)
    require(file_hash(base['static_export_path'])==STATIC_SHA and static['schema']=='winprob-boots-static-v1'
            and file_hash(base['catalogue_path'])==CATALOGUE_SHA,'Public static/catalogue identity changed')
    validate_catalogue(catalogue)
    for values in static['patches'].values():
        for source in values['source_files']:
            require(file_hash(source['path'])==source['sha256'],'Native public static changed');bindings[source['path']]=source['sha256']
    by_patch={patch:goal_ids(static['patches'][patch],catalogue) for patch in PATCHES}
    for patch,ids in by_patch.items():
        require(static['patches'][patch]['version']==patch+'.1' and len(ids)==106
                and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==GOAL_IDS_SHA,
                'Exact106 public-static goal definition changed')
    vocabulary=label_vocabulary(catalogue,by_patch)
    result={'schema':'goal-label-definition-v1','state':'definition_ready','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'base_manifest_path':str(Path(base_manifest).resolve()),'base_manifest_sha256':BASE_SHA,
            'history_manifest_path':str(Path(history_manifest).resolve()),'history_manifest_sha256':file_hash(history_manifest),
            'static_export_path':base['static_export_path'],'static_export_sha256':STATIC_SHA,
            'catalogue_path':base['catalogue_path'],'catalogue_sha256':CATALOGUE_SHA,
            'admission_path':base['source_admission_path'],'admission_sha256':base['source_admission_sha256'],
            'goal_item_ids_by_patch':by_patch,'goal_item_count_per_patch':106,'goal_item_ids_sha256':GOAL_IDS_SHA,
            'goal_rule':'Supported by fixed full-action catalogue; map11=true; purchasable=true; baseprice>0; nonemptyfrom; depth>=3 OR nointo; noConsumable/Trinket/GoldPer; noBoots tag onselfor anyrecursivefromancestor; entiregraph missing/cycle fails.',
            'label_vocabulary':vocabulary,'vocabulary_basis':'All goal intersections of the fixed full-action catalogue, before any future labels are read.',
            'horizon_ms':HORIZON_MS,'modes':list(MODES),'qualifying_visit':'First chronological real purchase node with at leastone goal copy and visit_end_ms<=current_start+360000.',
            'including_current':'Search nodes atcurrent ordinal orlater, including current only after its visit closure within horizon.',
            'strictly_later':'Search realpurchase nodes with ordinal>current; no syntheticSAVE goal purchase.',
            'NONE_rule':'Search qualifying visits first; ifnone and absorbing_terminal_ms<=horizon then NONE_TERMINAL_HORIZON, else NONE_FULL_HORIZON.',
            'MULTIPLE_rule':'Preserve the entire sorted counted qualifying subset from the firstvisit.2+totalcopies isMULTIPLE; no min-ID selection or merging later visits.',
            'terminal_time_rule':'timestamp_ms[last]+elapsed_ms[last], require exactlylast terminated=True; reward.npy never opened.',
            'conditional_entropy':'Raw plug-in natural-log H(exactgoal_multiset | currentmainbasket,patch), with one unified EMPTY goal identity for both NONEstatuses; descriptive only; state-weighted.',
            'NONE_identity':'NONE_FULL_HORIZON and NONE_TERMINAL_HORIZON are separate diagnosticstatuses but the SAME EMPTY identity forentropy/variation.',
            'current_completion_coarsening':'Qualifying currentvisit goal is a deterministic countedsubset of(currentmainbasket,nativepatch); report its prevalence and latergoals differing fromthis subset.',
            'population':{'states':392448,'episodes':33212,'games':4001,'role':'original admittedFIT','selection_or_replacements':'none'},
            'bindings_sha256':bindings,'future_action_payloads_read':False,'terminal_timing_arrays_read':False,'reward_array_read':False,
            'training_performed':False,'policy_accuracy_computed':False,'raw_or_source_database_read':False,
            'interpretation':'Observed completion supervision, not verified private intent or additional observed information; any later benefit would be learned regularization.',
            'limits':['Same retrospectively qualified complete, catalogue-supported FIT population; not a representative shop census.',
                      'No component-only subset, winner/reward conditioning, model score or hidden confirmation payload.',
                      'NONE terminal category uses future end timing as a label only, never an observed input.']}
    revalidate(bindings);output.mkdir(parents=True,exist_ok=False);write_json(output/'definition.json',result);return result


def assign_episode_labels(timestamps,ends,goal_codes,terminated,elapsed,horizon_ms=HORIZON_MS):
    n=len(timestamps)
    require(n>0 and all(np.shape(value)==(n,) for value in (ends,goal_codes,terminated,elapsed)),'Unaligned episode arrays')
    require(np.all(np.diff(timestamps)>0) and np.all(ends>=timestamps) and np.all(ends[:-1]<=timestamps[1:]),'Incomplete/overlapping episode')
    require(terminated.dtype==np.bool_ and int(terminated.sum())==1 and bool(terminated[-1]) and np.all(elapsed>0)
            and np.array_equal(elapsed[:-1],np.diff(timestamps)),'Absorbing terminal timing contract failed')
    terminal=int(timestamps[-1])+int(elapsed[-1]);require(np.all(ends<=terminal),'Visit closes after terminal')
    horizons=timestamps+horizon_ms;none=np.where(terminal<=horizons,1,0).astype(np.int32)
    qualified=np.flatnonzero(goal_codes>=2);result={}
    for mode,side in (('including_current','left'),('strictly_later','right')):
        positions=np.searchsorted(qualified,np.arange(n),side=side);has=positions<len(qualified)
        candidate=np.full(n,-1,dtype=np.int64);candidate[has]=qualified[positions[has]]
        valid=has.copy();valid[has]=ends[candidate[has]]<=horizons[has]
        labels=none.copy();labels[valid]=goal_codes[candidate[valid]];candidate[~valid]=-1
        result[mode]=(labels,candidate)
    return result,terminal


def summarize_mode(labels,selected_visit,main,patch,game,vocabulary,current_goal_codes=None):
    n=len(labels);row_counts=Counter();game_sets=defaultdict(set);class_counts=Counter();class_games=defaultdict(set)
    pair=Counter();joint=Counter();pair_games=defaultdict(set);pair_labels=defaultdict(set)
    triple=Counter();pair_game=Counter()
    per_item=Counter();item_games=defaultdict(set)
    identity_lookup={};identities=[]
    for entry in vocabulary:
        key=tuple(map(tuple,entry['goal_counts']))
        if key not in identity_lookup:identity_lookup[key]=len(identity_lookup)
        identities.append(identity_lookup[key])
    for index in range(n):
        label=int(labels[index]);entry=vocabulary[label];gid=int(game[index]);key=(int(main[index]),int(patch[index]))
        origin='none' if selected_visit[index]<0 else 'current_completion' if selected_visit[index]==index else 'later_completion'
        for category in (origin,entry['type']):row_counts[category]+=1;game_sets[category].add(gid)
        identity=identities[label]
        class_counts[label]+=1;class_games[label].add(gid);pair[key]+=1;joint[(key,identity)]+=1
        pair_games[key].add(gid);pair_labels[key].add(identity);triple[(key,identity,gid)]+=1;pair_game[(key,gid)]+=1
        for item,count in entry['goal_counts']:per_item[item]+=count;item_games[item].add(gid)
    entropy=-sum(count/n*math.log(count/pair[key]) for (key,label),count in joint.items())
    varied={key for key,values in pair_labels.items() if len(values)>1}
    across={key for key in varied if len(pair_games[key])>=2}
    total_cross=sum(count*count for count in pair.values())-sum(count*count for count in pair_game.values())
    same_cross=sum(count*count for count in joint.values())-sum(count*count for count in triple.values())
    rare={'n1':sum(count==1 for count in pair.values()),'n2_to4':sum(2<=count<=4 for count in pair.values()),
          'n5_to9':sum(5<=count<=9 for count in pair.values()),'n_less10':sum(count<10 for count in pair.values()),
          'single_game_groups':sum(len(games)==1 for games in pair_games.values())}
    result={'states':n,'games':len(set(map(int,game))),'state_counts':dict(row_counts),
        'unique_game_counts':{key:len(value) for key,value in game_sets.items()},
        'goal_label_distribution':[{'label_id':label,**vocabulary[label],'states':count,'games':len(class_games[label])} for label,count in sorted(class_counts.items())],
        'goal_item_copy_distribution':[{'item_id':item,'goal_copies_across_state_labels':count,'games':len(item_games[item])} for item,count in sorted(per_item.items())],
        'conditional_variation':{'conditioning':'(currentmainbasket, nativepatch)','observed_conditioning_groups':len(pair),
            'goal_identity':'exact countedmultiset; EMPTY unifies both NONE diagnosticstatuses',
            'groups_with_multiple_goal_labels':len(varied),'states_in_varying_groups':sum(pair[key] for key in varied),
            'groups_with_variation_across_atleast2games':len(across),'states_in_crossgame_varying_groups':sum(pair[key] for key in across),
            'distinct_game_ordered_state_pairs':total_cross,'different_goal_distinct_game_ordered_state_pairs':total_cross-same_cross,
            'crossgame_goal_disagreement_fraction':(total_cross-same_cross)/total_cross if total_cross else None,
            'plugin_H_G_given_A_patch_nats':entropy,'plugin_H_G_given_A_patch_bits':entropy/math.log(2),'rare_group_counts':rare,
            'entropy_limit':'Finite-sample raw plug-in entropy, state-weighted and generally downward-biased; repeated states/games are not independent; not an information gain or policy score.'}}
    if current_goal_codes is not None:
        require(np.shape(current_goal_codes)==(n,),'Current goal coarsening not aligned')
        current_identity=np.where(current_goal_codes>=2,current_goal_codes,0)
        future_identity=np.where(labels>=2,labels,0)
        current=selected_visit==np.arange(n);later=selected_visit>np.arange(n);goals=labels>=2
        require(np.array_equal(labels[current],current_goal_codes[current]),'Currentcompletion is not exact currentaction coarsening')
        result['current_action_coarsening']={
            'current_actions_containing_qualified_goals':int((current_goal_codes>=2).sum()),
            'labels_taken_from_current_completion':int(current.sum()),
            'fraction_all_labels_current_completion':float(current.mean()),
            'fraction_nonempty_labels_current_completion':float(current.sum()/goals.sum()) if goals.any() else None,
            'later_completion_labels':int(later.sum()),
            'later_goal_different_from_current_goal_subset':int((later&(future_identity!=current_identity)).sum()),
            'later_goal_different_when_current_action_already_has_goal':int((later&(current_goal_codes>=2)&(future_identity!=current_identity)).sum()),
            'later_goal_same_when_current_action_already_has_goal':int((later&(current_goal_codes>=2)&(future_identity==current_identity)).sum()),
            'meaning':'Descriptive observed-label relationships; currentcompletion is a coarsening of the action target, not extra observed information.'}
    return result


def census(definition_path,output,*,deadline_seconds=300,guard=lambda:None):
    started=time.monotonic();require(1<=deadline_seconds<=300,'Census ceiling300seconds')
    output=Path(output).resolve();require(not output.exists(),'Refuse census overwrite')
    def budget():guard();require(time.monotonic()-started<=deadline_seconds,'Census deadline exceeded')
    bindings=code_bindings();definition=bound(definition_path,bindings)
    require(definition['schema']=='goal-label-definition-v1' and definition['state']=='definition_ready'
            and definition['horizon_ms']==HORIZON_MS and definition['goal_item_ids_sha256']==GOAL_IDS_SHA,'Unreviewed definition')
    for path,digest in definition['bindings_sha256'].items():
        require(path not in bindings or bindings[path]==digest,'Definition code changed before labels');bindings[path]=digest
    revalidate(bindings);budget()
    base,history,ad=input_metadata(definition['base_manifest_path'],definition['history_manifest_path'],bindings)
    for prefix in ('base_manifest','history_manifest','static_export','catalogue','admission'):
        path=Path(definition[prefix+'_path']).resolve(strict=True)
        reject_quarantined(path)
        require(file_hash(path)==definition[prefix+'_sha256'], 'Published definition input changed: '+prefix)
        require(str(path) in bindings and bindings[str(path)]==definition[prefix+'_sha256'], 'Unbound definition input: '+prefix)
    base_dir=Path(definition['base_manifest_path']).parent;hist_dir=Path(definition['history_manifest_path']).parent
    catalogue=bound(definition['catalogue_path'],bindings);vocabulary=label_vocabulary(catalogue,definition['goal_item_ids_by_patch'])
    require(vocabulary==definition['label_vocabulary'],'Pre-published static goal vocabulary changed')
    lookup={tuple(map(tuple,row['goal_counts'])):i for i,row in enumerate(vocabulary) if i>=2}
    mapping=np.full((3,len(catalogue['full_actions'])),-1,dtype=np.int32)
    for pi,patch in enumerate(PATCHES):
        for action,basket in enumerate(catalogue['full_actions']):
            key=goal_multiset(basket,set(definition['goal_item_ids_by_patch'][patch]))
            if key:mapping[pi,action]=lookup[key]
    def array(meta,directory,name):
        require(name!='reward','Win reward must never be opened')
        spec=meta['arrays'][name];path=(directory/spec['file']).resolve(strict=True)
        reject_quarantined(path)
        require(str(path) not in bindings or bindings[str(path)]==meta['files'][spec['file']], 'Conflicting census array digest')
        require(path.is_relative_to(directory.resolve()) and file_hash(path)==meta['files'][spec['file']],'Consumed census array changed')
        bindings[str(path)]=meta['files'][spec['file']];value=np.load(path,mmap_mode='r',allow_pickle=False)
        require(str(value.dtype)==spec['dtype'] and list(value.shape)==spec['shape'],'Array schema differs')
        return value
    # Definition has already been published and verified. This is the first
    # future-action/terminal-timing access in the new workflow.
    names=('timestamp_ms','full_action_id','main_group_id','patch_index','game_index','episode_index','terminated','elapsed_ms')
    a={name:array(base,base_dir,name) for name in names};ends=array(history,hist_dir,'visit_end_ms');n=len(ends)
    ep_path=base_dir/base['episode_metadata_file'];require(file_hash(ep_path)==base['files'][ep_path.name],'Episode metadata changed')
    bindings[str(ep_path)]=base['files'][ep_path.name];episodes=read(ep_path)
    require(n==392448 and len(episodes)==33212,'Census population changed')
    require(np.all((a['full_action_id']>=0)&(a['full_action_id']<len(catalogue['full_actions'])))
            and np.array_equal(a['main_group_id'],np.asarray(catalogue['full_to_group'])[a['full_action_id']]),'Complete/main target mapping changed')
    outputs={mode:{'label_id':np.empty(n,dtype=np.int32),'qualifying_state_index':np.full(n,-1,dtype=np.int64)} for mode in MODES}
    cursor=0
    for ei,episode in enumerate(episodes):
        if ei%250==0:budget()
        lo=episode['first_state'];hi=lo+episode['states'];game=base['game_metadata'][episode['game_index']]
        require(lo==cursor and hi>lo and episode['match_id']==game['match_id'] and episode['patch']==game['patch']
                and np.all(a['game_index'][lo:hi]==episode['game_index']) and np.all(a['episode_index'][lo:hi]==ei)
                and np.all(a['patch_index'][lo:hi]==PATCHES.index(game['patch'])),'Episode crosses a game/patch')
        codes=mapping[a['patch_index'][lo:hi],a['full_action_id'][lo:hi]]
        result,terminal=assign_episode_labels(a['timestamp_ms'][lo:hi],ends[lo:hi],codes,a['terminated'][lo:hi],a['elapsed_ms'][lo:hi])
        for mode,(labels,visits) in result.items():
            outputs[mode]['label_id'][lo:hi]=labels
            outputs[mode]['qualifying_state_index'][lo:hi]=np.where(visits<0,-1,visits+lo)
        cursor=hi
    require(cursor==n,'Census did not cover every original state')
    budget();summaries={};current_codes=mapping[a['patch_index'],a['full_action_id']]
    main_mapping=np.full((3,len(catalogue['group_main'])),-1,dtype=np.int32)
    for pi,patch in enumerate(PATCHES):
        qualified=set(definition['goal_item_ids_by_patch'][patch])
        for gid,basket in enumerate(catalogue['group_main']):
            key=goal_multiset(basket,qualified)
            if key:main_mapping[pi,gid]=lookup[key]
    require(np.array_equal(current_codes,main_mapping[a['patch_index'],a['main_group_id']]),
            'Current goal subset is not a deterministic mainbasket/nativepatch coarsening')
    for mode in MODES:
        summaries[mode]=summarize_mode(outputs[mode]['label_id'],outputs[mode]['qualifying_state_index'],a['main_group_id'],a['patch_index'],a['game_index'],vocabulary,current_codes)
        budget()
    require(summaries['strictly_later']['state_counts'].get('current_completion',0)==0,'Strict later label included current visit')
    revalidate(bindings);require(not {g['match_id'] for g in base['game_metadata']}&current_reservations(ad),'Reservation changed before publication');budget()
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'labels.npz',**{mode+'_'+name:value for mode,parts in outputs.items() for name,value in parts.items()})
    write_json(output/'label_vocabulary.json',vocabulary)
    result={'schema':'goal-completion-census-v1','state':'complete','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'definition_path':str(Path(definition_path).resolve()),'definition_sha256':file_hash(definition_path),
            'population':definition['population'],'horizon_ms':HORIZON_MS,'summaries':summaries,
            'bindings_sha256':bindings,'files':{name:file_hash(output/name) for name in ('labels.npz','label_vocabulary.json')},
            'definition_published_before_future_labels':True,'legacy_source_target_id_used':False,
            'reward_array_read':False,'terminal_timing_arrays_read':['terminated.npy','elapsed_ms.npy'],
            'raw_payloads_or_source_database_read':False,'source_code_imported':False,'test_or_confirmation_payload_reads':False,
            'policy_accuracy_computed':False,'model_loaded_or_trained':False,'new_observed_information_claim':False,
            'elapsed_s':time.monotonic()-started,
            'limits':definition['limits']+['These are observed completed-purchase labels, not identified intentions.',
                'Variation/entropy is descriptive; current action and goal supervision share future purchases, especially in the including-current label.',
                'No model, policy ranking, accuracy improvement, causal value or practical benefit is tested here.']}
    write_json(output/'report.json',result);return result
