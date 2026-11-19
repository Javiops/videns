"""Outcome-blind recorded component initiation/consumption census."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .history_probe_data import HistoryProbeData,write_json
from .iql_dataset_contract import file_hash,reject_quarantined
from .full_action_catalogue import load_catalogue

PAIR=(1029,1033)
FINISH={1029:3047,1033:3111}
WARDS={2055,772043}


def require(value,message):
    if not value:raise ValueError(message)


def ancestors(items,item,trail=()):
    require(item not in trail,'Cyclic native recipe')
    require(str(item) in items,'Missing native recipe item')
    result=set()
    for child in items[str(item)].get('from',[]):
        child=int(child);result.add(child);result.update(ancestors(items,child,(*trail,item)))
    return result


def money_status(estimate,cost):
    if estimate is None or cost is None or not np.isfinite(estimate):return 'unknown'
    if abs(estimate-cost)<=.001:return 'numerical_boundary'
    return 'estimated_sufficient' if estimate>cost else 'estimated_insufficient'


def completion_costs(inventory,items):
    if inventory.get(2422,0):return None
    return {str(boot):float(items[str(boot)]['gold']['total'])-
            sum(float(items[str(child)]['gold']['total'])*min(1,inventory.get(child,0))
                for child in (1001,component)) for component,boot in FINISH.items()}


def analyze_episode(nodes,tier2,recipes,items):
    """Nodes are public predecision inventories + retained labels; no outcomes."""
    first_boot=None;initiation=None;prior_boot=False
    for i,node in enumerate(nodes):
        basket=node['basket'];inventory=node['inventory']
        boots=[item for item,count in basket.items() if item in tier2 for _ in range(count)]
        if boots and first_boot is None:
            costs=completion_costs(inventory,items)
            first_boot={'index':i,'items':boots,'mixed_main':any(item not in tier2|WARDS for item in basket),
                        'remaining_static_costs':costs,'estimated_money':money_status(node['gold'],max(costs.values()) if costs else None)}
        bought=[item for item,count in basket.items() if item in PAIR for _ in range(count)]
        if initiation is None and bought and not prior_boot and not any(inventory.get(item,0) for item in tier2):
            status=('preowned_component' if any(inventory.get(item,0) for item in PAIR) else
                    'both_components' if len(set(bought))>1 else 'duplicate_component' if len(bought)!=1 else
                    'same_visit_boot' if boots else 'neutral_single')
            component=bought[0];pure=all(item in {component}|WARDS for item in basket)
            ward_cost=sum(items[str(item)]['gold']['total']*count for item,count in basket.items() if item in WARDS)
            initiation={'index':i,'component':component,'status':status,'pure_main':pure,
                        'estimated_pair_money':money_status(node['gold'],400+ward_cost if pure else None)}
        if boots or any(inventory.get(item,0) for item in tier2):prior_boot=True
    record={'first_boot':first_boot,'initiation':initiation,'trace':None}
    if initiation is None or initiation['status']!='neutral_single':return record
    start=initiation['index'];component=initiation['component'];trace={'status':'no_observed_consumption'}
    record['trace']=trace
    if start+1==len(nodes):trace['status']='no_following_state';return record
    if nodes[start]['end']>=nodes[start+1]['time']:trace['status']='initiation_closure_overlap';return record
    for j in range(start+1,len(nodes)):
        node=nodes[j];held=node['inventory'].get(component,0)
        if held!=1:
            trace['status']='component_absent' if held==0 else 'component_multiplicity';return record
        if node['basket'].get(component,0):trace['status']='component_reacquisition';return record
        consumers=[item for item,count in node['basket'].items() if component in recipes[item] for _ in range(count)]
        if j+1==len(nodes):
            if consumers:trace['status']='consumer_without_following_state'
            return record
        if node['end']>=nodes[j+1]['time']:trace['status']='later_closure_overlap';return record
        after=nodes[j+1]['inventory'].get(component,0)
        if after==1:continue
        if after>1:trace['status']='component_multiplicity';return record
        if len(consumers)!=1:
            trace['status']='multiple_compatible_consumers' if consumers else 'disappeared_without_recipe_receipt';return record
        item=consumers[0]
        trace.update(status='matching_defensive_boot' if item==FINISH[component] else 'other_item',
                     consumer_item=item,consumer_index=j,elapsed_ms=node['time']-nodes[start]['time'])
        return record
    return record


def summarize(records):
    counts=Counter();by_action=Counter();traces=Counter();money=Counter();paired_money=Counter();durations=[]
    trace_actions=Counter();trace_games=defaultdict(set);action_games=defaultdict(set)
    for row in records:
        counts['episodes']+=1;boot=row['first_boot'];init=row['initiation'];trace=row['trace']
        if boot:
            for item in boot['items']:counts['first_boot_item:'+str(item)]+=1
            counts['first_boot_present']+=1
        else:counts['no_recorded_tier2_boot']+=1
        if init is None:counts['no_initiation']+=1;continue
        counts['initiation:'+init['status']]+=1
        if init['status']!='neutral_single':continue
        counts['neutral_pure' if init['pure_main'] else 'neutral_mixed']+=1
        action_key=f"{row['fold']}|{init['component']}|{'pure' if init['pure_main'] else 'mixed'}"
        by_action[action_key]+=1;action_games[action_key].add(row['game_id'])
        money[init['estimated_pair_money']]+=1;traces[trace['status']]+=1
        trace_actions[str(init['component'])+'|'+trace['status']]+=1;trace_games[trace['status']].add(row['game_id'])
        if trace['status']=='matching_defensive_boot':
            durations.append(trace['elapsed_ms'])
            if boot and boot['index']==trace['consumer_index']:
                paired_money[init['estimated_pair_money']+'|'+boot['estimated_money']]+=1
                counts['trace_matches_first_boot']+=1
    return {'counts':dict(counts),'neutral_by_fold_component_and_purity':dict(by_action),'trace_statuses':dict(traces),
            'neutral_unique_games_by_fold_component_and_purity':{key:len(value) for key,value in action_games.items()},
            'trace_by_component':dict(trace_actions),'trace_unique_games':{key:len(value) for key,value in trace_games.items()},
            'initiation_estimated_money':dict(money),'same_trace_early_late_estimated_money':dict(paired_money),
            'unique_games':len({row['game_id'] for row in records}),
            'matching_boot_duration_ms_quantiles':np.percentile(durations,[0,25,50,75,100]).tolist() if durations else None}


def run(output,protocol,guard=lambda:None):
    started=time.monotonic();output=Path(output).resolve();protocol=Path(protocol).resolve(strict=True)
    require(not output.exists(),'Refuse immutable census overwrite');root=Path(__file__).resolve().parents[1]
    bindings={}
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);reject_quarantined(path)
        require(path.name not in ('reward.npy','terminated.npy','elapsed_ms.npy','next_state_index.npy','fold_action_counts.npy'),'Forbidden outcome/terminal dependency')
        want=file_hash(path) if digest is None else digest
        require(str(path) not in bindings or bindings[str(path)]==want,'Conflicting digest')
        require(file_hash(path)==want,'Changed bound input');bindings[str(path)]=want
    def budget():guard();require(time.monotonic()-started<=300,'Commitment census300second limit')
    for path in (Path(__file__),root/'scripts/commitment_census_v1.py',protocol):bind(path)
    data=HistoryProbeData(root/'data/runs/history-cache-20260930-v1/cache')
    require(len(data.games)==4001 and len(data.episodes)==33212,'Frozen population changed')
    for path,digest in data.bindings.items():bind(path,digest)
    base=data.base_manifest;catalogue,_=load_catalogue(base['catalogue_path']);bind(base['catalogue_path'],base['catalogue_sha256'])
    bind(base['static_export_path'],base['static_export_sha256']);static=json.loads(Path(base['static_export_path']).read_text())['patches']
    metadata=data.cache.json_member(base['feature_metadata_file']);inverse={int(index):int(item) for item,index in metadata['item_index'].items()}
    require(metadata['global_names'][2]=='own_gold_est_over_3000','Unexpected cached wallet coordinate')
    worlds={}
    for patch,world in static.items():
        items=world['items']
        for component,boot in FINISH.items():
            require(sorted(map(int,items[str(boot)]['from']))==sorted((1001,component)),'Unexpected defensive recipe')
        require(items['1029']['gold']['total']==300 and items['1033']['gold']['total']==400,'Changed component costs')
        tier2={int(item) for item in items['1001']['into'] if items[str(item)].get('maps',{}).get('11') is True and items[str(item)]['gold']['purchasable'] is True}
        recipes={item:ancestors(items,item) for item in inverse.values() if str(item) in items}
        worlds[patch]=(items,tier2,recipes)
    a={name:data.base_array(name) for name in ('full_action_id','timestamp_ms','inventory_offsets','inventory_item_indices',
                                            'inventory_item_counts','global_public')}
    ends=data.arrays['visit_end_ms'];focal={(row['match_id'],row['participant_id']) for row in data.selection['entries'] if row['status']=='selected'}
    records=[]
    for ei,episode in enumerate(data.episodes):
        if ei%128==0:budget()
        nodes=[];items,tier2,recipes=worlds[episode['patch']]
        for index in range(episode['first_state'],episode['first_state']+episode['states']):
            low=int(a['inventory_offsets'][index*10]);high=int(a['inventory_offsets'][index*10+1])
            inventory={inverse[int(item)]:int(count) for item,count in zip(a['inventory_item_indices'][low:high],a['inventory_item_counts'][low:high],strict=True)}
            action=int(a['full_action_id'][index])
            nodes.append({'time':int(a['timestamp_ms'][index]),'end':int(ends[index]),'inventory':inventory,
                          'basket':dict(catalogue['full_actions'][action]) if action else {},'gold':float(a['global_public'][index,2])*3000})
        row=analyze_episode(nodes,tier2,recipes,items)
        row.update(game_id=episode['match_id'],participant_id=episode['participant_id'],fold=episode['fold'],patch=episode['patch'],
                   focal=(episode['match_id'],episode['participant_id']) in focal,first_state=episode['first_state'])
        records.append(row)
    primary=[row for row in records if row['focal']];require(len(primary)==3350,'Focal availability changed')
    first=summarize(primary);all_rows=summarize(records);c=first['neutral_by_fold_component_and_purity']
    gates={'minimum500_neutral_pure_focal':first['counts'].get('neutral_pure',0)>=500,
           'minimum100_each_component_each_fold':all(c.get(f'{fold}|{component}|pure',0)>=100 for fold in (0,1) for component in PAIR),
           'minimum200_matching_focal_boot_traces':first['trace_statuses'].get('matching_defensive_boot',0)>=200}
    for path,digest in data.cache.bindings.items():bind(path,digest)
    data.revalidate();budget()
    for path,digest in bindings.items():require(file_hash(path)==digest,'Input changed during census')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        for row in records:stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
    result={'schema':'recorded-component-initiation-census-v1','state':'complete','stage':'counts_only',
            'primary_focal':first,'secondary_all_episodes':all_rows,'unavailable_focal_games':651,'gates':gates,
            'primary_by_patch':{patch:summarize([row for row in primary if row['patch']==patch]) for patch in sorted(static)},
            'primary_by_fold':{str(fold):summarize([row for row in primary if row['fold']==fold]) for fold in (0,1)},
            'source_validation_readiness':all(gates.values()),'bindings_sha256':bindings,'episodes_sha256':file_hash(output/'episodes.jsonl'),
            'outcomes_read':False,'models_fitted':False,'raw_or_source_payloads_parsed':False,'causal_claim':False,
            'slot_legality_established':False,'true_wallet_established':False,'elapsed_s':time.monotonic()-started,
            'limits':['Recorded component/recipe traces do not prove private intent or complete item-token lineage.',
                      'Conditioning on eventual completion would select on future information; all initiation/censoring categories are retained.',
                      'Estimated wallet comparisons and static recipe costs are diagnostic only, not historical legal-action certificates.',
                      'New4001FIT population differs from the old6000-game boot-support census; counts are not directly comparable.']}
    write_json(output/'report.json',result);return result
