"""Recovered inputs reach training and both displayed policies without leakage."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from footwear_runtime import activate
activate()
import torch
import train_prefix as tp
from app.ddragon import dragon_for_patch
from app.footwear_policy import FOOTWEAR_STATE_VERSION as VERSION
from app.predictor import Predictor
from app.conditional_policy import purchase_sequence
from eval_policy import check_displayed_purchases,serving_row
import eval_policy
from footwear_runtime import declaration

dragon=dragon_for_patch('16.17')

def row(inventory,budget,granted):
    return {'match_id':'SYNTHETIC','participant_id':1,'champion':'Ashe','champion_id':22,'team_id':100,
            'role':'MIDDLE','inventory':inventory,'gold':budget,'gold_est':budget,'gold_est_version':'prequential-v3',
            'slot_state_version':'role-slots-v1','bot_quest_complete':False,'unmodeled_regular_slots':0,
            'footwear_state_version':VERSION,'magical_footwear_selected':True,'footwear_granted':granted,
            'others':[],'label_counts':{},'target_encoding':'item-counts-v1','ts':100000,'level':8}

def scored(proposals):
    p=Predictor.__new__(Predictor)
    p.config={'max_basket_items':8,'footwear_state_version':VERSION}
    p.dragon=dragon;p.threshold=.8;p.label_ids=[i for i,n in proposals]
    p.label_index={item:i for i,item in enumerate(p.label_ids)}
    p._standalone=set(p.label_ids);p._final_components={}
    scores=torch.tensor([.99-i*.005 for i in range(len(proposals))])
    copies=torch.tensor([n for i,n in proposals])
    p._probs=lambda _row:(scores,scores,copies,None)
    p._target_probs=lambda _row:torch.zeros(len(proposals))
    return p

early=row([],500,False)
late=row([2422],800,True)
for state,proposals,expected,cost in ((early,[(1001,1),(1036,1)],[1036],350),(late,[(3020,1)],[3020],800)):
    before=deepcopy(state);p=scored(proposals)
    options=p.predict_options(state,n_options=1,budget_slack=0)
    assert Counter(i['item_id'] for i in options[0]['items'])==Counter(expected)
    assert sum(i['cost'] for i in options[0]['items'])==cost
    assert not check_displayed_purchases(options,state,state['gold'],dragon)
    assert Counter(i['item_id'] for i in p.predict(state,max_items=8,budget_slack=0)['basket'])==Counter(expected)
    assert state==before
assert purchase_sequence((1001,),early,500,dragon) is None
assert purchase_sequence((3020,),late,800,dragon)==[{'item_id':3020,'cost':800}]
# A dedicated support ward slot permits seven retained copies. The artifact's
# declared basket capacity, rather than a historical six-entry cap, controls
# the shown option; actual physical capacity is still independently replayed.
large={**row([],5000,False),'role':'UTILITY'}
large_proposals=[(1036,4),(1028,2),(2055,1)]
large_options=scored(large_proposals).predict_options(large,n_options=1,budget_slack=0)
assert Counter(i['item_id'] for i in large_options[0]['items'])==Counter(dict(large_proposals))
assert not check_displayed_purchases(large_options,large,5000,dragon)
bad={**early};bad.pop('footwear_granted')
try: scored([(1001,1)]).predict_options(bad,n_options=1,budget_slack=0)
except ValueError: pass
else: raise AssertionError('Missing footwear state accepted')

tp.apply_config({'gold_x':True,'gold_est':True,'extras':False,'runes':False,'query_dim':tp.BASE_QUERY_DIM,
                 'max_items':8,'max_copies':4,'inventory_encoding':'canonical-multiset-v1'})
labels={1001:0,1036:1,3020:2}
items={1001:1,1036:2,3020:3,2422:4}
dataset=tp.ShopDataset([early,late],{22:1},items,labels,dragon)
assert not bool(dataset.legal[0,0]) and not bool(dataset.legal[0,2])
assert bool(dataset.legal[0,1]) and bool(dataset.legal[1,2])
try: tp.ShopDataset([{**early,'label_counts':{'1001':1}}],{22:1},items,labels,dragon)
except ValueError: pass
else: raise AssertionError('Forced-positive mask admitted pregrant boots')

sample={**late,'label_counts':{'1036':1},'source_target_id':3153,'source_target_version':'full-source-completions-v1'}
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'visits_val.jsonl'
    path.write_text(json.dumps(sample)+'\n',encoding='utf-8')
    streamed=next(tp.stream_rows(path))
    assert streamed['footwear_granted'] is True and streamed['source_target_id']==3153
    target=next(tp.with_targets([streamed],dragon))
    assert target['target_id']==3153,'Masked source completion was replaced'
    clean=serving_row(target,800)
    assert 'source_target_id' not in clean and 'source_target_version' not in clean and 'target_id' not in clean
    assert clean['footwear_granted'] is True
    inputs_a=tp.ShopDataset([target],{22:1},items,{**labels,3153:3,3031:4},dragon)
    inputs_b=tp.ShopDataset([{**target,'source_target_id':3031,'target_id':3031}],{22:1},items,{**labels,3153:3,3031:4},dragon)
    for name in ('champs','sides','items','hist','query','inv','legal','budget'):
        assert torch.equal(getattr(inputs_a,name),getattr(inputs_b,name)),name
    assert int(inputs_a.target[0])!=int(inputs_b.target[0])
for bad_source in ({**late}, {**sample,'source_target_id':1036}, {**sample,'source_target_version':None}):
    try: list(tp.with_targets([bad_source],dragon))
    except ValueError: pass
    else: raise AssertionError('Recovered plan supervision silently fell back or accepted a component target')
# A model's claimed provenance cannot conceal different runtime/capacity
# settings in the config actually used by Predictor.
manifest={'export_fingerprint':'synthetic','eval_version':1,'footwear_runtime':declaration(),
          'input_schema':{'max_modeled_inventory_items':8,'max_copies_of_one_item':4,'max_basket_items':7}}
config={'gold_x':True,'gold_input':'prequential-v3','footwear_runtime':declaration(),
        'footwear_state_version':VERSION,'max_items':8,'max_copies':4,'max_basket_items':7}
provenance={'manifest_sha256':'bound','export_fingerprint':'synthetic','eval_version':1,
            'footwear_runtime':declaration(),'footwear_state_version':VERSION,
            'source_target_version':'full-source-completions-v1','gold_input':'prequential-v3',
            'training_split':'train','selection_split':'validation'}
predictor=SimpleNamespace(config=config,provenance=provenance,artifact=Path('synthetic.pt'))
with patch.object(eval_policy,'manifest_sha256',return_value='bound'):
    assert eval_policy.verify_artifact_provenance(predictor,manifest,False)['status']=='verified'
    for key,value in (('max_basket_items',6),('max_items',6),('footwear_runtime',None)):
        with patch.dict(config,{key:value}):
            try: eval_policy.verify_artifact_provenance(predictor,manifest,False)
            except SystemExit: pass
            else: raise AssertionError('Artifact config disagreed with frozen recovered schema')
print('ok recovered policy: both decoders, baseline, independent price replay, pregrant training masks, streamed state and source-only future labels')
