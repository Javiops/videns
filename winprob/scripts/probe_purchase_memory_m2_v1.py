"""Fixed purchase-memory reranking of frozen legal M2 options on859 dev games."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from winprob import admission
from winprob.history_probe_training import HistoryDataset
from winprob.history_probe_models import HistoryProbeActor,set_history_validation
from winprob.iql_training import TensorStore,RuntimeGuard,file_hash,set_below_normal_priority
from winprob.iql_dataset_contract import current_reservations
from winprob.iql_features import PublicFeatureCompiler
from winprob.iql_models import collate_public_boards
from winprob.purchase_memory_v1 import normalized,neighbors,distribution,shuffled_labels
from winprob.ordinary_actor_export_v2 import mutate_ignored_fields,compiled_equal
from winprob.ordinary_policy_evaluation import stable_order,assert_same_candidates,target_description,recommendation_credit,option_multiset
from winprob.ordinary_evaluation_io import (validate_source,validate_actor,validate_reference,validate_native19,
    validate_targets,load_catalogue,canonical)
from probe_purchase_memory_v1 import embedding,checked_hash,write,CHECKPOINTS,CACHE,OLD

PROTOCOL=ROOT/'docs/purchase_memory_m2_protocol_v1.md'
SOURCE=ROOT.parent/'build_tracker/data/exports/winprob-ordinary/ordinary-contexts-20260930-v1/manifest.json'
REFERENCE=SOURCE.parent.parent/'ordinary-reference-20260930-v3/manifest.json'
NATIVE=SOURCE.parent.parent/'ordinary-native19-20260930-v3/manifest.json'
ACTOR=ROOT/'data/runs/ordinary-actor-20260930-v2/export/manifest.json'


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    guard=RuntimeGuard(args.deadline_seconds);set_below_normal_priority()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def status(stage,**extra):
        value={'stage':stage,'elapsed_s':time.monotonic()-guard.started,**extra}
        (args.output/'status.json').write_text(json.dumps(value),encoding='utf-8');print(json.dumps(value),flush=True)
    try:
        status('validating_frozen_inputs')
        bindings={str(p.resolve()):file_hash(p) for p in [Path(__file__),PROTOCOL,ROOT/'scripts/probe_purchase_memory_v1.py']+sorted((ROOT/'winprob').glob('*.py'))}
        source,selection,contexts,admitted=validate_source(SOURCE,bindings)
        catalogue,lookup=load_catalogue(source['catalogue_path'])
        actor,actor_rows,old_arrays=validate_actor(ACTOR,SOURCE,source,contexts,catalogue,bindings)
        reference,refs,m2=validate_reference(REFERENCE,SOURCE,ACTOR,source,contexts,actor_rows,catalogue,bindings)
        native,native_rows,receipts=validate_native19(NATIVE,SOURCE,REFERENCE,ACTOR,source,contexts,refs,catalogue,bindings)
        del old_arrays,actor_rows,actor,native_rows;gc.collect()
        if len(contexts)!=859 or any(c['status']!='available' for c in contexts):raise ValueError('frozen859 available context ledger changed')
        bindings.update(checked_hash(CACHE/'manifest.json','1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b'))
        dataset=HistoryDataset(CACHE,ROOT/'docs/history_probe_protocol_v1.md');bindings.update(dataset.bindings)
        if dataset.catalogue['group_main']!=catalogue['group_main']:raise ValueError('M2/memory group identities differ')
        forbidden=current_reservations(admitted)
        if {c['match_id'] for c in contexts}&forbidden:raise ValueError('query entered reservation')
        if {g['match_id'] for g in dataset.games}&{c['match_id'] for c in contexts}:raise ValueError('query/encoder/memory games overlap')
        static_path=Path(dataset.base_manifest['static_export_path'])
        bindings.update(checked_hash(static_path,dataset.base_manifest['static_export_sha256']))
        compiler=PublicFeatureCompiler(json.loads(static_path.read_text())['patches'])
        if json.loads(json.dumps(compiler.metadata()))!=dataset.feature_metadata:raise ValueError('feature schema differs from fitted encoder')
        features=[];positions=[];fallbacks=[]
        for i,context in enumerate(contexts):
            try:feature=compiler.state(context['prestate'],context['patch'])
            except (ValueError,KeyError,TypeError) as error:fallbacks.append({'row':i,'reason':str(error)});continue
            changed=compiler.state(mutate_ignored_fields(context['prestate']),context['patch'])
            if not compiled_equal(feature,changed):raise ValueError('forbidden field changed query features')
            if not compiled_equal(feature,compiler.state(context['source_projection'],context['patch'])):raise ValueError('projection features differ')
            features.append(feature);positions.append(i)
        positions=np.asarray(positions,dtype=np.int64)
        write(args.output/'plan.json',{'protocol_sha256':file_hash(PROTOCOL),'bindings_sha256':bindings,
            'selection':selection,'feature_fallbacks':fallbacks,'query_count':859,'weights':{'m2':.5,'memory_each_fold':.25},
            'target_file_opened':False,'training_performed':False})
        device=torch.device('cuda:0');store=TensorStore(dataset,device,guard,preload_budget_gib=0.)
        a=dataset.arrays;champs=dataset.focal_champion_indices;n=859
        # These are scores only for the existing legal options, never extra cards.
        candidate_groups=[[o['group'] for o in ref['m2_options']] for ref in refs]
        baseline=[np.asarray(m2[ref['m2_tensor_row'],candidate_groups[i]],dtype=np.float64) for i,ref in enumerate(refs)]
        rank_scores={arm:[.5*x.copy() for x in baseline] for arm in ('memory','shuffled')}
        for missing in fallbacks:
            for arm in rank_scores:rank_scores[arm][missing['row']]=baseline[missing['row']].copy()
        nr=np.full((2,n,32),-1,dtype=np.int64);nw=np.zeros((2,n,32));nd=np.full((2,n,32),np.nan)
        query_champs=np.asarray([int(f['champion_indices'][0]) for f in features]);fold_facts={}
        for fold in (0,1):
            bindings.update(checked_hash(OLD/f'actor_state_fold{fold}.pt',CHECKPOINTS[fold]));guard.bind(bindings)
            saved=torch.load(OLD/f'actor_state_fold{fold}.pt',map_location='cpu',weights_only=True)
            allowed=sorted(g['match_id'] for g in dataset.games if g['fold']!=fold)
            if saved['arm']!='state' or saved['scored_fold']!=fold or sorted(saved['training_game_ids'])!=allowed:raise ValueError('encoder training set changed')
            model=HistoryProbeActor(dataset.num_champions,dataset.num_items).to(device)
            model.load_state_dict(saved['actor'],strict=True);model.eval();set_history_validation(model,False);del saved
            train=np.flatnonzero(a['fold']!=fold);keys=np.empty((len(train),128),dtype=np.float32)
            status('encoding_training_memory',fold=fold,rows=len(train))
            with torch.inference_mode():
                for lo in range(0,len(train),256):
                    guard();r=train[lo:lo+256];keys[lo:lo+len(r)]=embedding(model,store,r).cpu().numpy()
                q=np.empty((len(features),128),dtype=np.float32)
                for lo in range(0,len(features),256):
                    batch=collate_public_boards(features[lo:lo+256],device=device)
                    h=model.state(batch);h=torch.cat((h,torch.zeros(len(h),114,device=device)),dim=1)
                    q[lo:lo+len(h)]=model.combined[1](model.combined[0](h)).cpu().numpy()
            keys=normalized(keys);q=normalized(q)
            shuffled,info=shuffled_labels(a['main_group_id'],train,champs,a['patch_index'],a['timestamp_ms'],fold)
            by_champ={int(c):np.flatnonzero(champs[train]==c) for c in np.unique(query_champs)}
            status('retrieving',fold=fold,queries=len(features))
            for local,position in enumerate(positions):
                if local%64==0:guard()
                eligible=by_champ[int(query_champs[local])]
                ids,d,w=neighbors(q[local],keys[eligible],memory_rows=train[eligible],memory_games=a['game_index'][train[eligible]],query_game=-1)
                if np.any(a['fold'][ids]==fold) or len(np.unique(a['game_index'][ids]))!=len(ids):raise ValueError('memory membership failed')
                nr[fold,position,:len(ids)]=ids;nw[fold,position,:len(ids)]=w;nd[fold,position,:len(ids)]=d
                for arm,labels in (('memory',a['main_group_id']),('shuffled',shuffled)):
                    added=distribution(labels[ids],w,9333)[candidate_groups[position]] if len(ids) else baseline[position]
                    rank_scores[arm][position]+=.25*added
            fold_facts[str(fold)]={'training_games':len(allowed),'training_rows':len(train),'shuffle':info}
            del model,keys,q,shuffled;gc.collect();torch.cuda.empty_cache()
        predictions=[]
        for i,ref in enumerate(refs):
            options=ref['m2_options'];ordered={'m2':options}
            for arm in rank_scores:ordered[arm]=stable_order(options,rank_scores[arm][i]);assert_same_candidates(options,ordered[arm])
            if not all(r['feasible'] for r in receipts[i]['m2'].values()):raise ValueError('native19 option infeasible')
            predictions.append({'sample_id':contexts[i]['sample_id'],'ranked':ordered,
                'candidate_groups':candidate_groups[i],'candidate_scores':{k:v[i].tolist() for k,v in rank_scores.items()},
                'm2_candidate_probabilities':baseline[i].tolist()})
        write(args.output/'prediction_orders.json',predictions)
        prediction_sha=file_hash(args.output/'prediction_orders.json')
        np.savez_compressed(args.output/'neighbors.npz',rows=nr,weights=nw,distances=nd)
        status('scoring_frozen_predictions')
        targets=validate_targets(SOURCE.parent/'targets.jsonl',source,contexts,catalogue,lookup,bindings)
        credits={arm:{k:[] for k in ('main','typed_main','complete')} for arm in ('m2','memory','shuffled')}
        scored=[];ceiling=[]
        for context,pred,target in zip(contexts,predictions,targets,strict=True):
            truth=target_description(target['target']['label_counts']);row={'sample_id':context['sample_id'],'match_id':context['match_id'],'kind':truth['kind'],'credits':{}}
            for arm,options in pred['ranked'].items():
                credit=recommendation_credit(truth,options[0]);row['credits'][arm]=credit
                for key,value in credit.items():credits[arm][key].append(value)
            ceiling.append(float(any(option_multiset(o)==truth['main'] for o in pred['ranked']['m2'])))
            scored.append(row)
        values={arm:{key:np.asarray(v) for key,v in metrics.items()} for arm,metrics in credits.items()}
        if values['m2']['main'].sum()!=454 or values['m2']['complete'].sum()!=354 or sum(ceiling)!=591:raise ValueError('originalM2 result does not reproduce')
        rng=np.random.default_rng(20261001);draws=rng.integers(0,n,size=(2000,n));metrics={};paired={}
        for arm,types in values.items():
            metrics[arm]={key:{'hits':int(v.sum()),'accuracy':float(v.mean()),'interval95':np.quantile(v[draws].mean(1),[.025,.975]).tolist()} for key,v in types.items()}
        for control in ('m2','shuffled'):
            paired[control]={}
            for key in ('main','typed_main','complete'):
                d=values['memory'][key]-values[control][key]
                paired[control][key]={'gain':float(d.mean()),'interval95':np.quantile(d[draws].mean(1),[.025,.975]).tolist(),'corrected':int(np.sum(d>0)),'broken':int(np.sum(d<0))}
        primary=paired['m2']['main'];gate=primary['gain']>=.03 and primary['interval95'][0]>0 and paired['shuffled']['main']['interval95'][0]>0
        if file_hash(args.output/'prediction_orders.json')!=prediction_sha:raise ValueError('predictions changed after target read')
        for raw,want in bindings.items():
            guard()
            if file_hash(raw)!=want:raise ValueError('bound input/code changed: '+raw)
        admission.verify_manifest_sources(admitted);admission.verify_manifest_sources(dataset.admission)
        forbidden=current_reservations(admitted)
        if ({c['match_id'] for c in contexts}|{g['match_id'] for g in dataset.games})&forbidden:raise ValueError('reservation changed')
        write(args.output/'scored_rows.json',scored)
        result={'schema':'purchase-memory-m2-v1','state':'complete','scope':'previously exposed859-game development; fixed M2 candidate reranking',
            'games':n,'arms':metrics,'memory_minus_control':paired,'useful_development_gate_passed':bool(gate),
            'candidate_top3_hits_unchanged':int(sum(ceiling)),'feature_fallbacks':fallbacks,'forbidden_feature_mutation_checked':len(features),
            'fold_facts':fold_facts,'candidate_objects_changed':0,'native18_native19_legality_verified':True,
            'original_m2_reproduced':True,'prediction_orders_sha256_before_targets':prediction_sha,
            'bindings_sha256':bindings,'files':{name:file_hash(args.output/name) for name in ('prediction_orders.json','neighbors.npz','scored_rows.json')},
            'training_performed':False,'test_payload_reads':False,'confirmation_payload_reads':False,'outcomes_analyzed':False,
            'source_code_imported':False,'breakthrough_established':False,'promotion_authorized':False,
            'elapsed_s':time.monotonic()-guard.started,'host_memory':guard.memory_report(),
            'limits':['No fresh confirmation; this cohort was already exposed.','Legality is conditional on the unchanged estimated wallet.','Ranking existing main cards does not evaluate new candidate generation or ward advice.']}
        write(args.output/'report.json',result);status('complete',gate=bool(gate),report=str(args.output/'report.json'))
    except Exception as error:status('failed',error=str(error),error_type=type(error).__name__);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deadline-seconds',default=1200,type=int);run(parser.parse_args())
