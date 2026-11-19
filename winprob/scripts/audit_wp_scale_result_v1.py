"""Independent score/selection checks and complete scale-model inference replay."""
from collections import Counter
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from winprob.iql_dataset_contract import file_hash,current_reservations
from winprob.wp_scale_models_v1 import load_model_rows

RUN=ROOT/'data/runs/wp-scale-fit-20260930-v1/experiment'
OUT=ROOT/'data/reports/wp_scale_result_audit_v1.json'
HELPER=ROOT/'data/reports/wp_distillation_audit_v1.py'
spec=importlib.util.spec_from_file_location('independent_scores',HELPER)
metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def grouped_mean(values,games):
    # Independent sorted contiguous reduction, not production bincount aggregation.
    order=np.argsort(games,kind='stable');sorted_games=games[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_games))+1]
    counts=np.diff(np.r_[starts,len(order)])
    return np.add.reduceat(np.asarray(values)[order],starts)/counts


metric.per_game=grouped_mean


def independent_selection(report):
    refs=('rich_frozen','combat_frozen')
    candidates=('rich_pilot_recalibrated','combat_pilot_recalibrated','rich_full','combat_full')
    main=report['cohorts']['complement'];coverage=report['coverage'];eligible=[];gates={}
    for name in candidates:
        check={'minimum_full_fit':coverage['fit_games']>=11000,'minimum_full_calibration':coverage['calibration_games']>=2700,
               'minimum_complement_development':coverage['complement_games']>=1700,'minimum_lead_games':coverage['lead_games']>=1000}
        for ref in refs:
            overall=main['comparisons'][name+'_vs_'+ref];lead=main['lead_time']['comparisons'][name+'_vs_'+ref]
            check.update({'two_percent_log_loss_vs_'+ref:overall['relative_log_loss_reduction']>=.02,
                'positive_log_loss_vs_'+ref:overall['log_loss']['ci95'][0]>0,'positive_brier_vs_'+ref:overall['brier']['ci95'][0]>0,
                'half_percent_lead_loss_vs_'+ref:lead['relative_log_loss_reduction']>=.005,
                'positive_lead_loss_vs_'+ref:lead['log_loss']['ci95'][0]>0,'nonnegative_lead_brier_vs_'+ref:lead['brier']['estimate']>=0})
        gates[name]=check
        if all(check.values()):eligible.append(name)
    def wins(a,b):
        pair=main['comparisons'][a+'_vs_'+b]
        return pair['relative_log_loss_reduction']>=.005 and pair['log_loss']['ci95'][0]>0 and pair['brier']['estimate']>=0
    families={}
    for family in ('rich','combat'):
        recal=family+'_pilot_recalibrated';full=family+'_full'
        if recal in eligible:families[family]=full if full in eligible and wins(full,recal) else recal
        elif full in eligible:families[family]=full
    chosen=None
    if families:
        chosen=families.get('rich',families.get('combat'))
        if len(families)==2 and wins(families['combat'],families['rich']):chosen=families['combat']
    expected={'gates':gates,'eligible':eligible,'retained_by_family':families,'selected_candidate':chosen,
              'useful_development_signal':chosen is not None,'confirmation_completed':False,'practical_benefit_established':False}
    need(expected==report['decision'],'Independent selection differs')
    return chosen


def run():
    started=time.monotonic();need(not OUT.exists(),'Refuse audit overwrite')
    def guard():
        need(time.monotonic()-started<1200,'Result audit1200second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            m=Memory();m.length=ctypes.sizeof(m)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) and m.available>=.75*1024**3,'Audit RAM floor')
    need(read(RUN.parent/'launcher_status.json')['state']=='completed','Producer must be terminal')
    report=read(RUN/'report.json');need(report['state']=='complete' and report['schema']=='wp-full-scale-comparison-v1','Incomplete report')
    report_sha=file_hash(RUN/'report.json')
    need((RUN/'report.sha256').read_text().split()==[report_sha,'report.json'],'Report digest mismatch')
    for path,digest in report['bindings_sha256'].items():need(file_hash(path)==digest,'Changed bound input')
    for name,digest in report['files'].items():need(file_hash(RUN/name)==digest,'Changed output')
    for name,digest in report['artifacts'].items():need(file_hash(RUN/(name+'.joblib'))==digest,'Changed model')
    lock=read(RUN/'training_lock.json');before=read(RUN/'experiment_lock.json')
    need(lock['all_models_complete'] and lock['comparison_metrics_inspected'] is False and lock['artifacts']==report['artifacts'],'Training lock differs')
    need(lock['bindings_sha256']==before['bindings_sha256']==report['bindings_sha256'],'Input locks differ')
    cache_path=Path(report['cache_manifest_path']);data,cache,bounds,admitted=load_model_rows(cache_path,guard)
    audit=read(ROOT/'data/reports/wp_scale_cache_audit_v1.json')
    need(audit['state']=='passed' and audit['cache_manifest_sha256']==file_hash(cache_path),'Cache audit mismatch')
    fit,cal,dev=(data[k] for k in ('fit','calibration','development'))
    need(not(set(fit['game'])&set(cal['game']) or set(fit['game'])&set(dev['game']) or set(cal['game'])&set(dev['game'])),'Role overlap')
    times={role:[r['game_creation_ms'] for r in cache['games'] if r['role']==role] for role in data}
    need(max(times['fit'])<min(times['calibration']) and max(times['calibration'])<min(times['development']),'Chronological split overlap')
    need(before['fit_sampled_frames']==len(fit['y']) and before['pilot_fit_sampled_frames']==22554,'FIT receipt differs')
    with np.load(RUN/'development_predictions.npz',allow_pickle=False) as saved:p={k:saved[k] for k in saved.files}
    for key in ('y','game','timestamp_ms','patch','pilot'):need(np.array_equal(dev[key],p[key]),'Prediction identity differs')
    lead=(dev['end_ms']>=0)&(dev['end_ms']-dev['timestamp_ms']>=180000);primary=~dev['pilot']
    need(np.array_equal(lead,p['lead_time_mask']) and np.array_equal(dev['end_ms'],p['evaluation_only_end_ms']),'Lead cohort differs')
    coverage={'fit_games':len(set(fit['game'])),'calibration_games':len(set(cal['game'])),
        'complement_games':len(set(dev['game'][primary])),'lead_games':len(set(dev['game'][primary&lead]))}
    need(coverage==report['coverage'],'Coverage differs')
    paths={'rich_frozen':ROOT/'data/runs/rich-20260929-v1/experiment/rich_trajectory.joblib',
           'combat_frozen':ROOT/'data/runs/current-combat-fit-20260930-v1/experiment/combat_resources.joblib'}
    paths.update({k:RUN/(k+'.joblib') for k in report['artifacts']});identities={};errors={}
    params={'max_iter':150,'max_leaf_nodes':15,'learning_rate':.06,'l2_regularization':10,'min_samples_leaf':100,
            'early_stopping':False,'random_state':20260929}
    with threadpool_limits(limits=1):
        for name,path in paths.items():
            guard();model=joblib.load(path);columns=61 if name.startswith('rich_') else 173
            need(model.indices==list(range(columns)) and model.scaler is None,'Unexpected feature preprocessing')
            need(model.model.n_features_in_==columns and model.model.n_iter_==150 and all(model.model.get_params()[k]==v for k,v in params.items()),'Tree recipe differs')
            if columns==173:need(model.supports_counterfactual_items is False and model.live_safe is False,'Capability claim differs')
            raw=model.model.predict_proba(dev['x'][:,:columns])[:,1];calibrated=metric.calibrated(raw,model.calibrator)
            metric.close(raw,p[name+'_raw'],name+'/raw');metric.close(calibrated,p[name],name+'/calibrated')
            errors[name]=float(np.max(np.abs(calibrated-p[name])));identities[name]=joblib.hash((model.model,model.indices,model.scaler))
            del model,raw,calibrated
    for family in ('rich','combat'):
        need(identities[family+'_frozen']==identities[family+'_pilot_recalibrated']==before['raw_reference_identities'][family+'_frozen'],
             'Recalibration altered raw trees')
        need(identities[family+'_full']!=identities[family+'_frozen'],'Full model reused frozen trees')
    prior=ROOT/'data/runs/current-combat-fit-20260930-v1/experiment/development_predictions.npz'
    with np.load(prior,allow_pickle=False) as old:
        for key in ('y','game','timestamp_ms','patch'):need(np.array_equal(p[key][~primary],old[key]),'Old pilot population changed')
        need(np.array_equal(p['rich_frozen'][~primary],old['reference']) and np.array_equal(p['combat_frozen'][~primary],old['combat_resources']),'Frozen pilot inference differs')
    checked=Counter()
    for cohort,take in (('complement',primary),('pilot',~primary),('union',np.ones(len(primary),dtype=bool))):
        guard();record=report['cohorts'][cohort];need(record['games']==len(set(p['game'][take])) and record['frames']==int(take.sum()),'Cohort size differs')
        for suffix,mask,part in (('',take,record),('/lead',take&lead,record['lead_time'])):
            guard()
            for name,summary in part['methods'].items():
                metric.validate_summary(p['y'][mask],p[name][mask],p['game'][mask],summary,cohort+suffix+'/'+name);checked['summaries']+=1
            for name,pair in part['comparisons'].items():
                a,b=name.split('_vs_');metric.validate_pair(p['y'][mask],p[b][mask],p[a][mask],p['game'][mask],pair,cohort+suffix+'/'+name);checked['pairs']+=1
    for name,summary in report['raw_complement_methods'].items():
        metric.validate_summary(p['y'][primary],p[name+'_raw'][primary],p['game'][primary],summary,name+'/raw');checked['summaries']+=1
    selected=independent_selection(report);guard()
    for path,digest in report['bindings_sha256'].items():need(file_hash(path)==digest,'Bound input changed during audit')
    need(file_hash(RUN/'report.json')==report_sha and not {r['match_id'] for r in cache['games']}&current_reservations(admitted),'Final source check failed')
    result={'schema':'wp-scale-result-independent-audit-v1','state':'passed','report_sha256':report_sha,
        'audit_code_sha256':file_hash(__file__),'metric_helper_sha256':file_hash(HELPER),'verified_bindings':len(report['bindings_sha256']),
        'coverage':coverage,'prediction_models':6,'prediction_rows_replayed':6*len(dev['y']),'maximum_probability_errors':errors,
        'verified_summaries':checked['summaries'],'verified_paired_comparisons':checked['pairs'],'decision_reproduced':True,
        'selected_candidate':selected,'pilot_prediction_parity':True,'recalibration_raw_trees_unchanged':True,
        'models_refitted':False,'new_raw_payload_reads':False,'reserved_intersections':0,'elapsed_s':time.monotonic()-started,
        'limits':['Independent proper scores, bootstrap intervals, ECE and selection cover primary, old pilot and union including lead strata.',
                  'Secondary time/patch metric slices are not separately audited.',
                  'Cache loading is shared with the producer, after its separate exhaustive shard/parity/accounting audit.',
                  'Development selection does not establish confirmation, causal action value or practical benefit.']}
    with OUT.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n')
    OUT.with_suffix('.sha256').write_text(file_hash(OUT)+' '+OUT.name+'\n',encoding='ascii');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
