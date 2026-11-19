"""Fixed two-candidate observed-current-state WP comparison; no raw reads."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import sklearn
import setback_experiment as safe
from ablate_rich import verify_rich_run
from winprob import admission
from winprob.iql_dataset_contract import current_reservations
from winprob.experiment import sample_training_indices
from winprob.metrics import summarize
from winprob.wp_distillation import comparison
from winprob.features_current_combat_v1 import VERSION,FEATURES,HEALTH_FEATURES
from winprob.current_observed_model_v1 import CurrentObservedModel,CANDIDATES,choose_candidate

PROTOCOL=ROOT/'docs/current_combat_protocol_v1.md'
RICH=ROOT/'data/runs/rich-20260929-v1/experiment'


def need(value,message):
    if not value:raise ValueError(message)


def write_json(path,value):
    safe.write_json(path,value);Path(path).with_suffix('.sha256').write_text(safe.digest(path)+' '+Path(path).name+'\n',encoding='ascii')


def run(cache_manifest,output):
    started=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    safe.set_below_normal_priority();bindings={};snapshots={}
    def guard():
        need(time.monotonic()-started<600,'Observed WP600second stage ceiling')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,
                 'Observed WP RAM floor0.75GiB')
        for path,pair in snapshots.items():
            current=Path(path).stat();need((current.st_size,current.st_mtime_ns)==pair,'Bound input/code changed')
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);digest=safe.digest(path) if digest is None else digest
        need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting input digest')
        need(safe.digest(path)==digest,'Changed bound input');bindings[str(path)]=digest
        stat=path.stat();snapshots[str(path)]=(stat.st_size,stat.st_mtime_ns)
    def status(stage,**extra):write_json(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra})
    try:
        code=list(sorted((ROOT/'winprob').glob('*.py')))+[Path(__file__),PROTOCOL,ROOT/'scripts/ablate_rich.py',
                ROOT/'scripts/rich_experiment.py',ROOT/'scripts/setback_experiment.py']
        for path in code:bind(path)
        guard();status('validating_frozen_inputs')
        cache_manifest=Path(cache_manifest).resolve(strict=True);safe.verify_sidecar(cache_manifest);bind(cache_manifest)
        cache=json.loads(cache_manifest.read_text())
        need(cache['schema']=='current-observed-combat-cache-v1' and cache['state']=='complete' and cache['training_authorized'] is True
             and all(cache['quality_gates'].values()) and cache['feature_version']==VERSION,'Unavailable/failed-quality combat cache')
        need(cache['feature_names']==list(FEATURES) and cache['health_feature_names']==list(HEALTH_FEATURES), 'Feature schema changed')
        need(cache['outcome_fields_indexed'] is False and cache['cache_y_array_accessed'] is False
             and cache['sealed_or_confirmation_payload_reads'] is False,'Feature source boundary violated')
        for path,digest in cache['bindings_sha256'].items():
            need(not str(path).endswith('.json.gz'),'Training must not reopen raw source bindings');bind(path,digest)
        original=ROOT/'data/runs/w1-pilot-20260929-v3'
        source,old=safe.verify_inputs(original/'cache/states.npz',original/'cache/source_manifest.json',original/'experiment')
        manifest=json.loads(Path(source['admission_manifest']).read_text());mids={row['match_id'] for row in source['games']}
        need(len(mids)==6000 and not mids&current_reservations(manifest),'Reserved/duplicate model scope')
        for path,digest in old.items():bind(path,digest)
        data,rich_report,controls,frozen,_=verify_rich_run(RICH)
        for path,digest in frozen.items():bind(path,digest)
        feature_path=(cache_manifest.parent/cache['feature_file']).resolve(strict=True)
        need(feature_path.is_relative_to(cache_manifest.parent),'Feature file escaped artifact directory');bind(feature_path,cache['feature_sha256'])
        with np.load(feature_path,allow_pickle=False) as loaded:extra={name:loaded[name] for name in loaded.files}
        for key in ('game','role','timestamp_ms','patch'):need(np.array_equal(extra[key],data[key]),'New/base row identity differs: '+key)
        need(extra['feature_names'].tolist()==list(FEATURES) and extra['x_extra'].shape==(138831,112)
             and np.isfinite(extra['x_extra']).all(),'Incomplete/nonfinite observed features')
        x,y,games,roles,times=(data[key] for key in ('x','y','game','role','timestamp_ms'))
        fit=np.flatnonzero(roles=='fit');fit=fit[sample_training_indices(games[fit],times[fit])]
        cal=roles=='calibration';dev=roles=='development';need(len(fit)==22554 and cal.sum()==23171 and dev.sum()==23366,'Original sampling differs')
        feature_names=data['feature_names'].tolist()
        matrices={'health_availability':np.column_stack((x,extra['x_extra'][:,:4])),
                  'combat_resources':np.column_stack((x,extra['x_extra']))}
        schema={'health_availability':feature_names+list(HEALTH_FEATURES),'combat_resources':feature_names+list(FEATURES)}
        ending=extra['evaluation_only_end_ms'];lead=(ending[dev]>=0)&(ending[dev]-times[dev]>=180000)
        lead_games=len(np.unique(games[dev][lead]))
        write_json(output/'experiment_lock.json',{'stage':'before_new_model_fitting','candidates':list(CANDIDATES),
            'feature_names_by_model':schema,'fit_sampled_frames':22554,'population':rich_report['population'],
            'lead_time_development_games':lead_games,'lead_time_development_frames':int(lead.sum()),
            'end_time_is_feature':False,'bindings_sha256':bindings,'selection_rule':'current_combat_protocol_v1.md'})
        probabilities={'reference':controls['combined'].copy()};raw_predictions={};artifacts={};receipts={}
        with threadpool_limits(limits=1):
            for name in CANDIDATES:
                guard();status('fitting_current_observed_model',model=name)
                model=CurrentObservedModel(schema[name]).fit(matrices[name][fit],y[fit],games[fit]);guard()
                model.calibrate(matrices[name][cal],y[cal],games[cal]);guard()
                probabilities[name]=model.predict(matrices[name][dev]);raw_predictions[name]=model.raw(matrices[name][dev])
                # Terminal timing is a separate evaluation array, not an input.
                previous=ending.copy();ending[:]=0
                repeated=model.predict(matrices[name][dev]);ending[:]=previous
                need(np.array_equal(repeated,probabilities[name]),'End timing reached model inference')
                try:model.predict(np.column_stack((matrices[name][dev][:1],np.array([123456.]))))
                except ValueError:strict_width=True
                else:strict_width=False
                need(strict_width,'Model accepted an extra metadata column')
                path=output/(name+'.joblib');joblib.dump(model,path,compress=3)
                path.with_suffix('.sha256').write_text(safe.digest(path)+' '+path.name+'\n',encoding='ascii');artifacts[name]=safe.digest(path)
                receipts[name]={'fit_iterations':int(model.model.n_iter_),'feature_count':len(schema[name]),'fit_frames':22554,
                    'calibration_games':1000,'end_timing_mutation_inference_equal':True,'extra_metadata_column_rejected':True,
                    'supports_counterfactual_items':model.supports_counterfactual_items,'live_safe':model.live_safe}
        write_json(output/'training_lock.json',{'both_models_complete':True,'comparison_metrics_inspected':False,
            'artifacts':artifacts,'receipts':receipts,'bindings_sha256':bindings})
        np.savez_compressed(output/'development_predictions.npz',y=y[dev],game=games[dev],timestamp_ms=times[dev],patch=data['patch'][dev],
            lead_time_mask=lead,evaluation_only_end_ms=ending[dev],**probabilities,**{key+'_raw':value for key,value in raw_predictions.items()})
        guard();status('evaluating_complete_observational_predictions')
        methods={name:summarize(y[dev],p,games[dev],2000) for name,p in probabilities.items()}
        comparisons={name+'_vs_reference':comparison(y[dev],probabilities['reference'],probabilities[name],games[dev]) for name in CANDIDATES}
        comparisons['combat_resources_vs_health_availability']=comparison(y[dev],probabilities['health_availability'],probabilities['combat_resources'],games[dev])
        lead_comparisons={};lead_methods={}
        if lead_games>=30:
            lead_methods={name:summarize(y[dev][lead],p[lead],games[dev][lead],2000) for name,p in probabilities.items()}
            lead_comparisons={name+'_vs_reference':comparison(y[dev][lead],probabilities['reference'][lead],probabilities[name][lead],games[dev][lead]) for name in CANDIDATES}
        decision=choose_candidate(comparisons,lead_comparisons,lead_games)
        slices={}
        groups=[(f'{lo}-{hi}min',(times[dev]>=lo*60000)&(times[dev]<(hi if hi<40 else 41)*60000)) for lo,hi in ((3,10),(10,20),(20,30),(30,40))]
        groups += [(str(p),data['patch'][dev]==p) for p in np.unique(data['patch'][dev])]
        for label,take in groups:
            count=len(np.unique(games[dev][take]))
            slices[label]={'games':count,'status':'not_measured_below30_games'} if count<30 else {
                'games':count,'methods':{name:summarize(y[dev][take],p[take],games[dev][take],2000) for name,p in probabilities.items()}}
        for path,digest in bindings.items():need(safe.digest(path)==digest,'Input/code changed before report')
        admission.verify_manifest_sources(manifest);need(not mids&current_reservations(manifest),'Reservations changed during trial');guard()
        snapshot=output/'code_snapshot';snapshot.mkdir();snapshots_saved={}
        for path in code:
            relative=Path(path).resolve().relative_to(ROOT);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination);snapshots_saved[relative.as_posix()]=safe.digest(destination)
        write_json(snapshot/'manifest.json',{'files':snapshots_saved,'raw_data_copied':False})
        result={'schema':'current-observed-wp-screen-v1','state':'complete','decision':decision,'methods':methods,'comparisons':comparisons,
            'lead_time':{'minimum_ms':180000,'games':lead_games,'frames':int(lead.sum()),'methods':lead_methods,'comparisons':lead_comparisons,
                         'interpretation':'retrospective future-defined lead-time stratum; never a live selection rule'},
            'student_raw_diagnostics':{name:summarize(y[dev],p,games[dev],2000) for name,p in raw_predictions.items()},
            'slices':slices,'population':rich_report['population'],'feature_names_by_model':schema,'input_quality':cache['quality'],
            'artifacts':artifacts,'receipts':receipts,'bindings_sha256':bindings,'cache_manifest_path':str(cache_manifest),
            'cache_manifest_sha256':safe.digest(cache_manifest),'files':{name:safe.digest(output/name) for name in
                ('experiment_lock.json','training_lock.json','development_predictions.npz')},
            'raw_payload_reads':False,'sealed_or_confirmation_payload_reads':False,'end_timing_is_feature':False,
            'future_frame_inputs':False,'causal_claim':False,'counterfactual_item_use_valid':False,'live_safe':False,
            'practical_or_confirmatory_benefit_established':False,'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__},
            'elapsed_s':time.monotonic()-started,
            'limitations':['Two declared candidates on repeatedly exposed development; fresh confirmation is still required.',
                          'Health-positive, net gold outflow and reported stat summaries are observational proxies.',
                          'Item-derived/current combat inputs cannot serve as an action-neutral counterfactual item judge.',
                          'Game bootstrap conditions on fitted models; no model retraining uncertainty.']}
        write_json(output/'report.json',result);status('complete',report=str(output/'report.json'));return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    report=run(args.cache_manifest,args.output)
    print(json.dumps({'decision':report['decision'],'comparisons':report['comparisons'],'lead_time_comparisons':report['lead_time']['comparisons'],
                      'elapsed_s':report['elapsed_s']},indent=2))
