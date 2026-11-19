"""Fixed full-pool WP comparison with frozen and recalibration-only controls."""
from __future__ import annotations
import argparse
from copy import deepcopy
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
from winprob.iql_dataset_contract import current_reservations
from winprob.features_rich import ALL_RICH_FEATURES
from winprob.features_current_combat_v1 import FEATURES
from winprob.experiment import CalibratedModel
from winprob.current_observed_model_v1 import CurrentObservedModel
from winprob.wp_distillation import comparison
from winprob.metrics import summarize
from winprob.wp_scale_models_v1 import load_model_rows,select_scale_candidate,REFERENCES,CANDIDATES

PROTOCOL=ROOT/'docs/wp_scale_protocol_v1.md'
RICH=ROOT/'data/runs/rich-20260929-v1/experiment'
COMBAT=ROOT/'data/runs/current-combat-fit-20260930-v1/experiment'


def need(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    safe.write_json(path,value);Path(path).with_suffix('.sha256').write_text(safe.digest(path)+' '+Path(path).name+'\n',encoding='ascii')


def raw_identity(model):return joblib.hash((model.model,model.indices,model.scaler))


def run(cache_manifest,output):
    started=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    safe.set_below_normal_priority();bindings={};marks={}
    def guard():
        need(time.monotonic()-started<1800,'Scale fit/evaluation1800second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,'Scale fit RAM floor0.75GiB')
        for path,expected in marks.items():
            stat=Path(path).stat();need((stat.st_size,stat.st_mtime_ns)==expected,'Bound file changed during scale fit')
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);digest=safe.digest(path) if digest is None else digest
        need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting bound digest')
        need(safe.digest(path)==digest,'Changed bound input');bindings[str(path)]=digest
        stat=path.stat();marks[str(path)]=(stat.st_size,stat.st_mtime_ns)
    def status(stage,**extra):write(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra})
    try:
        code=list(sorted((ROOT/'winprob').glob('*.py')))+[Path(__file__),PROTOCOL,ROOT/'scripts/setback_experiment.py']
        for path in code:bind(path)
        guard();status('loading_verified_scale_shards')
        audit_path=ROOT/'data/reports/wp_scale_cache_audit_v1.json'
        audit=json.loads(audit_path.read_text());bind(audit_path)
        need(audit['state']=='passed' and audit['cache_manifest_sha256']==safe.digest(cache_manifest),
             'Passed independent cache accounting/parity audit required')
        data,cache,old_bindings,admission=load_model_rows(cache_manifest,guard)
        for path,digest in old_bindings.items():bind(path,digest)
        need(cache['rich_feature_names']==list(ALL_RICH_FEATURES) and cache['combat_feature_names']==list(FEATURES),'Scale feature definitions differ')
        fit,cal,dev=(data[role] for role in ('fit','calibration','development'))
        need(len(np.unique(fit['game']))>=11000 and len(np.unique(cal['game']))>=2700,'Full training/calibration scope too small')
        primary=~dev['pilot'];old_dev=dev['pilot'];lead=(dev['end_ms']>=0)&(dev['end_ms']-dev['timestamp_ms']>=180000)
        coverage={'fit_games':len(np.unique(fit['game'])),'calibration_games':len(np.unique(cal['game'])),
                  'complement_games':len(np.unique(dev['game'][primary])),'lead_games':len(np.unique(dev['game'][primary&lead]))}
        need(coverage['complement_games']>=1700 and coverage['lead_games']>=1000,'Primary complement/lead-time scope too small')
        need(len(np.unique(dev['game'][old_dev]))==998 and old_dev.sum()==23366 and cal['pilot'].sum()==23171,'Original evaluation/calibration rows changed')
        reports={'rich':json.loads((RICH/'report.json').read_text()),'combat':json.loads((COMBAT/'report.json').read_text())}
        bind(RICH/'report.json');bind(COMBAT/'report.json')
        paths={'rich_frozen':RICH/'rich_trajectory.joblib','combat_frozen':COMBAT/'combat_resources.joblib'}
        wanted={'rich_frozen':reports['rich']['artifacts']['rich_trajectory']['sha256'],
                'combat_frozen':reports['combat']['artifacts']['combat_resources']}
        for name,path in paths.items():bind(path,wanted[name])
        original_predictions=COMBAT/'development_predictions.npz';bind(original_predictions,reports['combat']['files']['development_predictions.npz'])
        with np.load(original_predictions,allow_pickle=False) as old:
            for key in ('game','y','timestamp_ms','patch'):need(np.array_equal(dev[key][old_dev],old[key]),'Old development identity differs')
            expected_p={'rich_frozen':old['reference'].copy(),'combat_frozen':old['combat_resources'].copy()}
        probabilities={};raw_probabilities={};artifacts={};receipts={}
        with threadpool_limits(limits=1):
            frozen={name:joblib.load(path) for name,path in paths.items()}
            raw_before={name:raw_identity(model) for name,model in frozen.items()}
            for name,model in frozen.items():
                columns=61 if name=='rich_frozen' else 173
                # No development-complement predictions are inspected before the lock.
                repeated=model.predict(dev['x'][old_dev,:columns])
                need(np.array_equal(repeated,expected_p[name]),'Frozen original pilot predictions do not reproduce exactly')
            write(output/'experiment_lock.json',{'stage':'before_new_calibration_or_fitting','candidates':list(CANDIDATES),
                'references':list(REFERENCES),'coverage':coverage,'fit_sampled_frames':len(fit['y']),
                'pilot_fit_sampled_frames':int(fit['pilot'].sum()),'cache_manifest_path':str(Path(cache_manifest).resolve()),
                'cache_manifest_sha256':safe.digest(cache_manifest),'raw_reference_identities':raw_before,'bindings_sha256':bindings,
                'primary_cohort':'development complement outside original6000 pilot; not confirmation'})
            for family in ('rich','combat'):
                columns=61 if family=='rich' else 173;reference=family+'_frozen';guard()
                status('recalibrating_fixed_pilot_trees',model=family)
                recal=deepcopy(frozen[reference]);recal.calibrate(cal['x'][:,:columns],cal['y'],cal['game']);guard()
                need(raw_identity(recal)==raw_before[reference] and raw_identity(frozen[reference])==raw_before[reference],
                     'Recalibration altered frozen raw trees/preprocessing')
                name=family+'_pilot_recalibrated';path=output/(name+'.joblib');joblib.dump(recal,path,compress=3)
                artifacts[name]=safe.digest(path);path.with_suffix('.sha256').write_text(artifacts[name]+' '+path.name+'\n',encoding='ascii')
                receipts[name]={'raw_tree_source':reference,'raw_identity':raw_identity(recal),'raw_trees_unchanged':True,
                                'fit_games':4002,'calibration_games':coverage['calibration_games'],'feature_count':columns}
                status('fitting_full_pool_model',model=family)
                if family=='rich':model=CalibratedModel('trajectory',list(range(61)))
                else:model=CurrentObservedModel(list(ALL_RICH_FEATURES)+list(FEATURES))
                model.fit(fit['x'][:,:columns],fit['y'],fit['game']);guard()
                model.calibrate(cal['x'][:,:columns],cal['y'],cal['game']);guard()
                name=family+'_full';path=output/(name+'.joblib');joblib.dump(model,path,compress=3)
                artifacts[name]=safe.digest(path);path.with_suffix('.sha256').write_text(artifacts[name]+' '+path.name+'\n',encoding='ascii')
                receipts[name]={'fresh_fit':True,'fit_games':coverage['fit_games'],'fit_frames':len(fit['y']),
                                'calibration_games':coverage['calibration_games'],'feature_count':columns,'iterations':int(model.model.n_iter_)}
                del recal,model
            write(output/'training_lock.json',{'all_models_complete':True,'comparison_metrics_inspected':False,
                'artifacts':artifacts,'receipts':receipts,'bindings_sha256':bindings,'unchanged_raw_reference_identities':raw_before})
            status('predicting_complete_development_cohorts')
            for name in (*REFERENCES,*CANDIDATES):
                guard();model=frozen[name] if name in frozen else joblib.load(output/(name+'.joblib'))
                columns=61 if name.startswith('rich_') else 173
                probabilities[name]=model.predict(dev['x'][:,:columns]);raw_probabilities[name]=model.raw(dev['x'][:,:columns])
                need(np.isfinite(probabilities[name]).all() and np.all((probabilities[name]>=0)&(probabilities[name]<=1)),'Invalid scale forecast')
                if name not in frozen:del model
            for name,model in frozen.items():need(raw_identity(model)==raw_before[name],'Frozen reference mutated')
        np.savez_compressed(output/'development_predictions.npz',y=dev['y'],game=dev['game'],timestamp_ms=dev['timestamp_ms'],
            patch=dev['patch'],pilot=dev['pilot'],evaluation_only_end_ms=dev['end_ms'],lead_time_mask=lead,
            **probabilities,**{name+'_raw':value for name,value in raw_probabilities.items()})
        pairs=[(candidate,reference) for candidate in CANDIDATES for reference in REFERENCES]
        pairs += [(family+'_full',family+'_pilot_recalibrated') for family in ('rich','combat')]
        pairs += [(combat,rich) for combat in ('combat_pilot_recalibrated','combat_full') for rich in ('rich_pilot_recalibrated','rich_full')]
        pairs += [('combat_frozen','rich_frozen')]
        cohort_reports={}
        for cohort,take in (('complement',primary),('pilot',old_dev),('union',np.ones(len(primary),dtype=bool))):
            guard();status('evaluating_fixed_cohort',cohort=cohort)
            methods={name:summarize(dev['y'][take],p[take],dev['game'][take],2000) for name,p in probabilities.items()}
            contrasts={a+'_vs_'+b:comparison(dev['y'][take],probabilities[b][take],probabilities[a][take],dev['game'][take]) for a,b in pairs}
            ahead=take&lead;ng=len(np.unique(dev['game'][ahead]));ahead_methods={};ahead_pairs={}
            if ng>=30:
                ahead_methods={name:summarize(dev['y'][ahead],p[ahead],dev['game'][ahead],2000) for name,p in probabilities.items()}
                ahead_pairs={a+'_vs_'+b:comparison(dev['y'][ahead],probabilities[b][ahead],probabilities[a][ahead],dev['game'][ahead])
                             for a,b in pairs if a in CANDIDATES and b in REFERENCES}
            cohort_reports[cohort]={'games':len(np.unique(dev['game'][take])),'frames':int(take.sum()),'methods':methods,
                'comparisons':contrasts,'lead_time':{'games':ng,'frames':int(ahead.sum()),'methods':ahead_methods,'comparisons':ahead_pairs}}
        main=cohort_reports['complement'];decision=select_scale_candidate(main['comparisons'],main['lead_time']['comparisons'],coverage)
        slices={}
        groups=[(f'{lo}-{hi}min',primary&(dev['timestamp_ms']>=lo*60000)&(dev['timestamp_ms']<(hi if hi<40 else 41)*60000))
                for lo,hi in ((3,10),(10,20),(20,30),(30,40))]
        groups += [(str(p),primary&(dev['patch']==p)) for p in np.unique(dev['patch'][primary])]
        for label,take in groups:
            ng=len(np.unique(dev['game'][take]))
            slices[label]={'games':ng,'status':'not_measured_below30_games'} if ng<30 else {'games':ng,
                'methods':{name:summarize(dev['y'][take],p[take],dev['game'][take],2000) for name,p in probabilities.items()}}
        guard()
        for path,digest in bindings.items():need(safe.digest(path)==digest,'Bound input/code changed before publication')
        for row in cache['games']:need(safe.digest(Path(cache_manifest).parent/row['file'])==row['sha256'],'Shard changed during fitting')
        need(not {row['match_id'] for row in cache['games']}&current_reservations(admission),'Reservation changed during scale study');guard()
        snapshot=output/'code_snapshot';snapshot.mkdir();copied={}
        for path in code:
            relative=Path(path).resolve().relative_to(ROOT);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination);copied[relative.as_posix()]=safe.digest(destination)
        write(snapshot/'manifest.json',{'files':copied,'raw_data_copied':False})
        result={'schema':'wp-full-scale-comparison-v1','state':'complete','decision':decision,'coverage':coverage,
            'cohorts':cohort_reports,'primary_cohort':'complement','secondary_slices_complement':slices,
            'raw_complement_methods':{name:summarize(dev['y'][primary],p[primary],dev['game'][primary],2000) for name,p in raw_probabilities.items()},
            'artifacts':artifacts,'receipts':receipts,'bindings_sha256':bindings,'cache_manifest_path':str(Path(cache_manifest).resolve()),
            'files':{name:safe.digest(output/name) for name in ('experiment_lock.json','training_lock.json','development_predictions.npz')},
            'original_pilot_prediction_parity':True,'raw_reference_identities_unchanged':raw_before,'calibration_only_controls_present':True,
            'new_payload_reads':False,'sealed_or_confirmation_payload_reads':False,'models_initialized_from_frozen_trees_for_refit':False,
            'retrospective_omni_only':True,'causal_claim':False,'practical_or_confirmatory_benefit_established':False,
            'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__},'elapsed_s':time.monotonic()-started,
            'limitations':['Newly evaluated development complement is not globally untouched or confirmation.',
                          'Four declared candidates and fixed hierarchy; intervals condition on fitted models.',
                          'Recalibration-only controls separate calibration data from expanded fitting data.',
                          'No intervention/live-safety claim for current-combat inputs; all1911 confirmation games remain closed.']}
        write(output/'report.json',result);status('complete',report=str(output/'report.json'));return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=run(args.cache_manifest,args.output)
    print(json.dumps({'decision':result['decision'],'coverage':result['coverage'],'elapsed_s':result['elapsed_s']},indent=2))
