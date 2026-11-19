"""Fixed FIT-only privileged-state teachers and current-only win students."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
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
from winprob.experiment import CalibratedModel,sample_training_indices
from winprob.features_rich import ALL_RICH_FEATURES
from winprob.metrics import summarize
from winprob.wp_distillation import (CurrentWinStudent,privileged_fit_matrix,teacher_plan,comparison,useful_gate,STUDENTS)

RICH=ROOT/'data/runs/rich-20260929-v1/experiment'
PROTOCOL=ROOT/'docs/wp_distillation_protocol_v1.md'
RICH_CACHE_SHA='65414f41331d4a82601ef38d9a95ff4b465bc90a758629730683593a93e729cb'
RICH_REPORT_SHA='f5e566b4187f02858c7fecdf7fb9ff64d689336461235d5080102d12d4f4808d'


def require(condition,message):
    if not condition:raise ValueError(message)


def write_json(path,value):
    safe.write_json(path,value)


def save_model(path,value):
    require(not path.exists(),'Refuse model overwrite');joblib.dump(value,path,compress=3)
    path.with_suffix('.sha256').write_text(safe.digest(path)+' '+path.name+'\n',encoding='ascii');return safe.digest(path)


def run(output):
    started=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    safe.set_below_normal_priority();bound={};snapshots={}
    def status(stage,**extra):write_json(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra})
    def guard():
        require(time.monotonic()-started<=600,'Win distillation600second stage cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,
                    'Win distillation RAM floor0.75GiB')
        for path,expected in snapshots.items():
            stat=Path(path).stat();require((stat.st_size,stat.st_mtime_ns)==expected,'Bound input/code changed during run')
    def bind(path,want=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);digest=safe.digest(path) if want is None else want
        require(str(path) not in bound or bound[str(path)]==digest,'Conflicting input/code binding')
        require(safe.digest(path)==digest,'Input/code digest differs');bound[str(path)]=digest
        stat=path.stat();snapshots[str(path)]=(stat.st_size,stat.st_mtime_ns)
    try:
        code=list(sorted((ROOT/'winprob').glob('*.py')))+[Path(__file__),PROTOCOL,ROOT/'scripts/ablate_rich.py',
             ROOT/'scripts/rich_experiment.py',ROOT/'scripts/setback_experiment.py']
        for path in code:bind(path)
        status('verifying_admission_before_array_access');guard()
        original=ROOT/'data/runs/w1-pilot-20260929-v3'
        source,early=safe.verify_inputs(original/'cache/states.npz',original/'cache/source_manifest.json',original/'experiment')
        manifest_path=Path(source['admission_manifest']);manifest=json.loads(manifest_path.read_text())
        mids={row['match_id'] for row in source['games']}
        require(len(mids)==6000 and not mids&current_reservations(manifest),'Reserved or duplicate source game')
        for path,want in early.items():bind(path,want)
        bind(RICH/'rich_states.npz',RICH_CACHE_SHA);bind(RICH/'report.json',RICH_REPORT_SHA)
        data,rich_report,controls,frozen,_=verify_rich_run(RICH)
        for path,want in frozen.items():bind(path,want)
        x,y,games,roles,times=(data[key] for key in ('x','y','game','role','timestamp_ms'))
        names=data['feature_names'].tolist();require(tuple(names)==ALL_RICH_FEATURES and x.shape==(138831,61),'Current rich feature population differs')
        for role,n_games,n_frames in (('fit',4002,92294),('calibration',1000,23171),('development',998,23366)):
            take=roles==role;require(int(take.sum())==n_frames and len(np.unique(games[take]))==n_games,'Frozen role population differs')
        source_by_index={int(row['game_index']):row for row in source['games']}
        require(len(source_by_index)==6000 and set(source_by_index)==set(map(int,np.unique(games))),
                'Source game-index mapping is not complete and unique')
        for game in np.unique(games):
            mask=games==game;entry=source_by_index[int(game)]
            require(len(np.unique(y[mask]))==1 and set(roles[mask])=={entry['role']},'Game label/role mismatch')
        status('constructing_FIT_only_teacher_alignment')
        fit,joint,future=privileged_fit_matrix(x,games,times,roles);sampled=sample_training_indices(games[fit],times[fit])
        require(len(sampled)==22554,'Existing deterministic FIT frame sample changed')
        folds,plans=teacher_plan(games[fit],source_by_index,sampled)
        present=future>=0;horizon=times[future[present]]-times[fit[present]]
        plan_receipts=[]
        for plan in plans:
            entry={'heldout_fold':plan['heldout_fold'],'calibration_fold':plan['calibration_fold']}
            for part in ('train','cal','target'):
                row_indices=plan[part];ids=sorted({source_by_index[int(game)]['match_id'] for game in games[fit[row_indices]]})
                entry[part+'_game_ids']=ids;entry[part+'_frames']=len(row_indices)
                entry[part+'_global_row_indices_sha256']=hashlib.sha256(fit[row_indices].astype(np.int64).tobytes()).hexdigest()
            plan_receipts.append(entry)
        np.savez_compressed(output/'teacher_alignment.npz',fit_global_indices=fit,future_global_indices=future,
                            sampled_fit_local_indices=sampled,teacher_fold=folds)
        write_json(output/'teacher_plan.json',{'plans':plan_receipts,'published_before_teacher_fit':True,'fit_games':4002,
            'fit_frames':92294,'student_sampled_frames':22554,'joint_features':123,'student_features':61,
            'available_future_frames':int(present.sum()),'missing_future_frames_retained':int((~present).sum()),
            'available_future_sampled_frames':int(present[sampled].sum()),
            'actual_horizon_ms_quantiles':np.percentile(horizon,[0,25,50,75,100]).tolist(),
            'future_rule':'earliest qualified same-game frame in currentclockbin+3; missing zeros+mask; FIT only',
            'alignment_sha256':safe.digest(output/'teacher_alignment.npz'),'bindings_sha256':bound})
        guard();current=x[fit];fit_y=y[fit];fit_games=games[fit]
        oof={kind:np.full(len(fit),np.nan) for kind in ('current_teacher','future_teacher')}
        oof_coverage={kind:np.zeros(len(fit),dtype=np.int8) for kind in oof};artifacts={};teacher_records=[]
        (output/'teachers').mkdir();(output/'students').mkdir()
        with threadpool_limits(limits=1):
            for plan in plans:
                train,cal,target=(plan[key] for key in ('train','cal','target'))
                for kind,features in (('current_teacher',current),('future_teacher',joint)):
                    guard();status('fitting_teacher',teacher=kind,target_fold=plan['heldout_fold'])
                    teacher=CalibratedModel('snapshot',list(range(features.shape[1])))
                    teacher.fit(features[train],fit_y[train],fit_games[train]);guard()
                    teacher.calibrate(features[cal],fit_y[cal],fit_games[cal]);guard()
                    p=teacher.predict(features[target]);require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid OOF teacher targets')
                    oof[kind][target]=p;oof_coverage[kind][target]+=1
                    filename=f'teachers/{kind}_fold{plan["heldout_fold"]}.joblib'
                    artifacts[filename]=save_model(output/filename,teacher)
                    teacher_records.append({'kind':kind,'target_fold':plan['heldout_fold'],'feature_count':features.shape[1],
                        'fit_iterations':int(teacher.model.n_iter_),'train_frames':len(train),'calibration_frames':len(cal),'target_frames':len(target),
                        'membership_receipt_index':plan['heldout_fold'],'artifact':filename,'privileged_training_only':True})
            require(all(np.isfinite(value).all() for value in oof.values()) and all(np.all(value==1) for value in oof_coverage.values()),
                    'OOF targets incomplete or duplicated')
            np.savez_compressed(output/'teacher_oof_targets.npz',fit_global_indices=fit,teacher_fold=folds,**oof)
            write_json(output/'teacher_receipts.json',{'teachers':teacher_records,'coverage_exactly_once':True,
                'source_roles_used':['fit'],'no_external_calibration_or_development_teacher_fit':True})
            probabilities={'frozen_rich_classifier':controls['combined'].copy()};raw_probabilities={};clipping={};student_receipts={}
            calibrated=roles=='calibration';dev=roles=='development'
            target_sets={'hard':fit_y,**oof}
            for kind in STUDENTS:
                guard();status('fitting_current_only_student',student=kind)
                student=CurrentWinStudent(names).fit(current[sampled],target_sets[kind][sampled],fit_games[sampled]);guard()
                student.calibrate(x[calibrated],y[calibrated],games[calibrated]);guard()
                filename=f'students/{kind}.joblib';artifacts[filename]=save_model(output/filename,student)
                clipping[kind]={}
                for role,features in (('fit_sampled',current[sampled]),('calibration',x[calibrated]),('development',x[dev])):
                    raw=student.unbounded_raw(features)
                    clipping[kind][role]={'frames':len(raw),'below_clip':int((raw<1e-7).sum()),'above_clip':int((raw>1-1e-7).sum())}
                probabilities[kind]=student.predict(x[dev]);raw_probabilities[kind]=student.raw(x[dev])
                # Inference is self-contained; changing training-only target arrays
                # cannot affect an already fitted current-only student's forecast.
                old=oof['future_teacher'].copy();oof['future_teacher'][:]=1-old
                repeated=student.predict(x[dev]);oof['future_teacher'][:]=old
                require(np.array_equal(repeated,probabilities[kind]),'Training-only targets reached inference')
                try:student.predict(np.zeros((1,123),dtype=np.float32))
                except ValueError:dimension_rejected=True
                else:dimension_rejected=False
                require(dimension_rejected,'Student accepted privileged feature dimensions')
                student_receipts[kind]={'fit_iterations':int(student.model.n_iter_),'fit_frames':22554,'fit_games':4002,
                    'calibration_games':1000,'current_feature_count':61,'privileged_matrix_rejected':True,'target_mutation_inference_equal':True,
                    'target_sha256':hashlib.sha256(np.asarray(target_sets[kind][sampled],dtype=np.float64).tobytes()).hexdigest(),
                    'artifact':filename}
        # Freeze every model and prediction before comparison metrics are inspected.
        write_json(output/'training_lock.json',{'artifacts':artifacts,'teacher_receipts':teacher_records,'student_receipts':student_receipts,
            'all_models_complete':True,'comparisons_inspected':False,'bindings_sha256':bound,'protocol_sha256':safe.digest(PROTOCOL)})
        np.savez_compressed(output/'development_predictions.npz',y=y[dev],game=games[dev],timestamp_ms=times[dev],patch=data['patch'][dev],
                            **probabilities,**{kind+'_raw':p for kind,p in raw_probabilities.items()})
        guard();status('evaluating_complete_current_only_predictions')
        methods={kind:summarize(y[dev],p,games[dev],2000) for kind,p in probabilities.items()}
        comparisons={'future_teacher_vs_'+control:comparison(y[dev],probabilities[control],probabilities['future_teacher'],games[dev])
                     for control in ('hard','frozen_rich_classifier','current_teacher')}
        gates=useful_gate(comparisons)
        slices={}
        for label,take in [(f'{low}-{high}min',(times[dev]>=low*60000)&(times[dev]<(high if high<40 else 41)*60000))
                           for low,high in ((3,10),(10,20),(20,30),(30,40))]+[(str(patch),data['patch'][dev]==patch) for patch in np.unique(data['patch'][dev])]:
            count=len(np.unique(games[dev][take]))
            slices[label]={'games':count,'status':'not_measured_below30_games'} if count<30 else {
                'games':count,'methods':{kind:summarize(y[dev][take],p[take],games[dev][take],2000) for kind,p in probabilities.items()}}
        teachers={kind:summarize(fit_y,p,fit_games,2000) for kind,p in oof.items()}
        teacher_pair=comparison(fit_y,oof['current_teacher'],oof['future_teacher'],fit_games)
        for path,want in bound.items():require(safe.digest(path)==want,'Bound code/input changed before publication')
        admission.verify_manifest_sources(manifest);require(not mids&current_reservations(manifest),'Reservation changed during experiment');guard()
        snapshot=output/'code_snapshot';snapshot.mkdir();copied={}
        for path in code:
            relative=Path(path).resolve().relative_to(ROOT);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination);copied[relative.as_posix()]=safe.digest(destination)
        write_json(snapshot/'manifest.json',{'files':copied,'raw_data_copied':False})
        proof={'original6000_game_population':True,'strict_teacher_role_and_game_separation':True,'complete_OOF_coverage':True,
               'same22554_current_student_fit_frames':True,'no_future_student_inputs':True,'all8_teachers_and3_students_complete':True,
               'unchanged_frozen_strong_control':True,'input_code_and_current_reservations_verified':True}
        result={'schema':'privileged-win-distillation-v1','state':'complete','methods':methods,'comparisons':comparisons,'gates':gates,
            'proof_gates':proof,'useful_internal_signal':all(gates.values()) and all(proof.values()),'teacher_oof_diagnostics':teachers,
            'teacher_future_vs_current_oof':teacher_pair,'student_raw_diagnostics':{kind:summarize(y[dev],p,games[dev],2000) for kind,p in raw_probabilities.items()},
            'raw_student_clipping':clipping,'slices':slices,'population':rich_report['population'],'current_feature_names':names,
            'artifacts':artifacts,'bindings_sha256':bound,'student_receipts':student_receipts,
            'files':{name:safe.digest(output/name) for name in ('teacher_alignment.npz','teacher_plan.json','teacher_oof_targets.npz',
                'teacher_receipts.json','training_lock.json','development_predictions.npz')},
            'raw_payload_reads':False,'sealed_or_confirmation_payload_reads':False,'outcomes_used_for_training':True,
            'development_is_fresh':False,'causal_claim':False,'practical_benefit_established':False,'future_available_at_inference':False,
            'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__},'elapsed_s':time.monotonic()-started,
            'limitations':['Retrospective OMNI current features; no live-safe or action-value claim.',
                'Cross-fitted estimated teachers need not equal the true conditional mean; the mathematical variance identity gives no finite-sample gain guarantee.',
                'All current frames retained when future unavailable; missingness is privileged teacher-only information.',
                'Repeated exposed development and model-conditional game intervals; all1911 confirmation games remain closed.']}
        write_json(output/'report.json',result);status('complete',report=str(output/'report.json'))
        return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=run(args.output)
    print(json.dumps({'useful_internal_signal':result['useful_internal_signal'],'comparisons':result['comparisons'],'elapsed_s':result['elapsed_s']},indent=2))
