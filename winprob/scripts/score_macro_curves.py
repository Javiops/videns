"""Score bounded requested replay curves with the frozen original W1 snapshot."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from export_macro_request import digest, load_request, read_bound_json
from winprob import admission, cache, features

DEFAULT_REQUEST = ROOT/'artifacts/macro-request/20260930-v1/request.json'
DEFAULT_MODEL = ROOT/'data/runs/w1-pilot-20260929-v3/experiment'
DEFAULT_OUTPUT = ROOT/'artifacts/macro-curves/20260930-v1'


def reject_quarantined(path):
    directory = Path(path).resolve()
    if not directory.is_dir():
        directory = directory.parent
    if any((parent/'QUARANTINED.md').exists() for parent in (directory,*directory.parents)):
        raise ValueError('Quarantined input is not admissible')


def verify_judge(directory):
    directory = Path(directory).resolve(strict=True)
    reject_quarantined(directory)
    report_path = directory/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('sealed_test_read') is not False or report.get('quarantined')
            or report.get('stage') != 'development_pilot'
            or report.get('gates',{}).get('snapshot_clears_logistic') is not True
            or report['gates'].get('negative_control_clear') is not True
            or report['population']['development']['games'] != 998):
        raise ValueError('Judge lacks the required clean 998-game original snapshot validation')
    for relative, expected in report['code_sha256'].items():
        if digest(ROOT/relative) != expected:
            raise ValueError('Frozen judge implementation changed: '+relative)
    model_path = directory/'snapshot.joblib'
    model_hash = digest(model_path)
    if model_hash != report['models']['snapshot']['artifact_sha256']:
        raise ValueError('Frozen snapshot model digest mismatch')
    source_cache = Path(report['source_cache']).resolve(strict=True)
    reject_quarantined(source_cache)
    meta_path = source_cache.parent/'cache_metadata.json'
    metadata, meta_hash = read_bound_json(meta_path)
    if digest(source_cache) != report['cache_sha256'] or metadata['npz_sha256'] != report['cache_sha256']:
        raise ValueError('Judge training cache binding mismatch')
    source_path = source_cache.parent/'source_manifest.json'
    if digest(source_path) != metadata['source_manifest_sha256']:
        raise ValueError('Judge source manifest digest mismatch')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if any(source.get(key) is not False for key in ('test_payload_reads','confirmatory_payload_reads','historical_validation_payload_reads')):
        raise ValueError('Judge source population is not clean')
    roles = {}
    for row in source['games']:
        if row['match_id'] in roles or row['role'] not in ('fit','calibration','development'):
            raise ValueError('Judge source exposure mapping is invalid')
        roles[row['match_id']] = row['role']
    evidence = {'state': 'validated_original_snapshot_development',
                'description': 'Original W1 snapshot passed its baseline and negative-control gates on 998 development games; retrospective diagnostic only.',
                'report_path': str(report_path), 'report_sha256': digest(report_path),
                'development_games': 998, 'development_frames': report['population']['development']['frames'],
                'log_loss': report['models']['snapshot']['log_loss'], 'brier': report['models']['snapshot']['brier'],
                'ece20': report['models']['snapshot']['ece20'],
                'not_pointwise_uncertainty': True}
    bindings = {str(report_path): evidence['report_sha256'], str(model_path): model_hash,
                str(source_cache): report['cache_sha256'], str(meta_path): meta_hash,
                str(source_path): metadata['source_manifest_sha256']}
    return model_path, model_hash, roles, evidence, bindings


def observed_windows(points):
    bins = {}
    for point in points:
        bins.setdefault(point['timestamp_ms']//60000, point)
    windows = []
    unavailable = 0
    for minute, start in sorted(bins.items()):
        if minute > 37:
            continue
        endpoint = bins.get(minute+3)
        if endpoint is None:
            unavailable += 1
            continue
        windows.append({'start_ms': start['timestamp_ms'], 'end_ms': endpoint['timestamp_ms'],
                        'delta_blue_pp': 100*(endpoint['p_blue']-start['p_blue'])})
    return windows, unavailable


def score_payloads(mid, match, timeline, predictor, model_digest, exposure_role, source_hashes):
    if match.get('info',{}).get('queueId') != 420:
        raise ValueError('Requested archive is not ranked solo')
    states = features.extract_game_states(match,timeline)
    base = {'schema':'winprob-macro-curve-v1','match_id':mid,'model_digest':model_digest,
            'exposure_role':exposure_role,'source_hashes':source_hashes,
            'interpretation':'Retrospective OMNI forecast changes; not action effects or pointwise confidence intervals.'}
    if not states:
        return {**base,'status':'unavailable','unavailable_reason':'no_valid_nonterminal_frames',
                'points':[],'observed_windows':[], 'completeness':{'valid_points':0}}
    if any(row['game_id']!=mid for row in states):
        raise ValueError('Feature identities differ from requested game')
    values=np.asarray([[row['x'][name] for name in features.ALL_FEATURES] for row in states],dtype=np.float32)
    predictions=np.asarray(predictor.predict(values),dtype=float)
    if predictions.shape!=(len(states),) or not np.isfinite(predictions).all() or not ((predictions>=0)&(predictions<=1)).all():
        raise ValueError('Judge emitted invalid probabilities')
    points=[{'timestamp_ms':int(row['timestamp_ms']),'p_blue':float(probability)} for row,probability in zip(states,predictions)]
    windows, missing_endpoints=observed_windows(points)
    stamps=[point['timestamp_ms'] for point in points]
    return {**base,'status':'available','unavailable_reason':None,'points':points,'observed_windows':windows,
            'completeness':{'valid_points':len(points),'first_ms':stamps[0],'last_ms':stamps[-1],
                            'max_gap_ms':max((b-a for a,b in zip(stamps,stamps[1:])),default=0),
                            'observed_window_count':len(windows),'missing_endpoint_bins':missing_endpoints,
                            'time_scope':'complete valid nonterminal frames from 3 through 40 minutes',
                            'windows_scope':'overlapping earliest-observed clock-bin pairs three minutes apart; no interpolation'}}


def run(request_path=DEFAULT_REQUEST, model_dir=DEFAULT_MODEL, output_dir=DEFAULT_OUTPUT, deadline_seconds=150):
    started=time.monotonic()
    request_path=Path(request_path).resolve(strict=True)
    request,request_hash,manifest=load_request(request_path)
    if len(request['games'])>34:
        raise ValueError('Bounded request exceeds 34 games')
    output=Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Curve export version already exists')
    model_path,model_hash,exposures,evidence,bindings=verify_judge(model_dir)
    code_paths=(Path(__file__),ROOT/'scripts/export_macro_request.py',Path(features.__file__),Path(cache.__file__),Path(admission.__file__))
    code_hashes={str(path.resolve()):digest(path) for path in code_paths}
    predictor=joblib.load(model_path)
    if predictor.kind!='snapshot' or predictor.indices!=[i for i,name in enumerate(features.ALL_FEATURES) if 'slope' not in name and 'lag' not in name]:
        raise ValueError('Expected the frozen original snapshot feature subset')
    archive=Path(manifest['source_data_root']).resolve(strict=True)/'raw_match_v5'
    output.mkdir(parents=True,exist_ok=False)
    status_path=output/'status.json'
    def status(state,**extra):
        status_path.write_bytes(admission.canonical_bytes({'state':state,'elapsed_s':time.monotonic()-started,**extra}))
    rows=[]
    try:
        with threadpool_limits(limits=1):
            for index,entry in enumerate(request['games'],1):
                if time.monotonic()-started>deadline_seconds:
                    raise RuntimeError('Diagnostic curve deadline exceeded')
                mid=entry['match_id']
                paths={endpoint:archive/endpoint/(mid+'.json.gz') for endpoint in ('match','timeline')}
                missing=[endpoint for endpoint,path in paths.items() if not path.is_file()]
                exposure=exposures.get(mid,'unseen')
                if missing:
                    row={'schema':'winprob-macro-curve-v1','match_id':mid,'status':'unavailable','model_digest':model_hash,
                         'exposure_role':exposure,'points':[],'observed_windows':[],'unavailable_reason':'missing_archive',
                         'missing_endpoints':missing,'source_hashes':{},'completeness':{'valid_points':0}}
                else:
                    for path in paths.values():
                        if not path.resolve(strict=True).is_relative_to(archive.resolve(strict=True)):
                            raise ValueError('Requested archive escaped its authorized root')
                    match,match_source=cache._read_payload(paths['match'],'match',mid)
                    timeline,timeline_source=cache._read_payload(paths['timeline'],'timeline',mid)
                    row=score_payloads(mid,match,timeline,predictor,model_hash,exposure,
                                       {'match':match_source,'timeline':timeline_source})
                    del match,timeline
                row['macro_role']=entry['macro_role']
                rows.append(row)
                status('scoring',processed_games=index,requested_games=len(request['games']))
        refreshed,refreshed_hash,_=load_request(request_path)
        if refreshed_hash!=request_hash or refreshed!=request:
            raise ValueError('Request changed while scoring')
        for path,expected in {**bindings,**code_hashes}.items():
            if digest(path)!=expected:
                raise ValueError('Frozen model/input/code changed while scoring: '+path)
        curves=output/'curves.jsonl'
        with curves.open('x',encoding='utf-8',newline='\n') as handle:
            for row in rows:
                handle.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        curves_hash=digest(curves)
        curves.with_suffix('.sha256').write_text(curves_hash+'  curves.jsonl\n',encoding='ascii')
        meta={'schema':'winprob-macro-curves-v1','version':output.name,'generated_utc':datetime.now(timezone.utc).isoformat(),
              'curves_sha256':curves_hash,'requested_game_ids':request['requested_game_ids'],
              'macro_split_sha256':request['macro_split_sha256'],'model_digest':model_hash,'model_validation':evidence,
              'request_path':str(request_path),'request_sha256':request_hash,'input_sha256':bindings,'code_sha256':code_hashes,
              'counts':{'requested':len(rows),'status':dict(Counter(row['status'] for row in rows)),
                        'exposure_role':dict(Counter(row['exposure_role'] for row in rows)),
                        'points':sum(len(row['points']) for row in rows),
                        'observed_windows':sum(len(row['observed_windows']) for row in rows)},
              'purpose':'retrospective_review_diagnostic_only','fitting_performed':False,'outcomes_exported':False,
              'test_payload_reads':False,'confirmatory_payload_reads':False,'macro_corpus_payload_reads':False,
              'limitations':['Curves are blue-team OMNI estimates, not live-POV probabilities.',
                             'Judge exposure is disclosed per game; fit/calibration games have in-sample judge exposure.',
                             'Observed changes are diagnostic and are not attributed to an action.',
                             'No per-point confidence interval is estimated; model validation is cohort-level.',
                             'Missing inputs remain unavailable; no network refetch or interpolation.']}
        meta_path=output/'meta.json'
        meta_path.write_bytes(admission.canonical_bytes(meta))
        meta_path.with_suffix('.sha256').write_text(digest(meta_path)+'  meta.json\n',encoding='ascii')
        status('complete',**meta['counts'])
        return meta
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,default=DEFAULT_REQUEST)
    parser.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--deadline-seconds',type=int,default=150)
    args=parser.parse_args()
    if not 1<=args.deadline_seconds<=150:
        parser.error('Deadline must be between one and 150 seconds')
    meta=run(args.request,args.model_dir,args.output,args.deadline_seconds)
    print(json.dumps({'output':str(args.output.resolve()),'counts':meta['counts'],'curves_sha256':meta['curves_sha256']},indent=2))


if __name__=='__main__':
    main()
