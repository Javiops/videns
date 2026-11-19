"""Bounded shard loading and fixed selection for the larger WP study."""
import json
from pathlib import Path
import numpy as np
from .experiment import sample_training_indices
from .wp_scale_scope_v1 import checked_json,verify_scope
from .iql_dataset_contract import file_hash,current_reservations

REFERENCES=('rich_frozen','combat_frozen')
CANDIDATES=('rich_pilot_recalibrated','combat_pilot_recalibrated','rich_full','combat_full')


def need(value,message):
    if not value:raise ValueError(message)


def validate_shard(arrays,entry):
    n=entry['rows'];gid=entry['game_index']
    need(set(arrays)=={'x_rich','x_combat','y','timestamp_ms','fit_selected_local'},'Unexpected shard fields')
    need(arrays['x_rich'].shape==(n,61) and arrays['x_combat'].shape==(n,112)
         and arrays['x_rich'].dtype==arrays['x_combat'].dtype==np.float32,'Shard feature shape/dtype differs')
    need(np.isfinite(arrays['x_rich']).all() and np.isfinite(arrays['x_combat']).all(),'Nonfinite shard features')
    y=arrays['y'];clock=arrays['timestamp_ms'];selected=arrays['fit_selected_local']
    need(y.shape==clock.shape==(n,) and y.dtype==np.int8 and clock.dtype==np.int64 and len(np.unique(y))==1
         and set(y.tolist())<={0,1} and np.all(np.diff(clock)>0) and clock.min()>=180000 and clock.max()<=2400000,
         'Shard label/time contract differs')
    expected=sample_training_indices(np.full(n,gid,dtype=np.int32),clock) if entry['role']=='fit' else np.empty(0,dtype=np.int64)
    need(np.array_equal(selected,expected) and len(selected)==entry['fit_sampled_frames'],'Per-game sampling differs')


def load_model_rows(path,guard=lambda:None):
    path=Path(path).resolve(strict=True);cache,digest=checked_json(path)
    need(cache['schema']=='wp-full-scale-feature-cache-v1' and cache['state']=='complete' and cache['training_authorized'] is True
         and all(cache['gates'].values()),'Complete quality-passing scale cache required')
    scope,scope_sha=checked_json(cache['scope_path']);need(scope_sha==cache['scope_sha256'],'Scope changed')
    admission,pilot,requested=verify_scope(scope)
    need(not {row['match_id'] for row in cache['games']}&current_reservations(admission),'Reserved model rows')
    bounds={str(path):digest,**cache['bindings_sha256']}
    for source,want in bounds.items():need(file_hash(source)==want,'Changed cache/source binding')
    expected={row['match_id']:row for row in requested};entries=cache['games']
    need([row['game_index'] for row in entries]==list(range(len(entries))) and len({r['match_id'] for r in entries})==len(entries),
         'Cache game indices/identities differ')
    for i,row in enumerate(entries):
        record=expected[row['match_id']]
        for key in ('role','source_role','patch','game_creation_ms','cohort','scope_index'):need(row[key]==record[key],'Shard metadata differs')
        if i<6000:need(row['match_id']==pilot['games'][i]['match_id'] and row['cohort']=='pilot','Pilot prefix changed')
    sizes={role:sum(row['fit_sampled_frames'] if role=='fit' else row['rows'] for row in entries if row['role']==role)
           for role in ('fit','calibration','development')}
    data={role:{'x':np.empty((n,173),dtype=np.float32),'y':np.empty(n,dtype=np.int8),'game':np.empty(n,dtype=np.int32),
                'timestamp_ms':np.empty(n,dtype=np.int64),'end_ms':np.empty(n,dtype=np.int64),'pilot':np.empty(n,dtype=bool),
                'patch':np.empty(n,dtype='U16')} for role,n in sizes.items()}
    offsets=dict.fromkeys(sizes,0)
    for j,row in enumerate(entries):
        if j%100==0:guard()
        shard=(path.parent/row['file']).resolve(strict=True)
        need(shard.is_relative_to(path.parent) and file_hash(shard)==row['sha256'],'Shard changed or escaped cache directory')
        with np.load(shard,allow_pickle=False) as saved:arrays={name:saved[name] for name in saved.files}
        validate_shard(arrays,row);role=row['role'];take=arrays['fit_selected_local'] if role=='fit' else np.arange(row['rows'])
        count=len(take);lo=offsets[role];hi=lo+count;out=data[role]
        out['x'][lo:hi,:61]=arrays['x_rich'][take];out['x'][lo:hi,61:]=arrays['x_combat'][take]
        out['y'][lo:hi]=arrays['y'][take];out['timestamp_ms'][lo:hi]=arrays['timestamp_ms'][take]
        out['game'][lo:hi]=row['game_index'];out['end_ms'][lo:hi]=row['evaluation_only_end_ms'];out['pilot'][lo:hi]=row['cohort']=='pilot'
        out['patch'][lo:hi]=row['patch'];offsets[role]=hi
    need(offsets==sizes,'Incomplete model row assembly')
    need(int(data['fit']['pilot'].sum())==22554,'Original sampled FIT population changed')
    return data,cache,bounds,admission


def additional_gain(item):
    return item['relative_log_loss_reduction']>=.005 and item['log_loss']['ci95'][0]>0 and item['brier']['estimate']>=0


def select_scale_candidate(comparisons,lead_comparisons,coverage):
    gates={};eligible=[]
    for name in CANDIDATES:
        checks={'minimum_full_fit':coverage['fit_games']>=11000,'minimum_full_calibration':coverage['calibration_games']>=2700,
                'minimum_complement_development':coverage['complement_games']>=1700,'minimum_lead_games':coverage['lead_games']>=1000}
        for reference in REFERENCES:
            overall=comparisons[name+'_vs_'+reference];lead=lead_comparisons.get(name+'_vs_'+reference)
            checks['two_percent_log_loss_vs_'+reference]=overall['relative_log_loss_reduction']>=.02
            checks['positive_log_loss_vs_'+reference]=overall['log_loss']['ci95'][0]>0
            checks['positive_brier_vs_'+reference]=overall['brier']['ci95'][0]>0
            checks['half_percent_lead_loss_vs_'+reference]=lead is not None and lead['relative_log_loss_reduction']>=.005
            checks['positive_lead_loss_vs_'+reference]=lead is not None and lead['log_loss']['ci95'][0]>0
            checks['nonnegative_lead_brier_vs_'+reference]=lead is not None and lead['brier']['estimate']>=0
        gates[name]=checks
        if all(checks.values()):eligible.append(name)
    retained={}
    for family in ('rich','combat'):
        pilot=family+'_pilot_recalibrated';full=family+'_full'
        if pilot in eligible:
            retained[family]=full if full in eligible and additional_gain(comparisons[full+'_vs_'+pilot]) else pilot
        elif full in eligible:retained[family]=full
    if len(retained)==2:
        rich,combat=retained['rich'],retained['combat']
        selected=combat if additional_gain(comparisons[combat+'_vs_'+rich]) else rich
    else:selected=next(iter(retained.values()),None)
    return {'gates':gates,'eligible':eligible,'retained_by_family':retained,'selected_candidate':selected,
            'useful_development_signal':selected is not None,'confirmation_completed':False,'practical_benefit_established':False}
