"""Single-game scale-cache conversion using unchanged feature definitions."""
from collections import Counter
import numpy as np
from .cache import _validate_match_metadata
from .features_rich import extract_rich_game_states,ALL_RICH_FEATURES
from .features_current_combat_v1 import (FEATURES,roster_identity,frame_features,exact_frame_map,evaluation_end_time)


def need(value,message):
    if not value:raise ValueError(message)


def convert_game(match,timeline,entry,patch_static):
    _validate_match_metadata(match,entry)
    rows=extract_rich_game_states(match,timeline)
    if not rows:return None
    clock=np.asarray([row['timestamp_ms'] for row in rows],dtype=np.int64)
    need(len(np.unique(clock))==len(clock) and np.all(np.diff(clock)>0),'Qualified frame clock is not strictly ordered')
    need(all(row['game_id']==entry['match_id'] and row['patch']==entry['patch'] for row in rows),'Frame identity differs')
    starts={row['game_start_ms'] for row in rows};need(len(starts)==1,'Game start differs across frames')
    rich=np.asarray([[row['x'][name] for name in ALL_RICH_FEATURES] for row in rows],dtype=np.float32)
    labels=np.asarray([row['y'] for row in rows],dtype=np.int8)
    need(set(labels.tolist())<={0,1} and len(np.unique(labels))==1 and np.isfinite(rich).all(),'Invalid game outcome/features')
    roster=roster_identity(match);frames=exact_frame_map(timeline);combat=np.empty((len(rows),len(FEATURES)),dtype=np.float32)
    health_series={pid:set() for pid,team,cid in roster};quality=Counter()
    gold_diff=ALL_RICH_FEATURES.index('gold_diff');gold_total=ALL_RICH_FEATURES.index('rich_gold_total_sum')
    for index,timestamp in enumerate(clock):
        values,q,gold,health=frame_features(roster,frames[int(timestamp)],patch_static['champion_tags'])
        need(np.float32(gold[0]-gold[1])==rich[index,gold_diff] and np.float32(sum(gold))==rich[index,gold_total],
             'Rich/combat gold mismatch')
        combat[index]=values;quality.update(q)
        for pid,pair in health.items():
            if all(value is not None for value in pair):health_series[pid].add(pair)
    ending=evaluation_end_time(timeline,int(clock.max()))
    return {'arrays':{'x_rich':rich,'x_combat':combat,'y':labels,'timestamp_ms':clock},
            'rows':len(rows),'game_start_ms':int(next(iter(starts))),'evaluation_only_end_ms':ending,
            'quality':dict(quality),'dynamics':{'participant_series':10,'varying_health_or_maximum':sum(len(value)>1 for value in health_series.values())}}


def verify_pilot_arrays(actual,rich,combat,labels,timestamps):
    for key,expected in (('x_rich',rich),('x_combat',combat),('y',labels),('timestamp_ms',timestamps)):
        need(actual[key].dtype==expected.dtype and np.array_equal(actual[key],expected),'Original pilot bytes/rows changed: '+key)
