"""Exact current player measurements before team pooling; retrospective OMNI."""
from collections import Counter
import numpy as np
from .features_rich import _position
from .features_current_combat_v1 import TAGS,STAT_SCALES,player_values,roster_identity,exact_frame_map

VERSION='wp-joint-player-measurements-v1'
POSITION_FIELDS=('position_x','position_y','position_observed')
FIELDS=POSITION_FIELDS+('total_gold_over10000','wallet_over2000','wallet_observed','net_gold_outflow_over10000',
    'health_over2000','health_observed','health_max_over2000','health_max_observed','health_ratio','health_ratio_observed',
    'health_positive','level_over18','level_observed')+tuple(name for field in STAT_SCALES for name in (field+'_scaled',field+'_observed'))+tuple(
    'tag_'+tag for tag in TAGS)+('side_blue_positive',)
assert len(FIELDS)==35


def need(value,message):
    if not value:raise ValueError(message)


def frame_tokens(roster,frame,champion_tags):
    need(len(roster)==10 and len({r[0] for r in roster})==10 and Counter(r[1] for r in roster)=={100:5,200:5},'Ten players/five per side required')
    tokens=[];counts=Counter(player_frames=10)
    for pid,side,champion in sorted(roster,key=lambda row:(row[1],row[0])):
        raw=frame['participantFrames'][str(pid)];p=player_values(pid,champion,raw,champion_tags);position=_position(raw)
        def val(key,scale=1):return p[key]/scale if p[key] is not None else 0.
        row=[*(position if position is not None else (0.,0.)),float(position is not None),p['gold']/10000,
             val('wallet',2000),float(p['wallet'] is not None),(p['gold']-p['wallet'])/10000 if p['wallet'] is not None else 0.,
             val('health',2000),float(p['health'] is not None),val('maximum',2000),float(p['maximum'] is not None),
             val('ratio'),float(p['ratio'] is not None),float(p['positive']),val('level',18),float(p['level'] is not None)]
        for field,scale in STAT_SCALES.items():
            value=p['stats'][field];row.extend((value/scale if value is not None else 0.,float(value is not None)))
        row.extend(float(tag in p['tags'])/len(p['tags']) for tag in TAGS);row.append(1. if side==100 else -1.)
        tokens.append(row);counts['position_observed']+=int(position is not None)
        for key in ('health','maximum','ratio','wallet','level'):counts[key+'_observed']+=int(p[key] is not None)
    result=np.asarray(tokens,dtype=np.float32)
    need(result.shape==(10,35) and np.isfinite(result).all(),'Invalid player tensor')
    need(np.array_equal(result[:5,-1],np.ones(5)) and np.array_equal(result[5:,-1],-np.ones(5)),'Side axis differs')
    return result,counts


def extract_player_tokens(match,timeline,timestamps,champion_tags):
    need(match['metadata']['matchId']==timeline['metadata']['matchId'],'Match/timeline identity differs')
    clock=np.asarray(timestamps);need(clock.ndim==1 and len(clock)>0 and np.issubdtype(clock.dtype,np.integer)
        and np.all(np.diff(clock)>0) and clock.min()>=180000 and clock.max()<=2400000,'Invalid admitted timestamps')
    roster=roster_identity(match);frames=exact_frame_map(timeline);quality=Counter();arrays=[]
    for t in clock:
        need(int(t) in frames,'Missing exact admitted frame; interpolation forbidden')
        values,counts=frame_tokens(roster,frames[int(t)],champion_tags);arrays.append(values);quality.update(counts)
    return np.stack(arrays),quality
