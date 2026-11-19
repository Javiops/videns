"""Synthetic information-loss witness for the frozen173-dimensional WP state."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import numpy as np
from test_wp_scale_cache_v1 import fixture
from winprob.wp_scale_cache_v1 import convert_game
from winprob.features_rich import ALL_RICH_FEATURES
from winprob.features_current_combat_v1 import FEATURES


def proximity_health(frame):
    """Illustrative observed statistic, not damage, win value or an action effect."""
    players=frame['participantFrames'];values={}
    for team,pids,enemies in ((100,range(1,6),range(6,11)),(200,range(6,11),range(1,6))):
        score=0.
        for pid in pids:
            own=players[str(pid)];position=own['position']
            distance=min(math.hypot(position['x']-players[str(other)]['position']['x'],
                                    position['y']-players[str(other)]['position']['y']) for other in enemies)
            health=own['championStats']['health']/own['championStats']['healthMax']
            score+=health*math.exp(-distance**2/(2*2000.**2))
        values[team]=score
    return values[100]-values[200]


def run():
    output=ROOT/'data/reports/wp_spatial_health_collision_v1.json'
    if output.exists():raise FileExistsError('Refuse witness overwrite')
    match,timeline,entry,static=fixture()
    for frame in timeline['info']['frames']:
        for pid,record in frame['participantFrames'].items():
            i=int(pid);record['totalGold']=2000;record['currentGold']=200
            record['championStats']['healthMax']=1000;record['championStats']['health']=800
            record['position']={'x':7400,'y':7400} if i==1 else ({'x':1500,'y':1500} if i==2 else
                {'x':3000+50*i,'y':3000} if i<=5 else {'x':7600+50*i,'y':7400})
        frame['participantFrames']['1']['championStats']['health']=1000
        frame['participantFrames']['2']['championStats']['health']=100
    changed=deepcopy(timeline)
    for frame in changed['info']['frames']:
        frame['participantFrames']['1']['championStats']['health']=100
        frame['participantFrames']['2']['championStats']['health']=1000
    a=convert_game(match,timeline,entry,static)['arrays'];b=convert_game(match,changed,entry,static)['arrays']
    xa=np.column_stack((a['x_rich'],a['x_combat']));xb=np.column_stack((b['x_rich'],b['x_combat']))
    if not np.array_equal(xa,xb):raise AssertionError('Constructed witness is not an exact collision')
    names=list(ALL_RICH_FEATURES)+list(FEATURES);stamps=a['timestamp_ms']
    first={int(f['timestamp']):f for f in timeline['info']['frames']};second={int(f['timestamp']):f for f in changed['info']['frames']}
    sa=[proximity_health(first[int(t)]) for t in stamps];sb=[proximity_health(second[int(t)]) for t in stamps]
    if not all(abs(x-y)>.5 for x,y in zip(sa,sb)):raise AssertionError('Relational statistic did not distinguish the states')
    result={'schema':'wp-spatial-health-information-loss-witness-v1','state':'verified_synthetic_collision',
        'real_game_payload_reads':0,'models_fitted':False,'feature_count':len(names),'equal_feature_rows':len(xa),
        'maximum_absolute_existing_feature_difference':float(np.max(np.abs(xa-xb))),
        'same_feature_bytes_sha256':hashlib.sha256(xa.tobytes()).hexdigest(),'timestamps_ms':stamps.tolist(),
        'state_a':'Blue player1 near red has1000HP; blue player2 far away has100HP; both have1000maxHP and equal resources.',
        'state_b':'Only the1000HP and100HP assignments are exchanged between blue players1 and2.',
        'proximity_health_a':sa,'proximity_health_b':sb,'proximity_kernel_sigma_world_units':2000,
        'claim':'Any deterministic predictor of these existing173 inputs gives identical outputs for these two constructed states.',
        'unproven':['Their true win probabilities differ.','This configuration is frequent in real games.',
                    'Adding this relation improves held-out predictions.','This statistic measures combat power or causal action value.'],
        'next_hypothesis':'Joint player-level health/resource/position relations can retain information that separate fixed summaries discard.',
        'method_references':['https://proceedings.mlr.press/v97/lee19d.html','https://arxiv.org/abs/1706.01427'],
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    output.with_suffix('.sha256').write_text(hashlib.sha256(output.read_bytes()).hexdigest()+' '+output.name+'\n',encoding='ascii')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
