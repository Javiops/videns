"""Fixed current-frame set summaries; observed WP, not an intervention model."""
from collections import Counter
import math
import numpy as np

VERSION='observed-current-combat-v1'
TAGS=('Assassin','Fighter','Mage','Marksman','Support','Tank')
STAT_SCALES={'attackDamage':500.,'abilityPower':1000.,'armor':100.,'magicResist':100.,'attackSpeed':100.,'movementSpeed':500.}
HEALTH_FIELDS=('health_observed_fraction','health_positive_fraction')
TEAM_FIELDS=HEALTH_FIELDS+(
    'health_max_observed_fraction','health_ratio_observed_fraction','health_fraction_mean',
    'health_current_sum_over_10000','health_max_sum_over_10000','level_observed_fraction','level_sum_over_90',
    'positive_health_level_sum_over_90','wallet_observed_fraction','wallet_sum_over_10000',
    'net_gold_outflow_sum_over_50000','positive_health_gold_sum_over_50000','positive_health_gold_share',
    'health_weighted_gold_share','positive_health_net_gold_outflow_sum_over_50000',
    'health_weighted_net_gold_outflow_sum_over_50000','health_wallet_observed_fraction','health_ratio_wallet_observed_fraction'
)+tuple(f'{field}_{suffix}' for field in STAT_SCALES for suffix in ('observed_fraction','scaled_sum','positive_health_scaled_sum'))+tuple(
    f'tag_{tag}_{suffix}' for tag in TAGS for suffix in ('roster_fraction','gold_excess_over_50000','health_weighted_net_outflow_over_50000'))
FEATURES=tuple(f'current_{name}_{op}' for name in TEAM_FIELDS for op in ('diff','sum'))
HEALTH_FEATURES=tuple(f'current_{name}_{op}' for name in HEALTH_FIELDS for op in ('diff','sum'))


def require(value,message):
    if not value:raise ValueError(message)


def number(value):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def roster_identity(match):
    rows=[]
    for raw in match['info']['participants']:
        values=[raw[key] for key in ('participantId','teamId','championId')]
        require(all(type(value) is int for value in values),'Noninteger immutable roster identity')
        rows.append(tuple(values))
    require(len(rows)==10 and {r[0] for r in rows}==set(range(1,11)) and Counter(r[1] for r in rows)=={100:5,200:5},
            'Exactly ten canonical participants/five per team required')
    return sorted(rows)


def player_values(pid,champion,raw,tags):
    stats=raw.get('championStats');stats=stats if isinstance(stats,dict) else {}
    gold=number(raw.get('totalGold'));require(gold is not None and gold>=0,'Total gold must match the qualified base frame')
    health=number(stats.get('health'));health=health if health is not None and health>=0 else None
    maximum=number(stats.get('healthMax'));maximum=maximum if maximum is not None and maximum>0 else None
    ratio=health/maximum if health is not None and maximum is not None and health<=maximum else None
    level=number(raw.get('level'));level=level if level is not None and 1<=level<=30 else None
    own_tags=tags.get(str(champion));require(isinstance(own_tags,list) and own_tags and len(set(own_tags))==len(own_tags)
        and set(own_tags)<=set(TAGS),'Missing/unknown native public champion tags')
    return {'pid':pid,'gold':gold,'wallet':number(raw.get('currentGold')),'health':health,'maximum':maximum,
            'ratio':ratio,'level':level,'positive':health is not None and health>0,
            'stats':{key:number(stats.get(key)) for key in STAT_SCALES},'tags':tuple(own_tags)}


def team_summary(players):
    require(len(players)==5,'Five players required');players=sorted(players,key=lambda p:p['pid'])
    def total(values):return math.fsum(value for value in values if value is not None)
    h=sum(p['health'] is not None for p in players);r=sum(p['ratio'] is not None for p in players)
    pos=sum(p['positive'] for p in players);gold=total(p['gold'] for p in players)
    positive_gold=total(p['gold'] for p in players if p['positive'])
    ratio_gold=total(p['gold']*p['ratio'] for p in players if p['ratio'] is not None)
    wallet=[p for p in players if p['wallet'] is not None]
    result={'health_observed_fraction':h/5,'health_positive_fraction':pos/5,
        'health_max_observed_fraction':sum(p['maximum'] is not None for p in players)/5,'health_ratio_observed_fraction':r/5,
        'health_fraction_mean':total(p['ratio'] for p in players)/5,
        'health_current_sum_over_10000':total(p['health'] for p in players)/10000,
        'health_max_sum_over_10000':total(p['maximum'] for p in players)/10000,
        'level_observed_fraction':sum(p['level'] is not None for p in players)/5,
        'level_sum_over_90':total(p['level'] for p in players)/90,
        'positive_health_level_sum_over_90':total(p['level'] for p in players if p['positive'])/90,
        'wallet_observed_fraction':len(wallet)/5,'wallet_sum_over_10000':total(p['wallet'] for p in wallet)/10000,
        'net_gold_outflow_sum_over_50000':total(p['gold']-p['wallet'] for p in wallet)/50000,
        'positive_health_gold_sum_over_50000':positive_gold/50000,
        'positive_health_gold_share':positive_gold/gold if gold else 0.,
        'health_weighted_gold_share':ratio_gold/gold if gold else 0.,
        'positive_health_net_gold_outflow_sum_over_50000':total(p['gold']-p['wallet'] for p in wallet if p['positive'])/50000,
        'health_weighted_net_gold_outflow_sum_over_50000':total((p['gold']-p['wallet'])*p['ratio'] for p in wallet if p['ratio'] is not None)/50000,
        'health_wallet_observed_fraction':sum(p['health'] is not None for p in wallet)/5,
        'health_ratio_wallet_observed_fraction':sum(p['ratio'] is not None for p in wallet)/5}
    for key,scale in STAT_SCALES.items():
        valid=[p for p in players if p['stats'][key] is not None]
        result[key+'_observed_fraction']=len(valid)/5
        result[key+'_scaled_sum']=total(p['stats'][key] for p in valid)/(5*scale)
        result[key+'_positive_health_scaled_sum']=total(p['stats'][key] for p in valid if p['positive'])/(5*scale)
    for tag in TAGS:
        tagged=[p for p in players if tag in p['tags']]
        result[f'tag_{tag}_roster_fraction']=total(1/len(p['tags']) for p in tagged)/5
        result[f'tag_{tag}_gold_excess_over_50000']=total((p['gold']-gold/5)/len(p['tags']) for p in tagged)/50000
        result[f'tag_{tag}_health_weighted_net_outflow_over_50000']=total(
            (p['gold']-p['wallet'])*p['ratio']/len(p['tags']) for p in tagged if p['wallet'] is not None and p['ratio'] is not None)/50000
    require(set(result)==set(TEAM_FIELDS),'Incomplete current-state feature map')
    return result,gold


def frame_features(roster,frame,champion_tags):
    require(len(roster)==10 and len({r[0] for r in roster})==10 and Counter(r[1] for r in roster)=={100:5,200:5},'Roster axis differs')
    teams={100:[],200:[]};quality=Counter(player_frames=10);dynamics={}
    for pid,team,champion in sorted(roster):
        raw=frame['participantFrames'][str(pid)];p=player_values(pid,champion,raw,champion_tags);teams[team].append(p)
        for field,key in (('health','health'),('health_max','maximum'),('health_ratio','ratio'),('level','level'),('wallet','wallet')):
            quality[field+'_observed']+=int(p[key] is not None)
        quality['health_zero']+=int(p['health']==0);quality['wallet_negative']+=int(p['wallet'] is not None and p['wallet']<0)
        for key,value in p['stats'].items():quality['stat_'+key+'_observed']+=int(value is not None)
        dynamics[pid]=(p['health'],p['maximum'])
    blue,bg=team_summary(teams[100]);red,rg=team_summary(teams[200])
    values=np.asarray([value for name in TEAM_FIELDS for value in (blue[name]-red[name],blue[name]+red[name])],dtype=np.float32)
    require(values.shape==(len(FEATURES),) and np.isfinite(values).all(),'Invalid observed-state vector')
    return values,quality,(bg,rg),dynamics


def exact_frame_map(timeline):
    frames=timeline['info']['frames'];mapping={int(frame['timestamp']):frame for frame in frames}
    require(len(mapping)==len(frames),'Duplicate source frame timestamp');return mapping


def evaluation_end_time(timeline,maximum_current_timestamp):
    """This output is evaluation metadata only and is never passed to features."""
    ends=[]
    for frame in timeline['info']['frames']:
        for event in frame.get('events',[]):
            if event.get('type')!='GAME_END':continue
            value=number(event.get('timestamp'))
            if value is None or value<0 or value!=int(value):return -1
            ends.append(int(value))
    if len(set(ends))!=1:return -1
    require(ends[0]>maximum_current_timestamp,'Known game end contradicts a qualified current frame')
    return ends[0]
