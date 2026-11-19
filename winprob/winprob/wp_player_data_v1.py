"""Verified bounded assembly of identical global rows and joint-player tokens."""
import json
from pathlib import Path
import numpy as np
from .iql_dataset_contract import file_hash,current_reservations
from .wp_scale_scope_v1 import checked_json,verify_scope
from .wp_scale_models_v1 import validate_shard
from .features_wp_players_v1 import FIELDS


def need(value,message):
    if not value:raise ValueError(message)


def load_player_data(path,guard=lambda:None):
    path=Path(path).resolve(strict=True);manifest,digest=checked_json(path)
    need(manifest['schema']=='wp-joint-player-cache-v1' and manifest['state']=='complete' and manifest['training_authorized']
         and manifest['game_count']==17195 and manifest['frames']==396667 and manifest['feature_names']==list(FIELDS),
         'Complete fixed player cache required')
    parent_path=Path(manifest['parent_manifest_path']).resolve(strict=True);parent,parent_sha=checked_json(parent_path)
    need(parent_sha==manifest['parent_manifest_sha256']=='7a6852da971246e67051b0ad7622b27b1ff20fd7ce8cb43d9fd76ccf41d89c14',
         'Wrong parent global cache')
    scope,scope_sha=checked_json(manifest['scope_path']);need(scope_sha==manifest['scope_sha256'],'Changed player scope')
    admission,pilot,scope_rows=verify_scope(scope)
    need(not {r['match_id'] for r in manifest['games']}&current_reservations(admission),'Reserved player rows')
    bindings={str(path):digest,str(parent_path):parent_sha,**manifest['bindings_sha256']}
    for source,expected in bindings.items():need(file_hash(source)==expected,'Changed player input/code binding')
    token_file=(path.parent/manifest['tensor_file']).resolve(strict=True)
    need(token_file.is_relative_to(path.parent) and file_hash(token_file)==manifest['tensor_sha256'],'Player tensor changed or escaped')
    bindings[str(token_file)]=manifest['tensor_sha256'];tokens=np.load(token_file,mmap_mode='r',allow_pickle=False)
    need(tokens.shape==(396667,10,35) and tokens.dtype==np.float32,'Player tensor shape/dtype changed')
    entries=manifest['games'];old=parent['games'];need(len(entries)==len(old)==17195,'Player game count differs')
    sizes={role:sum(r['fit_sampled_frames'] if role=='fit' else r['rows'] for r in old if r['role']==role)
           for role in ('fit','calibration','development')}
    need(sizes=={'fit':64422,'calibration':66279,'development':66864},'Fixed model row counts differ')
    result={role:{'x':np.empty((n,173),dtype=np.float32),'players':np.empty((n,10,35),dtype=np.float32),
        'y':np.empty(n,dtype=np.int8),'game':np.empty(n,dtype=np.int32),'timestamp_ms':np.empty(n,dtype=np.int64),
        'end_ms':np.empty(n,dtype=np.int64),'pilot':np.empty(n,dtype=bool),'patch':np.empty(n,dtype='U16')}
        for role,n in sizes.items()}
    offsets=dict.fromkeys(sizes,0);global_offset=0
    for gid,(row,original) in enumerate(zip(entries,old,strict=True)):
        if gid%100==0:guard()
        need(row['game_index']==gid and row['offset']==global_offset,'Player row offsets/order differ')
        for key,value in original.items():
            name={'file':'parent_shard_file','sha256':'parent_shard_sha256'}.get(key,key)
            need(row[name]==value,'Player metadata differs from parent')
        shard=(parent_path.parent/original['file']).resolve(strict=True)
        need(shard.is_relative_to(parent_path.parent) and file_hash(shard)==original['sha256'],'Parent shard changed')
        with np.load(shard,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
        validate_shard(a,original);n=row['rows'];role=row['role']
        selected=a['fit_selected_local'] if role=='fit' else np.arange(n);lo=offsets[role];hi=lo+len(selected)
        chunk=tokens[global_offset:global_offset+n]
        need(chunk.shape==(n,10,35) and np.isfinite(chunk).all(),'Invalid player rows')
        need(np.all(chunk[:,:5,-1]==1) and np.all(chunk[:,5:,-1]==-1),'Canonical stored team axes differ')
        dest=result[role];dest['x'][lo:hi,:61]=a['x_rich'][selected];dest['x'][lo:hi,61:]=a['x_combat'][selected]
        dest['players'][lo:hi]=chunk[selected];dest['y'][lo:hi]=a['y'][selected];dest['timestamp_ms'][lo:hi]=a['timestamp_ms'][selected]
        dest['game'][lo:hi]=gid;dest['end_ms'][lo:hi]=row['evaluation_only_end_ms'];dest['patch'][lo:hi]=row['patch']
        dest['pilot'][lo:hi]=row['cohort']=='pilot';offsets[role]=hi;global_offset+=n
    need(offsets==sizes and global_offset==396667 and result['fit']['pilot'].sum()==22554,'Assembled population differs')
    del tokens;guard()
    return result,manifest,parent,bindings,admission
