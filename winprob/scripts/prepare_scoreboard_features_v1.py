"""Admit the complete public scoreboard export and prepare fixed controls without targets."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from winprob.scoreboard_features_v1 import ScoreboardArtifact,normalized_scores,permutation_for_state,shuffle_within_team
from winprob.iql_dataset_contract import file_hash


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')


def run(source,output):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    code={str(p.resolve()):file_hash(p) for p in (Path(__file__),ROOT/'winprob/scoreboard_features_v1.py')}
    data=ScoreboardArtifact(source);n=len(data.scores)
    features=np.lib.format.open_memmap(output/'scores.npy',mode='w+',dtype=np.float32,shape=(n,10,3))
    permutations=np.lib.format.open_memmap(output/'permutations.npy',mode='w+',dtype=np.uint8,shape=(n,10))
    changed=0
    for ep in data.episodes:
        for row in range(ep['first_state'],ep['first_state']+ep['states']):
            permutations[row]=permutation_for_state(ep['match_id'],int(ep['participant_id']),int(data.events[row]))
    for lo in range(0,n,8192):
        hi=min(lo+8192,n);features[lo:hi]=normalized_scores(data.scores[lo:hi])
        shuffled=shuffle_within_team(features[lo:hi],permutations[lo:hi])
        changed+=int(np.any(shuffled!=features[lo:hi],axis=(1,2)).sum())
    features.flush();permutations.flush();del features,permutations
    data.revalidate()
    for raw,want in code.items():
        if file_hash(raw)!=want:raise ValueError('preparer code changed')
    manifest={'schema':'scoreboard-feature-cache-v1','state':'complete','states':n,'games':4001,
        'source_manifest_path':str(source),'source_manifest_sha256':data.digest,'bindings_sha256':data.bindings,'code_sha256':code,
        'files':{name:file_hash(output/name) for name in ('scores.npy','permutations.npy')},
        'score_columns':['kills_over_20','deaths_over_20','assists_over_20'],'focal_channel_zeroed':True,
        'shuffle':'whole KDA triples within allies1..4/enemies5..9; original identity hash; focal0 fixed',
        'shuffle_changed_feature_rows':changed,'independent_focal_parity_passed':True,'all_episode_participant_ids_verified':True,
        'training_performed':False,'purchase_target_arrays_opened':False,'outcome_arrays_opened':False,'raw_sources_opened':False,
        'source_runtime_imported':False,'test_payload_reads':False,'confirmation_payload_reads':False,
        'elapsed_s':time.monotonic()-started,'limits':['Feature preparation only; no purchase-accuracy evidence.']}
    write(output/'manifest.json',manifest);print(json.dumps({'state':'complete','states':n,'shuffle_changed_rows':changed,'elapsed_s':manifest['elapsed_s']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-manifest',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.source_manifest.resolve(),args.output.resolve())
