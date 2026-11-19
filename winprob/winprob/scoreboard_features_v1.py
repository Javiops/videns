"""Admit versioned public scoreboard features; never import source runtime."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from winprob import admission
from winprob.iql_dataset_contract import current_reservations,file_hash

BASE_SHA='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea'
HISTORY_SHA='1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b'


def require(ok,message):
    if not ok:raise ValueError(message)


def read_bound(path,bindings):
    path=Path(path).resolve(strict=True);sha=file_hash(path)
    require(path.with_suffix('.sha256').read_text().split()==[sha,path.name],'manifest sidecar mismatch')
    bindings[str(path)]=sha;return json.loads(path.read_text(encoding='utf-8')),sha


def normalized_scores(raw):
    x=np.asarray(raw)
    require(x.ndim==3 and x.shape[1:]==(10,3) and np.issubdtype(x.dtype,np.integer) and np.all(x>=0),'nonnegative integer ten-player KDA required')
    result=(x.astype(np.float64)/20.).astype(np.float32)
    result[:,0,:]=0.
    return result


def permutation_for_state(match_id,participant_id,event_index):
    require(isinstance(match_id,str) and bool(match_id) and type(participant_id) is int and 1<=participant_id<=10
            and type(event_index) is int and event_index>=0,'original state identity required')
    seed=int.from_bytes(hashlib.sha256(f'scoreboard-v1/shuffle/{match_id}|{participant_id}|{event_index}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed);result=np.arange(10,dtype=np.uint8)
    result[1:5]=rng.permutation(result[1:5]);result[5:]=rng.permutation(result[5:]);return result


def shuffle_within_team(features,permutations):
    x=np.asarray(features);p=np.asarray(permutations)
    require(x.ndim==3 and x.shape[1:]==(10,3) and p.shape==x.shape[:2] and np.issubdtype(p.dtype,np.integer),'aligned scoreboard/permutation arrays required')
    require(np.all(p[:,0]==0) and np.array_equal(np.sort(p[:,1:5],axis=1),np.broadcast_to(np.arange(1,5),p[:,1:5].shape))
            and np.array_equal(np.sort(p[:,5:],axis=1),np.broadcast_to(np.arange(5,10),p[:,5:].shape)),'permutation crosses teams/focal slot')
    require(np.isfinite(x).all() and np.all(x[:,0,:]==0),'focal channel must be masked before use')
    return x[np.arange(len(x))[:,None],p].copy()


class ScoreboardArtifact:
    def __init__(self,manifest_path):
        self.path=Path(manifest_path).resolve(strict=True);self.bindings={}
        m,sha=read_bound(self.path,self.bindings);self.manifest=m;self.digest=sha
        require(m.get('schema')=='winprob-scoreboard-v1' and m.get('state')=='complete' and m.get('states')==392448 and m.get('games')==4001,'complete original scoreboard release required')
        for key in ('raw_archive_reads','whole_database_scanned','test_payload_reads','confirmation_payload_reads','development_game_payload_reads',
                    'calibration_game_payload_reads','training_performed','consumer_runtime_imported','purchase_labels_used_or_analyzed','outcomes_used_or_analyzed'):
            require(m.get(key) is False,'source boundary failed: '+key)
        require(m.get('all_state_identities_matched') is True and m.get('focal_score_mismatches')==m.get('roster_mismatches')==0
                and m.get('mutated_example_games')==4001,'source coverage/parity/mutation gates failed')
        bp=Path(m['base_manifest_path']).resolve(strict=True);base,bsha=read_bound(bp,self.bindings)
        hp=Path(m['history_manifest_path']).resolve(strict=True);history,hsha=read_bound(hp,self.bindings)
        require(bsha==m['base_manifest_sha256']==BASE_SHA and hsha==m['history_manifest_sha256']==HISTORY_SHA,'original cache lineage changed')
        ap=Path(base['source_admission_path']);admitted,ash=read_bound(ap,self.bindings)
        require(ash==base['source_admission_sha256'] and ap.resolve()==Path(m['admission_path']).resolve(),'admission lineage mismatch')
        admission.verify_manifest_sources(admitted);forbidden=current_reservations(admitted)
        games=base['game_metadata'];ids={g['match_id'] for g in games};lookup={g['match_id']:g for g in admitted['games']}
        require(len(ids)==4001 and not ids&forbidden and all(lookup[mid]['role']=='train' and lookup[mid]['historical_validation'] is False for mid in ids),'non-FIT/reserved game before feature reads')
        # Admission precedes all feature-file reads, including checksum reads.
        for raw,want in {**m['bindings_sha256'],**m['code_sha256']}.items():
            require(file_hash(raw)==want,'source input/code changed: '+raw);self.bindings[raw]=want
        for name,want in m['files'].items():
            path=(self.path.parent/name).resolve(strict=True);require(path.is_relative_to(self.path.parent),'source member escaped directory')
            require(file_hash(path)==want,'source feature/member changed');self.bindings[str(path)]=want
        selection=json.loads((self.path.parent/'selection.json').read_text())
        ledger=json.loads((self.path.parent/'game_ledger.json').read_text())
        require(set(selection['game_ids'])==ids and len(ledger)==4001 and {g['match_id'] for g in ledger}==ids,'incomplete source game ledger')
        def base_member(name):
            path=bp.parent/name;require(file_hash(path)==base['files'][name],'base member changed');self.bindings[str(path)]=base['files'][name];return path
        self.episodes=json.loads(base_member(base['episode_metadata_file']).read_text());feature_meta=json.loads(base_member(base['feature_metadata_file']).read_text())
        globals_=np.load(base_member(base['arrays']['global_public']['file']),mmap_mode='r',allow_pickle=False)
        ep_counts={mid:0 for mid in ids};self.scores=np.load(self.path.parent/'player_kda.npy',mmap_mode='r',allow_pickle=False)
        self.pids=np.load(self.path.parent/'participant_ids.npy',mmap_mode='r',allow_pickle=False)
        require(self.scores.dtype==np.int32 and self.scores.shape==(392448,10,3) and self.pids.dtype==np.uint8 and self.pids.shape==(392448,10),'wrong score/identity schema')
        require(np.all(self.scores>=0) and np.array_equal(np.sort(self.pids,axis=1),np.broadcast_to(np.arange(1,11),self.pids.shape)),'invalid KDA/participant values')
        columns=[feature_meta['global_names'].index('own_'+k+'_over_20') for k in ('kills','deaths','assists')]
        focal=(self.scores[:,0,:].astype(np.float64)/20.).astype(np.float32)
        require(np.array_equal(focal,globals_[:,columns]),'independent focal-score parity failed')
        cursor=0
        for episode in self.episodes:
            lo=episode['first_state'];hi=lo+episode['states'];mid=episode['match_id']
            require(lo==cursor and hi<=392448 and np.all(self.pids[lo:hi,0]==episode['participant_id']),'focal episode identity/range changed')
            ep_counts[mid]+=episode['states'];cursor=hi
        require(cursor==392448 and all(row['matched_states']==ep_counts[row['match_id']] for row in ledger),'source row counts differ from complete episodes')
        ep=hp.parent/history['arrays']['source_event_index']['file'];require(file_hash(ep)==history['files'][ep.name],'event identity file changed')
        self.bindings[str(ep)]=history['files'][ep.name];self.events=np.load(ep,mmap_mode='r',allow_pickle=False)
        self.base=base;self.history=history;self.admission=admitted;self.games=games

    def revalidate(self):
        for path,want in self.bindings.items():require(file_hash(path)==want,'binding changed while preparing features')
        require(not {g['match_id'] for g in self.games}&current_reservations(self.admission),'FIT game became reserved')
