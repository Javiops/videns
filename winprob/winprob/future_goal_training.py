"""Strict action-only base plus audited future-supervision cache; no outcome reads."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from winprob import admission
from winprob.history_probe_training import HistoryDataset,bound_json,safe_file,shared_epoch_order
from winprob.iql_training import TensorStore,file_hash,refuse_quarantine
from winprob.iql_features import PublicFeatureCompiler
from winprob.future_goal_models import BasketTables,set_future_validation

SEED=20260930
EPOCHS=20
BATCH_SIZE=256
ARMS=('mainonly','future','shuffled')
DEFINITION_SHA256='4457a015b1d08a0893acfc19b8016b7a7104d98f220f2c893949d055144a9072'
CENSUS_SHA256='367293d55cdeae9aba107055be2eca8af0b4bd03c95de75fcc19dd424624d566'
CACHE_AUDIT_SHA256='6e85d442a9d8b764a6118c5433ce5387ed5b88b974ed15f9730cb93fe0f9a9b5'
DTYPES={'real_goal_id':'int64','shuffled_goal_id':'int64','goal_donor_state_index':'int64','selected_state_indices':'int64',
    'main_item_indices':'int64','main_item_counts':'float32','main_static':'float32','main_static_valid':'bool',
    'goal_item_indices':'int64','goal_item_counts':'float32','goal_static':'float32','goal_static_valid':'bool'}


def validate_independent_audit(audit,manifest_path,manifest_sha):
    if (audit.get('schema')!='future-goal-cache-independent-audit-v1' or audit.get('state')!='passed'
            or Path(audit['cache_manifest_path']).resolve()!=Path(manifest_path).resolve()
            or audit.get('cache_manifest_sha256')!=manifest_sha
            or audit.get('states')!=392448 or audit.get('games')!=4001 or audit.get('selected_games')!=3350
            or audit.get('goal_classes')!=258 or audit.get('reserved_intersections')!=0):
        raise ValueError('independent cache audit does not certify this fixed experiment')
    for name in ('all_shuffle_donors_and_targets_independently_recomputed','all_strict_later_goal_labels_verified',
                 'basket_counts_and_full_class_axes_verified','native_static_vectors_recomputed'):
        if audit.get(name) is not True:raise ValueError('incomplete independent future cache proof')
    for name in ('model_loaded_or_fitted','raw_or_database_reads','reward_or_terminal_timing_arrays_opened'):
        if audit.get(name) is not False:raise ValueError('independent audit crossed its declared boundary')


def validate_goal_control(base,arrays):
    n=len(base['main_group_id']);real=arrays['real_goal_id'];shuffled=arrays['shuffled_goal_id'];donors=arrays['goal_donor_state_index']
    if (real.shape!=(n,) or shuffled.shape!=(2,n) or donors.shape!=(2,n)
            or np.any(real<0) or np.any(real>=258)):
        raise ValueError('fixed258-goal supervision shape/range failed')
    reports={}
    for fold in (0,1):
        train=base['fold']!=fold;held=~train
        if np.any(shuffled[fold,held]!=-1) or np.any(donors[fold,held]!=-1):
            raise ValueError('held-out goals must never receive a training donor/label')
        d=donors[fold];valid=train&(d>=0);missing=train&(d==-1)
        if np.any(d<-1) or np.any(d>=n) or np.any(shuffled[fold,train]<0) or np.any(shuffled[fold,train]>=258):
            raise ValueError('invalid training goal donor/target')
        rows=np.flatnonzero(valid);source=d[rows]
        if (np.any(base['fold'][source]==fold) or np.any(base['game_index'][source]==base['game_index'][rows])
                or np.any(base['main_group_id'][source]!=base['main_group_id'][rows])
                or np.any(base['patch_index'][source]!=base['patch_index'][rows])
                or np.any((real[source]==0)!=(real[rows]==0))
                or not np.array_equal(shuffled[fold,rows],real[source])):
            raise ValueError('goal donor violates fold/game/current-basket/patch/EMPTY matching')
        if not np.array_equal(shuffled[fold,missing],real[missing]):
            raise ValueError('missing goal donor must retain the original target')
        reports[str(fold)]={'training_rows':int(train.sum()),'donor_rows':int(valid.sum()),
            'missing_donor_original_retained':int(missing.sum()),
            'same_identity_donor_rows':int(np.count_nonzero(valid&(shuffled[fold]==real))),
            'changed_goal_labels':int(np.count_nonzero(train&(shuffled[fold]!=real))),
            'training_EMPTY':int(np.count_nonzero(train&(real==0))),
            'training_nonempty':int(np.count_nonzero(train&(real!=0)))}
    return reports


def validate_public_tables(arrays,catalogue,goal_identities,compiler):
    """Verify all candidates from metadata; never mask/remove categorical classes."""
    if len(catalogue['group_main'])!=9333 or len(goal_identities)!=258 or goal_identities[0]!=[]:
        raise ValueError('fixed9333 action /258 goal identities required')
    fallbacks={}
    for kind,baskets in (('main',catalogue['group_main']),('goal',goal_identities)):
        idx=arrays[kind+'_item_indices'];cnt=arrays[kind+'_item_counts'];attrs=arrays[kind+'_static'];valid=arrays[kind+'_static_valid']
        if (idx.ndim!=2 or idx.shape[0]!=len(baskets) or cnt.shape!=idx.shape
                or attrs.shape!=(3,len(baskets),23) or valid.shape!=(3,len(baskets))
                or not np.isfinite(attrs).all() or not np.isfinite(cnt).all()
                or np.any(cnt<0) or not np.array_equal(cnt,np.round(cnt))):raise ValueError('native basket table schema failed')
        for code,basket in enumerate(baskets):
            expected=tuple((compiler.item_index[int(item)],int(count)) for item,count in basket)
            got=tuple((int(i),int(c)) for i,c in zip(idx[code],cnt[code],strict=True) if i)
            if got!=expected or np.any((idx[code]==0)!=(cnt[code]==0)):
                raise ValueError('basket table lost a counted item or changed vocabulary')
            for patch_index,patch in enumerate(('16.17','16.18','16.19')):
                missing=any(str(item) not in compiler.static[patch]['items'] for item,count in basket)
                if missing:
                    if kind=='goal' or valid[patch_index,code] or np.any(attrs[patch_index,code]!=0):
                        raise ValueError('unsupported native composition must be explicit zero, never an invented feature')
                    continue
                if not valid[patch_index,code]:raise ValueError('available native metadata was masked')
                if kind=='goal' and code==0:expected_static=np.zeros(23,dtype=np.float32)
                else:
                    tag='observed_no_buy_death' if not basket else 'retained_net_purchase'
                    _,expected_static=compiler.full_action(basket,tag,patch)
                if not np.array_equal(attrs[patch_index,code],expected_static):
                    raise ValueError('native public basket coordinates changed')
        fallbacks[kind]={patch:int(np.count_nonzero(~valid[i])) for i,patch in enumerate(('16.17','16.18','16.19'))}
    return fallbacks


class FutureGoalDataset(HistoryDataset):
    def __init__(self,directory,protocol_path):
        future_directory=Path(directory).resolve(strict=True);path=future_directory/'manifest.json'
        manifest,sha=bound_json(path)
        if (manifest.get('schema')!='future-goal-dataset-v1' or manifest.get('state')!='complete'
                or manifest.get('training_authorized') is not True or manifest.get('smoke') is not False
                or manifest.get('gates',{}).get('passed') is not True or manifest.get('goal_scope')!='strictly_later'
                or manifest.get('horizon_ms')!=360000
                or manifest.get('shuffle_scope')!=['main_group_id','patch_index','goal_is_EMPTY']):
            raise ValueError('fixed complete strictly-later future-supervision cache required')
        for flag in ('reward_arrays_opened','terminal_timing_arrays_opened','outcomes_analyzed','raw_payload_reads',
                     'source_code_imported','source_payloads_parsed','test_payload_reads','confirmation_payload_reads','training_performed',
                     'auxiliary_labels_are_inference_inputs'):
            if manifest.get(flag) is not False:raise ValueError('future source boundary failed: '+flag)
        protocol=Path(protocol_path).resolve(strict=True)
        if Path(manifest['protocol_path']).resolve()!=protocol or file_hash(protocol)!=manifest['protocol_sha256']:
            raise ValueError('future-supervision protocol changed')
        audit_path=Path(__file__).resolve().parents[1]/'data/reports/future_goal_cache_audit_v1.json'
        audit,audit_sha=bound_json(audit_path)
        if audit_sha!=CACHE_AUDIT_SHA256:raise ValueError('independent cache audit receipt changed')
        validate_independent_audit(audit,path,sha)
        parent=Path(manifest['history_manifest_path']).resolve(strict=True);history,hsha=bound_json(parent)
        if hsha!=manifest['history_manifest_sha256']:raise ValueError('history selection lineage changed')
        # The frozen action-only constructor never opens old reward/terminal arrays.
        super().__init__(parent.parent,history['protocol_path'])
        if (Path(manifest['base_manifest_path']).resolve()!=self.base_directory/'manifest.json'
                or manifest['base_manifest_sha256']!=file_hash(self.base_directory/'manifest.json')
                or Path(manifest['admission_path']).resolve()!=self.admission_path
                or manifest['admission_sha256']!=file_hash(self.admission_path)):
            raise ValueError('auxiliary and base population/admission disagree')
        for name in ('definition','census_report'):
            source=Path(manifest[name+'_path']).resolve(strict=True)
            expected=DEFINITION_SHA256 if name=='definition' else CENSUS_SHA256
            if manifest[name+'_sha256']!=expected or file_hash(source)!=expected:raise ValueError('frozen goal definition/census changed')
            self.bindings[str(source)]=expected
        definition=json.loads(Path(manifest['definition_path']).read_text())
        census=json.loads(Path(manifest['census_report_path']).read_text())
        self.goal_identities=[[]]+[row['goal_counts'] for row in definition['label_vocabulary'][2:]]
        self.future_arrays={}
        if set(manifest['arrays'])!=set(DTYPES):raise ValueError('unexpected future supervision arrays')
        for name,dtype in DTYPES.items():
            spec=manifest['arrays'][name];source=safe_file(future_directory,spec['file']);expected=manifest['files'][spec['file']]
            if file_hash(source)!=expected:raise ValueError('future array hash mismatch')
            value=np.load(source,mmap_mode='r',allow_pickle=False)
            if str(value.dtype)!=dtype or list(value.shape)!=spec['shape']:raise ValueError('future array dtype/shape mismatch')
            self.future_arrays[name]=value;self.bindings[str(source)]=expected
        if manifest.get('state_count')!=self.n or not np.array_equal(self.future_arrays['selected_state_indices'],self.selected):
            raise ValueError('future experiment must retain all old states and selected decisions')
        labels_path=Path(manifest['census_report_path']).resolve().parent/'labels.npz'
        if file_hash(labels_path)!=census['files']['labels.npz']:raise ValueError('bound census labels changed')
        with np.load(labels_path,allow_pickle=False) as labels:
            codes=labels['strictly_later_label_id']
        if (codes.shape!=(self.n,) or np.any(codes<0) or np.any(codes>=259)
                or not np.array_equal(self.future_arrays['real_goal_id'],np.where(codes<2,0,codes-1))):
            raise ValueError('real goal supervision is not the exact strict-later EMPTY merge')
        self.bindings[str(labels_path)]=census['files']['labels.npz'];del codes
        for name,expected in manifest['files'].items():
            source=safe_file(future_directory,name)
            if file_hash(source)!=expected:raise ValueError('future artifact changed')
            self.bindings[str(source)]=expected
        timing_allowed={}
        for name in ('terminated','elapsed_ms'):
            filename=self.base_manifest['arrays'][name]['file'];source=(self.base_directory/filename).resolve()
            expected=self.base_manifest['files'][filename]
            if census['bindings_sha256'].get(str(source))==expected:timing_allowed[str(source)]=expected
        self.timing_ancestor_bytes_hashed=[]
        for source,expected in manifest['bindings'].items():
            source=Path(source).resolve(strict=True);refuse_quarantine(source)
            if source.name in ('reward.npy','next_state_index.npy') or 'raw_match_v5' in source.parts:
                raise ValueError('future trainer refuses raw/reward/next-state reads')
            if source.name in ('terminated.npy','elapsed_ms.npy'):
                if timing_allowed.get(str(source))!=expected:raise ValueError('timing hash is not the exact census/base ancestor')
                self.timing_ancestor_bytes_hashed.append(str(source))
            if file_hash(source)!=expected:raise ValueError('future source binding changed')
            key=str(source)
            if key in self.bindings and self.bindings[key]!=expected:raise ValueError('conflicting ancestor digest')
            self.bindings[key]=expected
        static_path=Path(manifest['static_export_path']).resolve(strict=True)
        if file_hash(static_path)!=manifest['static_export_sha256'] or manifest['static_export_sha256']!=self.base_manifest['static_export_sha256']:
            raise ValueError('native public static lineage changed')
        compiler=PublicFeatureCompiler(json.loads(static_path.read_text())['patches'])
        if json.loads(json.dumps(compiler.metadata()))!=self.feature_metadata:raise ValueError('future and state vocabularies differ')
        self.native_fallbacks=validate_public_tables(self.future_arrays,self.catalogue,self.goal_identities,compiler)
        a=self.arrays
        if not self.future_arrays['main_static_valid'][a['patch_index'],a['main_group_id']].all():
            raise ValueError('observed target lacks native attributes; cannot drop rows')
        self.goal_control_audit=validate_goal_control(a,self.future_arrays)
        vocabulary=json.loads(safe_file(future_directory,manifest['goal_vocabulary_file']).read_text())
        if [row['goal_counts'] for row in vocabulary]!=self.goal_identities or [row['goal_id'] for row in vocabulary]!=list(range(258)):
            raise ValueError('goal table/class identity ordering changed')
        self.bindings.update({str(path):sha,str(path.with_suffix('.sha256')):file_hash(path.with_suffix('.sha256')),
                              str(protocol):file_hash(protocol),str(static_path):file_hash(static_path),
                              str(audit_path):audit_sha,str(audit_path.with_suffix('.sha256')):file_hash(audit_path.with_suffix('.sha256'))})
        self.independent_cache_audit={'path':str(audit_path),'sha256':audit_sha,'state':'passed'}
        self.future_directory=future_directory;self.history_manifest=self.manifest;self.manifest=manifest;self.future_validated=True

    def cpu_tables(self):
        return BasketTables(**{name:torch.from_numpy(np.array(self.future_arrays[name],copy=True)) for name in BasketTables.__dataclass_fields__})


class FutureGoalStore:
    def __init__(self,dataset,device,guard):
        self.dataset=dataset;self.device=torch.device(device);self.base=TensorStore(dataset,device,guard)
        if not self.base.preloaded:raise RuntimeError('fixed goal recipe requires resident validated CUDA states')
        a=dataset.arrays
        self.main_ids=torch.tensor(np.asarray(a['main_group_id']),dtype=torch.long,device=device)
        self.patches=torch.tensor(np.asarray(a['patch_index']),dtype=torch.long,device=device)
        # Goal labels remain CPU mappings; baseline and inference never gather them.
        self.real_goals=dataset.future_arrays['real_goal_id'];self.shuffled_goals=dataset.future_arrays['shuffled_goal_id']
        self.goal_label_gathers={arm:0 for arm in ARMS}
        torch.cuda.synchronize(device);guard()

    def inputs(self,rows):
        index=torch.as_tensor(np.asarray(rows),dtype=torch.long,device=self.device)
        return self.base.states(rows),self.patches[index]

    def main_labels(self,rows):
        return self.main_ids[torch.as_tensor(np.asarray(rows),dtype=torch.long,device=self.device)]

    def training_goal_labels(self,rows,heldout,arm):
        if arm=='mainonly':raise ValueError('baseline must not read future labels in its loss path')
        if arm not in ('future','shuffled') or heldout not in (0,1):raise ValueError('fixed supervised arm/fold required')
        rows=np.asarray(rows,dtype=np.int64)
        if np.any(self.dataset.arrays['fold'][rows]==heldout):raise ValueError('held-out future labels reached fitting')
        values=self.real_goals[rows] if arm=='future' else self.shuffled_goals[heldout,rows]
        if np.any(values<0) or np.any(values>=258):raise ValueError('invalid supervised goal identity')
        self.goal_label_gathers[arm]+=len(rows)
        return torch.as_tensor(np.array(values),dtype=torch.long,device=self.device)


def enable_validated_future_fastpath(model,dataset):
    if (getattr(dataset,'future_validated',False) is not True or not dataset.manifest.get('training_authorized')
                or dataset.manifest.get('smoke') is not False):raise ValueError('validated future cache required before trusted execution')
    set_future_validation(model,False)


def verify_optimizer_steps(model,optimizer,expected,arm):
    inactive={'goal_query.weight','goal_bias'} if arm=='mainonly' else set()
    evidence={}
    for name,parameter in model.named_parameters():
        value=optimizer.state.get(parameter)
        if name in inactive:
            if value or parameter.grad is not None:raise ValueError('baseline updated an inactive goal-only parameter')
            evidence[name]='inactive_no_state'
        else:
            if not value or int(value['step'])!=expected:raise ValueError('active parameter step count mismatch: '+name)
            evidence[name]=expected
    return evidence
