"""Bounded synthetic CPU regressions; no artifact payload or real fit."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from winprob.history_probe_models import HistoryProbeActor,HistoryBatch,main_action_loss,set_history_validation
from winprob.history_probe_training import (validate_history_arrays,verify_selection,HistoryTensorStore,
    shared_epoch_order,total_training_steps,enable_validated_history_fastpath)
from winprob.history_probe_data import make_selection,history_slots,focal_pid
from winprob.iql_training import fold_for_game,adam,tree_hash
from test_iql_torch import board

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_history_probe import runtime_projection,changed_selected_future_labels,score_model,prediction_receipts
torch.set_num_threads(1)


def fixture():
    found={}
    for i in range(100):
        name=f'HISTORY_SYNTH_{i}';found.setdefault(fold_for_game(name),name)
        if len(found)==2:break
    mids=sorted(found.values(),reverse=True)
    games=[{'game_index':i,'match_id':mid,'fold':fold_for_game(mid),'patch':'16.17'} for i,mid in enumerate(mids)]
    episodes=[{'first_state':i*4,'states':4,'game_index':i,'match_id':g['match_id'],'participant_id':focal_pid(g['match_id']),
               'fold':g['fold'],'patch':'16.17','trajectory_id':g['match_id']+':focal'} for i,g in enumerate(games)]
    a={'game_index':np.repeat(np.arange(2),4).astype(np.int32),'episode_index':np.repeat(np.arange(2),4).astype(np.int32),
       'fold':np.repeat([g['fold'] for g in games],4).astype(np.uint8),'timestamp_ms':np.array([100,150,200,300,400,450,500,600],dtype=np.int64),
       'full_action_id':np.array([1,0,2,3,3,0,2,1],dtype=np.int32),'champion_indices':np.tile(np.arange(1,11),(8,1)).astype(np.int32),
       'patch_index':np.zeros(8,dtype=np.uint8)}
    ends=np.array([140,170,300,310,430,470,600,610],dtype=np.int64)
    previous=np.array([[-1,-1],[0,-1],[0,-1],[-1,0],[-1,-1],[4,-1],[4,-1],[-1,4]],dtype=np.int64)
    present=previous>=0;safe=np.maximum(previous,0)
    ages=np.where(present,(a['timestamp_ms'][:,None]-ends[safe])/360000.,0).astype(np.float32)
    h={'prev_state_index':previous,'history_present':present,'history_age':ages,'visit_end_ms':ends,
       'donor_history_state_index':np.full((2,8),-1,dtype=np.int64),'selected_state_indices':np.array([3,7],dtype=np.int64)}
    return a,h,games,episodes


def cpu_store(a,h):
    store=object.__new__(HistoryTensorStore);store.device=torch.device('cpu')
    ids=torch.tensor([[0,0],[1,0],[2,0],[1,2]],dtype=torch.long)
    counts=torch.tensor([[0.,0.],[1.,0.],[1.,0.],[1.,2.]])
    statics=torch.zeros(3,4,23);statics[:,0,22]=1.;statics[:,1,0]=.1;statics[:,2,0]=.2;statics[:,3,0]=.4
    store.base=SimpleNamespace(action_indices=ids,action_counts=counts,action_static=statics,states=lambda rows:board(len(rows)))
    store.full_ids=torch.tensor(a['full_action_id'],dtype=torch.long);store.main_ids=store.full_ids%3
    store.patches=torch.tensor(a['patch_index'],dtype=torch.long);store.previous=torch.tensor(h['prev_state_index'])
    store.presence=torch.tensor(h['history_present']);store.ages=torch.tensor(h['history_age']);store.donors=torch.tensor(h['donor_history_state_index'])
    return store


class HistoryTrainingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(127)

    def test_save_removed_before_adjacency_and_tie_does_not_replace_purchase(self):
        a,h,games,episodes=fixture();report=validate_history_arrays(a,h,games,episodes)
        self.assertEqual(report['states'],8)
        np.testing.assert_array_equal(h['prev_state_index'][2],[0,-1])
        np.testing.assert_array_equal(h['prev_state_index'][3],[-1,0])
        bad=deepcopy(h);bad['prev_state_index'][2,0]=1
        with self.assertRaises(ValueError):validate_history_arrays(a,bad,games,episodes)
        bad=deepcopy(h);bad['prev_state_index'][3]=[0,-1]
        with self.assertRaises(ValueError):validate_history_arrays(a,bad,games,episodes)

    def test_age_uses_closure_not_start_and_donor_cannot_cross_training_boundary(self):
        a,h,games,episodes=fixture();bad=deepcopy(h)
        bad['history_age'][2,0]=(200-100)/360000.
        with self.assertRaises(ValueError):validate_history_arrays(a,bad,games,episodes)
        fold=int(a['fold'][7]);h['donor_history_state_index'][fold,7]=3
        validate_history_arrays(a,h,games,episodes)
        h['donor_history_state_index'][fold,7]=7
        with self.assertRaises(ValueError):validate_history_arrays(a,h,games,episodes)

    def test_selection_ledger_can_be_sorted_differently_from_base_games(self):
        a,h,games,episodes=fixture();entries=make_selection(games,episodes,a['timestamp_ms'])
        self.assertNotEqual([g['match_id'] for g in games],[e['match_id'] for e in entries])
        selected=np.array([e['selected_state_index'] for e in entries],dtype=np.int64)
        verify_selection(entries,selected,a,games,episodes)
        entries[0]['selected_state_index']+=1
        with self.assertRaises(ValueError):verify_selection(entries,selected,a,games,episodes)

    def test_shuffled_content_keeps_recipient_age_and_masks_even_missing_donor(self):
        a,h,_,_=fixture();fold=int(a['fold'][7]);h['donor_history_state_index'][fold,7]=3;store=cpu_store(a,h)
        _,real,_=store.batch(np.array([7]),fold,'history');_,shuffled,_=store.batch(np.array([7]),fold,'shuffled')
        self.assertTrue(torch.equal(real.age,shuffled.age));self.assertTrue(torch.equal(real.present,shuffled.present))
        self.assertFalse(torch.equal(real.item_indices,shuffled.item_indices))
        store.donors[fold,7]=-1;_,missing,_=store.batch(np.array([7]),fold,'shuffled')
        self.assertTrue(torch.equal(real.age,missing.age));self.assertTrue(torch.equal(real.present,missing.present))
        self.assertEqual(int(torch.count_nonzero(missing.item_indices)),0)

    def test_current_and_later_recipient_targets_cannot_change_features_or_prediction(self):
        a,h,_,_=fixture();store=cpu_store(a,h);actor=HistoryProbeActor(20,20,class_count=3)
        first,history,_=store.batch(np.array([2]),0,'history');prediction=actor(first,history)
        store.full_ids[2:4]=0
        second,changed,_=store.batch(np.array([2]),0,'history')
        self.assertTrue(torch.equal(history.item_indices,changed.item_indices));self.assertTrue(torch.equal(prediction,actor(second,changed)))

    def test_shared_embedding_receives_history_gradient_and_is_not_duplicated(self):
        a,h,_,_=fixture();store=cpu_store(a,h);actor=HistoryProbeActor(20,20,class_count=3)
        state,history,y=store.batch(np.array([7]),0,'history');main_action_loss(actor(state,history),y).backward()
        self.assertGreater(int(torch.count_nonzero(actor.state.inventory.embedding.weight.grad[2])),0)
        self.assertEqual(sum(name.endswith('inventory.embedding.weight') for name,_ in actor.named_parameters()),1)

    def test_all_three_arms_have_exact_validated_fastpath_gradient_adam_parity(self):
        a,h,_,_=fixture();store=cpu_store(a,h);initial=HistoryProbeActor(20,20,class_count=3)
        for arm in ('state','history','shuffled'):
            strict,fast=deepcopy(initial),deepcopy(initial);set_history_validation(fast,False)
            opts=[adam(model.parameters()) for model in (strict,fast)]
            state,history,y=store.batch(np.array([2,7]),0,arm)
            out1,out2=strict(state,history),fast(state,history);self.assertTrue(torch.equal(out1,out2))
            l1,l2=main_action_loss(out1,y),main_action_loss(out2,y);self.assertTrue(torch.equal(l1,l2))
            l1.backward();l2.backward()
            for p,q in zip(strict.parameters(),fast.parameters(),strict=True):self.assertTrue(torch.equal(p.grad,q.grad))
            for model,opt in zip((strict,fast),opts,strict=True):
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
            self.assertEqual(tree_hash(strict.state_dict()),tree_hash(fast.state_dict()))
            self.assertEqual(tree_hash(opts[0].state_dict()),tree_hash(opts[1].state_dict()))

    def test_fresh_initialization_and_batch_orders_are_arm_independent(self):
        torch.manual_seed(20261030);template=HistoryProbeActor(20,20,class_count=3)
        expected=tree_hash(template.state_dict());models=[HistoryProbeActor(20,20,class_count=3) for _ in range(3)]
        for model in models:model.load_state_dict(template.state_dict());self.assertEqual(tree_hash(model.state_dict()),expected)
        with torch.no_grad():next(models[0].parameters()).add_(1)
        self.assertEqual(tree_hash(models[1].state_dict()),expected);self.assertEqual(tree_hash(template.state_dict()),expected)
        x=np.arange(513);first=shared_epoch_order(x,0,0)
        self.assertTrue(np.array_equal(first,shared_epoch_order(x,0,0)));self.assertFalse(np.array_equal(first,shared_epoch_order(x,0,1)))
        np.testing.assert_array_equal(np.sort(first),x)

    def test_trusted_switch_and_history_save_nonfinite_guards(self):
        a,h,_,_=fixture();store=cpu_store(a,h);actor=HistoryProbeActor(20,20,class_count=3)
        with self.assertRaises(ValueError):enable_validated_history_fastpath(actor,SimpleNamespace(validated=False,manifest={}))
        state,history,y=store.batch(np.array([2]),0,'history');history.static_features[0,0,22]=1
        with self.assertRaises(ValueError):actor(state,history)
        with self.assertRaises(ValueError):main_action_loss(torch.tensor([[0.,float('nan'),1.]]),torch.tensor([0]))

    def test_projection_counts_all_arms_partial_batches_validation_and_overhead(self):
        fold=np.array([0]*195611+[1]*196837);selected=np.array([0]*1673+[1]*1677)
        r=runtime_projection(dict.fromkeys(('state','history','shuffled'),.01),dict.fromkeys(('state','history','shuffled'),.001),
            dict.fromkeys(('state','history','shuffled'),.02),fold,selected)
        self.assertEqual(r['training_updates'],92040);self.assertEqual(total_training_steps(fold),92040)
        self.assertEqual(r['selected_validation_batches_per_arm'],14)
        self.assertAlmostEqual(r['projected_remaining_s'],1.25*(92040*.011+2*42*.021)+180)

    def test_selected_future_mutation_preserves_training_donors_and_all_logits(self):
        a,h,games,episodes=fixture();fold=int(a['fold'][2]);store=cpu_store(a,h)
        selected=np.array([2,6]);dataset=SimpleNamespace(arrays=a,selected=selected,games=games)
        altered,count=changed_selected_future_labels(store,dataset,fold)
        self.assertEqual(count,2)
        self.assertTrue(torch.equal(altered[torch.from_numpy(a['fold']!=fold)],store.full_ids[torch.from_numpy(a['fold']!=fold)]))
        original=store.full_ids
        actor=HistoryProbeActor(20,20,class_count=3)
        for arm in ('state','history','shuffled'):
            summary,checked=score_model(actor,store,np.array([2]),fold,arm,lambda:None,altered)
            self.assertEqual(checked,1);self.assertEqual(summary['top3'].shape,(1,3));self.assertIs(store.full_ids,original)

    def test_prediction_receipts_bind_every_arm_label_and_identity(self):
        summary={'target_log_probability':np.array([-1.,-2.]),'top1':np.array([0,2]),'top3':np.array([[0,1,2],[2,1,0]])}
        scores={arm:deepcopy(summary) for arm in ('state','history','shuffled','frequency')}
        args=(['g1','g2'],np.array([10,20]),np.array([0,1]),np.array([0,2]),['observed_no_buy_death','other_purchase'])
        first=prediction_receipts(*args,scores)
        scores['shuffled']['target_log_probability'][0]=-.9
        second=prediction_receipts(*args,scores)
        self.assertNotEqual(first[0]['row_sha256'],second[0]['row_sha256']);self.assertEqual(first[1]['row_sha256'],second[1]['row_sha256'])
        scores['history']['target_log_probability'][0]=np.nan
        with self.assertRaises(ValueError):prediction_receipts(*args,scores)


if __name__=='__main__':unittest.main()
