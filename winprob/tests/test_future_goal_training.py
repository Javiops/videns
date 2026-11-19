"""Tiny CPU-only numerical, control, provenance and inference-boundary checks."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_iql_torch import board
from winprob.future_goal_models import BasketTables,FutureGoalActor,set_future_validation,future_supervision_loss
from winprob.future_goal_training import (FutureGoalStore,validate_goal_control,validate_independent_audit,
    verify_optimizer_steps,enable_validated_future_fastpath,shared_epoch_order)
from winprob.iql_training import adam,tree_hash
from train_future_goal import auxiliary_gradient_proof,runtime_projection,changed_heldout_targets,score_model,synthetic_inputs

torch.set_num_threads(1)


def tables():
    ids=torch.tensor([[0,0],[1,0],[2,3]],dtype=torch.long)
    counts=torch.tensor([[0.,0.],[1.,0.],[1.,2.]])
    main=torch.zeros(3,3,23);main[:,0,22]=1.;main[:,1,0]=.2;main[:,2,0]=.8
    goal=main.clone();goal[:,0]=0.;goal[:,2,1]=.7
    return BasketTables(ids,counts,main,torch.ones(3,3,dtype=torch.bool),ids.clone(),counts.clone(),goal,torch.ones(3,3,dtype=torch.bool))


def control_fixture():
    base={'fold':np.array([0,0,1,1]),'game_index':np.arange(4),
          'main_group_id':np.ones(4,dtype=np.int32),'patch_index':np.zeros(4,dtype=np.uint8)}
    arrays={'real_goal_id':np.array([1,2,3,4]),
            'shuffled_goal_id':np.array([[-1,-1,4,3],[2,1,-1,-1]]),
            'goal_donor_state_index':np.array([[-1,-1,3,2],[1,0,-1,-1]])}
    return base,arrays


class FutureGoalTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(19)

    def test_forward_has_no_goal_inputs_and_auxiliary_reaches_served_parameters(self):
        model=FutureGoalActor(20,20,tables());sample=board();patch=torch.tensor([0,1])
        main=model(sample,patch)['main_logits'];with_aux=model(sample,patch,compute_goal=True)
        self.assertTrue(torch.equal(main,with_aux['main_logits']))
        with self.assertRaises(TypeError):model(sample,patch,goal_labels=torch.tensor([1,2]))
        proof=auxiliary_gradient_proof(model,sample,patch,torch.tensor([1,2]))
        self.assertTrue(all(proof.values()))
        self.assertIs(model.state.inventory.embedding,dict(model.named_modules())['state.inventory.embedding'])

    def test_goal_empty_distinct_from_main_save_and_exact_counts(self):
        static=tables();self.assertEqual(static.main_static[0,0,22].item(),1.)
        self.assertEqual(static.goal_static[:,0].count_nonzero().item(),0)
        bad=replace(static,goal_static=static.main_static.clone())
        with self.assertRaisesRegex(ValueError,'future EMPTY'):FutureGoalActor(20,20,bad)
        model=FutureGoalActor(20,20,static)
        with torch.no_grad():model.state.inventory.embedding.weight[3].fill_(2.)
        got=model.state.inventory(static.main_item_indices,static.main_item_counts)
        self.assertTrue(torch.equal(got[2],torch.full((32,),4.)))

    def test_invalid_native_composition_zeroed_after_biased_mlp_without_masking_class(self):
        t=tables();valid=t.main_static_valid.clone();valid[1,2]=False
        model=FutureGoalActor(20,20,replace(t,main_static_valid=valid));sample=board();patch=torch.tensor([0,1])
        with torch.no_grad():model.basket[-1].bias.fill_(3.)
        self.assertTrue(torch.equal(model.basket_vectors('main',1)[2],torch.zeros(128)))
        expected=model.main_free(model.state(sample))[1,2]
        logits=model(sample,patch)['main_logits']
        self.assertEqual(logits.shape,(2,3));self.assertTrue(torch.equal(logits[1,2],expected))
        self.assertGreater(torch.softmax(logits,dim=1)[1,2].item(),0.)

    def test_baseline_never_activates_goal_optimizer_and_rejects_targets(self):
        model=FutureGoalActor(20,20,tables());optimizer=adam(model.parameters());sample=board();patch=torch.tensor([0,1]);labels=torch.tensor([1,2])
        out=model(sample,patch)
        with self.assertRaisesRegex(ValueError,'must not receive'):future_supervision_loss(out,labels,labels)
        with self.assertRaisesRegex(ValueError,'leave the goal head inactive'):
            future_supervision_loss(model(sample,patch,compute_goal=True),labels)
        loss,_,aux=future_supervision_loss(out,labels);self.assertIsNone(aux)
        loss.backward();optimizer.step();receipts=verify_optimizer_steps(model,optimizer,1,'mainonly')
        self.assertEqual(receipts['goal_query.weight'],'inactive_no_state');self.assertEqual(receipts['goal_bias'],'inactive_no_state')
        with self.assertRaises(ValueError):verify_optimizer_steps(model,optimizer,1,'future')

    def test_all_arms_strict_fast_loss_gradient_and_adam_step_exact(self):
        sample=board();patch=torch.tensor([0,1]);labels=torch.tensor([1,2]);goal_labels=torch.tensor([2,1])
        for arm in ('mainonly','future','shuffled'):
            with self.subTest(arm=arm):
                strict=FutureGoalActor(20,20,tables());fast=deepcopy(strict);set_future_validation(fast,False)
                opts=[adam(m.parameters()) for m in (strict,fast)];grads=[];outputs=[];losses=[]
                for model,opt in zip((strict,fast),opts):
                    output=model(sample,patch,compute_goal=arm!='mainonly')
                    loss,_,_=future_supervision_loss(output,labels,None if arm=='mainonly' else goal_labels,use_goal=arm!='mainonly')
                    loss.backward();grads.append({n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()})
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                    verify_optimizer_steps(model,opt,1,arm);outputs.append(output);losses.append(loss)
                self.assertTrue(torch.equal(outputs[0]['main_logits'],outputs[1]['main_logits']))
                self.assertTrue(torch.equal(losses[0],losses[1]))
                for name in grads[0]:
                    if grads[0][name] is None:self.assertIsNone(grads[1][name])
                    else:self.assertTrue(torch.equal(grads[0][name],grads[1][name]),name)
                self.assertEqual(tree_hash(strict.state_dict()),tree_hash(fast.state_dict()))
                self.assertEqual(tree_hash(opts[0].state_dict()),tree_hash(opts[1].state_dict()))

    def test_default_rejects_bad_input_and_fastpath_requires_audited_dataset(self):
        model=FutureGoalActor(20,20,tables());sample=board();bad=replace(sample,global_public=torch.full((2,37),float('nan')))
        with self.assertRaises(ValueError):model(bad,torch.tensor([0,1]))
        with self.assertRaises(ValueError):model(sample,torch.tensor([0,3]))
        with self.assertRaises(ValueError):enable_validated_future_fastpath(model,SimpleNamespace(manifest={}))
        enable_validated_future_fastpath(model,SimpleNamespace(future_validated=True,manifest={'training_authorized':True,'smoke':False}))
        self.assertFalse(model.check_input_contents)

    def test_control_preserves_current_basket_patch_empty_and_train_game_boundaries(self):
        base,arrays=control_fixture();result=validate_goal_control(base,arrays)
        self.assertEqual(result['0']['changed_goal_labels'],2)
        for field,row,value in (('fold',3,0),('game_index',3,2),('main_group_id',3,2),('patch_index',3,1)):
            with self.subTest(field=field):
                bad=deepcopy(base);bad[field][row]=value
                with self.assertRaises(ValueError):validate_goal_control(bad,arrays)
        bad=deepcopy(arrays);bad['real_goal_id'][3]=0
        with self.assertRaises(ValueError):validate_goal_control(base,bad)
        bad=deepcopy(arrays);bad['shuffled_goal_id'][0,0]=1
        with self.assertRaisesRegex(ValueError,'held-out'):validate_goal_control(base,bad)
        bad=deepcopy(arrays);bad['goal_donor_state_index'][0,2]=-1;bad['shuffled_goal_id'][0,2]=3
        self.assertEqual(validate_goal_control(base,bad)['0']['missing_donor_original_retained'],1)
        bad['shuffled_goal_id'][0,2]=4
        with self.assertRaisesRegex(ValueError,'retain the original'):validate_goal_control(base,bad)

    def test_store_refuses_baseline_or_heldout_goal_gathers(self):
        base,arrays=control_fixture();store=FutureGoalStore.__new__(FutureGoalStore)
        store.dataset=SimpleNamespace(arrays=base);store.real_goals=arrays['real_goal_id'];store.shuffled_goals=arrays['shuffled_goal_id']
        store.goal_label_gathers={'mainonly':0,'future':0,'shuffled':0};store.device=torch.device('cpu')
        with self.assertRaisesRegex(ValueError,'baseline'):store.training_goal_labels([2],0,'mainonly')
        with self.assertRaisesRegex(ValueError,'held-out'):store.training_goal_labels([0],0,'future')
        self.assertEqual(store.training_goal_labels([2,3],0,'future').tolist(),[3,4])
        self.assertEqual(store.training_goal_labels([2,3],0,'shuffled').tolist(),[4,3])
        self.assertEqual(store.goal_label_gathers,{'mainonly':0,'future':2,'shuffled':2})

    def test_target_mutation_heldout_only_and_exact_main_output_invariance(self):
        arrays={'game_index':np.array([0,0,1,1]),'timestamp_ms':np.array([100,200,100,200]),'fold':np.array([0,0,1,1])}
        dataset=SimpleNamespace(arrays=arrays,selected=np.array([1,3]),games=[{},{}])
        sample=board();patch=torch.tensor([0,1]);store=SimpleNamespace(device=torch.device('cpu'),main_ids=torch.tensor([1,2,1,2]),
            real_goals=np.array([1,2,1,2]),shuffled_goals=np.array([[-1,-1,1,2],[1,2,-1,-1]]))
        store.inputs=lambda rows:(sample,patch)
        store.main_labels=lambda rows:store.main_ids[torch.as_tensor(rows)]
        altered=changed_heldout_targets(store,dataset,0)
        self.assertEqual(altered[3],1);self.assertEqual(altered[0].tolist(),[1,3,1,2])
        np.testing.assert_array_equal(altered[1],[1,3,1,2]);np.testing.assert_array_equal(altered[2][:,2:],store.shuffled_goals[:,2:])
        old=(store.main_ids,store.real_goals,store.shuffled_goals)
        result,count=score_model(FutureGoalActor(20,20,tables()),store,np.array([1,3]),lambda:None,altered)
        self.assertEqual(count,2);self.assertTrue(np.isfinite(result['target_log_probability']).all())
        for a,b in zip(old,(store.main_ids,store.real_goals,store.shuffled_goals)):self.assertIs(a,b)

    def test_initialization_orders_and_full_recipe_projection(self):
        template=FutureGoalActor(20,20,tables());copies=[FutureGoalActor(20,20,tables()) for _ in range(3)]
        for model in copies:model.load_state_dict(template.state_dict())
        self.assertEqual(len({tree_hash(m.state_dict()) for m in copies}),1)
        self.assertEqual(len({tree_hash(adam(m.parameters()).state_dict()) for m in copies}),1)
        orders=[shared_epoch_order(np.arange(40),0,1) for _ in copies]
        for order in orders[1:]:np.testing.assert_array_equal(order,orders[0])
        fold=np.r_[np.zeros(195611,dtype=np.uint8),np.ones(196837,dtype=np.uint8)]
        selected=np.r_[np.zeros(1673,dtype=np.uint8),np.ones(1677,dtype=np.uint8)]
        arms=('mainonly','future','shuffled');result=runtime_projection(dict.fromkeys(arms,.01),dict.fromkeys(arms,.001),dict.fromkeys(arms,.004),fold,selected)
        self.assertEqual(result['training_updates'],92040);self.assertEqual(result['updates_per_arm'],30680)
        self.assertEqual(result['validation_forwards_per_batch'],2);self.assertEqual(result['work_margin'],1.25)
        self.assertAlmostEqual(result['projected_remaining_s'],1.25*(92040*.011+2*3*14*.005)+180.)
        dataset=SimpleNamespace(inventory_max=4,arrays={'patch_index':np.array([0]*8+[1]*2)})
        state,patch,_,_=synthetic_inputs(dataset,'cpu')
        self.assertEqual(state.inventory_indices.shape,(256,10,4));self.assertEqual(set(patch.tolist()),{0,1})

    def test_independent_audit_requires_exact_artifact_and_all_proofs(self):
        path=Path('manifest.json').resolve();audit={'schema':'future-goal-cache-independent-audit-v1','state':'passed',
            'cache_manifest_path':str(path),'cache_manifest_sha256':'a'*64,'states':392448,'games':4001,'selected_games':3350,
            'goal_classes':258,'reserved_intersections':0,'all_shuffle_donors_and_targets_independently_recomputed':True,
            'all_strict_later_goal_labels_verified':True,'basket_counts_and_full_class_axes_verified':True,'native_static_vectors_recomputed':True,
            'model_loaded_or_fitted':False,'raw_or_database_reads':False,'reward_or_terminal_timing_arrays_opened':False}
        validate_independent_audit(audit,path,'a'*64)
        for key,value in (('cache_manifest_sha256','b'*64),('reserved_intersections',1),('native_static_vectors_recomputed',False),('model_loaded_or_fitted',True)):
            bad=dict(audit);bad[key]=value
            with self.assertRaises(ValueError):validate_independent_audit(bad,path,'a'*64)


if __name__=='__main__':unittest.main()
