"""No real artifacts: native-wallet threshold and fixed-scope contract tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import diagnose_bc20_ordinary_feasibility_v1 as diagnostic


class FeasibilityTests(unittest.TestCase):
    def test_wallet_sensitive_finds_integer_boundary_without_mutating_source(self):
        row={'gold_est':150.5,'inventory':[1001],'role':'TOP'};before=deepcopy(row)
        def purchase(action,context,wallet,dragon):
            self.assertEqual(context['gold_est'],wallet)
            self.assertEqual(context['inventory'],row['inventory'])
            return [{'item_id':item,'cost':125} for item in action] if wallet>=250 else None
        result=diagnostic.diagnose_action((2003,2003),row,purchase,None)
        self.assertEqual(result['classification'],'wallet_sensitive')
        self.assertEqual(result['minimum_feasible_integer_wallet'],250)
        self.assertEqual(result['minimum_integer_wallet_shortfall'],99.5)
        self.assertEqual(result['minimum_wallet_receipt']['incremental_cost'],250)
        self.assertIsNone(result['lower_integer_wallet_receipt'])
        self.assertEqual(row,before)
        self.assertLessEqual(result['native_economic_calls'],24)

    def test_structural_remains_unresolved_and_never_changes_inventory(self):
        observed=[]
        def blocked(action,row,wallet,dragon):observed.append((row['inventory'],wallet));return None
        result=diagnostic.diagnose_action((3006,),{'gold_est':250,'inventory':[1001]*6},blocked,None)
        self.assertEqual(result['classification'],'structural_or_unresolved_at_generous_wallet')
        self.assertIsNone(result['minimum_feasible_integer_wallet'])
        self.assertEqual(len(observed),2);self.assertEqual(observed[1],([1001]*6,1000000.0))

    def test_already_feasible_is_recorded_not_silently_forced_into_wallet_class(self):
        result=diagnostic.diagnose_action((2003,),{'gold_est':100},lambda *a:[{'item_id':2003,'cost':50}],None)
        self.assertEqual(result['classification'],'feasible_at_original_wallet')
        self.assertEqual(result['native_economic_calls'],2)

    def test_sequence_copy_count_and_cost_receipts_are_enforced(self):
        for sequence in ([{'item_id':2003,'cost':30}],[{'item_id':2003,'cost':90}]*2,
                         [{'item_id':2003,'cost':-1}]*2):
            with self.assertRaises(ValueError):diagnostic.check_sequence(sequence,(2003,2003),100)

    def test_bisection_zero_cost_minimum(self):
        result=diagnostic.diagnose_action((2003,),{'gold_est':0},lambda *a:[{'item_id':2003,'cost':0}],None)
        self.assertEqual(result['classification'],'feasible_at_original_wallet')

    def test_timeout_guard_called_before_every_native_probe(self):
        with self.assertRaisesRegex(RuntimeError,'deadline'):
            diagnostic.diagnose_action((2003,),{'gold_est':50},lambda *a:None,None,
                                       lambda:(_ for _ in ()).throw(RuntimeError('deadline')))

    def test_invalid_or_save_target_rejected(self):
        for action in ((),(999999,),(True,)):
            with self.assertRaises(ValueError):diagnostic.diagnose_action(action,{'gold_est':0},lambda *a:None,None)

    def test_explicit_target_access_acknowledgment_precedes_any_artifact_read(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory)/'new',patch='16.18',deadline_seconds=120,read_observed_targets=False)
            with patch.object(diagnostic,'verify_report',side_effect=AssertionError('artifact opened')):
                with self.assertRaisesRegex(ValueError,'explicitly opens observed purchase targets'):diagnostic.run(args)

    def test_observed_group_mapping_preserves_multiplicity(self):
        groups=[[],[[2003,2]]]
        target={'encoding':'item-counts-v1','main_label_counts':{'2003':2},'main_group_id':1}
        self.assertEqual(diagnostic.target_main(target,groups),(1,[[2003,2]]))
        target['main_label_counts']['2003']=1
        with self.assertRaisesRegex(ValueError,'mismatch'):diagnostic.target_main(target,groups)

    def test_fixed_scope_rejects_tie_and_changed_receipt_identity(self):
        n=859;groups=[[],[[1001,1]],[[2003,1]],[[1052,1]]]
        contexts=[];targets=[];refs=[];predictions=[];scored=[]
        p=np.tile(np.array([.05,.8,.1,.05],dtype=np.float32),(n,1))
        for i in range(n):
            identity={'sample_id':f's{i}','match_id':f'm{i}'}
            contexts.append({**identity,'status':'available','participant_id':1,'role':'development','patch':'16.19',
                             'context_timestamp_ms':10,'source_visit_end_ms':11,'projection_sha256':'x','source_projection':{'gold_est':5}})
            targets.append({**identity,'status':'available','target':{'encoding':'item-counts-v1','main_label_counts':{'1001':1},
                            'main_group_id':1,'kind':'other_purchase'}})
            direct=[{'group':0}] if i==0 else [{'group':1}]
            refs.append({**identity,'reference_status':'scored','bc_tensor_row':i,'direct_bc_options':direct,'m2_options':[{'group':0}]})
            predictions.append({**identity,'catalogue_top_groups':[1,2,0],'direct_options':direct})
            scored.append({**identity,'catalogue_diagnostic_top3_main_credit':True,'secondary_direct_main_ceiling':i!=0})
        selected,counts=diagnostic.select_cases(contexts,targets,refs,predictions,scored,p,groups,expected=1)
        self.assertEqual(len(selected),1);self.assertEqual(counts,{'catalogue_only':1,'both':858})
        with self.assertRaisesRegex(ValueError,'scope changed'):
            diagnostic.select_cases(contexts,targets,refs,predictions,scored,p,groups,expected=108)
        changed=p.copy();changed[0]=[.1,.4,.4,.1]
        with self.assertRaisesRegex(ValueError,'ties'):
            diagnostic.select_cases(contexts,targets,refs,predictions,scored,changed,groups,expected=1)
        targets[0]['sample_id']='different'
        with self.assertRaisesRegex(ValueError,'identity'):
            diagnostic.select_cases(contexts,targets,refs,predictions,scored,p,groups,expected=1)


if __name__=='__main__':unittest.main()
