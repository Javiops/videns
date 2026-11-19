"""Synthetic checks for independent result arithmetic; no experiment reads."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from winprob.future_goal_evaluation import evaluate_future_goal_probe

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('future_result_audit',ROOT/'data/reports/future_goal_result_audit_v1.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FutureGoalResultAuditTests(unittest.TestCase):
    def test_arm_key_roundtrip_keeps_all_metrics(self):
        value={'arms':{'mainonly':{'nll':1.},'future':{'nll':.9}},
               'future_goal_contrasts':{'mainonly':{'nll_reduction':.1},'shuffled':{'top1_gain':.02}},
               'gates':{'two_pp_top1_gain_vs_mainonly':True}}
        self.assertEqual(audit.public_metric_view(audit.historical_metric_view(value)),value)

    def test_no_partial_report_or_checkpoint_access_without_completion_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'verified terminal process'):
                audit.run(SimpleNamespace(completed_run_confirmed=False,run=Path(tmp)/'absent',output=Path(tmp)/'audit.json'))

    def test_independent_bootstraps_and_gate_match_synthetic_production_report(self):
        n=3350;games=[f'synthetic-future-audit-{i}' for i in range(4001)];ids=games[:n]
        group=lambda mid:int.from_bytes(audit.hashlib.sha256(('iql-v1-fold:'+mid).encode()).digest()[:8],'big')%2
        fold=np.array([group(mid) for mid in ids],dtype=np.int64)
        allowed={str(f):[mid for mid in games if group(mid)!=f] for f in (0,1)}
        labels=1+np.arange(n,dtype=np.int64)%9332;selected=np.arange(n,dtype=np.int64);scores={}
        for arm,accuracy,loss in (('mainonly',70,1.),('future',75,.94),('shuffled',71,.98),('frequency',50,1.2)):
            one=np.where(np.arange(n)%100<accuracy,labels,(labels+3)%9333)
            scores[arm]={'target_log_probability':-(loss+.001*(np.arange(n)%7)),
                         'top1':one,'top3':np.column_stack((one,(one+1)%9333,(one+2)%9333))}
        kinds=['other_purchase']*n
        report=evaluate_future_goal_probe(ids,labels,scores,selected_state_indices=selected,prediction_fold=fold,
            training_game_ids_by_fold=allowed,full_selection_game_ids=games,
            selection_statuses=['selected']*n+['no_eligible_focal_episode']*(4001-n),target_kinds=kinds)
        prediction={'labels':labels,'prediction_fold':fold,'target_kind':np.asarray(kinds)}
        for arm,values in scores.items():
            for key,value in values.items():prediction[arm+'_'+key]=value
        result=audit.audit_metrics(prediction,report)
        self.assertTrue(result['metric_gate_passed'])
        self.assertEqual(result['gates'],report['gates'])
        report['future_goal_contrasts']['mainonly']['nll_reduction']+=.001
        with self.assertRaisesRegex(ValueError,'Mismatch'):
            audit.audit_metrics(prediction,report)


if __name__=='__main__':unittest.main()
