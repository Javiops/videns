from copy import deepcopy
import unittest
import numpy as np
from winprob.wp_scale_models_v1 import (REFERENCES,CANDIDATES,select_scale_candidate,validate_shard)
from winprob.experiment import sample_training_indices


def contrast(relative=.03,lower=.001,brier=.001):
    return {'relative_log_loss_reduction':relative,'log_loss':{'ci95':[lower,.02]},'brier':{'ci95':[brier,.02],'estimate':brier}}

def comparisons():
    overall={a+'_vs_'+b:contrast() for a in CANDIDATES for b in REFERENCES}
    lead={a+'_vs_'+b:contrast(.01) for a in CANDIDATES for b in REFERENCES}
    for family in ('rich','combat'):overall[family+'_full_vs_'+family+'_pilot_recalibrated']=contrast(.001,-.001)
    for combat in ('combat_pilot_recalibrated','combat_full'):
        for rich in ('rich_pilot_recalibrated','rich_full'):overall[combat+'_vs_'+rich]=contrast(.001,-.001)
    coverage={'fit_games':11468,'calibration_games':2866,'complement_games':1863,'lead_games':1800}
    return overall,lead,coverage


class ScaleModelTests(unittest.TestCase):
    def test_prefers_simple_recalibrated_rich_without_resolved_extra_gains(self):
        self.assertEqual(select_scale_candidate(*comparisons())['selected_candidate'],'rich_pilot_recalibrated')

    def test_full_fit_requires_additional_resolved_gain(self):
        c,l,n=comparisons();c['rich_full_vs_rich_pilot_recalibrated']=contrast(.006)
        self.assertEqual(select_scale_candidate(c,l,n)['selected_candidate'],'rich_full')

    def test_combat_upgrade_requires_both_hierarchy_comparisons(self):
        c,l,n=comparisons();c['combat_full_vs_combat_pilot_recalibrated']=contrast(.006)
        c['combat_full_vs_rich_pilot_recalibrated']=contrast(.006)
        self.assertEqual(select_scale_candidate(c,l,n)['selected_candidate'],'combat_full')
        c['combat_full_vs_rich_pilot_recalibrated']['brier']['estimate']=-.001
        self.assertEqual(select_scale_candidate(c,l,n)['selected_candidate'],'rich_pilot_recalibrated')

    def test_cannot_choose_only_the_weaker_reference(self):
        c,l,n=comparisons()
        for candidate in CANDIDATES:c[candidate+'_vs_combat_frozen']['relative_log_loss_reduction']=.001
        self.assertIsNone(select_scale_candidate(c,l,n)['selected_candidate'])

    def test_missing_lead_time_or_scale_coverage_fails(self):
        c,l,n=comparisons();n['lead_games']=999
        self.assertIsNone(select_scale_candidate(c,l,n)['selected_candidate'])
        c,l,n=comparisons();n['complement_games']=1699
        self.assertIsNone(select_scale_candidate(c,l,n)['selected_candidate'])

    def test_shard_checks_stable_game_label_and_exact_sample(self):
        clock=np.array([180000,240000,300000,360000],dtype=np.int64)
        selected=sample_training_indices(np.zeros(4,dtype=np.int32),clock)
        a={'x_rich':np.zeros((4,61),dtype=np.float32),'x_combat':np.zeros((4,112),dtype=np.float32),
           'y':np.ones(4,dtype=np.int8),'timestamp_ms':clock,'fit_selected_local':selected}
        row={'rows':4,'game_index':0,'role':'fit','fit_sampled_frames':len(selected)}
        validate_shard(a,row)
        changed=deepcopy(a);changed['y'][0]=0
        with self.assertRaisesRegex(ValueError,'label/time'):validate_shard(changed,row)
        changed=deepcopy(a);changed['fit_selected_local']=np.array([],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,'sampling'):validate_shard(changed,row)

if __name__=='__main__':unittest.main()
