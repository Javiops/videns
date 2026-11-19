from copy import deepcopy
import unittest
import numpy as np
from winprob.current_observed_model_v1 import CurrentObservedModel,choose_candidate,CANDIDATES


def item(relative=.03,lower=.001,brier=.001):
    return {'relative_log_loss_reduction':relative,'log_loss':{'ci95':[lower,.02]},'brier':{'ci95':[brier,.01],'estimate':brier}}

def fixtures():
    overall={name+'_vs_reference':item() for name in CANDIDATES}
    overall['combat_resources_vs_health_availability']=item(relative=.001,lower=-.001)
    lead={name+'_vs_reference':item(relative=.01) for name in CANDIDATES}
    return overall,lead


class ObservedModelTests(unittest.TestCase):
    def test_simpler_model_preferred_without_resolved_extra_gain(self):
        overall,lead=fixtures();result=choose_candidate(overall,lead,998)
        self.assertEqual(result['selected_candidate'],'health_availability')

    def test_full_model_needs_fixed_additional_gain(self):
        overall,lead=fixtures();overall['combat_resources_vs_health_availability']=item(relative=.006)
        self.assertEqual(choose_candidate(overall,lead,998)['selected_candidate'],'combat_resources')
        overall['combat_resources_vs_health_availability']['brier']['estimate']=-.0001
        self.assertEqual(choose_candidate(overall,lead,998)['selected_candidate'],'health_availability')

    def test_global_gain_cannot_rescue_failed_lead_time(self):
        overall,lead=fixtures()
        for name in CANDIDATES:lead[name+'_vs_reference']['log_loss']['ci95'][0]=-.001
        self.assertIsNone(choose_candidate(overall,lead,998)['selected_candidate'])
        self.assertIsNone(choose_candidate(overall,{},0)['selected_candidate'])

    def test_minimum_lead_time_population_is_required(self):
        overall,lead=fixtures();self.assertIsNone(choose_candidate(overall,lead,499)['selected_candidate'])

    def test_only_qualifying_candidate_selected(self):
        overall,lead=fixtures();overall['health_availability_vs_reference']['relative_log_loss_reduction']=.001
        self.assertEqual(choose_candidate(overall,lead,998)['selected_candidate'],'combat_resources')

    def test_brier_uncertainty_cannot_be_ignored(self):
        overall,lead=fixtures()
        for name in CANDIDATES:overall[name+'_vs_reference']['brier']['ci95'][0]=-.0001
        self.assertFalse(choose_candidate(overall,lead,998)['useful_development_signal'])

    def test_no_counterfactual_or_live_capability_claim(self):
        model=CurrentObservedModel(['x','z']);self.assertFalse(model.supports_counterfactual_items)
        self.assertFalse(model.live_safe)
        with self.assertRaisesRegex(ValueError,'CURRENT'):model.fit(np.ones((200,3)),np.arange(200)%2,np.arange(200))

    def test_prediction_rejects_additional_metadata(self):
        rng=np.random.default_rng(40);x=rng.normal(size=(300,2));y=(x[:,0]>0).astype(int);g=np.repeat(np.arange(100),3)
        model=CurrentObservedModel(['x','z']).fit(x,y,g).calibrate(x,y,g)
        original=model.predict(x)
        unrelated_end_times=np.arange(300);unrelated_end_times[:]=999999
        np.testing.assert_array_equal(original,model.predict(x))
        with self.assertRaisesRegex(ValueError,'CURRENT'):model.predict(np.column_stack((x,unrelated_end_times)))

if __name__=='__main__':unittest.main()
