from copy import deepcopy
import unittest
from winprob.wp_player_selection_v1 import select_player_candidate


def fixture():
    good={'relative_log_loss_reduction':.03,'log_loss':{'ci95':[.001,.02]},'brier':{'estimate':.005,'ci95':[.001,.009]}}
    pairs={key:deepcopy(good) for key in ('deepset_vs_combat_full','deepset_vs_global','attention_vs_combat_full',
        'attention_vs_global','attention_vs_attention_position_shuffled','attention_vs_deepset')}
    lead={key:deepcopy(good) for key in ('deepset_vs_combat_full','attention_vs_combat_full')}
    coverage={'fit_games':11468,'calibration_games':2866,'development_games':2861,'fit_sampled_frames':64422,
        'completed_models':12,'minimum_completed_epochs':30,'maximum_completed_epochs':30,'three_seed_ensembles':True,
        'inference_checks':True,'lead_games':2861}
    seeds={'deepset':[.01,.01,-.01],'attention':[.01,.01,.01]}
    return pairs,lead,coverage,seeds


class PlayerSelectionTests(unittest.TestCase):
    def test_both_qualify_attention_requires_additional_resolved_gain(self):
        p,l,c,s=fixture();self.assertEqual(select_player_candidate(p,l,c,s)['selected_candidate'],'attention')
        p['attention_vs_deepset']['log_loss']['ci95'][0]=-.001
        self.assertEqual(select_player_candidate(p,l,c,s)['selected_candidate'],'deepset')

    def test_attention_must_beat_matched_shuffle_even_if_reference_gain_large(self):
        p,l,c,s=fixture();p['attention_vs_attention_position_shuffled']['relative_log_loss_reduction']=.001
        self.assertEqual(select_player_candidate(p,l,c,s)['eligible'],['deepset'])

    def test_no_weaker_reference_or_small_significant_gain_rescues_candidate(self):
        p,l,c,s=fixture()
        for name in ('deepset','attention'):p[name+'_vs_combat_full']['relative_log_loss_reduction']=.019
        self.assertIsNone(select_player_candidate(p,l,c,s)['selected_candidate'])

    def test_partial_training_or_one_lucky_seed_is_not_eligible(self):
        p,l,c,s=fixture();c['completed_models']=11
        self.assertEqual(select_player_candidate(p,l,c,s)['eligible'],[])
        c['completed_models']=12;s['attention']=[.5,-.01,-.01];s['deepset']=[.2,-.01,-.01]
        self.assertEqual(select_player_candidate(p,l,c,s)['eligible'],[])

    def test_lead_time_and_brier_evidence_are_required(self):
        p,l,c,s=fixture();l['attention_vs_combat_full']['brier']['estimate']=-.0001
        p['deepset_vs_combat_full']['brier']['ci95'][0]=0
        self.assertEqual(select_player_candidate(p,l,c,s)['eligible'],[])


if __name__=='__main__':unittest.main()
