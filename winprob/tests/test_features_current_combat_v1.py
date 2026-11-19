from copy import deepcopy
import unittest
import numpy as np
from winprob.features_current_combat_v1 import (FEATURES,HEALTH_FEATURES,TAGS,frame_features,roster_identity,
                                               exact_frame_map,evaluation_end_time)


def fixture():
    roster=[(pid,100 if pid<=5 else 200,pid) for pid in range(1,11)]
    tags={str(pid):[TAGS[(pid-1)%6]] for pid in range(1,11)}
    frame={'timestamp':60000,'participantFrames':{str(pid):{'totalGold':1000+100*pid,'currentGold':200,'level':8,
        'championStats':{'health':1000,'healthMax':2000,'attackDamage':100,'abilityPower':100,'armor':50,
                         'magicResist':30,'attackSpeed':130,'movementSpeed':350}} for pid in range(1,11)},'events':[]}
    return roster,frame,tags


class CurrentCombatFeatureTests(unittest.TestCase):
    def test_feature_schema_and_exact_health_projection(self):
        roster,frame,tags=fixture();value,quality,gold,_=frame_features(roster,frame,tags)
        self.assertEqual(len(FEATURES),112);self.assertEqual(len(set(FEATURES)),112)
        self.assertEqual(FEATURES[:4],HEALTH_FEATURES)
        np.testing.assert_array_equal(value[:4],[0,2,0,2]);self.assertEqual(quality['player_frames'],10)
        self.assertEqual(gold,(6500.,9000.))

    def test_roster_and_dictionary_permutation_invariance(self):
        roster,frame,tags=fixture();before=frame_features(roster,frame,tags)[0]
        frame['participantFrames']=dict(reversed(list(frame['participantFrames'].items())))
        after=frame_features(roster[::-1],frame,tags)[0];np.testing.assert_array_equal(before,after)

    def test_team_swap_diff_sign_and_sum_parity(self):
        roster,frame,tags=fixture();before=frame_features(roster,frame,tags)[0]
        after=frame_features([(pid,300-team,cid) for pid,team,cid in roster],frame,tags)[0]
        np.testing.assert_array_equal(after[0::2],-before[0::2]);np.testing.assert_array_equal(after[1::2],before[1::2])

    def test_health_zero_removes_positive_resource_proxy(self):
        roster,frame,tags=fixture();before=frame_features(roster,frame,tags)[0]
        frame['participantFrames']['1']['championStats']['health']=0
        after,q,_,_=frame_features(roster,frame,tags)
        np.testing.assert_allclose(after[:4],[0,2,-.2,1.8],rtol=0,atol=1e-7)
        self.assertEqual(q['health_zero'],1)
        index=FEATURES.index('current_positive_health_gold_sum_over_50000_diff')
        self.assertAlmostEqual(float(after[index]-before[index]),-1100/50000,places=7)

    def test_missing_stats_are_flagged_without_dropping_frame(self):
        roster,frame,tags=fixture();frame['participantFrames']['1']['championStats']=None
        value,q,_,_=frame_features(roster,frame,tags)
        self.assertTrue(np.isfinite(value).all());self.assertEqual(q['health_observed'],9)
        self.assertEqual(q['stat_armor_observed'],9)

    def test_invalid_ratio_does_not_repair_observed_health(self):
        roster,frame,tags=fixture();frame['participantFrames']['1']['championStats'].update(health=2100,healthMax=2000)
        _,q,_,_=frame_features(roster,frame,tags)
        self.assertEqual(q['health_observed'],10);self.assertEqual(q['health_ratio_observed'],9)

    def test_champion_assignment_changes_labelled_gold_not_roster_mass(self):
        roster,frame,tags=fixture();before=frame_features(roster,frame,tags)[0]
        swapped=roster.copy();swapped[0]=(1,100,2);swapped[1]=(2,100,1)
        after=frame_features(swapped,frame,tags)[0]
        roster_columns=[i for i,name in enumerate(FEATURES) if 'roster_fraction' in name]
        np.testing.assert_array_equal(before[roster_columns],after[roster_columns])
        np.testing.assert_array_equal(before[:4],after[:4]);self.assertFalse(np.array_equal(before,after))

    def test_fractional_tags_conserve_roster_and_gold_excess(self):
        roster,frame,tags=fixture();tags['1']=['Tank','Support']
        value=frame_features(roster,frame,tags)[0]
        masses=[value[FEATURES.index(f'current_tag_{tag}_roster_fraction_sum')] for tag in TAGS]
        excess=[value[FEATURES.index(f'current_tag_{tag}_gold_excess_over_50000_sum')] for tag in TAGS]
        self.assertAlmostEqual(float(sum(masses)),2.,places=6);self.assertAlmostEqual(float(sum(excess)),0.,places=7)

    def test_outcome_and_final_participant_fields_not_used(self):
        roster,frame,tags=fixture()
        match={'info':{'participants':[{'participantId':pid,'teamId':team,'championId':cid,'win':False,'teamPosition':'WRONG'}
                                      for pid,team,cid in roster],'teams':[{'win':False},{'win':True}]}}
        before=frame_features(roster_identity(match),frame,tags)[0]
        for player in match['info']['participants']:player.update(win=True,teamPosition='CHANGED',goldEarned=999999)
        match['info']['teams'].reverse()
        np.testing.assert_array_equal(before,frame_features(roster_identity(match),frame,tags)[0])

    def test_future_frames_and_end_timing_are_not_feature_inputs(self):
        roster,frame,tags=fixture();later=deepcopy(frame);later['timestamp']=120000
        timeline={'info':{'frames':[frame,later]}};before=frame_features(roster,exact_frame_map(timeline)[60000],tags)[0]
        later['participantFrames']['1']['championStats']['health']=0
        later['events']=[{'type':'GAME_END','timestamp':130000,'winningTeam':100}]
        self.assertEqual(evaluation_end_time(timeline,60000),130000)
        np.testing.assert_array_equal(before,frame_features(roster,exact_frame_map(timeline)[60000],tags)[0])

    def test_duplicate_frames_or_post_end_qualification_fail(self):
        _,frame,_=fixture()
        with self.assertRaisesRegex(ValueError,'Duplicate'):exact_frame_map({'info':{'frames':[frame,frame]}})
        frame['events']=[{'type':'GAME_END','timestamp':60000}]
        with self.assertRaisesRegex(ValueError,'contradicts'):evaluation_end_time({'info':{'frames':[frame]}},60000)

    def test_missing_end_is_unknown_without_outcome_guess(self):
        _,frame,_=fixture();self.assertEqual(evaluation_end_time({'info':{'frames':[frame]}},60000),-1)

    def test_unknown_native_tag_fails(self):
        roster,frame,tags=fixture();tags['1']=['InventedRole']
        with self.assertRaisesRegex(ValueError,'tags'):frame_features(roster,frame,tags)

    def test_negative_finite_wallet_and_resistance_preserved(self):
        roster,frame,tags=fixture();frame['participantFrames']['1']['currentGold']=-50
        frame['participantFrames']['1']['championStats']['armor']=-20
        value,q,_,_=frame_features(roster,frame,tags)
        self.assertEqual(q['wallet_negative'],1);self.assertEqual(q['wallet_observed'],10)
        self.assertTrue(np.isfinite(value).all())

if __name__=='__main__':unittest.main()
