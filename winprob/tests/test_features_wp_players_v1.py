from copy import deepcopy
import unittest
import numpy as np
from test_wp_scale_cache_v1 import fixture
from winprob.features_wp_players_v1 import FIELDS,extract_player_tokens,frame_tokens
from winprob.features_current_combat_v1 import roster_identity


class PlayerTokenTests(unittest.TestCase):
    def setUp(self):self.match,self.timeline,self.entry,self.static=fixture();self.times=np.array([180000,240000],dtype=np.int64)
    def extract(self,timeline=None,match=None):
        return extract_player_tokens(match or self.match,timeline or self.timeline,self.times,self.static['champion_tags'])[0]

    def test_exact_shape_side_and_fractional_tags(self):
        self.static['champion_tags']['1']=['Mage','Support'];x=self.extract()
        self.assertEqual(x.shape,(2,10,35));self.assertEqual(x.dtype,np.float32)
        np.testing.assert_array_equal(x[:,:5,-1],1);np.testing.assert_array_equal(x[:,5:,-1],-1)
        self.assertEqual(x[0,0,FIELDS.index('tag_Mage')],.5);self.assertEqual(x[0,0,FIELDS.index('tag_Support')],.5)

    def test_unknown_position_and_health_keep_explicit_masks(self):
        frame=next(f for f in self.timeline['info']['frames'] if f['timestamp']==180000);raw=frame['participantFrames']['1']
        raw['position']={'x':0,'y':200};raw['championStats']['health']=float('nan')
        x=self.extract()[0,0];self.assertEqual(x[FIELDS.index('position_observed')],0)
        self.assertEqual(x[FIELDS.index('position_x')],0);self.assertEqual(x[FIELDS.index('health_observed')],0)
        self.assertEqual(x[FIELDS.index('health_ratio_observed')],0);self.assertEqual(x[FIELDS.index('health_positive')],0)

    def test_health_above_maximum_is_observed_but_ratio_unknown(self):
        frame=next(f for f in self.timeline['info']['frames'] if f['timestamp']==180000)
        frame['participantFrames']['1']['championStats'].update(health=2000,healthMax=1000)
        x=self.extract()[0,0];self.assertEqual(x[FIELDS.index('health_observed')],1)
        self.assertEqual(x[FIELDS.index('health_ratio_observed')],0)

    def test_outcomes_and_future_frames_do_not_change_inputs(self):
        before=self.extract();match=deepcopy(self.match);timeline=deepcopy(self.timeline)
        for team in match['info']['teams']:team['win']=not team['win']
        for p in match['info']['participants']:p.update(kills=999,totalDamageDealt=999999,teamPosition='irrelevant')
        for frame in timeline['info']['frames']:
            if frame['timestamp']>240000:
                for raw in frame['participantFrames'].values():raw['championStats']['health']=0;raw['position']={'x':1,'y':1}
        np.testing.assert_array_equal(before,self.extract(timeline,match))

    def test_same_team_health_swap_is_retained(self):
        before=self.extract();timeline=deepcopy(self.timeline)
        for frame in timeline['info']['frames']:
            frame['participantFrames']['1']['championStats']['health']=100
            frame['participantFrames']['2']['championStats']['health']=700
        a=self.extract(timeline);swapped=deepcopy(timeline)
        for frame in swapped['info']['frames']:
            frame['participantFrames']['1']['championStats']['health']=700
            frame['participantFrames']['2']['championStats']['health']=100
        b=self.extract(swapped);self.assertFalse(np.array_equal(a,b))
        np.testing.assert_allclose(a[:,:5].sum(axis=1),b[:,:5].sum(axis=1),atol=1e-6)

    def test_roster_dictionary_order_is_not_an_input(self):
        roster=roster_identity(self.match);frame=self.timeline['info']['frames'][3]
        a,_=frame_tokens(roster,frame,self.static['champion_tags'])
        b,_=frame_tokens(list(reversed(roster)),frame,self.static['champion_tags']);np.testing.assert_array_equal(a,b)

    def test_no_future_interpolation_or_identity_substitution(self):
        with self.assertRaisesRegex(ValueError,'exact admitted frame'):
            extract_player_tokens(self.match,self.timeline,np.array([180001]),self.static['champion_tags'])
        timeline=deepcopy(self.timeline);timeline['metadata']['matchId']='EUW1_999'
        with self.assertRaisesRegex(ValueError,'identity'):self.extract(timeline)


if __name__=='__main__':unittest.main()
