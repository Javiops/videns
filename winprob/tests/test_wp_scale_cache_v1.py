from copy import deepcopy
import unittest
import numpy as np
from test_features_rich import fixture as rich_fixture
from winprob.wp_scale_cache_v1 import convert_game,verify_pilot_arrays
from winprob.features_rich import ALL_RICH_FEATURES,extract_rich_game_states


def fixture():
    match,timeline=rich_fixture();match['info'].update(queueId=420,gameCreation=1699999999000)
    for player in match['info']['participants']:player['championId']=player['participantId']
    for minute,frame in enumerate(timeline['info']['frames']):
        for player in frame['participantFrames'].values():
            player['currentGold']=200
            player['championStats']={'health':800+minute*100,'healthMax':1000+minute*100,'armor':50,'magicResist':40,
                                    'abilityPower':50,'attackDamage':100,'attackSpeed':120,'movementSpeed':350}
    entry={'match_id':match['metadata']['matchId'],'patch':'16.19','game_creation_ms':1699999999000}
    static={'champion_tags':{str(i):['Mage'] for i in range(1,11)}}
    return match,timeline,entry,static


class ScaleCacheTests(unittest.TestCase):
    def test_frozen_rich_rows_and_labels_are_preserved(self):
        match,timeline,entry,static=fixture();expected=extract_rich_game_states(match,timeline)
        value=convert_game(match,timeline,entry,static)
        np.testing.assert_array_equal(value['arrays']['x_rich'],np.asarray([[r['x'][n] for n in ALL_RICH_FEATURES] for r in expected],dtype=np.float32))
        np.testing.assert_array_equal(value['arrays']['y'],[r['y'] for r in expected])
        self.assertEqual(value['arrays']['x_combat'].shape,(4,112));self.assertEqual(value['evaluation_only_end_ms'],450000)

    def test_outcome_mutation_changes_labels_not_features(self):
        match,timeline,entry,static=fixture();before=convert_game(match,timeline,entry,static)
        for team in match['info']['teams']:team['win']=not team['win']
        after=convert_game(match,timeline,entry,static)
        for key in ('x_rich','x_combat','timestamp_ms'):np.testing.assert_array_equal(before['arrays'][key],after['arrays'][key])
        np.testing.assert_array_equal(after['arrays']['y'],1-before['arrays']['y'])

    def test_future_mutation_does_not_change_prior_features(self):
        match,timeline,entry,static=fixture();before=convert_game(match,timeline,entry,static)
        for frame in timeline['info']['frames'][4:]:
            frame['participantFrames']['1']['totalGold']+=10000
            frame['participantFrames']['1']['championStats']['health']=0
        after=convert_game(match,timeline,entry,static)
        np.testing.assert_array_equal(before['arrays']['x_rich'][0],after['arrays']['x_rich'][0])
        np.testing.assert_array_equal(before['arrays']['x_combat'][0],after['arrays']['x_combat'][0])

    def test_zero_base_qualified_frames_are_explicit(self):
        match,timeline,entry,static=fixture()
        for frame in timeline['info']['frames']:frame['participantFrames'].pop('10')
        self.assertIsNone(convert_game(match,timeline,entry,static))

    def test_wrong_queue_or_patch_fails(self):
        match,timeline,entry,static=fixture();match['info']['queueId']=430
        with self.assertRaisesRegex(ValueError,'queue'):convert_game(match,timeline,entry,static)
        match,timeline,entry,static=fixture();entry['patch']='16.17'
        with self.assertRaisesRegex(ValueError,'patch'):convert_game(match,timeline,entry,static)

    def test_pilot_parity_rejects_changed_values_or_dtype(self):
        match,timeline,entry,static=fixture();a=convert_game(match,timeline,entry,static)['arrays'];copy={k:v.copy() for k,v in a.items()}
        verify_pilot_arrays(copy,a['x_rich'],a['x_combat'],a['y'],a['timestamp_ms'])
        copy['x_rich'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'pilot'):verify_pilot_arrays(copy,a['x_rich'],a['x_combat'],a['y'],a['timestamp_ms'])

if __name__=='__main__':unittest.main()
