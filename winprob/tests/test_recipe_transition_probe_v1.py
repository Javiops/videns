from dataclasses import replace
import unittest
import numpy as np
from winprob.recipe_transition_v1 import ChoiceInputs
from winprob.recipe_transition_probe_v1 import Standardizer, game_weights, shuffle_for_entry


def fixture():
    rng = np.random.default_rng(123)
    h = rng.normal(size=(5, 26)); h[:, 0] = 1.
    f = rng.normal(size=(5, 3, 26))
    p = np.array([[.2, .2, .2, .4]]*4+[[.6, 0., 0., .4]])
    f[4, 1:] = 0
    return ChoiceInputs(p, h, f).validate()


class StandardizerTests(unittest.TestCase):
    def test_games_and_present_candidates_have_declared_weights(self):
        data = fixture(); ids = ['a', 'a', 'a', 'b', 'c']
        w = game_weights(ids)
        self.assertAlmostEqual(w[:3].sum(), 1/3)
        self.assertAlmostEqual(w[3], 1/3)
        scale = Standardizer.fit(data, ids)
        candidate_mean = (data.candidates[:3].mean((0, 1)) + data.candidates[3].mean(0) + data.candidates[4, 0])/3
        np.testing.assert_allclose(scale.candidate_mean, candidate_mean, atol=1e-15)
        self.assertEqual(scale.context_mean[0], 0.)
        self.assertEqual(scale.context_scale[0], 1.)

    def test_raw_and_net_have_identical_common_inputs_and_reference(self):
        data = fixture(); scaler = Standardizer.fit(data, ['a', 'a', 'b', 'b', 'c'])
        before = data.candidates.copy()
        raw = scaler.transform(data, 'raw'); net = scaler.transform(data, 'net')
        np.testing.assert_array_equal(raw.candidates[:, :, :16], net.candidates[:, :, :16])
        np.testing.assert_array_equal(raw.context, net.context)
        np.testing.assert_array_equal(raw.reference, data.reference)
        np.testing.assert_array_equal(raw.candidates[:, :, 16:], 0.)
        np.testing.assert_array_equal(raw.candidates[4, 1:], 0.)
        np.testing.assert_array_equal(net.candidates[4, 1:], 0.)
        np.testing.assert_array_equal(data.candidates, before)

    def test_control_only_permutes_whole_net_vectors(self):
        data = fixture(); scaler = Standardizer.fit(data, list('abcde'))
        perm = np.array([[2, 0, 1]]*4+[[0, 1, 2]])
        shuffled = scaler.transform(data, 'shuffled_net', perm)
        net = scaler.transform(data, 'net')
        np.testing.assert_array_equal(shuffled.candidates[:, :, :16], net.candidates[:, :, :16])
        np.testing.assert_allclose(shuffled.candidates[:, :, 16:], np.take_along_axis(net.candidates[:, :, 16:], perm[:, :, None], axis=1), atol=0.)
        bad = perm.copy(); bad[-1] = [2, 0, 1]
        with self.assertRaises(ValueError):
            scaler.transform(data, 'shuffled_net', bad)

    def test_scaler_can_be_serialized_without_refitting_on_evaluation(self):
        data = fixture(); scaler = Standardizer.fit(data, list('abcde'))
        copy = Standardizer.from_dict(scaler.as_dict())
        context = data.context.copy(); context[:, 1:] *= 100
        changed = replace(data, context=context, candidates=data.candidates*10)
        result = copy.transform(changed, 'net')
        np.testing.assert_array_equal(copy.context_mean, scaler.context_mean)
        np.testing.assert_array_equal(copy.candidate_scale, scaler.candidate_scale)
        np.testing.assert_allclose(result.context, (changed.context-scaler.context_mean)/scaler.context_scale)

    def test_identity_hash_control_ignores_target_and_fold(self):
        entry = {'match_id': 'SYNTHETIC', 'participant_id': 3, 'source_event_index': 17}
        p = shuffle_for_entry(entry, 3)
        np.testing.assert_array_equal(p, shuffle_for_entry({**entry, 'target': 4, 'fold': 1}, 3))
        np.testing.assert_array_equal(shuffle_for_entry(entry, 1), [0, 1, 2])

    def test_constant_coordinates_have_unit_scale(self):
        data = fixture(); h = data.context.copy(); f = data.candidates.copy()
        h[:, 1] = 7.; f[:, :, 0] = 2.; f[4, 1:, 0] = 0.
        scaler = Standardizer.fit(replace(data, context=h, candidates=f), list('abcde'))
        self.assertEqual(scaler.context_scale[1], 1.)
        self.assertEqual(scaler.candidate_scale[0], 1.)


if __name__ == '__main__':
    unittest.main()
