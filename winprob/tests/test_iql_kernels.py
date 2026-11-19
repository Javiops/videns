"""Synthetic accounting/gradient/provenance tests; no real model or data."""
import unittest
import numpy as np

from winprob.iql_kernels import (actor_advantage_weights, assert_out_of_game_predictions,
    awr_actor_loss, bellman_targets, episodic_returns, expectile_value_loss, full_basket_key,
    game_folds, initialize_target, observed_critic_loss, observed_target_q,
    polyak_target, potential_shaped_rewards, targets_from_next_states, normalize_oof_actor_weights)


class IQLKernelTests(unittest.TestCase):
    def test_full_baskets_do_not_collapse_ward_variants_or_save(self):
        keys = [full_basket_key({'encoding': 'item-counts-v1', 'label_counts': counts,
                                'kind': 'observed_no_buy_death' if not counts else 'retained_net_purchase'})
                for counts in ({}, {'2055': 1}, {'2055': 2}, {'1036': 1}, {'1036': 1, '2055': 1})]
        self.assertEqual(len(set(keys)), 5)
        self.assertEqual(keys[2], ((2055, 2),))
        self.assertEqual(full_basket_key({'encoding': 'item-counts-v1', 'kind': 'observed_no_buy_death',
                                         'label_counts': {'999999': 1}}), ())
        with self.assertRaises(ValueError):
            full_basket_key({'encoding': 'item-counts-v1', 'label_counts': {'999999': 1}})

    def test_observed_q_queries_only_recorded_full_actions(self):
        calls = []
        def critic(states, actions):
            calls.append((states, actions))
            return np.array([.2, .8]), np.array([.3, .7])
        actions = [{'encoding': 'item-counts-v1', 'label_counts': {'1036': 1, '2055': 1}},
                   {'encoding': 'item-counts-v1', 'label_counts': {'2055': 2}}]
        result = observed_target_q(critic, ['s0', 's1'], actions)
        np.testing.assert_array_equal(result, [.2, .7])
        self.assertEqual(calls, [(('s0', 's1'), (((1036, 1), (2055, 1)), ((2055, 2),)))])

    def test_no_candidate_axis_is_accepted_as_observed_q(self):
        with self.assertRaisesRegex(ValueError, 'candidate matrix'):
            observed_critic_loss([[.2, .9]], [[.3, .8]], [.5])

    def test_terminal_target_ignores_nonexistent_next_value_without_nan(self):
        result = bellman_targets([0, 1], np.array([False, True]), [.6, np.nan])
        np.testing.assert_array_equal(result, [.6, 1])
        self.assertTrue(np.isfinite(result).all())

    def test_terminal_state_is_never_queried(self):
        calls = []
        def value(states):
            calls.append(states); return [.4]
        result = targets_from_next_states(value, ['real-next', None], [0, 1], np.array([False, True]))
        self.assertEqual(calls, [('real-next',)])
        np.testing.assert_array_equal(result, [.4, 1])
        calls.clear()
        np.testing.assert_array_equal(targets_from_next_states(value, [None], [0], np.array([True])), [0])
        self.assertEqual(calls, [])

    def test_expectile_is_on_predicted_q_not_realized_outcomes(self):
        q = np.array([.25, .75]); v = np.array([.5, .5])
        result = expectile_value_loss(q, v, .7)
        np.testing.assert_allclose(result['per_row_loss'], [.3*.25**2, .7*.25**2])
        # Swapping independently recorded win/loss labels cannot affect an API
        # whose target is Q. Replacing Q by Bernoulli labels yields another loss.
        self.assertNotEqual(result['loss'], expectile_value_loss([0, 1], v, .7)['loss'])

    def test_value_and_q_gradients_match_finite_differences(self):
        q, v = np.array([.2, .8]), np.array([.4, .3])
        got = expectile_value_loss(q, v, sampling_weights=[1, 2])
        for i in range(2):
            delta = np.zeros(2); delta[i] = 1e-6
            diff = (expectile_value_loss(q, v+delta, sampling_weights=[1, 2])['loss']
                    -expectile_value_loss(q, v-delta, sampling_weights=[1, 2])['loss'])/(2e-6)
            self.assertAlmostEqual(diff, got['gradient_wrt_online_v'][i], places=8)
        got = observed_critic_loss(q, v, [.3, .9], [1, 2])
        for i in range(2):
            delta = np.zeros(2); delta[i] = 1e-6
            diff = (observed_critic_loss(q+delta, v, [.3, .9], [1, 2])['loss']
                    -observed_critic_loss(q-delta, v, [.3, .9], [1, 2])['loss'])/(2e-6)
            self.assertAlmostEqual(diff, got['gradient_wrt_online_q1'][i], places=8)

    def test_target_initialization_and_polyak_are_independent_copies(self):
        online = {'weight': np.array([2., 4.], dtype=np.float32)}
        target = initialize_target(online)
        online['weight'][0] = 10
        np.testing.assert_array_equal(target['weight'], [2, 4])
        new = polyak_target(online, target, .005)
        np.testing.assert_allclose(new['weight'], .995*target['weight']+.005*online['weight'])
        self.assertFalse(np.shares_memory(new['weight'], target['weight']))
        self.assertFalse(np.shares_memory(new['weight'], online['weight']))
        self.assertEqual(new['weight'].dtype, np.float32)

    def test_awr_clip_is_stable_and_weights_are_not_renormalized(self):
        result = actor_advantage_weights([1000, .5, -1000], [0, .5, 0], temperature=.1, cap=20)
        np.testing.assert_allclose(result['weights'], [20, 1, 0])
        self.assertTrue(np.isfinite(result['weights']).all())
        loss = awr_actor_loss(np.log([.5, .5]), [20, 1], [1, 1])
        self.assertAlmostEqual(loss['loss'], 10.5*np.log(2))

    def test_terminal_reward_is_paid_once_across_complete_episode_boundaries(self):
        rewards = [0, 1, 0, 0]
        done = np.array([False, True, False, True])
        np.testing.assert_array_equal(episodic_returns(rewards, done), [1, 1, 0, 0])
        with self.assertRaises(ValueError):
            episodic_returns([0, 1], np.array([False, False]))

    def test_wp_shaping_uses_zero_terminal_potential_not_a_second_win(self):
        r = np.array([0., 0., 1.]); done = np.array([False, False, True])
        current = np.array([.2, .6, .7])
        shaped = potential_shaped_rewards(r, done, current, [.6, .7, 1.])
        np.testing.assert_allclose(shaped, [.4, .1, .3])
        np.testing.assert_allclose(episodic_returns(shaped, done), episodic_returns(r, done)-current)
        self.assertAlmostEqual(shaped.sum(), .8)

    def test_game_folds_are_order_and_player_count_invariant(self):
        base = game_folds(['MATCH_A', 'MATCH_B', 'MATCH_C'])
        duplicate = game_folds(['MATCH_C', 'MATCH_A', 'MATCH_A', 'MATCH_B', 'MATCH_C'])
        self.assertEqual(base, duplicate)
        self.assertTrue(set(base.values()) <= {0, 1})

    def test_crossfit_provenance_checks_entire_game_exposure(self):
        ok = assert_out_of_game_predictions(['A', 'A', 'B'], ['fold1', 'fold1', 'fold0'],
                                            {'fold1': ['B', 'C'], 'fold0': ['A', 'D']})
        self.assertEqual(ok['games'], 2)
        with self.assertRaisesRegex(ValueError, 'was seen'):
            assert_out_of_game_predictions(['A'], ['critic'], {'critic': ['A']})
        with self.assertRaisesRegex(ValueError, 'missing'):
            assert_out_of_game_predictions(['A'], ['unknown'], {})

    def test_global_fit_rescale_preserves_raw_fallback_and_mean_one(self):
        supported = np.array([True]*7+[False]*3)
        q1 = np.array([1., .8, .7, .6, .5, .4, .3, np.nan, np.nan, np.nan])
        q2 = q1.copy(); values = np.array([.5]*7+[np.nan]*3)
        result = normalize_oof_actor_weights(q1, q2, values, supported)
        np.testing.assert_array_equal(result['raw_weights'][~supported], 1)
        self.assertAlmostEqual(result['weights'].mean(), 1)
        self.assertEqual(result['support_fraction'], .7)
        self.assertEqual(result['fallback_transitions'], 3)
        self.assertLessEqual(result['effective_sample_size'], 10)
        # All rows share one constant, including unsupported raw-one fallbacks.
        np.testing.assert_allclose(result['weights']*result['global_fit_mean_raw_weight'], result['raw_weights'])
        supported[0] = False
        with self.assertRaisesRegex(ValueError, 'support'):
            normalize_oof_actor_weights(q1, q2, values, supported)

    def test_numpy_id_arrays_are_supported_for_fold_and_exposure_checks(self):
        ids = np.asarray(['A', 'B', 'A'])
        self.assertEqual(game_folds(ids), game_folds(ids.tolist()))
        result = assert_out_of_game_predictions(ids, np.asarray(['x', 'y', 'x']), {'x': ['B'], 'y': ['A']})
        self.assertEqual(result['exposure_overlap'], 0)


if __name__ == '__main__':
    unittest.main()
