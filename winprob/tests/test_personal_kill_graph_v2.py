from copy import deepcopy
import unittest
import numpy as np
from winprob.personal_kill_graph_v2 import (
    Kill, apply_kill, needed_kills, graph_for_observation, degree_control, edge_excess,
    excess_profile, control_seed)

TEAMS = np.array([100]*5+[200]*5)


def witness(events):
    kda = np.zeros((10, 3), dtype=np.int64); graph = np.zeros((10, 10), dtype=np.int64); env = np.zeros(10, dtype=np.int64)
    for event in events:
        apply_kill(event, TEAMS, kda, graph, env)
    return kda


class PersonalGraphTests(unittest.TestCase):
    def test_raw_assist_metadata_is_preserved_without_becoming_kill_edges(self):
        events = [Kill(100, 10, 5, (2, 3, 5, 6, 7, 9, 9))]
        expected = np.zeros((10, 3), dtype=np.int64)
        expected[9, 0] = 1; expected[4, 1] = 1
        for pid in events[0].assisters:
            expected[pid-1, 2] += 1
        graph, env, _ = graph_for_observation(events, expected, 100, TEAMS)
        self.assertEqual(graph.sum(), 1); self.assertEqual(graph[9, 4], 1)
        self.assertEqual(env.sum(), 0)
        self.assertEqual(expected[8, 2], 2)

    def test_identical_KDA_can_encode_different_personal_relationships(self):
        a = [Kill(100, 6, 1, (8,)), Kill(200, 6, 1, (8,)), Kill(300, 7, 2, (8,))]
        b = [Kill(100, 6, 1, (8,)), Kill(200, 6, 2, (8,)), Kill(300, 7, 1, (8,))]
        np.testing.assert_array_equal(witness(a), witness(b))
        ga, _, _ = graph_for_observation(a, witness(a), 500, TEAMS)
        gb, _, _ = graph_for_observation(b, witness(b), 500, TEAMS)
        profiles = np.zeros((10, 17)); profiles[5, 0] = 1.; profiles[6, 1] = 1.
        self.assertFalse(np.array_equal(excess_profile(ga, TEAMS, 1, profiles), excess_profile(gb, TEAMS, 1, profiles)))

    def test_stale_snapshot_does_not_add_later_kills_before_nominal_timestamp(self):
        events = [Kill(1000, 6, 1, ()), Kill(2000, 7, 2, ())]
        graph, _, audit = graph_for_observation(events, witness(events[:1]), 16000, TEAMS)
        self.assertEqual(graph.sum(), 1)
        self.assertEqual(audit['maximum_included_timestamp_ms'], 1000)

    def test_same_timestamp_keeps_original_prefix_order(self):
        events = [Kill(100, 6, 1, ()), Kill(100, 7, 2, ())]
        graph, _, _ = graph_for_observation(events, witness(events[:1]), 100, TEAMS)
        self.assertEqual(graph[5, 0], 1); self.assertEqual(graph[6, 1], 0)

    def test_future_suffix_and_nonkill_fields_are_not_consumed(self):
        timeline = {'info': {'frames': [{'events': [
            {'type': 'ITEM_PURCHASED', 'itemId': 999, 'label': 'unused'},
            {'type': 'CHAMPION_KILL', 'timestamp': 100, 'killerId': 6, 'victimId': 1, 'assistingParticipantIds': []},
            {'type': 'CHAMPION_KILL', 'timestamp': 999999, 'killerId': 99, 'victimId': None},
        ]}]}}
        original = needed_kills(timeline, 1)
        changed = deepcopy(timeline); changed['info']['frames'][0]['events'][0]['itemId'] = 1
        changed['info']['frames'][0]['events'][2] = {'type': 'CHAMPION_KILL', 'win': 'poison'}
        self.assertEqual(original, needed_kills(changed, 1))
        self.assertEqual(needed_kills({'poison': 'unused'}, 0), [])

    def test_future_included_event_and_wrong_witness_are_rejected(self):
        events = [Kill(100, 6, 1, ())]
        with self.assertRaises(ValueError):
            graph_for_observation(events, witness(events), 99, TEAMS)
        wrong = witness(events); wrong[6, 0] += 1
        with self.assertRaises(ValueError):
            graph_for_observation(events, wrong, 100, TEAMS)

    def test_environment_deaths_preserve_deaths_without_champion_edges(self):
        events = [Kill(100, 0, 1, ()), Kill(200, 6, 2, ())]
        graph, env, _ = graph_for_observation(events, witness(events), 200, TEAMS)
        self.assertEqual(env[0], 1); self.assertEqual(graph.sum(), 1)
        _, numerator, _ = edge_excess(graph, TEAMS, 1)
        np.testing.assert_array_equal(numerator, 0)

    def test_null_preserves_all_degrees_and_seed_does_not_use_labels(self):
        graph = np.zeros((10, 10), dtype=np.int64)
        graph[5, 0] = 4; graph[6, 1] = 3; graph[7, 2] = 2
        graph[0, 5] = 3; graph[1, 6] = 4
        entry = {'match_id': 'SYNTHETIC', 'participant_id': 1, 'source_event_index': 9}
        seed = control_seed(entry)
        self.assertEqual(seed, control_seed({**entry, 'target': 999, 'fold': 0}))
        changed = 0
        for value in range(20):
            control = degree_control(graph, TEAMS, value)
            np.testing.assert_array_equal(control.sum(0), graph.sum(0))
            np.testing.assert_array_equal(control.sum(1), graph.sum(1))
            self.assertFalse(np.any(control[TEAMS[:, None] == TEAMS[None, :]]))
            changed += int(not np.array_equal(control, graph))
        self.assertGreater(changed, 0)
        np.testing.assert_array_equal(degree_control(graph, TEAMS, seed), degree_control(graph, TEAMS, seed))

    def test_profile_is_invariant_to_roster_renaming_and_common_offset(self):
        graph = np.zeros((10, 10), dtype=np.int64); graph[5, 0] = 2; graph[6, 1] = 1
        rng = np.random.default_rng(8); profiles = rng.normal(size=(10, 17))
        value = excess_profile(graph, TEAMS, 1, profiles)
        permutation = np.array([6, 2, 5, 1, 7, 9, 0, 3, 4, 8]); buyer = int(np.where(permutation == 0)[0][0])+1
        permuted = excess_profile(graph[np.ix_(permutation, permutation)], TEAMS[permutation], buyer, profiles[permutation])
        np.testing.assert_allclose(value, permuted, atol=1e-15)
        np.testing.assert_allclose(value, excess_profile(graph, TEAMS, 1, profiles+2.3), atol=1e-15)
        np.testing.assert_array_equal(excess_profile(graph, TEAMS, 1, np.ones((10, 17))*.1), 0.)


if __name__ == '__main__':
    unittest.main()
