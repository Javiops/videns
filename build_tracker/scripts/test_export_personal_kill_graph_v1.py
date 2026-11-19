from copy import deepcopy
import unittest
from export_personal_kill_graph_v1 import needed_events, source_snapshot


class SourceGraphTests(unittest.TestCase):
    def fixture(self):
        raw = {'type': 'CHAMPION_KILL', 'timestamp': 100, 'killerId': 6, 'victimId': 1, 'assistingParticipantIds': [7]}
        payload = {'info': {'frames': [{'events': [raw, {'type': 'ITEM_PURCHASED', 'itemId': 999}]}]}}
        kda = [[0, 0, 0] for _ in range(10)]; kda[5][0] = 1; kda[0][1] = 1; kda[6][2] = 1
        return payload, kda

    def test_source_prefix_has_exact_counts(self):
        payload, kda = self.fixture(); graph, env, last = source_snapshot(needed_events(payload, 1), kda, [100]*5+[200]*5, 200)
        self.assertEqual(graph[5][0], 1); self.assertEqual(sum(env), 0); self.assertEqual(last, 100)

    def test_unused_suffix_not_validated_and_no_timestamp_resort(self):
        payload, kda = self.fixture()
        payload['info']['frames'][0]['events'].append({'type': 'CHAMPION_KILL', 'timestamp': 100, 'killerId': 99})
        self.assertEqual(len(needed_events(payload, 1)), 1)
        self.assertEqual(source_snapshot(needed_events(payload, 1), kda, [100]*5+[200]*5, 15000)[0][5][0], 1)

    def test_rejects_future_or_mismatched_KDA(self):
        payload, kda = self.fixture(); events = needed_events(payload, 1)
        with self.assertRaises(ValueError):
            source_snapshot(events, kda, [100]*5+[200]*5, 99)
        bad = deepcopy(kda); bad[6][2] = 0
        with self.assertRaises(ValueError):
            source_snapshot(events, bad, [100]*5+[200]*5, 200)

    def test_environment_is_not_an_enemy_edge(self):
        payload, kda = self.fixture(); raw = payload['info']['frames'][0]['events'][0]
        raw['killerId'] = 0; raw['assistingParticipantIds'] = []; kda[5][0] = 0; kda[6][2] = 0
        graph, env, _ = source_snapshot(needed_events(payload, 1), kda, [100]*5+[200]*5, 200)
        self.assertEqual(sum(map(sum, graph)), 0); self.assertEqual(env[0], 1)


if __name__ == '__main__':
    unittest.main()
