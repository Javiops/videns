"""Synthetic projection checks; never calls the real source reader."""
from copy import deepcopy
import unittest
from export_recipe_roles_v1 import role_projection, poison_non_roles


class RoleProjectionTests(unittest.TestCase):
    def fixture(self):
        entry = {'source_event_index': 2, 'timestamp_ms': 600000, 'participant_id': 1}
        board = [{'participant_id': i, 'champion_id': i if i <= 5 else i-5,
                  'team_id': 100 if i <= 5 else 200, 'is_self': i == 1,
                  'team_position': 'MIDDLE' if i in (1, 6) else ''} for i in range(1, 11)]
        return {'event_index': 2, 'ts': 600000, 'participant_id': 1, 'champion_id': 1,
                'team_id': 100, 'team_position': 'MIDDLE', 'board': board}, entry

    def test_exact_roles_and_opponent_mirror_champion(self):
        event, entry = self.fixture(); result = role_projection(event, entry)
        self.assertEqual(result['role'], 'MIDDLE')
        self.assertEqual(len(result['others']), 9)
        self.assertEqual(result['others'][4], {'participant_id': 6, 'champion_id': 1, 'team_id': 200, 'role': 'MIDDLE'})

    def test_forbidden_fields_cannot_change_projection(self):
        event, entry = self.fixture()
        self.assertEqual(role_projection(event, entry), role_projection(poison_non_roles(event), entry))

    def test_rejects_duplicate_or_changed_roster_identity(self):
        event, entry = self.fixture()
        for field, value in (('participant_id', 7), ('ts', 0), ('event_index', 99), ('champion_id', 999)):
            bad = deepcopy(event); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                role_projection(bad, entry)
        bad = deepcopy(event); bad['board'][1]['participant_id'] = 1
        with self.assertRaises(ValueError):
            role_projection(bad, entry)

    def test_empty_role_matches_native_fallback(self):
        event, entry = self.fixture(); event['team_position'] = None
        event['board'][3]['team_position'] = None
        got = role_projection(event, entry)
        self.assertEqual(got['role'], '')
        self.assertEqual(got['others'][2]['role'], '')

    def test_preserves_original_other_order(self):
        event, entry = self.fixture(); event['board'] = list(reversed(event['board']))
        self.assertEqual([p['participant_id'] for p in role_projection(event, entry)['others']], list(range(10, 1, -1)))


if __name__ == '__main__':
    unittest.main()
