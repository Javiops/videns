"""No raw reads: expanded scope guards and exact v1 certificate reuse."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from undo_pairs_expanded import ARRAY_NAMES, ROLE_COUNTS, SCHEMA, analyze_with_ids, stable_pair_id, validate_scope


class ExpandedUndoTests(unittest.TestCase):
    def scope(self):
        games = []
        for role, count in ROLE_COUNTS.items():
            for _ in range(count):
                number = len(games)
                games.append({'match_id': 'SYN_'+str(number), 'game_index': number,
                              'role': role, 'patch': '16.19' if role == 'development' else '16.18'})
        metadata = {'schema': SCHEMA, 'contains_outcome_array': False, 'outcome_fields_accessed': False,
                    'cache_y_accessed': False, 'role_game_counts': dict(ROLE_COUNTS)}
        return metadata, {'schema': SCHEMA, 'games': games, 'exclusions': []}

    def test_exact_new_population_is_accepted(self):
        metadata, source = self.scope()
        validate_scope(metadata, source, ARRAY_NAMES)

    def test_old_schema_or_outcome_array_is_rejected(self):
        metadata, source = self.scope()
        with self.assertRaisesRegex(ValueError, 'outcome-free'):
            validate_scope(metadata, source, ARRAY_NAMES | {'y'})
        metadata['schema'] = 'winprob-cache-v1'
        with self.assertRaisesRegex(ValueError, 'revision-context schema'):
            validate_scope(metadata, source, ARRAY_NAMES)

    def test_trained_patch_development_and_missing_game_fail_closed(self):
        metadata, source = self.scope()
        source['games'][-1]['patch'] = '16.18'
        with self.assertRaisesRegex(ValueError, 'wrong-patch'):
            validate_scope(metadata, source, ARRAY_NAMES)
        source['games'].pop()
        with self.assertRaisesRegex(ValueError, 'role counts'):
            validate_scope(metadata, source, ARRAY_NAMES)

    def test_duplicate_and_excluded_sources_are_rejected(self):
        metadata, source = self.scope()
        source['games'][1]['match_id'] = source['games'][0]['match_id']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_scope(metadata, source, ARRAY_NAMES)
        metadata, source = self.scope()
        source['exclusions'] = [{'reason': 'missing'}]
        with self.assertRaisesRegex(ValueError, 'exclude'):
            validate_scope(metadata, source, ARRAY_NAMES)

    def test_analyzer_output_preserved_except_stable_identity(self):
        row = {'participant_id': 3, 'feature_cutoff_ms': 100, 'undo_ms': 110,
               'decision_timestamp_ms': 120, 'action_a': 1036, 'action_b': 1029,
               'eligible_revision': True, 'rejection_reasons': [], 'random_focal_pair_weight': .1}
        original = copy.deepcopy(row)
        args = ({'immutable': 'match'}, {'immutable': 'timeline'}, {}, {}, [])
        with patch('undo_pairs_expanded.v1.analyze_game', return_value=([row], {'undo_events': 1})) as analyzer:
            rows, counts = analyze_with_ids('SYN_1', *args)
            analyzer.assert_called_once_with(*args)
        self.assertEqual(rows[0].pop('pair_id'), 'SYN_1|3|100|110|120|1036|1029')
        self.assertEqual(rows[0], original)
        self.assertEqual(counts, {'undo_events': 1})

    def test_pair_identity_changes_with_context_or_actions(self):
        row = {'participant_id': 3, 'feature_cutoff_ms': 100, 'undo_ms': 110,
               'decision_timestamp_ms': 120, 'action_a': 1036, 'action_b': 1029}
        first = stable_pair_id('SYN_1', row)
        row['action_b'] = 1033
        self.assertNotEqual(first, stable_pair_id('SYN_1', row))


if __name__ == '__main__':
    unittest.main()
