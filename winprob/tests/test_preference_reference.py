import copy
import hashlib
import unittest
import numpy as np

from winprob.preference_reference import (M2_IDENTITY, validate_header,
                                         validate_rows, validate_arrays, value_hash)


class ReferenceConsumerTests(unittest.TestCase):
    def header(self):
        return {'schema': 'winprob-m2-preference-v1', 'state': 'complete',
                'package_identity_sha256': M2_IDENTITY, 'group_count': 9333, 'pairs': 2,
                'test_payload_reads': False, 'confirmation_payload_reads': False,
                'training_performed': False, 'zero_residual_native_options_mismatches': 0,
                'parity_checked_pairs': 2, 'mutation_checked_pairs': 2,
                'whole_distribution_label_mutation_equal': True}

    def row(self):
        projection = {'ts': 100, 'gold_est': 200}
        identity = {'match_id': 'g1', 'participant_id': 1, 'feature_cutoff_ms': 100,
                    'undo_ms': 200, 'decision_timestamp_ms': 300, 'action_a': 100, 'action_b': 200}
        return {'match_id': 'g1', 'role': 'fit', 'patch': '16.18', 'pair_id': 'g1|1|100|200|300|100|200',
                'context_id': value_hash(identity), 'source_projection': projection, 'participant_id': 1,
                'projection_sha256': value_hash(projection), 'context_timestamp_ms': 100,
                'undo_timestamp_ms': 200, 'decision_timestamp_ms': 300, 'random_focal_pair_weight': .1,
                'action_a': 100, 'action_b': 200, 'group_a': 1, 'group_b': 2,
                'probability_a': .25, 'probability_b': .75, 'tensor_row': 0,
                'probability_row_sha256': hashlib.sha256(self.arrays()[0][0].tobytes()).hexdigest()}

    def admitted(self):
        return {'g1': {'role': 'train', 'patch': '16.18', 'historical_validation': False}}

    def arrays(self):
        p = np.zeros((1, 9333), dtype=np.float32)
        p[0, 1:3] = [.25, .75]
        groups = [[], [[100, 1]], [[200, 1]]] + [[[1000+i, 1]] for i in range(9330)]
        return p, groups

    def test_real_identity_requires_all_rows_checked(self):
        header = self.header()
        validate_header(header)
        for key in ('parity_checked_pairs', 'mutation_checked_pairs'):
            other = copy.deepcopy(header); other[key] = 0
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validate_header(other)

    def test_reserved_role_fails_before_array_read(self):
        row = self.row(); admitted = self.admitted()
        admitted['g1']['role'] = 'confirmatory_reserved'
        with self.assertRaisesRegex(ValueError, 'permitted'):
            validate_rows([row], admitted)

    def test_m2_exposed_development_cannot_replace_new_patch(self):
        row = self.row(); row['role'] = 'development'
        admitted = self.admitted(); admitted['g1']['role'] = 'development'
        with self.assertRaisesRegex(ValueError, 'permitted'):
            validate_rows([row], admitted)

    def test_mutated_projection_rejected(self):
        row = self.row(); row['source_projection']['gold_est'] = 999
        with self.assertRaisesRegex(ValueError, 'projection'):
            validate_rows([row], self.admitted())

    def test_duplicate_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows([self.row(), self.row()], self.admitted())

    def test_full_distribution_mass_and_mapping(self):
        p, groups = self.arrays(); row = self.row()
        self.assertTrue(validate_arrays([row], p, groups)[0])
        row['probability_b'] = .5
        with self.assertRaisesRegex(ValueError, 'tensor'):
            validate_arrays([row], p, groups)
        row = self.row(); groups[2] = [[200, 2]]
        with self.assertRaisesRegex(ValueError, 'mapping'):
            validate_arrays([row], p, groups)

    def test_reference_zeros_and_unmapped_pairs_not_fabricated(self):
        p, groups = self.arrays(); row = self.row()
        p[0, 1:3] = [0, 1]; row.update(probability_a=0., probability_b=1.)
        row['probability_row_sha256'] = hashlib.sha256(p[0].tobytes()).hexdigest()
        self.assertFalse(validate_arrays([row], p, groups)[0])
        row['group_a'] = None
        self.assertFalse(validate_arrays([row], p, groups)[0])


if __name__ == '__main__':
    unittest.main()
