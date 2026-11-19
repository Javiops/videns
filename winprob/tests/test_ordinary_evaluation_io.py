"""Synthetic artifact boundary tests; no research target files are read."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from winprob.ordinary_evaluation_io import validate_targets, digest, bind, load_manifest, validate_audit_entries, canonical
import hashlib


def fixture(counts, full_id, group_id, kind, physical, main, wards):
    identity = dict(sample_id='synthetic|1|0|100', match_id='synthetic', participant_id=1,
                    role='development', patch='16.19', selected_decision=dict(pid=1, event_index=0, ts=100))
    context = dict(identity, status='available')
    target = dict(identity, status='available', target=dict(encoding='item-counts-v1', kind=kind,
        label_counts=counts, physical_label_counts=physical, main_label_counts=main, ward_label_counts=wards,
        full_action_id=full_id, main_group_id=group_id, full_catalogue_supported=full_id is not None,
        main_catalogue_supported=group_id is not None))
    return context, target


class TargetArtifactTests(unittest.TestCase):
    catalogue = {'group_main': [[], [[1001, 1]]]}
    lookup = {('observed_no_buy_death', ()): 0, ('retained_net_purchase', ((1001, 1),)): 1,
              ('retained_net_purchase', ((2055, 1),)): 2}

    def validate(self, context, target):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'targets.jsonl'
            path.write_text(json.dumps(target)+'\n', encoding='utf-8')
            bindings = {}
            rows = validate_targets(path, {'files': {'targets.jsonl': digest(path)}}, [context],
                                    self.catalogue, self.lookup, bindings)
            self.assertEqual(len(bindings), 1)
            return rows

    def test_no_buy_and_ward_only_remain_distinct_but_share_main_group_zero(self):
        for args in [({'999999': 1}, 0, 0, 'observed_no_buy_death', {}, {}, {}),
                     ({'2055': 1}, 2, 0, 'ward_only_purchase', {'2055': 1}, {}, {'2055': 1})]:
            context, target = fixture(*args)
            self.assertEqual(self.validate(context, target)[0]['target']['main_group_id'], 0)

    def test_unknown_full_variant_does_not_erase_supported_main(self):
        context, target = fixture({'1001': 1, '2055': 2}, None, 1, 'mixed_purchase_with_wards',
                                  {'1001': 1, '2055': 2}, {'1001': 1}, {'2055': 2})
        self.validate(context, target)
        changed = deepcopy(target); changed['target']['full_catalogue_supported'] = True
        with self.assertRaisesRegex(ValueError, 'catalogue support'):
            self.validate(context, changed)

    def test_unsupported_main_stays_recorded_without_remapping(self):
        context, target = fixture({'4444': 2}, None, None, 'other_purchase', {'4444': 2}, {'4444': 2}, {})
        self.validate(context, target)
        changed = deepcopy(target); changed['target']['main_group_id'] = 1
        with self.assertRaisesRegex(ValueError, 'catalogue support'):
            self.validate(context, changed)

    def test_counts_and_selected_visit_identity_cannot_change(self):
        context, target = fixture({'1001': 1}, 1, 1, 'other_purchase', {'1001': 1}, {'1001': 1}, {})
        changed = deepcopy(target); changed['target']['main_label_counts']['1001'] = 2
        with self.assertRaisesRegex(ValueError, 'multiset projection'):
            self.validate(context, changed)
        changed = deepcopy(target); changed['selected_decision']['ts'] = 101
        with self.assertRaisesRegex(ValueError, 'Target identity'):
            self.validate(context, changed)

    def test_reconstructed_target_can_survive_unavailable_feature_projection(self):
        context, target = fixture({'1001': 1}, 1, 1, 'other_purchase', {'1001': 1}, {'1001': 1}, {})
        context['status'] = 'unavailable'
        self.validate(context, target)
        context['status'] = 'no_admitted_decision'
        with self.assertRaisesRegex(ValueError, 'without a selected decision'):
            self.validate(context, target)

    def test_digest_mismatch_is_not_a_zero_credit_data_row(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'artifact.json'; path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                bind(path, '0'*64, {})

    def test_missing_payload_binding_fails_before_payload_read(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps({'schema': 'winprob-ordinary-bc20-export-v1', 'state': 'complete', 'files': {}}), encoding='utf-8')
            path.with_suffix('.sha256').write_text(digest(path)+' '+path.name, encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'mandatory payload hash'):
                load_manifest(path, 'winprob-ordinary-bc20-export-v1', {})

    def test_opaque_target_cannot_be_reopened_through_extra_bindings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('selection.json', 'contexts.jsonl'):
                (root/name).write_text('[]', encoding='utf-8')
            # Deliberately no targets.jsonl exists: rejection must precede access.
            manifest = {'schema': 'winprob-ordinary-contexts-v1', 'state': 'complete',
                'files': {name: digest(root/name) for name in ('selection.json', 'contexts.jsonl')},
                'bindings': {str(root/'targets.jsonl'): '0'*64}}
            manifest['files']['targets.jsonl'] = '0'*64
            path = root/'manifest.json'; path.write_text(json.dumps(manifest), encoding='utf-8')
            path.with_suffix('.sha256').write_text(digest(path)+' '+path.name, encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'Target binding'):
                load_manifest(path, 'winprob-ordinary-contexts-v1', {}, skip_files=('targets.jsonl',))

    def test_economic_receipt_preserves_counts_and_incremental_costs(self):
        option = {'group': 1, 'kind': 'buy', 'items': {'1001': 2}, 'legal': True}
        receipt = {'group': 1, 'option_sha256': hashlib.sha256(canonical(option)).hexdigest(),
                   'feasible': True, 'incremental_cost': 600.,
                   'sequence': [{'item_id': 1001, 'cost': 300}, {'item_id': 1001, 'cost': 300}]}
        context = {'source_projection': {'gold_est': 650}}
        groups = [[], [[1001, 2]]]
        self.assertTrue(validate_audit_entries([receipt], [option], context, groups)['1']['feasible'])
        changed = deepcopy(receipt); changed['sequence'][1]['item_id'] = 1002
        with self.assertRaisesRegex(ValueError, 'counts/cost'):
            validate_audit_entries([changed], [option], context, groups)
        changed = dict(receipt, feasible=False, incremental_cost=None, sequence=None)
        self.assertFalse(validate_audit_entries([changed], [option], context, groups)['1']['feasible'])

    def test_economic_receipt_cannot_be_transferred_to_a_changed_option(self):
        option = {'group': 0, 'kind': 'save', 'items': {}, 'legal': True}
        receipt = {'group': 0, 'option_sha256': '0'*64, 'feasible': True, 'incremental_cost': 0., 'sequence': []}
        with self.assertRaisesRegex(ValueError, 'different complete option'):
            validate_audit_entries([receipt], [option], {'source_projection': {'gold_est': 100}}, [[]])


if __name__ == '__main__': unittest.main()
