"""Synthetic, patch-pinned tests for train-only paired displayed-policy evidence."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from app.conditional_policy import (ConditionalPolicy, action_of, keys_for,
    purchase_sequence, require_explicit_slot_state)
from app.dataset_artifacts import sha256, write_json
from app.ddragon import dragon_for_patch
from app.shop_econ import SAVE_ITEM
import eval_policy as ep
import train_prefix as tp


dragon = dragon_for_patch("16.17")


def row(inventory=(), gold=700, labels=(1036, 1036), match_id="train_A", save_kind="build_spend"):
    return {"match_id": match_id, "participant_id": 3, "champion": "Ashe", "champion_id": 22, "role": "BOTTOM",
        "inventory": list(inventory), "gold": gold, "gold_est": gold, "ts": 60000,
        "gold_est_version": "prequential-v3", "slot_state_version": "role-slots-v1",
        "bot_quest_complete": False, "unmodeled_regular_slots": 0,
        "others": [{"slot_state_version": "role-slots-v1", "bot_quest_complete": False}],
        "label_ids": list(labels), "label_counts": {str(i): n for i, n in Counter(labels).items()},
        "target_encoding": "item-counts-v1", "save_kind": save_kind}


def rejected(call, message):
    try:
        call()
    except (ValueError, SystemExit):
        return
    raise AssertionError(message)


# A full inventory requires a combine before the unordered ward IDs. Prices
# are independent patch facts, then replayed by the common policy checker.
ingredients = [1036, 1036, 2022, 1029, 1028, 1052]
combine_row = row(ingredients, 250, (2055, 2055, 3133))
before = deepcopy(combine_row)
sequence = purchase_sequence(action_of(combine_row), combine_row, 250, dragon)
assert sequence == [{"item_id": 3133, "cost": 100}, {"item_id": 2055, "cost": 75}, {"item_id": 2055, "cost": 75}]
assert not ep.check_displayed_purchases([{"kind": "buy", "items": sequence}], combine_row, 250, dragon)
assert purchase_sequence(action_of(combine_row), combine_row, 249, dragon) is None
assert combine_row == before
assert purchase_sequence((1036, 3133), row(gold=1050), 1050, dragon) is None
assert sum(i['cost'] for i in purchase_sequence((1036, 3133), row(gold=1400), 1400, dragon)) == 1400
assert action_of({"label_counts": {"1036": 5}, "target_encoding": "item-counts-v1"}) == (1036,) * 5
rejected(lambda: action_of({"save_kind": "no_buy_death", "label_ids": [1036]}), "Invented SAVE accepted")

# Unknown completion is allowed by conservative legacy decoding, but never by
# a development comparison (including another player's board state).
for changed in ({"slot_state_version": None}, {"bot_quest_complete": None},
                {"bot_quest_complete": 0}, {"unmodeled_regular_slots": None},
                {"others": [{"slot_state_version": "role-slots-v1", "bot_quest_complete": None}]}):
    rejected(lambda: require_explicit_slot_state({**row(), **changed}), "Unknown state admitted")


with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    dataset, run = base / 'dataset', base / 'run'
    dataset.mkdir(); run.mkdir()
    train_path, val_path = dataset / 'visits_train.jsonl', dataset / 'visits_val.jsonl'
    train = [row() for _ in range(5)] + [row(gold=0, labels=(SAVE_ITEM,), save_kind="no_buy_death")]
    validation = [row(match_id="val_A"), row(gold=0, labels=(SAVE_ITEM,), match_id="val_B", save_kind="no_buy_death")]
    for path, rows in ((train_path, train), (val_path, validation)):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    test_path = dataset / 'visits_test.jsonl'
    test_path.write_text('FINAL TEST MUST REMAIN SEALED', encoding='utf-8')
    manifest = {'eval_version': 'synthetic', 'export_fingerprint': 'frozen-synthetic', 'patch': '16.17',
        'ddragon_version': dragon.version, 'static_data_sha256': dragon.signature, 'probe': True,
        'gold_est_versions': {'prequential-v3': 3}}
    generation = {'purpose': 'patch_development', 'files': {'visits_train.jsonl': {'sha256': sha256(train_path)}}}
    assignments = {'train_A': 'train', 'val_A': 'val', 'val_B': 'val'}
    write_json(dataset / 'split_manifest.json', manifest)
    write_json(dataset / 'generation_manifest.json', generation)
    write_json(dataset / 'split_assignment.json', assignments)
    calls = []
    real_open = Path.open

    def guard_open(path, *args, **kwargs):
        assert path != test_path, "Final test payload was opened"
        return real_open(path, *args, **kwargs)

    def stream(path):
        calls.append(path.name)
        assert path == train_path
        yield from tp.stream_rows(path)

    with patch.object(Path, 'open', guard_open):
        policy = ConditionalPolicy.fit(train_path, stream, dragon, manifest, generation, assignments)
        assert calls == ['visits_train.jsonl', 'visits_train.jsonl']
        assert policy.evidence['train_rows'] == 6 and policy.evidence['train_games'] == 1
        clean = ep.serving_row(validation[0], 700)
        predicted = policy.predict_options(clean, allow_save=True)
        assert Counter(x['item_id'] for x in predicted[0]['items']) == Counter({1036: 2})
        assert all(key not in clean for key in ep.LABEL_FIELDS)
        score, over = ep.score_visit(policy, validation[0], Counter({1036: 2}), 'build_spend', 700, dragon, {1036})
        assert score['first_action_exact'] == score['action_family_match'] == 1 and not over
        assert policy.predict_options(ep.serving_row(validation[1], 0), allow_save=True)[0]['kind'] == 'save'
        assert policy.predict_options(ep.serving_row(validation[1], 0), allow_save=False) == []
        rejected(lambda: ConditionalPolicy.fit(val_path, stream, dragon, manifest, generation, assignments), 'Fit read VAL')
        rejected(lambda: ConditionalPolicy.fit(train_path, stream, dragon, manifest, generation, {'train_A': 'val'}), 'Wrong fold accepted')
        rejected(lambda: ConditionalPolicy.fit(train_path, stream, dragon, {**manifest, 'static_data_sha256': 'other'}, generation, assignments), 'Wrong patch static data accepted')
        rejected(lambda: ConditionalPolicy.fit(train_path, stream, dragon, manifest, {'files': {'visits_train.jsonl': {'sha256': 'changed'}}}, assignments), 'Changed TRAIN accepted')

        # Full CLI orchestration with synthetic frozen rows and a known policy.
        # Only expensive generation-chain validation is stubbed; artifact,
        # provenance, fitting, serving, scoring and report writing are real.
        artifact = run / 'candidate.pt'
        artifact.write_bytes(b'synthetic predictor fixture')

        class Candidate:
            def __init__(self, path):
                self.artifact = path
                self.artifact_sha256 = sha256(path)
                self.dragon = dragon
                self.config = {'gold_x': True, 'gold_input': 'prequential-v3'}
                self.label_ids = [1036]  # SAVE truth is outside this fixture's vocabulary.
                self.threshold = .8
                self.trained_at = 'synthetic'
                self.provenance = {**manifest, 'manifest_sha256': ep.manifest_sha256(dataset / 'split_manifest.json'),
                    'training_split': 'train', 'selection_split': 'validation', 'gold_input': 'prequential-v3',
                    'reconstruction_version': tp.RECONSTRUCTION_VERSION, 'run_plan_sha256': 'bound-plan',
                    'training_stage': 'graft'}

            def predict_options(self, served, **kwargs):
                assert not any(key in served for key in ep.LABEL_FIELDS)
                assert served['gold'] == served['gold_est']
                items = purchase_sequence((1036, 1036), served, served['gold'], dragon)
                return [{'kind': 'buy', 'items': items}] if items else []

        plan = {'primary_metrics': ['first_action_exact', 'action_family_match'],
                'training_stages': {'graft': {'PREFIX_OUT': 'candidate.pt'}},
                'evaluation_seed': 16, 'evaluation_bootstrap': 20}
        with patch.object(ep, 'ML_DIR', dataset), patch.object(ep, 'RUN_DIR', run), \
             patch.object(ep, 'SPLIT_MANIFEST_PATH', dataset / 'split_manifest.json'), \
             patch('app.dataset_generations.verify_generation', return_value=generation), \
             patch('app.dataset_generations.validate_run_directory'), \
             patch('app.dataset_generations.validate_development_run_plan', return_value=(plan, 'bound-plan')), \
             patch.dict(os.environ, {'PREFIX_RUN_PLAN': str(run / 'run_plan.json')}), \
             patch('app.predictor.Predictor', Candidate), \
             patch.object(sys, 'argv', ['eval_policy.py', '--artifact', 'candidate.pt', '--compare-conditional', '--games', '0', '--bootstrap', '20']):
            good_argv = list(sys.argv)
            for artifact_name in ('trunk.pt', str(artifact), '../candidate.pt'):
                with patch.object(sys, 'argv', [good_argv[0], '--artifact', artifact_name, *good_argv[3:]]):
                    rejected(ep.main, 'Unplanned/outside-run candidate accepted')
            with patch.object(sys, 'argv', [*good_argv, '--seed', '17']):
                rejected(ep.main, 'Changed bootstrap seed accepted')
            with patch.object(sys, 'argv', [*good_argv, '--bootstrap', '21']):
                rejected(ep.main, 'Changed bootstrap resample count accepted')

            class TrunkMasqueradingAsCandidate(Candidate):
                def __init__(self, path):
                    super().__init__(path)
                    self.provenance['training_stage'] = 'trunk'

            with patch('app.predictor.Predictor', TrunkMasqueradingAsCandidate):
                rejected(ep.main, 'Trunk artifact accepted under candidate filename')
            assert not (run / 'conditional_policy.json').exists(), 'Invalid run fitted/wrote baseline'
            ep.main()
        def reject_nonfinite(value):
            raise AssertionError(f'Nonstandard numeric JSON constant: {value}')
        report = json.loads((run / 'policy_eval_val_candidate_vs_conditional_policy.json').read_text(encoding='utf-8'),
                            parse_constant=reject_nonfinite)
        assert report['visits'] == 2 and report['games'] == 2
        assert report['metrics']['A']['first_action_exact']['value'] == .5
        assert report['metrics']['B']['first_action_exact']['value'] == 1
        assert report['paired_diff_a_minus_b']['first_action_exact']['diff'] == -.5
        assert report['paired_diff_a_minus_b']['first_action_exact']['n_a'] == 2
        assert report['conditional_baseline']['train_sha256'] == sha256(train_path)
        assert report['artifact_sha256']['B'] == sha256(run / 'conditional_policy.json')
        assert report['manifest']['patch'] == '16.17'
        assert report['promotion_allowed'] is report['final_test_evaluation'] is False
        undefined_summary = report['metrics']['A']['save_card_precision']
        assert undefined_summary['value'] is undefined_summary['ci95'] is None
        assert undefined_summary['n'] == 0 and undefined_summary['status'] == 'not_measured'
        assert undefined_summary['reason'] == 'no_population'
        undefined_pair = report['paired_diff_a_minus_b']['save_card_precision']
        assert undefined_pair['diff'] is undefined_pair['ci95'] is undefined_pair['a'] is None
        assert undefined_pair['b'] == 1 and undefined_pair['n_a'] == 0 and undefined_pair['n_b'] == 1
        assert undefined_pair['status'] == 'not_measured' and undefined_pair['reason'] == 'no_population'
        assert undefined_pair['missing_population'] == ['A']
        empty_family = report['paired_diff_a_minus_b']['action_family_match']
        assert empty_family['n_games'] == empty_family['n_a'] == empty_family['n_b'] == 0
        assert empty_family['diff'] is empty_family['ci95'] is None
        assert not (dataset / 'conditional_policy_val.json').exists()
        capture_path = run / 'policy_outcomes_val_candidate_vs_conditional.jsonl'
        capture = report['paired_outcomes']
        assert capture['path'] == str(capture_path) and capture['sha256'] == sha256(capture_path)
        assert capture['bytes'] == capture_path.stat().st_size and capture['rows'] == report['visits'] == 2
        captured = [json.loads(line) for line in capture_path.read_text(encoding='utf-8').splitlines()]
        assert [visit['match_id'] for visit in captured] == ['val_A', 'val_B']
        assert all(visit['participant_id'] == 3 and set(visit['policies']) == {'A', 'B'} for visit in captured)
        assert captured[0]['actual_item_counts'] == {'1036': 2}
        assert captured[0]['policies']['A']['displayed_options'][0]['item_counts'] == {'1036': 2}
        assert captured[0]['policies']['A']['displayed_options'][0]['purchase_sequence'] == [
            {'item_id': 1036, 'cost': 350}, {'item_id': 1036, 'cost': 350}]
        assert captured[1]['policies']['A']['unseen_truth_item_counts'] == {str(SAVE_ITEM): 1}
        assert captured[1]['policies']['A']['displayed_options'] == []
        assert all('puuid' not in visit and 'others' not in visit for visit in captured)
        for tag in ('A', 'B'):
            for metric in ep.METRICS:
                values = [visit['policies'][tag]['metrics'][metric] for visit in captured
                          if visit['policies'][tag]['metric_n'][metric]]
                assert len(values) == report['metrics'][tag][metric]['n']
                if values:
                    assert sum(values) / len(values) == report['metrics'][tag][metric]['value']

        # A failed stream retains partial evidence but never publishes a complete
        # path, and neither complete nor partial output is silently overwritten.
        failed_capture = run / 'interrupted.jsonl'
        try:
            with ep.outcome_capture(failed_capture) as (write, _binding):
                write({'visit': 1})
                raise ValueError('injected interruption')
        except ValueError:
            pass
        assert not failed_capture.exists() and failed_capture.with_suffix('.jsonl.partial').exists()
        def reopen(path):
            with ep.outcome_capture(path):
                raise AssertionError('An existing capture was overwritten')
        rejected(lambda: reopen(failed_capture), 'Existing partial capture accepted')
        rejected(lambda: reopen(capture_path), 'Existing final capture accepted')


# Paired ratios retain games with an option shown by only one policy.
a = {'one': [{'save_card_precision': 1.}], 'two': [{'save_card_precision': 0.}]}
b = {'one': [{'save_card_precision': None}], 'two': [{'save_card_precision': 0.}]}
paired = ep.paired_diff(a, b, 'save_card_precision', 20, 16)
assert paired['a'] == .5 and paired['b'] == 0. and paired['diff'] == .5
assert paired['n_a'] == 2 and paired['n_b'] == 1 and paired['n_games'] == 2
rejected(lambda: ep.paired_diff(a, {'one': b['one']}, 'save_card_precision', 20, 16), 'Unpaired games accepted')

print('ok paired conditional: frozen TRAIN only, pinned static data, explicit slots, retained quantities, ordered combine costs, label-free common scoring, paired intervals, isolated evidence and sealed final test')
