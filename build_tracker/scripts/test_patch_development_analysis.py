"""Analysis reads only synthetic hash-bound paired evidence, never model/data files."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json
from analyze_patch_development import analyze


def rejected(call, message):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError(message)


def synthetic_visit(index):
    metrics_a = {'first_action_exact': 0., 'action_family_match': 0., 'save_card_precision': 0.}
    metrics_b = {'first_action_exact': 1., 'action_family_match': 1., 'save_card_precision': None}
    return {'schema_version': 1, 'split': 'val', 'match_id': f'game_{index // 6}', 'participant_id': 3,
        'ts': 60000 * (index + 1), 'champion': 'Rare' if index < 2 else 'Ashe',
        'role': 'TOP' if index < 2 else 'BOTTOM', 'save_kind': 'build_spend',
        'minute': index + 1, 'budget': 700., 'inventory': [],
        'actual_item_counts': {'1036': 1, '2055': 1}, 'next_final_id': 3133,
        'policies': {
            'A': {'metrics': metrics_a, 'metric_n': {key: 1 for key in metrics_a},
                'unaffordable': 0, 'unseen_truth_item_counts': {'2055': 1} if index < 2 else {},
                'displayed_options': [{'kind': 'save', 'item_counts': {}, 'purchase_sequence': []}]},
            'B': {'metrics': metrics_b, 'metric_n': {key: int(value is not None) for key, value in metrics_b.items()},
                'unaffordable': 0, 'unseen_truth_item_counts': None,
                'displayed_options': [{'kind': 'buy', 'item_counts': {'1036': 1, '2055': 1},
                    'purchase_sequence': [{'item_id': 1036, 'cost': 350}, {'item_id': 2055, 'cost': 75}]}]},
        }}


def publish_fixture(run):
    rows = [synthetic_visit(i) for i in range(36)]
    outcome_path = run / 'policy_outcomes_val_fixture_vs_conditional.jsonl'
    outcome_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    metrics = {}
    for tag in ('A', 'B'):
        metrics[tag] = {'unaffordable': 0}
        for metric in rows[0]['policies'][tag]['metrics']:
            defined = [row['policies'][tag]['metrics'][metric] for row in rows
                       if row['policies'][tag]['metric_n'][metric]]
            value = sum(defined) / len(defined) if defined else None
            metrics[tag][metric] = {'n': len(defined), 'value': value,
                'ci95': [value, value] if defined else None, 'status': 'measured' if defined else 'not_measured'}
    report = {'purpose': 'patch_development', 'split': 'val', 'manifest': {'patch': '16.17',
        'export_fingerprint': 'synthetic_frozen_export', 'split_metadata': {'validation_exposure': 'synthetic only'}},
        'final_test_evaluation': False, 'promotion_allowed': False,
        'primary_metrics': ['first_action_exact', 'action_family_match'], 'metrics': metrics,
        'visits': 36, 'games': 6, 'artifacts': {'A': 'candidate.pt', 'B': 'conditional.json'},
        'artifact_sha256': {'A': 'synthetic_model', 'B': 'synthetic_baseline'},
        'paired_outcomes': {'path': str(outcome_path), 'sha256': sha256(outcome_path),
            'bytes': outcome_path.stat().st_size, 'rows': 36, 'schema_version': 1},
        'paired_diff_a_minus_b': {key: {'n_a': 36, 'n_b': 36, 'n_games': 6,
            'a': 0., 'b': 1., 'diff': -1., 'ci95': [-1., -1.]}
            for key in ('first_action_exact', 'action_family_match')}}
    report_path = run / 'policy_eval_val_fixture_vs_conditional.json'
    write_json(report_path, report)
    return report_path, report, outcome_path


with tempfile.TemporaryDirectory() as temp:
    run = Path(temp)
    report_path, report, outcome_path = publish_fixture(run)
    sealed_test = run / 'visits_test.jsonl'
    sealed_test.write_text('must stay sealed', encoding='utf-8')
    model_path = run / 'candidate.pt'
    model_path.write_text('must not load a model', encoding='utf-8')
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path in (sealed_test, model_path):
            raise AssertionError(f'Analysis opened forbidden input: {path.name}')
        return original_open(path, *args, **kwargs)
    with patch.object(Path, 'open', guarded_open):
        output = run / 'analysis'
        result = analyze(report_path, output)
        assert (output / 'analysis.md').is_file()
        assert result['visits'] == 36 and result['games'] == 6
        assert result['source_report_sha256'] == sha256(report_path)
        assert result['source_report'] == str(report_path)
        assert result['model_reevaluated'] is result['final_test_read'] is result['training_changes_executed'] is False
        assert result['global_primary_metrics']['first_action_exact']['paired_a_minus_b']['ci95'] == [-1., -1.]
        slices = {(item['dimension'], item['value']): item for item in result['slices']}
        sparse = slices[('champion', 'Rare')]
        assert sparse['coverage'] == 'sparse' and sparse['visits'] == 2 and sparse['games'] == 1
        assert sparse['metrics']['first_action_exact']['recommendation_support'] is False
        supported = slices[('champion', 'Ashe')]
        assert supported['coverage'] == 'supported' and supported['visits'] == 34 and supported['games'] == 6
        assert supported['metrics']['first_action_exact']['recommendation_support'] is True
        assert all(cell['value'] not in ('Rare', 'TOP') for cell in result['supported_cells_with_smallest_candidate_minus_baseline'])
        paired = supported['metrics']['first_action_exact']['paired_common_population']
        assert paired['n'] == 34 and paired['a_sum'] == 0. and paired['b_sum'] == 34. and paired['diff_a_minus_b'] == -1.
        # Save-card precision has no common population; it must not gain a
        # fabricated paired denominator or a numeric zero difference.
        save = result['global_recomputed']['metrics']['save_card_precision']
        assert save['A']['n'] == 36 and save['B']['n'] == 0
        assert save['paired_common_population']['n'] == 0
        assert save['paired_common_population']['diff_a_minus_b'] is None
        for kind in ('underbuy', 'save_false_positive', 'ward_miss'):
            error = result['error_classes'][kind]
            assert error['policies']['A']['n'] == 36 and error['policies']['B']['n'] == 0
            assert len(error['policies']['A']['examples']) == 3
            assert len({example['match_id'] for example in error['policies']['A']['examples']}) == 3
        assert result['error_classes']['unseen_label']['policies']['A']['n'] == 2
        # Ward misses caused entirely by leading SAVE are not evidence that
        # ward ranking itself needs changing; avoid duplicate causal proposals.
        assert result['error_classes']['ward_miss_while_buying']['policies']['A']['n'] == 0
        assert {item['hypothesis'] for item in result['next_controlled_changes']} == {'save_gate'}
        assert all(item['candidate_n'] == 36 and item['population_n'] == 36 for item in result['next_controlled_changes'])
        # Strict JSON consumer behavior, including null values, is preserved.
        def reject_constant(value): raise AssertionError(value)
        parsed = json.loads((output / 'analysis.json').read_text(encoding='utf-8'), parse_constant=reject_constant)
        assert parsed == result
        rejected(lambda: analyze(report_path, output), 'Existing analysis overwritten')
        rejected(lambda: analyze(sealed_test, run / 'bad'), 'Dataset payload accepted as report')
        rejected(lambda: analyze(report_path, run.parent / 'outside-analysis'), 'Outside run output accepted')

        # Hash-bound bytes and declared estimates both matter. No artifact is
        # published for corrupt evidence, even when it is parseable JSON.
        with outcome_path.open('ab') as stream: stream.write(b'\n')
        rejected(lambda: analyze(report_path, run / 'changed'), 'Changed outcomes accepted')
        assert not (run / 'changed').exists()
        report_path, report, outcome_path = publish_fixture(run)
        report['metrics']['A']['first_action_exact']['value'] = .5
        write_json(report_path, report)
        rejected(lambda: analyze(report_path, run / 'wrong-metric'), 'Unreproducible aggregate accepted')
        report_path, report, outcome_path = publish_fixture(run)
        report['paired_diff_a_minus_b']['first_action_exact']['diff'] = 0.
        write_json(report_path, report)
        rejected(lambda: analyze(report_path, run / 'wrong-paired'), 'Unreproducible paired difference accepted')
        report_path, report, outcome_path = publish_fixture(run)
        report['paired_outcomes']['path'] = str(sealed_test)
        write_json(report_path, report)
        rejected(lambda: analyze(report_path, run / 'test-outcomes'), 'Test payload accepted as outcomes')

print('ok development analysis: hash/population binding, strict JSON, common paired populations, sparse-cell thresholds, bounded concrete examples, evidence-conditioned proposals and no model/test reads')
