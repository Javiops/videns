"""Read completed route predictions and native receipts before target access."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.recipe_transition_data_v1 import RecipeData, require
ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'data/runs/route-probe-fit-20261003-v3/experiment/distribution'
PROTOCOL = ROOT/'docs/causal_student_probe_protocol_v1.md'
PROTOCOL_SHA = 'fc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e'
OLD = ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2'
from winprob.causal_student_evaluation_v1 import ARMS


def json_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


class CompletedStudent:
    def __init__(self, directory, decoded_manifest, native_manifest):
        self.directory = Path(directory).resolve(strict=True)
        require(self.directory.is_relative_to(ROOT/'data/runs'), 'fit directory outside own runs')
        self.bindings = {}; self.targets_loaded = False
        self.training, self.training_sha = bound_json(self.directory/'report.json', self.bindings)
        require(self.training['schema'] == 'causal-student-training-v1' and self.training['state'] == 'complete'
                and self.training['query_games'] == 2900 and self.training['selected_states'] == 56288
                and self.training['selected_games'] == 3518 and self.training['training_performed'] is True
                and self.training['updates'] == 26460 and len(self.training['models']) == 6
                and self.training['accuracy_computed'] is False, 'completed unscored six-student comparison required')
        self.dist_path = self.directory/'distribution/manifest.json'
        self.dist, self.dist_sha = bound_json(self.dist_path, self.bindings)
        require(self.dist_sha == self.training['distribution_manifest_sha256']
                and self.dist['schema'] == 'causal-student-distribution-v1' and self.dist['state'] == 'complete'
                and self.dist['selection_count'] == 2900 and self.dist['groups'] == 9333
                and self.dist['protocol_sha256'] == PROTOCOL_SHA
                and self.dist['arms'] == ['hard', 'distilled', 'shuffled_teacher']
                and self.dist['models'] == self.training['models']
                and self.dist['student_inference_uses_teacher_or_labels'] is False
                and self.training['student_inference_uses_teacher_or_labels'] is False,
                'frozen complete-student prediction contract changed')
        for flag in ('accuracy_computed', 'source_runtime_imported', 'test_payload_reads', 'confirmation_payload_reads'):
            require(self.training[flag] is False and self.dist[flag] is False, 'route boundary failed: '+flag)
        require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'route protocol changed')
        # The original loader admits every FIT identity before prediction payloads.
        self.base = RecipeData(OLD); self.bindings.update(self.base.bindings)
        self.ledger = None; self.matrices = {}
        for name, sha in self.dist['files'].items():
            p = (self.dist_path.parent/name).resolve()
            require(p.parent == self.dist_path.parent and file_hash(p) == sha, 'prediction artifact changed')
            self.bindings[str(p)] = sha
        self.ledger = json.loads((self.dist_path.parent/'selection.json').read_text(encoding='utf-8'))
        self.query = self.base.focal; self.folds = self.base.folds[self.query]
        expected = [{**self.base.entries[int(index)], 'query_ordinal': i} for i, index in enumerate(self.query)]
        require(self.ledger == expected and len(self.query) == 2900, 'original focal population changed')
        require(json.loads((self.dist_path.parent/'group_main.json').read_text()) == self.base.groups, 'full catalogue changed')
        for arm in ARMS:
            self.matrices[arm] = BoundedNpyRows(self.dist_path.parent/f'{arm}.npy', expected_shape=(2900, 9333),
                                              expected_dtype=np.float32 if arm == 'm2' else np.float64)
            matrix = self.matrices[arm]
            require(matrix.shape == (2900, 9333) and matrix.dtype == (np.float32 if arm == 'm2' else np.float64), 'distribution axis/precision changed')
            for lo in range(0, 2900, 128):
                block = matrix[lo:min(lo+128, 2900)]
                require(np.isfinite(block).all() and np.all(block >= 0)
                        and np.all(np.abs(block.sum(1, dtype=np.float64)-1) < (2e-6 if arm == 'm2' else 2e-12)), 'invalid probabilities')
        self.neutral = np.load(self.dist_path.parent/'neutral_flags.npy', allow_pickle=False)
        require(self.neutral.shape == (3, 2900) and self.neutral.dtype == bool, 'neutral flags changed')
        self.input_lock, ilh = bound_json(self.directory/'inputs_lock.json', self.bindings)
        self.pred_lock, plh = bound_json(self.directory/'predictions_lock.json', self.bindings)
        self.model_lock, mlh = bound_json(self.directory/'models_lock.json', self.bindings)
        self.control_lock, clh = bound_json(self.directory/'teacher_control_lock.json', self.bindings)
        self.inference_boundary, ibh = bound_json(self.directory/'inference_boundary.json', self.bindings)
        require(ilh == self.dist['inputs_lock_sha256'] and plh == self.dist['prediction_lock_sha256']
                and self.input_lock['state'] == 'student_inputs_frozen_before_targets'
                and self.input_lock['protocol_sha256'] == PROTOCOL_SHA
                and self.pred_lock['state'] == 'all_student_distributions_frozen_before_decoding_or_scoring'
                and self.pred_lock['files'] == self.dist['files']
                and self.pred_lock['models_lock_sha256'] == mlh and self.pred_lock['inference_boundary_sha256'] == ibh
                and self.model_lock['state'] == 'all_six_final_models_frozen_before_prediction'
                and self.model_lock['models'] == self.training['models'] and self.model_lock['updates'] == 26460
                and self.model_lock['teacher_controls_sha256'] == clh
                and self.control_lock['state'] == 'teacher_controls_frozen_before_actual_labels'
                and self.inference_boundary == {'state': 'student_supervision_storage_released',
                    'teacher_gpu_available': False, 'observed_labels_available': False, 'forward_accepts_only_public_inputs': True},
                'student model/control/inference freeze lineage changed')
        for name, sha in self.training['models'].items():
            path = (self.directory/name).resolve()
            require(path.parent == self.directory and file_hash(path) == sha, 'final model checkpoint changed')
            self.bindings[str(path)] = sha
        for fold in (0, 1):
            path = self.directory/f'teacher_permutation_fold{fold}.npy'
            require(file_hash(path) == self.control_lock['files'][str(fold)], 'teacher control changed')
            self.bindings[str(path)] = self.control_lock['files'][str(fold)]
        require(not self.neutral.any(), 'independent student should not request a baseline fallback')
        for path, sha in self.training['code_sha256'].items():
            p = Path(path).resolve()
            require(p.is_relative_to(ROOT) and file_hash(p) == sha
                    and file_hash(self.directory/'code_snapshot'/p.relative_to(ROOT)) == sha, 'forecast code/snapshot changed')
            self.bindings[str(p)] = sha
        native, nh = bound_json(NATIVE/'manifest.json', self.bindings)
        require(nh == 'b69a407f794bbebebaf0be7da8c52d50a61629b2f755df3f92c7d95df3173e2e'
                and file_hash(self.dist_path.parent/'m2.npy') == native['files']['m2.npy']
                and Path(self.dist['experiment_directory']).resolve() == self.directory, 'native focal prior/experiment identity changed')
        self.source = {}; self.rows = {}
        for world, path in (('16.18', Path(decoded_manifest).resolve(strict=True)), ('16.17', Path(native_manifest).resolve(strict=True))):
            require(path.is_relative_to(ROOT.parent/'build_tracker/data/exports/winprob-causal-student'), 'source decoder artifact outside declared scope')
            value, sha = bound_json(path, self.bindings)
            require(value['schema'] == ('source-causal-student-decoder-v1' if world == '16.18' else 'source-causal-student-native17-v1')
                    and value['state'] == 'complete' and value['world'] == world and value['selection_count'] == 2900
                    and value['distribution_manifest_sha256'] == self.dist_sha and value['protocol_sha256'] == PROTOCOL_SHA,
                    'wrong source decoder/audit')
            for flag in ('target_file_opened', 'test_payload_reads', 'confirmation_payload_reads', 'training_performed',
                         'outcomes_analyzed', 'accuracy_computed', 'consumer_runtime_imported'):
                require(value[flag] is False, 'source boundary failed: '+flag)
            if world == '16.18': require(value['m2_replay_mismatches'] == 0 and value['mutation_checked'] == 11600, 'native replay/mutation incomplete')
            for raw, want in {**value['bindings_sha256'], **value['code_sha256']}.items():
                require(Path(raw).name != 'targets.jsonl' and file_hash(raw) == want, 'source binding changed/contains targets')
                self.bindings[raw] = want
            rp = path.parent/'rows.jsonl'; require(file_hash(rp) == value['files'][rp.name], 'source result rows changed')
            self.bindings[str(rp)] = value['files'][rp.name]
            rows = json_lines(rp); require(len(rows) == 2900, 'source result query count changed')
            for i, (row, entry) in enumerate(zip(rows, self.ledger, strict=True)):
                require(row['ordinal'] == i and row['sample_id'] == entry['sample_id'], 'source result query order changed')
            self.source[world] = {'manifest': value, 'manifest_sha256': sha, 'manifest_path': str(path)}; self.rows[world] = rows
        self.legality = {}; self.top3 = {arm: np.full((2900, 3), -1, dtype=np.int32) for arm in ARMS}
        self.costs = {}
        for world in ('16.18', '16.17'):
            failures = {arm: 0 for arm in ARMS}; option_counts = {arm: 0 for arm in ARMS}
            first_cost = {arm: [] for arm in ARMS}
            for i, row in enumerate(self.rows[world]):
                for arm in ARMS:
                    options = self.rows['16.18'][i]['options'][arm]; audits = row['audits'][arm]
                    require(1 <= len(options) <= 3 and len(audits) == len(options), 'native option/audit dimensions changed')
                    ids = [option['group'] for option in options]
                    require(len(set(ids)) == len(ids) and all(type(g) is int and 0 <= g < 9333 for g in ids), 'duplicate/out-of-range options')
                    for option, check in zip(options, audits, strict=True):
                        g = option['group']
                        require(option == {'kind': 'save' if g == 0 else 'buy', 'items': {str(k): c for k, c in self.base.groups[g]},
                                           'group': g, 'legal': True} and check['group'] == g and type(check['feasible']) is bool,
                                'native counted action/audit identity changed')
                        failures[arm] += int(not check['feasible']); option_counts[arm] += 1
                    if audits[0]['feasible']:
                        cost = audits[0]['incremental_cost']; require(np.isfinite(cost) and cost >= 0, 'invalid native cost')
                        first_cost[arm].append(cost)
                    if world == '16.18': self.top3[arm][i, :len(ids)] = ids
            counts = self.source[world]['manifest']['counts']
            require(all(failures[arm] == counts.get(arm+'_infeasible', 0)
                        and option_counts[arm] == counts[arm+'_options'] for arm in ARMS), 'native audit totals do not reproduce')
            self.legality[world] = failures
            self.costs[world] = {arm: {'feasible_first_count': len(v), 'total_queries': 2900,
                'mean_incremental_cost_on_feasible_first': float(np.mean(v)) if v else None} for arm, v in first_cost.items()}
        self.revalidate()

    def scoring_lock(self):
        return {'state': 'inputs_frozen', 'native_manifest_sha256': self.base.digest,
                'feature_digests': self.base.feature_digests, 'distribution_manifest_sha256': self.dist_sha}

    def load_targets(self, scoring_lock_path):
        self.revalidate()
        require(json.loads(Path(scoring_lock_path).read_text(encoding='utf-8')) == self.scoring_lock(), 'scoring lock differs from complete predictions')
        full = self.base.load_targets(scoring_lock_path); self.targets_loaded = True
        return {name: values[self.query] for name, values in full.items()}, full

    def scores(self, labels):
        require(self.targets_loaded, 'scoring requires explicit target stage after all prediction/audit locks')
        y = np.asarray(labels['main']); require(y.shape == (2900,), 'scoring labels changed')
        result = {}
        for arm, matrix in self.matrices.items():
            mass = np.empty(2900, dtype=np.float64)
            for lo in range(0, 2900, 128):
                hi = min(lo+128, 2900); block = matrix[lo:hi]
                value = np.asarray(block[np.arange(hi-lo), y[lo:hi]], dtype=np.float64)
                if arm == 'm2': value /= block.sum(1, dtype=np.float64)
                mass[lo:hi] = value
            require(np.isfinite(mass).all() and np.all(mass > 0), 'target has no finite catalogue likelihood')
            result[arm] = {'top1': self.top3[arm][:, 0], 'top3': self.top3[arm], 'catalogue_target_logp': np.log(mass)}
        return result

    def revalidate(self):
        self.base.revalidate()
        for path, sha in self.bindings.items(): require(file_hash(path) == sha, 'completed route artifact changed')
