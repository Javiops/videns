"""Validate completed calibration maps and native receipts before scored targets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.recipe_transition_data_v1 import RecipeData
from winprob.native_bias_data_v1 import ROOT, REFERENCE, PROTOCOL, PROTOCOL_SHA, array_hash
from winprob.native_bias_evaluation_v1 import ARMS, require


def json_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


class CompletedNativeBias:
    def __init__(self, directory, decoded_manifest, native_manifest):
        self.directory = Path(directory).resolve(strict=True)
        require(self.directory.is_relative_to(ROOT/'data/runs'), 'fit outside own runs')
        self.bindings = {}; self.targets_loaded = False
        self.fit, self.fit_sha = bound_json(self.directory/'report.json', self.bindings)
        require(self.fit['schema'] == 'native-bias-fit-v1' and self.fit['state'] == 'complete'
                and self.fit['actual_fits'] == 2 and self.fit['selected_states'] == 56288
                and self.fit['selected_games'] == 3518 and self.fit['query_games'] == 2900
                and self.fit['parameters_per_model'] == 9333 and self.fit['training_performed'] is True
                and set(self.fit['models']) == {'0','1'}, 'two completed fixed calibrators required')
        self.dist_path = self.directory/'distribution/manifest.json'
        self.dist, self.dist_sha = bound_json(self.dist_path, self.bindings)
        require(self.dist_sha == self.fit['distribution_manifest_sha256']
                and self.dist['schema'] == 'native-bias-distribution-v1' and self.dist['state'] == 'complete'
                and self.dist['selection_count'] == 2900 and self.dist['groups'] == 9333
                and self.dist['arms'] == ['calibrated','permuted'] and self.dist['models'] == self.fit['models']
                and self.dist['reference_probabilities_used_at_inference'] is True
                and self.dist['protocol_sha256'] == PROTOCOL_SHA and file_hash(PROTOCOL) == PROTOCOL_SHA
                and Path(self.dist['experiment_directory']).resolve() == self.directory, 'frozen calibration distribution changed')
        for flag in ('accuracy_computed','source_runtime_imported','test_payload_reads','confirmation_payload_reads','current_labels_used_in_prediction'):
            require(self.fit[flag] is False and self.dist[flag] is False, 'calibration boundary failed: '+flag)
        require(self.fit['heldout_targets_used_by_its_calibrator'] is False, 'calibration fold boundary failed')
        self.base = RecipeData(REFERENCE); self.bindings.update(self.base.bindings)
        self.matrices = {}
        wanted_files = {'m2.npy','calibrated.npy','permuted.npy','selection.json','group_main.json','neutral_flags.npy'}
        require(set(self.dist['files']) == wanted_files, 'distribution members changed')
        for name, sha in self.dist['files'].items():
            p = (self.dist_path.parent/name).resolve()
            require(p.parent == self.dist_path.parent and file_hash(p) == sha, 'prediction bytes changed')
            self.bindings[str(p)] = sha
        self.ledger = json.loads((self.dist_path.parent/'selection.json').read_text(encoding='utf-8'))
        self.query = self.base.focal; self.folds = self.base.folds[self.query]
        expected = [{**self.base.entries[int(row)], 'query_ordinal': i} for i,row in enumerate(self.query)]
        require(self.ledger == expected and len(self.query) == 2900
                and json.loads((self.dist_path.parent/'group_main.json').read_text()) == self.base.groups, 'query/catalogue order changed')
        for arm in ARMS:
            dtype = np.float32 if arm == 'm2' else np.float64
            matrix = BoundedNpyRows(self.dist_path.parent/f'{arm}.npy', expected_shape=(2900,9333), expected_dtype=dtype)
            self.matrices[arm] = matrix
            for lo in range(0,2900,128):
                block = matrix[lo:min(lo+128,2900)]
                require(np.isfinite(block).all() and np.all(block >= 0)
                        and np.all(np.abs(block.sum(1,dtype=np.float64)-1) < (2e-6 if arm == 'm2' else 2e-12)), 'invalid probabilities')
        require(file_hash(self.dist_path.parent/'m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'native baseline changed')
        self.neutral = np.load(self.dist_path.parent/'neutral_flags.npy', allow_pickle=False)
        require(self.neutral.shape == (2,2900) and self.neutral.dtype == bool, 'neutral flags changed')
        self.input_lock, ilh = bound_json(self.directory/'inputs_lock.json', self.bindings)
        self.model_lock, mlh = bound_json(self.directory/'models_lock.json', self.bindings)
        self.pred_lock, plh = bound_json(self.directory/'predictions_lock.json', self.bindings)
        self.counts_lock, clh = bound_json(self.directory/'counts_lock.json', self.bindings)
        self.control_lock, coh = bound_json(self.directory/'control_lock.json', self.bindings)
        require(ilh == self.dist['inputs_lock_sha256'] and plh == self.dist['prediction_lock_sha256']
                and self.input_lock['state'] == 'inputs_frozen' and self.input_lock['protocol_sha256'] == PROTOCOL_SHA
                and self.input_lock['feature_digests'] == self.base.feature_digests
                and self.input_lock['folds_sha256'] == array_hash(self.base.folds)
                and self.model_lock['state'] == 'both_calibrators_frozen_before_prediction'
                and self.model_lock['models'] == self.fit['models'] and self.model_lock['counts_lock_sha256'] == clh
                and self.model_lock['control_lock_sha256'] == coh
                and self.counts_lock['state'] == 'opposite_fold_counts_frozen_before_fits'
                and self.counts_lock['actual_label_storage_released'] is True
                and self.control_lock['state'] == 'class_control_frozen_before_targets'
                and self.pred_lock['state'] == 'calibrated_distributions_frozen_before_native_scoring'
                and self.pred_lock['files'] == self.dist['files'] and self.pred_lock['models_lock_sha256'] == mlh,
                'calibration freeze lineage changed')
        for name, sha in self.counts_lock['files'].items():
            p = (self.directory/name).resolve()
            require(p.parent == self.directory and file_hash(p) == sha, 'frozen training-count receipt changed')
            self.bindings[str(p)] = sha
        for name,key in (('class_permutation.npy','permutation_sha256'),('basket_sizes.npy','sizes_sha256')):
            p = self.directory/name; require(file_hash(p) == self.control_lock[key], 'class control changed')
            self.bindings[str(p)] = self.control_lock[key]
        for fold,spec in self.fit['models'].items():
            p = (self.directory/spec['file']).resolve()
            require(p.parent == self.directory and file_hash(p) == spec['sha256'], 'bias file changed')
            self.bindings[str(p)] = spec['sha256']
            b = np.load(p,allow_pickle=False)
            require(b.shape == (9333,) and b.dtype == np.float64 and np.isfinite(b).all(), 'invalid bias vector')
            receipt, sha = bound_json(self.directory/f'solver_fold{fold}.json', self.bindings)
            require(sha == spec['solver_sha256'] and receipt == self.fit['fit_receipts'][fold]
                    and receipt['qualified'] is True and receipt['success'] is True and receipt['ridge'] == 1.
                    and receipt['parameters'] == 9333 and receipt['gradient_max_abs'] < 1e-5
                    and receipt['optimizer_evaluations'] <= 2000 and receipt['gradient_rechecks'] == 1,
                    'optimizer qualification changed')
            require(file_hash(self.directory/f'training_identity_fold{fold}.json') == spec['training_identity_sha256']
                    and file_hash(self.directory/f'train_counts_fold{fold}.npy') == spec['counts_sha256'], 'model training identity changed')
        for raw,sha in self.fit['code_sha256'].items():
            p = Path(raw).resolve()
            require(p.is_relative_to(ROOT) and file_hash(p) == sha
                    and file_hash(self.directory/'code_snapshot'/p.relative_to(ROOT)) == sha, 'fit code/snapshot changed')
            self.bindings[str(p)] = sha
        self.source = {}; self.rows = {}
        for world, path in (('16.18', Path(decoded_manifest).resolve(strict=True)), ('16.17', Path(native_manifest).resolve(strict=True))):
            require(path.is_relative_to(ROOT.parent/'build_tracker/data/exports/winprob-native-bias'), 'source decoder artifact outside declared scope')
            value, sha = bound_json(path, self.bindings)
            require(value['schema'] == ('source-native-bias-decoder-v1' if world == '16.18' else 'source-native-bias-native17-v1')
                    and value['state'] == 'complete' and value['world'] == world and value['selection_count'] == 2900
                    and value['distribution_manifest_sha256'] == self.dist_sha and value['protocol_sha256'] == PROTOCOL_SHA,
                    'wrong source decoder/audit')
            for flag in ('target_file_opened', 'test_payload_reads', 'confirmation_payload_reads', 'training_performed',
                         'outcomes_analyzed', 'accuracy_computed', 'consumer_runtime_imported'):
                require(value[flag] is False, 'source boundary failed: '+flag)
            if world == '16.18': require(value['m2_replay_mismatches'] == 0 and value['mutation_checked'] == 8700, 'native replay/mutation incomplete')
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
        for path, sha in self.bindings.items(): require(file_hash(path) == sha, 'completed calibration artifact changed')
