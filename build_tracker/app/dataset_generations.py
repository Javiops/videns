"""Readers accept only complete, content-bound dataset generations."""
import json
from pathlib import Path
from app.dataset_artifacts import sha256, canonical_sha

SPLITS = ('train','val','test')
PURPOSES = ('pipeline_smoke_only', 'patch_development', 'patch_development_1618', 'patch_development_1619')
DEVELOPMENT_PURPOSES = ('patch_development', 'patch_development_1618', 'patch_development_1619')
# Scoped development benchmarks: one entry per patch, same contract (patch day 16.19, 2026-09-29). The 16.18
# entry carries exactly the literals the validators used before; nothing about 16.18 evidence changes.
SCOPED_DEVELOPMENT = {
    'patch_development_1618': {'patch': '16.18', 'ddragon': '16.18.1', 'recovery_version': 'footwear-viego-16.18-v1',
        'allowed_reconstruction': ('decision-start-footwear-16.18-v1', 'decision-start-footwear-16.18-v2'),
        'authorizations': ('owner authorized steps 2-4 on 2026-09-12',
                           'owner instructed steps 1-3 on 2026-09-13',
                           'owner authorized design B expansion (snapshot, qualification, export, Bf d64) on 2026-09-23')},
    'patch_development_1619': {'patch': '16.19', 'ddragon': '16.19.1', 'recovery_version': 'footwear-viego-16.19-v1',
        'allowed_reconstruction': ('decision-start-footwear-16.19-v1',),
        'authorizations': ('owner job 2 (2026-09-29): qualify 16.19, fix a temporal cohort, measure frozen M2 and R1; no training',)},
}
BENCHMARK_1618_FILES = {'benchmark_admission.json', 'source_plan.json', 'original_qualification.json',
    'original_completion.json', 'original_protocol.json', 'original_static_review.json',
    'decision_admission.json', 'split_metadata.json', 'preserved_split_assignment.json',
    'original_split_plan.json','original_split_assignment.json','original_generation_manifest.json'}
REQUIRED = {'split_manifest.json','split_assignment.json','split_plan.json','coverage.json',
            'qualification.json','qualification_code.py','semantic_audit.json','audit_code.py','corpus_manifest.json'} | {f'visits_{s}.jsonl' for s in SPLITS}


def validate_recovered_evidence(qualification, audit, corpus, admission, admission_sha256):
    """A row mask is a versioned audit result, never an eligibility bypass."""
    version = 'footwear-timer-slots-v1'
    expected = {'version':version, 'passed':True, 'semantic_audit_sha256':qualification['semantic_audit_sha256'],
                'decision_admission_sha256':admission_sha256}
    if (corpus.get('recovery_version') != version or qualification.get('recovery_version') != version
        or corpus.get('patch') != '16.17' or admission.get('patch') != '16.17'
        or admission.get('schema') != 1 or admission.get('recovery_version') != version
        or admission.get('timing_guard_ms') != 1000 or admission.get('final_test_payload_opened') is not False
        or admission.get('purpose') != 'patch_development'
        or corpus.get('decision_admission_sha256') != admission_sha256
        or qualification.get('decision_admission_sha256') != admission_sha256
        or audit.get('decision_admission_sha256') != admission_sha256
        or qualification.get('footwear_qualification') != expected):
        raise ValueError('Recovered decision mask lacks bound footwear qualification')
    games = admission.get('games') or {}
    source_bindings = admission.get('source_bindings') or {}
    import re
    if (set(source_bindings) != {'source_plan', 'prior_qualification', 'recovery_summary', 'recovery_source_report'}
        or any(not isinstance(v.get('path'), str) or not v['path']
               or not re.fullmatch('[0-9a-f]{64}', str(v.get('sha256', ''))) for v in source_bindings.values())):
        raise ValueError('Recovered mask lacks frozen input hashes')
    if (len(games) != admission.get('games_count') or len(games) != corpus.get('games')
        or not set(qualification['accepted_match_ids']) <= games.keys()):
        raise ValueError('Recovered mask does not account for the corpus')
    totals = {'admitted_rows':0, 'boundary_excluded_rows':0}
    for mid, game in games.items():
        def keys(values):
            result = []
            for value in values:
                key = tuple(value.get(k) for k in ('event_index', 'participant_id', 'ts'))
                if any(type(v) is not int for v in key) or key[0] < 0 or not 1 <= key[1] <= 10 or key[2] < 0:
                    raise ValueError(f'{mid}: malformed decision identity')
                result.append(key)
            if len(result) != len(set(result)):
                raise ValueError(f'{mid}: duplicate admission identity')
            return set(result)
        admitted, excluded = keys(game.get('admitted_decision_ids', [])), keys(game.get('boundary_exclusions', []))
        if (not admitted or admitted & excluded or game.get('split') not in ('train', 'val')
            or game.get('source_cohort') not in ('prior_qualified', 'recovered_footwear')
            or len(admitted) != game.get('admitted_rows') or len(excluded) != game.get('boundary_excluded_rows')
            or len(admitted) + len(excluded) != game.get('modeled_rows_before_boundary_guard')
            or any(v.get('reason') not in ('all_board_anchor_near_inferred_grant', 'own_visit_near_or_spans_inferred_grant')
                   for v in game.get('boundary_exclusions', []))
            or game['source_cohort'] == 'prior_qualified' and excluded):
            raise ValueError(f'{mid}: invalid audited decision mask')
        totals['admitted_rows'] += len(admitted)
        totals['boundary_excluded_rows'] += len(excluded)
    if any(admission.get(key) != value for key,value in totals.items()):
        raise ValueError('Recovered decision totals disagree')


def validate_run_directory(dataset, run):
    from app.config import DATA_DIR
    dataset, run = Path(dataset).resolve(),Path(run).resolve()
    experiments = (DATA_DIR / 'experiments').resolve()
    if not run.is_relative_to(experiments) or run == experiments or run.is_relative_to(dataset) or dataset.is_relative_to(run):
        raise ValueError('Experiment outputs must be isolated under data/experiments, outside dataset and serving files')


def generation_fingerprint(data_files, qualification_sha256, row_builder_sha256, split_plan_sha256):
    return canonical_sha({'data_files':data_files,'qualification_sha256':qualification_sha256,
        'row_builder_sha256':row_builder_sha256,'split_plan_sha256':split_plan_sha256})


def validate_development_split_metadata(metadata):
    import re
    if (metadata.get('test_status') != 'unavailable_original_holdout_not_materialized'
        or type(metadata.get('reserved_test_games')) is not int or metadata['reserved_test_games'] <= 0
        or not re.fullmatch('[0-9a-f]{64}',str(metadata.get('reserved_test_ids_sha256','')))
        or any(not isinstance(metadata.get(k),str) or not metadata[k].strip()
               for k in ('validation_exposure','assignment_policy'))):
        raise ValueError('Development requires explicit original-test reservation and validation exposure metadata')


SPLIT_1618_POLICIES = {
    'preserve-original-temporal-windows-v1': 'reused development validation; not confirmatory',
    'forward-validation-v1': 'fresh forward development validation; unshopped at admission',
}


def validate_1618_split(assignment, metadata, source_plan):
    """Validate source metadata only, before any examples can enter this benchmark."""
    policy = metadata.get('policy')
    if (metadata.get('schema') != 1 or metadata.get('purpose') not in SCOPED_DEVELOPMENT
        or policy not in SPLIT_1618_POLICIES
        or metadata.get('final_test_payload_opened') is not False
        or metadata.get('validation_exposure') != SPLIT_1618_POLICIES[policy]
        or set(assignment.values()) != {'train', 'val'}):
        raise ValueError('16.18 requires a declared temporal development split and sealed test')
    rows = {row['match_id']:row for row in source_plan['matches']}
    if len(rows) != len(source_plan['matches']) or not assignment.keys() <= rows.keys():
        raise ValueError('16.18 split contains duplicate or unselected sources')
    reserved = set(source_plan['reserved_test_ids'])
    if sorted(reserved) != metadata.get('reserved_test_ids') or reserved & assignment.keys():
        raise ValueError('A reserved test game entered the benchmark')
    if policy == 'forward-validation-v1':
        boundary = metadata.get('forward_boundary_ms')
        if type(boundary) is not int or boundary <= 0:
            raise ValueError('Forward validation requires the frozen snapshot boundary')
        for mid, fold in assignment.items():
            row = rows[mid]
            prior = set(row.get('prior_assignments', {}).values())
            if 'test' in prior:
                raise ValueError('Prior held-out membership entered the benchmark')
            if fold == 'train':
                if row['game_creation'] > boundary or 'val' in prior:
                    raise ValueError('Training must predate the boundary and never reuse validation')
            else:
                if row['game_creation'] <= boundary or prior:
                    raise ValueError('Forward validation must be strictly newer and previously unassigned')
        if (max(rows[m]['game_creation'] for m, f in assignment.items() if f == 'train')
                >= min(rows[m]['game_creation'] for m, f in assignment.items() if f == 'val')):
            raise ValueError('Validation must follow every training game; ties stay together')
        return
    train_end, val_end = metadata.get('train_end_ms'), metadata.get('val_end_ms')
    if type(train_end) is not int or type(val_end) is not int or train_end >= val_end:
        raise ValueError('Invalid original temporal cutoffs')
    original = metadata.get('original_assignment') or {}
    stamps = metadata.get('original_game_creation_ms') or {}
    if not original or original.keys() != stamps.keys() or set(original.values()) != {'train','val','test'}:
        raise ValueError('Original temporal benchmark metadata is incomplete')
    for fold, end in (('train',train_end),('val',val_end)):
        if max(stamps[mid] for mid,f in original.items() if f == fold) != end:
            raise ValueError('Original temporal cutoff changed')
    if max(stamps[m] for m,f in original.items() if f=='train') >= min(stamps[m] for m,f in original.items() if f=='val'):
        raise ValueError('Original training and validation are not temporal')
    if val_end >= min(stamps[m] for m,f in original.items() if f=='test'):
        raise ValueError('Original validation and test are not temporal')
    for mid, fold in assignment.items():
        row = rows[mid]
        prior = set(row.get('prior_assignments',{}).values())
        if 'test' in prior or (fold == 'train' and 'val' in prior):
            raise ValueError('Prior held-out membership entered training')
        if mid in original and original[mid] != fold:
            raise ValueError('Prior frozen membership changed')
        expected = 'train' if row['game_creation'] <= train_end else 'val' if row['game_creation'] <= val_end else None
        if fold != expected:
            raise ValueError('A newer source crossed an original temporal boundary')
    if max(rows[m]['game_creation'] for m,f in assignment.items() if f=='train') >= min(rows[m]['game_creation'] for m,f in assignment.items() if f=='val'):
        raise ValueError('Validation must follow every training game; ties stay together')


def validate_1618_admission(directory, qualification=None):
    """Verify the new authorization and frozen reconstruction evidence; no test payloads."""
    directory = Path(directory).resolve()
    q = qualification or json.loads((directory/'qualification.json').read_text(encoding='utf-8'))
    admission_path = directory/'benchmark_admission.json'
    admission = json.loads(admission_path.read_text(encoding='utf-8'))
    scope = SCOPED_DEVELOPMENT.get(q.get('benchmark_purpose'))
    if scope is None:
        raise ValueError('16.18 lacks a separately authorized and bound admission record')
    purpose = q['benchmark_purpose']
    if (q.get('benchmark_purpose') != purpose or q.get('status') != 'qualified_subset'
        or q.get('purpose') != 'isolated_experiment_only' or q.get('training_clearance') != 'authorized benchmark IDs only'
        or q.get('promotion_allowed') is not False or q.get('production_hold_cleared') is not False
        or q.get('benchmark_admission_sha256') != sha256(admission_path)
        or admission.get('purpose') != purpose or admission.get('patch') != scope['patch']
        or admission.get('authorization') not in scope['authorizations']
        or admission.get('source_scope_unchanged') is not True
        or admission.get('final_test_payload_opened') is not False):
        raise ValueError('16.18 lacks a separately authorized and bound admission record')
    required = {'source_plan.json','original_qualification.json','original_completion.json',
                'original_protocol.json','original_static_review.json','corpus_manifest.json',
                'semantic_audit.json','decision_admission.json','split_metadata.json',
                'preserved_split_assignment.json','qualification_code.py','audit_code.py',
                'original_split_plan.json','original_split_assignment.json','original_generation_manifest.json'}
    bindings = admission.get('files') or {}
    if set(bindings) != required:
        raise ValueError('16.18 admission evidence set changed')
    for name,binding in bindings.items():
        path = directory/name
        if not path.is_file() or path.stat().st_size != binding['bytes'] or sha256(path) != binding['sha256']:
            raise ValueError(f'16.18 admission evidence changed: {name}')
    documents = {name:json.loads((directory/name).read_text(encoding='utf-8')) for name in required if name.endswith('.json')}
    source, original, completion, protocol, corpus, audit, mask, metadata, assignment = (documents[n] for n in (
        'source_plan.json','original_qualification.json','original_completion.json','original_protocol.json',
        'corpus_manifest.json','semantic_audit.json','decision_admission.json','split_metadata.json','preserved_split_assignment.json'))
    if (source.get('training_or_export_authorized') is not False or source.get('test_payload_reads') is not False
        or original.get('status') != 'qualified_reconstruction_snapshot' or original.get('training_clearance') is not False
        or original.get('export_created') is not False or original.get('test_payload_reads') is not False
        or completion.get('state') != 'complete' or completion.get('completed_stages') != ['build','audit']
        or completion.get('qualification_sha256') != bindings['original_qualification.json']['sha256']
        or completion.get('protected_unchanged') is not True or completion.get('test_payload_reads') is not False
        or protocol.get('source_plan_sha256') != bindings['source_plan.json']['sha256']
        or corpus.get('source_plan_sha256') != bindings['source_plan.json']['sha256']
        or original.get('corpus_manifest_sha256') != bindings['corpus_manifest.json']['sha256']
        or original.get('semantic_audit_sha256') != bindings['semantic_audit.json']['sha256']
        or audit.get('corpus_manifest_sha256') != bindings['corpus_manifest.json']['sha256']
        or audit.get('db_sha256') != corpus.get('db_sha256') or original.get('db_sha256') != corpus.get('db_sha256')
        or audit.get('passed') is not True or audit.get('hard_failures') or audit.get('quarantined_match_ids')
        or corpus.get('patch') != scope['patch'] or corpus.get('recovery_version') != scope['recovery_version']
        or mask.get('recovery_version') != corpus['recovery_version'] or mask.get('timing_guard_ms') != 1000
        or mask.get('test_payload_reads') is not False
        or corpus.get('decision_admission_sha256') != bindings['decision_admission.json']['sha256']
        or original.get('decision_admission_sha256') != bindings['decision_admission.json']['sha256']):
        raise ValueError('16.18 original reconstruction evidence chain is invalid')
    all_accepted = set(original['accepted_match_ids'])
    if (all_accepted != set(audit['accepted_match_ids']) or all_accepted != set(mask['games'])
        or len(all_accepted) != original['accepted_games'] or len(all_accepted) != corpus['games']
        or all_accepted & set(source['reserved_test_ids'])
        or audit.get('audit_code_sha256') != bindings['audit_code.py']['sha256']
        or audit['counts'].get('checks:reconstructible_whole_game_source_checks') != len(all_accepted)
        or audit['counts'].get('checks:reconstructible_exact_admission_mask') != len(all_accepted)
        or not audit['counts'].get('checks:reconstructible_causal_footwear_state')):
        raise ValueError('16.18 full Viego/footwear checks do not cover the corpus')
    validate_1618_split(assignment,metadata,source)
    if metadata['policy'] == 'preserve-original-temporal-windows-v1':
        original_plan = documents['original_split_plan.json']
        original_generation = documents['original_generation_manifest.json']
        if (metadata['original_assignment'] != documents['original_split_assignment.json']
            or original_plan['assignment'] != metadata['original_assignment']
            or original_plan['game_creation_ms'] != metadata['original_game_creation_ms']
            or original_generation['files']['split_plan.json']['sha256'] != bindings['original_split_plan.json']['sha256']
            or original_generation['files']['split_assignment.json']['sha256'] != bindings['original_split_assignment.json']['sha256']
            or bindings['original_split_assignment.json']['sha256'] not in source['split_metadata_sha256'].values()):
            raise ValueError('Original 16.18 split metadata is not source-bound')
    else:
        # forward-validation-v1: the copied "original_*" documents describe the
        # PRIOR benchmark generation whose validation window is now retired.
        prior_generation = documents['original_generation_manifest.json']
        prior_assignment = documents['original_split_assignment.json']
        if (metadata.get('prior_generation_manifest_sha256') != bindings['original_generation_manifest.json']['sha256']
            or prior_generation['files']['split_plan.json']['sha256'] != bindings['original_split_plan.json']['sha256']
            or prior_generation['files']['split_assignment.json']['sha256'] != bindings['original_split_assignment.json']['sha256']
            or bindings['original_split_assignment.json']['sha256'] not in source['split_metadata_sha256'].values()):
            raise ValueError('Prior 16.18 benchmark metadata is not source-bound')
        if any(mid in assignment for mid, fold in prior_assignment.items() if fold == 'val'):
            raise ValueError('A development-exposed prior validation game entered the forward benchmark')
    review = documents['original_static_review.json']
    if (source.get('static_review_sha256') != bindings['original_static_review.json']['sha256']
        or corpus.get('files',{}).get('static_review/static_review.json',{}).get('sha256') != bindings['original_static_review.json']['sha256']
        or review.get('patch') != scope['patch'] or review.get('ddragon_version') != scope['ddragon']
        or review.get('timer_economics_and_physical_rules_compatible') is not True):
        raise ValueError('16.18 admission lost its pinned mechanics review')
    selected = set(q['accepted_match_ids'])
    if (selected != set(assignment) or not selected <= all_accepted
        or len(selected) != q.get('quality_accepted_games') or selected & set(q['quarantined_match_ids'])
        or set(q.get('inventory_constraint_states',[])) != {'board_before','shop_before','shop_after'}
        or q.get('accepted_max_regular_slots',7) > 6
        or q.get('source_qualification_sha256') != bindings['original_qualification.json']['sha256']
        or q.get('semantic_audit_sha256') != bindings['semantic_audit.json']['sha256']
        or q.get('corpus_manifest_sha256') != bindings['corpus_manifest.json']['sha256']
        or q.get('qualification_code_sha256') != bindings['qualification_code.py']['sha256']
        or q.get('decision_admission_sha256') != bindings['decision_admission.json']['sha256']
        or q.get('db_sha256') != corpus['db_sha256']):
        raise ValueError('16.18 selected qualification or inventory checks changed')
    for mid in selected:
        game = mask['games'][mid]
        admitted = {(v['event_index'],v['participant_id'],v['ts']) for v in game['admitted_decision_ids']}
        excluded = {(v['event_index'],v['participant_id'],v['ts']) for v in game['boundary_exclusions']}
        if (len(admitted) != game['admitted_rows'] or len(excluded) != game['boundary_excluded_rows']
            or admitted & excluded or not admitted):
            raise ValueError('16.18 decision mask has duplicate, missing or overlapping rows')
    return {'admission':admission,'source_plan':source,'corpus':corpus,'audit':audit,'mask':mask,
            'metadata':metadata,'assignment':assignment,'qualification':q}


def validate_development_run_plan(dataset, run, generation, plan_path):
    """Bind development to one reviewed recipe; this never clears production hold."""
    dataset, run, plan_path = Path(dataset).resolve(), Path(run).resolve(), Path(plan_path).resolve()
    validate_run_directory(dataset, run)
    if plan_path != run / 'run_plan.json':
        raise ValueError('Development requires run_plan.json directly in the isolated run directory')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    purpose = generation.get('purpose')
    expected = {'schema':1, 'purpose':purpose, 'patch':SCOPED_DEVELOPMENT[purpose]['patch'] if purpose in SCOPED_DEVELOPMENT else '16.17',
        'export_fingerprint':generation['export_fingerprint'],
        'generation_manifest_sha256':sha256(dataset / 'generation_manifest.json'),
        'final_test_evaluation':False, 'promotion_allowed':False, 'production_hold_cleared':False}
    if (purpose not in DEVELOPMENT_PURPOSES
        or any(plan.get(k) != v or type(plan.get(k)) is not type(v) for k,v in expected.items())
        or Path(plan.get('dataset','')).resolve() != dataset
        or Path(plan.get('run_directory','')).resolve() != run):
        raise ValueError('Development run plan does not bind this scoped dataset and run')
    if any(not isinstance(plan.get(k),str) or not plan[k].strip() for k in ('question','stopping_rule')):
        raise ValueError('Development requires a declared question and stopping rule')
    if (plan.get('primary_metrics') != ['first_action_exact','action_family_match']
        or not isinstance(plan.get('populations'),list) or not plan['populations']
        or any(not isinstance(v,str) or not v.strip() for v in plan['populations'])
        or plan.get('evaluation') != {'split':'val','games':0,'compare_conditional':True}):
        raise ValueError('Development requires fixed paired, full-validation metrics and populations')
    if (type(plan.get('evaluation_seed')) is not int
        or type(plan.get('evaluation_bootstrap')) is not int or plan['evaluation_bootstrap'] <= 0):
        raise ValueError('Development requires a fixed integer evaluation seed and positive bootstrap count')
    budget = plan.get('resource_budget') or {}
    for key in ('max_wall_seconds','max_ram_gb','minimum_free_disk_gb'):
        value = budget.get(key)
        if type(value) not in (int,float) or not 0 < value < float('inf'):
            raise ValueError(f'Development requires a positive finite resource budget: {key}')
    if type(budget.get('max_train_processes')) is not int or budget['max_train_processes'] != 1:
        raise ValueError('Development permits one training process at a time')
    code = plan.get('code_sha256') or {}
    if 'scripts/train_prefix.py' not in code:
        raise ValueError('Development requires a bound trainer code digest')
    from app.config import DATA_DIR
    root = DATA_DIR.parent.resolve()
    for name,digest in code.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Development code differs from the fixed run plan: {name}')
    return plan, sha256(plan_path)


def verify_generation(directory: Path, *, verify_splits=('train','val'), staged=False):
    directory = directory.resolve()
    if not staged and (directory.name.startswith('.') or directory.name.endswith('.partial')):
        raise ValueError('An unfinished generation is not a training input')
    manifest = json.loads((directory / 'generation_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('state') != 'complete' or manifest.get('purpose') not in PURPOSES or manifest.get('production_hold_cleared') is not False or manifest.get('promotion_allowed') is not False:
        raise ValueError('Generation is not a complete, scoped dataset')
    files = manifest.get('files') or {}
    if not REQUIRED <= files.keys():
        raise ValueError('Generation is missing required artifacts')
    for name, binding in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != binding['bytes']:
            raise ValueError(f'Invalid or missing generation artifact: {name}')
        # Training/evaluation never opens final-test payloads, even for hashes.
        # Its digest was accumulated while the publisher wrote the stream.
        if name.startswith('visits_') and name.endswith('.jsonl') and name[7:-6] not in verify_splits:
            continue
        if sha256(path) != binding['sha256']:
            raise ValueError(f'Generation artifact changed: {name}')
    q = json.loads((directory / 'qualification.json').read_text(encoding='utf-8'))
    audit = json.loads((directory / 'semantic_audit.json').read_text(encoding='utf-8'))
    coverage = json.loads((directory / 'coverage.json').read_text(encoding='utf-8'))
    split = json.loads((directory / 'split_manifest.json').read_text(encoding='utf-8'))
    assignment = json.loads((directory / 'split_assignment.json').read_text(encoding='utf-8'))
    plan = json.loads((directory / 'split_plan.json').read_text(encoding='utf-8'))
    corpus = json.loads((directory / 'corpus_manifest.json').read_text(encoding='utf-8'))
    scope = SCOPED_DEVELOPMENT.get(manifest['purpose'])
    scoped_1618 = scope is not None
    if scoped_1618:
        if not BENCHMARK_1618_FILES <= files.keys():
            raise ValueError('16.18 generation lacks its frozen admission evidence')
        evidence_1618 = validate_1618_admission(directory,q)
        if (manifest.get('patch') != scope['patch'] or any(evidence_1618['assignment'].get(mid) != fold for mid,fold in plan['assignment'].items())
            or evidence_1618['metadata'] != plan.get('split_metadata')
            or manifest.get('footwear_runtime') != corpus.get('footwear_runtime')
            or split.get('footwear_runtime') != corpus.get('footwear_runtime')
            or manifest.get('source_target_version') != 'full-source-completions-v1'
            or split.get('source_target_version') != 'full-source-completions-v1'
            or split.get('input_schema',{}).get('source_target_version') != 'full-source-completions-v1'
            or split.get('footwear_state_versions') != {'footwear-timer-slots-v1':sum(split['rows_per_split'].values())}
            or plan.get('decision_admission_sha256') != files['decision_admission.json']['sha256']
            or coverage.get('decision_admission_sha256') != files['decision_admission.json']['sha256']):
            raise ValueError('16.18 generation changed its scoped split, runtime or target semantics')
    elif corpus.get('recovery_version') is not None:
        required_recovery = {'decision_admission.json', 'recovery_inputs.json', 'recovery_summary.json', 'recovered_audit_code.py'}
        if not required_recovery <= files.keys():
            raise ValueError('Recovered generation lacks frozen admission evidence')
        admission = json.loads((directory / 'decision_admission.json').read_text(encoding='utf-8'))
        validate_recovered_evidence(q, audit, corpus, admission, files['decision_admission.json']['sha256'])
        if (audit.get('audit_scope') != 'complete_recovered_corpus'
            or audit.get('recovered_audit_code_sha256') != files['recovered_audit_code.py']['sha256']
            or q.get('recovered_audit_code_sha256') != files['recovered_audit_code.py']['sha256']):
            raise ValueError('Recovered generation requires the full aggregate validator snapshot')
        inputs = json.loads((directory / 'recovery_inputs.json').read_text(encoding='utf-8'))
        summary = json.loads((directory / 'recovery_summary.json').read_text(encoding='utf-8'))
        if (inputs != admission['source_bindings']
            or files['recovery_summary.json']['sha256'] != inputs['recovery_summary']['sha256']
            or summary['source_report_sha256'] != inputs['recovery_source_report']['sha256']
            or summary['previous_qualification_sha256'] != inputs['prior_qualification']['sha256']):
            raise ValueError('Recovered input evidence chain changed')
        expected_runtime = corpus.get('footwear_runtime')
        if (not expected_runtime or manifest.get('footwear_runtime') != expected_runtime
            or split.get('footwear_runtime') != expected_runtime
            or coverage.get('decision_admission_sha256') != files['decision_admission.json']['sha256']
            or plan.get('decision_admission_sha256') != files['decision_admission.json']['sha256']
            or split.get('footwear_state_versions') != {'footwear-timer-slots-v1':sum(split['rows_per_split'].values())}
            or split.get('source_target_version') != 'full-source-completions-v1'
            or manifest.get('source_target_version') != 'full-source-completions-v1'
            or split.get('input_schema', {}).get('source_target_version') != 'full-source-completions-v1'
            or any(admission['games'][mid]['split'] != fold for mid,fold in assignment.items())):
            raise ValueError('Recovered runtime, mask or frozen folds disagree')
    if q.get('status') != 'qualified_subset' or q.get('promotion_allowed') is not False or not audit.get('passed') or audit.get('hard_failures'):
        raise ValueError('Generation lacks qualifying semantic evidence')
    if q['semantic_audit_sha256'] != files['semantic_audit.json']['sha256'] or q['corpus_manifest_sha256'] != files['corpus_manifest.json']['sha256']:
        raise ValueError('Qualification evidence chain is broken')
    if (q['qualification_code_sha256'] != files['qualification_code.py']['sha256']
        or audit['audit_code_sha256'] != files['audit_code.py']['sha256']
        or audit['corpus_manifest_sha256'] != files['corpus_manifest.json']['sha256']
        or q['db_sha256'] != corpus['db_sha256'] or audit['db_sha256'] != corpus['db_sha256']):
        raise ValueError('Audit/code/database evidence chain is broken')
    if coverage['qualification_sha256'] != files['qualification.json']['sha256'] or coverage['split_plan_sha256'] != files['split_plan.json']['sha256']:
        raise ValueError('Coverage evidence chain is broken')
    if (plan['assignment'] != assignment or plan['qualification_sha256'] != files['qualification.json']['sha256']
        or plan['row_builder_sha256'] != coverage['row_builder_sha256']
        or plan['row_counts'] != split['rows_per_split'] or plan['game_counts'] != split['counts']
        or q.get('production_hold_cleared') is not False or q.get('purpose') != 'isolated_experiment_only'
        or coverage.get('target_encoding') != 'item-counts-v1'
        or set(q.get('inventory_constraint_states',[])) != {'board_before','shop_before','shop_after'}):
        raise ValueError('Frozen split, multiset or qualification scope mismatch')
    expected_folds = {'train','val'} if manifest['purpose'] in DEVELOPMENT_PURPOSES else set(SPLITS)
    if set(assignment.values()) != expected_folds or not assignment.keys() <= set(q['accepted_match_ids']):
        raise ValueError('Split contains unqualified games or incomplete folds')
    data_files = {name:files[name] for name in ('split_assignment.json',*(f'visits_{s}.jsonl' for s in SPLITS))}
    fingerprint = generation_fingerprint(data_files,files['qualification.json']['sha256'],coverage['row_builder_sha256'],files['split_plan.json']['sha256'])
    if manifest['export_fingerprint'] != fingerprint or split['export_fingerprint'] != fingerprint or split.get('probe') is not True:
        raise ValueError('Dataset fingerprint or experiment scope changed')
    if plan.get('purpose') != manifest['purpose'] or split.get('purpose') != manifest['purpose']:
        raise ValueError('Dataset experiment purposes disagree')
    if manifest['purpose'] in DEVELOPMENT_PURPOSES:
        from app.reconstruct import SLOT_STATE_VERSION, RECONSTRUCTION_VERSION
        total = sum(split['rows_per_split'].values())
        expected_patch = scope['patch'] if scoped_1618 else '16.17'
        # Membership sets, not single literals: bumping an estimator or
        # reconstruction version must never make PAST generations unverifiable
        # (their evidence has to stay re-checkable), while mixes and legacy
        # tags stay refused. Exactly one version must cover every row.
        allowed_reconstruction = (
            scope['allowed_reconstruction']
            if scoped_1618 else (
                ('decision-start-footwear-v3', 'decision-start-footwear-v4')
                if corpus.get('recovery_version') else (RECONSTRUCTION_VERSION,)))
        allowed_gold = ('prequential-v2', 'prequential-v3')
        if (any(record.get('patch') != expected_patch for record in (manifest,split,plan,coverage,corpus))
            or coverage.get('purpose') != manifest['purpose']
            or not isinstance(plan.get('benchmark_id'),str) or not plan['benchmark_id'].strip()
            or type(plan.get('eval_version')) is not int or plan['eval_version'] < 1
            or split.get('benchmark_id') != plan['benchmark_id'] or split.get('eval_version') != plan['eval_version']
            or corpus.get('slot_state_version') != SLOT_STATE_VERSION
            or split.get('slot_state_versions') != {SLOT_STATE_VERSION:total}
            or split.get('reconstruction_versions') not in [{v:total} for v in allowed_reconstruction]
            or split.get('gold_est_versions') not in [{v:total} for v in allowed_gold]):
            raise ValueError('Development requires its scoped patch and current causal slot-state rows')
        name = 'preserved_split_assignment.json'
        if name not in files or plan.get('preserved_split_assignment_sha256') != files[name]['sha256']:
            raise ValueError('Development must bind prior split membership')
        prior = json.loads((directory / name).read_text(encoding='utf-8'))
        if any(prior.get(mid) != fold for mid,fold in assignment.items()):
            raise ValueError('Development reassigned a previously frozen game')
        metadata_name = 'split_metadata.json'
        if (metadata_name not in files or plan.get('split_metadata_sha256') != files[metadata_name]['sha256']
            or split['counts'].get('test') != 0 or split['rows_per_split'].get('test') != 0
            or files['visits_test.jsonl']['bytes'] != 0):
            raise ValueError('Development must preserve the unavailable original test without materializing examples')
        metadata = json.loads((directory / metadata_name).read_text(encoding='utf-8'))
        if scoped_1618:
            validate_1618_split(assignment,metadata,evidence_1618['source_plan'])
        else:
            validate_development_split_metadata(metadata)
        if plan.get('split_metadata') != metadata or split.get('split_metadata') != metadata:
            raise ValueError('Development holdout metadata changed')
    return manifest
