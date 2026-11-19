"""Validate completed source-owned FIT shards and stream complete IQL episodes.

Only versioned artifacts and identity metadata are consumed. No source Python
imports, raw archives, SQLite, mixed supervised NPZ, or fold reassignment.
Successful iterator exhaustion is mandatory before downstream publication.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from . import admission, full_action_catalogue, trajectory_contract

SOURCE_SCHEMA = 'winprob-fit-trajectories-v2'
BUNDLE_SCHEMA = 'winprob-iql-source-bundle-v1'
EXPECTED_PATCH_COUNTS = {'16.17': 3519, '16.18': 483}
SOURCE_FILES = {'trajectories.jsonl.gz', 'exclusions.jsonl', 'selection.json'}
FALSE_FLAGS = ('calibration_game_payload_reads', 'development_game_payload_reads', 'test_payload_reads',
               'confirmation_payload_reads', 'mixed_npz_or_training_export_read', 'policy_loaded',
               'training_performed', 'reference_compatibility_exported', 'whole_database_scanned')
TRUE_FLAGS = ('only_complete_catalogue_supported_player_chains', 'terminal_reward_reaches_last_action_once',
              'inferred_role_fields_absent_from_prestate')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bound_json(path, bindings):
    path = Path(path).resolve(strict=True)
    sidecar = path.with_suffix('.sha256')
    digest = file_hash(path)
    require(sidecar.read_text(encoding='ascii').split() == [digest, path.name], 'Artifact SHA256 sidecar mismatch: ' + str(path))
    bindings[str(path)] = digest
    bindings[str(sidecar)] = file_hash(sidecar)
    return json_file(path), digest


def reject_quarantined(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    require(not any((parent / 'QUARANTINED.md').exists() for parent in (directory, *directory.parents)), 'Quarantined source artifact')


def unique_ids(value, label):
    require(isinstance(value, list) and all(isinstance(mid, str) and admission.MATCH_ID.fullmatch(mid) for mid in value)
            and len(set(value)) == len(value), 'Invalid/duplicate identity list: ' + label)
    return set(value)


def metadata_binding(path, expected, bindings, code=False):
    path = Path(path).resolve(strict=True)
    require(path.suffix.lower() in ({'.py'} if code else {'.json', '.py', '.sha256'}),
            'Binding is not code/identity metadata: ' + str(path))
    require(file_hash(path) == expected, 'Source code/metadata changed: ' + str(path))
    bindings[str(path)] = expected


def current_reservations(manifest):
    admission.verify_manifest_sources(manifest)
    root = Path(manifest['source_data_root'])
    forbidden = set()
    for binding in manifest['split_metadata_sources']:
        mapping, observed = admission.read_assignment(root / binding['path'])
        require(observed == binding['sha256'], 'Reservation metadata changed during validation')
        forbidden.update(mid for mid, role in mapping.items() if role in ('test', 'val'))
    for binding in manifest.get('external_reservation_sources', []):
        ids, observed = admission.read_external_reservations(Path(binding['path']))
        require(observed == binding, 'External reservation changed')
        forbidden.update(ids)
    forbidden.update(row['match_id'] for row in manifest['games'] if row['role'] == 'confirmatory_reserved')
    prior = manifest.get('prior_manifest')
    seen = set()
    while prior:
        path = str(Path(prior['path']).resolve(strict=True))
        require(path not in seen, 'Cyclic prior admission')
        seen.add(path)
        old, binding = admission.read_prior_manifest(Path(path))
        require(binding == prior, 'Prior confirmation reservation changed')
        forbidden.update(row['match_id'] for row in old['games'] if row['role'] == 'confirmatory_reserved')
        prior = old.get('prior_manifest')
    return forbidden


def original_fit_from_metadata(source_manifest, admitted, forbidden):
    paths = [Path(path) for path in source_manifest['input_metadata_sha256'] if Path(path).name == 'source_manifest.json']
    require(len(paths) == 1, 'Require one bound original W1 source membership file')
    original = json_file(paths[0])
    require(all(original.get(key) is False for key in ('test_payload_reads', 'confirmatory_payload_reads', 'historical_validation_payload_reads')),
            'Original W1 source membership is not clean')
    fits = [row for row in original['games'] if row['role'] == 'fit']
    ids = {row['match_id'] for row in fits}
    require(len(fits) == len(ids) == 4002, 'Original W1 FIT universe is not exactly 4002 games')
    require(Counter(row['patch'] for row in fits) == EXPECTED_PATCH_COUNTS, 'Original 3519/483 patch scope changed')
    require(not ids & forbidden, 'Original FIT universe intersects a reserved game')
    for row in fits:
        entry = admitted.get(row['match_id'])
        require(entry is not None and entry['role'] == 'train' and entry.get('historical_validation') is False
                and row['source_role'] == 'train' and row['patch'] == entry['patch']
                and row['game_creation_ms'] == entry['game_creation_ms'], 'Original FIT membership/role/date changed')
    return ids, {row['match_id']: row['patch'] for row in fits}


def exclusion_accounting(directory, manifest, requested, queried):
    """Recover disjoint per-game denominators from exclusion identity metadata."""
    excluded_players = {mid: {} for mid in queried}
    game_exclusions = {}
    reasons = Counter()
    with (directory / 'exclusions.jsonl').open('r', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            mid, why = row.get('match_id'), row.get('reasons')
            require(mid in requested and row.get('role') == 'fit' and row.get('patch') == manifest['patch'],
                    'Exclusion identity/role differs from source selection')
            require(isinstance(why, list) and why and all(isinstance(reason, str) and reason for reason in why)
                    and len(set(why)) == len(why), 'Invalid exclusion reasons')
            reasons.update(why)
            if row.get('scope') == 'game':
                require(mid not in queried and mid not in game_exclusions, 'Duplicate or queried game qualification exclusion')
                game_exclusions[mid] = why
            elif row.get('scope') == 'player_game':
                pid = row.get('participant_id')
                require(mid in queried and type(pid) is int and 1 <= pid <= 10 and pid not in excluded_players[mid]
                        and row.get('trajectory_id') == f'{mid}:{pid}', 'Invalid/duplicate player-game exclusion')
                require(not ('complete_action_outside_frozen_catalogue' in why and why != ['complete_action_outside_frozen_catalogue'])
                        and not ('source_chain_validation_failed' in why and why != ['source_chain_validation_failed']),
                        'Exclusive source/catalogue exclusion branches were combined')
                excluded_players[mid][pid] = why
            else:
                raise ValueError('Unknown exclusion scope')
    require(set(game_exclusions) == requested - queried, 'Qualification exclusions do not partition requested IDs')
    qualification = manifest['qualification_exclusions']
    require(set(qualification) == set(game_exclusions)
            and all(game_exclusions[mid] == [reason] for mid, reason in qualification.items()), 'Qualification exclusions differ')
    expected_pids = {mid: set(range(1, 11)) - set(excluded_players[mid]) for mid in queried}
    exported = {mid: len(expected_pids.get(mid, set())) for mid in requested}
    complete, flag_complete = dict(exported), dict(exported)
    catalogue_excluded = validation_failed = 0
    for mid, by_pid in excluded_players.items():
        for why in by_pid.values():
            if why == ['complete_action_outside_frozen_catalogue']:
                complete[mid] += 1
                flag_complete[mid] += 1
                catalogue_excluded += 1
            elif why == ['source_chain_validation_failed']:
                flag_complete[mid] += 1
                validation_failed += 1
    counts = manifest['counts']
    eligible = sum(exported.values())
    n_excluded = sum(len(value) for value in excluded_players.values())
    require(counts.get('considered_player_games', 0) == 10 * len(queried)
            and counts.get('eligible_player_games', 0) == eligible == manifest['eligible_player_games'] == manifest['trajectory_count']
            and counts.get('excluded_player_games', 0) == n_excluded
            and counts.get('terminal_transitions', 0) == eligible
            and counts.get('qualification_excluded_games', 0) == len(game_exclusions), 'Source player/game accounting mismatch')
    require(dict(reasons) == manifest['exclusion_reason_counts'], 'Exclusion reason totals differ')
    eligible_ids = {mid for mid, count in exported.items() if count > 0}
    require(eligible_ids == unique_ids(manifest['eligible_game_ids'], 'eligible_game_ids')
            and len(eligible_ids) == manifest['eligible_games'], 'Eligible game count/list mismatch')
    return {'expected_player_ids': expected_pids, 'exported_player_games': exported,
            'source_complete_player_games': complete, 'source_flag_complete_player_games': flag_complete,
            'catalogue_excluded_player_games': catalogue_excluded, 'source_chain_validation_failed_player_games': validation_failed}


def validate_sources(directories, admission_path, catalogue_path, smoke=False):
    """Validate only completed artifacts; gzip episodes are streamed later."""
    require(type(smoke) is bool, 'Smoke flag must be boolean')
    paths = [Path(path).resolve(strict=True) for path in directories]
    require(paths and len(paths) == len(set(paths)), 'Missing or duplicate source directories')
    require(all((path / 'manifest.json').is_file() and (path / 'manifest.sha256').is_file() for path in paths),
            'A source shard is unfinished or lacks its publication sidecar')
    bindings, snapshots = {}, []
    for directory in paths:
        reject_quarantined(directory)
        manifest, manifest_hash = bound_json(directory / 'manifest.json', bindings)
        require(manifest.get('schema') == SOURCE_SCHEMA and manifest.get('state') == 'complete', 'Source shard is not complete FIT v2')
        snapshots.append((directory, manifest, manifest_hash))
    admission_path = Path(admission_path).resolve(strict=True)
    reject_quarantined(admission_path)
    current, admission_hash = bound_json(admission_path, bindings)
    require(current.get('schema') == 'winprob-admission-v1' and current.get('state') == 'temporal_manifest_ready', 'Wrong frozen admission')
    forbidden = current_reservations(current)
    admitted = {row['match_id']: row for row in current['games']}
    require(len(admitted) == len(current['games']), 'Duplicate admission game')
    catalogue_path = Path(catalogue_path).resolve(strict=True)
    reject_quarantined(catalogue_path)
    catalogue, lookup = full_action_catalogue.load_catalogue(catalogue_path)
    catalogue_hash = file_hash(catalogue_path)
    bindings[str(catalogue_path)] = catalogue_hash
    bindings[str(catalogue_path.with_suffix('.sha256'))] = file_hash(catalogue_path.with_suffix('.sha256'))
    sources = []
    all_fit = fit_patches = None
    requested_union, seen_patches, eligible_union = set(), set(), set()
    complete, exported, flag_complete = {}, {}, {}
    for directory, m, manifest_hash in snapshots:
        require(all(m.get(key) is False for key in FALSE_FLAGS) and all(m.get(key) is True for key in TRUE_FLAGS),
                'Source FIT payload or completeness scope changed')
        patch = m.get('patch')
        require(patch in EXPECTED_PATCH_COUNTS and patch not in seen_patches and m.get('role') == 'fit', 'Duplicate/non-FIT source patch')
        seen_patches.add(patch)
        require(m.get('rows_schema') == trajectory_contract.SCHEMA and m.get('rows_file') == 'trajectories.jsonl.gz'
                and m.get('rows_compression') == 'gzip' and m.get('gzip_compresslevel') == 1 and m.get('gzip_mtime') == 0,
                'Unknown source episode encoding')
        require(m.get('full_action_catalogue_sha256') == catalogue_hash == m.get('catalogue_sha256'), 'Source uses another catalogue')
        require(set(m['files']) == SOURCE_FILES and m['files']['trajectories.jsonl.gz'] == m['rows_sha256'], 'Incomplete source file binding')
        for path, expected in m['source_code_sha256'].items():
            metadata_binding(path, expected, bindings, code=True)
        for path, expected in m['input_metadata_sha256'].items():
            metadata_binding(path, expected, bindings)
        require(m['input_metadata_sha256'].get(str(admission_path)) == admission_hash, 'Source admission differs from current reservations')
        fit_ids, patches = original_fit_from_metadata(m, admitted, forbidden)
        require(unique_ids(m['original_fit_game_ids'], 'original_fit_game_ids') == fit_ids and m.get('original_fit_games') == 4002
                and m.get('original_fit_ids_sha256') == admission.id_digest(fit_ids)
                and m.get('original_fit_patch_counts') == EXPECTED_PATCH_COUNTS, 'Declared original FIT universe changed')
        if all_fit is None:
            all_fit, fit_patches = fit_ids, patches
        else:
            require(fit_ids == all_fit and patches == fit_patches, 'Shards have different original FIT populations')
        requested = unique_ids(m['requested_game_ids'], 'requested_game_ids')
        require(requested and requested <= fit_ids and not requested & requested_union
                and all(patches[mid] == patch for mid in requested), 'Requested games overlap or change patch')
        expected_patch_ids = {mid for mid in fit_ids if patches[mid] == patch}
        require(m['requested_patch_fit_games'] == len(expected_patch_ids), 'Original patch FIT count changed')
        if smoke:
            require(len(requested) <= 10, 'Smoke allows at most 10 requested FIT games per source')
        else:
            require(m.get('full_mode') is True and requested == expected_patch_ids, 'Full production shard omits/adds original FIT games')
        queried = unique_ids(m['queried_game_ids'], 'queried_game_ids')
        require(queried == unique_ids(m['qualified_game_ids'], 'qualified_game_ids') and queried <= requested, 'Query identities escaped selection')
        require(not requested & forbidden, 'Requested source identity is reserved')
        # Identity/role admission precedes even compressed episode-file hashing.
        for name, expected in m['files'].items():
            path = (directory / name).resolve(strict=True)
            require(path.is_relative_to(directory) and file_hash(path) == expected, 'Source artifact file changed: ' + name)
            bindings[str(path)] = expected
        selection = json_file(directory / 'selection.json')
        for key in ('schema', 'full_mode', 'patch', 'role', 'original_fit_games', 'original_fit_ids_sha256', 'original_fit_game_ids',
                    'original_fit_patch_counts', 'requested_game_ids', 'requested_patch_fit_games', 'qualified_game_ids',
                    'qualification_exclusions', 'input_metadata_sha256', 'source_code_sha256', 'catalogue_sha256'):
            require(selection.get(key) == m.get(key), 'Selection differs from manifest: ' + key)
        accounting = exclusion_accounting(directory, m, requested, queried)
        game_bindings = m['game_bindings']
        require(len(game_bindings) == len(queried) and {row['match_id'] for row in game_bindings} == queried, 'Per-game provenance coverage differs')
        require(all(row.get('raw_replay_matches_indexed_rows') is True and row.get('admitted_mask_reproduced') is True for row in game_bindings),
                'Per-game replay validation is incomplete')
        source = {'directory': str(directory), 'manifest': m, 'manifest_path': str(directory / 'manifest.json'),
                  'manifest_sha256': manifest_hash, 'rows_path': str(directory / 'trajectories.jsonl.gz'),
                  'requested_fit_ids': sorted(requested), 'queried_fit_ids': sorted(queried), **accounting}
        sources.append(source)
        requested_union.update(requested)
        eligible_union.update(m['eligible_game_ids'])
        complete.update(accounting['source_complete_player_games'])
        exported.update(accounting['exported_player_games'])
        flag_complete.update(accounting['source_flag_complete_player_games'])
    if not smoke:
        require(seen_patches == set(EXPECTED_PATCH_COUNTS) and requested_union == all_fit, 'Full corpus needs both disjoint 3519/483 shards')
    for module in (admission, full_action_catalogue, trajectory_contract):
        bindings[str(Path(module.__file__).resolve())] = file_hash(module.__file__)
    bindings[str(Path(__file__).resolve())] = file_hash(__file__)
    return {'schema': BUNDLE_SCHEMA, 'sources': sources, 'bindings': bindings, 'fit_ids': sorted(all_fit),
            'requested_fit_ids': sorted(requested_union), 'eligible_fit_ids': sorted(eligible_union),
            'catalogue': catalogue, 'lookup': lookup, 'catalogue_path': str(catalogue_path), 'catalogue_sha256': catalogue_hash,
            'admission_path': str(admission_path), 'admission_sha256': admission_hash, 'admitted': admitted,
            'source_complete_player_games': complete, 'exported_player_games': exported,
            'source_flag_complete_player_games': flag_complete, 'smoke': smoke,
            'training_scope_ready': not smoke and len(eligible_union) >= 1000, 'verified_complete': False, 'completed_passes': 0,
            'denominator_definition': 'Validated complete source chains before catalogue/feature filtering; eligible plus catalogue-only exclusions.'}


def _verify_current(bundle):
    reject_quarantined(bundle['admission_path'])
    reject_quarantined(bundle['catalogue_path'])
    for source in bundle['sources']:
        reject_quarantined(source['directory'])
    for path, expected in bundle['bindings'].items():
        require(file_hash(path) == expected, 'Bound source/code changed before/during iteration: ' + path)
    current = json_file(bundle['admission_path'])
    require(not set(bundle['requested_fit_ids']) & current_reservations(current), 'A requested FIT game became reserved')


def _compile_record(record, source, bundle, seen):
    m = source['manifest']
    mid, pid = record.get('match_id'), record.get('participant_id')
    require(mid in source['expected_player_ids'] and type(pid) is int and 1 <= pid <= 10
            and pid in source['expected_player_ids'][mid], 'Unexpected/source-excluded player-game was exported')
    identity = f'{mid}:{pid}'
    require(record.get('trajectory_id') == identity and identity not in seen, 'Duplicate/cross-source trajectory identity')
    require(record.get('role') == record.get('consumer_role') == 'fit' and record.get('patch') == m['patch']
            and record.get('source_role') == ('qualified_history_fit' if m['patch'] == '16.17' else 'v4_train'), 'Episode role/native patch differs')
    entry = bundle['admitted'][mid]
    require(entry['role'] == 'train' and entry.get('historical_validation') is False
            and record.get('game_creation_ms') == entry['game_creation_ms'], 'Episode is not original admitted FIT')
    require(record.get('full_action_catalogue_sha256') == bundle['catalogue_sha256'], 'Episode catalogue binding changed')
    require(type(record.get('team_id')) is int and record['team_id'] in (100, 200), 'Unknown focal team')
    quality = record.get('source_quality', {})
    index = quality.get('game_binding_index')
    require(type(index) is int and 0 <= index < len(m['game_bindings']), 'Invalid per-game provenance reference')
    binding = m['game_bindings'][index]
    require(binding['match_id'] == mid and quality.get('indexed_rows_sha256') == binding['indexed_rows_sha256'], 'Episode points to another game binding')
    require(all('reference_compatibility' not in node for node in record.get('decisions', [])), 'Reference-only projection entered training artifact')
    transitions = trajectory_contract.compile_transitions(record)
    for transition in transitions:
        action = transition['action']
        key = full_action_catalogue.counts_key(action['label_counts'])
        require(action.get('kind') == key[0] and action.get('wards_preserved') is True, 'Complete action semantic tag/ward contract changed')
        require(key in bundle['lookup'], 'Unsupported complete action was exported')
        expected = bundle['lookup'][key]
        group = bundle['catalogue']['full_to_group'][expected]
        require(type(action.get('full_action_id')) is int and action['full_action_id'] == expected
                and type(action.get('main_group_id')) is int and action['main_group_id'] == group, 'Action/group ID differs from canonical multiset')
        transition['full_action_id'] = expected
        transition['main_group_id'] = group
    seen.add(identity)
    return transitions


def iter_episodes(bundle):
    """Yield(source_index,record,transitions); only exhaustion verifies a pass."""
    require(bundle.get('schema') == BUNDLE_SCHEMA, 'Unknown source bundle')
    bundle['verified_complete'] = False
    bundle['iteration_attempts'] = bundle.get('iteration_attempts', 0) + 1
    attempt = bundle['iteration_attempts']
    def generate():
        _verify_current(bundle)
        seen, all_games, summaries = set(), set(), []
        total_episodes = total_transitions = 0
        for source_index, source in enumerate(bundle['sources']):
            m = source['manifest']
            uncompressed = hashlib.sha256()
            episodes = transition_count = 0
            players, observed_pids = Counter(), {}
            with gzip.open(source['rows_path'], 'rb') as handle:
                for line in handle:
                    uncompressed.update(line)
                    require(line.endswith(b'\n') and line.strip(), 'Blank or unterminated episode row')
                    record = json.loads(line.decode('utf-8'))
                    compiled = _compile_record(record, source, bundle, seen)
                    mid, pid = record['match_id'], record['participant_id']
                    players[mid] += 1
                    observed_pids.setdefault(mid, set()).add(pid)
                    all_games.add(mid)
                    episodes += 1
                    transition_count += len(compiled)
                    yield source_index, record, compiled
            require(uncompressed.hexdigest() == m['uncompressed_rows_sha256'], 'Uncompressed episode digest mismatch')
            require(episodes == m['trajectory_count'] == m['counts'].get('terminal_transitions', 0)
                    and transition_count == m['counts'].get('decision_nodes', 0), 'Episode/transition counts did not reproduce')
            require(dict(players) == {mid: n for mid, n in source['exported_player_games'].items() if n > 0}, 'Per-game output counts differ')
            require(observed_pids == {mid: pids for mid, pids in source['expected_player_ids'].items() if pids}, 'Participant membership differs from exclusions')
            summaries.append({'source_index': source_index, 'patch': m['patch'], 'episodes': episodes,
                              'transitions': transition_count, 'games': len(players), 'uncompressed_sha256': uncompressed.hexdigest()})
            total_episodes += episodes
            total_transitions += transition_count
        require(all_games == set(bundle['eligible_fit_ids']), 'Combined eligible game membership did not reproduce')
        _verify_current(bundle)
        require(bundle['iteration_attempts'] == attempt, 'Overlapping iteration cannot certify a complete pass')
        bundle['completed_passes'] += 1
        bundle['last_pass'] = {'sources': summaries, 'episodes': total_episodes, 'transitions': total_transitions, 'games': len(all_games)}
        bundle['verified_complete'] = True
    return generate()
