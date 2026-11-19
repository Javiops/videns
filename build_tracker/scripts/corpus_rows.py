"""Shared, read-only row construction for qualified-corpus coverage and export."""
from contextlib import closing, contextmanager
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
baseline = None  # Activated lazily after the scoped reconstruction runtime.
from app.dataset_artifacts import canonical_sha, sha256
from app.ddragon import dragon_for_patch
from app.reconstruct import GOLD_EST_VERSION, SLOT_STATE_VERSION
from app.purchase_multiset import label_counts, serialize_counts, TARGET_ENCODING


@dataclass(frozen=True)
class QualifiedCorpus:
    path: Path
    qualification_path: Path
    qualification: dict
    manifest: dict
    dragon: object
    decision_admission: dict | None = None


@contextmanager
def read_db(path):
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1',uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        yield conn


def row_builder_digest():
    global baseline
    if baseline is None:
        import importlib
        baseline = importlib.import_module('baseline')
    binding = {
        'functions':{f.__name__:inspect.getsource(f) for f in (baseline._example,baseline._label,baseline._item_ids,baseline._others,baseline._unmodeled_regular_slots)},
        'min_duration':baseline.MIN_DURATION_S,
        'corpus_rows':sha256(Path(__file__)),
        'static_classifier':sha256(ROOT / 'app/ddragon.py'),
        'shop_economics':sha256(ROOT / 'app/shop_econ.py'),
        'purchase_multiset':sha256(ROOT / 'app/purchase_multiset.py')}
    if getattr(baseline, 'FOOTWEAR_RUNTIME_VERSION', None):
        binding['footwear_runtime'] = {name:sha256(ROOT / name) for name in (
            'scripts/baseline.py', 'scripts/footwear_runtime.py', 'scripts/footwear_economics.py',
            'app/footwear_reconstruction.py', 'scripts/footwear_timer.py',
            'scripts/footwear_timer_inventory.py', 'scripts/magical_footwear.py')}
        if getattr(baseline,'FOOTWEAR_RUNTIME_VERSION',None) == 'reconstructible-16.18-v1':
            binding['footwear_runtime'].update({name:sha256(ROOT/name) for name in (
                'scripts/benchmark_1618_runtime.py','scripts/reconstructible_1618_runtime.py')})
    return canonical_sha(binding)


def load_qualified(path):
    global baseline
    q = json.loads(path.read_text(encoding='utf-8'))
    if q.get('status') != 'qualified_subset' or q.get('purpose') != 'isolated_experiment_only' or q.get('promotion_allowed') is not False:
        raise ValueError('Expected a qualified subset with an isolated-experiment scope')
    corpus = Path(q['corpus']).resolve()
    manifest_path = corpus / 'corpus_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('slot_state_version') != SLOT_STATE_VERSION:
        raise ValueError('Qualified exports require a freshly audited slot-state corpus')
    if sha256(manifest_path) != q['corpus_manifest_sha256'] or sha256(corpus / 'tracker.db') != q['db_sha256'] or manifest['db_sha256'] != q['db_sha256']:
        raise ValueError('Qualified corpus binding changed')
    audit_path = Path(q['semantic_audit'])
    if sha256(audit_path) != q['semantic_audit_sha256'] or sha256(path.parent / 'qualification_code.py') != q['qualification_code_sha256']:
        raise ValueError('Acceptance evidence changed')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or audit['hard_failures'] or audit['corpus_manifest_sha256'] != q['corpus_manifest_sha256']:
        raise ValueError('Audit does not certify these source checks')
    accepted = set(q['accepted_match_ids'])
    if len(accepted) != q['quality_accepted_games'] or not accepted <= set(audit['accepted_match_ids']) or accepted & q['quarantined_match_ids'].keys():
        raise ValueError('Invalid qualified game set')
    admission = None
    from app.dataset_generations import SCOPED_DEVELOPMENT          # patch_development_1618 / _1619, same contract
    scoped_1618 = q.get('benchmark_purpose') in SCOPED_DEVELOPMENT
    if scoped_1618:
        from app.dataset_generations import validate_1618_admission
        evidence = validate_1618_admission(path.parent,q)
        if evidence['corpus'] != manifest:
            raise ValueError('16.18 qualification no longer describes this corpus')
        # Source semantics are frozen snapshots, independent of this checkout's
        # newly implemented admission/export code. Never compare old hashes to
        # the modified app/dataset_generations.py in the new checkout.
        for name,digest in manifest['reconstruction_code_sha256'].items():
            if sha256(corpus/'reconstruction_code'/name) != digest:
                raise ValueError(f'Original reconstruction code snapshot changed: {name}')
        for name,binding in manifest['files'].items():
            if name.startswith(('static/','static_review/','reconstruction_variants/')):
                frozen = corpus/name
                if frozen.stat().st_size != binding['bytes'] or sha256(frozen) != binding['sha256']:
                    raise ValueError(f'Original static/runtime snapshot changed: {name}')
        if sha256(corpus/'decision_admission.json') != q['decision_admission_sha256']:
            raise ValueError('Original 16.18 decision mask changed')
        admission = evidence['mask']
        from footwear_runtime import activate_for_manifest
        activate_for_manifest(manifest,corpus_path=corpus)
        from app.ddragon import dragon_for_patch as scoped_dragon_for_patch
        dragon = scoped_dragon_for_patch(SCOPED_DEVELOPMENT[q['benchmark_purpose']]['patch'])
    elif manifest.get('recovery_version') is not None:
        from app.dataset_generations import validate_recovered_evidence
        admission_path = corpus / 'decision_admission.json'
        admission = json.loads(admission_path.read_text(encoding='utf-8'))
        validate_recovered_evidence(q, audit, manifest, admission, sha256(admission_path))
        from footwear_runtime import activate, declaration
        if manifest.get('footwear_runtime') != declaration():
            raise ValueError('Recovered corpus requires one explicit reviewed runtime')
        for name, digest in manifest['reconstruction_code_sha256'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Recovered source semantics changed after audit: {name}')
        activate()
        from app.ddragon import dragon_for_patch as recovered_dragon_for_patch
        dragon = recovered_dragon_for_patch(manifest['patch'])
    else:
        if q.get('decision_admission_sha256') or q.get('footwear_qualification'):
            raise ValueError('Decision masks require an explicitly recovered corpus')
        dragon = dragon_for_patch(manifest['patch'])
    import importlib
    baseline = importlib.import_module('baseline')
    if dragon.signature != manifest['static_data_sha256']:
        raise ValueError('Patch static data changed')
    for name in ('app/ddragon.py','app/shop_econ.py'):
        if sha256(ROOT / name) != manifest['reconstruction_code_sha256'][name]:
            raise ValueError(f'Row semantics changed since the source audit: {name}')
    return QualifiedCorpus(corpus,path,q,manifest,dragon,admission)


def iter_games(context):
    """Yield metadata, one bounded game's rows, and any export exclusion."""
    global baseline
    if baseline is None:
        import importlib
        baseline = importlib.import_module('baseline')
    accepted = set(context.qualification['accepted_match_ids'])
    with read_db(context.path / 'tracker.db') as conn:
        for game in conn.execute('SELECT * FROM games ORDER BY match_id'):
            mid = game['match_id']
            if mid not in accepted:
                continue
            metadata = dict(game)
            participants = json.loads(game['participants_json'])
            metadata['participants'] = participants
            if game['game_duration'] < baseline.MIN_DURATION_S:
                yield metadata, [], 'duration_below_600s'
                continue
            if len(participants) != 10:
                raise ValueError(f'{mid}: incomplete qualified lobby')
            by_puuid = {p['puuid']:p for p in participants}
            shoppers, rows, raw_count = set(), [], 0
            mask = None
            admitted_keys, excluded_keys, seen_model_keys = set(), set(), set()
            if getattr(context, 'decision_admission', None) is not None:
                from build_recovered_patch_corpus import decision_key
                mask = context.decision_admission['games'][mid]
                admitted_keys = {decision_key(value) for value in mask['admitted_decision_ids']}
                excluded_keys = {decision_key(value) for value in mask['boundary_exclusions']}
            source_completions = defaultdict(list)
            for record in conn.execute('SELECT payload_json FROM game_events WHERE match_id=? ORDER BY event_index',(mid,)):
                event = json.loads(record['payload_json'])
                raw_count += 1
                shoppers.add(event['puuid'])
                expected = {'reconstruction_version':context.manifest['reconstruction_version'],
                    'inventory_version':context.manifest['inventory_version'],
                    'label_source_version':context.manifest['label_source_version'],
                    'gold_est_version':GOLD_EST_VERSION,'patch':context.manifest['patch'],
                    'slot_state_version':SLOT_STATE_VERSION,
                    'ddragon_version':context.dragon.version,'static_data_sha256':context.dragon.signature}
                if any(event.get(k) != v for k,v in expected.items()) or event.get('training_eligible') is not True:
                    raise ValueError(f'{mid}: unqualified or mixed row provenance')
                if type(event.get('bot_quest_complete')) is not bool or any(
                    member.get('slot_state_version') != SLOT_STATE_VERSION
                    or type(member.get('bot_quest_complete')) is not bool
                    for member in event['board']
                ):
                    raise ValueError(f'{mid}: missing or unknown pre-decision slot state')
                if mask is not None and not event.get('is_save'):
                    # Plan-head targets are labels. Boundary omission cannot
                    # erase a real completion and substitute a later purchase.
                    for item in baseline._item_ids(event.get('label_bought', [])):
                        classification = context.dragon.classify(item)
                        if (not classification.get('target_skip', classification.get('skip'))
                            and (classification.get('is_completed') or classification.get('is_boots'))):
                            source_completions[event['participant_id']].append((event['ts'], item))
                event['match_id'] = mid
                event['perks'] = by_puuid[event['puuid']]
                row = baseline._example(event,context.dragon)
                if row:
                    if mask is not None:
                        key = decision_key(event)
                        if key in seen_model_keys or key not in admitted_keys | excluded_keys:
                            raise ValueError(f'{mid}: model row lies outside the audited decision mask')
                        seen_model_keys.add(key)
                        if key in excluded_keys:
                            continue
                    row['participant_id'] = event['participant_id']
                    row['label_counts'] = serialize_counts(label_counts(row))
                    row['target_encoding'] = TARGET_ENCODING
                    row.pop('label_ids')
                    # Keep the label-derived legacy wallet in the forensic DB,
                    # never in a qualified model export.
                    row.pop('gold_arrival_true',None)
                    rows.append(row)
            metadata['raw_shop_events'] = raw_count
            metadata['boundary_excluded_rows'] = len(excluded_keys)
            if mask is not None and (seen_model_keys != admitted_keys | excluded_keys or len(rows) != len(admitted_keys)):
                raise ValueError(f'{mid}: audited decision membership/count changed during export')
            if mask is not None:
                for values in source_completions.values():
                    values.sort()
                for row in rows:
                    completions = source_completions[row['participant_id']]
                    next_index = bisect_left(completions, (row['ts'], -1))
                    row['source_target_id'] = completions[next_index][1] if next_index < len(completions) else 0
                    row['source_target_version'] = 'full-source-completions-v1'
            if shoppers != by_puuid.keys():
                yield metadata, [], 'incomplete_shop_perspectives'
            elif not rows:
                yield metadata, [], 'no_purchase_or_save_targets'
            else:
                yield metadata, rows, None
