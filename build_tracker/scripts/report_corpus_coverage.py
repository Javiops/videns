"""Gate 3: coverage and a frozen split plan, before any training export."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
from corpus_rows import load_qualified, iter_games, row_builder_digest, read_db
from app.dataset_artifacts import sha256, write_json, publish_directory
from app.purchase_multiset import label_counts, TARGET_ENCODING
from app.dataset_generations import PURPOSES, SPLITS, DEVELOPMENT_PURPOSES


def preserve_assignment(created, prior_path):
    """Filter membership metadata without reading any old examples or test payload."""
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if not isinstance(prior,dict) or any(value not in (*SPLITS,'post_cutoff') for value in prior.values()):
        raise ValueError('Expected an explicit match-ID to frozen-split mapping')
    if not created.keys() <= prior.keys():
        raise ValueError('Eligible games lack prior frozen assignments; declare a reviewed split plan first')
    if any(prior[mid] not in SPLITS for mid in created):
        raise ValueError('Post-cutoff games cannot silently enter a development fold')
    return {mid:prior[mid] for mid in created}


def coverage(qualification, destination, *, benchmark_id='16.18-initial-pipeline-smoke-v2', eval_version=2,
             purpose='pipeline_smoke_only', split_assignment=None, split_metadata=None):
    if destination.exists():
        raise ValueError('Coverage generation exists; refusing overwrite')
    if purpose not in PURPOSES or not benchmark_id or type(eval_version) is not int or eval_version < 1:
        raise ValueError('An explicit valid scope, benchmark identity and positive evaluation version are required')
    if purpose in DEVELOPMENT_PURPOSES and (split_assignment is None or split_metadata is None or benchmark_id == '16.18-initial-pipeline-smoke-v2'):
        raise ValueError('Development requires an explicit benchmark identity, preserved assignment and holdout metadata')
    context = load_qualified(qualification)
    if purpose == 'patch_development' and context.manifest['patch'] != '16.17':
        raise ValueError('The authorized development benchmark is patch 16.17 only')
    from app.dataset_generations import SCOPED_DEVELOPMENT
    if purpose in SCOPED_DEVELOPMENT:
        if context.qualification.get('benchmark_purpose') != purpose or context.manifest['patch'] != SCOPED_DEVELOPMENT[purpose]['patch']:
            raise ValueError('16.18 coverage requires the separately authorized scoped qualification')
        if benchmark_id != context.qualification['benchmark_id'] or eval_version != context.qualification['eval_version']:
            raise ValueError('16.18 benchmark identity differs from its admission')
    elif context.qualification.get('benchmark_purpose') in SCOPED_DEVELOPMENT:
        raise ValueError('A 16.18 benchmark cannot be exported under another purpose')
    import baseline
    builder = row_builder_digest()
    state = json.loads((context.path / 'state.json').read_text(encoding='utf-8'))
    pros = {value['puuid']:name for name,value in state['pros'].items() if value.get('puuid')}
    pro_games_source, pro_games_quality = Counter(), Counter()
    quality = set(context.qualification['accepted_match_ids'])
    with read_db(context.path / 'tracker.db') as conn:
        for mid, participants in conn.execute('SELECT match_id,participants_json FROM games'):
            names = {pros[p['puuid']] for p in json.loads(participants) if p['puuid'] in pros}
            pro_games_source.update(names)
            if mid in quality: pro_games_quality.update(names)
    # Only counters and one game's rows are retained, never the whole export.
    created, game_counts, exclusions = {}, {}, {}
    role, champion_role, labels, kinds, region, pro_rows, pro_games = (Counter() for _ in range(7))
    max_inventory, max_basket, max_copies, total_rows, raw_events, masked_events = 0,0,0,0,0,0
    for game, rows, excluded in iter_games(context):
        mid = game['match_id']
        if excluded:
            exclusions[mid] = excluded
            continue
        created[mid] = game['game_creation']
        game_counts[mid] = len(rows)
        region[mid.split('_')[0]] += 1
        present = {pros[p['puuid']] for p in game['participants'] if p['puuid'] in pros}
        pro_by_pid = {p['participant_id']:pros[p['puuid']] for p in game['participants'] if p['puuid'] in pros}
        pro_games.update(present)
        raw_events += game['raw_shop_events']
        masked_events += game.get('boundary_excluded_rows', 0)
        for row in rows:
            counts = label_counts(row)
            total_rows += 1
            role[row['role']] += 1
            champion_role[f"{row['champion']}:{row['role']}"] += 1
            labels.update(counts)
            kinds[row['save_kind']] += 1
            if row['participant_id'] in pro_by_pid: pro_rows[pro_by_pid[row['participant_id']]] += 1
            max_inventory = max(max_inventory,len(row['inventory']),*(len(p['items']) for p in row['others']))
            max_basket = max(max_basket,sum(counts.values()))
            max_copies = max(max_copies,max(counts.values()))
    if not created or not total_rows:
        raise ValueError('No complete, qualified model examples')
    assignment = preserve_assignment(created,split_assignment) if split_assignment else baseline.temporal_split(created)
    splits = Counter({s:0 for s in SPLITS})
    splits.update(assignment.values())
    split_rows = Counter({s:0 for s in SPLITS})
    for mid, n in game_counts.items(): split_rows[assignment[mid]] += n
    required_folds = {'train','val'} if purpose in DEVELOPMENT_PURPOSES else set(SPLITS)
    if set(assignment.values()) != required_folds:
        raise ValueError('The qualified sample cannot support the required three-way split')
    if row_builder_digest() != builder:
        raise ValueError('Row builder changed during coverage')
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir(parents=True)
    plan = {'schema':1,'benchmark_id':benchmark_id,'eval_version':eval_version,
        'purpose':purpose,'patch':context.manifest['patch'],'qualification_sha256':sha256(qualification),
        'row_builder_sha256':builder,'assignment':assignment,'game_creation_ms':created,
        'game_counts':dict(splits),'row_counts':dict(split_rows),
        'policy':'whole games, temporal 70/15/15, timestamp ties together; no final-test evaluation'}
    if context.manifest.get('recovery_version'):
        plan['decision_admission_sha256'] = context.manifest['decision_admission_sha256']
    if split_assignment:
        shutil.copyfile(split_assignment,pending / 'preserved_split_assignment.json')
        plan.update(preserved_split_assignment_sha256=sha256(pending / 'preserved_split_assignment.json'),
            policy='whole games; preserve every eligible game prior split membership; no final-test evaluation')
    if split_metadata:
        metadata = json.loads(split_metadata.read_text(encoding='utf-8'))
        if purpose == 'patch_development':
            from app.dataset_generations import validate_development_split_metadata
            validate_development_split_metadata(metadata)
        elif purpose in SCOPED_DEVELOPMENT:
            from app.dataset_generations import validate_1618_admission, validate_1618_split
            bound = validate_1618_admission(qualification.parent,context.qualification)
            if metadata != bound['metadata'] or any(bound['assignment'].get(mid) != fold for mid,fold in assignment.items()):
                raise ValueError('16.18 coverage changed its admitted split metadata')
            validate_1618_split(assignment,metadata,bound['source_plan'])
        shutil.copyfile(split_metadata,pending / 'split_metadata.json')
        plan.update(split_metadata=metadata,split_metadata_sha256=sha256(pending / 'split_metadata.json'))
    write_json(pending / 'split_plan.json',plan)
    report = {'schema':1,'status':'coverage_complete','purpose':purpose,'patch':context.manifest['patch'],
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'qualification':str(qualification),'qualification_sha256':sha256(qualification),
        'corpus_manifest_sha256':context.qualification['corpus_manifest_sha256'],'row_builder_sha256':builder,
        'split_plan_sha256':sha256(pending / 'split_plan.json'),'source_games':context.qualification['source_games'],
        'quality_accepted_games':len(quality),'export_eligible_games':len(created),'export_exclusions':exclusions,
        'raw_shop_events_in_eligible_games':raw_events,'model_rows':total_rows,
        'boundary_excluded_model_rows':masked_events,'unlabelled_shop_events':raw_events-total_rows-masked_events,
        'region_games':dict(region),'role_rows':dict(role),'champion_role_rows':dict(champion_role),
        'label_item_occurrences':dict(labels),'save_kind_rows':dict(kinds),
        'max_modeled_inventory_items':max_inventory,'max_basket_items':max_basket,'max_copies_of_one_item':max_copies,
        'target_encoding':TARGET_ENCODING,
        'inventory_count_unit':'item copies, including stacked wards and role-slot items; not physical slots',
        'max_modeled_regular_slots':context.qualification['accepted_max_regular_slots'],
        'inventory_constraint_states':context.qualification['inventory_constraint_states'],
        'pro_games_in_source':dict(pro_games_source),'pro_games_after_quality':dict(pro_games_quality),
        'pro_games_export_eligible':dict(pro_games),'pro_shopper_rows':dict(pro_rows),
        'pro_accounts_resolved':len(pros),'unresolved_pros':context.manifest['unresolved_pros'],
        'game_counts_by_split':dict(splits),'row_counts_by_split':dict(split_rows),
        'smoke_ready':True,'release_ready':False,
        'limits':['Counts describe the qualified KR/EUW roster/pro sample, not every high-elo player or pro.',
            'Pro game memberships overlap and must not be summed as unique games.',
            'Sparse champion-role cells and exclusions limit representation; results apply only to this frozen patch sample.',
            'This is schema/population QA. No model has been scored and no final-test examples evaluated.']}
    if context.manifest.get('recovery_version'):
        report['decision_admission_sha256'] = context.manifest['decision_admission_sha256']
        report['recovery_version'] = context.manifest['recovery_version']
    write_json(pending / 'coverage.json',report)
    for name in ('report_corpus_coverage.py','corpus_rows.py'):
        shutil.copy2(ROOT / 'scripts' / name,pending / name)
    for path in pending.iterdir(): path.chmod(0o444)
    publish_directory(pending,destination)
    print(json.dumps({k:report[k] for k in ('export_eligible_games','model_rows','region_games','role_rows','game_counts_by_split','row_counts_by_split','max_modeled_inventory_items','max_basket_items','max_copies_of_one_item','pro_games_export_eligible')},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--benchmark-id',default='16.18-initial-pipeline-smoke-v2')
    p.add_argument('--eval-version',type=int,default=2)
    p.add_argument('--purpose',choices=PURPOSES,default='pipeline_smoke_only')
    p.add_argument('--split-assignment',type=Path,help='Prior frozen match-to-fold mapping; required for patch development')
    p.add_argument('--split-metadata',type=Path,help='Holdout reservation and prior validation exposure metadata')
    a = p.parse_args()
    if a.purpose in DEVELOPMENT_PURPOSES and '--eval-version' not in sys.argv:
        p.error('patch_development requires explicit --eval-version')
    coverage(a.qualification.resolve(),a.out.resolve(),benchmark_id=a.benchmark_id,eval_version=a.eval_version,
             purpose=a.purpose,split_assignment=a.split_assignment.resolve() if a.split_assignment else None,
             split_metadata=a.split_metadata.resolve() if a.split_metadata else None)
