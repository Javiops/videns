"""Gate 4: publish an immutable qualified export; update its pointer last."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
from corpus_rows import load_qualified, iter_games, row_builder_digest
from app.dataset_artifacts import sha256, write_json, publish_directory
from app.dataset_generations import SPLITS, PURPOSES, BENCHMARK_1618_FILES, SCOPED_DEVELOPMENT, generation_fingerprint, verify_generation
from app.reconstruct import SLOT_STATE_VERSION


def validate_slot_state(row):
    """A new qualified export must never rely on legacy unknown-slot decoding."""
    for member in (row, *row.get('others', [])):
        if member.get('slot_state_version') != SLOT_STATE_VERSION or type(member.get('bot_quest_complete')) is not bool:
            raise ValueError('New exports require explicit, observed pre-decision slot state')
    occupancy = row.get('unmodeled_regular_slots')
    if type(occupancy) is not int or not 0 <= occupancy <= 6:
        raise ValueError('New exports require explicit skipped-item slot occupancy')


def publish(coverage_dir, root, name, *, fault=None):
    """fault(stage) is for interruption tests, never a CLI bypass."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):
        raise ValueError('Invalid generation name')
    destination = root / name
    if destination.exists():
        raise ValueError('Immutable generation already exists')
    coverage_path, plan_path = coverage_dir / 'coverage.json',coverage_dir / 'split_plan.json'
    coverage = json.loads(coverage_path.read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    q_path = Path(coverage['qualification'])
    if coverage.get('status') != 'coverage_complete' or not coverage.get('smoke_ready') or sha256(plan_path) != coverage['split_plan_sha256'] or sha256(q_path) != coverage['qualification_sha256']:
        raise ValueError('Coverage/split plan is incomplete or changed')
    context = load_qualified(q_path)
    builder = row_builder_digest()
    purpose = plan.get('purpose')
    if builder != plan['row_builder_sha256'] or builder != coverage['row_builder_sha256'] or purpose not in PURPOSES:
        raise ValueError('Frozen row construction or experiment scope changed')
    assignment = plan['assignment']
    if not assignment.keys() <= set(context.qualification['accepted_match_ids']):
        raise ValueError('Split includes unqualified games')
    pending = root / f'.{name}.{uuid.uuid4().hex}.partial'
    pending.mkdir(parents=True)
    hashes = {s:hashlib.sha256() for s in SPLITS}
    rows, seen = Counter({s:0 for s in SPLITS}), set()
    versions, inventory_versions, label_versions = Counter(),Counter(),Counter()
    slot_versions = Counter()
    footwear_versions = Counter()
    with ExitStack() as stack:
        streams = {s:stack.enter_context((pending / f'visits_{s}.jsonl').open('xb')) for s in SPLITS}
        for game, examples, excluded in iter_games(context):
            mid = game['match_id']
            if excluded:
                if mid in assignment: raise ValueError(f'Frozen eligible game became ineligible: {mid}')
                continue
            if mid not in assignment or game['game_creation'] != plan['game_creation_ms'][mid]:
                raise ValueError(f'Game is outside the frozen split plan: {mid}')
            seen.add(mid)
            split = assignment[mid]
            for row in examples:
                validate_slot_state(row)
                if context.manifest.get('recovery_version'):
                    from footwear_runtime import validate_state
                    for member in (row, *row.get('others', [])):
                        validate_state(member)
                    if (row.get('source_target_version') != 'full-source-completions-v1'
                        or type(row.get('source_target_id')) is not int or row['source_target_id'] < 0):
                        raise ValueError('Recovered rows require explicit full-source completion labels')
                    footwear_versions[row['footwear_state_version']] += 1
                payload = (json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
                streams[split].write(payload)
                hashes[split].update(payload)
                rows[split] += 1
                versions[row['gold_est_version']] += 1
                inventory_versions[row['inventory_version']] += 1
                label_versions[row['label_source_version']] += 1
                slot_versions[row['slot_state_version']] += 1
            if fault: fault('game_written')
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
    if seen != assignment.keys() or dict(rows) != plan['row_counts'] or row_builder_digest() != builder:
        raise ValueError('Export rows, game membership, or code differ from frozen coverage')
    write_json(pending / 'split_assignment.json',assignment)
    copies = {'coverage.json':coverage_path,'split_plan.json':plan_path,'qualification.json':q_path,
        'semantic_audit.json':Path(context.qualification['semantic_audit']),
        'qualification_code.py':q_path.parent / 'qualification_code.py',
        'audit_code.py':Path(context.qualification['semantic_audit']).parent / 'audit_code.py',
        'corpus_manifest.json':context.path / 'corpus_manifest.json'}
    if purpose in SCOPED_DEVELOPMENT:
        for filename in BENCHMARK_1618_FILES:
            copies[filename] = q_path.parent/filename
    elif context.manifest.get('recovery_version'):
        for filename in ('decision_admission.json', 'recovery_inputs.json', 'recovery_summary.json'):
            copies[filename] = context.path / filename
        copies['recovered_audit_code.py'] = Path(context.qualification['semantic_audit']).parent / 'recovered_audit_code.py'
    if 'preserved_split_assignment_sha256' in plan:
        prior_path = coverage_dir / 'preserved_split_assignment.json'
        if sha256(prior_path) != plan['preserved_split_assignment_sha256']:
            raise ValueError('Prior frozen assignment changed after coverage')
        prior = json.loads(prior_path.read_text(encoding='utf-8'))
        if any(prior.get(mid) != fold for mid,fold in assignment.items()):
            raise ValueError('Frozen development membership changed')
        copies['preserved_split_assignment.json'] = prior_path
    if 'split_metadata_sha256' in plan:
        metadata_path = coverage_dir / 'split_metadata.json'
        if sha256(metadata_path) != plan['split_metadata_sha256']:
            raise ValueError('Split metadata changed after coverage')
        copies['split_metadata.json'] = metadata_path
    for name_out, source in copies.items():
        shutil.copyfile(source,pending / name_out)
    data_files = {f'visits_{s}.jsonl':{'sha256':hashes[s].hexdigest(),'bytes':(pending / f'visits_{s}.jsonl').stat().st_size,'rows':rows[s]} for s in SPLITS}
    data_files['split_assignment.json'] = {'sha256':sha256(pending / 'split_assignment.json'),'bytes':(pending / 'split_assignment.json').stat().st_size}
    fingerprint = generation_fingerprint(data_files,sha256(q_path),builder,sha256(plan_path))
    split_manifest = {'schema':1,'eval_version':plan['eval_version'],'benchmark_id':plan['benchmark_id'],
        'export_fingerprint':fingerprint,'probe':True,'purpose':purpose,'patch':context.manifest['patch'],
        'ddragon_version':context.dragon.version,'static_data_sha256':context.dragon.signature,
        'reconstruction_versions':{context.manifest['reconstruction_version']:sum(rows.values())},
        'gold_est_versions':dict(versions),'inventory_versions':dict(inventory_versions),'label_source_versions':dict(label_versions),
        'slot_state_versions':dict(slot_versions),
        'counts':plan['game_counts'],'rows_per_split':dict(rows),'strategy':plan['policy'],
        'post_cutoff_count':0,'created_at':datetime.now(timezone.utc).isoformat(),
        'qualification_sha256':sha256(q_path),'row_builder_sha256':builder,
        'target_encoding':coverage['target_encoding'],
        'input_schema':{k:coverage[k] for k in ('max_modeled_inventory_items','max_basket_items','max_copies_of_one_item')}}
    if 'split_metadata' in plan:
        split_manifest['split_metadata'] = plan['split_metadata']
    if context.manifest.get('recovery_version'):
        split_manifest['footwear_runtime'] = context.manifest['footwear_runtime']
        split_manifest['footwear_state_versions'] = dict(footwear_versions)
        split_manifest['decision_admission_sha256'] = context.manifest['decision_admission_sha256']
        split_manifest['input_schema']['source_target_version'] = 'full-source-completions-v1'
        split_manifest['source_target_version'] = 'full-source-completions-v1'
    write_json(pending / 'split_manifest.json',split_manifest)
    code_dir = pending / 'export_code'
    code_dir.mkdir()
    for script in ('publish_patch_dataset.py','corpus_rows.py'):
        shutil.copyfile(ROOT / 'scripts' / script,code_dir / script)
    shutil.copyfile(ROOT / 'app/purchase_multiset.py',code_dir / 'purchase_multiset.py')
    if context.manifest.get('recovery_version'):
        for filename in ('footwear_runtime.py', 'footwear_economics.py', 'baseline.py'):
            shutil.copyfile(ROOT / 'scripts' / filename, code_dir / filename)
        if purpose in SCOPED_DEVELOPMENT:
            tag = SCOPED_DEVELOPMENT[purpose]['patch'].replace('.', '')
            for filename in (f'benchmark_{tag}_runtime.py',f'reconstructible_{tag}_runtime.py',f'prepare_benchmark_{tag}.py'):
                shutil.copyfile(ROOT/'scripts'/filename,code_dir/filename)
    files = dict(data_files)
    for path in pending.rglob('*'):
        if path.is_file() and path.relative_to(pending).as_posix() not in files:
            files[path.relative_to(pending).as_posix()] = {'sha256':sha256(path),'bytes':path.stat().st_size}
    manifest = {'schema':1,'state':'complete','purpose':purpose,'patch':context.manifest['patch'],'export_fingerprint':fingerprint,
        'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,'generation_name':destination.name,
        'publisher_code_sha256':sha256(Path(__file__)),'production_hold_cleared':False,'promotion_allowed':False}
    if context.manifest.get('recovery_version'):
        manifest['footwear_runtime'] = context.manifest['footwear_runtime']
        manifest['source_target_version'] = 'full-source-completions-v1'
    write_json(pending / 'generation_manifest.json',manifest)
    verify_generation(pending,staged=True)
    if fault: fault('validated')
    for path in pending.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    publish_directory(pending,destination)
    if fault: fault('published')
    write_json(root / 'CURRENT.json',{'generation':str(destination),'generation_manifest_sha256':sha256(destination / 'generation_manifest.json'),
        'export_fingerprint':fingerprint,'purpose':purpose})
    print(json.dumps({'generation':str(destination),'export_fingerprint':fingerprint,'rows':dict(rows),'production_hold_cleared':False},indent=2))
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT / 'data/dataset_generations')
    p.add_argument('--name',required=True)
    a = p.parse_args()
    publish(a.coverage.resolve(),a.root.resolve(),a.name)
