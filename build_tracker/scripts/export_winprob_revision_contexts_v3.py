"""Expanded-census native pre-A contexts; each invocation owns one clean runtime.

No mixed training export is read. Frozen v1 producers and artifacts remain
unchanged. Full mode is staged for an explicitly reviewed durable launch.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import closing
from datetime import datetime,timezone
import importlib
import json
from pathlib import Path
import sqlite3
import time

import export_winprob_trajectories_v1 as producer
import align_winprob_revision_contexts_v1 as alignment

ROOT=producer.ROOT
SCHEMA='winprob-revision-source-alignment-v3'
CERTIFICATE_SHA256='13fbac82f399437658b86a303e6995cf9b8f62ab912b1d2fbb08bdadd27d563c'
PATCH_SOURCES={
    '16.17':{'corpus':'data/corpora/16.17-history-v1','evidence':'data/acceptance/16.17_history_20260927',
             'history':'data/history/16.17-history-v1/history_manifest.json'},
    '16.18':{'corpus':'data/corpora/16.18-reconstructible-v4','evidence':'data/dataset_generations/16.18-reconstructible-development-v4'},
    '16.19':{'corpus':'data/corpora/16.19-reconstructible-v1','evidence':'data/dataset_generations/16.19-development-v1'},
}
ROLE_MAP={'fit':'train','calibration':'calibration','development':'development'}


def require(condition,message):
    if not condition:raise ValueError(message)


def context_id(pair):
    fields=('match_id','participant_id','feature_cutoff_ms','undo_ms','decision_timestamp_ms','action_a','action_b')
    return producer.value_hash({key:pair[key] for key in fields})


def pair_id(pair):
    fields=('match_id','participant_id','feature_cutoff_ms','undo_ms','decision_timestamp_ms','action_a','action_b')
    return '|'.join(str(pair[key]) for key in fields)


def select_records(records,manifest,source_ids,denied,patch,roles,limit):
    """Outcome-free pair rows only, after their entire source universe is denied."""
    require(set(roles)<=set(ROLE_MAP) and roles,'Unknown or empty requested role set')
    require(type(limit) is int and limit>=0,'Invalid pair limit')
    admitted={row['match_id']:row for row in manifest['games']}
    require(not set(source_ids)&set(denied),'Census universe contains a reserved identity')
    output=[];seen=set()
    for line_number,pair in records:
        mid=pair['match_id']
        require(mid in source_ids and mid not in denied,'Census record outside admitted source universe')
        if pair.get('patch')!=patch or pair.get('role') not in roles or pair.get('eligible_revision') is not True:continue
        entry=admitted.get(mid)
        require(entry is not None and entry.get('historical_validation') is False
                and entry['role']==ROLE_MAP[pair['role']] and entry['patch']==patch,'Pair role/patch cannot be remapped')
        identity=context_id(pair)
        require(identity not in seen,'Duplicate canonical preference context')
        seen.add(identity);output.append({**pair,'source_line_number':line_number,'context_id':identity,'pair_id':pair_id(pair)})
        if limit and len(output)>=limit:break
    require(bool(output),'No eligible pairs in the fixed requested patch/roles')
    return output


def validate_expanded_claims(report,source,manifest,original_source):
    require(report.get('schema')=='undo-pairs-support-v2' and report.get('stage')=='outcome_blind_expanded_counts'
            and report.get('outcome_targets_extracted') is False and report.get('cache_y_accessed') is False,
            'Expected expanded outcome-blind census v2')
    require(report.get('certificate_changed_from_v1') is False and report.get('certificate_sha256')==CERTIFICATE_SHA256,
            'Expanded census changed the original receipt certificate')
    games=source['games'];ids={row['match_id'] for row in games}
    require(len(games)==len(ids)==5861 and Counter(row['role'] for row in games)=={'fit':4002,'calibration':1000,'development':859},
            'Expanded census must contain exactly4002fit/1000calibration/859development games')
    for role in ('fit','calibration'):
        require({row['match_id'] for row in games if row['role']==role}
                =={row['match_id'] for row in original_source['games'] if row['role']==role},
                'Expanded census changed original '+role+' membership')
    expected_dev={row['match_id'] for row in manifest['games'] if row['role']=='development' and row['patch']=='16.19'}
    actual_dev={row['match_id'] for row in games if row['role']=='development'}
    require(actual_dev==expected_dev and len(expected_dev)==859
            and all(row['patch']=='16.19' for row in games if row['role']=='development'),
            'Expanded development is not exactly all859 frozen16.19 games')


def load_census(census,manifest,denied,patch,roles,limit):
    census=Path(census).resolve(strict=True)
    require(not any((parent/'QUARANTINED.md').exists() for parent in (census,*census.parents)),'Quarantined census')
    report=producer.load_json(census/'report.json')
    require(report.get('schema')=='undo-pairs-support-v2' and report.get('stage')=='outcome_blind_expanded_counts'
            and report.get('outcome_targets_extracted') is False and report.get('cache_y_accessed') is False,
            'Expected frozen expanded outcome-blind revision census')
    bindings=dict(report['binding_sha256'])
    for path,digest in bindings.items():require(producer.file_hash(path)==digest,'Census binding changed: '+path)
    pairs_path=census/'pairs.jsonl'
    require(producer.file_hash(pairs_path)==report['pairs_sha256'],'Census pairs digest mismatch')
    source_paths=[Path(path) for path in bindings if Path(path).name=='source_manifest.json']
    require(len(source_paths)==1,'Require one census source manifest')
    source=producer.load_json(source_paths[0])
    require(all(source.get(key) is False for key in ('test_payload_reads','confirmatory_payload_reads','historical_validation_payload_reads')),
            'Unverified census source population')
    certificate=Path(report['certificate_implementation']).resolve(strict=True)
    require(producer.file_hash(certificate)==CERTIFICATE_SHA256 and bindings.get(str(certificate))==CERTIFICATE_SHA256,
            'Original certificate implementation is not identically bound')
    context_metadata_paths=[Path(path) for path in bindings if Path(path).name=='cache_metadata.json']
    require(len(context_metadata_paths)==1,'Require one expanded context metadata artifact')
    context_meta,context_meta_hash=producer.load_bound(context_metadata_paths[0])
    require(context_meta.get('schema')=='winprob-revision-context-v1' and context_meta.get('state')=='complete'
            and context_meta.get('contains_outcome_array') is False and context_meta_hash==bindings[str(context_metadata_paths[0])],
            'Expanded context cache contract changed')
    original_meta_path=Path(context_meta['base_cache_metadata_path']).resolve(strict=True)
    original_meta,original_meta_hash=producer.load_bound(original_meta_path)
    require(original_meta_hash==context_meta['base_cache_metadata_sha256'],'Original clean cache metadata changed')
    original_source_path=Path(original_meta['source_manifest_path']).resolve(strict=True)
    require(producer.file_hash(original_source_path)==original_meta['source_manifest_sha256'],'Original clean source membership changed')
    original_source=producer.load_json(original_source_path)
    validate_expanded_claims(report,source,manifest,original_source)
    bindings.update({str(original_meta_path):original_meta_hash,str(original_source_path):original_meta['source_manifest_sha256']})
    source_ids={row['match_id'] for row in source['games']}
    require(len(source_ids)==len(source['games']),'Duplicate census source game')
    require(not source_ids&denied,'Census source universe contains reserved games')
    with pairs_path.open('r',encoding='utf-8') as handle:
        pairs=select_records(((n,json.loads(line)) for n,line in enumerate(handle,1)),manifest,source_ids,denied,patch,roles,limit)
    bindings.update({str(census/'report.json'):producer.file_hash(census/'report.json'),str(pairs_path):report['pairs_sha256']})
    return pairs,bindings


def checked_file(path,expected,bindings):
    path=Path(path).resolve(strict=True)
    require(producer.file_hash(path)==expected,'Source evidence digest mismatch: '+str(path))
    bindings[str(path)]=expected
    return path


def native_metadata(patch,wanted):
    """Static qualification/identity metadata only; no corpus or raw payload."""
    spec=PATCH_SOURCES[patch];evidence=(ROOT/spec['evidence']).resolve(strict=True)
    corpus_path=(ROOT/spec['corpus']).resolve(strict=True)
    bindings={};excluded={}
    if patch=='16.17':
        history_path=ROOT/spec['history'];history=producer.load_json(history_path)
        require(history.get('state')=='complete' and history.get('purpose')=='prior_patch_transfer_pretraining_only'
                and history.get('test_payload_reads') is False and history.get('patch')==patch,'Unqualified history source')
        bindings[str(history_path)]=producer.file_hash(history_path)
        qpath=checked_file(evidence/'qualification.json',history['acceptance']['qualification_sha256'],bindings)
        cpath=checked_file(corpus_path/'corpus_manifest.json',history['corpus']['manifest_sha256'],bindings)
        apath=checked_file(evidence/'semantic_audit.json',history['acceptance']['semantic_audit_sha256'],bindings)
        ppath=checked_file(evidence/'source_plan.json',history['acceptance']['source_plan_sha256'],bindings)
        mask_path=checked_file(corpus_path/'decision_admission.json',history['corpus']['decision_admission_sha256'],bindings)
        # corpus_rows is deliberately not imported: its old all-game reader is
        # unsuitable for current cross-project reservations. All actually used
        # source projection/replay dependencies must still match history.
        for name,digest in history['row_builder_code'].items():
            if name!='scripts/corpus_rows.py':checked_file(ROOT/name,digest,bindings)
        source_assignment=None
    else:
        generation_path=evidence/'generation_manifest.json';generation=producer.load_json(generation_path)
        bindings[str(generation_path)]=producer.file_hash(generation_path)
        names=('qualification.json','corpus_manifest.json','semantic_audit.json','source_plan.json',
               'decision_admission.json','split_assignment.json','benchmark_admission.json')
        for name in names:
            binding=generation['files'][name]
            require((evidence/name).stat().st_size==binding['bytes'],'Source evidence size mismatch')
            checked_file(evidence/name,binding['sha256'],bindings)
        qpath=evidence/'qualification.json';cpath=evidence/'corpus_manifest.json';apath=evidence/'semantic_audit.json'
        ppath=evidence/'source_plan.json';mask_path=evidence/'decision_admission.json'
        source_assignment=producer.assignment(evidence/'split_assignment.json')
        frozen=evidence/'export_code/baseline.py'
        current_tree=ast.parse((ROOT/'scripts/baseline.py').read_text(encoding='utf-8-sig'))
        frozen_tree=ast.parse(frozen.read_text(encoding='utf-8-sig'))
        for name in ('_example','_label','_item_ids','_others','_unmodeled_regular_slots'):
            current=next(node for node in current_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
            old=next(node for node in frozen_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
            require(ast.dump(current,include_attributes=False)==ast.dump(old,include_attributes=False),'Reviewed projection changed: '+name)
        q_temp=producer.load_json(qpath)
        checked_file(corpus_path/'corpus_manifest.json',q_temp['corpus_manifest_sha256'],bindings)
    q=producer.load_json(qpath);corpus=producer.load_json(cpath);audit=producer.load_json(apath);plan=producer.load_json(ppath)
    require(q.get('patch')==patch and corpus.get('patch')==patch and audit.get('passed') is True and not audit.get('hard_failures'),
            'Native qualified source failed its audit')
    require(q['corpus_manifest_sha256']==producer.file_hash(cpath) and q['semantic_audit_sha256']==producer.file_hash(apath)
            and q['decision_admission_sha256']==producer.file_hash(mask_path)
            and q['db_sha256']==corpus['db_sha256']==audit['db_sha256'],'Native source evidence chain differs')
    require(Path(q['corpus']).resolve()==corpus_path and corpus.get('test_payload_reads') is False
            and plan.get('test_payload_reads') is False,'Native source path or reservation scope changed')
    accepted=set(q['accepted_match_ids']);records={row['match_id']:row for row in plan['matches']}
    approved=set()
    for mid in wanted:
        if mid not in accepted:excluded[mid]='game_not_in_native_qualified_corpus'
        elif mid not in records:excluded[mid]='game_missing_from_frozen_source_plan'
        elif source_assignment is not None and source_assignment.get(mid)!='train':excluded[mid]='game_not_in_native_source_train'
        else:approved.add(mid)
    masks=producer.selected_object_entries(mask_path,'games',approved) if approved else {}
    review=Path(plan['static_review']).resolve(strict=True)
    checked_file(review,plan['static_review_sha256'],bindings)
    return {'patch':patch,'corpus':corpus,'corpus_path':corpus_path,'evidence':evidence,'approved':approved,
            'excluded':excluded,'plans':{mid:records[mid] for mid in approved},'masks':masks,
            'review':review,'bindings':bindings,'db_reference_sha256':corpus['db_sha256']}


def activate_native(source):
    patch=source['patch'];suffix=patch.replace('.','')
    if patch=='16.17':
        runtime=importlib.import_module('reconstructible_1617_runtime')
        active=runtime.activate(source['review'])
    else:
        from footwear_runtime import activate_for_manifest
        active=activate_for_manifest(source['corpus'],corpus_path=source['corpus_path'])
    require(active==source['corpus']['footwear_runtime'],'Native source runtime differs from qualification')
    import baseline
    from app.reconstruct import reconstruct_game
    from app.ddragon import dragon_for_patch
    process=importlib.import_module('process_reconstructible_'+suffix)
    dragon=dragon_for_patch(patch)
    require(dragon.signature==source['corpus']['static_data_sha256'],'Native static data changed')
    return active,baseline,reconstruct_game,dragon,process.decision_mask


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Immutable context export already exists')
    require(args.patch in PATCH_SOURCES,'Unsupported native patch')
    require((args.full and args.limit==0) or (not args.full and 1<=args.limit<=10),'Use bounded≤10 smoke or separately reviewed --full --limit0')
    manifest,tests,validation,confirm,bindings=producer.validate_admission(args.admission)
    denied=tests|validation|confirm;roles=set(args.role or ['fit'])
    pairs,pair_bindings=load_census(args.census,manifest,denied,args.patch,roles,args.limit)
    bindings.update(pair_bindings);wanted={pair['match_id'] for pair in pairs}
    source=native_metadata(args.patch,wanted);bindings.update(source['bindings'])
    require(not source['approved']&denied,'Native admitted IDs intersect reservations')
    active,baseline,reconstruct_game,dragon,decision_mask=activate_native(source)
    attach_gold=reconstruct_game.__globals__['_attach_gold_estimates']
    code_paths=[Path(__file__),ROOT/'scripts/export_winprob_revision_contexts_v2.py',Path(producer.__file__),Path(alignment.__file__),ROOT/'scripts/baseline.py',
                ROOT/'scripts/footwear_runtime.py',ROOT/f'scripts/reconstructible_{args.patch.replace(".","")}_runtime.py',
                ROOT/f'scripts/process_reconstructible_{args.patch.replace(".","")}.py',ROOT/'app/footwear_reconstruction.py']
    code={str(path.resolve()):producer.file_hash(path) for path in code_paths}
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n')
    status('selected_before_payloads',pairs=len(pairs),source_games=len(wanted),qualified_games=len(source['approved']))
    selection={'schema':SCHEMA,'patch':args.patch,'roles':sorted(roles),'census':str(Path(args.census).resolve()),
               'selected_context_ids':[pair['context_id'] for pair in pairs],'selected_game_ids':sorted(wanted),
               'qualified_game_ids':sorted(source['approved']),'qualification_exclusions':source['excluded'],
               'input_metadata_sha256':bindings,'code_sha256':code,'global_tests_denied':len(tests),'confirmation_denied':len(confirm)}
    (output/'selection.json').write_text(json.dumps(selection,indent=2,sort_keys=True)+'\n')
    rows_path=output/'contexts.jsonl';records=[];loaded_mid=None;loaded=None;games_read=set()
    deadline=1200 if args.full else 150
    try:
        db_path=source['corpus_path']/'tracker.db';initial=db_path.stat()
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=2)) as conn, rows_path.open('x',encoding='utf-8',newline='\n') as handle:
            conn.execute('PRAGMA query_only=ON');conn.execute('PRAGMA cache_size=-2048')
            reader=producer.ApprovedGameReader(conn,source['approved'],denied) if source['approved'] else None
            for index,pair in enumerate(pairs,1):
                require(time.monotonic()-started<=deadline,'Native context export deadline exceeded')
                mid=pair['match_id']
                if mid not in source['approved']:
                    record={'schema':SCHEMA,'context_id':pair['context_id'],'match_id':mid,'participant_id':pair['participant_id'],
                            'source_pair_line':pair['source_line_number'],'role':pair['role'],'patch':pair['patch'],
                            'action_a':pair['action_a'],'action_b':pair['action_b'],'status':'unaligned','reason':source['excluded'][mid]}
                else:
                    if mid!=loaded_mid:
                        game,events=reader.game(mid)
                        raw,raw_hashes=producer.read_raw_pair(mid,source['plans'][mid],source['approved'],denied)
                        debug={};replayed_game,replayed_events=reconstruct_game(raw['match'],raw['timeline'],dragon,debug)
                        producer.verify_replay(game,events,replayed_game,replayed_events)
                        fresh=decision_mask(replayed_events,debug,dragon);frozen=source['masks'][mid]
                        require(all(fresh[key]==frozen[key] for key in ('admitted_decision_ids','boundary_exclusions','admitted_rows','boundary_excluded_rows')),
                                'Native decision mask no longer reproduces')
                        players={player['participant_id']:player for player in producer.load_participants(game)}
                        loaded=(events,players,debug,raw['timeline'],raw_hashes);loaded_mid=mid;games_read.add(mid)
                        del raw,replayed_game,replayed_events,game
                    events,players,debug,timeline,raw_hashes=loaded
                    try:
                        record=alignment.align_pair(pair,events,source['masks'][mid],players,lambda event:baseline._example(event,dragon),
                                                    attach_gold,debug,timeline,dragon)
                        record.update(schema=SCHEMA,context_id=pair['context_id'],raw_sources=raw_hashes,
                                      random_focal_pair_weight=pair['random_focal_pair_weight'],native_patch=args.patch)
                    except (ValueError,KeyError,TypeError) as error:
                        record={'schema':SCHEMA,'context_id':pair['context_id'],'match_id':mid,'participant_id':pair['participant_id'],
                                'source_pair_line':pair['source_line_number'],'role':pair['role'],'patch':pair['patch'],
                                'action_a':pair['action_a'],'action_b':pair['action_b'],'status':'unaligned','reason':str(error),
                                'error_type':type(error).__name__}
                record.update(pair_id=pair['pair_id'],undo_timestamp_ms=pair['undo_ms'],
                              context_timestamp_ms=pair['feature_cutoff_ms'],decision_timestamp_ms=pair['decision_timestamp_ms'])
                handle.write(producer.canonical(record).decode()+'\n');handle.flush()
                records.append({'status':record['status'],'reason':record.get('reason'),'role':pair['role']})
                if index%25==0 or index==len(pairs):status('aligning',processed_pairs=index,counts=dict(Counter(row['status'] for row in records)))
        require(db_path.stat().st_size==initial.st_size and db_path.stat().st_mtime_ns==initial.st_mtime_ns,'Native corpus changed during indexed reads')
        producer.validate_admission(args.admission)
        require(all(producer.file_hash(path)==digest for path,digest in {**bindings,**code}.items()),'Frozen native-context input/code changed')
        rows_hash=producer.file_hash(rows_path);rows_path.with_suffix('.sha256').write_text(rows_hash+'  contexts.jsonl\n',encoding='ascii')
        report={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'patch':args.patch,'requested_roles':sorted(roles),'full_mode':bool(args.full),'pairs_examined':len(pairs),
                'games_read':len(games_read),'status_counts':dict(Counter(row['status'] for row in records)),
                'status_by_role':{role:dict(Counter(row['status'] for row in records if row['role']==role)) for role in sorted(roles)},
                'unaligned_reasons':dict(Counter(row['reason'] for row in records if row['status']!='aligned')),
                'contexts_sha256':rows_hash,'selection_sha256':producer.file_hash(output/'selection.json'),
                'input_metadata_sha256':bindings,'code_sha256':code,'runtime':active,'source_database':str(db_path),
                'qualified_database_sha256_reference':source['db_reference_sha256'],'whole_database_scanned':False,
                'permitted_source_rows_equal_hash_bound_raw_replay':True,'outcomes_analyzed':False,
                'source_replay_internally_reads_final_match_fields':True,'policy_loaded':False,'training_performed':False,
                'test_payload_reads':False,'confirmation_payload_reads':False,'elapsed_s':time.monotonic()-started,
                'scope':'Native pre-A context alignment; exact M2 scoring is a separate16.18-pinned process.'}
        report_path=output/'report.json';report_path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        report_path.with_suffix('.sha256').write_text(producer.file_hash(report_path)+'  report.json\n',encoding='ascii')
        status('complete',counts=report['status_counts'])
        return report
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census',type=Path,default=ROOT.parent/'winprob/data/runs/undo-pairs-20260930-v2/experiment')
    parser.add_argument('--admission',type=Path,default=producer.ADMISSION)
    parser.add_argument('--patch',choices=tuple(PATCH_SOURCES),required=True)
    parser.add_argument('--role',choices=tuple(ROLE_MAP),action='append')
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args)
    print(json.dumps({key:report[key] for key in ('patch','requested_roles','pairs_examined','games_read','status_counts','unaligned_reasons','contexts_sha256','elapsed_s')},indent=2))


if __name__=='__main__':main()
