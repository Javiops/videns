"""Audit the exact frozen M2/direct-BC candidates in a clean native16.19 world.

Only versioned context/reference artifacts and source metadata/code are read.
No target, raw game, database, probability model or training corpus is opened.
Candidates are never filtered, replaced, or reranked by this audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import export_bc20_ordinary_reference_v1 as reference
producer=reference.producer
source=reference.source
require=reference.require
SCHEMA='winprob-ordinary-native19-audit-v1'


def candidate_union(record,groups):
    """Preserve membership/order of each arm; deduplicate exact objects only."""
    candidates={}
    for arm,key in (('m2','m2_options'),('direct_bc','direct_bc_options')):
        options=record[key]
        reference.validate_options(options,groups)
        for rank,option in enumerate(options):
            digest=producer.value_hash(option);identity=(option['group'],digest)
            if identity not in candidates:
                candidates[identity]={'group':option['group'],'option_sha256':digest,'option':option,'arms':{}}
            candidates[identity]['arms'][arm]=rank
    return list(candidates.values())


def validate_reference_rows(contexts,rows,groups,manifest):
    require(len(contexts)==len(rows)==859,'Reference ledger dropped selected games')
    available=0
    for ordinal,(context,row) in enumerate(zip(contexts,rows,strict=True)):
        require(all(row.get(key)==context.get(key) for key in reference.IDENTITY_KEYS)
                and row.get('ordinal')==ordinal and row.get('source_tensor_row')==context.get('tensor_row')
                and row.get('projection_sha256')==context.get('projection_sha256')
                and row.get('context_timestamp_ms')==context.get('context_timestamp_ms'),
                'Reference/source identity changed')
        if context['status']=='available':
            require(row.get('reference_status')=='scored' and type(row.get('m2_tensor_row')) is int
                    and row['m2_tensor_row']==available and row.get('zero_residual_native_options_equal') is True
                    and row.get('whole_distribution_label_mutation_equal') is True
                    and row.get('native_options_label_mutation_equal') is True,'Reference parity/mutation coverage incomplete')
            candidate_union(row,groups);available+=1
        else:
            require(row.get('m2_tensor_row') is None and row.get('m2_options')==[] and row.get('direct_bc_options')==[],
                    'Unavailable source row acquired recommendations')
    require(available==manifest.get('scored_rows')==manifest.get('parity_checked_rows')==manifest.get('mutation_checked_rows')
            and manifest.get('zero_residual_native_options_mismatches')==0,'Incomplete reference ledger coverage')


def load_reference(path,source_path,source_manifest,contexts):
    path=Path(path).resolve(strict=True);manifest,digest=producer.load_bound(path)
    require(manifest.get('schema')==reference.SCHEMA and manifest.get('state')=='complete'
            and manifest.get('selection_count')==859 and manifest.get('group_count')==9333
            and manifest.get('package_identity_sha256')==reference.PACKAGE_DIGEST
            and manifest.get('reference_world_patch')=='16.18','Frozen ordinary reference required')
    for key in ('target_file_opened','targets_analyzed','outcomes_analyzed','raw_payload_reads',
                'test_payload_reads','confirmation_payload_reads','training_performed','accuracy_computed'):
        require(manifest.get(key) is False,'Reference boundary flag failed: '+key)
    require(Path(manifest['source_manifest_path']).resolve()==Path(source_path).resolve()
            and manifest['source_manifest_sha256']==producer.file_hash(source_path)
            and manifest['source_selection_sha256']==source_manifest['files']['selection.json']
            and manifest['source_contexts_sha256']==source_manifest['files']['contexts.jsonl']
            and manifest['opaque_targets_sha256_not_opened']==source_manifest['files']['targets.jsonl'],
            'Reference lineage differs from native source contexts')
    bindings=reference.metadata_bindings({**manifest['bindings_sha256'],**manifest['code_sha256']})
    bindings[str(path)]=digest
    for name in ('rows.jsonl','probabilities.npz','group_main.json'):
        value=producer.file_hash(path.parent/name)
        require(value==manifest['files'][name],'Reference artifact changed: '+name)
        bindings[str(path.parent/name)]=value
    rows=reference.read_lines(path.parent/'rows.jsonl');groups=producer.load_json(path.parent/'group_main.json')
    require(len(groups)==9333 and groups[0]==[],'Reference main catalogue changed')
    require(manifest.get('row_order_sha256')==producer.value_hash([row['sample_id'] for row in rows]),'Reference ordering changed')
    validate_reference_rows(contexts,rows,groups,manifest)
    return manifest,rows,groups,bindings


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Refuse to overwrite native audit artifact')
    require(1<=args.deadline_seconds<=600,'Native audit deadline must be bounded at10minutes')
    manifest,contexts,selection,native_source,bindings=reference.load_contexts(args.source_manifest)
    ref,ref_rows,groups,ref_bindings=load_reference(args.reference_manifest,args.source_manifest,manifest,contexts)
    bindings.update(ref_bindings)
    active,_,_,dragon,_=source.native.activate_native(native_source)
    require(native_source['patch']=='16.19' and dragon.version=='16.19.1','Audit requires qualified native16.19.1')
    from app.conditional_policy import purchase_sequence
    code=reference.runtime_code_bindings();code[str(Path(__file__).resolve())]=producer.file_hash(__file__)
    reference.revalidate({**bindings,**code})
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n',encoding='utf-8')
    unique=0;counts=Counter();audit=[]
    try:
        for ordinal,(context,record) in enumerate(zip(contexts,ref_rows,strict=True)):
            require(time.monotonic()-started<=args.deadline_seconds,'Native19 audit deadline exceeded')
            result={**{key:context[key] for key in reference.IDENTITY_KEYS},'ordinal':ordinal,
                    'projection_sha256':context.get('projection_sha256'),'context_timestamp_ms':context.get('context_timestamp_ms'),
                    'reference_status':record['reference_status'],'candidates':[]}
            if context['status']=='available':
                row=reference.model_row(context)
                for candidate in candidate_union(record,groups):
                    entry=reference.audit_options([candidate['option']],row,groups,purchase_sequence,dragon)[0]
                    result['candidates'].append({**candidate,**entry})
                    unique+=1
                    for arm in candidate['arms']:
                        counts[arm+'_candidates']+=1
                        counts[arm+('_feasible' if entry['feasible'] else '_infeasible')]+=1
            audit.append(result)
            if ordinal%50==0:status('auditing',ledger_rows=ordinal+1,candidates=unique)
        reference.revalidate({**bindings,**code});producer.validate_admission(manifest['admission_path'])
        require(reference.runtime_code_bindings()==code,'Native audit loaded an unbound source module')
        with (output/'rows.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for row in audit:stream.write(producer.canonical(row).decode('utf-8')+'\n')
        files={'rows.jsonl':source.sidecar(output/'rows.jsonl')}
        result={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'files':files,'selection_count':859,'audited_contexts':sum(row['status']=='available' for row in contexts),
                'candidate_union_count':unique,'candidate_counts':dict(counts),'unknown_candidates':0,
                'row_order_sha256':producer.value_hash([row['sample_id'] for row in audit]),
                'native_world_patch':'16.19','ddragon_version':dragon.version,'static_data_sha256':dragon.signature,'native_runtime':active,
                'source_manifest_path':str(Path(args.source_manifest).resolve()),'source_manifest_sha256':producer.file_hash(args.source_manifest),
                'source_selection_sha256':manifest['files']['selection.json'],'source_contexts_sha256':manifest['files']['contexts.jsonl'],
                'reference_manifest_path':str(Path(args.reference_manifest).resolve()),'reference_manifest_sha256':producer.file_hash(args.reference_manifest),
                'package_identity_sha256':ref['package_identity_sha256'],'actor_manifest_sha256':ref['actor_manifest_sha256'],
                'opaque_targets_sha256_not_opened':manifest['files']['targets.jsonl'],
                'protocol_path':manifest['protocol_path'],'protocol_sha256':manifest['protocol_sha256'],
                'bindings_sha256':bindings,'code_sha256':code,'candidate_objects_unchanged':True,
                'cost_semantics':'Sum actual native19 purchase_sequence incremental costs at the unchanged causal gold_est wallet; group0 empty no-main costs0.',
                'target_file_opened':False,'targets_analyzed':False,'outcomes_analyzed':False,'raw_payload_reads':False,
                'test_payload_reads':False,'confirmation_payload_reads':False,'training_performed':False,'accuracy_computed':False,
                'policy_model_loaded':False,'candidates_filtered_or_replaced':False,'elapsed_s':time.monotonic()-started,
                'limits':['Legality/cost are conditional on the supplied estimated wallet and reconstructed inventory.',
                          'Audit failures remain in the candidate ledger and fail the gate; they never trigger replacements.']}
        reference.write_json(output/'manifest.json',result);status('complete',candidates=unique);return result
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True,type=Path)
    parser.add_argument('--reference-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deadline-seconds',type=int,default=600)
    result=run(parser.parse_args())
    print(json.dumps({key:result[key] for key in ('audited_contexts','candidate_union_count','candidate_counts','elapsed_s')}))


if __name__=='__main__':main()
