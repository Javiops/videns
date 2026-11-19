"""Synthetic-only coverage of context boundaries, exact decoding and native audit."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import export_bc20_ordinary_reference_v1 as ref
import audit_bc20_ordinary_native19_v1 as audit


GROUPS=[[],[[1001,1]],[[2003,2]]]


def option(group):
    return {'group':group,'kind':'save' if not group else 'buy',
            'items':{str(item):count for item,count in GROUPS[group]},'legal':True}


class FakePolicy:
    def __init__(self):self.calls=0
    def probabilities(self,rows):
        self.calls+=1
        return np.tile(np.array([.1,.7,.2],dtype=np.float32),(len(rows),1))
    def options(self,rows):
        return [[option(int(group)) for group in np.argsort(-p)] for p in self.probabilities(rows)]


def fixture():
    entries=[];rows=[]
    for ordinal in range(859):
        mid=f'TEST_{ordinal}';pid=ref.source.focal_player(mid)
        choice={'event_index':7,'pid':pid,'ts':60000}
        sid=f'{mid}|{pid}|7|60000'
        entries.append({'ordinal':ordinal,'match_id':mid,'pid':pid,'sample_id':sid,'selected_decision':choice})
        projection={'ts':60000,'gold_est':450.0,'inventory':[1001],'others':[]}
        rows.append({'ordinal':ordinal,'sample_id':sid,'match_id':mid,'participant_id':pid,'role':'development',
            'patch':'16.19','status':'available','selected_decision':choice,'tensor_row':ordinal,
            'context_timestamp_ms':60000,'prestate':deepcopy(projection),'source_projection':projection,
            'projection_sha256':ref.producer.value_hash(projection),
            'checks':{key:True for key in ('selected_identity_matches','prefix_wallet_reproduced','target_future_projection_mutation_equal')}})
    return {'available_contexts':859},{'entries':entries},rows


class ReferenceTests(unittest.TestCase):
    def test_capture_is_original_single_forward_and_override_preserves_objects(self):
        policy=FakePolicy();rows=[{'gold_est':500}]
        original=policy.probabilities
        probabilities,options=ref.capture_native(policy,rows)
        self.assertEqual(policy.calls,1)
        self.assertEqual(policy.probabilities,original)
        self.assertEqual(ref.options_override(policy,rows,probabilities),options)
        self.assertEqual(policy.calls,1)
        direct=ref.options_override(policy,rows,np.array([[.8,.1,.1]],dtype=np.float32))
        self.assertEqual(direct[0][0]['group'],0)
        self.assertEqual(policy.calls,1)
        self.assertEqual(policy.probabilities,original)

    def test_override_restores_even_on_decoder_exception(self):
        policy=FakePolicy();original=policy.probabilities
        with patch.object(policy,'options',side_effect=RuntimeError('decoder')):
            with self.assertRaises(RuntimeError):ref.options_override(policy,[],np.empty((0,3)))
        self.assertEqual(policy.probabilities,original)

    def test_capture_rejects_extra_forward(self):
        class Bad(FakePolicy):
            def options(self,rows):
                self.probabilities(rows);return super().options(rows)
        with self.assertRaisesRegex(ValueError,'exactly one'):ref.capture_native(Bad(),[{}])

    def test_override_rejects_changed_row_batch(self):
        class Bad(FakePolicy):
            def options(self,rows):return super().options(list(rows))
        with self.assertRaisesRegex(ValueError,'batch'):ref.options_override(Bad(),[{}],np.ones((1,3)))

    def test_probability_guard_never_renormalizes(self):
        p=np.array([[.1,.7,.2]],dtype=np.float32);saved=p.tobytes()
        ref.validate_probabilities(p,1,3);self.assertEqual(p.tobytes(),saved)
        for invalid in (p.astype(np.float64),p*1.00001,np.array([[.1,np.nan,.9]],dtype=np.float32),np.array([[-.1,.5,.6]],dtype=np.float32)):
            with self.assertRaises(ValueError):ref.validate_probabilities(invalid,1,3)

    def test_model_placeholder_never_uses_target_and_mutation_preserves_inputs(self):
        _,_,rows=fixture();context=rows[0]
        context['target']={'label_counts':{'3006':1}}
        model=ref.model_row(context);self.assertEqual(model['label_counts'],{'999999':1})
        changed=ref.mutate_ignored(model)
        for key in context['source_projection']:self.assertEqual(changed[key],model[key])
        self.assertNotEqual(changed['label_counts'],model['label_counts'])

    def test_context_rows_reject_retiming_swaps_targets_and_dropped_rows(self):
        manifest,selection,rows=fixture();self.assertEqual(ref.validate_context_rows(manifest,selection,rows),859)
        for mutation in ('time','target','order','index'):
            altered=deepcopy(rows)
            if mutation=='time':altered[0]['context_timestamp_ms']+=1
            if mutation=='target':altered[0]['target']={'item':1001}
            if mutation=='order':altered[0],altered[1]=altered[1],altered[0]
            if mutation=='index':altered[0]['tensor_row']=True
            with self.assertRaises(ValueError):ref.validate_context_rows(manifest,selection,altered)
        with self.assertRaises(ValueError):ref.validate_context_rows(manifest,selection,rows[:-1])

    def test_actor_rows_require_complete_byte_bound_tensor_and_mutation_coverage(self):
        _,_,rows=fixture();p=np.zeros((859,9333),dtype=np.float32);p[:,0]=1
        actor=[{**{key:row[key] for key in ref.IDENTITY_KEYS},'source_tensor_row':i,'bc_status':'scored',
                'bc_tensor_row':i,'ignored_field_mutation_equal':True,'row_sha256':{'group_probabilities':ref.array_hash(p[i])}}
               for i,row in enumerate(rows)]
        self.assertEqual(ref.validate_actor_rows(rows,actor,p),859)
        for field,value in (('bc_tensor_row',1),('ignored_field_mutation_equal',False),('match_id','WRONG')):
            changed=deepcopy(actor);changed[0][field]=value
            with self.assertRaises(ValueError):ref.validate_actor_rows(rows,changed,p)
        p[0,0]=.5
        with self.assertRaisesRegex(ValueError,'digest'):ref.validate_actor_rows(rows,actor,p)

    def test_bad_group_objects_rejected_including_multiplicity(self):
        ref.validate_options([option(0),option(2)],GROUPS)
        for options in ([option(1),option(1)],[],[{**option(2),'items':{'2003':1}}],[{**option(0),'kind':'buy'}]):
            with self.assertRaises(ValueError):ref.validate_options(options,GROUPS)

    def test_incremental_cost_uses_native_sequence_and_preserves_copies(self):
        calls=[]
        def purchase(action,row,wallet,dragon):
            calls.append((action,wallet));return [{'item_id':item,'cost':25} for item in action]
        result=ref.audit_options([option(0),option(2)],{'gold_est':100},GROUPS,purchase,None)
        self.assertEqual(calls,[((2003,2003),100.0)])
        self.assertEqual(result[0]['incremental_cost'],0)
        self.assertEqual(result[1]['incremental_cost'],50)
        self.assertEqual(result[1]['option_sha256'],ref.producer.value_hash(option(2)))
        impossible=ref.audit_options([option(1)],{'gold_est':10},GROUPS,lambda *a:None,None)[0]
        self.assertFalse(impossible['feasible']);self.assertIsNone(impossible['incremental_cost'])
        with self.assertRaisesRegex(ValueError,'wallet'):
            ref.audit_options([option(2)],{'gold_est':49},GROUPS,purchase,None)

    def test_candidate_union_never_filters_and_retains_both_arm_ranks(self):
        row={'m2_options':[option(1),option(2),option(0)],'direct_bc_options':[option(0),option(1)]}
        saved=deepcopy(row);union=audit.candidate_union(row,GROUPS)
        self.assertEqual(row,saved);self.assertEqual(len(union),3)
        self.assertEqual(union[0]['arms'],{'m2':0,'direct_bc':1})
        self.assertEqual(union[2]['arms'],{'m2':2,'direct_bc':0})

    def test_metadata_guard_forbids_target_even_for_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'targets.jsonl';target.write_text('NEVER READ')
            with patch.object(ref.producer,'file_hash',side_effect=AssertionError('attempted payload hash')):
                with self.assertRaisesRegex(ValueError,'Forbidden'):ref.metadata_bindings({str(target):'a'*64})

    def test_context_loader_does_not_open_or_hash_separate_targets(self):
        base,selection,rows=fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            metadata=directory/'metadata.json';metadata.write_text('{}')
            digest=ref.producer.file_hash(metadata)
            selection.update(admitted_ids=[row['match_id'] for row in rows],admission_sha256=digest,protocol_sha256=digest)
            ref.write_json(directory/'selection.json',selection)
            with (directory/'contexts.jsonl').open('w',encoding='utf-8') as stream:
                for row in rows:stream.write(json.dumps(row)+'\n')
            # targets.jsonl deliberately does not exist: only its opaque hash is available.
            manifest={**base,'schema':ref.source.CONTEXT_SCHEMA,'state':'complete','smoke':False,'full_cohort_complete':True,
                      'selection_count':859,'files':{'selection.json':ref.producer.file_hash(directory/'selection.json'),
                      'contexts.jsonl':ref.producer.file_hash(directory/'contexts.jsonl'),'targets.jsonl':'a'*64},
                      'bindings':{str(metadata):digest}}
            for flag in ('test_payload_reads','confirmation_payload_reads','outcomes_exported','outcomes_analyzed','training_performed'):manifest[flag]=False
            for key in ('admission','protocol','static_export','catalogue'):
                manifest[key+'_path']=str(metadata);manifest[key+'_sha256']=digest
            ref.write_json(directory/'manifest.json',manifest)
            with patch.object(ref.source,'load_selection',return_value=(selection,{}, {},set())):
                loaded=ref.load_contexts(directory/'manifest.json')
            self.assertEqual(len(loaded[1]),859)
            self.assertFalse((directory/'targets.jsonl').exists())


if __name__=='__main__':unittest.main()
