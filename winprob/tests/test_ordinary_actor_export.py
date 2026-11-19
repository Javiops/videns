"""Synthetic CPU checks; no ordinary source payload or trained actor is read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from winprob.iql_features import PublicFeatureCompiler
from winprob.iql_models import FullCategoricalActor
from winprob.ordinary_actor_export import (GroupProjection,canonical_bytes,compiled_equal,digest,
    mutate_ignored_fields,score_contexts,validate_context_ledger,load_contexts,PROTOCOL,SOURCE_SCHEMA)

torch.set_num_threads(1)


def static():
    value={'items':{'1':{'gold':{'total':100},'stats':{'FlatPhysicalDamageMod':1},'tags':['Damage']},
                    '2':{'gold':{'total':200},'stats':{'FlatHPPoolMod':20},'tags':['Health']}},
           'champion_tags':{str(i):[] for i in range(1,11)}}
    return {key:deepcopy(value) for key in ('16.17','16.18','16.19')}


def state():
    return {'team_id':100,'champion_id':1,'inventory':[1,1],'level':5,
        'gold_est_version':'prequential-v3','gold_est':150,'ts':600000,'cs':44,
        'kills':1,'deaths':2,'assists':3,'ally_obj':{'towers':1},'enemy_obj':{},
        'gold':777,'role':'legacy_final_role','total_gold':3000,
        'others':[{'team_id':100 if i<6 else 200,'champion_id':i,'items':[2],
                   'level':5,'gold':1000,'role':'ignored'} for i in range(2,11)]}


def row(identifier,status='available'):
    s=state()
    return {'sample_id':identifier,'match_id':identifier,'participant_id':1,
        'role':'development','patch':'16.19','status':status,'tensor_row':0 if status=='available' else None,
        'prestate':s,'source_projection':deepcopy(s)}


def ledger_fixture():
    entries=[];rows=[];games=[];tensor=0
    for ordinal in range(859):
        mid=f'SYNTHETIC_{ordinal}'
        pid=1+int.from_bytes(hashlib.sha256(('bc20-ordinary-v1/focal/'+mid).encode()).digest()[:8],'big')%10
        selected=ordinal!=0
        choice={'event_index':ordinal,'pid':pid,'ts':ordinal*1000} if selected else None
        sample=f'{mid}|{pid}|{ordinal}|{ordinal*1000}' if selected else f'{mid}|{pid}|none'
        entries.append({'ordinal':ordinal,'sample_id':sample,'match_id':mid,'pid':pid,
                        'selected_decision':choice,'status':'selected' if selected else 'no_admitted_decision'})
        projection={'synthetic':True,'ts':ordinal*1000}
        rows.append({'sample_id':sample,'match_id':mid,'participant_id':pid,'role':'development','patch':'16.19',
            'status':'available' if selected else 'no_admitted_decision','prestate':projection,'source_projection':projection,
            'projection_sha256':hashlib.sha256(canonical_bytes(projection)).hexdigest(),'tensor_row':tensor if selected else None,
            'selected_decision':deepcopy(choice),'context_timestamp_ms':ordinal*1000 if selected else None})
        if selected:tensor+=1
        games.append({'match_id':mid,'role':'development','patch':'16.19'})
    manifest={'schema':SOURCE_SCHEMA,'state':'complete','selection_count':859,'smoke':False,'full_cohort_complete':True}
    return manifest,{'entries':entries},rows,{'games':games}


class OrdinaryActorExportTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(913)

    def test_full_variants_sum_and_finite_logmass_survives_underflow(self):
        project=GroupProjection([0,0,1,2,2],3)
        values=project(torch.tensor([[0.,-1.,-1000.,-1001.,-1002.]],dtype=torch.float32))
        self.assertEqual(values['group_probabilities'][0,1],0.)
        self.assertEqual(values['group_probabilities'][0,2],0.)
        self.assertTrue(np.isfinite(values['group_log_probabilities']).all())
        self.assertGreater(values['group_logits'][0,1],values['group_logits'][0,2])
        np.testing.assert_array_equal(values['group_probabilities'][:,0],values['full_probabilities'][:,:2].sum(1))
        np.testing.assert_allclose(values['group_probabilities'].sum(1),1.,atol=2e-6,rtol=0)

    def test_group_partition_or_nonfinite_logits_fail(self):
        for mapping,count in (([0,0],2),([0,-1],1),([0.,1.],2)):
            with self.assertRaises(ValueError):GroupProjection(mapping,count)
        project=GroupProjection([0,1],2)
        with self.assertRaises(ValueError):project(torch.tensor([[0.,torch.nan]]))

    def test_real_feature_allowlist_ignores_labels_roles_and_privileged_gold(self):
        compiler=PublicFeatureCompiler(static());original=state();before=deepcopy(original)
        changed=mutate_ignored_fields(original)
        self.assertEqual(original,before)
        self.assertTrue(compiled_equal(compiler.state(original,'16.19'),compiler.state(changed,'16.19')))
        changed['gold_est']+=1
        self.assertFalse(compiled_equal(compiler.state(original,'16.19'),compiler.state(changed,'16.19')))

    def test_scoring_preserves_fallback_denominator_and_all_scored_row_mutations(self):
        compiler=PublicFeatureCompiler(static());actor=FullCategoricalActor(10,2,action_count=5)
        rows=[row('good'),row('unknown'),row('missing','unavailable'),row('empty','no_admitted_decision')]
        rows[1]['prestate']['inventory']=[999999]
        ledger,arrays,counts=score_contexts(actor,compiler,rows,[0,0,1,2,2],3)
        self.assertEqual([r['sample_id'] for r in ledger],[r['sample_id'] for r in rows])
        self.assertEqual([r['bc_status'] for r in ledger],['scored','bc_feature_fallback','source_unavailable','no_admitted_decision'])
        self.assertEqual([r['bc_tensor_row'] for r in ledger],[0,None,None,None])
        self.assertEqual(arrays['full_probabilities'].shape,(1,5))
        self.assertEqual(counts['ledger_rows'],4);self.assertEqual(counts['mutation_checked_rows'],1)
        self.assertTrue(ledger[0]['ignored_field_mutation_equal'])
        self.assertEqual(len(ledger[0]['row_sha256']),4)

    def test_public_projection_mismatch_invalidates_instead_of_silent_fallback(self):
        compiler=PublicFeatureCompiler(static());actor=FullCategoricalActor(10,2,action_count=5)
        sample=row('mismatch');sample['source_projection']['gold_est']+=100
        with self.assertRaisesRegex(ValueError,'projection public features differ'):
            score_contexts(actor,compiler,[sample],[0,0,1,2,2],3)

    def test_deliberately_leaky_compiler_fails_mutation_gate(self):
        class Leaky(PublicFeatureCompiler):
            def state(self,source,patch):
                result=super().state(source,patch);result['global_public'][0]=source['gold'];return result
        with self.assertRaisesRegex(ValueError,'forbidden label/future field'):
            score_contexts(FullCategoricalActor(10,2,action_count=5),Leaky(static()),[row('leaky')],[0,0,1,2,2],3)

    def test_exact_859_metadata_ledger_and_focal_choice_are_required(self):
        manifest,selection,rows,admitted=ledger_fixture()
        self.assertEqual(validate_context_ledger(manifest,selection,rows,admitted),858)
        rows[10]['participant_id']=11
        with self.assertRaises(ValueError):validate_context_ledger(manifest,selection,rows,admitted)

    def test_target_in_context_or_changed_projection_is_rejected(self):
        manifest,selection,rows,admitted=ledger_fixture()
        rows[1]['target']={'label_counts':{'1':1}}
        with self.assertRaises(ValueError):validate_context_ledger(manifest,selection,rows,admitted)
        del rows[1]['target'];rows[1]['source_projection']={'changed':True}
        with self.assertRaises(ValueError):validate_context_ledger(manifest,selection,rows,admitted)

    def test_preview_or_postdecision_state_cannot_be_scored_as_full_trial(self):
        manifest,selection,rows,admitted=ledger_fixture()
        manifest['smoke']=True
        with self.assertRaises(ValueError):validate_context_ledger(manifest,selection,rows,admitted)
        manifest['smoke']=False;manifest['full_cohort_complete']=False
        with self.assertRaises(ValueError):validate_context_ledger(manifest,selection,rows,admitted)
        manifest['full_cohort_complete']=True;rows[2]['context_timestamp_ms']+=1
        with self.assertRaises(ValueError):validate_context_ledger(manifest,selection,rows,admitted)

    def test_loader_never_opens_or_hashes_separate_targets(self):
        manifest,selection,rows,admitted=ledger_fixture()
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            def write(name,value):
                path=folder/name;path.write_text(json.dumps(value),encoding='utf-8');return path
            selection_path=write('selection.json',selection)
            context_path=folder/'contexts.jsonl';context_path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            admission_path=write('admission.json',admitted);static_path=write('static.json',{})
            code=folder/'producer.py';code.write_text('# synthetic producer\n',encoding='utf-8')
            manifest.update(files={'selection.json':digest(selection_path),'contexts.jsonl':digest(context_path),'targets.jsonl':'f'*64},
                admission_path=str(admission_path),admission_sha256=digest(admission_path),
                protocol_path=str(PROTOCOL),protocol_sha256=digest(PROTOCOL),static_export_path=str(static_path),static_export_sha256=digest(static_path),
                bindings={str(code):digest(code)},test_payload_reads=False,confirmation_payload_reads=False,outcomes_exported=False,outcomes_analyzed=False)
            manifest_path=write('manifest.json',manifest)
            manifest_path.with_suffix('.sha256').write_text(digest(manifest_path)+' manifest.json\n',encoding='ascii')
            self.assertFalse((folder/'targets.jsonl').exists())
            with patch('winprob.ordinary_actor_export.admission.verify_manifest_sources'):
                loaded,loaded_rows,_,_,available=load_contexts(manifest_path)
            self.assertEqual(available,858);self.assertEqual(len(loaded_rows),859)
            self.assertEqual(loaded['files']['targets.jsonl'],'f'*64)


if __name__=='__main__':unittest.main()
