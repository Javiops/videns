"""Synthetic admission and diagnostic-export tests; no actual game payloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import export_macro_request as request
import score_macro_curves as curves
from winprob import admission
from test_cache import payloads


class FakePredictor:
    kind='snapshot'
    indices=list(range(17))
    def predict(self,x):
        self.x=x.copy()
        return np.linspace(.4,.7,len(x))


class MacroCurveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        data=self.root/'data'
        data.mkdir()
        split=data/'split_assignment.json'
        split.write_text(json.dumps({'EUW1_1':'train','EUW1_2':'train','EUW1_3':'train','EUW1_4':'test'}))
        manifest=admission.prepare_manifest(data,[split],'synthetic-macro-v1')
        for row in manifest['games']:
            row['role']='confirmatory_reserved' if row['match_id']=='EUW1_2' else 'train'
        self.admitted,_=admission.write_manifest(manifest,self.root/'admission')
        self.split=self.root/'macro-split.json'
        self.split.write_text(json.dumps({'train':['EUW1_1','EUW1_2'],'val':['EUW1_3','EUW1_4'],
                                         'test':['EUW1_5'],'n_games':5}))

    def export(self):
        return request.export_request(self.split,self.admitted,self.root/'request')

    def test_tests_and_prior_confirmation_removed_before_any_payload(self):
        path,payload=self.export()
        self.assertEqual(payload['requested_game_ids'],['EUW1_1','EUW1_3'])
        self.assertEqual({row['match_id'] for row in payload['excluded']},{'EUW1_2','EUW1_4'})
        self.assertFalse(payload['raw_payload_reads'])
        loaded,_,_=request.load_request(path)
        self.assertEqual(loaded,payload)

    def test_injected_identity_rejected_even_with_updated_request_sidecar(self):
        path,payload=self.export()
        payload['games'].append({'match_id':'EUW1_5','macro_role':'val'})
        payload['requested_game_ids'].append('EUW1_5')
        path.write_bytes(admission.canonical_bytes(payload))
        path.with_suffix('.sha256').write_text(request.digest(path)+'  request.json\n')
        with self.assertRaisesRegex(ValueError,'Requested identities differ'):
            request.load_request(path)

    def test_split_change_invalidates_request(self):
        path,_=self.export()
        self.split.write_bytes(self.split.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Macro split changed'):
            request.load_request(path)

    def test_refuses_request_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_missing_archives_export_explicit_unavailable_without_reads(self):
        path,payload=self.export()
        model=self.root/'fake-model.joblib'
        model.write_bytes(b'synthetic-model-placeholder')
        evidence={'state':'synthetic','description':'test only','report_path':'synthetic',
                  'report_sha256':'synthetic'}
        with patch.object(curves,'verify_judge',return_value=(model,request.digest(model),{},evidence,{})), \
                patch.object(curves.joblib,'load',return_value=FakePredictor()), \
                patch.object(curves.cache,'_read_payload') as reader:
            output=self.root/'curves'
            metadata=curves.run(path,self.root,output)
            reader.assert_not_called()
        self.assertEqual(metadata['counts']['status'],{'unavailable':2})
        self.assertEqual(metadata['requested_game_ids'],payload['requested_game_ids'])
        rows=[json.loads(line) for line in (output/'curves.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['unavailable_reason']=='missing_archive' and row['points']==[] for row in rows))
        self.assertEqual((output/'curves.sha256').read_text().split(),[request.digest(output/'curves.jsonl'),'curves.jsonl'])
        self.assertEqual((output/'meta.sha256').read_text().split(),[request.digest(output/'meta.json'),'meta.json'])

    def test_score_exports_no_labels_and_predictor_gets_only_features(self):
        match,timeline=payloads('EUW1_1',1000000)
        predictor=FakePredictor()
        row=curves.score_payloads('EUW1_1',match,timeline,predictor,'digest','unseen',{})
        self.assertEqual(row['status'],'available')
        self.assertEqual(predictor.x.shape,(2,19))
        self.assertNotIn('y',row)
        self.assertNotIn('win',row)
        self.assertEqual(set(row['points'][0]),{'timestamp_ms','p_blue'})
        self.assertTrue(all(point['timestamp_ms']<300000 for point in row['points']))

    def test_windows_observed_endpoints_and_delta_without_interpolation(self):
        points=[{'timestamp_ms':180005,'p_blue':.4},{'timestamp_ms':240003,'p_blue':.5},
                {'timestamp_ms':360001,'p_blue':.7}]
        windows,missing=curves.observed_windows(points)
        self.assertEqual(len(windows),1)
        self.assertEqual(windows[0]['start_ms'],180005)
        self.assertEqual(windows[0]['end_ms'],360001)
        self.assertAlmostEqual(windows[0]['delta_blue_pp'],30)
        self.assertEqual(missing,2)

    def test_invalid_probability_and_wrong_queue_fail(self):
        match,timeline=payloads('EUW1_1',1000000)
        class Bad:
            def predict(self,x): return np.full(len(x),1.2)
        with self.assertRaisesRegex(ValueError,'invalid probabilities'):
            curves.score_payloads('EUW1_1',match,timeline,Bad(),'digest','unseen',{})
        match['info']['queueId']=450
        with self.assertRaisesRegex(ValueError,'ranked solo'):
            curves.score_payloads('EUW1_1',match,timeline,FakePredictor(),'digest','unseen',{})


if __name__=='__main__':
    unittest.main()
