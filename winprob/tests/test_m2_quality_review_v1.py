from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from winprob.m2_quality_review_v1 import strict_json,validate_and_summarize


def fixture():
    cases={'schema':'m2-purchase-quality-blind-cases-v1','version':'synthetic-test-only',
        'cases':[{'case_id':'C01','options':[{'label':label} for label in 'ABC']},{'case_id':'C02','options':[{'label':label} for label in 'AB']}]}
    review={'schema':'m2-purchase-quality-user-review-v1','version':'synthetic-test-only','cases_sha256':'synthetic-digest','submitted':True,
        'answers':{'C01':{'ratings':{'A':'preferred','B':'acceptable','C':'insufficient_information'},'reason':'Synthetic fixture, not a user judgement.'}}}
    return cases,review


class QualityReviewTests(unittest.TestCase):
    def test_unknown_and_unrated_choices_create_no_fake_preferences(self):
        cases,review=fixture();value=validate_and_summarize(review,cases,'synthetic-digest')
        first,second=value['cases'];self.assertEqual(first['strict_preferences_within_declared_rubric'],[{'preferred_label':'A','less_preferred_label':'B'}])
        self.assertEqual(first['insufficient_information_labels'],['C']);self.assertEqual(second['unrated_labels'],['A','B'])
        self.assertEqual(second['strict_preferences_within_declared_rubric'],[]);self.assertFalse(value['performance_improvement_measured'])

    def test_ties_are_retained_without_selecting_a_unique_winner(self):
        cases,review=fixture();review['answers']['C01']['ratings']={'A':'preferred','B':'preferred','C':'acceptable'}
        row=validate_and_summarize(review,cases,'synthetic-digest')['cases'][0]
        self.assertEqual(row['same_grade_pairs'],[['A','B']]);self.assertEqual(len(row['strict_preferences_within_declared_rubric']),2)

    def test_draft_and_boolean_impostor_are_rejected(self):
        cases,review=fixture()
        for flag in (False,1,'true'):
            changed=deepcopy(review);changed['submitted']=flag
            with self.assertRaisesRegex(ValueError,'explicit submission'):validate_and_summarize(changed,cases,'synthetic-digest')

    def test_wrong_version_unknown_case_or_option_cannot_be_joined(self):
        cases,review=fixture();changed=deepcopy(review);changed['cases_sha256']='another-digest'
        with self.assertRaisesRegex(ValueError,'another case version'):validate_and_summarize(changed,cases,'synthetic-digest')
        changed=deepcopy(review);changed['answers']['C99']=changed['answers']['C01']
        with self.assertRaisesRegex(ValueError,'Unknown case'):validate_and_summarize(changed,cases,'synthetic-digest')
        changed=deepcopy(review);changed['answers']['C01']['ratings']['Z']='preferred'
        with self.assertRaisesRegex(ValueError,'Unknown alternative'):validate_and_summarize(changed,cases,'synthetic-digest')

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):strict_json('{"submitted":false,"submitted":true}')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):strict_json('{"value":NaN}')

    def test_unblinding_fields_are_not_accepted_as_ratings(self):
        cases,review=fixture();review['answers']['C01']['original_m2_rank']=0
        with self.assertRaisesRegex(ValueError,'Unexpected case-answer'):validate_and_summarize(review,cases,'synthetic-digest')

    def test_context_note_does_not_fabricate_an_option_grade(self):
        cases,review=fixture();review['answers']['C01']={'ratings':{},'reason':'Missing tactical objective (synthetic fixture).'}
        value=validate_and_summarize(review,cases,'synthetic-digest')
        self.assertEqual(value['graded_options'],0);self.assertEqual(value['cases_with_context_notes'],1)
        self.assertFalse(value['source_origin_authenticated_by_parser'])

    def test_ingestion_preserves_submitted_bytes_and_refuses_overwrite(self):
        source=Path(__file__).resolve().parents[1]/'scripts/ingest_m2_quality_review_v1.py'
        spec=importlib.util.spec_from_file_location('synthetic_review_ingestion',source)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        cases,review=fixture()
        with tempfile.TemporaryDirectory(prefix='m2-review-synthetic-test-') as directory:
            temp=Path(directory).resolve()
            self.assertTrue(temp.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            case_file=temp/'cases.json';case_file.write_text(json.dumps(cases),encoding='utf-8')
            digest=hashlib.sha256(case_file.read_bytes()).hexdigest()
            case_file.with_suffix('.sha256').write_text(digest+' cases.json\n',encoding='ascii')
            review['cases_sha256']=digest
            raw=b'\xef\xbb\xbf'+json.dumps(review,indent=2).replace('\n','\r\n').encode()
            review_file=temp/'review.json';review_file.write_bytes(raw);output=temp/'receipt'
            result=cli.run(review_file,case_file,output,'Synthetic unit test, not a human review.')
            self.assertEqual((output/'submitted_review.json').read_bytes(),raw)
            self.assertEqual(result['source_review_sha256'],hashlib.sha256(raw).hexdigest())
            with self.assertRaises(FileExistsError):cli.run(review_file,case_file,output,'Synthetic unit test only.')


if __name__=='__main__':unittest.main()
