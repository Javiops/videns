"""Synthetic exact expanded-census contract tests; no real data payloads."""
from copy import deepcopy
import unittest

import export_winprob_revision_contexts_v3 as exporter


class ExpandedCensusTests(unittest.TestCase):
    def setUp(self):
        self.report={'schema':'undo-pairs-support-v2','stage':'outcome_blind_expanded_counts',
                     'outcome_targets_extracted':False,'cache_y_accessed':False,
                     'certificate_changed_from_v1':False,'certificate_sha256':exporter.CERTIFICATE_SHA256}
        self.source={'games':[{'match_id':f'FIT_{i}','role':'fit','patch':'16.17'} for i in range(4002)]
                      +[{'match_id':f'CAL_{i}','role':'calibration','patch':'16.18'} for i in range(1000)]
                      +[{'match_id':f'DEV_{i}','role':'development','patch':'16.19'} for i in range(859)]}
        self.original={'games':self.source['games'][:5002]+[{'match_id':'OLDDEV_1','role':'development','patch':'16.18'}]}
        self.manifest={'games':[{'match_id':f'DEV_{i}','role':'development','patch':'16.19'} for i in range(859)]}
    def test_exact_expanded_census_accepted(self):
        exporter.validate_expanded_claims(self.report,self.source,self.manifest,self.original)
    def test_old_schema_or_changed_certificate_rejected(self):
        for update in ({'schema':'undo-pairs-support-v1'},{'stage':'outcome_blind_counts'},
                       {'certificate_changed_from_v1':True},{'certificate_sha256':'changed'}):
            with self.assertRaises(ValueError):
                exporter.validate_expanded_claims({**self.report,**update},self.source,self.manifest,self.original)
    def test_same_counts_but_changed_fit_membership_rejected(self):
        source=deepcopy(self.source);source['games'][0]['match_id']='DIFFERENT_1'
        with self.assertRaisesRegex(ValueError,'fit membership'):
            exporter.validate_expanded_claims(self.report,source,self.manifest,self.original)
    def test_development_cannot_include_another_patch_or_changed_id(self):
        source=deepcopy(self.source);source['games'][-1]['patch']='16.18'
        with self.assertRaisesRegex(ValueError,'all859'):
            exporter.validate_expanded_claims(self.report,source,self.manifest,self.original)
        source=deepcopy(self.source);source['games'][-1]['match_id']='NEW_1'
        with self.assertRaisesRegex(ValueError,'all859'):
            exporter.validate_expanded_claims(self.report,source,self.manifest,self.original)


if __name__=='__main__':unittest.main()
