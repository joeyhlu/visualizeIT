"""Independent validation of the six sealed comparison reports; no inference."""
import copy
import json
from pathlib import Path
import unittest
from tools import workbench_suite_comparison as suite

REPORTS=Path(__file__).resolve().parents[1]/'artifacts/video-workbench/comparisons'

class SuiteComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths=[REPORTS/f"{case['file_prefix']}-full-27e06-vs-baseline-v1.json" for case in suite.SUITE]
        if not all(path.exists() for path in paths):raise unittest.SkipTest('Sealed full-video comparison fixtures unavailable')
        cls.documents=[json.loads(path.read_text()) for path in paths]
        cls.paths=paths

    def row(self,document,index=0):
        return suite.validate_report(document,source_name=self.paths[index].name,
            source_sha256=suite.SUITE[index]['report_sha256'])

    def test_exact_original_suite_and_order(self):
        rows=[self.row(copy.deepcopy(doc),i) for i,doc in enumerate(self.documents)]
        ordered=suite.validate_suite(list(reversed(rows)))
        result=suite.aggregate(ordered)
        self.assertEqual(result['rows_per_role'],1050);self.assertEqual(result['rows_total'],2100)
        self.assertEqual([row['clip_id'] for row in result['clips']],[case['clip_id'] for case in suite.SUITE])
        self.assertFalse(result['claims']['quality_improvement'])

    def test_substituted_run_rejected_even_with_updated_paths(self):
        doc=copy.deepcopy(self.documents[0]);run=doc['runs'][0];old=run['run_id']
        run['run_id']='run-20261021T000000-deadbeef'
        for item in run['input_files']:item['path']=item['path'].replace(old,run['run_id'])
        with self.assertRaises(suite.SuiteError):self.row(doc)

    def test_actual_source_sha_is_mandatory_and_exact(self):
        with self.assertRaises(suite.SuiteError):
            suite.validate_report(copy.deepcopy(self.documents[0]),source_sha256='0'*64)
        rows=[self.row(copy.deepcopy(doc),i) for i,doc in enumerate(self.documents)]
        rows[0]['source_report']['sha256']=None
        with self.assertRaises(suite.SuiteError):suite.validate_suite(rows,require_source_seals=True)

    def test_wrong_primary_or_baseline_weights_rejected(self):
        for role,key in [('primary_neural','installed_head_sha256'),('frozen_baseline','base_checkpoint_sha256')]:
            doc=copy.deepcopy(self.documents[0]);doc['runs'][0]['roles'][role]['identity'][key]='0'*64
            with self.assertRaises(suite.SuiteError):self.row(doc)

    def test_partial_window_and_missing_rows_rejected(self):
        for field,value in [('scope','partial'),('scheduled_frames',239),('run_completeness','partial')]:
            doc=copy.deepcopy(self.documents[0]);doc['runs'][0][field]=value
            with self.assertRaises(suite.SuiteError):self.row(doc)
        doc=copy.deepcopy(self.documents[0]);doc['runs'][0]['roles']['primary_neural']['missing_rows']=1
        with self.assertRaises(suite.SuiteError):self.row(doc)

    def test_duplicate_or_missing_clip_rejected(self):
        rows=[self.row(copy.deepcopy(doc),i) for i,doc in enumerate(self.documents)]
        for selected in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(suite.SuiteError):suite.validate_suite(selected)

    def test_accuracy_claim_and_cached_latency_rejected(self):
        doc=copy.deepcopy(self.documents[0]);doc['claims']['accuracy']=True
        with self.assertRaises(suite.SuiteError):self.row(doc)
        doc=copy.deepcopy(self.documents[0]);doc['runs'][0]['roles']['primary_neural']['latency']['source']='cached_original_run'
        with self.assertRaises(suite.SuiteError):self.row(doc)

    def test_duplicate_json_keys_and_nonfinite_numbers_rejected(self):
        for raw in [b'{"a":1,"a":2}',b'{"a":NaN}']:
            with self.assertRaises(suite.SuiteError):suite.loads_strict(raw,'fixture')

if __name__=='__main__':unittest.main()
