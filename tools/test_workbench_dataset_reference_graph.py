"""Codex checks for reporting the sealed dataset diagnostic, without inference."""
import copy,json
from pathlib import Path
import unittest
from tools import workbench_dataset_reference_graph as graph

SOURCE=graph.ROOT/'.cache/video-workbench/derived/dataset-reference/eval-v1/dataset-reference-projection.json'

@unittest.skipUnless(SOURCE.is_file(),'sealed local diagnostic unavailable')
class DatasetGraphTests(unittest.TestCase):
    def setUp(self):self.doc=json.loads(SOURCE.read_bytes())
    def test_exact_six_rows_preserve_values_and_denominators(self):
        rows=graph.build_rows(graph.validate_document(self.doc,source_sha256=graph.INPUT_SHA256))
        self.assertEqual(len(rows),6)
        for row in rows:
            source=self.doc['objects'][str(row['object_id'])]['roles'][row['role']]['A_anchor']
            self.assertEqual(row['anchor_median_px720'],source['median_px720'])
            self.assertEqual(row['anchor_reference_eligible'],110)
            self.assertEqual(row['anchor_errors_measured'],109)
            self.assertEqual(row['human_accuracy'],'unverified')
    def test_wrong_hash_and_changed_source_rejected(self):
        with self.assertRaises(graph.GraphError):graph.read_input(SOURCE,'0'*64)
        with self.assertRaises(graph.GraphError):graph.validate_document(self.doc,source_sha256='0'*64)
    def test_human_or_phone_promotion_and_different_head_rejected(self):
        for field in ['independent_human_accuracy','phone_attachment_verified']:
            changed=copy.deepcopy(self.doc);changed['flags'][field]=True
            with self.assertRaises(graph.GraphError):graph.validate_document(changed)
        self.doc['pins']['head_sha256']='0'*64
        with self.assertRaises(graph.GraphError):graph.validate_document(self.doc)
    def test_partial_run_missing_row_and_changed_closure_rejected(self):
        mutations=[lambda d:d['objects'].pop('2'),
                   lambda d:d['objects']['2']['roles']['primary_neural']['A_anchor_rows'].pop(),
                   lambda d:d['hash_closure'].update(equal=False),
                   lambda d:d['objects']['2'].update(run_id='substitute')]
        for mutate in mutations:
            changed=copy.deepcopy(self.doc);mutate(changed)
            with self.assertRaises(graph.GraphError):graph.validate_document(changed)
    def test_wrong_recomputed_quantile_and_nan_rejected(self):
        for wrong in [0.,float('nan')]:
            changed=copy.deepcopy(self.doc)
            changed['objects']['2']['roles']['primary_neural']['A_anchor']['median_px720']=wrong
            with self.assertRaises(graph.GraphError):graph.validate_document(changed)
    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(graph.GraphError):graph.loads_strict(b'{"x":1,"x":2}')

if __name__=='__main__':unittest.main()
