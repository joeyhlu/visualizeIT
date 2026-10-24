"""Codex verification of the source-bound completed diagnostic graph."""
import copy
import csv
import io
import unittest
from unittest import mock

from tools import workbench_mask_replay_graph as graph


class ReplayGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.sha = graph.load_source()

    def test_actual_four_cells_and_failure_denominators(self):
        points = graph.extract_points(self.doc, source_sha256=self.sha)
        self.assertEqual(len(points), 4)
        self.assertEqual([p['denominator'] for p in points], [110]*4)
        self.assertEqual([p['measured'] for p in points], [109]*4)
        self.assertEqual([p['artwork_rendered'] for p in points], [54,54,109,109])
        self.assertEqual([p['coverage_le_5_count'] for p in points], [26,26,39,39])
        self.assertEqual([p['coverage_le_10_count'] for p in points], [101,100,108,108])
        self.assertAlmostEqual(points[2]['conditional_median_px720'], 5.5846402217, places=8)

    def test_wrong_hash_rejected_before_json_parse(self):
        with mock.patch.object(graph.Path, 'read_bytes', return_value=b'not JSON'), \
             mock.patch.object(graph, '_loads_strict', side_effect=AssertionError('parsed')):
            with self.assertRaisesRegex(graph.GraphError, 'SHA-256'):
                graph.load_source()

    def test_claim_and_control_mutations_rejected(self):
        for flag in ('phone_quality', 'independent_human_accuracy', 'weights_changed', 'significance_tested'):
            doc = copy.deepcopy(self.doc)
            doc['flags'][flag] = True
            with self.subTest(flag=flag), self.assertRaises(graph.GraphError):
                graph.extract_points(doc)
        doc = copy.deepcopy(self.doc)
        doc['original_control']['passed'] = False
        with self.assertRaises(graph.GraphError):
            graph.extract_points(doc)

    def test_count_quantile_and_missing_frame_mutations_rejected(self):
        key = 'new_qualified_mask/primary_neural'
        for field, val in [('denominator',109), ('conditional_median_px720',0.0), ('coverage_le_10px720',1.0)]:
            doc = copy.deepcopy(self.doc)
            doc['cells'][key]['all_reference_eligible'][field] = val
            with self.subTest(field=field), self.assertRaises(graph.GraphError):
                graph.extract_points(doc)
        doc = copy.deepcopy(self.doc)
        row = next(r for r in doc['rows']['new']['primary_neural'] if r['error_px720'] is None)
        row['error_px720'] = 0
        with self.assertRaises(graph.GraphError):
            graph.extract_points(doc)

    def test_duplicate_frame_rejected(self):
        doc = copy.deepcopy(self.doc)
        rows = doc['rows']['original']['frozen_baseline']
        rows[1]['frame_id'] = rows[0]['frame_id']
        with self.assertRaises(graph.GraphError):
            graph.extract_points(doc)

    def test_deterministic_four_row_exports_and_no_readiness_claim(self):
        points = graph.extract_points(self.doc)
        record = graph.provenance(self.doc, self.sha, points)
        self.assertEqual(graph.json_bytes(record), graph.json_bytes(record))
        self.assertEqual(graph.csv_bytes(points), graph.csv_bytes(points))
        rows = list(csv.DictReader(io.StringIO(graph.csv_bytes(points).decode())))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['denominator'] == '110' for r in rows))
        self.assertTrue(all(v is False for v in record['claims'].values()))

    def test_existing_prefix_rejected_before_render(self):
        with mock.patch.object(graph.os.path, 'lexists', return_value=True), \
             mock.patch.object(graph, 'render_png', side_effect=AssertionError('rendered')):
            with self.assertRaisesRegex(graph.GraphError, 'already exists'):
                graph.build(graph.DEFAULT_INPUT, 'artifacts/video-workbench/progress/existing')

    def test_strict_json_rejects_nonfinite_and_duplicate_keys(self):
        for raw in (b'{"x":NaN}', b'{"x":1,"x":2}'):
            with self.subTest(raw=raw), self.assertRaises(graph.GraphError):
                graph._loads_strict(raw)


if __name__ == '__main__':
    unittest.main()
