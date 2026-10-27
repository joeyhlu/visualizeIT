"""Independent source-bound descriptive mask audit controls; no accuracy labels."""
import builtins
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_cli as cli
from bench import workbench_contract as wc
from bench import workbench_geometry as geom
from bench import workbench_mask_diagnostics as diagnostics
from tools.test_workbench_offline_export_integration import make_fixture, cv2


class IndependentMaskAuditChecks(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/'.cache'
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache, prefix='mask-audit-independent-')
        self.addCleanup(self.temp.cleanup)
        self.f = make_fixture(Path(self.temp.name))
        self.ws = self.f['workspace']
        self.clip_id = 'synthetic-offline-clip'
        self.mask_dir = self.ws.masks_dir/self.clip_id/'synthetic'
        self.out = self.ws.derived_dir/'mask-audit'

    def masks(self, sequence):
        manifest_path = self.mask_dir/'manifest.json'
        doc = wc.loads_strict(manifest_path.read_bytes())
        for index, dimensions in enumerate(sequence, 2):
            fid = f'f{index}'
            if dimensions is None:
                doc['masks'].pop(fid, None)
                continue
            mask = np.zeros((96, 128), np.uint8)
            height, width = dimensions
            mask[20:20+height, 20:20+width] = 255
            ok, encoded = cv2.imencode('.png', mask); self.assertTrue(ok)
            raw = encoded.tobytes()
            (self.mask_dir/f'{fid}.png').write_bytes(raw)
            doc['masks'][fid] = dict(path=f'{fid}.png', sha256=hashlib.sha256(raw).hexdigest(),
                content_sha256=geom.mask_sha256(mask > 127), empty=not bool(mask.any()),
                width=128, height=96)
        manifest_path.write_bytes(wc.canonical_json(doc))

    def audit(self):
        return diagnostics.audit_saved_masks(self.ws, clip_id=self.clip_id,
            mask_set='synthetic', output_dir=self.out, area_ratio_min=.75)

    def test_analytical_rectangles_and_explicit_missing_empty_denominators(self):
        self.masks([(4, 4), (4, 2), (2, 2), None, (0, 0), (4, 4)])
        original_import = builtins.__import__
        def no_model(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'torchvision', 'gotrack', 'sam2'):
                raise AssertionError('No model dependency: '+name)
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=no_model), \
             patch.object(cli.FrameReader, '__init__', side_effect=AssertionError('No RGB reader')):
            report = self.audit()
        rows = report['rows']
        self.assertEqual(report['frame_accounting'],
            dict(scheduled=6, observed=5, unavailable=1, empty=1, nonempty=4))
        # The first rectangle occupies x20..23,y20..23: exact area16,
        # exclusive bounds and arithmetic pixel-center centroid21.5.
        self.assertEqual((rows[0]['area_px'], rows[0]['bbox_xyxy'], rows[0]['centroid_xy']),
                         (16, [20, 20, 24, 24], [21.5, 21.5]))
        self.assertEqual(rows[1]['temporal_iou'], .5)
        self.assertEqual(rows[1]['area_ratio'], .5)
        self.assertEqual(rows[1]['centroid_distance_px'], 1.)
        self.assertEqual(rows[1]['centroid_distance_px720'], 7.5)
        self.assertEqual(rows[2]['baseline_area_ratio'], .25)
        self.assertIsNone(rows[3]['area_px'])
        self.assertEqual(rows[4]['area_px'], 0)
        self.assertIsNone(rows[4]['centroid_xy'])
        self.assertEqual(rows[5]['temporal_iou'], 0)
        self.assertIsNone(rows[5]['area_ratio'])
        self.assertEqual([e['frame_ids'] for e in report['episodes']], [['f3', 'f4'], ['f6']])
        self.assertTrue(all(e['timestamp_span_s'] is None for e in report['episodes']))
        self.assertIsNone(report['independent_accuracy'])
        self.assertFalse(report['quality_claim'])
        csv_rows = list(csv.DictReader(io.StringIO((self.out/'report.csv').read_text())))
        self.assertEqual([r['frame_id'] for r in csv_rows], report['scheduled_frame_ids'])
        self.assertEqual(csv_rows[3]['area_px'], '')
        self.assertEqual(csv_rows[4]['area_px'], '0')

    def test_source_gap_splits_episodes_and_never_creates_pair_metrics(self):
        self.masks([(4, 4), (4, 2), (2, 2), None, (0, 0), (4, 4)])
        path = self.ws.clip_path(self.clip_id)
        doc = wc.loads_strict(path.read_bytes())
        for frame in doc['frames']:
            if frame['source_frame_index'] >= 4:
                frame['source_frame_index'] += 2
        path.write_bytes(wc.canonical_json(doc))
        report = self.audit()
        row = report['rows'][2]
        self.assertEqual(row['pair_status'], 'source_gap')
        self.assertIsNone(row['temporal_iou'])
        self.assertIsNone(row['area_ratio'])
        self.assertEqual([e['frame_ids'] for e in report['episodes']], [['f3'], ['f4'], ['f6']])

    def test_empty_opening_baseline_is_not_backfilled_from_later_observation(self):
        self.masks([(0, 0), None, (4, 4), (2, 2), (0, 0), (4, 4)])
        report = self.audit()
        self.assertEqual(report['baseline'], dict(frame_id='f4', area_px=16))
        self.assertTrue(all(r['baseline_frame_id'] is None for r in report['rows'][:2]))
        self.assertTrue(all(r['baseline_area_ratio'] is None for r in report['rows'][:2]))
        self.assertEqual([e['frame_ids'] for e in report['episodes']], [['f5', 'f6']])

    def test_actual_cli_authenticates_registered_png_and_keeps_quality_unmeasured(self):
        self.masks([(4, 4)]*6)
        args = [sys.executable, '-B', '-m', 'tools.workbench_mask_diagnostics',
            '--workspace', str(self.ws.path), '--clip-id', self.clip_id,
            '--mask-set', 'synthetic', '--output', str(self.out), '--area-ratio-min', '.75']
        completed = subprocess.run(args, cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = wc.loads_strict((self.out/'report.json').read_bytes())
        self.assertEqual(len(report['rows']), 6)
        self.assertFalse(report['new_neural_inference'])
        self.assertIsNone(report['independent_accuracy'])
        self.assertIn('RGB pixels not verified', report['provenance']['rgb_binding'])


if __name__ == '__main__':
    unittest.main()
