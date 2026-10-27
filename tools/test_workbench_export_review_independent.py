"""Independent exact-frame sheet checks using manufactured images and FFV1."""
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from bench import workbench_contract as wc
from bench import workbench_offline_export as exporter
from bench import workbench_cli as cli
from tools.test_workbench_offline_export_integration import make_fixture, export_args, cv2
from tools.test_workbench_mesh_export_independent import plane_reference, selected_png


ROOT = Path(__file__).resolve().parents[1]


class IndependentExportReviewChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / '.cache').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.cache', prefix='export-review-independent-')
        self.addCleanup(self.tmp.cleanup)
        self.f = make_fixture(Path(self.tmp.name))
        self.export = exporter.export_saved_window(self.f['workspace'], **export_args(self.f), surface_mode='mesh')
        self.manifest_path = self.f['output_dir'] / 'manifest.json'
        self.sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.output = self.f['workspace'].derived_dir / 'review'

    def review(self, ids=('f7', 'f3', 'f2'), **kwargs):
        from bench.workbench_export_review import review_saved_export
        return review_saved_export(self.f['workspace'], **dict(export_dir=self.f['output_dir'],
            expected_manifest_sha256=self.sha, frame_ids=list(ids), output_dir=self.output,
            tile_width=128, **kwargs))

    def test_exact_panels_and_schedule_order_with_lost_frame(self):
        report = self.review()
        self.assertEqual([r['frame_id'] for r in report['rows']], ['f2', 'f3', 'f7'])
        self.assertFalse(report['quality_claim'])
        self.assertFalse(report['new_neural_inference'])
        self.assertIsNone(report['independent_accuracy'])
        with Image.open(self.output / report['sheet']['file']) as image:
            sheet = np.array(image.convert('RGB'))
        saved = wc.loads_strict((self.f['workspace'].runs_dir / self.f['run_id'] / 'neural-artwork.json').read_bytes())
        K = np.array(saved['effective_K']).reshape(3, 3)
        for row in report['rows']:
            fid = row['frame_id']
            original = next(r['original'] for r in self.export['rows'] if r['original']['frame_id'] == fid)
            expected_mask = np.repeat((self.f['masks'][fid].astype(np.uint8)*255)[..., None], 3, axis=2)
            if original['pose_state'] == 'tracking':
                expected_composite = plane_reference(self.f['images'][fid], selected_png(), self.f['masks'][fid],
                    saved['marker'], K, np.array(original['pose']), exporter.DEFAULT_PLACEMENT)
            else:
                expected_composite = self.f['images'][fid]
            expected = dict(source=self.f['images'][fid], mask=expected_mask, composite=expected_composite)
            for panel, pixels in expected.items():
                x, y, width, height = row['panels'][panel]
                self.assertEqual((width, height), (128, 96))
                np.testing.assert_array_equal(sheet[y:y+height, x:x+width], pixels)
            self.assertEqual(row['pose_state'], original['pose_state'])

    def test_decoder_misalignment_refuses_even_for_unselected_frame(self):
        # The selected first frame is still correct. A later corrupt decoded
        # frame must invalidate the whole saved-export review, not be skipped.
        from bench import workbench_export_review as review
        real = cv2.VideoCapture
        class CorruptCapture:
            def __init__(self, path):
                self.inner = real(path); self.index = 0
            def isOpened(self): return self.inner.isOpened()
            def get(self, key): return self.inner.get(key)
            def release(self): self.inner.release()
            def read(self):
                ok, frame = self.inner.read()
                if ok and self.index == 3:
                    frame = frame.copy(); frame[0, 0, 0] ^= 1
                self.index += 1
                return ok, frame
        with patch.object(cv2, 'VideoCapture', CorruptCapture):
            with self.assertRaises(Exception): self.review(ids=('f2',))
        self.assertFalse(self.output.exists())

    def test_no_model_imports_and_original_artifacts_unchanged(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.f['workspace'].path.rglob('*') if p.is_file()}
        real = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'torchvision', 'sam2', 'gotrack', 'onnxruntime'}:
                raise AssertionError('Unexpected model import: '+name)
            return real(name, *args, **kwargs)
        with patch('builtins.__import__', guarded): self.review()
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_late_source_decode_change_cannot_publish(self):
        original = cli.FrameReader.read
        counts = {}
        def changed(reader, frame):
            rgb, ident = original(reader, frame)
            counts[frame.frame_id] = counts.get(frame.frame_id, 0) + 1
            if counts[frame.frame_id] > 1:
                ident = {**ident, 'image_sha256': '0'*64}
            return rgb, ident
        with patch.object(cli.FrameReader, 'read', changed):
            with self.assertRaises(Exception): self.review()
        self.assertFalse(self.output.exists())

    def test_real_cli_and_manifest_pin(self):
        from tools.workbench_export_review import main
        argv = ['--workspace', str(self.f['workspace'].path), '--export', str(self.f['output_dir']),
                '--manifest-sha256', self.sha, '--frame-id', 'f6', '--frame-id', 'f4',
                '--output', str(self.output), '--tile-width', '128']
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(argv), 0)
        self.assertEqual(json.loads(stdout.getvalue())['status'], 'completed')
        report = json.loads((self.output / 'report.json').read_text())
        self.assertEqual(report['selected_frame_ids'], ['f4', 'f6'])
        self.assertEqual(report['rows'][0]['pose_state'], 'recovering')
        self.assertEqual(report['rows'][1]['mask_state'], 'empty')
        self.assertEqual(report['provenance']['manifest_sha256'], self.sha)
        self.assertIn('pillow', report['dependencies'])

    def test_selected_source_file_corruption_is_refused(self):
        path = self.f['workspace'].path / 'synthetic-inputs' / 'f2.png'
        with path.open('ab') as stream:
            stream.write(b'changed registered file')
        with self.assertRaises(Exception): self.review(ids=('f2',))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
