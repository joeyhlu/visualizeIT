"""Independent analytical controls for an explicitly reviewed seed-frame anchor.

Manufactured plane/video inputs only. These are software checks, not attachment
accuracy measurements on a real object or a neural-model evaluation.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_geometry as geom
from bench import workbench_offline_export as exporter
from bench import workbench_cli as cli
from tools.test_workbench_mesh_export_independent import plane_reference, selected_png
from tools.test_workbench_offline_export_integration import make_fixture, export_args, cv2, use_video_source


ROOT = Path(__file__).resolve().parents[1]
VERTICES = np.array([[-.2, -.15, 0], [.2, -.15, 0], [.2, .15, 0], [-.2, .15, 0]], dtype=np.float32).astype(float)
FACES = np.array([[0, 2, 1], [0, 3, 2]])


def request_file(fixture, pixel):
    path = fixture['workspace'].path / 'manual-anchor.json'
    path.write_bytes(wc.canonical_json(dict(schema='workbench_mesh_anchor_v1',
        clip_id='synthetic-offline-clip', frame_id='f0', pixel=list(pixel),
        reviewed=True, reviewed_by='independent manufactured control')))
    return path


def translated_plane_reference(rgb, mask, marker, pose, center):
    """Move the camera translation rather than using the production raycast.

    Pinhole projection of a plane relative to anchor A is R(X-A)+(t+RA).
    The existing independent plane oracle can therefore use an origin anchor
    with translation t+RA, while retaining physical footprint and texel axes.
    """
    shifted = np.array(pose, float)
    shifted[:3, 3] += shifted[:3, :3] @ center
    K = np.array([[200., 0, 63.5], [0, 200., 47.5], [0, 0, 1.]])
    return plane_reference(rgb, selected_png(), mask, marker, K, shifted,
                           exporter.DEFAULT_PLACEMENT)


class IndependentMeshAnchorChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / '.cache').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.cache', prefix='anchor-independent-')
        self.addCleanup(self.tmp.cleanup)
        self.fixture = make_fixture(Path(self.tmp.name))

    def export(self, pixel=(71.25, 42.75), **kwargs):
        f = self.fixture
        return exporter.export_saved_window(f['workspace'], **export_args(f),
            surface_mode='mesh', mesh_anchor_path=request_file(f, pixel), **kwargs)

    def test_analytic_surface_anchor_and_decoded_artwork_across_saved_poses(self):
        f = self.fixture
        run_dir = f['workspace'].runs_dir / f['run_id']
        originals = {name: (run_dir / name).read_bytes() for name in
                     ('neural-artwork.json', 'results.jsonl', 'run.json', 'report.json')}
        pixel = (71.25, 42.75)
        expected_center = np.array([(pixel[0]-63.5)/200, (pixel[1]-47.5)/200, 0.])
        report = self.export(pixel)
        marker = report['mesh']['derived_marker']
        actual_center = np.asarray(marker['barycentric']) @ VERTICES[FACES[marker['face_id']]]
        np.testing.assert_allclose(actual_center, expected_center, atol=1e-12, rtol=0)
        saved = wc.loads_strict(originals['neural-artwork.json'])
        self.assertNotEqual(marker['barycentric'], saved['marker']['barycentric'])
        self.assertAlmostEqual(marker['size_object_units'], saved['marker']['size_object_units'])
        self.assertEqual(report['provenance']['original_artwork_record_sha256'], saved['record_sha256'])
        self.assertEqual(report['frame_accounting']['scheduled'], 6)
        self.assertFalse(report['new_neural_inference'])
        capture = cv2.VideoCapture(str(f['output_dir'] / 'playback.avi'))
        self.addCleanup(capture.release)
        rendered = 0
        for row in report['rows']:
            ok, bgr = capture.read()
            self.assertTrue(ok)
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            fid = row['original']['frame_id']
            if row['render_reason'] == 'rendered_mesh_artwork':
                expected = translated_plane_reference(f['images'][fid], f['masks'][fid],
                    marker, row['original']['pose'], expected_center)
                np.testing.assert_array_equal(rgb, expected)
                rendered += 1
            else:
                np.testing.assert_array_equal(rgb, f['images'][fid])
        self.assertEqual(rendered, 2)
        self.assertFalse(capture.read()[0])
        for name, original in originals.items():
            self.assertEqual((run_dir / name).read_bytes(), original)

    def test_click_across_diagonal_changes_real_mesh_face(self):
        for i, pixel in enumerate(((72., 39.), (54., 56.))):
            self.fixture['output_dir'] = self.fixture['workspace'].derived_dir / ('face-' + str(i))
            r = self.export(pixel)
            marker = r['mesh']['derived_marker']
            center = np.asarray(marker['barycentric']) @ VERTICES[FACES[marker['face_id']]]
            np.testing.assert_allclose(center, [(pixel[0]-63.5)/200, (pixel[1]-47.5)/200, 0], atol=1e-12)
            self.assertEqual(marker['face_id'], i)

    def test_video_seed_is_decoded_even_when_not_scored(self):
        use_video_source(self.fixture)
        calls = []
        original = cli.FrameReader.read
        def record(reader, frame):
            calls.append(frame.frame_id)
            return original(reader, frame)
        with patch.object(cli.FrameReader, 'read', record):
            self.export()
        self.assertGreaterEqual(calls.count('f0'), 2)

    def test_seed_decode_change_refuses_publication_and_cleans_staging(self):
        original = cli.FrameReader.read
        calls = 0
        def changed(reader, frame):
            nonlocal calls
            rgb, identity = original(reader, frame)
            if frame.frame_id == 'f0':
                calls += 1
                if calls > 1:
                    rgb = rgb.copy(); rgb[0, 0, 0] ^= 1
                    identity = copy.deepcopy(identity)
                    identity['image_sha256'] = geom.array_sha256(rgb)
            return rgb, identity
        with patch.object(cli.FrameReader, 'read', changed):
            with self.assertRaisesRegex(Exception, 'seed|anchor|source'):
                self.export()
        self.assertFalse(self.fixture['output_dir'].exists())
        self.assertEqual(list(self.fixture['workspace'].derived_dir.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
