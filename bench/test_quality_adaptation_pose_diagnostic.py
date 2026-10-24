"""Manufactured pinhole controls; these are software tests, not model accuracy."""
import copy
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bench import quality_adaptation_pose_diagnostic as diagnostic


def fixture():
    yy, xx = np.mgrid[:280, :280]
    depth = .7 + .025 * np.sin(xx / 25.) * np.cos(yy / 30.)
    # Analytical camera: R=I, t=(0,0,.7)m, fx=fy=200, cx=cy=140.
    xyz = np.stack(((xx - 140) * depth / 200,
                    (yy - 140) * depth / 200, depth - .7), axis=-1).astype('float32')
    mask = np.zeros((280, 280), dtype=bool)
    mask[30:250:10, 30:250:10] = True
    features = {'template_xyz_m': xyz, 'template_mask': mask,
                'query_K': np.array([[200., 0, 140], [0, 200., 140], [0, 0, 1.]])}
    flow = np.zeros((2, 280, 280), dtype='float32')
    confidence = np.ones((280, 280), dtype='float32')
    labels = {'flow': flow.copy(), 'visible': mask.copy(), 'known': mask.copy()}
    return features, flow, confidence, labels


class PoseDiagnosticTests(unittest.TestCase):
    def test_manufactured_pose_metres_to_millimetres(self):
        features, flow, confidence, _ = fixture()
        result = diagnostic.predict_pose(features, flow, confidence)
        self.assertTrue(result['solver_success'])
        # Independently manufactured t=.7m must emerge as 700mm.
        np.testing.assert_allclose(np.asarray(result['crop_camera_from_object_mm'])[:3, 3],
                                   [0, 0, 700], atol=.002, rtol=0)
        np.testing.assert_allclose(np.asarray(result['crop_camera_from_object_mm'])[:3, :3],
                                   np.eye(3), atol=1e-5, rtol=0)

    def test_exact_projection_and_identical_branches(self):
        features, flow, confidence, labels = fixture()
        first = diagnostic.predict_pose(features, flow, confidence)
        second = diagnostic.predict_pose(features, flow, confidence)
        self.assertEqual(first, second)
        score = diagnostic.score_pose(features, first, labels)
        self.assertLess(score['projection_error_px']['maximum'], .001)
        self.assertEqual(score['within_2px_fraction_all_visible'], 1.)

    def test_confidence_cutoff_native_float32(self):
        features, flow, confidence, _ = fixture()
        confidence.fill(np.float32(.3))
        result = diagnostic.predict_pose(features, flow, confidence)
        self.assertEqual(result['retained_point_count'], 0)
        self.assertFalse(result['solver_success'])
        confidence.fill(np.nextafter(np.float32(.3), np.float32(1)))
        self.assertEqual(diagnostic.predict_pose(features, flow, confidence)['retained_point_count'],
                         int(features['template_mask'].sum()))

    def test_insufficient_and_all_bad_count_as_missing(self):
        features, flow, confidence, labels = fixture()
        confidence.fill(0)
        result = diagnostic.predict_pose(features, flow, confidence)
        score = diagnostic.score_pose(features, result, labels)
        self.assertFalse(result['solver_success'])
        self.assertEqual(score['projected_point_count'], 0)
        self.assertEqual(score['missing_projection_point_count'], int(labels['visible'].sum()))
        self.assertEqual(score['within_2px_fraction_all_visible'], 0.)

    def test_flow_outside_crop_is_not_solver_support(self):
        features, flow, confidence, _ = fixture()
        flow[0].fill(280)
        self.assertEqual(diagnostic.predict_pose(features, flow, confidence)['retained_point_count'], 0)

    def test_subsample_is_deterministic_row_major(self):
        features, flow, confidence, _ = fixture()
        features['template_mask'].fill(True)
        result = diagnostic.predict_pose(features, flow, confidence)
        # Same deterministic linspace sampling as the existing GoTrack refiner.
        expected = np.linspace(0, 280 * 280 - 1, 10000, dtype=int)
        np.testing.assert_array_equal(result['sampled_template_flat_indices'], expected)
        self.assertEqual(result['sampled_point_count'], 10000)

    def test_label_change_cannot_change_prediction(self):
        features, flow, confidence, labels = fixture()
        before = diagnostic.predict_pose(features, flow, confidence)
        labels['flow'][0, labels['visible']] += 4
        after = diagnostic.predict_pose(features, flow, confidence)
        self.assertEqual(before, after)
        score = diagnostic.score_pose(features, after, labels)
        self.assertAlmostEqual(score['projection_error_px']['mean'], 4., places=3)

    def test_analytical_translation_projection(self):
        features, flow, confidence, labels = fixture()
        result = diagnostic.predict_pose(features, flow, confidence)
        pose = np.eye(4)
        pose[:3, 3] = [10, 0, 700]
        result['crop_camera_from_object_mm'] = pose.tolist()
        score = diagnostic.score_pose(features, result, labels)
        # Independent pinhole formula: delta_u = fx * delta_x / camera_depth.
        z = features['template_xyz_m'][labels['visible'], 2].astype('float64') + .7
        self.assertAlmostEqual(score['projection_error_px']['mean'], float((2 / z).mean()), places=4)

    def test_behind_camera_is_not_silently_dropped(self):
        features, flow, confidence, labels = fixture()
        result = diagnostic.predict_pose(features, flow, confidence)
        pose = np.eye(4)
        pose[2, 3] = -700
        result['crop_camera_from_object_mm'] = pose.tolist()
        score = diagnostic.score_pose(features, result, labels)
        self.assertEqual(score['behind_camera_point_count'], int(labels['visible'].sum()))
        self.assertEqual(score['within_2px_fraction_all_visible'], 0.)

    def test_malformed_predictions_rejected(self):
        features, flow, confidence, _ = fixture()
        for bad_flow, bad_confidence in ((flow.astype('float64'), confidence),
                                        (flow, confidence * np.float32(2))):
            with self.subTest(dtype=bad_flow.dtype, max_confidence=bad_confidence.max()):
                with self.assertRaises((ValueError, TypeError)):
                    diagnostic.predict_pose(features, bad_flow, bad_confidence)
        bad = copy.deepcopy(flow)
        bad[0, 0, 0] = np.nan
        with self.assertRaises((ValueError, TypeError)):
            diagnostic.predict_pose(features, bad, confidence)

    def test_cli_seals_all_predictions_before_labels_and_counts_missing_rows(self):
        """Mock only head forwards/runtime; real PnP, persistence and scorer run."""
        features, flow, confidence, labels = fixture()
        features.update(baseline_flow=flow, baseline_confidence=confidence)
        for hook in diagnostic.data.HOOKS:
            features[f'tokens_{hook}'] = np.zeros((1,), dtype='float32')

        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
            def __getitem__(self, item): return Tensor(self.value[item])

        class Head:
            def load_state_dict(self, state, strict): self.state = state
            def requires_grad_(self, value): return self
            def eval(self): return self
            def __call__(self, tokens, shape):
                return Tensor(flow[None]), Tensor(confidence[None])

        fake_torch = SimpleNamespace(manual_seed=lambda _: None,
            use_deterministic_algorithms=lambda _: None,
            inference_mode=contextlib.nullcontext, from_numpy=lambda x: x)
        scratch = Path(__file__).resolve().parents[1] / '.cache'
        with tempfile.TemporaryDirectory(prefix='pose-diagnostic-test-', dir=scratch) as directory:
            root = Path(directory)
            label_path = root / 'labels.npz'
            np.savez(label_path, **labels)
            label_descriptor = dict(path='labels.npz', bytes=label_path.stat().st_size,
                                    sha256=diagnostic.data.sha256(label_path))
            rows = [dict(id='val-1', object_id='4', split='val', status='available',
                         features=dict(path='features.npz', bytes=1, sha256='1' * 64),
                         labels=label_descriptor),
                    dict(id='missing', object_id='4', split='val', status='unavailable',
                         failure_reason='original_capture_failed')]
            source_hash = diagnostic.data.sha256(diagnostic.runtime.LOCK)
            ds = diagnostic.data.Dataset(root, dict(rows=rows, provenance=dict(
                label_origin='synthetic_renderer_training_only',
                feature_recipe_sha256='2' * 64, source_lock_sha256=source_hash)), '3' * 64)
            manifest = root / 'manifest.json'
            manifest.write_text('{}')  # Dataset parsing alone is replaced by the controlled fixture.
            plan = dict(schema=diagnostic.PLAN_SCHEMA, dataset_sha256=ds.identity,
                        feature_recipe_sha256='2' * 64, source_lock_sha256=source_hash,
                        exports=[dict(name='candidate', path='head.pt', sha256='4' * 64,
                                      bytes=1, dataset_sha256='5' * 64, config_sha256='6' * 64)])
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            output = root / 'output'
            original_load = diagnostic.data.load_npz
            label_reads = []

            def guarded_labels(path, keys):
                self.assertEqual(keys, diagnostic.data.LABEL_KEYS)
                seal = json.loads((output / 'prediction-seal.json').read_text())
                self.assertEqual(set(seal['branches']), {'baseline', 'candidate'})
                self.assertFalse(seal['labels_opened'])
                for entry in seal['branches'].values():
                    self.assertEqual(diagnostic.data.sha256(output / entry['path']), entry['sha256'])
                label_reads.append(str(path))
                return original_load(path, keys)

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict('sys.modules', {'torch': fake_torch}))
                stack.enter_context(patch.object(diagnostic.runtime, 'cloud_guard'))
                stack.enter_context(patch.object(diagnostic.runtime, 'verify_sources', return_value={}))
                stack.enter_context(patch.object(diagnostic.runtime, 'runtime_versions', return_value={}))
                stack.enter_context(patch.object(diagnostic.runtime, 'load_head', side_effect=lambda **_: Head()))
                loader = stack.enter_context(patch.object(diagnostic.data, 'load_dataset', return_value=ds))
                stack.enter_context(patch.object(diagnostic, '_features', side_effect=lambda *_: features))
                export_loader = stack.enter_context(patch.object(diagnostic, '_export', return_value={'head': {}}))
                stack.enter_context(patch.object(diagnostic.data, 'load_npz', side_effect=guarded_labels))
                report = diagnostic.run(manifest, plan_path, diagnostic.data.sha256(plan_path), output,
                                        cloud_execution=True)
                loader.assert_called_once_with(manifest.absolute(), inspect_arrays=False)
                # A later branch authentication failure must leave labels unopened,
                # even after the baseline has produced all its poses.
                label_reads_before_failure = len(label_reads)
                output = root / 'failed-output'
                export_loader.side_effect = ValueError('manufactured export digest mismatch')
                with self.assertRaisesRegex(ValueError, 'export digest mismatch'):
                    diagnostic.run(manifest, plan_path, diagnostic.data.sha256(plan_path), output,
                                   cloud_execution=True)
                self.assertEqual(len(label_reads), label_reads_before_failure)
                self.assertFalse((output / 'prediction-seal.json').exists())
                self.assertFalse((output / 'report.json').exists())
            self.assertEqual(len(label_reads), 2)
            self.assertFalse(report['test_arrays_opened'])
            for branch in report['branches'].values():
                self.assertEqual(branch['requested_val_rows'], 2)
                self.assertEqual(branch['pose_available_rows'], 1)
                self.assertEqual(branch['failed_rows'], 1)
                self.assertEqual(branch['pose_availability'], .5)
            with self.assertRaises(FileExistsError):
                diagnostic._write(root / 'output', 'report.json', report)


if __name__ == '__main__':
    unittest.main()
