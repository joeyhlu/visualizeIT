"""Synthetic contracts for the sealed, private prototype report gallery."""
import math
import io
import json
from pathlib import Path
import unittest
import uuid
from contextlib import redirect_stdout
from unittest import mock

from . import prototype_gallery as gallery


def trace(states=None):
    states = states or ['tracking'] * 110
    return [dict(frameId=400 + index, logicalIndex=index + 10,
                 state=states[index], failureReason='synthetic loss' if states[index] != 'tracking' else None,
                 cvCameraFromSourceObject=[[1, 0, 0, 0], [0, 1, 0, 0],
                                           [0, 0, 1, 1], [0, 0, 0, 1]] if index == 20 else None,
                 forcedInterruption=50 <= index <= 54)
            for index in range(110)]


class PrototypeGalleryTests(unittest.TestCase):
    def test_sample_plan_keeps_fixed_points_and_adds_first_loss_and_recovery(self):
        states = ['tracking'] * 110
        states[40:43] = ['lost', 'recovering', 'tracking']
        plan = gallery.sample_plan(trace(states))
        by_index = {row['index']: row for row in plan}
        self.assertEqual(set(by_index), {0, 36, 40, 42, 73, 109})
        self.assertIn('first nontracking frame', by_index[40]['tags'])
        self.assertIn('first tracking frame after loss', by_index[42]['tags'])
        self.assertIn('fixed index 36', by_index[36]['tags'])

    def test_first_failure_is_not_replaced_by_a_stale_pose(self):
        frame = trace(['tracking'] * 110)[40]
        frame['state'] = 'lost'
        frame['cvCameraFromSourceObject'] = [[1, 0, 0, 3], [0, 1, 0, 4],
                                              [0, 0, 1, 5], [0, 0, 0, 1]]
        self.assertIsNone(gallery.estimated_pose_for_frame(frame))
        self.assertIsNotNone(gallery.estimated_pose_for_frame(
            {'state': 'tracking', 'cvCameraFromSourceObject': frame['cvCameraFromSourceObject']}))

    def test_forced_interruption_panels_show_blank_input_and_recording_only_as_reference(self):
        import numpy as np
        from PIL import Image
        frames = trace()
        frames[50]['cvCameraFromSourceObject'] = [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        rgb_spec = {'path': 'rgb/frame.png', 'sha256': 'a' * 64}
        calibration = {'cam_K': [100, 0, 24, 0, 100, 16, 0, 0, 1]}
        dataset_pose = {'cam_R_m2c': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        'cam_t_m2c': [0, 0, 1000]}
        inference_obj = {'frames': [
            {'rgb': rgb_spec, 'calibration': calibration} for _ in range(110)]}
        evaluator_obj = {'frames': [
            {'rgb': rgb_spec, 'calibration': calibration, 'evaluationPose': dataset_pose}
            for _ in range(110)]}
        recorded = np.empty((32, 48, 3), dtype=np.uint8)
        recorded[:] = [190, 60, 30]
        measured_depth = np.full((32, 48), 1.0, dtype=np.float32)
        context = type('Context', (), {})()
        context.dataset = Path('.')
        context.rgb_cache = {}
        context.depth_cache = {}
        context.mesh = object()
        context.transform = None
        written = {}

        def overlay(source_rgb, *_args):
            return Image.fromarray(source_rgb.copy(), mode='RGB'), 'measured_depth_occlusion'

        with (mock.patch.object(gallery, '_load_rgb', return_value=recorded),
              mock.patch.object(gallery, '_load_depth', return_value=measured_depth),
              mock.patch.object(gallery, '_overlay', side_effect=overlay) as overlay_mock):
            sample = gallery._sample_asset(context, inference_obj, evaluator_obj, frames,
                {'index': 50, 'frameId': frames[50]['frameId'], 'tags': ['forced interruption']},
                'recovery', 2, 'dataset_reference',
                lambda name, content: written.update({name: content}))

        self.assertTrue(sample['forcedBlankTrackerInput'])
        self.assertEqual(sample['estimatedRenderState'], 'suppressed_forced_blank_input')
        self.assertEqual(sample['inputPanelLabel'], 'Forced blank tracker input (zero RGB)')
        self.assertEqual(sample['estimatedPanelLabel'], 'Estimated design render (suppressed)')
        self.assertEqual(sample['referenceRgbSource'], 'original_recorded_rgb')
        self.assertEqual(sample['depthUseScope'], 'dataset-pose reference control only')
        self.assertEqual(sample['evaluationMaskUse'], 'not used')
        self.assertEqual(overlay_mock.call_count, 1)
        np.testing.assert_array_equal(overlay_mock.call_args.args[0], recorded)
        np.testing.assert_array_equal(overlay_mock.call_args.args[-1], measured_depth)

        with Image.open(io.BytesIO(written[sample['imagePath']])) as sheet:
            pixels = np.asarray(sheet.convert('RGB'))
        y = 24 + 16
        np.testing.assert_array_less(pixels[y, 30], [15, 15, 15])
        np.testing.assert_array_less(pixels[y, 350], [15, 15, 15])
        self.assertGreater(int(pixels[y, 670, 0]), 120)
        self.assertGreater(int(pixels[y, 670, 1]), 25)

    def test_timeline_retains_all110_frames_and_marks_unavailable_scores(self):
        frames = trace()
        frames[5]['state'] = 'lost'
        metrics = {'perFrameAttachment': [
            {'frameId': frames[5]['frameId'], 'pairedPoints': 0,
             'medianErrorPixels': None, 'p95ErrorPixels': None,
             'unavailableReason': 'depth unavailable'}]}
        rows = gallery.timeline_rows(frames, metrics)
        self.assertEqual(len(rows), 110)
        self.assertEqual(rows[5]['state'], 'lost')
        self.assertEqual(rows[5]['unavailableReason'], 'depth unavailable')
        self.assertNotIn('evaluationPose', rows[5])
        self.assertEqual(rows[0]['frameId'], 400)
        self.assertEqual(rows[-1]['frameId'], 509)

    def test_depth_occlusion_uses_measured_depth_and_never_an_eval_mask(self):
        import numpy as np
        triangle = np.array([[0, 1, -1]], dtype=np.int32)
        rendered = np.array([[2.0, 1.0, math.inf]], dtype=np.float64)
        measured = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
        visible, reason = gallery.physical_occlusion_mask(triangle, rendered, measured)
        self.assertEqual(visible.tolist(), [[False, True, False]])
        self.assertEqual(reason, 'measured_depth_occlusion')
        silhouette, reason = gallery.physical_occlusion_mask(triangle, rendered, None)
        self.assertEqual(silhouette.tolist(), [[True, True, False]])
        self.assertEqual(reason, 'unoccluded_depth_unavailable')

    def test_metrics_copy_strips_ground_truth_and_pose_fields(self):
        value = {'medianErrorPixels': 2.0,
                 'perFrameAttachment': [{'frameId': 1, 'evaluationPose': {'secret': 1},
                                         'cameraFromObject': [1, 2], 'pairedPoints': 4}]}
        copied = gallery._safe_metrics(value)
        self.assertEqual(copied['perFrameAttachment'], [{'frameId': 1, 'pairedPoints': 4}])
        self.assertEqual(copied['medianErrorPixels'], 2.0)

    def test_synthetic_control_is_one_smoke_object_and_zero_real_unseen(self):
        report = {'evidenceKind': 'synthetic_control', 'dataset': 'Synthetic fixture',
                  'realUnseenObjects': 0, 'attemptedObjects': 1,
                  'tenObjectsEvaluated': False, 'benchmarkGateEligible': False}
        seal = {'evidenceKind': 'synthetic_control', 'realUnseenObjects': 0}
        evidence = gallery._evidence_contract(report, seal, 1)
        self.assertEqual(evidence['evidenceKind'], 'synthetic_control')
        self.assertEqual(evidence['realUnseenObjects'], 0)
        self.assertIn('not a new physical identity', evidence['scopeNote'])
        self.assertNotIn('YCB-Video', evidence['provenance'])
        with self.assertRaises(gallery.GalleryError):
            gallery._evidence_contract(dict(report, realUnseenObjects=1), seal, 1)
        with self.assertRaises(gallery.GalleryError):
            gallery._evidence_contract(dict(report, benchmarkGatePassed=True), seal, 1)
        with self.assertRaises(gallery.GalleryError):
            gallery._evidence_contract(dict(report, tenObjectsEvaluated=True), seal, 1)

    def test_dataset_labels_are_derived_from_sealed_report_contract(self):
        report = {'evidenceKind': 'dataset_reference', 'dataset': 'BOP YCB-Video',
                  'realUnseenObjects': 10, 'tenObjectsEvaluated': True,
                  'provenance': 'dataset-reference only'}
        evidence = gallery._evidence_contract(report, {'evidenceKind': 'dataset_reference',
            'realUnseenObjects': 10}, 10)
        self.assertEqual(evidence['datasetName'], report['dataset'])
        self.assertEqual(evidence['provenance'], report['provenance'])
        self.assertEqual(evidence['realUnseenObjects'], 10)

    def test_dataset_reference_pilot_is_labeled_below_the_ten_object_gate(self):
        report = {'evidenceKind': 'dataset_reference', 'benchmarkMode': 'pilot',
                  'dataset': 'BOP YCB-Video', 'realUnseenObjects': 5,
                  'plannedObjects': 5, 'attemptedObjects': 5,
                  'tenObjectsEvaluated': False, 'benchmarkGateEligible': False}
        seal = {'evidenceKind': 'dataset_reference', 'benchmarkMode': 'pilot',
                'realUnseenObjects': 5, 'plannedObjects': 5,
                'tenObjectsEvaluated': False}
        evidence = gallery._evidence_contract(report, seal, 5)
        self.assertEqual(evidence['evidenceKind'], 'dataset_reference')
        self.assertEqual(evidence['benchmarkMode'], 'pilot')
        self.assertEqual(evidence['plannedObjects'], 5)
        self.assertFalse(evidence['benchmarkGateEligible'])
        self.assertFalse(evidence['tenObjectsEvaluated'])
        self.assertIn('not the ten-object benchmark', evidence['provenance'])
        self.assertIn('not evaluated', evidence['scopeNote'])
        with self.assertRaises(gallery.GalleryError):
            gallery._evidence_contract(dict(report, realUnseenObjects=4), seal, 5)
        with self.assertRaises(gallery.GalleryError):
            gallery._evidence_contract(dict(report, tenObjectsEvaluated=True), seal, 5)
        with self.assertRaises(gallery.GalleryError):
            gallery._evidence_contract(dict(report, benchmarkGateEligible=True), seal, 5)

    def test_json_serialization_escapes_markup_and_script_delimiters(self):
        raw = gallery._json_bytes({'name': '</script><img src=x onerror=alert(1)>'})
        text = raw.decode('utf-8')
        self.assertNotIn('<script', text.lower())
        self.assertIn('\\u003c/script\\u003e', text)
        self.assertIn('\\u0026', gallery._json_bytes({'value': '&'}).decode('utf-8'))

    def test_budget_rejects_only_the_write_that_crosses_the_private_cap(self):
        gallery._check_budget(gallery.MAX_GALLERY_BYTES - 10, 10)
        with self.assertRaises(gallery.GalleryBudgetExceeded):
            gallery._check_budget(gallery.MAX_GALLERY_BYTES - 10, 11)

    def test_output_policy_keeps_gallery_private_under_results_only(self):
        root = Path.cwd() / '.cache' / 'prototype-sprint' / 'results-v1'
        dataset = Path.cwd() / '.cache' / 'prototype-sprint' / 'dataset'
        accepted = root / f'demo-policy-check-{uuid.uuid4().hex}'
        results, data, output = gallery._validate_output(root, dataset, accepted)
        self.assertEqual(output, accepted.resolve())
        self.assertEqual(results, root.resolve())
        self.assertEqual(data, dataset.resolve())
        with self.assertRaises(gallery.GalleryError):
            gallery._validate_output(root, dataset, Path.cwd() / 'public' / 'demo')
        with self.assertRaises(gallery.GalleryError):
            gallery._validate_output(root, dataset, dataset / 'demo')

    def test_gallery_page_builds_dynamic_content_with_text_nodes(self):
        html = gallery._html_document()
        self.assertIn('n.textContent=String(text)', html)
        self.assertNotIn('.innerHTML', html)
        self.assertIn('all 110 frame states', html)
        self.assertIn('data.datasetName', html)
        self.assertIn('data.evidenceKind', html)
        self.assertIn('no live camera FPS claim', html)
        self.assertIn("/gate|check/i.test(String(key||''))", html)
        self.assertIn('format(value,key)', html)

    def test_cli_routes_required_paths_and_optional_output_to_generator(self):
        receipt = {'status': 'complete', 'objectCount': 5, 'generatedBytes': 12345}
        output = io.StringIO()
        with mock.patch.object(gallery, 'generate', return_value=receipt) as generate, \
             redirect_stdout(output):
            returned = gallery.main(['--dataset', 'dataset-pilot-v1',
                                     '--results', 'results-pilot-v1',
                                     '--output', 'results-pilot-v1/demo-custom'])
        self.assertEqual(returned, receipt)
        generate.assert_called_once_with(Path('dataset-pilot-v1'),
            Path('results-pilot-v1'), Path('results-pilot-v1/demo-custom'))
        self.assertEqual(json.loads(output.getvalue()), receipt)

        output = io.StringIO()
        with mock.patch.object(gallery, 'generate', return_value=receipt) as generate, \
             redirect_stdout(output):
            gallery.main(['--dataset', 'dataset-v1', '--results', 'results-v1'])
        generate.assert_called_once_with(Path('dataset-v1'), Path('results-v1'), None)


if __name__ == '__main__':
    unittest.main()
