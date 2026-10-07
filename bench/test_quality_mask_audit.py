"""Synthetic CPU tests for the frozen cached-mask audit; no real labels are created."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_mask_audit as audit
from .vision import cv2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png(mask):
    ok, encoded = cv2.imencode('.png', mask)
    if not ok:
        raise AssertionError('OpenCV fixture encoding failed')
    return encoded.tobytes()


def polygon(points):
    return [{'points': points}]


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class MaskAuditTests(unittest.TestCase):
    def setUp(self):
        self.canonical_selection_constants = dict(audit.FROZEN_ANNOTATION_SELECTIONS)
        self.window_patch = patch.dict(audit.ORIGINAL_WINDOWS, {
            'keyboard': {'setup_frame_id': 8, 'frame_ids': (9, 10, 11, 12), 'resolution': (32, 24)},
        }, clear=True)
        self.window_patch.start()
        self.selection_patch = patch.object(audit, 'FROZEN_ANNOTATION_SELECTIONS', {
            'keyboard': ((9, 'uniform'), (10, 'uniform'), (11, 'uniform')),
        })
        self.selection_patch.start()
        self.count_patch = patch.object(audit, 'ANNOTATION_FRAME_COUNT', 3)
        self.count_patch.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'input-bundle'
        self.bundle.mkdir()
        self.mask_root = self.root / 'segmentation'
        (self.mask_root / 'masks').mkdir(parents=True)
        (self.bundle / 'source.mp4').write_bytes(b'original RGB video fixture')
        (self.bundle / 'object.glb').write_bytes(b'fixed model fixture')
        self.input_path = self.bundle / 'input.json'
        self.results_path = self.root / 'complete.json'
        self.mask_results_path = self.mask_root / 'results.json'
        self.annotations_path = self.root / 'annotations.json'
        self.width, self.height = (32, 24)
        self.ids = [9, 10, 11, 12]
        self.setup_id = 8
        self.masks = {}
        for frame_id in [self.setup_id] + self.ids:
            mask = np.zeros((self.height, self.width), np.uint8)
            mask[3:8, 3:8] = 255
            self.masks[frame_id] = mask
        self.annotations = [
            self.annotation_row(frame_id, 'pending')
            for frame_id, _ in audit.FROZEN_ANNOTATION_SELECTIONS['keyboard']
        ]
        self.write_sources()

    def tearDown(self):
        self.temp.cleanup()
        self.count_patch.stop()
        self.selection_patch.stop()
        self.window_patch.stop()

    def annotation_row(self, frame_id, status, visible=None, hands=None, visibility=None):
        row = {
            'frame_id': frame_id,
            'image': f'{frame_id}.jpg',
            'group': dict(audit.FROZEN_ANNOTATION_SELECTIONS['keyboard']).get(frame_id, 'uniform'),
            'status': status,
            'visible_object': visible or [],
            'overlapping_hands': hands or [],
            'landmarks': [],
            'provenance': 'Independent source RGB human review fixture',
        }
        if visibility is not None:
            row['visibility'] = visibility
        return row

    def save_mask(self, frame_id, mask=None, corrupt=None):
        path = self.mask_root / 'masks' / f'{frame_id}.png'
        if corrupt is not None:
            path.write_bytes(corrupt)
            return sha(corrupt)
        encoded = png(self.masks[frame_id] if mask is None else mask)
        path.write_bytes(encoded)
        return sha(encoded)

    def write_sources(self, *, mask_provenance=True, detections=None, state_overrides=None):
        manifest = {
            'schema_version': 1,
            'object': 'keyboard',
            'video': 'source.mp4',
            'asset': 'object.glb',
            'native_resolution': [self.width, self.height],
            'setup_frame_id': self.setup_id,
            'frame_ids': self.ids,
            'source_hashes': {
                'source.mp4': sha((self.bundle / 'source.mp4').read_bytes()),
                'object.glb': sha((self.bundle / 'object.glb').read_bytes()),
            },
        }
        self.input_path.write_text(json.dumps(manifest), encoding='utf-8')
        input_sha = sha(self.input_path.read_bytes())

        mask_hashes = {}
        states = {}
        for frame_id in [self.setup_id] + self.ids:
            area = int(np.count_nonzero(self.masks[frame_id]))
            states[frame_id] = 'available' if area >= audit.USABLE_MASK_MIN_PIXELS else 'lost'
            if state_overrides and frame_id in state_overrides:
                states[frame_id] = state_overrides[frame_id]
            mask_hashes[frame_id] = self.save_mask(frame_id)

        adapter_hashes = {
            'quality_assets': 'assets-v1',
            'quality_sam2': 'sam2-v1',
            'quality_cnos': 'cnos-v1',
            'quality_detection_association': 'association-v1',
            'quality_foundpose': 'pose-v1',
        }
        identity_hashes = {
            'source_revisions': {'sam2': 'sam2-source', 'gotrack': 'gotrack-source'},
            'submodule_revisions': {'dinov2': 'dino-source'},
            'checkpoint_sha256': {'sam2.pt': 'sam2-checkpoint', 'dino.pt': 'dino-checkpoint'},
        }
        run_provenance = {
            'input_manifest_sha256': input_sha,
            'adapter_sha256': adapter_hashes,
            **identity_hashes,
        }
        cache_provenance = {
            'input_manifest_sha256': input_sha,
            'adapter_sha256': dict(adapter_hashes),
            **identity_hashes,
        }

        results_frames = []
        for frame_id in self.ids:
            results_frames.append({
                'frameId': frame_id,
                'mask_state': states[frame_id],
                'mask_path': f'masks/{frame_id}.png',
                'mask_sha256': mask_hashes[frame_id],
                'pose_state': 'tracking' if states[frame_id] == 'available' else 'lost',
                'render_state': 'visible' if states[frame_id] == 'available' else 'suppressed',
                'failure_reason': None,
            })
        results = {
            'object': 'keyboard', 'mode': 'complete', 'automatic': True,
            'diagnostic_control': False, 'complete': True, 'frame_ids': self.ids,
            'provenance': run_provenance,
            'tracking_settings': {'min_correspondences': 24, 'threshold': 0.9},
            'initialization': {'frameId': self.setup_id, 'mask_state': states[self.setup_id]},
            'frames': results_frames,
        }
        self.results_path.write_text(json.dumps(results), encoding='utf-8')

        mask_frames = []
        detections = detections or {}
        for frame_id in [self.setup_id] + self.ids:
            row = {
                'frameId': frame_id,
                'mask_state': states[frame_id],
                'path': f'masks/{frame_id}.png',
                'timings_ms': {'segmentation': 3.0},
                'automatic_detection_required': False,
            }
            row.update(detections.get(frame_id, {}))
            mask_frames.append(row)
        mask_result = {
            'frames': mask_frames,
            'provenance': cache_provenance if mask_provenance else None,
        }
        self.mask_results_path.write_text(json.dumps(mask_result), encoding='utf-8')
        self.write_annotations()

    def write_annotations(self):
        annotations = {
            'object': 'keyboard',
            'resolution': [self.width, self.height],
            'selection_status': 'reviewed_image_only',
            'operator': 'Synthetic test reviewer',
            'source_sha256': sha((self.bundle / 'source.mp4').read_bytes()),
            'frames': self.annotations,
        }
        self.annotations_path.write_text(json.dumps(annotations), encoding='utf-8')

    def run_audit(self):
        return audit.audit_cached_masks(
            'keyboard', self.input_path, self.results_path, self.mask_results_path,
            self.mask_root, self.annotations_path)

    def test_frozen_selection_literals_match_canonical_candidate_files(self):
        repository = Path(__file__).resolve().parents[1]
        for object_name, selection in self.canonical_selection_constants.items():
            path = repository / 'artifacts' / 'model-quality' / 'annotations' / object_name / 'annotations.json'
            canonical = json.loads(path.read_text(encoding='utf-8'))
            expected = [(frame_id, group, f'{frame_id}.jpg') for frame_id, group in selection]
            actual = [(row['frame_id'], row['group'], row['image']) for row in canonical['frames']]
            self.assertEqual(actual, expected, object_name)

    def test_nonempty_wrong_object_and_hand_mask_stays_available_and_is_flagged(self):
        target = np.zeros((self.height, self.width), np.uint8)
        target[3:10, 3:10] = 255
        hands = np.zeros_like(target)
        hands[3:10, 12:19] = 255
        predicted = target.copy()
        predicted[hands > 0] = 255
        self.masks[9] = predicted
        self.annotations[0] = self.annotation_row(9, 'reviewed', polygon(rectangle(3, 3, 9, 9)),
                                                   polygon(rectangle(12, 3, 18, 9)))
        self.write_sources()

        report = self.run_audit()

        self.assertEqual(report['frames'][0]['mask_state'], 'available')
        self.assertLess(report['frames'][0]['metrics']['iou'], 0.90)
        self.assertGreater(report['frames'][0]['metrics']['hand_leakage'], 0.01)
        deficient = report['available_but_deficient_reviewed_masks']
        self.assertEqual(deficient['count'], 1)
        self.assertEqual(deficient['denominator_available_reviewed'], 1)
        self.assertEqual(deficient['frame_ids_and_reasons'][0]['frame_id'], 9)
        self.assertEqual(report['mask_state_counts_over_original_240']['available'], 4)
        self.assertFalse(report['mask_gate_passed'])

    def test_empty_lost_prediction_against_visible_truth_is_zero_iou_and_boundary_failure(self):
        self.masks[9] = np.zeros((self.height, self.width), np.uint8)
        self.annotations[0] = self.annotation_row(9, 'reviewed', polygon(rectangle(3, 3, 9, 9)))
        self.write_sources()

        report = self.run_audit()
        metrics = report['frames'][0]['metrics']
        self.assertEqual(report['frames'][0]['mask_state'], 'lost')
        self.assertEqual(metrics['iou'], 0.0)
        self.assertIsNone(metrics['boundary_p95_720'])
        self.assertTrue(metrics['boundary_failure'])
        self.assertEqual(report['mask_scores']['boundary_failure_frames'], 1)

    def test_lost_mask_with_small_residual_is_hashed_but_scored_as_empty(self):
        residual = np.zeros((self.height, self.width), np.uint8)
        residual[0:2, 0:5] = 255
        self.masks[9] = residual
        self.annotations[0] = self.annotation_row(9, 'reviewed', polygon(rectangle(3, 3, 9, 9)))
        self.write_sources()

        report = self.run_audit()
        frame = report['frames'][0]
        self.assertEqual(frame['mask_state'], 'lost')
        self.assertEqual(frame['raw_mask_artifact_pixels'], 10)
        self.assertEqual(frame['predicted_pixels'], 0)
        self.assertEqual(frame['metrics']['predicted_pixels'], 0)
        self.assertEqual(frame['metrics']['iou'], 0.0)
        self.assertEqual(report['lost_mask_residual_artifact_pixels_by_frame'], [{'frame_id': 9, 'pixels': 10}])

    def test_small_successful_cnos_mask_remains_available(self):
        detected = np.zeros((self.height, self.width), np.uint8)
        detected[0:2, 0:5] = 255
        self.masks[9] = detected
        self.annotations[0] = self.annotation_row(9, 'reviewed', polygon(rectangle(20, 15, 28, 21)))
        self.write_sources(state_overrides={9: 'available'})

        report = self.run_audit()
        frame = report['frames'][0]
        self.assertEqual(frame['mask_state'], 'available')
        self.assertEqual(frame['raw_mask_artifact_pixels'], 10)
        self.assertEqual(frame['predicted_pixels'], 10)
        self.assertEqual(frame['metrics']['predicted_pixels'], 10)
        self.assertEqual(frame['metrics']['iou'], 0.0)
        self.assertEqual(report['available_but_deficient_reviewed_masks']['count'], 1)

    def test_fully_hidden_requires_explicit_reviewed_absence_and_counts_false_foreground(self):
        self.annotations[0] = self.annotation_row(9, 'reviewed', visibility='fully_hidden')
        self.write_sources()

        report = self.run_audit()
        hidden = report['fully_hidden_reviewed_frames']
        self.assertEqual(hidden['count'], 1)
        self.assertEqual(hidden['denominator_reviewed'], 1)
        self.assertEqual(hidden['false_foreground_frame_count'], 1)
        self.assertEqual(hidden['false_foreground_pixels_by_frame'][0]['pixels'], 25)
        self.assertEqual(report['frames'][0]['metrics']['iou'], 0.0)

    def test_empty_reviewed_target_without_explicit_absence_is_rejected(self):
        self.annotations[0] = self.annotation_row(9, 'reviewed')
        self.write_sources()
        with self.assertRaisesRegex(ValueError, 'explicit visibility'):
            self.run_audit()

    def test_pending_and_sparse_labels_do_not_become_recovery_proof(self):
        self.annotations[0] = self.annotation_row(9, 'reviewed', polygon(rectangle(3, 3, 7, 7)))
        self.annotations[1] = self.annotation_row(10, 'pending', polygon(rectangle(3, 3, 7, 7)))
        self.annotations[2] = self.annotation_row(11, 'reviewed', polygon(rectangle(3, 3, 7, 7)))
        self.write_sources(detections={
            9: {'timings_ms': {'segmentation': 3.0, 'detection': 1.25}, 'automatic_detection_reason': 'fixture',
                'automatic_detection_required': False},
            10: {'automatic_detection_required': True},
        })

        report = self.run_audit()
        self.assertEqual(report['reviewed_frame_ids'], [9, 11])
        self.assertEqual(report['pending_candidate_frame_ids'], [10])
        self.assertEqual(report['unreviewed_frame_ids'], [10, 12])
        self.assertEqual(report['mask_scores']['frames'], 2)
        self.assertFalse(report['independent_accuracy_ready'])
        self.assertFalse(report['mask_gate_passed'])
        self.assertFalse(report['overall_gate_passed'])
        self.assertFalse(report['temporal_evidence']['sparse_labels_are_recovery_proof'])
        self.assertEqual(report['automatic_detection_evidence']['recorded_evidence_frame_ids'], [9])
        self.assertEqual(report['automatic_detection_evidence']['unknown_frame_ids'], [10, 11, 12])
        self.assertEqual(report['frames'][1]['detection_evidence'], 'unknown')
        self.assertTrue(report['frames'][1]['automatic_detection_required_raw'])

    def test_zero_reviewed_labels_produce_null_scores_and_no_pass(self):
        report = self.run_audit()
        self.assertEqual(report['annotation_frames_reviewed'], 0)
        self.assertIsNone(report['mask_scores']['mean_iou'])
        self.assertIsNone(report['mask_scores']['worst_boundary_p95_720'])
        self.assertIsNone(report['mask_scores']['mean_hand_leakage'])
        self.assertFalse(report['independent_accuracy_ready'])
        self.assertFalse(report['mask_gate_passed'])

    def test_missing_and_corrupt_mask_artifacts_are_errors_even_for_pending_frames(self):
        (self.mask_root / 'masks' / '10.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing cached mask'):
            self.run_audit()

        self.write_sources()
        self.save_mask(10, corrupt=b'not a PNG')
        with self.assertRaisesRegex(ValueError, 'Corrupt cached mask'):
            self.run_audit()

    def test_wrong_native_resolution_and_frame_hash_mismatch_are_rejected(self):
        wrong_shape = np.zeros((self.height - 1, self.width), np.uint8)
        wrong_shape[1:4, 1:4] = 255
        self.masks[10] = wrong_shape
        self.write_sources()
        with self.assertRaisesRegex(ValueError, 'not native resolution'):
            self.run_audit()

        self.masks[10] = np.zeros((self.height, self.width), np.uint8)
        self.write_sources()
        results = json.loads(self.results_path.read_text())
        results['frames'][1]['mask_sha256'] = '0' * 64
        self.results_path.write_text(json.dumps(results), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash does not match'):
            self.run_audit()

    def test_global_provenance_differences_are_recorded_but_ordered_ids_still_enforced(self):
        self.write_sources()
        cache = json.loads(self.mask_results_path.read_text())
        cache['provenance']['input_manifest_sha256'] = '0' * 64
        cache['provenance']['adapter_sha256']['quality_sam2'] = 'later-sam-source'
        cache['provenance']['source_revisions']['sam2'] = 'later-sam-revision'
        cache['provenance']['submodule_revisions']['dinov2'] = 'later-dino-revision'
        cache['provenance']['checkpoint_sha256']['sam2.pt'] = 'later-sam-checkpoint'
        self.mask_results_path.write_text(json.dumps(cache), encoding='utf-8')
        report = self.run_audit()
        comparison = report['frozen_provenance']['segmentation_provenance_comparison']
        differences = comparison['recorded_metadata_differences']
        self.assertFalse(comparison['pose_run_global_metadata_compared_as_mask_generation_provenance'])
        self.assertIn('input_manifest_sha256', differences)
        self.assertIn('adapter_sha256', differences)
        self.assertIn('source_revisions', differences)
        self.assertIn('submodule_revisions', differences)
        self.assertIn('checkpoint_sha256', differences)
        self.assertEqual(comparison['explicit_mask_generation_claim']['present'], False)

        self.write_sources()
        results = json.loads(self.results_path.read_text())
        results['frames'][0], results['frames'][1] = results['frames'][1], results['frames'][0]
        self.results_path.write_text(json.dumps(results), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'ordered'):
            self.run_audit()

    def test_explicit_mask_generation_provenance_claim_is_checked_against_actual_inputs(self):
        self.write_sources()
        cache = json.loads(self.mask_results_path.read_text())
        cache['mask_generation_provenance'] = {'input_manifest_sha256': '0' * 64}
        self.mask_results_path.write_text(json.dumps(cache), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Explicit mask-generation provenance'):
            self.run_audit()

    def test_replacing_one_candidate_with_an_unselected_original_frame_is_rejected(self):
        self.annotations[2]['frame_id'] = 12
        self.annotations[2]['image'] = '12.jpg'
        self.write_annotations()
        with self.assertRaisesRegex(ValueError, 'immutable original candidate selection'):
            self.run_audit()

        self.annotations = [
            self.annotation_row(frame_id, 'pending')
            for frame_id, _ in audit.FROZEN_ANNOTATION_SELECTIONS['keyboard']
        ]
        self.annotations[0]['group'] = 'hard_candidate'
        self.write_annotations()
        with self.assertRaisesRegex(ValueError, 'group or original RGB image changed'):
            self.run_audit()

    def test_absent_segmentation_provenance_is_tied_by_every_frame_hash(self):
        self.write_sources(mask_provenance=False)
        report = self.run_audit()
        self.assertIn('PNG hashes against the automatic result', report['frozen_provenance']['mask_cache_match_basis'])
        self.assertIsNone(report['frozen_provenance']['segmentation_cache_provenance'])

    def test_explicit_diagnostic_or_nonautomatic_mask_index_is_rejected(self):
        self.write_sources(mask_provenance=False)
        cache = json.loads(self.mask_results_path.read_text())
        cache['diagnostic_control'] = True
        self.mask_results_path.write_text(json.dumps(cache), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Diagnostic or explicitly non-automatic'):
            self.run_audit()

    def test_matching_stress_test_run_and_mask_index_are_rejected(self):
        stress = {'occlusion_start': 10, 'source_frames': 15, 'input': 'black RGB frames'}
        self.write_sources()
        results = json.loads(self.results_path.read_text())
        cache = json.loads(self.mask_results_path.read_text())
        results['stress_test'] = stress
        cache['stress_test'] = stress
        self.results_path.write_text(json.dumps(results), encoding='utf-8')
        self.mask_results_path.write_text(json.dumps(cache), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Stress-test runs'):
            self.run_audit()

        self.write_sources()
        cache = json.loads(self.mask_results_path.read_text())
        cache['stress_test'] = stress
        self.mask_results_path.write_text(json.dumps(cache), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Stress-test runs'):
            self.run_audit()

        self.write_sources()
        results = json.loads(self.results_path.read_text())
        results['stress_test'] = stress
        self.results_path.write_text(json.dumps(results), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Stress-test runs'):
            self.run_audit()

        self.write_sources(mask_provenance=False)
        cache = json.loads(self.mask_results_path.read_text())
        cache['automatic'] = False
        self.mask_results_path.write_text(json.dumps(cache), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Diagnostic or explicitly non-automatic'):
            self.run_audit()

    def test_new_report_output_never_overwrites_an_existing_file(self):
        target = self.root / 'report.json'
        target.write_text('baseline')
        with self.assertRaises(FileExistsError):
            audit._write_new_report(target, {'new': True})
        self.assertEqual(target.read_text(), 'baseline')


if __name__ == '__main__':
    unittest.main()
