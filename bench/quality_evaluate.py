"""Independent image annotation evaluator. Never imported by inference stages."""
import argparse
import json
from pathlib import Path
import numpy as np
from .quality_contract import checked_pose, project
from .vision import cv2


def polygon_mask(paths, shape):
    mask = np.zeros(shape, np.uint8)
    for hole in (False, True):
        for path in paths:
            points = path['points'] if isinstance(path, dict) else path
            is_hole = path.get('hole', False) if isinstance(path, dict) else False
            if is_hole == hole and len(points) >= 3:
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 0 if hole else 255)
    return mask > 0


def mask_metrics(predicted, truth, hands):
    a, b, hand = (np.asarray(m, bool) for m in (predicted, truth, hands))
    if not a.shape == b.shape == hand.shape: raise ValueError('Mask resolutions disagree')
    union = np.count_nonzero(a | b)
    iou = np.count_nonzero(a & b)/union if union else 1.
    leakage = np.count_nonzero(a & hand)/max(1, np.count_nonzero(a))
    kernel = np.ones((3, 3), np.uint8)
    edge_a = a & ~cv2.erode(a.astype(np.uint8), kernel).astype(bool)
    edge_b = b & ~cv2.erode(b.astype(np.uint8), kernel).astype(bool)
    if not edge_a.any() and not edge_b.any(): error = 0.
    elif not edge_a.any() or not edge_b.any(): error = None
    else:
        da = cv2.distanceTransform((~edge_a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        db = cv2.distanceTransform((~edge_b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        error = float(np.percentile(np.r_[da[edge_b], db[edge_a]], 95)*720/a.shape[0])
    return dict(iou=float(iou), boundary_p95_720=error, hand_leakage=float(leakage),
                predicted_pixels=int(a.sum()), boundary_failure=error is None)


def evaluate(results, annotations, masks_root, intrinsics, expected_ids):
    # Import lazily to keep annotation and evaluator helpers out of inference
    # imports while validating all human labels before any scoring begins.
    from .quality_annotations import FROZEN_SCORED_FRAME_IDS, normalize_annotation_document
    annotations = normalize_annotation_document(annotations)
    frames = results['frames']
    ids = [f['frameId'] for f in frames]
    frozen_ids = list(FROZEN_SCORED_FRAME_IDS[annotations['object']])
    if expected_ids != frozen_ids or ids != frozen_ids:
        raise ValueError('Every frozen original frame must be present in order, including failures')
    result_id_set = set(ids)
    if len(ids) != len(result_id_set) or any(row['frame_id'] not in result_id_set for row in annotations['frames']):
        raise ValueError('Canonical annotation selection must fall inside the scored original window')
    if not results.get('automatic') or results.get('diagnostic_control'):
        scope = 'diagnostic control; excluded from automatic-pipeline gate'
    else: scope = 'automatic pipeline after setup'
    mask_reviewed = [a for a in annotations['frames'] if a['status'] == 'reviewed']
    pending = [a for a in annotations['frames'] if a['status'] == 'pending']
    visible_reviewed = [a for a in mask_reviewed if a['visibility'] == 'visible']
    hidden_reviewed = [a for a in mask_reviewed if a['visibility'] == 'fully_hidden']
    landmark_reviewed = [a for a in annotations['frames'] if a['landmark_status'] == 'reviewed']
    landmark_unobservable = [a for a in annotations['frames'] if a['landmark_status'] == 'unobservable']
    landmark_pending = [a for a in annotations['frames'] if a['landmark_status'] == 'pending']
    landmark_not_applicable = [a for a in annotations['frames']
                               if a['landmark_status'] == 'not_applicable_fully_hidden']
    by_id = {f['frameId']: f for f in frames}
    details = []
    landmark_details = []
    errors = []
    landmarks_missing = 0
    missing_pose_landmarks = 0
    invalid_projection_landmarks = 0
    landmark_count = 0
    landmark_error_by_frame = {}
    landmark_missing_by_frame = {}
    measured_landmark_frame_ids = []
    hidden_false_foreground_ids = []
    hidden_render_leakage_ids = []
    w, h = annotations['resolution']; accepted = 0; visible = 0
    for f in frames:
        tracking = f['pose_state'] == 'tracking' and f.get('cameraFromObject') is not None
        accepted += tracking; visible += f['render_state'] == 'visible'
        if f['render_state'] == 'visible' and (not tracking or f['mask_state'] != 'available'):
            raise ValueError('Stale/unmasked rendering cannot be counted as valid')

    # Landmark measurements have their own review state. Pending, unobservable,
    # and fully hidden rows add no coverage and never become synthetic labels.
    for a in landmark_reviewed:
        if a['frame_id'] not in by_id: raise ValueError('Annotation outside original window')
        record = by_id[a['frame_id']]
        visible_landmarks = [landmark for landmark in a['landmarks']
                             if landmark.get('visible', True) is True]
        landmark_count += len(visible_landmarks)
        measured_landmark_frame_ids.append(a['frame_id'])
        frame_errors = []
        frame_missing = 0
        frame_missing_pose = 0
        frame_invalid_projection = 0
        if record['pose_state'] != 'tracking' or record.get('cameraFromObject') is None:
            frame_missing = len(visible_landmarks)
            frame_missing_pose = frame_missing
        else:
            points = np.asarray([landmark['object_point_m'] for landmark in visible_landmarks], dtype=np.float64)
            try:
                pose = checked_pose(record['cameraFromObject'])
                uv, z = project(points, pose, intrinsics)
                projection_valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
            except (ValueError, TypeError, IndexError, np.linalg.LinAlgError, FloatingPointError):
                projection_valid = np.zeros(len(visible_landmarks), dtype=bool)
                uv = np.full((len(visible_landmarks), 2), np.nan, dtype=np.float64)
            for index, (landmark, valid) in enumerate(zip(visible_landmarks, projection_valid)):
                if not valid:
                    frame_missing += 1
                    frame_invalid_projection += 1
                    continue
                error = float(np.linalg.norm(uv[index] - landmark['pixel']) * 720 / h)
                if not np.isfinite(error):
                    frame_missing += 1
                    frame_invalid_projection += 1
                    continue
                errors.append(error)
                frame_errors.append(error)
        missing_pose_landmarks += frame_missing_pose
        invalid_projection_landmarks += frame_invalid_projection
        landmarks_missing += frame_missing
        landmark_error_by_frame[a['frame_id']] = frame_errors
        landmark_missing_by_frame[a['frame_id']] = frame_missing
        landmark_details.append(dict(
            frame_id=a['frame_id'], landmark_status=a['landmark_status'],
            measured_landmarks=len(visible_landmarks), missing_landmarks=frame_missing,
            missing_pose_landmarks=frame_missing_pose,
            invalid_projection_landmarks=frame_invalid_projection,
            attachment_errors_720=frame_errors, pose_state=record['pose_state'],
        ))

    for a in mask_reviewed:
        if a['frame_id'] not in by_id: raise ValueError('Annotation outside original window')
        record = by_id[a['frame_id']]
        truth = polygon_mask(a['visible_object'], (h, w))
        hand = polygon_mask(a['overlapping_hands'], (h, w))
        if np.any(truth & hand):
            raise ValueError('Visible object annotations must exclude overlapping hand pixels')
        mask_path = record.get('mask_path')
        mask = cv2.imread(str(masks_root / mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        mask_lost = record['mask_state'] != 'available' or mask is None
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        if mask.shape != (h, w):
            raise ValueError('Native resolution predicted masks required')
        if mask_lost:
            mask[:] = 0
        metrics = mask_metrics(mask, truth, hand)
        visibility = a['visibility']
        if visibility == 'fully_hidden':
            if metrics['predicted_pixels']:
                hidden_false_foreground_ids.append(a['frame_id'])
            # Rendering leakage is about visible render states only. A suppressed
            # render cannot leak even when its mask cache contains foreground.
            if record['render_state'] == 'visible':
                hidden_render_leakage_ids.append(a['frame_id'])
        frame_errors = landmark_error_by_frame.get(a['frame_id'], [])
        details.append(dict(
            frame_id=a['frame_id'], visibility=visibility,
            landmark_status=a['landmark_status'], mask=metrics,
            mask_lost=mask_lost,
            false_foreground_pixels=metrics['predicted_pixels'] if visibility == 'fully_hidden' else None,
            attachment_errors_720=frame_errors,
            attachment_missing_landmarks=landmark_missing_by_frame.get(a['frame_id'], 0),
            pose_state=record['pose_state'], render_state=record['render_state']))

    mask_rows = [d['mask'] for d in details]
    visible_mask_rows = [d['mask'] for d in details if d['visibility'] == 'visible']
    average_iou = float(np.mean([m['iou'] for m in mask_rows])) if mask_rows else None
    boundary = max((m['boundary_p95_720'] for m in mask_rows if m['boundary_p95_720'] is not None), default=None)
    leakage = float(np.mean([m['hand_leakage'] for m in mask_rows])) if mask_rows else None
    visible_average_iou = float(np.mean([m['iou'] for m in visible_mask_rows])) if visible_mask_rows else None
    visible_boundary = max((m['boundary_p95_720'] for m in visible_mask_rows
                            if m['boundary_p95_720'] is not None), default=None)
    visible_leakage = float(np.mean([m['hand_leakage'] for m in visible_mask_rows])) if visible_mask_rows else None
    median = float(np.median(errors)) if errors else None
    p95 = float(np.percentile(errors, 95)) if errors else None
    selection_ready = (annotations.get('selection_status') == 'reviewed_image_only' and
                       bool(annotations.get('operator')))
    mask_labels_ready = len(mask_reviewed) == 40 and selection_ready
    attachment_labels_ready = len(set(measured_landmark_frame_ids)) == 40 and selection_ready
    independent_ready = mask_labels_ready and attachment_labels_ready
    visible_mask_gate = None
    if mask_labels_ready and visible_mask_rows:
        visible_mask_gate = (
            visible_average_iou >= .9 and visible_boundary is not None and visible_boundary <= 3 and
            visible_leakage <= .01 and not any(m['boundary_failure'] for m in visible_mask_rows))
    attachment_gate = None
    if attachment_labels_ready:
        attachment_gate = (
            median is not None and median < 5 and p95 is not None and p95 < 10 and
            landmarks_missing == 0 and invalid_projection_landmarks == 0)
    hidden_false_foreground_gate = None
    hidden_render_leakage_gate = None
    if mask_labels_ready and hidden_reviewed:
        hidden_false_foreground_gate = not hidden_false_foreground_ids
        hidden_render_leakage_gate = not hidden_render_leakage_ids
    visible_mask_lost_ids = [d['frame_id'] for d in details
                             if d['visibility'] == 'visible' and d['mask_lost']]
    visible_mask_empty_ids = [d['frame_id'] for d in details
                              if d['visibility'] == 'visible' and d['mask']['predicted_pixels'] == 0]
    status_counts = {
        'reviewed': len(landmark_reviewed),
        'unobservable': len(landmark_unobservable),
        'pending': len(landmark_pending),
        'not_applicable_fully_hidden': len(landmark_not_applicable),
    }
    gates = dict(mask=visible_mask_gate, visible_mask=visible_mask_gate,
                 hidden_false_foreground=hidden_false_foreground_gate,
                 hidden_render_leakage=hidden_render_leakage_gate,
                 availability=accepted/len(frames) >= .9, attachment=attachment_gate)
    legacy_aggregate_metrics = dict(
        mask_mean_iou=average_iou,
        boundary_worst_frame_p95_720=boundary,
        mean_hand_leakage=leakage,
        attachment_median_720=median,
        attachment_p95_720=p95,
    )
    return dict(schema_version=2, scope=scope, frames=len(frames), failures=len(frames)-accepted,
        accepted_pose_availability=accepted/len(frames), rendering_visibility=visible/len(frames),
        annotation_frames_reviewed=len(mask_reviewed), annotation_frames_required=40,
        mask_labels_ready=mask_labels_ready, attachment_labels_ready=attachment_labels_ready,
        independent_accuracy_ready=independent_ready, combined_independent_accuracy_ready=independent_ready,
        visible_frames_reviewed=len(visible_reviewed), visible_reviewed_frame_ids=[a['frame_id'] for a in visible_reviewed],
        fully_hidden_frames_reviewed=len(hidden_reviewed), fully_hidden_reviewed_frame_ids=[a['frame_id'] for a in hidden_reviewed],
        pending_annotation_frames=len(pending), pending_annotation_frame_ids=[a['frame_id'] for a in pending],
        landmark_review_status_counts=status_counts,
        landmark_frames_reviewed=len(landmark_reviewed),
        landmark_reviewed_frame_ids=[a['frame_id'] for a in landmark_reviewed],
        landmark_unobservable_frame_ids=[a['frame_id'] for a in landmark_unobservable],
        landmark_pending_frame_ids=[a['frame_id'] for a in landmark_pending],
        landmark_not_applicable_fully_hidden_frame_ids=[a['frame_id'] for a in landmark_not_applicable],
        landmark_frame_coverage=len(set(measured_landmark_frame_ids))/40,
        fully_hidden_false_foreground_frames=len(hidden_false_foreground_ids),
        fully_hidden_false_foreground_frame_ids=hidden_false_foreground_ids,
        fully_hidden_false_foreground_reviewed_denominator=len(hidden_reviewed),
        fully_hidden_render_leakage_frames=len(hidden_render_leakage_ids),
        fully_hidden_render_leakage_frame_ids=hidden_render_leakage_ids,
        fully_hidden_render_leakage_reviewed_denominator=len(hidden_reviewed),
        mask_mean_iou=average_iou, boundary_worst_frame_p95_720=boundary, mean_hand_leakage=leakage,
        attachment_median_720=median, attachment_p95_720=p95,
        legacy_aggregate_metrics=legacy_aggregate_metrics,
        visible_mask_reviewed_frames=len(visible_mask_rows),
        visible_mask_reviewed_frame_ids=[d['frame_id'] for d in details if d['visibility'] == 'visible'],
        visible_mask_mean_iou=visible_average_iou,
        visible_mask_boundary_worst_frame_p95_720=visible_boundary,
        visible_mask_mean_hand_leakage=visible_leakage,
        visible_mask_lost_frames=len(visible_mask_lost_ids), visible_mask_lost_frame_ids=visible_mask_lost_ids,
        visible_mask_empty_prediction_frames=len(visible_mask_empty_ids),
        visible_mask_empty_prediction_frame_ids=visible_mask_empty_ids,
        visible_landmark_labels=landmark_count,
        landmark_frames_with_accepted_projection=len({row['frame_id'] for row in landmark_details
                                                       if row['measured_landmarks'] > row['missing_landmarks']}),
        visible_landmarks_without_accepted_pose=landmarks_missing,
        landmarks_without_accepted_pose=missing_pose_landmarks,
        visible_landmarks_with_invalid_projection=invalid_projection_landmarks,
        attachment_missing_landmarks=landmarks_missing,
        attachment_missing_frames=sum(row['missing_landmarks'] > 0 for row in landmark_details),
        gates=gates, overall_gate_passed=False,
        remaining_gates=['forced occlusion recovery', 'model prefix invariance', 'frozen additional windows'],
        details=details, landmark_details=landmark_details)


def prefix_equal(short, long):
    """Timings may vary; masks, states, failures and poses must not."""
    earlier = long['frames'][:len(short['frames'])]
    if len(earlier) != len(short['frames']): return False
    for a, b in zip(short['frames'], earlier):
        if not a.get('mask_sha256') or not b.get('mask_sha256'): return False
        for key in ('frameId', 'mask_state', 'pose_state', 'render_state', 'failure_reason', 'mask_sha256'):
            if a.get(key) != b.get(key): return False
        pa, pb = a.get('cameraFromObject'), b.get('cameraFromObject')
        if (pa is None) != (pb is None): return False
        if pa is not None and not np.allclose(pa, pb, atol=1e-6, rtol=0): return False
    return True


def secondary_pose_agreement(results, original, mesh, visibility_cache=None):
    """Estimated dataset labels are secondary, never the independent gate."""
    from .show3d_experiment import evaluation_truth
    k = np.array(original['cameraCalibration']).reshape(3, 3)
    points = np.linspace(0, len(mesh.positions)-1, min(128, len(mesh.positions)), dtype=int)
    reference = {r['frameId']: r for r in original['referenceFrames']}
    errors = []; reference_frames = 0
    for frame in results['frames']:
        control = reference[frame['frameId']]['cameraFromObject']
        if control is None: continue
        reference_frames += 1
        if frame['pose_state'] != 'tracking' or frame.get('cameraFromObject') is None: continue
        p = np.array(control)
        key = (id(mesh), frame['frameId'], p.tobytes(), k.tobytes(), tuple(original['nativeResolution']))
        if visibility_cache is not None and key in visibility_cache:
            ids, expected = visibility_cache[key]
        else:
            ids, expected = evaluation_truth(mesh, points, (p[:3, :3], p[:3, 3]), k, *original['nativeResolution'])
            if visibility_cache is not None: visibility_cache[key] = (ids, expected)
        predicted, _ = project(mesh.positions[points], checked_pose(frame['cameraFromObject']), k)
        errors.extend((np.linalg.norm(predicted[ids]-expected[ids], axis=1)*720/original['nativeResolution'][1]).tolist())
    return dict(reference_frames=reference_frames, accepted_point_samples=len(errors),
        median_720=float(np.median(errors)) if errors else None, p95_720=float(np.percentile(errors, 95)) if errors else None,
        scope='Secondary agreement with SHOW3D estimated poses from related model family; self-visibility only, no hand ground truth. Never used to pass independent attachment gate.')


def export_oracle_masks(predicted_root, annotation, output):
    """Evaluator-only diagnostic controls. Inference receives mask pixels only."""
    import shutil
    from .quality_annotations import validate_annotation_document
    annotation = validate_annotation_document(annotation)
    output.mkdir(parents=True, exist_ok=True); (output/'masks').mkdir(exist_ok=True)
    source = json.loads((predicted_root/'results.json').read_text())
    reviewed = {a['frame_id']: a for a in annotation['frames'] if a['status'] == 'reviewed'}
    w, h = annotation['resolution']
    for record in source['frames']:
        target = output/record['path']; target.parent.mkdir(parents=True, exist_ok=True)
        if record['frameId'] in reviewed:
            a = reviewed[record['frameId']]
            truth = polygon_mask(a['visible_object'], (h, w))
            if np.any(truth & polygon_mask(a['overlapping_hands'], (h, w))): raise ValueError('Object annotation overlaps hand')
            cv2.imwrite(str(target), truth.astype(np.uint8)*255)
            record['mask_state'] = 'available' if truth.any() else 'lost'
            record['failure_reason'] = None if truth.any() else 'diagnostic_annotated_full_occlusion'
            record['diagnostic_corrected_mask'] = True
        else: shutil.copyfile(predicted_root/record['path'], target)
    source.update(diagnostic_control=True, automatic=False, corrected_frames=sorted(reviewed),
                  scope='Corrected masks on independently reviewed frames only; never an automatic benchmark result.')
    (output/'results.json').write_text(json.dumps(source, indent=2))


def occlusion_recovery(results, start, count=15, deadline=30):
    hidden = [r for r in results['frames'] if start <= r['frameId'] < start+count]
    suppressed = len(hidden) == count and all(r['render_state'] == 'suppressed' for r in hidden)
    returned = next((r['frameId'] for r in results['frames'] if r['frameId'] >= start+count and r['pose_state'] == 'tracking'), None)
    latency = None if returned is None else returned-(start+count)
    return dict(hidden_frames=len(hidden), hidden_rendering_suppressed=suppressed, recovery_source_frames=latency,
                passed=suppressed and latency is not None and latency <= deadline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True); parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True); parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); m = json.loads(a.input.read_text())
    score = evaluate(json.loads(a.results.read_text()), json.loads(a.annotations.read_text()), a.masks, np.array(m['intrinsics']), m['frame_ids'])
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(score, indent=2, allow_nan=False))
