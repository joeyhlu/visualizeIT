"""CPU-only fixtures for the opt-in chronological mug observer."""
import json
import random
import copy
import builtins
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from .quality_assets import ROOT, digest
from .quality_contract import Frame
from .quality_mug_chronological_capture import (
    BRANCH_GUESS, BRANCHES, CACHE_PREPARE_RESERVATION_BYTES, CAPTURE_BUDGET_BYTES,
    CAPTURE_EXPERIMENT, CAPTURE_FRAME_IDS, CAPTURE_SCHEMA_VERSION,
    MAX_FULL_CORRESPONDENCES, MAX_NATIVE_CONTEXTS, MAX_NATIVE_CONTEXT_BYTES,
    MAX_PACKET_PAIR_BYTES, MAX_PACKET_PAIRS, MAX_PLANNED_CAPTURE_BYTES,
    MAX_PLANNED_CAPTURE_HEADROOM_BYTES, MAX_RETAINED_PACKET_BYTES,
    MAX_RESULT_PACKET_BYTES, MAX_SAMPLED_CORRESPONDENCES,
    METADATA_ARTIFACT_BUDGET_BYTES, MODEL_CACHE_BUDGET_BYTES,
    OLD_V1_CAPTURE_BUDGET_BYTES, WORKING_COPY_BUDGET_BYTES,
    RUN_FRAME_IDS, SETUP_FRAME_ID, MugChronologicalCapture, TraceOrdinalReader,
    _compare_trace_records, _frame_equal_prefix, _json_sha, _trace_records,
    _check_metadata_artifact_budget, _existing_cache_bytes, _metadata_artifact_bytes,
    _load_bound_packet, _native_context_bytes, _native_context_descriptor_bytes, _packet_descriptor_bytes,
    _preflight_model_cache_budget,
    _working_copy_peak_bytes,
    _validate_capture_branch, _validate_compatibility_review_record,
    _validate_legacy_segmentation_record, record_terminal_exit,
)
from . import quality_mug_chronological_capture as capture_module


def _pose(translation):
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = translation
    return value


def _comparison_frame(frame_id):
    return dict(frameId=frame_id, cameraFromObject=np.eye(4).tolist(),
        mask_state='available', pose_state='tracking', render_state='visible',
        failure_reason=None, timings_ms={'pose_total': 1.0},
        validation=dict(correspondences=26480, inliers=26480,
            median_reprojection_720=0.6475620681989869,
            p95_reprojection_720=1.6506356413748013,
            spatial_support=0.6913144266570733, score=0.974624663146826,
            model_motion=dict(state='disabled')),
        rejection_reasons=[], mask_path=f'masks/{frame_id}.png',
        mask_sha256=f'mask-{frame_id}')


def _base_manifest():
    return dict(
        schema_version=CAPTURE_SCHEMA_VERSION,
        experiment=CAPTURE_EXPERIMENT,
        status='prepared',
        setup_frame_id=SETUP_FRAME_ID,
        frame_ids=list(RUN_FRAME_IDS),
        capture_frame_ids=list(CAPTURE_FRAME_IDS),
        capture_iteration=4,
        branch_pnp_use_extrinsic_guess=dict(BRANCH_GUESS),
        capture_budget_bytes=CAPTURE_BUDGET_BYTES,
        metadata_artifact_budget_bytes=METADATA_ARTIFACT_BUDGET_BYTES,
        working_copy_budget_bytes=WORKING_COPY_BUDGET_BYTES,
        copied_array_bytes=0,
        capture_stopped=False,
        contexts={},
        branches={branch: dict(status='prepared', exit_code=None, frame_records=[])
                  for branch in BRANCHES},
    )


def _runtime_binding_fixture(root, capture_root):
    """Small complete source/mask fixture for branch-boundary freeze checks."""
    from .vision import cv2

    root = Path(root).resolve()
    spec_path = root / capture_module.CAPACITY_SPEC_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_bytes((Path(ROOT) / capture_module.CAPACITY_SPEC_PATH).read_bytes())
    bundle = root / '.cache/model-quality/inputs/mug'
    masks_root = root / '.cache/model-quality/results/mug/segmentation'
    bundle.mkdir(parents=True)
    masks_root.mkdir(parents=True)
    input_path = bundle / 'input.json'
    input_path.write_text('{"object":"mug","units":"metres"}', encoding='utf-8')

    producer = root / capture_module.LEGACY_SAM2_PRODUCER_PATH
    producer.parent.mkdir(parents=True)
    producer.write_bytes(b'fixture legacy SAM producer')
    producer_sha = digest(producer)
    provenance = dict(input_manifest_sha256=digest(input_path),
        adapter_sha256={'quality_sam2': producer_sha})

    frames = []
    mask_hashes = {}
    review_rows = []
    for frame_id in range(SETUP_FRAME_ID, 1067):
        relative = f'masks/{frame_id}.png'
        path = masks_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((3, 5), 255, dtype=np.uint8))
        sha = digest(path)
        mask_hashes[str(frame_id)] = sha
        frames.append(dict(frameId=frame_id, path=relative, mask_state='available',
            failure_reason=None, automatic_detection_required=False))
        review_rows.append(dict(frame_id=frame_id, path=relative, sha256=sha,
            width=5, height=3, state='available'))
    segmentation = dict(provenance=provenance, frames=frames)
    segmentation_path = masks_root / 'results.json'
    segmentation_path.write_text(json.dumps(segmentation), encoding='utf-8')

    v2_root = root / '.cache/model-quality/diagnostics/v2'
    v2_root.mkdir(parents=True)
    v2_experiment = dict(segmentation_provenance=provenance,
        mask_sha256_by_frame=mask_hashes)
    (v2_root / 'experiment.json').write_text(json.dumps(v2_experiment), encoding='utf-8')
    (v2_root / 'stage-evidence.json').write_text('{}', encoding='utf-8')
    (v2_root / 'report.json').write_text('{}', encoding='utf-8')

    source_bench = root / 'bench'
    source_bench.mkdir()
    repo_bench = Path(capture_module.__file__).resolve().parent
    for name in ('quality_gotrack.py', 'quality_runner.py', 'quality_mug_chronological_capture.py'):
        (source_bench / name).write_bytes((repo_bench / name).read_bytes())
    contract_source = source_bench / 'quality_contract.py'
    contract_source.write_bytes(b'fixture unchanged source')

    bank = root / '.cache/model-quality/banks/mug-foundpose.pt'
    bank.parent.mkdir(parents=True)
    bank.write_bytes(b'fixture foundpose bank')
    review_path = root / '.cache/model-quality/reviews/sol-compatibility.json'
    review_path.parent.mkdir(parents=True)
    legacy_review = dict(
        manifest_path=segmentation_path.relative_to(root).as_posix(),
        manifest_sha256=digest(segmentation_path),
        input_manifest_sha256=digest(input_path),
        producer_path=capture_module.LEGACY_SAM2_PRODUCER_PATH,
        producer_sha256=producer_sha,
        provenance_sha256=_json_sha(provenance),
        missing_flags=list(capture_module.LEGACY_MISSING_MODE_FLAGS),
        mask_rows=review_rows)
    review_path.write_text(json.dumps(dict(legacy_segmentation=legacy_review)), encoding='utf-8')

    frozen_files = {
        input_path.relative_to(root).as_posix(): digest(input_path),
        segmentation_path.relative_to(root).as_posix(): digest(segmentation_path),
        bank.relative_to(root).as_posix(): digest(bank),
    }
    for row in frames:
        path = masks_root / row['path']
        frozen_files[path.relative_to(root).as_posix()] = digest(path)
    current_source_hashes = {
        'bench/quality_contract.py': digest(contract_source),
        'bench/quality_gotrack.py': digest(source_bench / 'quality_gotrack.py'),
        'bench/quality_runner.py': digest(source_bench / 'quality_runner.py'),
    }
    current_source_sha256 = {name: current_source_hashes[name]
        for name in ('bench/quality_gotrack.py', 'bench/quality_runner.py')}
    run_provenance = dict(input_manifest_sha256=digest(input_path), adapter_sha256={},
        source_revisions={}, submodule_revisions={}, checkpoint_sha256={})
    with mock.patch.object(capture_module, 'LEGACY_SAM2_PRODUCER_SHA256', producer_sha):
        legacy_summary = _validate_legacy_segmentation_record(
            dict(legacy_segmentation=legacy_review), segmentation, segmentation_path,
            bundle, masks_root, v2_experiment, root)
    bindings = dict(
        source_root=str(root), bundle_root=str(bundle), masks_root=str(masks_root),
        v2_root=str(v2_root),
        current_source_hashes=current_source_hashes,
        current_source_sha256=current_source_sha256,
        frozen_files=frozen_files,
        legacy_review_record_path=review_path.relative_to(root).as_posix(),
        legacy_review_record_sha256=digest(review_path),
        v2_experiment_sha256=digest(v2_root / 'experiment.json'),
        v2_stage_evidence_sha256=digest(v2_root / 'stage-evidence.json'),
        v2_report_sha256=digest(v2_root / 'report.json'),
        new_capture_module_sha256=digest(source_bench / 'quality_mug_chronological_capture.py'),
        current_inference_provenance=dict(run_provenance,
            foundpose_bank_sha256=digest(bank)),
        mask_sha256_by_frame=mask_hashes,
        legacy_segmentation_review=legacy_summary,
        capacity_spec=dict(path=capture_module.CAPACITY_SPEC_PATH,
            sha256=capture_module.CAPACITY_SPEC_SHA256),
    )
    capture_root = Path(capture_root).resolve()
    capture_root.mkdir(parents=True)
    capture_manifest = _base_manifest()
    capture_manifest['bindings'] = bindings
    (capture_root / 'capture.json').write_text(json.dumps(capture_manifest), encoding='utf-8')
    return dict(root=root, bundle=bundle, masks_root=masks_root,
        capture_root=capture_root, producer_sha=producer_sha, run_provenance=run_provenance,
        review_path=review_path)


def _packet_arrays():
    count = 24
    source_ids = np.arange(count, dtype=np.int64)
    centers = np.column_stack((source_ids % 280 + .5, source_ids // 280 + .5)).astype(np.float64)
    flow = np.zeros((280, 280, 2), dtype=np.float32)
    confidence = np.full((280, 280), .8, dtype=np.float32)
    obj = np.column_stack((np.arange(count), np.arange(count) * .5, np.ones(count) * 20)).astype(np.float64)
    target = centers + flow.reshape(-1, 2)[source_ids]
    c = np.eye(4, dtype=np.float64)
    c[:3, :3] = np.array([[.999, -.03, 0], [.03, .999, 0], [0, 0, 1.]])
    c[:3, 3] = [.4, -.2, .1]
    crop_pose = _pose([2., 3., 650.])
    native_pose_m = np.linalg.inv(c) @ crop_pose
    native_pose_m[:3, 3] *= .001
    sample_ids = np.arange(count, dtype=np.int64)
    retained = dict(
        query_rgb_crop=np.zeros((1, 3, 280, 280), dtype=np.float32),
        template_rgb=np.zeros((1, 3, 280, 280), dtype=np.float32),
        rendered_depth_mm=np.ones((1, 280, 280), dtype=np.float32),
        rendered_mask=np.ones((1, 280, 280), dtype=bool),
        observed_crop_mask=np.ones((1, 280, 280), dtype=np.float32),
        full_flow_crop_px=flow,
        full_confidence=confidence,
        full_obj_points_mm=obj,
        full_target_crop_px=target,
        source_flat_indices=source_ids,
        source_crop_pixel_centers=centers,
        sample_ids=sample_ids,
        sample_obj_points_mm=obj[sample_ids],
        sample_target_crop_px=target[sample_ids],
        sample_weights=confidence.reshape(-1)[source_ids][sample_ids],
        crop_k=np.array([[700., 0., 140.], [0., 700., 140.], [0., 0., 1.]]),
        crop_from_orig=c,
        native_k=np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]]),
        seed_camera_from_object_m=_pose([0., 0., .65]),
        current_crop_camera_from_object_mm=crop_pose,
        initial_rvec=np.zeros((3, 1)),
        initial_tvec_mm=crop_pose[:3, 3].copy(),
    )
    inliers = np.arange(8, dtype=np.int32)
    result = dict(
        ransac_inlier_ids=inliers,
        retained_inlier_ids=sample_ids[inliers],
        refined_rvec=np.zeros((3,), dtype=np.float64),
        refined_tvec_mm=crop_pose[:3, 3].copy(),
        crop_camera_from_object_mm=crop_pose.copy(),
        native_camera_from_object_m=native_pose_m,
    )
    return retained, result


def _array_descriptor(shape, dtype):
    dtype = np.dtype(dtype)
    shape = list(shape)
    return dict(shape=shape, dtype=str(dtype),
                nbytes=int(np.prod(shape, dtype=np.int64)) * dtype.itemsize)


def _maximum_packet_descriptors():
    n, s = MAX_FULL_CORRESPONDENCES, MAX_SAMPLED_CORRESPONDENCES
    retained = {
        'query_rgb_crop': _array_descriptor((1, 3, 280, 280), 'float32'),
        'template_rgb': _array_descriptor((1, 3, 280, 280), 'float32'),
        'rendered_depth_mm': _array_descriptor((1, 280, 280), 'float32'),
        'rendered_mask': _array_descriptor((1, 280, 280), 'uint8'),
        'observed_crop_mask': _array_descriptor((1, 280, 280), 'float32'),
        'full_flow_crop_px': _array_descriptor((280, 280, 2), 'float32'),
        'full_confidence': _array_descriptor((280, 280), 'float32'),
        'full_obj_points_mm': _array_descriptor((n, 3), 'float64'),
        'full_target_crop_px': _array_descriptor((n, 2), 'float64'),
        'source_flat_indices': _array_descriptor((n,), 'int64'),
        'source_crop_pixel_centers': _array_descriptor((n, 2), 'float64'),
        'sample_ids': _array_descriptor((s,), 'int64'),
        'sample_obj_points_mm': _array_descriptor((s, 3), 'float64'),
        'sample_target_crop_px': _array_descriptor((s, 2), 'float64'),
        'sample_weights': _array_descriptor((s,), 'float32'),
        'crop_k': _array_descriptor((3, 3), 'float64'),
        'crop_from_orig': _array_descriptor((4, 4), 'float64'),
        'native_k': _array_descriptor((3, 3), 'float64'),
        'seed_camera_from_object_m': _array_descriptor((4, 4), 'float64'),
        'current_crop_camera_from_object_mm': _array_descriptor((4, 4), 'float64'),
        'initial_rvec': _array_descriptor((3, 1), 'float64'),
        'initial_tvec_mm': _array_descriptor((3,), 'float64'),
    }
    result = {
        'ransac_inlier_ids': _array_descriptor((s,), 'int64'),
        'retained_inlier_ids': _array_descriptor((s,), 'int64'),
        'refined_rvec': _array_descriptor((3,), 'float64'),
        'refined_tvec_mm': _array_descriptor((3,), 'float64'),
        'crop_camera_from_object_mm': _array_descriptor((4, 4), 'float64'),
        'native_camera_from_object_m': _array_descriptor((4, 4), 'float64'),
    }
    return retained, result


def _fake_v2(root):
    experiment = root / 'experiment.json'
    evidence = root / 'stage-evidence.json'
    report = root / 'report.json'
    experiment.write_text('{}', encoding='utf-8')
    evidence.write_text('{}', encoding='utf-8')
    report.write_text('{}', encoding='utf-8')
    provenance = dict(
        input_manifest_sha256='input', source_revisions={}, submodule_revisions={},
        checkpoint_sha256='checkpoint', foundpose_bank_sha256='bank', adapter_sha256={},
    )
    return dict(provenance=provenance, source_delta={}, auxiliary_source_delta={},
        current_source_hashes={}, experiment=dict(source_files={}, frozen_files={}),
        compatibility_review=dict(sha256='review-sha', relative_path='review.json',
            record=dict(review_id='review-v1', approved_delta_sha256='delta-sha')))


def _fake_refinement_inputs():
    """Tiny deterministic CPU GoTrack fixture; no model or renderer is loaded."""
    import torch
    from .vision import cv2

    intrinsics = np.array([[500., 0., 140.], [0., 500., 140.], [0., 0., 1.]], dtype=np.float64)
    pixels = np.zeros((1, 280, 280, 2), dtype=np.float32)
    xyz_camera = np.zeros((1, 280, 280, 3), dtype=np.float32)
    retained_mask = np.zeros((1, 280, 280), dtype=bool)
    for yi, y in enumerate((-30., -20., -10., 0., 10., 20., 30.)):
        for xi, x in enumerate((-36., -24., -12., 0., 12., 24., 36.)):
            z = float(((xi * 3 + yi * 5) % 7) - 3)
            point = np.array([x, y, z], dtype=np.float64)
            uv, _ = cv2.projectPoints(point[None], np.zeros((3, 1)),
                np.array([[0.], [0.], [500.]]), intrinsics, None)
            u, v = uv.reshape(2)
            px, py = int(np.floor(u)), int(np.floor(v))
            retained_mask[0, py, px] = True
            pixels[0, py, px] = (u, v)
            xyz_camera[0, py, px] = point + np.array([0., 0., 500.])
    template = types.SimpleNamespace(
        rgbs=torch.zeros((1, 3, 280, 280)),
        masks=torch.from_numpy(retained_mask),
        depths=torch.zeros((1, 1, 280, 280)),
    )
    data = dict(crop_rgbs=torch.zeros((1, 3, 280, 280)),
        crop_masks=torch.ones((1, 280, 280)), templates=template)
    utils_module = types.ModuleType('utils')
    utils_module.data_util = types.SimpleNamespace(
        compute_gotrack_inputs_from_init_poses=lambda **_: (data, [object()], torch.eye(4)[None]))
    utils_module.misc = types.SimpleNamespace(get_intrinsic_matrix=lambda _: intrinsics)
    utils_module.structs = types.SimpleNamespace(
        PinholePlaneCameraModel=lambda **kwargs: types.SimpleNamespace(**kwargs))
    utils_module.transform3d = types.SimpleNamespace(
        get_3d_points_from_depth=lambda *_: (torch.from_numpy(pixels), torch.from_numpy(xyz_camera)))

    class Renderer:
        vertices_m = np.zeros((1, 3), dtype=np.float64)

    def network(*_):
        return torch.zeros((1, 2, 280, 280)), torch.ones((1, 280, 280))

    frame = Frame(1016, np.zeros((280, 280, 3), dtype=np.uint8), intrinsics)
    mask = np.full((280, 280), 255, dtype=np.uint8)
    seed = np.eye(4, dtype=np.float64)
    seed[2, 3] = .5
    return utils_module, network, Renderer(), frame, mask, seed


def _rng_snapshot(torch):
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _rng_unchanged(torch, state):
    python_before, numpy_before, torch_before = state
    numpy_after = np.random.get_state()
    return (random.getstate() == python_before and numpy_after[0] == numpy_before[0] and
        np.array_equal(numpy_after[1], numpy_before[1]) and numpy_after[2:] == numpy_before[2:] and
        torch.equal(torch.get_rng_state(), torch_before))
    return dict(
        root=root,
        provenance=provenance,
        source_delta={},
        experiment=dict(frozen_files={}, source_files={}),
        evidence={},
    )


class MugChronologicalCaptureTests(unittest.TestCase):
    def test_resource_v2_maximum_descriptor_cohort_fits_without_allocating_packets(self):
        retained_descriptors, result_descriptors = _maximum_packet_descriptors()
        retained_bytes = _packet_descriptor_bytes('retained', {'iteration': 4}, retained_descriptors)
        result_bytes = _packet_descriptor_bytes('result', {'fit_state': 'refined'}, result_descriptors)
        context_bytes = _native_context_descriptor_bytes((1024, 1280, 3), 'uint8',
            (1024, 1280), 'uint8', (3, 3), 'float64')
        self.assertLessEqual(retained_bytes, MAX_RETAINED_PACKET_BYTES)
        self.assertLessEqual(result_bytes, MAX_RESULT_PACKET_BYTES)
        self.assertLessEqual(retained_bytes + result_bytes, MAX_PACKET_PAIR_BYTES)
        self.assertEqual(context_bytes, MAX_NATIVE_CONTEXT_BYTES)
        planned = MAX_PACKET_PAIRS * MAX_PACKET_PAIR_BYTES + MAX_NATIVE_CONTEXTS * context_bytes
        self.assertEqual(planned, MAX_PLANNED_CAPTURE_BYTES)
        self.assertEqual(CAPTURE_BUDGET_BYTES - planned, MAX_PLANNED_CAPTURE_HEADROOM_BYTES)
        self.assertEqual(MAX_PLANNED_CAPTURE_HEADROOM_BYTES, 51_192_224)
        self.assertGreater(planned, OLD_V1_CAPTURE_BUDGET_BYTES)
        self.assertLessEqual(planned, CAPTURE_BUDGET_BYTES)
        self.assertLessEqual(_working_copy_peak_bytes(retained_bytes, context_bytes), WORKING_COPY_BUDGET_BYTES)

    def test_oversized_packet_descriptor_is_rejected_before_any_packet_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = self._observer(root)
            self._step_to_target(observer)
            frame = Frame(1016, np.zeros((4, 5, 3), dtype=np.uint8), np.eye(3))
            mask = np.ones((4, 5), dtype=np.uint8)
            observer.begin_frame(frame, mask, True, '1016.png', 'mask-sha', 'available')
            context = observer.enter_refinement(1016)
            retained, _ = _packet_arrays()
            descriptors = {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                nbytes=int(value.nbytes)) for name, value in retained.items()}
            descriptors['query_rgb_crop']['shape'] = [1, 3, 281, 280]
            descriptors['query_rgb_crop']['nbytes'] = 1 * 3 * 281 * 280 * 4
            preserved = root / 'prior.bin'
            preserved.write_bytes(b'preserve me')
            self.assertFalse(observer.preflight(context, 'retained', dict(iteration=4), descriptors))
            self.assertTrue(observer.disabled)
            self.assertIn('descriptor rejected before copy', observer.failure)
            self.assertEqual(preserved.read_bytes(), b'preserve me')
            self.assertFalse((root / 'packets').exists())

    def test_cache_and_metadata_limits_are_preflighted_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / 'cache'
            cache_root.mkdir()
            (cache_root / 'existing.bin').write_bytes(b'x' * 100)
            cache = _preflight_model_cache_budget(cache_root,
                reservation_bytes=CACHE_PREPARE_RESERVATION_BYTES,
                budget_bytes=100 + CACHE_PREPARE_RESERVATION_BYTES)
            self.assertEqual(cache['current_cache_bytes'], 100)
            self.assertEqual(cache['projected_cache_bytes'], cache['cache_budget_bytes'])
            with self.assertRaisesRegex(ValueError, 'exceeds the 8 GiB budget'):
                _preflight_model_cache_budget(cache_root,
                    reservation_bytes=CACHE_PREPARE_RESERVATION_BYTES,
                    budget_bytes=99 + CACHE_PREPARE_RESERVATION_BYTES)

            artifact_root = root / 'capture'
            artifact_root.mkdir()
            (artifact_root / 'commands.json').write_bytes(b'c' * 10)
            (artifact_root / 'packets').mkdir()
            (artifact_root / 'packets' / 'raw.npz').write_bytes(b'p' * (METADATA_ARTIFACT_BUDGET_BYTES + 100))
            self.assertEqual(_metadata_artifact_bytes(artifact_root), 10)
            self.assertEqual(_check_metadata_artifact_budget(artifact_root,
                METADATA_ARTIFACT_BUDGET_BYTES - 10), METADATA_ARTIFACT_BUDGET_BYTES)
            with self.assertRaisesRegex(ValueError, 'allowance exceeded'):
                _check_metadata_artifact_budget(artifact_root,
                    METADATA_ARTIFACT_BUDGET_BYTES - 9)

    def test_closed_v1_six_packet_descriptors_match_preserved_raw_counter(self):
        diagnostics = ROOT / '.cache/model-quality/diagnostics'
        report_path = diagnostics / 'mug-v5-capacity-failure-v1.json'
        capture_root = diagnostics / 'mug-chronological-identity-v1'
        capture_path = capture_root / 'capture.json'
        if not report_path.is_file() or not capture_path.is_file():
            self.skipTest('preserved closed v1 packet evidence is unavailable in this checkout')
        expected_report_sha = '73d256937a84c7d1885a28fdb4842a11ab0b611d2fcec441db15c37629f4e9c6'
        self.assertEqual(digest(report_path), expected_report_sha)
        failure = json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(digest(capture_path), failure['artifact_sha256']['capture.json'])
        manifest = json.loads(capture_path.read_text(encoding='utf-8'))
        self.assertTrue(manifest['capture_stopped'])
        branch = manifest['branches']['control']
        self.assertEqual(branch['status'], 'partial')
        self.assertEqual(failure['neural_control_process_exit'], 0)
        self.assertEqual(failure['parent_driver_exit'], 1)
        self.assertFalse(failure['candidate_executed'])
        rows = [row for row in branch['frame_records'] if row.get('availability') == 'captured']
        self.assertEqual(len(rows), 6)
        unique_contexts = set()
        raw_total = 0
        for row in rows:
            for kind, key in (('retained', 'retained_packet'), ('result', 'result_packet')):
                info = row[key]
                arrays = _load_bound_packet(capture_root, info)
                descriptors = {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                    nbytes=int(value.nbytes)) for name, value in arrays.items()}
                size = _packet_descriptor_bytes(kind, info.get('event_metadata', {}), descriptors)
                self.assertEqual(size, info['array_bytes'])
                raw_total += size
                del arrays
            context_key = row['context_key']
            if context_key not in unique_contexts:
                unique_contexts.add(context_key)
                info = manifest['contexts'][context_key]
                with np.load(capture_root / info['path'], allow_pickle=False) as archive:
                    size = _native_context_bytes(archive['native_rgb'],
                        archive['observed_native_mask'], archive['native_k'])
                self.assertEqual(size, info['array_bytes'])
                raw_total += size
        self.assertEqual(len(unique_contexts), 6)
        self.assertEqual(raw_total, manifest['copied_array_bytes'])
        self.assertEqual(raw_total, failure['copied_array_bytes'])
        self.assertEqual(raw_total, 65_474_608)
        self.assertEqual(failure['frozen_cap_bytes'], OLD_V1_CAPTURE_BUDGET_BYTES)
        self.assertLessEqual(raw_total, CAPTURE_BUDGET_BYTES)

    def test_closed_v1_control_prefix_matches_hash_pinned_v2_control_and_setup(self):
        diagnostics = ROOT / '.cache/model-quality/diagnostics'
        v1_path = diagnostics / 'mug-chronological-identity-v1/control-201.json'
        v2_path = diagnostics / 'mug-pnp-full-window-v2/control-240.json'
        if not v1_path.is_file() or not v2_path.is_file():
            self.skipTest('preserved closed V1/V2 control outputs are unavailable in this checkout')
        self.assertEqual(digest(v1_path),
            '29b8b367890c786043ddcdc8c1bbf375d3f61d8b0a79d1a6ddac7fa66619ee1d')
        self.assertEqual(digest(v2_path),
            '170d47f4c192dfc1c099a59864871099774e5a290a58bad6b0fcdc908029ac5d')
        v1 = json.loads(v1_path.read_text(encoding='utf-8'))
        v2 = json.loads(v2_path.read_text(encoding='utf-8'))
        self.assertEqual(len(v1['frames']), 201)
        self.assertEqual(len(v2['frames']), 240)
        self.assertIn('initialization', v1)
        self.assertIn('initialization', v2)
        self.assertEqual([frame['frameId'] for frame in v1['frames']], list(RUN_FRAME_IDS))
        self.assertEqual([frame['frameId'] for frame in v2['frames'][:201]], list(RUN_FRAME_IDS))
        self.assertEqual(_frame_equal_prefix(v1, v2), (True, None))

    def _observer(self, root, branch='control'):
        root.mkdir(parents=True, exist_ok=True)
        _base_manifest_path = root / 'capture.json'
        _base_manifest_path.write_text(json.dumps(_base_manifest()), encoding='utf-8')
        return MugChronologicalCapture(root, branch, root / f'{branch}-201.trace.jsonl', source_root=root)

    @staticmethod
    def _step_to_target(observer, target=1016):
        k = np.array([[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]])
        for frame_id in range(SETUP_FRAME_ID, target):
            frame = Frame(frame_id, np.zeros((4, 5, 3), dtype=np.uint8), k)
            mask = np.ones((4, 5), dtype=np.uint8) * 255
            observer.begin_frame(frame, mask, True, f'{frame_id}.png', 'mask-sha', 'available')
            observer.end_frame(dict(pose_state='tracking', render_state='visible', mask_state='available',
                                    cameraFromObject=np.eye(4).tolist()))

    def test_compatibility_record_binds_full_maps_and_exact_classified_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v2_root = root / 'v2'
            v2_root.mkdir()
            for name in ('experiment.json', 'stage-evidence.json', 'report.json'):
                (v2_root / name).write_text('{}', encoding='utf-8')
            source_files = {
                **capture_module.V2_INSTRUMENTATION_BASELINES,
                capture_module.V2_AUXILIARY_SOURCE_PATH: capture_module.V2_AUXILIARY_SOURCE_BASELINE,
                'bench/quality_contract.py': '1' * 64,
            }
            current = {
                'bench/quality_gotrack.py': '2' * 64,
                'bench/quality_runner.py': '3' * 64,
                capture_module.V2_AUXILIARY_SOURCE_PATH: capture_module.CURRENT_AUXILIARY_SOURCE,
                'bench/quality_contract.py': '1' * 64,
            }
            classified = capture_module._classified_source_delta(source_files, current)
            experiment = {'source_files': source_files}
            legacy_review = dict(
                manifest_path='.cache/model-quality/results/mug/segmentation/results.json',
                manifest_sha256='a' * 64, input_manifest_sha256='b' * 64,
                producer_path=capture_module.LEGACY_SAM2_PRODUCER_PATH,
                producer_sha256=capture_module.LEGACY_SAM2_PRODUCER_SHA256,
                provenance_sha256='c' * 64,
                missing_flags=list(capture_module.LEGACY_MISSING_MODE_FLAGS),
                mask_rows=[dict(frame_id=frame_id, path=f'masks/{frame_id}.png',
                    sha256='d' * 64, width=5, height=3, state='available')
                    for frame_id in range(SETUP_FRAME_ID, 1067)])
            record = dict(
                schema_version=1, review_id='synthetic-review-v1', reviewer='Sol',
                decision='accepted', scope=capture_module.COMPATIBILITY_REVIEW_SCOPE,
                runtime_path_equivalence=True,
                v2_bindings=dict(root=str(v2_root.resolve()),
                    experiment_sha256=digest(v2_root / 'experiment.json'),
                    stage_evidence_sha256=digest(v2_root / 'stage-evidence.json'),
                    report_sha256=digest(v2_root / 'report.json')),
                captured_source_hashes=source_files, current_source_hashes=current,
                instrumentation_source_delta={name: classified[name]
                    for name in capture_module.INSTRUMENTED_SOURCE_PATHS},
                auxiliary_source_delta={capture_module.V2_AUXILIARY_SOURCE_PATH:
                    classified[capture_module.V2_AUXILIARY_SOURCE_PATH]},
                approved_delta_sha256=capture_module._json_sha(classified),
                legacy_segmentation=legacy_review)
            record_path = root / 'review.json'
            record_path.write_text(json.dumps(record), encoding='utf-8')
            accepted = _validate_compatibility_review_record(record, record_path, v2_root,
                experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                source_files, current)
            self.assertEqual(accepted['classified_delta'], classified)
            (v2_root / 'report.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different V2 experiment/evidence/report bytes'):
                _validate_compatibility_review_record(record, record_path, v2_root,
                    experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                    source_files, current)
            (v2_root / 'report.json').write_text('{}', encoding='utf-8')

            unlisted_current = dict(current, **{'bench/quality_contract.py': '4' * 64})
            unlisted = copy.deepcopy(record)
            unlisted['current_source_hashes'] = unlisted_current
            with self.assertRaisesRegex(ValueError, 'unreviewed path'):
                _validate_compatibility_review_record(unlisted, record_path, v2_root,
                    experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                    source_files, unlisted_current)

            changed_aux = copy.deepcopy(record)
            changed_aux['current_source_hashes'][capture_module.V2_AUXILIARY_SOURCE_PATH] = '5' * 64
            with self.assertRaisesRegex(ValueError, 'exact reviewed old/current hashes'):
                _validate_compatibility_review_record(changed_aux, record_path, v2_root,
                    experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                    source_files, changed_aux['current_source_hashes'])

            extra_field = dict(record, inference_override=True)
            with self.assertRaisesRegex(ValueError, 'unknown or incomplete schema'):
                _validate_compatibility_review_record(extra_field, record_path, v2_root,
                    experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                    source_files, current)

            incomplete_legacy = copy.deepcopy(record)
            incomplete_legacy['legacy_segmentation'].pop('producer_sha256')
            with self.assertRaisesRegex(ValueError, 'incomplete legacy mask schema'):
                _validate_compatibility_review_record(incomplete_legacy, record_path, v2_root,
                    experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                    source_files, current)
            truncated_legacy = copy.deepcopy(record)
            truncated_legacy['legacy_segmentation']['mask_rows'].pop()
            with self.assertRaisesRegex(ValueError, 'ordered 241 native mask rows'):
                _validate_compatibility_review_record(truncated_legacy, record_path, v2_root,
                    experiment, v2_root / 'stage-evidence.json', v2_root / 'report.json',
                    source_files, current)

    def test_legacy_mask_exception_requires_exact_producer_provenance_and_native_rows(self):
        from .vision import cv2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'bundle'
            bundle.mkdir()
            input_path = bundle / 'input.json'
            input_path.write_text('{"object":"mug"}', encoding='utf-8')
            masks_root = root / '.cache/model-quality/results/mug/segmentation'
            masks_root.mkdir(parents=True)
            producer = root / capture_module.LEGACY_SAM2_PRODUCER_PATH
            producer.parent.mkdir(parents=True)
            producer.write_bytes(b'synthetic exact legacy producer bytes')
            producer_sha = digest(producer)
            provenance = dict(input_manifest_sha256=digest(input_path),
                adapter_sha256={'quality_sam2': producer_sha})
            mask_hashes = {}
            mask_rows = []
            frames = []
            for frame_id in range(SETUP_FRAME_ID, 1067):
                relative = f'masks/{frame_id}.png'
                path = masks_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.full((4, 6), 255, dtype=np.uint8))
                sha = digest(path)
                mask_hashes[str(frame_id)] = sha
                frames.append(dict(frameId=frame_id, path=relative, mask_state='available',
                    failure_reason=None, automatic_detection_required=False))
                mask_rows.append(dict(frame_id=frame_id, path=relative, sha256=sha,
                    width=6, height=4, state='available'))
            segmentation = dict(provenance=provenance, frames=frames,
                model='synthetic fixture', device='cpu')
            segmentation_path = masks_root / 'results.json'
            segmentation_path.write_text(json.dumps(segmentation), encoding='utf-8')
            experiment = dict(segmentation_provenance=provenance,
                mask_sha256_by_frame=mask_hashes)
            record = dict(legacy_segmentation=dict(
                manifest_path=segmentation_path.relative_to(root).as_posix(),
                manifest_sha256=digest(segmentation_path),
                input_manifest_sha256=digest(input_path),
                producer_path=capture_module.LEGACY_SAM2_PRODUCER_PATH,
                producer_sha256=producer_sha,
                provenance_sha256=_json_sha(provenance),
                missing_flags=list(capture_module.LEGACY_MISSING_MODE_FLAGS),
                mask_rows=mask_rows))
            with mock.patch.object(capture_module, 'LEGACY_SAM2_PRODUCER_SHA256', producer_sha):
                accepted = _validate_legacy_segmentation_record(record, segmentation,
                    segmentation_path, bundle, masks_root, experiment, root)
                self.assertEqual(len(accepted['mask_rows']), 241)
                self.assertIn('original CLI arguments remain unknown', accepted['interpretation'])
                self.assertIn('no mask-accuracy claim', accepted['interpretation'])

                for key, contradictory in (('automatic', False),
                                           ('diagnostic_control', True),
                                           ('stress_test', {'occlusion_start': 1016})):
                    altered = copy.deepcopy(segmentation)
                    altered[key] = contradictory
                    with self.subTest(flag=key), self.assertRaisesRegex(ValueError, 'contradictory'):
                        _validate_legacy_segmentation_record(record, altered, segmentation_path,
                            bundle, masks_root, experiment, root)

                altered_record = copy.deepcopy(record)
                altered_record['legacy_segmentation']['mask_rows'][0]['width'] += 1
                with self.assertRaisesRegex(ValueError, 'does not bind exact provenance'):
                    _validate_legacy_segmentation_record(altered_record, segmentation,
                        segmentation_path, bundle, masks_root, experiment, root)

                for mutate in (
                    lambda value: value['legacy_segmentation'].pop('producer_sha256'),
                    lambda value: value['legacy_segmentation'].__setitem__('original_cli_arguments', []),
                ):
                    malformed = copy.deepcopy(record)
                    mutate(malformed)
                    with self.assertRaisesRegex(ValueError, 'exact legacy segmentation provenance group'):
                        _validate_legacy_segmentation_record(malformed, segmentation,
                            segmentation_path, bundle, masks_root, experiment, root)

                changed_provenance = copy.deepcopy(segmentation)
                changed_provenance['provenance']['input_manifest_sha256'] = 'a' * 64
                with self.assertRaisesRegex(ValueError, 'differs from the V2 frozen segmentation'):
                    _validate_legacy_segmentation_record(record, changed_provenance,
                        segmentation_path, bundle, masks_root, experiment, root)

                segmentation_path.write_text(json.dumps(segmentation) + ' ', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'does not bind exact provenance'):
                    _validate_legacy_segmentation_record(record, segmentation,
                        segmentation_path, bundle, masks_root, experiment, root)
                segmentation_path.write_text(json.dumps(segmentation), encoding='utf-8')

                first_mask = masks_root / frames[0]['path']
                first_mask.write_bytes(first_mask.read_bytes() + b'tamper')
                with self.assertRaisesRegex(ValueError, 'mask hash differs from V2'):
                    _validate_legacy_segmentation_record(record, segmentation,
                        segmentation_path, bundle, masks_root, experiment, root)
                cv2.imwrite(str(first_mask), np.full((4, 6), 255, dtype=np.uint8))

                producer.write_bytes(b'changed producer bytes')
                with self.assertRaisesRegex(ValueError, 'producer source bytes'):
                    _validate_legacy_segmentation_record(record, segmentation, segmentation_path,
                        bundle, masks_root, experiment, root)

    def test_runtime_prebranch_freeze_rejects_changed_review_record_or_mask(self):
        for changed in ('review_record', 'observed_mask'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = _runtime_binding_fixture(root, root / 'capture')
                manifest = json.loads((fixture['capture_root'] / 'capture.json').read_text(encoding='utf-8'))
                first_mask = fixture['masks_root'] / 'masks/826.png'
                if changed == 'review_record':
                    fixture['review_path'].write_text(
                        fixture['review_path'].read_text(encoding='utf-8') + ' ', encoding='utf-8')
                else:
                    first_mask.write_bytes(first_mask.read_bytes() + b'tamper')
                with mock.patch.object(capture_module, 'LEGACY_SAM2_PRODUCER_SHA256', fixture['producer_sha']), \
                     mock.patch.object(capture_module, 'inference_provenance',
                        return_value=fixture['run_provenance']):
                    with self.assertRaisesRegex(ValueError, 'review record bytes changed|Frozen V2'):
                        MugChronologicalCapture(fixture['capture_root'], 'control',
                            fixture['capture_root'] / 'control-201.trace.jsonl',
                            source_root=root, bundle=fixture['bundle'], masks_root=fixture['masks_root'])
                self.assertFalse((fixture['capture_root'] / 'control-201.trace.jsonl').exists())
                self.assertEqual(manifest['branches']['control']['status'], 'prepared')

    def test_runtime_postbranch_freeze_failure_stays_partial_for_record_or_mask_drift(self):
        for changed in ('review_record', 'observed_mask'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = _runtime_binding_fixture(root, root / 'capture')
                with mock.patch.object(capture_module, 'LEGACY_SAM2_PRODUCER_SHA256', fixture['producer_sha']), \
                     mock.patch.object(capture_module, 'inference_provenance',
                        return_value=fixture['run_provenance']):
                    observer = MugChronologicalCapture(fixture['capture_root'], 'control',
                        fixture['capture_root'] / 'control-201.trace.jsonl',
                        source_root=root, bundle=fixture['bundle'], masks_root=fixture['masks_root'])
                    observer.branch_row['frame_records'] = [dict(frame_id=frame_id,
                        availability='unavailable',
                        unavailable_reason='observed_mask_unavailable_at_frame_entry')
                        for frame_id in CAPTURE_FRAME_IDS]
                    output = fixture['capture_root'] / 'control-201.json'
                    output.write_text('{}', encoding='utf-8')
                    if changed == 'review_record':
                        fixture['review_path'].write_text(
                            fixture['review_path'].read_text(encoding='utf-8') + ' ', encoding='utf-8')
                    else:
                        first_mask = fixture['masks_root'] / 'masks/826.png'
                        first_mask.write_bytes(first_mask.read_bytes() + b'tamper')
                    observer.finish_branch(output)
                final = json.loads((fixture['capture_root'] / 'capture.json').read_text(encoding='utf-8'))
                row = final['branches']['control']
                self.assertEqual(row['status'], 'partial')
                self.assertTrue(row['capture_stopped'])
                self.assertIn('post-branch immutable binding check failed', row['capture_failure'])

    def test_nonzero_exit_closes_partial_branch_even_after_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _runtime_binding_fixture(root, root / 'capture')
            manifest_path = fixture['capture_root'] / 'capture.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['branches']['control']['status'] = 'running'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            (root / 'bench/quality_contract.py').write_bytes(b'source changed after failed process')
            (fixture['capture_root'] / 'control-201.trace.jsonl').write_bytes(b'{"partial":')
            failed = record_terminal_exit(fixture['capture_root'], 'control', 17)
            row = failed['branches']['control']
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['exit_code'], 17)
            self.assertEqual(row['failure_previous_status'], 'running')
            self.assertEqual(row['failure_artifacts']['trace']['path'], 'control-201.trace.jsonl')
            self.assertEqual(failed['status'], 'failed')
            self.assertNotEqual(row['status'], 'terminal')

    def test_zero_exit_still_requires_returned_stage_and_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _runtime_binding_fixture(root, root / 'capture')
            manifest_path = fixture['capture_root'] / 'capture.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['branches']['control']['status'] = 'partial'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Exit code 0 requires a completed runner stage'):
                record_terminal_exit(fixture['capture_root'], 'control', 0)
            self.assertEqual(json.loads(manifest_path.read_text(encoding='utf-8'))['branches']['control']['status'], 'partial')

            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            output = fixture['capture_root'] / 'control-201.json'
            trace = fixture['capture_root'] / 'control-201.trace.jsonl'
            output.write_text('{}', encoding='utf-8')
            trace.write_text('{"stage":"refine"}\n', encoding='utf-8')
            manifest['branches']['control'].update(status='stage_returned', captured_pairs=0,
                output=dict(path=output.name, sha256=digest(output)),
                trace=dict(path=trace.name, sha256=digest(trace)))
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            trace.write_text('{"stage":"refine"}\n{"partial":', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'trace changed before terminal exit binding'):
                record_terminal_exit(fixture['capture_root'], 'control', 0)

    def test_closed_trace_rejects_unterminated_and_malformed_tail_but_live_reader_waits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'trace.jsonl'
            complete = b'{"stage":"expected-prefix"}\n'
            path.write_bytes(complete + b'{"stage":"truncated"')
            with self.assertRaisesRegex(ValueError, 'unterminated final record'):
                _trace_records(path)
            reader = TraceOrdinalReader(path)
            self.assertEqual(reader.sync(), [(0, {'stage': 'expected-prefix'})])
            path.write_bytes(complete + b'{not-json}\n')
            with self.assertRaisesRegex(ValueError, 'invalid record'):
                _trace_records(path)

    def test_full_current_source_map_freeze_detects_any_byte_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'bench/quality_contract.py'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'frozen source bytes')
            expected = {'bench/quality_contract.py': digest(source)}
            self.assertEqual(capture_module._verify_full_current_source_map(root, expected), expected)
            source.write_bytes(b'changed source bytes')
            with self.assertRaisesRegex(ValueError, 'Full reviewed source freeze changed'):
                capture_module._verify_full_current_source_map(root, expected)

    def test_manifest_source_freeze_binds_full_map_frozen_files_review_and_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'bench/quality_contract.py'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'full map source')
            module_copy = root / 'bench/quality_mug_chronological_capture.py'
            module_copy.write_bytes(Path(capture_module.__file__).read_bytes())
            frozen = root / '.cache/immutable/input.bin'
            frozen.parent.mkdir(parents=True)
            frozen.write_bytes(b'input pin')
            record = root / '.cache/reviews/accepted.json'
            record.parent.mkdir(parents=True)
            record.write_text('{"decision":"accepted"}', encoding='utf-8')
            v2_root = root / '.cache/v2'
            v2_root.mkdir()
            v2_files = {}
            for name in ('experiment.json', 'stage-evidence.json', 'report.json'):
                path = v2_root / name
                path.write_text('{}', encoding='utf-8')
                v2_files[name] = digest(path)
            bindings = dict(
                source_root=str(root),
                current_source_hashes={'bench/quality_contract.py': digest(source)},
                frozen_files={frozen.relative_to(root).as_posix(): digest(frozen)},
                legacy_review_record_path=record.relative_to(root).as_posix(),
                legacy_review_record_sha256=digest(record),
                v2_root=str(v2_root),
                v2_experiment_sha256=v2_files['experiment.json'],
                v2_stage_evidence_sha256=v2_files['stage-evidence.json'],
                v2_report_sha256=v2_files['report.json'],
                new_capture_module_sha256=digest(module_copy))
            self.assertTrue(capture_module._verify_manifest_source_freeze(bindings, root))
            frozen.write_bytes(b'changed input pin')
            with self.assertRaisesRegex(ValueError, 'Frozen V2 input/checkpoint/mask/smoke binding'):
                capture_module._verify_manifest_source_freeze(bindings, root)
            frozen.write_bytes(b'input pin')
            self.assertTrue(capture_module._verify_manifest_source_freeze(bindings, root))
            record.write_text('{"decision":"changed"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'review record bytes changed'):
                capture_module._verify_manifest_source_freeze(bindings, root)
            record.write_text('{"decision":"accepted"}', encoding='utf-8')
            module_copy.write_bytes(b'changed module source')
            with self.assertRaisesRegex(ValueError, 'capture module source hash changed'):
                capture_module._verify_manifest_source_freeze(bindings, root)

    def test_mocked_actual_prepare_stays_out_of_annotation_and_evaluation_modules(self):
        from .quality_mug_chronological_capture import prepare_capture_root
        from .vision import cv2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'bundle'
            bundle.mkdir()
            input_path = bundle / 'input.json'
            input_value = dict(object='mug', units='metres', setup_frame_id=SETUP_FRAME_ID,
                frame_ids=list(range(827, 1067)))
            input_path.write_text(json.dumps(input_value), encoding='utf-8')
            smoke_path = root / '.cache/model-quality/smoke/mug-cpu-unlit-no-msaa.json'
            smoke_path.parent.mkdir(parents=True)
            smoke = dict(smoke_passed=True, unlit_templates=True, disable_multisampling=True)
            smoke_path.write_text(json.dumps(smoke), encoding='utf-8')
            masks_root = root / '.cache/model-quality/results/mug/segmentation'
            masks_root.mkdir(parents=True)
            producer = root / capture_module.LEGACY_SAM2_PRODUCER_PATH
            producer.parent.mkdir(parents=True)
            producer.write_bytes(b'fixture producer for mocked prepare')
            producer_sha = digest(producer)
            provenance = dict(input_manifest_sha256=digest(input_path),
                adapter_sha256={'quality_sam2': producer_sha})
            frames, mask_hashes, review_rows = [], {}, []
            for frame_id in range(SETUP_FRAME_ID, 1067):
                relative = f'masks/{frame_id}.png'
                mask_path = masks_root / relative
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), np.full((3, 5), 255, dtype=np.uint8))
                sha = digest(mask_path)
                mask_hashes[str(frame_id)] = sha
                frames.append(dict(frameId=frame_id, path=relative, mask_state='available',
                    failure_reason=None, automatic_detection_required=False))
                review_rows.append(dict(frame_id=frame_id, path=relative, sha256=sha,
                    width=5, height=3, state='available'))
            segmentation = dict(model='fixture', device='cpu', provenance=provenance, frames=frames)
            segmentation_path = masks_root / 'results.json'
            segmentation_path.write_text(json.dumps(segmentation), encoding='utf-8')
            v2_root = root / 'v2'
            v2_root.mkdir()
            for name in ('experiment.json', 'stage-evidence.json', 'report.json'):
                (v2_root / name).write_text('{}', encoding='utf-8')
            bench_source = root / 'bench'
            bench_source.mkdir()
            (bench_source / 'quality_mug_chronological_capture.py').write_bytes(
                Path(capture_module.__file__).read_bytes())
            contract_source = bench_source / 'quality_contract.py'
            contract_source.write_text('synthetic source freeze\n', encoding='utf-8')
            current_source_hashes = {'bench/quality_contract.py': digest(contract_source)}
            producer_record = dict(
                manifest_path=segmentation_path.relative_to(root).as_posix(),
                manifest_sha256=digest(segmentation_path), input_manifest_sha256=digest(input_path),
                producer_path=capture_module.LEGACY_SAM2_PRODUCER_PATH, producer_sha256=producer_sha,
                provenance_sha256=_json_sha(provenance),
                missing_flags=list(capture_module.LEGACY_MISSING_MODE_FLAGS), mask_rows=review_rows)
            compatibility_record = dict(approved_delta_sha256='fixture-delta',
                legacy_segmentation=producer_record)
            compatibility_path = root / 'sol-review.json'
            compatibility_path.write_text(json.dumps(compatibility_record), encoding='utf-8')
            input_relative = input_path.relative_to(root).as_posix()
            smoke_relative = smoke_path.relative_to(root).as_posix()
            v2_provenance = dict(input_manifest_sha256=digest(input_path), source_revisions={},
                submodule_revisions={}, checkpoint_sha256={}, foundpose_bank_sha256='bank-sha',
                adapter_sha256={})
            experiment = dict(object='mug', setup_frame_id=SETUP_FRAME_ID,
                expected_frame_ids=list(range(827, 1067)), run_mode='complete',
                reference_or_annotation_inputs=False,
                required_settings=dict(capture_module.REQUIRED_V4_SETTINGS),
                frozen_files={input_relative: digest(input_path), smoke_relative: digest(smoke_path)},
                source_files={'bench/quality_contract.py': '1' * 64},
                source_snapshot_files={}, device='cpu',
                mask_sha256_by_frame=mask_hashes, segmentation_provenance=provenance)
            v2 = dict(root=str(v2_root), experiment=experiment, evidence={}, report={},
                provenance=v2_provenance, source_delta={}, auxiliary_source_delta={},
                current_source_sha256={}, current_source_hashes=current_source_hashes,
                compatibility_review=dict(record=compatibility_record, relative_path='sol-review.json',
                    sha256=digest(compatibility_path)))
            run_provenance = copy.deepcopy(v2_provenance)
            run_provenance.pop('foundpose_bank_sha256')
            blocked_imports = {'bench.quality_annotations', 'bench.quality_evaluate',
                               'bench.quality_evaluate_saved', 'bench.quality_evaluation_reference'}
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name in blocked_imports or any(name.startswith(item + '.') for item in blocked_imports):
                    raise AssertionError(f'forbidden annotation/evaluation import: {name}')
                return original_import(name, *args, **kwargs)

            capture_root = root / 'capture'
            spec_path = root / capture_module.CAPACITY_SPEC_PATH
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_bytes((Path(ROOT) / capture_module.CAPACITY_SPEC_PATH).read_bytes())
            with mock.patch.object(capture_module, '_validate_v2_terminal', return_value=v2), \
                 mock.patch.object(capture_module, 'inference_provenance', return_value=run_provenance), \
                 mock.patch.object(capture_module, 'LEGACY_SAM2_PRODUCER_SHA256', producer_sha), \
                 mock.patch('builtins.__import__', side_effect=guarded_import):
                prepared = prepare_capture_root(bundle, masks_root, v2_root, capture_root,
                    root, compatibility_path)
            self.assertEqual(prepared['status'], 'prepared')
            self.assertTrue((capture_root / 'capture.json').is_file())
            commands = json.loads((capture_root / 'commands.json').read_text(encoding='utf-8'))
            self.assertEqual(commands['run_order'], ['control', 'candidate'])
            self.assertEqual(commands['branches']['control']['expected_ids'], list(RUN_FRAME_IDS))
            self.assertEqual(commands['branches']['candidate']['expected_ids'], list(RUN_FRAME_IDS))
            self.assertFalse(json.loads((capture_root / 'capture.json').read_text(encoding='utf-8'))[
                'reference_or_annotation_inputs'])

    def test_recovery_only_and_early_stop_are_explicit_chronological_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            observer = self._observer(Path(temporary))
            self._step_to_target(observer)
            k = np.eye(3)
            frame = Frame(1016, np.zeros((4, 5, 3), dtype=np.uint8), k)
            mask = np.ones((4, 5), dtype=np.uint8)
            observer.begin_frame(frame, mask, False, '1016.png', 'mask-sha', 'available')
            context = observer.enter_refinement(1016)
            with observer.trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': 'refine/1016/0'}) + '\n')
            observer(dict(kind='trace', context=context,
                metadata=dict(frame_id=1016, iteration=0, expected_stage='refine/1016/0')))
            with observer.trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': 'recover/1016'}) + '\n')
            observer.after_recovery(1016)
            observer.end_frame(dict(pose_state='tracking', render_state='visible', mask_state='available',
                                    cameraFromObject=np.eye(4).tolist()))
            row = observer.branch_row['frame_records'][0]
            self.assertEqual(row['availability'], 'unavailable')
            self.assertEqual(row['unavailable_reason'], 'recovery_only_at_frame_entry')
            self.assertFalse(row['entry_tracking'])
            self.assertEqual(row['recoveries'][0]['trace_ordinal'], 1)

            frame = Frame(1017, np.zeros((4, 5, 3), dtype=np.uint8), k)
            observer.begin_frame(frame, mask, True, '1017.png', 'mask-sha', 'available')
            context = observer.enter_refinement(1017)
            with observer.trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': 'refine/1017/0'}) + '\n')
            observer(dict(kind='trace', context=context,
                metadata=dict(frame_id=1017, iteration=0, expected_stage='refine/1017/0')))
            observer.clear_invocation()
            observer.end_frame(dict(pose_state='lost', render_state='suppressed', mask_state='available',
                                    failure_reason='early-stop', cameraFromObject=None))
            row = observer.branch_row['frame_records'][1]
            self.assertEqual(row['availability'], 'unavailable')
            self.assertEqual(row['unavailable_reason'], 'primary_refinement_ended_before_iteration_4')
            self.assertEqual(row['invocations'][0]['trace_ordinals'], [2])

    def test_iteration_four_packet_copy_centers_camera_units_and_immutable_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = self._observer(root)
            self._step_to_target(observer)
            native_rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            native_mask = np.ones((4, 5), dtype=np.uint8) * 255
            retained, result = _packet_arrays()
            frame = Frame(1016, native_rgb, retained['native_k'])
            observer.begin_frame(frame, native_mask, True, '1016.png', 'mask-sha', 'available')
            context = observer.enter_refinement(1016)
            for iteration in range(5):
                stage = f'refine/1016/{iteration}'
                with observer.trace_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'stage': stage}) + '\n')
                observer(dict(kind='trace', context=context,
                    metadata=dict(frame_id=1016, iteration=iteration, expected_stage=stage)))
            metadata = dict(frame_id=1016, iteration=4, rng_seed=1016 * 17 + 4,
                coordinate_convention={'source_pixel_centers': '(x+.5,y+.5)',
                    'initial_pose': 'millimetres', 'native_pose': 'metres'})
            from .quality_gotrack import GoTrackRefiner
            descriptors = GoTrackRefiner._chronological_array_descriptors(retained)
            self.assertTrue(observer.preflight(context, 'retained', metadata, descriptors))
            observer(dict(kind='retained', context=context, metadata=metadata, arrays=retained))
            result_meta = dict(metadata, fit_state='refined', solver_success=True)
            observer(dict(kind='result', context=context, metadata=result_meta, arrays=result))
            retained['full_flow_crop_px'][:] = 99
            observer.end_frame(dict(pose_state='lost', render_state='suppressed', mask_state='available',
                                    failure_reason='jump_rejected', cameraFromObject=None))
            row = observer.branch_row['frame_records'][0]
            self.assertEqual(row['availability'], 'captured')
            self.assertEqual(row['public_decision']['pose_state'], 'lost')
            loaded = capture_module._load_bound_packet(root, row['retained_packet'])
            np.testing.assert_array_equal(loaded['full_flow_crop_px'], np.zeros((280, 280, 2), dtype=np.float32))
            np.testing.assert_array_equal(loaded['source_crop_pixel_centers'][:3], [[.5, .5], [1.5, .5], [2.5, .5]])

            v2_root = root / 'v2'
            v2_root.mkdir()
            v2 = _fake_v2(v2_root)
            manifest = observer.manifest
            manifest['bindings'] = dict(
                v2_experiment_sha256=digest(v2_root / 'experiment.json'),
                v2_stage_evidence_sha256=digest(v2_root / 'stage-evidence.json'),
                v2_report_sha256=digest(v2_root / 'report.json'),
                prior_inference_provenance_sha256=_json_sha(v2['provenance']),
                instrumentation_only_source_delta={}, frozen_files={}, prior_source_files={},
                current_inference_provenance=dict(v2['provenance']), current_source_sha256={},
                captured_source_hashes={}, current_source_hashes={}, auxiliary_source_delta={},
                approved_classified_delta_sha256='delta-sha',
                legacy_review_record_sha256='review-sha', legacy_review_record_path='review.json',
                new_capture_module_sha256=digest(Path(capture_module.__file__)),
            )
            manifest['branches']['control']['status'] = 'terminal'
            manifest['branches']['control']['captured_pairs'] = 1
            for frame_id in CAPTURE_FRAME_IDS[1:]:
                manifest['branches']['control']['frame_records'].append(dict(
                    frame_id=frame_id, availability='unavailable',
                    unavailable_reason='observed_mask_unavailable_at_frame_entry',
                    mask_usable_at_entry=False, entry_tracking=True, invocations=[], recoveries=[]))
            source_root = Path(capture_module.__file__).parents[1]
            self.assertEqual(_validate_capture_branch(root, 'control', manifest, v2_root, v2,
                source_root), [])
            packet_path = root / row['retained_packet']['path']
            packet_path.write_bytes(packet_path.read_bytes() + b'tamper')
            issues = _validate_capture_branch(root, 'control', manifest, v2_root, v2, source_root)
            self.assertTrue(any('packet bytes do not match' in issue for issue in issues))

    def test_budget_overflow_keeps_prior_bytes_and_disables_new_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = self._observer(root)
            preserved = root / 'preserved.bin'
            preserved.write_bytes(b'previously captured bytes')
            observer.manifest['copied_array_bytes'] = CAPTURE_BUDGET_BYTES - 1
            observer._persist_manifest()
            self._step_to_target(observer)
            frame = Frame(1016, np.zeros((4, 5, 3), dtype=np.uint8), np.eye(3))
            mask = np.ones((4, 5), dtype=np.uint8)
            observer.begin_frame(frame, mask, True, '1016.png', 'mask-sha', 'available')
            context = observer.enter_refinement(1016)
            allowed = observer.preflight_nbytes(context, 'retained', dict(iteration=4), 2)
            self.assertFalse(allowed)
            self.assertTrue(observer.disabled)
            self.assertEqual(preserved.read_bytes(), b'previously captured bytes')
            self.assertEqual(observer.manifest['copied_array_bytes'], CAPTURE_BUDGET_BYTES - 1)
            observer.end_frame(dict(pose_state='tracking', render_state='visible', mask_state='available',
                                    cameraFromObject=np.eye(4).tolist()))
            self.assertEqual(observer.branch_row['frame_records'][0]['availability'], 'unavailable')

    def test_trajectory_prefix_rejects_pose_or_state_drift(self):
        frames = [_comparison_frame(frame_id) for frame_id in RUN_FRAME_IDS]
        short, long = dict(frames=copy.deepcopy(frames)), dict(frames=copy.deepcopy(frames))
        self.assertEqual(_frame_equal_prefix(short, long), (True, None))
        long['frames'][4]['cameraFromObject'][0][3] = 0.000002
        self.assertIn('pose mismatch', _frame_equal_prefix(short, long)[1])
        long['frames'] = copy.deepcopy(frames)
        long['frames'][3]['render_state'] = 'suppressed'
        self.assertIn('render_state mismatch', _frame_equal_prefix(short, long)[1])

    def test_trajectory_prefix_compares_nested_validation_and_rejection_reasons(self):
        frames = [_comparison_frame(frame_id) for frame_id in RUN_FRAME_IDS]
        short, long = dict(frames=copy.deepcopy(frames)), dict(frames=copy.deepcopy(frames))
        short['frames'][0]['timings_ms']['pose_total'] = 19.5
        self.assertEqual(_frame_equal_prefix(short, long), (True, None),
            'elapsed timing values are intentionally outside the decision comparison')

        long['frames'][37]['validation']['model_motion']['state'] = 'enabled'
        self.assertIn('validation.model_motion.state', _frame_equal_prefix(short, long)[1])
        long['frames'] = copy.deepcopy(frames)
        long['frames'][37]['validation']['inliers'] -= 1
        self.assertIn('validation.inliers', _frame_equal_prefix(short, long)[1])
        long['frames'] = copy.deepcopy(frames)
        long['frames'][37]['rejection_reasons'] = ['pose_jump_rejected']
        self.assertIn('rejection_reasons', _frame_equal_prefix(short, long)[1])
        long['frames'] = copy.deepcopy(frames)
        long['frames'][37]['rejection_reasons'] = ['second', 'first']
        short['frames'][37]['rejection_reasons'] = ['first', 'second']
        self.assertIn('rejection_reasons', _frame_equal_prefix(short, long)[1])

    def test_validation_float_tolerance_types_and_missing_keys_are_strict(self):
        frames = [_comparison_frame(frame_id) for frame_id in RUN_FRAME_IDS]
        short, long = dict(frames=copy.deepcopy(frames)), dict(frames=copy.deepcopy(frames))
        short_value = short['frames'][0]['validation']['score']
        long['frames'][0]['validation']['score'] = short_value + 9.0e-10
        self.assertEqual(_frame_equal_prefix(short, long), (True, None))
        long['frames'][0]['validation']['score'] = short_value + 1.2e-9
        self.assertIn('validation.score float differs', _frame_equal_prefix(short, long)[1])
        long['frames'] = copy.deepcopy(frames)
        long['frames'][0]['validation']['inliers'] = True
        self.assertIn('validation.inliers value type mismatch', _frame_equal_prefix(short, long)[1])
        long['frames'] = copy.deepcopy(frames)
        del long['frames'][0]['validation']['score']
        self.assertIn('validation key set mismatch', _frame_equal_prefix(short, long)[1])

    def test_validation_nulls_match_only_when_explicitly_present_on_both_sides(self):
        frames = [_comparison_frame(frame_id) for frame_id in RUN_FRAME_IDS]
        short, long = dict(frames=copy.deepcopy(frames)), dict(frames=copy.deepcopy(frames))
        short['frames'][0]['validation'] = None
        long['frames'][0]['validation'] = None
        self.assertEqual(_frame_equal_prefix(short, long), (True, None))

        long['frames'][0]['validation'] = copy.deepcopy(frames[0]['validation'])
        self.assertIn('validation null/object mismatch', _frame_equal_prefix(short, long)[1])

        long['frames'][0]['validation'] = None
        del long['frames'][0]['validation']
        self.assertIn('frame schema mismatch', _frame_equal_prefix(short, long)[1])

    def test_setup_initialization_decision_is_compared_when_present(self):
        frames = [_comparison_frame(frame_id) for frame_id in RUN_FRAME_IDS]
        setup = _comparison_frame(SETUP_FRAME_ID)
        setup.pop('mask_sha256')
        short, long = dict(frames=copy.deepcopy(frames), initialization=copy.deepcopy(setup)), \
                      dict(frames=copy.deepcopy(frames), initialization=copy.deepcopy(setup))
        self.assertEqual(_frame_equal_prefix(short, long), (True, None))
        long['initialization']['failure_reason'] = 'ambiguous_recovery_hypotheses'
        self.assertIn('setup initialization.failure_reason', _frame_equal_prefix(short, long)[1])
        long = dict(frames=copy.deepcopy(frames))
        self.assertIn('setup initialization decision presence mismatch', _frame_equal_prefix(short, long)[1])

    def test_trace_comparison_is_ordered_and_detects_array_hash_drift(self):
        reference = [dict(stage='render', array_sha256={'depth': 'a'}),
                     dict(stage='refine/1027/4', array_sha256={'flow': 'b'})]
        self.assertIsNone(_compare_trace_records(reference, list(reference)))
        changed = [dict(reference[0]), dict(reference[1], array_sha256={'flow': 'changed'})]
        self.assertIsNotNone(_compare_trace_records(reference, changed))
        self.assertIsNotNone(_compare_trace_records(reference, list(reversed(reference))))

    def test_observer_callback_path_is_rng_neutral_and_default_is_inert(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = self._observer(root)
            self._step_to_target(observer)
            frame = Frame(1016, np.zeros((4, 5, 3), dtype=np.uint8), np.eye(3))
            mask = np.ones((4, 5), dtype=np.uint8)
            observer.begin_frame(frame, mask, True, '1016.png', 'mask-sha', 'available')
            context = observer.enter_refinement(1016)
            packet_arrays, _ = _packet_arrays()
            metadata = dict(frame_id=1016, iteration=4)
            state_py = random.getstate()
            state_np = np.random.get_state()
            import torch
            state_torch = torch.get_rng_state().clone()
            from .quality_gotrack import GoTrackRefiner
            refiner = GoTrackRefiner(None, None, 1, 'cpu', chronological_capture_callback=observer)
            refiner.set_chronological_capture_context(context)
            refiner._emit_chronological_capture('retained', metadata, packet_arrays)
            self.assertEqual(random.getstate(), state_py)
            np.testing.assert_array_equal(np.random.get_state()[1], state_np[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), state_torch))
            packet_arrays['query_rgb_crop'][:] = -1
            info = observer.current_row['retained_packet']
            self.assertEqual(info['arrays']['query_rgb_crop']['sha256'],
                             capture_module._sha_bytes(np.zeros((1, 3, 280, 280), dtype=np.float32).tobytes()))
            default = GoTrackRefiner(None, None, 1, 'cpu')
            self.assertFalse(default._emit_chronological_capture('retained', metadata, packet_arrays))

    def test_adversarial_hook_preflight_cannot_change_rng_or_solver_arrays(self):
        import torch
        from .quality_gotrack import GoTrackRefiner

        class Adversary:
            def __init__(self):
                self.calls = []
                self.descriptor_only = True

            @staticmethod
            def _consume_rng():
                random.random()
                np.random.random()
                torch.rand(1)

            def should_capture(self, context, kind, metadata):
                self.calls.append(('should_capture', kind))
                self._consume_rng()
                selected = kind == 'trace' or metadata.get('iteration') == 4
                context['phase'] = 'recovery'
                metadata['iteration'] = -1
                return selected

            def preflight_nbytes(self, context, kind, metadata, nbytes):
                self.calls.append(('preflight_nbytes', kind))
                self._consume_rng()
                context['frame_id'] = -1
                metadata['iteration'] = -1
                return True

            def preflight(self, context, kind, metadata, arrays):
                self.calls.append(('preflight', kind))
                self._consume_rng()
                self.descriptor_only &= all(isinstance(value, dict) and
                    set(value) == {'shape', 'dtype', 'nbytes'} for value in arrays.values())
                descriptor = arrays.get('full_obj_points_mm')
                if self.descriptor_only:
                    descriptor['shape'][0] = -999
                    descriptor['nbytes'] = -1
                return True

            def __call__(self, packet):
                self.calls.append(('callback', packet['kind']))
                self._consume_rng()
                points = packet.get('arrays', {}).get('full_obj_points_mm')
                if points is not None:
                    points[:] = -12345

        def run(callback):
            utils_module, network, renderer, frame, mask, seed = _fake_refinement_inputs()
            random.seed(21)
            np.random.seed(22)
            torch.manual_seed(23)
            before = _rng_snapshot(torch)
            with mock.patch.dict(sys.modules, {'utils': utils_module}):
                refiner = GoTrackRefiner(network, renderer, 'mug', 'cpu',
                    chronological_capture_callback=callback)
                refiner.set_chronological_capture_context(dict(frame_id=1016,
                    phase='primary_tracking', invocation_ordinal=1))
                candidate = refiner.refine(frame, mask, seed)
            return candidate, _rng_unchanged(torch, before)

        utils_module, network, renderer, frame, mask, seed = _fake_refinement_inputs()
        with mock.patch.dict(sys.modules, {'utils': utils_module}):
            baseline = GoTrackRefiner(network, renderer, 'mug', 'cpu').refine(frame, mask, seed)
        adversary = Adversary()
        candidate, rng_ok = run(adversary)
        violations = []
        if not rng_ok:
            violations.append('Python/NumPy/Torch RNG changed in eligibility, budget, preflight, or callback hooks')
        if not np.array_equal(candidate.points_object_m, baseline.points_object_m):
            violations.append('preflight mutation changed solver-owned retained points')
        self.assertIn(('preflight', 'retained'), adversary.calls)
        self.assertTrue(adversary.descriptor_only, 'array preflight must receive descriptors only')
        self.assertEqual(violations, [], '; '.join(violations))

    def test_byte_preflight_exception_and_disable_hook_are_rng_neutral(self):
        import torch
        from .quality_gotrack import GoTrackRefiner

        class FailingPreflight:
            def __init__(self):
                self.disabled = []

            @staticmethod
            def _consume_rng():
                random.random()
                np.random.random()
                torch.rand(1)

            def should_capture(self, context, kind, metadata):
                self._consume_rng()
                return not self.disabled and (kind == 'trace' or metadata.get('iteration') == 4)

            def preflight_nbytes(self, context, kind, metadata, nbytes):
                self._consume_rng()
                raise RuntimeError('synthetic preflight refusal')

            def _disable_capture(self, reason):
                self._consume_rng()
                self.disabled.append(reason)

            def preflight(self, *args):
                raise AssertionError('array preflight must not run after byte refusal')

            def __call__(self, packet):
                raise AssertionError('no packet should be emitted after byte refusal')

        callback = FailingPreflight()
        utils_module, network, renderer, frame, mask, seed = _fake_refinement_inputs()
        random.seed(31)
        np.random.seed(32)
        torch.manual_seed(33)
        before = _rng_snapshot(torch)
        with mock.patch.dict(sys.modules, {'utils': utils_module}):
            refiner = GoTrackRefiner(network, renderer, 'mug', 'cpu',
                chronological_capture_callback=callback)
            refiner.set_chronological_capture_context(dict(frame_id=1016,
                phase='primary_tracking', invocation_ordinal=1))
            refiner.refine(frame, mask, seed)
        self.assertTrue(callback.disabled, 'byte-preflight failure must invoke the guarded disable hook')
        self.assertTrue(_rng_unchanged(torch, before),
            'eligibility, byte-preflight and disable callbacks must all preserve global RNG state')


if __name__ == '__main__':
    unittest.main()
