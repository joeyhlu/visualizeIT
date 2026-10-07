"""CPU integration fixtures for the bounded bottle zero-view caller."""
import hashlib
import contextlib
import io
import json
import shutil
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_identity_probe as source_capture
from . import quality_bottle_zero_view_probe as zero
from .quality_assets import MODELS, ROOT, digest


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    def __init__(self):
        self.cuda = types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)

    def from_numpy(self, value):
        return _FakeTensor(value)

    def inference_mode(self):
        class _NoOp:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False
        return _NoOp()


class BottleZeroViewProbeTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.capture_root = self.root / 'immutable-r1-capture'
        self.capture_root.mkdir()
        self.expected = self._make_capture(self.capture_root)

    def tearDown(self):
        self._temporary.cleanup()

    def _run_capture(self, *args, **kwargs):
        with patch.object(zero, 'EXPECTED_CAPTURE_SHA256', digest(self.capture_root / 'capture.json')):
            return zero.run_capture(*args, **kwargs)

    def _save_npz(self, path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        payload = path.read_bytes()
        return dict(
            path=path.relative_to(self.capture_root).as_posix(),
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
            arrays={name: audit.array_info(value) for name, value in arrays.items()},
        )

    def _context(self, context_id, role, frame_id, geometry_group, arrays, **metadata):
        entry = self._save_npz(
            self.capture_root / 'packets' / 'contexts' / f'{context_id}.npz', arrays)
        entry.update(context_id=context_id, role=role, frame_id=int(frame_id),
                     geometry_group=geometry_group, **metadata)
        return entry

    def _make_capture(self, capture_root):
        h = w = 280
        k = np.array([[380., 0., 140.], [0., 380., 140.], [0., 0., 1.]], dtype=np.float64)
        crop_from_native = np.eye(4, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = .9
        pose_mm = pose.copy()
        pose_mm[:3, 3] *= 1000.
        template_mask = np.zeros((h, w), dtype=bool)
        template_mask[16:264, 16:264] = True
        observed_mask = np.zeros((h, w), dtype=bool)
        observed_mask[35:245, 35:245] = True
        depth = np.zeros((h, w), dtype=np.float32)
        depth[template_mask] = 900.
        ids, pixels, points = audit.object_points_from_depth(depth, k, pose_mm, template_mask)
        rng = np.random.default_rng(20102026)
        source_rgb = rng.random((h, w, 3), dtype=np.float32)
        competitor_rgb = np.random.default_rng(705).random((h, w, 3), dtype=np.float32)
        contexts = []
        source_arrays_by_frame = {}

        # The saved snapshot files and their manifest hashes exercise the same
        # linked provenance checks used by the actual zero-view caller.
        source_snapshot = {}
        for relative in source_capture.SOURCE_FILES:
            source = ROOT / relative
            target = capture_root / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            source_snapshot[relative] = dict(
                path=target.relative_to(capture_root).as_posix(), sha256=digest(target))

        for frame_id in zero.CARRIERS:
            frame_arrays = dict(
                native_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
                native_mask=np.ones((24, 32), dtype=np.uint8),
                observed_crop_mask=observed_mask.astype(np.uint8),
                crop_k=k, crop_from_native=crop_from_native,
                native_k=k, seed_pose_m=pose,
            )
            frame_id_text = f'{frame_id:04d}'
            frame_entry = self._context(
                f'real-frame-{frame_id_text}', 'real_frame', frame_id,
                f'frame-{frame_id}', frame_arrays,
                native_rgb_sha256=audit.array_sha256(frame_arrays['native_rgb']),
                native_mask_sha256=audit.array_sha256(frame_arrays['native_mask'] > 0),
                input_rgb_sha256=audit.array_sha256(frame_arrays['native_rgb']),
                reference_or_annotation_inputs=False,
            )
            contexts.append(frame_entry)

            for offset, rgb, source_points in (
                    (0, source_rgb, points),
                    (180, competitor_rgb, points + np.array([2., 0., 0.]))):
                arrays = dict(
                    template_rgb=rgb.copy(), template_depth_mm=depth.copy(),
                    template_mask=template_mask.astype(np.uint8),
                    source_indices=ids.copy(), source_pixels_xy=pixels.copy(),
                    source_points_object_m=source_points.copy(),
                    observed_crop_mask=observed_mask.astype(np.uint8),
                    crop_k=k.copy(), crop_from_native=crop_from_native.copy(),
                    native_k=k.copy(), seed_pose_m=pose.copy(), template_pose_m=pose.copy(),
                )
                template_id = f'template-{frame_id_text}-{offset:03d}'
                template_entry = self._context(
                    template_id, 'template', frame_id, f'frame-{frame_id}', arrays,
                    template_offset_deg=offset,
                    geometry_hashes=dict(
                        depth_mm=audit.array_sha256(arrays['template_depth_mm']),
                        mask=audit.array_sha256(arrays['template_mask'] > 0)),
                    object_source_points=len(ids), reference_or_annotation_inputs=False,
                )
                bank = audit.build_template_patch_bank(
                    arrays['template_rgb'], arrays['template_mask'],
                    arrays['source_indices'], arrays['source_points_object_m'])
                template_entry['fixed_patch_bank'] = audit.fixed_patch_bank_manifest(bank, arrays)
                contexts.append(template_entry)
                if offset == 0:
                    source_arrays_by_frame[frame_id] = arrays

            # These known synthetic labels are available only to the post-capture
            # CPU scorer; no synthetic context is used for the six new forwards.
            plus8_mask = np.zeros((h, w), dtype=np.uint8)
            plus8_mask[50:230, 50:230] = 1
            plus8_arrays = dict(
                query_rgb_rgb=source_rgb.copy(), query_depth_mm=depth.copy(),
                synthetic_query_mask=plus8_mask,
                known_query_pose_m=pose.copy(), crop_k=k.copy(),
                crop_from_native=crop_from_native.copy(), native_k=k.copy(),
                observed_crop_mask=observed_mask.astype(np.uint8),
                query_render_mask=plus8_mask.copy(),
            )
            query_entry = self._context(
                f'synthetic-query-{frame_id_text}-008', 'synthetic_query', frame_id,
                f'frame-{frame_id}', plus8_arrays,
                query_offset_deg=8, synthetic_geometry_oracle=True,
                geometry_hashes=dict(depth_mm=audit.array_sha256(depth),
                                     render_mask=audit.array_sha256(plus8_mask > 0),
                                     synthetic_mask=audit.array_sha256(plus8_mask > 0)),
            )
            contexts.append(query_entry)

        conditions = source_capture._planned_conditions()
        forwards = []
        selected_ids = []
        for condition in conditions:
            condition_id = condition['condition_id']
            frame_id = int(condition['frame_id'])
            if (condition['kind'] == 'synthetic' and condition['query_offset_deg'] == 8 and
                    condition['query_appearance'] == 'rgb' and condition['template_offset_deg'] == 0 and
                    condition['template_appearance'] == 'rgb' and frame_id in zero.CARRIERS):
                condition['state'] = 'captured'
                condition['context_refs'] = dict(
                    query=f'synthetic-query-{frame_id:04d}-008',
                    template=f'template-{frame_id:04d}-000',
                    observed_frame=f'real-frame-{frame_id:04d}',
                )
                condition['elapsed_ms'] = 0.
                forward = dict(
                    flow=np.zeros((h, w, 2), dtype=np.float32),
                    confidence=np.full((h, w), .9, dtype=np.float32),
                )
                entry = self._save_npz(
                    capture_root / 'packets' / 'forwards' / f'{condition_id}.npz', forward)
                entry.update(
                    condition_id=condition_id, rng_seed=condition['rng_seed'],
                    output_hashes={key: audit.array_sha256(value) for key, value in forward.items()},
                    derived_array_hashes=audit._array_hashes_for_forward(
                        source_arrays_by_frame[frame_id], forward),
                    elapsed_ms=0., context_refs=condition['context_refs'],
                    forward_index=condition['forward_index'],
                )
                condition.update(packet_path=entry['path'], packet_sha256=entry['sha256'],
                                 output_hashes=entry['output_hashes'])
                forwards.append(entry)
                selected_ids.append(condition_id)
            else:
                condition.update(state='unavailable', reason='cpu-fixture-unselected-condition')

        packet_bytes = sum(entry['bytes'] for entry in contexts + forwards)
        capture = dict(
            schema_version=1, object='ranch', status='complete', complete=True,
            requested_real_frame_ids=list(audit.FRAME_IDS),
            requested_synthetic_carriers=list(audit.SYNTHETIC_CARRIERS),
            requested_query_offsets_deg=list(audit.QUERY_OFFSETS),
            requested_template_offsets_deg=list(audit.TEMPLATE_OFFSETS),
            requested_template_appearances=list(audit.APPEARANCES),
            synthetic_query_appearances=list(audit.APPEARANCES),
            planned_forward_cap=audit.FORWARD_CAP, forward_calls=len(forwards),
            packet_budget_bytes=audit.PACKET_BUDGET_BYTES, packet_bytes=packet_bytes,
            reference_or_annotations_loaded=False, source_snapshot=source_snapshot,
            contexts=contexts, forwards=forwards, packet_failures=[], failures=[],
            conditions=conditions, object_bbox_diagonal_m=1.0,
            model_checkpoint_sha256=MODELS['gotrack_checkpoint.pt']['sha256'],
            output_root=str(capture_root),
        )
        (capture_root / 'capture.json').write_text(
            json.dumps(capture, indent=2), encoding='utf-8')
        return dict(
            capture=capture, expected_rgb=source_rgb, expected_competitor_rgb=competitor_rgb,
            template_mask=template_mask, observed_mask=observed_mask,
            common_mask=template_mask & observed_mask,
            source_arrays_by_frame=source_arrays_by_frame,
            selected_ids=selected_ids,
        )

    def _run_with_stub(self, output_root, *, fail=False):
        fake_torch = _FakeTorch()
        network_inputs = []
        loader_observations = []
        original_source = self.expected['source_arrays_by_frame']

        def runtime_loader():
            report = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
            loader_observations.append(report)
            self.assertTrue(report['zero_flow_control_gate']['passed'])
            self.assertEqual(len(report['conditions']), 6)
            self.assertTrue(all(row['state'] == 'pending' for row in report['conditions']))
            self.assertEqual(report['forward_calls'], 0)
            self.assertTrue(all(row['state'] == 'distinctive_current_image_support'
                                for row in report['zero_flow_controls']))
            return fake_torch, network, 'mock-cpu'

        call_number = 0

        def network(query, template, mask):
            nonlocal call_number
            frame_id = zero.CARRIERS[call_number // 2]
            condition_name = ('full', 'clipped')[call_number % 2]
            arrays = original_source[frame_id]
            expected_query = arrays['template_rgb'].copy()
            if condition_name == 'clipped':
                expected_query[~self.expected['common_mask']] = np.float32(.5)
            expected_q = zero._rgb_tensor_array(expected_query)
            expected_t = zero._rgb_tensor_array(arrays['template_rgb'])
            expected_mask = arrays['template_mask'].astype(np.float32)[None]
            np.testing.assert_array_equal(query.value, expected_q)
            np.testing.assert_array_equal(template.value, expected_t)
            np.testing.assert_array_equal(mask.value, expected_mask)
            network_inputs.append((query.value.copy(), template.value.copy(), mask.value.copy()))
            call_number += 1
            if fail:
                raise RuntimeError('fixture failure at first neural boundary')
            return (
                _FakeTensor(np.zeros((1, 2, 280, 280), dtype=np.float32)),
                _FakeTensor(np.full((1, 280, 280), .9, dtype=np.float32)),
            )

        return (runtime_loader, network_inputs, loader_observations)

    def test_real_caller_runs_six_exact_inputs_after_fixed_controls_and_reuses_plus8_packets(self):
        output_root = self.root / 'zero-view'
        runtime_loader, network_inputs, loader_observations = self._run_with_stub(output_root)
        record = self._run_capture(self.capture_root, output_root, _runtime_loader=runtime_loader)

        self.assertEqual(len(loader_observations), 1)
        self.assertEqual(len(network_inputs), zero.FORWARD_CAP)
        self.assertEqual(record['status'], 'complete')
        self.assertTrue(record['complete'])
        self.assertEqual(record['forward_calls'], 6)
        self.assertTrue(record['network_released_before_cpu_scoring'])
        self.assertTrue(record['cpu_scoring_completed_after_network_release'])
        self.assertEqual(record['all_six_conditions_accounted'], True)
        self.assertEqual(record['captured_condition_count'], 6)
        self.assertEqual(record['failed_condition_count'], 0)
        self.assertEqual(record['unavailable_condition_count'], 0)
        self.assertEqual(len(record['stored_plus8_rgb_rows']), 3)
        self.assertTrue(record['plus8_recomputed_from_existing_packets'])

        rows_by_id = {row['condition_id']: row for row in record['conditions']}
        for frame_id in zero.CARRIERS:
            source_arrays = self.expected['source_arrays_by_frame'][frame_id]
            full = rows_by_id[f'zero-{frame_id}-full']
            clipped = rows_by_id[f'zero-{frame_id}-clipped']
            expected_full = source_arrays['template_rgb']
            expected_clipped = expected_full.copy()
            expected_clipped[~self.expected['common_mask']] = np.float32(.5)
            self.assertEqual(full['input_hashes']['query_rgb'], audit.array_info(expected_full))
            self.assertEqual(clipped['input_hashes']['query_rgb'], audit.array_info(expected_clipped))
            self.assertEqual(full['input_hashes']['template_rgb'], clipped['input_hashes']['template_rgb'])
            self.assertEqual(full['input_hashes']['template_mask'], clipped['input_hashes']['template_mask'])
            self.assertEqual(full['score']['visible_denominator']['count'],
                             clipped['score']['visible_denominator']['count'])
            self.assertEqual(full['score']['visible_denominator']['source_indices_sha256'],
                             clipped['score']['visible_denominator']['source_indices_sha256'])
            self.assertEqual(full['score']['visible_denominator']['shared_with_zero_conditions'], True)
            self.assertTrue(full['score']['gates']['qualified'])
            self.assertTrue(clipped['score']['gates']['qualified'])
        for frame_id, denominator in record['stored_plus8_denominators_differ_from_zero_view'].items():
            self.assertNotEqual(denominator['zero_view_count'], denominator['plus8_count'])
        serialized = json.dumps(record)
        self.assertNotIn('known_query_pose_m', serialized)
        self.assertNotIn('query_pose_native_m', serialized)

    def test_uppercase_capture_pin_and_mocked_cuda_use_zero_view_six_forward_cap(self):
        output_root = self.root / 'mocked-cuda-metadata'
        base_loader, network_inputs, _ = self._run_with_stub(output_root)

        def mocked_cuda_loader():
            torch, network, _ = base_loader()
            return torch, network, 'cuda'

        # The shared R1 runtime helper advertises its 108-forward ceiling. The
        # R3 caller must retain helper diagnostics but publish its own six-call cap.
        # The frozen pin is uppercase while the actual digest helper returns lowercase.
        helper_runtime = dict(device='cuda', network_forwards_max=108,
                              cuda_runtime='mock-cuda-runtime')
        actual_digest = digest(self.capture_root / 'capture.json')
        self.assertEqual(actual_digest, actual_digest.lower())
        with patch.object(zero, 'EXPECTED_CAPTURE_SHA256', actual_digest.upper()):
            with patch.object(zero.capture_probe, '_runtime_info', return_value=helper_runtime):
                record = zero.run_capture(
                    self.capture_root, output_root, _runtime_loader=mocked_cuda_loader)

        self.assertEqual(len(network_inputs), zero.FORWARD_CAP)
        self.assertEqual(record['forward_calls'], 6)
        self.assertEqual(record['runtime']['device'], 'cuda')
        self.assertEqual(record['runtime']['cuda_runtime'], 'mock-cuda-runtime')
        self.assertEqual(record['runtime']['network_forwards_max'], zero.FORWARD_CAP)
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['runtime']['network_forwards_max'], zero.FORWARD_CAP)

    def test_frozen_r1_cache_cpu_preflight_accepts_empty_opposite_bank(self):
        capture_path = zero.CAPTURE_ROOT / 'capture.json'
        if not capture_path.is_file():
            self.skipTest('Immutable frozen R1 v2 capture cache is unavailable')
        capture_root, capture = audit._load_capture_manifest(zero.CAPTURE_ROOT)
        actual_digest = digest(capture_root / 'capture.json')
        self.assertEqual(actual_digest.casefold(), zero.EXPECTED_CAPTURE_SHA256.casefold())
        self.assertIs(capture.get('reference_or_annotations_loaded'), False)

        # These calls read and verify the real frozen packets only. They do not
        # construct or load a network and cannot issue a neural forward.
        prepared = zero._load_inputs(capture_root, capture)
        expected_banks = {10: (23, 0), 50: (25, 2), 100: (28, 2)}
        for frame_id, (source_count, opposite_count) in expected_banks.items():
            item = prepared[frame_id]
            self.assertEqual(item['source_bank']['selected_anchor_count'], source_count)
            self.assertEqual(item['competitor_bank']['selected_anchor_count'], opposite_count)
            self.assertIsNotNone(item['source_entry']['fixed_patch_bank'])
            self.assertIsNotNone(item['competitor_entry']['fixed_patch_bank'])
        controls = zero._zero_flow_controls(prepared, float(capture['object_bbox_diagonal_m']))
        self.assertEqual([row['frame_id'] for row in controls], [10, 50, 100])
        self.assertEqual([row['state'] for row in controls],
                         ['distinctive_current_image_support'] * 3)
        self.assertEqual([row['supported'] for row in controls], [19, 13, 24])
        self.assertEqual([row['supported_cells_4x4'] for row in controls], [8, 8, 8])
        for row, expected_hull in zip(controls, (.22371134, .13223522, .27421652)):
            self.assertAlmostEqual(row['supported_hull_fraction'], expected_hull, places=6)

    def test_empty_source_and_no_distant_banks_are_unavailable_without_forwards(self):
        for case in ('empty_source', 'no_distant_competitor'):
            with self.subTest(case=case):
                prepared = zero._load_inputs(self.capture_root, self.expected['capture'])
                source = prepared[10]['source_bank']
                opposite = prepared[10]['competitor_bank']
                if case == 'empty_source':
                    source['anchors'] = []
                    source['selected_anchor_count'] = 0
                    source['eligible_source_patch_count'] = 0
                else:
                    for bank in (source, opposite):
                        for anchor in bank['anchors']:
                            anchor['object_xyz_m'] = [0., 0., 0.]
                controls = zero._zero_flow_controls(prepared, 1.0)
                self.assertEqual(controls[0]['state'], 'identity_bank_uninformative')
                self.assertEqual(controls[0]['patch_summary']['supported'], 0)

                output_root = self.root / f'{case}-unavailable'
                loader_calls = []

                def forbidden_loader():
                    loader_calls.append(True)
                    self.fail('Uninformative fixed banks must stop before runtime loading')

                with patch.object(zero, '_zero_flow_controls', return_value=controls):
                    record = self._run_capture(
                        self.capture_root, output_root, _runtime_loader=forbidden_loader)
                self.assertEqual(loader_calls, [])
                self.assertEqual(record['status'], 'unavailable')
                self.assertFalse(record['complete'])
                self.assertEqual(record['forward_calls'], 0)
                self.assertTrue(all(row['state'] == 'unavailable' for row in record['conditions']))

    def test_first_call_failure_is_accounted_and_report_survives(self):
        output_root = self.root / 'zero-view-failure'
        runtime_loader, network_inputs, _ = self._run_with_stub(output_root, fail=True)
        with self.assertRaisesRegex(RuntimeError, 'first neural boundary'):
            self._run_capture(self.capture_root, output_root, _runtime_loader=runtime_loader)
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertFalse(saved['complete'])
        self.assertEqual(saved['forward_calls'], 1)
        self.assertTrue(saved['all_six_conditions_accounted'])
        self.assertEqual(saved['conditions'][0]['state'], 'failed')
        self.assertTrue(all(row['state'] == 'unavailable' for row in saved['conditions'][1:]))
        self.assertEqual(saved['network_released_before_cpu_scoring'], True)
        self.assertEqual(len(saved['stored_plus8_rgb_rows']), 3)
        self.assertEqual(saved['packet_bytes'], 0)
        self.assertEqual(len(network_inputs), 1)

    def test_terminal_report_persistence_failure_never_returns_or_prints_complete(self):
        output_root = self.root / 'terminal-write-failure'
        runtime_loader, _, _ = self._run_with_stub(output_root)
        real_save_result = zero.save_result
        rejected = []

        def refuse_terminal(path, payload):
            if (Path(path) == output_root / 'zero_view.json' and
                    payload.get('status') == 'complete' and payload.get('complete') is True):
                rejected.append(True)
                raise PermissionError('fixture terminal atomic-write failure')
            return real_save_result(path, payload)

        output = io.StringIO()
        with patch.object(zero, 'save_result', side_effect=refuse_terminal):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(PermissionError, 'terminal atomic-write failure'):
                    self._run_capture(self.capture_root, output_root, _runtime_loader=runtime_loader)
        self.assertEqual(rejected, [True])
        self.assertEqual(output.getvalue(), '')
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'preflight_passed')
        self.assertFalse(saved['complete'])
        self.assertNotIn('cpu_scoring_completed_after_network_release', saved)
        self.assertEqual(saved['forward_calls'], zero.FORWARD_CAP)

    def test_network_exception_traceback_does_not_retain_model_during_cpu_score(self):
        class RaisingNetwork:
            def __call__(self, *_args):
                raise RuntimeError('fixture inference boundary failure')

        holder = [RaisingNetwork()]
        network_ref = weakref.ref(holder[0])

        def runtime_loader():
            return _FakeTorch(), holder.pop(), 'mock-cpu'

        score_calls = []

        def check_model_released(*_args):
            self.assertIsNone(network_ref(), 'traceback must not retain the inference model during CPU scoring')
            score_calls.append(True)

        output_root = self.root / 'traceback-release'
        with patch.object(zero, '_score_outputs', side_effect=check_model_released):
            with self.assertRaisesRegex(RuntimeError, 'inference boundary failure'):
                self._run_capture(self.capture_root, output_root, _runtime_loader=runtime_loader)
        self.assertEqual(score_calls, [True])
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertTrue(saved['network_released_before_cpu_scoring'])
        self.assertTrue(saved['cpu_scoring_completed_after_network_release'])
        self.assertEqual(saved['failure']['type'], 'RuntimeError')
        self.assertEqual(saved['failure']['message'], 'fixture inference boundary failure')

    def test_release_failure_skips_cpu_scoring_and_is_raised(self):
        output_root = self.root / 'release-failure'
        runtime_loader, network_inputs, _ = self._run_with_stub(output_root)
        score_calls = []
        with patch.object(zero, '_release_runtime', side_effect=RuntimeError('fixture release failure')):
            with patch.object(zero, '_score_outputs', side_effect=lambda *_args: score_calls.append(True)):
                with self.assertRaisesRegex(RuntimeError, 'fixture release failure'):
                    self._run_capture(self.capture_root, output_root, _runtime_loader=runtime_loader)
        self.assertEqual(len(network_inputs), zero.FORWARD_CAP)
        self.assertEqual(score_calls, [])
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertFalse(saved['complete'])
        self.assertFalse(saved.get('cpu_scoring_completed_after_network_release', False))
        self.assertEqual(saved['cpu_scoring_skipped_reason'], 'runtime_release_failed')
        self.assertEqual(saved['failure']['stage'], 'network_release')

    def test_early_unavailable_release_failure_does_not_return_normally(self):
        output_root = self.root / 'early-unavailable-release-failure'
        controls = [dict(frame_id=frame_id, state='unobservable_or_ambiguous', supported=0,
                         supported_cells_4x4=0, supported_hull_fraction=0.)
                    for frame_id in zero.CARRIERS]
        loader_calls = []
        score_calls = []

        def forbidden_loader():
            loader_calls.append(True)
            self.fail('runtime loader must not run after failed zero-flow controls')

        with patch.object(zero, '_zero_flow_controls', return_value=controls):
            with patch.object(zero, '_release_runtime', side_effect=RuntimeError('early release failure')):
                with patch.object(zero, '_score_outputs', side_effect=lambda *_args: score_calls.append(True)):
                    with self.assertRaisesRegex(RuntimeError, 'early release failure'):
                        self._run_capture(self.capture_root, output_root, _runtime_loader=forbidden_loader)
        self.assertEqual(loader_calls, [])
        self.assertEqual(score_calls, [])
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertFalse(saved['complete'])
        self.assertFalse(saved['calibration_available'])
        self.assertEqual(saved['forward_calls'], 0)
        self.assertEqual(saved['failure']['stage'], 'network_release')

    def test_failed_zero_flow_control_stops_before_runtime_loader(self):
        output_root = self.root / 'zero-flow-unavailable'
        failed_controls = [
            dict(frame_id=frame_id,
                 state=('unobservable_or_ambiguous' if frame_id == 50 else
                        'distinctive_current_image_support'),
                 supported=(7 if frame_id == 50 else 12),
                 supported_cells_4x4=4, supported_hull_fraction=.2)
            for frame_id in zero.CARRIERS
        ]
        loader_calls = []

        def forbidden_loader():
            loader_calls.append(True)
            raise AssertionError('runtime loader must not run')

        with patch.object(zero, '_zero_flow_controls', return_value=failed_controls):
            self._run_capture(self.capture_root, output_root, _runtime_loader=forbidden_loader)
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(loader_calls, [])
        self.assertEqual(saved['status'], 'unavailable')
        self.assertFalse(saved['calibration_available'])
        self.assertEqual(saved['forward_calls'], 0)
        self.assertEqual(saved['unavailable_condition_count'], zero.FORWARD_CAP)
        self.assertTrue(all(row['state'] == 'unavailable' for row in saved['conditions']))
        self.assertEqual(saved['forwards'], [])

    def test_aggregate_budget_failure_writes_no_partial_packet_and_preserves_counter(self):
        output_root = self.root / 'budget-root'
        runtime_loader, network_inputs, _ = self._run_with_stub(output_root)
        with patch.object(zero, 'PACKET_BUDGET_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'aggregate exceeds'):
                self._run_capture(self.capture_root, output_root, _runtime_loader=runtime_loader)
        saved = json.loads((output_root / 'zero_view.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['forward_calls'], 1)
        self.assertEqual(saved['packet_bytes'], 0)
        self.assertEqual(len(saved['packet_failures']), 1)
        self.assertEqual(saved['conditions'][0]['state'], 'failed')
        self.assertTrue(all(row['state'] == 'unavailable' for row in saved['conditions'][1:]))
        self.assertFalse((output_root / 'packets' / 'zero-10-full.npz').exists())
        self.assertEqual(len(network_inputs), 1)
        self.assertTrue(saved['network_released_before_cpu_scoring'])

    def test_existing_output_root_is_preserved(self):
        output_root = self.root / 'existing-output'
        output_root.mkdir()
        sentinel = output_root / 'sentinel.txt'
        sentinel.write_text('keep exactly', encoding='utf-8')
        before = digest(sentinel)
        with self.assertRaises(FileExistsError):
            self._run_capture(self.capture_root, output_root,
                              _runtime_loader=lambda: self.fail('must not load'))
        self.assertEqual(digest(sentinel), before)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep exactly')
        self.assertFalse((output_root / 'zero_view.json').exists())

    def test_inference_source_delta_requires_exact_independent_review_binding(self):
        relative = 'bench/quality_gotrack.py'
        capture = self.expected['capture']
        saved_entry = capture['source_snapshot'][relative]
        saved_path = self.capture_root / saved_entry['path']
        with saved_path.open('ab') as stream:
            stream.write(b'\\n# frozen-R1 source delta fixture\\n')
        saved_entry['sha256'] = digest(saved_path)
        capture_path = self.capture_root / 'capture.json'
        capture_path.write_text(json.dumps(capture, indent=2), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'independent accepted Sol'):
            zero._source_pins(self.capture_root, capture)

        captured_runtime = {name: capture['source_snapshot'][name]['sha256']
                            for name in source_capture.SOURCE_FILES}
        current_runtime = {name: digest(ROOT / name) for name in source_capture.SOURCE_FILES}
        changed = {name: dict(captured_sha256=captured_runtime[name],
                              current_sha256=current_runtime[name])
                   for name in source_capture.SOURCE_FILES
                   if captured_runtime[name] != current_runtime[name]}
        review = dict(
            schema_version=1, reviewer='Sol', decision='accepted',
            scope='bottle-zero-view-inference-source-delta', review_id='fixture-sol-accepted-delta',
            capture_sha256=digest(capture_path), captured_source_hashes=captured_runtime,
            current_source_hashes=current_runtime, inference_path_equivalence=True,
            approved_delta_sha256=zero._canonical_json_sha256(changed),
        )
        review_path = self.root / 'accepted-source-review.json'
        review_path.write_text(json.dumps(review, indent=2), encoding='utf-8')
        pins = zero._source_pins(self.capture_root, capture, review_path)
        self.assertEqual(pins['reviewed_source_deltas']['reviewer'], 'Sol')
        self.assertEqual(set(pins['reviewed_source_deltas']['changed_sources']), {relative})

        review['current_source_hashes'][relative] = '0' * 64
        review_path.write_text(json.dumps(review, indent=2), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not bind this exact source delta'):
            zero._source_pins(self.capture_root, capture, review_path)


if __name__ == '__main__':
    unittest.main()
