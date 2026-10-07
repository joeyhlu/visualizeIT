"""CPU caller-path fixture for the frozen bottle capture preflight."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_identity_probe as probe
from .vision import cv2


class _StopBeforeNeuralCall(RuntimeError):
    pass


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])

    def __truediv__(self, divisor):
        return _FakeTensor(self.value / divisor)

    def to(self, _device):
        return self

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeCamera:
    def __init__(self, width, height, f, c, T_world_from_eye):
        self.width, self.height = int(width), int(height)
        self.K = np.array([[f[0], 0., c[0]], [0., f[1], c[1]], [0., 0., 1.]],
                          dtype=np.float64)
        self.T_world_from_eye = np.asarray(T_world_from_eye, dtype=np.float64)


class _FakeRenderer:
    def __init__(self, expected_object_id):
        self.expected_object_id = int(expected_object_id)
        self.object_ids = []
        self.closed = False
        self.vertices_m = np.array([
            [-.04, -.03, -.08], [.05, -.03, -.02], [-.03, .04, .03], [.04, .03, .09],
        ], dtype=np.float32)
        self._render_index = 0

    def render_object_model(self, object_id, camera):
        self.object_ids.append(int(object_id))
        if int(object_id) != self.expected_object_id:
            raise AssertionError('Capture used an object id outside the input manifest')
        height, width = camera.height, camera.width
        image = np.full((height, width, 3), .5, dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)
        pad_x, pad_y = max(1, width // 5), max(1, height // 5)
        mask[pad_y:height-pad_y, pad_x:width-pad_x] = 1
        depth = np.zeros((height, width), dtype=np.float32)
        depth[mask > 0] = 900.
        rng = np.random.default_rng(9000 + self._render_index)
        image[mask > 0] = rng.random((int(mask.sum()), 3), dtype=np.float32)
        self._render_index += 1
        return {'rgb': image, 'depth': depth, 'mask': mask}

    def close(self):
        self.closed = True


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__('torch')
        self.__version__ = 'cpu-preflight-fixture'
        self.version = types.SimpleNamespace(cuda='mocked-no-cuda')
        self._tensor_calls = 0
        self.cuda = types.SimpleNamespace(
            get_device_name=lambda _index: 'mock-device-boundary',
            synchronize=lambda: None,
        )
        self.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=False, allow_tf32=False),
            cuda=types.SimpleNamespace(
                matmul=types.SimpleNamespace(allow_tf32=False), allow_tf32=False,
                enable_flash_sdp=lambda _value: None,
                enable_mem_efficient_sdp=lambda _value: None,
            ),
        )

    def from_numpy(self, value):
        self._tensor_calls += 1
        return _FakeTensor(value)

    def manual_seed(self, _seed):
        return None

    def set_num_threads(self, _count):
        return None

    def use_deterministic_algorithms(self, _enabled):
        return None

    def are_deterministic_algorithms_enabled(self):
        return True

    def get_num_threads(self):
        return 1

    def inference_mode(self):
        class _NoOpInference:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False
        return _NoOpInference()


class BottleCapturePreflightTests(unittest.TestCase):
    def test_run_capture_prepares_manifest_bound_contexts_before_mocked_network_stop(self):
        object_id = 731
        manifest = dict(
            object_id=object_id,
            asset='fake-bottle.obj', video='fake-input.mp4', source_hashes={'fixture': 'source-hash'},
            intrinsics=[[45., 0., 16.], [0., 45., 12.], [0., 0., 1.]],
            native_resolution=[32, 24],
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            cache = temp_root / 'cache'
            bundle = cache / 'inputs' / 'ranch'
            bundle.mkdir(parents=True)
            (bundle / 'input.json').write_text('{}', encoding='utf-8')
            (bundle / manifest['asset']).write_bytes(b'fixture asset')
            (bundle / manifest['video']).write_bytes(b'fixture video')
            checkpoint = cache / 'checkpoints' / 'gotrack_checkpoint.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fixture checkpoint')

            seed_path = cache / 'results' / 'ranch' / 'render-stability-appearance' / 'complete.json'
            seed_path.parent.mkdir(parents=True)
            seed_pose = np.eye(4, dtype=np.float64)
            seed_pose[2, 3] = .9
            seed_result = dict(
                object='ranch', mode='complete', complete=True, automatic=True,
                diagnostic_control=False,
                frames=[dict(frameId=frame_id, cameraFromObject=seed_pose.tolist(),
                             mask_path='masks/observed.png') for frame_id in audit.FRAME_IDS],
            )
            seed_path.write_text(json.dumps(seed_result), encoding='utf-8')
            mask_root = cache / 'results' / 'ranch' / 'segmentation' / 'masks'
            mask_root.mkdir(parents=True)
            observed_mask = np.zeros((24, 32), dtype=np.uint8)
            observed_mask[3:21, 5:27] = 255
            self.assertTrue(cv2.imwrite(str(mask_root / 'observed.png'), observed_mask))

            renderer = _FakeRenderer(object_id)
            network_calls = []
            capture_at_network_boundary = []

            def stop_network(query, template, mask):
                network_calls.append((query.value.shape, template.value.shape, mask.value.shape))
                capture_at_network_boundary.append(
                    json.loads((output_root / 'capture.json').read_text(encoding='utf-8')))
                raise _StopBeforeNeuralCall('CPU fixture stopped at the first network boundary')

            fake_torch = _FakeTorch()
            fake_gotrack = types.ModuleType('bench.quality_gotrack')
            fake_gotrack.upstream_path = lambda: None
            fake_gotrack.load_network = lambda device: self._load_mock_network(device, stop_network)
            fake_gotrack.TexturedRenderer = lambda *_args, **_kwargs: renderer

            render_types = types.SimpleNamespace(COLOR='rgb', DEPTH='depth', MASK='mask')
            renderer_base = types.SimpleNamespace(RenderType=render_types)

            def compute_fake_gotrack_inputs(**kwargs):
                self.assertEqual(kwargs['obj_ids'], [object_id])
                self.assertEqual(kwargs['crop_size'], (280, 280))
                self.assertIs(kwargs['input_masks'].__class__, _FakeTensor)
                camera = _FakeCamera(280, 280, (100., 101.), (140., 140.), np.eye(4))
                rendered = renderer.render_object_model(object_id, camera)
                chw = rendered['rgb'].transpose(2, 0, 1)
                data = dict(
                    crop_rgbs=_FakeTensor(chw[None]),
                    crop_masks=_FakeTensor(rendered['mask'][None].astype(np.float32)),
                    templates=types.SimpleNamespace(
                        rgbs=_FakeTensor(chw[None]),
                        depths=_FakeTensor(rendered['depth'][None]),
                        masks=_FakeTensor(rendered['mask'][None]),
                    ),
                )
                return data, [camera], _FakeTensor(np.eye(4, dtype=np.float64)[None])

            data_util = types.SimpleNamespace(
                compute_gotrack_inputs_from_init_poses=compute_fake_gotrack_inputs)
            misc = types.SimpleNamespace(get_intrinsic_matrix=lambda camera: camera.K)
            structs = types.SimpleNamespace(PinholePlaneCameraModel=_FakeCamera)
            fake_utils = types.ModuleType('utils')
            fake_utils.data_util = data_util
            fake_utils.misc = misc
            fake_utils.renderer_base = renderer_base
            fake_utils.structs = structs

            fake_rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
            output_root = temp_root / 'capture'
            with (
                patch.object(probe, 'CACHE', cache),
                patch.object(probe, 'read_input', return_value=manifest),
                patch.object(probe, 'read_rgb', return_value=fake_rgb),
                patch.object(probe, '_snapshot_sources', return_value={'fixture': {'sha256': 'pinned'}}),
                patch.object(probe, 'inference_provenance', return_value={'fixture': 'cpu-only'}),
                patch.dict('sys.modules', {
                    'torch': fake_torch,
                    'bench.quality_gotrack': fake_gotrack,
                    'utils': fake_utils,
                }),
            ):
                with self.assertRaisesRegex(_StopBeforeNeuralCall, 'first network boundary'):
                    probe.run_capture(output_root)

            capture = json.loads((output_root / 'capture.json').read_text(encoding='utf-8'))
            self.assertEqual(capture['object_id'], manifest['object_id'])
            self.assertEqual(capture['input_intrinsics'], manifest['intrinsics'])
            self.assertEqual(capture['native_resolution'], manifest['native_resolution'])
            self.assertEqual(capture['planned_forward_cap'], 108)
            self.assertEqual(len(capture['conditions']), 108)
            self.assertEqual(capture['forward_calls'], 0)
            self.assertEqual(capture['forwards'], [])
            self.assertEqual(len(network_calls), 1)
            self.assertTrue(renderer.closed)
            self.assertEqual(capture['status'], 'failed')
            self.assertFalse(capture['complete'])
            self.assertEqual(capture['failure']['type'], _StopBeforeNeuralCall.__name__)
            self.assertEqual(capture['conditions'][0]['state'], 'failed')
            self.assertEqual(len(capture['packet_failures']), 0)
            self.assertLessEqual(capture['packet_bytes'], audit.PACKET_BUDGET_BYTES)
            self.assertEqual(len(capture_at_network_boundary), 1)
            preflight = capture_at_network_boundary[0]
            self.assertEqual(preflight['object_id'], manifest['object_id'])
            self.assertEqual(preflight['planned_forward_cap'], audit.FORWARD_CAP)
            self.assertEqual(preflight['forward_calls'], 0)
            self.assertEqual(len(preflight['conditions']), audit.FORWARD_CAP)

            templates = [entry for entry in capture['contexts'] if entry.get('role') == 'template']
            synthetic = [entry for entry in capture['contexts']
                         if entry.get('role') == 'synthetic_query']
            self.assertEqual(len(templates), len(audit.FRAME_IDS) * len(audit.TEMPLATE_OFFSETS))
            self.assertEqual(len(synthetic), len(audit.SYNTHETIC_CARRIERS) * len(audit.QUERY_OFFSETS))
            self.assertTrue(all(entry.get('fixed_patch_bank') for entry in templates))
            boundary_templates = [entry for entry in preflight['contexts']
                                  if entry.get('role') == 'template']
            self.assertEqual(len(boundary_templates), len(templates))
            self.assertTrue(all(entry.get('fixed_patch_bank') for entry in boundary_templates))
            for entry in templates:
                bank = entry['fixed_patch_bank']
                self.assertEqual(set(bank['template_input_sha256']), set(audit._PATCH_BANK_INPUTS))
                self.assertLessEqual(bank['selected_anchor_count'], 64)
                arrays = audit._load_npz(output_root, entry)
                verified = audit.verify_fixed_patch_bank(entry, arrays)
                self.assertEqual(verified['selected_anchor_count'], bank['selected_anchor_count'])
            first_condition = capture['conditions'][0]
            self.assertEqual(first_condition['context_refs']['query'], 'synthetic-query-0010-008')
            self.assertEqual(first_condition['context_refs']['template'], 'template-0010-000')

    def _load_mock_network(self, device, stop_network):
        self.assertEqual(device, 'cuda')  # The model loader is mocked; no CUDA context is created.
        return stop_network


if __name__ == '__main__':
    unittest.main()
