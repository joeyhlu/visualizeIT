"""NumPy and mocked-network checks for the explicitly selected GoTrack profile."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from . import quality_gotrack_inference as inference


class InferenceOptionTests(unittest.TestCase):
    def test_no_options_returns_before_any_preflight(self):
        options = inference.options_from_values()
        self.assertIsNone(inference.prepare_inference(options, Path('missing-bundle'), {}))

    def test_option_pairs_and_profile_requirements_are_strict(self):
        for values in (
            {'head_path': 'head.pt'},
            {'head_sha256': '0' * 64},
            {'config_path': 'config.json'},
            {'config_sha256': '0' * 64},
            {'cloud_execution': True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                inference.options_from_values(**values)

    def test_compatibility_contract_arguments_are_an_authenticated_pair(self):
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            inference.options_from_values(
                config_path='request.json', config_sha256='0' * 64,
                compatibility_config_path='runtime.json', cloud_execution=True)

    def test_artifact_registry_keeps_old_and_expanded_exports_independent(self):
        self.assertEqual(set(inference.ARTIFACT_REGISTRY), {
            inference.EXPORT_SHA256, inference.EXPANDED_EXPORT_SHA256})
        old = inference.ARTIFACT_REGISTRY[inference.EXPORT_SHA256]
        expanded = inference.ARTIFACT_REGISTRY[inference.EXPANDED_EXPORT_SHA256]
        self.assertEqual(old['export_bytes'], 80_141_142)
        self.assertEqual(expanded['export_bytes'], 80_144_086)
        self.assertEqual(old['training']['dataset_sha256'], inference.DATASET_SHA256)
        self.assertEqual(expanded['training']['dataset_sha256'], inference.EXPANDED_DATASET_SHA256)
        self.assertNotEqual(old['export_sha256'], expanded['export_sha256'])

    def test_historical_expanded_contract_registry_and_pinned_closure(self):
        legacy_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        self.assertEqual(inference._sha256_file(legacy_path),
                         inference.LEGACY_CONTRACT_SHA256)
        contract_path = inference.ROOT / 'bench/runtime/gotrack-refinement-states-v1.json'
        contract = inference._loads_unique(contract_path.read_bytes(),
                                           'expanded inference contract')
        inference._validate_contract(contract)
        self.assertEqual(contract['training'], dict(inference._EXPANDED_TRAINING))
        self.assertEqual(contract['expanded_artifact_mask_producer_compatibility'],
                         inference.EXPANDED_MASK_PRODUCER_COMPATIBILITY)
        self.assertEqual(set(contract['additional_runtime_files']),
                         set(inference.EXPECTED_RUNTIME_FILES))
        self.assertNotIn(inference.EXPANDED_CONTRACT_RELATIVE_PATH,
                         contract['additional_runtime_files'])
        # This archived contract describes ROOT25. New observer admission and
        # an optional CNOS scratch-bank argument have separate current pins.
        changed = {'bench/quality_gotrack.py', 'bench/quality_cnos.py'}
        for relative, expected in contract['additional_runtime_files'].items():
            if relative in changed:
                continue
            with self.subTest(relative=relative):
                self.assertEqual(inference._sha256_file(inference.ROOT / relative), expected)
        lock = json.loads((inference.ROOT / 'bench/runtime/adaptation-sources.json').read_bytes())['gotrack']['files']
        replacements = contract['reviewed_adapter_replacements']
        self.assertEqual(set(replacements), {'bench/quality_gotrack.py', 'bench/quality_runner.py'})
        self.assertEqual(contract['unchanged_training_files'],
                         {path: value for path, value in lock.items() if path not in replacements})
        for relative, record in contract['reviewed_adapter_replacements'].items():
            with self.subTest(replacement=relative):
                self.assertEqual(record['training_sha256'], lock[relative])
                if relative not in changed:
                    self.assertEqual(inference._sha256_file(inference.ROOT / relative),
                                     record['inference_sha256'])
        self.assertEqual(contract['source_closure_sha256'], inference._source_closure_hash(
            contract['unchanged_training_files'],
            contract['reviewed_adapter_replacements'],
            contract['additional_runtime_files']))
        mixed = copy.deepcopy(contract)
        mixed['training']['dataset_sha256'] = inference.DATASET_SHA256
        with self.assertRaisesRegex(ValueError, 'Training/export identity'):
            inference._validate_contract(mixed)
        mixed = copy.deepcopy(contract)
        mixed['expanded_artifact_mask_producer_compatibility']['producer']['position_grid'] = 'other'
        with self.assertRaisesRegex(ValueError, 'ROOT3 mask producer compatibility'):
            inference._validate_contract(mixed)
        legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
        legacy['expanded_artifact_mask_producer_compatibility'] = (
            copy.deepcopy(inference.EXPANDED_MASK_PRODUCER_COMPATIBILITY))
        with self.assertRaisesRegex(ValueError, 'unexpected or missing fields'):
            inference._validate_contract(legacy)

    def test_larger_runtime_has_current_source_pins(self):
        contract = json.loads((inference.ROOT / 'bench/runtime/gotrack-larger-v1.json').read_bytes())
        inference._validate_contract(contract)
        self.assertEqual(set(contract['reviewed_adapter_replacements']), {
            'bench/quality_gotrack.py', 'bench/quality_runner.py', 'bench/quality_cnos.py'})
        for relative, expected in contract['additional_runtime_files'].items():
            with self.subTest(runtime=relative):
                self.assertEqual(inference._sha256_file(inference.ROOT / relative), expected)
        for relative, record in contract['reviewed_adapter_replacements'].items():
            with self.subTest(replacement=relative):
                self.assertEqual(inference._sha256_file(inference.ROOT / relative), record['inference_sha256'])
        self.assertEqual(contract['source_closure_sha256'], inference._source_closure_hash(
            contract['unchanged_training_files'], contract['reviewed_adapter_replacements'],
            contract['additional_runtime_files']))

    def test_cloud_assertion_is_required_before_config_file_access(self):
        options = inference.options_from_values(config_path='missing.json',
            config_sha256='0' * 64, cloud_execution=False)
        with patch.dict(inference.os.environ, {'VISUALIZEIT_CLOUD': '0'}), \
             patch.object(inference.platform, 'system', return_value='Linux'), \
             self.assertRaisesRegex(ValueError, 'cloud-execution'):
            inference.prepare_inference(options, Path('missing-bundle'), {})

    def test_import_and_no_option_path_do_not_import_torch(self):
        # This is an offline preflight module; model-runtime import belongs to
        # the later ordinary runner branch after every binding has passed.
        self.assertNotIn('torch', sys.modules)

    def test_only_the_authenticated_three_input_bindings_validate(self):
        rows = copy.deepcopy([dict(row) for row in inference.EXPECTED_ACCEPTED_INPUTS])
        inference._validate_accepted_input_contract(rows)
        rows[0]['video_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'authenticated three-bundle trial'):
            inference._validate_accepted_input_contract(rows)
        rows = copy.deepcopy([dict(row) for row in inference.EXPECTED_ACCEPTED_INPUTS])
        with self.assertRaisesRegex(ValueError, 'authenticated three-bundle trial'):
            inference._validate_accepted_input_contract(rows[::-1])

    def test_authenticated_contract_parser_rejects_duplicate_and_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            inference._loads_unique(b'{"profile":"one","profile":"two"}', 'test contract')
        contract_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        contract['unexpected'] = True
        with self.assertRaisesRegex(ValueError, 'unexpected or missing fields'):
            inference._validate_contract(contract)

    def test_profile_preflight_accepts_exact_synthetic_physical_time_and_corrects_k_once(self):
        contract_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        options = inference.options_from_values(config_path=contract_path,
            config_sha256=inference._sha256_file(contract_path), cloud_execution=True)
        accepted = dict(inference.EXPECTED_ACCEPTED_INPUTS[0])
        manifest = {
            'schema_version': 2,
            'provenance_kind': 'synthetic',
            'units': 'metres',
            'asset': 'object.glb',
            'video': 'synthetic.mp4',
            'sequence_id': accepted['sequence_id'],
            'physical_object_id': accepted['physical_object_id'],
            'native_resolution': accepted['native_resolution'],
            'intrinsics': accepted['native_intrinsics'],
            'source_hashes': {'object.glb': accepted['mesh_sha256'],
                              'synthetic.mp4': accepted['video_sha256']},
            '_timing': {'clock_mode': 'capture_seconds_v1'},
        }
        with tempfile.TemporaryDirectory(dir=inference.ROOT / '.cache/cloud-prepare/test-temp') as folder, \
             patch.dict(inference.os.environ, {'VISUALIZEIT_CLOUD': '1'}), \
             patch.object(inference.platform, 'system', return_value='Linux'), \
             patch.object(inference, 'EXPECTED_PYTHON', (tuple(sys.version_info[:2]),)), \
             patch.object(inference, 'verify_source_compatibility', return_value={
                 'source_closure_sha256': 'd' * 64,
                 'resolved_model_cache_root': inference.APPROVED_CACHE_MOUNT}), \
             patch.object(inference, '_validate_accepted_input', return_value=accepted):
            context = inference.prepare_inference(options, Path(folder), manifest,
                runtime_packages=dict(inference.EXPECTED_PACKAGES))
        np.testing.assert_array_equal(context.original_intrinsics,
            [[576., 0., 319.5], [0., 576., 239.5], [0., 0., 1.]])
        np.testing.assert_array_equal(context.effective_intrinsics,
            [[576., 0., 319.], [0., 576., 239.], [0., 0., 1.]])

    def test_preflight_rejects_capture_wrong_clock_and_physical_camera_before_torch(self):
        contract_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        options = inference.options_from_values(config_path=contract_path,
            config_sha256=inference._sha256_file(contract_path), cloud_execution=True)
        with tempfile.TemporaryDirectory(dir=inference.ROOT / '.cache/cloud-prepare/test-temp') as folder, \
             patch.dict(inference.os.environ, {'VISUALIZEIT_CLOUD': '1'}), \
             patch.object(inference.platform, 'system', return_value='Linux'), \
             patch.object(inference, 'EXPECTED_PYTHON', (tuple(sys.version_info[:2]),)), \
             patch.object(inference, 'verify_source_compatibility', return_value={
                 'source_closure_sha256': 'd' * 64,
                 'resolved_model_cache_root': inference.APPROVED_CACHE_MOUNT}):
            capture_manifest = {'_capture': object(),
                                '_timing': {'clock_mode': 'capture_seconds_v1'}}
            with self.assertRaisesRegex(ValueError, 'does not support capture bundles'):
                inference.prepare_inference(options, Path(folder), capture_manifest,
                    runtime_packages=dict(inference.EXPECTED_PACKAGES))
            wrong_clock = {'_capture': None, '_timing': {'clock_mode': 'legacy_v1'}}
            with self.assertRaisesRegex(ValueError, 'pinned synthetic physical-time'):
                inference.prepare_inference(options, Path(folder), wrong_clock,
                    runtime_packages=dict(inference.EXPECTED_PACKAGES))
            physical_camera = {
                'schema_version': 2, 'provenance_kind': 'physical',
                '_capture': None, '_timing': {'clock_mode': 'capture_seconds_v1'},
            }
            with self.assertRaisesRegex(ValueError, 'non-synthetic/unknown manifest provenance'):
                inference.prepare_inference(options, Path(folder), physical_camera,
                    runtime_packages=dict(inference.EXPECTED_PACKAGES))
        self.assertNotIn('torch', sys.modules)

    def test_source_compatibility_rejects_incomplete_or_changed_closure(self):
        source_root = inference.ROOT
        lock_path = source_root / inference.TRAINING_LOCK_RELATIVE_PATH
        temp_root = inference.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root = Path(folder)
            target_lock = root / inference.TRAINING_LOCK_RELATIVE_PATH
            target_lock.parent.mkdir(parents=True)
            target_lock.write_bytes(lock_path.read_bytes())
            cache = root / '.cache/model-quality'
            cache.mkdir(parents=True)
            approved_mount = cache.resolve().as_posix()
            contract = json.loads((source_root / inference.CONTRACT_RELATIVE_PATH).read_text(encoding='utf-8'))
            contract['source_cache_mount'] = approved_mount
            with patch.object(inference, 'ROOT', root), \
                 patch.object(inference, 'APPROVED_CACHE_MOUNT', approved_mount), \
                 patch.object(inference, '_verify_relative_file'):
                result = inference.verify_source_compatibility(contract)
                self.assertEqual(result['historical_file_count'], 320)
                changed = copy.deepcopy(contract)
                changed['additional_runtime_files'].pop(next(iter(changed['additional_runtime_files'])))
                with self.assertRaisesRegex(ValueError, 'runtime closure has missing or unexpected paths'):
                    inference.verify_source_compatibility(changed)
                changed = copy.deepcopy(contract)
                changed['reviewed_adapter_replacements']['bench/quality_runner.py']['inference_sha256'] = 'f' * 64
                with self.assertRaisesRegex(ValueError, 'source closure digest mismatch'):
                    inference.verify_source_compatibility(changed)

    def test_legacy_request_uses_its_head_artifact_with_separately_authenticated_runtime(self):
        old_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        old_bytes = old_path.read_bytes()
        old = json.loads(old_bytes)
        effective = copy.deepcopy(old)
        effective['training'] = dict(inference._EXPANDED_TRAINING)
        effective['legacy_runtime_compatibility'] = dict(inference.LEGACY_RUNTIME_COMPATIBILITY)
        effective['expanded_artifact_mask_producer_compatibility'] = copy.deepcopy(
            inference.EXPANDED_MASK_PRODUCER_COMPATIBILITY)
        effective_bytes = (json.dumps(effective, sort_keys=True, separators=(',', ':')) + '\n').encode()
        temp_root = inference.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        accepted = dict(inference.EXPECTED_ACCEPTED_INPUTS[0])
        manifest = {
            'schema_version': 2, 'provenance_kind': 'synthetic', 'units': 'metres',
            'asset': 'object.glb', 'video': 'synthetic.mp4',
            'sequence_id': accepted['sequence_id'],
            'physical_object_id': accepted['physical_object_id'],
            'native_resolution': accepted['native_resolution'],
            'intrinsics': accepted['native_intrinsics'],
            'source_hashes': {'object.glb': accepted['mesh_sha256'],
                              'synthetic.mp4': accepted['video_sha256']},
            '_timing': {'clock_mode': 'capture_seconds_v1'},
        }
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            runtime_path = Path(folder) / 'expanded-runtime.json'
            runtime_path.write_bytes(effective_bytes)
            options = inference.options_from_values(
                config_path=old_path, config_sha256=inference.LEGACY_CONTRACT_SHA256,
                compatibility_config_path=runtime_path,
                compatibility_config_sha256=inference._sha256_bytes(effective_bytes),
                cloud_execution=True)
            with patch.dict(inference.os.environ, {'VISUALIZEIT_CLOUD': '1'}), \
                 patch.object(inference.platform, 'system', return_value='Linux'), \
                 patch.object(inference, 'EXPECTED_PYTHON', (tuple(sys.version_info[:2]),)), \
                 patch.object(inference, 'verify_source_compatibility', return_value={
                     'source_closure_sha256': 'd' * 64,
                     'resolved_model_cache_root': inference.APPROVED_CACHE_MOUNT}) as verify, \
                 patch.object(inference, '_validate_accepted_input', return_value=accepted):
                context = inference.prepare_inference(options, Path(folder), manifest,
                    runtime_packages=dict(inference.EXPECTED_PACKAGES))
        self.assertEqual(context.artifact['artifact_id'], 'adaptation-smoke-v1')
        self.assertEqual(context.artifact['export_sha256'], inference.EXPORT_SHA256)
        self.assertEqual(context.requested_contract_sha256, inference.LEGACY_CONTRACT_SHA256)
        self.assertEqual(context.runtime_contract_sha256,
                         inference._sha256_bytes(effective_bytes))
        self.assertEqual(context.identity_dict()['config_sha256'], inference.LEGACY_CONTRACT_SHA256)
        self.assertEqual(context.identity_dict()['runtime_artifact_id'],
                         'adaptation-expanded-continuation-v1')
        self.assertEqual(verify.call_args.args[0]['training']['export_sha256'],
                         inference.EXPANDED_EXPORT_SHA256)

    def test_legacy_compatibility_runtime_contract_tampering_fails_its_sha_pin(self):
        old_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        expanded_path = inference.ROOT / inference.EXPANDED_CONTRACT_RELATIVE_PATH
        expanded_bytes = expanded_path.read_bytes()
        tampered_bytes = expanded_bytes + b' '
        temp_root = inference.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            runtime_path = Path(folder) / 'expanded-runtime.json'
            runtime_path.write_bytes(tampered_bytes)
            options = inference.options_from_values(
                config_path=old_path, config_sha256=inference.LEGACY_CONTRACT_SHA256,
                compatibility_config_path=runtime_path,
                compatibility_config_sha256=inference._sha256_bytes(expanded_bytes),
                cloud_execution=True)
            with patch.dict(inference.os.environ, {'VISUALIZEIT_CLOUD': '1'}), \
                 patch.object(inference.platform, 'system', return_value='Linux'), \
                 self.assertRaisesRegex(ValueError, 'Runtime compatibility contract SHA-256 mismatch'):
                inference.prepare_inference(options, Path('unused'),
                    {'_capture': None, '_timing': {'clock_mode': 'capture_seconds_v1'}})

    def test_legacy_contract_cannot_claim_current_closure_without_compatibility(self):
        old_path = inference.ROOT / inference.CONTRACT_RELATIVE_PATH
        options = inference.options_from_values(
            config_path=old_path, config_sha256=inference.LEGACY_CONTRACT_SHA256,
            cloud_execution=True)
        with patch.dict(inference.os.environ, {'VISUALIZEIT_CLOUD': '1'}), \
             patch.object(inference.platform, 'system', return_value='Linux'), \
             patch.object(inference, 'verify_source_compatibility', side_effect=
                           inference.InferenceContractError('Inference source closure digest mismatch')) as verify:
            with self.assertRaisesRegex(ValueError, 'source closure digest mismatch'):
                inference.prepare_inference(options, Path('unused'),
                    {'_capture': None, '_timing': {'clock_mode': 'capture_seconds_v1'}},
                    runtime_packages=dict(inference.EXPECTED_PACKAGES))
        self.assertEqual(verify.call_args.args[0]['training']['export_sha256'],
                         inference.EXPORT_SHA256)


class AtomicHeadInstallTests(unittest.TestCase):
    class Tensor:
        layout = 'strided'

        def __init__(self, values):
            self.values = np.asarray(values)
            self.shape = self.values.shape
            self.dtype = self.values.dtype

        def clone(self):
            return type(self)(self.values.copy())

        def is_floating_point(self):
            return self.dtype.kind == 'f'

        def is_complex(self):
            return self.dtype.kind == 'c'

    class RNGState:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return type(self)(self.value)

    class Head:
        def __init__(self, *, fail_load=False, fail_device=False):
            self.values = {'projection.weight': AtomicHeadInstallTests.Tensor(
                np.array([0.25, -0.5], dtype=np.float32))}
            self.fail_load = fail_load
            self.fail_device = fail_device
            self.load_calls = 0
            self.strict = None
            self.device = None
            self.training = True
            self.requires_grad_value = True

        def state_dict(self):
            return self.values

        def load_state_dict(self, state, strict):
            self.load_calls += 1
            self.strict = strict
            self.values = dict(state)
            if self.fail_load:
                raise RuntimeError('mock partial strict-load failure')

        def to(self, device):
            self.device = device
            if self.fail_device:
                raise RuntimeError('mock device preparation failure')
            return self

        def eval(self):
            self.training = False
            return self

        def requires_grad_(self, enabled):
            self.requires_grad_value = enabled
            return self

    class Network:
        def __init__(self, *, fail_load=False, fail_device=False):
            object.__setattr__(self, 'head_assignments', 0)
            self.pose_head = AtomicHeadInstallTests.Head(
                fail_load=fail_load, fail_device=fail_device)
            self.backbone = object()
            self.decoder = object()

        def __setattr__(self, name, value):
            if name == 'pose_head' and hasattr(self, 'pose_head'):
                object.__setattr__(self, 'head_assignments', self.head_assignments + 1)
            object.__setattr__(self, name, value)

    class TorchDouble:
        strided = 'strided'

        def __init__(self, export, expected_bytes):
            self.export = export
            self.expected_bytes = expected_bytes
            self.load_calls = 0
            self.cpu_state = AtomicHeadInstallTests.RNGState('cpu-before')
            self.cuda_state = [AtomicHeadInstallTests.RNGState('cuda-before')]
            self.random = types.SimpleNamespace(
                get_rng_state=lambda: self.cpu_state.clone(),
                set_rng_state=self._set_cpu_state)
            self.cuda = types.SimpleNamespace(
                is_initialized=lambda: True,
                get_rng_state_all=lambda: [state.clone() for state in self.cuda_state],
                set_rng_state_all=self._set_cuda_state)

        def _set_cpu_state(self, state):
            self.cpu_state = state.clone()

        def _set_cuda_state(self, states):
            self.cuda_state = [state.clone() for state in states]

        @staticmethod
        def is_tensor(value):
            return isinstance(value, AtomicHeadInstallTests.Tensor)

        @staticmethod
        def isfinite(value):
            return np.isfinite(value.values)

        def load(self, stream, *, map_location, weights_only):
            self.load_calls += 1
            if map_location != 'cpu' or weights_only is not True:
                raise AssertionError('export must use CPU weights_only loading')
            if stream.read() != self.expected_bytes:
                raise AssertionError('torch.load did not receive the bytes that were hashed')
            self.cpu_state = AtomicHeadInstallTests.RNGState('cpu-mutated')
            self.cuda_state = [AtomicHeadInstallTests.RNGState('cuda-mutated')]
            return self.export

    def _export(self, state=None, **changes):
        value = {
            'schema': inference.EXPORT_SCHEMA,
            'head': state if state is not None else {
                'projection.weight': self.Tensor(np.array([1., 2.], dtype=np.float32))},
            'source_lock_sha256': inference.TRAINING_LOCK_SHA256,
            'base_checkpoint_sha256': inference.BASE_CHECKPOINT_SHA256,
            'dataset_sha256': inference.DATASET_SHA256,
            'config_sha256': inference.TRAINING_CONFIG_SHA256,
            'validation': {'kind': 'diagnostic_only'},
            'quality_claim': False,
            'deployment': inference.EXPORT_DEPLOYMENT,
        }
        value.update(changes)
        return value

    def _install(self, export, *, requested_sha=None, fail_load=False, fail_device=False,
                 expected_bytes=None, device='cuda:7'):
        payload = b'mock authenticated tensor export'
        temp_root = inference.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            path = Path(folder) / 'head.pt'
            path.write_bytes(payload)
            requested = requested_sha or inference._sha256_bytes(payload)
            artifact = {
                'artifact_id': 'test-adapted',
                'export_sha256': requested,
                'export_bytes': len(payload) if expected_bytes is None else expected_bytes,
                'training': dict(inference._LEGACY_TRAINING),
            }
            context = types.SimpleNamespace(head_path=path,
                requested_head_sha256=requested,
                artifact=artifact,
                identity_dict=lambda installed=False: {
                    'head': {'installed_sha256': requested if installed else None},
                    'head_installation': {'requested': True, 'installed': bool(installed)},
                })
            network = self.Network(fail_load=fail_load, fail_device=fail_device)
            original = network.pose_head
            original_values = {key: value.values.copy() for key, value in original.state_dict().items()}
            backbone, decoder = network.backbone, network.decoder
            torch_mock = self.TorchDouble(export, payload)
            try:
                result = inference.install_adapted_head(network, context, device=device,
                                                         torch_module=torch_mock)
                error = None
            except Exception as exc:
                result, error = None, exc
            return (network, original, original_values, backbone, decoder, torch_mock, result, error)

    def test_each_registry_artifact_accepts_only_its_own_export_header(self):
        headers = {}
        for digest, artifact in inference.ARTIFACT_REGISTRY.items():
            training = artifact['training']
            header = {
                'schema': inference.EXPORT_SCHEMA,
                'head': {'projection.weight': object()},
                'source_lock_sha256': training['training_lock_sha256'],
                'base_checkpoint_sha256': training['base_checkpoint_sha256'],
                'dataset_sha256': training['dataset_sha256'],
                'config_sha256': training['training_config_sha256'],
                'validation': {'kind': 'diagnostic_only'},
                'quality_claim': False,
                'deployment': training['export_deployment'],
            }
            headers[digest] = header
            inference._validate_export_header(header, artifact)
        for digest, header in headers.items():
            for other_digest, artifact in inference.ARTIFACT_REGISTRY.items():
                if digest != other_digest:
                    with self.subTest(requested=digest, supplied=other_digest):
                        with self.assertRaisesRegex(ValueError, 'header'):
                            inference._validate_export_header(header, artifact)

    def test_valid_head_is_frozen_assigned_once_on_selected_device_and_preserves_rng(self):
        (network, original, _, backbone, decoder, torch_mock, result, error) = self._install(self._export())
        self.assertIsNone(error)
        self.assertIs(result[0], network)
        self.assertIsNot(network.pose_head, original)
        self.assertEqual(network.head_assignments, 1)
        self.assertEqual(network.pose_head.load_calls, 1)
        self.assertIs(network.pose_head.strict, True)
        self.assertEqual(network.pose_head.device, 'cuda:7')
        self.assertFalse(network.pose_head.training)
        self.assertFalse(network.pose_head.requires_grad_value)
        self.assertIs(network.backbone, backbone)
        self.assertIs(network.decoder, decoder)
        self.assertEqual(torch_mock.cpu_state.value, 'cpu-before')
        self.assertEqual(torch_mock.cuda_state[0].value, 'cuda-before')
        self.assertEqual(result[1]['head']['installed_sha256'], inference._sha256_bytes(b'mock authenticated tensor export'))

    def test_digest_mismatch_does_not_deserialize_or_mutate_live_network(self):
        payload = b'mock authenticated tensor export'
        temp_root = inference.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            path = Path(folder) / 'head.pt'
            path.write_bytes(payload)
            network = self.Network()
            original = network.pose_head
            torch_mock = self.TorchDouble(self._export(), payload)
            context = types.SimpleNamespace(head_path=path, requested_head_sha256='0' * 64)
            artifact = {'artifact_id': 'test-adapted',
                'export_sha256': '0' * 64, 'export_bytes': len(payload),
                'training': dict(inference._LEGACY_TRAINING)}
            context.artifact = artifact
            with self.assertRaisesRegex(ValueError, 'digest or size mismatch'):
                inference.install_adapted_head(network, context, device='cuda:7',
                                                torch_module=torch_mock)
            self.assertEqual(torch_mock.load_calls, 0)
            self.assertIs(network.pose_head, original)
            self.assertEqual(network.head_assignments, 0)

    def test_wrong_artifact_size_does_not_deserialize_or_mutate_live_network(self):
        (network, original, _, _, _, torch_mock, _, error) = self._install(
            self._export(), expected_bytes=len(b'mock authenticated tensor export') + 1)
        self.assertIsInstance(error, ValueError)
        self.assertRegex(str(error), 'digest or size mismatch')
        self.assertEqual(torch_mock.load_calls, 0)
        self.assertIs(network.pose_head, original)
        self.assertEqual(network.head_assignments, 0)

    def test_bad_schema_dtype_nonfinite_partial_load_and_device_failures_are_atomic(self):
        invalid_exports = [
            self._export(schema='unexpected'),
            self._export(state={'projection.weight': self.Tensor(np.array([1, 2], dtype=np.int32))}),
            self._export(state={'projection.weight': self.Tensor(np.array([np.nan, 2.], dtype=np.float32))}),
        ]
        for export in invalid_exports:
            with self.subTest(export=export):
                network, original, original_values, backbone, decoder, torch_mock, _, error = self._install(export)
                self.assertIsInstance(error, ValueError)
                self.assertIs(network.pose_head, original)
                self.assertEqual(network.head_assignments, 0)
                self.assertIs(network.backbone, backbone)
                self.assertIs(network.decoder, decoder)
                self.assertEqual(original.load_calls, 0)
                self.assertEqual(torch_mock.cpu_state.value, 'cpu-before')
                self.assertEqual(torch_mock.cuda_state[0].value, 'cuda-before')
                for key, values in original_values.items():
                    np.testing.assert_array_equal(original.state_dict()[key].values, values)
        for fail_load, fail_device in ((True, False), (False, True)):
            with self.subTest(fail_load=fail_load, fail_device=fail_device):
                payload = b'mock authenticated tensor export'
                temp_root = inference.ROOT / '.cache/cloud-prepare/test-temp'
                temp_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=temp_root) as folder:
                    path = Path(folder) / 'head.pt'
                    path.write_bytes(payload)
                    network = self.Network(fail_load=fail_load, fail_device=fail_device)
                    original = network.pose_head
                    original_values = {key: value.values.copy() for key, value in original.state_dict().items()}
                    backbone, decoder = network.backbone, network.decoder
                    torch_mock = self.TorchDouble(self._export(), payload)
                    context = types.SimpleNamespace(head_path=path,
                        requested_head_sha256=inference._sha256_bytes(payload),
                        artifact={'artifact_id': 'test-adapted',
                            'export_sha256': inference._sha256_bytes(payload),
                            'export_bytes': len(payload),
                            'training': dict(inference._LEGACY_TRAINING)})
                    with self.assertRaises(RuntimeError):
                        inference.install_adapted_head(network, context, device='cuda:7',
                                                        torch_module=torch_mock)
                    self.assertIs(network.pose_head, original)
                    self.assertEqual(network.head_assignments, 0)
                    self.assertIs(network.backbone, backbone)
                    self.assertIs(network.decoder, decoder)
                    self.assertEqual(original.load_calls, 0)
                    for key, values in original_values.items():
                        np.testing.assert_array_equal(original.state_dict()[key].values, values)
                    self.assertEqual(torch_mock.cpu_state.value, 'cpu-before')
                    self.assertEqual(torch_mock.cuda_state[0].value, 'cuda-before')


class IntegerGeometryTests(unittest.TestCase):
    @staticmethod
    def _rotation_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

    def _fixture(self):
        crop_k = np.array([[375., 0., 126.3], [0., 419., 143.8], [0., 0., 1.]])
        native_k = np.array([[576., 0., 319.], [0., 576., 239.], [0., 0., 1.]])
        crop_from_native = np.eye(4)
        crop_from_native[:3, :3] = self._rotation_y(.17)
        native_from_object = np.eye(4)
        native_from_object[:3, :3] = self._rotation_y(-.21)
        native_from_object[:3, 3] = [113., -49., 1420.]
        depth = np.full((280, 280), 1500., dtype=np.float64)
        flow = np.zeros((280, 280, 2), dtype=np.float64)
        weight = np.ones((280, 280), dtype=np.float64)
        rendered = np.ones((280, 280), dtype=np.float32)
        observed = np.ones((280, 280), dtype=np.float32)
        return (crop_k, native_k, crop_from_native, native_from_object,
                depth, flow, weight, rendered, observed)

    def test_projection_units_inverse_crop_and_half_pixel_control(self):
        from .quality_gotrack import _integer_profile_correspondences

        (crop_k, native_k, c, pose, depth, flow, weight,
         rendered, observed) = self._fixture()
        y, x = 140, 141
        flow[y, x] = [.25, .75]
        rows = _integer_profile_correspondences(depth, crop_k, c, pose, flow, weight,
                                                 rendered, observed, native_k)
        crop_pose, object_mm, target, native_uv, _, valid, integer_pixels, crop_xyz = rows

        # Expected XYZ is d * inv(K_crop)[x,y,1] / ray_z, then transformed
        # from crop-camera millimetres into the model coordinate system.
        pixel = np.array([float(x), float(y), 1.])
        ray = np.linalg.inv(crop_k) @ pixel
        expected_crop_mm = ray * (depth[y, x] / ray[2])
        expected_object_mm = crop_pose[:3, :3].T @ (expected_crop_mm - crop_pose[:3, 3])
        self.assertTrue(valid[y, x])
        np.testing.assert_array_equal(integer_pixels[y, x], [x, y])
        np.testing.assert_allclose(crop_xyz[y, x], expected_crop_mm, rtol=0., atol=1e-10)
        np.testing.assert_allclose(object_mm[y, x], expected_object_mm, rtol=0., atol=1e-10)
        np.testing.assert_allclose(target[y, x], [x + .25, y + .75], rtol=0., atol=1e-12)

        # Native UV uses C.R.T * inv(K_crop) * q, then K_native; it is an
        # inverse-crop ray mapping and does not depend on depth.
        q = np.array([x + .25, y + .75, 1.])
        native_ray = c[:3, :3].T @ np.linalg.inv(crop_k) @ q
        projected = native_k @ native_ray
        expected_native_uv = projected[:2] / projected[2]
        np.testing.assert_allclose(native_uv[y, x], expected_native_uv, rtol=0., atol=1e-10)

        # The historical renderer samples at (x+.5,y+.5); Kn=K0-(.5,.5)
        # makes integer pixel indices reproduce that ray. Deliberately lifting
        # at half centers through K_crop leaves a 0.5 px reprojection error.
        wrong_ray = np.linalg.inv(crop_k) @ np.array([x + .5, y + .5, 1.])
        wrong_crop_point = wrong_ray * (depth[y, x] / wrong_ray[2])
        wrong_projection = crop_k @ wrong_crop_point
        wrong_uv = wrong_projection[:2] / wrong_projection[2]
        self.assertAlmostEqual(float(np.max(np.abs(wrong_uv - [x, y]))), .5, places=10)
        self.assertGreater(float(np.linalg.norm(wrong_crop_point - crop_xyz[y, x])), 0.)

    def test_depth_target_confidence_mask_and_safe_boundary_rules(self):
        from .quality_gotrack import _integer_profile_correspondences

        (crop_k, native_k, c, pose, depth, flow, weight,
         rendered, observed) = self._fixture()
        depth[0, 0] = 0.
        depth[0, 1] = np.nan
        depth[0, 2] = -1.
        flow[10, 10] = [-10., -10.]       # target exactly (0,0): included
        flow[20, 20] = [260., 0.]         # target x=280: excluded
        flow[30, 30] = [-30.25, 0.]       # target x<0: excluded
        flow[40, 40] = [0., 239.999]      # target y<280: included, floor 279
        flow[50, 50] = [.999, .999]       # destination uses floor target
        flow[60, 59] = [1.1, 0.]          # floor destination [60,60] is masked
        observed[60, 60] = 0.
        observed[70, 70] = 0.             # source is masked
        weight[80, 80] = .3               # strict confidence cutoff
        rendered[90, 90] = 0.
        flow[100, 100] = [np.nan, 0.]
        flow[101, 101] = [np.inf, 0.]
        flow[102, 102] = [1e300, -1e300]  # finite but out of bounds before floor

        *_, valid, _, _ = _integer_profile_correspondences(
            depth, crop_k, c, pose, flow, weight, rendered, observed, native_k)
        for point in ((0, 0), (0, 1), (0, 2), (20, 20), (30, 30),
                      (60, 59), (70, 70), (80, 80), (90, 90),
                      (100, 100), (101, 101), (102, 102)):
            self.assertFalse(valid[point], point)
        self.assertTrue(valid[10, 10])
        self.assertTrue(valid[40, 40])
        self.assertTrue(valid[50, 50])

    def test_invalid_crop_camera_or_transform_fails_closed(self):
        from .quality_gotrack import _integer_profile_correspondences

        fixture = self._fixture()
        crop_k, native_k, c, pose, depth, flow, weight, rendered, observed = fixture
        singular = crop_k.copy()
        singular[0, 0] = 0.
        with self.assertRaises(ValueError):
            _integer_profile_correspondences(depth, singular, c, pose, flow, weight,
                                              rendered, observed, native_k)
        translated = c.copy()
        translated[0, 3] = .01
        with self.assertRaises(ValueError):
            _integer_profile_correspondences(depth, crop_k, translated, pose, flow,
                                              weight, rendered, observed, native_k)
        reflected = c.copy()
        reflected[0, 0] *= -1.
        with self.assertRaises(ValueError):
            _integer_profile_correspondences(depth, crop_k, reflected, pose, flow,
                                              weight, rendered, observed, native_k)

    def test_ordinary_refine_keeps_float_crop_tensors_and_runs_five_iterations(self):
        from .quality_contract import Frame
        from .quality_gotrack import GoTrackRefiner

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def __getitem__(self, item):
                return Tensor(self.value[item])

            def permute(self, *axes):
                return Tensor(self.value.transpose(axes))

            def float(self):
                return Tensor(self.value.astype(np.float32))

            def to(self, _device):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def detach(self):
                return self

            def __truediv__(self, value):
                return Tensor(self.value / value)

        (crop_k, native_k, c, pose, depth, flow, weight,
         rendered, observed) = self._fixture()
        query_rgb = np.full((1, 3, 280, 280), .375, dtype=np.float32)
        template_rgb = np.full((1, 3, 280, 280), .625, dtype=np.float32)
        crop_mask = np.ones((1, 280, 280), dtype=np.float32)
        template = types.SimpleNamespace(
            rgbs=Tensor(template_rgb), masks=Tensor(rendered[None]), depths=Tensor(depth[None]))
        data = {'crop_rgbs': Tensor(query_rgb), 'crop_masks': Tensor(crop_mask), 'templates': template}
        camera = object()
        upstream = MagicMock(return_value=(data, [camera], [Tensor(c)]))

        class Network:
            def __init__(self):
                self.queries = []

            def __call__(self, query, _template, _mask):
                self.queries.append(query)
                flow_chw = flow.transpose(2, 0, 1)[None]
                return Tensor(flow_chw), Tensor(weight[None])

        network = Network()
        renderer = types.SimpleNamespace(render_policy='legacy_v1',
            coordinate_mode='integer_centers_v1', unlit=True,
            disable_multisampling=True, vertices_m=np.zeros((8, 3)))
        refiner = GoTrackRefiner(network, renderer, 1, 'cpu',
            inference_geometry=inference.PROFILE)
        refiner._solve_pnp_ransac = MagicMock(side_effect=lambda _obj, _target, _k, rvec, tvec:
            (True, rvec, tvec, np.arange(6, dtype=np.int32).reshape((-1, 1))))
        torch_mock = types.SimpleNamespace(from_numpy=lambda value: Tensor(value),
                                           inference_mode=nullcontext)
        data_util = types.SimpleNamespace(compute_gotrack_inputs_from_init_poses=upstream)
        misc = types.SimpleNamespace(get_intrinsic_matrix=lambda _camera: crop_k)
        structs = types.SimpleNamespace(PinholePlaneCameraModel=lambda **kwargs: kwargs)
        utils = types.ModuleType('utils')
        utils.data_util, utils.misc, utils.structs = data_util, misc, structs
        utils.transform3d = types.SimpleNamespace()
        frame = Frame(7, np.zeros((480, 640, 3), dtype=np.uint8), native_k,
                      timestamp_s=1., clock_mode='capture_seconds_v1')
        seed = np.eye(4)
        seed[2, 3] = 1.42

        with patch.dict(sys.modules, {'torch': torch_mock, 'utils': utils}), \
             patch('bench.quality_gotrack.cv2.solvePnPRefineLM', side_effect=lambda _obj, _img, _k, _d, r, t: (r, t)):
            candidate = refiner.refine(frame, np.ones((480, 640), dtype=np.uint8), seed)

        self.assertIsNotNone(candidate)
        self.assertEqual(upstream.call_count, 5)
        self.assertEqual(len(network.queries), 5)
        self.assertEqual(refiner._solve_pnp_ransac.call_count, 5)
        for query in network.queries:
            # Each network call received the exact upstream float RGB crop.
            self.assertIs(query, data['crop_rgbs'])
            np.testing.assert_array_equal(query.numpy(), query_rgb)
        input_images = [call.kwargs['input_rgbs'] for call in upstream.call_args_list]
        self.assertTrue(all(image is input_images[0] for image in input_images))
        self.assertEqual(input_images[0].numpy().shape, (1, 3, 480, 640))
        self.assertFalse(np.any(input_images[0].numpy()))
        for call in upstream.call_args_list:
            self.assertEqual(call.kwargs['crop_size'], (280, 280))
            self.assertEqual(call.kwargs['crop_rel_pad'], .1)
            self.assertEqual(call.kwargs['cropping_type'], 'perspective_2d_box')


class PhysicalMaskProducerTests(unittest.TestCase):
    def _record_for(self, producer):
        accepted = {
            'input_sha256': '1' * 64,
            'mesh_path': 'object.glb', 'mesh_sha256': '2' * 64,
            'video_path': 'synthetic.mp4', 'video_sha256': '3' * 64,
        }
        record = copy.deepcopy(producer)
        record['input_manifest_sha256'] = accepted['input_sha256']
        record['source_sha256'] = {
            'object.glb': accepted['mesh_sha256'],
            'synthetic.mp4': accepted['video_sha256'],
        }
        record['provenance'] = {
            'input_manifest_sha256': accepted['input_sha256'],
            **copy.deepcopy(producer['provenance']),
        }
        return accepted, record

    def _legacy_context(self, accepted):
        return types.SimpleNamespace(profile=inference.PROFILE,
            contract={'mask_producer': inference.EXPECTED_SOURCE18_MASK_PRODUCER},
            artifact=inference.ARTIFACT_REGISTRY[inference.EXPORT_SHA256],
            accepted_input=accepted)

    def test_source18_physical_schema2_mask_producer_is_reused(self):
        accepted, record = self._record_for(inference.EXPECTED_SOURCE18_MASK_PRODUCER)
        context = self._legacy_context(accepted)
        inference.validate_physical_mask_producer(context, record)

    def test_expanded_artifact_accepts_only_the_audited_root3_producer(self):
        accepted, record = self._record_for(inference.EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER)
        contract_path = inference.ROOT / inference.EXPANDED_CONTRACT_RELATIVE_PATH
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        context = types.SimpleNamespace(profile=inference.PROFILE, contract=contract,
            artifact=inference.ARTIFACT_REGISTRY[inference.EXPANDED_EXPORT_SHA256],
            accepted_input=accepted)
        inference.validate_physical_mask_producer(context, record)

    def test_legacy_and_legacy_compatibility_contexts_reject_root3_producer(self):
        accepted, record = self._record_for(inference.EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER)
        with self.assertRaisesRegex(ValueError, 'unexpected producer field or artifact pairing'):
            inference.validate_physical_mask_producer(self._legacy_context(accepted),
                                                      copy.deepcopy(record))
        runtime_contract = json.loads((inference.ROOT /
            inference.EXPANDED_CONTRACT_RELATIVE_PATH).read_text(encoding='utf-8'))
        compatibility_context = types.SimpleNamespace(profile=inference.PROFILE,
            contract=runtime_contract,
            artifact=inference.ARTIFACT_REGISTRY[inference.EXPORT_SHA256],
            accepted_input=accepted)
        with self.assertRaisesRegex(ValueError, 'unexpected producer field or artifact pairing'):
            inference.validate_physical_mask_producer(compatibility_context, record)

    def test_expanded_root3_producer_rejects_every_other_producer_difference(self):
        accepted, record = self._record_for(inference.EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER)
        contract = json.loads((inference.ROOT /
            inference.EXPANDED_CONTRACT_RELATIVE_PATH).read_text(encoding='utf-8'))
        context = types.SimpleNamespace(profile=inference.PROFILE, contract=contract,
            artifact=inference.ARTIFACT_REGISTRY[inference.EXPANDED_EXPORT_SHA256],
            accepted_input=accepted)
        for path, value in (
                (('model',), 'SAM 2.1 changed'),
                (('provenance', 'adapter_sha256', 'quality_gotrack'), 'f' * 64),
                (('provenance', 'source_revisions', 'sam2'), '0' * 40)):
            changed = copy.deepcopy(record)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                inference.validate_physical_mask_producer(context, changed)

    def test_wrong_producer_file_or_input_hash_is_rejected(self):
        accepted, record = self._record_for(inference.EXPECTED_SOURCE18_MASK_PRODUCER)
        context = self._legacy_context(accepted)
        changed = copy.deepcopy(record)
        changed['provenance']['adapter_sha256']['quality_runner'] = 'f' * 64
        with self.assertRaises(ValueError):
            inference.validate_physical_mask_producer(context, changed)
        changed = copy.deepcopy(record)
        changed['source_sha256']['synthetic.mp4'] = 'f' * 64
        with self.assertRaises(ValueError):
            inference.validate_physical_mask_producer(context, changed)
        changed = copy.deepcopy(record)
        changed['provenance']['input_manifest_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            inference.validate_physical_mask_producer(context, changed)


if __name__ == '__main__':
    unittest.main()
