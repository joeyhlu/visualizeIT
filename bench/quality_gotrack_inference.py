"""Authenticated opt-in inference contract for the adapted GoTrack pose head.

This module is deliberately importable without Torch or the upstream model.
It validates the reviewed runtime closure, exact synthetic bundle bindings and
cloud assertion before the ordinary pose runner constructs any neural state.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
from types import MappingProxyType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = 'gotrack_inference_contract_v1'
PROFILE = 'gotrack_integer_legacy_synthetic_v1'
CONTRACT_RELATIVE_PATH = 'bench/runtime/gotrack-inference-v1.json'
EXPANDED_CONTRACT_RELATIVE_PATH = 'bench/runtime/gotrack-inference-expanded-v1.json'
AUTHENTICATED_PLAN_RELATIVE_PATH = '.cache/cloud-prepare/adaptation-smoke-plan-v2.json'
AUTHENTICATED_PLAN_SHA256 = 'ec5d5f8701f0b3340619637eb223ac2ea91bb544a872f987e353b5fe25407939'
AUTHENTICATED_DIAGNOSTIC_RELATIVE_PATH = '.cache/cloud-prepare/adaptation-label-diagnostic-v1.json'
AUTHENTICATED_DIAGNOSTIC_SHA256 = '504bb0a2cf638ed5274ed65c6a652616e34ad51f94c8da11abae69044b5de5b1'
TRAINING_LOCK_RELATIVE_PATH = 'bench/runtime/adaptation-sources.json'
TRAINING_LOCK_SHA256 = '44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33'
FEATURE_LOCK_SHA256 = 'eb6ecea80a2e3af037b0b1a9bfab26dd7bf2701a5a5e3cf6b615fac8ab1c53cf'
FEATURE_RECIPE_SHA256 = '2a6c03ac8f5a67795b3b6f0ea239ce45c788a898acf9472ff0c7dc80f914e596'
FEATURE_PRODUCER_SHA256 = '507eb4228a24d26b2d22b902a1ebbc4e2887694fb29b799e3be4bcc2c512df08'
PRELABEL_SEAL_SHA256 = '42eabdaca40bba15332b8db783f003b534d4c125105886ad5438115bb90f80a1'
DATASET_SHA256 = '1295079545086ef11abb48e7975e5544832677568d49b7b46760e45979f5b5d4'
TRAINING_CONFIG_SHA256 = '3b5da58a42f4ee2e622e16dfc39eb746b2df30d2349e0b1625df8fe9d3a75979'
BASE_CHECKPOINT_SHA256 = 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'
UPSTREAM_REVISION = '68f76055755f2a4a8967e13ece834f975f008bdf'
EXPORT_SHA256 = '942b35b52536fe303a9701296b27e90009b88a89216b0883cf045cf221ed7b41'
EXPORT_BYTES = 80_141_142
LEGACY_CONTRACT_SHA256 = '889104c72dd25b448d24e215f72efbc7e78064838e547ba5d2337f76cecc82d4'
EXPANDED_EXPORT_SHA256 = 'e23794bcf91f2b3de48d30c0a1fed9c088c6e0119d8d64b5c1b2718adc4e3b96'
EXPANDED_EXPORT_BYTES = 80_144_086
EXPANDED_TRAINING_LOCK_SHA256 = '44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33'
EXPANDED_FEATURE_LOCK_SHA256 = '44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33'
EXPANDED_FEATURE_RECIPE_SHA256 = 'a1ac3f71daf8d20d2d8fdc360357d894f807d387fbe9a9abef2bdad1f8e9aeac'
EXPANDED_PRELABEL_SEAL_SHA256 = '95e6daf560b08e5872781f97e0e91ec9ba8287f8bd40719dcedf6fc23e3cf8b3'
EXPANDED_DATASET_SHA256 = 'bcb07d208cc922a0871781231c5da13ab97b7d7d5d87ff492e7a4a0381632a78'
EXPANDED_TRAINING_CONFIG_SHA256 = '1e74728a85df74b00bc84b2e106414b1c05d698ee74c857a01c72027310de0ae'
EXPORT_SCHEMA = 'correspondence_head_export_v1'
EXPORT_DEPLOYMENT = 'explicit_cloud_research_only_pose_head_replacement'
RENDER_POLICY = 'legacy_v1'
COORDINATE_MODE = 'integer_centers_v1'
RASTER_CONVENTION = 'legacy_opengl_native_principal_minus_half_to_integer_v1'
APPROVED_CACHE_MOUNT = '/teamspace/studios/this_studio/visualizeit-cloud-v2/.cache/model-quality'
NATIVE_CORRECTION = (-0.5, -0.5)
EXPECTED_PYTHON = ((3, 10), (3, 11))
EXPECTED_PACKAGES = MappingProxyType({
    'torch': '2.5.1', 'torchvision': '0.20.1', 'numpy': '1.26.4', 'einops': '0.8.0',
})

_TRAINING_FIELDS = frozenset({
    'training_lock_path', 'training_lock_sha256', 'feature_lock_sha256',
    'feature_recipe_sha256', 'feature_producer_sha256', 'prelabel_seal_sha256',
    'dataset_sha256', 'training_config_sha256', 'base_checkpoint_sha256',
    'upstream_revision', 'export_sha256', 'export_bytes', 'export_schema',
    'export_deployment', 'head_interface',
})
_LEGACY_TRAINING = MappingProxyType({
    'training_lock_path': TRAINING_LOCK_RELATIVE_PATH,
    'training_lock_sha256': TRAINING_LOCK_SHA256,
    'feature_lock_sha256': FEATURE_LOCK_SHA256,
    'feature_recipe_sha256': FEATURE_RECIPE_SHA256,
    'feature_producer_sha256': FEATURE_PRODUCER_SHA256,
    'prelabel_seal_sha256': PRELABEL_SEAL_SHA256,
    'dataset_sha256': DATASET_SHA256,
    'training_config_sha256': TRAINING_CONFIG_SHA256,
    'base_checkpoint_sha256': BASE_CHECKPOINT_SHA256,
    'upstream_revision': UPSTREAM_REVISION,
    'export_sha256': EXPORT_SHA256,
    'export_bytes': EXPORT_BYTES,
    'export_schema': EXPORT_SCHEMA,
    'export_deployment': EXPORT_DEPLOYMENT,
    'head_interface': 'DPTHead-defaults/decoder-feature-map/280x280/reference-to-query-pixels/confidence-mask',
})
_EXPANDED_TRAINING = MappingProxyType({
    **dict(_LEGACY_TRAINING),
    'training_lock_sha256': EXPANDED_TRAINING_LOCK_SHA256,
    'feature_lock_sha256': EXPANDED_FEATURE_LOCK_SHA256,
    'feature_recipe_sha256': EXPANDED_FEATURE_RECIPE_SHA256,
    'prelabel_seal_sha256': EXPANDED_PRELABEL_SEAL_SHA256,
    'dataset_sha256': EXPANDED_DATASET_SHA256,
    'training_config_sha256': EXPANDED_TRAINING_CONFIG_SHA256,
    'export_sha256': EXPANDED_EXPORT_SHA256,
    'export_bytes': EXPANDED_EXPORT_BYTES,
})
LEGACY_RUNTIME_COMPATIBILITY = MappingProxyType({
    'schema': 'gotrack_legacy_runtime_compatibility_v1',
    'requested_contract_sha256': LEGACY_CONTRACT_SHA256,
    'requested_export_sha256': EXPORT_SHA256,
    'rule': 'same_geometry_inputs_packages_mask_producer_and_nonwrapper_runtime_bytes',
})
ARTIFACT_REGISTRY = MappingProxyType({
    EXPORT_SHA256: MappingProxyType({
        'artifact_id': 'adaptation-smoke-v1',
        'export_sha256': EXPORT_SHA256,
        'export_bytes': EXPORT_BYTES,
        'training': _LEGACY_TRAINING,
    }),
    EXPANDED_EXPORT_SHA256: MappingProxyType({
        'artifact_id': 'adaptation-expanded-continuation-v1',
        'export_sha256': EXPANDED_EXPORT_SHA256,
        'export_bytes': EXPANDED_EXPORT_BYTES,
        'training': _EXPANDED_TRAINING,
    }),
})
_RUNTIME_COMPATIBILITY_WRAPPERS = frozenset({
    'bench/quality_gotrack_inference.py',
    'bench/quality_runner.py',
    'bench/quality_offline.py',
})
EXPECTED_ACCEPTED_INPUTS = (
    {
        'sequence_id': 'ranker-multiview-hot3d17-v1-yaw0-pitchm25',
        'physical_object_id': 'hot3d:obj_000017',
        'split': 'train',
        'input_sha256': '600bee44fb89561804459aeacadb8d33763091a8dbf73598728437029305d77c',
        'video_path': 'synthetic.mp4',
        'video_sha256': 'c2b3464401e74dfa99b2681d5084379a26d55df9d116ce8ded3cad122594e79b',
        'mesh_path': 'object.glb',
        'mesh_sha256': 'e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc',
        'generation_receipt_path': '../generation.json',
        'generation_receipt_sha256': '4e8c03dbebd19b6f5a9ea1c7111948175e82b353ba40cd50fceab4c9ec14cef4',
        'native_resolution': [640, 480],
        'native_intrinsics': [[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]],
        'renderer_provenance': 'quality-ranker-synthetic-generation-v1/legacy_opengl_centers',
    },
    {
        'sequence_id': 'ranker-multiview-hot3d17-v1-yaw0-pitchp25',
        'physical_object_id': 'hot3d:obj_000017',
        'split': 'train',
        'input_sha256': '9688fd8df353f0421da3d2cf37bfc91d77ec0f953dcc89783b4c9e4449f3839c',
        'video_path': 'synthetic.mp4',
        'video_sha256': 'c0b049ec4170f0b4db1f2821002c844b51622efa8325775551540e66d01dbd37',
        'mesh_path': 'object.glb',
        'mesh_sha256': 'e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc',
        'generation_receipt_path': '../generation.json',
        'generation_receipt_sha256': 'b0c06f72a23b5f29dfff1d01596704a59b705e54f345fddb1dd2d47c0ecca04d',
        'native_resolution': [640, 480],
        'native_intrinsics': [[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]],
        'renderer_provenance': 'quality-ranker-synthetic-generation-v1/legacy_opengl_centers',
    },
    {
        'sequence_id': 'ranker-multiview-hot3d27-v1-yaw0-pitchm25',
        'physical_object_id': 'hot3d:obj_000027',
        'split': 'validation',
        'input_sha256': 'e078c5827d4913c12e600bacb0bbd4e1733c3cc2f8ef08938a6c5b0922eec507',
        'video_path': 'synthetic.mp4',
        'video_sha256': '3152c0997b04a368dfc468ae0ea47bf094c568586932167d078f0ccda8cdb6a6',
        'mesh_path': 'object.glb',
        'mesh_sha256': 'b5354f0785c6a4b97dcadb3fe2622e9dc7e2c794bd69d648b72482dc7bdb8e0a',
        'generation_receipt_path': '../generation.json',
        'generation_receipt_sha256': 'd74bf01b9bbc794b0c1c2ae2f8a01dafd01d20dc9e0e191e0257062c91eecb5c',
        'native_resolution': [640, 480],
        'native_intrinsics': [[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]],
        'renderer_provenance': 'quality-ranker-synthetic-generation-v1/legacy_opengl_centers',
    },
)

# Exact non-model runtime closure for this interface. The two historical
# adapter replacements also appear here because they are executed at runtime.
EXPECTED_RUNTIME_FILES = frozenset({
    'bench/quality_gotrack_inference.py',
    'bench/quality_gotrack.py',
    'bench/quality_runner.py',
    'bench/quality_offline.py',
    'bench/glb_model.py',
    'bench/show3d_experiment.py',
    'bench/renderer.py',
    'bench/quality_capture.py',
    'bench/quality_camera.py',
    'bench/quality_contract.py',
    'bench/quality_assets.py',
    'bench/quality_time.py',
    'bench/quality_pipeline.py',
    'bench/quality_pose_ranker.py',
    'bench/quality_confidence_rejection.py',
    'bench/quality_confidence_evaluate.py',
    'bench/quality_ranker_data.py',
    'bench/quality_ranker_features.py',
    'bench/quality_ranker_signal.py',
    'bench/quality_ranker_initialization.py',
    'bench/quality_ranker_train.py',
    'bench/quality_ordering_research.py',
    'bench/quality_trace.py',
    'bench/quality_render_stability.py',
    'bench/quality_foundpose.py',
    'bench/quality_neighbors.py',
    'bench/quality_sam2.py',
    'bench/quality_cnos.py',
    'bench/quality_detection_association.py',
    'bench/quality_memory.py',
    'bench/quality_appearance.py',
    'bench/vision.py',
})
EXPECTED_REPLACEMENTS = frozenset({'bench/quality_gotrack.py', 'bench/quality_runner.py'})
EXPECTED_SOURCE18_MASK_PRODUCER = {
    'schema_version': 2,
    'clock_mode': 'capture_seconds_v1',
    'model': 'SAM 2.1 Hiera Base Plus',
    'model_config': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'propagation': 'forward only; one ordinal-0 initial selection; no query-frame prompts',
    'position_grid': 'exact arange form of upstream cumsum(ones)',
    'checkpoint_sha256': 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5',
    'provenance': {
        'adapter_sha256': {
            'quality_appearance': '84848bbb6e80f805b52e36fc2621ac417a5c20c84835526f18f5fabafbd357ff',
            'quality_assets': '74c47bcb761ea655d89ccecffb13d27501263da7f01b96a344bc9684dfdf030b',
            'quality_cnos': '5227062e2eaf504dec265f387c2d72edf7899f4f387230913c28112babb42255',
            'quality_contract': '9a96f6602daffea0248e038b737231eaf27ccf5dfbe5247ac3d883ab0c0aaa0e',
            'quality_detection_association': 'e29ebb0e51e98bd13cfaff960050124aab0434ef7e23878906d087117d25ffbd',
            'quality_foundpose': 'c07f9f254f131aa3fecb5ba86c77d21afab8be081ac3111e4edac62eb170fb86',
            'quality_gotrack': '76df6cb3c1368d1d723acd5e00cde83a62bbd153ef032e54e2845998ff6d09ae',
            'quality_memory': '87e808d579176f46ccb1e58d38c5e70711c62760014494b6b72f24a9c3565e32',
            'quality_neighbors': '3eb720f9ccd86d76b9c68d79b7b1d03d5c2ba21fe3e105098510397d52be5188',
            'quality_render_stability': '66f907f0c6c3f9ed32c6edbedb996a9c45e5ab1aea1d1914f99026a7f725c6bd',
            'quality_runner': '2266cadf050a884c36a38a3cc39ef591176a190e9dbbe830ac3141aae761b70c',
            'quality_sam2': '51df8cff618711aaa8cc57092c4c67419a50a8d36db93474e0ab7bc82207b612',
            'quality_time': '87e1013af4ff6e8625fa9eb7020834e66aad0c187a56774d8f0ae78e72f21e54',
            'quality_trace': 'd80314188862c4908bbb23a92b77416ac1af1c5cd6a1e7ae78ea2724364d5473',
            'vision': '76c7e91e6fe73278b35f96cd933801d8cca299aec41d51276bfe95c0167f77da',
        },
        'source_revisions': {
            'gotrack': UPSTREAM_REVISION,
            'sam2': '2b90b9f5ceec907a1c18123530e92e794ad901a4',
        },
        'submodule_revisions': {
            'bop_toolkit': 'fa9bc5c3a1c7fc092fc0694e69270d275278032e',
            'dinov2': '12592a9bdfea0a6e4ee4767cd5a8d3195c0e3509',
        },
        'checkpoint_sha256': {
            'FastSAM-x.pt': '752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76',
            'dinov2_vitl14_pretrain.pth': 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428',
            'gotrack_checkpoint.pt': BASE_CHECKPOINT_SHA256,
            'sam2.1_hiera_base_plus.pt': 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5',
        },
    },
}
EXPANDED_MASK_REUSE_AUDIT_SHA256 = 'eb0b5c34b03e893313af7e6d5a4af3442722574e8a817e727d6fbe4780ab8f45'
EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER = copy.deepcopy(EXPECTED_SOURCE18_MASK_PRODUCER)
EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER['provenance']['adapter_sha256']['quality_runner'] = (
    'caaac2ae404e54b90a7c85c40e0224fdacc3604241612e871fd7d867d1cd1e35')
EXPANDED_MASK_PRODUCER_COMPATIBILITY = {
    'schema': 'gotrack_expanded_mask_producer_compatibility_v1',
    'artifact_sha256': EXPANDED_EXPORT_SHA256,
    'reuse_audit_sha256': EXPANDED_MASK_REUSE_AUDIT_SHA256,
    'producer': EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER,
}


class InferenceContractError(ValueError):
    """The requested GoTrack inference profile is not authenticated/supported."""


@dataclass(frozen=True, slots=True)
class GoTrackInferenceOptions:
    head_path: Path | None = None
    head_sha256: str | None = None
    config_path: Path | None = None
    config_sha256: str | None = None
    compatibility_config_path: Path | None = None
    compatibility_config_sha256: str | None = None
    cloud_execution: bool = False

    @property
    def configured(self) -> bool:
        return (self.head_path is not None or self.head_sha256 is not None or
                self.config_path is not None or self.config_sha256 is not None or
                self.compatibility_config_path is not None or
                self.compatibility_config_sha256 is not None)

    def validate(self) -> 'GoTrackInferenceOptions':
        if type(self.cloud_execution) is not bool:
            raise InferenceContractError('cloud_execution must be a bool')
        if not self.configured:
            if self.cloud_execution:
                raise InferenceContractError('--cloud-execution requires a GoTrack inference contract')
            return self
        if self.config_path is None or self.config_sha256 is None:
            raise InferenceContractError('Inference profile requires both --gotrack-inference-config and its SHA-256')
        if ((self.compatibility_config_path is None) !=
                (self.compatibility_config_sha256 is None)):
            raise InferenceContractError(
                'Compatibility contract path and SHA-256 must be supplied together')
        if (self.head_path is None) != (self.head_sha256 is None):
            raise InferenceContractError('Adapted head path and SHA-256 must be supplied together')
        if self.head_path is not None and self.head_sha256 is None:
            raise InferenceContractError('Adapted head path requires its SHA-256')
        _require_sha256(self.config_sha256, 'config_sha256')
        if self.compatibility_config_sha256 is not None:
            _require_sha256(self.compatibility_config_sha256,
                            'compatibility_config_sha256')
        if self.head_sha256 is not None:
            _require_sha256(self.head_sha256, 'head_sha256')
        return self


@dataclass(frozen=True, slots=True)
class GoTrackInferenceContext:
    profile: str
    geometry: str
    contract_path: Path
    contract_sha256: str
    contract: Any
    artifact: Any
    runtime_contract_sha256: str
    requested_contract_sha256: str
    bundle: Path
    accepted_input: Any
    original_intrinsics: np.ndarray
    effective_intrinsics: np.ndarray
    head_path: Path | None
    requested_head_sha256: str | None
    installed_head_sha256: str | None
    base_checkpoint_sha256: str
    runtime_packages: Any
    provenance: Any

    def identity_dict(self, *, installed: bool = False) -> dict:
        value = _thaw(self.provenance)
        value['runtime_contract_sha256'] = self.runtime_contract_sha256
        value['requested_contract_sha256'] = self.requested_contract_sha256
        if self.requested_head_sha256 is not None:
            value['head']['installed_sha256'] = self.requested_head_sha256 if installed else None
            value['head_installation'] = {'requested': True, 'installed': bool(installed)}
        else:
            value['head_installation'] = {'requested': False, 'installed': True}
        return value

    def frame_identity(self, *, mask_path=None, mask_sha256=None) -> dict:
        value = self.identity_dict()
        value['mask_cache_frame'] = {'path': mask_path, 'sha256': mask_sha256}
        return value


def options_from_values(*, head_path=None, head_sha256=None, config_path=None,
                        config_sha256=None, compatibility_config_path=None,
                        compatibility_config_sha256=None, cloud_execution=False) -> GoTrackInferenceOptions:
    return GoTrackInferenceOptions(
        None if head_path is None else Path(head_path), head_sha256,
        None if config_path is None else Path(config_path), config_sha256,
        None if compatibility_config_path is None else Path(compatibility_config_path),
        compatibility_config_sha256,
        cloud_execution).validate()


def prepare_inference(options: GoTrackInferenceOptions, bundle, manifest, *,
                      runtime_packages: dict | None = None) -> GoTrackInferenceContext | None:
    """Authenticate opt-in contract and one exact synthetic bundle before NN import.

    The omitted/default path returns immediately and does no new verification.
    """
    options.validate()
    if not options.configured:
        return None
    if not options.cloud_execution or os.environ.get('VISUALIZEIT_CLOUD') != '1' or platform.system() != 'Linux':
        raise InferenceContractError(
            'Configured GoTrack inference requires --cloud-execution, VISUALIZEIT_CLOUD=1 and Linux')
    if manifest.get('_capture') is not None:
        raise InferenceContractError('The adapted inference profile does not support capture bundles')
    if manifest.get('_timing', {}).get('clock_mode') != 'capture_seconds_v1':
        raise InferenceContractError('The adapted inference profile only accepts its pinned synthetic physical-time bundles')

    requested_path = _checked_regular_file(options.config_path, 'inference contract')
    requested_bytes = _read_limited(requested_path, 16 * 1024 * 1024, 'inference contract')
    requested_sha = _sha256_bytes(requested_bytes)
    if requested_sha != options.config_sha256.lower():
        raise InferenceContractError('Inference contract SHA-256 mismatch')
    requested_contract = _loads_unique(requested_bytes, 'inference contract')
    _validate_contract(requested_contract)
    requested_artifact = _artifact_for_contract(requested_contract)

    runtime_path = requested_path
    runtime_bytes = requested_bytes
    runtime_sha = requested_sha
    contract = requested_contract
    artifact = requested_artifact
    runtime_artifact = requested_artifact
    if options.compatibility_config_path is not None:
        if (requested_sha != LEGACY_CONTRACT_SHA256 or
                requested_artifact['export_sha256'] != EXPORT_SHA256):
            raise InferenceContractError(
                'Compatibility mapping is allowed only for the exact legacy request contract')
        runtime_path = _checked_regular_file(options.compatibility_config_path,
                                             'runtime compatibility contract')
        runtime_bytes = _read_limited(runtime_path, 16 * 1024 * 1024,
                                      'runtime compatibility contract')
        runtime_sha = _sha256_bytes(runtime_bytes)
        if runtime_sha != options.compatibility_config_sha256.lower():
            raise InferenceContractError('Runtime compatibility contract SHA-256 mismatch')
        contract = _loads_unique(runtime_bytes, 'runtime compatibility contract')
        _validate_contract(contract)
        runtime_artifact = _artifact_for_contract(contract)
        _validate_legacy_runtime_pair(requested_contract, requested_artifact,
                                      contract, runtime_artifact, requested_sha)
    elif requested_sha == LEGACY_CONTRACT_SHA256:
        # The archived legacy contract authenticates its original source closure.
        # After wrapper edits it must fail closed unless the caller provides the
        # separately pinned compatible runtime contract above.
        pass
    source_identity = verify_source_compatibility(contract)

    expected_packages = dict(contract['runtime']['packages'])
    actual_packages = _runtime_packages() if runtime_packages is None else dict(runtime_packages)
    if actual_packages != expected_packages:
        raise InferenceContractError('Inference runtime package versions differ from the reviewed contract')
    if sys.version_info[:2] not in EXPECTED_PYTHON:
        raise InferenceContractError('Configured inference requires the reviewed Python 3.10 or 3.11 runtime')

    bundle_path = Path(bundle).resolve(strict=True)
    accepted = _validate_accepted_input(bundle_path, manifest, contract['accepted_inputs'])
    original = _validated_native_intrinsics(manifest, accepted)
    effective = np.array(original, dtype=np.float64, copy=True)
    effective[0, 2] += NATIVE_CORRECTION[0]
    effective[1, 2] += NATIVE_CORRECTION[1]
    effective.setflags(write=False)
    original.setflags(write=False)

    head_path = None
    requested_head = options.head_sha256.lower() if options.head_sha256 is not None else None
    if options.head_path is not None:
        head_path = _checked_regular_file(options.head_path, 'GoTrack pose head export')
        if head_path.stat().st_size != artifact['export_bytes']:
            raise InferenceContractError('GoTrack pose head export byte size differs from the approved artifact')
        if requested_head != artifact['export_sha256']:
            raise InferenceContractError('Requested head SHA-256 differs from the approved export identity')
        if _sha256_file(head_path) != requested_head:
            raise InferenceContractError('GoTrack pose head export SHA-256 differs from the approved artifact')
    elif requested_head is not None:
        raise InferenceContractError('A head digest without a head path is unsupported')

    provenance = {
        'schema': CONTRACT_SCHEMA,
        'profile': PROFILE,
        'geometry': contract['geometry'],
        'head': {
            'requested_sha256': requested_head,
            'installed_sha256': None,
            'selection': 'frozen_base' if requested_head is None else 'adapted_pose_head',
            'artifact_id': artifact['artifact_id'],
            **{key: artifact['training'][key] for key in (
                'base_checkpoint_sha256', 'training_lock_sha256',
                'feature_lock_sha256', 'feature_recipe_sha256',
                'feature_producer_sha256', 'prelabel_seal_sha256',
                'dataset_sha256', 'training_config_sha256')},
        },
        'config_sha256': requested_sha,
        'runtime_contract_sha256': runtime_sha,
        'runtime_artifact_id': runtime_artifact['artifact_id'],
        'runtime': {
            'packages': expected_packages,
            'python': '.'.join(map(str, sys.version_info[:3])),
            'source_closure_sha256': source_identity['source_closure_sha256'],
            'resolved_model_cache_root': source_identity['resolved_model_cache_root'],
        },
        'input': {
            'input_sha256': accepted['input_sha256'],
            'video_sha256': accepted['video_sha256'],
            'mesh_sha256': accepted['mesh_sha256'],
            'generation_receipt_sha256': accepted['generation_receipt_sha256'],
            'sequence_id': accepted['sequence_id'],
            'physical_object_id': accepted['physical_object_id'],
            'split': accepted['split'],
            'clock_mode': 'capture_seconds_v1',
        },
        'calibration': {
            'native_intrinsics_original': original.tolist(),
            'native_intrinsics_effective': effective.tolist(),
            'native_principal_correction_px': list(NATIVE_CORRECTION),
            'raster_convention': RASTER_CONVENTION,
            'native_resolution': list(accepted['native_resolution']),
        },
        'head_installation': {'requested': requested_head is not None, 'installed': False},
    }
    return GoTrackInferenceContext(
        PROFILE, PROFILE, requested_path, requested_sha, _freeze(contract),
        _freeze(dict(artifact)), runtime_sha, requested_sha, bundle_path,
        _freeze(accepted), original, effective, head_path, requested_head,
        None, BASE_CHECKPOINT_SHA256,
        _freeze(expected_packages), _freeze(provenance))


def verify_source_compatibility(contract: dict) -> dict:
    """Verify the contract's exact historical source closure and adapter bytes."""
    lock_path = ROOT / TRAINING_LOCK_RELATIVE_PATH
    if _sha256_file(lock_path) != TRAINING_LOCK_SHA256:
        raise InferenceContractError('Historical GoTrack training source lock changed')
    lock = _loads_unique(_read_limited(lock_path, 2 * 1024 * 1024, 'historical source lock'),
                         'historical source lock')
    if not isinstance(lock, dict) or not isinstance(lock.get('gotrack'), dict):
        raise InferenceContractError('Historical GoTrack source lock is malformed')
    historical = lock['gotrack'].get('files')
    if not isinstance(historical, dict) or lock['gotrack'].get('revision') != UPSTREAM_REVISION:
        raise InferenceContractError('Historical GoTrack source map/revision is malformed')

    replacements = contract['reviewed_adapter_replacements']
    unchanged = contract['unchanged_training_files']
    if set(replacements) != EXPECTED_REPLACEMENTS:
        raise InferenceContractError('Reviewed adapter replacement set differs from the exact approved paths')
    expected_unchanged = {path: digest for path, digest in historical.items()
                          if path not in EXPECTED_REPLACEMENTS}
    if unchanged != expected_unchanged:
        raise InferenceContractError('Inference contract does not retain the complete historical training file map')
    cache_mount = contract.get('source_cache_mount')
    if cache_mount != APPROVED_CACHE_MOUNT:
        raise InferenceContractError('Inference contract declares an unexpected model-cache mount')
    model_cache = ROOT / '.cache/model-quality'
    try:
        resolved_cache = model_cache.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InferenceContractError('Pinned GoTrack model cache is unavailable') from exc
    if resolved_cache.as_posix() != APPROVED_CACHE_MOUNT:
        raise InferenceContractError('Resolved GoTrack cache differs from the approved shared-cache root')
    for relative, expected in unchanged.items():
        _verify_relative_file(relative, expected, 'historical training source',
                              resolved_cache=resolved_cache)
    for relative, row in replacements.items():
        if not isinstance(row, dict) or set(row) != {'training_sha256', 'inference_sha256', 'reason'}:
            raise InferenceContractError(f'Invalid reviewed replacement record: {relative}')
        if row['training_sha256'] != historical.get(relative):
            raise InferenceContractError(f'Historical replacement pin differs from the training lock: {relative}')
        if not isinstance(row['reason'], str) or not row['reason']:
            raise InferenceContractError(f'Reviewed replacement requires a reason: {relative}')
        _verify_relative_file(relative, row['inference_sha256'], 'reviewed adapter replacement',
                              resolved_cache=resolved_cache)

    runtime_files = contract['additional_runtime_files']
    if set(runtime_files) != EXPECTED_RUNTIME_FILES:
        raise InferenceContractError('Inference contract runtime closure has missing or unexpected paths')
    for relative, expected in runtime_files.items():
        _verify_relative_file(relative, expected, 'additional inference runtime source',
                              resolved_cache=resolved_cache)
    closure = _source_closure_hash(unchanged, replacements, runtime_files)
    if contract['source_closure_sha256'] != closure:
        raise InferenceContractError('Inference source closure digest mismatch')
    return {'source_closure_sha256': closure,
            'historical_file_count': len(unchanged) + len(replacements),
            'additional_runtime_file_count': len(runtime_files),
            'resolved_model_cache_root': resolved_cache.as_posix()}


def install_adapted_head(network, context: GoTrackInferenceContext, *, device='cpu', torch_module=None):
    """Verify then atomically replace only ``network.pose_head`` with the export."""
    if context.head_path is None or context.requested_head_sha256 is None:
        raise InferenceContractError('No adapted head was requested for this inference context')
    if torch_module is None:
        import torch as torch_module
    path = _checked_regular_file(context.head_path, 'GoTrack pose head export')
    artifact = context.artifact
    expected_bytes = artifact['export_bytes']
    if context.requested_head_sha256 != artifact['export_sha256']:
        raise InferenceContractError('Requested head differs from the selected immutable artifact record')
    data = _read_limited(path, expected_bytes, 'GoTrack pose head export')
    if len(data) != expected_bytes or _sha256_bytes(data) != context.requested_head_sha256:
        raise InferenceContractError('GoTrack pose head export digest or size mismatch')

    # Deserialization uses the exact hashed byte string, avoiding a path swap
    # between hash verification and torch.load. No unsafe pickle fallback exists.
    cpu_rng = torch_module.random.get_rng_state().clone()
    cuda_rng = None
    rng_restored = False
    try:
        cuda = getattr(torch_module, 'cuda', None)
        if cuda is not None and cuda.is_initialized():
            cuda_rng = [state.clone() for state in cuda.get_rng_state_all()]
        exported = torch_module.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        _validate_export_header(exported, artifact)
        live_head = getattr(network, 'pose_head', None)
        if live_head is None:
            raise InferenceContractError('Frozen GoTrack network has no pose_head')
        expected_state = live_head.state_dict()
        exported_state = exported.get('head')
        _validate_state_dict(exported_state, expected_state, torch_module)
        candidate = copy.deepcopy(live_head)
        candidate.load_state_dict(exported_state, strict=True)
        candidate.to(device)
        candidate.eval()
        candidate.requires_grad_(False)
        torch_module.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch_module.cuda.set_rng_state_all(cuda_rng)
        rng_restored = True
        # This is the sole mutation of the live network and happens only after
        # complete validation, isolated strict loading, and device preparation.
        network.pose_head = candidate
        provenance = context.identity_dict(installed=True)
        provenance['head_installation'] = {'requested': True, 'installed': True}
        return network, provenance
    finally:
        if not rng_restored:
            torch_module.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch_module.cuda.set_rng_state_all(cuda_rng)


def bind_runtime_mask_provenance(context: GoTrackInferenceContext, mask_manifest_sha256: str,
                                 mask_records: dict, *, profile_identity: dict | None = None) -> dict:
    """Return a detached run identity with exact joined mask-cache references."""
    _require_sha256(mask_manifest_sha256, 'mask_manifest_sha256')
    value = copy.deepcopy(profile_identity) if profile_identity is not None else context.identity_dict()
    value['mask_cache'] = {
        'manifest_sha256': mask_manifest_sha256.lower(),
        'records': {str(key): {'path': row.get('path'), 'sha256': row.get('mask_sha256')}
                    for key, row in sorted(mask_records.items())},
    }
    return value


def validate_physical_mask_producer(context: GoTrackInferenceContext, mask_manifest: dict) -> None:
    """Authenticate an artifact-authorized schema-2 SAM producer closure.

    The archived producer is the default for both immutable artifacts. The
    expanded artifact alone may also use the exact audited ROOT3 producer,
    whose only source difference is the runner adapter. The ordinary physical
    mask validator still owns timing, coverage, paths and mask bytes.
    """
    if context.profile != PROFILE:
        raise InferenceContractError('Physical mask producer pin applies only to the reviewed inference profile')
    if context.contract['mask_producer'] != EXPECTED_SOURCE18_MASK_PRODUCER:
        raise InferenceContractError('Inference contract mask producer differs from the reviewed source18 closure')

    artifact_sha = context.artifact.get('export_sha256')
    expected_producers = [EXPECTED_SOURCE18_MASK_PRODUCER]
    if artifact_sha == EXPANDED_EXPORT_SHA256:
        if context.contract.get('expanded_artifact_mask_producer_compatibility') != (
                EXPANDED_MASK_PRODUCER_COMPATIBILITY):
            raise InferenceContractError('Expanded artifact does not authenticate the audited ROOT3 producer')
        expected_producers.append(EXPECTED_EXPANDED_ROOT3_MASK_PRODUCER)

    selected_producer = None
    for expected_producer in expected_producers:
        matches = True
        for key, expected in expected_producer.items():
            if key == 'provenance':
                continue
            actual = mask_manifest.get(key)
            if key == 'checkpoint_sha256' and isinstance(actual, str):
                actual = actual.lower()
            if actual != expected:
                matches = False
                break
        actual_provenance = mask_manifest.get('provenance')
        if matches and isinstance(actual_provenance, dict):
            for field, expected in expected_producer['provenance'].items():
                actual = actual_provenance.get(field)
                if field in ('adapter_sha256', 'checkpoint_sha256') and isinstance(actual, dict):
                    actual = {key: str(value).lower() for key, value in actual.items()}
                if actual != expected:
                    matches = False
                    break
        elif matches:
            matches = False
        if matches:
            selected_producer = expected_producer
            break
    if selected_producer is None:
        raise InferenceContractError('Physical mask cache has an unexpected producer field or artifact pairing')

    accepted = context.accepted_input
    if str(mask_manifest.get('input_manifest_sha256', '')).lower() != accepted['input_sha256']:
        raise InferenceContractError('Physical mask cache input hash differs from the pinned synthetic bundle')
    expected_source_hashes = {
        accepted['mesh_path']: accepted['mesh_sha256'],
        accepted['video_path']: accepted['video_sha256'],
    }
    actual_source_hashes = mask_manifest.get('source_sha256')
    if (not isinstance(actual_source_hashes, dict) or set(actual_source_hashes) != set(expected_source_hashes) or
            {key: str(value).lower() for key, value in actual_source_hashes.items()} != expected_source_hashes):
        raise InferenceContractError('Physical mask producer source hashes do not match the pinned synthetic bundle')

    actual_provenance = mask_manifest.get('provenance')
    if not isinstance(actual_provenance, dict) or set(actual_provenance) != {
            'input_manifest_sha256', 'adapter_sha256', 'source_revisions',
            'submodule_revisions', 'checkpoint_sha256'}:
        raise InferenceContractError('Physical mask producer provenance schema is missing or unexpected')
    if str(actual_provenance['input_manifest_sha256']).lower() != accepted['input_sha256']:
        raise InferenceContractError('Physical mask producer is bound to a different input manifest')
    for field, expected in selected_producer['provenance'].items():
        actual = actual_provenance.get(field)
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise InferenceContractError(f'Physical mask producer {field} closure is missing or unexpected')
        if field in ('adapter_sha256', 'checkpoint_sha256'):
            actual = {key: str(value).lower() for key, value in actual.items()}
        if actual != expected:
            raise InferenceContractError(f'Physical mask producer {field} differs from the selected reviewed closure')


def _validate_contract(value: Any) -> None:
    required = {
        'schema', 'profile', 'training', 'unchanged_training_files',
        'reviewed_adapter_replacements', 'additional_runtime_files',
        'source_closure_sha256', 'source_cache_mount', 'geometry', 'accepted_inputs',
        'mask_producer', 'runtime',
    }
    if not isinstance(value, dict):
        raise InferenceContractError('inference contract must be an object')
    training = value.get('training')
    export_sha = training.get('export_sha256') if isinstance(training, dict) else None
    artifact = ARTIFACT_REGISTRY.get(export_sha)
    if artifact is None:
        raise InferenceContractError('Unknown or mixed GoTrack artifact identity')
    expected_keys = set(required)
    if artifact['export_sha256'] == EXPANDED_EXPORT_SHA256:
        expected_keys.update({
            'legacy_runtime_compatibility',
            'expanded_artifact_mask_producer_compatibility',
        })
    _exact_keys(value, expected_keys, 'inference contract')
    if value['schema'] != CONTRACT_SCHEMA or value['profile'] != PROFILE:
        raise InferenceContractError('Unknown GoTrack inference schema/profile')
    if value['source_cache_mount'] != APPROVED_CACHE_MOUNT:
        raise InferenceContractError('Inference contract cache mount differs from the reviewed cloud root')
    _exact_keys(training, _TRAINING_FIELDS, 'training identity')
    if training != dict(artifact['training']):
        raise InferenceContractError('Training/export identity differs from the immutable artifact registry')
    if artifact['export_sha256'] == EXPANDED_EXPORT_SHA256:
        if value['legacy_runtime_compatibility'] != dict(LEGACY_RUNTIME_COMPATIBILITY):
            raise InferenceContractError('Expanded contract has an unknown legacy runtime compatibility rule')
        if value['expanded_artifact_mask_producer_compatibility'] != (
                EXPANDED_MASK_PRODUCER_COMPATIBILITY):
            raise InferenceContractError('Expanded contract has an unknown ROOT3 mask producer compatibility rule')

    _validate_geometry(value['geometry'])
    _validate_accepted_input_contract(value['accepted_inputs'])
    if value['mask_producer'] != EXPECTED_SOURCE18_MASK_PRODUCER:
        raise InferenceContractError('Mask producer contract differs from the reviewed source18 schema-2 closure')
    runtime = value['runtime']
    _exact_keys(runtime, {'python', 'packages'}, 'runtime contract')
    if runtime['python'] != ['3.10', '3.11'] or runtime['packages'] != dict(EXPECTED_PACKAGES):
        raise InferenceContractError('Inference package or Python runtime pins differ from the reviewed trial')
    for name in ('source_closure_sha256',):
        _require_sha256(value[name], name)
    for field in ('unchanged_training_files', 'additional_runtime_files'):
        mapping = value[field]
        if not isinstance(mapping, dict) or not mapping:
            raise InferenceContractError(f'{field} must be a nonempty path-to-SHA256 mapping')
        for path, digest in mapping.items():
            _require_relative_path(path, field)
            _require_sha256(digest, f'{field}.{path}')
    replacements = value['reviewed_adapter_replacements']
    if not isinstance(replacements, dict):
        raise InferenceContractError('reviewed_adapter_replacements must be an object')
    for path, record in replacements.items():
        _require_relative_path(path, 'reviewed adapter path')
        if not isinstance(record, dict) or set(record) != {'training_sha256', 'inference_sha256', 'reason'}:
            raise InferenceContractError(f'Invalid reviewed adapter replacement record: {path}')
        _require_sha256(record['training_sha256'], f'{path}.training_sha256')
        _require_sha256(record['inference_sha256'], f'{path}.inference_sha256')


def _artifact_for_contract(contract: dict) -> MappingProxyType:
    training = contract.get('training') if isinstance(contract, dict) else None
    digest = training.get('export_sha256') if isinstance(training, dict) else None
    artifact = ARTIFACT_REGISTRY.get(digest)
    if artifact is None or training != dict(artifact['training']):
        raise InferenceContractError('Contract does not select one exact immutable GoTrack artifact')
    return artifact


def _validate_legacy_runtime_pair(requested: dict, requested_artifact: dict,
                                  runtime: dict, runtime_artifact: dict,
                                  requested_sha: str) -> None:
    if requested_artifact['export_sha256'] != EXPORT_SHA256:
        raise InferenceContractError('Compatibility request must use the legacy smoke artifact')
    if runtime_artifact['export_sha256'] != EXPANDED_EXPORT_SHA256:
        raise InferenceContractError('Compatibility runtime must use the expanded continuation artifact')
    if runtime.get('legacy_runtime_compatibility') != dict(LEGACY_RUNTIME_COMPATIBILITY):
        raise InferenceContractError('Expanded runtime does not authenticate the requested legacy contract')
    if requested_sha != LEGACY_CONTRACT_SHA256:
        raise InferenceContractError('Compatibility mapping requires the exact archived legacy contract SHA-256')
    for field in ('geometry', 'accepted_inputs', 'mask_producer', 'runtime',
                  'source_cache_mount', 'unchanged_training_files'):
        if requested.get(field) != runtime.get(field):
            raise InferenceContractError(
                f'Legacy request and expanded runtime differ in {field}')
    old_replacements = requested.get('reviewed_adapter_replacements', {})
    new_replacements = runtime.get('reviewed_adapter_replacements', {})
    if set(old_replacements) != set(new_replacements):
        raise InferenceContractError('Legacy request and expanded runtime replacement sets differ')
    for path in old_replacements:
        old, new = old_replacements[path], new_replacements[path]
        if old.get('training_sha256') != new.get('training_sha256') or old.get('reason') != new.get('reason'):
            raise InferenceContractError('Legacy request and expanded runtime adapter provenance differ')
        if path not in _RUNTIME_COMPATIBILITY_WRAPPERS and old.get('inference_sha256') != new.get('inference_sha256'):
            raise InferenceContractError('Legacy request and expanded runtime numerical adapter bytes differ')
    old_runtime = requested.get('additional_runtime_files', {})
    new_runtime = runtime.get('additional_runtime_files', {})
    old_numerical = {path: digest for path, digest in old_runtime.items()
                     if path not in _RUNTIME_COMPATIBILITY_WRAPPERS}
    new_numerical = {path: digest for path, digest in new_runtime.items()
                     if path not in _RUNTIME_COMPATIBILITY_WRAPPERS}
    if old_numerical != new_numerical:
        raise InferenceContractError('Legacy request and expanded runtime numerical source bytes differ')


def _validate_geometry(geometry: Any) -> None:
    _exact_keys(geometry, {
        'raster_convention', 'native_correction_px', 'renderer_coordinate_mode',
        'renderer_render_policy', 'unlit_templates', 'disable_multisampling',
        'crop_recipe', 'crop_size', 'depth_grid', 'flow_direction',
        'depth_units', 'pose_units', 'solver_iterations', 'confidence_threshold',
        'observed_mask_policy', 'pnp_settings',
    }, 'geometry policy')
    expected = {
        'raster_convention': RASTER_CONVENTION,
        'native_correction_px': [-0.5, -0.5],
        'renderer_coordinate_mode': COORDINATE_MODE,
        'renderer_render_policy': RENDER_POLICY,
        'unlit_templates': True,
        'disable_multisampling': True,
        'crop_recipe': 'ordinary_upstream_float_rgb_template_crop_280_v1',
        'crop_size': [280, 280],
        'depth_grid': 'integer_pixel_indices_xy_v1',
        'flow_direction': 'template_reference_to_query_crop_pixels',
        'depth_units': 'millimetres_crop_camera',
        'pose_units': 'native_camera_from_object_metres;PnP_crop_camera_from_object_millimetres',
        'solver_iterations': 5,
        'confidence_threshold': 0.3,
        'observed_mask_policy': 'template_rendered_mask_and_observed_crop_mask_source_and_floor_target_gt_0_5',
        'pnp_settings': {
            'subset_max': 10000, 'minimum_correspondences': 24,
            'ransac_seed': 'frame_id_times_17_plus_iteration', 'minimum_inliers': 6,
            'ransac_threshold_px': 2.0, 'ransac_iterations': 3000,
            'ransac_confidence': 0.999, 'ransac_flags': 'SOLVEPNP_ITERATIVE',
            'use_extrinsic_guess': True, 'refinement': 'solvePnPRefineLM',
        },
    }
    if geometry != expected:
        raise InferenceContractError('Geometry policy differs from the reviewed integer-center profile')


def _validate_accepted_input_contract(inputs: Any) -> None:
    if not isinstance(inputs, list) or len(inputs) != len(EXPECTED_ACCEPTED_INPUTS):
        raise InferenceContractError('The profile requires exactly three pinned synthetic trial bundles')
    if inputs != [dict(row) for row in EXPECTED_ACCEPTED_INPUTS]:
        raise InferenceContractError('Accepted synthetic input bindings differ from the authenticated three-bundle trial')
    seen = set()
    for row in inputs:
        _exact_keys(row, {
            'sequence_id', 'physical_object_id', 'split', 'input_sha256',
            'video_path', 'video_sha256', 'mesh_path', 'mesh_sha256',
            'generation_receipt_path', 'generation_receipt_sha256',
            'native_resolution', 'native_intrinsics', 'renderer_provenance',
        }, 'accepted input')
        if row['sequence_id'] in seen:
            raise InferenceContractError('Accepted input sequence IDs must be unique')
        seen.add(row['sequence_id'])
        if row['split'] not in ('train', 'validation'):
            raise InferenceContractError('Accepted synthetic input split is unsupported')
        if row['renderer_provenance'] != 'quality-ranker-synthetic-generation-v1/legacy_opengl_centers':
            raise InferenceContractError('Accepted input lacks the reviewed historical raster provenance')
        if row['native_resolution'] != [640, 480] or row['native_intrinsics'] != [
                [576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]]:
            raise InferenceContractError('Accepted input native camera does not match the reviewed synthetic generator')
        for key in ('input_sha256', 'video_sha256', 'mesh_sha256', 'generation_receipt_sha256'):
            _require_sha256(row[key], f'accepted input {key}')
        for key in ('video_path', 'mesh_path', 'generation_receipt_path'):
            if not isinstance(row[key], str) or not row[key] or Path(row[key]).is_absolute():
                raise InferenceContractError(f'Accepted input {key} must be a fixed relative path')
            if '..' in Path(row[key]).parts and key != 'generation_receipt_path':
                raise InferenceContractError(f'Accepted input {key} cannot escape its bundle')


def _validate_accepted_input(bundle: Path, manifest: dict, accepted_rows: list) -> dict:
    if manifest.get('schema_version') != 2 or manifest.get('provenance_kind') != 'synthetic':
        raise InferenceContractError('Configured inference rejects non-synthetic/unknown manifest provenance')
    if manifest.get('units') != 'metres' or manifest.get('asset') != 'object.glb' or manifest.get('video') != 'synthetic.mp4':
        raise InferenceContractError('Synthetic inference manifest asset/video schema is unsupported')
    sequence_id = manifest.get('sequence_id')
    match = next((row for row in accepted_rows if row['sequence_id'] == sequence_id), None)
    if match is None:
        raise InferenceContractError('Inference bundle sequence is not in the authenticated synthetic allowlist')
    if manifest.get('physical_object_id') != match['physical_object_id']:
        raise InferenceContractError('Synthetic bundle object identity differs from its approved binding')
    if manifest.get('native_resolution') != match['native_resolution']:
        raise InferenceContractError('Synthetic bundle raster dimensions differ from their approved binding')
    if manifest.get('source_hashes') != {
            match['mesh_path']: match['mesh_sha256'], match['video_path']: match['video_sha256']}:
        raise InferenceContractError('Synthetic source hash map differs from its approved video/mesh binding')

    input_path = _checked_regular_file(bundle / 'input.json', 'inference input manifest')
    input_sha = _sha256_file(input_path)
    if input_sha != match['input_sha256']:
        raise InferenceContractError('Synthetic input manifest SHA-256 differs from its approved binding')
    video_path = _checked_regular_file(bundle / match['video_path'], 'synthetic input video')
    mesh_path = _checked_regular_file(bundle / match['mesh_path'], 'synthetic input mesh')
    if _sha256_file(video_path) != match['video_sha256'] or _sha256_file(mesh_path) != match['mesh_sha256']:
        raise InferenceContractError('Synthetic input video or mesh differs from its approved binding')
    generation_path = _checked_relative_resource(bundle, match['generation_receipt_path'],
                                                 'generation receipt')
    generation_bytes = _read_limited(generation_path, 256 * 1024, 'generation receipt')
    if _sha256_bytes(generation_bytes) != match['generation_receipt_sha256']:
        raise InferenceContractError('Synthetic generation receipt SHA-256 differs from its approved binding')
    generation = _loads_unique(generation_bytes, 'generation receipt')
    if (generation.get('schema_version') != 1 or
            generation.get('kind') != 'quality-ranker-synthetic-generation-v1' or
            generation.get('sequence_id') != match['sequence_id'] or
            generation.get('physical_object_id') != match['physical_object_id'] or
            generation.get('split') != match['split'] or
            generation.get('video_sha256') != match['video_sha256'] or
            generation.get('mesh_sha256') != match['mesh_sha256'] or
            generation.get('inference_bundle_files') != ['input.json', 'object.glb', 'synthetic.mp4']):
        raise InferenceContractError('Synthetic generation metadata does not join the approved input bytes')
    # Deliberately never use generation label paths/hashes: poses.jsonl is not read.
    return {
        'sequence_id': match['sequence_id'], 'physical_object_id': match['physical_object_id'],
        'split': match['split'], 'input_sha256': input_sha,
        'video_path': match['video_path'], 'video_sha256': match['video_sha256'],
        'mesh_path': match['mesh_path'], 'mesh_sha256': match['mesh_sha256'],
        'generation_receipt_path': match['generation_receipt_path'],
        'generation_receipt_sha256': match['generation_receipt_sha256'],
        'native_resolution': tuple(match['native_resolution']),
    }


def _validated_native_intrinsics(manifest: dict, accepted: dict) -> np.ndarray:
    matrix = np.asarray(manifest.get('intrinsics'), dtype=np.float64)
    expected = np.asarray([[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]])
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
            matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or
            not np.allclose(matrix[0, 1], 0.0, rtol=0.0, atol=1e-12) or
            not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12) or
            not np.array_equal(matrix, expected)):
        raise InferenceContractError('Native camera matrix must be the exact finite zero-skew approved synthetic K')
    return matrix


def _validate_export_header(value: Any, artifact: MappingProxyType | dict) -> None:
    required = {'schema', 'head', 'source_lock_sha256', 'base_checkpoint_sha256',
                'dataset_sha256', 'config_sha256', 'validation', 'quality_claim', 'deployment'}
    _exact_keys(value, required, 'GoTrack head export')
    training = artifact['training']
    expected = {
        'schema': EXPORT_SCHEMA,
        'source_lock_sha256': training['training_lock_sha256'],
        'base_checkpoint_sha256': training['base_checkpoint_sha256'],
        'dataset_sha256': training['dataset_sha256'],
        'config_sha256': training['training_config_sha256'],
        'quality_claim': False,
        'deployment': training['export_deployment'],
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise InferenceContractError('GoTrack head export header does not match the authenticated research artifact')
    if type(value.get('quality_claim')) is not bool:
        raise InferenceContractError('GoTrack head export quality_claim must be a JSON boolean')
    if not isinstance(value.get('validation'), dict):
        raise InferenceContractError('GoTrack head export validation diagnostics are missing')


def _validate_state_dict(candidate: Any, expected: dict, torch_module) -> None:
    if not isinstance(candidate, dict) or set(candidate) != set(expected):
        raise InferenceContractError('GoTrack head state keys differ from the complete frozen pose_head')
    for key, reference in expected.items():
        value = candidate[key]
        if not torch_module.is_tensor(value) or not torch_module.is_tensor(reference):
            raise InferenceContractError(f'GoTrack head state entry is not a tensor: {key}')
        if value.layout != torch_module.strided or tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.dtype:
            raise InferenceContractError(f'GoTrack head tensor shape/dtype/layout mismatch: {key}')
        if value.is_floating_point() and not bool(torch_module.isfinite(value).all().item()):
            raise InferenceContractError(f'GoTrack head tensor contains nonfinite values: {key}')
        if value.is_complex():
            raise InferenceContractError(f'GoTrack head tensor has unsupported complex dtype: {key}')


def _source_closure_hash(unchanged: dict, replacements: dict, runtime_files: dict) -> str:
    payload = {'unchanged_training_files': unchanged,
               'reviewed_adapter_replacements': replacements,
               'additional_runtime_files': runtime_files}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return _sha256_bytes(encoded)


def _runtime_packages() -> dict:
    output = {}
    for name in EXPECTED_PACKAGES:
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InferenceContractError(f'Missing inference runtime package {name}') from exc
        output[name] = value.split('+', 1)[0]
    return output


def _verify_relative_file(relative: str, expected: str, name: str, *, resolved_cache: Path) -> None:
    _require_relative_path(relative, name)
    path = ROOT / Path(relative)
    if path.is_symlink():
        raise InferenceContractError(f'{name} leaf cannot be a symlink: {relative}')
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InferenceContractError(f'Missing {name} file: {relative}') from exc
    if relative.startswith('.cache/model-quality/'):
        suffix = Path(relative).relative_to('.cache/model-quality')
        cursor = resolved_cache
        for part in suffix.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise InferenceContractError(f'{name} has a symlink below the approved cache root: {relative}')
        if not resolved.is_relative_to(resolved_cache):
            raise InferenceContractError(f'{name} path escapes the approved cache root: {relative}')
    elif not resolved.is_relative_to(ROOT) or _has_symlink_component(path):
        raise InferenceContractError(f'{name} path escapes the project or contains a symlink: {relative}')
    if not resolved.is_file() or _sha256_file(resolved) != expected:
        raise InferenceContractError(f'{name} digest mismatch: {relative}')


def _checked_regular_file(path, name: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise InferenceContractError(f'{name} must not be a symlink')
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError) as exc:
        raise InferenceContractError(f'{name} is missing') from exc
    if not stat.S_ISREG(mode):
        raise InferenceContractError(f'{name} must be a regular file')
    return resolved


def _checked_relative_resource(bundle: Path, relative: str, name: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise InferenceContractError(f'{name} path must be relative')
    candidate = bundle / relative
    if _has_symlink_component(candidate):
        raise InferenceContractError(f'{name} must not use symlinked paths')
    return _checked_regular_file(candidate, name)


def _read_limited(path: Path, max_bytes: int, name: str) -> bytes:
    try:
        with Path(path).open('rb') as stream:
            value = stream.read(max_bytes + 1)
    except OSError as exc:
        raise InferenceContractError(f'Cannot read {name}') from exc
    if len(value) > max_bytes:
        raise InferenceContractError(f'{name} exceeds its reviewed byte limit')
    return value


def _loads_unique(raw: bytes, name: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InferenceContractError(f'Duplicate JSON key in {name}: {key}')
            result[key] = value
        return result
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              InferenceContractError(f'Invalid nonfinite JSON number in {name}: {token}')))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceContractError(f'Invalid JSON in {name}') from exc


def _exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise InferenceContractError(f'{name} has unexpected or missing fields')


def _require_sha256(value: Any, name: str) -> None:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in '0123456789abcdefABCDEF' for character in value)):
        raise InferenceContractError(f'{name} must be a SHA-256 hex digest')


def _require_relative_path(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise InferenceContractError(f'{name} must be a nonescaping relative path')


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _has_symlink_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value
