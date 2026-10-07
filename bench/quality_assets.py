"""Pinned research sources and bounded model cache; no host pip installation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from dataclasses import dataclass
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, distribution
from types import MappingProxyType
import urllib.request
import zipfile
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/model-quality'
CAPTURE_CACHE = CACHE / 'cache'
BUDGET = 8 * 1024**3


def save_result(path, value):
    """Atomic cache record so the viewer never reads half a JSON document."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def inference_provenance(bundle, *, capture_plan=None, capture_resources=None):
    capture_module = None
    asset = None
    validated_resources = None
    provenance_bundle = None
    input_manifest_sha256 = None
    if capture_plan is None and capture_resources is not None:
        raise ValueError('Capture resources require their authenticated capture plan')
    if capture_plan is not None:
        capture_module = _capture_require_issued_plan(capture_plan)
        try:
            provenance_bundle = Path(bundle).resolve(strict=True)
            issued_bundle = Path(capture_plan.bundle).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise capture_module.CaptureIntegrityError(
                'Capture provenance bundle cannot be resolved') from exc
        if provenance_bundle != issued_bundle:
            raise capture_module.CaptureIntegrityError(
                'Capture plan belongs to a different inference bundle')
        input_manifest_sha256 = digest(provenance_bundle / 'input.json')
        if input_manifest_sha256.upper() != capture_plan.input_manifest_sha256.upper():
            raise capture_module.CaptureIntegrityError(
                'Current input manifest differs from the issued capture plan')
        asset = _read_capture_asset_receipt_validated(capture_plan, capture_module)
        if capture_resources is not None:
            validated_resources = _capture_validate_stage_resources(
                capture_plan, asset, capture_resources, capture_module)

    modules = ('quality_assets', 'quality_contract', 'quality_time', 'quality_runner', 'quality_gotrack',
               'quality_sam2', 'quality_foundpose', 'quality_cnos', 'quality_detection_association', 'quality_neighbors', 'quality_trace', 'quality_memory', 'quality_appearance', 'quality_render_stability', 'vision')
    if input_manifest_sha256 is None:
        provenance_bundle = Path(bundle)
        input_manifest_sha256 = digest(provenance_bundle / 'input.json')
    result = dict(input_manifest_sha256=input_manifest_sha256,
                  adapter_sha256={name: digest(ROOT/'bench'/f'{name}.py') for name in modules},
                  source_revisions=SOURCES, submodule_revisions={k: v[1] for k, v in SUBMODULES.items()},
                  checkpoint_sha256={k: v['sha256'] for k, v in MODELS.items()})
    if capture_plan is None:
        return result
    resource_set = validated_resources
    stage = None if resource_set is None else resource_set.stage
    required = () if resource_set is None else resource_set.required
    used_resources = []
    for use in required:
        descriptor = use.descriptor
        used_resources.append({
            'purpose': use.purpose,
            'resource_kind': descriptor.resource_kind,
            'resource_key': descriptor.resource_key,
            'sidecar_path': descriptor.sidecar_path.relative_to(CAPTURE_CACHE).as_posix(),
            'sidecar_sha256': descriptor.sidecar_sha256,
            'artifact_sha256': descriptor.artifact_sha256,
            'artifact_byte_count': descriptor.artifact_byte_count,
            'recipe': _capture_plain(descriptor.recipe),
            'producer_source_closure': dict(descriptor.source_closure),
        })
    output_use = None if resource_set is None else resource_set.output
    capture_adapter_modules = ['quality_runner']
    if stage in _CAPTURE_STAGE_ADAPTER:
        capture_adapter_modules.extend(_CAPTURE_STAGE_ADAPTER[stage])
    stage_adapter_closure = {
        f'bench/{name}.py': digest(ROOT/'bench'/f'{name}.py')
        for name in dict.fromkeys(capture_adapter_modules)
    }
    result['capture_provenance'] = {
        'source_kind': 'rectified_image_sequence_v1',
        'input_manifest_sha256': capture_plan.input_manifest_sha256,
        'capture_table_sha256': capture_plan.capture_table_sha256,
        'output_camera_sha256': capture_plan.table['output_camera_sha256'],
        'coordinate_mode': 'integer_centers_v1',
        'sampling_policy': dict(_CAPTURE_SAMPLING_POLICY),
        'asset_receipt_sha256': asset.receipt_sha256,
        'renderer_proof_sha256': asset.renderer_proof_sha256,
        'renderer_proof': _capture_plain(asset.renderer_proof),
        'producer_source_closure': {
            path: value for path, value in capture_plan.table['converter']['source_closure'].items()
        },
        'stage_adapter_closure': stage_adapter_closure,
        'stage': stage,
        'render_flags': None if resource_set is None else dict(resource_set.render_flags),
        'used_resources': used_resources,
        'output_resource': None if output_use is None else {
            'resource_kind': output_use.resource_kind,
            'resource_key': output_use.resource_key,
            'sidecar_path': output_use.sidecar_path.relative_to(CAPTURE_CACHE).as_posix(),
            'recipe': _capture_plain(output_use.recipe),
            'producer_source_closure': dict(output_use.source_closure),
        },
    }
    return result


@dataclass(frozen=True, slots=True)
class CaptureAssetReceipt:
    """Detached, immutable meaning of a preflight-authenticated asset receipt."""

    receipt_sha256: str
    asset_sha256: str
    unit_receipt_sha256: str
    object_id: int
    object_name: str
    source_units: str
    conversion_to_metres: float
    document: Mapping[str, object]
    renderer_proof: Mapping[str, object] | None = None
    renderer_proof_sha256: str = ''


@dataclass(frozen=True, slots=True)
class CaptureResourceUse:
    """One required verified sidecar, or the deterministic plan for an output."""

    purpose: str
    resource_kind: str
    resource_key: str
    sidecar_path: Path
    recipe: Mapping[str, object]
    source_closure: Mapping[str, str]
    descriptor: object | None


@dataclass(frozen=True, slots=True)
class CaptureStageResources:
    """Verified prerequisites and a content-addressed output recipe for one stage."""

    stage: str
    asset_receipt: CaptureAssetReceipt
    render_flags: Mapping[str, bool]
    device: str
    required: tuple[CaptureResourceUse, ...]
    output: CaptureResourceUse | None


_CAPTURE_RECEIPT_FIELDS = frozenset({
    'schema_version', 'kind', 'object_id', 'object_name', 'asset_sha256',
    'original_asset_sha256', 'source_units', 'conversion_to_metres',
    'unit_receipt_sha256', 'official_unit_evidence', 'models_info',
    'post_node_bounds_m', 'object_frame', 'loaders', 'geometry_receipt_sha256',
    'renderer_receipt_sha256',
})
_CAPTURE_SAMPLING_POLICY = MappingProxyType({
    'channel_order': 'RGB',
    'interpolation': 'bilinear-four-neighbor',
    'neighbor_footprint': 'all four integer pixel centers must be in bounds and declared valid disk',
    'rounding': 'floor(value+0.5)',
    'clip_range': (0, 255),
    'invalid_fill_rgb': (128, 128, 128),
})
_CAPTURE_RENDERER_PROOF_LIMIT = 2 * 1024 * 1024
_CAPTURE_RENDER_POLICY = 'capture_zero_sample_v1'
_CAPTURE_COORDINATE_MODE = 'integer_centers_v1'
_CAPTURE_RENDERER_SOURCE_PATHS = (
    'bench/quality_gotrack.py',
    'bench/quality_render_stability.py',
    'bench/quality_camera.py',
    'bench/glb_model.py',
    'bench/model.py',
    'bench/renderer.py',
    '.cache/quality-windows/Lib/site-packages/pyrender/offscreen.py',
    '.cache/quality-windows/Lib/site-packages/pyrender/renderer.py',
    '.cache/quality-windows/Lib/site-packages/pyrender/camera.py',
    '.cache/quality-windows/Lib/site-packages/pyrender/mesh.py',
    '.cache/quality-windows/Lib/site-packages/pyrender/primitive.py',
    '.cache/quality-windows/Lib/site-packages/trimesh/exchange/gltf.py',
)
_CAPTURE_RENDERER_IDENTITY_PATHS = MappingProxyType({
    'quality_gotrack': 'bench/quality_gotrack.py',
    'quality_render_stability': 'bench/quality_render_stability.py',
    'pyrender_offscreen': '.cache/quality-windows/Lib/site-packages/pyrender/offscreen.py',
    'pyrender_renderer': '.cache/quality-windows/Lib/site-packages/pyrender/renderer.py',
})
_CAPTURE_RENDERER_MODULE_NAMES = MappingProxyType({
    'quality_gotrack': 'bench.quality_gotrack',
    'quality_render_stability': 'bench.quality_render_stability',
    'pyrender_offscreen': 'pyrender.offscreen',
    'pyrender_renderer': 'pyrender.renderer',
})
_CAPTURE_STAGE_REQUIREMENTS = {
    'smoke': (),
    'banks': ('smoke',),
    'cnos-bank': ('smoke',),
    'segment': ('cnos-bank',),
    'detect': ('cnos-bank',),
    'pose': ('smoke', 'foundpose-bank'),
}
_CAPTURE_STAGE_OUTPUT = {
    'smoke': 'smoke',
    'banks': 'foundpose-bank',
    'cnos-bank': 'cnos-bank',
}
_CAPTURE_STAGE_ADAPTER = {
    'smoke': ('quality_gotrack',),
    'banks': ('quality_foundpose', 'quality_gotrack'),
    'cnos-bank': ('quality_cnos', 'quality_gotrack'),
    'segment': ('quality_sam2',),
    'detect': ('quality_cnos',),
    'pose': ('quality_foundpose', 'quality_gotrack', 'quality_contract'),
}
_CAPTURE_RESOURCE_KIND = {
    'smoke': 'quality-capture-model-smoke-v1',
    'foundpose-bank': 'quality-capture-foundpose-bank-v1',
    'cnos-bank': 'quality-capture-cnos-bank-v1',
}
_CAPTURE_RESOURCE_MODULES = {
    'smoke': (
        'quality_assets', 'quality_camera', 'quality_contract', 'quality_gotrack',
        'quality_render_stability', 'quality_runner', 'quality_time', 'vision',
        'glb_model', 'renderer', 'model', 'show3d_experiment',
    ),
    'foundpose-bank': (
        'quality_assets', 'quality_camera', 'quality_foundpose', 'quality_gotrack',
        'quality_neighbors', 'quality_render_stability', 'quality_time', 'vision',
    ),
    'cnos-bank': (
        'quality_assets', 'quality_camera', 'quality_cnos', 'quality_contract',
        'quality_gotrack', 'quality_render_stability', 'vision',
    ),
}
_CAPTURE_RESOURCE_FLAGS = {
    'smoke': ('unlit_templates', 'disable_multisampling'),
    'foundpose-bank': ('unlit_templates', 'disable_multisampling'),
    'cnos-bank': ('unlit_templates', 'grayscale'),
}
_CAPTURE_STAGE_FLAGS = {
    'smoke': ('unlit_templates', 'disable_multisampling'),
    'banks': ('unlit_templates', 'disable_multisampling'),
    'cnos-bank': ('unlit_templates', 'grayscale'),
    'segment': (),
    'detect': (),
    'pose': ('unlit_templates', 'disable_multisampling'),
}
_CAPTURE_RESOURCE_UPSTREAM_FILES = {
    'smoke': {
        'gotrack': (
            'utils/crop_generation.py', 'utils/data_util.py', 'utils/dinov2_util.py',
            'utils/im_util.py', 'utils/logging.py', 'utils/misc.py',
            'utils/net_util.py', 'utils/poser_util.py', 'utils/renderer.py',
            'utils/renderer_base.py', 'utils/renderer_builder.py', 'utils/structs.py',
            'utils/torch_helpers.py', 'utils/transform3d.py',
            'model/blocks/attention.py', 'model/blocks/config.py',
            'model/blocks/cross_attention.py', 'model/blocks/decoder.py',
            'model/blocks/decoder_block.py', 'model/blocks/mlp.py',
            'model/blocks/rope2d.py', 'model/heads/dpt/config.py',
            'model/heads/dpt/fusion_block.py', 'model/heads/dpt/model.py',
            'model/heads/dpt/util.py',
        ),
        'bop_toolkit': ('bop_toolkit_lib/inout.py',),
        'dinov2': (
            'dinov2/hub/backbones.py', 'dinov2/hub/utils.py',
            'dinov2/models/vision_transformer.py', 'dinov2/layers/__init__.py',
            'dinov2/layers/block.py', 'dinov2/layers/attention.py',
            'dinov2/layers/layer_scale.py', 'dinov2/layers/mlp.py',
            'dinov2/layers/patch_embed.py',
        ),
    },
    'foundpose-bank': {
        'gotrack': (
            'utils/cluster_util.py', 'utils/crop_generation.py', 'utils/data_util.py',
            'utils/dinov2_util.py', 'utils/feature_util.py', 'utils/im_util.py',
            'utils/logging.py', 'utils/misc.py', 'utils/net_util.py', 'utils/pca_util.py',
            'utils/poser_util.py', 'utils/renderer.py', 'utils/renderer_base.py',
            'utils/renderer_builder.py', 'utils/structs.py', 'utils/template_util.py',
            'utils/torch_helpers.py', 'utils/transform3d.py',
            'model/blocks/attention.py', 'model/blocks/config.py',
            'model/blocks/cross_attention.py', 'model/blocks/decoder.py',
            'model/blocks/decoder_block.py', 'model/blocks/mlp.py',
            'model/blocks/rope2d.py', 'model/heads/dpt/config.py',
            'model/heads/dpt/fusion_block.py', 'model/heads/dpt/model.py',
            'model/heads/dpt/util.py',
        ),
        'bop_toolkit': ('bop_toolkit_lib/inout.py',),
        'dinov2': (
            'dinov2/hub/backbones.py', 'dinov2/hub/utils.py',
            'dinov2/models/vision_transformer.py', 'dinov2/layers/__init__.py',
            'dinov2/layers/block.py', 'dinov2/layers/attention.py',
            'dinov2/layers/layer_scale.py', 'dinov2/layers/mlp.py',
            'dinov2/layers/patch_embed.py',
        ),
    },
    'cnos-bank': {
        'gotrack': (
            'utils/crop_generation.py', 'utils/data_util.py', 'utils/im_util.py',
            'utils/logging.py', 'utils/misc.py', 'utils/poser_util.py',
            'utils/renderer.py', 'utils/renderer_base.py', 'utils/renderer_builder.py',
            'utils/structs.py', 'utils/torch_helpers.py', 'utils/transform3d.py',
        ),
        'bop_toolkit': ('bop_toolkit_lib/inout.py',),
        'dinov2': (
            'dinov2/hub/backbones.py', 'dinov2/hub/utils.py',
            'dinov2/models/vision_transformer.py', 'dinov2/layers/__init__.py',
            'dinov2/layers/block.py', 'dinov2/layers/attention.py',
            'dinov2/layers/layer_scale.py', 'dinov2/layers/mlp.py',
            'dinov2/layers/patch_embed.py',
        ),
    },
}
_CAPTURE_RENDERER_RUNTIME_FILES = {
    'pyrender': ('pyrender', (
        '__init__.py', 'camera.py', 'constants.py', 'light.py', 'mesh.py', 'node.py',
        'offscreen.py', 'primitive.py', 'renderer.py', 'scene.py', 'texture.py',
        'platforms/base.py', 'platforms/egl.py', 'platforms/pyglet_platform.py',
    )),
    'trimesh': ('trimesh', (
        '__init__.py', 'base.py', 'scene/__init__.py', 'scene/scene.py',
        'exchange/gltf.py', 'exchange/load.py',
        'visual/objects.py',
    )),
    'PyOpenGL': ('OpenGL', (
        '__init__.py', 'GL/__init__.py', 'platform/__init__.py',
        'platform/baseplatform.py',
    )),
    'pyglet': ('pyglet', (
        '__init__.py', 'app/win32.py', 'canvas/win32.py', 'gl/__init__.py',
        'window/__init__.py', 'window/win32/__init__.py',
    )),
}
_CAPTURE_SOURCE_ARCHIVE_SHA256 = {
    'gotrack': '7C7EDEC1820A8872892FF30957A1FA322FCDBEC396DA4616D46E38702F9DD02E',
    'bop_toolkit': 'F11F854DB50EC9EC7D702C0F10FD71082C840F20CA4083B50C54D53B4D5BABF0',
    'dinov2': 'DB796F1C3A8A1A3D4E78B26E076140D54B985AC0F263A66426955F5E03066070',
}
_CAPTURE_RESOURCE_CHECKPOINTS = {
    'smoke': ('gotrack_checkpoint.pt',),
    'foundpose-bank': ('gotrack_checkpoint.pt',),
    'cnos-bank': ('dinov2_vitl14_pretrain.pth',),
}
_CAPTURE_RESOURCE_SETTINGS = {
    'smoke': {
        'synthetic_pose_count': 3,
        'minimum_iou': 0.94,
        'maximum_median_common_depth_error_m': 0.002,
        'maximum_slanted_plane_ray_error_m': 0.00003,
        'negative_control_required': True,
    },
    'foundpose-bank': {
        'template_count': 798,
        'view_count': 57,
        'rolls_per_view': 14,
        'pca_components': 256,
        'kmeans_clusters': 2048,
        'color_ssaa_factor': 4,
        'depth_mask_ssaa_factor': 1,
    },
    'cnos-bank': {
        'view_count': 57,
        'ssaa_factor': 1,
    },
}


def _capture_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _capture_freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_capture_freeze(child) for child in value)
    return value


def _capture_plain(value):
    if isinstance(value, Mapping):
        return {key: _capture_plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_capture_plain(child) for child in value]
    if isinstance(value, list):
        return [_capture_plain(child) for child in value]
    return value


def _capture_sha(value, name):
    if (type(value) is not str or len(value) != 64 or value != value.upper() or
            any(character not in '0123456789ABCDEF' for character in value)):
        raise ValueError(f'{name} must be an uppercase SHA-256 digest')
    return value


def _capture_real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite real number')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite real number') from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = 'positive ' if positive else ''
        raise ValueError(f'{name} must be a finite {qualifier}real number')
    return result


def _capture_vector(value, name, *, positive=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f'{name} must contain three coordinates')
    result = tuple(_capture_real(item, f'{name}[{index}]') for index, item in enumerate(value))
    if positive and any(item <= 0 for item in result):
        raise ValueError(f'{name} coordinates must be positive')
    return result


def _capture_close(left, right):
    return all(abs(a-b) <= 1e-5 + 1e-4 * abs(b) for a, b in zip(left, right))


def _capture_receipt_json(raw):
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('Asset receipt exceeds the 2 MiB bound')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate asset receipt key: {key!r}')
            result[key] = value
        return result
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('Asset receipt must be bounded UTF-8 JSON') from exc


def _capture_renderer_proof_json(raw):
    if type(raw) is not bytes or len(raw) > _CAPTURE_RENDERER_PROOF_LIMIT:
        raise ValueError('Renderer proof must be bounded bytes no larger than 2 MiB')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate renderer-proof key: {key!r}')
            result[key] = value
        return result

    try:
        document = json.loads(
            raw.decode('utf-8'), object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError('Renderer proof must be valid bounded UTF-8 JSON') from exc

    def finite(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'Renderer proof contains a nonfinite number at {path}')
        if isinstance(value, dict):
            for key, child in value.items():
                finite(child, f'{path}.{key}')
        elif isinstance(value, list):
            for index, child in enumerate(value):
                finite(child, f'{path}[{index}]')

    if type(document) is not dict:
        raise ValueError('Renderer proof must be a JSON object')
    finite(document, '$')
    return document


def _capture_proof_map(value, name):
    if type(value) is not dict:
        raise ValueError(f'{name} must be an object')
    return value


def _capture_proof_true(value, name):
    if type(value) is not bool or value is not True:
        raise ValueError(f'{name} must be true in the renderer proof')


def _capture_proof_false(value, name):
    if type(value) is not bool or value is not False:
        raise ValueError(f'{name} must be false in the renderer proof')


def _capture_proof_int(value, name, *, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be an integer' +
                         (f' >= {minimum}' if minimum is not None else ''))
    return value


def _capture_proof_real(value, name, *, positive=False):
    return _capture_real(value, name, positive=positive)


def _capture_proof_close(left, right, name, *, atol=1e-8, rtol=1e-8):
    left_value = _capture_proof_real(left, f'{name} (actual)')
    right_value = _capture_proof_real(right, f'{name} (expected)')
    if abs(left_value - right_value) > atol + rtol * abs(right_value):
        raise ValueError(f'{name} is inconsistent with the independently recomputed value')


def _capture_proof_vector(value, name, *, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must contain {length} values')
    return tuple(_capture_proof_real(item, f'{name}[{index}]')
                 for index, item in enumerate(value))


def _capture_proof_dimensions(value, name):
    if (type(value) not in (list, tuple) or len(value) != 2 or
            any(type(item) is not int or item <= 0 for item in value)):
        raise ValueError(f'{name} must contain two positive integer dimensions')
    return tuple(value)


def _capture_proof_matrix(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f'{name} must be a 4 by 4 matrix')
    rows = tuple(_capture_proof_vector(row, f'{name}[{index}]', length=4)
                 for index, row in enumerate(value))
    expected_last = (0.0, 0.0, 0.0, 1.0)
    if any(abs(rows[3][i] - expected_last[i]) > 1e-9 for i in range(4)):
        raise ValueError(f'{name} must be an affine camera transform')
    return rows


def _capture_proof_det3(matrix):
    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)


def _capture_proof_validate_pose(row, index):
    name = f'renderer proof rows[{index}]'
    camera_from_object = _capture_proof_matrix(row.get('T_camera_from_object_m'),
                                               f'{name}.T_camera_from_object_m')
    object_from_camera = _capture_proof_matrix(row.get('T_object_from_cv_camera_m'),
                                               f'{name}.T_object_from_cv_camera_m')
    rotation = tuple(tuple(camera_from_object[r][c] for c in range(3)) for r in range(3))
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            _capture_proof_close(dot, 1.0 if i == j else 0.0,
                                 f'{name} rotation orthonormality ({i},{j})', atol=2e-6)
    _capture_proof_close(_capture_proof_det3(rotation), 1.0,
                         f'{name} rotation determinant', atol=2e-6)
    if not any(abs(rotation[r][c] - rotation[c][r]) > 1e-3
               for r in range(3) for c in range(r+1, 3)):
        raise ValueError(f'{name} must be a nonsymmetric front-oblique rotation')

    inverse_rotation = tuple(tuple(rotation[c][r] for c in range(3)) for r in range(3))
    translation = tuple(camera_from_object[r][3] for r in range(3))
    inverse_translation = tuple(-sum(inverse_rotation[r][c] * translation[c]
                                     for c in range(3)) for r in range(3))
    for r in range(3):
        for c in range(3):
            _capture_proof_close(
                object_from_camera[r][c], inverse_rotation[r][c],
                f'{name} inverse rotation[{r},{c}]', atol=2e-6)
        _capture_proof_close(object_from_camera[r][3], inverse_translation[r],
                             f'{name} inverse translation[{r}]', atol=2e-7)

    renderer_input = _capture_proof_matrix(
        row.get('T_object_from_cv_camera_renderer_input_mm'),
        f'{name}.T_object_from_cv_camera_renderer_input_mm')
    for r in range(3):
        for c in range(3):
            _capture_proof_close(renderer_input[r][c], object_from_camera[r][c],
                                 f'{name} renderer rotation[{r},{c}]', atol=2e-6)
        _capture_proof_close(renderer_input[r][3], object_from_camera[r][3] * 1000.0,
                             f'{name} renderer millimetre translation[{r}]', atol=2e-4)
    translation_mm = _capture_proof_vector(row.get('renderer_translation_input_mm'),
                                            f'{name}.renderer_translation_input_mm')
    for index_axis in range(3):
        _capture_proof_close(translation_mm[index_axis], renderer_input[index_axis][3],
                             f'{name} reported renderer translation[{index_axis}]', atol=1e-6)

    for field in ('proper_rotation', 'nonsymmetric_orientation',
                  'camera_transform_roundtrip_passed', 'cv_to_gl_axis_conversion_passed',
                  'renderer_mm_to_m_full_pose_passed', 'depth_gate_passed', 'mask_gate_passed'):
        _capture_proof_true(row.get(field), f'{name}.{field}')
    _capture_proof_close(_capture_proof_real(row.get('camera_space_min_z_m'),
                                              f'{name}.camera_space_min_z_m', positive=True),
                         _capture_proof_real(row.get('camera_space_min_z_m'),
                                              f'{name}.camera_space_min_z_m'),
                         f'{name} positive camera-space z')
    iou = _capture_proof_real(row.get('textured_mask_iou'), f'{name}.textured_mask_iou')
    if not 0.94 <= iou <= 1.0:
        raise ValueError(f'{name} textured mask IoU is below the geometry proof threshold')
    median = _capture_proof_real(row.get('median_common_absolute_depth_error_m'),
                                 f'{name}.median_common_absolute_depth_error_m')
    if median < 0 or median > 0.002:
        raise ValueError(f'{name} common-depth error is outside the geometry proof threshold')
    _capture_proof_int(row.get('common_depth_pixel_count'),
                       f'{name}.common_depth_pixel_count', minimum=1)
    if row.get('textured_renderer_boundary_coordinate_mode') != _CAPTURE_COORDINATE_MODE:
        raise ValueError(f'{name} does not use integer-center renderer coordinates')
    principal = _capture_proof_vector(row.get('pyrender_intrinsics_center_after_boundary_shift'),
                                      f'{name}.pyrender_intrinsics_center_after_boundary_shift',
                                      length=2)
    if any(value < 0 or value >= 280 for value in principal):
        raise ValueError(f'{name} has an invalid output-camera principal point')
    output_camera = _capture_proof_map(row.get('output_camera'), f'{name}.output_camera')
    if (_capture_proof_int(output_camera.get('width'), f'{name}.output_camera.width') != 280 or
            _capture_proof_int(output_camera.get('height'), f'{name}.output_camera.height') != 280):
        raise ValueError(f'{name} must bind the 280 by 280 crop output camera')
    if set(output_camera) != {'cx', 'cy', 'fx', 'fy', 'height', 'width'}:
        raise ValueError(f'{name} output camera fields differ from the fixed crop-camera schema')
    crop_scale = 280.0 / 720.0
    expected_camera = {
        'fx': 360.0 * crop_scale,
        'fy': 360.0 * crop_scale,
        'cx': (359.5 + 0.5) * crop_scale - 0.5,
        'cy': (359.5 + 0.5) * crop_scale - 0.5,
    }
    renderer_input = _capture_proof_map(
        row.get('textured_renderer_input_intrinsics'),
        f'{name}.textured_renderer_input_intrinsics')
    if set(renderer_input) != set(expected_camera):
        raise ValueError(f'{name} renderer input intrinsics fields differ from the fixed crop camera')
    for field, expected in expected_camera.items():
        _capture_proof_close(renderer_input.get(field), expected,
                             f'{name}.textured_renderer_input_intrinsics.{field}')
        _capture_proof_close(output_camera.get(field), expected,
                             f'{name}.output_camera.{field}')
        _capture_proof_close(renderer_input.get(field), output_camera.get(field),
                             f'{name} renderer input/output camera {field}')
    cpu_principal = _capture_proof_vector(row.get('cpu_camera_principal_point'),
                                          f'{name}.cpu_camera_principal_point', length=2)
    expected_cpu_principal = (360.0 * crop_scale, 360.0 * crop_scale)
    if cpu_principal != expected_cpu_principal:
        raise ValueError(f'{name} CPU crop camera principal point differs from integer-center coordinates')
    _capture_proof_close(principal[0], _capture_proof_real(
        output_camera.get('cx'), f'{name}.output_camera.cx') + 0.5,
        f'{name} cropped camera center u')
    _capture_proof_close(principal[1], _capture_proof_real(
        output_camera.get('cy'), f'{name}.output_camera.cy') + 0.5,
        f'{name} cropped camera center v')


def _capture_proof_validate_axis_camera(checks):
    checks = _capture_proof_map(checks, 'renderer proof camera_checks')
    for field in ('positive_x_right', 'positive_y_down', 'positive_z_forward'):
        _capture_proof_true(checks.get(field), f'renderer proof camera_checks.{field}')
    cpu = _capture_proof_vector(checks.get('cpu_principal_point'),
                                'renderer proof camera_checks.cpu_principal_point', length=2)
    sample_offset = _capture_proof_vector(checks.get('cpu_sample_offset_pixels'),
                                          'renderer proof camera_checks.cpu_sample_offset_pixels',
                                          length=2)
    textured = _capture_proof_vector(checks.get('textured_integer_principal_point'),
                                     'renderer proof camera_checks.textured_integer_principal_point',
                                     length=2)
    crop_scale = 280.0 / 720.0
    expected_cpu = (360.0 * crop_scale, 360.0 * crop_scale)
    expected_textured = ((359.5 + 0.5) * crop_scale - 0.5,
                         (359.5 + 0.5) * crop_scale - 0.5)
    if cpu != expected_cpu or sample_offset != (0.5, 0.5) or textured != expected_textured:
        raise ValueError('Renderer proof axis/pixel-center camera values differ from the fixed crop camera')
    ray = _capture_proof_vector(checks.get('center_sample_ray_xy'),
                                'renderer proof camera_checks.center_sample_ray_xy', length=2)
    _capture_proof_close(ray[0], (0.0 - textured[0]) / expected_cpu[0],
                         'renderer proof camera_checks center sample ray x')
    _capture_proof_close(ray[1], (0.0 - textured[1]) / expected_cpu[1],
                         'renderer proof camera_checks center sample ray y')
    axes = checks.get('axis_projection_pixels')
    expected = ((expected_textured[0], expected_textured[1]),
                (expected_textured[0] + expected_cpu[0], expected_textured[1]),
                (expected_textured[0], expected_textured[1] + expected_cpu[1]))
    if type(axes) is not list or len(axes) != len(expected):
        raise ValueError('Renderer proof axis projection evidence must contain the three fixed crop-camera points')
    for index, (actual, target) in enumerate(zip(axes, expected)):
        point = _capture_proof_vector(actual,
                                      f'renderer proof camera_checks.axis_projection_pixels[{index}]',
                                      length=2)
        if point != target:
            raise ValueError('Renderer proof axis projection pixels differ from integer-center crop geometry')


def _capture_proof_validate_array(array, name, *, shape, dtype):
    array = _capture_proof_map(array, name)
    if set(array) != {'dtype', 'order', 'sha256', 'shape'}:
        raise ValueError(f'{name} descriptor fields do not match the renderer array schema')
    _capture_sha(array.get('sha256'), f'{name}.sha256')
    if array.get('dtype') != dtype or array.get('order') != 'C':
        raise ValueError(f'{name} dtype or memory order differs from the renderer array contract')
    actual_shape = array.get('shape')
    if (type(actual_shape) is not list or
            any(type(dimension) is not int or dimension <= 0 for dimension in actual_shape) or
            tuple(actual_shape) != tuple(shape)):
        raise ValueError(f'{name} shape differs from the renderer array contract')
    return array


def _capture_proof_validate_viewport_rows(control, dimensions, name):
    width, height = dimensions
    camera = _capture_proof_map(control.get('camera'), f'{name}.camera')
    if set(camera) != {'cx', 'cy', 'fx', 'fy'}:
        raise ValueError(f'{name} camera must expose only the four queried intrinsics')
    size = width
    factor = size / 720.0
    expected_camera = {
        'fx': 367.5 * factor,
        'fy': 391.25 * factor,
        'cx': (338.75 + 0.5) * factor - 0.5,
        'cy': (352.125 + 0.5) * factor - 0.5,
    }
    for field, expected in expected_camera.items():
        _capture_proof_close(camera.get(field), expected, f'{name}.camera.{field}')
    rows = control.get('rows')
    if type(rows) is not list or len(rows) != 8:
        raise ValueError(f'{name} must retain exactly eight viewport analytic ray samples')
    seen = set()
    errors = []
    for index, value in enumerate(rows):
        row_name = f'{name}.rows[{index}]'
        row = _capture_proof_map(value, row_name)
        pixel = row.get('pixel')
        if type(pixel) not in (list, tuple) or len(pixel) != 2:
            raise ValueError(f'{row_name}.pixel must be a two-integer coordinate')
        u = _capture_proof_int(pixel[0], f'{row_name}.pixel.u')
        v = _capture_proof_int(pixel[1], f'{row_name}.pixel.v')
        if not (0 <= u < width and 0 <= v < height) or (u, v) in seen:
            raise ValueError(f'{row_name} pixel is outside the viewport or duplicated')
        seen.add((u, v))
        x = (u - expected_camera['cx']) / expected_camera['fx']
        y = (v - expected_camera['cy']) / expected_camera['fy']
        denominator = 1.0 - 0.25 * x + 0.15 * y
        if denominator <= 0:
            raise ValueError(f'{row_name} analytic plane intersection is behind the camera')
        expected = 1.0 / denominator
        actual = _capture_proof_real(row.get('actual_m'), f'{row_name}.actual_m', positive=True)
        error = abs(actual - expected)
        _capture_proof_close(row.get('expected_m'), expected,
                             f'{row_name}.expected_m', atol=1e-9, rtol=1e-9)
        _capture_proof_close(row.get('absolute_error_m'), error,
                             f'{row_name}.absolute_error_m', atol=1e-9, rtol=1e-9)
        errors.append(error)
    if len(seen) != 8:
        raise ValueError(f'{name} must use eight distinct viewport pixels')
    maximum = max(errors)
    _capture_proof_close(control.get('max_absolute_depth_error_m'), maximum,
                         f'{name}.max_absolute_depth_error_m', atol=1e-9, rtol=1e-9)
    if maximum > 3e-5:
        raise ValueError(f'{name} recomputed viewport depth error exceeds the fixed threshold')
    _capture_proof_true(control.get('depth_gate_passed'), f'{name}.depth_gate_passed')
    return maximum


def _capture_proof_reject_errors(value, name='renderer proof'):
    error_fields = {'error', 'renderer_close_error', 'query_error', 'allocation_error',
                    'binding_restore_error', 'context_release_error'}
    if isinstance(value, dict):
        for key, child in value.items():
            child_name = f'{name}.{key}'
            if key in error_fields and child is not None:
                raise ValueError(f'{child_name} records an error in an otherwise accepted proof')
            _capture_proof_reject_errors(child, child_name)
    elif type(value) is list:
        for index, child in enumerate(value):
            _capture_proof_reject_errors(child, f'{name}[{index}]')


def _capture_proof_source_slice(document):
    source_hashes = _capture_proof_map(document.get('source_sha256'),
                                       'renderer proof source_sha256')
    current = {}
    root = ROOT.resolve(strict=True)
    for relative in _CAPTURE_RENDERER_SOURCE_PATHS:
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f'Current renderer proof source escapes the repository: {relative}') from exc
        if not path.is_file():
            raise ValueError(f'Current renderer proof source is missing: {relative}')
        actual = digest(path).upper()
        record = _capture_proof_map(source_hashes.get(relative),
                                    f'renderer proof source_sha256[{relative!r}]')
        recorded_bytes = _capture_proof_int(record.get('bytes'),
                                            f'renderer proof source_sha256[{relative!r}].bytes',
                                            minimum=1)
        recorded = _capture_sha(record.get('sha256'),
                                f'renderer proof source_sha256[{relative!r}].sha256')
        if recorded_bytes != path.stat().st_size:
            raise ValueError(f'Renderer proof source byte count differs for {relative}')
        if recorded != actual:
            raise ValueError(f'Renderer proof is stale for current geometry source {relative}')
        current[relative] = actual
    return current


def _capture_proof_report_sources(report, current, name):
    sources = _capture_proof_map(report.get('source_sha256'), f'{name}.source_sha256')
    expected_paths = tuple(_CAPTURE_RENDERER_IDENTITY_PATHS.values())
    if set(sources) != set(expected_paths):
        raise ValueError(f'{name} source inventory differs from the four measured runtime identities')
    for relative in expected_paths:
        value = _capture_sha(sources.get(relative), f'{name}.source_sha256[{relative!r}]')
        if value != current[relative]:
            raise ValueError(f'{name} source digest differs for {relative}')


def _capture_proof_source_identities(metadata, current, name):
    identities = _capture_proof_map(metadata.get('source_identities'),
                                    f'{name}.source_identities')
    if set(identities) != set(_CAPTURE_RENDERER_IDENTITY_PATHS):
        raise ValueError(f'{name} must identify the four pinned renderer/runtime sources')
    for identity, relative in _CAPTURE_RENDERER_IDENTITY_PATHS.items():
        item = _capture_proof_map(identities.get(identity), f'{name}.source_identities.{identity}')
        if item.get('module') != _CAPTURE_RENDERER_MODULE_NAMES[identity]:
            raise ValueError(f'{name} has an unexpected source module for {identity}')
        expected_file = str((ROOT / relative).resolve(strict=True))
        if item.get('file') != expected_file:
            raise ValueError(f'{name} source path differs for {identity}')
        sha = _capture_sha(item.get('sha256'), f'{name}.source_identities.{identity}.sha256')
        if sha != current[relative]:
            raise ValueError(f'{name} source digest differs for {identity}')


def _capture_proof_allocation_pair(pair, dimensions, name):
    pair = _capture_proof_map(pair, name)
    if pair.get('passed') is not True:
        raise ValueError(f'{name} did not pass allocation-pair validation')
    actual_dimensions = _capture_proof_dimensions(pair.get('dimensions'), f'{name}.dimensions')
    if actual_dimensions != tuple(dimensions):
        raise ValueError(f'{name} dimensions do not match the queried framebuffer')
    values = []
    for key, format_name in (('color', 'GL_RGBA'), ('depth', 'GL_DEPTH_COMPONENT24')):
        row = _capture_proof_map(pair.get(key), f'{name}.{key}')
        if (row.get('target_name') != 'GL_RENDERBUFFER' or
                row.get('format_name') != format_name or
                _capture_proof_int(row.get('samples'), f'{name}.{key}.samples') != 4 or
                _capture_proof_int(row.get('width'), f'{name}.{key}.width') != dimensions[0] or
                _capture_proof_int(row.get('height'), f'{name}.{key}.height') != dimensions[1]):
            raise ValueError(f'{name}.{key} is not the pinned 4x ordinary-storage request')
        _capture_proof_true(row.get('ordinary_storage_delegated'),
                            f'{name}.{key}.ordinary_storage_delegated')
        _capture_proof_true(row.get('success'), f'{name}.{key}.success')
        identifier = _capture_proof_int(row.get('renderbuffer_id'),
                                        f'{name}.{key}.renderbuffer_id', minimum=1)
        values.append(identifier)
    if values[0] == values[1]:
        raise ValueError(f'{name} color and depth storage must use distinct renderbuffers')


def _capture_proof_zero_metadata(metadata, dimensions, current, name):
    metadata = _capture_proof_map(metadata, name)
    if metadata.get('status') != 'queried':
        raise ValueError(f'{name} is not a completed framebuffer query')
    for field in ('framebuffer_complete', 'dimension_match', 'framebuffer_bindings_restored',
                  'current_context_released', 'zero_sample_verified'):
        _capture_proof_true(metadata.get(field), f'{name}.{field}')
    for field in ('external_allocation_hook_used',):
        _capture_proof_false(metadata.get(field), f'{name}.{field}')
    if (_capture_proof_int(metadata.get('gl_samples'), f'{name}.gl_samples') != 0 or
            _capture_proof_int(metadata.get('gl_sample_buffers'), f'{name}.gl_sample_buffers') != 0):
        raise ValueError(f'{name} framebuffer is not verified zero-sample storage')
    if _capture_proof_int(metadata.get('framebuffer_status'), f'{name}.framebuffer_status') != 36053:
        raise ValueError(f'{name} framebuffer status is not GL_FRAMEBUFFER_COMPLETE')
    expected_dimensions = _capture_proof_dimensions(
        metadata.get('expected_dimensions'), f'{name}.expected_dimensions')
    if expected_dimensions != tuple(dimensions):
        raise ValueError(f'{name} expected dimensions differ from the requested viewport')
    if metadata.get('sample_positions_status') != 'single_sample_no_multisample_positions':
        raise ValueError(f'{name} sample-position query does not report single-sample storage')
    if metadata.get('sample_positions') != []:
        raise ValueError(f'{name} must not report multisample positions for zero-sample storage')
    fields = _capture_proof_map(metadata.get('framebuffer_fields'), f'{name}.framebuffer_fields')
    if _capture_proof_dimensions(fields.get('multisample_dimensions'),
                                 f'{name}.framebuffer_fields.multisample_dimensions') != tuple(dimensions):
        raise ValueError(f'{name} queried framebuffer dimensions differ')
    draw = _capture_proof_int(fields.get('multisample_draw_fbo'),
                              f'{name}.framebuffer_fields.multisample_draw_fbo', minimum=1)
    read = _capture_proof_int(fields.get('single_sample_read_fbo'),
                              f'{name}.framebuffer_fields.single_sample_read_fbo', minimum=1)
    if draw == read:
        raise ValueError(f'{name} draw and read framebuffer bindings must be distinct')
    bindings = _capture_proof_map(metadata.get('framebuffer_bindings_before_query'),
                                  f'{name}.framebuffer_bindings_before_query')
    _capture_proof_int(bindings.get('draw'), f'{name}.framebuffer_bindings_before_query.draw', minimum=0)
    _capture_proof_int(bindings.get('read'), f'{name}.framebuffer_bindings_before_query.read', minimum=0)

    policy = _capture_proof_map(metadata.get('production_policy_metadata'),
                                f'{name}.production_policy_metadata')
    if (_capture_proof_int(policy.get('schema_version'), f'{name}.policy.schema_version') != 1 or
            policy.get('render_policy') != _CAPTURE_RENDER_POLICY or
            policy.get('coordinate_mode') != _CAPTURE_COORDINATE_MODE or
            policy.get('depth_units') != 'millimetres'):
        raise ValueError(f'{name} renderer policy/coordinate/depth contract differs')
    for field in ('framebuffer_complete', 'dimension_match', 'framebuffer_bindings_restored',
                  'current_context_released'):
        _capture_proof_true(policy.get(field), f'{name}.policy.{field}')
        if metadata.get(field) is not policy.get(field):
            raise ValueError(f'{name} outer query and production policy disagree on {field}')
    if (_capture_proof_int(policy.get('gl_samples'), f'{name}.policy.gl_samples') != 0 or
            _capture_proof_int(policy.get('gl_sample_buffers'),
                               f'{name}.policy.gl_sample_buffers') != 0):
        raise ValueError(f'{name} renderer policy metadata reports multisampling')
    if (_capture_proof_int(metadata.get('gl_samples'), f'{name}.gl_samples') != policy['gl_samples'] or
            _capture_proof_int(metadata.get('gl_sample_buffers'), f'{name}.gl_sample_buffers') !=
            policy['gl_sample_buffers']):
        raise ValueError(f'{name} outer query and production policy disagree on sample counts')
    if 'external_allocation_hook_used' in policy:
        _capture_proof_false(policy.get('external_allocation_hook_used'),
                             f'{name}.policy.external_allocation_hook_used')
    if _capture_proof_dimensions(policy.get('dimensions'), f'{name}.policy.dimensions') != tuple(dimensions):
        raise ValueError(f'{name} renderer policy metadata has stale dimensions')
    if (_capture_proof_dimensions(metadata.get('expected_dimensions'),
                                  f'{name}.expected_dimensions') !=
            _capture_proof_dimensions(policy.get('dimensions'), f'{name}.policy.dimensions')):
        raise ValueError(f'{name} outer query and production policy disagree on dimensions')
    policy_fields = _capture_proof_map(policy.get('framebuffer_fields'),
                                      f'{name}.policy.framebuffer_fields')
    if policy_fields != fields:
        raise ValueError(f'{name} queried and production-policy framebuffer IDs/dimensions disagree')
    _capture_proof_int(policy.get('offscreen_identity'), f'{name}.policy.offscreen_identity', minimum=1)
    _capture_proof_int(policy.get('allocation_generation'),
                       f'{name}.policy.allocation_generation', minimum=1)
    if policy.get('color_storage_policy') != 'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage':
        raise ValueError(f'{name} color storage policy is not ordinary GL_RGBA')
    if policy.get('depth_storage_policy') != 'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage':
        raise ValueError(f'{name} depth storage policy is not ordinary GL_DEPTH_COMPONENT24')
    _capture_proof_source_identities(policy, current, f'{name}.policy')
    pair = policy.get('allocation_pair')
    _capture_proof_allocation_pair(pair, dimensions, f'{name}.policy.allocation_pair')
    calls = policy.get('allocation_calls')
    if type(calls) is not list:
        raise ValueError(f'{name}.policy.allocation_calls must be an array')
    if len(calls) not in (0, 2):
        raise ValueError(f'{name}.policy.allocation_calls must be empty on reuse or one color/depth pair')
    for index, call in enumerate(calls):
        call_map = _capture_proof_map(call, f'{name}.policy.allocation_calls[{index}]')
        if call_map.get('success') is not True:
            raise ValueError(f'{name}.policy.allocation_calls[{index}] records a failed allocation')
    if calls:
        _capture_proof_allocation_pair(
            {'passed': True, 'dimensions': list(dimensions),
             'color': calls[0], 'depth': calls[1]}, dimensions,
            f'{name}.policy fresh allocation log')
        for key in ('color', 'depth'):
            if pair[key] != calls[0 if key == 'color' else 1]:
                raise ValueError(f'{name}.policy allocation log differs from the recorded {key} attachment')
    return policy


def _capture_proof_validate_plane_samples(run, camera, plane, name):
    rows = run.get('rows')
    if type(rows) is not list or len(rows) != 8:
        raise ValueError(f'{name} must retain exactly eight analytic ray samples')
    width = _capture_proof_int(camera.get('width'), f'{name}.camera.width', minimum=1)
    height = _capture_proof_int(camera.get('height'), f'{name}.camera.height', minimum=1)
    fx = _capture_proof_real(camera.get('fx'), f'{name}.camera.fx', positive=True)
    fy = _capture_proof_real(camera.get('fy'), f'{name}.camera.fy', positive=True)
    cx = _capture_proof_real(camera.get('cx'), f'{name}.camera.cx')
    cy = _capture_proof_real(camera.get('cy'), f'{name}.camera.cy')
    slope_x = _capture_proof_real(plane.get('slope_x'), f'{name}.plane.slope_x')
    slope_y = _capture_proof_real(plane.get('slope_y'), f'{name}.plane.slope_y')
    seen_labels = set()
    errors = []
    legacy_errors = []
    reprojection_errors = []
    for index, row in enumerate(rows):
        row_name = f'{name}.rows[{index}]'
        row = _capture_proof_map(row, row_name)
        pixel = row.get('pixel_integer_center')
        if type(pixel) not in (list, tuple) or len(pixel) != 2:
            raise ValueError(f'{row_name}.pixel_integer_center must be an integer pixel pair')
        u = _capture_proof_int(pixel[0], f'{row_name}.pixel_integer_center.u')
        v = _capture_proof_int(pixel[1], f'{row_name}.pixel_integer_center.v')
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError(f'{row_name} pixel is outside the analytic image')
        label = row.get('sample_label')
        if type(label) is not str or not label or label in seen_labels:
            raise ValueError(f'{row_name} sample label is missing or duplicated')
        seen_labels.add(label)
        x = (u - cx) / fx
        y = (v - cy) / fy
        denominator = 1.0 - slope_x*x - slope_y*y
        if denominator <= 0:
            raise ValueError(f'{row_name} analytic plane intersection is behind the camera')
        expected_z = 1.0 / denominator
        expected_point = (x*expected_z, y*expected_z, expected_z)
        ray = _capture_proof_vector(row.get('integer_center_ray_xy'),
                                    f'{row_name}.integer_center_ray_xy', length=2)
        _capture_proof_close(ray[0], x, f'{row_name} integer-center ray x')
        _capture_proof_close(ray[1], y, f'{row_name} integer-center ray y')
        point = _capture_proof_vector(row.get('expected_camera_point_m'),
                                      f'{row_name}.expected_camera_point_m')
        for axis in range(3):
            _capture_proof_close(point[axis], expected_point[axis],
                                 f'{row_name} expected camera point[{axis}]')
        _capture_proof_close(row.get('expected_raw_depth_m'), expected_z,
                             f'{row_name} expected raw depth')
        _capture_proof_close(row.get('inverse_center_depth_A'), denominator,
                             f'{row_name} inverse center depth')
        _capture_proof_true(row.get('renderer_depth_sample_valid'),
                            f'{row_name}.renderer_depth_sample_valid')
        observed_mm = _capture_proof_real(row.get('renderer_depth_raw_mm'),
                                          f'{row_name}.renderer_depth_raw_mm', positive=True)
        observed_m = _capture_proof_real(row.get('renderer_depth_m_after_single_mm_conversion'),
                                          f'{row_name}.renderer_depth_m_after_single_mm_conversion',
                                          positive=True)
        _capture_proof_close(observed_m, observed_mm / 1000.0,
                             f'{row_name} exactly-one millimetre conversion', atol=1e-10)
        error = abs(observed_m - expected_z)
        _capture_proof_close(row.get('absolute_depth_error_m'), error,
                             f'{row_name} absolute depth error', atol=1e-10)
        _capture_proof_close(row.get('signed_depth_error_m_actual_minus_center'),
                             observed_m - expected_z, f'{row_name} signed depth error', atol=1e-10)
        legacy_x = (u + 0.5 - cx) / fx
        legacy_y = (v + 0.5 - cy) / fy
        legacy_denominator = 1.0 - slope_x*legacy_x - slope_y*legacy_y
        if legacy_denominator <= 0:
            raise ValueError(f'{row_name} legacy counterfactual plane intersection is invalid')
        legacy_z = 1.0 / legacy_denominator
        legacy_error = abs(observed_m - legacy_z)
        _capture_proof_close(row.get('legacy_unshifted_counterfactual_depth_m'), legacy_z,
                             f'{row_name} legacy unshifted depth')
        _capture_proof_close(row.get('legacy_unshifted_counterfactual_error_m'), legacy_error,
                             f'{row_name} legacy unshifted error', atol=1e-10)
        projected = _capture_proof_vector(row.get('reprojected_pixel'),
                                          f'{row_name}.reprojected_pixel', length=2)
        projected_expected = (point[0] / point[2] * fx + cx,
                              point[1] / point[2] * fy + cy)
        _capture_proof_close(projected[0], projected_expected[0],
                             f'{row_name} reprojected u', atol=1e-8)
        _capture_proof_close(projected[1], projected_expected[1],
                             f'{row_name} reprojected v', atol=1e-8)
        reprojection_error = max(abs(projected[0] - u), abs(projected[1] - v))
        _capture_proof_close(row.get('reprojection_error_pixels'), reprojection_error,
                             f'{row_name} reprojection error', atol=1e-8)
        _capture_proof_close(row.get('t_observed_A_minus_inverse_actual_depth'),
                             denominator - 1.0 / observed_m,
                             f'{row_name} actual inverse-depth residual', atol=2e-7)
        errors.append(error)
        legacy_errors.append(legacy_error)
        reprojection_errors.append(reprojection_error)
    return max(errors), min(legacy_errors), max(reprojection_errors)


def _capture_proof_validate_plane_report(run, camera, plane, name):
    maximum, minimum_legacy, maximum_reprojection = _capture_proof_validate_plane_samples(
        run, camera, plane, name)
    _capture_proof_close(run.get('max_absolute_depth_error_m'), maximum,
                         f'{name}.max_absolute_depth_error_m', atol=1e-10)
    _capture_proof_close(run.get('minimum_legacy_unshifted_counterfactual_error_m'),
                         minimum_legacy,
                         f'{name}.minimum_legacy_unshifted_counterfactual_error_m', atol=1e-10)
    _capture_proof_close(run.get('maximum_reprojection_error_pixels'), maximum_reprojection,
                         f'{name}.maximum_reprojection_error_pixels', atol=1e-9)
    threshold = _capture_proof_real(run.get('maximum_accepted_depth_error_m'),
                                    f'{name}.maximum_accepted_depth_error_m', positive=True)
    if threshold != 3e-5:
        raise ValueError(f'{name} changed the fixed 30 micrometre depth threshold')
    negative_threshold = _capture_proof_real(
        run.get('minimum_legacy_distinguishing_error_m'),
        f'{name}.minimum_legacy_distinguishing_error_m', positive=True)
    if negative_threshold != 2e-4 or minimum_legacy <= negative_threshold:
        raise ValueError(f'{name} did not retain the fixed half-pixel negative-control separation')
    _capture_proof_true(run.get('all_sample_depths_valid'), f'{name}.all_sample_depths_valid')
    _capture_proof_true(run.get('legacy_unshifted_negative_control_passed'),
                        f'{name}.legacy_unshifted_negative_control_passed')
    if run.get('depth_gate_passed') is not (maximum <= threshold):
        raise ValueError(f'{name}.depth_gate_passed contradicts its recomputed samples')
    if run.get('reprojection_gate_passed') is not (maximum_reprojection <= 1e-10):
        raise ValueError(f'{name}.reprojection_gate_passed contradicts its recomputed samples')
    return maximum, minimum_legacy, maximum_reprojection


def _capture_proof_validate_fixed_plane(camera, plane, camera_pose_cv_to_gl, name):
    camera = _capture_proof_map(camera, f'{name}.camera')
    if (_capture_proof_int(camera.get('width'), f'{name}.camera.width') != 64 or
            _capture_proof_int(camera.get('height'), f'{name}.camera.height') != 48):
        raise ValueError(f'{name} analytic ray camera must be 64 by 48')
    for field, expected in (('fx', 110.0), ('fy', 130.0),
                            ('cx', 23.25), ('cy', 18.75)):
        _capture_proof_close(camera.get(field), expected, f'{name}.camera.{field}')
    if camera_pose_cv_to_gl != (
            'T_world_from_eye_cv @ diag(1,-1,-1,1); identity CV fixture pose'):
        raise ValueError(f'{name} camera pose does not preserve the fixed CV-to-GL test convention')
    plane = _capture_proof_map(plane, f'{name}.plane')
    if plane.get('equation') != 'z = 1 + 0.25*x - 0.15*y':
        raise ValueError(f'{name} analytic plane equation changed')
    _capture_proof_close(plane.get('slope_x'), 0.25, f'{name}.plane.slope_x')
    _capture_proof_close(plane.get('slope_y'), -0.15, f'{name}.plane.slope_y')
    return camera, plane


def _capture_validate_renderer_proof(raw, *, asset_receipt):
    """Validate only the bounded, model-only public geometry proof and return a detached summary."""
    if type(asset_receipt) is not CaptureAssetReceipt:
        raise TypeError('Renderer proof validation requires the exact validated asset receipt')
    frozen_asset = asset_receipt.document
    if not isinstance(frozen_asset, Mapping):
        raise TypeError('Validated asset receipt must expose an immutable document')
    asset = _capture_plain(frozen_asset)
    expected_sha = _capture_sha(asset.get('renderer_receipt_sha256'),
                                'asset receipt renderer_receipt_sha256')
    actual_sha = hashlib.sha256(raw).hexdigest().upper() if type(raw) is bytes else None
    if actual_sha != expected_sha:
        raise ValueError('Renderer proof bytes do not match the issued asset receipt digest')
    proof = _capture_renderer_proof_json(raw)
    _capture_proof_reject_errors(proof)
    if (_capture_proof_int(proof.get('schema_version'), 'renderer proof schema_version') != 1 or
            proof.get('status') != 'completed'):
        raise ValueError('Renderer proof is not a completed schema-v1 receipt')
    _capture_proof_true(proof.get('diagnostic_passed'), 'renderer proof diagnostic_passed')
    for field in ('external_allocation_hook_used', 'captured_rgb_read', 'evaluator_data_read',
                  'inference_executed', 'neural_execution', 'network_used',
                  'model_stage_child_process_used', 'tracking_accuracy_claim'):
        _capture_proof_false(proof.get(field), f'renderer proof {field}')
    for field in ('public_production_render_policy_implemented',
                  'synthetic_mask_is_not_observed_foreground', 'torch_checkpoint_load_blocked'):
        _capture_proof_true(proof.get(field), f'renderer proof {field}')
    _capture_proof_false(proof.get('capture_stage_callers_integrated'),
                         'renderer proof capture_stage_callers_integrated')
    _capture_proof_false(proof.get('capture_depth_policy_integrated'),
                         'renderer proof capture_depth_policy_integrated')
    _capture_proof_true(proof.get('default_capture_policy_gate_failed'),
                        'renderer proof default_capture_policy_gate_failed')
    _capture_proof_true(proof.get('proposed_capture_policy_is_diagnostic_only'),
                        'renderer proof proposed_capture_policy_is_diagnostic_only')

    asset_sha = _capture_sha(proof.get('asset_sha256'), 'renderer proof asset_sha256')
    catalog_sha = _capture_sha(proof.get('catalog_sha256'), 'renderer proof catalog_sha256')
    geometry_sha = _capture_sha(proof.get('geometry_receipt_sha256'),
                                'renderer proof geometry_receipt_sha256')
    models = asset.get('models_info')
    if not isinstance(models, Mapping):
        raise ValueError('Validated asset receipt lacks its catalog record')
    if asset_sha != asset_receipt.asset_sha256 or catalog_sha != models.get('sha256'):
        raise ValueError('Renderer proof asset or catalog identity differs from the issued receipt')
    if geometry_sha != asset.get('geometry_receipt_sha256'):
        raise ValueError('Renderer proof geometry receipt differs from the issued asset')
    if (_capture_proof_int(proof.get('object_id'), 'renderer proof object_id') != asset_receipt.object_id or
            proof.get('object_name') != asset_receipt.object_name):
        raise ValueError('Renderer proof object identity differs from the issued asset')

    current_sources = _capture_proof_source_slice(proof)
    renderer = _capture_proof_map(proof.get('renderer'), 'renderer proof renderer')
    _capture_proof_validate_axis_camera(renderer.get('camera_checks'))
    for field in ('external_allocation_hook_used',):
        _capture_proof_false(renderer.get(field), f'renderer proof renderer.{field}')
    api = _capture_proof_map(renderer.get('textured_renderer_api'),
                             'renderer proof renderer.textured_renderer_api')
    if (api.get('constructor_signature') !=
            "(self, path, obj_id, unlit=False, disable_multisampling=False, coordinate_mode='legacy', *, render_policy='legacy_v1')" or
            api.get('explicit_coordinate_mode_supported') is not True or
            api.get('legacy_coordinate_mode_default') != 'legacy' or
            api.get('selected_coordinate_mode') != _CAPTURE_COORDINATE_MODE):
        raise ValueError('Renderer proof does not bind the reviewed public renderer API')
    settings = _capture_proof_map(renderer.get('render_settings'),
                                  'renderer proof renderer.render_settings')
    _capture_proof_false(settings.get('disable_multisampling'),
                         'renderer proof renderer.render_settings.disable_multisampling')
    _capture_proof_false(settings.get('textured_renderer_unlit'),
                         'renderer proof renderer.render_settings.textured_renderer_unlit')
    if (settings.get('textured_renderer_coordinate_mode') != _CAPTURE_COORDINATE_MODE or
            settings.get('textured_renderer_coordinate_mode_parameter') is not True):
        raise ValueError('Renderer proof did not measure integer-center coordinate mode')
    _capture_proof_close(settings.get('renderer_mm_to_m_scale'), 0.001,
                         'renderer proof millimetre-to-metre scale')

    bounds = models.get('object_bounds_m')
    bounds = _capture_proof_map(bounds, 'asset receipt models_info.object_bounds_m')
    post = _capture_proof_map(asset.get('post_node_bounds_m'), 'asset receipt post_node_bounds_m')
    catalog = _capture_proof_map(renderer.get('catalog'), 'renderer proof renderer.catalog')
    if (_capture_proof_int(catalog.get('object_id'), 'renderer proof catalog.object_id') != 8 or
            catalog.get('object_name') != 'mug_patterned'):
        raise ValueError('Renderer proof catalog identifies a different model')
    _capture_proof_true(catalog.get('asset_receipt_bounds_match'),
                        'renderer proof catalog.asset_receipt_bounds_match')
    if not _capture_close(_capture_proof_vector(catalog.get('size_m'),
                                                 'renderer proof catalog.size_m'),
                          _capture_proof_vector(bounds.get('extents'),
                                                 'asset receipt catalog extents')):
        raise ValueError('Renderer proof catalog extents differ from the issued asset')

    custom = _capture_proof_map(renderer.get('custom_glb_reader'),
                                 'renderer proof renderer.custom_glb_reader')
    trimesh = _capture_proof_map(renderer.get('trimesh_loader'),
                                 'renderer proof renderer.trimesh_loader')
    asset_loaders = frozen_asset.get('loaders')
    if type(asset_loaders) is not tuple or len(asset_loaders) != 2:
        raise ValueError('Validated asset receipt lost its two-loader source binding')
    loader_sources = (
        ('custom_glb', 'bench/glb_model.py'),
        ('trimesh_gltf', '.cache/quality-windows/Lib/site-packages/trimesh/exchange/gltf.py'),
    )
    for index, (loader_name, relative) in enumerate(loader_sources):
        loader = asset_loaders[index]
        if (loader.get('name') != loader_name or
                _capture_sha(loader.get('source_sha256'), f'asset receipt loaders[{index}].source_sha256') !=
                current_sources[relative]):
            raise ValueError(f'Asset receipt {loader_name} source pin differs from current loader bytes')
    for name, loader in (('custom_glb_reader', custom), ('trimesh_loader', trimesh)):
        if not _capture_close(_capture_proof_vector(loader.get('bounds_min_m'),
                                                     f'renderer.{name}.bounds_min_m'),
                              _capture_proof_vector(post.get('min'), 'asset post bounds min')):
            raise ValueError(f'Renderer proof {name} minimum bounds differ from the asset receipt')
        if not _capture_close(_capture_proof_vector(loader.get('bounds_max_m'),
                                                     f'renderer.{name}.bounds_max_m'),
                              _capture_proof_vector(post.get('max'), 'asset post bounds max')):
            raise ValueError(f'Renderer proof {name} maximum bounds differ from the asset receipt')
        if not _capture_close(_capture_proof_vector(loader.get('size_m'),
                                                     f'renderer.{name}.size_m'),
                              _capture_proof_vector(bounds.get('extents'), 'asset catalog extents')):
            raise ValueError(f'Renderer proof {name} extents differ from the asset receipt')
    _capture_proof_true(custom.get('surface_signature_match'),
                        'renderer proof custom_glb_reader.surface_signature_match')
    custom_signature = _capture_sha(custom.get('surface_position_uv_signature_sha256'),
                                    'renderer proof custom surface signature')
    trimesh_signature = _capture_sha(custom.get('trimesh_surface_position_uv_signature_sha256'),
                                     'renderer proof Trimesh surface signature')
    if custom_signature != trimesh_signature:
        raise ValueError('Independent GLB and Trimesh surface-position/UV signatures differ')
    vertex_count = _capture_proof_int(custom.get('vertex_count'),
                                      'renderer proof custom vertex_count', minimum=1)
    triangle_count = _capture_proof_int(custom.get('triangle_count'),
                                        'renderer proof custom triangle_count', minimum=1)
    if (_capture_proof_int(trimesh.get('vertex_count'), 'renderer proof Trimesh vertex_count', minimum=1) != vertex_count or
            _capture_proof_int(trimesh.get('triangle_count'), 'renderer proof Trimesh triangle_count', minimum=1) != triangle_count):
        raise ValueError('Independent GLB and Trimesh topology counts differ')
    uv_conversion = _capture_proof_map(custom.get('independent_trimesh_uv_conversion'),
                                       'renderer proof independent Trimesh UV conversion')
    if (uv_conversion.get('source_pin') != _CAPTURE_RENDERER_SOURCE_PATHS[-1] or
            _capture_sha(uv_conversion.get('source_sha256'), 'renderer proof Trimesh GLTF source') !=
            current_sources[_CAPTURE_RENDERER_SOURCE_PATHS[-1]]):
        raise ValueError('Renderer proof Trimesh UV conversion does not bind the current glTF loader')

    textured = _capture_proof_map(renderer.get('textured_renderer'),
                                  'renderer proof renderer.textured_renderer')
    if (_capture_sha(textured.get('asset_sha256'), 'renderer proof textured asset') != asset_receipt.asset_sha256 or
            textured.get('primitive_position_uv_matches_trimesh') is not True or
            textured.get('transformed_positions_match_trimesh') is not True):
        raise ValueError('Public textured renderer geometry differs from the independent Trimesh loader')
    if _capture_proof_int(textured.get('material_count'),
                          'renderer proof textured material_count', minimum=1) != 1:
        raise ValueError('Public textured renderer did not retain the single expected material')
    flat = _capture_proof_map(textured.get('flat_primitive_validation'),
                              'renderer proof textured flat_primitive_validation')
    _capture_proof_true(flat.get('connectivity_multiplicity_and_winding_preserved'),
                        'renderer proof flat primitive connectivity and winding')
    _capture_proof_true(flat.get('equality'), 'renderer proof flat primitive equality')
    embedded = _capture_proof_map(renderer.get('embedded_textures'),
                                   'renderer proof renderer.embedded_textures')
    material = _capture_proof_map(embedded.get('material'), 'renderer proof embedded material')
    if (material.get('base_color_image_index') != 0 or
            material.get('base_color_texture_index') != 0):
        raise ValueError('Renderer proof material does not bind the expected base-color texture')
    images, textures = embedded.get('images'), embedded.get('textures')
    if (type(images) is not list or not images or type(textures) is not list or not textures or
            textures[0].get('source') != 0):
        raise ValueError('Renderer proof embedded base-color image/texture relationship is invalid')
    image = _capture_proof_map(images[0], 'renderer proof embedded base image')
    if (_capture_proof_int(image.get('width'), 'renderer proof base image width', minimum=1) != 2048 or
            _capture_proof_int(image.get('height'), 'renderer proof base image height', minimum=1) != 2048):
        raise ValueError('Renderer proof base-color texture dimensions differ from the validated asset')
    pixel_sha = _capture_sha(image.get('rgba_pixel_sha256'),
                             'renderer proof base image pixel digest')
    _capture_proof_true(renderer.get('trimesh_base_color_texture_pixel_match'),
                        'renderer proof Trimesh base-color pixel match')
    nodes = trimesh.get('nodes')
    if type(nodes) is not list or not nodes:
        raise ValueError('Renderer proof Trimesh loader must retain its textured node')
    node = _capture_proof_map(nodes[0], 'renderer proof Trimesh node')
    node_texture = _capture_proof_map(node.get('base_color_texture'),
                                      'renderer proof Trimesh node base-color texture')
    if _capture_sha(node_texture.get('rgba_pixel_sha256'),
                    'renderer proof Trimesh texture pixel digest') != pixel_sha:
        raise ValueError('Renderer proof base-color texture pixels differ across loaders')

    runs = _capture_proof_map(renderer.get('slanted_plane_depth_runs'),
                              'renderer proof slanted_plane_depth_runs')
    baseline_names = ('default_msaa_enabled', 'proposed_msaa_rasterization_disabled')
    for index, name in enumerate(baseline_names):
        baseline = _capture_proof_map(runs.get(name), f'renderer proof baseline {name}')
        if baseline.get('mode') != name:
            raise ValueError(f'Baseline {name} has an inconsistent measured mode')
        _capture_proof_false(baseline.get('depth_gate_passed'), f'baseline {name}.depth_gate_passed')
        if _capture_proof_int(baseline.get('sample_count'), f'baseline {name}.sample_count') != 8:
            raise ValueError(f'Baseline {name} must retain all eight analytic samples')
        if index == 0:
            _capture_proof_false(baseline.get('disable_multisampling_option'),
                                 f'baseline {name}.disable_multisampling_option')
        else:
            _capture_proof_true(baseline.get('disable_multisampling_option'),
                                f'baseline {name}.disable_multisampling_option')
        for field in ('all_sample_depths_valid', 'reprojection_gate_passed',
                      'legacy_unshifted_negative_control_passed', 'model_only_control'):
            _capture_proof_true(baseline.get(field), f'baseline {name}.{field}')
        for field in ('observed_foreground', 'tracking_accuracy_claim'):
            _capture_proof_false(baseline.get(field), f'baseline {name}.{field}')
        if 'external_allocation_hook_used' in baseline:
            _capture_proof_false(baseline.get('external_allocation_hook_used'),
                                 f'baseline {name}.external_allocation_hook_used')
        _capture_proof_report_sources(baseline, current_sources, f'baseline {name}')
        baseline_meta = _capture_proof_map(baseline.get('gpu_sampling_metadata'),
                                           f'baseline {name}.gpu_sampling_metadata')
        if baseline_meta.get('status') != 'queried' or baseline_meta.get('sample_positions_status') != 'queried':
            raise ValueError(f'Baseline {name} does not retain its actual multisample query status')
        if (_capture_proof_int(baseline_meta.get('gl_samples'),
                               f'baseline {name}.gl_samples') != 4 or
                _capture_proof_int(baseline_meta.get('gl_sample_buffers'),
                                   f'baseline {name}.gl_sample_buffers') != 1):
            raise ValueError(f'Baseline {name} must retain its measured four-sample attachments')
        _capture_proof_true(baseline_meta.get('framebuffers_supported'),
                            f'baseline {name}.gpu_sampling_metadata.framebuffers_supported')
        _capture_proof_true(baseline_meta.get('framebuffer_bindings_restored'),
                            f'baseline {name}.gpu_sampling_metadata.framebuffer_bindings_restored')
        fields = _capture_proof_map(baseline_meta.get('framebuffer_fields'),
                                    f'baseline {name}.gpu_sampling_metadata.framebuffer_fields')
        if _capture_proof_dimensions(fields.get('multisample_dimensions'),
                                     f'baseline {name}.gpu_sampling_metadata.framebuffer_fields.multisample_dimensions') != (64, 48):
            raise ValueError(f'Baseline {name} framebuffer dimensions differ from its analytic camera')
        draw_id = _capture_proof_int(fields.get('multisample_draw_fbo'),
                                     f'baseline {name}.gpu_sampling_metadata.framebuffer_fields.multisample_draw_fbo',
                                     minimum=1)
        read_id = _capture_proof_int(fields.get('single_sample_read_fbo'),
                                     f'baseline {name}.gpu_sampling_metadata.framebuffer_fields.single_sample_read_fbo',
                                     minimum=1)
        if draw_id == read_id:
            raise ValueError(f'Baseline {name} draw and read framebuffers must be distinct')
        if type(baseline_meta.get('sample_positions')) is not list or len(baseline_meta['sample_positions']) != 4:
            raise ValueError(f'Baseline {name} must retain the four queried sample positions')
        sample_position_fields = {
            'sample_index', 'label', 'sample_x_from_left', 'sample_y_from_bottom',
            'du_from_pixel_center_x', 'dv_top_down_from_pixel_center_y',
        }
        for sample_index, value in enumerate(baseline_meta['sample_positions']):
            sample_name = f'baseline {name}.gpu_sampling_metadata.sample_positions[{sample_index}]'
            sample = _capture_proof_map(value, sample_name)
            if set(sample) != sample_position_fields:
                raise ValueError(f'{sample_name} fields differ from the queried sample-position schema')
            if _capture_proof_int(sample.get('sample_index'), f'{sample_name}.sample_index') != sample_index:
                raise ValueError(f'{sample_name}.sample_index is out of order')
            if type(sample.get('label')) is not str or sample['label'] != f'gl-sample-{sample_index}':
                raise ValueError(f'{sample_name}.label differs from its ordered GL sample index')
            sample_x = _capture_proof_real(sample.get('sample_x_from_left'),
                                           f'{sample_name}.sample_x_from_left')
            sample_y = _capture_proof_real(sample.get('sample_y_from_bottom'),
                                           f'{sample_name}.sample_y_from_bottom')
            if not 0.0 <= sample_x <= 1.0 or not 0.0 <= sample_y <= 1.0:
                raise ValueError(f'{sample_name} coordinates must be within the unit sample square')
            _capture_proof_close(sample.get('du_from_pixel_center_x'), sample_x - 0.5,
                                 f'{sample_name}.du_from_pixel_center_x')
            _capture_proof_close(sample.get('dv_top_down_from_pixel_center_y'), 0.5 - sample_y,
                                 f'{sample_name}.dv_top_down_from_pixel_center_y')
        if _capture_proof_int(baseline_meta.get('draw_framebuffer_bound_for_sample_query'),
                              f'baseline {name}.gpu_sampling_metadata.draw_framebuffer_bound_for_sample_query',
                              minimum=1) != draw_id:
            raise ValueError(f'Baseline {name} sample query is not bound to its measured draw framebuffer')
        bindings = _capture_proof_map(baseline_meta.get('framebuffer_bindings_before_query'),
                                      f'baseline {name}.gpu_sampling_metadata.framebuffer_bindings_before_query')
        _capture_proof_int(bindings.get('draw'),
                           f'baseline {name}.gpu_sampling_metadata.framebuffer_bindings_before_query.draw',
                           minimum=0)
        _capture_proof_int(bindings.get('read'),
                           f'baseline {name}.gpu_sampling_metadata.framebuffer_bindings_before_query.read',
                           minimum=0)
        baseline_camera, baseline_plane = _capture_proof_validate_fixed_plane(
            baseline.get('camera'), baseline.get('plane'),
            baseline.get('camera_pose_cv_to_gl'), f'baseline {name}')
        baseline_max, _, _ = _capture_proof_validate_plane_report(
            baseline, baseline_camera, baseline_plane, f'baseline {name}')
        if baseline_max <= 3e-5:
            raise ValueError(f'Baseline {name} no longer records the failing depth gate')
    _capture_proof_false(proof.get('default_capture_depth_gate_passed'),
                         'renderer proof default_capture_depth_gate_passed')
    _capture_proof_false(proof.get('proposed_capture_msaa_disabled_depth_gate_passed'),
                         'renderer proof proposed_capture_msaa_disabled_depth_gate_passed')
    _capture_proof_true(proof.get('verified_zero_sample_attachment_depth_gate_passed'),
                        'renderer proof verified_zero_sample_attachment_depth_gate_passed')

    candidate = _capture_proof_map(runs.get('verified_zero_sample_attachments'),
                                   'renderer proof verified zero-sample plane run')
    if _capture_plain(renderer.get('slanted_plane_ray_fixture')) != _capture_plain(candidate):
        raise ValueError('Renderer proof duplicate slanted-plane candidate evidence disagrees')
    if (candidate.get('mode') != 'verified_zero_sample_attachments' or
            candidate.get('production_render_policy') != _CAPTURE_RENDER_POLICY or
            candidate.get('coordinate_mode') != _CAPTURE_COORDINATE_MODE):
        raise ValueError('Renderer proof candidate does not use the reviewed capture render policy')
    for field in ('depth_gate_passed', 'reprojection_gate_passed', 'all_sample_depths_valid',
                  'legacy_unshifted_negative_control_passed', 'model_only_control'):
        _capture_proof_true(candidate.get(field), f'renderer proof candidate.{field}')
    for field in ('external_allocation_hook_used', 'observed_foreground',
                  'tracking_accuracy_claim', 'disable_multisampling_option'):
        _capture_proof_false(candidate.get(field), f'renderer proof candidate.{field}')
    if (candidate.get('depth_units') != {
            'renderer_output': 'millimetres', 'expected': 'metres', 'conversion_count': 1}):
        raise ValueError('Renderer proof candidate must document one millimetre-to-metre conversion')
    if candidate.get('capture_policy_status') != 'public_renderer_verified_stage_callers_not_integrated':
        raise ValueError('Renderer proof overstates capture-stage integration')
    camera, plane = _capture_proof_validate_fixed_plane(
        candidate.get('camera'), candidate.get('plane'),
        candidate.get('camera_pose_cv_to_gl'), 'renderer proof candidate')
    max_error, min_legacy, max_reprojection = _capture_proof_validate_plane_report(
        candidate, camera, plane, 'renderer proof candidate')
    if max_error > 3e-5 or min_legacy <= 2e-4 or max_reprojection > 1e-10:
        raise ValueError('Renderer proof analytic plane values do not meet the fixed geometry thresholds')
    candidate_policy = _capture_proof_zero_metadata(
        candidate.get('gpu_sampling_metadata'), (64, 48), current_sources,
        'renderer proof candidate framebuffer')
    if _capture_proof_int(candidate_policy.get('allocation_generation'),
                          'renderer proof candidate allocation generation', minimum=1) != 1:
        raise ValueError('Renderer proof candidate must be the first verified allocation generation')
    calls = candidate.get('zero_sample_storage_allocation_calls')
    if type(calls) is not list or len(calls) != 2:
        raise ValueError('Renderer proof candidate must record exactly two allocation calls')
    _capture_proof_allocation_pair(
        {'passed': True, 'dimensions': [64, 48], 'color': calls[0], 'depth': calls[1]},
        (64, 48), 'renderer proof candidate allocation pair')
    if calls != candidate_policy.get('allocation_calls'):
        raise ValueError('Renderer proof candidate allocation evidence disagrees with queried policy metadata')

    capture_depth = _capture_proof_map(renderer.get('capture_depth_policy'),
                                       'renderer proof capture_depth_policy')
    if (capture_depth.get('current_default_depth_gate_passed') is not False or
            capture_depth.get('current_default_disable_multisampling') is not False):
        raise ValueError('Renderer proof current default policy does not preserve its failed baseline')
    object_metadata = capture_depth.get('object_control_framebuffer_metadata')
    if type(object_metadata) is not list or len(object_metadata) != 4:
        raise ValueError('Renderer proof must retain three object-render queries and the final reuse query')
    object_policies = []
    for index, item in enumerate(object_metadata):
        object_policies.append(_capture_proof_zero_metadata(
            item, (280, 280), current_sources,
            f'renderer proof object framebuffer[{index}]'))
    object_identity = _capture_proof_int(object_policies[0].get('offscreen_identity'),
                                         'renderer proof object framebuffer[0].offscreen_identity',
                                         minimum=1)
    object_pair = object_policies[0].get('allocation_pair')
    object_generation = _capture_proof_int(object_policies[0].get('allocation_generation'),
                                           'renderer proof object framebuffer[0].allocation_generation',
                                           minimum=1)
    for index, policy in enumerate(object_policies):
        if (_capture_proof_int(policy.get('offscreen_identity'),
                               f'renderer proof object framebuffer[{index}].offscreen_identity',
                               minimum=1) != object_identity or
                _capture_proof_int(policy.get('allocation_generation'),
                                   f'renderer proof object framebuffer[{index}].allocation_generation',
                                   minimum=1) != object_generation or
                policy.get('allocation_pair') != object_pair):
            raise ValueError('Renderer proof object controls do not reuse one queried framebuffer allocation')
        expected_calls = (2, 0, 0, 0)[index]
        if len(policy.get('allocation_calls')) != expected_calls:
            raise ValueError('Renderer proof object-control allocation log contradicts first-use/reuse order')

    controls = renderer.get('production_viewport_controls')
    expected_dimensions = ((280, 280), (280, 280), (720, 720),
                           (1120, 1120), (280, 280), (280, 280))
    if type(controls) is not list or len(controls) != 6:
        raise ValueError('Renderer proof must retain all six viewport lifecycle controls')
    if _capture_plain(proof.get('production_viewport_controls')) != _capture_plain(controls):
        raise ValueError('Renderer proof duplicate top-level viewport controls disagree with renderer evidence')
    identities = []
    expected_generations = (2, 2, 3, 4, 5, 6)
    viewport_policies = []
    for index, (control, dimensions) in enumerate(zip(controls, expected_dimensions)):
        name = f'renderer proof viewport[{index}]'
        control = _capture_proof_map(control, name)
        if tuple(_capture_proof_vector(control.get('dimensions'), f'{name}.dimensions', length=2)) != dimensions:
            raise ValueError(f'{name} requested dimensions differ from the fixed lifecycle sequence')
        if _capture_proof_int(control.get('allocation_generation'), f'{name}.allocation_generation') != expected_generations[index]:
            raise ValueError(f'{name} allocation generation does not prove reuse/resize/recreate order')
        expected_reuse = index in (0, 1)
        if type(control.get('same_size_reuse')) is not bool or control['same_size_reuse'] is not expected_reuse:
            raise ValueError(f'{name} same-size reuse flag is inconsistent with the lifecycle')
        if (type(control.get('close_before_render')) is not bool or
                control['close_before_render'] is not (index == 5)):
            raise ValueError(f'{name} close/recreate flag is inconsistent with the lifecycle')
        _capture_proof_true(control.get('depth_gate_passed'), f'{name}.depth_gate_passed')
        _capture_proof_true(control.get('same_frame_mask_depth_agreement'),
                            f'{name}.same_frame_mask_depth_agreement')
        _capture_proof_validate_viewport_rows(control, dimensions, name)
        _capture_proof_validate_array(control.get('color_hash'), f'{name}.color_hash',
                                      shape=(dimensions[1], dimensions[0], 3), dtype='float32')
        _capture_proof_validate_array(control.get('depth_hash'), f'{name}.depth_hash',
                                      shape=(dimensions[1], dimensions[0]), dtype='float32')
        _capture_proof_validate_array(control.get('mask_hash'), f'{name}.mask_hash',
                                      shape=(dimensions[1], dimensions[0]), dtype='bool')
        policy = _capture_proof_zero_metadata(control.get('gpu_sampling_metadata'),
                                             dimensions, current_sources, name)
        viewport_policies.append(policy)
        generation = _capture_proof_int(policy.get('allocation_generation'),
                                        f'{name}.policy.allocation_generation')
        if generation != expected_generations[index]:
            raise ValueError(f'{name} policy metadata generation is inconsistent')
        identity = _capture_proof_int(policy.get('offscreen_identity'),
                                      f'{name}.policy.offscreen_identity', minimum=1)
        identities.append(identity)
        allocations = policy.get('allocation_calls')
        expected_allocation_count = 0 if expected_reuse else 2
        if len(allocations) != expected_allocation_count:
            raise ValueError(f'{name} allocation log does not match reuse versus fresh storage')
        if not expected_reuse:
            for call_index, call in enumerate(allocations):
                call = _capture_proof_map(call, f'{name}.allocation_calls[{call_index}]')
                if (call.get('width') != dimensions[0] or call.get('height') != dimensions[1] or
                        call.get('samples') != 4 or call.get('target_name') != 'GL_RENDERBUFFER'):
                    raise ValueError(f'{name} fresh allocation differs from the requested ordinary-storage size')
    if identities[0] != identities[1] or len(set(identities[1:])) != 5:
        raise ValueError('Renderer proof viewport control identities do not demonstrate reuse and recreation')
    initial_viewport_generation = _capture_proof_int(
        viewport_policies[0].get('allocation_generation'),
        'renderer proof viewport[0].policy.allocation_generation', minimum=1)
    if (object_generation != initial_viewport_generation or
            viewport_policies[0].get('allocation_pair') != viewport_policies[1].get('allocation_pair') or
            viewport_policies[0].get('allocation_pair') != object_pair or
            viewport_policies[0].get('offscreen_identity') != object_identity):
        raise ValueError('Initial viewport reuse is not bound to the object-control generation/framebuffer pair')

    rows = proof.get('rows')
    if type(rows) is not list or len(rows) != 3:
        raise ValueError('Renderer proof requires exactly three front-oblique object controls')
    recipes = []
    poses = []
    for index, row in enumerate(rows):
        row = _capture_proof_map(row, f'renderer proof rows[{index}]')
        _capture_proof_validate_pose(row, index)
        recipes.append(row.get('recipe'))
        poses.append(row.get('T_camera_from_object_m'))
        arrays = _capture_proof_map(row.get('arrays'), f'renderer proof rows[{index}].arrays')
        array_contracts = {
            'cpu_depth_m': ((280, 280), 'float64'),
            'cpu_mask': ((280, 280), 'bool'),
            'cpu_rgb': ((280, 280, 3), 'uint8'),
            'depth_abs_error_m': ((280, 280), 'float64'),
            'textured_depth_m': ((280, 280), 'float64'),
            'textured_depth_mm': ((280, 280), 'float32'),
            'textured_mask': ((280, 280), 'bool'),
            'textured_rgb': ((280, 280, 3), 'uint8'),
        }
        if set(arrays) != set(array_contracts):
            raise ValueError(f'Renderer proof rows[{index}] must retain exactly the eight measured array descriptors')
        validated_arrays = {}
        for array_name, (shape, dtype) in array_contracts.items():
            validated_arrays[array_name] = _capture_proof_validate_array(
                arrays.get(array_name), f'renderer proof rows[{index}].arrays.{array_name}',
                shape=shape, dtype=dtype)
    if recipes != ['front-oblique-a', 'front-oblique-b', 'front-oblique-c']:
        raise ValueError('Renderer proof object controls are not the three ordered front-oblique views')
    if len({json.dumps(pose, separators=(',', ':')) for pose in poses}) != 3:
        raise ValueError('Renderer proof object controls must use three distinct poses')

    descriptor = {
        'schema_version': 1,
        'proof_sha256': expected_sha,
        'asset_sha256': asset_sha,
        'catalog_sha256': catalog_sha,
        'geometry_receipt_sha256': geometry_sha,
        'object_id': asset_receipt.object_id,
        'object_name': asset_receipt.object_name,
        'render_policy': _CAPTURE_RENDER_POLICY,
        'coordinate_mode': _CAPTURE_COORDINATE_MODE,
        'renderer_depth_units': 'millimetres',
        'disable_multisampling': False,
        'unlit': False,
        'source_closure': current_sources,
        'bounds_m': _capture_plain(post),
        'candidate_plane_maximum_error_m': max_error,
        'candidate_plane_minimum_legacy_error_m': min_legacy,
        'candidate_plane_maximum_reprojection_error_pixels': max_reprojection,
        'object_control_count': len(rows),
        'viewport_control_dimensions': [list(item) for item in expected_dimensions],
    }
    return _capture_freeze(descriptor)


def _capture_read_renderer_proof(verified_plan, asset_receipt):
    relative_asset = verified_plan.manifest.get('asset_receipt')
    if relative_asset != 'assets/asset-receipt.json':
        raise ValueError('Capture renderer proof requires the fixed assets/asset-receipt.json bundle path')
    root = Path(verified_plan.bundle).resolve(strict=True)
    proof_path = (root / 'assets' / 'renderer-proof.json').resolve(strict=True)
    try:
        proof_path.relative_to(root)
    except ValueError as exc:
        raise ValueError('Fixed renderer proof path escapes the verified bundle') from exc
    if not proof_path.is_file() or proof_path.stat().st_size > _CAPTURE_RENDERER_PROOF_LIMIT:
        raise ValueError('Fixed renderer proof must be a regular file no larger than 2 MiB')
    raw = proof_path.read_bytes()
    return _capture_validate_renderer_proof(raw, asset_receipt=asset_receipt)


def _validate_capture_asset_document(document, *, manifest, verified_resources):
    if not isinstance(document, dict) or set(document) != _CAPTURE_RECEIPT_FIELDS:
        raise ValueError('Asset receipt fields do not match the pinned metric-asset schema')
    if type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        raise ValueError('Asset receipt schema_version must be 1')
    if document.get('kind') != 'quality-capture-metric-asset-v1':
        raise ValueError('Unsupported capture asset receipt kind')
    if type(document.get('object_id')) is not int or document['object_id'] != 8:
        raise ValueError('Capture asset receipt must describe object_id 8')
    if document.get('object_name') != 'hot3d_obj_000008' or document.get('object_name') != manifest.get('object'):
        raise ValueError('Capture asset receipt object name differs from the authenticated input')
    asset_sha = _capture_sha(document.get('asset_sha256'), 'asset_sha256')
    original_sha = _capture_sha(document.get('original_asset_sha256'), 'original_asset_sha256')
    if asset_sha != original_sha:
        raise ValueError('Metric capture asset must retain the exact original asset bytes')
    asset_path = manifest.get('asset')
    if verified_resources.get(asset_path) != asset_sha:
        raise ValueError('Asset receipt digest does not bind the verified input asset')
    unit_sha = _capture_sha(document.get('unit_receipt_sha256'), 'unit_receipt_sha256')
    if document.get('source_units') != 'metres':
        raise ValueError('Capture asset source_units must be metres')
    scale = _capture_real(document.get('conversion_to_metres'), 'conversion_to_metres', positive=True)
    if scale != 1.0:
        raise ValueError('Capture asset must retain the verified metric scale of 1.0')

    official = document.get('official_unit_evidence')
    if not isinstance(official, dict) or set(official) != {'identifier', 'sha256'}:
        raise ValueError('official_unit_evidence must contain only identifier and sha256')
    if (not isinstance(official.get('identifier'), str) or not official['identifier'].strip() or
            len(official['identifier']) > 512):
        raise ValueError('official unit evidence requires a bounded identifier')
    _capture_sha(official.get('sha256'), 'official_unit_evidence.sha256')

    models = document.get('models_info')
    if not isinstance(models, dict) or set(models) != {'sha256', 'object_bounds_m'}:
        raise ValueError('models_info must contain its digest and sanitized object bounds')
    _capture_sha(models.get('sha256'), 'models_info.sha256')
    catalog = models.get('object_bounds_m')
    if not isinstance(catalog, dict) or set(catalog) != {'min', 'extents', 'diameter'}:
        raise ValueError('models_info object_bounds_m fields mismatch')
    catalog_min = _capture_vector(catalog.get('min'), 'models_info.object_bounds_m.min')
    catalog_extents = _capture_vector(catalog.get('extents'), 'models_info.object_bounds_m.extents', positive=True)
    diameter = _capture_real(catalog.get('diameter'), 'models_info.object_bounds_m.diameter', positive=True)
    if diameter + 1e-5 < max(catalog_extents):
        raise ValueError('models_info diameter is smaller than its declared axis-aligned bounds')

    post = document.get('post_node_bounds_m')
    if not isinstance(post, dict) or set(post) != {'min', 'max', 'extents'}:
        raise ValueError('post_node_bounds_m fields mismatch')
    post_min = _capture_vector(post.get('min'), 'post_node_bounds_m.min')
    post_max = _capture_vector(post.get('max'), 'post_node_bounds_m.max')
    post_extents = _capture_vector(post.get('extents'), 'post_node_bounds_m.extents', positive=True)
    if not _capture_close(post_min, catalog_min) or not _capture_close(post_extents, catalog_extents):
        raise ValueError('Post-node metre bounds differ from documented object-8 catalog bounds')
    if not _capture_close(post_max, tuple(a+b for a, b in zip(post_min, post_extents))):
        raise ValueError('Post-node maximum is inconsistent with minimum plus extents')

    frame = document.get('object_frame')
    expected_frame = {
        'origin': 'original glTF model origin; no recentering',
        'axes': 'glTF right-handed +Y up; camera-from-object explicitly converted at consumers',
    }
    if frame != expected_frame:
        raise ValueError('Capture asset object_frame statement differs from the approved model convention')
    loaders = document.get('loaders')
    if (not isinstance(loaders, list) or len(loaders) != 2 or
            [item.get('name') if isinstance(item, dict) else None for item in loaders] !=
            ['custom_glb', 'trimesh_gltf']):
        raise ValueError('Capture asset receipt requires exactly the two named independent loaders')
    for index, loader in enumerate(loaders):
        if set(loader) != {'name', 'source_sha256'}:
            raise ValueError(f'loaders[{index}] fields mismatch')
        _capture_sha(loader.get('source_sha256'), f'loaders[{index}].source_sha256')
    _capture_sha(document.get('geometry_receipt_sha256'), 'geometry_receipt_sha256')
    _capture_sha(document.get('renderer_receipt_sha256'), 'renderer_receipt_sha256')
    return CaptureAssetReceipt(
        receipt_sha256='', asset_sha256=asset_sha, unit_receipt_sha256=unit_sha,
        object_id=8, object_name=document['object_name'], source_units='metres',
        conversion_to_metres=scale, document=_capture_freeze(document),
    )


def _capture_require_issued_plan(verified_plan):
    from . import quality_capture as capture

    if type(verified_plan) is not capture.VerifiedCapturePlan:
        raise TypeError('Capture asset access requires a public verified capture plan')
    capture._validate_verified_capture_plan_record(verified_plan)
    return capture


def _read_capture_asset_receipt_validated(verified_plan, capture):
    """Read the bounded receipt after the caller validates the issued seal."""
    relative = verified_plan.manifest['asset_receipt']
    expected = verified_plan.verified_resources.get(relative)
    if expected is None or verified_plan.manifest['source_hashes'].get(relative, '').upper() != expected:
        raise capture.CaptureIntegrityError('Asset receipt lacks its preflight resource binding')
    root = Path(verified_plan.bundle).resolve(strict=True)
    path = root.joinpath(*relative.split('/'))
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise capture.CaptureIntegrityError('Asset receipt escapes its verified bundle') from exc
    raw = bytearray()
    with resolved.open('rb') as stream:
        while True:
            block = stream.read(min(64 * 1024, 2 * 1024 * 1024 + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
            if len(raw) > 2 * 1024 * 1024:
                raise capture.CaptureIntegrityError('Asset receipt exceeds its 2 MiB bound')
    raw_bytes = bytes(raw)
    actual = hashlib.sha256(raw_bytes).hexdigest().upper()
    if actual != expected:
        raise capture.CaptureIntegrityError('Asset receipt changed after capture preflight')
    document = _capture_receipt_json(raw_bytes)
    parsed = _validate_capture_asset_document(
        document, manifest=verified_plan.manifest, verified_resources=verified_plan.verified_resources)
    renderer_proof = _capture_read_renderer_proof(verified_plan, parsed)
    renderer_proof_sha256 = _capture_sha(
        parsed.document.get('renderer_receipt_sha256'), 'asset receipt renderer_receipt_sha256')
    return CaptureAssetReceipt(
        receipt_sha256=actual, asset_sha256=parsed.asset_sha256,
        unit_receipt_sha256=parsed.unit_receipt_sha256, object_id=parsed.object_id,
        object_name=parsed.object_name, source_units=parsed.source_units,
        conversion_to_metres=parsed.conversion_to_metres, document=parsed.document,
        renderer_proof=renderer_proof, renderer_proof_sha256=renderer_proof_sha256,
    )


def read_capture_asset_receipt(verified_plan):
    """Read and validate the semantic receipt already hashed by capture preflight."""
    capture_plan = _capture_require_issued_plan(verified_plan)
    return _read_capture_asset_receipt_validated(verified_plan, capture_plan)


def _capture_flags(render_flags):
    allowed = {'unlit_templates', 'disable_multisampling', 'grayscale'}
    if render_flags is None:
        render_flags = {}
    if not isinstance(render_flags, Mapping) or set(render_flags) - allowed:
        raise ValueError('Capture render_flags contain unsupported options')
    normalized = {name: False for name in sorted(allowed)}
    for name, value in render_flags.items():
        if type(value) is not bool:
            raise ValueError(f'Capture render flag {name} must be bool')
        normalized[name] = value
    return MappingProxyType(normalized)


def _capture_validate_stage_flags(stage, flags):
    if flags.get('disable_multisampling') is True:
        raise ValueError('Capture resource recipes require disable_multisampling=false')
    unsupported = [name for name, enabled in flags.items()
                   if enabled and name not in _CAPTURE_STAGE_FLAGS[stage]]
    if unsupported:
        raise ValueError(
            f'Capture stage {stage!r} does not implement enabled render flags: {", ".join(unsupported)}')


def _capture_validate_smoke_descriptor(descriptor, *, asset_receipt,
                                       expected_recipe, expected_source_closure,
                                       expected_key, expected_device):
    """Validate fixed smoke artifacts after their ordinary sidecar is authenticated."""
    from . import quality_capture as capture

    if type(descriptor) is not capture.StageResourceDescriptor:
        raise capture.CaptureIntegrityError('Capture smoke requires a verified stage descriptor')
    key = _capture_sha(expected_key, 'expected smoke resource key')
    root = descriptor.sidecar_path.parent.resolve(strict=True)
    if (descriptor.resource_kind != _CAPTURE_RESOURCE_KIND['smoke'] or
            descriptor.resource_key != key or
            descriptor.sidecar_path.name != 'resource.json' or
            descriptor.artifact_path.name != 'smoke.json' or
            descriptor.artifact_path.parent.resolve(strict=True) != root or
            descriptor.sidecar_path.parent.resolve(strict=True) != root or
            _capture_plain(descriptor.recipe) != _capture_plain(expected_recipe) or
            _capture_plain(descriptor.source_closure) != _capture_plain(expected_source_closure)):
        raise capture.CaptureIntegrityError('Capture smoke descriptor bindings differ from the current recipe')
    expected_root = (CAPTURE_CACHE / 'capture-resources' / key).resolve(strict=True)
    if root != expected_root:
        raise capture.CaptureIntegrityError('Capture smoke descriptor is outside its fixed keyed resource directory')
    if (type(expected_device) is not str or
            descriptor.descriptor.get('runtime', {}).get('device') != expected_device):
        raise capture.CaptureIntegrityError('Capture smoke runtime device differs from the issued device')
    smoke_path = root / 'smoke.json'
    execution_path = root / 'renderer-execution.json'
    smoke_bytes = _capture_smoke_read_bytes(smoke_path, root, 'smoke.json')
    if (len(smoke_bytes) != descriptor.artifact_byte_count or
            hashlib.sha256(smoke_bytes).hexdigest().upper() != descriptor.artifact_sha256):
        raise capture.CaptureIntegrityError(
            'Capture smoke artifact changed after its descriptor was authenticated')
    execution_bytes = _capture_smoke_read_bytes(execution_path, root, 'renderer-execution.json')
    validate_capture_model_smoke_finalized(
        smoke_bytes, execution_bytes, asset_receipt=asset_receipt,
        expected_recipe=expected_recipe, expected_source_closure=expected_source_closure,
        expected_key=key)


_CAPTURE_MODEL_SMOKE_JSON_LIMIT = 2 * 1024 * 1024
_CAPTURE_MODEL_SMOKE_ROTATION = 'Rx@Ry@Rz@diag(1,-1,-1)'
_CAPTURE_MODEL_SMOKE_INTRINSICS = (
    (360.0, 0.0, 359.5),
    (0.0, 360.0, 359.5),
    (0.0, 0.0, 1.0),
)
_CAPTURE_MODEL_SMOKE_VIEWS = (
    {'case_id': 'a', 'euler_xyz_degrees': (9.0, -13.0, 7.0),
     'offset_extent_units': (0.02, -0.04, 3.0)},
    {'case_id': 'b', 'euler_xyz_degrees': (-19.0, 24.0, -12.0),
     'offset_extent_units': (-0.17, 0.11, 3.15)},
    {'case_id': 'c', 'euler_xyz_degrees': (27.0, 15.0, 21.0),
     'offset_extent_units': (0.15, -0.126, 3.3)},
)
_CAPTURE_MODEL_SMOKE_PUBLIC_METADATA_KEYS = frozenset({
    'schema_version', 'render_policy', 'coordinate_mode', 'depth_units', 'dimensions',
    'allocation_generation', 'offscreen_identity', 'framebuffer_fields', 'allocation_pair',
    'allocation_calls', 'framebuffer_complete', 'gl_samples', 'gl_sample_buffers',
    'framebuffer_bindings_restored', 'current_context_released', 'dimension_match',
    'color_storage_policy', 'depth_storage_policy', 'source_identities',
})
_CAPTURE_MODEL_SMOKE_EVENT_KEYS = frozenset({
    'render_index', 'renderer_id', 'case_id', 'ordinal', 'role', 'iteration', 'camera',
    'requested_render_types', 'return_tensors', 'requested_background',
    'render_policy_metadata', 'arrays', 'mask_equals_depth_positive', 'network_inputs',
})
_CAPTURE_MODEL_SMOKE_RENDER_DESCRIPTOR_KEYS = frozenset({
    'gpu_rgb_sha256', 'gpu_depth_mm_sha256', 'gpu_mask_sha256', 'cpu_rgb_sha256',
    'cpu_depth_m_sha256', 'cpu_mask_sha256', 'mask_iou',
    'median_common_depth_error_m', 'common_depth_pixel_count',
    'mask_equals_depth_positive', 'native_dimensions', 'coordinate_mode',
    'render_policy', 'unlit_templates', 'disable_multisampling',
})
_CAPTURE_MODEL_SMOKE_STATS_KEYS = frozenset({
    'correspondences', 'inliers', 'median_reprojection_720', 'p95_reprojection_720',
    'spatial_support', 'score', 'raw_correspondences', 'retained_correspondences',
    'unsupported_correspondences',
})
_CAPTURE_MODEL_SMOKE_BUILDER_KEYS = frozenset({
    'python', 'numpy', 'torch', 'opencv', 'pyrender', 'cuda_runtime', 'gpu_name',
    'dtype', 'batch_size', 'crop_size', 'iterations',
})
_CAPTURE_MODEL_SMOKE_RUNTIME_KEYS = frozenset({
    'schema_version', 'scope', 'clock', 'cuda_synchronization', 'timings_ms',
    'render_calls', 'memory',
})


def _capture_smoke_exact_keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f'{name} keys mismatch (got {actual})')
    return value


def _capture_smoke_int(value, name, *, minimum=None, maximum=None):
    if type(value) is not int or (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum):
        bounds = ''
        if minimum is not None:
            bounds += f' >= {minimum}'
        if maximum is not None:
            bounds += f' <= {maximum}'
        raise ValueError(f'{name} must be an integer{bounds}')
    return value


def _capture_smoke_bool(value, name, expected=None):
    if type(value) is not bool or (expected is not None and value is not expected):
        suffix = '' if expected is None else f'={expected}'
        raise ValueError(f'{name} must be bool{suffix}')
    return value


def _capture_smoke_string(value, name, *, nonempty=True):
    if type(value) is not str or (nonempty and not value.strip()):
        raise ValueError(f'{name} must be a {"nonempty " if nonempty else ""}string')
    return value


def _capture_smoke_close(actual, expected, name, *, tolerance=1e-8):
    actual_value = _capture_real(actual, f'{name} actual')
    expected_value = _capture_real(expected, f'{name} expected')
    if abs(actual_value - expected_value) > tolerance:
        raise ValueError(f'{name} differs from its fixed semantic value')


def _capture_smoke_exact_value(actual, expected, name):
    """Compare JSON trees without Python's bool/int equality shortcut."""
    if type(actual) is not type(expected):
        raise ValueError(f'{name} has the wrong JSON scalar/container type')
    if type(actual) is dict:
        if set(actual) != set(expected):
            raise ValueError(f'{name} keys differ from the authenticated value')
        for key in expected:
            _capture_smoke_exact_value(actual[key], expected[key], f'{name}.{key}')
    elif type(actual) is list:
        if len(actual) != len(expected):
            raise ValueError(f'{name} length differs from the authenticated value')
        for index, (left, right) in enumerate(zip(actual, expected)):
            _capture_smoke_exact_value(left, right, f'{name}[{index}]')
    elif actual != expected:
        raise ValueError(f'{name} differs from the authenticated value')


def _capture_smoke_parse(raw, name):
    if type(raw) is not bytes or len(raw) > _CAPTURE_MODEL_SMOKE_JSON_LIMIT:
        raise ValueError(f'{name} must be bytes no larger than 2 MiB')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate {name} key: {key!r}')
            result[key] = value
        return result

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f'{name} contains a nonfinite JSON number')
        return value

    def reject_constant(token):
        raise ValueError(f'{name} contains nonstandard JSON constant {token}')

    try:
        result = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                            parse_float=finite_float, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise ValueError(f'{name} must be bounded UTF-8 JSON with finite numbers') from exc
    if type(result) is not dict:
        raise ValueError(f'{name} must be a JSON object')
    return result


def _capture_smoke_read_bytes(path, root, name):
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f'{name} is missing or escapes the keyed resource directory') from exc
    if not resolved.is_file() or resolved.stat().st_size > _CAPTURE_MODEL_SMOKE_JSON_LIMIT:
        raise ValueError(f'{name} must be a regular file no larger than 2 MiB')
    raw = bytearray()
    with resolved.open('rb') as stream:
        while True:
            block = stream.read(min(64 * 1024,
                                    _CAPTURE_MODEL_SMOKE_JSON_LIMIT + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
            if len(raw) > _CAPTURE_MODEL_SMOKE_JSON_LIMIT:
                raise ValueError(f'{name} exceeds the 2 MiB bound')
    return bytes(raw)


def _capture_smoke_timestamp(value, name):
    text = _capture_smoke_string(value, name)
    try:
        result = datetime.fromisoformat(text[:-1] + '+00:00' if text.endswith('Z') else text)
    except ValueError as exc:
        raise ValueError(f'{name} must be an ISO-8601 UTC timestamp') from exc
    if result.tzinfo is None or result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError(f'{name} must use UTC')
    return result


def _capture_smoke_descriptor(value, *, name, shape, dtype):
    item = _capture_smoke_exact_keys(value, {'shape', 'dtype', 'order', 'sha256'}, name)
    dimensions = item['shape']
    if (type(dimensions) is not list or any(type(size) is not int or size <= 0 for size in dimensions) or
            dimensions != list(shape)):
        raise ValueError(f'{name}.shape differs from the measured array layout')
    if item['dtype'] != dtype or type(item['dtype']) is not str:
        raise ValueError(f'{name}.dtype differs from the measured array layout')
    if item['order'] != 'C' or type(item['order']) is not str:
        raise ValueError(f'{name}.order must be C')
    descriptor_sha = _capture_sha(item['sha256'], f'{name}.sha256')
    return descriptor_sha


def _capture_smoke_det3(matrix):
    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)


def _capture_smoke_pose(value, name):
    if type(value) is not list or len(value) != 4:
        raise ValueError(f'{name} must be a JSON 4 by 4 matrix')
    rows = []
    for index, row in enumerate(value):
        if type(row) is not list or len(row) != 4:
            raise ValueError(f'{name}[{index}] must contain four finite values')
        rows.append(tuple(_capture_real(cell, f'{name}[{index}][{column}]')
                          for column, cell in enumerate(row)))
    matrix = tuple(rows)
    if any(abs(matrix[3][column] - expected) > 1e-9
           for column, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f'{name} must have an affine last row')
    for left in range(3):
        for right in range(3):
            dot = sum(matrix[row][left] * matrix[row][right] for row in range(3))
            if abs(dot - (1.0 if left == right else 0.0)) > 2e-6:
                raise ValueError(f'{name} rotation is not orthonormal')
    if abs(_capture_smoke_det3(matrix) - 1.0) > 2e-6:
        raise ValueError(f'{name} rotation determinant must be +1')
    return matrix


def _capture_smoke_matrix_multiply(left, right):
    return tuple(tuple(sum(left[row][k] * right[k][column] for k in range(3))
                       for column in range(3)) for row in range(3))


def _capture_smoke_seed_rotation(degrees):
    x, y, z = (math.radians(value) for value in degrees)
    rx = ((1.0, 0.0, 0.0), (0.0, math.cos(x), -math.sin(x)),
          (0.0, math.sin(x), math.cos(x)))
    ry = ((math.cos(y), 0.0, math.sin(y)), (0.0, 1.0, 0.0),
          (-math.sin(y), 0.0, math.cos(y)))
    rz = ((math.cos(z), -math.sin(z), 0.0), (math.sin(z), math.cos(z), 0.0),
          (0.0, 0.0, 1.0))
    result = _capture_smoke_matrix_multiply(
        _capture_smoke_matrix_multiply(_capture_smoke_matrix_multiply(rx, ry), rz),
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    return result


def _capture_smoke_expected_seed(degrees, offset, bounds, name):
    minimum = _capture_vector(bounds['min'], f'{name}.bounds.min')
    maximum = _capture_vector(bounds['max'], f'{name}.bounds.max')
    extents = tuple(maximum[index] - minimum[index] for index in range(3))
    if any(value <= 0 for value in extents):
        raise ValueError(f'{name} post-node asset bounds must have positive extent')
    center = tuple((minimum[index] + maximum[index]) / 2 for index in range(3))
    extent = max(extents)
    rotation = _capture_smoke_seed_rotation(degrees)
    translation = tuple(extent * offset[row] -
                        sum(rotation[row][column] * center[column] for column in range(3))
                        for row in range(3))
    return tuple(tuple(rotation[row][column] for column in range(3)) + (translation[row],)
                 for row in range(3)) + ((0.0, 0.0, 0.0, 1.0),)


def _capture_smoke_pose_close(actual, expected, name):
    for row in range(3):
        for column in range(4):
            tolerance = 2e-7 if column == 3 else 2e-6
            if abs(actual[row][column] - expected[row][column]) > tolerance:
                raise ValueError(f'{name} differs from the fixed model-only pose recipe')


def _capture_smoke_inverse_camera(seed, name):
    rotation = tuple(tuple(seed[column][row] for column in range(3)) for row in range(3))
    translation = tuple(-sum(rotation[row][column] * seed[column][3]
                             for column in range(3)) * 1000.0 for row in range(3))
    return tuple(tuple(rotation[row][column] for column in range(3)) + (translation[row],)
                 for row in range(3)) + ((0.0, 0.0, 0.0, 1.0),)


def _capture_smoke_camera(value, *, width, height, name, native_seed=None):
    camera = _capture_smoke_exact_keys(
        value, {'width', 'height', 'fx', 'fy', 'cx', 'cy', 'T_world_from_eye_mm'}, name)
    if (_capture_smoke_int(camera['width'], f'{name}.width', minimum=1) != width or
            _capture_smoke_int(camera['height'], f'{name}.height', minimum=1) != height):
        raise ValueError(f'{name} dimensions do not match the render role')
    fx = _capture_real(camera['fx'], f'{name}.fx', positive=True)
    fy = _capture_real(camera['fy'], f'{name}.fy', positive=True)
    cx = _capture_real(camera['cx'], f'{name}.cx')
    cy = _capture_real(camera['cy'], f'{name}.cy')
    pose = _capture_smoke_pose(camera['T_world_from_eye_mm'], f'{name}.T_world_from_eye_mm')
    if native_seed is not None:
        for actual, expected, field in ((fx, 360.0, 'fx'), (fy, 360.0, 'fy'),
                                        (cx, 359.5, 'cx'), (cy, 359.5, 'cy')):
            _capture_smoke_close(actual, expected, f'{name}.{field}')
        _capture_smoke_pose_close(pose, _capture_smoke_inverse_camera(native_seed, name), name)
    return fx, fy, cx, cy, pose


def _capture_smoke_validate_build(build, *, device, name):
    value = _capture_smoke_exact_keys(build, _CAPTURE_MODEL_SMOKE_BUILDER_KEYS, name)
    for field in ('python', 'numpy', 'torch', 'opencv', 'pyrender'):
        _capture_smoke_string(value[field], f'{name}.{field}')
    if value['dtype'] != 'float32' or type(value['dtype']) is not str:
        raise ValueError(f'{name}.dtype must be float32')
    if _capture_smoke_int(value['batch_size'], f'{name}.batch_size') != 1:
        raise ValueError(f'{name}.batch_size must be one')
    if value['crop_size'] != [280, 280] or type(value['crop_size']) is not list or any(
            type(size) is not int for size in value['crop_size']):
        raise ValueError(f'{name}.crop_size must be [280, 280]')
    if _capture_smoke_int(value['iterations'], f'{name}.iterations') != 5:
        raise ValueError(f'{name}.iterations must be five')
    if device == 'cuda':
        _capture_smoke_string(value['cuda_runtime'], f'{name}.cuda_runtime')
        _capture_smoke_string(value['gpu_name'], f'{name}.gpu_name')
    else:
        if value['cuda_runtime'] is not None or value['gpu_name'] is not None:
            raise ValueError(f'{name} must not claim CUDA build details on CPU')


def _capture_smoke_validate_runtime(runtime, *, device, render_events, name):
    value = _capture_smoke_exact_keys(runtime, _CAPTURE_MODEL_SMOKE_RUNTIME_KEYS, name)
    if _capture_smoke_int(value['schema_version'], f'{name}.schema_version') != 1:
        raise ValueError(f'{name}.schema_version must be one')
    if value['scope'] != 'model-only-smoke' or type(value['scope']) is not str:
        raise ValueError(f'{name}.scope must be model-only-smoke')
    if value['clock'] != 'perf_counter_ns' or type(value['clock']) is not str:
        raise ValueError(f'{name}.clock must be perf_counter_ns')
    sync = 'phase-boundaries' if device == 'cuda' else 'not-applicable'
    if value['cuda_synchronization'] != sync or type(value['cuda_synchronization']) is not str:
        raise ValueError(f'{name}.cuda_synchronization differs from the selected device')
    timing = _capture_smoke_exact_keys(
        value['timings_ms'], {'worker_total', 'network_load', 'renderer_construct',
                              'renderer_close', 'cases'}, f'{name}.timings_ms')
    worker_total = _capture_real(timing['worker_total'], f'{name}.timings_ms.worker_total', positive=True)
    phases = []
    for field in ('network_load', 'renderer_construct', 'renderer_close'):
        measured = _capture_real(timing[field], f'{name}.timings_ms.{field}')
        if measured < 0:
            raise ValueError(f'{name}.timings_ms.{field} must be nonnegative')
        phases.append(measured)
    cases = timing['cases']
    if type(cases) is not list or len(cases) != 3:
        raise ValueError(f'{name}.timings_ms.cases must contain a/b/c')
    native_elapsed = {}
    refine_elapsed = {}
    for index, case_id in enumerate(('a', 'b', 'c')):
        case = _capture_smoke_exact_keys(
            cases[index], {'case_id', 'native_render', 'cpu_geometry', 'refine', 'validation'},
            f'{name}.timings_ms.cases[{index}]')
        if case['case_id'] != case_id or type(case['case_id']) is not str:
            raise ValueError(f'{name}.timings_ms.cases order must be a/b/c')
        for field in ('native_render', 'cpu_geometry', 'refine', 'validation'):
            measured = _capture_real(case[field], f'{name}.timings_ms.cases[{index}].{field}')
            if measured < 0:
                raise ValueError(
                    f'{name}.timings_ms.cases[{index}].{field} must be nonnegative')
            phases.append(measured)
            if field == 'native_render':
                native_elapsed[case_id] = measured
            elif field == 'refine':
                refine_elapsed[case_id] = measured
    if sum(phases) > worker_total + 1.0:
        raise ValueError(f'{name}.timings_ms phase sum exceeds worker_total')
    calls = value['render_calls']
    if type(calls) is not list or len(calls) != 18:
        raise ValueError(f'{name}.render_calls must contain all 18 delegated renders')
    render_elapsed = []
    for index, call in enumerate(calls):
        item = _capture_smoke_exact_keys(call, {'render_index', 'elapsed_ms'},
                                         f'{name}.render_calls[{index}]')
        if _capture_smoke_int(item['render_index'], f'{name}.render_calls[{index}].render_index') != index:
            raise ValueError(f'{name}.render_calls must preserve render order')
        elapsed = _capture_real(item['elapsed_ms'], f'{name}.render_calls[{index}].elapsed_ms')
        if elapsed < 0:
            raise ValueError(f'{name}.render_calls[{index}].elapsed_ms must be nonnegative')
        render_elapsed.append(elapsed)
        if index in (0, 6, 12):
            case_id = ('a', 'b', 'c')[(index // 6)]
            if elapsed != native_elapsed[case_id]:
                raise ValueError(f'{name} native render timing differs from its observer sample')
    for case_index, case_id in enumerate(('a', 'b', 'c')):
        crop_elapsed = sum(render_elapsed[case_index * 6 + 1:case_index * 6 + 6])
        if crop_elapsed > refine_elapsed[case_id] + 1.0:
            raise ValueError(
                f'{name} crop render timings exceed case {case_id} refinement duration')
    memory = _capture_smoke_exact_keys(
        value['memory'], {'measurement', 'cuda_total_bytes', 'cuda_peak_allocated_bytes',
                          'cuda_peak_reserved_bytes', 'process_peak_rss_bytes',
                          'process_peak_rss_method'}, f'{name}.memory')
    if device == 'cuda':
        if memory['measurement'] != 'torch-cuda-allocator-v1':
            raise ValueError(f'{name}.memory measurement must identify the CUDA allocator')
        total = _capture_smoke_int(memory['cuda_total_bytes'], f'{name}.memory.cuda_total_bytes', minimum=1)
        allocated = _capture_smoke_int(memory['cuda_peak_allocated_bytes'],
                                       f'{name}.memory.cuda_peak_allocated_bytes', minimum=1)
        reserved = _capture_smoke_int(memory['cuda_peak_reserved_bytes'],
                                      f'{name}.memory.cuda_peak_reserved_bytes', minimum=1)
        if allocated > reserved or reserved > total:
            raise ValueError(f'{name}.memory CUDA allocator totals are inconsistent')
    else:
        if memory['measurement'] != 'cpu-no-cuda' or any(
                memory[field] is not None for field in
                ('cuda_total_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')):
            raise ValueError(f'{name}.memory must report null CUDA metrics for CPU')
    rss, method = memory['process_peak_rss_bytes'], memory['process_peak_rss_method']
    if rss is None:
        if method is not None:
            raise ValueError(f'{name}.memory RSS method must be null when RSS is unavailable')
    else:
        _capture_smoke_int(rss, f'{name}.memory.process_peak_rss_bytes', minimum=1)
        if method not in ('GetProcessMemoryInfo.PeakWorkingSetSize', 'resource.ru_maxrss'):
            raise ValueError(f'{name}.memory process RSS method is not an approved measurement API')
    return value


def _capture_smoke_constructor(records, *, asset_receipt, recipe, name):
    if type(records) is not list or len(records) != 1:
        raise ValueError(f'{name} must contain exactly one public renderer constructor')
    item = _capture_smoke_exact_keys(
        records[0], {'renderer_id', 'api', 'object_id', 'asset_sha256', 'arguments',
                     'observed_configuration'}, f'{name}[0]')
    if _capture_smoke_int(item['renderer_id'], f'{name}[0].renderer_id', minimum=0) != 0:
        raise ValueError(f'{name} renderer id must be zero')
    if item['api'] != 'bench.quality_gotrack.TexturedRenderer' or type(item['api']) is not str:
        raise ValueError(f'{name} must identify the public TexturedRenderer constructor')
    if _capture_smoke_int(item['object_id'], f'{name}[0].object_id') != asset_receipt.object_id:
        raise ValueError(f'{name} object id differs from the authenticated asset')
    if _capture_sha(item['asset_sha256'], f'{name}[0].asset_sha256') != asset_receipt.asset_sha256:
        raise ValueError(f'{name} asset digest differs from the authenticated asset')
    contract = recipe['rendering_contract']
    unlit = contract['textured_renderer_unlit']
    arguments = _capture_smoke_exact_keys(
        item['arguments'], {'unlit', 'disable_multisampling', 'coordinate_mode', 'render_policy'},
        f'{name}[0].arguments')
    expected_args = {'unlit': unlit, 'disable_multisampling': False,
                     'coordinate_mode': _CAPTURE_COORDINATE_MODE,
                     'render_policy': _CAPTURE_RENDER_POLICY}
    _capture_smoke_exact_value(arguments, expected_args, f'{name}[0].arguments')
    observed = _capture_smoke_exact_keys(
        item['observed_configuration'], set(expected_args) | {
            'scene_bg_rgba', 'ambient_light_rgb', 'spotlight_intensity',
            'spotlight_inner_cone', 'spotlight_outer_cone'}, f'{name}[0].observed_configuration')
    for field, expected in expected_args.items():
        _capture_smoke_exact_value(observed[field], expected,
                                   f'{name}[0].observed_configuration.{field}')
    raw_background = observed['scene_bg_rgba']
    if type(raw_background) is not list or len(raw_background) != 4:
        raise ValueError(f'{name} background RGBA must have four values')
    for index, expected in enumerate((0.5, 0.5, 0.5, 0.0)):
        _capture_smoke_close(raw_background[index], expected, f'{name}[0].scene_bg_rgba[{index}]')
    ambient = observed['ambient_light_rgb']
    if type(ambient) is not list or len(ambient) != 3:
        raise ValueError(f'{name}[0].ambient_light_rgb must contain three values')
    for index, expected in enumerate((0.02, 0.02, 0.02)):
        _capture_smoke_close(ambient[index], expected, f'{name}[0].ambient_light_rgb[{index}]')
    for field, expected in (('spotlight_intensity', 2.4),
                            ('spotlight_inner_cone', math.pi / 16),
                            ('spotlight_outer_cone', math.pi / 6)):
        _capture_smoke_close(observed[field], expected, f'{name}[0].{field}', tolerance=2e-7)


def _capture_smoke_metadata(metadata, *, event, expected_generation, current_sources, name,
                            source_closure, lifecycle):
    value = _capture_smoke_exact_keys(
        metadata, _CAPTURE_MODEL_SMOKE_PUBLIC_METADATA_KEYS, name)
    width = 720 if event['role'] == 'native' else 280
    height = width
    expected_dims = [width, height]
    if _capture_smoke_int(value['schema_version'], f'{name}.schema_version') != 1:
        raise ValueError(f'{name}.schema_version must be one')
    if value['render_policy'] != _CAPTURE_RENDER_POLICY or type(value['render_policy']) is not str:
        raise ValueError(f'{name}.render_policy is not the pinned capture policy')
    if value['coordinate_mode'] != _CAPTURE_COORDINATE_MODE or type(value['coordinate_mode']) is not str:
        raise ValueError(f'{name}.coordinate_mode is not integer centers')
    if value['depth_units'] != 'millimetres' or type(value['depth_units']) is not str:
        raise ValueError(f'{name}.depth_units must be millimetres')
    if type(value['dimensions']) is not list or value['dimensions'] != expected_dims or any(
            type(size) is not int for size in value['dimensions']):
        raise ValueError(f'{name}.dimensions differ from the event camera')
    if _capture_smoke_int(value['allocation_generation'], f'{name}.allocation_generation') != expected_generation:
        raise ValueError(f'{name}.allocation_generation differs from the public resize lifecycle')
    _capture_smoke_int(value['offscreen_identity'], f'{name}.offscreen_identity', minimum=1)
    for field in ('framebuffer_complete', 'framebuffer_bindings_restored',
                  'current_context_released', 'dimension_match'):
        _capture_smoke_bool(value[field], f'{name}.{field}', True)
    for field in ('gl_samples', 'gl_sample_buffers'):
        if _capture_smoke_int(value[field], f'{name}.{field}') != 0:
            raise ValueError(f'{name}.{field} must verify zero-sample storage')
    if value['color_storage_policy'] != 'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage':
        raise ValueError(f'{name}.color_storage_policy differs from the reviewed renderer')
    if value['depth_storage_policy'] != 'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage':
        raise ValueError(f'{name}.depth_storage_policy differs from the reviewed renderer')
    fields = _capture_smoke_exact_keys(
        value['framebuffer_fields'], {'multisample_draw_fbo', 'single_sample_read_fbo',
                                      'multisample_dimensions'}, f'{name}.framebuffer_fields')
    draw = _capture_smoke_int(fields['multisample_draw_fbo'],
                              f'{name}.framebuffer_fields.multisample_draw_fbo', minimum=1)
    read = _capture_smoke_int(fields['single_sample_read_fbo'],
                              f'{name}.framebuffer_fields.single_sample_read_fbo', minimum=1)
    if (draw == read or type(fields['multisample_dimensions']) is not list or
            fields['multisample_dimensions'] != expected_dims or
            any(type(size) is not int for size in fields['multisample_dimensions'])):
        raise ValueError(f'{name}.framebuffer_fields do not identify the requested framebuffer')
    pair = _capture_smoke_exact_keys(value['allocation_pair'],
                                     {'color', 'depth', 'dimensions', 'passed'},
                                     f'{name}.allocation_pair')
    _capture_smoke_bool(pair['passed'], f'{name}.allocation_pair.passed', True)
    if (pair['dimensions'] != expected_dims or type(pair['dimensions']) is not list or
            any(type(size) is not int for size in pair['dimensions'])):
        raise ValueError(f'{name}.allocation_pair dimensions mismatch')
    allocation_rows = []
    for key, format_name in (('color', 'GL_RGBA'), ('depth', 'GL_DEPTH_COMPONENT24')):
        row = _capture_smoke_exact_keys(
            pair[key], {'target_name', 'format_name', 'samples', 'width', 'height',
                        'renderbuffer_id', 'ordinary_storage_delegated', 'success'},
            f'{name}.allocation_pair.{key}')
        if row['target_name'] != 'GL_RENDERBUFFER' or row['format_name'] != format_name:
            raise ValueError(f'{name}.allocation_pair.{key} has the wrong storage format')
        if (_capture_smoke_int(row['samples'], f'{name}.{key}.samples') != 4 or
                _capture_smoke_int(row['width'], f'{name}.{key}.width') != width or
                _capture_smoke_int(row['height'], f'{name}.{key}.height') != height):
            raise ValueError(f'{name}.allocation_pair.{key} differs from the ordinary storage request')
        _capture_smoke_int(row['renderbuffer_id'], f'{name}.{key}.renderbuffer_id', minimum=1)
        _capture_smoke_bool(row['ordinary_storage_delegated'],
                            f'{name}.{key}.ordinary_storage_delegated', True)
        _capture_smoke_bool(row['success'], f'{name}.{key}.success', True)
        allocation_rows.append(row)
    if allocation_rows[0]['renderbuffer_id'] == allocation_rows[1]['renderbuffer_id']:
        raise ValueError(f'{name} color and depth renderbuffers must be distinct')
    calls = value['allocation_calls']
    if type(calls) is not list:
        raise ValueError(f'{name}.allocation_calls must be a JSON list')
    fresh = event['render_index'] in (0, 1, 6, 7, 12, 13)
    if fresh:
        if len(calls) != 2:
            raise ValueError(f'{name} fresh allocation must log both storage calls')
        for index, call in enumerate(calls):
            _capture_smoke_exact_value(call, allocation_rows[index],
                                       f'{name}.allocation_calls[{index}]')
    elif calls:
        raise ValueError(f'{name} reused framebuffer must have an empty allocation log')
    sources = value['source_identities']
    if type(sources) is not dict or set(sources) != set(_CAPTURE_RENDERER_IDENTITY_PATHS):
        raise ValueError(f'{name}.source_identities inventory mismatch')
    for identity, source in sources.items():
        _capture_smoke_exact_keys(source, {'module', 'file', 'sha256'},
                                  f'{name}.source_identities.{identity}')
    _capture_proof_source_identities(value, current_sources, name)
    closure_identity_paths = {
        'quality_gotrack': 'bench/quality_gotrack.py',
        'quality_render_stability': 'bench/quality_render_stability.py',
        'pyrender_renderer': 'runtime/pyrender/renderer.py',
        'pyrender_offscreen': 'runtime/pyrender/offscreen.py',
    }
    for identity, closure_path in closure_identity_paths.items():
        source_sha = _capture_sha(sources[identity].get('sha256'),
                                  f'{name}.source_identities.{identity}.sha256')
        if source_closure.get(closure_path) != source_sha:
            raise ValueError(f'{name} source identity differs from producer closure {closure_path}')
    identities = _capture_plain(value['source_identities'])
    if not lifecycle:
        lifecycle[expected_generation] = {
            'dimensions': expected_dims, 'offscreen_identity': value['offscreen_identity'],
            'framebuffer_fields': _capture_plain(fields), 'allocation_pair': _capture_plain(pair),
            'source_identities': identities,
        }
    else:
        previous = lifecycle.get(expected_generation)
        if previous is None:
            lifecycle[expected_generation] = {
                'dimensions': expected_dims, 'offscreen_identity': value['offscreen_identity'],
                'framebuffer_fields': _capture_plain(fields), 'allocation_pair': _capture_plain(pair),
                'source_identities': identities,
            }
        else:
            expected_lifecycle = {
                'dimensions': expected_dims, 'offscreen_identity': value['offscreen_identity'],
                'framebuffer_fields': _capture_plain(fields), 'allocation_pair': _capture_plain(pair),
                'source_identities': identities,
            }
            if expected_lifecycle != previous:
                raise ValueError(f'{name} changes the reused framebuffer identity within a generation')
    return value


def _capture_smoke_validate_stats(stats, name):
    value = _capture_smoke_exact_keys(stats, _CAPTURE_MODEL_SMOKE_STATS_KEYS, name)
    correspondences = _capture_smoke_int(value['correspondences'], f'{name}.correspondences', minimum=24)
    inliers = _capture_smoke_int(value['inliers'], f'{name}.inliers', minimum=20)
    raw = _capture_smoke_int(value['raw_correspondences'], f'{name}.raw_correspondences', minimum=0)
    retained = _capture_smoke_int(value['retained_correspondences'],
                                  f'{name}.retained_correspondences', minimum=0)
    unsupported = _capture_smoke_int(value['unsupported_correspondences'],
                                     f'{name}.unsupported_correspondences', minimum=0)
    if (retained != correspondences or inliers > correspondences or
            raw < retained + unsupported or inliers / correspondences < 0.6):
        raise ValueError(f'{name} public correspondences, inliers, or support counts fail the gates')
    median = _capture_real(value['median_reprojection_720'], f'{name}.median_reprojection_720')
    p95 = _capture_real(value['p95_reprojection_720'], f'{name}.p95_reprojection_720')
    support = _capture_real(value['spatial_support'], f'{name}.spatial_support')
    score = _capture_real(value['score'], f'{name}.score')
    if (median < 0 or p95 < median or p95 > 8 or median > 3 or
            support < 0.12 or support > 1 or score < 0):
        raise ValueError(f'{name} public validation measurements fail their fixed thresholds')


def _capture_smoke_proof_sources(renderer_proof):
    proof = _capture_proof_map(
        _capture_plain(renderer_proof), 'normalized renderer proof summary')
    source_closure = _capture_proof_map(
        proof.get('source_closure'), 'normalized renderer proof source_closure')
    if set(source_closure) != set(_CAPTURE_RENDERER_SOURCE_PATHS):
        raise ValueError('Normalized renderer proof source inventory differs from the fixed physical paths')
    root = ROOT.resolve(strict=True)
    current = {}
    for relative in _CAPTURE_RENDERER_SOURCE_PATHS:
        recorded = _capture_sha(
            source_closure.get(relative),
            f'normalized renderer proof source_closure[{relative!r}]')
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f'Current normalized renderer proof source escapes the repository: {relative}') from exc
        if not path.is_file():
            raise ValueError(f'Current normalized renderer proof source is missing: {relative}')
        actual = digest(path).upper()
        if recorded != actual:
            raise ValueError(
                f'Normalized renderer proof is stale for current source {relative}')
        current[relative] = actual
    return current


def _capture_smoke_validate_runtime_bindings(recipe, source_closure, resource_key,
                                             asset_receipt, asset_sha, proof_sha,
                                             checkpoint_sha, device, build, runtime,
                                             constructor_records, events, rows):
    if type(recipe) is not dict:
        raise ValueError('Smoke recipe must be a JSON object')
    if _capture_sha(asset_sha, 'smoke asset receipt SHA') != asset_receipt.receipt_sha256:
        raise ValueError('Smoke artifact asset receipt digest differs from the verified receipt')
    if _capture_sha(proof_sha, 'smoke renderer proof SHA') != asset_receipt.renderer_proof_sha256:
        raise ValueError('Smoke artifact renderer proof digest differs from the verified proof')
    key = _capture_sha(resource_key, 'smoke resource key')
    closure = source_closure
    if type(closure) is not dict or not closure:
        raise ValueError('Smoke source_closure must be a nonempty object')
    for path, digest_value in closure.items():
        if type(path) is not str or not path or '\\' in path or path.startswith('/'):
            raise ValueError('Smoke source_closure contains an invalid source path')
        _capture_sha(digest_value, f'source_closure[{path!r}]')
    expected_checkpoint = MODELS['gotrack_checkpoint.pt']['sha256'].upper()
    checkpoint = _capture_sha(checkpoint_sha, 'smoke checkpoint SHA')
    checkpoint_path = '.cache/model-quality/checkpoints/gotrack_checkpoint.pt'
    if checkpoint != expected_checkpoint or closure.get(checkpoint_path) != expected_checkpoint:
        raise ValueError('Smoke checkpoint does not match the frozen GoTrack source closure')
    contract = recipe.get('rendering_contract')
    if type(contract) is not dict:
        raise ValueError('Smoke recipe rendering_contract is missing')
    flags = recipe.get('render_flags')
    if type(flags) is not dict or set(flags) != {'unlit_templates', 'disable_multisampling'}:
        raise ValueError('Smoke recipe must bind only the two fixed rendering flags')
    unlit = _capture_smoke_bool(flags['unlit_templates'], 'recipe.render_flags.unlit_templates')
    if _capture_smoke_bool(flags['disable_multisampling'], 'recipe.render_flags.disable_multisampling'):
        raise ValueError('Smoke recipe must keep disable_multisampling false')
    if contract.get('textured_renderer_unlit') is not unlit:
        raise ValueError('Smoke recipe mode differs from its renderer flag')
    if contract.get('disable_multisampling') is not False:
        raise ValueError('Smoke recipe disables multisampling')
    if recipe.get('producer_device') != device or type(recipe.get('producer_device')) is not str:
        raise ValueError('Smoke recipe producer device differs from the execution evidence')
    views = contract.get('model_smoke_views')
    expected_views = [{'case_id': row['case_id'],
                       'euler_xyz_degrees': list(row['euler_xyz_degrees']),
                       'offset_extent_units': list(row['offset_extent_units'])}
                      for row in _CAPTURE_MODEL_SMOKE_VIEWS]
    _capture_smoke_exact_value(views, expected_views, 'recipe.rendering_contract.model_smoke_views')
    expected_model_contract = {
        'model_smoke_rotation': _CAPTURE_MODEL_SMOKE_ROTATION,
        'model_smoke_native_intrinsics': [list(row) for row in _CAPTURE_MODEL_SMOKE_INTRINSICS],
        'model_smoke_observation': 'public_return_v1',
        'model_smoke_array_digest': 'quality-model-smoke-array-v1',
    }
    for field, expected in expected_model_contract.items():
        _capture_smoke_exact_value(contract.get(field), expected,
                                   f'recipe.rendering_contract.{field}')
    if type(events) is not list or len(events) != 18:
        raise ValueError('Smoke evidence must contain exactly 18 ordered render events')
    if type(rows) is not list or len(rows) != 3:
        raise ValueError('Smoke evidence must contain exactly three rows')
    _capture_smoke_validate_build(build, device=device, name='smoke build')
    _capture_smoke_validate_runtime(runtime, device=device, render_events=events,
                                    name='smoke runtime_evidence')
    _capture_smoke_constructor(constructor_records, asset_receipt=asset_receipt,
                               recipe=recipe, name='constructor_records')
    bounds = _capture_plain(asset_receipt.document)['post_node_bounds_m']
    proof_sources = _capture_smoke_proof_sources(asset_receipt.renderer_proof)
    lifecycle = {}
    expected_generations = (1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6)
    event_by_index = {}
    crop_event_by_case = {}
    native_event_by_case = {}
    network_inputs_count = 0
    expected_array_shapes = {
        'native': {
            'color_norm_f32': ([720, 720, 3], '<f4'), 'depth_mm': ([720, 720], '<f4'),
            'mask': ([720, 720], '|b1'), 'model_rgb_u8': ([720, 720, 3], '|u1'),
            'cpu_rgb_u8': ([720, 720, 3], '|u1'), 'cpu_depth_m': ([720, 720], '<f8'),
            'cpu_mask': ([720, 720], '|b1'),
        },
        'crop': {
            'color_norm_f32': ([280, 280, 3], '<f4'), 'depth_mm': ([280, 280], '<f4'),
            'mask': ([280, 280], '|b1'), 'model_rgb_u8': None, 'cpu_rgb_u8': None,
            'cpu_depth_m': None, 'cpu_mask': None,
        },
    }
    for index, event in enumerate(events):
        name = f'render_events[{index}]'
        value = _capture_smoke_exact_keys(event, _CAPTURE_MODEL_SMOKE_EVENT_KEYS, name)
        if _capture_smoke_int(value['render_index'], f'{name}.render_index') != index:
            raise ValueError('Smoke render events must be consecutively ordered')
        case_ordinal = index // 6
        iteration_in_case = index % 6
        case_id = ('a', 'b', 'c')[case_ordinal]
        role = 'native' if iteration_in_case == 0 else 'crop'
        iteration = None if role == 'native' else iteration_in_case - 1
        if (value['case_id'] != case_id or type(value['case_id']) is not str or
                _capture_smoke_int(value['ordinal'], f'{name}.ordinal') != case_ordinal or
                value['role'] != role or type(value['role']) is not str or
                value['iteration'] != iteration or
                (iteration is None and value['iteration'] is not None) or
                (iteration is not None and type(value['iteration']) is not int)):
            raise ValueError(f'{name} label does not match the ordered model-only view recipe')
        if _capture_smoke_int(value['renderer_id'], f'{name}.renderer_id') != 0:
            raise ValueError(f'{name} renderer_id must match the sole constructor')
        seed = None
        if role == 'native':
            view = _CAPTURE_MODEL_SMOKE_VIEWS[case_ordinal]
            row = _capture_smoke_exact_keys(rows[case_ordinal],
                                            {'pose_id', 'seed_camera_from_object_m', 'render',
                                             'refinement', 'validation'},
                                            f'rows[{case_ordinal}]')
            if row['pose_id'] != case_id or type(row['pose_id']) is not str:
                raise ValueError(f'rows[{case_ordinal}].pose_id differs from case id')
            seed = _capture_smoke_pose(row['seed_camera_from_object_m'],
                                       f'rows[{case_ordinal}].seed_camera_from_object_m')
            expected_seed = _capture_smoke_expected_seed(
                view['euler_xyz_degrees'], view['offset_extent_units'], bounds,
                f'rows[{case_ordinal}].seed_camera_from_object_m')
            _capture_smoke_pose_close(seed, expected_seed,
                                      f'rows[{case_ordinal}].seed_camera_from_object_m')
            native_event_by_case[case_id] = (value, row, seed)
        camera_width = 720 if role == 'native' else 280
        _capture_smoke_camera(value['camera'], width=camera_width, height=camera_width,
                              name=f'{name}.camera', native_seed=seed)
        expected_types = ['rgb', 'depth', 'mask'] if role == 'native' else ['rgb', 'mask', 'depth']
        _capture_smoke_exact_value(value['requested_render_types'], expected_types,
                                   f'{name}.requested_render_types')
        _capture_smoke_bool(value['return_tensors'], f'{name}.return_tensors', role == 'crop')
        expected_background = None if role == 'native' else [0.5, 0.5, 0.5]
        _capture_smoke_exact_value(value['requested_background'], expected_background,
                                   f'{name}.requested_background')
        _capture_smoke_bool(value['mask_equals_depth_positive'],
                            f'{name}.mask_equals_depth_positive', True)
        metadata = _capture_smoke_metadata(
            value['render_policy_metadata'], event=value,
            expected_generation=expected_generations[index], current_sources=proof_sources,
            name=f'{name}.render_policy_metadata', source_closure=source_closure,
            lifecycle=lifecycle)
        arrays = _capture_smoke_exact_keys(value['arrays'],
                                            set(expected_array_shapes[role]),
                                            f'{name}.arrays')
        for field, schema in expected_array_shapes[role].items():
            if schema is None:
                if arrays[field] is not None:
                    raise ValueError(f'{name}.arrays.{field} must be null for crop observations')
            else:
                shape, dtype = schema
                _capture_smoke_descriptor(arrays[field], name=f'{name}.arrays.{field}',
                                          shape=shape, dtype=dtype)
        if role == 'native':
            if value['network_inputs'] is not None:
                raise ValueError(f'{name}.network_inputs must be null for native renders')
            native_event_by_case[case_id] = (value, row, seed)
        else:
            inputs = _capture_smoke_exact_keys(
                value['network_inputs'], {'query_rgb_bchw', 'template_rgb_bchw',
                                          'template_mask_bhw', 'template_matches_observed_render',
                                          'call_returned'}, f'{name}.network_inputs')
            _capture_smoke_descriptor(inputs['query_rgb_bchw'],
                                      name=f'{name}.network_inputs.query_rgb_bchw',
                                      shape=[1, 3, 280, 280], dtype='<f4')
            _capture_smoke_descriptor(inputs['template_rgb_bchw'],
                                      name=f'{name}.network_inputs.template_rgb_bchw',
                                      shape=[1, 3, 280, 280], dtype='<f4')
            _capture_smoke_descriptor(inputs['template_mask_bhw'],
                                      name=f'{name}.network_inputs.template_mask_bhw',
                                      shape=[1, 280, 280], dtype='|b1')
            _capture_smoke_bool(inputs['template_matches_observed_render'],
                                f'{name}.network_inputs.template_matches_observed_render', True)
            _capture_smoke_bool(inputs['call_returned'], f'{name}.network_inputs.call_returned', True)
            crop_event_by_case.setdefault(case_id, []).append(value)
            network_inputs_count += 1
        event_by_index[index] = (value, metadata)
    if network_inputs_count != 15:
        raise ValueError('Smoke evidence must attest exactly 15 successful network calls')
    _capture_smoke_validate_runtime(runtime, device=device, render_events=events,
                                    name='smoke runtime_evidence')
    for case_index, case_id in enumerate(('a', 'b', 'c')):
        row = _capture_smoke_exact_keys(rows[case_index],
                                        {'pose_id', 'seed_camera_from_object_m', 'render',
                                         'refinement', 'validation'}, f'rows[{case_index}]')
        native_event, _, seed = native_event_by_case[case_id]
        render = _capture_smoke_exact_keys(row['render'], _CAPTURE_MODEL_SMOKE_RENDER_DESCRIPTOR_KEYS,
                                           f'rows[{case_index}].render')
        descriptors = native_event['arrays']
        joined = (
            ('gpu_rgb_sha256', descriptors['model_rgb_u8']),
            ('gpu_depth_mm_sha256', descriptors['depth_mm']),
            ('gpu_mask_sha256', descriptors['mask']),
            ('cpu_rgb_sha256', descriptors['cpu_rgb_u8']),
            ('cpu_depth_m_sha256', descriptors['cpu_depth_m']),
            ('cpu_mask_sha256', descriptors['cpu_mask']),
        )
        for field, descriptor in joined:
            if _capture_sha(render[field], f'rows[{case_index}].render.{field}') != descriptor['sha256']:
                raise ValueError(f'rows[{case_index}].render.{field} does not join its native array descriptor')
        iou = _capture_real(render['mask_iou'], f'rows[{case_index}].render.mask_iou')
        depth_error = _capture_real(render['median_common_depth_error_m'],
                                    f'rows[{case_index}].render.median_common_depth_error_m')
        common = _capture_smoke_int(render['common_depth_pixel_count'],
                                    f'rows[{case_index}].render.common_depth_pixel_count',
                                    minimum=1, maximum=720*720)
        if not 0.94 <= iou <= 1.0 or not 0 <= depth_error <= 0.002:
            raise ValueError(f'rows[{case_index}].render comparison fails its fixed thresholds')
        _capture_smoke_bool(render['mask_equals_depth_positive'],
                            f'rows[{case_index}].render.mask_equals_depth_positive', True)
        if (render['native_dimensions'] != [720, 720] or
                type(render['native_dimensions']) is not list or
                any(type(size) is not int for size in render['native_dimensions'])):
            raise ValueError(f'rows[{case_index}].render.native_dimensions must be [720, 720]')
        if render['coordinate_mode'] != _CAPTURE_COORDINATE_MODE or type(render['coordinate_mode']) is not str:
            raise ValueError(f'rows[{case_index}].render.coordinate_mode mismatch')
        if render['render_policy'] != _CAPTURE_RENDER_POLICY or type(render['render_policy']) is not str:
            raise ValueError(f'rows[{case_index}].render.render_policy mismatch')
        _capture_smoke_bool(render['unlit_templates'],
                            f'rows[{case_index}].render.unlit_templates',
                            recipe['render_flags']['unlit_templates'])
        _capture_smoke_bool(render['disable_multisampling'],
                            f'rows[{case_index}].render.disable_multisampling', False)
        refinement = _capture_smoke_exact_keys(
            row['refinement'], {'candidate_present', 'iterations', 'pose_camera_from_object_m',
                                'points_object_m_sha256', 'pixels_native_sha256', 'weights_sha256'},
            f'rows[{case_index}].refinement')
        _capture_smoke_bool(refinement['candidate_present'],
                            f'rows[{case_index}].refinement.candidate_present', True)
        candidate_pose = _capture_smoke_pose(refinement['pose_camera_from_object_m'],
                                             f'rows[{case_index}].refinement.pose_camera_from_object_m')
        for field in ('points_object_m_sha256', 'pixels_native_sha256', 'weights_sha256'):
            _capture_sha(refinement[field], f'rows[{case_index}].refinement.{field}')
        iterations = refinement['iterations']
        if type(iterations) is not list or len(iterations) != 5:
            raise ValueError(f'rows[{case_index}].refinement.iterations must contain five records')
        crops = crop_event_by_case.get(case_id, [])
        if len(crops) != 5:
            raise ValueError(f'rows[{case_index}] lacks five actual crop and network events')
        for iteration_index, iteration in enumerate(iterations):
            diag = _capture_smoke_exact_keys(
                iteration, {'index', 'solver_success', 'crop_dimensions', 'query_rewarp_factor',
                            'max_sampling_map_difference_px', 'sampling_map_eligible_count',
                            'render_generation'},
                f'rows[{case_index}].refinement.iterations[{iteration_index}]')
            if _capture_smoke_int(diag['index'], f'rows[{case_index}].iterations[{iteration_index}].index') != iteration_index:
                raise ValueError('Smoke refinement diagnostics must be in iteration order')
            _capture_smoke_bool(diag['solver_success'],
                                f'rows[{case_index}].iterations[{iteration_index}].solver_success', True)
            if (diag['crop_dimensions'] != [280, 280] or
                    type(diag['crop_dimensions']) is not list or
                    any(type(size) is not int for size in diag['crop_dimensions'])):
                raise ValueError('Smoke crop diagnostics must use 280 by 280')
            if _capture_real(diag['query_rewarp_factor'],
                             f'rows[{case_index}].iterations[{iteration_index}].query_rewarp_factor') != 1.0:
                raise ValueError('Smoke query rewarp factor must be one')
            map_error = _capture_real(diag['max_sampling_map_difference_px'],
                                      f'rows[{case_index}].iterations[{iteration_index}].max_sampling_map_difference_px')
            eligible = _capture_smoke_int(diag['sampling_map_eligible_count'],
                                          f'rows[{case_index}].iterations[{iteration_index}].sampling_map_eligible_count',
                                          minimum=1, maximum=280*280)
            if map_error < 0 or map_error >= 1:
                raise ValueError('Smoke sampling-map difference must be in [0, 1)')
            crop_metadata = event_by_index[case_index * 6 + iteration_index + 1][1]
            if _capture_smoke_int(diag['render_generation'],
                                  f'rows[{case_index}].iterations[{iteration_index}].render_generation') != crop_metadata['allocation_generation']:
                raise ValueError('Smoke diagnostic generation differs from the actual crop render')
        validation = _capture_smoke_exact_keys(row['validation'], {'passed', 'reason', 'stats'},
                                               f'rows[{case_index}].validation')
        _capture_smoke_bool(validation['passed'], f'rows[{case_index}].validation.passed', True)
        if validation['reason'] is not None:
            raise ValueError(f'rows[{case_index}].validation.reason must be null on success')
        _capture_smoke_validate_stats(validation['stats'], f'rows[{case_index}].validation.stats')
    return MappingProxyType({
        'resource_key': resource_key, 'case_ids': ('a', 'b', 'c'),
        'render_count': 18, 'network_call_count': 15,
        'asset_receipt_sha256': asset_sha, 'renderer_proof_sha256': proof_sha,
        'device': device, 'mode': recipe['render_flags']['unlit_templates'],
        'rows': tuple(_capture_freeze(row) for row in rows),
        'runtime_evidence': _capture_freeze(runtime),
    })


def validate_capture_model_smoke_pending(verified_plan, resources, pending):
    """Validate saved successful worker bytes against a fresh issued smoke plan."""
    capture = _capture_require_issued_plan(verified_plan)
    asset_receipt = _read_capture_asset_receipt_validated(verified_plan, capture)
    verified_resources = _capture_validate_stage_resources(
        verified_plan, asset_receipt, resources, capture)
    if (verified_resources.stage != 'smoke' or verified_resources.required != () or
            verified_resources.output is None or verified_resources.output.descriptor is not None):
        raise capture.CaptureIntegrityError('Pending smoke requires the exact current smoke output resource')
    value = _capture_smoke_parse(pending, 'pending model-smoke report')
    allowed = {
        'schema_version', 'kind', 'execution_id', 'resource_key', 'recipe',
        'source_closure', 'asset_receipt_sha256', 'renderer_proof_sha256', 'started_at_utc',
        'worker_ended_at_utc', 'checkpoint_sha256', 'device', 'build',
        'constructor_records', 'render_events', 'rows', 'status', 'error', 'close_error',
        'renderer_closed', 'captured_rgb_read', 'evaluator_data_read', 'failure_location',
        'runtime_evidence',
    }
    _capture_smoke_exact_keys(value, allowed, 'pending model-smoke report')
    output = verified_resources.output
    if (_capture_smoke_int(value['schema_version'], 'pending.schema_version') != 1 or
            value['kind'] != 'quality-capture-model-smoke-pending-v1' or
            type(value['kind']) is not str):
        raise ValueError('Pending report schema/kind is unsupported')
    if value['status'] != 'measured_success' or type(value['status']) is not str:
        raise ValueError('Pending report is not complete measured-success evidence')
    if value['error'] is not None or value['close_error'] is not None:
        raise ValueError('Successful pending report cannot contain worker or close errors')
    _capture_smoke_bool(value['renderer_closed'], 'pending.renderer_closed', True)
    _capture_smoke_bool(value['captured_rgb_read'], 'pending.captured_rgb_read', False)
    _capture_smoke_bool(value['evaluator_data_read'], 'pending.evaluator_data_read', False)
    if value['failure_location'] is not None:
        raise ValueError('Successful pending report failure_location must be null')
    _capture_smoke_string(value['execution_id'], 'pending.execution_id')
    start = _capture_smoke_timestamp(value['started_at_utc'], 'pending.started_at_utc')
    end = _capture_smoke_timestamp(value['worker_ended_at_utc'], 'pending.worker_ended_at_utc')
    if end < start:
        raise ValueError('Pending worker ended before it started')
    if _capture_sha(value['resource_key'], 'pending.resource_key') != output.resource_key:
        raise ValueError('Pending report differs from the freshly recomputed smoke resource')
    _capture_smoke_exact_value(value['recipe'], _capture_plain(output.recipe), 'pending.recipe')
    _capture_smoke_exact_value(value['source_closure'], _capture_plain(output.source_closure),
                               'pending.source_closure')
    if value['device'] != verified_resources.device or type(value['device']) is not str:
        raise ValueError('Pending report device differs from the verified smoke resources')
    _capture_smoke_validate_runtime_bindings(
        recipe=value['recipe'], source_closure=value['source_closure'],
        resource_key=value['resource_key'], asset_receipt=asset_receipt,
        asset_sha=value['asset_receipt_sha256'], proof_sha=value['renderer_proof_sha256'],
        checkpoint_sha=value['checkpoint_sha256'], device=value['device'],
        build=value['build'], runtime=value['runtime_evidence'],
        constructor_records=value['constructor_records'], events=value['render_events'],
        rows=value['rows'])
    forbidden = {'terminal_exit_code', 'process_reaped', 'actual_unmocked_execution',
                 'synthetic_test_execution'}
    if forbidden & set(value):
        raise ValueError('Worker pending report cannot author parent terminal authority')
    return _capture_freeze(value)


def validate_capture_model_smoke_finalized(smoke_bytes, renderer_execution_bytes, *,
                                          asset_receipt, expected_recipe,
                                          expected_source_closure, expected_key):
    """Validate a finalized smoke artifact pair with the same gates as pending evidence."""
    if type(asset_receipt) is not CaptureAssetReceipt:
        raise TypeError('Finalized smoke validation requires an exact verified asset receipt')
    smoke = _capture_smoke_parse(smoke_bytes, 'finalized smoke artifact')
    execution_receipt = _capture_smoke_parse(renderer_execution_bytes,
                                              'renderer execution receipt')
    _capture_smoke_exact_keys(smoke, {
        'schema_version', 'resource_kind', 'resource_key', 'recipe', 'source_closure',
        'asset_receipt_sha256', 'renderer_proof_sha256', 'renderer_execution_receipt_sha256',
        'execution', 'rows',
    }, 'finalized smoke artifact')
    _capture_smoke_exact_keys(execution_receipt, {
        'schema_version', 'kind', 'execution_id', 'resource_key', 'recipe', 'source_closure',
        'started_at_utc', 'ended_at_utc', 'terminal_exit_code', 'actual_unmocked_execution',
        'constructor_records', 'generations', 'cleanup', 'runtime_evidence',
    }, 'renderer execution receipt')
    key = _capture_sha(expected_key, 'expected finalized smoke key')
    expected_recipe_plain = _capture_plain(expected_recipe)
    expected_closure_plain = _capture_plain(expected_source_closure)
    if (_capture_smoke_int(smoke['schema_version'], 'smoke.schema_version') != 1 or
            smoke['resource_kind'] != _CAPTURE_RESOURCE_KIND['smoke'] or
            type(smoke['resource_kind']) is not str or
            _capture_sha(smoke['resource_key'], 'smoke.resource_key') != key):
        raise ValueError('Finalized smoke artifact has the wrong schema, kind, or key')
    _capture_smoke_exact_value(smoke['recipe'], expected_recipe_plain, 'smoke.recipe')
    _capture_smoke_exact_value(smoke['source_closure'], expected_closure_plain,
                               'smoke.source_closure')
    if (_capture_smoke_int(execution_receipt['schema_version'], 'renderer_execution.schema_version') != 1 or
            execution_receipt['kind'] != 'quality-capture-model-smoke-renderer-execution-v1' or
            type(execution_receipt['kind']) is not str or
            _capture_sha(execution_receipt['resource_key'], 'renderer_execution.resource_key') != key):
        raise ValueError('Renderer execution receipt has the wrong schema, kind, or key')
    _capture_smoke_exact_value(execution_receipt['recipe'], expected_recipe_plain,
                               'renderer_execution.recipe')
    _capture_smoke_exact_value(execution_receipt['source_closure'], expected_closure_plain,
                               'renderer_execution.source_closure')
    receipt_digest = hashlib.sha256(renderer_execution_bytes).hexdigest().upper()
    if _capture_sha(smoke['renderer_execution_receipt_sha256'],
                    'smoke.renderer_execution_receipt_sha256') != receipt_digest:
        raise ValueError('Smoke artifact does not bind the exact renderer execution receipt bytes')
    if _capture_sha(smoke['asset_receipt_sha256'], 'smoke.asset_receipt_sha256') != asset_receipt.receipt_sha256:
        raise ValueError('Smoke artifact asset receipt digest differs from the authenticated asset')
    if _capture_sha(smoke['renderer_proof_sha256'], 'smoke.renderer_proof_sha256') != asset_receipt.renderer_proof_sha256:
        raise ValueError('Smoke artifact renderer proof digest differs from the authenticated proof')
    cleanup = _capture_smoke_exact_keys(execution_receipt['cleanup'],
                                        {'renderer_closed', 'process_reaped'},
                                        'renderer_execution.cleanup')
    _capture_smoke_bool(cleanup['renderer_closed'], 'renderer_execution.cleanup.renderer_closed', True)
    _capture_smoke_bool(cleanup['process_reaped'], 'renderer_execution.cleanup.process_reaped', True)
    if _capture_smoke_int(execution_receipt['terminal_exit_code'],
                          'renderer_execution.terminal_exit_code') != 0:
        raise ValueError('Renderer execution receipt does not bind a zero terminal exit code')
    _capture_smoke_bool(execution_receipt['actual_unmocked_execution'],
                        'renderer_execution.actual_unmocked_execution', True)
    execution = _capture_smoke_exact_keys(smoke['execution'], {
        'execution_id', 'started_at_utc', 'ended_at_utc', 'terminal_exit_code',
        'actual_unmocked_execution', 'inference_executed', 'captured_rgb_read',
        'evaluator_data_read', 'checkpoint_sha256', 'device', 'build', 'renderer_closed',
        'process_reaped', 'runtime_evidence',
    }, 'smoke.execution')
    if (_capture_smoke_int(execution['terminal_exit_code'], 'smoke.execution.terminal_exit_code') != 0 or
            _capture_smoke_int(execution_receipt['terminal_exit_code'],
                               'renderer_execution.terminal_exit_code') != 0):
        raise ValueError('Finalized smoke requires a zero terminal exit code')
    _capture_smoke_bool(execution['actual_unmocked_execution'],
                        'smoke.execution.actual_unmocked_execution', True)
    _capture_smoke_bool(execution['inference_executed'], 'smoke.execution.inference_executed', True)
    _capture_smoke_bool(execution['captured_rgb_read'], 'smoke.execution.captured_rgb_read', False)
    _capture_smoke_bool(execution['evaluator_data_read'], 'smoke.execution.evaluator_data_read', False)
    _capture_smoke_bool(execution['renderer_closed'], 'smoke.execution.renderer_closed', True)
    _capture_smoke_bool(execution['process_reaped'], 'smoke.execution.process_reaped', True)
    _capture_smoke_string(execution['execution_id'], 'smoke.execution.execution_id')
    if execution['execution_id'] != execution_receipt['execution_id']:
        raise ValueError('Finalized smoke execution id differs from renderer receipt')
    if (execution_receipt['started_at_utc'] != execution['started_at_utc'] or
            execution_receipt['ended_at_utc'] != execution['ended_at_utc']):
        raise ValueError('Finalized smoke times differ from renderer execution receipt')
    start = _capture_smoke_timestamp(execution['started_at_utc'], 'smoke.execution.started_at_utc')
    end = _capture_smoke_timestamp(execution['ended_at_utc'], 'smoke.execution.ended_at_utc')
    if end < start:
        raise ValueError('Finalized smoke ended before it started')
    device = execution['device']
    if device not in ('cpu', 'cuda') or type(device) is not str:
        raise ValueError('Finalized smoke device is unsupported')
    _capture_smoke_validate_build(execution['build'], device=device, name='smoke.execution.build')
    _capture_smoke_exact_value(execution['runtime_evidence'], execution_receipt['runtime_evidence'],
                               'finalized runtime_evidence')
    checkpoint_sha = _capture_sha(execution['checkpoint_sha256'], 'smoke.execution.checkpoint_sha256')
    summary = _capture_smoke_validate_runtime_bindings(
        recipe=smoke['recipe'], source_closure=smoke['source_closure'],
        resource_key=smoke['resource_key'], asset_receipt=asset_receipt,
        asset_sha=smoke['asset_receipt_sha256'], proof_sha=smoke['renderer_proof_sha256'],
        checkpoint_sha=checkpoint_sha, device=device, build=execution['build'],
        runtime=execution['runtime_evidence'],
        constructor_records=execution_receipt['constructor_records'],
        events=execution_receipt['generations'], rows=smoke['rows'])
    if (execution_receipt['execution_id'] != execution['execution_id'] or
            not start <= _capture_smoke_timestamp(execution_receipt['started_at_utc'],
                                                   'renderer_execution.started_at_utc') or
            not end <= _capture_smoke_timestamp(execution_receipt['ended_at_utc'],
                                                 'renderer_execution.ended_at_utc')):
        raise ValueError('Renderer execution receipt times do not match finalized execution')
    return _capture_freeze({
        'resource_key': key, 'execution_id': execution['execution_id'],
        'started_at_utc': execution['started_at_utc'], 'ended_at_utc': execution['ended_at_utc'],
        'terminal_exit_code': 0, 'process_reaped': True, 'rows': summary['rows'],
        'render_count': summary['render_count'], 'network_call_count': summary['network_call_count'],
    })


def _capture_flags_for_resource(resource_name, flags):
    return MappingProxyType({
        name: flags[name] for name in _CAPTURE_RESOURCE_FLAGS[resource_name]
    })


def _capture_resource_settings(resource_name):
    return MappingProxyType(dict(_CAPTURE_RESOURCE_SETTINGS[resource_name]))


def _capture_bounded_lock():
    lock_path = CACHE / 'lock.json'
    root = CACHE.resolve(strict=True)
    resolved = lock_path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError('Pinned model lock escapes the bounded model cache') from exc
    if not resolved.is_file() or resolved.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Pinned model lock must be a bounded regular file')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate pinned model-lock key: {key!r}')
            result[key] = value
        return result
    try:
        lock = json.loads(resolved.read_text(encoding='utf-8'), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError('Pinned model lock must be valid UTF-8 JSON') from exc
    if not isinstance(lock, dict):
        raise ValueError('Pinned model lock must be an object')
    return lock


def _capture_source_record(lock, source_name):
    if source_name in SOURCES:
        revision = SOURCES[source_name]
    else:
        try:
            revision = SUBMODULES[source_name][1]
        except KeyError as exc:
            raise ValueError(f'No frozen source revision for {source_name!r}') from exc
    sources = lock.get('sources')
    if not isinstance(sources, dict):
        raise ValueError('Pinned source lock table must be an object')
    entry = sources.get(source_name)
    expected_archive = _CAPTURE_SOURCE_ARCHIVE_SHA256.get(source_name)
    if (not isinstance(entry, dict) or entry.get('revision') != revision or
            type(entry.get('archive_sha256')) is not str or
            entry['archive_sha256'].upper() != expected_archive):
        raise ValueError(f'Pinned {source_name} source lock differs from its frozen revision/archive')
    archive_name = f'{source_name}-{revision}.zip'
    archive = (CACHE / 'archives' / archive_name).resolve(strict=True)
    try:
        archive.relative_to(CACHE.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f'Pinned {source_name} source archive escapes the bounded cache') from exc
    actual = digest(archive).upper()
    if actual != expected_archive:
        raise ValueError(f'Pinned {source_name} source archive bytes changed')
    return revision, archive_name, actual


def _capture_renderer_runtime_closure():
    closure = {}
    for distribution_name, (package, files) in _CAPTURE_RENDERER_RUNTIME_FILES.items():
        try:
            installed = distribution(distribution_name)
        except PackageNotFoundError as exc:
            raise ValueError(
                f'Required capture renderer distribution {distribution_name!r} is not installed') from exc
        package_root = Path(installed.locate_file(package)).resolve(strict=True)
        if not package_root.is_dir():
            raise ValueError(f'Capture renderer package root is invalid: {distribution_name!r}')
        version = installed.version
        if type(version) is not str or not version:
            raise ValueError(f'Capture renderer distribution {distribution_name!r} has no version pin')
        closure[f'runtime/{distribution_name}/version'] = hashlib.sha256(
            version.encode('utf-8')).hexdigest().upper()
        for relative in files:
            source_path = (package_root / relative).resolve(strict=True)
            try:
                source_path.relative_to(package_root)
            except ValueError as exc:
                raise ValueError(f'Capture renderer source escapes {distribution_name}: {relative}') from exc
            if not source_path.is_file():
                raise ValueError(f'Capture renderer source is missing: {distribution_name}/{relative}')
            closure[f'runtime/{distribution_name}/{relative}'] = digest(source_path).upper()
    return closure


def _capture_source_closure(resource_name, plan):
    modules = _CAPTURE_RESOURCE_MODULES[resource_name]
    closure = dict(plan.table['converter']['source_closure'])
    root = ROOT.resolve(strict=True)
    for module in modules:
        path = f'bench/{module}.py'
        source = (root / path).resolve(strict=True)
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f'Capture producer source escapes the repository: {path}') from exc
        actual = digest(source).upper()
        if path in closure and (type(closure[path]) is not str or closure[path].upper() != actual):
            raise ValueError(f'Capture converter pin differs from current producer source {path}')
        closure[path] = actual

    lock = _capture_bounded_lock()
    for source_name, source_files in _CAPTURE_RESOURCE_UPSTREAM_FILES[resource_name].items():
        revision, archive_name, archive_sha = _capture_source_record(lock, source_name)
        closure[f'.cache/model-quality/archives/{archive_name}'] = archive_sha
        if source_name == 'gotrack':
            source_root = CACHE / 'sources' / 'gotrack'
        else:
            source_root = CACHE / 'sources' / 'gotrack' / 'external' / source_name
        for relative in source_files:
            source_path = (source_root / relative).resolve(strict=True)
            try:
                source_path.relative_to(source_root.resolve(strict=True))
            except ValueError as exc:
                raise ValueError(f'Pinned upstream source path escapes {source_name}: {relative}') from exc
            if not source_path.is_file():
                raise ValueError(f'Pinned upstream source file is missing: {source_name}/{relative}')
            closure[source_path.relative_to(ROOT).as_posix()] = digest(source_path).upper()

    closure.update(_capture_renderer_runtime_closure())

    for checkpoint in _CAPTURE_RESOURCE_CHECKPOINTS[resource_name]:
        spec = MODELS.get(checkpoint)
        if spec is None or not spec.get('sha256'):
            raise ValueError(f'Capture resource checkpoint {checkpoint!r} has no frozen SHA-256 pin')
        models = lock.get('models')
        if not isinstance(models, dict):
            raise ValueError('Pinned model lock table must be an object')
        lock_entry = models.get(checkpoint)
        expected = spec['sha256'].upper()
        if (not isinstance(lock_entry, dict) or
                type(lock_entry.get('sha256')) is not str or
                lock_entry['sha256'].upper() != expected):
            raise ValueError(f'Capture checkpoint {checkpoint!r} differs from its frozen model lock')
        checkpoint_path = (CACHE / 'checkpoints' / checkpoint).resolve(strict=True)
        try:
            checkpoint_path.relative_to(CACHE.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(f'Capture checkpoint {checkpoint!r} escapes the bounded cache') from exc
        if (type(lock_entry.get('bytes')) is not int or
                lock_entry['bytes'] != checkpoint_path.stat().st_size or
                digest(checkpoint_path).upper() != expected):
            raise ValueError(f'Capture checkpoint {checkpoint!r} bytes differ from its frozen lock')
        closure[f'.cache/model-quality/checkpoints/{checkpoint}'] = expected
    return MappingProxyType(closure)


def _capture_recipe(resource_name, asset_receipt, plan, render_flags, device):
    contract = MappingProxyType({
        'render_policy': _CAPTURE_RENDER_POLICY,
        'coordinate_mode': _CAPTURE_COORDINATE_MODE,
        'frame_dimensions': (720, 720),
        'color_storage': 'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage',
        'depth_storage': 'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage',
        'mask_source': 'same-frame-renderer-depth-greater-than-zero',
        'renderer_depth_units': 'millimetres',
        'renderer_depth_to_metres_conversion_count': 1,
        'pose_translation_input_units': 'millimetres',
        'crop_dimensions': (280, 280),
        'crop_coordinate_mode': _CAPTURE_COORDINATE_MODE,
        'foundpose_color_native_dimensions': (1120, 1120),
        'foundpose_color_resize_dimensions': (280, 280),
        'foundpose_color_interpolation': 'area',
        'foundpose_resized_intrinsics_rule': 's*fx,s*fy,s*(c+0.5)-0.5',
        'low_resolution_depth_mask_rendered_separately': True,
        'disable_multisampling': render_flags.get('disable_multisampling', False),
        'textured_renderer_unlit': render_flags.get('unlit_templates', False),
        'cnos_grayscale_postprocess': render_flags.get('grayscale', False),
    })
    if resource_name == 'smoke':
        contract = MappingProxyType({
            **dict(contract),
            'model_smoke_views': tuple(MappingProxyType({
                'case_id': view['case_id'],
                'euler_xyz_degrees': tuple(view['euler_xyz_degrees']),
                'offset_extent_units': tuple(view['offset_extent_units']),
            }) for view in _CAPTURE_MODEL_SMOKE_VIEWS),
            'model_smoke_rotation': _CAPTURE_MODEL_SMOKE_ROTATION,
            'model_smoke_native_intrinsics': _CAPTURE_MODEL_SMOKE_INTRINSICS,
            'model_smoke_observation': 'public_return_v1',
            'model_smoke_array_digest': 'quality-model-smoke-array-v1',
        })
    return MappingProxyType({
        'schema': 'quality-capture-resource-recipe-v1',
        'resource': resource_name,
        'resource_kind': _CAPTURE_RESOURCE_KIND[resource_name],
        'object_id': asset_receipt.object_id,
        'object_name': asset_receipt.object_name,
        'asset_sha256': asset_receipt.asset_sha256,
        'asset_receipt_sha256': asset_receipt.receipt_sha256,
        'unit_receipt_sha256': asset_receipt.unit_receipt_sha256,
        'renderer_proof_sha256': asset_receipt.renderer_proof_sha256,
        'output_camera_sha256': plan.table['output_camera_sha256'],
        'coordinate_mode': 'integer_centers_v1',
        'rendering_contract': contract,
        'render_flags': render_flags,
        'producer_device': device,
        'settings': _capture_resource_settings(resource_name),
    })


def _capture_resource_plan(resource_name, asset_receipt, plan, render_flags, device, *, descriptor=None):
    from . import quality_capture as capture

    recipe = _capture_recipe(resource_name, asset_receipt, plan, render_flags, device)
    closure = _capture_source_closure(resource_name, plan)
    key = capture.capture_resource_key(
        asset_receipt.document, plan.table['output_camera'], recipe, closure)
    sidecar = CAPTURE_CACHE / 'capture-resources' / key / 'resource.json'
    return CaptureResourceUse(
        purpose=resource_name, resource_kind=_CAPTURE_RESOURCE_KIND[resource_name],
        resource_key=key, sidecar_path=sidecar, recipe=recipe,
        source_closure=closure, descriptor=descriptor,
    )


def _capture_same_asset(left, right):
    return (
        type(left) is CaptureAssetReceipt and
        left.receipt_sha256 == right.receipt_sha256 and
        left.asset_sha256 == right.asset_sha256 and
        left.unit_receipt_sha256 == right.unit_receipt_sha256 and
        left.object_id == right.object_id and
        left.object_name == right.object_name and
        left.source_units == right.source_units and
        left.conversion_to_metres == right.conversion_to_metres and
        left.renderer_proof_sha256 == right.renderer_proof_sha256 and
        _capture_plain(left.renderer_proof) == _capture_plain(right.renderer_proof) and
        _capture_plain(left.document) == _capture_plain(right.document)
    )


def _capture_same_descriptor(left, right):
    return (
        type(left) is type(right) and
        left.sidecar_path == right.sidecar_path and
        left.sidecar_sha256 == right.sidecar_sha256 and
        left.resource_key == right.resource_key and
        left.resource_kind == right.resource_kind and
        left.artifact_path == right.artifact_path and
        left.artifact_sha256 == right.artifact_sha256 and
        left.artifact_byte_count == right.artifact_byte_count and
        _capture_plain(left.recipe) == _capture_plain(right.recipe) and
        _capture_plain(left.source_closure) == _capture_plain(right.source_closure) and
        _capture_plain(left.descriptor) == _capture_plain(right.descriptor)
    )


def _capture_validate_use(actual, expected, capture, *, descriptor_required):
    if type(actual) is not CaptureResourceUse:
        raise TypeError('Capture stage resource uses must be exact CaptureResourceUse records')
    if (actual.purpose != expected.purpose or actual.resource_kind != expected.resource_kind or
            actual.resource_key != expected.resource_key or
            type(actual.sidecar_path) is not type(expected.sidecar_path) or
            actual.sidecar_path != expected.sidecar_path or
            _capture_plain(actual.recipe) != _capture_plain(expected.recipe) or
            _capture_plain(actual.source_closure) != _capture_plain(expected.source_closure)):
        raise capture.CaptureIntegrityError('Capture stage resource differs from the current plan recipe')
    if not descriptor_required:
        if actual.descriptor is not None:
            raise capture.CaptureIntegrityError('Capture output resource cannot carry a prerequisite descriptor')
        return expected
    if type(actual.descriptor) is not capture.StageResourceDescriptor:
        raise TypeError('Capture prerequisites require verified stage-resource descriptors')
    verified = capture.verify_stage_resource_sidecar(
        expected.sidecar_path, expected_key=expected.resource_key,
        expected_recipe=expected.recipe, expected_source_closure=expected.source_closure)
    if not _capture_same_descriptor(actual.descriptor, verified):
        raise capture.CaptureIntegrityError('Capture prerequisite descriptor changed after verification')
    if (verified.resource_kind != expected.resource_kind or
            verified.descriptor['runtime']['device'] != expected.recipe['producer_device']):
        raise capture.CaptureIntegrityError('Capture prerequisite kind/device differs from its recipe')
    return CaptureResourceUse(
        purpose=expected.purpose, resource_kind=expected.resource_kind,
        resource_key=expected.resource_key, sidecar_path=expected.sidecar_path,
        recipe=expected.recipe, source_closure=expected.source_closure,
        descriptor=verified,
    )


def _capture_validate_stage_resources(verified_plan, asset_receipt, resource_set, capture):
    if type(resource_set) is not CaptureStageResources:
        raise TypeError('Capture provenance requires an exact CaptureStageResources record')
    if not _capture_same_asset(resource_set.asset_receipt, asset_receipt):
        raise capture.CaptureIntegrityError('Capture resources are bound to a different asset receipt')
    stage = resource_set.stage
    if type(stage) is not str or stage not in _CAPTURE_STAGE_REQUIREMENTS:
        raise ValueError('Capture resources have an unsupported stage')
    device = resource_set.device
    if type(device) is not str or device not in ('cpu', 'cuda'):
        raise ValueError('Capture resources have an unsupported producer device')
    if type(resource_set.render_flags) is not MappingProxyType:
        raise TypeError('Capture resource render flags must be the issued immutable mapping')
    flags = _capture_flags(resource_set.render_flags)
    if _capture_plain(resource_set.render_flags) != _capture_plain(flags):
        raise ValueError('Capture resource render flags are not in their canonical shape')
    _capture_validate_stage_flags(stage, flags)
    required_names = _CAPTURE_STAGE_REQUIREMENTS[stage]
    if type(resource_set.required) is not tuple or len(resource_set.required) != len(required_names):
        raise capture.CaptureIntegrityError('Capture resources do not match the fixed stage prerequisites')
    verified_required = []
    for actual, name in zip(resource_set.required, required_names):
        expected = _capture_resource_plan(
            name, asset_receipt, verified_plan, _capture_flags_for_resource(name, flags), device)
        verified_use = _capture_validate_use(
            actual, expected, capture, descriptor_required=True)
        if name == 'smoke':
            _capture_validate_smoke_descriptor(
                verified_use.descriptor, asset_receipt=asset_receipt,
                expected_recipe=expected.recipe,
                expected_source_closure=expected.source_closure,
                expected_key=expected.resource_key, expected_device=device)
        verified_required.append(verified_use)

    output_name = _CAPTURE_STAGE_OUTPUT.get(stage)
    if output_name is None:
        if resource_set.output is not None:
            raise capture.CaptureIntegrityError('Capture stage does not declare an output resource')
        verified_output = None
    else:
        if resource_set.output is None:
            raise capture.CaptureIntegrityError('Capture stage is missing its deterministic output resource')
        expected_output = _capture_resource_plan(
            output_name, asset_receipt, verified_plan,
            _capture_flags_for_resource(output_name, flags), device)
        verified_output = _capture_validate_use(
            resource_set.output, expected_output, capture, descriptor_required=False)
    return CaptureStageResources(
        stage=stage, asset_receipt=asset_receipt, render_flags=flags, device=device,
        required=tuple(verified_required), output=verified_output,
    )


def capture_stage_resources(verified_plan, *, stage, render_flags=None, device='cuda'):
    """Preflight exact capture resource prerequisites before any stage effects."""
    capture = _capture_require_issued_plan(verified_plan)
    if type(stage) is not str or stage not in _CAPTURE_STAGE_REQUIREMENTS:
        raise ValueError('Unsupported capture stage resource policy')
    if type(device) is not str or device not in ('cpu', 'cuda'):
        raise ValueError('Capture stage device must be cpu or cuda')
    flags = _capture_flags(render_flags)
    _capture_validate_stage_flags(stage, flags)
    asset_receipt = _read_capture_asset_receipt_validated(verified_plan, capture)
    required = []
    for resource_name in _CAPTURE_STAGE_REQUIREMENTS[stage]:
        resource_flags = _capture_flags_for_resource(resource_name, flags)
        expected = _capture_resource_plan(resource_name, asset_receipt, verified_plan, resource_flags, device)
        descriptor = capture.verify_stage_resource_sidecar(
            expected.sidecar_path, expected_key=expected.resource_key,
            expected_recipe=expected.recipe, expected_source_closure=expected.source_closure)
        if descriptor.resource_kind != expected.resource_kind:
            raise capture.CaptureIntegrityError(
                f'Capture prerequisite {resource_name!r} has the wrong fixed resource kind')
        if descriptor.descriptor['runtime']['device'] != device:
            raise capture.CaptureIntegrityError(
                f'Capture prerequisite {resource_name!r} was built for a different device')
        if resource_name == 'smoke':
            _capture_validate_smoke_descriptor(
                descriptor, asset_receipt=asset_receipt,
                expected_recipe=expected.recipe,
                expected_source_closure=expected.source_closure,
                expected_key=expected.resource_key, expected_device=device)
        required.append(CaptureResourceUse(
            purpose=expected.purpose, resource_kind=expected.resource_kind,
            resource_key=expected.resource_key, sidecar_path=expected.sidecar_path,
            recipe=expected.recipe, source_closure=expected.source_closure,
            descriptor=descriptor,
        ))
    output_name = _CAPTURE_STAGE_OUTPUT.get(stage)
    output = None if output_name is None else _capture_resource_plan(
        output_name, asset_receipt, verified_plan,
        _capture_flags_for_resource(output_name, flags), device)
    return CaptureStageResources(
        stage=stage, asset_receipt=asset_receipt, render_flags=flags,
        device=device, required=tuple(required), output=output,
    )
SOURCES = {
    'sam2': '2b90b9f5ceec907a1c18123530e92e794ad901a4',
    'gotrack': '68f76055755f2a4a8967e13ece834f975f008bdf',
}
SUBMODULES = {
    'bop_toolkit': ('thodan/bop_toolkit', 'fa9bc5c3a1c7fc092fc0694e69270d275278032e'),
    'dinov2': ('facebookresearch/dinov2', '12592a9bdfea0a6e4ee4767cd5a8d3195c0e3509'),
}
MODELS = {
    'gotrack_checkpoint.pt': {
        'url': f'https://media.githubusercontent.com/media/facebookresearch/gotrack/{SOURCES["gotrack"]}/gotrack_checkpoint.pt',
        'sha256': 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086',
        'maximum': 1608850339,
    },
    'sam2.1_hiera_base_plus.pt': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'sha256': 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5', 'maximum': 400 * 1024**2,
    },
    'dinov2_vitl14_pretrain.pth': {
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'sha256': 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428', 'maximum': 1400*1024**2,
    },
    'FastSAM-x.pt': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt',
        'sha256': '752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76', 'maximum': 300*1024**2,
    },
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def acquire_file(path, url, maximum, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = digest(path)
        if expected and actual != expected: raise ValueError(f'Hash mismatch: {path.name}')
        return actual
    used = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    temporary = path.with_suffix(path.suffix + '.part')
    count = 0
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as f:
            while chunk := response.read(4 * 1024**2):
                count += len(chunk)
                if count > maximum or used + count > BUDGET or shutil.disk_usage(CACHE).free < 512 * 1024**2:
                    raise ValueError('8 GiB model cache budget or disk reserve reached')
                f.write(chunk)
                if count % (128 * 1024**2) < len(chunk): print(path.name, count, 'bytes', flush=True)
        actual = digest(temporary)
        if expected and actual != expected: raise ValueError(f'Hash mismatch: {path.name}')
        if count < 1024: raise ValueError('Downloaded pointer/error instead of model/source')
        temporary.replace(path)
        return actual
    finally: temporary.unlink(missing_ok=True)


def acquire(sources=True, models=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    lock_path = CACHE/'lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {'sources': {}, 'models': {}}
    if sources:
        entries = [(n, 'facebookresearch/'+n, r, CACHE/'sources'/n) for n, r in SOURCES.items()]
        entries += [(n, repo, r, CACHE/'sources/gotrack/external'/n) for n, (repo, r) in SUBMODULES.items()]
        for name, repo, revision, target in entries:
            archive = CACHE/'archives'/f'{name}-{revision}.zip'
            sha = acquire_file(archive, f'https://codeload.github.com/{repo}/zip/{revision}', 64*1024**2)
            if not (target/'README.md').exists():
                with zipfile.ZipFile(archive) as z:
                    for entry in z.infolist():
                        relative = Path(*Path(entry.filename).parts[1:])
                        output = target/relative
                        if not output.resolve().is_relative_to(target.resolve()): raise ValueError('Unsafe archive path')
                        if entry.is_dir(): output.mkdir(parents=True, exist_ok=True)
                        else:
                            output.parent.mkdir(parents=True, exist_ok=True)
                            output.write_bytes(z.read(entry))
            lock['sources'][name] = {'revision': revision, 'archive_sha256': sha}
            print(name, revision, 'ready', flush=True)
    if models:
        for name, spec in MODELS.items():
            expected = spec['sha256'] or lock['models'].get(name, {}).get('sha256')
            sha = acquire_file(CACHE/'checkpoints'/name, spec['url'], spec['maximum'], expected)
            lock['models'][name] = {'url': spec['url'], 'sha256': sha, 'bytes': (CACHE/'checkpoints'/name).stat().st_size,
                                    'hash_origin': 'upstream Git LFS' if name == 'gotrack_checkpoint.pt' else 'verified official bootstrap download; locked for all subsequent use'}
            lock_path.write_text(json.dumps(lock, indent=2))
            print(name, sha, 'verified', flush=True)
    lock_path.write_text(json.dumps(lock, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources-only', action='store_true')
    args = p.parse_args()
    acquire(models=not args.sources_only)
