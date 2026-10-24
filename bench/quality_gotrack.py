"""Inference adapter for the pinned GoTrack network (CC BY-NC 4.0).

Uses upstream DINOv2, decoder, DPT head and perspective crop construction.
Training/dataset wrappers are omitted; network state loading is strict.
Runtime integration must pass the smoke gate before benchmark publication.
"""
import hashlib
import operator
import os
from pathlib import Path
import sys
import random
import time
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import numpy as np
from .quality_assets import CACHE, MODELS, digest
from .quality_contract import (PoseCandidate, checked_pose, pose_units, canonical_pose,
                               check_capture_frame, check_model_smoke_frame)
from .vision import cv2
from .quality_trace import trace


_LEGACY_COORDINATE_MODE = 'legacy'
_CAPTURE_COORDINATE_MODE = 'integer_centers_v1'
_LEGACY_RENDER_POLICY = 'legacy_v1'
_CAPTURE_RENDER_POLICY = 'capture_zero_sample_v1'
_GOTRACK_INTEGER_INFERENCE_PROFILE = 'gotrack_integer_legacy_synthetic_v1'


@dataclass(frozen=True, slots=True)
class ModelSmokeIterationDiagnostic:
    index: int
    solver_success: bool
    crop_dimensions: tuple[int, int]
    query_rewarp_factor: int
    max_sampling_map_difference_px: float | None
    sampling_map_eligible_count: int
    render_generation: int

    def as_dict(self):
        return {
            'index': self.index,
            'solver_success': self.solver_success,
            'crop_dimensions': list(self.crop_dimensions),
            'query_rewarp_factor': self.query_rewarp_factor,
            'max_sampling_map_difference_px': self.max_sampling_map_difference_px,
            'sampling_map_eligible_count': self.sampling_map_eligible_count,
            'render_generation': self.render_generation,
        }


@dataclass(frozen=True, slots=True)
class ModelSmokeRefinementResult:
    candidate: PoseCandidate | None
    iterations: tuple[ModelSmokeIterationDiagnostic, ...]
    case_id: str
    ordinal: int

    @property
    def metadata(self):
        return MappingProxyType({
            'model_only': True,
            'case_id': self.case_id,
            'ordinal': self.ordinal,
            'timestamp_s': None,
        })


def _freeze_render_policy_value(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_render_policy_value(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_render_policy_value(item) for item in value)
    return value


def _render_policy_source_identity(module):
    module_name = getattr(module, '__name__', None)
    source_path = getattr(module, '__file__', None)
    result = {'module': module_name, 'file': None, 'sha256': None}
    if source_path is None:
        return result
    path = Path(source_path)
    result['file'] = str(path.resolve())
    try:
        size = path.stat().st_size
        if size < 0 or size > 8 * 1024 * 1024:
            raise ValueError('renderer source identity exceeds its bounded size')
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f'cannot read renderer source identity: {path.name}') from exc
    if len(data) != size:
        raise ValueError('renderer source changed while its identity was read')
    result['sha256'] = hashlib.sha256(data).hexdigest().upper()
    return result


def _capture_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _capture_erode3(mask):
    """Full 3x3 AND erosion with an unsupported exterior boundary."""
    source = np.asarray(mask)
    if source.ndim != 2 or source.dtype != np.bool_:
        raise ValueError('Capture support erosion requires a 2D bool map')
    height, width = source.shape
    padded = np.pad(source, ((1, 1), (1, 1)), mode='constant', constant_values=False)
    result = np.ones((height, width), dtype=np.bool_)
    for y in range(3):
        for x in range(3):
            result &= padded[y:y + height, x:x + width]
    return result


def _capture_transform(value):
    transform = _capture_numpy(value).astype(np.float64, copy=False)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0., atol=1e-6)):
        raise ValueError('Capture crop transform must be a finite homogeneous 4x4 matrix')
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-6) or
            abs(float(np.linalg.det(rotation)) - 1.) > 1e-6):
        raise ValueError('Capture crop transform rotation must be proper')
    if not np.allclose(transform[:3, 3], 0., rtol=0., atol=1e-6):
        raise ValueError('Capture crop and native cameras must share the optical center')
    return transform


def _capture_native_uv(points, crop_k, rotation, native_k):
    """Map finite integer-center crop pixels back to native integer-center UV."""
    pixels = np.asarray(points, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('Capture pixels must be Nx2')
    output = np.full((len(pixels), 2), np.nan, dtype=np.float64)
    finite = np.isfinite(pixels).all(axis=1)
    ids = np.flatnonzero(finite)
    if not ids.size:
        return output
    homogeneous = np.column_stack((pixels[ids], np.ones(len(ids), dtype=np.float64)))
    crop_rays = homogeneous @ np.linalg.inv(crop_k).T
    native_rays = crop_rays @ rotation
    native_h = native_rays @ native_k.T
    z = native_h[:, 2]
    front = np.isfinite(native_h).all(axis=1) & np.isfinite(crop_rays).all(axis=1) & (crop_rays[:, 2] > 0.) & (z > 0.)
    if np.any(front):
        selected = ids[front]
        output[selected, 0] = native_h[front, 0] / z[front]
        output[selected, 1] = native_h[front, 1] / z[front]
    return output


def _integer_profile_correspondences(depth, crop_k, crop_from_orig, pose_mm,
                                     flow, weight, rendered_mask, observed_mask,
                                     native_k):
    """Build aligned integer-grid PnP records while preserving source crop tensors."""
    from .quality_capture import integer_depth_points

    def crop_map(value, name, *, dtype=None):
        array = _capture_numpy(value)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (280, 280):
            raise ValueError(f'{name} must be a 280x280 single-channel crop')
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    depth_value = _capture_numpy(depth)
    while depth_value.ndim > 2 and depth_value.shape[0] == 1:
        depth_value = depth_value[0]
    if depth_value.dtype.kind not in 'iuf' or depth_value.dtype.kind == 'b':
        raise ValueError('Rendered depth must be a real numeric millimetre map')
    depth_map = crop_map(depth_value, 'Rendered depth', dtype=np.float64)
    flow_value = _capture_numpy(flow)
    if flow_value.dtype.kind not in 'iuf' or flow_value.dtype.kind == 'b':
        raise ValueError('Predicted flow must be real numeric crop pixels')
    flow_map = flow_value.astype(np.float64, copy=False)
    if flow_map.shape != (280, 280, 2):
        raise ValueError('Predicted flow must be a 280x280x2 crop map')
    weight_value = _capture_numpy(weight)
    if weight_value.dtype.kind not in 'iuf' or weight_value.dtype.kind == 'b':
        raise ValueError('Predicted confidence must be real numeric weights')
    confidence = crop_map(weight_value, 'Predicted confidence', dtype=np.float64)
    rendered = crop_map(rendered_mask, 'Rendered template mask') > .5
    observed = crop_map(observed_mask, 'Observed crop mask', dtype=np.float64)

    camera = _capture_numpy(crop_k).astype(np.float64, copy=False)
    if (camera.shape != (3, 3) or not np.isfinite(camera).all() or
            camera[0, 0] <= 0. or camera[1, 1] <= 0. or
            not np.allclose(camera[2], [0., 0., 1.], rtol=0., atol=1e-8) or
            abs(float(np.linalg.det(camera))) <= 1e-12):
        raise ValueError('Ordinary crop camera K must be finite, nonsingular and have positive focal lengths')
    transform = _capture_transform(crop_from_orig)
    native = _capture_numpy(native_k).astype(np.float64, copy=False)
    if (native.shape != (3, 3) or not np.isfinite(native).all() or
            native[0, 0] <= 0. or native[1, 1] <= 0. or
            not np.allclose(native[0, 1], 0., rtol=0., atol=1e-12) or
            not np.allclose(native[2], [0., 0., 1.], rtol=0., atol=1e-12)):
        raise ValueError('Effective native camera K must be finite, zero-skew and calibrated')

    integer_pixels, crop_xyz_mm = integer_depth_points(depth_map, camera)
    crop_pose = transform @ _capture_numpy(pose_mm).astype(np.float64, copy=False)
    object_xyz_mm = (crop_xyz_mm - crop_pose[:3, 3]) @ crop_pose[:3, :3]
    target = integer_pixels.astype(np.float64) + flow_map
    finite_target = np.isfinite(target).all(axis=2)
    in_bounds = (finite_target & (target[..., 0] >= 0.) & (target[..., 0] < 280.) &
                 (target[..., 1] >= 0.) & (target[..., 1] < 280.))
    # Invalid/unbounded coordinates never reach floor or integer indexing.
    safe_target = np.zeros_like(target)
    safe_target[in_bounds] = target[in_bounds]
    target_x = np.floor(safe_target[..., 0]).astype(np.int64)
    target_y = np.floor(safe_target[..., 1]).astype(np.int64)
    valid = rendered & np.isfinite(depth_map) & (depth_map > 0.)
    valid &= np.isfinite(crop_xyz_mm).all(axis=2) & np.isfinite(object_xyz_mm).all(axis=2)
    valid &= np.isfinite(flow_map).all(axis=2) & np.isfinite(confidence) & (confidence > .3)
    valid &= in_bounds & (observed > .5)
    valid &= observed[target_y, target_x] > .5

    native_uv = _capture_native_uv(target.reshape((-1, 2)), camera,
                                   transform[:3, :3], native).reshape((280, 280, 2))
    valid &= np.isfinite(native_uv).all(axis=2)
    return (crop_pose, object_xyz_mm, target, native_uv, confidence, valid,
            integer_pixels, crop_xyz_mm)


def _capture_supported(valid, uv, helper):
    """Call the reviewed four-neighbor gate only on finite in-map points."""
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Capture UV must be Nx2')
    height, width = valid.shape
    result = np.zeros(len(points), dtype=np.bool_)
    finite = np.isfinite(points).all(axis=1)
    bounded = np.flatnonzero(finite & (points[:, 0] >= 0.) & (points[:, 0] < width - 1) &
                             (points[:, 1] >= 0.) & (points[:, 1] < height - 1))
    if bounded.size:
        result[bounded] = helper(valid, points[bounded])
    return result


def _capture_actual_sampling_map(src_camera, dst_camera, integer_grid):
    """Reproduce the pinned im_util.warp_image camera/map method chain.

    The negative-depth sentinel is applied before the float32 conversion, just
    as it is in warp_image before cv2.remap receives map_x/map_y.
    """
    pixels = np.asarray(integer_grid, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('Capture integer grid must be Nx2')
    try:
        destination_eye = np.asarray(dst_camera.window_to_eye(pixels), dtype=np.float64)
        world_points = np.asarray(dst_camera.eye_to_world(destination_eye), dtype=np.float64)
        source_eye = np.asarray(src_camera.world_to_eye(world_points), dtype=np.float64)
        source_window = np.asarray(src_camera.eye_to_window(source_eye), dtype=np.float64)
    except Exception as exc:
        raise ValueError('Capture returned-camera sampling map could not be evaluated') from exc
    count = len(pixels)
    if (destination_eye.shape != (count, 3) or world_points.shape != (count, 3) or
            source_eye.shape != (count, 3) or source_window.shape != (count, 2)):
        raise ValueError('Capture returned-camera sampling map has malformed dimensions')
    sentinel_window = np.array(source_window, dtype=np.float64, copy=True)
    sentinel_window[source_eye[:, 2] < 0.] = -1.
    return (destination_eye, world_points, source_eye,
            sentinel_window.astype(np.float32))


def upstream_path():
    source = CACHE/'sources/gotrack'
    for path in (source, source/'external/bop_toolkit', source/'external/dinov2'):
        if not path.exists(): raise RuntimeError('Run bench.quality_assets first')
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    os.environ.setdefault('XFORMERS_DISABLED', '1')


def load_network(device):
    upstream_path()
    import torch
    import dinov2.hub.backbones as backbones
    from utils.dinov2_util import DinoFeatureExtractor
    from model.blocks.decoder import Decoder
    from model.blocks.config import DecoderOpts
    from model.heads.dpt.model import DPTHead
    from model.heads.dpt.config import DPTHeadOpts
    checkpoint = CACHE/'checkpoints/gotrack_checkpoint.pt'
    if digest(checkpoint) != MODELS[checkpoint.name]['sha256']: raise ValueError('Untrusted GoTrack checkpoint')
    # The checkpoint carries the backbone. Avoid two upstream pretrained
    # downloads and an unpinned torch.hub fetch from the repository default branch.
    original_backbone = backbones.dinov2_vits14_reg
    original_hub_load = torch.hub.load
    backbones.dinov2_vits14_reg = lambda **kw: original_backbone(pretrained=False)
    torch.hub.load = lambda repo, name, **kw: backbones.__dict__[name](pretrained=False)
    try:
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DinoFeatureExtractor('dinov2_vits14-reg')
                self.decoder = Decoder(DecoderOpts())
                self.pose_head = DPTHead(DPTHeadOpts())

            def forward(self, query, template, mask):
                features = self.backbone(torch.cat((query, template), 0))['feature_maps']
                q, t = features.chunk(2, 0)
                _, reference = self.decoder(features_query=q, features_reference=t, crop_size=(280, 280))
                flows, confidences = self.pose_head(reference, (280, 280))
                return flows*mask[:, None], confidences*mask
        network = Network()
    finally:
        backbones.dinov2_vits14_reg = original_backbone
        torch.hub.load = original_hub_load
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    prefix = 'models.1.'
    selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not selected: raise ValueError('Expected upstream models.1 checkpoint namespace')
    network.load_state_dict(selected, strict=True)
    del state, selected
    return network.eval().to(device)


class TexturedRenderer:
    """Upstream renderer API with metre GLBs and explicit mm camera boundary.

    Retains every GLB node transform, texture and UV. Model depth is synthetic.
    Lighting follows the upstream camera spotlight/ambient defaults.
    """
    def __init__(self, path, obj_id, unlit=False, disable_multisampling=False,
                 coordinate_mode=_LEGACY_COORDINATE_MODE, *,
                 render_policy=_LEGACY_RENDER_POLICY):
        if (type(render_policy) is not str or
                render_policy not in (_LEGACY_RENDER_POLICY, _CAPTURE_RENDER_POLICY)):
            raise ValueError('Unknown TexturedRenderer render_policy')
        if (type(coordinate_mode) is not str or
                coordinate_mode not in (_LEGACY_COORDINATE_MODE, _CAPTURE_COORDINATE_MODE)):
            raise ValueError('Unknown TexturedRenderer coordinate_mode')
        if render_policy == _CAPTURE_RENDER_POLICY:
            if coordinate_mode != _CAPTURE_COORDINATE_MODE:
                raise ValueError('capture_zero_sample_v1 requires integer_centers_v1 coordinate_mode')
            if type(disable_multisampling) is not bool or disable_multisampling is not False:
                raise ValueError('capture_zero_sample_v1 requires disable_multisampling=False as a bool')
        upstream_path()
        import trimesh
        import pyrender
        self.pyrender = pyrender; self.obj_id = obj_id; self.offscreen = None; self.unlit = bool(unlit)
        self.coordinate_mode = coordinate_mode
        self.disable_multisampling = bool(disable_multisampling)
        self._render_policy = render_policy
        self._capture_allocation_generation = 0
        self._capture_verified_offscreen = None
        self._capture_verified_internal_renderer = None
        self._capture_verified_framebuffers = None
        self._capture_allocation_pair = None
        self._capture_dimensions = None
        self._capture_metadata = None
        self._capture_scene_poisoned = False
        self.asset_sha256 = digest(Path(path))
        self.scene = pyrender.Scene(bg_color=[.5, .5, .5, 0], ambient_light=[.02, .02, .02, 1.])
        loaded = trimesh.load(path, force='scene', process=False)
        self.vertices_m = []
        for node in loaded.graph.nodes_geometry:
            transform, geometry = loaded.graph[node]
            part = loaded.geometry[geometry].copy(); part.apply_transform(transform)
            self.vertices_m.append(part.vertices.copy())
            self.scene.add(pyrender.Mesh.from_trimesh(part, smooth=False))
        self.vertices_m = np.concatenate(self.vertices_m)
        if not np.isfinite(self.vertices_m).all() or np.ptp(self.vertices_m, axis=0).max() > 2:
            raise ValueError('Expected a metric everyday-object GLB')

    @property
    def render_policy(self):
        return getattr(self, '_render_policy', _LEGACY_RENDER_POLICY)

    @property
    def render_policy_metadata(self):
        if (self.render_policy != _CAPTURE_RENDER_POLICY or
                self.coordinate_mode != _CAPTURE_COORDINATE_MODE or
                type(self.disable_multisampling) is not bool or
                self.disable_multisampling is not False or
                self._capture_metadata is None or
                self.offscreen is None or
                self._capture_verified_offscreen is not self.offscreen or
                self._capture_verified_internal_renderer is not getattr(self.offscreen, '_renderer', None) or
                self._capture_verified_framebuffers != (
                    self._capture_metadata.get('framebuffer_fields', {}).get('multisample_draw_fbo'),
                    self._capture_metadata.get('framebuffer_fields', {}).get('single_sample_read_fbo'))):
            return None
        try:
            from .quality_render_stability import _capture_dimensions as validate_dimensions
            current_renderer = self.offscreen._renderer
            raw_framebuffers = (current_renderer._main_fb_ms, current_renderer._main_fb)
            if any(isinstance(value, (bool, np.bool_)) for value in raw_framebuffers):
                return None
            current_framebuffers = tuple(operator.index(value) for value in raw_framebuffers)
            current_dimensions = validate_dimensions(current_renderer._main_fb_dims)
            viewport_dimensions = (
                self.offscreen.viewport_width, self.offscreen.viewport_height)
            recorded_dimensions = validate_dimensions(self._capture_metadata['dimensions'])
        except (AttributeError, ImportError, KeyError, TypeError, ValueError, OverflowError):
            return None
        if (current_framebuffers != self._capture_verified_framebuffers or
                current_dimensions != self._capture_dimensions or
                current_dimensions != recorded_dimensions or
                viewport_dimensions != recorded_dimensions):
            return None
        return _freeze_render_policy_value(self._capture_metadata)

    def _validate_capture_render_configuration(self):
        self._capture_metadata = None
        if self._render_policy != _CAPTURE_RENDER_POLICY:
            raise ValueError('Capture rendering requires render_policy=capture_zero_sample_v1')
        if self.coordinate_mode != _CAPTURE_COORDINATE_MODE:
            raise ValueError('capture_zero_sample_v1 requires integer_centers_v1 coordinate_mode')
        if type(self.disable_multisampling) is not bool or self.disable_multisampling is not False:
            raise ValueError('capture_zero_sample_v1 requires disable_multisampling=False as a bool')

    def _invalidate_capture_evidence(self):
        self._capture_metadata = None
        self._capture_verified_offscreen = None
        self._capture_verified_internal_renderer = None
        self._capture_verified_framebuffers = None
        self._capture_allocation_pair = None
        self._capture_dimensions = None

    def render_object_model(self, obj_id, camera_model_c2w, render_types=None, return_tensors=False, background=None, **kwargs):
        capture = self.render_policy == _CAPTURE_RENDER_POLICY
        if capture:
            self._validate_capture_render_configuration()
        from .quality_render_stability import _render_lock
        with _render_lock:
            if capture:
                self._validate_capture_render_configuration()
                if self._capture_scene_poisoned:
                    raise RuntimeError('Capture renderer scene integrity is unknown; create a fresh renderer')
            return self._render_object_model_locked(
                obj_id, camera_model_c2w, render_types=render_types,
                return_tensors=return_tensors, background=background, **kwargs)

    def _render_object_model_locked(self, obj_id, camera_model_c2w, render_types=None,
                                    return_tensors=False, background=None, **kwargs):
        from utils.renderer_base import RenderType
        if obj_id != self.obj_id: raise ValueError('Wrong object asset')
        if self.render_policy == _CAPTURE_RENDER_POLICY:
            return self._render_capture_locked(obj_id, camera_model_c2w, RenderType,
                                               return_tensors=return_tensors)
        p = self.pyrender; camera = camera_model_c2w
        size = (camera.width, camera.height)
        if self.offscreen is None or size != (self.offscreen.viewport_width, self.offscreen.viewport_height):
            if self.offscreen: self.offscreen.delete()
            self.offscreen = p.OffscreenRenderer(*size)
            # Request stable color conversion while diagnosing tiny render
            # differences. This does not itself prove repeatability; separate
            # real inference prefix tests must verify that requirement.
            self.offscreen._platform.make_current()
            from OpenGL.GL import glDisable, GL_DITHER
            glDisable(GL_DITHER)
        camera_pose = np.array(camera.T_world_from_eye, copy=True)
        camera_pose[:3, 3] *= .001
        camera_pose = camera_pose@np.diag([1., -1., -1., 1.])
        principal = camera.c
        if self.coordinate_mode == _CAPTURE_COORDINATE_MODE:
            principal = (camera.c[0] + .5, camera.c[1] + .5)
        node = self.scene.add(p.IntrinsicsCamera(*camera.f, *principal, znear=.01, zfar=30.), pose=camera_pose)
        light = self.scene.add(p.SpotLight(color=np.ones(3), intensity=2.4, innerConeAngle=np.pi/16, outerConeAngle=np.pi/6), pose=camera_pose)
        try:
            from contextlib import nullcontext
            from .quality_render_stability import without_multisampling
            context = without_multisampling(p.renderer) if self.disable_multisampling else nullcontext()
            with context:
                color, depth = self.offscreen.render(self.scene, flags=p.RenderFlags.FLAT if self.unlit else p.RenderFlags.NONE)
        finally: self.scene.remove_node(node); self.scene.remove_node(light)
        trace('render', camera_pose=camera_pose, color=color, depth=depth)
        result = {RenderType.COLOR: color.astype(np.float32)/255,
                  RenderType.DEPTH: depth.astype(np.float32)*1000, RenderType.MASK: depth > 0}
        if return_tensors:
            import torch
            result = {k: torch.from_numpy(v.copy()) for k, v in result.items()}
        return result

    def _render_capture_locked(self, obj_id, camera, RenderType, *, return_tensors=False):
        self._capture_metadata = None
        self._validate_capture_render_configuration()
        if self._capture_scene_poisoned:
            raise RuntimeError('Capture renderer scene integrity is unknown; create a fresh renderer')
        width, height = camera.width, camera.height
        if (type(width) is not int or type(height) is not int or
                width <= 0 or height <= 0 or width > 1120 or height > 1120):
            raise ValueError('Capture viewport must use positive integer dimensions no larger than 1120')
        size = (width, height)
        p = self.pyrender
        if obj_id != self.obj_id:
            raise ValueError('Wrong object asset')

        if (self.offscreen is not None and
                self._capture_verified_offscreen is not self.offscreen):
            current_offscreen = self.offscreen
            previous_offscreen = self._capture_verified_offscreen
            self.offscreen = None
            self._invalidate_capture_evidence()
            close_errors = []
            seen_offscreens = []
            for candidate in (current_offscreen, previous_offscreen):
                if candidate is None or any(candidate is prior for prior in seen_offscreens):
                    continue
                seen_offscreens.append(candidate)
                try:
                    self._delete_capture_offscreen(candidate)
                except BaseException as exc:
                    close_errors.append(exc)
            if close_errors:
                raise RuntimeError(
                    f'Externally replaced capture renderer could not be closed: {close_errors[0]}') from close_errors[0]
            raise ValueError('Capture OffscreenRenderer identity changed outside the verified lifecycle')
        if self.offscreen is None and self._capture_verified_offscreen is not None:
            previous_offscreen = self._capture_verified_offscreen
            self._invalidate_capture_evidence()
            try:
                self._delete_capture_offscreen(previous_offscreen)
            except BaseException as exc:
                raise RuntimeError('Externally cleared capture renderer could not be closed') from exc
            raise ValueError('Capture OffscreenRenderer was cleared outside the verified lifecycle')

        if self.offscreen is not None:
            current_size = (getattr(self.offscreen, 'viewport_width', None),
                            getattr(self.offscreen, 'viewport_height', None))
            if current_size != size:
                self._dispose_capture_offscreen()
        fresh_offscreen = self.offscreen is None
        if fresh_offscreen and self._capture_verified_offscreen is not None:
            self._invalidate_capture_evidence()
        self._capture_metadata = None

        from .quality_render_stability import (
            capture_storage_scope, validate_capture_allocation_pairs,
            verify_capture_framebuffer,
        )
        from OpenGL import GL

        allocation_calls = []
        scene_nodes = []
        primary_error = None
        cleanup_errors = []
        result = None
        metadata = None
        pair = None
        next_generation = self._capture_allocation_generation
        internal_renderer = None
        framebuffer_identity = None
        try:
            with capture_storage_scope(
                    p.renderer, GL, dimensions=size,
                    allocation_calls=allocation_calls):
                if self.offscreen is None:
                    self.offscreen = p.OffscreenRenderer(width, height)
                    self.offscreen._platform.make_current()
                    GL.glDisable(GL.GL_DITHER)

                camera_pose = np.array(camera.T_world_from_eye, copy=True)
                camera_pose[:3, 3] *= .001
                camera_pose = camera_pose @ np.diag([1., -1., -1., 1.])
                principal = (camera.c[0] + .5, camera.c[1] + .5)
                node = self.scene.add(
                    p.IntrinsicsCamera(*camera.f, *principal, znear=.01, zfar=30.),
                    pose=camera_pose)
                scene_nodes.append(node)
                light = self.scene.add(
                    p.SpotLight(color=np.ones(3), intensity=2.4,
                                innerConeAngle=np.pi/16, outerConeAngle=np.pi/6),
                    pose=camera_pose)
                scene_nodes.append(light)
                color, depth = self.offscreen.render(
                    self.scene,
                    flags=p.RenderFlags.FLAT if self.unlit else p.RenderFlags.NONE)
                verified = verify_capture_framebuffer(
                    self.offscreen, GL, expected_dimensions=size)
                internal_renderer = getattr(self.offscreen, '_renderer', None)
                framebuffer_identity = (
                    verified['framebuffer_fields']['multisample_draw_fbo'],
                    verified['framebuffer_fields']['single_sample_read_fbo'])

                if allocation_calls:
                    pair = validate_capture_allocation_pairs(
                        allocation_calls, dimensions=size)
                    next_generation += 1
                else:
                    if fresh_offscreen:
                        raise ValueError('Fresh capture framebuffer did not allocate a logged color/depth pair')
                    if (self._capture_verified_offscreen is not self.offscreen or
                            self._capture_verified_internal_renderer is not internal_renderer or
                            self._capture_verified_framebuffers != framebuffer_identity or
                            self._capture_dimensions != size or
                            self._capture_allocation_pair is None):
                        raise ValueError('Reused capture framebuffer has no matching verified allocation identity')
                    pair = self._capture_allocation_pair

                import pyrender.offscreen as offscreen_module
                from . import quality_render_stability as stability_module
                renderer_module = getattr(p, 'renderer', None)
                if renderer_module is None:
                    raise ValueError('PyRender renderer module identity is unavailable')
                metadata = {
                    'schema_version': 1,
                    'render_policy': _CAPTURE_RENDER_POLICY,
                    'coordinate_mode': _CAPTURE_COORDINATE_MODE,
                    'depth_units': 'millimetres',
                    'dimensions': size,
                    'allocation_generation': next_generation,
                    'offscreen_identity': id(self.offscreen),
                    'framebuffer_fields': dict(verified['framebuffer_fields']),
                    'allocation_pair': dict(pair),
                    'allocation_calls': [dict(row) for row in allocation_calls],
                    'framebuffer_complete': verified['framebuffer_complete'],
                    'gl_samples': verified['gl_samples'],
                    'gl_sample_buffers': verified['gl_sample_buffers'],
                    'framebuffer_bindings_restored': verified['framebuffer_bindings_restored'],
                    'current_context_released': verified['current_context_released'],
                    'dimension_match': verified['dimension_match'],
                    'color_storage_policy': 'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage',
                    'depth_storage_policy': 'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage',
                    'source_identities': {
                        'quality_gotrack': _render_policy_source_identity(sys.modules[__name__]),
                        'quality_render_stability': _render_policy_source_identity(stability_module),
                        'pyrender_renderer': _render_policy_source_identity(renderer_module),
                        'pyrender_offscreen': _render_policy_source_identity(offscreen_module),
                    },
                }

            failed_scene_nodes = []
            for scene_node in reversed(scene_nodes):
                try:
                    self.scene.remove_node(scene_node)
                except BaseException as exc:
                    cleanup_errors.append(('remove_capture_scene_node', exc))
                    failed_scene_nodes.append(scene_node)
            scene_nodes[:] = failed_scene_nodes
            if failed_scene_nodes:
                self._capture_scene_poisoned = True
            if cleanup_errors:
                detail = '; '.join(f'{name}: {type(exc).__name__}: {exc}'
                                   for name, exc in cleanup_errors)
                raise RuntimeError(f'Capture scene-node cleanup failed: {detail}') from cleanup_errors[0][1]

            result = {RenderType.COLOR: color.astype(np.float32) / 255,
                      RenderType.DEPTH: depth.astype(np.float32) * 1000,
                      RenderType.MASK: depth > 0}
            if return_tensors:
                import torch
                result = {key: torch.from_numpy(value.copy())
                          for key, value in result.items()}
            trace('render', camera_pose=camera_pose, color=color, depth=depth)
        except BaseException as exc:
            primary_error = exc
        finally:
            failed_scene_nodes = []
            for scene_node in reversed(scene_nodes):
                try:
                    self.scene.remove_node(scene_node)
                except BaseException as exc:
                    cleanup_errors.append(('remove_capture_scene_node', exc))
                    failed_scene_nodes.append(scene_node)
            if failed_scene_nodes:
                self._capture_scene_poisoned = True
            scene_nodes[:] = failed_scene_nodes
            if primary_error is not None or cleanup_errors:
                self._invalidate_capture_evidence()
                try:
                    self._dispose_capture_offscreen()
                except BaseException as exc:
                    cleanup_errors.append(('close_capture_offscreen', exc))

        if primary_error is not None:
            if cleanup_errors:
                detail = '; '.join(f'{name}: {type(exc).__name__}: {exc}'
                                   for name, exc in cleanup_errors)
                raise RuntimeError(
                    f'Capture render failed ({type(primary_error).__name__}: {primary_error}); '
                    f'cleanup also failed ({detail})') from primary_error
            raise primary_error
        if cleanup_errors:
            detail = '; '.join(f'{name}: {type(exc).__name__}: {exc}'
                               for name, exc in cleanup_errors)
            raise RuntimeError(f'Capture render cleanup failed: {detail}') from cleanup_errors[0][1]
        self._capture_allocation_generation = next_generation
        self._capture_verified_offscreen = self.offscreen
        self._capture_verified_internal_renderer = internal_renderer
        self._capture_verified_framebuffers = framebuffer_identity
        self._capture_allocation_pair = pair
        self._capture_dimensions = size
        self._capture_metadata = metadata
        return result

    def _dispose_capture_offscreen(self):
        offscreen = self.offscreen
        self.offscreen = None
        self._invalidate_capture_evidence()
        if offscreen is not None:
            self._delete_capture_offscreen(offscreen)

    @staticmethod
    def _delete_capture_offscreen(offscreen):
        platform = getattr(offscreen, '_platform', None)
        cleanup_errors = []
        if platform is None:
            cleanup_errors.append(('release_before_delete',
                                   RuntimeError('Capture offscreen platform identity is unavailable')))
        else:
            try:
                platform.make_uncurrent()
            except BaseException as exc:
                cleanup_errors.append(('release_before_delete', exc))
        try:
            offscreen.delete()
        except BaseException as exc:
            cleanup_errors.append(('delete_offscreen', exc))
        if platform is not None:
            try:
                platform.make_uncurrent()
            except BaseException as exc:
                cleanup_errors.append(('release_after_delete', exc))
        if cleanup_errors:
            detail = '; '.join(f'{name}: {type(exc).__name__}: {exc}'
                               for name, exc in cleanup_errors)
            raise RuntimeError(f'Capture offscreen disposal failed: {detail}') from cleanup_errors[0][1]

    def close(self):
        from .quality_render_stability import _render_lock
        with _render_lock:
            if self.render_policy == _CAPTURE_RENDER_POLICY:
                offscreen = self.offscreen
                previously_verified = self._capture_verified_offscreen
                self.offscreen = None
                self._invalidate_capture_evidence()
                close_errors = []
                seen = []
                for candidate in (offscreen, previously_verified):
                    if candidate is None or any(candidate is prior for prior in seen):
                        continue
                    seen.append(candidate)
                    try:
                        self._delete_capture_offscreen(candidate)
                    except BaseException as exc:
                        close_errors.append(exc)
                if close_errors:
                    raise RuntimeError(
                        f'Capture renderer close failed: {close_errors[0]}') from close_errors[0]
            elif self.offscreen:
                self.offscreen.delete(); self.offscreen = None


class GoTrackRefiner:
    def __init__(self, network, renderer, obj_id, device, pnp_use_extrinsic_guess=True,
                 diagnostic_capture_callback=None, chronological_capture_callback=None,
                 *, inference_geometry='legacy', inference_observer_capability=None):
        if type(pnp_use_extrinsic_guess) is not bool:
            raise TypeError('pnp_use_extrinsic_guess must be a bool')
        if diagnostic_capture_callback is not None and not callable(diagnostic_capture_callback):
            raise TypeError('diagnostic_capture_callback must be callable or None')
        if chronological_capture_callback is not None and not callable(chronological_capture_callback):
            raise TypeError('chronological_capture_callback must be callable or None')
        if inference_geometry not in ('legacy', _GOTRACK_INTEGER_INFERENCE_PROFILE):
            raise ValueError('Unknown GoTrack inference geometry')
        if inference_observer_capability is not None:
            from .quality_gotrack_probe import InferenceObserverCapability
            from .quality_adaptation_refinement_states import RefStateObserverCapability
            from .quality_adaptation_larger import LargerObserverCapability
            if type(inference_observer_capability) not in (InferenceObserverCapability, RefStateObserverCapability, LargerObserverCapability):
                raise ValueError('Expected an authenticated inference observer capability')
            inference_observer_capability.validate(chronological_capture_callback, inference_geometry)
        if inference_geometry == _GOTRACK_INTEGER_INFERENCE_PROFILE:
            if diagnostic_capture_callback is not None or (chronological_capture_callback is not None
                    and inference_observer_capability is None):
                raise ValueError('The adapted inference profile requires an authenticated observer capability')
        self.network = network; self.renderer = renderer; self.obj_id = obj_id; self.device = device
        self.inference_geometry = inference_geometry
        self.pnp_use_extrinsic_guess = pnp_use_extrinsic_guess
        self.diagnostic_capture_callback = diagnostic_capture_callback
        self.chronological_capture_callback = chronological_capture_callback
        self._chronological_capture_context = None
        self._inference_observer_capability = inference_observer_capability
        self._constructed_inference_geometry = inference_geometry
        # Set only by the ranker pipeline. When false, the production forward
        # path is unchanged and no backbone hook is registered.
        self.collect_ranker_features = False
        self._ranker_appearance_summary_ms = 0.0

    @property
    def ranker_appearance_summary_ms(self):
        """Cumulative CPU-side patch-summary time for one tracker trace source."""
        return self._ranker_appearance_summary_ms

    def set_chronological_capture_context(self, context):
        self._chronological_capture_context = None if context is None else dict(context)

    def _ranker_forward(self, query, template, mask, observed_mask, rendered_mask):
        if not self.collect_ranker_features:
            return self.network(query, template, mask), None
        backbone = getattr(self.network, 'backbone', None)
        register_hook = getattr(backbone, 'register_forward_hook', None)
        if not callable(register_hook):
            return self.network(query, template, mask), None
        captured = {}

        def capture_backbone(_module, _inputs, output):
            if isinstance(output, Mapping):
                captured['feature_maps'] = output.get('feature_maps')

        handle = register_hook(capture_backbone)
        try:
            result = self.network(query, template, mask)
        finally:
            handle.remove()
        summary_begin = time.perf_counter()
        try:
            summary = self._summarize_ranker_features(
                captured.get('feature_maps'), observed_mask, rendered_mask)
        finally:
            summary_ms = (time.perf_counter() - summary_begin) * 1000
            self._ranker_appearance_summary_ms += summary_ms
        summary = dict(summary)
        summary['appearance_summary_ms'] = summary_ms
        return result, summary

    @staticmethod
    def _ranker_mask_grid(mask, shape):
        if mask is None:
            return None
        try:
            array = np.asarray(_capture_numpy(mask))
        except Exception:
            return None
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            return None
        height, width = shape
        if array.shape != shape:
            resized = cv2.resize(array.astype(np.float32), (width, height),
                                 interpolation=cv2.INTER_AREA)
        else:
            resized = array
        return np.isfinite(resized) & (resized > .5)

    @classmethod
    def _summarize_ranker_features(cls, feature_maps, observed_mask, rendered_mask):
        """Compare query/template backbone patches on a masked 2x2 crop grid.

        These correspondingly positioned frozen features are a ranking signal,
        not proof of object orientation or identity.
        """
        empty = {
            'appearance_patch_cosine_00': None,
            'appearance_patch_cosine_01': None,
            'appearance_patch_cosine_10': None,
            'appearance_patch_cosine_11': None,
            'appearance_valid_fraction': None,
        }
        if feature_maps is None:
            return empty
        try:
            features = np.asarray(_capture_numpy(feature_maps), dtype=np.float32)
        except Exception:
            return empty
        if (features.ndim != 4 or features.shape[0] != 2 or
                features.shape[1] < 1 or features.shape[2] < 2 or
                features.shape[3] < 2 or not np.isfinite(features).all()):
            return empty
        query_grid, template_grid = features[0], features[1]
        spatial_shape = features.shape[2:]
        query_valid = cls._ranker_mask_grid(observed_mask, spatial_shape)
        template_valid = cls._ranker_mask_grid(rendered_mask, spatial_shape)
        if query_valid is None or template_valid is None:
            return empty
        query_norm = np.linalg.norm(query_grid, axis=0)
        template_norm = np.linalg.norm(template_grid, axis=0)
        valid = (query_valid & template_valid & np.isfinite(query_norm) &
                 np.isfinite(template_norm) & (query_norm > 1e-12) &
                 (template_norm > 1e-12))
        valid_count = int(np.count_nonzero(valid))
        if not valid_count:
            return empty
        cosine = np.sum(query_grid * template_grid, axis=0) / np.maximum(
            query_norm * template_norm, 1e-12)
        cosine = np.clip(cosine, -1., 1.)
        height, width = spatial_shape
        y_edges = np.linspace(0, height, 3, dtype=int)
        x_edges = np.linspace(0, width, 3, dtype=int)
        summary = dict(empty)
        for row in range(2):
            for column in range(2):
                ys = slice(y_edges[row], y_edges[row + 1])
                xs = slice(x_edges[column], x_edges[column + 1])
                cell_valid = valid[ys, xs]
                if np.any(cell_valid):
                    name = f'appearance_patch_cosine_{row}{column}'
                    summary[name] = float(np.mean(cosine[ys, xs][cell_valid]))
        summary['appearance_valid_fraction'] = float(valid_count / valid.size)
        return summary

    @staticmethod
    def _clone_capture_metadata(value):
        """Copy the small built-in context/metadata containers passed to hooks.

        Array payloads are kept separate and are either descriptors or private
        packet copies before they reach a callback.
        """
        if isinstance(value, dict):
            return {key: GoTrackRefiner._clone_capture_metadata(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [GoTrackRefiner._clone_capture_metadata(item) for item in value]
        if isinstance(value, tuple):
            return tuple(GoTrackRefiner._clone_capture_metadata(item) for item in value)
        return value

    @staticmethod
    def _chronological_array_descriptors(arrays):
        """Return only bounded shape/dtype/size metadata, never solver arrays."""
        descriptors = {}
        for name, value in arrays.items():
            if value is None:
                continue
            shape = getattr(value, 'shape', None)
            if shape is None:
                array = np.asarray(value)
                shape = array.shape
                dtype = array.dtype
                nbytes = array.nbytes
            else:
                dtype = getattr(value, 'dtype', type(value).__name__)
                if hasattr(value, 'numel') and hasattr(value, 'element_size'):
                    nbytes = int(value.numel()) * int(value.element_size())
                else:
                    nbytes = int(np.asarray(value).nbytes)
            descriptors[name] = dict(shape=[int(dimension) for dimension in shape],
                                      dtype=str(dtype), nbytes=int(nbytes))
        return descriptors

    @staticmethod
    def _call_chronological_hook(callback, method_name, *args, boolean_result=False):
        """Call every observer method outside the tracker RNG stream.

        Eligibility, byte/array preflight, failure handling, and the final
        packet callback all share this boundary. Hook arguments are safe copies
        of metadata; array preflight gets descriptors, while the final callback
        gets only already-copied private packet arrays.
        """
        safe_args = tuple(GoTrackRefiner._clone_capture_metadata(value) for value in args)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch = None
        torch_rng_state = None
        cuda_rng_states = None
        try:
            torch = sys.modules.get('torch')
            if torch is None:
                raise LookupError('Torch has not been loaded by inference')
            torch_rng_state = torch.get_rng_state()
            if torch.cuda.is_initialized():
                cuda_rng_states = torch.cuda.get_rng_state_all()
        except Exception:
            torch = None
            torch_rng_state = None
            cuda_rng_states = None
        try:
            method = callback if method_name is None else getattr(callback, method_name)
            result = method(*safe_args)
            return bool(result) if boolean_result else result
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            if torch is not None and torch_rng_state is not None:
                try:
                    torch.set_rng_state(torch_rng_state)
                    if cuda_rng_states is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_states)
                except Exception:
                    pass

    @classmethod
    def _disable_chronological_capture(cls, callback, reason):
        try:
            cls._call_chronological_hook(callback, '_disable_capture', reason)
        except Exception:
            pass

    def _emit_diagnostic_capture(self, event, arrays):
        """Send private array copies to an explicitly enabled diagnostic callback.

        The callback is outside the RANSAC call and never receives solver-owned
        arrays. Restore Python and NumPy RNG state around it; each iteration also
        resets OpenCV's RNG immediately before RANSAC as in ordinary inference.
        """
        callback = self.diagnostic_capture_callback
        if callback is None:
            return
        copied = {
            name: np.array(value, copy=True, order='C')
            for name, value in arrays.items() if value is not None
        }
        packet = dict(event=event, arrays=copied)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        try:
            callback(packet)
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)

    def _emit_chronological_capture(self, kind, metadata, arrays=None):
        callback = self.chronological_capture_callback
        context = self._chronological_capture_context
        if callback is None or context is None:
            return False
        arrays = {} if arrays is None else arrays
        try:
            if not self._chronological_capture_wanted(kind, metadata):
                return False
            descriptors = self._chronological_array_descriptors(arrays)
            if kind != 'trace' and not self._call_chronological_hook(callback, 'preflight',
                    context, kind, metadata, descriptors, boolean_result=True):
                return False
        except Exception as exc:
            self._disable_chronological_capture(callback,
                f'capture preflight failed: {type(exc).__name__}: {exc}')
            return False

        copied = {}
        try:
            for name, value in arrays.items():
                if value is None: continue
                if hasattr(value, 'detach'):
                    value = value.detach().cpu().numpy()
                copied[name] = np.array(value, copy=True, order='C')
        except Exception as exc:
            self._disable_chronological_capture(callback,
                f'capture array copy failed: {type(exc).__name__}: {exc}')
            return False
        try:
            packet = dict(kind=kind, context=context, metadata=metadata, arrays=copied)
            self._call_chronological_hook(callback, None, packet)
        except Exception as exc:
            self._disable_chronological_capture(callback,
                f'capture callback failed: {type(exc).__name__}: {exc}')
        return True

    def _chronological_capture_wanted(self, kind, metadata):
        callback = self.chronological_capture_callback
        context = self._chronological_capture_context
        if callback is None or context is None:
            return False
        try:
            return self._call_chronological_hook(callback, 'should_capture',
                context, kind, metadata, boolean_result=True)
        except Exception as exc:
            self._disable_chronological_capture(callback,
                f'capture eligibility check failed: {type(exc).__name__}: {exc}')
            return False

    def _solve_pnp_ransac(self, obj_points, target_points, crop_k, rvec, tvec):
        return cv2.solvePnPRansac(obj_points, target_points, crop_k, None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=self.pnp_use_extrinsic_guess,
            iterationsCount=3000, reprojectionError=2., confidence=.999,
            flags=cv2.SOLVEPNP_ITERATIVE)

    def _refine_capture(self, frame, seed, capture_binding, native_foreground):
        candidate, _ = self._refine_supported_core(
            rgb=frame.rgb, intrinsics=frame.intrinsics,
            sampling_valid=frame.sampling_valid, seed=seed,
            native_foreground=native_foreground, capture_binding=capture_binding,
            capture_frame_id=frame.frame_id, capture_timestamp_s=frame.timestamp_s)
        return candidate

    def _refine_supported_core(self, *, rgb, intrinsics, sampling_valid, seed,
                               native_foreground, capture_binding=None,
                               capture_frame_id=None, capture_timestamp_s=None,
                               model_case_id=None, model_ordinal=None):
        from .quality_capture import (crop_validity, integer_depth_points,
                                      supported_points)
        import torch
        from utils import data_util, im_util, misc, structs

        capture_mode = capture_binding is not None
        pose = pose_units(seed, 1000)
        k = np.asarray(intrinsics, dtype=np.float64)
        h, w = rgb.shape[:2]
        camera = structs.PinholePlaneCameraModel(
            width=w, height=h, f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]),
            T_world_from_eye=np.eye(4))
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
        input_mask = torch.from_numpy(native_foreground.copy()).float()[None]
        native_valid_halo = _capture_erode3(sampling_valid)
        native_foreground_halo = _capture_erode3(native_foreground)
        final = None
        ranker_features = None
        model_iteration_rows = []
        with torch.inference_mode():
            for iteration in range(5):
                data, cameras, crop_from_orig = data_util.compute_gotrack_inputs_from_init_poses(
                    input_rgbs=image, input_cameras=[camera],
                    init_poses_cam_from_model=torch.from_numpy(pose.astype(np.float32))[None],
                    renderer=self.renderer, obj_ids=[self.obj_id],
                    object_vertices=[self.renderer.vertices_m * 1000],
                    crop_size=(280, 280), crop_rel_pad=.1,
                    cropping_type='perspective_2d_box', ssaa_factor=1.,
                    background_type='gray', input_masks=input_mask)

                template = data['templates']
                if cameras[0].width != 280 or cameras[0].height != 280:
                    raise ValueError('Capture crop camera must be 280x280')
                render_generation = None
                if not capture_mode:
                    render_metadata = getattr(self.renderer, 'render_policy_metadata', None)
                    if not isinstance(render_metadata, Mapping):
                        raise ValueError('Model smoke template render has no verified renderer metadata')
                    render_dimensions = render_metadata.get('dimensions')
                    if render_dimensions is None or tuple(render_dimensions) != (280, 280):
                        raise ValueError('Model smoke template render metadata must describe the 280x280 crop')
                    raw_generation = render_metadata.get('allocation_generation')
                    if isinstance(raw_generation, (bool, np.bool_)):
                        raise ValueError('Model smoke render generation must be a positive integer')
                    try:
                        render_generation = operator.index(raw_generation)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError('Model smoke render generation must be a positive integer') from exc
                    if render_generation <= 0:
                        raise ValueError('Model smoke render generation must be a positive integer')
                crop_k = np.asarray(misc.get_intrinsic_matrix(cameras[0]), dtype=np.float64)
                if (crop_k.shape != (3, 3) or not np.isfinite(crop_k).all() or
                        crop_k[0, 0] <= 0. or crop_k[1, 1] <= 0. or
                        abs(float(np.linalg.det(crop_k))) < 1e-12):
                    raise ValueError('Capture crop camera must have finite invertible intrinsics')
                C = _capture_transform(crop_from_orig[0])
                rotation = C[:3, :3]

                # Use one shared integer-center destination grid for the actual
                # upstream camera chain and the independent K/C geometry map.
                grid_y, grid_x = np.indices((280, 280), dtype=np.int64)
                integer_grid = np.stack((grid_x, grid_y), axis=-1).reshape((-1, 2))
                crop_valid = crop_validity(
                    native_valid_halo, k, crop_k, rotation, (280, 280))
                crop_foreground = crop_validity(
                    native_foreground_halo, k, crop_k, rotation, (280, 280))
                crop_foreground &= crop_valid

                independent_native_uv = _capture_native_uv(integer_grid, crop_k, rotation, k)
                actual = _capture_actual_sampling_map(camera, cameras[0], integer_grid)
                destination_eye, world_points, source_eye, actual_uv32 = actual
                eligible_map = crop_valid.reshape(-1) & crop_foreground.reshape(-1)
                eligible_ids = np.flatnonzero(eligible_map)
                if eligible_ids.size:
                    actual_components = (destination_eye, world_points, source_eye,
                                         actual_uv32, independent_native_uv)
                    if any(not np.isfinite(component[eligible_ids]).all()
                           for component in actual_components):
                        raise ValueError('Capture returned-camera sampling map is nonfinite on supported pixels')
                    actual_source_z = source_eye[eligible_ids, 2]
                    if np.any(actual_source_z <= 0.):
                        raise ValueError('Capture returned-camera source depth is nonpositive on supported pixels')
                    map_difference = np.linalg.norm(
                        actual_uv32[eligible_ids].astype(np.float64) -
                        independent_native_uv[eligible_ids], axis=1)
                    if not np.isfinite(map_difference).all():
                        raise ValueError('Capture returned-camera sampling map difference is nonfinite')
                    max_map_difference = float(map_difference.max())
                    if max_map_difference >= 1.:
                        raise ValueError(
                            'Capture returned-camera sampling map differs by at least one native pixel '
                            f'(Euclidean max={max_map_difference:.9g})')
                else:
                    max_map_difference = 0.

                # Rewarp only the query RGB through the actual returned crop
                # camera at factor one; the upstream crop RGB is not the
                # capture-branch network input.
                query_rgb = im_util.warp_image(
                    src_camera=camera, dst_camera=cameras[0], src_image=rgb,
                    interpolation=cv2.INTER_LINEAR, depth_check=True,
                    factor_to_downsample=1)
                query_rgb = np.asarray(query_rgb)
                if query_rgb.shape != (280, 280, 3) or query_rgb.dtype != np.uint8:
                    raise ValueError('Capture query rewarp must return 280x280 uint8 RGB')
                old_query = data['crop_rgbs']
                query_tensor = torch.from_numpy(np.ascontiguousarray(query_rgb)).permute(2, 0, 1).float()[None] / 255
                data['crop_rgbs'] = query_tensor.to(old_query.device)

                old_crop_mask = data.get('crop_masks')
                crop_mask_tensor = torch.from_numpy(crop_foreground.astype(np.float32, copy=True))[None]
                if old_crop_mask is not None:
                    crop_mask_tensor = crop_mask_tensor.to(old_crop_mask.device)
                data['crop_masks'] = crop_mask_tensor

                network_result, ranker_features = self._ranker_forward(
                    data['crop_rgbs'].to(self.device), template.rgbs.to(self.device),
                    template.masks.to(self.device), data['crop_masks'], template.masks)
                flow_tensor, weight_tensor = network_result
                flow = _capture_numpy(flow_tensor[0].permute(1, 2, 0)).astype(np.float64, copy=False)
                native_weight = _capture_numpy(weight_tensor[0])
                if flow.shape != (280, 280, 2) or native_weight.shape != (280, 280):
                    raise ValueError('Capture network returned unexpected flow/confidence shapes')
                # Preserve the historical strict cutoff in the confidence tensor's
                # native dtype before promoting numerical records for diagnostics,
                # candidate weights, and solver-emitted sample weights.
                native_confidence_valid = np.isfinite(native_weight) & (native_weight > .3)
                weight = native_weight.astype(np.float64, copy=False)

                if capture_mode:
                    trace_name = f'refine/{capture_frame_id}/{iteration}'
                else:
                    trace_name = f'refine/model-smoke/{model_case_id}/{model_ordinal}/{iteration}'
                trace(trace_name, seed=pose, image=rgb, query=data['crop_rgbs'],
                      template=template.rgbs, depth=template.depths, flow=flow,
                      weight=weight)
                if capture_mode:
                    common_meta = dict(
                        frame_id=int(capture_frame_id), timestamp_s=float(capture_timestamp_s),
                        iteration=int(iteration), coordinate_mode=_CAPTURE_COORDINATE_MODE,
                        capture_binding=dict(capture_binding), factor_to_downsample=1)
                    common_meta.update(
                        camera_map_norm='euclidean_native_pixels',
                        camera_map_max_difference_px=max_map_difference,
                        camera_map_eligible_count=int(eligible_ids.size))
                    self._emit_chronological_capture('trace', dict(
                        frame_id=int(capture_frame_id), iteration=int(iteration),
                        expected_stage=f'refine/{capture_frame_id}/{iteration}', **{
                            key: value for key, value in common_meta.items()
                            if key not in ('frame_id', 'iteration')}))
                else:
                    common_meta = dict(
                        model_only=True, case_id=model_case_id,
                        ordinal=int(model_ordinal), timestamp_s=None,
                        iteration=int(iteration), coordinate_mode=_CAPTURE_COORDINATE_MODE,
                        factor_to_downsample=1,
                        camera_map_norm='euclidean_native_pixels',
                        camera_map_max_difference_px=(
                            max_map_difference if eligible_ids.size else None),
                        camera_map_eligible_count=int(eligible_ids.size))
                    model_iteration_rows.append({
                        'index': int(iteration),
                        'solver_success': False,
                        'crop_dimensions': (int(cameras[0].width), int(cameras[0].height)),
                        'query_rewarp_factor': 1,
                        'max_sampling_map_difference_px': (
                            max_map_difference if eligible_ids.size else None),
                        'sampling_map_eligible_count': int(eligible_ids.size),
                        'render_generation': int(render_generation),
                    })

                depth_mm = _capture_numpy(template.depths[0])
                rendered_mask = _capture_numpy(template.masks[0])
                if depth_mm.shape != (280, 280) or rendered_mask.shape != (280, 280):
                    raise ValueError('Capture template depth/mask must be 280x280')
                depth_pixels, xyz_crop = integer_depth_points(depth_mm, crop_k)
                if not np.array_equal(depth_pixels.reshape((-1, 2)), integer_grid):
                    raise ValueError('Capture integer-depth helper changed the fixed crop pixel grid')
                source_pixels = integer_grid
                xyz_crop = xyz_crop.reshape((-1, 3))
                target = source_pixels.astype(np.float64) + flow.reshape((-1, 2))
                weights = weight.reshape(-1)
                template_visible = rendered_mask.reshape(-1) > 0
                depth_valid = (np.isfinite(depth_mm.reshape(-1)) &
                               (depth_mm.reshape(-1) > 0.) &
                               np.isfinite(xyz_crop).all(axis=1))
                target_finite = np.isfinite(target).all(axis=1)
                base = (template_visible & depth_valid & target_finite &
                        native_confidence_valid.reshape(-1))

                source_crop_valid = _capture_supported(crop_valid, source_pixels, supported_points)
                source_crop_foreground = _capture_supported(crop_foreground, source_pixels, supported_points)
                dest_crop_valid = _capture_supported(crop_valid, target, supported_points)
                dest_crop_foreground = _capture_supported(crop_foreground, target, supported_points)
                source_native_uv = independent_native_uv
                target_native_uv = _capture_native_uv(target, crop_k, rotation, k)
                source_native_valid = _capture_supported(sampling_valid, source_native_uv, supported_points)
                source_native_foreground = _capture_supported(native_foreground, source_native_uv, supported_points)
                dest_native_valid = _capture_supported(sampling_valid, target_native_uv, supported_points)
                dest_native_foreground = _capture_supported(native_foreground, target_native_uv, supported_points)

                source_support = (source_crop_valid & source_crop_foreground &
                                  source_native_valid & source_native_foreground)
                destination_support = (dest_crop_valid & dest_crop_foreground &
                                       dest_native_valid & dest_native_foreground)
                keep = base & source_support & destination_support
                source_count = int(np.count_nonzero(base & source_support))
                destination_count = int(np.count_nonzero(base & destination_support))
                source_discard_count = int(np.count_nonzero(base & ~source_support))
                destination_discard_count = int(np.count_nonzero(base & ~destination_support))

                crop_pose = C @ pose
                xyz_object = np.full_like(xyz_crop, np.nan, dtype=np.float64)
                if np.any(depth_valid):
                    xyz_object[depth_valid] = (xyz_crop[depth_valid] - crop_pose[:3, 3]) @ crop_pose[:3, :3]
                keep &= np.isfinite(xyz_object).all(axis=1) & np.isfinite(target_native_uv).all(axis=1)
                obj_points = xyz_object[keep].astype(np.float64, copy=False)
                target_points = target[keep].astype(np.float64, copy=False)
                candidate_uv = target_native_uv[keep].astype(np.float64, copy=False)
                retained_weights = weights[keep]
                ids = np.linspace(0, len(obj_points) - 1, min(len(obj_points), 10000), dtype=int)
                rng_identity = int(capture_frame_id) if capture_mode else int(model_ordinal)
                rng_seed = (rng_identity * 17 + iteration) & 0x7fffffff
                support_metadata = dict(
                    source_support_count=source_count,
                    source_discard_count=source_discard_count,
                    destination_support_count=destination_count,
                    destination_discard_count=destination_discard_count,
                    retained_count=int(len(obj_points)))
                retained_metadata = dict(
                    **common_meta, rng_seed=int(rng_seed), native_shape=[int(h), int(w)],
                    coordinate_convention=dict(
                        rendered_depth='millimetres in crop camera',
                        object_points='GoTrack object/model coordinates in millimetres',
                        crop_targets='integer-center crop image pixel coordinates',
                        source_crop_pixels='integer pixel indices; no half-pixel offset',
                        crop_from_orig='crop camera from native camera (C)',
                        initial_pose='crop camera from object; translation in millimetres',
                        native_pose='native camera from object; translation in metres',
                        observed_mask=('cached automatic native-resolution foreground mask'
                            if capture_mode else 'same-render model depth-positive foreground mask')),
                    **support_metadata)
                chronological_capture = (capture_mode and self._chronological_capture_wanted(
                    'retained', dict(iteration=int(iteration))))
                retained_arrays = None
                if chronological_capture:
                    retained_arrays = dict(
                        query_rgb_crop=data['crop_rgbs'], template_rgb=template.rgbs,
                        rendered_depth_mm=template.depths, rendered_mask=template.masks,
                        observed_crop_mask=data['crop_masks'], full_flow_crop_px=flow,
                        full_confidence=weight, full_obj_points_mm=obj_points,
                        full_target_crop_px=target_points, sample_ids=ids,
                        sample_obj_points_mm=obj_points[ids], sample_target_crop_px=target_points[ids],
                        crop_k=crop_k, crop_from_orig=C, native_k=k,
                        seed_camera_from_object_m=pose_units(pose, .001),
                        current_crop_camera_from_object_mm=crop_pose,
                        initial_rvec=cv2.Rodrigues(crop_pose[:3, :3])[0],
                        initial_tvec_mm=crop_pose[:3, 3])
                    tensor_size = lambda value: (int(value.numel()) * int(value.element_size())
                        if hasattr(value, 'numel') and hasattr(value, 'element_size')
                        else int(np.asarray(value).nbytes))
                    retained_count = int(np.count_nonzero(keep))
                    support_array_bytes = (
                        retained_count * np.dtype(np.int64).itemsize +
                        retained_count * 2 * source_pixels.dtype.itemsize +
                        len(ids) * retained_weights.dtype.itemsize)
                    estimated_bytes = (sum(tensor_size(value) for value in retained_arrays.values()) +
                                       support_array_bytes)
                    capture_callback = self.chronological_capture_callback
                    try:
                        fits = self._call_chronological_hook(
                            capture_callback, 'preflight_nbytes',
                            self._chronological_capture_context, 'retained',
                            retained_metadata, estimated_bytes, boolean_result=True)
                    except Exception as exc:
                        self._disable_chronological_capture(capture_callback,
                            f'capture size preflight failed: {type(exc).__name__}: {exc}')
                        fits = False
                    if fits:
                        retained_arrays.update(
                            source_flat_indices=np.flatnonzero(keep).astype(np.int64),
                            source_crop_integer_pixels=source_pixels[keep],
                            sample_weights=retained_weights[ids])
                        self._emit_chronological_capture('retained', retained_metadata, retained_arrays)

                diagnostic_capture = (capture_mode and
                                      self.diagnostic_capture_callback is not None)
                if diagnostic_capture:
                    diagnostic_event = dict(
                        stage='retained', **retained_metadata)
                    diagnostic_arrays = dict(
                        full_obj_points_mm=obj_points, full_target_crop_px=target_points,
                        full_weights=retained_weights, sample_ids=ids,
                        sample_obj_points_mm=obj_points[ids],
                        sample_target_crop_px=target_points[ids],
                        crop_k=crop_k, crop_from_orig=C, native_k=k,
                        current_crop_pose_mm=crop_pose,
                        initial_rvec=cv2.Rodrigues(crop_pose[:3, :3])[0],
                        initial_tvec_mm=crop_pose[:3, 3])
                    self._emit_diagnostic_capture(diagnostic_event, diagnostic_arrays)

                if len(obj_points) < 24:
                    result_meta = dict(**common_meta, rng_seed=int(rng_seed),
                        fit_state='insufficient_retained_correspondences',
                        solver_success=False, ransac_inlier_count=0, **support_metadata)
                    if chronological_capture:
                        self._emit_chronological_capture('result', result_meta,
                            dict(ransac_inlier_ids=np.empty(0, dtype=np.int32),
                                 retained_inlier_ids=np.empty(0, dtype=np.int64)))
                    if diagnostic_capture:
                        self._emit_diagnostic_capture(dict(
                            stage='result', **result_meta, ransac_inlier_ids=[]), {})
                    return None, tuple(model_iteration_rows)

                crop_rvec = cv2.Rodrigues(crop_pose[:3, :3])[0]
                crop_tvec = crop_pose[:3, 3].copy()
                cv2.setRNGSeed(rng_seed)
                success, rvec, tvec, inliers = self._solve_pnp_ransac(
                    obj_points[ids], target_points[ids], crop_k, crop_rvec, crop_tvec)
                if not success or inliers is None or len(inliers) < 6:
                    ransac_ids = (np.empty(0, dtype=np.int32) if inliers is None
                                  else np.asarray(inliers).reshape(-1))
                    result_meta = dict(**common_meta, rng_seed=int(rng_seed),
                        fit_state='ransac_unavailable', solver_success=bool(success),
                        ransac_inlier_count=int(len(ransac_ids)), **support_metadata)
                    if chronological_capture:
                        self._emit_chronological_capture('result', result_meta, dict(
                            ransac_inlier_ids=ransac_ids,
                            retained_inlier_ids=ids[ransac_ids]))
                    if diagnostic_capture:
                        self._emit_diagnostic_capture(dict(
                            stage='result', **result_meta, ransac_inlier_ids=ransac_ids.tolist()), dict(
                            ransac_inlier_ids=ransac_ids,
                            retained_inlier_ids=ids[ransac_ids]))
                    return None, tuple(model_iteration_rows)
                ransac_ids = np.asarray(inliers).reshape(-1)
                if np.any(ransac_ids < 0) or np.any(ransac_ids >= len(ids)):
                    raise ValueError('RANSAC returned indices outside the retained sample')
                chosen = ids[ransac_ids]
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_points[chosen], target_points[chosen], crop_k, None, rvec, tvec)
                refined = np.eye(4)
                refined[:3, :3] = cv2.Rodrigues(rvec)[0]
                refined[:3, 3] = np.asarray(tvec).reshape(3)
                native_pose = canonical_pose(np.linalg.inv(C) @ refined)
                # Feed the successful native-camera refinement forward in the
                # millimetre units expected by the next GoTrack crop/render.
                pose = native_pose
                candidate = PoseCandidate(
                    pose_units(pose, .001), obj_points * .001,
                    candidate_uv, retained_weights, source='refinement',
                    ranker_features=ranker_features)
                final = candidate
                if not capture_mode:
                    model_iteration_rows[-1]['solver_success'] = True
                result_meta = dict(**common_meta, rng_seed=int(rng_seed),
                    fit_state='refined', solver_success=True,
                    ransac_inlier_count=int(len(ransac_ids)), **support_metadata)
                result_arrays = dict(
                    ransac_inlier_ids=ransac_ids, retained_inlier_ids=chosen,
                    refined_rvec=np.asarray(rvec).reshape(3),
                    refined_tvec_mm=np.asarray(tvec).reshape(3),
                    crop_camera_from_object_mm=refined,
                    native_camera_from_object_m=final.pose,
                    candidate_points_object_m=final.points_object_m,
                    candidate_pixels_native=final.pixels_image,
                    candidate_weights=final.weights)
                if diagnostic_capture:
                    self._emit_diagnostic_capture(dict(
                        stage='result', **result_meta), result_arrays)
                if chronological_capture:
                    self._emit_chronological_capture('result', result_meta, result_arrays)
        return final, tuple(model_iteration_rows)

    def refine_model_smoke(self, frame, mask, seed):
        check_model_smoke_frame(frame)
        if getattr(self.renderer, 'render_policy', None) != _CAPTURE_RENDER_POLICY:
            raise ValueError('Model smoke refinement requires capture_zero_sample_v1 renderer policy')
        if getattr(self.renderer, 'coordinate_mode', None) != _CAPTURE_COORDINATE_MODE:
            raise ValueError('Model smoke refinement requires integer_centers_v1 renderer mode')
        from .quality_capture import restrict_foreground
        native_foreground = restrict_foreground(mask, frame.sampling_valid)
        candidate, rows = self._refine_supported_core(
            rgb=frame.rgb, intrinsics=frame.intrinsics,
            sampling_valid=frame.sampling_valid, seed=seed,
            native_foreground=native_foreground, model_case_id=frame.case_id,
            model_ordinal=frame.ordinal)
        diagnostics = tuple(ModelSmokeIterationDiagnostic(**row) for row in rows)
        return ModelSmokeRefinementResult(candidate, diagnostics,
                                          frame.case_id, frame.ordinal)

    def refine(self, frame, mask, seed):
        if self.inference_geometry != self._constructed_inference_geometry:
            raise ValueError('Inference geometry cannot be changed after construction')
        if self.inference_geometry == _GOTRACK_INTEGER_INFERENCE_PROFILE:
            if self.diagnostic_capture_callback is not None:
                raise ValueError('Integer inference does not admit legacy diagnostic observers')
            if self.chronological_capture_callback is not None:
                if self._inference_observer_capability is None:
                    raise ValueError('Integer inference observer capability is missing')
                self._inference_observer_capability.validate(
                    self.chronological_capture_callback, self.inference_geometry)
        capture_binding = check_capture_frame(frame)
        integer_profile = self.inference_geometry == _GOTRACK_INTEGER_INFERENCE_PROFILE
        if integer_profile:
            if capture_binding is not None:
                raise ValueError('The adapted integer profile does not support capture Frames')
            if (getattr(self.renderer, 'render_policy', None) != _LEGACY_RENDER_POLICY or
                    getattr(self.renderer, 'coordinate_mode', None) != _CAPTURE_COORDINATE_MODE or
                    getattr(self.renderer, 'unlit', None) is not True or
                    getattr(self.renderer, 'disable_multisampling', None) is not True):
                raise ValueError('Adapted inference requires legacy_v1 rendering, integer centers, unlit templates and no multisampling')
        if capture_binding is not None:
            if getattr(self.renderer, 'render_policy', None) != _CAPTURE_RENDER_POLICY:
                raise ValueError('Capture refinement requires capture_zero_sample_v1 renderer policy')
            if getattr(self.renderer, 'coordinate_mode', None) != _CAPTURE_COORDINATE_MODE:
                raise ValueError('Capture refinement requires integer_centers_v1 renderer mode')
            from .quality_capture import restrict_foreground
            native_foreground = restrict_foreground(mask, frame.sampling_valid)
            return self._refine_capture(frame, seed, capture_binding, native_foreground)
        import torch
        from utils import data_util, misc, structs, transform3d
        pose = pose_units(seed, 1000)
        k = frame.intrinsics; h, w = frame.rgb.shape[:2]
        camera = structs.PinholePlaneCameraModel(width=w, height=h, f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]), T_world_from_eye=np.eye(4))
        image = torch.from_numpy(frame.rgb.copy()).permute(2, 0, 1).float()[None]/255
        input_mask = torch.from_numpy((mask > 0).copy()).float()[None]
        final = None
        ranker_features = None
        with torch.inference_mode():
            for iteration in range(5):
                # Always rerender the fixed model, not previously inferred points.
                data, cameras, crop_from_orig = data_util.compute_gotrack_inputs_from_init_poses(
                    input_rgbs=image, input_cameras=[camera], init_poses_cam_from_model=torch.from_numpy(pose.astype(np.float32))[None],
                    renderer=self.renderer, obj_ids=[self.obj_id], object_vertices=[self.renderer.vertices_m*1000],
                    crop_size=(280, 280), crop_rel_pad=.1, cropping_type='perspective_2d_box', ssaa_factor=1.,
                    background_type='gray', input_masks=input_mask)
                template = data['templates']; crop_k = misc.get_intrinsic_matrix(cameras[0])
                network_result, ranker_features = self._ranker_forward(
                    data['crop_rgbs'].to(self.device), template.rgbs.to(self.device),
                    template.masks.to(self.device), data.get('crop_masks'), template.masks)
                flow, weight = network_result
                flow = flow[0].permute(1, 2, 0).cpu().numpy(); weight = weight[0].cpu().numpy()
                trace(f'refine/{frame.frame_id}/{iteration}', seed=pose, image=frame.rgb,
                      query=data['crop_rgbs'], template=template.rgbs, depth=template.depths,
                      flow=flow, weight=weight)
                self._emit_chronological_capture('trace', dict(frame_id=int(frame.frame_id),
                    iteration=int(iteration), expected_stage=f'refine/{frame.frame_id}/{iteration}'))
                if integer_profile:
                    (crop_pose, xyz, target, native_uv, confidence, valid,
                     integer_pixels, xyz_cam) = _integer_profile_correspondences(
                        template.depths, crop_k, crop_from_orig[0], pose, flow, weight,
                        template.masks[0], data['crop_masks'][0], k)
                    source_pixel_centers = integer_pixels
                    crop_mask = _capture_numpy(data['crop_masks'][0])
                    obj_points = xyz[valid].astype(np.float64)
                    target_points = target[valid].astype(np.float64)
                    candidate_pixels = native_uv[valid].astype(np.float64)
                    candidate_weights = confidence[valid].astype(np.float64)
                else:
                    # Upstream uses pixel centers (x+.5,y+.5) for rendered depth.
                    pixels, xyz_cam = transform3d.get_3d_points_from_depth(template.depths, torch.from_numpy(crop_k.astype(np.float32))[None])
                    source_pixel_centers = pixels[0].numpy()
                    target = source_pixel_centers+flow
                    crop_pose = crop_from_orig[0].numpy()@pose
                    xyz = (xyz_cam[0].numpy()-crop_pose[:3, 3])@crop_pose[:3, :3]
                    xx, yy = np.floor(target[..., 0]).astype(int), np.floor(target[..., 1]).astype(int)
                    valid = template.masks[0].numpy() & np.isfinite(target).all(2) & (weight > .3)
                    valid &= (xx >= 0) & (xx < 280) & (yy >= 0) & (yy < 280)
                    crop_mask = data['crop_masks'][0].numpy()
                    # The prior render can include surface behind a hand. Require
                    # current visible foreground at the source projection too;
                    # otherwise an occluded source can falsely match another part
                    # of the object and still land inside the destination mask.
                    valid &= crop_mask > .5
                    valid &= crop_mask[np.clip(yy, 0, 279), np.clip(xx, 0, 279)] > .5
                    obj_points = xyz[valid].astype(np.float64); target_points = target[valid].astype(np.float64)
                    candidate_pixels = None
                    candidate_weights = weight[valid]
                # Deterministic subsampling and frame-local RANSAC RNG make
                # early results invariant to the length of a later video suffix.
                ids = np.linspace(0, len(obj_points)-1, min(len(obj_points), 10000), dtype=int)
                rng_seed = (int(frame.frame_id)*17+iteration) & 0x7fffffff
                retained_metadata = dict(frame_id=int(frame.frame_id), iteration=int(iteration), rng_seed=int(rng_seed),
                    native_shape=[int(h), int(w)],
                    coordinate_convention=dict(
                        rendered_depth='millimetres in crop camera',
                        object_points='GoTrack object/model coordinates in millimetres',
                        crop_targets='crop image pixel coordinates',
                        source_pixel_centers=('integer crop pixel indices used by integer_depth_points'
                            if integer_profile else 'crop pixel centers returned by production depth unprojection'),
                        crop_from_orig='crop camera from native camera (C)',
                        initial_pose='crop camera from object; translation in millimetres',
                        native_pose='native camera from object; translation in metres',
                        observed_mask='cached automatic native-resolution foreground mask'))
                chronological_capture = self._chronological_capture_wanted('retained',
                    dict(iteration=int(iteration)))
                if chronological_capture:
                    retained_arrays = dict(query_rgb_crop=data['crop_rgbs'], template_rgb=template.rgbs,
                        rendered_depth_mm=template.depths, rendered_mask=template.masks,
                        observed_crop_mask=data['crop_masks'], full_flow_crop_px=flow,
                        full_confidence=weight, full_obj_points_mm=obj_points,
                        full_target_crop_px=target_points, sample_ids=ids,
                        sample_obj_points_mm=obj_points[ids], sample_target_crop_px=target_points[ids],
                        crop_k=crop_k, crop_from_orig=crop_from_orig[0].numpy(), native_k=k,
                        seed_camera_from_object_m=np.array(seed, copy=True),
                        current_crop_camera_from_object_mm=crop_pose,
                        initial_rvec=cv2.Rodrigues(crop_pose[:3, :3])[0],
                        initial_tvec_mm=crop_pose[:3, 3])
                    source_count = int(np.count_nonzero(valid))
                    source_center_bytes = source_count * 2 * source_pixel_centers.dtype.itemsize
                    sample_weight_bytes = len(ids) * weight.dtype.itemsize
                    tensor_size = lambda value: (int(value.numel()) * int(value.element_size())
                        if hasattr(value, 'numel') and hasattr(value, 'element_size') else int(np.asarray(value).nbytes))
                    estimated_bytes = sum(tensor_size(value) for value in retained_arrays.values())
                    estimated_bytes += source_count * np.dtype(np.int64).itemsize + source_center_bytes + sample_weight_bytes
                    capture_callback = self.chronological_capture_callback
                    try:
                        fits = self._call_chronological_hook(capture_callback, 'preflight_nbytes',
                            self._chronological_capture_context, 'retained', retained_metadata,
                            estimated_bytes, boolean_result=True)
                    except Exception as exc:
                        self._disable_chronological_capture(capture_callback,
                            f'capture size preflight failed: {type(exc).__name__}: {exc}')
                        fits = False
                    if fits:
                        retained_arrays.update(source_flat_indices=np.flatnonzero(valid).astype(np.int64),
                            source_crop_pixel_centers=source_pixel_centers[valid],
                            sample_weights=weight[valid][ids])
                        self._emit_chronological_capture('retained', retained_metadata, retained_arrays)
                diagnostic_capture = self.diagnostic_capture_callback is not None
                if diagnostic_capture:
                    initial_rvec = cv2.Rodrigues(crop_pose[:3, :3])[0]
                    initial_tvec = crop_pose[:3, 3].copy()
                    self._emit_diagnostic_capture(dict(
                        stage='retained', frame_id=int(frame.frame_id), iteration=int(iteration),
                        rng_seed=rng_seed,
                        native_shape=[int(h), int(w)],
                        coordinate_convention=dict(
                            object_points='GoTrack object/model coordinates in millimetres',
                            crop_targets='crop image pixel coordinates',
                            crop_from_orig='crop camera from native camera (C)',
                            initial_pose='crop camera from object; translation in millimetres',
                            native_pose='native camera from object; translation in metres',
                        )), dict(
                            full_obj_points_mm=obj_points,
                            full_target_crop_px=target_points,
                        full_weights=candidate_weights,
                            sample_ids=ids,
                            sample_obj_points_mm=obj_points[ids],
                            sample_target_crop_px=target_points[ids],
                            crop_k=crop_k,
                            crop_from_orig=crop_from_orig[0].numpy(),
                            native_k=k,
                            current_crop_pose_mm=crop_pose,
                            initial_rvec=initial_rvec,
                            initial_tvec_mm=initial_tvec,
                        ))
                if len(obj_points) < 24:
                    if chronological_capture:
                        self._emit_chronological_capture('result', dict(frame_id=int(frame.frame_id),
                            iteration=int(iteration), rng_seed=int(rng_seed), fit_state='insufficient_retained_correspondences',
                            solver_success=False, ransac_inlier_count=0),
                            dict(ransac_inlier_ids=np.empty(0, dtype=np.int32),
                                 retained_inlier_ids=np.empty(0, dtype=np.int64)))
                    if diagnostic_capture:
                        self._emit_diagnostic_capture(dict(
                            stage='result', frame_id=int(frame.frame_id), iteration=int(iteration),
                            rng_seed=rng_seed, fit_state='insufficient_retained_correspondences',
                            solver_success=False, ransac_inlier_ids=[]), {})
                    return None
                if not diagnostic_capture:
                    initial_rvec = cv2.Rodrigues(crop_pose[:3, :3])[0]
                    initial_tvec = crop_pose[:3, 3].copy()
                cv2.setRNGSeed(rng_seed)
                success, rvec, tvec, inliers = self._solve_pnp_ransac(
                    obj_points[ids], target_points[ids], crop_k, initial_rvec, initial_tvec)
                if not success or inliers is None or len(inliers) < 6:
                    if chronological_capture:
                        try:
                            ransac_inlier_ids = (np.empty(0, dtype=np.int32) if inliers is None
                                                 else np.asarray(inliers).reshape(-1))
                            retained_inlier_ids = ids[ransac_inlier_ids]
                            self._emit_chronological_capture('result', dict(frame_id=int(frame.frame_id),
                                iteration=int(iteration), rng_seed=int(rng_seed), fit_state='ransac_unavailable',
                                solver_success=bool(success), ransac_inlier_count=int(len(ransac_inlier_ids))),
                                dict(ransac_inlier_ids=ransac_inlier_ids,
                                     retained_inlier_ids=retained_inlier_ids))
                        except Exception as exc:
                            self._disable_chronological_capture(self.chronological_capture_callback,
                                f'capture solver-index snapshot failed: {type(exc).__name__}: {exc}')
                    if diagnostic_capture:
                        ransac_inlier_ids = (np.empty(0, dtype=np.int32) if inliers is None
                                             else np.asarray(inliers).reshape(-1))
                        self._emit_diagnostic_capture(dict(
                            stage='result', frame_id=int(frame.frame_id), iteration=int(iteration),
                            rng_seed=rng_seed, fit_state='ransac_unavailable',
                            solver_success=bool(success),
                            ransac_inlier_count=int(len(ransac_inlier_ids))), dict(
                                ransac_inlier_ids=ransac_inlier_ids,
                                retained_inlier_ids=ids[ransac_inlier_ids],
                            ))
                    return None
                chosen = ids[inliers.ravel()]
                if chronological_capture and self._inference_observer_capability is not None:
                    self._emit_chronological_capture('pre_lm', dict(frame_id=int(frame.frame_id),
                        iteration=int(iteration), rng_seed=int(rng_seed), solver_success=True,
                        ransac_inlier_count=int(len(inliers))), dict(
                        ransac_inlier_ids=np.asarray(inliers).reshape(-1), retained_inlier_ids=chosen,
                        ransac_rvec=np.asarray(rvec).reshape(3),
                        ransac_tvec_mm=np.asarray(tvec).reshape(3)))
                rvec, tvec = cv2.solvePnPRefineLM(obj_points[chosen], target_points[chosen], crop_k, None, rvec, tvec)
                refined = np.eye(4); refined[:3, :3] = cv2.Rodrigues(rvec)[0]; refined[:3, 3] = tvec.ravel()
                pose = canonical_pose(np.linalg.inv(_capture_numpy(crop_from_orig[0]))@refined)
                # Transform matched image rays back to native camera pixels.
                if integer_profile:
                    uv = candidate_pixels
                else:
                    rays = np.c_[target_points, np.ones(len(target_points))]@np.linalg.inv(crop_k).T
                    rays_orig = rays@crop_from_orig[0].numpy()[:3, :3]
                    projected = rays_orig@k.T; uv = projected[:, :2]/projected[:, 2:3]
                final = PoseCandidate(
                    pose_units(pose, .001), obj_points*.001, uv, candidate_weights,
                    ranker_features=ranker_features)
                if diagnostic_capture:
                    ransac_inlier_ids = np.asarray(inliers).reshape(-1)
                    self._emit_diagnostic_capture(dict(
                        stage='result', frame_id=int(frame.frame_id), iteration=int(iteration),
                        rng_seed=rng_seed, fit_state='refined', solver_success=True,
                        ransac_inlier_count=int(len(ransac_inlier_ids))), dict(
                        ransac_inlier_ids=ransac_inlier_ids,
                        retained_inlier_ids=chosen,
                        refined_rvec=np.asarray(rvec).reshape(3),
                        refined_tvec_mm=np.asarray(tvec).reshape(3),
                        crop_camera_from_object_mm=refined,
                        native_camera_from_object_m=final.pose,
                        candidate_points_object_m=final.points_object_m,
                        candidate_pixels_native=final.pixels_image,
                        candidate_weights=final.weights,
                    ))
                if chronological_capture:
                    ransac_inlier_ids = np.asarray(inliers).reshape(-1)
                    self._emit_chronological_capture('result', dict(frame_id=int(frame.frame_id),
                        iteration=int(iteration), rng_seed=int(rng_seed), fit_state='refined',
                        solver_success=True, ransac_inlier_count=int(len(ransac_inlier_ids))),
                        dict(ransac_inlier_ids=ransac_inlier_ids, retained_inlier_ids=chosen,
                             refined_rvec=np.asarray(rvec).reshape(3),
                             refined_tvec_mm=np.asarray(tvec).reshape(3),
                             crop_camera_from_object_mm=refined,
                             native_camera_from_object_m=final.pose))
        return final
