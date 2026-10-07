"""Inference-only contracts, same-frame validation and sequential recovery.

No dataset labels or evaluation annotations are accepted by this module.
Public poses: OpenCV camera-from-object, column vectors, translation in metres.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
from hashlib import sha256
from types import MappingProxyType
import time
import numpy as np
from .quality_time import LEGACY, PHYSICAL, FrameKey, TimePolicy, LEGACY_POLICY
from .vision import cv2


_CAPTURE_BINDING_FIELDS = (
    'capture_table_sha256', 'capture_row_sha256', 'rgb_pixel_sha256',
    'sampling_valid_sha256', 'warp_sha256', 'calibration_sha256',
)
_CAPTURE_K = np.array([[360., 0., 359.5], [0., 360., 359.5], [0., 0., 1.]],
                      dtype=np.float64)
_CAPTURE_K.setflags(write=False)


def _capture_digest(array):
    contiguous = np.ascontiguousarray(array)
    return sha256(memoryview(contiguous).cast('B')).hexdigest().upper()


def _capture_valid_digest(valid):
    # This is the framed camera digest used by DecodedCapture, not the raw PNG
    # pixel digest. The byte layout is deliberately explicit and stable.
    pixels = np.ascontiguousarray(valid, dtype=np.uint8)
    digest = sha256()
    digest.update(b'sampling_valid\0[720,720]\0|u1\0')
    digest.update(memoryview(pixels).cast('B'))
    return digest.hexdigest().upper()


def _capture_binding(value):
    if not isinstance(value, Mapping):
        raise ValueError('Capture binding must be a mapping')
    try:
        detached = dict(value)
    except Exception as exc:
        raise ValueError('Capture binding must be a readable mapping') from exc
    if set(detached) != set(_CAPTURE_BINDING_FIELDS):
        raise ValueError('Capture binding must contain exactly the six capture identities')
    for name in _CAPTURE_BINDING_FIELDS:
        item = detached[name]
        if (not isinstance(item, str) or len(item) != 64 or
                any(char not in '0123456789ABCDEF' for char in item)):
            raise ValueError(f'{name} must be an uppercase SHA-256 digest')
    return detached


def _capture_array_snapshot(value, *, shape, dtype, name):
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f'{name} must be a valid array') from exc
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(f'{name} must have shape {shape} and dtype {np.dtype(dtype)}')
    contiguous = np.ascontiguousarray(array)
    backing = contiguous.tobytes(order='C')
    snapshot = np.frombuffer(backing, dtype=np.dtype(dtype)).reshape(shape)
    return snapshot


def _immutable_bytes_snapshot(value):
    if not isinstance(value, np.ndarray) or value.flags.writeable:
        return False
    base = value
    while isinstance(base, np.ndarray):
        base = base.base
    return type(base) is bytes


@dataclass(frozen=True, slots=True)
class ModelSmokeFrame:
    """Synthetic model-rendered query image with no capture identity fields."""
    case_id: str
    ordinal: int
    rgb: np.ndarray
    intrinsics: np.ndarray
    sampling_valid: np.ndarray

    def __post_init__(self):
        if type(self.case_id) is not str or self.case_id not in ('a', 'b', 'c'):
            raise ValueError('Model smoke case_id must be one of a, b, or c')
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError('Model smoke ordinal must be a nonnegative synthetic integer')
        try:
            rgb = np.asarray(self.rgb)
            valid = np.asarray(self.sampling_valid)
            intrinsics = np.asarray(self.intrinsics)
        except Exception as exc:
            raise ValueError('Model smoke frame arrays are invalid') from exc
        if rgb.shape != (720, 720, 3) or rgb.dtype != np.uint8:
            raise ValueError('Model smoke RGB must be 720x720 uint8')
        if valid.shape != (720, 720) or valid.dtype != np.bool_:
            raise ValueError('Model smoke sampling validity must be a 720x720 bool map')
        if (intrinsics.shape != (3, 3) or intrinsics.dtype.kind not in 'iuf' or
                intrinsics.dtype.kind == 'b' or not np.isfinite(intrinsics).all() or
                not np.array_equal(intrinsics, _CAPTURE_K)):
            raise ValueError('Model smoke intrinsics must be the fixed 720 camera matrix')
        object.__setattr__(self, 'rgb', _capture_array_snapshot(
            rgb, shape=(720, 720, 3), dtype=np.uint8, name='RGB'))
        object.__setattr__(self, 'sampling_valid', _capture_array_snapshot(
            valid, shape=(720, 720), dtype=np.bool_, name='sampling_valid'))
        object.__setattr__(self, 'intrinsics', _capture_array_snapshot(
            _CAPTURE_K, shape=(3, 3), dtype=np.float64, name='intrinsics'))


def make_model_smoke_frame(*, case_id, ordinal, rgb, intrinsics, sampling_valid):
    """Construct a checked, immutable model-only frame from synthetic arrays."""
    return ModelSmokeFrame(case_id=case_id, ordinal=ordinal, rgb=rgb,
                           intrinsics=intrinsics, sampling_valid=sampling_valid)


def check_model_smoke_frame(frame):
    """Recheck the exact synthetic frame type and its byte-backed arrays."""
    if type(frame) is not ModelSmokeFrame:
        raise ValueError('Model smoke APIs require an exact ModelSmokeFrame')
    if type(frame.case_id) is not str or frame.case_id not in ('a', 'b', 'c'):
        raise ValueError('Model smoke case_id must be one of a, b, or c')
    if type(frame.ordinal) is not int or frame.ordinal < 0:
        raise ValueError('Model smoke ordinal must be a nonnegative synthetic integer')
    rgb = np.asarray(frame.rgb)
    valid = np.asarray(frame.sampling_valid)
    intrinsics = np.asarray(frame.intrinsics)
    if (rgb.shape != (720, 720, 3) or rgb.dtype != np.uint8 or
            valid.shape != (720, 720) or valid.dtype != np.bool_ or
            intrinsics.shape != (3, 3) or intrinsics.dtype != np.float64 or
            not np.array_equal(intrinsics, _CAPTURE_K)):
        raise ValueError('Model smoke frame arrays do not match the fixed raster and camera')
    if not all(_immutable_bytes_snapshot(value)
               for value in (rgb, valid, intrinsics)):
        raise ValueError('Model smoke RGB, validity, and intrinsics must be immutable byte-backed snapshots')


def check_capture_frame(frame):
    """Recheck capture-only frame identity and immutable pixel bindings.

    Returns the detached six-field binding for a capture frame and ``None`` for
    every historical frame. It intentionally does not authenticate the capture
    resource table; the reader owns that boundary.
    """
    if isinstance(frame, ModelSmokeFrame):
        raise ValueError('ModelSmokeFrame requires the model-only capture smoke API')
    sampling_valid = getattr(frame, 'sampling_valid', None)
    binding_value = getattr(frame, 'capture_binding', None)
    if sampling_valid is None and binding_value is None:
        return None
    if sampling_valid is None or binding_value is None:
        raise ValueError('Capture Frame requires both sampling_valid and capture_binding')
    if not isinstance(frame, Frame):
        raise ValueError('Capture data requires a quality Frame')
    try:
        key = FrameKey(frame.frame_id, frame.timestamp_s, frame.clock_mode)
    except Exception as exc:
        raise ValueError('Capture Frame has an invalid physical chronology key') from exc
    if key.clock_mode != PHYSICAL or key.timestamp_s is None:
        raise ValueError('Capture Frame requires the physical capture clock')
    rgb = np.asarray(frame.rgb)
    valid = np.asarray(sampling_valid)
    intrinsics = np.asarray(frame.intrinsics)
    if rgb.shape != (720, 720, 3) or rgb.dtype != np.uint8:
        raise ValueError('Capture Frame RGB must be 720x720 uint8')
    if valid.shape != (720, 720) or valid.dtype != np.bool_:
        raise ValueError('Capture sampling validity must be a 720x720 bool map')
    if intrinsics.shape != (3, 3) or intrinsics.dtype.kind not in 'iuf' or intrinsics.dtype.kind == 'b':
        raise ValueError('Capture intrinsics must be the fixed 720 camera matrix')
    if not np.array_equal(intrinsics, _CAPTURE_K):
        raise ValueError('Capture intrinsics must be the fixed 720 camera matrix')
    if (not _immutable_bytes_snapshot(rgb) or not _immutable_bytes_snapshot(valid) or
            not _immutable_bytes_snapshot(intrinsics)):
        raise ValueError('Capture RGB, validity, and intrinsics must be immutable byte-backed snapshots')
    if type(binding_value) is not MappingProxyType:
        raise ValueError('Capture binding must be a detached immutable mapping')
    binding = _capture_binding(binding_value)
    if _capture_digest(rgb) != binding['rgb_pixel_sha256']:
        raise ValueError('Capture RGB pixels no longer match their binding')
    if _capture_valid_digest(valid) != binding['sampling_valid_sha256']:
        raise ValueError('Capture validity no longer matches its framed binding')
    return MappingProxyType(binding)


def checked_pose(value):
    p = np.asarray(value, float)
    if p.shape != (4, 4) or not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1]):
        raise ValueError('Expected finite SE(3) camera-from-object')
    r = p[:3, :3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-4) or abs(np.linalg.det(r)-1) > 1e-4:
        raise ValueError('Improper object rotation')
    return p.copy()


def pose_units(pose, factor):
    p = checked_pose(pose); p[:3, 3] *= factor
    return p


def canonical_pose(value):
    """Remove float crop-transform roundoff without accepting invalid poses."""
    p = checked_pose(value)
    u, _, vt = np.linalg.svd(p[:3, :3])
    p[:3, :3] = u@vt
    return checked_pose(p)


def project(points, pose, intrinsics):
    p = np.asarray(points)@pose[:3, :3].T + pose[:3, 3]
    q = p@np.asarray(intrinsics).T
    return q[:, :2]/q[:, 2:3], p[:, 2]


@dataclass(frozen=True)
class Frame:
    frame_id: int
    rgb: np.ndarray
    intrinsics: np.ndarray
    timestamp_s: float | None = None
    clock_mode: str = LEGACY
    sampling_valid: np.ndarray | None = None
    capture_binding: Mapping | None = None

    def __post_init__(self):
        if any(isinstance(value, ModelSmokeFrame) for value in
               (self.frame_id, self.rgb, self.intrinsics, self.sampling_valid,
                self.capture_binding)):
            raise ValueError('ModelSmokeFrame cannot be wrapped as a quality Frame')
        key = FrameKey(self.frame_id, self.timestamp_s, self.clock_mode)
        object.__setattr__(self, 'frame_id', key.frame_id)
        object.__setattr__(self, 'timestamp_s', key.timestamp_s)
        if self.sampling_valid is not None or self.capture_binding is not None:
            if self.sampling_valid is None or self.capture_binding is None:
                raise ValueError('Capture Frame requires both sampling_valid and capture_binding')
            if key.clock_mode != PHYSICAL:
                raise ValueError('Capture Frame requires the physical capture clock')
            try:
                rgb = np.asarray(self.rgb)
                valid = np.asarray(self.sampling_valid)
                k = np.asarray(self.intrinsics)
            except Exception as exc:
                raise ValueError('Capture Frame arrays are invalid') from exc
            if rgb.shape != (720, 720, 3) or rgb.dtype != np.uint8:
                raise ValueError('Capture Frame RGB must be 720x720 uint8')
            if valid.shape != (720, 720) or valid.dtype != np.bool_:
                raise ValueError('Capture sampling validity must be a 720x720 bool map')
            if (k.shape != (3, 3) or k.dtype.kind not in 'iuf' or k.dtype.kind == 'b' or
                    not np.isfinite(k).all() or not np.array_equal(k, _CAPTURE_K)):
                raise ValueError('Capture intrinsics must be the fixed 720 camera matrix')
            binding = _capture_binding(self.capture_binding)
            rgb_snapshot = _capture_array_snapshot(rgb, shape=(720, 720, 3), dtype=np.uint8, name='RGB')
            valid_snapshot = _capture_array_snapshot(valid, shape=(720, 720), dtype=np.bool_, name='sampling_valid')
            if _capture_digest(rgb_snapshot) != binding['rgb_pixel_sha256']:
                raise ValueError('Capture RGB pixels do not match their binding')
            if _capture_valid_digest(valid_snapshot) != binding['sampling_valid_sha256']:
                raise ValueError('Capture validity does not match its framed binding')
            k_snapshot = _capture_array_snapshot(_CAPTURE_K, shape=(3, 3), dtype=np.float64,
                                                 name='intrinsics')
            object.__setattr__(self, 'rgb', rgb_snapshot)
            object.__setattr__(self, 'sampling_valid', valid_snapshot)
            object.__setattr__(self, 'intrinsics', k_snapshot)
            object.__setattr__(self, 'capture_binding', MappingProxyType(binding))
            return
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError('RGB uint8 image required')
        k = np.asarray(self.intrinsics)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError('Calibrated positive camera intrinsics required')

    @property
    def key(self):
        return FrameKey(self.frame_id, self.timestamp_s, self.clock_mode)


@dataclass
class MaskPrediction:
    frame_id: int
    mask: np.ndarray | None
    state: str = 'available'
    reason: str | None = None
    timings_ms: dict = field(default_factory=dict)
    timestamp_s: float | None = None
    clock_mode: str = LEGACY

    def __post_init__(self):
        key = FrameKey(self.frame_id, self.timestamp_s, self.clock_mode)
        self.frame_id = key.frame_id
        self.timestamp_s = key.timestamp_s

    @property
    def key(self):
        return FrameKey(self.frame_id, self.timestamp_s, self.clock_mode)

    def usable(self, shape):
        return self.state == 'available' and self.mask is not None and self.mask.shape == shape and np.any(self.mask)


@dataclass
class PoseCandidate:
    pose: np.ndarray
    points_object_m: np.ndarray
    pixels_image: np.ndarray
    weights: np.ndarray
    source: str = 'refinement'


@dataclass(frozen=True)
class ValidationSettings:
    visibility_weight: float = .3  # Upstream GoTrack confidence cutoff.
    min_correspondences: int = 24
    min_inliers: int = 20
    min_inlier_ratio: float = .6
    max_median_error_720: float = 3.
    max_p95_error_720: float = 8.
    min_spatial_support: float = .12
    recovery_ambiguity_ratio: float = .9
    recovery_rotation_degrees: float = 20.
    recovery_translation_m: float = .03
    # Conservative continuity bounds for these 60 FPS handheld recordings.
    # These reject jumps; they never make a weak correspondence acceptable.
    max_rotation_degrees_per_frame: float = 12.
    rotation_margin_degrees: float = 5.
    max_translation_m_per_frame: float = .02
    translation_margin_m: float = .01
    private_pose_memory_frames: int = 30


def _validate_candidate(candidate, frame, mask, settings, *, support_aware,
                        sampling_valid=None, restricted_foreground=None):
    p = checked_pose(candidate.pose)
    xyz = np.asarray(candidate.points_object_m, float).reshape(-1, 3)
    uv = np.asarray(candidate.pixels_image, float).reshape(-1, 2)
    weights = np.asarray(candidate.weights, float).reshape(-1)
    if not len(xyz) == len(uv) == len(weights): raise ValueError('Correspondence lengths disagree')
    h, w = frame.rgb.shape[:2]
    finite = np.isfinite(xyz).all(1) & np.isfinite(uv).all(1) & np.isfinite(weights)
    keep = finite & (weights >= settings.visibility_weight)
    if not support_aware:
        keep &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids = np.flatnonzero(keep)
        if mask is not None:
            xy = uv[ids].astype(int)
            keep[ids] &= mask[xy[:, 1], xy[:, 0]] > 0
    else:
        from .quality_capture import supported_points
        eligible = np.flatnonzero(keep)
        valid_support = np.zeros(len(keep), dtype=bool)
        foreground_support = np.zeros(len(keep), dtype=bool)
        # Do not hand enormous but finite coordinates to floor/int conversion in
        # the helper. Out-of-frame candidates remain unsupported, never clipped.
        bounded = eligible[(uv[eligible, 0] >= 0) & (uv[eligible, 0] < w - 1) &
                           (uv[eligible, 1] >= 0) & (uv[eligible, 1] < h - 1)]
        if bounded.size:
            valid_support[bounded] = supported_points(sampling_valid, uv[bounded])
            foreground_support[bounded] = supported_points(restricted_foreground, uv[bounded])
        supported = valid_support & foreground_support
        unsupported_count = int(np.count_nonzero(keep & ~supported))
        keep &= supported
    xyz, uv, weights = xyz[keep], uv[keep], weights[keep]
    stats = {'correspondences': len(xyz), 'inliers': 0, 'median_reprojection_720': None,
             'p95_reprojection_720': None, 'spatial_support': 0., 'score': 0.}
    if support_aware:
        stats.update(raw_correspondences=int(len(candidate.points_object_m)),
                     retained_correspondences=int(len(xyz)),
                     unsupported_correspondences=unsupported_count)
    if len(xyz) < settings.min_correspondences: return False, 'insufficient_visible_correspondences', stats
    predicted, z = project(xyz, p, frame.intrinsics)
    errors = np.linalg.norm(predicted-uv, axis=1)*720/h
    inliers = (z > .01) & np.isfinite(errors) & (errors <= settings.max_p95_error_720)
    count = int(inliers.sum()); stats['inliers'] = count
    if count < settings.min_inliers or count/len(xyz) < settings.min_inlier_ratio:
        return False, 'inconsistent_current_image_correspondences', stats
    e = errors[inliers]
    stats.update(median_reprojection_720=float(np.median(e)), p95_reprojection_720=float(np.percentile(e, 95)))
    support_area = cv2.contourArea(cv2.convexHull(uv[inliers].astype(np.float32)))
    if support_aware:
        yy, xx = np.nonzero(restricted_foreground)
        denominator = max((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1), 1) if len(xx) else h*w
    elif mask is not None:
        yy, xx = np.nonzero(mask)
        denominator = max((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1), 1) if len(xx) else h*w
    else: denominator = h*w
    stats['spatial_support'] = float(support_area/denominator)
    stats['score'] = float(np.mean(weights[inliers])*(count/len(xyz))*min(1., stats['spatial_support']/.3))
    if stats['median_reprojection_720'] > settings.max_median_error_720:
        return False, 'high_reprojection_error', stats
    if stats['spatial_support'] < settings.min_spatial_support:
        return False, 'localized_or_ambiguous_support', stats
    return True, None, stats


def validate(candidate, frame, mask, settings=ValidationSettings()):
    """Validate against current RGB correspondences, never silhouette agreement.

    Mask rejection occurs before fitting. The refiner must likewise mask its
    target correspondences before solving. This second check guards adapters.
    """
    capture_binding = check_capture_frame(frame)
    restricted_foreground = None
    if capture_binding is not None:
        if mask is None:
            raise ValueError('Capture validation requires a binary current foreground mask')
        from .quality_capture import restrict_foreground
        restricted_foreground = restrict_foreground(mask, frame.sampling_valid)
    return _validate_candidate(
        candidate, frame, mask, settings, support_aware=capture_binding is not None,
        sampling_valid=(frame.sampling_valid if capture_binding is not None else None),
        restricted_foreground=restricted_foreground)


def validate_model_smoke(candidate, frame, mask, settings=ValidationSettings()):
    """Apply the public correspondence checks to one synthetic model-smoke view."""
    check_model_smoke_frame(frame)
    if mask is None:
        raise ValueError('Model smoke validation requires the same-render depth-positive foreground')
    from .quality_capture import restrict_foreground
    restricted_foreground = restrict_foreground(mask, frame.sampling_valid)
    return _validate_candidate(candidate, frame, mask, settings, support_aware=True,
                               sampling_valid=frame.sampling_valid,
                               restricted_foreground=restricted_foreground)


def distinct_pose(a, b, settings):
    angle = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T@b[:3, :3])-1)/2, -1, 1)))
    return angle > settings.recovery_rotation_degrees or np.linalg.norm(a[:3, 3]-b[:3, 3]) > settings.recovery_translation_m


class SequentialTracker:
    """Backend: refine(frame, mask, seed), recover(frame, mask, top_k=5).

    Segmentation runs in a separate process and is supplied as a cache record.
    No stale pose is returned. A private candidate can seed recovery verification.
    """
    def __init__(self, backend, initial_pose=None, settings=ValidationSettings(), *,
                 time_policy=LEGACY_POLICY, initial_frame_id=None, initial_timestamp_s=None):
        if not isinstance(time_policy, TimePolicy):
            raise ValueError('time_policy must be a TimePolicy')
        self.backend = backend; self.settings = settings; self.time_policy = time_policy
        if time_policy.mode == LEGACY:
            if initial_frame_id is not None or initial_timestamp_s is not None:
                raise ValueError('Legacy controlled seeds cannot specify frame/time keys')
        else:
            if initial_pose is None:
                if initial_frame_id is not None or initial_timestamp_s is not None:
                    raise ValueError('Physical seed keys require an initial pose')
            elif initial_frame_id is None or initial_timestamp_s is None:
                raise ValueError('Physical initial pose requires its explicit frame and timestamp')
        self._check_backend_policy()
        self.accepted = None if initial_pose is None else checked_pose(initial_pose)
        self.pending = None
        self.pending_id = None
        self.pending_time_s = None
        self.last_id = None
        self.last_time_s = None
        self.last_valid_pose = None if self.accepted is None else self.accepted.copy()
        self.last_valid_id = None
        self.last_valid_time_s = None
        self.rejections = []
        self.terminal = False
        self.terminal_key = None
        if time_policy.mode == PHYSICAL and self.accepted is not None:
            seed_key = FrameKey(initial_frame_id, initial_timestamp_s, PHYSICAL)
            if seed_key.frame_id != 0:
                raise ValueError('Physical inference ordinals must start at zero')
            self.last_id = seed_key.frame_id
            self.last_time_s = seed_key.timestamp_s
            self.last_valid_id = seed_key.frame_id
            self.last_valid_time_s = seed_key.timestamp_s

    def _check_backend_policy(self):
        if self.time_policy.mode != PHYSICAL:
            return
        supported = getattr(self.backend, 'clock_modes_supported', ())
        if (getattr(self.backend, 'time_policy', None) != self.time_policy or
                PHYSICAL not in supported or not callable(getattr(self.backend, 'clear_motion', None))):
            raise ValueError('Physical backend must declare this policy, PHYSICAL support and clear_motion')

    def _preflight_key(self, key):
        if not isinstance(key, FrameKey) or key.clock_mode != self.time_policy.mode:
            raise ValueError('Tracker clock mode mismatch')
        if not self.terminal:
            self._check_backend_policy()
        previous = self.terminal_key if self.terminal and self.terminal_key is not None else None
        previous_id = previous.frame_id if previous is not None else self.last_id
        previous_time = previous.timestamp_s if previous is not None else self.last_time_s
        if previous_id is not None and key.frame_id != previous_id + 1:
            raise ValueError('Chronological consecutive frames required')
        if self.time_policy.mode == PHYSICAL:
            if previous_id is None and key.frame_id != 0:
                raise ValueError('Physical inference ordinals must start at zero')
            if previous_time is not None and key.timestamp_s <= previous_time:
                raise ValueError('Physical timestamps must be strictly increasing')

    def _base_result(self, key, mask_state, reason=None, timings=None):
        result = dict(frameId=key.frame_id, cameraFromObject=None, mask_state=mask_state,
                      pose_state='lost', render_state='suppressed', failure_reason=reason,
                      timings_ms={} if timings is None else dict(timings), validation=None)
        if key.clock_mode == PHYSICAL:
            result['timestamp_s'] = key.timestamp_s
            result['clock_mode'] = key.clock_mode
        return result

    def _terminal_result(self, key, mask_state='unavailable', reason='motion_cleanup_unconfirmed', timings=None):
        self.terminal_key = key
        # Cleanup may have partially mutated a backend, so no accepted or
        # pending pose remains eligible for public output or later seeding.
        self.accepted = None
        self.pending = None
        self.pending_id = None
        self.pending_time_s = None
        if (self.time_policy.mode == PHYSICAL and self.last_valid_time_s is not None and
                key.timestamp_s - self.last_valid_time_s > self.time_policy.private_pose_memory_s):
            self.last_valid_pose = None
            self.last_valid_id = None
            self.last_valid_time_s = None
        result = self._base_result(key, mask_state, reason, timings)
        result['terminal'] = True
        result['unmeasured_failure'] = True
        return result

    def _clear_motion(self, key, reason):
        try:
            self.backend.clear_motion(key, reason=reason)
        except Exception:
            self.terminal = True
            return False
        return True

    def _advance_unavailable(self, key, reason, mask_state='unavailable', timings=None):
        if self.time_policy.mode == PHYSICAL:
            if self.terminal:
                return self._terminal_result(key, mask_state, 'motion_cleanup_unconfirmed', timings)
            if not self._clear_motion(key, reason):
                return self._terminal_result(key, mask_state, 'motion_cleanup_unconfirmed', timings)
        self.last_id = key.frame_id
        self.last_time_s = key.timestamp_s
        self.accepted = None
        self.pending = None
        self.pending_id = None
        self.pending_time_s = None
        if (self.time_policy.mode == PHYSICAL and self.last_valid_time_s is not None and
                key.timestamp_s - self.last_valid_time_s > self.time_policy.private_pose_memory_s):
            self.last_valid_pose = None
            self.last_valid_id = None
            self.last_valid_time_s = None
        result = self._base_result(key, mask_state, reason, timings)
        result['motion_cleared'] = self.time_policy.mode == PHYSICAL
        return result

    def update_unavailable(self, key: FrameKey, *, reason: str):
        """Advance a known source row without RGB or mask evidence."""
        if self.time_policy.mode != PHYSICAL:
            raise ValueError('Unavailable rows require the explicit physical-time adapter')
        if not isinstance(key, FrameKey):
            raise ValueError('update_unavailable requires a FrameKey')
        if not isinstance(reason, str) or not reason:
            raise ValueError('Unavailable row requires a reason')
        self._preflight_key(key)
        return self._advance_unavailable(key, reason)

    def update(self, frame, segmentation):
        if not isinstance(frame, Frame) or not isinstance(segmentation, MaskPrediction):
            raise ValueError('Tracker update requires a Frame and MaskPrediction')
        key = frame.key
        self._preflight_key(key)
        if segmentation.key != key:
            raise ValueError('Mask/frame key mismatch')
        if self.terminal:
            return self._terminal_result(key, segmentation.state,
                                         timings=segmentation.timings_ms)
        try:
            return self._update_after_preflight(frame, segmentation)
        except Exception:
            # A backend operation may fail after partially changing optical-flow
            # or model memory.  Physical chronology cannot safely resume from an
            # assumed cleanup state, so suppress this and every later row without
            # making another backend call.  Keep the old exception behavior for
            # authenticated legacy callers.
            if self.time_policy.mode != PHYSICAL:
                raise
            self.terminal = True
            return self._terminal_result(key, segmentation.state,
                                         'motion_cleanup_unconfirmed',
                                         segmentation.timings_ms)

    def _update_after_preflight(self, frame, segmentation):
        key = frame.key
        usable = segmentation.usable(frame.rgb.shape[:2])
        if self.time_policy.mode == PHYSICAL and not usable:
            return self._advance_unavailable(key, segmentation.reason or 'visible_mask_unavailable',
                                             segmentation.state, segmentation.timings_ms)

        expired = (self.time_policy.mode == PHYSICAL and (
            (self.last_valid_time_s is not None and
             key.timestamp_s - self.last_valid_time_s > self.time_policy.private_pose_memory_s) or
            (self.pending_time_s is not None and
             key.timestamp_s - self.pending_time_s > self.time_policy.private_pose_memory_s)))
        if expired:
            if not self._clear_motion(key, 'physical_pose_memory_expired'):
                return self._terminal_result(key, segmentation.state, timings=segmentation.timings_ms)
            self.accepted = None; self.pending = None
            self.pending_id = None; self.pending_time_s = None
            self.last_valid_pose = None; self.last_valid_id = None; self.last_valid_time_s = None

        self.last_id = key.frame_id
        self.last_time_s = key.timestamp_s
        if self.last_valid_pose is not None and self.last_valid_id is None:
            # Historical legacy seed behavior is authenticated by the runner.
            self.last_valid_id = frame.frame_id-1
        self.rejections = []
        timings = dict(segmentation.timings_ms)
        result = self._base_result(key, segmentation.state, segmentation.reason, timings)
        if hasattr(self.backend, 'observe'):
            begin = time.perf_counter()
            self.backend.observe(frame, segmentation.mask if usable else None)
            timings['model_motion_observation'] = (time.perf_counter()-begin)*1000
        if not usable:
            self.accepted = None; self.pending = None
            result['failure_reason'] = segmentation.reason or 'visible_mask_unavailable'
            return result
        start = time.perf_counter()
        recovery = self.accepted is None
        candidates = []
        if self.time_policy.mode == PHYSICAL:
            memory_live = (self.last_valid_time_s is not None and
                           key.timestamp_s - self.last_valid_time_s <= self.time_policy.private_pose_memory_s)
        else:
            memory_live = (self.last_valid_id is not None and
                           frame.frame_id-self.last_valid_id <= self.settings.private_pose_memory_frames)
        if self.accepted is not None or self.pending is not None or memory_live:
            seed = self.accepted if self.accepted is not None else self.pending if self.pending is not None else self.last_valid_pose
            if self.accepted is None and hasattr(self.backend, 'motion_seed'):
                motion_arg = frame if self.time_policy.mode == PHYSICAL else frame.frame_id
                motion_seed = self.backend.motion_seed(motion_arg)
                if motion_seed is not None: seed = motion_seed
            candidates = [self.backend.refine(frame, segmentation.mask, seed)]
        evaluated = self._evaluate(candidates, frame, segmentation.mask)
        if not evaluated:
            recovery = True; self.accepted = None
            begin = time.perf_counter()
            hypotheses = self.backend.recover(frame, segmentation.mask, top_k=5)
            timings['recovery'] = (time.perf_counter()-begin)*1000
            candidates = [self.backend.refine(frame, segmentation.mask, p) for p in hypotheses[:5]]
            evaluated = self._evaluate(candidates, frame, segmentation.mask)
        timings['pose_total'] = (time.perf_counter()-start)*1000
        result['rejection_reasons'] = self.rejections.copy()
        if not evaluated:
            self.pending = None
            self.pending_id = None; self.pending_time_s = None
            result['failure_reason'] = 'implausible_pose_jump' if 'implausible_pose_jump' in self.rejections else 'no_valid_current_image_pose'
            if 'contradicts_observed_model_motion' in self.rejections:
                result['failure_reason'] = 'contradicts_observed_model_motion'
            if 'contradicts_rendered_texture' in self.rejections:
                result['failure_reason'] = 'contradicts_rendered_texture'
            return result
        evaluated.sort(key=lambda v: v[1]['score'], reverse=True)
        pose, stats = evaluated[0]; result['validation'] = stats
        if recovery and any(s['score'] >= stats['score']*self.settings.recovery_ambiguity_ratio and
                            distinct_pose(pose, other, self.settings) for other, s in evaluated[1:]):
            self.pending = None
            self.pending_id = None; self.pending_time_s = None
            result['failure_reason'] = 'ambiguous_recovery_hypotheses'
            return result
        if recovery and (self.pending is None or distinct_pose(pose, self.pending, self.settings)):
            self.pending = pose
            self.pending_id = frame.frame_id
            self.pending_time_s = frame.timestamp_s if self.time_policy.mode == PHYSICAL else None
            result.update(pose_state='recovering', failure_reason='awaiting_second_validated_pose')
            return result
        self.accepted = pose; self.pending = None
        self.pending_id = None; self.pending_time_s = None
        self.last_valid_pose = pose.copy(); self.last_valid_id = frame.frame_id
        if self.time_policy.mode == PHYSICAL:
            self.last_valid_time_s = key.timestamp_s
        if hasattr(self.backend, 'commit'):
            begin = time.perf_counter()
            self.backend.commit(frame, segmentation.mask, pose)
            timings['model_memory_refresh'] = (time.perf_counter()-begin)*1000
        result.update(cameraFromObject=pose.tolist(), pose_state='tracking', render_state='visible', failure_reason=None)
        return result

    def _evaluate(self, candidates, frame, mask):
        good = []
        for c in candidates:
            if c is None: continue
            ok, reason, stats = validate(c, frame, mask, self.settings)
            if not ok:
                self.rejections.append(reason)
                continue
            pose = checked_pose(c.pose)
            if hasattr(self.backend, 'validate_motion'):
                motion_arg = frame if self.time_policy.mode == PHYSICAL else frame.frame_id
                motion_ok, motion_reason, motion_stats = self.backend.validate_motion(c, motion_arg)
                stats['model_motion'] = motion_stats
                if not motion_ok:
                    self.rejections.append(motion_reason)
                    continue
            if self.last_valid_id is not None:
                if self.time_policy.mode == PHYSICAL:
                    elapsed = frame.timestamp_s - self.last_valid_time_s
                    memory_live = elapsed <= self.time_policy.private_pose_memory_s
                else:
                    gap = frame.frame_id-self.last_valid_id
                    elapsed = None
                    memory_live = gap <= self.settings.private_pose_memory_frames
                if memory_live:
                    angle = np.degrees(np.arccos(np.clip((np.trace(self.last_valid_pose[:3, :3].T@pose[:3, :3])-1)/2, -1, 1)))
                    distance = float(np.linalg.norm(self.last_valid_pose[:3, 3]-pose[:3, 3]))
                    if self.time_policy.mode == PHYSICAL:
                        stats.update(rotation_from_last_valid_degrees=float(angle),
                                     translation_from_last_valid_m=distance,
                                     seconds_since_last_valid=elapsed)
                        max_angle = self.settings.rotation_margin_degrees + self.time_policy.max_angular_rate_deg_s*elapsed
                        max_distance = self.settings.translation_margin_m + self.time_policy.max_translation_rate_m_s*elapsed
                    else:
                        stats.update(rotation_from_last_valid_degrees=float(angle), translation_from_last_valid_m=distance,
                                     frames_since_last_valid=gap)
                        max_angle = self.settings.rotation_margin_degrees + gap*self.settings.max_rotation_degrees_per_frame
                        max_distance = self.settings.translation_margin_m + gap*self.settings.max_translation_m_per_frame
                    # The physical limits are inclusive. Rotation reconstruction
                    # through trace/arccos can exceed an exact boundary by a few
                    # ulps. Preserve the historical legacy comparison exactly.
                    if self.time_policy.mode == PHYSICAL:
                        angle_ok = (angle <= max_angle or
                                    np.isclose(angle, max_angle, rtol=1e-12, atol=1e-9))
                        distance_ok = (distance <= max_distance or
                                       np.isclose(distance, max_distance, rtol=1e-12, atol=1e-12))
                    else:
                        angle_ok = angle <= max_angle
                        distance_ok = distance <= max_distance
                    if not angle_ok or not distance_ok:
                        self.rejections.append('implausible_pose_jump')
                        continue
            good.append((pose, stats))
        return good
