"""Versioned RGB-only wrapper for the initialized rigid tracker.

The legacy :class:`RigidTracker` API remains available for existing experiments.
New streaming callers should use ``PrototypeTrackerV1`` so packet identity,
calibration, recovery and public-pose visibility are checked in one place.
"""
from dataclasses import dataclass
import numpy as np

from .tracker import RigidTracker, TrackingResult, reconcile_flow_fresh
from .vision import cv2


API_VERSION = "prototype-tracking/1"
FRAME_INDEX_CLOCK = "frame_index_no_fps"
PHYSICAL_CLOCK = "physical"


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _identity_value(value):
    if isinstance(value, str):
        return bool(value.strip())
    return _is_int(value) and int(value) >= 0


def _matrix3(value, label):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f'{label} must be a finite 3x3 matrix.')
    return matrix


def _canonical_intrinsics(k):
    matrix = _matrix3(k, 'Intrinsics')
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or
            not np.allclose(matrix[0, 1], 0, atol=1e-10, rtol=0) or
            not np.allclose(matrix[1, 0], 0, atol=1e-10, rtol=0) or
            not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0) or
            abs(float(np.linalg.det(matrix))) <= 1e-12):
        raise ValueError('Intrinsics must be a nonsingular pinhole matrix with positive focal lengths.')
    return matrix


def _pixel_transform(value):
    if value is None:
        return np.eye(3, dtype=np.float64)
    matrix = _matrix3(value, 'native_from_input')
    if (not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0) or
            not np.allclose(matrix[0, 1], 0, atol=1e-10, rtol=0) or
            not np.allclose(matrix[1, 0], 0, atol=1e-10, rtol=0) or
            matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or
            abs(float(np.linalg.det(matrix))) <= 1e-12):
        raise ValueError('native_from_input must describe positive axis-aligned crop/resize geometry.')
    return matrix


def crop_resize_native_from_input(native_size, crop_xywh, output_size):
    """Return the OpenCV-center transform from resized/cropped pixels to native pixels.

    ``native_size`` and ``output_size`` are ``(width, height)``. ``crop_xywh``
    uses native pixel-center coordinates and pixel counts. The matching forward
    transform is ``u' = sx*(u-x0+.5)-.5`` and likewise for ``v``.
    """
    if (not isinstance(native_size, (tuple, list)) or len(native_size) != 2 or
            not all(_is_int(d) and int(d) > 0 for d in native_size)):
        raise ValueError('native_size must contain positive integer width and height.')
    if (not isinstance(output_size, (tuple, list)) or len(output_size) != 2 or
            not all(_is_int(d) and int(d) > 0 for d in output_size)):
        raise ValueError('output_size must contain positive integer width and height.')
    crop = np.asarray(crop_xywh, dtype=np.float64)
    if crop.shape != (4,) or not np.isfinite(crop).all():
        raise ValueError('crop_xywh must contain finite x, y, width and height.')
    x0, y0, crop_width, crop_height = crop
    native_width, native_height = map(int, native_size)
    output_width, output_height = map(int, output_size)
    if (x0 < 0 or y0 < 0 or crop_width <= 0 or crop_height <= 0 or
            x0+crop_width > native_width or y0+crop_height > native_height):
        raise ValueError('Crop must be positive and stay within native image bounds.')
    sx, sy = output_width/crop_width, output_height/crop_height
    input_from_native = np.array([
        [sx, 0., sx*(.5-x0)-.5],
        [0., sy, sy*(.5-y0)-.5],
        [0., 0., 1.],
    ], dtype=np.float64)
    return np.linalg.inv(input_from_native)


def crop_resize_input_from_native(native_size, crop_xywh, output_size):
    """Return the OpenCV-center transform from native pixels to cropped/resized pixels."""
    return np.linalg.inv(crop_resize_native_from_input(native_size, crop_xywh, output_size))


def intrinsics_for_input(k_native, native_from_input=None):
    """Propagate native OpenCV intrinsics into the supplied crop/resize image."""
    k = _canonical_intrinsics(k_native)
    native_from_input_matrix = _pixel_transform(native_from_input)
    result = np.linalg.inv(native_from_input_matrix)@k
    if not np.isfinite(result).all() or result[0, 0] <= 0 or result[1, 1] <= 0:
        raise ValueError('Transformed input intrinsics are invalid.')
    return result


def transform_pixels(pixels, transform):
    """Apply an affine pixel-center transform to an ``Nx2`` array."""
    points = np.asarray(pixels, dtype=np.float64)
    matrix = _matrix3(transform, 'pixel transform')
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('pixels must be a finite Nx2 array.')
    homogeneous = np.column_stack([points, np.ones(len(points))])@matrix.T
    if not np.isfinite(homogeneous).all() or (np.abs(homogeneous[:, 2]) <= 1e-12).any():
        raise ValueError('Pixel transform produced an invalid coordinate.')
    return homogeneous[:, :2]/homogeneous[:, 2, None]


def project_points_cv(points, rotation, translation, k, near_metres=.03):
    """Project source-object metre points in CV camera axes (X right, Y down, Z forward)."""
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    k = _canonical_intrinsics(k)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all() or
            rotation.shape != (3, 3) or not is_proper_rotation(rotation) or
            translation.shape != (3,) or not np.isfinite(translation).all() or
            not np.isfinite(near_metres) or near_metres < 0):
        raise ValueError('Projection inputs must be finite metres and a proper CV pose.')
    camera = points@rotation.T+translation
    if not np.isfinite(camera).all() or (camera[:, 2] <= near_metres).any():
        raise ValueError('Projection contains points behind or inside the near plane.')
    homogeneous = camera@k.T
    pixels = homogeneous[:, :2]/homogeneous[:, 2, None]
    if not np.isfinite(pixels).all():
        raise ValueError('Projection produced nonfinite pixels.')
    return pixels, camera[:, 2].copy()


def is_proper_rotation(rotation, tolerance=1e-5):
    matrix = np.asarray(rotation, dtype=np.float64)
    return bool(matrix.shape == (3, 3) and np.isfinite(matrix).all() and
                np.allclose(matrix@matrix.T, np.eye(3), atol=tolerance, rtol=0) and
                abs(float(np.linalg.det(matrix))-1.) <= tolerance)


def cv_intrinsics_to_renderer(k_cv):
    """Convert integer-centered OpenCV K to the renderer's corner-centered K."""
    result = _canonical_intrinsics(k_cv).copy()
    result[0, 2] += .5
    result[1, 2] += .5
    return result


def raster_centers_to_cv(pixels):
    """Convert renderer raster coordinates ``(i+.5,j+.5)`` to OpenCV centers ``(i,j)``."""
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('Raster centers must be a finite Nx2 array.')
    return points-.5


def camera_from_source_to_bench(rotation_cv_from_source, translation_cv_from_source, bench_from_source):
    """Express a source-object camera mapping in bench coordinates, including reflections."""
    rotation = np.asarray(rotation_cv_from_source, dtype=np.float64)
    translation = np.asarray(translation_cv_from_source, dtype=np.float64).reshape(-1)
    transform = np.asarray(bench_from_source, dtype=np.float64)
    if (not is_proper_rotation(rotation) or translation.shape != (3,) or
            not np.isfinite(translation).all() or transform.shape != (4, 4) or
            not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0) or
            abs(float(np.linalg.det(transform[:3, :3]))) <= 1e-12):
        raise ValueError('Pose and source-to-bench basis must be finite and invertible.')
    camera_from_source = np.eye(4)
    camera_from_source[:3, :3] = rotation
    camera_from_source[:3, 3] = translation
    return camera_from_source@np.linalg.inv(transform)


def triangles_for_basis(triangles, source_to_bench):
    """Reverse face winding exactly when a basis conversion reflects orientation."""
    faces = np.asarray(triangles)
    transform = np.asarray(source_to_bench, dtype=np.float64)
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or
            not np.issubdtype(faces.dtype, np.integer) or transform.shape != (4, 4) or
            not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0) or
            abs(float(np.linalg.det(transform[:3, :3]))) <= 1e-12):
        raise ValueError('Triangle indices and source-to-bench transform are invalid.')
    return faces[:, [0, 2, 1]].copy() if np.linalg.det(transform[:3, :3]) < 0 else faces.copy()


@dataclass(frozen=True)
class FrameInputV1:
    """Whitelisted tracker input; evaluator annotations and depth are not fields."""
    session_id: object
    object_id: object
    reference_id: object
    frame_id: object
    logical_index: object
    timestamp_seconds: object
    timestamp_kind: object
    rgb: object
    k_native: object
    native_size: object
    native_from_input: object = None

    @classmethod
    def from_mapping(cls, mapping):
        allowed = {
            'sessionId', 'objectId', 'referenceId', 'frameId', 'logicalIndex',
            'timestampSeconds', 'timestampKind', 'rgb', 'KNative', 'nativeSize',
            'nativeFromInput',
        }
        if not isinstance(mapping, dict) or set(mapping)-allowed:
            raise ValueError('Frame mapping contains unsupported tracker fields.')
        required = {'sessionId', 'objectId', 'referenceId', 'frameId', 'logicalIndex',
                    'timestampKind', 'rgb', 'KNative', 'nativeSize'}
        if not required.issubset(mapping):
            raise ValueError('Frame mapping is missing required tracker fields.')
        return cls(
            mapping['sessionId'], mapping['objectId'], mapping['referenceId'],
            mapping['frameId'], mapping['logicalIndex'], mapping.get('timestampSeconds'),
            mapping['timestampKind'], mapping['rgb'], mapping['KNative'],
            mapping['nativeSize'], mapping.get('nativeFromInput'),
        )


@dataclass(frozen=True)
class TrackingOutputV1:
    session_id: object
    object_id: object
    reference_id: object
    frame_id: object
    logical_index: object
    timestamp_seconds: object
    timestamp_kind: object
    state: str
    reason: str | None
    camera_from_object: np.ndarray | None
    statistics: dict

    def manifest(self):
        matrix = None
        if self.state == 'tracking' and self.camera_from_object is not None:
            matrix = np.asarray(self.camera_from_object, dtype=np.float64).tolist()
        return {
            **self.statistics,
            'apiVersion': API_VERSION,
            'sessionId': self.session_id,
            'objectId': self.object_id,
            'referenceId': self.reference_id,
            'frameId': self.frame_id,
            'logicalIndex': self.logical_index,
            'timestampSeconds': self.timestamp_seconds,
            'timestampKind': self.timestamp_kind,
            'state': self.state,
            'failureReason': self.reason,
            'cameraFromObject': matrix,
            'units': 'metres',
            'cameraConvention': 'CV X right, Y down, Z forward; column vectors',
        }


@dataclass
class _Candidate:
    frame_id: int
    logical_index: int
    gray: np.ndarray
    input_pixels: np.ndarray
    native_pixels: np.ndarray
    points: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    statistics: dict


class PrototypeTrackerV1:
    """Strict stream wrapper around fixed onboarding references and legacy ORB/PnP.

    Lost frames are searched from the immutable onboarding descriptor bank. A pose
    becomes public only after two consecutive descriptor/PnP candidates agree with
    forward-backward optical flow of the first candidate's inlier anchors.
    """

    def __init__(self, reference_points, descriptors, model_positions, *,
                 session_id, object_id, reference_id, onboarding_seed=None,
                 seed_pixels=None, seed_points=None):
        if not isinstance(session_id, str) or not session_id.strip() or not _identity_value(object_id) or not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError('Session, object and reference bindings must be explicit identifiers.')
        supplied_seed = (onboarding_seed is not None, seed_pixels is not None, seed_points is not None)
        if any(supplied_seed) and not all(supplied_seed):
            raise ValueError('An onboarding seed requires its frame, pixels and source anchors together.')
        self.session_id = session_id
        self.object_id = object_id
        self.reference_id = reference_id
        self.rigid = RigidTracker(reference_points, descriptors, model_positions)
        self._pending = None
        self._active = False
        self._calibration_key = None
        self._last_seen_frame_id = None
        self._last_seen_logical_index = None
        self._last_physical_timestamp = None
        self._clock_kind = None
        self._last_sequence_frame_id = None
        self._last_sequence_logical_index = None
        if onboarding_seed is not None:
            seed = self._validate_seed(onboarding_seed, seed_pixels, seed_points)
            self._calibration_key = seed[0]
            self._last_seen_frame_id = self._last_sequence_frame_id = int(onboarding_seed.frame_id)
            self._last_seen_logical_index = self._last_sequence_logical_index = int(onboarding_seed.logical_index)
            self._clock_kind = onboarding_seed.timestamp_kind
            if onboarding_seed.timestamp_seconds is not None:
                self._last_physical_timestamp = float(onboarding_seed.timestamp_seconds)
            self.rigid.previous_gray = seed[1]
            self.rigid.previous_pixels = seed[2]
            self.rigid.previous_points = seed[3]
            self.rigid.lost = False
            self._active = True

    def _validate_seed(self, frame, seed_pixels, seed_points):
        prepared = self._prepare_frame(frame, check_identity=True)
        if isinstance(prepared, str):
            raise ValueError(f'Invalid onboarding seed: {prepared}.')
        key, gray, _, _, _ = prepared
        pixels = np.asarray(seed_pixels, dtype=np.float64)
        points = np.asarray(seed_points, dtype=np.float64)
        if (pixels.ndim != 2 or pixels.shape[1:] != (2,) or points.shape != (len(pixels), 3) or
                len(points) < 12 or not np.isfinite(pixels).all() or not np.isfinite(points).all()):
            raise ValueError('Onboarding seed pixels and source anchors must be finite and paired.')
        if ((pixels[:, 0] < 0).any() or (pixels[:, 1] < 0).any() or
                (pixels[:, 0] >= gray.shape[1]).any() or (pixels[:, 1] >= gray.shape[0]).any()):
            raise ValueError('Onboarding seed pixels must lie inside the seed image.')
        bank = {point.tobytes() for point in self.rigid.reference}
        if any(point.tobytes() not in bank for point in points):
            raise ValueError('Onboarding seed anchors must retain exact frozen reference identity.')
        return key, gray, pixels.copy(), points.copy()

    def reset_temporal(self):
        """Clear image flow and recovery evidence while retaining identity high-watermarks."""
        self.rigid.reset_temporal()
        self._pending = None
        self._active = False

    def _prepare_frame(self, frame, check_identity=True):
        if not isinstance(frame, FrameInputV1):
            return 'invalid_frame_packet'
        if check_identity and (frame.session_id != self.session_id or
                               type(frame.object_id) is not type(self.object_id) or frame.object_id != self.object_id or
                               frame.reference_id != self.reference_id):
            return 'identity_mismatch'
        if (not _is_int(frame.frame_id) or int(frame.frame_id) < 0 or
                not _is_int(frame.logical_index) or int(frame.logical_index) < 0):
            return 'invalid_frame_order'
        if frame.timestamp_kind == FRAME_INDEX_CLOCK:
            if frame.timestamp_seconds is not None:
                return 'invalid_clock_metadata'
        elif frame.timestamp_kind == PHYSICAL_CLOCK:
            if (isinstance(frame.timestamp_seconds, (bool, np.bool_)) or
                    not isinstance(frame.timestamp_seconds, (int, float, np.integer, np.floating)) or
                    not np.isfinite(frame.timestamp_seconds)):
                return 'invalid_timestamp'
        else:
            return 'invalid_clock_metadata'
        if (not isinstance(frame.native_size, (tuple, list)) or len(frame.native_size) != 2 or
                not all(_is_int(d) and int(d) > 0 for d in frame.native_size)):
            return 'invalid_calibration'
        try:
            k_native = _canonical_intrinsics(frame.k_native)
            native_from_input = _pixel_transform(frame.native_from_input)
            k_input = np.linalg.inv(native_from_input)@k_native
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return 'invalid_calibration'
        rgb = np.asarray(frame.rgb) if isinstance(frame.rgb, np.ndarray) else None
        if (rgb is None or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or
                rgb.shape[0] <= 0 or rgb.shape[1] <= 0):
            return 'invalid_rgb_frame'
        native_width, native_height = map(int, frame.native_size)
        if frame.native_from_input is None and (rgb.shape[1] != native_width or rgb.shape[0] != native_height):
            return 'calibration_resolution_mismatch'
        # Validate pixel-edge support, including explicit identity transforms. A
        # crop/resize must describe actual pixels within the calibrated source.
        image_corners = np.array([[-.5, -.5], [rgb.shape[1]-.5, -.5],
                                  [-.5, rgb.shape[0]-.5],
                                  [rgb.shape[1]-.5, rgb.shape[0]-.5]])
        try:
            native_corners = transform_pixels(image_corners, native_from_input)
        except ValueError:
            return 'invalid_calibration'
        if ((native_corners[:, 0] < -.5-1e-6).any() or (native_corners[:, 1] < -.5-1e-6).any() or
                (native_corners[:, 0] > native_width-.5+1e-6).any() or
                (native_corners[:, 1] > native_height-.5+1e-6).any()):
            return 'calibration_resolution_mismatch'
        if not np.isfinite(k_input).all() or k_input[0, 0] <= 0 or k_input[1, 1] <= 0:
            return 'invalid_calibration'
        # Input pixels are integer-centered; B maps them back to native centers.
        key = (native_width, native_height, tuple(k_native.ravel()),
               tuple(native_from_input.ravel()), rgb.shape[1], rgb.shape[0])
        owned_rgb = np.array(rgb, copy=True, order='C')
        try:
            gray = cv2.cvtColor(owned_rgb, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            return 'invalid_rgb_frame'
        return key, gray, k_native, native_from_input, np.linalg.inv(native_from_input)

    def _output(self, frame, state, reason, rotation=None, translation=None, statistics=None):
        matrix = None
        if state == 'tracking' and rotation is not None and translation is not None:
            translation = np.asarray(translation, dtype=np.float64).reshape(-1)
            if is_proper_rotation(rotation) and translation.shape == (3,) and np.isfinite(translation).all():
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, :3] = rotation
                matrix[:3, 3] = translation
            else:
                state, reason, matrix = 'lost', 'invalid_or_improper_rotation', None
                self.reset_temporal()
        elif state == 'tracking':
            state, reason = 'lost', 'missing_tracking_pose'
            self.reset_temporal()
        return TrackingOutputV1(
            getattr(frame, 'session_id', self.session_id),
            getattr(frame, 'object_id', self.object_id),
            getattr(frame, 'reference_id', self.reference_id),
            getattr(frame, 'frame_id', None), getattr(frame, 'logical_index', None),
            getattr(frame, 'timestamp_seconds', None), getattr(frame, 'timestamp_kind', None),
            state, reason, matrix, dict(statistics or {}),
        )

    def _failed_frame(self, frame, reason):
        self.reset_temporal()
        return self._output(frame, 'lost', reason)

    def _advance_identity_clock(self, frame):
        """Advance high-watermarks before payload validation to reject replayed packets."""
        if (not isinstance(frame, FrameInputV1) or frame.session_id != self.session_id or
                type(frame.object_id) is not type(self.object_id) or frame.object_id != self.object_id or
                frame.reference_id != self.reference_id):
            return 'identity_mismatch', False
        if (not _is_int(frame.frame_id) or not _is_int(frame.logical_index) or
                int(frame.frame_id) < 0 or int(frame.logical_index) < 0):
            return 'invalid_frame_order', False
        frame_id, logical_index = int(frame.frame_id), int(frame.logical_index)
        previous_frame = self._last_seen_frame_id
        previous_logical = self._last_seen_logical_index
        ordered = ((previous_frame is None or frame_id > previous_frame) and
                   (previous_logical is None or logical_index > previous_logical))
        clock_changed = self._clock_kind is not None and frame.timestamp_kind != self._clock_kind
        if clock_changed:
            ordered = False
        elif self._clock_kind is None and frame.timestamp_kind in (FRAME_INDEX_CLOCK, PHYSICAL_CLOCK):
            self._clock_kind = frame.timestamp_kind
        self._last_seen_frame_id = frame_id if previous_frame is None else max(previous_frame, frame_id)
        self._last_seen_logical_index = logical_index if previous_logical is None else max(previous_logical, logical_index)
        if frame.timestamp_kind == PHYSICAL_CLOCK and isinstance(frame.timestamp_seconds, (int, float, np.integer, np.floating)) and not isinstance(frame.timestamp_seconds, (bool, np.bool_)) and np.isfinite(frame.timestamp_seconds):
            timestamp = float(frame.timestamp_seconds)
            if self._last_physical_timestamp is not None and timestamp <= self._last_physical_timestamp:
                ordered = False
            self._last_physical_timestamp = timestamp if self._last_physical_timestamp is None else max(self._last_physical_timestamp, timestamp)
        if clock_changed:
            return 'clock_kind_changed', ordered
        return (None if ordered else 'out_of_order_frame'), ordered

    @staticmethod
    def _deduplicate(pixels, points):
        if not len(points):
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        _, unique = np.unique(points, axis=0, return_index=True)
        unique.sort()
        return np.asarray(pixels)[unique], np.asarray(points)[unique]

    def _solve_candidate(self, gray, k_native, native_from_input, input_from_native, frame_id):
        pixels_input, points = self.rigid.match(gray)
        pixels_native = transform_pixels(pixels_input, native_from_input)
        solved = self.rigid.solve(pixels_native, points, k_native, frame_id)
        if not isinstance(solved, tuple):
            return solved
        result, inlier_pixels_native, inlier_points = solved
        return _Candidate(
            int(frame_id), -1, gray.copy(), transform_pixels(inlier_pixels_native, input_from_native),
            inlier_pixels_native.copy(), inlier_points.copy(), result.rotation.copy(),
            result.translation.copy(), dict(result.statistics),
        )

    def _temporal_consistency(self, pending, current, k_native, native_from_input):
        original = np.asarray(pending.input_pixels, dtype=np.float32).reshape(-1, 1, 2)
        empty = dict(temporalLinks=0, temporalConsistent=0, consistencyFraction=0.,
                     consistencyMedianErrorPixels=None, consistencyP95ErrorPixels=None)
        expected = len(original)
        try:
            flowed, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                pending.gray, current.gray, original, None, winSize=(21, 21), maxLevel=3)
            if flowed is None or forward_status is None:
                return False, empty
            # Validate the forward result before feeding it to a backward pass.
            # A malformed/truncated LK result is a lost consensus, not an exception.
            flowed = np.asarray(flowed)
            forward_status = np.asarray(forward_status).reshape(-1)
            if flowed.size != 2*expected or forward_status.size != expected:
                return False, empty
            flowed = flowed.reshape(-1, 1, 2)
            returned, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                current.gray, pending.gray, flowed, None, winSize=(21, 21), maxLevel=3)
            if returned is None or reverse_status is None:
                return False, empty
            returned = np.asarray(returned)
            reverse_status = np.asarray(reverse_status).reshape(-1)
            if returned.size != 2*expected or reverse_status.size != expected:
                return False, empty
        except cv2.error:
            return False, empty
        except (TypeError, ValueError, IndexError):
            return False, empty
        flowed = flowed.reshape(-1, 2)
        returned = returned.reshape(-1, 2)
        status = forward_status == 1
        status &= reverse_status == 1
        round_trip = returned-original.reshape(-1, 2)
        round_trip_native = round_trip@native_from_input[:2, :2].T
        status &= np.isfinite(flowed).all(axis=1)&np.isfinite(round_trip_native).all(axis=1)
        status &= np.linalg.norm(round_trip_native, axis=1) <= 1.
        status &= (flowed[:, 0] >= 0)&(flowed[:, 1] >= 0)
        status &= (flowed[:, 0] < current.gray.shape[1])&(flowed[:, 1] < current.gray.shape[0])
        count = int(status.sum())
        stats = dict(empty)
        stats['temporalLinks'] = count
        if count < 12:
            return False, stats
        linked_native = transform_pixels(flowed[status], native_from_input)
        anchors = pending.points[status]
        try:
            projected_native, _ = project_points_cv(anchors, current.rotation, current.translation, k_native)
        except ValueError:
            return False, stats
        errors = np.linalg.norm(projected_native-linked_native, axis=1)
        if not np.isfinite(errors).all():
            return False, stats
        consistent = errors <= 6.
        fraction = float(consistent.mean())
        median = float(np.median(errors))
        p95 = float(np.percentile(errors, 95))
        stats.update(temporalConsistent=int(consistent.sum()), consistencyFraction=fraction,
                     consistencyMedianErrorPixels=median, consistencyP95ErrorPixels=p95)
        passed = bool(fraction >= .5 and median <= 3. and p95 <= 6. and
                      current.statistics.get('featureCoverage', 0) >= .2)
        return passed, stats

    def _cache_active(self, candidate, gray, input_from_native):
        self.rigid.previous_gray = gray.copy()
        self.rigid.previous_pixels = transform_pixels(candidate.native_pixels, input_from_native)
        self.rigid.previous_points = candidate.points.copy()
        self.rigid.lost = False
        self._active = True
        self._pending = None

    def update(self, frame):
        order_error, ordered = self._advance_identity_clock(frame)
        if order_error is not None:
            return self._failed_frame(frame, order_error)
        frame_id = int(frame.frame_id)
        logical_index = int(frame.logical_index)
        previous_frame = self._last_sequence_frame_id
        previous_logical = self._last_sequence_logical_index
        is_consecutive = (previous_frame is None or frame_id == previous_frame+1) and (
            previous_logical is None or logical_index == previous_logical+1)
        self._last_sequence_frame_id = frame_id
        self._last_sequence_logical_index = logical_index
        prepared = self._prepare_frame(frame)
        if isinstance(prepared, str):
            return self._failed_frame(frame, prepared)
        key, gray, k_native, native_from_input, input_from_native = prepared
        calibration_changed = self._calibration_key is not None and key != self._calibration_key
        if calibration_changed or not is_consecutive:
            self.reset_temporal()
        self._calibration_key = key

        if self._active:
            flow_pixels, flow_points = self.rigid.flow(gray, pixel_to_native=native_from_input)
            fresh_pixels, fresh_points = np.empty((0, 2)), np.empty((0, 3))
            if self.rigid.iteration%5 == 0 or len(flow_points) < 12:
                fresh_pixels, fresh_points = self.rigid.match(gray)
            native_pixels, points, proposals, fusion = reconcile_flow_fresh(
                flow_pixels, flow_points, fresh_pixels, fresh_points,
                k_native, frame_id, self.rigid.solve, pixel_to_native=native_from_input)
            if fusion['fusionFailure'] is not None:
                solved = TrackingResult(frame_id, 'lost', fusion['fusionFailure'], None, None, dict(fusion))
            else:
                solved = self.rigid.solve(native_pixels, points, k_native, frame_id,
                                          proposal_count=proposals)
                result_object = solved[0] if isinstance(solved, tuple) else solved
                result_object.statistics.update(fusion)
            self.rigid.iteration += 1
            if isinstance(solved, tuple):
                result, inlier_pixels_native, inlier_points = solved
                self.rigid.previous_gray = gray.copy()
                self.rigid.previous_pixels = transform_pixels(inlier_pixels_native, input_from_native)
                self.rigid.previous_points = inlier_points.copy()
                self.rigid.lost = False
                return self._output(frame, 'tracking', None, result.rotation, result.translation, result.statistics)
            self.reset_temporal()
            return self._output(frame, solved.state, solved.reason, statistics=solved.statistics)

        candidate = self._solve_candidate(gray, k_native, native_from_input, input_from_native, frame_id)
        if not isinstance(candidate, _Candidate):
            self._pending = None
            self.rigid.reset_temporal()
            return self._output(frame, candidate.state, candidate.reason, statistics=candidate.statistics)
        candidate.logical_index = logical_index
        if self._pending is None or candidate.frame_id != self._pending.frame_id+1 or candidate.logical_index != self._pending.logical_index+1:
            self._pending = candidate
            statistics = dict(candidate.statistics)
            statistics['recoveryCandidates'] = 1
            reason = 'awaiting_recovery_consensus'
            if previous_frame is not None and not is_consecutive:
                reason = 'recovery_restarted_after_frame_gap'
            elif calibration_changed:
                reason = 'recovery_restarted_after_calibration_change'
            return self._output(frame, 'recovering', reason, statistics=statistics)

        passed, consistency = self._temporal_consistency(self._pending, candidate, k_native, native_from_input)
        candidate.statistics.update(consistency)
        candidate.statistics['recoveryCandidates'] = 2
        if not passed:
            self._pending = candidate
            return self._output(frame, 'recovering', 'recovery_consensus_failed', statistics=candidate.statistics)
        self._cache_active(candidate, gray, input_from_native)
        return self._output(frame, 'tracking', None, candidate.rotation, candidate.translation, candidate.statistics)

