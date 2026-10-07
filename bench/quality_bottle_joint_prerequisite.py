"""Frozen, CPU-only Stage A1 preparation for the bottle joint-pose study.

This module declares inputs, coordinate adapters, source/query partitions and
geometric support.  It deliberately contains no matching loop, optimizer,
photometric scoring path, production renderer or neural entry point.  A1 cache
preparation and manifest writes require an explicit parent scheduling flag;
A2 and B are hard-closed in this revision.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache' / 'model-quality'
SPEC_PATH = ROOT / '.cache' / 'bottle-joint-photometric-spec-v1.md'
CAPTURE_ROOT = CACHE / 'diagnostics' / 'bottle-identity-v2'
ZERO_ROOT = CACHE / 'diagnostics' / 'bottle-zero-view-calibration-v3'
R5_ROOT = CACHE / 'diagnostics' / 'bottle-source-observability-pose-ablation-v1'
R5_TERMINAL = CACHE / 'diagnostics' / 'bottle-r5-parent-v1' / 'terminal.json'
R6_V1_ROOT = CACHE / 'diagnostics' / 'bottle-patch-pose-calibration-v1'
R6_V2_ROOT = CACHE / 'diagnostics' / 'bottle-patch-pose-calibration-v2'
R6_V2_TERMINAL = CACHE / 'diagnostics' / 'bottle-r6-parent-v2' / 'terminal.json'
RANCH_INPUT_ROOT = CACHE / 'inputs' / 'ranch'
RANCH_INPUT_MANIFEST = RANCH_INPUT_ROOT / 'input.json'
RANCH_MESH = RANCH_INPUT_ROOT / 'object.glb'

SPEC_SHA256 = '82b7add0ddf5dcde101139979186511487b4e3f6e6b7f10fc2778f2b33410d57'
R1_CAPTURE_SHA256 = '33efa32778c4094a5df0690d49658cb3bfb0be522204b3dc4e1279d9f0d03063'
R1_INPUT_MANIFEST_SHA256 = 'f012f28918534f2a7ac3efd59841a347a8bad384d9585130db88eaac56377d5d'
R1_MESH_SHA256 = 'ed3ac3767202c8da56d30051d766bf7d444d3d356c30671f4bbb2a73e3a1b321'
R3_REPORT_SHA256 = 'dccbff28e6ea84693fe2a64d40006304f318532daefcff955646827b6129819d'
R5_REPORT_SHA256 = 'a394bf54f51ddf527549ee5b3b6c7e8f623546b2c277a49051df2ce1c4d95259'
R5_TERMINAL_SHA256 = '7d193b09d6d9dda241e039fce757b0a19d5c82e0fc6bc84b287d73e8915392ae'
R6_V1_REPORT_SHA256 = 'b03c83967efc84f818a504f284ce1c1647271e2dda3b61d1e74f8eff27aa73c'
R6_V2_REPORT_SHA256 = '857a52df0982adee17388e3de5cb82836ea2bbd0249ef878c4741cb909c341ad'
R6_V2_TERMINAL_SHA256 = '214ed8c051d7865d46bb254821a662dcb85b1a0e776d88ecc727e5da46c97ccf'

IMAGE_WIDTH = 280
IMAGE_HEIGHT = 280
K_FIXTURE = np.asarray(((800., 0., 140.), (0., 800., 140.), (0., 0., 1.)), dtype=np.float64)
RADIUS_M = 0.12
HALF_HEIGHT_M = 0.12
QUERY_YAW_DEG = 8.0
SOURCE_YAWS_DEG = (0.0, 180.0)
CHALLENGE_YAW_DEG = (-180.0, -90.0, -15.0, 15.0, 90.0, 180.0)
PATCH_RADIUS = 5
PATCH_SIZE = 11
EROSION_PIXELS = 6
SIGMA = 2.0
GAUSSIAN_KERNEL_SIZE = 17
MAX_WITNESSES = 32
MAX_WITNESSES_PER_CELL = 2
MIN_WITNESSES = 8
MAX_FIT_SAMPLES = 3000
MIN_FIT_SAMPLES = 24
MIN_VERIFY_CELLS = 3
MIN_VERIFY_HULL_FRACTION = 0.12
MAX_WORKING_BYTES = 128 * 1024**2
MAX_OUTPUT_BYTES = 8 * 1024**2
MAX_CACHE_BYTES = 8 * 1024**3
MAX_RAY_CHUNK = 256
RAY_PAIR_SCRATCH_BYTES = 256
GEOMETRY_EPSILON_ABS_M = 1e-5
GEOMETRY_EPSILON_DIAGONAL_FRACTION = 1e-4
CONTROL_IDS = (
    'distinct_markers', 'blank_constant', 'repeated_2d_periodic',
    'coherent_wrong_180_packet', 'query_occlusion', 'mask_leakage_surrogate',
    'global_affine', 'appearance_mismatch',
)
MANDATORY_CONTROL_PREPREREQUISITE_VARIANTS = (
    'distinct_markers', 'repeated_2d_periodic',
    'coherent_wrong_180_packet', 'global_affine',
)
EXPECTED_A1_BENCH_IMPORTS = (
    'bench', 'bench.glb_model', 'bench.model', 'bench.quality_assets',
    'bench.quality_bottle_identity_audit', 'bench.quality_bottle_identity_probe',
    'bench.quality_bottle_joint_prerequisite',
    'bench.quality_bottle_patch_pose_calibration', 'bench.quality_bottle_pose_ablation',
    'bench.quality_bottle_zero_view_probe', 'bench.quality_contract',
    'bench.quality_runner', 'bench.quality_texture_detail', 'bench.vision',
)
CARRIERS = (10, 50, 100)


class PrerequisiteError(ValueError):
    """A frozen Stage A1 precondition failed; no IDs may be retuned."""


class StageGateError(RuntimeError):
    """A stage is not scheduled or is not implemented in this A1-only code."""


class DependencyViolation(PrerequisiteError):
    """A proposed pixel read crosses its predeclared raw-pixel partition."""


class ResourceLimitError(PrerequisiteError):
    """A prospective allocation or private output exceeds its frozen budget."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def array_record(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                nbytes=int(array.nbytes), sha256=sha256_bytes(array.tobytes(order='C')))


def ordered_id_record(ids: Sequence[int] | np.ndarray) -> dict:
    values = np.asarray(ids, dtype='<i8').reshape(-1)
    return dict(count=int(len(values)), sha256=sha256_bytes(values.tobytes(order='C')),
                ordered_ids=values.tolist())


def _checked_bytes(shape: Sequence[int], dtype: object, *, live_bytes: int = 0,
                   limit_bytes: int = MAX_WORKING_BYTES, label: str = 'array') -> int:
    if any(int(size) < 0 for size in shape):
        raise ResourceLimitError(f'{label} has a negative shape')
    required = math.prod(int(size) for size in shape) * np.dtype(dtype).itemsize
    if int(live_bytes) + required > int(limit_bytes):
        raise ResourceLimitError(
            f'{label} would raise live buffers to {int(live_bytes) + required} bytes, '
            f'over {int(limit_bytes)}')
    return int(required)


def guarded_zeros(shape: Sequence[int], dtype: object, *, live_bytes: int = 0,
                  label: str = 'array') -> np.ndarray:
    _checked_bytes(shape, dtype, live_bytes=live_bytes, label=label)
    return np.zeros(tuple(int(x) for x in shape), dtype=dtype)


def guarded_empty(shape: Sequence[int], dtype: object, *, live_bytes: int = 0,
                  label: str = 'array') -> np.ndarray:
    _checked_bytes(shape, dtype, live_bytes=live_bytes, label=label)
    return np.empty(tuple(int(x) for x in shape), dtype=dtype)


def ndarray_tree_bytes(root: Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ResourceLimitError('Model-cache tree contains a symlink; cache size is not auditable')
        if path.is_file():
            total += int(path.stat().st_size)
            if total > MAX_CACHE_BYTES:
                return total
    return int(total)


def prospective_resource_budget(*, live_bytes: int, requested_bytes: int,
                                output_bytes: int = 0,
                                cache_bytes: int | None = None) -> dict:
    live_after = int(live_bytes) + int(requested_bytes)
    if min(live_bytes, requested_bytes, output_bytes) < 0:
        raise ResourceLimitError('Resource reservations must be nonnegative')
    if live_after > MAX_WORKING_BYTES:
        raise ResourceLimitError(f'working buffers need {live_after} bytes > {MAX_WORKING_BYTES}')
    if output_bytes > MAX_OUTPUT_BYTES:
        raise ResourceLimitError(f'private output needs {output_bytes} bytes > {MAX_OUTPUT_BYTES}')
    current_cache = ndarray_tree_bytes(CACHE) if cache_bytes is None else int(cache_bytes)
    if current_cache + output_bytes > MAX_CACHE_BYTES:
        raise ResourceLimitError(
            f'model cache plus prospective output needs {current_cache + output_bytes} bytes '
            f'> {MAX_CACHE_BYTES}')
    return dict(live_bytes=int(live_bytes), requested_bytes=int(requested_bytes),
                peak_bytes=live_after, working_budget_bytes=MAX_WORKING_BYTES,
                output_bytes=int(output_bytes), output_budget_bytes=MAX_OUTPUT_BYTES,
                current_model_cache_bytes=current_cache,
                model_cache_budget_bytes=MAX_CACHE_BYTES)


def reflect101(index: int, length: int) -> int:
    if int(length) <= 0:
        raise ValueError('Image dimension must be positive')
    if length == 1:
        return 0
    period = 2 * (int(length) - 1)
    value = int(index) % period
    return value if value < length else period - value


def gaussian_kernel_1d(size: int = GAUSSIAN_KERNEL_SIZE, sigma: float = SIGMA) -> np.ndarray:
    if int(size) != 17 or not math.isclose(float(sigma), 2.0, rel_tol=0., abs_tol=0.):
        raise ValueError('A1 freezes the 17-tap float32 sigma-2 Gaussian')
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel, dtype=np.float64)
    result = kernel.astype(np.float32)
    result.setflags(write=False)
    return result


def gaussian_blur_kernel_record() -> dict:
    kernel = gaussian_kernel_1d()
    outer = np.asarray(kernel[:, None] * kernel[None, :], dtype=np.float64)
    return dict(size=[17, 17], sigma=2.0, border='BORDER_REFLECT_101',
                coefficients_1d_f32=kernel.tolist(),
                coefficient_count=int(np.count_nonzero(kernel)),
                coefficients_2d_nonzero=int(np.count_nonzero(outer)),
                kernel_sha256=array_record(kernel)['sha256'],
                outer_kernel_sha256=array_record(outer)['sha256'])


def gaussian_raw_axis_dependencies(highpass_index: int, length: int) -> tuple[int, ...]:
    """Raw axis pixels used by one high-pass pixel under the frozen 17-tap blur."""
    return tuple(sorted({reflect101(int(highpass_index) + offset, int(length))
                         for offset in range(-8, 9)}))


def _bilinear_axis_taps(array_coordinate: float, length: int) -> tuple[tuple[int, float], ...]:
    """Return only nonzero taps; reject a partial footprint instead of clipping it."""
    coordinate = float(array_coordinate)
    if not math.isfinite(coordinate):
        return ()
    lower = math.floor(coordinate)
    fraction = coordinate - lower
    taps = []
    if 1.0 - fraction != 0.0:
        taps.append((lower, 1.0 - fraction))
    if fraction != 0.0:
        taps.append((lower + 1, fraction))
    if any(index < 0 or index >= int(length) for index, _ in taps):
        raise DependencyViolation('bilinear sample has a nonzero out-of-bounds tap')
    return tuple(taps)


def _bilinear_axis_dependencies(array_coordinate: float, length: int) -> tuple[int, ...]:
    return tuple(index for index, _ in _bilinear_axis_taps(array_coordinate, length))


def bilinear_raw_dependencies(center_xy: Sequence[float], width: int, height: int,
                              *, edge_coordinates: bool = True) -> tuple[int, ...]:
    """Nonzero raw RGB bilinear neighbors, in sorted row-major pixel IDs."""
    q = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(q).all():
        return ()
    x = float(q[0] - 0.5) if edge_coordinates else float(q[0])
    y = float(q[1] - 0.5) if edge_coordinates else float(q[1])
    xs = _bilinear_axis_dependencies(x, width)
    ys = _bilinear_axis_dependencies(y, height)
    return tuple(sorted(int(yy * width + xx) for yy in ys for xx in xs))


def highpass_patch_raw_dependencies(center_xy: Sequence[float], width: int,
                                    height: int, radius: int = PATCH_RADIUS) -> tuple[int, ...]:
    """Exact union of raw RGB pixels behind all nonzero high-pass patch taps."""
    xs, ys = dependency_axes(center_xy, width, height, radius)
    return tuple(sorted(int(y * width + x) for y in ys for x in xs))


def guarded_bilinear_sample(image: np.ndarray, center_xy: Sequence[float],
                            allowed_raw_pixels: np.ndarray, *,
                            edge_coordinates: bool = True) -> np.ndarray:
    """Sample only nonzero taps after proving every raw read is in the partition."""
    value = np.asarray(image)
    allowed = np.asarray(allowed_raw_pixels, dtype=bool)
    if value.ndim < 2 or value.shape[:2] != allowed.shape:
        raise ValueError('Image and raw-pixel partition shapes differ')
    q = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(q).all():
        raise DependencyViolation('bilinear sample center is nonfinite')
    shift = .5 if edge_coordinates else 0.
    x_taps = _bilinear_axis_taps(float(q[0] - shift), value.shape[1])
    y_taps = _bilinear_axis_taps(float(q[1] - shift), value.shape[0])
    if not x_taps or not y_taps:
        raise DependencyViolation('bilinear sample has no finite nonzero taps')
    deps = tuple(sorted(y * value.shape[1] + x for y, _ in y_taps for x, _ in x_taps))
    if not _mask_for_ids(allowed, deps):
        raise DependencyViolation('bilinear raw dependency footprint escapes its frozen partition')
    extra = (None,) * (value.ndim - 2)
    result = np.zeros(value.shape[2:] or (), dtype=np.result_type(value.dtype, np.float64))
    for y, wy in y_taps:
        for x, wx in x_taps:
            result = result + value[y, x] * (wx * wy)
    return result


def partition_sanitized_image(image: np.ndarray, allowed_raw_pixels: np.ndarray) -> np.ndarray:
    """Zero excluded raw pixels before any whole-image filtering operation."""
    value = np.asarray(image)
    allowed = np.asarray(allowed_raw_pixels, dtype=bool)
    if value.ndim < 2 or value.shape[:2] != allowed.shape:
        raise ValueError('Image and raw-pixel partition shapes differ')
    _checked_bytes(value.shape, value.dtype,
                   live_bytes=int(value.nbytes + allowed.nbytes),
                   label='partition-sanitized image')
    result = np.zeros_like(value)
    result[allowed] = value[allowed]
    return result


def sample_highpass_patch(highpass: np.ndarray, center_xy: Sequence[float],
                          allowed_raw_pixels: np.ndarray) -> np.ndarray:
    """Build the frozen 11×11 descriptor values with a pre-read D(q) gate."""
    image = np.asarray(highpass)
    allowed = np.asarray(allowed_raw_pixels, dtype=bool)
    require_dependencies_inside(center_xy, allowed, highpass_patch=True,
                                label='high-pass patch')
    patch = np.empty((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    q = np.asarray(center_xy, dtype=np.float64).reshape(2)
    for row, dy in enumerate(range(-PATCH_RADIUS, PATCH_RADIUS + 1)):
        for column, dx in enumerate(range(-PATCH_RADIUS, PATCH_RADIUS + 1)):
            patch[row, column] = guarded_bilinear_sample(
                image, q + (dx, dy), allowed, edge_coordinates=True)
    return patch


def dependency_axes(center_xy: Sequence[float], width: int, height: int,
                    radius: int = PATCH_RADIUS) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Cartesian factorization of D(q), preserving zero-weight exclusions."""
    q = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if int(radius) < 0:
        raise ValueError('High-pass patch radius cannot be negative')
    if not np.isfinite(q).all():
        return (), ()

    def axis_dependencies(coordinate: float, length: int) -> tuple[int, ...]:
        if int(length) <= 0:
            raise ValueError('Image dimension must be positive')
        first = float(coordinate - 0.5 - int(radius))
        last = float(coordinate - 0.5 + int(radius))
        first_lower = math.floor(first)
        last_lower = math.floor(last)
        last_fraction = last - last_lower
        last_upper = last_lower + int(last_fraction != 0.0)
        if first_lower < 0 or last_upper >= int(length):
            raise DependencyViolation('bilinear sample has a nonzero out-of-bounds tap')
        raw = range(first_lower - 8, last_upper + 8 + 1)
        return tuple(sorted({reflect101(index, int(length)) for index in raw}))

    return axis_dependencies(float(q[0]), width), axis_dependencies(float(q[1]), height)


def _mask_for_ids(mask: np.ndarray, ids: Sequence[int]) -> bool:
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    values = np.asarray(ids, dtype=np.int64).reshape(-1)
    return bool(len(values) and np.all((values >= 0) & (values < len(flat))) and flat[values].all())


def dependencies_inside_partition(center_xy: Sequence[float], partition: np.ndarray,
                                  *, highpass_patch: bool = True) -> bool:
    allowed = np.asarray(partition, dtype=bool)
    h, w = allowed.shape
    deps = (highpass_patch_raw_dependencies(center_xy, w, h) if highpass_patch else
            bilinear_raw_dependencies(center_xy, w, h))
    return _mask_for_ids(allowed, deps)


def require_dependencies_inside(center_xy: Sequence[float], partition: np.ndarray,
                                *, highpass_patch: bool = True,
                                label: str = 'candidate') -> tuple[int, ...]:
    allowed = np.asarray(partition, dtype=bool)
    h, w = allowed.shape
    dependencies = (highpass_patch_raw_dependencies(center_xy, w, h) if highpass_patch else
                    bilinear_raw_dependencies(center_xy, w, h))
    if not _mask_for_ids(allowed, dependencies):
        raise DependencyViolation(f'{label} raw dependency footprint escapes its frozen partition')
    return dependencies


def guarded_alternative_reads(centers_xy: Sequence[Sequence[float]], raw_rgb: np.ndarray,
                              partition: np.ndarray,
                              reader: Callable[[np.ndarray, Sequence[int]], object]) -> list[dict]:
    """Check every alternative before a callback can read query pixels.

    Low-NCC and losing alternatives belong in ``centers_xy`` too.  This helper
    makes no score and reads no pixel for a footprint that crosses the mask.
    """
    image = np.asarray(raw_rgb)
    allowed = np.asarray(partition, dtype=bool)
    if image.shape[:2] != allowed.shape:
        raise ValueError('RGB and raw-pixel partition shapes differ')
    sanitized = partition_sanitized_image(image, allowed)
    outcomes = []
    for index, center in enumerate(centers_xy):
        try:
            dependencies = require_dependencies_inside(
                center, allowed, highpass_patch=True, label=f'alternative[{index}]')
        except DependencyViolation as error:
            try:
                dependency_count = len(highpass_patch_raw_dependencies(
                    center, image.shape[1], image.shape[0]))
            except DependencyViolation:
                dependency_count = None
            outcomes.append(dict(index=index, state='rejected_before_read',
                                 dependency_count=dependency_count,
                                 reason=str(error)))
            continue
        # The callback receives only a copy with every forbidden raw pixel fixed
        # to zero, so ignoring the dependency list cannot leak the other split.
        value = reader(sanitized, dependencies)
        outcomes.append(dict(index=index, state='read_after_dependency_check',
                             dependency_count=len(dependencies), value=value))
    return outcomes


def raw_pixel_partitions(width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError('Partition dimensions must be positive')
    _checked_bytes((height, width), np.bool_, live_bytes=0, label='partition masks')
    y, x = np.indices((height, width), dtype=np.int64)
    cell_x = (4 * x) // int(width)
    cell_y = (4 * y) // int(height)
    verification = ((cell_x + cell_y) % 2) == 0
    fitting = ~verification
    return verification, fitting


def encode_bitmap(mask: np.ndarray) -> dict:
    value = np.asarray(mask, dtype=np.bool_)
    packed = np.packbits(value.reshape(-1), bitorder='little')
    return dict(shape=list(value.shape), order='C', bitorder='little', count=int(value.sum()),
                sha256=sha256_bytes(packed.tobytes()),
                packed_base64=base64.b64encode(packed.tobytes()).decode('ascii'))


def decode_bitmap(record: Mapping[str, object]) -> np.ndarray:
    shape = tuple(int(x) for x in record['shape'])
    raw = base64.b64decode(str(record['packed_base64']), validate=True)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')[:math.prod(shape)]
    return bits.reshape(shape).astype(bool)


def _integral(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.int64)
    return np.pad(value.cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64), ((1, 0), (1, 0)))


def _rect_sum(prefix: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> int:
    return int(prefix[y1 + 1, x1 + 1] - prefix[y0, x1 + 1] -
               prefix[y1 + 1, x0] + prefix[y0, x0])


def _integer_center_dep_rect(center_x: int, center_y: int, width: int,
                             height: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return dependency_axes((center_x + 0.5, center_y + 0.5), width, height)


def _axis_contiguous(values: Sequence[int]) -> tuple[int, int] | None:
    if not values:
        return None
    low, high = int(values[0]), int(values[-1])
    if tuple(values) != tuple(range(low, high + 1)):
        return None
    return low, high


def allowed_target_center_bitmap(observed_mask: np.ndarray, raw_partition: np.ndarray,
                                 eroded_mask: np.ndarray | None = None) -> np.ndarray:
    """Freeze centers using legacy all-four mask taps plus exact raw D(q)."""
    observed = np.asarray(observed_mask, dtype=bool)
    partition = np.asarray(raw_partition, dtype=bool)
    if observed.ndim != 2 or observed.shape != partition.shape:
        raise ValueError('Observed mask and partition must be equal 2D arrays')
    h, w = observed.shape
    if eroded_mask is None:
        # The cache path supplies the unchanged R5/OpenCV erosion.  The fixture
        # path can pass a previously frozen erosion; absence means an explicit
        # conservative 13x13 support check, never an inferred target score.
        eroded = _square_erode(observed, EROSION_PIXELS)
    else:
        eroded = np.asarray(eroded_mask, dtype=bool)
        if eroded.shape != observed.shape:
            raise ValueError('Eroded mask shape differs from observed mask')
    allowed = np.asarray(partition, dtype=bool)
    bad_prefix = _integral(~allowed)
    result = guarded_zeros((h, w), np.bool_, live_bytes=observed.nbytes + partition.nbytes +
                           eroded.nbytes + bad_prefix.nbytes,
                           label='target-center bitmap')
    eroded_prefix = _integral(~eroded)
    # R6's unchanged _bilinear_patch requires all four eroded-mask taps even
    # when a bilinear coefficient is zero. Integer half-pixel centers therefore
    # require the asymmetric [x-5,x+6]×[y-5,y+6] support.
    for y in range(PATCH_RADIUS, h - PATCH_RADIUS - 1):
        for x in range(PATCH_RADIUS, w - PATCH_RADIUS - 1):
            if _rect_sum(eroded_prefix, x - PATCH_RADIUS, x + PATCH_RADIUS + 1,
                         y - PATCH_RADIUS, y + PATCH_RADIUS + 1):
                continue
            axis_x, axis_y = _integer_center_dep_rect(x, y, w, h)
            rect_x, rect_y = _axis_contiguous(axis_x), _axis_contiguous(axis_y)
            if rect_x is None or rect_y is None:
                # This can occur only at REFLECT_101 borders; use the exact
                # Cartesian set when the reflected support is not a rectangle.
                dep = highpass_patch_raw_dependencies((x + .5, y + .5), w, h)
                if not _mask_for_ids(allowed, dep):
                    continue
            elif _rect_sum(bad_prefix, rect_x[0], rect_x[1], rect_y[0], rect_y[1]):
                continue
            result[y, x] = True
    return result


def _square_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Conservative square support for fixture-only tests without OpenCV."""
    value = np.asarray(mask, dtype=bool)
    h, w = value.shape
    if radius < 0:
        raise ValueError('Erosion radius cannot be negative')
    prefix = _integral(~value)
    result = guarded_zeros(value.shape, np.bool_, live_bytes=value.nbytes + prefix.nbytes,
                           label='fixture erosion mask')
    diameter = 2 * radius + 1
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            result[y, x] = _rect_sum(prefix, x-radius, x+radius, y-radius, y+radius) == 0
    return result


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def source_grid_cells(points_xy: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Map edge-coordinate source centers into the fixed 4x4 source bbox grid."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    bbox = _mask_bbox(observed_mask)
    if bbox is None:
        return np.full(len(points), -1, dtype=np.int16)
    x0, y0, x1, y1 = bbox
    finite = np.isfinite(points).all(axis=1)
    raw_x = np.zeros(len(points), dtype=np.int64)
    raw_y = np.zeros(len(points), dtype=np.int64)
    raw_x[finite] = np.floor(points[finite, 0]).astype(np.int64)
    raw_y[finite] = np.floor(points[finite, 1]).astype(np.int64)
    width, height = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    col = np.clip((4 * (raw_x - x0)) // width, 0, 3)
    row = np.clip((4 * (raw_y - y0)) // height, 0, 3)
    valid = finite & (raw_x >= x0) & (raw_x <= x1) & (raw_y >= y0) & (raw_y <= y1)
    return np.where(valid, row * 4 + col, -1).astype(np.int16)


def source_appearance_support(center_xy: Sequence[float], observed_mask: np.ndarray,
                              depth_mm: np.ndarray) -> dict:
    """Check mask and positive-depth support for every raw high-pass input pixel."""
    mask = np.asarray(observed_mask, dtype=bool)
    depth = np.asarray(depth_mm)
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError('Source mask and depth must be equal 2D arrays')
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(center).all():
        return dict(state='unsupported', reasons=['nonfinite_source_center'],
                    dependency_count=0, unsupported_mask_count=0,
                    invalid_depth_count=0, out_of_bounds=False)
    try:
        dependencies = highpass_patch_raw_dependencies(center, mask.shape[1], mask.shape[0])
    except DependencyViolation:
        return dict(state='unsupported', reasons=['dependency_out_of_bounds'],
                    dependency_count=0, unsupported_mask_count=0,
                    invalid_depth_count=0, out_of_bounds=True)
    if not dependencies:
        return dict(state='unsupported', reasons=['empty_dependency_footprint'],
                    dependency_count=0, unsupported_mask_count=0,
                    invalid_depth_count=0, out_of_bounds=False)
    indices = np.asarray(dependencies, dtype=np.int64)
    flat_mask = mask.reshape(-1)
    flat_depth = depth.reshape(-1)
    invalid_mask = ~flat_mask[indices]
    values = flat_depth[indices]
    invalid_depth = ~np.isfinite(values) | (values <= 0)
    reasons = []
    if np.any(invalid_mask):
        reasons.append('source_mask_false_raw_dependency')
    if np.any(invalid_depth):
        reasons.append('source_depth_invalid_raw_dependency')
    return dict(state='unsupported' if reasons else 'supported', reasons=reasons,
                dependency_count=int(len(dependencies)),
                unsupported_mask_count=int(invalid_mask.sum()),
                invalid_depth_count=int(invalid_depth.sum()), out_of_bounds=False)


def _source_appearance_support_for_ids(source_ids: Sequence[int], points_xy: np.ndarray,
                                       observed_mask: np.ndarray,
                                       depth_mm: np.ndarray, *,
                                       resident_bytes: int = 0) -> tuple[dict, np.ndarray]:
    """Freeze per-ID raw dependency eligibility without choosing replacements."""
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(observed_mask, dtype=bool)
    depth = np.asarray(depth_mm)
    if (len(ids) != len(points) or mask.ndim != 2 or depth.shape != mask.shape):
        raise ValueError('Source IDs/centers or source mask/depth shapes differ')
    if len(ids) and np.any(ids[:-1] >= ids[1:]):
        raise ValueError('Source IDs must be unique and increasing')
    height, width = mask.shape
    # The common interior D(q) is a 27x27 rectangle. Integral masks make the
    # all-pixel test constant-time per source ID; reflected border footprints
    # use the exact Cartesian axis set. Reserve both prefixes and their
    # construction temporaries before allocating them.
    _checked_bytes((5, height + 1, width + 1), np.int64,
                   live_bytes=max(int(resident_bytes), int(mask.nbytes + depth.nbytes +
                               points.nbytes + ids.nbytes)),
                   label='source appearance dependency prefixes')
    invalid_mask = ~mask
    invalid_depth = ~np.isfinite(depth) | (depth <= 0)
    invalid_mask_prefix = _integral(invalid_mask)
    invalid_depth_prefix = _integral(invalid_depth)
    support = np.zeros(len(ids), dtype=bool)
    reason_ids: dict[str, list[int]] = {}
    for row, source_id in enumerate(ids.tolist()):
        try:
            axis_x, axis_y = dependency_axes(points[row], width, height)
        except DependencyViolation:
            support[row] = False
            reason_ids.setdefault('dependency_out_of_bounds', []).append(int(source_id))
            continue
        if not axis_x or not axis_y:
            reason_ids.setdefault('empty_dependency_footprint', []).append(int(source_id))
            continue
        rect_x, rect_y = _axis_contiguous(axis_x), _axis_contiguous(axis_y)
        if rect_x is not None and rect_y is not None:
            mask_bad = _rect_sum(invalid_mask_prefix, rect_x[0], rect_x[1],
                                 rect_y[0], rect_y[1])
            depth_bad = _rect_sum(invalid_depth_prefix, rect_x[0], rect_x[1],
                                  rect_y[0], rect_y[1])
        else:
            index = np.ix_(np.asarray(axis_y, dtype=np.int64),
                           np.asarray(axis_x, dtype=np.int64))
            mask_bad = int(np.count_nonzero(invalid_mask[index]))
            depth_bad = int(np.count_nonzero(invalid_depth[index]))
        support[row] = mask_bad == 0 and depth_bad == 0
        if mask_bad:
            reason_ids.setdefault('source_mask_false_raw_dependency', []).append(int(source_id))
        if depth_bad:
            reason_ids.setdefault('source_depth_invalid_raw_dependency', []).append(int(source_id))
    supported_ids = ids[support]
    unsupported_ids = ids[~support]
    record = dict(
        state='computed', source_count=int(len(ids)),
        supported_count=int(support.sum()), unsupported_count=int((~support).sum()),
        supported_ids=supported_ids.tolist(),
        unsupported_ids=unsupported_ids.tolist(),
        unsupported_reason_ids={reason: values
                                for reason, values in sorted(reason_ids.items())},
        support_bits_sha256=array_record(support)['sha256'],
        support_rule='all nonzero raw pixels in exact high-pass D(q) are in original source mask and have finite positive source depth')
    return record, support


def _appearance_reason_map(record: Mapping[str, object]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for reason, id_record in record.get('unsupported_reason_ids', {}).items():
        source_ids = id_record.get('ordered_ids', []) if isinstance(id_record, Mapping) else id_record
        for source_id in source_ids:
            result.setdefault(int(source_id), []).append(str(reason))
    return {source_id: sorted(reasons) for source_id, reasons in result.items()}


def native_mesh_topology(positions_m: np.ndarray, triangles: np.ndarray) -> dict:
    """Find nonmanifold or inconsistently wound geometric edges without repair."""
    positions = np.asarray(positions_m, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Native mesh positions and triangles must be Nx3 and Mx3')
    if not np.isfinite(positions).all() or (faces.size and
            (int(faces.min()) < 0 or int(faces.max()) >= len(positions))):
        raise PrerequisiteError('Native mesh topology inputs are nonfinite or out of range')
    extent_m = float(np.ptp(positions, axis=0).max()) if len(positions) else 0.0
    position_tolerance_m = max(
        16.0 * np.finfo(np.float64).eps * max(extent_m, 1.0),
        2.0 * np.finfo(np.float32).eps * max(extent_m, np.finfo(np.float64).tiny))
    # Reserve a conservative Python-edge-table allowance before constructing it.
    _checked_bytes((len(faces) * 3, 256), np.uint8,
                   live_bytes=int(positions.nbytes + faces.nbytes),
                   label='native mesh geometric edge table')
    incidence: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[tuple[int, int]]] = {}
    reasons_by_triangle: dict[int, set[str]] = {}
    degenerate_count = 0
    for triangle_id, face in enumerate(faces):
        vertices = [tuple(int(round(float(value) / position_tolerance_m))
                          for value in positions[int(index)]) for index in face]
        for start, end in ((0, 1), (1, 2), (2, 0)):
            first, second = vertices[start], vertices[end]
            if first == second:
                reasons_by_triangle.setdefault(int(triangle_id), set()).add('degenerate_geometric_edge')
                degenerate_count += 1
                continue
            if first < second:
                edge, direction = (first, second), 1
            else:
                edge, direction = (second, first), -1
            incidence.setdefault(edge, []).append((int(triangle_id), direction))
    boundary_count = 0
    nonmanifold_edges = []
    inconsistent_edges = []
    for edge, incident in incidence.items():
        if len(incident) == 1:
            boundary_count += 1
            continue
        if len(incident) > 2:
            issue = 'nonmanifold_edge'
            nonmanifold_edges.append((edge, incident))
        elif incident[0][1] == incident[1][1]:
            issue = 'inconsistent_winding_edge'
            inconsistent_edges.append((edge, incident))
        else:
            continue
        for triangle_id, _ in incident:
            reasons_by_triangle.setdefault(triangle_id, set()).add(issue)
    def issue_record(values, issue):
        return [dict(edge_sha256=sha256_bytes(np.asarray(edge, dtype='<f8').tobytes()),
                     triangle_ids=ordered_id_record([row[0] for row in incident]),
                     reason=issue)
                for edge, incident in sorted(values, key=lambda item: item[0])]
    unsupported_ids = sorted(reasons_by_triangle)
    return dict(
        state='unsupported' if unsupported_ids else 'supported',
        vertex_count=int(len(positions)), triangle_count=int(len(faces)),
        geometric_position_tolerance_m=float(position_tolerance_m),
        geometric_position_key='round(position_m / reported tolerance); tolerance is two float32 position-encoding ulps at mesh extent; input is never modified',
        boundary_edge_count=int(boundary_count),
        nonmanifold_edge_count=int(len(nonmanifold_edges)),
        inconsistent_winding_edge_count=int(len(inconsistent_edges)),
        degenerate_geometric_edge_count=int(degenerate_count),
        nonmanifold_edges=issue_record(nonmanifold_edges, 'nonmanifold_edge'),
        inconsistent_winding_edges=issue_record(inconsistent_edges, 'inconsistent_winding_edge'),
        unsupported_triangle_ids=unsupported_ids,
        unsupported_triangle_id_order=ordered_id_record(unsupported_ids),
        unsupported_reasons_by_triangle={str(triangle_id): sorted(reasons)
                                         for triangle_id, reasons in sorted(reasons_by_triangle.items())},
        topology_rule='geometric-position adjacency; boundary edges allowed; >2 incident faces or two same-direction faces unsupported; winding and triangle IDs preserved')


def _convex_hull_area(points: Sequence[tuple[int, int]]) -> float:
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return 0.0
    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1]) -
                (first[1] - origin[1]) * (second[0] - origin[0]))
    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    return abs(sum(hull[index][0] * hull[(index + 1) % len(hull)][1] -
                   hull[index][1] * hull[(index + 1) % len(hull)][0]
                   for index in range(len(hull)))) / 2.0


def qh_verification_support_upper_bound(qh_centers: np.ndarray,
                                        observed_mask: np.ndarray) -> dict:
    """Bound verification support using only allowed centers and the observed bbox."""
    allowed = np.asarray(qh_centers, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool)
    if allowed.ndim != 2 or observed.shape != allowed.shape:
        raise ValueError('Q_H center bitmap and observed mask must be equal 2D arrays')
    bbox = _mask_bbox(observed)
    ys, xs = np.nonzero(allowed)
    if bbox is None:
        bbox_area = 0
        represented = []
        hull_area = 0.0
        bbox_record = None
    else:
        x0, y0, x1, y1 = bbox
        bbox_area = max(1, x1 - x0 + 1) * max(1, y1 - y0 + 1)
        _checked_bytes((len(xs), 96), np.uint8,
                       live_bytes=int(allowed.nbytes + observed.nbytes + xs.nbytes + ys.nbytes),
                       label='Q_H hull point workspace')
        centers_xy = np.column_stack((xs.astype(np.float64) + 0.5,
                                      ys.astype(np.float64) + 0.5))
        cells = source_grid_cells(centers_xy, observed)
        represented = sorted(set(int(value) for value in cells.tolist() if value >= 0))
        hull_area = _convex_hull_area(list(zip(xs.astype(int).tolist(), ys.astype(int).tolist())))
        bbox_record = [x0, y0, x1, y1]
    hull_fraction = float(hull_area / bbox_area) if bbox_area else 0.0
    reasons = []
    if len(represented) < MIN_VERIFY_CELLS:
        reasons.append('fewer_than_3_possible_verification_cells')
    if hull_fraction < MIN_VERIFY_HULL_FRACTION:
        reasons.append('maximum_verification_hull_below_0.12')
    return dict(state='known_impossible' if reasons else 'possible_upper_bound',
        allowed_center_count=int(len(xs)), observed_bbox=bbox_record,
        observed_bbox_area_px2=int(bbox_area), represented_cell_count=int(len(represented)),
        represented_cells=represented, maximum_hull_area_px2=float(hull_area),
        maximum_hull_fraction=float(hull_fraction), hull_fraction=float(hull_fraction),
        impossible_reasons=reasons,
        upper_bound_only=True, sufficient_support=False,
        rule='Q_H centers and observed bbox only; fewer than 3 represented cells or max hull fraction below 0.12 is impossible, passing this upper bound is not sufficient')


def select_witness_ids(source_ids: Sequence[int], points_xy: np.ndarray,
                       std_values: Sequence[float], eligible: Sequence[bool],
                       observed_mask: np.ndarray,
                       dependency_fn: Callable[[Sequence[float]], Sequence[int]],
                       bank_dependencies: Iterable[int] = (), *,
                       max_witnesses: int = MAX_WITNESSES,
                       per_cell: int = MAX_WITNESSES_PER_CELL) -> dict:
    """Freeze up to two ranked IDs per source-bbox cell before target reads."""
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    std = np.asarray(std_values, dtype=np.float64).reshape(-1)
    keep = np.asarray(eligible, dtype=bool).reshape(-1)
    if not (len(ids) == len(xy) == len(std) == len(keep)):
        raise ValueError('Witness selector input lengths differ')
    if len(ids) and (np.any(ids[:-1] >= ids[1:])):
        raise ValueError('Source IDs must be unique and increasing')
    cells = source_grid_cells(xy, observed_mask)
    bank = set(int(x) for x in bank_dependencies)
    reserved_dependencies: set[int] = set()
    selected = []
    cell_counts = {cell: 0 for cell in range(16)}
    rejected_overlap_ids = []
    candidate_count = 0
    for cell in range(16):
        rows = np.flatnonzero(keep & np.isfinite(std) & (cells == cell))
        rows = sorted(rows.tolist(), key=lambda row: (-float(std[row]), int(ids[row])))
        for row in rows:
            if cell_counts[cell] >= int(per_cell) or len(selected) >= int(max_witnesses):
                break
            candidate_count += 1
            deps = set(int(x) for x in dependency_fn(xy[row]))
            if deps & reserved_dependencies:
                rejected_overlap_ids.append(int(ids[row]))
                continue
            selected.append(dict(source_id=int(ids[row]), row=int(row), cell=int(cell),
                                 source_std=float(std[row]),
                                 raw_dependency_overlap_with_fixed_bank=bool(deps & bank),
                                 raw_dependencies=tuple(sorted(deps))))
            reserved_dependencies.update(deps)
            cell_counts[cell] += 1
    selected_ids = [item['source_id'] for item in selected]
    count = len(selected_ids)
    reason = None
    if count < MIN_WITNESSES:
        reason = 'fewer_than_8_frozen_source_witness_ids'
    unusable = sum(item['raw_dependency_overlap_with_fixed_bank'] for item in selected)
    usable = count - int(unusable)
    if usable < MIN_WITNESSES:
        reason = 'fewer_than_8_usable_frozen_source_witness_ids_after_fixed_bank_exclusion'
    return dict(state='frozen' if usable >= MIN_WITNESSES else 'unavailable', reason=reason,
                selected=selected, ordered_ids=ordered_id_record(selected_ids),
                denominator=count, usable_count=int(usable),
                unusable_fixed_bank_overlap_count=int(unusable),
                cell_counts={str(k): int(v) for k, v in cell_counts.items() if v},
                candidate_rows_considered=int(candidate_count),
                rejected_witness_overlap_ids=rejected_overlap_ids,
                dependency_union=tuple(sorted(reserved_dependencies)),
                fixed_bank_dependency_ids_count=len(bank))


def select_fit_ids(source_ids: Sequence[int], points_xy: np.ndarray,
                   eligible: Sequence[bool], observed_mask: np.ndarray,
                   dependency_fn: Callable[[Sequence[float]], Sequence[int]],
                   heldout_dependencies: Iterable[int], *,
                   max_samples: int = MAX_FIT_SAMPLES) -> dict:
    """Select source-only fit IDs round-robin by bbox cell, increasing ID."""
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    keep = np.asarray(eligible, dtype=bool).reshape(-1)
    if not (len(ids) == len(xy) == len(keep)):
        raise ValueError('Fit selector input lengths differ')
    cells = source_grid_cells(xy, observed_mask)
    forbidden = set(int(x) for x in heldout_dependencies)
    buckets = []
    for cell in range(16):
        rows = np.flatnonzero(keep & (cells == cell))
        rows = sorted(rows.tolist(), key=lambda row: int(ids[row]))
        accepted = []
        for row in rows:
            deps = set(int(x) for x in dependency_fn(xy[row]))
            if deps & forbidden:
                continue
            accepted.append(row)
        buckets.append(accepted)
    selected = []
    depth = 0
    while len(selected) < int(max_samples):
        advanced = False
        for cell_rows in buckets:
            if depth < len(cell_rows) and len(selected) < int(max_samples):
                selected.append(cell_rows[depth])
                advanced = True
        if not advanced:
            break
        depth += 1
    selected_ids = [int(ids[row]) for row in selected]
    return dict(state='frozen' if len(selected_ids) >= MIN_FIT_SAMPLES else 'unavailable',
                reason=None if len(selected_ids) >= MIN_FIT_SAMPLES else 'fewer_than_24_fit_sources',
                selected_rows=[int(x) for x in selected],
                ordered_ids=ordered_id_record(selected_ids), count=len(selected_ids),
                max_count=int(max_samples), cell_counts={str(cell): int(len(rows))
                    for cell, rows in enumerate(buckets) if rows})


def crop_to_native_pose_m(crop_pose_m: np.ndarray, crop_from_native: np.ndarray) -> np.ndarray:
    """Frozen metre→millimetre→C^-1→metre adapter, including canonical rotation."""
    pose = np.asarray(crop_pose_m, dtype=np.float64).copy()
    transform = np.asarray(crop_from_native, dtype=np.float64)
    if pose.shape != (4, 4) or transform.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(transform).all():
        raise ValueError('Pose and crop transform must be finite 4x4 matrices')
    pose[:3, 3] *= 1000.0
    native_mm = np.linalg.inv(transform) @ pose
    u, _, vt = np.linalg.svd(native_mm[:3, :3])
    native_mm[:3, :3] = u @ vt
    if np.linalg.det(native_mm[:3, :3]) < 0:
        u[:, -1] *= -1
        native_mm[:3, :3] = u @ vt
    native_mm[:3, 3] *= .001
    return native_mm


def native_to_crop_pose_mm(native_pose_m: np.ndarray, crop_from_native: np.ndarray) -> np.ndarray:
    pose = np.asarray(native_pose_m, dtype=np.float64).copy()
    transform = np.asarray(crop_from_native, dtype=np.float64)
    if pose.shape != (4, 4) or transform.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(transform).all():
        raise ValueError('Pose and crop transform must be finite 4x4 matrices')
    pose[:3, 3] *= 1000.0
    return transform @ pose


def project_crop_points(points_object_m: np.ndarray, pose_crop_m: np.ndarray,
                        intrinsics: np.ndarray, center_object_m: Sequence[float] = (0., 0., 0.)) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_object_m, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_crop_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    center = np.asarray(center_object_m, dtype=np.float64).reshape(3)
    if pose.shape != (4, 4) or k.shape != (3, 3):
        raise ValueError('Projection requires a 4x4 pose and 3x3 intrinsics')
    p = pose[:3, :3] @ center + pose[:3, 3]
    camera = (points - center) @ pose[:3, :3].T + p
    homogeneous = camera @ k.T
    xy = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.) & np.isfinite(homogeneous).all(axis=1)
    xy[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    valid &= np.isfinite(xy).all(axis=1)
    return xy, valid


def crop_ray_to_native_xy(crop_xy: Sequence[float], crop_k: np.ndarray,
                          crop_from_native: np.ndarray,
                          native_k: np.ndarray) -> np.ndarray:
    """Convert the original measured crop ray, never a candidate projection."""
    q = np.asarray(crop_xy, dtype=np.float64).reshape(2)
    k_crop = np.asarray(crop_k, dtype=np.float64)
    c = np.asarray(crop_from_native, dtype=np.float64)
    k_native = np.asarray(native_k, dtype=np.float64)
    if not np.isfinite(q).all() or k_crop.shape != (3, 3) or c.shape != (4, 4) or k_native.shape != (3, 3):
        return np.full(2, np.nan, dtype=np.float64)
    crop_ray = np.asarray((q[0], q[1], 1.), dtype=np.float64) @ np.linalg.inv(k_crop).T
    native_ray = crop_ray @ c[:3, :3]
    projected = native_ray @ k_native.T
    if not np.isfinite(projected).all() or projected[2] == 0.:
        return np.full(2, np.nan, dtype=np.float64)
    return projected[:2] / projected[2]


def skew(value: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64).reshape(3)
    return np.asarray(((0., -z, y), (z, 0., -x), (-y, x, 0.)), dtype=np.float64)


def so3_exp(omega: Sequence[float]) -> np.ndarray:
    vector = np.asarray(omega, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError('Rotation vector must be finite')
    theta = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if theta < 1e-8:
        a = 1. - theta * theta / 6. + theta**4 / 120.
        b = .5 - theta * theta / 24. + theta**4 / 720.
    else:
        a = math.sin(theta) / theta
        b = (1. - math.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * matrix + b * (matrix @ matrix)


def local_pose_from_normalized(seed_pose_m: np.ndarray, center_object_m: Sequence[float],
                               diagonal_m: float, xi: Sequence[float]) -> np.ndarray:
    seed = np.asarray(seed_pose_m, dtype=np.float64)
    center = np.asarray(center_object_m, dtype=np.float64).reshape(3)
    value = np.asarray(xi, dtype=np.float64).reshape(6)
    if seed.shape != (4, 4) or not np.isfinite(seed).all() or not np.isfinite(center).all():
        raise ValueError('Seed pose and center must be finite')
    if not math.isfinite(float(diagonal_m)) or float(diagonal_m) <= 0.:
        raise ValueError('Object diagonal must be positive and finite')
    rotation = seed[:3, :3] @ so3_exp(value[:3])
    p0 = seed[:3, :3] @ center + seed[:3, 3]
    p = p0 + float(diagonal_m) * value[3:]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = p - rotation @ center
    return result


def yaw_challenge_pose(seed_pose_m: np.ndarray, center_object_m: Sequence[float],
                       axis_object: Sequence[float], yaw_degrees: float) -> np.ndarray:
    seed = np.asarray(seed_pose_m, dtype=np.float64)
    center = np.asarray(center_object_m, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_object, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if seed.shape != (4, 4) or not np.isfinite(seed).all() or norm == 0.:
        raise ValueError('Invalid challenge seed or axis')
    angle = math.radians(float(yaw_degrees))
    delta = so3_exp(axis / norm * angle)
    rotation = seed[:3, :3] @ delta
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = seed[:3, 3] + seed[:3, :3] @ center - rotation @ center
    return result


def canonical_yaw_degrees(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    return -180.0 if result == 180.0 else result


def declared_challenge_poses(seed_poses: Sequence[np.ndarray], center_object_m: Sequence[float],
                             axis_object: Sequence[float],
                             offsets_degrees: Sequence[float] = CHALLENGE_YAW_DEG,
                             cap: int = 12) -> list[dict]:
    rows = []
    seen = set()
    for seed_index, seed in enumerate(seed_poses):
        candidates = [(None, np.asarray(seed, dtype=np.float64))]
        canonical_offsets = []
        for theta in offsets_degrees:
            canonical = canonical_yaw_degrees(theta)
            if canonical not in canonical_offsets:
                canonical_offsets.append(canonical)
        for theta in canonical_offsets:
            candidates.append((theta, yaw_challenge_pose(seed, center_object_m, axis_object, theta)))
        for theta, pose in candidates:
            key = np.ascontiguousarray(pose, dtype='<f8').tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(seed_index=int(seed_index), yaw_offset_deg=theta,
                             pose_crop_m=pose.tolist(), pose_sha256=sha256_bytes(key)))
            if len(rows) >= int(cap):
                return rows
    return rows


def central_difference(function: Callable[[np.ndarray], np.ndarray], value: Sequence[float],
                       step: float = 1e-6) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(point).all() or not math.isfinite(step) or step <= 0.:
        raise ValueError('Finite differences require a finite point and positive step')
    base = np.asarray(function(point), dtype=np.float64)
    jac = np.empty((base.size, point.size), dtype=np.float64)
    for column in range(point.size):
        delta = np.zeros_like(point)
        delta[column] = step
        plus = np.asarray(function(point + delta), dtype=np.float64)
        minus = np.asarray(function(point - delta), dtype=np.float64)
        if plus.shape != base.shape or minus.shape != base.shape:
            raise ValueError('Finite-difference outputs changed shape')
        jac[:, column] = ((plus - minus) / (2. * step)).reshape(-1)
    return jac


def project_nuisance_jacobian(j_pose: np.ndarray, j_ab: np.ndarray) -> np.ndarray:
    """Project gain/bias columns out without allocating an N×N projector."""
    pose = np.asarray(j_pose, dtype=np.float64)
    nuisance = np.asarray(j_ab, dtype=np.float64)
    if pose.ndim != 2 or nuisance.ndim != 2 or pose.shape[0] != nuisance.shape[0]:
        raise ValueError('Pose and nuisance Jacobians must have the same row count')
    if not np.isfinite(pose).all() or not np.isfinite(nuisance).all():
        raise ValueError('Jacobian inputs must be finite')
    if nuisance.shape[1] == 0:
        return pose.copy()
    return pose - nuisance @ (np.linalg.pinv(nuisance) @ pose)


def fixture_cylinder_mesh() -> dict:
    """Exact 64-segment cylinder topology and UV seam from the frozen design."""
    # Reserve the NumPy mesh arrays and their one-time normal/index temporaries
    # before constructing them. Python tuple-list construction is bounded at
    # 132 vertices / 256 faces and remains below 256 KiB.
    _checked_bytes((132 * 3 + 256 * 3 + 132 * 2 + 256 * 3, ), np.float64,
                   label='fixture mesh arrays and normal scratch')
    vertices = []
    uv = []
    for j in range(65):
        angle = 2.0 * math.pi * j / 64.0
        x, z = RADIUS_M * math.sin(angle), RADIUS_M * math.cos(angle)
        vertices.extend(((x, -HALF_HEIGHT_M, z), (x, HALF_HEIGHT_M, z)))
        uv.extend(((j / 64.0, 0.0), (j / 64.0, 1.0)))
    lower_center = len(vertices)
    vertices.extend(((0., -HALF_HEIGHT_M, 0.), (0., HALF_HEIGHT_M, 0.)))
    uv.extend(((.5, .5), (.5, .5)))
    faces = []
    for j in range(64):
        b0, t0 = 2 * j, 2 * j + 1
        b1, t1 = 2 * (j + 1), 2 * (j + 1) + 1
        faces.extend(((b0, b1, t0), (b1, t1, t0)))
    for j in range(64):
        b0, b1 = 2 * j, 2 * ((j + 1) % 64)
        faces.append((lower_center, b1, b0))
    for j in range(64):
        b0, b1 = 2 * j, 2 * ((j + 1) % 64)
        t0, t1 = b0 + 1, b1 + 1
        faces.append((lower_center + 1, t0, t1))
    positions = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    uv_array = np.asarray(uv, dtype=np.float64)
    if (triangles.shape != (256, 3) or len(positions) != 132 or lower_center != 130 or
            not np.array_equal(triangles[[128, 191, 192, 255]],
                               np.asarray(((130, 2, 0), (130, 0, 126),
                                           (131, 1, 3), (131, 127, 1)), dtype=np.int64))):
        raise AssertionError('Fixture cylinder topology drifted from its frozen 256 IDs')
    normals = np.cross(positions[triangles[:, 1]] - positions[triangles[:, 0]],
                       positions[triangles[:, 2]] - positions[triangles[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.):
        raise AssertionError('Fixture contains a degenerate triangle')
    normals /= lengths[:, None]
    return dict(positions=positions, triangles=triangles, uv=uv_array,
                triangle_normals=normals,
                triangle_ids=np.arange(256, dtype=np.int64),
                radius_m=RADIUS_M, height_m=2. * HALF_HEIGHT_M,
                center_m=np.zeros(3, dtype=np.float64),
                diagonal_m=float(np.linalg.norm(np.ptp(positions, axis=0))))


def fixture_texture(kind: str = 'markers') -> np.ndarray:
    """Return the exact frozen 256×256 float32 RGB texture."""
    # Integer grids, marker arithmetic, float64 channels and masks coexist for
    # a moment; reserve a conservative 18 MiB before any fixture allocation.
    _checked_bytes((18 * 1024**2,), np.uint8, label='fixture texture construction peak')
    tx, ty = np.meshgrid(np.arange(256, dtype=np.int64), np.arange(256, dtype=np.int64))
    if kind == 'constant':
        value = np.full((256, 256, 3), .6, dtype=np.float64)
        return value.astype(np.float32)
    if kind == 'periodic_2d':
        u = tx % 128
        v = ty
        p_int = (37 * u + 17 * v + 13 * u * v + 11) % 251
        p = p_int.astype(np.float64) / 250.0
        gray = .15 + .70 * p
        return np.stack((gray, gray, gray), axis=2).astype(np.float32)
    if kind not in ('markers', 'base'):
        raise ValueError(f'Unknown frozen fixture texture {kind!r}')
    g = .25 + .5 * ((tx // 8) % 2)
    texture = np.stack((g, g, g), axis=2).astype(np.float64)
    if kind == 'base':
        return texture.astype(np.float32)
    rectangles = ((92, 164, 12, 72, 11), (92, 164, 88, 168, 53),
                  (92, 164, 184, 244, 97))
    colors = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for (x0, x1, y0, y1, seed), order in zip(rectangles, colors):
        inside = (tx >= x0) & (tx < x1) & (ty >= y0) & (ty < y1)
        local_x, local_y = tx - x0, ty - y0
        p_int = (37 * local_x + 17 * local_y + 13 * local_x * local_y + seed) % 251
        p = p_int.astype(np.float64) / 250.0
        c0, c1, c2 = ((.15 + .75 * p, .15 + .30 * p, .20 + .15 * p)
                      if order == (0, 1, 2) else
                      (.20 + .15 * p, .15 + .75 * p, .15 + .30 * p)
                      if order == (1, 2, 0) else
                      (.15 + .30 * p, .20 + .15 * p, .15 + .75 * p))
        texture[inside, 0] = c0[inside]
        texture[inside, 1] = c1[inside]
        texture[inside, 2] = c2[inside]
    return texture.astype(np.float32)


def _pose_y(rotation_y_degrees: float, translation=(0., 0., 1.)) -> np.ndarray:
    theta = math.radians(float(rotation_y_degrees))
    c, s = math.cos(theta), math.sin(theta)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ((c, 0., s), (0., 1., 0.), (-s, 0., c))
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def raycast_first_hits(origins_m: np.ndarray, directions: np.ndarray,
                       vertices_m: np.ndarray, triangles: np.ndarray, *,
                       chunk_size: int | None = None,
                       resident_bytes: int = 0) -> dict:
    """Two-sided Möller–Trumbore, streamed by ray chunks, with frozen tie rule."""
    origins = np.asarray(origins_m, dtype=np.float64).reshape(-1, 3)
    rays = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(vertices_m, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(origins) != len(rays) or not len(faces) or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError('Invalid ray or triangle arrays')
    if not np.isfinite(origins).all() or not np.isfinite(rays).all() or not np.isfinite(vertices).all():
        raise ValueError('Ray and mesh inputs must be finite')
    tri_bytes = int(vertices.nbytes + faces.nbytes + len(faces) * 3 * 8 * 3)
    input_bytes = int(origins.nbytes + rays.nbytes + vertices.nbytes + faces.nbytes)
    base_live = int(resident_bytes) + tri_bytes + input_bytes
    _checked_bytes((len(rays),), np.float64, live_bytes=base_live,
                   label='ray normalization workspace')
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 0.):
        raise ValueError('Ray directions must be nonzero')
    output_bytes = int(len(rays) * (3 * 8 + 8 + 8 + 3 * 8 + 1))
    _checked_bytes((output_bytes,), np.uint8, live_bytes=base_live + norms.nbytes,
                   label='ray-hit output')
    available = MAX_WORKING_BYTES - base_live - norms.nbytes - output_bytes
    allowed_chunk = min(MAX_RAY_CHUNK, max(1, available // max(1, len(faces) * RAY_PAIR_SCRATCH_BYTES)))
    chunk = allowed_chunk if chunk_size is None else min(int(chunk_size), allowed_chunk)
    if chunk < 1:
        raise ResourceLimitError('No ray chunk fits the 128 MiB working-buffer limit')
    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0
    distance = guarded_empty((len(rays),), np.float64,
                             live_bytes=resident_bytes + tri_bytes + output_bytes - len(rays) * 8,
                             label='ray distance output')
    triangle_id = guarded_empty((len(rays),), np.int64,
                                live_bytes=resident_bytes + tri_bytes + output_bytes - len(rays) * 8,
                                label='ray triangle output')
    barycentric = guarded_empty((len(rays), 3), np.float64,
                                live_bytes=resident_bytes + tri_bytes + output_bytes - len(rays) * 24,
                                label='ray barycentric output')
    hit_point = guarded_empty((len(rays), 3), np.float64,
                              live_bytes=resident_bytes + tri_bytes + output_bytes - len(rays) * 24,
                              label='ray point output')
    hit = guarded_zeros((len(rays),), np.bool_,
                        live_bytes=resident_bytes + tri_bytes + output_bytes - len(rays),
                        label='ray validity output')
    for start in range(0, len(rays), chunk):
        stop = min(len(rays), start + chunk)
        ray_block = rays[start:stop] / norms[start:stop, None]
        origin_block = origins[start:stop]
        scratch_bytes = int(len(ray_block) * len(faces) * RAY_PAIR_SCRATCH_BYTES)
        _checked_bytes((scratch_bytes,), np.uint8,
                       live_bytes=base_live + norms.nbytes + output_bytes +
                       len(ray_block) * 3 * 8,
                       limit_bytes=MAX_WORKING_BYTES,
                       label='streamed triangle intersection scratch')
        pvec = np.cross(ray_block[:, None, :], e2[None, :, :])
        det = np.einsum('td,rtd->rt', e1, pvec, optimize=False)
        det_ok = np.abs(det) > 1e-12
        safe_det = np.where(det_ok, det, 1.0)
        tvec = origin_block[:, None, :] - v0[None, :, :]
        u = np.einsum('rtd,rtd->rt', tvec, pvec, optimize=False) / safe_det
        qvec = np.cross(tvec, e1[None, :, :])
        v = np.einsum('rd,rtd->rt', ray_block, qvec, optimize=False) / safe_det
        t = np.einsum('td,rtd->rt', e2, qvec, optimize=False) / safe_det
        valid = (det_ok & (u >= -1e-12) & (v >= -1e-12) &
                 (u + v <= 1.0 + 1e-12) & (t > 0.0) & np.isfinite(t))
        candidate_t = np.where(valid, t, np.inf)
        minimum = np.min(candidate_t, axis=1)
        tied = valid & (candidate_t <= minimum[:, None] + 1e-12)
        chosen = np.argmax(tied, axis=1)  # triangle IDs are ascending: first wins ties.
        has_hit = np.isfinite(minimum)
        rows = np.arange(stop - start)
        chosen_u = u[rows, chosen]
        chosen_v = v[rows, chosen]
        distance[start:stop] = np.where(has_hit, minimum, np.nan)
        triangle_id[start:stop] = np.where(has_hit, chosen, -1)
        barycentric[start:stop] = np.stack((1. - chosen_u - chosen_v, chosen_u, chosen_v), axis=1)
        hit_point[start:stop] = origin_block + ray_block * np.where(has_hit, minimum, 0.)[:, None]
        hit[start:stop] = has_hit
        del pvec, det, det_ok, safe_det, tvec, u, qvec, v, t, valid, candidate_t, tied
    return dict(hit=hit, distance_m=distance, triangle_id=triangle_id,
                barycentric=barycentric, point_m=hit_point,
                chunk_size=int(chunk), estimated_scratch_bytes=int(chunk * len(faces) * RAY_PAIR_SCRATCH_BYTES),
                ray_count=int(len(rays)), triangle_count=int(len(faces)))


def _pixel_rays(width: int, height: int, intrinsics: np.ndarray,
                start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(intrinsics, dtype=np.float64)
    inv = np.linalg.inv(k)
    flat = np.arange(start, stop, dtype=np.int64)
    x = flat % width
    y = flat // width
    centers = np.column_stack((x.astype(np.float64) + .5,
                               y.astype(np.float64) + .5,
                               np.ones(len(flat), dtype=np.float64)))
    directions = centers @ inv.T
    origins = np.zeros_like(directions)
    return origins, directions


def _render_fixture_view(mesh: Mapping[str, np.ndarray], texture: np.ndarray,
                         pose_camera_from_object_m: np.ndarray,
                         intrinsics: np.ndarray = K_FIXTURE,
                         width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT,
                         resident_bytes: int = 0) -> dict:
    vertices = np.asarray(mesh['positions'], dtype=np.float64)
    triangles = np.asarray(mesh['triangles'], dtype=np.int64)
    pose = np.asarray(pose_camera_from_object_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    tex = np.asarray(texture, dtype=np.float32)
    pixels = width * height
    pair_bytes = pixels * (3 * 4 + 1 + 8 + 1)
    ray_bytes = int(min(MAX_RAY_CHUNK, width * height) * len(triangles) * RAY_PAIR_SCRATCH_BYTES)
    geometry_bytes = int(vertices.nbytes + triangles.nbytes + mesh['uv'].nbytes + tex.nbytes)
    # Includes full-frame hit/depth/barycentric/camera/object buffers, hit
    # shading temporaries and retained caller arrays during the second view.
    frame_working = pixels * 160
    _checked_bytes((1,), np.uint8,
                   live_bytes=int(resident_bytes + frame_working + ray_bytes + geometry_bytes),
                   label='fixture render working set')
    rgb = np.full((height, width, 3), np.float32(.5), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    depth = np.zeros((height, width), dtype=np.float64)
    flat_hit = np.zeros(width * height, dtype=bool)
    flat_distance = np.full(width * height, np.nan, dtype=np.float64)
    flat_tri = np.full(width * height, -1, dtype=np.int64)
    flat_bary = np.zeros((width * height, 3), dtype=np.float64)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    transformed = vertices @ rotation.T + translation
    for start in range(0, width * height, MAX_RAY_CHUNK):
        stop = min(width * height, start + MAX_RAY_CHUNK)
        origins, rays = _pixel_rays(width, height, k, start, stop)
        result = raycast_first_hits(origins, rays, transformed, triangles,
                                    chunk_size=MAX_RAY_CHUNK,
                                    resident_bytes=resident_bytes + geometry_bytes + frame_working)
        flat_hit[start:stop] = result['hit']
        flat_distance[start:stop] = result['distance_m']
        flat_tri[start:stop] = result['triangle_id']
        flat_bary[start:stop] = result['barycentric']
    hit_ids = np.flatnonzero(flat_hit)
    if len(hit_ids):
        tri_ids = flat_tri[hit_ids]
        bary = flat_bary[hit_ids]
        uv_vertices = np.asarray(mesh['uv'], dtype=np.float64)[triangles[tri_ids]]
        uv = np.einsum('ni,nij->nj', bary, uv_vertices, optimize=False)
        side = tri_ids < 128
        colors = np.full((len(hit_ids), 3), .6, dtype=np.float64)
        if np.any(side):
            tex_x = np.floor(256. * uv[side, 0]).astype(np.int64) % 256
            tex_y = np.clip(np.floor(256. * uv[side, 1]).astype(np.int64), 0, 255)
            colors[side] = tex[tex_y, tex_x].astype(np.float64)
        rgb.reshape(-1, 3)[hit_ids] = colors.astype(np.float32)
        mask.reshape(-1)[hit_ids] = True
        starts = hit_ids
        y = starts // width
        x = starts % width
        q = np.column_stack((x.astype(np.float64) + .5, y.astype(np.float64) + .5,
                             np.ones(len(starts), dtype=np.float64)))
        ray = q @ np.linalg.inv(k).T
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        depth.reshape(-1)[hit_ids] = ray[:, 2] * flat_distance[hit_ids] * 1000.
    camera_points = np.zeros((width * height, 3), dtype=np.float64)
    for start in range(0, width * height, MAX_RAY_CHUNK):
        stop = min(width * height, start + MAX_RAY_CHUNK)
        if not np.any(flat_hit[start:stop]):
            continue
        origins, rays = _pixel_rays(width, height, k, start, stop)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        block_ids = np.flatnonzero(flat_hit[start:stop]) + start
        camera_points[block_ids] = origins[flat_hit[start:stop]] + rays[flat_hit[start:stop]] * flat_distance[block_ids, None]
    object_points = (camera_points - translation) @ rotation
    return dict(rgb=rgb, observed_mask=mask, depth_mm=depth,
                hit_triangle_ids=flat_tri.reshape(height, width),
                hit_barycentric=flat_bary.reshape(height, width, 3),
                source_points_object_m=object_points.reshape(height, width, 3),
                camera_from_object_m=pose.copy(), intrinsics=k.copy(),
                raster_center_convention='projected q is edge-coordinate; array index=q-0.5 once',
                ray_chunk_size=MAX_RAY_CHUNK)


def _source_bank_ids(sample_ids: np.ndarray, eligible: np.ndarray,
                     width: int, height: int, maximum: int = 64) -> list[int]:
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    keep = np.asarray(eligible, dtype=bool).reshape(-1)
    if len(ids) != len(keep):
        raise ValueError('Bank selector eligibility length mismatch')
    id_to_row = {int(value): row for row, value in enumerate(ids.tolist())}
    xs = np.clip(np.rint((np.arange(16) + .5) * width / 16.).astype(int),
                 PATCH_RADIUS, max(PATCH_RADIUS, width - PATCH_RADIUS - 1))
    ys = np.clip(np.rint((np.arange(16) + .5) * height / 16.).astype(int),
                 PATCH_RADIUS, max(PATCH_RADIUS, height - PATCH_RADIUS - 1))
    candidates = []
    for y in ys:
        for x in xs:
            flat_id = int(y * width + x)
            row = id_to_row.get(flat_id)
            if row is not None and keep[row]:
                candidates.append(flat_id)
    if len(candidates) > maximum:
        selected = np.linspace(0, len(candidates) - 1, maximum, dtype=int)
        candidates = [candidates[int(index)] for index in selected]
    return candidates


def _gray_rgb_cv2(rgb: np.ndarray, cv2_module) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Expected H×W×3 RGB')
    return cv2_module.cvtColor(value, cv2_module.COLOR_RGB2GRAY).astype(np.float32)


def _opencv_gaussian_path_audit(cv2_module) -> dict:
    """Prove the legacy sigma-2 path has the frozen 17×17 support before use."""
    kernel = np.asarray(cv2_module.getGaussianKernel(17, 2.0, cv2_module.CV_32F), dtype=np.float32).reshape(-1)
    if kernel.shape != (17,) or not np.isfinite(kernel).all() or np.any(kernel <= 0.):
        return dict(state='failed', reason='invalid_explicit_17_tap_kernel')
    impulse = np.zeros((65, 65), dtype=np.float32)
    impulse[32, 32] = 1.
    legacy = cv2_module.GaussianBlur(impulse, (0, 0), 2.0)
    explicit = cv2_module.GaussianBlur(impulse, (17, 17), 2.0,
                                       borderType=cv2_module.BORDER_REFLECT_101)
    active = np.flatnonzero(np.abs(legacy[32]) > 0.)
    support_width = int(active[-1] - active[0] + 1) if len(active) else 0
    max_error = float(np.max(np.abs(legacy.astype(np.float64) - explicit.astype(np.float64))))
    exact_support = bool(support_width == 17 and np.array_equal(legacy, explicit))
    return dict(state='passed' if exact_support else 'failed',
                reason=None if exact_support else 'legacy_sigma2_path_not_bitwise_17x17_reflect101',
                legacy_support_width=support_width,
                legacy_nonzero_coefficients=legacy[32, 32 - 8:32 + 9].tolist(),
                explicit_kernel_f32=kernel.tolist(),
                explicit_nonzero_coefficients=int(np.count_nonzero(kernel)),
                explicit_response_sha256=array_record(explicit)['sha256'],
                legacy_response_sha256=array_record(legacy)['sha256'],
                max_abs_response_error=max_error,
                border='BORDER_REFLECT_101', bitwise_parity=bool(np.array_equal(legacy, explicit)))


def _highpass_statistics(rgb: np.ndarray, mask: np.ndarray, centers_xy: np.ndarray,
                         cv2_module, audit_module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the frozen R5 source eligibility and mask-footprint rules."""
    gray = audit_module._gray_image(np.asarray(rgb, dtype=np.float32))
    high = audit_module._highpass(gray, 2.)
    binary = np.asarray(mask, dtype=bool)
    eroded = audit_module._eroded_mask(binary, EROSION_PIXELS)
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    std = np.full(len(centers), np.nan, dtype=np.float64)
    eligible = np.zeros(len(centers), dtype=bool)
    for row, q in enumerate(centers):
        patch, _ = audit_module._bilinear_patch(high, q, eroded)
        if patch is None:
            continue
        descriptor, current_std = audit_module._normalize_patch(patch)
        if descriptor is None:
            std[row] = current_std
            continue
        std[row] = current_std
        eligible[row] = True
    return eligible, std, eroded


def _compress_dependencies(dependencies: Sequence[int]) -> str:
    values = np.asarray(dependencies, dtype='<u4').reshape(-1)
    return base64.b64encode(values.tobytes(order='C')).decode('ascii')


def _serializable_witness_bank(bank: Mapping[str, object], source_image_id: str,
                               source_image_sha256: str) -> dict:
    return dict(state=bank['state'], reason=bank['reason'], denominator=bank['denominator'],
                usable_count=bank['usable_count'], source_image_id=source_image_id,
                source_rgb_sha256=source_image_sha256,
                ordered_ids=bank['ordered_ids'], cell_counts=bank['cell_counts'],
                candidate_rows_considered=bank['candidate_rows_considered'],
                rejected_witness_overlap_ids=bank['rejected_witness_overlap_ids'],
                unusable_fixed_bank_overlap_count=bank['unusable_fixed_bank_overlap_count'],
                witnesses=[dict(source_id=item['source_id'], row=item['row'], cell=item['cell'],
                                source_std=item['source_std'],
                                raw_dependency_overlap_with_fixed_bank=item['raw_dependency_overlap_with_fixed_bank'],
                                raw_dependency_count=len(item['raw_dependencies']),
                                raw_dependency_ids_u32_le_base64=_compress_dependencies(item['raw_dependencies']),
                                raw_dependency_sha256=sha256_bytes(
                                    np.asarray(item['raw_dependencies'], dtype='<u4').tobytes()))
                           for item in bank['selected']],
                dependency_union=dict(image_qualified=True, source_image_id=source_image_id,
                    source_rgb_sha256=source_image_sha256,
                    count=len(bank['dependency_union']),
                    sha256=sha256_bytes(np.asarray(bank['dependency_union'], dtype='<u4').tobytes()),
                    ids_u32_le_base64=_compress_dependencies(bank['dependency_union'])))


def association_filtered_source_support(
        witness_ids: Sequence[int], witness_usable_before_ids: Sequence[int],
        fit_selected_ids: Sequence[int], fit_usable_before_ids: Sequence[int],
        association_supported_by_id: Mapping[int, bool]) -> dict:
    """Apply frozen geometry-support penalties without replacing selected IDs.

    Witness IDs remain in their original reserved denominator. Unsupported
    witness and fit IDs are accounted as unusable; no selector is rerun and no
    replacement ID is selected.
    """
    def _ordered_ids(values, label):
        result = [int(value) for value in values]
        if len(result) != len(set(result)):
            raise PrerequisiteError(f'Duplicate source ID in {label}')
        return result

    witnesses = _ordered_ids(witness_ids, 'witness selection')
    witness_before = set(_ordered_ids(witness_usable_before_ids, 'pre-association witnesses'))
    fits = _ordered_ids(fit_selected_ids, 'fit selection')
    fit_before = set(_ordered_ids(fit_usable_before_ids, 'pre-association fits'))
    if not witness_before.issubset(witnesses) or not fit_before.issubset(fits):
        raise PrerequisiteError('Pre-association support IDs must be selected IDs')
    supported = {int(source_id): bool(value)
                 for source_id, value in association_supported_by_id.items()}

    def _filtered(selected, eligible_before):
        usable = [source_id for source_id in selected
                  if source_id in eligible_before and supported.get(source_id, False)]
        unsupported = [source_id for source_id in selected
                       if source_id in eligible_before and not supported.get(source_id, False)]
        return usable, unsupported

    witness_usable, witness_unsupported = _filtered(witnesses, witness_before)
    fit_usable, fit_unsupported = _filtered(fits, fit_before)
    return dict(
        witness_selected_ids=witnesses,
        witness_reserved_denominator=len(witnesses),
        witness_usable_ids=witness_usable,
        witness_usable_count=len(witness_usable),
        witness_usable_count_before_association=len(witness_before),
        witness_unsupported_selected_ids=witness_unsupported,
        witness_state='frozen' if len(witness_usable) >= MIN_WITNESSES else 'unavailable',
        fit_selected_ids=fits,
        fit_usable_ids=fit_usable,
        fit_usable_count=len(fit_usable),
        fit_usable_count_before_association=len(fit_before),
        fit_unsupported_selected_ids=fit_unsupported,
        fit_state='frozen' if len(fit_usable) >= MIN_FIT_SAMPLES else 'unavailable')


def source_row_prerequisite_state(*, witness_state: str, fit_state: str,
                                  fixed_bank_supported: bool,
                                  qh_upper_bound_state: str,
                                  unsupported_association_count: int = 0) -> dict:
    """Evaluate frozen source/query prerequisites; associations are per-ID penalties."""
    reasons = []
    if witness_state != 'frozen':
        reasons.append('source_witness_support_unavailable')
    if fit_state != 'frozen':
        reasons.append('source_fit_support_unavailable')
    if not bool(fixed_bank_supported):
        reasons.append('fixed_bank_source_appearance_unsupported')
    if qh_upper_bound_state == 'known_impossible':
        reasons.append('verification_support_upper_bound_known_impossible')
    elif qh_upper_bound_state != 'possible_upper_bound':
        reasons.append('verification_support_upper_bound_unavailable')
    return dict(state='prepared' if not reasons else 'terminal_unavailable',
                unavailable_reasons=reasons,
                unsupported_association_count=int(unsupported_association_count),
                association_count_is_row_gate=False)


def _account_terminal_rows(planned_ids: Sequence[str], outcomes: Mapping[str, Mapping[str, object]],
                           default_reason: str) -> list[dict]:
    rows = []
    for condition_id in planned_ids:
        result = outcomes.get(condition_id)
        if result is None:
            rows.append(dict(condition_id=condition_id, state='terminal_unavailable',
                             reason=default_reason))
        else:
            row = dict(result)
            row['condition_id'] = condition_id
            if row.get('state') not in ('prepared', 'terminal_unavailable'):
                row['state'] = 'terminal_unavailable'
            rows.append(row)
    return rows


def _actual_source_manifest(r5, r6, audit, bundle, plans, selections, banks,
                            cv2_module, mesh) -> tuple[dict[str, dict], dict[str, dict]]:
    capture = bundle['capture']
    center = np.asarray(capture['object_bbox_center_m'], dtype=np.float64)
    axis = np.asarray(capture['object_axis_longest_svd'], dtype=np.float64)
    diagonal = float(capture['object_bbox_diagonal_m'])
    if (center.shape != (3,) or axis.shape != (3,) or not np.isfinite(center).all() or
            not np.isfinite(axis).all() or abs(float(np.linalg.norm(axis)) - 1.) > 1e-9 or
            not math.isfinite(diagonal) or diagonal <= 0.):
        raise PrerequisiteError('Frozen R1 center, signed axis, or diagonal is invalid')
    view_records: dict[str, dict] = {}
    row_records: dict[str, dict] = {}
    template_plans = {}
    for plan in plans:
        template_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        template_plans.setdefault(template_id, plan)
    mesh_topology = native_mesh_topology(mesh.positions, mesh.triangles)
    mesh_unsupported_ids = set(mesh_topology['unsupported_triangle_ids'])
    for template_id, plan in template_plans.items():
        entry, source = r5._condition_source_only_inputs(bundle, plan)
        rgb = np.asarray(source['template_rgb'], dtype=np.float32)
        observed = np.asarray(source['template_mask'], dtype=bool) & np.asarray(source['observed_crop_mask'], dtype=bool)
        ids = np.asarray(source['source_indices'], dtype=np.int64).reshape(-1)
        xy = np.asarray(source['source_pixels_xy'], dtype=np.float64).reshape(-1, 2)
        points = np.asarray(source['source_points_object_m'], dtype=np.float64).reshape(-1, 3)
        eligible_r5 = np.asarray(selections[template_id]['eligible'], dtype=bool).reshape(-1)
        if not (len(ids) == len(xy) == len(points) == len(eligible_r5)):
            raise PrerequisiteError(f'R5 source selector shape mismatch in {template_id}')
        source_live_bytes = sum(int(value.nbytes) for value in source.values()
                                if isinstance(value, np.ndarray))
        appearance_record, appearance_supported = _source_appearance_support_for_ids(
            ids, xy, observed, np.asarray(source['template_depth_mm']),
            resident_bytes=source_live_bytes)
        appearance_reasons = _appearance_reason_map(appearance_record)
        eligible_local, std, eroded = _highpass_statistics(rgb, observed, xy, cv2_module, audit)
        eligible = eligible_r5 & eligible_local
        if not np.array_equal(np.flatnonzero(eligible), np.flatnonzero(eligible_r5)):
            raise PrerequisiteError(f'R7 recomputed source eligibility differs from frozen R5 in {template_id}')
        anchors = banks[template_id]['anchors']
        bank_dep_by_id = {}
        for anchor in anchors:
            q = anchor.get('source_xy_crop')
            deps = highpass_patch_raw_dependencies(q, rgb.shape[1], rgb.shape[0])
            bank_dep_by_id[int(anchor['source_index'])] = deps
        opposite_id = r5._source_template_refs(plan['frame_id'], 180 - int(plan['template_offset_deg']))
        source_bank_dependencies = set(dep for dependencies in bank_dep_by_id.values()
                                       for dep in dependencies)
        opposite_bank_dependencies = set()
        opposite_bank_dep_by_id = {}
        for anchor in banks[opposite_id]['anchors']:
            dependencies = highpass_patch_raw_dependencies(
                anchor.get('source_xy_crop'), rgb.shape[1], rgb.shape[0])
            opposite_bank_dep_by_id[int(anchor['source_index'])] = dependencies
            opposite_bank_dependencies.update(dependencies)
        source_image_sha = selections[template_id]['source_array_hashes']['template_rgb']['sha256']
        opposite_image_sha = selections[opposite_id]['source_array_hashes']['template_rgb']['sha256']
        index_by_id = {int(value): row for row, value in enumerate(ids.tolist())}
        witnesses = select_witness_ids(
            ids, xy, std, eligible, observed,
            lambda q: highpass_patch_raw_dependencies(q, rgb.shape[1], rgb.shape[0]),
            source_bank_dependencies)
        fits = select_fit_ids(
            ids, xy, eligible, observed,
            lambda q: highpass_patch_raw_dependencies(q, rgb.shape[1], rgb.shape[0]),
            witnesses['dependency_union'])
        witness_selected_rows = np.asarray([item['row'] for item in witnesses['selected']], dtype=np.int64)
        witness_supported = (appearance_supported[witness_selected_rows]
                             if len(witness_selected_rows) else np.zeros(0, dtype=bool))
        witness_appearance_unsupported_ids = [
            int(item['source_id']) for item, supported in zip(witnesses['selected'], witness_supported)
            if not supported]
        witness_appearance_usable = sum(
            bool(supported) and not bool(item['raw_dependency_overlap_with_fixed_bank'])
            for item, supported in zip(witnesses['selected'], witness_supported))
        witness_usable_before_association_ids = [
            int(item['source_id']) for item, supported in zip(witnesses['selected'], witness_supported)
            if supported and not bool(item['raw_dependency_overlap_with_fixed_bank'])]
        witness_baseline_state = witnesses['state']
        witness_baseline_reason = witnesses['reason']
        witness_effective_state = 'frozen' if witness_appearance_usable >= MIN_WITNESSES else 'unavailable'
        witness_effective_reason = (None if witness_effective_state == 'frozen' else
            'fewer_than_8_source_appearance_supported_witnesses_without_fixed_bank_overlap')
        witness_records = _serializable_witness_bank(witnesses, template_id, source_image_sha)
        witness_records.update(
            selector_state=witness_baseline_state,
            selector_reason=witness_baseline_reason,
            state=witness_effective_state,
            reason=witness_effective_reason,
            reserved_denominator=int(witnesses['denominator']),
            usable_count=int(witness_appearance_usable),
            usable_count_before_appearance_support=int(witnesses['usable_count']),
            appearance_unsupported_ids=ordered_id_record(witness_appearance_unsupported_ids))
        for witness_row in witness_records['witnesses']:
            source_id = int(witness_row['source_id'])
            witness_row['appearance_supported'] = bool(appearance_supported[index_by_id[source_id]])
            witness_row['appearance_reasons'] = appearance_reasons.get(source_id, [])
        fit_selected_ids = [int(value) for value in fits['ordered_ids']['ordered_ids']]
        fit_supported_ids = [source_id for source_id in fit_selected_ids
                             if appearance_supported[index_by_id[source_id]]]
        fit_appearance_unsupported_ids = [source_id for source_id in fit_selected_ids
                                          if not appearance_supported[index_by_id[source_id]]]
        fit_effective_state = 'frozen' if len(fit_supported_ids) >= MIN_FIT_SAMPLES else 'unavailable'
        fit_record = dict(
            state=fit_effective_state,
            reason=None if fit_effective_state == 'frozen' else
                   'fewer_than_24_source_appearance_supported_fit_ids_without_refill',
            selector_state=fits['state'], selector_reason=fits['reason'],
            count=len(fit_supported_ids), selected_count=int(fits['count']),
            max_count=int(fits['max_count']),
            ordered_ids=ordered_id_record(fit_supported_ids),
            selector_ordered_ids=fits['ordered_ids'],
            appearance_unsupported_ids=ordered_id_record(fit_appearance_unsupported_ids),
            cell_counts=fits['cell_counts'])
        selected_ids = set(fit_selected_ids)
        selected_ids.update(witnesses['ordered_ids']['ordered_ids'])
        selected_ids.update(bank_dep_by_id)
        selected_ids_ordered = sorted(selected_ids)
        rows_for_assoc = np.asarray([index_by_id[source_id] for source_id in selected_ids_ordered
                                     if source_id in index_by_id], dtype=np.int64)
        if len(rows_for_assoc) != len(selected_ids_ordered):
            raise PrerequisiteError(f'Selected source association is missing from {template_id}')
        source_pose = np.asarray(source['template_pose_m'], dtype=np.float64)
        k = np.asarray(source['crop_k'], dtype=np.float64)
        q = xy[rows_for_assoc]
        rays_cam = np.column_stack((q, np.ones(len(q), dtype=np.float64))) @ np.linalg.inv(k).T
        rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
        rays_obj = rays_cam @ source_pose[:3, :3]
        origins_obj = np.broadcast_to(-source_pose[:3, :3].T @ source_pose[:3, 3], rays_obj.shape).copy()
        ray = raycast_first_hits(origins_obj, rays_obj, mesh.positions,
                                 mesh.triangles, resident_bytes=int(rgb.nbytes + points.nbytes +
                                 xy.nbytes + ids.nbytes + std.nbytes))
        point_error = np.linalg.norm(ray['point_m'] - points[rows_for_assoc], axis=1)
        eps = max(GEOMETRY_EPSILON_ABS_M,
                  GEOMETRY_EPSILON_DIAGONAL_FRACTION * diagonal)
        associated = ray['hit'] & np.isfinite(point_error) & (point_error <= eps)
        topology_bad = np.asarray([int(value) in mesh_unsupported_ids
                                   for value in ray['triangle_id']], dtype=bool)
        topology_bad &= ray['hit']
        topology_bad_source_ids = np.asarray(selected_ids_ordered, dtype=np.int64)[topology_bad]
        effective_association_supported = associated & ~topology_bad
        support = dict(sample_count=int(len(rows_for_assoc)), associated_count=int(associated.sum()),
                       unsupported_count=int((~associated).sum()),
                       effective_association_supported_count=int(effective_association_supported.sum()),
                       effective_association_unsupported_count=int((~effective_association_supported).sum()),
                       epsilon_m=float(eps),
                        topology_unsupported_association_count=int(topology_bad.sum()),
                        topology_unsupported_source_ids=ordered_id_record(topology_bad_source_ids),
                       source_ids_sha256=sha256_bytes(np.asarray(selected_ids_ordered, dtype='<i8').tobytes()),
                       triangle_ids_sha256=sha256_bytes(np.asarray(ray['triangle_id'], dtype='<i8').tobytes()),
                       point_error_sha256=array_record(point_error)['sha256'],
                       ray_chunk_size=int(ray['chunk_size']),
                       max_ray_scratch_bytes=int(ray['estimated_scratch_bytes']))
        association_supported_by_id = {
            int(source_id): bool(valid)
            for source_id, valid in zip(selected_ids_ordered, effective_association_supported)}
        association_detail_by_id = {
            int(source_id): dict(
                supported=bool(valid),
                ray_hit=bool(ray['hit'][row]),
                geometry_point_supported=bool(associated[row]),
                topology_supported=not bool(topology_bad[row]),
                triangle_id=int(ray['triangle_id'][row]),
                point_error_m=(float(point_error[row]) if math.isfinite(float(point_error[row])) else None))
            for row, (source_id, valid) in enumerate(zip(selected_ids_ordered,
                                                         effective_association_supported))}
        association_filter = association_filtered_source_support(
            [int(item['source_id']) for item in witnesses['selected']],
            witness_usable_before_association_ids,
            fit_selected_ids, fit_supported_ids,
            association_supported_by_id)
        if association_filter['witness_reserved_denominator'] != int(witnesses['denominator']):
            raise PrerequisiteError(f'Native association filtering changed the witness denominator in {template_id}')
        witness_effective_state = association_filter['witness_state']
        witness_records.update(
            state=witness_effective_state,
            reason=(None if witness_effective_state == 'frozen' else
                    'fewer_than_8_geometry_supported_witnesses_without_refill'),
            usable_count=association_filter['witness_usable_count'],
            usable_count_before_native_association=association_filter[
                'witness_usable_count_before_association'],
            native_association_unsupported_ids=ordered_id_record(
                association_filter['witness_unsupported_selected_ids']))
        for witness_row in witness_records['witnesses']:
            source_id = int(witness_row['source_id'])
            association_detail = association_detail_by_id.get(source_id)
            witness_row['native_association'] = (association_detail if association_detail is not None
                                                  else dict(supported=False, reason='association_not_sampled'))
        fit_effective_state = association_filter['fit_state']
        fit_supported_ids = association_filter['fit_usable_ids']
        fit_record.update(
            state=fit_effective_state,
            reason=(None if fit_effective_state == 'frozen' else
                    'fewer_than_24_geometry_supported_fit_ids_without_refill'),
            count=association_filter['fit_usable_count'],
            count_before_native_association=association_filter['fit_usable_count_before_association'],
            ordered_ids=ordered_id_record(fit_supported_ids),
            native_association_unsupported_ids=ordered_id_record(
                association_filter['fit_unsupported_selected_ids']))
        support['association_supported_ids'] = ordered_id_record(
            [source_id for source_id in selected_ids_ordered
             if association_supported_by_id.get(source_id, False)])
        support['association_unsupported_ids'] = ordered_id_record(
            [source_id for source_id in selected_ids_ordered
             if not association_supported_by_id.get(source_id, False)])
        support['association_support_by_source_id'] = [
            dict(source_id=source_id, **association_detail_by_id[source_id])
            for source_id in selected_ids_ordered]
        source_bank_mask_records = [dict(
            source_image_id=template_id, source_template_id=template_id,
            rgb_sha256=source_image_sha, source_id=int(anchor['source_index']),
            count=len(bank_dep_by_id[int(anchor['source_index'])]),
            sha256=sha256_bytes(np.asarray(bank_dep_by_id[int(anchor['source_index'])], dtype='<u4').tobytes()),
            ids_u32_le_base64=_compress_dependencies(bank_dep_by_id[int(anchor['source_index'])]))
            for anchor in anchors]
        opposite_bank_mask_records = [dict(
            source_image_id=opposite_id, source_template_id=opposite_id,
            rgb_sha256=opposite_image_sha, source_id=int(anchor['source_index']),
            count=len(opposite_bank_dep_by_id[int(anchor['source_index'])]),
            sha256=sha256_bytes(np.asarray(opposite_bank_dep_by_id[int(anchor['source_index'])], dtype='<u4').tobytes()),
            ids_u32_le_base64=_compress_dependencies(opposite_bank_dep_by_id[int(anchor['source_index'])]))
            for anchor in banks[opposite_id]['anchors']]
        fixed_bank_record = dict(
            anchor_count=len(anchors), source_image_id=template_id,
            source_rgb_sha256=source_image_sha,
            anchor_ids=ordered_id_record([int(anchor['source_index']) for anchor in anchors]),
            dependency_ids_count=len(source_bank_dependencies),
            dependency_ids_sha256=sha256_bytes(np.asarray(sorted(source_bank_dependencies), dtype='<u4').tobytes()),
            dependency_image_qualified=True,
            source_dependency_image=dict(source_image_id=template_id,
                source_template_id=template_id, rgb_sha256=source_image_sha,
                count=len(source_bank_dependencies),
                ids_sha256=sha256_bytes(np.asarray(sorted(source_bank_dependencies), dtype='<u4').tobytes())),
            opposite_dependency_image=dict(source_image_id=opposite_id,
                source_template_id=opposite_id,
                anchor_ids=ordered_id_record([int(anchor['source_index']) for anchor in banks[opposite_id]['anchors']]),
                rgb_sha256=opposite_image_sha, count=len(opposite_bank_dependencies),
                ids_sha256=sha256_bytes(np.asarray(sorted(opposite_bank_dependencies), dtype='<u4').tobytes())),
            raw_dependency_masks=source_bank_mask_records,
            opposite_raw_dependency_masks=opposite_bank_mask_records,
            source_appearance_unsupported_anchor_ids=ordered_id_record([
                int(source_id) for source_id in bank_dep_by_id
                if not appearance_supported[index_by_id[int(source_id)]]]),
            native_association_unsupported_anchor_ids=ordered_id_record([
                int(source_id) for source_id in bank_dep_by_id
                if not association_supported_by_id.get(int(source_id), False)]),
            manifest_sha256=banks[template_id]['manifest_sha256'])
        qh_partition_record, qf_partition_record = raw_pixel_partitions(rgb.shape[1], rgb.shape[0])
        view_records[template_id] = dict(
            source_packet_path=entry['path'], source_packet_sha256=entry['sha256'],
            rgb=array_record(rgb), observed_source_mask=array_record(observed),
            source_indices=array_record(ids), source_pixels_xy=array_record(xy),
            source_points_object_m=array_record(points),
            source_appearance_support=appearance_record,
            r5_eligibility=selections[template_id]['eligible_array'],
            eligibility_count=int(eligible.sum()), source_bbox=list(_mask_bbox(observed) or ()),
            source_cells_sha256=sha256_bytes(np.asarray(source_grid_cells(xy, observed), dtype='<i2').tobytes()),
            witness_bank=witness_records,
            fit_sources=fit_record,
            fixed_bank=fixed_bank_record,
            native_mesh_topology=mesh_topology,
            native_triangle_association=support,
            query_raw_partitions=dict(verification=array_record(qh_partition_record),
                                      fitting=array_record(qf_partition_record)))
        del source, rgb, observed, ids, xy, points, std, eroded, ray, origins_obj, rays_obj
    for plan in plans:
        row_id = plan['condition_id']
        source_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        source_view = r5._condition_source_only_inputs(bundle, plan)[1]
        observed = np.asarray(source_view['observed_crop_mask'], dtype=bool)
        verify_partition, fit_partition = raw_pixel_partitions(observed.shape[1], observed.shape[0])
        # Use the exact, unchanged OpenCV R5 erosion path on cached rows.
        eroded = cv2_module.distanceTransform(observed.astype(np.uint8), cv2_module.DIST_L2, 5) >= float(EROSION_PIXELS)
        qh = allowed_target_center_bitmap(observed, verify_partition, eroded)
        qf = allowed_target_center_bitmap(observed, fit_partition, eroded)
        verification_upper = qh_verification_support_upper_bound(qh, observed)
        view_records[source_id].setdefault('query_domains', {})[row_id] = dict(
            observed_mask_sha256=array_record(observed)['sha256'],
            qh_allowed_center_count=int(qh.sum()), qf_allowed_center_count=int(qf.sum()),
            qh_allowed_centers=encode_bitmap(qh), qf_allowed_centers=encode_bitmap(qf),
            verification_support_upper_bound=verification_upper,
            qh_dependency_partition='Q_H: raw 4×4 cell parity even; every D(q) inside before read',
            qf_dependency_partition='Q_F: raw 4×4 cell parity odd; every D(q) inside before read')
        row_prerequisite = source_row_prerequisite_state(
            witness_state=view_records[source_id]['witness_bank']['state'],
            fit_state=view_records[source_id]['fit_sources']['state'],
            fixed_bank_supported=(not bool(view_records[source_id]['fixed_bank'][
                'source_appearance_unsupported_anchor_ids']['count'])),
            qh_upper_bound_state=verification_upper['state'],
            unsupported_association_count=view_records[source_id][
                'native_triangle_association']['effective_association_unsupported_count'])
        row_records[row_id] = dict(
            source_template_id=source_id, query_mask_sha256=array_record(observed)['sha256'],
            qh_count=int(qh.sum()), qf_count=int(qf.sum()),
            verification_support_upper_bound=verification_upper,
            state=row_prerequisite['state'],
            unavailable_reasons=row_prerequisite['unavailable_reasons'],
            association_count_is_row_gate=row_prerequisite['association_count_is_row_gate'])
        del source_view
    return view_records, row_records


def _planned_synthetic_case_ids() -> list[str]:
    return [f'{variant}-seed-{int(seed):03d}' for variant in CONTROL_IDS for seed in SOURCE_YAWS_DEG]


def control_case_declarations() -> list[dict]:
    rows = []
    for variant_id in CONTROL_IDS:
        for seed in SOURCE_YAWS_DEG:
            rows.append(dict(case_id=f'{variant_id}-seed-{int(seed):03d}',
                             variant=variant_id, source_seed_deg=float(seed),
                             query_yaw_deg=QUERY_YAW_DEG,
                             challenge_yaw_offsets_deg=list(CHALLENGE_YAW_DEG),
                             control_state='declared_only', stage_a1_action='no_fit_or_score'))
    return rows


def _fixture_variants() -> dict[str, dict]:
    return {
        'distinct_markers': dict(source_texture='markers', query_texture='markers', edit=None),
        'blank_constant': dict(source_texture='constant', query_texture='constant', edit=None),
        'repeated_2d_periodic': dict(source_texture='periodic_2d', query_texture='periodic_2d', edit=None),
        'coherent_wrong_180_packet': dict(source_texture='markers', query_texture='markers', edit=None),
        'query_occlusion': dict(source_texture='markers', query_texture='markers', edit='occlusion'),
        'mask_leakage_surrogate': dict(source_texture='markers', query_texture='markers', edit='leakage'),
        'global_affine': dict(source_texture='markers', query_texture='markers', edit='affine'),
        'appearance_mismatch': dict(source_texture='markers', query_texture='base', edit=None),
    }


def _fixture_live_budget(mesh: Mapping[str, np.ndarray] | None = None,
                         textures: Sequence[np.ndarray] = ()) -> dict:
    """Conservative pre-allocation peak for two views plus source/query inputs."""
    mesh_bytes = (sum(int(np.asarray(mesh[name]).nbytes)
                      for name in ('positions', 'triangles', 'uv', 'triangle_normals'))
                  if mesh is not None else 132 * 3 * 8 + 256 * 3 * 8 + 132 * 2 * 8 + 256 * 3 * 8)
    texture_bytes = (sum(int(np.asarray(value).nbytes) for value in textures)
                     if textures else 2 * 256 * 256 * 3 * 4)
    pixels = IMAGE_WIDTH * IMAGE_HEIGHT
    view_resident = pixels * 160 + min(MAX_RAY_CHUNK, pixels) * 256 * RAY_PAIR_SCRATCH_BYTES
    generator_temporaries = 18 * 1024**2 + 18 * 1024**2 + 6 * 1024**2
    source_query_copies = pixels * 96
    selector_preflight = pixels * 48 + 3 * (IMAGE_WIDTH + 1) * (IMAGE_HEIGHT + 1) * 8
    estimate = (mesh_bytes + texture_bytes + 2 * view_resident + generator_temporaries +
                source_query_copies + selector_preflight)
    # Pure fixtures do not write into the model cache.  An explicit zero keeps
    # repeated unit fixture generation from rescanning a multi-gigabyte cache;
    # actual A1 preparation takes and enforces one real cache-size snapshot.
    return prospective_resource_budget(live_bytes=0, requested_bytes=int(estimate),
                                        cache_bytes=0)


def generate_control_case(variant_id: str, source_seed_deg: float) -> dict:
    """Build one deterministic image/control case with evaluator truth isolated."""
    if variant_id not in _fixture_variants() or source_seed_deg not in SOURCE_YAWS_DEG:
        raise ValueError('Unknown frozen control or source seed')
    variant = _fixture_variants()[variant_id]
    # This estimate precedes mesh, texture, raster or source-sample allocation.
    budget = _fixture_live_budget()
    mesh = fixture_cylinder_mesh()
    source_texture = fixture_texture(variant['source_texture'])
    query_texture = fixture_texture(variant['query_texture'])
    source_pose = _pose_y(source_seed_deg)
    truth_pose = _pose_y(QUERY_YAW_DEG)
    source = _render_fixture_view(mesh, source_texture, source_pose)
    source_live_bytes = sum(int(value.nbytes) for value in source.values() if isinstance(value, np.ndarray))
    query = _render_fixture_view(mesh, query_texture, truth_pose, resident_bytes=source_live_bytes)
    query_rgb = query['rgb'].copy()
    observed_query_mask = query['observed_mask'].copy()
    if variant['edit'] == 'occlusion':
        region = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
        region[:, 105:175] = True
        query_rgb[region] = np.float32(.5)
        observed_query_mask[region] = False
    elif variant['edit'] == 'leakage':
        xs = np.arange(IMAGE_WIDTH, dtype=np.int64)
        stripe = .25 + .5 * ((xs // 8) % 2)
        band = (xs >= 105) & (xs < 175)
        query_rgb[:, band, :] = stripe[band][None, :, None]
    elif variant['edit'] == 'affine':
        query_rgb = (.8 * query_rgb.astype(np.float64) + .05).astype(np.float32)
    if variant_id == 'coherent_wrong_180_packet':
        endpoint_pose = _pose_y(188.)
    else:
        endpoint_pose = truth_pose
    source_rows = np.flatnonzero(source['observed_mask'].reshape(-1))
    source_points = source['source_points_object_m'].reshape(-1, 3)[source_rows].copy()
    q_source = np.column_stack(((source_rows % IMAGE_WIDTH).astype(np.float64) + .5,
                                (source_rows // IMAGE_WIDTH).astype(np.float64) + .5))
    q0, valid = project_crop_points(source_points, endpoint_pose, K_FIXTURE,
                                    center_object_m=mesh['center_m'])
    q0[valid] += np.asarray((2., -2.), dtype=np.float64)
    q0[~valid] = np.nan
    estimator_inputs = dict(
        source_rgb=source['rgb'], query_rgb=query_rgb,
        source_observed_mask=source['observed_mask'],
        query_observed_mask=observed_query_mask,
        source_depth_mm=source['depth_mm'],
        source_ids=source_rows.astype(np.int64), source_pixels_xy=q_source,
        source_points_object_m=source_points,
        source_triangle_ids=source['hit_triangle_ids'].reshape(-1)[source_rows].copy(),
        source_barycentric=source['hit_barycentric'].reshape(-1, 3)[source_rows].copy(),
        source_uv=np.einsum('ni,nij->nj',
            source['hit_barycentric'].reshape(-1, 3)[source_rows],
            mesh['uv'][mesh['triangles'][source['hit_triangle_ids'].reshape(-1)[source_rows]]],
            optimize=False),
        source_confidence=np.ones(len(source_rows), dtype=np.float32),
        fixed_mesh=dict(positions_m=mesh['positions'], triangles=mesh['triangles'], uv=mesh['uv'],
                        # The estimator's fixed texture is attached from its
                        # source packet. Query-only appearance is never copied
                        # into the model asset (notably variant 8).
                        texture_rgb=source_texture),
        intrinsics=K_FIXTURE.copy(), source_seed_pose_m=source_pose.copy(),
        endpoint_stub_q0=q0, endpoint_stub_valid=valid,
        source_seed_deg=float(source_seed_deg),
        challenge_yaw_offsets_deg=np.asarray(CHALLENGE_YAW_DEG, dtype=np.float64),
    )
    evaluator_truth = dict(query_pose_camera_from_object_m=truth_pose,
                           query_depth_mm=query['depth_mm'],
                           endpoint_packet_pose_m=endpoint_pose)
    return dict(case_id=f'{variant_id}-seed-{int(source_seed_deg):03d}',
                variant=variant_id, source_seed_deg=float(source_seed_deg),
                estimator_inputs=estimator_inputs, evaluator_truth=evaluator_truth,
                generator_arrays=dict(mesh_positions=mesh['positions'], mesh_triangles=mesh['triangles'],
                    mesh_uv=mesh['uv'], mesh_triangle_normals=mesh['triangle_normals'],
                    source_texture=source_texture, query_texture=query_texture),
                geometry=dict(radius_m=RADIUS_M, height_m=2. * HALF_HEIGHT_M,
                              center_m=mesh['center_m'].copy(),
                              yaw_axis_object= np.asarray((0., 1., 0.), dtype=np.float64),
                              diagonal_m=mesh['diagonal_m'], intrinsics=K_FIXTURE.copy()),
                resource_budget=budget,
                generator_rules=dict(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                    query_yaw_deg=QUERY_YAW_DEG, triangle_count=256,
                    raster='pixel ray K^-1(x+.5,y+.5,1), two-sided float64 Moller-Trumbore',
                    determinant_tolerance=1e-12, barycentric_tolerance=1e-12,
                    ray_tie_tolerance_m=1e-12, texture_sampling='nearest floor(256*u/v)',
                    antialiasing=False, lighting=False, shadows=False,
                    query_edit=variant['edit']))


def fixture_case_digest(case: Mapping[str, object]) -> dict:
    inputs = case['estimator_inputs']
    truth = case['evaluator_truth']
    return dict(case_id=case['case_id'], variant=case['variant'],
                source_seed_deg=case['source_seed_deg'],
                estimator_inputs={name: array_record(value) for name, value in inputs.items()
                                  if isinstance(value, np.ndarray)},
                fixed_mesh={name: array_record(value) for name, value in inputs['fixed_mesh'].items()},
                generator_arrays={name: array_record(value)
                                  for name, value in case['generator_arrays'].items()},
                geometry={name: (array_record(value) if isinstance(value, np.ndarray)
                                 else value)
                          for name, value in case['geometry'].items()},
                generator_rules=case['generator_rules'],
                resource_budget=case['resource_budget'],
                evaluator_truth={name: array_record(value) for name, value in truth.items()
                                 if isinstance(value, np.ndarray)},
                truth_fields_separate_from_estimator=True,
                stage_a1_state='declared_arrays_only_no_fit')


def _array_tree_bytes(value: object) -> int:
    """Count unique owning NumPy buffers under a generated case mapping."""
    seen_containers: set[int] = set()
    seen_owners: set[int] = set()
    total = 0

    def visit(item: object) -> None:
        nonlocal total
        if isinstance(item, np.ndarray):
            owner = item
            while isinstance(getattr(owner, 'base', None), np.ndarray):
                owner = owner.base
            if id(owner) not in seen_owners:
                seen_owners.add(id(owner))
                total += int(owner.nbytes)
        elif isinstance(item, Mapping):
            if id(item) in seen_containers:
                return
            seen_containers.add(id(item))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            if id(item) in seen_containers:
                return
            seen_containers.add(id(item))
            for child in item:
                visit(child)

    visit(value)
    return int(total)


def _qualified_dependency_union_record(source_image_id: str, source_rgb_sha256: str,
                                       dependencies: Sequence[int]) -> dict:
    values = np.asarray(sorted(set(int(x) for x in dependencies)), dtype='<u4')
    return dict(source_image_id=source_image_id, source_rgb_sha256=source_rgb_sha256,
                count=int(len(values)), sha256=sha256_bytes(values.tobytes()),
                ids_u32_le_base64=_compress_dependencies(values))


def _fixture_case_a1_preflight(case: Mapping[str, object], cv2_module,
                               audit_module) -> dict:
    """Freeze source-only selectors and both query partitions; read no outcomes."""
    inputs = case['estimator_inputs']
    source_rgb = np.asarray(inputs['source_rgb'], dtype=np.float32)
    source_mask = np.asarray(inputs['source_observed_mask'], dtype=bool)
    source_ids = np.asarray(inputs['source_ids'], dtype=np.int64)
    source_xy = np.asarray(inputs['source_pixels_xy'], dtype=np.float64)
    source_points = np.asarray(inputs['source_points_object_m'], dtype=np.float64)
    query_mask = np.asarray(inputs['query_observed_mask'], dtype=bool)
    resident = _array_tree_bytes(case)
    appearance_record, appearance_supported = _source_appearance_support_for_ids(
        source_ids, source_xy, source_mask, np.asarray(inputs['source_depth_mm']),
        resident_bytes=resident)
    appearance_reasons = _appearance_reason_map(appearance_record)
    fixed_mesh = inputs['fixed_mesh']
    mesh_topology = native_mesh_topology(fixed_mesh['positions_m'], fixed_mesh['triangles'])
    mesh_unsupported_ids = set(mesh_topology['unsupported_triangle_ids'])
    source_triangle_ids = np.asarray(inputs['source_triangle_ids'], dtype=np.int64).reshape(-1)
    if len(source_triangle_ids) != len(source_ids):
        raise PrerequisiteError('Fixture source triangle association shape mismatch')
    fixture_association_supported_by_id = {
        int(source_id): bool(int(triangle_id) >= 0 and
                             int(triangle_id) not in mesh_unsupported_ids)
        for source_id, triangle_id in zip(source_ids, source_triangle_ids)}
    image_id = str(case['case_id']) + ':source'
    image_sha = array_record(source_rgb)['sha256']
    pixels = int(source_rgb.shape[0] * source_rgb.shape[1])
    source_count = int(len(source_ids))
    scratch = pixels * 52 + source_count * 192 + 8 * 1024**2
    budget = prospective_resource_budget(live_bytes=resident, requested_bytes=scratch,
                                         cache_bytes=0)
    eligible, std, eroded = _highpass_statistics(source_rgb, source_mask, source_xy,
                                                  cv2_module, audit_module)
    bank = audit_module.build_template_patch_bank(
        source_rgb, source_mask, source_ids, source_points, grid_size=16, max_anchors=64)
    bank_dependencies_by_id = {}
    for anchor in bank['anchors']:
        bank_dependencies_by_id[int(anchor['source_index'])] = highpass_patch_raw_dependencies(
            anchor['source_xy_crop'], source_rgb.shape[1], source_rgb.shape[0])
    bank_dependencies = set(dep for deps in bank_dependencies_by_id.values() for dep in deps)
    witnesses = select_witness_ids(
        source_ids, source_xy, std, eligible, source_mask,
        lambda q: highpass_patch_raw_dependencies(q, source_rgb.shape[1], source_rgb.shape[0]),
        bank_dependencies)
    fits = select_fit_ids(
        source_ids, source_xy, eligible, source_mask,
        lambda q: highpass_patch_raw_dependencies(q, source_rgb.shape[1], source_rgb.shape[0]),
        witnesses['dependency_union'])
    witness_supported = [bool(appearance_supported[int(item['row'])])
                         for item in witnesses['selected']]
    witness_unsupported_ids = [int(item['source_id'])
                               for item, supported in zip(witnesses['selected'], witness_supported)
                               if not supported]
    witness_appearance_usable = sum(
        supported and not bool(item['raw_dependency_overlap_with_fixed_bank'])
        for item, supported in zip(witnesses['selected'], witness_supported))
    witness_usable_before_association_ids = [
        int(item['source_id']) for item, supported in zip(witnesses['selected'], witness_supported)
        if supported and not bool(item['raw_dependency_overlap_with_fixed_bank'])]
    fit_selected_ids = [int(value) for value in fits['ordered_ids']['ordered_ids']]
    fit_appearance_supported_ids = [source_id for source_id in fit_selected_ids
                                    if appearance_supported[int(np.searchsorted(source_ids, source_id))]]
    fit_unsupported_ids = [source_id for source_id in fit_selected_ids
                           if not appearance_supported[int(np.searchsorted(source_ids, source_id))]]
    association_filter = association_filtered_source_support(
        [int(item['source_id']) for item in witnesses['selected']],
        witness_usable_before_association_ids, fit_selected_ids,
        fit_appearance_supported_ids, fixture_association_supported_by_id)
    if association_filter['witness_reserved_denominator'] != int(witnesses['denominator']):
        raise PrerequisiteError('Fixture native association filtering changed the witness denominator')
    witness_effective_state = association_filter['witness_state']
    fit_supported_ids = association_filter['fit_usable_ids']
    fit_effective_state = association_filter['fit_state']
    fit_association_unsupported_ids = association_filter['fit_unsupported_selected_ids']
    fit_dependencies = set()
    all_fit_dependencies = set()
    fit_dependency_by_id = {}
    id_to_row = {int(source_id): row for row, source_id in enumerate(source_ids.tolist())}
    for source_id in fit_selected_ids:
        deps = highpass_patch_raw_dependencies(source_xy[id_to_row[int(source_id)]],
                                               source_rgb.shape[1], source_rgb.shape[0])
        fit_dependency_by_id[int(source_id)] = deps
        all_fit_dependencies.update(deps)
        if (appearance_supported[id_to_row[int(source_id)]] and
                fixture_association_supported_by_id.get(int(source_id), False)):
            fit_dependencies.update(deps)
    if all_fit_dependencies.intersection(witnesses['dependency_union']):
        raise PrerequisiteError('Fixture fit source dependencies intersect held-out dependencies')
    if witnesses['denominator'] != len(witnesses['ordered_ids']['ordered_ids']):
        raise PrerequisiteError('Fixture witness reservation denominator lost original IDs')
    source_bank_masks = [dict(source_image_id=image_id, source_rgb_sha256=image_sha,
        source_id=int(anchor['source_index']),
        dependencies=_qualified_dependency_union_record(
            image_id, image_sha, bank_dependencies_by_id[int(anchor['source_index'])]))
        for anchor in bank['anchors']]
    witness_records = []
    for witness, appearance_ok in zip(witnesses['selected'], witness_supported):
        source_id = int(witness['source_id'])
        witness_records.append(dict(source_image_id=image_id, source_rgb_sha256=image_sha,
            source_id=source_id, cell=int(witness['cell']),
            source_std=float(witness['source_std']),
            fixed_bank_overlap=bool(witness['raw_dependency_overlap_with_fixed_bank']),
            appearance_supported=bool(appearance_ok),
            native_association_supported=bool(
                fixture_association_supported_by_id.get(source_id, False)),
            appearance_reasons=appearance_reasons.get(source_id, []),
            dependencies=_qualified_dependency_union_record(
                image_id, image_sha, witness['raw_dependencies'])))
    qh, qf = raw_pixel_partitions(query_mask.shape[1], query_mask.shape[0])
    qeroded = audit_module._eroded_mask(query_mask, EROSION_PIXELS)
    qh_centers = allowed_target_center_bitmap(query_mask, qh, qeroded)
    qf_centers = allowed_target_center_bitmap(query_mask, qf, qeroded)
    verification_upper = qh_verification_support_upper_bound(qh_centers, query_mask)
    bank_unsupported_ids = [int(source_id) for source_id in bank_dependencies_by_id
                            if not appearance_supported[id_to_row[int(source_id)]]]
    topology_unsupported_source_ids = source_ids[np.isin(source_triangle_ids,
                                                         list(mesh_unsupported_ids))]
    row_prerequisite = source_row_prerequisite_state(
        witness_state=witness_effective_state,
        fit_state=fit_effective_state,
        fixed_bank_supported=not bool(bank_unsupported_ids),
        qh_upper_bound_state=verification_upper['state'],
        unsupported_association_count=int(sum(
            not fixture_association_supported_by_id.get(int(source_id), False)
            for source_id in set(fit_selected_ids) |
                {int(item['source_id']) for item in witnesses['selected']})))
    selected_topology_unsupported_ids = list(dict.fromkeys(
        association_filter['witness_unsupported_selected_ids'] +
        association_filter['fit_unsupported_selected_ids']))
    prerequisite_reasons = []
    reason_names = {
        'source_witness_support_unavailable': 'source_witness_appearance_or_geometry_support_unavailable',
        'source_fit_support_unavailable': 'source_fit_appearance_or_geometry_support_unavailable',
        'fixed_bank_source_appearance_unsupported': 'fixed_bank_source_appearance_unsupported',
        'verification_support_upper_bound_known_impossible': 'verification_support_upper_bound_known_impossible',
        'verification_support_upper_bound_unavailable': 'verification_support_upper_bound_unavailable'}
    prerequisite_reasons.extend(reason_names.get(reason, reason)
                                for reason in row_prerequisite['unavailable_reasons'])
    return dict(state='frozen_source_only_preflight',
        source_image_id=image_id, source_rgb_sha256=image_sha,
        source_population_count=source_count,
        source_population_order=ordered_id_record(source_ids),
        source_xy=array_record(source_xy), source_points=array_record(source_points),
        source_appearance_support=appearance_record,
        source_eligibility=dict(eligible_count=int(eligible.sum()),
            ordered_ids=ordered_id_record(source_ids[eligible]),
            std_sha256=array_record(std)['sha256'], eroded_mask=array_record(eroded)),
        fixed_bank=dict(state=bank['state'], anchor_count=int(bank['selected_anchor_count']),
            anchor_ids=ordered_id_record([int(anchor['source_index']) for anchor in bank['anchors']]),
            source_image_id=image_id, source_rgb_sha256=image_sha,
            dependency_masks=source_bank_masks,
            dependency_union=_qualified_dependency_union_record(image_id, image_sha,
                                                                  sorted(bank_dependencies)),
            unsupported_appearance_ids=ordered_id_record(bank_unsupported_ids),
            manifest_sha256=sha256_bytes(canonical_json_bytes(
                [dict(source_index=row['source_index'], highpass_std=row['highpass_std'])
                 for row in bank['serialized_anchors']]))),
        witness_bank=dict(state=witness_effective_state,
            reason=None if witness_effective_state == 'frozen' else
                'fewer_than_8_source_appearance_and_geometry_supported_witnesses_without_fixed_bank_overlap',
            selector_state=witnesses['state'], selector_reason=witnesses['reason'],
            denominator=witnesses['denominator'], reserved_denominator=witnesses['denominator'],
            usable_count=association_filter['witness_usable_count'],
            usable_count_before_native_association=association_filter[
                'witness_usable_count_before_association'],
            usable_count_before_appearance_support=witnesses['usable_count'],
            native_association_unsupported_ids=ordered_id_record(
                association_filter['witness_unsupported_selected_ids']),
            appearance_unsupported_ids=ordered_id_record(witness_unsupported_ids),
            source_image_id=image_id, source_rgb_sha256=image_sha,
            ordered_ids=witnesses['ordered_ids'], cell_counts=witnesses['cell_counts'],
            rejected_overlap_ids=witnesses['rejected_witness_overlap_ids'],
            unusable_fixed_bank_overlap_count=witnesses['unusable_fixed_bank_overlap_count'],
            dependencies=witness_records,
            dependency_union=_qualified_dependency_union_record(
                image_id, image_sha, witnesses['dependency_union'])),
        fit_sources=dict(state=fit_effective_state,
            reason=None if fit_effective_state == 'frozen' else
                'fewer_than_24_source_appearance_and_geometry_supported_fit_ids_without_refill',
            selector_state=fits['state'], selector_reason=fits['reason'],
            count=len(fit_supported_ids), selected_count=int(fits['count']),
            count_before_native_association=association_filter['fit_usable_count_before_association'],
            ordered_ids=ordered_id_record(fit_supported_ids),
            selector_ordered_ids=fits['ordered_ids'],
            appearance_unsupported_ids=ordered_id_record(fit_unsupported_ids),
            native_association_unsupported_ids=ordered_id_record(fit_association_unsupported_ids),
            cell_counts=fits['cell_counts'],
            source_image_id=image_id, source_rgb_sha256=image_sha,
            dependency_union=_qualified_dependency_union_record(
                image_id, image_sha, sorted(fit_dependencies)),
            dependency_id_count=sum(len(fit_dependency_by_id[source_id])
                                    for source_id in fit_supported_ids),
            dependencies_disjoint_from_witnesses=True),
        query=dict(query_image_id=str(case['case_id']) + ':query',
            query_rgb_sha256=array_record(inputs['query_rgb'])['sha256'],
            observed_mask=array_record(query_mask), qh_partition=encode_bitmap(qh),
            qf_partition=encode_bitmap(qf),
            qh_centers=encode_bitmap(qh_centers), qf_centers=encode_bitmap(qf_centers),
            verification_support_upper_bound=verification_upper,
            qh_center_count=int(qh_centers.sum()), qf_center_count=int(qf_centers.sum()),
            source_query_namespaces_distinct=True,
            center_rule='legacy all-four eroded-mask taps; exact nonzero D(q) partition test'),
        geometric_max_support=dict(source_ray_hit_count=int(source_mask.sum()),
            source_sample_count=source_count,
            source_triangle_association_ids=array_record(inputs['source_triangle_ids']),
            topology_unsupported_source_ids=ordered_id_record(topology_unsupported_source_ids),
            selected_topology_unsupported_ids=ordered_id_record(selected_topology_unsupported_ids),
            native_mesh_topology=mesh_topology,
            source_barycentric=array_record(inputs['source_barycentric']),
            state='generator_ray_hits_declared_only_not_independent_native_mesh_validation',
            source_topology_association_state='unsupported' if len(topology_unsupported_source_ids)
                else 'supported'),
        prerequisite_state=('preflight_possible' if row_prerequisite['state'] == 'prepared'
                            else 'terminal_unavailable'),
        prerequisite_reasons=prerequisite_reasons,
        resource_budget=budget,
        estimator_truth_separate=True, matching_started=False, fitting_started=False,
        ncc_calls=0, forward_calls=0, optimization_calls=0, production_render_calls=0,
        rank_verified=False, performance_verified=False)


def build_procedural_control_manifest(*, cv2_module=None, audit_module=None,
                                     wall_started: float | None = None,
                                     wall_limit_seconds: int = 900) -> dict:
    """Build and seal all 8×2 CPU fixture inputs; never fit or score controls."""
    rows = []
    row_bytes = 1024
    for declaration in control_case_declarations():
        if wall_started is not None:
            _check_wall_deadline(wall_started, limit_seconds=wall_limit_seconds)
        case = generate_control_case(declaration['variant'], declaration['source_seed_deg'])
        record = fixture_case_digest(case)
        record.update(declaration)
        if cv2_module is not None and audit_module is not None:
            record['a1_source_split_preflight'] = _fixture_case_a1_preflight(
                case, cv2_module, audit_module)
        elif cv2_module is None and audit_module is None:
            record['a1_source_split_preflight'] = dict(
                state='not_run', reason='pinned_opencv_source_helpers_required')
        else:
            raise ValueError('Pinned cv2 and R5 audit helpers must be supplied together')
        prospective_resource_budget(live_bytes=0,
            requested_bytes=_array_tree_bytes(case), cache_bytes=0)
        row_bytes += len(canonical_json_bytes(record)) + 1
        if row_bytes > MAX_OUTPUT_BYTES:
            raise ResourceLimitError('Procedural A1 control manifest exceeds its 8 MiB artifact cap')
        rows.append(record)
        del case
        if wall_started is not None:
            _check_wall_deadline(wall_started, limit_seconds=wall_limit_seconds)
    coverage = summarize_control_preflight_coverage(rows)
    return dict(schema_version=1, spec_sha256=SPEC_SHA256,
                status='declared_only', planned_count=16, accounted_count=len(rows),
                forward_calls=0, fit_calls=0, ncc_calls=0, production_render_calls=0,
                optimization_calls=0, rank_verified=False, performance_verified=False,
                generator_source_sha256=file_sha256(Path(__file__)),
                preflight_coverage=coverage,
                controls=rows,
                note='Procedural CPU arrays and ray hits are declared inputs only; no hypothesis fitting, matching, NCC or performance evaluation occurred.')


def summarize_control_preflight_coverage(control_rows: Sequence[Mapping[str, object]]) -> dict:
    """Require support for mandatory positive/ambiguity controls; allow abstentions."""
    rows = list(control_rows)
    mandatory = set(MANDATORY_CONTROL_PREPREREQUISITE_VARIANTS)
    mandatory_unavailable = []
    source_failures = []
    blank_abstentions = []
    case_ids = [str(row.get('case_id', '')) for row in rows]
    expected_case_ids = {row['case_id'] for row in control_case_declarations()}
    missing_case_ids = sorted(expected_case_ids - set(case_ids))
    unexpected_case_ids = sorted(set(case_ids) - expected_case_ids)
    for row in rows:
        case_id = str(row.get('case_id', ''))
        variant = str(row.get('variant', ''))
        preflight = row.get('a1_source_split_preflight', {})
        if not isinstance(preflight, Mapping) or preflight.get('state') != 'frozen_source_only_preflight':
            source_failures.append(case_id)
            continue
        state = preflight.get('prerequisite_state')
        if variant == 'blank_constant' and state == 'terminal_unavailable':
            blank_abstentions.append(case_id)
        if variant in mandatory and state != 'preflight_possible':
            mandatory_unavailable.append(case_id)
    complete = (len(rows) == 16 and len(set(case_ids)) == 16 and
                not missing_case_ids and not unexpected_case_ids and not source_failures)
    passed = complete and not mandatory_unavailable
    return dict(state='passed' if passed else 'failed', all_rows_accounted=bool(complete),
        accounted_count=int(len(rows)), mandatory_variants=list(MANDATORY_CONTROL_PREPREREQUISITE_VARIANTS),
        mandatory_unavailable_case_ids=sorted(mandatory_unavailable),
        blank_negative_abstention_case_ids=sorted(blank_abstentions),
        source_preflight_failures=sorted(source_failures),
        missing_case_ids=missing_case_ids, unexpected_case_ids=unexpected_case_ids,
        rule='all 16 source preflights remain terminal-accounted; required positive and variant3 rows need possible support; blank-negative and other declared abstentions remain reportable')


def gaussian_dependency_fixture_record(center_xy: Sequence[float], width: int,
                                       height: int) -> dict:
    xs, ys = dependency_axes(center_xy, width, height)
    deps = highpass_patch_raw_dependencies(center_xy, width, height)
    return dict(center_xy=[float(x) for x in center_xy],
                axis_x=sorted(xs), axis_y=sorted(ys), dependency_count=len(deps),
                dependency_sha256=sha256_bytes(np.asarray(deps, dtype='<u4').tobytes()),
                gaussian=gaussian_blur_kernel_record())


def _source_dependency_mask(center_xy: Sequence[float], width: int, height: int) -> np.ndarray:
    deps = highpass_patch_raw_dependencies(center_xy, width, height)
    mask = guarded_zeros((height, width), np.bool_, label='one source dependency mask')
    if deps:
        mask.reshape(-1)[np.asarray(deps, dtype=np.int64)] = True
    return mask


def _actual_source_paths() -> dict[str, Path]:
    return {
        'bench/quality_bottle_joint_prerequisite.py': ROOT / 'bench' / 'quality_bottle_joint_prerequisite.py',
        'bench/test_quality_bottle_joint_prerequisite.py': ROOT / 'bench' / 'test_quality_bottle_joint_prerequisite.py',
        'bench/quality_bottle_pose_ablation.py': ROOT / 'bench' / 'quality_bottle_pose_ablation.py',
        'bench/quality_bottle_patch_pose_calibration.py': ROOT / 'bench' / 'quality_bottle_patch_pose_calibration.py',
        'bench/quality_bottle_identity_audit.py': ROOT / 'bench' / 'quality_bottle_identity_audit.py',
        'bench/quality_bottle_zero_view_probe.py': ROOT / 'bench' / 'quality_bottle_zero_view_probe.py',
        'bench/quality_contract.py': ROOT / 'bench' / 'quality_contract.py',
        'bench/quality_assets.py': ROOT / 'bench' / 'quality_assets.py',
        'bench/vision.py': ROOT / 'bench' / 'vision.py',
        'bench/glb_model.py': ROOT / 'bench' / 'glb_model.py',
        'bench/model.py': ROOT / 'bench' / 'model.py',
    }


def _source_freeze_before_import() -> dict:
    result = {}
    for name, path in _actual_source_paths().items():
        if not path.is_file():
            result[name] = dict(state='missing')
        else:
            result[name] = dict(state='bound', bytes=int(path.stat().st_size),
                                sha256=file_sha256(path))
    return result


def _source_freeze_revalidate(before: Mapping[str, Mapping[str, object]]) -> dict:
    after = _source_freeze_before_import()
    matched = all(name in after and after[name] == dict(before[name])
                  for name in before)
    return dict(state='matched' if matched else 'changed', before=dict(before), after=after)


def expected_actual_row_ids() -> list[str]:
    rows = []
    for frame in CARRIERS:
        rows.append(f'syn-{frame}-q8-rgb-t0-rgb')
        rows.extend((f'zero-{frame}-full', f'zero-{frame}-clipped'))
        rows.append(f'syn-{frame}-q8-rgb-t180-rgb')
    return rows


def _verify_frozen_files() -> dict:
    expected = {
        SPEC_PATH: SPEC_SHA256,
        CAPTURE_ROOT / 'capture.json': R1_CAPTURE_SHA256,
        RANCH_INPUT_MANIFEST: R1_INPUT_MANIFEST_SHA256,
        RANCH_MESH: R1_MESH_SHA256,
        ZERO_ROOT / 'zero_view.json': R3_REPORT_SHA256,
        R5_ROOT / 'pose_ablation.json': R5_REPORT_SHA256,
        R5_TERMINAL: R5_TERMINAL_SHA256,
        R6_V1_ROOT / 'calibration.json': R6_V1_REPORT_SHA256,
        R6_V2_ROOT / 'calibration.json': R6_V2_REPORT_SHA256,
        R6_V2_TERMINAL: R6_V2_TERMINAL_SHA256,
    }
    records = []
    for path, expected_hash in expected.items():
        if not path.is_file():
            records.append(dict(path=str(path), state='missing', expected_sha256=expected_hash))
            continue
        actual = file_sha256(path)
        records.append(dict(path=str(path), state='matched' if actual == expected_hash else 'changed',
                            bytes=int(path.stat().st_size), expected_sha256=expected_hash,
                            actual_sha256=actual))
    return dict(state='matched' if all(row['state'] == 'matched' for row in records) else 'failed',
                files=records)


def _contract_digest(contract: Mapping[str, object]) -> str:
    unsigned = {str(key): value for key, value in contract.items() if key != 'contract_sha256'}
    return sha256_bytes(canonical_json_bytes(unsigned))


def _loaded_project_import_closure() -> list[dict]:
    records = {}
    root = ROOT.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != 'bench' and not name.startswith('bench.'):
            continue
        path = getattr(module, '__file__', None)
        if not path:
            continue
        try:
            resolved = Path(path).resolve()
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if resolved.suffix.casefold() not in ('.py', '.pyc'):
            continue
        source = resolved
        if resolved.suffix.casefold() == '.pyc':
            spec = getattr(module, '__spec__', None)
            origin = getattr(spec, 'origin', None)
            if origin and str(origin).casefold().endswith('.py'):
                source = Path(origin).resolve()
            else:
                source = resolved.parents[1] / (resolved.name.split('.', 1)[0] + '.py')
        if source.is_file():
            records[name] = dict(module=name, path=source.as_posix(),
                                 relative_path=source.relative_to(root).as_posix(),
                                 sha256=file_sha256(source))
    return [records[name] for name in sorted(records)]


def _runtime_identity_record(runtime_root: Path,
                             runtime_pins: Mapping[str, str]) -> dict:
    executable = Path(sys.executable).resolve()
    return dict(implementation=sys.implementation.name,
                cache_tag=sys.implementation.cache_tag,
                platform=sys.platform,
                python_version=sys.version,
                prefix=str(Path(sys.prefix).resolve()),
                base_prefix=str(Path(sys.base_prefix).resolve()),
                executable=str(executable),
                executable_sha256=file_sha256(executable),
                site_packages=str(Path(runtime_root).resolve()),
                runtime_pins_sha256=sha256_bytes(canonical_json_bytes(runtime_pins)))


def _contract_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        if '..' in path.parts:
            raise StageGateError('Prebinding paths cannot traverse above their declared root')
        resolved = (root / path).resolve()
    return resolved


def _verify_contract_file_map(file_map: Mapping[str, str], *, root: Path,
                              label: str, allow_absolute: bool = False) -> list[dict]:
    if not isinstance(file_map, Mapping):
        raise StageGateError(f'Prebinding {label} must be a path-to-SHA256 mapping')
    rows = []
    for raw_path, raw_hash in sorted(file_map.items(), key=lambda item: str(item[0])):
        path_text = str(raw_path)
        candidate = Path(path_text)
        if candidate.is_absolute() and not allow_absolute:
            raise StageGateError(f'Prebinding {label} path must be root-relative: {path_text}')
        target = _contract_path(path_text, root)
        if not target.is_file() or target.is_symlink():
            rows.append(dict(path=path_text, state='missing_or_not_regular'))
            continue
        actual = file_sha256(target)
        expected = str(raw_hash).casefold()
        rows.append(dict(path=path_text, state='matched' if actual == expected else 'changed',
                         sha256=actual, expected_sha256=expected))
    if any(row['state'] != 'matched' for row in rows):
        raise StageGateError(f'Prebinding {label} files changed or are missing')
    return rows


def validate_prebinding_contract(contract: Mapping[str, object] | None, *,
                                 verify_files: bool = True,
                                 require_loaded_closure: bool = False) -> dict:
    """Validate the root launcher seal; file checks are mandatory in A1 prep.

    ``verify_files=False`` is provided only for isolated structural unit tests.
    ``prepare_actual_a1`` always uses the full file and import checks.
    """
    if not isinstance(contract, Mapping):
        raise StageGateError('A sealed root prebinding contract is required')
    if contract.get('schema_version') != 1 or contract.get('status') != 'sealed_preimport':
        raise StageGateError('Prebinding contract schema/status is not the frozen A1 contract')
    workspace_text = str(contract.get('workspace_root', '')).strip()
    if not workspace_text:
        raise StageGateError('Prebinding workspace root is missing')
    workspace = Path(workspace_text).resolve()
    if workspace != ROOT.resolve():
        raise StageGateError('Prebinding workspace root differs from this VisualizeIT checkout')
    expected_digest = str(contract.get('contract_sha256', '')).casefold()
    actual_digest = _contract_digest(contract)
    if not expected_digest or expected_digest != actual_digest:
        raise StageGateError('Prebinding contract digest is missing or invalid')
    expected_limits = dict(working_bytes=MAX_WORKING_BYTES, wall_seconds=900,
                           artifact_bytes=MAX_OUTPUT_BYTES, model_cache_bytes=MAX_CACHE_BYTES)
    if contract.get('limits') != expected_limits:
        raise StageGateError('Prebinding limits differ from the frozen A1 hard caps')
    interpreter = contract.get('interpreter')
    runtime_root_text = str(contract.get('runtime_root', '')).strip()
    if not runtime_root_text:
        raise StageGateError('Prebinding runtime root is missing')
    runtime_root = Path(runtime_root_text).resolve()
    runtime_identity = contract.get('runtime_identity')
    if not isinstance(interpreter, Mapping) or not runtime_identity or not str(runtime_root):
        raise StageGateError('Prebinding interpreter/runtime identity is incomplete')
    current_executable = Path(sys.executable).resolve()
    if Path(str(interpreter.get('path', ''))).resolve() != current_executable:
        raise StageGateError('Current Python executable differs from the root prebinding')
    pinned_executable = (ROOT / '.cache' / 'quality-windows' / 'Scripts' / 'python.exe').resolve()
    if current_executable != pinned_executable:
        raise StageGateError('Current Python executable is not the pinned quality environment')
    expected_runtime_root = (current_executable.parents[1] / 'Lib' / 'site-packages').resolve()
    if runtime_root != expected_runtime_root or not runtime_root.is_relative_to(ROOT.resolve()):
        raise StageGateError('Prebinding runtime root differs from the pinned interpreter site-packages')
    if str(interpreter.get('python_version', '')) != sys.version:
        raise StageGateError('Current Python version differs from the root prebinding')
    if not Path(sys.executable).is_file() or file_sha256(Path(sys.executable)) != str(
            interpreter.get('sha256', '')).casefold():
        raise StageGateError('Current Python executable hash differs from the root prebinding')
    closure = contract.get('import_closure')
    source_pins = contract.get('source_pins')
    runtime_pins = contract.get('runtime_pins', {})
    if (not isinstance(closure, list) or not closure or not isinstance(source_pins, Mapping) or
            not isinstance(runtime_pins, Mapping) or not runtime_pins):
        raise StageGateError('Prebinding must list a nonempty project import closure and source pins')
    expected_closure = {}
    closure_order = []
    for record in closure:
        if not isinstance(record, Mapping):
            raise StageGateError('Malformed import-closure record')
        name = str(record.get('module', ''))
        relative = str(record.get('relative_path', ''))
        path_text = str(record.get('path', ''))
        sha = str(record.get('sha256', '')).casefold()
        if not name or not relative or not path_text or len(sha) != 64:
            raise StageGateError('Import-closure record lacks module/path/hash')
        if name != 'bench' and not name.startswith('bench.'):
            raise StageGateError(f'Import closure includes a non-bench module: {name}')
        resolved = _contract_path(path_text, ROOT)
        if resolved != _contract_path(relative, ROOT) or not resolved.is_relative_to(ROOT.resolve()):
            raise StageGateError('Import-closure source path is inconsistent or outside the workspace')
        if source_pins.get(relative) is None or str(source_pins[relative]).casefold() != sha:
            raise StageGateError(f'Import closure is not source-pinned: {relative}')
        if name in expected_closure:
            raise StageGateError(f'Duplicate module in prebinding closure: {name}')
        expected_closure[name] = dict(path=resolved, sha256=sha)
        closure_order.append(name)
    if closure_order != sorted(closure_order):
        raise StageGateError('Prebinding import closure must be ordered by module name')
    if verify_files and tuple(closure_order) != EXPECTED_A1_BENCH_IMPORTS:
        raise StageGateError('Prebinding closure differs from the frozen 14-module A1 bench import set')
    runtime_expected = _runtime_identity_record(runtime_root, runtime_pins)
    runtime_observed = dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
    runtime_hash_fields = ('executable_sha256', 'runtime_pins_sha256')
    for hash_field in runtime_hash_fields:
        if hash_field in runtime_expected and hash_field in runtime_observed:
            runtime_expected[hash_field] = str(runtime_expected[hash_field]).casefold()
            runtime_observed[hash_field] = str(runtime_observed[hash_field]).casefold()
    if runtime_observed != runtime_expected:
        raise StageGateError('Current interpreter/runtime identity differs from root prebinding')
    if not isinstance(contract.get('output_root'), str) or not contract.get('output_root'):
        raise StageGateError('Prebinding output root is missing')
    output_root = Path(str(contract['output_root'])).resolve()
    private_root = (CACHE / 'diagnostics').resolve()
    if not output_root.is_relative_to(private_root):
        raise StageGateError('Prebinding output root is outside the private diagnostics cache')
    if output_root.is_symlink():
        raise StageGateError('Prebinding output root cannot be a symlink')
    summary = dict(state='structurally_valid', contract_sha256=actual_digest,
                   workspace_root=str(workspace), interpreter=dict(path=str(Path(sys.executable).resolve()),
                       sha256=str(interpreter['sha256']).casefold(), python_version=sys.version),
                   runtime_root=str(runtime_root), runtime_identity=runtime_identity,
                   output_root=str(output_root), limits=expected_limits,
                   expected_import_closure=[dict(module=name, path=str(value['path']),
                                                 sha256=value['sha256'])
                                            for name, value in sorted(expected_closure.items())])
    if not verify_files:
        return summary
    summary['source_pins'] = _verify_contract_file_map(source_pins, root=ROOT,
                                                       label='source pins')
    for path_text in runtime_pins:
        runtime_file = _contract_path(str(path_text), runtime_root)
        if not runtime_file.is_relative_to(runtime_root):
            raise StageGateError('Runtime pin escapes the pinned runtime root')
    summary['runtime_pins'] = _verify_contract_file_map(runtime_pins, root=runtime_root,
                                                        label='runtime pins', allow_absolute=True)
    loaded = _loaded_project_import_closure()
    loaded_by_name = {row['module']: row for row in loaded}
    for name, expected in expected_closure.items():
        actual = loaded_by_name.get(name)
        if actual is not None and (Path(actual['path']).resolve() != expected['path'] or
                                   actual['sha256'] != expected['sha256']):
            raise StageGateError(f'Loaded module differs from its prebound source: {name}')
    actual_extra = [row for name, row in loaded_by_name.items() if name not in expected_closure]
    if require_loaded_closure and (set(loaded_by_name) != set(expected_closure)):
        raise StageGateError('Loaded project import closure differs from the prebound closure')
    summary.update(state='files_matched', loaded_project_import_closure=loaded,
                   unanticipated_loaded_project_imports=actual_extra)
    if require_loaded_closure:
        summary['state'] = 'matched'
    return summary


def _check_wall_deadline(started: float, *, limit_seconds: int = 900) -> float:
    elapsed = time.monotonic() - started
    if elapsed > int(limit_seconds):
        raise ResourceLimitError(f'A1 preparation exceeded its {int(limit_seconds)} second wall cap')
    return float(elapsed)


def _existing_regular_bytes(root: Path, cap: int) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    if root.is_symlink():
        raise StageGateError('Output roots cannot be symlinks')
    total = 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise StageGateError('Output roots cannot contain symlinks')
        if path.is_file():
            total += int(path.stat().st_size)
            if total > int(cap):
                return total
    return int(total)


def _bounded_json_payload(value: object, *, cap: int) -> bytes:
    """Encode JSON incrementally, stopping before retaining an oversized payload."""
    encoder = json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False)
    parts: list[bytes] = []
    total = 0
    for chunk in encoder.iterencode(value):
        encoded = chunk.encode('utf-8')
        total += len(encoded)
        if total > int(cap):
            raise ResourceLimitError(f'JSON artifact exceeds its {int(cap)} byte cap')
        parts.append(encoded)
    return b''.join(parts)


def prepare_actual_a1(*, root_scheduled: bool = False,
                      output_path: Path | None = None,
                      prebinding_contract: Mapping[str, object] | None = None) -> dict:
    """Prepare the bounded read-only A1 manifest; never fit or write.

    The launcher seal is mandatory and checked before dynamic project imports.
    ``root_scheduled`` is a deliberate orchestration gate.  Manifest writing is
    a separate operation and cannot be requested through this preparation API.
    """
    if not root_scheduled:
        raise StageGateError('Actual Stage A1 cache preparation requires explicit root scheduling')
    if output_path is not None:
        raise StageGateError('Preparation is read-only; call the separately gated manifest writer')
    started = time.monotonic()
    prebinding = validate_prebinding_contract(prebinding_contract, verify_files=True,
                                              require_loaded_closure=False)
    _check_wall_deadline(started)
    output_root = Path(prebinding['output_root'])
    output_existing_bytes = _existing_regular_bytes(output_root, MAX_OUTPUT_BYTES)
    cache_bytes_before = ndarray_tree_bytes(CACHE)
    prospective_resource_budget(live_bytes=0, requested_bytes=0,
                                output_bytes=output_existing_bytes,
                                cache_bytes=cache_bytes_before)
    pre_import_sources = _source_freeze_before_import()
    file_pins = _verify_frozen_files()
    row_ids = expected_actual_row_ids()

    # Import the sealed bench closure even when a frozen data pin has failed, so
    # the terminal manifest still accounts all sixteen independent procedural
    # inputs and the launcher can verify the same post-import closure.
    r5 = importlib.import_module('bench.quality_bottle_pose_ablation')
    r6 = importlib.import_module('bench.quality_bottle_patch_pose_calibration')
    audit = importlib.import_module('bench.quality_bottle_identity_audit')
    cv2_module = importlib.import_module('bench.vision').cv2
    glb = importlib.import_module('bench.glb_model')
    _check_wall_deadline(started)
    blur_audit = _opencv_gaussian_path_audit(cv2_module)
    actual_preflight: dict = dict(state='not_run')
    view_records: dict = {}
    capture_binding = None
    mesh_binding = None
    outcomes: dict = {}

    if file_pins['state'] != 'matched':
        outcomes = {condition_id: dict(state='terminal_unavailable',
            reason='frozen_input_hash_mismatch') for condition_id in row_ids}
    elif blur_audit['state'] != 'passed':
        outcomes = {condition_id: dict(state='terminal_unavailable',
            reason='opencv_sigma2_path_failed_frozen_17x17_parity') for condition_id in row_ids}
        actual_preflight = dict(state='not_run', reason=blur_audit['reason'])
    else:
        plans = r5._expected_condition_plan()
        actual_ids = [row['condition_id'] for row in plans]
        if actual_ids != row_ids:
            raise PrerequisiteError('R5 actual row plan differs from the frozen twelve-row order')
        actual_preflight = r6.preflight_actual_cache()
        if (actual_preflight.get('matching_started') is not False or
                actual_preflight.get('fitting_started') is not False):
            raise PrerequisiteError('R6 closed-input preflight crossed the no-match/no-fit boundary')
        bundle = r6._load_r6_bundle()
        selections = r6._freeze_and_bind_source_selections(bundle, plans)
        banks = r6._load_fixed_banks(bundle)
        mesh = glb.read_glb(RANCH_MESH, 'ranch')
        view_records, outcomes = _actual_source_manifest(r5, r6, audit, bundle, plans,
                                                          selections, banks, cv2_module, mesh)
        capture_binding = dict(
            center_object_m=np.asarray(bundle['capture']['object_bbox_center_m']).tolist(),
            diagonal_m=float(bundle['capture']['object_bbox_diagonal_m']),
            signed_yaw_axis=np.asarray(bundle['capture']['object_axis_longest_svd']).tolist(),
            center_field='object_bbox_center_m', diagonal_field='object_bbox_diagonal_m',
            axis_field='object_axis_longest_svd')
        mesh_binding = dict(path=str(RANCH_MESH), sha256=R1_MESH_SHA256,
            vertices=array_record(mesh.positions), triangles=array_record(mesh.triangles),
            uv=array_record(mesh.uv), triangle_winding_preserved=True)
        # Release the loaded cached arrays before the procedural source banks
        # are allocated.  The control builder accounts its own shared peak.
        del bundle, selections, banks, mesh, plans
        import gc
        gc.collect()
        del r5, r6, glb

    rows = _account_terminal_rows(row_ids, outcomes,
        'fixed source support or independent witness preflight unavailable')
    if blur_audit['state'] == 'passed':
        controls = build_procedural_control_manifest(cv2_module=cv2_module,
            audit_module=audit, wall_started=started)
    else:
        controls = build_procedural_control_manifest(wall_started=started)
    _check_wall_deadline(started)
    source_check = _source_freeze_revalidate(pre_import_sources)
    if source_check['state'] != 'matched':
        raise PrerequisiteError('A1 local source snapshot changed during preparation')
    prebinding_post = validate_prebinding_contract(prebinding_contract, verify_files=True,
                                                   require_loaded_closure=True)
    cache_bytes_after = ndarray_tree_bytes(CACHE)
    prospective_resource_budget(live_bytes=0, requested_bytes=0,
        output_bytes=output_existing_bytes, cache_bytes=cache_bytes_after)
    control_rows = controls['controls']
    control_preflight_states = [row['a1_source_split_preflight']['state'] for row in control_rows]
    control_coverage = controls['preflight_coverage']
    controls_preflight_complete = (all(state == 'frozen_source_only_preflight'
                                       for state in control_preflight_states) and
                                   control_coverage['state'] == 'passed')
    source_rows_prepared = bool(rows) and all(row['state'] == 'prepared' for row in rows)
    preparation = dict(schema_version=1, stage='A1',
        status='prepared' if source_rows_prepared and controls_preflight_complete
        else 'terminal_unavailable',
        spec_sha256=SPEC_SHA256, frozen_input_pins=file_pins,
        root_scheduled=True, input_mode='closed R1/R3/R5/R6 inputs only',
        r6_loader_preflight=actual_preflight, gaussian_path_audit=blur_audit,
        capture_bindings=capture_binding, mesh_binding=mesh_binding,
        source_views=view_records, rows=rows, planned_rows=12, terminal_rows=len(rows),
        procedural_controls=controls, planned_controls=control_case_declarations(),
        control_count=len(control_rows), control_case_ids=[row['case_id'] for row in control_rows],
        control_split_state_counts={state: control_preflight_states.count(state)
                                    for state in sorted(set(control_preflight_states))},
        control_preflight_complete=controls_preflight_complete,
        control_preflight_coverage=control_coverage,
        matching_started=False, fitting_started=False, ncc_calls=0,
        production_render_calls=0, neural_calls=0,
        working_buffer_budget_bytes=MAX_WORKING_BYTES,
        private_output_budget_bytes=MAX_OUTPUT_BYTES,
        model_cache_budget_bytes=MAX_CACHE_BYTES,
        model_cache_bytes=int(cache_bytes_after),
        existing_output_root_bytes=int(output_existing_bytes),
        prebinding_contract=prebinding_post,
        source_freeze=dict(scope='partial 11-file local snapshot; sealed contract is full closure',
                           local_source_snapshot=source_check),
        elapsed_wall_seconds=_check_wall_deadline(started))
    payload_size = len(_bounded_json_payload(preparation,
                        cap=MAX_OUTPUT_BYTES - output_existing_bytes))
    prospective_resource_budget(live_bytes=0, requested_bytes=payload_size,
        output_bytes=output_existing_bytes + payload_size,
        cache_bytes=cache_bytes_after)
    return preparation


def write_frozen_a1_manifest(preparation: Mapping[str, object], output_path: Path, *,
                             root_scheduled: bool = False,
                             prebinding_contract: Mapping[str, object] | None = None) -> dict:
    if not root_scheduled:
        raise StageGateError('Writing the actual frozen A1 manifest requires explicit root scheduling')
    prebinding = validate_prebinding_contract(prebinding_contract, verify_files=True,
                                              require_loaded_closure=True)
    target = Path(output_path).resolve()
    allowed_root = (CACHE / 'diagnostics').resolve()
    if not target.is_relative_to(allowed_root):
        raise StageGateError('A1 manifest output must remain in the private model-quality diagnostics cache')
    output_root = Path(prebinding['output_root']).resolve()
    if not target.is_relative_to(output_root):
        raise StageGateError('A1 manifest must be written inside the sealed exclusive output root')
    prep_contract = preparation.get('prebinding_contract', {})
    if prep_contract.get('contract_sha256') != prebinding['contract_sha256']:
        raise StageGateError('Preparation and writer prebinding digests differ')
    existing_bytes = _existing_regular_bytes(output_root, MAX_OUTPUT_BYTES)
    payload = _bounded_json_payload(preparation, cap=MAX_OUTPUT_BYTES - existing_bytes)
    cache_bytes = ndarray_tree_bytes(CACHE)
    prospective_resource_budget(live_bytes=0, requested_bytes=len(payload),
        output_bytes=existing_bytes + len(payload), cache_bytes=cache_bytes)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(target), bytes=len(payload), sha256=sha256_bytes(payload),
                budget_bytes=MAX_OUTPUT_BYTES, aggregate_before_bytes=existing_bytes,
                aggregate_after_bytes=existing_bytes + len(payload),
                model_cache_bytes_before=int(cache_bytes), exclusive_create=True)


def run_stage_a2(*_args, **_kwargs) -> None:
    raise StageGateError('Stage A2 is closed in the A1 prerequisite; Sol review and parent scheduling are required')


def run_stage_b(*_args, **_kwargs) -> None:
    raise StageGateError('Stage B is closed in the A1 prerequisite; A2 review and parent scheduling are required')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('a1', 'a2', 'b'), required=True)
    parser.add_argument('--root-scheduled', action='store_true')
    parser.add_argument('--prebinding-contract', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.stage == 'a2':
        run_stage_a2()
    if args.stage == 'b':
        run_stage_b()
    contract = None
    if args.prebinding_contract is not None:
        contract = json.loads(args.prebinding_contract.read_text(encoding='utf-8'))
    prepared = prepare_actual_a1(root_scheduled=args.root_scheduled,
                                 prebinding_contract=contract)
    if args.output is not None:
        receipt = write_frozen_a1_manifest(prepared, args.output,
            root_scheduled=args.root_scheduled, prebinding_contract=contract)
        print(json.dumps(receipt, sort_keys=True))
    else:
        print(json.dumps(prepared, sort_keys=True, indent=2, allow_nan=False))
    return 0 if prepared.get('status') == 'prepared' else 2


if __name__ == '__main__':
    raise SystemExit(main())
