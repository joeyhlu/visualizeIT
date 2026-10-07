"""Pure source-first witness reservation for the frozen R8 capacity screen.

This module only freezes source-side evidence IDs and their raw dependency
union.  It does not inspect a query, match descriptors, fit a pose, or claim
that caller-supplied support bits are geometrically verified.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .quality_bottle_source_reservation import (
    MAX_WALL_SECONDS,
    MAX_WORKING_BYTES,
    SPEC_V2_SHA256,
    SourceCapacityLimitError,
    SourceReservationError,
    _check_budget,
)


MAX_WITNESSES = 32
MAX_WITNESSES_PER_CELL = 2
MIN_WITNESSES = 8
MAX_FIT_SAMPLES = 3000
MIN_FIT_SAMPLES = 24
MAX_BANK_ANCHORS = 64
BANK_GRID_SIZE = 16


def _as_source_array(value: object, name: str, *, ndim: int,
                     kinds: str, item_sizes: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SourceReservationError(f"{name} must be a preallocated NumPy array")
    array = value
    if (array.ndim != ndim or not array.flags.c_contiguous or
            array.dtype.kind not in kinds or array.dtype.itemsize not in item_sizes):
        raise SourceReservationError(
            f"{name} must be C-contiguous {ndim}D data with a supported numeric dtype")
    return array


def _namespace_json(image_namespace: object) -> str:
    if not isinstance(image_namespace, Mapping):
        raise SourceReservationError(
            "Image namespace must bind a source context ID and source RGB SHA256")
    context_id = image_namespace.get("context_id")
    rgb_hash = image_namespace.get("source_rgb_sha256")
    if (not isinstance(context_id, str) or not context_id.strip() or
            not isinstance(rgb_hash, str) or len(rgb_hash) != 64 or
            any(character not in "0123456789abcdefABCDEF" for character in rgb_hash)):
        raise SourceReservationError(
            "Image namespace requires nonempty context_id and 64-hex source_rgb_sha256")
    if set(image_namespace) != {"context_id", "source_rgb_sha256"}:
        raise SourceReservationError(
            "Image namespace must contain exactly context_id and source_rgb_sha256")
    normalized = {"context_id": context_id,
                  "source_rgb_sha256": rgb_hash.casefold()}
    try:
        value = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SourceReservationError("Image namespace must be finite canonical JSON data") from exc
    return value


def _ids_sha256(ids: np.ndarray, namespace_json: str) -> str:
    digest = hashlib.sha256()
    digest.update(namespace_json.encode("utf-8"))
    digest.update(b"\0source-ids-i64le\0")
    canonical = np.ascontiguousarray(ids, dtype="<i8")
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _dependency_sha256(dependencies: Sequence[int], namespace_json: str) -> str:
    values = np.ascontiguousarray(np.asarray(dependencies, dtype="<i8").reshape(-1))
    digest = hashlib.sha256()
    digest.update(namespace_json.encode("utf-8"))
    digest.update(b"\0raw-dependency-ids-i64le\0")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def reserve_witnesses(ids: np.ndarray, pixels: np.ndarray, std: np.ndarray,
                      r5_eligible: np.ndarray, source_supported: np.ndarray,
                      observed_mask: np.ndarray,
                      dependency_fn: Callable[[Sequence[float]], Sequence[int]], *,
                      image_namespace: object) -> dict:
    """Reserve the deterministic source-only witness set before any new bank.

    Inputs may be shuffled, but each row is inseparably bound by its original
    row-major source ID: ``id == floor(y)*W + floor(x)`` and the pixel must be
    the frozen center ``[x+.5, y+.5]``.  Caller support is explicitly treated
    as an assertion from an upstream source-only adapter; this function does
    not certify geometry, depth, or appearance support itself.
    """
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    namespace_json = _namespace_json(image_namespace)
    raw_ids = _as_source_array(ids, "ids", ndim=1, kinds="iu", item_sizes=(4, 8))
    raw_pixels = _as_source_array(pixels, "pixels", ndim=2, kinds="f", item_sizes=(4, 8))
    raw_std = _as_source_array(std, "std", ndim=1, kinds="f", item_sizes=(4, 8))
    raw_eligible = _as_source_array(r5_eligible, "r5_eligible", ndim=1,
                                    kinds="bu", item_sizes=(1,))
    raw_supported = _as_source_array(source_supported, "source_supported", ndim=1,
                                     kinds="bu", item_sizes=(1,))
    raw_mask = _as_source_array(observed_mask, "observed_mask", ndim=2,
                                kinds="bu", item_sizes=(1,))
    n = int(raw_ids.shape[0])
    if (raw_pixels.shape != (n, 2) or raw_std.shape != (n,) or
            raw_eligible.shape != (n,) or raw_supported.shape != (n,)):
        raise SourceReservationError("Witness source arrays must align row-for-row")
    height, width = map(int, raw_mask.shape)
    if height <= 0 or width <= 0 or n > height * width:
        raise SourceReservationError("Observed-mask dimensions or source population are invalid")
    if not callable(dependency_fn):
        raise SourceReservationError("The frozen raw-dependency function is required")
    # The selector makes aligned arrays, a stable sort index, cell buckets and
    # short-lived Python row lists.  Reserve a conservative 192 bytes per
    # source row plus the retained dependency bitmap and fixed interpreter
    # overhead before sorting or asking for any dependency set.
    input_bytes = int(raw_ids.nbytes + raw_pixels.nbytes + raw_std.nbytes +
                      raw_eligible.nbytes + raw_supported.nbytes + raw_mask.nbytes)
    # Pixel-center D(q) is at most 27x27=729 raw IDs under the frozen R7
    # 11x11 patch / 17-tap REFLECT_101 path.  Reserve retained tuples, the
    # selector's dependency set/union, and canonical ID/hash work up front.
    fixed_dependency_bytes = MAX_WITNESSES * 729 * 256 + 256 * 1024
    projected_bytes = (input_bytes + 192 * n + int(raw_mask.size) +
                       fixed_dependency_bytes + 1024 * 1024)
    _check_budget(projected_bytes, "source witness reservation")
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source witness reservation exceeded the 900-second wall cap")
    if raw_eligible.dtype.kind == "u" and np.any(raw_eligible > 1):
        raise SourceReservationError("R5 eligibility bits must contain only zero or one")
    if raw_supported.dtype.kind == "u" and np.any(raw_supported > 1):
        raise SourceReservationError("Source support bits must contain only zero or one")
    if raw_mask.dtype.kind == "u" and np.any(raw_mask > 1):
        raise SourceReservationError("Observed mask must contain only zero or one")
    if n and np.any(raw_ids < 0):
        raise SourceReservationError("Source IDs must be nonnegative original row-major indices")
    order = np.argsort(raw_ids, kind="stable")
    sorted_ids = raw_ids[order].astype(np.int64, copy=False)
    if len(sorted_ids) > 1 and np.any(sorted_ids[:-1] == sorted_ids[1:]):
        raise SourceReservationError("Original source IDs must be unique within one image namespace")
    sorted_pixels = raw_pixels[order].astype(np.float64, copy=False)
    sorted_std = raw_std[order].astype(np.float64, copy=False)
    sorted_eligible = raw_eligible[order].astype(bool, copy=False)
    sorted_supported = raw_supported[order].astype(bool, copy=False)
    if not np.isfinite(sorted_pixels).all():
        raise SourceReservationError("Original source pixel centers must be finite")
    if n and int(sorted_ids[-1]) >= height * width:
        raise SourceReservationError("An original source ID lies outside its observed mask")
    # Validate row-major pixel-center identity in bounded blocks.  This is a
    # schema check; IDs are never remapped to a nearby pixel.
    for start in range(0, n, 4096):
        if time.monotonic() > deadline:
            raise SourceCapacityLimitError("Source witness validation exceeded the wall cap")
        stop = min(n, start + 4096)
        block_ids = sorted_ids[start:stop]
        expected_x = block_ids % width + 0.5
        expected_y = block_ids // width + 0.5
        if (not np.array_equal(sorted_pixels[start:stop, 0], expected_x) or
                not np.array_equal(sorted_pixels[start:stop, 1], expected_y)):
            raise SourceReservationError(
                "Original source IDs do not match their frozen row-major pixel centers")

    qualified = sorted_eligible & sorted_supported
    # Importing the unchanged R7 selector here keeps this standalone core's
    # dependency contract explicit and avoids copying its cell/ranking rules.
    from . import quality_bottle_joint_prerequisite as r7

    def checked_dependencies(center_xy: Sequence[float]) -> tuple[int, ...]:
        if time.monotonic() > deadline:
            raise SourceCapacityLimitError("Source dependency reservation exceeded the wall cap")
        values = dependency_fn(center_xy)
        try:
            count = len(values)
        except TypeError:
            bounded = []
            for value in values:
                bounded.append(value)
                if len(bounded) > 729:
                    raise SourceReservationError(
                        "A pixel-center R7 high-pass set exceeds its frozen 729-ID bound")
            values = bounded
            count = len(values)
        if count > 729:
            raise SourceReservationError(
                "A pixel-center R7 high-pass set exceeds its frozen 729-ID bound")
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise SourceReservationError("Raw dependency IDs must be a one-dimensional integer sequence")
        if raw.size == 0:
            raise SourceReservationError("A high-pass witness must have at least one raw dependency")
        if raw.size > 729:
            raise SourceReservationError(
                "A pixel-center R7 high-pass set exceeds its frozen 729-ID bound")
        deps = np.asarray(raw, dtype=np.int64)
        if np.any(deps < 0) or np.any(deps >= height * width):
            raise SourceReservationError("A raw dependency lies outside its source image")
        return tuple(sorted(set(int(value) for value in deps)))

    # R7 itself only stores dependencies for selected witnesses.  Passing an
    # empty fixed-bank set is essential: the new bank has not been built yet.
    selected = r7.select_witness_ids(
        sorted_ids, sorted_pixels, sorted_std, qualified, raw_mask,
        checked_dependencies, bank_dependencies=(), max_witnesses=MAX_WITNESSES,
        per_cell=MAX_WITNESSES_PER_CELL)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source witness selection exceeded the 900-second wall cap")

    selected_rows = selected["selected"]
    selected_ids = np.asarray([item["source_id"] for item in selected_rows], dtype=np.int64)
    qualified_ids = sorted_ids[qualified].copy()
    dependency_union_ids = np.asarray(selected["dependency_union"], dtype=np.int64)
    if len(dependency_union_ids) and (
            np.any(dependency_union_ids[1:] <= dependency_union_ids[:-1]) or
            int(dependency_union_ids[0]) < 0 or int(dependency_union_ids[-1]) >= height * width):
        raise SourceReservationError("R7 selector returned a malformed dependency union")
    dependency_union_mask = np.zeros((height, width), dtype=bool)
    if len(dependency_union_ids):
        dependency_union_mask.reshape(-1)[dependency_union_ids] = True
    witness_records = []
    for item in selected_rows:
        dependencies = tuple(int(value) for value in item["raw_dependencies"])
        if tuple(sorted(set(dependencies))) != dependencies:
            raise SourceReservationError("R7 selector returned an unordered witness dependency set")
        witness_records.append({
            "source_id": int(item["source_id"]),
            "cell": int(item["cell"]),
            "std": float(item["source_std"]),
            "dependency_count": len(dependencies),
            "dependency_sha256": _dependency_sha256(dependencies, namespace_json),
        })
    reserved_count = int(selected["denominator"])
    state = "reserved" if selected["state"] == "frozen" else "unavailable"
    result = {
        "spec_v2_sha256": SPEC_V2_SHA256,
        "state": state,
        "reason": selected["reason"],
        "image_namespace": json.loads(namespace_json),
        "image_namespace_sha256": hashlib.sha256(namespace_json.encode("utf-8")).hexdigest(),
        "source_count": n,
        "r5_eligible_count": int(np.count_nonzero(sorted_eligible)),
        "source_supported_count": int(np.count_nonzero(sorted_supported)),
        "qualified_count": int(np.count_nonzero(qualified)),
        "nonfinite_std_qualified_count": int(np.count_nonzero(qualified & ~np.isfinite(sorted_std))),
        "ordered_ids": sorted_ids,
        "qualified_ids": qualified_ids,
        "ordered_ids_sha256": _ids_sha256(sorted_ids, namespace_json),
        "qualified_ids_sha256": _ids_sha256(qualified_ids, namespace_json),
        "qualified_source_identity": {
            "image_namespace": json.loads(namespace_json),
            "source_ids": qualified_ids,
            "source_ids_sha256": _ids_sha256(qualified_ids, namespace_json),
        },
        "selected_ids": selected_ids,
        "selected_ids_sha256": _ids_sha256(selected_ids, namespace_json),
        "selected_source_identity": {
            "image_namespace": json.loads(namespace_json),
            "source_ids": selected_ids,
            "source_ids_sha256": _ids_sha256(selected_ids, namespace_json),
        },
        "denominator": reserved_count,
        "reserved_count": reserved_count,
        "cell_counts": dict(selected["cell_counts"]),
        "candidate_rows_considered": int(selected["candidate_rows_considered"]),
        "rejected_witness_overlap_ids": np.asarray(
            selected["rejected_witness_overlap_ids"], dtype=np.int64),
        "witness_records": witness_records,
        "dependency_union_ids": dependency_union_ids,
        "dependency_union_count": int(len(dependency_union_ids)),
        "dependency_union_sha256": _ids_sha256(dependency_union_ids, namespace_json),
        "dependency_union_mask": dependency_union_mask,
        "support_provenance": "caller_asserted_only",
        "full_source_support_provenance_required": True,
        "capacity_only": True,
        "pose_quality_claim": False,
        "estimated_working_bytes": int(projected_bytes),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source witness result sealing exceeded the wall cap")
    return result


def _plain_array_sha256(value: np.ndarray) -> str:
    if not value.flags.c_contiguous:
        raise SourceReservationError("Source bank inputs must be C-contiguous original arrays")
    if value.size == 0:
        return hashlib.sha256(b"").hexdigest()
    digest = hashlib.sha256()
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _checked_raw_dependencies(dependency_fn: Callable[[Sequence[float]], Sequence[int]],
                              center_xy: Sequence[float], width: int, height: int,
                              deadline: float) -> tuple[int, ...]:
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Fit/bank dependency pass exceeded the wall cap")
    values = dependency_fn(center_xy)
    try:
        value_count = len(values)
    except TypeError:
        bounded = []
        for value in values:
            bounded.append(value)
            if len(bounded) > 729:
                raise SourceReservationError("A pixel-center R7 dependency set exceeds 729 IDs")
        values = bounded
        value_count = len(values)
    if value_count <= 0 or value_count > 729:
        raise SourceReservationError("A pixel-center R7 dependency set must contain 1..729 IDs")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise SourceReservationError("Raw dependency IDs must be a one-dimensional integer sequence")
    deps = raw.astype(np.int64, copy=False)
    if np.any(deps < 0) or np.any(deps >= width * height):
        raise SourceReservationError("A raw dependency lies outside its source image")
    return tuple(sorted(set(int(value) for value in deps)))


def _pool_dependency_row_digest(source_id: int, dependencies: Sequence[int]) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"R8-pool-row-v1\0")
    digest.update(np.asarray([source_id], dtype="<i8").tobytes())
    digest.update(np.asarray(dependencies, dtype="<i8").tobytes())
    return digest.digest()


def _dependency_union_hash(ids: np.ndarray, namespace_json: str) -> str:
    return _ids_sha256(ids, namespace_json)


def select_fit_and_bank(source_arrays: Mapping[str, object], ids: np.ndarray,
                        pixels: np.ndarray, points: np.ndarray, std: np.ndarray,
                        r5_eligible: np.ndarray, source_supported: np.ndarray,
                        witness: Mapping[str, object],
                        dependency_fn: Callable[[Sequence[float]], Sequence[int]],
                        bank_builder: Callable[..., Mapping[str, object]] | None = None,
                        *, image_namespace: object) -> dict:
    """Freeze the S_H-disjoint fit pool and build its one new source bank.

    The bank builder receives the original, unmodified full source RGB/mask
    and the entire filtered pool of original IDs/3D points.  The pool is not
    capped to the fit sample count.  This is source-selection bookkeeping only;
    a ready result does not establish identity discrimination or pose quality.
    """
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    if not isinstance(source_arrays, Mapping):
        raise SourceReservationError("Source arrays must be a mapping of whitelisted source inputs")
    try:
        raw_rgb = source_arrays["template_rgb"]
        raw_mask = source_arrays["template_mask"]
        raw_observed_mask = source_arrays["observed_crop_mask"]
    except KeyError as exc:
        raise SourceReservationError(
            "Source selection requires template_rgb, full template_mask, and observed_crop_mask") from exc
    rgb = _as_source_array(raw_rgb, "template_rgb", ndim=3, kinds="fiu", item_sizes=(1, 2, 4, 8))
    mask = _as_source_array(raw_mask, "template_mask", ndim=2, kinds="bu", item_sizes=(1,))
    observed_mask = _as_source_array(raw_observed_mask, "observed_crop_mask", ndim=2,
                                     kinds="bu", item_sizes=(1,))
    if (rgb.shape[:2] != mask.shape or observed_mask.shape != mask.shape or
            rgb.shape[2] != 3 or not rgb.size or not mask.size):
        raise SourceReservationError("Original template RGB and mask dimensions are invalid")
    namespace_json = _namespace_json(image_namespace)
    namespace = json.loads(namespace_json)

    raw_ids = _as_source_array(ids, "ids", ndim=1, kinds="iu", item_sizes=(4, 8))
    raw_pixels = _as_source_array(pixels, "pixels", ndim=2, kinds="f", item_sizes=(4, 8))
    raw_points = _as_source_array(points, "points", ndim=2, kinds="f", item_sizes=(4, 8))
    raw_std = _as_source_array(std, "std", ndim=1, kinds="f", item_sizes=(4, 8))
    raw_eligible = _as_source_array(r5_eligible, "r5_eligible", ndim=1,
                                    kinds="bu", item_sizes=(1,))
    raw_supported = _as_source_array(source_supported, "source_supported", ndim=1,
                                     kinds="bu", item_sizes=(1,))
    n = int(raw_ids.shape[0])
    if (raw_pixels.shape != (n, 2) or raw_points.shape != (n, 3) or
            raw_std.shape != (n,) or raw_eligible.shape != (n,) or
            raw_supported.shape != (n,)):
        raise SourceReservationError("Fit/bank source rows must align by original source ID")
    height, width = map(int, mask.shape)
    if n > height * width:
        raise SourceReservationError("Source population exceeds the original image ID space")
    input_bytes = int(rgb.nbytes + mask.nbytes + observed_mask.nbytes +
                      raw_ids.nbytes + raw_pixels.nbytes +
                      raw_points.nbytes + raw_std.nbytes + raw_eligible.nbytes + raw_supported.nbytes)
    # Reserve R7 sorting/bucket lists, three full-image workspace planes,
    # builder conversion/filter/descriptor work, and both source-dependency
    # unions before any copies, per-ID traversal or bank construction.
    pixel_count = int(height * width)
    dependency_workspace = (MAX_WITNESSES * 729 * 256 +
                            MAX_BANK_ANCHORS * 729 * 256 + 256 * 1024)
    image_workspace = 8 * pixel_count * 4 + 40 * pixel_count
    bank_anchor_workspace = MAX_BANK_ANCHORS * 11 * 11 * 3 * 4 + 1024 * 1024
    projected_bytes = (input_bytes + 192 * n + dependency_workspace +
                       image_workspace + bank_anchor_workspace)
    _check_budget(projected_bytes, "source fit-pool and bank selection")
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source fit-pool and bank selection exceeded the wall cap")

    if mask.dtype.kind == "u" and np.any(mask > 1):
        raise SourceReservationError("Original template mask must contain only zero or one")
    if observed_mask.dtype.kind == "u" and np.any(observed_mask > 1):
        raise SourceReservationError("Observed crop mask must contain only zero or one")
    if rgb.dtype.kind == "f" and not np.isfinite(rgb).all():
        raise SourceReservationError("Original template RGB contains nonfinite values")
    if raw_eligible.dtype.kind == "u" and np.any(raw_eligible > 1):
        raise SourceReservationError("R5 eligibility bits must contain only zero or one")
    if raw_supported.dtype.kind == "u" and np.any(raw_supported > 1):
        raise SourceReservationError("Source support bits must contain only zero or one")
    if not np.isfinite(raw_points).all():
        raise SourceReservationError("Original source object points must be finite")
    rgb_sha256 = _plain_array_sha256(rgb)
    if rgb_sha256 != namespace["source_rgb_sha256"]:
        raise SourceReservationError("Image namespace RGB hash does not match the unchanged source RGB")
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source RGB binding exceeded the wall cap")

    common_observed_mask = np.logical_and(mask.astype(bool, copy=False),
                                          observed_mask.astype(bool, copy=False))
    common_mask_sha256 = _plain_array_sha256(common_observed_mask)
    # Recompute the deterministic reservation from these exact arrays and
    # compare every identity-bearing field before consuming its S_H union.
    expected_witness = reserve_witnesses(
        raw_ids, raw_pixels, raw_std, raw_eligible, raw_supported, common_observed_mask,
        dependency_fn, image_namespace=namespace)
    if _namespace_json(witness.get("image_namespace")) != namespace_json:
        raise SourceReservationError("Witness source namespace differs from the bank source image")
    for key in ("spec_v2_sha256", "state", "reason", "image_namespace_sha256",
                "source_count", "r5_eligible_count", "source_supported_count", "qualified_count",
                "denominator", "reserved_count",
                "ordered_ids_sha256", "qualified_ids_sha256", "selected_ids_sha256",
                "dependency_union_count", "dependency_union_sha256", "cell_counts",
                "candidate_rows_considered", "witness_records", "support_provenance",
                "full_source_support_provenance_required", "capacity_only", "pose_quality_claim"):
        if witness.get(key) != expected_witness.get(key):
            raise SourceReservationError(f"Frozen witness identity field changed: {key}")
    for key in ("ordered_ids", "qualified_ids", "selected_ids",
                "rejected_witness_overlap_ids", "dependency_union_ids"):
        if not np.array_equal(np.asarray(witness.get(key)), expected_witness[key]):
            raise SourceReservationError(f"Frozen witness ID vector changed: {key}")
    witness_mask = np.asarray(witness.get("dependency_union_mask"), dtype=bool)
    if witness_mask.shape != mask.shape or not np.array_equal(
            witness_mask, expected_witness["dependency_union_mask"]):
        raise SourceReservationError("Frozen witness S_H mask differs from its source IDs")
    for key in ("qualified_source_identity", "selected_source_identity"):
        actual_identity = witness.get(key)
        expected_identity = expected_witness[key]
        if (not isinstance(actual_identity, Mapping) or
                _namespace_json(actual_identity.get("image_namespace")) != namespace_json or
                actual_identity.get("source_ids_sha256") != expected_identity["source_ids_sha256"] or
                not np.array_equal(np.asarray(actual_identity.get("source_ids")),
                                   expected_identity["source_ids"])):
            raise SourceReservationError(f"Frozen witness image-qualified identity changed: {key}")
    if witness.get("support_provenance") != "caller_asserted_only":
        raise SourceReservationError("Witness support bits lack the required upstream provenance warning")

    reasons = []
    if expected_witness["state"] != "reserved":
        reasons.append(expected_witness["reason"] or "fewer_than_8_frozen_source_witness_ids")

    order = np.argsort(raw_ids, kind="stable")
    sorted_ids = raw_ids[order].astype(np.int64, copy=False)
    sorted_pixels = raw_pixels[order].astype(np.float64, copy=False)
    sorted_points = raw_points[order]
    sorted_std = raw_std[order].astype(np.float64, copy=False)
    sorted_eligible = raw_eligible[order].astype(bool, copy=False)
    sorted_supported = raw_supported[order].astype(bool, copy=False)
    qualified = sorted_eligible & sorted_supported
    if (not np.isfinite(sorted_pixels).all() or
            (n and (np.any(sorted_ids[:-1] == sorted_ids[1:]) or
                    int(sorted_ids[0]) < 0 or int(sorted_ids[-1]) >= pixel_count))):
        raise SourceReservationError("Original source ID/pixel rows are invalid or duplicated")
    if n:
        for start in range(0, n, 4096):
            stop = min(n, start + 4096)
            block_ids = sorted_ids[start:stop]
            if (not np.array_equal(sorted_pixels[start:stop, 0], block_ids % width + 0.5) or
                    not np.array_equal(sorted_pixels[start:stop, 1], block_ids // width + 0.5)):
                raise SourceReservationError("Source ID/pixel-center binding changed before bank selection")

    witness_union_ids = expected_witness["dependency_union_ids"]
    witness_union_flat = expected_witness["dependency_union_mask"].reshape(-1)
    original_input_hashes = {
        "template_rgb": rgb_sha256,
        "template_mask": _plain_array_sha256(mask),
        "observed_crop_mask": _plain_array_sha256(observed_mask),
        "ids": _plain_array_sha256(raw_ids),
        "pixels": _plain_array_sha256(raw_pixels),
        "points": _plain_array_sha256(raw_points),
        "std": _plain_array_sha256(raw_std),
        "r5_eligible": _plain_array_sha256(raw_eligible),
        "source_supported": _plain_array_sha256(raw_supported),
    }
    row_dependency_digests = np.zeros((n, 32), dtype=np.uint8)
    fit_pool_mask = np.zeros(n, dtype=bool)
    fit_pool_rows = []
    pool_dependency_digest = hashlib.sha256()
    pool_dependency_digest.update(namespace_json.encode("utf-8"))
    pool_dependency_digest.update(b"\0R8-fit-pool-dependencies-v1\0")
    overlap_rejected = 0
    for row in np.flatnonzero(qualified):
        if time.monotonic() > deadline:
            raise SourceCapacityLimitError("Full source fit-pool scan exceeded the wall cap")
        deps = _checked_raw_dependencies(dependency_fn, sorted_pixels[row], width, height, deadline)
        row_hash = _pool_dependency_row_digest(int(sorted_ids[row]), deps)
        row_dependency_digests[row] = np.frombuffer(row_hash, dtype=np.uint8)
        if any(witness_union_flat[dependency_id] for dependency_id in deps):
            overlap_rejected += 1
            continue
        fit_pool_mask[row] = True
        fit_pool_rows.append(int(row))
        pool_dependency_digest.update(np.asarray([sorted_ids[row]], dtype="<i8").tobytes())
        pool_dependency_digest.update(row_hash)
    fit_pool_ids = sorted_ids[np.asarray(fit_pool_rows, dtype=np.int64)].copy()
    fit_pool_identity = {
        "image_namespace": namespace,
        "source_ids": fit_pool_ids,
        "source_ids_sha256": _ids_sha256(fit_pool_ids, namespace_json),
    }

    from . import quality_bottle_joint_prerequisite as r7

    def verified_fit_dependencies(center_xy: Sequence[float]) -> tuple[int, ...]:
        deps = _checked_raw_dependencies(dependency_fn, center_xy, width, height, deadline)
        x = int(math.floor(float(center_xy[0])))
        y = int(math.floor(float(center_xy[1])))
        source_id = y * width + x
        row = int(np.searchsorted(sorted_ids, source_id))
        if row >= n or int(sorted_ids[row]) != source_id or not fit_pool_mask[row]:
            raise SourceReservationError("R7 fit selector requested an ID outside the frozen fit pool")
        expected_row_hash = bytes(row_dependency_digests[row].tolist())
        if _pool_dependency_row_digest(source_id, deps) != expected_row_hash:
            raise SourceReservationError("Raw dependency set changed after fit-pool selection")
        return deps

    fit_record = r7.select_fit_ids(
        sorted_ids, sorted_pixels, fit_pool_mask, common_observed_mask, verified_fit_dependencies,
        witness_union_ids, max_samples=MAX_FIT_SAMPLES)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("R7 fit-ID selection exceeded the wall cap")
    fit_rows = np.asarray(fit_record["selected_rows"], dtype=np.int64)
    fit_ids = sorted_ids[fit_rows].copy()
    fit_dependency_union_mask = np.zeros(mask.shape, dtype=bool)
    fit_selected_dependency_digest = hashlib.sha256()
    fit_selected_dependency_digest.update(namespace_json.encode("utf-8"))
    fit_selected_dependency_digest.update(b"\0R8-selected-fit-dependencies-v1\0")
    for row in fit_rows:
        deps = verified_fit_dependencies(sorted_pixels[int(row)])
        fit_dependency_union_mask.reshape(-1)[np.asarray(deps, dtype=np.int64)] = True
        fit_selected_dependency_digest.update(
            np.asarray([sorted_ids[int(row)]], dtype="<i8").tobytes())
        fit_selected_dependency_digest.update(_pool_dependency_row_digest(int(sorted_ids[int(row)]), deps))
    fit_dependency_union_ids = np.flatnonzero(fit_dependency_union_mask.reshape(-1)).astype(np.int64)
    fit_union_intersection = int(np.count_nonzero(
        fit_dependency_union_mask & expected_witness["dependency_union_mask"]))
    if fit_union_intersection:
        raise SourceReservationError("Selected fit dependency union intersects frozen witness S_H")

    from . import quality_bottle_identity_audit as audit
    frozen_bank_builder = audit.build_template_patch_bank
    if bank_builder is None:
        bank_builder = frozen_bank_builder
    bank_builder_is_frozen = bank_builder is frozen_bank_builder
    if not callable(bank_builder):
        raise SourceReservationError("The unchanged source-only patch-bank builder is required")
    filtered_points = sorted_points[np.asarray(fit_pool_rows, dtype=np.int64)]
    filtered_ids_sha256 = _plain_array_sha256(fit_pool_ids)
    filtered_points_sha256 = _plain_array_sha256(filtered_points)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source bank construction exceeded the wall cap")
    bank_builder_calls = 1
    bank = bank_builder(rgb, mask, fit_pool_ids, filtered_points,
                        grid_size=BANK_GRID_SIZE, max_anchors=MAX_BANK_ANCHORS)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source bank builder exceeded the 900-second wall cap")
    post_builder_hashes = {
        "template_rgb": _plain_array_sha256(rgb),
        "template_mask": _plain_array_sha256(mask),
        "observed_crop_mask": _plain_array_sha256(observed_mask),
        "ids": _plain_array_sha256(raw_ids),
        "pixels": _plain_array_sha256(raw_pixels),
        "points": _plain_array_sha256(raw_points),
        "std": _plain_array_sha256(raw_std),
        "r5_eligible": _plain_array_sha256(raw_eligible),
        "source_supported": _plain_array_sha256(raw_supported),
    }
    if post_builder_hashes != original_input_hashes:
        raise SourceReservationError("Source bank builder mutated an original source-side input")
    if (_plain_array_sha256(fit_pool_ids) != filtered_ids_sha256 or
            _plain_array_sha256(filtered_points) != filtered_points_sha256):
        raise SourceReservationError("Source bank builder mutated its filtered full-pool arguments")
    if not isinstance(bank, Mapping):
        raise SourceReservationError("Source bank builder returned a malformed record")
    anchors = bank.get("anchors")
    serialized_anchors = bank.get("serialized_anchors")
    if not isinstance(anchors, (list, tuple)) or not isinstance(serialized_anchors, (list, tuple)):
        raise SourceReservationError("Source bank must expose ordered anchors and serialized anchors")
    if (len(anchors) != len(serialized_anchors) or len(anchors) > MAX_BANK_ANCHORS or
            int(bank.get("selected_anchor_count", len(anchors))) != len(anchors)):
        raise SourceReservationError("Source bank anchor accounting is inconsistent")
    anchor_ids = np.empty(len(anchors), dtype=np.int64)
    bank_dependency_union_mask = np.zeros(mask.shape, dtype=bool)
    bank_dependency_digest = hashlib.sha256()
    bank_dependency_digest.update(namespace_json.encode("utf-8"))
    bank_dependency_digest.update(b"\0R8-bank-anchor-dependencies-v1\0")
    bank_anchor_dependency_records = []
    for index, (anchor, public_anchor) in enumerate(zip(anchors, serialized_anchors)):
        if time.monotonic() > deadline:
            raise SourceCapacityLimitError("Source bank anchor validation exceeded the wall cap")
        if not isinstance(anchor, Mapping) or not isinstance(public_anchor, Mapping):
            raise SourceReservationError("Source bank contains a malformed anchor record")
        source_id = anchor.get("source_index")
        if isinstance(source_id, bool) or not isinstance(source_id, (int, np.integer)):
            raise SourceReservationError("Source bank anchor ID is not an original integer source ID")
        source_id = int(source_id)
        anchor_ids[index] = source_id
        row = int(np.searchsorted(sorted_ids, source_id))
        if row >= n or int(sorted_ids[row]) != source_id or not fit_pool_mask[row]:
            raise SourceReservationError("Source bank anchor is outside the full S_H-disjoint fit pool")
        expected_xy = sorted_pixels[row]
        anchor_xy = np.asarray(anchor.get("source_xy_crop"), dtype=np.float64)
        anchor_point = np.asarray(anchor.get("object_xyz_m"), dtype=np.float64)
        if (anchor_xy.shape != (2,) or not np.array_equal(anchor_xy, expected_xy) or
                anchor_point.shape != (3,) or
                not np.array_equal(anchor_point, np.asarray(sorted_points[row], dtype=np.float64))):
            raise SourceReservationError("Source bank anchor pixel or 3D point detached from its original ID")
        if (public_anchor.get("source_index") != source_id or
                public_anchor.get("source_xy_crop") != anchor.get("source_xy_crop") or
                public_anchor.get("object_xyz_m") != anchor.get("object_xyz_m")):
            raise SourceReservationError("Serialized source-bank anchor changed its original ID/pixel/point")
        deps = _checked_raw_dependencies(dependency_fn, expected_xy, width, height, deadline)
        if _pool_dependency_row_digest(source_id, deps) != bytes(row_dependency_digests[row].tolist()):
            raise SourceReservationError("Bank-anchor dependencies changed after fit-pool selection")
        if any(witness_union_flat[dependency_id] for dependency_id in deps):
            raise SourceReservationError("New source-bank dependency overlaps frozen witness S_H")
        bank_dependency_union_mask.reshape(-1)[np.asarray(deps, dtype=np.int64)] = True
        row_hash = _pool_dependency_row_digest(source_id, deps)
        bank_dependency_digest.update(np.asarray([source_id], dtype="<i8").tobytes())
        bank_dependency_digest.update(row_hash)
        bank_anchor_dependency_records.append({
            "source_id": source_id,
            "dependency_count": len(deps),
            "dependency_sha256": _dependency_sha256(deps, namespace_json),
        })
    if len(anchor_ids) > 1 and len(np.unique(anchor_ids)) != len(anchor_ids):
        raise SourceReservationError("Source bank contains duplicate original source IDs")
    # The bank builder is read-only by contract. Verify the byte identity of
    # every caller-owned input it received or that defines its source rows.
    bank_dependency_union_ids = np.flatnonzero(bank_dependency_union_mask.reshape(-1)).astype(np.int64)
    bank_intersection = int(np.count_nonzero(
        bank_dependency_union_mask & expected_witness["dependency_union_mask"]))
    if bank_intersection:
        raise SourceReservationError("New bank S_F dependency union intersects frozen witness S_H")
    bank_manifest_record = {
        "selector": {"grid_size": BANK_GRID_SIZE, "max_anchors": MAX_BANK_ANCHORS},
        "state": str(bank.get("state", "unknown")),
        "candidate_grid_count": int(bank.get("candidate_grid_count", 0)),
        "eligible_source_patch_count": int(bank.get("eligible_source_patch_count", 0)),
        "selected_anchor_count": int(len(anchor_ids)),
        "anchors": list(serialized_anchors),
    }
    bank_public_json = json.dumps(bank_manifest_record, sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=True,
                                  allow_nan=False)
    bank_manifest_sha256 = hashlib.sha256(
        (namespace_json + "\0" + bank_public_json).encode("utf-8")).hexdigest()
    if int(fit_record["count"]) < MIN_FIT_SAMPLES:
        reasons.append("fewer_than_24_s_h_disjoint_fit_sources")
    if len(anchor_ids) < 1:
        reasons.append("no_s_h_disjoint_structural_bank_anchor")
    if _plain_array_sha256(rgb) != rgb_sha256:
        reasons.append("source_rgb_changed_during_bank_build")
    fit_state = "frozen" if int(fit_record["count"]) >= MIN_FIT_SAMPLES else "unavailable"
    bank_state = str(bank.get("state", "unknown"))
    state = "source_pool_and_bank_ready" if not reasons else "unavailable"
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Source fit/bank result sealing exceeded the 900-second wall cap")
    return {
        "spec_v2_sha256": SPEC_V2_SHA256,
        "state": state,
        "fit_bank_state": state,
        "reasons": reasons,
        "image_namespace": namespace,
        "common_observed_mask_sha256": common_mask_sha256,
        "support_provenance": "caller_asserted_only",
        "full_source_support_provenance_required": True,
        "witness_selected_ids": expected_witness["selected_ids"].copy(),
        "witness_selected_ids_sha256": expected_witness["selected_ids_sha256"],
        "witness_selected_source_identity": expected_witness["selected_source_identity"],
        "witness_denominator": int(expected_witness["denominator"]),
        "witness_dependency_union_count": int(expected_witness["dependency_union_count"]),
        "witness_dependency_union_sha256": expected_witness["dependency_union_sha256"],
        "fit_pool_ids": fit_pool_ids,
        "fit_pool_count": int(len(fit_pool_ids)),
        "fit_pool_source_identity": fit_pool_identity,
        "fit_pool_ids_sha256": fit_pool_identity["source_ids_sha256"],
        "fit_pool_dependency_sha256": pool_dependency_digest.hexdigest(),
        "fit_pool_rejected_witness_overlap_count": int(overlap_rejected),
        "fit_ids": fit_ids,
        "fit_count": int(len(fit_ids)),
        "fit_state": fit_state,
        "fit_ids_sha256": _ids_sha256(fit_ids, namespace_json),
        "fit_cell_counts": dict(fit_record["cell_counts"]),
        "fit_dependency_sha256": fit_selected_dependency_digest.hexdigest(),
        "fit_dependency_union_ids": fit_dependency_union_ids,
        "fit_dependency_union_count": int(len(fit_dependency_union_ids)),
        "fit_dependency_union_sha256": _dependency_union_hash(fit_dependency_union_ids, namespace_json),
        "fit_union_intersection_with_witness_count": fit_union_intersection,
        "bank_builder_calls": int(bank_builder_calls),
        "bank_builder_is_frozen_audit_function": bool(bank_builder_is_frozen),
        "bank_builder_identity": (
            f"{getattr(bank_builder, '__module__', '')}."
            f"{getattr(bank_builder, '__qualname__', type(bank_builder).__name__)}"),
        "bank_state": bank_state,
        "bank_candidate_grid_count": int(bank.get("candidate_grid_count", 0)),
        "bank_eligible_source_patch_count": int(bank.get("eligible_source_patch_count", 0)),
        "bank_anchor_count": int(len(anchor_ids)),
        "bank_anchor_ids": anchor_ids,
        "bank_anchor_source_identity": {
            "image_namespace": namespace,
            "source_ids": anchor_ids,
            "source_ids_sha256": _ids_sha256(anchor_ids, namespace_json),
        },
        "bank_anchor_ids_sha256": _ids_sha256(anchor_ids, namespace_json),
        "bank_anchor_dependency_records": bank_anchor_dependency_records,
        "bank_anchor_dependency_sha256": bank_dependency_digest.hexdigest(),
        "bank_dependency_union_ids": bank_dependency_union_ids,
        "bank_dependency_union_count": int(len(bank_dependency_union_ids)),
        "bank_dependency_union_sha256": _dependency_union_hash(bank_dependency_union_ids, namespace_json),
        "bank_intersection_with_witness_count": bank_intersection,
        "bank_manifest_sha256": bank_manifest_sha256,
        "estimated_working_bytes": int(projected_bytes),
        "elapsed_seconds": float(time.monotonic() - started),
        "capacity_only": True,
        "capacity_screen_complete": False,
        "pose_quality_claim": False,
    }
