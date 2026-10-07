"""Compose one source-only R8 capacity row from reviewed CPU helpers.

The caller binds the frozen plan and supplies only the packet loader's eleven
source arrays and the legitimate observed mask. This module neither opens a
packet nor reads query RGB, reference truth, matching results, or a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping

import numpy as np

from . import quality_bottle_joint_prerequisite as r7
from . import quality_bottle_pose_ablation as r5
from . import quality_bottle_reservation_selection as selection
from . import quality_bottle_source_packet as packet
from . import quality_bottle_source_reservation as geometry


CAPACITY_SCHEMA_VERSION = 1
MAX_CAPACITY_BYTES = 128 * 1024**2
MAX_CAPACITY_SECONDS = 900.0
SOURCE_KEYS = frozenset(packet.SOURCE_ARRAY_KEYS) - {"template_gray_rgb", "seed_pose_m"}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_FALSE_CLAIMS = {"fullscreen": False, "accuracy": False, "pose": False}


class SourceCapacityError(ValueError):
    """A source-only row has invalid provenance, shape, or resource use."""


def _deadline_check(deadline: float, phase: str) -> None:
    if time.monotonic() >= deadline:
        raise geometry.SourceCapacityLimitError(f"R8 source capacity exceeded its wall cap at {phase}")


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SourceCapacityError(f"{label} must be a SHA256 digest")
    return value.casefold()


def _raw_array_sha(value: np.ndarray) -> str:
    if not value.flags.c_contiguous:
        raise SourceCapacityError("Source evidence arrays must be C-contiguous")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _json_compact(value: object, label: str) -> object:
    """Detach a small helper receipt without retaining any source-ID arrays."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SourceCapacityError(f"{label} is not compact JSON evidence") from exc


def _int_field(record: Mapping, key: str) -> int:
    value = record.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise SourceCapacityError(f"Reviewed helper omitted nonnegative integer {key}")
    return int(value)


def _budget(phase: str, peak: int, peaks: dict[str, int]) -> None:
    peaks[phase] = int(peak)
    if peak > MAX_CAPACITY_BYTES:
        raise geometry.SourceCapacityLimitError(
            f"R8 {phase} requires {peak} bytes above the 128 MiB working cap")


def _unique_array_bytes(*values: np.ndarray) -> int:
    seen = set()
    total = 0
    for value in values:
        if not isinstance(value, np.ndarray) or value.dtype.hasobject or not value.flags.c_contiguous:
            raise SourceCapacityError("Source, mask, and mesh inputs must be contiguous numeric arrays")
        if id(value) not in seen:
            total += int(value.nbytes)
            seen.add(id(value))
    return total


def evaluate_source_capacity(
    source_arrays: Mapping[str, object],
    *,
    source_context_id: str,
    packet_sha256: str,
    mesh_positions_m: np.ndarray,
    mesh_triangles: np.ndarray,
    mesh_sha256: str,
    query_observed_mask: np.ndarray | None = None,
    resident_bytes: int = 0,
    deadline_monotonic: float | None = None,
) -> dict[str, object]:
    """Evaluate one source-first row without opening cached evidence.

    Caller support bits are never trusted: appearance is checked across every
    original raw D(q), and the full original ID population is ray-associated
    before witness or fit selection. Passing only establishes source capacity.
    """
    started = time.monotonic()
    deadline = started + MAX_CAPACITY_SECONDS
    if deadline_monotonic is not None:
        if (isinstance(deadline_monotonic, bool) or
                not isinstance(deadline_monotonic, (int, float)) or
                not math.isfinite(float(deadline_monotonic))):
            raise SourceCapacityError("Caller deadline must be a finite monotonic timestamp")
        deadline = min(deadline, float(deadline_monotonic))
    _deadline_check(deadline, "entry")
    if not isinstance(source_arrays, Mapping) or set(source_arrays) != SOURCE_KEYS:
        raise SourceCapacityError("Source arrays must contain exactly the eleven packet-loader projection keys")
    if not isinstance(source_context_id, str) or not source_context_id.strip():
        raise SourceCapacityError("A pinned source context ID is required")
    packet_hash = _sha(packet_sha256, "packet_sha256")
    mesh_hash = _sha(mesh_sha256, "mesh_sha256")
    if isinstance(resident_bytes, bool) or not isinstance(resident_bytes, int) or resident_bytes < 0:
        raise SourceCapacityError("resident_bytes must be a nonnegative integer")
    arrays = {key: source_arrays[key] for key in SOURCE_KEYS}
    if not all(isinstance(value, np.ndarray) for value in arrays.values()):
        raise SourceCapacityError("Every projected source field must be a NumPy array")
    vertices = mesh_positions_m
    triangles = mesh_triangles
    if not isinstance(vertices, np.ndarray) or not isinstance(triangles, np.ndarray):
        raise SourceCapacityError("Native mesh arrays must be NumPy arrays")
    rgb = arrays["template_rgb"]
    depth = arrays["template_depth_mm"]
    template_mask = arrays["template_mask"]
    observed_source = arrays["observed_crop_mask"]
    ids = arrays["source_indices"]
    xy = arrays["source_pixels_xy"]
    points = arrays["source_points_object_m"]
    current_observed = observed_source if query_observed_mask is None else query_observed_mask
    if not isinstance(current_observed, np.ndarray):
        raise SourceCapacityError("Current observed mask must be a source-side NumPy array")
    base = int(resident_bytes) + _unique_array_bytes(*arrays.values(), vertices, triangles, current_observed)
    peaks: dict[str, int] = {}
    _budget("input_residency", base, peaks)
    if (rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3 or
            depth.shape != rgb.shape[:2] or template_mask.shape != depth.shape or
            observed_source.shape != depth.shape or current_observed.shape != depth.shape or
            template_mask.dtype.kind not in "bu" or observed_source.dtype.kind not in "bu" or
            current_observed.dtype.kind not in "bu" or
            ids.ndim != 1 or ids.dtype.kind not in "iu" or
            xy.shape != (len(ids), 2) or points.shape != (len(ids), 3) or
            not len(ids) or vertices.ndim != 2 or vertices.shape[1] != 3 or
            triangles.ndim != 2 or triangles.shape[1] != 3):
        raise SourceCapacityError("Source image, original rows, observed mask, or mesh shape is invalid")
    height, width = depth.shape
    validation_peak = (base + 8 * len(ids) + 4 * rgb.size +
                       vertices.size + triangles.size + 2 * height * width)
    _budget("input_validation", validation_peak, peaks)
    if (template_mask.dtype.kind == "u" and np.any(template_mask > 1) or
            observed_source.dtype.kind == "u" and np.any(observed_source > 1) or
            current_observed.dtype.kind == "u" and np.any(current_observed > 1)):
        raise SourceCapacityError("Observed and template masks must contain only zero or one")
    if len(ids) > 1 and np.any(ids[:-1] >= ids[1:]):
        raise SourceCapacityError("Original source IDs must be increasing and unique")
    if not np.isfinite(vertices).all() or not np.isfinite(points).all():
        raise SourceCapacityError("Native mesh and source points must be finite")
    model_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not math.isfinite(model_diagonal) or model_diagonal <= 0:
        raise SourceCapacityError("Native mesh diagonal must be positive")
    _deadline_check(deadline, "before R5 eligibility")

    # R5 and R7 use exactly the same common source mask. Freeze and compare
    # both eligibility bitmaps before any geometry or witness choice.
    highpass_peak = base + 80 * height * width + 96 * len(ids) + 2 * 1024**2
    _budget("source_highpass", highpass_peak, peaks)
    common_mask = np.logical_and(template_mask, observed_source)
    r5_record = r5.freeze_source_selection(arrays)
    r5_bits = np.asarray(r5_record["eligible"])
    if r5_bits.dtype != np.bool_ or r5_bits.shape != (len(ids),):
        raise SourceCapacityError("R5 eligibility did not cover every original source ID")
    r5_count = _int_field(r5_record, "eligible_count")
    if r5_count != int(r5_bits.sum()):
        raise SourceCapacityError("R5 eligibility count differs from its bitmap")
    del r5_record
    r7_bits, std, eroded = r7._highpass_statistics(rgb, common_mask, xy, r5.cv2, r5.audit)
    if (r7_bits.shape != r5_bits.shape or r7_bits.dtype != np.bool_ or
            std.shape != (len(ids),) or eroded.shape != common_mask.shape or
            not np.array_equal(r5_bits, r7_bits)):
        raise SourceCapacityError("R7 source STD/eligibility differs from frozen R5")
    del r7_bits, eroded
    _deadline_check(deadline, "after R5/R7 eligibility")

    # Appearance's Python ID lists and prefixes must not remain live beside
    # the all-source native ray kernel. Keep only counts and the support bits.
    prefix_bytes = 4 * (height + 1) * (width + 1) * 8
    appearance_peak = base + common_mask.nbytes + r5_bits.nbytes + std.nbytes
    appearance_peak += prefix_bytes + 4 * height * width + 256 * len(ids) + 2 * 1024**2
    _budget("all_id_appearance", appearance_peak, peaks)
    appearance_record, appearance_bits = r7._source_appearance_support_for_ids(
        ids, xy, common_mask, depth,
        resident_bytes=base + common_mask.nbytes + r5_bits.nbytes + std.nbytes)
    if (appearance_bits.shape != (len(ids),) or appearance_bits.dtype != np.bool_ or
            _int_field(appearance_record, "source_count") != len(ids) or
            _int_field(appearance_record, "supported_count") != int(appearance_bits.sum())):
        raise SourceCapacityError("R7 appearance support did not account for all original IDs")
    appearance_reasons = {name: len(values)
                          for name, values in appearance_record["unsupported_reason_ids"].items()}
    appearance_count = int(appearance_bits.sum())
    appearance_sha = _raw_array_sha(appearance_bits)
    del appearance_record
    _deadline_check(deadline, "before all-ID native association")

    # The reviewed geometry helper counts its own ID/depth/point/mesh arrays.
    # Pass only other live residency, while separately reserving its 43N
    # outputs, topology table and streamed ray scratch in this row ledger.
    native_counted = ids.nbytes + xy.nbytes + depth.nbytes + points.nbytes + vertices.nbytes + triangles.nbytes
    native_external = base - native_counted + common_mask.nbytes + r5_bits.nbytes + std.nbytes + appearance_bits.nbytes
    native_peak = base + common_mask.nbytes + r5_bits.nbytes + std.nbytes + appearance_bits.nbytes
    native_peak += 43 * len(ids) + 3 * len(triangles) * 256 + 4 * 1024**2
    _budget("all_id_native", native_peak, peaks)
    native = geometry.native_source_association_all(
        ids, xy, depth, points,
        crop_k=arrays["crop_k"], crop_from_native=arrays["crop_from_native"],
        native_k=arrays["native_k"], template_pose_m=arrays["template_pose_m"],
        mesh_positions_m=vertices, mesh_triangles=triangles,
        model_diagonal_m=model_diagonal, source_context_id=source_context_id,
        packet_sha256=packet_hash, mesh_sha256=mesh_hash,
        resident_bytes=max(0, native_external), deadline_monotonic=deadline)
    if (not isinstance(native, Mapping) or
            not isinstance(native.get("source_ids"), np.ndarray) or
            not np.array_equal(native["source_ids"], ids) or
            not isinstance(native.get("support_bits"), np.ndarray) or
            native["support_bits"].shape != (len(ids),) or
            native["support_bits"].dtype != np.bool_ or
            _int_field(native, "source_count") != len(ids) or
            _int_field(native, "associated_count") != int(native["support_bits"].sum()) or
            native.get("raycast_kernel_is_frozen_r7") is not True or
            native.get("packet_sha256") != packet_hash or
            native.get("mesh_sha256") != mesh_hash):
        raise SourceCapacityError("All-original native association changed IDs, kernel, or pins")
    native_resource = native.get("resource")
    if not isinstance(native_resource, Mapping):
        raise SourceCapacityError("All-original native association omitted its resource ledger")
    native_actual_peak = _int_field(native_resource, "estimated_peak_bytes")
    if native_resource.get("state") != "within_budget":
        raise SourceCapacityError("All-original native association did not certify its resource cap")
    _budget("all_id_native", max(native_peak, native_actual_peak), peaks)
    native_bits = native["support_bits"]
    native_count = int(native_bits.sum())
    native_record = native["record"]
    if not isinstance(native_record, Mapping):
        raise SourceCapacityError("Native association omitted compact provenance")
    association_sha = _sha(native_record["geometry"]["association_sha256"], "association_sha256")
    topology_state = str(native_record["geometry"]["native_topology_state"])
    native_reason_counts = dict(native_record["reason_counts"])
    native_support_sha = _raw_array_sha(native_bits)
    native_compact = _json_compact({
        "schema_version": native_record["schema_version"],
        "source_context_id": native_record["source_context_id"],
        "packet_sha256": native_record["packet_sha256"],
        "mesh_sha256": native_record["mesh_sha256"],
        "source_count": native_record["source_count"],
        "associated_count": native_record["associated_count"],
        "unsupported_count": native_record["unsupported_count"],
        "reason_counts": native_record["reason_counts"],
        "epsilon_m": native_record["epsilon_m"],
        "geometry": native_record["geometry"],
        "raycast_kernel_identity": native_record["raycast_kernel_identity"],
        "raycast_kernel_is_frozen_r7": native_record["raycast_kernel_is_frozen_r7"],
        "ray": native_record["ray"],
        "resource": native_record["resource"],
        "topology": native_record["topology"],
    }, "native association")
    del native
    source_supported = np.logical_and(appearance_bits, native_bits)
    source_count = int(source_supported.sum())
    source_support_sha = _raw_array_sha(source_supported)
    _deadline_check(deadline, "before source-first reservation")

    namespace = {"context_id": source_context_id, "source_rgb_sha256": r5.audit.array_sha256(rgb)}
    def dependencies(center_xy):
        _deadline_check(deadline, "raw dependency enumeration")
        return r7.highpass_patch_raw_dependencies(center_xy, width, height)

    # Selection helpers include their own arrays/workspace; reserve all
    # additional source, mesh, query-mask, and support vectors here.
    selection_extra = 32 * 729 * 256 + 64 * 729 * 256 + 192 * len(ids)
    selection_extra += 72 * height * width + 2 * 1024**2
    selection_peak = base + common_mask.nbytes + r5_bits.nbytes + std.nbytes
    selection_peak += appearance_bits.nbytes + native_bits.nbytes + source_supported.nbytes + selection_extra
    _budget("witness_fit_bank", selection_peak, peaks)
    witness = selection.reserve_witnesses(
        ids, xy, std, r5_bits, source_supported, common_mask,
        dependencies, image_namespace=namespace)
    witness_count = _int_field(witness, "denominator")
    witness_sha = _sha(witness["selected_ids_sha256"], "witness IDs")
    if witness_count != len(witness["selected_ids"]):
        raise SourceCapacityError("Frozen witness denominator differs from selected original IDs")
    if witness.get("image_namespace") != namespace:
        raise SourceCapacityError("Frozen witness namespace differs from this source image")
    witness_compact = _json_compact({
        "spec_v2_sha256": witness["spec_v2_sha256"],
        "image_namespace": witness["image_namespace"],
        "image_namespace_sha256": witness["image_namespace_sha256"],
        "state": witness["state"],
        "reason": witness["reason"],
        "source_count": witness["source_count"],
        "r5_eligible_count": witness["r5_eligible_count"],
        "source_supported_count": witness["source_supported_count"],
        "qualified_count": witness["qualified_count"],
        "nonfinite_std_qualified_count": witness["nonfinite_std_qualified_count"],
        "ordered_ids_sha256": witness["ordered_ids_sha256"],
        "qualified_ids_sha256": witness["qualified_ids_sha256"],
        "selected_ids_sha256": witness["selected_ids_sha256"],
        "denominator": witness["denominator"],
        "cell_counts": witness["cell_counts"],
        "candidate_rows_considered": witness["candidate_rows_considered"],
        "rejected_witness_overlap_count": len(witness["rejected_witness_overlap_ids"]),
        "dependency_union_count": witness["dependency_union_count"],
        "dependency_union_sha256": witness["dependency_union_sha256"],
        "support_provenance": witness["support_provenance"],
    }, "witness reservation")
    _deadline_check(deadline, "after witness reservation")
    fit_bank = selection.select_fit_and_bank(
        arrays, ids, xy, points, std, r5_bits, source_supported,
        witness, dependencies, image_namespace=namespace)
    fit_pool_count = _int_field(fit_bank, "fit_pool_count")
    fit_count = _int_field(fit_bank, "fit_count")
    bank_count = _int_field(fit_bank, "bank_anchor_count")
    fit_pool_sha = _sha(fit_bank["fit_pool_ids_sha256"], "fit-pool IDs")
    fit_sha = _sha(fit_bank["fit_ids_sha256"], "fit IDs")
    bank_sha = _sha(fit_bank["bank_anchor_ids_sha256"], "bank anchor IDs")
    bank_manifest_sha = _sha(fit_bank["bank_manifest_sha256"], "new bank")
    if (fit_pool_count != len(fit_bank["fit_pool_ids"]) or
            fit_count != len(fit_bank["fit_ids"]) or
            bank_count != len(fit_bank["bank_anchor_ids"]) or
            fit_count > 3000 or fit_count > fit_pool_count or
            fit_bank.get("witness_denominator") != witness_count or
            fit_bank.get("witness_selected_ids_sha256") != witness_sha or
            fit_bank.get("bank_builder_calls") != 1 or
            fit_bank.get("bank_builder_is_frozen_audit_function") is not True or
            fit_bank.get("image_namespace") != namespace or
            fit_bank.get("fit_union_intersection_with_witness_count") != 0 or
            fit_bank.get("bank_intersection_with_witness_count") != 0):
        raise SourceCapacityError("Source-first fit/bank selection violated reviewed identities or intersections")
    bank_state = str(fit_bank["bank_state"])
    if bank_count and bank_state != "eligible":
        raise SourceCapacityError("A counted bank anchor lacks the reviewed eligible bank state")
    fit_pool_overlap_rejected = _int_field(fit_bank, "fit_pool_rejected_witness_overlap_count")
    fit_bank_compact = _json_compact({
        "spec_v2_sha256": fit_bank["spec_v2_sha256"],
        "image_namespace": fit_bank["image_namespace"],
        "state": fit_bank["state"],
        "reasons": fit_bank["reasons"],
        "common_observed_mask_sha256": fit_bank["common_observed_mask_sha256"],
        "witness_denominator": fit_bank["witness_denominator"],
        "witness_selected_ids_sha256": fit_bank["witness_selected_ids_sha256"],
        "witness_dependency_union_count": fit_bank["witness_dependency_union_count"],
        "witness_dependency_union_sha256": fit_bank["witness_dependency_union_sha256"],
        "fit_pool_count": fit_bank["fit_pool_count"],
        "fit_pool_ids_sha256": fit_bank["fit_pool_ids_sha256"],
        "fit_pool_dependency_sha256": fit_bank["fit_pool_dependency_sha256"],
        "fit_pool_rejected_witness_overlap_count": fit_bank["fit_pool_rejected_witness_overlap_count"],
        "fit_count": fit_bank["fit_count"],
        "fit_state": fit_bank["fit_state"],
        "fit_ids_sha256": fit_bank["fit_ids_sha256"],
        "fit_cell_counts": fit_bank["fit_cell_counts"],
        "fit_dependency_sha256": fit_bank["fit_dependency_sha256"],
        "fit_dependency_union_count": fit_bank["fit_dependency_union_count"],
        "fit_dependency_union_sha256": fit_bank["fit_dependency_union_sha256"],
        "fit_union_intersection_with_witness_count": fit_bank["fit_union_intersection_with_witness_count"],
        "bank_builder_calls": fit_bank["bank_builder_calls"],
        "bank_builder_is_frozen_audit_function": fit_bank["bank_builder_is_frozen_audit_function"],
        "bank_builder_identity": fit_bank["bank_builder_identity"],
        "bank_state": fit_bank["bank_state"],
        "bank_candidate_grid_count": fit_bank["bank_candidate_grid_count"],
        "bank_eligible_source_patch_count": fit_bank["bank_eligible_source_patch_count"],
        "bank_anchor_count": fit_bank["bank_anchor_count"],
        "bank_anchor_ids_sha256": fit_bank["bank_anchor_ids_sha256"],
        "bank_anchor_dependency_sha256": fit_bank["bank_anchor_dependency_sha256"],
        "bank_dependency_union_count": fit_bank["bank_dependency_union_count"],
        "bank_dependency_union_sha256": fit_bank["bank_dependency_union_sha256"],
        "bank_intersection_with_witness_count": fit_bank["bank_intersection_with_witness_count"],
        "bank_manifest_sha256": fit_bank["bank_manifest_sha256"],
        "support_provenance": fit_bank["support_provenance"],
    }, "source fit and bank")
    del witness, fit_bank
    _deadline_check(deadline, "before QH/QF mask-only bounds")

    qh_prefix = 4 * (height + 1) * (width + 1) * 8
    qh_peak = base + common_mask.nbytes + r5_bits.nbytes + std.nbytes
    qh_peak += appearance_bits.nbytes + native_bits.nbytes + source_supported.nbytes
    qh_peak += qh_prefix + 8 * height * width + 96 * height * width + 2 * 1024**2
    _budget("qh_qf_bounds", qh_peak, peaks)
    query_mask = current_observed.astype(bool, copy=False)
    qh, qf = r7.raw_pixel_partitions(width, height)
    eroded_query = r5.audit._eroded_mask(query_mask, 6)
    qh_centers = r7.allowed_target_center_bitmap(query_mask, qh, eroded_query)
    qf_centers = r7.allowed_target_center_bitmap(query_mask, qf, eroded_query)
    upper = r7.qh_verification_support_upper_bound(qh_centers, query_mask)
    qh_cells = _int_field(upper, "represented_cell_count")
    qh_count = _int_field(upper, "allowed_center_count")
    hull = upper.get("maximum_hull_fraction")
    if not isinstance(hull, (int, float)) or not math.isfinite(float(hull)) or not 0 <= float(hull) <= 1:
        raise SourceCapacityError("QH helper omitted a finite maximum hull fraction")
    if qh_count != int(qh_centers.sum()):
        raise SourceCapacityError("QH center count differs from frozen bitmap")
    qf_count = int(qf_centers.sum())
    qh_sha = _raw_array_sha(qh_centers)
    qf_sha = _raw_array_sha(qf_centers)
    _deadline_check(deadline, "after QH/QF bounds")

    gates = {
        "witnesses_ge_8": witness_count >= 8,
        "fit_ge_24": fit_count >= 24,
        "bank_ge_1": bank_count >= 1,
        "qh_possible_cells_ge_3": qh_cells >= 3,
        "qh_maximum_bbox_hull_ge_0_12": float(hull) >= 0.12,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "state": "available" if not failed else "unavailable",
        "capacity_available": not failed,
        "capacity_only": True,
        "full_28_case_screen_complete": False,
        "counterpart_context_bindings_verified": False,
        "source_plan_bindings_verified": False,
        "query_observed_mask_origin": (
            "source_observed_fallback" if query_observed_mask is None else "caller_supplied_observed"
        ),
        "source_context_id": source_context_id,
        "packet_sha256": packet_hash,
        "mesh_sha256": mesh_hash,
        "source_population_count": len(ids),
        "source_ids_sha256": _raw_array_sha(ids),
        "r5_eligible_count": r5_count,
        "r5_eligible_sha256": _raw_array_sha(r5_bits),
        "r7_eligible_count": r5_count,
        "r7_eligible_sha256": _raw_array_sha(r5_bits),
        "r5_r7_bit_exact": True,
        "appearance_supported_count": appearance_count,
        "appearance_support_sha256": appearance_sha,
        "appearance_unsupported_reason_counts": appearance_reasons,
        "native_supported_count": native_count,
        "native_support_sha256": native_support_sha,
        "native_reason_counts": native_reason_counts,
        "native_association_sha256": association_sha,
        "native_topology_state": topology_state,
        "native_raycast_kernel_frozen_r7": True,
        "native_association": native_compact,
        "source_supported_count": source_count,
        "source_support_sha256": source_support_sha,
        "witness_count": witness_count,
        "witness_ids_sha256": witness_sha,
        "witness_reservation": witness_compact,
        "fit_pool_count": fit_pool_count,
        "fit_pool_ids_sha256": fit_pool_sha,
        "fit_pool_rejected_witness_overlap_count": fit_pool_overlap_rejected,
        "fit_count": fit_count,
        "fit_ids_sha256": fit_sha,
        "bank_anchor_count": bank_count,
        "bank_ids_sha256": bank_sha,
        "new_bank_manifest_sha256": bank_manifest_sha,
        "bank_state": bank_state,
        "source_fit_bank": fit_bank_compact,
        "fit_union_intersection_with_witness_count": 0,
        "bank_intersection_with_witness_count": 0,
        "qh_possible_center_count": qh_count,
        "qh_possible_cell_count": qh_cells,
        "qh_maximum_bbox_hull_fraction": float(hull),
        "qh_centers_sha256": qh_sha,
        "qf_possible_center_count": qf_count,
        "qf_centers_sha256": qf_sha,
        "gates": gates,
        "failed_conjuncts": failed,
        "claims": dict(_FALSE_CLAIMS),
        "source_support_provenance": "full_original_id_appearance_and_native_association_before_selection",
        "query_rgb_or_evaluator_truth_read": False,
        "matching_started": False,
        "fitting_started": False,
        "resource": {
            "limit_bytes": MAX_CAPACITY_BYTES,
            "input_resident_bytes": base,
            "phase_peak_estimates_bytes": peaks,
            "wall_limit_seconds": MAX_CAPACITY_SECONDS,
            "elapsed_seconds": float(time.monotonic() - started),
        },
    }


__all__ = ["CAPACITY_SCHEMA_VERSION", "MAX_CAPACITY_BYTES", "MAX_CAPACITY_SECONDS",
           "SourceCapacityError", "evaluate_source_capacity"]
