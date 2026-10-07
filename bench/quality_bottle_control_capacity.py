"""Source-only capacity screen for the sixteen frozen R7 control cases.

The generator builds each deterministic control privately.  This adapter
projects only the reviewed source arrays, fixed mesh and observed query mask
into the frozen R8 capacity helper, then releases the full generated case.
It never uses query RGB, evaluator truth, endpoint stubs, matching or fitting.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import quality_bottle_joint_prerequisite as generator
from . import quality_bottle_source_capacity as capacity


CONTROL_SCHEMA_VERSION = "r8-procedural-control-capacity-v1"
MAX_WORKING_BYTES = 128 * 1024**2
MAX_WALL_SECONDS = 900.0
MAX_RECEIPT_BYTES = 8 * 1024**2
HASH_CHUNK_BYTES = 1024**2
REPORT_RESERVE_BYTES = 2 * 1024**2
MAPPING_VERSION = "r8-r7-control-source-projection-v1"
CONTROL_SPEC_SHA256 = "889e6e4ab708fd552f9134f72ae14370c2e900389130fba80ff1f5465e2757b5"
SOURCE_SPEC_SHA256 = "32758dda6ad550a5edd94f5354b7e8401ff206a63b884e8c88ab37f71ee6d9cb"
SOURCE_ADAPTER_SPEC_SHA256 = "e6c7df9b9b3d61fcfb1b845f94006573a4dba46575bc58e6e23ae894625426e8"
ARRAY_MAP = (
    ("template_rgb", "source_rgb"),
    ("template_mask", "source_observed_mask"),
    ("template_depth_mm", "source_depth_mm"),
    ("source_indices", "source_ids"),
    ("source_pixels_xy", "source_pixels_xy"),
    ("source_points_object_m", "source_points_object_m"),
    ("observed_crop_mask", "source_observed_mask"),
    ("crop_k", "intrinsics"),
    ("crop_from_native", None),
    ("native_k", "intrinsics"),
    ("template_pose_m", "source_seed_pose_m"),
)
GATE_NAMES = (
    "witnesses_ge_8", "fit_ge_24", "bank_ge_1",
    "qh_possible_cells_ge_3", "qh_maximum_bbox_hull_ge_0_12",
)
CONTROL_VARIANTS = (
    "distinct_markers", "blank_constant", "repeated_2d_periodic",
    "coherent_wrong_180_packet", "query_occlusion", "mask_leakage_surrogate",
    "global_affine", "appearance_mismatch",
)
SEEDS = (0, 180)
CANONICAL_CASES = tuple(
    (f"{variant}-seed-{seed:03d}", variant, float(seed))
    for variant in CONTROL_VARIANTS for seed in SEEDS
)
PINNED_CORE_SPECS: dict[str, str] = {
    ".cache/r8-control-capacity-spec-v1.md": CONTROL_SPEC_SHA256,
    ".cache/bottle-source-reservation-r8-spec-v2.md": SOURCE_SPEC_SHA256,
    "docs/r8-source-adapter-spec.md": SOURCE_ADAPTER_SPEC_SHA256,
    "bench/quality_bottle_source_adapter.py": "e57eb564573c76ce9db724814b211e9f9350853e5776059669fec9e8f9238bc6",
    "bench/quality_bottle_joint_prerequisite.py": "cf74fdf73a2e6f36a1ae4bf783091ac23e4b2fd5364ce5b5e721304cb7e1684a",
    "bench/quality_bottle_source_capacity.py": "8196e0bc188971affa15aa2361034d456cf20aa5756f15ab3571678cd456eef1",
    "bench/quality_bottle_source_packet.py": "8dcaac8c79c1cc560bf0634a17d0c3cd46bbc44a52843a5d7917854286e6201f",
    "bench/quality_bottle_source_reservation.py": "679876b7bca979bca6ab61e96e3d9ff90de3d0fba2f261da3cadeba39ff79b39",
    "bench/quality_bottle_reservation_selection.py": "753d51cc4e9ed1ef1b40e5bda64d9f83bcbf46d24432f42ec8edc9d4e7d5b944",
    "bench/quality_bottle_pose_ablation.py": "54bdd2f4ca4fb82551492e5397ab42cea2e464c5dad1f83360f7b23d33926542",
    "bench/quality_bottle_identity_audit.py": "954d4b4e3a460187963894988cce53a85a077abaf07e085a0833131d14418f6a",
    "bench/quality_bottle_zero_view_probe.py": "6e6ef7c0fc7629028fd1cd6bd711f55a006397e74c312ded0464ed7f21f06b78",
    "bench/quality_bottle_identity_probe.py": "57dfd1d500c911d501976df4bb23b067e8eae14d691e906ac7d1138e75339a27",
    "bench/quality_assets.py": "7258340f9f7c0fd27d5676ae9f37d50f311a4fd094033cc674e3880ca3f676fc",
    "bench/quality_contract.py": "587493cfaae1684fe2cfbacdfac064f80f6f939db0f4ca1d1905bb470e157756",
    "bench/vision.py": "76c7e91e6fe73278b35f96cd933801d8cca299aec41d51276bfe95c0167f77da",
}
_FALSE_CLAIMS = {
    "full_28_case_screen_complete": False,
    "pose": False,
    "accuracy": False,
    "live_fps": False,
    "mobile_ready": False,
    "matching": False,
    "fitting": False,
}


class ControlCapacityError(ValueError):
    """Terminal control-screen failure with fixed sixteen-row accounting."""

    def __init__(self, message: str, receipt: Mapping[str, object]):
        super().__init__(message)
        self.receipt = dict(receipt)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _check_deadline(deadline: float, phase: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"R8 control capacity exceeded the shared 900-second deadline at {phase}")


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _verify_core_pins(deadline: float) -> dict[str, dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    checked: dict[str, dict[str, object]] = {}
    for relative, expected in PINNED_CORE_SPECS.items():
        _check_deadline(deadline, "source pin verification")
        path = root.joinpath(*Path(relative).parts)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Pinned control dependency is not a regular file: {relative}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValueError(f"Pinned control dependency changed before read: {relative}")
            while True:
                _check_deadline(deadline, "source pin hashing")
                block = stream.read(HASH_CHUNK_BYTES)
                if not block:
                    break
                size += len(block)
                digest.update(block)
                del block
            after = os.fstat(stream.fileno())
        sha = digest.hexdigest()
        if (opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns) or sha != expected:
            raise ValueError(f"Pinned control dependency differs from its reviewed SHA256: {relative}")
        checked[relative] = {"bytes": int(size), "sha256": sha}
    return checked


def _array_record(value: object, label: str) -> dict[str, object]:
    if (not isinstance(value, np.ndarray) or value.dtype.hasobject or
            not value.flags.c_contiguous or value.dtype.kind not in "biuf"):
        raise ValueError(f"{label} must be a contiguous numeric NumPy array")
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "nbytes": int(value.nbytes),
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
    }


def _compact_graph_bytes(value: object, deadline: float, label: str) -> int:
    """Count only adapter-owned JSON receipt graphs, never producer mappings or arrays."""
    stack = [value]
    seen: set[int] = set()
    total = 0
    visited = 0
    while stack:
        item = stack.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        visited += 1
        if visited % 1024 == 0:
            _check_deadline(deadline, f"{label} retained graph accounting")
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif item is None or isinstance(item, (str, int, float, bool)):
            pass
        else:
            raise ValueError(f"{label} contains a non-JSON object")
    return int(total)


def _detach_exception_graph(error: BaseException) -> None:
    """Remove traceback-held producer buffers before a compact terminal error escapes."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending and len(seen) < 32:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        cause = current.__cause__
        context = current.__context__
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def _unique_array_bytes(values: tuple[np.ndarray, ...]) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _validate_projection(source_arrays: Mapping[str, np.ndarray], positions: np.ndarray,
                         triangles: np.ndarray, query_mask: np.ndarray) -> None:
    if tuple(source_arrays) != tuple(key for key, _ in ARRAY_MAP):
        raise ValueError("Control source projection changed its frozen eleven-key order")
    rgb = source_arrays["template_rgb"]
    template_mask = source_arrays["template_mask"]
    observed_mask = source_arrays["observed_crop_mask"]
    depth = source_arrays["template_depth_mm"]
    ids = source_arrays["source_indices"]
    xy = source_arrays["source_pixels_xy"]
    points = source_arrays["source_points_object_m"]
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2] or
            template_mask.shape != depth.shape or observed_mask.shape != depth.shape or
            query_mask.shape != depth.shape or ids.ndim != 1 or
            xy.shape != (len(ids), 2) or points.shape != (len(ids), 3) or
            positions.ndim != 2 or positions.shape[1] != 3 or
            triangles.ndim != 2 or triangles.shape[1] != 3):
        raise ValueError("Control source arrays, masks, or fixed mesh have invalid shapes")
    if source_arrays["crop_from_native"].shape != (4, 4) or not np.array_equal(
            source_arrays["crop_from_native"], np.eye(4, dtype=np.float64)):
        raise ValueError("Control bridge must use the frozen identity crop transform")
    if (source_arrays["crop_k"] is not source_arrays["native_k"] or
            template_mask is not observed_mask):
        raise ValueError("Control projection must preserve the frozen intrinsics and source-mask aliases")


def _capacity_record(record: object, *, case_id: str, packet_sha256: str,
                     mesh_sha256: str, namespace: dict[str, str],
                     expected_input_resident_bytes: int,
                     deadline: float) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(record, Mapping):
        raise ValueError("Frozen capacity helper did not return a receipt mapping")
    if (record.get("schema_version") != capacity.CAPACITY_SCHEMA_VERSION or
            record.get("source_context_id") != case_id or
            record.get("packet_sha256") != packet_sha256 or
            record.get("mesh_sha256") != mesh_sha256 or
            record.get("query_observed_mask_origin") != "caller_supplied_observed" or
            record.get("capacity_only") is not True or
            record.get("full_28_case_screen_complete") is not False or
            record.get("matching_started") is not False or
            record.get("fitting_started") is not False or
            record.get("query_rgb_or_evaluator_truth_read") is not False):
        raise ValueError("Capacity helper receipt crossed its frozen source-only boundary")
    for name in ("counterpart_context_bindings_verified", "source_plan_bindings_verified"):
        if record.get(name) is not False:
            raise ValueError(f"Capacity helper {name} must remain false for private procedural controls")
    witness = record.get("witness_reservation")
    fit_bank = record.get("source_fit_bank")
    if (not isinstance(witness, Mapping) or not isinstance(fit_bank, Mapping) or
            witness.get("image_namespace") != namespace or fit_bank.get("image_namespace") != namespace):
        raise ValueError("Capacity helper witness or fit/bank namespace differs from this exact control source")
    if (record.get("fit_union_intersection_with_witness_count") != 0 or
            record.get("bank_intersection_with_witness_count") != 0 or
            fit_bank.get("fit_union_intersection_with_witness_count") != 0 or
            fit_bank.get("bank_intersection_with_witness_count") != 0):
        raise ValueError("Capacity helper fit or bank intersects held-out witnesses")

    witness_count = _nonnegative_int(record.get("witness_count"), "witness_count")
    fit_count = _nonnegative_int(record.get("fit_count"), "fit_count")
    bank_count = _nonnegative_int(record.get("bank_anchor_count"), "bank_anchor_count")
    qh_cells = _nonnegative_int(record.get("qh_possible_cell_count"), "qh_possible_cell_count")
    hull = record.get("qh_maximum_bbox_hull_fraction")
    if (isinstance(hull, bool) or not isinstance(hull, (int, float)) or
            not math.isfinite(float(hull)) or not 0.0 <= float(hull) <= 1.0):
        raise ValueError("Capacity helper omitted its finite QH hull bound")
    expected_gates = {
        "witnesses_ge_8": witness_count >= 8,
        "fit_ge_24": fit_count >= 24,
        "bank_ge_1": bank_count >= 1,
        "qh_possible_cells_ge_3": qh_cells >= 3,
        "qh_maximum_bbox_hull_ge_0_12": float(hull) >= .12,
    }
    gates = record.get("gates")
    if (not isinstance(gates, Mapping) or tuple(gates) != GATE_NAMES or
            any(type(gates.get(key)) is not bool or gates[key] is not value
                for key, value in expected_gates.items())):
        raise ValueError("Capacity helper gates differ from their measured structural counts")
    failed = [key for key, passed in expected_gates.items() if not passed]
    if (record.get("failed_conjuncts") != failed or
            record.get("capacity_available") is not (not failed) or
            record.get("state") != ("available" if not failed else "unavailable")):
        raise ValueError("Capacity helper availability disagrees with its five frozen gates")
    claims = record.get("claims")
    if (not isinstance(claims, Mapping) or set(claims) != {"fullscreen", "accuracy", "pose"} or
            any(claims.get(key) is not False for key in ("fullscreen", "accuracy", "pose"))):
        raise ValueError("Capacity helper made an impermissible outcome or pose claim")
    resource = record.get("resource")
    if (not isinstance(resource, Mapping) or resource.get("limit_bytes") != MAX_WORKING_BYTES or
            resource.get("wall_limit_seconds") != MAX_WALL_SECONDS):
        raise ValueError("Capacity helper omitted its frozen 128 MiB resource ledger")
    elapsed = resource.get("elapsed_seconds")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
            not math.isfinite(float(elapsed)) or not 0 <= float(elapsed) <= MAX_WALL_SECONDS):
        raise ValueError("Capacity helper omitted its bounded wall-time receipt")
    phase_peaks = resource.get("phase_peak_estimates_bytes")
    if not isinstance(phase_peaks, Mapping) or not phase_peaks:
        raise ValueError("Capacity helper omitted nonempty phase peak estimates")
    clean_peaks: dict[str, int] = {}
    for name, peak in phase_peaks.items():
        clean_peaks[str(name)] = _nonnegative_int(peak, f"capacity phase {name}")
        if clean_peaks[str(name)] > MAX_WORKING_BYTES:
            raise MemoryError(f"Capacity helper phase {name} exceeded 128 MiB")
    input_resident = _nonnegative_int(
        resource.get("input_resident_bytes"), "capacity input_resident_bytes")
    if (input_resident != expected_input_resident_bytes or
            max(clean_peaks.values(), default=0) < expected_input_resident_bytes):
        raise ValueError("Capacity helper resource ledger undercounts caller and unique input arrays")
    _check_deadline(deadline, "capacity receipt validation")
    compact = json.loads(_canonical_json(record).decode("utf-8"))
    if not isinstance(compact, dict):
        raise ValueError("Capacity helper receipt did not serialize as a JSON object")
    summary = {
        "witness_count": witness_count,
        "fit_count": fit_count,
        "bank_anchor_count": bank_count,
        "qh_possible_cell_count": qh_cells,
        "qh_maximum_bbox_hull_fraction": float(hull),
        "gates": dict(expected_gates),
        "failed_conjuncts": failed,
        "capacity_available": not failed,
        "phase_peak_estimates_bytes": clean_peaks,
    }
    return compact, summary


def _screen_one(case_id: str, variant: str, seed: float, *, external_resident: int,
                deadline: float, resource: dict[str, Any],
                render_ledger: list[dict[str, object]]) -> dict[str, object]:
    """Generate, whitelist, release the full case, then screen its source."""
    case: object = None
    estimator_inputs: object = None
    fixed_mesh: object = None
    source_arrays: dict[str, np.ndarray] = {}
    positions: object = None
    triangles: object = None
    query_mask: object = None
    capacity_record: object = None
    case_generated = False
    try:
        _check_deadline(deadline, f"before generation {case_id}")
        budget = generator._fixture_live_budget()
        if not isinstance(budget, Mapping):
            raise ValueError("Frozen generator omitted its pre-allocation resource budget")
        generator_peak = _nonnegative_int(budget.get("peak_bytes"), "generator peak_bytes")
        report_bytes = int(resource["completed_report_graph_bytes"])
        generation_peak = external_resident + report_bytes + REPORT_RESERVE_BYTES + generator_peak
        resource["generator_peak_bytes"] = max(int(resource["generator_peak_bytes"]), generator_peak)
        resource["generation_prospective_peak_bytes"] = max(
            int(resource["generation_prospective_peak_bytes"]), generation_peak)
        resource["phase_peak_estimates_bytes"][f"{case_id}:generation"] = generation_peak
        if generation_peak > MAX_WORKING_BYTES:
            raise MemoryError(f"Control {case_id} generator peak {generation_peak} exceeds 128 MiB")
        _check_deadline(deadline, f"generation start {case_id}")
        render_ledger.append({
            "case_id": case_id,
            "state": "generation_started",
            "procedural_generator_render_calls": "unknown_or_partial",
        })
        case = generator.generate_control_case(variant, seed)
        case_generated = True
        render_ledger[-1].update({
            "state": "generated",
            "procedural_generator_render_calls": 2,
        })
        _check_deadline(deadline, f"after generation {case_id}")
        if not isinstance(case, Mapping):
            raise ValueError("Frozen generator returned a non-mapping control case")
        if (case["case_id"] != case_id or case["variant"] != variant or
                float(case["source_seed_deg"]) != seed):
            raise ValueError("Frozen generator returned a case with the wrong selected identity")
        estimator_inputs = case["estimator_inputs"]
        if not isinstance(estimator_inputs, Mapping):
            raise ValueError("Generated control case omitted estimator inputs")
        for capacity_key, producer_key in ARRAY_MAP:
            if producer_key is None:
                source_arrays[capacity_key] = np.eye(4, dtype=np.float64)
            else:
                source_arrays[capacity_key] = estimator_inputs[producer_key]
        fixed_mesh = estimator_inputs["fixed_mesh"]
        if not isinstance(fixed_mesh, Mapping):
            raise ValueError("Generated control case omitted its frozen fixed mesh")
        positions = fixed_mesh["positions_m"]
        triangles = fixed_mesh["triangles"]
        query_mask = estimator_inputs["query_observed_mask"]
        _validate_projection(source_arrays, positions, triangles, query_mask)

        source_records = {
            key: _array_record(value, key) for key, value in source_arrays.items()
        }
        mesh_positions_record = _array_record(positions, "mesh positions")
        mesh_triangles_record = _array_record(triangles, "mesh triangles")
        query_record = _array_record(query_mask, "observed query mask")
        namespace = {"context_id": case_id,
                     "source_rgb_sha256": source_records["template_rgb"]["sha256"]}
        mesh_sha = hashlib.sha256(_canonical_json({
            "schema_version": "r8-control-native-mesh-v1",
            "positions_m": mesh_positions_record,
            "triangles": mesh_triangles_record,
        })).hexdigest()
        packet_sha = hashlib.sha256(_canonical_json({
            "mapping_version": MAPPING_VERSION,
            "control_spec_sha256": CONTROL_SPEC_SHA256,
            "source_spec_v2_sha256": SOURCE_SPEC_SHA256,
            "generator_spec_sha256": str(generator.SPEC_SHA256).lower(),
            "case_id": case_id,
            "variant": variant,
            "source_seed_deg": seed,
            "source_array_order": [key for key, _ in ARRAY_MAP],
            "source_array_records": source_records,
        })).hexdigest()
        if (not isinstance(positions, np.ndarray) or not isinstance(triangles, np.ndarray) or
                not isinstance(query_mask, np.ndarray)):
            raise ValueError("Generated mesh and observed mask must remain NumPy arrays")
        selected_bytes = _unique_array_bytes(
            tuple(source_arrays.values()) + (positions, triangles, query_mask))
        capacity_external = external_resident + report_bytes + REPORT_RESERVE_BYTES
        capacity_input_peak = capacity_external + selected_bytes
        resource["selected_input_bytes"] = max(int(resource["selected_input_bytes"]), selected_bytes)
        resource["capacity_external_resident_bytes"] = max(
            int(resource["capacity_external_resident_bytes"]), capacity_external)
        resource["capacity_prospective_input_bytes"] = max(
            int(resource["capacity_prospective_input_bytes"]), capacity_input_peak)
        resource["phase_peak_estimates_bytes"][f"{case_id}:capacity_input"] = capacity_input_peak
        if capacity_input_peak > MAX_WORKING_BYTES:
            raise MemoryError(f"Control {case_id} projected capacity inputs exceed 128 MiB")

        # Drop all producer-private containers before crossing the capacity boundary.
        case = None
        estimator_inputs = None
        fixed_mesh = None
        gc.collect()
        _check_deadline(deadline, f"before capacity {case_id}")
        resource["capacity_stage_calls"] = int(resource["capacity_stage_calls"]) + 1
        capacity_record = capacity.evaluate_source_capacity(
            source_arrays,
            source_context_id=case_id,
            packet_sha256=packet_sha,
            mesh_positions_m=positions,
            mesh_triangles=triangles,
            mesh_sha256=mesh_sha,
            query_observed_mask=query_mask,
            resident_bytes=capacity_external,
            deadline_monotonic=deadline,
        )
        compact_capacity, summary = _capacity_record(
            capacity_record, case_id=case_id, packet_sha256=packet_sha,
            mesh_sha256=mesh_sha, namespace=namespace,
            expected_input_resident_bytes=capacity_input_peak, deadline=deadline)
        helper_peak = max(summary["phase_peak_estimates_bytes"].values(), default=0)
        resource["capacity_helper_phase_peak_bytes"] = max(
            int(resource["capacity_helper_phase_peak_bytes"]), helper_peak)
        for name, peak in summary["phase_peak_estimates_bytes"].items():
            resource["phase_peak_estimates_bytes"][f"{case_id}:capacity:{name}"] = peak
        _check_deadline(deadline, f"after capacity {case_id}")
        mandatory = variant != "blank_constant"
        row = {
            "case_id": case_id,
            "variant": variant,
            "source_seed_deg": seed,
            "mandatory": mandatory,
            "blank_may_abstain": variant == "blank_constant",
            "state": "measured",
            "capacity_available": summary["capacity_available"],
            "witness_count": summary["witness_count"],
            "fit_count": summary["fit_count"],
            "bank_anchor_count": summary["bank_anchor_count"],
            "qh_possible_cell_count": summary["qh_possible_cell_count"],
            "qh_maximum_bbox_hull_fraction": summary["qh_maximum_bbox_hull_fraction"],
            "gates": summary["gates"],
            "failed_conjuncts": summary["failed_conjuncts"],
            "namespace": namespace,
            "source_array_records": source_records,
            "mesh_array_records": {"positions_m": mesh_positions_record,
                                   "triangles": mesh_triangles_record},
            "packet_sha256": packet_sha,
            "mesh_sha256": mesh_sha,
            "query_observed_mask": query_record,
            "procedural_generator_render_calls": 2,
            "capacity_receipt": compact_capacity,
            "resource": {
                "generation_prospective_peak_bytes": generation_peak,
                "capacity_prospective_input_bytes": capacity_input_peak,
                "capacity_helper_phase_peak_bytes": helper_peak,
            },
        }
        return row
    except BaseException:
        if case_generated:
            # Keep the known two procedural renders even when a later capacity
            # check fails; generator failures are marked by the outer caller.
            pass
        raise
    finally:
        if isinstance(source_arrays, dict):
            source_arrays.clear()
        case = None
        estimator_inputs = None
        fixed_mesh = None
        positions = None
        triangles = None
        query_mask = None
        capacity_record = None


def _selected_cases() -> tuple[tuple[str, str, float], ...]:
    if tuple(generator.CONTROL_IDS) != CONTROL_VARIANTS or tuple(generator.SOURCE_YAWS_DEG) != (0.0, 180.0):
        raise ValueError("Frozen generator control declaration constants changed")
    declarations = generator.control_case_declarations()
    selected = tuple((row["case_id"], row["variant"], float(row["source_seed_deg"]))
                     for row in declarations)
    if selected != CANONICAL_CASES:
        raise ValueError("Frozen generator declaration order differs from the fixed sixteen control IDs")
    return selected


def _terminal_rows() -> list[dict[str, object]]:
    return [{"case_id": case_id, "variant": variant, "source_seed_deg": seed,
             "state": "not_evaluated_due_terminal_failure"}
            for case_id, variant, seed in CANONICAL_CASES]


def _base_receipt() -> dict[str, object]:
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "state": "failed",
        "complete": False,
        "capacity_available": False,
        "advance_gate_passed": False,
        "full_28_case_screen_complete": False,
        "planned_procedural_generator_render_calls": 32,
        "procedural_generator_render_calls": 0,
        "capacity_stage_calls": 0,
        "capacity_stage_production_render_calls": 0,
        "capacity_stage_model_calls": 0,
        "capacity_stage_ncc_calls": 0,
        "capacity_stage_fitting_calls": 0,
        "capacity_stage_optimizer_calls": 0,
        "mandatory_case_ids": [case_id for case_id, variant, _ in CANONICAL_CASES
                               if variant != "blank_constant"],
        "mandatory_unavailable_case_ids": [],
        "blank_abstentions": [],
        "render_ledger": [],
        "source_records": [],
        "claims": dict(_FALSE_CLAIMS),
        "resource": {
            "limit_bytes": MAX_WORKING_BYTES,
            "wall_limit_seconds": MAX_WALL_SECONDS,
            "caller_resident_bytes": 0,
            "completed_report_graph_bytes": 0,
            "receipt_graph_bytes": 0,
            "adapter_reserve_bytes": REPORT_RESERVE_BYTES,
            "generator_peak_bytes": 0,
            "generation_prospective_peak_bytes": 0,
            "selected_input_bytes": 0,
            "capacity_external_resident_bytes": 0,
            "capacity_prospective_input_bytes": 0,
            "capacity_helper_phase_peak_bytes": 0,
            "capacity_stage_calls": 0,
            "receipt_json_bytes": 0,
            "phase_peak_estimates_bytes": {},
        },
        "rows": _terminal_rows(),
    }


def screen_control_rows(*, resident_bytes: int = 0,
                        deadline_monotonic: float | None = None) -> dict[str, object]:
    """Screen the exact sixteen frozen controls with source-only capacity gates."""
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    if deadline_monotonic is not None:
        if (isinstance(deadline_monotonic, bool) or
                not isinstance(deadline_monotonic, (int, float)) or
                not math.isfinite(float(deadline_monotonic))):
            receipt = _base_receipt()
            receipt["failure"] = {"stage": "entry", "reason": "deadline must be a finite monotonic timestamp"}
            raise ControlCapacityError("Invalid shared control deadline", receipt)
        deadline = min(deadline, float(deadline_monotonic))
    try:
        resident_bytes = _nonnegative_int(resident_bytes, "resident_bytes")
    except Exception as exc:
        receipt = _base_receipt()
        receipt["failure"] = {"stage": "entry", "reason": str(exc)[:512]}
        raise ControlCapacityError("Invalid control caller residency", receipt) from None

    receipt = _base_receipt()
    resource = receipt["resource"]
    assert isinstance(resource, dict)
    resource["caller_resident_bytes"] = resident_bytes
    source_records: list[dict[str, object]] = []
    completed_case_ids: set[str] = set()
    current_case: str | None = None
    stage = "source_pin_check"
    failure_render_state: str | None = None
    try:
        if resident_bytes > MAX_WORKING_BYTES:
            raise MemoryError("Caller retained bytes exceed the 128 MiB control-screen cap")
        _check_deadline(deadline, "entry")
        initial_pin_peak = resident_bytes + HASH_CHUNK_BYTES + REPORT_RESERVE_BYTES
        resource["phase_peak_estimates_bytes"]["initial_source_pin_check"] = initial_pin_peak
        if initial_pin_peak > MAX_WORKING_BYTES:
            raise MemoryError("Caller residency plus source-pin hashing reserve exceeds 128 MiB")
        input_pins = _verify_core_pins(deadline)
        stage = "case_selection"
        selected_cases = _selected_cases()
        if selected_cases != CANONICAL_CASES or len(selected_cases) != 16:
            raise ValueError("Control screen did not select all sixteen canonical cases before generation")
        stage = "generation_and_capacity"
        for case_id, variant, seed in selected_cases:
            current_case = case_id
            prior_bytes = (_compact_graph_bytes(source_records, deadline, "completed control records")
                           if source_records else 0)
            resource["completed_report_graph_bytes"] = prior_bytes
            external_resident = resident_bytes
            _check_deadline(deadline, f"case {case_id} preflight")
            ledger_before = len(receipt["render_ledger"])
            try:
                row = _screen_one(case_id, variant, seed,
                                  external_resident=external_resident,
                                  deadline=deadline, resource=resource,
                                  render_ledger=receipt["render_ledger"])
            except BaseException as exc:
                if len(receipt["render_ledger"]) == ledger_before:
                    failure_render_state = "not_started"
                elif receipt["render_ledger"][-1].get("state") == "generation_started":
                    receipt["render_ledger"][-1]["state"] = "generation_failed_or_incomplete"
                    failure_render_state = "unknown_or_partial"
                else:
                    failure_render_state = 2
                # Do not retain implementation frames and their producer-private
                # arrays through an exception chain in the compact terminal receipt.
                _detach_exception_graph(exc)
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from None
            completed_case_ids.add(case_id)
            receipt["render_ledger"][-1]["state"] = "generated_and_screened"
            source_records.append(row)
            receipt["capacity_stage_calls"] = int(resource["capacity_stage_calls"])
            resource["completed_report_graph_bytes"] = _compact_graph_bytes(
                source_records, deadline, "completed control records")
            receipt["procedural_generator_render_calls"] = 2 * len(completed_case_ids)
            _check_deadline(deadline, f"after case {case_id}")
        current_case = None
        stage = "final_pin_recheck"
        final_report_graph_bytes = _compact_graph_bytes(
            source_records, deadline, "final completed control records")
        final_pin_peak = (resident_bytes + final_report_graph_bytes + HASH_CHUNK_BYTES +
                          REPORT_RESERVE_BYTES)
        resource["phase_peak_estimates_bytes"]["final_source_pin_recheck"] = final_pin_peak
        if final_pin_peak > MAX_WORKING_BYTES:
            raise MemoryError("Final control reports plus pin-hashing reserve exceed 128 MiB")
        final_pins = _verify_core_pins(deadline)
        if input_pins != final_pins:
            raise ValueError("A pinned control dependency changed during the sixteen-case screen")
        stage = "receipt_seal"
        mandatory_rows = [row for row in source_records if row["mandatory"] is True]
        mandatory_unavailable = [str(row["case_id"]) for row in mandatory_rows
                                 if row["capacity_available"] is not True]
        blank_abstentions = [str(row["case_id"]) for row in source_records
                             if row["blank_may_abstain"] is True and row["capacity_available"] is not True]
        receipt.update({
            "state": "complete",
            "complete": True,
            "capacity_available": not mandatory_unavailable,
            "advance_gate_passed": not mandatory_unavailable,
            "source_records": source_records,
            "rows": source_records,
            "mandatory_unavailable_case_ids": mandatory_unavailable,
            "blank_abstentions": blank_abstentions,
            "mandatory_case_ids": [str(row["case_id"]) for row in mandatory_rows],
            "capacity_stage_calls": int(resource["capacity_stage_calls"]),
            "procedural_generator_render_calls": sum(
                item["procedural_generator_render_calls"]
                for item in receipt["render_ledger"]
                if isinstance(item.get("procedural_generator_render_calls"), int)
            ),
            "matching_started": False,
            "fitting_started": False,
            "elapsed_seconds": float(time.monotonic() - started),
            "input_core_pins": final_pins,
        })
        resource["receipt_graph_bytes"] = _compact_graph_bytes(
            receipt, deadline, "final control receipt")
        receipt_seal_peak = (resident_bytes + int(resource["receipt_graph_bytes"]) +
                             REPORT_RESERVE_BYTES + MAX_RECEIPT_BYTES)
        resource["phase_peak_estimates_bytes"]["receipt_seal"] = receipt_seal_peak
        if receipt_seal_peak > MAX_WORKING_BYTES:
            raise MemoryError("Control receipt plus bounded serialization reserve exceeds 128 MiB")
        receipt_json = _canonical_json(receipt)
        receipt_bytes = len(receipt_json)
        del receipt_json
        resource["receipt_json_bytes"] = receipt_bytes
        receipt_json = _canonical_json(receipt)
        receipt_bytes = len(receipt_json)
        del receipt_json
        resource["receipt_json_bytes"] = receipt_bytes
        if receipt_bytes > MAX_RECEIPT_BYTES:
            raise MemoryError("Control capacity receipt exceeds the frozen 8 MiB artifact cap")
        _check_deadline(deadline, "final receipt return")
        return receipt
    except BaseException as exc:
        # Source records remain compact and complete rows stay available for
        # diagnosis; authoritative rows are all unmeasured after any terminal
        # condition, even if it occurred after the sixteenth helper returned.
        _detach_exception_graph(exc)
        failure = {
            "stage": stage,
            "reason": f"{type(exc).__name__}: {exc}"[:1024],
        }
        if current_case is not None:
            failure["case_id"] = current_case
        if failure_render_state is not None:
            failure["current_case_generator_render_calls"] = failure_render_state
        receipt.update({
            "state": "failed",
            "complete": False,
            "capacity_available": False,
            "advance_gate_passed": False,
            "rows": _terminal_rows(),
            "source_records": source_records,
            "failure": failure,
            "elapsed_seconds": float(time.monotonic() - started),
            "procedural_generator_render_calls": sum(
                item["procedural_generator_render_calls"]
                for item in receipt["render_ledger"]
                if isinstance(item.get("procedural_generator_render_calls"), int)
            ),
            "capacity_stage_calls": int(resource["capacity_stage_calls"]),
            "claims": dict(_FALSE_CLAIMS),
            "matching_started": False,
            "fitting_started": False,
        })
        receipt["resource"]["completed_report_graph_bytes"] = _compact_graph_bytes(
            source_records, deadline, "failed control records")
        receipt["resource"]["receipt_json_bytes"] = 0
        raise ControlCapacityError("R8 procedural control capacity screen failed", receipt) from None


__all__ = [
    "ARRAY_MAP", "CANONICAL_CASES", "CONTROL_SCHEMA_VERSION", "MAX_RECEIPT_BYTES",
    "MAX_WALL_SECONDS", "MAX_WORKING_BYTES", "PINNED_CORE_SPECS",
    "ControlCapacityError", "screen_control_rows",
]
