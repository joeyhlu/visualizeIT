"""Synthetic CPU contract tests for the sixteen-control R8 capacity adapter.

All procedural producer calls are replaced with small in-memory mappings. The
source-capacity helper is mocked with its compact, real-schema receipt; this
suite never renders a control or opens a cache/model artifact.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import unittest
import weakref
from collections import UserDict
from typing import Any
from unittest import mock

import numpy as np

from . import quality_bottle_control_capacity as control
from . import quality_bottle_joint_prerequisite as generator


_VARIANTS = (
    "distinct_markers", "blank_constant", "repeated_2d_periodic",
    "coherent_wrong_180_packet", "query_occlusion", "mask_leakage_surrogate",
    "global_affine", "appearance_mismatch",
)
_SEEDS = (0, 180)
_ARRAY_KEYS = (
    "template_rgb", "template_mask", "template_depth_mm", "source_indices",
    "source_pixels_xy", "source_points_object_m", "observed_crop_mask", "crop_k",
    "crop_from_native", "native_k", "template_pose_m",
)
_GATES = (
    "witnesses_ge_8", "fit_ge_24", "bank_ge_1",
    "qh_possible_cells_ge_3", "qh_maximum_bbox_hull_ge_0_12",
)
_WORKING_CAP = 128 * 1024 * 1024
_FROZEN_GENERATOR_PEAK = 117_457_720
_MIB = 1024 * 1024
_FALSE_CLAIMS = {"fullscreen": False, "accuracy": False, "pose": False}
_SOURCE_SPEC_SHA256 = "32758DDA6AD550A5EDD94F5354B7E8401FF206A63B884E8C88AB37F71EE6D9CB"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _typed_record(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(memoryview(contiguous).cast("B"))
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "nbytes": int(contiguous.nbytes),
        "sha256": digest.hexdigest(),
    }


def _walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _graph_bytes(value: Any) -> int:
    stack = [value]
    seen: set[int] = set()
    total = 0
    while stack:
        item = stack.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return total


class _PoisonMapping(UserDict):
    """Weak-referenceable mapping that fails on forbidden producer fields."""

    def __init__(self, data=None, *, forbidden=(), access_log=None):
        super().__init__(data or {})
        self.forbidden = frozenset(forbidden)
        self.access_log = access_log if access_log is not None else []

    def _record(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden producer field accessed: {key}")
        self.access_log.append(key)

    def __getitem__(self, key):
        self._record(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._record(key)
        return super().get(key, default)

    def items(self):
        for key in self.data:
            self._record(key)
            yield key, self.data[key]

    def values(self):
        for key in self.data:
            self._record(key)
            yield self.data[key]


class ControlCapacityTests(unittest.TestCase):
    def setUp(self):
        self.generator_calls: list[tuple[str, float]] = []
        self.capacity_calls: list[dict[str, object]] = []
        self.budget_calls = 0
        self.mock_generator_peak = _FROZEN_GENERATOR_PEAK
        self.capacity_state = None
        self.capacity_mutator = None
        self.references: dict[str, dict[str, object]] = {}
        self.source_records: dict[str, dict[str, object]] = {}
        self.forbidden_access_logs: dict[str, list[str]] = {}
        self._deadline_seen: list[float | None] = []
        # Fail closed if a test forgets to install the synthetic producer or
        # capacity helper. Individual tests override these guards in a nested
        # patch stack with their small fixture callables.
        self._real_generator_guard = mock.patch.object(
            generator, "generate_control_case", new=self._reject_real_generator,
        )
        self._real_capacity_guard = mock.patch.object(
            control.capacity, "evaluate_source_capacity", new=self._reject_real_capacity,
        )
        self._real_generator_guard.start()
        self._real_capacity_guard.start()
        self.addCleanup(self._real_generator_guard.stop)
        self.addCleanup(self._real_capacity_guard.stop)

    def _reject_real_generator(self, *args, **kwargs):
        raise AssertionError("real procedural generator reached by synthetic test")

    def _reject_real_capacity(self, *args, **kwargs):
        raise AssertionError("real source-capacity helper reached by synthetic test")

    def _fixture_live_budget(self, *args, **kwargs):
        self.budget_calls += 1
        return {
            "peak_bytes": self.mock_generator_peak,
            "working_budget_bytes": _WORKING_CAP,
            "live_bytes": 0,
            "requested_bytes": self.mock_generator_peak,
        }

    def _make_case(self, variant: str, seed_deg: float) -> _PoisonMapping:
        case_id = f"{variant}-seed-{int(seed_deg):03d}"
        code = _VARIANTS.index(variant) * 2 + _SEEDS.index(int(seed_deg)) + 1
        source_rgb = np.full((3, 4, 3), code / 32.0, dtype=np.float32)
        source_mask = np.asarray(
            ((1, 1, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1)), dtype=np.uint8,
        )
        source_depth = np.full((3, 4), 850.0 + code, dtype=np.float32)
        source_ids = np.asarray((2, 7, 11), dtype=np.int64)
        source_pixels = np.asarray(((0.5, 0.5), (1.5, 1.5), (3.5, 2.5)), dtype=np.float64)
        source_points = np.asarray(
            ((-0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.1, -0.1, 0.0)),
            dtype=np.float64,
        )
        intrinsics = np.asarray(
            ((800.0, 0.0, 1.25), (0.0, 760.0, 1.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        source_pose = np.eye(4, dtype=np.float64)
        if int(seed_deg) == 180:
            source_pose[0, 0] = -1.0
            source_pose[2, 2] = -1.0
        query_mask = np.asarray(
            ((1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 1, 1)), dtype=np.uint8,
        )
        query_rgb = np.full((3, 4, 3), 0.25, dtype=np.float32)
        query_depth = np.full((3, 4), 1000.0, dtype=np.float32)
        positions = np.asarray(
            ((-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (0.0, 0.2, 0.0)),
            dtype=np.float64,
        )
        triangles = np.asarray(((0, 1, 2),), dtype=np.int64)

        field_logs: dict[str, list[str]] = {}
        forbidden_input = {
            "query_rgb", "query_depth_mm", "endpoint_stub_q0", "endpoint_stub_valid",
            "source_triangle_ids", "source_barycentric", "source_uv", "source_confidence",
            "challenge_yaw_offsets_deg", "source_seed_deg",
        }
        mesh = _PoisonMapping(
            {"positions_m": positions, "triangles": triangles,
             "uv": object(), "triangle_normals": object(), "texture_rgb": object()},
            forbidden={"uv", "triangle_normals", "texture_rgb"},
            access_log=field_logs.setdefault("fixed_mesh", []),
        )
        inputs = _PoisonMapping(
            {
                "source_rgb": source_rgb,
                "source_observed_mask": source_mask,
                "source_depth_mm": source_depth,
                "source_ids": source_ids,
                "source_pixels_xy": source_pixels,
                "source_points_object_m": source_points,
                "query_observed_mask": query_mask,
                "intrinsics": intrinsics,
                "source_seed_pose_m": source_pose,
                "fixed_mesh": mesh,
                "query_rgb": query_rgb,
                "query_depth_mm": query_depth,
                "endpoint_stub_q0": object(),
                "endpoint_stub_valid": object(),
                "source_triangle_ids": object(),
                "source_barycentric": object(),
                "source_uv": object(),
                "source_confidence": object(),
                "challenge_yaw_offsets_deg": object(),
            },
            forbidden=forbidden_input,
            access_log=field_logs.setdefault("estimator_inputs", []),
        )
        truth = _PoisonMapping(
            {"query_pose_camera_from_object_m": object(), "query_depth_mm": query_depth,
             "endpoint_packet_pose_m": object()},
            forbidden={"query_pose_camera_from_object_m", "query_depth_mm",
                       "endpoint_packet_pose_m"},
            access_log=field_logs.setdefault("evaluator_truth", []),
        )
        generator_arrays = _PoisonMapping(
            {"mesh_positions": positions, "mesh_triangles": triangles,
             "query_texture": query_rgb},
            forbidden={"mesh_positions", "mesh_triangles", "query_texture"},
            access_log=field_logs.setdefault("generator_arrays", []),
        )
        case = _PoisonMapping(
            {
                "case_id": case_id,
                "variant": variant,
                "source_seed_deg": float(seed_deg),
                "estimator_inputs": inputs,
                "evaluator_truth": truth,
                "generator_arrays": generator_arrays,
                "geometry": object(),
                "resource_budget": object(),
                "generator_rules": object(),
            },
            forbidden={"evaluator_truth", "generator_arrays", "geometry",
                       "resource_budget", "generator_rules"},
            access_log=field_logs.setdefault("case", []),
        )

        source_arrays = {
            "template_rgb": source_rgb,
            "template_mask": source_mask,
            "template_depth_mm": source_depth,
            "source_indices": source_ids,
            "source_pixels_xy": source_pixels,
            "source_points_object_m": source_points,
            "observed_crop_mask": source_mask,
            "crop_k": intrinsics,
            "native_k": intrinsics,
            "template_pose_m": source_pose,
        }
        records = {key: _typed_record(value) for key, value in source_arrays.items()}
        # C is adapter-created, so record the expected identity transform too.
        identity = np.eye(4, dtype=np.float64)
        records["crop_from_native"] = _typed_record(identity)
        ordered_records = {key: records[key] for key in _ARRAY_KEYS}
        namespace = {
            "context_id": case_id,
            "source_rgb_sha256": ordered_records["template_rgb"]["sha256"],
        }
        packet_digest = _sha(json.dumps(
            {"mapping_version": control.MAPPING_VERSION,
             "control_spec_sha256": control.CONTROL_SPEC_SHA256,
             "source_spec_v2_sha256": control.SOURCE_SPEC_SHA256,
             "generator_spec_sha256": str(generator.SPEC_SHA256).lower(),
             "case_id": case_id,
             "variant": variant,
             "source_seed_deg": float(seed_deg),
             "source_array_order": list(_ARRAY_KEYS),
             "source_array_records": ordered_records},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8"))
        mesh_records = {
            "positions": _typed_record(positions),
            "triangles": _typed_record(triangles),
        }
        mesh_digest = _sha(json.dumps(
            {"schema_version": "r8-control-native-mesh-v1",
             "positions_m": mesh_records["positions"],
             "triangles": mesh_records["triangles"]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8"))
        self.references[case_id] = {
            "case": weakref.ref(case),
            "inputs": weakref.ref(inputs),
            "fixed_mesh": weakref.ref(mesh),
            "truth": weakref.ref(truth),
            "generator_arrays": weakref.ref(generator_arrays),
            "query_rgb": weakref.ref(query_rgb),
            "query_depth": weakref.ref(query_depth),
            "source_arrays": {key: weakref.ref(value) for key, value in source_arrays.items()},
            "source_observed_mask": weakref.ref(source_mask),
            "query_observed_mask": weakref.ref(query_mask),
            "mesh_positions": weakref.ref(positions),
            "mesh_triangles": weakref.ref(triangles),
            "intrinsics": weakref.ref(intrinsics),
            "field_logs": field_logs,
        }
        self.source_records[case_id] = {
            "namespace": namespace,
            "source_array_records": ordered_records,
            "packet_sha256": packet_digest,
            "mesh_sha256": mesh_digest,
            "mesh_records": mesh_records,
            "query_observed_mask_record": _typed_record(query_mask),
        }
        self.forbidden_access_logs[case_id] = field_logs
        return case

    def _generate(self, variant_id: str, source_seed_deg: float):
        case_id = f"{variant_id}-seed-{int(source_seed_deg):03d}"
        self.generator_calls.append((variant_id, float(source_seed_deg)))
        # A prior case's full producer package and source projection must be
        # gone before the next generator allocation starts.
        for prior_id, refs in self.references.items():
            if prior_id == case_id:
                continue
            if prior_id in {item[0] for item in self.generator_calls[:-1]}:
                self.assertIsNone(refs["case"](), "previous full control case remained live")
                self.assertIsNone(refs["inputs"](), "previous estimator inputs remained live")
                self.assertIsNone(refs["fixed_mesh"](), "previous fixed mesh mapping remained live")
                self.assertTrue(all(ref() is None for ref in refs["source_arrays"].values()),
                                "previous source projection remained live into next generation")
        return self._make_case(variant_id, source_seed_deg)

    def _capacity_receipt(self, case_id: str, packet_sha: str, mesh_sha: str,
                          namespace: dict[str, str], *, available: bool = True) -> dict[str, object]:
        gates = {
            "witnesses_ge_8": bool(available),
            "fit_ge_24": bool(available),
            "bank_ge_1": True,
            "qh_possible_cells_ge_3": bool(available),
            "qh_maximum_bbox_hull_ge_0_12": bool(available),
        }
        failed = [name for name, value in gates.items() if not value]
        return {
            "schema_version": 1,
            "state": "available" if available else "unavailable",
            "capacity_available": available,
            "capacity_only": True,
            "full_28_case_screen_complete": False,
            "counterpart_context_bindings_verified": False,
            "source_plan_bindings_verified": False,
            "query_observed_mask_origin": "caller_supplied_observed",
            "source_context_id": case_id,
            "packet_sha256": packet_sha,
            "mesh_sha256": mesh_sha,
            "source_population_count": 3,
            "source_ids_sha256": _sha(b"ids" + case_id.encode("ascii")),
            "r5_eligible_count": 3,
            "r5_eligible_sha256": _sha(b"r5" + case_id.encode("ascii")),
            "r7_eligible_count": 3,
            "r7_eligible_sha256": _sha(b"r7" + case_id.encode("ascii")),
            "r5_r7_bit_exact": True,
            "appearance_supported_count": 3,
            "appearance_support_sha256": _sha(b"appearance" + case_id.encode("ascii")),
            "appearance_unsupported_reason_counts": {},
            "native_supported_count": 3,
            "native_support_sha256": _sha(b"native" + case_id.encode("ascii")),
            "native_reason_counts": {},
            "native_association_sha256": _sha(b"association" + case_id.encode("ascii")),
            "native_topology_state": "supported",
            "native_raycast_kernel_frozen_r7": True,
            "native_association": {
                "schema_version": 1,
                "source_context_id": case_id,
                "source_count": 3,
                "associated_count": 3,
                "unsupported_count": 0,
                "reason_counts": {},
                "raycast_kernel_is_frozen_r7": True,
            },
            "source_supported_count": 3,
            "source_support_sha256": _sha(b"support" + case_id.encode("ascii")),
            "witness_count": 8 if available else 7,
            "witness_ids_sha256": _sha(b"witness" + case_id.encode("ascii")),
            "witness_reservation": {
                "spec_v2_sha256": _SOURCE_SPEC_SHA256,
                "image_namespace": namespace,
                "state": "eligible" if available else "unavailable",
                "denominator": 8 if available else 7,
                "selected_ids_sha256": _sha(b"witness" + case_id.encode("ascii")),
            },
            "fit_pool_count": 24 if available else 23,
            "fit_pool_ids_sha256": _sha(b"fit-pool" + case_id.encode("ascii")),
            "fit_count": 24 if available else 23,
            "fit_ids_sha256": _sha(b"fit" + case_id.encode("ascii")),
            "bank_anchor_count": 1,
            "bank_ids_sha256": _sha(b"bank" + case_id.encode("ascii")),
            "new_bank_manifest_sha256": _sha(b"manifest" + case_id.encode("ascii")),
            "bank_state": "eligible",
            "source_fit_bank": {
                "spec_v2_sha256": _SOURCE_SPEC_SHA256,
                "image_namespace": namespace,
                "state": "eligible",
                "witness_denominator": 8 if available else 7,
                "witness_selected_ids_sha256": _sha(b"witness" + case_id.encode("ascii")),
                "fit_pool_count": 24 if available else 23,
                "fit_pool_ids_sha256": _sha(b"fit-pool" + case_id.encode("ascii")),
                "fit_count": 24 if available else 23,
                "fit_ids_sha256": _sha(b"fit" + case_id.encode("ascii")),
                "bank_anchor_count": 1,
                "bank_state": "eligible",
                "bank_anchor_ids_sha256": _sha(b"bank" + case_id.encode("ascii")),
                "bank_anchor_dependency_sha256": _sha(b"bank-dependencies" + case_id.encode("ascii")),
                "bank_dependency_union_sha256": _sha(b"bank-union" + case_id.encode("ascii")),
                "bank_builder_calls": 1,
                "bank_builder_is_frozen_audit_function": True,
                "fit_union_intersection_with_witness_count": 0,
                "bank_intersection_with_witness_count": 0,
            },
            "fit_union_intersection_with_witness_count": 0,
            "bank_intersection_with_witness_count": 0,
            "qh_possible_center_count": 7 if available else 0,
            "qh_possible_cell_count": 3 if available else 2,
            "qh_maximum_bbox_hull_fraction": 0.12 if available else 0.11,
            "gates": gates,
            "failed_conjuncts": failed,
            "claims": dict(_FALSE_CLAIMS),
            "source_support_provenance": "full_original_id_appearance_and_native_association_before_selection",
            "query_rgb_or_evaluator_truth_read": False,
            "matching_started": False,
            "fitting_started": False,
            "resource": {
                "limit_bytes": _WORKING_CAP,
                "input_resident_bytes": 0,
                "phase_peak_estimates_bytes": {"synthetic_mock": 4096},
                "wall_limit_seconds": 900.0,
                "elapsed_seconds": 0.001,
            },
        }

    def _capacity(self, source_arrays, **kwargs):
        case_id = kwargs["source_context_id"]
        refs = self.references[case_id]
        self.assertEqual(tuple(source_arrays), _ARRAY_KEYS)
        self.assertIsNone(refs["case"](), "full producer case survived into capacity")
        self.assertIsNone(refs["inputs"](), "estimator input mapping survived into capacity")
        self.assertIsNone(refs["fixed_mesh"](), "fixed mesh mapping survived into capacity")
        self.assertIsNone(refs["truth"](), "evaluator truth mapping survived into capacity")
        self.assertIsNone(refs["generator_arrays"](), "generator arrays mapping survived into capacity")
        self.assertIsNone(refs["query_rgb"](), "query RGB survived into capacity")
        self.assertIsNone(refs["query_depth"](), "query depth survived into capacity")
        expected_refs = refs["source_arrays"]
        mapping = {
            "template_rgb": "source_rgb",
            "template_mask": "source_observed_mask",
            "template_depth_mm": "source_depth_mm",
            "source_indices": "source_ids",
            "source_pixels_xy": "source_pixels_xy",
            "source_points_object_m": "source_points_object_m",
            "observed_crop_mask": "source_observed_mask",
            "crop_k": "intrinsics",
            "native_k": "intrinsics",
            "template_pose_m": "source_seed_pose_m",
        }
        for capacity_key, producer_key in mapping.items():
            self.assertIs(source_arrays[capacity_key], expected_refs[capacity_key]())
        identity = source_arrays["crop_from_native"]
        self.assertIsInstance(identity, np.ndarray)
        self.assertEqual(identity.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(identity, np.eye(4, dtype=np.float64))
        self.assertIsNot(identity, source_arrays["crop_k"])
        self.assertIs(kwargs["query_observed_mask"], refs["query_observed_mask"]())
        self.assertIs(kwargs["mesh_positions_m"], refs["mesh_positions"]())
        self.assertIs(kwargs["mesh_triangles"], refs["mesh_triangles"]())
        self.assertEqual(kwargs["deadline_monotonic"], self._expected_deadline)
        expected = self.source_records[case_id]
        self.assertEqual(kwargs["packet_sha256"], expected["packet_sha256"])
        self.assertEqual(kwargs["mesh_sha256"], expected["mesh_sha256"])
        self.assertEqual(
            kwargs["source_context_id"], case_id,
            "source capacity namespace must include the exact frozen case ID",
        )
        namespace = expected["namespace"]
        available = True
        if callable(self.capacity_state):
            state = self.capacity_state(case_id, len(self.capacity_calls), kwargs)
            if isinstance(state, dict):
                available = state.get("available", True)
        else:
            state = self.capacity_state or {}
            available = state.get(case_id, True) if isinstance(state, dict) else bool(state)
        record = self._capacity_receipt(
            case_id, kwargs["packet_sha256"], kwargs["mesh_sha256"],
            namespace, available=available,
        )
        if self.capacity_mutator is not None:
            self.capacity_mutator(case_id, record)
        unique_nbytes = sum(
            value.nbytes for value in {
                id(value): value for value in [*source_arrays.values(),
                                               kwargs["mesh_positions_m"],
                                               kwargs["mesh_triangles"],
                                               kwargs["query_observed_mask"]]
            }.values()
        )
        record["resource"]["input_resident_bytes"] = kwargs["resident_bytes"] + unique_nbytes
        record["resource"]["phase_peak_estimates_bytes"] = {
            "synthetic_mock": kwargs["resident_bytes"] + unique_nbytes + 4096,
        }
        self.capacity_calls.append({
            "case_id": case_id,
            "keys": tuple(source_arrays),
            "resident_bytes": kwargs["resident_bytes"],
            "deadline": kwargs["deadline_monotonic"],
            "packet_sha256": kwargs["packet_sha256"],
            "mesh_sha256": kwargs["mesh_sha256"],
            "query_mask_hash": _typed_record(kwargs["query_observed_mask"])["sha256"],
        })
        return record

    def _patch_runtime(self, stack, *, generator_fn=None, capacity_fn=None):
        stack.enter_context(mock.patch.object(
            control.generator, "generate_control_case", new=generator_fn or self._generate,
        ))
        stack.enter_context(mock.patch.object(
            control.generator, "_fixture_live_budget", new=self._fixture_live_budget,
        ))
        stack.enter_context(mock.patch.object(
            control.capacity, "evaluate_source_capacity", new=capacity_fn or self._capacity,
        ))

    def _run(self, *, resident_bytes=0, deadline=None):
        from contextlib import ExitStack

        self._expected_deadline = deadline if deadline is not None else time.monotonic() + 600.0
        with ExitStack() as stack:
            self._patch_runtime(stack)
            return control.screen_control_rows(
                resident_bytes=resident_bytes,
                deadline_monotonic=self._expected_deadline,
            )

    def _assert_terminal_receipt(self, exc_context):
        self.assertIsInstance(exc_context.exception, control.ControlCapacityError)
        receipt = exc_context.exception.receipt
        self.assertEqual(receipt["state"], "failed")
        self.assertFalse(receipt["complete"])
        expected_ids = [f"{variant}-seed-{seed:03d}" for variant in _VARIANTS for seed in _SEEDS]
        self.assertEqual([row["case_id"] for row in receipt["rows"]], expected_ids)
        self.assertTrue(all(
            row["state"] == "not_evaluated_due_terminal_failure"
            for row in receipt["rows"]
        ))
        self.assertFalse(receipt["capacity_available"])
        self.assertFalse(receipt["full_28_case_screen_complete"])
        self.assertFalse(receipt["matching_started"])
        self.assertFalse(receipt["fitting_started"])
        self.assertFalse(receipt["claims"]["accuracy"])
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return receipt

    def test_success_maps_source_only_fields_for_all_sixteen_exact_cases(self):
        self.capacity_state = {"blank_constant-seed-000": False,
                               "blank_constant-seed-180": False}
        result = self._run(resident_bytes=4096)

        expected_ids = [f"{variant}-seed-{seed:03d}" for variant in _VARIANTS for seed in _SEEDS]
        self.assertEqual(result["state"], "complete")
        self.assertTrue(result["complete"])
        self.assertEqual([row["case_id"] for row in result["rows"]], expected_ids)
        self.assertEqual(
            self.generator_calls,
            [(variant, float(seed)) for variant in _VARIANTS for seed in _SEEDS],
        )
        self.assertEqual([call["case_id"] for call in self.capacity_calls], expected_ids)
        self.assertEqual(self.budget_calls, 16)
        self.assertTrue(result["advance_gate_passed"])
        self.assertEqual(control.CONTROL_SPEC_SHA256,
                         "889e6e4ab708fd552f9134f72ae14370c2e900389130fba80ff1f5465e2757b5")
        self.assertEqual(control.SOURCE_SPEC_SHA256, _SOURCE_SPEC_SHA256.lower())
        self.assertEqual(tuple(key for key, _ in control.ARRAY_MAP), _ARRAY_KEYS)
        self.assertEqual(result["planned_procedural_generator_render_calls"], 32)
        self.assertEqual(result["procedural_generator_render_calls"], 32)
        self.assertEqual(result["capacity_stage_calls"], 16)
        rows_by_id = {row["case_id"]: row for row in result["rows"]}
        for index, case_id in enumerate(expected_ids):
            row = rows_by_id[case_id]
            variant, seed_text = case_id.rsplit("-seed-", 1)
            self.assertEqual(row["variant"], variant)
            self.assertEqual(row["source_seed_deg"], int(seed_text))
            self.assertEqual(row["procedural_generator_render_calls"], 2)
            self.assertEqual(row["mandatory"], variant != "blank_constant")
            self.assertEqual(row["blank_may_abstain"], variant == "blank_constant")
            self.assertEqual(row["namespace"]["context_id"], case_id)
            self.assertEqual(row["namespace"], self.source_records[case_id]["namespace"])
            self.assertEqual(row["source_array_records"],
                             self.source_records[case_id]["source_array_records"])
            self.assertEqual(row["packet_sha256"], self.source_records[case_id]["packet_sha256"])
            self.assertEqual(row["mesh_sha256"], self.source_records[case_id]["mesh_sha256"])
            self.assertEqual(row["query_observed_mask"],
                             self.source_records[case_id]["query_observed_mask_record"])
            self.assertEqual(row["capacity_receipt"]["source_context_id"], case_id)
            self.assertEqual(row["capacity_receipt"]["packet_sha256"], row["packet_sha256"])
            self.assertEqual(row["capacity_receipt"]["mesh_sha256"], row["mesh_sha256"])
            # A forbidden lookup raises immediately from the mapping.
        self.assertEqual(len({row["namespace"]["source_rgb_sha256"] for row in result["rows"]}), 16)
        self.assertTrue(all(call["deadline"] == self._expected_deadline for call in self.capacity_calls))
        self.assertLess(self.capacity_calls[0]["resident_bytes"], self.capacity_calls[-1]["resident_bytes"])
        resource = result["resource"]
        self.assertEqual(resource["caller_resident_bytes"], 4096)
        self.assertEqual(resource["adapter_reserve_bytes"], control.REPORT_RESERVE_BYTES)
        self.assertGreater(resource["completed_report_graph_bytes"], 0)
        self.assertGreater(resource["capacity_prospective_input_bytes"],
                           resource["capacity_external_resident_bytes"])
        self.assertLessEqual(resource["capacity_prospective_input_bytes"], _WORKING_CAP)
        self.assertLessEqual(max(resource["phase_peak_estimates_bytes"].values()), _WORKING_CAP)
        prior_report_bytes = _graph_bytes([result["source_records"][0]])
        generation_peak = result["rows"][1]["resource"]["generation_prospective_peak_bytes"]
        self.assertEqual(
            generation_peak,
            4096 + prior_report_bytes + 24 + control.REPORT_RESERVE_BYTES + _FROZEN_GENERATOR_PEAK,
            "the one-entry append-grown source_records list is part of retained report residency",
        )
        self.assertEqual(
            self.capacity_calls[1]["resident_bytes"],
            4096 + prior_report_bytes + 24 + control.REPORT_RESERVE_BYTES,
        )
        self.assertEqual(result["capacity_stage_production_render_calls"], 0)
        self.assertEqual(result["capacity_stage_model_calls"], 0)
        self.assertEqual(result["capacity_stage_ncc_calls"], 0)
        self.assertEqual(result["capacity_stage_fitting_calls"], 0)
        self.assertEqual(result["capacity_stage_optimizer_calls"], 0)
        self.assertFalse(result["claims"]["full_28_case_screen_complete"])
        self.assertFalse(result["claims"]["pose"])
        self.assertFalse(result["claims"]["accuracy"])
        self.assertFalse(result["claims"]["matching"])
        self.assertFalse(result["claims"]["fitting"])
        serialized = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertLessEqual(len(serialized.encode("utf-8")), control.MAX_RECEIPT_BYTES)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in _walk(result)))
        self.assertFalse(any(key in {"query_rgb", "evaluator_truth", "query_depth_mm",
                                    "endpoint_stub_q0", "generator_arrays"}
                             for key in _walk(result)))

    def test_mandatory_unavailability_fails_advance_while_blank_can_abstain(self):
        self.capacity_state = {
            "blank_constant-seed-000": False,
            "blank_constant-seed-180": False,
            "query_occlusion-seed-000": False,
        }
        result = self._run()
        self.assertFalse(result["advance_gate_passed"])
        blank_rows = [row for row in result["rows"] if row["variant"] == "blank_constant"]
        self.assertEqual(len(blank_rows), 2)
        self.assertTrue(all(row["blank_may_abstain"] and not row["capacity_available"]
                            for row in blank_rows))
        mandatory_row = next(row for row in result["rows"]
                             if row["case_id"] == "query_occlusion-seed-000")
        self.assertTrue(mandatory_row["mandatory"])
        self.assertFalse(mandatory_row["capacity_available"])
        self.assertIn("witnesses_ge_8", mandatory_row["failed_conjuncts"])

    def test_generator_peak_plus_caller_reservation_fails_before_generation(self):
        caller_resident = _WORKING_CAP - _FROZEN_GENERATOR_PEAK + 1
        from contextlib import ExitStack
        with ExitStack() as stack:
            self._patch_runtime(stack)
            with self.assertRaises(control.ControlCapacityError) as context:
                control.screen_control_rows(resident_bytes=caller_resident,
                                            deadline_monotonic=time.monotonic() + 500)
        receipt = self._assert_terminal_receipt(context)
        self.assertEqual(self.generator_calls, [])
        self.assertEqual(self.capacity_calls, [])
        self.assertEqual(receipt["failure"]["stage"], "generation_and_capacity")
        self.assertEqual(receipt["resource"]["generator_peak_bytes"], _FROZEN_GENERATOR_PEAK)
        self.assertEqual(
            receipt["resource"]["generation_prospective_peak_bytes"],
            caller_resident + control.REPORT_RESERVE_BYTES + _FROZEN_GENERATOR_PEAK,
        )
        self.assertGreater(receipt["resource"]["generation_prospective_peak_bytes"], _WORKING_CAP)

    def test_shared_deadline_is_clamped_and_passed_through_to_capacity(self):
        from contextlib import ExitStack

        clock = [100.0]
        self._expected_deadline = 1000.0
        with mock.patch.object(control.time, "monotonic", new=lambda: clock[0]):
            with ExitStack() as stack:
                self._patch_runtime(stack)
                result = control.screen_control_rows(
                    resident_bytes=0, deadline_monotonic=5000.0,
                )
        self.assertTrue(result["complete"])
        self.assertTrue(all(call["deadline"] == 1000.0 for call in self.capacity_calls))
        self.assertEqual(result["resource"]["wall_limit_seconds"], control.MAX_WALL_SECONDS)

    def test_deadline_expiring_during_generation_stops_before_capacity_with_unknown_renders(self):
        clock = [10.0]

        def expiring_generator(variant_id, seed):
            case = self._generate(variant_id, seed)
            clock[0] = 20.0
            return case

        from contextlib import ExitStack
        with mock.patch.object(control.time, "monotonic", new=lambda: clock[0]):
            with ExitStack() as stack:
                self._patch_runtime(stack, generator_fn=expiring_generator)
                with self.assertRaises(control.ControlCapacityError) as context:
                    control.screen_control_rows(resident_bytes=128,
                                                deadline_monotonic=15.0)
        receipt = self._assert_terminal_receipt(context)
        self.assertEqual(len(self.generator_calls), 1)
        self.assertEqual(self.capacity_calls, [])
        self.assertEqual(receipt["failure"]["stage"], "generation_and_capacity")
        self.assertEqual(receipt["failure"]["case_id"], "distinct_markers-seed-000")
        self.assertEqual(receipt["failure"]["current_case_generator_render_calls"], 2)

    def test_generator_and_capacity_failures_keep_sixteen_unmeasured_rows(self):
        from contextlib import ExitStack
        for stage in ("generator", "capacity"):
            with self.subTest(stage=stage):
                self.setUp()
                def failing_generator(variant_id, seed):
                    if len(self.generator_calls) == 1:
                        self.generator_calls.append((variant_id, float(seed)))
                        raise RuntimeError("synthetic generation failure")
                    return self._generate(variant_id, seed)

                def failing_capacity(source_arrays, **kwargs):
                    if len(self.capacity_calls) == 1:
                        raise RuntimeError("synthetic capacity helper failure")
                    return self._capacity(source_arrays, **kwargs)

                deadline = time.monotonic() + 500
                self._expected_deadline = deadline
                with ExitStack() as stack:
                    self._patch_runtime(
                        stack,
                        generator_fn=failing_generator if stage == "generator" else self._generate,
                        capacity_fn=self._capacity if stage == "generator" else failing_capacity,
                    )
                    with self.assertRaises(control.ControlCapacityError) as context:
                        control.screen_control_rows(resident_bytes=256,
                                                    deadline_monotonic=deadline)
                receipt = self._assert_terminal_receipt(context)
                if stage == "generator":
                    self.assertEqual(len(receipt["source_records"]), 1)
                    self.assertEqual(receipt["failure"]["current_case_generator_render_calls"],
                                     "unknown_or_partial")
                else:
                    self.assertEqual(len(receipt["source_records"]), 1)
                    self.assertEqual(receipt["failure"]["current_case_generator_render_calls"], 2)
                self.assertEqual(len(receipt["rows"]), 16)

    def test_namespace_packet_or_mesh_mismatch_is_terminal(self):
        for mismatch in ("namespace", "packet", "mesh"):
            with self.subTest(mismatch=mismatch):
                self.setUp()
                def corrupt(_case_id, record):
                    if len(self.capacity_calls) != 0:
                        return
                    if mismatch == "namespace":
                        record["source_fit_bank"]["image_namespace"]["context_id"] = "wrong-case"
                    elif mismatch == "packet":
                        record["packet_sha256"] = "0" * 64
                    else:
                        record["mesh_sha256"] = "0" * 64

                self.capacity_mutator = corrupt
                with self.assertRaises(control.ControlCapacityError) as context:
                    self._run()
                receipt = self._assert_terminal_receipt(context)
                self.assertEqual(len(receipt["rows"]), 16)
                self.assertEqual(receipt["failure"]["stage"], "generation_and_capacity")


if __name__ == "__main__":
    unittest.main()
