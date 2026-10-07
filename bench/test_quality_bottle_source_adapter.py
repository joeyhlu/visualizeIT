"""Independent CPU tests for the R8 actual-source adapter.

The adapter control flow is exercised with producer-shaped synthetic metadata
and mocked packet decoding, row capacity evaluation, and GLB loading. No
actual-cache packet, model, renderer, or evaluator truth is used here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from . import quality_bottle_patch_pose_calibration as r6
from . import quality_bottle_pose_ablation as r5
from . import quality_bottle_source_adapter as adapter
from . import quality_bottle_source_capacity as capacity_module
from . import glb_model as glb_module
from . import quality_bottle_source_packet as packet
from .test_quality_bottle_source_bindings import _expected_plans, _fixture as _metadata_fixture


_FRAMES = (10, 50, 100)
_MODEL_QUALITY_PINS = (
    "diagnostics/bottle-identity-v2/capture.json",
    "diagnostics/bottle-zero-view-calibration-v3/zero_view.json",
    "inputs/ranch/input.json",
    "inputs/ranch/object.glb",
    "diagnostics/bottle-source-observability-pose-ablation-v1/pose_ablation.json",
    "diagnostics/bottle-r5-parent-v1/terminal.json",
    "diagnostics/bottle-patch-pose-calibration-v2/calibration.json",
    "diagnostics/bottle-r6-parent-v2/terminal.json",
)
_SOURCE_ARRAY_KEYS = tuple(packet.SOURCE_ARRAY_KEYS)
_PROJECTED_SOURCE_KEYS = tuple(
    key for key in _SOURCE_ARRAY_KEYS if key not in {"template_gray_rgb", "seed_pose_m"}
)
_FALSE_CLAIMS = {"fullscreen": False, "accuracy": False, "pose": False}
_CAPACITY_GATES = (
    "witnesses_ge_8", "fit_ge_24", "bank_ge_1",
    "qh_possible_cells_ge_3", "qh_maximum_bbox_hull_ge_0_12",
)
_MIB = 1024 * 1024


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _typed_array_sha(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8") + b"\0"
    digest = hashlib.sha256(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _synthetic_glb() -> bytes:
    """A tiny valid GLB 2.0 envelope for the adapter's metadata preflight."""
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 102}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36},
            {"buffer": 0, "byteOffset": 72, "byteLength": 24},
            {"buffer": 0, "byteOffset": 96, "byteLength": 6},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3,
        }]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary_chunk = bytes(104)  # GLB chunks are padded to four-byte alignment.
    total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    return (
        struct.pack("<III", 0x46546C67, 2, total) +
        struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk +
        struct.pack("<II", len(binary_chunk), 0x004E4942) + binary_chunk
    )


def _array_info(entry: dict, name: str) -> dict:
    # The accepted metadata fixture deliberately guards every non-whitelisted
    # counterpart field. This helper reaches only the binder's safe projection.
    return copy.deepcopy(entry["arrays"][name])


def _small_projection(source_entry: dict) -> dict[str, np.ndarray]:
    """Make a fresh loader-shaped eleven-field projection for one view."""
    arrays = {}
    for key in _PROJECTED_SOURCE_KEYS:
        metadata = source_entry["arrays"][key]
        arrays[key] = np.zeros(tuple(metadata["shape"]), dtype=np.dtype(metadata["dtype"]))
    arrays["template_mask"].fill(1)
    arrays["observed_crop_mask"].fill(1)
    return arrays


class _TemporaryInputs:
    """Synthetic pinned directory tree; reports are opaque hash-only bytes."""

    def __init__(self, temporary: tempfile.TemporaryDirectory):
        self.root = Path(temporary.name) / "model-quality"
        self.capture_root = self.root / "diagnostics" / "bottle-identity-v2"
        self.zero_root = self.root / "diagnostics" / "bottle-zero-view-calibration-v3"
        self.input_root = self.root / "inputs" / "ranch"
        self.capture_root.mkdir(parents=True)
        self.zero_root.mkdir(parents=True)
        self.input_root.mkdir(parents=True)
        self.payloads: dict[str, bytes] = {}
        for index, relative in enumerate(_MODEL_QUALITY_PINS):
            if relative.endswith("capture.json") or relative.endswith("zero_view.json"):
                payload = b"{}\n"
            elif relative.endswith("input.json"):
                payload = b'{"fixture":"synthetic-r8-input"}\n'
            elif relative.endswith("object.glb"):
                payload = _synthetic_glb()
            else:
                # These closed R5/R6 provenance artifacts must be hashed, not
                # parsed as JSON or read for outcome fields.
                payload = bytes((0xFF,)) + f"sealed-report-{index}".encode("ascii")
            path = self.root.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.payloads[relative] = payload

    def path_for(self, relative: str) -> Path:
        return self.root.joinpath(*relative.split("/"))

    def patch_pin_specs(self):
        """Build test pins while retaining the implementation's value shape."""
        current = adapter.PINNED_FILE_SPECS
        specs = {}
        for relative, payload in self.payloads.items():
            old = current[relative]
            digest = _sha(payload)
            if isinstance(old, dict):
                item = dict(old)
                item["bytes"] = len(payload)
                item["sha256"] = digest
            elif isinstance(old, tuple):
                item = tuple(digest if isinstance(value, str) else len(payload)
                             for value in old)
            else:
                raise AssertionError(f"Unexpected pin spec representation for {relative}: {old!r}")
            specs[relative] = item
        return mock.patch.object(adapter, "PINNED_FILE_SPECS", specs)

    def mutate(self, relative: str, *, same_length: bool = False) -> None:
        path = self.path_for(relative)
        value = path.read_bytes()
        if same_length:
            changed = bytes((value[0] ^ 1,)) + value[1:]
        else:
            changed = value + b"x"
        path.write_bytes(changed)

    def file_set(self) -> set[str]:
        return {str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()}


class ActualSourceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.inputs = _TemporaryInputs(self.temporary)
        self.capture, self.zero, self.plans, self.counterparts = _metadata_fixture()
        self._materialize_source_packet_pins()
        self.closed_ids = [plan["condition_id"] for plan in self.plans]
        self.mesh_positions = np.asarray(
            ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0)),
            dtype=np.float64,
        )
        # Keep this winding exactly as returned by the mocked native GLB read.
        self.mesh_triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
        self.mesh = SimpleNamespace(
            name="ranch", positions=self.mesh_positions,
            triangles=self.mesh_triangles,
            normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float64),
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)), dtype=np.float64),
        )
        self._renderer_attribute_refs = (
            weakref.ref(self.mesh.normals), weakref.ref(self.mesh.uv),
        )
        self._renderer_attribute_bytes = self.mesh.normals.nbytes + self.mesh.uv.nbytes
        self.mesh_file_sha = _sha(self.inputs.payloads["inputs/ranch/object.glb"])
        self.logs: dict[str, list] = {"loader": [], "capacity": [], "mesh": []}
        self._weak_arrays: list[weakref.ReferenceType] = []
        self._namespace_sha_by_context: dict[str, str] = {}
        self._next_capacity_state = None
        self._on_capacity = None

    def _read_json(self, stream, *args, **kwargs):
        path = Path(stream.name)
        if path.name == "capture.json":
            return self.capture
        if path.name == "zero_view.json":
            return self.zero
        if path.name == "input.json":
            return {"fixture": "synthetic-r8-input"}
        raise AssertionError(f"Closed R5/R6 report was parsed as JSON: {path}")

    def _capacity_record(self, context_id: str, packet_sha: str, mesh_sha: str,
                         *, all_gates: bool = True, bank_count: int = 1,
                         source_rgb_sha256: str | None = None,
                         input_resident_bytes: int = 1024) -> dict:
        namespace_sha = source_rgb_sha256 or _sha(context_id.encode("ascii"))
        gates = {name: bool(all_gates) for name in _CAPACITY_GATES}
        gates["bank_ge_1"] = bank_count >= 1
        failed = [name for name, passed in gates.items() if not passed]
        available = not failed
        return {
            "schema_version": 1,
            "state": "available" if available else "unavailable",
            "capacity_available": available,
            "capacity_only": True,
            "full_28_case_screen_complete": False,
            "counterpart_context_bindings_verified": False,
            "source_plan_bindings_verified": False,
            "query_observed_mask_origin": "caller_supplied_observed",
            "source_context_id": context_id,
            "packet_sha256": packet_sha,
            "mesh_sha256": mesh_sha,
            "source_population_count": 64,
            "source_ids_sha256": _sha(context_id.encode("ascii")),
            "source_supported_count": 64,
            "source_support_sha256": _sha((context_id + "-support").encode("ascii")),
            "witness_count": 8 if all_gates else 7,
            "witness_ids_sha256": _sha((context_id + "-witness").encode("ascii")),
            "fit_count": 24 if all_gates else 23,
            "bank_anchor_count": bank_count,
            "bank_ids_sha256": _sha((context_id + "-bank").encode("ascii")),
            "bank_state": "eligible" if bank_count else "unavailable",
            "new_bank_manifest_sha256": _sha((context_id + "-manifest").encode("ascii")),
            "qh_possible_cell_count": 3 if all_gates else 2,
            "qh_possible_center_count": 7 if all_gates else 0,
            "qh_maximum_bbox_hull_fraction": 0.12 if all_gates else 0.11,
            "gates": gates,
            "failed_conjuncts": failed,
            "claims": dict(_FALSE_CLAIMS),
            "source_support_provenance": "full_original_id_appearance_and_native_association_before_selection",
            "fit_union_intersection_with_witness_count": 0,
            "bank_intersection_with_witness_count": 0,
            "source_fit_bank": {
                "spec_v2_sha256": "32758DDA6AD550A5EDD94F5354B7E8401FF206A63B884E8C88AB37F71EE6D9CB",
                "image_namespace": {
                    "context_id": context_id,
                    "source_rgb_sha256": namespace_sha,
                },
                "state": "eligible" if bank_count else "unavailable",
                "witness_denominator": 8 if all_gates else 7,
                "witness_selected_ids_sha256": _sha((context_id + "-witness").encode("ascii")),
                "fit_pool_count": 24 if all_gates else 23,
                "fit_pool_ids_sha256": _sha((context_id + "-pool").encode("ascii")),
                "fit_count": 24 if all_gates else 23,
                "fit_ids_sha256": _sha((context_id + "-fit").encode("ascii")),
                "bank_anchor_count": bank_count,
                "bank_state": "eligible" if bank_count else "unavailable",
                "bank_anchor_ids_sha256": _sha((context_id + "-bank").encode("ascii")),
                "bank_anchor_dependency_sha256": _sha((context_id + "-bank-deps").encode("ascii")),
                "bank_dependency_union_sha256": _sha((context_id + "-bank-union").encode("ascii")),
                "bank_builder_calls": 1,
                "bank_builder_is_frozen_audit_function": True,
                "fit_union_intersection_with_witness_count": 0,
                "bank_intersection_with_witness_count": 0,
            },
            "witness_reservation": {
                "spec_v2_sha256": "32758DDA6AD550A5EDD94F5354B7E8401FF206A63B884E8C88AB37F71EE6D9CB",
                "state": "eligible" if all_gates else "unavailable",
                "image_namespace": {
                    "context_id": context_id,
                    "source_rgb_sha256": namespace_sha,
                },
                "denominator": 8 if all_gates else 7,
                "selected_ids_sha256": _sha((context_id + "-witness").encode("ascii")),
            },
            "matching_started": False,
            "fitting_started": False,
            "bank_intersection_with_witness_count": 0,
            "query_rgb_or_evaluator_truth_read": False,
            "resource": {
                "limit_bytes": 128 * _MIB,
                "input_resident_bytes": input_resident_bytes,
                "phase_peak_estimates_bytes": {"synthetic_mock": input_resident_bytes + 2048},
                "wall_limit_seconds": 900.0,
                "elapsed_seconds": 0.001,
            },
        }

    def _materialize_source_packet_pins(self):
        packet_root = self.inputs.capture_root / "packets" / "contexts"
        packet_root.mkdir(parents=True)
        for entry in self.capture["contexts"]:
            if entry["role"] != "template":
                continue
            context_id = entry["context_id"]
            payload = b"synthetic-source-packet:" + context_id.encode("ascii")
            path = packet_root / f"{context_id}.npz"
            path.write_bytes(payload)
            # The accepted producer-shaped fixture uses guarded mappings. Keep
            # their safe template pin fields aligned with these opaque bytes.
            dict.__setitem__(entry, "bytes", len(payload))
            dict.__setitem__(entry, "sha256", _sha(payload))

    def _read_mesh(self, path, name):
        self.logs["mesh"].append((Path(path), name))
        return self.mesh

    def _loader(self, capture_root, entry, **kwargs):
        for reference in self._weak_arrays:
            self.assertIsNone(reference(), "previous source view survived into the next loader call")
        self.assertIsNone(self._renderer_attribute_refs[0](),
                          "GLB renderer normals remained live during source loading")
        self.assertIsNone(self._renderer_attribute_refs[1](),
                          "GLB renderer UVs remained live during source loading")
        context_id = kwargs["expected_context_id"]
        frame_id = kwargs["expected_frame_id"]
        offset = kwargs["expected_offset_deg"]
        source_entry = entry
        arrays = _small_projection(source_entry)
        self.assertEqual(set(arrays), set(_PROJECTED_SOURCE_KEYS))
        self._weak_arrays = [weakref.ref(value) for value in arrays.values()]
        source_sha = source_entry["sha256"]
        self._namespace_sha_by_context[context_id] = source_entry["arrays"]["template_rgb"]["sha256"]
        array_metadata = {
            key: value for key, value in source_entry["arrays"].items()
            if key in _PROJECTED_SOURCE_KEYS
        }
        counterpart_mask = source_entry["arrays"]["observed_crop_mask"]
        # The adapter must independently bind both masks before each unique
        # view is decoded, including views shared by self and synthetic rows.
        self.assertEqual(kwargs["real_frame_mask_metadata"], counterpart_mask)
        self.assertEqual(kwargs["synthetic_mask_metadata"], counterpart_mask)
        self.assertTrue(kwargs["require_synthetic_mask"])
        self.logs["loader"].append({
            "context_id": context_id,
            "frame_id": frame_id,
            "offset": offset,
            "resident_bytes": kwargs["resident_bytes"],
            "deadline": kwargs["deadline_monotonic"],
            "real_mask": copy.deepcopy(kwargs["real_frame_mask_metadata"]),
            "synthetic_mask": copy.deepcopy(kwargs["synthetic_mask_metadata"]),
        })
        receipt = {
            "schema_version": 1,
            "state": "loaded_source_template",
            "context_id": context_id,
            "frame_id": frame_id,
            "template_offset_deg": offset,
            "packet": {
                "relative_path": source_entry["path"],
                "bytes": source_entry["bytes"],
                "sha256": source_sha,
            },
            "source_array_metadata": array_metadata,
            "observed_crop_mask_metadata": counterpart_mask,
            "counterpart_mask_checks": [
                {"label": "real_frame", "matches_source_observed_crop_mask": True},
                {"label": "synthetic_query", "matches_source_observed_crop_mask": True},
            ],
            "archive_member_count": 13,
            "packet_sha256_rechecked": True,
            "r5_source_only_loader_calls": 1,
            "dropped_source_arrays": ["template_gray_rgb", "seed_pose_m"],
            "resident_bytes": kwargs["resident_bytes"],
            "loaded_array_bytes": sum(value.nbytes for value in arrays.values()),
            "prospective_peak_bytes": (
                kwargs["resident_bytes"] + 2 * sum(value.nbytes for value in arrays.values()) +
                max(value.nbytes for value in arrays.values()) + packet.HASH_CHUNK_BYTES +
                packet.DECODE_FIXED_ALLOWANCE_BYTES
            ),
            "working_limit_bytes": 128 * _MIB,
            "query_or_evaluator_data_read": False,
            "fitting_or_matching_started": False,
            "capacity_screen_complete": False,
            "counterpart_context_bindings_verified": False,
            "observed_mask_provenance": "caller_supplied_matching_array_metadata_only",
            "deadline_seconds_remaining": 100.0,
        }
        return {
            "arrays": arrays,
            "receipt": receipt,
            "image_namespace": {
                "context_id": context_id,
                "source_rgb_sha256": source_entry["arrays"]["template_rgb"]["sha256"],
            },
        }

    def _capacity(self, source_arrays, **kwargs):
        context_id = kwargs["source_context_id"]
        self.assertEqual(set(source_arrays), set(_PROJECTED_SOURCE_KEYS))
        self.assertIs(kwargs["query_observed_mask"], source_arrays["observed_crop_mask"])
        self.assertIs(kwargs["mesh_positions_m"], self.mesh_positions)
        self.assertIs(kwargs["mesh_triangles"], self.mesh_triangles)
        self.assertEqual(kwargs["mesh_sha256"], self.mesh_file_sha)
        self.assertEqual(kwargs["deadline_monotonic"], self._shared_deadline)
        record_kwargs = {
            "context_id": context_id,
            "packet_sha": kwargs["packet_sha256"],
            "mesh_sha": kwargs["mesh_sha256"],
            "source_rgb_sha256": self._namespace_sha_by_context[context_id],
            "input_resident_bytes": (
                kwargs["resident_bytes"] + sum(value.nbytes for value in source_arrays.values()) +
                self.mesh_positions.nbytes + self.mesh_triangles.nbytes
            ),
        }
        if callable(self._next_capacity_state):
            state = self._next_capacity_state(context_id, kwargs)
        else:
            state = self._next_capacity_state or {}
        result = self._capacity_record(**record_kwargs, **state)
        self.logs["capacity"].append({
            "context_id": context_id,
            "packet_sha256": kwargs["packet_sha256"],
            "mesh_sha256": kwargs["mesh_sha256"],
            "resident_bytes": kwargs["resident_bytes"],
            "deadline": kwargs["deadline_monotonic"],
            "query_mask_was_source_observed": True,
            "position_hash": _typed_array_sha(kwargs["mesh_positions_m"]),
            "triangle_hash": _typed_array_sha(kwargs["mesh_triangles"]),
        })
        if self._on_capacity is not None:
            self._on_capacity(len(self.logs["capacity"]), context_id)
        return result

    def _metadata_reader_patch(self):
        return mock.patch.object(adapter.json, "load", side_effect=self._read_json)

    def _runtime_patches(self, *, loader=None, capacity=None, mesh=None):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(self.inputs.patch_pin_specs())
        self._synthetic_core_root = self._patch_synthetic_core_tree(stack)
        stack.enter_context(self._metadata_reader_patch())
        stack.enter_context(mock.patch.object(packet, "load_source_template",
                                               new=loader or self._loader))
        stack.enter_context(mock.patch.object(capacity_module, "evaluate_source_capacity",
                                               new=capacity or self._capacity))
        stack.enter_context(mock.patch.object(glb_module, "read_glb",
                                               new=mesh or self._read_mesh))
        stack.enter_context(mock.patch.object(r5, "_load_bound_manifests",
                                               side_effect=AssertionError("R5 cache evaluator loader forbidden")))
        stack.enter_context(mock.patch.object(r5, "_load_condition_payload",
                                               side_effect=AssertionError("R5 forward packet loader forbidden")))
        stack.enter_context(mock.patch.object(r5, "_query_truth_for_plan",
                                               side_effect=AssertionError("R5 evaluator truth forbidden")))
        stack.enter_context(mock.patch.object(r6, "preflight_actual_cache",
                                               side_effect=AssertionError("R7/R6 actual preflight forbidden")))
        return stack

    def _patch_synthetic_core_tree(self, stack):
        repo_root = Path(self.temporary.name) / "synthetic-repo"
        specs = {}
        for relative in adapter.PINNED_CORE_SPECS:
            payload = ("synthetic-core:" + relative).encode("utf-8")
            path = repo_root.joinpath(*Path(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            specs[relative] = _sha(payload)
        stack.enter_context(mock.patch.object(adapter, "PINNED_CORE_SPECS", specs))
        stack.enter_context(mock.patch.object(adapter, "_repo_root", return_value=repo_root))
        return repo_root

    def _run(self, *, resident_bytes=137, deadline=None):
        self._shared_deadline = time.monotonic() + 500.0 if deadline is None else deadline
        return adapter.bind_and_screen_actual_rows(
            self.inputs.capture_root,
            self.inputs.zero_root,
            self.inputs.input_root,
            resident_bytes=resident_bytes,
            deadline_monotonic=self._shared_deadline,
        )

    def _terminal_receipt(self, context):
        self.assertIsInstance(context.exception, adapter.SourceAdapterError)
        receipt = context.exception.receipt
        self.assertEqual(receipt["state"], "failed")
        self.assertFalse(receipt["complete"])
        self.assertEqual([row["condition_id"] for row in receipt["rows"]], self.closed_ids)
        self.assertTrue(all(
            row["state"] == "not_evaluated_due_terminal_failure"
            for row in receipt["rows"]
        ))
        self.assertFalse(receipt["full_28_case_screen_complete"])
        self.assertFalse(receipt["matching_started"])
        self.assertFalse(receipt["fitting_started"])
        self.assertFalse(receipt["accuracy"])
        json.dumps(receipt, separators=(",", ":"), allow_nan=False)
        return receipt

    def test_success_screens_six_distinct_views_once_and_emits_twelve_compact_rows(self):
        before = self.inputs.file_set()
        self._next_capacity_state = {"all_gates": True, "bank_count": 1}
        with self._runtime_patches():
            result = self._run()

        self.assertEqual(result["state"], "complete")
        self.assertTrue(result["complete"])
        self.assertEqual([row["condition_id"] for row in result["rows"]], self.closed_ids)
        expected_contexts = [
            f"template-{frame_id:04d}-{offset:03d}"
            for frame_id in _FRAMES for offset in (0, 180)
        ]
        self.assertEqual([row["context_id"] for row in result["source_views"]], expected_contexts)
        self.assertEqual([row["context_id"] for row in self.logs["loader"]], expected_contexts)
        self.assertEqual([row["context_id"] for row in self.logs["capacity"]], expected_contexts)
        self.assertEqual(len({(row["context_id"], row["packet_sha256"])
                              for row in self.logs["capacity"]}), 6)
        self.assertTrue(all(row["query_mask_was_source_observed"]
                            for row in self.logs["capacity"]))
        self.assertTrue(all(row["deadline"] == self._shared_deadline
                            for row in self.logs["loader"] + self.logs["capacity"]))
        mesh_retained = self.mesh_positions.nbytes + self.mesh_triangles.nbytes
        self.assertTrue(all(row["resident_bytes"] >= 137 + self.mesh_positions.nbytes +
                            self.mesh_triangles.nbytes for row in self.logs["loader"]))
        self.assertTrue(all(row["resident_bytes"] >= 137 + self.mesh_positions.nbytes +
                            self.mesh_triangles.nbytes + 2 * _MIB
                            for row in self.logs["loader"]))
        self.assertTrue(all(row["resident_bytes"] >= 137
                            for row in self.logs["capacity"]))
        self.assertEqual(self.logs["mesh"], [(self.inputs.input_root / "object.glb", "ranch")])
        self.assertEqual(self.logs["capacity"][0]["triangle_hash"],
                         _typed_array_sha(self.mesh_triangles))
        self.assertEqual(result["mesh"]["positions"]["typed_sha256"],
                         _typed_array_sha(self.mesh_positions))
        self.assertEqual(result["mesh"]["triangles"]["typed_sha256"],
                         _typed_array_sha(self.mesh_triangles))
        self.assertEqual(result["mesh"]["triangle_order_and_winding"],
                         "original_CPU_GLTF_reader_output")
        resource = result["resource"]
        self.assertLessEqual(resource["prospective_peak_estimate_bytes"], 128 * _MIB)
        self.assertEqual(resource["adapter_external_resident_bytes"], 137)
        phase_peaks = resource["phase_peak_estimates_bytes"]
        self.assertGreaterEqual(phase_peaks["mesh_reader"],
                                137 + result["mesh"]["reader_peak_estimate_bytes"])
        self.assertGreaterEqual(phase_peaks["mesh_retained"], 137 + mesh_retained)
        self.assertTrue(all(value <= 128 * _MIB for value in phase_peaks.values()))
        self.assertEqual(result["mesh"]["released_renderer_attributes_bytes"],
                         self._renderer_attribute_bytes)
        for plan, row in zip(self.plans, result["rows"]):
            self.assertTrue(row["plan_binding_verified"])
            self.assertTrue(row["counterpart_binding_verified"])
            self.assertIn("counterpart_array_metadata_sha256", row)
            self.assertEqual(row["primary_source_view"]["context_id"],
                             f"template-{plan['frame_id']:04d}-{plan['template_offset_deg']:03d}")
            expected_competitor = 180 - plan["template_offset_deg"]
            self.assertEqual(row["competitor_source_view"]["context_id"],
                             f"template-{plan['frame_id']:04d}-{expected_competitor:03d}")
            self.assertEqual(row["counterpart_observed_crop_mask_metadata"]["observed_frame"],
                             self.counterparts[plan["frame_id"]]["observed_crop_mask"])
            if plan["kind"] != "self_control":
                self.assertEqual(row["counterpart_observed_crop_mask_metadata"]["synthetic_query"],
                                 self.counterparts[plan["frame_id"]]["observed_crop_mask"])
            expected_refs = (
                {"query": f"synthetic-query-{plan['frame_id']:04d}-008",
                 "template": f"template-{plan['frame_id']:04d}-{plan['template_offset_deg']:03d}",
                 "observed_frame": f"real-frame-{plan['frame_id']:04d}"}
                if plan["kind"] != "self_control" else
                {"source_template": f"template-{plan['frame_id']:04d}-000",
                 "competitor_template": f"template-{plan['frame_id']:04d}-180",
                 "observed_frame": f"real-frame-{plan['frame_id']:04d}"}
            )
            self.assertEqual(row["counterpart_context_refs"], expected_refs)
        self.assertEqual(self.inputs.file_set(), before, "adapter wrote output files")
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertLessEqual(len(encoded.encode("utf-8")), adapter.MAX_RECEIPT_BYTES)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in _walk(result)))
        self.assertFalse(result["full_28_case_screen_complete"])
        self.assertFalse(result["matching_started"])
        self.assertFalse(result["fitting_started"])
        for claim in ("accuracy", "pose", "full_28_case_screen_complete", "evaluator_truth_read"):
            self.assertFalse(result["claims"][claim])

    def test_capacity_is_explicitly_primary_gate_and_competitor_bank_only(self):
        def mixed_capacity(context_id, _kwargs):
            if context_id.endswith("-180"):
                # The 180 view has a structurally valid source-supported bank,
                # while its own W/fit/QH conjuncts fail. Those extra failures
                # must not veto rows where 180 is only the competitor.
                return {"all_gates": False, "bank_count": 1}
            return {"all_gates": True, "bank_count": 1}

        self._next_capacity_state = mixed_capacity
        with self._runtime_patches():
            result = self._run()
        rows = {row["condition_id"]: row for row in result["rows"]}
        for frame_id in _FRAMES:
            for condition_id in (
                f"syn-{frame_id}-q8-rgb-t0-rgb",
                f"zero-{frame_id}-full",
                f"zero-{frame_id}-clipped",
            ):
                row = rows[condition_id]
                self.assertEqual(row["primary_source_view"]["context_id"],
                                 f"template-{frame_id:04d}-000")
                self.assertEqual(row["competitor_source_view"]["context_id"],
                                 f"template-{frame_id:04d}-180")
                self.assertTrue(row["capacity_possible"])
                self.assertEqual(row["competitor_structural_bank"]["anchor_count"], 1)
                self.assertTrue(row["competitor_structural_bank"]["passed"])
                self.assertFalse(row["competitor_other_gates"]["witnesses_ge_8"])
                self.assertFalse(row["competitor_other_gates"]["fit_ge_24"])
                self.assertFalse(row["competitor_other_gates"]["qh_possible_cells_ge_3"])
            negative = rows[f"syn-{frame_id}-q8-rgb-t180-rgb"]
            self.assertEqual(negative["primary_source_view"]["context_id"],
                             f"template-{frame_id:04d}-180")
            self.assertFalse(negative["capacity_possible"])
            self.assertTrue(negative["primary_gates"]["bank_ge_1"])
            self.assertIn("witnesses_ge_8", negative["failed_conjuncts"])

        # A missing competitor bank is the one competitor-side veto.
        self.logs = {"loader": [], "capacity": [], "mesh": []}
        self._weak_arrays = []
        self._next_capacity_state = lambda context_id, _kwargs: (
            {"all_gates": True, "bank_count": 0}
            if context_id.endswith("-180")
            else {"all_gates": True, "bank_count": 1}
        )
        with self._runtime_patches():
            no_bank = self._run()
        zero_rows = [row for row in no_bank["rows"]
                     if row["primary_source_view"]["context_id"].endswith("-000")]
        self.assertTrue(all(not row["capacity_possible"] for row in zero_rows))
        self.assertTrue(all("competitor_source_supported_disjoint_bank_ge_1" in row["failed_conjuncts"]
                            for row in zero_rows))

    def test_each_pinned_file_hash_or_length_mismatch_precedes_decode_and_mesh(self):
        for relative in _MODEL_QUALITY_PINS:
            for same_length in (True, False):
                with self.subTest(relative=relative, same_length=same_length):
                    self.logs = {"loader": [], "capacity": [], "mesh": []}
                    self._weak_arrays = []
                    self.inputs.mutate(relative, same_length=same_length)
                    with self._runtime_patches():
                        with self.assertRaises(adapter.SourceAdapterError) as context:
                            self._run()
                    self._terminal_receipt(context)
                    self.assertEqual(self.logs["loader"], [])
                    self.assertEqual(self.logs["capacity"], [])
                    self.assertEqual(self.logs["mesh"], [])
                    # Restore the fixture payload and pin before the next case.
                    payload = _initial_payload(relative)
                    self.inputs.path_for(relative).write_bytes(payload)

    def test_residency_and_expired_deadline_fail_before_source_or_mesh_allocation(self):
        with self._runtime_patches():
            with self.assertRaises(adapter.SourceAdapterError) as cap_context:
                self._run(resident_bytes=128 * _MIB + 1)
        self._terminal_receipt(cap_context)
        self.assertEqual(self.logs["loader"], [])
        self.assertEqual(self.logs["capacity"], [])
        self.assertEqual(self.logs["mesh"], [])

        self.logs = {"loader": [], "capacity": [], "mesh": []}
        self._weak_arrays = []
        with self._runtime_patches():
            with self.assertRaises(adapter.SourceAdapterError) as deadline_context:
                self._run(deadline=time.monotonic() - 1.0)
        self._terminal_receipt(deadline_context)
        self.assertEqual(self.logs["loader"], [])
        self.assertEqual(self.logs["capacity"], [])
        self.assertEqual(self.logs["mesh"], [])

    def test_terminal_loader_and_capacity_failures_preserve_all_rows_and_completed_views(self):
        for failure_stage in ("loader", "capacity"):
            with self.subTest(failure_stage=failure_stage):
                self.logs = {"loader": [], "capacity": [], "mesh": []}
                self._weak_arrays = []

                def failing_loader(*args, **kwargs):
                    if len(self.logs["loader"]) == 1:
                        raise packet.PacketResourceLimitError("synthetic packet failure")
                    return self._loader(*args, **kwargs)

                def failing_capacity(source_arrays, **kwargs):
                    if len(self.logs["capacity"]) == 1:
                        raise RuntimeError("synthetic capacity helper failure")
                    return self._capacity(source_arrays, **kwargs)

                loader = failing_loader if failure_stage == "loader" else self._loader
                capacity = failing_capacity if failure_stage == "capacity" else self._capacity
                with self._runtime_patches(loader=loader, capacity=capacity):
                    with self.assertRaises(adapter.SourceAdapterError) as context:
                        self._run()
                receipt = self._terminal_receipt(context)
                self.assertEqual(len(receipt["source_views"]), 1)
                self.assertEqual(receipt["source_views"][0]["context_id"],
                                 "template-0010-000")
                self.assertEqual(receipt["failure"]["context_id"],
                                 "template-0010-180")
                self.assertFalse(any(
                    row["state"] != "not_evaluated_due_terminal_failure"
                    for row in receipt["rows"]
                ))

    def test_final_source_input_and_core_pin_recheck_catches_changes(self):
        for relative in (
            "diagnostics/bottle-identity-v2/capture.json",
            "inputs/ranch/input.json",
            "diagnostics/bottle-r6-parent-v2/terminal.json",
            "packets/contexts/template-0100-180.npz",
            "bench/quality_bottle_source_capacity.py",
        ):
            with self.subTest(relative=relative):
                self.logs = {"loader": [], "capacity": [], "mesh": []}
                self._weak_arrays = []

                def mutate_on_last_capacity(count, _context_id):
                    if count == 6:
                        if relative.startswith("bench/"):
                            core_path = self._synthetic_core_root.joinpath(*Path(relative).parts)
                            value = core_path.read_bytes()
                            core_path.write_bytes(bytes((value[0] ^ 1,)) + value[1:])
                        elif relative.startswith("packets/"):
                            packet_path = self.inputs.capture_root.joinpath(*relative.split("/"))
                            value = packet_path.read_bytes()
                            packet_path.write_bytes(bytes((value[0] ^ 1,)) + value[1:])
                        else:
                            self.inputs.mutate(relative, same_length=True)

                self._on_capacity = mutate_on_last_capacity
                with self._runtime_patches():
                    with self.assertRaises(adapter.SourceAdapterError) as context:
                        self._run()
                receipt = self._terminal_receipt(context)
                self.assertEqual(len(self.logs["loader"]), 6)
                self.assertEqual(len(self.logs["capacity"]), 6)
                self.assertEqual(len(receipt["source_views"]), 6)
                self.assertEqual(receipt["failure"]["stage"], "final_pin_recheck")
                if relative.startswith("bench/"):
                    core_path = self._synthetic_core_root.joinpath(*Path(relative).parts)
                    core_path.write_bytes(("synthetic-core:" + relative).encode("utf-8"))
                elif relative.startswith("packets/"):
                    packet_path = self.inputs.capture_root.joinpath(*relative.split("/"))
                    context_id = packet_path.stem
                    payload = b"synthetic-source-packet:" + context_id.encode("ascii")
                    packet_path.write_bytes(payload)
                else:
                    self.inputs.path_for(relative).write_bytes(_initial_payload(relative))
                self._on_capacity = None


def _initial_payload(relative: str) -> bytes:
    if relative.endswith("capture.json") or relative.endswith("zero_view.json"):
        return b"{}\n"
    if relative.endswith("input.json"):
        return b'{"fixture":"synthetic-r8-input"}\n'
    if relative.endswith("object.glb"):
        return _synthetic_glb()
    index = _MODEL_QUALITY_PINS.index(relative)
    return bytes((0xFF,)) + f"sealed-report-{index}".encode("ascii")


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


if __name__ == "__main__":
    unittest.main()
