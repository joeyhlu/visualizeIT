"""Independent, metadata-only tests for the R8 source-plan binder."""
from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import os
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from bench.quality_bottle_source_bindings import (
    SourceBindingError,
    bind_source_plans,
)


_FRAMES = (10, 50, 100)
_ARRAY_SHAPES = {
    "template_rgb": ([280, 280, 3], "float32"),
    "template_gray_rgb": ([280, 280, 3], "float32"),
    "template_depth_mm": ([280, 280], "float32"),
    "template_mask": ([280, 280], "uint8"),
    "source_indices": ([64], "int64"),
    "source_pixels_xy": ([64, 2], "float64"),
    "source_points_object_m": ([64, 3], "float64"),
    "observed_crop_mask": ([280, 280], "uint8"),
    "crop_k": ([3, 3], "float64"),
    "crop_from_native": ([4, 4], "float64"),
    "native_k": ([3, 3], "float64"),
    "seed_pose_m": ([4, 4], "float64"),
    "template_pose_m": ([4, 4], "float64"),
}
_SOURCE_ARRAYS = tuple(_ARRAY_SHAPES)
_COUNTERPART_ARRAYS = (
    "observed_crop_mask", "crop_k", "crop_from_native", "native_k")
def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _PoisonMapping(dict):
    """A parsed-metadata-shaped mapping that fails if forbidden fields are read."""

    def __init__(self, *args, forbidden=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._forbidden = frozenset(forbidden)

    def _check(self, key):
        if key in self._forbidden:
            raise AssertionError(f"forbidden metadata field was accessed: {key}")

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._check(key)
        return super().__contains__(key)

    def items(self):
        for key in super().keys():
            self._check(key)
        return super().items()

    def values(self):
        for key in super().keys():
            self._check(key)
        return super().values()


def _descriptor(label: str, shape=None, dtype=None):
    if shape is None or dtype is None:
        shape, dtype = [1], "uint8"
    return {"shape": list(shape), "dtype": dtype, "sha256": _sha(label)}


def _common_array_metadata(frame_id: int):
    return {
        "observed_crop_mask": _descriptor(
            f"frame-{frame_id}-observed-mask", *_ARRAY_SHAPES["observed_crop_mask"]),
        "crop_k": _descriptor(f"frame-{frame_id}-crop-k", *_ARRAY_SHAPES["crop_k"]),
        "crop_from_native": _descriptor(
            f"frame-{frame_id}-crop-from-native", *_ARRAY_SHAPES["crop_from_native"]),
        "native_k": _descriptor(f"frame-{frame_id}-native-k", *_ARRAY_SHAPES["native_k"]),
    }


def _all_template_arrays(context_id: str, common):
    result = {}
    for key, (shape, dtype) in _ARRAY_SHAPES.items():
        if key in common:
            result[key] = copy.deepcopy(common[key])
        else:
            result[key] = _descriptor(f"{context_id}-{key}", shape, dtype)
    return result


def _expected_plans():
    plans = []
    for frame_id in _FRAMES:
        plans.append(dict(
            condition_id=f"syn-{frame_id}-q8-rgb-t0-rgb", frame_id=frame_id,
            kind="synthetic_positive", template_offset_deg=0, appearance="rgb"))
        for condition in ("full", "clipped"):
            plans.append(dict(
                condition_id=f"zero-{frame_id}-{condition}", frame_id=frame_id,
                kind="self_control", template_offset_deg=0, appearance="rgb",
                self_condition=condition))
        plans.append(dict(
            condition_id=f"syn-{frame_id}-q8-rgb-t180-rgb", frame_id=frame_id,
            kind="synthetic_negative", template_offset_deg=180, appearance="rgb"))
    return plans


def _r1_index_and_seed(frame_id: int, offset: int):
    # The six selected forwards occupy these slots in the producer's frozen
    # four-frame synthetic plan, which starts with 16 combinations per frame.
    frame_slot = (10, 50, 100, 209).index(frame_id)
    forward_index = frame_slot * 16 + (2 if offset == 180 else 0)
    rng_seed = frame_id * 17 + (2 if offset == 180 else 0)
    return forward_index, rng_seed


def _context(context_id, role, frame_id, geometry_group, arrays, **metadata):
    return dict(
        context_id=context_id, role=role, frame_id=frame_id,
        geometry_group=geometry_group, arrays=arrays, **metadata)


def _source_context(frame_id: int, offset: int, common):
    context_id = f"template-{frame_id:04d}-{offset:03d}"
    arrays = _all_template_arrays(context_id, common)
    return _PoisonMapping(dict(
        context_id=context_id,
        role="template",
        frame_id=frame_id,
        geometry_group=f"frame-{frame_id}",
        template_offset_deg=offset,
        path=f"packets/contexts/{context_id}.npz",
        bytes=12345 + frame_id + offset,
        sha256=_sha(f"source-packet-{context_id}"),
        arrays=arrays,
        geometry_hashes={
            "depth_mm": arrays["template_depth_mm"]["sha256"],
            "mask": arrays["template_mask"]["sha256"],
        },
        # Producer metadata outside the source-only loader projection must be
        # ignored by this pure binder.
        appearance_ablation=_PoisonMapping({"details": object()}, forbidden={"details"}),
        fixed_patch_bank=object(), object_source_points=64,
        reference_or_annotation_inputs=False,
    ), forbidden={"appearance_ablation", "fixed_patch_bank", "object_source_points",
                  "reference_or_annotation_inputs"})


def _counterpart_context(context_id, role, frame_id, geometry_group, common,
                         *, offset=None):
    arrays = dict(common)
    array_forbidden = set()
    if role == "real_frame":
        arrays.update({
            "native_rgb": _PoisonMapping({"sha256": object()}, forbidden={"sha256"}),
            "native_mask": object(), "crop_rgb": object(), "seed_pose_m": object(),
        })
        array_forbidden.update({"native_rgb", "native_mask", "crop_rgb", "seed_pose_m"})
        extra = dict(
            native_rgb_sha256=object(), native_mask_sha256=object(),
            observed_mask_file_sha256=object(), seed_pose_sha256=object(),
            input_rgb_sha256=object(), reference_or_annotation_inputs=False,
        )
    else:
        arrays.update({
            "query_rgb_rgb": object(), "query_rgb_gray": object(),
            "query_depth_mm": object(), "query_render_mask": object(),
            "synthetic_query_mask": object(), "synthetic_native_mask": object(),
            "query_pose_native_m": object(), "known_query_pose_m": object(),
        })
        array_forbidden.update({
            "query_rgb_rgb", "query_rgb_gray", "query_depth_mm", "query_render_mask",
            "synthetic_query_mask", "synthetic_native_mask", "query_pose_native_m",
            "known_query_pose_m",
        })
        extra = dict(
            query_offset_deg=offset,
            query_appearance_ablation=_PoisonMapping(
                {"result": object()}, forbidden={"result"}),
            geometry_hashes=object(), synthetic_geometry_oracle=True,
        )
    return _PoisonMapping(
        dict(
            context_id=context_id, role=role, frame_id=frame_id,
            geometry_group=geometry_group,
            path=object(), bytes=object(), sha256=object(),
            arrays=_PoisonMapping(arrays, forbidden=array_forbidden),
            reference_pose=object(), annotation=object(), evaluation_result=object(),
            **extra,
        ),
        forbidden={"path", "bytes", "sha256", "reference_pose", "annotation",
                   "evaluation_result", "query_appearance_ablation",
                   "geometry_hashes", "native_rgb_sha256", "native_mask_sha256",
                   "observed_mask_file_sha256", "seed_pose_sha256",
                   "input_rgb_sha256"},
    )


def _forward(condition_id, index, refs, *, rng_seed=None):
    row = dict(
        condition_id=condition_id, forward_index=index, context_refs=copy.deepcopy(refs),
        path=object(), bytes=object(), sha256=object(), output_hashes=object(),
        derived_array_hashes=object(), elapsed_ms=object(), flow=object(), confidence=object(),
    )
    if rng_seed is not None:
        row["rng_seed"] = rng_seed
    return _PoisonMapping(row, forbidden={
        "path", "bytes", "sha256", "output_hashes", "derived_array_hashes",
        "elapsed_ms", "flow", "confidence",
    })


def _r1_condition(frame_id: int, offset: int, refs):
    condition_id = f"syn-{frame_id}-q8-rgb-t{offset}-rgb"
    index, seed = _r1_index_and_seed(frame_id, offset)
    return _PoisonMapping(dict(
        condition_id=condition_id, kind="synthetic", frame_id=frame_id,
        query_offset_deg=8, query_appearance="rgb", template_offset_deg=offset,
        template_appearance="rgb", rng_seed=seed, state="captured",
        forward_index=index, context_refs=copy.deepcopy(refs),
        packet_path=object(), packet_sha256=object(), output_hashes=object(),
        result=object(), score=object(), reference_pose=object(), annotation=object(),
    ), forbidden={"packet_path", "packet_sha256", "output_hashes", "result",
                  "score", "reference_pose", "annotation"})


def _r3_condition(frame_id: int, kind: str, index: int, refs):
    condition_id = f"zero-{frame_id}-{kind}"
    return _PoisonMapping(dict(
        condition_id=condition_id, frame_id=frame_id, template_offset_deg=0,
        condition=kind, appearance="rgb", state="captured", forward_index=index,
        context_refs=copy.deepcopy(refs), packet_path=object(), packet_sha256=object(),
        output_hashes=object(), input_hashes=object(), result=object(), score=object(),
        annotation=object(), reference_pose=object(),
    ), forbidden={"packet_path", "packet_sha256", "output_hashes", "input_hashes",
                  "result", "score", "annotation", "reference_pose"})


def _fixture():
    contexts = []
    r1_conditions = []
    r1_forwards = []
    r3_conditions = []
    r3_forwards = []
    common_by_frame = {}
    for frame_index, frame_id in enumerate(_FRAMES):
        geometry_group = f"frame-{frame_id}"
        common = _common_array_metadata(frame_id)
        common_by_frame[frame_id] = common
        real_id = f"real-frame-{frame_id:04d}"
        contexts.append(_counterpart_context(
            real_id, "real_frame", frame_id, geometry_group, common))
        for offset in (0, 180):
            contexts.append(_source_context(frame_id, offset, common))
        query_id = f"synthetic-query-{frame_id:04d}-008"
        contexts.append(_counterpart_context(
            query_id, "synthetic_query", frame_id, geometry_group, common, offset=8))
        for offset in (0, 180):
            condition_id = f"syn-{frame_id}-q8-rgb-t{offset}-rgb"
            refs = dict(query=query_id,
                        template=f"template-{frame_id:04d}-{offset:03d}",
                        observed_frame=real_id)
            condition = _r1_condition(frame_id, offset, refs)
            r1_conditions.append(condition)
            r1_forwards.append(_forward(
                condition_id, _r1_index_and_seed(frame_id, offset)[0], refs,
                rng_seed=_r1_index_and_seed(frame_id, offset)[1]))
        for condition_offset, kind in enumerate(("full", "clipped")):
            condition_id = f"zero-{frame_id}-{kind}"
            refs = dict(source_template=f"template-{frame_id:04d}-000",
                        competitor_template=f"template-{frame_id:04d}-180",
                        observed_frame=real_id)
            index = frame_index * 2 + condition_offset
            r3_conditions.append(_r3_condition(frame_id, kind, index, refs))
            r3_forwards.append(_forward(condition_id, index, refs))

    # Make the exact closed plan explicit without importing the R5 evaluator.
    expected = _expected_plans()
    capture = _PoisonMapping(dict(
        contexts=contexts, conditions=r1_conditions, forwards=r1_forwards,
        reference_or_annotations_loaded=False,
        query_reference_values=object(), evaluation_rows=object(), scores=object(),
    ), forbidden={"query_reference_values", "evaluation_rows", "scores"})
    zero = _PoisonMapping(dict(
        conditions=r3_conditions, forwards=r3_forwards,
        zero_view_scores=object(), results=object(), annotations=object(),
    ), forbidden={"zero_view_scores", "results", "annotations"})
    return capture, zero, expected, common_by_frame


def _bind(capture, zero, expected):
    return bind_source_plans(capture, zero, expected)


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.capture, self.zero, self.plans, self.common = _fixture()

    def test_binds_exact_twelve_rows_and_six_view_namespaces(self):
        result = _bind(self.capture, self.zero, self.plans)
        expected_ids = [row["condition_id"] for row in self.plans]
        self.assertEqual(set(result), {
            "schema_version", "metadata_bindings_verified",
            "counterpart_context_bindings_verified", "source_packet_file_pins_verified",
            "source_packet_members_verified", "source_packet_arrays_decoded",
            "mesh_pin_verified", "capacity_screen_complete", "matching_started",
            "fitting_started", "template_bindings", "plan_bindings", "counts",
        })
        self.assertEqual(result["schema_version"], "r8-source-metadata-bindings-v1")
        self.assertEqual(result["counts"], {"template_bindings": 6, "plan_bindings": 12})
        self.assertEqual([row["condition_id"] for row in result["plan_bindings"]], expected_ids)
        expected_templates = [
            f"template-{frame_id:04d}-{offset:03d}"
            for frame_id in _FRAMES for offset in (0, 180)
        ]
        self.assertEqual([row["context_id"] for row in result["template_bindings"]],
                         expected_templates)
        namespaces = [row["source_namespace"] for row in result["template_bindings"]]
        self.assertEqual(len({(row["context_id"], row["source_rgb_sha256"])
                              for row in namespaces}), 6)
        for row in result["template_bindings"]:
            self.assertEqual(set(row), {
                "context_id", "frame_id", "template_offset_deg", "geometry_group",
                "source_namespace", "array_metadata", "source_entry",
            })
            self.assertEqual(row["source_namespace"]["context_id"], row["context_id"])
            self.assertEqual(row["source_namespace"]["source_rgb_sha256"],
                             row["source_entry"]["arrays"]["template_rgb"]["sha256"])
            self.assertEqual(set(row["array_metadata"]), set(_COUNTERPART_ARRAYS))
            for desc in row["array_metadata"].values():
                self.assertEqual(set(desc), {"shape", "dtype", "sha256"})

    def test_source_entry_is_loader_compatible_and_metadata_only(self):
        result = _bind(self.capture, self.zero, self.plans)
        required_source_fields = {
            "context_id", "role", "frame_id", "template_offset_deg", "path",
            "bytes", "sha256", "arrays", "geometry_hashes",
        }
        for binding in result["template_bindings"]:
            source = binding["source_entry"]
            self.assertEqual(set(source), required_source_fields)
            self.assertEqual(source["context_id"], binding["context_id"])
            self.assertEqual(source["role"], "template")
            self.assertEqual(source["frame_id"], binding["frame_id"])
            self.assertEqual(source["template_offset_deg"], binding["template_offset_deg"])
            self.assertEqual(source["path"],
                             f"packets/contexts/{binding['context_id']}.npz")
            self.assertGreater(source["bytes"], 0)
            self.assertEqual(len(source["sha256"]), 64)
            self.assertEqual(set(source["arrays"]), set(_SOURCE_ARRAYS))
            for name, desc in source["arrays"].items():
                self.assertEqual(set(desc), {"shape", "dtype", "sha256"}, name)
                self.assertEqual(desc["shape"], _ARRAY_SHAPES[name][0], name)
                self.assertEqual(desc["dtype"], _ARRAY_SHAPES[name][1], name)
            self.assertEqual(set(source["geometry_hashes"]), {"depth_mm", "mask"})
            self.assertEqual(binding["array_metadata"], {
                name: source["arrays"][name] for name in _COUNTERPART_ARRAYS
            })

    def test_plan_bindings_keep_producer_refs_indices_rng_and_counterpart_hashes(self):
        result = _bind(self.capture, self.zero, self.plans)
        rows = result["plan_bindings"]
        required = {
            "condition_id", "kind", "frame_id", "template_offset_deg",
            "template_context_id", "observed_frame_context_id",
            "synthetic_query_context_id", "geometry_group", "source_namespace",
            "forward_index", "rng_seed", "self_condition", "context_refs",
            "counterpart_mask_metadata", "counterpart_array_metadata_sha256",
            "plan_binding_verified", "counterpart_binding_verified",
        }
        self.assertEqual(len(rows), 12)
        for plan, row in zip(self.plans, rows):
            self.assertEqual(set(row), required)
            self.assertEqual(row["condition_id"], plan["condition_id"])
            self.assertEqual(row["kind"], plan["kind"])
            self.assertEqual(row["frame_id"], plan["frame_id"])
            self.assertEqual(row["template_offset_deg"], plan["template_offset_deg"])
            self.assertEqual(row["template_context_id"],
                             f"template-{plan['frame_id']:04d}-{plan['template_offset_deg']:03d}")
            self.assertEqual(row["observed_frame_context_id"],
                             f"real-frame-{plan['frame_id']:04d}")
            self.assertTrue(row["plan_binding_verified"])
            self.assertTrue(row["counterpart_binding_verified"])
            self.assertEqual(row["geometry_group"], f"frame-{plan['frame_id']}")
            if plan["kind"].startswith("synthetic"):
                offset = plan["template_offset_deg"]
                expected_index, expected_seed = _r1_index_and_seed(plan["frame_id"], offset)
                query_id = f"synthetic-query-{plan['frame_id']:04d}-008"
                expected_refs = dict(
                    query=query_id,
                    template=row["template_context_id"],
                    observed_frame=row["observed_frame_context_id"],
                )
                self.assertEqual(row["forward_index"], expected_index)
                self.assertEqual(row["rng_seed"], expected_seed)
                self.assertEqual(row["synthetic_query_context_id"], query_id)
                self.assertIsNone(row["self_condition"])
                self.assertEqual(row["context_refs"], expected_refs)
                counterpart_names = {"observed_frame", "synthetic_query"}
                self.assertEqual(set(row["counterpart_mask_metadata"]), counterpart_names)
                self.assertEqual(set(row["counterpart_array_metadata_sha256"]),
                                 counterpart_names)
            else:
                condition_offset = 0 if plan["self_condition"] == "full" else 1
                expected_index = _FRAMES.index(plan["frame_id"]) * 2 + condition_offset
                expected_refs = dict(
                    source_template=f"template-{plan['frame_id']:04d}-000",
                    competitor_template=f"template-{plan['frame_id']:04d}-180",
                    observed_frame=row["observed_frame_context_id"],
                )
                self.assertEqual(row["forward_index"], expected_index)
                self.assertIsNone(row["rng_seed"])
                self.assertIsNone(row["synthetic_query_context_id"])
                self.assertEqual(row["self_condition"], plan["self_condition"])
                self.assertEqual(row["context_refs"], expected_refs)
                self.assertEqual(set(row["counterpart_mask_metadata"]), {"observed_frame"})
                self.assertEqual(set(row["counterpart_array_metadata_sha256"]),
                                 {"observed_frame"})
            for name, descriptor in row["counterpart_mask_metadata"].items():
                self.assertEqual(set(descriptor), {"shape", "dtype", "sha256"}, name)
                self.assertEqual(descriptor,
                                 self.common[plan["frame_id"]]["observed_crop_mask"], name)
            for digest in row["counterpart_array_metadata_sha256"].values():
                self.assertEqual(len(digest), 64)
            counterpart_names = row["counterpart_array_metadata_sha256"]
            for counterpart_name in counterpart_names:
                descriptors = {
                    name: self.common[plan["frame_id"]][name]
                    for name in _COUNTERPART_ARRAYS
                }
                expected_digest = hashlib.sha256(json.dumps(
                    descriptors, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
                self.assertEqual(counterpart_names[counterpart_name], expected_digest)

    def test_receipt_is_compact_json_and_has_no_false_capability_claims(self):
        result = _bind(self.capture, self.zero, self.plans)
        false_claims = {
            "source_packet_file_pins_verified", "source_packet_members_verified",
            "source_packet_arrays_decoded", "mesh_pin_verified",
            "capacity_screen_complete", "matching_started", "fitting_started",
        }
        self.assertTrue(false_claims.issubset(result))
        for name in false_claims:
            self.assertIs(result[name], False, name)
        self.assertIs(result["metadata_bindings_verified"], True)
        self.assertIs(result["counterpart_context_bindings_verified"], True)
        payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertLessEqual(len(payload.encode("utf-8")), 64 * 1024)
        round_trip = json.loads(payload)
        self.assertEqual(round_trip, result)

        def check_json_tree(value):
            self.assertNotIsInstance(value, (bytes, bytearray, memoryview))
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertIsInstance(key, str)
                    check_json_tree(child)
            elif isinstance(value, list):
                for child in value:
                    check_json_tree(child)
            else:
                self.assertIsNone(value) if value is None else self.assertIsInstance(
                    value, (str, int, float, bool))
        check_json_tree(result)

    def test_pure_mapping_call_does_not_open_files_npz_or_import_runtime_tools(self):
        original_import = builtins.__import__
        forbidden_imports = (
            "numpy", "torch", "cv2", "trimesh", "pyrender",
            "bench.quality_bottle_source_packet", "bench.quality_bottle_source_capacity",
            "bench.quality_bottle_source_reservation", "bench.quality_bottle_reservation_selection",
            "bench.quality_bottle_joint_prerequisite", "bench.quality_bottle_pose_ablation",
            "bench.quality_bottle_identity_audit", "bench.glb_model",
        )

        def guarded_import(name, *args, **kwargs):
            if any(name == prefix or name.startswith(prefix + ".")
                   for prefix in forbidden_imports):
                raise AssertionError(f"forbidden runtime import: {name}")
            return original_import(name, *args, **kwargs)

        failure = AssertionError("metadata binder attempted file/NPZ access")
        with mock.patch("builtins.__import__", side_effect=guarded_import), \
                mock.patch("builtins.open", side_effect=failure), \
                mock.patch.object(Path, "open", side_effect=failure), \
                mock.patch.object(io, "open", side_effect=failure), \
                mock.patch.object(os, "open", side_effect=failure), \
                mock.patch.object(zipfile, "ZipFile", side_effect=failure):
            result = _bind(self.capture, self.zero, self.plans)
        self.assertEqual(len(result["plan_bindings"]), 12)

    def test_plan_order_missing_duplicate_and_extra_rows_are_rejected(self):
        variants = []
        reordered = copy.deepcopy(self.plans)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        variants.append(reordered)
        variants.append(copy.deepcopy(self.plans[:-1]))
        duplicated = copy.deepcopy(self.plans)
        duplicated[-1] = copy.deepcopy(duplicated[0])
        variants.append(duplicated)
        extra = copy.deepcopy(self.plans) + [copy.deepcopy(self.plans[0])]
        variants.append(extra)
        for plans in variants:
            with self.subTest(count=len(plans), first=plans[0]["condition_id"]):
                with self.assertRaises(SourceBindingError):
                    _bind(self.capture, self.zero, plans)

    def test_exact_plan_mappings_survive_sorted_key_json_round_trip(self):
        parsed = json.loads(json.dumps(self.plans, sort_keys=True))
        result = _bind(self.capture, self.zero, parsed)
        self.assertEqual(
            [row["condition_id"] for row in result["plan_bindings"]],
            [row["condition_id"] for row in self.plans],
        )

    def test_missing_duplicate_or_mismatched_condition_and_forward_metadata_rejected(self):
        capture, zero, plans, _ = _fixture()
        del capture["conditions"][0]
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        capture, zero, plans, common = _fixture()
        capture["conditions"].append(dict(capture["conditions"][0]))
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        capture, zero, plans, common = _fixture()
        capture["forwards"][0] = dict(capture["forwards"][0])
        capture["forwards"][0]["condition_id"] = "unexpected-condition"
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        capture, zero, plans, common = _fixture()
        capture["forwards"][0] = dict(capture["forwards"][0])
        capture["forwards"][0]["forward_index"] += 1
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

    def test_r1_state_refs_seed_index_and_r3_self_metadata_are_verified(self):
        mutations = (
            ("state", "failed"),
            ("query_offset_deg", 9),
            ("query_appearance", "gray"),
            ("template_offset_deg", 180),
            ("template_appearance", "gray"),
            ("forward_index", 999),
            ("rng_seed", 999),
        )
        for key, value in mutations:
            capture, zero, plans, _ = _fixture()
            capture["conditions"][0] = dict(capture["conditions"][0])
            capture["conditions"][0][key] = value
            with self.subTest(record="r1 condition", field=key):
                with self.assertRaises(SourceBindingError):
                    _bind(capture, zero, plans)

        capture, zero, plans, _ = _fixture()
        capture["forwards"][0] = dict(capture["forwards"][0])
        capture["forwards"][0]["rng_seed"] += 1
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        capture, zero, plans, _ = _fixture()
        capture["conditions"][0] = dict(capture["conditions"][0])
        capture["conditions"][0]["context_refs"] = dict(
            capture["conditions"][0]["context_refs"], template="wrong-template")
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        for field, value in (("state", "failed"), ("condition", "clipped"),
                             ("frame_id", 50), ("forward_index", 99)):
            capture, zero, plans, _ = _fixture()
            zero["conditions"][0] = dict(zero["conditions"][0])
            zero["conditions"][0][field] = value
            with self.subTest(record="r3 condition", field=field):
                with self.assertRaises(SourceBindingError):
                    _bind(capture, zero, plans)

        capture, zero, plans, _ = _fixture()
        zero["forwards"][0] = dict(zero["forwards"][0])
        zero["forwards"][0]["context_refs"] = dict(
            zero["forwards"][0]["context_refs"], competitor_template="wrong")
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

    def test_context_role_frame_offset_geometry_and_counterpart_arrays_are_checked(self):
        cases = (
            ("role", "real_frame"),
            ("frame_id", 50),
            ("template_offset_deg", 180),
            ("geometry_group", "other-frame"),
        )
        for key, value in cases:
            capture, zero, plans, _ = _fixture()
            context_index = next(i for i, row in enumerate(capture["contexts"])
                                 if row["context_id"] == "template-0010-000")
            capture["contexts"][context_index] = dict(capture["contexts"][context_index])
            capture["contexts"][context_index][key] = value
            with self.subTest(field=key):
                with self.assertRaises(SourceBindingError):
                    _bind(capture, zero, plans)

        for counterpart_role, counterpart_id in (
                ("real_frame", "real-frame-0010"),
                ("synthetic_query", "synthetic-query-0010-008")):
            for array_name in _COUNTERPART_ARRAYS:
                for metadata_field, changed in (
                        ("shape", [279, 280]),
                        ("dtype", "float32"),
                        ("sha256", _sha("mismatched-camera-or-mask"))):
                    capture, zero, plans, _ = _fixture()
                    context_index = next(i for i, row in enumerate(capture["contexts"])
                                         if row["context_id"] == counterpart_id)
                    entry = capture["contexts"][context_index]
                    arrays = dict(entry["arrays"])
                    arrays[array_name] = dict(arrays[array_name])
                    arrays[array_name][metadata_field] = changed
                    replacement = dict(entry)
                    replacement["arrays"] = arrays
                    capture["contexts"][context_index] = replacement
                    with self.subTest(role=counterpart_role, array=array_name,
                                      metadata=metadata_field):
                        with self.assertRaises(SourceBindingError):
                            _bind(capture, zero, plans)

    def test_missing_duplicate_contexts_and_malformed_source_descriptors_rejected(self):
        capture, zero, plans, _ = _fixture()
        capture["contexts"] = [row for row in capture["contexts"]
                               if row["context_id"] != "synthetic-query-0010-008"]
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        capture, zero, plans, _ = _fixture()
        capture["contexts"].append(dict(capture["contexts"][1]))
        with self.assertRaises(SourceBindingError):
            _bind(capture, zero, plans)

        for mutate in (
            lambda arrays: arrays["crop_k"].update(extra="unexpected"),
            lambda arrays: arrays["template_rgb"].update(sha256="bad"),
            lambda arrays: arrays.pop("template_pose_m"),
        ):
            capture, zero, plans, _ = _fixture()
            index = next(i for i, row in enumerate(capture["contexts"])
                         if row["context_id"] == "template-0010-000")
            entry = dict(capture["contexts"][index])
            arrays = copy.deepcopy(entry["arrays"])
            mutate(arrays)
            entry["arrays"] = arrays
            capture["contexts"][index] = entry
            with self.assertRaises(SourceBindingError):
                _bind(capture, zero, plans)


if __name__ == "__main__":
    unittest.main()
