"""Independent, source-only regressions for frozen R8 template NPZ preflight."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

from . import quality_bottle_source_packet as packet


SOURCE_KEYS = (
    "template_rgb", "template_gray_rgb", "template_depth_mm", "template_mask",
    "source_indices", "source_pixels_xy", "source_points_object_m",
    "observed_crop_mask", "crop_k", "crop_from_native", "native_k",
    "seed_pose_m", "template_pose_m",
)
SPEC_V2_SHA256 = "32758DDA6AD550A5EDD94F5354B7E8401FF206A63B884E8C88AB37F71EE6D9CB"
FIXED_SCHEMA = {
    "template_rgb": ((280, 280, 3), "float32", 4),
    "template_gray_rgb": ((280, 280, 3), "float32", 4),
    "template_depth_mm": ((280, 280), "float32", 4),
    "template_mask": ((280, 280), "uint8", 1),
    "observed_crop_mask": ((280, 280), "uint8", 1),
    "crop_k": ((3, 3), "float64", 8),
    "crop_from_native": ((4, 4), "float64", 8),
    "native_k": ((3, 3), "float64", 8),
    "seed_pose_m": ((4, 4), "float64", 8),
    "template_pose_m": ((4, 4), "float64", 8),
    "source_indices": ((2,), "int64", 8),
    "source_pixels_xy": ((2, 2), "float64", 8),
    "source_points_object_m": ((2, 3), "float64", 8),
}


def _array_metadata():
    result = {}
    for key, (shape, dtype, itemsize) in FIXED_SCHEMA.items():
        raw_bytes = 1
        for dimension in shape:
            raw_bytes *= dimension
        raw_bytes *= itemsize
        result[key] = {
            "shape": list(shape),
            "dtype": dtype,
            "sha256": hashlib.sha256((key + ":synthetic-source-array").encode()).hexdigest(),
            "_raw_bytes": raw_bytes,
        }
    return result


class TemplatePacketPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.capture_root = Path(self.temporary.name)
        self.context_id = "template-0010-000"
        self.relative_path = f"packets/contexts/{self.context_id}.npz"
        self.packet_path = self.capture_root.joinpath(*self.relative_path.split("/"))
        self.packet_path.parent.mkdir(parents=True)
        self.metadata = _array_metadata()
        self.entry = {
            "context_id": self.context_id,
            "role": "template",
            "frame_id": 10,
            "template_offset_deg": 0,
            "path": self.relative_path,
            "arrays": self._public_metadata(),
            "geometry_hashes": {
                "depth_mm": self.metadata["template_depth_mm"]["sha256"],
                "mask": self.metadata["template_mask"]["sha256"],
            },
        }
        self._write_archive([(f"{key}.npy", self.metadata[key]["_raw_bytes"] + 128)
                             for key in SOURCE_KEYS])

    def _public_metadata(self):
        return {
            key: {field: value for field, value in record.items() if field != "_raw_bytes"}
            for key, record in self.metadata.items()
        }

    def _write_archive(self, members):
        with zipfile.ZipFile(self.packet_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for member in members:
                name, size = member[:2]
                info = zipfile.ZipInfo(name)
                if len(member) > 2 and member[2] == "symlink":
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b"x" * size)
        self.entry["bytes"] = self.packet_path.stat().st_size
        self.entry["sha256"] = hashlib.sha256(self.packet_path.read_bytes()).hexdigest()

    def _preflight(self, entry=None, **kwargs):
        return packet.preflight_template_packet(
            self.capture_root,
            self.entry if entry is None else entry,
            expected_context_id=self.context_id,
            expected_frame_id=10,
            expected_offset_deg=0,
            **kwargs,
        )

    def test_valid_thirteen_member_receipt_uses_metadata_only(self):
        with mock.patch.object(zipfile.ZipFile, "open",
                               side_effect=AssertionError("NPZ payload member was opened")) as opened:
            receipt = self._preflight()

        self.assertEqual(receipt["state"], "validated_source_packet")
        self.assertEqual(receipt["context_id"], self.context_id)
        self.assertEqual(receipt["packet"]["relative_path"], self.relative_path)
        self.assertEqual(receipt["archive"]["member_count"], 13)
        self.assertFalse(receipt["archive"]["payloads_decoded"])
        self.assertFalse(receipt["decode_authorized"])
        self.assertEqual(set(receipt["array_metadata"]), set(SOURCE_KEYS))
        self.assertEqual(opened.call_count, 0)
        self.assertTrue(receipt["budget"]["peak_estimate_bytes"] < packet.MAX_WORKING_BYTES)

    def test_dynamic_source_arrays_reject_wrong_trailing_dimensions(self):
        for key, shape in (("source_indices", [2, 1]),
                           ("source_pixels_xy", [2, 4]),
                           ("source_points_object_m", [2, 2])):
            with self.subTest(key=key, shape=shape):
                altered = copy.deepcopy(self.entry)
                altered["arrays"][key]["shape"] = shape
                with self.assertRaises(packet.PacketPreflightError):
                    self._preflight(altered)

        mismatched_counts = copy.deepcopy(self.entry)
        mismatched_counts["arrays"]["source_indices"]["shape"] = [3]
        with self.assertRaises(packet.PacketPreflightError):
            self._preflight(mismatched_counts)

    def test_context_role_frame_and_offset_mismatches_fail_closed(self):
        mutations = (
            ("context_id", "template-0010-180"),
            ("role", "real_frame"),
            ("frame_id", 50),
            ("template_offset_deg", 180),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                altered = copy.deepcopy(self.entry)
                altered[field] = value
                with self.assertRaises(packet.PacketPreflightError):
                    self._preflight(altered)

    def test_extra_zip_member_and_wrong_packet_hash_are_rejected(self):
        with self.subTest(case="extra member count"):
            members = [(f"{key}.npy", self.metadata[key]["_raw_bytes"] + 128)
                       for key in SOURCE_KEYS]
            members.append(("unexpected.npy", 256))
            self._write_archive(members)
            with self.assertRaises(packet.PacketPreflightError):
                self._preflight()

        with self.subTest(case="wrong SHA256"):
            self._write_archive([(f"{key}.npy", self.metadata[key]["_raw_bytes"] + 128)
                                 for key in SOURCE_KEYS])
            altered = copy.deepcopy(self.entry)
            altered["sha256"] = "0" * 64
            with self.assertRaises(packet.PacketPreflightError):
                self._preflight(altered)

    def test_counterpart_observed_mask_metadata_is_compared_without_opening_packets(self):
        source_mask = copy.deepcopy(self.entry["arrays"]["observed_crop_mask"])
        matching = {
            "real-frame": copy.deepcopy(source_mask),
            "synthetic-query": copy.deepcopy(source_mask),
        }
        receipt = self._preflight(counterpart_mask_metadata=matching)
        self.assertEqual(receipt["counterpart_mask_checks"], [
            {"label": "real-frame", "matches_source_observed_crop_mask": True},
            {"label": "synthetic-query", "matches_source_observed_crop_mask": True},
        ])

        mismatch = copy.deepcopy(matching)
        mismatch["synthetic-query"]["sha256"] = "f" * 64
        with mock.patch.object(packet, "_hash_and_list_members",
                               side_effect=AssertionError("counterpart mismatch must fail before packet I/O")):
            with self.assertRaises(packet.PacketPreflightError):
                self._preflight(counterpart_mask_metadata=mismatch)

    def test_object_dtype_private_arrays_and_private_entry_fields_are_handled_safely(self):
        object_dtype = copy.deepcopy(self.entry)
        object_dtype["arrays"]["template_rgb"]["dtype"] = "object"
        with self.assertRaises(packet.PacketPreflightError):
            self._preflight(object_dtype)

        private_array = copy.deepcopy(self.entry)
        private_array["arrays"]["query_rgb"] = {
            "shape": [280, 280, 3], "dtype": "float32", "sha256": "a" * 64,
        }
        with self.assertRaises(packet.PacketPreflightError):
            self._preflight(private_array)

        class PoisonEntry(dict):
            _FORBIDDEN = {"query_rgb", "query_depth_mm", "evaluator_truth",
                          "annotations", "reference_pose"}

            def get(self, key, default=None):
                if key in self._FORBIDDEN:
                    raise AssertionError(f"preflight read forbidden source field {key}")
                return super().get(key, default)

        poison = PoisonEntry(self.entry)
        poison.update({key: object() for key in PoisonEntry._FORBIDDEN})
        receipt = self._preflight(poison)
        serialized = repr(receipt).lower()
        for forbidden in PoisonEntry._FORBIDDEN:
            self.assertNotIn(forbidden, serialized)

    def test_zip_duplicate_pathlike_symlink_and_oversized_member_names_fail_closed(self):
        base_members = [(f"{key}.npy", self.metadata[key]["_raw_bytes"] + 128)
                        for key in SOURCE_KEYS]
        cases = {}

        duplicate = list(base_members)
        duplicate[-1] = duplicate[0]
        cases["duplicate name"] = duplicate

        pathlike = list(base_members)
        pathlike[0] = ("../template_rgb.npy", pathlike[0][1])
        cases["path-like member"] = pathlike

        linked = list(base_members)
        linked[0] = (linked[0][0], linked[0][1], "symlink")
        cases["symlink member"] = linked

        oversized = list(base_members)
        oversized[0] = (oversized[0][0],
                        self.metadata["template_rgb"]["_raw_bytes"] +
                        packet.NPY_HEADER_ALLOWANCE_BYTES + 1)
        cases["oversized member"] = oversized

        for label, members in cases.items():
            with self.subTest(case=label):
                self._write_archive(members)
                with mock.patch.object(zipfile.ZipFile, "open",
                                       side_effect=AssertionError("preflight must not read members")):
                    with self.assertRaises(packet.PacketPreflightError):
                        self._preflight()

    def test_packet_path_traversal_and_symlink_are_rejected(self):
        traversal = copy.deepcopy(self.entry)
        traversal["path"] = "../escape.npz"
        with mock.patch.object(packet, "_hash_and_list_members",
                               side_effect=AssertionError("unsafe path must fail before opening")):
            with self.assertRaises(packet.PacketPreflightError):
                self._preflight(traversal)

        external = self.capture_root / "outside.npz"
        external.write_bytes(self.packet_path.read_bytes())
        self.packet_path.unlink()
        try:
            self.packet_path.symlink_to(external)
        except (OSError, NotImplementedError):
            original_lstat = Path.lstat

            def report_symlink(path, *args, **kwargs):
                if path == self.packet_path:
                    return SimpleNamespace(st_mode=stat.S_IFLNK)
                return original_lstat(path, *args, **kwargs)

            with mock.patch.object(Path, "lstat", report_symlink):
                with self.assertRaises(packet.PacketPreflightError):
                    self._preflight()
        else:
            with self.assertRaises(packet.PacketPreflightError):
                self._preflight()

    def test_packet_byte_pin_and_in_place_change_during_hashing_are_rejected(self):
        wrong_bytes = copy.deepcopy(self.entry)
        wrong_bytes["bytes"] += 1
        with self.assertRaises(packet.PacketPreflightError):
            self._preflight(wrong_bytes)

        original_check = packet._check_deadline
        changed = False

        def append_during_hash(deadline, label):
            nonlocal changed
            if label == "packet hashing" and not changed:
                changed = True
                with self.packet_path.open("ab") as stream:
                    stream.write(b"concurrent-change")
            return original_check(deadline, label)

        with mock.patch.object(packet, "_check_deadline", side_effect=append_during_hash):
            with self.assertRaises(packet.PacketPreflightError):
                self._preflight()
        self.assertTrue(changed)

    def test_resource_cap_rejects_before_hashing_or_zip_directory_allocation(self):
        with (mock.patch.object(packet, "_hash_and_list_members",
                                side_effect=AssertionError("over-cap packet must fail before hashing")) as hash_spy,
              mock.patch.object(zipfile.ZipFile, "infolist",
                                side_effect=AssertionError("over-cap packet must not allocate ZipInfo list")) as list_spy):
            with self.assertRaises(packet.PacketResourceLimitError):
                self._preflight(resident_bytes=packet.MAX_WORKING_BYTES - 1)
        self.assertEqual(hash_spy.call_count, 0)
        self.assertEqual(list_spy.call_count, 0)

    def test_malformed_expired_and_clamped_deadlines(self):
        for deadline in (True, math.nan, math.inf, "later"):
            with self.subTest(deadline=deadline):
                with self.assertRaises(packet.PacketPreflightError):
                    self._preflight(deadline_monotonic=deadline)
        with mock.patch.object(packet.time, "monotonic", return_value=100.0):
            with self.assertRaises(packet.PacketResourceLimitError):
                self._preflight(deadline_monotonic=99.0)

        with mock.patch.object(packet.time, "monotonic", return_value=100.0):
            receipt = self._preflight(deadline_monotonic=10**12)
        self.assertEqual(receipt["budget"]["deadline_seconds_remaining"],
                         packet.MAX_WALL_SECONDS)

    def _load_with_stub(self, stub, *, real_mask=None, synthetic_mask=None,
                        require_synthetic=False, deadline=None, array_info=None):
        from . import quality_bottle_pose_ablation as r5

        declared = {
            key: {field: value for field, value in record.items() if field != "_raw_bytes"}
            for key, record in self.metadata.items()
        }
        sentinels = {key: object() for key in SOURCE_KEYS}
        info_by_identity = {id(sentinels[key]): declared[key] for key in SOURCE_KEYS}
        if array_info is None:
            array_info = lambda value: info_by_identity[id(value)]
        real_mask = real_mask or declared["observed_crop_mask"]
        kwargs = {
            "expected_context_id": self.context_id,
            "expected_frame_id": 10,
            "expected_offset_deg": 0,
            "real_frame_mask_metadata": real_mask,
            "require_synthetic_mask": require_synthetic,
        }
        if synthetic_mask is not None:
            kwargs["synthetic_mask_metadata"] = synthetic_mask
        if deadline is not None:
            kwargs["deadline_monotonic"] = deadline
        with (mock.patch.object(r5, "_condition_source_only_inputs", side_effect=stub) as loader_spy,
              mock.patch.object(r5.audit, "array_info", side_effect=array_info),
              mock.patch.object(np, "load", side_effect=AssertionError("NPZ must not be decoded here")) as load_spy,
              mock.patch.object(zipfile.ZipFile, "open",
                                side_effect=AssertionError("NPZ member payload must not be opened")) as open_spy):
            result = packet.load_source_template(self.capture_root, self.entry, **kwargs)
        return result, sentinels, loader_spy, load_spy, open_spy, declared

    def test_loader_uses_minimal_source_bundle_and_returns_exact_projection(self):
        from . import quality_bottle_pose_ablation as r5

        calls = []
        declared_entry_keys = {
            "context_id", "role", "frame_id", "template_offset_deg", "path",
            "bytes", "sha256", "arrays", "geometry_hashes",
        }

        def source_only_loader(bundle, plan):
            calls.append((bundle, plan))
            self.assertEqual(set(bundle), {"capture_root", "r1_contexts"})
            self.assertEqual(bundle["capture_root"], self.capture_root.resolve())
            self.assertEqual(set(bundle["r1_contexts"]), {self.context_id})
            sanitized = bundle["r1_contexts"][self.context_id]
            self.assertEqual(set(sanitized), declared_entry_keys)
            self.assertEqual(sanitized["context_id"], self.context_id)
            self.assertEqual(sanitized["role"], "template")
            self.assertEqual(sanitized["arrays"].keys(), set(SOURCE_KEYS))
            self.assertEqual(set(plan), {"frame_id", "template_offset_deg", "condition_id"})
            self.assertEqual(plan["condition_id"], f"source-packet-preflight:{self.context_id}")
            return sanitized, sentinels

        sentinels = {key: object() for key in SOURCE_KEYS}
        declared = {key: {field: value for field, value in rec.items() if field != "_raw_bytes"}
                    for key, rec in self.metadata.items()}
        info_by_identity = {id(sentinels[key]): declared[key] for key in SOURCE_KEYS}
        with (mock.patch.object(r5, "_condition_source_only_inputs", side_effect=source_only_loader) as loader_spy,
              mock.patch.object(r5.audit, "array_info", side_effect=lambda value: info_by_identity[id(value)]),
              mock.patch.object(np, "load", side_effect=AssertionError("NPZ must not be decoded here")) as load_spy,
              mock.patch.object(zipfile.ZipFile, "open",
                                side_effect=AssertionError("NPZ member payload must not be opened")) as open_spy):
            result = packet.load_source_template(
                self.capture_root, self.entry,
                expected_context_id=self.context_id,
                expected_frame_id=10,
                expected_offset_deg=0,
                real_frame_mask_metadata=declared["observed_crop_mask"],
            )

        self.assertEqual(loader_spy.call_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(result), {"arrays", "receipt", "image_namespace"})
        projected = set(SOURCE_KEYS) - {"template_gray_rgb", "seed_pose_m"}
        self.assertEqual(set(result["arrays"]), projected)
        for key in projected:
            self.assertIs(result["arrays"][key], sentinels[key])
        self.assertEqual(result["image_namespace"], {
            "context_id": self.context_id,
            "source_rgb_sha256": declared["template_rgb"]["sha256"],
        })
        receipt = result["receipt"]
        self.assertEqual(receipt["r5_source_only_loader_calls"], 1)
        self.assertTrue(receipt["packet_sha256_rechecked"])
        self.assertFalse(receipt["query_or_evaluator_data_read"])
        self.assertFalse(receipt["fitting_or_matching_started"])
        self.assertFalse(receipt["capacity_screen_complete"])
        self.assertEqual(load_spy.call_count, 0)
        self.assertEqual(open_spy.call_count, 0)

    def test_loader_requires_matching_real_and_plan_required_synthetic_masks_before_r5(self):
        from . import quality_bottle_pose_ablation as r5

        declared_mask = self.entry["arrays"]["observed_crop_mask"]
        different = copy.deepcopy(declared_mask)
        different["sha256"] = "f" * 64
        with mock.patch.object(r5, "_condition_source_only_inputs",
                               side_effect=AssertionError("mask mismatch must precede R5")) as spy:
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0, real_frame_mask_metadata=different)
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0, real_frame_mask_metadata=declared_mask,
                    require_synthetic_mask=True)
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0, real_frame_mask_metadata=declared_mask,
                    synthetic_mask_metadata=different, require_synthetic_mask=True)
        self.assertEqual(spy.call_count, 0)

    def test_loader_rejects_in_place_entry_mutation_and_nonidentical_return_before_array_checks(self):
        from . import quality_bottle_pose_ablation as r5

        def mutate_same_entry(bundle, plan):
            entry = bundle["r1_contexts"][self.context_id]
            entry["role"] = "real_frame"
            return entry, {key: object() for key in SOURCE_KEYS}

        with (mock.patch.object(r5, "_condition_source_only_inputs", side_effect=mutate_same_entry) as spy,
              mock.patch.object(r5.audit, "array_info",
                                side_effect=AssertionError("mutated entry must fail before array checks")) as info_spy):
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0,
                    real_frame_mask_metadata=self.entry["arrays"]["observed_crop_mask"],
                )
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(info_spy.call_count, 0)

        def copied_entry(bundle, plan):
            return dict(bundle["r1_contexts"][self.context_id]), {key: object() for key in SOURCE_KEYS}

        with mock.patch.object(r5, "_condition_source_only_inputs", side_effect=copied_entry):
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0,
                    real_frame_mask_metadata=self.entry["arrays"]["observed_crop_mask"],
                )

    def test_loader_rejects_wrong_array_set_and_metadata_without_leaking_projection(self):
        from . import quality_bottle_pose_ablation as r5

        declared = {key: {field: value for field, value in rec.items() if field != "_raw_bytes"}
                    for key, rec in self.metadata.items()}
        masks = declared["observed_crop_mask"]
        cases = (
            ("missing source key", {key: object() for key in SOURCE_KEYS if key != "template_pose_m"}),
            ("private extra", {**{key: object() for key in SOURCE_KEYS}, "query_rgb": object()}),
        )
        for label, loaded in cases:
            with self.subTest(case=label):
                with (mock.patch.object(r5, "_condition_source_only_inputs",
                                        side_effect=lambda bundle, plan: (
                                            bundle["r1_contexts"][self.context_id], loaded)) as spy,
                      mock.patch.object(r5.audit, "array_info",
                                        side_effect=AssertionError("bad key set precedes array checks")) as info_spy):
                    with self.assertRaises(packet.PacketPreflightError):
                        packet.load_source_template(
                            self.capture_root, self.entry,
                            expected_context_id=self.context_id, expected_frame_id=10,
                            expected_offset_deg=0, real_frame_mask_metadata=masks)
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(info_spy.call_count, 0)

        sentinels = {key: object() for key in SOURCE_KEYS}
        wrong_info = dict(declared["template_depth_mm"])
        wrong_info["sha256"] = "e" * 64
        info_by_identity = {id(sentinels[key]): declared[key] for key in SOURCE_KEYS}
        info_by_identity[id(sentinels["template_depth_mm"])] = wrong_info
        with (mock.patch.object(r5, "_condition_source_only_inputs",
                                side_effect=lambda bundle, plan: (
                                    bundle["r1_contexts"][self.context_id], sentinels)),
              mock.patch.object(r5.audit, "array_info",
                                side_effect=lambda value: info_by_identity[id(value)])):
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0, real_frame_mask_metadata=masks)

    def test_loader_rehashes_after_r5_and_obeys_global_deadline(self):
        from . import quality_bottle_pose_ablation as r5

        declared = {key: {field: value for field, value in rec.items() if field != "_raw_bytes"}
                    for key, rec in self.metadata.items()}
        sentinels = {key: object() for key in SOURCE_KEYS}
        info_by_identity = {id(sentinels[key]): declared[key] for key in SOURCE_KEYS}

        def mutate_packet(bundle, plan):
            with self.packet_path.open("ab") as stream:
                stream.write(b"post-preflight-mutation")
            return bundle["r1_contexts"][self.context_id], sentinels

        with (mock.patch.object(r5, "_condition_source_only_inputs", side_effect=mutate_packet),
              mock.patch.object(r5.audit, "array_info", side_effect=lambda value: info_by_identity[id(value)])):
            with self.assertRaises(packet.PacketPreflightError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0,
                    real_frame_mask_metadata=declared["observed_crop_mask"],
                )

        with mock.patch.object(r5, "_condition_source_only_inputs",
                               side_effect=AssertionError("expired loader must not call R5")) as spy:
            with self.assertRaises(packet.PacketResourceLimitError):
                packet.load_source_template(
                    self.capture_root, self.entry,
                    expected_context_id=self.context_id, expected_frame_id=10,
                    expected_offset_deg=0,
                    real_frame_mask_metadata=declared["observed_crop_mask"],
                    deadline_monotonic=packet.time.monotonic() - 1.0,
                )
        self.assertEqual(spy.call_count, 0)

        # Holding monotonic time fixed lets us verify the loader clamps its
        # inherited deadline to 900 seconds and reports the same global budget.
        self._write_archive([(f"{key}.npy", self.metadata[key]["_raw_bytes"] + 128)
                             for key in SOURCE_KEYS])
        sentinels = {key: object() for key in SOURCE_KEYS}
        info_by_identity = {id(sentinels[key]): declared[key] for key in SOURCE_KEYS}
        with (mock.patch.object(r5, "_condition_source_only_inputs",
                               side_effect=lambda bundle, plan:
                               (bundle["r1_contexts"][self.context_id], sentinels)),
              mock.patch.object(r5.audit, "array_info", side_effect=lambda value: info_by_identity[id(value)]),
              mock.patch.object(packet.time, "monotonic", return_value=100.0)):
            result = packet.load_source_template(
                self.capture_root, self.entry,
                expected_context_id=self.context_id, expected_frame_id=10,
                expected_offset_deg=0,
                real_frame_mask_metadata=declared["observed_crop_mask"],
                deadline_monotonic=10**12,
            )
        self.assertEqual(result["receipt"]["deadline_seconds_remaining"], packet.MAX_WALL_SECONDS)


class FrozenSourcePacketSpecTests(unittest.TestCase):
    def test_helper_and_checked_in_spec_use_the_frozen_v2_digest(self):
        root = Path(__file__).resolve().parents[1]
        spec_path = root / ".cache" / "bottle-source-reservation-r8-spec-v2.md"
        self.assertEqual(hashlib.sha256(spec_path.read_bytes()).hexdigest().upper(), SPEC_V2_SHA256)
        self.assertEqual(packet.SPEC_V2_SHA256, SPEC_V2_SHA256)


if __name__ == "__main__":
    unittest.main()
