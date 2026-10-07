"""Independent bounded acceptance tests for the HOT3D capture protocol.

All bundle resources are synthetic opaque bytes. The reader's fake decoder
returns constant arrays and never invokes an image library or opens media.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from . import quality_capture as capture


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PIXELS = 720 * 720
_VALID_BYTES = bytes([1]) * _PIXELS
_CAMERA_VALID_DIGEST = hashlib.sha256(
    b"sampling_valid\0[720,720]\0|u1\0" + _VALID_BYTES
).hexdigest().upper()
_LIMITATION = "Raw capture acquisition only; warp/selection/producer integration pending"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _digest_label(value: str) -> str:
    return _sha(value.encode("ascii"))


def _plain(value):
    """Make an independent mutable JSON-shaped copy of frozen plan values."""
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _immutable(value):
    """Freeze JSON-shaped test mutations without using producer internals."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(child) for child in value)
    return value


def _rgb_value(ordinal: int) -> tuple[int, int, int]:
    return (17 + ordinal, 43 + ordinal, 91 + ordinal)


def _rgb_pixel_digest(ordinal: int) -> str:
    return _sha(bytes(_rgb_value(ordinal)) * _PIXELS)


def _fake_png(kind: str, ordinal: int) -> bytes:
    """A dimension-bearing PNG header followed by a fake-codec marker only."""
    color_type = 2 if kind == "RGB" else 0
    ihdr = (720).to_bytes(4, "big") + (720).to_bytes(4, "big")
    ihdr += bytes((8, color_type, 0, 0, 0))
    return (_PNG_SIGNATURE + (13).to_bytes(4, "big") + b"IHDR" + ihdr +
            b"\0\0\0\0" + b"FAKE|" + kind.encode("ascii") + b"|" +
            str(ordinal).encode("ascii"))


class _SyntheticDecoder:
    """Decode only this test's fake markers into constant in-memory arrays."""

    def __init__(self, *, fail_on: tuple[str, int] | None = None,
                 rgb_override: tuple[int, tuple[int, int, int]] | None = None):
        self.fail_on = fail_on
        self.rgb_override = rgb_override
        self.calls: list[tuple[str, int]] = []

    def __call__(self, encoded: bytes):
        import numpy as np

        marker = encoded[33:].decode("ascii")
        prefix, kind, ordinal_text = marker.split("|")
        if prefix != "FAKE":
            raise AssertionError("decoder received a resource outside the fake-codec boundary")
        ordinal = int(ordinal_text)
        self.calls.append((kind, ordinal))
        if self.fail_on == (kind, ordinal):
            raise RuntimeError("injected fake decoder failure")
        if kind == "RGB":
            value = _rgb_value(ordinal)
            if self.rgb_override is not None and self.rgb_override[0] == ordinal:
                value = self.rgb_override[1]
            result = np.empty((720, 720, 3), dtype=np.uint8)
            result[:] = value
            return result, "RGB"
        if kind == "VALID":
            result = np.full((720, 720), 255, dtype=np.uint8)
            return result, "GRAY"
        raise AssertionError(f"unexpected fake resource marker: {kind}")


class _SyntheticBundle:
    """Minimal public-preflight bundle with three synthetic, joined rows."""

    def __init__(self, root: Path, *, unavailable: frozenset[int] = frozenset()):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.unavailable = unavailable
        self.asset_path = "assets/object.glb"
        self.receipt_path = "assets/asset-receipt.json"
        self.table_path = "captures.json"
        self.write(self.asset_path, b"synthetic asset identity bytes")
        self.write(self.receipt_path, b"synthetic receipt identity bytes")

        source_ids = (260, 262, 263)
        timestamps = (1_000_000_000, 1_033_333_333, 1_066_666_666)
        self.rows: list[dict[str, object]] = []
        inventory_rows: list[dict[str, object]] = []
        for ordinal, (source_id, timestamp_ns) in enumerate(zip(source_ids, timestamps)):
            rgb_path = f"raw/{source_id}.image_214-1.jpg"
            calibration_path = f"calibration/{source_id}.json"
            raw_rgb = f"opaque fake raw source {source_id}".encode("ascii")
            raw_calibration = f"opaque fake calibration {source_id}".encode("ascii")
            self.write(rgb_path, raw_rgb)
            self.write(calibration_path, raw_calibration)
            raw_rgb_sha = _sha(raw_rgb)
            raw_calibration_sha = _sha(raw_calibration)
            camera_metadata_sha = _digest_label(f"camera metadata {source_id}")
            inventory_rows.append({
                "source_frame_id": source_id,
                "calibration_path": calibration_path,
                "calibration_sha256": raw_calibration_sha,
                "raw_camera_metadata_sha256": camera_metadata_sha,
                "rgb_path": rgb_path,
                "rgb_sha256": raw_rgb_sha,
                "source_timestamp_ns": timestamp_ns,
            })

            calibration_values_sha = _digest_label(f"calibration values {ordinal}")
            warp_sha = _digest_label(f"warp {ordinal}")
            map_sha = _digest_label(f"map {ordinal}")
            geometric_sha = _digest_label(f"geometric validity {ordinal}")
            domain_sha = _digest_label(f"domain validity {ordinal}")
            raw_pixel_sha = _digest_label(f"decoded fake raw pixels {ordinal}")
            rgb_pixel_sha = _rgb_pixel_digest(ordinal)
            rgb_path_out = f"rectified/rgb/{ordinal}.png"
            valid_path_out = f"rectified/sampling-valid/{ordinal}.png"
            metadata_path = f"rectified/metadata/{ordinal}.json"
            rgb_bytes = _fake_png("RGB", ordinal)
            valid_bytes = _fake_png("VALID", ordinal)
            if ordinal not in unavailable:
                self.write(rgb_path_out, rgb_bytes)
                self.write(valid_path_out, valid_bytes)

            row: dict[str, object] = {
                "frame_id": ordinal,
                "source_frame_id": source_id,
                "source_timestamp_ns": timestamp_ns,
                "timestamp_s": float(Fraction(timestamp_ns, 1_000_000_000)),
                "role": "scored",
                "raw_rgb": {"path": rgb_path, "sha256": raw_rgb_sha,
                            "byte_count": len(raw_rgb)},
                "decoded_raw_rgb_pixel_sha256": raw_pixel_sha,
                "raw_calibration": {"path": calibration_path,
                                     "sha256": raw_calibration_sha,
                                     "byte_count": len(raw_calibration)},
                "raw_camera_metadata_sha256": camera_metadata_sha,
                "calibration_values_sha256": calibration_values_sha,
                "warp_sha256": warp_sha,
                "map_sha256": map_sha,
                "geometric_valid_sha256": geometric_sha,
                "domain_valid_sha256": domain_sha,
                "sampling_valid_sha256": _CAMERA_VALID_DIGEST,
                "output_rgb": {
                    "path": rgb_path_out, "file_sha256": _sha(rgb_bytes),
                    "byte_count": len(rgb_bytes), "pixel_sha256": rgb_pixel_sha,
                },
                "sampling_valid": {
                    "path": valid_path_out, "file_sha256": _sha(valid_bytes),
                    "byte_count": len(valid_bytes), "pixel_sha256": _sha(_VALID_BYTES),
                },
                "conversion_state": "available",
                "failure_reason": None,
                "conversion_metadata": None,
                "conversion_metadata_sha256": None,
                "capture_row_sha256": "",
            }
            if ordinal in unavailable:
                row.update({
                    "calibration_values_sha256": None,
                    "warp_sha256": None,
                    "map_sha256": None,
                    "geometric_valid_sha256": None,
                    "domain_valid_sha256": None,
                    "sampling_valid_sha256": None,
                    "output_rgb": None,
                    "sampling_valid": None,
                    "conversion_state": "unavailable",
                    "failure_reason": "raw_decode_failed",
                })
            else:
                metadata = {
                    "schema": "quality-camera-warped-rgb-v1",
                    "source_rgb_sha256": raw_pixel_sha,
                    "calibration_sha256": raw_calibration_sha,
                    "canonical_calibration_sha256": calibration_values_sha,
                    "warp_sha256": warp_sha,
                    "sampling_valid_sha256": _CAMERA_VALID_DIGEST,
                    "output_rgb_sha256": rgb_pixel_sha,
                    "output": copy.deepcopy(capture.OUTPUT_CAMERA),
                    "sampling": {
                        "channel_order": "RGB",
                        "interpolation": "bilinear-four-neighbor",
                        "neighbor_footprint": "all four integer pixel centers must be in bounds and declared valid disk",
                        "rounding": "floor(value+0.5)",
                        "clip_range": [0, 255],
                        "invalid_fill_rgb": [128, 128, 128],
                    },
                }
                metadata_bytes = _canonical(metadata)
                self.write(metadata_path, metadata_bytes)
                metadata_sha = _sha(metadata_bytes)
                row["conversion_metadata"] = {
                    "path": metadata_path, "file_sha256": metadata_sha,
                    "byte_count": len(metadata_bytes),
                }
                row["conversion_metadata_sha256"] = metadata_sha
            self.rows.append(row)

        archive_sha = _digest_label("synthetic archive identity")
        self.inventory: dict[str, object] = {
            "schema_version": 1,
            "source_revision": "synthetic-source-revision",
            "clip_id": 1944,
            "archive_sha256": archive_sha,
            "source_units": "nanoseconds",
            "frames": inventory_rows,
            "inference_ready": False,
            "limitation": _LIMITATION,
        }
        self.table: dict[str, object] = {
            "schema_version": 1,
            "source_kind": capture.SOURCE_KIND,
            "source_revision": "synthetic-source-revision",
            "clip_id": 1944,
            "stream_id": "214-1",
            "archive_sha256": archive_sha,
            "source_inventory_sha256": "",
            "output_camera": copy.deepcopy(capture.OUTPUT_CAMERA),
            "output_camera_sha256": _sha(_canonical(capture.OUTPUT_CAMERA)),
            "coordinate_mode": capture.COORDINATE_MODE,
            "converter": {
                "source_closure": self.source_closure(),
                "runtime_decoder": "synthetic-fake-marker-decoder-v1",
            },
            "coverage_complete": True,
            "planned_count": len(self.rows),
            "rows": self.rows,
        }
        self.manifest: dict[str, object] = {
            "schema_version": 2,
            "object": "synthetic_object_8",
            "object_id": 8,
            "units": "metres",
            "asset": self.asset_path,
            "asset_receipt": self.receipt_path,
            "native_resolution": [720, 720],
            "intrinsics": [list(row) for row in capture.OUTPUT_INTRINSICS],
            "rgb_source": {"kind": capture.SOURCE_KIND, "table": self.table_path},
            "source_hashes": {},
            "clock": {
                "mode": "capture_seconds_v1",
                "units": "seconds",
                "source_units": "nanoseconds",
                "time_base": {"num": 1, "den": 1_000_000_000},
                "timestamp_source": "HOT3D source-captures.json:source_timestamp_ns;stream214-1",
                "nominal_fps": 30,
            },
            "setup_frame_id": None,
            "frame_ids": [0, 1, 2],
            "timeline": [
                {"frame_id": ordinal, "source_frame_id": source_id,
                 "source_timestamp": timestamp_ns,
                 "timestamp_s": float(Fraction(timestamp_ns, 1_000_000_000)),
                 "role": "scored"}
                for ordinal, (source_id, timestamp_ns) in enumerate(zip(source_ids, timestamps))
            ],
            "source_gaps": [{"previous_source_frame_id": 260,
                             "current_source_frame_id": 262,
                             "missing_capture_count": 1}],
            "selection": {},
        }
        self.input_bytes = b""
        self.table_bytes = b""
        self.seal()

    def source_closure(self) -> dict[str, str]:
        repository_root = Path(capture.__file__).resolve().parent.parent
        return {
            relative: _sha((repository_root / Path(relative)).read_bytes())
            for relative in ("bench/quality_capture.py", "bench/quality_camera.py",
                             "bench/quality_time.py")
        }

    def path(self, relative: str) -> Path:
        return self.root.joinpath(*relative.split("/"))

    def write(self, relative: str, data: bytes) -> None:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def seal(self, *, sync_selection: bool = True) -> None:
        inventory_bytes = _canonical(self.inventory)
        self.write("source-captures.json", inventory_bytes)
        self.table["source_inventory_sha256"] = _sha(inventory_bytes)
        for row in self.rows:
            row["capture_row_sha256"] = _sha(_canonical({
                key: value for key, value in row.items() if key != "capture_row_sha256"
            }))
        self.table_bytes = _canonical(self.table)
        self.write(self.table_path, self.table_bytes)
        asset_sha = _sha(self.path(self.asset_path).read_bytes())
        receipt_sha = _sha(self.path(self.receipt_path).read_bytes())
        self.manifest["source_hashes"] = {
            self.asset_path: asset_sha,
            self.receipt_path: receipt_sha,
            self.table_path: _sha(self.table_bytes),
        }
        if sync_selection:
            first = self.rows[0]
            self.manifest["selection"] = {
                "reviewed": True,
                "frame_id": 0,
                "source_frame_id": first["source_frame_id"],
                "timestamp_s": first["timestamp_s"],
                "points": [[10.0, 10.0]],
                "labels": [1],
                "capture_row_sha256": first["capture_row_sha256"],
                "output_rgb_sha256": first["output_rgb"]["pixel_sha256"],
                "sampling_valid_sha256": first["sampling_valid_sha256"],
                "origin": "image_only_once",
            }
        self.input_bytes = _canonical(self.manifest)


class CaptureProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hot3d-capture-protocol-")
        self.bundle = _SyntheticBundle(Path(self.temporary.name) / "bundle")

    def tearDown(self):
        self.temporary.cleanup()

    def new_bundle(self, name: str, *, unavailable: frozenset[int] = frozenset()) -> _SyntheticBundle:
        return _SyntheticBundle(Path(self.temporary.name) / name, unavailable=unavailable)

    def assert_rejected_before_path_resolution(self, action, *, expected_exception=ValueError):
        with patch.object(Path, "resolve", side_effect=AssertionError("filesystem resolution reached")) as poison:
            with self.assertRaises(expected_exception):
                action()
        self.assertEqual(poison.call_count, 0)

    def test_public_preflight_joins_source_ids_timestamps_gaps_and_selection(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        self.assertEqual([row["frame_id"] for row in verified.rows], [0, 1, 2])
        self.assertEqual([row["source_frame_id"] for row in verified.rows], [260, 262, 263])
        self.assertEqual([row["source_timestamp_ns"] for row in verified.rows],
                         [1_000_000_000, 1_033_333_333, 1_066_666_666])
        self.assertEqual([key.timestamp_s for key in verified.timing["scored_keys"]],
                         [float(Fraction(row["source_timestamp_ns"], 1_000_000_000))
                          for row in verified.rows])
        self.assertEqual(self.bundle.manifest["source_gaps"], [{
            "previous_source_frame_id": 260, "current_source_frame_id": 262,
            "missing_capture_count": 1,
        }])
        self.assertEqual(verified.selection["source_frame_id"], 260)
        self.assertEqual(verified.selection["capture_row_sha256"],
                         verified.rows[0]["capture_row_sha256"])

    def test_source_gaps_are_optional_only_when_timeline_is_contiguous(self):
        contiguous = self.new_bundle("contiguous_no_gaps")
        contiguous_ids = (260, 261, 262)
        for ordinal, source_id in enumerate(contiguous_ids):
            contiguous.rows[ordinal]["source_frame_id"] = source_id
            contiguous.inventory["frames"][ordinal]["source_frame_id"] = source_id
            contiguous.manifest["timeline"][ordinal]["source_frame_id"] = source_id
        contiguous.manifest.pop("source_gaps")
        contiguous.seal()
        verified = capture.load_capture_plan(contiguous.root, contiguous.input_bytes)
        self.assertEqual([row["source_frame_id"] for row in verified.rows], list(contiguous_ids))

        mismatched = self.new_bundle("mismatched_gaps")
        mismatched.manifest["source_gaps"][0]["missing_capture_count"] = 2
        mismatched.input_bytes = _canonical(mismatched.manifest)
        with self.assertRaises(ValueError):
            capture.load_capture_plan(mismatched.root, mismatched.input_bytes)

        invalid_timing = (
            ("missing_gap_declaration", []),
            ("bool_gap_count", [{"previous_source_frame_id": 260,
                                  "current_source_frame_id": 262,
                                  "missing_capture_count": True}]),
        )
        for name, gaps in invalid_timing:
            with self.subTest(name=name):
                bad = self.new_bundle(name)
                bad.manifest["source_gaps"] = gaps
                bad.input_bytes = _canonical(bad.manifest)
                self.assert_rejected_before_path_resolution(
                    lambda bad=bad: capture.load_capture_plan(bad.root, bad.input_bytes))

        bad_unknown = self.new_bundle("unknown_evaluator_input")
        bad_unknown.manifest["evaluator_pose"] = {"translation": [0, 0, 0]}
        bad_unknown.input_bytes = _canonical(bad_unknown.manifest)
        self.assert_rejected_before_path_resolution(
            lambda: capture.load_capture_plan(bad_unknown.root, bad_unknown.input_bytes))

    def test_authentic_source_only_inventory_fields_are_bounded_and_not_evaluator_data(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        self.assertFalse(self.bundle.inventory["inference_ready"])
        self.assertEqual(self.bundle.inventory["limitation"], _LIMITATION)
        self.assertEqual(len(verified.rows), 3)

        for name, mutate in (
            ("inference_ready_true", lambda inv: inv.update(inference_ready=True)),
            ("evaluator_unknown", lambda inv: inv.update(evaluator_pose={"x": 1})),
            ("limitation_not_string", lambda inv: inv.update(limitation=False)),
            ("limitation_too_long", lambda inv: inv.update(limitation="x" * 513)),
        ):
            with self.subTest(name=name):
                bad = self.new_bundle(name)
                mutate(bad.inventory)
                bad.seal()
                with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
                    capture.load_capture_plan(bad.root, bad.input_bytes)

    def test_public_preflight_rejects_timeline_selection_and_traversal_mutations(self):
        bad_timeline = self.new_bundle("timeline_source_id")
        bad_timeline.rows[1]["source_frame_id"] = 999
        bad_timeline.seal()
        with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
            capture.load_capture_plan(bad_timeline.root, bad_timeline.input_bytes)

        bad_selection = self.new_bundle("selection_source_id")
        bad_selection.seal()
        bad_selection.manifest["selection"]["source_frame_id"] = 261
        bad_selection.input_bytes = _canonical(bad_selection.manifest)
        with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
            capture.load_capture_plan(bad_selection.root, bad_selection.input_bytes)

        bad_path = self.new_bundle("traversal_output")
        bad_path.rows[0]["output_rgb"]["path"] = "../outside.png"
        bad_path.seal()
        with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
            capture.load_capture_plan(bad_path.root, bad_path.input_bytes)

        bad_camera = self.new_bundle("changed_output_camera")
        bad_camera.table["output_camera"]["fx"] = 361.0
        bad_camera.seal()
        with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
            capture.load_capture_plan(bad_camera.root, bad_camera.input_bytes)

    def test_replaced_public_plan_tokens_do_not_authorize_changed_semantics(self):
        plan = capture.prepare_capture_plan(self.bundle.input_bytes, self.bundle.table_bytes)
        changed_selection = _plain(plan.selection)
        changed_selection["source_frame_id"] = 261
        changed_manifest = _plain(plan.manifest)
        changed_manifest["object"] = "replaced-object"
        changed_rows = list(plan.rows)
        changed_rows[0] = {**_plain(changed_rows[0]), "source_frame_id": 261}
        changed_timestamp_rows = list(plan.rows)
        changed_timestamp_rows[0] = {
            **_plain(changed_timestamp_rows[0]),
            "timestamp_s": plan.rows[0]["timestamp_s"] + 0.25,
        }
        from .quality_time import FrameKey, PHYSICAL
        changed_timing = _plain(plan.timing)
        changed_timing["scored_keys"] = tuple(
            [FrameKey(0, plan.timing["scored_keys"][0].timestamp_s + 0.25, PHYSICAL)] +
            list(plan.timing["scored_keys"][1:]))
        changed_table = _plain(plan.table)
        changed_table["stream_id"] = "unauthenticated-stream"
        changes = (
            {"manifest": _immutable(changed_manifest)},
            {"timing": _immutable(changed_timing)},
            {"selection": _immutable(changed_selection)},
            {"rows": tuple(_immutable(row) for row in changed_rows)},
            {"rows": tuple(_immutable(row) for row in changed_timestamp_rows)},
            {"table": _immutable(changed_table)},
            {"input_manifest_sha256": _digest_label("changed input digest")},
            {"capture_table_sha256": _digest_label("changed table digest")},
        )
        for replacement in changes:
            with self.subTest(replacement=tuple(replacement)):
                mutated = replace(plan, **replacement)
                with patch.object(Path, "resolve", side_effect=AssertionError("bundle I/O reached")):
                    with self.assertRaises(capture.CaptureIntegrityError):
                        capture.verify_capture_resources(self.bundle.root, mutated)

        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        changed_verified_plan = replace(
            verified.plan, input_manifest_sha256=_digest_label("changed verified input digest"))
        changed_verified = replace(verified, plan=changed_verified_plan)
        decoder = _SyntheticDecoder()
        with patch.object(Path, "resolve", side_effect=AssertionError("bundle I/O reached")):
            with self.assertRaises(capture.CaptureIntegrityError):
                capture.CaptureReader(self.bundle.root, changed_verified, decoder=decoder)
        self.assertEqual(decoder.calls, [])

    def test_direct_unissued_records_and_subclasses_fail_before_path_resolution(self):
        plan = capture.prepare_capture_plan(self.bundle.input_bytes, self.bundle.table_bytes)
        direct_plan = capture.CapturePlan(
            manifest=plan.manifest, table=plan.table, timing=plan.timing, rows=plan.rows,
            input_manifest_sha256=plan.input_manifest_sha256,
            capture_table_sha256=plan.capture_table_sha256, selection=plan.selection)

        class PlanSubclass(capture.CapturePlan):
            __slots__ = ()

        subclass_plan = PlanSubclass(
            manifest=plan.manifest, table=plan.table, timing=plan.timing, rows=plan.rows,
            input_manifest_sha256=plan.input_manifest_sha256,
            capture_table_sha256=plan.capture_table_sha256, selection=plan.selection)
        for candidate, expected_exception in (
            (direct_plan, ValueError),
            (subclass_plan, TypeError),
        ):
            with self.subTest(candidate_type=type(candidate).__name__):
                self.assert_rejected_before_path_resolution(
                    lambda candidate=candidate: capture.verify_capture_resources(
                        self.bundle.root, candidate),
                    expected_exception=expected_exception)

        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        direct_verified = capture.VerifiedCapturePlan(
            plan=verified.plan, bundle=verified.bundle,
            verified_resources=verified.verified_resources)

        class VerifiedSubclass(capture.VerifiedCapturePlan):
            __slots__ = ()

        subclass_verified = VerifiedSubclass(
            plan=verified.plan, bundle=verified.bundle,
            verified_resources=verified.verified_resources)
        decoder = _SyntheticDecoder()
        for candidate in (direct_verified, subclass_verified):
            with self.subTest(candidate_type=type(candidate).__name__):
                self.assert_rejected_before_path_resolution(
                    lambda candidate=candidate: capture.CaptureReader(
                        self.bundle.root, candidate, decoder=decoder))
        self.assertEqual(decoder.calls, [])

        capability = capture.SingleFrameCapability(
            input_manifest_sha256=verified.input_manifest_sha256,
            capture_table_sha256=verified.capture_table_sha256,
            ordinal=0, source_frame_id=verified.rows[0]["source_frame_id"],
            capture_row_sha256=verified.rows[0]["capture_row_sha256"])

        class CapabilitySubclass(capture.SingleFrameCapability):
            __slots__ = ()

        subclass_capability = CapabilitySubclass(
            input_manifest_sha256=verified.input_manifest_sha256,
            capture_table_sha256=verified.capture_table_sha256,
            ordinal=0, source_frame_id=verified.rows[0]["source_frame_id"],
            capture_row_sha256=verified.rows[0]["capture_row_sha256"])
        for candidate in (capability, subclass_capability):
            with self.subTest(candidate_type=type(candidate).__name__):
                self.assert_rejected_before_path_resolution(
                    lambda candidate=candidate: capture.CaptureReader.from_capability(
                        self.bundle.root, verified, candidate, decoder=decoder))
        self.assertEqual(decoder.calls, [])

    def test_mutable_equal_value_replacements_fail_before_path_resolution(self):
        plan = capture.prepare_capture_plan(self.bundle.input_bytes, self.bundle.table_bytes)
        mutable_variants = (
            replace(plan, manifest=_plain(plan.manifest)),
            replace(plan, table=_plain(plan.table)),
            replace(plan, rows=tuple(_plain(row) for row in plan.rows)),
            replace(plan, selection=_plain(plan.selection)),
        )
        for index, candidate in enumerate(mutable_variants):
            with self.subTest(plan_field=index):
                self.assert_rejected_before_path_resolution(
                    lambda candidate=candidate: capture.verify_capture_resources(
                        self.bundle.root, candidate))

        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        mutable_resources = replace(verified, verified_resources=dict(verified.verified_resources))
        mutable_nested_plan = replace(verified, plan=replace(
            verified.plan, selection=_plain(verified.plan.selection)))
        decoder = _SyntheticDecoder()
        for candidate in (mutable_resources, mutable_nested_plan):
            with self.subTest(candidate=candidate):
                self.assert_rejected_before_path_resolution(
                    lambda candidate=candidate: capture.CaptureReader(
                        self.bundle.root, candidate, decoder=decoder))
        self.assertEqual(decoder.calls, [])

    def test_verified_bundle_resources_and_nested_plan_replacements_are_rejected(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        other_bundle = self.bundle.root.parent / "different-bundle"
        changed_resources = _plain(verified.verified_resources)
        changed_resources["unbound/resource.bin"] = _digest_label("unbound resource")
        changed_resources = _immutable(changed_resources)
        changed_plan = replace(
            verified.plan,
            capture_table_sha256=_digest_label("replaced nested table digest"))
        candidates = (
            replace(verified, bundle=other_bundle),
            replace(verified, verified_resources=changed_resources),
            replace(verified, plan=changed_plan),
        )
        decoder = _SyntheticDecoder()
        for candidate in candidates:
            with self.subTest(bundle=candidate.bundle, plan_digest=candidate.plan.capture_table_sha256):
                self.assert_rejected_before_path_resolution(
                    lambda candidate=candidate: capture.CaptureReader(
                        self.bundle.root, candidate, decoder=decoder))
        self.assertEqual(decoder.calls, [])

    def test_genuine_unchanged_issued_records_remain_reusable(self):
        plan = capture.prepare_capture_plan(self.bundle.input_bytes, self.bundle.table_bytes)
        verified = capture.verify_capture_resources(self.bundle.root, plan)
        verified_again = capture.verify_capture_resources(self.bundle.root, plan)
        self.assertEqual(verified.input_manifest_sha256, verified_again.input_manifest_sha256)
        self.assertEqual(verified.capture_table_sha256, verified_again.capture_table_sha256)
        reader = capture.CaptureReader(self.bundle.root, verified, decoder=_SyntheticDecoder())
        first = reader.read(0)
        reader.allow(1)
        second = reader.read(1)
        self.assertEqual(first.source_frame_id, 260)
        self.assertEqual(second.source_frame_id, 262)

    def test_changed_declared_bytes_fail_preflight_before_fake_decoder(self):
        self.bundle.write("rectified/rgb/0.png", b"changed after table sealing")
        with patch.object(capture, "_decode_result",
                          side_effect=AssertionError("decoder reached during preflight")) as decoder:
            with self.assertRaises(capture.CaptureIntegrityError):
                capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        self.assertEqual(decoder.call_count, 0)

    def test_reader_enforces_causal_allowance_cache_and_immutable_arrays(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        decoder = _SyntheticDecoder()
        reader = capture.CaptureReader(self.bundle.root, verified, decoder=decoder)
        for bad_ordinal in (True, 1.0, -1, 3):
            with self.subTest(bad_ordinal=bad_ordinal):
                with self.assertRaises(ValueError):
                    reader.read(bad_ordinal)
        for bad_allowance in (True, 2):
            with self.subTest(bad_allowance=bad_allowance):
                with self.assertRaises(ValueError):
                    reader.allow(bad_allowance)
        self.assertEqual(decoder.calls, [])
        self.assertEqual(reader.snapshot(0)["attempts"], 0)

        first = reader.read(0)
        cached = reader.read(0)
        self.assertIs(first, cached)
        self.assertEqual(reader.snapshot(0)["cache_hits"], 1)
        self.assertFalse(first.rgb.flags.writeable)
        self.assertFalse(first.sampling_valid.flags.writeable)
        with self.assertRaises(ValueError):
            first.rgb[0, 0, 0] = 255
        with self.assertRaises(ValueError):
            first.sampling_valid[0, 0] = False

        reader.allow(1)
        second = reader.read(1)
        self.assertEqual(second.source_frame_id, 262)
        self.assertEqual(reader.snapshot(1)["cache_ordinal"], 1)
        reread = reader.read(0)
        self.assertIsNot(first, reread)
        self.assertEqual(reread.rgb_pixel_sha256, first.rgb_pixel_sha256)
        self.assertEqual(reader.snapshot(0)["past_reads"], 1)
        self.assertEqual(reader.snapshot(0)["decode_calls"], 4)

    def test_one_frame_capability_reads_only_its_bound_row(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        capability = capture.single_frame_capability(verified, 2)
        decoder = _SyntheticDecoder()
        reader = capture.CaptureReader.from_capability(
            self.bundle.root, verified, capability, decoder=decoder)
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.read(0)
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.allow(1)
        self.assertEqual(decoder.calls, [])
        row = reader.read(2)
        self.assertEqual(row.source_frame_id, 263)
        self.assertEqual(decoder.calls, [("RGB", 2), ("VALID", 2)])

    def test_known_unavailable_row_is_nonterminal_and_later_row_remains_readable(self):
        bundle = self.new_bundle("unavailable", unavailable=frozenset({1}))
        verified = capture.load_capture_plan(bundle.root, bundle.input_bytes)
        decoder = _SyntheticDecoder()
        reader = capture.CaptureReader(bundle.root, verified, decoder=decoder)
        reader.read(0)
        reader.allow(1)
        with self.assertRaises(capture.CaptureUnavailable) as caught:
            reader.read(1)
        self.assertEqual(caught.exception.source_frame_id, 262)
        self.assertEqual(caught.exception.reason, "raw_decode_failed")
        self.assertFalse(reader.snapshot(1)["terminal"])
        reader.allow(2)
        self.assertEqual(reader.read(2).source_frame_id, 263)

    def test_decoder_failure_terminalizes_reader_and_blocks_later_rows(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        decoder = _SyntheticDecoder(fail_on=("RGB", 1))
        reader = capture.CaptureReader(self.bundle.root, verified, decoder=decoder)
        reader.read(0)
        reader.allow(1)
        with self.assertRaises(capture.CaptureDecodeError) as caught:
            reader.read(1)
        self.assertEqual(caught.exception.ordinal, 1)
        self.assertEqual(caught.exception.source_frame_id, 262)
        self.assertTrue(reader.snapshot(1)["terminal"])
        calls_after_failure = len(decoder.calls)
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.allow(2)  # terminal state rejects this without reaching the decoder
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.read(2)
        self.assertEqual(len(decoder.calls), calls_after_failure)

    def test_post_preflight_resource_change_is_rejected_before_decode(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        decoder = _SyntheticDecoder()
        reader = capture.CaptureReader(self.bundle.root, verified, decoder=decoder)
        self.bundle.write("rectified/rgb/0.png", b"post-preflight replacement")
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.read(0)
        self.assertEqual(decoder.calls, [])
        self.assertTrue(reader.snapshot(0)["terminal"])
        self.assertEqual(reader.snapshot(0)["decode_calls"], 0)

    def test_decoded_pixel_digest_mismatch_is_terminal(self):
        verified = capture.load_capture_plan(self.bundle.root, self.bundle.input_bytes)
        decoder = _SyntheticDecoder(rgb_override=(0, (1, 2, 3)))
        reader = capture.CaptureReader(self.bundle.root, verified, decoder=decoder)
        with self.assertRaises(capture.CaptureIntegrityError) as caught:
            reader.read(0)
        self.assertIn("RGB pixel digest mismatch", str(caught.exception))
        self.assertTrue(reader.snapshot(0)["terminal"])
        self.assertEqual(decoder.calls, [("RGB", 0), ("VALID", 0)])
        calls_after_failure = tuple(decoder.calls)
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.allow(1)
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.read(0)
        with self.assertRaises(capture.CaptureIntegrityError):
            reader.read(1)
        self.assertEqual(tuple(decoder.calls), calls_after_failure)
        self.assertEqual(reader.snapshot(0)["returns"], 0)
        self.assertEqual(reader.snapshot(1)["decode_calls"], 0)

    def test_conversion_metadata_allowlist_and_row_binding_are_preflighted(self):
        for name, mutate in (
            ("missing_sampling", lambda metadata: metadata.pop("sampling")),
            ("unexpected_evaluator_field", lambda metadata: metadata.update(evaluator_score=1.0)),
            ("wrong_pixel_binding", lambda metadata: metadata.update(output_rgb_sha256=_digest_label("wrong rgb"))),
        ):
            with self.subTest(name=name):
                bundle = self.new_bundle(name)
                row = bundle.rows[0]
                resource = row["conversion_metadata"]
                metadata = json.loads(bundle.path(resource["path"]).read_text(encoding="utf-8"))
                mutate(metadata)
                encoded = _canonical(metadata)
                bundle.write(resource["path"], encoded)
                resource["file_sha256"] = _sha(encoded)
                resource["byte_count"] = len(encoded)
                row["conversion_metadata_sha256"] = _sha(encoded)
                bundle.seal()
                with self.assertRaises(capture.CaptureIntegrityError):
                    capture.load_capture_plan(bundle.root, bundle.input_bytes)

    def test_content_key_and_sidecar_allowlist_authenticate_inert_artifact(self):
        asset_receipt = {
            "asset_sha256": _digest_label("asset"),
            "unit_receipt_sha256": _digest_label("unit receipt"),
            "object_id": 8,
        }
        recipe = {"name": "synthetic-model-smoke", "render_flags": {"back_face_culling": False}}
        closure = {
            "bench/quality_capture.py": _digest_label("capture source"),
            "bench/quality_camera.py": _digest_label("camera source"),
        }
        key = capture.capture_resource_key(asset_receipt, capture.OUTPUT_CAMERA, recipe, closure)
        reversed_key = capture.capture_resource_key(
            dict(reversed(tuple(asset_receipt.items()))), capture.OUTPUT_CAMERA,
            dict(reversed(tuple(recipe.items()))), dict(reversed(tuple(closure.items()))))
        self.assertEqual(key, reversed_key)
        self.assertNotEqual(key, capture.capture_resource_key(
            asset_receipt, capture.OUTPUT_CAMERA, {**recipe, "variant": 2}, closure))

        resource_dir = self.bundle.root / ".cache" / "capture-resources" / key
        resource_dir.mkdir(parents=True)
        artifact = b"opaque synthetic artifact; never deserialized"
        (resource_dir / "artifact.bin").write_bytes(artifact)
        sidecar = {
            "schema_version": 1,
            "resource_kind": "synthetic-test-artifact",
            "resource_key": key,
            "recipe": recipe,
            "dimensions": [720, 720],
            "coordinate_mode": capture.COORDINATE_MODE,
            "artifact": {"path": "artifact.bin", "sha256": _sha(artifact),
                         "byte_count": len(artifact)},
            "runtime": {"device": "synthetic", "build": "unit-test"},
            "source_closure": closure,
        }
        sidecar_path = resource_dir / "resource.json"
        sidecar_bytes = _canonical(sidecar)
        sidecar_path.write_bytes(sidecar_bytes)
        descriptor = capture.verify_stage_resource_sidecar(
            sidecar_path, expected_key=key, expected_recipe=recipe,
            expected_source_closure=closure)
        self.assertEqual(descriptor.artifact_sha256, _sha(artifact))
        self.assertEqual(descriptor.artifact_byte_count, len(artifact))
        self.assertEqual(descriptor.sidecar_sha256, _sha(sidecar_bytes))

        sidecar["unreviewed_extra"] = True
        sidecar_path.write_bytes(_canonical(sidecar))
        with self.assertRaisesRegex(ValueError, "stage sidecar fields mismatch") as caught:
            capture.verify_stage_resource_sidecar(
                sidecar_path, expected_key=key, expected_recipe=recipe,
                expected_source_closure=closure)
        self.assertIs(type(caught.exception), ValueError)
        sidecar.pop("unreviewed_extra")
        sidecar_path.write_bytes(_canonical(sidecar))
        (resource_dir / "artifact.bin").write_bytes(b"artifact changed")
        with self.assertRaises(capture.CaptureIntegrityError):
            capture.verify_stage_resource_sidecar(
                sidecar_path, expected_key=key, expected_recipe=recipe,
                expected_source_closure=closure)


if __name__ == "__main__":
    unittest.main()
