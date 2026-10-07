"""Synthetic acceptance for missing-source capture conversion rows.

The fake camera and codec services operate only on opaque byte markers and
small in-memory arrays. No real image decoder or calibration implementation is
used.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import bench
from . import quality_capture as capture


_ROWS = (
    (260, 1_000_000_000, b"good-calibration"),
    (261, 1_033_333_333, b"invalid-calibration"),
    (262, 1_066_666_666, b"warp-failure-calibration"),
)
_SAMPLING = {
    "channel_order": "RGB",
    "interpolation": "bilinear-four-neighbor",
    "neighbor_footprint": "all four integer pixel centers must be in bounds and declared valid disk",
    "rounding": "floor(value+0.5)",
    "clip_range": [0, 255],
    "invalid_fill_rgb": [128, 128, 128],
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _digest_label(value: str) -> str:
    return _sha(value.encode("ascii"))


def _write(root: Path, relative: str, data: bytes) -> None:
    target = root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _inventory(frames: list[dict[str, object]]) -> bytes:
    return _canonical({
        "schema_version": 1,
        "source_revision": "synthetic-missing-input-protocol-v1",
        "clip_id": 0,
        "archive_sha256": _digest_label("opaque synthetic archive"),
        "source_units": "nanoseconds",
        "inference_ready": False,
        "limitation": "Synthetic acquisition fixture; no image or model data",
        "frames": frames,
    })


class _SyntheticConversionFixture:
    def __init__(self, root: Path, *, count: int = 3,
                 calibration_override: tuple[int, bytes] | None = None,
                 available_rgb_ids: frozenset[int] | None = None):
        self.source_root = root / "source"
        self.bundle_root = root / "bundle"
        self.source_root.mkdir(parents=True)
        if available_rgb_ids is None:
            available_rgb_ids = frozenset((_ROWS[0][0],))
        self.entries: list[dict[str, object]] = []
        self.calibration_bytes: dict[int, bytes] = {}
        for ordinal, (source_id, timestamp_ns, calibration) in enumerate(_ROWS[:count]):
            rgb_path = f"raw/{source_id}.image_214-1.jpg"
            calibration_path = f"calibration/{source_id}.json"
            expected_rgb = f"opaque rgb bytes for source {source_id}".encode("ascii")
            self.calibration_bytes[source_id] = calibration
            actual_calibration = calibration
            if calibration_override is not None and calibration_override[0] == source_id:
                actual_calibration = calibration_override[1]
            _write(self.source_root, calibration_path, actual_calibration)
            # Ordinal zero is the sole available source row so the public input
            # manifest can bind its required initial image selection. Later RGB
            # paths intentionally remain absent while their hashes stay pinned.
            if source_id in available_rgb_ids:
                _write(self.source_root, rgb_path, expected_rgb)
            self.entries.append({
                "source_frame_id": source_id,
                "calibration_path": calibration_path,
                "calibration_sha256": _sha(calibration),
                "raw_camera_metadata_sha256": _digest_label(f"camera metadata {source_id}"),
                "rgb_path": rgb_path,
                "rgb_sha256": _sha(expected_rgb),
                "source_timestamp_ns": timestamp_ns,
            })
        self.inventory_bytes = _inventory(self.entries)
        self.inventory_path = self.source_root / "source-captures.json"
        self.inventory_path.write_bytes(self.inventory_bytes)


class _FakeCameraServices:
    def __init__(self, *, invalid_calibration: bytes, warp_failure_calibration: bytes):
        self.normalize_calls: list[bytes] = []
        self.build_calls: list[bytes] = []
        self.warp_calls: list[bytes] = []
        self.invalid_calibration = invalid_calibration
        self.warp_failure_calibration = warp_failure_calibration
        self.module = types.ModuleType("bench.quality_camera")
        self.module.PinholeOutput = self._pinhole_output
        self.module.normalize_calibration = self._normalize_calibration
        self.module.build_warp = self._build_warp
        self.module.warp_rgb = self._warp_rgb

    @staticmethod
    def _pinhole_output():
        return types.SimpleNamespace(
            width=720, height=720, fx=360.0, fy=360.0, cx=359.5, cy=359.5,
            rotation_output_from_source=tuple(capture.OUTPUT_CAMERA["rotation_output_from_source"]),
            pixel_center_convention=capture.OUTPUT_CAMERA["pixel_center_convention"],
        )

    def _normalize_calibration(self, raw: bytes):
        self.normalize_calls.append(raw)
        if raw == self.invalid_calibration:
            raise ValueError("injected malformed calibration bytes")
        kind = "warp-failure" if raw == self.warp_failure_calibration else "good"
        return types.SimpleNamespace(
            source_bytes=raw,
            source_kind=kind,
            image_width=3,
            image_height=2,
            stream_id="214-1",
            raw_calibration_sha256=_sha(raw),
            canonical_values_sha256=_digest_label(f"canonical calibration {raw.decode('ascii')}"),
        )

    def _build_warp(self, calibration, *, tile_rows: int):
        self.build_calls.append(calibration.source_bytes)
        if calibration.source_bytes == self.warp_failure_calibration:
            raise RuntimeError("injected warp construction failure")
        valid_digest = hashlib.sha256(
            b"sampling_valid\0[720,720]\0|u1\0" + bytes([1]) * (720 * 720)
        ).hexdigest().upper()
        return types.SimpleNamespace(
            calibration=calibration,
            warp_sha256=_digest_label("synthetic warp"),
            map_sha256=_digest_label("synthetic map"),
            geometric_valid_sha256=_digest_label("synthetic geometric validity"),
            domain_valid_sha256=_digest_label("synthetic domain validity"),
            sampling_valid_sha256=valid_digest,
        )

    def _warp_rgb(self, decoded_rgb, warp):
        import numpy as np

        self.warp_calls.append(warp.calibration.source_bytes)
        rgb = np.full((720, 720, 3), (31, 61, 91), dtype=np.uint8)
        sampling_valid = np.ones((720, 720), dtype=np.bool_)
        source_digest = hashlib.sha256(memoryview(np.ascontiguousarray(decoded_rgb)).cast("B")).hexdigest().upper()
        output_digest = hashlib.sha256(memoryview(np.ascontiguousarray(rgb)).cast("B")).hexdigest().upper()
        metadata = {
            "schema": "quality-camera-warped-rgb-v1",
            "source_rgb_sha256": source_digest,
            "calibration_sha256": warp.calibration.raw_calibration_sha256,
            "canonical_calibration_sha256": warp.calibration.canonical_values_sha256.upper(),
            "warp_sha256": warp.warp_sha256.upper(),
            "sampling_valid_sha256": warp.sampling_valid_sha256.upper(),
            "output_rgb_sha256": output_digest,
            "output": copy.deepcopy(capture.OUTPUT_CAMERA),
            "sampling": copy.deepcopy(_SAMPLING),
        }
        metadata_bytes = _canonical(metadata)
        return types.SimpleNamespace(
            rgb=rgb,
            sampling_valid=sampling_valid,
            metadata_json_bytes=metadata_bytes,
            metadata_sha256=_sha(metadata_bytes),
            export_metadata=lambda: metadata_bytes,
        )


class _FakePngWriter:
    def __init__(self):
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.decoded_arrays: dict[bytes, tuple[object, str]] = {}

    def __call__(self, array, image_format: str, channel_order: str) -> bytes:
        import numpy as np

        self.calls.append((channel_order, tuple(array.shape)))
        payload = f"opaque-fake-png-{len(self.calls)}".encode("ascii")
        self.decoded_arrays[payload] = (np.array(array, copy=True), channel_order)
        return payload


class _FakeImageDecoder:
    def __init__(self, raw_payload: bytes, writer: _FakePngWriter):
        import numpy as np

        self.raw_payload = raw_payload
        self.writer = writer
        self.raw_array = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))
        self.calls: list[bytes] = []

    def __call__(self, encoded: bytes):
        import numpy as np

        self.calls.append(encoded)
        if encoded == self.raw_payload:
            return np.array(self.raw_array, copy=True), "RGB"
        if encoded in self.writer.decoded_arrays:
            array, channel_order = self.writer.decoded_arrays[encoded]
            return np.array(array, copy=True), channel_order
        raise AssertionError("fake decoder received an unknown opaque payload")


class CaptureConversionFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hot3d-conversion-failure-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _install_fake_camera(self, camera):
        return (
            patch.object(bench, "quality_camera", camera.module, create=True),
            patch.dict(sys.modules, {"bench.quality_camera": camera.module}),
        )

    def _input_manifest(self, fixture: _SyntheticConversionFixture, receipt,
                        asset_bytes: bytes, asset_receipt_bytes: bytes) -> bytes:
        first = receipt.rows[0]
        frame_ids = [row["frame_id"] for row in receipt.rows]
        timeline = [
            {
                "frame_id": ordinal,
                "source_frame_id": source_id,
                "source_timestamp": timestamp_ns,
                "timestamp_s": float(Fraction(timestamp_ns, 1_000_000_000)),
                "role": "scored",
            }
            for ordinal, (source_id, timestamp_ns, _) in enumerate(_ROWS[:len(receipt.rows)])
        ]
        return _canonical({
            "schema_version": 2,
            "object": "synthetic_obj_000001",
            "object_id": 1,
            "units": "metres",
            "asset": "assets/object.glb",
            "asset_receipt": "assets/receipt.json",
            "native_resolution": [720, 720],
            "intrinsics": [list(row) for row in capture.OUTPUT_INTRINSICS],
            "rgb_source": {"kind": capture.SOURCE_KIND, "table": "captures.json"},
            "source_hashes": {
                "assets/object.glb": _sha(asset_bytes),
                "assets/receipt.json": _sha(asset_receipt_bytes),
                "captures.json": receipt.capture_table_sha256,
            },
            "clock": {
                "mode": "capture_seconds_v1",
                "units": "seconds",
                "source_units": "nanoseconds",
                "time_base": {"num": 1, "den": 1_000_000_000},
                "timestamp_source": "HOT3D source-captures.json:source_timestamp_ns;stream214-1",
                "nominal_fps": 30,
            },
            "setup_frame_id": None,
            "frame_ids": frame_ids,
            "timeline": timeline,
            "selection": {
                "reviewed": True,
                "frame_id": 0,
                "source_frame_id": first["source_frame_id"],
                "timestamp_s": first["timestamp_s"],
                "points": [[300.0, 300.0]],
                "labels": [1],
                "capture_row_sha256": first["capture_row_sha256"],
                "output_rgb_sha256": first["output_rgb"]["pixel_sha256"],
                "sampling_valid_sha256": first["sampling_valid_sha256"],
                "origin": "image_only_once",
            },
        })

    def test_missing_rgb_preserves_reason_counts_and_timeline_through_public_preflight(self):
        fixture = _SyntheticConversionFixture(self.root, count=3)
        invalid_calibration = fixture.calibration_bytes[261]
        warp_failure_calibration = fixture.calibration_bytes[262]
        camera = _FakeCameraServices(
            invalid_calibration=invalid_calibration,
            warp_failure_calibration=warp_failure_calibration)
        raw_payload = f"opaque rgb bytes for source 260".encode("ascii")
        writer = _FakePngWriter()
        decoder = _FakeImageDecoder(raw_payload, writer)
        output_root = fixture.bundle_root

        camera_patch, module_patch = self._install_fake_camera(camera)
        with camera_patch, module_patch:
            receipt = capture.convert_capture_sequence(
                fixture.source_root,
                fixture.inventory_path,
                output_root,
                expected_inventory_sha256=_sha(fixture.inventory_bytes),
                decoder=decoder,
                writer=writer,
            )

        self.assertEqual(receipt.planned_count, 3)
        self.assertEqual(receipt.available_count, 1)
        self.assertEqual(receipt.unavailable_count, 2)
        self.assertEqual([row["frame_id"] for row in receipt.rows], [0, 1, 2])
        self.assertEqual([row["source_frame_id"] for row in receipt.rows], [260, 261, 262])
        self.assertEqual([row["source_timestamp_ns"] for row in receipt.rows],
                         [1_000_000_000, 1_033_333_333, 1_066_666_666])
        self.assertEqual([row["timestamp_s"] for row in receipt.rows], [
            float(Fraction(1_000_000_000, 1_000_000_000)),
            float(Fraction(1_033_333_333, 1_000_000_000)),
            float(Fraction(1_066_666_666, 1_000_000_000)),
        ])

        for ordinal, source_id in ((1, 261), (2, 262)):
            row = receipt.rows[ordinal]
            self.assertEqual(row["conversion_state"], "unavailable")
            self.assertEqual(row["failure_reason"], "raw_missing")
            self.assertIsNone(row["raw_rgb"]["path"])
            self.assertIsNone(row["raw_rgb"]["byte_count"])
            self.assertEqual(row["raw_rgb"]["sha256"], fixture.entries[ordinal]["rgb_sha256"])
            self.assertEqual(row["raw_calibration"]["path"], fixture.entries[ordinal]["calibration_path"])
            self.assertEqual(row["raw_calibration"]["sha256"], fixture.entries[ordinal]["calibration_sha256"])
            self.assertEqual(row["raw_calibration"]["byte_count"], len(fixture.calibration_bytes[source_id]))
            self.assertIsNone(row["decoded_raw_rgb_pixel_sha256"])
            self.assertIsNone(row["calibration_values_sha256"])
            self.assertIsNone(row["warp_sha256"])
            self.assertIsNone(row["output_rgb"])
            self.assertIsNone(row["sampling_valid"])
            self.assertIsNone(row["conversion_metadata"])

        # The present first row uses the fake services; no service sees either
        # missing RGB path or its calibration bytes.
        self.assertEqual(camera.normalize_calls, [fixture.calibration_bytes[260]])
        self.assertEqual(camera.build_calls, [fixture.calibration_bytes[260]])
        self.assertEqual(camera.warp_calls, [fixture.calibration_bytes[260]])
        self.assertEqual(decoder.calls.count(raw_payload), 1)
        self.assertEqual(len(decoder.calls), 3)  # raw frame 0 and its two opaque PNG round trips
        self.assertEqual(len(writer.calls), 2)
        self.assertEqual([order for order, _ in writer.calls], ["RGB", "GRAY"])

        asset_bytes = b"opaque synthetic asset bytes"
        asset_receipt_bytes = b"opaque synthetic asset receipt bytes"
        _write(output_root, "assets/object.glb", asset_bytes)
        _write(output_root, "assets/receipt.json", asset_receipt_bytes)
        for ordinal, entry in enumerate(fixture.entries):
            source_id = entry["source_frame_id"]
            calibration_path = entry["calibration_path"]
            _write(output_root, calibration_path, fixture.calibration_bytes[source_id])
            if ordinal == 0:
                _write(output_root, entry["rgb_path"], raw_payload)

        input_bytes = self._input_manifest(fixture, receipt, asset_bytes, asset_receipt_bytes)
        verified = capture.load_capture_plan(output_root, input_bytes)
        self.assertEqual([row["source_frame_id"] for row in verified.rows], [260, 261, 262])
        self.assertEqual([row["timestamp_s"] for row in verified.rows], [
            1.0, float(Fraction(1_033_333_333, 1_000_000_000)),
            float(Fraction(1_066_666_666, 1_000_000_000)),
        ])
        self.assertEqual(verified.rows[1]["failure_reason"], "raw_missing")
        self.assertEqual(verified.rows[2]["failure_reason"], "raw_missing")
        self.assertEqual(verified.verified_resources[fixture.entries[1]["calibration_path"]],
                         fixture.entries[1]["calibration_sha256"])
        self.assertEqual(verified.verified_resources[fixture.entries[2]["calibration_path"]],
                         fixture.entries[2]["calibration_sha256"])

    def test_missing_rgb_does_not_bypass_hash_authentication_of_present_calibration(self):
        wrong_calibration = b"different bytes at the pinned calibration path"
        fixture = _SyntheticConversionFixture(
            self.root, count=1, calibration_override=(260, wrong_calibration),
            available_rgb_ids=frozenset())
        camera = _FakeCameraServices(
            invalid_calibration=b"never normalized",
            warp_failure_calibration=b"never warped")
        writer = _FakePngWriter()
        decoder = _FakeImageDecoder(b"opaque rgb bytes for source 260", writer)

        camera_patch, module_patch = self._install_fake_camera(camera)
        with camera_patch, module_patch:
            with self.assertRaisesRegex(capture.CaptureIntegrityError, "raw calibration hash mismatch"):
                capture.convert_capture_sequence(
                    fixture.source_root,
                    fixture.inventory_path,
                    fixture.bundle_root,
                    expected_inventory_sha256=_sha(fixture.inventory_bytes),
                    decoder=decoder,
                    writer=writer,
                )
        self.assertEqual(camera.normalize_calls, [])
        self.assertEqual(camera.build_calls, [])
        self.assertEqual(camera.warp_calls, [])
        self.assertEqual(decoder.calls, [])
        self.assertEqual(writer.calls, [])


if __name__ == "__main__":
    unittest.main()
