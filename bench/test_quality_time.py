"""Pure physical/legacy time-policy checks; no model or media execution."""

import hashlib
import json
import subprocess
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

from .quality_time import (
    LEGACY,
    PHYSICAL,
    LEGACY_POLICY,
    FrameKey,
    TimePolicy,
    parse_timing_manifest,
)


def _manifest_bytes(*, source_units="microseconds", timestamps=(1_000_000, 1_033_333),
                    timestamp_s=None, source_ids=(260, 262), nominal_fps=30,
                    source_gaps="derived", time_base=None, timeline_span_seconds=None):
    scale = {"seconds": 1, "milliseconds": 1_000, "microseconds": 1_000_000,
             "nanoseconds": 1_000_000_000}[source_units]
    converted = [value / scale for value in timestamps]
    if timestamp_s is not None:
        converted = list(timestamp_s)
    document = {
        "schema_version": 2,
        "clock": {
            "mode": PHYSICAL,
            "units": "seconds",
            "source_units": source_units,
            "timestamp_source": "declared capture metadata/PTS",
            "nominal_fps": nominal_fps,
        },
        "timeline": [
            {"frame_id": ordinal, "source_frame_id": source_id,
             "source_timestamp": source_time, "timestamp_s": seconds,
             "role": "setup" if ordinal == 0 else "scored"}
            for ordinal, (source_id, source_time, seconds) in enumerate(
                zip(source_ids, timestamps, converted)
            )
        ],
        "setup_frame_id": 0,
        "frame_ids": [1],
    }
    if source_gaps == "derived":
        gaps = [
            {"previous_source_frame_id": previous, "current_source_frame_id": current,
             "missing_capture_count": current - previous - 1}
            for previous, current in zip(source_ids, source_ids[1:])
            if current - previous > 1
        ]
        if gaps:
            document["source_gaps"] = gaps
    elif source_gaps is not None:
        document["source_gaps"] = source_gaps
    if time_base is not None:
        document["clock"]["time_base"] = time_base
    if timeline_span_seconds is not None:
        document["clock"]["timeline_span_seconds"] = timeline_span_seconds
    return json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _table_sha256(document):
    rows = [
        {key: row[key] for key in ("frame_id", "source_frame_id", "timestamp_s", "role")}
        for row in document["timeline"]
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


class TimePolicyTests(unittest.TestCase):
    def setUp(self):
        guard = mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("subprocess execution is forbidden in tests"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    def test_legacy_defaults_are_explicit_and_immutable(self):
        self.assertEqual(LEGACY, "legacy_nominal_60_v1")
        self.assertEqual(PHYSICAL, "capture_seconds_v1")
        self.assertEqual(LEGACY_POLICY.mode, LEGACY)
        self.assertEqual(LEGACY_POLICY.max_angular_rate_deg_s, 720.0)
        self.assertEqual(LEGACY_POLICY.max_translation_rate_m_s, 1.2)
        self.assertEqual(LEGACY_POLICY.private_pose_memory_s, 0.5)
        with self.assertRaises(FrozenInstanceError):
            LEGACY_POLICY.mode = PHYSICAL

    def test_policy_rejects_bool_nonfinite_and_nonpositive_limits(self):
        invalid = (
            {"mode": "unknown"},
            {"max_angular_rate_deg_s": True},
            {"max_translation_rate_m_s": float("nan")},
            {"private_pose_memory_s": float("inf")},
            {"max_angular_rate_deg_s": 0.0},
            {"max_translation_rate_m_s": -1.0},
            {"private_pose_memory_s": 0.0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                TimePolicy(**overrides)

    def test_frame_key_has_no_fps_or_unit_autodetection(self):
        key = FrameKey(7, 0.0, PHYSICAL)
        self.assertEqual((key.frame_id, key.timestamp_s, key.clock_mode), (7, 0.0, PHYSICAL))
        # Zero and negative values are valid clock origins; chronology checks
        # belong to the stream consumer, not FrameKey construction.
        self.assertEqual(FrameKey(0, -2.5, PHYSICAL).timestamp_s, -2.5)
        for bad in (True, -1, 1.25):
            with self.subTest(frame_id=bad), self.assertRaises((TypeError, ValueError)):
                FrameKey(bad)
        for bad in (None, float("nan"), float("inf"), "1.0"):
            with self.subTest(timestamp_s=bad), self.assertRaises((TypeError, ValueError)):
                FrameKey(0, bad, PHYSICAL)
        with self.assertRaises((TypeError, ValueError)):
            FrameKey(0, 1.0, LEGACY)

    def test_table_digest_oracle_is_independent_of_parser(self):
        # This fixture is consumed by the parser acceptance test below after
        # construction, using the spec's normalized row contract as oracle.
        timeline = [
            {"frame_id": 0, "source_frame_id": 260, "source_timestamp": 1234567890123,
             "timestamp_s": 1234567890123 / 1_000_000, "role": "setup"},
            {"frame_id": 1, "source_frame_id": 262, "source_timestamp": 1234567891123,
             "timestamp_s": 1234567891123 / 1_000_000, "role": "scored"},
        ]
        normalized = [
            {key: row[key] for key in ("frame_id", "source_frame_id", "timestamp_s", "role")}
            for row in timeline
        ]
        expected = hashlib.sha256(json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest().upper()
        self.assertEqual(len(expected), 64)
        self.assertEqual(expected, expected.upper())

    def test_physical_manifest_preserves_source_ids_and_table_digest(self):
        raw = _manifest_bytes()
        document = json.loads(raw)
        parsed = parse_timing_manifest(raw)
        self.assertEqual(parsed["clock_mode"], PHYSICAL)
        self.assertEqual(parsed["time_policy"], TimePolicy(mode=PHYSICAL))
        self.assertEqual(parsed["manifest_sha256"], hashlib.sha256(raw).hexdigest().upper())
        self.assertEqual(parsed["timestamp_table_sha256"], _table_sha256(document))
        self.assertEqual(parsed["setup_key"], FrameKey(0, 1.0, PHYSICAL))
        self.assertEqual(parsed["scored_keys"], (FrameKey(1, 1.033333, PHYSICAL),))
        self.assertEqual(parsed["source_frame_ids"], {0: 260, 1: 262})

    def test_legacy_manifest_is_explicit_and_has_no_physical_table_digest(self):
        raw = json.dumps({
            "schema_version": 1,
            "setup_frame_id": 1148,
            "frame_ids": [1149, 1150],
        }, separators=(",", ":")).encode("utf-8")
        parsed = parse_timing_manifest(raw)
        self.assertEqual(parsed["clock_mode"], LEGACY)
        self.assertEqual(parsed["time_policy"], LEGACY_POLICY)
        self.assertEqual(parsed["timestamp_table_sha256"], None)
        self.assertEqual(parsed["setup_key"], FrameKey(1148, None, LEGACY))
        self.assertEqual(parsed["scored_keys"], (
            FrameKey(1149, None, LEGACY), FrameKey(1150, None, LEGACY),
        ))
        self.assertEqual(parsed["source_frame_ids"], {1148: 1148, 1149: 1149, 1150: 1150})

    def test_all_declared_source_units_convert_using_explicit_units(self):
        cases = (
            ("seconds", (1.0, 1.25), (1.0, 1.25)),
            ("milliseconds", (1000, 1250), (1.0, 1.25)),
            ("microseconds", (1_000_000, 1_250_000), (1.0, 1.25)),
            ("nanoseconds", (1_000_000_000, 1_250_000_000), (1.0, 1.25)),
        )
        for units, source, expected in cases:
            with self.subTest(source_units=units):
                parsed = parse_timing_manifest(_manifest_bytes(
                    source_units=units, timestamps=source, timestamp_s=expected,
                ))
                self.assertEqual([key.timestamp_s for key in parsed["scored_keys"]], [1.25])

    def test_physical_clock_accepts_zero_and_negative_origins(self):
        parsed = parse_timing_manifest(_manifest_bytes(
            source_units="seconds", timestamps=(-1.0, 0.0),
            timestamp_s=(-1.0, 0.0), source_ids=(900, 901),
        ))
        self.assertEqual(parsed["setup_key"], FrameKey(0, -1.0, PHYSICAL))
        self.assertEqual(parsed["scored_keys"][0], FrameKey(1, 0.0, PHYSICAL))

    def test_nominal_fps_does_not_change_physical_dt_or_table_identity(self):
        at_30 = parse_timing_manifest(_manifest_bytes(nominal_fps=30))
        at_60 = parse_timing_manifest(_manifest_bytes(nominal_fps=60))
        self.assertEqual(at_30["time_policy"], at_60["time_policy"])
        self.assertEqual(at_30["scored_keys"], at_60["scored_keys"])
        self.assertEqual(at_30["timestamp_table_sha256"], at_60["timestamp_table_sha256"])
        self.assertNotEqual(at_30["manifest_sha256"], at_60["manifest_sha256"])

    def test_source_gap_records_are_exact_and_not_inferred_as_rows(self):
        parsed = parse_timing_manifest(_manifest_bytes())
        self.assertEqual(parsed["source_frame_ids"], {0: 260, 1: 262})
        missing = json.loads(_manifest_bytes())
        del missing["source_gaps"]
        wrong_count = json.loads(_manifest_bytes())
        wrong_count["source_gaps"][0]["missing_capture_count"] = 2
        extra = json.loads(_manifest_bytes())
        extra["source_gaps"].append({
            "previous_source_frame_id": 100,
            "current_source_frame_id": 101,
            "missing_capture_count": 0,
        })
        reordered = json.loads(_manifest_bytes(
            source_ids=(260, 262, 265),
            timestamps=(1_000_000, 1_033_333, 1_066_666),
            timestamp_s=(1.0, 1.033333, 1.066666),
        ))
        reordered["frame_ids"] = [1, 2]
        reordered["source_gaps"].reverse()
        for candidate in (missing, wrong_count, extra, reordered):
            with self.subTest(source_gaps=candidate.get("source_gaps")), self.assertRaises(ValueError):
                parse_timing_manifest(json.dumps(candidate).encode("utf-8"))
        contiguous = parse_timing_manifest(_manifest_bytes(source_ids=(260, 261)))
        self.assertEqual(contiguous["source_frame_ids"], {0: 260, 1: 261})

    def test_optional_time_base_and_span_are_consistency_checks(self):
        accepted = parse_timing_manifest(_manifest_bytes(
            time_base={"num": 1, "den": 1_000_000},
            timeline_span_seconds=0.0333335,
        ))
        self.assertEqual(accepted["clock_mode"], PHYSICAL)
        for time_base in (
            {"num": 1, "den": 1000},
            {"num": True, "den": 1_000_000},
            {"num": 1, "den": 0},
            {"num": 1, "den": 1_000_000, "extra": 0},
        ):
            with self.subTest(time_base=time_base), self.assertRaises(ValueError):
                parse_timing_manifest(_manifest_bytes(time_base=time_base))
        with self.assertRaises(ValueError):
            parse_timing_manifest(_manifest_bytes(timeline_span_seconds=0.033335))

    def test_declared_conversion_mismatch_and_bad_clock_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_timing_manifest(_manifest_bytes(timestamp_s=(1.0, 1.034)))
        malformed = json.loads(_manifest_bytes())
        malformed["clock"]["units"] = "milliseconds"
        with self.assertRaises(ValueError):
            parse_timing_manifest(json.dumps(malformed).encode("utf-8"))
        malformed = json.loads(_manifest_bytes())
        malformed["clock"]["source_units"] = "frames"
        with self.assertRaises(ValueError):
            parse_timing_manifest(json.dumps(malformed).encode("utf-8"))

    def test_timeline_rejects_duplicate_ids_nonmonotonic_time_and_role_mismatch(self):
        mutations = []
        document = json.loads(_manifest_bytes())
        duplicate_source = json.loads(json.dumps(document))
        duplicate_source["timeline"][1]["source_frame_id"] = duplicate_source["timeline"][0]["source_frame_id"]
        mutations.append(duplicate_source)
        duplicate_ordinal = json.loads(json.dumps(document))
        duplicate_ordinal["timeline"][1]["frame_id"] = 0
        mutations.append(duplicate_ordinal)
        nonmonotonic = json.loads(json.dumps(document))
        nonmonotonic["timeline"][1]["timestamp_s"] = 0.5
        mutations.append(nonmonotonic)
        bad_role = json.loads(json.dumps(document))
        bad_role["timeline"][1]["role"] = "setup"
        mutations.append(bad_role)
        wrong_scored = json.loads(json.dumps(document))
        wrong_scored["frame_ids"] = [0]
        mutations.append(wrong_scored)
        for candidate in mutations:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                parse_timing_manifest(json.dumps(candidate).encode("utf-8"))

    def test_parser_rejects_bool_ids_bad_unit_types_oversized_ticks_and_bool_schema(self):
        mutations = []

        bad_frame_id = json.loads(_manifest_bytes())
        bad_frame_id["timeline"][0]["frame_id"] = True
        mutations.append(bad_frame_id)

        bad_schema = json.loads(_manifest_bytes())
        bad_schema["schema_version"] = True
        mutations.append(bad_schema)
        fractional_schema = json.loads(_manifest_bytes())
        fractional_schema["schema_version"] = 2.0
        mutations.append(fractional_schema)

        missing_setup_id = json.loads(_manifest_bytes())
        missing_setup_id.pop("setup_frame_id")
        mutations.append(missing_setup_id)

        for invalid_units in ([], {}, None, 1):
            bad_units = json.loads(_manifest_bytes())
            bad_units["clock"]["source_units"] = invalid_units
            mutations.append(bad_units)

        oversized_tick = json.loads(_manifest_bytes())
        oversized_tick["timeline"][0]["source_timestamp"] = 10 ** 1000
        mutations.append(oversized_tick)

        for candidate in mutations:
            raw = json.dumps(candidate, separators=(",", ":"), allow_nan=False).encode("utf-8")
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                parse_timing_manifest(raw)

        huge_exponent = _manifest_bytes().replace(
            b'"source_timestamp":1000000', b'"source_timestamp":1e999', 1,
        )
        with self.assertRaises(ValueError):
            parse_timing_manifest(huge_exponent)

    def test_parser_enforces_raw_byte_and_timeline_limits_before_acceptance(self):
        oversized = b" " * (2 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            parse_timing_manifest(oversized)
        document = json.loads(_manifest_bytes(
            source_units="seconds", timestamps=(0, 1), timestamp_s=(0, 1),
            source_ids=(0, 1),
        ))
        row = dict(document["timeline"][1])
        document["timeline"] = [
            {**row, "frame_id": i, "source_frame_id": i,
             "source_timestamp": i, "timestamp_s": i,
             "role": "setup" if i == 0 else "scored"}
            for i in range(10_001)
        ]
        document["setup_frame_id"] = 0
        document["frame_ids"] = list(range(1, 10_001))
        document.pop("source_gaps", None)
        with self.assertRaises(ValueError):
            parse_timing_manifest(json.dumps(document, separators=(",", ":")).encode("utf-8"))

    def test_timeline_limit_includes_exactly_ten_thousand_rows(self):
        rows = [
            {"frame_id": i, "source_frame_id": i, "source_timestamp": i,
             "timestamp_s": float(i), "role": "setup" if i == 0 else "scored"}
            for i in range(10_000)
        ]
        document = {
            "schema_version": 2,
            "clock": {"mode": PHYSICAL, "units": "seconds", "source_units": "seconds",
                      "timestamp_source": "declared capture metadata/PTS", "nominal_fps": 30},
            "timeline": rows,
            "setup_frame_id": 0,
            "frame_ids": list(range(1, 10_000)),
        }
        raw = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(raw), 2 * 1024 * 1024)
        result = parse_timing_manifest(raw)
        self.assertEqual(len(result["scored_keys"]), 9_999)

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        valid = _manifest_bytes()
        duplicate = valid.replace(b'"schema_version":2,',
                                  b'"schema_version":2,"schema_version":2,', 1)
        nonfinite = valid.replace(b'"timestamp_s":1.033333', b'"timestamp_s":NaN')
        for raw in (duplicate, nonfinite):
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                parse_timing_manifest(raw)


if __name__ == "__main__":
    unittest.main()
