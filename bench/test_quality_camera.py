"""Independent synthetic checks for the HOT3D FISHEYE624 image warp.

The suite uses only synthetic rays and in-memory RGB arrays. Its small upstream
oracle closure is AST-extracted from authenticated, pinned text after the
license and all source hashes have been checked; the toolkit is never imported.
Equation, validity-domain, and sampling behavior live in separate test classes
so their evidence remains distinguishable in the guarded harness receipt.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_camera as camera


ROOT = Path(__file__).resolve().parents[1]
ORACLE_ID = "6059fa325e3048e3a3be941ea7672883"
ORACLE_REVISION = "bc628e9286c18444b47a0971a89d075ea6f00747"
ORACLE_ROOT = (
    ROOT
    / ".cache/model-quality/generalization-metadata/camera-oracle"
    / ORACLE_REVISION
    / ORACLE_ID
)
ORACLE_PINS = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "hand_tracking_toolkit/camera.py": (
        "20565a0294208f6472e72216db258bef96f33f5e0c78c70ea0e72e7b1431a400"
    ),
    "hand_tracking_toolkit/camera_distortion.py": (
        "83815230ee4401c1e45208149263f1606164452e710e0090179b18d2ef5cd047"
    ),
    "hand_tracking_toolkit/math_utils.py": (
        "1209672b2c59e57f9cd2a8a3252637cf39e65bbd5fcabbe7e47beee635a362c9"
    ),
}
RECEIPT_PATH = (
    ROOT
    / ".cache/experiment-history/receipts"
    / f"camera-source-{ORACLE_ID}.json"
)


def _record(
    *,
    width=1200,
    height=900,
    fx=420.0,
    fy=510.0,
    cx=604.25,
    cy=417.75,
    coefficients=None,
    layout=16,
    theta=1.0,
    valid_radius=None,
    **extra,
):
    coeffs = list(coefficients if coefficients is not None else [0.0] * 12)
    if layout == 15:
        params = [fx, cx, cy, *coeffs]
    elif layout == 16:
        params = [fx, fy, cx, cy, *coeffs]
    else:
        raise AssertionError("test fixture layout must be 15 or 16")
    result = {
        "image_width": width,
        "image_height": height,
        "projection_model_type": "CameraModelType.FISHEYE624",
        "projection_params": params,
        "max_solid_angle": theta,
        "valid_radius": valid_radius,
        "label": "synthetic-camera",
        "serial_number": "synthetic-serial",
        "stream_id": "214-1",
    }
    result.update(extra)
    return result


def _rays_for_equations():
    # All rays are asymmetric in x/y, front-facing, and well away from zero.
    return np.asarray(
        [
            [0.58, -0.31, 0.62],
            [-0.42, 0.27, 0.71],
            [0.19, 0.53, 0.66],
            [-0.63, -0.22, 0.58],
        ],
        dtype=np.float64,
    )


def _scalar_624(ray, cal):
    """Independent scalar statement of the angular and OVR624 equations."""

    x, y, z = (float(value) for value in ray)
    rho = math.hypot(x, y)
    angle = math.atan2(rho, z)
    if rho == 0.0:
        ax = ay = 0.0
    else:
        ax = x * angle / rho
        ay = y * angle / rho
    r2 = min(ax * ax + ay * ay, math.pi * math.pi)
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r4 * r4
    r10 = r8 * r2
    r12 = r6 * r6
    k1, k2, k3, k4, k5, k6 = cal.radial
    scale = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
    xr, yr = ax * scale, ay * scale
    radial_r2 = xr * xr + yr * yr
    radial_r4 = radial_r2 * radial_r2
    tx = xr + 2.0 * cal.p2 * xr * yr + cal.p1 * (radial_r2 + 2.0 * xr * xr)
    ty = yr + 2.0 * cal.p1 * xr * yr + cal.p2 * (radial_r2 + 2.0 * yr * yr)
    xd = tx + cal.prism[0] * radial_r2 + cal.prism[1] * radial_r4
    yd = ty + cal.prism[2] * radial_r2 + cal.prism[3] * radial_r4
    return np.asarray([cal.fx * xd + cal.cx, cal.fy * yd + cal.cy])


def _expected_domain(ray, uv, cal):
    if not np.isfinite(ray).all() or not np.any(ray) or ray[2] <= 0.0:
        return False
    angle = math.atan2(math.hypot(float(ray[0]), float(ray[1])), float(ray[2]))
    u, v = (float(value) for value in uv)
    in_rect = -0.5 <= u <= cal.image_width - 0.5 and -0.5 <= v <= cal.image_height - 0.5
    in_disk = cal.valid_radius is None or math.hypot(u - cal.cx, v - cal.cy) <= cal.valid_radius
    return angle <= cal.max_solid_angle and in_rect and in_disk


def _method_node(tree, class_name, method_name):
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    method.returns = None
    method.name = "oracle_" + method_name
    return method


def _load_authenticated_upstream_oracle():
    """Execute only three exact, independently pinned projection methods."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        receipt.get("id") != ORACLE_ID
        or receipt.get("source_revision") != ORACLE_REVISION
        or receipt.get("source_only") is not True
        or receipt.get("code_executed") is not False
        or receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
    ):
        raise AssertionError("pinned oracle acquisition receipt is not an authenticated source-only pass")
    rows = {row.get("path"): row for row in receipt.get("rows", [])}
    if set(rows) != set(ORACLE_PINS):
        raise AssertionError("pinned oracle receipt does not contain the exact four-file closure")

    source_text = {}
    for relative_path, expected_sha256 in ORACLE_PINS.items():
        path = ORACLE_ROOT / Path(relative_path)
        content = path.read_bytes()
        observed_sha256 = hashlib.sha256(content).hexdigest()
        row = rows[relative_path]
        if (
            observed_sha256 != expected_sha256
            or row.get("sha256") != expected_sha256
            or row.get("bytes") != len(content)
            or row.get("status") != "acquired"
        ):
            raise AssertionError(f"pinned upstream text failed its receipt/SHA check: {relative_path}")
        source_text[relative_path] = content.decode("utf-8")
    if sum(row["bytes"] for row in rows.values()) != 45_696:
        raise AssertionError("pinned upstream text closure byte count changed")

    camera_tree = ast.parse(source_text["hand_tracking_toolkit/camera.py"])
    distortion_tree = ast.parse(source_text["hand_tracking_toolkit/camera_distortion.py"])
    methods = [
        _method_node(camera_tree, "CameraModel", "eye_to_window"),
        _method_node(distortion_tree, "ArctanProjection", "project"),
        _method_node(distortion_tree, "OVR624Distortion", "evaluate"),
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "math": math,
        "cast": lambda _target_type, value: value,
        "_StoredDistortion": object,
    }
    exec(compile(ast.fix_missing_locations(module), "<pinned HOT3D camera methods>", "exec"), namespace)
    return namespace["oracle_eye_to_window"], namespace["oracle_project"], namespace["oracle_evaluate"]


def _oracle_pixels(rays, cal, oracle):
    eye_to_window, project, evaluate = oracle
    params = [
        *cal.radial,
        cal.p1,
        cal.p2,
        *cal.prism,
    ]

    class Distortion:
        def evaluate(self, points):
            return evaluate(params, np.array(points, dtype=np.float64, copy=True))

    synthetic_camera = SimpleNamespace(
        project=project,
        distort=Distortion(),
        f=np.asarray([cal.fx, cal.fy], dtype=np.float64),
        c=np.asarray([cal.cx, cal.cy], dtype=np.float64),
    )
    return eye_to_window(synthetic_camera, np.asarray(rays, dtype=np.float64))


def _array_digest(name, value, dtype):
    canonical = np.asarray(value, dtype=np.dtype(dtype), order="C")
    digest = hashlib.sha256()
    digest.update(name.encode("ascii") + b"\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(dtype.encode("ascii") + b"\0")
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _warp_map_digest(map_x, map_y):
    digest = hashlib.sha256()
    for name, value in (("map_x", map_x), ("map_y", map_y)):
        canonical = np.asarray(value, dtype="<f8", order="C")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(b"<f8\0")
        digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _resign_test_map(warp, map_x, map_y):
    """Bind a synthetic, contract-valid coordinate tweak for sampler tests."""

    map_digest = _warp_map_digest(map_x, map_y)
    warp_digest = _canonical_sha256(
        {
            "schema": "quality-camera-warp-v1",
            "calibration_canonical_sha256": warp.calibration.canonical_values_sha256,
            "map_sha256": map_digest,
            "geometric_valid_sha256": warp.geometric_valid_sha256,
            "domain_valid_sha256": warp.domain_valid_sha256,
            "sampling_valid_sha256": warp.sampling_valid_sha256,
            "output": {
                "width": 720,
                "height": 720,
                "fx": 360.0,
                "fy": 360.0,
                "cx": 359.5,
                "cy": 359.5,
                "rotation_output_from_source": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "pixel_center_convention": "integer-centers; edges[-0.5,size-0.5]",
            },
            "sampling": {
                "channel_order": "RGB",
                "interpolation": "bilinear-four-neighbor",
                "neighbor_footprint": "all four integer pixel centers must be in bounds and declared valid disk",
                "rounding": "floor(value+0.5)",
                "clip_range": [0, 255],
                "invalid_fill_rgb": [128, 128, 128],
            },
            "tile_rows": warp.tile_rows,
            "working_memory_limit_bytes": 64 * 1024 * 1024,
        }
    )
    map_x.setflags(write=False)
    map_y.setflags(write=False)
    return replace(warp, map_x=map_x, map_y=map_y, map_sha256=map_digest, warp_sha256=warp_digest)


class PinnedOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = _load_authenticated_upstream_oracle()

    def test_acquisition_receipt_and_exact_text_pins_are_authenticated_before_execution(self):
        # setUpClass has already refused to compile any selected AST method
        # unless the immutable source-only receipt and all four SHAs matched.
        self.assertEqual(camera.UPSTREAM_REVISION, ORACLE_REVISION)
        self.assertEqual(camera.UPSTREAM_CAMERA_SHA256, ORACLE_PINS["hand_tracking_toolkit/camera.py"])
        self.assertEqual(camera.UPSTREAM_DISTORTION_SHA256, ORACLE_PINS["hand_tracking_toolkit/camera_distortion.py"])

    def test_both_parameter_layouts_match_pinned_camera_api_to_one_micro_pixel(self):
        coefficients = [
            0.021, -0.006, 0.0017, -0.00042, 0.00013, -0.000031,
            0.004, -0.007, 0.0023, -0.00051, -0.0031, 0.00072,
        ]
        rays = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.0, -0.3, 1.0],
                [0.58, -0.31, 0.62],
                [-0.42, 0.27, 0.71],
                [0.65, 0.38, 0.55],
                [0.001, -0.002, 1.0],
            ],
            dtype=np.float64,
        )
        for layout in (15, 16):
            with self.subTest(parameter_count=layout):
                record = _record(layout=layout, coefficients=coefficients)
                cal = camera.normalize_calibration(record)
                uv, geometric, _domain = camera.project_source_rays(rays, cal)
                oracle_uv = _oracle_pixels(rays, cal, self.oracle)
                self.assertTrue(geometric.all())
                self.assertTrue(np.isfinite(oracle_uv).all())
                self.assertLessEqual(float(np.max(np.abs(uv - oracle_uv))), 1e-6)

    def test_every_distortion_coefficient_independently_matches_scalar_and_pinned_oracles(self):
        # Ordering: k1..k6, p1,p2, s1..s4. The same asymmetric rays make
        # tangent-axis swaps and prism-axis swaps produce distinct errors.
        coefficient_names = (
            "k1", "k2", "k3", "k4", "k5", "k6",
            "p1", "p2", "s1", "s2", "s3", "s4",
        )
        rays = _rays_for_equations()
        for index, name in enumerate(coefficient_names):
            coefficients = [0.0] * 12
            coefficients[index] = 0.035 if index % 2 == 0 else -0.027
            with self.subTest(coefficient=name):
                cal = camera.normalize_calibration(
                    _record(layout=16, coefficients=coefficients, fx=380.0, fy=470.0)
                )
                observed, geometric, _domain = camera.project_source_rays(rays, cal)
                scalar = np.vstack([_scalar_624(ray, cal) for ray in rays])
                oracle = _oracle_pixels(rays, cal, self.oracle)
                self.assertTrue(geometric.all())
                np.testing.assert_allclose(observed, scalar, rtol=0.0, atol=1e-10)
                self.assertLessEqual(float(np.max(np.abs(observed - oracle))), 1e-6)


class CalibrationTests(unittest.TestCase):
    def test_15_and_16_parameter_calibrations_keep_their_distinct_focal_layouts(self):
        shared = [0.0] * 12
        equal = camera.normalize_calibration(
            _record(layout=15, fx=333.25, fy=999.0, cx=17.5, cy=21.25, coefficients=shared)
        )
        anisotropic = camera.normalize_calibration(
            _record(layout=16, fx=333.25, fy=410.75, cx=17.5, cy=21.25, coefficients=shared)
        )
        self.assertEqual((equal.fx, equal.fy, equal.cx, equal.cy), (333.25, 333.25, 17.5, 21.25))
        self.assertEqual((anisotropic.fx, anisotropic.fy), (333.25, 410.75))
        self.assertEqual((anisotropic.cx, anisotropic.cy), (17.5, 21.25))
        self.assertEqual(equal.projection_model_type, "CameraModelType.FISHEYE624")
        self.assertEqual(anisotropic.projection_model_type, "CameraModelType.FISHEYE624")

    def test_exact_json_bytes_and_canonical_values_have_separate_digests(self):
        record = _record(layout=16, coefficients=[0.0] * 12)
        compact = json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")
        spaced = json.dumps(record, indent=2, allow_nan=False).encode("utf-8")
        compact_cal = camera.normalize_calibration(compact)
        spaced_cal = camera.normalize_calibration(spaced)
        self.assertEqual(compact_cal.raw_calibration_sha256, hashlib.sha256(compact).hexdigest())
        self.assertEqual(spaced_cal.raw_calibration_sha256, hashlib.sha256(spaced).hexdigest())
        self.assertNotEqual(compact_cal.raw_calibration_sha256, spaced_cal.raw_calibration_sha256)
        self.assertEqual(compact_cal.canonical_values_sha256, spaced_cal.canonical_values_sha256)
        metadata = camera.calibration_metadata(spaced_cal)
        self.assertEqual(metadata["raw_calibration_sha256"], hashlib.sha256(spaced).hexdigest())
        self.assertEqual(metadata["source_width"], 1200)
        self.assertEqual(metadata["source_height"], 900)
        self.assertEqual(metadata["label"], "synthetic-camera")
        self.assertEqual(metadata["serial_number"], "synthetic-serial")
        self.assertEqual(metadata["stream_id"], "214-1")
        self.assertEqual(metadata["thin_prism_s1_s2_s3_s4"], [0.0] * 4)
        json.dumps(metadata, allow_nan=False)

    def test_wrapped_records_preserve_extrinsics_in_raw_digest_without_using_them_as_projection(self):
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        shifted = [row[:] for row in identity]
        shifted[0][3] = 8.0
        first = _record(T_device_from_camera=identity)
        second = {"calibration": _record(T_device_from_camera=shifted)}
        cal_a = camera.normalize_calibration(first)
        cal_b = camera.normalize_calibration(second)
        rays = _rays_for_equations()
        uv_a, _, _ = camera.project_source_rays(rays, cal_a)
        uv_b, _, _ = camera.project_source_rays(rays, cal_b)
        np.testing.assert_array_equal(uv_a, uv_b)
        self.assertNotEqual(cal_a.raw_calibration_sha256, cal_b.raw_calibration_sha256)
        self.assertEqual(cal_a.canonical_values_sha256, cal_b.canonical_values_sha256)

    def test_frozen_calibration_record_rejects_field_mutation(self):
        cal = camera.normalize_calibration(_record())
        with self.assertRaises(FrozenInstanceError):
            cal.fx = 0.0

    def test_invalid_models_lengths_scalars_dimensions_and_annotation_fields_are_rejected(self):
        invalid_records = [
            _record(projection_model_type="CameraModelType.FISHEYE62"),
            _record(projection_model_type="fisheye624"),
            _record(projection_params=[1.0] * 14),
            _record(projection_params=[1.0] * 17),
            _record(projection_params=[True, *[0.0] * 15]),
            _record(fx=0.0),
            _record(fy=-1.0),
            _record(coefficients=[float("nan")] + [0.0] * 11),
            _record(coefficients=[float("inf")] + [0.0] * 11),
            _record(max_solid_angle=True),
            _record(max_solid_angle=0.0),
            _record(max_solid_angle=-0.1),
            _record(max_solid_angle=math.pi / 2),
            _record(max_solid_angle=float("inf")),
            _record(valid_radius=False),
            _record(valid_radius=0.0),
            _record(valid_radius=-1.0),
            _record(width=True),
            _record(width=0),
            _record(width=1409),
            _record(width=1408, height=1409),
            _record(hand_pose={"joints": []}),
            _record(world_from_camera=[[1.0] * 4] * 4),
            _record(label=17),
        ]
        for record in invalid_records:
            with self.subTest(record=record):
                with self.assertRaises(camera.CalibrationError):
                    camera.normalize_calibration(record)

    def test_invalid_extrinsics_json_duplicates_and_unbounded_rays_are_rejected(self):
        invalid = _record(T_device_from_camera=[[1.0, 2.0, 3.0]])
        with self.assertRaises(camera.CalibrationError):
            camera.normalize_calibration(invalid)
        duplicate = (
            b'{"image_width":1200,"image_width":1200,"image_height":900,'
            b'"projection_model_type":"CameraModelType.FISHEYE624",'
            b'"projection_params":[420,604.25,417.75,0,0,0,0,0,0,0,0,0,0,0,0],'
            b'"max_solid_angle":1.0}'
        )
        with self.assertRaises(camera.CalibrationError):
            camera.normalize_calibration(duplicate)
        with self.assertRaises(camera.CalibrationError):
            camera.normalize_calibration(b'{"image_width":1e999}')
        with self.assertRaises(camera.CalibrationError):
            camera.normalize_calibration(b" " * (camera.MAX_CALIBRATION_BYTES + 1))

        cal = camera.normalize_calibration(_record())
        too_many = np.broadcast_to(np.zeros((1, 3), dtype=np.float32), (camera.MAX_SOURCE_PIXELS + 1, 3))
        with self.assertRaises(ValueError):
            camera.project_source_rays(too_many, cal)


class CameraNormalizedProvenanceTests(unittest.TestCase):
    def test_valid_normalized_reuse_retains_identity_and_raw_bytes_hash(self):
        raw = json.dumps(_record()).encode('utf-8')
        value = camera.normalize_calibration(raw)
        self.assertIs(camera.normalize_calibration(value), value)
        self.assertEqual(value.raw_calibration_sha256, hashlib.sha256(raw).hexdigest())

    def test_replaced_valid_values_cannot_keep_stale_canonical_provenance(self):
        value = camera.normalize_calibration(_record())
        for field, altered in (('fx', value.fx + 1), ('cx', value.cx + 0.25),
                               ('valid_radius', 100.0), ('label', 'another camera'),
                               ('stream_id', 'another stream')):
            with self.subTest(field=field), self.assertRaises(camera.CalibrationError):
                camera.normalize_calibration(replace(value, **{field: altered}))

    def test_constructed_invalid_fields_reject_before_any_map_allocation(self):
        valid = camera.normalize_calibration(_record())
        changes = (('image_width', True), ('image_height', 1409),
                   ('projection_model_type', 'CameraModelType.PINHOLE'),
                   ('fx', 0.0), ('fy', float('nan')), ('cx', True),
                   ('max_solid_angle', math.pi / 2), ('valid_radius', -1.0),
                   ('radial', list(valid.radial)), ('radial', (0.0,) * 5),
                   ('prism', (0.0, float('inf'), 0.0, 0.0)), ('p1', False),
                   ('serial_number', 17), ('canonical_values_sha256', 'bad'),
                   ('canonical_values_sha256', '0' * 64), ('raw_calibration_sha256', 'not a hash'))
        for field, altered in changes:
            with self.subTest(field=field), patch.object(camera.np, 'empty',
                    side_effect=AssertionError('invalid calibration reached map allocation')), \
                    patch.object(camera.np, 'full', side_effect=AssertionError('invalid calibration reached full allocation')), \
                    patch.object(camera.np, 'zeros', side_effect=AssertionError('invalid calibration reached zeros allocation')):
                with self.assertRaises(camera.CalibrationError):
                    # Direct construction tests the public dataclass boundary;
                    # no parser creates or signs this changed record.
                    values = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
                    values[field] = altered
                    constructed = camera.NormalizedCalibration(**values)
                    camera.build_warp(constructed)

    def test_every_projection_and_metadata_entry_revalidates_records(self):
        valid = camera.normalize_calibration(_record())
        calls = (camera.normalize_calibration, camera.calibration_metadata,
                 lambda cal: camera.project_source_rays(np.asarray([[0.0, 0.0, 1.0]]), cal),
                 camera.build_warp)
        for call in calls:
            with self.subTest(call=call), self.assertRaises(camera.CalibrationError):
                call(replace(valid, fx=valid.fx + 2))

    def test_subclasses_reject_and_numeric_values_rematerialize_canonically(self):
        valid = camera.normalize_calibration(_record())
        class DerivedCalibration(camera.NormalizedCalibration):
            pass
        values = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
        with self.assertRaises(camera.CalibrationError):
            camera.normalize_calibration(DerivedCalibration(**values))
        mixed = replace(valid, fx=int(valid.fx), fy=int(valid.fy))
        normalized = camera.normalize_calibration(mixed)
        self.assertIs(type(normalized.fx), float)
        self.assertIs(type(normalized.fy), float)
        self.assertEqual(normalized.raw_calibration_sha256, valid.raw_calibration_sha256)
        self.assertEqual(normalized.canonical_values_sha256, valid.canonical_values_sha256)

    def test_oversized_constructed_labels_reject_before_canonical_serialization(self):
        valid = camera.normalize_calibration(_record())
        for field in ('label', 'serial_number', 'stream_id'):
            altered = replace(valid, **{field: 'x' * (camera.MAX_CALIBRATION_BYTES + 1)})
            with self.subTest(field=field), patch.object(camera, '_canonical_json',
                    side_effect=AssertionError('oversized record reached canonical serialization')):
                with self.assertRaises(camera.CalibrationError):
                    camera.normalize_calibration(altered)

    def test_oversized_canonical_metadata_rejects_before_json_decode(self):
        oversized = b' ' * (camera.MAX_CALIBRATION_BYTES + 1)
        with patch.object(camera.json, 'loads', side_effect=AssertionError('oversized metadata reached JSON decoder')):
            with self.assertRaises(camera.WarpError):
                camera._decode_metadata_json(oversized)


class CameraEquationTests(unittest.TestCase):
    def test_equidistant_zero_distortion_optical_axis_and_near_axis_continuity(self):
        cal = camera.normalize_calibration(_record(coefficients=[0.0] * 12))
        rays = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1e-14, -2e-14, 1.0],
                [0.3, -0.2, 0.9],
                [-0.5, 0.4, 0.8],
            ],
            dtype=np.float64,
        )
        uv, geometric, domain = camera.project_source_rays(rays, cal)
        expected = np.vstack([_scalar_624(ray, cal) for ray in rays])
        self.assertTrue(geometric.all())
        self.assertTrue(domain.all())
        np.testing.assert_allclose(uv, expected, rtol=0.0, atol=1e-11)
        self.assertEqual(tuple(uv[0]), (cal.cx, cal.cy))
        self.assertLess(abs(uv[1, 0] - cal.cx), 1e-10)
        self.assertLess(abs(uv[1, 1] - cal.cy), 1e-10)

    def test_projection_is_direction_only_across_safe_and_extreme_ray_scales(self):
        cal = camera.normalize_calibration(_record(coefficients=[0.0] * 12))
        ray = np.asarray([0.43, -0.28, 0.81], dtype=np.float64)
        scales = [1e-250, 0.01, 1.0, 100.0, 1e250]
        batch = np.vstack([ray * scale for scale in scales])
        uv, geometric, domain = camera.project_source_rays(batch, cal)
        self.assertTrue(geometric.all())
        self.assertTrue(domain.all())
        for point in uv[1:]:
            np.testing.assert_allclose(point, uv[2], rtol=0.0, atol=1e-10)

    def test_resize_and_crop_follow_integer_pixel_center_intrinsics(self):
        ray = np.asarray([[0.38, -0.24, 0.91]], dtype=np.float64)
        original = camera.normalize_calibration(
            _record(layout=15, width=640, height=480, fx=503.0, cx=311.25, cy=229.75)
        )
        crop_left, crop_top, scale = 16.0, 12.0, 0.5
        cropped_resized = camera.normalize_calibration(
            _record(
                layout=15,
                width=312,
                height=228,
                fx=503.0 * scale,
                cx=(311.25 - crop_left + 0.5) * scale - 0.5,
                cy=(229.75 - crop_top + 0.5) * scale - 0.5,
            )
        )
        original_uv, original_geo, _ = camera.project_source_rays(ray, original)
        transformed_uv, transformed_geo, _ = camera.project_source_rays(ray, cropped_resized)
        self.assertTrue(original_geo.all() and transformed_geo.all())
        expected_uv = np.asarray(
            [[(original_uv[0, 0] - crop_left + 0.5) * scale - 0.5,
              (original_uv[0, 1] - crop_top + 0.5) * scale - 0.5]]
        )
        np.testing.assert_allclose(transformed_uv, expected_uv, rtol=0.0, atol=1e-10)

    def test_projector_rejects_nonarrays_shapes_and_nonreal_types(self):
        cal = camera.normalize_calibration(_record())
        bad_inputs = (
            [[0.0, 0.0, 1.0]],
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((1, 3), dtype=np.bool_),
            np.zeros((1, 3), dtype=np.complex128),
            np.zeros((1, 3), dtype=object),
        )
        for rays in bad_inputs:
            with self.subTest(dtype=getattr(rays, "dtype", type(rays))):
                with self.assertRaises(ValueError):
                    camera.project_source_rays(rays, cal)


class CameraDomainTests(unittest.TestCase):
    def test_geometric_validity_is_separate_from_cone_rect_and_disk_domain(self):
        rays = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [math.sin(0.9), 0.0, math.cos(0.9)],
                [math.sin(1.0001), 0.0, math.cos(1.0001)],
                [math.sin(0.1), 0.0, math.cos(0.1)],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [float("nan"), 0.0, 1.0],
                [float("inf"), 1.0, 1.0],
                [math.sin(0.1), 0.0, math.cos(0.1)],
            ],
            dtype=np.float64,
        )
        record = _record(width=1000, height=1000, fx=100.0, fy=100.0,
                         cx=500.0, cy=500.0, theta=1.0)
        cal = camera.normalize_calibration(record)
        uv, geometric, domain = camera.project_source_rays(rays, cal)
        np.testing.assert_array_equal(
            geometric,
            np.asarray([True, True, True, True, False, False, False, False, True]),
        )
        # 0.9 radians is inside a one-radian polar cone even though a mistaken
        # steradian interpretation would impose a much smaller half-angle.
        self.assertTrue(domain[1])
        self.assertFalse(domain[2])
        self.assertTrue(domain[3])
        self.assertTrue(domain[8])
        for index in range(4):
            self.assertEqual(bool(domain[index]), _expected_domain(rays[index], uv[index], cal))
        self.assertTrue(np.isnan(uv[4:8]).all())

        disk_cal = camera.normalize_calibration(
            _record(width=1000, height=1000, fx=100.0, fy=100.0,
                    cx=500.0, cy=500.0, theta=1.0, valid_radius=5.0)
        )
        disk_uv, disk_geometric, disk_domain = camera.project_source_rays(rays[[3]], disk_cal)
        self.assertTrue(disk_geometric[0])
        self.assertTrue(np.isfinite(disk_uv[0]).all())
        self.assertFalse(disk_domain[0])

    def test_theta_and_valid_disk_boundaries_are_inclusive_and_independent(self):
        theta = 0.7
        cal = camera.normalize_calibration(
            _record(width=1000, height=1000, fx=100.0, fy=100.0, cx=500.0, cy=500.0,
                    theta=theta, valid_radius=None)
        )
        rays = np.asarray(
            [
                [math.sin(theta), 0.0, math.cos(theta)],
                [math.sin(theta + 1e-6), 0.0, math.cos(theta + 1e-6)],
                [math.sin(0.2), 0.0, math.cos(0.2)],
            ],
            dtype=np.float64,
        )
        uv, geometric, domain = camera.project_source_rays(rays, cal)
        self.assertTrue(geometric.all())
        self.assertTrue(domain[0])
        self.assertFalse(domain[1])
        self.assertTrue(domain[2])
        self.assertAlmostEqual(float(uv[0, 0]), cal.cx + 70.0, places=10)

        disk_edge = camera.normalize_calibration(
            _record(width=40, height=40, fx=10.0, fy=12.0, cx=20.0, cy=20.0,
                    theta=1.0, valid_radius=2.0)
        )
        disk_rays = np.asarray(
            [[math.sin(0.2), 0.0, math.cos(0.2)],
             [math.sin(0.20001), 0.0, math.cos(0.20001)]],
            dtype=np.float64,
        )
        disk_uv, disk_geometric, disk_domain = camera.project_source_rays(disk_rays, disk_edge)
        self.assertTrue(disk_geometric.all())
        self.assertAlmostEqual(float(disk_uv[0, 0] - disk_edge.cx), 2.0, places=10)
        self.assertTrue(disk_domain[0])
        self.assertFalse(disk_domain[1])


class WarpGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One sizeable but bounded, entirely synthetic rectangular sensor is
        # shared by orientation, tile, digest, and RGB sampler tests.
        cls.record = _record(
            width=1408,
            height=1000,
            fx=500.0,
            fy=450.0,
            cx=700.25,
            cy=498.75,
            theta=1.0,
            coefficients=[0.0] * 12,
        )
        cls.warp = camera.build_warp(cls.record, tile_rows=16)

    def test_fixed_output_camera_rotation_and_camera_object_pose_composition(self):
        metadata = camera.warp_metadata(self.warp)
        output = metadata["output"]
        rotation = np.asarray(output["rotation_output_from_source"]).reshape(3, 3)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=0.0)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=15)
        np.testing.assert_array_equal(rotation @ np.asarray([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(rotation @ np.asarray([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0])
        self.assertEqual((output["fx"], output["fy"], output["cx"], output["cy"]),
                         (360.0, 360.0, 359.5, 359.5))
        self.assertEqual(output["pixel_center_convention"], "integer-centers; edges[-0.5,size-0.5]")

        # A source-camera-to-world pose right-multiplies the inverse camera
        # rotation, while an object-to-source-camera pose is left-multiplied.
        angle = 0.31
        world_from_source = np.eye(4, dtype=np.float64)
        world_from_source[:3, :3] = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        world_from_source[:3, 3] = [1.2, -0.7, 0.4]
        source_from_output = np.eye(4, dtype=np.float64)
        source_from_output[:3, :3] = rotation.T
        world_from_output = world_from_source @ source_from_output
        object_point = np.asarray([0.2, -0.1, 2.0, 1.0])
        source_from_object = np.eye(4, dtype=np.float64)
        source_from_object[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        source_from_object[:3, 3] = [0.1, 0.2, 0.3]
        output_from_source = np.eye(4, dtype=np.float64)
        output_from_source[:3, :3] = rotation
        output_from_object = output_from_source @ source_from_object
        np.testing.assert_allclose(
            world_from_output @ output_from_object @ object_point,
            world_from_source @ source_from_object @ object_point,
            rtol=0.0,
            atol=1e-15,
        )

    def test_destination_ray_orientation_and_half_pixel_center_are_independently_reprojected(self):
        cal = camera.normalize_calibration(self.record)
        rotation = np.asarray(camera.PinholeOutput().rotation_output_from_source).reshape(3, 3)
        destinations = [(359, 359), (360, 360), (359, 500), (500, 359), (64, 101), (655, 619)]
        for v, u in destinations:
            with self.subTest(output_pixel=(u, v)):
                output_ray = np.asarray(
                    [(u - 359.5) / 360.0, (v - 359.5) / 360.0, 1.0], dtype=np.float64
                )
                source_ray = rotation.T @ output_ray
                source_uv = _scalar_624(source_ray, cal)
                observed = np.asarray([self.warp.map_x[v, u], self.warp.map_y[v, u]])
                np.testing.assert_allclose(observed, source_uv, rtol=0.0, atol=1e-10)

                angular_x = (source_uv[0] - cal.cx) / cal.fx
                angular_y = (source_uv[1] - cal.cy) / cal.fy
                angle = math.hypot(angular_x, angular_y)
                sinc = 1.0 if angle == 0.0 else math.sin(angle) / angle
                recovered_source_ray = np.asarray(
                    [angular_x * sinc, angular_y * sinc, math.cos(angle)], dtype=np.float64
                )
                recovered_output_ray = rotation @ recovered_source_ray
                projected_output = np.asarray(
                    [360.0 * recovered_output_ray[0] / recovered_output_ray[2] + 359.5,
                     360.0 * recovered_output_ray[1] / recovered_output_ray[2] + 359.5]
                )
                self.assertLessEqual(float(np.max(np.abs(projected_output - [u, v]))), 0.5)

        # The positive source x ray points output-down; the positive source y
        # ray points output-left. This catches a transposed or double rotation.
        np.testing.assert_array_equal(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(rotation @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])

    def test_tile_bounds_content_digests_and_metadata_are_stable(self):
        tiled = camera.build_warp(self.record, tile_rows=32)
        self.assertEqual((self.warp.tile_rows, tiled.tile_rows), (16, 32))
        self.assertEqual(tiled.map_sha256, self.warp.map_sha256)
        self.assertEqual(tiled.geometric_valid_sha256, self.warp.geometric_valid_sha256)
        self.assertEqual(tiled.domain_valid_sha256, self.warp.domain_valid_sha256)
        self.assertEqual(tiled.sampling_valid_sha256, self.warp.sampling_valid_sha256)
        self.assertEqual(tiled.warp_sha256, camera.warp_metadata(tiled)["warp_sha256"])
        self.assertEqual(tiled.peak_working_bytes_limit, 64 * 1024 * 1024)
        metadata = camera.warp_metadata(self.warp)
        self.assertEqual(metadata["map_sha256"], _warp_map_digest(self.warp.map_x, self.warp.map_y))
        self.assertEqual(metadata["geometric_valid_sha256"], _array_digest("geometric_valid", self.warp.geometric_valid, "|u1"))
        self.assertEqual(metadata["domain_valid_sha256"], _array_digest("domain_valid", self.warp.domain_valid, "|u1"))
        self.assertEqual(metadata["sampling_valid_sha256"], _array_digest("sampling_valid", self.warp.sampling_valid, "|u1"))
        self.assertEqual(metadata["output"]["rotation_output_from_source"], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(metadata["sampling"]["rounding"], "floor(value+0.5)")
        self.assertEqual(metadata["sampling"]["invalid_fill_rgb"], [128, 128, 128])
        json.dumps(metadata, allow_nan=False)

    def test_warp_records_and_maps_are_immutable_and_digest_tampering_is_detected(self):
        with self.assertRaises(FrozenInstanceError):
            self.warp.tile_rows = 1
        with self.assertRaises(ValueError):
            self.warp.map_x[0, 0] = 0.0
        self.assertFalse(self.warp.map_x.flags.writeable)
        self.assertFalse(self.warp.map_y.flags.writeable)
        self.assertFalse(self.warp.sampling_valid.flags.writeable)

        altered_x = self.warp.map_x.copy()
        altered_x[359, 359] += 0.25
        altered_x.setflags(write=False)
        tampered = replace(self.warp, map_x=altered_x)
        with self.assertRaises(camera.WarpError):
            camera.warp_metadata(tampered)

    def test_tile_rows_reject_bool_out_of_range_and_noninteger_values_before_mapping(self):
        for tile_rows in (True, 0, 33, 1.5, "16"):
            with self.subTest(tile_rows=tile_rows), self.assertRaises(ValueError):
                camera.build_warp(self.record, tile_rows=tile_rows)


class CameraSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Add a small disk-limited camera whose unsupported output pixels must
        # receive neutral RGB fill, plus the same wide map used for synthetic
        # checkerboard samples.
        cls.record = _record(
            width=1408,
            height=1000,
            fx=500.0,
            fy=450.0,
            cx=700.25,
            cy=498.75,
            theta=1.0,
            coefficients=[0.0] * 12,
        )
        cls.warp = camera.build_warp(cls.record, tile_rows=16)
        cls.disk_record = _record(
            width=6,
            height=6,
            fx=280.0,
            fy=280.0,
            cx=2.5,
            cy=2.5,
            theta=1.0,
            valid_radius=1.4,
            coefficients=[0.0] * 12,
        )
        cls.disk_warp = camera.build_warp(cls.disk_record, tile_rows=8)

    def test_sampling_map_matches_an_independent_full_four_neighbor_and_disk_check(self):
        cal = self.disk_warp.calibration
        expected = np.zeros((720, 720), dtype=np.bool_)
        for row, col in np.argwhere(self.disk_warp.domain_valid):
            row, col = int(row), int(col)
            x = float(self.disk_warp.map_x[row, col])
            y = float(self.disk_warp.map_y[row, col])
            x0, y0 = math.floor(x), math.floor(y)
            if not (0 <= x0 and 0 <= y0 and x0 + 1 < cal.image_width and y0 + 1 < cal.image_height):
                continue
            neighbors = ((x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1))
            if cal.valid_radius is not None and any(
                math.hypot(px - cal.cx, py - cal.cy) > cal.valid_radius
                for px, py in neighbors
            ):
                continue
            expected[row, col] = True
        np.testing.assert_array_equal(self.disk_warp.sampling_valid, expected)

        # Find a projected center that is inside the declared disk while at
        # least one of its bilinear footprint corners lies outside the disk.
        center_inside_but_footprint_outside = []
        for row, col in np.argwhere(self.disk_warp.domain_valid):
            x = float(self.disk_warp.map_x[row, col])
            y = float(self.disk_warp.map_y[row, col])
            x0, y0 = math.floor(x), math.floor(y)
            if not (0 <= x0 and 0 <= y0 and x0 + 1 < cal.image_width and y0 + 1 < cal.image_height):
                continue
            if math.hypot(x - cal.cx, y - cal.cy) > cal.valid_radius:
                continue
            if any(
                math.hypot(px - cal.cx, py - cal.cy) > cal.valid_radius
                for px, py in ((x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1))
            ):
                center_inside_but_footprint_outside.append((row, col))
                break
        self.assertTrue(center_inside_but_footprint_outside)
        row, col = center_inside_but_footprint_outside[0]
        self.assertTrue(self.disk_warp.domain_valid[row, col])
        self.assertFalse(self.disk_warp.sampling_valid[row, col])

    def test_synthetic_constant_rgb_and_neutral_fill_preserve_validity_separately(self):
        rgb = np.empty((6, 6, 3), dtype=np.uint8)
        rgb[...] = [23, 117, 241]
        warped = camera.warp_rgb(rgb, self.disk_warp)
        valid = warped.sampling_valid
        self.assertTrue(valid.any())
        self.assertTrue((~valid).any())
        np.testing.assert_array_equal(warped.rgb[valid], np.broadcast_to([23, 117, 241], (int(valid.sum()), 3)))
        np.testing.assert_array_equal(warped.rgb[~valid], np.broadcast_to([128, 128, 128], (int((~valid).sum()), 3)))
        self.assertFalse(warped.rgb.flags.writeable)
        self.assertFalse(warped.sampling_valid.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            warped.rgb = rgb
        self.assertEqual(warped.calibration_sha256, self.disk_warp.calibration.raw_calibration_sha256)
        self.assertEqual(warped.warp_sha256, self.disk_warp.warp_sha256)
        self.assertEqual(warped.sampling_valid_sha256, self.disk_warp.sampling_valid_sha256)
        self.assertEqual(warped.rgb_sha256, hashlib.sha256(memoryview(rgb).cast("B")).hexdigest())
        self.assertEqual(warped.output_sha256, hashlib.sha256(memoryview(warped.rgb).cast("B")).hexdigest())
        json.dumps(warped.metadata, allow_nan=False)

    def test_checkerboard_rgb_uses_all_four_values_and_rounds_half_up(self):
        row = np.arange(1000, dtype=np.uint16)[:, None]
        col = np.arange(1408, dtype=np.uint16)[None, :]
        red = np.broadcast_to(((row + col) & 1) * 255, (1000, 1408)).astype(np.uint8)
        green = np.broadcast_to(col % 256, (1000, 1408)).astype(np.uint8)
        blue = np.broadcast_to(row % 256, (1000, 1408)).astype(np.uint8)
        checker = np.stack((red, green, blue), axis=-1)
        warped = camera.warp_rgb(checker, self.warp)
        candidates = np.argwhere(warped.sampling_valid[350:370, 350:370])
        row_out, col_out = next(
            (int(r + 350), int(c + 350)) for r, c in candidates
            if 0.2 < self.warp.map_x[r + 350, c + 350] - math.floor(self.warp.map_x[r + 350, c + 350]) < 0.8
            and 0.2 < self.warp.map_y[r + 350, c + 350] - math.floor(self.warp.map_y[r + 350, c + 350]) < 0.8
        )
        x, y = float(self.warp.map_x[row_out, col_out]), float(self.warp.map_y[row_out, col_out])
        x0, y0 = math.floor(x), math.floor(y)
        dx, dy = x - x0, y - y0
        expected = (
            checker[y0, x0].astype(np.float64) * ((1.0 - dx) * (1.0 - dy))
            + checker[y0, x0 + 1].astype(np.float64) * (dx * (1.0 - dy))
            + checker[y0 + 1, x0].astype(np.float64) * ((1.0 - dx) * dy)
            + checker[y0 + 1, x0 + 1].astype(np.float64) * (dx * dy)
        )
        expected = np.floor(np.clip(expected, 0.0, 255.0) + 0.5).astype(np.uint8)
        np.testing.assert_array_equal(warped.rgb[row_out, col_out], expected)

        # An exact half-value exercises floor(value + 0.5) independently of
        # NumPy's platform-dependent ties-to-even round convention.
        map_x, map_y = self.warp.map_x.copy(), self.warp.map_y.copy()
        row_tie, col_tie = 359, 359
        map_x[row_tie, col_tie] = 10.5
        map_y[row_tie, col_tie] = 10.5
        tie_warp = _resign_test_map(self.warp, map_x, map_y)
        tie_rgb = np.zeros((1000, 1408, 3), dtype=np.uint8)
        tie_rgb[10, 10] = [2, 6, 10]
        tie_result = camera.warp_rgb(tie_rgb, tie_warp)
        np.testing.assert_array_equal(tie_result.rgb[row_tie, col_tie], [1, 2, 3])

    def test_rgb_contract_rejects_wrong_shape_dtype_or_channel_count(self):
        with self.assertRaises(camera.WarpError):
            camera.warp_rgb(np.zeros((1000, 1408), dtype=np.uint8), self.warp)
        with self.assertRaises(camera.WarpError):
            camera.warp_rgb(np.zeros((1000, 1408, 4), dtype=np.uint8), self.warp)
        with self.assertRaises(camera.WarpError):
            camera.warp_rgb(np.zeros((1000, 1408, 3), dtype=np.float32), self.warp)

    def test_authoritative_metadata_isolated_from_top_level_and_nested_export_edits(self):
        rgb = np.full((1000, 1408, 3), [23, 91, 171], dtype=np.uint8)
        result = camera.warp_rgb(rgb, self.warp)
        original = copy.deepcopy(result.metadata)
        exported = result.metadata
        exported['warp_sha256'] = '0' * 64
        exported['output']['rotation_output_from_source'][0] = 999.0
        exported['sampling']['invalid_fill_rgb'][0] = 17
        self.assertEqual(result.metadata, original)
        serialized = json.dumps(result.metadata, sort_keys=True, separators=(',', ':'),
                                allow_nan=False).encode('ascii')
        self.assertEqual(result.metadata_sha256, hashlib.sha256(serialized).hexdigest())

    def test_constructor_rejects_changed_metadata_with_unchanged_digest(self):
        rgb = np.full((1000, 1408, 3), [23, 91, 171], dtype=np.uint8)
        result = camera.warp_rgb(rgb, self.warp)
        altered = copy.deepcopy(result.metadata)
        altered['warp_sha256'] = '0' * 64
        with self.assertRaises(camera.WarpError):
            camera.WarpedRGB(rgb=result.rgb, sampling_valid=result.sampling_valid,
                             calibration_sha256=result.calibration_sha256,
                             warp_sha256=result.warp_sha256, rgb_sha256=result.rgb_sha256,
                             sampling_valid_sha256=result.sampling_valid_sha256,
                             output_sha256=result.output_sha256, metadata=altered,
                             metadata_sha256=result.metadata_sha256)

    def test_replaced_actual_pixels_and_validity_cannot_keep_old_snapshot(self):
        rgb = np.full((1000, 1408, 3), [23, 91, 171], dtype=np.uint8)
        result = camera.warp_rgb(rgb, self.warp)
        changed_rgb = result.rgb.copy()
        changed_rgb[359, 359, 0] ^= 1
        changed_rgb.setflags(write=False)
        with self.assertRaises(camera.WarpError):
            replace(result, rgb=changed_rgb)
        changed_valid = result.sampling_valid.copy()
        changed_valid[359, 359] = not changed_valid[359, 359]
        changed_valid.setflags(write=False)
        with self.assertRaises(camera.WarpError):
            replace(result, sampling_valid=changed_valid)

