"""Independent fake-only checks for the frozen HOT3D capture foundation drafts.

All capture inputs and resource files in this suite are opaque synthetic bytes.
The suite never runs a producer, loads a model, decodes media, or creates a GL
context. Hash-shaped fixture receipts validate schema plumbing only; they do
not certify a metric asset, renderer, or model smoke.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import copy
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from . import quality_capture as capture
from . import test_quality_capture_protocol as protocol


_ROOT = Path(__file__).resolve().parents[1]
_ASSETS_PATH = _ROOT / "bench" / "quality_assets.py"
_RUNNER_PATH = _ROOT / "bench" / "quality_runner.py"
_ASSETS_SHA256 = "D20E8669D7D7C658C14A45D42E2CF8128F2B897D11ED209C2E320CEA1F178009"
_RUNNER_SHA256 = "9A66B30027A4F105502A1A628556221F861302A00AD407268C8039E397FF8757"
_ASSETS_MODULE_NAME = "bench._quality_assets_capture_wave1_draft"
_RUNNER_MODULE_NAME = "bench._quality_runner_capture_wave1_draft"


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _assert_frozen_sources() -> None:
    expected_assets = _ROOT / "bench" / "quality_assets.py"
    expected_runner = _ROOT / "bench" / "quality_runner.py"
    if _ASSETS_PATH != expected_assets or _RUNNER_PATH != expected_runner:
        raise AssertionError("Capture foundation loaders must target the applied bench sources")
    if _ASSETS_PATH.resolve(strict=True).parent != (_ROOT / "bench").resolve(strict=True):
        raise AssertionError("Capture assets loader resolved outside the production bench directory")
    if _RUNNER_PATH.resolve(strict=True).parent != (_ROOT / "bench").resolve(strict=True):
        raise AssertionError("Capture runner loader resolved outside the production bench directory")
    if _source_sha256(_ASSETS_PATH) != _ASSETS_SHA256:
        raise AssertionError("Applied capture assets SHA-256 changed")
    if _source_sha256(_RUNNER_PATH) != _RUNNER_SHA256:
        raise AssertionError("Applied capture runner SHA-256 changed")


def _load_path_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load frozen source by path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_assets_draft():
    _assert_frozen_sources()
    module = sys.modules.get(_ASSETS_MODULE_NAME)
    return module if module is not None else _load_path_module(_ASSETS_PATH, _ASSETS_MODULE_NAME)


def _load_runner_draft(assets):
    """Route only the draft runner's package-local assets import, then restore it."""
    _assert_frozen_sources()
    module = sys.modules.get(_RUNNER_MODULE_NAME)
    if module is not None:
        return module
    package = importlib.import_module("bench")
    missing = object()
    old_attribute = getattr(package, "quality_assets", missing)
    old_module = sys.modules.get("bench.quality_assets", missing)
    try:
        package.quality_assets = assets
        sys.modules["bench.quality_assets"] = assets
        return _load_path_module(_RUNNER_PATH, _RUNNER_MODULE_NAME)
    finally:
        if old_attribute is missing:
            package.__dict__.pop("quality_assets", None)
        else:
            package.quality_assets = old_attribute
        if old_module is missing:
            sys.modules.pop("bench.quality_assets", None)
        else:
            sys.modules["bench.quality_assets"] = old_module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


class _PoisonPath:
    """A path-like sentinel proving rejection occurs before coercion."""

    def __init__(self):
        self.fspath_calls = 0

    def __fspath__(self):
        self.fspath_calls += 1
        raise AssertionError("poison path was coerced before the capture seal")


def _write_bundle_input(bundle):
    bundle.write("input.json", bundle.input_bytes)


def _label_sha(label: str) -> str:
    return _sha(("synthetic-only|" + label).encode("utf-8"))


def _install_synthetic_asset_receipt(bundle):
    """Install schema-shaped inert bytes; this fixture is not metric evidence."""
    bundle.manifest["object"] = "hot3d_obj_000008"
    asset_sha = _sha(bundle.path(bundle.asset_path).read_bytes())
    minimum = [0.0, 0.0, 0.0]
    extents = [0.1, 0.2, 0.3]
    document = {
        "schema_version": 1,
        "kind": "quality-capture-metric-asset-v1",
        "object_id": 8,
        "object_name": "hot3d_obj_000008",
        "asset_sha256": asset_sha,
        "original_asset_sha256": asset_sha,
        "source_units": "metres",
        "conversion_to_metres": 1.0,
        "unit_receipt_sha256": _label_sha("unit-receipt"),
        "official_unit_evidence": {
            "identifier": "synthetic-schema-fixture-only",
            "sha256": _label_sha("official-unit-fixture"),
        },
        "models_info": {
            "sha256": _label_sha("models-info-fixture"),
            "object_bounds_m": {
                "min": minimum,
                "extents": extents,
                "diameter": 0.3,
            },
        },
        "post_node_bounds_m": {
            "min": minimum,
            "max": [minimum[i] + extents[i] for i in range(3)],
            "extents": extents,
        },
        "object_frame": {
            "origin": "original glTF model origin; no recentering",
            "axes": "glTF right-handed +Y up; camera-from-object explicitly converted at consumers",
        },
        "loaders": [
            {"name": "custom_glb", "source_sha256": _label_sha("custom-loader")},
            {"name": "trimesh_gltf", "source_sha256": _label_sha("trimesh-loader")},
        ],
        "geometry_receipt_sha256": _label_sha("geometry-fixture"),
        "renderer_receipt_sha256": _label_sha("renderer-fixture"),
    }
    bundle.write(bundle.receipt_path, _canonical(document))
    bundle.seal()
    return document


def _write_stage_sidecar(use, *, resource_kind=None, resource_key=None,
                         recipe=None, source_closure=None, device=None):
    """Write an opaque structural sidecar with no producer artifact semantics."""
    artifact_bytes = b"opaque fake-only artifact; never deserialized"
    directory = use.sidecar_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / "artifact.bin"
    artifact_path.write_bytes(artifact_bytes)
    document = {
        "schema_version": 1,
        "resource_kind": use.resource_kind if resource_kind is None else resource_kind,
        "resource_key": use.resource_key if resource_key is None else resource_key,
        "recipe": protocol._plain(use.recipe) if recipe is None else recipe,
        "dimensions": [1],
        "coordinate_mode": capture.COORDINATE_MODE,
        "artifact": {
            "path": "artifact.bin",
            "sha256": _sha(artifact_bytes),
            "byte_count": len(artifact_bytes),
        },
        "runtime": {
            "device": use.recipe["producer_device"] if device is None else device,
            "build": "synthetic-fake-only",
        },
        "source_closure": dict(use.source_closure) if source_closure is None else source_closure,
    }
    use.sidecar_path.write_bytes(_canonical(document))
    return document


@contextmanager
def _fake_resource_tree(assets, plan, root: Path):
    """Create tiny declared source/runtime/checkpoint fixtures in a temp root."""
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache" / "model-quality"
    for child in ("archives", "sources", "checkpoints"):
        (cache / child).mkdir(parents=True, exist_ok=True)

    source_revisions = {
        "gotrack": "synthetic-gotrack-revision",
        "bop_toolkit": "synthetic-bop-revision",
        "dinov2": "synthetic-dinov2-revision",
    }
    archive_bytes = {
        name: ("opaque fake archive|" + name).encode("ascii")
        for name in source_revisions
    }
    archive_hashes = {name: _sha(data) for name, data in archive_bytes.items()}
    sources_pin = {"gotrack": source_revisions["gotrack"]}
    submodules_pin = {
        "bop_toolkit": ("synthetic/bop_toolkit", source_revisions["bop_toolkit"]),
        "dinov2": ("synthetic/dinov2", source_revisions["dinov2"]),
    }
    expected_archive_pin = dict(archive_hashes)
    source_lock = {}
    for name, revision in source_revisions.items():
        archive_name = f"{name}-{revision}.zip"
        (cache / "archives" / archive_name).write_bytes(archive_bytes[name])
        source_lock[name] = {
            "revision": revision,
            "archive_sha256": archive_hashes[name],
        }

    original_models = assets.MODELS
    model_pins = {name: dict(spec) for name, spec in original_models.items()}
    model_lock = {}
    for checkpoint in sorted({
        item for names in assets._CAPTURE_RESOURCE_CHECKPOINTS.values() for item in names
    }):
        data = ("opaque inert checkpoint fixture|" + checkpoint).encode("ascii")
        model_sha = _sha(data)
        model_pins[checkpoint]["sha256"] = model_sha
        model_lock[checkpoint] = {"sha256": model_sha, "bytes": len(data)}
        (cache / "checkpoints" / checkpoint).write_bytes(data)

    lock = {"sources": source_lock, "models": model_lock}
    (cache / "lock.json").write_bytes(_canonical(lock))

    for resource_sources in assets._CAPTURE_RESOURCE_UPSTREAM_FILES.values():
        for source_name, files in resource_sources.items():
            source_root = (cache / "sources" / "gotrack" if source_name == "gotrack"
                           else cache / "sources" / "gotrack" / "external" / source_name)
            for relative in files:
                target = source_root.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(("opaque source fixture|" + source_name + "|" + relative).encode("utf-8"))

    bench_root = root / "bench"
    bench_root.mkdir(parents=True, exist_ok=True)
    capture_source_root = Path(capture.__file__).resolve().parent
    for name in ("quality_capture.py", "quality_camera.py", "quality_time.py"):
        (bench_root / name).write_bytes((capture_source_root / name).read_bytes())
    provenance_modules = (
        "quality_assets", "quality_contract", "quality_time", "quality_runner",
        "quality_gotrack", "quality_sam2", "quality_foundpose", "quality_cnos",
        "quality_detection_association", "quality_neighbors", "quality_trace",
        "quality_memory", "quality_appearance", "quality_render_stability", "vision",
    )
    source_modules = set(provenance_modules)
    for names in assets._CAPTURE_RESOURCE_MODULES.values():
        source_modules.update(names)
    for names in assets._CAPTURE_STAGE_ADAPTER.values():
        source_modules.update(names)
    for name in source_modules:
        target = bench_root / f"{name}.py"
        if not target.exists():
            target.write_bytes(("opaque adapter fixture|" + name).encode("ascii"))

    runtime_root = root / "fake-runtime"
    runtime_locations = {}
    for distribution_name, (package, files) in assets._CAPTURE_RENDERER_RUNTIME_FILES.items():
        package_root = runtime_root / package
        package_root.mkdir(parents=True, exist_ok=True)
        for relative in files:
            target = package_root.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(("opaque runtime fixture|" + distribution_name + "|" + relative).encode("utf-8"))
        runtime_locations[distribution_name] = package_root

    class _FakeDistribution:
        def __init__(self, name):
            self.version = "synthetic-runtime-version-" + name
            self.name = name

        def locate_file(self, package):
            expected_package = assets._CAPTURE_RENDERER_RUNTIME_FILES[self.name][0]
            if str(package) != expected_package:
                raise AssertionError("Draft requested an unpinned runtime package")
            return runtime_locations[self.name]

    def fake_distribution(name):
        if name not in runtime_locations:
            raise AssertionError("Draft requested an undeclared runtime distribution")
        return _FakeDistribution(name)

    original = {
        "ROOT": assets.ROOT,
        "CACHE": assets.CACHE,
        "CAPTURE_CACHE": assets.CAPTURE_CACHE,
        "BUDGET": assets.BUDGET,
        "SOURCES": assets.SOURCES,
        "SUBMODULES": assets.SUBMODULES,
        "MODELS": assets.MODELS,
        "_CAPTURE_SOURCE_ARCHIVE_SHA256": assets._CAPTURE_SOURCE_ARCHIVE_SHA256,
        "distribution": assets.distribution,
    }
    try:
        assets.ROOT = root
        assets.CACHE = cache
        assets.CAPTURE_CACHE = cache / "cache"
        assets.BUDGET = 8 * 1024**3
        assets.SOURCES = sources_pin
        assets.SUBMODULES = submodules_pin
        assets.MODELS = model_pins
        assets._CAPTURE_SOURCE_ARCHIVE_SHA256 = expected_archive_pin
        assets.distribution = fake_distribution
        yield {"root": root, "cache": cache, "lock": lock,
               "archive_bytes": archive_bytes, "archive_hashes": archive_hashes}
    finally:
        for name, value in original.items():
            setattr(assets, name, value)


def _capture_binding(plan, row):
    output = row.get("output_rgb")
    return {
        "capture_table_sha256": plan.capture_table_sha256,
        "capture_row_sha256": row["capture_row_sha256"],
        "rgb_pixel_sha256": None if output is None else output.get("pixel_sha256"),
        "sampling_valid_sha256": row.get("sampling_valid_sha256"),
        "warp_sha256": row.get("warp_sha256"),
        "calibration_sha256": row.get("calibration_values_sha256"),
    }


def _rewrite_checkpoint_pin(assets, cache: Path, checkpoint: str, data: bytes) -> None:
    """Change a declared fake pin and its inert file/lock entry together."""
    sha = _sha(data)
    assets.MODELS[checkpoint]["sha256"] = sha
    (cache / "checkpoints" / checkpoint).write_bytes(data)
    lock_path = cache / "lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["models"][checkpoint] = {"sha256": sha, "bytes": len(data)}
    lock_path.write_bytes(_canonical(lock))


def _mask_cache_document(runner, plan, manifest, requested_ids):
    planned = list(manifest["frame_ids"])
    table_rows = {row["frame_id"]: row for row in plan.rows}
    timeline = {row["frame_id"]: row for row in manifest["timeline"]}
    frames = []
    for ordinal in planned:
        timing_row = timeline[ordinal]
        outside = ordinal not in requested_ids
        frames.append({
            "frameId": ordinal,
            "sourceFrameId": timing_row["source_frame_id"],
            "timestamp_s": timing_row["timestamp_s"],
            "clock_mode": runner.PHYSICAL,
            "role": timing_row["role"],
            "capture_binding": _capture_binding(plan, table_rows[ordinal]),
            "mask_state": "lost",
            "measurement_state": "unmeasured",
            "path": None,
            "mask_sha256": None,
            "failure_reason": "outside_requested_prefix" if outside else "synthetic_unmeasured_fixture",
        })
    return {
        "schema_version": 2,
        "clock_mode": runner.PHYSICAL,
        "input_manifest_sha256": manifest["_timing"]["manifest_sha256"],
        "timestamp_table_sha256": manifest["_timing"]["timestamp_table_sha256"],
        "source_kind": capture.SOURCE_KIND,
        "capture_table_sha256": plan.capture_table_sha256,
        "coordinate_mode": capture.COORDINATE_MODE,
        "coverage_complete": True,
        "status": "diagnostic_prefix",
        "stage_completed": True,
        "expected_setup_frames": 0,
        "expected_scored_frames": len(planned),
        "planned_frame_ids": planned,
        "requested_frame_ids": list(requested_ids),
        "mask_association": False,
        "frames": frames,
    }


class CaptureFoundationDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = _load_assets_draft()
        cls.runner = _load_runner_draft(cls.assets)

    def setUp(self):
        _assert_frozen_sources()
        self.temporary = tempfile.TemporaryDirectory(prefix="hot3d-capture-foundation-")
        self.temporary_root = Path(self.temporary.name)
        self.bundle = protocol._SyntheticBundle(self.temporary_root / "bundle")
        _install_synthetic_asset_receipt(self.bundle)
        _write_bundle_input(self.bundle)

    def tearDown(self):
        self.temporary.cleanup()
        _assert_frozen_sources()

    def _issued_plan(self, bundle=None):
        source = self.bundle if bundle is None else bundle
        _write_bundle_input(source)
        return capture.load_capture_plan(source.root, source.input_bytes)

    def _new_bundle(self, name, *, unavailable=frozenset()):
        bundle = protocol._SyntheticBundle(
            self.temporary_root / name, unavailable=unavailable)
        _install_synthetic_asset_receipt(bundle)
        return bundle

    def _public_resource_uses(self, plan, *, flags=None, device="cpu"):
        smoke = self.assets.capture_stage_resources(
            plan, stage="smoke", render_flags=flags or {}, device=device)
        _write_stage_sidecar(smoke.output)
        banks = self.assets.capture_stage_resources(
            plan, stage="banks", render_flags=flags or {}, device=device)
        cnos = self.assets.capture_stage_resources(
            plan, stage="cnos-bank", render_flags=flags or {}, device=device)
        return {"smoke": smoke, "banks": banks, "cnos-bank": cnos}

    def test_loader_paths_are_the_applied_production_sources(self):
        expected_assets = (_ROOT / "bench" / "quality_assets.py").resolve(strict=True)
        expected_runner = (_ROOT / "bench" / "quality_runner.py").resolve(strict=True)
        self.assertEqual(_ASSETS_PATH.resolve(strict=True), expected_assets)
        self.assertEqual(_RUNNER_PATH.resolve(strict=True), expected_runner)
        self.assertEqual(expected_assets.parent, expected_runner.parent)
        self.assertEqual(expected_assets.parent, (_ROOT / "bench").resolve(strict=True))
        self.assertNotIn(".cache", expected_assets.parts)
        self.assertNotIn(".cache", expected_runner.parts)
        self.assertEqual(_source_sha256(expected_assets), _ASSETS_SHA256)
        self.assertEqual(_source_sha256(expected_runner), _RUNNER_SHA256)

    def test_unissued_plan_rejects_poison_paths_before_receipt_mask_or_decoder(self):
        unissued = capture.prepare_capture_plan(self.bundle.input_bytes, self.bundle.table_bytes)
        poisoned = _PoisonPath()
        forbidden = AssertionError("filesystem, mask, or decode hook reached before seal rejection")
        with (
            patch.object(Path, "resolve", side_effect=forbidden),
            patch.object(Path, "open", side_effect=forbidden),
            patch.object(Path, "read_bytes", side_effect=forbidden),
            patch.object(self.assets, "_read_capture_asset_receipt_validated", side_effect=forbidden),
            patch.object(self.runner, "_read_json_unique", side_effect=forbidden),
            patch.object(capture, "_decode_result", side_effect=forbidden),
        ):
            with self.assertRaises(TypeError):
                self.assets.read_capture_asset_receipt(unissued)
            with self.assertRaises(TypeError):
                self.assets.capture_stage_resources(unissued, stage="smoke")
            with self.assertRaises(TypeError):
                self.runner.read_decoded_frame(poisoned, {"_capture": unissued}, 0)
            with self.assertRaises(TypeError):
                self.runner._physical_mask_cache(
                    {"_capture": unissued}, poisoned)
        self.assertEqual(poisoned.fspath_calls, 0)

    def test_mutated_issued_nested_record_rejects_before_path_and_effect_hooks(self):
        issued = self._issued_plan()
        poisoned = _PoisonPath()
        replaced = replace(issued, bundle=issued.bundle.parent / "different-bundle")
        manifest = {"_capture": replaced}
        forbidden = AssertionError("a mutated issued record reached an effect hook")
        with (
            patch.object(Path, "resolve", side_effect=forbidden),
            patch.object(Path, "open", side_effect=forbidden),
            patch.object(Path, "read_bytes", side_effect=forbidden),
            patch.object(self.assets, "_read_capture_asset_receipt_validated", side_effect=forbidden),
            patch.object(self.runner, "_read_json_unique", side_effect=forbidden),
            patch.object(capture, "_decode_result", side_effect=forbidden),
        ):
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.read_capture_asset_receipt(replaced)
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.capture_stage_resources(replaced, stage="smoke")
            with self.assertRaises(capture.CaptureIntegrityError):
                self.runner.read_decoded_frame(poisoned, manifest, 0)
            with self.assertRaises(capture.CaptureIntegrityError):
                self.runner._physical_mask_cache(manifest, poisoned)
        self.assertEqual(poisoned.fspath_calls, 0)

    def test_capture_provenance_rejects_different_bundle_before_asset_or_resources(self):
        issued = self._issued_plan()
        other = protocol._SyntheticBundle(self.temporary_root / "other-bundle")
        _write_bundle_input(other)
        forbidden = AssertionError("provenance read asset/resources before bundle identity check")
        with (
            patch.object(self.assets, "_read_capture_asset_receipt_validated", side_effect=forbidden) as receipt,
            patch.object(self.assets, "_capture_validate_stage_resources", side_effect=forbidden) as resources,
        ):
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(other.root, capture_plan=issued)
        receipt.assert_not_called()
        resources.assert_not_called()

    def test_capture_provenance_rejects_changed_current_input_before_asset_or_resources(self):
        issued = self._issued_plan()
        self.bundle.write("input.json", b"changed after public preflight")
        forbidden = AssertionError("provenance read asset/resources before current-input check")
        with (
            patch.object(self.assets, "_read_capture_asset_receipt_validated", side_effect=forbidden) as receipt,
            patch.object(self.assets, "_capture_validate_stage_resources", side_effect=forbidden) as resources,
        ):
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(self.bundle.root, capture_plan=issued)
        receipt.assert_not_called()
        resources.assert_not_called()

    def test_asset_receipt_semantics_and_post_preflight_bytes_are_enforced(self):
        mutations = (
            ("extra_field", lambda doc: doc.update({"unreviewed": True})),
            ("source_units", lambda doc: doc.update({"source_units": "millimetres"})),
            ("scale", lambda doc: doc.update({"conversion_to_metres": 0.001})),
            ("object_frame", lambda doc: doc["object_frame"].update({"axes": "unknown"})),
            ("bounds", lambda doc: doc["post_node_bounds_m"].update({
                "max": [0.1, 0.2, 0.31]})),
        )
        for label, mutate in mutations:
            with self.subTest(receipt_mutation=label):
                candidate = self._new_bundle("invalid-receipt-" + label)
                document = json.loads(candidate.path(candidate.receipt_path).read_text(encoding="utf-8"))
                mutate(document)
                candidate.write(candidate.receipt_path, _canonical(document))
                candidate.seal()
                plan = self._issued_plan(candidate)
                with self.assertRaises(ValueError):
                    self.assets.read_capture_asset_receipt(plan)

        candidate = self._new_bundle("changed-receipt-bytes")
        plan = self._issued_plan(candidate)
        candidate.write(candidate.receipt_path, b"{}")
        with self.assertRaises(capture.CaptureIntegrityError):
            self.assets.read_capture_asset_receipt(plan)

    def test_asset_receipt_and_smoke_sidecar_use_the_canonical_nested_key(self):
        plan = self._issued_plan()
        fake_root = self.temporary_root / "fake-resource-root"
        with _fake_resource_tree(self.assets, plan, fake_root):
            receipt = self.assets.read_capture_asset_receipt(plan)
            declared_receipt_sha = plan.verified_resources[plan.manifest["asset_receipt"]]
            self.assertEqual(receipt.receipt_sha256, declared_receipt_sha)
            self.assertEqual(receipt.asset_sha256, plan.verified_resources[plan.manifest["asset"]])
            self.assertEqual(receipt.document["kind"], "quality-capture-metric-asset-v1")

            smoke_set = self.assets.capture_stage_resources(
                plan, stage="smoke", render_flags={"unlit_templates": True}, device="cpu")
            smoke = smoke_set.output
            self.assertIsNotNone(smoke)
            expected_key = capture.capture_resource_key(
                receipt.document, plan.table["output_camera"],
                smoke.recipe, smoke.source_closure)
            self.assertEqual(smoke.resource_key, expected_key)
            self.assertEqual(
                smoke.sidecar_path,
                self.assets.CAPTURE_CACHE / "capture-resources" / expected_key / "resource.json")
            self.assertTrue(smoke.sidecar_path.is_relative_to(self.assets.CACHE))
            self.assertTrue(smoke.sidecar_path.is_relative_to(fake_root))

            _write_stage_sidecar(smoke)
            banks = self.assets.capture_stage_resources(
                plan, stage="banks", render_flags={"unlit_templates": True}, device="cpu")
            self.assertEqual(tuple(use.purpose for use in banks.required), ("smoke",))
            descriptor = banks.required[0].descriptor
            self.assertEqual(descriptor.resource_key, expected_key)
            self.assertEqual(descriptor.resource_kind, smoke.resource_kind)
            self.assertEqual(protocol._plain(descriptor.recipe), protocol._plain(smoke.recipe))
            self.assertEqual(dict(descriptor.source_closure), dict(smoke.source_closure))
            self.assertEqual(descriptor.descriptor["runtime"]["device"], "cpu")

            mutations = (
                ("resource_key", {"resource_key": _label_sha("wrong-sidecar-key")}),
                ("resource_kind", {"resource_kind": "wrong-fixed-kind"}),
                ("recipe", {"recipe": {**protocol._plain(smoke.recipe),
                                         "output_camera_sha256": _label_sha("wrong-recipe")}}),
                ("source_closure", {"source_closure": {
                    **dict(smoke.source_closure),
                    "bench/quality_capture.py": _label_sha("wrong-source-closure")}}),
                ("device", {"device": "cuda"}),
            )
            for label, overrides in mutations:
                with self.subTest(sidecar_binding=label):
                    _write_stage_sidecar(smoke)
                    if label == "device":
                        sidecar = json.loads(smoke.sidecar_path.read_text(encoding="utf-8"))
                        sidecar["runtime"]["device"] = overrides["device"]
                        smoke.sidecar_path.write_bytes(_canonical(sidecar))
                    else:
                        values = {
                            "resource_kind": smoke.resource_kind,
                            "resource_key": smoke.resource_key,
                            "recipe": protocol._plain(smoke.recipe),
                            "source_closure": dict(smoke.source_closure),
                        }
                        values.update(overrides)
                        _write_stage_sidecar(
                            smoke, resource_kind=values["resource_kind"],
                            resource_key=values["resource_key"], recipe=values["recipe"],
                            source_closure=values["source_closure"])
                    with self.assertRaises(capture.CaptureIntegrityError):
                        self.assets.capture_stage_resources(
                            plan, stage="banks", render_flags={"unlit_templates": True}, device="cpu")

    def test_mask_cache_joins_all_six_fields_and_nullable_unavailable_rows(self):
        unavailable = self._new_bundle("unavailable-row", unavailable=frozenset({1}))
        plan = self._issued_plan(unavailable)
        manifest = self.runner.read_input(unavailable.root)
        masks_root = self.temporary_root / "masks"
        masks_root.mkdir()

        for requested in ([0, 1], [0]):
            with self.subTest(requested_prefix=requested):
                document = _mask_cache_document(self.runner, plan, manifest, requested)
                cache_path = masks_root / "results.json"
                cache_path.write_bytes(_canonical(document))
                returned, by_id = self.runner._physical_mask_cache(manifest, masks_root)
                self.assertEqual(returned["requested_frame_ids"], requested)
                nullable = by_id[1]["capture_binding"]
                self.assertEqual(nullable["capture_table_sha256"], plan.capture_table_sha256)
                self.assertEqual(nullable["capture_row_sha256"], plan.rows[1]["capture_row_sha256"])
                for name in ("rgb_pixel_sha256", "sampling_valid_sha256", "warp_sha256",
                             "calibration_sha256"):
                    self.assertIsNone(nullable[name])
                if 1 not in requested:
                    self.assertEqual(by_id[1]["failure_reason"], "outside_requested_prefix")

        available_plan = self._issued_plan()
        available_manifest = self.runner.read_input(self.bundle.root)
        baseline = _mask_cache_document(self.runner, available_plan, available_manifest, [0])
        cache_path = masks_root / "results.json"
        for field in (
            "capture_table_sha256", "capture_row_sha256", "rgb_pixel_sha256",
            "sampling_valid_sha256", "warp_sha256", "calibration_sha256",
        ):
            with self.subTest(mutated_capture_binding=field):
                changed = copy.deepcopy(baseline)
                changed["frames"][0]["capture_binding"][field] = _label_sha("changed-" + field)
                cache_path.write_bytes(_canonical(changed))
                with self.assertRaises(ValueError):
                    self.runner._physical_mask_cache(available_manifest, masks_root)

        changed_outside = copy.deepcopy(baseline)
        changed_outside["frames"][2]["failure_reason"] = "unmeasured_for_another_reason"
        cache_path.write_bytes(_canonical(changed_outside))
        with self.assertRaises(ValueError):
            self.runner._physical_mask_cache(available_manifest, masks_root)

        false_available = _mask_cache_document(
            self.runner, plan, manifest, [0, 1])
        false_available["frames"][1].update({
            "mask_state": "available", "measurement_state": "measured",
            "path": "claimed-mask.png", "mask_sha256": _label_sha("false-mask"),
            "failure_reason": None,
        })
        cache_path.write_bytes(_canonical(false_available))
        with self.assertRaises(ValueError):
            self.runner._physical_mask_cache(manifest, masks_root)

    def test_runner_frame_bridge_never_advances_or_decodes_a_future_row(self):
        plan_manifest = self.runner.read_input(self.bundle.root)
        plan = plan_manifest["_capture"]
        decoder = protocol._SyntheticDecoder()
        reader = capture.CaptureReader(self.bundle.root, plan, decoder=decoder)
        first = self.runner.read_decoded_frame(
            self.bundle.root, plan_manifest, 0, reader=reader)
        self.assertEqual(first.source_frame_id, plan.rows[0]["source_frame_id"])
        prior_calls = tuple(decoder.calls)
        with self.assertRaises(capture.CaptureIntegrityError):
            self.runner.read_decoded_frame(
                self.bundle.root, plan_manifest, 1, reader=reader)
        snapshot = reader.snapshot(1)
        self.assertEqual(snapshot["allowed_through_ordinal"], 0)
        self.assertEqual(snapshot["attempts"], 0)
        self.assertEqual(tuple(decoder.calls), prior_calls)
        reader.close()

    def test_stage_resource_purpose_output_and_asset_mixing_are_rejected(self):
        plan = self._issued_plan()
        fake_root = self.temporary_root / "fake-resource-root"
        with _fake_resource_tree(self.assets, plan, fake_root):
            smoke = self.assets.capture_stage_resources(
                plan, stage="smoke", render_flags={}, device="cpu")
            self.assertEqual(smoke.required, ())
            self.assertEqual(smoke.output.purpose, "smoke")
            _write_stage_sidecar(smoke.output)
            banks = self.assets.capture_stage_resources(
                plan, stage="banks", render_flags={}, device="cpu")
            self.assertEqual(tuple(use.purpose for use in banks.required), ("smoke",))
            self.assertEqual(banks.output.purpose, "foundpose-bank")

            mixed_stage = replace(smoke, stage="banks")
            with self.assertRaises((TypeError, ValueError, capture.CaptureIntegrityError)):
                self.assets.inference_provenance(
                    self.bundle.root, capture_plan=plan, capture_resources=mixed_stage)

            wrong_output = replace(smoke.output, purpose="foundpose-bank")
            mixed_output = replace(smoke, output=wrong_output)
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(
                    self.bundle.root, capture_plan=plan, capture_resources=mixed_output)

            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(
                    self.bundle.root, capture_plan=plan,
                    capture_resources=replace(smoke, output=banks.output))

            changed_recipe = replace(
                smoke.output,
                recipe={**protocol._plain(smoke.output.recipe), "producer_device": "cuda"})
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(
                    self.bundle.root, capture_plan=plan,
                    capture_resources=replace(smoke, output=changed_recipe))

            other = self._new_bundle("different-asset")
            other.write(other.asset_path, b"different synthetic asset bytes")
            _install_synthetic_asset_receipt(other)
            other_plan = self._issued_plan(other)
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(
                    other.root, capture_plan=other_plan, capture_resources=smoke)
            with self.assertRaises(capture.CaptureIntegrityError):
                self.assets.inference_provenance(
                    other.root, capture_plan=other_plan, capture_resources=banks)

    def test_same_asset_bank_recipe_reuses_across_clips_with_each_issued_plan(self):
        first_plan = self._issued_plan()
        second = self._new_bundle("clip-two")
        second.table["clip_id"] = 1945
        second.inventory["clip_id"] = 1945
        second.seal()
        second_plan = self._issued_plan(second)
        self.assertNotEqual(first_plan.capture_table_sha256, second_plan.capture_table_sha256)
        self.assertNotEqual(first_plan.input_manifest_sha256, second_plan.input_manifest_sha256)

        fake_root = self.temporary_root / "fake-resource-root"
        with _fake_resource_tree(self.assets, first_plan, fake_root):
            first_smoke = self.assets.capture_stage_resources(
                first_plan, stage="smoke", render_flags={"unlit_templates": False}, device="cpu")
            second_smoke = self.assets.capture_stage_resources(
                second_plan, stage="smoke", render_flags={"unlit_templates": False}, device="cpu")
            self.assertEqual(first_smoke.output.resource_key, second_smoke.output.resource_key)
            _write_stage_sidecar(first_smoke.output)

            first_banks = self.assets.capture_stage_resources(
                first_plan, stage="banks", render_flags={"unlit_templates": False}, device="cpu")
            second_banks = self.assets.capture_stage_resources(
                second_plan, stage="banks", render_flags={"unlit_templates": False}, device="cpu")
            self.assertEqual(first_banks.output.resource_key, second_banks.output.resource_key)
            self.assertEqual(first_banks.required[0].resource_key,
                             second_banks.required[0].resource_key)

            first_provenance = self.assets.inference_provenance(
                self.bundle.root, capture_plan=first_plan, capture_resources=first_banks)
            second_provenance = self.assets.inference_provenance(
                second.root, capture_plan=second_plan, capture_resources=second_banks)
            first_capture = first_provenance["capture_provenance"]
            second_capture = second_provenance["capture_provenance"]
            self.assertNotEqual(first_capture["capture_table_sha256"],
                                second_capture["capture_table_sha256"])
            self.assertEqual(first_capture["used_resources"][0]["resource_key"],
                             second_capture["used_resources"][0]["resource_key"])
            self.assertEqual(first_capture["output_resource"]["resource_key"],
                             second_capture["output_resource"]["resource_key"])

    def test_legacy_provenance_keeps_the_unmodified_bundle_digest_shape(self):
        plan = self._issued_plan()
        legacy_bytes = _canonical({"schema_version": 1, "fixture_kind": "legacy-input"})
        self.bundle.write("input.json", legacy_bytes)
        with _fake_resource_tree(self.assets, plan, self.temporary_root / "fake-resource-root"):
            result = self.assets.inference_provenance(self.bundle.root)
        expected = hashlib.sha256(legacy_bytes).hexdigest()
        self.assertEqual(result["input_manifest_sha256"], expected)
        self.assertEqual(result["input_manifest_sha256"], result["input_manifest_sha256"].lower())

    def test_legacy_runner_input_still_uses_exact_raw_manifest_and_hashes(self):
        bundle_root = self.temporary_root / "legacy-bundle"
        bundle_root.mkdir()
        source = bundle_root / "source.bin"
        source.write_bytes(b"opaque synthetic legacy video-source identity")
        source_hashes = {source.name: _sha(source.read_bytes()).lower()}
        manifest = {
            "schema_version": 1,
            "object": "keyboard",
            "units": "metres",
            "source_hashes": source_hashes,
            "setup_frame_id": 10,
            "frame_ids": [11],
        }
        raw = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        manifest_sha = self.runner.parse_timing_manifest(raw)["manifest_sha256"]
        with patch.dict(self.runner._LEGACY_INPUTS, {
                manifest_sha: {"object": "keyboard", "source_hashes": source_hashes}}, clear=True):
            (bundle_root / "input.json").write_bytes(raw)
            parsed = self.runner.read_input(bundle_root)
            self.assertEqual(parsed["_timing"]["clock_mode"], self.runner.LEGACY)
            (bundle_root / "input.json").write_bytes(raw + b" ")
            with self.assertRaises(ValueError):
                self.runner.read_input(bundle_root)

    def test_capture_rgb_path_rejects_video_fallback_before_video_capture(self):
        manifest = self.runner.read_input(self.bundle.root)
        with patch.object(
                self.runner.cv2, "VideoCapture",
                side_effect=AssertionError("capture sequence fell through to video decode")) as video:
            with self.assertRaises(ValueError):
                self.runner.read_rgb(self.bundle.root, manifest, 0)
        video.assert_not_called()

    def test_public_stage_resource_graph_has_fixed_non_circular_prerequisites(self):
        plan = self._issued_plan()
        fake_root = self.temporary_root / "fake-resource-root"
        with _fake_resource_tree(self.assets, plan, fake_root):
            uses = self._public_resource_uses(plan)
            _write_stage_sidecar(uses["banks"].output)
            _write_stage_sidecar(uses["cnos-bank"].output)
            segment = self.assets.capture_stage_resources(plan, stage="segment", device="cpu")
            detect = self.assets.capture_stage_resources(plan, stage="detect", device="cpu")
            pose = self.assets.capture_stage_resources(plan, stage="pose", device="cpu")

            self.assertEqual(tuple(use.purpose for use in uses["smoke"].required), ())
            self.assertEqual(tuple(use.purpose for use in uses["banks"].required), ("smoke",))
            self.assertEqual(tuple(use.purpose for use in uses["cnos-bank"].required), ("smoke",))
            self.assertEqual(tuple(use.purpose for use in segment.required), ("cnos-bank",))
            self.assertEqual(tuple(use.purpose for use in detect.required), ("cnos-bank",))
            self.assertEqual(tuple(use.purpose for use in pose.required),
                             ("smoke", "foundpose-bank"))
            self.assertIsNone(segment.output)
            self.assertIsNone(detect.output)
            self.assertIsNone(pose.output)

            producer = {
                resource: stage
                for stage, resource in self.assets._CAPTURE_STAGE_OUTPUT.items()
            }
            stage_sets = {
                "smoke": uses["smoke"], "banks": uses["banks"],
                "cnos-bank": uses["cnos-bank"], "segment": segment,
                "detect": detect, "pose": pose,
            }
            dependencies = {
                stage: tuple(producer[use.purpose] for use in stage_set.required)
                for stage, stage_set in stage_sets.items()
            }
            visiting = set()
            visited = set()

            def visit(stage):
                if stage in visiting:
                    raise AssertionError(f"capture resource prerequisite cycle at {stage}")
                if stage in visited:
                    return
                visiting.add(stage)
                for dependency in dependencies[stage]:
                    visit(dependency)
                visiting.remove(stage)
                visited.add(stage)

            for stage in stage_sets:
                visit(stage)

    def test_extracted_layerscale_bytes_invalidate_all_producer_keys(self):
        plan = self._issued_plan()
        fake_root = self.temporary_root / "fake-resource-root"
        with _fake_resource_tree(self.assets, plan, fake_root) as tree:
            before = self._public_resource_uses(plan)
            before_keys = {name: value.output.resource_key for name, value in before.items()}
            layer_scale = (tree["cache"] / "sources" / "gotrack" / "external" /
                           "dinov2" / "dinov2" / "layers" / "layer_scale.py")
            self.assertTrue(layer_scale.is_file())
            original_layer = layer_scale.read_bytes()
            archive_paths = {
                name: tree["cache"] / "archives" / f"{name}-{revision}.zip"
                for name, revision in (
                    ("gotrack", self.assets.SOURCES["gotrack"]),
                    ("bop_toolkit", self.assets.SUBMODULES["bop_toolkit"][1]),
                    ("dinov2", self.assets.SUBMODULES["dinov2"][1]),
                )
            }
            archive_before = {name: path.read_bytes() for name, path in archive_paths.items()}

            layer_scale.write_bytes(original_layer + b"\nchanged extracted LayerScale fixture")
            changed_smoke = self.assets.capture_stage_resources(
                plan, stage="smoke", device="cpu").output
            self.assertNotEqual(changed_smoke.resource_key, before_keys["smoke"])
            with self.assertRaises(OSError):
                self.assets.capture_stage_resources(plan, stage="banks", device="cpu")

            _write_stage_sidecar(changed_smoke)
            after_layer = {
                "smoke": self.assets.capture_stage_resources(plan, stage="smoke", device="cpu"),
                "banks": self.assets.capture_stage_resources(plan, stage="banks", device="cpu"),
                "cnos-bank": self.assets.capture_stage_resources(plan, stage="cnos-bank", device="cpu"),
            }
            after_layer_keys = {
                name: value.output.resource_key for name, value in after_layer.items()
            }
            for name in ("smoke", "banks", "cnos-bank"):
                with self.subTest(producer=name):
                    self.assertNotEqual(before_keys[name], after_layer_keys[name])
            for name, path in archive_paths.items():
                self.assertEqual(path.read_bytes(), archive_before[name])
                self.assertEqual(_sha(path.read_bytes()), tree["archive_hashes"][name])

            sam_consumer = fake_root / "bench" / "quality_sam2.py"
            sam_consumer.write_bytes(b"unrelated SAM consumer edit fixture")
            after_sam = self._public_resource_uses(plan)
            self.assertEqual(after_sam["smoke"].output.resource_key, after_layer_keys["smoke"])
            self.assertEqual(after_sam["banks"].output.resource_key, after_layer_keys["banks"])
            self.assertEqual(after_sam["cnos-bank"].output.resource_key,
                             after_layer_keys["cnos-bank"])

            _rewrite_checkpoint_pin(
                self.assets, tree["cache"], "gotrack_checkpoint.pt",
                b"changed opaque GoTrack checkpoint pin")
            after_gotrack = self._public_resource_uses(plan)
            self.assertNotEqual(after_gotrack["smoke"].output.resource_key,
                                after_sam["smoke"].output.resource_key)
            self.assertNotEqual(after_gotrack["banks"].output.resource_key,
                                after_sam["banks"].output.resource_key)
            self.assertEqual(after_gotrack["cnos-bank"].output.resource_key,
                             after_sam["cnos-bank"].output.resource_key)

            _rewrite_checkpoint_pin(
                self.assets, tree["cache"], "dinov2_vitl14_pretrain.pth",
                b"changed opaque DINO ViT-L checkpoint pin")
            after_cnos = self._public_resource_uses(plan)
            self.assertEqual(after_cnos["smoke"].output.resource_key,
                             after_gotrack["smoke"].output.resource_key)
            self.assertEqual(after_cnos["banks"].output.resource_key,
                             after_gotrack["banks"].output.resource_key)
            self.assertNotEqual(after_cnos["cnos-bank"].output.resource_key,
                                after_gotrack["cnos-bank"].output.resource_key)
            for name, path in archive_paths.items():
                self.assertEqual(path.read_bytes(), archive_before[name])

    def test_flags_device_and_producer_settings_change_only_bound_keys(self):
        plan = self._issued_plan()
        fake_root = self.temporary_root / "fake-resource-root"
        with _fake_resource_tree(self.assets, plan, fake_root):
            baseline = self._public_resource_uses(plan)
            baseline_keys = {name: use.output.resource_key for name, use in baseline.items()}

            flagged = self._public_resource_uses(
                plan, flags={"unlit_templates": True}, device="cpu")
            for name in baseline_keys:
                with self.subTest(render_flag=name):
                    self.assertNotEqual(baseline_keys[name], flagged[name].output.resource_key)

            other_device = self._public_resource_uses(plan, flags={}, device="cuda")
            for name in baseline_keys:
                with self.subTest(device=name):
                    self.assertNotEqual(baseline_keys[name], other_device[name].output.resource_key)

            changed_settings = copy.deepcopy(self.assets._CAPTURE_RESOURCE_SETTINGS)
            changed_settings["foundpose-bank"]["template_count"] += 1
            with patch.object(self.assets, "_CAPTURE_RESOURCE_SETTINGS", changed_settings):
                changed = self._public_resource_uses(plan)
            self.assertEqual(changed["smoke"].output.resource_key, baseline_keys["smoke"])
            self.assertNotEqual(changed["banks"].output.resource_key, baseline_keys["banks"])
            self.assertEqual(changed["cnos-bank"].output.resource_key,
                             baseline_keys["cnos-bank"])


if __name__ == "__main__":
    unittest.main()
