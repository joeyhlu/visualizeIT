"""Manufactured finalized-window integration; no real footage, weights or inference.

Fixture identity: offline-planar-export-fixture-v1. Source images, a plane GLB,
masks and declared producer metadata are manufactured controls, not model results.
"""
from __future__ import annotations

import base64
import builtins
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from bench import workbench_artwork_replay as ar
from bench import workbench_cli as cli
from bench import workbench_contract as wc
from bench import workbench_frame_boundary as fb
from bench import workbench_geometry as wg
from bench import workbench_neural as wn
from bench import workbench_offline_export as exporter
from tools import design_context as dc

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ID = "offline-planar-export-fixture-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wc.canonical_json(value))


def plane_glb(path):
    # Analytical plane in metres, front normal -Z; camera looks along +Z.
    vertices = np.array([[-.2, -.15, 0], [.2, -.15, 0], [.2, .15, 0], [-.2, .15, 0]], dtype="<f4")
    indices = np.array([0, 2, 1, 0, 3, 2], dtype="<u2")
    binary = vertices.tobytes() + indices.tobytes()
    gltf = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
                        {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": indices.nbytes}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                       "min": [-.2, -.15, 0], "max": [.2, .15, 0]},
                      {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
    encoded = json.dumps(gltf, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<I4s", len(encoded), b"JSON") + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def make_fixture(root, *, timestamps=False, role="primary_neural", schedule_start=2):
    """Create only synthetic inputs. Paths vary by host; pixel/pose controls do not."""
    root = Path(root)
    ws = cli.Workspace(root / "workspace")
    ws.init()
    inputs = ws.path / "synthetic-inputs"
    inputs.mkdir()
    width, height, count = 128, 96, 8
    images, masks, frames = {}, {}, []
    irregular = [0., .1, .24, .4, .53, .66, .84, 1.02]
    for index in range(count):
        fid = "f" + str(index)
        # Frame index is an independent full-width color marker for order checks.
        image = np.empty((height, width, 3), np.uint8)
        image[:] = [10 + index * 17, 24, 70]
        image[:3, :, :] = [index * 23, 220, 30]
        path = inputs / (fid + ".png")
        Image.fromarray(image, "RGB").save(path)
        images[fid] = image
        mask = np.zeros((height, width), bool)
        mask[13:83, 16:112] = True
        if fid == "f6":
            mask[:] = False
        masks[fid] = mask
        frames.append({"frame_id": fid, "source_frame_index": index,
            "timestamp_s": irregular[index] if timestamps else None,
            "media_time_s": irregular[index] if timestamps else None,
            "image_path": path.name, "image_sha256": sha(path)})
    selection = {"frame_id": "f0", "positive": [[63.5, 47.5]], "negative": [],
                 "reviewed": True, "reviewed_by": "synthetic-test-control"}
    clip_doc = {"schema": wc.CLIP_SCHEMA, "clip_id": "synthetic-offline-clip",
        "title": FIXTURE_ID, "source_kind": "image_sequence",
        "clock_kind": "timestamp_seconds" if timestamps else "frame_index",
        "clock_source": "manufactured timestamps, not capture" if timestamps else None,
        "width": width, "height": height,
        "media": {"root_id": "workbench", "path": "synthetic-inputs", "sha256": None, "bytes": None},
        "frames": frames, "intrinsics": {"status": "dataset_reference",
            "K": [200., 0., 63.5, 0., 200., 47.5, 0., 0., 1.], "source": "analytical synthetic camera"},
        "onboarding": {"keyframe_ids": ["f0"] if schedule_start == 1 else ["f0", "f1"], "selection": selection},
        "scoring_frame_ids": ["f" + str(i) for i in range(schedule_start, count)],
        "evidence_role": "synthetic_test_only", "evaluator_inputs_read": False,
        "provenance": {"fixture_id": FIXTURE_ID, "not_model_execution": True}}
    clip = wc.ClipSpec.from_dict(clip_doc)
    ws.save_clip(clip)
    mask_dir = ws.masks_dir / clip.clip_id / "synthetic"
    mask_dir.mkdir(parents=True)
    entries = {}
    for fid, mask in masks.items():
        path = mask_dir / (fid + ".png")
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)
        entries[fid] = {"path": path.name, "sha256": sha(path)}
    mask_doc = {"schema": cli.MASK_SCHEMA, "clip_id": clip.clip_id,
        "producer": FIXTURE_ID, "evaluator_inputs_read": False, "masks": entries}
    write_json(mask_dir / "manifest.json", mask_doc)
    run_doc = {"schema": wc.RUN_SCHEMA, "run_id": "synthetic-offline-run", "clip_id": clip.clip_id,
        "mode": "focused", "onboarding": clip_doc["onboarding"],
        "scheduled_frame_ids": clip_doc["scoring_frame_ids"],
        "roles": ["primary_neural", "frozen_baseline"] if role == "frozen_baseline" else [role],
        "primary": {key: wn.PRIMARY[key] for key in ("artifact_id", "export_sha256", "export_bytes")},
        "surface_package_id": None, "artwork_kind": "checkerboard", "config": {"mask_set": "synthetic", "fixture": True}}
    run = wc.RunSpec.from_dict(run_doc)
    mesh = inputs / "synthetic-plane.glb"
    plane_glb(mesh)
    pose = np.eye(4)
    pose[2, 3] = 1
    bundle_doc = {"schema": "workbench_metric_bundle_v1", "family": "synthetic-offline-family",
        "glb_path": str(mesh), "glb_sha256": sha(mesh), "obj_id": 1,
        "principal_point_origin": "integer_centers", "seed_pose_m": pose.tolist(),
        "seed_label": "controlled_init:manufactured plane fixture; not evaluator data",
        "seed_frame_id": "f0", "metric_provenance": "dataset_metric_cad"}
    bundle_path = inputs / "metric-bundle.json"
    write_json(bundle_path, bundle_doc)
    bundle = wn.MetricBundle.from_file(bundle_path)
    binding = fb.neural_binding(clip, run, family=bundle.family, bundle=bundle)
    context = fb.FrameContextV1.build(clip, run, role, binding)
    metadata = {"fixture": FIXTURE_ID, "not_serving_admission": True,
        "source_closure": {"current_files": {name: sha(ROOT / name) for name in ar.CONTRACT_CLOSURE_FILES}}}
    contract_sha = wc.canonical_sha256(metadata)
    admission = inputs / "synthetic-metadata-only-admission"
    write_json(admission / "contract.json", {**metadata, "contract_sha256": contract_sha})
    identity = {**(wn.primary_identity() if role == "primary_neural" else wn.baseline_identity()),
        "installed_head_sha256": wn.PRIMARY["export_sha256"] if role == "primary_neural" else None,
        "admission_contract_sha256": contract_sha, "initialization": bundle.seed_label,
        "seed_frame_id": bundle.seed_frame_id, "seed_target_frame_id": binding["seed_target_frame_id"],
        "glb_sha256": bundle.glb_sha256, "metric_units": "metres",
        "principal_point_origin": bundle.principal_point_origin, "effective_K": clip_doc["intrinsics"]["K"],
        "synthetic_declared_identity_only": True}
    rows = []
    for index in range(schedule_start, count):
        frame = clip.frame("f" + str(index))
        state = "lost" if index == 3 else "recovering" if index == 4 else "tracking"
        frame_pose = pose.copy()
        if index == 5:
            frame_pose[2, 3] = -1
        if index == 7:
            frame_pose[0, 3] = .02
        row = wc.FrameResult(run.run_id, clip.clip_id, frame.frame_id, frame.source_frame_index,
            frame.timestamp_s, frame.media_time_s, role, copy.deepcopy(identity),
            "observed", state, "rendered" if state == "tracking" else "suppressed",
            None if state == "tracking" else "synthetic saved " + state,
            tuple(map(tuple, frame_pose)) if state == "tracking" else None,
            "metric" if state == "tracking" else None,
            wg.array_sha256(images[frame.frame_id]) if state == "tracking" else None,
            frame.frame_id, wg.mask_sha256(masks[frame.frame_id]), {}, {})
        rows.append(row.to_dict())
    run_dir = ws.runs_dir / run.run_id
    write_json(run_dir / "run.json", {"clip_id": clip.clip_id, "run": run.to_dict()})
    all_rows = copy.deepcopy(rows)
    if role == "frozen_baseline":
        for baseline_row in rows:
            peer = copy.deepcopy(baseline_row)
            peer["backend_role"] = "primary_neural"
            peer["backend_identity"].pop("head", None)
            peer["backend_identity"].update(wn.primary_identity())
            peer["backend_identity"]["installed_head_sha256"] = wn.PRIMARY["export_sha256"]
            all_rows.append(peer)
    (run_dir / "results.jsonl").write_bytes(b"\n".join(wc.canonical_json(row) for row in all_rows) + b"\n")
    original = wn.NeuralArtwork(bundle, run.onboarding.selection, run.artwork_kind,
        clip_K=clip.intrinsics.K, width=width, height=height, selection_mask=masks["f0"])
    write_json(run_dir / "neural-artwork.json", {**ar._plain(original.record), "record_sha256": original.sha256})
    report = {"run_id": run.run_id, "clip_id": clip.clip_id, "status": "completed",
        "fixture": FIXTURE_ID, "not_model_execution": True,
        "dependencies": {"sources": {name: sha(ROOT / "bench" / (name + ".py")) for name in ar.PRODUCER_MODULES}},
        "stages": {role: {"status": "completed", "execution": cli.boundary_provenance(context)},
                   "masks": {"identity": cli.MaskStore(ws, clip.clip_id, "synthetic").identity()}}}
    if role == "frozen_baseline":
        peer_context = fb.FrameContextV1.build(clip, run, "primary_neural", binding)
        report["stages"]["primary_neural"] = {"status": "completed", "execution": cli.boundary_provenance(peer_context)}
    write_json(run_dir / "report.json", report)
    artwork = np.zeros((16, 32, 4), np.uint8)
    artwork[:8, :16] = [240, 20, 30, 255]
    artwork[:8, 16:] = [10, 220, 40, 128]
    artwork[8:, :16] = [20, 40, 230, 255]
    artwork[8:, 16:] = [220, 220, 20, 0]
    buffer = io.BytesIO()
    Image.fromarray(artwork, "RGBA").save(buffer, format="PNG")
    project = dc.new_project(FIXTURE_ID)
    project.update(projectId="synthetic-offline-project", projectRevisionId="synthetic-project-revision")
    project["selectedArtwork"] = dc.make_artwork(buffer.getvalue(), {"kind": "imported", "producer": None,
        "brief": None, "prompt": None, "effectiveRequest": None, "source": "synthetic asymmetric alpha control"},
        artwork_id="synthetic-alpha-artwork")
    project_path = inputs / "selected.visualizeit-project.json"
    project_path.write_bytes(dc.encode_document(project))
    return dict(workspace=ws, run_id=run.run_id, role=role, project_path=project_path,
        expected_revision=project["projectRevisionId"], bundle_path=bundle_path, bundle_sha256=sha(bundle_path),
        admission_dir=admission, output_dir=ws.derived_dir / "synthetic-export", playback_fps=10.,
        images=images, masks=masks, rows=rows, fixture_id=FIXTURE_ID)


def export_args(fixture):
    return {key: value for key, value in fixture.items() if key not in ("workspace", "images", "masks", "rows", "fixture_id")}


def use_video_source(fixture, *, registered=True):
    """Lossless synthetic input video; all matching producer metadata is rebuilt."""
    f = fixture
    ws = f["workspace"]
    path = ws.path / "synthetic-inputs" / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10., (128, 96))
    if not writer.isOpened():
        raise RuntimeError("Synthetic FFV1 source writer unavailable")
    for index in range(8):
        writer.write(np.ascontiguousarray(f["images"]["f" + str(index)][:, :, ::-1]))
    writer.release()
    clip_path = ws.clip_path("synthetic-offline-clip")
    document = wc.loads_strict(clip_path.read_bytes())
    document["source_kind"] = "video_file"
    document["media"] = {"root_id": "workbench", "path": "synthetic-inputs/source.avi",
        "sha256": sha(path) if registered else None, "bytes": path.stat().st_size if registered else None}
    for frame in document["frames"]:
        frame["image_path"] = None
        frame["image_sha256"] = None
    clip = wc.ClipSpec.from_dict(document)
    write_json(clip_path, clip.to_dict())
    run_dir = ws.runs_dir / f["run_id"]
    run = wc.RunSpec.from_dict(wc.loads_strict((run_dir / "run.json").read_bytes())["run"])
    bundle = wn.MetricBundle.from_file(f["bundle_path"])
    binding = fb.neural_binding(clip, run, family=bundle.family, bundle=bundle)
    report_path = run_dir / "report.json"
    report = wc.loads_strict(report_path.read_bytes())
    for role in run.roles:
        context = fb.FrameContextV1.build(clip, run, role, binding)
        report["stages"][role]["execution"] = cli.boundary_provenance(context)
    write_json(report_path, report)
    return f


class OfflineVideoIntegrationTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="offline-export-test-", dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.fixture = make_fixture(Path(self.temp.name))

    def test_complete_saved_window_roundtrip_failure_frames_and_exact_source_timing(self):
        fixture = self.fixture
        result = exporter.export_saved_window(fixture["workspace"], **export_args(fixture))
        output = fixture["output_dir"]
        self.assertTrue((output / "manifest.json").is_file())
        saved = wc.loads_strict((output / "manifest.json").read_bytes())
        self.assertEqual(saved, result)
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual([row["original"]["frame_id"] for row in result["rows"]], ["f" + str(i) for i in range(2, 8)])
        capture = cv2.VideoCapture(str(output / "playback.avi"))
        self.addCleanup(capture.release)
        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10., places=6)
        decoded = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded.append(np.ascontiguousarray(image[:, :, ::-1]))
        self.assertEqual(len(decoded), 6)
        for index, rgb in enumerate(decoded):
            fid = "f" + str(index + 2)
            # Independent color-order controls exist in every frame's top strip.
            np.testing.assert_array_equal(rgb[:3], fixture["images"][fid][:3])
            self.assertIsNone(result["rows"][index]["original"]["timestamp_s"])
            self.assertIsNone(result["rows"][index]["original"]["media_time_s"])
            outside = ~fixture["masks"][fid]
            np.testing.assert_array_equal(rgb[outside], fixture["images"][fid][outside])
        for fid in ("f3", "f4", "f5", "f6"):
            np.testing.assert_array_equal(decoded[int(fid[1:]) - 2], fixture["images"][fid])
        self.assertFalse(np.array_equal(decoded[0], fixture["images"]["f2"]))
        self.assertFalse(result["new_neural_inference"])
        self.assertEqual(result["source_clip"]["evidence_role"], "synthetic_test_only")
        self.assertEqual(result["frame_accounting"]["scheduled"], 6)
        self.assertEqual(result["frame_accounting"]["exported"], 6)
        self.assertEqual(result["frame_accounting"]["rendered"], 2)
        self.assertEqual(result["frame_accounting"]["suppressed"], 4)

    def test_cli_publishes_only_synthetic_artifacts_on_matching_source(self):
        f = self.fixture
        command = [sys.executable, "-B", "-m", "tools.workbench_offline_export",
            "--workspace", str(f["workspace"].path), "--run-id", f["run_id"], "--role", f["role"],
            "--project", str(f["project_path"]), "--expected-revision", f["expected_revision"],
            "--bundle", str(f["bundle_path"]), "--bundle-sha256", f["bundle_sha256"],
            "--admission-dir", str(f["admission_dir"]), "--output", str(f["output_dir"]), "--playback-fps", "10"]
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual({p.name for p in f["output_dir"].iterdir()}, {"manifest.json", "playback.avi"})

    def test_missing_row_and_changed_head_refuse_before_publication(self):
        f = self.fixture
        path = f["workspace"].runs_dir / f["run_id"] / "results.jsonl"
        original = path.read_bytes()
        path.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
        with self.assertRaisesRegex(ar.Refused, "every scheduled"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(f["output_dir"].exists())
        rows = copy.deepcopy(f["rows"])
        rows[0]["backend_identity"]["installed_head_sha256"] = "0" * 64
        path.write_bytes(b"\n".join(wc.canonical_json(row) for row in rows) + b"\n")
        with self.assertRaisesRegex(ar.Refused, "pinned installed/export"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(f["output_dir"].exists())

    def test_source_mutation_while_composing_refuses_and_preserves_original_run(self):
        f = self.fixture
        original = exporter.compose_planar
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                changed = True
                source = f["workspace"].path / "synthetic-inputs" / "f2.png"
                source.write_bytes(source.read_bytes() + b"changed")
            return result
        with patch.object(exporter, "compose_planar", side_effect=mutate), self.assertRaisesRegex((ar.Refused, cli.WorkbenchCLIError), "hash|changed"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(f["output_dir"].exists())

    def test_explicit_fps_does_not_rewrite_nonuniform_source_times(self):
        f = make_fixture(Path(self.temp.name) / "irregular", timestamps=True)
        result = exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertEqual([row["original"]["timestamp_s"] for row in result["rows"]],
                         [row["timestamp_s"] for row in f["rows"]])
        self.assertIn("CFR", result["playback"]["timing"])

    def test_existing_output_and_budget_refuse_without_overwriting(self):
        f = self.fixture
        f["output_dir"].mkdir()
        sentinel = f["output_dir"] / "preserve.txt"
        sentinel.write_bytes(b"original")
        with self.assertRaisesRegex(ar.Refused, "already exists"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertEqual(sentinel.read_bytes(), b"original")
        args = export_args(f)
        args["output_dir"] = f["workspace"].derived_dir / "budget-refused"
        with patch.object(f["workspace"], "require_budget", side_effect=cli.WorkbenchCLIError("synthetic budget refusal")), self.assertRaisesRegex(cli.WorkbenchCLIError, "budget refusal"):
            exporter.export_saved_window(f["workspace"], **args)
        self.assertFalse(args["output_dir"].exists())

    def test_frozen_baseline_never_claims_primary_installed_weights(self):
        f = make_fixture(Path(self.temp.name) / "baseline", role="frozen_baseline")
        result = exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertEqual(result["role"], "frozen_baseline")
        self.assertTrue(all(row["original"]["backend_identity"]["installed_head_sha256"] is None for row in result["rows"]))
        self.assertIsNone(result["provenance"]["model_pins"]["installed_head_sha256"])

    def test_complete_export_never_imports_or_constructs_a_model(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torchvision", "gotrack"):
                raise AssertionError("Forbidden model import: " + name)
            return original_import(name, *args, **kwargs)
        f = self.fixture
        with patch.object(builtins, "__import__", side_effect=guarded), \
             patch.object(wn, "load_admission", side_effect=AssertionError("No runtime admission")) as admission, \
             patch.object(wn, "GoTrackMetricBackend", side_effect=AssertionError("No backend")) as backend, \
             patch.object(wn, "load_primary_network", side_effect=AssertionError("No weights")) as network:
            result = exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(result["weights_loaded"])
        admission.assert_not_called()
        backend.assert_not_called()
        network.assert_not_called()

    def test_decoded_corruption_refuses_publication_and_cleans_own_staging(self):
        real_capture = cv2.VideoCapture
        class CorruptDecodedVideo:
            def __init__(self, *args, **kwargs):
                self.inner = real_capture(*args, **kwargs)
            def __getattr__(self, name):
                return getattr(self.inner, name)
            def read(self):
                ok, pixels = self.inner.read()
                if ok:
                    pixels[0, 0, 0] ^= 255
                return ok, pixels
        f = self.fixture
        with patch.object(cv2, "VideoCapture", side_effect=CorruptDecodedVideo), \
             self.assertRaisesRegex(ar.Refused, "decoded RGB differs"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(f["output_dir"].exists())
        self.assertEqual(list(f["workspace"].derived_dir.glob(".offline-export-*")), [])

    def test_observed_mask_tampering_refuses_without_success(self):
        f = self.fixture
        mask = f["workspace"].masks_dir / "synthetic-offline-clip" / "synthetic" / "f2.png"
        mask.write_bytes(mask.read_bytes() + b"synthetic tampering")
        with self.assertRaisesRegex(cli.WorkbenchCLIError, "mask file.*changed"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(f["output_dir"].exists())

    def test_video_source_requires_registered_digest_and_size(self):
        valid = use_video_source(self.fixture)
        result = exporter.export_saved_window(valid["workspace"], **export_args(valid))
        self.assertEqual(result["playback"]["frames"], 6)
        invalid = use_video_source(make_fixture(Path(self.temp.name) / "unregistered"), registered=False)
        with self.assertRaisesRegex(ar.Refused, "registered.*SHA|registered.*hash"):
            exporter.export_saved_window(invalid["workspace"], **export_args(invalid))
        self.assertFalse(invalid["output_dir"].exists())

    def test_clip_mutation_after_parse_cannot_rebind_a_new_digest(self):
        f = self.fixture
        actual_project = exporter._project
        def change_clip(*args, **kwargs):
            result = actual_project(*args, **kwargs)
            path = f["workspace"].clip_path("synthetic-offline-clip")
            doc = wc.loads_strict(path.read_bytes())
            doc["intrinsics"]["K"][0] = 999.
            write_json(path, doc)
            return result
        with patch.object(exporter, "_project", side_effect=change_clip), \
             self.assertRaisesRegex(ar.Refused, "clip|closure|changed"):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertFalse(f["output_dir"].exists())

    def test_publication_race_preserves_a_new_empty_destination(self):
        f = self.fixture
        publish = exporter._publish_new_directory
        def create_competing_output(source, destination):
            # Interleave AFTER the exporter's last existence check. POSIX rename
            # would replace this empty directory; exclusive publication must not.
            Path(destination).mkdir()
            return publish(source, destination)
        with patch.object(exporter, "_publish_new_directory", side_effect=create_competing_output), \
             self.assertRaises((FileExistsError, ar.Refused)):
            exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertTrue(f["output_dir"].is_dir())
        self.assertEqual(list(f["output_dir"].iterdir()), [])
        self.assertEqual(list(f["workspace"].derived_dir.glob(".offline-export-*")), [])


if __name__ == "__main__":
    unittest.main()
