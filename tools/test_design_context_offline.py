"""Offline adapter synthetic checks; no browser/native or tracking evidence."""
import base64
from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from tools import design_context as dc
from tools import design_context_offline as offline
from tools.design_studio import StudioStore


def png():
    stream = io.BytesIO()
    Image.new("RGBA", (256, 256), (32, 64, 90, 255)).save(stream, format="PNG")
    return stream.getvalue()


class OfflineContextTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".cache"
        self.temp = tempfile.TemporaryDirectory(prefix="context-offline-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = offline.OfflineProjects(self.root / "projects")
        self.doc = self.adapter.create("Synthetic offline project")

    def draft(self, text="blue waves", capability="waves"):
        self.doc = dc.edit_draft(self.adapter.store, self.doc["projectId"],
            self.doc["projectRevisionId"], text, capability=capability)

    def test_A18_legacy_v1_adoption_preserves_bundles_and_unknown_provenance(self):
        studio = StudioStore(self.root / "studio")
        image = png()
        recipe = {**offline.DEFAULT_RECIPE, "width": 256, "height": 256}
        metadata = studio.create_design({"schema": "visualizeit_design_v1", "name": "Synthetic",
            "prompt": "cream stripes", "recipe": recipe,
            "png_base64": base64.b64encode(image).decode()})
        original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (self.root / "studio").rglob("*") if p.is_file()}
        self.doc = self.adapter.adopt_studio(self.doc["projectId"], self.doc["projectRevisionId"], studio, metadata["id"])
        art = self.doc["selectedArtwork"]
        self.assertEqual(art["id"], "studio-" + metadata["id"])
        self.assertIsNone(art["provenance"]["brief"])
        self.assertIsNone(art["provenance"]["producer"])
        self.assertEqual(base64.b64decode(art["png"]["base64"]), image)
        self.assertEqual(original, {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (self.root / "studio").rglob("*") if p.is_file()})
        reopened = offline.OfflineProjects(self.root / "projects").open(self.doc["projectId"])
        self.assertEqual(reopened["selectedArtwork"], art)

    def test_A04_supported_generation_saved_before_callback_and_repeat_is_current(self):
        self.draft()
        recipe = {**offline.DEFAULT_RECIPE, "pattern": "waves", "width": 256, "height": 256}
        observed = []
        def render(request):
            persisted = self.adapter.store.open(self.doc["projectId"])
            observed.append(persisted)
            self.assertEqual(request, recipe)
            self.assertEqual(persisted["acceptedBrief"]["text"], "blue waves")
            self.assertEqual(dc.draft_status(persisted), "clean")
            self.assertEqual(persisted["generationAttempt"]["effectiveRequest"], recipe)
            return png()
        for _ in range(2):
            self.doc = self.adapter.generate(self.doc["projectId"], self.doc["projectRevisionId"], recipe, render, supported=True)
            self.assertIsNotNone(self.doc["selectedArtwork"])
        self.assertEqual(len(observed), 2)

    def test_A04_save_failure_never_starts_renderer_and_render_failure_keeps_image(self):
        self.doc = self.adapter.import_png(self.doc["projectId"], self.doc["projectRevisionId"], png())
        old_artwork = copy.deepcopy(self.doc["selectedArtwork"])
        self.draft()
        recipe = {**offline.DEFAULT_RECIPE, "pattern": "waves", "width": 256, "height": 256}
        called = []
        def fail_save(stage):
            if stage == "beforeActivation":
                raise OSError("Synthetic interrupted save")
        self.adapter.store.fault = fail_save
        with self.assertRaises(OSError):
            self.adapter.generate(self.doc["projectId"], self.doc["projectRevisionId"], recipe,
                lambda request: called.append(request), supported=True)
        self.assertEqual(called, [])
        self.adapter.store.fault = lambda stage: None
        self.doc = self.adapter.store.open(self.doc["projectId"])
        def fail_render(request):
            raise ValueError("Synthetic renderer failure")
        with self.assertRaises(ValueError):
            self.adapter.generate(self.doc["projectId"], self.doc["projectRevisionId"], recipe, fail_render, supported=True)
        after = self.adapter.store.open(self.doc["projectId"])
        self.assertEqual(after["selectedArtwork"], old_artwork)
        self.assertEqual(after["acceptedBrief"]["text"], "blue waves")
        self.assertIsNotNone(after["generationAttempt"])

    def test_A03_A06_suggestions_are_explicit_and_unsupported_does_not_render(self):
        self.draft("A dragon illustration", "waves")
        suggestions = offline.suggestions(self.doc)
        self.assertTrue(any(item["label"] == "Upload artwork" for item in suggestions))
        self.assertFalse(any("Generate dragon" in item["label"] for item in suggestions))
        self.assertLessEqual(len(suggestions), 5)
        with self.assertRaises(dc.DocumentError):
            self.adapter.generate(self.doc["projectId"], self.doc["projectRevisionId"],
                {**offline.DEFAULT_RECIPE, "pattern": "waves"}, lambda _: self.fail("renderer called"), supported=False)
        self.assertIsNone(self.adapter.store.open(self.doc["projectId"])["acceptedBrief"])

    def test_A05_contextual_chip_only_edits_draft_and_rejects_stale(self):
        self.draft("green dots", "dots")
        offered = offline.suggestions(self.doc)
        first = next(item for item in offered if item["chip"]["actionId"] == "blue-pattern")
        before = copy.deepcopy(self.doc)
        self.doc = dc.apply_chip(self.adapter.store, self.doc["projectId"], self.doc["projectRevisionId"], first["chip"])
        self.assertEqual(self.doc["promptDraft"]["text"], "blue dots")
        self.assertEqual(self.doc["acceptedBrief"], before["acceptedBrief"])
        with self.assertRaises(dc.ConflictError):
            dc.apply_chip(self.adapter.store, self.doc["projectId"], self.doc["projectRevisionId"], first["chip"])

    def test_A07_delayed_output_keeps_other_project_selected_and_remains_in_origin_catalog(self):
        self.draft()
        origin_id = self.doc["projectId"]
        other = self.adapter.create("Other")
        self.adapter.open(origin_id)
        def render(request):
            self.adapter.open(other["projectId"])
            return png()
        result = self.adapter.generate(origin_id, self.doc["projectRevisionId"],
            {**offline.DEFAULT_RECIPE, "pattern": "waves", "width": 256, "height": 256}, render, supported=True)
        self.assertIsNone(result["selectedArtwork"])
        self.assertIsNone(self.adapter.store.open(other["projectId"])["selectedArtwork"])
        self.assertEqual(len(self.adapter.store.catalog(origin_id)), 1)
        self.assertEqual(self.adapter.active_project_id, other["projectId"])

    def test_A09_A19_snapshot_owns_bytes_context_only_and_cancel(self):
        image = png()
        external = self.root / "synthetic.png"
        external.write_bytes(image)
        self.doc = self.adapter.import_png(self.doc["projectId"], self.doc["projectRevisionId"], offline.read_bounded(external, 16_777_216))
        external.unlink()
        snapshot = self.adapter.store.export_snapshot(self.doc["projectId"])
        destination = offline.OfflineProjects(self.root / "destination")
        imported = destination.import_snapshot(snapshot)
        self.assertEqual(snapshot, destination.store.export_snapshot(imported["projectId"]))
        self.assertEqual(base64.b64decode(imported["selectedArtwork"]["png"]["base64"]), image)
        current = destination.active_project_id
        self.assertIsNone(destination.import_snapshot(snapshot, "cancel"))
        self.assertEqual(current, destination.active_project_id)
        empty = destination.create("Context only")
        self.assertIsNone(dc.decode_document(destination.store.export_snapshot(empty["projectId"]))["selectedArtwork"])

    def test_A22_cli_export_requires_scope_confirmation_and_does_not_overwrite(self):
        target = self.root / "project.visualizeit-project.json"
        out, err = io.StringIO(), io.StringIO()
        args = ["--root", str(self.root / "projects"), "export", self.doc["projectId"], "--output", str(target)]
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(offline.main(args), 2)
        self.assertFalse(target.exists())
        self.assertIn("saved messages", err.getvalue())
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(offline.main(args + ["--confirm-full-transcript"]), 0)
        saved = target.read_bytes()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(offline.main(args + ["--confirm-full-transcript"]), 2)
        self.assertEqual(target.read_bytes(), saved)

    def test_review_A17_cli_recovery_is_explicit_outside_portable_state(self):
        previous = copy.deepcopy(self.doc)
        current = dc.append_message(self.adapter.store, self.doc["projectId"],
            self.doc["projectRevisionId"], "New revision")
        pointer = self.adapter.store._pointer(current["projectId"])
        self.adapter.store._manifest_path(current["projectId"], pointer["current"]).write_bytes(b"corrupt")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            result = offline.main(["--root", str(self.root / "projects"), "open", current["projectId"]])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(out.getvalue()), previous)
        self.assertIn("Recovered the previous complete design", err.getvalue())
        self.assertNotIn("recovered", json.loads(out.getvalue()))


if __name__ == "__main__":
    unittest.main()
