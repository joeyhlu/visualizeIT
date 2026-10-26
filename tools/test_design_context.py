"""Synthetic core checks. These do not assert native/runtime/attachment behavior."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

from PIL import Image
from tools import design_context as dc


TEMP_ROOT = Path(__file__).resolve().parents[1] / ".cache" / "design-context-core-tests"


def temporary_directory():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEMP_ROOT)


def png(width=2, height=3, color="red"):
    stream = io.BytesIO()
    Image.new("RGBA", (width, height), color).save(stream, format="PNG")
    return stream.getvalue()


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)


def imported(data=None, identifier=None):
    return dc.make_artwork(data or png(), {"kind": "imported", "producer": None,
        "brief": None, "prompt": None, "effectiveRequest": None, "source": None}, identifier)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = temporary_directory()
        self.addCleanup(self.tmp.cleanup)
        self.store = dc.ProjectStore(self.tmp.name)
        self.doc = self.store.create("Synthetic")

    def change(self, function, *args, **kwargs):
        self.doc = function(self.store, self.doc["projectId"], self.doc["projectRevisionId"], *args, **kwargs)
        return self.doc

    def accepted(self, text="B1"):
        return self.change(dc.accept_brief, text)

    def prepared(self):
        self.change(dc.edit_draft, "Actual typography", "typography", {"text": "Actual typography"})
        self.doc, attempt = dc.prepare_generation(self.store, self.doc["projectId"], self.doc["projectRevisionId"],
            "typography", {"text": "Actual typography"}, True)
        return attempt

    def snapshot_files(self):
        return {str(path.relative_to(self.tmp.name)): path.read_bytes() for path in Path(self.tmp.name).rglob("*") if path.is_file()}

    def test_A01_retains_35_messages_brief_draft_and_exact_png(self):
        self.doc["messages"] = [{"id": "message-" + str(i), "role": "user", "source": "user", "text": "Line " + str(i), "artworkIds": []} for i in range(35)]
        self.doc = self.store.update(self.doc, self.doc["projectRevisionId"])
        self.accepted()
        self.change(dc.edit_draft, "Unaccepted draft")
        art = imported()
        self.change(dc.select_artwork, art)
        reopened = dc.ProjectStore(self.tmp.name).open(self.doc["projectId"])
        self.assertEqual(self.doc, reopened)
        self.assertEqual(len(reopened["messages"]), 35)
        self.assertEqual(dc.draft_status(reopened), "unaccepted")
        self.assertEqual(reopened["selectedArtwork"], art)

    def test_A02_chat_does_not_accept_and_proposal_stales_manual_draft(self):
        self.accepted()
        self.change(dc.edit_draft, "D2")
        self.change(dc.append_message, "C3", "assistant", source="local-proposal")
        self.assertEqual(self.doc["acceptedBrief"]["text"], "B1")
        self.assertEqual(self.doc["promptDraft"]["text"], "D2")
        self.change(dc.accept_brief, "C3")
        self.assertEqual(dc.draft_status(self.doc), "stale")
        self.assertEqual(self.doc["promptDraft"]["text"], "D2")
        self.change(dc.accept_brief)
        self.assertEqual(dc.draft_status(self.doc), "clean")

    def test_A03_unsupported_idea_persists_without_substitution(self):
        art = imported()
        self.change(dc.select_artwork, art)
        self.accepted("Dragon illustration")
        before = self.snapshot_files()
        with self.assertRaises(dc.DocumentError):
            dc.prepare_generation(self.store, self.doc["projectId"], self.doc["projectRevisionId"], "dragon", {}, False)
        self.assertEqual(before, self.snapshot_files())
        self.assertEqual(self.store.open(self.doc["projectId"])["selectedArtwork"], art)

    def test_A04_accept_before_execution_repeat_and_save_failure(self):
        self.change(dc.select_artwork, imported())
        previous_art = self.doc["selectedArtwork"]
        attempt = self.prepared()
        self.assertEqual(attempt["contextRevisionId"], self.doc["projectRevisionId"])
        self.assertEqual(self.store.open(self.doc["projectId"])["generationAttempt"], attempt)
        self.assertEqual(dc.draft_status(self.doc), "clean")
        self.doc, repeated = dc.prepare_generation(self.store, self.doc["projectId"], self.doc["projectRevisionId"], "typography", {"text": "Actual typography"}, True)
        self.assertNotEqual(attempt["id"], repeated["id"])
        with self.assertRaises(dc.DocumentError):
            dc.complete_generation(self.store, repeated, b"not PNG", {"app": "synthetic", "version": "1", "capability": "typography"}, self.doc["projectId"])
        self.assertEqual(self.store.open(self.doc["projectId"])["selectedArtwork"], previous_art)
        self.assertEqual(len(self.store.catalog(self.doc["projectId"])), 1)
        self.assertEqual(self.doc["acceptedBrief"]["text"], "Actual typography")
        calls = []
        def fail(stage):
            if stage == "beforeActivation":
                raise OSError("synthetic disk full")
        self.store.fault = fail
        with self.assertRaises(OSError):
            prepared, request = dc.prepare_generation(self.store, self.doc["projectId"], self.doc["projectRevisionId"], "typography", {}, True)
            calls.append("renderer would start here")
        self.assertEqual(calls, [])
        self.assertEqual(self.store.open(self.doc["projectId"]), self.doc)

    def chip(self, kind, payload):
        return {"id": dc.new_id(), "actionId": "synthetic-" + kind, "projectId": self.doc["projectId"],
            "contextRevisionId": self.doc["projectRevisionId"],
            "draftRevisionId": self.doc["promptDraft"]["revisionId"] if self.doc["promptDraft"] else None,
            "briefRevisionId": self.doc["acceptedBrief"]["revisionId"] if self.doc["acceptedBrief"] else None,
            "effect": {"type": kind, "payload": payload}}

    def test_A05_all_chip_effect_types_exactly_once_and_stale(self):
        for kind, payload in (("clarify", {"question": "Which color?"}), ("import", {}),
                              ("draft", {"text": "Explicit typography", "capability": "typography", "settings": {}})):
            with self.subTest(kind=kind):
                chip = self.chip(kind, payload)
                brief = copy.deepcopy(self.doc["acceptedBrief"])
                self.change(dc.apply_chip, chip)
                self.assertEqual(self.doc["acceptedBrief"], brief)
                self.assertIsNone(self.doc["generationAttempt"])
                self.assertEqual(self.doc["chosenSuggestionEffects"][-1]["effect"], chip["effect"])
                with self.assertRaises(dc.ConflictError):
                    self.change(dc.apply_chip, chip)
        chip = self.chip("draft", {"text": "Stale"})
        self.change(dc.edit_draft, "New edit")
        with self.assertRaises(dc.ConflictError):
            self.change(dc.apply_chip, chip)

    def test_A06_explicit_typography_is_labelled_typography(self):
        self.accepted("Dragon illustration")
        attempt = self.prepared()
        self.doc = dc.complete_generation(self.store, attempt, png(), {"app": "synthetic", "version": "1", "capability": "typography"}, self.doc["projectId"])
        self.assertEqual(self.doc["selectedArtwork"]["provenance"]["producer"]["capability"], "typography")
        self.assertEqual(self.doc["selectedArtwork"]["provenance"]["brief"]["text"], "Actual typography")

    def test_A07_project_switch_and_delayed_result_do_not_contaminate(self):
        attempt = self.prepared()
        a = copy.deepcopy(self.doc)
        b = self.store.create("B")
        b = dc.append_message(self.store, b["projectId"], b["projectRevisionId"], "B-only")
        result = dc.complete_generation(self.store, attempt, png(), {"app": "synthetic", "version": None, "capability": "typography"}, b["projectId"])
        self.assertEqual(self.store.open(b["projectId"]), b)
        self.assertEqual(result, a)
        self.assertIn(attempt["id"], self.store.catalog(a["projectId"]))
        self.assertEqual(result["generationAttempt"]["effectiveRequest"], attempt["effectiveRequest"])
        self.assertIsNone(self.store.open(a["projectId"])["selectedArtwork"])
        self.assertEqual(self.store.open(b["projectId"])["projectRevisionId"], b["projectRevisionId"])

    def test_A08_creation_record_survives_new_intent_and_selection(self):
        attempt = self.prepared()
        self.doc = dc.complete_generation(self.store, attempt, png(), {"app": "synthetic", "version": "1", "capability": "typography"}, self.doc["projectId"])
        art = copy.deepcopy(self.doc["selectedArtwork"])
        self.accepted("B2")
        self.change(dc.select_artwork, art)
        self.assertEqual(self.doc["acceptedBrief"]["text"], "B2")
        self.assertEqual(self.doc["selectedArtwork"]["provenance"]["brief"], attempt["brief"])

    def test_A09_A10_A11_snapshot_round_trip_context_only_and_local_history(self):
        fixture = Path(__file__).parent / "fixtures/design-context/context-only.visualizeit-project.json"
        context = dc.decode_document(fixture.read_bytes())
        self.store.import_snapshot(dc.encode_document(context))
        self.assertEqual(dc.decode_document(self.store.export_snapshot(context["projectId"])), context)
        self.assertFalse(dc.artwork_available(self.store, context["projectId"], "historical-excluded-artwork"))
        y = imported(identifier="Y")
        self.change(dc.select_artwork, y)
        self.change(dc.append_message, "Historical Y", artwork_ids=["Y"])
        x = imported(png(color="blue"), "X")
        x["provenance"]["effectiveRequest"] = {"unknownGenerator": {"opaque": "file:///never-execute"}}
        self.change(dc.select_artwork, x)
        self.doc["extensions"] = {"opaque": ["<script>inert</script>", {"custom": 7}]}
        self.doc = self.store.update(self.doc, self.doc["projectRevisionId"])
        data = self.store.export_snapshot(self.doc["projectId"])
        with temporary_directory() as destination:
            other = dc.ProjectStore(destination)
            other.import_snapshot(data)
            self.assertEqual(other.export_snapshot(self.doc["projectId"]), data)
            self.assertNotIn("Y", other.catalog(self.doc["projectId"]))
        self.assertIn("Y", self.store.catalog(self.doc["projectId"]))
        self.assertEqual(dc.decode_document(data)["messages"][-1]["artworkIds"], ["Y"])
        self.assertEqual(set(self.store.catalog(self.doc["projectId"])), {"X", "Y"})

    def test_A12_import_idempotent_update_copy_cancel(self):
        initial = copy.deepcopy(self.doc)
        encoded = dc.encode_document(initial)
        files = self.snapshot_files()
        self.assertEqual(self.store.import_snapshot(encoded), initial)
        self.assertEqual(files, self.snapshot_files())
        self.change(dc.append_message, "Changed")
        with self.assertRaises(dc.ConflictError):
            self.store.import_snapshot(encoded)
        files = self.snapshot_files()
        self.assertIsNone(self.store.import_snapshot(encoded, "cancel"))
        self.assertEqual(files, self.snapshot_files())
        copied = self.store.import_snapshot(encoded, "copy")
        self.assertNotEqual(copied["projectId"], initial["projectId"])
        self.assertNotEqual(copied["projectRevisionId"], initial["projectRevisionId"])
        self.assertEqual(copied["origin"], {"projectId": initial["projectId"], "revisionId": initial["projectRevisionId"]})
        self.assertEqual(self.store.import_snapshot(encoded, "update"), initial)

    def test_A13_semantic_and_artwork_conflicts_preserve_all_files(self):
        self.change(dc.select_artwork, imported(identifier="immutable"))
        candidate = copy.deepcopy(self.doc)
        candidate["extensions"] = {"conflict": True}
        before = self.snapshot_files()
        with self.assertRaises(dc.ConflictError):
            self.store.import_snapshot(dc.encode_document(candidate), "update")
        self.assertEqual(before, self.snapshot_files())
        candidate = copy.deepcopy(self.doc)
        candidate["projectRevisionId"] = dc.new_id()
        candidate["selectedArtwork"]["provenance"]["source"] = "fabricated change"
        with self.assertRaises(dc.ConflictError):
            self.store.import_snapshot(dc.encode_document(candidate), "update")
        self.assertEqual(before, self.snapshot_files())

    def test_A14_json_invalid_utf8_duplicates_nonfinite_unicode_and_depth(self):
        bad = [b"{", b"\xff", b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":"\\ud800"}', b"[" * 33 + b"]" * 33]
        for data in bad:
            with self.subTest(data=data[:25]), self.assertRaises(dc.DocumentError):
                dc.decode_document(data)
        for mutation in (lambda d: d.update(unknown=True), lambda d: d.update(schemaVersion=2),
                         lambda d: d.update(messages=[{"id": "dup", "role": "user", "source": "user", "text": "", "artworkIds": []}] * 2),
                         lambda d: d.update(promptDraft={"edited": True})):
            doc = copy.deepcopy(self.doc)
            mutation(doc)
            with self.assertRaises(dc.DocumentError):
                dc.validate_document(doc)

    def test_A14_png_invalid_framing_crc_base64_length_hash_and_decoder(self):
        art = imported()
        variants = [b"no PNG", png()[:-1], png() + b"trailing", png()[:20] + b"bad" + png()[23:]]
        for raw in variants:
            with self.subTest(raw=raw[:15]), self.assertRaises(dc.DocumentError):
                imported(raw)
        for field, value in (("byteCount", 1), ("sha256", "0" * 64), ("width", 1), ("base64", art["png"]["base64"] + "\n"),
                             ("base64", "data:image/png;base64," + art["png"]["base64"]), ("base64", "AB==")):
            candidate = copy.deepcopy(self.doc)
            candidate["selectedArtwork"] = copy.deepcopy(art)
            candidate["selectedArtwork"]["png"][field] = value
            with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(dc.DocumentError):
                dc.validate_document(candidate)
        invalid_raster = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 3, 8, 6, 0, 0, 0)) + chunk(b"IDAT", b"not-zlib") + chunk(b"IEND", b"")
        with self.assertRaises(dc.DocumentError):
            dc.validate_png(invalid_raster)

    def test_A15_exact_and_over_metadata_extensions_messages_effects_depth_file(self):
        doc = copy.deepcopy(self.doc)
        doc["title"] = ""
        doc["title"] = "x" * (dc.MAX_METADATA - len(dc._json(doc)))
        self.assertEqual(len(dc._json(doc)), dc.MAX_METADATA)
        dc.validate_document(doc)
        doc["title"] += "x"
        with self.assertRaises(dc.DocumentError):
            dc.validate_document(doc)
        doc = copy.deepcopy(self.doc)
        doc["extensions"] = {"x": "x" * (dc.MAX_EXTENSIONS - len(dc._json({"x": ""})))}
        dc.validate_document(doc)
        doc["extensions"]["x"] += "x"
        with self.assertRaises(dc.DocumentError):
            dc.validate_document(doc)
        for key in ("messages", "chosenSuggestionEffects"):
            doc = copy.deepcopy(self.doc)
            if key == "messages":
                entry = {"id": "x", "role": "user", "source": "user", "text": "", "artworkIds": []}
            else:
                entry = {"id": "x", "actionId": "a", "projectId": doc["projectId"], "contextRevisionId": doc["projectRevisionId"],
                    "draftRevisionId": None, "briefRevisionId": None, "effect": {"type": "import", "payload": {}}, "resultDraftRevisionId": None}
            doc[key] = [dict(entry, id="i" + str(i)) for i in range(1000)]
            dc.validate_document(doc)
            doc[key].append(dict(entry, id="over"))
            with self.subTest(cap=key), self.assertRaises(dc.DocumentError):
                dc.validate_document(doc)
        doc = copy.deepcopy(self.doc)
        nested = doc["extensions"]
        for _ in range(30):
            nested["x"] = {}
            nested = nested["x"]
        dc.validate_document(doc)
        nested["x"] = {}
        with self.assertRaises(dc.DocumentError):
            dc.validate_document(doc)
        data = dc.encode_document(self.doc)
        padded = data + b" " * (dc.MAX_FILE - len(data))
        self.assertEqual(dc.decode_document(padded), self.doc)
        with self.assertRaises(dc.DocumentError):
            dc.decode_document(padded + b" ")

    def test_A15_exact_and_over_png_bytes_dimensions_and_pixels(self):
        raw = png()
        raw = raw[:-12] + chunk(b"vpAg", b"x" * (dc.MAX_PNG - len(raw) - 12)) + raw[-12:]
        self.assertEqual(len(raw), dc.MAX_PNG)
        self.assertEqual(dc.validate_png(raw), (2, 3))
        with self.assertRaises(dc.DocumentError):
            dc.validate_png(raw + b"x")
        self.assertEqual(dc.validate_png(png(4096, 4096)), (4096, 4096))
        for width, height in ((4097, 1), (1, 4097), (4097, 4096)):
            data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            with self.subTest(width=width, height=height), self.assertRaises(dc.DocumentError):
                dc.validate_png(data)

    def test_A16_fault_each_copy_on_write_boundary_recovers_complete_state(self):
        for stage in ("assetStaging", "assetCreated", "manifestStaging", "beforeActivation", "afterActivation"):
            with self.subTest(stage=stage), temporary_directory() as root:
                store = dc.ProjectStore(root)
                initial = store.create()
                initial = dc.select_artwork(store, initial["projectId"], initial["projectRevisionId"], imported())
                candidate = copy.deepcopy(initial)
                candidate["selectedArtwork"] = imported(png(color="blue"))
                def fail(point):
                    if point == stage:
                        raise OSError("synthetic " + stage)
                store.fault = fail
                with self.assertRaises(OSError):
                    store.update(candidate, initial["projectRevisionId"])
                reopened = dc.ProjectStore(root).open(initial["projectId"])
                if stage == "afterActivation":
                    self.assertEqual(reopened["selectedArtwork"], candidate["selectedArtwork"])
                    pointer = store._pointer(initial["projectId"])
                    previous, _ = store._manifest(initial["projectId"], pointer["previous"])
                    self.assertEqual(previous, initial)
                else:
                    self.assertEqual(reopened, initial)

    def test_A17_corrupt_current_recovers_previous_both_preserved(self):
        initial = copy.deepcopy(self.doc)
        self.change(dc.append_message, "New state")
        pointer = self.store._pointer(self.doc["projectId"])
        directory = self.store._directory(self.doc["projectId"]) / "revisions"
        (self.store._manifest_path(self.doc["projectId"], pointer["current"])).write_bytes(b"corrupt")
        self.assertEqual(self.store.open(self.doc["projectId"]), initial)
        self.assertTrue(self.store.recovered)
        (self.store._manifest_path(self.doc["projectId"], pointer["previous"])).write_bytes(b"also corrupt")
        files = self.snapshot_files()
        with self.assertRaises(dc.RecoveryError):
            self.store.open(self.doc["projectId"])
        self.assertEqual(files, self.snapshot_files())

    def test_A19_owned_bytes_survive_deleted_source_and_cancel_has_no_side_effect(self):
        source = Path(self.tmp.name) / "external.png"
        source.write_bytes(png())
        art = imported(source.read_bytes())
        self.change(dc.select_artwork, art)
        source.unlink()
        self.assertEqual(self.store.open(self.doc["projectId"])["selectedArtwork"], art)
        self.assertEqual(base64.b64decode(art["png"]["base64"]), png())

    def test_A20_A22_no_geometry_or_execution_and_no_transcript_trimming(self):
        text = "C:\\private\\user-note <script>not markup</script> https://invalid.example/never-fetch"
        self.change(dc.append_message, text)
        self.assertEqual(dc.decode_document(self.store.export_snapshot(self.doc["projectId"]))["messages"][0]["text"], text)
        for key in ("pose", "mask", "attachment", "cachePath", "geometry"):
            doc = copy.deepcopy(self.doc)
            doc[key] = {}
            with self.assertRaises(dc.DocumentError):
                dc.validate_document(doc)

    def test_stale_writer_stale_result_and_forged_attempt(self):
        attempt = self.prepared()
        stale = copy.deepcopy(self.doc)
        self.change(dc.append_message, "Revision changed")
        with self.assertRaises(dc.ConflictError):
            self.store.update(stale, stale["projectRevisionId"])
        result = dc.complete_generation(self.store, attempt, png(), {"app": "synthetic", "version": None, "capability": "typography"}, self.doc["projectId"])
        self.assertEqual(result, self.doc)
        forged = dict(attempt, effectiveRequest={"changed": True})
        with self.assertRaises(dc.ConflictError):
            dc.complete_generation(self.store, forged, png(), {"app": "synthetic", "version": None, "capability": "typography"}, self.doc["projectId"])

    def test_imported_opaque_ids_never_become_storage_paths(self):
        for identifier in (".", "..", ":", "C:", "CON", "aux", "trailing."):
            with self.subTest(identifier=identifier):
                document = dc.new_project(identifier)
                document["projectId"] = identifier
                document["projectRevisionId"] = ".."
                imported_doc = self.store.import_snapshot(dc.encode_document(document))
                self.assertEqual(imported_doc, document)
                self.assertEqual(self.store.open(identifier), document)
                directory = self.store._directory(identifier)
                self.assertEqual(directory.parent, Path(self.tmp.name) / "projects")
                self.assertEqual(len(directory.name), 64)
                self.assertIn(identifier, {entry["projectId"] for entry in self.store.list()})

    def test_cross_project_artwork_identity_conflict_rejects_before_activation(self):
        self.change(dc.select_artwork, imported(identifier="same-id"))
        another = dc.new_project("Other")
        another["selectedArtwork"] = imported(png(color="blue"), "same-id")
        files = self.snapshot_files()
        with self.assertRaises(dc.ConflictError):
            self.store.import_snapshot(dc.encode_document(another))
        self.assertEqual(files, self.snapshot_files())

    def test_missing_assets_recover_previous_and_malformed_history_is_inert(self):
        initial = copy.deepcopy(self.doc)
        self.change(dc.select_artwork, imported())
        digest = self.doc["selectedArtwork"]["png"]["sha256"]
        (Path(self.tmp.name) / "assets" / (digest + ".png")).unlink()
        self.assertEqual(self.store.open(self.doc["projectId"]), initial)
        self.assertTrue(self.store.recovered)
        # An unrelated orphan manifest is never traversed as committed history.
        directory = self.store._directory(self.doc["projectId"]) / "revisions"
        (directory / "unreferenced.json").write_text("[]")
        self.doc = self.store.update(initial, initial["projectRevisionId"])
        self.assertEqual(self.store.open(self.doc["projectId"]), self.doc)
        self.assertEqual(self.store.import_snapshot(dc.encode_document(self.doc)), self.doc)

    def test_store_serializes_multiple_instances_and_processes(self):
        project_id, revision = self.doc["projectId"], self.doc["projectRevisionId"]
        def append_from_instance(index):
            try:
                dc.append_message(dc.ProjectStore(self.tmp.name), project_id, revision, str(index))
                return "ok"
            except dc.ConflictError:
                return "conflict"
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(append_from_instance, range(4)))
        self.assertEqual(outcomes.count("ok"), 1)
        self.assertEqual(outcomes.count("conflict"), 3)
        current = self.store.open(project_id)
        script = """import sys
from tools import design_context as dc
store=dc.ProjectStore(sys.argv[1])
try:
    dc.append_message(store,sys.argv[2],sys.argv[3],sys.argv[4])
    print('ok')
except dc.ConflictError:
    print('conflict')
"""
        processes = [subprocess.Popen([sys.executable, "-B", "-c", script, self.tmp.name, project_id,
            current["projectRevisionId"], str(index)], cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for index in range(2)]
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            outcomes.append(stdout.strip())
        self.assertEqual(sorted(outcomes), ["conflict", "ok"])
        self.assertEqual(len(self.store.open(project_id)["messages"]), 2)
        self.assertTrue((Path(self.tmp.name) / ".store.lock").exists())

    def test_dangling_structural_references_and_missing_bytes_reject(self):
        # Excluded historical artwork IDs are allowed; malformed/non-ID targets
        # and foreign attempts cannot gain execution authority.
        mutations = [lambda d: d.update(messages=[{"id": "m", "role": "user", "source": "user", "text": "", "artworkIds": [""]}]),
                     lambda d: d.update(selectedArtwork={"id": "a", "png": {}, "provenance": {}}),
                     lambda d: d.update(parentRevisionId=d["projectRevisionId"])]
        files = self.snapshot_files()
        for mutate in mutations:
            document = copy.deepcopy(self.doc)
            mutate(document)
            with self.assertRaises(dc.DocumentError):
                self.store.import_snapshot(dc._json(document))
        self.assertEqual(files, self.snapshot_files())

    def test_review_A15_A16_depth_boundary_persists_and_reopens_with_catalog(self):
        for selected in (False, True):
            document = dc.new_project("Exact portable nesting boundary")
            if selected:
                artwork = imported()
                nested = artwork["provenance"]["effectiveRequest"] = {}
                for _ in range(28):
                    nested["x"] = {}
                    nested = nested["x"]
                document["selectedArtwork"] = artwork
            else:
                nested = document["extensions"]
                for _ in range(30):
                    nested["x"] = {}
                    nested = nested["x"]
            dc.validate_document(document)
            persisted = self.store.import_snapshot(dc.encode_document(document))
            self.assertEqual(self.store.open(document["projectId"]), persisted)
            self.assertEqual(self.store.export_snapshot(document["projectId"]), dc.encode_document(document))
            reopened = dc.ProjectStore(self.tmp.name)
            self.assertIn(document["projectId"], {row["projectId"] for row in reopened.list()})

    def test_review_A16_local_envelope_overflow_rejects_before_any_write(self):
        pointer = self.store._pointer(self.doc["projectId"])
        current_path = self.store._manifest_path(self.doc["projectId"], pointer["current"])
        limit = len(current_path.read_bytes()) + 50
        candidate = copy.deepcopy(self.doc)
        candidate["title"] = "x" * 2000
        before = self.snapshot_files()
        with patch.object(dc, "MAX_FILE", limit), self.assertRaises(dc.DocumentError):
            self.store.update(candidate, self.doc["projectRevisionId"])
        self.assertEqual(before, self.snapshot_files())
        self.assertEqual(self.store.open(self.doc["projectId"]), self.doc)

    def test_review_A13_bool_integer_nested_provenance_is_immutable(self):
        for value in (True, {"nested": [False, {"value": True}]}):
            document = dc.new_project("Typed provenance")
            artwork = imported()
            artwork["provenance"]["effectiveRequest"] = {"opaque": value}
            document["selectedArtwork"] = artwork
            self.store.import_snapshot(dc.encode_document(document))
            conflict = copy.deepcopy(document)
            conflict["parentRevisionId"] = document["projectRevisionId"]
            conflict["projectRevisionId"] = dc.new_id()
            replacement = 1 if value is True else {"nested": [0, {"value": 1}]}
            conflict["selectedArtwork"]["provenance"]["effectiveRequest"] = {"opaque": replacement}
            before = self.snapshot_files()
            with self.assertRaises(dc.ConflictError):
                self.store.import_snapshot(dc.encode_document(conflict), choice="update")
            self.assertEqual(before, self.snapshot_files())
            self.assertEqual(self.store.open(document["projectId"]), document)

    def test_review_A13_forged_attempt_changes_json_type_rejects(self):
        self.change(dc.edit_draft, "blue dots", "dots")
        self.doc, attempt = dc.prepare_generation(self.store, self.doc["projectId"],
            self.doc["projectRevisionId"], "dots", {"opaque": True}, True)
        forged = copy.deepcopy(attempt)
        forged["effectiveRequest"] = {"opaque": 1}
        before = self.snapshot_files()
        with self.assertRaises(dc.ConflictError):
            dc.complete_generation(self.store, forged, png(),
                {"app": "Synthetic", "version": "1", "capability": "dots"}, self.doc["projectId"])
        self.assertEqual(before, self.snapshot_files())


if __name__ == "__main__":
    unittest.main()
