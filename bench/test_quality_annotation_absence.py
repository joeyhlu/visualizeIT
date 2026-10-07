"""CPU-only tests for explicit fully hidden image annotations."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_annotations as annotations
from . import quality_evaluate as evaluator
from .vision import cv2


FIXTURE_SELECTION = tuple((100 + index, "uniform") for index in range(40))
FIXTURE_SCORED_IDS = tuple(range(100, 340))
FIXTURE_RESOLUTION = (32, 24)
FIXTURE_SOURCE_HASH = "a" * 64


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def polygon(points, hole=False):
    return [{"points": points, "hole": hole}]


def png(mask):
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise AssertionError("Could not encode synthetic mask")
    return encoded.tobytes()


class AnnotationAbsenceTests(unittest.TestCase):
    def setUp(self):
        self.trusted_selections = annotations.FROZEN_ANNOTATION_SELECTIONS
        self.trusted_source_hashes = annotations.FROZEN_SOURCE_SHA256
        self.trusted_resolution = annotations.FROZEN_RESOLUTION
        self._patches = [
            patch.object(annotations, "FROZEN_ANNOTATION_SELECTIONS", {"keyboard": FIXTURE_SELECTION}),
            patch.object(annotations, "FROZEN_SCORED_FRAME_IDS", {"keyboard": FIXTURE_SCORED_IDS}),
            patch.object(annotations, "FROZEN_RESOLUTION", FIXTURE_RESOLUTION),
            patch.object(annotations, "FROZEN_SOURCE_SHA256", {"keyboard": FIXTURE_SOURCE_HASH}),
        ]
        for item in self._patches:
            item.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        for item in reversed(self._patches):
            item.stop()

    def document(self):
        return {
            "schema_version": 1,
            "object": "keyboard",
            "resolution": list(FIXTURE_RESOLUTION),
            "selection_status": annotations.PENDING_SELECTION_STATUS,
            "operator": None,
            "source_sha256": FIXTURE_SOURCE_HASH,
            "frames": [
                {
                    "frame_id": frame_id,
                    "image": f"{frame_id}.jpg",
                    "group": group,
                    "status": "pending",
                    "visible_object": [],
                    "overlapping_hands": [],
                    "landmarks": [],
                    "review_notes": "",
                }
                for frame_id, group in FIXTURE_SELECTION
            ],
        }

    def hidden_row(self, *, hands=False):
        doc = self.document()
        row = doc["frames"][0]
        row["status"] = "reviewed"
        row["visibility"] = "fully_hidden"
        if hands:
            row["overlapping_hands"] = polygon(rectangle(20, 15, 28, 21))
        doc["operator"] = "Synthetic reviewer"
        return doc

    def patch_import_root(self):
        repository = self.root / "repository"
        canonical = repository / "artifacts/model-quality/annotations/keyboard/annotations.json"
        canonical.parent.mkdir(parents=True)
        canonical.write_text(json.dumps(self.document()), encoding="utf-8")
        return repository

    def test_trusted_selection_literals_match_all_canonical_pending_rows(self):
        from . import quality_mask_audit as mask_audit

        self.assertEqual(dict(self.trusted_selections),
                         dict(mask_audit.FROZEN_ANNOTATION_SELECTIONS))
        repository = Path(__file__).resolve().parents[1]
        for alias, selection in self.trusted_selections.items():
            canonical = json.loads((repository / "artifacts/model-quality/annotations" /
                                   alias / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(canonical["source_sha256"], self.trusted_source_hashes[alias])
            self.assertEqual(canonical["resolution"], list(self.trusted_resolution))
            actual = [(row["frame_id"], row["group"], row["image"]) for row in canonical["frames"]]
            expected = [(frame_id, group, f"{frame_id}.jpg") for frame_id, group in selection]
            self.assertEqual(actual, expected, alias)
            self.assertTrue(all(row["status"] == "pending" for row in canonical["frames"]))
        with patch.multiple(
            annotations,
            FROZEN_ANNOTATION_SELECTIONS=self.trusted_selections,
            FROZEN_SOURCE_SHA256=self.trusted_source_hashes,
            FROZEN_RESOLUTION=self.trusted_resolution,
        ):
            for alias in self.trusted_selections:
                canonical = json.loads((repository / "artifacts/model-quality/annotations" /
                                       alias / "annotations.json").read_text(encoding="utf-8"))
                self.assertIs(annotations.validate_annotation_document(canonical, alias), canonical)

    def test_fully_hidden_review_accepts_no_landmarks_and_optional_closed_hands(self):
        no_hands = self.hidden_row()
        self.assertIs(annotations.validate_annotation_document(no_hands), no_hands)
        with_hands = self.hidden_row(hands=True)
        self.assertIs(annotations.validate_annotation_document(with_hands), with_hands)

    def test_fully_hidden_review_rejects_contradictory_target_or_landmarks(self):
        with_target = self.hidden_row()
        with_target["frames"][0]["visible_object"] = polygon(rectangle(3, 3, 8, 9))
        with self.assertRaisesRegex(ValueError, "Fully hidden"):
            annotations.validate_annotation_document(with_target)

        with_landmark = self.hidden_row()
        with_landmark["frames"][0]["landmarks"] = [{
            "landmark_id": "point-a", "object_point_m": [0.0, 0.0, 0.0],
            "pixel": [5, 5], "visible": True, "reviewed": True,
        }]
        with self.assertRaisesRegex(ValueError, "Fully hidden"):
            annotations.validate_annotation_document(with_landmark)

    def test_reviewed_empty_target_requires_explicit_fully_hidden_choice(self):
        missing_choice = self.hidden_row()
        missing_choice["frames"][0].pop("visibility")
        with self.assertRaisesRegex(ValueError, "explicit visibility"):
            annotations.validate_annotation_document(missing_choice)

        declared_visible = self.hidden_row()
        declared_visible["frames"][0]["visibility"] = "visible"
        with self.assertRaisesRegex(ValueError, "target polygon"):
            annotations.validate_annotation_document(declared_visible)

    def test_visible_review_requires_valid_target_and_reviewed_landmark(self):
        doc = self.document()
        doc["operator"] = "Synthetic reviewer"
        row = doc["frames"][0]
        row.update(status="reviewed", visibility="visible",
                   visible_object=polygon(rectangle(3, 3, 8, 9)))
        with self.assertRaisesRegex(ValueError, "reviewed visible landmarks"):
            annotations.validate_annotation_document(doc)
        row["landmarks"] = [{
            "landmark_id": "point-a", "object_point_m": [0.0, 0.0, 0.0],
            "pixel": [5, 5], "visible": True, "reviewed": True,
        }]
        self.assertIs(annotations.validate_annotation_document(doc), doc)

    def test_malformed_and_degenerate_polygons_are_rejected(self):
        doc = self.document()
        doc["frames"][0]["overlapping_hands"] = polygon([[2, 2], [4, 4], [6, 6]])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            annotations.validate_annotation_document(doc)

        doc["frames"][0]["overlapping_hands"] = polygon([[2, 2], [4, 4], [6, 2], ["bad", 3]])
        with self.assertRaisesRegex(ValueError, "coordinates"):
            annotations.validate_annotation_document(doc)

    def test_import_accepts_reviewed_hidden_row_and_writes_only_to_patched_root(self):
        repository = self.patch_import_root()
        source = self.root / "edited.json"
        source.write_text(json.dumps(self.hidden_row(hands=True)), encoding="utf-8")
        with patch.object(annotations, "ROOT", repository):
            annotations.import_labels(source)
        imported = json.loads((repository / "artifacts/model-quality/annotations/keyboard/annotations.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(imported["frames"][0]["visibility"], "fully_hidden")
        self.assertEqual(imported["frames"][0]["landmarks"], [])
        self.assertEqual(len(imported["frames"][0]["overlapping_hands"]), 1)

    def test_import_rejects_metadata_and_frozen_row_substitution(self):
        cases = {
            "object": lambda d: d.update(object="mug"),
            "resolution": lambda d: d.update(resolution=[31, 24]),
            "source hash": lambda d: d.update(source_sha256="b" * 64),
            "frame ID": lambda d: d["frames"][0].update(frame_id=101),
            "group": lambda d: d["frames"][0].update(group="hard_candidate"),
            "image": lambda d: d["frames"][0].update(image="other.jpg"),
        }
        repository = self.patch_import_root()
        with patch.object(annotations, "ROOT", repository):
            for label, mutate in cases.items():
                with self.subTest(label=label):
                    incoming = self.document()
                    mutate(incoming)
                    source = self.root / f"edited-{label.replace(' ', '-')}.json"
                    source.write_text(json.dumps(incoming), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        annotations.import_labels(source)

    def evaluation_fixture(self):
        mask = np.zeros((FIXTURE_RESOLUTION[1], FIXTURE_RESOLUTION[0]), np.uint8)
        mask[3:6, 3:6] = 255
        mask_path = self.root / "mask.png"
        mask_path.write_bytes(png(mask))
        doc = self.document()
        doc["selection_status"] = annotations.REVIEWED_SELECTION_STATUS
        doc["operator"] = "Synthetic reviewer"
        for index, visibility in ((0, "fully_hidden"), (1, "fully_hidden"), (2, "visible")):
            row = doc["frames"][index]
            row["status"] = "reviewed"
            row["visibility"] = visibility
        doc["frames"][0]["overlapping_hands"] = polygon(rectangle(20, 15, 28, 21))
        doc["frames"][2]["visible_object"] = polygon(rectangle(2, 2, 8, 9))
        doc["frames"][2]["landmarks"] = [{
            "landmark_id": "point-a", "object_point_m": [0.0, 0.0, 0.0],
            "pixel": [16, 12], "visible": True, "reviewed": True,
        }]
        pose = np.eye(4)
        pose[2, 3] = 1.0
        result_frames = []
        for frame_id in FIXTURE_SCORED_IDS:
            result_frames.append({
                "frameId": frame_id,
                "mask_state": "available",
                "mask_path": "mask.png",
                "pose_state": "tracking",
                "cameraFromObject": pose.tolist(),
                "render_state": "suppressed" if frame_id == 101 else "visible",
            })
        results = {"automatic": True, "frames": result_frames}
        intrinsics = np.array([[1.0, 0, 16], [0, 1.0, 12], [0, 0, 1.0]])
        return results, doc, intrinsics, mask_path.parent

    def test_mixed_review_accounting_keeps_hidden_foreground_and_render_leakage_separate(self):
        results, doc, intrinsics, masks_root = self.evaluation_fixture()
        report = evaluator.evaluate(results, doc, masks_root, intrinsics, list(FIXTURE_SCORED_IDS))
        self.assertEqual(report["visible_frames_reviewed"], 1)
        self.assertEqual(report["visible_reviewed_frame_ids"], [102])
        self.assertEqual(report["fully_hidden_frames_reviewed"], 2)
        self.assertEqual(report["fully_hidden_reviewed_frame_ids"], [100, 101])
        self.assertEqual(report["pending_annotation_frames"], 37)
        self.assertEqual(report["pending_annotation_frame_ids"][:2], [103, 104])
        self.assertEqual(report["fully_hidden_false_foreground_frames"], 2)
        self.assertEqual(report["fully_hidden_false_foreground_frame_ids"], [100, 101])
        self.assertEqual(report["fully_hidden_false_foreground_reviewed_denominator"], 2)
        self.assertEqual(report["fully_hidden_render_leakage_frames"], 1)
        self.assertEqual(report["fully_hidden_render_leakage_frame_ids"], [100])
        self.assertEqual(report["fully_hidden_render_leakage_reviewed_denominator"], 2)
        hidden = {row["frame_id"]: row for row in report["details"]}
        self.assertEqual(hidden[100]["mask"]["iou"], 0.0)
        self.assertEqual(hidden[100]["false_foreground_pixels"], 9)
        self.assertEqual(hidden[101]["render_state"], "suppressed")
        self.assertEqual(report["visible_landmark_labels"], 1)
        self.assertEqual(report["visible_landmarks_without_accepted_pose"], 0)
        self.assertFalse(report["independent_accuracy_ready"])
        self.assertIsNone(report["gates"]["attachment"])

    def test_pending_frames_are_reported_but_never_scored(self):
        results, doc, intrinsics, masks_root = self.evaluation_fixture()
        for row in doc["frames"]:
            row["status"] = "pending"
        report = evaluator.evaluate(results, doc, masks_root, intrinsics, list(FIXTURE_SCORED_IDS))
        self.assertEqual(report["annotation_frames_reviewed"], 0)
        self.assertEqual(report["pending_annotation_frames"], 40)
        self.assertEqual(report["details"], [])
        self.assertIsNone(report["mask_mean_iou"])
        self.assertFalse(report["independent_accuracy_ready"])

    def test_oracle_export_keeps_fully_hidden_truth_empty(self):
        _, doc, _, _ = self.evaluation_fixture()
        predicted_root = self.root / "predicted"
        (predicted_root / "masks").mkdir(parents=True)
        mask = np.zeros((FIXTURE_RESOLUTION[1], FIXTURE_RESOLUTION[0]), np.uint8)
        mask[3:6, 3:6] = 255
        (predicted_root / "masks/100.png").write_bytes(png(mask))
        source = {"frames": [{"frameId": 100, "path": "masks/100.png", "mask_state": "available"}]}
        (predicted_root / "results.json").write_text(json.dumps(source), encoding="utf-8")
        output = self.root / "oracle"
        evaluator.export_oracle_masks(predicted_root, doc, output)
        exported = cv2.imread(str(output / "masks/100.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(int(np.count_nonzero(exported)), 0)
        export_index = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(export_index["frames"][0]["mask_state"], "lost")

    def browser_document(self):
        selection = annotations.FROZEN_ANNOTATION_SELECTIONS["keyboard"]
        return {
            "schema_version": 1, "object": "keyboard", "resolution": list(annotations.FROZEN_RESOLUTION),
            "selection_status": annotations.PENDING_SELECTION_STATUS, "operator": None,
            "source_sha256": annotations.FROZEN_SOURCE_SHA256["keyboard"],
            "frames": [{
                "frame_id": frame_id, "image": f"{frame_id}.jpg", "group": group,
                "status": "pending", "visible_object": [], "overlapping_hands": [], "landmarks": [],
                "review_notes": "",
            } for frame_id, group in selection],
        }

    def test_generated_javascript_parses_and_visibility_geometry_handlers_reset_status(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is unavailable for JavaScript syntax and handler validation")
        html = annotations.render_html()
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            script_path = Path(temporary) / "annotations.js"
            fixture_path = Path(temporary) / "canonical.json"
            script_path.write_text(script, encoding="utf-8")
            fixture_path.write_text(json.dumps(self.browser_document()), encoding="utf-8")
            checked = subprocess.run([node, "--check", str(script_path)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

            harness = r"""
const fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(process.argv[1],'utf8');
const fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const elements={};
const drawing={clearRect(){},drawImage(){},beginPath(){},moveTo(){},lineTo(){},closePath(){},fill(){},stroke(){},arc(){},fillText(){}};
function el(id){if(!elements[id])elements[id]={id,value:'',textContent:'',checked:false,files:[],width:1024,height:1280,
 getContext(){return drawing;},getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height};},
 replaceChildren(...children){this.children=children;if(id==='frame'&&children.length)this.value=children[0].value;},click(){}};return elements[id];}
const sandbox={console,document:{getElementById:el,createElement(){return {}}},Image:class{async decode(){}},
 fetch:async()=>({ok:true,json:async()=>fixture}),Blob:class{},URL:{createObjectURL(){return 'blob:'},revokeObjectURL(){}},
 setTimeout(){return 0;}};
el('object').value='keyboard';
vm.createContext(sandbox);vm.runInContext(source,sandbox);
(async()=>{await new Promise(resolve=>setTimeout(resolve,0));await new Promise(resolve=>setTimeout(resolve,0));
 if(vm.runInContext('typeof data',sandbox)==='undefined')throw Error('annotation fixture failed to load: '+el('status').textContent);
 let row=vm.runInContext('current()',sandbox);row.visibility='visible';row.status='reviewed';el('visibility').value='fully_hidden';
 vm.runInContext("$('visibility').onchange()",sandbox);
 row=vm.runInContext('current()',sandbox);if(row.status!=='pending'||row.visibility!=='fully_hidden')throw Error('visibility change did not reset reviewed status');
 row.status='reviewed';row.visibility='fully_hidden';el('tool').value='visible_object';
 vm.runInContext("canvas.onclick({clientX:20,clientY:20})",sandbox);
 row=vm.runInContext('current()',sandbox);if(row.status!=='pending'||vm.runInContext('draft.length',sandbox)!==1)throw Error('geometry change did not reset reviewed status');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
            checked_handlers = subprocess.run(
                [node, "-e", harness, str(script_path), str(fixture_path)],
                capture_output=True, text=True,
            )
            self.assertEqual(checked_handlers.returncode, 0, checked_handlers.stderr)


if __name__ == "__main__":
    unittest.main()
