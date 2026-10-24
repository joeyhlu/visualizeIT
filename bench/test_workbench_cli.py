"""Integration checks for orchestration and source identity, without inference."""
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from bench import workbench_cli as cli
from bench import workbench_contract as wc
from bench import workbench_cache as cache
from bench.test_workbench_contract import clip_document


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        doc = clip_document()
        doc.update(source_kind="image_sequence")
        doc["media"] = {"root_id": "fixture", "path": "images", "sha256": None, "bytes": None}
        for i, frame in enumerate(doc["frames"]):
            content = f"source image identity {i}".encode()
            name = f"f{i}.png"
            (self.images / name).write_bytes(content)
            frame.update(image_path=name, image_sha256=hashlib.sha256(content).hexdigest())
        self.clip = wc.ClipSpec.from_dict(doc)
        self.ws = cli.Workspace(self.root / "workspace")
        self.ws.init({"fixture": str(self.root)})
        self.ws.save_clip(self.clip)

    def tearDown(self):
        self.temp.cleanup()

    def pipeline(self, **kwargs):
        run = cli.build_run_spec(self.clip, mode="smoke", roles=("primary_neural", "frozen_baseline"))
        return cli.Pipeline(self.ws, self.clip, run, cli.RunOptions(), progress=lambda _: None, **kwargs)

    def test_changed_image_rejected_before_cache_can_be_reused(self):
        pipe = self.pipeline()
        reader = cli.FrameReader(self.ws, self.clip)
        masks = cli.MaskStore(self.ws, self.clip.clip_id, None)
        original = pipe._input_identity(pipe.run.scheduled_frame_ids, masks, reader)
        key = cache.stage_key("tracking", original, {}, {"head": "a" * 64})
        doc = self.clip.to_dict()
        content = b"new image"
        (self.images / "f2.png").write_bytes(content)
        with self.assertRaisesRegex(cli.WorkbenchCLIError, "hash differs"):
            pipe._input_identity(pipe.run.scheduled_frame_ids, masks, reader)
        doc["frames"][2]["image_sha256"] = hashlib.sha256(content).hexdigest()
        pipe.clip = self.clip = wc.ClipSpec.from_dict(doc)
        pipe.frames = {f.frame_id: f for f in self.clip.frames}
        reader = cli.FrameReader(self.ws, self.clip)
        changed = pipe._input_identity(pipe.run.scheduled_frame_ids, masks, reader)
        self.assertNotEqual(key, cache.stage_key("tracking", changed, {}, {"head": "a" * 64}))

    def test_budget_block_keeps_every_scheduled_row_without_decoding(self):
        pipe = self.pipeline()
        with patch.object(self.ws, "require_budget", side_effect=cli.WorkbenchCLIError("reserve")), \
                patch.object(cli.FrameReader, "read", side_effect=AssertionError("must not decode")):
            result = pipe.execute()
        self.assertEqual(result["status"], "blocked")
        _, rows, report = cli.load_results(self.ws, pipe.run.run_id)
        self.assertEqual(len(rows), len(pipe.run.scheduled_frame_ids) * 2)
        for role in pipe.run.roles:
            self.assertEqual(report["roles"][role]["missing_rows"], [])
            self.assertTrue(all(r.pose is None for r in rows))

    def test_cancelled_run_keeps_all_frames_without_loading_network(self):
        cancel = threading.Event()
        cancel.set()
        pipe = self.pipeline(cancel=cancel)
        with patch.object(self.ws, "require_budget"), patch.object(cli, "stage_cache", return_value=(None, "test")), \
                patch("bench.workbench_neural.load_primary_network", side_effect=AssertionError("no neural work")):
            result = pipe.execute()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(pipe.results), len(pipe.run.scheduled_frame_ids) * 2)

    def test_preparation_exception_still_accounts_for_every_requested_role(self):
        run = cli.build_run_spec(self.clip, mode="smoke",
                                 roles=("relative_geometry", "primary_neural", "frozen_baseline"))
        pipe = cli.Pipeline(self.ws, self.clip, run, cli.RunOptions(), progress=lambda _: None)
        with patch.object(cli, "stage_cache", return_value=(None, "fixture")), \
             patch.object(pipe, "_relative", side_effect=ValueError("prep\n\tfailed " * 2000)), \
             patch.object(pipe, "_neural", side_effect=RuntimeError("neural fixture failed")) as neural, \
             patch.object(cli.FrameReader, "read", side_effect=AssertionError("must not decode")):
            result = pipe.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(neural.call_count, 2)
        _, rows, report = cli.load_results(self.ws, run.run_id)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({(row.backend_role, row.frame_id) for row in rows}), 6)
        self.assertTrue(all(row.pose is None and row.pose_state == "unavailable" for row in rows))
        self.assertTrue(all(not role["missing_rows"] for role in report["roles"].values()))
        self.assertEqual(len(list((pipe.run_dir / "errors").glob("*.json"))), 3)

    def test_partial_results_survive_exception_without_duplicate_rows(self):
        pipe = self.pipeline()

        def interrupted(role, *_args):
            first = pipe.frames[pipe.run.scheduled_frame_ids[0]]
            row = wc.blocked_result(pipe.run.run_id, self.clip, first, role, {"role": role},
                                    "fixture already produced", pose_state="unavailable", mask_state="unavailable")
            pipe.results.append(row)
            if role == "primary_neural":
                pipe.results.append(row)
            raise RuntimeError("failure after partial work")

        with patch.object(cli, "stage_cache", return_value=(None, "fixture")), \
             patch.object(pipe, "_neural", side_effect=interrupted):
            result = pipe.execute()
        self.assertEqual(result["status"], "failed")
        _, rows, report = cli.load_results(self.ws, pipe.run.run_id)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({(row.backend_role, row.frame_id) for row in rows}), 4)
        self.assertEqual(sum(row.failure_reason == "fixture already produced" for row in rows), 2)
        self.assertTrue(all(not role["missing_rows"] for role in report["roles"].values()))
        self.assertEqual(report["stages"]["accounting"]["dropped_duplicate_or_unscheduled_rows"], 1)


if __name__ == "__main__":
    unittest.main()
