"""Causal clock registration is checked without reading image content."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench import workbench_contract as contracts
from bench.test_workbench_contract import clip_document
from tools import workbench_timed_prefix as prefix


class TimedPrefixTests(unittest.TestCase):
    def test_actual_time_bound_and_unique_keyframes(self):
        times = [i / 60 for i in range(900)]
        chosen = prefix.select_keyframes(times, 826, 827, seconds=10., keyframes=24)
        self.assertEqual(len(chosen), 24)
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertEqual(chosen[-1], 826)
        self.assertAlmostEqual(times[chosen[-1]] - times[chosen[0]], 10.)
        self.assertTrue(all(i < 827 for i in chosen))

    def test_later_timestamps_cannot_change_frozen_prefix(self):
        times = [i / 60 for i in range(900)]
        original = prefix.select_keyframes(times, 826, 827, seconds=10., keyframes=24)
        times[827:] = [1000. + i for i in range(len(times) - 827)]
        self.assertEqual(original, prefix.select_keyframes(times, 826, 827, seconds=10., keyframes=24))

    def test_limits_and_scoring_boundary_cannot_be_relaxed(self):
        for seconds, keyframes, seed in ((11., 24, 5), (10., 25, 5), (0., 24, 5), (10., 24, 6)):
            with self.subTest(seconds=seconds, keyframes=keyframes, seed=seed):
                with self.assertRaises(prefix.TimedPrefixError):
                    prefix.select_keyframes(list(range(10)), seed, 6, seconds=seconds, keyframes=keyframes)

    def fixture(self):
        document = clip_document(clock_kind="frame_index", start_index=0, media_times=False)
        document["onboarding"]["selection"]["frame_id"] = "f1"
        document["provenance"].update(metric_cad={"path": "forbidden-geometry"},
                                           initialization_candidate={"pose": "forbidden-pose"})
        source = contracts.ClipSpec.from_dict(document)
        workspace = SimpleNamespace(load_clip=lambda _: source, media_path=lambda _: Path("not-opened.mp4"))
        reader = Mock()
        reader.verify_media.return_value = {"kind": "video_file", "bytes": 4096, "sha256": "a" * 64}
        probe = {"media_times": [0., .1, .2, .3], "pts_reliable": True,
                 "width": 640, "height": 480, "frame_count": 4}
        return source, workspace, reader, probe

    def test_registration_preserves_scoring_and_omits_geometry_inputs(self):
        source, workspace, reader, probe = self.fixture()
        original = copy.deepcopy(source.to_dict())
        with patch.object(prefix.cli, "FrameReader", return_value=reader), \
             patch.object(prefix.cli, "probe_video", return_value=probe):
            built = prefix.build(workspace, source.clip_id, "new-experiment", seconds=10., keyframes=24)
        clip = built["clip"]
        self.assertEqual(source.to_dict(), original)
        self.assertEqual(clip.scoring_frame_ids, source.scoring_frame_ids)
        self.assertEqual([clip.frame(fid).source_frame_index for fid in clip.scoring_frame_ids], [2, 3])
        self.assertEqual(clip.onboarding.selection, source.onboarding.selection)
        self.assertNotIn("metric_cad", clip.provenance)
        self.assertNotIn("initialization_candidate", clip.provenance)
        self.assertEqual([f.timestamp_s for f in clip.frames], probe["media_times"])
        self.assertTrue(built["receipt"]["prefix_attestation"]["prefix_bound_attested"])
        reader.read.assert_not_called()

    def test_unreliable_decoder_clock_never_becomes_guessed_time(self):
        source, workspace, reader, probe = self.fixture()
        probe["pts_reliable"] = False
        with patch.object(prefix.cli, "FrameReader", return_value=reader), \
             patch.object(prefix.cli, "probe_video", return_value=probe):
            with self.assertRaisesRegex(prefix.TimedPrefixError, "refusing to guess"):
                prefix.build(workspace, source.clip_id, "new-experiment", seconds=10., keyframes=24)


if __name__ == "__main__":
    unittest.main()
