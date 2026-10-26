"""Offline tests for real benchmark acquisition, selection and label isolation."""
import binascii
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from . import tracking_real_data as data


def archive_fixture():
    output = io.BytesIO()
    contents = {"rgb/a.png": b"real-source-fixture" * 32, "depth/a.png": bytes(range(200)),
                "mask/a.png": b"mask" * 500, "sentinel": b"end"}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in contents.items():
            archive.writestr(name, raw)
    payload = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = {row.filename: dict(offset=row.header_offset, method=row.compress_type,
                   flags=row.flag_bits, compressed=row.compress_size, size=row.file_size,
                   crc=row.CRC) for row in archive.infolist()}
    return payload, contents, entries


class BoundedRangeTests(unittest.TestCase):
    def test_coalesced_records_roundtrip_and_tampering_fails(self):
        payload, contents, entries = archive_fixture()
        spans = data.ranges_for_members(entries, list(contents)[:-1], archive_size=len(payload))
        self.assertEqual(len(spans), 1)
        start, stop, names = spans[0]
        for name in names:
            actual = data.extract_record(payload[start:stop], entries[name]["offset"]-start, name, entries[name])
            self.assertEqual(actual, contents[name])
        broken = dict(entries["rgb/a.png"], crc=0)
        with self.assertRaises(ValueError):
            data.extract_record(payload, entries["rgb/a.png"]["offset"], "rgb/a.png", broken)

    def test_final_member_range_is_clamped_to_archive(self):
        payload, _, entries = archive_fixture()
        spans = data.ranges_for_members(entries, ["sentinel"], archive_size=len(payload))
        self.assertLessEqual(spans[0][1], len(payload))

    def test_range_limits_and_unsafe_names_are_rejected(self):
        _, _, entries = archive_fixture()
        with self.assertRaises(ValueError):
            data.ranges_for_members(entries, ["rgb/a.png"], range_limit=3)
        for name in ("../escape", "/absolute", "a//b", "a\\b", "a/../b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                data.safe_member(name)

    def test_frozen_outputs_are_idempotent_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"manifest.json"
            data.atomic_write(path, b"a")
            data.atomic_write(path, b"a")
            with self.assertRaises(ValueError):
                data.atomic_write(path, b"b")

    def test_release_archive_is_reproducible_and_contains_license_readme(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root/"source"
            source.mkdir()
            (source/"rgb.png").write_bytes(b"public-fixture")
            lock = dict(resources=[dict(member="rgb.png", bytes=14, sha256=data.sha(b"public-fixture"))])
            for name, key in (("selection.json", "selectionSHA256"), ("inference.json", "inferenceSHA256"), ("evaluator.json", "evaluatorSHA256")):
                raw = data.canonical({"fixture": name})
                (source/name).write_bytes(raw)
                lock[key] = data.sha(raw)
            (source/"sources.lock.json").write_bytes(data.canonical(lock))
            first = data.package(source, root/"first.tar.gz")
            second = data.package(source, root/"second.tar.gz")
            self.assertEqual(first["sha256"], second["sha256"])
            with tarfile.open(root/"first.tar.gz") as archive:
                self.assertEqual(archive.extractfile("real-v1/rgb.png").read(), b"public-fixture")
                self.assertIn(b"CC0 1.0", archive.extractfile("real-v1/README.md").read())


class SelectionAndManifestTests(unittest.TestCase):
    def metadata(self):
        result = {}
        for scene_index, scene in enumerate(data.SCENES):
            ids = list(range(scene_index*3+1, scene_index*3+4))
            result[scene] = {"scene_gt.json": {}, "scene_camera.json": {}, "scene_gt_info.json": {}}
            for frame in range(data.FRAME_COUNT):
                result[scene]["scene_gt.json"][str(frame)] = [dict(obj_id=obj, cam_R_m2c=[1,0,0,0,1,0,0,0,1], cam_t_m2c=[0,0,500]) for obj in ids]
                result[scene]["scene_camera.json"][str(frame)] = dict(cam_K=[500,0,320,0,500,240,0,0,1], depth_scale=1)
                result[scene]["scene_gt_info.json"][str(frame)] = [dict(visib_fract=.25) for _ in ids]
        return result

    def test_selection_retains_low_visibility_and_all_frames(self):
        metadata = self.metadata()
        selected = data.selection_for(metadata)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["objectId"] for row in selected}), 12)
        self.assertTrue(all(row["frameIds"] == list(range(310)) for row in selected))

    def test_missing_or_duplicate_instances_fail_fixed_policy(self):
        metadata = self.metadata()
        metadata[data.SCENES[0]]["scene_gt.json"]["2"].pop()
        with self.assertRaises(ValueError):
            data.selection_for(metadata)

    def test_inference_never_receives_scored_pose_depth_or_mask(self):
        metadata = self.metadata()
        selected = data.selection_for(metadata)
        class Hashes(dict):
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                return "a"*64
        inference, evaluator = data.manifests(selected, metadata, Hashes(), "b"*64)
        self.assertEqual(inference["evaluatorManifestSHA256"], data.sha(data.canonical(evaluator)))
        for obj, reference in zip(inference["objects"], evaluator["objects"]):
            self.assertEqual(len(obj["onboarding"]), 10)
            self.assertEqual(len(obj["frames"]), 300)
            self.assertIn("onboardingMask", obj["onboarding"][0])
            self.assertEqual([r["frameId"] for r in obj["frames"]], [r["frameId"] for r in reference["frames"]])
            for record in obj["frames"]:
                self.assertTrue({"evaluationPose", "depth", "evaluationMask", "evaluationInfo", "onboardingMask"}.isdisjoint(record))
            self.assertTrue(all("evaluationPose" in r for r in reference["frames"]))


if __name__ == "__main__":
    unittest.main()
