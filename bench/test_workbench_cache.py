from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from collections import namedtuple
from pathlib import Path
from unittest import mock

from bench import workbench_cache as wc
from bench.workbench_cache import (
    CacheBudgetError,
    CacheCollisionError,
    CacheIntegrityError,
    CacheLockTimeout,
    CacheValueError,
    StageCache,
    canonical_json,
    hash_file,
    stage_key,
)

DiskUsage = namedtuple("DiskUsage", "total used free")
HUGE = 1 << 50


def _key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _snapshot(root: Path, *, ignore_scratch_dir_times: bool = False) -> dict[str, tuple[int, int]]:
    result = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        rel = path.relative_to(root).as_posix()
        if ignore_scratch_dir_times and rel == "tmp":
            # Writers release the lock through tmp/, touching only its directory time.
            result[rel] = (0, 0)
            continue
        result[rel] = (st.st_size, st.st_mtime_ns)
    return result


def _symlink_or_skip(test: unittest.TestCase, link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        if directory and os.name == "nt":
            import _winapi  # Junctions need no privilege and are reparse points too.

            try:
                _winapi.CreateJunction(str(target), str(link))
                return
            except OSError:
                pass
        test.skipTest(f"symlinks unavailable: {exc}")


def _rewrite_record(cache: StageCache, key: str, mutate) -> None:
    path = cache.root / "entries" / key / "record.json"
    record = json.loads(path.read_bytes())
    mutate(record)
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    path.write_bytes(canonical_json(record))


class _TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "cache"
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        patcher = mock.patch.object(wc.shutil, "disk_usage", return_value=DiskUsage(HUGE, 0, HUGE))
        self.disk = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def cache(self, **kwargs) -> StageCache:
        kwargs.setdefault("max_bytes", 1 << 20)
        kwargs.setdefault("reserve_bytes", 0)
        kwargs.setdefault("lock_timeout", 5.0)
        return StageCache(self.root, **kwargs)

    def assert_no_staging(self) -> None:
        temp = self.root / "tmp"
        self.assertEqual(list(temp.iterdir()) if temp.exists() else [], [])
        self.assertFalse((self.root / "writer.lock").exists())


class HashAndKeyTests(_TempCase):
    def test_hash_file_matches_sha256_and_tracks_mutation(self) -> None:
        path = self.inputs / "weights.bin"
        path.write_bytes(b"checkpoint-a" * 100_000)
        digest = hash_file(path)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(digest, digest.lower())
        path.write_bytes(b"checkpoint-b" * 100_000)
        self.assertNotEqual(hash_file(path), digest)

    def test_hash_file_rejects_directories_traversal_and_symlinks(self) -> None:
        with self.assertRaises(CacheValueError):
            hash_file(self.inputs)
        with self.assertRaises(CacheValueError):
            hash_file(self.inputs / ".." / "inputs" / "x")
        target = self.inputs / "real.bin"
        target.write_bytes(b"x")
        link = self.inputs / "link.bin"
        _symlink_or_skip(self, link, target)
        with self.assertRaises(CacheValueError):
            hash_file(link)

    def test_stage_key_is_canonical_and_invalidated_by_checkpoint_config_version(self) -> None:
        base = dict(
            stage="segment",
            inputs={"video": "a" * 64, "checkpoint": "c" * 64},
            configuration={"threshold": 0.5, "size": [640, 480]},
            versions={"code": "1.2.0", "model": "sam2"},
        )
        key = stage_key(**base)
        self.assertRegex(key, r"^[0-9a-f]{64}$")
        reordered = dict(
            stage="segment",
            inputs={"checkpoint": "c" * 64, "video": "a" * 64},
            configuration={"size": [640, 480], "threshold": 0.5},
            versions={"model": "sam2", "code": "1.2.0"},
        )
        self.assertEqual(stage_key(**reordered), key)
        variants = [
            {**base, "stage": "pose"},
            {**base, "inputs": {**base["inputs"], "checkpoint": "d" * 64}},
            {**base, "configuration": {**base["configuration"], "threshold": 0.6}},
            {**base, "configuration": {**base["configuration"], "threshold": 1}},
            {**base, "versions": {**base["versions"], "code": "1.2.1"}},
        ]
        keys = {stage_key(**variant) for variant in variants}
        self.assertEqual(len(keys), len(variants))
        self.assertNotIn(key, keys)

    def test_stage_key_rejects_nonfinite_unsupported_and_ambiguous_values(self) -> None:
        good = dict(stage="s", inputs={}, configuration={}, versions={})
        bad_configs = [
            {"x": float("nan")},
            {"x": float("inf")},
            {"x": (1, 2)},
            {"x": {1, 2}},
            {"x": Path("a")},
            {"x": b"bytes"},
            {1: "int key"},
            {"x": {"nested": float("-inf")}},
            {"x": "\ud800"},
        ]
        for config in bad_configs:
            with self.subTest(config=repr(config)), self.assertRaises(CacheValueError):
                stage_key(**{**good, "configuration": config})
        for stage in ("", " s", None, 3):
            with self.subTest(stage=stage), self.assertRaises(CacheValueError):
                stage_key(**{**good, "stage": stage})
        with self.assertRaises(CacheValueError):
            stage_key(**{**good, "inputs": [("a", 1)]})
        cyclic: list = []
        cyclic.append(cyclic)
        with self.assertRaises(CacheValueError):
            stage_key(**{**good, "configuration": {"c": cyclic}})


class StoreLookupTests(_TempCase):
    def test_round_trip_with_bytes_and_external_path_sources(self) -> None:
        source = self.inputs / "mask.png"
        source.write_bytes(b"\x89PNG" + bytes(range(256)) * 50)
        cache = self.cache()
        key = _key("roundtrip")
        record = cache.store(key, {"frames": 3, "score": 0.25}, files={"masks/0001.png": source, "log.txt": b"ok\n"})
        self.assertEqual(record["key"], key)
        self.assertEqual(record["payload"], {"frames": 3, "score": 0.25})
        self.assertEqual(record["payload_sha256"], hashlib.sha256(canonical_json(record["payload"])).hexdigest())
        self.assertEqual(set(record["files"]), {"masks/0001.png", "log.txt"})
        for name, meta in record["files"].items():
            stored = cache.root / meta["path"]
            self.assertFalse(Path(meta["path"]).is_absolute())
            self.assertTrue(meta["path"].startswith(f"entries/{key}/files/"))
            self.assertEqual(hashlib.sha256(stored.read_bytes()).hexdigest(), meta["sha256"])
            self.assertEqual(stored.stat().st_size, meta["size"])
        self.assertEqual(record["files"]["masks/0001.png"]["sha256"], hash_file(source))
        self.assertTrue(source.exists())
        self.assertEqual(cache.lookup(key), record)
        self.assert_no_staging()

    def test_lookup_missing_never_writes(self) -> None:
        cache = self.cache()
        self.assertIsNone(cache.lookup(_key("absent")))
        self.assertFalse(self.root.exists())
        cache.store(_key("present"), {"a": 1})
        before = _snapshot(self.root)
        self.assertIsNone(cache.lookup(_key("absent")))
        self.assertIsNotNone(cache.lookup(_key("present")))
        self.assertEqual(_snapshot(self.root), before)

    def test_invalid_keys_and_payloads_are_rejected_without_writes(self) -> None:
        cache = self.cache()
        for key in ("A" * 64, "a" * 63, "a" * 65, "g" * 64, 123, None):
            with self.subTest(key=key), self.assertRaises(CacheValueError):
                cache.store(key, {})
            with self.subTest(key=key), self.assertRaises(CacheValueError):
                cache.lookup(key)
        for payload in ([], {"x": float("nan")}, {"x": (1,)}, {2: "x"}, {"x": object()}):
            with self.subTest(payload=repr(payload)), self.assertRaises(CacheValueError):
                cache.store(_key("p"), payload)
        self.assertFalse(self.root.exists())

    def test_unsafe_file_names_and_sources_are_rejected(self) -> None:
        cache = self.cache()
        bad_names = [
            "/abs.bin", "../escape.bin", "a/../../b", "a\\b", "C:evil", "c:/x", "a//b", "./a", "a/./b",
            "", "CON", "nul.txt", "trailing.", "sp ace", "x:stream", "a/" + "b" * 200,
        ]
        for name in bad_names:
            with self.subTest(name=name), self.assertRaises(CacheValueError):
                cache.store(_key("names"), {}, files={name: b"x"})
        for files in ({"A.txt": b"1", "a.txt": b"2"}, {"a": b"1", "a/b": b"2"}, {"D/x": b"1", "d": b"2"}):
            with self.subTest(files=list(files)), self.assertRaises(CacheValueError):
                cache.store(_key("names"), {}, files=files)
        with self.assertRaises(CacheValueError):
            cache.store(_key("names"), {}, files={"x": str(self.inputs / "f")})
        with self.assertRaises(CacheValueError):
            cache.store(_key("names"), {}, files={"x": self.inputs})
        with self.assertRaises(CacheValueError):
            cache.store(_key("names"), {}, files=[("x", b"1")])
        self.assertIsNone(cache.lookup(_key("names")))

    def test_identical_store_returns_existing_without_rewrite(self) -> None:
        cache = self.cache()
        key = _key("same")
        first = cache.store(key, {"v": 1}, files={"a.bin": b"abc"})
        before = _snapshot(self.root, ignore_scratch_dir_times=True)
        second = cache.store(key, {"v": 1}, files={"a.bin": b"abc"})
        self.assertEqual(first, second)
        self.assertEqual(_snapshot(self.root, ignore_scratch_dir_times=True), before)

    def test_collisions_never_overwrite(self) -> None:
        cache = self.cache()
        key = _key("collide")
        cache.store(key, {"v": 1}, files={"a.bin": b"abc"})
        before = _snapshot(self.root, ignore_scratch_dir_times=True)
        attempts = [
            ({"v": 2}, {"a.bin": b"abc"}),
            ({"v": 1.0}, {"a.bin": b"abc"}),
            ({"v": 1}, {"a.bin": b"abd"}),
            ({"v": 1}, {"b.bin": b"abc"}),
            ({"v": 1}, None),
        ]
        for payload, files in attempts:
            with self.subTest(payload=payload, files=files), self.assertRaises(CacheCollisionError):
                cache.store(key, payload, files=files)
        self.assertEqual(_snapshot(self.root, ignore_scratch_dir_times=True), before)
        self.assertEqual(cache.lookup(key)["payload"], {"v": 1})


class IntegrityTests(_TempCase):
    def setUp(self) -> None:
        super().setUp()
        self.c = self.cache()
        self.key = _key("integrity")
        self.c.store(self.key, {"v": 1}, files={"dir/a.bin": b"hello world", "b.bin": b"bee"})
        self.entry = self.root / "entries" / self.key

    def assert_corrupt(self) -> None:
        with self.assertRaises(CacheIntegrityError):
            self.c.lookup(self.key)
        with self.assertRaises(CacheIntegrityError):
            self.c.store(self.key, {"v": 1}, files={"dir/a.bin": b"hello world", "b.bin": b"bee"})
        self.assertTrue(self.entry.exists())
        self.assertFalse((self.root / "writer.lock").exists())

    def test_same_length_byte_mutation(self) -> None:
        (self.entry / "files" / "dir" / "a.bin").write_bytes(b"hello WORLD")
        self.assert_corrupt()

    def test_truncation_and_growth(self) -> None:
        target = self.entry / "files" / "b.bin"
        target.write_bytes(b"be")
        self.assert_corrupt()
        target.write_bytes(b"beee")
        self.assert_corrupt()

    def test_missing_extra_and_symlinked_files(self) -> None:
        (self.entry / "files" / "extra.bin").write_bytes(b"")
        self.assert_corrupt()
        (self.entry / "files" / "extra.bin").unlink()
        (self.entry / "files" / "emptydir").mkdir()
        self.assert_corrupt()
        (self.entry / "files" / "emptydir").rmdir()
        self.assertIsNotNone(self.c.lookup(self.key))
        target = self.entry / "files" / "b.bin"
        outside = self.inputs / "b.bin"
        shutil.move(str(target), str(outside))
        self.assert_corrupt()

    def test_symlinked_file_inside_entry(self) -> None:
        target = self.entry / "files" / "b.bin"
        outside = self.inputs / "b.bin"
        shutil.move(str(target), str(outside))
        _symlink_or_skip(self, target, outside)
        self.assert_corrupt()

    def test_dangling_partial_entry(self) -> None:
        (self.entry / "record.json").unlink()
        self.assert_corrupt()
        partial = self.root / "entries" / _key("partial")
        partial.mkdir()
        with self.assertRaises(CacheIntegrityError):
            self.c.lookup(_key("partial"))

    def test_tampered_payload_and_record(self) -> None:
        record_path = self.entry / "record.json"
        original = record_path.read_bytes()
        record_path.write_bytes(original.replace(b'"v":1', b'"v":2'))
        self.assert_corrupt()
        record_path.write_bytes(original + b"\n")
        self.assert_corrupt()
        record_path.write_bytes(original.replace(b'{"files"', b'{"format":"x","files"', 1))
        self.assert_corrupt()
        record_path.write_bytes(original.replace(b'"payload":{"v":1}', b'"payload":{"v":NaN}'))
        self.assert_corrupt()
        record_path.write_bytes(original.replace(b'"payload":{"v":1}', b'"payload":{"v":1e999}'))
        self.assert_corrupt()
        record_path.write_bytes(b"\xff" + original)
        self.assert_corrupt()
        record_path.write_bytes(original)
        self.assertIsNotNone(self.c.lookup(self.key))

    def test_resealed_record_with_untrusted_size_path_or_format(self) -> None:
        mutations = [
            lambda r: r["files"]["b.bin"].update(size=2),
            lambda r: r["files"]["b.bin"].update(sha256=hashlib.sha256(b"x").hexdigest()),
            lambda r: r["files"]["b.bin"].update(path="../../../inputs/b.bin"),
            lambda r: r["files"].update({"../x": r["files"].pop("b.bin")}),
            lambda r: r.update(format="visualizeit.workbench_cache.v0"),
            lambda r: r.update(key=_key("other")),
            lambda r: r.update(extra=True),
        ]
        original = (self.entry / "record.json").read_bytes()
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                _rewrite_record(self.c, self.key, mutate)
                with self.assertRaises(CacheIntegrityError):
                    self.c.lookup(self.key)
                (self.entry / "record.json").write_bytes(original)
        self.assertIsNotNone(self.c.lookup(self.key))

    def test_symlinked_entry_and_root_are_rejected(self) -> None:
        moved = self.base / "moved-entry"
        shutil.move(str(self.entry), str(moved))
        _symlink_or_skip(self, self.entry, moved, directory=True)
        with self.assertRaises(CacheIntegrityError):
            self.c.lookup(self.key)
        root_link = self.base / "root-link"
        _symlink_or_skip(self, root_link, self.root, directory=True)
        with self.assertRaises(CacheValueError):
            StageCache(root_link)
        with self.assertRaises(CacheValueError):
            StageCache(root_link / "nested")


class SourceSafetyTests(_TempCase):
    def test_symlinked_source_chain_is_rejected(self) -> None:
        real_dir = self.inputs / "real"
        real_dir.mkdir()
        (real_dir / "f.bin").write_bytes(b"data")
        link_dir = self.inputs / "linked"
        _symlink_or_skip(self, link_dir, real_dir, directory=True)
        cache = self.cache()
        with self.assertRaises(CacheValueError):
            cache.store(_key("src"), {}, files={"f.bin": link_dir / "f.bin"})
        self.assertIsNone(cache.lookup(_key("src")))

    def test_source_changed_during_copy_is_detected(self) -> None:
        source = self.inputs / "grow.bin"
        source.write_bytes(b"a" * 10)
        cache = self.cache()
        real_stage = StageCache._stage

        def mutate_then_stage(this, *args):
            source.write_bytes(b"b" * 10)
            return real_stage(this, *args)

        with mock.patch.object(StageCache, "_stage", mutate_then_stage):
            with self.assertRaises(wc.CacheSourceChangedError):
                cache.store(_key("grow"), {}, files={"g.bin": source})
        self.assertIsNone(cache.lookup(_key("grow")))
        self.assert_no_staging()


class BudgetTests(_TempCase):
    def _entry_bytes(self, key: str, payload: dict, files: dict) -> int:
        probe = StageCache(self.base / f"probe-{key[:8]}", max_bytes=HUGE, reserve_bytes=0)
        probe.store(key, payload, files=files)
        return probe.usage_bytes()

    def test_lock_adds_no_bytes(self) -> None:
        self.assertEqual(wc.LOCK_BYTES, 0)
        cache = self.cache()
        cache._prepare_root()
        with cache._writer_lock():
            self.assertTrue((self.root / "writer.lock").is_dir())
            self.assertEqual(cache.usage_bytes(), 0)
        self.assert_no_staging()

    def test_cap_boundary_is_exact_including_record_bytes(self) -> None:
        key, payload, files = _key("cap"), {"n": 1}, {"a.bin": b"x" * 5000}
        entry_bytes = self._entry_bytes(key, payload, files)
        self.assertGreater(entry_bytes, 5000)
        tight = self.cache(max_bytes=entry_bytes - 1)
        with self.assertRaises(CacheBudgetError):
            tight.store(key, payload, files=files)
        self.assertIsNone(tight.lookup(key))
        self.assert_no_staging()
        exact = self.cache(max_bytes=entry_bytes)
        exact.store(key, payload, files=files)
        self.assertEqual(exact.usage_bytes(), entry_bytes)

    def test_existing_entries_count_toward_cap(self) -> None:
        files = {"a.bin": b"y" * 4000}
        one = self._entry_bytes(_key("first"), {}, files)
        cache = self.cache(max_bytes=2 * one - 1)
        cache.store(_key("first"), {}, files=files)
        with self.assertRaises(CacheBudgetError):
            cache.store(_key("secnd"), {}, files=files)
        self.assertIsNotNone(cache.lookup(_key("first")))
        self.assert_no_staging()

    def test_stray_temporary_bytes_count_toward_cap(self) -> None:
        cache = self.cache(max_bytes=100_000)
        cache.store(_key("seed"), {})
        (self.root / "tmp" / "foreign.partial").write_bytes(b"z" * 95_000)
        with self.assertRaises(CacheBudgetError):
            cache.store(_key("next"), {}, files={"a.bin": b"q" * 6000})
        self.assertTrue((self.root / "tmp" / "foreign.partial").exists())

    def test_reserve_violation_is_detected_before_lock_is_created(self) -> None:
        cache = self.cache(reserve_bytes=1000)
        cache._prepare_root()
        self.disk.return_value = DiskUsage(HUGE, 0, 999)
        with mock.patch.object(wc.os, "mkdir", wraps=os.mkdir) as spy:
            with self.assertRaises(CacheBudgetError):
                cache.store(_key("reserve"), {})
        self.assertNotIn(str(self.root / "writer.lock"), [str(call.args[0]) for call in spy.call_args_list])
        self.assert_no_staging()

    def test_reserve_checked_before_write(self) -> None:
        cache = self.cache(reserve_bytes=1000)
        self.disk.return_value = DiskUsage(HUGE, 0, 1500)
        with self.assertRaises(CacheBudgetError):
            cache.store(_key("reserve"), {}, files={"a.bin": b"r" * 600})
        self.assertIsNone(cache.lookup(_key("reserve")))
        self.assert_no_staging()
        self.disk.return_value = DiskUsage(HUGE, 0, 1000 + 600 + 4096)
        self.assertIsNotNone(cache.store(_key("reserve"), {}, files={"a.bin": b"r" * 600}))

    def test_reserve_checked_during_chunked_write(self) -> None:
        cache = self.cache(max_bytes=HUGE, reserve_bytes=1 << 20)
        calls = {"n": 0}

        def shrinking(_path):
            calls["n"] += 1  # 1: before lock, 2: budget, 3: first chunk, 4: second chunk
            free = HUGE if calls["n"] <= 3 else (1 << 20)
            return DiskUsage(HUGE, 0, free)

        self.disk.side_effect = shrinking
        data = b"c" * (wc.CHUNK_BYTES * 3)
        with self.assertRaises(CacheBudgetError):
            cache.store(_key("during"), {}, files={"big.bin": data})
        self.assertEqual(calls["n"], 4)
        self.assertIsNone(cache.lookup(_key("during")))
        self.assert_no_staging()

    def test_concurrent_writers_never_exceed_strict_cap(self) -> None:
        files = {"a.bin": b"k" * 200_000}
        entry = self._entry_bytes(_key("w0"), {"n": 0}, files)
        cap = 3 * entry + entry // 2
        cache = self.cache(max_bytes=cap, lock_timeout=30.0)
        peaks: list[int] = []
        real_check = StageCache._check_reserve

        def observing(this, remaining):
            peaks.append(this.usage_bytes() + remaining)
            return real_check(this, remaining)

        barrier = threading.Barrier(6)
        results: list[object] = [None] * 6

        def worker(index: int) -> None:
            barrier.wait()
            try:
                results[index] = cache.store(_key(f"w{index}"), {"n": index}, files=files)
            except Exception as exc:  # noqa: BLE001 - recorded for assertions
                results[index] = exc

        with mock.patch.object(StageCache, "_check_reserve", observing), warnings.catch_warnings():
            warnings.simplefilter("error")
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)
        stored = [r for r in results if isinstance(r, dict)]
        refused = [r for r in results if isinstance(r, CacheBudgetError)]
        self.assertEqual((len(stored), len(refused)), (3, 3), results)
        self.assertTrue(peaks)
        self.assertLessEqual(max(peaks), cap)
        self.assertLessEqual(cache.usage_bytes(), cap)
        self.assert_no_staging()


_PROCESS_SCRIPT = r"""
import hashlib, json, sys, warnings
warnings.simplefilter("error")
from bench.workbench_cache import StageCache
root, worker = sys.argv[1], int(sys.argv[2])
cache = StageCache(root, max_bytes=1 << 30, reserve_bytes=0, lock_timeout=60)
digests = []
for i in range(8):
    shared = hashlib.sha256(f"shared{i}".encode()).hexdigest()
    own = hashlib.sha256(f"own{worker}-{i}".encode()).hexdigest()
    first = cache.store(shared, {"i": i}, files={"a.bin": bytes([i]) * 50000})
    second = cache.store(own, {"w": worker, "i": i}, files={"a.bin": bytes([worker]) * 20000})
    assert cache.lookup(shared) == first and cache.lookup(own) == second
    digests.append(first["record_sha256"])
print(json.dumps(digests))
"""


class ConcurrencyTests(_TempCase):
    def _race(self, payloads: list[dict], root: Path | None = None, keys: list[str] | None = None) -> list[object]:
        cache = StageCache(root or self.root, max_bytes=1 << 24, reserve_bytes=0, lock_timeout=30.0)
        keys = keys or [_key("race")] * len(payloads)
        barrier = threading.Barrier(len(payloads))
        results: list[object] = [None] * len(payloads)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                results[index] = cache.store(keys[index], payloads[index], files={"a.bin": b"w" * 300_000})
            except Exception as exc:  # noqa: BLE001 - recorded for assertions
                results[index] = exc

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(payloads))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)
        return results

    def test_two_identical_writers_share_one_entry(self) -> None:
        first, second = self._race([{"v": 1}, {"v": 1}])
        self.assertIsInstance(first, dict)
        self.assertEqual(first, second)
        self.assert_no_staging()

    def test_two_conflicting_writers_one_wins(self) -> None:
        results = self._race([{"v": 1}, {"v": 2}])
        winners = [r for r in results if isinstance(r, dict)]
        losers = [r for r in results if isinstance(r, CacheCollisionError)]
        self.assertEqual((len(winners), len(losers)), (1, 1), results)
        self.assertEqual(self.cache().lookup(_key("race")), winners[0])
        self.assert_no_staging()

    def test_repeated_mixed_thread_races_with_warnings_as_errors(self) -> None:
        for round_index in range(15):
            root = self.base / f"round-{round_index}"
            same, clash = _key(f"same{round_index}"), _key(f"clash{round_index}")
            keys = [same, same, clash, clash, _key(f"a{round_index}"), _key(f"b{round_index}")]
            payloads = [{"v": 1}, {"v": 1}, {"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]
            results = self._race(payloads, root=root, keys=keys)
            collisions = [r for r in results if isinstance(r, CacheCollisionError)]
            others = [r for r in results if not isinstance(r, (dict, CacheCollisionError))]
            self.assertEqual(others, [], f"round {round_index}")
            self.assertEqual(len(collisions), 1, f"round {round_index}: {results}")
            self.assertEqual(results[0], results[1])
            self.assertEqual(os.listdir(root / "tmp"), [], f"round {round_index}")
            self.assertFalse((root / "writer.lock").exists())

    def test_concurrent_processes_with_warnings_as_errors(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        procs = [
            subprocess.Popen(
                [sys.executable, "-B", "-W", "error", "-c", _PROCESS_SCRIPT, str(self.root), str(worker)],
                cwd=repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for worker in range(4)
        ]
        outputs = []
        for proc in procs:
            out, err = proc.communicate(timeout=180)
            self.assertEqual((proc.returncode, err), (0, ""), err)
            outputs.append(json.loads(out))
        self.assertTrue(all(output == outputs[0] for output in outputs))
        cache = self.cache()
        self.assertEqual(len(os.listdir(self.root / "entries")), 8 + 4 * 8)
        for name in os.listdir(self.root / "entries"):
            self.assertIsNotNone(cache.lookup(name))
        self.assert_no_staging()

    def test_foreign_lock_times_out_and_is_preserved(self) -> None:
        cache = self.cache(lock_timeout=0.2)
        cache.store(_key("seed"), {})
        lock = self.root / "writer.lock"
        lock.mkdir()
        marker = lock / wc._owner_name(os.getpid() + 1, "f" * 32)
        marker.touch()
        with self.assertRaises(CacheLockTimeout):
            cache.store(_key("blocked"), {})
        self.assertEqual(os.listdir(lock), [marker.name])
        self.assertIsNone(cache.lookup(_key("blocked")))
        self.assertIsNotNone(cache.lookup(_key("seed")))
        shutil.rmtree(lock)

    def test_own_stale_lock_is_reclaimed(self) -> None:
        cache = self.cache(lock_timeout=0.5)
        cache.store(_key("seed"), {})
        token = "e" * 32
        with wc._STATE_LOCK:
            wc._ISSUED_LOCK_TOKENS.add(token)
        lock = self.root / "writer.lock"
        lock.mkdir()
        (lock / wc._owner_name(os.getpid(), token)).touch()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNotNone(cache.store(_key("after"), {}))
        self.assert_no_staging()

    @unittest.skipUnless(os.name == "nt", "Windows sharing-violation semantics")
    def test_release_retries_transient_sharing_violation(self) -> None:
        cache = self.cache()
        real_rename = os.rename
        failures = {"n": 0}

        def flaky(src, dst):
            if Path(src).name == "writer.lock" and failures["n"] < 3:
                failures["n"] += 1
                raise PermissionError(13, "The process cannot access the file", str(src), 32)
            return real_rename(src, dst)

        with mock.patch.object(wc.os, "rename", flaky), warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNotNone(cache.store(_key("transient"), {}))
        self.assertEqual(failures["n"], 3)
        self.assert_no_staging()

    def test_persistent_release_failure_warns_then_own_process_reclaims(self) -> None:
        cache = self.cache(lock_timeout=2.0)
        real_rename = os.rename

        def stuck(src, dst):
            if Path(src).name == "writer.lock":
                raise PermissionError(13, "The process cannot access the file", str(src), 32)
            return real_rename(src, dst)

        with mock.patch.object(wc.os, "rename", stuck), mock.patch.object(wc, "TRANSIENT_RETRY_SECONDS", 0.05):
            with self.assertWarnsRegex(RuntimeWarning, "could not release cache lock"):
                first = cache.store(_key("stuck"), {})
        self.assertIsNotNone(first)
        self.assertTrue((self.root / "writer.lock").is_dir())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNotNone(cache.store(_key("after-stuck"), {}))
        self.assert_no_staging()


if __name__ == "__main__":
    unittest.main()
