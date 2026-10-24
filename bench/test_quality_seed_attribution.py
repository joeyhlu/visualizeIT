import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from .quality_contract import Frame
from .quality_seed_attribution import (
    SeedAttributionBackend,
    SeedAttributionObserver,
)
from .quality_time import LEGACY, PHYSICAL


K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
MASK = np.ones((2, 2), dtype=np.uint8)


def _pose(angle=0.0, translation=(0.0, 0.0, 1.0)):
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)
    result = np.eye(4)
    result[:3, :3] = [[cosine, -sine, 0.0],
                      [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    result[:3, 3] = translation
    return result


def _frame(frame_id=0, timestamp=1.0):
    return Frame(frame_id, np.zeros((2, 2, 3), dtype=np.uint8), K,
                 timestamp, PHYSICAL)


def _candidate(pose):
    return SimpleNamespace(pose=pose)


class _Backend:
    def __init__(self, recoveries=(), refine_results=()):
        self.recoveries = list(recoveries)
        self.refine_results = list(refine_results)
        self.recover_calls = 0
        self.refine_calls = 0
        self.seeds_seen = []

    def recover(self, frame, mask, top_k=5):
        self.recover_calls += 1
        return self.recoveries.pop(0)

    def refine(self, frame, mask, seed):
        self.refine_calls += 1
        self.seeds_seen.append(np.array(seed, copy=True))
        value = self.refine_results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _trace_row(frame, candidates, *, object_id="object-a", sequence_id="seq-a",
               source_frame_id=None):
    return {
        "object_id": object_id,
        "sequence_id": sequence_id,
        "source_frame_id": frame.frame_id + 100 if source_frame_id is None else source_frame_id,
        "frame_id": frame.frame_id,
        "timestamp_s": frame.timestamp_s,
        "clock_mode": frame.clock_mode,
        "candidates": candidates,
        "pose_state": "lost",
        "render_state": "suppressed",
        "failure_reason": None,
    }


def _trace_candidate(stage, index, pose):
    return {
        "stage": stage,
        "original_index": index,
        "pose": None if pose is None else np.asarray(pose).tolist(),
    }


def _write_trace(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8")


class SeedAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_tempdir = tempfile.tempdir
        cls._temp_root = Path.cwd() / ".cache" / "test-temp-seed-attribution"
        cls._temp_root.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(cls._temp_root)

    @classmethod
    def tearDownClass(cls):
        tempfile.tempdir = cls._old_tempdir

    def _observer(self, source_ids=None, **kwargs):
        return SeedAttributionObserver(
            object_id="object-a", sequence_id="seq-a",
            source_frame_ids=source_ids or {0: 100, 1: 101}, **kwargs)

    def test_backend_delegates_once_preserves_return_identity_and_inputs(self):
        frame = _frame()
        seed = _pose(15.0)
        duplicate = _pose(25.0)
        recovery_result = [duplicate, duplicate]
        returned = _candidate(_pose(5.0))
        backend = _Backend([recovery_result], [returned])
        observer = self._observer()
        wrapped = SeedAttributionBackend(backend, observer)

        actual_recovery = wrapped.recover(frame, MASK, top_k=5)
        actual_return = wrapped.refine(frame, MASK, seed)

        self.assertIs(actual_recovery, recovery_result)
        self.assertIs(actual_return, returned)
        self.assertEqual(backend.recover_calls, 1)
        self.assertEqual(backend.refine_calls, 1)
        np.testing.assert_array_equal(seed, _pose(15.0))
        np.testing.assert_array_equal(backend.seeds_seen[0], seed)
        self.assertIs(wrapped.refiner if hasattr(backend, "refiner") else backend,
                      backend)

    def test_prior_refinement_duplicate_recovery_and_paired_dispersion(self):
        frame = _frame()
        initial_seed = _pose(3.0)
        seeds = [_pose(0.0), _pose(0.0), _pose(60.0)]
        returned = [_pose(1.0), _pose(2.0), _pose(3.0)]
        initial_return = _pose(4.0)
        backend = _Backend(
            [seeds],
            [_candidate(initial_return)] + [_candidate(value) for value in returned],
        )
        observer = self._observer()
        wrapped = SeedAttributionBackend(backend, observer)
        wrapped.refine(frame, MASK, initial_seed)
        generated = wrapped.recover(frame, MASK)
        for seed in generated:
            wrapped.refine(frame, MASK, seed)

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "candidates.jsonl"
            _write_trace(trace, [_trace_row(frame, [
                _trace_candidate("refinement", 0, initial_return),
                *[_trace_candidate("recovery", index, value)
                  for index, value in enumerate(returned)],
            ])])
            report = observer.report(trace)

        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["generated_seeds"], 3)
        self.assertEqual(report["counts"]["generated_seed_snapshots"], 3)
        self.assertEqual(report["counts"]["consumed_refinement_seeds"], 1)
        self.assertEqual(report["counts"]["consumed_recovery_seeds"], 3)
        self.assertEqual(report["counts"]["matched_attempts"], 4)
        self.assertEqual(report["paired_recovery_dispersion"]["pair_count"], 3)
        self.assertEqual(
            report["paired_recovery_dispersion"]["seed_dispersion"]["rotation_deg"]["min"],
            0.0,
        )

    def test_none_return_is_failed_attempt_but_complete_when_trace_matches(self):
        frame = _frame()
        backend = _Backend(refine_results=[None])
        observer = self._observer()
        result = SeedAttributionBackend(backend, observer).refine(frame, MASK, _pose())
        self.assertIsNone(result)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(
                frame, [_trace_candidate("refinement", 0, None)])])
            report = observer.report(trace)
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["none_returns"], 1)
        self.assertEqual(report["failed_attempts"][0]["outcome"], "none")
        self.assertIsNone(report["seed_to_return"]["all"])
        self.assertIsNone(report["paired_recovery_dispersion"])

    def test_intermediate_success_proxy_final_none_uses_actual_return(self):
        class ProxyBackend(_Backend):
            def refine(self, frame, mask, seed):
                self.refine_calls += 1
                self.intermediate = _candidate(_pose(45.0))
                return None

        frame = _frame()
        backend = ProxyBackend()
        observer = self._observer()
        self.assertIsNone(SeedAttributionBackend(backend, observer).refine(
            frame, MASK, _pose()))
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(
                frame, [_trace_candidate("refinement", 0, None)])])
            report = observer.report(trace)
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["returned_candidates"], 0)
        self.assertEqual(report["counts"]["none_returns"], 1)

    def test_invalid_return_and_backend_exception_are_incomplete_and_exception_wins(self):
        frame = _frame()
        observer = self._observer()
        invalid = _candidate(np.zeros((3, 3)))
        returned = SeedAttributionBackend(
            _Backend(refine_results=[invalid]), observer).refine(frame, MASK, _pose())
        self.assertIs(returned, invalid)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(
                frame, [_trace_candidate("refinement", 0, None)])])
            report = observer.report(trace)
        self.assertFalse(report["complete"])
        self.assertGreater(report["failures"]["total"], 0)

        error = LookupError("backend sentinel")
        observer2 = self._observer()
        with self.assertRaises(LookupError) as caught:
            SeedAttributionBackend(
                _Backend(refine_results=[error]), observer2).refine(
                    frame, MASK, _pose())
        self.assertIs(caught.exception, error)

    def test_mutating_failing_observer_cannot_change_frame_seed_return_or_rng(self):
        class MutatingObserver(SeedAttributionObserver):
            def _refine_started(self, frame_descriptor, seed, snapshot_error=None):
                random.random()
                np.random.random()
                frame_descriptor["frame_id"] = 999
                seed[:] = -999.0
                raise RuntimeError("observer start failure")

            def _refine_finished(self, token, returned_pose, outcome,
                                 exception_type=None, snapshot_error=None):
                random.random()
                np.random.random()
                returned_pose[:] = 777.0
                raise RuntimeError("observer finish failure")

        frame = _frame()
        original_rgb = frame.rgb.copy()
        seed = _pose(12.0)
        original_seed = seed.copy()
        candidate_pose = _pose(6.0)
        result = _candidate(candidate_pose.copy())
        backend = _Backend(refine_results=[result])
        observer = MutatingObserver(
            object_id="object-a", sequence_id="seq-a",
            source_frame_ids={0: 100})
        wrapped = SeedAttributionBackend(backend, observer)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        actual = wrapped.refine(frame, MASK, seed)
        observed_python = random.random()
        observed_numpy = np.random.random()
        random.setstate(python_state)
        expected_python = random.random()
        np.random.set_state(numpy_state)
        expected_numpy = np.random.random()

        self.assertIs(actual, result)
        np.testing.assert_array_equal(frame.rgb, original_rgb)
        np.testing.assert_array_equal(seed, original_seed)
        np.testing.assert_array_equal(backend.seeds_seen[0], original_seed)
        np.testing.assert_array_equal(result.pose, candidate_pose)
        self.assertEqual(observed_python, expected_python)
        self.assertEqual(observed_numpy, expected_numpy)
        self.assertGreater(observer._failure_count, 0)

    def test_existing_torch_module_rng_is_restored_without_import(self):
        class FakeCuda:
            def __init__(self, owner): self.owner = owner
            def is_initialized(self): return True
            def get_rng_state_all(self): return list(self.owner.cuda_state)
            def set_rng_state_all(self, state): self.owner.cuda_state = list(state)

        class FakeTorch:
            def __init__(self):
                self.state = 7
                self.cuda_state = [11]
                self.cuda = FakeCuda(self)
            def get_rng_state(self): return self.state
            def set_rng_state(self, state): self.state = state

        fake_torch = FakeTorch()

        class TorchMutator(SeedAttributionObserver):
            def _refine_started(self, *args, **kwargs):
                sys.modules["torch"].state = 99
                sys.modules["torch"].cuda_state = [101]
                return super()._refine_started(*args, **kwargs)

        observer = TorchMutator(object_id="object-a", sequence_id="seq-a",
                                source_frame_ids={0: 100})
        backend = _Backend(refine_results=[None])
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            SeedAttributionBackend(backend, observer).refine(
                _frame(), MASK, _pose())
        self.assertEqual(fake_torch.state, 7)
        self.assertEqual(fake_torch.cuda_state, [11])

    def test_full_identity_occurrence_join_and_stale_collision_rejection(self):
        frame0 = _frame(0, 1.0)
        frame1 = _frame(1, 2.0)
        poses = [_pose(1.0), _pose(2.0), _pose(3.0)]
        observer = self._observer()
        wrapped = SeedAttributionBackend(
            _Backend(refine_results=[_candidate(value) for value in poses]), observer)
        wrapped.refine(frame0, MASK, _pose())
        wrapped.refine(frame1, MASK, _pose())
        wrapped.refine(frame0, MASK, _pose())
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            rows = [
                _trace_row(frame0, [_trace_candidate("refinement", 0, poses[0])]),
                _trace_row(frame1, [_trace_candidate("refinement", 0, poses[1])]),
                _trace_row(frame0, [_trace_candidate("refinement", 0, poses[2])]),
            ]
            _write_trace(trace, rows)
            report = observer.report(trace)
            self.assertTrue(report["complete"])

            stale = Path(directory) / "stale.jsonl"
            rows[2]["source_frame_id"] = 101
            _write_trace(stale, rows)
            stale_report = observer.report(stale)
        self.assertFalse(stale_report["complete"])
        self.assertGreater(stale_report["failures"]["total"], 0)

    def test_zero_call_frame_is_complete_and_unrequested_inventory_is_absent(self):
        observer = self._observer(source_ids={0: 100})
        frame = _frame()
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(frame, [])])
            report = observer.report(trace)
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["trace_frames"], 1)
        self.assertEqual(report["counts"]["zero_call_trace_frames"], 1)
        self.assertEqual(report["counts"]["consumed_refine_attempts"], 0)

    def test_event_and_pair_caps_make_receipt_incomplete(self):
        frame = _frame()
        observer = self._observer(max_events=1)
        wrapped = SeedAttributionBackend(
            _Backend(recoveries=[[_pose()]], refine_results=[None]), observer)
        seed = wrapped.recover(frame, MASK)[0]
        wrapped.refine(frame, MASK, seed)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(
                frame, [_trace_candidate("recovery", 0, None)])])
            report = observer.report(trace)
        self.assertFalse(report["complete"])
        self.assertGreater(report["counts"]["dropped_events"], 0)
        self.assertTrue(any(row["kind"] == "event_cap_exhausted"
                            for row in report["failures"]["reported"]))

        seeds = [_pose(0.0), _pose(30.0), _pose(60.0)]
        returns = [_pose(1.0), _pose(2.0), _pose(3.0)]
        observer2 = self._observer(max_pairs=1)
        wrapped2 = SeedAttributionBackend(
            _Backend([seeds], [_candidate(value) for value in returns]), observer2)
        for seed in wrapped2.recover(frame, MASK):
            wrapped2.refine(frame, MASK, seed)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(frame, [
                _trace_candidate("recovery", index, value)
                for index, value in enumerate(returns)
            ])])
            report2 = observer2.report(trace)
        self.assertFalse(report2["complete"])
        self.assertEqual(report2["paired_recovery_dispersion"]["pair_count"], 1)

    def test_pose_copy_is_shape_bounded_and_recovery_snapshot_overflow_is_incomplete(self):
        class HugePose:
            shape = (100_000, 100_000)
            def __array__(self, *args, **kwargs):
                raise AssertionError("unexpected huge pose copy")

        frame = _frame()
        huge_candidate = _candidate(HugePose())
        observer = self._observer()
        actual = SeedAttributionBackend(
            _Backend(refine_results=[huge_candidate]), observer).refine(
                frame, MASK, _pose())
        self.assertIs(actual, huge_candidate)
        self.assertGreater(observer._failure_count, 0)

        seeds = [_pose(0.0), _pose(10.0), _pose(20.0)]
        observer2 = self._observer(max_events=8)
        backend = _Backend(recoveries=[seeds])
        actual_seeds = SeedAttributionBackend(backend, observer2).recover(
            frame, MASK, top_k=2)
        self.assertIs(actual_seeds, seeds)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace, [_trace_row(frame, [])])
            report = observer2.report(trace)
        self.assertFalse(report["complete"])
        self.assertEqual(report["counts"]["generated_seeds"], 3)
        self.assertEqual(report["counts"]["generated_seed_snapshots"], 2)
        self.assertTrue(any(row["kind"] == "recovery_seed_snapshot_overflow"
                            for row in report["failures"]["reported"]))

    def test_empty_trace_is_missing_evidence_while_explicit_zero_call_is_valid(self):
        observer = self._observer(source_ids={0: 100})
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "empty.jsonl"
            trace.write_bytes(b"")
            report = observer.report(trace)
        self.assertFalse(report["complete"])
        self.assertTrue(any(row["kind"] == "candidate_trace_validation_failed"
                            for row in report["failures"]["reported"]))

    def test_finalize_writes_derived_only_report_and_clears_snapshots(self):
        frame = _frame()
        observer = self._observer()
        result = _pose(8.0)
        SeedAttributionBackend(
            _Backend(refine_results=[_candidate(result)]), observer).refine(
                frame, MASK, _pose())
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            output = Path(directory) / "report.json"
            _write_trace(trace, [_trace_row(
                frame, [_trace_candidate("refinement", 0, result)])])
            report = observer.finalize(trace, output)
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["complete"])
        self.assertEqual(loaded, report)
        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)
        self.assertNotIn("seed_pose", set(keys(report)))
        self.assertNotIn("returned_pose", set(keys(report)))
        self.assertEqual(observer._events, [])
        with self.assertRaises(RuntimeError):
            observer.report(trace)

    def test_finalize_collision_preserves_winner_and_cleans_private_state(self):
        frame = _frame()
        observer = self._observer()
        result = _pose(9.0)
        SeedAttributionBackend(
            _Backend(refine_results=[_candidate(result)]), observer).refine(
                frame, MASK, _pose())
        winner = b"concurrent-winner-bytes"
        real_link = os.link

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            output = root / "report.json"
            _write_trace(trace, [_trace_row(
                frame, [_trace_candidate("refinement", 0, result)])])

            def collide(source, destination):
                Path(destination).write_bytes(winner)
                return real_link(source, destination)

            with mock.patch("bench.quality_seed_attribution.os.link",
                            side_effect=collide):
                report = observer.finalize(trace, output)

            self.assertEqual(output.read_bytes(), winner)
            self.assertEqual(list(root.glob(".*.part")), [])

        self.assertFalse(report["complete"])
        self.assertTrue(any(row["kind"] == "report_output_collision"
                            for row in report["failures"]["reported"]))
        self.assertEqual(observer._events, [])
        self.assertIsNone(observer._active_recovery)

    def test_runner_forwards_physical_observer_installs_adapter_and_preserves_mug_guard(self):
        from . import quality_runner

        delegate = object()
        self.assertIs(quality_runner._install_seed_attribution(delegate, None), delegate)
        observer = self._observer()
        installed = quality_runner._install_seed_attribution(delegate, observer)
        self.assertIsInstance(installed, SeedAttributionBackend)
        self.assertIs(installed.backend, delegate)

        physical_manifest = {
            "object": "carton",
            "_timing": {"clock_mode": PHYSICAL},
        }
        sentinel = object()
        with (mock.patch.object(quality_runner, "read_input",
                               return_value=physical_manifest),
              mock.patch.object(quality_runner, "_pnp_tracking_settings",
                                return_value={"pnp_use_extrinsic_guess": True}),
              mock.patch.object(quality_runner, "_physical_mask_cache",
                                return_value=({"frames": []}, {})),
              mock.patch.object(quality_runner, "_pose_stage_physical",
                                return_value=sentinel) as physical):
            actual = quality_runner.pose_stage(
                Path("bundle"), Path("masks"), Path("output"), "cpu", "complete",
                seed_attribution_observer=observer)
        self.assertIs(actual, sentinel)
        self.assertIs(physical.call_args.args[-1], observer)

        legacy_manifest = {"object": "mug", "_timing": {"clock_mode": LEGACY}}
        with mock.patch.object(quality_runner, "read_input",
                               return_value=legacy_manifest):
            with self.assertRaisesRegex(ValueError, "only physical-time"):
                quality_runner.pose_stage(
                    Path("bundle"), Path("masks"), Path("output"), "cpu", "complete",
                    seed_attribution_observer=observer)

        with (mock.patch.object(quality_runner, "read_input",
                               return_value=legacy_manifest),
              mock.patch.object(quality_runner, "_pnp_tracking_settings",
                                return_value={"pnp_use_extrinsic_guess": True})):
            with self.assertRaisesRegex(ValueError, "requires both capture root"):
                quality_runner.pose_stage(
                    Path("bundle"), Path("masks"), Path("output"), "cpu", "complete",
                    chronological_mug_capture_root=Path("capture"))


if __name__ == "__main__":
    unittest.main()
