"""Independent fake-runtime acceptance tests for the frozen SAM 2 adapter.

No model, media decoder, image library, child process, or network service is
used. The real-session tests below call the producer's actual residency methods
against tiny device-aware tensor stubs.
"""
import builtins
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import quality_sam2 as producer
from .quality_time import LEGACY, PHYSICAL


ROOT = Path(__file__).resolve().parents[1]


def encode(document):
    return json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")


def physical_document(*, width=32, height=24):
    source_ids = (260, 262, 263)
    ticks = (1_000_000, 1_033_333, 1_066_667)
    return {
        "schema_version": 2,
        "object": "synthetic-selected-object",
        "video": "video.mp4",
        "asset": "object.glb",
        "source_hashes": {"video.mp4": "0" * 64, "object.glb": "0" * 64},
        "native_resolution": [width, height],
        "clock": {
            "mode": PHYSICAL,
            "units": "seconds",
            "source_units": "microseconds",
            "timestamp_source": "synthetic capture clock",
            "nominal_fps": 30,
        },
        "timeline": [
            {
                "frame_id": index,
                "source_frame_id": source_id,
                "source_timestamp": tick,
                "timestamp_s": tick / 1_000_000,
                "role": "setup" if index == 0 else "scored",
            }
            for index, (source_id, tick) in enumerate(zip(source_ids, ticks))
        ],
        "setup_frame_id": 0,
        "frame_ids": [1, 2],
        "source_gaps": [
            {
                "previous_source_frame_id": 260,
                "current_source_frame_id": 262,
                "missing_capture_count": 1,
            }
        ],
        "selection": {
            "reviewed": True,
            "frame_id": 0,
            "source_frame_id": 260,
            "timestamp_s": 1.0,
            "points": [[2, 3], [12, 13]],
            "labels": [1, 0],
        },
        "units": "metres",
        "intrinsics": [[20, 0, width / 2], [0, 20, height / 2], [0, 0, 1]],
    }


def write_physical_bundle(root, *, width=32, height=24, bad_video_hash=False):
    bundle = Path(root) / "input"
    bundle.mkdir(parents=True)
    bodies = {
        "video.mp4": b"synthetic resource placeholder; never decoded",
        "object.glb": b"synthetic asset placeholder; never loaded",
    }
    document = physical_document(width=width, height=height)
    for name, body in bodies.items():
        (bundle / name).write_bytes(body)
        document["source_hashes"][name] = hashlib.sha256(body).hexdigest()
    if bad_video_hash:
        document["source_hashes"]["video.mp4"] = "f" * 64
    (bundle / "input.json").write_bytes(encode(document))
    return bundle, document


def write_legacy_bundle(root, object_name):
    source = ROOT / ".cache" / "model-quality" / "inputs" / object_name / "input.json"
    raw = source.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    bundle = Path(root) / "input"
    bundle.mkdir(parents=True)
    (bundle / "input.json").write_bytes(raw)
    for name in manifest["source_hashes"]:
        resource = bundle / name
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_bytes(b"synthetic legacy placeholder; injected runtime never reads it")
    return bundle, raw, manifest


class Mask:
    def __init__(self, width, height, foreground_pixels):
        self.width = width
        self.height = height
        self.foreground_pixels = foreground_pixels

    def __gt__(self, threshold):
        return SyntheticBoolMask(self, threshold)


class SyntheticBoolMask:
    def __init__(self, source, threshold):
        self.source = source
        self.threshold = threshold
        self.dtype = "bool"


class ProtocolFakeSession:
    """Public runtime seam with metadata-only masks and placeholder artifacts."""

    def __init__(self, plan, manifest, *, fault=None):
        self.plan = plan
        self.manifest = manifest
        self.fault = fault
        self.width, self.height = manifest["native_resolution"]
        self.calls = []
        self.provenance = {"synthetic_only": True}
        self.checkpoint_sha256 = "c" * 64
        key = plan["selection_key"]
        source_id = plan["source_frame_ids"][key.frame_id]
        self.failure_exception = producer.SourceDecodeError(key.frame_id, source_id)

    def initialize(self):
        self.calls.append(("initialize",))
        if self.fault == "initial_decode":
            raise self.failure_exception

    def add_initial_selection(self, selection):
        self.calls.append(("selection",))

    def step(self, index):
        self.calls.append(("step", index))
        if self.fault == "middle_decode" and index == 1:
            raise self.failure_exception
        obj_id = 1
        if index == 1 and self.fault == "float_obj_id":
            obj_id = 1.0
        elif index == 1 and self.fault == "bool_obj_id":
            obj_id = True
        blank = self.fault == "blank_source" and index == 1
        foreground = 80
        if index == 1 and self.fault in {
                "zero_detector", "bad_restore", "positive_detector", "unreaped_child",
                "failed_child", "legacy_detector_error"}:
            foreground = 0
        return {
            "frame_idx": index,
            "obj_ids": [obj_id],
            "mask": Mask(self.width, self.height, foreground),
            "blank": blank,
            "timings_ms": {"segmentation": 0.1},
            "source": self.snapshot_source(index),
        }

    def mask_info(self, mask):
        return {
            "shape": (mask.height, mask.width),
            "dtype": "uint8",
            "binary_0_255": True,
            "foreground_pixels": mask.foreground_pixels,
        }

    def empty_mask(self):
        return Mask(self.width, self.height, 0)

    def write_mask(self, path, mask):
        path = Path(path)
        payload = f"fake-mask:{mask.width}:{mask.height}:{mask.foreground_pixels}".encode("ascii")
        if self.fault == "partial_writer" and path.name == "1.png":
            path.write_bytes(payload[: max(1, len(payload) // 2)])
            raise OSError("synthetic writer failed after a partial artifact")
        path.write_bytes(payload)
        return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}

    def offload_for_detector(self):
        self.calls.append(("offload",))
        return {
            "state_after": {"non_cpu_tensor_count": 0},
            "model_after": {"model_tensors_on_wrong_device": 0},
            "live_gpu_bytes_after_empty_cache": 0,
        }

    def run_detector(self, index, key, *, prior_frame_id=None):
        self.calls.append(("detector", index))
        positive = self.fault == "positive_detector"
        failed_child = self.fault == "failed_child"
        unreaped_child = self.fault == "unreaped_child"
        legacy_error = self.fault == "legacy_detector_error"
        mask = Mask(self.width, self.height, 80 if positive else 0)
        self.detector_mask = mask
        if legacy_error:
            return {
                "mask": None,
                "reason": "cnos_runtime_failure",
                "timings_ms": {"detection": 0.2},
                "process_returncode": 1,
                "process_reaped": True,
                "artifact_sha256": None,
            }
        return {
            "identity": {
                "frameId": key.frame_id,
                "sourceFrameId": self.plan["source_frame_ids"][key.frame_id],
                "timestamp_s": key.timestamp_s,
                "clock_mode": PHYSICAL,
                "input_manifest_sha256": self.plan["input_manifest_sha256"],
                "timestamp_table_sha256": self.plan["timestamp_table_sha256"],
            },
            "mask": mask,
            "process_returncode": 7 if failed_child else 0,
            "process_reaped": not unreaped_child,
            "artifact_sha256": "d" * 64,
            "timings_ms": {"detection": 0.2},
            "reason": None,
        }

    def restore_after_detector(self):
        self.calls.append(("restore",))
        mismatches = 1 if self.fault == "bad_restore" else 0
        return {"model": {"model_tensors_on_wrong_device": 0},
                "state": {"device_mismatches": mismatches}}

    def resume_after_detector(self, index):
        self.calls.append(("resume", index))

    def condition(self, index, mask):
        self.calls.append(("condition", index))

    def snapshot_source(self, index):
        return {
            "decode_attempted": 1,
            "decode_returned": 1,
            "decoder_requests": 1,
            "cache_hits": 0,
            "past_reads": 0,
            "decode_ms": 0.1,
            "decode_failure": None,
        }

    def close(self):
        self.calls.append(("close",))
        return {
            "iterator_released": True,
            "decoder_released": True,
            "predictor_released": True,
            "loader_hook_restored": True,
        }


class ProtocolFakeRuntime:
    def __init__(self, *, fault=None):
        self.fault = fault
        self.opens = 0
        self.session = None

    def open_session(self, *, plan, manifest, bundle, output, device):
        self.opens += 1
        self.session = ProtocolFakeSession(plan, manifest, fault=self.fault)
        return self.session


class PublicRuntimeProtocolTests(unittest.TestCase):
    def run_physical(self, root, *, fault=None, width=32, height=24):
        bundle, _ = write_physical_bundle(root, width=width, height=height)
        output = Path(root) / "masks"
        runtime = ProtocolFakeRuntime(fault=fault)
        result = producer.run(bundle, output, device="cpu", runtime=runtime)
        return result, runtime, output

    def test_blank_decoded_source_and_zero_detector_result_remain_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            blank, blank_runtime, _ = self.run_physical(Path(folder) / "blank", fault="blank_source")
            blank_row = blank["frames"][1]
            self.assertEqual(blank_row["source_state"], "decoded")
            self.assertEqual(blank_row["failure_reason"], "blank_current_image")
            self.assertFalse(any(call[0] == "detector" for call in blank_runtime.session.calls))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "zero"
            bundle, _ = write_physical_bundle(root)
            output = root / "masks"
            zero_runtime = ActualMethodRuntime(fault="zero_detector")
            zero = producer.run(bundle, output, device="cuda", runtime=zero_runtime)
            zero_row = zero["frames"][1]
            calls = zero_runtime.session.calls
            self.assertEqual(zero_row["failure_reason"], "automatic_detection_no_usable_foreground")
            self.assertEqual([call for call in calls if call[0] == "detector"], [("detector", 1)])
            self.assertIn(("resume", 1), calls)
            self.assertFalse(any(call[0] == "condition" for call in calls))
            self.assertEqual(zero["frames"][2]["mask_state"], "available")
            self.assertEqual(zero_runtime.session.predictor_ref.propagations[0][:2], (0, False))
            self.assertEqual(zero_runtime.session.predictor_ref.propagations[1][:2], (2, False))

    def test_float_and_boolean_iterator_object_ids_are_rejected(self):
        for fault in ("float_obj_id", "bool_obj_id"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                bundle, _ = write_physical_bundle(Path(folder))
                output = Path(folder) / "masks"
                runtime = ProtocolFakeRuntime(fault=fault)
                with self.assertRaisesRegex(RuntimeError, "physical cache stage failed"):
                    producer.run(bundle, output, device="cpu", runtime=runtime)
                record = json.loads((output / "results.json").read_text(encoding="utf-8"))
                self.assertFalse(record["stage_completed"])
                self.assertEqual(record["frames"][1]["measurement_state"], "unmeasured")
                self.assertIsNone(record["frames"][1]["path"])

    def test_failed_restore_receipt_blocks_resume_and_conditioning(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _ = write_physical_bundle(Path(folder))
            output = Path(folder) / "masks"
            runtime = ProtocolFakeRuntime(fault="bad_restore")
            with self.assertRaisesRegex(RuntimeError, "physical cache stage failed"):
                producer.run(bundle, output, device="cpu", runtime=runtime)
            calls = runtime.session.calls
            self.assertIn(("detector", 1), calls)
            self.assertIn(("restore",), calls)
            self.assertFalse(any(call[0] in ("resume", "condition") for call in calls))

    def test_actual_positive_detector_conditioning_uses_restored_state_and_next_ordinal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "positive"
            bundle, _ = write_physical_bundle(root)
            output = root / "masks"
            runtime = ActualMethodRuntime(fault="positive_detector")
            result = producer.run(bundle, output, device="cuda", runtime=runtime)

            row = result["frames"][1]
            self.assertEqual(row["mask_state"], "available")
            self.assertTrue(row["automatic_conditioning"])
            self.assertEqual(result["frames"][2]["mask_state"], "available")
            calls = runtime.session.calls
            tags = [call[0] for call in calls]
            resumed_index = next(i for i, call in enumerate(calls)
                                  if call[0] == "propagate" and call[2] == 2)
            self.assertLess(tags.index("detector"), tags.index("restore_actual"))
            self.assertLess(tags.index("restore_actual"), tags.index("condition_actual"))
            self.assertLess(tags.index("condition_actual"), tags.index("add_new_mask"))
            self.assertLess(tags.index("add_new_mask"), resumed_index)

            [condition] = runtime.session.predictor_ref.mask_calls
            self.assertEqual(condition["frame_idx"], 1)
            self.assertEqual(condition["obj_id"], 1)
            self.assertIs(condition["state"], runtime.session.predictor_ref.propagations[1][2])
            self.assertIs(condition["mask"].source, runtime.session.detector_mask)
            self.assertEqual(condition["mask"].threshold, 0)
            self.assertEqual(str(condition["state"]["obj_ptr"].device), "cuda:0")
            self.assertEqual(runtime.session.predictor_ref.propagations[1][:2], (2, False))

    def test_actual_init_and_prompt_hook_failures_restore_loader_and_cleanup(self):
        for fault in ("init_after_hook", "prompt"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / fault
                bundle, _ = write_physical_bundle(root)
                output = root / "masks"
                runtime = ActualMethodRuntime(fault=fault)
                with self.assertRaisesRegex(RuntimeError, "physical cache stage failed"):
                    producer.run(bundle, output, device="cuda", runtime=runtime)

                session = runtime.session
                record = json.loads((output / "results.json").read_text(encoding="utf-8"))
                self.assertIs(session.predictor_module_ref.load_video_frames,
                              session.original_loader_hook)
                self.assertTrue(session._hook_restored)
                self.assertEqual(record["cleanup"]["complete"], True)
                self.assertEqual(record["cleanup"]["loader_hook_restored"], True)
                self.assertEqual(record["stage_completed"], False)
                self.assertFalse(any(call[0] in ("step", "detector") for call in session.calls))

                if fault == "init_after_hook":
                    self.assertEqual(record["terminal_error"].split(":", 1)[0], "RuntimeError")
                    self.assertFalse(any(call[0] == "selection_actual" for call in session.calls))
                    self.assertIsNotNone(session.predictor_ref.loaded_video)
                    loader = session.predictor_ref.loaded_video[0]
                    self.assertTrue(session.predictor_ref.loaded_cap.released)
                    self.assertIsNone(loader.cap)
                    self.assertEqual(record["prompt_ledger"]["initial_selection"]["attempted"], 0)
                else:
                    prompt = record["prompt_ledger"]["initial_selection"]
                    self.assertEqual((prompt["attempted"], prompt["returned"]), (1, 0))
                    self.assertEqual({key: prompt[key] for key in
                                      ("frame_id", "source_frame_id", "timestamp_s", "points",
                                       "labels", "reviewed")},
                                     {"frame_id": 0, "source_frame_id": 260, "timestamp_s": 1.0,
                                      "points": [[2.0, 3.0], [12.0, 13.0]],
                                      "labels": [1, 0], "reviewed": True})
                    self.assertEqual(session.predictor_ref.mask_calls, [])
                    self.assertIsNotNone(session.predictor_ref.loaded_video)
                    loader = session.predictor_ref.loaded_video[0]
                    self.assertTrue(session.predictor_ref.loaded_cap.released)
                    self.assertIsNone(loader.cap)

    def test_unreaped_and_failed_children_are_terminal_without_restore_or_future_step(self):
        for fault in ("unreaped_child", "failed_child"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                bundle, _ = write_physical_bundle(Path(folder))
                output = Path(folder) / "masks"
                runtime = ProtocolFakeRuntime(fault=fault)
                with self.assertRaisesRegex(RuntimeError, "physical cache stage failed"):
                    producer.run(bundle, output, device="cpu", runtime=runtime)

                record = json.loads((output / "results.json").read_text(encoding="utf-8"))
                calls = runtime.session.calls
                self.assertEqual(record["stage_completed"], False)
                self.assertEqual(record["frames"][1]["failure_reason"],
                                 "detector_or_recovery_failure")
                self.assertEqual(record["frames"][1]["measurement_state"], "unmeasured")
                self.assertEqual(record["frames"][2]["failure_reason"],
                                 "segmentation_not_run_after_terminal_failure")
                self.assertFalse(any(call[0] in ("restore", "resume", "condition")
                                     for call in calls))
                self.assertEqual([call for call in calls if call[0] == "step"],
                                 [("step", 0), ("step", 1)])
                self.assertEqual(record["frames"][1]["operation_counts"]["returned"].get("detector", 0),
                                 0 if fault == "unreaped_child" else 1)

    def test_partial_writer_failure_keeps_partial_file_and_marks_suffix_unmeasured(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _ = write_physical_bundle(Path(folder))
            output = Path(folder) / "masks"
            runtime = ProtocolFakeRuntime(fault="partial_writer")
            with self.assertRaisesRegex(RuntimeError, "physical cache stage failed"):
                producer.run(bundle, output, device="cpu", runtime=runtime)

            partial = output / "masks" / "1.png"
            self.assertTrue(partial.is_file())
            self.assertGreater(partial.stat().st_size, 0)
            record = json.loads((output / "results.json").read_text(encoding="utf-8"))
            current = record["frames"][1]
            self.assertEqual(current["measurement_state"], "unmeasured")
            self.assertIsNone(current["path"])
            self.assertEqual(current["operation_counts"]["attempted"]["mask_write"], 1)
            self.assertEqual(current["operation_counts"]["returned"].get("mask_write", 0), 0)
            self.assertEqual(record["frames"][2]["failure_reason"],
                             "segmentation_not_run_after_terminal_failure")
            self.assertEqual([call for call in runtime.session.calls if call[0] == "step"],
                             [("step", 0), ("step", 1)])

    def test_bad_default_resource_hash_fails_before_runtime_or_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _ = write_physical_bundle(Path(folder), bad_video_hash=True)
            output = Path(folder) / "masks"
            created = []

            class ForbiddenRuntime:
                def __init__(self):
                    created.append("runtime")
                    raise AssertionError("default runtime constructed before resource authentication")

            class ForbiddenSession:
                def __init__(self, *args, **kwargs):
                    created.append("session")
                    raise AssertionError("real session constructed before resource authentication")

            native_roots = {"numpy", "cv2", "torch", "PIL", "sam2", "cnos"}
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name.split(".", 1)[0] in native_roots:
                    raise AssertionError("native/model import attempted before resource authentication: " + name)
                return original_import(name, *args, **kwargs)

            with patch.object(producer, "_RealRuntime", ForbiddenRuntime), \
                    patch.object(producer, "_RealSession", ForbiddenSession), \
                    patch("builtins.__import__", guarded_import):
                with self.assertRaisesRegex(ValueError, "source hash mismatch: video.mp4"):
                    producer.run(bundle, output, device="cpu")
            self.assertEqual(created, [])
            self.assertFalse(output.exists())

    def test_native_pixel_cap_output_confinement_and_freshness_precede_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "pixel-cap"
            root.mkdir()
            bundle, _ = write_physical_bundle(root, width=5000, height=4000)
            output = root / "masks"
            runtime = ProtocolFakeRuntime()
            with self.assertRaisesRegex(ValueError, "16,777,216-pixel"):
                producer.run(bundle, output, device="cpu", runtime=runtime)
            self.assertEqual(runtime.opens, 0)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "confinement"
            root.mkdir()
            bundle, _ = write_physical_bundle(root)
            runtime = ProtocolFakeRuntime()
            with self.assertRaisesRegex(ValueError, "outside the immutable input bundle"):
                producer.run(bundle, bundle / "nested-output", device="cpu", runtime=runtime)
            self.assertEqual(runtime.opens, 0)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "freshness"
            root.mkdir()
            bundle, _ = write_physical_bundle(root)
            output = root / "existing-output"
            output.mkdir()
            runtime = ProtocolFakeRuntime()
            with self.assertRaises(FileExistsError):
                producer.run(bundle, output, device="cpu", runtime=runtime)
            self.assertEqual(runtime.opens, 0)


class Device:
    def __init__(self, name):
        self.name = str(name)
        self.type = self.name.split(":", 1)[0]

    def __str__(self):
        return self.name


class SyntheticTensor:
    def __init__(self, device, *, stuck=False, values=(1.25, -7.5), dtype="float16"):
        self.device = Device(device)
        self.stuck = stuck
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
        self.requires_grad = True

    def detach(self):
        return self

    def to(self, device):
        if self.stuck:
            return self
        return SyntheticTensor(device, values=self.values, dtype=self.dtype)

    def numel(self):
        return 4

    def element_size(self):
        return 4


class FakeCuda:
    def __init__(self, live_bytes=0):
        self.synchronizations = 0
        self.empty_cache_calls = 0
        self.live_bytes = live_bytes

    def synchronize(self):
        self.synchronizations += 1

    def empty_cache(self):
        self.empty_cache_calls += 1

    def memory_allocated(self):
        return self.live_bytes


class FakeIterator:
    def __init__(self):
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def close(self):
        self.closed = True


class FakePredictor:
    def __init__(self, *, stuck_on_cpu=False, trace=None):
        self._parameters = [SyntheticTensor("cuda:0")]
        self._buffers = [SyntheticTensor("cuda:0")]
        self.stuck_on_cpu = stuck_on_cpu
        self.propagations = []
        self.trace = trace if trace is not None else []
        self.mask_calls = []
        self.module = None
        self.state_to_return = None
        self.init_fault = None
        self.prompt_fault = None
        self.loaded_cap = None

    def init_state(self, video_path, **kwargs):
        self.trace.append(("predictor_init_state", video_path, kwargs))
        if self.module is not None:
            loader, height, width = self.module.load_video_frames(video_path, image_size=2)
            self.loaded_video = (loader, height, width)
            self.loaded_cap = loader.cap
        if self.init_fault == "init_after_hook":
            raise RuntimeError("synthetic init warmup failure after loader hook use")
        return self.state_to_return

    def add_new_points_or_box(self, state, *, frame_idx, obj_id, points, labels):
        self.trace.append(("add_new_points_or_box", state, frame_idx, obj_id, points, labels))
        if self.prompt_fault == "prompt":
            raise RuntimeError("synthetic initial prompt failure")

    def add_new_mask(self, state, *, frame_idx, obj_id, mask):
        self.trace.append(("add_new_mask", state, frame_idx, obj_id, mask))
        self.mask_calls.append({"state": state, "frame_idx": frame_idx,
                                "obj_id": obj_id, "mask": mask})

    def parameters(self):
        return list(self._parameters)

    def buffers(self):
        return list(self._buffers)

    def to(self, device):
        if device == "cpu" and self.stuck_on_cpu:
            return self
        self._parameters = [tensor.to(device) for tensor in self._parameters]
        self._buffers = [tensor.to(device) for tensor in self._buffers]
        return self

    def propagate_in_video(self, state, *, start_frame_idx, reverse):
        self.propagations.append((start_frame_idx, reverse, state))
        self.trace.append(("propagate", state, start_frame_idx, reverse))
        return FakeIterator()


class FakeLoader:
    def __init__(self, length):
        self.length = length
        self.closed = False

    def __len__(self):
        return self.length

    def close(self):
        self.closed = True

    def snapshot(self, index):
        return {}


class RealSessionResidencyTests(unittest.TestCase):
    def session(self, plan, *, stuck_state=False, stuck_model=False, live_bytes=0):
        session = object.__new__(producer._RealSession)
        session.device = "cuda"
        session.torch = types.SimpleNamespace(Tensor=SyntheticTensor, cuda=FakeCuda(live_bytes))
        session.predictor = FakePredictor(stuck_on_cpu=stuck_model)
        pointer = SyntheticTensor("cuda:0", stuck=stuck_state,
                                  values=(3.125, -0.5, 9.75), dtype="bfloat16")
        storage = SyntheticTensor("cpu", values=(0, 255, 17), dtype="uint8")
        session.state = {
            "obj_ptr": pointer,
            "pointer_aliases": [pointer],
            "cpu_storage": storage,
            "cached_features": {"old": "clear before offload"},
        }
        session.original_pointer = pointer
        session.original_storage = storage
        session.iterator = FakeIterator()
        session.loader = FakeLoader(len(plan["planned_keys"]))
        session._state_device_transport = None
        session._restored_devices = None
        session.original_loader = None
        session._hook_restored = True
        session._closed = False
        return session

    def test_actual_real_session_offload_restore_aliases_and_zero_detection_resume(self):
        bundle = ROOT / ".cache" / "model-quality" / "inputs" / "keyboard" / "input.json"
        plan = producer.prepare_cache_plan(bundle.read_bytes(), limit=2)
        session = self.session(plan)
        old_iterator = session.iterator

        offload = session.offload_for_detector()
        self.assertTrue(old_iterator.closed)
        self.assertIsNone(session.iterator)
        self.assertEqual(offload["state_after"]["non_cpu_tensor_count"], 0)
        self.assertEqual(offload["model_after"]["model_tensors_on_wrong_device"], 0)
        self.assertEqual(offload["live_gpu_bytes_after_empty_cache"], 0)
        self.assertEqual(offload["state_original_device_counts"], {"cpu": 1, "cuda:0": 1})
        self.assertEqual(session.state["cached_features"], {})
        self.assertEqual(session.state["obj_ptr"].values, (3.125, -0.5, 9.75))
        self.assertEqual(session.state["obj_ptr"].dtype, "bfloat16")
        self.assertEqual(session.state["obj_ptr"].shape, (3,))
        self.assertEqual(session.state["cpu_storage"].values, (0, 255, 17))
        self.assertEqual(session.state["cpu_storage"].dtype, "uint8")

        restored = session.restore_after_detector()
        pointer = session.state["obj_ptr"]
        self.assertIs(pointer, session.state["pointer_aliases"][0])
        self.assertEqual(str(pointer.device), "cuda:0")
        self.assertEqual(str(session.state["cpu_storage"].device), "cpu")
        self.assertIsNot(pointer, session.original_pointer)
        self.assertEqual(pointer.values, (3.125, -0.5, 9.75))
        self.assertEqual(pointer.dtype, "bfloat16")
        self.assertEqual(pointer.shape, (3,))
        self.assertEqual(pointer.requires_grad, session.original_pointer.requires_grad)
        self.assertEqual(session.state["cpu_storage"].values, (0, 255, 17))
        self.assertEqual(session.state["cpu_storage"].dtype, "uint8")
        self.assertEqual(restored["model"]["model_tensors_on_wrong_device"], 0)
        self.assertEqual(restored["state"]["device_mismatches"], 0)

        session.resume_after_detector(1)
        self.assertEqual(session.predictor.propagations[0][:2], (2, False))
        self.assertIs(session.predictor.propagations[0][2], session.state)
        self.assertIsNotNone(session.iterator)

    def test_actual_offload_blocks_residual_state_or_model_devices(self):
        plan = producer.prepare_cache_plan(
            (ROOT / ".cache" / "model-quality" / "inputs" / "keyboard" / "input.json").read_bytes(),
            limit=2,
        )
        for stuck_state, stuck_model in ((True, False), (False, True)):
            with self.subTest(stuck_state=stuck_state, stuck_model=stuck_model):
                session = self.session(plan, stuck_state=stuck_state, stuck_model=stuck_model)
                with self.assertRaisesRegex(RuntimeError, "offload audit failed"):
                    session.offload_for_detector()

    def test_actual_offload_rejects_live_fake_gpu_allocation(self):
        plan = producer.prepare_cache_plan(
            (ROOT / ".cache" / "model-quality" / "inputs" / "keyboard" / "input.json").read_bytes(),
            limit=2,
        )
        session = self.session(plan, live_bytes=128)
        with self.assertRaisesRegex(RuntimeError, "offload audit failed"):
            session.offload_for_detector()

    def test_actual_restore_rejects_unreceipted_tensor_transport(self):
        plan = producer.prepare_cache_plan(
            (ROOT / ".cache" / "model-quality" / "inputs" / "keyboard" / "input.json").read_bytes(),
            limit=2,
        )
        session = self.session(plan)
        session.offload_for_detector()
        session.state["late_tensor"] = SyntheticTensor("cpu")
        with self.assertRaisesRegex(RuntimeError, "unaudited tensor"):
            session.restore_after_detector()


class ActualMethodProtocolSession(ProtocolFakeSession, producer._RealSession):
    """Public-run fake whose residency and resume hooks are real SAM methods."""

    def __init__(self, plan, manifest, *, fault=None):
        ProtocolFakeSession.__init__(self, plan, manifest, fault=fault)
        self.device = "cuda"
        self.np = FakeNumpy()
        self.cv2 = FakeCV(PixelFrame(self.width, self.height))
        self.torch = FakeTorch()
        self.torch.Tensor = SyntheticTensor
        self.torch.cuda = FakeCuda()
        pointer = SyntheticTensor("cuda:0")
        self.state = {
            "obj_ptr": pointer,
            "pointer_aliases": [pointer],
            "cpu_storage": SyntheticTensor("cpu"),
            "cached_features": {"discard": "before detector"},
        }
        self.predictor = FakePredictor(trace=self.calls)
        self.predictor_ref = self.predictor
        self.predictor.state_to_return = self.state
        self.predictor.init_fault = fault
        self.predictor.prompt_fault = fault
        self.original_loader_hook = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("original predictor loader hook should be replaced during init"))
        self.predictor_module = types.SimpleNamespace(load_video_frames=self.original_loader_hook)
        self.predictor_module_ref = self.predictor_module
        self.predictor.module = self.predictor_module
        self.iterator = None
        self.loader = None
        self._state_device_transport = None
        self._restored_devices = None
        self.original_loader = None
        self._hook_restored = True
        self._closed = False
        self.loaded_video = None

    def initialize(self):
        self.calls.append(("initialize_actual",))
        return producer._RealSession.initialize(self)

    def add_initial_selection(self, selection):
        self.calls.append(("selection_actual",))
        return producer._RealSession.add_initial_selection(self, selection)

    def offload_for_detector(self):
        self.calls.append(("offload_actual",))
        return producer._RealSession.offload_for_detector(self)

    def restore_after_detector(self):
        self.calls.append(("restore_actual",))
        return producer._RealSession.restore_after_detector(self)

    def resume_after_detector(self, index):
        self.calls.append(("resume", index))
        return producer._RealSession.resume_after_detector(self, index)

    def close(self):
        self.calls.append(("close_actual",))
        return producer._RealSession.close(self)

    def condition(self, index, mask):
        self.calls.append(("condition_actual", index))
        return producer._RealSession.condition(self, index, mask)


class ActualMethodRuntime:
    def __init__(self, *, fault=None):
        self.fault = fault
        self.session = None

    def open_session(self, *, plan, manifest, bundle, output, device):
        self.session = ActualMethodProtocolSession(plan, manifest, fault=self.fault)
        self.session.bundle = Path(bundle)
        return self.session


class PixelFrame:
    def __init__(self, width, height, *, dtype="uint8", ndim=3):
        self.shape = (height, width, 3) if ndim == 3 else (height, width)
        self.dtype = dtype
        self.ndim = ndim


class NumericArray:
    """Tiny nested-list image that keeps preprocessing tests library-free."""

    def __init__(self, pixels, *, dtype="uint8"):
        self.pixels = [[list(pixel) for pixel in row] for row in pixels]
        self.dtype = dtype
        self.ndim = 3
        self.shape = (len(self.pixels), len(self.pixels[0]), len(self.pixels[0][0]))

    def copy(self):
        return NumericArray(self.pixels, dtype=self.dtype)

    def __setitem__(self, key, value):
        if key != slice(None):
            raise AssertionError("synthetic array only supports full-frame assignment")
        self.pixels = [[list(value) for _ in row] for row in self.pixels]


class NumericTensor:
    def __init__(self, values, *, dtype="uint8"):
        self.values = values
        self.dtype = dtype

    def permute(self, *axes):
        if axes != (2, 0, 1):
            raise AssertionError(f"unexpected channel permutation: {axes!r}")
        height = len(self.values)
        width = len(self.values[0])
        channels = len(self.values[0][0])
        return NumericTensor([
            [[self.values[y][x][channel] for x in range(width)] for y in range(height)]
            for channel in range(channels)
        ], dtype=self.dtype)

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]), len(self.values[0][0]))

    def float(self):
        return NumericTensor(self.values, dtype="float32")

    def _map(self, function):
        return NumericTensor([[list(map(function, row)) for row in plane]
                              for plane in self.values], dtype="float32")

    def __truediv__(self, value):
        if isinstance(value, ChannelVector):
            return NumericTensor([
                [[pixel / value.values[channel] for pixel in plane_row]
                 for plane_row in plane]
                for channel, plane in enumerate(self.values)
            ], dtype="float32")
        return self._map(lambda item: item / value)

    def __sub__(self, value):
        if not isinstance(value, ChannelVector):
            return self._map(lambda item: item - value)
        return NumericTensor([
            [[pixel - value.values[channel] for pixel in plane_row]
             for plane_row in plane]
            for channel, plane in enumerate(self.values)
        ], dtype="float32")


class ChannelVector:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, key):
        return self


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.read_count = 0
        self.seeks = []
        self.released = False

    def set(self, prop, source_frame_id):
        self.seeks.append((prop, source_frame_id))
        return True

    def read(self):
        self.read_count += 1
        return True, self.frame

    def release(self):
        self.released = True


class FakeNumpy:
    def __init__(self, *, fail_std=False):
        self.fail_std = fail_std

    def std(self, frame):
        if self.fail_std:
            raise AssertionError("bad frame metadata reached pixel inspection")
        if isinstance(frame, NumericArray):
            values = [channel for row in frame.pixels for pixel in row for channel in pixel]
            average = sum(values) / len(values)
            return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))
        return 2

    def array(self, value):
        return value

    float32 = "float32"
    int32 = "int32"

    def asarray(self, value, dtype=None):
        return {"values": value, "dtype": dtype}


class FakeImageTensor:
    def __getitem__(self, key):
        return self

    def permute(self, *axes):
        return self

    def float(self):
        return self

    def __truediv__(self, value):
        return self

    def __sub__(self, value):
        return self


class FakeTorch:
    def inference_mode(self):
        return contextlib.nullcontext()

    def from_numpy(self, value):
        if isinstance(value, NumericArray):
            return NumericTensor(value.pixels, dtype=value.dtype)
        return FakeImageTensor()

    def tensor(self, value):
        if isinstance(value, (list, tuple)):
            return ChannelVector(value)
        return FakeImageTensor()


class FakeCV:
    CAP_PROP_POS_FRAMES = 1
    COLOR_BGR2RGB = 2

    def __init__(self, frame):
        self.capture = FakeCapture(frame)
        self.converted_codes = []

    def VideoCapture(self, path):
        return self.capture

    def cvtColor(self, frame, code):
        self.converted_codes.append(code)
        if isinstance(frame, NumericArray):
            return NumericArray([[pixel[::-1] for pixel in row] for row in frame.pixels],
                                dtype=frame.dtype)
        return frame


class FakePILImage:
    def __init__(self, image, resize_calls):
        self.image = image
        self.resize_calls = resize_calls

    def resize(self, size):
        self.resize_calls.append(size)
        return self.image


class SourceDecoderProtocolTests(unittest.TestCase):
    def source_session(self, plan, *, frame, fail_std=False):
        manifest = {
            "native_resolution": [32, 24],
            "video": "placeholder.mp4",
        }
        cv = FakeCV(frame)
        session = types.SimpleNamespace(
            np=FakeNumpy(fail_std=fail_std),
            cv2=cv,
            torch=FakeTorch(),
            plan=plan,
            manifest=manifest,
            bundle=Path("synthetic-bundle"),
        )
        return session, cv

    def test_wrong_native_rgb_dimensions_and_type_fail_before_pixels(self):
        document = physical_document()
        plan = producer.prepare_cache_plan(encode(document))
        bad_frames = (
            (PixelFrame(32, 23), "rgb_dimension_mismatch"),
            (PixelFrame(32, 24, dtype="float32"), "rgb_type_mismatch"),
        )
        for frame, reason in bad_frames:
            with self.subTest(reason=reason):
                session, cv = self.source_session(plan, frame=frame, fail_std=True)
                loader = producer._SourceFrames(session, image_size=4)
                loader.allow(0)
                with self.assertRaises(producer.SourceDecodeError) as error:
                    loader[0]
                self.assertEqual(error.exception.reason, reason)
                self.assertEqual(cv.capture.read_count, 1)
                self.assertEqual(loader.snapshot(0)["decode_failure"], reason)
                loader.close()

    def test_legacy_cache_hit_keeps_fifo_eviction_and_limited_length(self):
        raw = (ROOT / ".cache" / "model-quality" / "inputs" / "keyboard" / "input.json").read_bytes()
        plan = producer.prepare_cache_plan(raw, limit=5)
        self.assertEqual(len(plan["planned_keys"]), 241)
        self.assertEqual(len(plan["requested_keys"]), 6)
        session, cv = self.source_session(plan, frame=PixelFrame(32, 24))
        session.manifest["native_resolution"] = [32, 24]

        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(
            fromarray=lambda image: types.SimpleNamespace(resize=lambda size: image)
        )
        with patch.dict(sys.modules, {"PIL": pil}):
            loader = producer._SourceFrames(session, image_size=4)
            self.assertEqual(len(loader), 6)
            for index in range(5):
                loader.allow(index)
                loader[index]
            self.assertEqual(list(loader.cache), [1, 2, 3, 4])
            cached = loader[1]
            self.assertEqual(list(loader.cache), [1, 2, 3, 4])
            self.assertEqual(loader.snapshot(1)["cache_hits"], 1)
            loader.allow(5)
            loader[5]
            self.assertEqual(list(loader.cache), [2, 3, 4, 5])
            self.assertIsNotNone(cached)
            loader.close()

        physical = producer.prepare_cache_plan(encode(physical_document()), limit=1)
        physical_session, _ = self.source_session(physical, frame=PixelFrame(32, 24))
        physical_loader = producer._SourceFrames(physical_session, image_size=4)
        self.assertEqual(len(physical_loader), 3)
        physical_loader.close()

    def test_legacy_rgb_float32_normalization_and_original_source_seeks(self):
        bgr_pixels = [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ]
        frame = NumericArray(bgr_pixels)
        resize_calls = []
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(
            fromarray=lambda image: FakePILImage(image, resize_calls)
        )
        means = (0.485, 0.456, 0.406)
        deviations = (0.229, 0.224, 0.225)
        rgb_pixels = [[pixel[::-1] for pixel in row] for row in bgr_pixels]
        expected = [
            [[(rgb_pixels[y][x][channel] / 255.0 - means[channel]) / deviations[channel]
              for x in range(2)] for y in range(2)]
            for channel in range(3)
        ]

        with patch.dict(sys.modules, {"PIL": pil}):
            for object_name, pin in LegacyPinTests.PINS.items():
                with self.subTest(object_name=object_name):
                    raw = (ROOT / ".cache" / "model-quality" / "inputs" /
                           object_name / "input.json").read_bytes()
                    plan = producer.prepare_cache_plan(raw, limit=2)
                    session, cv = self.source_session(plan, frame=frame)
                    session.manifest["native_resolution"] = [2, 2]
                    loader = producer._SourceFrames(session, image_size=2)
                    tensors = []
                    for index in range(3):
                        loader.allow(index)
                        tensors.append(loader[index])
                    cached = loader[1]
                    self.assertIs(cached, tensors[1])
                    self.assertEqual(loader.snapshot(1)["cache_hits"], 1)
                    self.assertEqual([source for _, source in cv.capture.seeks],
                                     [pin["setup"], pin["first"], pin["first"] + 1])
                    self.assertEqual(cv.converted_codes, [cv.COLOR_BGR2RGB] * 3)
                    self.assertEqual(tensors[0].dtype, "float32")
                    self.assertEqual(tensors[0].shape, (3, 2, 2))
                    for channel in range(3):
                        for y in range(2):
                            for x in range(2):
                                self.assertAlmostEqual(tensors[0].values[channel][y][x],
                                                       expected[channel][y][x], places=7)
                    self.assertEqual(resize_calls[-3:], [(2, 2)] * 3)
                    loader.close()

    def test_zero_valued_decoded_source_is_classified_as_blank_pixels(self):
        plan = producer.prepare_cache_plan(encode(physical_document()))
        frame = NumericArray([[[0, 0, 0] for _ in range(32)] for _ in range(24)])
        session, _ = self.source_session(plan, frame=frame)
        loader = producer._SourceFrames(session, image_size=4)
        loader.allow(0)
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(
            fromarray=lambda image: FakePILImage(image, [])
        )
        with patch.dict(sys.modules, {"PIL": pil}):
            loader[0]
        self.assertIn(0, loader.blank)
        self.assertEqual(loader.snapshot(0)["decode_returned"], 1)
        self.assertIsNone(loader.snapshot(0)["decode_failure"])
        loader.close()


class LegacyRuntimeExceptionTests(unittest.TestCase):
    def test_original_initial_and_middle_decode_errors_are_rethrown(self):
        for fault in ("initial_decode", "middle_decode"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                bundle, _, _ = write_legacy_bundle(Path(folder), "keyboard")
                runtime = ProtocolFakeRuntime(fault=fault)
                with self.assertRaises(producer.SourceDecodeError) as error:
                    producer.run(bundle, Path(folder) / "masks", device="cpu", limit=1,
                                 runtime=runtime)
                self.assertIs(error.exception, runtime.session.failure_exception)


class LegacyPinTests(unittest.TestCase):
    LEGACY_SOURCE_SNAPSHOT = (
        ROOT / ".cache" / "model-quality" / "diagnostics" /
        "mug-chronological-resource-v2-source-snapshot" / "bench" / "quality_sam2.py"
    )
    LEGACY_SOURCE_SHA256 = "C1F8C24C6BB41F57CB2133E15753557AD76F17D9E0D61757C61284C78B9CD736"

    PINS = {
        "keyboard": {
            "sha256": "3CFC13197E096739698B5867383CD34808EB571FFA5A2E387C17E84819D00B80",
            "object": "keyboard",
            "setup": 1148,
            "first": 1149,
            "last": 1388,
            "video": "dd638f8f8de4413bfbbd92e8e21e0451c13efd7fb29b1f5d5e89f12040546a02",
            "asset": "2e20d137a3357a30c978447be48134fecc3c294a9f908bf6f1767e5efee7313e",
        },
        "mug": {
            "sha256": "4422C7373A09564112B0069153AAB40C803148F109EACEAA737BFB0398A268A3",
            "object": "mug",
            "setup": 826,
            "first": 827,
            "last": 1066,
            "video": "ad9e6c4d70adfa62889b08a9f31e0d98a781619f86c5efdce48caef482be3844",
            "asset": "26b2884fa61a9b9870ae089de04ea9bccdf8eb8e1467bdd185e7d514ae259195",
        },
        "ranch": {
            "sha256": "F012F28918534F2A7AC3EFD59841A347A8BAD384D9585130DB88EAAC56377D5D",
            "object": "ranch",
            "setup": 9,
            "first": 10,
            "last": 249,
            "video": "6dcd8c1dda7381d69db178e2b619447506cfcce6cfe4afce8666a43c6363283e",
            "asset": "ed3ac3767202c8da56d30051d766bf7d444d3d356c30671f4bbb2a73e3a1b321",
        },
    }

    def test_exact_three_legacy_raw_pins_and_full_and_limited_plan_ids(self):
        self.assertEqual(set(producer._LEGACY_INPUTS), {pin["sha256"] for pin in self.PINS.values()})
        for object_name, pin in self.PINS.items():
            with self.subTest(object_name=object_name):
                raw = (ROOT / ".cache" / "model-quality" / "inputs" / object_name / "input.json").read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest().upper(), pin["sha256"])
                document = json.loads(raw.decode("utf-8"))
                self.assertEqual(document["object"], pin["object"])
                self.assertEqual(document["setup_frame_id"], pin["setup"])
                self.assertEqual(document["source_hashes"], {
                    "source.mp4": pin["video"],
                    "object.glb": pin["asset"],
                })

                scored_ids = list(range(pin["first"], pin["last"] + 1))
                expected_ids = [pin["setup"], *scored_ids]
                plan = producer.prepare_cache_plan(raw)
                self.assertEqual(plan["clock_mode"], LEGACY)
                self.assertEqual([key.frame_id for key in plan["planned_keys"]], expected_ids)
                self.assertEqual(list(plan["requested_frame_ids"]), expected_ids)
                self.assertEqual(plan["source_frame_ids"], {frame_id: frame_id for frame_id in expected_ids})
                self.assertEqual(plan["expected_setup_frames"], 1)
                self.assertEqual(plan["expected_scored_frames"], 240)

                limited = producer.prepare_cache_plan(raw, limit=2)
                self.assertEqual(list(limited["requested_frame_ids"]), [pin["setup"], *scored_ids[:2]])
                self.assertEqual(len(limited["requested_keys"]), 3)
                self.assertEqual(len(limited["planned_keys"]), 241)

    def test_pinned_legacy_failure_row_keeps_original_visible_reason_and_fields(self):
        original = self.LEGACY_SOURCE_SNAPSHOT.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest().upper(), self.LEGACY_SOURCE_SHA256)
        self.assertIn(b"sam2_visible_object_unavailable", original)
        self.assertIn(b"automatic_detection_reason", original)

        with tempfile.TemporaryDirectory() as folder:
            bundle, _, manifest = write_legacy_bundle(Path(folder), "keyboard")
            output = Path(folder) / "masks"
            runtime = ProtocolFakeRuntime(fault="legacy_detector_error")
            producer.run(bundle, output, device="cpu", limit=1, runtime=runtime)
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["setup_frame_id"], self.PINS["keyboard"]["setup"])
        first, failed = saved["frames"]
        legacy_fields = {
            "frameId", "mask_state", "path", "failure_reason", "timings_ms",
            "automatic_detection_required",
        }
        self.assertEqual(set(first), legacy_fields)
        self.assertEqual(first["frameId"], self.PINS["keyboard"]["setup"])
        self.assertEqual(first["mask_state"], "available")
        self.assertEqual(first["path"], f"masks/{self.PINS['keyboard']['setup']}.png")
        self.assertIsNone(first["failure_reason"])
        self.assertFalse(first["automatic_detection_required"])

        self.assertEqual(set(failed), legacy_fields | {"automatic_detection_reason"})
        self.assertEqual(failed["frameId"], self.PINS["keyboard"]["first"])
        self.assertEqual(failed["mask_state"], "lost")
        self.assertEqual(failed["path"], f"masks/{self.PINS['keyboard']['first']}.png")
        self.assertEqual(failed["failure_reason"], "sam2_visible_object_unavailable")
        self.assertEqual(failed["automatic_detection_reason"], "cnos_runtime_failure")
        self.assertTrue(failed["automatic_detection_required"])
        self.assertEqual(set(failed["timings_ms"]), {"segmentation", "detection"})
        self.assertTrue(all(value >= 0 for value in failed["timings_ms"].values()))


if __name__ == "__main__":
    unittest.main()
