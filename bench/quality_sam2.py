"""Forward-only SAM 2.1 mask production with authenticated capture clocks.

Importing this module is inert: native image, tensor and model packages are
loaded only after ``prepare_cache_plan`` has accepted the complete input.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from numbers import Real


_MAX_INPUT_BYTES = 2 * 1024 * 1024
_MAX_PHYSICAL_SOURCE_PIXELS = 16_777_216
_LEGACY_INPUTS = {
    "3CFC13197E096739698B5867383CD34808EB571FFA5A2E387C17E84819D00B80": {
        "object": "keyboard",
        "source_hashes": {
            "source.mp4": "dd638f8f8de4413bfbbd92e8e21e0451c13efd7fb29b1f5d5e89f12040546a02",
            "object.glb": "2e20d137a3357a30c978447be48134fecc3c294a9f908bf6f1767e5efee7313e",
        },
    },
    "4422C7373A09564112B0069153AAB40C803148F109EACEAA737BFB0398A268A3": {
        "object": "mug",
        "source_hashes": {
            "source.mp4": "ad9e6c4d70adfa62889b08a9f31e0d98a781619f86c5efdce48caef482be3844",
            "object.glb": "26b2884fa61a9b9870ae089de04ea9bccdf8eb8e1467bdd185e7d514ae259195",
        },
    },
    "F012F28918534F2A7AC3EFD59841A347A8BAD384D9585130DB88EAAC56377D5D": {
        "object": "ranch",
        "source_hashes": {
            "source.mp4": "6dcd8c1dda7381d69db178e2b619447506cfcce6cfe4afce8666a43c6363283e",
            "object.glb": "ed3ac3767202c8da56d30051d766bf7d444d3d356c30671f4bbb2a73e3a1b321",
        },
    },
}


class SourceDecodeError(ValueError):
    """A known planned source row could not be decoded as the declared image."""

    def __init__(self, frame_id, source_frame_id, reason="rgb_decode_failed"):
        super().__init__(reason)
        self.frame_id = frame_id
        self.source_frame_id = source_frame_id
        self.reason = reason


class CausalFrameAccessError(ValueError):
    """The SAM video loader requested an unplanned, future or invalid index."""


def deterministic_position_grid(layer, size):
    """Exact closed form of upstream cumsum(ones), avoiding CUDA cumsum."""
    import torch

    h, w = size
    device = layer.positional_encoding_gaussian_matrix.device
    yy = (torch.arange(1, h + 1, device=device, dtype=torch.float32) - .5) / h
    xx = (torch.arange(1, w + 1, device=device, dtype=torch.float32) - .5) / w
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return layer._pe_encoding(torch.stack([x, y], dim=-1)).permute(2, 0, 1)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number is not permitted: {value}")


def _load_manifest(raw_input_bytes):
    if not isinstance(raw_input_bytes, bytes):
        raise ValueError("raw_input_bytes must be bytes")
    if len(raw_input_bytes) > _MAX_INPUT_BYTES:
        raise ValueError("Input manifest exceeds the 2 MiB limit")
    try:
        manifest = json.loads(raw_input_bytes.decode("utf-8", errors="strict"),
                              object_pairs_hook=_strict_object,
                              parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Input manifest must be valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Input manifest must be a JSON object")
    return manifest


def _valid_int(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _valid_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validated_timings_ms(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object of nonnegative finite milliseconds")
    result = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        elapsed = _valid_finite(raw, f"{name}.{key}")
        if elapsed < 0:
            raise ValueError(f"{name}.{key} must be nonnegative")
        result[key] = elapsed
    return result


def _validate_selection(manifest, clock_mode, selection_key, source_frame_id):
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("reviewed") is not True:
        raise ValueError("A reviewed initial object/hand/background selection is required")
    size = manifest.get("native_resolution")
    if (not isinstance(size, list) or len(size) != 2 or
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)):
        raise ValueError("native_resolution must be positive [width, height]")
    width, height = size
    if clock_mode == "capture_seconds_v1" and width * height > _MAX_PHYSICAL_SOURCE_PIXELS:
        raise ValueError("Physical source image exceeds the 16,777,216-pixel preflight ceiling")
    points = selection.get("points")
    labels = selection.get("labels")
    if not isinstance(points, list) or not points or not isinstance(labels, list) or len(points) != len(labels):
        raise ValueError("Selection points and labels must be nonempty lists of equal length")
    positives = 0
    normalized_points = []
    normalized_labels = []
    for point, label in zip(points, labels):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each selection point must be an [x, y] pair")
        x = _valid_finite(point[0], "selection x")
        y = _valid_finite(point[1], "selection y")
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError("Selection points must be inside the declared native image")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError("Selection labels must be binary integers")
        positives += label == 1
        normalized_points.append([x, y])
        normalized_labels.append(label)
    if positives < 1:
        raise ValueError("At least one positive object selection point is required")

    if clock_mode == "capture_seconds_v1":
        if _valid_int(selection.get("frame_id"), "selection frame_id") != selection_key.frame_id:
            raise ValueError("Physical selection must bind to ordinal 0")
        if _valid_int(selection.get("source_frame_id"), "selection source_frame_id") != source_frame_id:
            raise ValueError("Physical selection source ID differs from its planned ordinal")
        stamp = _valid_finite(selection.get("timestamp_s"), "selection timestamp_s")
        if stamp != selection_key.timestamp_s:
            raise ValueError("Physical selection timestamp differs from its planned ordinal")
        for key, value in selection.items():
            lowered = str(key).casefold()
            if ("annotation" in lowered or "image" in lowered or "mask" in lowered or
                    "pose" in lowered):
                raise ValueError("Physical selection cannot reference an unbound image, annotation, mask or pose")
    return {"points": normalized_points, "labels": normalized_labels, "reviewed": True}


def _validate_forced_occlusion(value, clock_mode, plan_length):
    if value is None:
        return None
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or
            any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
        raise ValueError("forced_occlusion must be a pair of integer endpoints")
    start, end = value
    if end <= start:
        raise ValueError("forced_occlusion must be a nonempty half-open interval")
    if clock_mode == "capture_seconds_v1" and not (0 <= start < end <= plan_length):
        raise ValueError("Physical forced_occlusion endpoints must be within planned ordinals")
    return (start, end)


def prepare_cache_plan(raw_input_bytes, *, limit=None, forced_occlusion=None,
                       mask_association=False):
    """Purely authenticate input timing/options and freeze SAM's ordinal plan.

    This function uses only the standard library and the pure ``quality_time``
    parser. It does not read paths, decode pixels, load native libraries, create
    output directories, or initialize model runtime.
    """
    manifest = _load_manifest(raw_input_bytes)
    from .quality_time import LEGACY, PHYSICAL, parse_timing_manifest

    timing = parse_timing_manifest(raw_input_bytes)
    mode = timing["clock_mode"]
    if type(mask_association) is not bool:
        raise ValueError("mask_association must be a bool")
    if mode == PHYSICAL and mask_association:
        raise ValueError("Physical mask association is unsupported while prior age uses frame count")
    if limit is not None:
        _valid_int(limit, "limit", minimum=1)

    if mode == LEGACY:
        pin = _LEGACY_INPUTS.get(timing["manifest_sha256"])
        if pin is None:
            raise ValueError("Untimed SAM input is not one of the three authenticated historical manifests")
        if manifest.get("object") != pin["object"] or manifest.get("source_hashes") != pin["source_hashes"]:
            raise ValueError("Historical SAM input differs from its authenticated object/source hashes")
        setup = timing["setup_key"]
        scored = list(timing["scored_keys"])
        if limit is not None and limit > len(scored):
            raise ValueError("limit exceeds the historical scored-frame count")
        selected_scored = scored if limit is None else scored[:limit]
        planned = ([setup] if setup is not None else []) + scored
        roles = ["setup"] if setup is not None else []
        roles += ["scored"] * len(scored)
        requested_ids = ([setup.frame_id] if setup is not None else []) + [key.frame_id for key in selected_scored]
    else:
        timeline = manifest["timeline"]
        from .quality_time import FrameKey
        planned = [FrameKey(row["frame_id"], row["timestamp_s"], PHYSICAL)
                   for row in timeline]
        roles = [row["role"] for row in timeline]
        setup = timing["setup_key"]
        scored = list(timing["scored_keys"])
        if limit is not None and limit >= len(scored):
            raise ValueError("A finite physical limit must select a strict diagnostic scored prefix; omit it for a full run")
        selected_scored = scored if limit is None else scored[:limit]
        selected_ids = {key.frame_id for key in selected_scored}
        requested_ids = [key.frame_id for key, role in zip(planned, roles)
                         if role == "setup" or key.frame_id in selected_ids]

    if not planned:
        raise ValueError("SAM requires a nonempty planned setup/scored timeline")
    if len(roles) != len(planned):
        raise ValueError("Planned SAM keys and roles differ in length")
    source_ids = dict(timing["source_frame_ids"])
    selection_key = planned[0]
    selection = _validate_selection(manifest, mode, selection_key,
                                    source_ids[selection_key.frame_id])
    occlusion = _validate_forced_occlusion(forced_occlusion, mode, len(planned))

    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("SAM input requires source hashes")
    for name, sha in source_hashes.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute() or
                ".." in Path(name).parts or not isinstance(sha, str) or len(sha) != 64 or
                any(character not in "0123456789abcdefABCDEF" for character in sha)):
            raise ValueError("SAM source hash entries must bind relative paths to SHA-256 digests")
    if mode == PHYSICAL:
        for field in ("video", "asset"):
            relative = manifest.get(field)
            if not isinstance(relative, str) or relative not in source_hashes:
                raise ValueError(f"Physical input must hash its declared {field}")
        if set(source_hashes) != {manifest["video"], manifest["asset"]}:
            raise ValueError("Physical SAM inputs may hash only the source video and metric asset")

    role_by_id = {key.frame_id: role for key, role in zip(planned, roles)}
    requested_set = set(requested_ids)
    return {
        "clock_mode": mode,
        "input_manifest_sha256": timing["manifest_sha256"],
        "timestamp_table_sha256": timing["timestamp_table_sha256"],
        "planned_keys": tuple(planned),
        "source_frame_ids": source_ids,
        "roles": tuple(roles),
        "role_by_frame_id": role_by_id,
        "requested_keys": tuple(key for key in planned if key.frame_id in requested_set),
        "requested_frame_ids": tuple(requested_ids),
        "selection_key": selection_key,
        "selection": selection,
        "expected_setup_frames": sum(role == "setup" for role in roles),
        "expected_scored_frames": sum(role == "scored" for role in roles),
        "requested_scored_frames": len(selected_scored),
        "diagnostic_prefix": mode == PHYSICAL and limit is not None,
        "forced_occlusion": occlusion,
        "mask_association": mask_association,
        "effective_options": {
            "limit": limit,
            "forced_occlusion": occlusion,
            "mask_association": mask_association,
        },
        "source_hashes": dict(source_hashes),
    }


def _read_input(bundle):
    path = Path(bundle) / "input.json"
    with path.open("rb") as stream:
        raw = stream.read(_MAX_INPUT_BYTES + 1)
    if len(raw) > _MAX_INPUT_BYTES:
        raise ValueError("Input manifest exceeds the 2 MiB limit")
    return raw


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _unmeasured_row(key, source_frame_id, clock_mode, role, reason):
    return {
        "frameId": key.frame_id,
        "sourceFrameId": source_frame_id,
        "timestamp_s": key.timestamp_s,
        "clock_mode": clock_mode,
        "role": role,
        "source_state": "not_read",
        "mask_state": "lost",
        "measurement_state": "unmeasured",
        "path": None,
        "mask_sha256": None,
        "failure_reason": reason,
        "source_failure_reason": None,
        "detector_reason": None,
        "timings_ms": {},
        "operation_counts": {"attempted": {}, "returned": {}},
        "automatic_conditioning": False,
    }


def _legacy_frame_row(key, source_frame_id, role, state, path, reason, timings,
                      automatic_detection_required, detection_reason=None,
                      detection_attempted=False):
    row = {
        "frameId": source_frame_id,
        "mask_state": state,
        "path": path,
        "failure_reason": reason,
        "timings_ms": timings,
        "automatic_detection_required": automatic_detection_required,
    }
    if detection_attempted:
        row["automatic_detection_reason"] = detection_reason
    return row


def _move_tensors(value, torch, device, memo=None, transport=None):
    """Move every tensor reachable from a SAM state container, preserving aliases."""
    if memo is None:
        memo = {}
    if transport is None:
        transport = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        original_device = str(value.device)
        moved = value.detach().to(device)
        memo[identity] = moved
        record = {"tensor": moved, "original_device": original_device}
        transport[identity] = record
        transport[id(moved)] = record
        return moved
    if isinstance(value, dict):
        memo[identity] = value
        for key in tuple(value):
            value[key] = _move_tensors(value[key], torch, device, memo, transport)
        return value
    if isinstance(value, list):
        memo[identity] = value
        for index, item in enumerate(value):
            value[index] = _move_tensors(item, torch, device, memo, transport)
        return value
    if isinstance(value, tuple):
        moved = tuple(_move_tensors(item, torch, device, memo, transport) for item in value)
        memo[identity] = moved
        return moved
    return value


def _restore_tensors(value, transport, torch, memo=None, restored_devices=None):
    """Restore each transported state tensor to its own pre-detector device."""
    if memo is None:
        memo = {}
    if restored_devices is None:
        restored_devices = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        target = transport.get(identity)
        if target is None:
            raise RuntimeError("SAM state acquired an unaudited tensor during detector offload")
        restored = value.to(target["original_device"])
        memo[identity] = restored
        restored_devices[id(restored)] = target["original_device"]
        return restored
    if isinstance(value, dict):
        memo[identity] = value
        for key in tuple(value):
            value[key] = _restore_tensors(value[key], transport, torch, memo, restored_devices)
        return value
    if isinstance(value, list):
        memo[identity] = value
        for index, item in enumerate(value):
            value[index] = _restore_tensors(item, transport, torch, memo, restored_devices)
        return value
    if isinstance(value, tuple):
        restored = tuple(_restore_tensors(item, transport, torch, memo, restored_devices) for item in value)
        memo[identity] = restored
        return restored
    return value


def _restored_device_audit(value, restored_devices, torch):
    seen = set()
    counts = {}
    mismatches = 0

    def visit(item):
        nonlocal mismatches
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            expected = restored_devices.get(identity)
            current = str(item.device)
            counts[current] = counts.get(current, 0) + 1
            if expected is None or current != expected:
                mismatches += 1
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple, set)):
            for child in item:
                visit(child)

    visit(value)
    return {"tensor_device_counts": counts, "device_mismatches": mismatches}


def _tensor_audit(value, torch, seen=None):
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"tensor_count": 0, "non_cpu_tensor_count": 0, "bytes": 0}
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return {
            "tensor_count": 1,
            "non_cpu_tensor_count": int(value.device.type != "cpu"),
            "bytes": value.numel() * value.element_size(),
        }
    totals = {"tensor_count": 0, "non_cpu_tensor_count": 0, "bytes": 0}
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple, set)):
        children = value
    else:
        return totals
    for child in children:
        row = _tensor_audit(child, torch, seen)
        for key in totals:
            totals[key] += row[key]
    return totals


class _SourceFrames:
    """Four-frame causal LRU used by the pinned SAM loader replacement."""

    def __init__(self, session, image_size):
        from collections import OrderedDict

        self.session = session
        self.np = session.np
        self.cv2 = session.cv2
        self.torch = session.torch
        self.image_size = image_size
        self.width, self.height = session.manifest["native_resolution"]
        self.cap = self.cv2.VideoCapture(str(session.bundle / session.manifest["video"]))
        self.cache = OrderedDict()
        self.blank = set()
        self.max_index = 0
        self.metrics = {}
        self.max_cache_frames = 4

    def __len__(self):
        if self.session.plan["clock_mode"] == "capture_seconds_v1":
            return len(self.session.plan["planned_keys"])
        return len(self.session.plan["requested_keys"])

    def allow(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or index < self.max_index:
            raise CausalFrameAccessError("SAM source allowance cannot move backward")
        if index >= len(self):
            raise CausalFrameAccessError("SAM requested an index outside the frozen plan")
        if index > self.max_index + 1:
            raise CausalFrameAccessError("SAM source allowance must advance one planned ordinal at a time")
        self.max_index = index

    def _stats(self, index):
        return self.metrics.setdefault(index, {
            "decoder_requests": 0,
            "decode_attempted": 0,
            "decode_returned": 0,
            "cache_hits": 0,
            "past_reads": 0,
            "decode_ms": 0.0,
            "decode_failure": None,
        })

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise CausalFrameAccessError("SAM requested an invalid source index")
        if index > self.max_index:
            raise CausalFrameAccessError("SAM attempted to read a future source frame")
        stats = self._stats(index)
        stats["decoder_requests"] += 1
        if index < self.max_index:
            stats["past_reads"] += 1
        if index in self.cache:
            stats["cache_hits"] += 1
            if self.session.plan["clock_mode"] == "capture_seconds_v1":
                self.cache.move_to_end(index)
            return self.cache[index]

        key = self.session.plan["planned_keys"][index]
        source_frame_id = self.session.plan["source_frame_ids"][key.frame_id]
        started = time.perf_counter()
        stats["decode_attempted"] += 1
        try:
            seek_ok = self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, source_frame_id)
            if self.session.plan["clock_mode"] == "capture_seconds_v1" and not seek_ok:
                raise SourceDecodeError(key.frame_id, source_frame_id, "rgb_decode_failed")
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                raise SourceDecodeError(key.frame_id, source_frame_id)
            if getattr(bgr, "ndim", None) != 3 or tuple(bgr.shape) != (self.height, self.width, 3):
                raise SourceDecodeError(key.frame_id, source_frame_id, "rgb_dimension_mismatch")
            if str(getattr(bgr, "dtype", "")) != "uint8":
                raise SourceDecodeError(key.frame_id, source_frame_id, "rgb_type_mismatch")
            forced = self.session.plan["forced_occlusion"]
            if forced is not None:
                index_for_range = index if self.session.plan["clock_mode"] == "capture_seconds_v1" else source_frame_id
                if forced[0] <= index_for_range < forced[1]:
                    bgr = bgr.copy()
                    bgr[:] = 0
            if self.np.std(bgr) < 1:
                self.blank.add(index)
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            from PIL import Image
            resized = self.np.array(Image.fromarray(rgb).resize((self.image_size, self.image_size)))
            tensor = self.torch.from_numpy(resized).permute(2, 0, 1).float() / 255
            tensor = (tensor - self.torch.tensor([.485, .456, .406])[:, None, None]) / self.torch.tensor(
                [.229, .224, .225])[:, None, None]
            self.cache[index] = tensor
            stats["decode_returned"] += 1
            while len(self.cache) > self.max_cache_frames:
                self.cache.popitem(last=False)
            return tensor
        except SourceDecodeError as exc:
            stats["decode_failure"] = exc.reason
            raise
        finally:
            stats["decode_ms"] += (time.perf_counter() - started) * 1000.0

    def snapshot(self, index):
        return dict(self.metrics.get(index, {}))

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cache.clear()


class _RealSession:
    """Pinned SAM2/CNOS runtime, created only after the complete pure preflight."""

    def __init__(self, plan, manifest, bundle, output, device):
        import numpy as np
        import cv2
        import torch

        self.np = np
        self.cv2 = cv2
        self.torch = torch
        self.plan = plan
        self.manifest = manifest
        self.bundle = Path(bundle)
        self.output = Path(output)
        self.device = device
        from .quality_assets import CACHE, MODELS, digest, inference_provenance

        self.cache_root = CACHE
        self.checkpoint = CACHE / "checkpoints/sam2.1_hiera_base_plus.pt"
        if digest(self.checkpoint).lower() != MODELS[self.checkpoint.name]["sha256"].lower():
            raise ValueError("SAM 2 checkpoint hash mismatch")
        self.checkpoint_sha256 = digest(self.checkpoint).upper()
        self.provenance = inference_provenance(self.bundle)
        source = CACHE / "sources/sam2"
        sys.path.insert(0, str(source))
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.modeling.position_encoding import PositionEmbeddingRandom
        import sam2.sam2_video_predictor as predictor_module

        PositionEmbeddingRandom.forward = deterministic_position_grid
        self.predictor_module = predictor_module
        self.predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_b+.yaml", str(self.checkpoint), device=device,
            apply_postprocessing=False, vos_optimized=False)
        self.loader = None
        self.state = None
        self.iterator = None
        self.original_loader = None
        self._hook_restored = True
        self._closed = False
        self._state_device_transport = None
        self._restored_devices = None
        self.initialization_ms = 0.0
        self.selection_prompt_ms = 0.0
        self.selection_prompt_attempted = False
        self.selection_prompt_returned = False
        self.model_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.position_grid = "exact arange form of upstream cumsum(ones)"
        self.runtime_attached = True

    def _load_original(self, video_path, image_size, **kwargs):
        self.loader = _SourceFrames(self, image_size)
        return self.loader, self.loader.height, self.loader.width

    def initialize(self):
        initialize_started = time.perf_counter()
        self.original_loader = self.predictor_module.load_video_frames
        self.predictor_module.load_video_frames = self._load_original
        self._hook_restored = False
        try:
            with self.torch.inference_mode():
                self.loader = None
                self.state = self.predictor.init_state(
                    "original-source", offload_video_to_cpu=True, offload_state_to_cpu=True,
                    async_loading_frames=False)
        finally:
            self.predictor_module.load_video_frames = self.original_loader
            self._hook_restored = True
            self.initialization_ms = (time.perf_counter() - initialize_started) * 1000.0

    def add_initial_selection(self, selection):
        selection_started = time.perf_counter()
        points = self.np.asarray(selection["points"], dtype=self.np.float32)
        labels = self.np.asarray(selection["labels"], dtype=self.np.int32)
        try:
            self.selection_prompt_attempted = True
            with self.torch.inference_mode():
                self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=1,
                                                     points=points, labels=labels)
            self.selection_prompt_returned = True
            with self.torch.inference_mode():
                self.iterator = iter(self.predictor.propagate_in_video(
                    self.state, start_frame_idx=0, reverse=False))
        finally:
            self.selection_prompt_ms = (time.perf_counter() - selection_started) * 1000.0

    def step(self, index):
        if self.loader is None or self.iterator is None:
            raise RuntimeError("SAM session was not initialized")
        self.loader.allow(index)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                frame_index, object_ids, logits = next(self.iterator)
        except StopIteration as exc:
            raise RuntimeError("SAM iterator ended before the planned ordinal") from exc
        finally:
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        try:
            if frame_index != index or object_ids != [1]:
                raise RuntimeError("Nonchronological SAM frame/object output")
            mask = (logits[0, 0].detach().cpu().numpy() > 0).astype(self.np.uint8) * 255
        finally:
            del logits
        height = self.loader.height
        width = self.loader.width
        if tuple(mask.shape) != (height, width):
            raise RuntimeError("SAM mask dimensions differ from the declared native image")
        return {
            "frame_idx": frame_index,
            "obj_ids": list(object_ids),
            "mask": mask,
            "blank": index in self.loader.blank,
            "timings_ms": {"segmentation": elapsed},
            "source": self.loader.snapshot(index),
        }

    def mask_info(self, mask):
        array = self.np.asarray(mask)
        if array.ndim != 2 or str(array.dtype) != "uint8":
            raise ValueError("Masks must be native two-dimensional uint8 arrays")
        if array.shape != (self.loader.height, self.loader.width):
            raise ValueError("Mask dimensions differ from the native image")
        if not self.np.isin(array, (0, 255)).all():
            raise ValueError("Mask pixels must be binary 0/255 values")
        return {
            "shape": tuple(int(value) for value in array.shape),
            "dtype": "uint8",
            "binary_0_255": True,
            "foreground_pixels": int(self.np.count_nonzero(array)),
        }

    def empty_mask(self):
        return self.np.zeros((self.loader.height, self.loader.width), dtype=self.np.uint8)

    def write_mask(self, path, mask):
        path = Path(path)
        if not self.cv2.imwrite(str(path), mask):
            raise OSError(f"OpenCV could not write mask artifact: {path}")
        return {"sha256": _sha256_path(path), "bytes": path.stat().st_size}

    def _model_device_audit(self, expected):
        seen = 0
        wrong = 0
        for tensor in list(self.predictor.parameters()) + list(self.predictor.buffers()):
            seen += 1
            wrong += tensor.device.type != expected
        return {"model_tensor_count": seen, "model_tensors_on_wrong_device": int(wrong)}

    def offload_for_detector(self):
        if self.iterator is not None:
            close = getattr(self.iterator, "close", None)
            if callable(close):
                close()
            self.iterator = None
        if self.state is None or self.predictor is None:
            raise RuntimeError("SAM state is unavailable for detector offload")
        cached = self.state.get("cached_features")
        if isinstance(cached, dict):
            cached.clear()
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        self.predictor.to("cpu")
        state_audit = _tensor_audit(self.state, self.torch)
        transport = {}
        _move_tensors(self.state, self.torch, "cpu", transport=transport)
        self._state_device_transport = transport
        if self.device == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()
            live_gpu_bytes = int(self.torch.cuda.memory_allocated())
        else:
            live_gpu_bytes = 0
        state_after = _tensor_audit(self.state, self.torch)
        model_after = self._model_device_audit("cpu")
        if (state_after["non_cpu_tensor_count"] or model_after["model_tensors_on_wrong_device"] or
                live_gpu_bytes != 0):
            raise RuntimeError("SAM CPU offload audit failed; refusing to start the detector child")
        return {
            "state_before": state_audit,
            "state_after": state_after,
            "model_after": model_after,
            "live_gpu_bytes_after_empty_cache": live_gpu_bytes,
            "state_original_device_counts": {
                device: sum(entry["original_device"] == device
                            for entry in {id(item): item for item in transport.values()}.values())
                for device in sorted({entry["original_device"] for entry in transport.values()})
            },
            "cpu_transport": "recursive state tensor .to(cpu); model parameters/buffers CPU audited",
        }

    def run_detector(self, index, key, *, prior_frame_id=None):
        source_id = self.plan["source_frame_ids"][key.frame_id]
        frame_arg = key.frame_id if self.plan["clock_mode"] == "capture_seconds_v1" else source_id
        detection_path = self.output / f"detection-{frame_arg}.npz"
        command = [sys.executable, "-B", "-m", "bench.quality_runner", "detect",
                   "--bundle", str(self.bundle), "--frame", str(frame_arg),
                   "--output", str(detection_path), "--device", self.device]
        if self.plan["clock_mode"] != "capture_seconds_v1" and self.plan["mask_association"] and prior_frame_id is not None:
            prior_name = self.output / "masks" / f"{prior_frame_id}.png"
            command += ["--detection-prior", str(prior_name), "--prior-frame", str(prior_frame_id)]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed = (time.perf_counter() - started) * 1000.0
        if completed.returncode != 0 or not detection_path.is_file():
            error_path = self.output / f"detection-{frame_arg}-error.txt"
            error_path.write_text((completed.stderr or "detector did not create its result")[-8000:],
                                  encoding="utf-8")
            if self.plan["clock_mode"] == "capture_seconds_v1":
                return {
                    "mask": None,
                    "reason": "cnos_runtime_failure" if completed.returncode else "cnos_result_missing",
                    "timings_ms": {"detection": elapsed},
                    "process_returncode": completed.returncode,
                    "process_reaped": True,
                    "artifact_sha256": None,
                    "error_path": error_path.name,
                }
            return {
                "mask": None, "reason": "cnos_runtime_failure", "timings_ms": {"detection": elapsed},
                "process_returncode": completed.returncode, "process_reaped": True,
                "artifact_sha256": None,
            }
        with self.np.load(detection_path, allow_pickle=False) as archive:
            if "mask" not in archive:
                raise ValueError("Detector NPZ is missing its mask")
            mask = archive["mask"].copy()
            reason = self._archive_text(archive, "reason")
            diagnostics = self._archive_text(archive, "diagnostics")
            identity = None
            if self.plan["clock_mode"] == "capture_seconds_v1":
                identity = {
                    "frameId": self._archive_scalar(archive, "frameId"),
                    "sourceFrameId": self._archive_scalar(archive, "sourceFrameId"),
                    "timestamp_s": self._archive_scalar(archive, "timestamp_s"),
                    "clock_mode": self._archive_text(archive, "clock_mode"),
                    "input_manifest_sha256": self._archive_text(archive, "input_manifest_sha256"),
                    "timestamp_table_sha256": self._archive_text(archive, "timestamp_table_sha256"),
                }
        return {
            "mask": mask,
            "reason": reason or None,
            "diagnostics": diagnostics,
            "identity": identity,
            "timings_ms": {"detection": elapsed},
            "process_returncode": completed.returncode,
            "process_reaped": True,
            "artifact_sha256": _sha256_path(detection_path),
            "artifact_path": detection_path.name,
        }

    @staticmethod
    def _archive_scalar(archive, key):
        if key not in archive:
            raise ValueError(f"Detector NPZ is missing authenticated field {key}")
        value = archive[key]
        if value.shape != ():
            raise ValueError(f"Detector NPZ field {key} must be a scalar")
        return value.item()

    @classmethod
    def _archive_text(cls, archive, key):
        value = cls._archive_scalar(archive, key)
        if not isinstance(value, str):
            raise ValueError(f"Detector NPZ field {key} must be text")
        return value

    def restore_after_detector(self):
        self.predictor.to(self.device)
        model_after = self._model_device_audit("cuda" if self.device == "cuda" else "cpu")
        if model_after["model_tensors_on_wrong_device"]:
            raise RuntimeError("SAM restore audit failed after detector child exit")
        if self._state_device_transport is None:
            raise RuntimeError("SAM state transport receipt is unavailable for restoration")
        restored_devices = {}
        _restore_tensors(self.state, self._state_device_transport, self.torch,
                         restored_devices=restored_devices)
        state_after = _restored_device_audit(self.state, restored_devices, self.torch)
        if state_after["device_mismatches"]:
            raise RuntimeError("SAM retained state did not return to its original per-tensor devices")
        self._restored_devices = restored_devices
        self._state_device_transport = None
        return {"model": model_after, "state": state_after}

    def resume_after_detector(self, index):
        """Continue the same causal SAM state after an authenticated no-detection."""
        if index + 1 < len(self.loader):
            self.iterator = iter(self.predictor.propagate_in_video(
                self.state, start_frame_idx=index + 1, reverse=False))

    def condition(self, index, mask):
        with self.torch.inference_mode():
            self.predictor.add_new_mask(self.state, frame_idx=index, obj_id=1, mask=mask > 0)
        if index + 1 < len(self.loader):
            self.iterator = iter(self.predictor.propagate_in_video(
                self.state, start_frame_idx=index + 1, reverse=False))

    def snapshot_source(self, index):
        if self.loader is None:
            return {}
        return self.loader.snapshot(index)

    def close(self):
        status = {"iterator_released": True, "decoder_released": True,
                  "predictor_released": True, "loader_hook_restored": self._hook_restored}
        try:
            if self.iterator is not None:
                close = getattr(self.iterator, "close", None)
                if callable(close):
                    close()
                self.iterator = None
        except Exception:
            status["iterator_released"] = False
        try:
            if self.loader is not None:
                self.loader.close()
                self.loader = None
        except Exception:
            status["decoder_released"] = False
        try:
            if self._hook_restored and self.original_loader is not None:
                self.predictor_module.load_video_frames = self.original_loader
            self.predictor = None
            self.state = None
            if self._state_device_transport is not None:
                self._state_device_transport.clear()
            self._state_device_transport = None
            if self._restored_devices is not None:
                self._restored_devices.clear()
            self._restored_devices = None
        except Exception:
            status["predictor_released"] = False
        self._closed = True
        return status


class _RealRuntime:
    def open_session(self, *, plan, manifest, bundle, output, device):
        return _RealSession(plan, manifest, bundle, output, device)


def _validate_physical_detector_payload(payload, key, plan, session):
    if not isinstance(payload, dict):
        raise ValueError("Detector runtime must return a bound result object")
    expected = {
        "frameId": key.frame_id,
        "sourceFrameId": plan["source_frame_ids"][key.frame_id],
        "timestamp_s": key.timestamp_s,
        "clock_mode": "capture_seconds_v1",
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "timestamp_table_sha256": plan["timestamp_table_sha256"],
    }
    identity = payload.get("identity")
    if not isinstance(identity, dict) or set(identity) != set(expected):
        raise ValueError("Physical detector result lacks its authenticated six-field identity")
    for field, wanted in expected.items():
        value = identity.get(field)
        if field in ("frameId", "sourceFrameId"):
            if isinstance(value, bool) or not isinstance(value, int) or value != wanted:
                raise ValueError(f"Physical detector {field} differs from the current source row")
        elif field == "timestamp_s":
            if isinstance(value, bool) or not isinstance(value, Real) or float(value) != wanted:
                raise ValueError("Physical detector timestamp differs from the current source row")
        elif value != wanted:
            raise ValueError(f"Physical detector {field} differs from the current input")
    returncode = payload.get("process_returncode")
    if (isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0 or
            payload.get("process_reaped") is not True):
        raise ValueError("Physical detector child did not complete and reap successfully")
    mask = payload.get("mask")
    info = session.mask_info(mask)
    if info.get("dtype") != "uint8" or info.get("binary_0_255") is not True:
        raise ValueError("Physical detector mask must be native binary uint8 0/255")
    return mask, info


def _validate_offload_receipt(receipt):
    if not isinstance(receipt, dict):
        raise RuntimeError("SAM offload must return an auditable residency receipt")
    state = receipt.get("state_after")
    model = receipt.get("model_after")
    gpu_bytes = receipt.get("live_gpu_bytes_after_empty_cache")
    state_non_cpu = state.get("non_cpu_tensor_count") if isinstance(state, dict) else None
    model_wrong = model.get("model_tensors_on_wrong_device") if isinstance(model, dict) else None
    if (isinstance(state_non_cpu, bool) or not isinstance(state_non_cpu, int) or state_non_cpu != 0 or
            isinstance(model_wrong, bool) or not isinstance(model_wrong, int) or model_wrong != 0 or
            isinstance(gpu_bytes, bool) or not isinstance(gpu_bytes, int) or gpu_bytes != 0):
        raise RuntimeError("SAM residency receipt does not prove CPU-only child staging")
    return receipt


def _validate_restore_receipt(receipt):
    if not isinstance(receipt, dict):
        raise RuntimeError("SAM restore must return an auditable residency receipt")
    model = receipt.get("model")
    state = receipt.get("state")
    model_wrong = model.get("model_tensors_on_wrong_device") if isinstance(model, dict) else None
    state_mismatches = state.get("device_mismatches") if isinstance(state, dict) else None
    if (isinstance(model_wrong, bool) or not isinstance(model_wrong, int) or model_wrong != 0 or
            isinstance(state_mismatches, bool) or not isinstance(state_mismatches, int) or
            state_mismatches != 0):
        raise RuntimeError("SAM return-path receipt does not prove its original device map")
    return receipt


def _source_counts(source):
    source = source if isinstance(source, dict) else {}
    def count(name):
        value = source.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Source metric {name} must be a nonnegative integer")
        return value
    attempted = {
        "decode": count("decode_attempted"),
        "decoder_request": count("decoder_requests"),
    }
    returned = {
        "decode": count("decode_returned"),
        "cache_hit": count("cache_hits"),
        "past_read": count("past_reads"),
    }
    return attempted, returned


def _physical_row_from_result(plan, index, step, mask, mask_info, artifact,
                              *, usable, reason, detector_reason, detector_sha,
                              source, automatic_conditioning, residency=None,
                              operations=None, timings=None):
    key = plan["planned_keys"][index]
    row = _unmeasured_row(key, plan["source_frame_ids"][key.frame_id],
                          plan["clock_mode"], plan["roles"][index], reason)
    row["source_state"] = ("decoded" if source.get("decode_returned", 0) or
                            source.get("cache_hits", 0) else "unverified")
    row["mask_state"] = "available" if usable else "lost"
    row["measurement_state"] = "measured"
    row["path"] = artifact["path"]
    row["mask_sha256"] = artifact["sha256"]
    row["failure_reason"] = None if usable else reason
    row["source_failure_reason"] = None
    row["detector_reason"] = detector_reason
    row["timings_ms"] = _validated_timings_ms(step.get("timings_ms"), "SAM timings_ms")
    row["timings_ms"].update(_validated_timings_ms(
        step.get("detector_timings_ms"), "detector timings_ms"))
    row["timings_ms"].update(_validated_timings_ms(timings, "row timings_ms"))
    if source.get("decode_ms") is not None:
        decode_elapsed = _valid_finite(source["decode_ms"], "decode timing")
        if decode_elapsed < 0:
            raise ValueError("Decode timing must be nonnegative")
        row["timings_ms"]["decode"] = decode_elapsed
    attempted, returned = _source_counts(source)
    attempted["sam_step"] = 1
    returned["sam_step"] = 1
    attempted.update({"detector": int(bool(step.get("detector_attempted"))),
                      "sam_offload": int(bool(step.get("detector_attempted"))),
                      "detector_restore": int(bool(step.get("detector_returned"))),
                      "automatic_conditioning": int(bool(automatic_conditioning)),
                      "mask_write": 1})
    returned.update({"detector": int(bool(step.get("detector_returned"))),
                     "sam_offload": int(bool(step.get("detector_attempted"))),
                     "detector_restore": int(bool(step.get("detector_returned"))),
                     "automatic_conditioning": int(bool(automatic_conditioning)),
                     "mask_write": 1})
    if index == 0:
        attempted.update({
            "initialization": 1,
            "initial_selection": 1,
        })
        returned.update({
            "initialization": 1,
            "initial_selection": 1,
        })
    if operations:
        attempted.update(operations.get("attempted", {}))
        returned.update(operations.get("returned", {}))
    row["operation_counts"] = {"attempted": attempted, "returned": returned}
    row["automatic_conditioning"] = bool(automatic_conditioning)
    if detector_sha is not None:
        row["detector_artifact_sha256"] = detector_sha
    if residency is not None:
        row["detector_residency"] = residency
    row["foreground_pixels"] = int(mask_info["foreground_pixels"])
    return row


def _validate_mask_info(session, mask, width, height):
    info = session.mask_info(mask)
    if not isinstance(info, dict):
        raise ValueError("Runtime mask_info must return a metadata object")
    if tuple(info.get("shape", ())) != (height, width):
        raise ValueError("Mask dimensions differ from the declared native image")
    if info.get("dtype") != "uint8" or info.get("binary_0_255") is not True:
        raise ValueError("Mask must contain native binary uint8 0/255 values")
    pixels = info.get("foreground_pixels")
    if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 0 or pixels > width * height:
        raise ValueError("Runtime foreground pixel count is invalid")
    return info


def _validate_runtime_manifest(manifest, bundle, plan, *, verify_resources):
    if manifest.get("units") != "metres":
        raise ValueError("Metric object asset required")
    width, height = manifest["native_resolution"]
    intrinsics = manifest.get("intrinsics")
    if not isinstance(intrinsics, list) or len(intrinsics) != 3 or any(
            not isinstance(row, list) or len(row) != 3 for row in intrinsics):
        raise ValueError("SAM input requires a 3x3 camera calibration")
    for row in intrinsics:
        for value in row:
            _valid_finite(value, "intrinsics value")
    root = Path(bundle).resolve()
    for name, expected in plan["source_hashes"].items():
        resource = (root / name).resolve()
        if not resource.is_relative_to(root) or not resource.is_file():
            raise ValueError("SAM source path is missing or escapes its input bundle")
        if verify_resources and _sha256_path(resource).lower() != expected.lower():
            raise ValueError(f"SAM source hash mismatch: {name}")
    if not isinstance(manifest.get("video"), str) or not isinstance(manifest.get("asset"), str):
        raise ValueError("SAM input must identify its source video and metric asset")
    if width <= 0 or height <= 0:
        raise ValueError("SAM native image dimensions must be positive")


def _runtime_manifest_view(manifest):
    """Keep annotations/query labels out of every inference runtime object."""
    allowed = ("object", "video", "asset", "native_resolution", "intrinsics",
               "source_hashes", "units")
    return {key: manifest[key] for key in allowed if key in manifest}


def _default_provenance(plan):
    return {
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "adapter_sha256": {"quality_sam2": _sha256_path(Path(__file__))},
    }


def _initial_physical_record(plan, manifest, device, provenance):
    rows = []
    requested = set(plan["requested_frame_ids"])
    for index, key in enumerate(plan["planned_keys"]):
        reason = "stage_not_started" if key.frame_id in requested else "outside_requested_prefix"
        rows.append(_unmeasured_row(key, plan["source_frame_ids"][key.frame_id],
                                    plan["clock_mode"], plan["roles"][index], reason))
    return {
        "model": "SAM 2.1 Hiera Base Plus",
        "model_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint_sha256": None,
        "device": device,
        "propagation": "forward only; one ordinal-0 initial selection; no query-frame prompts",
        "position_grid": "exact arange form of upstream cumsum(ones)",
        "provenance": provenance,
        "schema_version": 2,
        "clock_mode": plan["clock_mode"],
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "timestamp_table_sha256": plan["timestamp_table_sha256"],
        "source_sha256": dict(plan["source_hashes"]),
        "status": "running",
        "coverage_complete": True,
        "stage_completed": False,
        "expected_setup_frames": plan["expected_setup_frames"],
        "expected_scored_frames": plan["expected_scored_frames"],
        "requested_frame_ids": list(plan["requested_frame_ids"]),
        "planned_frame_ids": [key.frame_id for key in plan["planned_keys"]],
        "frames": rows,
        "operation_counts": {"attempted": {}, "returned": {}},
        "cleanup": {"status": "pending"},
        "diagnostic_control": plan["forced_occlusion"] is not None,
        "mask_association": plan["mask_association"],
        "options": dict(plan["effective_options"]),
        "prompt_ledger": {
            "initial_selection": {
                "frame_id": plan["selection_key"].frame_id,
                "source_frame_id": plan["source_frame_ids"][plan["selection_key"].frame_id],
                "timestamp_s": plan["selection_key"].timestamp_s,
                "reviewed": True,
                "points": plan["selection"]["points"],
                "labels": plan["selection"]["labels"],
                "attempted": 0,
                "returned": 0,
            },
            "automatic_mask_conditioning": [],
        },
    }


def _increment(mapping, key, amount=1):
    mapping[key] = mapping.get(key, 0) + amount


def _mark_terminal_rows(record, plan, index, reason, source=None, *, operations=None,
                        timings=None, residency=None, detector_reason=None,
                        detector_sha=None, automatic_conditioning=False):
    requested = set(plan["requested_frame_ids"])
    if index is not None and 0 <= index < len(record["frames"]):
        key = plan["planned_keys"][index]
        row = record["frames"][index]
        row["source_state"] = "decode_failed" if isinstance(reason, str) and reason in {
            "rgb_decode_failed", "rgb_dimension_mismatch", "rgb_type_mismatch"} else row["source_state"]
        if source and (source.get("decode_returned", 0) or source.get("cache_hits", 0)):
            row["source_state"] = "decoded"
        row["failure_reason"] = reason
        row["source_failure_reason"] = reason if row["source_state"] == "decode_failed" else None
        row["detector_reason"] = detector_reason
        row["automatic_conditioning"] = bool(automatic_conditioning)
        if detector_sha is not None:
            row["detector_artifact_sha256"] = detector_sha
        row["measurement_state"] = "unmeasured"
        row["mask_state"] = "lost"
        row["path"] = None
        row["mask_sha256"] = None
        if source:
            row["timings_ms"].update({"decode": float(source.get("decode_ms", 0.0))})
            row["operation_counts"]["attempted"].update({
                "decode": int(source.get("decode_attempted", 0)),
                "decoder_request": int(source.get("decoder_requests", 0)),
            })
            row["operation_counts"]["returned"].update({
                "decode": int(source.get("decode_returned", 0)),
                "cache_hit": int(source.get("cache_hits", 0)),
                "past_read": int(source.get("past_reads", 0)),
            })
        if operations:
            for direction in ("attempted", "returned"):
                row["operation_counts"][direction].update(operations.get(direction, {}))
        if timings:
            row["timings_ms"].update(timings)
        if residency is not None:
            row["detector_residency"] = residency
        start = index + 1
    else:
        start = 0
    for remaining in range(start, len(record["frames"])):
        row = record["frames"][remaining]
        if row["frameId"] not in requested or row["measurement_state"] == "measured":
            continue
        row["source_state"] = "not_read"
        row["mask_state"] = "lost"
        row["measurement_state"] = "unmeasured"
        row["path"] = None
        row["mask_sha256"] = None
        row["failure_reason"] = "segmentation_not_run_after_terminal_failure"


def _physical_receipt_write(output, record):
    _write_json_atomic(Path(output) / "results.json", record)


def _run_plan(plan, manifest, bundle, output, device, runtime):
    physical = plan["clock_mode"] == "capture_seconds_v1"
    output = Path(output)
    if runtime is None:
        runtime = _RealRuntime()
    opener = getattr(runtime, "open_session", None)
    if not callable(opener):
        raise TypeError("Injected SAM runtime must provide open_session(**plan_context)")
    if physical:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / "masks"
    masks_dir.mkdir(exist_ok=not physical)

    session = None
    if physical:
        provisional_provenance = _default_provenance(plan)
        record = _initial_physical_record(plan, manifest, device, provisional_provenance)
        _physical_receipt_write(output, record)
        requested_set = set(plan["requested_frame_ids"])
        all_rows = record["frames"]
        legacy_rows = []
    else:
        record = None
        all_rows = None
        legacy_rows = []
        requested_set = set(plan["requested_frame_ids"])

    planned = plan["planned_keys"]
    width, height = manifest["native_resolution"]
    last_usable_id = None
    terminal_error = None
    terminal_index = None
    initial_timings = {}
    initial_operations = {"attempted": {}, "returned": {}}
    if physical:
        _increment(record["operation_counts"]["attempted"], "session_open")
        initial_operations["attempted"]["session_open"] = 1
        _physical_receipt_write(output, record)
    try:
        session_open_started = time.perf_counter()
        try:
            session = opener(plan=plan, manifest=manifest, bundle=Path(bundle),
                             output=output, device=device)
        finally:
            initial_timings["session_open"] = (time.perf_counter() - session_open_started) * 1000.0
        if physical and hasattr(session, "provenance"):
            record["provenance"] = session.provenance
        if hasattr(session, "checkpoint_sha256"):
            if physical:
                record["checkpoint_sha256"] = session.checkpoint_sha256
        if hasattr(session, "model_config") and physical:
            record["model_config"] = session.model_config
        _increment(record["operation_counts"]["returned"] if physical else {}, "session_open")
        initial_operations["returned"]["session_open"] = 1
        if physical:
            _physical_receipt_write(output, record)
        initial_operations["attempted"]["initialization"] = 1
        _increment(record["operation_counts"]["attempted"] if physical else {}, "initialization")
        initialization_started = time.perf_counter()
        try:
            session.initialize()
            initial_operations["returned"]["initialization"] = 1
            _increment(record["operation_counts"]["returned"] if physical else {}, "initialization")
        finally:
            initialization_ms = getattr(
                session, "initialization_ms", (time.perf_counter() - initialization_started) * 1000.0)
            initial_timings["initialization"] = _valid_finite(initialization_ms, "initialization timing")
            if initial_timings["initialization"] < 0:
                raise ValueError("Initialization timing must be nonnegative")
        _increment(record["operation_counts"]["attempted"] if physical else {}, "initial_selection")
        initial_operations["attempted"]["initial_selection"] = 1
        if physical:
            record["prompt_ledger"]["initial_selection"]["attempted"] = 1
            _physical_receipt_write(output, record)
        selection_started = time.perf_counter()
        selection_method_returned = False
        try:
            session.add_initial_selection(plan["selection"])
            selection_method_returned = True
        finally:
            click_returned = (selection_method_returned or
                              getattr(session, "selection_prompt_returned", False) is True)
            if click_returned:
                initial_operations["returned"]["initial_selection"] = 1
                _increment(record["operation_counts"]["returned"] if physical else {},
                           "initial_selection")
            if physical:
                record["prompt_ledger"]["initial_selection"]["returned"] = int(click_returned)
            selection_ms = getattr(
                session, "selection_prompt_ms", (time.perf_counter() - selection_started) * 1000.0)
            initial_timings["initial_selection"] = _valid_finite(selection_ms, "selection timing")
            if initial_timings["initial_selection"] < 0:
                raise ValueError("Selection timing must be nonnegative")
        for index, key in enumerate(planned):
            if key.frame_id not in requested_set:
                continue
            row = None
            source = {}
            attempted_detection = False
            detector_restore_attempted = False
            detector_returned = False
            automatic_conditioning = False
            conditioning_attempted = False
            conditioning_returned = False
            sam_attempted = False
            sam_returned = False
            writer_attempted = False
            writer_returned = False
            row_operations = {direction: {} for direction in ("attempted", "returned")}
            row_timings = dict(initial_timings) if index == 0 and physical else {}
            if index == 0 and physical:
                for direction in ("attempted", "returned"):
                    row_operations[direction].update(initial_operations[direction])
            detector_reason = None
            detector_sha = None
            residency = None
            detector_timings = {}
            try:
                sam_attempted = True
                _increment(record["operation_counts"]["attempted"] if physical else {}, "sam_step")
                row_operations["attempted"]["sam_step"] = 1
                sam_started = time.perf_counter()
                try:
                    step = session.step(index)
                finally:
                    row_timings["segmentation"] = (time.perf_counter() - sam_started) * 1000.0
                if not isinstance(step, dict):
                    raise ValueError("SAM session step must return a result object")
                source = step.get("source") or {}
                sam_returned = True
                row_operations["returned"]["sam_step"] = 1
                if physical:
                    _increment(record["operation_counts"]["returned"], "sam_step")
                if (isinstance(step.get("frame_idx"), bool) or
                        not isinstance(step.get("frame_idx"), int) or step.get("frame_idx") != index):
                    raise ValueError("SAM iterator ordinal differs from the frozen plan")
                object_ids = step.get("obj_ids")
                if (not isinstance(object_ids, list) or len(object_ids) != 1 or
                        isinstance(object_ids[0], bool) or not isinstance(object_ids[0], int) or
                        object_ids[0] != 1):
                    raise ValueError("SAM iterator returned an unexpected object identity")
                info = _validate_mask_info(session, step.get("mask"), width, height)
                mask = step["mask"]
                blank = step.get("blank", False)
                if type(blank) is not bool:
                    raise ValueError("SAM source blank-stress state must be a bool")
                hidden = blank
                if hidden:
                    mask = session.empty_mask()
                    info = _validate_mask_info(session, mask, width, height)
                usable = not hidden and info["foreground_pixels"] >= 24
                if not usable and not hidden:
                    attempted_detection = True
                    _increment(record["operation_counts"]["attempted"] if physical else {}, "detector")
                    row_operations["attempted"]["detector"] = 1
                    row_operations["attempted"]["sam_offload"] = 1
                    _increment(record["operation_counts"]["attempted"] if physical else {}, "sam_offload")
                    offload_started = time.perf_counter()
                    try:
                        residency = _validate_offload_receipt(session.offload_for_detector())
                    finally:
                        row_timings["sam_offload"] = (time.perf_counter() - offload_started) * 1000.0
                    row_operations["returned"]["sam_offload"] = 1
                    _increment(record["operation_counts"]["returned"] if physical else {}, "sam_offload")
                    residency["offload"] = {key: value for key, value in residency.items()
                                              if key != "restore"}
                    detector_started = time.perf_counter()
                    try:
                        payload = session.run_detector(index, key, prior_frame_id=last_usable_id)
                    finally:
                        detector_timings["detection"] = (time.perf_counter() - detector_started) * 1000.0
                    row_timings.update(detector_timings)
                    if not isinstance(payload, dict):
                        raise ValueError("Detector runtime must return a result object")
                    detector_reason = payload.get("reason")
                    payload_timings = payload.get("timings_ms")
                    if payload_timings is not None:
                        if not isinstance(payload_timings, dict):
                            raise ValueError("Detector timings_ms must be an object")
                        for timing_name, timing_value in payload_timings.items():
                            elapsed_ms = _valid_finite(timing_value, f"detector timing {timing_name}")
                            if elapsed_ms < 0:
                                raise ValueError("Detector timings must be nonnegative")
                            detector_timings[str(timing_name)] = elapsed_ms
                    row_timings.update(detector_timings)
                    detector_returned = payload.get("process_reaped") is True
                    if physical:
                        if detector_returned:
                            _increment(record["operation_counts"]["returned"], "detector")
                            row_operations["returned"]["detector"] = 1
                        if payload.get("process_returncode") != 0:
                            raise RuntimeError("Physical detector child failed after being reaped")
                        detected, detection_info = _validate_physical_detector_payload(
                            payload, key, plan, session)
                        detection_info = _validate_mask_info(session, detected, width, height)
                        detector_sha = payload.get("artifact_sha256")
                        if (not isinstance(detector_sha, str) or len(detector_sha) != 64 or
                                any(character not in "0123456789abcdefABCDEF" for character in detector_sha)):
                            raise ValueError("Physical detector result requires an artifact SHA-256")
                        detector_restore_attempted = True
                        row_operations["attempted"]["detector_restore"] = 1
                        _increment(record["operation_counts"]["attempted"] if physical else {},
                                   "detector_restore")
                        restore_started = time.perf_counter()
                        try:
                            residency["restore"] = _validate_restore_receipt(session.restore_after_detector())
                        finally:
                            row_timings["detector_restore"] = (time.perf_counter() - restore_started) * 1000.0
                        row_operations["returned"]["detector_restore"] = 1
                        _increment(record["operation_counts"]["returned"] if physical else {},
                                   "detector_restore")
                        if detection_info["foreground_pixels"] >= 24:
                            mask = detected
                            info = detection_info
                            usable = True
                            conditioning_attempted = True
                    else:
                        detected = payload.get("mask")
                        detector_restore_attempted = True
                        row_operations["attempted"]["detector_restore"] = 1
                        _increment(record["operation_counts"]["attempted"] if physical else {},
                                   "detector_restore")
                        restore_started = time.perf_counter()
                        try:
                            _validate_restore_receipt(session.restore_after_detector())
                        finally:
                            row_timings["detector_restore"] = (time.perf_counter() - restore_started) * 1000.0
                        row_operations["returned"]["detector_restore"] = 1
                        _increment(record["operation_counts"]["returned"] if physical else {},
                                   "detector_restore")
                        if detected is not None:
                            detected_info = _validate_mask_info(session, detected, width, height)
                            if detected_info["foreground_pixels"] > 0:
                                mask = detected
                                info = detected_info
                                usable = True
                                conditioning_attempted = True
                    if conditioning_attempted:
                        _increment(record["operation_counts"]["attempted"] if physical else {},
                                   "automatic_conditioning")
                        row_operations["attempted"]["automatic_conditioning"] = 1
                        conditioning_entry = None
                        if physical:
                            conditioning_entry = {
                                "frame_id": key.frame_id,
                                "source_frame_id": plan["source_frame_ids"][key.frame_id],
                                "timestamp_s": key.timestamp_s,
                                "detector_artifact_sha256": detector_sha,
                                "attempted": True,
                                "returned": False,
                            }
                            record["prompt_ledger"]["automatic_mask_conditioning"].append(
                                conditioning_entry)
                        condition_started = time.perf_counter()
                        try:
                            session.condition(index, mask)
                        finally:
                            row_timings["automatic_conditioning"] = (time.perf_counter() - condition_started) * 1000.0
                        conditioning_returned = True
                        automatic_conditioning = True
                        row_operations["returned"]["automatic_conditioning"] = 1
                        _increment(record["operation_counts"]["returned"] if physical else {},
                                   "automatic_conditioning")
                        if physical:
                            conditioning_entry["returned"] = True
                    elif detector_returned:
                        if physical:
                            row_operations["attempted"]["sam_resume"] = 1
                            _increment(record["operation_counts"]["attempted"], "sam_resume")
                        resume_started = time.perf_counter()
                        try:
                            session.resume_after_detector(index)
                        finally:
                            row_timings["sam_resume"] = (time.perf_counter() - resume_started) * 1000.0
                        if physical:
                            row_operations["returned"]["sam_resume"] = 1
                            _increment(record["operation_counts"]["returned"], "sam_resume")
                if not usable and not hidden and detector_returned:
                    if physical and info["foreground_pixels"] < 24 and detector_sha is not None:
                        # A valid zero/small current-image detector result is evidence
                        # of no detection; preserve that returned mask as the row.
                        detected_info = session.mask_info(payload.get("mask"))
                        if tuple(detected_info.get("shape", ())) == (height, width):
                            mask = payload["mask"]
                            info = _validate_mask_info(session, mask, width, height)
                if hidden:
                    reason = "blank_current_image"
                elif usable:
                    reason = None
                elif detector_returned and physical:
                    reason = "automatic_detection_no_usable_foreground"
                elif detector_returned and detector_reason == "cnos_runtime_failure":
                    reason = "cnos_runtime_failure"
                else:
                    reason = "sam2_visible_object_unavailable"

                filename_id = key.frame_id if physical else plan["source_frame_ids"][key.frame_id]
                relative_path = f"masks/{filename_id}.png"
                if physical and index == 0:
                    step = dict(step)
                    step["timings_ms"] = dict(step.get("timings_ms") or {})
                    step["timings_ms"].update(initial_timings)
                writer_attempted = True
                _increment(record["operation_counts"]["attempted"] if physical else {}, "mask_write")
                row_operations["attempted"]["mask_write"] = 1
                write_started = time.perf_counter()
                mask_path = output / relative_path
                try:
                    artifact = session.write_mask(mask_path, mask)
                finally:
                    row_timings["mask_write"] = (time.perf_counter() - write_started) * 1000.0
                if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
                    raise OSError("SAM runtime did not return a mask artifact digest")
                if physical and (len(artifact["sha256"]) != 64 or any(
                        character not in "0123456789abcdefABCDEF" for character in artifact["sha256"])):
                    raise OSError("SAM runtime returned a malformed mask artifact SHA-256")
                if physical:
                    resolved_mask = mask_path.resolve()
                    if not resolved_mask.is_relative_to(output.resolve()) or not resolved_mask.is_file():
                        raise OSError("SAM runtime did not create the planned mask artifact")
                    actual_sha = _sha256_path(resolved_mask)
                    if actual_sha.lower() != artifact["sha256"].lower():
                        raise OSError("SAM runtime mask artifact digest differs from its written bytes")
                    if "bytes" in artifact:
                        byte_count = artifact["bytes"]
                        if (isinstance(byte_count, bool) or not isinstance(byte_count, int) or
                                byte_count != resolved_mask.stat().st_size):
                            raise OSError("SAM runtime mask artifact byte count is inconsistent")
                _increment(record["operation_counts"]["returned"] if physical else {}, "mask_write")
                row_operations["returned"]["mask_write"] = 1
                writer_returned = True

                if physical:
                    if not source:
                        source = session.snapshot_source(index)
                    row = _physical_row_from_result(
                        plan, index, dict(step, detector_attempted=attempted_detection,
                                          detector_returned=detector_returned,
                                          detector_timings_ms=detector_timings),
                        mask, info,
                        {"path": relative_path, "sha256": artifact["sha256"]},
                        usable=usable, reason=reason, detector_reason=detector_reason,
                        detector_sha=detector_sha, source=source,
                        automatic_conditioning=automatic_conditioning, residency=residency,
                        operations=row_operations, timings=row_timings)
                    record["frames"][index] = row
                    _physical_receipt_write(output, record)
                else:
                    legacy_timings = dict(step.get("timings_ms") or {})
                    if attempted_detection:
                        legacy_timings.update(detector_timings)
                    row = _legacy_frame_row(
                        key, plan["source_frame_ids"][key.frame_id], plan["roles"][index],
                        "available" if usable else "lost", relative_path,
                        "blank_current_image" if hidden else None if usable else "sam2_visible_object_unavailable",
                        legacy_timings, not usable, detector_reason,
                        detection_attempted=attempted_detection)
                    legacy_rows.append(row)
                    record_legacy = {
                        "model": "SAM 2.1 Hiera Base Plus", "device": device,
                        "propagation": "forward only; setup frame prompts only",
                        "provenance": getattr(session, "provenance", _default_provenance(plan)),
                        "frames": legacy_rows,
                        "position_grid": "exact arange form of upstream cumsum(ones)",
                        "stress_test": None if plan["forced_occlusion"] is None else {
                            "occlusion_start": plan["forced_occlusion"][0],
                            "source_frames": plan["forced_occlusion"][1] - plan["forced_occlusion"][0],
                            "input": "black RGB frames",
                        },
                        "mask_association": plan["mask_association"],
                        "deterministic_algorithms": session.torch.are_deterministic_algorithms_enabled()
                            if hasattr(session, "torch") else False,
                    }
                    _write_json_atomic(output / "results.json", record_legacy)
                if usable:
                    last_usable_id = plan["source_frame_ids"][key.frame_id]
                if index % 40 == 0:
                    print(manifest.get("object", "SAM"), "SAM2", index, "/",
                          len(plan["requested_keys"]),
                          "available" if usable else "lost", flush=True)
            except SourceDecodeError as exc:
                if not physical:
                    raise
                terminal_error = exc.reason
                terminal_index = index
                if physical:
                    try:
                        source = session.snapshot_source(index)
                    except Exception:
                        source = {}
                    _mark_terminal_rows(record, plan, index, exc.reason, source,
                                        operations=row_operations, timings=row_timings,
                                        residency=residency, detector_reason=detector_reason,
                                        detector_sha=detector_sha,
                                        automatic_conditioning=automatic_conditioning)
                    _physical_receipt_write(output, record)
                break
            except Exception as exc:
                terminal_error = "sam2_runtime_failure"
                terminal_index = index
                if attempted_detection:
                    terminal_error = "detector_or_recovery_failure"
                if physical:
                    try:
                        source = session.snapshot_source(index)
                    except Exception:
                        source = {}
                    _mark_terminal_rows(record, plan, index, terminal_error, source,
                                        operations=row_operations, timings=row_timings,
                                        residency=residency, detector_reason=detector_reason,
                                        detector_sha=detector_sha,
                                        automatic_conditioning=automatic_conditioning)
                    record["terminal_error"] = f"{type(exc).__name__}: {exc}"[:2000]
                    _physical_receipt_write(output, record)
                else:
                    raise
                break

    except SourceDecodeError as exc:
        if not physical:
            raise
        terminal_error = exc.reason
        terminal_index = exc.frame_id
        try:
            source = session.snapshot_source(exc.frame_id) if session is not None else {}
        except Exception:
            source = {}
        _mark_terminal_rows(record, plan, exc.frame_id, exc.reason, source,
                            operations=initial_operations, timings=initial_timings)
        record["terminal_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        _physical_receipt_write(output, record)
    except Exception as exc:
        if not physical:
            raise
        if terminal_error is None:
            terminal_error = "sam2_runtime_initialization_failure"
            record["terminal_error"] = f"{type(exc).__name__}: {exc}"[:2000]
            terminal_index = 0
            try:
                source = session.snapshot_source(0) if session is not None else {}
            except Exception:
                source = {}
            _mark_terminal_rows(record, plan, 0, terminal_error, source,
                                operations=initial_operations, timings=initial_timings)
            _physical_receipt_write(output, record)
    finally:
        cleanup = {"session_opened": session is not None}
        if session is not None:
            try:
                cleanup.update(session.close())
            except Exception as exc:
                cleanup["close_error"] = f"{type(exc).__name__}: {exc}"[:2000]
                cleanup["session_closed"] = False
        else:
            cleanup.update({"iterator_released": False, "decoder_released": False,
                            "predictor_released": False, "loader_hook_restored": True})
        if physical:
            required_cleanup = ("iterator_released", "decoder_released",
                                "predictor_released", "loader_hook_restored")
            cleanup_complete = all(cleanup.get(key) is True for key in required_cleanup)
            cleanup["complete"] = cleanup_complete
            record["cleanup"] = cleanup
            if not cleanup_complete:
                terminal_error = terminal_error or "runtime_cleanup_unconfirmed"
            requested_done = all(
                record["frames"][index]["measurement_state"] == "measured"
                for index, key in enumerate(planned) if key.frame_id in requested_set)
            if terminal_error is None and requested_done:
                record["stage_completed"] = True
                record["status"] = "diagnostic_prefix" if plan["diagnostic_prefix"] else "complete"
            else:
                record["stage_completed"] = False
                record["status"] = "failed"
                if terminal_error:
                    record.setdefault("terminal_error", terminal_error)
            _physical_receipt_write(output, record)

    if physical:
        if terminal_error is not None:
            raise RuntimeError(f"SAM physical cache stage failed: {terminal_error}")
        return record
    return {"frames": legacy_rows}


def run(bundle, output, device="cuda", limit=None, forced_occlusion=None,
        mask_association=False, *, runtime=None):
    """Produce masks after a pure input/options preflight.

    ``runtime`` is an optional test seam. An injected object provides
    ``open_session(plan=..., manifest=..., bundle=..., output=..., device=...)``;
    the returned session implements ``initialize()``,
    ``add_initial_selection(selection)``, ``step(index)``, ``mask_info(mask)``,
    ``empty_mask()``, ``write_mask(path, mask)``, ``offload_for_detector()``,
    ``run_detector(index, key, prior_frame_id=...)``,
    ``restore_after_detector()``, ``resume_after_detector(index)``,
    ``condition(index, mask)``, ``snapshot_source(index)`` and ``close()``.
    Offload and restore calls return auditable device receipts; close must
    explicitly confirm iterator, decoder, predictor and loader-hook release.
    The default pinned runtime is not imported or constructed when an injected
    session is supplied.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    raw = _read_input(bundle)
    plan = prepare_cache_plan(raw, limit=limit, forced_occlusion=forced_occlusion,
                              mask_association=mask_association)
    manifest = _load_manifest(raw)
    _validate_runtime_manifest(manifest, bundle, plan, verify_resources=runtime is None)
    output_path = Path(output)
    if plan["clock_mode"] == "capture_seconds_v1":
        bundle_path = Path(bundle).resolve()
        resolved_output = output_path.resolve()
        if resolved_output == bundle_path or resolved_output.is_relative_to(bundle_path):
            raise ValueError("Physical SAM output must be outside the immutable input bundle")
        if output_path.exists():
            raise FileExistsError("Physical SAM output must be a fresh directory")
    runtime_manifest = _runtime_manifest_view(manifest)
    return _run_plan(plan, runtime_manifest, bundle, output_path, device, runtime)


