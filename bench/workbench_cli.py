"""Command line and pipeline for the video validation workbench.

    python -m bench.workbench_cli --help
    python -m bench.workbench_cli inspect
    python -m bench.workbench_cli init [--root NAME=PATH ...]
    python -m bench.workbench_cli import video PATH --clip-id ID (--K fx,fy,cx,cy | --calibration-unavailable)
    python -m bench.workbench_cli import images DIR --clip-id ID ...
    python -m bench.workbench_cli import masks DIR --clip-id ID --mask-set NAME --producer TEXT
    python -m bench.workbench_cli suite [--verify-images]
    python -m bench.workbench_cli run CLIP_ID --mode smoke|focused|regression [--masks SET]
        [--admission DIR --metric-bundle FILE --head head.pt --device cpu]
    python -m bench.workbench_cli report RUN_ID
    python -m bench.workbench_cli segment CLIP_ID --mask-set NAME      (SAM 2.1 tap-to-mask)
    python -m bench.workbench_cli gate bundle|case|contract|geometry|parity|verify ...

Workspace: ``.cache/video-workbench``. Source originals are referenced in place
through registered roots and never copied. Budgets: newly imported inputs
<= 250,000,000 bytes; derived + scratch <= 1 GiB; >= 1 GiB free disk reserve.
Exit codes: 0 completed, 1 failed, 2 usage, 3 blocked (prerequisites missing),
4 dry run (nothing executed).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from . import workbench_contract as wc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / ".cache" / "video-workbench"
CONFIG_SCHEMA = "workbench_config_v1"
MASK_SCHEMA = "workbench_observed_masks_v1"
EXIT_COMPLETED, EXIT_FAILED, EXIT_USAGE, EXIT_BLOCKED, EXIT_DRY_RUN = 0, 1, 2, 3, 4
VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_DECODE_FRAMES = 20_000
BUILTIN_ROOTS = {
    "model_quality_inputs": ".cache/model-quality/inputs",
    "ycb_pilot": ".cache/prototype-sprint/dataset-pilot-v1",
}
TRACKER_VERSION = "relative_tracker_v1"
# Ordinary runs execute every relative/neural frame through the portable boundary. Tracking cache
# keys and payloads bind this path id; entries from the earlier direct-session path never qualify.
ADOPTED_EXECUTION_PATH = "workbench_frame_boundary_v1_main_pipeline"
BOUNDARY_SOURCE = "bench/workbench_frame_boundary.py"


def boundary_context_digest(context) -> str:
    """Run-ID-independent digest of a canonical boundary context (clip incl. K/time/frames, role, order,
    seed/warm-up, binding, prefix attestation, requested primary, artwork and onboarding)."""
    return wc.canonical_sha256({
        "schema": context.schema, "clip_sha256": context.clip_sha256, "role": context.role,
        "source_mode": context.source_mode, "warmup_frame_ids": list(context.warmup_frame_ids),
        "scheduled_frame_ids": list(context.scheduled_frame_ids), "permitted_order": list(context.permitted_order),
        "K": list(context.K), "width": context.width, "height": context.height,
        "binding_sha256": context.binding_sha256, "prefix_attestation": wc.loads_strict(context.prefix_attestation_json),
        "run_primary": dict(context.run.primary), "artwork_kind": context.run.artwork_kind,
        "onboarding": context.run.onboarding.to_dict()})


def boundary_provenance(context) -> dict:
    """Execution provenance bound into adopted tracking cache keys and payloads."""
    from . import workbench_frame_boundary as fb
    return {"execution_path": ADOPTED_EXECUTION_PATH, "boundary_version": fb.BOUNDARY_VERSION,
            "boundary_source_sha256": sha256_file(ROOT / BOUNDARY_SOURCE),
            "context_digest": boundary_context_digest(context), "binding_sha256": context.binding_sha256}


class WorkbenchCLIError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[workbench] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(wc.canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> Any:
    return wc.loads_strict(Path(path).read_bytes())


def _tree_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _d, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def contained(root: Path, relative: str) -> Path:
    """Resolve a POSIX relative path under ``root`` refusing traversal and symlinks."""
    wc._relative_path(relative, "path")
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise WorkbenchCLIError("registered paths cannot traverse symlinks")
    resolved = current.resolve(strict=False)
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise WorkbenchCLIError("path escapes its registered root")
    return resolved


# --------------------------------------------------------------------------
# workspace

class Workspace:
    def __init__(self, path: Path | str = DEFAULT_WORKSPACE, *, repo_root: Path = ROOT):
        self.path = Path(path).resolve()
        self.repo_root = repo_root
        self.config_path = self.path / "config.json"

    # layout
    @property
    def clips_dir(self): return self.path / "clips"
    @property
    def imports_dir(self): return self.path / "imports"
    @property
    def masks_dir(self): return self.path / "masks"
    @property
    def derived_dir(self): return self.path / "derived"
    @property
    def scratch_dir(self): return self.path / "scratch"
    @property
    def runs_dir(self): return self.path / "runs"
    @property
    def suite_dir(self): return self.path / "suite"

    def init(self, roots: Mapping[str, str] | None = None) -> dict:
        for d in (self.path, self.clips_dir, self.imports_dir, self.masks_dir, self.derived_dir, self.scratch_dir,
                  self.runs_dir, self.suite_dir):
            d.mkdir(parents=True, exist_ok=True)
        config = self.config()
        for name, path in (roots or {}).items():
            if not wc.ROOT_ID_RE.fullmatch(name) or name in ("workbench",) or name in BUILTIN_ROOTS:
                raise WorkbenchCLIError(f"invalid or reserved root name {name}")
            resolved = Path(path).expanduser().resolve(strict=True)
            if not resolved.is_dir():
                raise WorkbenchCLIError(f"root {name} must be an existing directory")
            config["roots"][name] = str(resolved)
        tmp = self.config_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(wc.canonical_json(config))
        os.replace(tmp, self.config_path)
        return config

    def config(self) -> dict:
        if self.config_path.is_file():
            value = _read_json(self.config_path)
            if value.get("schema") != CONFIG_SCHEMA:
                raise WorkbenchCLIError("workspace config schema differs")
            return value
        return {"schema": CONFIG_SCHEMA, "roots": {}}

    def roots(self) -> dict[str, Path]:
        out = {"workbench": self.path}
        for name, rel in BUILTIN_ROOTS.items():
            p = self.repo_root / rel
            if p.is_dir():
                out[name] = p.resolve()
        for name, path in self.config().get("roots", {}).items():
            out[name] = Path(path)
        return out

    def resolve(self, root_id: str, relative: str) -> Path:
        roots = self.roots()
        if root_id not in roots:
            raise WorkbenchCLIError(f"unregistered root {root_id}")
        return contained(roots[root_id], relative)

    def locate(self, path: Path) -> tuple[str, str]:
        """Find the registered root containing ``path`` (in-place reference)."""
        resolved = Path(path).resolve(strict=True)
        best = None
        for name, root in self.roots().items():
            root = root.resolve()
            if root in resolved.parents and (best is None or len(str(root)) > len(str(best[1]))):
                best = (name, root)
        if best is None:
            raise WorkbenchCLIError(f"{resolved} is not inside a registered root; use `init --root NAME=DIR` first")
        return best[0], resolved.relative_to(best[1]).as_posix()

    # budgets
    def budget(self) -> dict:
        inputs = _tree_size(self.imports_dir) + sum(c.get("provenance", {}).get("counted_input_bytes", 0)
                                                    for c in self._clip_dicts())
        derived = sum(_tree_size(d) for d in (self.derived_dir, self.scratch_dir, self.runs_dir, self.masks_dir,
                                              self.suite_dir))
        self.path.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.path).free
        return {"new_inputs_bytes": inputs, "new_inputs_limit": wc.NEW_INPUT_BUDGET_BYTES,
                "derived_bytes": derived, "derived_limit": wc.DERIVED_BUDGET_BYTES,
                "free_bytes": free, "free_reserve": wc.FREE_RESERVE_BYTES,
                "within_budget": inputs <= wc.NEW_INPUT_BUDGET_BYTES and derived <= wc.DERIVED_BUDGET_BYTES
                and free >= wc.FREE_RESERVE_BYTES}

    def require_budget(self, *, extra_inputs: int = 0, extra_derived: int = 0) -> None:
        b = self.budget()
        if b["new_inputs_bytes"] + extra_inputs > wc.NEW_INPUT_BUDGET_BYTES:
            raise WorkbenchCLIError("new-input budget of 250,000,000 bytes would be exceeded")
        if b["derived_bytes"] + extra_derived > wc.DERIVED_BUDGET_BYTES:
            raise WorkbenchCLIError("derived+scratch budget of 1 GiB would be exceeded")
        if b["free_bytes"] - extra_derived - extra_inputs < wc.FREE_RESERVE_BYTES:
            raise WorkbenchCLIError("1 GiB free-disk reserve would be violated")

    # clips
    def _clip_dicts(self) -> list[dict]:
        out = []
        if self.clips_dir.is_dir():
            for p in sorted(self.clips_dir.glob("*/clip.json")):
                try:
                    out.append(_read_json(p))
                except (OSError, ValueError):
                    continue
        return out

    def clip_path(self, clip_id: str) -> Path:
        wc._ident(clip_id, "clip_id")
        return self.clips_dir / clip_id / "clip.json"

    def save_clip(self, clip: wc.ClipSpec) -> Path:
        path = self.clip_path(clip.clip_id)
        body = clip.to_dict()
        if path.exists():
            if _read_json(path) == wc.loads_strict(wc.canonical_json(body)):
                return path
            raise WorkbenchCLIError(f"clip {clip.clip_id} already exists with different content; choose a new clip id")
        _write_new(path, body)
        return path

    def load_clip(self, clip_id: str) -> wc.ClipSpec:
        path = self.clip_path(clip_id)
        if not path.is_file():
            raise WorkbenchCLIError(f"unknown clip {clip_id}")
        return wc.ClipSpec.from_dict(_read_json(path))

    def list_clips(self) -> list[wc.ClipSpec]:
        out = []
        for row in self._clip_dicts():
            try:
                out.append(wc.ClipSpec.from_dict(row))
            except wc.ContractError:
                continue
        return out

    def media_path(self, clip: wc.ClipSpec) -> Path:
        return self.resolve(clip.media.root_id, clip.media.path)


# --------------------------------------------------------------------------
# streaming decoding

class FrameReader:
    """Streams exact source frames; never dumps a full video."""

    def __init__(self, workspace: Workspace, clip: wc.ClipSpec):
        self.ws, self.clip = workspace, clip
        self._cap = None
        self._next = None
        self._media = workspace.media_path(clip)

    def verify_media(self) -> dict:
        if self.clip.source_kind != "video_file":
            return {"kind": "image_sequence", "verified_per_frame": True}
        if not self._media.is_file():
            raise WorkbenchCLIError("clip media is unavailable")
        size = self._media.stat().st_size
        if self.clip.media.bytes is not None and size != self.clip.media.bytes:
            raise WorkbenchCLIError("clip media size changed")
        digest = sha256_file(self._media)
        if self.clip.media.sha256 is not None and digest != self.clip.media.sha256:
            raise WorkbenchCLIError("clip media SHA-256 changed")
        return {"kind": "video_file", "sha256": digest, "bytes": size}

    def verify_frames(self, frame_ids: Sequence[str]) -> dict[str, str | None]:
        """Re-hash source files now (image sequences). Videos are covered by the
        whole-file hash from ``verify_media``; their per-frame entry is None."""
        out: dict[str, str | None] = {}
        for fid in frame_ids:
            frame = self.clip.frame(fid)
            if self.clip.source_kind != "image_sequence":
                out[fid] = None
                continue
            root = self._media if self._media.is_dir() else self._media.parent
            digest = sha256_file(contained(root, frame.image_path))
            if digest != frame.image_sha256:
                raise WorkbenchCLIError(f"image {frame.image_path} hash differs from the registered clip")
            out[fid] = digest
        return out

    def read(self, frame: wc.FrameSpec) -> tuple[np.ndarray, dict]:
        from .vision import cv2
        if self.clip.source_kind == "image_sequence":
            root = self._media if self._media.is_dir() else self._media.parent
            path = contained(root, frame.image_path)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != frame.image_sha256:
                raise WorkbenchCLIError(f"image {frame.image_path} hash differs")
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise WorkbenchCLIError(f"cannot decode {frame.image_path}")
            media_time = None
        else:
            if self._cap is None:
                self._cap = cv2.VideoCapture(str(self._media))
                self._next = 0
                if not self._cap.isOpened():
                    raise WorkbenchCLIError("cannot open clip media")
            if self._next != frame.source_frame_index:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame.source_frame_index)
            ok, bgr = self._cap.read()
            position = int(round(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if not ok or bgr is None:
                raise WorkbenchCLIError(f"decoder returned no frame {frame.source_frame_index}")
            if position != frame.source_frame_index + 1:
                raise WorkbenchCLIError(f"inexact seek for frame {frame.source_frame_index} (decoder at {position - 1})")
            self._next = position
            msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            media_time = float(msec) / 1000.0 if np.isfinite(msec) else None
            if frame.media_time_s is not None and (media_time is None or abs(media_time - frame.media_time_s) > 1e-3):
                raise WorkbenchCLIError(f"decoder PTS differs for frame {frame.frame_id}")
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        if rgb.shape[:2] != (self.clip.height, self.clip.width):
            raise WorkbenchCLIError(f"frame {frame.frame_id} dimensions differ from the clip")
        from . import workbench_geometry as wg
        return rgb, {"frame_id": frame.frame_id, "source_frame_index": frame.source_frame_index,
                     "image_sha256": wg.array_sha256(rgb), "decoder_media_time_s": media_time}

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def probe_video(path: Path, *, max_frames: int = MAX_DECODE_FRAMES) -> dict:
    """Sequentially decode once (streaming) to count frames and read decoder PTS."""
    from .vision import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise WorkbenchCLIError("cannot open video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    times = []
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            times.append(float(msec) / 1000.0 if np.isfinite(msec) and msec >= 0 else None)
            if len(times) > max_frames:
                raise WorkbenchCLIError(f"video exceeds {max_frames} frames")
    finally:
        cap.release()
    if not times:
        raise WorkbenchCLIError("video has no decodable frames")
    reliable = all(t is not None for t in times) and all(b > a for a, b in zip(times, times[1:]))
    return {"width": width, "height": height, "frame_count": len(times), "media_times": times,
            "pts_reliable": reliable}


# --------------------------------------------------------------------------
# observed masks

class MaskStore:
    """Observed masks produced elsewhere (e.g. reviewed segmentation), keyed by frame id."""

    def __init__(self, workspace: Workspace, clip_id: str, mask_set: str | None):
        self.ws, self.clip_id, self.mask_set = workspace, clip_id, mask_set
        self.manifest = None
        if mask_set:
            path = workspace.masks_dir / clip_id / mask_set / "manifest.json"
            if path.is_file():
                manifest = _read_json(path)
                if manifest.get("schema") != MASK_SCHEMA or manifest.get("clip_id") != clip_id:
                    raise WorkbenchCLIError("mask manifest schema/clip differs")
                if manifest.get("evaluator_inputs_read") is not False:
                    raise WorkbenchCLIError("mask set declares evaluator inputs; rejected")
                self.manifest = manifest
                self.dir = path.parent

    @property
    def available(self) -> bool:
        return self.manifest is not None

    def identity(self) -> dict | None:
        if self.manifest is None:
            return None
        return {"mask_set": self.mask_set, "producer": self.manifest.get("producer"),
                "manifest_sha256": wc.canonical_sha256(self.manifest)}

    def entry(self, frame_id: str) -> dict | None:
        if self.manifest is None:
            return None
        row = self.manifest["masks"].get(frame_id)
        if row is None:
            return None
        return {**row, "abs_path": str(contained(self.dir, row["path"]))}

    def verify(self, frame_ids: Sequence[str]) -> dict[str, str | None]:
        """Re-hash the mask files now; a modified mask can never reuse a stale manifest hash."""
        out: dict[str, str | None] = {}
        for fid in frame_ids:
            row = self.entry(fid)
            if row is None:
                out[fid] = None
                continue
            digest = sha256_file(Path(row["abs_path"]))
            if digest != row["sha256"]:
                raise WorkbenchCLIError(f"mask file for {fid} changed after registration")
            out[fid] = digest
        return out

    def load(self, frame_id: str, shape: tuple[int, int]) -> tuple[np.ndarray | None, dict | None]:
        from .vision import cv2
        from . import workbench_geometry as wg
        row = self.entry(frame_id)
        if row is None:
            return None, None
        raw = Path(row["abs_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise WorkbenchCLIError(f"mask file for {frame_id} hash differs")
        m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != shape:
            raise WorkbenchCLIError(f"mask for {frame_id} undecodable or wrong size")
        mask = m > 127
        return mask, {"mask_frame_id": frame_id, "mask_sha256": wg.mask_sha256(mask), "file_sha256": row["sha256"]}


def import_masks(ws: Workspace, clip: wc.ClipSpec, source_dir: Path, mask_set: str, producer: str) -> dict:
    """Copy observed mask PNGs named <frame_id>.png into the workspace (derived budget)."""
    from .vision import cv2
    wc._ident(mask_set, "mask_set")
    dest = ws.masks_dir / clip.clip_id / mask_set
    if dest.exists():
        raise WorkbenchCLIError("mask set already exists; masks are immutable")
    files = {p.stem: p for p in Path(source_dir).glob("*.png")}
    rows = {}
    total = sum(p.stat().st_size for p in files.values())
    ws.require_budget(extra_derived=total)
    dest.mkdir(parents=True)
    for frame in clip.frames:
        p = files.get(frame.frame_id)
        if p is None:
            continue
        raw = p.read_bytes()
        m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (clip.height, clip.width):
            raise WorkbenchCLIError(f"mask {p.name} undecodable or wrong size")
        (dest / p.name).write_bytes(raw)
        rows[frame.frame_id] = {"path": p.name, "sha256": hashlib.sha256(raw).hexdigest(),
                                "width": clip.width, "height": clip.height}
    manifest = {"schema": MASK_SCHEMA, "clip_id": clip.clip_id, "mask_set": mask_set,
                "producer": producer, "evaluator_inputs_read": False, "masks": rows,
                "note": "observed foreground masks; projected silhouettes are not accepted here"}
    _write_new(dest / "manifest.json", manifest)
    return {"mask_set": mask_set, "frames": len(rows)}


def run_segmentation(ws: Workspace, clip: wc.ClipSpec, onboarding: wc.Onboarding, scheduled: Sequence[str],
                     mask_set: str, *, device: str = "cpu", cancel: threading.Event | None = None) -> dict:
    """Tap-to-mask: prompt SAM 2.1 with the reviewed selection and write observed
    masks for the onboarding prefix and scheduled frames (causal release)."""
    from . import workbench_neural as wn
    from . import workbench_geometry as wg
    from .vision import cv2
    if onboarding.selection is None:
        raise WorkbenchCLIError("segmentation needs a reviewed positive/negative selection")
    wc.check_onboarding(clip, onboarding, scheduled)
    wc._ident(mask_set, "mask_set")
    dest = ws.masks_dir / clip.clip_id / mask_set
    if dest.exists():
        raise WorkbenchCLIError("mask set already exists; masks are immutable")
    pre = wn.segmentation_preflight(verify_hash=False)
    if not pre["ready"]:
        raise WorkbenchCLIError(f"segmentation blocked: {pre['blocked_reasons']}")
    ws.require_budget(extra_derived=len(scheduled) * 200_000)
    position = clip.frame_position()
    order = sorted(set(onboarding.keyframe_ids) | set(scheduled), key=lambda f: position[f])
    prompt_index = order.index(onboarding.selection.frame_id)
    reader = FrameReader(ws, clip)
    dest.mkdir(parents=True)
    rows: dict[str, dict] = {}

    def read_rgb(index: int) -> np.ndarray:
        return reader.read(clip.frame(order[index]))[0]

    def write_mask(index: int, mask: np.ndarray) -> None:
        if cancel is not None and cancel.is_set():
            raise WorkbenchCLIError("segmentation cancelled before writing the next mask")
        fid = order[index]
        ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not ok:
            raise WorkbenchCLIError("mask encoding failed")
        data = buf.tobytes()
        ws.require_budget(extra_derived=len(data))  # checked before every mask write
        name = f"m{index:05d}.png"
        with (dest / name).open("xb") as stream:
            stream.write(data)
        rows[fid] = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "width": clip.width,
                     "height": clip.height, "content_sha256": wg.mask_sha256(mask), "empty": bool(not mask.any())}
    class _PersistedSamples(list):
        """Each sample is also flushed to disk so it survives a native abort."""

        def append(self, item):  # noqa: D401
            super().append(item)
            try:
                with (dest / "resource-samples.jsonl").open("ab") as stream:
                    stream.write(json.dumps(item, sort_keys=True, default=str).encode("utf-8") + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                pass

    samples: list = _PersistedSamples()
    from . import workbench_telemetry as wt
    try:
        with wt.gpu_stage("segmentation"):
            result = wn.segment_prompted(read_rgb, len(order), prompt_index, onboarding.selection, height=clip.height,
                                         width=clip.width, write_mask=write_mask, device=device, cancel=cancel,
                                         resource_samples=samples, disk_path=ws.path)
    except BaseException as exc:  # preserve partial masks + best-effort failure telemetry
        try:
            samples.append(wt.resource_sample("failure", disk_path=ws.path))
        except Exception as sample_exc:  # noqa: BLE001 - broken CUDA etc.
            samples.append({"label": "failure", "sample_error": f"{type(sample_exc).__name__}: {sample_exc}"[:300]})
        try:
            _write_new(dest / "failure.json", {"error": f"{type(exc).__name__}: {exc}"[:2000],
                                               "frames_written": len(rows), "resource_samples": samples,
                                               "masks_written": rows})
        except Exception:  # noqa: BLE001 - never hide the original failure
            pass
        raise
    finally:
        reader.close()
    manifest = {"schema": MASK_SCHEMA, "clip_id": clip.clip_id, "mask_set": mask_set,
                "producer": {"kind": "sam2.1_video_prompted", "checkpoint_sha256": result["checkpoint_sha256"],
                             "config": result["config"], "device": device,
                             "selection": onboarding.selection.to_dict(), "prompt_frame_id": onboarding.selection.frame_id,
                             "point_convention": result["point_convention"], "causal_guard": result["causal_guard"]},
                "evaluator_inputs_read": False, "masks": rows, "cancelled": result["cancelled"],
                "telemetry": {**result["telemetry"],
                              "segmentation_ms_by_frame_id": {order[int(k)]: v for k, v in
                                                              result["telemetry"]["segmentation_ms_per_frame"].items()}},
                "note": "observed foreground masks from reviewed prompts; never projected silhouettes"}
    _write_new(dest / "manifest.json", manifest)
    return {"mask_set": mask_set, "frames": len(rows), "cancelled": result["cancelled"]}


# --------------------------------------------------------------------------
# stage cache (separately owned module; imported lazily)

def stage_cache(ws: Workspace):
    """Open the separately owned StageCache sized to the *remaining* shared 1 GiB
    derived budget (minus a lock/record margin), so cache growth cannot push the
    workspace aggregate over the cap."""
    try:
        from . import workbench_cache  # type: ignore
    except ImportError as exc:
        return None, f"dependency_unavailable: bench.workbench_cache ({exc})"
    root = ws.derived_dir / "stage-cache"
    try:
        other = ws.budget()["derived_bytes"] - _tree_size(root)
        remaining = wc.DERIVED_BUDGET_BYTES - other - CACHE_MARGIN_BYTES
        if remaining <= 0:
            return None, "budget_blocked: no derived budget remains for the stage cache"
        cache = workbench_cache.StageCache(root, max_bytes=int(remaining), reserve_bytes=wc.FREE_RESERVE_BYTES)
        return (workbench_cache, cache), None
    except Exception as exc:  # noqa: BLE001 - interface differences reported, not hidden
        return None, f"dependency_unavailable: StageCache init failed ({type(exc).__name__}: {exc})"


CACHE_MARGIN_BYTES = 1 << 20  # covers lock files/temp records beyond the cache's own accounting


def _module_versions(*names: str) -> dict:
    out = {}
    for name in names:
        p = ROOT / "bench" / f"{name}.py"
        out[name] = sha256_file(p) if p.is_file() else None
    return out


# --------------------------------------------------------------------------
# pipeline

@dataclass
class RunOptions:
    mask_set: str | None = None
    colmap: str | None = None
    head_path: str | None = None
    admission: Any = None              # ServingAdmission (verified by caller) or None
    neural_backends: Mapping[str, Any] = field(default_factory=dict)  # role -> NeuralBackend (tests inject fakes)
    family: str | None = None
    write_overlays: bool = True
    reconstruction_limits: Any = None
    metric_bundle: str | None = None   # workbench_metric_bundle_v1 file (pinned metric GLB + labelled seed)
    device: str = "cpu"


def build_run_spec(clip: wc.ClipSpec, *, mode: str, onboarding: wc.Onboarding | None = None,
                   first_frame: str | None = None, count: int | None = None,
                   roles: Sequence[str] = ("primary_neural", "frozen_baseline", "relative_geometry"),
                   artwork_kind: str = "checkerboard", run_id: str | None = None, config: Mapping | None = None) -> wc.RunSpec:
    from . import workbench_neural as wn
    onboarding = onboarding or clip.onboarding
    if onboarding is None or onboarding.selection is None:
        raise WorkbenchCLIError("clip has no reviewed selection; supply positive/negative points first")
    scheduled = wc.schedule_frames(clip, mode, onboarding, first_frame_id=first_frame, count=count)
    run_id = run_id or f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run = wc.RunSpec(run_id, clip.clip_id, mode, onboarding, scheduled, tuple(roles),
                     {k: wn.PRIMARY[k] for k in ("artifact_id", "export_sha256", "export_bytes")},
                     None, artwork_kind, dict(config or {}))
    run.validate_against(clip)
    return wc.RunSpec.from_dict(run.to_dict())


class Pipeline:
    """Executes one run with real stage code; writes only new run directories."""

    def __init__(self, ws: Workspace, clip: wc.ClipSpec, run: wc.RunSpec, options: RunOptions,
                 cancel: threading.Event | None = None, progress: Callable[[str], None] = log):
        self.ws, self.clip, self.run, self.opt = ws, clip, run, options
        self.cancel = cancel or threading.Event()
        self.progress = progress
        self.run_dir = ws.runs_dir / run.run_id
        self.stages: dict[str, dict] = {}
        self.results: list[wc.FrameResult] = []
        self.frames = {f.frame_id: f for f in clip.frames}
        self.media_identity: dict = {}
        from . import workbench_telemetry as wt
        self.tel = wt.Telemetry()
        self._last_lookup_ms: float | None = None

    def _cache_lookup(self, store, key):
        started = time.perf_counter()
        try:
            return store.lookup(key)
        except Exception as exc:  # noqa: BLE001 - a broken cache never fakes a hit
            self._stage(f"stage_cache_lookup_{len(self.stages):02d}", "failed", reason=f"{type(exc).__name__}: {exc}")
            return None
        finally:
            self._last_lookup_ms = (time.perf_counter() - started) * 1000.0

    def _cache_store(self, cache, key, payload, files=None):
        try:
            cache[1].store(key, payload, files=files or None)
        except Exception as exc:  # noqa: BLE001 - outputs remain in the run directory
            self._stage(f"stage_cache_store_{len(self.stages):02d}", "failed", reason=f"{type(exc).__name__}: {exc}")

    def _defer_cache_store(self, cache, key, payload, files=None, *, stage: str):
        """Queue a commit; it is made only after every owned resource (incl. the decoder) closed cleanly."""
        self._pending_commits = getattr(self, "_pending_commits", [])
        self._pending_commits.append((stage, cache, key, payload, dict(files or {})))

    def _settle_pending_commits(self, cleanup_failure: str | None) -> None:
        pending, self._pending_commits = getattr(self, "_pending_commits", []), []
        if not pending:
            return
        if cleanup_failure is not None:
            self._stage("cache_commits", "skipped", reason=f"owned resource cleanup failed ({cleanup_failure}); "
                                                            "no cache entry is committed for this run",
                        stages=[p[0] for p in pending])
            return
        for _stage, cache, key, payload, files in pending:
            self._cache_store(cache, key, payload, files)

    def _materialize(self, cache, record, dest: Path) -> dict[str, Path]:
        """Copy immutable cached files into a new directory and re-verify each hash."""
        dest.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, meta in record.get("files", {}).items():
            if "/" in name or name in (".", ".."):
                raise WorkbenchCLIError("cached file names must be flat")
            src = contained(cache[1].root, meta["path"])
            target = dest / name
            with src.open("rb") as reader_, target.open("xb") as writer:
                shutil.copyfileobj(reader_, writer)
            if sha256_file(target) != meta["sha256"] or target.stat().st_size != meta["size"]:
                raise WorkbenchCLIError(f"cached file {name} failed re-verification after copy")
            out[name] = target
        return out

    def _input_identity(self, frame_ids: Sequence[str], masks: "MaskStore", reader: "FrameReader") -> dict:
        """Complete, freshly re-verified input identity for cache keys."""
        frame_files = reader.verify_frames(frame_ids)
        mask_files = masks.verify(frame_ids)
        return {
            "clip_sha256": wc.canonical_sha256(self.clip.to_dict()),
            "media_sha256_verified": self.media_identity.get("sha256"),
            "frames": [[fid, self.frames[fid].source_frame_index, frame_files[fid]] for fid in frame_ids],
            "masks": [[fid, mask_files[fid]] for fid in frame_ids],
            "mask_manifest_sha256": (masks.identity() or {}).get("manifest_sha256"),
            "onboarding": self.run.onboarding.to_dict(),
            "intrinsics": self.clip.intrinsics.to_dict(),
            "dimensions": [self.clip.width, self.clip.height],
        }

    def _stage(self, name: str, status: str, **detail) -> dict:
        from .workbench_reconstruction import safe_text
        for key in ("reason", "detail"):
            if isinstance(detail.get(key), str):
                detail[key] = safe_text(detail[key], 2000)[0]
        row = {"stage": name, "status": status, **detail, "at": time.time()}
        self.stages[name] = row
        self._stage_seq = getattr(self, "_stage_seq", 0) + 1
        _write_new(self.run_dir / "stages" / f"{name}-{self._stage_seq:03d}.json", row)
        self.progress(f"{self.run.run_id} {name}: {status}" + (f" ({detail.get('reason')})" if detail.get("reason") else ""))
        return row

    def _record_error(self, scope: str, exc: BaseException) -> str:
        """Small typed error receipt (bounded traceback) for an unexpected exception."""
        import traceback
        from .workbench_reconstruction import safe_text
        reason = safe_text(f"internal_error: {type(exc).__name__}: {exc}", 600)[0]
        receipt = {"scope": scope, "type": type(exc).__name__, "message": safe_text(str(exc), 2000)[0],
                   "traceback": safe_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                                          4000)[0], "at": time.time()}
        try:
            self._error_seq = getattr(self, "_error_seq", 0) + 1
            _write_new(self.run_dir / "errors" / f"{scope}-{self._error_seq:03d}.json", receipt)
        except OSError:
            pass
        self._role_failures[scope] = reason
        try:
            self._stage(scope, "failed", reason=reason)
        except OSError:
            pass
        return reason

    def _run_role(self, role: str, fn, *args) -> None:
        """Role work never aborts the run: ordinary exceptions become failed
        role evidence; KeyboardInterrupt/SystemExit still propagate."""
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            self._record_error(role, exc)

    def execute(self) -> dict:
        attestation = self.run.validate_against(self.clip)
        if self.opt.mask_set is None and isinstance(self.run.config.get("mask_set"), str):
            self.opt.mask_set = self.run.config["mask_set"]
        self._role_failures: dict[str, str] = {}
        budget_reason = None
        try:
            self.ws.require_budget()
        except WorkbenchCLIError as exc:
            budget_reason = f"budget_blocked: {exc}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        _write_new(self.run_dir / "run.json", {"run": self.run.to_dict(), "clip_id": self.clip.clip_id,
                                               "prefix_attestation": attestation, "created_at": time.time()})
        if budget_reason:
            # Only small receipts are written; no decoding or derived outputs.
            self._stage("budget", "blocked", reason=budget_reason, budget=self.ws.budget())
            return self._finish(blocked_reason=budget_reason)
        reader = None
        blocked_reason = None
        try:
            reader = FrameReader(self.ws, self.clip)
            masks = MaskStore(self.ws, self.clip.clip_id, self.opt.mask_set)
            cache, cache_reason = stage_cache(self.ws)
            if cache is None:
                self._stage("stage_cache", "unavailable", reason=cache_reason)
            self.media_identity: dict = {}
            try:
                with self.tel.measure("source_verification"):
                    self.media_identity = reader.verify_media()
                self._stage("source", "completed", **self.media_identity)
            except Exception as exc:  # noqa: BLE001 - no role runs; finalized below after cleanup
                self._stage("source", "failed", reason=f"{type(exc).__name__}: {exc}")
                blocked_reason = f"source_unverified: {exc}"
            if blocked_reason is None:
                self._stage("masks", "completed" if masks.available else "blocked",
                            reason=None if masks.available else "observed masks unavailable: run `segment` (SAM 2.1 tap-to-mask) "
                                                                "or `import masks`",
                            identity=masks.identity())
                if "relative_geometry" in self.run.roles:
                    self._run_role("relative_geometry", self._relative, reader, masks, attestation, cache)
                for role in ("primary_neural", "frozen_baseline"):
                    if role in self.run.roles:
                        self._run_role(role, self._neural, role, reader, masks, cache)
        except Exception as exc:  # noqa: BLE001 - finalize evidence for every scheduled frame
            self._record_error("pipeline", exc)
        finally:
            close_error = None
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:  # noqa: BLE001 - recorded; rows are kept; no cache commit
                    close_error = f"{type(exc).__name__}: {exc}"[:500]
            if close_error is not None:
                self._stage("reader_cleanup", "failed", reason=f"decoder close failed: {close_error}")
            self._settle_pending_commits(close_error)
        # The single final report is written only after every owned resource closed and commits settled.
        return self._finish(blocked_reason=blocked_reason)

    # relative geometry ---------------------------------------------------
    def _relative(self, reader, masks, attestation, cache):
        from . import workbench_reconstruction as wr
        from . import workbench_geometry as wg
        role = "relative_geometry"
        identity = {"role": role, "tracker": TRACKER_VERSION, "quality_claim": False}
        if self.cancel.is_set():
            return self._blocked_role(role, identity, "cancelled")
        if self.clip.intrinsics.status == "unavailable":
            self._stage("reconstruction", "blocked", reason="calibration_unavailable")
            return self._blocked_role(role, identity, "calibration_unavailable: measured K required")
        K = self.clip.intrinsics.matrix()
        prefix_ids = list(self.run.onboarding.keyframe_ids)
        prefix_masks = {fid: masks.entry(fid) for fid in prefix_ids}
        if not masks.available or any(v is None for v in prefix_masks.values()):
            self._stage("reconstruction", "blocked", reason="observed prefix masks unavailable")
            return self._blocked_role(role, identity, "surface_package_unavailable: observed prefix masks unavailable")
        pre = wr.preflight(self.opt.colmap)
        if not pre.get("available"):
            self._stage("reconstruction", "blocked", reason=pre.get("reason"))
            return self._blocked_role(role, identity, f"surface_package_unavailable: {pre.get('reason')}")
        attestation = dict(attestation, first_scored_source_frame_index=self.frames[self.run.scheduled_frame_ids[0]].source_frame_index)
        limits = self.opt.reconstruction_limits or wr.Limits(baseline_bytes=self.ws.budget()["derived_bytes"])
        out = self.ws.derived_dir / "reconstructions" / f"{self.clip.clip_id}-{self.run.run_id}"
        package_dir = None
        key = None
        if cache is not None:
            mod, store = cache
            try:
                inputs = self._input_identity(prefix_ids, masks, reader)
            except WorkbenchCLIError as exc:
                self._stage("reconstruction", "failed", reason=f"input_reverification_failed: {exc}")
                return self._blocked_role(role, identity, f"input_reverification_failed: {exc}", pose_state="unavailable")
            key = mod.stage_key("reconstruction", inputs,
                                {"limits": {k: v for k, v in limits.to_dict().items() if k != "baseline_bytes"},
                                 "artwork": self.run.artwork_kind, "attestation": attestation},
                                {"colmap": pre.get("version"), "colmap_executable": pre.get("executable"),
                                 **_module_versions("workbench_reconstruction", "workbench_geometry",
                                                    "workbench_contract", "workbench_cli")})
            hit = self._cache_lookup(store, key)
            if hit:
                try:
                    self._materialize(cache, hit, out / "package")
                    package_dir = out / "package"
                    self.tel.cache_event("reconstruction", "cache_hit", self._last_lookup_ms)
                    self.tel.reuse_cached("reconstruction", hit["payload"].get("telemetry"))
                    self._stage("reconstruction", "cache_hit", key=key, package_dir=str(package_dir))
                except (OSError, WorkbenchCLIError) as exc:
                    self._stage("reconstruction_cache", "failed", reason=f"materialize failed: {exc}")
                    shutil.rmtree(out, ignore_errors=True)
                    package_dir = None
        if package_dir is None:
            # decode the causal prefix to scratch PNG (bounded <= 24 frames)
            from .vision import cv2
            scratch = self.ws.scratch_dir / self.run.run_id / "prefix"
            scratch.mkdir(parents=True, exist_ok=False)
            prefix = []
            for ordinal, fid in enumerate(prefix_ids):
                rgb, ident = reader.read(self.frames[fid])
                path = scratch / f"p{ordinal:03d}.png"
                cv2.imwrite(str(path), rgb[:, :, ::-1])
                self.ws.require_budget()
                prefix.append({"frame_id": fid, "image_path": str(path), "image_sha256": sha256_file(path),
                               "content_sha256": ident["image_sha256"], "width": self.clip.width,
                               "height": self.clip.height, "source_frame_index": self.frames[fid].source_frame_index,
                               "timestamp_s": self.frames[fid].timestamp_s if attestation["prefix_bound_attested"] else None})
            mask_refs = {fid: {"path": row["abs_path"], "sha256": row["sha256"], "width": row["width"],
                               "height": row["height"]} for fid, row in prefix_masks.items()}
            if key is not None:
                self.tel.cache_event("reconstruction", "miss", self._last_lookup_ms)
            with self.tel.measure("reconstruction") as recon_time:
                result = wr.reconstruct(prefix, mask_refs, K, output_dir=out, limits=limits,
                                        cancel=self.cancel, colmap=self.opt.colmap, clip_id=self.clip.clip_id,
                                        package_id=f"{self.clip.clip_id}-{self.run.run_id}",
                                        selection=self.run.onboarding.selection, artwork_kind=self.run.artwork_kind,
                                        prefix_attestation=attestation, preflight_result=pre)
            package_dir = Path(result["package_dir"])
            status = "completed" if result["status"] == "available" else "failed"
            self._stage("reconstruction", status, reason=result.get("reason"), package_dir=str(package_dir))
            if result["status"] == "available" and key is not None:
                files = {p.name: p for p in sorted(package_dir.iterdir()) if p.is_file()}
                self._defer_cache_store(cache, key, {"package_id": result["package"].package_id,
                                                     "file_names": sorted(files),
                                                     "telemetry": {"reconstruction_ms": recon_time["ms"]}}, files,
                                        stage="reconstruction")
        self._track(reader, masks, package_dir, identity, cache)

    # portable-boundary helpers -------------------------------------------
    def _boundary_input(self, reader, masks, frame):
        """FrameInputV1 for one frame. A decoder/read failure becomes an explicit RGB-unavailable packet
        (counted, invalidates tracking); a mask failure takes the safe missing-mask path."""
        from . import workbench_frame_boundary as fb
        rgb, ident, rgb_reason = None, None, None
        try:
            rgb, ident = reader.read(frame)
        except Exception as exc:  # noqa: BLE001
            rgb_reason = f"{type(exc).__name__}: {exc}"[:400]
        mask, mident, mask_reason = None, None, None
        try:
            mask, mident = masks.load(frame.frame_id, (self.clip.height, self.clip.width))
            if mask is None:
                mask_reason = "no observed mask for this frame"
        except Exception as exc:  # noqa: BLE001
            mask, mident, mask_reason = None, None, f"{type(exc).__name__}: {exc}"[:400]
        try:
            return fb.FrameInputV1.from_reader(frame, rgb, ident, mask, mident, rgb_unavailable_reason=rgb_reason,
                                              mask_unavailable_reason=mask_reason)
        except fb.BoundaryRejected as exc:
            return fb.FrameInputV1.from_reader(frame, None, None, None, None,
                                              rgb_unavailable_reason=f"reader packet rejected: {exc}"[:400],
                                              mask_unavailable_reason=mask_reason)

    def _adopted_lookup(self, store, key, role, provenance, *, context=None):
        """Lookup for boundary-adopted tracking stages -> (record | None, invalid_reason | None).

        The WHOLE entry is validated before any materialization or append: provenance, the exact
        scheduled rows of this role, every row's manifest clip/frame/index/timestamp/media time,
        role units, mask identity form, one constant backend identity carrying the pinned head/base
        and the current admission contract (and, with ``context``, the bound bundle/seed/origin/K or
        package identity), and overlay evidence at exactly ``evidence/<role>/<frame>.png`` matching
        the entry's file set and the current ``write_overlays``. Anything else fails closed."""
        started = time.perf_counter()
        try:
            record = store.lookup(key)
        except Exception as exc:  # noqa: BLE001
            return None, f"cache_entry_invalid: lookup failed: {type(exc).__name__}: {exc}"
        finally:
            self._last_lookup_ms = (time.perf_counter() - started) * 1000.0
        if record is None:
            return None, None
        try:
            reason = self._validate_adopted_entry(record, role, provenance, context)
            if reason is None:
                reason = self._check_lookup_context(role, context)
        except Exception as exc:  # noqa: BLE001 - malformed entries never reach replay
            reason = f"malformed entry ({type(exc).__name__}: {exc})"
        return (None, f"cache_entry_invalid: {reason}"[:600]) if reason else (record, None)

    def _check_lookup_context(self, role, context) -> str | None:
        """An adopted hit requires the validated boundary context of THIS run/role; without it the entry
        is refused consistently (the basic checks above still report their own specific reasons first)."""
        from . import workbench_frame_boundary as fb
        if context is None:
            return "the validated boundary context is required for an adopted cache hit"
        if (type(context) is not fb.FrameContextV1 or context.role != role
                or context.clip_sha256 != wc.canonical_sha256(self.clip.to_dict())
                or list(context.scheduled_frame_ids) != list(self.run.scheduled_frame_ids)):
            return "boundary context does not belong to this run/role"
        return None

    NEURAL_BUNDLE_IDENTITY_FIELDS = ("principal_point_origin", "initialization", "seed_frame_id", "seed_target_frame_id",
                                     "glb_sha256", "metric_units", "effective_K")

    def _expected_identity_checks(self, role, identities, context) -> str | None:
        from . import workbench_neural as wn
        if len({wc.canonical_sha256(i) for i in identities}) != 1:
            return "backend identity varies across cached rows"
        ident = identities[0]
        if role == "relative_geometry":
            if context is not None:
                b = context.binding
                expected = {"role": role, "tracker": TRACKER_VERSION, "quality_claim": False,
                            "package_id": b["package_id"], "package_sha256": b["surface_json_sha256"]}
                if ident != expected:
                    return "relative backend identity differs from the bound package/tracker"
            elif ident.get("role") != role or ident.get("tracker") != TRACKER_VERSION:
                return "relative backend identity differs from the tracker"
            return None
        if (ident.get("base_checkpoint_sha256") != wn.PRIMARY["base_checkpoint_sha256"]
                or ident.get("quality_claim") is not False
                or (self.opt.admission is not None
                    and ident.get("admission_contract_sha256") != self.opt.admission.body.get("contract_sha256"))):
            return "cached base/admission/claim identity differs from the current pins"
        if role == "primary_neural" and (ident.get("installed_head_sha256") != wn.PRIMARY["export_sha256"]
                                         or ident.get("export_sha256") != wn.PRIMARY["export_sha256"]):
            return "cached installed/export head differs from the pinned primary"
        if role == "frozen_baseline" and (ident.get("installed_head_sha256") is not None
                                          or ident.get("head") != "frozen_base_pose_head"):
            return "cached baseline identity carries an installed head"
        bundle = None if context is None else context.binding.get("bundle")
        if bundle is not None:
            # Every real metric backend emits these; deleting a field fails exactly like changing it.
            missing = [k for k in self.NEURAL_BUNDLE_IDENTITY_FIELDS if k not in ident]
            if missing:
                return f"cached identity lacks required bound-bundle fields {missing}"
            expected = {"principal_point_origin": bundle["principal_point_origin"], "initialization": bundle["seed_label"],
                        "seed_frame_id": bundle["seed_frame_id"], "glb_sha256": bundle["glb_sha256"],
                        "seed_target_frame_id": context.binding["seed_target_frame_id"], "metric_units": "metres"}
            if any(ident[k] != v for k, v in expected.items()):
                return "cached bundle/seed/origin/units identity differs from the bound bundle"
            K = ident["effective_K"]
            if (not isinstance(K, list) or len(K) != 9 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                               or not np.isfinite(float(x)) for x in K)):
                return "cached effective K is not 9 finite numbers"
            expected_K = wn.native_K(np.asarray(context.K, float), bundle["principal_point_origin"]).ravel().tolist()
            if [float(x) for x in K] != [float(x) for x in expected_K]:
                return "cached effective K differs from the current calibration/origin"
        return None

    def _validate_adopted_entry(self, record, role, provenance, context) -> str | None:
        if not isinstance(record, Mapping):
            return "record is not an object"
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or payload.get("execution") != {**provenance, "computed_by": "fresh_boundary_execution"}:
            return "execution provenance is not this adopted boundary path"
        rows = payload.get("results")
        scheduled = list(self.run.scheduled_frame_ids)
        if (not isinstance(rows, list) or any(not isinstance(r, Mapping) for r in rows)
                or [r.get("frame_id") for r in rows] != scheduled or any(r.get("backend_role") != role for r in rows)):
            return "cached rows do not cover exactly the scheduled frames of this role"
        files = record.get("files") or {}
        if not isinstance(files, Mapping):
            return "file table is malformed"
        units = "object_unit" if role == "relative_geometry" else "metric"
        overlays, parsed = set(), []
        for raw in rows:
            row = wc.FrameResult.from_dict(raw)
            frame = self.frames[row.frame_id]
            if row.clip_id != self.clip.clip_id or row.source_frame_index != frame.source_frame_index:
                return f"{row.frame_id}: clip/source index differs from the manifest"
            if row.timestamp_s != frame.timestamp_s:
                return f"{row.frame_id}: timestamp differs from the manifest clock"
            if frame.media_time_s is not None and row.media_time_s != frame.media_time_s:
                return f"{row.frame_id}: media time differs from the manifest"
            if frame.media_time_s is None and role != "relative_geometry" and row.media_time_s is not None:
                return f"{row.frame_id}: neural rows carry the manifest media time only"
            if row.pose_units not in (None, units):
                return f"{row.frame_id}: pose units {row.pose_units} are not {units}"
            if row.mask_state == "observed" and (row.mask_frame_id != row.frame_id or row.mask_sha256 is None):
                return f"{row.frame_id}: observed mask identity is not this frame's"
            evidence = row.evidence or {}
            if evidence.get("execution_provenance") != "fresh_boundary_execution":
                return f"{row.frame_id}: row was not computed through the adopted boundary"
            overlay = evidence.get("overlay")
            if overlay is not None:
                if not self.opt.write_overlays or overlay != f"evidence/{role}/{row.frame_id}.png":
                    return f"{row.frame_id}: overlay evidence path {overlay!r} is not this role/frame's"
                if row.render_state != "rendered" or f"{row.frame_id}.png" not in files:
                    return f"{row.frame_id}: overlay evidence without a rendered row and its cached file"
                overlays.add(f"{row.frame_id}.png")
            elif self.opt.write_overlays and row.render_state == "rendered":
                return f"{row.frame_id}: rendered row lacks its overlay artifact"
            if "artifact_write_error" in evidence:
                return f"{row.frame_id}: entries with artifact write errors are never valid"
            parsed.append(row)
        if set(files) != overlays:
            return "cached file set differs from the rows' overlay evidence"
        for name, meta in files.items():
            if (not isinstance(meta, Mapping) or not isinstance(meta.get("sha256"), str)
                    or type(meta.get("size")) is not int or meta["size"] <= 0):
                return f"cached file {name} lacks integrity metadata"
        return self._expected_identity_checks(role, [r.backend_identity for r in parsed], context)

    def _replay_adopted_hit(self, role, record, cache) -> str | None:
        """Materialize (each file re-hashed after copy) and append an entry already validated by
        ``_adopted_lookup``; no boundary/model construction. Returns an invalid reason or None."""
        try:
            materialized = self._materialize(cache, record, self.run_dir / "evidence" / role)
            if set(materialized) != set(record.get("files") or {}):
                return "cache_entry_invalid: materialized files differ from the entry"
            path_id = record["payload"]["execution"]["execution_path"]
            rows = []
            for row in record["payload"]["results"]:
                evidence = dict(row.get("evidence") or {})
                evidence.update(execution_provenance="cache_replay_of_boundary_computed_rows",
                                computed_execution_path=path_id, cached_source_run_id=row.get("run_id"))
                rows.append(wc.FrameResult.from_dict(dict(row, run_id=self.run.run_id, evidence=evidence)))
        except (OSError, WorkbenchCLIError, wc.ContractError, KeyError, TypeError, ValueError) as exc:
            return f"cache_entry_invalid: {type(exc).__name__}: {exc}"[:600]
        self.results.extend(rows)
        return None

    def _fail_closed(self, role, identity, stage, reason):
        self._stage(stage, "failed", reason=reason)
        for fid in self.run.scheduled_frame_ids:
            self.results.append(wc.blocked_result(self.run.run_id, self.clip, self.frames[fid], role, identity, reason,
                                                  pose_state="unavailable", mask_state="unavailable"))

    def _track(self, reader, masks, package_dir, identity, cache):
        import dataclasses
        from . import workbench_frame_boundary as fb
        from . import workbench_geometry as wg
        from .vision import cv2
        role = "relative_geometry"
        try:   # authenticated package bytes, re-verified exactly as the loader does
            proof = fb.RelativePackageProof.from_package_dir(package_dir)
            pkg, _arrays, _digests = proof.verify()
        except (fb.BoundaryError, wc.ContractError, OSError) as exc:
            self._stage("relative_tracking", "failed", reason=f"surface_package_unverified: {exc}")
            return self._blocked_role(role, identity, f"surface_package_unverified: {exc}", pose_state="unavailable")
        if pkg.status != "available":
            return self._blocked_role(role, identity, f"surface_package_unavailable: {pkg.unavailable_reason}",
                                      pose_state="unavailable")
        identity = {**identity, "package_id": pkg.package_id, "package_sha256": proof.surface_json_sha256}
        if self.run.artwork_kind == "imported":
            return self._blocked_role(role, identity,
                                      "imported_artwork_unavailable: no imported artwork file is bound to this run")
        tracker = wg.RelativeTracker()
        try:
            binding = fb.relative_binding(proof, artwork_kind=self.run.artwork_kind)
            context = fb.FrameContextV1.build(self.clip, self.run, role, binding)
            provenance = boundary_provenance(context)
        except (fb.BoundaryError, wc.ContractError) as exc:
            tracker.close()
            self._stage("relative_tracking", "blocked", reason=f"boundary_context_unavailable: {exc}")
            return self._blocked_role(role, identity, f"boundary_context_unavailable: {exc}", pose_state="unavailable")
        key = None
        if cache is not None:
            mod, store = cache
            try:
                inputs = self._input_identity(list(self.run.scheduled_frame_ids), masks, reader)
            except WorkbenchCLIError as exc:
                tracker.close()
                self._stage("relative_tracking", "failed", reason=f"input_reverification_failed: {exc}")
                return self._blocked_role(role, identity, f"input_reverification_failed: {exc}", pose_state="unavailable")
            inputs["package_sha256"] = identity["package_sha256"]
            inputs["boundary"] = {"context_digest": provenance["context_digest"], "binding_sha256": provenance["binding_sha256"]}
            key = mod.stage_key("relative_tracking", inputs,
                                {"tracker": tracker.cfg, "artwork": self.run.artwork_kind,
                                 "write_overlays": self.opt.write_overlays},
                                {**_module_versions("workbench_geometry", "workbench_cli", "workbench_contract",
                                                    "workbench_frame_boundary", "workbench_reconstruction"),
                                 "execution": provenance})
            hit, invalid = self._adopted_lookup(store, key, role, provenance, context=context)
            if invalid is None and hit is not None:
                tracker.close()   # a genuine adopted hit never constructs the boundary
                invalid = self._replay_adopted_hit(role, hit, cache)
                if invalid is None:
                    self.tel.cache_event("relative_tracking", "cache_hit", self._last_lookup_ms)
                    self.tel.reuse_cached("relative_tracking", hit["payload"].get("telemetry"))
                    self._stage("relative_tracking", "cache_hit", key=key, frames=len(hit["payload"]["results"]),
                                execution={**provenance, "replayed": "cache_replay_of_boundary_computed_rows"})
                    return
            if invalid is not None:
                tracker.close()
                return self._fail_closed(role, identity, "relative_tracking", invalid)
            self.tel.cache_event("relative_tracking", "miss", self._last_lookup_ms)
        boundary = fb.FrameBoundary()
        with self.tel.measure("relative_initialization") as init_time:
            try:
                boundary.initialize(context, fb.RelativeTrackerAdapter(tracker, proof))   # closes the tracker on failure
            except Exception as exc:  # noqa: BLE001
                self._stage("relative_tracking", "failed", reason=f"boundary_initialization_failed: {exc}",
                            cleanup_errors=list(boundary.cleanup_errors))
                return self._blocked_role(role, identity, f"boundary_initialization_failed: {exc}", pose_state="unavailable")
        identity = boundary.identity()["backend_identity"]
        frame_timings, produced, artifact_errors, files = {}, [], [], {}
        try:
            for fid in self.run.scheduled_frame_ids:
                frame = self.frames[fid]
                if self.cancel.is_set():
                    produced.append(wc.blocked_result(self.run.run_id, self.clip, frame, role, identity, "cancelled"))
                    continue
                t_frame = time.perf_counter()
                out = boundary.process(self._boundary_input(reader, masks, frame))
                row, evidence = out.result, dict(out.result.evidence)
                self._write_overlay(role, fid, out.overlay_rgba, evidence, artifact_errors, files)
                # The adapter tracks and renders inside one boundary call: only the combined time is measured.
                timing = {"tracking_ms": None, "rendering_ms": None, "boundary_process_ms": out.process_ms,
                          "end_to_end_latency_ms": (time.perf_counter() - t_frame) * 1000.0}
                self.tel.frame(role, fid, **timing)
                frame_timings[fid] = timing
                evidence["execution_provenance"] = "fresh_boundary_execution"
                produced.append(dataclasses.replace(row, metrics={**row.metrics, **timing}, evidence=evidence))
        finally:
            # Rows already produced are preserved even if the loop raised.
            self.results.extend(produced)
            boundary.close()
            cleanup = list(boundary.cleanup_errors)
            if cleanup:
                self._stage("relative_tracking_cleanup", "failed", errors=cleanup)
        self._stage("relative_tracking", "cancelled" if self.cancel.is_set() else "completed", frames=len(produced),
                    execution={**provenance, "computed_by": "fresh_boundary_execution"})
        if artifact_errors:
            self._stage("relative_tracking_artifacts", "failed",
                        reason=f"{len(artifact_errors)} overlay PNG write(s) failed; cache commit skipped",
                        frame_ids=[e["frame_id"] for e in artifact_errors], errors=artifact_errors[:20])
        if (cache is not None and key is not None and not self.cancel.is_set() and not artifact_errors and not cleanup):
            self._defer_cache_store(cache, key, {"results": [r.to_dict() for r in produced],
                                                 "execution": {**provenance, "computed_by": "fresh_boundary_execution"},
                                                 "telemetry": {"initialization_ms": init_time["ms"],
                                                               "frames": frame_timings}}, files,
                                    stage="relative_tracking")

    def _write_overlay(self, role, fid, overlay_rgba, evidence, artifact_errors, files) -> None:
        """Shared artifact policy: write only this frame's current overlay when write_overlays is on
        (neural: atomic ``write_png_rgba``; relative: its existing cv2 encoder, exclusive). Any
        mkdir/convert/encode/write/collision error keeps the computed row and its pose/render state,
        records artifact_write_error and blocks the cache commit. Never raises."""
        from . import workbench_image_io as wio
        if overlay_rgba is None:
            return
        if not self.opt.write_overlays:
            evidence["overlay_artifact"] = "not_written: write_overlays disabled"
            return
        path = self.run_dir / "evidence" / role / f"{fid}.png"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if role == "relative_geometry":
                self._write_relative_png(path, overlay_rgba)
            else:
                wio.write_png_rgba(path, overlay_rgba)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"[:500]
            artifact_errors.append({"frame_id": fid, "reason": reason})
            evidence["artifact_write_error"] = reason
        else:
            files[path.name] = path
            evidence["overlay"] = f"evidence/{role}/{fid}.png"

    @staticmethod
    def _write_relative_png(path: Path, rgba) -> None:
        """The relative stage's existing encoder (cv2 BGRA PNG), made exclusive: never overwrites,
        and removes only a partial file it created itself before re-raising."""
        from .vision import cv2
        if os.path.lexists(path):
            raise FileExistsError(f"overlay {path.name} already exists; evidence is never overwritten")
        try:
            if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
                raise OSError("cv2.imwrite returned False")
        except BaseException:
            if os.path.lexists(path):
                os.unlink(path)
            raise

    # neural --------------------------------------------------------------
    def _neural_block(self, role, reason, **detail):
        from . import workbench_neural as wn
        self._stage(role, "blocked", reason=reason, **detail)
        self.results.extend(wn.blocked_results(self.run.run_id, self.clip, self.run.scheduled_frame_ids, role, reason))

    def _neural_plan(self):
        """Bundle, seed target, causal warm-up frames and the shared artwork (built once)."""
        from . import workbench_neural as wn
        if hasattr(self, "_neural_plan_cache"):
            return self._neural_plan_cache
        plan = {"bundle": None, "seed_target": None, "warmup": [], "artwork": None, "artwork_reason": None,
                "error": None}
        try:
            bundle = wn.MetricBundle.from_file(Path(self.opt.metric_bundle))
            plan["bundle"] = bundle
            order = [f.frame_id for f in self.clip.frames]
            if bundle.seed_frame_id not in order:
                raise WorkbenchCLIError(f"seed frame {bundle.seed_frame_id} is not a clip frame")
            seed_pos = order.index(bundle.seed_frame_id)
            if seed_pos + 1 >= len(order):
                raise WorkbenchCLIError("no frame follows the seed frame")
            first = order.index(self.run.scheduled_frame_ids[0])
            if first <= seed_pos:
                raise WorkbenchCLIError("scheduled frames start at or before the seed frame")
            onboarding_last = max(order.index(k) for k in self.run.onboarding.keyframe_ids)
            if seed_pos > onboarding_last:
                raise WorkbenchCLIError("seed frame lies after the onboarding prefix")
            plan["seed_target"] = order[seed_pos + 1]
            # Frames between the seed and the first scheduled frame are replayed
            # causally (not scored) so a focused start never jumps the seed forward.
            plan["warmup"] = order[seed_pos + 1:first]
        except Exception as exc:  # noqa: BLE001
            plan["error"] = f"neural_plan_unavailable: {type(exc).__name__}: {exc}"
            self._neural_plan_cache = plan
            return plan
        try:
            sel = self.run.onboarding.selection
            masks = MaskStore(self.ws, self.clip.clip_id, self.opt.mask_set)
            sel_mask, _ = masks.load(sel.frame_id, (self.clip.height, self.clip.width))
            artwork = wn.NeuralArtwork(plan["bundle"], sel, self.run.artwork_kind, clip_K=self.clip.intrinsics.K,
                                       width=self.clip.width, height=self.clip.height, selection_mask=sel_mask)
            plan["artwork"] = artwork
            _write_new(self.run_dir / "neural-artwork.json", {**artwork.record, "record_sha256": artwork.sha256})
        except Exception as exc:  # noqa: BLE001 - rows stay unrendered, never fake-rendered
            plan["artwork_reason"] = f"artwork_unavailable: {type(exc).__name__}: {exc}"
            self._stage("neural_artwork", "unavailable", reason=plan["artwork_reason"])
        self._neural_plan_cache = plan
        return plan

    def _neural(self, role, reader, masks, cache):
        from . import workbench_frame_boundary as fb
        from . import workbench_neural as wn
        from . import workbench_image_io as wio
        import dataclasses
        pre = wn.neural_preflight(head_path=Path(self.opt.head_path) if self.opt.head_path else None)
        if self.clip.intrinsics.status == "unavailable":
            return self._neural_block(role, "calibration_unavailable: the metric neural profile needs K")
        backend = self.opt.neural_backends.get(role)
        admission = self.opt.admission
        admitted = admission is not None and getattr(admission, "admitted", False)
        if not admitted or (backend is None and not self.opt.metric_bundle):
            reason = wn.neural_stage_status(pre)
            if not admitted:
                reason = f"{reason}; serving admission not verified"
                if admission is not None:
                    reason += f" ({'; '.join(admission.reasons[:3])})"
            else:
                reason = f"{reason}; no metric bundle (--metric-bundle) for the admitted backend"
            return self._neural_block(role, reason, preflight=pre)
        plan = {"bundle": None, "seed_target": None, "warmup": [], "artwork": None, "artwork_reason": None}
        if self.opt.metric_bundle:
            plan = self._neural_plan()
            if plan.get("error"):
                return self._neural_block(role, plan["error"])
        family = self.opt.family or (plan["bundle"].family if plan["bundle"] else None)
        artwork = plan["artwork"]
        # Canonical portable-boundary context (seed plan, K, order, complete bundle binding), before any cache work.
        try:
            binding = fb.neural_binding(self.clip, self.run, family=family or "", bundle=plan["bundle"])
            context = fb.FrameContextV1.build(self.clip, self.run, role, binding)
            if list(context.warmup_frame_ids) != list(plan["warmup"]) or binding["seed_target_frame_id"] != plan["seed_target"]:
                raise fb.BoundaryRejected("boundary seed plan differs from the pipeline seed plan")
            provenance = boundary_provenance(context)
        except (fb.BoundaryError, wc.ContractError) as exc:
            return self._neural_block(role, f"boundary_context_unavailable: {exc}")
        identity_fallback = wn.primary_identity() if role == "primary_neural" else wn.baseline_identity()
        key = None
        if cache is not None and backend is None:
            mod, store = cache
            try:
                frames = list(plan["warmup"]) + list(self.run.scheduled_frame_ids)
                inputs = self._input_identity(frames, masks, reader)
                inputs["warmup_frame_ids"] = list(plan["warmup"])
                inputs["metric_bundle_sha256"] = sha256_file(Path(self.opt.metric_bundle))
                inputs["artwork_record_sha256"] = artwork.sha256 if artwork else plan["artwork_reason"]
                inputs["boundary"] = {"context_digest": provenance["context_digest"],
                                      "binding_sha256": provenance["binding_sha256"]}
                if role == "primary_neural":
                    if not self.opt.head_path:
                        raise WorkbenchCLIError("primary head path missing")
                    model = sha256_file(Path(self.opt.head_path))
                    if model != wn.PRIMARY["export_sha256"]:
                        raise WorkbenchCLIError("head.pt bytes differ from the pinned export")
                else:
                    model = sha256_file(self.ws.repo_root / wn.BASE_CHECKPOINT_RELATIVE)
                    if model != wn.PRIMARY["base_checkpoint_sha256"]:
                        raise WorkbenchCLIError("base checkpoint bytes differ from the pin")
            except (WorkbenchCLIError, OSError) as exc:
                return self._neural_block(role, f"input_reverification_failed: {exc}")
            key = mod.stage_key(role, inputs, {"device": self.opt.device, "write_overlays": self.opt.write_overlays},
                                {"model_sha256_verified": model, "base_checkpoint_sha256": wn.PRIMARY["base_checkpoint_sha256"],
                                 "admission": dict(admission.body), "execution": provenance,
                                 **_module_versions("workbench_neural", "workbench_cli", "workbench_contract",
                                                    "workbench_geometry", "workbench_render", "workbench_reconstruction", "quality_gotrack",
                                                    "quality_gotrack_inference", "quality_render_stability", "workbench_image_io",
                                                    "workbench_frame_boundary")})
            hit, invalid = self._adopted_lookup(store, key, role, provenance, context=context)
            if invalid is None and hit is not None:
                invalid = self._replay_adopted_hit(role, hit, cache)   # no model, GPU stage or boundary construction
                if invalid is None:
                    self.tel.cache_event(role, "cache_hit", self._last_lookup_ms)
                    self.tel.reuse_cached(role, hit["payload"].get("telemetry"))
                    self._stage(role, "cache_hit", key=key, frames=len(hit["payload"]["results"]),
                                warmup=hit["payload"].get("warmup"),
                                execution={**provenance, "replayed": "cache_replay_of_boundary_computed_rows"})
                    return
            if invalid is not None:
                return self._fail_closed(role, identity_fallback, role, invalid)
            self.tel.cache_event(role, "miss", self._last_lookup_ms)
        from contextlib import ExitStack
        from . import workbench_telemetry as wt
        render_ms = {"v": None}

        def timed_render(pose_m, observed_mask, clip_K):
            t_render = time.perf_counter()
            try:
                return artwork.render(pose_m, observed_mask, clip_K)
            finally:
                render_ms["v"] = (time.perf_counter() - t_render) * 1000.0
        if self.cancel.is_set():
            return self._neural_block(role, "cancelled")
        stack = ExitStack()
        try:
            # One neural/GPU stage at a time (shared with SAM segmentation).
            stack.enter_context(wt.gpu_stage(role))
        except RuntimeError as exc:
            return self._neural_block(role, str(exc))
        pending: list = []
        release_error = None
        try:
            if backend is None:
                if self.cancel.is_set():
                    return self._neural_block(role, "cancelled")
                try:
                    self._stage(f"{role}_load", "running", reason=f"loading {role} network on {self.opt.device}",
                                limits={"initialization": plan["bundle"].seed_label, "seed_target": plan["seed_target"],
                                        "warmup_frames": len(plan["warmup"])})
                    with self.tel.measure(f"{role}_initialization", sync_gpu=True):
                        backend = wn.GoTrackMetricBackend(role, plan["bundle"], device=self.opt.device,
                                                          head_path=Path(self.opt.head_path) if self.opt.head_path else None)
                except Exception as exc:  # noqa: BLE001 - missing runtime/weights are reported, never faked
                    return self._neural_block(role, f"neural_backend_unavailable: {type(exc).__name__}: {exc}", preflight=pre)
            if plan["seed_target"] is not None and hasattr(backend, "seed_target_frame_id"):
                backend.seed_target_frame_id = plan["seed_target"]
            try:
                session = wn.NeuralSession(role, backend, admission, run_id=self.run.run_id, clip=self.clip,
                                           family=family or "", renderer=timed_render if artwork else None)
            except Exception as exc:  # noqa: BLE001
                close = getattr(backend, "close", None)
                if callable(close) and backend is not self.opt.neural_backends.get(role):
                    close()
                return self._neural_block(role, f"neural_session_unavailable: {exc}")
            # The boundary owns the session (and so its backend) from here on, including on refusal.
            boundary = fb.FrameBoundary()
            try:
                boundary.initialize(context, fb.NeuralSessionAdapter(session))
            except Exception as exc:  # noqa: BLE001
                self._stage(f"{role}_boundary", "failed", reason=f"{type(exc).__name__}: {exc}",
                            cleanup_errors=list(boundary.cleanup_errors))
                return self._neural_block(role, f"boundary_initialization_failed: {type(exc).__name__}: {exc}")
            identity = boundary.identity()["backend_identity"]
            produced, warm, frame_timings = [], {}, {}
            files = {}
            artifact_errors: list[dict] = []
            try:
                for fid in plan["warmup"]:
                    if self.cancel.is_set():
                        break
                    out = boundary.process(self._boundary_input(reader, masks, self.frames[fid]))
                    warm[out.result.pose_state] = warm.get(out.result.pose_state, 0) + 1
                if plan["warmup"]:
                    self._stage(f"{role}_warmup", "completed", frames=len(plan["warmup"]), pose_states=warm,
                                note="causal replay from the seed frame through the boundary; not scored")
                for fid in self.run.scheduled_frame_ids:
                    frame = self.frames[fid]
                    if self.cancel.is_set():
                        produced.append(wc.blocked_result(self.run.run_id, self.clip, frame, role, identity, "cancelled"))
                        continue
                    t_frame = time.perf_counter()
                    inp = self._boundary_input(reader, masks, frame)
                    render_ms["v"] = None
                    wt.gpu_synchronize()
                    t_proc = time.perf_counter()
                    out = boundary.process(inp)
                    wt.gpu_synchronize()
                    process_ms = (time.perf_counter() - t_proc) * 1000.0
                    row = out.result
                    evidence = dict(row.evidence)
                    self._write_overlay(role, fid, out.overlay_rgba, evidence, artifact_errors, files)
                    if row.render_state != "rendered" and row.pose is not None and artwork is None:
                        evidence["render"] = plan["artwork_reason"]
                    # tracking_ms = boundary call (incl. its validation) minus the measured render time.
                    timing = {"tracking_ms": process_ms - (render_ms["v"] or 0.0), "rendering_ms": render_ms["v"],
                              "boundary_process_ms": out.process_ms,
                              "end_to_end_latency_ms": (time.perf_counter() - t_frame) * 1000.0}
                    self.tel.frame(role, fid, **timing)
                    frame_timings[fid] = timing
                    evidence["execution_provenance"] = "fresh_boundary_execution"
                    produced.append(dataclasses.replace(row, metrics={**row.metrics, **timing}, evidence=evidence))
            finally:
                # Rows already produced are preserved even if the loop raised.
                self.results.extend(produced)
                boundary.close()
                cleanup = list(boundary.cleanup_errors)
                if cleanup:
                    self._stage(f"{role}_cleanup", "failed", errors=cleanup)
            self._stage(role, "cancelled" if self.cancel.is_set() else "completed", frames=len(produced),
                        identity=identity, warmup=warm, execution={**provenance, "computed_by": "fresh_boundary_execution"})
            if artifact_errors:
                # The neural stage itself completed; the evidence set is incomplete, so this
                # stage fails the run and the cache entry is never committed.
                self._stage(f"{role}_artifacts", "failed",
                            reason=f"{len(artifact_errors)} overlay PNG write(s) failed; cache commit skipped",
                            frame_ids=[e["frame_id"] for e in artifact_errors], errors=artifact_errors[:20])
            if key is not None and cache is not None and not self.cancel.is_set() and not artifact_errors and not cleanup:
                pending.append((key, {"results": [r.to_dict() for r in produced], "warmup": warm,
                                      "execution": {**provenance, "computed_by": "fresh_boundary_execution"},
                                      "telemetry": {"frames": frame_timings,
                                                    "initialization_ms": (self.tel.stages.get(f"{role}_initialization") or [None])[-1]}},
                                files))
        finally:
            # Exactly one release. A release failure keeps every computed row, is recorded as a failed
            # stage and forbids the commit; an original body error is never replaced.
            try:
                stack.close()
            except Exception as exc:  # noqa: BLE001
                release_error = f"{type(exc).__name__}: {exc}"[:500]
                try:
                    self._stage(f"{role}_gpu_release", "failed",
                                reason=f"GPU stage release failed: {release_error}; computed rows kept, cache commit skipped")
                except OSError:
                    pass
            if release_error is None:
                for pending_key, payload, pending_files in pending:
                    self._defer_cache_store(cache, pending_key, payload, pending_files, stage=role)

    def _blocked_role(self, role, identity, reason, pose_state="blocked"):
        for fid in self.run.scheduled_frame_ids:
            self.results.append(wc.blocked_result(self.run.run_id, self.clip, self.frames[fid], role, identity, reason,
                                                  pose_state=pose_state,
                                                  mask_state="blocked" if pose_state == "blocked" else "unavailable"))
        if role not in self.stages or self.stages.get(role, {}).get("status") != "blocked":
            self._stage(role, "blocked", reason=reason)

    def _finish(self, blocked_reason: str | None = None) -> dict:
        # Exactly one row per (requested role, scheduled frame): keep already
        # produced rows (first wins), fill only missing pairs, never fabricate poses.
        failures = getattr(self, "_role_failures", {})
        seen: set[tuple[str, str]] = set()
        unique: list[wc.FrameResult] = []
        duplicates = 0
        scheduled = set(self.run.scheduled_frame_ids)
        for row in self.results:
            key = (row.backend_role, row.frame_id)
            if key in seen or row.backend_role not in self.run.roles or row.frame_id not in scheduled:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(row)
        filled = {}
        for role in self.run.roles:
            reason = failures.get(role) or failures.get("pipeline")
            for fid in self.run.scheduled_frame_ids:
                if (role, fid) in seen:
                    continue
                if reason:
                    row = wc.blocked_result(self.run.run_id, self.clip, self.frames[fid], role, {"role": role},
                                            f"role_failed: {reason}", pose_state="unavailable",
                                            mask_state="unavailable")
                elif blocked_reason:
                    row = wc.blocked_result(self.run.run_id, self.clip, self.frames[fid], role, {"role": role},
                                            blocked_reason)
                else:
                    row = wc.blocked_result(self.run.run_id, self.clip, self.frames[fid], role, {"role": role},
                                            f"not_produced: {role} produced no result for this frame",
                                            pose_state="unavailable", mask_state="unavailable")
                unique.append(row)
                seen.add((role, fid))
                filled[role] = filled.get(role, 0) + 1
        self.results = unique
        if filled or duplicates:
            unexplained = [r for r in filled if r not in failures and "pipeline" not in failures and not blocked_reason]
            self._stage("accounting", "failed" if (duplicates or unexplained) else "completed",
                        filled_rows=filled, dropped_duplicate_or_unscheduled_rows=duplicates,
                        unexplained_roles=unexplained)
        with (self.run_dir / "results.jsonl").open("xb") as stream:
            for row in self.results:
                stream.write(wc.canonical_json(row.to_dict()) + b"\n")
        report = wc.build_report(self.run, self.clip, self.results, stage_status=self.stages)
        from . import workbench_telemetry as wt
        self.tel.finished = time.perf_counter()
        report["telemetry"] = self.tel.snapshot()
        heads = {}
        for role in self.run.roles:
            row = next((r for r in self.results if r.backend_role == role), None)
            ident = row.backend_identity if row is not None else {}
            heads[role] = {k: ident.get(k) for k in ("installed_head_sha256", "installed_head_state_sha256",
                                                     "export_sha256", "base_checkpoint_sha256", "package_sha256",
                                                     "principal_point_origin", "effective_K")
                           if k in ident}
        report["dependencies"] = {"versions": wt.dependency_versions(), "heads": heads,
                                  "sources": _module_versions("workbench_cli", "workbench_contract", "workbench_geometry",
                                                              "workbench_neural", "workbench_reconstruction",
                                                              "workbench_telemetry", "workbench_cache",
                                                              "quality_gotrack", "quality_gotrack_inference",
                                                              "workbench_image_io"),
                                  "note": "installed_head_sha256 is null when no head was actually loaded"}
        statuses = [s["status"] for s in self.stages.values() if s["stage"] not in ("stage_cache",)]
        if self.cancel.is_set():
            status = "cancelled"
        elif any(s == "failed" for s in statuses):
            # Any failed stage (role exception, COLMAP step, cache commit, accounting) makes the run
            # failed; already produced rows keep their actual states.
            status = "failed"
        elif any(s == "blocked" for s in statuses) or blocked_reason:
            status = "blocked"
        else:
            status = "completed"
        report["status"] = status
        _write_new(self.run_dir / "report.json", report)
        self.progress(f"{self.run.run_id} finished: {status}")
        return {"run_id": self.run.run_id, "status": status, "run_dir": str(self.run_dir), "report": report}


def execute_run(ws: Workspace, clip: wc.ClipSpec, run: wc.RunSpec, options: RunOptions | None = None,
                cancel: threading.Event | None = None, progress: Callable[[str], None] = log) -> dict:
    return Pipeline(ws, clip, run, options or RunOptions(), cancel, progress).execute()


def load_results(ws: Workspace, run_id: str) -> tuple[dict, list[wc.FrameResult], dict]:
    wc._ident(run_id, "run_id")
    d = ws.runs_dir / run_id
    run = _read_json(d / "run.json")
    rows = [wc.FrameResult.from_dict(wc.loads_strict(line)) for line in
            (d / "results.jsonl").read_bytes().splitlines() if line.strip()]
    report = _read_json(d / "report.json") if (d / "report.json").is_file() else {}
    return run, rows, report


# --------------------------------------------------------------------------
# import helpers

def parse_point(text: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("points are X,Y in source pixels")
    return float(parts[0]), float(parts[1])


def parse_K(text: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--K is fx,fy,cx,cy")
    return tuple(parts)  # type: ignore[return-value]


def _intrinsics(args) -> wc.Intrinsics:
    if bool(args.K) == bool(args.calibration_unavailable):
        raise WorkbenchCLIError("supply exactly one of --K fx,fy,cx,cy or --calibration-unavailable")
    if args.calibration_unavailable:
        return wc.Intrinsics("unavailable", None, "user marked calibration unavailable")
    fx, fy, cx, cy = args.K
    if args.principal_origin == "corner":
        cx, cy = cx - 0.5, cy - 0.5
    return wc.Intrinsics.from_dict({"status": "measured", "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                                    "source": f"user measured ({args.principal_origin} principal origin)"})


def _onboarding(args, frames: Sequence[wc.FrameSpec]) -> wc.Onboarding | None:
    if args.keyframes is None:
        return None
    by_index = {f.source_frame_index: f.frame_id for f in frames}
    start, _, end = args.keyframes.partition(":")
    ids = [by_index[i] for i in range(int(start), int(end) + 1) if i in by_index]
    selection = None
    if args.select_frame is not None:
        if not args.positive:
            raise WorkbenchCLIError("selection needs at least one --positive X,Y")
        selection = wc.Selection(by_index[args.select_frame], tuple(args.positive), tuple(args.negative or ()),
                                 True, args.reviewed_by or "cli user")
    return wc.Onboarding.from_dict({"keyframe_ids": ids, "selection": None if selection is None else selection.to_dict()})


def import_video(ws: Workspace, path: Path, args) -> wc.ClipSpec:
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise WorkbenchCLIError("unsupported video extension")
    root_id, rel = ws.locate(path)
    size = path.stat().st_size
    counted = 0 if root_id in BUILTIN_ROOTS else size
    ws.require_budget(extra_inputs=counted)
    log(f"hashing {path.name} ({size} bytes)")
    digest = sha256_file(path)
    log("probing decoder timing (streaming, no frames stored)")
    probe = probe_video(path)
    frames = []
    for i in range(probe["frame_count"]):
        t = probe["media_times"][i] if probe["pts_reliable"] else None
        frames.append(wc.FrameSpec(f"f{i:06d}", i, t, t))
    onboarding = _onboarding(args, frames)
    last = -1
    if onboarding is not None:
        pos = {f.frame_id: i for i, f in enumerate(frames)}
        last = max(pos[k] for k in onboarding.keyframe_ids)
    clip = wc.ClipSpec(
        clip_id=args.clip_id, title=args.title or path.name, source_kind="video_file",
        clock_kind="timestamp_seconds" if probe["pts_reliable"] else "frame_index",
        width=probe["width"], height=probe["height"], media=wc.MediaRef(root_id, rel, digest, size),
        frames=tuple(frames), intrinsics=_intrinsics(args), onboarding=onboarding,
        scoring_frame_ids=tuple(f.frame_id for f in frames[last + 1:]), evidence_role="user_import_development",
        clock_source="decoder_pts_opencv_pos_msec" if probe["pts_reliable"] else None,
        provenance={"imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "counted_input_bytes": counted,
                    "pts_reliable": probe["pts_reliable"], "copied": False})
    clip = wc.ClipSpec.from_dict(clip.to_dict())
    ws.save_clip(clip)
    return clip


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def import_images(ws: Workspace, directory: Path, args) -> wc.ClipSpec:
    directory = Path(directory).resolve(strict=True)
    root_id, rel = ws.locate(directory)
    files = sorted((p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()),
                   key=lambda p: _natural_key(p.name))
    if not files:
        raise WorkbenchCLIError("no images found")
    from .vision import cv2
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise WorkbenchCLIError("first image is unreadable")
    frames = tuple(wc.FrameSpec(f"f{i:06d}", i, None, None, p.name, sha256_file(p)) for i, p in enumerate(files))
    onboarding = _onboarding(args, frames)
    last = -1
    if onboarding is not None:
        pos = {f.frame_id: i for i, f in enumerate(frames)}
        last = max(pos[k] for k in onboarding.keyframe_ids)
    size = sum(p.stat().st_size for p in files)
    counted = 0 if root_id in BUILTIN_ROOTS else size
    ws.require_budget(extra_inputs=counted)
    clip = wc.ClipSpec(args.clip_id, args.title or directory.name, "image_sequence", "frame_index",
                       first.shape[1], first.shape[0], wc.MediaRef(root_id, rel, None, None), frames,
                       _intrinsics(args), onboarding, tuple(f.frame_id for f in frames[last + 1:]),
                       "user_import_development", provenance={"counted_input_bytes": counted, "copied": False})
    clip = wc.ClipSpec.from_dict(clip.to_dict())
    ws.save_clip(clip)
    return clip


# --------------------------------------------------------------------------
# CLI

def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def cmd_inspect(ws: Workspace, args) -> int:
    from . import workbench_neural as wn
    from . import workbench_reconstruction as wr
    cache, cache_reason = stage_cache(ws)
    clips = ws.list_clips()
    out = {"workspace": str(ws.path), "initialized": ws.path.is_dir(), "budget": ws.budget(),
           "roots": {k: str(v) for k, v in ws.roots().items()},
           "clips": [{"clip_id": c.clip_id, "frames": len(c.frames), "clock": c.clock_kind,
                      "intrinsics": c.intrinsics.status, "selection": bool(c.onboarding and c.onboarding.selection),
                      "evidence_role": c.evidence_role} for c in clips],
           "stage_cache": "available" if cache else cache_reason,
           "colmap": wr.preflight(args.colmap),
           "neural": wn.neural_preflight(head_path=Path(args.head) if args.head else None,
                                         verify_base_hash=args.verify_base)}
    out["colmap"].pop("commands", None) if not args.verbose else None
    _print(out)
    return EXIT_COMPLETED


def cmd_init(ws: Workspace, args) -> int:
    roots = {}
    for item in args.root or []:
        name, sep, path = item.partition("=")
        if not sep:
            raise WorkbenchCLIError("--root uses NAME=PATH")
        roots[name] = path
    _print(ws.init(roots))
    return EXIT_COMPLETED


def cmd_import(ws: Workspace, args) -> int:
    ws.init()
    if args.kind == "video":
        clip = import_video(ws, Path(args.path), args)
    elif args.kind == "images":
        clip = import_images(ws, Path(args.path), args)
    else:
        clip = ws.load_clip(args.clip_id)
        _print(import_masks(ws, clip, Path(args.path), args.mask_set, args.producer))
        return EXIT_COMPLETED
    _print({"clip_id": clip.clip_id, "frames": len(clip.frames), "clock_kind": clip.clock_kind,
            "intrinsics": clip.intrinsics.status, "selection": bool(clip.onboarding and clip.onboarding.selection)})
    return EXIT_COMPLETED


def cmd_suite(ws: Workspace, args) -> int:
    from . import workbench_suite as ws_suite
    ws.init()
    result = ws_suite.export_suite(ws, verify_images=args.verify_images)
    _print(result)
    return EXIT_COMPLETED if not result.get("errors") else EXIT_FAILED


def cmd_run(ws: Workspace, args) -> int:
    clip = ws.load_clip(args.clip_id)
    roles = tuple(args.roles.split(",")) if args.roles else ("primary_neural", "frozen_baseline", "relative_geometry")
    run = build_run_spec(clip, mode=args.mode, first_frame=args.first_frame, count=args.count, roles=roles,
                         artwork_kind=args.artwork, config={"mask_set": args.masks} if args.masks else None)
    admission = None
    if args.admission:
        from . import workbench_neural as wn
        admission = wn.load_admission(Path(args.admission))
        log(f"admission {'verified' if admission.admitted else 'REJECTED'}: {list(admission.reasons)[:4]}")
    options = RunOptions(mask_set=args.masks, colmap=args.colmap, head_path=args.head, admission=admission,
                         metric_bundle=args.metric_bundle, device=args.device)
    if args.dry_run:
        from . import workbench_neural as wn
        from . import workbench_reconstruction as wr
        _print({"status": "dry_run", "executed": False, "run": run.to_dict(),
                "colmap": wr.preflight(args.colmap).get("reason") or "available",
                "neural": wn.neural_preflight(head_path=Path(args.head) if args.head else None)["blocked_reasons"],
                "masks": MaskStore(ws, clip.clip_id, args.masks).available})
        return EXIT_DRY_RUN
    ws.init()
    result = execute_run(ws, clip, run, options)
    _print({k: result[k] for k in ("run_id", "status", "run_dir")} | {"roles": {
        r: v["counts"]["pose_state"] for r, v in result["report"]["roles"].items()}})
    return {"completed": EXIT_COMPLETED, "blocked": EXIT_BLOCKED}.get(result["status"], EXIT_FAILED)


def cmd_report(ws: Workspace, args) -> int:
    _run, rows, report = load_results(ws, args.run_id)
    if args.frames:
        report = dict(report, frames=[r.to_dict() for r in rows])
    _print(report)
    return EXIT_COMPLETED


def cmd_segment(ws: Workspace, args) -> int:
    from . import workbench_neural as wn
    clip = ws.load_clip(args.clip_id)
    if clip.onboarding is None or clip.onboarding.selection is None:
        raise WorkbenchCLIError("clip has no reviewed selection")
    scheduled = wc.schedule_frames(clip, args.mode, clip.onboarding, first_frame_id=args.first_frame, count=args.count)
    pre = wn.segmentation_preflight(verify_hash=args.verify_hash)
    if args.dry_run or not pre["ready"]:
        _print({"status": "dry_run" if args.dry_run else "blocked", "executed": False, "preflight": pre,
                "frames": len(set(clip.onboarding.keyframe_ids) | set(scheduled))})
        return EXIT_DRY_RUN if args.dry_run else EXIT_BLOCKED
    ws.init()
    _print(run_segmentation(ws, clip, clip.onboarding, scheduled, args.mask_set, device=args.device))
    return EXIT_COMPLETED


def cmd_gate(ws: Workspace, args) -> int:
    """Earn (or diagnose) workbench_integer_metric_v1 serving admission."""
    from . import workbench_neural as wn
    from .vision import cv2
    if args.gate == "bundle":
        body = wn.historical_metric_bundle(args.name, principal_point_origin=args.principal_origin)
        _write_new(Path(args.out), body)
        _print(body)
        return EXIT_COMPLETED
    if args.gate == "case":
        clip = ws.load_clip(args.clip_id)
        bundle = wn.MetricBundle.from_file(Path(args.metric_bundle))
        if bundle.seed_pose_m is None:
            raise WorkbenchCLIError("bundle has no labelled initialization pose")
        frame = clip.frame(args.frame)
        store = MaskStore(ws, clip.clip_id, args.mask_set)
        reader = FrameReader(ws, clip)
        try:
            rgb, _ = reader.read(frame)
        finally:
            reader.close()
        mask, mident = store.load(frame.frame_id, (clip.height, clip.width))
        if mask is None:
            raise WorkbenchCLIError("no observed mask for this frame in the mask set")
        out_dir = ws.derived_dir / "gate-cases" / f"{clip.clip_id}-{frame.frame_id}"
        out_dir.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(out_dir / "rgb.png"), rgb[:, :, ::-1])
        cv2.imwrite(str(out_dir / "mask.png"), mask.astype(np.uint8) * 255)
        case = {"case_id": f"{clip.clip_id}-{frame.frame_id}".replace("_", "-"), "family": bundle.family,
                "rgb_path": str(out_dir / "rgb.png"), "rgb_sha256": sha256_file(out_dir / "rgb.png"),
                "mask_path": str(out_dir / "mask.png"), "mask_sha256": sha256_file(out_dir / "mask.png"),
                "segmentation": {"producer": json.dumps(store.identity()["producer"], sort_keys=True)[:500],
                                 "manifest_sha256": store.identity()["manifest_sha256"]},
                "K": list(clip.intrinsics.K), "principal_point_origin": bundle.principal_point_origin,
                "initial_pose_m": bundle.seed_pose_m.tolist(), "initialization_source": "onboarding:" + bundle.seed_label,
                "glb_path": str(bundle.glb_path), "glb_sha256": bundle.glb_sha256, "obj_id": bundle.obj_id,
                "metric_provenance": bundle.metric_provenance}
        _write_new(Path(args.out), {"cases": [case]})
        _print(case)
        return EXIT_COMPLETED
    cases = []
    for path in getattr(args, "cases", None) or []:
        cases.extend(_read_json(Path(path))["cases"])
    if args.gate == "contract":
        contract = wn.prepare_workbench_contract(cases, head_path=Path(args.head), reference_path=Path(args.reference),
                                                 verify_base=not args.skip_base_hash)
        wn.save_contract(contract, Path(args.out))
        closure = contract.body["source_closure"]
        _print({"contract_sha256": contract.sha256, "families": contract.body["families"],
                "reviewed_replacements": sorted(closure["reviewed_replacements"]),
                "unreviewed_replacements": closure["unreviewed_replacements"],
                "review_evidence_present": closure["review_evidence_present"]})
        return EXIT_COMPLETED
    if args.gate == "geometry":
        from . import workbench_telemetry as wt
        if args.production:
            # Actual renderer + crop pipeline: same single GPU stage as SAM/pose/parity.
            evidence = Path(args.out).parent / "geometry-evidence"
            with wt.gpu_stage("gate_geometry_production"):
                production = wn.measure_production_crop(ws.scratch_dir / "geometry-probe", evidence_dir=evidence)
                probe = wn.production_renderer_probe(ws.scratch_dir / "geometry-probe")
                body = wn.validate_metric_geometry(renderer_probe=probe, production=production, evidence_dir=evidence)
        else:
            body = wn.validate_metric_geometry()
        _write_new(Path(args.out), body)
        _print({"status": body["status"], "admission_eligible": body["admission_eligible"]})
        return EXIT_COMPLETED if body["status"] != "failed" else EXIT_FAILED
    if args.gate == "parity":
        contract = wn.load_contract(Path(args.contract))
        loaded = []
        for case in cases:
            rgb = cv2.imread(case["rgb_path"], cv2.IMREAD_COLOR)
            mask = cv2.imread(case["mask_path"], cv2.IMREAD_GRAYSCALE)
            if rgb is None or mask is None:
                raise WorkbenchCLIError(f"case {case['case_id']} images unreadable")
            loaded.append({**case, "rgb": np.ascontiguousarray(rgb[:, :, ::-1]), "mask": mask > 127})
        from . import workbench_telemetry as wt
        with wt.gpu_stage("gate_parity"):
            receipt = wn.run_fullimage_head_parity(loaded, contract=contract, head_path=Path(args.head),
                                                   reference_path=Path(args.reference), output_dir=Path(args.out),
                                                   device=args.device)
        _print({"all_passed": receipt["all_passed"], "receipt_sha256": receipt["receipt_sha256"]})
        return EXIT_COMPLETED if receipt["all_passed"] else EXIT_FAILED
    if args.gate == "verify":
        admission = wn.load_admission(Path(args.admission))
        _print(admission.to_dict())
        return EXIT_COMPLETED if admission.admitted else EXIT_BLOCKED
    raise WorkbenchCLIError("unknown gate command")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bench.workbench_cli",
                                description="VisualizeIT video validation workbench (local, no downloads).")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="workspace directory (.cache/video-workbench)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="diagnose workspace, budgets, COLMAP and neural prerequisites")
    s.add_argument("--colmap")
    s.add_argument("--head", help="path of the flow-terminal head.pt export")
    s.add_argument("--verify-base", action="store_true", help="hash the 1.6 GB base checkpoint")
    s.add_argument("--verbose", action="store_true")
    s = sub.add_parser("init", help="create the workspace and register media roots")
    s.add_argument("--root", action="append", metavar="NAME=PATH")
    s = sub.add_parser("import", help="register a local video/image sequence in place, or observed masks")
    s.add_argument("kind", choices=("video", "images", "masks"))
    s.add_argument("path")
    s.add_argument("--clip-id", required=True)
    s.add_argument("--title")
    s.add_argument("--K", type=parse_K, help="measured fx,fy,cx,cy")
    s.add_argument("--principal-origin", choices=("integer_centers", "corner"), default="integer_centers")
    s.add_argument("--calibration-unavailable", action="store_true")
    s.add_argument("--keyframes", help="onboarding source frame range START:END (<=24 frames)")
    s.add_argument("--select-frame", type=int, help="source frame index of the reviewed selection")
    s.add_argument("--positive", type=parse_point, action="append", help="X,Y source pixel (repeatable)")
    s.add_argument("--negative", type=parse_point, action="append", help="X,Y source pixel (repeatable)")
    s.add_argument("--reviewed-by")
    s.add_argument("--mask-set", default="observed-v1")
    s.add_argument("--producer", default="unspecified external segmentation")
    s = sub.add_parser("suite", help="export the deterministic six-case development suite")
    s.add_argument("--verify-images", action="store_true", help="hash every YCB frame image")
    s = sub.add_parser("run", help="run smoke/focused/regression on a registered clip")
    s.add_argument("clip_id")
    s.add_argument("--mode", choices=wc.RUN_MODES, default="smoke")
    s.add_argument("--first-frame")
    s.add_argument("--count", type=int)
    s.add_argument("--roles", help="comma list of primary_neural,frozen_baseline,relative_geometry")
    s.add_argument("--masks", help="observed mask set name")
    s.add_argument("--colmap")
    s.add_argument("--head")
    s.add_argument("--artwork", choices=("checkerboard", "directional"), default="checkerboard")
    s.add_argument("--admission", help="admission dir {contract.json, geometry.json, parity/}; re-verified each run")
    s.add_argument("--metric-bundle", help="workbench_metric_bundle_v1 file (pinned metric GLB + labelled seed)")
    s.add_argument("--device", default="cpu", help="torch device for admitted neural roles (parent-scheduled)")
    s.add_argument("--dry-run", action="store_true", help="print the plan; executes nothing (exit 4)")
    s = sub.add_parser("report", help="print a run report")
    s.add_argument("run_id")
    s.add_argument("--frames", action="store_true")
    s = sub.add_parser("segment", help="tap-to-mask: SAM 2.1 prompted observed masks (explicit neural action)")
    s.add_argument("clip_id")
    s.add_argument("--mask-set", required=True)
    s.add_argument("--mode", choices=wc.RUN_MODES, default="smoke")
    s.add_argument("--first-frame")
    s.add_argument("--count", type=int)
    s.add_argument("--device", default="cpu")
    s.add_argument("--verify-hash", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("gate", help="earn/verify workbench_integer_metric_v1 serving admission")
    g = s.add_subparsers(dest="gate", required=True)
    gs = g.add_parser("bundle", help="metric bundle for a historical case (controlled_init seed)")
    gs.add_argument("name", choices=("keyboard", "mug", "ranch"))
    gs.add_argument("--principal-origin", required=True, choices=("integer_centers", "corner"),
                    help="documented principal-point origin of the clip K")
    gs.add_argument("--out", required=True)
    gs = g.add_parser("case", help="freeze one unlabelled parity case from a clip frame + observed mask")
    gs.add_argument("clip_id")
    gs.add_argument("--frame", required=True)
    gs.add_argument("--mask-set", required=True)
    gs.add_argument("--metric-bundle", required=True)
    gs.add_argument("--out", required=True)
    gs = g.add_parser("contract", help="authenticate model/source/input bindings")
    gs.add_argument("--cases", nargs="+", required=True)
    gs.add_argument("--head", required=True)
    gs.add_argument("--reference", required=True, help="run/best.pt (0c211531)")
    gs.add_argument("--skip-base-hash", action="store_true")
    gs.add_argument("--out", required=True)
    gs = g.add_parser("geometry", help="manufactured-scene geometry validation")
    gs.add_argument("--production", action="store_true", help="measure the actual renderer + crop pipeline")
    gs.add_argument("--out", required=True)
    gs = g.add_parser("parity", help="full-image reference-vs-export head parity (runs the network twice)")
    gs.add_argument("--contract", required=True)
    gs.add_argument("--cases", nargs="+", required=True)
    gs.add_argument("--head", required=True)
    gs.add_argument("--reference", required=True)
    gs.add_argument("--device", default="cpu")
    gs.add_argument("--out", required=True, help="new parity directory")
    gs = g.add_parser("verify", help="recompute admission from an admission directory")
    gs.add_argument("--admission", required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ws = Workspace(args.workspace)
    handlers = {"inspect": cmd_inspect, "init": cmd_init, "import": cmd_import, "suite": cmd_suite,
                "run": cmd_run, "report": cmd_report, "segment": cmd_segment, "gate": cmd_gate}
    try:
        return handlers[args.command](ws, args)
    except (WorkbenchCLIError, wc.ContractError, FileNotFoundError, KeyError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - admission/runtime prerequisites
        name = type(exc).__name__
        if name in ("NeuralUnavailable", "AdmissionError", "ReconstructionUnavailable"):
            print(f"blocked: {name}: {exc}", file=sys.stderr)
            return EXIT_BLOCKED
        raise


if __name__ == "__main__":
    raise SystemExit(main())
