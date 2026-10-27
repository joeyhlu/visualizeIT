"""Streaming, reference-free continuity descriptions of registered observed masks.

Only clip metadata and mask PNGs are read. These measurements cannot establish
segmentation accuracy, visibility, pose quality, or a tracking failure.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
import shutil
import struct
import uuid

import numpy as np

from . import workbench_cli as cli
from . import workbench_contract as wc
from .workbench_geometry import mask_sha256
from .workbench_offline_export import _publish_new_directory

ROOT = Path(__file__).resolve().parents[1]
VERSION = "workbench_mask_diagnostics_v1"
MAX_FRAMES = 1000
MAX_PIXELS = 8_000_000
MAX_PNG_BYTES = 8 << 20
MAX_JSON_BYTES = 8 << 20
MAX_OUTPUT_BYTES = 16 << 20
SOURCES = ("bench/workbench_mask_diagnostics.py", "tools/workbench_mask_diagnostics.py",
           "bench/workbench_cli.py", "bench/workbench_contract.py", "bench/workbench_geometry.py",
           "bench/workbench_offline_export.py", "bench/workbench_artwork_replay.py", "bench/vision.py")
ROW_FIELDS = ("frame_id", "source_frame_index", "timestamp_s", "media_time_s", "mask_state",
              "file_sha256", "content_sha256", "area_px", "area_fraction", "bbox_xyxy", "centroid_xy",
              "previous_frame_id", "pair_status", "temporal_iou", "area_ratio", "centroid_distance_px",
              "centroid_distance_px720", "baseline_frame_id", "baseline_area_ratio", "area_drop",
              "low_baseline_area")


def refuse(message):
    raise cli.WorkbenchCLIError(message)


def _no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        refuse("diagnostic paths cannot traverse symlinks")
    return path


def _read_bounded(path, limit):
    path = _no_symlinks(path)
    if not path.is_file() or path.stat().st_size > limit:
        refuse(f"missing or oversized input: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        refuse("input grew beyond its byte limit")
    return raw


def _source_hashes():
    return {name: hashlib.sha256(_read_bounded(ROOT / name, MAX_JSON_BYTES)).hexdigest() for name in SOURCES}


def _load_mask(store, frame_id, shape):
    """Bound PNG allocation before using the existing authenticated mask decoder."""
    entry = store.entry(frame_id)
    if entry is None:
        return None, None
    path = _no_symlinks(entry["abs_path"])
    raw = _read_bounded(path, MAX_PNG_BYTES)
    if (len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
            or struct.unpack(">I", raw[8:12])[0] != 13):
        refuse("mask is not a PNG with a bounded IHDR")
    width, height = struct.unpack(">II", raw[16:24])
    if not width or not height or width * height > MAX_PIXELS or (height, width) != shape:
        refuse("PNG header dimensions differ or exceed pixel limit")
    if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
        refuse("mask PNG hash differs from registration")
    # Decode this exact bounded byte snapshot, avoiding a second unbounded read.
    from .vision import cv2
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.shape != shape:
        refuse("mask PNG undecodable or wrong size")
    mask = decoded > 127
    content = mask_sha256(mask)
    for key, expected in (("width", width), ("height", height)):
        if key in entry and (type(entry[key]) is not int or entry[key] != expected):
            refuse(f"mask declared {key} differs")
    if "empty" in entry and (type(entry["empty"]) is not bool or entry["empty"] != (not bool(mask.any()))):
        refuse("mask declared empty differs")
    if "content_sha256" in entry and entry["content_sha256"] != content:
        refuse("mask content_sha256 differs")
    # Keep MaskStore's file verifier as the registered source of file integrity.
    if store.verify([frame_id])[frame_id] != hashlib.sha256(raw).hexdigest():
        refuse("mask changed during decode")
    return mask, {"file_sha256": entry["sha256"], "content_sha256": content}


def _episode(rows, clock_kind):
    first, last = rows[0], rows[-1]
    return dict(frame_ids=[r["frame_id"] for r in rows], start_frame_id=first["frame_id"],
                end_frame_id=last["frame_id"], frame_count=len(rows),
                start_source_frame_index=first["source_frame_index"],
                end_source_frame_index=last["source_frame_index"],
                timestamp_span_s=(last["timestamp_s"] - first["timestamp_s"]
                                  if clock_kind == "timestamp_seconds" else None))


def audit_saved_masks(workspace, *, clip_id, mask_set, output_dir, area_ratio_min=.5, split="scoring"):
    """Publish report.json/report.csv in a new direct child of workspace.derived_dir."""
    if not isinstance(workspace, cli.Workspace):
        workspace = cli.Workspace(workspace)
    ws = workspace
    if type(area_ratio_min) not in (int, float) or not math.isfinite(area_ratio_min) or not 0 < area_ratio_min < 1:
        refuse("area_ratio_min must be a finite number strictly between zero and one")
    if split not in ("scoring", "onboarding", "all"):
        refuse("unknown split")
    wc._ident(clip_id, "clip_id")
    wc._ident(mask_set, "mask_set")
    _no_symlinks(ws.path)
    derived = _no_symlinks(ws.derived_dir)
    output = _no_symlinks(Path(output_dir).absolute())
    if output.parent.resolve() != derived.resolve() or output.exists():
        refuse("output must be a new direct child of derived")
    if not derived.is_dir():
        refuse("workspace derived directory is unavailable")
    sources = _source_hashes()
    config_raw = _read_bounded(ws.config_path, MAX_JSON_BYTES) if ws.config_path.exists() else None
    clip_path = ws.clip_path(clip_id)
    clip_raw = _read_bounded(clip_path, MAX_JSON_BYTES)
    clip = wc.ClipSpec.from_dict(wc.loads_strict(clip_raw))
    if clip.clip_id != clip_id:
        refuse("clip identity differs")
    if clip.width * clip.height > MAX_PIXELS:
        refuse("clip exceeds pixel limit")
    chosen = (set(clip.scoring_frame_ids) if split == "scoring" else
              set(clip.onboarding.keyframe_ids if clip.onboarding else ()) if split == "onboarding" else
              {f.frame_id for f in clip.frames})
    frames = [f for f in clip.frames if f.frame_id in chosen]
    if not 1 <= len(frames) <= MAX_FRAMES:
        refuse("selected schedule is empty or exceeds 1000 frames")
    ids = [f.frame_id for f in frames]
    manifest_path = ws.masks_dir / clip_id / mask_set / "manifest.json"
    manifest_raw = _read_bounded(manifest_path, MAX_JSON_BYTES)
    manifest = wc.loads_strict(manifest_raw)
    store = cli.MaskStore(ws, clip_id, mask_set)
    if store.manifest != manifest or manifest.get("mask_set", mask_set) != mask_set:
        refuse("mask manifest changed or mask_set differs")
    if not isinstance(manifest.get("masks"), dict):
        refuse("mask manifest masks must be an object")
    for fid in ids:
        if fid in manifest["masks"] and not isinstance(manifest["masks"][fid], dict):
            refuse("declared mask entry must be an object")
    rows, episodes, active = [], [], []
    baseline = previous_mask = previous_row = None
    files = {}
    for frame in frames:
        mask, identity = _load_mask(store, frame.frame_id, (clip.height, clip.width))
        row = dict.fromkeys(ROW_FIELDS)
        row.update(frame_id=frame.frame_id, source_frame_index=frame.source_frame_index,
                   timestamp_s=frame.timestamp_s, media_time_s=frame.media_time_s,
                   mask_state="unavailable" if mask is None else "observed")
        files[frame.frame_id] = None if identity is None else identity["file_sha256"]
        contiguous = previous_row is not None and frame.source_frame_index == previous_row["source_frame_index"] + 1
        row["pair_status"] = ("current_missing" if mask is None else "first_frame" if previous_row is None
                              else "source_gap" if not contiguous else "previous_missing"
                              if previous_mask is None else "available")
        if mask is not None:
            row.update(identity)
            area = int(np.count_nonzero(mask))
            row.update(area_px=area, area_fraction=area / (clip.width * clip.height))
            if area:
                # Axis reductions bound auxiliary memory independently of foreground area.
                xs = np.flatnonzero(mask.any(axis=0))
                ys = np.flatnonzero(mask.any(axis=1))
                row["bbox_xyxy"] = [int(xs[0]), int(ys[0]), int(xs[-1])+1, int(ys[-1])+1]
                row["centroid_xy"] = [float(np.dot(mask.sum(axis=0), np.arange(clip.width)) / area),
                                       float(np.dot(mask.sum(axis=1), np.arange(clip.height)) / area)]
                if baseline is None:
                    baseline = dict(frame_id=frame.frame_id, area_px=area)
            if baseline is not None:
                row.update(baseline_frame_id=baseline["frame_id"], baseline_area_ratio=area / baseline["area_px"])
                row["low_baseline_area"] = row["baseline_area_ratio"] < area_ratio_min
            if contiguous and previous_mask is not None:
                row["previous_frame_id"] = previous_row["frame_id"]
                intersection = int(np.count_nonzero(mask & previous_mask))
                union = area + previous_row["area_px"] - intersection
                if not union:
                    row["pair_status"] = "both_empty"
                elif not previous_row["area_px"]:
                    row["pair_status"] = "previous_empty"
                elif not area:
                    row["pair_status"] = "current_empty"
                if union:
                    row["temporal_iou"] = intersection / union
                if previous_row["area_px"]:
                    row["area_ratio"] = area / previous_row["area_px"]
                    row["area_drop"] = row["area_ratio"] < area_ratio_min
                if area and previous_row["area_px"]:
                    row["centroid_distance_px"] = math.dist(row["centroid_xy"], previous_row["centroid_xy"])
                    row["centroid_distance_px720"] = row["centroid_distance_px"] * 720 / clip.height
        if active and (not contiguous or row["low_baseline_area"] is not True):
            episodes.append(_episode(active, clip.clock_kind))
            active = []
        if row["low_baseline_area"] is True:
            active.append(row)
        rows.append(row)
        previous_mask, previous_row = mask, row
    if active:
        episodes.append(_episode(active, clip.clock_kind))
    observed = sum(r["mask_state"] == "observed" for r in rows)
    report = dict(schema=VERSION, status="completed" if observed else "unavailable", clip_id=clip_id,
        mask_set=mask_set, split=split, scheduled_frame_ids=ids,
        policy=dict(area_ratio_min=float(area_ratio_min), comparison="strict_less_than", status="proposed_not_validated",
                    purpose="descriptive diagnostic only; never an inference or tracking decision", px720_scale="720 / native height"),
        baseline=baseline, frame_accounting=dict(scheduled=len(rows), observed=observed,
                    unavailable=len(rows)-observed, empty=sum(r["area_px"] == 0 for r in rows),
                    nonempty=sum(r["area_px"] is not None and r["area_px"] > 0 for r in rows)),
        episodes=episodes, rows=rows, independent_accuracy=None, quality_claim=False, new_neural_inference=False,
        provenance=dict(clip_raw_sha256=hashlib.sha256(clip_raw).hexdigest(),
            mask_manifest_raw_sha256=hashlib.sha256(manifest_raw).hexdigest(), mask_manifest=store.identity(),
            masks=files, sources=sources, config_sha256=None if config_raw is None else hashlib.sha256(config_raw).hexdigest(),
            evaluator_inputs_read=False, rgb_binding="clip metadata only; RGB pixels not verified or opened",
            inputs_read=["registered clip metadata", "registered observed mask manifest", "selected mask PNG files"],
            clock_kind=clip.clock_kind, clock_source=clip.clock_source))
    raw = wc.canonical_json(report)
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=ROW_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: wc.canonical_json(v).decode() if isinstance(v, (list, dict)) else v for k, v in row.items()})
    csv_raw = csv_stream.getvalue().encode("utf-8")
    if len(raw) + len(csv_raw) > MAX_OUTPUT_BYTES:
        refuse("diagnostic output exceeds byte limit")
    ws.require_budget(extra_derived=len(raw) + len(csv_raw))
    staging = derived / (".mask-diagnostics-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name, value in (("report.json", raw), ("report.csv", csv_raw)):
            with (staging / name).open("xb") as stream:
                stream.write(value)
        ws.require_budget()
        # Recheck raw snapshots and every selected file immediately before publication.
        fresh_config = _read_bounded(ws.config_path, MAX_JSON_BYTES) if ws.config_path.exists() else None
        if (_read_bounded(clip_path, MAX_JSON_BYTES) != clip_raw or fresh_config != config_raw
                or _read_bounded(manifest_path, MAX_JSON_BYTES) != manifest_raw or _source_hashes() != sources):
            refuse("input/source/config closure changed before publication")
        fresh = cli.MaskStore(ws, clip_id, mask_set)
        for fid in ids:
            entry = fresh.entry(fid)
            if entry is not None:
                path = _no_symlinks(entry["abs_path"])
                if hashlib.sha256(_read_bounded(path, MAX_PNG_BYTES)).hexdigest() != files[fid]:
                    refuse("mask file changed before publication")
            elif files[fid] is not None:
                refuse("mask entry disappeared before publication")
        if fresh.verify(ids) != files or fresh.identity() != store.identity():
            refuse("mask input closure changed before publication")
        _no_symlinks(output)
        _no_symlinks(staging)
        if output.exists():
            refuse("output appeared before publication")
        _publish_new_directory(staging, output)
        return report
    finally:
        if staging.exists() and not staging.is_symlink() and staging.parent.resolve() == derived.resolve():
            shutil.rmtree(staging)
