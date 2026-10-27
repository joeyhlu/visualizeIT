"""Authenticate a saved export and publish a bounded, reference-free inspection sheet."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from . import workbench_cli as cli, workbench_contract as wc, workbench_geometry as wg
from . import workbench_mask_diagnostics as md
from .workbench_offline_export import _publish_new_directory

ROOT = Path(__file__).resolve().parents[1]
VERSION = "workbench_export_review_v1"
MAX_PNG_BYTES = 16 << 20
MAX_JSON_BYTES = 8 << 20
SOURCES = tuple(dict.fromkeys((*md.SOURCES, "bench/workbench_export_review.py",
                             "tools/workbench_export_review.py")))


def refuse(message):
    raise cli.WorkbenchCLIError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        refuse("SHA256 must be lowercase 64-character hexadecimal")
    return value


def _sources():
    return {name: _sha(md._read_bounded(ROOT / name, MAX_JSON_BYTES)) for name in SOURCES}


def _file_identity(path):
    md._no_symlinks(path)
    if not path.is_file():
        refuse("input must be a regular file")
    return dict(bytes=path.stat().st_size, sha256=cli.sha256_file(path))


def review_saved_export(workspace, *, export_dir, expected_manifest_sha256, frame_ids,
                        output_dir, tile_width=256):
    """Inspect authenticated historical pixels; never infer poses or assess accuracy."""
    from .vision import cv2
    ws = workspace if isinstance(workspace, cli.Workspace) else cli.Workspace(workspace)
    _digest(expected_manifest_sha256)
    if type(tile_width) is not int or not 64 <= tile_width <= 512:
        refuse("tile width must be an integer between 64 and 512")
    if (type(frame_ids) not in (list, tuple) or not 1 <= len(frame_ids) <= 24
            or any(type(fid) is not str for fid in frame_ids) or len(set(frame_ids)) != len(frame_ids)):
        refuse("select 1..24 unique frame IDs")
    derived = md._no_symlinks(ws.derived_dir).resolve(strict=True)
    export = md._no_symlinks(Path(export_dir).absolute())
    output = md._no_symlinks(Path(output_dir).absolute())
    if export.parent.resolve() != derived or not export.is_dir():
        refuse("export must be an existing direct child of derived")
    if output.parent.resolve() != derived or output.exists() or output == export:
        refuse("output must be a new other direct child of derived")
    sources = _sources()
    config = md._read_bounded(ws.config_path, MAX_JSON_BYTES) if ws.config_path.exists() else None
    manifest_path = export / "manifest.json"
    raw = md._read_bounded(manifest_path, 32 << 20)
    if _sha(raw) != expected_manifest_sha256:
        refuse("export manifest SHA256 differs")
    manifest = wc.loads_strict(raw)
    if (manifest.get("schema") != "workbench_offline_export_v1"
            or manifest.get("source") not in ("offline_saved_pose_planar_artwork", "offline_saved_pose_mesh_artwork")):
        refuse("unsupported saved export")
    provenance = manifest["provenance"]
    clip_id = manifest["source_clip"]["clip_id"]
    wc._ident(clip_id, "clip_id")
    clip_path = ws.clip_path(clip_id)
    clip_raw = md._read_bounded(clip_path, MAX_JSON_BYTES)
    if _sha(clip_raw) != _digest(provenance["clip_sha256"]):
        refuse("registered clip raw SHA256 differs")
    clip = wc.ClipSpec.from_dict(wc.loads_strict(clip_raw))
    if clip.clip_id != clip_id:
        refuse("registered clip ID differs")
    width, height = clip.width, clip.height
    if min(width, height) < 2 or max(width, height) > 1280 or width*height > 2_000_000 or width % 2 or height % 2:
        refuse("source dimensions exceed even 1280 edge / 2 MP limits")
    if clip.source_kind == "video_file" and (clip.media.sha256 is None or clip.media.bytes is None):
        refuse("source video requires registered hash and byte count")
    ids, records = manifest["scheduled_frame_ids"], manifest["rows"]
    if (type(ids) is not list or not 1 <= len(ids) <= 1000 or any(type(fid) is not str for fid in ids)
            or len(set(ids)) != len(ids) or type(records) is not list or len(records) != len(ids)):
        refuse("invalid export frame schedule")
    for key in ("scheduled", "exported"):
        if type(manifest["frame_accounting"].get(key)) is not int or manifest["frame_accounting"][key] != len(ids):
            refuse("export frame accounting differs")
    playback = manifest["playback"]
    for key, expected in (("width", width), ("height", height), ("frames", len(ids))):
        if type(playback.get(key)) is not int or playback[key] != expected:
            refuse("playback metadata differs")
    fps = playback.get("fps")
    if (playback.get("codec") != "FFV1" or playback.get("container") != "AVI"
            or type(fps) not in (int, float) or not math.isfinite(fps) or not .1 <= fps <= 120):
        refuse("unsupported playback metadata")
    if manifest.get("role") not in ("primary_neural", "frozen_baseline"):
        refuse("unsupported saved role")
    wc._ident(manifest["run_id"], "run_id")
    originals = {}
    last = -1
    for index, (fid, record) in enumerate(zip(ids, records)):
        if type(record.get("output_index")) is not int or record["output_index"] != index:
            refuse("output index/order differs")
        original = wc.FrameResult.from_dict(record["original"])
        frame = clip.frame(fid)
        for key in ("frame_id", "source_frame_index", "timestamp_s", "media_time_s"):
            if (getattr(original, key) != getattr(frame, key)
                    or type(record["original"][key]) is not type(getattr(frame, key))):
                refuse("original frame metadata differs")
        if (original.clip_id != clip_id or original.run_id != manifest["run_id"]
                or original.backend_role != manifest["role"] or frame.source_frame_index <= last):
            refuse("original run/role/chronology differs")
        last = frame.source_frame_index
        for key in ("source_rgb_sha256", "expected_rgb_sha256", "decoded_rgb_sha256"):
            _digest(record[key])
        if type(record.get("render_reason")) is not str:
            refuse("render reason must be a string")
        originals[fid] = original
    if not set(frame_ids).issubset(ids):
        refuse("unknown selected frame ID")
    selected = [fid for fid in ids if fid in frame_ids]
    tile_height = max(1, (height*tile_width*2 + width)//(2*width))
    sheet_width, sheet_height = 3*tile_width, 40+len(selected)*(tile_height+40)
    if max(sheet_width, sheet_height) > 4096 or sheet_width*sheet_height > 16_000_000:
        refuse("sheet exceeds canvas limits")
    ws.require_budget(extra_derived=MAX_PNG_BYTES+MAX_JSON_BYTES)
    video = export / "playback.avi"
    declared_video = manifest["video"]
    if (declared_video.get("file") != "playback.avi" or type(declared_video.get("bytes")) is not int
            or declared_video["bytes"] <= 0):
        refuse("fixed playback.avi identity required")
    video_identity = _file_identity(video)
    if video_identity != dict(bytes=declared_video["bytes"], sha256=_digest(declared_video["sha256"])):
        refuse("video identity differs")
    mask_identity = provenance["mask_manifest"]
    mask_set = mask_identity["mask_set"]
    wc._ident(mask_set, "mask_set")
    mask_path = ws.masks_dir / clip_id / mask_set / "manifest.json"
    mask_raw = md._read_bounded(mask_path, MAX_JSON_BYTES)
    mask_doc = wc.loads_strict(mask_raw)
    store = cli.MaskStore(ws, clip_id, mask_set)
    if store.manifest != mask_doc or store.identity() != mask_identity:
        refuse("observed mask registry differs")
    mask_files = store.verify(selected)
    if any(fid not in provenance["masks"] or provenance["masks"][fid] != mask_files[fid] for fid in selected):
        refuse("selected mask file provenance differs")
    # FrameReader enforces registered file hashes; reject traversed links before it
    # resolves media paths, so the identity cannot silently move to another tree.
    media_path = md._no_symlinks(ws.media_path(clip))
    if clip.source_kind == "image_sequence":
        source_root = media_path if media_path.is_dir() else media_path.parent
        for fid in selected:
            md._no_symlinks(source_root / clip.frame(fid).image_path)
    # Decode all composites, retaining only selected frames. Failed frames never disappear.
    composites = {}
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or not math.isclose(cap.get(cv2.CAP_PROP_FPS), fps, rel_tol=1e-6, abs_tol=1e-6):
            refuse("video cannot open or header FPS differs")
        for fid, record in zip(ids, records):
            ok, bgr = cap.read()
            if not ok or bgr is None or bgr.shape != (height, width, 3):
                refuse("video frame missing or dimensions differ")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            digest = wg.array_sha256(rgb)
            if digest != record["expected_rgb_sha256"] or digest != record["decoded_rgb_sha256"]:
                refuse("decoded composite hash/order differs")
            if fid in selected:
                composites[fid] = rgb
        if cap.read()[0]:
            refuse("video contains extra frames")
    finally:
        cap.release()
    image = Image.new("RGB", (sheet_width, sheet_height), (32, 32, 32))
    draw = ImageDraw.Draw(image)
    for i, title in enumerate(("SOURCE", "OBSERVED MASK", "EXPORTED COMPOSITE")):
        draw.text((i*tile_width+4, 12), title, fill="white")
    reader = cli.FrameReader(ws, clip)
    staging = None
    try:
        media, source_files = reader.verify_media(), reader.verify_frames(selected)
        if media != provenance["media"] or any(provenance["source_frames"].get(fid) != source_files[fid] for fid in selected):
            refuse("source file provenance differs")
        rows = []
        by_id = dict(zip(ids, records))
        for i, fid in enumerate(selected):
            original, record = originals[fid], by_id[fid]
            rgb, ident = reader.read(clip.frame(fid))
            if ident["image_sha256"] != record["source_rgb_sha256"] or (original.image_sha256 is not None and original.image_sha256 != ident["image_sha256"]):
                refuse("source decoded RGB differs")
            mask, mask_info = md._load_mask(store, fid, (height, width))
            if original.mask_state == "observed" and (mask is None or original.mask_frame_id != fid or original.mask_sha256 != mask_info["content_sha256"]):
                refuse("same-frame observed mask content differs")
            mask_state = "unavailable" if mask is None else "empty" if not mask.any() else "observed"
            mask_rgb = np.zeros((height, width, 3), np.uint8) if mask is None else np.repeat((mask.astype(np.uint8)*255)[:, :, None], 3, axis=2)
            y = 40+i*(tile_height+40)
            panels = {}
            for x, (name, panel) in enumerate((("source", rgb), ("mask", mask_rgb), ("composite", composites.pop(fid)))):
                rect = [x*tile_width, y+40, tile_width, tile_height]
                image.paste(Image.fromarray(panel).resize((tile_width, tile_height), Image.Resampling.NEAREST), tuple(rect[:2]))
                panels[name] = rect
            draw.text((4, y+2), f"{fid} source={original.source_frame_index} {original.pose_state}", fill="white")
            draw.text((4, y+17), f"{record['render_reason']} | mask={mask_state} | scale={tile_width}/{width}", fill="white")
            rows.append(dict(frame_id=fid, source_frame_index=original.source_frame_index,
                pose_state=original.pose_state, render_reason=record["render_reason"], mask_state=mask_state,
                original_mask_state=original.mask_state, panels=panels, source_rgb_sha256=ident["image_sha256"],
                decoded_rgb_sha256=record["decoded_rgb_sha256"], mask_sha256=None if mask_info is None else mask_info["content_sha256"],
                mask_file_sha256=mask_files[fid]))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
        if len(png) > MAX_PNG_BYTES:
            refuse("PNG exceeds 16 MiB")
        report = dict(schema=VERSION, status="completed", scope="artifact_inspection", new_neural_inference=False,
            quality_claim=False, independent_accuracy=None, clip_id=clip_id, run_id=manifest["run_id"], role=manifest["role"],
            limitation="Saved artifact inspection only; no new tracking admission, visibility proof or independent accuracy.",
            scheduled_frame_ids=ids, selected_frame_ids=selected, rows=rows,
            frame_accounting=dict(scheduled=len(ids), decoded=len(ids), selected=len(selected)),
            sheet=dict(file="review.png", width=sheet_width, height=sheet_height, tile_width=tile_width,
                tile_height=tile_height, resampling="nearest", bytes=len(png), sha256=_sha(png)),
            provenance=dict(manifest_sha256=expected_manifest_sha256, video=video_identity, clip_sha256=_sha(clip_raw),
                mask_manifest=mask_identity, mask_manifest_file_sha256=_sha(mask_raw), masks=mask_files,
                media=media, source_frames=source_files, sources=sources, config_sha256=None if config is None else _sha(config)),
            dependencies=dict(numpy=np.__version__, opencv=cv2.__version__, pillow=pillow_version))
        encoded = wc.canonical_json(report)
        if len(encoded) > MAX_JSON_BYTES:
            refuse("report exceeds 8 MiB")
        ws.require_budget(extra_derived=len(png)+len(encoded))
        staging = derived / (".export-review-"+uuid.uuid4().hex)
        staging.mkdir()
        (staging / "review.png").write_bytes(png)
        (staging / "report.json").write_bytes(encoded)
        if (_sha(md._read_bounded(manifest_path, 32 << 20)) != expected_manifest_sha256
                or _file_identity(video) != video_identity or md._read_bounded(clip_path, MAX_JSON_BYTES) != clip_raw
                or md._read_bounded(mask_path, MAX_JSON_BYTES) != mask_raw or _sources() != sources
                or (md._read_bounded(ws.config_path, MAX_JSON_BYTES) if ws.config_path.exists() else None) != config):
            refuse("input/source closure changed before publication")
        fresh = cli.MaskStore(ws, clip_id, mask_set)
        if fresh.identity() != mask_identity or fresh.verify(selected) != mask_files:
            refuse("mask registry/files changed before publication")
        if reader.verify_media() != media or reader.verify_frames(selected) != source_files:
            refuse("source media changed before publication")
        for row in rows:
            if reader.read(clip.frame(row["frame_id"]))[1]["image_sha256"] != row["source_rgb_sha256"]:
                refuse("source decoded pixels changed before publication")
        ws.require_budget()
        _publish_new_directory(staging, output)
        staging = None
        return report
    finally:
        reader.close()
        if staging is not None:
            shutil.rmtree(staging)
