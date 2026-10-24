"""Observed CNOS selection proposal and explicit review finalization.

This adapter treats detector output as observed foreground evidence. It never
reads pose labels, renderer masks or evaluator sidecars, and it never creates a
review receipt. Heavy runner/image/model dependencies are imported only inside
the operations that need them.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
from zipfile import BadZipFile, ZipFile

import numpy as np


PROPOSAL_SCHEMA_VERSION = 1
PROPOSAL_KIND = "quality-ranker-initialization-proposal-v1"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_KIND = "quality-ranker-initialization-review-receipt-v1"
FINALIZATION_SCHEMA_VERSION = 1
MAX_PHYSICAL_SOURCE_PIXELS = 16_777_216
MAX_DETECTOR_NPZ_BYTES = 64 * 1024 * 1024
MAX_DETECTOR_UNPACKED_BYTES = 32 * 1024 * 1024
MAX_PREVIEW_BYTES = 64 * 1024 * 1024
MAX_FINAL_BUNDLE_BYTES = 256 * 1024 * 1024
REVIEW_STATEMENT = (
    "Codex agent visually inspected the RGB, observed-mask overlay and selected point; "
    "this is not independent human accuracy evidence."
)
_DETECTION_FIELDS = {
    "mask", "reason", "diagnostics", "frameId", "sourceFrameId", "timestamp_s",
    "clock_mode", "input_manifest_sha256", "timestamp_table_sha256",
}
_PROPOSAL_FIELDS = {
    "schema_version", "kind", "created_at_utc", "source_binding", "timing_binding",
    "rgb_binding", "detector_binding", "cnos_pins", "selection", "preview_binding",
    "proposal_sha256",
}
_RECEIPT_FIELDS = {
    "schema_version", "kind", "proposal_sha256", "reviewed_at_utc", "decision",
    "reviewer_kind", "reviewer_identifier", "statement", "source_binding",
    "timing_binding", "rgb_binding", "detector_binding", "selection", "preview_binding",
    "cnos_pins",
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_unique(path, *, maximum=2 * 1024 * 1024):
    raw = Path(path).read_bytes()
    if len(raw) > maximum:
        raise ValueError("Initialization JSON exceeds its byte limit")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid initialization JSON: {path}") from exc
    return value, raw


def _parse_unique_json(raw, name):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {name}") from exc


def _read_npz_scalar(archive, name):
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype.kind not in ("U", "S", "i", "u", "f"):
        raise ValueError(f"CNOS detector field {name} must be a non-object scalar")
    item = value.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"CNOS detector field {name} is not UTF-8") from exc
    return item


def _validate_timestamp(value, name):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite timestamp")
    return float(value)


def _read_cnos_pins(manifest, diagnostics):
    """Bind the exact CNOS bank, locked weights, lock and source adapters."""
    from .quality_assets import CACHE, digest

    lock_path = CACHE / "lock.json"
    lock_raw = lock_path.read_bytes()
    lock = json.loads(lock_raw.decode("utf-8"))
    models = lock.get("models")
    if not isinstance(models, dict):
        raise ValueError("CNOS model lock lacks a models inventory")
    checkpoints = {}
    for name in ("FastSAM-x.pt", "dinov2_vitl14_pretrain.pth"):
        entry = models.get(name)
        if not isinstance(entry, dict) or type(entry.get("sha256")) is not str:
            raise ValueError(f"CNOS model lock lacks the {name} digest")
        path = CACHE / "checkpoints" / name
        actual = digest(path)
        expected = entry["sha256"].lower()
        if actual.lower() != expected:
            raise ValueError(f"CNOS model checkpoint changed: {name}")
        checkpoints[name] = actual
    bank_path = CACHE / "banks" / f"{manifest['object']}-cnos.npz"
    bank_sha = digest(bank_path)
    diagnostics_bank = diagnostics.get("bank_sha256")
    if type(diagnostics_bank) is not str or diagnostics_bank.lower() != bank_sha.lower():
        raise ValueError("CNOS detector bank digest differs from the current descriptor bank")
    with np.load(bank_path, allow_pickle=False) as bank:
        bank_asset = _read_npz_scalar(bank, "asset_sha256")
    if str(bank_asset).lower() != manifest["source_hashes"][manifest["asset"]].lower():
        raise ValueError("CNOS descriptor bank is bound to another metric asset")
    module_dir = Path(__file__).resolve().parent
    return {
        "model_lock_sha256": _sha256_bytes(lock_raw),
        "checkpoints": checkpoints,
        "descriptor_bank_sha256": bank_sha,
        "descriptor_bank_asset_sha256": str(bank_asset).lower(),
        "quality_cnos_source_sha256": _sha256_file(module_dir / "quality_cnos.py"),
        "quality_runner_source_sha256": _sha256_file(module_dir / "quality_runner.py"),
    }


def _interior_positive_point(mask):
    """Choose a maximum-clearance foreground pixel, with row-major tie breaking."""
    binary = np.asarray(mask)
    if binary.ndim != 2 or binary.dtype != np.uint8 or not np.isin(binary, (0, 255)).all():
        raise ValueError("Observed mask must be native HxW uint8 binary 0/255")
    foreground = binary == 255
    if not np.any(foreground):
        return None
    # scipy is already a pinned runtime dependency. The false border models the
    # contract's implicit one-pixel background outside the native image.
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    distance = distance_transform_edt(padded)[1:-1, 1:-1]
    distance[~foreground] = -1.
    maximum = float(np.max(distance))
    y, x = np.argwhere(distance == maximum)[0]
    return [int(x), int(y)]


def _make_preview(rgb, mask, point):
    from .vision import cv2

    image = np.asarray(rgb)
    binary = np.asarray(mask) == 255
    overlay_color = np.array([255., 32., 32.], dtype=np.float64)
    preview = image.copy()
    if np.any(binary):
        preview[binary] = np.clip(np.rint(
            image[binary].astype(np.float64) * 0.55 + overlay_color * 0.45),
            0, 255).astype(np.uint8)
    bgr = preview[..., ::-1].copy()
    x, y = point
    radius = max(3, min(image.shape[0], image.shape[1]) // 160)
    cv2.circle(bgr, (x, y), radius, (0, 255, 0), thickness=-1,
               lineType=cv2.LINE_8)
    ok, encoded = cv2.imencode(
        ".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError("OpenCV could not encode the bounded selection preview")
    raw = encoded.tobytes()
    if len(raw) > MAX_PREVIEW_BYTES:
        raise ValueError("Observed-mask review preview exceeds its 64 MiB bound")
    return raw, {
        "sha256": _sha256_bytes(raw), "format": "PNG",
        "width": int(image.shape[1]), "height": int(image.shape[0]),
        "channel_order": "RGB", "overlay_rgb": [255, 32, 32],
        "overlay_mix": [0.55, 0.45], "point_rgb": [0, 255, 0],
        "point_radius_px": radius, "png_compression": 3,
    }


def _auth_raw_bundle(bundle, detector_npz):
    """Authenticate raw physical input and detector identity through the runner."""
    from . import quality_runner
    from .quality_time import PHYSICAL

    bundle_argument = Path(bundle)
    if bundle_argument.is_symlink():
        raise ValueError("Raw inference bundle path cannot be a symlink")
    bundle = Path(bundle).resolve(strict=True)
    detector_npz = Path(detector_npz).resolve(strict=True)
    input_path = bundle / "input.json"
    if (input_path.is_symlink() or not input_path.is_file() or
            input_path.stat().st_size > 2 * 1024 * 1024):
        raise ValueError("Raw input.json must be a regular file no larger than 2 MiB")
    raw_manifest = input_path.read_bytes()
    pre_manifest = _parse_unique_json(raw_manifest, "raw inference manifest")
    if not isinstance(pre_manifest, dict):
        raise ValueError("Raw inference manifest must be a JSON object")
    pre_video, pre_asset = pre_manifest.get("video"), pre_manifest.get("asset")
    pre_hashes = pre_manifest.get("source_hashes")
    if (type(pre_video) is not str or type(pre_asset) is not str or
            not isinstance(pre_hashes, dict) or set(pre_hashes) != {pre_video, pre_asset}):
        raise ValueError("Raw physical input must hash only its declared video and metric asset")
    for resource_name in (pre_video, pre_asset):
        relative = Path(resource_name)
        if (relative.is_absolute() or ".." in relative.parts or
                len(relative.parts) != 1 or relative.name != resource_name):
            raise ValueError("Raw source resources must be direct contained files")
    expected_entries = {"input.json", pre_video, pre_asset}
    if len(expected_entries) != 3:
        raise ValueError("Raw input video and asset names must be distinct")
    initial_entries = list(bundle.iterdir())
    if (any(entry.is_symlink() or not entry.is_file() for entry in initial_entries) or
            {entry.name for entry in initial_entries} != expected_entries):
        raise ValueError("Raw inference bundle must contain only input.json, its video and metric asset")

    manifest = quality_runner.read_input(bundle)
    timing = manifest.get("_timing")
    if not isinstance(timing, dict) or timing.get("clock_mode") != PHYSICAL:
        raise ValueError("CNOS initialization requires an authenticated physical-time bundle")
    if "selection" in manifest:
        raise ValueError("Initialization proposal requires the unselected raw input bundle")
    resolution = manifest.get("native_resolution")
    if (not isinstance(resolution, list) or len(resolution) != 2 or
            any(type(value) is not int or value <= 0 for value in resolution) or
            resolution[0] * resolution[1] > MAX_PHYSICAL_SOURCE_PIXELS):
        raise ValueError("Raw initialization frame exceeds the bounded physical pixel policy")
    video_name, asset_name = manifest.get("video"), manifest.get("asset")
    source_hashes = manifest.get("source_hashes")
    if (type(video_name) is not str or type(asset_name) is not str or
            not isinstance(source_hashes, dict) or set(source_hashes) != {video_name, asset_name}):
        raise ValueError("Raw physical input must hash only its declared video and metric asset")
    if Path(video_name).is_absolute() or Path(asset_name).is_absolute() or ".." in Path(video_name).parts or ".." in Path(asset_name).parts:
        raise ValueError("Raw source resources must use contained relative paths")
    allowed_names = {"input.json", video_name, asset_name}
    if len(allowed_names) != 3:
        raise ValueError("Raw input video and asset names must be distinct")
    bundle_bytes = len(raw_manifest)
    for resource_name in (video_name, asset_name):
        resource_path = bundle / resource_name
        size = resource_path.stat().st_size
        bundle_bytes += size
    if bundle_bytes > MAX_FINAL_BUNDLE_BYTES:
        raise ValueError("Raw three-file bundle exceeds the 256 MiB initialization bound")
    entries = list(bundle.iterdir())
    if (any(entry.is_symlink() or not entry.is_file() for entry in entries) or
            {entry.name for entry in entries} != allowed_names):
        raise ValueError("Raw inference bundle must contain only input.json, its video and metric asset")
    source_paths = {}
    for resource_name, relative in (("video", video_name), ("asset", asset_name)):
        source = (bundle / relative).resolve(strict=True)
        if not source.is_relative_to(bundle) or not source.is_file():
            raise ValueError("Raw source path is missing or escapes its bundle")
        source_paths[resource_name] = source
        expected = source_hashes[relative].lower()
        if _sha256_file(source) != expected:
            raise ValueError(f"Raw {resource_name} changed after runner authentication")

    timeline = manifest.get("timeline")
    if not isinstance(timeline, list) or not timeline or timeline[0].get("frame_id") != 0:
        raise ValueError("Initialization requires planned ordinal 0 as the first timing row")
    first = timeline[0]
    frame_id = first.get("frame_id")
    source_frame_id = first.get("source_frame_id")
    timestamp = _validate_timestamp(first.get("timestamp_s"), "first planned timestamp")
    if type(frame_id) is not int or frame_id != 0 or type(source_frame_id) is not int:
        raise ValueError("First planned physical row must bind ordinal 0 and an integer source frame ID")
    expected_source_id = timing.get("source_frame_ids", {}).get(0)
    if expected_source_id != source_frame_id:
        raise ValueError("First timeline source ID differs from the authenticated timing table")
    rgb = quality_runner.read_rgb(bundle, manifest, 0)
    rgb = np.asarray(rgb)
    resolution = manifest.get("native_resolution")
    if (rgb.ndim != 3 or rgb.dtype != np.uint8 or rgb.shape[2] != 3 or
            not isinstance(resolution, list) or len(resolution) != 2 or
            rgb.shape != (resolution[1], resolution[0], 3)):
        raise ValueError("Runner-decoded first frame differs from declared native RGB resolution")
    rgb = np.ascontiguousarray(rgb)
    rgb_digest = _sha256_bytes(rgb.tobytes(order="C"))

    if not detector_npz.is_file() or detector_npz.is_symlink() or detector_npz.stat().st_size > MAX_DETECTOR_NPZ_BYTES:
        raise ValueError("CNOS detector NPZ must be a regular file no larger than 64 MiB")
    detector_raw = detector_npz.read_bytes()
    if len(detector_raw) > MAX_DETECTOR_NPZ_BYTES:
        raise ValueError("CNOS detector NPZ grew beyond its 64 MiB bound")
    try:
        with ZipFile(io.BytesIO(detector_raw)) as archive_file:
            infos = archive_file.infolist()
            expected_npz_names = {name + ".npy" for name in _DETECTION_FIELDS}
            if (len(infos) != len(expected_npz_names) or
                    {info.filename for info in infos} != expected_npz_names or
                    sum(info.file_size for info in infos) > MAX_DETECTOR_UNPACKED_BYTES):
                raise ValueError("CNOS detector NPZ has an unexpected or oversized archive inventory")
    except BadZipFile as exc:
        raise ValueError("CNOS detector evidence is not a valid NPZ archive") from exc
    with np.load(io.BytesIO(detector_raw), allow_pickle=False) as archive:
        if set(archive.files) != _DETECTION_FIELDS:
            raise ValueError("CNOS detector NPZ has unknown or missing identity/diagnostic fields")
        mask = np.asarray(archive["mask"])
        reason_value = _read_npz_scalar(archive, "reason")
        diagnostics_value = _read_npz_scalar(archive, "diagnostics")
        detector_identity = {
            "frame_id": _read_npz_scalar(archive, "frameId"),
            "source_frame_id": _read_npz_scalar(archive, "sourceFrameId"),
            "timestamp_s": _read_npz_scalar(archive, "timestamp_s"),
            "clock_mode": _read_npz_scalar(archive, "clock_mode"),
            "input_manifest_sha256": _read_npz_scalar(archive, "input_manifest_sha256"),
            "timestamp_table_sha256": _read_npz_scalar(archive, "timestamp_table_sha256"),
        }
        mask = mask.copy()
    if (mask.shape != rgb.shape[:2] or mask.dtype != np.uint8 or
            not np.isin(mask, (0, 255)).all()):
        raise ValueError("CNOS output must be native HxW uint8 binary 0/255")
    try:
        diagnostics_raw = str(diagnostics_value).encode("utf-8")
        diagnostics = _parse_unique_json(diagnostics_raw, "CNOS diagnostics")
    except (UnicodeEncodeError, json.JSONDecodeError) as exc:
        raise ValueError("CNOS diagnostics must contain valid JSON") from exc
    if not isinstance(diagnostics, dict):
        raise ValueError("CNOS diagnostics must be a JSON object")
    if "reason" not in diagnostics:
        raise ValueError("CNOS diagnostics must explicitly report its reason field")
    expected_identity = {
        "frame_id": 0, "source_frame_id": source_frame_id,
        "timestamp_s": timestamp, "clock_mode": PHYSICAL,
        "input_manifest_sha256": timing["manifest_sha256"],
        "timestamp_table_sha256": timing["timestamp_table_sha256"],
    }
    for key, expected in expected_identity.items():
        actual = detector_identity[key]
        if key == "timestamp_s":
            actual = _validate_timestamp(actual, "CNOS timestamp_s")
        elif key in ("frame_id", "source_frame_id") and type(actual) is not int:
            raise ValueError(f"CNOS {key} identity must be an integer")
        elif key in ("clock_mode", "input_manifest_sha256", "timestamp_table_sha256") and type(actual) is not str:
            raise ValueError(f"CNOS {key} identity must be a string")
        if actual != expected:
            raise ValueError(f"CNOS detector identity mismatch for {key}")
    diagnostic_reason = diagnostics.get("reason")
    if (reason_value is not None and type(reason_value) is not str or
            diagnostic_reason is not None and type(diagnostic_reason) is not str):
        raise ValueError("CNOS reason fields must be empty strings or null")
    detector_reason = reason_value or ""
    if detector_reason != (diagnostic_reason or ""):
        raise ValueError("CNOS top-level and diagnostic reason fields disagree")
    no_detection = bool(detector_reason) or not np.any(mask == 255)

    # Reauthenticate source streams, manifest and the decoded first frame before
    # returning the handoff record. This catches concurrent source mutation.
    if input_path.read_bytes() != raw_manifest:
        raise ValueError("Raw input manifest changed during CNOS proposal authentication")
    for key, source in source_paths.items():
        relative = manifest["video"] if key == "video" else manifest["asset"]
        if _sha256_file(source) != source_hashes[relative].lower():
            raise ValueError(f"Raw {key} changed during CNOS proposal authentication")
    rgb_again = np.ascontiguousarray(quality_runner.read_rgb(bundle, manifest, 0))
    if rgb_again.shape != rgb.shape or _sha256_bytes(rgb_again.tobytes(order="C")) != rgb_digest:
        raise ValueError("Runner-decoded first RGB frame changed during proposal authentication")
    if _sha256_file(detector_npz) != _sha256_bytes(detector_raw):
        raise ValueError("CNOS detector evidence changed during proposal authentication")
    if no_detection:
        return {
            "status": "no_detection",
            "reason": (str(detector_reason) if detector_reason else "empty_observed_foreground"),
            "detector_npz_sha256": _sha256_bytes(detector_raw),
            "mask_sha256": _sha256_bytes(np.ascontiguousarray(mask).tobytes(order="C")),
            "rgb_sha256": rgb_digest, "diagnostics": diagnostics,
        }
    point = _interior_positive_point(mask)
    return {
        "bundle": bundle, "manifest": manifest, "raw_manifest": raw_manifest,
        "timing": timing, "source_paths": source_paths,
        "source_binding": {
            "input_manifest_sha256": timing["manifest_sha256"],
            "video_sha256": source_hashes[video_name].lower(),
            "asset_sha256": source_hashes[asset_name].lower(),
        },
        "timing_binding": {
            "clock_mode": PHYSICAL, "timestamp_table_sha256": timing["timestamp_table_sha256"],
            "frame_id": 0, "source_frame_id": source_frame_id,
            "timestamp_s": timestamp,
        },
        "rgb": rgb, "rgb_binding": {
            "sha256": rgb_digest, "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]), "channels": 3,
            "channel_order": "RGB", "dtype": "uint8", "memory_order": "C",
        },
        "mask": mask, "point": point, "diagnostics": diagnostics,
        "detector_binding": {
            "npz_sha256": _sha256_bytes(detector_raw),
            "mask_sha256": _sha256_bytes(np.ascontiguousarray(mask).tobytes(order="C")),
            "reason": "", "identity": detector_identity,
        },
    }


def create_proposal(bundle, detector_npz, proposal_path, preview_path):
    """Create an unsigned-by-review observed selection proposal and preview."""
    raw = _auth_raw_bundle(bundle, detector_npz)
    if raw.get("status") == "no_detection":
        return raw
    preview_bytes, preview_binding = _make_preview(raw["rgb"], raw["mask"], raw["point"])
    manifest = raw["manifest"]
    cnos_pins = _read_cnos_pins(manifest, raw["diagnostics"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    proposal = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "kind": PROPOSAL_KIND,
        "created_at_utc": now,
        "source_binding": raw["source_binding"],
        "timing_binding": raw["timing_binding"],
        "rgb_binding": raw["rgb_binding"],
        "detector_binding": raw["detector_binding"],
        "cnos_pins": cnos_pins,
        "selection": {"points": [raw["point"]], "labels": [1]},
        "preview_binding": preview_binding,
    }
    proposal["proposal_sha256"] = _sha256_bytes(_json_bytes(proposal))
    if set(proposal) != _PROPOSAL_FIELDS:
        raise RuntimeError("Internal proposal schema drift")
    proposal_path, preview_path = Path(proposal_path), Path(preview_path)
    proposal_resolved = proposal_path.resolve(strict=False)
    preview_resolved = preview_path.resolve(strict=False)
    if proposal_resolved == preview_resolved:
        raise ValueError("Proposal JSON and preview require different paths")
    if (proposal_resolved.is_relative_to(raw["bundle"]) or
            preview_resolved.is_relative_to(raw["bundle"])):
        raise ValueError("Proposal and review preview must remain outside the inference bundle")
    if proposal_path.exists() or proposal_path.is_symlink() or preview_path.exists() or preview_path.is_symlink():
        raise FileExistsError("Initialization proposal and preview destinations must be fresh")
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    claims, created = [], []
    try:
        claims.append(_claim(preview_path))
        claims.append(_claim(proposal_path))

        def write_owned(path, raw_bytes):
            with Path(path).open("xb") as stream:
                created.append(Path(path))
                stream.write(raw_bytes)

        write_owned(preview_path, preview_bytes)
        write_owned(proposal_path, _json_bytes(proposal) + b"\n")
    except Exception:
        # An incomplete proposal/preview pair cannot be reviewed. Remove only
        # paths this call created; detector evidence remains untouched.
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for claim in reversed(claims):
            try:
                claim.unlink()
            except FileNotFoundError:
                pass
    return {"status": "proposal_created", "proposal": str(proposal_path),
            "preview": str(preview_path), "proposal_sha256": proposal["proposal_sha256"],
            "preview_sha256": preview_binding["sha256"],
            "selection": proposal["selection"], "reviewed": False}


def _write_new_bytes(path, raw):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(raw)


def _validate_proposal(proposal):
    if not isinstance(proposal, dict) or set(proposal) != _PROPOSAL_FIELDS:
        raise ValueError("Proposal has unknown or missing fields")
    if type(proposal.get("schema_version")) is not int or proposal["schema_version"] != PROPOSAL_SCHEMA_VERSION or proposal.get("kind") != PROPOSAL_KIND:
        raise ValueError("Unsupported initialization proposal schema")
    if "reviewed" in proposal or "reviewed:true" in json.dumps(proposal).lower():
        raise ValueError("Proposal cannot declare itself reviewed")
    sha = proposal.get("proposal_sha256")
    if type(sha) is not str or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("Proposal self-digest must be a lowercase SHA-256")
    unsigned = {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    if _sha256_bytes(_json_bytes(unsigned)) != sha:
        raise ValueError("Initialization proposal digest mismatch")
    _parse_utc(proposal.get("created_at_utc"), "proposal created_at_utc")
    return proposal


def _parse_utc(value, name):
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp ending in Z") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must use the UTC offset")
    return parsed


def _validate_review_receipt(receipt, proposal):
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("Review receipt has unknown or missing fields")
    if (type(receipt.get("schema_version")) is not int or
            receipt["schema_version"] != RECEIPT_SCHEMA_VERSION or
            receipt.get("kind") != RECEIPT_KIND):
        raise ValueError("Unsupported visual review receipt schema")
    if receipt.get("proposal_sha256") != proposal["proposal_sha256"]:
        raise ValueError("Visual review receipt belongs to a different proposal")
    review_time = _parse_utc(receipt.get("reviewed_at_utc"), "reviewed_at_utc")
    if review_time < _parse_utc(proposal["created_at_utc"], "proposal created_at_utc"):
        raise ValueError("Visual review receipt predates its proposal")
    if receipt.get("decision") != "approve":
        raise ValueError("Visual review receipt did not explicitly approve the selection")
    if receipt.get("reviewer_kind") != "codex_agent":
        raise ValueError("Visual review receipt must identify a Codex agent reviewer")
    if type(receipt.get("reviewer_identifier")) is not str or not receipt["reviewer_identifier"].strip():
        raise ValueError("Visual review receipt requires a reviewer identifier")
    if receipt.get("statement") != REVIEW_STATEMENT:
        raise ValueError("Visual review receipt must state that this is not independent human accuracy evidence")
    for field in ("source_binding", "timing_binding", "rgb_binding", "detector_binding",
                  "cnos_pins", "selection", "preview_binding"):
        if receipt.get(field) != proposal.get(field):
            raise ValueError(f"Visual review receipt {field} differs from the proposal")
    return receipt


def _verify_current_proposal(bundle, detector_npz, proposal, preview_path):
    raw = _auth_raw_bundle(bundle, detector_npz)
    if raw.get("status") == "no_detection":
        raise ValueError("Current detector evidence has no observed foreground")
    expected = {
        "source_binding": raw["source_binding"],
        "timing_binding": raw["timing_binding"],
        "rgb_binding": raw["rgb_binding"],
        "detector_binding": raw["detector_binding"],
        "selection": {"points": [raw["point"]], "labels": [1]},
    }
    for field, value in expected.items():
        if proposal.get(field) != value:
            raise ValueError(f"Initialization proposal is stale or altered: {field}")
    current_pins = _read_cnos_pins(raw["manifest"], raw["diagnostics"])
    if proposal.get("cnos_pins") != current_pins:
        raise ValueError("Initialization proposal CNOS bank/model/source pins changed")
    preview_bytes, current_preview = _make_preview(raw["rgb"], raw["mask"], raw["point"])
    if proposal.get("preview_binding") != current_preview:
        raise ValueError("Initialization proposal preview metadata changed")
    preview_path = Path(preview_path).resolve(strict=True)
    if _sha256_file(preview_path) != current_preview["sha256"] or preview_path.read_bytes() != preview_bytes:
        raise ValueError("Review preview bytes differ from the proposal-bound RGB/mask overlay")
    proposal_selection = proposal["selection"]
    if (not isinstance(proposal_selection, dict) or
            set(proposal_selection) != {"points", "labels"} or
            proposal_selection != expected["selection"]):
        raise ValueError("Proposal selection must be its deterministic observed interior point")
    return raw


def _claim(path):
    path = Path(path)
    claim = path.with_name(path.name + ".claim")
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    return claim


def finalize_selection(bundle, detector_npz, proposal_path, preview_path,
                       receipt_path, output_bundle, *, finalization_receipt_path=None):
    """Reauthenticate an approved receipt and promote a fresh three-file bundle."""
    proposal_path = Path(proposal_path).resolve(strict=True)
    receipt_path = Path(receipt_path).resolve(strict=True)
    preview_path = Path(preview_path).resolve(strict=True)
    detector_npz = Path(detector_npz).resolve(strict=True)
    proposal, proposal_raw = _read_json_unique(proposal_path)
    _validate_proposal(proposal)
    receipt, receipt_raw = _read_json_unique(receipt_path)
    _validate_review_receipt(receipt, proposal)
    proposal_file_sha = _sha256_bytes(proposal_raw)
    review_receipt_file_sha = _sha256_bytes(receipt_raw)
    raw = _verify_current_proposal(bundle, detector_npz, proposal, preview_path)

    output_bundle = Path(output_bundle).resolve(strict=False)
    finalization_receipt = (Path(finalization_receipt_path).absolute()
        if finalization_receipt_path is not None else
        Path(receipt_path).absolute().with_name(Path(receipt_path).name + ".finalization.json"))
    finalization_receipt = finalization_receipt.resolve(strict=False)
    if finalization_receipt == output_bundle or finalization_receipt.is_relative_to(output_bundle):
        raise ValueError("Finalization receipt must be outside the inference bundle")
    output_bundle_resolved = output_bundle
    finalization_receipt_resolved = finalization_receipt
    if finalization_receipt.is_relative_to(raw["bundle"]):
        raise ValueError("Finalization receipt must remain outside the raw inference bundle")
    source_paths = {
        raw["bundle"] / "input.json",
        raw["source_paths"]["video"],
        raw["source_paths"]["asset"],
        proposal_path, receipt_path, preview_path, detector_npz,
    }
    if finalization_receipt_resolved in source_paths:
        raise ValueError("Finalization receipt path collides with a source or review artifact")
    if output_bundle_resolved in source_paths or any(
            path.is_relative_to(output_bundle_resolved) for path in source_paths):
        raise ValueError("Finalized bundle path collides with a source or review artifact")
    if output_bundle.exists() or output_bundle.is_symlink() or finalization_receipt.exists() or finalization_receipt.is_symlink():
        raise FileExistsError("Finalized bundle and receipt destinations must be fresh")
    if output_bundle.resolve(strict=False).is_relative_to(raw["bundle"]):
        raise ValueError("Finalized bundle cannot be placed inside the raw inference bundle")
    if finalization_receipt.resolve(strict=False).is_relative_to(output_bundle.resolve(strict=False)):
        raise ValueError("Finalization receipt must remain outside the inference bundle")

    raw_manifest = _parse_unique_json(raw["raw_manifest"], "raw inference manifest")
    if "selection" in raw_manifest:
        raise ValueError("Raw inference manifest already contains a selection")
    if raw_manifest.get("video") != "synthetic.mp4" or raw_manifest.get("asset") != "object.glb":
        raise ValueError("Finalization supports the approved synthetic.mp4/object.glb source names")
    selection = {
        "reviewed": True,
        "frame_id": 0,
        "source_frame_id": raw["timing_binding"]["source_frame_id"],
        "timestamp_s": raw["timing_binding"]["timestamp_s"],
        "points": proposal["selection"]["points"],
        "labels": proposal["selection"]["labels"],
    }
    final_manifest = dict(raw_manifest)
    final_manifest["selection"] = selection
    final_manifest_bytes = _json_bytes(final_manifest) + b"\n"
    planned_bundle_bytes = len(final_manifest_bytes) + sum(
        source.stat().st_size for source in raw["source_paths"].values())
    if planned_bundle_bytes > MAX_FINAL_BUNDLE_BYTES:
        raise ValueError("Finalized three-file bundle exceeds the 256 MiB initialization bound")

    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    finalization_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_claim = receipt_claim = None
    stage_bundle = Path(tempfile.mkdtemp(prefix=f".{output_bundle.name}.stage-",
                                         dir=output_bundle.parent))
    stage_receipt = finalization_receipt.with_name(
        f".{finalization_receipt.name}.stage-{os.getpid()}")
    promoted_bundle = False
    stage_receipt_created = False
    try:
        output_claim = _claim(output_bundle)
        receipt_claim = _claim(finalization_receipt)
        if stage_receipt.exists() or stage_receipt.is_symlink():
            raise FileExistsError("Finalization receipt staging path is already in use")
        source_specs = (
            (raw["source_paths"]["video"], stage_bundle / "synthetic.mp4",
             raw["source_binding"]["video_sha256"]),
            (raw["source_paths"]["asset"], stage_bundle / "object.glb",
             raw["source_binding"]["asset_sha256"]),
        )
        for source, target, expected_sha in source_specs:
            with source.open("rb") as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
            if _sha256_file(target) != expected_sha or _sha256_file(source) != expected_sha:
                raise ValueError("Raw source changed while copying the reviewed bundle")
        if sum(path.stat().st_size for path in stage_bundle.iterdir() if path.is_file()) + len(final_manifest_bytes) > MAX_FINAL_BUNDLE_BYTES:
            raise ValueError("Staged three-file bundle exceeds the 256 MiB initialization bound")
        input_path = stage_bundle / "input.json"
        _write_new_bytes(input_path, final_manifest_bytes)
        expected_names = ["input.json", "object.glb", "synthetic.mp4"]
        if sorted(path.name for path in stage_bundle.iterdir()) != expected_names:
            raise RuntimeError("Finalized inference bundle must contain exactly three files")

        # Run the same source/timing and SAM physical-selection preflight used by
        # actual inference before promoting any bundle bytes.
        from .quality_runner import read_input
        from .quality_sam2 import prepare_cache_plan

        parsed = read_input(stage_bundle)
        if parsed.get("selection") != selection:
            raise ValueError("Runner changed the reviewed selection identity")
        if _parse_unique_json(input_path.read_bytes(), "final input manifest") != final_manifest:
            raise ValueError("Final input fields differ from the approved manifest plus selection")
        cache_plan = prepare_cache_plan(final_manifest_bytes)
        if cache_plan.get("selection") != {
                "points": selection["points"], "labels": selection["labels"], "reviewed": True}:
            raise ValueError("SAM preflight changed the reviewed positive selection")
        for _source, target, expected_sha in source_specs:
            if _sha256_file(target) != expected_sha:
                raise ValueError("Finalized source bytes changed after runner/SAM preflight")
        final_input_sha = _sha256_bytes(final_manifest_bytes)
        final_receipt = {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "kind": "quality-ranker-initialization-finalization-v1",
            "proposal_sha256": proposal["proposal_sha256"],
            "review_receipt_sha256": review_receipt_file_sha,
            "raw_input_manifest_sha256": raw["source_binding"]["input_manifest_sha256"],
            "final_input_manifest_sha256": final_input_sha,
            "timestamp_table_sha256": raw["timing_binding"]["timestamp_table_sha256"],
            "rgb_pixel_sha256": raw["rgb_binding"]["sha256"],
            "detector_npz_sha256": raw["detector_binding"]["npz_sha256"],
            "observed_mask_sha256": raw["detector_binding"]["mask_sha256"],
            "review_preview_sha256": proposal["preview_binding"]["sha256"],
            "selection": selection,
            "bundle_inventory": expected_names,
            "bundle_path": output_bundle.name,
            "bundle_files_sha256": {
                "input.json": final_input_sha,
                "synthetic.mp4": raw["source_binding"]["video_sha256"],
                "object.glb": raw["source_binding"]["asset_sha256"],
            },
        }
        with stage_receipt.open("xb") as stream:
            stage_receipt_created = True
            stream.write(_json_bytes(final_receipt) + b"\n")

        # Repeat the full authentication after staging and preflight. Bind the
        # finalization record to the exact bytes loaded and reviewed above.
        current_raw = _verify_current_proposal(bundle, detector_npz, proposal, preview_path)
        for field in ("source_binding", "timing_binding", "rgb_binding", "detector_binding"):
            if current_raw[field] != raw[field]:
                raise ValueError(f"Initialization source changed before promotion: {field}")
        if current_raw["raw_manifest"] != raw["raw_manifest"]:
            raise ValueError("Raw input manifest changed before promotion")
        if _sha256_file(proposal_path) != proposal_file_sha:
            raise ValueError("Initialization proposal bytes changed before promotion")
        if _sha256_file(receipt_path) != review_receipt_file_sha:
            raise ValueError("Visual review receipt bytes changed before promotion")
        if _sha256_file(preview_path) != proposal["preview_binding"]["sha256"]:
            raise ValueError("Review preview bytes changed before promotion")
        if _sha256_file(detector_npz) != raw["detector_binding"]["npz_sha256"]:
            raise ValueError("CNOS detector evidence changed before promotion")
        if (raw["bundle"] / "input.json").read_bytes() != raw["raw_manifest"]:
            raise ValueError("Raw input manifest bytes changed before promotion")
        for resource_name, source in raw["source_paths"].items():
            expected_sha = raw["source_binding"][f"{resource_name}_sha256"]
            if _sha256_file(source) != expected_sha:
                raise ValueError(f"Raw {resource_name} changed before promotion")
        for _source, target, expected_sha in source_specs:
            if _sha256_file(target) != expected_sha:
                raise ValueError("Finalized source bytes changed immediately before promotion")

        if output_bundle.exists() or finalization_receipt.exists():
            raise FileExistsError("Finalized bundle or receipt destination was claimed concurrently")
        os.replace(stage_bundle, output_bundle)
        promoted_bundle = True
        try:
            os.replace(stage_receipt, finalization_receipt)
        except Exception:
            os.replace(output_bundle, stage_bundle)
            promoted_bundle = False
            raise
        return {"status": "finalized", "bundle": str(output_bundle),
                "receipt": str(finalization_receipt),
                "final_input_manifest_sha256": final_input_sha,
                "bundle_inventory": expected_names}
    except Exception as exc:
        if stage_receipt_created and stage_receipt.exists():
            try:
                stage_receipt.unlink()
            except OSError:
                pass
        if stage_bundle.exists():
            partial = sorted({path.name: path.stat().st_size
                              for path in stage_bundle.rglob("*") if path.is_file()}.items())
            try:
                _write_new_bytes(stage_bundle / "incomplete.json", _json_bytes({
                    "schema_version": 1, "status": "incomplete", "promoted": False,
                    "error_type": type(exc).__name__, "error": str(exc)[:512],
                    "partial_file_bytes": dict(partial),
                    "proposal_sha256": proposal["proposal_sha256"],
                }) + b"\n")
            except Exception:
                pass
        if promoted_bundle and output_bundle.exists() and not finalization_receipt.exists():
            try:
                os.replace(output_bundle, stage_bundle)
            except Exception:
                pass
        raise
    finally:
        for claim in (output_claim, receipt_claim):
            if claim is not None:
                try:
                    claim.unlink()
                except FileNotFoundError:
                    pass


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose", help="create an observed-mask review proposal")
    propose.add_argument("--bundle", type=Path, required=True)
    propose.add_argument("--detector-npz", type=Path, required=True,
                         help="existing quality_runner detect --frame 0 output")
    propose.add_argument("--proposal", type=Path, required=True)
    propose.add_argument("--preview", type=Path, required=True)
    finalize = commands.add_parser("finalize", help="finalize only a parent-authored visual-review receipt")
    finalize.add_argument("--bundle", type=Path, required=True)
    finalize.add_argument("--detector-npz", type=Path, required=True)
    finalize.add_argument("--proposal", type=Path, required=True)
    finalize.add_argument("--preview", type=Path, required=True)
    finalize.add_argument("--review-receipt", type=Path, required=True,
                          help="explicit parent-authored Codex visual review receipt")
    finalize.add_argument("--output-bundle", type=Path, required=True)
    finalize.add_argument("--finalization-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "propose":
        result = create_proposal(args.bundle, args.detector_npz,
                                 args.proposal, args.preview)
    else:
        result = finalize_selection(
            args.bundle, args.detector_npz, args.proposal, args.preview,
            args.review_receipt, args.output_bundle,
            finalization_receipt_path=args.finalization_receipt)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
