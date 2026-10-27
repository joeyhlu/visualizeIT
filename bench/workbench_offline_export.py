"""Bounded CPU composition of a selected PNG onto authenticated saved metric poses."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
from types import MappingProxyType
import uuid

import numpy as np

from . import workbench_artwork_replay as ar
from . import workbench_contract as wc
from . import workbench_geometry as wg

ROOT = Path(__file__).resolve().parents[1]
VERSION = "workbench_offline_export_v1"
SOURCES = (*ar.REPLAY_SOURCES, "bench/workbench_offline_export.py", "tools/workbench_offline_export.py",
           "tools/design_context.py", "bench/vision.py", "bench/workbench_tracking_policy.py")
MAX_FRAMES = 1000
DEFAULT_PLACEMENT = dict(scale=1.0, rotation_deg=0.0, offset_t_m=0.0, offset_b_m=0.0, opacity=1.0)


def refuse(reason):
    raise ar.Refused("refused", reason)


def _saved_display_decision(row, *, current_mask_trusted, epoch=0):
    """Adapt authenticated historical evidence without retrospectively re-running tracking.

    A saved tracking row is measured evidence. Legacy loss never authorizes a prediction,
    and the historical schema cannot carry proposed prediction candidates.
    """
    from .workbench_tracking_policy import DisplayDecision
    measured = row.pose_state == "tracking" and row.pose is not None and row.pose_units == "metric"
    state = "measured" if measured else "recovering" if row.pose_state == "recovering" else "lost"
    return DisplayDecision(state=state, pose=tuple(np.asarray(row.pose, dtype=float).ravel()) if measured else None,
        opacity=1.0 if measured and current_mask_trusted else 0.0,
        reason="saved_accepted_measurement" if measured and current_mask_trusted else
               "current_observed_mask_unavailable" if measured else "saved_" + row.pose_state,
        age_frames=0 if measured else None, age_ms=None, epoch=epoch, frame_id=row.frame_id,
        accepted_measured=measured, current_mask_trusted=current_mask_trusted)


def _anchor_integrity(artwork, marker=None):
    """Validate the actual persistent anchor consumed by composition, not a summary label."""
    marker = artwork.marker if marker is None else marker
    vertices, faces = np.asarray(artwork.vertices), np.asarray(artwork.faces)
    fid = marker.get("face_id")
    bary = np.asarray(marker.get("barycentric"), dtype=np.float64)
    basis = np.asarray([marker.get(k) for k in ("tangent", "bitangent", "normal")], dtype=np.float64)
    size = marker.get("size_object_units")
    if (type(fid) is not int or not 0 <= fid < len(faces) or bary.shape != (3,)
            or not np.isfinite(bary).all() or (bary < 0).any() or not np.isclose(bary.sum(), 1., atol=1e-8)
            or basis.shape != (3, 3) or not np.isfinite(basis).all()
            or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-7)
            or not np.allclose(np.cross(basis[0], basis[1]), basis[2], atol=1e-7)
            or type(size) not in (float, int) or not math.isfinite(size) or size <= 0
            or vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()):
        refuse("anchor/chart integrity invalid")
    return wc.canonical_sha256(ar._plain(marker))


def _mesh_anchor_request(path, clip, bundle, selection):
    """Parse and hash one bounded request snapshot, with no alternative pose authority."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        refuse("mesh anchor request must be a regular non-symlink file")
    raw = ar._read_bounded(path, 64 << 10)
    request = wc.loads_strict(raw)
    keys = {"schema", "clip_id", "frame_id", "pixel", "reviewed", "reviewed_by"}
    if not isinstance(request, dict) or set(request) != keys:
        refuse("mesh anchor request must contain exactly the v1 fields")
    if (request["schema"] != "workbench_mesh_anchor_v1"
            or type(request["clip_id"]) is not str or request["clip_id"] != clip.clip_id
            or type(request["frame_id"]) is not str
            or request["frame_id"] != bundle.seed_frame_id or request["frame_id"] != selection.frame_id
            or request["reviewed"] is not True or type(request["reviewed_by"]) is not str
            or not request["reviewed_by"].strip()):
        refuse("mesh anchor requires reviewed matching clip and original authorized seed frame")
    pixel = request["pixel"]
    if (type(pixel) is not list or len(pixel) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in pixel)
            or not (-0.5 <= pixel[0] < clip.width - .5 and -.5 <= pixel[1] < clip.height - .5)):
        refuse("mesh anchor requires finite source pixel coordinates within frame bounds")
    return request, hashlib.sha256(raw).hexdigest()


def _mesh_anchor_seed_identity(workspace, clip, seed_frame_id):
    from .workbench_cli import FrameReader
    reader = FrameReader(workspace, clip)
    try:
        media, frames = reader.verify_media(), reader.verify_frames([seed_frame_id])
        _, decoded = reader.read(clip.frame(seed_frame_id))
        return dict(media=media, source_frames=frames, decoded=decoded)
    finally:
        reader.close()


def _experimental_admission(clip, run, binding, artwork, bundle, K):
    """Refuse unsupported late windows; no checkpoint/backend restoration is implied."""
    from .workbench_tracking_policy import SurfaceAdmission
    frames = [frame.frame_id for frame in clip.frames]
    seed = frames.index(binding["seed_frame_id"])
    scheduled = list(run.scheduled_frame_ids)
    end = frames.index(scheduled[-1])
    if scheduled != frames[seed + 1:end + 1]:
        refuse("experimental tracking requires the complete causal prefix after the authorized seed; checkpoint restore unsupported")
    # This exporter supports a tangent plane, not arbitrary curved-surface charts.
    # Admit a real planar chart only after checking the geometry itself.
    marker = artwork.marker
    n = np.asarray(marker["normal"], float)
    center = np.asarray(marker["barycentric"], float) @ artwork.vertices[artwork.faces[marker["face_id"]]]
    distances = (artwork.vertices - center) @ n
    patch = artwork.patch
    if (np.max(np.abs(distances)) > 1e-7 or len(patch.get("folded_faces", []))
            or len(patch["faces"]) == 0 or patch.get("overlap_counts") is None
            or np.max(patch["overlap_counts"]) > 1):
        refuse("experimental export requires a verified planar, unfolded, overlap-free chart")
    chart = dict(chart_id="saved-planar-chart", domain="plane", orientation_valid=True,
        overlap_free=True, distortion=1.0, max_distortion=1.0,
        marker=ar._plain(marker), patch=ar._plain(patch), tolerance_m=1e-7)
    anchor = dict(anchor_id="saved-persistent-anchor", geometry_sha256=bundle.glb_sha256,
        chart_id=chart["chart_id"], marker=ar._plain(marker))
    calibration_sha = wc.canonical_sha256(dict(K=K.ravel().tolist(), width=clip.width, height=clip.height,
        principal_point_origin=bundle.principal_point_origin))
    admission = SurfaceAdmission(units="metric", metric_provenance=bundle.metric_provenance,
        calibration_sha256=calibration_sha, geometry_sha256=bundle.glb_sha256,
        anchor_sha256=wc.canonical_sha256(anchor), chart_sha256=wc.canonical_sha256(chart))
    admission.verify(calibration_sha256=calibration_sha, geometry_sha256=bundle.glb_sha256,
        anchor_record=anchor, chart_record=chart)
    return dict(mode="experimental_saved_measurements_v1", causal_prefix_frame_ids=scheduled,
        prefix_verified=True, checkpoint_restored=False, epoch=0,
        surface=dict(units="metres", metric_provenance=bundle.metric_provenance,
            calibration_sha256=calibration_sha, geometry_sha256=bundle.glb_sha256,
            anchor_sha256=wc.canonical_sha256(anchor), chart_sha256=wc.canonical_sha256(chart)),
        predictions_enabled=False, prediction_limitation="Historical loss/unavailability cannot authorize predictions.")


def _publish_new_directory(staging, output):
    """Atomic no-replace directory publication, with no check/rename fallback."""
    if sys.platform == "win32":
        # Windows MoveFile semantics used by os.rename refuse an existing target.
        os.rename(staging, output)
        return
    if sys.platform != "linux":
        refuse("exclusive directory publication is unsupported on this platform")
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        refuse("exclusive directory publication requires Linux libc renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. The caller supplies same-volume paths.
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            refuse("exclusive directory publication is unsupported by this Linux kernel/filesystem")
        raise OSError(error, os.strerror(error), os.fspath(output))


def placement_values(value=None):
    value = DEFAULT_PLACEMENT.copy() if value is None else value
    if not isinstance(value, dict) or set(value) != set(DEFAULT_PLACEMENT):
        refuse("placement must contain exactly scale, rotation_deg, offset_t_m, offset_b_m, opacity")
    bounds = dict(scale=(0.05, 4), rotation_deg=(-180, 180), offset_t_m=(-0.5, 0.5),
                  offset_b_m=(-0.5, 0.5), opacity=(0, 1))
    for key, (low, high) in bounds.items():
        x = value[key]
        if type(x) not in (int, float) or not math.isfinite(x) or not low <= x <= high:
            refuse(f"placement {key} outside [{low}, {high}]")
    return {key: float(value[key]) for key in DEFAULT_PLACEMENT}


def compose_planar(rgb, rgba, mask, pose, K, vertices, faces, marker, placement=None):
    """Straight alpha, nearest sampling, float64 geometry; no curved-surface visibility claim.

    Returns (RGB, reason, projected corners). Integrity errors raise; geometrically invisible
    planes suppress. PNG top maps to +bitangent, so image y increases toward -bitangent.
    """
    from .vision import cv2
    p = placement_values(placement)
    pose, K = wg.check_pose(pose), wg.check_K(K)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        refuse("source must be uint8 RGB")
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4 or min(rgba.shape[:2]) < 2:
        refuse("artwork must be uint8 RGBA with edges >=2")
    if np.asarray(mask).shape != rgb.shape[:2]:
        refuse("observed mask dimensions differ")
    center = np.asarray(marker["barycentric"], np.float64) @ np.asarray(vertices, np.float64)[faces[int(marker["face_id"])]]
    t, b, n = (np.asarray(marker[k], np.float64) for k in ("tangent", "bitangent", "normal"))
    basis = np.stack((t, b, n))
    size = float(marker["size_object_units"])
    if (not np.isfinite(center).all() or not np.isfinite(basis).all() or not math.isfinite(size)
            or size <= 0 or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-7)
            or not np.allclose(np.cross(t, b), n, atol=1e-7)):
        return rgb.copy(), "invalid_plane", None
    center = center + p["offset_t_m"] * t + p["offset_b_m"] * b
    angle = math.radians(p["rotation_deg"])
    u = math.cos(angle) * t + math.sin(angle) * b
    v = -math.sin(angle) * t + math.cos(angle) * b
    h, w = rgba.shape[:2]
    sx, sy = size * p["scale"] * w / max(h, w), size * p["scale"] * h / max(h, w)
    corners = center + np.asarray([[-1, 1], [1, 1], [1, -1], [-1, -1]])[:, :1] * u * sx / 2 \
        + np.asarray([[1], [1], [-1], [-1]]) * v * sy / 2
    cam = wg.to_camera(corners, pose)
    if (cam[:, 2] <= 1e-8).any():
        return rgb.copy(), "behind_camera", None
    camera_center = wg.to_camera(center[None], pose)[0]
    if np.dot(pose[:3, :3] @ n, camera_center) >= -1e-12:
        return rgb.copy(), "backfacing", None
    projected = cam @ K.T
    projected = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(projected).all() or abs(cv2.contourArea(projected.astype(np.float32))) < 1e-6:
        return rgb.copy(), "degenerate_projection", None
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]), projected.astype(np.float32))
    warped = cv2.warpPerspective(rgba, transform, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3].astype(np.float64) * (p["opacity"] / 255.0)
    alpha *= np.asarray(mask, bool)
    if not (alpha > 0).any():
        return rgb.copy(), "empty_or_outside_observed_mask", projected.tolist()
    result = np.floor(warped[:, :, :3].astype(np.float64) * alpha[:, :, None]
                      + rgb.astype(np.float64) * (1-alpha[:, :, None]) + 0.5).astype(np.uint8)
    return result, "rendered_planar_approximation", projected.tolist()


def _project(path, revision):
    from tools import design_context as dc
    raw = ar._read_bounded(Path(path), 25_165_824)
    doc = dc.decode_document(raw)
    if doc["projectRevisionId"] != revision or doc["selectedArtwork"] is None:
        refuse("project revision differs or selected PNG is unavailable")
    png = base64.b64decode(doc["selectedArtwork"]["png"]["base64"], validate=True)
    from .vision import cv2
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] not in (3, 4) or min(decoded.shape[:2]) < 2:
        refuse("selected PNG must have RGB/RGBA channels and edges >=2")
    rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA if decoded.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
    return doc, rgba, hashlib.sha256(raw).hexdigest(), hashlib.sha256(png).hexdigest()


def prepare_mesh_artwork(rgba, marker, placement=None):
    """Center-letterbox to a square without resizing; odd padding goes bottom/right.

    The surface sampler uses floor/clamp texel cells, with PNG top at +bitangent.
    Return independent arrays/records; the persistent anchor is never edited.
    """
    p = placement_values(placement)
    if p["offset_t_m"] != 0 or p["offset_b_m"] != 0:
        refuse("mesh offsets require a separately admitted persistent anchor")
    if (not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.ndim != 3
            or rgba.shape[2] != 4 or min(rgba.shape[:2]) < 2 or max(rgba.shape[:2]) > 1024):
        refuse("mesh artwork must be uint8 RGBA with edges 2..1024")
    h, w = rgba.shape[:2]
    edge = max(8, h, w)
    padded = np.zeros((edge, edge, 4), np.uint8)
    top, left = (edge-h)//2, (edge-w)//2
    padded[top:top+h, left:left+w] = rgba
    derived = ar._plain(marker)
    t, b = (np.asarray(marker[k], np.float64) for k in ("tangent", "bitangent"))
    angle = math.radians(p["rotation_deg"])
    derived["tangent"] = (math.cos(angle)*t + math.sin(angle)*b).tolist()
    derived["bitangent"] = (-math.sin(angle)*t + math.cos(angle)*b).tolist()
    # Minimum sampler canvas edge must not shrink a tiny selected PNG physically.
    derived["size_object_units"] = float(marker["size_object_units"]) * p["scale"] * edge/max(h, w)
    derived["artwork_sha256"] = wg.array_sha256(padded)
    return padded, derived


def _immutable_array(value):
    """Own array bytes with a read-only backing object, not a mutable view."""
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _immutable_value(value):
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    return value


def _mesh_geometry_placement(placement):
    # Opacity is a current display decision; it does not affect patch geometry.
    p = placement_values(placement)
    return tuple(p[key] for key in ("scale", "rotation_deg", "offset_t_m", "offset_b_m"))


@dataclass(frozen=True, slots=True)
class PreparedMeshArtwork:
    """Owned, export-local fixed artwork and patch; never owns a frame pose/mask."""
    _texture: np.ndarray
    _marker: MappingProxyType
    _patch: MappingProxyType
    _vertices: np.ndarray
    _faces: np.ndarray
    _support: np.ndarray
    _source: tuple
    _placement: tuple
    _chart_limit: float | None
    _owner: str | None

    def verify(self, rgba, vertices, faces, face_support, marker, placement,
               *, max_chart_stretch=None, owner=None):
        actual = (wg.array_sha256(rgba), wg.array_sha256(vertices),
                  wg.array_sha256(faces), wg.array_sha256(face_support),
                  wc.canonical_sha256(ar._plain(marker)))
        if (actual != self._source or _mesh_geometry_placement(placement) != self._placement
                or wg.check_chart_stretch_limit(max_chart_stretch) != self._chart_limit
                or (None if owner is None else str(owner)) != self._owner):
            refuse("prepared mesh artwork source, placement, policy or export owner changed")


def prepare_mesh_artwork_state(rgba, vertices, faces, face_support, marker, placement=None,
                               *, max_chart_stretch=None, owner=None):
    """Prepare only fixed mesh/artwork data, with private immutable ownership."""
    p = placement_values(placement)
    limit = wg.check_chart_stretch_limit(max_chart_stretch)
    texture, derived = prepare_mesh_artwork(rgba, marker, p)
    owned_vertices = _immutable_array(vertices)
    owned_faces = _immutable_array(faces)
    owned_support = _immutable_array(face_support)
    owned_marker = _immutable_value(ar._plain(derived))
    patch = wg.marker_patch(owned_vertices, owned_faces, owned_support, owned_marker,
                            max_chart_stretch=limit, conservative_chart_reach=True)
    return PreparedMeshArtwork(_immutable_array(texture), owned_marker, _immutable_value(patch),
        owned_vertices, owned_faces, owned_support,
        (wg.array_sha256(rgba), wg.array_sha256(vertices), wg.array_sha256(faces),
         wg.array_sha256(face_support), wc.canonical_sha256(ar._plain(marker))),
        _mesh_geometry_placement(p), limit, None if owner is None else str(owner))


def compose_mesh(rgb, rgba, mask, pose, K, vertices, faces, face_support, marker,
                 placement=None, *, max_chart_stretch=None, artwork_region_raster=False,
                 prepared_mesh_artwork=None, prepared_owner=None):
    """Actual registered mesh visibility and straight-alpha, half-up RGB composition."""
    if type(artwork_region_raster) is not bool:
        refuse("artwork_region_raster must be boolean")
    p = placement_values(placement)
    limit = wg.check_chart_stretch_limit(max_chart_stretch)
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        refuse("source must be uint8 RGB")
    if mask is None or np.asarray(mask).shape != rgb.shape[:2]:
        refuse("observed mask dimensions differ")
    if prepared_mesh_artwork is None:
        texture, derived = prepare_mesh_artwork(rgba, marker, p)
        render_vertices, render_faces, render_support, patch = vertices, faces, face_support, None
    else:
        if type(prepared_mesh_artwork) is not PreparedMeshArtwork:
            refuse("prepared mesh artwork must be an owned PreparedMeshArtwork")
        prepared_mesh_artwork.verify(rgba, vertices, faces, face_support, marker, p,
                                     max_chart_stretch=limit, owner=prepared_owner)
        texture, derived = prepared_mesh_artwork._texture, prepared_mesh_artwork._marker
        render_vertices, render_faces, render_support = (prepared_mesh_artwork._vertices,
                                                         prepared_mesh_artwork._faces,
                                                         prepared_mesh_artwork._support)
        patch = prepared_mesh_artwork._patch
    h, w = rgb.shape[:2]
    rendered = wg.render_marker_rgba(render_vertices, render_faces, render_support, derived, texture, K, pose,
                                    w, h, mask, max_chart_stretch=limit,
                                     artwork_region_raster=artwork_region_raster,
                                     _prepared_patch=patch)
    overlay, stats = rendered["rgba"], dict(rendered["stats"])
    alpha = overlay[:, :, 3].astype(np.float64) * (p["opacity"] / 255.0)
    alpha *= np.asarray(mask, bool)
    stats["composited_pixels"] = int(np.count_nonzero(alpha > 0))
    if not stats["composited_pixels"]:
        return rgb.copy(), "mesh_" + (stats.get("reason") or "empty_or_outside_observed_mask"), stats
    result = np.floor(overlay[:, :, :3].astype(np.float64)*alpha[:, :, None]
                      + rgb.astype(np.float64)*(1-alpha[:, :, None]) + .5).astype(np.uint8)
    return result, "rendered_mesh_artwork", stats


def export_saved_window(workspace, *, run_id, role, project_path, expected_revision, bundle_path,
                        bundle_sha256, admission_dir, output_dir, playback_fps, placement=None, repo_root=ROOT,
                        experimental_tracking=False, surface_mode="planar", max_chart_stretch=None,
                        mesh_anchor_path=None, artwork_region_raster=False, prepare_mesh_once=False):
    """Publish a new derived directory only after complete video decode and input reauthentication."""
    from . import workbench_cli as cli, workbench_frame_boundary as fb, workbench_neural as wn
    from .vision import cv2
    if role not in ar.NEURAL_ROLES:
        refuse("only a saved neural role is supported")
    if type(playback_fps) not in (int, float) or not math.isfinite(playback_fps) or not 0.1 <= playback_fps <= 120:
        refuse("explicit CFR playback FPS must be between 0.1 and 120")
    placement = placement_values(placement)
    if surface_mode not in ("planar", "mesh"):
        refuse("surface_mode must be planar or mesh")
    if type(artwork_region_raster) is not bool:
        refuse("artwork_region_raster must be boolean")
    if type(prepare_mesh_once) is not bool:
        refuse("prepare_mesh_once must be boolean")
    if artwork_region_raster and surface_mode != "mesh":
        refuse("artwork_region_raster requires mesh surface mode")
    if prepare_mesh_once and surface_mode != "mesh":
        refuse("prepare_mesh_once requires mesh surface mode")
    if surface_mode == "planar" and max_chart_stretch is not None:
        refuse("max_chart_stretch requires mesh surface mode")
    if mesh_anchor_path is not None and surface_mode != "mesh":
        refuse("mesh_anchor_path requires mesh surface mode")
    if surface_mode == "mesh":
        max_chart_stretch = wg.check_chart_stretch_limit(max_chart_stretch)
        if placement["offset_t_m"] != 0 or placement["offset_b_m"] != 0:
            refuse("mesh offsets require a separately admitted persistent anchor")
        if experimental_tracking:
            refuse("experimental tracking contract currently supports planar export only")
    if type(experimental_tracking) is not bool:
        refuse("experimental_tracking must be boolean")
    output = Path(output_dir)
    if output.is_symlink() or output.exists():
        refuse("output already exists")
    derived = workspace.derived_dir.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    if parent != derived or output.name in ("", ".", ".."):
        refuse("output must be a new direct child of workspace derived")
    output = parent / output.name
    project_path, bundle_path = Path(project_path).resolve(strict=True), Path(bundle_path).resolve(strict=True)
    service = ar.ArtworkRamReplay(workspace, bundle_resolver=lambda _: (bundle_path, bundle_sha256),
                                  admission_dir=Path(admission_dir), repo_root=repo_root)
    state = service._load_run(run_id)
    run, report = state["run"], state["report"]
    if role not in run.roles:
        refuse("role is absent from saved run")
    ids = list(run.scheduled_frame_ids)
    if not 1 <= len(ids) <= MAX_FRAMES:
        refuse("scheduled window exceeds 1000 frames or is empty")
    # Bind the parsed object and baseline digest to the SAME bounded byte snapshot.
    clip_path = workspace.clip_path(run.clip_id)
    clip_raw = ar._read_bounded(clip_path, ar.MAX_SMALL_JSON_BYTES)
    clip_sha = hashlib.sha256(clip_raw).hexdigest()
    clip = wc.ClipSpec.from_dict(wc.loads_strict(clip_raw))
    if clip.clip_id != run.clip_id:
        refuse("registered clip identity differs from saved run")
    if clip.source_kind == "video_file" and (clip.media.sha256 is None or clip.media.bytes is None):
        refuse("video source requires registered SHA256 hash and byte count")
    if clip.width > ar.MAX_EDGE or clip.height > ar.MAX_EDGE or clip.width * clip.height > ar.MAX_PIXELS:
        refuse("frame exceeds 1280 edge / 2 MP")
    # FFV1/OpenCV can truncate odd dimensions; refuse them before opening a writer.
    if min(clip.width, clip.height) < 2 or clip.width % 2 or clip.height % 2 or clip.intrinsics.K is None:
        refuse("FFV1 export requires even dimensions and calibrated K")
    rows = [state["table"][(role, fid)] for fid in ids]
    stage = (report.get("stages") or {}).get(role) or {}
    if stage.get("status") not in ("completed", "cache_hit"):
        refuse("saved role stage did not complete")
    bundle, bundle_files = service._bundle(clip.clip_id)
    binding = fb.neural_binding(clip, run, family=bundle.family, bundle=bundle)
    context = fb.FrameContextV1.build(clip, run, role, binding)
    provenance = cli.boundary_provenance(context)
    if any((stage.get("execution") or {}).get(k) != v for k, v in provenance.items()):
        refuse("saved boundary context differs")
    K = wn.native_K(np.asarray(clip.intrinsics.K, float).ravel(), bundle.principal_point_origin)
    contract = service._admission_contract(rows[0].backend_identity.get("admission_contract_sha256"))
    producer = service._producer_sources(report, stage, contract)
    for row in rows:
        frame = clip.frame(row.frame_id)
        for key in ("frame_id", "source_frame_index", "timestamp_s", "media_time_s"):
            if getattr(row, key) != getattr(frame, key) or type(getattr(row, key)) is not type(getattr(frame, key)):
                refuse(f"saved frame {key} differs")
        service._identity_checks(role, row.backend_identity, bundle, binding, K.ravel().tolist())
        if row.backend_identity.get("admission_contract_sha256") != contract["contract_sha256"]:
            refuse("row admission contract differs")
        if row.pose_state == "tracking" and (row.pose is None or row.pose_units != "metric" or row.image_sha256 is None):
            refuse("tracking row lacks saved metric pose or decoded RGB identity")
    mask_set = (run.config or {}).get("mask_set")
    store = cli.MaskStore(workspace, clip.clip_id, mask_set)
    recorded_masks = ((report.get("stages") or {}).get("masks") or {}).get("identity")
    if not store.available or store.identity() != recorded_masks:
        refuse("mask manifest differs")
    selection = run.onboarding.selection
    all_masks = list(dict.fromkeys(ids + [selection.frame_id]))
    mask_files = store.verify(all_masks)
    selection_mask, _ = store.load(selection.frame_id, (clip.height, clip.width))
    if selection_mask is None:
        refuse("saved anchor selection mask unavailable")
    artwork = wn.NeuralArtwork(bundle, selection, run.artwork_kind, clip_K=clip.intrinsics.K,
        width=clip.width, height=clip.height, selection_mask=selection_mask, max_faces=ar.MAX_FACES)
    saved = dict(state["artwork_record"])
    saved_sha = saved.pop("record_sha256", None)
    if artwork.sha256 != saved_sha or wc.canonical_json(ar._plain(artwork.record)) != wc.canonical_json(saved):
        refuse("saved anchor/original artwork record differs")
    anchor_sha = _anchor_integrity(artwork)
    mesh_marker = artwork.marker
    mesh_anchor = None
    if mesh_anchor_path is not None:
        request, request_sha = _mesh_anchor_request(mesh_anchor_path, clip, bundle, selection)
        seed_identity = _mesh_anchor_seed_identity(workspace, clip, bundle.seed_frame_id)
        hit = wg.anchor_from_pixel(artwork.vertices, artwork.faces, K, bundle.seed_pose_m,
            *request["pixel"], width=clip.width, height=clip.height, observed_mask=selection_mask)
        mesh_marker = wg.make_marker(artwork.vertices, artwork.faces, hit, bundle.seed_pose_m,
            artwork.marker["size_object_units"], artwork.marker["artwork_sha256"])
        mesh_anchor = dict(schema="workbench_mesh_anchor_revision_v1", source="manual_reviewed_seed_placement",
            historical_render_reproduction=False, request=request, request_sha256=request_sha,
            marker=mesh_marker, marker_sha256=_anchor_integrity(artwork, mesh_marker),
            anchor_object_xyz_m=(np.asarray(mesh_marker["barycentric"]) @
                artwork.vertices[artwork.faces[mesh_marker["face_id"]]]).tolist(),
            original_artwork_record_sha256=saved_sha, original_anchor_sha256=anchor_sha,
            seed=dict(**seed_identity, pose_authority="metric_bundle_authorized_seed_pose",
                frame_id=bundle.seed_frame_id, seed_label=bundle.seed_label,
                pose_m=ar._plain(bundle.seed_pose_m), pose_sha256=wc.canonical_sha256(ar._plain(bundle.seed_pose_m)),
                observed_mask_sha256=wg.array_sha256(selection_mask), mask_files=mask_files,
                bundle_files={key: str(value) if isinstance(value, Path) else value
                              for key, value in bundle_files.items()}, glb_sha256=bundle.glb_sha256,
                effective_K=K.ravel().tolist(), principal_point_origin=bundle.principal_point_origin))
        mesh_anchor_sha = wc.canonical_sha256(mesh_anchor)
    tracking = (_experimental_admission(clip, run, binding, artwork, bundle, K) if experimental_tracking else
        dict(mode="legacy_saved_context", prefix_verified=False, checkpoint_restored=False, epoch=0,
             predictions_enabled=False, limitation="Authenticated historical rows; causal prefix is not replayed."))
    from .workbench_tracking_policy import POLICY_VERSION
    tracking["decision_contract_version"] = POLICY_VERSION
    if experimental_tracking:
        for row in rows:
            epoch = row.evidence.get("epoch", 0)
            if type(epoch) is not int or epoch != tracking["epoch"]:
                refuse("saved tracking epoch differs; cross-epoch replay is unsupported")
        tracking["saved_evidence_identity"] = dict(source_sha256=clip_sha,
            mask_sha256=wc.canonical_sha256(recorded_masks),
            head_sha256=rows[0].backend_identity.get("installed_head_sha256") or
                rows[0].backend_identity.get("base_checkpoint_sha256"),
            config_sha256=wc.canonical_sha256(ar._plain(run.config)),
            policy_sha256=ar._sha_file(Path(repo_root) / "bench/workbench_tracking_policy.py"),
            saved_rows_sha256=state["files"]["results.jsonl"], epoch=0)
    project, rgba, project_sha, png_sha = _project(project_path, expected_revision)
    mesh_identity = None
    prepared_mesh = None
    prepared_owner = None
    if surface_mode == "mesh":
        if prepare_mesh_once:
            prepared_owner = f"{run_id}:{output}"
        def current_mesh_identity():
            if prepared_mesh is None:
                current_texture, current_marker = prepare_mesh_artwork(rgba, mesh_marker, placement)
            else:
                prepared_mesh.verify(rgba, artwork.vertices, artwork.faces, artwork.support,
                                     mesh_marker, placement, max_chart_stretch=max_chart_stretch,
                                     owner=prepared_owner)
                current_texture, current_marker = prepared_mesh._texture, prepared_mesh._marker
            h, w = rgba.shape[:2]
            edge = current_texture.shape[0]
            physical = float(current_marker["size_object_units"])
            identity = dict(surface_mode=surface_mode, max_chart_stretch=max_chart_stretch,
                derived_marker=ar._plain(current_marker),
                derived_marker_sha256=wc.canonical_sha256(ar._plain(current_marker)),
                selected_rgba_sha256=wg.array_sha256(rgba), texture_sha256=wg.array_sha256(current_texture),
                texture_shape=list(current_texture.shape), png_sha256=png_sha,
                selected_shape=list(rgba.shape),
                texel_padding=dict(top=(edge-h)//2, bottom=edge-h-(edge-h)//2,
                                   left=(edge-w)//2, right=edge-w-(edge-w)//2),
                selected_rectangle_size_m=dict(width=physical*w/edge, height=physical*h/edge),
                vertices_sha256=wg.array_sha256(artwork.vertices), faces_sha256=wg.array_sha256(artwork.faces),
                support_sha256=wg.array_sha256(artwork.support), placement=dict(placement),
                sampler="floor_clamp_nearest_texel_cells; PNG top=+bitangent",
                padding="transparent centered square, no resizing; odd extra bottom/right",
                support_source="metric CAD geometry (not observation-derived)")
            if artwork_region_raster:
                identity["artwork_region_raster"] = True
            if prepare_mesh_once:
                identity["prepare_mesh_once"] = True
            if mesh_anchor is not None:
                identity["mesh_anchor_revision_sha256"] = wc.canonical_sha256(mesh_anchor)
            return identity
        mesh_identity = current_mesh_identity()
        mesh_identity_sha = wc.canonical_sha256(mesh_identity)
    sources = service._source_hashes(SOURCES)
    # Conservative incompressible frame allowance + container/index/sidecar headroom.
    frame_allowance = clip.width * clip.height * 8 + 65536
    estimate = len(ids) * (frame_allowance + 32768) + (4 << 20)
    workspace.require_budget(extra_derived=estimate)
    staging = parent / (".offline-export-" + uuid.uuid4().hex)
    staging.mkdir()
    writer = reader = None
    try:
        video = staging / "playback.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), float(playback_fps), (clip.width, clip.height))
        if not writer.isOpened():
            refuse("FFV1 codec unavailable; no fallback")
        reader = cli.FrameReader(workspace, clip)
        media, frame_files = reader.verify_media(), reader.verify_frames(ids)
        records = []
        for index, row in enumerate(rows):
            rgb, image_ident = reader.read(clip.frame(row.frame_id))
            if row.image_sha256 is not None and image_ident["image_sha256"] != row.image_sha256:
                refuse("saved decoded source RGB hash differs")
            mask, mask_ident = store.load(row.frame_id, (clip.height, clip.width))
            if row.mask_state == "observed":
                if (row.mask_frame_id != row.frame_id or mask is None or mask_ident["mask_sha256"] != row.mask_sha256):
                    refuse("observed row mask identity differs")
            reason, corners, composite = "saved_pose_or_fresh_mask_unavailable", None, rgb
            geometry_stats = None
            trusted = row.mask_state == "observed" and mask is not None and bool(np.asarray(mask, bool).any())
            decision = _saved_display_decision(row, current_mask_trusted=trusted, epoch=tracking["epoch"])
            if decision.state == "measured" and decision.accepted_measured and decision.current_mask_trusted:
                if _anchor_integrity(artwork) != anchor_sha:
                    refuse("anchor/chart changed before composition")
                if surface_mode == "mesh":
                    # Construct the selected-artwork patch only when a current
                    # measured frame can actually render. All-lost exports keep
                    # the historical preparation and original-RGB path.
                    if prepare_mesh_once and prepared_mesh is None:
                        prepared_mesh = prepare_mesh_artwork_state(rgba, artwork.vertices,
                            artwork.faces, artwork.support, mesh_marker, placement,
                            max_chart_stretch=max_chart_stretch, owner=prepared_owner)
                    if wc.canonical_sha256(current_mesh_identity()) != mesh_identity_sha:
                        refuse("mesh composition identity changed")
                    composite, reason, geometry_stats = compose_mesh(rgb, rgba, mask,
                        np.asarray(decision.pose, np.float64).reshape(4, 4), K, artwork.vertices,
                        artwork.faces, artwork.support, mesh_marker,
                        {**placement, "opacity": placement["opacity"] * decision.opacity},
                        max_chart_stretch=max_chart_stretch, artwork_region_raster=artwork_region_raster,
                        prepared_mesh_artwork=prepared_mesh, prepared_owner=prepared_owner)
                else:
                    composite, reason, corners = compose_planar(rgb, rgba, mask, np.asarray(decision.pose, np.float64).reshape(4, 4), K,
                        artwork.vertices, artwork.faces, artwork.marker, {**placement, "opacity": placement["opacity"] * decision.opacity})
            digest = wg.array_sha256(composite)
            writer.write(np.ascontiguousarray(composite[:, :, ::-1]))
            if video.exists() and video.stat().st_size > (index+1) * frame_allowance + (1 << 20):
                refuse("writer exceeded conservative headroom")
            workspace.require_budget(extra_derived=(len(ids)-index-1)*frame_allowance + (4 << 20))
            records.append(dict(output_index=index, original=row.to_dict(), source_rgb_sha256=image_ident["image_sha256"],
                                display_decision=ar._plain(decision.to_dict()), expected_rgb_sha256=digest,
                                render_reason=reason, projected_corners_px=corners))
            if surface_mode == "mesh":
                records[-1]["geometry_render_stats"] = geometry_stats
        writer.release()
        writer = None
        reader.close()
        reader = None
        if video.stat().st_size > estimate:
            refuse("video exceeds reserved budget")
        decoded = cv2.VideoCapture(str(video))
        try:
            if not decoded.isOpened():
                refuse("cannot reopen encoded FFV1")
            decoded_fps = float(decoded.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(decoded_fps) or not math.isclose(decoded_fps, playback_fps, rel_tol=1e-6, abs_tol=1e-6):
                refuse("FFV1 decoded header FPS differs from explicit playback FPS")
            for record in records:
                ok, bgr = decoded.read()
                if not ok or bgr is None or bgr.shape != (clip.height, clip.width, 3):
                    refuse("FFV1 decoded frame count/dimensions differ")
                actual = wg.array_sha256(np.ascontiguousarray(bgr[:, :, ::-1]))
                if actual != record["expected_rgb_sha256"]:
                    refuse("FFV1 decoded RGB differs from composite/order")
                record["decoded_rgb_sha256"] = actual
            if decoded.read()[0]:
                refuse("FFV1 contains extra frames")
        finally:
            decoded.release()
        after = service._load_run(run_id)
        if _anchor_integrity(artwork) != anchor_sha:
            refuse("anchor/chart changed before publication")
        if surface_mode == "mesh" and wc.canonical_sha256(current_mesh_identity()) != mesh_identity_sha:
            refuse("mesh composition identity changed before publication")
        if experimental_tracking:
            current_tracking = _experimental_admission(clip, run, binding, artwork, bundle, K)
            if current_tracking["surface"] != tracking["surface"]:
                refuse("experimental surface identity changed before publication")
        fresh = cli.MaskStore(workspace, clip.clip_id, mask_set)
        if (after["files"] != state["files"] or service._source_hashes(SOURCES) != sources
                or ar._sha_file(workspace.clip_path(clip.clip_id), ar.MAX_SMALL_JSON_BYTES) != clip_sha
                or service._admission_contract(contract["contract_sha256"]) != contract
                or service._producer_sources(after["report"], (after["report"].get("stages") or {}).get(role) or {}, contract) != producer
                or service._bundle(clip.clip_id)[1] != bundle_files
                or fresh.identity() != recorded_masks or fresh.verify(all_masks) != mask_files
                or _project(project_path, expected_revision)[2:] != (project_sha, png_sha)):
            refuse("input/source closure changed before publication")
        reader = cli.FrameReader(workspace, clip)
        if reader.verify_media() != media or reader.verify_frames(ids) != frame_files:
            refuse("source media changed before publication")
        # Re-decode even for video inputs: file hashes alone are not decoded RGB identity.
        for row, record in zip(rows, records):
            if reader.read(clip.frame(row.frame_id))[1]["image_sha256"] != record["source_rgb_sha256"]:
                refuse("source decoded RGB changed before publication")
        reader.close()
        reader = None
        if mesh_anchor is not None:
            if (_mesh_anchor_request(mesh_anchor_path, clip, bundle, selection) != (request, request_sha)
                    or _mesh_anchor_seed_identity(workspace, clip, bundle.seed_frame_id) != seed_identity
                    or wg.array_sha256(fresh.load(selection.frame_id, (clip.height, clip.width))[0]) !=
                        mesh_anchor["seed"]["observed_mask_sha256"]
                    or _anchor_integrity(artwork, mesh_marker) != mesh_anchor["marker_sha256"]
                    or wc.canonical_sha256(mesh_anchor) != mesh_anchor_sha):
                refuse("manual mesh anchor input/seed/revision changed before publication")
        reasons = {}
        for record in records:
            reason = record["render_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        success_reason = "rendered_mesh_artwork" if surface_mode == "mesh" else "rendered_planar_approximation"
        rendered = reasons.get(success_reason, 0)
        recorded_identity = rows[0].backend_identity
        states = {key: 0 for key in ("accepted_measurements", "proposed_predictions", "recovering", "lost", "unavailable", "render_unavailable")}
        for record in records:
            original, decision = record["original"], record["display_decision"]
            if decision["accepted_measured"]:
                states["accepted_measurements"] += 1
            elif decision["state"] == "predicted":
                states["proposed_predictions"] += 1
            elif original["pose_state"] in ("recovering", "lost"):
                states[original["pose_state"]] += 1
            else:
                states["unavailable"] += 1
            if record["render_reason"] != success_reason:
                states["render_unavailable"] += 1
        manifest = dict(schema=VERSION, source="offline_saved_pose_planar_artwork", new_neural_inference=False,
            new_pose_estimation=False, weights_loaded=False, quality_claim=False, audio_copied=False,
            limitation="Tangent-plane approximation; no curved attachment, mesh occlusion, or surface certainty.",
            run_id=run_id, role=role, original_run_status=report["status"], scheduled_frame_ids=ids,
            source_clip=dict(clip_id=clip.clip_id, source_kind=clip.source_kind, evidence_role=clip.evidence_role,
                clock_kind=clip.clock_kind, clock_source=clip.clock_source, provenance=ar._plain(clip.provenance)),
            frame_accounting=dict(scheduled=len(ids), exported=len(records), rendered=rendered,
                suppressed=len(records)-rendered, counts_by_render_reason=reasons, quality_metrics=False, **states),
            tracking_policy=tracking,
            playback=dict(codec="FFV1", container="AVI", fps=float(playback_fps), timing="explicit CFR playback; original times only in rows",
                          verified_header_fps=decoded_fps, width=clip.width, height=clip.height, frames=len(records)),
            project=dict(project_id=project["projectId"], revision=expected_revision, file_sha256=project_sha, png_sha256=png_sha),
            placement=placement, plane=dict(marker=artwork.marker, units="metres", effective_K=K.ravel().tolist(),
                anchor_object_xyz_m=(np.asarray(artwork.marker["barycentric"], np.float64)
                    @ artwork.vertices[artwork.faces[int(artwork.marker["face_id"])]]).tolist()),
            provenance=dict(run_files=state["files"], clip_sha256=clip_sha, bundle_sha256=bundle_files["bundle_sha256"],
                glb_sha256=bundle.glb_sha256, original_artwork_record_sha256=saved_sha, masks=mask_files,
                mask_manifest=recorded_masks, media=media, source_frames=frame_files, admission=contract,
                producer_sources=producer, exporter_sources=sources, boundary=provenance,
                model_pins=dict(backend_role=role, installed_head_sha256=recorded_identity.get("installed_head_sha256"),
                    export_sha256=recorded_identity.get("export_sha256"), head=recorded_identity.get("head"),
                    base_checkpoint_sha256=recorded_identity["base_checkpoint_sha256"], weights_loaded=False)),
            dependencies=dict(numpy=np.__version__, opencv=cv2.__version__),
            video=dict(file="playback.avi", bytes=video.stat().st_size, sha256=ar._sha_file(video)), rows=records)
        if surface_mode == "mesh":
            manifest["source"] = "offline_saved_pose_mesh_artwork"
            manifest["limitation"] = ("Registered mesh depth/front/chart checks and current observed mask only; "
                "no independent visibility proof, real curved attachment accuracy, or physical-phone validation.")
            manifest["surface_mode"] = surface_mode
            manifest["mesh"] = mesh_identity
            manifest["provenance"]["mesh_composition_sha256"] = mesh_identity_sha
            manifest.pop("plane")
            if mesh_anchor is not None:
                manifest["mesh_anchor"] = mesh_anchor
        raw = wc.canonical_json(manifest)
        if len(raw) > 32 << 20:
            refuse("sidecar exceeds 32 MiB")
        workspace.require_budget(extra_derived=len(raw))
        (staging / "manifest.json").write_bytes(raw)
        workspace.require_budget()
        if output.exists():
            refuse("output appeared before publication")
        _publish_new_directory(staging, output)
        return manifest
    finally:
        if writer is not None:
            writer.release()
        if reader is not None:
            reader.close()
        if staging.exists():
            if staging.parent.resolve() == derived and staging.name.startswith(".offline-export-") and not staging.is_symlink():
                shutil.rmtree(staging)
