"""Deterministic six-case development suite export.

Historical cached cases (keyboard, mug, ranch) come from
``.cache/model-quality/inputs/<name>/input.json`` with their declared source
hashes, reviewed selection points and exact 240-frame windows. Three YCB-Video
pilot objects (IDs 2, 4, 18) come from
``.cache/prototype-sprint/dataset-pilot-v1/inference.json``; only RGB
path/hash/availability and ``cam_K`` are extracted. Onboarding evaluation
poses, depth and camera extrinsics are never copied, and ``evaluator.json`` is
never opened. These are development cases: not new/unseen objects, not
independent accuracy evidence. Old manifests are read only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from . import workbench_contract as wc

SUITE_SCHEMA = "workbench_development_suite_v1"
HISTORICAL = ("keyboard", "mug", "ranch")
HISTORICAL_FRAMES = {"keyboard": (1149, 1388), "mug": (827, 1066), "ranch": (10, 249)}
YCB_OBJECT_IDS = (2, 4, 18)
YCB_EXCLUDED = ("evaluationPose", "depth", "cam_R_w2c", "cam_t_w2c", "depth_scale", "model (CAD) for relative path",
                "evaluator.json (never opened)")
HISTORICAL_EXCLUDED = ("controlled_initial_pose (not used by relative reconstruction)",
                       "object.glb (not used by relative reconstruction)", "additional_window")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def historical_clip(repo: Path, name: str) -> wc.ClipSpec:
    base = repo / ".cache/model-quality/inputs" / name
    data = wc.loads_strict((base / "input.json").read_bytes())
    video = base / data["video"]
    declared = data["source_hashes"][data["video"]]
    actual = _sha(video)
    if actual != declared:
        raise wc.ContractError(f"{name}: source.mp4 hash differs from declared source_hashes")
    ids = list(data["frame_ids"])
    lo, hi = HISTORICAL_FRAMES[name]
    if ids != list(range(lo, hi + 1)):
        raise wc.ContractError(f"{name}: frame window differs from the preserved {lo}..{hi}")
    setup = int(data["setup_frame_id"])
    if setup != lo - 1:
        raise wc.ContractError(f"{name}: setup frame does not immediately precede the window")
    width, height = (int(v) for v in data["native_resolution"])
    K = [float(v) for row in data["intrinsics"] for v in row]
    if not (0 < K[2] < width and 0 < K[5] < height):
        raise wc.ContractError(f"{name}: native_resolution orientation is inconsistent with K")
    prefix = list(range(max(0, setup - wc.MAX_KEYFRAMES + 1), setup + 1))
    frames = [wc.FrameSpec(f"s{i:06d}", i) for i in prefix + ids]
    pts, labels = data["selection"]["points"], data["selection"]["labels"]
    if len(pts) != len(labels):
        raise wc.ContractError(f"{name}: selection points/labels differ in length")
    selection = {"frame_id": f"s{setup:06d}",
                 "positive": [[float(x), float(y)] for (x, y), l in zip(pts, labels) if l == 1],
                 "negative": [[float(x), float(y)] for (x, y), l in zip(pts, labels) if l == 0],
                 "reviewed": bool(data["selection"].get("reviewed")),
                 "reviewed_by": data["selection"].get("reviewed_by")}
    glb = base / data["asset"]
    clip = {
        "schema": wc.CLIP_SCHEMA, "clip_id": f"dev-{name}", "title": f"Historical development: {name}",
        "source_kind": "video_file", "clock_kind": "frame_index", "clock_source": None,
        "width": width, "height": height,
        "media": {"root_id": "model_quality_inputs", "path": f"{name}/{data['video']}", "sha256": actual,
                  "bytes": video.stat().st_size},
        "frames": [f.to_dict() for f in frames],
        "intrinsics": {"status": "dataset_reference", "K": K,
                       "source": "input.json intrinsics; principal-point origin undocumented"},
        "onboarding": {"keyframe_ids": [f"s{i:06d}" for i in prefix], "selection": selection},
        "scoring_frame_ids": [f"s{i:06d}" for i in ids],
        "evidence_role": "historical_development_case",
        "evaluator_inputs_read": False, "excluded_source_fields": list(HISTORICAL_EXCLUDED),
        "provenance": {
            "input_json_sha256": _sha(base / "input.json"), "setup_frame_id": setup,
            "window": [lo, hi], "selection_scope": data["selection"].get("scope"),
            "prefix_policy": "causal frames ending at the reviewed setup frame (<=24)",
            "metric_cad": {"path": f"{name}/{data['asset']}", "sha256": data["source_hashes"][data["asset"]],
                           "use": "neural metric profile only (not relative reconstruction)",
                           "present": glb.is_file()},
            "initialization_candidate": {"source": "onboarding:controlled_initial_pose",
                                         "note": "controlled onboarding pose; neural profile only"},
            "principal_point_origin": "undocumented",
            "independent_annotations": False,
        },
    }
    return wc.ClipSpec.from_dict(clip)


def ycb_clips(repo: Path, *, verify_images: bool = False) -> list[wc.ClipSpec]:
    root = repo / ".cache/prototype-sprint/dataset-pilot-v1"
    manifest_path = root / "inference.json"
    manifest = wc.loads_strict(manifest_path.read_bytes())
    manifest_sha = _sha(manifest_path)
    out = []
    for obj in manifest["objects"]:
        if obj["objectId"] not in YCB_OBJECT_IDS:
            continue
        rows = list(obj["onboarding"]) + list(obj["frames"])
        Ks = {tuple(float(v) for v in r["calibration"]["cam_K"]) for r in rows}
        if len(Ks) != 1:
            raise wc.ContractError(f"object {obj['objectId']}: cam_K varies across frames")
        K = list(Ks.pop())
        dirs = {str(PurePosixPath(r["rgb"]["path"]).parent) for r in rows}
        if len(dirs) != 1:
            raise wc.ContractError(f"object {obj['objectId']}: RGB frames span multiple directories")
        rgb_dir = dirs.pop()
        frames = []
        for i, r in enumerate(rows):
            if r["timestampKind"] != "frame_index_no_fps" or r["timestampSeconds"] is not None:
                raise wc.ContractError("YCB pilot frames must be frame-index clocked")
            if not r["rgb"]["available"]:
                raise wc.ContractError(f"frame {r['frameId']} RGB unavailable")
            path = root / r["rgb"]["path"]
            if not path.is_file():
                raise wc.ContractError(f"missing RGB {r['rgb']['path']}")
            if verify_images or i in (0, len(obj["onboarding"]) - 1, len(obj["onboarding"]), len(rows) - 1):
                if _sha(path) != r["rgb"]["sha256"]:
                    raise wc.ContractError(f"RGB {r['rgb']['path']} hash differs")
            frames.append(wc.FrameSpec(f"s{int(r['frameId']):06d}", int(r["frameId"]), None, None,
                                       PurePosixPath(r["rgb"]["path"]).name, r["rgb"]["sha256"]))
        from .vision import cv2
        first = cv2.imread(str(root / rows[0]["rgb"]["path"]), cv2.IMREAD_COLOR)
        if first is None:
            raise wc.ContractError("first YCB frame unreadable")
        n_on = len(obj["onboarding"])
        clip = {
            "schema": wc.CLIP_SCHEMA, "clip_id": f"dev-ycb-{obj['canonicalYcbId'].replace('_', '-')}",
            "title": f"YCB-Video pilot development: {obj['name']}", "source_kind": "image_sequence",
            "clock_kind": "frame_index", "clock_source": None, "width": int(first.shape[1]),
            "height": int(first.shape[0]), "media": {"root_id": "ycb_pilot", "path": rgb_dir, "sha256": None,
                                                     "bytes": None},
            "frames": [f.to_dict() for f in frames],
            "intrinsics": {"status": "dataset_reference", "K": K,
                           "source": "inference.json cam_K; principal-point origin undocumented"},
            "onboarding": None,
            "scoring_frame_ids": [f.frame_id for f in frames[n_on:]],
            "evidence_role": "development_dataset_reference",
            "evaluator_inputs_read": False, "excluded_source_fields": list(YCB_EXCLUDED),
            "provenance": {
                "dataset": manifest["dataset"], "dataset_revision": manifest["datasetRevision"],
                "inference_manifest_sha256": manifest_sha, "object_id": obj["objectId"],
                "canonical_id": obj["canonicalYcbId"], "scene_id": obj["sceneId"],
                "onboarding_frame_ids": [f.frame_id for f in frames[:n_on]],
                "suggested_onboarding": "all onboarding frames; user positive/negative selection required",
                "labels": "not new/unseen; development dataset reference only",
                "geometry_prepared_control": {
                    "status": "not_prepared",
                    "would_use": ["onboarding evaluationPose", "onboarding depth"],
                    "label": "geometry-prepared control: uses dataset onboarding pose; not RGB-only, not unseen",
                },
                "rgb_hashes_verified": "all" if verify_images else "onboarding_first_last_and_scoring_first_last",
                "principal_point_origin": "undocumented", "independent_annotations": False,
            },
        }
        out.append(wc.ClipSpec.from_dict(clip))
    if len(out) != len(YCB_OBJECT_IDS):
        raise wc.ContractError("expected YCB objects 2, 4 and 18 in the pilot manifest")
    return out


def build_suite(repo: Path, *, verify_images: bool = False) -> dict:
    clips, errors = [], []
    for name in HISTORICAL:
        try:
            clips.append(historical_clip(repo, name))
        except (OSError, KeyError, wc.ContractError) as exc:
            errors.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
    try:
        clips.extend(ycb_clips(repo, verify_images=verify_images))
    except (OSError, KeyError, wc.ContractError) as exc:
        errors.append({"case": "ycb_pilot", "error": f"{type(exc).__name__}: {exc}"})
    body = {"schema": SUITE_SCHEMA,
            "cases": [{"clip_id": c.clip_id, "clip_sha256": wc.canonical_sha256(c.to_dict()),
                       "evidence_role": c.evidence_role, "frames": len(c.frames),
                       "scoring_frames": len(c.scoring_frame_ids),
                       "selection_reviewed": bool(c.onboarding and c.onboarding.selection),
                       "intrinsics": c.intrinsics.status} for c in clips],
            "errors": errors,
            "claims": "development cases only; no independent annotations; accuracy gates unverified"}
    return {"suite": body, "clips": clips}


def export_suite(ws, *, verify_images: bool = False) -> dict:
    built = build_suite(ws.repo_root, verify_images=verify_images)
    for clip in built["clips"]:
        ws.save_clip(clip)
    out = ws.suite_dir / "development-suite-v1.json"
    data = wc.canonical_json(built["suite"])
    if out.exists() and out.read_bytes() != data:
        out = ws.suite_dir / f"development-suite-v1-{hashlib.sha256(data).hexdigest()[:12]}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return {**built["suite"], "path": str(out), "sha256": hashlib.sha256(data).hexdigest()}
