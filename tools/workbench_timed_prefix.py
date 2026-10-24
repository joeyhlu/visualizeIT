"""Timestamped onboarding-prefix ingestion for an existing historical video clip.

Creates a NEW, separately pinned experimental ClipSpec (``clock_kind =
timestamp_seconds``) from a registered ``video_file`` clip:

* up to 24 onboarding keyframes spread uniformly in *decoder* time over at
  most 10 seconds, ending exactly at the existing reviewed selection (seed)
  frame and strictly before the first scored frame;
* the original scored frames (IDs, source indices, window) unchanged;
* every frame's ``timestamp_s`` and ``media_time_s`` equal the observed
  OpenCV decoder PTS of its ORIGINAL source frame index (no cropping,
  re-encoding or re-indexing).

Only clock metadata is probed (``cv2.VideoCapture.grab`` + ``CAP_PROP_POS_MSEC``
via ``bench.workbench_cli.probe_video``); no pixel data is retrieved, analysed or
passed to preparation. Unreliable or non-monotonic PTS fails; frame rates are
never used to guess time. Evaluator inputs, depth, masks and poses are never
read; inherited metric-CAD / controlled-initial-pose references are removed.

Default output is an artifact directory only; ``--register`` additionally
creates the new clip in the workspace (never overwriting).

    python tools/workbench_timed_prefix.py --source-clip dev-mug \\
        --clip-id exp-mug-prefix10sec-v1 --out artifacts/video-workbench/timed-prefix/exp-mug-prefix10sec-v1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import workbench_cli as cli  # noqa: E402
from bench import workbench_contract as wc  # noqa: E402

SCHEMA = "workbench_timed_prefix_v1"
CLOCK_SCHEMA = "workbench_decoder_clock_v1"
CLOCK_SOURCE = "opencv_ffmpeg_decoder_pts_probe_v1"
MAX_SECONDS = wc.MAX_PREFIX_SECONDS
MAX_KEYFRAMES = wc.MAX_KEYFRAMES
MAX_SIDECAR_BYTES = 1 << 20
INHERITED_REMOVED = ("metric_cad", "initialization_candidate")


class TimedPrefixError(RuntimeError):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return cli.sha256_file(path)


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    try:
        from bench.vision import cv2
        out["cv2_runtime"] = cv2.__version__
    except Exception:  # noqa: BLE001
        out["cv2_runtime"] = None
    return out


def _source_closure() -> dict:
    files = ("tools/workbench_timed_prefix.py", "bench/workbench_cli.py", "bench/workbench_contract.py")
    return {rel: _sha_file(ROOT / rel) for rel in files}


def select_keyframes(times: Sequence[float], seed_index: int, first_scored_index: int, *,
                     seconds: float, keyframes: int) -> list[int]:
    """Uniform true-time selection of <= ``keyframes`` unique source indices in
    ``[t_seed - seconds, t_seed]``, always including the seed, all < first scored."""
    if not 0 < seconds <= MAX_SECONDS:
        raise TimedPrefixError(f"--seconds must be within (0, {MAX_SECONDS}]")
    if not 1 <= keyframes <= MAX_KEYFRAMES:
        raise TimedPrefixError(f"--keyframes must be within 1..{MAX_KEYFRAMES}")
    if not 0 <= seed_index < first_scored_index:
        raise TimedPrefixError("seed frame must precede the first scored frame")
    t_seed = times[seed_index]
    candidates = [i for i in range(0, seed_index + 1) if t_seed - times[i] <= seconds + 1e-9]
    span = t_seed - times[candidates[0]]
    if keyframes == 1 or span <= 0:
        return [seed_index]
    chosen: list[int] = []
    for k in range(keyframes):
        target = t_seed - span * (keyframes - 1 - k) / (keyframes - 1)
        best = min(candidates, key=lambda i: (abs(times[i] - target), i))
        if best not in chosen:
            chosen.append(best)
    if chosen[-1] != seed_index:
        chosen = [i for i in chosen if i != seed_index] + [seed_index]
    chosen = sorted(set(chosen))
    if chosen[-1] != seed_index or times[seed_index] - times[chosen[0]] > seconds + 1e-9:
        raise TimedPrefixError("internal selection error")
    if any(i >= first_scored_index for i in chosen):
        raise TimedPrefixError("selection reached a scored frame")
    return chosen


def build(ws: cli.Workspace, source_id: str, new_id: str, *, seconds: float, keyframes: int) -> dict:
    wc._ident(new_id, "clip_id")
    if not 0 < seconds <= MAX_SECONDS or not 1 <= keyframes <= MAX_KEYFRAMES:
        raise TimedPrefixError(f"--seconds must be within (0, {MAX_SECONDS}] and --keyframes within 1..{MAX_KEYFRAMES}")
    if new_id == source_id:
        raise TimedPrefixError("the new clip id must differ from the source clip")
    src = ws.load_clip(source_id)
    if src.source_kind != "video_file":
        raise TimedPrefixError("source clip must be a video_file clip")
    if src.onboarding is None or src.onboarding.selection is None or not src.onboarding.selection.reviewed:
        raise TimedPrefixError("source clip needs a reviewed onboarding selection")
    seed_id = src.onboarding.selection.frame_id
    if src.onboarding.keyframe_ids[-1] != seed_id:
        raise TimedPrefixError("reviewed selection must be on the last onboarding frame")
    by_id = {f.frame_id: f for f in src.frames}
    seed_index = by_id[seed_id].source_frame_index
    scored = [by_id[f] for f in src.scoring_frame_ids]
    first_scored_index = min(f.source_frame_index for f in scored)

    reader = cli.FrameReader(ws, src)
    media = reader.verify_media()          # whole-file hash/size against the registered clip
    media_path = ws.media_path(src)
    probe = cli.probe_video(media_path)    # decoder PTS only; frames are grabbed, never retrieved
    times = probe["media_times"]
    if not probe["pts_reliable"] or any(t is None or not math.isfinite(t) for t in times):
        raise TimedPrefixError("decoder PTS are missing or not strictly increasing; refusing to guess time")
    if (probe["width"], probe["height"]) != (src.width, src.height):
        raise TimedPrefixError("probed dimensions differ from the registered clip")
    max_index = max(f.source_frame_index for f in src.frames)
    if max_index >= probe["frame_count"]:
        raise TimedPrefixError("registered frames exceed the decoded frame count")

    chosen = select_keyframes(times, seed_index, first_scored_index, seconds=seconds, keyframes=keyframes)
    clock = {"schema": CLOCK_SCHEMA, "clock_source": CLOCK_SOURCE, "video_sha256": media["sha256"],
             "video_bytes": media["bytes"], "frame_count": probe["frame_count"],
             "pts_strictly_increasing": True, "media_times_s": times,
             "first_s": times[0], "last_s": times[-1],
             "min_delta_s": min(b - a for a, b in zip(times, times[1:])) if len(times) > 1 else None,
             "max_delta_s": max(b - a for a, b in zip(times, times[1:])) if len(times) > 1 else None,
             "method": "cv2.VideoCapture.grab + CAP_PROP_POS_MSEC; pixel data never retrieved"}
    clock_bytes = wc.canonical_json(clock)
    if len(clock_bytes) > MAX_SIDECAR_BYTES:
        raise TimedPrefixError("clock sidecar exceeds 1 MiB")

    def frame_id_for(index: int) -> str:
        existing = next((f.frame_id for f in src.frames if f.source_frame_index == index), None)
        return existing or f"s{index:06d}"
    prefix = [wc.FrameSpec(frame_id_for(i), i, times[i], times[i]) for i in chosen]
    scored_new = [wc.FrameSpec(f.frame_id, f.source_frame_index, times[f.source_frame_index],
                               times[f.source_frame_index]) for f in scored]
    if len({f.frame_id for f in prefix} | {f.frame_id for f in scored_new}) != len(prefix) + len(scored_new):
        raise TimedPrefixError("prefix frame ids collide with scored ids")
    provenance = {k: v for k, v in src.provenance.items() if k not in INHERITED_REMOVED}
    provenance.update({
        "experiment": "timed_prefix_automatic_preparation_v1",
        "labels": "pure-RGB automatic preparation experiment; development data; not new/unseen; no accuracy or "
                  "mobile claim",
        "source_clip_id": src.clip_id, "source_clip_sha256": wc.canonical_sha256(src.to_dict()),
        "source_window": [scored[0].source_frame_index, scored[-1].source_frame_index],
        "frozen_prefix_source_indices": chosen, "clock_sidecar_sha256": _sha_bytes(clock_bytes),
        "removed_inherited_fields": list(INHERITED_REMOVED), "accuracy_claim": False, "mobile_claim": False,
        "independent_annotations": False,
    })
    body = src.to_dict()
    body.update({
        "clip_id": new_id, "title": f"Timed-prefix experiment from {src.clip_id}",
        "clock_kind": "timestamp_seconds", "clock_source": CLOCK_SOURCE,
        "frames": [f.to_dict() for f in prefix + scored_new],
        "onboarding": {"keyframe_ids": [f.frame_id for f in prefix], "selection": src.onboarding.selection.to_dict()},
        "scoring_frame_ids": list(src.scoring_frame_ids),
        "evidence_role": "timed_prefix_automatic_experiment",
        "evaluator_inputs_read": False,
        "excluded_source_fields": sorted(set(src.excluded_source_fields) |
                                         {"metric_cad (removed from automatic preparation)",
                                          "controlled_initial_pose (removed from automatic preparation)"}),
        "provenance": provenance,
    })
    clip = wc.ClipSpec.from_dict(body)
    attestation = wc.check_onboarding(clip, clip.onboarding, clip.scoring_frame_ids)
    if not attestation["prefix_bound_attested"] or attestation["prefix_duration_s"] > seconds + 1e-9:
        raise TimedPrefixError("prefix is not attested within the requested seconds")
    if [f.frame_id for f in clip.frames if f.frame_id in set(clip.scoring_frame_ids)] != list(src.scoring_frame_ids):
        raise TimedPrefixError("scored frames changed")
    receipt = {
        "schema": SCHEMA, "new_clip_id": new_id, "source_clip_id": src.clip_id,
        "source_clip_sha256": provenance["source_clip_sha256"], "video": media,
        "seed": {"frame_id": seed_id, "source_frame_index": seed_index, "decoder_pts_s": times[seed_index]},
        "first_scored": {"frame_id": scored[0].frame_id, "source_frame_index": first_scored_index},
        "prefix": {"source_indices": chosen, "frame_ids": [f.frame_id for f in prefix],
                   "decoder_pts_s": [times[i] for i in chosen], "duration_s": attestation["prefix_duration_s"],
                   "count": len(chosen), "requested_seconds": seconds, "requested_keyframes": keyframes},
        "scoring": {"count": len(scored), "first_id": scored[0].frame_id, "last_id": scored[-1].frame_id,
                    "source_indices_unchanged": True, "frame_ids_unchanged": True},
        "decoded_pts_range_s": [times[0], times[-1]], "clock_sidecar_sha256": _sha_bytes(clock_bytes),
        "prefix_attestation": attestation,
        "separation": "Only decoder clock metadata (grab + POS_MSEC) was probed; no pixel data was retrieved, "
                      "analysed or passed to preparation. Later preparation receives only the frozen onboarding "
                      "prefix RGB/masks. No evaluator, depth, mask or pose input was read; no model, segmentation, "
                      "GPU or reconstruction was launched.",
        "evaluator_inputs_read": False, "accuracy_claim": False, "mobile_claim": False,
        "source_closure": _source_closure(), "versions": _versions(),
    }
    return {"clip": clip, "clock_bytes": clock_bytes, "receipt": receipt}


def _atomic_write(directory: Path, name: str, data: bytes) -> str:
    tmp = directory / f".{name}.tmp"
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, directory / name)
    return _sha_bytes(data)


def write_outputs(out: Path, built: dict) -> dict:
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise TimedPrefixError(f"output directory {out} already exists; outputs are never overwritten")
    parent = out.parent.resolve()
    if any(p.is_symlink() for p in [out.parent, *out.parent.parents] if p.exists()):
        raise TimedPrefixError("output path traverses a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < wc.FREE_RESERVE_BYTES + 2 * MAX_SIDECAR_BYTES:
        raise TimedPrefixError("1 GiB free-disk reserve would be violated")
    final = parent / out.name
    final.mkdir(exist_ok=False)
    hashes = {"clock.json": _atomic_write(final, "clock.json", built["clock_bytes"]),
              "clip.json": _atomic_write(final, "clip.json", wc.canonical_json(built["clip"].to_dict()))}
    receipt = dict(built["receipt"], output_hashes=dict(hashes))
    hashes["receipt.json"] = _atomic_write(final, "receipt.json", wc.canonical_json(receipt))
    return {"directory": str(final), "hashes": hashes}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create a timestamped onboarding-prefix experiment clip "
                                            "(decoder PTS metadata only; no pixels, models or GPU).")
    p.add_argument("--workspace", default=str(cli.DEFAULT_WORKSPACE))
    p.add_argument("--source-clip", required=True)
    p.add_argument("--clip-id", required=True, help="new clip id (must not exist)")
    p.add_argument("--seconds", type=float, default=MAX_SECONDS, help="prefix span in decoder seconds (<= 10)")
    p.add_argument("--keyframes", type=int, default=MAX_KEYFRAMES, help="onboarding keyframes (<= 24)")
    p.add_argument("--out", required=True, help="new artifact directory (must not exist)")
    p.add_argument("--register", action="store_true", help="also create the new clip in the workspace")
    args = p.parse_args(argv)
    try:
        ws = cli.Workspace(args.workspace)
        if args.register and ws.clip_path(args.clip_id).exists():
            raise TimedPrefixError(f"clip {args.clip_id} already exists; never overwritten")
        built = build(ws, args.source_clip, args.clip_id, seconds=args.seconds, keyframes=args.keyframes)
        written = write_outputs(Path(args.out), built)
        registered = None
        if args.register:
            if shutil.disk_usage(ws.path).free < wc.FREE_RESERVE_BYTES:
                raise TimedPrefixError("1 GiB free-disk reserve would be violated")
            registered = str(ws.save_clip(built["clip"]))
        r = built["receipt"]
        print(json.dumps({"status": "completed", **written, "registered_clip": registered,
                          "prefix_source_indices": r["prefix"]["source_indices"],
                          "prefix_duration_s": r["prefix"]["duration_s"],
                          "decoded_pts_range_s": r["decoded_pts_range_s"], "scoring": r["scoring"]}, indent=2))
        return 0
    except (TimedPrefixError, cli.WorkbenchCLIError, wc.ContractError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
