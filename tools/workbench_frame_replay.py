"""Opt-in recorded replay through the portable frame boundary (library entrypoints + safe inspect CLI).

``replay_recorded`` drives one role over the context's frozen order: the causal warm-up first,
then the scheduled scored frames. It uses the existing ``FrameReader`` (exact decode, pixel
identity, decoder PTS) and ``MaskStore`` (observed masks only), and calls the boundary for every
planned frame. Every planned row is accounted for: a read or mask failure becomes a counted row,
and an initialization or adapter failure blocks every remaining planned row with its reason.
Warm-up and scored rows and their measured wall-clock timings are kept separate. The decoder and
the session/tracker/renderer resources are always closed, and cleanup errors are reported.

Ownership: ``replay_recorded`` owns the adapter and reader it receives and closes both on every
path. The builders take ownership of the session or tracker they receive, close it if building
fails, and otherwise hand it to the returned adapter.

``run_neural_replay`` is the documented end-to-end neural sequence:
1. load the bundle;
2. bind the seed plan;
3. verify the admission;
4. build the artwork;
5. enter ``gpu_stage``;
6. construct the backend and set its seed target;
7. construct the ``NeuralSession``;
8. replay and clean up.

Every component is real by default. Test doubles are allowed only through ``NeuralFactories`` and
are always recorded in ``result['provenance']``; they are never promoted to real admission or model
evidence.

Nothing happens on import, ``--help`` or ``inspect``: no reads of models, data, GT or runs, no GPU
import, no writes and no inference. Results stay in memory, and overlays go only to an optional
callback. This path is NOT cached and does not migrate the legacy CLI pipeline. It makes no
phone, continuous-camera or accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import workbench_contract as wc  # noqa: E402
from bench import workbench_frame_boundary as fb  # noqa: E402

SCHEMA_REPLAY = "workbench_frame_replay_v1"
SOURCE_FILES = ("bench/workbench_frame_boundary.py", "tools/workbench_frame_replay.py", "bench/workbench_contract.py",
                "bench/workbench_neural.py", "bench/workbench_geometry.py", "bench/workbench_cli.py",
                "bench/workbench_reconstruction.py")
FLAGS = {"opt_in_boundary": True, "legacy_pipeline_migrated": False, "cached": False,
         "phone_or_continuous_camera_claim": False, "accuracy_claim": False,
         "timings": "wall-clock measured in this process (read, mask, boundary process, end-to-end); not phone performance",
         "live_neural_camera": "unsupported until a reviewed source/admission contract exists"}


def source_identity(root: Path = ROOT) -> dict:
    """SHA-256 of the boundary, replay and wrapped source files (for future reports)."""
    out = {}
    for rel in SOURCE_FILES:
        path = root / rel
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return out


def _count(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r[key])] = out.get(str(r[key]), 0) + 1
    return dict(sorted(out.items()))


def _new_result(context: fb.FrameContextV1) -> dict:
    return {"schema": SCHEMA_REPLAY, "status": None, "role": context.role, "run_id": context.run.run_id,
            "clip_id": context.clip.clip_id, "source_mode": context.source_mode, "context": context.to_dict(),
            "warmup": [], "scored": [], "timings": {"warmup": [], "scored": []}, "overlays_delivered": 0,
            "identity": None, "media": None, "failure": None, "cleanup_errors": [], "flags": dict(FLAGS),
            "sources": source_identity()}


def _finish(context: fb.FrameContextV1, result: dict, done: int, failure: str | None, status_on_failure: str) -> dict:
    planned = list(context.permitted_order)
    if failure is not None:
        identity = (result["identity"] or {}).get("backend_identity") or {"role": context.role, "identity": "unavailable"}
        for fid in planned[done:]:
            frame = context.frame(fid)
            row = wc.FrameResult(
                run_id=context.run.run_id, clip_id=context.clip.clip_id, frame_id=fid,
                source_frame_index=frame.source_frame_index, timestamp_s=frame.timestamp_s,
                media_time_s=frame.media_time_s, backend_role=context.role, backend_identity=identity,
                mask_state="blocked", pose_state="blocked", render_state="unavailable",
                failure_reason=failure[:2000], pose=None, pose_units=None, image_sha256=None, mask_frame_id=None,
                mask_sha256=None, metrics={}, evidence={"replay": "not processed"})
            result[context.phase(fid)].append(row.to_dict())
        result["failure"] = failure[:2000]
    result["status"] = "completed" if failure is None else status_on_failure
    if result["cleanup_errors"] and failure is None:
        result["status"] = "completed_with_cleanup_errors"
    if (len(result["scored"]) != len(context.scheduled_frame_ids)
            or len(result["warmup"]) != len(context.warmup_frame_ids)):
        raise RuntimeError("replay accounting invariant violated")      # pragma: no cover - defensive
    result["counts"] = {phase: {"rows": len(result[phase]), "pose_state": _count(result[phase], "pose_state"),
                                "mask_state": _count(result[phase], "mask_state"),
                                "render_state": _count(result[phase], "render_state")}
                        for phase in ("warmup", "scored")}
    return result


def blocked_replay(context: fb.FrameContextV1, reason: str, *, cleanup_errors: list[str] | None = None) -> dict:
    """Every planned row blocked with ``reason`` (nothing was opened or processed)."""
    result = _new_result(context)
    result["cleanup_errors"] = list(cleanup_errors or [])
    return _finish(context, result, 0, reason, "blocked")


class ReplayRefused(fb.BoundaryRejected):
    """The replay context was refused before processing; the transferred adapter and reader were closed
    exactly once. ``cleanup_errors`` lists any close failures."""

    def __init__(self, message: str, cleanup_errors: list[str]):
        super().__init__(message + (f"; cleanup errors: {cleanup_errors}" if cleanup_errors else ""))
        self.cleanup_errors = list(cleanup_errors)


def replay_recorded(context: fb.FrameContextV1, adapter: fb.FrameAdapter, reader: Any, masks: Any, *,
                    on_output: Callable[[fb.FrameOutputV1], None] | None = None, verify_media: bool = True,
                    clock: Callable[[], float] = time.perf_counter) -> dict:
    """Process warm-up then scored frames of ``context.permitted_order`` through a fresh boundary.

    ``reader``: ``FrameReader``-like (``read(frame) -> (rgb, ident)``, optional ``verify_media()``,
    ``close()``). ``masks``: ``MaskStore``-like (``load(frame_id, (h, w)) -> (mask | None, mident | None)``).
    Takes ownership of ``adapter`` and ``reader``: each is closed exactly once on every path. An invalid
    context raises ``ReplayRefused`` (after closing both) instead of inventing rows."""
    try:
        canonical, _fingerprint = fb.canonical_context(context)   # schema, binding, attestation, every field
        result = _new_result(canonical)
    except BaseException as exc:
        errors = fb._safe_close(adapter) + fb._safe_close(reader)
        raise ReplayRefused(f"replay refused before processing: {type(exc).__name__}: {exc}; adapter and reader closed",
                            errors) from exc
    boundary = fb.FrameBoundary()
    done = 0
    failure: str | None = None
    status_on_failure = "failed"
    try:
        verify = getattr(reader, "verify_media", None)
        if verify_media and callable(verify):
            try:
                result["media"] = verify()
            except Exception as exc:  # noqa: BLE001
                failure, status_on_failure = f"source_unverified: {type(exc).__name__}: {exc}", "blocked"
        if failure is None:
            try:
                result["identity"] = boundary.initialize(context, adapter)
            except Exception as exc:  # noqa: BLE001 - initialize already closed the adapter
                failure, status_on_failure = f"initialization_failed: {type(exc).__name__}: {exc}", "blocked"
        if failure is None:
            for fid in canonical.permitted_order:
                frame = canonical.frame(fid)
                phase = canonical.phase(fid)
                t0 = clock()
                rgb, ident, rgb_reason = None, None, None
                try:
                    rgb, ident = reader.read(frame)
                except Exception as exc:  # noqa: BLE001 - counted as an RGB-unavailable row
                    rgb_reason = f"{type(exc).__name__}: {exc}"
                t1 = clock()
                mask, mident, mask_reason = None, None, None
                try:
                    mask, mident = masks.load(fid, (canonical.height, canonical.width))
                    if mask is None:
                        mask_reason = "no observed mask for this frame"
                except Exception as exc:  # noqa: BLE001 - counted through the safe missing-mask path
                    mask, mident, mask_reason = None, None, f"{type(exc).__name__}: {exc}"
                t2 = clock()
                try:
                    inp = fb.FrameInputV1.from_reader(frame, rgb, ident, mask, mident, rgb_unavailable_reason=rgb_reason,
                                                      mask_unavailable_reason=mask_reason)
                except fb.BoundaryRejected as exc:
                    inp = fb.FrameInputV1.from_reader(frame, None, None, None, None,
                                                      rgb_unavailable_reason=f"reader packet rejected: {exc}",
                                                      mask_unavailable_reason=mask_reason)
                out = boundary.process(inp)
                t3 = clock()
                result[phase].append(out.result.to_dict())
                result["timings"][phase].append({"frame_id": fid, "read_ms": (t1 - t0) * 1000.0,
                                                 "mask_ms": (t2 - t1) * 1000.0, "process_ms": out.process_ms,
                                                 "end_to_end_ms": (t3 - t0) * 1000.0})
                done += 1
                if out.overlay_rgba is not None:
                    result["overlays_delivered"] += 1
                if on_output is not None:
                    on_output(out)
    except Exception as exc:  # noqa: BLE001 - every remaining planned row is still accounted for
        failure = failure or f"replay_failed: {type(exc).__name__}: {exc}"
    finally:
        if boundary.state == "new":
            result["cleanup_errors"] += fb._safe_close(adapter)   # never opened: still release its resources
        else:
            boundary.close()
        result["cleanup_errors"] += boundary.cleanup_errors
        result["cleanup_errors"] += fb._safe_close(reader)
    return _finish(canonical, result, done, failure, status_on_failure)


# --------------------------------------------------------------------------
# library entrypoints


def build_neural_replay(workspace: Any, clip: wc.ClipSpec, run: wc.RunSpec, session: Any, *, role: str, family: str,
                        mask_set: str | None, bundle: Any = None, source_mode: str = "recorded_finite") -> dict:
    """Context + adapter + reader + masks for an ACTUAL caller-built ``NeuralSession``. Takes ownership of
    ``session``: on any builder failure the session (and a created reader) is closed and the error re-raised."""
    from bench.workbench_cli import FrameReader, MaskStore
    reader = None
    try:
        binding = fb.neural_binding(clip, run, family=family, bundle=bundle)
        context = fb.FrameContextV1.build(clip, run, role, binding, source_mode=source_mode)
        reader = FrameReader(workspace, clip)
        masks = MaskStore(workspace, clip.clip_id, mask_set)
        return {"context": context, "adapter": fb.NeuralSessionAdapter(session), "reader": reader, "masks": masks}
    except BaseException:
        fb._safe_close(session)
        fb._safe_close(reader)
        raise


def build_relative_replay(workspace: Any, clip: wc.ClipSpec, run: wc.RunSpec, *, package_dir: Path,
                          mask_set: str | None, tracker: Any = None, source_mode: str = "recorded_finite") -> dict:
    """Context + adapter + reader + masks for the actual RelativeTracker over a byte-verified package
    directory (``RelativePackageProof.from_package_dir``). Closes the tracker on builder failure."""
    from bench import workbench_geometry as wg
    from bench.workbench_cli import FrameReader, MaskStore
    tracker = tracker if tracker is not None else wg.RelativeTracker()
    reader = None
    try:
        proof = fb.RelativePackageProof.from_package_dir(package_dir)
        binding = fb.relative_binding(proof, artwork_kind=run.artwork_kind)
        context = fb.FrameContextV1.build(clip, run, fb.RELATIVE_ROLE, binding, source_mode=source_mode)
        reader = FrameReader(workspace, clip)
        masks = MaskStore(workspace, clip.clip_id, mask_set)
        return {"context": context, "adapter": fb.RelativeTrackerAdapter(tracker, proof), "reader": reader, "masks": masks}
    except BaseException:
        fb._safe_close(tracker)
        fb._safe_close(reader)
        raise


@dataclass(frozen=True)
class NeuralFactories:
    """Optional replacements for ``run_neural_replay`` components. ``None`` means the real component.
    Any replacement is recorded in the result provenance and marks the result as test-double evidence."""
    bundle_loader: Callable[[Path], Any] | None = None
    admission_loader: Callable[[Path], Any] | None = None
    backend_factory: Callable[..., Any] | None = None
    artwork_factory: Callable[..., Any] | None = None
    gpu_stage: Callable[[str], Any] | None = None
    reader_factory: Callable[[Any, wc.ClipSpec], Any] | None = None
    masks_factory: Callable[[Any, str, str | None], Any] | None = None

    def injected(self) -> list[str]:
        return sorted(f.name for f in fields(self) if getattr(self, f.name) is not None)


def run_neural_replay(workspace: Any, clip: wc.ClipSpec, run: wc.RunSpec, *, role: str, metric_bundle_path: Path,
                      admission_dir: Path, head_path: Path | None, mask_set: str | None, device: str = "cuda",
                      factories: NeuralFactories | None = None) -> dict:
    """End-to-end neural recorded replay (parent-scheduled; loads weights on ``device`` by default).

    Real by default:
    - ``MetricBundle.from_file`` (re-hashes the GLB);
    - ``load_admission`` (recomputes the admission, never trusts booleans);
    - ``NeuralArtwork``;
    - ``workbench_telemetry.gpu_stage`` around backend construction and replay;
    - ``GoTrackMetricBackend``;
    - ``FrameReader`` and ``MaskStore``."""
    from bench import workbench_neural as wn
    f = factories or NeuralFactories()
    provenance = {"injected_factories": f.injected(),
                  "evidence_class": "test_double_execution" if f.injected() else "real_components",
                  "admission": "injected_test_double" if f.admission_loader else "workbench_neural.load_admission",
                  "backend": "injected_test_double" if f.backend_factory else "workbench_neural.GoTrackMetricBackend",
                  "artwork": "injected_test_double" if f.artwork_factory else "workbench_neural.NeuralArtwork"}
    bundle = (f.bundle_loader or wn.MetricBundle.from_file)(Path(metric_bundle_path))
    binding = fb.neural_binding(clip, run, family=bundle.family, bundle=bundle)
    context = fb.FrameContextV1.build(clip, run, role, binding)

    def done(result: dict) -> dict:
        result["provenance"] = provenance
        return result
    try:
        admission = (f.admission_loader or wn.load_admission)(Path(admission_dir))
    except Exception as exc:  # noqa: BLE001
        return done(blocked_replay(context, f"serving_admission_unverified: {type(exc).__name__}: {exc}"))
    if getattr(admission, "admitted", False) is not True:
        return done(blocked_replay(context, "serving_admission_unverified: "
                                   + "; ".join(list(getattr(admission, "reasons", ()))[:3])))
    if f.masks_factory is not None:
        masks = f.masks_factory(workspace, clip.clip_id, mask_set)
    else:
        from bench.workbench_cli import MaskStore
        masks = MaskStore(workspace, clip.clip_id, mask_set)
    artwork, artwork_reason = None, None
    try:
        sel = run.onboarding.selection
        sel_mask, _ = masks.load(sel.frame_id, (clip.height, clip.width))
        artwork = (f.artwork_factory or wn.NeuralArtwork)(bundle, sel, run.artwork_kind, clip_K=clip.intrinsics.K,
                                                          width=clip.width, height=clip.height, selection_mask=sel_mask)
    except Exception as exc:  # noqa: BLE001 - rows stay unrendered, never fake-rendered
        artwork_reason = f"artwork_unavailable: {type(exc).__name__}: {exc}"
    provenance["artwork_reason"] = artwork_reason
    if f.gpu_stage is not None:
        stage = f.gpu_stage
    else:
        from bench import workbench_telemetry as wt
        stage = wt.gpu_stage
    try:
        gpu = stage(role)
        gpu.__enter__()
    except Exception as exc:  # noqa: BLE001 - busy/foreign lock: nothing was constructed
        return done(blocked_replay(context, f"gpu_stage_unavailable: {type(exc).__name__}: {exc}"))

    def staged() -> dict:
        backend = None
        try:
            if f.backend_factory is not None:
                backend = f.backend_factory(role, bundle, head_path=head_path, device=device)
            else:
                backend = wn.GoTrackMetricBackend(role, bundle, device=device,
                                                  head_path=Path(head_path) if head_path else None)
            backend.seed_target_frame_id = binding["seed_target_frame_id"]
            session = wn.NeuralSession(role, backend, admission, run_id=run.run_id, clip=clip, family=bundle.family,
                                       renderer=artwork.render if artwork is not None else None)
        except Exception as exc:  # noqa: BLE001 - never faked; the constructed backend is closed
            errors = fb._safe_close(backend) if backend is not None else []
            return blocked_replay(context, f"neural_backend_unavailable: {type(exc).__name__}: {exc}",
                                  cleanup_errors=errors)
        reader_factory = f.reader_factory
        if reader_factory is None:
            from bench.workbench_cli import FrameReader
            reader_factory = FrameReader
        try:
            reader = reader_factory(workspace, clip)
        except Exception as exc:  # noqa: BLE001
            return blocked_replay(context, f"reader_unavailable: {type(exc).__name__}: {exc}",
                                  cleanup_errors=fb._safe_close(session))
        return replay_recorded(context, fb.NeuralSessionAdapter(session), reader, masks)

    # The stage is released exactly once. A release failure never discards the completed or blocked
    # rows: it is appended to cleanup_errors (completed -> completed_with_cleanup_errors; blocked/failed kept).
    try:
        result = staged()
    except BaseException as body_exc:
        try:
            gpu.__exit__(type(body_exc), body_exc, body_exc.__traceback__)
        except BaseException:  # noqa: BLE001 - the original body failure is the one re-raised
            pass
        raise
    try:
        gpu.__exit__(None, None, None)
    except Exception as exc:  # noqa: BLE001
        result["cleanup_errors"].append(f"gpu_stage.release: {type(exc).__name__}: {exc}"[:500])
        if result["status"] == "completed":
            result["status"] = "completed_with_cleanup_errors"
    return done(result)


def inspect() -> dict:
    """Static capabilities only: no data, model, run or GT reads; no GPU import; no writes; no inference."""
    return {"schema": "workbench_frame_replay_inspect_v1", "boundary_version": fb.BOUNDARY_VERSION,
            "records": [fb.SCHEMA_CONTEXT, fb.SCHEMA_INPUT, fb.SCHEMA_OUTPUT],
            "lifecycle": "initialize(context, adapter) -> process(FrameInputV1)* -> reset('replay_restart') -> close()",
            "adapters": {"NeuralSessionAdapter": list(fb.NeuralSessionAdapter.roles),
                         "RelativeTrackerAdapter": list(fb.RelativeTrackerAdapter.roles)},
            "relative_input": "RelativePackageProof.from_package_dir | .from_memory (manufactured bytes)",
            "source_modes": {"recorded_finite": "all roles",
                             "camera_style_synthetic": "relative_geometry software API only; never inherits a neural admission",
                             "live_neural_camera": "UNSUPPORTED"},
            "entrypoints": ["replay_recorded(context, adapter, reader, masks, on_output=None)",
                            "build_neural_replay(workspace, clip, run, session, role=, family=, mask_set=, bundle=None)",
                            "build_relative_replay(workspace, clip, run, package_dir=, mask_set=, tracker=None)",
                            "run_neural_replay(workspace, clip, run, role=, metric_bundle_path=, admission_dir=, "
                            "head_path=, mask_set=, device='cuda', factories=None)"],
            "neural_cli_runner": "not provided: an actual backend needs parent-scheduled weights and a verified admission",
            "flags": FLAGS, "sources": source_identity()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in recorded replay through the portable frame boundary. "
                                                 "Replay is a library API; this CLI only inspects capabilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="print static capabilities and source identities (no data reads, no inference)")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
