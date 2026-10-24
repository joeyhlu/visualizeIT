"""Sealed CPU-only evidence recovery for the failed current-source mug run.

The parent's run ``run-20261021T071542-8205a748`` finished both neural roles
(480 unique rows) but exited 1: two ``frozen_baseline`` overlay PNGs
(``s000974``, ``s000975``) were never written, and the stage-cache commit then
failed on the first missing file. That run, its terminal seal and the original
mug run ``run-20261021T002156-1b2de77d`` stay immutable.

This tool writes a NEW derivative run that reuses the saved predictions:

* every row is byte-for-byte the source row except ``run_id``;
* the 369 existing overlay PNGs are copied and re-verified (bytes, strict RGBA,
  size, binary alpha, alpha count == recorded ``rendered_pixels``);
* the two missing overlays are re-rendered on the CPU with the existing
  ``NeuralArtwork.render`` from the saved pose, the frame's observed mask and
  the frozen artwork rebuilt exactly (record SHA pinned). Every recorded
  ``render_stats`` field must match before the checked PNG writer publishes
  the result. These are deterministic reconstructions that agree with the
  recorded render state; the historical missing bytes cannot be proven.

No network, model, GPU stage, neural session, segmentation or stage cache is
touched. Guards (``BaseException``, restored on every path) block them and any
model-runtime import. No evaluator, reference pose, ground truth or depth is
read. The report is published last, after every check, with the literal status
``completed_evidence_recovery``; a failure leaves the partial derivative plus a
failure receipt and never promotes a completed status.

    prepare --out FRESH-plan.json
    run --plan FILE --expected-plan-sha256 HEX
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN_SCHEMA = "workbench_mug_artifact_recovery_plan_v1"
FAILURE_SCHEMA = "workbench_mug_artifact_recovery_failure_v1"
RECOVERY_STATUS = "completed_evidence_recovery"
WORKSPACE = ".cache/video-workbench"
CLIP_ID = "dev-mug"
SOURCE_RUN_ID = "run-20261021T071542-8205a748"
ORIGINAL_RUN_ID = "run-20261021T002156-1b2de77d"
SEAL_PATH = ".cache/video-workbench/derived/validation/mug-failed-run-seal-v1/seal.json"
SEAL_SHA256 = "f87d3620110e27e54a52bd8ba11821b60c0f4592b391b1dc9e37a2ee55c70975"
LAUNCHER_PLAN_PATH = ".cache/video-workbench/derived/validation/mug-current-source-revalidation-plan-v1.json"
LAUNCHER_PLAN_SHA256 = "ca23fe7ebacabfa3ec7ea28c9feadc48123a653f180a5e6f16dc051361b98a50"
ADMISSION_DIR = ".cache/video-workbench/admission/attempt-06"
BUNDLE_PATH = ".cache/video-workbench/admission/bundle-mug-v1.json"
ARTWORK_RECORD_SHA256 = "d20f8b89f9c5cb8e574875e7d44d947425b3e5c9b6af2d257ead94f52bd7aace"
ARTWORK_PATCH_FACES = 2585
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
ROLES = ("primary_neural", "frozen_baseline")
FRAME_IDS = tuple(f"s{i:06d}" for i in range(827, 1067))
SEED_FRAME_ID = "s000826"
MISSING = {"evidence/frozen_baseline/s000974.png": ("frozen_baseline", "s000974", 29),
           "evidence/frozen_baseline/s000975.png": ("frozen_baseline", "s000975", 38)}
EXPECTED_COUNTS = {"primary_neural": {"tracking": 225, "rendered": 187},
                   "frozen_baseline": {"tracking": 224, "rendered": 184}}
EXPECTED_STAGES = {"source": "completed", "masks": "completed", "primary_neural_load": "running",
                   "primary_neural": "completed", "frozen_baseline_load": "running", "frozen_baseline": "completed",
                   "stage_cache_store_06": "failed"}
# The computational closure that produced the saved rows; must still be byte-identical.
COMPUTATIONAL_SOURCES = {
    "workbench_neural": "86d0caf6d3f32607d28ddb9cfc7c9b063e6767ded8710fa5d8aa0384968d6078",
    "workbench_render": "da9d79157289d093399cc055ab98e8c8534f3aa2f0cbc52ac288c327d2fdf57a",
    "workbench_geometry": "80a6369d01208f07b2dd8a78b437fa815fd2a51832fef89d3564926e5adca7ed",
}
# Current exporter side (recorded, separate from the source producer snapshots).
EXPORTER_MODULES = ("workbench_cli", "workbench_contract", "workbench_image_io", "workbench_telemetry", "vision")
SEAL_RUN_FILES = 380
SEAL_SNAPSHOTS = 10
MAX_PLAN_BYTES = 256 << 10
MAX_DERIVATIVE_BYTES = 8 << 20
MAX_REPORT_BYTES = 512 << 10
MAX_REGENERATED_PNG_BYTES = 256 << 10
MAX_RECEIPT_BYTES = 64 << 10
MARGIN_BYTES = 512 << 10
FORBIDDEN_IMPORTS = ("torch", "torchvision", "sam2", "pyrender", "OpenGL", "bench.quality_gotrack",
                     "bench.quality_gotrack_inference")
CLAIMS = ("Evidence recovery only: saved predictions of the failed source run are reused unchanged; no new neural "
          "inference, no stage-cache hit, no accuracy, phone or speed claim. Two overlays are deterministic CPU "
          "reconstructions that agree with the recorded render state; their historical bytes cannot be proven.")


class RecoveryError(RuntimeError):
    """Pin, closure, invariant or output failure (fail closed)."""


class RecoveryViolation(BaseException):
    """A guarded route (model, GPU stage, session, segmentation, cache) was attempted."""


# --------------------------------------------------------------------------
# small helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    from bench import workbench_contract as wc
    return wc.canonical_json(value)


def loads_strict(raw: bytes, what: str) -> Any:
    from bench import workbench_contract as wc
    try:
        return wc.loads_strict(raw)
    except Exception as exc:  # noqa: BLE001
        raise RecoveryError(f"{what}: invalid JSON ({exc})") from None


def _abs(rel: str) -> Path:
    return ROOT / Path(rel.replace("\\", "/"))


def _rel(path: Path) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise RecoveryError(f"{path} is outside the repository") from None


def _regular(path: Path, what: str) -> Path:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise RecoveryError(f"{what} is missing or not a regular file: {_rel(p)}")
    return p


def _pinned(rel: str, sha: str, what: str) -> bytes:
    raw = _regular(_abs(rel), what).read_bytes()
    if sha256_bytes(raw) != sha:
        raise RecoveryError(f"{what} differs from its pin ({rel})")
    return raw


def _module_sha(name: str) -> str:
    return sha256_file(_regular(ROOT / "bench" / f"{name}.py", f"bench/{name}.py"))


def _tree_size(path: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            total += (Path(base) / name).stat().st_size
    return total


def _tree_files(path: Path) -> dict[str, Path]:
    out = {}
    for base, dirs, files in os.walk(path):
        for name in dirs + files:
            if (Path(base) / name).is_symlink():
                raise RecoveryError(f"link inside {_rel(path)}: {name}")
        for name in files:
            p = Path(base) / name
            out[p.relative_to(path).as_posix()] = p
    return out


# --------------------------------------------------------------------------
# pure checks (parent tests can call these with manufactured inputs)

def check_seal(seal: Mapping) -> None:
    # Exact types first: Python equality would accept True for 1 and 1 for True.
    for key in ("exit_code", "rows"):
        if type(seal.get(key)) is not int:
            raise RecoveryError(f"seal.{key} must be an integer (got {type(seal.get(key)).__name__})")
    for key in ("worker_exited", "original_run_immutable", "no_new_neural_execution"):
        if seal.get(key) is not True:
            raise RecoveryError(f"seal.{key} must be the boolean true (got {seal.get(key)!r})")
    want = {"run_id": SOURCE_RUN_ID, "exit_code": 1, "original_report_status": "failed",
            "rows": 480, "role_stages": {"primary_neural": "completed", "frozen_baseline": "completed"},
            "missing_overlays": sorted(MISSING)}
    for key, value in want.items():
        got = seal.get(key)
        if (sorted(got) if key == "missing_overlays" and isinstance(got, list) else got) != value:
            raise RecoveryError(f"seal.{key} is {got!r}, expected {value!r}")
    if seal.get("launcher_plan", {}).get("sha256") != LAUNCHER_PLAN_SHA256 or \
            _norm(seal.get("launcher_plan", {}).get("path", "")) != LAUNCHER_PLAN_PATH:
        raise RecoveryError("seal does not pin the launcher plan")
    if not isinstance(seal.get("run_files"), Mapping) or len(seal["run_files"]) != SEAL_RUN_FILES:
        raise RecoveryError(f"seal must pin exactly {SEAL_RUN_FILES} physical run files")
    if not isinstance(seal.get("source_snapshot"), Mapping) or len(seal["source_snapshot"]) != SEAL_SNAPSHOTS:
        raise RecoveryError(f"seal must pin exactly {SEAL_SNAPSHOTS} producer source snapshots")
    if set(seal.get("failure_stage") or {}) != {"stage_cache_store_06"}:
        raise RecoveryError("seal must record exactly the stage_cache_store_06 failure")
    models = seal.get("models") or {}
    if (models.get("primary_neural", {}).get("installed_head_sha256") != HEAD_SHA256
            or any(models.get(r, {}).get("base_checkpoint_sha256") != BASE_SHA256 for r in ROLES)):
        raise RecoveryError("seal model identities differ from the head/base pins")


def _norm(rel: str) -> str:
    return str(rel).replace("\\", "/")


def check_source_report(report: Mapping) -> None:
    if report.get("status") != "failed" or report.get("run_id") != SOURCE_RUN_ID:
        raise RecoveryError("source report must be the failed source run")
    stages = report.get("stages") or {}
    got = {k: (v or {}).get("status") for k, v in stages.items()}
    if got != EXPECTED_STAGES:
        raise RecoveryError(f"source stages differ from the known failure: {got}")
    reason = str(stages["stage_cache_store_06"].get("reason") or "")
    if not reason.startswith("FileNotFoundError") or "s000974.png" not in reason:
        raise RecoveryError("source failure is not the known missing-overlay cache-store failure")
    tel = report.get("telemetry") or {}
    if tel.get("gpu_used") is not True or tel.get("run_cache_mode") != "cold" or len(tel.get("frames") or []) != 480:
        raise RecoveryError("source telemetry is not the cold neural execution")


def check_source_rows(rows: Sequence[Mapping], contract_sha256: str) -> dict:
    """Invariants of the 480 saved rows; returns {overlay_path: row} for rows with an overlay."""
    if len(rows) != 480:
        raise RecoveryError(f"expected 480 rows, found {len(rows)}")
    seen, overlays = set(), {}
    counts = {r: {"tracking": 0, "rendered": 0} for r in ROLES}
    for row in rows:
        role, fid = row.get("backend_role"), row.get("frame_id")
        if role not in ROLES or fid not in FRAME_IDS or (role, fid) in seen:
            raise RecoveryError(f"unexpected or duplicate row {role}/{fid}")
        seen.add((role, fid))
        ident = row.get("backend_identity") or {}
        if (row.get("run_id") != SOURCE_RUN_ID or row.get("clip_id") != CLIP_ID or row.get("mask_state") != "observed"
                or row.get("mask_frame_id") != fid or ident.get("role") != role
                or ident.get("base_checkpoint_sha256") != BASE_SHA256
                or ident.get("admission_contract_sha256") != contract_sha256):
            raise RecoveryError(f"row {role}/{fid} identity/mask differs from the pins")
        if role == "primary_neural" and (ident.get("export_sha256") != HEAD_SHA256
                                         or ident.get("installed_head_sha256") != HEAD_SHA256):
            raise RecoveryError(f"row {role}/{fid} head identity differs from {HEAD_SHA256[:8]}")
        if role == "frozen_baseline" and ident.get("installed_head_sha256") is not None:
            raise RecoveryError(f"row {role}/{fid} claims an installed head")
        evidence = row.get("evidence") or {}
        if "artifact_write_error" in evidence:
            raise RecoveryError(f"row {role}/{fid} records an artifact write error")
        rendered = row.get("render_state") == "rendered"
        counts[role]["tracking"] += row.get("pose_state") == "tracking"
        counts[role]["rendered"] += rendered
        overlay = evidence.get("overlay")
        if rendered != bool(overlay):
            raise RecoveryError(f"row {role}/{fid} overlay presence differs from its render state")
        if overlay:
            if overlay != f"evidence/{role}/{fid}.png":
                raise RecoveryError(f"row {role}/{fid} overlay path is not canonical")
            stats = evidence.get("render_stats") or {}
            if not isinstance(stats.get("rendered_pixels"), int) or stats["rendered_pixels"] <= 0:
                raise RecoveryError(f"row {role}/{fid} rendered without recorded pixels")
            overlays[overlay] = row
    if counts != EXPECTED_COUNTS:
        raise RecoveryError(f"row counts differ from the sealed execution: {counts}")
    for path, (role, fid, pixels) in MISSING.items():
        row = overlays.get(path)
        if row is None or row.get("pose_state") != "tracking" or row.get("failure_reason") is not None \
                or row["evidence"]["render_stats"].get("rendered_pixels") != pixels:
            raise RecoveryError(f"missing overlay row {path} differs from the recorded state")
    return overlays


def finalize_render_stats(stats: Mapping) -> dict:
    """The exact post-processing NeuralSession.process applies to renderer stats."""
    out = dict(stats)
    if out.get("reason") is None and "rendered_pixels" in out and out["rendered_pixels"] == 0:
        out["reason"] = "no visible supported artwork pixels"
    return out


def check_regenerated(recorded_stats: Mapping, drawn: Mapping, shape: tuple[int, int], expected_pixels: int):
    """Every recorded render_stats field (incl. reason) must match; RGBA must be exact-format with binary alpha."""
    import numpy as np
    stats = finalize_render_stats(drawn.get("stats") or {})
    if stats != dict(recorded_stats) or canonical(stats) != canonical(dict(recorded_stats)):
        raise RecoveryError(f"re-render stats {stats} differ from the recorded {dict(recorded_stats)}")
    rgba = drawn.get("rgba")
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.shape != (shape[0], shape[1], 4):
        raise RecoveryError("re-render is not HxWx4 uint8 RGBA")
    alpha = rgba[:, :, 3]
    if not np.isin(alpha, (0, 255)).all():
        raise RecoveryError("re-render alpha is not binary")
    on = int(np.count_nonzero(alpha == 255))
    if on != stats.get("rendered_pixels") or on != expected_pixels:
        raise RecoveryError(f"re-render alpha count {on} differs from recorded {stats.get('rendered_pixels')}")
    return rgba


def check_copied_png(data: bytes, shape: tuple[int, int], rendered_pixels: int) -> dict:
    from bench import workbench_image_io as wio
    try:
        info = wio.inspect_png_rgba(data, expected_shape=shape)
    except wio.ImageWriteError as exc:
        raise RecoveryError(f"source overlay is not strict RGBA: {exc}") from None
    if not info["alpha_binary"] or info["alpha_on_pixels"] != rendered_pixels:
        raise RecoveryError(f"source overlay alpha ({info['alpha_on_pixels']}, binary={info['alpha_binary']}) "
                            f"differs from recorded rendered_pixels {rendered_pixels}")
    return info


def derivative_rows(rows: Sequence[Mapping], new_run_id: str) -> list[dict]:
    return [dict(row, run_id=new_run_id) for row in rows]


def rows_equal_except_run_id(old: Sequence[Mapping], new: Sequence[Mapping]) -> bool:
    if len(old) != len(new):
        return False
    return all(canonical({k: v for k, v in a.items() if k != "run_id"}) ==
               canonical({k: v for k, v in b.items() if k != "run_id"}) for a, b in zip(old, new))


def derivative_config(source_config: Mapping) -> dict:
    return {"mask_set": source_config["mask_set"], "evidence_recovery_only": True,
            "development_diagnostic_only": True, "source_neural_run_id": SOURCE_RUN_ID,
            "source_run_config": dict(source_config),
            "recovery_note": "saved predictions reused; 2 missing frozen_baseline overlays re-rendered on CPU; "
                             "no new neural inference"}


def budget_reserve(copied_png_bytes: int, results_bytes: int, run_json_bytes: int, artwork_bytes: int) -> int:
    """Conservative derivative bytes incl. one temp of the largest file (hard-link publication)."""
    largest = max(results_bytes, MAX_REPORT_BYTES)
    return (copied_png_bytes + results_bytes + run_json_bytes + 8192 + artwork_bytes + MAX_REPORT_BYTES
            + 2 * MAX_REGENERATED_PNG_BYTES + MAX_RECEIPT_BYTES + largest + MARGIN_BYTES)


# --------------------------------------------------------------------------
# guards (restored on every path)

class _ImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, attempts: list):
        self.attempts = attempts

    def find_spec(self, name, path=None, target=None):  # noqa: D401
        if any(name == m or name.startswith(m + ".") for m in FORBIDDEN_IMPORTS):
            self.attempts.append(f"import {name}")
            raise RecoveryViolation(f"model-runtime import attempted: {name}")
        return None


@contextmanager
def recovery_guards() -> Iterator[list]:
    from bench import workbench_cache as wcache, workbench_cli as cli, workbench_neural as wn, workbench_telemetry as wt
    loaded = sorted(m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS))
    if loaded:
        raise RecoveryViolation(f"model runtime already imported: {loaded[:5]}")
    attempts: list[str] = []
    saved: list[tuple[Any, str, Any]] = []
    blocker = _ImportBlocker(attempts)

    def deny(what: str):
        def _deny(*_a, **_k):
            attempts.append(what)
            raise RecoveryViolation(what)
        return _deny

    def patch(owner, name, value):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    try:
        patch(wt, "gpu_stage", deny("GPU stage entered"))
        patch(wn, "GoTrackMetricBackend", deny("neural backend construction attempted"))
        patch(wn, "NeuralSession", deny("NeuralSession construction attempted"))
        patch(wn, "segment_prompted", deny("SAM segmentation attempted"))
        patch(cli, "execute_run", deny("pipeline execution attempted"))
        patch(cli.Pipeline, "execute", deny("pipeline execution attempted"))
        patch(wcache.StageCache, "store", deny("stage-cache store attempted"))
        sys.meta_path.insert(0, blocker)
        yield attempts
    finally:
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


# --------------------------------------------------------------------------
# binding (prepare, and closure before/after run)

def bind_inputs() -> tuple[dict, dict]:
    """Verify every pinned input read-only. Returns (binding, context); the binding is JSON-able."""
    from bench import workbench_cli as cli, workbench_contract as wc, workbench_geometry as wg
    from bench import workbench_neural as wn, workbench_telemetry as wt
    ws = cli.Workspace(ROOT / WORKSPACE)
    seal = loads_strict(_pinned(SEAL_PATH, SEAL_SHA256, "terminal seal"), "seal")
    check_seal(seal)
    launcher = loads_strict(_pinned(LAUNCHER_PLAN_PATH, LAUNCHER_PLAN_SHA256, "launcher plan"), "launcher plan")
    if (launcher.get("prospective_run_id") != SOURCE_RUN_ID or launcher.get("original_run_id") != ORIGINAL_RUN_ID
            or launcher.get("head_sha256") != HEAD_SHA256 or launcher.get("base_sha256") != BASE_SHA256):
        raise RecoveryError("launcher plan identities differ")
    log = seal.get("terminal_log") or {}
    _pinned(_norm(log.get("path", "")), log.get("sha256", ""), "terminal log")
    # producer source snapshots (old CLI bytes etc.), kept separate from the current exporter
    snapshots = {}
    for name, entry in seal["source_snapshot"].items():
        _pinned(_norm(entry["path"]), entry["sha256"], f"source snapshot {name}")
        snapshots[name] = entry["sha256"]
    # failed source run: exactly the sealed 380 physical files, the 2 known overlays absent
    run_dir = ws.runs_dir / SOURCE_RUN_ID
    sealed = {_norm(k): v for k, v in seal["run_files"].items()}
    physical = _tree_files(run_dir)
    if set(physical) != set(sealed):
        raise RecoveryError(f"source run files differ from the seal (extra {sorted(set(physical) - set(sealed))[:3]}, "
                            f"missing {sorted(set(sealed) - set(physical))[:3]})")
    if any(p in physical for p in MISSING):
        raise RecoveryError("a sealed-missing overlay now exists in the source run")
    blobs: dict[str, bytes] = {}
    for rel, pin in sealed.items():
        raw = physical[rel].read_bytes()
        if sha256_bytes(raw) != pin["sha256"] or len(raw) != pin["bytes"]:
            raise RecoveryError(f"source run file {rel} differs from the seal")
        blobs[rel] = raw
    run_doc = loads_strict(blobs["run.json"], "source run.json")
    report = loads_strict(blobs["report.json"], "source report.json")
    lines = [line for line in blobs["results.jsonl"].split(b"\n") if line]
    rows = [loads_strict(line, "source results row") for line in lines]
    if b"".join(canonical(r) + b"\n" for r in rows) != blobs["results.jsonl"]:
        raise RecoveryError("source results.jsonl is not canonical")
    spec = run_doc.get("run") or {}
    if canonical(spec) != canonical(launcher.get("run")) or spec.get("run_id") != SOURCE_RUN_ID:
        raise RecoveryError("source RunSpec differs from the launcher plan")
    if tuple(spec.get("scheduled_frame_ids") or ()) != FRAME_IDS or tuple(spec.get("roles") or ()) != ROLES:
        raise RecoveryError("source window/roles differ (expected s000827..s001066, both neural roles)")
    check_source_report(report)
    if canonical((report.get("dependencies") or {}).get("sources")) != canonical(
            {k: v for k, v in snapshots.items() if k in (report.get("dependencies") or {}).get("sources", {})}) \
            or snapshots.get("workbench_render") != (launcher.get("source_pins") or {}).get("workbench_render"):
        raise RecoveryError("producer source snapshots differ from the recorded producer sources")
    # admission files pinned by the launcher plan; contract self-hash bound into every row
    adm_files = launcher.get("admission_files") or {}
    for name, sha in adm_files.items():
        _pinned(f"{ADMISSION_DIR}/{name}", sha, f"admission {name}")
    contract = loads_strict(_abs(f"{ADMISSION_DIR}/contract.json").read_bytes(), "admission contract")
    contract_sha = contract.pop("contract_sha256", None)
    if not contract_sha or wc.canonical_sha256(contract) != contract_sha:
        raise RecoveryError("admission contract self-hash differs")
    overlays = check_source_rows(rows, contract_sha)
    present = {p for p in physical if p.startswith("evidence/")}
    if present != set(overlays) - set(MISSING) or len(present) != 369:
        raise RecoveryError("source overlay files differ from the rows' overlay references")
    # clip, masks (file + semantic digests), bundle, GLB
    clip_raw = _pinned(_rel(ws.clip_path(CLIP_ID)), launcher["clip_sha256"], "clip record")
    clip = wc.ClipSpec.from_dict(loads_strict(clip_raw, "clip"))
    if wc.canonical_sha256(clip.to_dict()) != launcher["clip_sha256"]:
        raise RecoveryError("clip record differs")
    mask_set = (spec.get("config") or {}).get("mask_set")
    manifest_rel = _rel(ws.masks_dir / CLIP_ID / str(mask_set) / "manifest.json")
    _pinned(manifest_rel, launcher["mask_manifest_sha256"], "mask manifest")
    masks = cli.MaskStore(ws, CLIP_ID, mask_set)
    shape = (clip.height, clip.width)
    mask_files, semantic = {}, {}
    for fid in (SEED_FRAME_ID,) + FRAME_IDS:
        mask, mident = masks.load(fid, shape)
        if mask is None:
            raise RecoveryError(f"observed mask for {fid} is unavailable")
        mask_files[fid], semantic[fid] = mident["file_sha256"], mident["mask_sha256"]
    for row in rows:
        if row["mask_sha256"] != semantic[row["frame_id"]]:
            raise RecoveryError(f"row {row['backend_role']}/{row['frame_id']} semantic mask digest differs")
    _pinned(BUNDLE_PATH, launcher["bundle_sha256"], "metric bundle")
    bundle = wn.MetricBundle.from_file(_abs(BUNDLE_PATH), root=ROOT)
    if bundle.glb_sha256 != launcher["glb_sha256"] or sha256_file(_regular(bundle.glb_path, "GLB")) != launcher["glb_sha256"]:
        raise RecoveryError("physical GLB differs from the pin")
    sel = (spec.get("onboarding") or {}).get("selection") or {}
    if bundle.seed_frame_id != SEED_FRAME_ID or sel.get("frame_id") != SEED_FRAME_ID or sel.get("reviewed") is not True:
        raise RecoveryError("artwork anchor must be the reviewed selection on the seed frame")
    artwork_doc = loads_strict(blobs["neural-artwork.json"], "source neural-artwork.json")
    if artwork_doc.get("record_sha256") != ARTWORK_RECORD_SHA256 or artwork_doc.get("patch_faces") != ARTWORK_PATCH_FACES:
        raise RecoveryError("source artwork record differs from the frozen record")
    # computational closure must be the producer's; exporter recorded separately
    computational = {m: _module_sha(m) for m in COMPUTATIONAL_SOURCES}
    if computational != COMPUTATIONAL_SOURCES or any(snapshots.get(m) != s for m, s in COMPUTATIONAL_SOURCES.items()):
        raise RecoveryError(f"computational sources differ from the producer pins: {computational}")
    original = {}
    orig_dir = ws.runs_dir / ORIGINAL_RUN_ID
    for name in ("run.json", "report.json", "results.jsonl", "neural-artwork.json"):
        original[name] = sha256_file(_regular(orig_dir / name, f"original {name}"))
    if original["run.json"] != launcher.get("original_run_sha256"):
        raise RecoveryError("original mug run.json differs from the launcher pin")
    binding = {
        "seal": {"path": SEAL_PATH, "sha256": SEAL_SHA256},
        "launcher_plan": {"path": LAUNCHER_PLAN_PATH, "sha256": LAUNCHER_PLAN_SHA256},
        "terminal_log": {"path": _norm(log["path"]), "sha256": log["sha256"]},
        "source_run": {"run_id": SOURCE_RUN_ID, "files": len(physical),
                       "files_sha256": wc.canonical_sha256({k: sealed[k] for k in sorted(sealed)}),
                       "missing": sorted(MISSING)},
        "producer_source_snapshots": snapshots,
        "original_run": {"run_id": ORIGINAL_RUN_ID, "files": original},
        "clip": {"clip_id": CLIP_ID, "sha256": launcher["clip_sha256"]},
        "masks": {"mask_set": mask_set, "manifest_sha256": launcher["mask_manifest_sha256"],
                  "files_sha256": wc.canonical_sha256(mask_files), "semantic_sha256": wc.canonical_sha256(semantic)},
        "bundle": {"path": BUNDLE_PATH, "sha256": launcher["bundle_sha256"]},
        "glb": {"path": _rel(bundle.glb_path), "sha256": launcher["glb_sha256"]},
        "admission": {"dir": ADMISSION_DIR, "files": dict(adm_files), "contract_sha256": contract_sha},
        "computational_sources": computational,
        "exporter_sources": {**{m: _module_sha(m) for m in EXPORTER_MODULES}, "tool": sha256_file(Path(__file__))},
        "runtime": wt.dependency_versions(),
        "model_verification": seal.get("model_bytes_verification"),
    }
    context = {"ws": ws, "seal": seal, "run_doc": run_doc, "report": report, "rows": rows, "blobs": blobs,
               "clip": clip, "masks": masks, "bundle": bundle, "overlays": overlays, "artwork_doc": artwork_doc,
               "shape": shape}
    return binding, context


# --------------------------------------------------------------------------
# prepare

def _publish(path: Path, data: bytes, what: str) -> dict:
    from bench import workbench_image_io as wio
    try:
        return wio.publish_new_bytes(path, data)
    except wio.ImageWriteError as exc:
        raise RecoveryError(f"{what}: {exc}") from None


def build_plan() -> dict:
    from bench import workbench_contract as wc
    with recovery_guards():
        binding, ctx = bind_inputs()
    ws, blobs = ctx["ws"], ctx["blobs"]
    new_run_id = f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if len(new_run_id) != len(SOURCE_RUN_ID) or os.path.lexists(ws.runs_dir / new_run_id):
        raise RecoveryError("derivative run id is not fresh")
    spec = dict(ctx["run_doc"]["run"], run_id=new_run_id, config=derivative_config(ctx["run_doc"]["run"]["config"]))
    run_spec = wc.RunSpec.from_dict(spec)
    if canonical(run_spec.to_dict()) != canonical(spec):
        raise RecoveryError("derivative RunSpec does not round-trip")
    attestation = run_spec.validate_against(ctx["clip"])
    if canonical(attestation) != canonical(ctx["run_doc"].get("prefix_attestation")):
        raise RecoveryError("derivative prefix attestation differs from the source")
    copied = sum(len(v) for k, v in blobs.items() if k.startswith("evidence/"))
    reserve = budget_reserve(copied, len(blobs["results.jsonl"]), len(blobs["run.json"]),
                             len(blobs["neural-artwork.json"]))
    if reserve > MAX_DERIVATIVE_BYTES:
        raise RecoveryError(f"derivative reserve {reserve} exceeds {MAX_DERIVATIVE_BYTES} bytes")
    regenerate = {}
    for path, (role, fid, _pixels) in MISSING.items():
        regenerate[path] = {"role": role, "frame_id": fid,
                            "recorded_render_stats": ctx["overlays"][path]["evidence"]["render_stats"]}
    return {"schema": PLAN_SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool_sha256": sha256_file(Path(__file__)), "binding": binding,
            "derivative": {"run_id": new_run_id, "dir": _rel(ws.runs_dir / new_run_id), "run_spec": spec,
                           "prefix_attestation": attestation},
            "artwork_record_sha256": ARTWORK_RECORD_SHA256, "copy_overlays": 369, "regenerate": regenerate,
            "budget": {"reserve_bytes": reserve, "cap_bytes": MAX_DERIVATIVE_BYTES, "copied_png_bytes": copied},
            "claims": CLAIMS}


def write_plan(out: Path) -> dict:
    from bench import workbench_cli as cli
    target = Path(os.path.abspath(out))
    _rel(target)
    if target.suffix != ".json" or os.path.lexists(target) or not target.parent.is_dir():
        raise RecoveryError("plan output must be a fresh .json path in an existing directory")
    plan = build_plan()
    data = canonical(plan)
    if len(data) > MAX_PLAN_BYTES:
        raise RecoveryError(f"plan is {len(data)} bytes, above {MAX_PLAN_BYTES}")
    try:
        cli.Workspace(ROOT / WORKSPACE).require_budget(extra_derived=len(data))
    except cli.WorkbenchCLIError as exc:
        raise RecoveryError(f"budget: {exc}") from None
    sha = _publish(target, data, "plan")["sha256"]
    return {"plan": _rel(target), "plan_sha256": sha, "derivative_run_id": plan["derivative"]["run_id"],
            "reserve_bytes": plan["budget"]["reserve_bytes"]}


# --------------------------------------------------------------------------
# run

def load_plan(path: Path, expected_sha256: str) -> dict:
    raw = _regular(Path(path), "plan").read_bytes()
    if len(raw) > MAX_PLAN_BYTES or sha256_bytes(raw) != expected_sha256:
        raise RecoveryError("plan bytes differ from the expected SHA-256 (or exceed the cap)")
    plan = loads_strict(raw, "plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise RecoveryError("plan schema differs")
    if plan.get("tool_sha256") != sha256_file(Path(__file__)):
        raise RecoveryError("tool differs from the version that built the plan")
    if plan["budget"]["reserve_bytes"] > MAX_DERIVATIVE_BYTES:
        raise RecoveryError("plan reserve exceeds the derivative cap")
    return plan


def _require_room(ws, target: Path, upcoming: int, reserve: int) -> None:
    from bench import workbench_cli as cli
    used = _tree_size(target) if target.exists() else 0
    if used + upcoming > MAX_DERIVATIVE_BYTES:
        raise RecoveryError(f"derivative would exceed {MAX_DERIVATIVE_BYTES} bytes")
    try:
        ws.require_budget(extra_derived=max(0, reserve - used))
    except cli.WorkbenchCLIError as exc:
        raise RecoveryError(f"budget: {exc}") from None


def _recover(plan: Mapping, plan_sha256: str, stages: dict, started: float) -> dict:
    """All writes happen here, inside the guards, after the closure-before check."""
    import numpy as np
    from bench import workbench_contract as wc, workbench_image_io as wio, workbench_neural as wn
    binding, ctx = bind_inputs()
    if canonical(binding) != canonical(plan["binding"]):
        diff = sorted(k for k in set(binding) | set(plan["binding"])
                      if canonical(binding.get(k)) != canonical(plan["binding"].get(k)))
        raise RecoveryError(f"closure before recovery differs from the plan: {diff}")
    stages["closure_before"] = {"stage": "closure_before", "status": "completed"}
    ws, blobs, shape, rows = ctx["ws"], ctx["blobs"], ctx["shape"], ctx["rows"]
    deriv = plan["derivative"]
    target = ws.runs_dir / deriv["run_id"]
    if _rel(target) != deriv["dir"] or os.path.lexists(target):
        raise RecoveryError("derivative run directory is not fresh")
    reserve = plan["budget"]["reserve_bytes"]
    _require_room(ws, target, reserve, reserve)
    run_spec = wc.RunSpec.from_dict(deriv["run_spec"])
    if canonical(run_spec.to_dict()) != canonical(deriv["run_spec"]) or \
            canonical(run_spec.validate_against(ctx["clip"])) != canonical(deriv["prefix_attestation"]):
        raise RecoveryError("derivative RunSpec/attestation differs from the plan")
    # frozen artwork rebuilt exactly before anything renders
    sel = run_spec.onboarding.selection
    sel_mask, _ = ctx["masks"].load(sel.frame_id, shape)
    artwork = wn.NeuralArtwork(ctx["bundle"], sel, run_spec.artwork_kind, clip_K=ctx["clip"].intrinsics.K,
                               width=ctx["clip"].width, height=ctx["clip"].height, selection_mask=sel_mask)
    artwork_bytes = canonical({**artwork.record, "record_sha256": artwork.sha256})
    if artwork.sha256 != ARTWORK_RECORD_SHA256 or artwork.record["patch_faces"] != ARTWORK_PATCH_FACES \
            or artwork_bytes != blobs["neural-artwork.json"]:
        raise RecoveryError("rebuilt artwork differs from the frozen record")
    stages["artwork_rebuild"] = {"stage": "artwork_rebuild", "status": "completed", "record_sha256": artwork.sha256,
                                 "patch_faces": ARTWORK_PATCH_FACES}
    # re-render the two missing overlays in memory and check every recorded stat first
    regenerated = {}
    for path, (role, fid, pixels) in MISSING.items():
        row = ctx["overlays"][path]
        recorded = row["evidence"]["render_stats"]
        if canonical(recorded) != canonical(plan["regenerate"][path]["recorded_render_stats"]):
            raise RecoveryError(f"{path}: recorded stats differ from the plan")
        mask, mident = ctx["masks"].load(fid, shape)
        if mident["mask_sha256"] != row["mask_sha256"]:
            raise RecoveryError(f"{path}: observed mask digest differs from the row")
        drawn = artwork.render(np.asarray(row["pose"], float), np.asarray(mask) > 0)
        regenerated[path] = check_regenerated(recorded, drawn, shape, pixels)
    stages["rerender_check"] = {"stage": "rerender_check", "status": "completed", "frames": sorted(regenerated)}
    # derivative directory: run.json, artwork, copied overlays, regenerated overlays, rows
    target.mkdir(parents=False, exist_ok=False)
    stages["derivative_created"] = {"stage": "derivative_created", "status": "completed", "dir": deriv["dir"]}
    run_json = canonical({"run": deriv["run_spec"], "clip_id": CLIP_ID, "prefix_attestation": deriv["prefix_attestation"],
                          "created_at": time.time()})
    _publish(target / "run.json", run_json, "run.json")
    _publish(target / "neural-artwork.json", artwork_bytes, "neural-artwork.json")
    for role in ROLES:
        (target / "evidence" / role).mkdir(parents=True, exist_ok=False)
    copied_bytes = 0
    _require_room(ws, target, plan["budget"]["copied_png_bytes"], reserve)
    for rel in sorted(k for k in blobs if k.startswith("evidence/")):
        data = blobs[rel]
        check_copied_png(data, shape, ctx["overlays"][rel]["evidence"]["render_stats"]["rendered_pixels"])
        out = _publish(target / rel, data, rel)
        if out["sha256"] != _sealed_sha(ctx["seal"], rel):
            raise RecoveryError(f"{rel}: copied bytes differ from the seal")
        copied_bytes += out["bytes"]
    stages["overlay_copy"] = {"stage": "overlay_copy", "status": "completed", "files": 369, "bytes": copied_bytes}
    _require_room(ws, target, 2 * MAX_REGENERATED_PNG_BYTES, reserve)
    regen_info = []
    for path, rgba in regenerated.items():
        role, fid, pixels = MISSING[path]
        try:
            out = wio.write_png_rgba(target / path, rgba, max_bytes=MAX_REGENERATED_PNG_BYTES)
        except wio.ImageWriteError as exc:
            raise RecoveryError(f"{path}: checked write failed: {exc}") from None
        published = (target / path).read_bytes()
        if not np.array_equal(wio.decode_png_rgba(published, expected_shape=shape), rgba):
            raise RecoveryError(f"{path}: published PNG is not pixel-exact")
        regen_info.append({"path": path, "role": role, "frame_id": fid, "rendered_pixels": pixels,
                           "render_stats": ctx["overlays"][path]["evidence"]["render_stats"],
                           "sha256": out["sha256"], "bytes": out["bytes"],
                           "label": "deterministic CPU reconstruction from the recorded pose, observed mask and frozen "
                                    "artwork; agrees with every recorded render_stats field; the historical missing "
                                    "bytes cannot be proven"})
    stages["overlay_regeneration"] = {"stage": "overlay_regeneration", "status": "completed", "files": len(regen_info)}
    new_rows = derivative_rows(rows, deriv["run_id"])
    results = [wc.FrameResult.from_dict(r) for r in new_rows]
    if any(canonical(fr.to_dict()) != canonical(r) for fr, r in zip(results, new_rows)) \
            or not rows_equal_except_run_id(rows, new_rows):
        raise RecoveryError("derivative rows are not the source rows with only run_id changed")
    results_bytes = b"".join(canonical(r) + b"\n" for r in new_rows)
    _require_room(ws, target, len(results_bytes), reserve)
    _publish(target / "results.jsonl", results_bytes, "results.jsonl")
    stages["rows"] = {"stage": "rows", "status": "completed", "rows": len(new_rows)}
    # verify the derivative before the report
    files = _tree_files(target)
    want = {"run.json", "neural-artwork.json", "results.jsonl", *ctx["overlays"]}
    if set(files) != want:
        raise RecoveryError(f"derivative files differ (extra {sorted(set(files) - want)[:3]}, "
                            f"missing {sorted(want - set(files))[:3]})")
    reread = [loads_strict(line, "derivative row") for line in (target / "results.jsonl").read_bytes().split(b"\n") if line]
    if not rows_equal_except_run_id(rows, reread) or any(r["run_id"] != deriv["run_id"] for r in reread):
        raise RecoveryError("re-read derivative rows differ")
    # closure after: every pinned input, source and runtime unchanged
    after, _ = bind_inputs()
    if canonical(after) != canonical(plan["binding"]):
        raise RecoveryError("closure after recovery differs from the plan")
    stages["closure_after"] = {"stage": "closure_after", "status": "completed"}
    wall_ms = (time.perf_counter() - started) * 1000.0
    src_report = ctx["report"]
    report = wc.build_report(run_spec, ctx["clip"], results, stage_status=stages)
    report["status"] = RECOVERY_STATUS
    report["telemetry"] = {
        "schema": "workbench_artifact_recovery_telemetry_v1", "run_cache_mode": "artifact_recovery",
        "cache": {}, "cached_original_timings": {}, "frames": [], "per_role_frame_timings": {}, "stages": {},
        "run_wall_ms": None, "gpu_used": False, "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
        "process_peak_rss_bytes": None,
        "note": "evidence recovery: no neural frame was executed; the rows' timings are the source run's historical "
                "measurements (see recovery.source_neural_telemetry)"}
    report["dependencies"] = {
        "versions": binding["runtime"], "heads": (src_report.get("dependencies") or {}).get("heads"),
        "sources": binding["exporter_sources"],
        "note": "heads are REUSED from the source run's rows/report (recorded by the parent launcher, which hashed "
                "head/base before that execution); no model was loaded or rehashed by this recovery. sources are the "
                "current exporter modules; the producer modules are recovery.source_neural_dependencies"}
    report["recovery"] = {
        "status": RECOVERY_STATUS, "source_neural_run_id": SOURCE_RUN_ID, "source_report_status": "failed",
        "source_exit_code": 1, "source_role_stages": dict(ctx["seal"]["role_stages"]),
        "reused_saved_predictions": True, "new_neural_frames": 0, "new_neural_execution": False,
        "stage_cache_hit": False, "stage_cache_writes": 0, "gpu_used": False,
        "overlays_copied": 369, "overlays_regenerated": len(regen_info), "regenerated": regen_info,
        "source_neural_dependencies": src_report.get("dependencies"),
        "source_neural_telemetry": src_report.get("telemetry"),
        "source_neural_telemetry_label": "historical timings of the failed source neural run; not this recovery",
        "producer_source_snapshots": binding["producer_source_snapshots"],
        "computational_sources": binding["computational_sources"],
        "exporter_sources": binding["exporter_sources"], "runtime": binding["runtime"],
        "current_cpu_wall_ms": wall_ms,
        "current_cpu_wall_label": "measured wall time of this CPU evidence recovery (not inference latency)",
        "artwork_record_sha256": artwork.sha256, "plan_sha256": plan_sha256, "seal_sha256": SEAL_SHA256,
        "launcher_plan_sha256": LAUNCHER_PLAN_SHA256, "model_verification": binding["model_verification"],
        "claims": CLAIMS}
    data = canonical(report)
    if len(data) > MAX_REPORT_BYTES:
        raise RecoveryError(f"report is {len(data)} bytes, above {MAX_REPORT_BYTES}")
    _require_room(ws, target, len(data), reserve)
    derivative_bytes = _tree_size(target) + len(data)
    if derivative_bytes > MAX_DERIVATIVE_BYTES:
        raise RecoveryError(f"derivative would be {derivative_bytes} bytes, above {MAX_DERIVATIVE_BYTES}")
    summary = {"derivative_run_id": deriv["run_id"], "status": RECOVERY_STATUS, "report_sha256": sha256_bytes(data),
               "overlays_copied": 369, "overlays_regenerated": len(regen_info), "derivative_bytes": derivative_bytes,
               "current_cpu_wall_ms": wall_ms}
    # Report last; publication verifies (and rolls back its own link) itself. Nothing fallible follows.
    _publish(target / "report.json", data, "report.json")
    stages["report_published"] = {"stage": "report_published", "status": "completed"}
    return summary


def _sealed_sha(seal: Mapping, rel: str) -> str:
    for key, value in seal["run_files"].items():
        if _norm(key) == rel:
            return value["sha256"]
    raise RecoveryError(f"{rel} is not sealed")


class RecoveryFailed(RecoveryError):
    """The recovery started and failed; a failure receipt (or stderr record) exists."""


def run(plan_path: Path, expected_sha256: str) -> dict:
    """Returns the success summary, or raises RecoveryFailed after writing a failure receipt.
    Plan errors raise RecoveryError before anything is written."""
    started = time.perf_counter()
    plan = load_plan(plan_path, expected_sha256)
    stages: dict[str, dict] = {}
    attempts: list[str] = []
    try:
        with recovery_guards() as attempts:
            return _recover(plan, expected_sha256, stages, started)
    except BaseException as exc:  # noqa: BLE001 - every failure leaves a reviewable receipt
        _failure_receipt(plan, expected_sha256, stages, attempts, exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RecoveryFailed(f"recovery failed: {type(exc).__name__}: {exc}") from None


def _failure_receipt(plan: Mapping, plan_sha256: str, stages: Mapping, attempts: Sequence[str], exc: BaseException) -> None:
    """Written only into a derivative directory THIS run created; otherwise stderr."""
    from bench import workbench_cli as cli
    target = ROOT / plan["derivative"]["dir"]
    body = {"schema": FAILURE_SCHEMA, "status": "failed_evidence_recovery", "plan_sha256": plan_sha256,
            "derivative_run_id": plan["derivative"]["run_id"], "source_neural_run_id": SOURCE_RUN_ID,
            "reason": f"{type(exc).__name__}: {str(exc)[:1500]}", "stages_completed": sorted(stages),
            "guard_attempts": list(attempts)[:20], "report_published": "report_published" in stages,
            "note": "partial derivative preserved; no completed status was promoted"}
    data = canonical(body)
    try:
        if "derivative_created" not in stages or len(data) > MAX_RECEIPT_BYTES:
            raise OSError("no derivative directory created by this run")
        cli.Workspace(ROOT / WORKSPACE).require_budget(extra_derived=len(data))
        _publish(target / "recovery-failure.json", data, "failure receipt")
    except Exception:  # noqa: BLE001 - fall back to the terminal record
        sys.stderr.write("failure receipt not persisted; terminal record:\n" + data.decode("ascii") + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 success; 1 the recovery started and failed (receipt in the derivative directory if this run
    created it, otherwise a stderr record); 2 plan/usage/prepare error before any recovery write."""
    parser = argparse.ArgumentParser(description="Sealed CPU-only evidence recovery of the failed mug run.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="read-only verification; writes one fresh plan")
    s.add_argument("--out", required=True)
    s = sub.add_parser("run", help="CPU recovery into the derivative run frozen in the plan")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = write_plan(Path(args.out))
        else:
            result = run(Path(args.plan), args.expected_plan_sha256)
    except RecoveryFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (Exception, RecoveryViolation) as exc:  # noqa: BLE001 - typed, nothing promoted
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
