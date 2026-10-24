"""Loopback-only HTTP workbench service for registered clips and runs.

Run::

    python -m bench.workbench_service --workspace .cache/video-workbench --port 8765

The service binds 127.0.0.1/::1 only, serves a fixed static page, and exposes
clips registered in the workbench workspace (``bench.workbench_cli``). Request
paths never become filesystem paths: media, frames and evidence resolve only
through registered roots, clip/frame/run identifiers and validated result
evidence. No URL is fetched, no code is executed from requests. One job runs
at a time through the real pipeline; missing prerequisites produce a blocked
job whose results still contain one unavailable row per scheduled frame.

Prepared metric CAD controls are configured only by the operator, either per
clip (``--metric-bundle-for CLIP_ID=PATH``, ``--masks-for CLIP_ID=SET``) or
with a legacy global bundle that is explicitly bound to one clip
(``--metric-bundle PATH --metric-bundle-clip CLIP_ID``). An explicit per-clip
mapping wins. A bound global bundle applies only to that exact clip ID, and
the size, seed and hash checks still apply. An unbound global bundle is never
used for any clip, because clip identity is never inferred from frame size or
seed frame. ``--masks`` remains a default only for clips that contain that set. Each bundle file's SHA-256 is bound at startup and
rehashed before every job. A changed, missing, invalid or incompatible bundle
for the requested clip blocks the neural roles explicitly and never falls back
to another clip's geometry. API callers cannot name a bundle path; only IDs,
statuses and hashes are returned.

Supplemental execution-only YCB CAD controls are configured with
``--execution-control-for CLIP_ID=DIRECTORY`` (repeatable; the directory is an
emitted ``tools/workbench_ycb_execution_control.py`` output). The pinned
reviewed adapter is imported lazily and only when such a mapping exists. Its
``validate_output`` launch gate re-derives the whole closure at startup, at run
request, at job start and immediately before the pipeline. The control must
also bind exactly to the clip: object, canonical id, inference manifest,
scene, 640x480 family, manifest K and the seed frame's RGB. The validated
identity (bundle, GLB, sidecar, receipt, adapter and map hashes) is fixed at
startup and written into the run config. Any difference blocks the neural roles
with no fallback. A clip with both an execution control and a metric bundle is
refused at startup.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from . import workbench_contract as wc

MAX_JSON_BYTES = 1024 * 1024
MAX_UPLOAD_BYTES = 250_000_000          # approved decimal cap (not MiB)
MAX_IMPORT_TOTAL_BYTES = 250_000_000    # aggregate across stored uploads
MAX_RANGE_BYTES = 8 * 1024 * 1024
ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
HEX32 = re.compile(r"^[0-9a-f]{32}$")
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/workbench.css": ("workbench.css", "text/css; charset=utf-8"),
    "/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
}
JOB_STATES = ("queued", "running", "blocked", "failed", "completed", "cancelled")
TERMINAL = ("blocked", "failed", "completed", "cancelled")


class WorkbenchError(ValueError):
    """A request does not fit the local workbench contract."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _safe_member(root: Path, relative: Any, *, require_file: bool = True) -> Path:
    """Resolve an explicit registered relative member without traversing symlinks."""
    if (not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative
            or ":" in relative or relative.startswith("/")):
        raise WorkbenchError("Registered path must be a safe relative path.")
    member = PurePosixPath(relative)
    if member.is_absolute() or any(part in ("", ".", "..") for part in member.parts):
        raise WorkbenchError("Registered path escapes its configured root.")
    root = Path(root).resolve()
    current = root
    for part in member.parts:
        current = current / part
        if current.is_symlink():
            raise WorkbenchError("Registered paths cannot traverse symlinks.")
    resolved = current.resolve(strict=False)
    if not _inside(root, resolved):
        raise WorkbenchError("Registered path escapes its configured root.")
    if require_file and not resolved.is_file():
        raise WorkbenchError("Registered media or evidence file is unavailable.")
    return resolved


def safe_public_frame(row: Any) -> dict:
    """Re-validate a result through the FrameResult contract before publishing.

    A pose survives only with tracking state and a fresh observed mask for the
    same frame; rendered state requires that pose. Invalid rows raise.
    """
    parsed = wc.FrameResult.from_dict(row)
    out = parsed.to_dict()
    evidence = out.get("evidence") or {}
    out["evidence"] = {k: evidence[k] for k in ("overlay", "render_stats", "mask_file_sha256", "render",
                                                 "initialization", "limits") if k in evidence}
    return out


NEURAL_ROLES = ("primary_neural", "frozen_baseline")
DEFAULT_ROLES = ("primary_neural", "frozen_baseline", "relative_geometry")
CONTROL_READY = "prepared_metric_cad_control"
CONTROL_LABEL_READY = "prepared metric CAD control"
CONTROL_LABEL_UNAVAILABLE = "unavailable; automatic preparation is a separate experimental path"
_FAMILY_SIZE = re.compile(r"_(\d{2,5})x(\d{2,5})$")

# Supplemental execution-only YCB CAD controls (``--execution-control-for``).
# The reviewed adapter and its fixed reviewed map are pinned; the adapter is
# imported lazily, only when such a control is configured.
REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_ADAPTER_REL = "tools/workbench_ycb_execution_control.py"
EXECUTION_ADAPTER_SHA256 = "57a08ff834ceac404f3adbeed1561d1bb29dadfd23d3486caad75b2676f6d32b"
EXECUTION_MAP_REL = "tools/workbench_ycb_execution_controls.reviewed.json"
EXECUTION_MAP_SHA256 = "c5ae3b009007ac1600af345c56f1b719e8881591baf37187a917d361eb99ca20"
EXECUTION_CONTROL_FILES = ("model.glb", "metric-bundle.json", "execution-control-sidecar.json", "emission-receipt.json")
EXECUTION_CONTROL_KIND = "execution_only_supplemental_cad"
EXECUTION_CONTROL_READY = "execution_only_metric_cad_control"
EXECUTION_CONTROL_LABEL = "execution-only known metric CAD control (supplemental YCB)"
EXECUTION_LABELS = {
    "control": "execution-only known metric CAD (dataset CAD + final onboarding pose seed)",
    "placement": "parent qualitative placement reviewed on both opening frames (not independent annotation)",
    "correspondence": "independent correspondence / attachment accuracy unverified",
    "uv": "raw-PLY vs loaded-GLTF GPU relative convention passed on both opening frames; full-UV diagnostic unchanged",
    "automatic_rgb_geometry": "automatic RGB geometry preparation is a separate experiment (not used)",
    "admission": "640x480 neural serving admission is still required and is not satisfied or altered by this control",
    "primary_model": "main head 27e06a51… and the frozen baseline are unchanged",
    "principal_point_origin": "integer_centers (assumed provenance)",
}
YCB_DATASET = "BOP YCB-Video"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _bind_bundle(value: Any, label: str) -> dict:
    """Resolve an operator-configured local bundle file and bind its SHA-256."""
    if not isinstance(value, (str, os.PathLike)) or not str(value) or "://" in str(value):
        raise WorkbenchError(f"{label}: metric bundle must be a local file path")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise WorkbenchError(f"{label}: metric bundle is not a regular file")
    sha, size = _sha256_file(path)
    return {"path": path, "sha256": sha, "bytes": size}


def _execution_adapter():
    """Pinned reviewed adapter (lazy import). The adapter file and its fixed
    reviewed map must equal their reviewed SHA-256 before every use."""
    for rel, pin in ((EXECUTION_ADAPTER_REL, EXECUTION_ADAPTER_SHA256), (EXECUTION_MAP_REL, EXECUTION_MAP_SHA256)):
        path = REPO_ROOT / rel
        if path.is_symlink() or not path.is_file():
            raise WorkbenchError(f"execution-control file {rel} is missing")
        if _sha256_file(path)[0] != pin:
            raise WorkbenchError(f"execution-control file {rel} differs from its reviewed SHA-256")
    import importlib
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    module = importlib.import_module("tools.workbench_ycb_execution_control")
    if (Path(module.__file__).resolve() != (REPO_ROOT / EXECUTION_ADAPTER_REL).resolve()
            or Path(module.MAP_PATH).resolve() != (REPO_ROOT / EXECUTION_MAP_REL).resolve()):
        raise WorkbenchError("imported execution-control adapter is not the pinned project file")
    return module


def _control_directory(value: Any, clip_id: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value) or "://" in str(value):
        raise WorkbenchError(f"execution control for clip {clip_id} must be a local directory")
    path = Path(os.path.abspath(Path(value).expanduser()))
    if path.is_symlink() or not path.is_dir():
        raise WorkbenchError(f"execution control for clip {clip_id} is not a real directory")
    return path


def parse_clip_mapping(pairs: Any, kind: str) -> dict[str, str]:
    """``CLIP_ID=VALUE`` strings, (clip_id, value) pairs or a dict; duplicates are rejected."""
    if pairs is None:
        return {}
    items = list(pairs.items()) if isinstance(pairs, dict) else list(pairs)
    out: dict[str, str] = {}
    for item in items:
        if isinstance(item, str):
            clip_id, sep, value = item.partition("=")
            if not sep:
                raise WorkbenchError(f"{kind} mappings use CLIP_ID=VALUE")
        else:
            clip_id, value = item
        try:
            clip_id = wc._ident(clip_id, "clip_id")
        except wc.ContractError as exc:
            raise WorkbenchError(f"{kind} mapping has an invalid clip id: {exc}") from exc
        if not isinstance(value, str) or not value:
            raise WorkbenchError(f"{kind} mapping for clip {clip_id} needs a value")
        if clip_id in out:
            raise WorkbenchError(f"duplicate {kind} mapping for clip {clip_id}")
        out[clip_id] = value
    return out


def _clip_summary(clip: wc.ClipSpec) -> dict:
    return {"clip_id": clip.clip_id, "title": clip.title, "source_kind": clip.source_kind,
            "clock_kind": clip.clock_kind, "width": clip.width, "height": clip.height,
            "frame_count": len(clip.frames), "scoring_frames": len(clip.scoring_frame_ids),
            "intrinsics": clip.intrinsics.status, "evidence_role": clip.evidence_role,
            "selection_ready": bool(clip.onboarding and clip.onboarding.selection)}


class WorkbenchService:
    """Workspace-backed API with a single serialized job runner."""

    def __init__(self, workspace: str | Path, *, roots: dict[str, str | Path] | None = None,
                 prepare_adapter: Callable[..., dict] | None = None, use_pipeline: bool = True,
                 run_options: dict | None = None):
        from . import workbench_cli as cli
        self.cli = cli
        path = Path(workspace).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.ws = cli.Workspace(path.resolve(strict=True))
        self.ws.init({k: str(v) for k, v in (roots or {}).items()})
        self.web_root = Path(__file__).resolve().parent / "web" / "workbench"
        self.service_root = self.ws.path / "service"
        self.import_root = self.ws.imports_dir
        self.jobs_root = self.service_root / "jobs"
        self.requests_root = self.service_root / "requests"
        self.selections_root = self.service_root / "selections"
        for directory in (self.service_root, self.import_root, self.jobs_root, self.requests_root,
                          self.selections_root):
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink() or not _inside(self.ws.path, directory.resolve()):
                raise WorkbenchError("Workbench state must remain inside its workspace.")
        self.prepare_adapter = prepare_adapter
        self.use_pipeline = use_pipeline
        self.run_options = dict(run_options or {})
        self._mask_map = {}
        for clip_id, mask_set in parse_clip_mapping(self.run_options.get("masks_by_clip"), "mask set").items():
            try:
                self._mask_map[clip_id] = wc._ident(mask_set, "mask_set")
            except wc.ContractError as exc:
                raise WorkbenchError(f"mask set mapping for clip {clip_id} is invalid: {exc}") from exc
        self._bundle_map = {}
        for clip_id, value in parse_clip_mapping(self.run_options.get("metric_bundles_by_clip"),
                                                 "metric bundle").items():
            binding = _bind_bundle(value, f"metric bundle for clip {clip_id}")
            from . import workbench_neural as wn
            try:
                wn.MetricBundle.from_file(binding["path"])
            except Exception as exc:  # noqa: BLE001 - strict at startup; operator must fix the mapping
                raise WorkbenchError(f"metric bundle for clip {clip_id} is invalid: {type(exc).__name__}: {exc}") from exc
            self._bundle_map[clip_id] = binding
        self._global_bundle = None
        self._global_clip = None
        if self.run_options.get("metric_bundle_clip") is not None:
            try:
                self._global_clip = wc._ident(self.run_options["metric_bundle_clip"], "metric_bundle_clip")
            except wc.ContractError as exc:
                raise WorkbenchError(f"--metric-bundle-clip is invalid: {exc}") from exc
            if not self.run_options.get("metric_bundle"):
                raise WorkbenchError("--metric-bundle-clip requires --metric-bundle")
        if self.run_options.get("metric_bundle"):
            try:
                self._global_bundle = _bind_bundle(self.run_options["metric_bundle"], "global metric bundle")
            except (OSError, WorkbenchError) as exc:
                # Backward compatible: report as blocked per clip instead of refusing to start.
                self._global_bundle = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        # Supplemental execution-only controls: fully validated at startup; the
        # bound identity is immutable for the life of the service.
        self._exec_map: dict[str, dict] = {}
        exec_pairs = parse_clip_mapping(self.run_options.get("execution_controls_by_clip"), "execution control")
        for clip_id, value in exec_pairs.items():
            if clip_id in self._bundle_map or (self._global_bundle is not None and self._global_clip == clip_id):
                raise WorkbenchError(f"clip {clip_id} has conflicting explicit geometry mappings "
                                     "(--execution-control-for and a metric bundle); configure exactly one")
            directory = _control_directory(value, clip_id)
            try:
                clip = self.ws.load_clip(clip_id)
            except (self.cli.WorkbenchCLIError, wc.ContractError) as exc:
                raise WorkbenchError(f"execution control for clip {clip_id}: clip is not registered") from exc
            identity = self._validate_execution_control(clip, directory)
            self._exec_map[clip_id] = {"directory": directory, "identity": identity,
                                       "identity_sha256": wc.canonical_sha256(identity), "validated_at": time.time()}
        self._state_lock = threading.RLock()
        self._job_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._active_job: str | None = None
        self._cancel: threading.Event | None = None
        self._jobs: dict[str, dict] = {}
        self._artwork_replayer = None   # RAM-only CPU artwork replay (created on first request; no cache)
        self._recover_jobs()

    # ---------------------------------------------------------------- persistence
    def _write_new(self, path: Path, value: Any) -> None:
        with path.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())

    def _write_mutable(self, path: Path, value: Any) -> None:
        tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        with tmp.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)

    def _recover_jobs(self) -> None:
        for path in self.jobs_root.glob("*.json"):
            try:
                row = wc.loads_strict(path.read_bytes())
            except (OSError, ValueError):
                continue
            if row.get("status") in ("queued", "running"):
                row["status"] = "failed"
                row["error"] = "Service restarted before the job reached a terminal state."
                row["updated_at"] = time.time()
                self._write_mutable(path, row)
            if isinstance(row.get("job_id"), str):
                self._jobs[row["job_id"]] = row

    # ---------------------------------------------------------------- status/clips
    def _admission_status(self) -> dict:
        """Re-verify the configured admission directory on every call (read-only,
        no GPU/model loads). Unreadable or stale evidence stays blocked."""
        configured = self.run_options.get("admission")
        if not configured:
            return {"configured": False, "admitted": False, "reasons": ["no admission directory configured"]}
        from . import workbench_neural as wn
        try:
            admission = wn.load_admission(Path(configured))
        except Exception as exc:  # noqa: BLE001 - report honestly, never admit
            return {"configured": True, "admitted": False,
                    "reasons": [f"admission unreadable: {type(exc).__name__}: {str(exc)[:300]}"]}
        body = admission.body
        return {"configured": True, "admitted": bool(admission.admitted), "reasons": list(admission.reasons)[:8],
                "profile": body.get("profile"), "families": body.get("families"),
                "contract_sha256": body.get("contract_sha256"),
                "scope": body.get("scope", "experimental execution compatibility only; never model quality"),
                "quality_claim": False}

    def _runtime_device(self) -> dict:
        """Configured device and CUDA build/env metadata without importing torch."""
        import importlib.metadata
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        cuda_build = None
        if torch_version and "+cu" in torch_version:
            cuda_build = torch_version.split("+", 1)[1]
        elif torch_version and "+cpu" in torch_version:
            cuda_build = "cpu"
        return {"configured_device": self.run_options.get("device") or "cpu", "torch_version": torch_version,
                "torch_build": cuda_build, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "note": "metadata only; CUDA is not initialized by status"}

    def status(self) -> dict:
        from . import workbench_reconstruction as wr
        with self._state_lock:
            active = self._jobs.get(self._active_job) if self._active_job else None
        colmap = wr.preflight(self.run_options.get("colmap"))
        from . import workbench_neural as wn
        seg = wn.segmentation_preflight()
        admission = self._admission_status()
        neural = wn.neural_preflight(head_path=Path(self.run_options["head"]) if self.run_options.get("head") else None,
                                     admission_path=Path(self.run_options["admission"]) if admission["admitted"] else None)
        blocked = list(neural["blocked_reasons"])
        if admission["admitted"]:
            # The serving admission was freshly re-verified above; the remaining
            # preflight items (weights, sources, runtime) still apply.
            blocked = [b for b in blocked if b not in ("serving_admission", "reference_best_checkpoint")]
        return {
            "service": "visualizeit-workbench", "schema": "workbench_service_v2", "local_only": True,
            "registered_clip_count": len(self.ws.list_clips()),
            "import_limit_bytes": MAX_IMPORT_TOTAL_BYTES, "budget": self.ws.budget(),
            "capabilities": {"pipeline": self.use_pipeline or self.prepare_adapter is not None,
                             "colmap": colmap.get("available"), "colmap_reason": colmap.get("reason"),
                             "segmentation": seg["ready"], "segmentation_blocked": seg["blocked_reasons"],
                             "neural_blocked": blocked,
                             "admission_dir": bool(self.run_options.get("admission"))},
            "admission": admission,
            "runtime_device": self._runtime_device(),
            "defaults": {"mask_set": self.run_options.get("mask_set"),
                         "metric_bundle_configured": bool(self.run_options.get("metric_bundle")),
                         "head_configured": bool(self.run_options.get("head")),
                         "masks_by_clip": dict(self._mask_map),
                         "execution_controls_by_clip": sorted(self._exec_map)},
            "metric_controls": self._control_overview(),
            "primary_model": neural["primary"],
            "active_job": self._public_job(active) if active else None,
        }

    # ---------------------------------------------------------------- prepared controls
    def _bundle_for_clip(self, clip_id: str) -> tuple[dict | None, str | None]:
        """Explicit identity only: a per-clip mapping, or the global bundle bound to this exact clip ID."""
        if clip_id in self._bundle_map:
            return self._bundle_map[clip_id], "per_clip"
        if self._global_bundle is not None and self._global_clip == clip_id:
            return self._global_bundle, "global_bound"
        return None, None

    @staticmethod
    def _control_mismatches(bundle: Any, clip: wc.ClipSpec, selection_frame_id: str | None) -> list[str]:
        issues = []
        if clip.intrinsics.status == "unavailable":
            issues.append("clip calibration is unavailable")
        size = _FAMILY_SIZE.search(str(bundle.family))
        if size is None:
            issues.append(f"bundle family {bundle.family} declares no frame size")
        elif (int(size.group(1)), int(size.group(2))) != (clip.width, clip.height):
            issues.append(f"bundle family {bundle.family} does not match clip size {clip.width}x{clip.height}")
        if bundle.seed_pose_m is None:
            issues.append("bundle has no labelled seed pose")
        if bundle.seed_frame_id not in {f.frame_id for f in clip.frames}:
            issues.append(f"bundle seed frame {bundle.seed_frame_id} is not a frame of clip {clip.clip_id}")
        elif selection_frame_id is not None and bundle.seed_frame_id != selection_frame_id:
            issues.append(f"bundle seed frame {bundle.seed_frame_id} differs from the reviewed selection frame "
                          f"{selection_frame_id}")
        return issues

    def _resolve_control(self, clip: wc.ClipSpec, selection_frame_id: str | None, *,
                         full: bool = False) -> tuple[dict, dict | None]:
        """Public readiness summary plus the private binding (path never leaves the service)."""
        if clip.clip_id in self._exec_map:
            return self._resolve_execution_control(clip, selection_frame_id, full=full)
        binding, source = self._bundle_for_clip(clip.clip_id)
        out = {"clip_id": clip.clip_id, "source": source, "status": "unavailable", "ready": False,
               "label": CONTROL_LABEL_UNAVAILABLE, "reasons": [], "bundle_sha256": None, "glb_sha256": None,
               "family": None, "obj_id": None, "principal_point_origin": None, "seed_frame_id": None,
               "metric_provenance": None,
               "principal_point_origin_assumption": ("declared by the bundle; the clip record stores no origin, "
                                                     "so independent calibration of the origin is not validated"),
               "note": "Readiness only; serving admission and the GLB hash are re-verified by the job."}
        if binding is None:
            if self._global_bundle is not None and self._global_clip is None:
                out["reasons"] = ["the global --metric-bundle has no clip binding and is never applied by inference; "
                                  "configure --metric-bundle-for CLIP=PATH or bind the legacy bundle with "
                                  "--metric-bundle-clip CLIP_ID"]
            else:
                out["reasons"] = ["no prepared metric CAD control is configured for this clip"]
            return out, None
        if binding.get("error"):
            out.update(status="blocked", reasons=[f"configured metric bundle unavailable: {binding['error']}"])
            return out, None
        out["bundle_sha256"] = binding["sha256"]
        problems: list[str] = []
        try:
            if _sha256_file(binding["path"])[0] != binding["sha256"]:
                problems.append("configured metric bundle changed since the service was configured")
        except OSError:
            problems.append("configured metric bundle file is missing or unreadable")
        mismatches: list[str] = []
        if not problems:
            try:
                from . import workbench_neural as wn
                bundle = wn.MetricBundle.from_file(binding["path"])
            except Exception as exc:  # noqa: BLE001 - invalid content never becomes ready
                problems.append(f"metric bundle invalid: {type(exc).__name__}: {str(exc)[:200]}")
            else:
                out.update(glb_sha256=bundle.glb_sha256, family=bundle.family, obj_id=bundle.obj_id,
                           principal_point_origin=bundle.principal_point_origin,
                           seed_frame_id=bundle.seed_frame_id, metric_provenance=bundle.metric_provenance)
                mismatches = self._control_mismatches(bundle, clip, selection_frame_id)
        if problems:
            out.update(status="blocked", reasons=problems)
            return out, None
        if mismatches:
            # Every binding is explicit (per clip or bound global), so a misfit is an operator error.
            out.update(status="blocked", reasons=mismatches)
            return out, None
        if selection_frame_id is None:
            out.update(status="pending_selection",
                       reasons=["save a reviewed selection; the seed frame must equal its selection frame"])
            return out, None
        out.update(status=CONTROL_READY, ready=True, label=CONTROL_LABEL_READY)
        return out, binding

    # ---------------------------------------------------------------- execution-only controls
    def _execution_binding_issues(self, xc: Any, clip: wc.ClipSpec, result: dict, bundle: Any) -> list[str]:
        """The validated control must belong to exactly this clip: object, dataset
        manifest, scene, size, calibration and seed frame (RGB) all agree."""
        issues = self._control_mismatches(bundle, clip, None)
        yc = xc.yc
        object_id = result.get("object_id")
        prov = clip.provenance if isinstance(clip.provenance, dict) else {}
        if prov.get("dataset") != YCB_DATASET:
            issues.append(f"clip {clip.clip_id} is not a registered {YCB_DATASET} clip")
        if type(object_id) is not int or prov.get("object_id") != object_id or bundle.obj_id != object_id:
            issues.append(f"control object {object_id} does not match clip object {prov.get('object_id')}")
            return issues
        if prov.get("canonical_id") != yc.OBJECTS[object_id][0]:
            issues.append("clip canonical YCB id differs from the control object")
        try:
            manifest, manifest_sha = yc.load_manifest(yc.DEFAULT_MANIFEST)
            obj = yc.select_object(manifest, object_id)
        except Exception as exc:  # noqa: BLE001 - fail closed
            issues.append(f"authenticated inference manifest unavailable: {type(exc).__name__}: {str(exc)[:160]}")
            return issues
        if prov.get("inference_manifest_sha256") != manifest_sha:
            issues.append("clip was registered from another inference manifest")
        if str(obj.get("sceneId")).zfill(6) != str(prov.get("scene_id")).zfill(6):
            issues.append("clip scene differs from the control object's scene")
        rows = {r.get("logicalIndex"): r for r in obj.get("onboarding", []) if isinstance(r, dict)}
        final = rows.get(xc.FINAL_ONBOARDING_LOGICAL)
        if not isinstance(final, dict) or type(final.get("frameId")) is not int:
            issues.append("final onboarding row unavailable in the manifest")
            return issues
        seed_id = f"s{final['frameId']:06d}"
        if bundle.seed_frame_id != seed_id or result.get("seed_frame_id") != seed_id:
            issues.append(f"control seed {bundle.seed_frame_id} is not the final onboarding frame {seed_id}")
        if seed_id not in (prov.get("onboarding_frame_ids") or []):
            issues.append(f"seed frame {seed_id} is not an onboarding frame of clip {clip.clip_id}")
        try:
            frame = clip.frame(seed_id)
        except KeyError:
            issues.append(f"seed frame {seed_id} is not a frame of clip {clip.clip_id}")
        else:
            if frame.source_frame_index != final["frameId"]:
                issues.append("seed frame source index differs from the manifest")
            if frame.image_sha256 != (final.get("rgb") or {}).get("sha256"):
                issues.append("seed frame RGB differs from the authenticated manifest RGB")
        cam_k = (final.get("calibration") or {}).get("cam_K")
        clip_k = None if clip.intrinsics.K is None else [float(v) for v in clip.intrinsics.K]
        if not isinstance(cam_k, list) or clip_k != [float(v) for v in cam_k]:
            issues.append("clip calibration differs from the manifest's final onboarding K")
        return issues

    def _validate_execution_control(self, clip: wc.ClipSpec, directory: Path) -> dict:
        """Full launch-gate validation (re-derives evidence/runtime/map/qualitative
        closure) plus exact clip binding. Returns the public identity."""
        xc = _execution_adapter()
        try:
            result = xc.validate_output(directory)
        except Exception as exc:  # noqa: BLE001 - every failure blocks
            raise WorkbenchError(f"execution control for clip {clip.clip_id} failed validation: "
                                 f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        files = {}
        for name in EXECUTION_CONTROL_FILES:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise WorkbenchError(f"execution control file {name} is missing")
            files[name] = _sha256_file(path)[0]
        if (result.get("status") != "valid_execution_only_control"
                or files["metric-bundle.json"] != result.get("bundle_sha256")
                or files["model.glb"] != result.get("glb_sha256")):
            raise WorkbenchError(f"execution control for clip {clip.clip_id} does not match its validation result")
        from . import workbench_neural as wn
        try:
            bundle = wn.MetricBundle.from_file(directory / "metric-bundle.json")
        except Exception as exc:  # noqa: BLE001
            raise WorkbenchError(f"execution control bundle unreadable: {type(exc).__name__}: {exc}") from exc
        if bundle.glb_sha256 != files["model.glb"] or Path(bundle.glb_path).resolve() != (directory / "model.glb").resolve():
            raise WorkbenchError("execution control bundle does not name its own validated model.glb")
        issues = self._execution_binding_issues(xc, clip, result, bundle)
        if issues:
            raise WorkbenchError(f"execution control does not belong to clip {clip.clip_id}: {'; '.join(issues)}")
        for name in EXECUTION_CONTROL_FILES:
            if _sha256_file(directory / name)[0] != files[name]:
                raise WorkbenchError(f"execution control file {name} changed during validation")
        return {"kind": EXECUTION_CONTROL_KIND, "clip_id": clip.clip_id, "object_id": result["object_id"],
                "canonical_ycb_id": xc.yc.OBJECTS[result["object_id"]][0],
                "bundle_sha256": files["metric-bundle.json"], "glb_sha256": files["model.glb"],
                "sidecar_sha256": files["execution-control-sidecar.json"],
                "receipt_sha256": files["emission-receipt.json"],
                "adapter_sha256": EXECUTION_ADAPTER_SHA256, "map_sha256": EXECUTION_MAP_SHA256,
                "family": bundle.family, "obj_id": bundle.obj_id, "seed_frame_id": bundle.seed_frame_id,
                "metric_provenance": bundle.metric_provenance,
                "principal_point_origin": bundle.principal_point_origin, "flags": dict(result.get("flags") or {})}

    def _resolve_execution_control(self, clip: wc.ClipSpec, selection_frame_id: str | None, *,
                                   full: bool) -> tuple[dict, dict | None]:
        entry = self._exec_map[clip.clip_id]
        ident = entry["identity"]
        out = {"clip_id": clip.clip_id, "source": "execution_control", "control_kind": EXECUTION_CONTROL_KIND,
               "status": "blocked", "ready": False, "label": EXECUTION_CONTROL_LABEL, "labels": dict(EXECUTION_LABELS),
               "reasons": [], **{k: ident[k] for k in ("bundle_sha256", "glb_sha256", "sidecar_sha256", "receipt_sha256",
                                                       "adapter_sha256", "map_sha256", "family", "obj_id",
                                                       "seed_frame_id", "metric_provenance", "principal_point_origin",
                                                       "object_id", "canonical_ycb_id")},
               "identity_sha256": entry["identity_sha256"],
               "validation": "full re-validation now" if full else
                             "files unchanged since startup validation (full re-validation at request, job start "
                             "and immediately before the pipeline)",
               "note": "Execution-only. Serving admission and the main head are re-verified by the job."}
        problems: list[str] = []
        try:
            for rel, pin in ((EXECUTION_ADAPTER_REL, EXECUTION_ADAPTER_SHA256), (EXECUTION_MAP_REL, EXECUTION_MAP_SHA256)):
                if _sha256_file(REPO_ROOT / rel)[0] != pin:
                    problems.append(f"{rel} changed since startup")
            for name, key in (("model.glb", "glb_sha256"), ("metric-bundle.json", "bundle_sha256"),
                              ("execution-control-sidecar.json", "sidecar_sha256"),
                              ("emission-receipt.json", "receipt_sha256")):
                if _sha256_file(entry["directory"] / name)[0] != ident[key]:
                    problems.append(f"control file {name} changed since startup")
        except OSError:
            problems.append("execution control files are missing or unreadable")
        if full and not problems:
            try:
                current = self._validate_execution_control(clip, entry["directory"])
            except WorkbenchError as exc:
                problems.append(str(exc))
            else:
                if wc.canonical_sha256(current) != entry["identity_sha256"]:
                    problems.append("execution control identity differs from the startup identity")
        if problems:
            out["reasons"] = problems
            return out, None
        if selection_frame_id is None:
            out.update(status="pending_selection",
                       reasons=["save a reviewed selection; the seed frame must equal its selection frame"])
            return out, None
        if selection_frame_id != ident["seed_frame_id"]:
            out["reasons"] = [f"control seed frame {ident['seed_frame_id']} differs from the reviewed selection "
                              f"frame {selection_frame_id}"]
            return out, None
        out.update(status=EXECUTION_CONTROL_READY, ready=True)
        binding = {"path": entry["directory"] / "metric-bundle.json", "sha256": ident["bundle_sha256"],
                   "kind": EXECUTION_CONTROL_KIND, "identity_sha256": entry["identity_sha256"],
                   "run_config": {**ident, "identity_sha256": entry["identity_sha256"]}}
        return out, binding

    def _execution_files_unchanged(self, clip_id: str) -> bool:
        entry = self._exec_map[clip_id]
        ident = entry["identity"]
        try:
            return all(_sha256_file(entry["directory"] / name)[0] == ident[key] for name, key in (
                ("model.glb", "glb_sha256"), ("metric-bundle.json", "bundle_sha256"),
                ("execution-control-sidecar.json", "sidecar_sha256"), ("emission-receipt.json", "receipt_sha256")))
        except OSError:
            return False

    def _mask_sets(self, clip_id: str) -> list[str]:
        root = self.ws.masks_dir / clip_id
        if not root.is_dir():
            return []
        return sorted(m.parent.name for m in root.glob("*/manifest.json") if wc.ID_RE.fullmatch(m.parent.name))

    def _default_masks(self, clip_id: str) -> dict:
        available = self._mask_sets(clip_id)
        if clip_id in self._mask_map:
            name = self._mask_map[clip_id]
            return {"default": name, "source": "per_clip", "present": name in available, "available": available}
        configured = self.run_options.get("mask_set")
        if configured and configured in available:
            return {"default": configured, "source": "global", "present": True, "available": available}
        return {"default": None, "source": None, "present": False, "available": available}

    def _control_overview(self) -> dict:
        clips = {}
        for clip_id in sorted(set(self._bundle_map) | set(self._mask_map) | set(self._exec_map)):
            try:
                clip = self.ws.load_clip(clip_id)
            except (self.cli.WorkbenchCLIError, wc.ContractError):
                clips[clip_id] = {"status": "unavailable", "reasons": ["clip is not registered"]}
                continue
            selection = clip.onboarding.selection.frame_id if clip.onboarding and clip.onboarding.selection else None
            control, _ = self._resolve_control(clip, selection)
            clips[clip_id] = {**{k: control[k] for k in ("status", "ready", "label", "source", "bundle_sha256",
                                                         "glb_sha256", "seed_frame_id", "reasons")},
                              **{k: control[k] for k in ("control_kind", "labels", "sidecar_sha256", "identity_sha256")
                                 if k in control},
                              "default_mask_set": self._default_masks(clip_id)["default"]}
        glob = self._global_bundle
        return {"per_clip": clips,
                "global": {"configured": glob is not None, "bound_clip_id": self._global_clip,
                           "bundle_sha256": glob.get("sha256") if glob else None,
                           "error": glob.get("error") if glob else None,
                           "applies_to": ("only the bound clip ID (when it has no per-clip mapping)" if self._global_clip
                                          else "no clip: unbound global bundles are never applied")}}

    def list_clips(self) -> dict:
        clips = [_clip_summary(c) for c in self.ws.list_clips()]
        imports = []
        for meta in sorted(self.import_root.glob("*.meta.json")):
            try:
                row = wc.loads_strict(meta.read_bytes())
                imports.append({k: row.get(k) for k in ("source_id", "original_name", "bytes", "media_type",
                                                        "registration_status", "clip_id", "blocked_reason")})
            except (OSError, ValueError):
                continue
        return {"clips": clips, "imports": imports}

    def clip(self, clip_id: str) -> wc.ClipSpec:
        try:
            return self.ws.load_clip(clip_id)
        except (self.cli.WorkbenchCLIError, wc.ContractError) as exc:
            raise KeyError(clip_id) from exc

    def clip_detail(self, clip_id: str) -> dict:
        clip = self.clip(clip_id)
        body = clip.to_dict()
        body.pop("provenance", None)
        media_url = None
        if clip.source_kind == "video_file" and Path(clip.media.path).suffix.lower() in VIDEO_TYPES:
            media_url = f"/api/clips/{clip.clip_id}/media"
        selection = clip.onboarding.selection.frame_id if clip.onboarding and clip.onboarding.selection else None
        control, _ = self._resolve_control(clip, selection)
        return {"clip": body, "summary": _clip_summary(clip), "media_url": media_url,
                "frame_url_template": f"/api/clips/{clip.clip_id}/frames/{{frame_id}}/image",
                "runs": self.list_runs(clip_id)["runs"],
                "prepared_control": control, "masks": self._default_masks(clip.clip_id)}

    def media_path(self, clip_id: str) -> tuple[Path, str]:
        clip = self.clip(clip_id)
        if clip.source_kind != "video_file":
            raise KeyError(clip_id)
        path = self.ws.media_path(clip)
        media_type = VIDEO_TYPES.get(path.suffix.lower())
        if media_type is None or not path.is_file():
            raise KeyError(clip_id)
        return path, media_type

    def frame_image(self, clip_id: str, frame_id: str) -> tuple[bytes, str]:
        """Exact source frame by frame ID: the source PNG/JPEG for image sequences,
        or a decoded-frame PNG at the exact source_frame_index for videos."""
        clip = self.clip(clip_id)
        try:
            frame = clip.frame(frame_id)
        except KeyError:
            raise KeyError(frame_id)
        if clip.source_kind == "image_sequence":
            root = self.ws.media_path(clip)
            path = _safe_member(root, frame.image_path)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != frame.image_sha256:
                raise WorkbenchError("Source image changed after registration.")
            return raw, "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        from .vision import cv2
        with self._frame_lock:
            reader = self.cli.FrameReader(self.ws, clip)
            try:
                rgb, _ident = reader.read(frame)
            finally:
                reader.close()
        ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])
        if not ok:
            raise WorkbenchError("Frame encoding failed.")
        return buf.tobytes(), "image/png"

    # ---------------------------------------------------------------- uploads
    def accept_upload(self, filename: str, media_type: str, data: bytes, *, clip_id: str | None = None,
                      intrinsics: str | None = None) -> dict:
        safe_name = Path(str(filename).replace("\\", "/")).name
        if not safe_name or len(safe_name) > 240 or any(ord(ch) < 32 for ch in safe_name):
            raise WorkbenchError("Upload filename is invalid.")
        extension = Path(safe_name).suffix.lower()
        expected = VIDEO_TYPES.get(extension)
        if expected is None or media_type.split(";", 1)[0].strip().lower() not in (expected, "application/octet-stream"):
            raise WorkbenchError("Upload must be an MP4, M4V, MOV or WebM video.")
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise WorkbenchError(f"Each upload must be between 1 and {MAX_UPLOAD_BYTES:,} bytes.")
        current = sum(p.stat().st_size for p in self.import_root.iterdir()
                      if p.is_file() and not p.name.endswith(".json"))
        if current + len(data) > MAX_IMPORT_TOTAL_BYTES:
            raise WorkbenchError(f"Imports would exceed the {MAX_IMPORT_TOTAL_BYTES:,}-byte aggregate input budget.")
        source_id = uuid.uuid4().hex
        media = self.import_root / f"{source_id}{extension}"
        meta_path = self.import_root / f"{source_id}.meta.json"
        row = {"source_id": source_id, "original_name": safe_name, "extension": extension, "media_type": expected,
               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "created_at": time.time(),
               "registration_status": "needs_registration", "clip_id": None,
               "blocked_reason": "Supply a clip id and measured K (or mark calibration unavailable) to register."}
        with media.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            self._write_new(meta_path, row)
        except Exception:
            media.unlink(missing_ok=True)
            raise
        if clip_id:
            try:
                clip = self.register_import(source_id, clip_id, intrinsics)
                row.update(registration_status="registered", clip_id=clip.clip_id, blocked_reason=None)
            except (WorkbenchError, self.cli.WorkbenchCLIError, wc.ContractError) as exc:
                row.update(registration_status="failed", blocked_reason=f"Registration failed: {exc}")
            self._write_mutable(meta_path, row)
        return {k: row[k] for k in ("source_id", "original_name", "bytes", "sha256", "registration_status",
                                    "clip_id", "blocked_reason")}

    def register_import(self, source_id: str, clip_id: str, intrinsics: str | None) -> wc.ClipSpec:
        if not HEX32.fullmatch(source_id):
            raise WorkbenchError("Invalid source id.")
        meta = wc.loads_strict((self.import_root / f"{source_id}.meta.json").read_bytes())
        media = _safe_member(self.import_root, f"{source_id}{meta['extension']}")
        args = argparse.Namespace(clip_id=wc._ident(clip_id, "clip_id"), title=meta["original_name"], K=None,
                                  calibration_unavailable=False, principal_origin="integer_centers",
                                  keyframes=None, select_frame=None, positive=None, negative=None, reviewed_by=None)
        if intrinsics in (None, "", "unavailable"):
            args.calibration_unavailable = True
        else:
            args.K = self.cli.parse_K(intrinsics)
        return self.cli.import_video(self.ws, media, args)

    # ---------------------------------------------------------------- selection and runs
    def create_selection(self, payload: Any) -> dict:
        if not isinstance(payload, dict):
            raise WorkbenchError("Selection must be an object.")
        clip = self.clip(str(payload.get("clip_id")))
        onboarding = wc.Onboarding.from_dict({
            "keyframe_ids": payload.get("keyframe_ids"),
            "selection": {"frame_id": payload.get("frame_id"), "positive": payload.get("positive", []),
                          "negative": payload.get("negative", []), "reviewed": True,
                          "reviewed_by": str(payload.get("reviewed_by") or "local workbench user")[:120]}})
        attestation = wc.check_onboarding(clip, onboarding, ())
        selection_id = uuid.uuid4().hex
        receipt = {"selection_id": selection_id, "clip_id": clip.clip_id, "onboarding": onboarding.to_dict(),
                   "prefix_attestation": attestation, "created_at": time.time(),
                   "coordinates": "source pixels, integer pixel centres"}
        self._write_new(self.selections_root / f"{selection_id}.json", receipt)
        return receipt

    def create_run(self, payload: Any) -> dict:
        if not isinstance(payload, dict):
            raise WorkbenchError("Run request must be an object.")
        clip = self.clip(str(payload.get("clip_id")))
        onboarding = None
        selection_id = payload.get("selection_id")
        if selection_id is not None:
            if not isinstance(selection_id, str) or not HEX32.fullmatch(selection_id):
                raise WorkbenchError("Invalid selection id.")
            receipt = wc.loads_strict((self.selections_root / f"{selection_id}.json").read_bytes())
            if receipt["clip_id"] != clip.clip_id:
                raise WorkbenchError("Selection belongs to another clip.")
            onboarding = wc.Onboarding.from_dict(receipt["onboarding"])
        roles = payload.get("roles") or list(DEFAULT_ROLES)
        if (not isinstance(roles, list) or not all(isinstance(r, str) and r in wc.BACKEND_ROLES for r in roles)
                or len(set(roles)) != len(roles)):
            raise WorkbenchError(f"Roles must be unique names from {list(wc.BACKEND_ROLES)}.")
        if payload.get("mask_set"):
            try:
                mask_set = wc._ident(payload["mask_set"], "mask_set")
            except wc.ContractError as exc:
                raise WorkbenchError(str(exc)) from exc
            mask_source = "request"
        else:
            masks = self._default_masks(clip.clip_id)
            mask_set, mask_source = masks["default"], masks["source"]
        config = {"mask_set": mask_set} if mask_set else {}
        try:
            run = self.cli.build_run_spec(clip, mode=str(payload.get("mode", "smoke")), onboarding=onboarding,
                                          first_frame=payload.get("first_frame"), count=payload.get("count"),
                                          roles=tuple(roles), artwork_kind=str(payload.get("artwork_kind", "checkerboard")),
                                          config=config)
        except self.cli.WorkbenchCLIError as exc:
            raise WorkbenchError(str(exc)) from exc
        control = {"status": "not_requested", "ready": False, "bundle_sha256": None}
        if any(role in NEURAL_ROLES for role in run.roles):
            control, binding = self._resolve_control(clip, run.onboarding.selection.frame_id, full=True)
            if binding is not None and binding.get("kind") == EXECUTION_CONTROL_KIND:
                # Server-derived identity only (never caller-provided): bound into the run config;
                # every other run field is copied unchanged.
                try:
                    run = wc.RunSpec.from_dict({**run.to_dict(),
                                                "config": {**run.config, "metric_control": binding["run_config"]}})
                    run.validate_against(clip)
                except wc.ContractError as exc:
                    raise WorkbenchError(str(exc)) from exc
        record = {"run_id": run.run_id, "run": run.to_dict(), "selection_id": selection_id, "created_at": time.time(),
                  "status": "requested", "mask_set_source": mask_source, "metric_control": control}
        self._write_new(self.requests_root / f"{run.run_id}.json", record)
        return record

    def _request(self, run_id: str) -> dict:
        if not isinstance(run_id, str) or not re.fullmatch(ID_PATTERN, run_id):
            raise KeyError(run_id)
        path = self.requests_root / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(run_id)
        return wc.loads_strict(path.read_bytes())

    def list_runs(self, clip_id: str | None = None) -> dict:
        runs = []
        for path in sorted(self.ws.runs_dir.glob("*/report.json")):
            try:
                report = wc.loads_strict(path.read_bytes())
            except (OSError, ValueError):
                continue
            if clip_id and report.get("clip_id") != clip_id:
                continue
            runs.append({"run_id": report.get("run_id"), "clip_id": report.get("clip_id"),
                         "mode": report.get("mode"), "status": report.get("status"),
                         "scheduled_frames": report.get("scheduled_frames")})
        return {"runs": runs}

    def run_detail(self, run_id: str) -> dict:
        try:
            run, rows, report = self.cli.load_results(self.ws, run_id)
        except (OSError, ValueError, wc.ContractError) as exc:
            raise KeyError(run_id) from exc
        from . import workbench_artwork_replay as war
        return {"run": run, "report": report, "frame_results": [safe_public_frame(r.to_dict()) for r in rows],
                "artwork_sources": war.artwork_sources(run_id, rows, self.ws.runs_dir / run_id)}

    # ---------------------------------------------------------------- RAM-only artwork replay
    def _artwork_bundle(self, clip_id: str) -> tuple[Path, str] | None:
        """Server-registered geometry only (never a browser path): execution control, per-clip or bound global."""
        if clip_id in self._exec_map:
            entry = self._exec_map[clip_id]
            return entry["directory"] / "metric-bundle.json", entry["identity"]["bundle_sha256"]
        binding, _source = self._bundle_for_clip(clip_id)
        if binding is None or binding.get("error"):
            return None
        return Path(binding["path"]), binding["sha256"]

    def artwork_replay(self, run_id: str, frame_id: str, role: str) -> tuple[int, dict]:
        """CPU replay of saved-pose artwork, in memory only (see bench/workbench_artwork_replay.py)."""
        from . import workbench_artwork_replay as war
        with self._state_lock:
            active = self._active_job and (self._jobs.get(self._active_job) or {}).get("run_id") == run_id
            if self._artwork_replayer is None:
                admission = self.run_options.get("admission")
                self._artwork_replayer = war.ArtworkRamReplay(self.ws, bundle_resolver=self._artwork_bundle,
                                                              admission_dir=Path(admission) if admission else None)
        if active:
            return 409, {"schema": war.SCHEMA, "status": "not_ready", "run_id": run_id, "frame_id": frame_id, "role": role,
                         "reason": "the run is still executing", **war.FLAGS}
        return self._artwork_replayer.replay(run_id, frame_id, role)

    def run_evidence(self, run_id: str, frame_id: str, kind: str, role: str = "relative_geometry") -> Path:
        _run, rows, _report = self.cli.load_results(self.ws, run_id)
        if kind == "overlay":
            if role not in wc.BACKEND_ROLES:
                raise KeyError(role)
            for row in rows:
                if (row.frame_id == frame_id and row.backend_role == role and row.render_state == "rendered"
                        and row.pose is not None and row.evidence.get("overlay")):
                    return _safe_member(self.ws.runs_dir / run_id, row.evidence["overlay"])
            raise KeyError(kind)
        if kind == "mask":
            run = _run["run"]
            mask_set = (run.get("config") or {}).get("mask_set")
            store = self.cli.MaskStore(self.ws, run["clip_id"], mask_set)
            entry = store.entry(frame_id)
            if entry is None or not any(r.frame_id == frame_id and r.mask_state == "observed" for r in rows):
                raise KeyError(kind)
            path = Path(entry["abs_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise WorkbenchError("Mask evidence changed.")
            return path
        raise KeyError(kind)

    # ---------------------------------------------------------------- jobs
    def _public_job(self, job: dict | None) -> dict | None:
        if job is None:
            return None
        return {k: job.get(k) for k in ("job_id", "kind", "run_id", "mask_set", "status", "created_at", "updated_at",
                                        "error", "blocked_reason", "cancel_requested", "frame_result_count", "run_dir",
                                        "frames", "roles", "metric_control")}

    def _unavailable_rows(self, run: wc.RunSpec, clip: wc.ClipSpec, reason: str) -> list[dict]:
        rows = []
        for role in run.roles:
            for fid in run.scheduled_frame_ids:
                rows.append(wc.blocked_result(run.run_id, clip, clip.frame(fid), role, {"role": role}, reason).to_dict())
        return rows

    def _persist_unavailable(self, job: dict, run: wc.RunSpec, clip: wc.ClipSpec, reason: str) -> None:
        rows = self._unavailable_rows(run, clip, reason)
        path = self.jobs_root / f"{job['job_id']}.results.jsonl"
        with path.open("xb") as stream:
            for row in rows:
                stream.write(wc.canonical_json(row) + b"\n")
        job["frame_result_count"] = len(rows)
        job["frame_results_file"] = path.name

    def create_job(self, run_id: str) -> dict:
        request = self._request(run_id)
        run = wc.RunSpec.from_dict(request["run"])
        clip = self.clip(run.clip_id)
        run.validate_against(clip)
        neural = any(role in NEURAL_ROLES for role in run.roles)
        control, binding = ({"status": "not_requested", "ready": False, "bundle_sha256": None}, None)
        if neural:
            control, binding = self._resolve_control(clip, run.onboarding.selection.frame_id, full=True)
        recorded = request.get("metric_control")
        run_control = run.config.get("metric_control") if isinstance(run.config, dict) else None
        with self._state_lock:
            job_id = uuid.uuid4().hex
            now = time.time()
            job = {"job_id": job_id, "run_id": run_id, "status": "queued", "created_at": now, "updated_at": now,
                   "roles": list(run.roles), "mask_set": run.config.get("mask_set"), "metric_control": control}
            path = self.jobs_root / f"{job_id}.json"
            self._write_new(path, job)
            self._jobs[job_id] = job
            reason = None
            if not self.use_pipeline and self.prepare_adapter is None:
                reason = "Preparation pipeline is not connected; no model or geometry work was started."
            elif (self.ws.runs_dir / run_id).exists():
                reason = "This run already executed; create a new run so earlier outputs are preserved."
            elif neural and control["status"] == "blocked":
                reason = (f"Prepared metric control for clip {clip.clip_id} is blocked: "
                          f"{'; '.join(control['reasons'])}. No other clip's geometry is substituted.")
            elif neural and isinstance(recorded, dict) and recorded.get("bundle_sha256") != control.get("bundle_sha256"):
                reason = ("The prepared metric control changed since this run was requested; "
                          "create a new run so its bundle identity is explicit.")
            elif neural and (run_control is not None or clip.clip_id in self._exec_map) and (
                    binding is None or not isinstance(run_control, dict)
                    or run_control.get("identity_sha256") != binding.get("identity_sha256")
                    or wc.canonical_sha256({k: v for k, v in run_control.items() if k != "identity_sha256"})
                    != binding.get("identity_sha256")):
                reason = (f"The execution-only control for clip {clip.clip_id} is not ready or differs from the "
                          "identity bound when this run was requested; create a new run. No other geometry is used.")
            elif not self._job_lock.acquire(blocking=False):
                reason = "Another job is active; jobs are serialized. Retry after it finishes or cancel it."
            if reason is not None:
                job.update(status="blocked", blocked_reason=reason, updated_at=time.time())
                self._persist_unavailable(job, run, clip, reason)
                self._write_mutable(path, job)
                return self._public_job(job)
            self._active_job = job_id
            self._cancel = threading.Event()
            thread = threading.Thread(target=self._run_job, args=(job_id, run, clip, self._cancel, binding),
                                      name=f"workbench-{job_id[:8]}", daemon=True)
            thread.start()
            return self._public_job(job)

    def _run_job(self, job_id: str, run: wc.RunSpec, clip: wc.ClipSpec, cancel: threading.Event,
                 binding: dict | None = None) -> None:
        path = self.jobs_root / f"{job_id}.json"
        with self._state_lock:
            job = self._jobs[job_id]
            job.update(status="running", updated_at=time.time())
            self._write_mutable(path, job)
        try:
            if binding is not None:
                try:
                    current = _sha256_file(binding["path"])[0]
                except OSError:
                    current = None
                if current != binding["sha256"]:
                    reason = "Prepared metric bundle changed or vanished before the pipeline started; nothing was run."
                    with self._state_lock:
                        job.update(status="blocked", blocked_reason=reason, updated_at=time.time())
                        self._persist_unavailable(job, run, clip, reason)
                        self._write_mutable(path, job)
                    return
            if binding is not None and binding.get("kind") == EXECUTION_CONTROL_KIND:
                # Launch gate immediately before the pipeline: full re-derivation and
                # exact equality with both the startup identity and the run's bound identity.
                gate, fresh = self._resolve_control(clip, run.onboarding.selection.frame_id, full=True)
                bound = run.config.get("metric_control") if isinstance(run.config, dict) else None
                if (fresh is None or fresh.get("identity_sha256") != binding.get("identity_sha256")
                        or not isinstance(bound, dict) or bound.get("identity_sha256") != binding.get("identity_sha256")):
                    reason = ("Execution-only control failed re-validation immediately before the pipeline "
                              f"({'; '.join(gate.get('reasons') or ['identity changed'])}); nothing was run.")
                    with self._state_lock:
                        job.update(status="blocked", blocked_reason=reason[:2000], updated_at=time.time())
                        self._persist_unavailable(job, run, clip, reason[:2000])
                        self._write_mutable(path, job)
                    return
                binding = fresh
            if self.prepare_adapter is not None:
                result = self.prepare_adapter(run, clip, cancel)
            else:
                admission = None
                if self.run_options.get("admission"):
                    from . import workbench_neural as wn
                    admission = wn.load_admission(Path(self.run_options["admission"]))
                options = self.cli.RunOptions(mask_set=run.config.get("mask_set"),
                                              colmap=self.run_options.get("colmap"),
                                              head_path=self.run_options.get("head"), admission=admission,
                                              metric_bundle=str(binding["path"]) if binding else None,
                                              device=self.run_options.get("device") or "cpu")
                result = self.cli.execute_run(self.ws, clip, run, options, cancel)
            status = result.get("status") if isinstance(result, dict) else None
            if cancel.is_set():
                status = "cancelled"
            if status not in TERMINAL:
                status = "failed"
            control_changed = (binding is not None and binding.get("kind") == EXECUTION_CONTROL_KIND
                               and not self._execution_files_unchanged(clip.clip_id))
            if control_changed:
                status = "failed"
            with self._state_lock:
                job.update(status=status, updated_at=time.time(), run_dir=result.get("run_dir") if isinstance(result, dict) else None)
                if control_changed:
                    job["error"] = ("Execution-only control files changed while the job ran; outputs are retained "
                                    "but are not attributable to the validated control.")
                if status == "blocked":
                    job["blocked_reason"] = "One or more stages lacked real prerequisites; see run stages."
                report = (result or {}).get("report", {}) if isinstance(result, dict) else {}
                job["frame_result_count"] = sum(v.get("result_rows", 0) for v in report.get("roles", {}).values())
                if not (self.ws.runs_dir / run.run_id / "results.jsonl").is_file():
                    self._persist_unavailable(job, run, clip, f"job_{status}: no pipeline results")
                self._write_mutable(path, job)
        except Exception as exc:  # noqa: BLE001 - surface failure, preserve receipts
            with self._state_lock:
                status = "cancelled" if cancel.is_set() else "failed"
                job.update(status=status, updated_at=time.time(), error=f"{type(exc).__name__}: {exc}"[:2000])
                if not (self.ws.runs_dir / run.run_id / "results.jsonl").is_file():
                    try:
                        self._persist_unavailable(job, run, clip, f"job_{status}: {type(exc).__name__}")
                    except FileExistsError:
                        pass
                self._write_mutable(path, job)
        finally:
            with self._state_lock:
                if self._active_job == job_id:
                    self._active_job = None
                    self._cancel = None
            self._job_lock.release()

    def create_segmentation_job(self, payload: Any) -> dict:
        """Explicit user action: prompt SAM 2.1 with a persisted reviewed selection."""
        if not isinstance(payload, dict):
            raise WorkbenchError("Segmentation request must be an object.")
        clip = self.clip(str(payload.get("clip_id")))
        selection_id = payload.get("selection_id")
        if not isinstance(selection_id, str) or not HEX32.fullmatch(selection_id):
            raise WorkbenchError("Segmentation requires a persisted selection_id.")
        receipt = wc.loads_strict((self.selections_root / f"{selection_id}.json").read_bytes())
        if receipt["clip_id"] != clip.clip_id:
            raise WorkbenchError("Selection belongs to another clip.")
        onboarding = wc.Onboarding.from_dict(receipt["onboarding"])
        mask_set = wc._ident(payload.get("mask_set") or f"sam2-{selection_id[:12]}", "mask_set")
        scheduled = wc.schedule_frames(clip, str(payload.get("mode", "smoke")), onboarding,
                                       first_frame_id=payload.get("first_frame"), count=payload.get("count"))
        from . import workbench_neural as wn
        pre = wn.segmentation_preflight()
        with self._state_lock:
            job_id = uuid.uuid4().hex
            now = time.time()
            job = {"job_id": job_id, "kind": "segmentation", "run_id": None, "mask_set": mask_set,
                   "status": "queued", "created_at": now, "updated_at": now}
            path = self.jobs_root / f"{job_id}.json"
            self._write_new(path, job)
            self._jobs[job_id] = job
            reason = None
            if not pre["ready"]:
                reason = f"Segmentation prerequisites missing: {pre['blocked_reasons']}"
            elif not self._job_lock.acquire(blocking=False):
                reason = "Another job is active; jobs are serialized."
            if reason is not None:
                job.update(status="blocked", blocked_reason=reason, updated_at=time.time())
                self._write_mutable(path, job)
                return self._public_job(job)
            self._active_job = job_id
            self._cancel = threading.Event()
            cancel = self._cancel

        def target():
            try:
                with self._state_lock:
                    job.update(status="running", updated_at=time.time())
                    self._write_mutable(path, job)
                result = self.cli.run_segmentation(self.ws, clip, onboarding, scheduled, mask_set,
                                                   device=self.run_options.get("device") or "cpu", cancel=cancel)
                status = "cancelled" if result.get("cancelled") else "completed"
                with self._state_lock:
                    job.update(status=status, updated_at=time.time(), frames=result.get("frames"))
                    self._write_mutable(path, job)
            except Exception as exc:  # noqa: BLE001
                with self._state_lock:
                    job.update(status="cancelled" if cancel.is_set() else "failed", updated_at=time.time(),
                               error=f"{type(exc).__name__}: {exc}"[:2000])
                    self._write_mutable(path, job)
            finally:
                with self._state_lock:
                    if self._active_job == job_id:
                        self._active_job = None
                        self._cancel = None
                self._job_lock.release()
        threading.Thread(target=target, name=f"workbench-seg-{job_id[:8]}", daemon=True).start()
        return self._public_job(job)

    def get_job(self, job_id: str) -> dict:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self._public_job(job)

    def job_results(self, job_id: str) -> dict:
        with self._state_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if (self.ws.runs_dir / job["run_id"] / "results.jsonl").is_file():
            return self.run_detail(job["run_id"])
        name = job.get("frame_results_file")
        if not name:
            return {"frame_results": []}
        rows = [safe_public_frame(wc.loads_strict(line)) for line in
                (self.jobs_root / name).read_bytes().splitlines() if line.strip()]
        return {"frame_results": rows}

    def cancel_job(self, job_id: str) -> dict:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.get("status") in ("queued", "running") and self._active_job == job_id and self._cancel:
                self._cancel.set()
                job.update(cancel_requested=True, updated_at=time.time())
                self._write_mutable(self.jobs_root / f"{job_id}.json", job)
            return self._public_job(job)


# ---------------------------------------------------------------------- HTTP

class WorkbenchHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def create_server(app: WorkbenchService, host: str = "127.0.0.1", port: int = 8765) -> WorkbenchHTTPServer:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise WorkbenchError("Workbench listener must be a loopback IP address.") from exc
    if not address.is_loopback:
        raise WorkbenchError("Workbench listener is loopback-only.")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise WorkbenchError("Port is outside the supported range.")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "VisualizeITWorkbench/2"

        @property
        def app(self) -> WorkbenchService:
            return self.server.app  # type: ignore[attr-defined]

        def log_message(self, fmt: str, *args: Any) -> None:
            print("workbench: " + (fmt % args), flush=True)

        def _headers(self, content_type: str, length: int | None = None) -> None:
            self.send_header("Content-Type", content_type)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; "
                             "script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self._headers(content_type, len(data))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, value: Any, status: int = 200) -> None:
            self._send_bytes(_json_bytes(value), "application/json; charset=utf-8", status)

        def _error(self, status: int, message: str) -> None:
            self._send_json({"error": message, "status": status}, status)

        def _path(self) -> tuple[str, dict]:
            parsed = urlsplit(self.path)
            try:
                path = unquote(parsed.path, errors="strict")
            except (UnicodeDecodeError, ValueError) as exc:
                raise WorkbenchError("Request path encoding is invalid.") from exc
            if "\\" in path or "\x00" in path or ".." in path.split("/"):
                raise WorkbenchError("Request path is invalid.")
            return path, parse_qs(parsed.query)

        def _check_host(self) -> bool:
            port_ = self.server.server_port
            raw_host = self.headers.get("Host", "")
            accepted = {f"127.0.0.1:{port_}", f"localhost:{port_}", f"[::1]:{port_}"}
            if raw_host not in accepted:
                self._error(403, "This service accepts requests for its loopback host only.")
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{h}" for h in accepted}:
                self._error(403, "Cross-origin requests are not accepted.")
                return False
            return True

        def _body(self, limit: int) -> bytes:
            raw = self.headers.get("Content-Length")
            if self.headers.get("Transfer-Encoding"):
                raise WorkbenchError("Chunked request bodies are not accepted.")
            if raw is None or not raw.isascii() or not raw.isdecimal():
                raise WorkbenchError("Content-Length is required.")
            length = int(raw)
            if length > limit:
                raise WorkbenchError("Request body exceeds the bounded request size.")
            data = self.rfile.read(length)
            if len(data) != length:
                raise WorkbenchError("Request body ended before Content-Length.")
            return data

        def _json_body(self) -> Any:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise WorkbenchError("Content-Type must be application/json.")
            try:
                return wc.loads_strict(self._body(MAX_JSON_BYTES))
            except (UnicodeDecodeError, ValueError) as exc:
                raise WorkbenchError(f"Request JSON is invalid: {exc}") from exc

        def _serve_file(self, path: Path, media_type: str, *, allow_range: bool) -> None:
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            raw_range = self.headers.get("Range") if allow_range else None
            if raw_range:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw_range.strip())
                if not match or (not match.group(1) and not match.group(2)):
                    self._error(416, "Only a single byte range is supported.")
                    return
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = int(right) if right else size - 1
                else:
                    start, end = max(0, size - int(right)), size - 1
                if start >= size or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self._headers("application/json; charset=utf-8", 0)
                    self.end_headers()
                    return
                end = min(end, size - 1, start + MAX_RANGE_BYTES - 1)
                status = 206
            length = max(0, end - start + 1)
            self.send_response(status)
            self._headers(media_type, length)
            if allow_range:
                self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    block = stream.read(min(64 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_GET(self) -> None:  # noqa: N802
            if not self._check_host():
                return
            try:
                path, query = self._path()
                if path in STATIC_FILES:
                    name, media_type = STATIC_FILES[path]
                    file_path = _safe_member(self.app.web_root, name)
                    self._serve_file(file_path, media_type, allow_range=False)
                    return
                if path == "/api/status":
                    return self._send_json(self.app.status())
                if path == "/api/clips":
                    return self._send_json(self.app.list_clips())
                if path == "/api/runs":
                    clip_id = (query.get("clip_id") or [None])[0]
                    return self._send_json(self.app.list_runs(clip_id))
                m = re.fullmatch(rf"/api/clips/({ID_PATTERN})", path)
                if m:
                    return self._send_json(self.app.clip_detail(m.group(1)))
                m = re.fullmatch(rf"/api/clips/({ID_PATTERN})/media", path)
                if m:
                    media, media_type = self.app.media_path(m.group(1))
                    return self._serve_file(media, media_type, allow_range=True)
                m = re.fullmatch(rf"/api/clips/({ID_PATTERN})/frames/({ID_PATTERN})/image", path)
                if m:
                    data, media_type = self.app.frame_image(m.group(1), m.group(2))
                    return self._send_bytes(data, media_type)
                m = re.fullmatch(rf"/api/runs/({ID_PATTERN})", path)
                if m:
                    return self._send_json(self.app.run_detail(m.group(1)))
                m = re.fullmatch(rf"/api/runs/({ID_PATTERN})/frames/({ID_PATTERN})/artwork-replay", path)
                if m:
                    role = (query.get("role") or ["primary_neural"])[0]
                    status, body = self.app.artwork_replay(m.group(1), m.group(2), role)
                    if status == 429:
                        data = _json_bytes(body)
                        self.send_response(status)
                        self._headers("application/json; charset=utf-8", len(data))
                        self.send_header("Retry-After", "1")
                        self.end_headers()
                        self.wfile.write(data)
                        return None
                    return self._send_json(body, status)
                m = re.fullmatch(rf"/api/runs/({ID_PATTERN})/frames/({ID_PATTERN})/(overlay|mask)", path)
                if m:
                    role = (query.get("role") or ["relative_geometry"])[0]
                    asset = self.app.run_evidence(m.group(1), m.group(2), m.group(3), role)
                    return self._serve_file(asset, "image/png", allow_range=False)
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
                if m:
                    return self._send_json(self.app.get_job(m.group(1)))
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/results", path)
                if m:
                    return self._send_json(self.app.job_results(m.group(1)))
                self._error(404, "Route was not found.")
            except KeyError:
                self._error(404, "Requested item is not registered.")
            except (WorkbenchError, wc.ContractError) as exc:
                self._error(400, str(exc))
            except Exception as exc:  # noqa: BLE001
                self._error(500, f"Workbench request failed: {type(exc).__name__}.")

        def do_POST(self) -> None:  # noqa: N802
            if not self._check_host():
                return
            try:
                path, _query = self._path()
                if path == "/api/import":
                    result = self.app.accept_upload(self.headers.get("X-Filename", ""),
                                                    self.headers.get("Content-Type", "application/octet-stream"),
                                                    self._body(MAX_UPLOAD_BYTES),
                                                    clip_id=self.headers.get("X-Clip-Id") or None,
                                                    intrinsics=self.headers.get("X-Intrinsics") or None)
                    return self._send_json(result, 201)
                if path == "/api/selections":
                    return self._send_json(self.app.create_selection(self._json_body()), 201)
                if path == "/api/segment":
                    return self._send_json(self.app.create_segmentation_job(self._json_body()), 202)
                if path == "/api/runs":
                    return self._send_json(self.app.create_run(self._json_body()), 201)
                if path == "/api/jobs":
                    body = self._json_body()
                    if not isinstance(body, dict) or not isinstance(body.get("run_id"), str):
                        raise WorkbenchError("Job request requires run_id.")
                    return self._send_json(self.app.create_job(body["run_id"]), 202)
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", path)
                if m:
                    return self._send_json(self.app.cancel_job(m.group(1)))
                self._error(404, "Route was not found.")
            except KeyError:
                self._error(404, "Requested item is not registered.")
            except (WorkbenchError, wc.ContractError) as exc:
                self._error(400, str(exc))
            except Exception as exc:  # noqa: BLE001
                self._error(500, f"Workbench request failed: {type(exc).__name__}.")

        def do_PUT(self) -> None:  # noqa: N802
            self._error(405, "Method is not allowed.")

        def do_DELETE(self) -> None:  # noqa: N802
            self._error(405, "Method is not allowed.")

    import socket

    class BoundServer(WorkbenchHTTPServer):
        address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET

    server = BoundServer((str(address), port), Handler)
    server.app = app  # type: ignore[attr-defined]
    return server


def _parse_root(value: str) -> tuple[str, str]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("Roots use NAME=PATH.")
    return name, path


def _parse_mapping(value: str) -> tuple[str, str]:
    clip_id, sep, target = value.partition("=")
    if not sep or not clip_id or not target:
        raise argparse.ArgumentTypeError("Per-clip mappings use CLIP_ID=VALUE.")
    return clip_id, target


def main(argv: list[str] | None = None) -> int:
    from . import workbench_cli as cli
    parser = argparse.ArgumentParser(description="Run the loopback-only VisualizeIT video workbench.")
    parser.add_argument("--workspace", default=str(cli.DEFAULT_WORKSPACE))
    parser.add_argument("--root", action="append", default=[], type=_parse_root, metavar="NAME=PATH",
                        help="register an allowlisted media root (repeatable)")
    parser.add_argument("--host", default="127.0.0.1", help="loopback IP only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--colmap", help="installed COLMAP executable")
    parser.add_argument("--head", help="flow-terminal head.pt path")
    parser.add_argument("--masks", help="default observed mask set name")
    parser.add_argument("--admission", help="admission directory (re-verified for every job)")
    parser.add_argument("--metric-bundle", help="legacy global metric bundle; applies only to --metric-bundle-clip")
    parser.add_argument("--metric-bundle-clip", metavar="CLIP_ID",
                        help="the one clip the legacy --metric-bundle belongs to (required for it to be used)")
    parser.add_argument("--metric-bundle-for", action="append", default=[], type=_parse_mapping,
                        metavar="CLIP_ID=PATH", help="prepared metric bundle for one clip (repeatable; wins over global)")
    parser.add_argument("--masks-for", action="append", default=[], type=_parse_mapping,
                        metavar="CLIP_ID=MASK_SET", help="default observed mask set for one clip (repeatable)")
    parser.add_argument("--execution-control-for", action="append", default=[], type=_parse_mapping,
                        metavar="CLIP_ID=DIRECTORY",
                        help="emitted execution-only supplemental YCB control directory for one clip (repeatable; "
                             "fully validated at startup, request, job start and before the pipeline)")
    parser.add_argument("--device", default="cpu", help="torch device for explicitly triggered neural jobs")
    args = parser.parse_args(argv)
    try:
        roots = dict(args.root)
        if len(roots) != len(args.root):
            raise WorkbenchError("Root names must be unique.")
        app = WorkbenchService(args.workspace, roots=roots,
                               run_options={"colmap": args.colmap, "head": args.head, "mask_set": args.masks,
                                            "admission": args.admission, "metric_bundle": args.metric_bundle,
                                            "metric_bundle_clip": args.metric_bundle_clip,
                                            "metric_bundles_by_clip": args.metric_bundle_for,
                                            "masks_by_clip": args.masks_for,
                                            "execution_controls_by_clip": args.execution_control_for,
                                            "device": args.device})
        server = create_server(app, args.host, args.port)
    except (WorkbenchError, OSError, ValueError, cli.WorkbenchCLIError) as exc:
        parser.error(str(exc))
    print(f"Workbench ready at http://127.0.0.1:{server.server_port}/ (local only)", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
