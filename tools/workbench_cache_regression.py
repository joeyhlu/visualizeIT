"""Cache-only regression replay of the seven fixed completed paired full-window runs.

This measures ACTUAL reuse of the unchanged pipeline's stage cache. For each case
it rebuilds the original ``RunSpec`` (new ``run_id`` only), the original admission
attempt, metric bundle, mask set and head, then calls the real
``workbench_cli.execute_run``. The pipeline therefore recomputes and
re-verifies every cache key itself (input files, masks, onboarding, model and
checkpoint hashes, admission body, module hashes, warm-up, bundle and
artwork). This is not a re-read of sealed reports.

Guards make new inference impossible. While a case runs, entering the GPU
stage, constructing ``GoTrackMetricBackend``/``NeuralSession``, running SAM
segmentation, or storing into the stage cache raises ``CacheOnlyViolation``,
a ``BaseException`` so the core's generic ``except Exception`` handlers cannot
swallow it. A cache miss reaches the GPU-stage guard and stops the case with
an explicit cache-miss failure. It never yields "unavailable" rows that look
like success. All patches are restored on every path.

Each completed case must show:

* both role stages ``cache_hit``;
* zero new frame telemetry and no CUDA initialisation;
* complete frame accounting;
* new rows equal to the original rows exactly, removing ONLY ``run_id``;
* overlay evidence bytes equal to the originals;
* ``neural-artwork.json`` equal to the original.

Cached per-frame timings are reported as *cached historical timings* (never
inference latency); the current replay wall time is measured separately.

No model-quality, pooling, head-superiority or mobile claim. The original and
qualified cracker runs remain distinct cases, and nothing is promoted into a
default suite.

    plan --out FRESH.json
    run --plan FILE --expected-plan-sha256 HEX --out FRESH-summary.json [--case KEY]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
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

PLAN_SCHEMA = "workbench_cache_regression_plan_v1"
SUMMARY_SCHEMA = "workbench_cache_regression_summary_v1"
WORKSPACE = ".cache/video-workbench"
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
ROLES = ("primary_neural", "frozen_baseline")
_ADM = ".cache/video-workbench/admission"
_CTL = ".cache/video-workbench/derived/execution-controls/v1"
CASES = {
    "mug": {"run_id": "run-20261021T002156-1b2de77d", "frames": 240, "admission": f"{_ADM}/attempt-03",
            "bundle": f"{_ADM}/bundle-mug-v1.json"},
    "keyboard": {"run_id": "run-20261021T013943-538cfa03", "frames": 240, "admission": f"{_ADM}/attempt-04",
                 "bundle": f"{_ADM}/bundle-keyboard-v1.json"},
    "ranch": {"run_id": "run-20261021T020454-01d11a29", "frames": 240, "admission": f"{_ADM}/attempt-05",
              "bundle": f"{_ADM}/bundle-ranch-v1.json"},
    "cracker-original": {"run_id": "run-20261021T030702-475adc39", "frames": 110, "admission": f"{_ADM}/attempt-06",
                         "bundle": f"{_CTL}/object-000002/metric-bundle.json"},
    "soup": {"run_id": "run-20261021T031729-0986ef46", "frames": 110, "admission": f"{_ADM}/attempt-06",
             "bundle": f"{_CTL}/object-000004/metric-bundle.json"},
    "marker": {"run_id": "run-20261021T032932-539343a7", "frames": 110, "admission": f"{_ADM}/attempt-06",
               "bundle": f"{_CTL}/object-000018/metric-bundle.json"},
    "cracker-qualified": {"run_id": "run-20261021T060541-e8894b79", "frames": 110, "admission": f"{_ADM}/attempt-06",
                          "bundle": f"{_CTL}/object-000002/metric-bundle.json"},
}
# Modules hashed into the neural stage key (workbench_cli._neural); drift => the key changes => certain miss.
KEY_MODULES = ("workbench_neural", "workbench_cli", "workbench_contract", "workbench_geometry", "workbench_render",
               "workbench_reconstruction", "quality_gotrack", "quality_gotrack_inference", "quality_render_stability")
REPORT_MODULES = ("workbench_cli", "workbench_contract", "workbench_geometry", "workbench_neural",
                  "workbench_reconstruction", "workbench_telemetry", "workbench_cache", "quality_gotrack",
                  "quality_gotrack_inference")
ORIGINAL_FILES = ("run.json", "results.jsonl", "report.json", "neural-artwork.json")
MAX_JSON_BYTES = 256 << 10
OUTPUT_MARGIN_BYTES = 1 << 20


class RegressionError(RuntimeError):
    """Plan, closure or comparison failure (fail closed)."""


class CacheOnlyViolation(BaseException):
    """Raised by the guards; a BaseException so core ``except Exception`` handlers cannot swallow it."""


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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def loads_strict(raw: bytes, what: str) -> Any:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise RegressionError(f"{what}: duplicate JSON key {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise RegressionError(f"{what}: non-finite JSON value {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegressionError(f"{what}: invalid JSON ({exc})") from None


def _rel(path: Path) -> str:
    resolved = Path(os.path.abspath(path))
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise RegressionError(f"{path} is outside the repository") from None


def _file_id(path: Path) -> dict:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise RegressionError(f"{_rel(p)} is missing or not a regular file")
    return {"path": _rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def _tree_id(directory: Path) -> dict:
    """Aggregate identity of every regular file under a directory (no links allowed)."""
    d = Path(directory)
    if d.is_symlink() or not d.is_dir():
        raise RegressionError(f"{_rel(d)} is missing or not a directory")
    rows = []
    for p in sorted(d.rglob("*")):
        if p.is_symlink():
            raise RegressionError(f"{_rel(p)} is a link")
        if p.is_file():
            rows.append([p.relative_to(d).as_posix(), sha256_file(p), p.stat().st_size])
    return {"path": _rel(d), "files": len(rows), "bytes": sum(r[2] for r in rows), "sha256": sha256_bytes(canonical(rows))}


def row_semantics(row_dict: Mapping) -> dict:
    """FrameResult dict with ONLY run_id removed (the exact comparison basis)."""
    out = dict(row_dict)
    out.pop("run_id", None)
    return out


def compare_rows(old: Sequence[Mapping], new: Sequence[Mapping]) -> dict:
    """Exact semantic equality per (role, frame), ignoring only run_id; also checks completeness/duplicates."""
    def index(rows, what):
        out = {}
        for r in rows:
            k = (r.get("backend_role"), r.get("frame_id"))
            if k in out:
                raise RegressionError(f"{what}: duplicate row {k}")
            out[k] = r
        return out
    a, b = index(old, "original"), index(new, "replay")
    missing, extra = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    differing = [k for k in sorted(set(a) & set(b)) if canonical(row_semantics(a[k])) != canonical(row_semantics(b[k]))]
    return {"rows_original": len(a), "rows_replay": len(b), "missing": [list(k) for k in missing[:10]],
            "extra": [list(k) for k in extra[:10]], "differing": [list(k) for k in differing[:10]],
            "differing_count": len(differing), "equal": not missing and not extra and not differing}


# --------------------------------------------------------------------------
# plan (read-only; no torch import)


def compare_versions(original: Any, current: Any) -> dict:
    """Exact runtime comparison of ``report.dependencies.versions`` with
    ``workbench_telemetry.dependency_versions()``. Missing or extra fields count as
    drift (fail closed); nothing is normalised."""
    if not isinstance(original, Mapping):
        return {"<versions>": {"original": None, "current": "present" if isinstance(current, Mapping) else None}}
    if not isinstance(current, Mapping):
        return {"<versions>": {"original": "present", "current": None}}
    return {k: {"original": original.get(k, "<missing>"), "current": current.get(k, "<missing>")}
            for k in sorted(set(original) | set(current)) if k not in original or k not in current
            or original[k] != current[k]}


def _module_hashes(names: Sequence[str]) -> dict:
    out = {}
    for name in names:
        p = ROOT / "bench" / f"{name}.py"
        out[name] = sha256_file(p) if p.is_file() else None
    return out


def shared_identity() -> dict:
    from bench import workbench_neural as wn, workbench_telemetry as wt
    head = ROOT / wn.DEFAULT_MODEL_DIR / "head.pt"
    base = ROOT / wn.BASE_CHECKPOINT_RELATIVE
    head_id, base_id = _file_id(head), _file_id(base)
    if head_id["sha256"] != HEAD_SHA256 or wn.PRIMARY["export_sha256"] != HEAD_SHA256:
        raise RegressionError("head.pt differs from the fixed head 27e06a51")
    if base_id["sha256"] != BASE_SHA256 or wn.PRIMARY["base_checkpoint_sha256"] != BASE_SHA256:
        raise RegressionError("base checkpoint differs from the fixed base f7d127ab")
    return {"head": head_id, "base_checkpoint": base_id,
            "current_key_modules": _module_hashes(KEY_MODULES), "current_report_modules": _module_hashes(REPORT_MODULES),
            "tool_sha256": sha256_file(Path(__file__)), "runtime": wt.dependency_versions(),
            "python_executable": _rel(Path(sys.executable)) if ROOT.resolve() in Path(sys.executable).resolve().parents
            else Path(sys.executable).name}


def case_identity(key: str, shared: Mapping) -> dict:
    """Bind one original run and judge eligibility. Never reads reference/evaluator/raw GT files."""
    from bench import workbench_cli as cli, workbench_neural as wn
    spec = CASES[key]
    ws = cli.Workspace(ROOT / WORKSPACE)
    run_dir = ws.runs_dir / spec["run_id"]
    files = {name: _file_id(run_dir / name) for name in ORIGINAL_FILES}
    run_doc = loads_strict((run_dir / "run.json").read_bytes(), "run.json")
    report = loads_strict((run_dir / "report.json").read_bytes(), "report.json")
    run = run_doc.get("run") or {}
    problems: list[str] = []
    if run.get("run_id") != spec["run_id"] or report.get("run_id") != spec["run_id"]:
        problems.append("run id differs inside run.json/report.json")
    if tuple(run.get("roles") or ()) != ROLES:
        problems.append(f"roles {run.get('roles')} are not exactly {list(ROLES)}")
    scheduled = run.get("scheduled_frame_ids") or []
    if run.get("mode") != "regression" or len(scheduled) != spec["frames"] or len(set(scheduled)) != len(scheduled):
        problems.append("not a full regression window with the expected unique frame count")
    if report.get("status") != "completed":
        problems.append(f"original report status {report.get('status')}")
    stages = report.get("stages") or {}
    for role in ROLES:
        if (stages.get(role) or {}).get("status") != "completed":
            problems.append(f"original {role} stage not completed")
    _run, rows, _rep = cli.load_results(ws, spec["run_id"])
    pairs = [(r.backend_role, r.frame_id) for r in rows]
    if len(pairs) != len(set(pairs)) or set(pairs) != {(r, f) for r in ROLES for f in scheduled} \
            or any(r.run_id != spec["run_id"] for r in rows):
        problems.append("original results are not exactly one row per role and scheduled frame")
    overlays = sorted({r.evidence.get("overlay") for r in rows if r.evidence.get("overlay")})
    overlay_rows = []
    for rel in overlays:
        p = run_dir / rel
        overlay_rows.append([rel, sha256_file(p) if p.is_file() else None])
    if any(h is None for _r, h in overlay_rows):
        problems.append("an original overlay referenced by results is missing")
    # admission attempt: contract SHA must equal the backend identity recorded by both roles
    contract = wn.load_contract(ROOT / spec["admission"] / "contract.json")
    recorded_contract = {(stages.get(r) or {}).get("identity", {}).get("admission_contract_sha256") for r in ROLES}
    if recorded_contract != {contract.sha256}:
        problems.append(f"admission {spec['admission']} contract {contract.sha256[:12]} differs from the recorded "
                        f"backend admission_contract_sha256 {sorted(map(str, recorded_contract))}")
    # metric bundle + GLB
    bundle = wn.MetricBundle.from_file(ROOT / spec["bundle"])
    glb_id = _file_id(Path(bundle.glb_path))
    recorded_glb = {(stages.get(r) or {}).get("identity", {}).get("glb_sha256") for r in ROLES}
    if glb_id["sha256"] != bundle.glb_sha256 or recorded_glb != {bundle.glb_sha256}:
        problems.append("bundle GLB differs from its pin or from the recorded backend glb_sha256")
    # inputs: clip record and mask manifest of the ORIGINAL config mask set
    clip_file = _file_id(ws.clip_path(run_doc.get("clip_id")))
    mask_set = (run.get("config") or {}).get("mask_set")
    mask_manifest = _file_id(ws.masks_dir / run_doc.get("clip_id") / str(mask_set) / "manifest.json") if mask_set else None
    if not mask_set:
        problems.append("original config records no mask_set")
    # source drift: original recorded module hashes vs current
    recorded = (report.get("dependencies") or {}).get("sources") or {}
    current = shared["current_report_modules"]
    drift = {m: {"original": recorded.get(m), "current": current.get(m)}
             for m in REPORT_MODULES if recorded.get(m) != current.get(m)}
    key_drift = sorted(m for m in drift if m in KEY_MODULES)
    unrecorded_key_modules = sorted(m for m in KEY_MODULES if m not in recorded)
    # runtime drift: original report.dependencies.versions must equal the current dependency_versions() EXACTLY
    runtime_drift = compare_versions((report.get("dependencies") or {}).get("versions"), shared["runtime"])
    eligible = not problems and not drift and not runtime_drift
    run_tree = _tree_id(run_dir)
    reason = None
    if problems:
        reason = "original_invalid: " + "; ".join(problems)
    elif drift:
        reason = (f"source_drift: original recorded {sorted(drift)} differ from current code"
                  + (f" (stage-key modules {key_drift}: the current key cannot equal the original key, so a cache hit "
                     "is impossible without the original sources)" if key_drift else ""))
    elif runtime_drift:
        reason = f"runtime_drift: original recorded runtime differs from the current runtime in {sorted(runtime_drift)}"
    return {"key": key, "run_id": spec["run_id"], "clip_id": run_doc.get("clip_id"), "frames": spec["frames"],
            "eligible": eligible, "ineligible_reason": reason,
            "original_files": files, "original_run_dir": run_tree,
            "overlay_evidence": {"count": len(overlay_rows), "sha256": sha256_bytes(canonical(overlay_rows))},
            "runspec_sha256": sha256_bytes(canonical(run)), "mask_set": mask_set, "mask_manifest": mask_manifest,
            "clip_record": clip_file, "admission": {"dir": spec["admission"], "contract_sha256": contract.sha256,
                                                     "closure": _tree_id(ROOT / spec["admission"])},
            "bundle": {**_file_id(ROOT / spec["bundle"]), "glb": glb_id},
            "source_drift": drift, "runtime_drift": runtime_drift,
            "unrecorded_key_modules": unrecorded_key_modules,
            "unrecorded_key_modules_note": "not recorded by the original report; equality is established only by an "
                                           "actual cache hit (the key binds them), never claimed as verified here",
            "prospective_output_bytes": int(run_tree["bytes"] * 1.1) + OUTPUT_MARGIN_BYTES}


def build_plan(keys: Sequence[str] = tuple(CASES)) -> dict:
    if "torch" in sys.modules:
        raise RegressionError("plan must be built before torch is imported")
    shared = shared_identity()
    return {"schema": PLAN_SCHEMA, "workspace": WORKSPACE, "shared": shared,
            "cases": {k: case_identity(k, shared) for k in keys},
            "claims": {"model_quality": False, "pooling": False, "head_superiority": False, "mobile": False,
                       "suite_promotion": False}}


# --------------------------------------------------------------------------
# bounded, immutable writes


def _write_new(path: Path, data: bytes, extra_derived: int = 0) -> str:
    from bench import workbench_cli as cli
    target = Path(os.path.abspath(path))
    _rel(target)
    if len(data) > MAX_JSON_BYTES:
        raise RegressionError(f"{target.name} exceeds {MAX_JSON_BYTES} bytes")
    if os.path.lexists(target):
        raise RegressionError(f"{_rel(target)} already exists; outputs are immutable")
    cli.Workspace(ROOT / WORKSPACE).require_budget(extra_derived=len(data) + extra_derived)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, target)
    except FileExistsError:
        raise RegressionError(f"{_rel(target)} already exists; outputs are immutable") from None
    finally:
        tmp.unlink(missing_ok=True)
    return sha256_bytes(data)


def write_plan(out: Path) -> dict:
    plan = build_plan()
    data = canonical(plan)
    sha = _write_new(out, data)
    return {"plan": _rel(Path(out)), "plan_sha256": sha,
            "eligible": [k for k, c in plan["cases"].items() if c["eligible"]],
            "ineligible": {k: c["ineligible_reason"] for k, c in plan["cases"].items() if not c["eligible"]}}


# --------------------------------------------------------------------------
# destination and closure checks (pure/runtime helpers)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_link_or_reparse(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    import stat as stat_mod
    return stat_mod.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def validate_destination(out_path: Path, *, reserve_bytes: int = 0) -> Path:
    """Validate the summary destination BEFORE any case runs.

    It must be a ``.json`` path that does not exist yet. Its parent must be an
    existing directory inside the repository, with no symlink or reparse point
    on the path from the repository root. The parent must be writable, which is
    checked with a fresh owned temp file that is removed straight away; no
    existing data is touched. The workspace budget (1 GiB derived limit, 1 GiB
    free reserve) and the parent disk must fit ``reserve_bytes`` plus the
    summary cap."""
    from bench import workbench_cli as cli, workbench_contract as wc
    target = Path(os.path.abspath(out_path))
    root = ROOT.resolve()
    if target.suffix != ".json" or not target.name or target.name.startswith("."):
        raise RegressionError("summary destination must be a visible .json file name")
    if root not in target.parents:
        raise RegressionError("summary destination is outside the repository")
    if os.path.lexists(target):
        raise RegressionError("summary destination already exists; outputs are immutable")
    parent = target.parent
    if not parent.is_dir():
        raise RegressionError("summary destination parent must be an existing directory")
    for part in [parent, *parent.parents]:
        if part == root.parent:
            break
        if _is_link_or_reparse(part):
            raise RegressionError(f"summary destination path traverses a link/reparse point: {part}")
    if parent.resolve() != parent:
        raise RegressionError("summary destination parent does not resolve to itself")
    probe = parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.probe"
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        os.close(fd)
    except OSError as exc:
        raise RegressionError(f"summary destination parent is not writable: {exc}") from None
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
    need = int(reserve_bytes) + MAX_JSON_BYTES
    try:
        cli.Workspace(ROOT / WORKSPACE).require_budget(extra_derived=need)
    except cli.WorkbenchCLIError as exc:
        raise RegressionError(f"budget/reserve cannot fit the planned outputs: {exc}") from None
    import shutil
    if shutil.disk_usage(parent).free - need < wc.FREE_RESERVE_BYTES:
        raise RegressionError("1 GiB free-disk reserve would be violated at the summary destination")
    return target


def closure_status(plan: Mapping, key: str) -> dict:
    """Recompute the COMPLETE shared identity (weights, current module hashes, runtime, tool) and the case
    identity (originals, inputs, admission, bundle, drift) and compare both with the plan."""
    out = {"shared_equal": False, "case_equal": False, "error": None}
    try:
        shared_now = shared_identity()
        out["shared_equal"] = canonical(shared_now) == canonical(plan["shared"])
        if not out["shared_equal"]:
            out["shared_differs"] = sorted(k for k in set(shared_now) | set(plan["shared"])
                                           if canonical(shared_now.get(k)) != canonical(plan["shared"].get(k)))
        case_now = case_identity(key, shared_now)
        out["case_equal"] = canonical(case_now) == canonical(plan["cases"][key])
        if not out["case_equal"]:
            out["case_differs"] = sorted(k for k in set(case_now) | set(plan["cases"][key])
                                         if canonical(case_now.get(k)) != canonical(plan["cases"][key].get(k)))
    except Exception as exc:  # noqa: BLE001 - an unverifiable closure is a failed closure
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out["ok"] = out["shared_equal"] and out["case_equal"] and out["error"] is None
    return out


def plan_file_unchanged(path: Path, expected_sha256: str) -> bool:
    try:
        return sha256_file(Path(path)) == expected_sha256
    except OSError:
        return False


def load_plan(path: Path, expected_sha256: str) -> dict:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or \
            any(c not in "0123456789abcdef" for c in expected_sha256):
        raise RegressionError("--expected-plan-sha256 must be a lowercase SHA-256")
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_JSON_BYTES:
        raise RegressionError("plan must be a regular bounded file")
    raw = p.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RegressionError("plan differs from the externally expected SHA-256")
    plan = loads_strict(raw, "plan")
    if plan.get("schema") != PLAN_SCHEMA or canonical(plan) != raw or set(plan.get("cases") or {}) != set(CASES):
        raise RegressionError("not a canonical cache-regression plan for the seven fixed cases")
    return plan


# --------------------------------------------------------------------------
# guards (restored on every path)


@contextmanager
def cache_only_guards() -> Iterator[list]:
    """Forbid every route to new inference or cache writes for the duration of a case."""
    from bench import workbench_cache as wcache, workbench_cli as cli, workbench_neural as wn, workbench_telemetry as wt
    attempts: list[str] = []
    saved: list[tuple[Any, str, Any]] = []

    def deny(what: str):
        def _deny(*_a, **_k):
            attempts.append(what)
            raise CacheOnlyViolation(what)
        return _deny

    def patch(owner, name, value):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    try:
        patch(wt, "gpu_stage", deny("cache miss: the pipeline tried to enter the GPU stage"))
        patch(wn, "GoTrackMetricBackend", deny("neural backend construction attempted"))
        patch(wn, "NeuralSession", deny("NeuralSession construction attempted"))
        patch(wn, "segment_prompted", deny("SAM segmentation attempted"))
        patch(cli.Pipeline, "_cache_store", deny("stage-cache store attempted"))
        patch(wcache.StageCache, "store", deny("stage-cache store attempted"))
        yield attempts
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


# --------------------------------------------------------------------------
# run one case


def _cuda_initialized() -> bool | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001
        return None


def verify_replay(case: Mapping, new_run_id: str) -> dict:
    """Read the new run's outputs and check them strictly against the original."""
    from bench import workbench_cli as cli
    ws = cli.Workspace(ROOT / WORKSPACE)
    _o_run, old_rows, old_report = cli.load_results(ws, case["run_id"])
    _n_run, new_rows, new_report = cli.load_results(ws, new_run_id)
    checks: dict[str, Any] = {}
    stages = new_report.get("stages") or {}
    checks["role_stages_cache_hit"] = all((stages.get(r) or {}).get("status") == "cache_hit" for r in ROLES)
    checks["no_failed_or_blocked_stage"] = not any(s.get("status") in ("failed", "blocked") for s in stages.values())
    checks["report_completed"] = new_report.get("status") == "completed"
    tel = new_report.get("telemetry") or {}
    checks["telemetry_cache_hits"] = all((tel.get("cache") or {}).get(r, {}).get("status") == "cache_hit" for r in ROLES)
    checks["zero_new_frame_inference"] = tel.get("frames") == [] and not tel.get("per_role_frame_timings")
    checks["no_role_initialization"] = not any(k.endswith("_initialization") for k in (tel.get("stages") or {}))
    checks["no_gpu_used"] = tel.get("gpu_used") is False
    checks["cached_historical_timings_present"] = all(
        (tel.get("cached_original_timings") or {}).get(r, {}).get("source") == "cached_original_run" for r in ROLES)
    comparison = compare_rows([r.to_dict() for r in old_rows], [r.to_dict() for r in new_rows])
    checks["rows_equal_except_run_id"] = comparison["equal"]
    checks["frame_accounting_complete"] = comparison["rows_replay"] == len(ROLES) * case["frames"]
    old_dir, new_dir = ws.runs_dir / case["run_id"], ws.runs_dir / new_run_id
    overlays = sorted({r.evidence.get("overlay") for r in new_rows if r.evidence.get("overlay")})
    bad = [rel for rel in overlays if not (new_dir / rel).is_file() or not (old_dir / rel).is_file()
           or sha256_file(new_dir / rel) != sha256_file(old_dir / rel)]
    checks["overlay_evidence_equal"] = not bad and len(overlays) == case["overlay_evidence"]["count"]
    art_old, art_new = old_dir / "neural-artwork.json", new_dir / "neural-artwork.json"
    checks["artwork_record_equal"] = art_new.is_file() and art_new.read_bytes() == art_old.read_bytes()
    return {"checks": checks, "passed": all(checks.values()), "row_comparison": comparison,
            "overlay_mismatches": bad[:10], "overlays_compared": len(overlays),
            "cached_historical_timings": {
                "label": "cached historical timings from the ORIGINAL run (not inference latency of this replay)",
                "per_role": (old_report.get("telemetry") or {}).get("per_role_frame_timings")},
            "replay_cache": tel.get("cache"), "replay_run_cache_mode": tel.get("run_cache_mode")}


def _fail(out: dict, reason: str) -> None:
    """Record a failure reason without discarding an earlier one (guard/closure reasons are preserved)."""
    out["status"] = "failed"
    out["reason"] = f"{out['reason']}; {reason}" if out.get("reason") else reason


def run_case(plan: Mapping, key: str) -> dict:
    """One serial, guarded cache-only case. The outcome is decided only AFTER the
    complete shared+case closure has been re-checked, on every execution path."""
    from bench import workbench_cli as cli, workbench_contract as wc, workbench_neural as wn
    case = plan["cases"][key]
    out: dict[str, Any] = {"key": key, "original_run_id": case["run_id"], "status": "failed", "new_run_id": None,
                           "reason": None, "guard_attempts": []}
    if not case["eligible"]:
        return {**out, "status": "ineligible", "reason": case["ineligible_reason"]}
    pre = closure_status(plan, key)
    out["closure_before"] = pre
    if not pre["ok"]:
        _fail(out, "closure before execution differs from the plan")
        return out
    executed = False
    attempts: list[str] = []
    try:
        ws = cli.Workspace(ROOT / WORKSPACE)
        try:
            ws.require_budget(extra_derived=case["prospective_output_bytes"] + MAX_JSON_BYTES)
        except cli.WorkbenchCLIError as exc:
            out["status"], out["reason"] = "blocked", f"budget: {exc}"
            return out                                        # nothing executed; closure-after still runs below
        run_doc = loads_strict((ws.runs_dir / case["run_id"] / "run.json").read_bytes(), "run.json")
        clip = ws.load_clip(run_doc["clip_id"])
        new_run_id = f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run_spec = wc.RunSpec.from_dict({**run_doc["run"], "run_id": new_run_id})
        run_spec.validate_against(clip)
        out["new_run_id"] = new_run_id
        admission = wn.load_admission(ROOT / CASES[key]["admission"])
        if not admission.admitted or admission.body.get("contract_sha256") != case["admission"]["contract_sha256"]:
            _fail(out, "admission did not re-verify for the recorded contract")
            return out
        options = cli.RunOptions(mask_set=case["mask_set"], head_path=str(ROOT / plan["shared"]["head"]["path"]),
                                 admission=admission, metric_bundle=str(ROOT / CASES[key]["bundle"]), device="cuda")
        messages: list[str] = []
        started = time.perf_counter()
        try:
            executed = True
            with cache_only_guards() as attempts:
                result = cli.execute_run(ws, clip, run_spec, options, progress=lambda m: messages.append(str(m)[:300]))
        except CacheOnlyViolation as exc:
            out["current_wall_ms"] = (time.perf_counter() - started) * 1000.0
            out["progress"] = messages[-20:]
            _fail(out, f"cache-only guard: {exc}")
            return out
        out["current_wall_ms"] = (time.perf_counter() - started) * 1000.0
        out["current_wall_label"] = "measured wall time of this cache-only replay (not inference latency)"
        out["pipeline_status"] = result.get("status")
        out["cuda_initialized"] = _cuda_initialized()
        verification = verify_replay(case, new_run_id)
        out["verification"] = verification
        if not verification["passed"]:
            _fail(out, "replay checks failed: " + ", ".join(k for k, v in verification["checks"].items() if not v))
        elif attempts:
            _fail(out, "guard attempts recorded")
        elif out["cuda_initialized"] not in (None, False):
            _fail(out, "CUDA was initialised")
        else:
            out["status"] = "verified_pending_closure"
        return out
    except Exception as exc:  # noqa: BLE001 - ordinary errors are recorded, never flattened into success
        _fail(out, f"{type(exc).__name__}: {str(exc)[:500]}")
        return out
    finally:
        out["guard_attempts"] = list(attempts)
        out["executed"] = executed
        post = closure_status(plan, key)                    # every path: success, guard, ordinary error, budget
        out["closure_after"] = post
        if not post["ok"]:
            _fail(out, "closure after execution differs from the plan "
                       f"(shared {post.get('shared_differs')}, case {post.get('case_differs')}, error {post['error']})")
        elif out["status"] == "verified_pending_closure":
            out["status"] = "completed"


STATUSES = ("completed", "failed", "ineligible", "blocked", "not_run")


def run(plan_path: Path, expected_sha256: str, out_path: Path, case_key: str | None = None) -> dict:
    plan = load_plan(plan_path, expected_sha256)
    if sha256_file(Path(__file__)) != plan["shared"]["tool_sha256"]:
        raise RegressionError("tool differs from the version that built the plan")
    if case_key and case_key not in CASES:
        raise RegressionError(f"unknown case {case_key!r}; choose from {list(CASES)}")
    keys = [case_key] if case_key else list(CASES)
    reserve = sum(plan["cases"][k]["prospective_output_bytes"] for k in keys if plan["cases"][k]["eligible"])
    target = validate_destination(out_path, reserve_bytes=reserve)    # before ANY case or pipeline write
    results: dict[str, dict] = {}
    plan_ok = True
    for key in keys:                                         # serial; no GPU stage is ever entered
        if not plan_file_unchanged(plan_path, expected_sha256):
            plan_ok = False
            results[key] = {"key": key, "status": "not_run", "reason": "pinned plan file changed before this case"}
            continue
        try:
            results[key] = run_case(plan, key)
        except BaseException as exc:  # noqa: BLE001 - keep a reviewable record, then stop
            results[key] = {"key": key, "status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}
            if not isinstance(exc, Exception):
                break
    if not plan_file_unchanged(plan_path, expected_sha256):     # after the last case, before the summary
        plan_ok = False
    if not plan_ok:
        for r in results.values():
            if r["status"] == "completed":
                r["status"] = "failed"
                r["reason"] = "pinned plan file changed during the replay"
    for key in keys:
        results.setdefault(key, {"key": key, "status": "not_run", "reason": "stopped before this case"})
    counts = {s: sum(1 for r in results.values() if r["status"] == s) for s in STATUSES}
    summary = {"schema": SUMMARY_SCHEMA, "plan_sha256": expected_sha256, "plan_file_unchanged": plan_ok,
               "requested_cases": keys, "cases": results, "counts": counts,
               "all_requested_completed": plan_ok and counts["completed"] == len(keys),
               "claims": plan["claims"],
               "note": "cache reuse of the unchanged pipeline only; cached timings are historical, not inference"}
    data = canonical(summary)
    try:
        sha = _write_new(target, data)
    except Exception as exc:  # noqa: BLE001 - never lose the record silently
        sys.stderr.write("summary could not be written; terminal record:\n" + data.decode("ascii") + "\n")
        raise RegressionError(f"summary not persisted: {exc}") from exc
    return {"summary": _rel(target), "summary_sha256": sha, "counts": counts, "plan_file_unchanged": plan_ok,
            "requested_cases": keys}


def exit_code(result: Mapping) -> int:
    """0 only when every requested case completed and the pinned plan stayed unchanged."""
    counts = result.get("counts") or {}
    bad = sum(int(counts.get(s, 0)) for s in STATUSES if s != "completed")
    requested = result.get("requested_cases")
    complete = int(counts.get("completed", 0)) == len(requested) if requested is not None else int(counts.get("completed", 0)) > 0
    return 0 if bad == 0 and complete and result.get("plan_file_unchanged", True) is not False else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache-only regression replay of the seven fixed paired runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("plan", help="read-only binding of the seven originals; writes one fresh plan")
    s.add_argument("--out", required=True)
    s = sub.add_parser("run", help="guarded cache-only replay (no inference, no GPU stage)")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--case", choices=sorted(CASES))
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = write_plan(Path(args.out))       # eligibility findings are data, not a format error
            print(json.dumps(result, indent=2))
            return 0
        result = run(Path(args.plan), args.expected_plan_sha256, Path(args.out), args.case)
    except (RegressionError, OSError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
