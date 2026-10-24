"""Offline paired-comparison report for existing workbench runs (no inference).

Reads one or more finished, failed or partial run directories through the
existing read-only workbench APIs (``Workspace.load_clip``, ``load_results``,
``workbench_contract`` and ``workbench_evaluate``) and writes an immutable
JSON report, a Markdown summary and, only on request, a PNG chart.

Each run is reported separately per backend role. Nothing is pooled across
clips or runs and nothing is ranked. Every scheduled frame stays in the
denominator: a missing row counts as unavailable and never improves a
fraction. Pose availability (``tracking`` rows) and the raw artwork render
fraction (``rendered`` rows) are reported separately. Neither one is an
accuracy measure.

Attachment error, wrong-surface events and recovery correctness stay
``null`` and unverified unless the evaluator has independent annotations.
This tool never supplies annotations. The roles run serially on a shared
host, so latency differences are descriptive and do not show a causal
speed-up.

The primary role counts as a verified neural comparison only when every
primary row reports the actually installed head ``27e06a51…b3fb9311`` and the
evaluator confirms that a completed or verified-cache stage ran. The requested
head in the run metadata is reported separately and never substitutes for the
installed head. Rows that claim a tracking pose without that installed head on
every row (spoofed or missing identity) are rejected. Rows that carry the head
but have no confirmed stage, such as a crashed run without ``report.json``, are
still counted, but the run is marked unverified. Both neural roles must also
report the pinned base checkpoint ``f7d127ab…2a01086`` on every row. A role
whose rows claim a pose with a wrong or missing base is rejected. Identity
verification uses only the rows' actual identity and never the requested run
metadata. Blocked or all-unavailable rows that claim no pose remain unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import secrets
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import workbench_contract as wc  # noqa: E402

SCHEMA = "workbench_paired_comparison_v1"
TOOL_VERSION = "1"
PRIMARY_HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
PINNED_BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
SCOPES = ("smoke", "full")
LATENCY_METRICS = ("end_to_end_latency_ms", "tracking_ms", "rendering_ms")
UNVERIFIED_METRICS = ("attachment_median_px_720", "attachment_p95_px_720", "wrong_surface_events",
                      "correct_recovery_fraction")
MAX_RUNS = 32
EXIT_OK, EXIT_ERROR = 0, 2


class ComparisonError(Exception):
    """The inputs cannot support an honest comparison report; nothing is written."""


# --------------------------------------------------------------------------
# pure helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def percentile_linear(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (the NumPy default / Hyndman-Fan type 7)."""
    data = sorted(float(v) for v in values)
    if not data:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must be within [0, 100]")
    position = (len(data) - 1) * q / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(data) - 1)
    return data[lower] + (position - lower) * (data[upper] - data[lower])


def latency_summary(values: Sequence[float | None], scheduled: int) -> dict:
    measured = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"measured_frames": len(measured), "scheduled_frames": scheduled,
            "median_ms": percentile_linear(measured, 50.0), "p95_ms": percentile_linear(measured, 95.0),
            "max_ms": max(measured) if measured else None,
            "method": "linear interpolation (NumPy default) over measured per-frame values only"}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_role(role: str, scheduled: Sequence[str], rows: Sequence[Any], *,
                   cache_status: str | None = None, cached_original: Any = None) -> dict:
    """Count every scheduled frame for one role. ``rows`` are validated FrameResults."""
    by_frame = {row.frame_id: row for row in rows}
    if len(by_frame) != len(rows):
        raise ComparisonError(f"{role}: duplicate frame rows")
    missing = [fid for fid in scheduled if fid not in by_frame]
    n = len(scheduled)
    pose = Counter(row.pose_state for row in rows)
    render = Counter(row.render_state for row in rows)
    mask = Counter(row.mask_state for row in rows)
    reasons = Counter((row.failure_reason or "").split(":", 1)[0][:96] for row in rows if row.failure_reason)
    tracking, rendered = pose.get("tracking", 0), render.get("rendered", 0)
    latency_source = ("cached_original_run" if cache_status == "cache_hit"
                      else "measured_this_run" if rows else "unavailable")
    return {
        "role": role,
        "scheduled_frames": n,
        "result_rows": len(rows),
        "missing_rows": len(missing),
        "missing_frame_ids": missing,
        "pose_state_counts": {state: pose.get(state, 0) for state in wc.POSE_STATES},
        "render_state_counts": {state: render.get(state, 0) for state in wc.RENDER_STATES},
        "mask_state_counts": {state: mask.get(state, 0) for state in wc.MASK_STATES},
        "pose_available_frames": tracking,
        "pose_availability_fraction": _fraction(tracking, n),
        "artwork_rendered_frames": rendered,
        "raw_artwork_render_fraction": _fraction(rendered, n),
        "recovering_frames": pose.get("recovering", 0),
        "lost_frames": pose.get("lost", 0),
        "unavailable_frames_including_missing": pose.get("unavailable", 0) + pose.get("blocked", 0) + len(missing),
        "failure_reason_prefixes": dict(sorted(reasons.items())),
        "latency": {
            "source": latency_source,
            "cache_status": cache_status,
            **{name: latency_summary([row.metrics.get(name) for row in rows], n) for name in LATENCY_METRICS},
        },
        "cached_original_timings": cached_original,
        "definitions": {
            "pose_availability_fraction": "tracking rows / all scheduled frames (missing rows count as unavailable)",
            "raw_artwork_render_fraction": "rendered rows / all scheduled frames; not correctness or expected visibility",
        },
    }


def paired_frames(scheduled: Sequence[str], primary: Sequence[Any], baseline: Sequence[Any]) -> dict:
    """Frame-by-frame agreement of pose availability; missing rows count as unavailable."""
    p = {row.frame_id: row.pose_state == "tracking" for row in primary}
    b = {row.frame_id: row.pose_state == "tracking" for row in baseline}
    cells = Counter((p.get(fid, False), b.get(fid, False)) for fid in scheduled)
    pr = {row.frame_id: row.render_state == "rendered" for row in primary}
    br = {row.frame_id: row.render_state == "rendered" for row in baseline}
    rcells = Counter((pr.get(fid, False), br.get(fid, False)) for fid in scheduled)
    return {
        "scheduled_frames": len(scheduled),
        "pose_both_available": cells[(True, True)], "pose_primary_only": cells[(True, False)],
        "pose_baseline_only": cells[(False, True)], "pose_neither": cells[(False, False)],
        "render_both": rcells[(True, True)], "render_primary_only": rcells[(True, False)],
        "render_baseline_only": rcells[(False, True)], "render_neither": rcells[(False, False)],
        "note": "Descriptive agreement only. Availability does not establish correct attachment.",
    }


def identity_check(role: str, rows: Sequence[Any], executed: bool, requested_head: str | None) -> dict:
    """Validate the actual row identities for one role; raise on spoofed or missing identity.

    Both neural roles must report the pinned base checkpoint on every row before
    any pose claim is accepted. Only the rows' own identity is used; requested
    run metadata never fills in a missing base or head.
    """
    identities = {wc.canonical_sha256(row.backend_identity) for row in rows}
    installed = {row.backend_identity.get("installed_head_sha256") for row in rows}
    bases = {row.backend_identity.get("base_checkpoint_sha256") for row in rows}
    base_ok = bool(rows) and bases == {PINNED_BASE_SHA256}
    claims_pose = any(row.pose_state == "tracking" for row in rows)
    for row in rows:
        if row.backend_identity.get("role") not in (None, role):
            raise ComparisonError(f"{role}: row {row.frame_id} identity names role {row.backend_identity.get('role')}")
    if role in ("primary_neural", "frozen_baseline") and claims_pose and not base_ok:
        raise ComparisonError(f"{role} rows claim tracking poses without the pinned base checkpoint "
                              f"{PINNED_BASE_SHA256} on every row (actual: {sorted(map(str, bases))})")
    row_bases = sorted(map(str, bases))
    if role == "primary_neural":
        head_ok = installed == {PRIMARY_HEAD_SHA256}
        verified = bool(rows) and executed and head_ok and base_ok
        if claims_pose and not head_ok:
            raise ComparisonError(
                "primary_neural rows claim tracking poses without the installed head "
                f"{PRIMARY_HEAD_SHA256} on every row (installed: {sorted(map(str, installed))})")
        status = ("verified_installed_head" if verified
                  else "row_identity_only_stage_unconfirmed" if rows and head_ok and base_ok
                  else "unavailable_no_verified_execution")
        return {"identity_verified": verified,
                "installed_head_sha256": PRIMARY_HEAD_SHA256 if verified else None,
                "row_installed_head_sha256": sorted(map(str, installed)),
                "base_checkpoint_sha256": PINNED_BASE_SHA256 if base_ok else None,
                "row_base_checkpoint_sha256": row_bases,
                "requested_head_sha256_from_run_metadata": requested_head,
                "requested_matches_installed": verified and requested_head == PRIMARY_HEAD_SHA256,
                "distinct_row_identities": len(identities),
                "status": status}
    if role == "frozen_baseline":
        if installed - {None}:
            raise ComparisonError(f"frozen_baseline rows claim an installed new head {sorted(installed - {None})}")
        heads = {row.backend_identity.get("head") for row in rows}
        if claims_pose and heads != {"frozen_base_pose_head"}:
            raise ComparisonError(f"frozen_baseline rows claim poses with unexpected head identity {sorted(map(str, heads))}")
        head_ok = bool(rows) and heads == {"frozen_base_pose_head"}
        verified = head_ok and base_ok
        return {"identity_verified": verified,
                "installed_head_sha256": None, "head": "frozen_base_pose_head" if head_ok else None,
                "base_checkpoint_sha256": PINNED_BASE_SHA256 if base_ok else None,
                "row_base_checkpoint_sha256": row_bases,
                "status": "verified_pinned_frozen_base" if verified else "unavailable_unverified_baseline_identity",
                "distinct_row_identities": len(identities)}
    return {"identity_verified": None, "distinct_row_identities": len(identities)}


# --------------------------------------------------------------------------
# collection (read-only)


def _read(path: Path, files: list) -> bytes:
    data = path.read_bytes()
    files.append({"path": path.resolve().as_posix(), "bytes": len(data), "sha256": sha256_bytes(data)})
    return data


def _parse_rows(raw: bytes) -> tuple[list, int]:
    rows, invalid = [], 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(wc.FrameResult.from_dict(wc.loads_strict(line)))
        except (wc.ContractError, ValueError, TypeError):
            invalid += 1
    return rows, invalid


def collect_run(workspace: Path, run_id: str, scope: str) -> dict:
    """Read one run directory and validate it against its registered clip."""
    from bench import workbench_cli as cli
    from bench import workbench_evaluate as ev

    if scope not in SCOPES:
        raise ComparisonError(f"scope must be one of {SCOPES}")
    wc._ident(run_id, "run_id")
    ws = cli.Workspace(workspace)
    run_dir = ws.runs_dir / run_id
    if not run_dir.is_dir():
        raise ComparisonError(f"run directory not found: {run_dir}")
    files: list[dict] = []
    run_doc = wc.loads_strict(_read(run_dir / "run.json", files))
    if not isinstance(run_doc, dict) or "run" not in run_doc:
        raise ComparisonError(f"{run_id}: run.json lacks the sealed run specification")
    run = wc.RunSpec.from_dict(run_doc["run"])
    if run.run_id != run_id or run_doc.get("clip_id") != run.clip_id:
        raise ComparisonError(f"{run_id}: run identity differs from its directory or wrapper")
    _read(ws.clip_path(run.clip_id), files)
    clip = ws.load_clip(run.clip_id)
    run.validate_against(clip)
    expected = wc.schedule_frames(clip, "regression" if scope == "full" else "smoke", run.onboarding)
    expected_mode = "regression" if scope == "full" else "smoke"
    if run.mode != expected_mode or tuple(run.scheduled_frame_ids) != tuple(expected):
        raise ComparisonError(f"{run_id}: run (mode {run.mode}, {len(run.scheduled_frame_ids)} frames) is not "
                              f"the {scope} scope of clip {clip.clip_id} ({len(expected)} frames)")

    results_path, report_path = run_dir / "results.jsonl", run_dir / "report.json"
    parse = {"results_file": results_path.is_file(), "report_file": report_path.is_file(), "invalid_lines": 0}
    rows: list = []
    report: dict = {}
    if parse["results_file"]:
        raw = _read(results_path, files)
        try:
            _, rows, report = cli.load_results(ws, run_id)
            parse["loader"] = "workbench_cli.load_results"
        except (wc.ContractError, ValueError, OSError):
            rows, parse["invalid_lines"] = _parse_rows(raw)
            parse["loader"] = "per_line_fallback (invalid lines counted, their frames stay unavailable)"
    if parse["report_file"]:
        report_bytes = _read(report_path, files)
        if not report:
            try:
                report = wc.loads_strict(report_bytes)
            except (wc.ContractError, ValueError):
                report, parse["report_file"] = {}, False

    scheduled = list(run.scheduled_frame_ids)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.backend_role, row.frame_id)
        if row.run_id != run.run_id or row.clip_id != clip.clip_id:
            raise ComparisonError(f"{run_id}: row {key} belongs to another run or clip")
        if row.backend_role not in run.roles:
            raise ComparisonError(f"{run_id}: row role {row.backend_role} was not scheduled")
        if row.frame_id not in scheduled:
            raise ComparisonError(f"{run_id}: row frame {row.frame_id} was not scheduled")
        if key in seen:
            raise ComparisonError(f"{run_id}: duplicate row identity {key}")
        if row.source_frame_index != clip.frame(row.frame_id).source_frame_index:
            raise ComparisonError(f"{run_id}: row {key} source frame index differs from the clip")
        seen.add(key)

    telemetry = report.get("telemetry") if isinstance(report.get("telemetry"), Mapping) else {}
    cache = telemetry.get("cache") if isinstance(telemetry.get("cache"), Mapping) else {}
    cached_original = telemetry.get("cached_original_timings") if isinstance(
        telemetry.get("cached_original_timings"), Mapping) else {}
    roles: dict[str, dict] = {}
    for role in run.roles:
        role_rows = [row for row in rows if row.backend_role == role]
        executed = False
        if role == "primary_neural":
            try:
                executed = bool(ev.evaluate(clip, run, role_rows, role=role, run_evidence=report)["inference_executed"])
            except ValueError as exc:
                raise ComparisonError(f"{run_id}: evaluator rejected primary rows: {exc}") from exc
        status = cache.get(role, {}).get("status") if isinstance(cache.get(role), Mapping) else None
        summary = summarize_role(role, scheduled, role_rows, cache_status=status,
                                 cached_original=cached_original.get(role))
        summary["identity"] = identity_check(role, role_rows, executed, run.primary.get("export_sha256"))
        stage = report.get("stages", {}).get(role) if isinstance(report.get("stages"), Mapping) else None
        summary["stage_status"] = stage.get("status") if isinstance(stage, Mapping) else None
        summary["report_telemetry_summary"] = (telemetry.get("per_role_frame_timings") or {}).get(role)
        roles[role] = summary

    paired = None
    if {"primary_neural", "frozen_baseline"} <= set(roles):
        paired = paired_frames(scheduled, [r for r in rows if r.backend_role == "primary_neural"],
                               [r for r in rows if r.backend_role == "frozen_baseline"])
    complete = all(r["missing_rows"] == 0 for r in roles.values()) and parse["invalid_lines"] == 0
    neural = roles.get("primary_neural", {}).get("identity", {}).get("identity_verified", False)
    baseline_ok = roles.get("frozen_baseline", {}).get("identity", {}).get("identity_verified", False)
    return {
        "run_id": run.run_id, "clip_id": clip.clip_id, "scope": scope, "mode": run.mode,
        "scheduled_frames": len(scheduled), "roles_scheduled": list(run.roles),
        "role_order_note": "roles execute serially on a shared host; latency differences are not causal",
        "report_status": report.get("status"),
        "run_completeness": "complete" if complete and parse["report_file"] else "partial_or_failed",
        "neural_comparison": ("verified_primary_vs_frozen_baseline" if neural and baseline_ok and paired
                              else "unavailable_primary_identity_or_execution_unverified" if not neural
                              else "unavailable_baseline_identity_unverified" if paired
                              else "primary_only_no_baseline"),
        "parse": parse,
        "roles": roles,
        "paired_frames": paired,
        "run_telemetry": {key: telemetry.get(key) for key in (
            "run_wall_ms", "run_cache_mode", "process_peak_rss_bytes", "gpu_used",
            "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")} if telemetry else None,
        "accuracy": {"status": "unverified_no_independent_annotations",
                     **{name: None for name in UNVERIFIED_METRICS},
                     "note": "Availability and render fractions are not accuracy; no value is derived from them."},
        "input_files": files,
    }


def build_report(runs: Sequence[dict], scope: str, command: Sequence[str]) -> dict:
    return {
        "schema": SCHEMA, "tool": "tools/workbench_comparison.py", "tool_version": TOOL_VERSION,
        "tool_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "scope": scope, "command": list(command),
        "pinned_primary_head_sha256": PRIMARY_HEAD_SHA256,
        "pinned_base_checkpoint_sha256": PINNED_BASE_SHA256,
        "claims": {"pooled_quality": False, "leaderboard": False, "accuracy": False, "causal_speed": False,
                   "mobile_readiness": False, "new_inference": False},
        "note": "Each run and clip is reported separately; nothing is pooled or ranked.",
        "runs": list(runs),
    }


# --------------------------------------------------------------------------
# rendering


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# Presentation-only labels for already-recorded telemetry. The exported JSON keeps the raw values.
# A legacy ``gpu_used`` boolean proves neither per-run GPU work nor GPU idleness, and the process
# memory peaks are process-scoped observations, not per-run peaks.
MEMORY_SCOPE_NOTE = "Process memory observations do not establish GPU work or per-run memory peaks."


def legacy_gpu_label(value: Any) -> str:
    """Only a real JSON boolean is shown (no coercion of 0/1/strings); missing/other -> not recorded."""
    if value is True:
        return "true (recorded legacy value)"
    if value is False:
        return "false (recorded legacy value)"
    return "not recorded"


def recorded_bytes_label(value: Any) -> str:
    """A valid non-negative integer byte count (zero is a valid zero), otherwise 'not recorded'.
    Bool, negative, non-finite, fractional and unsafe (> 2**53 - 1, as in JS) values are invalid."""
    if isinstance(value, bool):
        return "not recorded"
    if isinstance(value, float):
        if not (math.isfinite(value) and value.is_integer()):
            return "not recorded"
        value = int(value)
    if not isinstance(value, int) or value < 0 or value > 2 ** 53 - 1:
        return "not recorded"
    return f"{value} B"


def telemetry_scope_sentence(tel: Mapping[str, Any]) -> str:
    return (f"Run telemetry (shared by both serial roles): wall {_fmt(tel.get('run_wall_ms'), 0)} ms, "
            f"cache mode {_fmt(tel.get('run_cache_mode'))}; "
            f"host process peak RSS (not a per-run peak): {recorded_bytes_label(tel.get('process_peak_rss_bytes'))}; "
            f"legacy GPU field (run scope unverified): {legacy_gpu_label(tel.get('gpu_used'))}; "
            f"process CUDA allocator peak allocated (cumulative; epoch unrecorded): "
            f"{recorded_bytes_label(tel.get('gpu_peak_allocated_bytes'))}; "
            f"process CUDA allocator peak reserved (cumulative; epoch unrecorded): "
            f"{recorded_bytes_label(tel.get('gpu_peak_reserved_bytes'))}. {MEMORY_SCOPE_NOTE}")


def render_markdown(report: dict) -> str:
    lines = [f"# Workbench paired comparison ({report['scope']} scope)", "",
             "Offline report from existing run artifacts. No new inference was run. Each run is reported "
             "separately, with no pooling or ranking. Availability and artwork rendering are **not** accuracy. "
             "Attachment error, wrong-surface events and recovery correctness are **unverified**, because there "
             "are no independent annotations. The roles ran serially on a shared host, so latency differences "
             "are not causal.", ""]
    for run in report["runs"]:
        lines += [f"## {run['run_id']} — clip `{run['clip_id']}`", "",
                  f"- Scope **{run['scope']}** (mode `{run['mode']}`), {run['scheduled_frames']} scheduled frames per role; "
                  f"completeness: {run['run_completeness']}; report status: {_fmt(run['report_status'])}",
                  f"- Neural comparison: **{run['neural_comparison']}**", ""]
        roles = run["roles"]
        header = "| Measure | " + " | ".join(roles) + " |"
        lines += [header, "|" + " --- |" * (len(roles) + 1)]

        def row(label, fn):
            lines.append(f"| {label} | " + " | ".join(fn(r) for r in roles.values()) + " |")

        row("Installed head", lambda r: (r["identity"].get("installed_head_sha256") or "none")[:12])
        row("Identity status", lambda r: str(r["identity"].get("status", r["identity"].get("identity_verified"))))
        row("Pose available (tracking)", lambda r: f"{r['pose_available_frames']}/{r['scheduled_frames']} "
                                                  f"({_fmt(r['pose_availability_fraction'] and 100 * r['pose_availability_fraction'])}%)")
        row("Artwork rendered (raw)", lambda r: f"{r['artwork_rendered_frames']}/{r['scheduled_frames']} "
                                               f"({_fmt(r['raw_artwork_render_fraction'] and 100 * r['raw_artwork_render_fraction'])}%)")
        row("Recovering", lambda r: str(r["recovering_frames"]))
        row("Lost", lambda r: str(r["lost_frames"]))
        row("Unavailable incl. missing", lambda r: str(r["unavailable_frames_including_missing"]))
        row("Missing rows", lambda r: str(r["missing_rows"]))
        row("Latency source", lambda r: r["latency"]["source"])
        for name in LATENCY_METRICS:
            row(f"{name} median / P95 (n)", lambda r, n=name: (
                f"{_fmt(r['latency'][n]['median_ms'])} / {_fmt(r['latency'][n]['p95_ms'])} "
                f"({r['latency'][n]['measured_frames']})"))
        row("Attachment error / wrong surface / recovery", lambda r: "unverified")
        lines.append("")
        if run["paired_frames"]:
            p = run["paired_frames"]
            lines += [f"Paired pose availability per frame: both {p['pose_both_available']}, primary only "
                      f"{p['pose_primary_only']}, baseline only {p['pose_baseline_only']}, neither {p['pose_neither']}. "
                      f"Paired rendering: both {p['render_both']}, primary only {p['render_primary_only']}, "
                      f"baseline only {p['render_baseline_only']}, neither {p['render_neither']}.", ""]
        tel = run["run_telemetry"]
        if tel:
            lines += [telemetry_scope_sentence(tel), ""]
        lines += ["Input files:", ""] + [f"- `{f['path']}` {f['bytes']} B sha256 `{f['sha256']}`"
                                          for f in run["input_files"]] + [""]
    return "\n".join(lines)


def render_png(report: dict) -> bytes:
    """Grouped availability chart plus per-role latency panel; imports matplotlib lazily."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415

    runs = report["runs"]
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 4.6 * len(runs)), squeeze=False)
    colors = {"primary_neural": "#2f6db3", "frozen_baseline": "#9a9a9a", "relative_geometry": "#c27c2c"}
    for i, run in enumerate(runs):
        roles = list(run["roles"].values())
        ax = axes[i][0]
        categories = [("Pose available", "pose_available_frames"), ("Artwork rendered", "artwork_rendered_frames"),
                      ("Recovering", "recovering_frames"), ("Lost", "lost_frames"),
                      ("Unavailable\n+missing", "unavailable_frames_including_missing")]
        width = 0.8 / max(len(roles), 1)
        for j, r in enumerate(roles):
            xs = [k + (j - (len(roles) - 1) / 2) * width for k in range(len(categories))]
            values = [r[key] for _, key in categories]
            bars = ax.bar(xs, values, width, label=r["role"], color=colors.get(r["role"], "#555555"))
            for bar, value in zip(bars, values):
                ax.annotate(f"{value}/{r['scheduled_frames']}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c for c, _ in categories], fontsize=8)
        ax.set_ylabel("frames (of scheduled)")
        ax.set_title(f"{run['run_id']} | clip {run['clip_id']} | {run['scope']} scope, "
                     f"{run['scheduled_frames']} frames/role\n{run['neural_comparison']}", fontsize=9)
        ax.legend(fontsize=8)
        ax.text(0.99, 0.97, "Attachment accuracy: UNVERIFIED\n(no independent annotations;\navailability is not accuracy)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox={"facecolor": "#fff6d6", "edgecolor": "#c9a400"})
        ax2 = axes[i][1]
        labels, medians, p95s = [], [], []
        for r in roles:
            lat = r["latency"]["end_to_end_latency_ms"]
            labels.append(f"{r['role']}\nn={lat['measured_frames']}/{lat['scheduled_frames']}\n{r['latency']['source']}")
            medians.append(lat["median_ms"])
            p95s.append(lat["p95_ms"])
        xs = range(len(labels))
        b1 = ax2.bar([x - 0.2 for x in xs], [v or 0.0 for v in medians], 0.4, label="median", color="#6aa84f")
        b2 = ax2.bar([x + 0.2 for x in xs], [v or 0.0 for v in p95s], 0.4, label="P95", color="#e69138")
        for bars, values in ((b1, medians), (b2, p95s)):
            for bar, value in zip(bars, values):
                ax2.annotate("n/a" if value is None else f"{value:.0f}",
                             (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha="center", va="bottom", fontsize=7)
        ax2.set_xticks(list(xs))
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_ylabel("end-to-end latency per frame (ms)")
        ax2.set_title("Measured per-frame latency (serial roles, shared host; not causal)", fontsize=9)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# publication


def output_paths(prefix: Path, png: bool) -> dict[str, Path]:
    prefix = Path(os.path.abspath(prefix))
    paths = {"json": prefix.with_name(prefix.name + ".json"), "md": prefix.with_name(prefix.name + ".md")}
    if png:
        paths["png"] = prefix.with_name(prefix.name + ".png")
    return paths


def refuse_existing(paths: Mapping[str, Path]) -> None:
    existing = [str(p) for p in paths.values() if os.path.lexists(p)]
    if existing:
        raise ComparisonError(f"refusing to overwrite existing output: {existing}")


def publish(paths: Mapping[str, Path], blobs: Mapping[str, bytes]) -> None:
    """Write every artifact via an exclusive temp file plus a no-overwrite hard link."""
    refuse_existing(paths)
    created: list[Path] = []
    try:
        for kind, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(blobs[kind])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temp, path)
                created.append(path)
            finally:
                os.unlink(temp)
    except BaseException:
        for path in created:
            os.unlink(path)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline paired comparison of existing workbench runs.")
    parser.add_argument("--workspace", type=Path, default=ROOT / ".cache" / "video-workbench")
    parser.add_argument("--run", action="append", required=True, dest="runs", help="run id (repeatable)")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--out-prefix", type=Path, required=True, help="fresh output prefix (.json/.md[/.png])")
    parser.add_argument("--png", action="store_true", help="also render a PNG chart (requires matplotlib)")
    args = parser.parse_args(argv)
    try:
        if len(args.runs) > MAX_RUNS or len(set(args.runs)) != len(args.runs):
            raise ComparisonError("runs must be unique and at most 32")
        paths = output_paths(args.out_prefix, args.png)
        refuse_existing(paths)
        runs = [collect_run(args.workspace, run_id, args.scope) for run_id in args.runs]
        report = build_report(runs, args.scope, ["tools/workbench_comparison.py", *(argv or sys.argv[1:])])
        blobs = {"json": (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
                 "md": render_markdown(report).encode("utf-8")}
        if args.png:
            try:
                blobs["png"] = render_png(report)
            except ImportError as exc:
                raise ComparisonError(f"PNG requested but matplotlib is unavailable: {exc}") from exc
        publish(paths, blobs)
    except (ComparisonError, ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps({"status": "written", **{k: str(v) for k, v in paths.items()},
                      **{f"{k}_sha256": sha256_bytes(v) for k, v in blobs.items()}}, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
