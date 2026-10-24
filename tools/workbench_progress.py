"""Plot saved evidence, without running models or treating missing metrics as zero.

Run with the project's existing plotting Python environment. Outputs are new,
timestamped PNG/JSON/CSV files, never overwrites of previous snapshots.

Every tracking metric gets its own per-clip panel; a missing value is drawn as
UNMEASURED (or UNVERIFIED for attachment accuracy without independent
annotations), never as a zero bar. The cached-token development error is a
separate panel and is never attachment accuracy. Clips are never pooled, and no
baseline comparison is drawn from primary-only receipts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
RECORDS_SHA256 = "09226dda6f08c739b05c42540156784b4783be5bf3d65b34cf0549b9959104aa"
TRACKING_METRICS = {
    "attachment_median_px_720": "Attachment median (720p px)",
    "attachment_p95_px_720": "Attachment P95 (720p px)",
    "availability_fraction": "All-frame pose availability",
    "artwork_render_fraction": "Artwork rendered / all frames",
    "wrong_surface_events": "Wrong-surface / flip events",
    "correct_recovery_fraction": "Correct recovery fraction",
    "latency_p95_ms": "Host per-frame latency P95 (ms)",
    "peak_memory_bytes": "Host process peak RSS (bytes)",
}
# Scope of the two resource metrics as produced by bench/workbench_evaluate.py and workbench_cli.
METRIC_SCOPE_NOTES = {
    "latency_p95_ms": "P95 of the shared host per-frame block (frame decode + cached mask load + tracking + "
                      "render + overlay write); excludes model initialization and segmentation preparation; "
                      "not phone FPS",
    "peak_memory_bytes": "host process peak RSS (resident set size); not GPU VRAM",
}
PANEL_TITLES = {
    "latency_p95_ms": "Host per-frame latency P95 (ms) · not phone FPS",
    "peak_memory_bytes": "Host process peak RSS (MiB) · not GPU memory",
}
ATTACHMENT_METRICS = ("attachment_median_px_720", "attachment_p95_px_720")
INTEGER_METRICS = ("wrong_surface_events", "peak_memory_bytes")
COUNT_FIELDS = ("scheduled_frames", "clip_scoring_frames", "result_rows", "failed_frames")
KNOWN_MODES = ("smoke", "regression")
FREE_RESERVE_BYTES = 1 << 30
OUTPUT_ALLOWANCE_BYTES = 32 << 20          # declared PNG+JSON+CSV allowance for one snapshot
CSV_FIELDS = (
    "row_kind", "id", "baseline_epe_px", "adapted_epe_px", "delta_epe_px",
    "clip_id", "run_id", "mode", "mode_label", "evidence_kind", "independent_annotations_complete",
    "coverage_status", "scheduled_frames", "clip_scoring_frames", "result_rows", "failed_frames",
    "completed_window_fraction", *TRACKING_METRICS, "missing_metrics", "comparison_status", "source_sha256",
)
ROW_KIND_DEVELOPMENT = "development_cached_token_epe"
ROW_KIND_VIDEO = "video_evaluation"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    if len(raw) > 16 * 1024**2:
        raise ValueError(f"Evidence receipt exceeds 16 MiB: {path}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate evidence field: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Non-finite evidence value: {value}")
    document = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(document, dict):
        raise ValueError("Evidence receipt must be an object")
    return document, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# pure semantics (tested without pixel matching)

def validate_metric(name: str, value: Any) -> Any:
    """None stays None (unmeasured). Rejects bools/non-finite; fractions in 0..1; px/ms >= 0;
    events and memory bytes are non-negative integers."""
    if value is None:
        return None
    if name in INTEGER_METRICS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Metric outside its domain: {name} must be a non-negative integer")
        return value
    number = _finite(value, name)
    if number < 0 or (name.endswith("fraction") and number > 1):
        raise ValueError(f"Metric outside its domain: {name}")
    return value


def validate_counts(report: Mapping) -> dict:
    """Structured coverage from the receipt's own counts. Missing counts stay unknown
    (backward compatible); a full-window claim with unequal counts is rejected."""
    counts = {}
    for name in COUNT_FIELDS:
        v = report.get(name)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise ValueError(f"{name} must be a non-negative integer")
        counts[name] = v
    scheduled, total = counts["scheduled_frames"], counts["clip_scoring_frames"]
    rows, failed = counts["result_rows"], counts["failed_frames"]
    # Bound every count by every known denominator, even when another count is missing.
    for name, value in (("scheduled_frames", scheduled), ("result_rows", rows), ("failed_frames", failed)):
        if value is not None and total is not None and value > total:
            raise ValueError(f"{name} exceeds clip_scoring_frames")
    if rows is not None and scheduled is not None and rows > scheduled:
        raise ValueError("result_rows exceeds scheduled_frames")
    if failed is not None and scheduled is not None and failed > scheduled:
        raise ValueError("failed_frames exceeds scheduled_frames")
    full = report.get("full_scoring_window_scheduled")
    if full is not None and not isinstance(full, bool):
        raise ValueError("full_scoring_window_scheduled must be a boolean")
    if full is True and scheduled is not None and total is not None and scheduled != total:
        raise ValueError("Receipt claims the full scoring window with unequal counts")
    # Complete: a non-empty frozen window, fully scheduled, every frame has a result row
    # (failed frames are rows and stay in the denominator), and the full-window flag is set.
    if full is True and total is not None and total > 0 and scheduled == total and rows == total:
        status = "complete"
    elif total is not None and total > 0 and ((scheduled is not None and scheduled < total)
                                               or (rows is not None and rows < total)):
        status = "partial"
    else:
        status = "unknown"
    completed = rows / total if rows is not None and total else None
    return {**counts, "full_window_claimed": full is True, "coverage_status": status,
            "completed_window_fraction": completed,
            "basis": "receipt counts only: result_rows / clip_scoring_frames (frozen scoring window)"}


def mode_label(mode: Any) -> str:
    return mode if mode in KNOWN_MODES else "scope unverified"


def coverage_label(row: Mapping) -> str:
    cov = row["coverage"]
    q = lambda v: "?" if v is None else v  # noqa: E731
    return f"{q(cov['scheduled_frames'])}/{q(cov['clip_scoring_frames'])} sched · " \
           f"{q(cov['result_rows'])} rows · {cov['coverage_status']}"


def metric_cell(row: Mapping, name: str) -> dict:
    """{status: measured|unmeasured|unverified, value, plot_value, display}; never 0 for missing."""
    value = (row.get("metrics") or {}).get(name)
    attachment = name in ATTACHMENT_METRICS
    annotated = row.get("independent_annotations_complete") is True
    if value is None:
        if attachment and not annotated:
            return {"status": "unverified", "value": None, "plot_value": None,
                    "display": "UNVERIFIED (no independent annotations)"}
        return {"status": "unmeasured", "value": None, "plot_value": None, "display": "UNMEASURED"}
    if name == "peak_memory_bytes":
        plot_value, display = value / 2**20, f"{value / 2**20:,.0f} MiB"
    elif name == "latency_p95_ms":
        plot_value, display = float(value), f"{value:,.0f} ms"
    elif name.endswith("fraction"):
        plot_value, display = float(value), f"{value:.3f}"
    elif name == "wrong_surface_events":
        plot_value, display = float(value), str(value)
    else:
        plot_value, display = float(value), f"{value:.3g} px"
    if attachment and not annotated:
        display += " (annotations incomplete: unverified)"
    return {"status": "measured", "value": value, "plot_value": plot_value, "display": display}


def report_labels(row: Mapping) -> dict:
    status = row.get("comparison_status")
    return {
        "row": f"{row['clip_id']} · {mode_label(row.get('mode'))} · {coverage_label(row)}",
        "mode": mode_label(row.get("mode")),
        "coverage": coverage_label(row),
        "annotations": "independent annotations complete" if row.get("independent_annotations_complete") is True
        else "independent annotations incomplete: attachment accuracy unverified",
        "comparison": ("not provided" if status is None else status if isinstance(status, str)
                       else json.dumps(status, sort_keys=True)) + " (as recorded; no liveness claim)",
    }


def check_unique(rows: Sequence[Mapping]) -> None:
    """Duplicates never inflate coverage: one row per clip+mode, per run and per source."""
    seen_clip_mode, seen_run, seen_source = set(), set(), set()
    for row in rows:
        key = (row["clip_id"], row.get("mode"))
        if key in seen_clip_mode:
            raise ValueError(f"Duplicate video evaluation for clip {row['clip_id']!r} mode {row.get('mode')!r}; "
                             "rows are never pooled")
        seen_clip_mode.add(key)
        if row.get("run_id") is not None:
            if row["run_id"] in seen_run:
                raise ValueError(f"Duplicate evidence run {row['run_id']!r}")
            seen_run.add(row["run_id"])
        if row.get("source_sha256") in seen_source:
            raise ValueError("The same evaluation receipt was supplied twice")
        seen_source.add(row.get("source_sha256"))


def coverage_summary(rows: Sequence[Mapping]) -> dict:
    clips = sorted({r["clip_id"] for r in rows})
    complete_clips = sorted({r["clip_id"] for r in rows if r["coverage"]["coverage_status"] == "complete"})
    return {
        "receipts": len(rows), "distinct_clip_ids": len(clips),
        "modes": {m: sum(1 for r in rows if mode_label(r.get("mode")) == m) for m in (*KNOWN_MODES, "scope unverified")},
        "complete_window_clip_ids": complete_clips,
        "partial_or_unknown_clip_ids": sorted(set(clips) - set(complete_clips)),
        "independent_annotations_complete": sum(1 for r in rows if r.get("independent_annotations_complete") is True),
        "independent_video_count": "not established by these receipts (object cases are not independent videos)",
        "pooled": False,
    }


def status_lines(snapshot: Mapping) -> list[str]:
    rows = snapshot["video_evaluations"]
    summary = coverage_summary(rows)
    lines = [f"Generated (UTC): {snapshot['generated_at_utc']}", f"Primary head: {snapshot['head_sha256']}"]
    if rows:
        lines.append(f"Video receipts: {summary['receipts']} per-clip "
                     f"(smoke {summary['modes']['smoke']}, regression {summary['modes']['regression']}, "
                     f"scope unverified {summary['modes']['scope unverified']}); never pooled")
        lines.append(f"Complete frozen-window coverage: {len(summary['complete_window_clip_ids'])} of "
                     f"{summary['distinct_clip_ids']} clip IDs; independent-video count not established")
        lines.append(f"Independent annotations complete: {summary['independent_annotations_complete']} of "
                     f"{summary['receipts']} receipts")
        for row in rows:
            lines.append(f"Comparison status [{row['clip_id']}]: {report_labels(row)['comparison']}")
    else:
        lines.append("Video receipts: none supplied; all tracking metrics UNMEASURED")
    checks = snapshot.get("software_checks")
    if checks is None:
        lines.append("Software validation: pending")
    elif checks.get("stale_source_files"):
        lines.append("Earlier software checks: source changed; rerun required")
    else:
        lines.append(f"Software checks: {checks.get('passed', 'unknown')} passed, "
                     f"{checks.get('failed', 'unknown')} failed, {checks.get('skipped', 'unknown')} skipped")
    lines.append("Phone inference / sustained performance: UNMEASURED")
    lines.append("Latency is host per-frame timing (not phone FPS); memory is host process peak RSS (not GPU VRAM).")
    lines.append("No baseline comparison is drawn from primary-only receipts.")
    lines.append("Availability, software checks and parity are not accuracy evidence.")
    return lines


def csv_rows(snapshot: Mapping) -> list[dict]:
    """Development cached-token rows and per-clip video rows in one table, separated by row_kind.
    Missing values are empty cells (and listed in missing_metrics), never 0."""
    def blank(v):
        return "" if v is None else v
    out = []
    for row in snapshot["paired_development_rows"]:
        out.append({**{k: "" for k in CSV_FIELDS}, "row_kind": ROW_KIND_DEVELOPMENT, "id": row["id"],
                    "baseline_epe_px": row["baseline_epe_px"], "adapted_epe_px": row["adapted_epe_px"],
                    "delta_epe_px": row["delta_epe_px"]})
    for row in snapshot["video_evaluations"]:
        cov = row["coverage"]
        metrics = row.get("metrics") or {}
        rec = {**{k: "" for k in CSV_FIELDS}, "row_kind": ROW_KIND_VIDEO, "id": row["clip_id"],
               "clip_id": row["clip_id"], "run_id": blank(row.get("run_id")), "mode": blank(row.get("mode")),
               "mode_label": mode_label(row.get("mode")), "evidence_kind": row.get("evidence_kind"),
               "independent_annotations_complete": row.get("independent_annotations_complete") is True,
               "coverage_status": cov["coverage_status"],
               **{k: blank(cov[k]) for k in COUNT_FIELDS},
               "completed_window_fraction": blank(cov["completed_window_fraction"]),
               "missing_metrics": ";".join(n for n in TRACKING_METRICS if metrics.get(n) is None),
               "comparison_status": "" if row.get("comparison_status") is None else (
                   row["comparison_status"] if isinstance(row["comparison_status"], str)
                   else json.dumps(row["comparison_status"], sort_keys=True)),
               "source_sha256": blank(row.get("source_sha256"))}
        for name in TRACKING_METRICS:
            rec[name] = blank(metrics.get(name))
        out.append(rec)
    return out


def csv_text(snapshot: Mapping) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(csv_rows(snapshot))
    return buffer.getvalue()


# --------------------------------------------------------------------------
# collection

def collect(records_path: Path, *, evaluation_reports: list[Path] | None = None,
            checks_path: Path | None = None) -> dict:
    records, digest = _read(records_path)
    if digest != RECORDS_SHA256:
        raise ValueError("Saved training records differ from the verified receipt")
    development = records["development-comparison.json"]
    if development["export_sha256"] != HEAD_SHA256:
        raise ValueError("Development evidence names a different checkpoint")
    pairs = development["per_frame"]
    if len(pairs) != development["validation_available"]:
        raise ValueError("Paired validation rows do not match the recorded denominator")
    if len({row["id"] for row in pairs}) != len(pairs):
        raise ValueError("Repeated validation frame identity")
    for row in pairs:
        baseline = _finite(row["baseline_epe_px"], "baseline_epe_px")
        adapted = _finite(row["adapted_epe_px"], "adapted_epe_px")
        if not math.isclose(adapted - baseline, _finite(row["delta_epe_px"], "delta"),
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError("Paired error difference is inconsistent")
    snapshot = {
        "schema": "workbench_progress_snapshot_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "head_sha256": HEAD_SHA256,
        "sources": [{"path": str(records_path.resolve()), "sha256": digest}],
        "development": {key: development[key] for key in (
            "baseline_epe_px", "adapted_epe_px", "baseline_pooled", "adapted_pooled",
            "confidence_parity", "optimizer_updates", "validation_available",
            "validation_total", "validation_objects", "train_rows", "train_objects")},
        "development_label": "cached-token EPE on development crops; not surface attachment accuracy",
        "metric_scope_notes": dict(METRIC_SCOPE_NOTES),
        "paired_development_rows": pairs,
        "video_evaluations": [],
        "software_checks": None,
        "phone_validation": "unmeasured",
        "quality_claim": False,
    }
    for report_path in evaluation_reports or []:
        report, report_digest = _read(report_path)
        if report.get("schema") != "workbench_evaluation_v1":
            raise ValueError("Only explicit workbench_evaluation_v1 receipts are charted as video metrics")
        if report.get("installed_head_sha256") != HEAD_SHA256:
            raise ValueError("Video evaluation must verify the actual installed candidate head")
        if report.get("inference_executed") is not True:
            raise ValueError("Blocked or unexecuted runs cannot supply measured video metrics")
        if not isinstance(report.get("clip_id"), str) or not report["clip_id"]:
            raise ValueError("Video evaluation needs a clip_id")
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("Video evaluation metrics must be an object")
        for name in TRACKING_METRICS:
            validate_metric(name, metrics.get(name))
        coverage = validate_counts(report)
        snapshot["video_evaluations"].append({
            "clip_id": report["clip_id"], "metrics": metrics,
            "mode": report.get("mode", "scope unverified"),
            "clip_scoring_frames": report.get("clip_scoring_frames"),
            "full_scoring_window_scheduled": report.get("full_scoring_window_scheduled") is True,
            "comparison_status": report.get("comparison_status"),
            "evidence_kind": report.get("evidence_kind", "unverified"),
            "scheduled_frames": report.get("scheduled_frames"),
            "failed_frames": report.get("failed_frames"),
            "independent_annotations_complete": report.get("independent_annotations_complete") is True,
            "result_rows": report.get("result_rows"),
            "run_id": report.get("run_id"),
            "backend_role": report.get("backend_role"),
            "coverage": coverage,
            "source_sha256": report_digest,
        })
        snapshot["sources"].append({"path": str(report_path.resolve()), "sha256": report_digest})
    check_unique(snapshot["video_evaluations"])
    snapshot["coverage"] = coverage_summary(snapshot["video_evaluations"])
    if checks_path is not None:
        checks, checks_digest = _read(checks_path)
        if checks.get("schema") != "workbench_software_checks_v1":
            raise ValueError("Software checks require a typed receipt")
        checks = dict(checks)
        changed = []
        for relative, expected in checks.get("source_files", {}).items():
            source = (ROOT / relative).resolve()
            if ROOT not in source.parents:
                raise ValueError("Software-check source path escapes the project")
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                changed.append(relative)
        checks["stale_source_files"] = changed
        snapshot["software_checks"] = checks
        snapshot["sources"].append({"path": str(checks_path.resolve()), "sha256": checks_digest})
    return snapshot


# --------------------------------------------------------------------------
# rendering (matplotlib imported lazily)

_INK, _MUTED, _WARN, _GOOD, _BAD, _BAR = "#384354", "#546174", "#a24f24", "#227458", "#b85b31", "#4a6fa5"


def tick_labels(rows: Sequence[Mapping]) -> list[str]:
    """Compact metric-panel ticks: the clip ID only; the mode is appended only when the same
    clip ID appears more than once. Scope and counts live in the coverage panel, status and CSV."""
    ids = [r["clip_id"] for r in rows]
    return [f"{r['clip_id']} ({mode_label(r.get('mode'))})" if ids.count(r["clip_id"]) > 1 else r["clip_id"]
            for r in rows]


def _metric_panel(ax, rows: Sequence[Mapping], name: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, weight="bold", color=_INK)
    if not rows:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(.5, .5, "NO VIDEO RECEIPTS\nUNMEASURED", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, weight="bold", color=_WARN)
        return
    cells = [metric_cell(r, name) for r in rows]
    measured = [c["plot_value"] for c in cells if c["status"] == "measured"]
    ys = list(range(len(rows)))
    ax.set_yticks(ys, tick_labels(rows), fontsize=8)
    ax.set_ylim(len(rows) - .5, -.5)
    if name.endswith("fraction"):
        upper = 1.0
    else:
        upper = max(measured) * 1.35 if measured and max(measured) > 0 else 1.0
    ax.set_xlim(0, upper)
    trans = ax.get_yaxis_transform()        # x in axes coordinates, y in data rows
    for y, cell in zip(ys, cells):
        if cell["status"] == "measured":
            ax.barh(y, cell["plot_value"], color=_BAR, height=.6)
            ax.text(min(cell["plot_value"] / upper, .98) + .01 if upper else .02, y, cell["display"],
                    transform=trans, va="center", fontsize=8, color=_INK, clip_on=False)
        else:
            ax.text(.01, y, cell["display"], transform=trans, va="center", fontsize=8, weight="bold",
                    color=_WARN)
    if not measured:
        ax.set_xticks([])
        word = "UNVERIFIED" if all(c["status"] == "unverified" for c in cells) else "UNMEASURED"
        ax.text(.98, .02, f"ALL {word} · no bar means no measurement, not zero", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=_WARN)
    elif name == "peak_memory_bytes":
        ax.set_xlabel("MiB", fontsize=8, color=_MUTED)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)


def _coverage_panel(ax, rows: Sequence[Mapping]) -> None:
    ax.set_title("Completed frozen scoring-window coverage", loc="left", fontsize=11, weight="bold", color=_INK)
    if not rows:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(.5, .5, "NO VIDEO RECEIPTS\nUNMEASURED", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, weight="bold", color=_WARN)
        return
    ys = list(range(len(rows)))
    ax.set_yticks(ys, tick_labels(rows), fontsize=8)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xlim(0, 1)
    trans = ax.get_yaxis_transform()
    for y, row in zip(ys, rows):
        cov = row["coverage"]
        frac = cov["completed_window_fraction"]
        rows_text = f"{'?' if cov['result_rows'] is None else cov['result_rows']} rows / " \
                    f"{'?' if cov['clip_scoring_frames'] is None else cov['clip_scoring_frames']} window"
        label = f"{mode_label(row.get('mode'))} · {rows_text} · {cov['coverage_status']}"
        if frac is None:
            ax.text(.01, y, f"UNMEASURED · {label}", transform=trans, va="center", fontsize=8, weight="bold",
                    color=_WARN)
        else:
            color = _GOOD if cov["coverage_status"] == "complete" else _BAD
            ax.barh(y, frac, color=color, height=.6)
            ax.text(.02, y, label, transform=trans, va="center", fontsize=8, color="white" if frac > .5 else _INK)
    ax.grid(axis="x", alpha=.18)
    ax.tick_params(axis="x", labelsize=8)


def render_png(snapshot: Mapping) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = snapshot["video_evaluations"]
    fig = plt.figure(figsize=(17, 27))
    fig.set_facecolor("#f6f7f9")
    gs = fig.add_gridspec(6, 2, height_ratios=[1.3, 1, 1, 1, 1, 1], hspace=.6, wspace=.42,
                          top=.915, bottom=.04, left=.13, right=.96)
    dev = fig.add_subplot(gs[0, 0])
    pairs = snapshot["paired_development_rows"]
    differences = [row["delta_epe_px"] for row in pairs]
    dev.bar(range(len(pairs)), differences, color=[_GOOD if v < 0 else _BAD for v in differences], width=.75)
    dev.axhline(0, color=_INK, linewidth=1)
    dev.set_title("DEVELOPMENT ONLY · cached-token EPE\n(not surface attachment accuracy)", loc="left",
                  fontsize=11, weight="bold", color=_INK)
    dev.set_xlabel("Paired cached validation input (dependent views)", fontsize=8)
    dev.set_ylabel("Candidate minus frozen EPE (crop px)", fontsize=8)
    dev.grid(axis="y", alpha=.18)
    dev.set_axisbelow(True)
    d = snapshot["development"]
    dev.text(0, -.2, f"Mean EPE {d['baseline_epe_px']:.6f} → {d['adapted_epe_px']:.6f} px · "
             f"{d['validation_available']}/{d['validation_total']} inputs · {d['validation_objects']} object(s)",
             transform=dev.transAxes, va="top", fontsize=8, color=_MUTED)
    status = fig.add_subplot(gs[0, 1])
    status.axis("off")
    status.set_title("Evidence status (saved receipts only)", loc="left", fontsize=11, weight="bold", color=_INK)
    status.text(0, .98, "\n".join(status_lines(snapshot)), transform=status.transAxes, va="top", fontsize=8,
                color=_INK, linespacing=1.55, wrap=True)
    for index, (name, title) in enumerate(TRACKING_METRICS.items()):
        ax = fig.add_subplot(gs[1 + index // 2, index % 2])
        _metric_panel(ax, rows, name, PANEL_TITLES.get(name, title))
    _coverage_panel(fig.add_subplot(gs[5, 0]), rows)
    notes = fig.add_subplot(gs[5, 1])
    notes.axis("off")
    notes.set_title("How to read", loc="left", fontsize=11, weight="bold", color=_INK)
    notes.text(0, .95, "\n".join([
        "UNMEASURED: the receipt has no value; never drawn as zero.",
        "UNVERIFIED: attachment accuracy needs independent annotations.",
        "Each clip is its own row; nothing is pooled across clips.",
        "Mode, scheduled/window counts and rows: coverage panel, status, CSV.",
        "Object cases are not a count of independent videos.",
        "Latency: host per-frame block (decode + cached mask load + tracking",
        "  + render + overlay write); excludes model init and segmentation prep.",
        "Memory: host process peak RSS, not GPU VRAM.",
        "Phone FPS / performance: UNMEASURED. No inference ran for this chart.",
    ]), transform=notes.transAxes, va="top", fontsize=9, color=_INK, linespacing=1.6)
    fig.suptitle("VisualizeIT · saved-evidence results", x=.03, y=.985, ha="left", fontsize=20, weight="bold")
    fig.text(.03, .962, f"Primary head {snapshot['head_sha256']} · generated {snapshot['generated_at_utc']} UTC",
             fontsize=9, color=_MUTED)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def plot(snapshot: dict, output: Path) -> None:
    _write_new(Path(output), render_png(snapshot))


_REPARSE_POINT = 0x400


def _output_dir(path: Path) -> Path:
    """Containment by the real filesystem, checked before anything is created: every existing
    component under ROOT must be a plain directory (no symlink/junction/reparse alias), and the
    resolved path (including not-yet-existing children) must stay strictly inside the real ROOT."""
    import stat
    lexical_root = Path(os.path.abspath(ROOT))
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(lexical_root)
    except ValueError:
        raise ValueError("--output-dir must be inside the project") from None
    if not relative.parts:
        raise ValueError("--output-dir must be a directory inside the project, not the project root")
    current = lexical_root
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            break                          # the rest will be created below a verified real directory
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
            raise ValueError("--output-dir traverses a symlink/junction; refusing an alias outside the project")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("--output-dir traverses a non-directory")
    real_root = Path(os.path.realpath(ROOT))
    real = Path(os.path.realpath(candidate))
    if real == real_root or real_root not in real.parents:
        raise ValueError("--output-dir resolves outside the project")
    return candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / ".cache/cloud-prepare/flow-terminal-small-records-20261021.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/video-workbench/progress")
    parser.add_argument("--evaluation-report", type=Path, action="append", default=[])
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args(argv)
    snapshot = collect(args.records, evaluation_reports=args.evaluation_report, checks_path=args.checks)
    out_dir = _output_dir(args.output_dir)
    json_bytes = json.dumps(snapshot, indent=2, allow_nan=False).encode("utf-8")
    csv_bytes = csv_text(snapshot).encode("utf-8")
    png_bytes = render_png(snapshot)
    total = len(json_bytes) + len(csv_bytes) + len(png_bytes)
    if total > OUTPUT_ALLOWANCE_BYTES:
        raise ValueError(f"Snapshot outputs ({total} bytes) exceed the declared {OUTPUT_ALLOWANCE_BYTES} byte allowance")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = _output_dir(out_dir)         # re-verify the created path before writing
    if shutil.disk_usage(out_dir).free - OUTPUT_ALLOWANCE_BYTES < FREE_RESERVE_BYTES:
        raise ValueError("Writing the snapshot would violate the 1 GiB free-disk reserve")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = out_dir / f"results-{stamp}"
    paths = {suffix: stem.with_suffix(suffix) for suffix in (".json", ".csv", ".png")}
    if any(os.path.lexists(p) for p in paths.values()):
        raise ValueError("A snapshot with this timestamp already exists; nothing overwritten")
    _write_new(paths[".json"], json_bytes)
    _write_new(paths[".csv"], csv_bytes)
    _write_new(paths[".png"], png_bytes)
    print(json.dumps({"png": str(paths[".png"]), "json": str(paths[".json"]), "csv": str(paths[".csv"]),
                      "inference_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
