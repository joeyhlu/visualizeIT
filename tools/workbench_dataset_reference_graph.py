"""Dataset-reference diagnostic graph for the three original YCB full runs (CPU only).

Reads ONLY the sealed post-run diagnostic
``.cache/video-workbench/derived/dataset-reference/eval-v1/dataset-reference-projection.json``
(``workbench_ycb_dataset_reference_projection_v1``). Its exact SHA-256 is pinned
and must also equal ``--expected-input-sha256``. The tool runs no model and no
inference, and never reads raw labels such as ``evaluator.json``, masks or depth.
Every summary it plots is first checked against the per-frame rows already
inside the sealed document.

This is a development DATASET-REFERENCE diagnostic, not independent human
attachment evidence: initialisation uses known dataset poses, the CAD geometry
and calibration are shared with the reference, and the public dataset may have
been seen in training. The phone is unverified. The 5 / 10 px720 lines are
provisional targets, not a passing gate. The new head and the frozen baseline
are shown separately per clip and never pooled, and no statistical difference
is claimed.

    python tools/workbench_dataset_reference_graph.py \
        --input .cache/video-workbench/derived/dataset-reference/eval-v1/dataset-reference-projection.json \
        --expected-input-sha256 83bccd35f98f70a46730d99bb15ec33883e6e6f3d421f13dc5c9410a451250b1 \
        --out-prefix artifacts/video-workbench/progress/dataset-reference-ycb-v1
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
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "workbench_dataset_reference_graph_v1"
INPUT_SCHEMA = "workbench_ycb_dataset_reference_projection_v1"
INPUT_SHA256 = "83bccd35f98f70a46730d99bb15ec33883e6e6f3d421f13dc5c9410a451250b1"
PLAN_SHA256 = "3c85e9c0e6f4b7b7414fec200a00a6071c4406f64e2e9bfa76f5f4e4b6de92f6"
EVALUATOR_TOOL_SHA256 = "d62d47b81ead461567699ff41be89a913634a9e717cf7a44ed1a0ca2ca31c538"
EVALUATOR_TOOL = "tools/workbench_ycb_dataset_evaluate.py"
EXPECTED_PINS = {
    "base_sha256": "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086",
    "evaluator_sha256": "c54534d6835653eb367f8d6fee34bd56cd3417056d48f5468d4f6e7d565b1870",
    "head_sha256": "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311",
    "inference_sha256": "9b936de31c5511e3644e065538dedf47d0cdcd22f2ba17cee083e75a25403945",
    "reviewed_map_sha256": "c5ae3b009007ac1600af345c56f1b719e8881591baf37187a917d361eb99ca20",
}
EXPECTED_FLAGS = {
    "dataset_reference_diagnostic": True, "development_only": True, "full_printed_uv_correspondence": "inconclusive",
    "human_gates": "unverified", "independent_human_accuracy": False, "phone_attachment_verified": False,
    "report_accuracy_overwritten": False,
}
# Fixed object order with the ORIGINAL full runs.
OBJECTS = (
    {"object_id": 2, "label": "YCB cracker box", "clip_id": "dev-ycb-003-cracker-box",
     "run_id": "run-20261021T030702-475adc39"},
    {"object_id": 4, "label": "YCB soup can", "clip_id": "dev-ycb-005-tomato-soup-can",
     "run_id": "run-20261021T031729-0986ef46"},
    {"object_id": 18, "label": "YCB marker", "clip_id": "dev-ycb-040-large-marker",
     "run_id": "run-20261021T032932-539343a7"},
)
FRAMES = 110
ROLES = ("primary_neural", "frozen_baseline")
ROLE_LABELS = {"primary_neural": "New head 27e06a51", "frozen_baseline": "Frozen baseline f7d127ab"}
PROVISIONAL_PX720 = (5.0, 10.0)
MAX_INPUT_BYTES = 8 << 20
MAX_OUTPUT_BYTES = 8 << 20
FREE_RESERVE_BYTES = 1 << 30
TOLERANCE = 1e-9
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FOOTER = ("Development DATASET-REFERENCE diagnostic, not independent human truth: known-pose CAD initialisation (dataset "
          "onboarding seed); CAD geometry and calibration (K, assumed integer_centers origin) shared with the reference; "
          "public YCB-Video exposure possible; phone attachment unverified; wrong-surface and recovery correctness "
          "unverified. Provisional 5/10 px720 lines are not a passing gate; no pooling and no statistical claim.")


class GraphError(ValueError):
    """The sealed diagnostic does not support this graph."""


# --------------------------------------------------------------------------
# strict reading


def loads_strict(raw: bytes) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise GraphError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise GraphError(f"non-finite JSON value {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphError(f"invalid JSON ({exc})") from None


def read_input(path: Path, expected_sha256: str) -> tuple[dict, str]:
    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise GraphError("--expected-input-sha256 must be a lowercase SHA-256")
    if expected_sha256 != INPUT_SHA256:
        raise GraphError("expected SHA-256 is not the reviewed sealed diagnostic for this graph version")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise GraphError("input must be an existing regular file")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise GraphError("input exceeds its size bound")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise GraphError(f"input SHA-256 {actual[:12]}... differs from the expected sealed diagnostic")
    doc = loads_strict(raw)
    if not isinstance(doc, dict):
        raise GraphError("input must be a JSON object")
    return doc, actual


# --------------------------------------------------------------------------
# pure validation (unit-testable with mutated authentic documents)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphError(message)


def _num_or_none(value: Any, what: str) -> float | None:
    if value is None:
        return None
    _require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)),
             f"{what} must be a finite number or null")
    return float(value)


def _same(recorded: Any, recomputed: float | None, what: str) -> float | None:
    value = _num_or_none(recorded, what)
    if recomputed is None:
        _require(value is None, f"{what} must be null when nothing was measured (never zero)")
        return None
    _require(value is not None and abs(value - recomputed) <= TOLERANCE * max(1.0, abs(recomputed)),
             f"{what} differs from the sealed per-frame rows")
    return value


def _percentiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def _anchor(summary: Any, rows: Sequence[Mapping], where: str) -> dict:
    _require(isinstance(summary, dict), f"{where}: anchor summary missing")
    eligible = [r for r in rows if r.get("reference_status") == "eligible"]
    errors = []
    for r in eligible:
        err = _num_or_none(r.get("error_px720"), f"{where} error_px720")
        if err is not None:
            _require(err >= 0 and r.get("prediction_status") == "ok", f"{where}: measured error without an ok prediction")
            errors.append(err)
        else:
            _require(r.get("prediction_status") != "ok", f"{where}: ok prediction without a measured error")
    _require(summary.get("reference_eligible") == len(eligible), f"{where}: reference_eligible differs from rows")
    _require(summary.get("errors_measured") == len(errors), f"{where}: errors_measured differs from rows")
    median, p95 = _percentiles(errors)
    out = {"reference_eligible": len(eligible), "errors_measured": len(errors),
           "median_px720": _same(summary.get("median_px720"), median, f"{where} median_px720"),
           "p95_px720": _same(summary.get("p95_px720"), p95, f"{where} p95_px720"),
           "denominator": summary.get("denominator")}
    for limit in PROVISIONAL_PX720:
        key = f"coverage_le_{int(limit)}px720"
        recomputed = (sum(1 for e in errors if e <= limit) / len(eligible)) if eligible else None
        out[key] = _same(summary.get(key), recomputed, f"{where} {key}")
        out[f"covered_le_{int(limit)}px720_frames"] = sum(1 for e in errors if e <= limit)
    return out


def validate_role(role: str, body: Any, where: str) -> dict:
    w = f"{where} {role}"
    _require(isinstance(body, dict), f"{w}: role record missing")
    rows = body.get("A_anchor_rows")
    _require(isinstance(rows, list) and len(rows) == FRAMES and all(isinstance(r, dict) for r in rows),
             f"{w}: exactly {FRAMES} anchor rows are required")
    frame_ids = [r.get("frame_id") for r in rows]
    _require(len(set(frame_ids)) == FRAMES and all(isinstance(f, str) for f in frame_ids), f"{w}: anchor frames not unique")
    anchor = _anchor(body.get("A_anchor"), rows, f"{w} A_anchor")
    _require(anchor["reference_eligible"] == FRAMES, f"{w}: not all {FRAMES} frames are reference-eligible")
    visible_rows = [r for r in rows if r.get("reference_visible") == "visible"]
    anchor_visible = _anchor(body.get("C_anchor_reference_visible"), visible_rows, f"{w} C_anchor_reference_visible")
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("prediction_status"))] = statuses.get(str(r.get("prediction_status")), 0) + 1
    _require(body.get("A_prediction_status_counts") == statuses, f"{w}: prediction status counts differ from rows")
    avail = body.get("D_availability")
    _require(isinstance(avail, dict) and avail.get("frames") == FRAMES, f"{w}: availability record incomplete")
    arows = avail.get("rows")
    _require(isinstance(arows, list) and len(arows) == FRAMES and [r.get("frame_id") for r in arows] == frame_ids,
             f"{w}: availability rows must cover the same {FRAMES} frames")
    rendered = sum(1 for r in arows if r.get("render_state") == "rendered")
    tracking = sum(1 for r in arows if r.get("tracking_pose") is True)
    _require(avail.get("render_state_rendered") == rendered and avail.get("tracking_pose") == tracking,
             f"{w}: availability counts differ from rows")
    human = body.get("human_metrics")
    _require(isinstance(human, dict) and human.get("status") == "unverified"
             and human.get("wrong_surface_events") is None and human.get("recovery_correctness") is None,
             f"{w}: human metrics must remain unverified and null")
    cad = body.get("B_cad_samples")
    _require(isinstance(cad, dict) and cad.get("frames") == FRAMES, f"{w}: CAD-sample summary incomplete")
    return {"role": role, "frames": FRAMES, "anchor": anchor, "anchor_reference_visible": anchor_visible,
            "prediction_status_counts": statuses, "tracking_pose_frames": tracking, "artwork_rendered_frames": rendered,
            "artwork_render_fraction": rendered / FRAMES,
            "rendered_pixels_median_when_positive": _num_or_none(avail.get("rendered_pixels_median_when_positive"),
                                                                 f"{w} rendered pixels"),
            "cad_samples": {k: v for k, v in cad.items() if k != "note"},
            "human_metrics": {"status": "unverified", "wrong_surface_events": None, "recovery_correctness": None}}


def validate_object(case: Mapping, body: Any) -> dict:
    where = f"object {case['object_id']}"
    _require(isinstance(body, dict) and body.get("object_id") == case["object_id"], f"{where}: record missing")
    _require(body.get("run_id") == case["run_id"], f"{where}: run {body.get('run_id')} is not the original run "
             f"{case['run_id']}")
    _require(body.get("frames") == FRAMES, f"{where}: expected {FRAMES} frames")
    roles = body.get("roles")
    _require(isinstance(roles, dict) and set(roles) == set(ROLES), f"{where}: roles must be exactly {list(ROLES)}")
    role_rows = {role: validate_role(role, roles[role], where) for role in ROLES}
    _require(role_rows["primary_neural"]["anchor"]["errors_measured"] > 0
             or role_rows["frozen_baseline"]["anchor"]["errors_measured"] > 0, f"{where}: no measured anchor errors")
    vis = body.get("visibility_counts")
    _require(isinstance(vis, dict) and sum(v for v in vis.values() if type(v) is int) == FRAMES,
             f"{where}: visibility counts do not cover {FRAMES} frames")
    overlap = body.get("mask_overlap_sam_vs_dataset")
    _require(isinstance(overlap, list) and len(overlap) == FRAMES, f"{where}: mask overlap rows incomplete")
    ious = []
    for r in overlap:
        _require(isinstance(r, dict), f"{where}: mask overlap row malformed")
        if r.get("status") == "ok":
            iou = _num_or_none(r.get("iou"), f"{where} iou")
            _require(iou is not None and 0 <= iou <= 1, f"{where}: IoU outside [0, 1]")
            ious.append(iou)
    summary = body.get("mask_overlap_summary")
    _require(isinstance(summary, dict) and summary.get("frames_with_labels") == len(ious),
             f"{where}: mask overlap summary differs from rows")
    iou_median = _percentiles(ious)[0]
    paired = body.get("paired_anchor")
    _require(isinstance(paired, dict), f"{where}: paired anchor summary missing")
    return {"object_id": case["object_id"], "label": case["label"], "clip_id": case["clip_id"], "run_id": case["run_id"],
            "frames_per_role": FRAMES, "roles": role_rows, "visibility_counts": dict(vis),
            "mask_overlap": {"scope": "shared SAM mask input vs dataset mask_visib; not head-specific",
                             "frames_with_labels": len(ious),
                             "iou_median": _same(summary.get("iou_median"), iou_median, f"{where} iou_median"),
                             "leakage_median": _num_or_none(summary.get("leakage_median"), f"{where} leakage")},
            "paired_anchor": {k: (v if k in ("note", "paired_frames") else _num_or_none(v, f"{where} paired {k}"))
                              for k, v in paired.items()}}


def validate_document(doc: Any, *, source_sha256: str | None = None) -> dict:
    """The sealed diagnostic -> validated per-object data (no recomputation of the evaluation)."""
    _require(isinstance(doc, dict) and doc.get("schema") == INPUT_SCHEMA, f"input is not {INPUT_SCHEMA}")
    if source_sha256 is not None:
        _require(source_sha256 == INPUT_SHA256, "input is not the sealed diagnostic for this graph version")
    _require(doc.get("plan_sha256") == PLAN_SHA256, "evaluation plan differs from the reviewed plan")
    _require(doc.get("tool_sha256") == EVALUATOR_TOOL_SHA256
             and (doc.get("sources") or {}).get(EVALUATOR_TOOL) == EVALUATOR_TOOL_SHA256,
             "evaluator tool differs from the reviewed tool")
    _require(doc.get("pins") == EXPECTED_PINS, "model/data pins differ")
    _require(doc.get("flags") == EXPECTED_FLAGS, "flags differ (no human, phone or accuracy promotion is allowed)")
    closure = doc.get("hash_closure")
    _require(isinstance(closure, dict) and closure.get("equal") is True
             and closure.get("before_read") == closure.get("after_read")
             and isinstance(closure.get("before_read"), dict) and closure["before_read"], "hash closure not equal")
    seal = closure.get("terminal_seal")
    _require(isinstance(seal, dict) and seal.get("before") == seal.get("after") == doc.get("terminal_seal_sha256"),
             "terminal seal differs")
    objects = doc.get("objects")
    _require(isinstance(objects, dict) and set(objects) == {str(c["object_id"]) for c in OBJECTS},
             "objects must be exactly 2, 4 and 18")
    rows = [validate_object(case, objects[str(case["object_id"])]) for case in OBJECTS]
    return {"objects": rows, "limitations": list(doc.get("limitations") or []),
            "metric_spec": dict(doc.get("metric_spec") or {}), "terminal_seal_sha256": doc["terminal_seal_sha256"]}


def build_rows(validated: Mapping) -> list[dict]:
    """One flat row per object x role (exact values; missing stays None)."""
    out = []
    for obj in validated["objects"]:
        for role in ROLES:
            r = obj["roles"][role]
            a = r["anchor"]
            out.append({"object_id": obj["object_id"], "label": obj["label"], "clip_id": obj["clip_id"],
                        "run_id": obj["run_id"], "role": role, "frames": FRAMES,
                        "anchor_median_px720": a["median_px720"], "anchor_p95_px720": a["p95_px720"],
                        "anchor_errors_measured": a["errors_measured"], "anchor_reference_eligible": a["reference_eligible"],
                        "metric_scope_median_p95": f"conditional on {a['errors_measured']}/{a['reference_eligible']} "
                                                   "measured frames",
                        "coverage_le_5px720": a["coverage_le_5px720"], "covered_le_5_frames": a["covered_le_5px720_frames"],
                        "coverage_le_10px720": a["coverage_le_10px720"],
                        "covered_le_10_frames": a["covered_le_10px720_frames"],
                        "metric_scope_coverage": f"failure-aware over all {a['reference_eligible']} reference-eligible frames",
                        "tracking_pose_frames": r["tracking_pose_frames"],
                        "artwork_rendered_frames": r["artwork_rendered_frames"],
                        "artwork_render_fraction": r["artwork_render_fraction"],
                        "sam_vs_dataset_iou_median": obj["mask_overlap"]["iou_median"],
                        "iou_scope": "shared SAM mask input; not head-specific",
                        "human_accuracy": "unverified"})
    return out


def csv_bytes(rows: Sequence[Mapping]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# plot


def figure_png(validated: Mapping, *, source_sha256: str, generated_utc: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    objs = validated["objects"]
    colors = {"primary_neural": "#2563eb", "frozen_baseline": "#9ca3af"}
    labels = [f"{o['label']}\n{o['run_id'][-8:]}" for o in objs]
    x = np.arange(len(objs))
    w = 0.36
    fig = plt.figure(figsize=(13.0, 9.6), dpi=150, facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.30], hspace=0.55, wspace=0.16,
                            left=0.06, right=0.985, top=0.845, bottom=0.025)
    fig.suptitle("YCB dataset-reference diagnostic: new head 27e06a51 vs frozen baseline f7d127ab (3 original full runs)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.977)
    fig.text(0.02, 0.935, "Development DATASET-REFERENCE diagnostic, not independent human attachment evidence. "
             "Per clip, never pooled. No reliable head improvement established; no statistical test.",
             fontsize=10, color="#b91c1c", ha="left")
    fig.legend(handles=[Patch(color=colors[r], label=ROLE_LABELS[r]) for r in ROLES], loc="upper left",
               bbox_to_anchor=(0.012, 0.928), ncol=2, fontsize=9.5, frameon=False)

    def style(ax, title, ylabel):
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x, labels, fontsize=8.8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # A: conditional anchor median / P95
    ax = fig.add_subplot(grid[0, 0])
    tops = [o["roles"][r]["anchor"]["p95_px720"] or 0.0 for o in objs for r in ROLES]
    ymax = max(max(tops), PROVISIONAL_PX720[1]) * 1.3
    for offset, role in ((-w / 2, "primary_neural"), (w / 2, "frozen_baseline")):
        for xi, o in zip(x + offset, objs):
            a = o["roles"][role]["anchor"]
            if a["median_px720"] is None:
                ax.text(xi, 0.5, "n/a", ha="center", fontsize=8)
                continue
            ax.bar(xi, a["median_px720"], w, color=colors[role])
            ax.vlines(xi, a["median_px720"], a["p95_px720"], color="#111827", linewidth=1.0)
            ax.scatter([xi], [a["p95_px720"]], marker="_", s=300, linewidths=2.2, color="#111827", zorder=3)
            ax.text(xi, a["p95_px720"] + ymax * 0.02, f"{a['median_px720']:.2f} /\n{a['p95_px720']:.2f}",
                    ha="center", va="bottom", fontsize=7.4)
            ax.text(xi, ymax * 0.02, f"n={a['errors_measured']}/{a['reference_eligible']}", ha="center", va="bottom",
                    fontsize=6.8, color="white" if a["median_px720"] > ymax * 0.12 else "#111827", rotation=90)
    for limit in PROVISIONAL_PX720:
        ax.axhline(limit, color="#b45309", linestyle="--", linewidth=1.0)
        ax.text(x[-1] + 0.42, limit, f"provisional\n{limit:g} px720", fontsize=7.2, color="#b45309", va="center", ha="left")
    ax.set_xlim(-0.6, x[-1] + 1.0)
    ax.set_ylim(0, ymax)
    style(ax, "A. Anchor error, px720: median (bar) / P95 (tick)\n    conditional on measured frames (n shown)", "px at 720p")

    # B: failure-aware coverage
    ax = fig.add_subplot(grid[0, 1])
    q = w / 2
    for i, (limit, hatch) in enumerate(((5, "///"), (10, ""))):
        for j, role in enumerate(ROLES):
            pos = x + (-1.5 + 2 * i + j) * q
            vals = [o["roles"][role]["anchor"][f"coverage_le_{limit}px720"] for o in objs]
            for xi, v, o in zip(pos, vals, objs):
                a = o["roles"][role]["anchor"]
                if v is None:
                    ax.text(xi, 1, "n/a", ha="center", fontsize=7)
                    continue
                ax.bar(xi, 100 * v, q, color=colors[role], hatch=hatch, edgecolor="white" if not hatch else "#1f2937",
                       linewidth=0.4)
                ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 112)
    ax.legend(handles=[Patch(facecolor="white", edgecolor="#1f2937", hatch="///", label="<= 5 px720 (hatched)"),
                       Patch(facecolor="#d1d5db", label="<= 10 px720 (solid)")], fontsize=8, loc="upper left",
              frameon=False, ncol=2)
    style(ax, "B. Failure-aware anchor coverage, % of all 110 reference-eligible frames\n"
              "    (missing predictions count as not covered)", "% of 110 frames")

    # C: shared SAM vs dataset mask IoU
    ax = fig.add_subplot(grid[1, 0])
    for xi, o in zip(x, objs):
        m = o["mask_overlap"]
        if m["iou_median"] is None:
            ax.text(xi, 0.02, "n/a", ha="center", fontsize=8)
            continue
        ax.bar(xi, m["iou_median"], 0.5, color="#0f766e")
        ax.text(xi, m["iou_median"] + 0.02, f"{m['iou_median']:.3f}\n({m['frames_with_labels']}/{FRAMES} frames)",
                ha="center", va="bottom", fontsize=7.8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style(ax, "C. Observed SAM mask vs dataset mask_visib, median IoU\n"
              "    shared mask input for both roles (not head-specific)", "IoU")

    # D: availability / actual artwork render counts
    ax = fig.add_subplot(grid[1, 1])
    for i, (key, hatch) in enumerate((("tracking_pose_frames", ""), ("artwork_rendered_frames", "///"))):
        for j, role in enumerate(ROLES):
            pos = x + (-1.5 + 2 * i + j) * q
            for xi, o in zip(pos, objs):
                v = o["roles"][role][key]
                ax.bar(xi, v, q, color=colors[role], hatch=hatch, edgecolor="#1f2937" if hatch else "white", linewidth=0.4)
                ax.text(xi, v + 1.5, f"{v}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 132)
    ax.axhline(FRAMES, color="#6b7280", linewidth=0.8, linestyle=":")
    ax.legend(handles=[Patch(facecolor="#d1d5db", label="tracking pose (solid)"),
                       Patch(facecolor="white", edgecolor="#1f2937", hatch="///", label="artwork rendered (hatched)")],
              fontsize=8, loc="upper left", frameon=False, ncol=2)
    style(ax, "D. Availability counts of 110 frames (NOT accuracy;\n"
              "    high tracking availability does not mean accurate attachment)", "frames")

    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    runs = "   ".join(f"{o['label']}: {o['run_id']}" for o in objs)
    ax.text(0.0, 1.0, _wrap(FOOTER, 175) + "\n"
            f"Source {INPUT_SHA256[:16]}... (plan {PLAN_SHA256[:8]}, evaluator tool {EVALUATOR_TOOL_SHA256[:8]}); "
            f"head {EXPECTED_PINS['head_sha256'][:8]}, base {EXPECTED_PINS['base_sha256'][:8]}; generated {generated_utc}.\n"
            f"Runs: {runs}", fontsize=8.3, va="top", ha="left")
    fig.canvas.draw()
    check_layout(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def check_layout(fig: Any) -> None:
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "get_window_extent")):
        if not text.get_text() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 0.5 or box.y0 < bounds.y0 - 0.5 or box.x1 > bounds.x1 + 0.5 or box.y1 > bounds.y1 + 0.5:
            raise GraphError(f"graph text is clipped: {text.get_text()[:60]!r}")


# --------------------------------------------------------------------------
# outputs


def _project_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"<outside-project>/{Path(path).name}"


def _publish(path: Path, data: bytes) -> str:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise GraphError(f"{path.name} already exists; outputs are immutable") from None
    finally:
        tmp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def build(input_path: Path, expected_sha256: str, out_prefix: Path) -> dict:
    out_prefix = Path(os.path.abspath(out_prefix))
    targets = {ext: out_prefix.with_name(out_prefix.name + ext) for ext in (".json", ".csv", ".png")}
    if any(os.path.lexists(p) for p in targets.values()):
        raise GraphError("an output with this prefix already exists; outputs are immutable")
    for part in [out_prefix.parent, *out_prefix.parent.parents]:
        if part.exists() and part.is_symlink():
            raise GraphError("output path traverses a symlink")
    doc, actual = read_input(Path(input_path), expected_sha256)
    validated = validate_document(doc, source_sha256=actual)          # before any plotting/publishing
    rows = build_rows(validated)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    png = figure_png(validated, source_sha256=actual, generated_utc=generated)
    csv_data = csv_bytes(rows)
    record = {
        "schema": SCHEMA, "tool": "tools/workbench_dataset_reference_graph.py",
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "generated_utc": generated,
        "source": {"path": _project_relative(Path(input_path)), "sha256": actual, "schema": INPUT_SCHEMA,
                   "plan_sha256": PLAN_SHA256, "evaluator_tool_sha256": EVALUATOR_TOOL_SHA256,
                   "terminal_seal_sha256": validated["terminal_seal_sha256"]},
        "models": {"primary_neural": {"label": ROLE_LABELS["primary_neural"], "head_sha256": EXPECTED_PINS["head_sha256"]},
                   "frozen_baseline": {"label": ROLE_LABELS["frozen_baseline"],
                                       "base_sha256": EXPECTED_PINS["base_sha256"]}},
        "pins": dict(EXPECTED_PINS), "flags": dict(EXPECTED_FLAGS),
        "provisional_targets_px720": list(PROVISIONAL_PX720), "provisional_targets_are_gates": False,
        "objects": validated["objects"], "rows": rows, "limitations": validated["limitations"],
        "metric_spec": validated["metric_spec"], "footer": FOOTER,
        "claims": {"independent_human_accuracy": False, "phone_attachment": False, "statistical_difference": False,
                   "pooled": False, "passing_gate": False, "new_inference": False},
        "outputs": {"csv_sha256": hashlib.sha256(csv_data).hexdigest(), "png_sha256": hashlib.sha256(png).hexdigest()},
    }
    json_data = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    total = len(json_data) + len(csv_data) + len(png)
    if total > MAX_OUTPUT_BYTES:
        raise GraphError("outputs exceed their byte bound")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_prefix.parent).free - total < FREE_RESERVE_BYTES:
        raise GraphError("1 GiB free-disk reserve would be violated")
    if hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != actual:
        raise GraphError("input changed during the build; nothing published")
    hashes = {ext: _publish(targets[ext], data) for ext, data in ((".csv", csv_data), (".png", png), (".json", json_data))}
    return {"outputs": {str(targets[k]): v for k, v in hashes.items()}, "generated_utc": generated}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YCB dataset-reference diagnostic graph (CPU only, no inference).")
    parser.add_argument("--input", required=True, help="sealed dataset-reference-projection.json")
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--out-prefix", required=True, help="fresh output prefix (.json, .csv, .png are added)")
    args = parser.parse_args(argv)
    try:
        result = build(Path(args.input), args.expected_input_sha256, Path(args.out_prefix))
    except (GraphError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
