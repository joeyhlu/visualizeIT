"""Source-bound graph for the completed qualified-mask replay diagnostic (cracker).

Reads ONLY the completed evaluation
``.cache/video-workbench/derived/mask-replay-evaluation/eval-v1/replay-evaluation.json``.
Its SHA-256 is pinned and checked BEFORE parsing. The tool reads no labels,
masks, meshes or reference files and runs no model. It validates the actual
recorded schema (v2), flags, pins and the original-control proof. It then
re-derives each cell's counts and quantiles from the recorded per-frame rows,
and plots ONLY the recorded values: four cells (original vs qualified mask x
new head vs frozen baseline), never pooled, with no statistical test and no
normalisation beyond the recorded px720.

    python tools/workbench_mask_replay_graph.py --out-prefix <fresh prefix>
    (``--input`` defaults to the pinned path)

This is a DEVELOPMENT dataset-reference 2-D anchor diagnostic. The weights are
unchanged; only the opening prompt and SAM seed changed. It establishes no
independent human accuracy, phone, orientation or footprint result and no
head superiority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ".cache/video-workbench/derived/mask-replay-evaluation/eval-v1/replay-evaluation.json"
INPUT_SHA256 = "1a87b22fda4a21dfe4d05561d66d95c78851ca8f3d9d64fc24201a741bc1463f"
INPUT_SCHEMA = "workbench_mask_replay_anchor_evaluation_v2"
SCHEMA = "workbench_mask_replay_graph_v1"
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
PROMPT_SHA256 = "15dce4be95eb362c9044f808c872abd810f98f072cc8f9bd96b79439ddd308a5"
# Run IDs are not recorded inside the evaluation JSON; pinned here from the brief and labelled as such.
RUN_IDS = {"original": "run-20261021T030702-475adc39", "new": "run-20261021T060541-e8894b79"}
MASK_SETS = {"original": "sam2-workbench-01", "new": "sam2-opening-box-v1-qualified"}
CELL_ORDER = (("original_mask", "original", "primary_neural"), ("original_mask", "original", "frozen_baseline"),
              ("new_qualified_mask", "new", "primary_neural"), ("new_qualified_mask", "new", "frozen_baseline"))
ROLE_LABEL = {"primary_neural": "new head 27e06a51", "frozen_baseline": "frozen baseline f7d127ab"}
MASK_LABEL = {"original_mask": "original SAM mask", "new_qualified_mask": "qualified opening-box mask"}
DENOMINATOR = 110
TARGETS = {"median_px720": 5.0, "p95_px720": 10.0}
EXPECTED_FLAGS = {"development_diagnostic_only": True, "independent_human_accuracy": False,
                  "isolated_head_improvement_claimed": False, "orientation_or_footprint_claimed": False,
                  "original_reports_overwritten": False, "phone_quality": False, "pooled_objects": False,
                  "prompt_and_sam_seed_changed": True, "significance_tested": False, "weights_changed": False}
STRING_FLAGS = ("geometry_and_mask_closure", "inference_source_closure")
MAX_INPUT_BYTES = 8 << 20
MAX_OUTPUT_BYTES = 2 << 20
FREE_RESERVE_BYTES = 1 << 30
TOL = 1e-9
LIMITATIONS = [
    "development cached dataset-reference 2-D anchor diagnostic only (one cracker clip, adjacent frames are not "
    "independent replications)",
    "weights unchanged (head 27e06a51, base f7d127ab); only the opening prompt (RGB s000370 whole box + original "
    "points) and SAM seed changed; not new training",
    "median anchor error still misses the provisional < 5 px720 target in every cell; P95 under 10 px720 is not the "
    "whole attachment gate",
    "independent human accuracy, phone quality, orientation and footprint are unverified; no head-superiority claim",
    "no statistical test was performed; cells are not pooled",
    "availability counts (validated poses, rendered artwork) are not accuracy",
]


class GraphError(ValueError):
    """The completed evaluation does not support this graph."""


# --------------------------------------------------------------------------
# source loading (SHA before parse)


def _loads_strict(raw: bytes) -> Any:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise GraphError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise GraphError(f"non-finite JSON value {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphError(f"invalid JSON ({exc})") from None


def _contained(path: Path) -> Path:
    resolved = Path(os.path.abspath(path))
    if ROOT.resolve() not in resolved.parents:
        raise GraphError(f"{path} is outside the repository")
    return resolved


def load_source(path: Path | str = DEFAULT_INPUT) -> tuple[dict, str]:
    """Verify the pinned SHA-256 of the raw bytes BEFORE parsing."""
    p = _contained(ROOT / path if not Path(path).is_absolute() else Path(path))
    if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_INPUT_BYTES:
        raise GraphError("input must be a regular, bounded file")
    raw = p.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != INPUT_SHA256:
        raise GraphError(f"input SHA-256 {actual[:12]}... is not the pinned completed evaluation")
    doc = _loads_strict(raw)
    if not isinstance(doc, dict):
        raise GraphError("input must be a JSON object")
    return doc, actual


# --------------------------------------------------------------------------
# pure validation / extraction


def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise GraphError(msg)


def _count(v: Any, what: str) -> int:
    _req(type(v) is int and v >= 0, f"{what} must be a non-negative integer")
    return v


def _num_or_none(v: Any, what: str) -> float | None:
    if v is None:
        return None
    _req(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v)), f"{what} must be finite")
    return float(v)


def _quantiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def _same(recorded: float | None, derived: float | None, what: str) -> None:
    if derived is None:
        _req(recorded is None, f"{what} must be null when nothing was measured (never zero)")
    else:
        _req(recorded is not None and abs(recorded - derived) <= TOL * max(1.0, abs(derived)),
             f"{what} differs from the recorded per-frame rows")


def validate(doc: Any, *, source_sha256: str | None = None) -> None:
    """Document-level checks: schema, flags, pins, original control, exact four cells."""
    _req(isinstance(doc, dict) and doc.get("schema") == INPUT_SCHEMA, f"input is not {INPUT_SCHEMA}")
    if source_sha256 is not None:
        _req(source_sha256 == INPUT_SHA256, "input is not the pinned completed evaluation")
    flags = doc.get("flags")
    _req(isinstance(flags, dict) and set(flags) == set(EXPECTED_FLAGS) | set(STRING_FLAGS), "flag set differs")
    _req(all(flags[k] is v for k, v in EXPECTED_FLAGS.items()), "flags make or drop a claim (human/phone/head/"
                                                                  "significance/weights/prompt)")
    _req(all(isinstance(flags[k], str) and flags[k] for k in STRING_FLAGS), "closure flags must be non-empty text")
    pins = doc.get("pins")
    _req(isinstance(pins, dict) and pins.get("head_sha256") == HEAD_SHA256 and pins.get("base_sha256") == BASE_SHA256,
         "head/base pins differ")
    _req(isinstance(pins.get("configs"), dict) and pins["configs"].get("prompt") == PROMPT_SHA256,
         "opening-box prompt pin differs")
    control = doc.get("original_control")
    _req(isinstance(control, dict) and control.get("passed") is True and control.get("rows_reproduced") == 220
         and control.get("tolerance") == {"atol_px": 1e-08, "rtol": 0.0}, "original control did not pass 220 rows at 1e-8")
    cells = doc.get("cells")
    _req(isinstance(cells, dict) and set(cells) == {f"{m}/{r}" for m, _k, r in CELL_ORDER},
         "cells must be exactly the four mask x role cells")
    rows = doc.get("rows")
    _req(isinstance(rows, dict) and set(rows) == {"original", "new"}
         and all(isinstance(rows[k], dict) and set(rows[k]) == {"primary_neural", "frozen_baseline"} for k in rows),
         "per-frame rows must be exactly original/new x two roles")
    consumption = ((doc.get("run_authentication") or {}).get("mask_consumption") or {})
    for mask_key, kind, _role in CELL_ORDER:
        _req((consumption.get(mask_key) or {}).get("mask_set") == MASK_SETS[kind], f"{mask_key} mask set differs")


def _cell(doc: Mapping, mask_key: str, kind: str, role: str) -> dict:
    where = f"{mask_key}/{role}"
    cell = doc["cells"][where]
    _req(isinstance(cell, dict) and cell.get("frames") == DENOMINATOR, f"{where}: expected {DENOMINATOR} frames")
    _req(cell.get("reference_status_counts") == {"eligible": DENOMINATOR}, f"{where}: not all frames reference-eligible")
    are = cell.get("all_reference_eligible")
    _req(isinstance(are, dict) and are.get("denominator") == DENOMINATOR, f"{where}: denominator must be {DENOMINATOR}")
    measured = _count(are.get("measured"), f"{where}.measured")
    status = are.get("prediction_status_counts")
    _req(isinstance(status, dict) and sum(_count(v, "status") for v in status.values()) == DENOMINATOR
         and status.get("ok", 0) == measured and status == cell.get("prediction_status_counts"),
         f"{where}: prediction status counts do not account for {DENOMINATOR} frames")
    # re-derive from the recorded per-frame rows (validation only; plotted values are the recorded ones)
    row_list = doc["rows"][kind][role]
    _req(isinstance(row_list, list) and len(row_list) == DENOMINATOR, f"{where}: expected {DENOMINATOR} rows")
    _req(len({r.get("frame_id") for r in row_list}) == DENOMINATOR, f"{where}: frame ids not unique")
    errors = []
    for r in row_list:
        _req(r.get("reference_status") == "eligible", f"{where}: a row is not reference-eligible")
        e = _num_or_none(r.get("error_px720"), f"{where} error_px720")
        _req((e is None) == (r.get("prediction_status") != "ok"), f"{where}: error presence differs from status")
        if e is not None:
            _req(e >= 0, f"{where}: negative error")
            errors.append(e)
    _req(len(errors) == measured, f"{where}: measured differs from the rows")
    median, p95 = _quantiles(errors)
    rec_median = _num_or_none(are.get("conditional_median_px720"), f"{where} median")
    rec_p95 = _num_or_none(are.get("conditional_p95_px720"), f"{where} p95")
    _same(rec_median, median, f"{where} conditional median")
    _same(rec_p95, p95, f"{where} conditional P95")
    coverage = {}
    for limit in (5, 10):
        frac = _num_or_none(are.get(f"coverage_le_{limit}px720"), f"{where} coverage {limit}")
        count = sum(1 for e in errors if e <= limit)
        _req(frac is not None and 0 <= frac <= 1 and abs(frac - count / DENOMINATOR) <= TOL,
             f"{where}: coverage <= {limit} px720 differs from rows over the full denominator")
        coverage[limit] = {"count": count, "fraction": frac}
    _req(coverage[5]["count"] <= coverage[10]["count"] <= measured, f"{where}: coverage counts inconsistent")
    avail = cell.get("availability")
    _req(isinstance(avail, dict) and avail.get("frames") == DENOMINATOR, f"{where}: availability incomplete")
    poses = _count(avail.get("validated_tracking_poses"), f"{where}.validated_tracking_poses")
    rendered = _count(avail.get("render_state_rendered"), f"{where}.render_state_rendered")
    render_counts, pose_counts = avail.get("render_state_counts"), avail.get("pose_state_counts")
    _req(isinstance(render_counts, dict) and sum(_count(v, "render") for v in render_counts.values()) == DENOMINATOR
         and render_counts.get("rendered", 0) == rendered, f"{where}: render counts inconsistent")
    _req(isinstance(pose_counts, dict) and sum(_count(v, "pose") for v in pose_counts.values()) == DENOMINATOR
         and pose_counts.get("tracking", 0) == poses and poses <= DENOMINATOR, f"{where}: pose counts inconsistent")
    fixed = cell.get("fixed_visible")
    _req(isinstance(fixed, dict) and fixed.get("denominator") == DENOMINATOR, f"{where}: fixed_visible incomplete")
    return {"cell": where, "mask": mask_key, "mask_label": MASK_LABEL[mask_key], "mask_set": MASK_SETS[kind],
            "run_id": RUN_IDS[kind], "role": role, "role_label": ROLE_LABEL[role],
            "model_sha256": HEAD_SHA256 if role == "primary_neural" else BASE_SHA256,
            "denominator": DENOMINATOR, "measured": measured, "status_counts": dict(status),
            "conditional_median_px720": rec_median, "conditional_p95_px720": rec_p95,
            "coverage_le_5_count": coverage[5]["count"], "coverage_le_5_fraction": coverage[5]["fraction"],
            "coverage_le_10_count": coverage[10]["count"], "coverage_le_10_fraction": coverage[10]["fraction"],
            "validated_tracking_poses": poses, "artwork_rendered": rendered,
            "fixed_visible_plotted": False}


def extract_points(doc: Any, *, source_sha256: str | None = None) -> list[dict]:
    """Validated graph points for the four cells in fixed order (recorded values only)."""
    validate(doc, source_sha256=source_sha256)
    return [_cell(doc, m, k, r) for m, k, r in CELL_ORDER]


def provenance(doc: Mapping, source_sha256: str, points: Sequence[Mapping]) -> dict:
    """Deterministic source receipt (no wall-clock time)."""
    return {
        "schema": SCHEMA, "tool": "tools/workbench_mask_replay_graph.py",
        "source": {"path": DEFAULT_INPUT, "sha256": source_sha256, "schema": INPUT_SCHEMA,
                   "seal_sha256": doc.get("seal_sha256"), "terminal_record_sha256": doc.get("terminal_record_sha256"),
                   "plan_sha256": doc.get("plan_sha256"), "copied_or_rewritten": False},
        "run_ids": dict(RUN_IDS), "run_ids_source": "pinned by this tool (not recorded in the evaluation JSON)",
        "mask_sets": dict(MASK_SETS),
        "models": {"head_sha256": HEAD_SHA256, "base_sha256": BASE_SHA256, "weights_changed": False},
        "prompt_sha256": PROMPT_SHA256,
        "original_control": {"passed": True, "rows_reproduced": 220, "atol_px": 1e-08, "rtol": 0.0},
        "flags": {k: doc["flags"][k] for k in EXPECTED_FLAGS},
        "targets_px720": dict(TARGETS), "targets_are_provisional": True,
        "points": [dict(p) for p in points], "limitations": list(LIMITATIONS),
        "claims": {"statistical_test": False, "pooled": False, "independent_human_accuracy": False, "phone": False,
                   "head_superiority": False, "orientation_or_footprint": False},
    }


def csv_bytes(points: Sequence[Mapping]) -> bytes:
    fields = ["cell", "mask_label", "mask_set", "run_id", "role", "model_sha256", "denominator", "measured",
              "conditional_median_px720", "conditional_p95_px720", "coverage_le_5_count", "coverage_le_5_fraction",
              "coverage_le_10_count", "coverage_le_10_fraction", "validated_tracking_poses", "artwork_rendered"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for p in points:
        w.writerow({k: ("" if p[k] is None else repr(p[k]) if isinstance(p[k], float) else p[k]) for k in fields})
    return buf.getvalue().encode("utf-8")


def json_bytes(record: Mapping) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# lazy matplotlib render


def render_png(points: Sequence[Mapping], source_sha256: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"primary_neural": "#2563eb", "frozen_baseline": "#9ca3af"}
    labels = [f"{p['mask_label']}\n{p['role_label']}" for p in points]
    fig = plt.figure(figsize=(16, 12), dpi=100, facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.42, left=0.15, right=0.97, top=0.88, bottom=0.06)
    fig.suptitle("Cracker development diagnostic: opening box prompt + original points", fontsize=17,
                 fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.935, "Weights unchanged (head 27e06a51, base f7d127ab; not new training). Original vs qualified "
             "SAM mask; per cell, never pooled; no statistical test.", fontsize=11, color="#b91c1c", ha="left")
    fig.legend(handles=[Patch(color=colors[r], label=ROLE_LABEL[r]) for r in ("primary_neural", "frozen_baseline")],
               loc="upper left", bbox_to_anchor=(0.012, 0.925), ncol=2, fontsize=10.5, frameon=False)
    y = list(range(len(points)))[::-1]

    def style(ax, title):
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_yticks(y, labels, fontsize=9.5)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # 1. conditional anchor median / P95
    ax = fig.add_subplot(grid[0, 0])
    h = 0.36
    xmax = max([TARGETS["p95_px720"]] + [p["conditional_p95_px720"] or 0 for p in points]) * 1.3
    for yi, p in zip(y, points):
        for off, key, alpha in ((h / 2, "conditional_median_px720", 1.0), (-h / 2, "conditional_p95_px720", 0.45)):
            v = p[key]
            if v is None:
                ax.text(0.2, yi + off, "unmeasured", va="center", fontsize=8.5, color="#6b7280")
                continue
            ax.barh(yi + off, v, h, color=colors[p["role"]], alpha=alpha)
            ax.text(v + xmax * 0.01, yi + off, f"{'median' if 'median' in key else 'P95'} {v:.2f}", va="center",
                    fontsize=8.5)
        ax.text(xmax * 0.99, yi, f"{p['measured']}/{p['denominator']} measured", va="center", ha="right", fontsize=8,
                color="#374151")
    for key, label in (("median_px720", "provisional median target 5"), ("p95_px720", "provisional P95 target 10")):
        ax.axvline(TARGETS[key], color="#b45309", linestyle="--", linewidth=1.1)
        ax.text(TARGETS[key], len(points) - 0.45, label, color="#b45309", fontsize=8.5, ha="center", va="bottom")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(points) - 0.1)
    ax.set_xlabel("anchor error, px at 720p (conditional on measured frames)", fontsize=9.5)
    style(ax, "1. Conditional anchor error: median (solid) and P95 (light), px720")

    # 2. validated poses vs artwork rendered
    ax = fig.add_subplot(grid[0, 1])
    for yi, p in zip(y, points):
        for off, key, hatch, name in ((h / 2, "validated_tracking_poses", "", "validated poses"),
                                      (-h / 2, "artwork_rendered", "///", "artwork rendered")):
            v = p[key]
            ax.barh(yi + off, v, h, color=colors[p["role"]], hatch=hatch, edgecolor="#1f2937" if hatch else "white",
                    linewidth=0.4)
            ax.text(v + 1.5, yi + off, f"{name} {v}/{DENOMINATOR}", va="center", fontsize=8.5)
    ax.set_xlim(0, DENOMINATOR * 1.55)
    ax.set_xlabel("frames of 110 (availability, NOT accuracy)", fontsize=9.5)
    style(ax, "2. Validated poses (solid) vs artwork rendered (hatched)")

    # 3. failure-aware coverage over 110
    ax = fig.add_subplot(grid[1, 0])
    for yi, p in zip(y, points):
        for off, limit, hatch in ((h / 2, 10, ""), (-h / 2, 5, "///")):
            c, f = p[f"coverage_le_{limit}_count"], p[f"coverage_le_{limit}_fraction"]
            ax.barh(yi + off, 100 * f, h, color=colors[p["role"]], hatch=hatch,
                    edgecolor="#1f2937" if hatch else "white", linewidth=0.4)
            ax.text(100 * f + 1.2, yi + off, f"<= {limit}: {c}/{DENOMINATOR} ({100 * f:.1f}%)", va="center", fontsize=8.5)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of all 110 reference-eligible frames (the failed frame counts as not covered)", fontsize=9.5)
    style(ax, "3. Failure-aware coverage <= 10 px720 (solid), <= 5 px720 (hatched)")

    # 4. limitations / source / control
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    import textwrap
    lines = ["4. Limitations, source and control", ""]
    for item in LIMITATIONS:
        lines.extend(textwrap.wrap("- " + item, 72))
    lines += ["", f"Original control: 220 original rows reproduced at 1e-8 px.",
              f"Runs: original {RUN_IDS['original']} ({MASK_SETS['original']})",
              f"      new {RUN_IDS['new']} ({MASK_SETS['new']})",
              f"Prompt {PROMPT_SHA256[:16]}...  Source {source_sha256[:16]}..."]
    ax.text(-0.25, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9.5, transform=ax.transAxes)
    fig.canvas.draw()
    check_layout(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


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
# bounded, atomic, never-overwriting publication


def _budget_check(total: int, target_dir: Path) -> None:
    from bench import workbench_cli as cli          # pure helper; imports neither torch nor SAM
    ws = cli.Workspace(ROOT / ".cache/video-workbench")
    ws.require_budget(extra_derived=total)           # 1 GiB derived+scratch and 1 GiB free reserve (workspace disk)
    if shutil.disk_usage(target_dir).free - total < FREE_RESERVE_BYTES:
        raise GraphError("1 GiB free-disk reserve would be violated at the output location")


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


def build(input_path: Path | str, out_prefix: Path | str) -> dict:
    prefix = _contained(ROOT / out_prefix if not Path(out_prefix).is_absolute() else Path(out_prefix))
    targets = {ext: prefix.with_name(prefix.name + ext) for ext in (".png", ".json", ".csv")}
    if any(os.path.lexists(t) for t in targets.values()):
        raise GraphError("an output with this prefix already exists; outputs are immutable")
    for part in [prefix.parent, *prefix.parent.parents]:
        if part.exists() and part.is_symlink():
            raise GraphError("output path traverses a symlink")
    doc, sha = load_source(input_path)
    points = extract_points(doc, source_sha256=sha)               # before any rendering or writing
    png = render_png(points, sha)
    record = provenance(doc, sha, points)
    csv_data = csv_bytes(points)
    record["outputs"] = {"png_sha256": hashlib.sha256(png).hexdigest(), "csv_sha256": hashlib.sha256(csv_data).hexdigest(),
                         "json": "not self-hashed; hash externally"}
    json_data = json_bytes(record)
    total = len(png) + len(csv_data) + len(json_data)
    if total > MAX_OUTPUT_BYTES:
        raise GraphError("new exports exceed 2 MiB")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _budget_check(total, prefix.parent)
    if load_source(input_path)[1] != sha:
        raise GraphError("input changed during the build; nothing published")
    hashes = {ext: _publish(targets[ext], data) for ext, data in ((".png", png), (".csv", csv_data), (".json", json_data))}
    return {"outputs": {str(targets[k].relative_to(ROOT.resolve())): v for k, v in hashes.items()}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Source-bound qualified-mask replay graph (no inference).")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-prefix", required=True, help="fresh repository-contained prefix (.png/.json/.csv)")
    args = parser.parse_args(argv)
    try:
        result = build(args.input, args.out_prefix)
    except (GraphError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
