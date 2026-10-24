"""Consolidated six-case full-video paired suite graph (no inference).

Reads exactly the six sealed ``workbench_paired_comparison_v1`` reports produced by
``tools/workbench_comparison.py`` (pinned tool SHA-256). It reuses their already
validated numbers, recomputes nothing about the models and writes one aggregate
JSON, one CSV and one PNG at a fresh, immutable output prefix.

The model identities are actual. The reports are only references, and no
inference is performed. Pose availability and raw artwork-render fraction are
not accuracy. Independent attachment accuracy is unverified, because there are
no independent image landmarks. The roles ran serially on a shared host, so
latency is not a causal performance comparison. 720p attachment median/P95 are
unavailable, phone performance is unmeasured, clips are never pooled and no
statistical significance is claimed.

    python tools/workbench_suite_comparison.py \
        --comparison artifacts/video-workbench/comparisons/mug-full-27e06-vs-baseline-v1.json \
        ... (all six) ... \
        --out-prefix artifacts/video-workbench/comparisons/suite-full-27e06-vs-baseline-v1
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
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "workbench_paired_suite_graph_v1"
SOURCE_SCHEMA = "workbench_paired_comparison_v1"
SOURCE_TOOL = "tools/workbench_comparison.py"
SOURCE_TOOL_SHA256 = "223350da74bfff9075b501510df01dbaa7bfb802793b37e119d0048833fb8784"
SOURCE_TOOL_VERSION = "1"
PRIMARY_HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_CHECKPOINT_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
ROLES = ("primary_neural", "frozen_baseline")
ROLE_LABELS = {"primary_neural": "New head 27e06a51", "frozen_baseline": "Frozen baseline f7d127ab"}
ROLE_STATUS = {"primary_neural": "verified_installed_head", "frozen_baseline": "verified_pinned_frozen_base"}
# Frozen six-case development suite (fixed order): expected full windows plus the
# sealed ORIGINAL run ID and exact source comparison-report SHA-256 per case.
# A later trial needs a separately reviewed source version; nothing selects "latest".
SUITE = (
    {"clip_id": "dev-mug", "file_prefix": "mug", "label": "Mug",
     "clip_sha256": "7779a0c32117ab2e3c52dbe5fbf141ad832056440ada1444695de626b1f7600a", "frames": 240,
     "run_id": "run-20261021T002156-1b2de77d",
     "report_sha256": "9559bec78864038b481d80c8043164332ed3d0e29a69eea589b45b4adfa3788d"},
    {"clip_id": "dev-keyboard", "file_prefix": "keyboard", "label": "Keyboard",
     "clip_sha256": "e3c954cb7a279f1e45d028b8cc615023ca9bb790c83bacfbc6b2d1e122afe0fe", "frames": 240,
     "run_id": "run-20261021T013943-538cfa03",
     "report_sha256": "17ace1cafb74dd3ac25d7bcbbfa6d42ef91fb38dc2c6afbd2ba7c3a1a2bb8cc5"},
    {"clip_id": "dev-ranch", "file_prefix": "ranch", "label": "Ranch",
     "clip_sha256": "0f6250e71ba55476d0c779c9f2b3b5648166456bec7b23bf199965149b0f63b4", "frames": 240,
     "run_id": "run-20261021T020454-01d11a29",
     "report_sha256": "a315056b47ddf4842237fc8074efc52b856f08c22125d9c8a294f703003258e5"},
    {"clip_id": "dev-ycb-003-cracker-box", "file_prefix": "ycb-cracker-box", "label": "YCB cracker box",
     "clip_sha256": "afbb18691fef64ee595c1c22e603f1999ea9b19f5e0c740b47db811fb2b646c4", "frames": 110,
     "run_id": "run-20261021T030702-475adc39",
     "report_sha256": "184d0904061a08554e056d566ed67af87cc8b7eb45eebdbdf82ab9b3a73e9feb"},
    {"clip_id": "dev-ycb-005-tomato-soup-can", "file_prefix": "ycb-soup-can", "label": "YCB soup can",
     "clip_sha256": "75169a1c0ddc9ee1be12922317fc285723bffa67bd65feb0015a074770900fd8", "frames": 110,
     "run_id": "run-20261021T031729-0986ef46",
     "report_sha256": "efd867d7003692f2c060dc6fcb14592db280cb5d12309c628b686b444c9af69d"},
    {"clip_id": "dev-ycb-040-large-marker", "file_prefix": "ycb-marker", "label": "YCB marker",
     "clip_sha256": "72ed3b4d18e1d07c905026ee8b79baa6108add73afb32a2b4ae51c9b76a3d076", "frames": 110,
     "run_id": "run-20261021T032932-539343a7",
     "report_sha256": "37d8844aa60eca066be782d16f829586acca4a5b2f28f03ca48050c1bf5b55f8"},
)
SUITE_BY_CLIP = {case["clip_id"]: case for case in SUITE}
EXPECTED_ROWS_PER_ROLE = sum(case["frames"] for case in SUITE)        # 1050
SOURCE_NAME = "{prefix}-full-27e06-vs-baseline-v1.json"
MAX_SOURCE_BYTES = 1 << 20
MAX_OUTPUT_BYTES = 8 << 20
FREE_RESERVE_BYTES = 1 << 30
FRACTION_TOLERANCE = 1e-12
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^run-\d{8}T\d{6}-[0-9a-f]{8}$")
FALSE_CLAIMS = ("accuracy", "causal_speed", "leaderboard", "mobile_readiness", "new_inference", "pooled_quality")
STATEMENTS = {
    "accuracy": "UNVERIFIED — no independent image landmarks",
    "availability": "pose availability and raw artwork-render fraction are NOT accuracy",
    "latency": "roles ran serially on a shared host; latency is not a causal performance comparison",
    "attachment_720p": "720p attachment median/P95 unavailable",
    "phone": "phone performance unmeasured",
    "pooling": "clips are not pooled; no statistical significance is claimed",
    "inference": "no inference performed; numbers are reused from the sealed comparison reports",
}


class SuiteError(ValueError):
    """The comparison reports do not form the exact frozen six-case suite."""


# --------------------------------------------------------------------------
# strict reading


def loads_strict(raw: bytes, what: str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise SuiteError(f"{what}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise SuiteError(f"{what}: non-finite JSON value {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiteError(f"{what}: invalid JSON ({exc})") from None


def read_report(path: Path) -> tuple[dict, str, bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SuiteError(f"{path.name}: comparison report is missing or not a regular file")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise SuiteError(f"{path.name}: comparison report exceeds its size bound")
    raw = path.read_bytes()
    doc = loads_strict(raw, path.name)
    if not isinstance(doc, dict):
        raise SuiteError(f"{path.name}: comparison report must be a JSON object")
    return doc, hashlib.sha256(raw).hexdigest(), raw


def project_relative(value: Any) -> str:
    """Project-relative POSIX path, or a marked basename for paths outside the project."""
    if not isinstance(value, str) or not value:
        raise SuiteError("input file path must be a non-empty string")
    path = Path(value)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return f"<outside-project>/{path.name}"


# --------------------------------------------------------------------------
# pure validation (unit-testable with mutated authentic documents)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuiteError(message)


def _int(value: Any, what: str) -> int:
    _require(type(value) is int and value >= 0, f"{what} must be a non-negative integer")
    return value


def _finite(value: Any, what: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
             and float(value) >= 0, f"{what} must be a finite non-negative number")
    return float(value)


def _counts(value: Any, keys: Sequence[str], total: int, what: str) -> dict:
    _require(isinstance(value, dict) and set(value) == set(keys), f"{what} must have exactly {sorted(keys)}")
    counts = {k: _int(value[k], f"{what}.{k}") for k in keys}
    _require(sum(counts.values()) == total, f"{what} does not account for all {total} scheduled frames")
    return counts


def validate_role(role: str, body: Any, frames: int, where: str) -> dict:
    w = f"{where} {role}"
    _require(isinstance(body, dict) and body.get("role") == role, f"{w}: role record missing or mislabelled")
    _require(_int(body.get("scheduled_frames"), f"{w}.scheduled_frames") == frames,
             f"{w}: scheduled frames differ from the frozen window {frames}")
    _require(_int(body.get("result_rows"), f"{w}.result_rows") == frames, f"{w}: result rows differ from {frames}")
    _require(body.get("missing_rows") == 0 and body.get("missing_frame_ids") == [], f"{w}: missing rows")
    _require(body.get("stage_status") == "completed", f"{w}: stage not completed")
    ident = body.get("identity")
    _require(isinstance(ident, dict), f"{w}: identity missing")
    _require(ident.get("status") == ROLE_STATUS[role] and ident.get("identity_verified") is True,
             f"{w}: identity not verified as {ROLE_STATUS[role]}")
    _require(ident.get("base_checkpoint_sha256") == BASE_CHECKPOINT_SHA256
             and ident.get("row_base_checkpoint_sha256") == [BASE_CHECKPOINT_SHA256],
             f"{w}: base checkpoint differs from the frozen pin")
    _require(ident.get("distinct_row_identities") == 1, f"{w}: rows mix identities")
    if role == "primary_neural":
        _require(ident.get("installed_head_sha256") == PRIMARY_HEAD_SHA256
                 and ident.get("requested_head_sha256_from_run_metadata") == PRIMARY_HEAD_SHA256
                 and ident.get("requested_matches_installed") is True
                 and ident.get("row_installed_head_sha256") == [PRIMARY_HEAD_SHA256],
                 f"{w}: installed head differs from the main head pin")
    else:
        _require(ident.get("installed_head_sha256") is None and ident.get("head") == "frozen_base_pose_head"
                 and ident.get("row_installed_head_sha256", []) in ([], [None]),
                 f"{w}: the frozen baseline must be the frozen base pose head with no installed head")
    pose = _counts(body.get("pose_state_counts"), ("tracking", "recovering", "lost", "blocked", "unavailable"),
                   frames, f"{w}.pose_state_counts")
    render = _counts(body.get("render_state_counts"), ("rendered", "suppressed", "unavailable"), frames,
                     f"{w}.render_state_counts")
    _counts(body.get("mask_state_counts"), ("observed", "rejected", "blocked", "unavailable"), frames,
            f"{w}.mask_state_counts")
    available = _int(body.get("pose_available_frames"), f"{w}.pose_available_frames")
    rendered = _int(body.get("artwork_rendered_frames"), f"{w}.artwork_rendered_frames")
    _require(available == pose["tracking"], f"{w}: pose_available_frames differs from tracking rows")
    _require(rendered == render["rendered"], f"{w}: artwork_rendered_frames differs from rendered rows")
    _require(body.get("recovering_frames") == pose["recovering"] and body.get("lost_frames") == pose["lost"],
             f"{w}: recovering/lost counts inconsistent")
    for key, count in (("pose_availability_fraction", available), ("raw_artwork_render_fraction", rendered)):
        _require(abs(_finite(body.get(key), f"{w}.{key}") - count / frames) <= FRACTION_TOLERANCE,
                 f"{w}: {key} differs from its counts")
    lat = body.get("latency")
    _require(isinstance(lat, dict) and lat.get("source") == "measured_this_run",
             f"{w}: latency must be measured in this run (no cached timings)")
    e2e = lat.get("end_to_end_latency_ms")
    _require(isinstance(e2e, dict), f"{w}: end-to-end latency missing")
    median, p95 = _finite(e2e.get("median_ms"), f"{w}.median_ms"), _finite(e2e.get("p95_ms"), f"{w}.p95_ms")
    measured = _int(e2e.get("measured_frames"), f"{w}.measured_frames")
    _require(0 < measured <= frames and e2e.get("scheduled_frames") == frames and median <= p95,
             f"{w}: end-to-end latency summary inconsistent")
    return {"role": role, "scheduled_frames": frames, "pose_available_frames": available,
            "pose_availability_fraction": available / frames, "artwork_rendered_frames": rendered,
            "raw_artwork_render_fraction": rendered / frames, "recovering_frames": pose["recovering"],
            "lost_frames": pose["lost"], "pose_state_counts": pose, "render_state_counts": render,
            "end_to_end_median_ms": median, "end_to_end_p95_ms": p95, "end_to_end_measured_frames": measured,
            "latency_cache_status": lat.get("cache_status"), "identity_status": ident["status"],
            "installed_head_sha256": ident.get("installed_head_sha256"),
            "base_checkpoint_sha256": ident["base_checkpoint_sha256"]}


def validate_report(doc: Any, *, expected_clip: str | None = None, source_name: str | None = None,
                    source_sha256: str | None = None) -> dict:
    """One sealed comparison report -> one validated suite row (no recomputation)."""
    _require(isinstance(doc, dict), "comparison report must be an object")
    where = source_name or "report"
    _require(doc.get("schema") == SOURCE_SCHEMA and doc.get("scope") == "full", f"{where}: not a full-scope "
             f"{SOURCE_SCHEMA}")
    _require(doc.get("tool") == SOURCE_TOOL and doc.get("tool_sha256") == SOURCE_TOOL_SHA256
             and doc.get("tool_version") == SOURCE_TOOL_VERSION, f"{where}: produced by another comparison tool")
    _require(doc.get("pinned_primary_head_sha256") == PRIMARY_HEAD_SHA256, f"{where}: wrong main head pin")
    _require(doc.get("pinned_base_checkpoint_sha256") == BASE_CHECKPOINT_SHA256, f"{where}: wrong base pin")
    claims = doc.get("claims")
    _require(isinstance(claims, dict) and all(claims.get(k) is False for k in FALSE_CLAIMS),
             f"{where}: report makes a claim this graph cannot carry")
    runs = doc.get("runs")
    _require(isinstance(runs, list) and len(runs) == 1, f"{where}: exactly one run per report is required")
    run = runs[0]
    _require(isinstance(run, dict), f"{where}: run record malformed")
    clip_id = run.get("clip_id")
    _require(clip_id in SUITE_BY_CLIP, f"{where}: clip {clip_id!r} is not in the frozen six-case suite")
    _require(expected_clip is None or clip_id == expected_clip, f"{where}: report is for {clip_id}, "
             f"expected {expected_clip}")
    case = SUITE_BY_CLIP[clip_id]
    if source_name is not None:
        _require(source_name == SOURCE_NAME.format(prefix=case["file_prefix"]),
                 f"{where}: file name differs from the sealed {SOURCE_NAME.format(prefix=case['file_prefix'])}")
    frames = case["frames"]
    _require(isinstance(run.get("run_id"), str) and RUN_ID.fullmatch(run["run_id"]) is not None,
             f"{where}: run id malformed")
    _require(run["run_id"] == case["run_id"], f"{where}: run {run['run_id']} is not the sealed original run "
             f"{case['run_id']} for {clip_id}")
    if source_sha256 is not None:
        _require(source_sha256 == case["report_sha256"], f"{where}: report SHA-256 {source_sha256[:12]}... differs from "
                 f"the sealed original report {case['report_sha256'][:12]}...")
    _require(run.get("scope") == "full" and run.get("mode") == "regression", f"{where}: run is not full regression")
    _require(run.get("run_completeness") == "complete" and run.get("report_status") == "completed",
             f"{where}: run is not complete")
    _require(run.get("neural_comparison") == "verified_primary_vs_frozen_baseline", f"{where}: pairing unverified")
    _require(_int(run.get("scheduled_frames"), f"{where}.scheduled_frames") == frames,
             f"{where}: scheduled frames differ from the frozen window {frames}")
    _require(isinstance(run.get("roles_scheduled"), list) and sorted(run["roles_scheduled"]) == sorted(ROLES),
             f"{where}: scheduled roles must be exactly the paired neural roles")
    roles = run.get("roles")
    _require(isinstance(roles, dict) and set(roles) == set(ROLES), f"{where}: roles must be exactly {list(ROLES)}")
    parse = run.get("parse")
    _require(isinstance(parse, dict) and parse.get("invalid_lines") == 0 and parse.get("report_file") is True
             and parse.get("results_file") is True, f"{where}: source run parse incomplete")
    files = run.get("input_files")
    _require(isinstance(files, list) and len(files) == 4, f"{where}: input file identities missing")
    identities = {}
    for row in files:
        _require(isinstance(row, dict) and isinstance(row.get("sha256"), str) and HEX64.fullmatch(row["sha256"])
                 is not None, f"{where}: input file identity malformed")
        name = Path(str(row.get("path"))).name
        _require(name in ("run.json", "clip.json", "results.jsonl", "report.json") and name not in identities,
                 f"{where}: unexpected or duplicate input file {name}")
        identities[name] = {"path": project_relative(row["path"]), "sha256": row["sha256"], "bytes": row.get("bytes")}
    _require(identities["clip.json"]["sha256"] == case["clip_sha256"], f"{where}: clip record differs from the "
             "frozen suite clip")
    _require(f"/runs/{run['run_id']}/" in "/" + identities["run.json"]["path"] or
             identities["run.json"]["path"].startswith("<outside-project>"), f"{where}: run.json is not this run's")
    role_rows = {role: validate_role(role, roles[role], frames, where) for role in ROLES}
    paired = run.get("paired_frames")
    _require(isinstance(paired, dict) and paired.get("scheduled_frames") == frames, f"{where}: paired counts missing")
    for kind, prim, base in (("pose", "pose_available_frames", "pose_available_frames"),
                             ("render", "artwork_rendered_frames", "artwork_rendered_frames")):
        both_key = "pose_both_available" if kind == "pose" else "render_both"
        both, p_only = _int(paired.get(both_key), both_key), _int(paired.get(f"{kind}_primary_only"), kind)
        b_only, neither = _int(paired.get(f"{kind}_baseline_only"), kind), _int(paired.get(f"{kind}_neither"), kind)
        _require(both + p_only + b_only + neither == frames, f"{where}: paired {kind} counts do not cover {frames}")
        _require(both + p_only == role_rows["primary_neural"][prim]
                 and both + b_only == role_rows["frozen_baseline"][base], f"{where}: paired {kind} counts inconsistent")
    acc = run.get("accuracy")
    _require(isinstance(acc, dict) and acc.get("status") == "unverified_no_independent_annotations"
             and acc.get("attachment_median_px_720") is None and acc.get("attachment_p95_px_720") is None,
             f"{where}: accuracy must remain unverified with no 720p attachment values")
    return {"clip_id": clip_id, "label": case["label"], "file_prefix": case["file_prefix"], "frames_per_role": frames,
            "run_id": run["run_id"], "mode": run["mode"], "scope": run["scope"],
            "source_report": {"name": source_name, "sha256": source_sha256},
            "source_tool": {"path": SOURCE_TOOL, "sha256": doc["tool_sha256"], "version": doc["tool_version"]},
            "run_input_files": identities,
            "admission_identity": "not recorded by the comparison report; bound indirectly through run.json SHA-256",
            "roles": role_rows,
            "paired_frames": {k: paired[k] for k in sorted(paired) if k != "note"},
            "run_telemetry": run.get("run_telemetry"),
            "accuracy": {"status": acc["status"], "statement": STATEMENTS["accuracy"],
                         "attachment_median_px_720": None, "attachment_p95_px_720": None}}


def validate_suite(rows: Sequence[Mapping], *, require_source_seals: bool = True) -> list[dict]:
    """Exactly the six frozen clips, once each, in the fixed suite order. With
    ``require_source_seals`` (always used by ``build``), every row must carry the
    externally computed SHA-256 of its source report, equal to the sealed pin."""
    _require(len(rows) == len(SUITE), f"exactly {len(SUITE)} comparison reports are required, got {len(rows)}")
    by_clip: dict[str, Mapping] = {}
    for row in rows:
        _require(row["clip_id"] not in by_clip, f"duplicate report for clip {row['clip_id']}")
        by_clip[row["clip_id"]] = row
    _require(set(by_clip) == set(SUITE_BY_CLIP), f"suite clips differ: missing {sorted(set(SUITE_BY_CLIP) - set(by_clip))}")
    for clip_id, row in by_clip.items():
        case = SUITE_BY_CLIP[clip_id]
        _require(row["run_id"] == case["run_id"], f"{clip_id}: run {row['run_id']} is not the sealed original run")
        sha = row["source_report"]["sha256"]
        if require_source_seals or sha is not None:
            _require(sha == case["report_sha256"], f"{clip_id}: source report is not the sealed original report")
    run_ids = [row["run_id"] for row in rows]
    _require(len(set(run_ids)) == len(run_ids), "two reports name the same run")
    hashes = [row["source_report"]["sha256"] for row in rows if row["source_report"]["sha256"]]
    _require(len(set(hashes)) == len(hashes), "two inputs are the same report")
    ordered = [dict(by_clip[case["clip_id"]]) for case in SUITE]
    for role in ROLES:
        total = sum(row["roles"][role]["scheduled_frames"] for row in ordered)
        _require(total == EXPECTED_ROWS_PER_ROLE, f"{role}: suite covers {total} frames, expected {EXPECTED_ROWS_PER_ROLE}")
    return ordered


def aggregate(ordered: Sequence[Mapping]) -> dict:
    return {
        "schema": SCHEMA, "tool": "tools/workbench_suite_comparison.py",
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": {"primary_neural": {"label": ROLE_LABELS["primary_neural"], "installed_head_sha256": PRIMARY_HEAD_SHA256,
                                      "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256},
                   "frozen_baseline": {"label": ROLE_LABELS["frozen_baseline"],
                                       "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256}},
        "source_tool": {"path": SOURCE_TOOL, "sha256": SOURCE_TOOL_SHA256, "version": SOURCE_TOOL_VERSION},
        "suite": [{k: case[k] for k in ("clip_id", "label", "clip_sha256", "frames", "run_id", "report_sha256")}
                  for case in SUITE],
        "source_seals": {case["clip_id"]: {"run_id": case["run_id"], "report_sha256": case["report_sha256"],
                                           "verified": True} for case in SUITE},
        "rows_per_role": EXPECTED_ROWS_PER_ROLE, "rows_total": EXPECTED_ROWS_PER_ROLE * len(ROLES),
        "clips": list(ordered), "statements": dict(STATEMENTS),
        "claims": {"accuracy": False, "quality_improvement": False, "pooled": False, "statistical_significance": False,
                   "causal_latency": False, "phone_performance": False, "new_inference": False},
    }


def csv_bytes(ordered: Sequence[Mapping]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["clip_id", "label", "run_id", "role", "model_sha256", "scheduled_frames", "pose_available_frames",
                     "pose_availability_fraction", "artwork_rendered_frames", "raw_artwork_render_fraction",
                     "recovering_frames", "lost_frames", "end_to_end_median_ms", "end_to_end_p95_ms",
                     "end_to_end_measured_frames", "accuracy", "source_report_sha256", "sealed_run_id",
                     "sealed_report_sha256"])
    for row in ordered:
        case = SUITE_BY_CLIP[row["clip_id"]]
        for role in ROLES:
            r = row["roles"][role]
            writer.writerow([row["clip_id"], row["label"], row["run_id"], role,
                             r["installed_head_sha256"] or r["base_checkpoint_sha256"], r["scheduled_frames"],
                             r["pose_available_frames"], f"{r['pose_availability_fraction']:.6f}",
                             r["artwork_rendered_frames"], f"{r['raw_artwork_render_fraction']:.6f}",
                             r["recovering_frames"], r["lost_frames"], f"{r['end_to_end_median_ms']:.1f}",
                             f"{r['end_to_end_p95_ms']:.1f}", r["end_to_end_measured_frames"],
                             "UNVERIFIED (no independent image landmarks)", row["source_report"]["sha256"],
                             case["run_id"], case["report_sha256"]])
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# plot


def figure_png(ordered: Sequence[Mapping]) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"primary_neural": "#2563eb", "frozen_baseline": "#9ca3af"}
    labels = [f"{row['label']}\n{row['frames_per_role']} frames" for row in ordered]
    narrow_labels = [f"{row['label'].replace('YCB ', 'YCB' + chr(10))}\n{row['frames_per_role']} frames" for row in ordered]
    x = np.arange(len(ordered))
    width = 0.38
    fig = plt.figure(figsize=(13.0, 9.0), dpi=150, facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.24], hspace=0.62, wspace=0.12,
                            left=0.055, right=0.985, top=0.865, bottom=0.03)
    fig.suptitle("Full-video paired suite: new head 27e06a51 vs frozen baseline f7d127ab (6 development clips)",
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.928, "Per clip, never pooled. Availability and render fraction are NOT accuracy. "
             "Accuracy: UNVERIFIED — no independent image landmarks.", fontsize=10.5, color="#b91c1c", ha="left")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=colors[r], label=ROLE_LABELS[r]) for r in ROLES], loc="upper left",
               bbox_to_anchor=(0.012, 0.922), ncol=2, fontsize=9.5, frameon=False)

    def bars(ax, key, title):
        for offset, role in ((-width / 2, "primary_neural"), (width / 2, "frozen_baseline")):
            values = [100.0 * row["roles"][role][key] for row in ordered]
            rects = ax.bar(x + offset, values, width, color=colors[role], label=ROLE_LABELS[role])
            for rect, value in zip(rects, values):
                ax.text(rect.get_x() + rect.get_width() / 2, min(value, 100) + 1.2, f"{value:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("% of scheduled frames", fontsize=9)
        ax.set_xticks(x, narrow_labels, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    ax_pose = fig.add_subplot(grid[0, 0])
    bars(ax_pose, "pose_availability_fraction", "Pose availability (tracking rows / scheduled)")
    ax_render = fig.add_subplot(grid[0, 1])
    bars(ax_render, "raw_artwork_render_fraction", "Raw artwork render fraction (rendered / scheduled)")
    ax_render.set_ylabel("")

    ax_lat = fig.add_subplot(grid[1, :])
    top = max(row["roles"][r]["end_to_end_p95_ms"] for row in ordered for r in ROLES) / 1000.0
    for offset, role in ((-width / 2, "primary_neural"), (width / 2, "frozen_baseline")):
        med = [row["roles"][role]["end_to_end_median_ms"] / 1000.0 for row in ordered]
        p95 = [row["roles"][role]["end_to_end_p95_ms"] / 1000.0 for row in ordered]
        ax_lat.bar(x + offset, med, width, color=colors[role])
        ax_lat.scatter(x + offset, p95, marker="_", s=420, linewidths=2.4, color="#111827", zorder=3)
        for xi, m, p in zip(x + offset, med, p95):
            ax_lat.vlines(xi, m, p, color="#111827", linewidth=1.0, zorder=2)
            ax_lat.text(xi, p + top * 0.025, f"{m:.1f} / {p:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax_lat.scatter([], [], marker="_", s=420, linewidths=2.4, color="#111827", label="P95 (tick)")
    ax_lat.set_ylim(0, top * 1.25)
    ax_lat.set_title("End-to-end latency per frame, seconds (bar = median, tick = P95; labels median / P95). "
                     "Serial shared host: not a causal comparison.", fontsize=11.5, loc="left")
    ax_lat.set_ylabel("seconds", fontsize=9)
    ax_lat.set_xticks(x, labels, fontsize=8.5)
    ax_lat.legend(loc="upper right", fontsize=8.5, ncol=3, frameon=True)
    ax_lat.grid(axis="y", alpha=0.25)
    ax_lat.spines[["top", "right"]].set_visible(False)

    ax_note = fig.add_subplot(grid[2, :])
    ax_note.axis("off")
    runs = [f"{row['label']}: {row['run_id']}" for row in ordered]
    ax_note.text(0.0, 1.0,
                 "Independent attachment accuracy: UNVERIFIED — no independent image landmarks.   "
                 "720p attachment median/P95: unavailable.   Phone performance: unmeasured.\n"
                 f"No inference performed; numbers reused from the six sealed reports (workbench_comparison "
                 f"{SOURCE_TOOL_SHA256[:8]}). {EXPECTED_ROWS_PER_ROLE} frames per role, {EXPECTED_ROWS_PER_ROLE * 2} rows; "
                 "no pooling, ranking or significance claim.\n"
                 "Runs: " + "   ".join(runs[:3]) + "\n          " + "   ".join(runs[3:]),
                 fontsize=8.6, va="top", ha="left", wrap=False)
    fig.canvas.draw()
    check_layout(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def check_layout(fig: Any) -> None:
    """Every text artist must lie fully inside the figure (no clipped labels)."""
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "get_window_extent")):
        if not text.get_text() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 0.5 or box.y0 < bounds.y0 - 0.5 or box.x1 > bounds.x1 + 0.5 or box.y1 > bounds.y1 + 0.5:
            raise SuiteError(f"graph text is clipped: {text.get_text()[:60]!r}")


# --------------------------------------------------------------------------
# outputs


def _publish(path: Path, data: bytes) -> str:
    """Atomic, never-overwriting publication (hard link of a fully written temporary file)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise SuiteError(f"{path.name} already exists; outputs are immutable") from None
    finally:
        tmp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def build(comparisons: Sequence[Path], out_prefix: Path) -> dict:
    out_prefix = Path(os.path.abspath(out_prefix))
    targets = {ext: out_prefix.with_name(out_prefix.name + ext) for ext in (".json", ".csv", ".png")}
    if any(os.path.lexists(p) for p in targets.values()):
        raise SuiteError("an output with this prefix already exists; outputs are immutable")
    rows = []
    for path in comparisons:
        doc, sha, _raw = read_report(Path(path))
        row = validate_report(doc, source_name=Path(path).name, source_sha256=sha)
        row["source_report"]["path"] = project_relative(str(Path(path)))
        rows.append(row)
    ordered = validate_suite(rows, require_source_seals=True)
    record = aggregate(ordered)
    png = figure_png(ordered)
    csv_data = csv_bytes(ordered)
    record["outputs"] = {"csv_sha256": hashlib.sha256(csv_data).hexdigest(),
                         "png_sha256": hashlib.sha256(png).hexdigest()}
    json_data = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    total = len(json_data) + len(csv_data) + len(png)
    if total > MAX_OUTPUT_BYTES:
        raise SuiteError("outputs exceed their byte bound")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_prefix.parent).free - total < FREE_RESERVE_BYTES:
        raise SuiteError("1 GiB free-disk reserve would be violated")
    # Seal re-check immediately before publishing: every source file is still the
    # sealed original report (externally computed hash, never self-declared).
    for path, row in zip(comparisons, rows):
        sealed = SUITE_BY_CLIP[row["clip_id"]]["report_sha256"]
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sealed or row["source_report"]["sha256"] != sealed:
            raise SuiteError(f"{Path(path).name}: source report changed or is not the sealed original; nothing published")
    hashes = {}
    for ext, data in ((".csv", csv_data), (".png", png), (".json", json_data)):
        hashes[ext] = _publish(targets[ext], data)
    return {"outputs": {str(targets[k]): v for k, v in hashes.items()}, "clips": [r["clip_id"] for r in ordered]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidated six-case paired suite graph (no inference).")
    parser.add_argument("--comparison", action="append", required=True,
                        help="sealed *-full-27e06-vs-baseline-v1.json report (exactly six, repeated)")
    parser.add_argument("--out-prefix", required=True, help="fresh output prefix (.json, .csv, .png are added)")
    args = parser.parse_args(argv)
    try:
        result = build([Path(p) for p in args.comparison], Path(args.out_prefix))
    except (SuiteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
