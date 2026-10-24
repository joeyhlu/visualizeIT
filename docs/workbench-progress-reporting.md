# Workbench progress reporting (saved evidence only)

`tools/workbench_progress.py` is the existing two-hour results generator. It
draws **saved evidence only**: it runs no model and no inference, reads no
ground truth or results beyond the explicit receipts, and downloads nothing.
This revision completes the graph and CSV within the same generator and
interface.

## Inputs (unchanged interface)

```
python tools/workbench_progress.py \
  [--records .cache/cloud-prepare/flow-terminal-small-records-20261021.json] \
  --evaluation-report .cache/video-workbench/evaluation-mug-full-primary.json \
  --evaluation-report ... (one per clip) \
  [--checks SOFTWARE_CHECKS.json] [--output-dir artifacts/video-workbench/progress]
```

- **Records:** the training-records receipt must hash to the pinned
  `RECORDS_SHA256`, and its development comparison must name head `27e06a51…`.
- **Evaluation receipts:** these must be `workbench_evaluation_v1`, with the
  installed head `27e06a51…`, `inference_executed: true` and a `clip_id`.
- **Software checks:** these must be `workbench_software_checks_v1`. Changed
  source files mark the checks as stale.

## Validation (rejected with `ValueError`)

**Metrics**
- `None` means unmeasured and stays `None`.
- Booleans and non-finite values are rejected.
- Fractions must be in 0..1, and px and ms values must be ≥ 0.
- `wrong_surface_events` and `peak_memory_bytes` must be non-negative integers.

**Counts**
- `scheduled_frames`, `clip_scoring_frames`, `result_rows` and `failed_frames`,
  when present, must be non-negative integers and not booleans.
- Every count is bounded by every known denominator, even when another count
  is missing. Scheduled frames, rows and failures can't exceed a known window,
  and rows and failures can't exceed known scheduled frames. For example, a
  window of 10 with 11 rows is rejected even without `scheduled_frames`.
- `full_scoring_window_scheduled` must be a boolean. Claiming the full window
  with unequal counts is rejected.
- Missing counts stay unknown, for backward compatibility.

**Duplicates** are rejected rather than pooled, so they can't inflate
coverage:
- the same `clip_id` and `mode` twice;
- the same `run_id` twice;
- the same receipt file twice.

## Coverage and labels

Each row's `coverage` comes only from the receipt's own counts:

- `complete`: all four of these hold:
  - the window is non-empty;
  - the full window is claimed;
  - the scheduled frames equal the window;
  - every frame has a result row (`result_rows` equals the window; failed
    frames are rows and stay in the denominator).

  A full-window claim with missing or fewer result rows is not complete;
- `partial`: a non-empty window where the scheduled frames, or the known result
  rows, are fewer than the window;
- `unknown`: anything else, including missing counts and an empty (all-zero)
  window. An empty window has a `null` completed fraction.

`completed_window_fraction` is `result_rows / clip_scoring_frames`, or `null`
when either is unknown.

- **Mode:** shown as `smoke` or `regression`. Anything else is shown as
  `scope unverified`.
- **Snapshot coverage summary:** counts distinct clip IDs. Object cases are
  **not** counted as independent videos; `independent_video_count` says it is
  not established.
- **Comparison status:** printed exactly as recorded, with "as recorded; no
  liveness claim". If the receipt has none, it shows "not provided". The old
  unconditional "paired baseline still running" line is removed.
- **Unchanged fields:** `independent_annotations_complete` and `evidence_kind`
  are preserved unchanged.

## PNG layout (one timestamped file)

| Row | Left | Right |
|---|---|---|
| Header | Title, full primary head pin and generated UTC timestamp | |
| 1 | **Development only**: the cached-token EPE deltas (never attachment accuracy) | Evidence status lines (`status_lines`) |
| 2 | Attachment median (720p px) | Attachment P95 (720p px) |
| 3 | All-frame pose availability | Artwork rendered / all frames |
| 4 | Wrong-surface / flip events | Correct recovery fraction |
| 5 | Host per-frame latency P95 (ms) · not phone FPS | Host process peak RSS (MiB) · not GPU memory |
| 6 | Completed frozen scoring-window coverage | How to read |

**Metric panels**
- Ticks are compact: the clip ID only, with the mode added only if the same
  clip ID appears more than once.
- Mode, scheduled/window counts, rows and the complete/partial/unknown status
  appear in the coverage panel (inside each bar), the status lines and the CSV.
- A missing value is a text marker, never a zero bar. `UNVERIFIED (no
  independent annotations)` is used for attachment metrics when annotations are
  incomplete; `UNMEASURED` is used otherwise.
- A panel with no measured values hides its axis and states "ALL UNMEASURED"
  (or "ALL UNVERIFIED") · "no bar means no measurement, not zero".
- With no receipts, every panel says "NO VIDEO RECEIPTS · UNMEASURED".

**Status lines** also say:
- phone inference and performance are UNMEASURED;
- latency is host per-frame timing and memory is host process peak RSS;
- no baseline comparison is drawn from primary-only receipts;
- availability, software checks and parity are not accuracy evidence.

## Scope of the resource metrics

These come from `bench/workbench_evaluate.py` and the CLI frame loop. The JSON
records them as `metric_scope_notes`.

- **`latency_p95_ms`** is the P95 of the shared **host** per-frame block:
  frame decode, cached mask load, tracking, render and overlay write. It
  excludes model initialisation and segmentation preparation, and it is **not**
  phone FPS.
- **`peak_memory_bytes`** is the **host process peak RSS** (resident set size).
  It is **not** GPU VRAM, and no GPU memory claim is made.

## CSV (one table, `row_kind` separates kinds)

**Development rows** (`row_kind = development_cached_token_epe`) fill `id`,
`baseline_epe_px`, `adapted_epe_px` and `delta_epe_px`.

**Video rows** (`row_kind = video_evaluation`) fill:
- `clip_id`, `run_id`, `mode`, `mode_label` and `evidence_kind`;
- `independent_annotations_complete`;
- `coverage_status`, the four counts and `completed_window_fraction`;
- **all 8 metrics**;
- `missing_metrics` (names joined by `;`), `comparison_status` and
  `source_sha256`.

A missing metric is an **empty cell**, never `0`.

## JSON

The JSON keeps every earlier `collect` field. It adds `development_label`,
`metric_scope_notes`, a `coverage` block with `result_rows`, `run_id`,
`backend_role` and `source_sha256` on each video row, and the snapshot-level
`coverage` summary. The schema name is unchanged
(`workbench_progress_snapshot_v1`). The metric display labels in
`TRACKING_METRICS` now read "Host per-frame latency P95" and "Host process peak
RSS"; the metric keys are unchanged.

## Output safety

- `--output-dir` must really be inside the project. This is checked before
  anything is created:
  - every existing path component under the project must be a plain
    directory. A symlink, junction or other reparse alias is rejected, even if
    the final child doesn't exist yet;
  - the resolved path must stay strictly inside the resolved project root.

  The check runs again after `mkdir` and before writing.
- All three files are rendered in memory first. Their total must fit the
  declared 32 MiB allowance, and free disk minus the allowance must remain at
  least 1 GiB.
- Each file is written exclusively (`xb`) with a microsecond UTC stamp.
  Existing snapshots are never overwritten.
- Matplotlib is imported only when rendering.

## Pure helpers (for tests without pixel matching)

| Helper | Purpose |
|---|---|
| `validate_metric(name, value)` | Metric domain checks |
| `validate_counts(report)` | The coverage dictionary |
| `mode_label(mode)`, `coverage_label(row)`, `report_labels(row)` | Labels for mode, coverage, annotations and the exact comparison status |
| `tick_labels(rows)` | Compact metric-panel ticks |
| `metric_cell(row, name)` | `{status: measured|unmeasured|unverified, value, plot_value, display}` |
| `check_unique(rows)` | Duplicate rejection |
| `coverage_summary(rows)` | Snapshot coverage summary |
| `status_lines(snapshot)` | The evidence status lines |
| `csv_rows(snapshot)` / `csv_text(snapshot)` | The CSV content |
| `render_png(snapshot)` | PNG bytes |
| `plot(snapshot, path)` | Exclusive PNG write |

## Limits

- **Attachment accuracy:** the six current receipts have null attachment, wrong-surface
  and recovery metrics. These show as UNVERIFIED or UNMEASURED; nothing is
  inferred from availability or rendering.
- **Phone:** not measured.
- **Clips:** never pooled, and no improvement claim is made.
