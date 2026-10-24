# Workbench cache/timing provenance (recorded-run strip)

The recorded-run provenance strip under **4. Inspect frames** now also shows
where a run's timings come from. These lines read **only** the loaded report's
`detail.report.telemetry` (from the existing `GET /api/runs/{id}`).

- Mask, prompt and config lines still read only `RunSpec.config`.
- Nothing is inferred from the run ID, mask names or the create-run form.
- No extra request is made.
- The lines are cleared with the rest of the strip on clip open, run open (also
  on failure), and run start.

## Lines

| Line | Source field | Shown |
|---|---|---|
| Recorded cache mode | `run_cache_mode` | `cached`, `cold` or `partially_cached`; anything else is `unknown (not recorded or unrecognised)` |
| Recorded pipeline wall (excludes outer command validation) | `run_wall_ms` | seconds, when finite and ≥ 0; otherwise `not recorded` |
| Recorded GPU use | `gpu_used` | `yes` or `no` for a boolean; otherwise `not recorded` |
| Per-frame timings | `cache.{role}.status`, `cached_original_timings.{role}.source`, `frames[].role` | see below |

The per-frame rule is applied to each scheduled role (`RunSpec.roles`; if that
is missing, the roles present in the telemetry):

- **Historical:** `cache.{role}.status == "cache_hit"` **and**
  `cached_original_timings.{role}.source == "cached_original_run"` →
  "historical from cached original run (…); not new inference".
- **This run:** not a cache hit, a cache entry is present, **and** `frames`
  contains rows for that role → "recorded frame timings from this run (…)".
- **Unknown:** anything else.

A mixed run lists each group separately.

If there is no telemetry object, the strip shows `Timing provenance: not
recorded (no report telemetry)`.

## Limits

- The wall time is the pipeline's own recorded wall time. It excludes the outer
  replay command's validation, and it is not end-to-end or inference latency.
- Per-frame timings of cached roles are the **original run's** measurements.
  They are not new inference, and they are not a phone, fps or speed-up
  measurement.
- No statistics are derived; values are shown as recorded. Attachment accuracy
  stays unverified.
- All text is set with `textContent`; no HTML or CSS changed.

## Expected display (parent browser check)

- **Cached replays** (`run-20261021T071807-7f091375`,
  `run-20261021T072130-746e1fcc`): cache mode `cached`, GPU use `no`, and
  per-frame timings "historical from cached original run (primary_neural,
  frozen_baseline); not new inference".
- **Original cold runs** (`run-20261021T013943-538cfa03`,
  `run-20261021T060541-e8894b79`): cache mode `cold`, and per-frame timings
  "recorded frame timings from this run (primary_neural, frozen_baseline)".
