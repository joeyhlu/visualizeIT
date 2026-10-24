# Workbench paired comparison report

`tools/workbench_comparison.py` builds an **offline** report from existing workbench run
directories. It runs no inference, uses no GPU and installs nothing. It reads runs only through
the existing APIs: `Workspace.load_clip`, `workbench_cli.load_results`, `workbench_contract`
and `workbench_evaluate.evaluate`.

```
python tools/workbench_comparison.py --workspace .cache/video-workbench \
    --run RUN_ID [--run RUN_ID ...] --scope {smoke,full} --out-prefix PATH [--png]
```

- **Output files:** `PATH.json` (immutable report) and `PATH.md`. `--png` adds `PATH.png`;
  matplotlib is imported only when this flag is given. If any of these files already exists,
  the tool refuses before writing anything. Each file is written to a temporary file and then
  hard-linked into place, so nothing is ever overwritten.
- **Scope:** must be declared explicitly. `full` requires a `regression` run whose scheduled
  frames are exactly the clip's full eligible scoring window. `smoke` requires the
  10-frame smoke schedule. Any other run is rejected.
- **Validation:** every row's run, clip, role, scheduled frame and source frame index is
  checked, and the (role, frame) pair must be unique.
- **Identity:**
  - Rows that claim a primary tracking pose are accepted only if every primary row reports the
    actually installed head `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`.
    Otherwise they are rejected as spoofed or missing identity.
  - A baseline row that claims the new head is rejected.
  - Both neural roles must report the pinned base checkpoint
    `f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086` on every row. A role
    whose rows claim a pose with a wrong, mixed or missing base is rejected. The baseline is
    verified only when its rows show the pinned base and `frozen_base_pose_head`. The verified
    base hash is reported for both roles. Blocked or all-unavailable rows that claim no pose
    are kept but stay unverified, and run metadata never fills in a missing actual identity.
  - The head *requested* in the run metadata is reported separately from the installed head.
  - The run is labelled `verified_primary_vs_frozen_baseline` only when the evaluator also
    confirms a completed or verified-cache stage and both roles' identities are verified.
- **Failed or partial runs:** a missing `report.json` or an unparseable trailing line is
  counted, not hidden. Every missing scheduled row counts as unavailable and stays in the
  denominator.
- **Per role and run:** reported separately, never pooled or ranked:
  - pose availability (tracking ÷ scheduled) and the raw artwork render fraction
    (rendered ÷ scheduled);
  - recovering, lost and unavailable-including-missing counts;
  - per-frame paired agreement;
  - linear-interpolated median and P95 of the measured per-frame `end_to_end_latency_ms`,
    `tracking_ms` and `rendering_ms`. A cache hit is labelled `cached_original_run`, and the
    stored cached timings are shown separately;
  - the telemetry summary from `report.json`, shown next to these values but not merged with
    them.
- **Run telemetry:** wall time, peak RSS and GPU peaks, as observed. Both roles share these
  values, because the roles run serially on a shared host. Latency differences are therefore
  descriptive, not causal.
- **Accuracy:** attachment error, wrong-surface events and recovery correctness remain `null`
  and unverified without independent annotations. Nothing is derived from availability.
- **Provenance:** every input file read (`run.json`, `clip.json`, `results.jsonl`,
  `report.json`) is listed with its size and SHA-256. The report also records the tool's own
  SHA-256.

## Read-only check on the full mug run (2026-10-21)

Run `run-20261021T002156-1b2de77d`, clip `dev-mug`, full scope, 240 frames per role:

| Role | Pose available | Rendered | Recovering | Lost | Missing | E2E median / P95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| primary_neural (verified 27e06, pinned base) | 225/240 | 187/240 | 15 | 0 | 0 | 5291.4 / 7549.7 |
| frozen_baseline (verified pinned base) | 224/240 | 184/240 | 16 | 0 | 0 | 4276.2 / 6642.7 |

- **Paired pose availability per frame:** both 220, primary only 5, baseline only 4,
  neither 11.
- **Telemetry difference:** the medians equal the stored telemetry summary. The stored P95
  values (7546.7 and 6635.7) use the telemetry's own method, which is why they differ slightly.
- **Accuracy:** unverified.
