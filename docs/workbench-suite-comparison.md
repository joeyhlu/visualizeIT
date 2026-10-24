# Workbench six-case paired suite graph

`tools/workbench_suite_comparison.py` turns the six sealed full-video paired
comparison reports into one consolidated JSON, CSV and PNG. Those reports are
`artifacts/video-workbench/comparisons/*-full-27e06-vs-baseline-v1.json`,
produced by `tools/workbench_comparison.py` (`223350da…`).

It performs no inference and recomputes nothing about the models. The
numbers come straight from the reports, after strict validation.

## Command

```powershell
python tools/workbench_suite_comparison.py `
  --comparison artifacts/video-workbench/comparisons/mug-full-27e06-vs-baseline-v1.json `
  --comparison artifacts/video-workbench/comparisons/keyboard-full-27e06-vs-baseline-v1.json `
  --comparison artifacts/video-workbench/comparisons/ranch-full-27e06-vs-baseline-v1.json `
  --comparison artifacts/video-workbench/comparisons/ycb-cracker-box-full-27e06-vs-baseline-v1.json `
  --comparison artifacts/video-workbench/comparisons/ycb-soup-can-full-27e06-vs-baseline-v1.json `
  --comparison artifacts/video-workbench/comparisons/ycb-marker-full-27e06-vs-baseline-v1.json `
  --out-prefix artifacts/video-workbench/comparisons/suite-full-27e06-vs-baseline-v1
```

This writes `<prefix>.json`, `<prefix>.csv` and `<prefix>.png` (1950×1350,
about 250 kB). Each file is published atomically through a hard link, and an
existing output is never overwritten.

## Validation (fails closed)

Each report must be a full-scope `workbench_paired_comparison_v1` from the
pinned comparison tool and must have the exact sealed file name. It must also
satisfy all of the following:

- **Model pins:** main head `27e06a51…` and frozen base `f7d127ab…`. The
  installed head must be verified on every row. The baseline must be the frozen
  base pose head with no installed head.
- **Run:** exactly one complete `regression` run, with
  `verified_primary_vs_frozen_baseline` pairing.
- **Roles:** exactly the two paired roles.
- **Window:** the frozen window of 240 frames for mug/keyboard/ranch and 110 for
  YCB.
- **Clip:** `clip.json` SHA-256 equal to the frozen suite clip.
- **Rows:** no missing rows and a clean parse.
- **Accounting:** pose, render and mask counts account for every scheduled
  frame; fractions equal their counts; paired-frame counts match both roles.
- **Latency:** measured in that run (no cached timings).
- **Accuracy:** must stay `unverified_no_independent_annotations` with null 720p
  attachment values. All report claims must be false.

For the suite as a whole: exactly six reports, one per frozen clip, with no
duplicate clip, run or report. That gives 1050 frames per role and 2100 rows,
placed in the fixed order mug, keyboard, ranch, YCB cracker box, YCB soup can,
YCB marker.

## Original-run seals

Each suite case pins its **original run ID** and the **exact SHA-256 of its
source comparison report**:

| Case | Original run | Report SHA-256 |
|---|---|---|
| Mug | `run-20261021T002156-1b2de77d` | `9559bec7…` |
| Keyboard | `run-20261021T013943-538cfa03` | `17ace1ca…` |
| Ranch | `run-20261021T020454-01d11a29` | `a315056b…` |
| YCB cracker box | `run-20261021T030702-475adc39` | `184d0904…` |
| YCB soup can | `run-20261021T031729-0986ef46` | `efd867d7…` |
| YCB marker | `run-20261021T032932-539343a7` | `37d8844a…` |

- `validate_report` always enforces the expected run ID. When given the
  actual report SHA-256, it also requires that SHA to equal the pin.
- `validate_suite` requires the pinned SHA-256 by default
  (`require_source_seals=True`).
- `build` and the CLI always compute each report's SHA-256 from its bytes, never
  from a self-declared name. Immediately before publishing, they re-hash every
  source file against its seal.

So a later trial cannot be substituted for an original, even when it has the
same clip, models, scope and file name or a rewritten run ID. A re-exported
report is refused too. There is no override, force option or automatic seal
update. A future completed export or trial needs a separately reviewed source
version of this tool. The latest report is never selected silently.

The seals are written into the aggregate JSON (`suite`, `source_seals`) and
into the CSV (`sealed_run_id`, `sealed_report_sha256`).

The JSON keeps each clip's full IDs, run ID, the source report's
project-relative path and SHA-256, and the run's input-file hashes. The
reports do not record admission identity; it is bound only indirectly through
the `run.json` SHA-256, and the JSON says so. The PNG contains no private
paths.

The pure helpers `validate_report`, `validate_role`, `validate_suite`,
`aggregate` and `csv_bytes` take parsed documents, so mutated authentic
reports can be unit-tested directly. Unit tests may omit the actual SHA-256 by
passing `source_sha256=None` and `require_source_seals=False`. The run-ID seal
is always enforced.

## Reading the graph

- **Panels:** pose availability and raw artwork-render fraction per clip (new
  head vs frozen baseline), plus end-to-end latency per frame (bar = median,
  tick = P95).
- **Not accuracy:** availability and render fraction are **not** accuracy.
  Independent attachment accuracy is **UNVERIFIED** because there are no
  independent image landmarks.
- **Latency:** the roles ran serially on a shared host, so latency is not a
  causal performance comparison.
- **Unavailable or unmeasured:** 720p attachment median/P95 are unavailable,
  and phone performance is unmeasured.
- **No pooling:** clips are never pooled, and no statistical significance or
  quality improvement is claimed.
