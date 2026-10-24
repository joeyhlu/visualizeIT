# Cache-only regression replay (seven fixed paired runs)

`tools/workbench_cache_regression.py` replays the seven completed paired
full-window runs **through the unchanged pipeline**, using only the existing
stage cache. It measures actual cache reuse; it does not re-read sealed reports.
It makes no model-quality, pooling, head-superiority or mobile claim. The
original and qualified cracker runs are separate cases, and nothing is promoted
into a default suite.

| Case | Original run | Frames | Admission | Metric bundle |
|---|---|---|---|---|
| `mug` | `run-20261021T002156-1b2de77d` | 240 | attempt-03 | `admission/bundle-mug-v1.json` |
| `keyboard` | `run-20261021T013943-538cfa03` | 240 | attempt-04 | `admission/bundle-keyboard-v1.json` |
| `ranch` | `run-20261021T020454-01d11a29` | 240 | attempt-05 | `admission/bundle-ranch-v1.json` |
| `cracker-original` | `run-20261021T030702-475adc39` | 110 | attempt-06 | `execution-controls/v1/object-000002` |
| `soup` | `run-20261021T031729-0986ef46` | 110 | attempt-06 | `execution-controls/v1/object-000004` |
| `marker` | `run-20261021T032932-539343a7` | 110 | attempt-06 | `execution-controls/v1/object-000018` |
| `cracker-qualified` | `run-20261021T060541-e8894b79` | 110 | attempt-06 | `execution-controls/v1/object-000002` |

Head `27e06a51…` and base `f7d127ab…` are fixed, and both are re-hashed before
every case.

## Commands (the parent executes them)

```powershell
.cache\quality-windows\Scripts\python.exe tools\workbench_cache_regression.py plan --out <fresh-plan.json>
.cache\quality-windows\Scripts\python.exe tools\workbench_cache_regression.py run `
  --plan <fresh-plan.json> --expected-plan-sha256 <sha> --out <fresh-summary.json> [--case keyboard]
```

## Plan (read-only, no torch)

For each case, the plan binds:

- the original `run.json`, `results.jsonl`, `report.json` and
  `neural-artwork.json` (SHA and size);
- the whole original run folder (aggregate identity);
- the overlay evidence hashes;
- the RunSpec;
- the clip record and the original config's mask-set manifest;
- the metric bundle and GLB, which must match the recorded backend
  `glb_sha256`;
- the admission attempt closure, whose contract SHA must equal both roles'
  recorded `admission_contract_sha256`.

It also records the head and base checkpoint, current module hashes, runtime
versions and the tool SHA.

It **rejects** an original with the wrong roles, mode or frame count, duplicate
or missing rows, an incomplete report, or a mismatched admission or GLB.

It marks a case ineligible for either kind of drift:

- **Source drift:** any module recorded in the original report differs from
  current code.
- **Runtime drift:** `report.dependencies.versions` differs from the current
  `workbench_telemetry.dependency_versions()` in any way. This covers Python,
  platform, numpy, OpenCV, torch, pyrender, trimesh and Pillow. Missing fields
  count as drift, and nothing is normalised. Run the plan with the same
  interpreter used for execution (`.cache/quality-windows`).

It reads no reference, evaluator or raw ground-truth file.

### Known gap: `mug` is ineligible under current code

The mug report records `workbench_cli f7312a7d`, `workbench_neural 939fffdf`
and `workbench_reconstruction e34d8807`. Current code is `edf76881`, `86d0caf6`
and `9909d059`. These modules are hashed into the neural stage key
(`workbench_cli._neural`), so the current key cannot equal mug's cached key and
a cache hit is impossible. The plan records this as `source_drift` instead of
inventing pins. A mug replay would need the original sources, which this tool
does not provide.

The six other originals show no recorded drift.

`workbench_render` and `quality_render_stability` are in the key but are not
recorded by the original reports. Their equality is established only by an
actual cache hit; this is listed per case.

## Run (guarded, serial, never enters the GPU stage)

**Before any case (repair 1):** `validate_destination` checks the summary
path:

- it is a visible `.json` file that doesn't exist yet;
- it lies inside the repository, and its parent is an existing directory
  that resolves to itself;
- there is no symlink or Windows reparse point (junction) on the path from the
  repository root;
- the parent is writable, which is checked with a fresh owned temp file that is
  deleted straight away; no existing data is touched;
- the workspace budget (1 GiB derived limit and 1 GiB reserve) and the parent
  disk can fit the 256 KiB summary plus the reserve for every requested eligible
  case.

An invalid destination means **no** case runs.

**Per case:**

1. The complete shared identity (weights, current module hashes, runtime, tool)
   and the case identity are recomputed and must equal the plan
   (`closure_status`).
2. The budget must fit the original run size + 10% + 1 MiB plus the summary
   cap.
3. The original RunSpec is reused with only a new `run_id`.
4. `load_admission` must re-verify the recorded contract.

**Execution:** it calls the real `workbench_cli.execute_run` with the original
mask set, head, admission, metric bundle and `device="cuda"`, which is needed
for the exact key. The pipeline recomputes the key and re-verifies all inputs
itself. No `neural_backends` are injected, so the cache is not disabled.

**Guards, restored on every path:** these raise `CacheOnlyViolation`, a
`BaseException`, so core `except Exception` handlers cannot swallow it:

- `workbench_telemetry.gpu_stage` (a cache miss reaches this);
- `workbench_neural.GoTrackMetricBackend`, `NeuralSession` and
  `segment_prompted`;
- `Pipeline._cache_store` and `StageCache.store`.

A miss stops the case as `failed: cache miss`. It never produces unavailable
rows that count as success.

**Checks per completed case:**

- both role stages are `cache_hit`, and the report is `completed`;
- telemetry has zero new frames, no role initialisation, `gpu_used: false`, and
  `cached_original_timings` for both roles;
- CUDA is not initialised;
- every `FrameResult` is **exactly equal** to the original with only `run_id`
  removed, and frame accounting is complete;
- overlay evidence bytes and `neural-artwork.json` equal the originals. The
  relative evidence paths are identical, so no path normalisation is used;
- the original closure is unchanged afterwards.

**Timings:** the original per-frame timings are reported as **cached historical
timings** (not inference). The replay's own wall time is measured separately.

**Closure after every path (repair 1):** the shared and case closures are
checked again on every execution path: success, guard violation, ordinary error,
budget block or admission failure. A case becomes `completed` only after this
check passes. A closure failure is **appended** to any earlier reason (for
example the cache-miss guard reason), never replacing it, and guard attempts are
kept.

**Plan file:** the externally pinned plan file is re-hashed before each case
and after the last one, before the summary is written. If it changed, later
cases are `not_run` and any `completed` case becomes `failed`.

**Exit code:** `run` exits 0 only when every requested case is `completed` and
the plan file is unchanged. Any `failed`, `blocked`, `ineligible` or `not_run`
case gives exit 1, and errors give 2. `plan` exits 0 even when cases are
ineligible; that is data, not an error.

**Output:** the summary (≤ 256 KiB, immutable, atomic) records every requested
case with its status, reason and both closure results. If it cannot be written,
the full record goes to stderr. The new run folders are evidence and are never
deleted.

## Limitations

- An interruption leaves partial run folders as evidence. The command attempts
  a partial summary and marks remaining cases `not_run`; a hard process termination
  can prevent that summary from being saved.
- A hit proves that the key matched the cached entry; it does not re-measure
  inference.
- The reserve remains 1 GiB. A case blocks when its planned output and summary
  cannot fit above that reserve.

## API (for parent tests)

| Function | Purpose |
|---|---|
| `row_semantics(row)` / `compare_rows(old, new)` | Exact per-(role, frame) equality with only `run_id` removed. |
| `compare_versions(original, current)` | Exact runtime drift; missing fields count as drift. |
| `closure_status(plan, key)` | Full shared and case re-check; returns `{ok, shared_equal, case_equal, *_differs, error}`. |
| `plan_file_unchanged(path, sha)` | Re-hash of the pinned plan file. |
| `validate_destination(out, *, reserve_bytes)` | The pre-run destination and budget checks. |
| `exit_code(result)` | The exit-code rule above. |
| `cache_only_guards()` | The six guards; restored on every path. |
| `load_plan`, `build_plan`, `shared_identity`, `case_identity`, `verify_replay`, `run_case`, `run`, `main` | Plan, execution and CLI. |

## Test suggestions (parent)

**Pure helpers:**

- `compare_rows` and `row_semantics`: a single field change, duplicates,
  missing and extra rows, and a `run_id`-only difference (which must pass).
- `load_plan`: a tampered SHA, non-canonical bytes, and a wrong case set.
- `case_identity` on a copied fake run: wrong roles, a missing row, an
  admission contract mismatch, and a drifted recorded source.
- `cache_only_guards`: every patch is restored after a `CacheOnlyViolation`
  and after a normal exit, and the violation is not caught by
  `except Exception`.

**With mocks:** run `run_case` with `execute_run` patched to call the guarded
`gpu_stage` (a miss must give `failed: cache miss`) and with a fake report
lacking `cache_hit` (verification must fail).

**Real execution:** run one real case first (for example `--case keyboard`),
then the others.
