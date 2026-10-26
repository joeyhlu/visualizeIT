# Reproducible bounded native recorded runs

`tools/tracking_native_recorded.py` supervises the **existing** workbench CLI.
It does not replace SAM, the integer-metric refiner, temporal tracking, or any
admission/quality gate. It cannot train or run LIVE. Prepare the pinned isolated
CPU runtime using [GoTrack](tracking-neural-diagnostic.md) and
[SAM2](tracking-sam-cpu-runtime.md); use the explicit
[fresh recorded-artifact lineage](tracking-recorded-artifact.md) if the exact
historical receipt bundle is unavailable. No receipt/configuration is fabricated.

Copy this tool and `requirements-tracking-sam-cpu.txt` into the corresponding
`tools/` and `bench/runtime/` directories of the already-restored isolated
worktree. Run from that worktree with its CPU `.venv`. All normal model/source,
geometry, parity, clip, metric CAD, mask, and seed checks still apply. Local
head/best paths are supplied by the operator; no private weight is downloaded or
published by this runner.

## Invocation and evidence

```bash
.venv/bin/python tools/tracking_native_recorded.py --output NEW_EVIDENCE_DIR \
  -- --workspace WORKSPACE gate verify --admission ADMISSION \
  --recorded-artifact LOCAL-PROOF.json
```

Only native `gate`, `segment`, and finite recorded `run` commands are accepted.
Non-CPU devices are refused before model construction.
`NEW_EVIDENCE_DIR` must not already exist. The tool creates:

- `input-freeze.json`: command, source/runner SHA-256, exact resource/thread budget;
- `runtime.json`: observed Torch/OpenCV/thread/platform settings;
- `worker.log`: real CLI output and genuine blocking/failure reasons;
- `observed-full-forward-counts.json`: actual backbone/decoder/head calls per
  loaded GoTrack network; no tensors are captured or altered;
- `supervision.json`: child exit, guard/failure, elapsed wall, sampled peak RSS,
  source drift, and declared limits. Nonzero exit, guard, or source drift fails
  the supervisor even if a native file was partially produced.

The first network corresponds to the first *actually loaded* role, not a role
name invented by the counter. Verify run order, cache state and native iteration
counts before mapping counters to roles. A cache hit is not a model forward.
Native receipts/FrameResults remain authoritative; a forward count or successful
invocation alone is not model accuracy.

Torch uses 8 threads/interop 1, OpenBLAS 1, OpenCV 1, FP32, seed0, no CUDA, software
EGL and no SAM CUDA extension. A security-only scope forces
`torch.load(weights_only=True, map_location='cpu')` **only inside**
`wn.load_base_network`; the native export/reference BytesIO loads remain
untouched and safely deserialized by their own checked code. No unsafe fallback,
extra safe globals, protected loader edit, or model numerical change is allowed.

The supervisor samples aggregate child-tree RSS every 250 ms, caps it at 7 GiB,
preserves 10 GiB free disk and caps **new output growth** at 512 MiB. It automatically
watches the native command's output/mask/derived/scratch/run paths; extra output
paths can be supplied with `--watch-output PATH`. Pre-existing datasets/weights
are not counted as new output. RSS is sampled rather than a continuous peak;
retain native `ru_maxrss`/resource telemetry as a separate measurement. Run one
heavy model/tracker benchmark stage at a time; this helper is not a multi-job
scheduler. SIGTERM/thenSIGKILL bounds children and retains partial evidence.

### Explicit measured wall profiles

Default `--budget-profile pilot` caps every stage at900s. The separately approved
`--budget-profile recorded300` is bounded at:

| Native stage | Maximum wall time |
|---|---:|
| `segment` (301 inputs =1 onboarding +300 scored) |1800s|
| `run` (one 300-frame pose role) |2700s|
| Any `gate` |900s|

These larger finite bounds follow the measured pilot (~4s/SAM frame,
~6–6.5s/five-iteration pose frame); they do not change model/solver/temporal
thresholds, RSS/output/disk limits, or admission. `--max-seconds` may lower the
selected ceiling but cannot raise it. The unchanged native contract forbids
baseline-only runs: baseline is a comparator and must accompany primary. For a
full comparison select `--budget-profile recorded300pair` with explicit ordered
`--roles primary_neural,frozen_baseline`. This runs600 rows internally serially
in one normal native job, capped at5400s (90min), with measured expected active
time about65min. The pair profile refuses missing/reversed/other role sets;
its geometry/parity gates remain900s. Primary-only work can use2700s, but must
not be misrepresented as a comparison. Keep all failures/partial outputs; never
silently retry with different settings or relax the native role contract.

## Example: actual SAM followed by causal metric pose

Use a registered native clip whose only prompt/pose seed is in onboarding.
Inputs after onboarding must exclude scored poses, depth and annotation masks.
Native RGB/calibration must be preserved; lossless re-encoding is permissible
only with decoded-pixel equality verified. Do not invent physical FPS for a
frame-index dataset.

```bash
# One observed SAM mask per input, causally released at native resolution.
.venv/bin/python tools/tracking_native_recorded.py --output EVIDENCE/SAM \
  --budget-profile recorded300 -- --workspace WORKSPACE segment CLIP-ID \
  --mask-set NEW-SAM-MASK-SET --mode regression --device cpu --verify-hash

# Freeze the actual SAM onboarding output, not a scored annotation/silhouette.
.venv/bin/python -m bench.workbench_cli --workspace WORKSPACE gate case CLIP-ID \
  --frame ONBOARDING-FRAME --mask-set NEW-SAM-MASK-SET \
  --metric-bundle METRIC-BUNDLE.json --out ACTUAL-SAM-CASE.json
```

Earn normal contract → production geometry → full-image reference/export parity
→ admission verification with explicit `--recorded-artifact LOCAL-PROOF.json`
where required; wrap each native command as in the first example. Geometry is
artifact-independent and has no recorded-artifact flag. Reusing existing
admission is allowed only where the normal verifier accepts its exact source,
model, calibration-family and case evidence; the runner grants no exception.

After all heavy jobs are serially cleared, run the normal combined comparison:

```bash
.venv/bin/python tools/tracking_native_recorded.py --output EVIDENCE/combined \
  --budget-profile recorded300pair -- --workspace WORKSPACE run CLIP-ID \
  --mode regression --roles primary_neural,frozen_baseline --masks NEW-SAM-MASK-SET \
  --metric-bundle METRIC-BUNDLE.json --head LOCAL-HEAD.pt --device cpu \
  --admission ADMISSION --recorded-artifact LOCAL-PROOF.json
```

Both roles execute sequentially inside the unchanged native pipeline. Baseline
is a comparator and cannot replace primary: do not fake a primary result or
relax RunSpec to split this into baseline-only work. One combined prediction
seal covers both roles before comparative scoring. The explicit expanded
development/reused-reference seal records prior GT access and run-local
inference isolation; it must never claim globally unopened reference labels.

The first pose seed is single-use and applies only to the immediately following
frame. Later frames use the previous checked candidate; public tracking requires
two consistent solutions. Temporal-only recovering retains that candidate.
After hard solver/mask loss there is **no automatic reacquisition/reseeding**:
the native loop continues to count every frame but skips further NN work when no
prior/authorized seed is available. A much shorter failed run is not faster
successful tracking. Native pose/overlay limits remain distinct; unavailable
artwork must not be called rendered.

Seal the complete combined prediction set, source/input hashes and schedules **before** opening
scored GT. Score missing/recovering/lost frames in all-frame denominators, and
report conditional errors separately. Native `end_to_end_latency_ms` excludes
precomputed SAM and model initialization: it is a pose-stage latency, not a
measured integrated SAM+pose streaming latency. One-object training-overlap or
reused pilot-prefix data are functional/development evidence, not held-out-object
generalization, phone performance, or a statistically established model win.
Keep raw receipts (which may contain local/private weight paths), weights and
unreviewed outputs local; publish only authorized, sanitized summaries.

## Lightweight tests

```bash
.venv/bin/python -m unittest tools.test_tracking_native_recorded -q
```

Twenty tests exercise base-only safe-load scope/restoration/no fallback,
observational/deepcopied-head hook cleanup without retaining prior models,
atomic-temp-file accounting, explicit stage wall budgets, resource controls, real tiny
child success/failure/timeout/RSS/output aborts, source drift, orphan-process cleanup and immutable output
directories. No test loads weights, constructs a model, renders, or uses CUDA.
The eight real-process tests require the pinned `psutil`; they explicitly skip in
the geometric-only environment without it. This public refactoring is distinct
from the original private pilot runner: recorded measurements retain that exact
runner's SHA-256, rather than being relabeled as executed by this new file.
