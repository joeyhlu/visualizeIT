# Larger GoTrack head adaptation

## October 20 cached-data checkpoint

A separate pilot completed **285 optimizer updates over three epochs**, using
95 available TRAIN examples from four objects and 20 available development VAL
examples from one object. This does not complete the later 312-example,
13-object target. The historical preparation records below describe earlier runs.

The exported head and resumable checkpoints are saved in the private Kaggle
dataset `achitaan/visualizeit-existing-head-20261020-520ba2eb`, version 1.
Mean validation correspondence error changed from 21.2094 to 21.0884 crop pixels;
visibility recall at the existing confidence cutoff fell from 95.41% to 93.73%.
These development results do not justify promoting the head over the baseline.

### Check the export before a pose replay

`bench.quality_adaptation_export_parity` installs the export through the existing
atomic inference loader and compares its outputs with the selected training
checkpoint. Both heads receive the same cached decoder tokens. The command reads
only available VAL feature archives; supervision and TEST payloads stay unopened.
It loads the DPT head alone, without SAM, backbone or decoder weights.

Use a reviewed cloud source root whose files satisfy the original training lock
and whose inference installer matches the new artifact manifest. Restore the
existing pinned runtime, source, dataset metadata, selected feature archives,
`run/best.pt`, and `run/head.pt` in cloud storage first. This command does not
download assets or provision resources. Keep the historical source roots intact.

From that source root, using its pinned Python runtime:

```bash
export VISUALIZEIT_CLOUD=1
# Set these paths to existing cloud files. Set DEADLINE_UTC to a newly approved
# absolute UTC deadline; the October 20 17:15 UTC training window has expired.
timeout --signal=TERM --kill-after=10s 900s "$PYTHON" -B \
  -m bench.quality_adaptation_export_parity \
  --artifact-manifest bench/runtime/gotrack-existing-head-20261020.json \
  --artifact-manifest-sha256 69f2ad7e8619881a7023ea2a9543fff30f7df2ece3e764313bf9e80e18a1c190 \
  --export "$PRIVATE_PACKAGE/run/head.pt" \
  --reference-checkpoint "$PRIVATE_PACKAGE/run/best.pt" \
  --dataset "$FEATURE_ROOT/manifest.json" \
  --output "$NEW_OUTPUT" --deadline-utc "$DEADLINE_UTC" \
  --max-seconds 900 --max-rows 3 --device cpu --cloud-execution
```

The initial feature root was
`/tmp/visualizeit-expanded-features-v1/group1-continuation-v1`; a stopped Kaggle
session may have lost it. Restore verified cloud files rather than assuming the
old session's paths still exist. `NEW_OUTPUT` must not already exist. Inspect its
`receipt.json` and the process exit code; an exit code of zero means the checked
flow and confidence outputs agree within absolute tolerance `1e-6`, with zero
relative tolerance. The command records input identities and retains a failure
receipt when possible. It defaults to CPU; `--device cuda` requires explicit CUDA
availability and uses the heads sequentially.

Passing this check establishes cached-head export consistency. It does not run
the complete image network, compare pose accuracy, prove checkpoint resumption,
or establish real-time speed. A renewed cloud execution window and an actual run
are still required. The new artifact is not admitted to the historical pose
runner profiles: its RAW geometry and serving inputs need a separate verified
integration before a baseline/adapted recorded-video comparison.

This experiment expands the previous 16 training crops from one object to a fixed
plan of 240 training frames across 10 HOT3D meshes. Three different objects supply
24 validation frames and 72 refinement states. These are synthetic renders, not
new real-world recordings. Test object 7 stays unopened. Keyboard, mug, bottle,
and other previously used object identities are excluded from the new split.

Only the released GoTrack correspondence head is optimized. Its backbone and
decoder, SAM 2.1, CNOS and FoundPose stay frozen. The released head initializes
this trial; earlier experimental heads remain separate.

## Actual preparation results

The free cloud CPU completed the first object's 798-template FoundPose bank in
533.49 seconds. Two fixed views then completed rendering, a 57-template CNOS
descriptor bank, observed CNOS/SAM masks, estimated FoundPose initialization and
ordinary GoTrack refinement-crop capture in 273.70 seconds. Both requested crops
were available. The largest sampled process-tree RSS was 3.49 GB.

Visual inspection found one full mask and one partial mask. Mask availability
does not establish segmentation correctness. Both predeclared inputs remain in
the experiment; reference poses and rendered silhouettes are never substituted
for observed masks. Labels are attached only after exact crop/geometry checks and
strict frozen-prediction parity checks pass.

The first bank's timing projects to about 116 minutes for 13 banks alone. Masks,
capture, feature extraction and training add work, so the complete CPU trial
cannot fit its two-hour limit. Its original clock, inputs, masks, crops, bank,
failure inventory and logs were preserved in cloud storage (235,799,933 bytes).
**No optimizer updates have run on the larger dataset yet.**

## Execution interfaces

The implementation lives in `bench/quality_adaptation_larger.py` and
`bench/quality_adaptation_larger_prepare.py`. Both default to CPU and accept an
explicit `--device cuda`. Missing CUDA raises an error. Each expensive stage runs
sequentially, with process-tree memory and time limits and explicit cleanup.

The preparation interface provides `freeze`, `render`, `bank`, `cnos-bank`,
`masks` and `assemble`. The controller provides `validate`, `capture`, `extract`,
`seal`, `labels`, `train`, `resume` and `export`. Use `--help` on each subcommand
for its pinned input arguments. The historical extractor and trainer run from
the authenticated original cloud checkout.

The prepared cloud operator is
`.cache/cloud-prepare/larger_cloud_cuda_execute_v2.py`. It requires an explicitly
approved, hashed authorization file and does not provision a GPU itself:

```sh
VISUALIZEIT_CLOUD=1 python -B larger_cloud_cuda_execute_v2.py \
  --authorization /path/to/approved-authorization.json \
  --authorization-sha256 <actual-sha256>
```

This particular operator is a pinned experiment artifact. Do not invent an
authorization file or reuse its allocation timestamp for a later resource run.
The approved T4 budget was at most 1.10 existing credits for two allocated hours,
at the account's observed rate of 0.55 credits/hour, subject to a fresh quote and
human approval. Work stops by 6,600 seconds from allocation, reserving ten minutes
for provider shutdown. Python exiting does not prove billing has stopped; the
organizer must verify the studio is Sleeping.

The CUDA variant uses its own immutable clock and preserves the CPU clock. It
uses one thread, deterministic algorithms, disabled TF32 and AMP disabled. An
actual first-bank/two-input timing forecast must admit the remaining fixed work.
There are no automatic retries, object replacements or iteration substitutions.

## Training, monitoring and resume

After the full input inventory is captured, frozen decoder features are cached
once. The seal compares actual frozen predictions with capture predictions
before labels are opened (flow tolerance 1e-4; confidence tolerance 1e-6).
Training runs two epochs, batch size one, learning rate 1e-5, with a ceiling of
480 optimizer updates. An intentional stop after 40 updates is followed by resume
from the saved optimizer, RNG and data-order state. Validation selects the export.
The operator then loads that export into a fresh frozen GoTrack network, verifies
exact agreement with the selected checkpoint's weights, and compares predictions
against the cached-feature head on every available validation row. All 72 planned
validation states remain in its report. It requires the existing flow/confidence
tolerances and reads no reference-pose or label arrays during this check. The
operator cannot report completion until this verification passes.

Cloud stage logs and supervision receipts report timing, sampled peak RSS,
availability and cleanup. The operator report records every planned state,
including failures, and the training `status.json`, `history.json` and checkpoint
hashes. Large datasets, frozen features and checkpoints remain in cloud storage.
Only small reports and previews are copied to the laptop.

If ephemeral storage is lost, resume requires regenerating the exact authenticated
data/features or restoring their pinned bytes. The workflow refuses a changed
dataset or configuration; it does not promise seamless cross-session recovery.

## Interpretation

Local focused checks passed 213 tests without importing Torch. The installed cloud
source also passed all 213 tests in 13.14 seconds; 39 operator tests passed locally
and independently, including ten export-verification cases. Earlier missing-fixture and temporary-directory failures were
preserved, then fixed without changing model mathematics or test assertions.
The reviewed v2 operator was uploaded, authenticated and executed on a Tesla T4
on 2026-10-16 UTC. Its ten probe stages completed: the first bank contained 798
templates, and both requested probe masks/crops were available. Bank preparation
took 378.80 seconds including process overhead; the two mask stages took 55.02
and 50.24 seconds, and two-input capture took 90.24 seconds. These cold probe
measurements do not establish steady-state inference speed or mask accuracy.

The conservative full-workload forecast was 35,460 seconds (9.85 hours), exceeding
the remaining approved time. It includes cold starts and is an extrapolation,
not a measured full run. The workflow stopped before labels, feature extraction
or training. **There were zero larger-data optimizer updates.** The supervisor
recorded a generic worker-exit error; the specific forecast rejection was
reproduced from the pinned operator and measured stage timings without neural
execution. Both records are retained.

The 82 input/probe files (168,496,316 bytes) were preserved in private cloud
storage. Sampled peak process-tree RSS was 3,918,376,960 bytes; GPU peak memory
was not established. Process cleanup completed, and the provider subsequently
showed Sleeping with no machine assigned. Displayed credits changed from 3.94
to 3.77; this rounded difference is not a final billing statement. Evidence:
`.cache/cloud-prepare/larger-t4-probe-preservation-actual-v1.json` and
`.cache/cloud-prepare/larger-t4-stopped-v1.png`.

Larger-data training, export parity and causal video evaluation remain pending.
The next engineering task is to profile preparation, amortize repeated model
loading where valid, and reuse authenticated prepared inputs before another GPU
allocation. Keep the full frozen split and original failed attempt. A later
allocation needs its own clock and budget; do not rerun the previous authorization.

### Preparation audit after the T4 probe

A model-free source/timing audit reproduces the exact 9.85-hour forecast and
attributes 72.78% to observed-mask generation and pose capture. It identifies two
specific repeated operations on the planned all-available path:

- Every query view constructs FastSAM, DINO and SAM 2.1 again (264 constructions
  of each across the full split).
- Every capture view reloads its object bank and builds 799 FoundPose search
  indices. That is 210,936 index constructions across 264 views.

These are source-derived operation counts, not a measured runtime breakdown.
Capture already loads GoTrack once per command; another network cache would
duplicate existing behavior. Measure model-loading, hashing, renderer creation,
index construction and neural inference separately before assigning speedups.

The proposed implementation batches proposal, descriptor and SAM phases while
keeping one large model resident at a time, then reuses immutable object search
indices with independent per-view refinement state. It must pass same-input
mask, association, candidate, crop and frozen-prediction comparisons. Exact
authenticated CUDA artifacts from the stopped attempt can avoid rebuilding
successful inputs after compatible producer admission. The audit is recorded in
`.cache/cloud-prepare/larger-preparation-performance-audit-v1.json`; no inference
optimization has been executed or measured yet.

## Opt-in amortized preparation

The local implementation now provides `legacy-v1` (default) and `amortized-v1`.
The latter is frozen into the preparation spec, source closure, plan and receipts;
requesting a different variant is rejected. Its additive modules are
`quality_adaptation_larger_optimized.py` and
`quality_adaptation_larger_parity.py`. The CNOS, FoundPose, FAISS and GoTrack model
modules retain their exact original bytes.

The new path processes an object's views with FastSAM, then DINO, then SAM, loading
one model for each phase. Floating proposal masks remain unchanged until DINO
cropping consumes them. Capture shares one object's bank and search indices while
constructing a fresh renderer, refiner and capture sink for each physical view.
Model, index and temporary-file cleanup is explicit. Timings cover setup and
per-view operations, with CUDA synchronization when applicable.

The parity tools compare actual output files and input hashes, retain the full
requested-state inventory, and reject missing or nonfinite evidence. Separate
diagnostic APIs run held model/search state through A→B→A; they do not add duplicate
training frames. Independent source review passes; actual neural comparisons have
not run. The review found and fixed mismatched CNOS failure evidence and removal
of pre-existing empty scratch directories. Failed views remain in comparisons.

The final 29 focused model-free tests pass with actual local disk conditions.
The independent reviewer passed 242 combined tests using an explicit disk-space
fixture because the laptop had 0.70 GiB free, below the unchanged 1 GiB production
reserve. These checks provide software evidence only. Nine source-installer
checks and 18 CPU-operator checks also pass. The parent independently reran the
29, nine and 18 focused checks.

Source-only ROOT30 packaging preserves ROOT29 and includes separate legacy and
amortized comparison specs with identical object/view splits. The installer,
builder and model-free cloud preflight are under `.cache/cloud-prepare/` with the
`larger_optimized_` names. This package is prepared locally; installation and real
cloud resource checks remain pending. The previous GPU v2 operator still targets
the preserved ROOT29 implementation. A new bounded CPU comparison operator is
prepared and independently reviewed before any further paid allocation or
full-training attempt. It has its own immutable 900-second diagnostic clock and
does not reset either previous experiment's clock. Both CPU branches use identical
authenticated CUDA-produced banks; this diagnostic does not claim CPU/CUDA bank
equivalence. Large inputs and outputs remain in the cloud.

Final reviewed source manifest: `19c242dd8abe9448c8a7210a41dd32be817cacbb055d59cba190f384ac407015`.
CPU comparison operator: `fa4a6ffeb0bb628183cef90641866fa68deb03395cf667981b53fcfafb43f435`.
Review receipt: `.cache/cloud-prepare/larger-preparation-optimization-review-v1.json`,
SHA-256 `ed3dbd77face1cdb23aaa220fd19c81b9fe6e6f2c2c31c921acf6336ef119674`.
The studio is still asleep. Automatic approval review rejected opening its Start
selector because it might wake the previous paid T4. A separate capped startup
approval or a user-selected free CPU profile is required before executing this
comparison. No new larger-data training updates have occurred.
The learned head stays experimental. Training completion alone cannot establish
better rotations, accurate surface attachment, generalization, 30 FPS or mobile
readiness. Compare against the frozen baseline on the same rows and report
availability alongside errors before any promotion.

## Full amortized execution driver

The new `larger_amortized_cuda_execute_v1.py` connects the reviewed preparation
variant to the existing extraction, sealing, labeling, training, resume and export
APIs. It first requires completed CPU numerical evidence, then runs a separate
CUDA parity check before preparing the full frozen split. Preparation uses object
batches; the forecast accounts for model setup and warm frame work separately.
The original experiment clocks and source bundle remain intact.

Before sleeping or switching the CPU machine, preserve its referenced crop and
geometry arrays using the new evidence bridge. The evidence bundle is a JSON map
of actual CPU report/snapshot file hashes, not training labels. Create it from the
completed CPU comparison; a source review or test fixture cannot replace it.
On that CPU host, with the actual bundle path and hash supplied:

```bash
VISUALIZEIT_CLOUD=1 python -B larger_amortized_cuda_execute_v1.py seal-cpu-evidence \
  --evidence-bundle "$CPU_EVIDENCE_BUNDLE" \
  --evidence-bundle-sha256 "$CPU_EVIDENCE_SHA256" \
  --output /teamspace/studios/this_studio/visualizeit-private-larger-v1/amortized-cpu-gates/v1
```

After separate approval of a fresh GPU allocation, supply its pinned authorization:

```bash
VISUALIZEIT_CLOUD=1 python -B larger_amortized_cuda_execute_v1.py execute \
  --authorization "$NEW_AUTHORIZATION" \
  --authorization-sha256 "$NEW_AUTHORIZATION_SHA256"
```

Keep the reviewed CPU operator and old CUDA helper beside this new driver. They
are execution helpers, not additional members of the 344-file ROOT30 source
manifest. The package `.cache/cloud-prepare/larger-amortized-full-workflow-source-v1.zip`
contains the original reviewed preparation package, the new driver/tests and its
review receipts. Its SHA-256 is
`0ff88bb1a16456c4e1a7536b1ea32d26ecc261cea55e70242a613f178a4bf1e9`.
Verify it before extraction, then extract the nested preparation package into the
same helper directory. This archive contains no models or datasets and has not
been uploaded or run in the cloud.

Final independent source review passed: 36 model-free tests in 25.189 seconds.
It found and verified fixes for duplicate counting of shared model-cache aliases
and disagreement with the existing trainer's durable-storage accounting. All
private larger-run output history counts toward the existing 1.2-billion-byte
limit; scratch remains under its separate limit. Tests use tiny files and explicit
resource fixtures because the laptop is below the production disk reserve.

Operator SHA-256:
`51e7a6d6b4b8f587ef38dbc6ba13cea5c82fc15186ba8a4f178787136645eb17`.
Review receipt: `.cache/cloud-prepare/larger-amortized-full-operator-review-v1.json`,
SHA-256 `5b26e1f0eaf2d51b9f7f4fd48ba3ae36d4367ba7c83d4b80c7a1131f4a86c9f2`.
Actual CPU/CUDA parity, measured timing fit, larger-data training and fresh-network
export verification remain pending. There are still zero larger-data optimizer
updates. A successful Python exit does not establish that cloud billing stopped;
the organizer must verify the provider's Sleeping state.
### Additional Linux filesystem check

A local Ubuntu WSL probe passed seven checks using real POSIX symlinks against
the exact reviewed driver. Shared-cache aliases counted once, distinct physical
caches accumulated, and escaping, internal, ancestor, broken and cyclic links
were rejected. It used tiny synthetic files, imported no Torch, and removed its
temporary fixture. The receipt is
`.cache/cloud-prepare/larger-amortized-local-linux-cache-probe-v1.json`.

This check ran on local Linux/Python 3.14.4. It does not establish compatibility
with the cloud environment or numerical parity. Linux NumPy and psutil were
absent; no dependencies were installed. Cloud installation, resource inventory,
CPU/CUDA comparisons and training still require actual execution.
### Provider startup attempt, 2026-10-16

The approved startup selected Lightning's explicitly free 4-CPU profile. It stayed
queued for cloud capacity, and source upload never reached a file chooser.
Lightning's official status page reported unresolved emergency maintenance:
https://status.lightning.ai/incidents/h20stxqvwg8r . The allocation was cancelled
and the studio's Sleeping state verified. Displayed credits remained 3.74; that
rounded display is not a final invoice. No model process or training step ran.

For a fallback or fresh instance, all three pinned source trees are now archived
without needing access to the old studio:
`.cache/cloud-prepare/larger-amortized-three-root-source-v1.zip` contains 968 source
files and their three exact manifests, with every member authenticated after
round-trip extraction. It is 5,700,194 bytes; SHA-256:
`cbc1e698fce71f1be1561ac1033a1a7bbbe82fa1a8ddbd6050036ba56b6f8a7a`.
The source trees retain their existing hashes. No models or datasets are included,
and this archive has not been uploaded. A new provider still needs its execution
adapter, authenticated model/data assets and actual runtime validation.

Kaggle's user-selected ZenKqi Google profile currently reaches unfinished
registration. GPU availability cannot be checked until account setup/sign-in is
complete. No new Kaggle account or training session was created by the agent.
