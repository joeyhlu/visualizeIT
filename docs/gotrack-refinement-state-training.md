# GoTrack training on later refinement states

This additive development workflow captures training crops along the current
integer-coordinate inference path. It reuses the existing frozen feature
extractor, supervision projection, manifest assembler, head trainer and export
format. The original released head remains the serving default.

## Experiment

Use the three existing synthetic sequences and their cached automatic SAM masks.
Each physical frame runs refinement once from the first quality-sorted FoundPose
estimate. No reference-pose fallback is permitted.

| Split | Physical object | Frames | Prescribed iterations | Requested crops |
| --- | --- | --- | --- | ---: |
| TRAIN | 17, two pitches | 0–7 in each sequence | 0,1,0,3,0,1,0,3 by frame | 16 |
| Previously used VAL | 27, one pitch | 0–7 | 0,1,3 for every frame | 24 |
| Sealed TEST | 7 | None | None | 0 |

Missing later iterations remain unavailable; they are never replaced with an
iteration-zero crop. A valid retained crop can survive a subsequent PnP failure.
Capture, extraction, label and final-pose availability are recorded separately.
All requested rows, including failures, remain in the inventories.

The head starts from the original released weights, with one epoch, batch one,
seed 42 and at most 16 updates. Backbone, decoder and SAM stay frozen. This tests
a suspected training/inference coverage gap; it does not establish its cause.
A controlled comparison needs the same physical TRAIN frames at iteration zero,
the same VAL state inventory and the same completed update count. Different
completed update counts must be reported as an incomplete comparison.

## Source and label separation

Capture requires its new source contract and input plan. The preserved source-18
checkout supplies feature extraction, assembly, training and export in separate
processes, with its original manifest and source lock authenticated. Historical
locks and experiments are not rewritten to accept changed code.

The capture plan contains observed inputs and estimated geometry. Generation,
recipe and split files remain unopened until the prelabel seal authenticates the
capture, geometry, feature inventories and frozen prediction parity. Only the
separate label stage uses synthetic reference geometry. TEST has metadata only.

## Cloud commands

Run these in the reviewed capture checkout after its source package and existing
cloud paths have passed preflight. The commands use the existing Linux runtime;
they do not install dependencies, fetch data or provision hardware.

```bash
export VISUALIZEIT_CLOUD=1 PYOPENGL_PLATFORM=egl XFORMERS_DISABLED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SAM2_BUILD_CUDA=0

VIT_PREP=.cache/cloud-prepare
VIT_PRIVATE=/teamspace/studios/this_studio/visualizeit-private-refinement-states-v1
VIT_CAPTURE="$VIT_PRIVATE/capture"
VIT_FEATURES="$VIT_PRIVATE/features"
VIT_RUN="$VIT_PRIVATE/run"
VIT_SOURCE18=/teamspace/studios/this_studio/visualizeit-cloud-v18
VIT_PLAN="$VIT_PREP/refinement-state-input-plan-v1.json"
VIT_LABELS="$VIT_PREP/refinement-state-label-plan-v1.json"
VIT_CONFIG="$VIT_PREP/refinement-state-train-config-v1.json"

# Verify these against the reviewed packet before using their hashes.
VIT_PLAN_SHA=$(sha256sum "$VIT_PLAN" | cut -d ' ' -f 1)
VIT_LABEL_SHA=$(sha256sum "$VIT_LABELS" | cut -d ' ' -f 1)
VIT_CONFIG_SHA=$(sha256sum "$VIT_CONFIG" | cut -d ' ' -f 1)
VIT_CMD=(python -B -m bench.quality_adaptation_refinement_states)
VIT_DATA=(--root "$VIT_CAPTURE" --features "$VIT_FEATURES")
VIT_HIST=(--training-source-root "$VIT_SOURCE18"
  --training-source-manifest "$VIT_PREP/source18-cloud-source-manifest-v1.json")

"${VIT_CMD[@]}" capture --plan "$VIT_PLAN" --plan-sha256 "$VIT_PLAN_SHA" \
  --output "$VIT_CAPTURE" --cloud-execution
"${VIT_CMD[@]}" extract "${VIT_DATA[@]}" "${VIT_HIST[@]}" --cloud-execution
"${VIT_CMD[@]}" seal "${VIT_DATA[@]}" --cloud-execution
"${VIT_CMD[@]}" labels "${VIT_DATA[@]}" "${VIT_HIST[@]}" \
  --label-plan "$VIT_LABELS" --label-plan-sha256 "$VIT_LABEL_SHA" --cloud-execution

VIT_TRAIN=("${VIT_DATA[@]}" "${VIT_HIST[@]}" --config "$VIT_CONFIG"
  --config-sha256 "$VIT_CONFIG_SHA" --output "$VIT_RUN" --cloud-execution)
"${VIT_CMD[@]}" train "${VIT_TRAIN[@]}" --stop-after-steps 8
"${VIT_CMD[@]}" train "${VIT_TRAIN[@]}" --resume
"${VIT_CMD[@]}" export "${VIT_TRAIN[@]}" --export-output "$VIT_RUN/head.pt"
```

Monitor the stage supervisor logs/receipts and the existing trainer's `status.json`
and `history.json`. An intentional step-eight stop leaves `last.pt`; resume
rechecks dataset, labels, configuration and saved run artifacts. An unavailable
TRAIN/VAL inventory, failed seal or failed cleanup must be investigated before
continuing. No later crop is synthesized to make the run pass.

The shared limit is 3,600 seconds from capture start, including gaps between
commands. Stage limits are 900 seconds for capture and 900 for extraction. The
reviewed continuation also enforces 1,200 seconds of combined wall time across
the step-eight training invocation and its resume; the trainer's per-invocation
limit alone does not enforce that combined allowance. Each extraction output is
bounded at 512 MiB. All private outputs, including preserved failed extraction
outputs and diagnostics, count toward 2 GiB, with a 1 GiB free-disk
reserve. A run is bounded at 1 GiB and the existing model cache at 8 GiB.
Old artifacts are preserved. Expired or insufficient budgets are not increased
automatically. Stopping Python does not prove that provider billing stopped;
verify the studio is sleeping when finished.

## Verification status

The combined model-free suite passed **149 tests** with no Torch import. The
31 new tests also passed in independent review. They cover manufactured rotated
camera geometry, a mocked 24-frame/40-state capture, actual generic manifest
assembly and feature-receipt validation, unavailable rows, source and label
mutation, resume/export forwarding and watchdog cleanup. Manufactured tests do
not establish real model quality or real renderer parity.

Preflight authenticates every input/video/mesh pin against the serving allowlist
before opening any prospective data file. A regression test rejects foreign
pins in the final row without reading any data or input manifest. An earlier
version checked this too late; its source packet is retained and superseded.

The new source and 40-state plan are pinned in
`bench/runtime/gotrack-refinement-states-v1.json` and the prepared cloud packet.
The existing runtime contracts and original measured probe are preserved.
The initial combined check exposed stale historical source fixtures; narrow
test-only changes now select the new runtime and explicitly double historical
source authentication for manufactured archive-join tests. Production source
verification remains enabled. The initial failures and final results are kept.

The actual cloud source/input preflight passed on the existing free four-CPU
studio: 149 tests, no Torch import, matching dependency versions and a matching
released GoTrack checkpoint. An earlier cloud preflight rejected bank paths
through an existing cache alias. The corrected plan uses their canonical paths;
both files retained their verified byte counts and SHA-256 hashes. The failed
preflight and source checkout were preserved. This checks infrastructure, not
prediction quality.

The first actual capture retained all 40 prescribed crops. Its four-thread
extraction produced all 40 feature files, but the prelabel seal rejected two
confidence differences above `1e-6`. All 40 geometry, RGB and mask checks passed;
38 prediction comparisons passed. That attempt stopped before labels or
training, and its source, features and failure reports are preserved.

A diagnostic on the first failing crop used the unchanged extractor with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Its flow and confidence differences
were both zero. The measured Torch thread count was one; deterministic
algorithms remained disabled in that extractor. This supports a numerical
execution-policy explanation for that crop, rather than demonstrating an
accuracy improvement. Full-inventory parity is still required before labels.
The earlier frozen cloud operator with four-thread settings is a preserved
failed packet; use the one-thread setting in the commands above.

### Completed cloud training smoke

The corrected one-thread continuation completed on the existing free four-CPU
Lightning studio. It reused the original capture and its original 3,600-second
deadline, retaining the failed feature output and both diagnostics.

| Check | Actual result |
| --- | --- |
| Capture and extraction | 40/40 prescribed states available |
| Strict prelabel prediction parity | 40/40 passed |
| Training | One epoch, 16 TRAIN crops, 16 optimizer updates |
| Interruption and resume | Stopped at update 8; restored optimizer state and completed update 16 |
| Validation inventory | 24 crops from 8 dependent frames of one previously inspected object |
| Export reload | Changed head loaded into a fresh frozen network; all 24 VAL crops matched saved-head outputs exactly |
| Combined training wall time | 76.329 seconds, including both invocations and their checks |
| Corrected continuation, including report archive | 330.931 seconds |
| Highest sampled stage process-tree RSS | 2,952,482,816 bytes |
| Private artifacts, including failed attempts | 1,306,748,914 bytes |
| Held-out TEST payloads opened | 0 |

Validation correspondence EPE was **2.011783 px frozen / 1.972604 px adapted**
on the same 24 crops. This is a small, repeatedly inspected synthetic diagnostic
from one object; independent-object uncertainty cannot be estimated. It does
not establish improved pose accuracy, fewer rotation failures or better surface
attachment. The matched iteration-zero training control and complete causal
development replays remain unperformed.

A subsequent audit split those same saved predictions by refinement iteration;
it ran no model and changed no training settings. Each row below contains the
same eight physical validation frames, so the rows are dependent.

| Refinement iteration | Frozen EPE, px | Adapted EPE, px | Crops with lower / higher error |
| --- | ---: | ---: | ---: |
| 0 | 2.325672 | 2.316946 | 5 / 3 |
| 1 | 2.001063 | 1.967671 | 7 / 1 |
| 3 | 1.708612 | 1.633196 | 8 / 0 |

The larger change on later crops is an exploratory observation. These values
come from fixed crops along frozen-head trajectories, not trajectories produced
by the adapted head. They cannot replace the matched training control or causal
video replay.

The exported head remains in cloud storage:

```text
/teamspace/studios/this_studio/visualizeit-private-refinement-states-v1/run-thread1-v1/head.pt
80,150,742 bytes
SHA-256: 0a8d647bc1aeb4cda8bb445d9375dd01ea7f6e8c7e284f822a59d785071c8c2f
```

The original exporter writes `inference_parity_executed=false` in its sidecar;
the separate, subsequent `export-parity.json` records the actual fresh-network
verification. Keep both records intact. The authenticated report archive is
119,966 bytes with SHA-256
`8f06d8f6378b194765df00fa17b5e8c178ce1de39192e93f67fdb0f4da4130c3`.
All 85 archive members were checked locally. Large data and checkpoints stayed
in the cloud.

After collection, no model workers or stage lock remained. The provider then
showed the studio sleeping with no machine assigned. Credits displayed 3.94
before and after this free-CPU session; that rounded display does not measure
an exact credit delta. No paid hardware or storage upgrade was enabled.

This proves the capture-to-training/resume/export bridge ran. Admission of this
new head into the offline CLI's immutable artifact registry is still pending,
as are the controlled comparison and video replays. The released head remains
the serving default. Export alone does not authorize promotion into the
offline pipeline.

These dependent views of one TRAIN and one repeatedly inspected VAL object do
not establish generalization. The separate held-out test object stays unopened.
Phone performance, 180-degree error reduction and surface attachment accuracy
remain unmeasured for this experiment.
