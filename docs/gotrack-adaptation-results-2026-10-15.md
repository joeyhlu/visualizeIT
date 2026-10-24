# GoTrack correspondence-head smoke — October 15, 2026

Actual cloud optimization, interruption/resume, export and fresh-network reload
completed on the existing free Lightning four-CPU machine. This was a two-step
engineering smoke, with two synthetic training crops of carton object 17 and
one synthetic validation crop of dinosaur object 27. It is insufficient evidence
for a tracking accuracy or generalization claim.

## What ran

1. Reused the pinned SAM masks, FoundPose banks and GoTrack network. Missing
   training masks were produced with frozen SAM; SAM was not fine-tuned.
2. Captured actual estimated-seed GoTrack query/template crops. No reference
   poses were used for candidate generation or feature extraction.
3. Cached the frozen backbone/decoder outputs and sealed their identities before
   opening the separate synthetic generation recipe to create supervision.
4. Updated only the existing GoTrack correspondence head: 20,029,955 parameters,
   batch size 1, CPU float32, AdamW learning rate 0.00001, two optimizer steps.
5. Stopped deliberately after step one, restored the optimizer and batch cursor,
   and trained on the second crop. Exported and attached the head to a freshly
   constructed original GoTrack network.

The backbone and decoder tensor hashes were identical before and after attaching
the export. Cached-token and full RGB inference both agreed exactly with the
trained head in this run. A wrong export digest was rejected without mutation.
The untouched test object 7 remained unopened. The legacy pipeline remains the
default; this research export is not automatically selected by `quality_runner`.

## Measurements

These metrics use independently rendered synthetic correspondence supervision
on the one validation crop, not independent real-world attachment annotations.

| Metric | Frozen head | Adapted head |
| --- | ---: | ---: |
| Mean visible correspondence endpoint error | 1.330979 px | 1.310551 px |
| Visible correspondences within 1 px | 31.8816% | 33.7434% |
| Visible correspondences within 3 px | 97.1299% | 97.1077% |
| Visibility false positives / false negatives | 272 / 0 | 272 / 0 |
| Available validation crops | 1 / 1 | 1 / 1 |

No confidence interval is reported because there is only one validation object.
The small endpoint-error change and mixed threshold results do not establish an
improvement. Pose errors, severe rotations, surface slipping, real occlusion
recovery and Android throughput were not evaluated in this smoke.

- Actual capture: 46.352 s; frozen extraction: 13.223 s; sealing: 0.265 s.
- Continuation including label generation, two training invocations, export and
  verification: 38.715 s. First training invocation: 7.805 s; resume: 6.289 s.
- Peak sampled process-tree memory: 3,086,270,464 bytes during fresh-network
  verification. One RGB forward: 1.669 s on this CPU; this is not pipeline FPS.
- Export: 80,141,142 bytes. Private outputs including optimizer checkpoints:
  597,339,932 bytes; source packet: 6,598,777 bytes; shared model cache:
  3,790,245,300 bytes. Heavy artifacts stayed in cloud storage.
- The machine was verified **sleeping with no allocation** after completion.
  Displayed credits remained 3.94; no paid resources or new pretrained weights
  were used. The displayed balance is not an itemized billing statement.

## Failure retained and repaired

The first run failed before optimization because the label adapter expected an
absent `recipe_kind` field and compared historical `validation` directly with
internal `val`. All failed rows and receipts were retained. The repair validates
the actual historical generation schema, maps the split name explicitly, and
keeps every recipe/mesh/video/object/sequence hash check.

The continuation reused the exact sealed features in a new directory with an
explicit compatibility receipt for unchanged extraction dependencies. It did
not rerun capture or extraction, overwrite failed labels, relax identity checks,
or reset the original execution deadline. It completed before the conservative
08:49 UTC cutoff.

## Reproduce and load

For a fresh cloud smoke with a reviewed, hash-pinned input and label plan:

```bash
export VISUALIZEIT_CLOUD=1 PYOPENGL_PLATFORM=egl
bash scripts/cloud_adaptation_smoke.sh \
  /cloud/plan.json /cloud/label-plan.json \
  bench/runtime/adaptation-smoke-cpu.json /cloud/private/smoke-1
```

The launcher performs capture, extraction, seal, labels, step one, interruption
inspection, resume, export and verification in separate sequential processes.
The complete input contracts and individual train/resume/export commands are in
[the adaptation guide](model-adaptation-cloud.md). The private dated continuation
operator is an incident recovery artifact, not the general launch command.

The actual cloud export is:

```text
/teamspace/studios/this_studio/visualizeit-private-adaptation-v1/run-v2/head.pt
SHA256 942b35b52536fe303a9701296b27e90009b88a89216b0883cf045cf221ed7b41
```

For a separately reviewed inference replay, load it using the existing
`quality_adaptation_runtime.load_export_into_network` helper with that expected
digest and the original compatible network. This exact loading path was verified
against full RGB inference. It does not promote the head to production.

Source lock: `44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33`.
Pinned upstream GoTrack revision: `68f76055755f2a4a8967e13ece834f975f008bdf`.
Focused local tests: 60 passed, without importing Torch or running local models.
The cloud repeated these tests successfully.

## Offline inference integration

A separate compatibility contract now loads this export through the ordinary
offline pose runner. The original training source lock and export remain
unchanged. The default legacy path remains selectable. The opt-in contract is
restricted to three authenticated historical synthetic bundles; it is not yet a
general real-camera adapter. See [the inference commands](gotrack-adapted-inference.md).

All 76 focused local inference checks passed without importing Torch. Independent
review also checked source identity, atomic failure handling, integer-center
projection, one-time principal-point correction and native/crop transforms.
These software checks do not establish model quality.

For the next paired development comparison, frozen SAM generated the first eight
observed masks of each of the same three synthetic sequences. All 24 requested
masks were available; the other 48 inventory frames remain explicitly unmeasured.
Preparation took 405.09 seconds on free CPU, with no optimizer steps. All three
frame-zero mask hashes match those used for the original training capture.
Sampled frame 0/3/7 overlays were visually inspected; no independent mask IoU was
measured. The cloud machine was verified sleeping afterward.

An actual cloud gate then ran both heads from identical estimated FoundPose
seeds, with five ordinary GoTrack refinement/PnP iterations per case. All six
case/head combinations returned finite poses. The first crop's five arrays and
crop transform matched the historical captured inputs exactly; there were 30
full-network forwards in total.

| Development case | Frozen/adapted rotation disagreement | Translation disagreement |
| --- | ---: | ---: |
| TRAIN object 17, minus 25 degrees | 0.092863 degrees | 0.168565 mm |
| TRAIN object 17, plus 25 degrees | 0.327551 degrees | 1.249450 mm |
| Used validation object 27, minus 25 degrees | 0.110058 degrees | 0.549735 mm |

These are differences between predictions, not errors against reference poses.
They show that the exported head executes through full refinement. They do not
show which head is more accurate. Correspondence counts are not RANSAC inlier
counts. No reference poses were read during this inference comparison.

The first sequential CLI attempt stopped before loading the network because the
existing CPU/unlit/no-MSAA model-smoke prerequisite was missing. That failed
attempt and all not-run rows were retained. A bounded continuation prepares the
two required frozen model smokes and reuses the completed crop gates; it does
not rerun training. That continuation completed all planned comparisons.

### Actual sequential development comparison

Both conditions use the same corrected synthetic geometry, legacy ranker,
observed masks and original recovery rules. This isolates the head replacement;
it is not a comparison against every aspect of the old legacy geometry.

| Eight-frame sequence | Frozen tracking / requested | Adapted tracking / requested | Frozen stage wall | Adapted stage wall |
| --- | ---: | ---: | ---: | ---: |
| TRAIN object 17, minus 25 degrees | 7 / 8 | 7 / 8 | 203.56 s | 205.65 s |
| TRAIN object 17, plus 25 degrees | 7 / 8 | 7 / 8 | 205.20 s | 206.13 s |
| Used validation object 27, minus 25 degrees | 7 / 8 | 7 / 8 | 200.87 s | 201.70 s |

All 48 requested condition/frame evaluations have outputs and observed masks.
Each head had 21/24 tracking/visible frames and 3/24 recovering/suppressed frames.
Every suppression was the first frame awaiting a second validated pose; there
were no lost states in this prefix. These are state/availability counts, not
independently verified pose correctness or invalid-overlay measurements.

Separate four-frame runs for both heads exactly matched their respective first
four eight-frame poses, candidates and decisions after removing profiling fields
only. Each short run had three tracking frames and one initialization frame.
Those eight repeated evaluations are causality checks, not extra independent
data. This prefix does not test recovery after an occlusion and return.

The eight-frame process-tree RSS samples ranged from 2,343,043,072 to
2,544,640,000 bytes. The four-frame controls peaked at 2,661,687,296 bytes;
the earlier full-refinement gate peaked at 2,676,248,576 bytes. These are sampled
RSS values, not guaranteed maxima or GPU VRAM. CPU stage times include loading,
initialization, inference and output; masks were cached. There were no repeated
timing trials or warm-up controls, so these figures do not support a speed claim
or end-to-end FPS estimate.

The eight-frame trace's separate single-process high-water RSS measurement was
2,561.105–2,562.023 MiB. Its scope and measurement method differ from sampled
process-tree RSS; neither is evidence that adaptation reduced memory use.

Continuation preflight, two model smokes and eight replay processes took
1,556.07 s. With the retained original gate/failure time, the cumulative bounded
experiment took 1,786.67 s, within the original 1,800 s budget. Mask preparation
was a separate 405.09 s stage. The final source tree used 10,211,149 bytes and
the shared model cache 3,790,247,355 bytes. No optimizer steps or reference-pose
reads occurred in the serving comparison. Test object 7 stayed unopened.

The compact report contains 42 authenticated files (270,312 archive bytes):
SHA256 `f44c19f4dbf9847b687922ac2f13078935103aa72ca349dacab7e17ee39a0e26`.
It is preserved locally at `.cache/adapted-paired-continuation-v1` alongside the
original failed report. The 80 MB head and heavy data remain in cloud storage.
The provider was verified sleeping with no allocated machine after completion;
the displayed credit balance remained 3.94.

Independent review reauthenticated every archive member and all 347 current
source pins, verified all requested head/mask/frame joins, and recomputed causal
equality across all nonprofiling trace fields, including provenance and memory
state. Each output remains explicitly `diagnostic_prefix`, not a completed
24-frame capture or segmentation experiment.

### Post-seal synthetic-reference pose evaluation

After all predictions were sealed, a separate evaluator authenticated the three
original generation receipts and pose sidecars. These six files total 36,441
bytes. They were downloaded while the cloud machine remained asleep. The
evaluator runs locally with NumPy and the existing pose-error functions; it
does not run inference, train, render or feed reference poses into tracking.

Translation error below is the Euclidean distance between emitted and reference
camera-from-object translations, converted from metres to millimetres. It
describes agreement with these synthetic renderer references.

| Development sequence | Emitted / requested, each head | Frozen translation mean / p95 | Adapted translation mean / p95 |
| --- | ---: | ---: | ---: |
| TRAIN object 17, minus 25 degrees | 7 / 8 | 1.081 / 1.406 mm | 1.048 / 1.389 mm |
| TRAIN object 17, plus 25 degrees | 7 / 8 | 6.105 / 8.255 mm | 5.581 / 7.853 mm |
| Used validation object 27, minus 25 degrees | 7 / 8 | 3.686 / 5.955 mm | 3.110 / 4.810 mm |

The two TRAIN views are resubstitution/development evidence from one physical
object: their pooled mean translation errors are 3.593 mm frozen and 3.314 mm
adapted on 14/16 emitted frames. The used-validation object has 7/8 emitted
frames. It remains one previously used validation object, not an independent
test or seven independent replicates.

Temporal translation-increment residual is the norm of the difference between
predicted and reference translation changes. It is evaluated only when both
consecutive frame endpoints have emitted poses; missing frames are not bridged.

| Development sequence | Valid / possible transitions, each head | Frozen increment residual mean / p95 | Adapted increment residual mean / p95 |
| --- | ---: | ---: | ---: |
| TRAIN object 17, minus 25 degrees | 6 / 7 | 0.895 / 1.344 mm | 0.995 / 1.606 mm |
| TRAIN object 17, plus 25 degrees | 6 / 7 | 1.277 / 2.111 mm | 1.384 / 2.063 mm |
| Used validation object 27, minus 25 degrees | 6 / 7 | 2.287 / 5.277 mm | 2.541 / 5.047 mm |

This is a mixed result: mean translation error is lower for the adapted
head in these cases, while mean translation-increment residual is higher.
Availability is unchanged. Increment residual is not visible surface slipping
or attachment accuracy. No confidence interval or population claim is made
from two objects and three dependent development sequences.

All 36 recorded candidates per head had valid poses and passed the existing
gates; each head had candidate coverage on 24/24 frames. This does not establish
that the candidates were correct. Candidate minima before and after gates are
identical in this sample. Their per-frame translation errors are retained as
counterfactuals within each head's own candidate sets. Candidate coverage
includes initialization, unlike the 21 emitted poses; these denominators must
not be interchanged.

All three original reference sequences are explicitly marked observationally
ambiguous. Primary rotation error and severe-rotation failure count/rate are
therefore **unmeasured**, with zero eligible orientation samples. They are not
reported as zero errors. Surface attachment, mask IoU, occlusion-return recovery,
untouched-test accuracy and phone/GPU throughput remain unmeasured.

The new `bench.quality_postseal_pose_evaluate` CLI has 17 focused analytical and
failure tests. Its actual run took 0.459 s with Python 3.12.14 / NumPy 2.5.3.
All 50 input files remained byte-identical; no Torch or inference modules were
loaded. It generated a 211,486-byte report at
`.cache/adapted-pose-evaluation-v1/report.json`, SHA256
`c4f323daf4236e420c97318ea750c11d3ebc49175dd40d740952c29a097d5685`.
The report includes separate TRAIN/used-validation summaries, every requested
frame's availability and null errors for unavailable poses, and paired
differences on the common emitted/reference intersection. The four-frame
causal repeats and frames 8–23 add no quality samples.

The learned head remains opt-in. These measurements do not justify promoting
it over the frozen baseline or claiming a fix for 180-degree rotation failures.
Reproduction commands are in [the inference guide](gotrack-adapted-inference.md).

Independent review recomputed every error, candidate minimum, paired difference,
temporal row and distribution directly from the sealed outputs and original
sidecars using standard-library arithmetic. Results agree within 1e-12 metres.
It also verified all 48 availability rows, separate TRAIN/used-validation
denominators, and unchanged input/source hashes. The independent audit is
`.cache/cloud-prepare/postseal-pose-eval-review-v1-actual-audit.json`,
SHA256 `6a90f619dc02a28e909b4afa7d81acc8ddeeb48ba30e4da7262b208aa635b27f`.

A second invocation through the standalone `python -m` CLI, using a fresh output
filename, produced the exact same report bytes and SHA256. It repeated numerical
evaluation only; no training or model inference was repeated.

This comparison reuses development objects 17 and 27. It is not an untouched test
or evidence of generalization. A larger disjoint experiment and independent
surface-attachment measurements remain necessary. SAM fine-tuning still needs
independent mask annotations and evidence that segmentation is a bottleneck.
