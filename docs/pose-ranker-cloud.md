# Offline pose ranking and cloud training

## Experimental ordering purpose (October 15 implementation)

`experimental_ordering_v1` is an explicit development fitting purpose. Its JSON
artifact uses schema 4 and loads only with `make_ranker("ordering-research", ...)`.
The ordinary `learned` factory rejects it. Production schema 1, score-only
integration schema 2 and confidence-veto schema 3 keep their existing contracts.
Both research architectures have now completed cloud training, resume/export
checks and fixed-candidate development comparisons. The selected MLP also
completed a causal offline replay. See [measured October 15 results](pose-ranker-results-2026-10-15.md)
for counts, artifact locations and limitations. Earlier dated results below
remain historical evidence; they describe the state before this experiment.

The substantive gate still requires eligible good/bad alternatives within frames
in both TRAIN and VAL. Research fitting instead requires at least one TRAIN
frame containing genuinely observable angular good/bad supervision, a positive
pairwise loss weight, and both quality classes among eligible VAL candidates.
VAL classes may occur in separate singleton frames. Such VAL data can support
available-dimension quality calibration and checkpoint selection, but cannot
measure improvement in choosing between alternatives within a frame.

The target is named `frozen_available_dimension_quality`. The frozen ambiguity
policy remains authoritative: an ambiguous rotation is excluded from rotational
supervision; translation errors remain errors. A positive translation-only label
does not establish full pose or full rotation correctness. The artifact records
the original cache audit ranking-signal flags separately from research
eligibility, frame/sequence/object supervision units, dependent pair counts,
observability counts, calibration scope, and unsupported claims. Four pairs from
one frame remain one frame of evidence.

### Complete-cache descriptor and extraction provenance

Create a small descriptor on the approved cloud host. All paths may be absolute
or relative to the descriptor. Each file digest below is SHA-256 of exact file
bytes. All complete source caches contribute their full TRAIN and VAL contents;
there is no filtering for successes or failures.

```json
{
  "schema_version": 1,
  "artifact_purpose": "experimental_ordering_v1",
  "sources": [
    {
      "manifest": "/cache/original/cache_manifest.json",
      "manifest_sha256": "<file digest>",
      "provenance": "/cache/original/extraction-provenance.json",
      "provenance_sha256": "<file digest>"
    },
    {
      "manifest": "/cache/added/cache_manifest.json",
      "manifest_sha256": "<file digest>",
      "provenance": "/cache/added/extraction-provenance.json",
      "provenance_sha256": "<file digest>"
    }
  ],
  "encoder_pins": {
    "encoder_weights_sha256": "<exact original weights digest>",
    "frozen_feature_configuration_sha256": "<exact extraction configuration digest>"
  }
}
```

Each extraction provenance sidecar has exactly these fields:

```json
{
  "cache_manifest_sha256": "<source cache manifest file digest>",
  "source_trace_sha256": "<source cache source_trace_sha256>",
  "encoder_pins": {
    "encoder_weights_sha256": "<same exact weights digest>",
    "frozen_feature_configuration_sha256": "<same exact configuration digest>"
  },
  "source_receipts": {
    "original_prediction_seal": "<original receipt file digest>",
    "original_stage_receipt": "<original stage receipt file digest>"
  },
  "inference_sealed_before_labels": true
}
```

The encoder-pin and receipt dictionaries allow additional named digests, but
must be nonempty and contain lowercase SHA-256 values. The operator must derive
pins from the actual original extraction weights/configuration and retain those
receipts. These sidecars are explicit operator assertions; the existing cache
schema contains no independently recoverable weight provenance. Matching feature
names alone do not prove matching encoders. The loader checks sidecar/file pins,
cache/trace binding and exact pin equality across sources.

`bench.quality_ordering_research.load_research_dataset(Path(descriptor))` audits
each source with `training_only=True`, checks the frozen feature schema and
complete shard metadata, and validates a union of full frozen split inventories.
It retains original physical-object, scene, mesh, alias, frame/source-frame and
split assignments, including unavailable inventory. Exact duplicate frame/label
rows are deduplicated by complete identity; any conflicting overlap or policy
fails. TEST and DEV inventory and shard hashes stay bound as metadata, while
their feature/label shard files remain unopened. No source cache is rewritten.

### Cloud fitting, interrupted resume and export

```bash
python -m bench.quality_ranker_train ordering-smoke \
  --manifest ordering-manifest.json --output ordering-linear-smoke \
  --architecture linear --cloud-execution
python -m bench.quality_ranker_train ordering-train \
  --manifest ordering-manifest.json --output ordering-linear \
  --architecture linear --cloud-execution
python -m bench.quality_ranker_train ordering-resume \
  --manifest ordering-manifest.json --output ordering-linear \
  --checkpoint ordering-linear/last.pt --architecture linear --cloud-execution
python -m bench.quality_ranker_train ordering-export \
  --manifest ordering-manifest.json --output ordering-linear/model.json \
  --checkpoint ordering-linear/best.pt --cloud-execution
```

Use `--architecture mlp64_32` for the second bounded head, or `--config` for an
explicit recorded `TrainConfig`. Resume must use the exact original config and
directory. Optimization is prohibited on this workstation and requires the
existing explicit approved Linux cloud guard.

The loss remains equal-frame BCE plus within-frame pairwise softplus, retaining
every eligible labeled candidate, including all-bad frames. Normalization fits
TRAIN only. VAL alone selects checkpoints and calibrates acceptance and the
distinct-pose margin. A one-class top-choice validation set retains the existing
maximum rejection margin; the research purpose does not relax recovery gates.
Source files, descriptor/cache/split/shard hashes, config/seed/runtime, label
policy, encoder provenance, supervision counts and purpose are bound through
resume and export. Incomplete runs cannot export: schema 4 requires the retained
validation-selected `best.pt`, completed `last.pt`, and matching completion
receipt. JSON and ONNX export compare both scores and actual decisions; ONNX
metadata also carries the research purpose and run hashes.

`ordering-smoke` runs two real cloud epochs, interrupts after one completed
optimizer step, retains partial evidence before resume overwrites `last.pt`,
checks exact selected/final model and optimizer/scheduler/scaler/RNG parity,
and performs JSON/ONNX score and decision parity. Local analytical tests only
check contracts and manufactured examples; they establish no model quality,
rotation improvement, generalization, attachment accuracy, or phone FPS.

## Status and scope

The implementation adds optional ranking to the existing video tracker and a
portable cloud training workflow. SAM 2.1, CNOS, FoundPose, GoTrack and their
encoders stay frozen. The learned ranker is opt-in. Android setup, emulator
builds, correspondence training and SAM fine-tuning remain later work.

**Current verified state (October 14):** cloud training completed for the linear
confidence head (7 epochs) and MLP 64/32 (40 epochs). Both passed exact
interruption/resume and JSON/ONNX export checks. The MLP checkpoint completed a
16-frame causal replay through the existing offline tracker. Compact verified
heads and reports are in `.cache/confidence-research-v1`; the provider Studio is
sleeping with no allocated machine and 3.94 displayed credits.

These heads use the explicit `confidence-research` mode to veto a doubtful legacy
selection. The sample still has zero eligible good/bad alternatives within a
frame, so substantive pose-ordering training remains unavailable. No rotation,
unseen-object or independent surface-attachment improvement has been established.
See the final confidence workflow and causal replay sections for measured results
and launch/resume/load commands. The preparation history below preserves earlier
trials and their outcomes.

### Additional candidate-coverage check — October 14

A fixed batch of four previously unprocessed initial views completed on the
existing free AWS 4-CPU Studio. The order was carton/dinosaur at yaw 90°, pitch
−25°, then carton/dinosaur at yaw 90°, pitch +25°. Each case ran observed SAM
segmentation followed by the existing complete legacy pose stage with frozen
patch features. The carton retained its default renderer; the dinosaur retained
its previously smoke-tested explicit no-multisampling diagnostic configuration.

All eight stages completed. Each view produced an available observed mask and
five actual pose candidates. The complete prediction batch was sealed before
reference labels were opened. Existing completed attempts were preserved, and
only the four exact unprocessed frame identities were replaced in a new cache.
Source, model weights, split, label policy and tracking gates remained fixed.

| Quantity | Train carton | Validation dinosaur |
| --- | ---: | ---: |
| Actual traced frames, including earlier attempts | 54 | 21 |
| Recorded candidates | 87 | 45 |
| Eligible candidates | 74 | 45 |
| Eligible good/bad candidate groups | 0 | 0 |
| Eligible all-bad groups | 2 | 4 |

The resulting inventory contains 132 candidates and 75 actual traces across 384
planned frames; 309 frames remain unprocessed. Empty inventory rows must not be
reported as observed mask or model failures. The two new carton views were
rejected as ambiguous recovery hypotheses. The dinosaur views were recovering,
awaiting a second validated pose. All four initial-frame overlays were suppressed;
these single-frame checks do not measure subsequent recovery or temporal slipping.

The shared batch took 460.86 seconds, including pre-stage resource checks. Each
one-frame segmentation stage took 15.10–17.36 seconds and each pose stage took
92.64–95.40 seconds, including model setup. Peak sampled stage process-tree RSS
was 2,739,048,448 bytes (2.55 GiB). These are CPU stage measurements, not steady
frame latency or phone performance.

The audit still supplies no candidate-ordering supervision under the frozen
rules. No optimizer or calibration ran on this extension; the previously trained
confidence heads remain bound to their original v6 training cache. Ambiguous
orientations retain their existing labels. Dataset-reference errors are recorded
in the audit, but this small synthetic development sample does not establish
independent physical attachment accuracy or unseen-object improvements.

Cloud results are preserved at
`/teamspace/studios/this_studio/visualizeit-cloud-v8/cloud-runs/remaining-initial-views-v2`.
The 21,396-byte report archive was downloaded, its cloud-observed SHA-256 and all
file hashes verified, and its five files extracted to
`.cache/remaining-initial-views-v2`. Raw traces, masks, feature caches and large
weights remain in cloud storage. The preceding `remaining-initial-views-v1`
attempt stopped during input-hash preflight before any vision stage and is
preserved separately.

### Diagnose missing ranking supervision

The reusable `quality_ranker_signal` command reads the audited train/validation
cache and reports each group's quality classes, gate rejections, missing labels,
and distances between recorded poses. It does not load vision models, alter
labels, open test/dev shards, or fit parameters. The report contains derived
distances and label diagnostics rather than pose matrices or feature vectors.

```bash
python3 -B -m bench.quality_ranker_signal \
  --manifest /cloud/candidate-cache/cache_manifest.json \
  --output /cloud/reports/candidate-signal.json
```

Create the output directory first and choose a fresh filename outside the cache.
Existing files and symlinked destinations are refused. The command fails rather
than truncating reports above 100,000 pose pairs or 16 MiB. Cache, split and source
identities are checked before and after reading.

The reviewed command ran against the frozen v7 cloud cache in 0.52 seconds with
38,567,936 bytes of sampled peak process-tree RSS. All 45 focused analytical and
integration checks passed locally without importing PyTorch. The cloud report
and all package hashes were verified after downloading only the 40,671-byte
archive to `.cache/candidate-signal-v1`.

| Finding in the recorded sample | Train carton | Validation dinosaur |
| --- | ---: | ---: |
| Actual frames with angular supervision | 3 | 6 |
| Actual frames with ambiguous orientation labels | 51 | 15 |
| Eligible groups containing only acceptable candidates | 51 | 17 |
| Eligible groups containing only unacceptable candidates | 2 | 4 |
| Eligible groups containing both classes | 0 | 0 |
| Recorded candidate-pair angular disagreement, maximum | 179.23° | 3.62° |

Twenty eligible training pairs have acceptable quality targets and ambiguous
orientation labels despite large angular differences, up to 179.23°. Their
targets retain translation-based supervision; these angular differences must
not be relabeled as rotation errors from the reference pose. Conversely, the
validation alternatives have little recorded angular diversity. Each of its
four unacceptable groups contains a single candidate, leaving no alternative
for an ordering model to choose.

The broader synthetic recipe initially marks every generated orientation as
ambiguous. This is a conservative annotation policy, not a measurement that all
of those images are visually ambiguous. The frozen reference sidecar supplies
angular supervision for the nine frames counted above. Most frames still lack
that supervision. Further rotation experiments need a prospectively defined
source of resolvable orientation labels; changing old labels after inspecting
candidate errors would invalidate the existing comparison.

The sole training group with mixed quality before gates is source frame 9 of
`ranker-multiview-hot3d17-v1-yaw0-pitchm25`. Its candidate at 6.64 mm translation
error is rejected for `localized_or_ambiguous_support`, while an eligible recovery
candidate has 72.45 mm error. Rotation supervision is unavailable for that frame.
This establishes which reference-quality class was removed; it does not prove
that the geometric gate is wrong or justify allowing unsupported poses through.
Nine unavailable targets correspond to attempts with no returned candidate pose.

Here, "before gates" already means **after GoTrack refinement**. Seed poses were
not recorded, so similar final poses cannot establish whether FoundPose supplied
similar seeds or GoTrack made distinct seeds converge. Within-stage pairs and
cross-stage comparisons are reported separately. A missing pair is marked as no
dispersion evidence rather than zero dispersion.

These findings direct the next data work toward prospective, independently
observable orientation supervision and useful candidate diversity. A bounded
seed-versus-refined-pose trace can resolve the remaining generation/refinement
question before more inference is spent. The current sample still does not
support ordinary rotation-ordering training. Existing confidence checkpoints,
tracking gates, feature schemas and label policy are unchanged.

### Trace candidate generation and refinement

`quality_seed_attribution` adds a disabled-by-default observer around the existing
FoundPose recovery backend. `pose_stage(..., seed_attribution_observer=observer)`
supports physical-time recorded-video bundles. The fixed historical mug capture
experiment retains its existing options and guards.

Create `SeedAttributionObserver` with the bundle's physical object ID, sequence ID
and source-frame map. Also enable the existing candidate trace through
`RankingOptions(trace=..., patch_features=True)`. After the pose stage, call
`observer.finalize(candidate_trace, report_path)` in cleanup and require its
`complete` field before interpreting attribution. Use a fresh report path.

The observer snapshots only detached metric 4×4 poses and frame metadata. It
records recovery order, refinement entry seeds, and actual returned candidates,
including `None` and exceptions. Reports join by full frame and occurrence
identity, preserve duplicate seeds, and compare dispersion over the same
surviving candidate pairs. They distinguish missing pairs from zero dispersion.
Raw snapshots are discarded during finalization; the report contains derived
distances, counts and failure reasons. No reference poses are accepted.

Observer failures produce incomplete diagnostic evidence. Backend calls, return
objects, exceptions and tracking decisions retain their existing behavior.
Reports use atomic publication that refuses a concurrent destination. The
focused local suite passes 55 tests, including the output-collision regression,
without importing PyTorch or running local training.

#### Measured two-view cloud probe — October 14

The observer ran on frame 0 of the existing carton and dinosaur
`yaw90-pitchm25` development sequences. Both reused their authenticated observed
masks, FoundPose banks, pretrained weights and original renderer settings. No
segmentation, optimizer, model download or label read ran in this probe. Both
prediction stages were sealed before comparison with the saved source8 outputs.

Each view generated five recovery seeds and returned five refined candidates.
All ten attempts matched their full candidate-trace identities, with no missing
returns, exceptions, dropped records or unmatched candidates. Each dispersion
measurement below uses the same ten surviving candidate pairs before and after
refinement.

| Development view | Maximum seed-pair rotation | Maximum returned-pair rotation | Pose stage, including setup |
| --- | ---: | ---: | ---: |
| Train carton | 178.71° | 179.23° | 105.43 s |
| Validation dinosaur | 22.30° | 0.16° | 99.44 s |

The dinosaur seeds converge during refinement in this case; the carton retains
large angular disagreement already present among its seeds. Neither result
establishes which orientation is correct. They do not justify changing tracking
gates, relaxing ambiguity rules, or fine-tuning GoTrack. Orientation supervision
and useful good/bad alternatives remain necessary for rotation-ordering training.

Both views passed the frozen legacy-output comparison: exact frame identities,
candidate IDs/order, eligibility, rejection reasons, ranking decisions and public
states; pose/feature/validation floats used absolute tolerance `1e-6`, relative
tolerance zero. Only enumerated timing, resource and runner-provenance fields were
excluded. The GoTrack, FoundPose, contract, feature, ranker and training sources
remained byte-identical to the prior source package.

The batch took 212.24 seconds with maximum sampled stage process-tree RSS of
2,644,680,704 bytes (2.46 GiB). This is a CPU diagnostic including model setup,
not steady-frame or mobile latency. The 19,977-byte report archive and all six
file hashes were verified locally in `.cache/seed-attribution-v1`. Raw candidate
traces remain in cloud storage; transient seed snapshots were discarded after
deriving the reports. The Studio was verified sleeping with no allocated machine
and 3.94 displayed credits, unchanged across this probe.

### Initial preparation history

Lightning AI is the current provider choice. The account initially showed the
Free plan, 5 organization credits, no saved payment card, and auto-reload off.
On October 14, 2026, Python 3.11.9 and PyTorch 2.5.1+cu124 ran successfully on an
AWS Tesla T4. Cloud model and rendering smoke checks, FoundPose/CNOS banks, and
a complete 24-frame synthetic carton replay passed. The Studio was explicitly
put to sleep and its sleeping status verified. The displayed credit balance
changed from 5.00 to 4.71; this is the total balance change, not an itemized bill.

The train-only audit found 28 eligible candidates, all meeting the frozen
10-degree/2-cm quality limits. It found no groups with both good and bad
candidates, so the small sample lacks ranking supervision. No optimizer ran.
Substantive ranking training and real-video accuracy comparisons remain pending.
The separate engineering smoke below now verifies actual cloud optimization,
interruption/resume, export and NumPy checkpoint loading. Local tests use analytical
candidates; neither those tests nor this synthetic replay establish improved
tracking, independently verified surface attachment, or mobile readiness.

### Candidate-preparation run — October 14, 2026

The reviewed source archive was deployed and all 242 source hashes verified.
Sixteen frozen clips (384 frames) were rendered from the two already acquired
metric meshes. CNOS supplied observed initialization masks; reviewed positive
points were finalized in fresh inference bundles. Candidate inference reached
55 frames and recorded 92 poses. The other 329 frames are explicitly marked
unprocessed within the compute budget, rather than counted as model failures.
Raw renderer labels remained outside inference. Angular supervision was limited
to visually reviewed first-frame views; other orientations remain conservative.

| Audit quantity | Train carton | Validation dinosaur |
| --- | ---: | ---: |
| Recorded candidates | 77 | 15 |
| Eligible candidates | 64 | 15 |
| Eligible good/bad candidate groups | 0 | 0 |
| Incorrect legacy selections with usable reference targets | 2 | 0 |
| All-bad eligible groups | 2 | 0 |

One training group had mixed quality before geometric gates, but none retained
mixed quality afterward. This sample does not support the required ranking
experiment; an all-bad set needs better candidates before selection can fix it.
That preparation run did not start an optimizer or create a trained checkpoint.
The later engineering smoke completed optimization and export. The final
confidence experiment subsequently completed its causal replay, described below.
These counts describe synthetic renderer-reference agreement, not independently
verified physical attachment accuracy or held-out test performance.

The dinosaur's default renderer smoke failed at silhouette IoU 0.9364 against
the existing 0.94 gate. The existing explicit `--disable-multisampling` option
passed a separate smoke; dinosaur replays therefore use an experimental rendering
configuration. The gate was not relaxed, and default/failed trials are preserved.
Full carton pose stages took 67.43 and 83.17 seconds, including setup and feature
extraction; these are stage times, not real-time frame latency.

The Studio was verified sleeping with no machine allocated before the run's
deadline. Displayed credits changed from 4.71 to 3.95 (0.76 credits; rounded UI
balances, not an itemized bill). Cloud artifacts remain under
`/teamspace/studios/this_studio/visualizeit-cloud-v3/cloud-data/multiview-v1`:
`candidate-audit-bounded-v4.json`, `cloud-run-summary-v4.json`, and the sealed
candidate cache. Large assets remain in cloud storage. The next data experiment
must supply useful eligible alternatives in both splits before training begins.

### Engineering integration smoke

The separate `integration-smoke` command checks optimizer, interruption/resume,
export and checkpoint loading on audited real candidate caches. It permits a
cache without ranking pairs while retaining integrity and split checks. It uses
a fixed linear head, two epochs, CPU execution, seed 42 and one shared 300-second
deadline. It does not run the vision encoders or download data.

```bash
VISUALIZEIT_CLOUD=1 python3 -B -m bench.quality_ranker_train integration-smoke \
  --cloud-execution \
  --manifest /teamspace/studios/this_studio/visualizeit-cloud-v3/cloud-data/multiview-v1/candidate-cache-bounded-v4/cache_manifest.json \
  --output /teamspace/studios/this_studio/visualizeit-cloud-v4/cloud-runs/integration-smoke-v1
```

That output directory now contains the verified run. Choose a fresh name such as
`integration-smoke-v2` when repeating it. Normal `resume` and `export` deliberately
reject integration-only checkpoints; the engineering command performs its own
controlled resume check. The normal commands below apply to substantive runs
with qualifying training and validation data.

Monitor `uninterrupted/events.jsonl` and `resumed/events.jsonl`. A successful
`integration-smoke-receipt.json` records observed optimizer steps, finite losses,
changed parameters, exact resumed final state, and JSON/ONNX score and decision
parity. The controlled interruption is after one completed optimizer step.

`integration-model.json` uses a separate schema-v2 contract: raw sigmoid scores,
unavailable calibration and `selection_enabled=false`. The existing inference
factory can load it with NumPy, but every pose is rejected with
`integration_only_selection_disabled`. ONNX is a score graph with the same
restriction in its metadata and export receipt. This artifact proves checkpoint
integration; it cannot enable overlays or establish improved ranking. Ordinary
`train`, `resume`, `export`, `smoke` and production schema-v1 behavior retain their
existing requirements. A successful engineering smoke does not complete the
substantive ranking experiment.

#### Verified run — October 14, 2026

The engineering command completed on Lightning's **free AWS 4-CPU machine**,
using the already frozen cloud feature cache. It trained 33 linear parameters
for two epochs and 102 optimizer steps. Training supervision contained 61 good
and three bad eligible candidates across 51 groups; validation contained 15 good
candidates across three groups. Both splits had zero ordering pairs, so this
run exercised quality loss without pairwise supervision or fitted calibration.

The interrupted run stopped at step one. Its selected and final model, optimizer,
scheduler, precision scaler, RNG and progress matched uninterrupted execution
exactly. Selected/final parameters were finite and changed from initialization.
Maximum score differences were `3.34e-8` for PyTorch/NumPy and `5.96e-8` for
PyTorch/ONNX, below the documented tolerances. Ordering and disabled decisions
also matched. A separate cloud process loaded the exported JSON through the
existing ranker factory, read five actual cached candidate features without
labels, and rejected selection in normal and recovery contexts without importing
PyTorch. This is checkpoint integration evidence, not a video tracking replay.

The monitored training/export process took 9.57 seconds and peaked at
445,054,976 bytes of process-tree RSS (about 424 MiB). This excludes the previous
vision feature extraction and is not a phone or full-pipeline latency result.
The JSON head is 8,132 bytes, ONNX 1,446 bytes, and the compressed package with
receipts 7,541 bytes. Only that package was downloaded; datasets, frozen vision
weights and resumable checkpoints remain in the cloud. Local NumPy loading and
an analytical sigmoid check passed with no local training or PyTorch import.

Cloud run: `/teamspace/studios/this_studio/visualizeit-cloud-v4/cloud-runs/integration-smoke-v1`.
Local package: `.cache/ranker-integration-v1/`; its `artifact-manifest.json`
binds every downloaded file hash. `integration-smoke-receipt.json` and
`local-load-receipt.json` contain the verified training and loading evidence.
Source archive SHA256 is
`e6c6fc07237784fe38d8a2e7c40320bc59415f3b144d872a9c197025a0c0a331`.

The provider was verified sleeping with no machine allocated after execution.
Its displayed balance remained 3.94 credits before/after this free CPU run.
No GPU, payment card, top-up or new pretrained model was needed. The normal
legacy pipeline remains selected by default. The full ranking milestone remains
open: useful ordering data, validation errors for calibration, MLP experiments,
fixed-candidate comparisons, sequential video replays and independent attachment
evaluation are still required. No accuracy improvement is claimed.

## Components and flow

1. `quality_sam2` propagates observed masks and invokes existing CNOS recovery.
2. `quality_foundpose.FoundPoseRecovery` proposes poses; `GoTrackRefiner` refines
   them using the current image and known metric mesh.
3. `SequentialTracker` applies its existing correspondence, motion and temporal
   gates. An optional `PoseRanker.rank(context, candidates)` selects among the
   eligible candidates or rejects them all.
4. The existing recovery consensus still requires two consistent validated poses.
   Lost, stale or rejected poses suppress rendering.
5. Existing rendering/player and Unity result interfaces consume the same public
   pose and mask/pose/render states. Candidate traces are separate diagnostics.

The existing SAM producer requires a reviewed initial object selection bound to
the first RGB frame, source frame ID, and timestamp. Raw synthetic/BOP bundles
do not supply that selection automatically. The first cloud replay preserved
the resulting preflight failure, then used the existing `quality_runner detect`
command to obtain a CNOS observed mask. After RGB/contour inspection, a fresh
bundle received a positive point inside that mask. Its original video, mesh,
clock, candidate settings, and accuracy limits were retained. No evaluator pose
or projected silhouette was used to initialize segmentation. A raw bundle needs
this initialization step before the `segment` stage can run.

`quality_pipeline` supplies thin protocols around existing `Frame`,
`MaskPrediction`, `PoseCandidate`, backends and renderer; it owns no large model.
`quality_offline` runs stages in separate processes, which release their models
on completion. It records elapsed time, sampled process-tree memory, failures and
logs. CPU execution uses the existing `--device cpu` path where supported.

Ranking choices:

| Choice | Behavior |
| --- | --- |
| `legacy` (default) | Original validation-score sorting and recovery ambiguity ratio. |
| `deterministic` | Fixed geometric, appearance and causal temporal evidence. Experimental. |
| `learned` | Small JSON linear/64–32 MLP head, shared feature schema, normalization and validation-calibrated rejection thresholds. |

Learned probabilities never replace the geometric validation score. The legacy
score ratio is not applied to learned logits. Spatial feature summaries reuse the
already executed GoTrack backbone through an optional hook. Missing evidence has
explicit indicators. Object IDs, labels and reference poses are not features.

## Provider and storage

Use the verified Lightning account's existing credits for the bounded smoke run
after checking the teamspace allocation and current compute/storage costs. Keep
auto-reload off. The optional offer of 25 additional credits requires adding a
card and is not needed to prepare this workflow. Kaggle remains a free alternative:
check **Settings → Accelerator → GPU** and any account verification requirement.
GPU quota and available hardware are account-dependent. No script provisions paid compute,
enables top-ups or purchases storage. Use RunPod only after approval of its
deposit, GPU and storage costs. Hardware availability is not guaranteed.

Check the live [RunPod pricing](https://www.runpod.io/pricing) and
[Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage) before launch.
Keep the checkout, weights, datasets, feature cache and resumable checkpoints on
the cloud machine's approved persistent volume. Template generation and feature
extraction can cost more than training the small head. Include them in the budget.

The model cache retains the existing 8 GiB cap. The launcher caps each run's
output growth and the trainer defaults to 64 MiB for logs/checkpoints. Provider
storage billing includes environments and datasets outside those directories too.

## Cloud setup

Use a Linux cloud image with Python 3.10/3.11, a CUDA-compatible GPU driver, EGL,
OpenGL, `libglib2.0` and `libgomp`. This workflow does not use Docker. Place the
reviewed repository revision **including these changes** on the cloud volume.
The scripts record source hashes as well as the Git revision.

Inside that already approved workspace:

```bash
export VISUALIZEIT_CLOUD=1
# Lightning: select Python 3.11.9 in Studio Environment, then use its existing
# managed environment. It blocks creation of additional conda/venv environments.
export VISUALIZEIT_MANAGED_ENV=1
bash scripts/cloud_ranker_bootstrap.sh
export PYOPENGL_PLATFORM=egl XFORMERS_DISABLED=1 SAM2_BUILD_CUDA=0
python3 -m bench.quality_cloud preflight --root . \
  --output .cache/cloud-preflight.json --require-models
```

Bootstrap downloads the existing pinned pretrained weights directly into the
cloud cache. It does not download a new model architecture or train any weights.
On a regular Linux host, omit `VISUALIZEIT_MANAGED_ENV` and activate
`.venv-cloud/bin/activate` after bootstrap. Managed mode explicitly permits
installation into the provider's existing environment and must remain set for
the training wrapper. It changes setup only; inference and training use the
same dependency pins. The first actual Lightning setup attempt failed at venv
creation before package/model downloads; its log is retained.

Dependencies are pinned in `bench/runtime/requirements-ranker.txt`; the resolved
environment is saved in `.cache/cloud-runtime/resolved-requirements.txt` for
managed mode, or `.venv-cloud/resolved-requirements.txt` otherwise.

## Freeze data before generation

Create a frame inventory with explicit physical object, sequence, scene and
background identities. For observed backgrounds, supply a complete scene object
inventory; a target object's annotation cannot prove other objects are absent.
Keep complete object identities and sequences in one split. Previously inspected
keyboard, mug, bottle and known pilot identities are development data.

The data CLI can adapt the existing bounded BOP/YCB inference and evaluator
manifests without downloading a full dataset:

```bash
python3 -m bench.quality_ranker_data bop-adapter \
  --inference /cloud/data/inference.json --evaluator /cloud/data/evaluator.json \
  --scene-inventory /cloud/data/complete-scene-inventory.json \
  --output-dir /cloud/data/ranker-inventory
```

The adapter converts BOP millimetres to metres in the **separate evaluator
labels**. Review its inventory, missing background information and assignments
before freezing. Example command (the error limits are configurable development
criteria, not measured performance):

```bash
python3 -m bench.quality_ranker_data freeze \
  --inventory /cloud/data/inventory.json --assignments /cloud/data/assignments.json \
  --max-rotation-error-deg 10 --max-translation-error-m 0.02 \
  --output /cloud/data/split.json
```

For BOP images, convert the acquired input to the existing video/metric-mesh
bundle. Supply a separate timestamp sidecar with `timestamp_source`,
`nominal_fps`, and `frames:[{"source_frame_id":10,"timestamp_s":1.0}, ...]`.
Use original capture timestamps with a cited source. The prototype frame IDs do
not establish a frame rate; conversion refuses to guess one.

```bash
python3 -m bench.quality_ranker_bop_bundle \
  --inference /cloud/data/inference.json --reference-id YOUR_REFERENCE_ID \
  --split-manifest /cloud/data/split.json --sequence-id YOUR_FROZEN_SEQUENCE \
  --timestamps /cloud/data/timestamps.json --data-root /cloud/data/acquired \
  --output /cloud/data/train-object --cloud-execution
```

This converts millimetre PLY vertices to metre GLB vertices without recentering,
checks constant intrinsics, and verifies every lossless FFV1 frame against its
source. The output keeps original BOP IDs for label joins. Evaluator and
onboarding poses are not consumed. The media conversion requires a cloud smoke;
local coverage checks its metadata and mocked failure paths only.

Synthetic generation uses permitted metric meshes and the existing renderer,
checks the frozen split before rendering, and writes a video bundle plus separate
reference poses. The renderer's true foreground is evaluation material; SAM/CNOS
must generate the actual inference masks. Software fixtures remain separately
identified from rendered training examples.

```bash
python3 -m bench.quality_ranker_synthetic --mesh /cloud/data/permitted-object.glb \
  --mesh-units metres --split-manifest /cloud/data/split.json \
  --sequence-id YOUR_FROZEN_SYNTHETIC_SEQUENCE --output-dir /cloud/data/synthetic-object \
  --frames 90 --observationally-ambiguous yes --cloud-execution
```

Declare orientation observability explicitly. `yes` excludes orientation labels
that cannot be resolved from the observations. Set `no` only when justified for
that mesh and rendered view. Freeze the exact GLB SHA-256 and procedural gray
background in this synthetic sequence's inventory before rendering.

The held-out mesh may be supplied at inference. “Unseen object” means unseen by
the trained ranker, not necessarily unseen in a foundation encoder's pretraining.

## Actual candidate extraction

The existing bundle contains video, measured intrinsics, a metric mesh, hashes
and timestamps. New videos use the existing schema-2 physical-time format.
Original authenticated keyboard/mug/bottle schema-1 bundles remain supported.

```bash
python3 -m bench.quality_offline --bundle /cloud/data/train-object \
  --output /cloud/runs/train-object-candidates --device cuda --patch-features
```

This executes `smoke → banks → cnos-bank → segment → pose`, sequentially. The
existing smoke compares two renderers and checks refinement before bank generation.
For new physical-time bundles without an onboarding pose, it uses a virtual view
derived from mesh extent. That smoke consumes no recorded pixels or reference
poses; its rendered foreground is never passed to video segmentation. Each
stage has a 30-minute default limit; change it explicitly after estimating costs.
The existing FoundPose bank uses 798 views; this first setup is not free in GPU
time. Reuse it for subsequent rankers. `--dry-run` prints commands without models.

For a bounded prefix, add `--limit N`. Preserve its original frame identities
and use the same prefix in the frozen development inventory. No evaluation pose
is used to initialize this command: it always selects `complete` mode.

Each pose run saves public states and `candidates.jsonl`, including failed frames,
candidate poses, eligibility, features, selection reasons, source identities,
timing and memory. Existing masks retain their paths and hashes. Stage outputs
and traces require fresh run directories; an interrupted trace is not silently
appended to a different trajectory.

Join reference labels **after** inference, then inspect the audit:

```bash
python3 -m bench.quality_ranker_data prepare --split-manifest /cloud/data/split.json \
  --trace /cloud/data/candidates.jsonl --reference-labels /cloud/data/poses.jsonl \
  --output-dir /cloud/data/ranker-cache
python3 -m bench.quality_ranker_data audit --cache-dir /cloud/data/ranker-cache --training-only
```

For multiple sequences, combine complete per-frame JSONL files in frozen
inventory order; preserve all failures and original identities. The cache keeps
features and labels separate, split-specific and hashed. The audit reports good
candidate coverage before/after gates, legacy choice, all-bad/missing/ambiguous
groups and best candidate errors. If good alternatives are absent, a ranker
cannot repair those cases. Do not automatically switch to correspondence training.
The trainer refuses a cache without eligible good/bad alternatives in both
training and validation; an integrity pass alone does not establish that signal.

## Broader synthetic collection

The first cloud sample produced 28 eligible candidates, all within the frozen
10-degree/2-centimetre criteria, and no mixed-quality groups. It has not been
used to optimize a ranker. The next fixed collection uses eight starting views
per object, 24 frames per view, and a scheduled image-space occluder. Carton17
is training, dinosaur27 is validation, and coffee-pot7 remains reserved. This
is an exploratory synthetic collection; success would not establish real-video
accuracy or generalization across objects.

Reuse the two pinned GLBs already acquired in cloud storage. No download occurs
in this preparation command:

```bash
VISUALIZEIT_CLOUD=1 python3 scripts/cloud_ranker_prepare_multiview.py \
  --assets /cloud/data/existing-assets --output-dir /cloud/data/multiview-v1
```

It freezes 16 complete sequences and the renderer recipe, binding both the
validated split digest and its exact file bytes. For each frozen sequence, use
its matching mesh and sequence ID:

```bash
python3 -m bench.quality_ranker_synthetic \
  --mesh /cloud/data/existing-assets/obj_000017.glb --mesh-units metres \
  --split-manifest /cloud/data/multiview-v1/split.json \
  --recipe /cloud/data/multiview-v1/recipe.json \
  --sequence-id ranker-multiview-hot3d17-v1-yaw0-pitchm25 \
  --output-dir /cloud/data/multiview-v1/yaw0-pitchm25-train \
  --frames 24 --width 640 --height 480 --fps 10 \
  --observationally-ambiguous yes --cloud-execution
```

Recipe-mode output places the three inference files in `bundle/`; labels and
generation provenance stay alongside that directory. Preserve the conservative
raw labels. Review orientation observability using rendered RGB before candidate
extraction, and record any per-frame orientation supervision in a separate
reviewed label sidecar. Do not choose labels after inspecting prediction errors.
Use observed CNOS initialization and the explicit selection-review adapter before
SAM. Run fresh trackers for each sequence, reuse compatible banks, preserve
failures, then join labels to sealed traces using the existing cache preparation
and audit commands. Training still requires mixed eligible good/bad groups in
both training and validation.

Initialization uses the existing detector stage and a separate, explicit review:

```bash
python3 -m bench.quality_runner detect --bundle RAW_BUNDLE --frame 0 \
  --device cuda --output DIAGNOSTICS/detection.npz
python3 -m bench.quality_ranker_initialization propose --bundle RAW_BUNDLE \
  --detector-npz DIAGNOSTICS/detection.npz \
  --proposal DIAGNOSTICS/proposal.json --preview DIAGNOSTICS/preview.png
# Inspect preview.png, then write an approval receipt bound to proposal.json.
python3 -m bench.quality_ranker_initialization finalize --bundle RAW_BUNDLE \
  --detector-npz DIAGNOSTICS/detection.npz \
  --proposal DIAGNOSTICS/proposal.json --preview DIAGNOSTICS/preview.png \
  --review-receipt DIAGNOSTICS/review.json --output-bundle FRESH_INFERENCE_BUNDLE
```

Build compatible CNOS banks before detection. Proposal generation does not
approve a selection. Finalization rejects a missing or mismatched review and
creates only the video, mesh and input manifest in the fresh bundle. A Codex
visual-selection receipt must explicitly identify agent review; it is not an
independent human accuracy annotation.

The fixed first collection has ceilings of 90 GPU minutes, one existing credit,
8 GiB model cache and 1 GiB aggregate new data/results. Monitor these separately:
Python runtime/storage limits cannot prove provider billing has stopped.

## Train, monitor, resume and export

After the audit, this script runs an interruption/resume/export smoke and then a
bounded experiment for each head on identical cached features:

```bash
bash scripts/cloud_ranker_train.sh /cloud/data/ranker-cache /cloud/runs/rankers-v1
```

The small heads can use the cloud CPU after GPU feature extraction. Set
`VISUALIZEIT_RANKER_DEVICE=cpu` for the same smoke, training and export sequence;
the launcher disables mixed precision and records the resolved configurations
in the run directory. The default remains CUDA. Resume with that run's
`config-linear.json` or `config-mlp.json` so the device and precision settings
match the saved checkpoint. This option does not switch or stop the provider's
GPU. CPU head training can run inside the existing GPU VM, whose compute billing
continues until provider shutdown. Check the actual migration quote before
changing machines: this account's CPU migration quoted 1.56 credits, so it was
not used in the one-credit run.

Each head uses candidate-quality plus pairwise supervision. All-bad groups still
contribute quality supervision. Normalization uses training rows only; validation
selects epochs and acceptance/ambiguity thresholds. No test labels are read by
the trainer. Frozen encoders are never loaded by the head-training loop.

Monitor `events.jsonl`, `eda.json`, stage logs and the provider's credit/storage
console. `last.pt` and `best.pt` are bounded atomic checkpoints. Resume in the
same output directory with the same data, code, configuration and runtime:

```bash
python3 -m bench.quality_ranker_train resume --cloud-execution \
  --manifest /cloud/data/ranker-cache/cache_manifest.json \
  --config bench/runtime/ranker-linear.json \
  --checkpoint /cloud/runs/rankers-v1/linear/last.pt \
  --output /cloud/runs/rankers-v1/linear
```

The cloud smoke compares interrupted and uninterrupted optimizer progress and
exports the head to JSON and ONNX. Numerical tolerances are `atol=1e-6`,
`rtol=1e-5`; successful receipts also require identical candidate ordering,
selection and rejection decisions on validation groups. Calibration uses
midpoints between observed scores where possible to avoid cutoff ties. The
production schema-v1 smoke and calibrated-decision checks have not yet executed
on a cloud host. The separate engineering schema-v2 smoke passed the numerical
and disabled-decision checks recorded above.

```bash
python3 -m bench.quality_ranker_train export --cloud-execution \
  --manifest /cloud/data/ranker-cache/cache_manifest.json \
  --checkpoint /cloud/runs/rankers-v1/linear/best.pt \
  --output /cloud/runs/rankers-v1/linear/model.json
```

Download only `model.json`, `model.onnx`, their export receipt and reports. Keep
large resume checkpoints and datasets in cloud storage. The JSON package
contains weights, exact feature names/version, normalization, calibrated
thresholds and provenance. Inference loads it with NumPy and does not import the
trainer. ONNX describes the small ranking head; it does not make the entire
vision stack Android-ready.

## Compare and replay

First compare all selectors on identical cached candidate sets using development
or validation data. The evaluator reports missing independent surface evidence
as unavailable, not zero error. Uncertainty is grouped by object/sequence.

```bash
python3 -m bench.quality_ranker_evaluate --cache-dir /cloud/data/ranker-cache \
  --split validation --learned-checkpoint /cloud/runs/rankers-v1/linear/model.json \
  --output /cloud/runs/rankers-v1/linear/comparison.json
```

Repeat for the MLP. After selecting settings on validation, replay the selected
checkpoint through the complete tracker using the same cached masks:

```bash
python3 -m bench.quality_offline --bundle /cloud/data/validation-object \
  --output /cloud/runs/validation-learned --stages pose \
  --masks /cloud/runs/validation-object-candidates/masks \
  --ranker learned --checkpoint /cloud/runs/rankers-v1/linear/model.json
```

Run analogous `--ranker legacy` and `--ranker deterministic` comparisons. Changed
selections change subsequent seeds: cached reranking alone cannot establish a
closed-loop tracking improvement. Use the existing independent evaluator for
surface attachment, temporal slipping and occlusion recovery. Keep dataset
reference agreement distinct from independent image/attachment labels. Open the
untouched test split only after model and thresholds are frozen.

When done, stop the GPU in the provider console and verify its stopped state.
Ending Python or exhausting this script's runtime budget does **not** stop cloud
billing. Persistent volumes can remain chargeable after GPU shutdown.

## Additional CPU replay evidence — October 14, 2026

The same free AWS 4CPU Studio completed an additional frozen dinosaur validation
clip, starting at yaw 180° and pitch +25°. No new weights or datasets were
downloaded. Inference used the observed SAM masks, the supplied metric mesh and
camera calibration; reference labels were attached after sealing predictions.
The validation renderer used the previously checked explicit no-MSAA diagnostic
configuration. This is separate from the unchanged default-renderer baseline.

| Stage | Recorded scope | Wall time | Peak process memory |
| --- | --- | ---: | ---: |
| GoTrack model smoke | Virtual geometry control, CPU | 35.12 s | 2.48 GiB |
| SAM propagation | First 16 recorded frames | 329.55 s | 2.94 GiB |
| Legacy pose replay | Same 16 frames, cached observed masks | 368.59 s | 2.50 GiB |

Stages ran sequentially in separate processes. All 16 frames had available mask
outputs; mask accuracy was not independently measured. Frame 0 remained
suppressed pending recovery confirmation. Frames 1–15 were displayed as tracking,
but the frozen synthetic reference evaluator found unacceptable translation on
frames 9, 10, 14 and 15: respectively 8.84, 2.07, 2.83 and 2.38 cm. Each of these
frames had only one candidate. A different ordering cannot supply a correct pose
on those frames. Their angular labels remain ambiguous and are excluded from
rotation-error claims. The clip did not lose tracking, so it does not demonstrate
recovery from complete occlusion or independently verified surface attachment.

The resulting cache contains 112 actual candidates across a 384-frame frozen
inventory; 313 inventory frames remain explicitly unprocessed. Cache integrity
passed. Legacy and deterministic selectors were compared on the same cached
sets: both selected 49 acceptable and 2 unacceptable training poses, and 15
acceptable and 4 unacceptable validation poses. These are synthetic reference
comparisons on two development identities, not evidence of general improvement.
No eligible frame contains both an acceptable and an unacceptable candidate, so
the existing substantive ranking-training gate remains unsatisfied.

The previously trained engineering checkpoint was also measured using NumPy
against 20 actual cached candidates in four validation groups: 30 warm-up passes,
200 measured passes, 800 observations; median 0.08064 ms, p95 0.108623 ms.
This measures the ranking head only, excludes feature extraction, and was taken
while the separate CPU SAM process was active. The checkpoint still disables
pose selection. It was not retrained on the extended cache.

Small reports were downloaded and verified locally in `.cache/cpu-validation-v1`.
The 40,257-byte report archive has SHA-256
`fc81dd0dd1747ad38fbd1f44b289e46e927d8db8f26945d37d2f1cb74c1986c5`.
Data, masks, feature caches and large checkpoints remain in cloud storage.
Provider shutdown was verified: Studio sleeping, no machine, 3.94 displayed
credits before and after this free CPU work. No paid resource was enabled.

## Confidence rejection research workflow

Cache v6 has correct and incorrect legacy choices across frames, but no eligible
good/bad alternatives within a frame. A separate research mode uses that quality
supervision to veto the legacy-selected pose. It preserves the legacy ordering,
scores, ambiguity decisions and downstream tracking checks. It cannot select a
replacement pose. The ordinary substantive ranking gate remains unchanged.

The source includes explicit `confidence-train`, `confidence-resume`,
`confidence-export` and `confidence-smoke` commands, plus a train/validation-only
evaluator. The integrated local analytical suite passed 157 checks, and independent source
review approved the final version. Cloud training, exact interruption/resume and
JSON/ONNX export checks now pass for both architectures. The earlier engineering
checkpoint is separate and remains selection-disabled.

### Portable comparison command

The repository command groups the ten stages used by the cloud experiment:
smoke, training, export, training evaluation and validation evaluation, first
linear and then MLP 64/32. It uses the existing fixed CPU configurations and
subprocess supervisor. Paths are supplied by the caller; no provider account or
provisioning API is involved.

From the repository in your approved Linux cloud environment:

```bash
cache=/cloud/path/to/candidate-cache
new_run=/cloud/path/to/a-new-run
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -B -m bench.quality_confidence_workflow \
  --manifest "$cache/cache_manifest.json" --output "$new_run" --cloud-execution
```

Use a new output directory. The shared deadline is 880 seconds and new outputs
are limited to 256 MiB. `confidence-workflow.json` records commands, cache/config/
source hashes, completed stages and failure details. Successful completion also
requires the trainer completion, export-parity and evaluation artifacts to pass
validation. It does not select a production model or schedule a video replay.

Preview the commands locally or in the cloud using the same arguments plus
`--dry-run`; that mode reads no training cache and creates no output. Monitor a
live cloud run through the aggregate receipt, stage logs and trainer events:

```bash
tail -f "$new_run/train-linear.log"
tail -f "$new_run/mlp/events.jsonl"
```

If interrupted, retain its output and use the existing `confidence-resume`,
`confidence-export` and evaluation commands below with the same manifest/config.
The comparison command refuses to overwrite an existing run. Its original
aggregate remains incomplete; a separate successful model completion receipt
and export verification are required after manual resume. Stop compute through
the provider console and verify shutdown after finishing.

This repository command was verified on the existing free cloud CPU: all ten
stages completed in 26.67 seconds, with sampled peak process memory of 529,207,296
bytes. Both exported inference payloads exactly matched the earlier verified
linear and MLP models. Resume, JSON/ONNX parity, completion hashes and evaluation
source bindings passed through the command's actual artifact validation. This
repeat checks workflow reproducibility; it adds no new accuracy evidence.

The verified run is at
`/teamspace/studios/this_studio/visualizeit-cloud-v9/cloud-runs/portable-confidence-v1`.
Its 109,419-byte compact download has SHA-256
`cef4870b9f1e53cabd323385a4dec8ad23858f454a94216026b8566590f82531`.
All 23 included weights/report files were hash checked before extraction to
`.cache/portable-workflow-v1`, and both JSON heads loaded through the local NumPy
factory without Torch or local training. The provider verified sleeping again,
with no allocated machine and the same 3.94 displayed credits.

For the verified Lightning workspace, start the existing free CPU Studio and
use a fresh run name, for example:

```bash
cd /teamspace/studios/this_studio/visualizeit-cloud-v9
cache=/teamspace/studios/this_studio/visualizeit-cloud-v3/cloud-data/multiview-v1/candidate-cache-bounded-v6
new_run=/teamspace/studios/this_studio/visualizeit-cloud-v9/cloud-runs/portable-confidence-v2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -B -m bench.quality_confidence_workflow \
  --manifest "$cache/cache_manifest.json" --output "$new_run" --cloud-execution
```

Run these commands inside the approved cloud workspace. The following paths use
the existing frozen cloud cache and a fresh run directory; do not execute training
on the laptop. No vision encoder is loaded while optimizing these cached features.

```bash
export VISUALIZEIT_CLOUD=1
cache=/teamspace/studios/this_studio/visualizeit-cloud-v3/cloud-data/multiview-v1/candidate-cache-bounded-v6
run=/teamspace/studios/this_studio/visualizeit-cloud-v8/cloud-runs/confidence-v1

python3 -B -m bench.quality_ranker_train confidence-smoke \
  --manifest "$cache/cache_manifest.json" --output "$run/smoke-linear" \
  --architecture linear --cloud-execution

python3 -B -m bench.quality_ranker_train confidence-train \
  --manifest "$cache/cache_manifest.json" --output "$run/linear" \
  --config bench/runtime/ranker-confidence-linear-cpu.json --cloud-execution

python3 -B -m bench.quality_ranker_train confidence-export \
  --manifest "$cache/cache_manifest.json" --checkpoint "$run/linear/best.pt" \
  --output "$run/linear/model.json" --cloud-execution

python3 -B -m bench.quality_confidence_evaluate \
  --cache-dir "$cache" --split validation --checkpoint "$run/linear/model.json" \
  --output "$run/linear/validation.json"
```

Repeat the train/export/evaluation commands using
`ranker-confidence-mlp-cpu.json` and the `mlp` output directory. The predeclared
comparison runs linear first and the 64/32 MLP second with seed 42, 40 epochs,
batch size 8, accumulation 4, unchanged loss weights, CPU execution and a
300-second limit per architecture. Retain both outcomes and learning curves.
Use the external process watchdog as well as these cooperative trainer limits.

Resume an interrupted run with its original configuration and output directory:

```bash
python3 -B -m bench.quality_ranker_train confidence-resume \
  --manifest "$cache/cache_manifest.json" --checkpoint "$run/linear/last.pt" \
  --output "$run/linear" --config bench/runtime/ranker-confidence-linear-cpu.json \
  --cloud-execution
```

Monitor `events.jsonl` and `eda.json` in each run directory. Atomic `last.pt` and
validation-selected `best.pt` stay in cloud storage. A hash-bound `completion.json`
records successful completion or early stopping; a runtime-limited run remains
incomplete and must be resumed. Export accepts the selected `best.pt` only after
verifying that receipt and the final checkpoint. A completed run cannot be resumed.
Export produces compact
JSON/ONNX artifacts with schema 3, purpose, feature normalization, calibrated
threshold, hashes and development-fitting provenance. Ordinary `--ranker learned`
rejects this schema. Explicit research replay uses the existing offline pipeline:

```bash
python3 -B -m bench.quality_offline --bundle /cloud/data/validation-object \
  --output /cloud/runs/confidence-replay-v1 --stages pose --device cpu \
  --masks /cloud/runs/validation-object-masks --ranker confidence-research \
  --checkpoint "$run/linear/model.json"
```

That replay requires the existing device/renderer smoke and template bank for the
bundle. A changed rejection changes later tracking history, so cached evaluation
alone does not establish a tracking improvement. Validation labels were already
inspected and are used for checkpoint selection and calibration: results are
development diagnostics. Ambiguous angular labels remain excluded; the target
is quality in the available labeled dimensions. The reserved test object remains
untouched. No production promotion, ordering gain or physical attachment claim
follows from this experiment.

The first confidence cloud smoke completed two-epoch training and exact resume
checks, then exposed an export-verification bug: probability diagnostics were
compared exactly alongside decisions. All 192 frame decisions matched, with
maximum probability difference 9.384331089989928e-9. The corrected source keeps
probability tolerances at rtol 1e-5 / atol 1e-6 and compares every decision field
exactly; crossing the acceptance boundary still fails. Source packet v5 and its
failed run remain preserved. The reviewed correction uses a fresh v6 workspace
with the same frozen data and settings. Complete calibrated exports and model
comparison remain pending until that actual cloud run succeeds.

The corrected v6 cloud smoke passed training, exact interrupted/resumed state
checks, JSON export and ONNX probability/decision verification. Its final JSON
receipt then rejected the initial unobserved best-loss infinity from the partial
checkpoint. The reviewed v7 correction reports only epoch/cursor/step in that
partial summary; the full binary resume state and training settings are unchanged.
Both prior trials remain preserved. The full linear/MLP comparison is pending.

The v7 cloud comparison completed: linear stopped at 7 epochs/14 steps; the
64/32 MLP completed 40 epochs/80 steps. Both passed two-epoch resume and export
checks. On 19 labeled validation legacy choices, linear retained 12 acceptable
and 3 unacceptable poses; MLP retained all 15 acceptable and 2 unacceptable poses.
These are inspected development results. The cache uses the explicit placeholder
reason not_processed_within_approved_compute_budget; source v8 recognizes it
alongside not_processed so reports count 71 actual traces and 313 unprocessed
frames in the 384-frame inventory. Prior reports with incorrect inventory counts
remain preserved; model settings and scored-choice metrics were not changed.

### Verified final cloud run

The final source-v8 run completed all ten smoke, training, export and evaluation
stages on the existing free AWS 4CPU Studio. The complete head comparison took
19.90 seconds and peaked at 504 MiB process memory; source and new run artifacts
occupied 8.27 MB. It reused the frozen cached features, with no new encoder or
dataset download. This CPU head training occurred in the cloud, not on the laptop.

| Model | Parameters | Epochs / optimizer steps | Good retained / rejected | Bad accepted / rejected | Validation balanced accuracy | JSON / ONNX bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 33 | 7 / 14, early stopping | 12 / 3 | 3 / 1 | 0.525 | 10,933 / 1,328 |
| MLP 64/32 | 4,225 | 40 / 80 | 15 / 0 | 2 / 2 | 0.750 | 145,138 / 18,715 |

Both two-epoch smoke runs finished four optimizer steps, verified exact resumed
training state, and exported the same complete veto decisions. Maximum absolute
probability differences versus PyTorch were 9.39e-9 / 3.85e-8 for linear
JSON / ONNX, and 3.20e-8 / 5.25e-7 for MLP JSON / ONNX. The numerical tolerance
remains rtol 1e-5 / atol 1e-6. Decision fields are compared exactly.

The predeclared development selection chose MLP by validation balanced accuracy,
then validation BCE, with a linear tie preference. Its fitted temperature is 0.5,
bias -0.5, and acceptance threshold 0.42703312555300144. This fitting uses the
already inspected validation set and does not establish generalization. In the
19 validation choices it suppressed two of four unacceptable poses and retained
all 15 acceptable poses. It remains an opt-in confidence veto, not a replacement
candidate ordering model or the default pipeline.

Corrected coverage is 52 actual training traces and 19 validation traces: 71 of
384 frozen frames, with 313 explicitly unprocessed frames. The cache contains
112 candidates but zero eligible mixed-quality groups and zero ordering pairs.
Therefore these results do not demonstrate correction of 180-degree rotations.
The reserved test object has not been read or evaluated.

Both exported JSON checkpoints load through the actual inference factory without
PyTorch, reference poses or label reads. Across 19 cached validation groups, 30
warm-up repeats and 100 measured repeats, median ranking-only latency was 0.00287
ms legacy, 0.03221 ms deterministic, 0.01002 ms linear, and 0.01432 ms MLP. These
measurements exclude frozen-feature extraction and do not demonstrate real-time
video performance.

### Completed causal video replay

The selected MLP checkpoint completed the actual 16-frame replay through
`quality_runner`, using the same cached observed SAM masks as legacy. Predictions
and candidate traces were sealed before the frozen reference labels were read.
Both result files retain the full 24-frame inventory: frames 16–23 are explicitly
unprocessed outside the requested prefix. All 16 requested frames are counted,
including suppressed and failed frames.

| Development clip metric | Legacy | Confidence MLP |
| --- | ---: | ---: |
| Available masks | 16/16 | 16/16 |
| Visible overlays | 15/16 | 13/16 |
| Visible poses exceeding the 2 cm reference translation limit | 4 | 1 |
| Mean translation error among visible poses | 16.89 mm | 8.82 mm |
| p95 translation error among visible poses | 46.36 mm | 16.73 mm |
| Stale/unmasked/state-invalid visible overlays | 0 | 0 |
| Nonambiguous angular measurements | 0 | 0 |
| Pose-stage wall time, one CPU run | 368.59 s | 303.81 s |
| Sampled peak process memory | 2.50 GiB | 2.36 GiB |

The confidence head vetoed frame 9. Frame 10 stayed hidden pending a second
validated pose, and tracking resumed at frame 11. That causal change altered
later candidates; frame 13 still exceeded the translation limit at 22.01 mm.
Legacy displayed unacceptable reference agreement on frames 9, 10, 14 and 15.
This is a tradeoff on one inspected synthetic development clip, not an
independent accuracy or speed improvement claim. Visible-only error averages
use different frame sets, so availability and failures must accompany them.
Partial occlusion and recovery after a veto do not demonstrate recovery from
complete object disappearance. Angular ambiguity prevents a rotation claim;
independent surface attachment and temporal slipping remain unmeasured.

To reproduce this specific opt-in diagnostic, use a fresh output and trace path:

```bash
python3 -B -m bench.quality_runner pose \
  --bundle "$bundle" --masks "$observed_masks" --output "$new_replay/pose.json" \
  --device cpu --limit 16 --mode complete --ranker confidence-research \
  --ranker-checkpoint "$run/mlp/model.json" \
  --candidate-trace "$new_replay/candidates.jsonl" --ranker-patch-features \
  --disable-multisampling
```

Use the existing preflight/template-bank checks and external timeout/storage
watchdog. The no-MSAA flag reproduces this checked diagnostic; it does not change
the default-renderer baseline. The legacy mode remains the default.

The compact artifact archive is 169,206 bytes, SHA-256
`999bcda46cbaec2b5117ed972861a93b25843591762866b4b5268a1ef3057012`.
Its 35 files were checked against a strict whitelist and per-file hashes before
extraction to `.cache/confidence-research-v1`. Both JSON heads loaded locally
through the NumPy inference factory and round-tripped exactly, without importing
PyTorch or training locally. Large datasets, feature caches, raw traces, frozen
encoder weights and resumable binary checkpoints remain in cloud storage.

The provider dashboard verified Studio sleeping, no allocated machine, and 3.94
displayed credits before and after this free CPU work. No paid resource or storage
add-on was enabled. There are no continuing cloud processes from this run.
## Prospective orientation data: existing meshes

`scripts/cloud_ranker_prepare_orientation.py` adds a separate, opt-in study.
It reuses the acquired HOT3D carton17 training mesh and dinosaur27 validation
mesh. The original broader recipe and its labels remain unchanged.

The new recipe fixes four sequences of 12 frames at 640×480 and 10 Hz. Each
object uses paired views `(45°, -15°)` and `(225°, +15°)`, with a 20° yaw sweep
and no occluder. Preparation authenticates both metric meshes, freezes complete
object/sequence assignments, and binds the generator and preparation source
hashes. Rendering validates every scheduled transform and vertex projection.

Run in the existing Linux cloud environment, with the two pinned metric GLBs
already present in `ASSETS` and a fresh `DATA` destination:

```bash
export VISUALIZEIT_CLOUD=1
python3 scripts/cloud_ranker_prepare_orientation.py \
  --assets "$ASSETS" --output-dir "$DATA"
python3 -m bench.quality_ranker_synthetic \
  --mesh "$ASSETS/obj_000017.glb" \
  --split-manifest "$DATA/split.json" --recipe "$DATA/recipe.json" \
  --sequence-id ranker-orientation-hot3d17-v1-yaw45-pitchm15 \
  --output-dir "$DATA/ranker-orientation-hot3d17-v1-yaw45-pitchm15" \
  --frames 12 --width 640 --height 480 --fps 10 \
  --mesh-units metres --observationally-ambiguous yes --cloud-execution
```

Render all four sequences in the recipe's frozen order. The inference bundle
contains exactly `input.json`, `object.glb` and `synthetic.mp4`; raw pose labels
remain outside it and conservatively mark every orientation ambiguous.
Review all rendered RGB frames before candidate inference, record visible
orientation cues and uncertainty, and freeze a separate reviewed labels sidecar.
Use the existing observed-mask initialization review, SAM, FoundPose/GoTrack,
cache preparation and training commands afterward. Never feed the labels into
those inference stages.

Actual free AWS4CPU preparation/rendering completed on 2026-10-14: four clips,
48 frames, 22.43 seconds for the watched render batch, peak process-tree RSS
753,307,648 bytes. All 49 analytical/software integration checks passed on Linux
with no skips. These are preparation measurements, not tracking FPS or accuracy.
The blind image review identified orientation cues in 36 frames; the other
12 carton frames retain ambiguity because their visible artwork repeats.
Candidate collection and ordinary ranking training on this new batch remain
pending.

The training decision uses the existing quality thresholds and requires actual
eligible, orientation-supervised good/bad angular alternatives in both splits.
Report both translation errors. A wrong rotation of an off-origin mesh may also
have a wrong origin translation, so that joint pose negative remains relevant.
Report pairs with both translations within 20 mm separately when assessing
isolated rotation discrimination. This small, inspected synthetic development
sample cannot establish independent real-world accuracy or generalization.


#### Observed orientation-data collection result

The orientation-v1 free CPU session rendered four clips (48 RGB frames), passed
all 49 Linux software checks, and produced four CNOS mask proposals. The agent
visually checked all four observed-mask overlays and finalized their existing
initialization bundles. A private collection adapter initially rejected matching
manifest hashes because it compared uppercase and lowercase hex strings. That
failed attempt is retained; a four-case analytical check covered the narrow fix.

Within the original runtime limit, the first clip completed six pose frames and
recorded ten eligible candidates. All ten were already good under the frozen
synthetic labels; the legacy selector chose a good candidate on all six frames.
The validation clip produced no pose traces before the cutoff. The other two
clips were not collected. All 48 planned frames remain in the audit, including
42 explicit unprocessed/empty rows. These are runtime and coverage limitations,
not evidence of model accuracy on the unprocessed frames.

The train/validation audit passed integrity checks but found zero eligible mixed
good/bad candidate groups. No new ordinary ranker was trained. Frozen SAM,
FoundPose, GoTrack and their encoders were used only for inference. The earlier
confidence-veto heads remain separate research artifacts. The cloud machine was
verified sleeping afterward, with 3.94 displayed credits and no allocated machine.

#### Single-view feasibility input

The additive `scripts/cloud_ranker_prepare_single_view.py` freezes one pinned
HOT3D17 training initialization at yaw 75°, pitch −15°, with one 640×480 frame.
It authenticates the existing cloud GLB, split, source files and recipe before
rendering. It downloads nothing and starts no models or training. The normal
synthetic render CLI accepts its recipe with `--frames 1`; its raw label remains
ambiguous until a separate RGB review. Existing broader/orientation recipes are
preserved; all 24 prior orientation poses were independently checked unchanged.

The actual free CPU render completed in 9.06 seconds, with approximately 660 MiB
peak process-tree RSS. Its image did not provide a defensible globally unique
yaw label: repeated artwork dominated the view and the adjacent panel was too
oblique. The agent retained ambiguity and stopped before candidate inference.
This probe created no new training checkpoint. The machine was verified sleeping
with no allocation afterward; the displayed balance remained 3.94 credits.

Local verification for the final source ran 53 focused analytical checks:
52 passed and one Windows symlink privilege check was skipped. Independent
review approved the exact final source after correcting the new recipe's
review-required metadata flag. No Torch or local model execution was involved.
# Fixed normalized synthetic views

The synthetic renderer also accepts `--view-mode normalized24-v1 --view-index N`
for any supplied metric mesh within the existing renderer's clipping domain.
Freeze 24 separate one-frame sequences per object before rendering. Indices
0–23 cover three elevation rings (−60°, 0°, 60°), each with azimuths
0°, 45°, …, 315°. Camera distance follows mesh radius; mesh coordinates and units
stay unchanged. Render the complete schedule before inspecting candidate quality.

```bash
python -B -m bench.quality_ranker_synthetic \
  --mesh /cloud/assets/object.glb --split-manifest /cloud/data/split.json \
  --sequence-id object-normalized24-view00 --output-dir /cloud/data/view00 \
  --frames 1 --width 640 --height 480 --fps 10 --mesh-units metres \
  --observationally-ambiguous yes --cloud-execution \
  --view-mode normalized24-v1 --view-index 0
```

Each output contains an isolated `bundle/` with RGB video, mesh and input
manifest. Reference poses and a generation receipt are sibling files. The receipt
binds the selected view, complete schedule, normalization, source, split and
output hashes. Raw labels remain ambiguous until a separate RGB review;
geometric containment does not prove that orientation is observable. Each view
is an independent sequence at timestamp zero, so the schedule must not be used
as a continuous tracking trajectory. Omitting `--view-mode` retains legacy
rendering and existing recipe behavior.



### Normalized-view cloud smoke evidence (2026-10-14)

The optional normalized24 mode rendered all 48 prospectively selected views of
HOT3D training objects 31 (mouse) and 33 (DVD remote) on the existing free CPU
studio. Rendering took 153.108 seconds; sampled peak process-tree memory was
776,198,688 bytes. Source, mesh, split and compact report hashes were checked.
RGB-only review found visible orientation cues in 22 views and retained ambiguity
for 26. Raw renderer labels remain unchanged. No candidate extraction or optimizer
ran in this render smoke test, and these results establish neither tracking
accuracy nor FPS.
The earlier confidence-rejection checkpoints remain distinct from the unfinished
ordinary pose-ordering ranker.

### Observed mouse replay (2026-10-15)

The existing frozen models prepared 798 FoundPose templates and 57 CNOS
descriptors for object 31. Template preparation took 724.024 seconds on Free
4×CPU; all encoder checkpoint hashes remained unchanged.

For normalized view 00, CNOS detection took 26.001 seconds. Its observed-mask
preview and interior point were inspected before finalizing the inference
bundle. SAM segmentation then completed in 14.152 seconds and the legacy pose
stage in 52.484 seconds, with spatial patch features enabled. These are cold
single-frame CPU stage timings, not sustained tracking throughput.

Of five recorded candidate attempts, two returned eligible poses and three
returned none. After inference outputs were sealed, a separate label derivative
applied the prior RGB-only orientation review. The two poses had rotation errors
167.363° and 178.004°, with translation errors 89.449 mm and 19.415 mm against the
synthetic renderer reference. Neither satisfied the frozen quality policy. The
legacy selector rejected the competing hypotheses as ambiguous and suppressed
the overlay. This candidate set cannot supply a correct alternative for ranking.
It remains useful negative-quality evidence; it is not independent physical
surface-attachment evidence or evidence of a trained ranker's accuracy.

The cache integrity audit passed with the other 47 frozen views explicitly
marked unprocessed. No optimizer ran. The original failed launch is preserved:
`segment --limit 1` is invalid when the physical video contains only one scored
frame. Omit `--limit` for the complete single-frame run:

```bash
python -B -m bench.quality_runner segment \
  --bundle /cloud/reviewed/bundle --output /cloud/run/masks --device cpu
python -B -m bench.quality_runner pose \
  --bundle /cloud/reviewed/bundle --masks /cloud/run/masks \
  --output /cloud/run/poses.json --device cpu --ranker legacy \
  --candidate-trace /cloud/run/candidates.jsonl --ranker-patch-features
```

### Eight-view mouse candidate audit (2026-10-15)

The next seven preselected mouse views (01–07) completed observed-mask review,
SAM segmentation and legacy FoundPose/GoTrack inference. Detection/proposals took
182.925 seconds; the seven sequential finalized replays took 846.179 seconds on
Free 4×CPU. Sampled peak process-tree memory was 3,494,748,160 bytes during
detection and 2,768,019,456 bytes during the replay stages. The same frozen mesh
template banks were reused. All seven single-frame outputs stayed in recovery
with rendering suppressed while awaiting a second validated pose; this experiment
does not measure temporal recovery or overlay availability on a video sequence.

Together with view 00, the audit contains 40 attempts across eight isolated
synthetic frames: 34 accurate poses under the frozen policy, three inaccurate
poses and three attempts returning no pose. The accurate poses have rotation
errors 0.277–1.998° and translation errors 0.205–1.692 mm against renderer
references. One inaccurate pose in view 04 was already ineligible; view 00 has
only inaccurate returned poses. Consequently no frame contains both an eligible
accurate pose and an eligible wrong-rotation alternative. These data supply
quality-supervision examples but do not satisfy the current experimental gate
for learning rotation-based candidate ordering. That gate is a protocol choice,
not a universal mathematical requirement for training a quality classifier.

The integrity audit passed, the other 40 frozen views remain explicitly
unprocessed, and all reference joins occurred after inference sealing. No
optimizer ran, no held-out test data were read, and no encoder weights changed.
Compact reports were downloaded and verified; large artifacts remain in cloud
storage. The studio was verified sleeping with no allocated machine and a
displayed balance of 3.94 credits. These measurements establish neither improved
ranker accuracy nor independent physical surface attachment accuracy.

### Rotation-bearing training example (2026-10-15)

Four additional preselected mouse views (16–19) completed the same observed-mask
and frozen-model pipeline. Detection/proposals took 80.679 seconds and sequential
finalization, segmentation and pose replays took 442.198 seconds on Free 4×CPU.
Sampled peak process-tree memory was 3,490,439,168 bytes during detection and
2,769,076,224 bytes during replay. No optimizer ran.

The twelve-view audit now contains 60 attempts: 53 accurate poses, four inaccurate
poses and three no-return attempts. In view 18, four eligible accurate candidates
have rotation errors 0.661–1.529° and translation errors 1.253–2.124 mm. An
eligible inaccurate alternative has a 61.919° rotation error and 19.521 mm
translation error. This supplies four ordering pairs from **one frame**, sharing
the same inaccurate candidate; it is not four independent examples. All five
translations satisfy the 20 mm diagnostic bound. The frozen quality thresholds,
inference gates, raw labels and model weights were unchanged.

This establishes a usable training comparison under the current protocol. A
separate validation comparison and calibrated export remain necessary before an
ordinary ordering ranker can be trained and assessed. The first originally
planned six-frame dinosaur validation prefix completed with ten accurate
candidates and no inaccurate candidate. Its result is retained in the composite
audit; it is insufficient for rejection calibration. The second preplanned view
completed separately in 259.813 seconds, compared with 258.489 seconds for the
first prefix. Both contain six actual frames and ten candidates. Across the two
prefixes, all 20 candidates satisfy the quality policy; rotation errors span
1.158–7.726°. Sampled peak process-tree memory was 3,154,182,144 and 3,166,601,216
bytes respectively.

The combined cache preserves all 96 original inventory frames and their original
object/sequence assignments. It contains 70 training attempts (63 accurate,
four inaccurate, three no-return) and 20 validation attempts (all accurate).
Integrity checks pass. The training comparison is now available, but the
ordinary training/calibration gate remains unsatisfied because validation has no
inaccurate alternatives. No optimizer ran or new checkpoint was produced. The
cloud studio was verified sleeping afterward with no allocated machine and a
displayed balance of 3.94 credits. These remain synthetic development
measurements, with no independent physical attachment or untouched-test accuracy
claim.
