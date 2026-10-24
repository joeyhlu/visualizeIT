# Experimental model adaptation in a cloud workspace

This adds two opt-in training routes. On October 15, the GoTrack correspondence
head completed a real two-step cloud smoke, interruption/resume, export and
fresh-network loading with exact cached-token and RGB parity. See the
[measured result and limitations](gotrack-adaptation-results-2026-10-15.md).
SAM fine-tuning has not run. Existing frozen SAM/GoTrack inference and
pose-ranker training remain the defaults. The smoke does not establish pose
accuracy, generalization, live-tracking or mobile readiness.

| Route | Reused implementation | Trainable parameters | Required data |
| --- | --- | --- | --- |
| Correspondence head | Released GoTrack `DPTHead`, initialized from `models.1.pose_head.*` | All DPT head parameters; DINOv2 and transformer decoder stay frozen | Actual inference crops and independent or explicitly synthetic training correspondence labels |
| Optional segmentation | Released SAM2 `training.trainer.Trainer` and `SAM2Train` | `sam_mask_decoder.*` only | Independently annotated image/mask sequences and reviewed segmentation-bottleneck evidence |

The cached [GoTrack release](https://github.com/facebookresearch/gotrack/tree/68f76055755f2a4a8967e13ece834f975f008bdf)
provides inference components and loss utilities, but no released end-to-end
training command or original training-data recipe. The correspondence route is
a new head-adaptation recipe using those components. It does not reproduce the
original GoTrack training procedure or fine-tune the whole network. GoTrack's
CC BY-NC 4.0 license continues to apply to its code and weights.

The optional route uses the
[released SAM2 trainer](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4/training).
Its upstream example assumes large GPUs. CPU execution is supported by that
trainer's device interface, but runtime, memory fit and convergence of this
adapter have not been established on the currently available cloud CPU host.

## Runtime and boundaries

### Three-crop CPU systems smoke

`quality_adaptation_smoke` wires actual estimated FoundPose seeds, the actual
GoTrack refiner's retained iteration 0 callback, frozen DINO/decoder extraction,
two DPT optimization steps, interruption/resume, strict export loading, and
fresh full-network RGB inference. It requires a parent-reviewed input plan with
the fixed frame 0 views: TRAIN object 17 yaw 0/pitch -25 and +25; VAL object 27
yaw 0/pitch -25. TEST object 7 stays unavailable and unopened. Two training
crops and one validation object cannot establish model quality or a useful
independent confidence interval; loss improvement is not a smoke pass condition.

The historical query videos used the legacy OpenGL raster convention. Their
integer-center native camera has principal point `cx-.5, cy-.5`. The smoke
applies this known camera correction in memory **before** estimated FoundPose
and refinement, while authenticating unchanged video, mesh, input manifest and
cached automatic masks. The original and effective camera matrices are recorded.
Templates and separate synthetic query labels use the existing integer-center
renderer. A manufactured rotated/scaled camera roundtrip checks the historical
raster correction; actual returned crop cameras are checked against retained
K/C sampling maps. This is an explicitly versioned research crop recipe.
The legacy PnP solver's half-center behavior remains a limitation for a future
pose replay; this smoke uses the pre-solver first crop and makes no pose claim.

Retained RGB stays exact float32 `[0,1]`; rounding interpolated/rendered RGB to
uint8 would alter frozen features. Existing uint8 extraction remains supported.
The feature recipe records the encoding version and captured inference output
must match extracted frozen output before the pre-label seal is written.
Float crop hashes are framed by encoding, while exactly byte-representable
float crops retain the uint8 identity so encoding changes cannot bypass split
leakage checks. Available rows always need real crop hashes and extractor
receipts. An unavailable row may use `query_rgb_sha256: null` only with a
nonempty failure reason and `unavailable_provenance`; test hashes are never
invented or obtained by opening test pixels.

The parent can convert its authenticated observed-input metadata without opening
cloud files or label payloads:

```python
from bench.quality_adaptation_data import read_json
from bench.quality_adaptation_smoke import plan_from_receipt
plan, label_plan = plan_from_receipt(read_json(parent_receipt_path))
```

Serialize both plans before capture. Each input, generation receipt, mask index,
mask and FoundPose bank has an exact absolute path/hash/byte pin. The separate
label plan pins the original broader synthetic recipe and split metadata; these
are parsed only after crop/frozen-feature sealing. Query generation receipt,
mesh, video, recipe, sequence and original split must all agree. Synthetic depth
and support come from an independent query render at the retained crop camera,
not the estimated template silhouette. Every requested failed row is retained.

```bash
export VISUALIZEIT_CLOUD=1
bash scripts/cloud_adaptation_smoke.sh /cloud/plan.json /cloud/label-plan.json \
  bench/runtime/adaptation-smoke-cpu.json /cloud/private-adaptation-run
```

The output must be outside the 256 MiB source directory. The launcher runs fresh
processes sequentially under one 1,800-second watchdog and a private 1.5 GiB
artifact limit. The first training invocation uses `--stop-after-steps 1` and
must return incomplete (exit 3). This is an invocation stop, so unchanged
manifest/config can resume to step 2 under the original total budget. The
checkpoint cursor, exact Adam-state digest and row order are checked for
repeat/skip. A fresh pinned full network strictly loads the export, preserves
backbone/decoder tensor hashes, and compares attached/standalone head outputs
on cached VAL tokens and actual RGB inference. A bad export digest must fail
before mutating the loaded head. Paired endpoint/visibility metrics use the
same fixed support; availability, train loss/gradient norm, times, memory and
file bytes are reported without an improvement requirement.

If a reviewed metadata repair is needed after extraction, keep the original
failed label directory and sealed feature recipe/index unchanged. The explicit
compatibility entry in the source lock admits only the original `eb6ece...`
extraction when every feature dependency and producer hash remains unchanged;
only the label/checkpoint metadata adapters may differ. Historical schema1
generation receipts use `kind: quality-ranker-synthetic-generation-v1` and may
lack `recipe_kind`; the separately authenticated recipe kind remains mandatory.
Historical `validation` maps explicitly to the smoke dataset's `val` name.

```bash
python -m bench.quality_adaptation_smoke reuse-features \
  --root /cloud/private-adaptation-run/capture \
  --features /cloud/private-adaptation-run/features \
  --output /cloud/private-adaptation-run/features-v2 --cloud-execution
python -m bench.quality_adaptation_smoke labels \
  --plan /cloud/plan.json --label-plan /cloud/label-plan.json \
  --root /cloud/private-adaptation-run/capture \
  --features /cloud/private-adaptation-run/features-v2 --cloud-execution
```

Continue training/export/verification against the new manifest and a new run
directory, retaining the original absolute parent deadline and total private
artifact budget. Do not invoke the complete smoke runner again: it would repeat
capture/extraction and establish a new invocation timer. `reuse.json` records
the old/new source identities and exact copied artifact hashes without claiming
that extraction ran under the repaired code. No inference or optimization is
performed by that copy helper.

Before any optimization, the parent measured actual cloud CPU construction of
the unchanged pinned network: 155,825,027 total parameters, 20,029,955 head
parameters (80,119,820 parameter bytes), 5.871 seconds loading and 2.681 GB
process peak RSS. This establishes construction fit only. Actual smoke training,
resume and inference results must come from cloud receipts, not the manufactured
and mocked local tests. After deployment, monitor invalid inputs, model load
errors, timeouts, availability, visibility calibration and paired prediction
quality; no deployment or automatic promotion is performed here.

Use an approved Linux cloud workspace with Python 3.10/3.11. The new requirements
files retain Torch 2.5.1, torchvision 0.20.1 and the existing frozen vision pins;
SAM extras have a separate requirements file. Dependencies and checkpoints must
already be present. None of these commands installs packages, downloads weights,
provisions resources, selects paid compute, or launches another cloud job.

`bench/runtime/adaptation-sources.json` binds the actual cached upstream Python
and configuration files to the GoTrack/SAM revisions and binds the existing
GoTrack loader. A changed source causes preflight failure. Review a new source
lock deliberately when changing the frozen runtime. Do not regenerate the lock
to bypass an unexpected mismatch. The official checkpoint hashes are also checked
before loading. Resumable checkpoints and exports are trusted, private artifacts
of the same run; do not supply arbitrary third-party checkpoint files.

Model source/checkpoint cache stays within 8 GiB. Default head run output is
bounded to 1.5 GiB; optional SAM output to 3 GiB. Temporary checkpoint space is
reserved before writing, with an additional 1 GiB free-disk reserve. These are
private research artifacts. There is no upload command, and a large SAM export
may exceed the separate 256 MiB public-viewer budget.

Run the stages sequentially. Finish and release the GoTrack process before
starting SAM. Only the parent organizer schedules execution. The explicit
`VISUALIZEIT_CLOUD=1` and `--cloud-execution` assertions plus Linux checks prevent
accidental workstation training; they are operator assertions, not cloud-host
attestation.

## Correspondence inputs and labels

The model predicts template-to-query **displacement**, not 3D coordinates:
`query_uv = template_integer_uv + flow_xy`. Upstream DPT returns float flow
`[B,2,280,280]` and sigmoid visibility `[B,280,280]`. Object coordinates belong
to the template pixel and are supplied separately in **metres**. Camera axes are
OpenCV: right, down, forward. All UVs and intrinsics use integer pixel centers in
the 280×280 pinhole crop. Native-image/fisheye conversion remains the existing
refiner's responsibility; do not interpret crop errors as native-image errors.

Prepare one `.npz` per actual inference crop with precisely these arrays:

| Key | Shape/dtype | Meaning |
| --- | --- | --- |
| `query_rgb`, `template_rgb` | `[280,280,3] uint8` | RGB inputs; no annotations |
| `template_mask` | `[280,280] bool` | Rendered object support |
| `template_xyz_m` | `[280,280,3] float32` | Template surface in object-frame metres; zero outside support |
| `query_K` | `[3,3] float64` | Query crop intrinsics |

`quality_adaptation_prepare.crops_from_retained_capture(arrays)` converts the
existing `GoTrackRefiner` chronological `retained` callback arrays, including
its millimetre depth/estimated template pose, into this contract. The caller must
capture the estimated-seed inference branch, never reference-pose replay. The
function accepts the existing callback's `query_rgb_crop`, `template_rgb`,
`rendered_depth_mm`, `rendered_mask`, `crop_k`, and
`current_crop_camera_from_object_mm`. Save its returned mapping with `np.savez`.
This is an adapter API, not a new automatic dataset acquisition/capture command.
No real training crop or annotation dataset is included in this change.

Create an input-only extraction manifest:

```json
{
  "schema": "correspondence_input_crops_v1",
  "conventions": {
    "flow": "template_to_query_displacement_xy_crop_pixels",
    "pixels": "integer_centers_v1",
    "object_xyz": "object_frame_metres",
    "camera": "opencv_x_right_y_down_z_forward",
    "crop_hw": [280, 280]
  },
  "rows": [
    {"id": "train-object-sequence-frame", "crops": {"path": "crop.npz", "sha256": "<actual lowercase SHA256>", "bytes": 123}}
  ]
}
```

The example descriptors are placeholders; replace both hash and size with actual
file identities. Paths are relative to the manifest directory. Crops must come
from a reviewed, fixed sampling plan. Keep every failed frame in the final
inventory; do not choose successful views after examining validation results.

On the approved cloud host:

```bash
export VISUALIZEIT_CLOUD=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
timeout --signal=TERM --kill-after=120 1200 \
  python -m bench.quality_adaptation_prepare \
  --manifest /cloud/data/crops.json --output /cloud/data/features-run1 \
  --device cpu --max-seconds 900 --max-rows 64 --cloud-execution
```

Extraction uses the real frozen backbone, decoder and DPT head. It saves decoder
hooks 2/5/8/11 as `[400,768] float32`, template geometry and frozen flow/confidence.
ImageNet normalization happens inside the upstream backbone. `recipe.json` and
`index.json` bind features to the input-only extractor and source lock. Failed or
unprocessed rows stay in the index. No reference poses or labels enter this call.

Labels are separate `.npz` files containing `flow [2,280,280] float32`, `visible
[280,280] bool`, and `known [280,280] bool`. `visible ⊆ known ⊆ template_mask`.
Unknown pixels do not train visibility; occluded known pixels do. Visible targets
must lie inside the query crop. All values must be finite; no implicit imputation
or unit conversion occurs in training.

For independently annotated poses or explicitly synthetic renderer training
data, `project_supervision(...)` provides a pure numerical label helper. Pass
template object XYZ, object-to-query-camera pose in metres, crop intrinsics,
independent metric query depth and an independent observed object mask. Visibility
requires depth agreement **and** observed foreground; projected silhouette alone
does not qualify. Zero depth means unknown. Choose and record a depth tolerance
from the annotation/sensor specification before training. This helper creates
supervision only; its pose/depth/mask arguments are never inference inputs.
Symmetric objects require independently justified surface identities; these
labels do not resolve symmetry ambiguity automatically.

## Split inventory and launch

Place the separately authored label archives under the extraction directory.
Create `labels.json` with schema `correspondence_label_inventory_v1`,
`label_origin` equal to `independent_annotations` or
`synthetic_renderer_training_only`, and `rows`. Every row has string `id`,
`object_id`, globally scoped `sequence_id`, `capture_group`, `query_rgb_sha256`,
`split` (`train`, `val`, `test`) and `status` (`available`, `unavailable`). An
unavailable row may declare a null RGB hash with explicit provenance as described
in the systems smoke section above.

`query_rgb_sha256` is SHA256 of the decoded `uint8[280,280,3]` query **crop**
array's C-order bytes (`hashlib.sha256(rgb.tobytes(order="C")).hexdigest()`).
It is not a JPEG/file hash or a full-image hash. The extractor computes this
identity from its validated input pixels and records it in `index.json`.
Assembly and training preflight require the available train/validation inventory
hashes to match that receipt. Identical query crops with different templates or
feature archives therefore cannot cross training/validation splits. Sealed-test
query identities remain declared metadata because test pixels are never opened.

Available training/validation rows additionally have a `labels` descriptor.
Available sealed-test rows have both `features` and `labels` descriptors, which
are checked as metadata but never opened. Unavailable rows instead have a
`failure_reason`. Objects, capture families/sequences, query RGB hashes and
feature-archive hashes must not cross splits. Every extracted row, including failures,
must appear. This strict policy evaluates new-object transfer; it is not a
same-object adaptation split. The audits validate declared identities and file
content, not the truth of an untrusted annotation claim.
Training preflight independently rejects duplicate extraction IDs and missing
extracted rows, including failures and unprocessed rows, even for a manifest
edited after assembly.

```bash
python -m bench.quality_adaptation_manifest \
  --extraction /cloud/data/features-run1 --labels /cloud/data/labels.json \
  --output /cloud/data/features-run1/manifest.json

bash scripts/cloud_adaptation_head.sh train \
  /cloud/data/features-run1/manifest.json \
  bench/runtime/adaptation-head-cpu.json /cloud/runs/correspondence-1

# Use the exact same manifest/config/run after a bounded partial invocation.
bash scripts/cloud_adaptation_head.sh resume \
  /cloud/data/features-run1/manifest.json \
  bench/runtime/adaptation-head-cpu.json /cloud/runs/correspondence-1

python -m bench.quality_adaptation_train export \
  --run /cloud/runs/correspondence-1 --output /cloud/runs/correspondence-1/head.pt \
  --cloud-execution
```

Preflight prints train foreground/flow distributions, missingness policy and split
availability before optimization. Loss is per-frame visible-pixel SmoothL1
(`beta=1` crop pixel, averaged over XY) plus known-pixel visibility BCE. No spatial
augmentation is applied to cached tokens/geometry. Validation selects by mean
endpoint error per object, with object bootstrap intervals when at least two
independent objects exist. Frozen predictions provide the paired baseline.
Training/validation curves are in `history.json`; stopped/failed rows and total
availability remain explicit. Test evaluation is deliberately a separate future
stage, and synthetic labels cannot support independent real-world accuracy claims.

Monitor `status.json`, `history.json`, `preflight.json` and the foreground shell.
Exit code 3 means incomplete. Head resume restores optimizer, AMP scaler, Torch
random state and deterministic epoch/batch cursor. A partial validation pass is
rerun in full. Last and best checkpoints are atomically replaced, with no epoch
checkpoint accumulation. The total time/step cap does not reset on resume. An
exhausted total budget requires a newly reviewed experiment configuration, not an
automatic extension. Shell watchdogs terminate/reap a stuck invocation; a hard
kill may lose work since the last checkpoint.

The exported state uses exact upstream `pose_head` keys. In a separately
authorized cloud replay, construct the existing frozen GoTrack network, then
call `quality_adaptation_runtime.load_export_into_network(network, path,
expected_sha256=receipt_hash, cloud_execution=True)`. It performs strict head
loading and leaves backbone/decoder weights unchanged. This does not modify the
normal frozen loader or promote the model into `quality_runner`. Fresh-network
cached-token and RGB inference parity passed in the dated smoke above. Complete
tracking replay, PnP availability and pose accuracy with the adapted head remain
unverified.

## Optional SAM route

SAM is gated on a reviewed `segmentation_bottleneck_v1` JSON report containing
`reviewed: true`, a nonempty `reason`, a finite `baseline_iou` in `[0,1]`, and
`evaluation_protocol: "independent_masks_initial_prompt_only"`. The dataset
manifest references it with an actual path/hash/size descriptor. A poor pose score
alone is not evidence that mask fine-tuning is the appropriate next experiment.

The dataset schema is `sam2_adaptation_dataset_v1`, with
`label_origin: "independent_annotations"`, `segmentation_bottleneck`, and
`sequences`. Each sequence has `id`, `object_id`, `capture_group`, `split`,
`status`; available sequences have `frames`, unavailable sequences a
`failure_reason`. A frame has `rgb` and `mask` descriptors. IDs use letters,
digits, underscore or hyphen. Files follow the actual upstream DAVIS-style loader:
`images/<id>/00000.jpg`, `masks/<id>/00000.png`, and so on. Require RGB JPEG,
matching palette/grayscale PNG with background 0/object 1, foreground in the
first frame, 2–8 frames per sequence, and no extra files in those folders.
The first release intentionally supports one object per sequence. Object and
capture families and RGB content must remain disjoint across train/val/test;
test files stay unopened.

```bash
bash scripts/cloud_adaptation_sam.sh train \
  /cloud/sam-data/manifest.json bench/runtime/adaptation-sam-cpu.json /cloud/runs/sam-1
bash scripts/cloud_adaptation_sam.sh resume \
  /cloud/sam-data/manifest.json bench/runtime/adaptation-sam-cpu.json /cloud/runs/sam-1
python -m bench.quality_adaptation_sam export --run /cloud/runs/sam-1 \
  --output /cloud/runs/sam-1/sam-adapted.pt --cloud-execution
```

The script explicitly supplies `--enable-sam-adaptation`. The adapter resolves
the pinned official base-plus training template, uses one process and zero loader
workers, disables cluster scheduling, and trains only the mask decoder. It keeps
the upstream segmentation loss, optimizer and training loop. Validation uses
the first annotated mask as an initial prompt, then scores later frames without
correction prompts. The initial frame is excluded. IoU/Dice and object bootstrap
intervals describe this prompt-conditioned segmentation protocol, not automatic
detector recovery or pose accuracy. The frozen model is evaluated before training;
early stopping selects the best validation IoU, which can remain the frozen
baseline if adaptation does not improve it.

SAM resume uses the last **completed epoch**, restoring optimizer/scaler and
random states; an interrupted partial epoch is replayed. `checkpoint.pt` and
`best.pt` are written atomically by the adapter. The upstream loader's silent
replacement of failed training samples is disabled. The upstream diagnostic that
prints all environment values is suppressed to keep cloud credentials out of
logs. Monitor `validation.json`, upstream train/val statistics, and `status.json`.

SAM export has the upstream `{"model": state_dict}` format and records the
compatible inference config `configs/sam2.1/sam2.1_hiera_b+.yaml`. Existing frozen
SAM wrappers intentionally reject its different hash. A separate reviewed cloud
replay using upstream `build_sam2`/video predictor is required before integration.
No actual SAM trainer import/optimization, CUDA test, checkpoint resume or export
load has been run against installed SAM training dependencies for this change.

## Local checks performed

Tests use manufactured analytical geometry, NumPy metrics, fake tensors and
mocked serializer/model interfaces. They cover units, visibility/occlusion,
object/sequence/content leakage, sealed test isolation, source/config identity,
bounded atomic failure cleanup, deterministic resume ordering, real upstream
configuration field wiring, and strict exported head loading. They do not call a
neural forward pass, optimizer, cloud service or download.
