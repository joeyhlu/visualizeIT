# Kaggle execution, evidence and scale planning

The evidence and scale tools below do not train a model. The standalone adapter
in `scripts/kaggle_adaptation.py` installs authenticated source and launches the
existing ROOT18 adaptation APIs. Its separate bootstrap installs dependencies
into an isolated directory on Kaggle. A verified source archive establishes
its bytes only. The original cloud baseline requires Linux, Python 3.10/3.11,
Torch 2.5.1 and the pinned `bench/runtime/requirements-*.txt` dependencies.
The observed Kaggle CPU image is separate from any eventual GPU image; actual
compatibility, GPU availability, crop/token parity and resource measurements
remain required before training. No automatic dependency repair is attempted.

Run these commands from the current source checkout or a separately identified
supplement. Adding these modules to an old pinned source archive changes its
identity and does not inherit that archive's admission. The historical ROOT30
operator and its exact source manifest remain unchanged. The additive adapter
is subject to independent source review and actual cloud integration checks.
Output reports are created
exclusively; choose a fresh output path each time.

```powershell
python -m bench.quality_kaggle_preflight --request preflight-request.json --output preflight-report.json
python -m bench.quality_adaptation_scale --inventory candidate-inventory.json --seed 42 --shard-size 64 --output scale-plan.json
python -m bench.quality_adaptation_learning_curve --history run/history.json --status run/status.json --inventory-plan scale-plan.json --output curve-report.json
```

Preflight request fields are `provenance` (explicit `provider: "kaggle"`,
`notebook_url`, `observed_at`, `evidence_source`), `proposed_roots` (named paths),
`source_archive` (`path`, lowercase `sha256`, `bytes`) and `assets` (the same
descriptors). The probe reads installed distribution metadata, archive bytes and
filesystem free space. It does not import frameworks, probe hardware or hash
large checkpoint contents. Its unresolved execution and parity blockers are
intentional; this probe cannot establish runtime readiness. Exit code 2 means
blockers remain. The current account UI requires human phone verification to
enable GPU and Internet. The observed CPU notebook had Internet disabled and
`No persistence`; do not assume preparation or resume checkpoints survive a
session stop. Recheck the eventual runtime and explicit storage/persistence
settings before allocating compute.

Candidate inventory is metadata only, not a correspondence-label adapter. It
uses the existing ranker `label_policy` and complete observed `background_catalog`
format. Each `records` entry requires a stable `id`, `refinement_state` integer,
audited `inputs_sha256`, exact metric `mesh_sha256`, `physical_object_id`,
`scene_id`, per-object `sequence_id`, `frame_id`, `source_frame_id`, explicit
`provenance_kind` (`real`, `synthetic`, or development-only `fixture`),
`background_id` and `background_kind`. Optional `aliases` connect equivalent
physical identities. For observed backgrounds, the catalog must include every
co-occurring physical object with `inventory_complete: true`. Paths embedded in
records are hashed as metadata and never opened by the planner. Historical
experiments and their frozen manifests remain separate.

The planner needs at least three independent nondevelopment connected groups;
otherwise it reports infeasibility and makes no preparation shards. Its seeded
group order assigns one independent group each to validation and test and the
remaining groups to training. This is a conservative deterministic allocation,
not a balanced sampling optimizer. All previously inspected identities and any
connected exposure remain development only. Source frames and refinement states
are counted separately. Requests for 1000/2000/4000 training states describe
targets, not acquired data. Four raw float32 token arrays use 4,915,200 bytes per
state; the full raw state estimate is 7,659,272 bytes, about 30.6 GB for 4000
states before source assets, temporary files and checkpoints. Compression and
fit within the account quota are unproven.

Preparation integration uses Python APIs `validate_preparation_receipt` and
`assess_admission` in `quality_adaptation_scale`. Each shard receipt binds the
global `manifest_sha256`, `shard_sha256`, pinned `producer_sha256`,
`config_sha256`, measured `preparation_seconds`, and every row's `id` and
`inputs_sha256`. A row is `available` with an output descriptor (`path` relative
to the explicit output root, `sha256`, `bytes`), or `failed` with a nonempty
`failure_reason`. Failed rows stay counted. Unknown or sealed test identities
are rejected before output files are opened. Resume rechecks actual output bytes
and rejects changed producer, configuration, input, manifest or shard identity.
Validation also reconstructs the split manifest and derived counts from global
state records, and requires exactly one matching TRAIN/VAL shard entry per state.
Resealing inconsistent metadata does not bypass these checks. Admission requires
zero failed preparation rows; failures remain preserved and counted.

`assess_admission(plan, receipts, producer_sha256=..., config_sha256=...,
output_root=..., measurements=...)` revalidates receipts. Measurements must bind
`plan_sha256`, select a frozen `train_states_requested`, and include positive
`measured_training_steps`, `training_seconds`, `planned_optimizer_steps`,
`actual_free_bytes`, `storage_limit_bytes`, `time_limit_seconds` and `reserve_bytes`.
Storage observations must use `storage_baseline: "after_preparation"`: all verified
preparation outputs are already resident when current free space is measured.
Also provide integer `measured_resident_bytes` for **all** existing usage charged
to the same storage quota, including preparation outputs, source assets,
checkpoints and any other existing occupancy; it must cover the verified prepared
bytes. Provide nonnegative integer `measured_additional_peak_bytes` for the
largest additional occupancy needed after that observation, including any new
checkpoints, assets and temporary files. Existing resident assets must not be
included again in additional occupancy. The free-space check is additional peak
plus reserve against current free bytes. The independent total-quota check is
resident plus additional peak plus reserve against the storage limit. Exact
boundary equality fits. Missing evidence blocks admission.
Resource admission covers the prepared inventory only and is not runtime,
training quality or quota approval. Measurements are caller-supplied evidence;
verify their origin and representative workload. Assemble one immutable global
dataset with the existing adaptation manifest/data tools before using the
existing trainer. Shards are preparation/storage units, never replacement
manifests for successive resumes.

The learning-curve tool reads existing JSON history/status without loading a
checkpoint. It reports composite training loss, adapted and frozen validation
EPE, steps, availability and optional inventory diversity. It does not subtract
loss from EPE or infer underfitting from sample counts. Optional
`--matched-metrics metrics.json` accepts a list of explicit TRAIN/validation
records with `steps`, `split`, `value`, `metric_definition`, `units`, `aggregation`,
`coverage_policy`, `dataset_manifest_sha256`, `available` and `total`. Each side
must have complete coverage; counts may differ. Definitions, units, aggregation,
coverage policy and global manifest must match before a descriptive gap is
reported. Independent labels, uncertainty and online quality checks remain
necessary for a generalization claim.

## Standalone source installation

The new launcher uses Python's standard library only. It can install source with
the observed Kaggle CPU Python 3.13 before runtime setup. Installation is not
training admission. Kaggle actually expanded the approved private ZIP into 971
files under three directories. The mounted input observed on this account is:

```bash
python3 /kaggle/working/kaggle_adaptation.py install-source \
  --source-directory /kaggle/input/datasets/achitaan/visualizeit-reviewed-training-source-v1 \
  --destination /kaggle/working/reviewed-source
```

This authenticates the three original manifest hashes, exact membership and all
971 file hashes before any destination writes. Its receipt records
`kind: "expanded_directory"` and manifest hashes; it does **not** claim to have
verified the absent ZIP's bytes. Extra files/directories, aliases, symlinks and
corruption fail before destination creation. A source with an additional
installation receipt is not a pristine expanded input.

Where the original ZIP actually exists, the alternative is:

```bash
python3 /kaggle/working/kaggle_adaptation.py install-source \
  --archive /kaggle/input/ACTUAL_PRIVATE_DATASET/larger-amortized-three-root-source-v1.zip \
  --destination /kaggle/working/reviewed-source
python3 /kaggle/working/kaggle_adaptation.py verify-source \
  --destination /kaggle/working/reviewed-source
```

Replace `ACTUAL_PRIVATE_DATASET` with the mounted dataset path shown by Kaggle.
The archive is exactly 5,700,194 bytes with SHA-256
`cbc1e698fce71f1be1561ac1033a1a7bbbe82fa1a8ddbd6050036ba56b6f8a7a`.
It contains `visualizeit-cloud-v18`, `visualizeit-cloud-v29` and
`visualizeit-cloud-v30`: respectively 287, 337 and 344 source files, plus each
root's `cloud-source-manifest.json` (971 members total). The launcher verifies
the whole archive, exact manifests, every member hash and CRC before creating
the destination. Duplicate names, traversal, symlinks, nonregular members,
unlisted files and overwrites fail. Archive receipts record `kind: "archive"`
and the actually verified byte count/hash. An I/O failure after authentication can leave
a partial destination; its absent completion receipt prevents execution. Keep
that failed directory for inspection and choose a fresh destination for retry.

The historical `.cache/cloud-prepare/larger_amortized_cuda_execute_v1.py` retains
SHA-256 `51e7a6d6b4b8f587ef38dbc6ba13cea5c82fc15186ba8a4f178787136645eb17`.
Its Lightning authorization, clocks and preservation rules are not adapted by
moving it to a new directory. The launcher uses ROOT18's actual portable APIs.

## Isolated runtime bootstrap

Run `scripts/kaggle_runtime_bootstrap.sh` only in the actual Kaggle Linux session
after Internet is available. Supply an independently authenticated, compatible
`uv==0.12.22` wheel and its SHA-256; the script does not fetch or execute an
unpinned installer. Obtain the wheel from the official
[uv package release](https://pypi.org/project/uv/0.12.22/), keeping the release
file identity with the run. Linux architecture and glibc compatibility still
need checking against the eventual GPU image.

```bash
bash /kaggle/working/kaggle_runtime_bootstrap.sh \
  /kaggle/working/kaggle_adaptation.py \
  /kaggle/working/reviewed-source \
  /kaggle/working/isolated-runtime \
  /kaggle/input/ACTUAL_PRIVATE_DATASET/EXACT_UV_0_12_22_LINUX_WHEEL.whl \
  INDEPENDENTLY_VERIFIED_UV_WHEEL_SHA256
```

This pins managed CPython **3.11.13**, `torch==2.5.1+cu124`,
`torchvision==0.20.1+cu124`, and ROOT18's existing
`bench/runtime/requirements-ranker.txt` / `requirements-quality.txt`. The managed
Python install and uv cache live under the fresh runtime root using
`UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR`; the venv uses the explicit managed
interpreter. The notebook interpreter only inspects files and installs the
verified uv wheel into a dedicated target directory. The interpreter selection
follows [uv's exact-version documentation](https://github.com/astral-sh/uv/blob/0.12.22/docs/concepts/python-versions.md).

The final `runtime-receipt.json` records the executable hash, exact patch,
distribution metadata, source requirement hashes, verified uv wheel descriptor
and resolved requirement file descriptor. It also records `runtime_root` and
the exact internal symlink layout as `runtime_links: [{"path": "relative/path",
"target": "literal-link-target"}]`. Only those authenticated runtime links are
allowed in resident accounting. Physical targets count once; changed, unlisted,
broken, cyclic or escaping links fail. Source, input and output paths retain
their prohibition on symlink components. No receipt is emitted for a partial
installation. These records support identity checks; package metadata and
`pip freeze` are not a cryptographic attestation of every installed package file.
Dependency resolution, CUDA driver support and EGL/rendering remain actual
environment checks. The bootstrap never calls `cloud_ranker_bootstrap.sh`,
`bench.quality_assets`, model imports, training or automatic resource creation.
Model asset acquisition is a separate explicit operation through the existing
pinned asset APIs and retains the 8 GiB model-cache limit. No substitute weights
or architecture are permitted.

## Execution request and admission

Every stage takes `--request` and the SHA-256 of that exact JSON file. Paths are
absolute. Input descriptors have exactly `path`, nonnegative integer `bytes`,
and lowercase `sha256`. All relevant TRAIN/validation inputs, source assets,
checkpoint inputs and detailed parity/input-review reports must be inventoried;
sealed test arrays must not be included or opened. Paths must stay in the
explicit `/kaggle/input` or `/kaggle/working` roots without symlink components.

The required request shape is below. `FILE_DESCRIPTOR` denotes a real existing
file descriptor, not literal JSON to submit. Fields marked stage-specific are
added only when those files exist; retain all other identity fields unchanged
through the run. Keep `crop_manifest`, `labels`, `extraction`, `inputs` and output
paths fixed from the first stage. `global_inventory` is the frozen scale plan
containing its `plan_sha256`, never a succession of per-shard datasets.

```json
{
  "schema": "kaggle_adaptation_request_v1",
  "mode": "extract",
  "source_install": "/kaggle/working/reviewed-source",
  "python": "/kaggle/working/isolated-runtime/venv/bin/python",
  "input_root": "/kaggle/input/ACTUAL_PRIVATE_DATASET",
  "working_root": "/kaggle/working",
  "output_root": "/kaggle/working/adaptation-run-001",
  "runtime": "FILE_DESCRIPTOR: isolated-runtime/runtime-receipt.json",
  "config": "FILE_DESCRIPTOR: existing CUDA TrainConfig JSON",
  "global_inventory": "FILE_DESCRIPTOR: frozen global scale plan JSON",
  "scale_sources": {
    "quality_adaptation_scale": "FILE_DESCRIPTOR: unchanged reviewed quality_adaptation_scale.py supplement",
    "quality_kaggle_preflight": "FILE_DESCRIPTOR: unchanged reviewed quality_kaggle_preflight.py supplement"
  },
  "identity_bridge": "FILE_DESCRIPTOR: authoritative identity bridge JSON described below",
  "inputs": [
    {"role": "parity_report", "file": "FILE_DESCRIPTOR: detailed passing crop/token parity report"},
    {"role": "input_review", "file": "FILE_DESCRIPTOR: authentic-input and split review"},
    {"role": "training_input", "file": "FILE_DESCRIPTOR: each relevant non-test input file"},
    {"role": "checkpoint", "file": "FILE_DESCRIPTOR: ROOT18/.cache/model-quality/checkpoints/gotrack_checkpoint.pt"}
  ],
  "crop_manifest": "FILE_DESCRIPTOR: actual correspondence_input_crops_v1 manifest",
  "labels": "FILE_DESCRIPTOR: actual correspondence_label_inventory_v1 inventory",
  "extraction": {"max_rows": 64, "max_seconds": 900, "max_output_bytes": 536870912},
  "paths": {
    "extraction_dir": "/kaggle/working/adaptation-run-001/extraction",
    "manifest_output": "/kaggle/working/adaptation-run-001/extraction/manifest.json",
    "train_dir": "/kaggle/working/adaptation-run-001/train",
    "export_output": "/kaggle/working/adaptation-run-001/train/head.pt"
  },
  "preparation": {
    "producer": "FILE_DESCRIPTOR: authenticated ROOT18/bench/quality_adaptation_prepare.py",
    "config": "FILE_DESCRIPTOR: exact extraction settings plus device cuda",
    "output_root": "/kaggle/working/adaptation-run-001/extraction",
    "receipts": [],
    "measurements": "STAGE-SPECIFIC FILE_DESCRIPTOR: original scale admission measurements JSON"
  },
  "dataset": "STAGE-SPECIFIC FILE_DESCRIPTOR: assembled manifest.json, preflight/train/resume/export",
  "scale_admission": "STAGE-SPECIFIC FILE_DESCRIPTOR: existing assess_admission result, train/resume/export",
  "evidence": "FILE_DESCRIPTOR: current stage's actual execution evidence JSON",
  "limits": {
    "max_seconds": 900,
    "max_rss_bytes": 17179869184,
    "max_output_bytes": 2147483648,
    "max_log_bytes": 8388608,
    "reserve_bytes": 1073741824
  }
}
```

Generate a descriptor without importing a framework:

```python
import hashlib, json
from pathlib import Path
def descriptor(filename):
    path = Path(filename).absolute()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
print(json.dumps(descriptor("/kaggle/working/isolated-runtime/runtime-receipt.json")))
```

The CUDA config is passed unchanged to the existing `TrainConfig` reader;
explicit `device: "cuda"` is mandatory. The supervisor deadline and output cap
must cover its `max_seconds` and `max_output_bytes`. Neither a changed config nor
a different global dataset may be used to extend an existing checkpoint.

Export must be a fresh **direct-child file in `train_dir`**, as required by the
existing trainer. A sibling such as `adaptation-run-001/head.pt` is invalid.

First populate the stable request fields and runtime descriptor, then compute
the evidence binding (this command does not admit execution):

```bash
python3 /kaggle/working/kaggle_adaptation.py request-identity --request /kaggle/working/request.json
```

Write actual observations into an evidence file with this shape. No passing
evidence file is shipped; missing phone verification, GPU, parity, authentic
inputs or measured fit must remain a blocker.

```json
{
  "schema": "kaggle_execution_evidence_v1",
  "provider": "kaggle",
  "stage": "train",
  "identity_sha256": "OUTPUT_OF_REQUEST_IDENTITY",
  "notebook_url": "https://www.kaggle.com/code/achitaan/ACTUAL_NOTEBOOK",
  "session_id": "ACTUAL_CURRENT_SESSION_IDENTIFIER",
  "evidence_source": "PATH_OR_DESCRIPTION_OF_ACTUAL_OBSERVATIONS",
  "observed_at": "ACTUAL_UTC_ISO_TIMESTAMP",
  "expires_at": "UTC_ISO_TIMESTAMP_WITHIN_ONE_HOUR_OF_OBSERVATION",
  "quota": {"paid": false, "gpu_enabled": true, "gpu_remaining_seconds": 900, "session_remaining_seconds": 900},
  "gpu": {"uuid": "ACTUAL_NVIDIA_SMI_UUID", "name": "ACTUAL_NVIDIA_SMI_NAME"},
  "parity_passed": true,
  "authentic_inputs_reviewed": true,
  "test_arrays_opened": false,
  "measurements": {
    "measured_units": 1,
    "measured_seconds": 1,
    "planned_units": 2000,
    "storage_baseline": "current_resident_before_stage",
    "resident_bytes": 1,
    "additional_peak_bytes": 1,
    "storage_limit_bytes": 1
  }
}
```

The numeric measurement entries above illustrate types, and intentionally do
not establish resource fit. Replace them with representative actual observations
for the exact stage/config/runtime/inputs. Training projection must cover the
full frozen `max_steps`, extraction projection all crop rows. Inventory every
resident byte charged to the same working storage quota; include logs and
temporary checkpoints in additional peak. Nested roots and hard links are
counted once; filesystem free space is never added twice for shared storage.
The runtime checks current `nvidia-smi` identity, explicit Kaggle mounts/session,
recent evidence, free quota and actual free space. This bounded route currently
selects exactly one observed GPU UUID and sets `CUDA_VISIBLE_DEVICES` for the
child. It does not authenticate Kaggle account claims
or independently prove caller-supplied parity/measurements. Preserve the original
observations and review the detailed reports.

For train/resume/export also provide the real `assess_admission` report described
above, admitted with no blockers and matching the global plan. Its existing
after-preparation accounting remains unchanged. The adapter's current-resident
check supervises the additional invocation; it does not replace that admission.

After writing evidence and updating its descriptor, compute the exact request
SHA and execute the desired stage, for example:

```bash
request_sha=$(python3 -I -c 'import hashlib;print(hashlib.sha256(open("/kaggle/working/request.json","rb").read()).hexdigest())')
python3 /kaggle/working/kaggle_adaptation.py train \
  --request /kaggle/working/request.json --request-sha256 "$request_sha"
python3 /kaggle/working/kaggle_adaptation.py status \
  --output-root /kaggle/working/adaptation-run-001
```

Use separate invocations in order: `extract` invokes
`bench.quality_adaptation_prepare`; `assemble` invokes
`bench.quality_adaptation_manifest`; `preflight`, `train`, `resume` and `export`
invoke the corresponding `bench.quality_adaptation_train` commands. Resume is
only for an incomplete run; export requires completed training. After assembly,
add the resulting manifest descriptor as `dataset`. Each invocation needs fresh
stage evidence and request SHA. No automatic sequence, scheduler, resource
upgrade or spending occurs.

After extraction, copy the already authored label files to their frozen
`label_output` locations inside the extraction directory before `assemble`.
These are byte copies of the original authenticated labels, never new labels.
The next metadata check verifies their exact bytes. The extraction directory
must be fresh for `extract`; pre-populating it with labels would violate the
existing extractor's creation guard. Keep failed rows and original outputs;
changing the frozen input/label inventory requires a new run identity.

The extraction manifest must already describe authentic estimated-seed crops.
The label inventory must already contain independently authored correspondence
labels, or the existing explicitly documented synthetic-training-only labels.
The generic assembly command does not create supervision, merge feature shards,
choose new splits or rescue unavailable rows. Porting the historical larger-run
operator's capture/preparation orchestration to Kaggle is separate unfinished
work. If only raw recordings or a scale plan exist, this adapter cannot train.
Use the existing preparation receipt APIs for sharded preparation, assemble one
global dataset outside this bounded single-extraction route, and never resume
successive shards as different datasets. Any such assembly needs its own review
and authentic receipts; the launcher does not fabricate them.

Every child runs with the explicit venv Python, authenticated ROOT18 as its
working directory, user site disabled and Python/user-base/library-path overrides
removed. The parent records actual command, exit, capped log, deadline, sampled
group RSS, state and checkpoint descriptors in `adapter-state.json`. Linux
process groups receive termination and then kill, including surviving descendants;
timeouts and interruptions retain outputs. RSS includes shared pages
conservatively and polling can observe a storage/RSS overshoot after it occurs;
these checks are not kernel-enforced filesystem or memory quotas. ROOT18's own
atomic/checkpoint/reserve guards remain authoritative as well.

After train/resume, the adapter reads the existing trainer's `status.json` and
records `optimizer_progress.last_confirmed_steps`, epoch, elapsed time, trainer
status, status-file SHA-256, producer-source SHA-256 and observation timestamp.
`status` rereads that bounded JSON file without loading any checkpoint. Counts
are last confirmed observations; `current_progress_known` remains false.
Missing, malformed, older-than-invocation or more-than-120-second-old metadata,
failed stages and unconfirmed adapter liveness cannot confirm completion.
Unexpected post-invocation status changes retain the earlier confirmed count
with an explicit freshness warning. Historical child outcomes remain in the
stage receipts and `adapter_status`; the effective `status` becomes incomplete
when current training completion cannot be confirmed. A child exit 0 without a
fresh completed trainer status is recorded as incomplete and cannot enable export.

Exit 0 means that stage completed, 2 means blocked/invalid/failed admission, and
3 means incomplete or a failed child. A child's exit 3 is always incomplete.
The persisted stage distinguishes incomplete from failed children, and a parent
crash can leave `running` plus a lock file rather than falsely reporting success.
Before manually removing a stale lock, inspect its PID and ensure no child group
survives. `status` is read only. Completed extraction or preflight is not a model
quality or deployment claim.

## Frozen-plan identity bridge and original receipt validation

The old dataset schema omits mesh, scene, background and refinement identities.
Matching two independent JSON hashes is insufficient admission. Every
array-processing stage requires the original producer's identity provenance,
frozen **before** scale planning. The SHA-256 of each provenance JSON file's
exact bytes must be that state's `inputs_sha256` in the frozen scale plan. An
existing opaque digest cannot be retroactively replaced with a new bundle.
Such a plan blocks until its authentic original provenance is available.

Each producer provenance file has this schema (descriptors refer to authentic
original files; no sample arrays or supervision are supplied):

```json
{
  "schema": "kaggle_input_identity_v1",
  "identity": {
    "id": "ORIGINAL_STATE_ID",
    "physical_object_id": "ORIGINAL_PHYSICAL_OBJECT_ID",
    "scene_id": "ORIGINAL_SCENE_ID",
    "sequence_id": "ORIGINAL_SEQUENCE_ID",
    "frame_id": 0,
    "source_frame_id": 0,
    "refinement_state": 0,
    "mesh_sha256": "EXACT_METRIC_MESH_FILE_SHA256",
    "background_id": "ORIGINAL_BACKGROUND_ID",
    "background_kind": "observed",
    "provenance_kind": "real",
    "aliases": []
  },
  "capture_group": "ORIGINAL_SCENE_ID",
  "status": "available",
  "query_rgb_sha256": "ACTUAL_PRODUCER_QUERY_RGB_HASH",
  "source_frame": "FILE_DESCRIPTOR: original source frame",
  "mesh": "FILE_DESCRIPTOR: exact metric mesh",
  "crop": "FILE_DESCRIPTOR: actual input-only estimated-seed crop",
  "labels": "FILE_DESCRIPTOR: original separately authored label archive",
  "label_output": "labels-ORIGINAL_STATE_ID.npz"
}
```

Use actual existing identity values, units and hashes; the integers above
illustrate field types. Paths in these provenance descriptors may be relative
to the provenance file's parent, preserving identity when packaged with a
prepared dataset. `label_output` is relative to the global extraction directory.
Dataset `object_id`, `sequence_id` and `capture_group` must map exactly to the
frozen physical object, sequence and scene; query RGB hashes must agree with
the actual extractor index. Unavailable provenance requires
`status: "unavailable"` and `failure_reason`; its failed row remains counted.
TEST/development descriptors remain metadata only and are never opened. The
bridge retains every plan state, including development metadata. The trainer
inventory contains exact TRAIN/validation/TEST states, mapping `validation` to
the existing `val` spelling.

The bridge file shape is:

```json
{
  "schema": "kaggle_correspondence_identity_bridge_v1",
  "plan_sha256": "ORIGINAL_PLAN_CANONICAL_SEAL",
  "manifest_sha256": "ORIGINAL_GLOBAL_MANIFEST_CANONICAL_SEAL",
  "rows": [{"id": "ORIGINAL_STATE_ID", "input_provenance": "ORIGINAL_PROVENANCE_FILE_DESCRIPTOR"}],
  "bridge_sha256": "CANONICAL_HASH_OF_ALL_OTHER_BRIDGE_FIELDS"
}
```

Canonical seals use SHA-256 of UTF-8 JSON with sorted keys, separators `,` and
`:`, and `allow_nan=False`, matching the existing scale planner. After assigning
the actual plan and original provenance descriptors to `bridge`:

```python
import hashlib, json
bridge["bridge_sha256"] = hashlib.sha256(json.dumps(
    bridge, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
```

The launcher authenticates two unchanged external helpers. They are separate
from the old three-root archive and require their own reviewed supplement:

| Helper | Required SHA-256 |
|---|---|
| `quality_adaptation_scale.py` | `614c12856f91f13b9c5c9cf4b88be3958293df855d11183521189386d953f141` |
| `quality_kaggle_preflight.py` | `74167feb34929151e8fd0fd6c37c6000c7ee961b71bf8cb81ee2974e148184d8` |

ROOT18 supplies byte-identical `quality_ranker_data.py`, SHA-256
`47ebe96f0c083ff549042273ee0405d2df6159d64bc28b37bfcbbe8f4e44d4d4`.
Only those exact modules load into a closed metadata namespace in the explicit
isolated Python subprocess. It imports NumPy for the existing module, never
Torch, model code or NPZ arrays. `validate_plan` reconstructs the original
grouping, exposure rules, split assignments, counts, seals and shards.
Each helper is read once, and those exact bytes are authenticated, compiled and
executed. Cached helper bytecode cannot substitute for the reviewed source.

The metadata join checks every state, object, sequence, scene, background,
metric mesh, source/crop/query identity, label descriptor, extraction
recipe/index, feature descriptor and final dataset row. It rejects TEST-to-TRAIN
reassignment, omitted failures, duplicated source/query/feature content across
splits, unrelated datasets and extra execution payloads before opening any
feature/label bytes. Only then may non-test payload hashes be read.
Before those reads, active payloads and explicitly requested inputs are checked
against all sealed/development paths and existing physical file identities.
Shared paths and hardlink aliases reject using filesystem metadata alone;
missing held-out payloads remain unopened and need not be acquired. Extraction's
automatic provenance audit reads source/mesh/crop bytes and defers original
label bytes to assembly. Label descriptors and overlap checks remain required,
and explicitly declared execution inputs retain their identity checks.

For assemble/preflight/train/resume/export, `preparation.receipts` contains all
original shard receipt descriptors. Available outputs must match actual index
feature descriptors, with every failed row retained. The producer descriptor is
the pinned ROOT18 extractor. The config file contains exactly `max_rows`,
`max_seconds`, `max_output_bytes` and `device: "cuda"` from the original
extraction. The original validator rechecks every receipt. For
train/resume/export, `preparation.measurements` supplies the original actual
`assess_admission` measurements; recomputed admission must match the supplied
`scale_admission` report exactly. Those receipt/measurement files become
immutable within the run once used. The adapter does not invent preparation
time or produce missing historical receipts.

## Already prepared immutable input datasets

Set `mode: "prepared"` for an authentic existing global feature/label dataset.
Use its read-only input directory for `paths.extraction_dir` and its manifest
for `paths.manifest_output` and `dataset.path`, for example:

```json
{
  "mode": "prepared",
  "input_root": "/kaggle/input/datasets/achitaan/ACTUAL_PREPARED_DATASET",
  "paths": {
    "extraction_dir": "/kaggle/input/datasets/achitaan/ACTUAL_PREPARED_DATASET/prepared",
    "manifest_output": "/kaggle/input/datasets/achitaan/ACTUAL_PREPARED_DATASET/prepared/manifest.json",
    "train_dir": "/kaggle/working/adaptation-run-001/train",
    "export_output": "/kaggle/working/adaptation-run-001/train/head.pt"
  }
}
```

This overrides part of the full request above. Keep source/runtime and
training/checkpoint/export outputs under working storage. All required
metadata, the original bridge, crop/label inventories, recipe/index, shard
receipts and original measurements remain required. Extract and assemble are
disallowed; invoke `preflight`, then `train` and any needed `resume`/`export`.
The unchanged lazy loader reads prepared non-test features and labels directly.
Raw source/crop files may remain historical provenance only; prepared mode
does not reopen or reacquire them. Their frozen source/crop identities and
authentic preparation metadata remain mandatory.

Read-only prepared inputs are not added to the working filesystem's resident
usage. The original scale report retains its original prepared-inventory/storage
scope; the current invocation separately requires observed working storage,
time and GPU/session quota fit. No storage upload, merger or streaming system is
added. This route does not establish that a valid larger prepared dataset exists.

There is no automatic pilot or measurement bootstrap. `assess_admission`
requires actual positive measured training steps, and full training remains
blocked without them. A separately reviewed bounded pilot and parity producer
must supply observations using the same immutable config/global dataset.
Do not bypass admission or invent numbers for the evidence template. Actual
end-to-end Kaggle training remains unfinished until those prerequisites and
provider/runtime integration checks pass.
