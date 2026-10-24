# Prepared Kaggle adaptation v2

This is a source path for a bounded research run. No cloud execution, model
training, accuracy result, larger-data acquisition or completed target is implied
by these files. Model-free fixtures exercise software contracts only.

The unchanged ROOT18 extractor produces the decoder tokens. The unchanged
ROOT18 assembler and trainer consume one immutable global dataset. The original
extractor recipe and index are preserved byte for byte. A separately audited
final index adds unavailable capture/TEST rows; it preserves every original
successful and failed extractor row. Development states have their own full
accounting and never enter the trainer's train/val/test schema.

## Before raw freeze

Use `quality_adaptation_portable_prepare.bounded_recipe(recipe, audit)` before
the existing raw freeze command. It reads metadata only. The first increment
uses the existing 24 normalized views and iterations `[0]`. It does not repeat
states or resample after outcomes. The audit schema is
`kaggle_exposure_audit_v2`, with `complete: true`,
`scope: "visualizeit_experimental_exposure_only"`, the canonical
`recipe_sha256`, pinned JSON `evidence_descriptors`, and `assets` covering every
selected declared asset. Each asset binds `physical_object_id`, `mesh_sha256`,
`aliases`, `eligible`, and `historical_roles`. Prior TRAIN requires explicit
`allow_prior_train: true` and its deterministic assignment must remain TRAIN.
Prior validation may remain validation with `allow_prior_validation_reuse: true`;
it is disclosed reused development validation for checkpoint selection. Prior
TEST may remain unavailable metadata-only TEST with
`retain_prior_test_metadata_only: true`. Validation/TEST retention requires
`frozen_seed` equal to the recipe's frozen seed. Mixed/conflicting, development,
unknown or reassigned historical roles are rejected. `val` and `validation`
normalize to the same validation role.

The audit is an externally reviewed experimental exposure declaration. It does
not establish unknown released encoder pretraining history or independently
prove a catalog's truth. The existing 22 candidate-only rows and incomplete
catalog audit are insufficient by themselves. Preserve the original audit's
incomplete flag. Known inspected development objects remain excluded. Validation
reuse and retained sealed TEST preserve their roles and never establish fresh
independent validation or TEST performance. Aliases and mesh descriptors must
agree with the selected declaration. The existing split planner also enforces
the keyboard/mug/ranch exclusions.

For the reviewed known-role route, use
`completeness_basis: "selected_known_roles_over_pinned_evidence"`: `complete`
means every listed selected asset has a positive known role over the listed
evidence. It does not mean exhaustive history for all 33 catalog objects, known
pretraining history, or absence of exposure for unknown candidates. Empty known
roles are rejected in this declaration. Freeze seed **521** and iterations `[0]`
before generation or availability outcomes. The unchanged planner yields either
the known12 option (240 requested TRAIN views, 24 reused VAL4 views and 24 sealed
TEST7 metadata rows) or the known15 option adding preserved TRAIN17/31/33 (312
requested TRAIN views, the same VAL4 and TEST7). Every actual usable count is
still zero until real preparation succeeds. Admission metadata carries
`exposure_disclosure`, including reused validation and no independent validation
or TEST claim.

This packet's proposed local files are
`.cache/cloud-prepare/kaggle-known15-recipe-proposed-v1.json` and
`.cache/cloud-prepare/kaggle-known15-exposure-audit-local-proposed-v1.json`.
They are **UNREVIEWED** pending independent scientific/source reassessment and
make no execution-admission claim. Mesh descriptors name future assets below
`/kaggle/working/visualizeit-known-development-inputs-v1/assets`; nothing has been
acquired or opened. The outer evidence descriptors currently bind real local
JSON bytes for prospective validation. Package exactly that small evidence
closure under the declared `evidence/` layout, then rebind outer descriptor paths
to verified actual cloud locations and freeze a new audit/request hash. Nested
historical paths in those JSON documents remain descriptive. No runnable cloud
request or observed mount is claimed by this local pair. Audit147 and the
original incomplete catalog remain byte-identical.
Only TRAIN/VAL asset payloads may be acquired or materialized. TEST7's mesh
descriptor is metadata only; its path must remain unopened.

The preflight rejects a complete proposed run whose conservative feature,
label, metadata and atomic temporary envelope exceeds 8 GiB. Its per-active-state
allowance is 16 MiB plus one 64 MiB margin for the global tree. This is an enforced
planning allowance, not a measured size forecast. It cannot establish the
1000/2000/4000 usable TRAIN targets; unmet targets remain explicit. The combined
provider quota separately counts raw outputs, runtime, both fixed source/model
caches, feature tree, temporary files, logs and training outputs. Hardlinked
files count once; device IDs do not establish independent provider quotas.

## Preparation requests and stages

All request JSON bytes require an externally recorded SHA256. Pin paths with
`quality_adaptation_portable_prepare.pin`. The request names `source_root`,
`source_files` for the three new production modules, the frozen `adapter` script,
`source_install`, exact existing `python`, `runtime`, `context`, `raw_seal`,
`exposure_audit`, `execution_root`, `features_root`, `input_root`, `working_root`,
`limits`, `quota_roots`, `storage_limit_bytes`, and `neural_inputs`. The latter
uses the original adapter's role/file checkpoint descriptors for ROOT18's
existing checkpoint. No cache remapping, acquisition or installation occurs.
Include working and the actual isolated runtime root in `quota_roots` and include
all source/model/preparation trees under that combined quota. The source package
must be clean of bytecode and use the raw context's actual source/cache paths.
The raw output, scratch and fixed model-cache roots must also be included.
The two fixed GoTrack loaders must share the same authenticated checkpoint file
through a real hardlink; duplicate checkpoint copies are rejected. No cache
symlink or loader remapping is accepted.

Preparation evidence has schema `kaggle_preparation_evidence_v2`: `provider`,
`session_id`, `observed_at`, `expires_at`, exact `raw_seal`, `context` and
`source_files`, selected `gpu: {uuid, name}`, and actual free `quota` observations.
Quota fields are `paid: false`, `gpu_enabled: true`, `gpu_remaining_seconds`,
`session_remaining_seconds`, `storage_scope` equal to `quota_roots`, and
`storage_limit_bytes`. The evidence is externally reviewed; hashes preserve its
origin rather than proving an operator declaration. GPU identity is checked
against `nvidia-smi`, and evidence expires within one hour.

Run each stage in a fresh process in the approved Kaggle runtime:

```bash
python -B -m bench.quality_adaptation_portable_prepare identity --request PREP.json --request-sha256 SHA
VISUALIZEIT_CLOUD=1 python -B -m bench.quality_adaptation_portable_prepare extract --request PREP.json --request-sha256 SHA
python -B -m bench.quality_adaptation_portable_prepare seal-features --request PREP.json --request-sha256 SHA
VISUALIZEIT_CLOUD=1 python -B -m bench.quality_adaptation_portable_prepare labels --request PREP.json --request-sha256 SHA
```

`identity` verifies the actual raw seal, including every failed and sealed state.
It emits per-state `kaggle_input_identity_v2` bytes binding recipe identity,
actual observed descriptors, RGB hash and distinct crop/input availability.
Generation/pose/label data is absent. Only `inputs_sha256` changes in a copied
prospective inventory. The same original seed/settings reconstruct the execution
plan; all other state fields and exact split bytes remain unchanged. The raw
closing stage receipt is authenticated separately because it is published after
the raw seal. Original raw and prospective files remain immutable.

`extract` runs exactly one unchanged authenticated ROOT18 extractor over the
available TRAIN/VAL crop manifest. The original CUDA bootstrap enforces the same
deterministic, no-TF32 policy as capture and writes actual runtime evidence. The
shared exclusive stage lock prevents concurrent neural stages. A fresh output
directory is required. Failed extraction attempts preserve receipts and rows.

`seal-features` reconstructs crops from retained depth/estimated camera geometry
using `crops_from_retained_capture(preserve_float_rgb=True)`, validates retained
camera transforms/native/crop calibration and sampling evidence, and compares
actual frozen flow/confidence against extracted outputs with the existing
numerical parity validator. At least one real parity row is required. It seals
actual artifacts and availability; caller booleans cannot substitute for parity.

`labels` launches a fresh supervised renderer worker. Before the first label
plan read, the worker recomputes the full prelabel seal. Supply `label_plan` with
schema `kaggle_label_plan_v2`, exact original `split_manifest` descriptor, and
`generations: [{sequence_id, generation}]` covering active rendered sequences
only. Existing normalized24 generation validation binds mesh/source/video/native
geometry/split. The unchanged renderer supervision and projection functions
produce labels separately. Good crops/features survive later label failures.
The unchanged ROOT18 assembler creates `correspondence_head_dataset_v1` with all
frozen nondev IDs, including unavailable sealed TEST. Failure inventories stay
available if labels or assembly cannot finish.

## Prepared consumer, pilot and resume

Use `scripts.kaggle_prepared_v2.request_document(...)` to build a prepared-only
request from the actual artifact paths, runtime/config pins, full limits and
fresh execution evidence. It hashes existing metadata; it makes no readiness
claim. The consumer completes the full metadata join, including the last sealed
state, before opening feature/crop/label payloads. Paths, inode aliases and
cross-split content hashes are checked. It distinguishes available crops,
available features and jointly usable training rows. Ordinary observed failures
count against availability; omitted accounting, changed source/provenance,
parity errors, leakage and insufficient quota block execution.

Consumer evidence is `kaggle_execution_evidence_v2` with the same provider/session,
time/GPU/quota fields and `prepared_identity_sha256`. The immutable prepared
identity binds source/runtime/config, raw/prelabel seals, execution/identity
bridge, original/final receipts, labels and canonical global dataset. Its hash
can be obtained by `metadata_join(request)['identity_sha256']` before attaching
execution evidence; evidence and output paths are deliberately outside that
data/config identity. They are checked independently on every launch.

Profiles are `{kind: "bounded_smoke", train_states_requested: N}` with N >= 1,
or `{kind: "target", train_states_requested: 1000|2000|4000}`. Requests count
actual jointly usable TRAIN states, never requested or cropped-only states.
A tiny integration smoke makes no claim of exceeding the earlier 16 TRAIN crop
experiment. Request at least 17 usable TRAIN states explicitly for that next
experiment. At least one usable validation state is always required.

A pilot request has schema `kaggle_pilot_request_v2`, `training` equal to the
prepared request, pinned original `pilot_helper`, `stop_after_steps` in 1..4,
and `full_run` with the same prepared identity and a separate fresh output.
The pilot's wall/RSS/output limits are <=900 seconds, <=8 GiB and <=2 GiB, with
at least 1 GiB reserve. It calls the genuine unchanged trainer's
`--stop-after-steps`. Original weights-only inspection reconciles actual optimizer
steps, checkpoint identity, RNG/scaler/optimizer state, epoch and data cursor.
A clean exit 3 means incomplete training with measured real updates. It does not
establish model quality. No pilot checkpoint is promoted into the full run.

The frozen inspection helper owns its own non-nestable interrupting timer.
V2 work between inspections uses separate timer contexts with the same original
absolute deadline; no segment resets the allowance. Timers and prior handlers
are restored on success and failure. Cleanup retains its bounded join allowance,
and a cleanup failure invalidates measurement and full-run admission.

The pilot retains actual wall/RSS/storage observations. A prefix linear forecast
is explicitly unrepresentative of full epoch/validation cost. For bounded smoke,
the full-scope bound is the enforced whole-global-dataset wall/RSS/output and
unchanged cumulative trainer cap; admission permits a bounded attempt without
claiming measured completion fit. It can interrupt and resume. Target mode
withholds full-run admission from prefix-only timing. No invented historical
timing or placeholder measurements are emitted.

```bash
VISUALIZEIT_CLOUD=1 python -B scripts/kaggle_pilot_v2.py --request PILOT.json --request-sha256 SHA
python -B scripts/kaggle_prepared_v2.py inspect --request FULL.json --request-sha256 SHA
VISUALIZEIT_CLOUD=1 python -B scripts/kaggle_prepared_v2.py preflight --request FULL.json --request-sha256 SHA
VISUALIZEIT_CLOUD=1 python -B scripts/kaggle_prepared_v2.py train --request FULL.json --request-sha256 SHA
python -B scripts/kaggle_prepared_v2.py monitor --request FULL.json --request-sha256 SHA
VISUALIZEIT_CLOUD=1 python -B scripts/kaggle_prepared_v2.py resume --request FULL.json --request-sha256 SHA
VISUALIZEIT_CLOUD=1 python -B scripts/kaggle_prepared_v2.py export --request FULL.json --request-sha256 SHA
```

Attach the actual `pilot-measurement.json` descriptor as `pilot_measurement` in
the full request. The prepared identity remains unchanged. Train/resume/export
reuse original commands and supervisor. Config/dataset/source/runtime identity
and every retained checkpoint/run file are checked before resume/export.
Export remains inside `train/export.pt` and requires the unchanged trainer's
completed, validation-selected checkpoint. A new export needs a concrete later
runtime/inference profile admission and replay against its actual bytes and
metadata. Existing profiles pin known checkpoints and cannot admit a new head by
substitution. The existing offline CLI's `--gotrack-head`/SHA256,
`--gotrack-inference-config`/SHA256 and `--gotrack-runtime-compat-config`/SHA256
are the eventual load boundary after that separate admission. This source makes
no load-success, deployment, mobile-readiness or 30 fps claim. `monitor` reports
recorded confirmed steps, never extrapolation.

Limits are sampled process/tree controls, not kernel-enforced instantaneous
peaks. OS termination may leave the existing stage lock; investigate it manually.
Preserve failed trials. Rollback consists of stopping the supervised process and
using a new run directory with the previously authenticated source request; do
not overwrite original raw/prospective files, historical archives or v1 evidence.
