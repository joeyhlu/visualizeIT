# Bounded first TRAIN resource measurement

`scripts/kaggle_pilot.py` adds a separate pilot admission path around the frozen
ROOT18 trainer. It obtains the first measured optimizer-step cost needed by
the existing scale and full-training gates. It does not change those gates,
select a smaller dataset, edit trainer parameters, or start a new training loop.

The path is: byte-authenticated pilot request → authenticated frozen adapter →
original installed source/runtime checks → original full global metadata join
at `preflight` → fresh free Kaggle GPU/session evidence and detailed parity/input
review → unchanged trainer `train --stop-after-steps N` → original status API plus
isolated, authenticated `last.pt` inspection → resource observations.

This is source-only implementation. No pilot has run on Kaggle. Authentic raw
data preparation, original provenance/receipts, complete global features/labels,
runtime bootstrap and actual geometry/token parity remain separate unfinished
prerequisites. The old larger controllers still contain `/teamspace` assumptions
and 312-state caps; this pilot does not port their data preparation. Phone-gated
GPU/Internet and provider availability must be checked by the parent before any
neural execution. Local import, help and tests never acquire or run a model.

## Immutable inputs and admission

The adapter source must be exactly SHA-256
`1823fc93ef95057a3f2e4928c3d43123f130ce719683260fdce4e5f58212b388`.
The pilot reads, hashes and compiles those same source bytes without using a
cached loader. The adapter authenticates its original ROOT18/29/30 source
manifests, runtime interpreter/bootstrap/requirements and pinned metadata helpers.
The old original source archive remains
`cbc1e698fce71f1be1561ac1033a1a7bbbe82fa1a8ddbd6050036ba56b6f8a7a`;
the reviewed 59 KB execution package remains
`3314861b6bcd8df1283f2f6332461eb655f02cc0206e0bbfb34c70d82eeb1ff8`.
Neither archive includes this new pilot source.

The nested `training` document is the full
`kaggle_adaptation_request_v1` described in [kaggle-training.md](kaggle-training.md).
Supply its original full config, dataset, input inventory, global plan, identity
bridge, crop/label metadata, preparation receipts and authenticated scale helpers.
`scale_admission`, preparation `measurements` and full-stage `evidence` need not
exist to obtain the first pilot observation. They are still mandatory when the
unchanged adapter later admits full training. The pilot calls the original
`run_metadata_check(request, 'preflight', root18)` and never calls
`validate_request` with invented timings or substitutes admission functions.

The complete original global join rejects split reassignment, missing failures,
unrelated feature/dataset/label identities, unauthenticated original provenance,
and active inputs that alias sealed TEST/development payloads. TEST arrays remain
unopened. The unchanged trainer also runs its original full preflight, TRAIN EDA
and feature checks. The leakage question is unchanged: can training contain
indirect information about a held-out object, scene, sequence, image or feature?
The original grouping and exposure checks answer it; the pilot cannot resplit.

Each file descriptor is exactly `path`, `bytes`, `sha256`, where the digest
authenticates the file's original bytes. Paths are absolute and must remain
inside the explicit Kaggle input/working roots. No fixture/subset manifest or
parameter replacement is supported. `pilot_output_root` must be fresh and
disjoint from installed source, isolated runtime and eventual full-run output.
The full trainer directory must also be fresh. Pilot retries need a new output
and UUID; failures are retained.

## Request and execution

Construct a JSON object with these fields. This is a field inventory, not a
runnable evidence fixture; use actual descriptors and observations.

| Field | Value |
|---|---|
| `schema` | `kaggle_train_pilot_request_v1` |
| `pilot_id`, `full_run_id` | Distinct canonical UUIDs |
| `adapter` | Exact frozen adapter source descriptor |
| `training` | Unchanged complete full adapter request |
| `pilot_output_root` | Fresh dedicated child of working root |
| `stop_after_steps` | Integer 1–4 optimizer steps in this invocation |
| `train_states_requested` | An existing target in the original frozen plan |
| `storage_limit_bytes` | Current observed working storage quota, at most 128 GiB |
| `limits` | Fields listed below |
| `evidence` | Actual pilot evidence JSON descriptor |

`limits` contains exactly `max_seconds` (positive, at most 900),
`max_rss_bytes` (at most 8 GiB), `max_output_bytes` (at most 2 GiB),
`max_log_bytes` (at most 64 MiB and below output cap), and `reserve_bytes`
(at least 1 GiB). The original trainer config output cap must fit the pilot
output cap. Its full config remains unchanged, including `max_steps`, epochs,
batch size, seed and optimizer. Initial admission runs before the execution
timer, using the adapter's finite subprocess timeouts. Once that timer starts,
inspection, launch revalidation, training, checkpoint/status hashing and late
export all share the same absolute deadline. Export never starts a fresh budget.

On the admitted Linux main thread, an interrupting `SIGALRM` timer bounds late
hashing, filesystem and revalidation calls as well as the frozen metadata and
GPU-query subprocesses. A subprocess interrupted through `subprocess.run` is
killed and reaped; neural children retain the original process-group supervisor.
The timer/handler are restored afterward. An existing active real-time timer or
an unsupported host/thread rejects the operation rather than silently running
an unbounded exporter. Windows storage fixtures explicitly mock this Linux-only
mechanism; the real interruption/cleanup control runs under Linux/WSL.

If genuine measurements have already been confirmed and late export exhausts
the shared deadline, the result stays `measurement_complete` with its actual
observations, while `admission_records_emitted` is false and the reason records
the deadline. Expiry before checkpoint/status reconciliation cannot establish a
successful measurement. Process cleanup/reaping and final failure/observation
receipt persistence still run after body expiry; their elapsed time is included
in `total_observed_seconds`. The wall cap bounds productive execution, not a
real-time guarantee for OS scheduling, signal delivery or final retention I/O.

Current resident working bytes, actual free storage, the output cap and reserve
must fit both free space and declared storage quota. No prior step timing is
required. Linux process-group RSS, output bytes, logs, wall time and filesystem
reserve are supervised using the frozen adapter. These are polling controls,
not operating-system cgroup limits; brief overshoot between checks is possible.
Only one process group is launched at a time, including checkpoint inspection.

Pilot evidence has schema `kaggle_train_pilot_evidence_v1`, provider `kaggle`,
stage `pilot`, `session_id`, `evidence_source`, canonical public `notebook_url`,
`observed_at`, `expires_at`, `identity_sha256`, and `plan_sha256`. Observations
must be at most five minutes old and expire within five minutes. `quota`
contains `paid: false`, `gpu_enabled: true`, and actual positive
`gpu_remaining_seconds`/`session_remaining_seconds` sufficient for the pilot cap.
`gpu` identifies its actual `uuid` and `name`; live `nvidia-smi` checks those
values and observes positive free/total memory. The exact UUID becomes
`CUDA_VISIBLE_DEVICES`. CPU fallback is rejected.

Also require `parity_passed: true`, `authentic_inputs_reviewed: true`,
`test_arrays_opened: false`. The nested `training.inputs` contains exactly one
`parity_report` and one `input_review` descriptor. Each report must contain
`passed: true`, `test_arrays_opened: false`, `dataset_sha256` and
`config_sha256` matching those **file byte descriptors**, `plan_sha256` matching
the original plan seal, and a nonempty `checks` array. Every check names itself,
has `passed: true`, and retains `observed` and `expected` values. Supply actual
detailed reports; hashes and operator assertions alone do not prove parity.
Generating authentic parity reports is a separate prerequisite.

To export legacy full-run resource records, `quota` additionally carries the
current provider `storage_limit_bytes` equal to the request quota and
`storage_scope` equal to the absolute working root. The existing observation
must still be fresh after the pilot and its GPU/session seconds must cover the
full request's wall cap. Missing or expired full-run quota evidence withholds
the records while preserving a successful pilot's observations.

Build the evidence identity with `seal(pilot_identity(adapter, request,
verified_runtime))`, using the authenticated adapter and actual verified runtime.
It excludes the evidence descriptor to avoid a circular hash. Then save the
evidence, set its descriptor, save the request, and compute SHA-256 of the final
request bytes. The request binds both full-run identity and dedicated pilot
identity, including all paths, original dataset/config descriptors and caps.

On an authorized verified Kaggle GPU session, the parent may invoke:

```sh
python -B /ABSOLUTE/REVIEWED/kaggle_pilot.py \
  --request /ABSOLUTE/ACTUAL/pilot-request.json \
  --request-sha256 ACTUAL_SHA256_OF_REQUEST_BYTES
```

There is no install, download, cloud start, upload or resume side effect in this
CLI. Execution requires the existing isolated cloud interpreter in the request.

## Success, failures and measurement scope

A successful **measurement attempt** requires supervisor exit 3 with the original
trainer's `incomplete` / `invocation_step_limit` status. The final `last.pt` must
match its authenticated file hash before/during/after a `weights_only=True`
CPU load inside the verified isolated interpreter. That child uses the original
`TrainConfig` defaults/validator and dataset loader to derive canonical config,
dataset, source-lock and runtime identities. The checkpoint must agree with
those identities and the original status observation, retain resume state, and
have a positive optimizer step delta no greater than the cap. Actual per-parameter
optimizer step counters must agree too. Fractional counters cannot be truncated
into acceptance.

Zero progress, excess steps, stale status, identity mismatch, deadline, crash,
log/storage/RSS failure and unconfirmed checkpoint inspection never yield usable
measurement records. `pilot-result.json`, logs and original outputs are retained.
The result distinguishes `measurement_complete` from `trainer_status:
incomplete`; training and model quality are never marked complete.

The success result always contains actual observations and the separately
labeled forecast. `admission_records_emitted` reports whether the producer also
emitted `scale_measurements` with the existing `assess_admission` keys and
`stage_measurements` with the existing adapter stage evidence keys. Keep the
full result and request alongside any extracted records. Trainer invocation
seconds, authenticated steps and resident/free snapshots are observations.
Wall time includes startup, original preflight and checkpoint writing, making
the first few-step projection conservative in some respects, but it does not
measure later validation/checkpoint regimes or guarantee full-run fit.

Storage high-water samples are **lower bounds**, labeled `peak_is_exact: false`.
The enforced output cap is a separate field and is never called a measured peak.
RSS is process-group sampling (shared pages may count more than once); GPU peak
memory is not measured. `forecast` contains a separately labeled linear time
projection. Full-training admission remains false in the pilot result.

The original consumers ignore lower-bound provenance. The producer therefore
never exports the sampled lower bound into their storage-admission fields.
Before it emits either legacy record, code reauthenticates the original dataset,
canonical config and source lock, rejoins the full plan/receipts, checks its
original TRAIN-state target and full optimizer-step cap, and requires the
prepared inventory and eventual outputs to share the measured working quota.
The original trainer cap must fit three retained/atomic checkpoint allowances
and its metadata floor. Existing full-output bytes plus the full trainer cap,
full log cap and two adapter JSON allowances must fit the full supervisor cap.

The entire full supervisor `max_output_bytes` becomes the conservative
**additional output allowance**, without subtracting existing pilot/prepared
files. Code requires that allowance plus the greater pilot/full reserve to fit
both newly observed free bytes and current resident bytes plus fresh quota.
The legacy fields named `measured_additional_peak_bytes` and
`additional_peak_bytes` receive this bound because those unchanged consumers
have no separate upper-bound field. `full_run_allowance.is_measured_peak: false`
and `legacy_additional_peak_field_basis:
enforced_full_run_upper_bound_not_measured_peak` identify the substitution.
The sampled observation remains separate and is never relabeled a measured
exact peak. The deliberate overcount protects later best/last/atomic checkpoint
and log growth even when the first few steps wrote very little.

Insufficient headroom, changed identities, missing or stale full-run quota, an
inadequate output envelope, or a prepared inventory outside the working quota
withholds **both** legacy records. The result retains successful pilot
observations and explains `admission_records_withheld_reason`. Read-only input
mounts need separate original storage-scope evidence and an appropriate reviewed
producer; this implementation cannot infer their quota from working storage and
does not emit legacy records for that case. A completed pilot measurement can
therefore remain useful while full admission is blocked.

Emitted records still require the unchanged full scale and adapter gates, fresh
execution-stage evidence and a review of time representativeness. A short early
prefix does not measure later validation regimes or establish full-run fit.

The checkpoint is a standalone pilot artifact with the complete original
dataset/config identities and RNG/optimizer/cursor state. It is not automatically
promoted or adopted by the existing full-run adapter. A later explicitly reviewed
resume handoff must preserve those identities and prove compatible state/run
ownership; until then full training starts only through its unchanged gate.

## Local verification and deployment checks

Run `science-python -B -m unittest bench.test_kaggle_pilot -v`. Tests use tiny
manufactured metadata and opaque bytes, mocked neural child interfaces, and a
real benign stdlib subprocess for elapsed measurement and timeout cleanup. They
check frozen source/archive pins, no-Torch help/import, exact trainer argv,
caps/path/source/runtime/evidence rejection, checkpoint/config/status/optimizer
reconciliation, failure preservation and legacy scale-consumer compatibility.
The low-headroom regression uses the actual unchanged consumer: a sampled
64 MiB increment can fit while the enforced full-run allowance cannot; the
producer withholds both records and the consumer remains blocked. A separate
conservative case passes the consumer with the full bound, while excessive time
projection still blocks. Identity, quota/scope and checkpoint/log envelope
regressions retain observations without emitting consumable records.
The real Linux deadline regression gives export 50 ms while a benign metadata
child would take two seconds. The child is interrupted and reaped, the timer is
restored, observations remain available and no legacy records are emitted. An
in-process delayed validation control is also interrupted before completion.
These tests establish integration contracts, not training accuracy.

Before a real pilot, independently review the final source and packaging,
authenticate the actual Kaggle mount/runtime/data/parity, verify GPU/quota and
one-process scheduling, then observe the real incomplete checkpoint and final
shutdown. Before any deployment or quality claim, run representative full
training/resume/export and independent held-out evaluation with failures counted,
availability, validation-selected checkpoints and uncertainty. Real traffic
checks must still cover object/scene distributions, stale features, serving
thresholds, fallback and pose/render availability; this resource pilot changes
none of those product decision rules.
