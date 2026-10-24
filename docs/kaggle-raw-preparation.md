# Portable Kaggle RAW preparation

This source implements the boundary from a frozen metric-mesh recipe to retained
GoTrack input crops and estimated geometry. It stops at `raw-seal.json`. It does
not extract features, open labels, train, export a head, create a final execution
inventory, or declare training readiness. Local fixture tests establish software
behavior; they establish no mask, pose, or model accuracy.

The entry point is `scripts/kaggle_raw_prepare.py`; orchestration and contracts
are in `bench/quality_adaptation_portable.py`. The six stages are `freeze`,
`render`, `banks`, `masks`, `capture`, and `seal-capture`. Execute each expensive
stage in a fresh Python process, sequentially. No stage downloads or installs
anything. Only the parent organizer schedules actual neural execution.

Storage checks use no-follow metadata traversal and count each regular inode
once across overlapping roots. If an atomic render publication removes a name
during traversal, the monitor discards the incomplete sum and retries the
complete scan. Three unstable scans cancel the stage. Symlinks, permission
errors and other filesystem errors remain failures. The strict check after a
stage finishes remains in place. A source update requires a fresh authenticated
context and fresh output directories; failed runs and their original source
records must be retained.

## Inputs and immutable boundaries

The recipe has exactly these fields:

```json
{
  "schema": "kaggle_raw_recipe_v1",
  "assets": [
    {
      "object_id": "101",
      "physical_object_id": "independently-audited-object:101",
      "aliases": [],
      "mesh": {
        "path": "/kaggle/input/APPROVED_PRIVATE_DATASET/object101.glb",
        "bytes": 12345,
        "sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SHA256"
      }
    }
  ],
  "iterations": [0, 1, 3],
  "seed": 42,
  "targets": [1000, 2000, 4000]
}
```

The example is a schema illustration, not an executable inventory. Supply every
permitted independent asset before freezing. Declare source units honestly:
the existing renderer accepts metric GLBs only. Repeated identical mesh bytes
are rejected. Every asset receives the existing 24 distinct normalized views,
with one physical frame at timestamp zero per view. Refinement states are the
selected existing iterations 0/1/3; they are not additional physical frames.

`quality_adaptation_scale.plan_scale` supplies the grouping, development-object
exclusions, and deterministic split assignments. Shared objects, aliases, mesh
bytes and scene exposures stay together. Fewer than three independent nondev
groups is infeasible. Insufficient TRAIN coverage leaves each requested target
explicitly unmet; it does not cause repeated identical states to be generated.
TEST and development states stay metadata only, including during preflight.
Only procedural gray backgrounds are generated. No held-out observed background
or co-occurring object is used for training preparation.

The context is immutable JSON bytes authenticated by an external SHA256. Use
`context_document` to build a reviewable context from the actual installed
package; it computes the current wrapper closure and pins the unchanged
`bench/runtime/adaptation-sources.json`. This helper performs no asset/model
acquisition or execution:

```python
from pathlib import Path
import json
from bench.quality_adaptation_portable import context_document, pin, sha256

context = context_document(
    output_root="/kaggle/working/visualizeit-raw-run-001",
    scratch_root="/tmp/visualizeit-raw-run-001",
    parity={"reference": pin(Path("/kaggle/input/APPROVED_EVIDENCE/reference.json")),
            "candidate": pin(Path("/kaggle/input/APPROVED_EVIDENCE/candidate.json"))},
    device="cuda", wall_seconds=7200,
    rss_bytes=8 * 1024**3, storage_bytes=18 * 1024**3,
    reserve_bytes=1024**3)
Path("/kaggle/working/raw-context.json").write_text(
    json.dumps(context, indent=2) + "\n", encoding="utf-8")
print(sha256(Path("/kaggle/working/raw-context.json")))
```

The original context and recipe bytes, including pretty-printing and trailing
newlines, survive freeze exactly. A reviewed source package and externally
checked context hash remain prerequisites; computing one's own hash is not an
independent review. Changed wrappers are a separate source closure. The immutable
upstream revisions and checkpoint digests come from the original pinned lock
and existing model declarations. Existing serving contracts, checkpoints,
historical archives, and pilot/adapter pins are not resealed or changed.

The context must name the actual source root and its actual
`.cache/model-quality` directory. Existing GoTrack/CNOS/SAM loaders resolve that
directory; arbitrary cache remapping is rejected. Before neural imports,
preflight checks source closure, existing checkpoint hashes and descriptor lock,
pinned packages, source-only upstream trees, and actual array parity using the
existing parity comparator. Parity snapshots must include actual available rows;
equal failures do not establish parity. The selected device must match the
candidate evidence. Clean packages must contain no Python bytecode, including
upstream bytecode, so unverified cached code cannot override source pins. Use
`python -B` and a fresh process. No cleanup or acquisition is automatic.

The intended neural runtime is the separately prepared pinned Python 3.11 /
Torch 2.5.1 environment. The shared verifier also recognizes its existing Python
3.10 compatibility range. This source does not turn Kaggle's observed Python
3.13 CPU environment into that runtime. Actual runtime, GPU availability,
geometry/parity evidence, private assets and approved package upload remain
environment prerequisites.

## Stage execution

Freeze is metadata only and may be inspected locally without opening mesh
payloads. Use a fresh run directory. Record the exact recipe hash first:

```bash
python -B scripts/kaggle_raw_prepare.py freeze \
  --context /kaggle/working/raw-context.json --context-sha256 CONTEXT_SHA256 \
  --recipe /kaggle/working/raw-recipe.json --recipe-sha256 RECIPE_SHA256

VISUALIZEIT_CLOUD=1 python -B scripts/kaggle_raw_prepare.py render \
  --context /kaggle/working/raw-context.json --context-sha256 CONTEXT_SHA256 \
  --cloud-execution
```

Run `banks`, `masks`, `capture`, and `seal-capture` with the same context arguments
and explicit cloud flag, in that order. Production stages require Linux, the
Kaggle input mount, output below `/kaggle/working`, scratch below `/tmp`, the
shared existing stage lock, and the remaining cumulative budget. Direct calls to
the producer functions also require their guarded stage context. Stage attempts
derive required model preflight from the stage name; callers cannot opt out with
a boolean. Stage attempts
are exclusive: create a new prospective run after an attempted stage fails;
do not overwrite old receipts or silently resplit.

The renderer is the existing `_render_bundle(..., view_mode="normalized24-v1")`.
It writes native K, metric meshes and physical timing with renderer truth in a
separate generation sidecar. That sidecar is not part of the inference input
projection and is never opened by mask/capture/seal stages. Input parsing rejects
foreign annotation fields. The existing native principal point correction is
reused without new geometry mathematics.

Banks reuse `build_descriptors` and the standard 798-template `build_bank`.
Descriptor residency ends before pose-network residency. Masks reuse
`run_mask_phases`: FastSAM proposals, DINO/CNOS identity scoring, then SAM image
segmentation, with phase cleanup. Observed masks never come from a projected
silhouette. Failed renders/banks/masks remain failures. Capture uses actual
`PreparedFoundPoseSearch.recover(..., top_k=5)`, selects the existing first sorted
estimated seed, observes the actual selected crop sampling maps, and writes
retained arrays through the existing exact `RefStateCapture` type. A distinct
portable capability factory runs genuine input-only preflight itself. The
unchanged GoTrack constructor validates that capability and integer geometry.
Legacy capability defaults and source checks are preserved.

## Resource and failure behavior

All expensive stages share one cumulative wall allowance, a Linux process-tree
RSS monitor, physical storage accounting and the existing exclusive model-stage
lock. A stage timer interrupts Python execution; storage/RSS cancellation is an
explicit `ResourceError` derived from `BaseException`, so ordinary per-frame
`except Exception` handlers cannot absorb it. Callers supervising a stage should
handle it alongside interruption, retain receipts, and report the failed stage.
The controller attempts renderer, search-index, network/device and FAISS-worker
cleanup on success and failure. The unchanged CNOS descriptor builder creates
its renderer before entering its own cleanup block; a constructor/model-load
failure there remains a process-lifetime cleanup boundary. Run each stage in a
fresh process and validate actual GPU/OpenGL cleanup in the approved runtime.
A killed OS process can leave the shared lock behind;
operator investigation is required. No stale-lock stealing is implemented.

The budget conservatively counts output, scratch and model cache together even
when their device IDs differ. Files sharing a physical inode are counted once.
Distinct devices are not assumed to have independent provider quotas. Each
device must preserve its own free-space reserve. The model cache has an
additional 8 GiB limit. Metadata writes are atomic; model/data writes reserve
space in advance and a 64 MiB margin is retained for failure metadata. The
portable JSON reader and writer share a 16 MiB per-document bound. Reads reject
duplicate keys, nonfinite numbers and nesting deeper than 32; the existing
shared 8 MiB legacy helper is unchanged. A metadata document above 16 MiB still
fails explicitly, and storage admission can fail below that size. A failed
freeze can leave a partial run without a successful `freeze.json` receipt.
An earlier `.<name>.part` blocks a new atomic write and is preserved byte for
byte; cleanup removes only a temporary created by that writer invocation. The
existing crop writer enforces a per-file bound and total run bound. Watchdog
checks are sampled, so provider enforcement remains necessary for a hard
kernel-level limit during native model operations. Local tests use tiny storage
fixtures; they do not prove provider quota enforcement or neural peak RSS.

The full storage scan is intentionally conservative. Existing historical cache
trees can make accounting slow. Within one full input preflight, shared object
banks are hash-verified only once. A new preflight hashes again; no persistent
stat/hash shortcut weakens mutation detection. Capability creation and validation
still repeat full preflight, which may be costly at larger scale. Measure that
cost in the approved runtime before admitting a larger run; no throughput claim
is made by this source packet.

## Raw seal contract and next boundary

`physical-index.json` contains every frozen physical frame, including TEST/dev,
with separate `render`, `bank`, `mask` and `seed` states. `raw/capture-index.json`
contains every selected refinement state, including explicit failure reasons.
A retained crop may be available even when its subsequent PnP result is
unavailable; that distinction is preserved. Failed and interrupted stages retain
their requested inventory and cleanup diagnostics. Successful raw sealing does
not mean every frame succeeded.

`raw-seal.json` has schema `kaggle_raw_capture_seal_v1`. It binds the exact
context, prospective plan, actual observed input/mask/bank descriptors, estimated
seed records, crop/geometry artifacts and all failures. Its `seal_sha256` is the
canonical hash of the document without that field. `training_ready` and
`final_execution_inventory` are always false.

The prospective `inputs_sha256` values identify recipes only. A later producer
must replace only those values with the SHA256 of actual
`kaggle_input_identity_v1` bytes and prove that every other field and split
assignment is unchanged. It must join feature and later label availability
without dropping failed states. Existing v1 consumers currently require stricter
matching availability and may reject any failed preparation; this raw producer
does not forge readiness to satisfy them. Feature extraction, independent
prelabel sealing, separate label generation, the final consumer bridge and
actual cloud validation remain separate required work.

Historical research contracts must be reproduced from their original
authenticated archives. Their capture-module hashes will not match the new
portable wrappers. Never silently reseal those old profiles or results.
