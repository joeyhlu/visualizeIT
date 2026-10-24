# Sharded GoTrack training preparation

This workflow prepares one global dataset from the completed smoke dataset and
three newly rendered RAW shards. The original RAW, feature, label, dataset and
trainer implementations remain the numerical authorities. The sharded modules
freeze ownership and role projections, aggregate genuine child outputs, and
bind each stage to its source, provider evidence and resource limits.

## Preparation sequence

1. Build the full seed-521, iteration-zero recipe and its exposure audit. Call
   `bench.quality_adaptation_sharded_v1.freeze_partition(...)` with those
   objects and the imported smoke context/seal pins. This freezes the full
   train/validation/sealed-test plan, assigns each globally owned state once,
   and records the duplicate validation rows as non-owning child context.
2. Run the original RAW producer stages for each new child recipe. Call
   `seal_shard(...)` only after that child's original RAW seal is complete.
   `aggregate_raw(...)` accepts the four frozen references and writes a new,
   separately identified aggregate RAW root. It preserves child lineage and
   excludes sealed-test payload descriptors.
3. Create each new child request with
   `scripts.kaggle_sharded_prepared_v1.child_request_document(...,
   label_plan=None, defer_label_plan=True)`. Advance
   one stage at a time: `child-identity`, `child-extract`,
   and `child-seal-features`. Then run `child-bind-labels --output BOUND_REQUEST`
   with the sealed request and its SHA. This verifies actual numerical sealing
   before creating any generation descriptors, writes the plan in its frozen
   control-directory slot, and binds that plan to the unchanged input identity.
   Use the returned bound request for `child-labels`. Before each later stage, refresh
   actual provider evidence with `refresh-child` and that next stage name.
   Finalize and metadata-verify the request with `child-finalize` and
   `child-verify`.
4. After RAW aggregation, call `prospective_preparation_identity(...)` with the
   frozen partition, child requests, paths, and initial UI observation. It
   validates the same normalized metadata as publication and writes no files.
   Use that identity to create the initial execution-evidence document, then
   call `preparation_request_document(...)` once to validate its evidence and
   publish the immutable binding. Advance through `global-identity`,
   `global-extract`, `global-seal-features`, and
   `global-labels`, refreshing provider evidence before every later stage.
   The global numerical prelabel seal must complete before child labels are
   inspected or joined. Then run `global-finalize`, `global-verify`, and
   `inspect`.
5. `inspect` reports row availability and the fixed admission gates. Training
   is admitted only with at least 240 usable training rows, all 13 frozen
   training objects represented with at least 12 usable rows each, and at least
   one usable reused-development validation row. Validation outcomes do not
   choose rows or thresholds. The 24 validation views reuse one development
   object; availability and observational ambiguity remain visible in the
   report.

## Request refresh and command form

Every CLI operation takes the current request file and its exact SHA-256. A
refresh writes a new request file; use the returned pin for the following
stage. UI context and execution evidence must describe the current provider
session and cannot reset prior time, GPU or quota observations.

```sh
python -E -s -B -m scripts.kaggle_sharded_prepared_v1 refresh-global \
  --request global-request.json \
  --request-sha256 CURRENT_REQUEST_SHA256 \
  --output global-extract-request.json \
  --evidence-stage extract \
  --ui-context fresh-ui-context.json \
  --execution-evidence fresh-execution-evidence.json

python -E -s -B -m scripts.kaggle_sharded_prepared_v1 global-extract \
  --request global-extract-request.json \
  --request-sha256 REFRESHED_REQUEST_SHA256
```

Use `refresh-child` with child stage names for child requests. `child-label-worker`
is an authenticated internal subprocess entry point; invoke `child-labels` for
the supervised stage. `child-finalize` and `global-finalize` require
`--output`; `child-bind-labels` also requires `--output` and is used before the
labels evidence refresh. Existing pinned-plan requests keep their original
identity behavior. `child-verify`, `global-verify`, and `inspect` print compact JSON
summaries rather than serializing internal filesystem metadata.

Run the CLI through the reviewed outer launcher, which authenticates the source
package and supervises the complete operation. Keep Python bytecode writing
disabled (`-E -s -B`) so a metadata-only invocation cannot dirty the pinned
source tree.

## Training and evidence limits

The proposed full configuration is FP32, batch size one, three epochs and up to
936 optimizer updates. A separate pilot is capped at four updates and writes to
its own output directory. Initial full training uses `train`; interruption
recovery uses `resume`, and export uses `export`. The full dataset and run
identity stay pinned across those stages.

The focused software fixtures use analytical arrays and mocked rendering,
extraction or provider observations where stated in their tests. They verify
metadata, ordering, ownership, copy/link provenance and resource gates; they do
not demonstrate neural training quality, model generalization, or successful
cloud execution. A valid prepared dataset and passing admission checks are
required before any training claim.

The expanded quota scans the declared writable and retained source, cache,
runtime, RAW, scratch, feature and support roots. `input_root` remains available
to consumers as a read-only boundary and is excluded from recursive residency
scans and write reservations, so mounted Kaggle inputs and sealed TEST assets
are not traversed by the quota watcher. The model-free global integration test
requires the Linux fixture roots supplied by its runner on distinct WORK and
feature-cache filesystems; other platforms run the remaining focused guards but
skip that filesystem-dependent path.
