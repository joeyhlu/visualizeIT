# Recorded SAM → metric tracking: reproducible setup order

This is the **finite recorded research** path: real RGB → observed SAM masks →
normal metric GoTrack tracking → sealed, post-inference dataset-reference
scoring. It is not stock `f58d532` serving, historical-receipt reconstruction,
LIVE/phone/frontend admission, automatic object discovery, or a quality claim.
Keep private weights/configuration, raw receipts and unreviewed results local.
The primary `27e06…` artifact already had paired full-window video testing on
October 21. The new 300-frame fresh-VM run is **additional evaluation**, not its
first video test. No completed full-300 result is claimed here.

## 1. Prepare the dataset separately

Follow the [dataset card](../datasets/offline-sam-pose-v1/README.md): acquire or
reuse the [licensed media release](https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1),
prepare **all 12 objects × 300 scored frames**, inspect the onboarding review
sheets, truthfully acknowledge that review, and verify the pack. Preparation
runs no model. RGB/model inputs and scored evaluator masks/depth/poses stay in
separate trees; never register the evaluator tree as an inference root.

Use a data-preparation environment with `bench/requirements-tracking.txt`, not
the separate NumPy 1.26 neural environment below. For one bounded full case:

```bash
REPO=$(git rev-parse --show-toplevel)
DATA_PY="$REPO/.venv-offline-data/bin/python"  # prepared as in the dataset card
PACK="$REPO/.cache/tracking-benchmark/offline-sam-pose-v1"
WS="$PACK/workbench-full-lossless"
CLIP=offline-hb-scene000001-object000020-full
"$DATA_PY" -m bench.tracking_offline_dataset register \
  --workspace "$WS" --object-id 20 --profile full --lossless-rgb
BUNDLE="$WS/metric-bundles/$CLIP.json"
```

Source frame 9 is the only authorized known-pose seed and annotation-assisted
reviewed point prompt; source frames 10–309 are scored. Native IDs/K/source axes are preserved,
GLBs use metres, and the clock is `frame_index`, not invented physical FPS.
Lossless PNG re-encoding must prove decoded pixel equality and count real bytes
under the unchanged **global 250 MB** native input budget. The one-object run
does not shrink the published 12-object dataset. Object 20 overlaps training;
neither this case nor reused references establish unseen-object generalization.

## 2. Restore an isolated historical worktree FIRST

The restoration helper **requires** a linked, detached worktree at the exact
historical HEAD. Do not run it in the benchmark checkout or an active run tree.
The benchmark source commit used here is the publicly verified
`29667146b5c746ea0cdd874f7f7a15f4f8897ece`; it must exist in the operator's
checkout. This 53-file commit is published on the benchmark branch. Fetch that
branch first; the **newer extras below need their own final commit pin**.

```bash
HISTORICAL_HEAD=f58d532bebd8700c07fc3f529b18bd646bd14a73
BENCHMARK_COMMIT=29667146b5c746ea0cdd874f7f7a15f4f8897ece
NEURAL="$REPO/.cache/tracking-benchmark/repro-neural"  # NEW, not an active tree
PRIVATE_HEAD=/operator/local/head.pt
PRIVATE_BEST=/operator/local/best.pt
git -C "$REPO" cat-file -e "$BENCHMARK_COMMIT^{commit}"
git -C "$REPO" worktree add --detach "$NEURAL" "$HISTORICAL_HEAD"
mkdir -p "$NEURAL/tools"
git -C "$REPO" show "$BENCHMARK_COMMIT:tools/restore_tracking_neural_sources.py" \
  > "$NEURAL/tools/restore_tracking_neural_sources.py"
(cd "$NEURAL" && python tools/restore_tracking_neural_sources.py \
  --head "$PRIVATE_HEAD" --best "$PRIVATE_BEST" --download-public-base)
```

Alternatively supply the exact authenticated public base with `--base` instead
of downloading. See [restoration/runtime instructions](tracking-neural-diagnostic.md)
for weight sizes/hashes, noncommercial GoTrack licensing and disk guards. The
helper restores exactly these historical CRLF-byte identities **before** the
new workbench port is overlaid:

| Restored historical file | SHA-256 |
|---|---|
| `bench/runtime/adaptation-sources.json` | `44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33` |
| `bench/quality_gotrack.py` | `0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29` |
| `bench/quality_runner.py` | `bca306d1342dce2566c786fed94a7564ae090e11bf42243da2ea7bc9ce841ad1` |

Do not globally normalize line endings, replace these restored files with LF
copies, fabricate old receipts, or change the protected installer.

## 3. Overlay the ACTUAL fresh-recorded port, not only tools

**Copying the runner alone is insufficient.** The restored stock workbench does
not contain `gate artifact` or the explicit fresh-recorded proof propagation.
Copy all three tracked port files from the pinned benchmark commit:

| Port file at `29667146…` | SHA-256 |
|---|---|
| `bench/workbench_neural.py` | `d4ce4297d94366c911316bc0df874cc9f33654585cfbe0522c8286f3314f382b` |
| `bench/workbench_cli.py` | `3e1e15180947acf1174554f035870c992b9927133396951b2ebb6786f92ec64e` |
| `bench/workbench_live_pose.py` | `ab60f59c039e7a9c866da2f6257502293dd18401432f27f74a54567c02dfde8b` |

```bash
for path in bench/workbench_neural.py bench/workbench_cli.py \
  bench/workbench_live_pose.py \
  bench/runtime/requirements-tracking-neural-cpu.txt \
  bench/runtime/requirements-tracking-sam-cpu.txt \
  bench/runtime/tracking-neural-cpu-wheels.json \
  bench/runtime/tracking-sam-cpu-setup.json \
  bench/test_workbench_recorded_artifact.py; do
  mkdir -p "$NEURAL/$(dirname "$path")"
  git -C "$REPO" show "$BENCHMARK_COMMIT:$path" > "$NEURAL/$path"
done
git -C "$NEURAL" rev-parse HEAD  # STILL exact f58d532…; deliberately dirty port
(cd "$NEURAL" && sha256sum bench/workbench_{neural,cli,live_pose}.py)
```

Record both the immutable historical HEAD **and the dirty local port/source
hashes**. Do not describe that tree as unmodified stock f58. The normal fresh
proof and subsequent receipts bind the actual current bytes, packages and paths.

Install the pinned Python 3.12 CPU environment using the
[GoTrack dependency/build steps](tracking-neural-diagnostic.md#cpu-runtime),
then restore exact SAM 2 sources/checkpoint and install the
[complete SAM CPU lock](tracking-sam-cpu-runtime.md). Retain verified source and
wheel hashes; no CUDA extension or unpinned dependency upgrade is required.

**Publication boundary:** the new generic supervisor
`tools/tracking_native_recorded.py`/its test, the post-seal
`bench/tracking_offline_evaluate.py`/its test, and
`tools/tracking_recorded_seal.py`/its test are newer reviewed work, **not files
supplied by 2966714**. Pin their separate final commit once published:

```bash
EXTRAS_COMMIT='<final published commit containing these reviewed extras>'
for path in tools/tracking_native_recorded.py tools/test_tracking_native_recorded.py \
  tools/tracking_recorded_seal.py tools/test_tracking_recorded_seal.py \
  bench/tracking_offline_evaluate.py bench/test_tracking_offline_evaluate.py; do
  git -C "$REPO" show "$EXTRAS_COMMIT:$path" > "$NEURAL/$path"
done
```

Until that commit exists, an operator-reviewed local snapshot with recorded
hashes is required; a clone of 2966714 alone does not contain these commands.
No private checkpoint, raw configuration or historical receipt is supplied by
either overlay. Copy the extras **before** earning proof/gates and freezing a
new run; do not edit an active worktree to update them.

## 4. Produce actual SAM masks, then an actual-SAM onboarding case

From the restored/ported tree with its CPU `.venv`, use the
[bounded native supervisor](tracking-native-recorded-runner.md). All heavy
SAM/geometry/parity/tracking stages run serially; never overlap speed trials.
All `EVIDENCE` outputs must be fresh directories.

```bash
cd "$NEURAL"
PY="$NEURAL/.venv/bin/python"
SUP="$NEURAL/tools/tracking_native_recorded.py"
EVIDENCE=/operator/local/new-offline-evidence
MASKS=sam2-observed-full-v1
CASE="$WS/cases/actual-sam-f000009.json"
ADMISSION=/operator/local/new-recorded-admission
mkdir -p "$ADMISSION" "$(dirname "$CASE")"
"$PY" "$SUP" --output "$EVIDENCE/SAM" --budget-profile recorded300 -- \
  --workspace "$WS" segment "$CLIP" --mask-set "$MASKS" \
  --mode regression --device cpu --verify-hash
"$PY" -m bench.workbench_cli --workspace "$WS" gate case "$CLIP" \
  --frame f000009 --mask-set "$MASKS" --metric-bundle "$BUNDLE" --out "$CASE"
```

Never substitute scored annotation masks, rendered silhouettes, cached feature
proxies or the dataset's deliberately blocked case template for observed SAM.

## 5. Earn fresh-local recorded proof AND all normal gates

Follow the exact [fresh-recorded lineage commands](tracking-recorded-artifact.md):
`gate artifact` → `gate contract` → `gate geometry --production` → full-image
reference/export `gate parity` → `gate verify`. Use the actual case above and
the exact local private head/best. Every relevant contract/parity/verify/run
command must explicitly receive the same `--recorded-artifact PROOF.json`.
Geometry is artifact-independent and has no such flag. Wrap heavy gates in the
supervisor's 900 s gate profile; do not waive production renderer/crop/parity checks.

Choose either the actual matching private `--training-config` or explicit
`--training-config-metadata-only` when only its authenticated checkpoint identity
is available. The latter does not recreate or verify unavailable raw config
bytes. A fresh proof says historical receipts were **not** verified and grants
no execution admission itself. Omitted proof selects the strict historical
default, not this new lineage. Never invent the old small-records/backup/loader
receipts, skip base hashing, or weaken pins. Package/path/source changes require
new proof and newly valid downstream evidence.

## 6. Run the NORMAL combined comparator, cache enabled-but-cold

```bash
PROOF="$ADMISSION/artifact-proof.json"  # genuinely produced by gate artifact
CACHE_BEFORE="$EVIDENCE/cache-before.json"
"$PY" tools/tracking_recorded_seal.py inventory --workspace "$WS" \
  --clip-id "$CLIP" --mask-set "$MASKS" --output "$CACHE_BEFORE"
"$PY" "$SUP" --output "$EVIDENCE/combined" --budget-profile recorded300pair -- \
  --workspace "$WS" run "$CLIP" --mode regression \
  --roles primary_neural,frozen_baseline --masks "$MASKS" \
  --metric-bundle "$BUNDLE" --head "$PRIVATE_HEAD" --device cpu \
  --admission "$ADMISSION" --recorded-artifact "$PROOF"
```

Unchanged native `RunSpec` requires baseline to accompany primary: ONE combined
job, 300 scheduled frames/role, 600 rows internally sequential. No standalone
baseline, fake/duplicate primary or role-contract relaxation. Pair wall ceiling
is 5,400 s; RSS 7 GiB, output growth 512 MiB and free disk 10 GiB guards remain unchanged.
Cache is **ENABLED-BUT-COLD**, not disabled: retain pre-run inventory, native
MISS/cold telemetry and genuine module-hook counts matching the SUM of actual
refinement iterations. A cache hit invalidates inference timing evidence.
After hard mask/solver loss, no automatic reacquisition/GT reseed is added;
cheap state-only lost rows are not faster model inference.

## 7. Seal terminal predictions, then explicitly score reused development

Only after the complete combined job successfully terminates, use the public
label-free sealer to bind all 600 rows, run/report/stages, exact schedules and
observed mask manifest, source/input freezes and supervisor records:

```bash
RUN_ID='<actual native run ID from the completed job>'
SEAL="$EVIDENCE/combined-prediction-seal.json"
"$PY" tools/tracking_recorded_seal.py seal --workspace "$WS" \
  --run "$WS/runs/$RUN_ID" --stage-record "$EVIDENCE/combined" \
  --cache-inventory "$CACHE_BEFORE" --output "$SEAL"
```

The inventory may contain unrelated prior cache entries: global emptiness is
not required for later objects. The sealer instead requires actual native
MISS/cold telemetry, exact observed hook counts versus iterations, terminal
success, frozen sources, actual installed head states and label-free input
bindings. The default identity is anchored to the public runtime config and
published diagnostic state hashes; an optional `--identity-pin` needs an
independently supplied `--identity-pin-sha256`. No weights, scored references,
model construction or admission are performed. All outputs are new immutable
files; raw operator receipts remain local.

Communicate the printed SHA-256 independently before scoring. One seal containing BOTH roles completes the
cohort; no companion seal is needed. Retain failures/partial outputs without
inventing a successful seal or changing settings silently.

The v2 seal uses `schema: offline_recorded_prediction_seal_v2`,
`evaluationProtocol: expanded_development_reused_reference`, `evaluated:false`,
`prior_reference_access:true`, `inference_process_scored_GT_access:false`,
`scored_GT_feedback_to_inference:false`, `weights_math_thresholds_pinned:true`,
`comparative_scoring_requires_all_role_seals:true`, and
`expectedComparativeRoles:[primary_neural,frozen_baseline]`. **Omit** the old
`scoredGroundTruthOpenedBeforeSeal` field: earlier geometric/preparation/pilot
work already accessed these references. This is expanded development with
inference isolation, not an independent held-out test or globally unopened GT.

```bash
"$PY" -m bench.tracking_offline_evaluate \
  --workspace "$WS" --run ACTUAL_COMBINED_RUN_DIR \
  --seal "$SEAL" --seal-sha256 EXTERNALLY_COMMUNICATED_SHA \
  --allow-reused-reference-development --pack "$PACK" \
  --reference-root "$REPO/.cache/tracking-benchmark/real-v1" \
  --stage-record "$EVIDENCE/combined" --sam-record "$EVIDENCE/SAM" \
  --destination /operator/local/new-private-evaluation
```

Missing/lost/recovering outputs remain failures in all-frame denominators;
conditional pose errors are separate. Report native 5 px corresponding-model
reprojection/ADD/rotation/translation and mask IoU as dataset-reference proxies,
not human attachment accuracy. Native `end_to_end_latency_ms` excludes cached
SAM and model loading; summed stage times are not measured live-stream latency.
Existing measurements retain their actual runner/source hashes; this entry does
not relabel old private-runner measurements as executions of the newer generic
supervisor. No new public result release is authorized by these instructions.
