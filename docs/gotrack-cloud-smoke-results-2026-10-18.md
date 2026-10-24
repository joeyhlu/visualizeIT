# GoTrack cloud smoke results — 2026-10-18

## Scope and data

This completed free Kaggle T4 run verifies the adaptation workflow. It is not a large-data quality result. The real released GoTrack DPT pose head (20,029,955 parameters) was optimized; SAM, FoundPose/CNOS and visual encoders remained frozen.

Training used 72 usable synthetic refinement crops from three HOT3D mesh identities (1, 6, 10), 24 views each. Development validation used 20 usable crops out of 24 requested views of object 4; four failures remain counted. These are 280-by-280 image examples, not videos. Object 4 has been repeatedly inspected and does not provide independent validation. Object 7 test payloads remain unopened. No YCB-Video data was used in this run.

The existing frozen inference components generated observed masks, pose seeds and correspondence features. Labels were attached separately after feature/geometry verification. No reference poses were supplied to candidate generation or the serving compatibility check.

## Measured training

Three epochs, batch size one, completed 216 optimizer updates. The same 72 examples repeat each epoch. A separate four-update pilot was not promoted. The main run was interrupted after four updates and resumed with optimizer/data-order state to update 216. Its trainer reports 94.114 seconds. Raw preparation took approximately 43.8 minutes; feature extraction, checks and labeling took another 6.4 minutes. Installation and debugging are additional session time.

The first AMP pilot failed on nonfinite gradients at zero updates. Its evidence was retained. A fresh FP32 run disabled AMP while retaining finite-gradient checks; numerical failures were not ignored.

| Epoch | Updates | Mean training loss | Adapted development EPE, pixels |
|---|---:|---:|---:|
| 1 | 72 | 18.152195 | 21.097919 |
| 2 | 144 | 17.752930 | 21.003629 |
| 3 | 216 | 17.533511 | 20.918034 |

The frozen-head development EPE was 21.209391 pixels. The descriptive difference is 0.291357 pixels (1.37%). EPE measures image correspondence displacement error; it does not directly measure final 6D pose or visible attachment accuracy. With only one reused validation object, no independent uncertainty interval or generalization improvement is supported. Some views remain badly wrong and some worsen after adaptation.

## Checkpoint and inference compatibility

The exported head is 80,148,566 bytes, SHA-256 `e699e9cb45f06d93a762a5251d23c2753c5884474c2ecbed9be29099cd9b8b62`.

An actual cloud probe constructed the full pinned frozen GoTrack network, loaded this export through `load_export_into_network`, and compared its head outputs with the saved PyTorch best head on one authenticated cached TRAIN input. Flow and confidence both had maximum absolute difference 0.0 (required atol/rtol 1e-5). No labels or test payloads were opened. The bounded probe completed in 17.689 seconds, sampled peak process-group RSS 3,439,980,544 bytes, peak CUDA allocation 942,072,832 bytes. The original probe import failure and corrected attempt remain separate.

This establishes checkpoint loading and cached-input head parity. It does not establish full recorded-video replay parity, quality improvement, mobile compatibility or end-to-end FPS. The single cached-head timing is not an FPS benchmark.

The checkpoint remains in Kaggle. Notebook QuickSave failed, so a separate private dataset now preserves the checkpoint, configuration, provenance and reports: `achitaan/visualizeit-gotrack-smoke-20261018-c216-e699e9cb`. The dataset UI confirms Private, version 1, 40 files (545.41 MB). A separate cloud readback verified the manifest and every payload's byte count and SHA-256: 545,394,217 payload bytes. This establishes checkpoint-package durability independently of the notebook save. Dataset arrays and the interpreter environment are not in that package.

## Remaining work

Run complete pose-level legacy/adapted sequential replays; finish preparing more objects under the frozen split and resource budget; evaluate independent test objects only after settings are frozen. Both private backup packages are complete and verified, as described below. No default pipeline promotion, Android deployment or 180-degree rotation-fix claim follows from this smoke run.

Actual local evidence: `.cache/cloud-prepare/kaggle-fp32-training-status-final-actual-20261018.json`, `kaggle-fp32-history-actual-20261018.json`, `kaggle-fp32-export-receipt-actual-20261018.json`, and `kaggle-export-parity-v2-actual-20261018.json`. The measured figure is `kaggle-fp32-training-summary-actual-20261018.png` in that same directory.

## All-view development diagnostic

A fresh bounded cloud inference/plot process completed in22.659seconds with all20usable validation views in fixed ID order. Fifteen views had lower mean correspondence error and five worsened. The recomputed means (frozen21.20939058px, adapted20.91803385px) reproduce the trainer curve. They remain descriptive measurements from one reused object.

Four actual unavailable views remain in the denominator:08 lacked a render bank or observed mask;10,14,15 lacked an estimated pose seed. Fine-tuning the head cannot recover frames that never reach it. Some nominally available views have very small visible support (102 or278scored pixels), so a complete assessment must inspect support/ambiguity alongside error.

Four sheets show all20available query/template crops and frozen/adapted error heatmaps using one fixed0–75pixel color scale. White pixels are unscored, not successful predictions. Per-view errors are plotted from a zero origin. The arrays were read for labels only after each prediction; no TEST payload was opened. Local files are `.cache/cloud-prepare/validation-examples-01-actual-20261018.png` through`04`, `validation-per-view-actual-20261018.png`, and `kaggle-validation-images-report-actual-20261018.json`.

The notebook QuickSave returned an HTML/JSON server error; a version 1 entry with Starting status is not evidence of durable data. The separate official KaggleHub 1.0.2 checkpoint upload and all-file remote readback both completed successfully. Original files remain intact; no model was downloaded to the laptop.

A separate private data/source upload completed in 34.84 seconds with 2,139 files and 1,786,373,806 payload bytes: `achitaan/visualizeit-gotrack-smoke-data-20261018-87598b0c`. It preserves synthetic TRAIN/VAL inputs, frozen features, labels and source/provenance, with TEST metadata only. A separate cloud process then verified the remote manifest and every payload's byte count and SHA-256 in 52.19 seconds. The dataset UI confirms Private. Pretrained base weights and the interpreter environment are excluded, so those still need their pinned restoration process in a new session. Neither archive was downloaded to the laptop.

## Split and next expansion

The split is by mesh object identity, rather than a random image percentage, to prevent nearly identical rendered views of one object crossing TRAIN and validation/test. The current smoke run has TRAIN objects 1, 6 and 10; reused development VAL object 4; sealed TEST object 7. Validation was evaluated after each epoch. The four validation failures remain counted, and all 20 usable views have diagnostic images.

The next implementation targets 312 requested TRAIN views across 13 objects (24 per object), preserving the original 24 requested development VAL views and 24 sealed TEST metadata rows. This intermediate expansion has not started training and is not the eventual 1,000–4,000-example diversity target. Additional views of the same mesh are correlated observations, not independent scenes. Preparation runs in bounded sequential shards to reuse the working inference components and fit cloud resource limits.

At the actual cloud observation on 2026-10-18 at approximately 22:40 UTC, the first expanded shard had rendered all 120 active views successfully and was building object banks on one T4. Its render worker completed in 104.263 seconds; driver time through rendering and admission was 119.402 seconds. The retained data, source and runtime occupied 14,630,265,009 unique bytes across the declared roots, within the combined 32 GiB cap. The observed working filesystem still had 6,123,139,072 bytes free. These are preparation measurements, not new training or quality results.

The expanded partition retains the original validation data as its only owned validation set. Validation views generated again for each shard support the unchanged inference preparation, but do not replace failed original validation examples. The training integration remains under independent review; no larger-run optimizer update is claimed. The exact preparation observation is recorded in `.cache/cloud-prepare/kaggle-expanded-group1-raw-status-actual-20261018.txt`.

The first expanded shard subsequently completed all 120 object banks (1,119.612 seconds in the bank worker), then stopped before loading SAM. The mask stage's physical storage check applied its 5,898,415,104-byte proposal reservation to the working filesystem, although proposals are written to the separate scratch filesystem. Working free space was 5,598,273,536 bytes; the additional 1 GiB safety reserve made this check fail. All successful render/bank outputs and the failed mask receipt are preserved. No expanded training has occurred. A destination-aware recovery adapter is being implemented and independently reviewed; it has not yet been deployed. The actual failure is recorded in `.cache/cloud-prepare/kaggle-expanded-group1-mask-failure-actual-20261018.txt`.

The recovery adapter subsequently passed all 11 tests on the actual Kaggle Linux CPU, with no skips or model imports, and received independent source review. Its retry driver started at approximately 23:28 UTC in the same free session. The initial observed stage imports the existing renders and banks into a fresh context using hardlinks, preserving the failed original run and its elapsed-time charges. Mask/capture completion and expanded training are still unverified at this observation. The original inference model operations remain unchanged; the new adapter assigns proposal reservations to scratch and separately reserves mask/control writes on working.
