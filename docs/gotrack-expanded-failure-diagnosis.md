# Expanded GoTrack: saved-output failure audit

## What this establishes

The expanded head remains experimental. The
[full-video comparison](gotrack-expanded-inference-results-2026-10-15.md)
found worse mean translation errors and lower availability despite a small
cached correspondence-error improvement.

The new `bench.quality_expanded_failure_diagnosis` command joins the sealed
candidate traces with the existing post-seal evaluation. It covers all 144
full-run records: two heads, three sequences and 24 requested frames each.
Suppressed poses and failures stay in the inventory. Four-frame causal repeats
do not enlarge the quality sample. The command performs no inference, training
or reference-sidecar reads and does not recompute pose errors.

## Why reranking cannot replace these three poses

The three largest adapted errors in this reviewed study each have exactly one
recorded candidate. That candidate was selected, emitted and admitted by every
recorded gate. Its before- and after-gate minimum candidate error equals the
emitted error; there is no better recorded alternative to select.

| View / role | Frame | Translation error, mm | Candidate attempts | Translation from previous valid pose, mm |
|---|---:|---:|---:|---:|
| Object 17, -25 / TRAIN | 9 | 100.181 | 1 | 99.530 |
| Object 17, +25 / TRAIN | 9 | 104.539 | 1 | 100.023 |
| Object 27, -25 / used VAL | 11 | 187.862 | 1 | 114.839 |

A ranker could potentially reject a sole candidate. This audit does not train
or validate such a rejection rule. It does establish that changing the order
of these recorded candidates cannot produce a better pose.

The fixed context windows use these study frame IDs and retain both heads'
records for two frames before and after each. They are exploratory development
windows, not a claim about independent test data or the maxima of future runs.

## Why the recorded gates admitted them

The authenticated physical-time policy permits translation changes of
`0.01 + 1.2 * elapsed_seconds` metres and rotation changes of
`5 + 720 * elapsed_seconds` degrees. Over 0.1 seconds those limits are about
130 mm and 77 degrees. The three recorded changes satisfy both limits.
These are comparisons with the previous accepted pose, not pose errors against
the reference. The policy was unchanged during this audit.

Backend motion checking was disabled. A recorded `backend_motion: true` in this
case is not evidence of an additional independent motion check.

The current-image validator also passed, with retained correspondence /
residual-validation-inlier counts of 1302/1165, 299/292 and 5760/4519. These
inliers count agreement with the candidate's native-image reprojection. They
are not OpenCV RANSAC inliers and do not establish independent pose correctness.
Observed mask availability likewise does not establish segmentation accuracy.

## Coverage and missing evidence

Across all 72 records per head, frozen runs have 66 single-attempt and 6
multiple-attempt frames; adapted runs have 64 and 8. Counting valid candidate
poses instead, frozen has 67 single, 4 multiple and 1 without a valid candidate;
adapted has 65 single and 7 multiple. Attempts can contain failed candidates,
so these counts measure different things.

The training archive separately verifies 16 TRAIN captures and 8 cached VAL
captures, all from frames 0-7 and refinement iteration 0. The three largest
replay errors occur outside that captured frame range. This is a data-coverage
gap worth investigating, not proof that it caused the failures. Object 27 stays
used validation; object 7 remains unopened test data.

The saved candidate traces do not contain per-iteration flow/confidence,
three-dimensional correspondence conditioning, RANSAC sample/inlier indices or
the PnP/LM solver trajectory. They explain admission, but cannot identify the
neural or solver mechanism that produced the bad pose. Appearance timing is
not appearance-quality evidence. Rotation accuracy, independent surface
attachment and recovery accuracy remain unmeasured.

The next useful probe is to collect those missing intermediate values for both
heads on identical images, observed masks and estimated seeds, then analyze
them after sealing. Reference poses must remain outside inference. Broader
training should follow that evidence; tighter motion limits or additional
epochs have not been validated by this audit. The frozen baseline is unchanged.

## Reproduce the offline report

With the existing small saved artifact package available locally:

```powershell
python -B -m bench.quality_expanded_failure_diagnosis `
  --sealed-root .cache/expanded-full-video-development-v1/sealed `
  --sealed-manifest-sha256 545922617b8f9b74affd0c84fe39bf473a9a26a6ab0f3568f94037f872809e7a `
  --sealed-archive .cache/expanded-full-video-development-v1/download.zip `
  --evaluation-policy .cache/expanded-full-video-development-v1/evaluation-policy.json `
  --evaluation-policy-sha256 06d725aafa3d15aac5bc1ced863b4367515971cb4918511acb0744d8cb60acd6 `
  --evaluation-report .cache/expanded-full-video-development-v1/evaluation-report.json `
  --evaluation-report-sha256 7e8a48f829bbe971cde8147242f0d5ed666c613329c26bdaf9a4e9b0446ac70f `
  --output NEW_REPORT.json
```

Use a new output filename. Input hashes, identities and consumed-member receipts
are checked before the report is built. Existing files cannot be overwritten.
The report copies the evaluator's translation errors, candidate minima,
distributions and availability denominators. Candidate minima are post-seal
counterfactuals within each head's candidate set, never replacement predictions.
Selected-minus-minimum attribution is available only when the emitted pose
exactly matches the selected candidate; it remains unknown for suppressed poses.

## Verification

The actual report at `.cache/expanded-failure-diagnosis-v1/report-v2.json`
contains 3,243,477 bytes with SHA-256
`7800c5f008ef2b0ac31e7e8bb8920bbe65fe5ceadfa4f116e39ac8c7d2fd393b`.
The first report is preserved separately. Eight focused diagnostic tests and
the combined 83-test inference, evaluation and pipeline suite passed. The suite
did not import Torch, and all 32 pinned existing runtime files remained unchanged.
These checks establish software behavior, not improved model accuracy.
