# GoTrack terminal-flow training: October 21, 2026 UTC

## Completed experiment

The final flow-output weights completed three epochs and **285 optimizer updates**
on the existing cloud cache. Only 514 parameters trained. The official GoTrack
backbone, decoder, DPT shared layers and confidence-output row stayed frozen.
SAM and the pose rankers were unchanged.

The run deliberately stopped at update 8 and resumed to 285. Resume validated the
checkpoint policy, optimizer settings/state, frozen tensors, deterministic cursor
and random-generator state. This demonstrates continuation of the recorded run;
an independent uninterrupted 285-step trajectory was not also run.

Training used seed 521, learning rate 1e-5, batch size 1, FP32, AdamW decay 1e-4
and gradient clipping at 1. The candidate was fixed to the final epoch. Validation
was evaluated after training, with no intermediate validation selection or
threshold fitting.

## Data and limitations

- TRAIN: 95 available examples across objects 13, 17, 18 and 20; one unavailable
  example remains recorded out of 96 requested.
- Development VAL: 20 available examples of object 4; four failures remain
  recorded out of 24 requested.
- TEST: object 7's 24 metadata entries stayed sealed; no payloads were opened.
- Inputs are cached frozen decoder features from the existing synthetic-render
  experiments, with separately generated correspondence supervision.
- No new rendering, segmentation, candidate capture or feature extraction ran.

The validation object has been inspected repeatedly. Its views are dependent;
these are development measurements, without independent-object uncertainty or
real-video accuracy evidence. This run does not satisfy the separate frozen
312-example/13-object target.

## Measured results

EPE is mean correspondence endpoint error in crop pixels. TP and FP use the
existing native float32 confidence decision, strictly greater than 0.3, on
identical known/visible validation support.

| Condition | VAL EPE, px | TP | FP |
|---|---:|---:|---:|
| Frozen pretrained head | 21.209391 | 65,462 | 56,209 |
| Earlier full-head LR 1e-5 trial | 21.088400 | 64,309 | 54,764 |
| Earlier full-head LR 1e-6 trial | 21.171269 | 65,418 | 56,094 |
| New terminal-flow-only trial | **21.183127** | **65,462** | **56,209** |

The new trial reduced EPE by 0.026264 px, approximately **0.124%**, relative to the
frozen head. Fifteen validation examples improved and five regressed; the largest
regression was 0.064799 px. It does not beat the earlier adapted heads on EPE.
Its distinguishing result is unchanged confidence: all 115 available TRAIN+VAL
inputs had maximum confidence difference **0.0** and zero changed threshold
decisions. Frozen state identity was checked separately.

The predeclared development guard passed: at least 0.001 px EPE reduction, exact
baseline TP/FP counts and preserved confidence. This does not promote the model.
Changed flow can still affect downstream geometric rejection and tracking.

Per-epoch TRAIN losses were 14.438275, 14.434187 and 14.431322. Recorded training
loop time across the interrupted and resumed invocations was 63.000 seconds.
CUDA peak allocated memory was 388.75 MiB, with 634 MiB reserved. These are
head-only training measurements, excluding full vision inference, provider setup
and data restoration; they are not tracking FPS or phone memory requirements.

## Verification and artifacts

- Existing local adaptation/loader contract suite: 44 tests passed.
- New parent/reviewer local suite: 10 passed, one optional Torch fixture skipped.
- Actual Kaggle CPU suite: all 11 tests passed, including the tiny Torch fixture.
- Generic inference installer: three cached validation inputs had exact exported
  versus checkpoint flow/confidence parity. No labels entered that check.
- Full RGB/video pose replay, attachment, rotation and recovery were not tested
  for this checkpoint. The baseline remains selected.

Export: 80,165,526 bytes, SHA256
`27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`.
The standard nine-field export contains ordinary DPT state keys and works with
the existing experimental cached-head installer.

The private version-1 package is
[visualizeit-flow-terminal-20261021-27e06a51](https://www.kaggle.com/datasets/achitaan/visualizeit-flow-terminal-20261021-27e06a51).
All 37 payload files, totaling 240,911,158 bytes, were hash-verified from Kaggle's
provider-mounted copy, plus the package manifest. It includes optimizer/RNG
checkpoints, source, configuration, logs and verification receipts. Dataset
arrays and full pretrained base models are excluded. No weights were downloaded
to the laptop.

Kaggle's SDK initially returned a stale shared private mount, including when a
fresh output directory was requested. Both checksum failures were retained.
Explicitly attaching the exact new dataset through the notebook UI provided the
correct manifest; all payloads then matched. Do not trust the generic mount path
without checking the expected manifest hash.

The completed checkpoint cannot be resumed under the same completed-run policy.
The recorded update-8 interruption/resume is a separate, successful check.

## Reproduction

Source entry point: `bench.quality_adaptation_flow_terminal`, with `preflight`,
`train`, `resume` and `export` subcommands. It reuses the existing authenticated
data/runtime helpers; it does not modify the previous trainer.

The private package's `source/flow-terminal-controller-20261021.py` records the
actual launch sequence, original source/data hashes, deterministic CUDA
bootstrap and stop-at-8/resume/export commands. It expects the authenticated
runtime and cache restored at the recorded cloud paths and a fresh output root.
Use the packaged bootstrap rather than invoking a CUDA run without its required
numerical policy. The package also includes the exact source and tests.

Next work should broaden verified data coverage and evaluate the checkpoint
through sequential video inference. Further tuning against this same validation
object alone would not establish generalization.
