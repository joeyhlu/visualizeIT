# GoTrack cached-data expansion: 167 TRAIN examples

## Actual execution

The free Kaggle workflow completed **501 optimizer updates over three epochs**.
It stopped after eight updates, resumed from that checkpoint, completed final
evaluation, and exported an ordinary DPT head. Workflow PID 405 exited 0.
Recorded training-loop time was 103.425 seconds, excluding restoration,
assembly, the transfer diagnostic, preflight, and export.

Only 514 scalars in the final two flow-output rows were trained. The shared
decoder and confidence row remained frozen. This is a small correspondence-head
adaptation, not training SAM or the complete GoTrack network.

## Data and preserved provenance

The new union contains 167 available TRAIN examples across seven objects
(1, 6, 10, 13, 17, 18, 20), from 168 requested examples. It combines the existing
95-example cache with all 72 TRAIN examples from the earlier smoke cache.
Original manifests, indices, recipes, final requests, failures, and split
inventories remain in the union lineage. The extraction transforms match;
their original crop-manifest identities remain distinct.

Validation remains the same 20 available / 24 requested examples of object 4.
Object 7's 24 TEST entries remain sealed metadata. No TEST payload was opened.
The data and labels are synthetic; these are development comparisons, not
independent real-video attachment measurements. The larger 312-example,
13-object target remains unfulfilled.

The old 72 TRAIN feature/label pairs were restored from the existing authenticated
cloud archive: 144 files, 551,687,760 bytes. No new rendering, masks, feature
extraction, pretrained-model downloads, or local ML training occurred.

## Recorded results

| Fixed candidate | Development flow EPE, pixels |
|---|---:|
| Frozen baseline | 21.209390579 |
| Prior 95-example terminal-flow adaptation | 21.183126755 |
| New 167-example terminal-flow adaptation | 21.188126366 |

The new candidate reduces error by about 0.10% relative to the baseline, but is
slightly worse than the previous 95-example candidate. Fourteen validation
frames improve and six regress; the largest regression is 0.201268 pixels.
This is insufficient evidence of a useful tracking improvement. Serving was
not changed, and no independent uncertainty/generalization claim is made.

Confidence outputs were exactly identical on all 187 available TRAIN/VAL
inputs, with zero threshold-decision changes. Validation TP/FP/FN remain
65,462 / 56,209 / 3,151. Peak allocated CUDA memory was 407,636,992 bytes;
this excludes full segmentation, pose generation, and rendering residency.

Before fitting, the fixed prior checkpoint was evaluated on all 72 added TRAIN
examples without fitting or threshold selection. All 72 completed, with zero
failures and no VAL/TEST arrays read. That CPU diagnostic took 70.443 seconds;
it is not a video FPS measurement.

## Implementation and checks

`quality_adaptation_feature_union.py` assembles and validates the explicit union
and runs the fixed-checkpoint transfer diagnostic. The terminal trainer accepts
`--feature-union`; the mode, recipe identity, and union source hash are bound to
the checkpoint policy. Core train/data/runtime/LOCK and the ordinary installer
remain unchanged.

Kaggle's 18 focused CPU tests passed in 2.232 seconds. The broader local check
ran 27 tests: 23 passed, two skipped, and two failed because of confirmed fixture
errors (reusing an already-created destination, and expecting a hash rejection
after changing file length). Those failures remain disclosed and unrepaired.
Independent review approved the production union, trainer, and controller.

Export SHA256:
`3593e1cd7164251da2b41137c55826064fca0100283ec1cad8168d3e985dfca7`

The export is 80,176,406 bytes and passed strict head-state loading during export.
The ordinary generic parity CLI does not yet accept union receipts; generic
installer parity remains pending. This does not establish phone or video readiness.

## Location and remaining work

Cloud run: `/kaggle/working/visualizeit-flow-terminal-union167-20261021/`.
The generated `union-spec.json`, `config.json`, stage logs, lineage, and resumable
interruption evidence are there. The completed run cannot be resumed under its
unchanged three-epoch configuration.

Local small result record: `.cache/cloud-prepare/union167-result-summary-20261021.json`.
This experiment's recovery and backup verification downloaded only small reports
to the laptop; its model weights and datasets remain in cloud storage.

The first checkpoint was lost when Kaggle reset before its approved backup could
be verified. That failure remains part of the experiment history.

Recovery completed in Kaggle batch version `360670664`. It reproduced all 501
updates, the same exported checkpoint SHA256, and the same development errors.
The reproduced optimizer loop took 104.977 seconds, excluding restoration,
preflight, evaluation, export and backup. Peak training CUDA allocation was
407,636,992 bytes; peak reservation was 662,700,032 bytes.

The checkpoint is now available in private Kaggle dataset
`achitaan/visualizeit-union167-20261021-3593e1cd`, version 1. A separate CPU session
read back and verified every one of its 54 payload files: 242,160,725 bytes,
including exported weights, training checkpoints, configuration and reports.
The manifest SHA256 is
`a930c0fb6bf328621b11ce95fa05a8f6bffc686ef5cdef01c1ef547484418d03`.
Privacy was also verified in the Kaggle UI. The verification session was stopped.

The local verification receipt is
`.cache/cloud-prepare/union167-remote-readback-20261021.json`; eight small result
reports are in `.cache/cloud-prepare/union167-reproduced-20261021-reports/`.
Recovery used existing pinned inputs, no new capture or held-out TEST payloads.
The larger 312-example/13-object target and generic union installer parity remain
unfinished. The learned checkpoint has not been promoted over the baseline.
