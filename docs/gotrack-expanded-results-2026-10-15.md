# Expanded GoTrack head adaptation — 2026-10-15

## What actually ran

The pretrained GoTrack correspondence head completed **16 AdamW updates**, one
epoch over 16 training crops. Its backbone and decoder remained frozen. This
was executed on the existing **free Lightning 4-CPU instance**, with no paid
GPU enabled. SAM and the pose rankers were not trained during this experiment.

Inputs were the first eight frames of three existing synthetic development
videos, with known metric meshes and cached automatic SAM foreground masks:

| Split | Physical object identity | Views | Available / requested |
| --- | --- | --- | --- |
| Training | HOT3D 17 | Pitch −25° and +25° | 16 / 16 |
| Previously inspected validation | HOT3D 27 | Pitch −25° | 8 / 8 |
| Held-out test | HOT3D 7 | Payload unopened | No examples materialized |

The two training views share one object identity. The eight validation frames
share another identity and sequence. These are dependent development frames;
there is only one validation object, so a population confidence interval is
unavailable. This experiment does not establish generalization to real videos.

Each frame used a fresh estimated FoundPose seed and retained the first actual
GoTrack refinement crop. Inference used no reference pose. All 24 crops and
frozen features passed their input-only seal and parity checks before synthetic
supervision was opened. This samples initialization, not causal tracking.

## Measured correspondence results

The comparison uses identical visible/known pixel support for the frozen
pretrained head and the newly adapted head: 70,133 visible and 72,501 known
pixels across the eight validation crops. Values below are means over frames.

| Metric | Frozen head | Adapted head |
| --- | ---: | ---: |
| Endpoint error, crop pixels ↓ | 2.325672 | 2.305137 |
| Correspondences within 1 pixel ↑ | 22.5454% | 23.2082% |
| Correspondences within 3 pixels ↑ | 74.3942% | 74.7848% |
| Visibility binary cross entropy ↓ | 0.189846 | 0.187979 |
| Visibility Brier score ↓ | 0.038827 | 0.038511 |

Mean endpoint error decreased by **0.020535 pixels (0.883%)**. Six frame errors
decreased and two increased. The largest frame regression was 0.132117 pixels.
The effect is small and comes from an already inspected validation sequence.
It is not evidence that 180° rotation failures, surface attachment or temporal
slipping improved. Those outcomes were not evaluated for this new head.

## Checkpoint and integration checks

- Training deliberately stopped after update 8 and resumed to update 16.
  The saved cursor, deterministic next sample, 60 Adam state entries and
  optimizer-state hash matched on resume; all final Adam step counters were 16.
- The export contains 20,029,955 head parameters, is 80,144,086 bytes, and has
  SHA-256 `e23794bcf91f2b3de48d30c0a1fed9c088c6e0119d8d64b5c1b2718adc4e3b96`.
- A fresh full network loaded the exported head. Cached-feature and full-RGB
  predictions matched the standalone exported head on all eight validation
  crops, with observed maximum absolute differences of zero. The configured
  tolerances were 1e-6 for cached predictions and 1e-4 for RGB predictions.
- Frozen backbone/decoder hashes stayed unchanged. An incorrect export digest
  was rejected without changing the loaded head.
- A separately reviewed expanded inference contract now accepts this exact
  export through the ordinary CLI. Its ordinary-refiner gate passed 15 actual
  forwards. Complete sequential comparisons are reported separately in the
  [full-video comparison](gotrack-expanded-inference-results-2026-10-15.md).
  It remains an opt-in research artifact in cloud storage.

Cloud checkpoint:

```text
/teamspace/studios/this_studio/visualizeit-private-adaptation-expanded-continuation-v1/run/head.pt
```

## Failure, repair and resource accounting

The first attempt captured all 24 crops, then failed before feature extraction
because JSON reformatting changed the saved plan's byte hash. The plan's
contents were identical. No labels or optimization ran in that failed attempt.
Its source, wrong-format plan and complete captures remain preserved.

The reviewed repair copies exact authenticated plan bytes. The continuation
copied all 48 crop/geometry archives without changing their bytes and reused the
actual capture. Before continuation, checkpoint/output caps were reduced from
384/1,200 MiB to 320/1,000 MiB so retaining both attempts fit the original total
budget. Learning settings, sampling, source model, seed and runtime limit were
unchanged. The trainer still reserves space for best, last and an atomic
temporary checkpoint.

| Measured stage | Seconds |
| --- | ---: |
| Original capture | 310.08 |
| Original failed extraction | 0.51 |
| Continuation focused tests | 2.27 |
| Frozen feature extraction | 86.32 |
| Input-only seal | 3.02 |
| Synthetic labels | 21.90 |
| Train through update 8 | 21.41 |
| Inspect interruption | 4.28 |
| Resume through update 16 | 24.20 |
| Export | 4.28 |
| Fresh-network verification | 32.23 |

The cumulative supervised workflow took **518.14 seconds**, including the
original failed attempt and authentication/copying overhead. The continuation
alone took 205.33 seconds. These timings exclude provider startup and operator
review time and are not live-video FPS or GPU measurements. The sampled peak
model process-tree memory was **3.09 GiB**. Final retained artifacts for both
attempts were **868.75 MiB**, within the original 1.5 GiB limit.

No model workers remained after completion. The provider then showed the Studio
as **sleeping**, with no allocated machine. The account displayed 3.94 credits
before and after, at its displayed two-decimal precision; exact consumption is
not established by that display. No paid resources or top-ups were enabled.

## Reproducibility and remaining work

Configuration: CPU, seed 42, batch 1, one epoch, learning rate 1e-5, weight decay
1e-4, visibility loss weight 1, gradient clip 1, checkpoint every 8 updates.
The historical training source lock remains `44e4c8a…d33`. Corrected source-v21
has 297 authenticated files and manifest SHA-256
`b902fdcda3d7a7c9d5cccc29307a3fc008f21476edc6829cd3e178e9fc5788c1`.
The failed source-v20 and previous exports were unchanged.

Local focused tests: 19 passed without importing PyTorch. Cloud focused tests:
19 passed in the continuation; the original cloud preflight also passed its
68-test combined suite. The 87,224-byte report archive has 45 authenticated
members and SHA-256
`5f8daba44589b69e432a42be406c588e73823b07c5f9b72c90166abd6ee4f15d`.
Reports are retained locally under `.cache/adaptation-expanded-continuation-v1`;
large features, data and checkpoints remain in cloud storage.

The [complete sequential comparison](gotrack-expanded-inference-results-2026-10-15.md)
subsequently found worse mean pose errors and does not support promotion.
Next: diagnose those failures and collect broader disjoint training objects plus
orientation-resolvable independent evaluation. SAM fine-tuning requires
independent foreground labels and evidence that segmentation is limiting the
system. Android deployment and 30-FPS operation remain unvalidated.
