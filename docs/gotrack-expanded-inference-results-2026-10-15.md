# Expanded GoTrack head: full-video development comparison

## Status

The expanded correspondence head was trained, exported and installed in the
ordinary recorded-video inference path. Six full-video replays and two causal
repeats completed and were scored after sealing. **The head is not promoted:**
mean translation errors and temporal residuals regressed, and availability fell
from 64/72 to 63/72.

## What changed

Only the GoTrack correspondence head was updated during the preceding training
experiment. The backbone, decoder, SAM and pose-ranking models stayed frozen.
The exported head is SHA-256
`e23794bcf91f2b3de48d30c0a1fed9c088c6e0119d8d64b5c1b2718adc4e3b96`.
See [training results](gotrack-expanded-results-2026-10-15.md) for the actual
16 updates, interruption/resume checks and correspondence metrics.

Serving uses `bench/runtime/gotrack-inference-expanded-v1.json`, SHA-256
`1495f84ceb3bb6513d3fb76602f68b88bec7e538b20e088a394b29df9afe7287`.
The frozen control uses the same geometry, runtime and inputs. The legacy
selection algorithm is used for both conditions. The existing legacy contract
and historical runs are preserved.

## Comparison design

The fixed comparison uses three synthetic videos, with 24 samples per video
at timestamps 0.0 through 2.3 seconds:

| Sequence | Role |
|---|---|
| Object 17, pitch -25 degrees | TRAIN resubstitution |
| Object 17, pitch +25 degrees | TRAIN resubstitution |
| Object 27, pitch -25 degrees | Previously inspected validation |

These are two object identities, with dependent views and frames. They cannot
support a generalization claim or confidence intervals treating frames as
independent observations. Untouched test object 7 remains excluded.

Both conditions use the same 72 observed SAM masks: 48 authenticated existing
masks and one fresh 24-frame propagation pass. Projected silhouettes are not
used as observed masks. The selected mask montages are a visual check, not an
independent mask-accuracy measurement.

Six new full-video replays compare the frozen and adapted heads. Two additional
four-frame replays check causal prefix consistency and do not enlarge the
quality sample. Large stages execute sequentially on the free four-CPU cloud
machine. There is no local neural execution or paid GPU allocation.

Reference sidecars are scored separately after the prediction archive is
sealed. They are not inference inputs. All requested frames, including failures
and suppressed poses, remain in the availability denominator.

## Verification already completed

- 75 focused local tests passed after the evaluator repair below; the original
  cloud source passed 72 tests. Neither test run imported Torch.
- Historical legacy evaluation remained byte-identical.
- The adapted ordinary-refiner gate passed 15 actual forwards across all three
  cases, including exact original crop/transform checks and checkpoint identity.
- All six 24-frame replays completed. Both four-frame repeats matched full-run
  pose and candidate semantics, excluding only existing profiling fields.
- All 51 archived members passed hash and inventory checks. An independent
  standard-library audit reproduced translation, temporal and paired statistics
  within 1.39e-17 metres.

## Measured pose results

Errors below are **millimetres** against synthetic references on emitted poses.
Unavailable poses remain in each 24-frame availability denominator and have no
translation error value.

| View / role | Head | Emitted / requested | Mean | Median | Maximum |
|---|---|---:|---:|---:|---:|
| Object 17, -25 / TRAIN | Frozen | 20/24 | 1.753 | 1.838 | 3.245 |
| Object 17, -25 / TRAIN | Adapted | 21/24 | 6.012 | 1.178 | 100.181 |
| Object 17, +25 / TRAIN | Frozen | 21/24 | 3.936 | 3.391 | 8.416 |
| Object 17, +25 / TRAIN | Adapted | 21/24 | 7.291 | 2.204 | 104.539 |
| Object 27, -25 / used VAL | Frozen | 23/24 | 8.179 | 3.885 | 79.458 |
| Object 27, -25 / used VAL | Adapted | 21/24 | 15.877 | 2.815 | 187.862 |

Adapted medians are lower, but large errors make all three means worse. The
adapted maxima occur at frame 9 in both TRAIN views and frame 11 in used VAL.
These are emitted incorrect translations; suppression did not catch every
error. Pooled emitted means are 4.779 mm frozen and 9.727 mm adapted, on different
emitted-frame sets.

On frames emitted by both conditions, adapted minus frozen mean translation
error is -0.450 mm on 20 common frames in the first TRAIN view, -1.478 mm on 20
in the second, and **+7.453 mm on 21 used-VAL frames**. Across 61 common frames
it is +1.934 mm. TRAIN intersection improvements do not erase the extra erroneous
emissions or VAL regression.

Consecutive translation-increment residual means are 3.823 mm frozen versus
7.864 mm adapted, on 59 and 57 eligible pairs respectively. This compares
predicted and reference translation changes; it is not surface-slip error.
Frozen suppressed 8/72 frames (7 recovering, 1 lost); adapted suppressed 9/72
(7 recovering, 2 lost). Independent recovery accuracy remains unmeasured.

Valid candidates existed before gates on 71/72 frozen and 72/72 adapted frames;
after gates, 71/72 and 70/72 respectively. Candidate support does not certify a
correct emitted pose.

## CPU timing and resources

| View | Frozen pose-stage seconds | Adapted pose-stage seconds |
|---|---:|---:|
| Object 17, -25 | 542.55 | 545.25 |
| Object 17, +25 | 472.71 | 563.00 |
| Object 27, -25 | 454.87 | 535.09 |

Each stage covers 24 requested samples, including initialization and suppressed
or failed samples. Segmentation was cached: these are pose-stage timings, not
camera-to-overlay latency or mobile FPS. Largest sampled process-tree RSS across
full runs was 2.308 GiB frozen and 2.471 GiB adapted; transient peaks may be missed.

Cumulative charged experiment time was 4,084.93 seconds (68.08 minutes), including
mask preparation, conservatively charged failed metadata admissions, gates,
replays, prefixes and packing. Provider startup and manual review are excluded.
Free four-CPU compute was used; no paid GPU or top-up was enabled. No model
workers remained, then the provider was verified sleeping with no machine.
Displayed credits remained 3.94 at two-decimal precision, which does not establish
exact consumption.

## Artifact and evaluator provenance

Prediction archive: 481,985 bytes, SHA-256
`41cf212aad863fd077a9b2c0a840030fc676383050a2a6a7a54cfad8b3f1e039`.
Master manifest:
`545922617b8f9b74affd0c84fe39bf473a9a26a6ab0f3568f94037f872809e7a`.
Separate completion receipt: 439 bytes,
`c8c8202530cb9f0a767143677924f1dff07e186cc292504c7f41903a15ba63ce`.
Post-seal policy:
`06d725aafa3d15aac5bc1ced863b4367515971cb4918511acb0744d8cb60acd6`.
Evaluation report: 545,206 bytes,
`7e8a48f829bbe971cde8147242f0d5ed666c613329c26bdaf9a4e9b0446ac70f`.
Small artifacts are under `.cache/expanded-full-video-development-v1`; large
caches and the checkpoint stay in cloud storage.

The first scoring attempt refused the 2,191,101-byte paired report before
reference-pose reads because its parser default was 2 MiB. The reviewed repair
permits 3 MiB only for the expanded policy's paired report; legacy and other JSON
limits remain 2 MiB and the archive cap remains 32 MiB. Predictions, settings and
metric mathematics were unchanged. The refusal receipt is preserved. Repaired
evaluator SHA:
`33ae82cdeefdc7235deb07bbf87c25a70b8f834251ab45c574bbbc73e5961e8f`.

## Interpretation limits

These errors measure synthetic reference agreement. Rotation references are
observationally ambiguous, so rotation accuracy and severe-rotation error rates
remain unmeasured. Visible surface attachment and independent occlusion recovery
accuracy also remain unmeasured. CPU replay timing is not Android throughput or
evidence of 30 FPS operation.

SAM fine-tuning still requires independently labeled masks and a measured
segmentation bottleneck. Broader GoTrack training requires disjoint data and
orientation-resolvable evaluation before model promotion.
