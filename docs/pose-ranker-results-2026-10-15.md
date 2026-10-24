# Pose-ranker development results — October 15, 2026

Both the linear scorer and the 64/32-unit MLP completed actual cloud training.
The MLP was selected by the predeclared minimum validation objective. It remains
an explicit `ordering-research` option; the legacy pipeline is still the default.

## Training and verification

| Measurement | Linear | MLP 64/32 |
| --- | ---: | ---: |
| Completed epochs | 40 | 40 |
| Optimizer steps | 120 | 120 |
| Selected validation objective, lower is better | 0.88792634 | 0.29349631 |
| Parameters in the head | 33 | 4,225 |
| Exported ONNX bytes | 1,738 | 19,125 |
| Exported JSON bytes, including provenance | 26,075 | 160,159 |

Training used the existing frozen feature caches on Lightning's free 4-CPU
machine, with no new pretrained weights or paid GPU. Both two-epoch smoke runs
verified exact interrupted/resumed model, optimizer, scheduler, scaler, RNG and
progress state. Final JSON/ONNX exports matched the saved PyTorch decisions on
216 validation groups; maximum score differences were below 1.24e-7.
Both compact JSON artifacts also loaded locally through the actual NumPy
ranker factory, without importing Torch. Loading them through the ordinary
`learned` or `confidence-research` factories was correctly rejected.

## Same-candidate validation comparison

The complete frozen validation inventory contains 216 frames. Only 33 frames
have usable eligible candidates in these bounded caches; unprocessed or failed
frames remain in the inventory. These counts measure selection on recorded
candidate sets, rather than a complete new tracker replay of all 216 frames.

| Method | Selected frames | Quality-label pass | Quality-label fail | No selection |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 33 | 29 | 4 | 183 |
| Deterministic | 33 | 29 | 4 | 183 |
| Linear | 30 | 28 | 2 | 186 |
| MLP | 31 | 29 | 2 | 185 |

The MLP rejected two baseline failures while retaining all 29 positive-label
frames. The linear scorer rejected the same two failures and one positive-label
frame. Both also changed some choices between positive-label candidates.
These are development results on data reused for checkpoint selection and
threshold calibration, with one seed, not independent test performance.

All four bad baseline validation selections are translation failures with
observationally ambiguous rotation. Positive labels on such frames certify
only the available error dimensions. They do not establish correct orientation.
There are **zero mixed-quality validation candidate groups**. The four angular
training pairs share one mouse frame and one incorrect candidate. The dataset
therefore does not establish a reduction in 180-degree rotation failures.

Rotation was scorable on 18 selected frames for every method. Median reference
rotation error was 2.350 degrees for legacy, 2.020 for deterministic, 2.212 for
linear and 2.020 for MLP; every method's 95th percentile was 7.448 degrees and
every method had zero severe rotation failures on those scorable frames.
Ambiguous rotations were excluded. These descriptive values show no measured
MLP advantage over deterministic ranking on this rotation subset.

All validation observations concern one synthetic object. No independent human
attachment labels were present. Object-level uncertainty cannot be estimated
from one object. The individual comparison reports retain sequence bootstrap
intervals for cluster means; those intervals are not intervals for these pooled
counts. Untouched test objects have not been opened. Physical surface attachment
and real-video generalization remain unmeasured.

## Causal offline integration replay

The selected MLP completed the public `bench.quality_runner pose` path on the
existing 16-frame synthetic occlusion development sequence, using its observed
SAM masks and collecting actual frozen patch features. The process returned
exit code 0 after 365.78 seconds on CPU, with 20 candidate attempts and sampled
peak process-tree RSS of 2,420,027,392 bytes. The replay receipt reports:

- 12 tracking frames with visible overlays.
- 2 recovering frames, suppressed pending a second validated pose.
- 2 lost frames, suppressed because acceptance probability was too low.

Inference outputs were sealed without reading reference labels. This replay
was not accuracy-scored; its state totals are recorded in the cloud receipt,
while the raw trace remains cloud-side. Cached segmentation was reused, so the
timing is a pose-stage integration measurement, not full camera-to-overlay
performance or a phone FPS estimate. It does not establish recovery accuracy.

## Artifacts and loading

The downloaded compact package and replay reports passed exact archive-size,
SHA-256, member-whitelist, duplicate/path/symlink and per-file hash checks.
No videos, meshes, reference-pose files, frozen feature caches or resumable
binary training checkpoints were downloaded.

- Local package: `.cache/ordering-research-v1/` (42 files including manifest).
- Local replay reports: `.cache/ordering-causal-replay-v1/` (5 files).
- Cloud run: `/teamspace/studios/this_studio/visualizeit-cloud-v16/cloud-runs/ordering-research-v1/`.
- Cloud replay: the same root's `cloud-runs/ordering-causal-replay-v1/`.
- Research archive SHA-256: `58a73023f5abacf2a45a6522c7b246fe009c5ddc1ca5a8d5c3c54256d97ebecd`.
- Replay archive SHA-256: `0c412f70ac279de8f5fc25a8d4dd16ae1d833bc1b5d2d0a6222e2472eb7eba95`.

Load the small local package explicitly:

```python
from pathlib import Path
from bench.quality_pose_ranker import make_ranker

ranker = make_ranker(
    "ordering-research", Path(".cache/ordering-research-v1/mlp64_32.json")
)
```

For a separately scheduled offline replay with an existing calibrated bundle
and observed mask directory:

```bash
python -B -m bench.quality_runner pose \
  --bundle /path/to/inference-bundle --masks /path/to/observed-masks \
  --output /path/to/new-run/selected.json --device cpu --limit 16 --mode complete \
  --ranker ordering-research \
  --ranker-checkpoint .cache/ordering-research-v1/mlp64_32.json \
  --candidate-trace /path/to/new-run/candidates.jsonl --ranker-patch-features
```

The measured headless cloud replay additionally used
`--disable-multisampling`, matching its existing rendering setup. This command
requires the pinned frozen vision runtime and input assets; the compact head
alone does not provide candidate generation or segmentation.

The Studio was verified sleeping with no allocated machine after completion;
the displayed credit balance remained 3.94. The next scientific task is a
prospectively fixed validation sample with observable incorrect alternatives,
followed by independent test evaluation after settings are frozen. Additional
correspondence-head adaptation subsequently completed a two-step cloud smoke,
resume, export and fresh-network loading; see
[the measured GoTrack result](gotrack-adaptation-results-2026-10-15.md).
SAM fine-tuning has not run. Neither result establishes improved tracking.
