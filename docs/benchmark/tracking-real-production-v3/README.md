# Full real corpus — original mesh-bank initialization

**Geometric tracking component only: no SAM and no pretrained pose network.**
This separate run fixes the custom-initialization limitation of the earlier
[simplified experiment](../tracking-results-2026-10-26.md). It is not the offline
SAM → trained-pose benchmark, and its results are not pooled with that experiment.

All 12 HB objects × 300 scored frames were replayed once, with 20 disposable
warm-up frames and fresh tracker state. All three trackers receive the exact
original `prototype_benchmark.build_bank` output: mesh-ray intersections, ten
authorized onboarding RGB-D/pose frames, ORB up to 2,000/view, no extra feature
cap. Verified cached banks were prepared before serial timed replay. No masks,
depth or reference poses enter scored geometric updates. Scoring follows the
sealed predictions; the input/evaluator/source identities are in `freeze.json`.

| Tracker | Visible coverage, mean projection <5 native px | Public / all frames | Attempted update FPS | Decode+update replay FPS |
|---|---:|---:|---:|---:|
| Original rigid (`df788084`) | 4.86% | 6.25% | 30.69 | 26.10 |
| Current rigid | 4.52% | 5.36% | 29.87 | 25.75 |
| Current stream wrapper | 4.52% | 5.36% | 29.76 | 25.63 |

There are 3,536 reference-visible frames (visibility ≥0.5), 3,600 attempted
updates per system, **zero initialization failures and zero skipped updates**.
The coverage denominator retains failed/lost outputs. Corresponding-point mean
projection is a 2D proxy on deterministically sampled model vertices, not an
official BOP score, symmetry-aware accuracy, or independent physical attachment
assessment. Public-only 3D errors remain available per object in `results.json`.

The original initializer removes the earlier HB33 initialization failure and
materially changes measured quality. Nevertheless sustained geometric coverage
is low under this particular known-object protocol. Neither result establishes
the quality of the separate pretrained tracker. The current wrapper produces
fewer public poses; these numbers alone do not prove a causal algorithm defect
or a universally preferable safety/availability policy.

One repeat on a shared 9-logical-CPU AMD EPYC VM is a functional comparison, not
a statistically stable speed ranking. OpenCV/BLAS used one thread, CPU only.
Bank construction, setup, scoring and cold initialization are excluded from
update/replay FPS and must not be described as end-to-end throughput. Exact
versions, configuration, source and dataset hashes are in `freeze.json`.
`trace-index.json` records sealed raw prediction identities; raw traces stay
local rather than becoming additional public Release assets.

- `real-production-versions.csv`: exact graph-ready aggregate values.
- `real-production-cases.csv`: per-object graph data and public/quality counts.
- `results.json`: full aggregate and per-sequence summaries.
- [Independent units/geometry audit](../tracking-hb-protocol-v1/README.md).

The already published [licensed dataset Release](https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1)
and [offline SAM/pose dataset adapter](../../../datasets/offline-sam-pose-v1/README.md)
are separate from these model-quality results. No training, live camera,
frontend test, or main-branch merge was performed.
