# Sealed synthetic tracking results, 2026-10-26

Completed 24 sequences ×190 scored frames ×3 repeated runs ×3 exact systems. These are controlled synthetic tests, not trained GoTrack, real-world generalization or live-phone performance. The real corpus is still running; it will be reported separately.

The primary metric is visibility>=0.5 coverage with mean corresponding-model reprojection error<5 native pixels at320×240, not generic6D or independent physical attachment accuracy. Three low-texture initialization failures retain570unique failed frames per system in quality denominators, while FPS uses only3990unique attempted frames. See the dataset card for the fixed3000-anchor CPU initialization adapter and timing exclusions.

| System | <5px visible coverage | Update-only FPS | Decode/hash-inclusive replay FPS |
|---|---:|---:|---:|
| Exact historical rigid df788084 |78.86%|40.96|38.88|
| Current rigid |78.29%|35.64|34.01|
| Current stream wrapper |76.84%|33.36|31.92|

This is a quality/availability tradeoff, not evidence that the new code is uniformly worse: wrong-threshold visible public poses fall56→28→25 on unique frames, while public-pose P95 mean-model error improves3.61→3.10px. The wrapper also withholds good outputs and can recover more slowly; one controlled return requires41observable frames versus4historically. Paired details and the final real-corpus report will follow.

`results.json` contains per-sequence/repeat summaries; CSVs support version/stress graphs; `freeze.json` records hardware, source/data hashes, versions, warmup and repeats. All failure frames remain in denominators. Full raw traces are local and will be archived after the real run finishes.
