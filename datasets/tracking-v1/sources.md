# Real footage provenance and licensing

The real component is **HB / HomebrewedDB, BOP Primesense validation** at
revision `985ebaef7b1bd4bffb2039f18dd81faa647fd060`. The official
[BOP dataset catalog](https://bop.felk.cvut.cz/datasets/) lists HB as
**CC0 1.0 Universal**. The [pinned upstream dataset card](https://huggingface.co/datasets/bop-benchmark/hb/blob/985ebaef7b1bd4bffb2039f18dd81faa647fd060/README.md) independently declares `cc0-1.0`.
Both were checked on 2026-10-26. The original TUM project/license links now
redirect to a general university homepage; we do not treat that redirect as
a working primary license file.

Credit: Roman Kaskman, Ivan Shugurov, Sergey Zakharov, Slobodan Ilic,
*HomebrewedDB: RGB-D Dataset for 6D Pose Estimation of 3D Objects*, ICCVW 2019;
BOP-format conversion by the BOP authors.

## What is in Git, versus acquired locally

- `sources.hb-selection.json`: exact predetermined scene/object/frame selection.
- `sources.hb-lock.json`: upstream archive/member identities, resource byte counts
  and SHA-256 hashes, and seals for both local manifests.
- `bench/tracking_real_data.py`: bounded, resumable download/verify recipe.
- `real-examples/`: eight byte-identical licensed real RGB examples, a display
  contact sheet, and deterministic compressed **complete** inference/evaluator
  annotation packs (including all 3,600 scored object-frame reference records).
- **Not the full raw corpus in Git:** beyond the eight examples, RGB/depth/mask/model resources are
  acquired into the ignored `.cache/tracking-benchmark/real-v1/`. This avoids
  duplicating more than a gigabyte of upstream media as Git blobs. This is a
  reproducible manifest-backed dataset addition, not a claim all media bytes are
  embedded in the repository. The CC0 source permits redistribution, but size
  and source-of-truth considerations favor pinned upstream acquisition.

```bash
python -m bench.tracking_real_data acquire \
  --output .cache/tracking-benchmark/real-v1 \
  --recipe-output datasets/tracking-v1
python -m bench.tracking_real_data verify \
  --output .cache/tracking-benchmark/real-v1
```

## Selection and evidence boundaries

The deterministic coverage rule takes the first three globally unused object
IDs present exactly once in frames 0–309 of each of HB scenes 1, 2, 3 and 4.
No tracker result, error, motion threshold or visibility threshold chooses the
clips. For each object, frames 0–9 are explicitly authorized **annotated
onboarding**, and frames 10–309 are scored. All scored frames remain, including
poor visibility and failures. Footage is real RGB-D; the pose/depth/mask labels
are supplied by the original dataset, not newly created independent human
annotations. The dataset was previously used by this project and pretrained
exposure is unknown: this is not an unseen-object generalization claim.

`inference.json` supplies only RGB/calibration/identity for scored frames.
Only onboarding records include pose, depth, and `onboardingMask`. The separate
`evaluator.json` includes scored pose, depth, visibility, and visible masks.
Metrics compare against dataset reference geometry; they do not establish
physical attachment, phone performance, official BOP benchmark standing, or
end-to-end automatic detection. Source timestamps/FPS are not asserted.

The downloader requests only validated HTTP 206 bounded ZIP byte ranges
(at most 16 MiB each), verifies member size/CRC, records SHA-256, refuses full
archive fallback, caps selected expanded resources at 3 GiB, and retains a
1 GiB free-space reserve. It never downloads the synthetic PBR training set or
private user media. The dedicated large-PLY expansion limit is 64 MiB; all
other member types are limited to 32 MiB. The network range cap remains 16 MiB.

Recreate the real examples and complete compressed annotation manifests with:

```bash
python -m bench.tracking_real_data examples \
  --output .cache/tracking-benchmark/real-v1 \
  --examples-output datasets/tracking-v1/real-examples
```

## Full-media release package

A deterministic full-media archive can be prepared separately from the small
Git recipe:

```bash
python -m bench.tracking_real_data package \
  --output .cache/tracking-benchmark/real-v1 \
  --archive artifacts/tracking-benchmark/releases/visualizeit-hb-tracking-v1.tar.gz
```

The package includes every locked real resource, both manifests, the lock and
selection, and a CC0 attribution/integrity README. Archive headers have fixed
ownership, mode and time; gzip has no source filename and mtime=0. A neighboring
JSON receipt records its size/SHA-256. Packaging does **not** upload anything.
The local full-media package was prepared; GitHub Release upload remains a
separate publishing step. Do not describe the raw media as GitHub-hosted until
the uploaded asset URL/hash have been verified.
