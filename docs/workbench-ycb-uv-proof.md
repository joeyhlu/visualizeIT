# YCB candidate UV-convention proof (staged, diagnostic)

`tools/workbench_ycb_uv_proof.py` tests whether the three blocked official-YCB
candidate GLBs use the same texture-coordinate convention as the BOP renderer.
It never promotes a candidate, writes no metric bundle and has no override.
The candidates stay `blocked_unverified_uv` until a separate review decides,
and any acceptance would be pinned separately.

## Scope and data

- **Scope:** a known-onboarding metric-CAD diagnostic. It is **not** automatic
  RGB preparation and **not** held-out accuracy.
- **Rows used:** only onboarding rows with logicalIndex 0 and 9 from the pinned
  inference manifest.

  | Object | Frames |
  |---|---|
  | 2 | 361, 370 |
  | 4 | 177, 186 |
  | 18 | 175, 184 |

- **Data used:** their RGB, `cam_K` and known onboarding pose.
- **Never read:** scored frames, poses or labels, `evaluator.json`, masks and
  depth.
- **Authentication:** reuses the existing `tools/workbench_ycb_controls.py`
  pins: manifest SHA, PLY SHA, official PNG (object, revision, member, size,
  SHA, full decode) and the candidate receipt/GLB binding.

## Phases

### `prepare` (pure CPU; numpy + PIL; no torch, GL or matplotlib)

```powershell
.cache\quality-windows\Scripts\python.exe tools\workbench_ycb_uv_proof.py prepare --object-id 18 `
  --candidate-dir artifacts/video-workbench/ycb-controls/validation-authenticated-v1/object-000018 `
  --out artifacts/video-workbench/ycb-uv-proof/prepare-object-000018-v1 [--landmarks frozen.json]
```

**`source_convention_pass`** requires all of the following:

- the triangle indices and their order match the PLY exactly;
- GLB positions equal PLY millimetres × 0.001 within 0.001 mm;
- the GLB has a single identity node;
- projection `K(R·PLY_mm + t_mm)` matches `K(R·GLB_m + t_m)` within 0.001 px
  with positive depth;
- geometry-only and translation-only ×1000 mutations fail. Scaling both
  together leaves the projection unchanged, so it does not prove units;
- this tool's own GLB parser reads `TEXCOORD_0 = (u, 1 − v_ply)`;
- the embedded image is the official PNG;
- the files the **current runtime would actually import** match exact reviewed
  SHA-256 pins: `bench/workbench_geometry.py`, `tools/workbench_ycb_controls.py`,
  BOP `inout.py`/`renderer_py.py`, trimesh glTF, pyrender `mesh.py`/`texture.py`/
  `flat.frag`/`sampler.py`/`material.py`/`constants.py`, trimesh
  `visual/material.py`/`visual/texture.py` (the sampler/material pins were
  verified by Sol in the actual GPU env), and `quality_gotrack.py`. They are
  resolved with `importlib.util.find_spec`; no alternative site-packages can
  be named. A drifted or missing file fails the source convention, and
  `render` refuses to start. There is no hash or string-match fallback. The
  cited statements are recorded for information only;
- the **actual trimesh loader's** UVs equal the raw PLY UVs. When trimesh isn't
  importable, this check is recorded as unavailable and the pass stays false.

**Source reference (independent of the renderer):**

- it uses the existing fixed-point rasterizer, `bench/workbench_geometry`
  with 1/256-subpixel quantized projected vertices, then perspective-correct
  interpolation of depth (mm) and the raw PLY UV. This is **not** exact
  ray/triangle intersection, which is a limitation especially for
  high-frequency texture. The method is recorded in every receipt, and the
  thresholds are fixed;
- it is coloured with OpenGL normalized clamp bilinear sampling at texel
  centres, `x = u·W − 0.5`, `y = (1 − v)·H − 0.5`, on the original top-down PNG;
- deterministic interior samples are frozen (up to 1024) away from triangle
  edges, depth discontinuities, UV seams and borders. Fewer than 128 makes the
  image inconclusive;
- exclusion counts are recorded.

**`onboarding_correspondence_pass`** comes only from a frozen
`workbench_uv_landmarks_v1` file, which is strictly validated:

- unique, non-empty IDs;
- no two IDs at the same **physical point**: the interpolated `point_mm` must
  be at least 1.0 mm from every other landmark, so a shared vertex declared
  through different triangles, new IDs or texture pixels up to 4 texels
  apart is still one point and the file is rejected;
- texture locations at least 0.25 texel apart;
- per frame, an RGB location within 0.05 px of another landmark's location
  in that frame never counts as another distinct landmark there. Two IDs
  that coincide in every co-visible frame are rejected outright;
- one observation per landmark and frame;
- integer triangle indices; nonnegative barycentrics summing to 1;
- finite numeric coordinates within texture and image bounds;
- boolean visibility;
- NaN, Inf, booleans or strings in numeric fields are rejected.

The file is copied into the prepare directory, and its hash is re-validated
by `render`. Further requirements:

- each landmark binds the texture/RGB SHA, triangle, barycentric, texture pixel
  and independently selected image pixel, plus its visibility;
- each image needs at least 3 visible landmarks, with a median error ≤ 5 px and
  a maximum ≤ 10 px;
- the texture binding must be within 4 texels;
- without such a file the result is `inconclusive`. No labels are fabricated
  and nothing is refined or optimized.

**Outputs** (new immutable directory, 1 GiB free-disk reserve checked):

- `prepared.json` with every input, source and runtime hash;
- `<frame>-samples.npz`, `<frame>-reference-depth-mm.npy` and
  `<frame>-reference-rgb.png`;
- `<frame>-side-by-side.png`: source RGB | reference | overlay | sample pixels.

### `render` (explicit parent-only command; GPU env; not run by the implementer)

```powershell
.cache\quality-windows\Scripts\python.exe tools\workbench_ycb_uv_proof.py render `
  --prepared artifacts/video-workbench/ycb-uv-proof/prepare-object-000018-v1 `
  --expected-prepared-sha256 <exact prepared.json SHA from prepare> `
  --out artifacts/video-workbench/ycb-uv-proof/render-object-000018-v1
```

**Before creating any output or importing a renderer**, `render` checks and
re-derives everything from authenticated inputs:

- the externally expected `prepared.json` SHA-256 (this is identity, not
  proof);
- strict JSON, rejecting duplicate keys and non-finite values;
- the exact generated file inventory: regular files only, no extras, size
  bounded, every hash checked;
- the current tool, thresholds, reference method, source pins and candidate
  pins;
- the candidate GLB and receipt, re-authenticated against their original
  pins. The recorded candidate path must be the canonical
  `<candidate dir>/candidate-model.glb`, so any other basename or form is
  rejected. The renderer loads a scratch copy of the **authenticated bytes**
  (re-hashed against the pin), never the recorded literal path;
- exactly two image rows (zero, one or three are rejected) in the fixed
  onboarding order, each with its frame ID, logical index, RGB SHA, strictly
  numeric K/R/t and exact row keys, all equal to the authenticated input. The
  image inventory, raster stats and `reference_ready` are reconciled with the
  re-derived results;
- the source-convention checks and landmark evaluation, re-run and compared
  with the recorded results and pass flags;
- the reference samples, recomputed on the CPU and required to equal the
  stored arrays. These are checked for exact keys, shapes and dtypes; finite
  values; distinct in-image pixels; valid barycentrics, depth and RGB; and
  counts matching the declaration. Fewer than 128 legitimate samples is
  `inconclusive`; samples are never padded.

`render` also refuses to start when the re-derived `source_convention_pass`
is false. After importing, and before any render, it checks that the
imported `pyrender`/`trimesh` packages are the directories whose files were
pinned, then re-hashes every pin.

**Failures:** any setup, render or cleanup failure writes `failure.json`, and
only the tool's own scratch directory is removed. `render.json` is written
only after a complete run.

**Sampler oracle:**

- the actual trimesh → pyrender texture uses pyrender's default sampler
  (`GL_LINEAR_MIPMAP_LINEAR` minification, repeat wrap). The CPU reference is
  base-level clamp bilinear. This is a **known oracle mismatch, not evidence
  of a wrong UV**;
- the serving comparison records the actual sampler state and is labelled
  `inconclusive: reference_sampler_mismatch`. It can set
  `serving_render_pass` to true only if the actual serving sampler is itself
  base-level clamp bilinear;
- a separate `base_level_sampler_control` re-renders the final onboarding
  image (candidate and opposite: 2 renders per object). It uses a
  **diagnostic renderer instance** whose texture samplers are set to
  `GL_LINEAR` min/mag and `CLAMP_TO_EDGE`; the production renderer and shaders
  are unchanged. This control can corroborate orientation but can **never**
  set `serving_render_pass`;
- a source-reviewed mipmapped reference remains future work.

- **What is rendered:** under the repository GPU stage, each onboarding image
  (640×480) is rendered with the serving `WorkbenchZeroStorageRenderer` twice:
  - with the candidate GLB;
  - with an opposite-UV hypothesis, in which only the `TEXCOORD_0` v values are
    flipped and every other byte is identical (checked; its hash is recorded).

  That is 4 renders per object, 12 in total.
- **What is compared:** each render against the frozen reference samples.
- **What is recorded:** allocation proofs, renders and comparison strips
  (source | reference | candidate | opposite).
- **Cleanup:** renderers are always closed, and the scratch candidate and
  opposite GLBs are deleted after use.

### Fixed acceptance (declared before any run)

| Check | Threshold |
|---|---|
| Depth | median ≤ 0.1 mm, P99 ≤ 0.5 mm |
| RGB per channel | median ≤ 2/255, P99 ≤ 8/255 |
| Discrimination | candidate median RGB error ≥ 4× lower than the opposite hypothesis, and opposite median > 8/255; otherwise inconclusive |

**Final status:**

- `inconclusive` unless all three passes hold;
- `failed` if any image fails;
- `eligible_for_review` only when everything passes. It is never promoted.

## Current state (CPU checks only; no rendering executed)

- **`prepare`** ran into temporary directories for all three objects using
  `.cache/quality-windows` Python (about 12 s each):
  - `source_convention_pass = true`;
  - trimesh 4.4.9 loaded UVs equal the raw PLY UVs within 4.5e-8;
  - projection parity is about 2e-6 px, and both ×1000 mutations fail;
  - samples: 1024 per image for objects 2 and 4, and 319/340 for object 18;
  - correspondence is `inconclusive` because no landmark file has been frozen.
- **System Python** without trimesh records the loader check as unavailable,
  so the pass stays false there.
- **Render phase:** implemented but **not executed**. Earlier prepare
  outputs were bound to the previous tool hash, so `render` will reject
  them; a fresh `prepare` is required.
- **Known risk to measure, not assume:** pyrender may sample the 4096² texture
  with mipmapped minification. The reference uses base-level bilinear sampling,
  so the RGB tolerances may report `inconclusive` or `fail` regardless of the
  UV convention. The thresholds must not be relaxed after results are seen.
