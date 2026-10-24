# YCB raw-source GPU UV reference

`tools/workbench_ycb_gpu_reference.py` is a separate **shared-backend relative
experiment**. It is not a repair of the accepted CPU proof
(`tools/workbench_ycb_uv_proof.py`, frozen at `764f58f6…`), which it calls
read-only and never edits.

## Why

The accepted proof keeps the serving render `inconclusive`: the serving
texture sampler uses mipmapped minification and repeat wrap, while its CPU
oracle uses base-level clamp bilinear sampling. This experiment draws an
independent reference built from the **authenticated original PLY** through
the **same** pyrender sampler, flat shader, material settings and zero-storage
framebuffer as the GLB serving path. The GLB loader/conversion boundary can
therefore be tested without modelling mipmaps.

## Reference construction

- **Positions:** PLY `positions_mm` converted to `float32(float64(mm)/1000)`
  metres, in the source face order.
- **UVs:** raw `(texture_u, texture_v)`, with no v flip.
- **Normals:** independent face normals.
- **Layout:** unindexed per-corner arrays, exactly as
  `Mesh.from_trimesh(smooth=False)` lays out the serving mesh.
- **Not used:** GLB vertex arrays, the exporter/controls UV helper and any glTF
  round trip.
- **Texture:** a fresh `pyrender.Texture` from the authenticated official PNG,
  as RGBA uint8 with alpha 255. Its `source` must equal that array, not the
  candidate pixels. It uses a default `pyrender.Sampler()`, the same as the
  serving texture.
- **Material:** only scalar settings are copied from the constructor-loaded
  candidate material, for state equality: alpha mode/cutoff, double-sided,
  smooth, wireframe, base-colour factor, metallic, roughness and emissive.
- **Renderer:** a fresh `qg.TexturedRenderer` (`legacy_v1`,
  `integer_centers_v1`, unlit, no multisampling) wrapped by
  `WorkbenchZeroStorageRenderer(inner=…)`.
  - Its constructor necessarily loads the candidate GLB. That mesh node is
    removed **before the first draw** and never used for reference arrays.
  - `inner.asset_sha256` is relabelled `raw_ply_source_reference:<hash>`.
- **Unchanged:** the production factory, sources, shaders and serving sampler.

## Commands

**`preflight`** is CPU only and writes no output. It doesn't import pyrender,
OpenGL or torch. It checks:

- the frozen proof module SHA and path;
- `proof.validate_prepared` with the required externally expected SHA. This
  re-authenticates the candidate/receipt pins, onboarding rows 0/9 (RGB, K and
  poses), samples and landmarks, and re-derives the source convention, which
  must pass;
- that the source bindings and raw-source arrays/texture are valid;
- the scratch GLB size (≤ 32 MiB each);
- the runtime pins.

```powershell
.cache\quality-windows\Scripts\python.exe tools\workbench_ycb_gpu_reference.py preflight `
  --prepared .cache/video-workbench/derived/uv-proof/attempt-02/object-000018 `
  --expected-prepared-sha256 <exact prepared.json SHA>
```

**`run`** is an explicit parent-scheduled GPU command; the implementer has
not executed it.

```powershell
.cache\quality-windows\Scripts\python.exe tools\workbench_ycb_gpu_reference.py run `
  --prepared .cache/video-workbench/derived/uv-proof/attempt-02/object-000018 `
  --expected-prepared-sha256 <exact prepared.json SHA> `
  --out .cache/video-workbench/derived/uv-gpu-reference/attempt-01/object-000018
```

`run` does the following:

1. Checks that the resolved `--out` lies **strictly inside**
   `.cache/video-workbench/derived`; anything outside is rejected before any
   work. It then runs preflight, the budget checks and creates a fresh output
   directory (1 GiB global derived budget, 1 GiB free reserve, 128 MiB own
   cap).
2. Writes scratch copies of the canonical candidate bytes and of the
   opposite-UV GLB (only `TEXCOORD_0` v is flipped, verified by
   `proof.opposite_uv_glb`), and verifies their SHA-256.
3. Checks after import that the loaded pyrender/trimesh are the pinned
   packages, and checks the proof and runtime pins again.
4. Takes **one `gpu_stage`** for the whole experiment.
5. For each onboarding frame:
   - constructs the reference, candidate and opposite renderers;
   - asserts their scene, primitive-format, material, sampler and texture-format
     equality **before any draw**;
   - requires all three `Texture.source` arrays to equal the independently
     decoded authenticated PNG RGBA exactly (shape, dtype and every pixel)
     before any draw, via the pure helper `require_texture_sources`. Shape,
     dtype and SHA evidence is recorded for each;
   - draws each with the unchanged default serving samplers (no base-level
     override);
   - records the allocation proof and the actual GL texture state (filters,
     wrap, base/max level, level 0/1 size and format), which must be equal
     across hypotheses. It checks that pre/post-draw state is unchanged and
     records draw timing (measured only);
   - saves RGB, depth (mm, float32) and a strip: source RGB | reference |
     candidate | opposite.

   That is 6 renders per object, 18 in total.
6. Closes every renderer. Any setup, render or cleanup failure writes
   `failure.json` with the partial frames, and the scratch folder is always
   removed. `gpu-reference.json` is written only after a complete run.

## Comparison (fixed thresholds of the frozen proof)

Candidate and opposite are compared with the **GPU reference** (not the CPU
colour or depth) at **all** frozen CPU interior sample pixels. There is no
dropping, registration, rescaling, pixel shift or fitting, and no natural-RGB
full-image metric.

| Check | Rule |
|---|---|
| Samples | 128..1024 distinct validated pixels, otherwise `inconclusive` |
| Coverage | reference incomplete → `inconclusive`; candidate missing/non-finite depth → `fail` |
| Opposite geometry | full-image mask identical to the candidate and depth within thresholds, otherwise `inconclusive` |
| Depth | candidate vs reference median ≤ 0.1 mm, P99 ≤ 0.5 mm, otherwise `fail` |
| Ambiguity | candidate RGB equal to opposite RGB at the samples → `inconclusive` |
| Discrimination | pooled candidate median RGB error × 4 ≤ the opposite's, and the opposite's median > 8/255, otherwise `inconclusive` |
| RGB | per channel: median ≤ 2/255 and P99 ≤ 8/255, otherwise `fail` |

**Results:**

- `relative_serving_uv_convention_pass` is `true` only if both frames pass;
  any failing frame makes it `fail`.
- `onboarding_correspondence_pass` is re-derived by the validated proof.
- `final_status` is `eligible_for_review` only when both are true. It is
  `failed` if either fails; otherwise it is `inconclusive`.
- Without frozen landmarks, eligibility stays false even if the convention
  passes. Nothing is ever promoted, and the proof's `serving_render_pass` is
  never rewritten.

## Limitations (recorded in every receipt)

- Reference and candidate share the pyrender rasterizer, texture upload, mipmap
  generation, sampler, shader and framebuffer, so common-mode faults are
  invisible.
- This is relative evidence of the conversion boundary only. Printed-landmark
  correspondence is still required before any promotion.
- It makes no claim of independent tracking accuracy, mobile readiness or
  automatic RGB preparation.
- Backend runtime pins (`pyrender` primitive/utils/shader_program/flat.vert/
  scene/renderer/offscreen and `bench` workbench_render/workbench_telemetry/
  quality_render_stability) were recorded by the implementer from the actual GPU
  env and are pending Sol review. They are enforced exactly.

## Current state (CPU only)

- `preflight` passes for objects 2, 4 and 18 (parent `attempt-02`
  preparations). The runtime pins match, samples are 1024/1024, 1024/1024 and
  319/340, and correspondence is `inconclusive` because no landmarks are
  frozen.
- In a CPU-only emulation of the constructor path for object 18 (trimesh load
  plus `Mesh.from_trimesh`; no GL context, renderer or draw), the reference and
  candidate states are equal apart from content:
  - the same default sampler (wrap 10497 = repeat, filters unset);
  - the same material, buffer flags and shapes;
  - both texture sources equal the PNG RGBA;
  - positions are bit-identical, and UVs differ by at most 6e-8.
- The GPU `run` has **not** been executed.
