# YCB prepared-metric CAD controls

`tools/workbench_ycb_controls.py` exports **known-pose controlled diagnostic** inputs for three
supplemental frozen YCB-Video clips. It is not automatic RGB preparation, and its outputs support
no claim about independent accuracy, tracking quality or mobile readiness.

## Scope and pins

| Object | Canonical id | Frozen PLY SHA-256 | Trusted texture SHA-256 (bytes) |
| --- | --- | --- | --- |
| 2 | `003_cracker_box` | `121df764d2214a16d59fd77a31e168ca9e79ba0f30a5e529de86c8cd19c1686f` | `9a04bc0e7a10766e44aee4bdfc719462b0b6cc53d186bc6d4e8b890691acdf59` (10406584) |
| 4 | `005_tomato_soup_can` | `fea7ae381b760b4852463fce4effa53fe9535dc5c8c1ee89de2be85a30b32211` | `33183a894eb75ae04ae5188a3ae689fe8d06ec6b9f9624da1bf853c2a25e70d2` (9350703) |
| 18 | `040_large_marker` | `aa6f0f674e68e4affd95f7d21ae052539fe817590492d485f12e34dd80489a2a` | `0351acbfd666baabc23261c3f7d5d7645c5b55bd47cef0dbc101572552c888fb` (4403500) |

- **Inference manifest:** `.cache/prototype-sprint/dataset-pilot-v1/inference.json`
  (SHA-256 `9b936de31c5511e3644e065538dedf47d0cdcd22f2ba17cee083e75a25403945`).
- **Texture source:** the immutable `TRUSTED_TEXTURES` map. Each entry binds the object id,
  dataset revision `5c2c4aa229800355648cd268040aa814f8dc94f0`, the archive
  `https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/5c2c4aa229800355648cd268040aa814f8dc94f0/ycbv_models.zip`,
  the member `models/obj_0000NN.png`, the SHA-256, the byte size and 4096×4096 dimensions.
- **What is never read:** `evaluator.json`, scored labels, scored-frame poses and RGB/depth
  pixels.

## Usage

```
python tools/workbench_ycb_controls.py --object-id {2,4,18} --out NEW_DIR \
    --principal-origin {integer_centers,corner} \
    [--allow-dataset-onboarding-control] [--manifest PATH] [--texture OBJ_PNG]
```

- **`--principal-origin`** is required. The operator declares it; the tool never infers it.
- **`--allow-dataset-onboarding-control`**: without this flag, no pose is read. With it, the
  tool uses only the final onboarding row (the highest `logicalIndex`):
  - `seed_pose_m`: rotation `cam_R_m2c`, translation `cam_t_m2c` × 0.001 in metres;
  - `seed_label`: starts with `controlled_init:dataset_onboarding:`;
  - `seed_frame_id`: `sXXXXXX`, the BOP image `frameId`.
- **`--texture`** goes through the authenticated gate below.

## Statuses

| Status | When | Outputs | Exit code |
| --- | --- | --- | --- |
| `blocked_unsupported_appearance` | No vertex colours and no texture supplied | `model.glb` (geometry only), receipt | 3 |
| `blocked_unverified_uv` | Authenticated official texture | `candidate-model.glb`, receipt; **no metric bundle** | 3 |
| `prepared_control` | PLY vertex colours, so no UVs are involved | `model.glb`, `metric-bundle.json`, receipt | 0 |
| error | Any failed check | nothing published | 2 |

A `blocked_unverified_uv` candidate must not be consumed by the neural pipeline. The receipt
records `uv_orientation_verified: false` and `neural_consumption_permitted: false`. Promotion
requires a separate exact-source orientation proof, which the parent schedules after verifying
the onboarding RGB, followed by review of that proof contract. **No flag or caller assertion
bypasses this.**

## Texture admission gate (runs before any GLB is built)

1. The PLY's `TextureFile` comment must be exactly the trusted member name, and the supplied
   file must have that name. A matching name is necessary but never sufficient.
2. The manifest's `datasetRevision` must equal the trusted revision.
3. The byte size and SHA-256 must equal the trusted pin. A hash mismatch is rejected even when
   the name matches.
4. The file must decode fully with the existing Pillow: `verify()`, then `load()` and
   conversion to an array. The decoded format must be PNG, and the decoded size, the header
   size and the array shape must all equal the pinned 4096×4096.

A malformed, truncated, renamed or substituted texture fails before anything is published.

## Conversion

- **Units:** BOP millimetres × 0.001 give metres. This is applied exactly once, to the vertex
  positions and to the seed translation.
- **Basis:** the original right-handed BOP model frame is kept. There is no axis swap,
  recentring or rescaling.
- **Face order:** source face vertex order is preserved. The receipt records the **aggregate**
  signed volume. A positive value says only that the total is positive; it is not a per-face
  orientation proof.
- **UVs:** the exporter writes `TEXCOORD_0 = (u, 1 − v)`, assuming the PLY `texture_v` has a
  bottom-left origin. This is an **unverified assumption**. The UV round-trip check reapplies
  the same flip, so it only confirms serialisation and is not evidence of orientation. Source
  normals are renormalised.
- **Material:** a flat `KHR_materials_unlit` made only from authenticated source appearance.
  It is embedded unchanged, with linear filtering and clamp-to-edge. Nothing is invented.

All outputs are published atomically into a new directory and never overwritten. Before
publishing, the staged GLB is checked with `bench.workbench_neural.glb_inventory`, which imports
only NumPy. Bounds: the GLB is at most 20 MB, derived output is under 1 GiB, and at least 1 GiB
of disk must remain free.

## Validation records (CPU only)

### Geometry only, 2026-10-21 (`validation-20261021/`)

Status `blocked_unsupported_appearance`. These records are retained unchanged.

| Object | GLB SHA-256 | Bytes | Vertices / triangles | Extent (m) |
| --- | --- | --- | --- | --- |
| 2 | `a18158d97253e7f482e82b6d9ec7b54a0540c6893cc370b5c5af481616748e53` | 455204 | 8291 / 15728 | 0.07173 × 0.16398 × 0.21349 |
| 4 | `84bc916a435ba05be17dfb59c150ec2413f827a190023acd142cf83e24880e9c` | 458832 | 8404 / 15728 | 0.06785 × 0.06775 × 0.10204 |
| 18 | `6cd9b970a72881d1727327399ef47f79f4b367073e000cb5757ad65a0bcd9a40` | 470620 | 14043 / 15728 | 0.02104 × 0.12084 × 0.01888 |

### Authenticated textures, 2026-10-21 (`validation-authenticated-v1/`)

Run with `--principal-origin integer_centers` and the textures in
`.cache/video-workbench/imports/ycb-textures-v1/`. No seed flag was used.

| Object | Status | Candidate GLB SHA-256 | Bytes | Receipt SHA-256 |
| --- | --- | --- | --- | --- |
| 2 | blocked_unverified_uv | `a5d6e179b5020406a5f3f0143d3958cb4a3fda749863a89e2133d866356b65de` | 10862316 | `26f70830c28614333c2098899a34f12dee5e00158e902ea2c646ef12d96989f7` |
| 4 | blocked_unverified_uv | `2bc906380ceccda6f4524db667b4f90f13a43cf078f393ca6edf6751e9496e80` | 9810064 | `d7fa080363df14b6acd9c99623372313f4c6dfb222ab364f497a57e5d6aefaec` |
| 18 | blocked_unverified_uv | `980c1615a22799b4c14b430bf4c239ca8460a492efb286c8774e34ab8bc216a8` | 4874648 | `08b11f7e8c8d0980985c477aa7425467092c53defd772814ecb220dcb22a20cd` |

- **Texture checks:** every texture matched its trusted pin and decoded fully to RGB
  4096×4096×3.
- **Geometry:** identical to the geometry-only run, including the aggregate signed volume.
- **Outputs:** no metric bundle and no seed were written. The total new derived size is
  25,563,135 bytes.
