# Workbench offline portability audit

`tools/workbench_portability_audit.py` is an accounting audit only. It
authenticates the explicit runtime payload files of each requested row and
measures them against three targets: **20,000,000 bytes**, **20,000
triangles** and a **1024 px overlay-artwork edge**. Embedded CAD material
images are diagnostic only.

It makes no claim of mobile readiness, full-runtime fit, automatic geometry or
compression that preserves attachment. `mobile_verified` is always `false`, and
the full mobile runtime is unverified.

## Command

```
python tools/workbench_portability_audit.py \
  --prepared-inventory .cache/video-workbench/derived/validation/prepared-cad-target-inventory-v1.json \
  --artwork mug=.cache/video-workbench/runs/run-20261021T082701-5c96f003/neural-artwork.json \
  --artwork keyboard=... --artwork ranch=... --artwork cracker=... --artwork soup=... --artwork marker=... \
  [--surface-package DIR ...] --out NEW.json
```

- **Artwork map:** exactly one `--artwork CASE=PATH` for each inventory case.
  Missing, extra or duplicate cases are rejected with exit 2.
- **Surface-only:** `--surface-package DIR` can be given without an inventory.
- **No discovery:** the audit never searches for runs or files.
- **Paths:** relative CLI paths and inventory strings (Windows `\` is
  normalised to `/`) are relative to the repository root, and absolute paths
  must lie inside it. These are rejected:
  - drive or stream syntax (`:`), `..`, empty parts, or escaping the root;
  - any symlink or junction on the path;
  - a path that doesn't resolve to itself.
- **Output:** one fresh `.json` smaller than 2 MiB, published atomically with
  the checked publisher (`workbench_image_io.publish_new_bytes`). An existing
  file is never overwritten, and no assets are copied.
- **Budget check before writing (`output_budget`):** this uses `lstat` sizes
  only and reads no file contents, so no clip records are read.
  - It counts the same trees as the workspace's derived usage: `derived`,
    `scratch`, `runs`, `masks` and `suite` under the fixed
    `.cache/video-workbench` root.
  - Regular-file sizes are counted. Links and junctions are counted by their own
    `lstat` size and are never followed. Directory metadata isn't counted.
  - It reserves 2 × 2 MiB: the output cap plus one same-size temp file at the
    peak before the hard link.
  - Derived usage plus the reserve must stay within 1 GiB, and free disk minus
    the reserve must stay at least 1 GiB.
  - The audit adds no new inputs, so `new_input_bytes_added` is 0. The 250 MB
    new-input budget is **not** revalidated here; the parent verifies it
    separately.
  - Everything printed after publication (the output SHA and summary) is
    prepared beforehand, so no filesystem step can fail after the commit.

## Inputs read

Only these are read:

- the inventory (provenance only; its boolean claims are not trusted);
- each row's metric bundle and the GLB it names;
- each explicit `neural-artwork.json`;
- each package's `surface.json` and its FileRefs;
- the hashes of the audit's own sources and the runtime identity.

These are never read: results, reports, evaluator data, reference poses, masks,
seed RGB, clip records and checkpoints. The head (`27e06a51…`) and base
(`f7d127ab…`) pins are recorded as `verification: not_read` and
`excluded_from_surface_bytes: true`.

## Accounting (per row)

| Field | Meaning |
|---|---|
| `reference_bytes` | Encoded physical bytes per dependency reference, with duplicates kept |
| `unique_path_bytes` | Bytes of distinct dependency paths |
| `unique_file_object_bytes` | Deduplicated by filesystem identity (dev, inode). Diagnostic only; `null` if any identity is unknown |
| `generated_artwork_raw_bytes` | The regenerated builtin RGB `uint8` `nbytes` (256×256×3 = 196,608) |
| `portable_budget_bytes` | `unique_path_bytes + generated_artwork_raw_bytes`; `null` if any part is unknown. Compared with 20 M |
| `portable_lower_bound_bytes` | Known bytes only; can prove a FAIL, never a PASS |
| `allocated_disk_bytes` | Always `null`; a stat size is not disk allocation |

The budget never subtracts hard links or files with equal content.

- **Prepared controls:** the payload is the bundle, the GLB and the artwork
  record. Embedded GLB buffers and images are already inside the GLB bytes and
  are counted once.
- **Packages:** the payload is `surface.json`, every numeric FileRef and every
  camera depth.

## Prepared metric CAD control (`audit_prepared_control`)

These controls are a MetricBundle, a GLB and `neural-artwork.json`. They are
**not** SurfacePackages, and nothing is converted or exported.

**Bundle and pins**
- `check_bundle` applies the same field checks as `MetricBundle.from_file`
  to the JSON already read; no loader is called. It requires:
  - the exact `workbench_metric_bundle_v1` keys;
  - bounded strings for `family`, `glb_path` and `seed_label`, and a valid
    `seed_frame_id`;
  - `obj_id` an integer from 0 to 2³¹−1, and a `glb_sha256` hash;
  - origin `integer_centers` or `corner`;
  - provenance `measured_metric_cad` or `dataset_metric_cad`.
- A prepared artwork control also needs a seed. A null `seed_pose_m` is a FAIL.
  The pose must be a finite, proper 4×4 (`wc._pose`), and its label must start
  with `controlled_init:` or `onboarding:`; an evaluator label is rejected.
- The bundle's `glb_path` is checked like any other input path.
- The GLB's physical hash must equal the bundle pin. The CLI also checks it
  against the inventory pin.

**Artwork record**
- Strict JSON. `record_sha256` must be the canonical hash of the record without
  that field.
- `glb_sha256`, `anchor_frame_id` (the bundle's seed frame), `seed_label` and
  the principal point origin must match the bundle, and units must be metres.
- The marker must be valid, with the same artwork hash. The marker and patch
  faces must lie within `mesh_faces`.
- `mesh_faces` must equal the producer-decoded count. That count comes from
  `workbench_neural.glb_mesh(max_faces=200_000)`, called only after the strict
  audit parser has accepted the file.

**Artwork regeneration**
- Only `fixed_builtin_v1` checkerboard or directional artwork is regenerated,
  with `builtin_artwork` and `array_sha256`. Hash, shape, dtype and `nbytes`
  must all match.
- Any other artwork is UNVERIFIED.
- The audit never anchors, selects, reads a seed mask, renders or constructs
  `NeuralArtwork`.

**Triangle counts (reported separately)**

| Field | Meaning |
|---|---|
| `raw_indexed_triangles` | Each mesh counted once |
| `scene_instanced_triangles` | glTF scene semantics |
| `producer_traversal_triangles` | The `glb_mesh` traversal |
| `producer_degenerate_triangles` / `producer_filtered_triangles` | Degenerate faces, and the count after `glb_mesh` drops them |
| `target_triangles` | The largest of the raw, scene and producer-traversal counts; degenerate faces are never deducted |

**Inventory cross-check (CLI)**
- Bundle and GLB hashes, GLB path and bytes, and `bundle_plus_glb_bytes` must
  match.
- `triangles` must equal the producer-decoded count.
- The artwork edge must match the record.

## Strict GLB subset (`parse_glb`)

The parser accepts dense, embedded triangle geometry only. It checks:

- **Container:** header and length; JSON chunk first; at most one BIN chunk,
  directly after it; 4-byte aligned chunks within bounds.
- **JSON:** strict (no duplicate keys, no NaN); asset version 2.x.
- **Buffers:** buffer 0 must match the BIN chunk length.
- **Buffer views:** bounds and stride.
- **Accessors:** every accessor's bounds and alignment.
- **Primitives:** `POSITION` must be float32 VEC3; indices must be unsigned
  SCALAR, with a count that is a multiple of 3 and every index in range.
- **Nodes:** strict trees (no cycles, at most one parent each), valid
  mesh and scene references, finite TRS or matrix transforms, and finite
  transformed vertices.

Failure handling:

| Input | Result |
|---|---|
| `extensionsRequired` | UNVERIFIED |
| Sparse accessor, or accessor without a buffer view | UNVERIFIED |
| Non-triangle mode, or non-float positions | UNVERIFIED |
| Data-URI or external buffers | UNVERIFIED |
| External image URIs | UNVERIFIED; recorded, **never opened** |
| Anything malformed | FAIL |

Semantics outside the static subset are recorded in `geometry_unsupported` and
make the row UNVERIFIED:

- `extensions` on the root, a node, a mesh or a primitive (for example
  `EXT_mesh_gpu_instancing`);
- skins or node `skin`;
- animations;
- morph `targets`, or mesh or node `weights`.

None of these extensions is implemented. Strict bounds and counts still run, and
the triangle count is then a lower bound (`target_triangles_is_lower_bound`),
so it can prove a FAIL but never a PASS. The producer decoder is **not** called
for such files.

Embedded image sizes are read from PNG IHDR or JPEG SOF headers without
decoding.

## SurfacePackage (`audit_surface_package`)

**Manifest and contract**
- `surface.json` is at most 1 MB, strict, and has the schema.
- `wc.SurfacePackage.from_dict` is run. A contract violation is a FAIL; this
  includes its own 20 MB and 20 k caps, so an oversized manifest is both
  closure FAIL and target FAIL.

**Files**
- All 5 numeric FileRefs plus every `camera.depth` must be top-level regular
  files.
- Each must match its FileRef size, then its hash, with duplicates kept.
- Nested, missing, linked or tampered files are a FAIL.

**NPY arrays**
- Before allocation: the header must be version 1 or 2, the dtype must not be
  an object type, and dtype, shape, row bounds and exact payload length must
  match.
- After loading with `np.load(allow_pickle=False)`:
  - values are finite;
  - counts equal `vertex_count` and `face_count`, and every index is in range;
  - there is one non-negative support value per face;
  - descriptor point and value rows are present together and equal in number;
  - `validate_mesh` passes.

**Other checks**
- Depth bytes are hash-verified, but there is no depth encoding contract, so a
  package with depth is UNVERIFIED.
- Builtin artwork is regenerated. Imported or unknown artwork has no locator and
  is UNVERIFIED. A package without artwork has no artwork dependency, and its
  texture target is `null`.
- Image and mask hashes, the reconstruction history and preview provenance are
  listed as not runtime payload and are not dereferenced. This is not a
  reproducibility audit.
- Unreferenced entries, sidecars or links in the package directory prevent a
  standalone pass (UNVERIFIED).
- An `unavailable` package is UNAVAILABLE with null targets. Its tiny manifest
  never passes, and an invalid manifest is a FAIL.
- A structurally VERIFIED package, with or without a descriptor bank, is byte
  and triangle accounting only. Its runtime usability stays **unverified**, and
  no tracker admission is implied (`package.runtime_usability`).

## Outcomes and exit codes

Each row has `closure_status` (`VERIFIED`, `UNVERIFIED`, `FAIL` or
`UNAVAILABLE`), `bytes_target`, `triangles_target` and `artwork_target` (each
`PASS`, `FAIL` or `null`), and `mobile_verified: false`.

- **PASS** only when the closure is VERIFIED and the measured value is within
  the limit.
- **FAIL** whenever a measured value, or the known lower bound for bytes,
  exceeds the limit, even if the closure is incomplete.
- **`null`** otherwise.

| Exit | When |
|---|---|
| 0 | Every requested row is VERIFIED and all three targets PASS |
| 1 | Confirmed corruption or inconsistency, or any target FAIL |
| 3 | Incomplete, unsupported or unavailable, with no confirmed failure |
| 2 | Invalid CLI, an output or write error (the record goes to stderr), a guard violation, or a model runtime already imported at start |

**Closure:** source hashes (this tool and the contract, geometry, neural,
reconstruction, image I/O and telemetry modules) and runtime identity (Python
executable and version, NumPy version, platform) are recorded before and after.
Every input read is re-hashed after the audit. Any change makes every row FAIL.

**Guards** apply for every API call and are restored on every path. They
raise a `BaseException` on:

- imports of torch, torchvision, sam2, pyrender, OpenGL, pycolmap,
  `quality_gotrack*` or `workbench_render`;
- the GPU stage, `GoTrackMetricBackend`, `NeuralSession`, `NeuralArtwork`,
  `segment_prompted`, `render_marker` and `rasterize`.

## Decisions to review

- **`artwork_target` gates only the regenerated overlay artwork** (edge
  ≤ 1024). Embedded GLB material images are reported separately in
  `embedded_texture_diagnostic` (count, largest edge, number over 1024, number
  with unknown size) and never gate the target. Their bytes are counted once,
  inside the GLB. A 2048 px material with 256 px artwork does not fail the
  artwork target.
- **Unavailable diagnostics:** `embedded_texture_diagnostic.inspection_status`
  is `inspected` only when `parse_glb` completed. Its counts are then measured,
  so a GLB with no images reports a real `0`. If the GLB was unread or invalid,
  or an unsupported feature (for example a required `KHR_texture_transform`)
  stopped parsing, the status is `not_inspected`. `images`, `max_edge_px`,
  `over_1024` and `unknown_dimensions` are then `null`, meaning unknown, not
  zero.
- **Triangle target rule:** the target uses the largest of the raw, scene and
  producer-traversal counts.
- **GLBs without scenes:** these use all parentless nodes, which is
  implementation-defined. The producer traversal (all nodes) is reported
  separately.

## Expected on the actual inputs (not run here)

Per the inventory, mug's 31,606 triangles are a triangle FAIL even with a
verified closure, so the run exits 1. The other five are at most 20 k
triangles.

## API

| Function | Purpose |
|---|---|
| `audit_prepared_control(bundle_path, artwork_record_path, *, root)` | One prepared-control row |
| `audit_surface_package(package_dir, *, root)` | One package row |
| `main(argv=None)` | The CLI |
| `parse_glb(raw)` | Pure GLB check and counts |
| `check_bundle(doc)` | Pure bundle field checks |
| `output_budget(out, *, workspace=None)` | Stat-only budget check before writing |
| `account(references, generated_artwork_raw_bytes)` | Pure accounting |
| `check_artwork_record(record, *, bundle, producer_mesh_faces)` | Artwork record checks |
| `regenerate_builtin(kind, edge, sha)` | Builtin artwork regeneration |
| `image_dims(data)` | PNG/JPEG header size |
| `resolve_input(root, value, *, kind)` | Path containment checks |
| `crosscheck_inventory(row, inv_row)` | Inventory cross-check |
| `finalize_row(row)` | Closure status and targets |
| `exit_code(rows)` | The exit-code rule |
| `audit_guards()` | The guards |
| `validate_output(out, root)` | Output path checks |
| `source_pins()`, `runtime_pins()` | Closure pins |

The functions work on tiny manufactured assets under any `root`.
