# Saved-window offline video export

`python -m tools.workbench_offline_export --help` describes the local CPU command.
It uses a finalized saved run, one saved neural role, a self-contained local
VisualizeIT project snapshot with a selected PNG, its exact expected revision,
an explicitly selected local metric bundle and SHA256, and admission metadata.
The snapshot's recipes and descriptive extensions are never executed or used to
choose geometry, models, or source paths.

Example (replace placeholders with local authenticated evidence):

```powershell
python -m tools.workbench_offline_export --workspace .cache/workbench --run-id saved-run --role primary_neural --project selected.visualizeit-project.json --expected-revision saved-revision --bundle metric-bundle.json --bundle-sha256 BUNDLE_SHA256 --admission-dir admission --output .cache/workbench/derived/new-export --playback-fps 24
```

The output must be a new direct child of the existing workspace `derived`
directory. Existing destinations refuse. No weights are loaded and no new pose,
tracking, segmentation, or quality measurement runs. Only the complete scheduled
window is exported, with every row retained in saved order. Lost, recovering, or
otherwise unavailable metric-pose/mask rows retain original RGB. A missing source
frame or integrity mismatch refuses the whole export. Null saved RGB hashes on
failure rows are covered by registered source-file/media authentication and a new
decoded RGB hash; existing row RGB hashes must match exactly.
Video-file sources additionally require both a nonnull registered SHA256 digest
and byte count before any decoding. A digest first computed during export is
insufficient source authentication. The registered clip is strictly parsed and
hashed from one bounded byte snapshot; its path is rehashed before publication.

An optional `--placement` JSON object must contain exactly `scale`,
`rotation_deg`, `offset_t_m`, `offset_b_m`, and `opacity`. Defaults are
`{"scale":1,"rotation_deg":0,"offset_t_m":0,"offset_b_m":0,"opacity":1}`.
Scale is bounded to [0.05,4], rotation to [-180,180] degrees, each tangent/bitangent
offset to [-0.5,0.5] metres, opacity to [0,1]. Booleans, nonfinite numbers, and
unknown keys refuse. The longest PNG dimension is saved marker size times scale;
the other preserves aspect ratio. PNG top maps to +saved bitangent and PNG right
to +saved tangent before rotation. Anchor position is the saved mesh face's
barycentric point. Translation and geometry use metres and projection uses the
authenticated effective camera K. Behind-camera, backfacing, degenerate or
unsupported planes suppress. Pixels outside the same-frame observed mask retain
source RGB. Nearest-neighbour RGBA warping uses straight-alpha composition with
explicit round-half-up integer output; transparent RGB contributes no color.

This is a **tangent-plane approximation**, labelled
`offline_saved_pose_planar_artwork`. It provides no curved wrapping, physical
surface certainty, mesh occlusion, or new current-tracking claim. Original
frame/pose/state/model identity and nullable/nonuniform timestamps remain in the
sidecar. Caller FPS sets an explicit constant-rate playback clock and does not
recover source timing. Audio is not copied.

The summary model identity is copied from the selected saved role: a frozen
baseline has no installed/export primary head. Recorded pins describe the saved
producer; the exporter never loads those weights. `source_clip` preserves the
clip evidence role, source clock kind/source, and inert provenance, so synthetic
controls remain labelled as such. `frame_accounting` records the complete
scheduled/exported count and rendered/suppressed counts by reason. These are
software accounting counts, not tracking quality metrics.

Output contains `playback.avi` (FFV1 only) and `manifest.json`. There is no codec
fallback or package installation. Even dimensions, <=1280 pixel edges, <=2MP and
<=1000 scheduled frames bound work. Input RGB is streamed one frame at a time;
no image dump is written. A conservative incompressible allowance and repeated
Workspace budget checks enforce the existing 1GiB derived/scratch limit and 1GiB
free reserve. The exporter releases and decodes the video, checking every RGB
hash, dimension, count and order, then reauthenticates all sources, input files,
masks, bundle/GLB, admission metadata, selected PNG/project revision and decoded
source RGB before atomically publishing its private staging directory. Failure
removes only that staging directory and publishes no success receipt.
Publication refuses a competing destination even if it appears after the final
existence check. Windows uses its no-replace directory rename semantics. Linux
requires libc `renameat2` with `RENAME_NOREPLACE`; missing kernel/filesystem/libc
support refuses without a fallback. Other platforms explicitly refuse
publication. Staging and destination must remain on the same filesystem.
The reopened stream's FPS header must also agree with explicit playback FPS
(relative/absolute tolerance 1e-6); otherwise publication refuses.

Python API: `export_saved_window(workspace, *, run_id, role, project_path,
expected_revision, bundle_path, bundle_sha256, admission_dir, output_dir,
playback_fps, placement=None, repo_root=ROOT)` returns the verified manifest.
`compose_planar(rgb, rgba, mask, pose, K, vertices, faces, marker, placement=None)`
returns `(RGB, reason, projected_corners_or_None)` for manufactured checks.
Run `python -m unittest tools.test_workbench_offline_export` for analytical CPU
geometry, alpha, orientation, suppression and import guard tests. Independent
integration tests own full finalized synthetic fixtures and actual codec checks.

For a portable VM check, use these commands from a checkout containing the exact
reviewed new exporter, CLI, unit test, integration test, and this document:

```powershell
python -B -m unittest tools.test_workbench_offline_export tools.test_workbench_offline_export_integration
python -B -m tools.workbench_offline_export --help
```

The integration fixture identity is `offline-planar-export-fixture-v1`.
`tools.test_workbench_offline_export_integration.make_fixture(root)` manufactures
its own plane GLB, eight PNGs, masks, project, and metadata-only saved run under
the provided workspace-local root. Tests export only the six scheduled frames
and exercise the ordinary CLI as well as the callable API. They do not copy
private footage or weights. The synthetic declared producer metadata is not
evidence of actual model execution or a serving admission. Run against matching
reviewed sources: an older benchmark checkout is not an equivalent test.
