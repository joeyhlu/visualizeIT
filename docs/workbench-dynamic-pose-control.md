# Synthetic dynamic pose control (`tools/workbench_dynamic_pose_control.py`, v2)

## Question

This is a bounded synthetic experiment with two questions:
1. Does an *exact geometric flow* recover nonzero pose changes through the actual production
   `WorkbenchIntegerRefiner.trace`? The trace covers the perspective crop, the legacy unlit
   integer-centre template render, the production correspondence gates, RANSAC + LM PnP and the
   crop inverse.
2. Does the unchanged new head (27e06 on base f7d1) stay stable on the same synthetic input?

**Labels:** `synthetic_only`, `human_accuracy=false`, `phone_ready=false`, `development_only=true`,
`automatic_rgb_preparation=false`, `scored_inputs_read=false`,
`oracle_network_is_diagnostic_not_trained_head=true`, and `physical_pose_correction_claimed=false`.

**Not read:** scored inputs, `evaluator.json`, dataset depth or masks, and real RGB.

## CLI

```
python -B tools/workbench_dynamic_pose_control.py preflight              # manufactured CPU checks only; writes nothing
python -B tools/workbench_dynamic_pose_control.py prepare --out .cache/video-workbench/derived/dynamic-pose-control/plan-v2
python -B tools/workbench_dynamic_pose_control.py run --plan <plan.json> --expected-plan-sha256 <external pin> \
    --out .cache/video-workbench/derived/dynamic-pose-control/run-v2 --device cuda      # parent-scheduled single GPU stage
```

- **Interpreter:** run `prepare` and `run` in the pinned GPU Python. `plan.runtime` is compared
  exactly.
- **Approval:** no `--force` and no automatic approval.
- **Plan check in `run`:** before any admission or GPU work, `run` **rebuilds the whole plan from
  the authenticated inputs** and requires exact equality. A self-consistent plan with any changed
  relationship is refused, even if its external SHA is correct.

## Synthetic target (finding 1)

The authenticated bundle seeds are slightly non-orthogonal:

| Object | Raw max\|RᵀR − I\| | det | Projection delta |
| --- | --- | --- | --- |
| 2 | 1.41e-6 | 0.99999953 | 7.6e-7 |
| 4 | 8.91e-7 | 0.99999928 | 4.6e-7 |
| 18 | 1.25e-6 | 1.00000092 | 6.6e-7 |

The previous `acos(trace)` error therefore reported about 0.055–0.069° for P\* compared with
itself.

**Synthetic experiment only:** `synthetic_target_pose(seed)` returns the deterministic SVD SO(3)
projection `R = U diag(1, 1, det(U Vᵀ)) Vᵀ`, with the translation unchanged. It rejects a raw seed
with max|RᵀR − I| > 1e-4, or a projection delta max|R_proper − R_raw| > 1e-4.

**Recorded per object:**
- `raw_seed_pose_m`;
- `synthetic_target` (`projection_max_abs_delta`, `raw_orthogonality_max_abs`,
  `raw_determinant`, `target_orthogonality_max_abs`, `target_determinant`, `method`);
- `pstar_m`, the proper target.

The same proper target is used for the query render, both initialisations, the perturbation and
all errors. The source bundles are unchanged.

`pose_errors` uses the stable `atan2(‖vee(R_rel − R_relᵀ)/2‖, (tr R_rel − 1)/2)`, which gives
(0, 0) for P\* against itself. **The thresholds are unchanged:** 1e-5 m, 1e-3°, check-point P95
≤ 1e-3 px, at least 24 inliers. The fixed perturbation is unchanged: Rz(2) Ry(−2) Rx(1),
(0.005, −0.004, 0.010) m.

## Learned-head identity (finding 2)

The strict installer's provenance follows the actual `HeadContext` schema:
`head.requested_sha256`, `head.installed_sha256`, `head_installation`,
`artifact_manifest_sha256`, `installed_head_state_sha256` and `installer_sha256`.
`validate_head_provenance` requires:
- both head SHAs to be 27e06;
- `head_installation = {requested: true, installed: true}`;
- the manifest to be the pinned record `05fe3012…`;
- a 64-hex tensor digest.

The provenance has **no base field**, and nothing is defaulted. `verify_base_checkpoint()`
independently checks the actual `.cache/model-quality/checkpoints/gotrack_checkpoint.pt`
(1 608 850 339 bytes, `f7d127ab…`) in `prepare`, in the rebuild, and again immediately before
model load (`evidence.base_before_load`).

The load, the identity validation and all learned work share one cleanup scope. Network
references are always dropped and the CUDA cache is emptied, including when validation fails.

## Closure (finding 3)

**Pinned (`production_pins`, enforced):**

| File | SHA-256 |
| --- | --- |
| `bench/workbench_neural.py` | `86d0caf6…` |
| `bench/quality_gotrack.py` | `0b185168…` |
| `bench/workbench_render.py` | `da9d7915…` |
| reviewed map | `c5ae3b00…` |
| strict installer `bench/quality_gotrack_inference.py` | `d1cd703f…` |
| context `bench/quality_adaptation_export_parity.py` | `910370f2…` |
| primary record `bench/runtime/gotrack-flow-terminal-20261021.json` | `05fe3012…` |
| upstream `utils/crop_generation.py` | `3d38667f…` |
| upstream `utils/poser_util.py` | `0a864891…` |
| upstream `utils/im_util.py` | `97610c92…` |
| upstream `utils/transform3d.py` | `cbb0d47b…` |

**Recorded and rehashed:**
- **`sources`:** the pinned files plus `bench/__init__.py`, `vision.py`, `quality_capture.py`,
  `quality_render_stability.py`, `workbench_telemetry.py`, `workbench_contract.py` and
  `workbench_geometry.py`; upstream `data_util.py`, `misc.py` and `structs.py`; and this tool.
- **`inputs`:**
  - the inference manifest, `head.pt` and the base checkpoint;
  - the small-records file (`09226dda…`);
  - every admission-06 file;
  - each control GLB and bundle.
- **`admission_source_closure`:** the canonical SHA of `workbench_neural.source_closure(ROOT)`,
  the broader closure that admission verifies. `run` also calls `load_admission(attempt-06)`
  and requires contract `83c041f9…`.

**After the run:** the same sources, inputs, runtime and admission closure must be unchanged.
Otherwise the status is `incomplete_closure_changed`.

## Manufactured CPU preflight (finding 4)

The preflight uses an analytic 40 mm sphere z-depth on a 280×280 crop, the correct oracle
correspondences, a preflight replica of the consumer and solver, and the same unchanged gate:

| Variant | Corruption | Translation error (m) | Check-point P95 (px) | Gate |
| --- | --- | --- | --- | --- |
| `correct` | none | 6.9e-18 | 5.7e-14 | pass |
| `reversed_flow` | correspondences | 0.0229 | 27.6 | fails |
| `half_pixel_displacement` | correspondences | 6.8e-4 | 1.22 | fails |
| `wrong_crop_inverse` | **final conversion** (`C @ refined`) | 0.105 | 192 | fails |
| `omitted_mm_to_m` | **final conversion** (no ÷1000) | 599.8 | 71.3 | fails |
| `bad_flow_oracle_uses_inverse_crop` | bad oracle flow (labelled) | 0.0060 | 8.38 | fails |
| `bad_flow_oracle_target_in_metres` | bad oracle flow (labelled) | 0.0119 | 13.8 | fails |

`crop_to_native(refined_crop_mm, C, mode)` is the pure conversion. `production` is
`inv(C) @ refined` with translation ÷ 1000. The **diagnostic-only** modes `crop_not_inverted`
and `omit_mm_to_m` are used for the negative controls and never change production.

## Support diagnostics (finding 5)

`support_diagnostics(points)` works on Nx3 object points or Nx2 image points. It centres the
points, takes the singular values and normalises them by s1, then records `ratio_2_1`,
`ratio_3_1` (3D), `planar`, `collinear`, `sufficient`, `finite` and `accepted`.

- **Rejected:** non-finite points, fewer than 24 points, or an effectively collinear set
  (s2/s1 < 1e-6, prospective).
- **Planar (rank 2) sets are allowed**, and planarity is only recorded.

**Each iteration:** `correspondence_diagnostics` independently reconstructs the object points
from the captured template depth, crop K, C and the current pose, and the crop targets from
pixel + trace flow. It covers the **retained**, **production-sampled** (the production
`linspace` over at most 10 000 points) and **inlier** sets, in both 3D and 2D.

**Effect on the oracle:** an oracle iteration is accepted only if every set is accepted
(`oracle_failure = support_rejected` otherwise). For the learned branch the diagnostics are
recorded only.

**Oracle confidence:** confidence is `template_mask & oracle_flow.support`, and flow is zero
elsewhere. Unsupported zero-flow pixels therefore get zero confidence, so the unchanged 0.3 gate
excludes them. Each iteration records `confident_pixels` and
`excluded_template_mask_without_support`. The production gates are unchanged.

## Run ordering and failures (finding 6)

1. **Before the GPU stage:** the plan pin, the full plan reconstruction and equality, the budget
   check, and admission 06.
2. **Inside `gpu_stage("dynamic_pose_control")`, ORACLE for all objects first:**
   - one query per object at the proper target;
   - exact and perturbed initialisations, 5 production iterations each.
3. **Stop conditions before any learned load:** any oracle failure (`oracle_failed`) or any
   renderer or refiner cleanup error (`incomplete_cleanup_failed`).
4. **LEARNED:**
   - verify the base checkpoint again, then load through the unchanged strict installer and
     validate the nested provenance;
   - the same in-memory queries and initialisations, with descriptive metrics only;
   - always release the network and CUDA resources.
5. **Failure receipt:** `failure.json` keeps the stage, the error, the counters, any cleanup
   errors, `results_so_far` (all completed small branch rows), `evidence` (base verification and
   head provenance), `closure_before`, the production pins and the plan pins. It never contains
   image or array dumps.
6. **Bounds:** at most 63 renders (3 query + 60 template) and 30 primary forwards, enforced
   before every render and forward.

## Oracle contract (from source)

- Production consumes `crop_pose = crop_from_orig @ pose_mm`.
- Object points are `(d · inv(Kc)[x, y, 1] − t) @ R`. This is z-depth in mm, at
  `(x = col, y = row)` integer centres.
- `target = pixel + flow`, so the flow is template → query crop.
- `native = inv(C) @ refined`, with translation ÷ 1000.

The oracle builds `A = C P0_mm` and `B = C P*_mm`, then sets `flow = proj(Kc, B · R_Aᵀ(Xc − t_A)) − [u, v]`.
It reads its inputs from a scoped observer of `utils.data_util.compute_gotrack_inputs_from_init_poses`,
which is restored and verified on exit. The captured values are template depth, crop K (via
`misc.get_intrinsic_matrix`), `crop_from_orig[0]`, and the initial pose, which must agree within
1e-2 mm.

## Helper API (pure; for parent regressions)

| Group | Functions |
| --- | --- |
| Poses | `rot_x/rot_y/rot_z`, `synthetic_target_pose`, `perturbed_pose`, `metres_to_mm`, `proper_pose`, `pose_errors` |
| Geometry | `lift_zdepth_mm`, `independent_object_points`, `project_points`, `oracle_flow`, `crop_to_native` |
| Checks | `support_diagnostics`, `checkpoint_errors`, `oracle_iteration_check`, `area_samples` |
| Identity | `validate_head_provenance`, `verify_base_checkpoint` |
| Preflight | `production_like_solve(..., conversion=, support=)`, `manufactured_preflight` |
| Plan | `build_plan` |

The run components are `Counters`, `CropObserver`, `OracleNetwork`, `CountingNetwork`,
`render_query`, `correspondence_diagnostics` and `run_branch`.

## Synthetic checks (worker; no GPU, model, scored data or actual prepare)

All of these passed:
- **Synthetic targets:** all three actual seeds produce proper targets (orthogonality about
  1e-15), with `pose_errors(P*, P*) = (0, 0)`. A seed with orthogonality 2e-3 is rejected.
- **Preflight:** the table above, with the correct oracle support accepted (non-planar sphere).
- **Support diagnostics:**
  - 2D diagonal collinear (nonzero x/y extent) and 3D collinear sets are rejected;
  - a 3D planar set is accepted and flagged `planar`;
  - too few points and non-finite points are rejected.
- **Head provenance:** the actual `HeadContext.identity_dict(installed=True)` schema is accepted
  (no base field present). These are rejected: a top-level-only `installed_head_sha256`, an
  installed SHA of `None`, an installation that did not complete, and another record.
- **Oracle confidence:** with a template mask larger than the support, 5380 pixels are confident
  and 73 020 excluded, and confidence is exactly `support & mask`.
- **Plan build:**
  - `build_plan` is deterministic, giving equal plans across two builds;
  - it covers 22 sources and 41 inputs, with an admission closure SHA recorded;
  - a fresh process does not import torch;
  - `run` with a self-consistent but tampered plan (one perturbed pose changed by 1e-9) is
    **refused before admission or GPU work**, and no output is created.

## Interpretation (no cause asserted from this experiment alone)

- **Oracle fails:** this identifies a coordinate, solver or instrumentation issue in the tested
  path.
- **Oracle succeeds but the learned branch drifts:** this localises the *synthetic* failure to
  learned correspondence, confidence and selection.
- **Both succeed:** real appearance, segmentation and initialisation remain unresolved.
- Production is not altered.
