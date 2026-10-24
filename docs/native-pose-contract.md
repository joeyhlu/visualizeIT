# Native trained-pose contract v1

Source delivery only. No inference, runtime execution, model download, live
camera upload or physical-phone validation was performed for this change.
Torch is currently unavailable and the existing serving admission is invalid.
The default service consequently reports live support unavailable. The trained
head is exactly `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`;
an installed head is reported only from actual backend install provenance.
There is no fallback to the baseline or the legacy `/v1/frames` endpoint.

`bench/workbench_live_pose.py` defines a separate unbounded camera lifecycle.
It does not manufacture a finite ClipSpec or relax the recorded FrameBoundary
or NeuralSession contracts. `tools/native_pose_bridge.py` provides the bounded
HTTP adapter. `tools/serve_native_pose.py` is the operator entrypoint. Starting
the service, restoring a runtime, measuring admission and sending phone frames
remain separate, explicitly authorized operator actions.

## Transport and consent

All routes require `Authorization: Bearer <operator token>`. The server creates
a random process token unless `--token-env` names an existing token environment
variable (at least 32 non-whitespace characters). A generated token is printed
once to the operator console and is never included in responses or request logs.
The token is a shared operator credential, not a multi-user identity system.

Non-loopback serving requires `--cert certificate.pem --key private-key.pem`.
The physical phone must trust that HTTPS certificate. Do not bypass TLS trust
in the app. Loopback HTTP is available for development only. Do not publish
this listener through the public artwork viewer. No CORS origins are granted;
requests containing Origin are rejected. No client paths, module names, masks,
poses or readiness flags are accepted. HTTP route matching is exact.

The native app must obtain explicit user consent before source images leave the
phone. Capabilities/session metadata do not imply that consent. Earlier unused
camera permission does not authorize a new upload. No actual frames were sent
while implementing this source. Artwork generation remains a separate action.

Example operator command after legitimate configuration and authorization:

```powershell
python tools/serve_native_pose.py --host 0.0.0.0 --port 8894 --cert <certificate.pem> --key <private-key.pem> --config <operator.json> --token-env NATIVE_POSE_TOKEN
```

Requests need exact bounded Content-Length and application/json. Chunked bodies,
Expect requests and browser Origin are rejected. Maximum body is 6 MiB; JPEG is
4 MiB; maximum raster is 1920 by 1920 with at most 2,073,600 pixels. Dimensions
are checked from the JPEG header before RGB decode. Only JPEG without EXIF
rotation is accepted. Eight HTTP workers and one processing operation are
allowed. One live session is admitted per server process. Operators must run
only one GPU service/stage; this is not a machine-wide GPU scheduler.
TLS negotiation runs only inside one of the eight bounded connection workers,
after the accepted socket receives a ten-second timeout. It never runs in the
accept loop. Handshake failure closes its socket and releases the worker slot;
server shutdown closes accepted sockets as well as the listener. Startup errors
also close the bridge's worker/session resources.

## Routes and schemas

`GET /v1/native/capabilities` returns:

```json
{
  "schema_version": 1,
  "model_head": "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311",
  "installed_head_sha256": null,
  "admission_ready": false,
  "live_camera_supported": false,
  "recorded_finite_supported": true,
  "blockers": ["no reviewed live GoTrack operator configuration"],
  "limits": {
    "max_width": 1920, "max_height": 1920, "max_pixels": 2073600,
    "max_jpeg_bytes": 4194304, "max_body_bytes": 6291456,
    "max_sessions": 1, "max_inflight": 1
  }
}
```

`recorded_finite_supported` identifies the separate repository workbench source
contract; it is not admission to run recorded inference or a route on this
listener. Capability checks never construct a model or decode an upload.

`POST /v1/native/sessions` requires exactly:

```json
{
  "schema_version": 1,
  "expected_model_head": "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311",
  "camera": {
    "calibration_revision": "actual-device-calibration-id",
    "width": 640, "height": 480,
    "fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5,
    "units": "pixels", "orientation": "landscape_right", "mirrored": false,
    "crop": {"x": 0, "y": 0, "width": 640, "height": 480}
  },
  "asset_id": "operator-registered-object",
  "onboarding": {
    "selection_tap": {"x": 320.0, "y": 240.0},
    "artwork_revision": "generated-artwork-id"
  }
}
```

The numeric intrinsics above illustrate JSON shape only. They are not a camera
calibration and must never be sent as measured values. AVFoundation must deliver
actual per-frame intrinsics for the exact encoded sensor raster. Keep raw VGA
pixels in their captured landscape orientation; convert portrait preview taps
back through the real aspect-fill crop and camera orientation. Any image
rotation, mirroring, rescaling, lens correction, format change or camera change
requires corresponding measured calibration and a new session. Intrinsics use
integer pixel centres, pixels as units and an undistorted pinhole raster. Crop
origin documents the sensor crop; width/height equal the encoded unscaled raster.
The supported orientation strings are portrait, portrait_upside_down,
landscape_left and landscape_right. Mirrored input is currently rejected.

The exact complete calibration fingerprint must already occur in independently
measured live admission. An unrecognized device/calibration remains unavailable.
`asset_id` selects only a server registry entry, never a path. The tap is an
observational prompt, not an object pose or foreground bounding box. Client
artwork revisions are retained as display identity for pose-only sessions.
Rendering additionally requires an exact server-registered artwork revision and
texture/geometry files; generated client image IDs do not claim this binding.

Creation returns HTTP 201 with `schema_version`, random `session_id`,
`generation: 1`, `state: "onboarding"`, `calibration_revision`,
`artwork_revision`, `model_head` and `installed_head_sha256` (initially null).
The next session has a different unpredictable ID. A second active session is
HTTP 409. Abandoned sessions are closed by a one-second idle reaper once their
operator-configured TTL elapses, even without another HTTP request.

`POST /v1/native/sessions/{session_id}/frames` requires exactly:

```json
{
  "schema_version": 1,
  "generation": 1,
  "sequence": 0,
  "timestamp_ns": 123456789,
  "jpeg_base64": "<encoded JPEG>",
  "jpeg_sha256": "<lowercase SHA256 of encoded JPEG bytes>",
  "calibration_revision": "actual-device-calibration-id",
  "artwork_revision": "generated-artwork-id"
}
```

Sequence and capture timestamp must both strictly increase. Timestamp is the
camera's monotonic capture clock in nanoseconds, not wall time or request time.
The server computes its own SHA256 of decoded uint8 RGB including dtype/shape
using `workbench_geometry.array_sha256`; the client hashes encoded JPEG bytes,
so cross-platform JPEG decoder rounding cannot cause false hash mismatch.
An accepted frame consumes its sequence before provider work. Never retry that
image under a new sequence after backend failure.

Each successful HTTP 200 response contains `schema_version`, `session_id`,
`generation`, `frame_id` (`session:generation:sequence`), `sequence`,
`timestamp_ns`, `calibration_revision`, `artwork_revision`, `model_head`,
actual `installed_head_sha256`, server `rgb_sha256`, `mask_sha256`, `state`,
nullable `pose`, nullable `render` and nullable `reason`. States are recovering,
tracking, lost or unavailable. A current tracking pose is:

```json
{
  "object_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]],
  "units": "metres",
  "camera_axes": "opencv_x_right_y_down_z_forward"
}
```

The matrix acts on object-frame column vectors. It is object-to-camera, not a
world anchor, camera pose, ARKit transform or phone screen-space transform.
The matrix shown illustrates units only, not a measured estimate. Tracking
requires the backend's two consecutive consistent solutions after onboarding.
Every other state has `pose: null`; no last-known pose is republished as current.
Separate `mask_state`, `pose_state` and `render_state` fields accompany the
compatible `state` alias. An admitted observation, including an observed empty
mask, is `mask_state: observed`; pose_state equals state. Actual pixels produce
render_state rendered; a missing renderer on a current pose is unavailable;
without a valid pose rendering is suppressed. Error responses also contain
these fields, null pose/render and empty outlines, with mask_state rejected
for mask/image integrity failures and unavailable for earlier failures.

An optional actual render contains `frame_id`, `rgb_sha256`, `artwork_revision`
and `rgba_png_base64`. The server verifies matching current frame/RGB/metric GLB,
uint8 full-size RGBA and nonzero alpha pixels before including it. No renderer,
missing texture binding or render failure means `render: null`. Backend render
statistics alone never count as rendered artwork. The native client must clear
an old overlay when frame/session/generation/revisions differ, when state is not
tracking, when the actual installed SHA differs, or when render is null. Never
paste an older response over a newer camera image. Physical phone latency and
display alignment remain unverified.

`DELETE /v1/native/sessions/{session_id}` has no body and closes provider/model
state. A closed/unknown ID is 404 or 410. Errors are JSON with `schema_version`,
`error`, `reason`; invalid input 400/422, auth 401, browser origin 403,
conflict/stale/inflight 409, oversized 413, unavailable admission/provider/model
503, processing deadline 504. Missing admission is checked before base64 decode,
image decode, provider import and model construction.

Responses also contain `outline_segments`, an array of closed polylines
`[[[xNormalized,yNormalized],...],...]` extracted only from this frame's verified
observed mask. Coordinates divide raw raster x by width-1 and y by height-1.
Limits are 32 segments, 512 points per segment and 2048 points overall. Empty
foreground or unavailable contour extraction returns `[]`. Current observed
outlines may be displayed during recovering/lost even when pose is unavailable;
their RGB/frame identity checks and exact processed-image display still apply.
No bounding-box or projected-silhouette outline is substituted.

## Actual live lifecycle

1. Recompute independent live admission, operator configuration and all pins.
2. Validate metadata/JPEG hash/calibration, decode one current RGB raster.
3. Trusted provider observes actual RGB and returns `ObservedMask(frame_id,
   rgb_sha256, boolean_HxW, "observed_foreground")`. A silhouette, bounding box,
   empty/reused mask, mismatched RGB/frame or bad shape cannot estimate a pose.
4. On the first nonempty observation, a trusted initializer returns
   `LiveInitialization` bound to that exact RGB/frame and registered metric GLB,
   with a valid metre object-to-camera pose and `onboarding:` provenance.
5. Construct the actual `GoTrackMetricBackend("primary_neural", live_bundle,
   head_path=..., device=...)`. Verify actual installed head/base provenance.
   The single-use seed applies only to the immediate sequence successor.
6. Subsequent frames call `GoTrackMetricBackend.estimate(rgb, current_mask, K,
   state)`. After lost masks/solver failure the prior is cleared. No automatic
   reacquisition or old seed reuse is implemented.
7. Recheck install/bundle/source/config/admission identities after processing,
   validate any current pose/render and emit only that frame's result.

One request holds the processing lock through consumption of its result. A
deadline cannot safely preempt a running Python/CUDA call. Timed-out work stays
quarantined under the same exclusive lock until it actually exits, then its
session is closed. A second operation cannot load another model alongside it.
Shutdown also waits for the worker before freeing its model. A permanently hung
driver therefore requires operator process termination; no unsafe resource-free
or additional overlapping worker is attempted.

## Concrete operator composition

The shipped factory entrypoint is
`bench.workbench_live_pose:sam2_metric_reference_provider`. It implements:

- SAM2ImagePredictor with the existing pinned SAM2.1 base-plus checkpoint,
  loaded explicitly from local files on CPU, with no download API. It recomputes
  embeddings/masks for each current RGB. Observed feature flow carries foreground
  point prompts from the previous actual image; it does not emit poses or masks.
  Missing forward/backward-consistent prompt tracks emit an empty observed mask.
- One-time SIFT feature matching against server-registered descriptor-to-metric
  XYZ references, followed by RANSAC PnP with at least 24 inliers and positive
  depth. This produces onboarding initialization only. Trained GoTrack owns
  every subsequent published pose; onboarding is never a fallback after loss.
- No artwork renderer. Pose-only responses are supported; custom texture
  registration/rendering is a separate provider capability.

These are new numerical behaviors whose empirical segmentation/initialization
accuracy, thresholds, CPU/GPU memory, timing and real camera performance remain
unvalidated. Source checks do not establish those claims. CPU SAM2 avoids
simultaneous SAM2/GoTrack GPU residency, but CPU memory and speed still require
measurement. The camera must supply an actual visible registered metric object;
an arbitrary surface/image does not manufacture metric references.

Reference features use a non-pickle NPZ with exactly `descriptors` float32 Nx128
(SIFT), `xyz_m` float32 Nx3 in the registered GLB's object frame, and a scalar
string `glb_sha256`. N is 24..50,000; all values must be finite. Reference points
must have genuine measured/dataset provenance. Do not derive them from evaluator
labels, a current guessed pose or a projected silhouette. No reference producer,
new data acquisition or reference-quality validation was run here.

SAM2 API source: [Meta's image predictor example](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb).
The implementation was also read against the existing local SAM2 source.

The operator JSON has exactly these keys (placeholders are intentionally not an
admitted runnable configuration):

```json
{
  "schema": "native_gotrack_operator_v1",
  "head_path": ".cache/video-workbench/models/flow-terminal-20261021/head.pt",
  "device": "cuda:0",
  "serving_admission_dir": "<reviewed existing measured serving admission directory>",
  "live_admission_path": "<independently reviewed live admission JSON>",
  "live_admission_sha256": "<exact reviewed receipt file SHA256>",
  "provider_factory": "bench.workbench_live_pose:sam2_metric_reference_provider",
  "provider_sources": ["bench/workbench_live_pose.py", "<all reviewed provider dependency source files>"],
  "provider_resources": ["<local SAM2 checkpoint>", "<SAM2 config>", "<registered reference NPZ>"],
  "provider_config": {
    "sam2_checkpoint": ".cache/model-quality/checkpoints/sam2.1_hiera_base_plus.pt",
    "sam2_checkpoint_sha256": "<existing quality_assets.py pin>",
    "sam2_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "reference_features": "<registered reference NPZ>",
    "minimum_sam_score": 0.7
  },
  "assets": {
    "operator-registered-object": {
      "metric_bundle": "<workbench_metric_bundle_v1 JSON>",
      "references": ["<registered reference NPZ>", "<reference provenance>"],
      "artwork_revision": "server-registered-artwork-id",
      "artwork_files": []
    }
  },
  "frame_timeout_s": 30,
  "session_idle_timeout_s": 60
}
```

All resource/source paths resolve within this checkout. The live metric bundle
uses measured_metric_cad or dataset_metric_cad, integer_centers intrinsics, an
authenticated GLB, and null seed_pose_m/seed_label/seed_frame_id. A recorded
seed is rejected. Operator inputs are never taken from HTTP. Additional custom
providers implement observe/initialize/close and optional render with the exact
current-frame bindings above; their entire executable dependency closure and
resources require fresh reviewed pins. Imports occur only after admission.

## Independent admission and remaining verification

There is deliberately no CLI that mints live admission by setting a flag.
An operator-reviewed receipt must have exactly `schema: native_live_admission_v1`,
unique `contract_id`, `bindings`, `serving_contract_sha256`, `calibrations`, and `evidence`.
`bindings` must equal `ConfiguredGoTrackFactory._pins()` recomputed from actual
live service/provider files, resources, provider config, assets, actual base and
head bytes, runtime package versions, selected device and deadlines. The
existing `load_admission` re-verifies measured metric geometry, full-image head
parity, source closure and the registered asset families; its recorded admission
alone never grants live support. Exact complete camera calibrations are listed
separately. Config pins the reviewed live receipt's exact file SHA256.
The shipped SAM2 provider also automatically includes every Python/YAML file
under its local SAM2 source checkout, quality_assets.py, the selected checkpoint
and registered feature NPZ. Loaded SAM2 modules must resolve inside that checkout.

`evidence` contains an operator-local path and SHA256 of a reviewed
`native_live_measurements_v1` receipt with method
`measured_live_camera_geometry_and_observed_mask_v1`, recomputed
matching `contract_id`, `bindings_sha256`, ordered `calibration_sha256`, `checks` and `trials`. Required
checks are geometry_units, pixel_centres, orientation, mirror, crop,
observed_mask_binding, seed_frame_binding, source_drift_rejected,
stale_frame_rejected and loss_suppresses_pose. Trials contain kind
positive/negative, local path and SHA256 of preserved actual trial evidence;
both kinds are required and their current bytes are authenticated. These
measurement/review receipts are trusted operator evidence, not proof generated
by HTTP or a cryptographic attestation of a physical experiment. The reviewer
must assess their actual positive/negative observations, not just booleans.
Hashing assertions does not establish tracking quality.

Before runtime readiness can be claimed: restore a separately authorized pinned
runtime, re-earn current serving admission, supply genuine metric references,
measure live calibration/mask/onboarding positive and negative trials, independently
review and pin that complete evidence, exercise source/backend failure paths,
measure resources, obtain fresh camera-upload consent, and verify on a physical
phone. No inference/training/install/download or these measurements was performed
for this source delivery. Native UI build/signing and physical display alignment
remain pending as separately reported by the native app owner.
