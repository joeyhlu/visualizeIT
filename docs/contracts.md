# Integration contracts for later milestones

These boundaries are specifications for subsequent implementation. No scan client/service or moving-object tracker is wired into the fixture app yet.

## Coordinate and capture boundary

Unity world coordinates use metres, +X right, +Y up, +Z forward. Images and intrinsics use pixel coordinates with +X right/+Y down, calibrated to the captured image resolution. An OpenCV camera uses +X right/+Y down/+Z forward. Define intrinsics `fx, fy, cx, cy`, source width/height, distortion model/coefficients, timestamp clock and image orientation explicitly; display rotation/crop must never silently change calibration.

Camera pose is a row-major 4×4 matrix acting on column vectors, named `worldFromCamera`, recorded in the same AR session/clock as the image. Supply optional depth/confidence with their own calibration, resolution, timestamp and image-to-depth transform. Reject unmatched timestamps rather than pretending depth is synchronized. Capture metadata includes schema version, session ID, platform, units and coordinate basis.

GLB assets use metre-scaled right-handed glTF coordinates. `native/pose.h` converts a rigid `cvCameraFromGltfObject` into `unityCameraFromUnityObject` with explicit camera/object basis reflections. It is tested independently; Unity's glTF import and camera display conventions still require end-to-end device projection tests before integration.

## Scan jobs and artifacts

Version the eventual API under `/v1/scans`: create a job; upload an immutable manifest/bundle with content hashes; explicitly start processing; inspect status/progress; cancel; download artifacts; delete. Use job states `created`, `uploading`, `queued`, `processing`, `succeeded`, `failed`, `cancelled`. Idempotency keys, resumable uploads, bounded size, artifact validation and retained failure reasons are required. Processing must not block rendering.

Each object asset contains `schemaVersion`, asset ID/content hash, GLB mesh/material paths, `arWorldFromObject`, metric alignment quality, surface confidence, coverage/reconstruction statistics, UV scale/seam metadata, and tracking reference views/features linked to 3D points. Low-confidence regions remain explicitly unavailable. Reference geometry may retain more detail than the display mesh. Cache assets locally and check schema/hash before use.

The future worker segments selected objects using a released, verified SAM version; reconstructs using COLMAP with calibrated views; fuses trustworthy depth; aligns geometry to recorded AR metric scale; prepares normals/UVs/mesh LODs; and produces reference correspondences. Pin actual supported releases when that milestone begins. Do not download models or inherit the hackathon's dependency list during the fixture gate.

## Tracker and saved project

The eventual native tracker accepts immutable object assets plus calibrated timestamped frames. Output `schemaVersion`, frame timestamp, camera-from-object rigid transform, confidence, state (`initializing`, `tracking`, `limited`, `lost`) and a failure reason. A room anchor does not replace the separate tracked object pose. Deformation modes additionally supply validated vertex positions with unchanged topology/UV identity and region confidence.

Native matching/PnP/flow/refinement/filtering and relocalization are not implemented in the mobile library. The independent Python prototype now implements initialized ORB/flow/PnP tracking and recovery for recorded data; it is not wired into this native API. Plain symmetric objects must trigger an ambiguity state rather than arbitrary texture rotation. Neither the current coordinate helper nor the fixture visibility state machine estimates object pose.

### Offline prototype boundary

Version 2 NPZ object assets retain geometry, metric UVs, source-face/vertex mappings, chart IDs, trustworthy faces and, for video objects, a `benchFromSource` matrix, reference 3D points and ORB descriptors. BOP source axes are right-handed and positions/poses are converted from millimetres to metres. These assets are not GLB exports and must not be fed directly into the glTF-specific native pose converter.

`RigidTracker.update(rgb, K, frame_id)` returns frame ID, state, failure reason, an optional proper `cvCameraFromSourceObject` rigid transform, and correspondence/inlier/reprojection/coverage statistics. Confidence is a threshold status, not a calibrated probability. It receives no evaluation annotations, masks or depth after onboarding. The compositor separately uses measured depth; the evaluator separately uses annotated poses and visibility masks. Zero-distortion source calibration is assumed; arbitrary phone calibration/display rotations are not yet supported by this prototype.

The existing local fixture save is schema version 1 and retains fixture dimensions, alignment, design and material/pattern settings. Later scanned projects must reference content-addressed object assets and saved design settings. Persistent world anchors need a separate platform-specific contract; existing saves deliberately require fresh placement.
