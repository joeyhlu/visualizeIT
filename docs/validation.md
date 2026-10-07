# Initial testing and the live AR exit gate

## 1. Repeatable source checks

Run `tools/verify.ps1` before editor import. The suite exercises the actual mesh generator used by `FixtureSurface`, including outward triangle winding, normals, metre-based bounds/UVs, invalid geometry inputs and visibility behavior when session/anchor/foreground tracking is lost. The C++ test applies coordinate conversions to physical points and rejects non-rigid/non-finite matrices.

Export that generator using `tools/ExportGeometry.cs`, then plot the exported JSON with `tools/geometry_report.py`. Save the PNG and JSON metrics as source evidence. It does not measure reconstruction, camera reprojection, occlusion or material appearance.

Run [the virtual camera bench](../bench/README.md) while phone setup is deferred. It checks perspective interpolation against independent ray intersections, renders actual exported mesh UVs through a synthetic calibrated camera, and injects known alignment errors. Its synthetic depth/card test and pose sensitivities are mathematical controls. They do not establish real device occlusion, measured tracker accuracy or natural material appearance.

## 2. Unity integration

Use the exact pinned editor/packages. Record package-resolution results and retain generated package lock/settings/scene assets. Require successful Android and iOS compilation, no shader errors, correct AR loader assignment, camera permission behavior and image picker operation. Test picker cancel, PNG/JPEG/transparent designs, large images, save/load, screenshot, app suspension, and supported/unsupported capability labels.

In the editor material preview, inspect a checkerboard on every face, cylinder curvature, UV seam, pattern size/rotation, imported image aspect ratio, roughness and fade. This is a rendering inspection only; mark every screenshot as editor preview. Compile both Vulkan/GLES3 Android shader variants and the Metal iOS variant in platform builds.

## 3. Measured physical fixtures

Start with a rigid matte box approximately 10 × 20 × 15 cm, a printed rigid cylinder 10 cm in diameter/20 cm tall, and a horizontal plane with measured markings. Measure the actual objects; enter those measurements rather than assuming nominal dimensions. Use a stable table, good overlapping camera views, and a repeatable initial placement. Mark corners and identifiable points on the physical fixture for later annotation.

The checkerboard texture contains eight squares across its tile: a 4 cm tile gives 5 mm squares. Check that physical scale is consistent and inspect seams separately. Cylinder texture width is its circumference; visible curvature should follow the surface as the camera moves. The shell offset is a rendering control, not a correction for pose error.

For each supported device record:

1. Stationary phone for 30 seconds; small lateral movement; approach/recede; full slow camera orbit. Check edge alignment, scale, perspective, texture stability, and curvature.
2. Move a hand in front and behind the object. Repeat with another rigid occluder. Save depth-available and depth-unavailable cases separately. Unsupported depth must be reported explicitly; those devices cannot claim the same occlusion gate.
3. Cover the camera, leave/re-enter the view, dim the lights, interrupt/resume the app. The overlay must fade during invalid tracking and return only after stable tracking. Suspension currently requires re-placement.
4. Repeat indoors under diffuse and directional light, and outdoors. Review light/color mismatch, floating edges, reflection behavior, flicker, shell gaps, cap/side seams, and over-darkening.
5. Run ten minutes with the fixture visible and save a session report at the end. Capture profiler frame time, native tracking latency where available, memory and platform thermal state separately. The in-app report does not measure thermal state or tracker latency.

Use a LiDAR iPhone, a non-LiDAR iPhone, and an ARCore depth-capable Android. Record exact hardware, OS, editor/package versions, build revision, active graphics API, and actual depth/light capability. No device has been validated yet. Start internal APK distribution and signed iOS device builds; TestFlight follows signing setup.

## 4. Score attachment honestly

Normalize footage/annotations to a 720p frame with preserved aspect ratio. For each sampled frame, annotate visible physical reference points and the corresponding rendered checkerboard/mesh points. Include frames where the surface is sufficiently visible but the app has lost tracking; set `valid=0` and leave observed coordinates empty. Exclude fully invisible physical points rather than guessing their position. Document annotation uncertainty and sampling cadence.

CSV header:

```csv
frame,point,valid,expected_x,expected_y,observed_x,observed_y
```

Use `tools/evaluate_tracking.py`. It computes Euclidean point error, interpolated median/p95, and frame-based tracking availability. All point rows within a frame must share its tracking validity. Invalid frames remain in the availability denominator. It rejects duplicate points and non-finite valid coordinates. Review aggregate and per-sequence scores so one easy view cannot hide a difficult view.

Attachment targets: median <5 px and p95 <10 px during valid tracking, with availability reported. The scorer uses a provisional minimum availability of 90%, configurable using `--minimum-availability`; the original plan did not specify an availability threshold. Agree a final threshold using observed device results. Error is not a replacement for appearance/occlusion review.

Performance target: sustained >=30 FPS over ten minutes, without severe thermal degradation. Report average and slow frames, inspect per-frame profiler data and avoid relying only on a high session average. In-app p95 uses the first 40,000 sampled frames; total frame count, worst frame and slow-frame count cover the full session. Long stalls are retained.

## Evidence and gate decision

Keep builds, footage, annotations, session JSON, profiler traces and a defects log under `artifacts/device/<device>/<sequence>/`. Record reviewer, date and accepted limitations. The natural-AR gate requires build success plus stable placement, convincing curvature/perspective, appropriate lighting and correct supported occlusion on the reference matrix. Any missing result leaves the gate pending.

Automatic scanning starts only after this gate passes. Mug/irregular/plain/shiny geometry, pickup/rotation, reference relocalization, scan failure handling, and deformation benchmarks belong to later milestones; fixture placement cannot establish those capabilities.
