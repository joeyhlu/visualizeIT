# Surface mapping model and virtual camera bench

This is a working, deterministic simulation. It uses the mobile app's C# meshes and pattern-mapping exports, a calibrated virtual camera, and a CPU reference renderer. It does not require Unity Editor activation or a phone. It is **not a trained vision model or an object pose estimator**.

## Run

Use Python 3.12 with the pinned NumPy, Pillow and Matplotlib versions in `requirements.txt`. The geometry export uses the standalone Mono/Roslyn compiler bundled in an existing Unity editor; this does not launch Unity or require its license.

```powershell
powershell -ExecutionPolicy Bypass -File tools/run-bench.ps1 -Python <python-executable>
```

In this workspace the Python executable is `C:\Users\Achita\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`; the existing isolated Matplotlib installation is under `.cache/plotting`. Ordinary development environments can install `bench/requirements.txt` in a virtual environment instead. No new Unity installation is needed.

Optional arguments: `-Frames 24` (12–120), `-Shape cylinder` (`plane`, `box`, `cylinder`), and `-UnityEditorRoot <existing-editor-path>`. Run from any directory; the wrapper sets the project working directory. Outputs are regenerated under ignored `artifacts/bench/`; save copies elsewhere if retaining multiple experiments.

Individual steps:

```text
powershell -ExecutionPolicy Bypass -File tools/export-geometry.ps1
python -m unittest bench.test_renderer
python -m bench.run --frames 24 --shape cylinder
python -m bench.plot_report
python -m bench.build_report
```

Open `artifacts/bench/index.html` directly in a browser. It embeds its numerical report and uses only local assets; there are no remote scripts, APIs or camera permission requests. For an app browser panel, serve the output directory on localhost:

```text
python -m http.server 8874 --bind 127.0.0.1 --directory artifacts/bench
```

## Model

`model.py` loads the real C# fixture vertices, normals, triangles and texture coordinates. It checks its separate Python mapping formula against the C# exports for fitted/unfitted repeats, rotated patterns and rectangular images. The camera stores explicit pixel intrinsics, dimensions, near plane and pose. World coordinates are Unity's X-right/Y-up/Z-forward in metres; camera coordinates are X-right/Y-down/Z-forward. Conversion between those handedness conventions is explicit in `cameraFromUnityWorld`.

`renderer.py` clips triangles at the near plane, projects them with the calibrated camera, rasterizes at pixel centres, and uses reciprocal-depth interpolation for UVs and normals. A depth buffer handles self-occlusion and a synthetic foreground card. The mesh retains its UVs when transformed. Shading is basic diffuse illumination for inspecting shape; Unity's PBR shader, real-world lighting and device occlusion are not tested here.

An independent ray/triangle intersection function checks interpolated UVs and depths. Ten renderer tests cover calibration axes, projection/unprojection, perspective correctness, depth ordering, clipping, shared edges, occlusion and invalid inputs. The deliberately incorrect affine renderer provides a negative control, visibly bending the checkerboard along triangle boundaries.

## Experiments and artifacts

The default 24-view camera orbit uses a 10 cm diameter/20 cm tall cylinder, 1280 × 720 frames, `fx=fy=850 px`, a horizontal orbit radius of 36 cm, and camera height 24 cm looking toward object height 10 cm. Presentation timestamps are simulated at 0.13 s intervals; these are not sensor timestamps or a frame-rate benchmark. Rendered controls use zero surface lift.

For each view, visible triangle centroids serve as known reference points. Projected points are checked against the rendered depth/object mask with a 2 mm sampling tolerance for pixel-centre depth variation. There is no feature detection, correspondence uncertainty or relocalization. The simulation injects position offsets (3/5 mm), yaw (2°), scale error (+2%), focal-length error (+2%), surface lifts (2/0.25 mm), three lost frames, and an object that moves while its world anchor stays put.

The surface-lift experiments shift vertices along normals, as the app shader does, while keeping physical reference landmarks unchanged. They test how a rendering bias affects alignment, even with perfect pose. They do not establish which lift avoids real device depth artefacts; that remains a phone test.

Outputs include a six-panel model preview, camera orbit GIF plus full-resolution frames, perspective comparison, synthetic occlusion, injected-error visualization, calibrated camera manifests, per-condition CSV annotations, `report.json`, and the interactive HTML report/chart. The scorer is the same one intended for future manually annotated footage. Synthetic threshold results are labelled `simulatedThresholdsMet`; `physicalValidationPassed` always remains false.

Results depend on shape, viewing distance, camera calibration and chosen reference points. The known-pose control has zero error by construction. The bench measures sensitivity and detects mathematical defects; it cannot predict a real tracker's performance, generate scanned geometry, or establish natural AR appearance. Perfect synthetic depth does not validate ARCore/ARKit occlusion.

## Real scanned objects

The next experiment uses actual public scans of a mug, mustard bottle, bowl, banana and power drill. `real_objects.py` applies metric UV charts, measures local distortion and chart seams, and renders photographic originals, checker patterns, quality masks and eight-view camera orbits. See [source attribution, reproducible commands and measurement limits](YCB_SOURCES.md). These are existing real object scans inspected with synthetic cameras; no mobile tracking or reconstruction service is implied.
