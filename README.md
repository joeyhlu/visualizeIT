# VisualizeIt

A new Unity mobile implementation for previewing whole-surface designs on physical objects. The first milestone is a **stationary fixture AR app**: manually align a plane, box, or cylinder to a measured physical object. Accurate automatic scanning and object tracking follow only after the live AR quality gate passes.

The mobile source is implemented, but it has **not been compiled with Unity 6 or validated on a phone**. This is not yet an installable release. See [current evidence and blockers](docs/status.md) and [device validation](docs/validation.md).

The [working surface mapping model](bench/README.md) runs without activating Unity: exported app meshes, checkerboard rendering, calibrated camera orbits, synthetic depth occlusion, and measured alignment-error experiments. It generates an interactive report under `artifacts/bench/index.html`. These are simulation results; the live AR gate is still pending.

The [real object scan extension](bench/YCB_SOURCES.md) now tests a mug, mustard bottle, bowl, banana and power drill from the attributed YCB dataset. It produces original/checker/quality views, local distortion and seam measurements, and eight-view synthetic orbits under `artifacts/real-objects/`. This tests mapping on existing scanned shapes; it does not establish phone tracking or our own reconstruction pipeline.

The [offline mapping/tracking pipeline](bench/OFFLINE_PIPELINE.md) adds safeguarded chart improvements, repeating/single-artwork comparisons, bounded public RGB-D footage acquisition, initialized ORB/flow/PnP tracking, depth compositing and independent pose scoring. Reports are under `artifacts/mapping-improvements/` and `artifacts/video-tracking/`. Supplied-pose controls and estimated tracking are labelled separately. This is an offline Python prototype, not the mobile native tracker or an AR release.

## Open and build

The current offline quality experiment is described in [bench/MODEL_QUALITY.md](bench/MODEL_QUALITY.md). SAM 2.1 and GoTrack now run in a separate Windows environment while Docker remains stopped. The original 240-frame keyboard window has been processed, improving accepted-pose availability from 64 to 203 frames, but recovery still contains a large rotation jump and the prefix-invariance check failed. Independent accuracy gates remain unverified. Mug and bottle evaluation is underway. The 120 annotation candidates are prepared, not completed; existing benchmark results remain intact.

1. Install **Unity 6.3 LTS, 6000.3.25f1**, through Unity Hub. Activate an appropriate Unity license. Install Android Build Support including SDK, NDK, and OpenJDK; install iOS Build Support on the Mac used for iOS builds.
2. Add `mobile/` as the Unity project. Let its pinned packages resolve. Do not open it in Unity 2022 or downgrade its package versions.
3. The editor bootstrap creates the URP assets, AR background renderer feature, XR loaders, and app scene. If needed, run **VisualizeIt → Configure project**. Open **VisualizeIt → Open app scene**.
4. Play Mode provides an explicitly labelled material/mesh preview. It does not provide camera AR or satisfy a device gate.
5. Run **VisualizeIt → Build Android APK**, then install `mobile/Builds/Android/VisualizeIt.apk` on a supported phone. For iOS, **VisualizeIt → Export iOS project**, open the export in Xcode on macOS, select a signing team, and build to the reference iPhone. Signing and TestFlight distribution have not been configured.

Build methods are also available for CI as `VisualizeIt.Editor.ProjectBootstrap.BuildAndroid` and `VisualizeIt.Editor.ProjectBootstrap.BuildIOS`. Build Support modules and licensing must be present. The source bootstrap and native photo pickers still require their first platform compilation.

Pinned packages include AR Foundation/ARKit/ARCore **6.6.2**, URP **17.3.0**, Input System **1.17.0**, XR Core Utils **2.6.0**, and XR Management **4.5.4**. The complete list is in `mobile/Packages/manifest.json`. Unity will produce `packages-lock.json` on successful package resolution; that lock should be retained with the generated scene/settings assets after the first editor import.

## Use the fixture app

Import a PNG/JPEG design using the native image picker, or use the built-in checkerboard. Find a supporting plane, aim the crosshair at the intended bottom centre of the object, and tap Place. Enter measured dimensions in centimetres and adjust local offsets and yaw to align the mesh. The plane uses width/depth; the cylinder uses width as its diameter and height.

Pattern controls set tile width in metres, rotation, roughness, and a small normal offset to avoid coincident surfaces. UV coordinates are physical distances attached to mesh vertices. Box side faces unwrap around the perimeter; caps/top/bottom use separate planar coordinates. Those boundaries have seams that must be evaluated explicitly.

For cylinders, move the wrap seam with its angle control or use **Fit cylinder repeats at 0°** to choose a whole number of repeats within the allowed tile-width range. This resets pattern rotation to zero and slightly adjusts physical tile width. It aligns repeat phase across the side seam; it cannot make nonperiodic image borders or cap/side boundaries seamless. The standalone mapping analyzer measures local scale/stretch on arbitrary triangle data and is tested against deliberately malformed mappings.

The renderer uses URP physically based shading, available AR light estimates and environment probes. It requests environment/people depth occlusion, reports current availability, and hides the design when tracking becomes invalid. Providers determine which capabilities exist. Natural appearance and correct occlusion are requirements to test, not established outcomes of this source implementation.

Keep the physical object still. The fixture follows a world anchor and does not follow a picked-up object. Returning from camera suspension requires placement confirmation. Saved projects retain the design and fixture/material settings locally; they do not restore room anchors across sessions. Screenshots and diagnostic reports are saved under Unity's app-specific `persistentDataPath`.

## Local verification

Run `powershell -ExecutionPolicy Bypass -File tools/verify.ps1`. This uses the standalone Mono/Roslyn tools bundled in an installed Unity editor, plus `g++`. Override `-UnityEditorRoot` and `-Cxx` when necessary. Using the Unity 2022 **compiler** for engine-independent tests does not build the Unity 6 app.

Run `python -m unittest discover -s tools -p test_evaluate_tracking.py` for the annotation scorer's synthetic unit tests. Score real annotated footage with:

```text
python tools/evaluate_tracking.py annotations.csv --output artifacts/device/attachment.json
```

Run `powershell -ExecutionPolicy Bypass -File tools/export-geometry.ps1` to compile `tools/ExportGeometry.cs` and export the same C# mesh generator used by the app. Add `-Plot -Python <python-executable>` to plot those exports with NumPy and Matplotlib. Local output lives in ignored `artifacts/`; the script also supports this workspace's isolated plotting installation in `.cache/plotting`. The report is geometry evidence, not an AR screenshot.

## Layout and milestone boundaries

| Directory | Contents |
|---|---|
| `mobile/` | Fixture AR app, metric meshes, PBR shader, native image import, local settings, diagnostics, Unity bootstrap |
| `native/` | Tested C++ coordinate conversion boundary; no moving-object tracker yet |
| `server/` | Reconstruction milestone specification; no server or model worker is running |
| `reference/` | Original hackathon Python source retained as reference, with its commit recorded |
| `tools/` | Geometry/syntax/native verification, geometry export/report, attachment scorer |
| `bench/` | Working synthetic surface model, CPU renderer, camera/error experiments and interactive report |
| `docs/` | Validation procedure, observed status, future integration contracts |

The next milestones are scanned stationary objects, C++ rigid-object pose tracking, then separate skin and fabric modes. Imported designs come first; AI design generation, public accounts, and sharing are deferred. Transparent, shiny, plain, or deforming objects need their own measured gates.
