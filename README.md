# VisualizeIt

VisualizeIt explores how designs can be mapped onto 3D objects. The comparison viewer plays saved frames, masks, and poses from prerecorded footage; it does not track a live camera. The separate initialized prototype below provides PC frame processing and a browser camera client.

Tracking and mapping results are experimental. Independent accuracy, broader object coverage, live performance, and phone validation remain unverified. The Unity mobile project is a deferred source scaffold, and the reconstruction server has not been implemented.

An initialized geometric tracking prototype includes a local frame-processing service, a browser camera client, and frozen dataset-reference evaluations. See [prototype launch and evidence](docs/prototype.md). The ten-object run completed with zero quality passes. A later tracker repair passes synthetic controls; independent human accuracy, sustained live 30 FPS and physical-phone validation remain unmet.

## CPU tracking regression benchmark

The first full comparison uses **shared simplified initialization**, not the
full production pipeline. Original mesh-raycast initialization is being checked
separately; do not infer production readiness or failure solely from that first
comparison.

A portable CPU-only harness compares the exact historical rigid tracker with
the current rigid tracker and production stream wrapper on variable-length
sequences. The versioned dataset has 3,600 real HB reference-labeled scored
object-frames and 4,560 explicitly synthetic stress frames, with isolated
evaluation labels and repeatable setup. See [dataset, licenses and reproduction](datasets/tracking-v1/README.md).
These geometric measurements do not stand in for trained GoTrack or live-phone
performance. Full-media archive publication and measured results are recorded
separately rather than implied by the presence of a download recipe.

## Run the offline tools

The synthetic camera bench renders the fixture meshes and measures mapping behavior with known camera poses. It does not estimate poses or represent phone performance. Use Python 3.12 with the pinned packages in `bench/requirements.txt`. The wrapper also exports the C# meshes using the standalone compiler bundled with an installed Unity Editor; this geometry export does not build the phone app. Set `-UnityEditorRoot` if the editor is outside the wrapper's default path:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run-bench.ps1 -Python '<python-executable>' -UnityEditorRoot '<installed-editor-path>'
```

The offline mapping and recorded-footage pipeline has separate stages. Use Python 3.12 with the pinned packages in `bench/requirements-real.txt` and pass its executable explicitly; the wrapper's default points to a Codex-specific runtime. Its verification entry point is:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run-offline.ps1 -Stage Verify -Python '<python-executable>'
```

See [the bench guide](bench/README.md) and [offline pipeline guide](bench/OFFLINE_PIPELINE.md) for prerequisites, experiment stages, and report locations. Generated reports are written under `artifacts/`. To serve existing reports locally:

```powershell
python -m http.server 8876 --bind 127.0.0.1 --directory artifacts
```

Open the report's path under `http://127.0.0.1:8876/`. A source-level geometry and native-code check is available with `powershell -ExecutionPolicy Bypass -File tools/verify.ps1`; it does not compile the Unity project.

## Mobile and server status

`native-ios/` contains the new native SwiftUI iPhone application and Xcode project, with AVFoundation camera capture, glass design screens and an authenticated client for the pinned GoTrack pose bridge. It does not embed a website. Open [the iPhone setup guide](native-ios/README.md) for installation from a Mac. This is source delivery: no signed app, iOS build, exact visual comparison or physical-phone tracking validation has been completed. See [native mobile status](docs/native-mobile.md) and [native pose contract](docs/native-pose-contract.md) for the required runtime, geometry and admission configuration.

`mobile/` contains a Unity fixture-app scaffold. Unity 6 import, platform compilation, camera AR, and physical-device behavior have not been verified; no installable phone build is available. Unity setup remains deferred. When resumed, the editor bootstrap exposes **VisualizeIt → Build Android APK** and **VisualizeIt → Export iOS project**. These entry points still require the supported Unity editor and platform modules, and their builds need validation. See [device validation](docs/validation.md) and [observed status](docs/status.md).

`server/` contains the reconstruction milestone specification only. There is no implemented or deployed server or model worker. See [server scope](server/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `bench/` | Synthetic surface and camera experiments, scanned-object mapping, and offline recorded-footage tracking research |
| `mobile/` | Deferred Unity source scaffold for fixture placement and design previews |
| `native-ios/` | Native SwiftUI iPhone UI, camera and trained-pose service client; build/device validation pending |
| `native/` | C++ coordinate-conversion boundary; no moving-object tracker |
| `server/` | Reconstruction service specification; no implementation |
| `reference/` | Original hackathon source retained as a pinned reference; see its [README](reference/README.md) |
| `tools/` | Offline run, geometry export, source verification, and annotation-scoring utilities |
| `docs/` | Project status and validation procedures |

The original source and dependency pins are retained under `reference/qhacks2025/`; the reference is not a dependency of the current project. Current experiment limitations and evidence are recorded in [model-quality status](bench/MODEL_QUALITY.md) and [project status](docs/status.md).
