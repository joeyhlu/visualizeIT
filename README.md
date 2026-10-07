# VisualizeIt

VisualizeIt explores how designs can be mapped onto 3D objects. The active work is offline experimentation and comparison of prerecorded footage. The browser viewer plays saved frames, masks, and poses; it does not track a live camera.

Tracking and mapping results are experimental. Independent accuracy, broader object coverage, live performance, and phone validation remain unverified. The Unity mobile project is a deferred source scaffold, and the reconstruction server has not been implemented.

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

`mobile/` contains a Unity fixture-app scaffold. Unity 6 import, platform compilation, camera AR, and physical-device behavior have not been verified; no installable phone build is available. Unity setup remains deferred. When resumed, the editor bootstrap exposes **VisualizeIt → Build Android APK** and **VisualizeIt → Export iOS project**. These entry points still require the supported Unity editor and platform modules, and their builds need validation. See [device validation](docs/validation.md) and [observed status](docs/status.md).

`server/` contains the reconstruction milestone specification only. There is no implemented or deployed server or model worker. See [server scope](server/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `bench/` | Synthetic surface and camera experiments, scanned-object mapping, and offline recorded-footage tracking research |
| `mobile/` | Deferred Unity source scaffold for fixture placement and design previews |
| `native/` | C++ coordinate-conversion boundary; no moving-object tracker |
| `server/` | Reconstruction service specification; no implementation |
| `reference/` | Original hackathon source retained as a pinned reference; see its [README](reference/README.md) |
| `tools/` | Offline run, geometry export, source verification, and annotation-scoring utilities |
| `docs/` | Project status and validation procedures |

The original source and dependency pins are retained under `reference/qhacks2025/`; the reference is not a dependency of the current project. Current experiment limitations and evidence are recorded in [model-quality status](bench/MODEL_QUALITY.md) and [project status](docs/status.md).
