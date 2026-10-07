# Public scan provenance and attribution

This offline mapping extension uses five `google_16k` textured meshes of physical objects from the [YCB Object and Model Set](https://ycb-benchmarks.s3.amazonaws.com/index.html). The official dataset page identifies the scanning systems, download variants, incomplete-scan limitations, and dataset license.

Data authors: Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, Aaron M. Dollar. Data license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Preserve this attribution when sharing derived renders. The authors do not endorse VisualizeIt.

| Scan | Shape challenges |
|---|---|
| `025_mug` | Handle opening, interior, curved wall, rim |
| `006_mustard_bottle` | Narrow neck, shoulders, transitions between curved and flat regions |
| `024_bowl` | Concave interior, convex exterior, rim |
| `011_banana` | Curved irregular body and narrow ends |
| `035_power_drill` | Handle, cavities, sharp transitions, detailed irregular shell |

`ycb_assets.py` pins each archive's SHA-256 and copies only the OBJ, MTL and photographic texture from regular archive entries; no scripts are executed. Cached data and regenerated outputs are ignored by Git. Each result includes its original download URL, archive hash, attribution and measurements. This is a small asset download, not a downloaded neural model or a trained network.

Changes: exchange source Y/Z axes and reverse triangle winding, place the object at a bottom-centred origin, derive vertex normals from geometry for UV generation, replace nonfinite source normals when needed, generate new metric UV charts, and render a replacement checker pattern or quality colours. **Positions and topology are retained**, apart from coordinate conversion/translation; no holes are filled or geometry invented. Nominal source coordinates are treated as metres, but dimensions have not been independently measured. Invalid UV/geometry triangles remain recorded and are not accepted as trustworthy mapping.

UV charts use the released [xatlas Python binding](https://github.com/mworchel/xatlas-python), version 0.0.11, to the existing C++ xatlas library. Independent NumPy measurements compute surface/UV Jacobian singular values, physical-area-weighted distortion and chart-boundary coordinate jumps. UV unwrapping and seam measurements run on the cached meshes; they do not establish capture, reconstruction, pose tracking, or live AR.

## Rebuild

Use Python 3.12, with `bench/requirements-real.txt` installed in a virtual environment. In the existing workspace, the bundled Python already provides NumPy/Pillow and xatlas is isolated under `.cache/vision`; no Unity installation is needed.

```text
python -m bench.ycb_assets
python -m unittest bench.test_renderer bench.test_scan_mapping
python tools/test_evaluate_tracking.py
python -m bench.real_objects --views 8
python -m http.server 8875 --bind 127.0.0.1 --directory artifacts/real-objects
```

Open `artifacts/real-objects/index.html` directly or visit localhost port 8875. It uses local assets and no remote scripts. Outputs include the original/photo, checker and per-face quality renders, eight known-pose orbit views per object, a GIF, metric UV mesh arrays, camera intrinsics/transforms, raw quality results and a five-object contact sheet. The orbit is sparse inspection footage, not a measured tracking rate. CPU batch render duration is not a mobile FPS measurement.

The camera uses 960 × 540 pixels and `fx=fy=637.5 px`, the same field of view as the earlier 720p bench. Distance adjusts to frame each unchanged object; complete poses are recorded per view. Per-face distortion is independent of camera pose.

Provisional diagnostics accept a triangle when both UV-to-surface singular values are within 2% of unit scale and anisotropy is ≤1.05. The area denominator includes all measurable geometry; collapsed UV regions count against coverage. Quantiles use valid triangles and physical-area weights; invalid area is reported separately. Seam phase is the length-weighted median of the mean wrapped coordinate jump sampled along chart edges, in full texture tiles. It does not measure rotation continuity or guarantee seamless imported designs. The quality image marks local distortions only; a teal chart can still have a bad boundary seam.

No full natural-appearance or physical-device gate passes from this experiment. Next address chart orientation/phase, seam placement and unreliable areas, then compare the same designs against physical phone footage once mobile builds are available.
