# Inspect a saved offline export

The review command creates an exact-frame contact sheet from a finalized
offline export. Each selected frame has three panels: source RGB, its registered
observed mask, and the decoded exported composite. Frame identity, pose state
and render reason stay visible alongside the images.

```text
python -B -m tools.workbench_export_review --workspace WORKSPACE --export WORKSPACE/derived/EXPORT --manifest-sha256 SHA256 --frame-id FRAME_A --frame-id FRAME_B --output WORKSPACE/derived/NEW_REVIEW --tile-width 256
```

Supply the SHA-256 of the export's complete `manifest.json`. Choose frame IDs
from its scheduled window. Repeated `--frame-id` options select up to 24 unique
frames; the sheet follows source schedule order rather than argument order.
The command refuses unknown or repeated IDs. Both the export and the new output
must be direct children of the workspace's derived directory. Existing output
is never overwritten.

The tool verifies the manifest, fixed FFV1 video, registered source clip and
observed-mask registry. It decodes the entire exported video, checking the
expected image hash for every scheduled frame, including frames omitted from
the sheet. It opens source RGB and masks only for the selected frames. Before
publication, it rechecks the input identities and selected decoded source
images. Input changes produce a refusal rather than a partial published review.

Outputs are `review.png` and `report.json`. The report records exact frame IDs,
source indices, panel rectangles, input hashes and original states. A lost or
recovering pose remains labelled as such. A black mask panel can represent an
observed empty mask; inspect the recorded mask state to distinguish empty and
unavailable evidence. No diagnostic changes the original mask or visibility
decision.

Panel resizing uses nearest-neighbour sampling. It is intended for inspection,
not boundary-distance measurements or pixel-accurate annotation. Use the
original registered images for those tasks. Sheet dimensions and encoded output
sizes are bounded; select fewer frames or a smaller tile width if the requested
sheet exceeds the limits. Frames remain in bounded memory and are not dumped
as separate full-resolution image files.

This is saved-artifact inspection. It does not rerun segmentation, pose tracking
or rendering; it does not establish independent accuracy or certify the original
model's admission. It uses no evaluator labels, reference poses, mesh files or
weights. The original run, masks and export are preserved. A review sheet should
be interpreted together with the original export's limitations and provenance.

Model-free synthetic checks:

```text
python -B -m unittest tools.test_workbench_export_review tools.test_workbench_export_review_independent
```
