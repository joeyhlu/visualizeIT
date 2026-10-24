# Workbench recorded-run provenance strip

The **4. Inspect frames** panel shows a small provenance strip directly under its
heading, above the transport controls and the four panes. It identifies the
**loaded recorded run**, not whatever the create-run form currently contains.

## Create form vs recorded replay

- **Create form (3. Run):** `#run-masks` and the other controls describe a run
  you might create *next*. They keep the clip's default (for example
  `sam2-workbench-01`) by design, and the strip never reads them.
- **Recorded replay (4. Inspect frames):** the strip reads only the loaded run's
  recorded `RunSpec.config`, from the existing `GET /api/runs/{id}` response.
  That response carries `run` (the `run.json` content), and the strip uses
  `run.run.config`, or `run.config` for the fallback shape. There is no extra
  request and no backend change. Nothing is inferred from a mask-set name.

## Fields shown

| Line | Source | When absent |
|---|---|---|
| Loaded run | `RunSpec.run_id` | `unknown` |
| Recorded observed mask set | `config.mask_set` | `not recorded` |
| Recorded segmentation prompt SHA-256 (full, wrapped) | `config.segmentation_prompt_sha256` | `not recorded` |
| Run class | shown only if `config.development_diagnostic_only === true`: "Development diagnostic", followed by the recorded `config.segmentation_diagnostic` text verbatim when present | line omitted |

**Unknown states:**

- If the run has no recorded config object, the mask and prompt lines read
  `unknown (no recorded run configuration)`.
- A recorded value that isn't a non-empty string reads `unrecognised recorded
  value`.
- The form is never used as a fallback.

**What the strip does not say:** it does not claim that weights are unchanged,
because no recorded field states it. It makes no human-accuracy, independence or
phone-performance claim, and ends with: "Recorded configuration for this replay.
Attachment accuracy: unverified."

## Lifecycle (no stale identity)

- **Cleared** when a clip is opened, when a run is opened (before the request,
  so a failed open leaves it empty), when a new run is started, and at the start
  of rendering a run.
- **Filled** only after the loaded run's data has been applied to the viewer.

## Rendering

- All external strings are set with `textContent`; no HTML is built from data.
- It reuses the existing `hint` and `mono` styles, with no new CSS or HTML file
  change. A long SHA wraps with an inline `overflow-wrap: anywhere`, so the frame
  slider and pane layout are unchanged.
- Everything else in the viewer is unchanged: per-role recorded render-pixel
  captions, the pretrained-baseline identity label, view headers, counts, model
  proof, footer, bookmarks, scheduling and forms.

## Expected display (for browser verification)

- `run-20261021T030702-475adc39`: mask set `sam2-workbench-01`; prompt SHA
  `not recorded`; no run-class line.
- `run-20261021T060541-e8894b79`: mask set `sam2-opening-box-v1-qualified`;
  prompt SHA
  `15dce4be95eb362c9044f808c872abd810f98f072cc8f9bd96b79439ddd308a5`; run class
  "Development diagnostic (recorded segmentation: opening-only whole-box+original
  points, isolated SAM states)".
- Switching back to the original run clears the prompt line to `not recorded`.
