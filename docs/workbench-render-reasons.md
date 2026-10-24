# Workbench viewer: recorded render exclusion counts

This is a diagnostic display only. It adds no head, accuracy or phone-quality claim.

## What is shown

The run viewer adds a compact caption below each scheduled role's artwork pane (relative
geometry, new head and frozen baseline), so the image is never covered:

```
Recorded pixels: rendered 0; excluded by mask 2734; back-facing 0; overlap 0
```

- **Source:** the values come from that role's own `FrameResult` for the **current frame**,
  `evidence.render_stats`: `rendered_pixels`, `mask_rejected_pixels`,
  `backfacing_rejected_pixels` and `overlap_rejected_pixels`. The service already publishes
  these through `safe_public_frame`. Each role uses only its own row; the primary's values are
  never copied to the baseline.
- **Missing values:** a missing, non-numeric, non-finite or negative count is shown as
  `not recorded`, never as 0. A row without `render_stats`, such as a suppressed recovering
  frame, shows `Recorded pixels: not recorded for this frame`. A missing row shows
  `… (no result row for this frame)`.
- **Frame changes:** every role's caption is cleared on each frame change before the current
  frame is drawn, so no earlier frame's values survive.
- **Not shown:** internal identity, patch-face counts and `folded_faces`.
- **No interpretation:** no cause is inferred, and nothing says whether the foreground mask is
  correct. These are recorded counts only.
- **Unchanged:** the mask, pose and render states, overlay suppression, the per-frame state
  table, bookmarks and controls.
- **Safe rendering:** text is set with `textContent` only, with no HTML. The caption element
  (`p.render-counts`) is created by `workbench.js` because `index.html` is unchanged.

## Example: cracker box run `run-20261021T030702-475adc39`

| Frame | Role | Caption |
| --- | --- | --- |
| s000434 | primary | rendered 2458; excluded by mask 403; back-facing 0; overlap 0 |
| s000434 | baseline | rendered 2446; excluded by mask 406; back-facing 0; overlap 0 |
| s000435 | primary | rendered 0; excluded by mask 2734; back-facing 0; overlap 0 |
| s000435 | baseline | rendered 0; excluded by mask 2734; back-facing 0; overlap 0 |
| s000371 (recovering) | both | not recorded for this frame |
