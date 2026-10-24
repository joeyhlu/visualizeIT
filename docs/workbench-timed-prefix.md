# Timed onboarding-prefix experiment clips

`tools/workbench_timed_prefix.py` creates a **new, separately pinned
experimental clip** from an existing historical `video_file` clip. It never
changes the frozen six-case suite or any run already in progress.

```powershell
python tools/workbench_timed_prefix.py --source-clip dev-mug --clip-id exp-mug-prefix10sec-v1 `
    --out artifacts/video-workbench/timed-prefix/exp-mug-prefix10sec-v1          # artifact only
# add --register to also create .cache/video-workbench/clips/<clip-id>/clip.json (never overwritten)
```

## What the new clip contains

- **Clock:** `clock_kind = timestamp_seconds`. The clock source is the OpenCV
  decoder PTS probe (`opencv_ffmpeg_decoder_pts_probe_v1`).
- **Frame timestamps:** each frame's `timestamp_s` and `media_time_s` equal the
  observed PTS of its *original* source frame index. Nothing is cropped,
  re-encoded or re-indexed.
- **Onboarding prefix:**
  - up to 24 keyframes (`--keyframes`), chosen uniformly in decoder time over
    at most 10 seconds (`--seconds`);
  - it ends exactly at the existing reviewed selection frame, includes that
    endpoint, and never reaches the first scored frame;
  - the reviewed selection points stay on that last frame.
- **Scored frames:** the original scored frame IDs, source indices and window,
  unchanged.
- **Unchanged from the source:** media identity, K and its dataset-reference
  uncertainty, and coordinate conventions.
- **Provenance:** records:
  - the source clip ID and hash, and its window;
  - the frozen prefix source indices;
  - the clock sidecar hash;
  - the label "pure-RGB automatic preparation experiment".
- **Removed:** the inherited metric-CAD and controlled-initial-pose references,
  so they are never passed to automatic preparation.
- **Claims:** `evaluator_inputs_read = false`; there are no accuracy, mobile or
  unseen-object claims.

## Separation

- The source video is verified by whole-file hash (`FrameReader.verify_media`).
- Timing comes from `probe_video`, which calls `grab` and reads
  `CAP_PROP_POS_MSEC` for each frame. Pixel data is never retrieved, analysed
  or passed to preparation.
- Later preparation receives only the frozen onboarding-prefix RGB and masks.
- Missing or non-increasing decoder timestamps make the tool fail; frame rates
  are never used to guess time.
- No evaluator, depth, mask or pose input is read, and no model, segmentation,
  GPU or reconstruction job is started.

## Outputs (new directory only; atomic writes; 1 GiB free-disk reserve checked)

| File | Content |
|---|---|
| `clock.json` | Decoder PTS for every source frame, video hash, frame count, min/max frame interval (≤ 1 MiB) |
| `clip.json` | The validated `workbench_clip_v1` ClipSpec |
| `receipt.json` | Seed, first scored frame, prefix indices/IDs/PTS/duration, scored-frame checks, PTS range, separation statement, tool/source hashes, runtime versions, output hashes |

The tool refuses an existing output directory or clip ID, a path through a
symlink, more than 10 seconds or more than 24 keyframes.

## First generated experiment (artifact only, not registered)

`dev-mug` → `exp-mug-prefix10sec-v1`:

- **Prefix:** source frames 226, 252, … 800, 826 (24 frames, 10.000 s).
- **Seed:** `s000826`.
- **Scored frames:** `s000827`–`s001066` (240 frames), unchanged.
- **Decoded PTS range:** 0–29.767 s.
