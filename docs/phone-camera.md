# Phone camera selection preview

Open `/camera` from the design studio, or use **Try this artwork with the camera**
to transfer the current artwork recipe locally. The transfer contains no camera
image, token or photograph and is removed from session storage when consumed. Use
**Back to design studio** to return the current generated artwork for editing,
PNG download or saving after pairing. Unsubmitted prompt drafts are excluded;
the returned design is unsaved and is never uploaded automatically.

The camera home has three actions: **Ask**, **Analyze** and **Design**. The
Analyze circle is exactly 1.3 times the diameter of the side circles. Open the
rear camera, then hold Analyze to request the center object's outline, or tap
an object directly. The camera stays live. A fresh observed outline exposes
**Use object**; artwork attachment still requires separate confirmation and Apply.
The glow indicates the press, not successful tracking.

The six interface states share frosted glass, a restrained pink/purple/peach rim,
system typography and safe-area spacing:

- **Camera:** circular Ask / Analyze / Design controls; the rim brightens only
  while Analyze is held and clears on release or cancellation.
- **Ask:** a full-screen glass conversation with its own close control. This is
  an explicitly labelled local, rule-based idea helper. It collects object,
  shape, material, feature, pattern and color preferences and transfers the brief
  into Generate; it is not a conversational model or photo-understanding service.
- **Add a design:** a bottom chooser with Generate and Upload.
- **Generate design:** one prompt field, character count, horizontally scrolling
  text-only object ideas and a Generate image action. The existing local engine
  draws finite patterns, colors and quoted text. Object-only requests show an
  actionable capability message rather than fabricate object generation.
- **Analyze held:** the same camera layout with an energized ring and rim.
- **Design generated/uploaded:** an independent floating 2D artwork card, with a
  bottom result sheet offering Regenerate and Upload. It spins up from below,
  settles facing forward and persists until closed. Drag tilts it; Move allows
  translation. Release returns the pose to centered rest; pinch, wheel or plus
  and minus adjust zoom. More view controls exposes Rotate, Move, Reset and the
  separately gated Apply action. No 3D geometry or scene anchoring is inferred.

Settings contains camera switching, selection reset, Stop, artwork options,
Refresh app and Back to studio. Scene taps are blocked behind open surfaces;
keyboard focus stays within the active dialog. Reduced motion removes the long
spin, and reduced transparency supplies stronger material backing. Text-bearing
panels and gradient actions use contrast-budgeted foregrounds over bright footage.
Native phone status bars and browser chrome belong to the operating system.

Import accepts PNG, JPEG and WebP content up to 12 MiB and 24 megapixels. Headers
are checked before decode, EXIF orientation is respected, and the image is fitted
without distortion into a transparent 1024-pixel square. One decode and at most
one newest waiting file are retained. Imported images stay in memory. Their studio
action exports a PNG rather than manufacturing a procedural recipe; generated
recipes retain the existing exact camera/studio round trip.

This is an **experimental live 2D translation preview**. A bounded local patch
tracker follows textured regions while the mask is pending. The mask is accepted
only with surviving, spatially distributed features from inside the original
selection. Uncertain tracking hides the artwork and requires another tap. It
cannot estimate rotation, scale, perspective, geometry or curved attachment.
The trained GoTrack head remains pinned and is not used by this auxiliary path.
Stop, hiding the tab and leaving the page close camera tracks and pending work.

## Start locally and enable phone HTTPS

The existing runtime serves the page without starting any computer-side model:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py tools.serve_design_studio
```

Desktop loopback camera checks can use `http://127.0.0.1:8891/camera`.
Phones need HTTPS with a certificate they trust. Provide a certificate whose SAN
matches the bound private IP, plus its private key:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py tools.serve_design_studio --lan-host 10.0.0.240 --tls-cert C:/path/to/trusted-cert.pem --tls-key C:/path/to/private-key.pem
```

Open `https://10.0.0.240:8891/camera` on the same private network. Replace the IP
and certificate paths with the actual operator configuration. A self-signed
certificate that the phone has not trusted is insufficient. Certificate trust,
firewall access and physical iPhone/Android checks have not been configured or
verified by this change. The existing public comparison tunnel is unchanged.
Camera selection requires no pairing token; optional computer artwork storage
still requires pairing in the design studio. Certificate/key paths are never
served. TLS handshakes have a three-second timeout inside bounded workers.

### Home-screen launch

The camera includes a standalone web manifest, an iPhone touch icon and safe-area
layout. Open the HTTPS camera URL in Safari and use Share → Add to Home Screen,
with Open as Web App enabled when offered. Safari remains supported for quick
iteration. The app menu's Refresh action retrieves current source; the server
serves application assets with `Cache-Control: no-store` and registers no service
worker. The computer and local network connection remain required. Manifest,
icon and browser layouts are software-verified; physical standalone iPhone and
Android validation remains pending.

## Local inference and assets

Camera frames, masks and corrections stay in browser memory. Segmentation runs
inside a browser worker; no frame-upload or server-inference endpoint exists.
The page loads same-origin model/runtime assets. Loading assets requires a
connection; this release does not implement offline installation or persistent
model caching. Explicit PNG downloads are the only camera-preview export.

The pinned runtime is MediaPipe Tasks Vision **1.1.1**, with the cached v2
interactive segmentation task:

```text
38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03
```

JavaScript/WASM assets occupy about 41 MB under `.cache/phone-camera-runtime/1.1.1`.
The existing 30.5 MB model is served in place. To reproduce staging, run:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py tools.phone_runtime --prepare
```

Staging verifies the pinned npm archive, exact extracted members and manifest,
preserving a 1 GiB free-space reserve. It executes no npm scripts. The server
checks pinned assets and the worker verifies model/module bytes before use.
Missing or incompatible assets explicitly disable automatic selection. No
fabricated mask or server inference is substituted. The live flow currently uses
tap selection; frozen brush correction is deferred. Version 1.1.1 declares
positive/negative brush values 1/2 but omits
the JavaScript enum export; the adapter applies those values only to this version.

One segmentation request runs at a time, only after a tap or deliberate Analyze hold. Captured reference
frames have a maximum 640-pixel edge; tracking uses at most 192 pixels per edge
and 192 features, without a frame-history queue. The camera image and its valid
artwork overlay are composed from the same checked frame. The underlying video
stays mounted and renderable for Safari. When verified frames are unavailable,
the page shows raw video only: masks, artwork, Confirm and Apply are disabled.
The first-frame watchdog also runs while the playback promise is pending.
Successful fresh copies commit pixels and geometry together; rejected frames
cannot resize or clear the last accepted source. A desktop CPU
compatibility smoke took about eight seconds for the first manufactured-image
selection. Phone speed, mask quality and sustained operation remain unverified.

The camera fills the viewport behind the Ask / Analyze / Design controls. The
Generate screen owns its prompt; the camera home has no always-visible composer.
A contextual **Use object** button appears only for a fresh valid outline.
**View last design** is in Settings. A labeled marker acknowledges selection while
segmentation runs; it is not an object outline. A nonempty, registered mask with
fresh valid correspondence produces a mint contour with a dark halo. That contour
remains visible after Apply. Empty/rejected masks and lost tracking have separate
persistent explanations, and late replies cannot announce a usable outline after
selection was lost.

Feature selection reserves four eligible points in each of 8×6 normalized image
cells, then backfills unused slots by the original score ordering, capped at 192.
Quality, patch spacing, mask footprint, matching, inlier, spatial-support and timing
gates are unchanged. Manufactured tests show improved support for a lower-contrast
target surrounded by stronger texture. This does not establish physical tracking
accuracy. A transparent target can still match background texture visible through
it, which is not evidence of surface attachment.

## Verification

Focused commands:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py unittest tools.test_phone_static
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe --test tools/test_phone_selection.js tools/test_design_studio_engine.js
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe tools/test_phone_tracker.js
```

`tools/check_phone_camera_browser.js` preserves the earlier frozen-flow regression
fixture. `tools/check_live_camera_browser.js` preserves the earlier continuous
preview fixture. The earlier glass and inspector fixtures retain their historical
layout contracts. Current six-state checks are `tools/check_six_state_ui_browser.js`,
`tools/check_six_state_contrast_browser.js`,
`tools/check_six_state_startup_order_browser.js`, `tools/check_glass_import_browser.js`
and `tools/check_studio_return_browser.js`. Camera and mask fixtures are
manufactured; image imports use actual browser encoders/decoders. These do not
establish real-camera segmentation or physical phone readiness. Tracker controls use
known array-index translations, occlusion, timing failures and delayed masks.

`tools/check_iphone_camera_debug_browser.js` covers startup callback starvation,
copy failures, geometry changes, raw-video recovery, delayed playback promises,
camera ownership after selection, control overlap and cleanup. Run camera tests
in a foreground browser; compositor callbacks may stop in an occluded window.
`tools/check_phone_glass_dock_browser.js` retains the superseded dock layout
contract, and `tools/check_phone_reference_camera_browser.js` retains the
superseded compact-composer contract. The current camera home and six surfaces
are covered by `tools/check_six_state_ui_browser.js`.
`tools/check_phone_outline_states_browser.js` covers contour visibility,
Apply, empty/rejected masks, persistent loss and delayed replies.
`tools/test_phone_feature_coverage.js` contains independent coverage and distractor
controls. `tools/profile_phone_tracker.js` reports bounded manufactured desktop CPU
measurements; those measurements are not phone performance or surface accuracy.

If tapping does not select an object, open Settings and expand
**Camera diagnostics · local only**. Refresh diagnostics to inspect frame
verification, runtime/worker readiness, selection errors and tracker failure
reasons. This disclosure contains no images or device identifiers, is not saved,
and sends no telemetry. A flat or reflective object may yield too few features
for the current experimental 2D tracker even when segmentation finds an outline;
the page reports that failure instead of enabling an unsupported attachment.

Diagnostics retain at most 32 numeric/state trace entries and a copied snapshot
of the first correspondence failure, including failures while segmentation is
pending. New taps and explicit selection resets clear the previous snapshot.
Stopping the camera, hiding the page or leaving it also clears that snapshot;
use **Copy diagnostics** before ending the session. Clipboard denial reveals
selectable JSON for manual copying. None of this is required for normal use, and
it does not establish that the reported physical-phone tracking failure is fixed.

Retain the existing design storage reserve. Save regressions can fail with HTTP
507 when disk space is below that reserve; camera processing itself does not
write captured frames to the computer.


## Connection to the tracking workbench

Open **Settings > Tracking pipeline** and pair with the startup token printed by
the local studio server. The token stays in browser memory. Refresh reads the
real workbench status and clip catalog through the existing fixed-loopback,
same-origin bridge. Disconnect, camera cancellation and hiding/leaving the page
abort requests and clear pairing. There is no automatic polling.

The connection distinguishes reachability from execution readiness. It pins
head `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`,
reports an installed head only when one is actually reported, and blocks older
or conflicting model identities. A disabled workbench cannot be described as
execution-ready. Status is an allowlisted metadata projection; server paths and
arbitrary upstream errors are not exposed to the phone.

**Save artwork for workbench** explicitly saves the current generated pattern
recipe and PNG through the existing StudioStore, then creates the existing
manual-import handoff for an optional catalog clip. This is an artwork handoff,
not automatic application to a neural tracking run. Imported raster images use
PNG export instead of a fabricated procedural recipe. Pairing and Refresh send
no artwork; camera frames and masks are never part of these requests.

The new route `/api/studio/tracking-status` requires the existing pairing token
and Host/Origin guards. `/api/studio/tracking-frames` rejects live requests before
reading their bodies. The original `/v1/frames` descriptor/PnP service is not the
trained GoTrack backend and is never selected as a fallback.

At the 2026-10-24 connection check, the real workbench reported nine registered
clips and the correct requested head, but no installed head, a missing Torch
runtime and missing serving admission. The connection catalog runs with
execution disabled. Trained neural sources remain restricted to authenticated
finite recordings. Live use also needs a reviewed frame contract, calibrated
camera, prepared metric geometry, valid initialization and current observed
foreground; this connection does not supply those prerequisites or establish
physical-phone tracking accuracy.

Software checks: `tools/test_phone_pipeline_bridge.py` covers readiness identity,
authentication, sanitization, rejected frame bodies and the bounded read deadline.
`tools/check_phone_pipeline_bridge_browser.js` covers the dialog, explicit
artwork-only handoff, cancellation, invalid-token recovery and raster protection.
