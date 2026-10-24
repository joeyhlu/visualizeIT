# Native mobile delivery

The native iPhone application source lives in `native-ios/`. It uses SwiftUI for the interface and AVFoundation for the camera; it does not embed the studio website. It has not been built or installed yet. The existing `mobile/` Unity project and browser studio remain separate.

## Visual specification

The user's six selected images are the specification for home, Ask, Add a design, Generate design, Analyze and Design generated. Match the proportions, translucent materials, pink/violet/peach edge lighting, centered wordmark, circular actions and persistent floating artwork. The system supplies the real status bar and home indicator. Do not draw a fake Dynamic Island, battery, clock or navigation chrome.

Reference dimensions are 941 × 1672 pixels. Use proportions and device safe areas rather than treating screenshot pixels as points. At the reference proportions the header icon circles are about 102 pixels wide, side action circles about 130, and the central Analyze circle about 170. Body text, sheets and controls must remain usable with Dynamic Type and accessibility settings. Reduced motion can replace the spin with a fade; reduced transparency can increase backing opacity.

Generation produces the supported local pattern/color/text artwork. The wood-slats option also has a finite native SceneKit preview with eighteen extruded slats, wood grain, pink/violet lighting and a soft floor ellipse. It supports independent orbit, pan, pinch and rotation, with a spinning entrance from below. Other generated/imported artwork remains a flat image preview. This predefined slatted mesh is not general prompt-to-3D generation, reconstructed object geometry or persistent scene attachment. Pattern and export options are in Settings so they do not add rows to the reference generator/result screens.

## Tracking boundary

The native client and separate native pose service use the versioned contract in `docs/native-pose-contract.md`. The current trained head remains pinned to `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311`.

Readiness, frame transport, object-pose inference and artwork rendering are distinct states. Camera motion is not object pose. A tap or rectangle is not a foreground mask. A previous pose cannot be drawn over a newer camera frame without a justified synchronization contract. A runtime/admission failure must suppress attachment and remain visible as a failure.

Source implementation does not establish that the current machine can run the model. Torch/runtime restoration, a valid live admission, calibrated capture and trusted metric assets/observed-mask providers remain deployment prerequisites. The app must request fresh, temporary local-computer processing consent before any camera frame leaves the phone. No cloud-frame permission is implied.

## Validation status

This milestone was requested as coding first, with device tests deferred. Source inspection and configuration/syntax checks are not an iOS build, a pixel comparison or a model evaluation. Exact visual fidelity remains unverified until native screenshots of all six states can be compared against the selected images on Apple tooling. Physical iPhone camera, keyboard, lifecycle, VoiceOver, memory and thermal behavior remain pending. Android native UI is not delivered by this iPhone target.

On a Mac, open the Xcode project and follow `native-ios/README.md` to select a signing team, build and install. Use an actual iPhone for camera behavior; simulator screenshots alone cannot verify capture or tracking. Do not label the app ready based solely on a successful compile.
