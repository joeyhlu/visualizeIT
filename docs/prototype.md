# Initialized tracking prototype

The prototype runs geometric pose tracking on the Windows PC and exposes a browser camera client. It requires a prepared object reference bank and calibrated camera stream. The browser displays each delayed pose over its matching captured RGB frame. This does not provide automatic object discovery, a foreground mask, or a native iOS build.

The original sprint experiment is a five-object YCB-Video pilot. A subsequent frozen HB selection has now evaluated ten additional project-unused object identities: all 2,200 trial rows are accounted for, but zero objects pass attachment or recovery. External dataset poses, depth and visibility are evaluator references; they are not independent human annotations. These runs use a geometric tracker with zero neural updates. Live-camera 30 FPS and physical iPhone alignment remain unverified.

The existing pilot inputs occupy 199,948,565 bytes and can be reused in place. Viewing the sealed report requires no dataset download or copy. The local cache audit also found development footage and partial HOPE/HOT3D inputs; it did not establish ten complete previously unused identities. A bounded supplemental HB acquisition started after disk headroom and validation recovered. Camera and physical-phone testing are deferred at the user's request. Keep the 1 GiB free-space reserve before creating new test or benchmark outputs.

The subsequent compact HB reader and output-budget changes passed independent source review. On that frozen source, 33 read-only tracking/rendering/bank-geometry tests passed, and a cached YCBV RGB/PLY stream-boundary check preserved pixels and geometry exactly. After disk headroom recovered on October 8 Toronto time, the full 48-test storage/benchmark/gallery/dataset suite and eight acquisition controls passed. Initial sandbox permission failures are retained in their logs; the final storage suite ran outside the sandbox's confirmed Windows strict-path restriction, with workspace temporary files and unchanged source. These checks establish storage and interface contracts, not real tracking quality.

The HB acquisition completed on October 9. On October 11 Toronto time, read-only validation verified all 2,569 members against compressed and expanded hashes, CRCs and lengths, plus receipt/manifest bindings and the exact frozen selection. Retained HB inputs occupy 413,077,875 bytes. The first inference attempt stopped at the 1 GiB free-space guard while preparing object 3; it retained both complete trials for objects 1 and 2 but produced no ten-object seal. Free space subsequently recovered to roughly 11.9 GB. One unchanged, bounded retry completed into `hb-results-v2`, followed by separate scoring on October 12 Toronto time. All ten objects failed availability; two also failed paired attachment error limits. The five-object measurements below remain the preserved original results. The original eight-hour deadline passed before this continued validation; it has not been reset or claimed met.

The user subsequently authorized up to 5,000,000,000 bytes total for test inputs, temporary files and results. The current frozen HB run retains its tighter reviewed guards: 416 MiB input cache, 128 MiB results, 32 MiB scratch and 1,000,000,000 bytes aggregate. No input, configuration or guard was changed merely to consume the larger allowance.

## Start the desktop demo

Run from the repository root using the existing pinned Python 3.12 environment. The wrapper defaults to this machine's Codex runtime; pass `-Python` to select an equivalent environment with the existing OpenCV, NumPy and Pillow dependencies. It uses explicit package paths and disables site startup hooks.

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_service --assets .cache/prototype-sprint/results-v1 --calibrations .cache/prototype-sprint/calibrations-pilot-v1.json --demo .cache/prototype-sprint/results-v1/demo-v2 --host 127.0.0.1 --port 8766
```

Open `http://127.0.0.1:8766/`. The recorded benchmark link shows both trials, every scored frame's state, sampled renders and dataset-reference controls. Its playback is precomputed. The camera page is a separate frame-processing client. Dataset calibration entries are intentionally absent from the physical-camera selector.

An explicit synthetic fallback is also available:

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_service --assets .cache/prototype-sprint/synthetic-results-v1 --calibrations .cache/prototype-sprint/calibrations-synthetic.json --demo .cache/prototype-sprint/synthetic-results-v1/demo-v2 --host 127.0.0.1 --port 8767
```

Synthetic results demonstrate execution and geometry only. They contain zero real unseen identities.

## Local HTTPS and iPhone preparation

Use a private LAN address for the PC and keep camera traffic local. Generate a fresh certificate directory, replacing `192.168.1.20` with that address:

```powershell
./tools/run-prototype.ps1 -Module tools.prototype_certificate --output .cache/prototype-sprint/https-phone --ip 192.168.1.20
./tools/run-prototype.ps1 -Module bench.prototype_service --assets .cache/prototype-sprint/results-v1 --calibrations .cache/prototype-sprint/calibrations-physical.json --demo .cache/prototype-sprint/results-v1/demo-v2 --host 192.168.1.20 --port 8765 --cert .cache/prototype-sprint/https-phone/server-cert.pem --key .cache/prototype-sprint/https-phone/server-key.pem
```

The generator creates a seven-day local certificate and prints its fingerprint. It does not change Windows or phone trust. Transfer only `server-cert.cer` to the phone through a local method. The device operator must install and explicitly trust that certificate, verify its fingerprint, and open the matching `https://PC-LAN-IP:8765/` address in Safari. Keep `server-key.pem` private on the PC. Certificate installation, firewall access and physical testing have not been performed by this sprint. Do not put this camera endpoint on the existing public comparison tunnel. The supplied physical registry is empty so the endpoint can start with pose output explicitly unavailable.

Supply a measured camera registry before expecting physical pose output:

```json
{
  "schemaVersion": 1,
  "calibrations": [{
    "calibrationId": "iphone-rear-landscape-640x480-v1",
    "kind": "physical_measured",
    "objectId": 2,
    "width": 640,
    "height": 480,
    "K": [[500, 0, 319.5], [0, 500, 239.5], [0, 0, 1]]
  }]
}
```

The numbers above are a schema example, not measured phone intrinsics. Replace them with calibration for the selected lens, resolution, orientation and crop. Save the registry privately under `.cache/prototype-sprint/`. Missing or mismatched calibration suppresses the pose. Calibration of distortion and stream preprocessing must match the pinhole input expected by the tracker. Rotating the phone or switching lenses requires matching calibration and restarting the session.

Select the matching prepared physical object and calibration, start the camera, and inspect single-frame results before enabling continuous capture. Camera preview rate is not inference rate. The page separately reports completed results, browser-reported distinct-frame valid poses, neural updates, processing time and capture-to-response delay. Browser frame callbacks are not independent sensor timestamp certification.

## Reproduce the frozen pilot

Preserve existing outputs. Inference refuses an existing destination; choose a fresh version for each rerun.

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_benchmark infer --dataset .cache/prototype-sprint/dataset-pilot-v1 --output .cache/prototype-sprint/results-v2
./tools/run-prototype.ps1 -Module bench.prototype_benchmark evaluate --dataset .cache/prototype-sprint/dataset-pilot-v1 --output .cache/prototype-sprint/results-v2
./tools/run-prototype.ps1 -Module bench.prototype_gallery --dataset .cache/prototype-sprint/dataset-pilot-v1 --results .cache/prototype-sprint/results-v2
```

`selection.json` freezes identities and windows. `inference.json` contains ten permitted onboarding records and 110 RGB/calibration test records per object. Test poses, masks and depth stay in `evaluator.json`. `freeze.json`, `source-snapshot/` and `inference-seal.json` bind sources, runtime, inputs and outputs. Evaluation checks the seal before reading labels. Natural and interrupted trials retain every scheduled frame; interruption blanks five frames at scored indices 50–54.

The dataset cache is capped at 384 MiB, below the user's 1 GB ceiling, and acquisition preserves at least 1 GiB free. No existing data was deleted. The original failed ten-object selection is retained separately under `.cache/prototype-sprint/dataset-v1/`.

Attachment gates require median error below 5 px, P95 below 10 px at 720p-equivalent scale, and availability at least 90%. Recovery requires observed loss, a correct first public reacquired pose, two consecutive correct visible poses within 15 observable frames, and whole-trial attachment/availability. Failed or missing frames remain in the denominator. A five-object pilot cannot pass the ten-object aggregate gate.

The frozen `results-v1` measurements are:

| Object | Natural median / P95 (720p-equivalent px) | Natural availability | Attachment gate | Recovery gate |
|---|---:|---:|---|---|
| Cracker box | 3.51 / 7.17 | 100% | Pass | Pass |
| Tomato soup can | 3.20 / 7.83 | 100% | Pass | Fail: incorrect first reacquisition |
| Tuna fish can | 2.14 / 4.36 | 100% | Pass | Fail: interrupted availability 80.9% |
| Potted meat can | 1.96 / 5.09 | 50.9% | Fail: availability | Fail: no qualifying recovery |
| Large marker | 5.03 / 9.13 | 100% | Fail: median error | Fail: no qualifying recovery |

These paired error summaries exclude frames without a public tracking pose; the availability column accounts for those failures. Three of five pass natural attachment, and one of five passes full recovery. All 550 distinct held-out frames and both 550-frame trials are retained. No configuration was tuned against these outcomes. See the sealed `report.json` and sampled failure panels for the complete evidence.

The inspected handoff gallery is `results-v1/demo-v2/`. Earlier galleries remain preserved. During forced interruptions, its input and estimated panels show the actual blank input; the separately labeled dataset-pose control shows the original recording. The gallery is an evaluator visualization and is never sent to the tracker.

Replay the API over local HTTPS independently of the benchmark:

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_service --assets .cache/prototype-sprint/results-v1 --calibrations .cache/prototype-sprint/calibrations-pilot-v1.json --demo .cache/prototype-sprint/results-v1/demo-v2 --host 127.0.0.1 --port 8765 --cert .cache/prototype-sprint/https/server-cert.pem --key .cache/prototype-sprint/https/server-key.pem
```

In a second terminal:

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_replay --dataset .cache/prototype-sprint/dataset-pilot-v1 --results .cache/prototype-sprint/results-v1 --url https://127.0.0.1:8765 --ca-file .cache/prototype-sprint/https/server-cert.pem --output .cache/prototype-sprint/service-replay-v4.json
```

Choose a fresh report name. The replay journal records each scheduled outcome as it completes. API replay uses JPEG quality 85 and descriptor-based initialization, while the frozen benchmark starts from the last authorized onboarding frame. Its pose availability is a separate service control, not a rerun of benchmark accuracy. Rates use measured replay-loop wall time; they are offline throughput and do not establish live-camera 30 FPS.

Final HTTPS replay `service-replay-v3.json` completed 550 requests in 32.48 seconds: 16.93 responses/sec, 11.73 valid public poses/sec, 381/550 pose availability (69.3%), zero transport drops, timeouts or stalls, and zero neural updates. End-to-end latency was 48.80 ms median and 146.43 ms P95. This is a 32-second offline replay, with input validation and final report writing excluded from the timing denominator; no ten-minute real-camera test occurred.

## Frozen ten-object HB follow-up

The private input directory is `.cache/prototype-sprint/hb-compact-v1/`, pinned to revision `985ebaef7b1bd4bffb2039f18dd81faa647fd060` and selection digest `af9d9980659b76213e1182c79551d0128dadb8b5d055650fc9b55cc8d6c86bb5`. Stable-order object IDs 1–10 each use onboarding frames 0–9 and scored frames 10–119. The 1,100 scored object-frames reuse 720 distinct RGB frames across six scenes; they are not 1,100 independent videos or object identities. The two trials schedule 2,200 output rows, retaining tracking loss, missing outputs and the five-frame interruption at scored indices 50–54.

`hb-integrity-v1.json` records the parent verification. `acquisition-receipt.json` binds the original losslessly stored ZIP members and manifests. The inference reader permits only model geometry, onboarding RGB/depth and scored RGB; evaluator poses, masks and scored depth remain outside its whitelist. Evaluation must run only after a valid inference seal. Pretrained exposure is unknown. This is initialized tracking transfer with known-pose onboarding, not automatic object discovery.

The completed retry uses `hb-results-v2`; the aborted `hb-results-v1` and its stop receipt remain preserved. The temporary space-drop cause has not been established, and no files or system settings were changed to recover space. `hb-seal-audit-v2.json` verifies eleven source/snapshot pins, 23 runtime pins, all twenty trial hashes and every scheduled row. Its 2,420 logged resource accesses match the inference whitelist; evaluator references were opened only by the subsequent scoring process. To reproduce the frozen configuration, choose a different, nonexistent result directory and preserve every prior result. The reviewed storage guard may stop a rerun if retained trials plus new outputs exceed its stricter 1 GB aggregate limit; the user's larger allowance does not silently change that guard.

The sealed HB report is `hb-results-v2/report.json` (SHA256 `9bb77adca89ff79b20b252266e8aeae874734fd70d2075439ea15e6a4befd764`). Natural-trial measurements are:

| HB ID | Median / P95 error, 720p-equivalent px | Tracking availability | Attachment | Recovery |
|---|---:|---:|---|---|
| 1 | 1.93 / 6.35 | 13.6% | Fail | Fail |
| 2 | 1.37 / 7.86 | 14.5% | Fail | Fail |
| 3 | 1.37 / 4.43 | 3.6% | Fail | Fail |
| 4 | 5.33 / 22.71 | 16.4% | Fail | Fail |
| 5 | 1.78 / 8.01 | 12.7% | Fail | Fail |
| 6 | 1.14 / 2.78 | 1.8% | Fail | Fail |
| 7 | 1.21 / 6.22 | 10.0% | Fail | Fail |
| 8 | 5.82 / 23.11 | 6.4% | Fail | Fail |
| 9 | 0.76 / 1.18 | 0.9% | Fail | Fail |
| 10 | 3.38 / 7.36 | 3.6% | Fail | Fail |

Errors are conditional on available poses and visible paired surface points. Object 9 contributes only one paired frame; a small error there does not establish sustained attachment. The natural trial tracks 92 of 1,100 scheduled object-frames (8.4%). All objects are already lost before the fixed interruption, and none subsequently publishes a reacquired pose. Therefore these trials establish no successful controlled recovery. There are zero evaluator-unavailable frames. No threshold, identity, window or configuration was tuned against the scores.

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_benchmark infer --dataset .cache/prototype-sprint/hb-compact-v1 --output .cache/prototype-sprint/hb-results-reproduction-v1
./tools/run-prototype.ps1 -Module bench.prototype_benchmark evaluate --dataset .cache/prototype-sprint/hb-compact-v1 --output .cache/prototype-sprint/hb-results-reproduction-v1
./tools/run-prototype.ps1 -Module bench.prototype_gallery --dataset .cache/prototype-sprint/hb-compact-v1 --results .cache/prototype-sprint/hb-results-reproduction-v1
```

The completed HB gallery is available at `http://127.0.0.1:8882/demo/`. It contains 100 sampled comparisons and all 2,200 trial state entries, with 5,502,645 generated bytes and zero unavailable samples. Every JPEG decoded successfully, every served image matched its local bytes, and all timeline IDs/states and fifty interruption rows match the sealed traces. The desktop browser loaded all 100 images without errors and displayed all ten objects/twenty trials. Initial attachment and subsequent overlay suppression were inspected against the separately labeled dataset-pose controls. The gallery is precomputed, with no new video export or public upload. The existing five-object demo remains preserved.

The prepared `calibrations-hb-v1.json` registry binds the ten objects to dimensions decoded from verified RGB and unchanged native intrinsics across their selected frames. The existing service loader has accepted all ten tracking assets and calibration bindings. There are zero physical-phone calibrations; asset loading is not evidence of physical alignment, tracking quality or live performance.

The isolated HB service starts with:

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_service --assets .cache/prototype-sprint/hb-results-v2 --calibrations .cache/prototype-sprint/calibrations-hb-v1.json --demo .cache/prototype-sprint/hb-results-v2/demo --host 127.0.0.1 --port 8882
```

Its camera client at `http://127.0.0.1:8882/` was inspected in the desktop browser: all ten prepared objects appear, and the calibration selector correctly shows no measured camera calibration. No camera permission or capture was performed. Navigation to the completed recorded gallery was also verified. A three-frame HTTP probe to `/v1/frames` returned 200 with matching identities for source frames 10–12 of HB object 1, but every response reported insufficient correspondences, lost pose, unavailable mask and suppressed rendering. This checks transport and state reporting; it does not demonstrate successful attachment. The probe uses JPEG quality 85 and descriptor-based startup; the benchmark uses decoded original RGB and a last-onboarding-frame seed. Both encoding and initialization differ, so this comparison does not isolate the cause. The initial probe's mistaken singular URL returned 404 and is retained separately; the corrected probe changed no application source. Evidence is `hb-service-functional-v2/receipt.json`, with no performance or live-camera claim.

## Reviewed tracker repair, October 12

The repaired tracker is independently accepted for the frozen source in `tracker-repair-candidate-v2`. It rejects ambiguous anchor/pixel identities, checks the actual accepted 3D support, removes invalid refined correspondences, and reconciles conflicting optical flow and descriptor observations using independent support. It still requires consecutive consistent observations before recovery and suppresses nontracking poses. Numerical/image regressions pass 51/51; service/replay checks pass 24/24.

One synthetic development scene was rerun with the original thresholds and all 220 scheduled trial rows. Errors below are conditional on available poses, in 720p-equivalent pixels; the paired populations differ.

| Trial | Baseline tracking | Repaired tracking | Baseline median / P95 | Repaired median / P95 |
|---|---:|---:|---:|---:|
| Uninterrupted | 104/110 | 103/110 (93.6%) | 1.335 / 4.142 | 1.123 / 5.688 |
| Five-frame interruption | 87/110 | 100/110 (90.9%) | 1.282 / 3.408 | 1.387 / 9.647 |

Repaired interrupted recovery passes in three observable frames. Increased availability also comes with a worse conditional error tail; this is no uniform accuracy improvement. The synthetic attachment checks pass, but this scene establishes zero real unseen objects and no independent human accuracy. The historical failed HB results above remain unchanged and were not used for repair tuning. The repaired configuration has not yet been tested on a new real-object benchmark.

The inspected synthetic comparison is at `http://127.0.0.1:8883/demo/`: 12 decoded and browser-loaded images, both complete 110-frame timelines, and explicit blank-input/suppressed-overlay and recovered samples. A separate three-frame RGB-only HTTP probe returned matching identities, first a recovering response with no public pose, then two valid tracking responses. Masks remain unavailable and service rendering is a diagnostic wireframe. This is a functional offline probe, not a live-camera or throughput measurement. The gallery itself is precomputed.

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_service --assets .cache/prototype-sprint/tracker-repair-candidate-v2 --calibrations .cache/prototype-sprint/calibrations-tracker-repair-v2.json --demo .cache/prototype-sprint/tracker-repair-candidate-v2/demo --host 127.0.0.1 --port 8883
```

Reproduce to a fresh output directory; the commands preserve the prior versions:

```powershell
./tools/run-prototype.ps1 -Module bench.prototype_benchmark infer --dataset .cache/prototype-sprint/synthetic-control-v1 --output .cache/prototype-sprint/tracker-repair-reproduction-v3
./tools/run-prototype.ps1 -Module bench.prototype_benchmark evaluate --dataset .cache/prototype-sprint/synthetic-control-v1 --output .cache/prototype-sprint/tracker-repair-reproduction-v3
./tools/run-prototype.ps1 -Module unittest bench.test_tracker_repair bench.test_prototype_tracking bench.test_prototype_benchmark
./tools/run-prototype.ps1 -Module unittest bench.test_prototype_service bench.test_prototype_replay
```

## Original requirement acceptance

| Original requirement | Status | Evidence and remaining work |
|---|---|---|
| Validate coordinates and fix orientation, drift and recovery | Fail: real-object evidence | Coordinate controls and the reviewed synthetic repair pass; sustained real-object attachment/recovery have not passed. The preserved historical ten-object run fails. |
| Measure against independent human annotations | Blocked | No independent annotator or completed human landmark labels. External dataset-reference scores are reported separately. |
| Test a frozen configuration on at least ten unseen objects | Pass: execution | Ten project-unused identities were selected before inference; all 1,100 scheduled object-frames and both trials were scored. Zero objects pass the quality gates. Pretrained exposure is unknown. |
| Demonstrate actual sustained 30 FPS live inference | Unverified | No ten-minute real-camera test. Preserved five-object HTTPS replay is offline: 11.73 valid poses/sec, 69.3% availability and zero neural updates. Recording/preview FPS and processing-only timing are not live throughput. |
| Integrate and validate the phone app | Blocked | Windows service/browser integration is tested. No measured phone calibration, hands-on iPhone validation or validated native installation. Unity remains outside this sprint. |

The original eight-hour deadline was missed. The reviewed numerical repair and reproducible prototype do not complete the requested working tracking app. Further validation needs a reviewed real-video onboarding adapter and held-out tests on the existing recordings; another new-unseen benchmark must exclude already observed identities. The current frozen experiments remain intact and must not be used for tuning.

## Checks

```powershell
./tools/run-prototype.ps1 -Module unittest bench.test_prototype_tracking bench.test_prototype_benchmark bench.test_prototype_dataset bench.test_prototype_service bench.test_prototype_gallery bench.test_prototype_replay
node bench/test_prototype_client.js
```

The service tests create loopback listeners. Sandbox socket restrictions may require the same local-network execution permission used for the service. Geometric controls, schema tests and browser integration establish their tested contracts; they do not establish real attachment accuracy or phone readiness.

Current limitations: object-specific initialized tracking, descriptor ambiguity, drift and tracking loss; no observed foreground segmentation in the live service; diagnostic wireframes only; offline galleries use measured dataset depth for explicitly labeled occlusion controls; no sustained real-camera measurement; no physical iPhone validation or native installation.
