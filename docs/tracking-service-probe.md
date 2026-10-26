# Bounded production tracking-service probe

The new variable-length corpus can exercise the actual current
`bench.prototype_service` over HTTP without changing production tracker code or
relaxing the old 110-frame replay validator.

```bash
# From the repository root with the tracking environment active:
python -m unittest bench.test_tracking_service_probe -v
python -m bench.tracking_service_probe \
  --dataset .cache/tracking-benchmark/synthetic-v1 \
  --dataset .cache/tracking-benchmark/real-v1 \
  --frames 20 \
  --output artifacts/tracking-benchmark/service-probe-v1.json
```

Choose fresh output/assets paths for another run; earlier evidence is not
silently overwritten. One or two corpus roots are allowed, with **at most twenty
RGB requests per corpus**. A shorter stream is permitted. By default the first
listed sequence from each inference manifest is used, not a score-selected case.

## What actually runs

1. Read `inference.json` only. The probe never opens `evaluator.json` or evaluation
   depth, masks, poses, visibility, or stress labels.
2. Construct the shared fixed descriptor/3-D bank from exactly ten explicitly
   authorized masked RGB-D/known-pose onboarding frames (maximum 300 ORB anchors
   per view). Export ordinary SHA-bound `asset.json` / `tracking-asset.npz` files.
3. Verify those files through the **unchanged production asset and calibration
   loaders**, including model/source coordinate mapping and array budgets.
4. Start the real `PrototypeService` with the real current `PrototypeTrackerV1`
   using `create_server('127.0.0.1', 0, ...)`. Port zero selects an ephemeral
   loopback port; no public or LAN listener is started.
5. Check `/healthz`, JPEG-encode RGB at quality 85, and POST the service's explicit
   identity/calibration/RGB whitelist to `/v1/frames`; then inspect `/v1/status`.
6. Require exact response identities, recognized states, finite nonnegative
   processing times, proper finite metre SE(3) when a pose is public, and no stale
   pose or diagnostic overlay in nontracking responses. Close the server even if
   an HTTP exchange fails.

The actual service **cold-starts from its immutable onboarding bank**. It receives
no offline last-frame optical-flow seed. That difference is intentional and must
not be hidden when comparing this smoke with seeded offline benchmark results.

Synthetic assets are truthfully labeled `synthetic_control` and use a `synthetic`
calibration. Real HB assets use `dataset_reference`. Both prerecorded streams are
sent with the service's `frame_index_no_fps` clock and null capture timestamps,
even if a synthetic trajectory has analytic timestamps. Neither is relabeled
`physical_measured`, and neither claims fresh live-camera capture.

## Observed local run: 2026-10-26

All six unit tests passed. The full-corpus bounded probe ran only after the serial
geometric timing benchmark finished.

| Input | HTTP requests | Service states | Valid public poses |
|---|---:|---|---:|
| Synthetic `train-smooth-27181` | 20 accepted / 20 HTTP 200 | 1 recovering, 19 tracking | 19 |
| Real HB object 20 | 20 accepted / 20 HTTP 200 | 20 lost (`insufficient_correspondences`) | 0 |

All forty request/response contracts passed: matching identities, valid states,
finite timings, no stale nontracking pose/overlay, and no false live-camera
freshness. Both loopback servers closed. Each asset used 3,000 onboarding anchors.

**This proves that the API pipeline can be exercised, not that real tracking is
working accurately.** The real sequence returned no tracking poses in this short
cold-start prefix. No accuracy score, live-camera FPS, phone/device validation,
neural inference, or readiness claim is made by this probe.

Evidence: `artifacts/tracking-benchmark/service-probe-v1.json`, with source/input
hashes, per-frame contracts, state/failure counts and actual service admission
counts. The fixture tests use only temporary generated resources and need no
network access or downloaded weights.

## Reuse the prepared production assets

The actual probe leaves standard assets under the ignored artifacts tree. To
manually start the same real-HB service on loopback:

```bash
python -m bench.prototype_service \
  --assets artifacts/tracking-benchmark/service-probe-v1-assets/dataset-02 \
  --calibrations artifacts/tracking-benchmark/service-probe-v1-assets/dataset-02/calibrations.json \
  --host 127.0.0.1 --port 8765
# In another terminal, inspect only local API status:
curl --fail http://127.0.0.1:8765/healthz
curl --fail http://127.0.0.1:8765/v1/status
```

Use `dataset-01` instead for the synthetic service. These dataset-reference
assets/calibrations are **not** physical-camera calibration or a phone-ready
setup. The production service intentionally rejects physical-clock requests with
these prerecorded calibrations. Stop a manually launched server with Ctrl-C.
Neither the probe nor these commands load private trained weights.
