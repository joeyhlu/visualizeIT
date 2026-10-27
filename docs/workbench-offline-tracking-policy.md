# Offline tracking policy: Packet A

This packet adds model-free contracts and saved-video export accounting. It does
not establish tracking accuracy, backend reacquisition, independent attachment
correctness, or physical-phone readiness. No neural inference is required.

`bench/workbench_tracking_policy.py` separates accepted measured evidence from
display proposals. Predictions never become `FrameResult` measurements or supply
object visibility. The experimental policy permits at most two consecutive
hold-last-pose proposals with opacities 0.67 and 0.33. Genuine source timestamps
also limit age to 100 ms. Null timestamps use ordinals only, with no elapsed-time
guarantee. A reset starts a new epoch and clears the usable prior. These limits
are proposed experiment gates, not validated performance thresholds.

Recovery requires explicit current user re-onboarding and two fresh consistent
observations. Recovery diagnostics do not authorize artwork while the public row
is recovering. No automatic candidate generator, backend reseeding, or backend
checkpoint restoration is provided by this packet.

## Saved exporter behavior

Run `python -B -m tools.workbench_offline_export --help` for the CLI.
The exporter retains each original `FrameResult` dictionary under `original`.
It adds a separate `display_decision` using the shared decision contract, including
flat row-major pose values, epoch, measurement status and current-mask trust.
The original saved pose remains a nested 4 by 4 matrix.

The default `legacy_saved_context` arm uses authenticated historical measurements
without claiming to replay a causal prefix. Measured tracking rows compose only
against their verified, same-frame, nonempty observed mask. Recovering, lost,
unavailable and blocked rows preserve original RGB. Empty masks, back-facing or
behind-camera artwork and unsupported projections produce no successful artwork.
Historical saved rows cannot authorize a predicted pose; prediction accounting is
therefore explicitly zero for this exporter. This preserves historical output
pixels while making the measurement and display distinction visible.

`--experimental-tracking` requires the entire contiguous saved causal schedule
after the authorized seed through the output window. A late window with a missing
prefix is refused; an authenticated metadata checkpoint alone cannot restore an
unimplemented backend state. Every export is a fresh offline display epoch zero;
no historical prior is imported into it. This arm verifies actual planar geometry,
persistent anchor, orientation and chart fold/overlap before composition. The
surface admission binds calibration and dimensions, metric CAD provenance, mesh
digest, and the exact anchor/chart records. Relative reconstruction is ineligible
for this metric path and is never relabelled as metres. Curved charts are outside
this exporter's experimental admission.

The renderer remains a tangent-plane approximation. It does not claim curved
wrapping, independently verified surface identity, mesh occlusion or independent
visibility labels. The mask is an observation source, not a quality certificate.
The accounting separately reports scheduled/exported rows, accepted measurements,
proposed predictions, recovering, lost, unavailable, rendered and render-unavailable
counts. Render-unavailable includes accepted poses whose artwork was suppressed;
it is not another mutually exclusive pose state. Independent quality metrics remain
unavailable, not zero error.

## Integrity and compatibility

Input RGB, masks, metric bundle/mesh, calibration, saved producer closure,
admission record, anchor, project PNG and exporter source closure are authenticated.
The exporter includes the tracking policy source in its closure and rechecks input
closure before exclusive publication. Integrity failures refuse the export instead
of becoming normal tracking loss. Existing runs remain immutable. A changed producer
closure can require matching newly admitted saved runs; old reports must not be
rewritten to pretend to match new source. Synthetic matching fixtures demonstrate
software behavior only.

The integration fixture creates `.cache` itself, including on pristine checkouts.
Run the model-free checks with:

```
python -B -m unittest tools.test_workbench_offline_export tools.test_workbench_offline_export_integration tools.test_workbench_tracking_export
```

Exact-commit independent VM validation remains required before promotion. Neural
execution still requires the separate legacy-worker ownership and runtime gates.
