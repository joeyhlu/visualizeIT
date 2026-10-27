# Metric object identity at input boundaries

`MetricBundle.from_file` and `prepare_workbench_contract` require `obj_id` to be
an exact integer in **0..2³¹−1**. Zero remains valid. Booleans, floating-point
values (including `1.0`), numeric strings, null, negative values and values above
the bound are rejected rather than converted into another object identity.
Direct Python callers must also supply a built-in integer, rather than a NumPy
integer or integer subclass.

The bundle loader checks the ID before reading the GLB. Contract preparation
checks every case ID before model authentication and retains a shallow mapping
snapshot of each case, so later prerequisite callbacks cannot replace a checked
ID in the caller's mapping. Existing checks for geometry, source, weights,
segmentation, intrinsics and serving parity remain required. This does not add
new execution admission or change the direct `MetricBundle(...)` constructor.

Valid integer bundle fields, contract bodies and canonical contract hashes match
the previous implementation under identical prerequisites. The documented ID
range already existed in the portability audit; the ingress loaders now enforce
it consistently. Historical receipts remain immutable. A source change still
requires fresh source compatibility checks before real execution.

```sh
python -B -m unittest bench.test_workbench_object_identity tools.test_workbench_object_identity_independent bench.test_workbench_neural
```

Tests reproduce the old coercion against the actual pinned Git parent, check
valid boundaries and historical IDs, and verify that rejected IDs reach neither
GLB nor checkpoint authentication. Small manufactured prerequisites are explicitly
mocked for contract tests; they do not authenticate real checkpoints or establish
neural parity, tracking quality or performance. Independent bundle tests use a
tiny real GLB, and an observer test checks the retained object identity when the
caller's mapping changes during a prerequisite callback.
