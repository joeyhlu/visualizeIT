# Fresh-machine artifact provenance for finite recorded inference

The default workbench profile is unchanged: it authenticates the exact historical
`flow-terminal-small-records-20261021.json` bundle before installing the primary
head or earning admission. A fresh measurement cannot truthfully reproduce that
historical bundle, backup, or loader receipt.

An **explicit opt-in** instead creates `workbench_fresh_recorded_artifact_v1`
evidence. This is a new local lineage, **not** a replacement historical receipt.
It supports finite registered recorded clips only. It does not admit live-camera
serving, automatic object discovery, unknown-scale reconstruction, or a model
quality claim. The native live factory refuses this scope independently.

## What is authenticated

`gate artifact` hashes the actual local files and safely reads authenticated
checkpoint metadata. It imports Torch but constructs/runs no model:

- The already-pinned public runtime JSON and its exact primary identity.
- Exact byte size and SHA-256 of the base, adapted export, and reference best
  checkpoint; exact installer and historical source-lock hashes.
- The exact training configuration **identity** crossbound in the pinned runtime
  JSON and in both safely deserialized, exact-hash export/reference checkpoints.
  Source-lock/dataset identities and export base/deployment/quality fields must
  also agree. Each checkpoint is byte-authenticated **before**
  `torch.load(weights_only=True, map_location="cpu")`; no unrestricted fallback
  or added safe-global allowlist exists.
- The caller must explicitly choose either a configuration JSON whose canonical
  JSON identity matches the pin, or `--training-config-metadata-only` when the raw
  dictionary is unavailable. Metadata-only proof truthfully says
  `training_config_verification=identity_crossbound_in_authenticated_export_and_reference_metadata`
  (within `training_configuration.verification`) and `raw_config_available=false`.
  It does not synthesize a dictionary, verify unavailable raw bytes, or certify a
  historical training run. If a raw dictionary is supplied, its canonical identity
  and current presentation bytes are both bound, in addition to checkpoint metadata.
- The complete current/pinned GoTrack source closure, only the existing exact
  reviewed replacements, their pinned review evidence, and the current
  workbench source hashes. Missing or unreviewed sources block the proof.
- The current workbench runtime package inventory and repository location.

The self-hashed proof says `historical_small_records_verified: false`,
`historical_receipts_verified: false`, `execution_admission_granted: false`,
`live_eligible: false`, and `quality_claim: false`. Self-hashing alone grants
nothing: all bindings are re-derived on each use. A relocated checkout or changed
file/package/source requires a new proof and subsequently new admission evidence.
Local proof/contract files may contain local paths; do not publish private model
files, private package metadata, or unreviewed local paths.

## Opt-in commands

Use the normal local workbench environment and separately prepare truthful
ClipSpec, reviewed RGB prompt selection, predicted SAM2 mask set, metric CAD GLB,
and a single authorized onboarding pose seed. Never import held-out scoring masks
or poses into those inference resources. Config/source/model restoration is
operator-controlled; this command does not download or train anything.

```sh
python -m bench.workbench_cli gate artifact \
  --head HEAD.pt --reference BEST.pt --training-config-metadata-only \
  --out LOCAL-PROOF.json

python -m bench.workbench_cli gate contract \
  --cases CASES.json --head HEAD.pt --reference BEST.pt \
  --recorded-artifact LOCAL-PROOF.json --out ADMISSION/contract.json

python -m bench.workbench_cli gate geometry --production \
  --out ADMISSION/geometry.json

python -m bench.workbench_cli gate parity \
  --contract ADMISSION/contract.json --cases CASES.json \
  --head HEAD.pt --reference BEST.pt --device cpu \
  --recorded-artifact LOCAL-PROOF.json --out ADMISSION/parity

python -m bench.workbench_cli gate verify --admission ADMISSION \
  --recorded-artifact LOCAL-PROOF.json

python -m bench.workbench_cli --workspace WORKSPACE run CLIP-ID \
  --mode focused --first-frame f000010 --count 20 \
  --roles primary_neural,frozen_baseline --masks PREDICTED-SAM2-MASK-SET \
  --metric-bundle METRIC-BUNDLE.json --head HEAD.pt --device cpu \
  --admission ADMISSION --recorded-artifact LOCAL-PROOF.json
```

If the exact raw training configuration is available, replace the explicit
`--training-config-metadata-only` option with `--training-config MATCHING-CONFIG.json`.
These alternatives are mutually exclusive; neither is an implicit fallback.

Create `ADMISSION` before these commands; outputs are new files/directories and
are not overwritten. Replace the example frame ID with the frame immediately
after the declared seed (or permit normal causal warm-up from that frame).
Heavy SAM, production geometry, parity, and pose stages should run serially on a
resource-bounded machine. See the workbench documentation for normal `segment`
and `gate case` commands. The ordinary model loaders and numerical behavior are
unchanged by this provenance repair.

The proof must be explicitly supplied to contract preparation, parity, admission
verification, and recorded execution. It must exactly match the proof embedded in
the contract. Omitting it does not silently select the new mode. Historical and
fresh-recorded install/admission lineages cannot be mixed.

## Gates remain measured, not asserted

The artifact proof cannot set `admitted=true`. Existing manufactured-plane/cube
geometry and wrong-control rejection, production renderer/crop measurements,
full RGB/backbone/decoder/head reference-versus-export parity, stored-array
recomparison, source/runtime/input revalidation, and temporal session checks all
remain required. Fresh recorded mode additionally requires full base and source
verification; `--skip-base-hash` is not permitted for it. Invalid/missing evidence
remains blocked. Frozen baseline is a separate comparator, never a fallback.

Once admitted, the normal backend starts from the single onboarding seed, uses
its previous validated pose sequentially, and reports lost/unavailable honestly.
It does not silently reseed after loss. Predicted segmentation and initialization
are still meaningful limitations: passing compatibility gates is not a tracking
accuracy result, end-to-end detector score, phone validation, or live FPS claim.

## Tests

`python -m unittest bench.test_workbench_recorded_artifact -v` uses tiny temporary
files, fixture-only patched pins and mocked checkpoint metadata, never model
execution or downloads. It checks strict
historical default, exact file/config/source/runtime identities, missing/size/
symlink/tamper/drift rejection, scope/history-claim tampering even after rehash,
explicit opt-in and mode matching, metadata mismatches and false raw-config
claims, strict weights-only/authentication ordering, no admission without measured parity, mixed
session lineage refusal, and explicit native live rejection. Production pins and
the atomic installer are not changed by these tests or by the opt-in path.
