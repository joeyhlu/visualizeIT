# Load the experimental GoTrack head in offline inference

The ordinary pose runner now accepts a separately authenticated inference
contract and the exported correspondence head. The default path retains the
existing frozen GoTrack model and legacy geometry. The new profile is opt-in,
cloud-only, and restricted to three existing synthetic development bundles.

Training completed a two-step systems smoke test; see
[the measured training results](gotrack-adaptation-results-2026-10-15.md).
The new CLI passed 76 focused local checks and an actual paired cloud replay:
six eight-frame runs plus two four-frame causal checks. Each head produced
21 tracking/visible frames out of 24 requested development frames; initialization
was suppressed in each sequence. Both causal prefix comparisons matched exactly.
These results establish serving integration, not improved rotations, surface
attachment, mobile support, or real-time performance. The detailed measurements
and limits are in the linked results report.

## Expanded checkpoint

A second opt-in contract, `bench/runtime/gotrack-inference-expanded-v1.json`,
accepts the 16-update correspondence-head export described in
[the expanded training report](gotrack-expanded-results-2026-10-15.md).
Its SHA-256 is
`1495f84ceb3bb6513d3fb76602f68b88bec7e538b20e088a394b29df9afe7287`.
The checkpoint SHA-256 is
`e23794bcf91f2b3de48d30c0a1fed9c088c6e0119d8d64b5c1b2718adc4e3b96`.
SAM, the visual backbone and the decoder remain frozen.

The expanded contract uses the same reviewed geometry and exact three development
bundles. It also accepts two authenticated historical full-video TRAIN mask caches;
their original producer headers are preserved. This exception does not apply to
the original checkpoint. All other producer fields must match exactly.

To select the expanded export in the reviewed cloud checkout:

```bash
python -B -m bench.quality_runner pose \
  --bundle "$BUNDLE" --masks "$MASKS" --output "$NEW_OUTPUT" \
  --device cpu --mode complete --ranker legacy \
  --unlit-templates --disable-multisampling \
  --gotrack-inference-config bench/runtime/gotrack-inference-expanded-v1.json \
  --gotrack-inference-config-sha256 1495f84ceb3bb6513d3fb76602f68b88bec7e538b20e088a394b29df9afe7287 \
  --gotrack-head /teamspace/studios/this_studio/visualizeit-private-adaptation-expanded-continuation-v1/run/head.pt \
  --gotrack-head-sha256 e23794bcf91f2b3de48d30c0a1fed9c088c6e0119d8d64b5c1b2718adc4e3b96 \
  --cloud-execution
```

Omitting `--limit` requests the complete physical video. Omit the two
`--gotrack-head` arguments to compare the frozen head under the same geometry and
runtime contract. Use an authenticated observed-mask cache and a fresh output path.
The required existing frozen-model smoke cache must already be available.

The immutable original contract still runs in its archived checkout. In the current
checkout, selecting that original contract additionally requires both
`--gotrack-runtime-compat-config bench/runtime/gotrack-inference-expanded-v1.json`
and `--gotrack-runtime-compat-config-sha256` with the expanded contract hash above.
This preserves the original checkpoint identity while authenticating the current
runtime. The compatibility flags do not permit the expanded checkpoint or its
additional historical mask producer under the original contract.

The expanded serving code passed 75 focused tests, including three added parser
boundary tests and exact legacy evaluation parity. Complete 24-frame comparisons
are in the [expanded comparison](gotrack-expanded-inference-results-2026-10-15.md)
and do not support promotion. The earlier eight-frame numbers describe the original two-step
checkpoint only. Neither result establishes real camera accuracy or phone speed.

## Inputs and compatibility

The profile `gotrack_integer_legacy_synthetic_v1` accepts these exact pinned
inputs, videos, meshes, and generation receipts:

- `ranker-multiview-hot3d17-v1-yaw0-pitchm25`
- `ranker-multiview-hot3d17-v1-yaw0-pitchp25`
- `ranker-multiview-hot3d27-v1-yaw0-pitchm25`

Their physical timestamps remain intact. The inference adapter applies the
documented native principal-point correction once, uses integer-center template
depth geometry, and preserves the ordinary floating RGB crop tensors. This
correction is specific to these historical synthetic renders. Other synthetic
raster conventions, real camera recordings, and capture bundles are rejected.

The original training source lock and export are immutable. The separate
`bench/runtime/gotrack-inference-v1.json` contract pins the reviewed source
replacements, unchanged upstream files, runtime dependencies, source18 observed
mask producer, and exact export identity. Editing a pinned source requires a
newly reviewed contract; a mismatch aborts before neural inference.

## Cloud invocation

Use the existing reviewed Linux environment with pre-existing model weights and
mask caches. These commands do not provision resources or download models.
The approved cache mount is currently specific to the existing Studio.

```bash
export VISUALIZEIT_CLOUD=1 PYOPENGL_PLATFORM=egl XFORMERS_DISABLED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SAM2_BUILD_CUDA=0

CONTRACT=bench/runtime/gotrack-inference-v1.json
CONTRACT_SHA=889104c72dd25b448d24e215f72efbc7e78064838e547ba5d2337f76cecc82d4
RUNTIME_CONTRACT=bench/runtime/gotrack-inference-expanded-v1.json
RUNTIME_CONTRACT_SHA=1495f84ceb3bb6513d3fb76602f68b88bec7e538b20e088a394b29df9afe7287
HEAD=/teamspace/studios/this_studio/visualizeit-private-adaptation-v1/run-v2/head.pt
HEAD_SHA=942b35b52536fe303a9701296b27e90009b88a89216b0883cf045cf221ed7b41

# Set BUNDLE and MASKS to one of the authenticated bundle/cache pairs.
# The coordinator prepares the required model smoke before pose inference.
python -B -m bench.quality_offline \
  --bundle "$BUNDLE" --masks "$MASKS" \
  --output "$NEW_OUTPUT" --stages smoke pose \
  --device cpu --limit 8 --ranker legacy \
  --unlit-templates --disable-multisampling \
  --gotrack-inference-config "$CONTRACT" \
  --gotrack-inference-config-sha256 "$CONTRACT_SHA" \
  --gotrack-runtime-compat-config "$RUNTIME_CONTRACT" \
  --gotrack-runtime-compat-config-sha256 "$RUNTIME_CONTRACT_SHA" \
  --gotrack-head "$HEAD" --gotrack-head-sha256 "$HEAD_SHA" \
  --cloud-execution
```

The paired frozen-head geometry control uses the same command with the two
`--gotrack-head` arguments omitted. The original legacy baseline omits all
GoTrack inference-contract/head arguments and `--cloud-execution`; retain its
original experiment settings. Those are distinct controls.

The coordinator runs the existing model-smoke prerequisite and generates a
separate pose filename and candidate trace for each profile/head/ranker identity.
The low-level `quality_runner pose` command requires its smoke cache to exist
already; a missing prerequisite stops before loading the pose network.
Use a new output directory
for every condition and prefix. Conflicting capture, appearance, motion-memory,
forced-occlusion, and PnP-ablation options are rejected for this profile.

The coordinator's stage receipt reads the installation identity from the child output;
a successful process exit without a valid inference output is a failure.

## Outputs and cleanup

The pose JSON and candidate trace include the profile, calibration convention,
source/config/checkpoint identities, mask joins, pose/render states, and timings.
Requested and installed head identities are separate, including on failure.
The adapter verifies the exact export bytes before safe deserialization and
loads a separate copy of the head before replacing the live module. A failed
installation preserves the frozen network and random state and aborts the run.

The runner closes renderers and releases model resources on exit. A bounded
experiment should also enforce a wall-clock and storage budget around the
process, preserve partial failures, and put the cloud machine to sleep afterward.
Stopping Python alone does not demonstrate that the provider allocation stopped.

Correspondence counts in candidate outputs are not PnP RANSAC inlier counts.
Unavailable internal solver counters must remain unknown. Development comparisons
must retain failed frames and report availability alongside any pose errors.

## Score saved predictions separately

The evaluator reads the sealed prediction report before opening separate
reference sidecars. It reuses the existing pose-error functions and performs no
model inference, training, rendering or network access. References are evaluation
inputs only; they are never fed back into the recorded pipeline.

For the existing three development sequences, preserve each original
`generation.json` and `poses.jsonl` pair in its sequence directory under the
reference root. These six files total 36,441 bytes; their hashes join the
generation identities in the sealed prediction report. Heavy model and image
files remain in cloud storage.

```powershell
New-Item -ItemType Directory -Force .cache/adapted-pose-evaluation-v1 | Out-Null
$poseReport = ".cache/adapted-pose-evaluation-v1/report-" + [guid]::NewGuid().ToString("N") + ".json"
& 'C:/Users/Achita/.codex/tools/science/Scripts/python.exe' -B -m bench.quality_postseal_pose_evaluate `
  --sealed-root .cache/adapted-paired-continuation-v1 `
  --sealed-manifest-sha256 c0564566222712bc8e322d0eef9fa051b31ac1d2ea8e8505fc6e847a68dcfe69 `
  --reference-root .cache/adapted-pose-evaluation-inputs-v1 `
  --diagnostic .cache/cloud-prepare/adaptation-label-diagnostic-v1.json `
  --output $poseReport
```

The example selects a fresh output filename so earlier reports are preserved.
Evaluation reports translation error for
emitted poses, candidate coverage and counterfactual minimum candidate errors,
paired head differences, and consecutive translation-increment residuals.
The last metric compares predicted translation changes with reference changes;
it is not surface attachment error. Suppressed poses remain unavailable even
when a candidate was selected.

Only frames 0â€“7 in each eight-frame run contribute quality samples. Four-frame
causal repeats and unprocessed frames 8â€“23 do not enlarge the sample. Original
orientation ambiguity is retained: primary rotation and severe-rotation metrics
are unmeasured with an eligible denominator of zero. These synthetic development
references do not establish independent real-world attachment accuracy.
