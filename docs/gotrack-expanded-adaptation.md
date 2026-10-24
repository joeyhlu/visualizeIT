# Expanded GoTrack development experiment

This extends the three-crop, two-step correspondence-head smoke to the first
eight frames of three existing synthetic sequences. It is a small development
experiment with dependent video frames, one training object and one previously
inspected validation object. It cannot establish unseen-object accuracy or a
solution to ambiguous rotations. SAM remains frozen.

| Split | Object | Views | Requested crops |
| --- | --- | --- | --- |
| Training | HOT3D 17 | Pitch −25° and +25° | 16 |
| Used validation | HOT3D 27 | Pitch −25° | 8 |
| Held-out test | HOT3D 7 | Unopened | No payload access |

Each frame starts with its own estimated FoundPose seed. This samples
initialization, rather than causal tracking. The first actual GoTrack refinement
crop is retained with its original floating RGB, camera transform and calibration.
Cached automatic SAM masks are observed foreground; templates do not replace them.

## Runtime and data separation

The additive controller runs in a copy of the authenticated historical training
runtime, whose source lock is
`44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33`.
Existing modules, the original smoke and previous checkpoints remain unchanged.
The controller and its configuration have a separate capture contract. The
current inference checkout has different source hashes; use the prepared cloud
runtime instead of weakening verification to run training from that checkout.

The stages run sequentially in separate processes:

1. Capture all 24 input crops using estimated seeds and matching frame timestamps.
2. Run the unchanged frozen feature extractor.
3. Authenticate every crop and feature, and check parity against actual inference.
4. Open the separate synthetic label recipe and render query supervision.
5. Train only the GoTrack DPT correspondence head for one epoch, up to 16 updates.
6. Check interruption/resume, export, and fresh-network loading.

Missing crops or features remain in the inventories and stop this fixed
experiment before optimization. The backbone and decoder stay frozen. Test
payloads are inaccessible to the training dataset loader.

## Launch and monitor

The corrected cloud source is prepared as `/teamspace/studios/this_studio/visualizeit-cloud-v21`.
The original source-v20 and its failed run are preserved. The command below is
for a new experiment output; it must not be used to restart the preserved trial.
No models, datasets or environment packages are installed by this command.

```bash
cd /teamspace/studios/this_studio/visualizeit-cloud-v21
export VISUALIZEIT_CLOUD=1 PYOPENGL_PLATFORM=egl XFORMERS_DISABLED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SAM2_BUILD_CUDA=0
python -B -m bench.quality_adaptation_expanded run \
  --plan .cache/cloud-prepare/gotrack-expanded-plan-v2.json \
  --label-plan .cache/cloud-prepare/gotrack-expanded-label-plan-v1.json \
  --config bench/runtime/adaptation-expanded-cpu-v1.json \
  --output /teamspace/studios/this_studio/visualizeit-private-adaptation-expanded-new-run \
  --cloud-execution
```

The output must be new and outside the source directory. This run has a
30-minute aggregate limit, a 1.5 GiB private output limit and a 1 GiB free-disk
reserve. It uses CPU because that is the existing free cloud machine; it is not
a GPU performance measurement. Paid GPU resources require separate approval.

Read `stages.json`, `logs/*.receipt.json` and `logs/*.log` for preparation progress.
During optimization, read `run/status.json` and `run/preflight.json`; the final
verification is `run/expanded-verify.json`. Model files and feature archives stay
in cloud storage. Download only the small reports.

The normal workflow deliberately stops after update 8 and resumes through the
same trainer. For an unexpected interruption, preserve the run directory and
check its saved configuration, status and remaining experiment budget before
using the controller's `resume` command. Resuming a training checkpoint does not
rerun missing preparation stages. A standalone `resume` restores the trainer's
elapsed time; it does **not** restore the outer workflow's deadline. For manual
continuation, the operator must subtract the saved total preparation and training
time from the original 1,800 seconds and supervise all remaining stages under
that remaining allowance, including cleanup. The planned update-8 interruption
and resume within `run` already share one outer watchdog.

After collecting the reports, check that the controller and its model workers
have exited, then use the Studio's **Sleep** action. Verify that the Studios
table says **sleeping** and shows no allocated machine. A Python process ending
does not establish that cloud compute or billing stopped. Keep paid upgrades,
automatic top-ups and additional chargeable storage disabled unless approved.

## Loading and interpreting the checkpoint

Fresh-network verification compares exported and attached heads on cached
validation tokens and then compares full RGB inference with those predictions.
It also checks that frozen weights remain unchanged and that an incorrect
checkpoint hash is rejected without modifying the network.

The separately reviewed expanded inference contract accepts this exact expanded
export through the ordinary offline CLI. See the
[loading instructions](gotrack-adapted-inference.md) and
[full-video comparison](gotrack-expanded-inference-results-2026-10-15.md).
A different export or dataset/configuration identity needs a separately reviewed
contract. Export verification alone does not promote a head over the baseline.

## Execution status

The first expanded attempt captured all 24 requested inputs in 310.08 seconds.
Feature extraction then failed before loading the neural extractor: writing the
saved plan changed its JSON formatting, which violated its exact byte hash.
The plan's contents were unchanged. No labels or optimizer steps ran in that
attempt. Its complete captures, source and failure logs are preserved.

The repair copies the authenticated plan bytes without reformatting. A bounded
continuation reuses the same 24 crop and geometry archives, with their original
hashes, and subtracts the prior 312.802 seconds from the original 1,800-second
allowance. Storage accounting includes both attempts. Before further execution,
checkpoint and training-output caps were lowered to 320 MiB and 1,000 MiB;
learning settings, seeds and data selection are unchanged. This is an amendment
to resource limits, not another learning experiment.

The continuation completed all 16 optimizer updates, resumed from update 8,
exported its head, and passed fresh-network loading/RGB parity checks. All
24 requested crops and features were available. See
[the measured results](gotrack-expanded-results-2026-10-15.md) for the small
development-validation effect and its limitations. The Studio is sleeping with
no allocated machine. Large artifacts remain in cloud storage.
