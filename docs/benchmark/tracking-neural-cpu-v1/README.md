# Actual CPU GoTrack inference diagnostic — 2026-10-26

This documents the initial model-only diagnostic, not the later causal SAM
pilot. A subsequent explicit [fresh-machine recorded-only verification
path](../../tracking-recorded-artifact.md) permits newly measured normal
geometry/parity/admission checks without claiming historical receipts. The
strict historical default and live-serving restrictions remain unchanged.

**Executed, not mocked:** the frozen GoTrack backbone/decoder, saved reference
head, exported trained head and upstream baseline all ran on this VM's CPU.
The trained export was installed through the unchanged atomic installer.
No training, CUDA use, checkpoint selection or serving-admission change occurred.

The supervised process exited **0** in **147.77 seconds**, with **2.93 GiB**
peak sampled process-group RSS (3,147,182,080 bytes, sampled every 250 ms).
Bounds were 900 seconds, 7 GiB RSS, 512 MiB output and 10 GiB free-disk reserve.
Torch/OpenBLAS/OpenCV used one thread, with software EGL/llvmpipe rendering.
Every checkpoint was authenticated and safely deserialized with
`weights_only=True`; there was no unsafe-pickle fallback.

## Measured results

| Variant | Full network forwards | Returned refinement candidates | Instrumented refinement time per call |
|---|---:|---:|---:|
| Saved `best.pt` head reference | 10 | 2 / 2 | 21.07–22.43 s |
| Installed trained export `27e06a51…` | 10 | 2 / 2 | 20.74–21.41 s |
| Frozen upstream baseline | 10 | 2 / 2 | 20.65–22.58 s |

Each call performed all five ordinary refinement iterations: **30 actual
RGB→DINOv2→decoder→DPT forwards**, with corresponding module hooks confirming
10 backbone, 10 decoder and 10 pose-head calls per variant. There were no model,
renderer or solver exceptions.

Reference versus installed export agreed **exactly**: every captured input,
flow/confidence tensor and final pose matched, with maximum absolute difference
**0.0** (predeclared tolerance was 1e-6). Their installed tensor-state SHA-256
also matched. On byte-identical first-iteration inputs, the trained head changed
maximum flow by **0.23952 px / 0.19660 px** relative to the upstream baseline;
confidence was unchanged. A changed flow is not evidence of improved accuracy.

## Scope: model/refinement check, not the tracking leaderboard

These are **two unique authorized onboarding images**, HB object 20, frames 8
and 9, initialized from known preceding onboarding poses 7 and 8 and supplied
onboarding visible-mask annotations. They contribute **zero additional scored
tracking frames**. All variants receive identical independent seeds. This does
not test automatic segmentation, causal tracking, reacquisition, held-out
accuracy or physical attachment. The legacy refinement geometry here is not
`workbench_integer_metric_v1` admission.

The elapsed times include instrumentation/capture and are observations from one
bounded diagnostic, not a repeated neural speed benchmark, end-to-end tracking
FPS or a claim about phone/GPU performance. Full workbench admission remains
separate: its exact historical small-records file is still required, followed
by normal contract/geometry/parity/admission stages. Missing historical backup
and loader receipts are preflight diagnostics, not additional hard admission
gates.

## Evidence and reproduction

- `summary.json`: compact reviewed measurements, artifact/source/input hashes,
  installed-head provenance, runtime versions, comparisons and supervisor result.
- `capture-hash-index.json`: hashes, dimensions and dtypes for all 30 local
  forward captures (14,238,271 bytes); raw tensors and private weights are not
  published.
- [Reproduction and exact restoration](../../tracking-neural-diagnostic.md)
  describes the isolated worktree, supported research APIs, safe loader,
  dependency lock, public upstream licence and bounded execution command.

The two model heads and base retain their documented experimental/noncommercial
research limits; successful CPU execution does not promote the trained head.
