# Opt-in refiner observations for the offline 15 FPS investigation

The P0 target is at least 15 **processed** recorded-video frames per second for
trained tracking plus artwork overlay. The corresponding serial frame budget is
66.667 ms. Encoded playback FPS, cached replay and quick failed frames do not
establish this target. No speedup or current model-throughput measurement is
provided by this instrumentation.

`WorkbenchIntegerRefiner` accepts an optional `profiler`. Its default is `None`:
no profiling clock or synchronization callback is called. Existing trace keys,
arrays, solver parameters, network argument expressions and numerical operations
are preserved. The backend still uses five refinement iterations by default.
Original frame-result and cache schemas are unchanged.

`bench.workbench_refiner_timing.RefinerStageProfiler` collects bounded,
out-of-band observations in memory. Its seven disjoint phase names are:

| Phase | Included work |
|---|---|
| `preparation` | Runtime imports, renderer lookup/first construction, camera, pose and RGB/mask tensor preparation. |
| `crop_template` | Pose-dependent crop/template construction and crop intrinsics. |
| `network_with_input_transfers` | The unchanged network call, including its argument transfers. |
| `flow_conf_to_host` | Flow/confidence conversion and host copies. |
| `correspondences` | Correspondences and retained object/image point arrays. |
| `pnp_solver` | Existing seeded PnP and refinement, or its insufficient-point path. |
| `diagnostic_materialization` | Existing returned diagnostic arrays and result dictionary. |

Network execution and input transfers remain combined to avoid changing
temporary tensor lifetimes. Preparation can include first-use initialization;
separate cold and steady observations. The optional whole-trace interval includes
the phases and must not be added to their durations. These observations omit
earlier SAM production, outer frame decode/mask reads, final artwork composition
and video encoding. They are not complete frame latency.

Without a synchronization callback, the mode explicitly states
`host_unsynchronized_NOT_GPU_latency`. Those intervals can measure host work or
asynchronous dispatch; they do not prove completed GPU execution. A caller can
provide an explicit synchronization callback and a declared device. Such records
are marked as perturbed, synchronized-by-callback host measurements. The helper
does not verify a device or import a neural runtime. Synchronization cost is
included at phase exit; phase-entry synchronization precedes the phase clock.
Profiled and normal-production throughput must therefore be measured separately.

For a future **admitted** benchmark, an operator can create a profiler and pass
this factory through the existing backend `refiner_factory` hook:

```python
profiler = RefinerStageProfiler(max_traces=256, context={"workload": "pinned-run"})
refiner_factory = lambda network, device: WorkbenchIntegerRefiner(
    network, device, profiler=profiler
)
```

This example constructs no model. Benchmark orchestration must still verify
runtime, weights, device, inputs and current-source admission, and satisfy the
legacy-job ownership gate before execution. This source change does not renew
old admission receipts. Record both the neural module and profiler helper source
hashes, runtime versions, workload identities and synchronization policy with
future measurements.

Call `profiler.snapshot()` for a detached JSON-safe record. Durations are seconds.
Trace ordinals and supplied RNG seeds are recorded; frame identities are not
inferred. Caller context is declared provenance, not verified authority. Correlate
observations with actual frame and iteration outcomes in the benchmark ledger.
Missing/failed stages remain explicit rather than becoming zero-cost successes.
Clock, callback and runtime exceptions propagate.

The helper retains at most `max_traces` records, with bounded metadata and device
text. At capacity, it increments `dropped_traces` and observes no further clocks
or synchronization calls; tracking work still proceeds. A full profile must
account for these drops. It writes no files and is intended for sequential use
by one refiner.

Software checks use logical clocks and a stub runtime. They verify stage order,
exception/capacity handling, and default-versus-profiled numerical/call/RNG parity.
Their execution time is not model performance:

```text
python -B -m unittest bench.test_workbench_refiner_timing bench.test_workbench_refiner_timing_independent bench.test_workbench_neural
```
