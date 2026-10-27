"""Opt-in, bounded host stage observations; never imports a neural runtime.

Synchronizing callbacks perturb execution. Unsynchronized durations measure host
work and dispatch, and must not be interpreted as completed GPU latency.
"""
from contextlib import contextmanager, nullcontext
import json
import math
from time import perf_counter


PHASES = (
    "preparation", "crop_template", "network_with_input_transfers",
    "flow_conf_to_host", "correspondences", "pnp_solver",
    "diagnostic_materialization",
)
MAX_METADATA_BYTES = 16384
MAX_DEVICE_BYTES = 256


def _detached(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _bounded_metadata(value):
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("context and trace metadata must fit within 16384 UTF-8 bytes")
    return json.loads(encoded)


class _DroppedTrace:
    def stage(self, name):
        return nullcontext()


class _Trace:
    def __init__(self, profiler, row):
        self.profiler = profiler
        self.row = row
        self.active = False
        self.closed = False

    @contextmanager
    def stage(self, name):
        if self.closed or self.active:
            raise ValueError("stages require an open trace and cannot overlap")
        if name not in PHASES or any(p["name"] == name for p in self.row["phases"]):
            raise ValueError("stage must be a known, unrepeated phase")
        phase = {"name": name, "outcome": "failed", "elapsed_seconds": None}
        self.row["phases"].append(phase)
        self.active = True
        try:
            self.profiler._sync()
            start = self.profiler._now()
            try:
                yield
            finally:
                self.profiler._sync()
                phase["elapsed_seconds"] = self.profiler._elapsed(start)
            phase["outcome"] = "completed"
        except BaseException as exc:
            phase["error"] = type(exc).__name__
            raise
        finally:
            self.active = False


class RefinerStageProfiler:
    """Keep at most ``max_traces`` JSON-safe observations in memory.

    ``device`` is a caller declaration, not a verified runtime property. No
    implicit synchronization occurs. Once capacity is reached, dropped traces
    perform no clock or synchronization calls. Exceptions always propagate.
    The optional inclusive elapsed time overlaps phases and is not additive.
    Instances are intended for sequential use by one refiner.
    """
    def __init__(self, *, max_traces=256, clock=perf_counter, synchronize=None,
                 device=None, context=None, include_total=False):
        if type(max_traces) is not int or not 1 <= max_traces <= 10000:
            raise ValueError("max_traces must be an integer in [1, 10000]")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable or None")
        if device is not None and (not isinstance(device, str) or not device.strip()
                                   or len(device.encode("utf-8")) > MAX_DEVICE_BYTES):
            raise ValueError("device must be None or a nonempty string of at most 256 UTF-8 bytes")
        if synchronize is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("synchronized profiling requires a declared device")
        if type(include_total) is not bool:
            raise TypeError("include_total must be bool")
        self.max_traces = max_traces
        self.clock = clock
        self.synchronize = synchronize
        self.device = _detached(device)
        self.context = _bounded_metadata(context)
        self.include_total = include_total
        self._traces = []
        self._dropped = 0

    def _sync(self):
        if self.synchronize is not None:
            self.synchronize()

    def _now(self):
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("clock must return a finite real number")
        return value

    def _elapsed(self, start):
        elapsed = self._now() - start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("clock elapsed time must be finite and nonnegative")
        return elapsed

    @contextmanager
    def trace(self, metadata=None):
        if len(self._traces) >= self.max_traces:
            self._dropped += 1
            yield _DroppedTrace()
            return
        row = {"ordinal": len(self._traces), "metadata": _bounded_metadata(metadata),
               "outcome": "failed", "phases": []}
        self._traces.append(row)
        handle = _Trace(self, row)
        if self.include_total:
            row["elapsed_seconds"] = None
        try:
            start = self._now() if self.include_total else None
            try:
                yield handle
            finally:
                if start is not None:
                    row["elapsed_seconds"] = self._elapsed(start)
            row["outcome"] = "completed"
        except BaseException as exc:
            row["error"] = type(exc).__name__
            raise
        finally:
            handle.closed = True

    def snapshot(self):
        return _detached({
            "schema": "workbench_refiner_stage_timing_v1",
            "mode": ("host_synchronized_by_callback" if self.synchronize is not None
                     else "host_unsynchronized_NOT_GPU_latency"),
            "perturbation": self.synchronize is not None,
            "synchronization_overhead_included": self.synchronize is not None,
            "device": self.device, "context": self.context,
            "max_traces": self.max_traces, "dropped_traces": self._dropped,
            "traces": self._traces,
        })
