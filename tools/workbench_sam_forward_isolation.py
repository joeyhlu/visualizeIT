"""Prospective SAM 2.1 state-isolation ablation for the original cracker-box prompt.

Question: does same-state reverse-prefix memory influence the original forward
predictions? Production ``bench/workbench_neural.py::segment_prompted`` prompts
local frame 9, propagates in reverse 9->0 and then forward 9->119 in ONE SAM
inference state, so the non-conditioning prefix memories feed the forward pass.
This ablation changes ONLY the state history and order. The prompt, weights,
config and settings stay fixed, and the cause of the original mask
fragmentation remains UNPROVEN either way.

State-isolation contract:

1. Both states keep ``num_frames = 120`` and the original indices (prompt =
   local 9). Shortening the reverse state would change SAM 2's object-pointer
   temporal normalisation (``max_obj_ptrs_in_encoder = min(num_frames, ...)``),
   so it is forbidden.
2. FORWARD FIRST: a fresh causal reader (count 120, released 9) and a fresh
   ``init_state``. ``init_state`` warms local 0, the opening RGB (recorded). The
   original prompt goes on local 9, and one forward propagation yields exactly
   9..119. Each next RGB is released only after the previous mask is emitted.
   There is no reverse call on this state.
3. The forward state, iterator and reader are destroyed. A fresh SEPARATE
   reader (count 120, released 9) and a fresh state get the same prompt. One
   reverse propagation yields 9..0, of which only 0..8 are kept. The duplicate
   local 9 must binary-equal the forward seed (and the original seed). Nothing
   above 9 is ever read, and there is never a forward pass on this state.
4. Masks are combined by authenticated local index and frame ID: exactly 120
   unique masks, with the seed and all 110 scored masks from the forward state
   only. Expected counters: 111 forward + 10 reverse yields, 120 writes, 2
   prompt adds, 2 fresh state initialisations and 2 teardowns.

Commands (the parent executes; the implementer runs neither):

``prepare`` (CPU only; no torch/SAM import) writes a fresh immutable plan under
``<workspace>/derived/sam-forward-isolation/plans/<plan-id>/plan.json``. It pins
the source/runtime closure, the SAM checkpoint/config/Python/YAML closure, the
clip, the original mask manifest, the original selection and seed mask, and all
120 RGB file and decoded hashes.

``run --plan PATH --expected-plan-sha256 HEX`` reconstructs and compares the
full plan before any GPU work. It then writes to the plan's fresh private
staging directory under ONE ``gpu_stage``, re-checks the closure afterwards and
writes a terminal receipt only after every identity/closure/budget check.
Outputs are NEVER registered as a workspace mask set. There is no ``--force``.

The frozen qualification is descriptive consistency, NOT independent accuracy.
Qualitative review is the parent's separate, mandatory step, and the receipt
never claims it was done. Success or failure does not prove original foreground
accuracy or any head improvement.

Pure, SAM-free API for tests: ``run_state_isolated``, ``qualify``, ``iou``,
``largest_component_fraction``, ``encode_mask_png``, ``decode_mask_png``,
``OutputWriter``, ``canonical_bytes``, ``validate_plan_shape``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN_SCHEMA = "workbench_sam_forward_isolation_plan_v1"
RECEIPT_SCHEMA = "workbench_sam_forward_isolation_receipt_v1"
MASK_SCHEMA = "workbench_observed_masks_v1"
PRODUCER_KIND = "sam2.1_video_prompted_state_isolated_forward_first"
EXPERIMENT_DIR = "sam-forward-isolation"

# ---- frozen experiment identity (original clip, prompt, masks; no new selection)
CLIP_ID = "dev-ycb-003-cracker-box"
CLIP_SHA256 = "afbb18691fef64ee595c1c22e603f1999ea9b19f5e0c740b47db811fb2b646c4"
COUNT = 120
FIRST_SOURCE_INDEX = 361
PROMPT_LOCAL = 9
PROMPT_FRAME_ID = "s000370"
SCORED_LOCAL = (10, 119)          # inclusive: s000371..s000480 (110 frames)
ORIGINAL_MASK_SET = "sam2-workbench-01"
ORIGINAL_MANIFEST_SHA256 = "f8fc14a551ce1792439b04f1c2afe40134eb3e107df1ea36f982a15be1c81fe3"
ORIGINAL_SEED = {"path": "m00009.png", "file_sha256": "1c8585a8829e48cf06c16540ed825034a01f0f8188d5ff8796b32769eacc7a9a",
                 "content_sha256": "7f6bcc144178bed955c3e05a694367c2dd94af8e71e40d8cd5736065431cffa9", "area": 18329,
                 "largest_component_fraction_recorded": 0.980359}
SELECTION_FILE = ".cache/video-workbench/derived/validation/ycb-reviewed-selections-v1.json"
SELECTION_FILE_SHA256 = "bd601c76a329452d4ce5f9ac71cfd4f0fa81c8030b028cba74516cf00df36623"
SELECTION_ID = "4efc716fa7744813b1973aec99e7b8d9"
POSITIVE = ((414.0, 110.0), (462.0, 213.0), (545.0, 88.0))
NEGATIVE = ((520.0, 165.0), (430.0, 290.0), (350.0, 210.0))
POINT_CONVENTION = "integer pixel centres + 0.5 -> SAM corner-origin coordinates"
SAM2_CHECKPOINT_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
SAM_SETTINGS = {"apply_postprocessing": False, "vos_optimized": False, "offload_video_to_cpu": True,
                "offload_state_to_cpu": True, "async_loading_frames": False, "obj_id": 1,
                "deterministic_position_grid": "bench.quality_sam2.deterministic_position_grid",
                "inference_mode": True, "mask_threshold": "logits[0, 0] > 0", "seed_or_precision_changes": None}
BENCH_SOURCES = ("bench/__init__.py", "bench/vision.py", "bench/workbench_neural.py", "bench/workbench_cli.py",
                 "bench/workbench_contract.py", "bench/workbench_geometry.py", "bench/workbench_telemetry.py",
                 "bench/quality_sam2.py", "bench/quality_assets.py")
PACKAGES = ("torch", "numpy", "pillow", "opencv-python", "opencv-python-headless", "hydra-core", "omegaconf",
            "iopath", "tqdm")
PREVIEW_LOCAL = (0, 9, 10, 25, 45, 65, 85, 105, 119)
PREVIEW_SIZE = (960, 810)
THRESHOLDS = {"seed_largest_component_min": 0.95, "seed_area_exact": 18329,
              "scored_area_min_fraction_of_candidate_seed": 0.5, "scored_consecutive_iou_min": 0.5,
              "zero_union_iou": 0.0, "connectivity": 8}
EXPECTED_COUNTERS = {"forward_yields": COUNT - PROMPT_LOCAL, "reverse_yields": PROMPT_LOCAL + 1, "writes": COUNT,
                     "prompt_adds": 2, "state_inits": 2, "teardowns": 2}
OUTPUT_CAP_BYTES = 8 << 20
FLAGS = {"human_accuracy": False, "independent_accuracy": False, "phone": False, "trained_head_test": False,
         "development_only": True, "qualitative_review_done": False, "registered_mask_set": False, "promotion": False}


class IsolationError(RuntimeError):
    """The state-isolation contract, a plan or an output check failed (fail closed)."""


# --------------------------------------------------------------------------
# small pure helpers


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def loads_strict(raw: bytes, what: str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise IsolationError(f"{what}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise IsolationError(f"{what}: non-finite JSON value {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolationError(f"{what}: invalid JSON ({exc})") from None


def frame_ids_for(count: int = COUNT, first: int = FIRST_SOURCE_INDEX) -> list[str]:
    return [f"s{first + i:06d}" for i in range(count)]


def _bool_mask(mask: Any, shape: tuple[int, int] | None = None) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2 or (shape is not None and m.shape != tuple(shape)):
        raise IsolationError(f"mask must be 2-D{'' if shape is None else ' with shape ' + str(tuple(shape))}")
    if m.dtype != np.bool_:
        raise IsolationError("mask must be boolean")
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU; the zero-union convention is 0.0 (never a pass)."""
    a, b = _bool_mask(a), _bool_mask(b)
    union = int(np.count_nonzero(a | b))
    return 0.0 if union == 0 else float(np.count_nonzero(a & b)) / union


def largest_component_fraction(mask: np.ndarray, connectivity: int = 8) -> float:
    from bench.vision import cv2
    m = _bool_mask(mask)
    area = int(m.sum())
    if area == 0:
        return 0.0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=connectivity)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / area if n > 1 else 0.0


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Same encoding as ``workbench_cli.run_segmentation`` (0/255 single-channel PNG)."""
    from bench.vision import cv2
    ok, buf = cv2.imencode(".png", _bool_mask(mask).astype(np.uint8) * 255)
    if not ok:
        raise IsolationError("mask PNG encoding failed")
    return buf.tobytes()


def decode_mask_png(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Strict binary decode: 8-bit single channel, exact shape, values only 0/255."""
    from bench.vision import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2 or img.dtype != np.uint8 or img.shape != tuple(shape):
        raise IsolationError("mask PNG is not an 8-bit single-channel image of the expected size")
    values = np.unique(img)
    if not set(values.tolist()) <= {0, 255}:
        raise IsolationError("mask PNG is not strictly binary (0/255)")
    return img == 255


def mask_content_sha256(mask: np.ndarray) -> str:
    from bench import workbench_geometry as wg
    return wg.mask_sha256(_bool_mask(mask))


# --------------------------------------------------------------------------
# pure state-isolation engine (adapter-driven; SAM-free and testable with fakes)


def _plain(value: Any) -> Any:
    """JSON-safe copy of small adapter return values (never mask arrays)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"ndarray": list(value.shape), "dtype": str(value.dtype)}
    return repr(value)[:200]


class IsolationLedger:
    """Caller-owned progress ledger. The engine records every reader allocation,
    state initialisation, read, release, yield, write and teardown attempt here
    *as it happens*, so a failure at any point keeps real non-null evidence.
    ``snapshot()`` is JSON-safe and never contains mask arrays."""

    BRANCH_TEMPLATE = {"reader_allocated": False, "reader_length": None, "state_init_attempted": False,
                       "state_init_completed": False, "prompt_added": False, "yields": [],
                       "teardown": {"attempted": False, "completed": False, "steps": {}, "errors": []},
                       "primary_error": None}

    def __init__(self, frame_ids: Sequence[str] = ()):
        self.frame_ids = list(frame_ids)
        self.counters = {k: 0 for k in EXPECTED_COUNTERS}
        self.events: list[dict] = []
        self.reads: dict[str, list[int]] = {}
        self.released: dict[str, int] = {}
        self.branches: dict[str, dict] = {}
        self.written: dict[int, dict] = {}
        self.cleanup_errors: list[str] = []
        self.outer: dict[str, Any] = {}

    # causal guard (independent of the adapter's own frame guard)
    def open(self, label: str, released: int) -> dict:
        if label in self.branches:
            raise IsolationError(f"state label {label} opened twice")
        self.released[label], self.reads[label] = released, []
        self.branches[label] = json.loads(json.dumps(self.BRANCH_TEMPLATE))
        return self.branches[label]

    def on_read(self, label: str, index: int) -> None:
        if type(index) is not int or not 0 <= index < len(self.frame_ids):
            raise IsolationError(f"{label}: read of an invalid local index {index!r}")
        if index > self.released[label]:
            raise IsolationError(f"{label}: read of unreleased (future) local frame {index}")
        self.reads[label].append(index)
        self.events.append({"event": "read", "state": label, "local": index, "frame_id": self.frame_ids[index]})

    def release(self, label: str, frames: Any, index: int) -> None:
        if index != self.released[label] + 1:
            raise IsolationError(f"{label}: frames must be released one at a time")
        frames.release(index)
        self.released[label] = index
        self.events.append({"event": "release", "state": label, "local": index, "frame_id": self.frame_ids[index]})

    def snapshot(self) -> dict:
        return {"counters": dict(self.counters), "events": [dict(e) for e in self.events],
                "reads": {k: list(v) for k, v in self.reads.items()}, "released": dict(self.released),
                "branches": json.loads(json.dumps(_plain(self.branches))),
                "written": {str(k): dict(v) for k, v in sorted(self.written.items())},
                "written_frame_ids": [v["frame_id"] for _k, v in sorted(self.written.items())],
                "cleanup_errors": list(self.cleanup_errors), "outer_cleanup": _plain(self.outer)}


def _attach_cleanup(primary: BaseException, errors: Sequence[str]) -> None:
    if errors:
        try:
            primary.isolation_cleanup_errors = list(getattr(primary, "isolation_cleanup_errors", [])) + list(errors)
        except Exception:  # noqa: BLE001 - some exception types refuse attributes; the ledger still has them
            pass


def run_state_isolated(adapter: Any, *, frame_ids: Sequence[str], prompt_index: int, points: np.ndarray,
                       labels: np.ndarray, shape: tuple[int, int],
                       write_mask: Callable[[int, str, np.ndarray, str], None],
                       events: list | None = None, clock: Callable[[], float] = time.perf_counter,
                       ledger: IsolationLedger | None = None) -> dict:
    """Forward-first, fully separate reverse state (see the module docstring).

    Adapter protocol:
      * ``make_frames(label, count, released, on_read)`` -> frames with ``__len__``
        and ``release(i)``; every RGB read must call ``on_read(label, i)``;
      * ``open_state(frames)``, ``add_prompt(state, index, points, labels)``,
        ``propagate(state, start, reverse)`` -> iterator of
        ``(frame_index, obj_ids, logits)``, ``to_mask(logits)`` -> bool HxW;
      * ``close_state(state, frames)`` must leave ``len(state) == 0``;
      * optional ``close_frames(frames)`` (reader/cache) and ``release_device()``.

    Cleanup ownership starts at reader allocation. Each branch's teardown makes
    INDEPENDENT attempts (iterator close, state close, frames close, device
    release), each recorded with its result. A primary failure is re-raised
    unchanged, with the cleanup errors attached as ``isolation_cleanup_errors``
    and kept in the ledger. A failed forward teardown stops before any reverse
    initialisation. ``ledger`` (caller-owned) holds all progress even on failure.
    """
    count = len(frame_ids)
    if not 0 <= prompt_index < count - 1:
        raise IsolationError("prompt index outside the clip")
    ledger = ledger if ledger is not None else IsolationLedger(frame_ids)
    ledger.frame_ids = list(frame_ids)
    if events is not None:
        ledger.events = events
    counters, events = ledger.counters, ledger.events
    timings: dict[str, float] = {}
    forward_seed: list[np.ndarray] = []

    def persist(index: int, mask: np.ndarray, label: str) -> None:
        if index in ledger.written:
            raise IsolationError(f"local {index} would be written twice")
        write_mask(index, frame_ids[index], mask, label)
        ledger.written[index] = {"frame_id": frame_ids[index], "state": label}
        counters["writes"] += 1
        events.append({"event": "write", "state": label, "local": index, "frame_id": frame_ids[index]})

    def teardown(label: str, frames: Any, state: Any, iterator: Any, close_state: bool) -> list[str]:
        record = ledger.branches[label]["teardown"]
        record["attempted"] = True
        errors: list[str] = []

        def attempt(step: str, fn: Callable[[], Any]) -> None:
            try:
                record["steps"][step] = _plain(fn()) if fn is not None else "skipped"
                if record["steps"][step] is None:
                    record["steps"][step] = "ok"
            except Exception as exc:  # noqa: BLE001 - every cleanup step is attempted independently
                record["steps"][step] = "failed"
                errors.append(f"{label} {step}: {type(exc).__name__}: {exc}"[:500])

        if iterator is not None and hasattr(iterator, "close"):
            attempt("iterator_close", iterator.close)
        if state is not None and close_state:
            attempt("close_state", lambda: adapter.close_state(state, frames))
            try:
                if len(state) != 0:
                    errors.append(f"{label} close_state: state still holds references after teardown")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label} close_state: state length unavailable ({type(exc).__name__})")
        if frames is not None and hasattr(adapter, "close_frames"):
            attempt("close_frames", lambda: adapter.close_frames(frames))
        if hasattr(adapter, "release_device"):
            attempt("release_device", adapter.release_device)
        record["errors"] = list(errors)
        record["completed"] = not errors
        if not errors:
            counters["teardowns"] += 1
        ledger.cleanup_errors.extend(errors)
        events.append({"event": "teardown", "state": label, "completed": not errors})
        return errors

    def branch(label: str, reverse: bool, forbidden_state: Any = None) -> Any:
        info = ledger.open(label, prompt_index)
        frames = state = iterator = None
        owns_state = False
        primary: BaseException | None = None
        try:
            frames = adapter.make_frames(label, count, prompt_index, ledger.on_read)
            info["reader_allocated"] = True
            info["reader_length"] = len(frames)
            if len(frames) != count:
                raise IsolationError(f"{label}: the reader must keep num_frames = {count}")
            info["state_init_attempted"] = True
            state = adapter.open_state(frames)
            if forbidden_state is not None and state is forbidden_state:
                raise IsolationError(f"{label}: the forward state was reused (contamination)")
            owns_state = True
            info["state_init_completed"] = True
            counters["state_inits"] += 1
            events.append({"event": "state_init", "state": label, "num_frames": len(frames)})
            if not hasattr(state, "__len__") or len(state) == 0:
                raise IsolationError(f"{label}: a fresh state must be non-empty after initialisation")
            adapter.add_prompt(state, prompt_index, points, labels)
            info["prompt_added"] = True
            counters["prompt_adds"] += 1
            events.append({"event": "prompt_add", "state": label, "local": prompt_index,
                           "frame_id": frame_ids[prompt_index]})
            iterator = iter(adapter.propagate(state, prompt_index, reverse))
            order = range(prompt_index, -1, -1) if reverse else range(prompt_index, count)
            for expected in order:
                started = clock()
                try:
                    frame_index, obj_ids, logits = next(iterator)
                except StopIteration:
                    raise IsolationError(f"{label}: missing yield for local {expected}") from None
                if type(frame_index) is not int or frame_index != expected or list(obj_ids) != [1]:
                    raise IsolationError(f"{label}: yield {frame_index!r} differs from expected local {expected}")
                mask = _bool_mask(adapter.to_mask(logits), shape).copy()
                counters["reverse_yields" if reverse else "forward_yields"] += 1
                info["yields"].append(frame_index)
                events.append({"event": "yield", "state": label, "local": frame_index,
                               "frame_id": frame_ids[frame_index]})
                if frame_index == prompt_index and not reverse:
                    forward_seed.append(mask.copy())
                    persist(frame_index, mask, label)
                elif frame_index == prompt_index:
                    if not forward_seed or not np.array_equal(mask, forward_seed[0]):
                        raise IsolationError("reverse duplicate seed disagrees with the forward seed")
                    events.append({"event": "duplicate_seed_equal", "state": label, "local": frame_index})
                else:
                    persist(frame_index, mask, label)
                timings[f"{label}:{frame_ids[frame_index]}"] = (clock() - started) * 1000.0
                if not reverse and frame_index + 1 < count:
                    ledger.release(label, frames, frame_index + 1)
            try:
                extra = next(iterator)
            except StopIteration:
                extra = None
            if extra is not None:
                raise IsolationError(f"{label}: an unexpected extra yield")
        except BaseException as exc:  # noqa: BLE001 - re-raised below after cleanup
            primary = exc
            info["primary_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        errors = teardown(label, frames, state, iterator, close_state=owns_state)
        iterator = None
        frames = None
        if primary is not None:
            _attach_cleanup(primary, errors)
            raise primary
        if errors:
            raise IsolationError(f"{label} teardown incomplete; no later state is started: {errors}")
        return state if owns_state else None

    forward_shell = branch("forward", False)            # emptied shell kept only for the identity check
    branch("reverse", True, forbidden_state=forward_shell)
    forward_shell = None

    written = {k: v["state"] for k, v in ledger.written.items()}
    if sorted(written) != list(range(count)):
        raise IsolationError(f"combined masks are not exactly 0..{count - 1}")
    if any(written[i] != "forward" for i in range(prompt_index, count)) or \
            any(written[i] != "reverse" for i in range(prompt_index)):
        raise IsolationError("seed/scored masks must come from the forward state and 0..8 from the reverse state")
    expected_counters = {"forward_yields": count - prompt_index, "reverse_yields": prompt_index + 1, "writes": count,
                         "prompt_adds": 2, "state_inits": 2, "teardowns": 2}
    if counters != expected_counters:
        raise IsolationError(f"counters {counters} differ from the contract {expected_counters}")
    if any(i > prompt_index for i in ledger.reads.get("reverse", [])):
        raise IsolationError("reverse state read a frame above the prompt")
    return {"counters": dict(counters), "events": events, "forward_seed": forward_seed[0], "written_by": written,
            "reads": {k: list(v) for k, v in ledger.reads.items()}, "released": dict(ledger.released),
            "timings_ms": timings, "teardown": {k: v["teardown"] for k, v in ledger.branches.items()},
            "ledger": ledger}


# --------------------------------------------------------------------------
# frozen qualification (descriptive consistency; NOT independent accuracy)


def qualify(masks: Mapping[int, np.ndarray], *, original_seed: np.ndarray, prompt_index: int = PROMPT_LOCAL,
            scored: tuple[int, int] = SCORED_LOCAL, positive: Iterable = POSITIVE, negative: Iterable = NEGATIVE,
            thresholds: Mapping[str, Any] = THRESHOLDS) -> dict:
    lo, hi = scored
    if sorted(masks) != list(range(hi + 1)):
        raise IsolationError("qualification needs every local mask 0..119")
    seed = _bool_mask(masks[prompt_index])
    original = _bool_mask(original_seed, seed.shape)
    area = int(seed.sum())
    largest = largest_component_fraction(seed, thresholds["connectivity"])
    inside = [bool(seed[int(y), int(x)]) for x, y in positive]
    outside = [not bool(seed[int(y), int(x)]) for x, y in negative]
    gates = {
        "seed_largest_component": {"value": largest, "threshold": thresholds["seed_largest_component_min"],
                                   "passed": largest >= thresholds["seed_largest_component_min"]},
        "seed_prompts": {"positive_inside": inside, "negative_outside": outside, "passed": all(inside) and all(outside)},
        "seed_equals_original": {"candidate_content_sha256": mask_content_sha256(seed),
                                 "original_content_sha256": mask_content_sha256(original), "area": area,
                                 "area_expected": thresholds["seed_area_exact"],
                                 "passed": bool(np.array_equal(seed, original)) and area == thresholds["seed_area_exact"]},
    }
    min_area = thresholds["scored_area_min_fraction_of_candidate_seed"] * area
    areas = {i: int(_bool_mask(masks[i], seed.shape).sum()) for i in range(lo, hi + 1)}
    below = [i for i, a in areas.items() if a < min_area]
    gates["scored_area"] = {"min_area": min_area, "frames_below": below, "min_observed": min(areas.values()),
                            "passed": not below}
    pairs = [(i, i + 1, iou(masks[i], masks[i + 1])) for i in range(lo, hi)]      # 109 scored pairs only
    failing = [(a, b, v) for a, b, v in pairs if v < thresholds["scored_consecutive_iou_min"]]
    gates["scored_consecutive_iou"] = {"pairs": len(pairs), "threshold": thresholds["scored_consecutive_iou_min"],
                                       "min_observed": min(v for _a, _b, v in pairs), "failing_pairs": failing,
                                       "passed": not failing and len(pairs) == hi - lo}
    return {"qualified": all(g["passed"] for g in gates.values()), "gates": gates, "scored_areas": areas,
            "consecutive_iou": [[a, b, v] for a, b, v in pairs], "thresholds": dict(thresholds),
            "scope": "descriptive consistency only; not independent accuracy, not qualitative review"}


# --------------------------------------------------------------------------
# bounded, contained output writer


class OutputWriter:
    """Fresh directory; every write is contained, size-capped and budget-checked BEFORE writing."""

    def __init__(self, directory: Path, *, cap_bytes: int = OUTPUT_CAP_BYTES,
                 budget_check: Callable[[int], None] | None = None):
        self.dir = Path(directory)
        self.cap, self.budget_check = cap_bytes, budget_check
        self.total = 0
        self.files: dict[str, str] = {}

    def create(self) -> None:
        if os.path.lexists(self.dir):
            raise IsolationError(f"output {self.dir} already exists; outputs are immutable")
        for part in [self.dir.parent, *self.dir.parent.parents]:
            if part.exists() and part.is_symlink():
                raise IsolationError("output path traverses a symlink")
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.dir.mkdir()

    def write(self, name: str, data: bytes) -> str:
        if (not isinstance(name, str) or not name or name != PurePosixPath(name).name or name.startswith(".")
                or "\\" in name or ":" in name):
            raise IsolationError(f"output name {name!r} is not a plain contained file name")
        if name in self.files or os.path.lexists(self.dir / name):
            raise IsolationError(f"output {name} already exists")
        if self.total + len(data) > self.cap:
            raise IsolationError(f"output cap of {self.cap} bytes would be exceeded by {name}")
        if self.budget_check is not None:
            self.budget_check(len(data))
        path = self.dir / name
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.total += len(data)
        self.files[name] = sha256_bytes(data)
        return self.files[name]


# --------------------------------------------------------------------------
# plan (prepare: CPU only, no torch/SAM import)


def _project_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _contained(root: Path, path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise IsolationError(f"{path} is outside its strict root {root}")
    return resolved


def sam_closure(sam_root: Path) -> dict:
    """Every non-bytecode file of the planned SAM 2 Python package (sources and YAML configs)."""
    pkg = sam_root / "sam2"
    if not pkg.is_dir() or pkg.is_symlink():
        raise IsolationError("planned SAM 2 package directory is missing")
    files = {}
    for path in sorted(pkg.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise IsolationError(f"SAM 2 closure contains a symlink: {path}")
        files[path.relative_to(sam_root).as_posix()] = sha256_file(path)
    return {"files": files, "sha256": sha256_bytes(canonical_bytes(files)), "count": len(files)}


def runtime_identity(device: str) -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python_version": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": _project_relative(Path(sys.executable)), "prefix": _project_relative(Path(sys.prefix)),
            "platform": platform.platform(), "packages": packages, "device": device}


def build_plan(workspace: Path, plan_id: str, device: str, *, allow_loaded_torch: bool = False) -> dict:
    """Deterministic plan (no timestamps) so ``run`` can reconstruct and compare it exactly.
    ``prepare`` never imports torch; only ``run``'s after-closure check may call this with torch loaded."""
    from bench import workbench_cli as cli, workbench_contract as wc, workbench_geometry as _wg, workbench_neural as wn
    wc._ident(plan_id, "plan_id")
    if "torch" in sys.modules and not allow_loaded_torch:
        raise IsolationError("prepare/plan reconstruction must run before torch is imported")
    ws = cli.Workspace(Path(workspace))
    _contained(ROOT, ws.path)
    # sources (strict roots)
    bench_sources = {}
    for rel in BENCH_SOURCES:
        path = _contained(ROOT / "bench", ROOT / rel)
        bench_sources[rel] = sha256_file(path)
    for mod in ("bench.workbench_cli", "bench.workbench_neural", "bench.workbench_geometry", "bench.workbench_contract"):
        _contained(ROOT / "bench", Path(sys.modules[mod].__file__))
    if wn.SAM2_CONFIG != "configs/sam2.1/sam2.1_hiera_b+.yaml":
        raise IsolationError("production SAM 2 config constant changed")
    sam_root = _contained(ROOT / ".cache/model-quality", ROOT / wn.SAM2_SOURCES_RELATIVE)
    ckpt = _contained(ROOT / ".cache/model-quality", ROOT / wn.SAM2_CHECKPOINT_RELATIVE)
    if not ckpt.is_file() or sha256_file(ckpt) != SAM2_CHECKPOINT_SHA256:
        raise IsolationError("SAM 2.1 checkpoint differs from its pin")
    closure = sam_closure(sam_root)
    if wn.SAM2_CONFIG not in {k.split("sam2/", 1)[-1] for k in closure["files"]}:
        raise IsolationError("planned SAM 2 config file is missing from the closure")
    # inputs
    clip_path = ws.clip_path(CLIP_ID)
    if sha256_file(clip_path) != CLIP_SHA256:
        raise IsolationError("clip record differs from its pin")
    clip = ws.load_clip(CLIP_ID)
    ids = frame_ids_for()
    if [f.frame_id for f in clip.frames] != ids or (clip.width, clip.height) != (640, 480):
        raise IsolationError("clip frames are not exactly s000361..s000480 at 640x480")
    sel_path = _contained(ws.derived_dir, ROOT / SELECTION_FILE)
    sel_raw = sel_path.read_bytes()
    if sha256_bytes(sel_raw) != SELECTION_FILE_SHA256:
        raise IsolationError("original selection record differs from its pin")
    sel_doc = loads_strict(sel_raw, "selection record")
    records = [r for r in sel_doc.get("records", []) if r.get("clip_id") == CLIP_ID]
    if len(records) != 1:
        raise IsolationError("exactly one original selection for the clip is required")
    sel = records[0]["selection"]
    onboarding = sel["onboarding"]["selection"]
    if (sel.get("selection_id") != SELECTION_ID or onboarding.get("frame_id") != PROMPT_FRAME_ID
            or [tuple(p) for p in onboarding.get("positive", [])] != list(POSITIVE)
            or [tuple(p) for p in onboarding.get("negative", [])] != list(NEGATIVE)
            or sel["onboarding"].get("keyframe_ids") != ids[:PROMPT_LOCAL + 1]):
        raise IsolationError("original prompt differs from the frozen points/frames (no new selection)")
    mask_dir = ws.masks_dir / CLIP_ID / ORIGINAL_MASK_SET
    man_raw = (mask_dir / "manifest.json").read_bytes()
    if sha256_bytes(man_raw) != ORIGINAL_MANIFEST_SHA256:
        raise IsolationError("original mask manifest differs from its pin")
    manifest = loads_strict(man_raw, "original mask manifest")
    seed_row = manifest["masks"].get(PROMPT_FRAME_ID) or {}
    if (len(manifest["masks"]) != COUNT or seed_row.get("path") != ORIGINAL_SEED["path"]
            or seed_row.get("sha256") != ORIGINAL_SEED["file_sha256"]
            or seed_row.get("content_sha256") != ORIGINAL_SEED["content_sha256"]
            or manifest["producer"].get("checkpoint_sha256") != SAM2_CHECKPOINT_SHA256):
        raise IsolationError("original seed mask record differs from its pins")
    seed_raw = (mask_dir / ORIGINAL_SEED["path"]).read_bytes()
    seed = decode_mask_png(seed_raw, (480, 640))
    if sha256_bytes(seed_raw) != ORIGINAL_SEED["file_sha256"] or mask_content_sha256(seed) != ORIGINAL_SEED["content_sha256"] \
            or int(seed.sum()) != ORIGINAL_SEED["area"]:
        raise IsolationError("original seed mask bytes/content/area differ")
    # all 120 RGB inputs: file hash (verified) and decoded identity
    reader = cli.FrameReader(ws, clip)
    try:
        file_hashes = reader.verify_frames(ids)
        frames = []
        for i, fid in enumerate(ids):
            frame = clip.frame(fid)
            _rgb, ident = reader.read(frame)
            frames.append({"local": i, "frame_id": fid, "source_frame_index": frame.source_frame_index,
                           "image_path": frame.image_path, "file_sha256": file_hashes[fid],
                           "decoded_sha256": ident["image_sha256"]})
    finally:
        reader.close()
    out_rel = _project_relative(ws.derived_dir / EXPERIMENT_DIR / "runs" / plan_id)
    return {
        "schema": PLAN_SCHEMA, "plan_id": plan_id, "tool": "tools/workbench_sam_forward_isolation.py",
        "tool_sha256": sha256_file(Path(__file__)), "workspace": _project_relative(ws.path), "planned_output": out_rel,
        "question": "does same-state reverse-prefix memory influence the original forward predictions?",
        "hypothesis_status": "cause of the original fragmentation UNPROVEN; results are conditional",
        "clip": {"clip_id": CLIP_ID, "clip_sha256": CLIP_SHA256, "count": COUNT, "frames": frames,
                 "prompt_local": PROMPT_LOCAL, "prompt_frame_id": PROMPT_FRAME_ID, "scored_local": list(SCORED_LOCAL)},
        "prompt": {"selection_file": SELECTION_FILE, "selection_file_sha256": SELECTION_FILE_SHA256,
                   "selection_id": SELECTION_ID, "positive": [list(p) for p in POSITIVE],
                   "negative": [list(p) for p in NEGATIVE], "point_convention": POINT_CONVENTION},
        "original": {"mask_set": ORIGINAL_MASK_SET, "manifest_sha256": ORIGINAL_MANIFEST_SHA256, "seed": ORIGINAL_SEED,
                     "seed_largest_component_fraction": largest_component_fraction(seed)},
        "sam": {"checkpoint": _project_relative(ckpt), "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
                "config": wn.SAM2_CONFIG, "sources_root": _project_relative(sam_root), "closure": closure,
                "settings": dict(SAM_SETTINGS)},
        "bench_sources": bench_sources, "runtime": runtime_identity(device),
        "state_isolation": {"order": ["forward 9..119 (fresh state)", "teardown", "reverse 9..0 (fresh state)"],
                            "num_frames_both_states": COUNT, "init_warm_frame_local": 0,
                            "expected_counters": dict(EXPECTED_COUNTERS),
                            "seed_and_scored_source": "forward state only", "prefix_source": "reverse state 0..8",
                            "duplicate_seed_rule": "reverse local 9 must binary-equal the forward and original seeds"},
        "thresholds": dict(THRESHOLDS),
        "preview": {"local": list(PREVIEW_LOCAL), "frame_ids": [ids[i] for i in PREVIEW_LOCAL],
                    "size": list(PREVIEW_SIZE), "layout": ".cache/workbench-ycb-mask-review.py 3x3 tiles"},
        "budgets": {"output_cap_bytes": OUTPUT_CAP_BYTES, "derived_limit_bytes": wc.DERIVED_BUDGET_BYTES,
                    "new_input_limit_bytes": wc.NEW_INPUT_BUDGET_BYTES, "free_reserve_bytes": wc.FREE_RESERVE_BYTES,
                    "new_inputs_acquired": 0},
        "flags": dict(FLAGS),
    }


def validate_plan_shape(plan: Any) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise IsolationError("not a state-isolation plan")
    clip = plan.get("clip") or {}
    frames = clip.get("frames")
    if (clip.get("count") != COUNT or not isinstance(frames, list) or len(frames) != COUNT
            or [f.get("frame_id") for f in frames] != frame_ids_for() or [f.get("local") for f in frames] != list(range(COUNT))
            or clip.get("prompt_local") != PROMPT_LOCAL):
        raise IsolationError("plan frame accounting differs from the frozen 120-frame clip")
    if plan.get("thresholds") != THRESHOLDS or plan.get("flags") != FLAGS:
        raise IsolationError("plan thresholds/flags differ from the frozen values")
    if (plan.get("preview") or {}).get("local") != list(PREVIEW_LOCAL):
        raise IsolationError("plan preview indices differ")


def prepare(workspace: Path, plan_id: str, device: str = "cuda") -> dict:
    from bench import workbench_cli as cli
    plan = build_plan(Path(workspace), plan_id, device)
    validate_plan_shape(plan)
    ws = cli.Workspace(Path(workspace))
    plan_dir = ws.derived_dir / EXPERIMENT_DIR / "plans" / plan_id
    data = canonical_bytes(plan)
    ws.require_budget(extra_derived=len(data))
    writer = OutputWriter(plan_dir, budget_check=lambda n: ws.require_budget(extra_derived=n))
    writer.create()
    sha = writer.write("plan.json", data)
    return {"plan": _project_relative(plan_dir / "plan.json"), "plan_sha256": sha, "planned_output": plan["planned_output"]}


# --------------------------------------------------------------------------
# run (explicit parent-scheduled single-GPU execution)


class SamAdapter:
    """Real SAM 2.1 adapter mirroring production ``segment_prompted`` calls exactly.
    Teardown is split into independent steps the engine attempts separately:
    ``close_state`` (clears the inference state), ``close_frames`` (reader cache
    and frame reader) and ``release_device`` (``_release_device_memory``; CUDA
    faults propagate)."""

    def __init__(self, ws, clip, plan: Mapping, predictor, predictor_module, torch, wn, cli):
        self.ws, self.clip, self.plan = ws, clip, plan
        self.predictor, self.predictor_module, self.torch, self.wn, self.cli = predictor, predictor_module, torch, wn, cli
        self.readers: dict[int, Any] = {}
        self.decoded = {f["local"]: f["decoded_sha256"] for f in plan["clip"]["frames"]}

    def make_frames(self, label, count, released, on_read):
        reader = self.cli.FrameReader(self.ws, self.clip)
        frame_ids = [f["frame_id"] for f in self.plan["clip"]["frames"]]

        def read_rgb(index: int) -> np.ndarray:
            on_read(label, index)
            rgb, ident = reader.read(self.clip.frame(frame_ids[index]))
            if ident["image_sha256"] != self.decoded[index]:
                raise IsolationError(f"decoded RGB identity of {frame_ids[index]} differs from the plan")
            return rgb

        try:
            frames = self.wn._CausalFrames(read_rgb, count, self.predictor.image_size, released,
                                           self.clip.height, self.clip.width)
        except BaseException:
            reader.close()
            raise
        self.readers[id(frames)] = reader
        return frames

    def open_state(self, frames):
        torch, module = self.torch, self.predictor_module
        original = module.load_video_frames
        module.load_video_frames = lambda *a, **k: (frames, self.clip.height, self.clip.width)
        try:
            with torch.inference_mode():
                return self.predictor.init_state("workbench-causal-source", offload_video_to_cpu=True,
                                                 offload_state_to_cpu=True, async_loading_frames=False)
        finally:
            module.load_video_frames = original

    def add_prompt(self, state, index, points, labels):
        with self.torch.inference_mode():
            self.predictor.add_new_points_or_box(state, frame_idx=index, obj_id=1, points=points, labels=labels)

    def propagate(self, state, start, reverse):
        torch = self.torch
        inner = self.predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse)

        def gen() -> Iterator:
            try:
                while True:
                    with torch.inference_mode():
                        try:
                            frame_index, obj_ids, logits = next(inner)
                        except StopIteration:
                            return
                    yield frame_index, obj_ids, logits
            finally:
                inner.close()
        return gen()

    def to_mask(self, logits):
        from bench import workbench_telemetry as wt
        with self.torch.inference_mode():
            wt.gpu_synchronize()
            return logits[0, 0].detach().cpu().numpy() > 0

    def close_state(self, state, frames):
        state.clear()
        return {"state_cleared": True}

    def close_frames(self, frames):
        errors, reads = [], None
        try:
            reads = list(getattr(frames, "reads", []))
            frames.cache.clear()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cache: {type(exc).__name__}: {exc}")
        reader = self.readers.pop(id(frames), None)
        if reader is not None:
            try:
                reader.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader: {type(exc).__name__}: {exc}")
        if errors:
            raise IsolationError(f"frame/reader cleanup failed: {errors}")
        return {"cache_cleared": True, "reader_closed": reader is not None, "frames_read": reads}

    def release_device(self):
        return self.wn._release_device_memory(self.torch)

    def close_all(self) -> list[str]:
        """Outer safety net: close any reader still registered (never raises)."""
        errors = []
        for key in list(self.readers):
            reader = self.readers.pop(key)
            try:
                reader.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader: {type(exc).__name__}: {exc}")
        return errors


def _load_plan(plan_path: Path, expected_sha256: str) -> tuple[dict, bytes]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or \
            any(c not in "0123456789abcdef" for c in expected_sha256):
        raise IsolationError("--expected-plan-sha256 must be a lowercase SHA-256")
    path = Path(plan_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > (16 << 20):
        raise IsolationError("plan must be a regular bounded file")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise IsolationError("plan differs from the externally expected SHA-256")
    plan = loads_strict(raw, "plan")
    validate_plan_shape(plan)
    if canonical_bytes(plan) != raw:
        raise IsolationError("plan file is not in canonical form")
    return plan, raw


def _check_imported_sam(sam_root: Path) -> dict:
    files = {}
    for name, module in list(sys.modules.items()):
        if name == "sam2" or name.startswith("sam2."):
            origin = getattr(module, "__file__", None)
            if origin is None:
                continue
            path = Path(origin).resolve()
            if sam_root.resolve() not in path.parents:
                raise IsolationError(f"imported {name} from {path}, outside the planned SAM 2 root")
            files[name] = _project_relative(path)
    return files


def run(plan_path: Path, expected_plan_sha256: str) -> dict:
    from bench import workbench_cli as cli, workbench_neural as wn, workbench_telemetry as wt
    plan, raw = _load_plan(plan_path, expected_plan_sha256)
    workspace = ROOT / plan["workspace"]
    if canonical_bytes(build_plan(workspace, plan["plan_id"], plan["runtime"]["device"])) != raw:
        raise IsolationError("reconstructed plan differs (sources, runtime, checkpoint or inputs changed)")
    ws = cli.Workspace(workspace)
    out_dir = ROOT / plan["planned_output"]
    _contained(ws.derived_dir / EXPERIMENT_DIR / "runs", out_dir)
    if "sam2" in sys.modules:
        raise IsolationError("a SAM 2 module was imported before the planned root was installed")
    clip = ws.load_clip(CLIP_ID)
    ids = [f["frame_id"] for f in plan["clip"]["frames"]]
    writer = OutputWriter(out_dir, budget_check=lambda n: ws.require_budget(extra_derived=n))
    ws.require_budget(extra_derived=OUTPUT_CAP_BYTES)
    writer.create()
    ledger = IsolationLedger(ids)
    stage, manifest_masks, outer_errors = "setup", {}, []
    try:
        import torch
        sam_root = ROOT / plan["sam"]["sources_root"]
        sys.path.insert(0, str(sam_root))
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        from sam2.modeling.position_encoding import PositionEmbeddingRandom  # type: ignore
        import sam2.sam2_video_predictor as predictor_module  # type: ignore
        from bench.quality_sam2 import deterministic_position_grid
        imported = _check_imported_sam(sam_root)
        points = np.array([[x + 0.5, y + 0.5] for x, y in POSITIVE + NEGATIVE], np.float32)
        labels = np.array([1] * len(POSITIVE) + [0] * len(NEGATIVE), np.int32)

        def write_mask(index, frame_id, mask, label):
            data = encode_mask_png(mask)
            if not np.array_equal(decode_mask_png(data, (clip.height, clip.width)), mask):
                raise IsolationError(f"mask PNG round trip differs for {frame_id}")
            writer.write(f"m{index:05d}.png", data)
            manifest_masks[frame_id] = mask_record(index, mask, data, label, (clip.height, clip.width))

        samples = [wt.resource_sample("before_gpu_stage", disk_path=ws.path)]
        stage = "gpu_stage"
        original_forward = PositionEmbeddingRandom.forward
        result = None
        with wt.gpu_stage("sam_forward_isolation"):
            PositionEmbeddingRandom.forward = deterministic_position_grid
            predictor = adapter = None
            try:
                stage = "build_predictor"
                predictor = build_sam2_video_predictor(
                    plan["sam"]["config"], str(ROOT / plan["sam"]["checkpoint"]), device=plan["runtime"]["device"],
                    apply_postprocessing=False, vos_optimized=False)
                imported = _check_imported_sam(sam_root)
                adapter = SamAdapter(ws, clip, plan, predictor, predictor_module, torch, wn, cli)
                stage = "state_isolated_propagation"
                result = run_state_isolated(adapter, frame_ids=ids, prompt_index=PROMPT_LOCAL, points=points,
                                            labels=labels, shape=(clip.height, clip.width), write_mask=write_mask,
                                            ledger=ledger)
                samples.append(wt.resource_sample("after_propagation", disk_path=ws.path))
            finally:
                # Outer cleanup INSIDE the GPU stage: every step attempted independently, never raising here.
                try:
                    PositionEmbeddingRandom.forward = original_forward
                except Exception as exc:  # noqa: BLE001
                    outer_errors.append(f"position grid restore: {type(exc).__name__}: {exc}")
                if adapter is not None:
                    outer_errors.extend(adapter.close_all())
                adapter = predictor = None
                try:
                    ledger.outer["final_device_release"] = _plain(wn._release_device_memory(torch))
                except Exception as exc:  # noqa: BLE001
                    outer_errors.append(f"final device release: {type(exc).__name__}: {exc}")
                ledger.outer["errors"] = list(outer_errors)
                ledger.cleanup_errors.extend(outer_errors)
        if outer_errors:
            raise IsolationError(f"outer predictor/device cleanup incomplete: {outer_errors}")
        stage = "preview"
        masks = {row["local"]: decode_mask_png((out_dir / row["path"]).read_bytes(), (clip.height, clip.width))
                 for row in manifest_masks.values()}
        preview = _preview_png(ws, clip, plan, masks)
        stage = "write_outputs"
        writer.write("preview.png", preview)
        writer.write("events.jsonl", b"".join(canonical_bytes(e) + b"\n" for e in ledger.events))
        manifest = {"schema": MASK_SCHEMA, "clip_id": CLIP_ID, "mask_set": f"sam2-forward-isolation-{plan['plan_id']}",
                    "producer": {"kind": PRODUCER_KIND, "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
                                 "config": plan["sam"]["config"], "device": plan["runtime"]["device"],
                                 "selection": {"frame_id": PROMPT_FRAME_ID, "positive": [list(p) for p in POSITIVE],
                                               "negative": [list(p) for p in NEGATIVE], "reviewed": True,
                                               "selection_id": SELECTION_ID},
                                 "prompt_frame_id": PROMPT_FRAME_ID, "point_convention": POINT_CONVENTION,
                                 "causal_guard": "unreleased-frame reads raise; forward frames released one at a time; "
                                                 "reverse state never reads above the prompt",
                                 "state_isolation": plan["state_isolation"], "plan_sha256": expected_plan_sha256},
                    "evaluator_inputs_read": False, "cancelled": False, "masks": manifest_masks,
                    "telemetry": {"segmentation_ms_by_state_frame": result["timings_ms"],
                                  "process_peak_rss_bytes": wt.process_peak_rss_bytes(), **wt.gpu_peak_bytes(),
                                  "resource_samples": samples},
                    "note": "private diagnostic staging output; NOT registered as a workspace mask set"}
        writer.write("manifest.json", canonical_bytes(manifest))
        stage = "closure_after"
        if canonical_bytes(build_plan(workspace, plan["plan_id"], plan["runtime"]["device"],
                                      allow_loaded_torch=True)) != raw:
            raise IsolationError("source/input/runtime closure changed during the run")
        if _check_imported_sam(sam_root) != imported:
            raise IsolationError("imported SAM 2 modules changed during the run")
        stage = "readback_verification"          # after ALL writes and closure checks, before the receipt
        verified = verify_staged_outputs(out_dir, recorded=dict(writer.files), frame_ids=ids,
                                         prompt_index=PROMPT_LOCAL, shape=(clip.height, clip.width))
        if not np.array_equal(verified["masks"][PROMPT_LOCAL], result["forward_seed"]):
            raise IsolationError("verified seed differs from the forward seed")
        stage = "qualification"
        original_seed = decode_mask_png((ws.masks_dir / CLIP_ID / ORIGINAL_MASK_SET / ORIGINAL_SEED["path"]).read_bytes(),
                                        (clip.height, clip.width))
        qualification = qualify(verified["masks"], original_seed=original_seed)
        stage = "receipt"
        receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "plan_sha256": expected_plan_sha256,
                   "counters": result["counters"], "expected_counters": EXPECTED_COUNTERS,
                   "ledger": ledger.snapshot(), "imported_sam_modules": imported, "duplicate_seed_equal": True,
                   "readback_verification": verified["proof"], "qualification": qualification,
                   "qualified_for_parent_review": qualification["qualified"],
                   "closure": {"before_equals_plan": True, "after_equals_plan": True},
                   "flags": dict(FLAGS),
                   "receipt_hash": "not self-hashed; the parent hashes receipt.json externally after the process",
                   "qualitative_review": "NOT done; mandatory separate parent review of preview.png",
                   "hypothesis": "conditional: isolated-state success or failure proves neither original foreground "
                                 "accuracy nor any head improvement"}
        writer.write("receipt.json", canonical_bytes(receipt))
        return {"directory": _project_relative(out_dir), "qualified": qualification["qualified"],
                "receipt_sha256_at_write": writer.files["receipt.json"]}
    except BaseException as exc:
        failure = {"schema": RECEIPT_SCHEMA, "status": "failed", "stage": stage,
                   "primary_error": f"{type(exc).__name__}: {exc}"[:4000],
                   "cleanup_errors": list(getattr(exc, "isolation_cleanup_errors", [])) + list(ledger.cleanup_errors),
                   "plan_sha256": expected_plan_sha256, "ledger": ledger.snapshot(),
                   "expected_frame_ids": list(ids), "written_frame_ids": sorted(manifest_masks),
                   "missing_frame_ids": [f for f in ids if f not in manifest_masks],
                   "partial_mask_identities": manifest_masks, "outputs_recorded": dict(writer.files),
                   "flags": dict(FLAGS), "note": "incomplete; no qualification, promotion or registration"}
        try:
            writer.write("failure.json", canonical_bytes(failure))
        except BaseException as persist_exc:  # noqa: BLE001 - reserve/cap/CUDA may prevent persistence
            failure["failure_json_persisted"] = False
            failure["persist_error"] = f"{type(persist_exc).__name__}: {persist_exc}"[:1000]
            try:
                sys.stderr.write("SAM isolation FAILED and failure.json could not be written; terminal record:\n"
                                 + json.dumps(failure, sort_keys=True, default=str) + "\n")
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
        raise


def mask_record(local: int, mask: np.ndarray, png: bytes, state: str, shape: tuple[int, int]) -> dict:
    """Manifest row for one written mask (no arrays)."""
    m = _bool_mask(mask, shape)
    return {"path": f"m{local:05d}.png", "sha256": sha256_bytes(png), "width": shape[1], "height": shape[0],
            "content_sha256": mask_content_sha256(m), "empty": bool(not m.any()), "state": state, "local": local}


MASK_ROW_KEYS = {"path", "sha256", "width", "height", "content_sha256", "empty", "state", "local"}
STAGED_EXTRAS = ("events.jsonl", "preview.png", "manifest.json")


def verify_staged_outputs(directory: Path, *, recorded: Mapping[str, str], frame_ids: Sequence[str],
                          prompt_index: int = PROMPT_LOCAL, shape: tuple[int, int] = (480, 640),
                          clip_id: str = CLIP_ID, preview_size: tuple[int, int] | None = PREVIEW_SIZE) -> dict:
    """Read back EVERY staged output (receipt excluded, it is not written yet).

    Checks, in order:
    * the exact inventory: ``m00000.png``.. plus events/preview/manifest, with no
      extra, missing, symlinked, hard-linked or non-regular entry;
    * every file's actual bytes re-hashed against the recorded write hash (so
      the same pixels in different PNG bytes are rejected);
    * the manifest (strict JSON): schema, clip, flags, exact frame set, row keys,
      path/local/branch ownership and file hash per frame;
    * each mask strictly decoded (binary 0/255, exact shape), with its content
      SHA and ``empty`` flag matching the manifest;
    * the events counts (forward/reverse yields and writes);
    * the preview PNG size.

    Returns ``{"masks": {local: bool array}, "manifest": dict, "proof": dict}``.
    """
    d = Path(directory)
    n = len(frame_ids)
    expected = {f"m{i:05d}.png" for i in range(n)} | set(STAGED_EXTRAS)
    if set(recorded) != expected:
        raise IsolationError(f"recorded output inventory differs: missing {sorted(expected - set(recorded))[:5]}, "
                             f"extra {sorted(set(recorded) - expected)[:5]}")
    present = {}
    for entry in os.scandir(d):
        st = os.stat(entry.path, follow_symlinks=False)
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise IsolationError(f"staged output {entry.name} is not a regular file")
        if st.st_nlink != 1:
            raise IsolationError(f"staged output {entry.name} is hard-linked")
        present[entry.name] = st.st_size
    if set(present) != expected:
        raise IsolationError(f"staged inventory differs: missing {sorted(expected - set(present))[:5]}, "
                             f"extra {sorted(set(present) - expected)[:5]}")
    data, files = {}, {}
    for name in sorted(expected):
        path = _contained(d, d / name)
        blob = path.read_bytes()
        digest = sha256_bytes(blob)
        if digest != recorded[name]:
            raise IsolationError(f"{name}: actual bytes differ from the recorded write hash")
        data[name], files[name] = blob, {"sha256": digest, "bytes": len(blob)}
    manifest = loads_strict(data["manifest.json"], "staged manifest")
    if (not isinstance(manifest, dict) or manifest.get("schema") != MASK_SCHEMA or manifest.get("clip_id") != clip_id
            or manifest.get("evaluator_inputs_read") is not False or manifest.get("cancelled") is not False
            or not isinstance(manifest.get("masks"), dict) or set(manifest["masks"]) != set(frame_ids)):
        raise IsolationError("staged manifest schema/clip/flags/frame set differ")
    masks, contents = {}, {}
    for local, frame_id in enumerate(frame_ids):
        row = manifest["masks"][frame_id]
        owner = "reverse" if local < prompt_index else "forward"
        if (not isinstance(row, dict) or set(row) != MASK_ROW_KEYS or row["path"] != f"m{local:05d}.png"
                or row["local"] != local or row["state"] != owner or row["sha256"] != files[row["path"]]["sha256"]
                or row["width"] != shape[1] or row["height"] != shape[0]):
            raise IsolationError(f"{frame_id}: manifest row differs from its file, index or branch owner")
        mask = decode_mask_png(data[row["path"]], shape)
        content = mask_content_sha256(mask)
        if content != row["content_sha256"] or row["empty"] is not bool(not mask.any()):
            raise IsolationError(f"{frame_id}: decoded mask content differs from the manifest")
        masks[local], contents[frame_id] = mask, content
    counts = {"forward_yields": 0, "reverse_yields": 0, "writes": 0}
    for line in data["events.jsonl"].splitlines():
        event = loads_strict(line, "staged events")
        if not isinstance(event, dict) or "event" not in event:
            raise IsolationError("staged events line malformed")
        if event["event"] == "yield":
            counts[f"{event.get('state')}_yields"] = counts.get(f"{event.get('state')}_yields", 0) + 1
        elif event["event"] == "write":
            counts["writes"] += 1
    expected_counts = {"forward_yields": n - prompt_index, "reverse_yields": prompt_index + 1, "writes": n}
    if counts != expected_counts:
        raise IsolationError(f"staged events counts {counts} differ from {expected_counts}")
    if preview_size is not None:
        from io import BytesIO
        from PIL import Image
        try:
            with Image.open(BytesIO(data["preview.png"])) as img:
                if img.format != "PNG" or img.size != tuple(preview_size):
                    raise IsolationError("preview is not the planned PNG size")
        except IsolationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IsolationError(f"preview unreadable: {type(exc).__name__}") from None
    proof = {"inventory": sorted(expected), "files": files, "mask_count": n,
             "mask_content_sha256": contents, "events_counts": counts,
             "manifest_sha256": files["manifest.json"]["sha256"],
             "method": "post-write readback: exact inventory, per-file SHA-256, strict binary decode, manifest identity"}
    return {"masks": masks, "manifest": manifest, "proof": proof}


def _preview_png(ws, clip, plan: Mapping, masks: Mapping[int, np.ndarray]) -> bytes:
    """Same 3x3 layout as .cache/workbench-ycb-mask-review.py (960x810, 320x240 tiles, green overlay)."""
    from io import BytesIO
    from PIL import Image, ImageDraw
    from bench import workbench_cli as cli
    reader = cli.FrameReader(ws, clip)
    decoded = {f["local"]: f["decoded_sha256"] for f in plan["clip"]["frames"]}
    sheet = Image.new("RGB", PREVIEW_SIZE, "white")
    draw = ImageDraw.Draw(sheet)
    try:
        for i, index in enumerate(PREVIEW_LOCAL):
            frame = clip.frames[index]
            rgb, ident = reader.read(frame)
            if ident["image_sha256"] != decoded[index]:
                raise IsolationError(f"preview RGB {frame.frame_id} differs from the plan")
            mask = masks[index]
            rgb[mask] = (rgb[mask] * .55 + np.array([0, 255, 80]) * .45).astype(np.uint8)
            panel = Image.fromarray(rgb)
            panel.thumbnail((320, 240))
            x, y = (i % 3) * 320, (i // 3) * 270
            sheet.paste(panel, (x, y + 25))
            branch = "seed" if index == PROMPT_LOCAL else "reverse" if index < PROMPT_LOCAL else "forward"
            draw.text((x + 4, y + 4), f"{frame.frame_id} ({branch})", fill="black")
    finally:
        reader.close()
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAM 2.1 forward-first state-isolation ablation (original prompt).")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="CPU-only immutable plan (no torch/SAM import)")
    s.add_argument("--workspace", default=".cache/video-workbench")
    s.add_argument("--plan-id", required=True)
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("run", help="explicit parent-scheduled single-GPU execution")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(Path(args.workspace), args.plan_id, args.device)
        else:
            result = run(Path(args.plan), args.expected_plan_sha256)
    except (IsolationError, OSError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
