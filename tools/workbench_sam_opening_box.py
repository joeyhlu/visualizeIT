"""Opening-only box-prompt SAM 2.1 diagnostic (state-isolated, original cracker clip).

ONE frozen prompt chosen by the parent from the opening RGB s000370 only: the
whole-box ``[388, 48, 583, 278]`` (xyxy, integer pixel centres) plus the
ORIGINAL three positive and three negative points. The drill negative
(520, 165) deliberately lies inside the box. No scored RGB, ground truth,
pose, depth or mesh was used to choose it. There are no alternatives,
sweeps, retries, registration or neural replay.

The engine, ledger, real adapter, PNG codec, writer, teardown and readback
come unchanged from the accepted ``tools/workbench_sam_forward_isolation.py``
(SHA-256 ``0aef88a8…``, verified before every plan or run). Nothing in it is
modified or monkeypatched. This wrapper adds:

* ``BoxSamAdapter``: the real ``SamAdapter`` with ONLY ``add_prompt``
  overridden, to call ``predictor.add_new_points_or_box(state, frame_idx=9,
  obj_id=1, points=<original +0.5>, labels=int32[1,1,1,0,0,0],
  box=float32[388.5, 48.5, 583.5, 278.5], clear_old_points=True,
  normalize_coords=True)``. SAM itself prepends the box corners (labels 2, 3)
  and normalises by 640x480 and its image size. The coordinates are never
  pre-normalised or half-pixel shifted twice, and every actual call is recorded;
* distinct plan/receipt schemas, producer kind, prompt ID, mask set and the
  fresh contained directory ``derived/sam-opening-box/``;
* a NEW qualifier (``qualify_box``) that never compares the seed with the
  original seed, because that check is inapplicable to a changed prompt.

Commands (parent executes after review/tests; the implementer runs neither)::

    prepare --workspace .cache/video-workbench --plan-id cracker-opening-box-isolated-v1 --device cuda
    run --plan <plan.json> --expected-plan-sha256 <sha>

This is a development-only diagnostic. It makes no accuracy, phone or
automatic-geometry claim, and the parent's qualitative review is mandatory
and separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_sam_forward_isolation as iso  # noqa: E402  (accepted; read-only reuse)

IsolationError = iso.IsolationError
PLAN_SCHEMA = "workbench_sam_opening_box_plan_v1"
RECEIPT_SCHEMA = "workbench_sam_opening_box_receipt_v1"
PRODUCER_KIND = "sam2.1_video_prompted_opening_box_state_isolated_forward_first"
EXPERIMENT_DIR = "sam-opening-box"

ACCEPTED_ISOLATION_SHA256 = "0aef88a8de9b9e7918e3a23a73251d4fcaa61816df9d9e0df717a033209aca9e"
ORIGINAL_PLAN_ID = "original-prompt-forward-isolated-v1"
ORIGINAL_PLAN_SHA256 = "7304cf296cc4ed14108d4c8ac79277e22982842003336c98c52f7ca919830acc"
PROMPT_FILE = ".cache/video-workbench/derived/validation/ycb-cracker-opening-box-v1.json"
PROMPT_SHA256 = "15dce4be95eb362c9044f808c872abd810f98f072cc8f9bd96b79439ddd308a5"
PROMPT_ID = "cracker-rgb370-whole-box-v1"
RGB370_FILE_SHA256 = "822ffafadb2b403e754e7a2790b971ba221a70664326bdddff269dbb85512fee"
RGB370_DECODED_SHA256 = "00090d9fb67ffc81ff7781ad930433a9e7330f461c08d64811a557c76ab19bf8"
BOX_XYXY = (388, 48, 583, 278)
IMAGE_SIZE = (640, 480)
PROMPT_KEYS = {"box_conversion", "box_format", "box_xyxy_integer_centres", "clip_id", "experiment_id", "frame_id",
               "human_annotation", "image_size", "inference_started", "local_index", "negative_points",
               "original_selection_id", "parent_opening_overlay_review", "point_convention", "positive_points",
               "prompt_id", "rgb_decoded_sha256", "rgb_file_sha256", "sam_box_xyxy", "sam_point_conversion", "schema",
               "selection_basis", "status"}
PROMPT_SCHEMA = "workbench-opening-box-selection-v1"
THRESHOLDS = {"seed_largest_component_min": 0.95, "scored_area_min_fraction_of_new_seed": 0.5,
              "scored_consecutive_iou_min": 0.5, "zero_union_iou": 0.0, "connectivity": 8, "seed_nonempty": True}
OUTPUT_CAP_BYTES = 8 << 20      # ALL new outputs: plan + run outputs (masks, events, preview, manifest, receipt/failure)
FLAGS = {"human_accuracy": False, "independent_accuracy": False, "phone": False, "trained_head_test": False,
         "automatic_geometry": False, "development_only": True, "qualitative_review_done": False,
         "registered_mask_set": False, "promotion": False, "neural_replay": False}


# --------------------------------------------------------------------------
# pure prompt helpers (no torch/SAM)


def _int_coord(value: Any, what: str) -> int:
    if type(value) is not int:
        raise IsolationError(f"{what} must be an integer (not bool/float)")
    return value


def validate_prompt(doc: Any) -> dict:
    """Exact frozen prompt identity: keys/schema/IDs/frame/size/local index,
    integer in-image coordinates, ordered box corners, exact original points and
    every positive inside the box. Negatives MAY lie inside the box."""
    if not isinstance(doc, dict) or set(doc) != PROMPT_KEYS or doc.get("schema") != PROMPT_SCHEMA:
        raise IsolationError("prompt record must have exactly the frozen keys and schema")
    if (doc["prompt_id"] != PROMPT_ID or doc["clip_id"] != iso.CLIP_ID or doc["frame_id"] != iso.PROMPT_FRAME_ID
            or type(doc["local_index"]) is not int or doc["local_index"] != iso.PROMPT_LOCAL
            or doc["image_size"] != list(IMAGE_SIZE) or doc["box_format"] != "xyxy"
            or doc["point_convention"] != "integer_centers" or doc["human_annotation"] is not False
            or doc["inference_started"] is not False or doc["rgb_file_sha256"] != RGB370_FILE_SHA256
            or doc["rgb_decoded_sha256"] != RGB370_DECODED_SHA256):
        raise IsolationError("prompt identity (ID/clip/frame/local/size/format/RGB) differs from the pin")
    box = doc["box_xyxy_integer_centres"]
    if not isinstance(box, list) or len(box) != 4:
        raise IsolationError("box must be four integers")
    x0, y0, x1, y1 = (_int_coord(v, "box coordinate") for v in box)
    w, h = IMAGE_SIZE
    if not (0 <= x0 < x1 <= w - 1 and 0 <= y0 < y1 <= h - 1):
        raise IsolationError("box corners must be ordered and inside the 640x480 image")
    if (x0, y0, x1, y1) != BOX_XYXY:
        raise IsolationError("box differs from the frozen whole-box prompt")
    sam_box = doc["sam_box_xyxy"]
    if (not isinstance(sam_box, list) or len(sam_box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in sam_box)
            or [float(v) for v in sam_box] != [v + 0.5 for v in BOX_XYXY]):
        raise IsolationError("recorded SAM box must be the integer box + 0.5 exactly once")
    for name in ("positive_points", "negative_points"):
        if not isinstance(doc[name], list) or any(not isinstance(p, list) or len(p) != 2 for p in doc[name]):
            raise IsolationError(f"{name} must be a list of [x, y] pairs")
    pos = [[_int_coord(c, "positive point") for c in p] for p in doc["positive_points"]]
    neg = [[_int_coord(c, "negative point") for c in p] for p in doc["negative_points"]]
    if [tuple(p) for p in pos] != [tuple(int(c) for c in p) for p in iso.POSITIVE] or \
            [tuple(p) for p in neg] != [tuple(int(c) for c in p) for p in iso.NEGATIVE]:
        raise IsolationError("points differ from the exact original positive/negative points")
    for x, y in pos + neg:
        if not (0 <= x < w and 0 <= y < h):
            raise IsolationError("a point lies outside the image")
    if not all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in pos):
        raise IsolationError("every positive centre must lie within the box")
    negatives_inside = [[x, y] for x, y in neg if x0 <= x <= x1 and y0 <= y <= y1]
    return {"prompt_id": PROMPT_ID, "box_xyxy": list(BOX_XYXY), "positive": pos, "negative": neg,
            "negatives_inside_box": negatives_inside, "review": doc["parent_opening_overlay_review"]}


def box_call_arguments() -> dict:
    """The exact ``add_new_points_or_box`` arguments (pure numpy; +0.5 applied once)."""
    points = np.array([[x + 0.5, y + 0.5] for x, y in iso.POSITIVE + iso.NEGATIVE], np.float32)
    labels = np.array([1] * len(iso.POSITIVE) + [0] * len(iso.NEGATIVE), np.int32)
    box = np.array([v + 0.5 for v in BOX_XYXY], np.float32)
    return {"frame_idx": iso.PROMPT_LOCAL, "obj_id": 1, "points": points, "labels": labels, "box": box,
            "clear_old_points": True, "normalize_coords": True}


def describe_call(kwargs: Mapping[str, Any]) -> dict:
    """JSON-safe record of one actual call (dtypes and shapes included)."""
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            out[key] = {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        else:
            out[key] = value
    return out


def expected_call_record() -> dict:
    return describe_call(box_call_arguments())


def cache_identity() -> dict:
    body = {"prompt_sha256": PROMPT_SHA256, "prompt_id": PROMPT_ID, "call": expected_call_record(),
            "original_plan_sha256": ORIGINAL_PLAN_SHA256, "accepted_isolation_sha256": ACCEPTED_ISOLATION_SHA256}
    return {**body, "sha256": hashlib.sha256(iso.canonical_bytes(body)).hexdigest()}


# --------------------------------------------------------------------------
# adapter: the real SamAdapter with ONLY add_prompt overridden


class BoxSamAdapter(iso.SamAdapter):
    """Adds the frozen box together with the original points; records every actual call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def add_prompt(self, state, index, points, labels):
        call = box_call_arguments()
        if index != call["frame_idx"] or not np.array_equal(np.asarray(points), call["points"]) or \
                not np.array_equal(np.asarray(labels), call["labels"]):
            raise IsolationError("engine prompt differs from the frozen original points/labels")
        self.calls.append(describe_call(call))
        with self.torch.inference_mode():
            self.predictor.add_new_points_or_box(state, frame_idx=call["frame_idx"], obj_id=call["obj_id"],
                                                 points=call["points"], labels=call["labels"], box=call["box"],
                                                 clear_old_points=call["clear_old_points"],
                                                 normalize_coords=call["normalize_coords"])


# --------------------------------------------------------------------------
# NEW qualifier (pure; never compares with the original seed)


def qualify_box(masks: Mapping[int, np.ndarray], *, actual_calls: Sequence[Mapping], duplicate_seed_equal: bool,
                preview_local: Sequence[int], prompt_index: int = iso.PROMPT_LOCAL,
                scored: tuple[int, int] = iso.SCORED_LOCAL, positive: Sequence = iso.POSITIVE,
                negative: Sequence = iso.NEGATIVE, thresholds: Mapping[str, Any] = THRESHOLDS) -> dict:
    lo, hi = scored
    if sorted(masks) != list(range(hi + 1)):
        raise IsolationError("qualification needs every local mask 0..119")
    seed = iso._bool_mask(masks[prompt_index])
    area = int(seed.sum())
    largest = iso.largest_component_fraction(seed, thresholds["connectivity"])
    inside = [bool(seed[int(y), int(x)]) for x, y in positive]
    outside = [not bool(seed[int(y), int(x)]) for x, y in negative]
    expected = expected_call_record()
    gates = {
        "prompt_call_agreement": {"calls": len(actual_calls),
                                  "passed": len(actual_calls) == 2 and all(dict(c) == expected for c in actual_calls)},
        "duplicate_seed_exact": {"passed": duplicate_seed_equal is True},
        "seed_nonempty": {"area": area, "passed": area > 0},
        "seed_largest_component": {"value": largest, "threshold": thresholds["seed_largest_component_min"],
                                   "passed": area > 0 and largest >= thresholds["seed_largest_component_min"]},
        "seed_prompts": {"positive_inside": inside, "negative_outside": outside, "passed": all(inside) and all(outside)},
        "preview_ids": {"local": list(preview_local), "passed": tuple(preview_local) == iso.PREVIEW_LOCAL},
    }
    min_area = thresholds["scored_area_min_fraction_of_new_seed"] * area
    areas = {i: int(iso._bool_mask(masks[i], seed.shape).sum()) for i in range(lo, hi + 1)}
    below = [i for i, a in areas.items() if a < min_area]
    gates["scored_area"] = {"min_area": min_area, "frames_below": below, "min_observed": min(areas.values()),
                            "passed": area > 0 and not below}
    pairs = [(i, i + 1, iso.iou(masks[i], masks[i + 1])) for i in range(lo, hi)]      # 109 scored pairs only
    failing = [(a, b, v) for a, b, v in pairs if v < thresholds["scored_consecutive_iou_min"]]
    gates["scored_consecutive_iou"] = {"pairs": len(pairs), "threshold": thresholds["scored_consecutive_iou_min"],
                                       "min_observed": min(v for _a, _b, v in pairs), "failing_pairs": failing,
                                       "passed": not failing and len(pairs) == hi - lo}
    return {"qualified": all(g["passed"] for g in gates.values()), "gates": gates,
            "original_seed_equality": {"applicable": False,
                                       "reason": "changed prompt: equality with the original s000370 seed and area "
                                                 "18329 is inapplicable and is NOT used as a gate"},
            "new_seed": {"area": area, "content_sha256": iso.mask_content_sha256(seed), "descriptive_only": True},
            "scored_areas": areas, "consecutive_iou": [[a, b, v] for a, b, v in pairs], "thresholds": dict(thresholds),
            "scope": "descriptive consistency only; not independent accuracy, not qualitative review"}


# --------------------------------------------------------------------------
# plan


def verify_accepted_source() -> str:
    path = Path(iso.__file__).resolve()
    if path != (ROOT / "tools/workbench_sam_forward_isolation.py").resolve():
        raise IsolationError("imported isolation module is not the accepted project file")
    actual = iso.sha256_file(path)
    if actual != ACCEPTED_ISOLATION_SHA256:
        raise IsolationError("accepted isolation tool differs from its frozen SHA-256")
    return actual


def build_plan(workspace: Path, plan_id: str, device: str, *, allow_loaded_torch: bool = False) -> dict:
    """Deterministic; rebuilds the ORIGINAL control plan's full closure against its immutable hash."""
    from bench import workbench_cli as cli, workbench_contract as wc
    wc._ident(plan_id, "plan_id")
    verify_accepted_source()
    ws = cli.Workspace(Path(workspace))
    original_path = ws.derived_dir / iso.EXPERIMENT_DIR / "plans" / ORIGINAL_PLAN_ID / "plan.json"
    original_raw = original_path.read_bytes()
    if iso.sha256_bytes(original_raw) != ORIGINAL_PLAN_SHA256:
        raise IsolationError("original control plan differs from its immutable hash")
    rebuilt = iso.build_plan(Path(workspace), ORIGINAL_PLAN_ID, device, allow_loaded_torch=allow_loaded_torch)
    if iso.canonical_bytes(rebuilt) != original_raw:
        raise IsolationError("original control plan closure no longer reproduces (sources/runtime/SAM/inputs changed)")
    original = iso.loads_strict(original_raw, "original plan")
    prompt_path = iso._contained(ws.derived_dir, ROOT / PROMPT_FILE)
    prompt_raw = prompt_path.read_bytes()
    if iso.sha256_bytes(prompt_raw) != PROMPT_SHA256:
        raise IsolationError("frozen opening-box prompt differs from its pin")
    prompt = validate_prompt(iso.loads_strict(prompt_raw, "prompt"))
    seed_frame = original["clip"]["frames"][iso.PROMPT_LOCAL]
    if (seed_frame["frame_id"] != iso.PROMPT_FRAME_ID or seed_frame["file_sha256"] != RGB370_FILE_SHA256
            or seed_frame["decoded_sha256"] != RGB370_DECODED_SHA256):
        raise IsolationError("RGB s000370 identity differs between the prompt and the original plan")
    out_rel = iso._project_relative(ws.derived_dir / EXPERIMENT_DIR / "runs" / plan_id)
    return {
        "schema": PLAN_SCHEMA, "plan_id": plan_id, "tool": "tools/workbench_sam_opening_box.py",
        "tool_sha256": iso.sha256_file(Path(__file__)), "accepted_isolation_sha256": ACCEPTED_ISOLATION_SHA256,
        "workspace": iso._project_relative(ws.path), "planned_output": out_rel,
        "original_plan": {"path": iso._project_relative(original_path), "sha256": ORIGINAL_PLAN_SHA256,
                          "rebuilt_byte_identical": True},
        "prompt": {"path": PROMPT_FILE, "sha256": PROMPT_SHA256, **prompt, "call": expected_call_record()},
        "cache_identity": cache_identity(),
        "clip": original["clip"],
        "sam": {k: original["sam"][k] for k in ("checkpoint", "checkpoint_sha256", "config", "sources_root", "settings")},
        "runtime": original["runtime"], "state_isolation": original["state_isolation"],
        "thresholds": dict(THRESHOLDS), "preview": original["preview"],
        "budgets": {"output_cap_bytes_all_new_outputs": OUTPUT_CAP_BYTES,
                    **{k: v for k, v in original["budgets"].items() if k != "output_cap_bytes"}},
        "flags": dict(FLAGS),
        "question": "does the frozen opening whole-box prompt (with the original points) give consistent forward "
                    "masks under the accepted state isolation?",
        "hypothesis_status": "conditional diagnostic; no accuracy, phone or automatic-geometry claim",
    }


def validate_plan_shape(plan: Any) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise IsolationError("not an opening-box plan")
    clip = plan.get("clip") or {}
    frames = clip.get("frames")
    if (not isinstance(frames, list) or [f.get("frame_id") for f in frames] != iso.frame_ids_for()
            or [f.get("local") for f in frames] != list(range(iso.COUNT)) or clip.get("prompt_local") != iso.PROMPT_LOCAL):
        raise IsolationError("plan frame accounting differs from the frozen 120-frame clip")
    if plan.get("thresholds") != THRESHOLDS or plan.get("flags") != FLAGS:
        raise IsolationError("plan thresholds/flags differ from the frozen values")
    if (plan.get("preview") or {}).get("local") != list(iso.PREVIEW_LOCAL):
        raise IsolationError("plan preview indices differ")
    prompt = plan.get("prompt") or {}
    if (prompt.get("sha256") != PROMPT_SHA256 or prompt.get("prompt_id") != PROMPT_ID
            or prompt.get("box_xyxy") != list(BOX_XYXY) or prompt.get("call") != expected_call_record()):
        raise IsolationError("plan prompt/call differs from the frozen prompt")
    if plan.get("cache_identity") != cache_identity():
        raise IsolationError("plan cache identity differs (prompt SHA/coordinates)")
    if (plan.get("original_plan") or {}).get("sha256") != ORIGINAL_PLAN_SHA256 or \
            plan.get("accepted_isolation_sha256") != ACCEPTED_ISOLATION_SHA256:
        raise IsolationError("plan does not bind the original control plan / accepted isolation source")


def prepare(workspace: Path, plan_id: str, device: str = "cuda") -> dict:
    from bench import workbench_cli as cli
    plan = build_plan(Path(workspace), plan_id, device)
    validate_plan_shape(plan)
    ws = cli.Workspace(Path(workspace))
    data = iso.canonical_bytes(plan)
    writer = iso.OutputWriter(ws.derived_dir / EXPERIMENT_DIR / "plans" / plan_id, cap_bytes=OUTPUT_CAP_BYTES,
                              budget_check=lambda n: ws.require_budget(extra_derived=n))
    writer.create()
    sha = writer.write("plan.json", data)
    return {"plan": iso._project_relative(writer.dir / "plan.json"), "plan_sha256": sha,
            "planned_output": plan["planned_output"], "run_output_cap_bytes": OUTPUT_CAP_BYTES - len(data)}


def _load_plan(plan_path: Path, expected_sha256: str) -> tuple[dict, bytes]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or \
            any(c not in "0123456789abcdef" for c in expected_sha256):
        raise IsolationError("--expected-plan-sha256 must be a lowercase SHA-256")
    path = Path(plan_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > OUTPUT_CAP_BYTES:
        raise IsolationError("plan must be a regular bounded file")
    raw = path.read_bytes()
    if iso.sha256_bytes(raw) != expected_sha256:
        raise IsolationError("plan differs from the externally expected SHA-256")
    plan = iso.loads_strict(raw, "plan")
    validate_plan_shape(plan)
    if iso.canonical_bytes(plan) != raw:
        raise IsolationError("plan file is not in canonical form")
    return plan, raw


# --------------------------------------------------------------------------
# run (explicit parent-scheduled single-GPU execution)


def run(plan_path: Path, expected_plan_sha256: str) -> dict:
    from bench import workbench_cli as cli, workbench_neural as wn, workbench_telemetry as wt
    plan, raw = _load_plan(plan_path, expected_plan_sha256)
    workspace = ROOT / plan["workspace"]
    if iso.canonical_bytes(build_plan(workspace, plan["plan_id"], plan["runtime"]["device"])) != raw:
        raise IsolationError("reconstructed plan differs (original closure, prompt, sources or runtime changed)")
    ws = cli.Workspace(workspace)
    out_dir = ROOT / plan["planned_output"]
    iso._contained(ws.derived_dir / EXPERIMENT_DIR / "runs", out_dir)
    if "sam2" in sys.modules:
        raise IsolationError("a SAM 2 module was imported before the planned root was installed")
    clip = ws.load_clip(iso.CLIP_ID)
    ids = [f["frame_id"] for f in plan["clip"]["frames"]]
    cap = OUTPUT_CAP_BYTES - len(raw)                      # the plan counts against the 8 MiB for all new outputs
    writer = iso.OutputWriter(out_dir, cap_bytes=cap, budget_check=lambda n: ws.require_budget(extra_derived=n))
    ws.require_budget(extra_derived=cap)
    writer.create()
    ledger = iso.IsolationLedger(ids)
    stage, manifest_masks, outer_errors, adapter_calls = "setup", {}, [], []
    try:
        import torch
        sam_root = ROOT / plan["sam"]["sources_root"]
        sys.path.insert(0, str(sam_root))
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        from sam2.modeling.position_encoding import PositionEmbeddingRandom  # type: ignore
        import sam2.sam2_video_predictor as predictor_module  # type: ignore
        from bench.quality_sam2 import deterministic_position_grid
        imported = iso._check_imported_sam(sam_root)
        call = box_call_arguments()

        def write_mask(index, frame_id, mask, label):
            data = iso.encode_mask_png(mask)
            if not np.array_equal(iso.decode_mask_png(data, (clip.height, clip.width)), mask):
                raise IsolationError(f"mask PNG round trip differs for {frame_id}")
            writer.write(f"m{index:05d}.png", data)
            manifest_masks[frame_id] = iso.mask_record(index, mask, data, label, (clip.height, clip.width))

        samples = [wt.resource_sample("before_gpu_stage", disk_path=ws.path)]
        stage = "gpu_stage"
        original_forward = PositionEmbeddingRandom.forward
        result = None
        with wt.gpu_stage("sam_opening_box"):
            PositionEmbeddingRandom.forward = deterministic_position_grid
            predictor = adapter = None
            try:
                stage = "build_predictor"
                predictor = build_sam2_video_predictor(
                    plan["sam"]["config"], str(ROOT / plan["sam"]["checkpoint"]), device=plan["runtime"]["device"],
                    apply_postprocessing=False, vos_optimized=False)
                imported = iso._check_imported_sam(sam_root)
                adapter = BoxSamAdapter(ws, clip, plan, predictor, predictor_module, torch, wn, cli)
                adapter_calls = adapter.calls
                stage = "state_isolated_propagation"
                result = iso.run_state_isolated(adapter, frame_ids=ids, prompt_index=iso.PROMPT_LOCAL,
                                                points=call["points"], labels=call["labels"],
                                                shape=(clip.height, clip.width), write_mask=write_mask, ledger=ledger)
                samples.append(wt.resource_sample("after_propagation", disk_path=ws.path))
            finally:
                # Outer cleanup INSIDE the GPU stage; every step attempted independently, never raising here.
                try:
                    PositionEmbeddingRandom.forward = original_forward
                except Exception as exc:  # noqa: BLE001
                    outer_errors.append(f"position grid restore: {type(exc).__name__}: {exc}")
                if adapter is not None:
                    outer_errors.extend(adapter.close_all())
                adapter = predictor = None
                try:
                    ledger.outer["final_device_release"] = iso._plain(wn._release_device_memory(torch))
                except Exception as exc:  # noqa: BLE001
                    outer_errors.append(f"final device release: {type(exc).__name__}: {exc}")
                ledger.outer["errors"] = list(outer_errors)
                ledger.outer["actual_prompt_calls"] = list(adapter_calls)
                ledger.cleanup_errors.extend(outer_errors)
        if outer_errors:
            raise IsolationError(f"outer predictor/device cleanup incomplete: {outer_errors}")
        stage = "preview"
        masks = {row["local"]: iso.decode_mask_png((out_dir / row["path"]).read_bytes(), (clip.height, clip.width))
                 for row in manifest_masks.values()}
        writer.write("preview.png", iso._preview_png(ws, clip, plan, masks))
        writer.write("events.jsonl", b"".join(iso.canonical_bytes(e) + b"\n" for e in ledger.events))
        manifest = {"schema": iso.MASK_SCHEMA, "clip_id": iso.CLIP_ID, "mask_set": f"sam2-opening-box-{plan['plan_id']}",
                    "producer": {"kind": PRODUCER_KIND, "checkpoint_sha256": plan["sam"]["checkpoint_sha256"],
                                 "config": plan["sam"]["config"], "device": plan["runtime"]["device"],
                                 "prompt_id": PROMPT_ID, "prompt_sha256": PROMPT_SHA256, "box_xyxy": list(BOX_XYXY),
                                 "selection": {"frame_id": iso.PROMPT_FRAME_ID,
                                               "positive": [list(p) for p in iso.POSITIVE],
                                               "negative": [list(p) for p in iso.NEGATIVE], "box": list(BOX_XYXY)},
                                 "prompt_frame_id": iso.PROMPT_FRAME_ID, "point_convention": iso.POINT_CONVENTION,
                                 "cache_identity_sha256": plan["cache_identity"]["sha256"],
                                 "state_isolation": plan["state_isolation"], "plan_sha256": expected_plan_sha256},
                    "evaluator_inputs_read": False, "cancelled": False, "masks": manifest_masks,
                    "telemetry": {"segmentation_ms_by_state_frame": result["timings_ms"],
                                  "process_peak_rss_bytes": wt.process_peak_rss_bytes(), **wt.gpu_peak_bytes(),
                                  "resource_samples": samples},
                    "note": "private diagnostic staging output; NOT registered as a workspace mask set"}
        writer.write("manifest.json", iso.canonical_bytes(manifest))
        stage = "closure_after"
        if iso.canonical_bytes(build_plan(workspace, plan["plan_id"], plan["runtime"]["device"],
                                          allow_loaded_torch=True)) != raw:
            raise IsolationError("original closure/prompt/source/runtime changed during the run")
        if iso._check_imported_sam(sam_root) != imported:
            raise IsolationError("imported SAM 2 modules changed during the run")
        stage = "readback_verification"             # 123 artifacts: 120 masks + events + preview + manifest
        verified = iso.verify_staged_outputs(out_dir, recorded=dict(writer.files), frame_ids=ids,
                                             prompt_index=iso.PROMPT_LOCAL, shape=(clip.height, clip.width))
        if not np.array_equal(verified["masks"][iso.PROMPT_LOCAL], result["forward_seed"]):
            raise IsolationError("verified seed differs from the forward seed")
        stage = "qualification"
        duplicate_equal = any(e.get("event") == "duplicate_seed_equal" for e in ledger.events)
        qualification = qualify_box(verified["masks"], actual_calls=adapter_calls, duplicate_seed_equal=duplicate_equal,
                                    preview_local=plan["preview"]["local"])
        stage = "receipt"
        receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "plan_sha256": expected_plan_sha256,
                   "prompt_sha256": PROMPT_SHA256, "cache_identity": plan["cache_identity"],
                   "actual_prompt_calls": list(adapter_calls), "counters": result["counters"],
                   "expected_counters": iso.EXPECTED_COUNTERS, "ledger": ledger.snapshot(),
                   "imported_sam_modules": imported, "readback_verification": verified["proof"],
                   "qualification": qualification, "qualified_for_parent_review": qualification["qualified"],
                   "closure": {"before_equals_plan": True, "after_equals_plan": True,
                               "original_plan_rebuilt_byte_identical": True},
                   "flags": dict(FLAGS),
                   "receipt_hash": "not self-hashed; the parent hashes receipt.json externally after the process",
                   "qualitative_review": "NOT done; mandatory separate parent review of preview.png",
                   "hypothesis": "conditional diagnostic; no accuracy, phone or automatic-geometry claim; any failed "
                                 "gate or unacceptable preview means no registration and no neural replay"}
        writer.write("receipt.json", iso.canonical_bytes(receipt))
        return {"directory": iso._project_relative(out_dir), "qualified": qualification["qualified"],
                "receipt_sha256_at_write": writer.files["receipt.json"]}
    except BaseException as exc:
        failure = {"schema": RECEIPT_SCHEMA, "status": "failed", "stage": stage,
                   "primary_error": f"{type(exc).__name__}: {exc}"[:4000],
                   "cleanup_errors": list(getattr(exc, "isolation_cleanup_errors", [])) + list(ledger.cleanup_errors),
                   "plan_sha256": expected_plan_sha256, "prompt_sha256": PROMPT_SHA256,
                   "actual_prompt_calls": list(adapter_calls), "ledger": ledger.snapshot(),
                   "expected_frame_ids": list(ids), "written_frame_ids": sorted(manifest_masks),
                   "missing_frame_ids": [f for f in ids if f not in manifest_masks],
                   "partial_mask_identities": manifest_masks, "outputs_recorded": dict(writer.files),
                   "flags": dict(FLAGS), "note": "incomplete; no qualification, registration or neural replay"}
        try:
            writer.write("failure.json", iso.canonical_bytes(failure))
        except BaseException as persist_exc:  # noqa: BLE001 - reserve/cap/CUDA may prevent persistence
            failure["failure_json_persisted"] = False
            failure["persist_error"] = f"{type(persist_exc).__name__}: {persist_exc}"[:1000]
            try:
                sys.stderr.write("SAM opening-box diagnostic FAILED and failure.json could not be written; "
                                 "terminal record:\n" + json.dumps(failure, sort_keys=True, default=str) + "\n")
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
        raise


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opening-only box-prompt SAM 2.1 diagnostic (state-isolated).")
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
