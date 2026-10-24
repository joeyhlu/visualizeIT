"""HB scene 000003 / CAD 4 bounded METADATA assembly (fifth distinct recording; prepared metric control).

This assembles exactly four JSON files from ALREADY accepted media and the separately accepted actual controlled
seed, all named by ONE parent-frozen registry (``hb-scene3-assembly-inputs-v1.json``, pinned by path + SHA-256 +
size). It is a prepared metric control for head 27e06a51..., NOT automatic geometry, NOT an evaluator-free run,
NOT head accuracy, NOT mobile/phone proof. It never opens GT, the raw camera file, RGB PNGs or the source PLY,
never decodes the GLB geometry or the video, never runs a model/GPU/SAM, never registers or admits anything, and
never copies or links the GLB/video (they are only sealed by bounded streaming SHA-256 at their accepted sizes).

    inspect_assembly(ctx)            -> authenticate everything and build the outputs in memory (no writes)
    assemble(ctx, *, out)            -> publish metric-bundle / clip-proposal / execution-control-sidecar, then the
                                        assembly receipt LAST; returns the receipt hash after post-publication closure
    validate_output(ctx, *, out)     -> re-authenticate, rebuild the expected outputs and re-verify the published ones

No caller-supplied hash/object/frame/ordinal/K/origin/seed can create trust: the registry pin is a source
constant; ``AssemblyContext.test_only_registry_pin`` (API only) labels everything TEST_ONLY and production refuses
it. Trust: historical local integrity (``upstream_authenticated: false``). Exit codes: 0 ok; 1 failure; 2 usage;
3 blocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (checked helpers; never mutated)
from tools import workbench_hb_scene3_profile as pf  # noqa: E402  (schema constants only)
from tools import workbench_hb_scene3_media as media  # noqa: E402  (sealed reads, publisher, counter; stages never called)
from tools import workbench_hb_scene3_seed as sd  # noqa: E402  (schema constants + reviewed quarantine; stages never called)
from bench import workbench_contract as wc  # noqa: E402
from bench import workbench_neural as wn  # noqa: E402  (MetricBundle loader / native_K only; nothing executed)

TOOL_VERSION = "1"
SCHEMA_REGISTRY = "workbench_hb_scene3_assembly_inputs_v1"
SCHEMA_SIDECAR = "workbench_hb_scene3_execution_control_sidecar_v1"
SCHEMA_RECEIPT = "workbench_hb_scene3_assembly_receipt_v1"
SCHEMA_FAILURE = "workbench_hb_scene3_assembly_failure_v1"
BUNDLE_SCHEMA = "workbench_metric_bundle_v1"
REGISTRY_PIN = {"path": ".cache/video-workbench/derived/validation/hb-scene3-assembly-inputs-v1.json",
                "sha256": "c9ea5036252308cb795bf02cf9199a8fb22388c0f3b914547a20aef92bb01a21", "bytes": 9156}
HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
SCENE, OBJECT_ID, FAMILY = "000003", 4, "hb_primesense_640x480"
FRAME_COUNT, WIDTH, HEIGHT = 120, 640, 480
OPENING, SCORING = tuple(range(0, 10)), tuple(range(10, 120))
SEED_FRAME_ID, FIRST_SCORED_FRAME_ID = "s000009", "s000010"
BUNDLE_NAME, CLIP_NAME = "metric-bundle.json", "clip-proposal.json"
SIDECAR_NAME, RECEIPT_NAME = "execution-control-sidecar.json", "assembly-receipt.json"
OUTPUT_NAMES = (BUNDLE_NAME, CLIP_NAME, SIDECAR_NAME, RECEIPT_NAME)
FAILURE_NAME = "failure.json"
REGISTRY_CAP = 64 << 10
SUBTREE_CAP = 96 << 10
OUTPUT_CAPS = {BUNDLE_NAME: 4 << 10, CLIP_NAME: 24 << 10, SIDECAR_NAME: 8 << 10, RECEIPT_NAME: 24 << 10}
FAILURE_CAP = 4 << 10
MAX_INPUT_CAPS = {"metadata": 256 << 10, "geometry_derivative": 4 << 20, "video_derivative": 6 << 20}
ARTWORK = {"kind": "checkerboard", "edge_px": 256, "size_fraction": 0.25, "max_faces": 200_000,
           "generation": "fixed_builtin_v1"}
PORTABLE_TRIANGLE_TARGET = 20_000
METADATA_KEYS = ("prerequisites", "allowlist", "model_receipt", "video_receipt", "profile", "camera_k", "association",
                 "seed_policy", "seed_plan", "controlled_seed", "seed_binding", "seed_source_acceptance",
                 "actual_seed_plan", "actual_seed_extraction", "actual_seed_acceptance", "actual_media",
                 "origin_review", "origin_independent_review", "opening_placement")
IN_SCOPE_KEYS = ("prerequisites", "allowlist", "model_receipt", "video_receipt", "profile", "camera_k", "seed_plan",
                 "controlled_seed", "seed_binding", "glb", "video")
INPUT_KINDS = {**{k: "metadata" for k in METADATA_KEYS}, "glb": "geometry_derivative", "video": "video_derivative"}
REVIEW_KEYS = ("seed_source_acceptance", "actual_seed_plan", "actual_seed_extraction", "actual_seed_acceptance",
               "actual_media", "origin_review", "origin_independent_review", "opening_placement")
SOURCE_FILES = ("tools/workbench_hb_scene3_seed.py", "tools/workbench_hb_scene3_media.py",
                "tools/workbench_hb_scene3_profile.py", "tools/workbench_hb_camera_k.py",
                "tools/workbench_hb_metric_inputs.py", "tools/workbench_hb_opening_seed.py", "bench/workbench_neural.py",
                "bench/workbench_geometry.py", "bench/workbench_contract.py")
EXPECTED_CONSTRAINTS = {
    "assembly_subtree_cap_bytes": SUBTREE_CAP, "seed_subtree_cap_bytes": 65536, "scope_cap_bytes": 12582912,
    "scope_metadata_cap_bytes": 524288, "derived_cap_bytes": 1073741824, "free_floor_bytes": 1073741824,
    "new_inputs_cap_bytes": 250000000, "no_asset_copy_or_link": True, "annotation_opens": 0, "raw_camera_opens": 0,
    "raw_rgb_or_ply_opens": 0, "neural_execution": False, "registration": False, "serving_admission": False,
    "origin": "integer_centers", "origin_evidence": "reviewed_assumption", "physical_calibration_verified": False,
    "fixed_point_xy": [320, 340], "fixed_roi_xyxy": [285, 270, 370, 440], "seed_frame_id": SEED_FRAME_ID,
    "first_scored_frame_id": FIRST_SCORED_FRAME_ID, "selection_reviewed": False, "observed_mask_review_pending": True,
    "frame_count": FRAME_COUNT, "physical_fps": None, "presentation_fps": 30, "mesh_triangles": 118136,
    "portable_triangle_target_passed": False}
FLAGS = {"prepared_metric_control": True, "controlled_ground_truth_seed": True, "automatic_geometry": False,
         "evaluator_free": False, "independent_accuracy_claim": False, "head_accuracy_claim": False,
         "phone_performance_claim": False, "mobile_verified": False, "physical_fps_claim": False,
         "annotation_opens": 0, "raw_camera_opens": 0, "raw_rgb_or_ply_opens": 0, "neural_executed": False,
         "gpu_used": False, "sam_executed": False, "segmentation": False, "registered": False, "admitted": False,
         "asset_copied_or_linked": False, "head_sha256": HEAD_SHA256, "family": FAMILY}
TEST_ONLY_FAMILY = f"TEST_ONLY_{FAMILY}"


def output_family(label: str) -> str:
    """Production keeps the accepted family; any synthetic run carries an unmistakable UNADMITTED family through
    every output (bundle, clip provenance, sidecar, flags, receipt), so a TEST_ONLY bundle cannot be detached
    from its sidecar and mistaken for production."""
    return FAMILY if label == "PRODUCTION_PINNED" else TEST_ONLY_FAMILY


def output_flags(label: str) -> dict:
    return {**FLAGS, "family": output_family(label), "label": label}


UNRESOLVED = ["observed_sam_review", "selection_review", "registration", "serving_admission",
              "runtime_execution_and_evaluation", "physical_principal_point_calibration"]
GT_DISCLOSURE = ("PRIOR CONTROLLED GROUND-TRUTH INITIALISATION: the accepted controlled seed was derived earlier from "
                 "the historical BOP scene_gt.json (all bytes read and hashed in four bounded passes; only the frame-9 "
                 "obj_id 4 pose interpreted). This clip is NOT evaluator-free. evaluator_inputs_read:false refers only "
                 "to this clip's own input path (the video derivative); the assembly itself opened no annotation")


class ExecutionControlError(mi.InputsError):
    """A registry/input/relationship/output contradiction (fail closed)."""


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ExecutionControlError(message)


def _same(value: Any, expected: Any, what: str) -> None:
    """Exact equality with JSON types (true vs 1, 1 vs 1.0 differ)."""
    if mi.dump_json(value) != mi.dump_json(expected):
        raise ExecutionControlError(f"{what} contradicts the frozen registry / accepted evidence")


# --------------------------------------------------------------------------
# context


@dataclass(frozen=True)
class SyntheticRegistryPin:
    """TEST-ONLY registry pin (API only; never settable from the CLI). The registry must then carry
    ``label: TEST_ONLY`` and every output is labelled TEST_ONLY; production refuses it."""
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class AssemblyContext:
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"
    test_only_registry_pin: SyntheticRegistryPin | None = None


def _pin(ctx: AssemblyContext) -> dict:
    s = ctx.test_only_registry_pin
    if s is None:
        return {**REGISTRY_PIN, "label": "PRODUCTION_PINNED"}
    if (not isinstance(s, SyntheticRegistryPin) or not isinstance(s.path, str) or not mi._is_sha(s.sha256)
            or type(s.bytes) is not int or not 0 < s.bytes <= REGISTRY_CAP):
        raise mi.UsageError("test-only registry pin is malformed")
    return {"path": s.path, "sha256": s.sha256, "bytes": s.bytes, "label": "TEST_ONLY"}


# --------------------------------------------------------------------------
# bounded authenticated reads


def _rel_path(value: Any, what: str) -> str:
    _need(isinstance(value, str) and value and "\\" not in value and ":" not in value and not value.startswith("/")
          and all(p not in ("", ".", "..") for p in value.split("/")), f"{what}: path must be repository-relative POSIX")
    return value


def _contained(ctx: AssemblyContext, rel: str, what: str) -> Path:
    """Regular, link-free (every ancestor from the repository root), inside the repository."""
    mctx = mi.Context(root=ctx.root, workspace=ctx.workspace)
    path = mi.contained_input(mctx, _rel_path(rel, what), what)
    mi._no_links(Path(os.path.abspath(ctx.root)), path)
    return path


def _read_metadata(path: Path, *, expected_sha: str, expected_bytes: int, cap: int, what: str) -> tuple[bytes, Any]:
    st = os.lstat(path)
    _need(stat.S_ISREG(st.st_mode) and st.st_size == expected_bytes and expected_bytes <= cap,
          f"{what}: not a regular file of the registered size within its cap")
    raw = mi.read_bounded(path, cap, what)
    _need(len(raw) == expected_bytes and mi.sha256_bytes(raw) == expected_sha,
          f"{what}: bytes differ from the registry (checked before parsing)")
    return raw, mi.strict_json(raw, what)


def _seal_derivative(path: Path, *, expected_sha: str, expected_bytes: int, cap: int, what: str) -> None:
    """Bounded streaming SHA-256 at the accepted size; never decoded, copied or linked."""
    st = os.lstat(path)
    _need(stat.S_ISREG(st.st_mode) and st.st_size == expected_bytes and expected_bytes <= cap,
          f"{what}: not a regular file of the registered size within its cap")
    _need(media.sealed_sha256(path, expected_bytes) == expected_sha, f"{what}: sealed SHA-256 differs from the registry")


def _sources() -> dict:
    extra = {rel: mi.sha256_file(ROOT / rel) for rel in ("bench/workbench_neural.py", "bench/workbench_geometry.py",
                                                           "bench/workbench_contract.py")}
    return dict(sorted({**sd._sources(), **extra,
                        Path(__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(__file__))}.items()))


# --------------------------------------------------------------------------
# registry and relations (pure checks over authenticated documents)


def validate_registry(doc: Any, *, label: str) -> dict:
    _need(isinstance(doc, dict) and set(doc) == {"schema", "timestamp_utc", "reviewer", "label", "scene", "object_id",
                                                 "family", "head_sha256", "preparation_scope", "output_directory",
                                                 "inputs", "accepted_source_versions", "trust", "constraints"},
          "registry has missing or unknown fields")
    _same(doc["schema"], SCHEMA_REGISTRY, "registry schema")
    _same(doc["label"], label, "registry label (a TEST_ONLY registry can never be production)")
    _same([doc["scene"], doc["object_id"], doc["family"], doc["head_sha256"]], [SCENE, OBJECT_ID, FAMILY, HEAD_SHA256],
          "registry scene/object/family/head")
    _same(doc["trust"], {"trust_level": "historical_local_integrity", "upstream_authenticated": False}, "registry trust")
    _same(doc["constraints"], EXPECTED_CONSTRAINTS, "registry constraints")
    scope = _rel_path(doc["preparation_scope"], "preparation_scope")
    _same(doc["output_directory"], f"{scope}/assembly-v1", "registry output directory")
    _need(isinstance(doc["timestamp_utc"], str) and isinstance(doc["reviewer"], str) and doc["reviewer"].strip(),
          "registry timestamp/reviewer malformed")
    inputs = doc["inputs"]
    _need(isinstance(inputs, Mapping) and set(inputs) == set(INPUT_KINDS), "registry inputs differ from the frozen set")
    for key, item in inputs.items():
        _need(isinstance(item, Mapping) and set(item) == {"path", "sha256", "bytes", "cap_bytes", "kind"},
              f"registry input {key} malformed")
        _rel_path(item["path"], key)
        _need(mi._is_sha(item["sha256"]) and item["kind"] == INPUT_KINDS[key]
              and type(item["bytes"]) is int and type(item["cap_bytes"]) is int
              and 0 < item["bytes"] <= item["cap_bytes"] <= MAX_INPUT_CAPS[item["kind"]],
              f"registry input {key} hash/kind/size/cap invalid")
        inside = PurePosixPath(scope) in PurePosixPath(item["path"]).parents
        _need(inside == (key in IN_SCOPE_KEYS), f"registry input {key} lies on the wrong side of the preparation scope")
    versions = doc["accepted_source_versions"]
    _need(isinstance(versions, Mapping) and set(versions) == set(SOURCE_FILES)
          and all(mi._is_sha(v) for v in versions.values()), "registry accepted source versions malformed")
    return doc


def _check_relations(reg: Mapping, docs: Mapping, *, label: str, root: Path) -> dict:
    """Every accepted record must agree with the registry and with each other. Returns the derived facts."""
    inputs = reg["inputs"]
    sha = {k: v["sha256"] for k, v in inputs.items()}
    path = {k: v["path"] for k, v in inputs.items()}
    c = EXPECTED_CONSTRAINTS
    pre, allow, model, video = docs["prerequisites"], docs["allowlist"], docs["model_receipt"], docs["video_receipt"]
    # accepted media chain
    _same([pre.get("schema"), pre.get("status"), pre.get("label"), pre.get("scene")],
          [media.SCHEMA_PREREQ, "prepared_media_only", label, SCENE], "prerequisites identity")
    for key in ("allowlist", "model_receipt", "video_receipt", "profile", "camera_k", "association"):
        _same(pre["inputs"][key]["sha256"], sha[key], f"prerequisites binding of {key}")
    _same([pre["outputs"]["glb"]["sha256"], pre["outputs"]["video"]["sha256"]], [sha["glb"], sha["video"]],
          "prerequisites output bindings")
    _same(pre["association"]["instance_ordinal"], None, "prerequisites association ordinal")
    _same([allow.get("schema"), allow.get("label"), allow["model"]["object_id"]], [media.SCHEMA_ALLOWLIST, label, OBJECT_ID],
          "allowlist identity")
    for key in ("profile", "camera_k", "association"):
        _same(allow["inputs"][key]["sha256"], sha[key], f"allowlist binding of {key}")
    _same([model.get("schema"), model.get("label"), model.get("object_id"), model["allowlist"]["sha256"]],
          [media.SCHEMA_MODEL, label, OBJECT_ID, sha["allowlist"]], "model receipt identity")
    _same([model["glb"]["sha256"], model["glb"]["bytes"], model["glb"]["path"]],
          [sha["glb"], inputs["glb"]["bytes"], path["glb"]], "model receipt GLB")
    _same([model["geometry"]["triangles"], model["geometry"]["decimated"], model["portable_targets"]["triangles"]],
          [c["mesh_triangles"], False, "FAIL" if c["mesh_triangles"] > PORTABLE_TRIANGLE_TARGET else "PASS"],
          "model geometry / portability")
    _need(c["portable_triangle_target_passed"] == (c["mesh_triangles"] <= PORTABLE_TRIANGLE_TARGET),
          "portable triangle target flag")
    _same([video.get("schema"), video.get("label"), video["allowlist"]["sha256"], video["video"]["sha256"],
           video["video"]["bytes"], video["video"]["path"]],
          [media.SCHEMA_VIDEO, label, sha["allowlist"], sha["video"], inputs["video"]["bytes"], path["video"]],
          "video receipt identity")
    _same([video["decode"]["sequential_frames"], video["decode"]["end_of_video_after"], video["decode"]["dimensions"],
           video["decode"]["frame_reader_seek_check"], len(video["frames"])],
          [FRAME_COUNT, FRAME_COUNT, [WIDTH, HEIGHT], {"exact": True, "frame_id": FIRST_SCORED_FRAME_ID}, FRAME_COUNT],
          "video decode record")
    _same([video["clock"]["physical_fps"], video["clock"]["physical_timestamps"], video["clock"]["presentation_fps"]],
          [None, None, c["presentation_fps"]], "video clock")
    _same([f["frame_id"] for f in video["frames"]], [f"s{i:06d}" for i in range(FRAME_COUNT)], "video frame order")
    # profile / K / association
    profile, kdoc, assoc = docs["profile"], docs["camera_k"], docs["association"]
    _same([profile.get("schema"), profile.get("scene"), len(profile["frames"]["rows"]), profile["image"]["width"],
           profile["image"]["height"]], [pf.SCHEMA_PROFILE, SCENE, FRAME_COUNT, WIDTH, HEIGHT], "profile identity")
    _same([kdoc.get("schema"), kdoc["profile"]["sha256"], kdoc["adjustments"]["origin_shift"], kdoc["adjustments"]["crop"],
           kdoc["adjustments"]["rescale"]], [pf.SCHEMA_K, sha["profile"], None, None, None], "camera-K record")
    raw_k = list(kdoc["K"]["values"])
    _need(len(raw_k) == 9 and all(type(v) is float and math.isfinite(v) for v in raw_k), "raw K malformed")
    _same([assoc.get("schema"), assoc.get("scene"), assoc["cad"]["object_id"], assoc["instance_ordinal"],
           assoc["selection"]["frame_id"], assoc["selection"]["positive_point_xy"], assoc["selection"]["roi_xyxy"],
           assoc["profile"]["sha256"]],
          [media.SCHEMA_ASSOCIATION, SCENE, OBJECT_ID, None, SEED_FRAME_ID, c["fixed_point_xy"], c["fixed_roi_xyxy"],
           sha["profile"]], "association (original ordinal must stay null)")
    # seed chain
    policy, plan, seed, binding = docs["seed_policy"], docs["seed_plan"], docs["controlled_seed"], docs["seed_binding"]
    _same([policy.get("schema"), policy["object_id"], policy["frame_id"], policy["association_sha256"],
           policy["prerequisites_sha256"]],
          [sd.SCHEMA_POLICY, OBJECT_ID, SEED_FRAME_ID, sha["association"], sha["prerequisites"]], "seed policy")
    _same([plan.get("schema"), plan.get("label"), plan["policy"]["sha256"], plan["inputs"]["prerequisites"]["sha256"]],
          [sd.SCHEMA_PLAN, label, sha["seed_policy"], sha["prerequisites"]], "seed plan")
    ordinal = seed.get("instance_ordinal")
    _need(type(ordinal) is int and 0 <= ordinal < 64, "seed instance ordinal malformed")
    seed_label = (f"controlled_init:ground_truth:HB:{mi.DATASET_REVISION}:scene{SCENE}:obj{OBJECT_ID:06d}:"
                  f"instance{ordinal}:{SEED_FRAME_ID}")
    _same([seed.get("schema"), seed.get("label"), seed["object_id"], seed["frame_id"], seed["frame_index"],
           seed["seed_label"], seed["plan"]["sha256"], seed["applies_once_to_frame_id"], seed["reuse_on_recovery"],
           seed["other_pose_values_converted"], seed["principal_point_origin"], seed["K_operation"]],
          [sd.SCHEMA_SEED, label, OBJECT_ID, SEED_FRAME_ID, 9, seed_label, sha["seed_plan"], FIRST_SCORED_FRAME_ID,
           False, 0, None, None], "controlled seed")
    _same({k: seed["flags"][k] for k in ("ground_truth_seed", "controlled_init", "initial_ground_truth_read",
                                          "automatic_geometry_annotation_access", "scored_pose_values_materialized")},
          {"ground_truth_seed": True, "controlled_init": True, "initial_ground_truth_read": True,
           "automatic_geometry_annotation_access": False, "scored_pose_values_materialized": False}, "seed flags")
    try:
        pose = [list(row) for row in wc._pose(seed["seed_pose_m"], "seed_pose_m")]      # finite proper SE(3)
    except (wc.ContractError, TypeError, ValueError):
        raise ExecutionControlError("seed pose is not a finite proper rigid transform") from None
    _same(pose, seed["seed_pose_m"], "seed pose (metres; used verbatim, never rescaled/inverted)")
    _same([binding.get("schema"), binding.get("label"), binding["status"], binding["seed"]["sha256"],
           binding["seed"]["bytes"], binding["plan"]["sha256"], binding["derived_instance_ordinal"], binding["seed_label"],
           binding["frame9"]["obj_id_4_matches"], binding["frame9"]["other_pose_values_converted"],
           binding["association"]["instance_ordinal"], binding["association"]["unchanged"]],
          [sd.SCHEMA_BINDING, label, "controlled_seed_extracted", sha["controlled_seed"], inputs["controlled_seed"]["bytes"],
           sha["seed_plan"], ordinal, seed_label, 1, 0, None, True], "seed binding")
    # independent acceptance and actual final-closure execution (binding alone is insufficient)
    src_acc, act_plan, act_ext, act_acc = (docs["seed_source_acceptance"], docs["actual_seed_plan"],
                                           docs["actual_seed_extraction"], docs["actual_seed_acceptance"])
    _same([src_acc.get("schema"), src_acc["verdict"], src_acc["independent_review"], src_acc["codex_failures"],
           src_acc["source_pins"]["tools/workbench_hb_scene3_seed.py"]],
          ["workbench_hb_scene3_seed_source_acceptance_v1", "accepted", "accepted", 0,
           reg["accepted_source_versions"]["tools/workbench_hb_scene3_seed.py"]], "seed source acceptance")
    for doc, stage in ((act_plan, "plan"), (act_ext, "extract")):
        _same([doc.get("schema"), doc["stage"], doc["status"], doc["final_closure_returned_successfully"],
               doc["source_acceptance"]["sha256"], doc["result"]["exit"], doc["result"]["label"],
               doc["neural_executed"], doc["registered_or_admitted"]],
              ["workbench_hb_scene3_actual_seed_execution_v1", stage, "completed", True, sha["seed_source_acceptance"],
               0, label, False, False], f"actual seed {stage} execution")
    _same([act_plan["result"]["plan_sha256"], act_plan["result"]["annotation_opens"]], [sha["seed_plan"], 0],
          "actual seed plan result")
    _same([act_ext["result"]["seed_sha256"], act_ext["result"]["binding_sha256"], act_ext["result"]["instance_ordinal"],
           act_ext["result"]["seed_label"], act_ext["result"]["annotation_opens"]],
          [sha["controlled_seed"], sha["seed_binding"], ordinal, seed_label, 4], "actual seed extraction result")
    _same([act_acc.get("schema"), act_acc["verdict"], act_acc["final_closure_accepted"], act_acc["object_id"],
           act_acc["frame_id"], act_acc["instance_ordinal"], act_acc["unique_matching_records"],
           act_acc["other_pose_values_converted"], act_acc["translation_units"]],
          ["workbench_hb_scene3_actual_seed_acceptance_v1", "accepted", True, OBJECT_ID, SEED_FRAME_ID, ordinal, 1, 0,
           "metres"], "independent actual seed acceptance")
    for key in ("actual_seed_extraction", "actual_seed_plan", "controlled_seed", "opening_placement", "seed_binding",
                "seed_plan"):
        _same(act_acc["evidence"][key]["sha256"], sha[key], f"actual seed acceptance evidence {key}")
    # actual media execution
    am = docs["actual_media"]
    arts = {str(k).replace("\\", "/"): v for k, v in am["artifacts"].items()}
    _same([am.get("schema"), am["tool_sha256"], am["neural_executed"], am["gt_or_raw_camera_read"],
           am["registered_or_admitted"], am["model"]["triangles"]],
          ["workbench_hb_scene3_actual_media_execution_v1", reg["accepted_source_versions"]["tools/workbench_hb_scene3_media.py"],
           False, False, False, c["mesh_triangles"]], "actual media execution")
    for key in ("allowlist", "glb", "model_receipt", "prerequisites", "video_receipt", "video"):
        _same(arts.get(path[key]), {"bytes": inputs[key]["bytes"], "sha256": sha[key]}, f"actual media artifact {key}")
    # origin (reviewed assumption only)
    org, ind = docs["origin_review"], docs["origin_independent_review"]
    _same([org.get("schema"), org["selected_origin"], org["evidence"], org["verified_dataset_convention"],
           org["physical_calibration_proof"], org["origin_shift"], org["crop"], org["rescale"], org["raw_K"],
           org["effective_K"], org["camera_k"]["sha256"]],
          ["workbench_hb_scene3_principal_point_review_v1", c["origin"], c["origin_evidence"], False, False, [0.0, 0.0],
           None, None, raw_k, raw_k, sha["camera_k"]], "origin review (raw K unchanged, no shift/crop/rescale)")
    _same([ind.get("schema"), ind["verdict"], ind["reviewed_sha256"], ind["principal_point_origin"], ind["raw_K_unchanged"],
           ind["origin_shift"], ind["verified_dataset_convention"], ind["physical_calibration_proof"], ind["GT_used"]],
          ["workbench_hb_scene3_origin_independent_review_v1", "accepted_as_reviewed_assumption_only",
           sha["origin_review"], c["origin"], True, [0, 0], False, False, False], "independent origin review")
    _same(list(wn.native_K(raw_k, c["origin"]).ravel()), raw_k, "integer-centre K (must equal the raw K)")
    # opening placement (frozen ray hit only; observed mask pending)
    pl = docs["opening_placement"]
    art = pl["artwork"]
    _same([pl.get("schema"), pl["verdict"], pl["observed_mask_checked"], pl["selection_reviewed"],
           pl["point_changed_or_searched"], pl["positive_point_xy"], pl["roi_xyxy"], pl["frame_id"],
           pl["principal_point_origin"], pl["origin_evidence"], pl["physical_calibration_verified"], pl["mesh_triangles"],
           pl["portable_triangle_target_passed"], pl["registered_or_admitted"], pl["neural_executed"],
           pl["scored_frame_used"], pl["segmentation"], pl["annotation_or_raw_camera_opens"], pl["texture_edge_px"],
           pl["actual_seed_receipt"]["sha256"]],
          ["workbench_hb_scene3_opening_placement_review_v1", "frozen_opening_ray_hit; observed_mask_review_pending",
           False, False, False, c["fixed_point_xy"], c["fixed_roi_xyxy"], SEED_FRAME_ID, c["origin"],
           c["origin_evidence"], False, c["mesh_triangles"], False, False, False, False, False, 0, ARTWORK["edge_px"],
           sha["actual_seed_extraction"]], "opening placement review (ray hit only; mask review pending)")
    for key, reg_key in (("glb", "glb"), ("seed", "controlled_seed"), ("binding", "seed_binding"), ("K", "camera_k"),
                         ("association", "association"), ("origin", "origin_review")):
        _same(pl["inputs"][key]["sha256"], sha[reg_key], f"opening placement input {key}")
    _same([art.get("schema"), art["glb_sha256"], art["seed_label"], art["anchor_pixel"], art["anchor_frame_id"],
           art["effective_K"], art["principal_point_origin"], art["anchor_mask_checked"], art["mesh_faces"], art["units"],
           art["artwork"]["kind"], art["artwork"]["edge_px"], art["artwork"]["generation"],
           art["marker"]["artwork_sha256"], art["artwork"]["sha256"]],
          ["workbench_neural_artwork_v1", sha["glb"], seed_label, [float(v) for v in c["fixed_point_xy"]], SEED_FRAME_ID,
           raw_k, c["origin"], False, c["mesh_triangles"], "metres", ARTWORK["kind"], ARTWORK["edge_px"],
           ARTWORK["generation"], art["artwork"]["sha256"], art["artwork"]["sha256"]],
          "frozen NeuralArtwork record")
    _same(wc.canonical_sha256(art), pl["artwork_record_sha256"], "artwork record hash (recomputed)")
    _validate_placement_geometry(pl, art, pose=pose, raw_k=raw_k, mesh_faces=c["mesh_triangles"],
                                 fixed_xy=c["fixed_point_xy"])
    for rel in ("bench/workbench_contract.py", "bench/workbench_geometry.py", "bench/workbench_neural.py"):
        _same(pl["sources"][rel], reg["accepted_source_versions"][rel], f"placement source {rel}")
    return {"raw_k": raw_k, "seed_pose": pose, "seed_label": seed_label, "ordinal": ordinal,
            "artwork_record_sha256": pl["artwork_record_sha256"], "artwork_sha256": art["artwork"]["sha256"],
            "placement": {k: pl[k] for k in ("face_id", "barycentric", "point_object_m", "depth_m")},
            "video_workspace_relative": video["video"]["workspace_relative"]}


ARTWORK_RECORD_KEYS = frozenset({"schema", "units", "glb_sha256", "marker", "anchor_pixel", "anchor_frame_id",
                                 "seed_label", "artwork", "support_source", "principal_point_origin", "effective_K",
                                 "patch_faces", "mesh_faces", "anchor_mask_checked"})
SUPPORT_SOURCE = "metric CAD geometry (not observation-derived)"
MAX_PROJECTION_RESIDUAL_PX = 1e-3
_CHECKER_DIGEST: list = []


def checker_digest() -> str:
    """Semantic digest of the EXISTING fixed built-in checkerboard (pure NumPy; no renderer/model)."""
    if not _CHECKER_DIGEST:
        from bench import workbench_geometry as wg
        _CHECKER_DIGEST.append(wg.array_sha256(wg.builtin_artwork(ARTWORK["kind"], ARTWORK["edge_px"])))
    return _CHECKER_DIGEST[0]


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_placement_geometry(pl: Mapping, art: Mapping, *, pose: Sequence, raw_k: Sequence, mesh_faces: int,
                                 fixed_xy: Sequence) -> None:
    """Complete placement/marker validation (booleans are never numbers):
    * the artwork record has exactly the ``NeuralArtwork.record`` keys; its marker passes ``wc.Marker.from_dict``
      (barycentric simplex, orthonormal right-handed chart, positive size), and the placement's face/barycentric
      are bound EXACTLY to it; the face is a strict int in [0, mesh_faces);
    * the artwork digest equals the RECOMPUTED digest of the fixed built-in checkerboard;
    * depth is a finite non-bool > 0; the recorded residual is a finite non-bool in [0, 1e-3] px and matches the
      residual recomputed below.
    FROZEN-POINT CONSISTENCY ONLY: the object-space hit point, transformed by the accepted seed pose, must lie at the
    recorded depth and project with the raw K onto the frozen pixel. One point cannot verify units, origin or a
    unique pose (another proper rigid pose rotating about that point passes too); units and pose provenance are
    inherited from the pinned seed/model, and the integer-centre origin remains a reviewed assumption."""
    _need(set(art) == ARTWORK_RECORD_KEYS and art["support_source"] == SUPPORT_SOURCE, "artwork record fields differ")
    _need(type(art["patch_faces"]) is int and 0 < art["patch_faces"] <= mesh_faces, "artwork patch_faces invalid")
    _need(isinstance(art["artwork"], Mapping) and set(art["artwork"]) == {"kind", "edge_px", "sha256", "generation"},
          "artwork descriptor fields differ")
    digest = checker_digest()
    _same([art["artwork"]["sha256"], art["marker"].get("artwork_sha256") if isinstance(art["marker"], Mapping) else None],
          [digest, digest], "artwork digest (recomputed fixed checkerboard)")
    try:
        marker = wc.Marker.from_dict(art["marker"])
    except (wc.ContractError, TypeError, ValueError):
        raise ExecutionControlError("artwork marker is not a valid contract Marker") from None
    face, bary = pl["face_id"], pl["barycentric"]
    _need(type(face) is int and 0 <= face < mesh_faces and marker.face_id == face,
          "placement face_id must be a strict int in [0, mesh_faces) equal to the marker face")
    _need(isinstance(bary, list) and len(bary) == 3 and all(_finite(b) and b >= 0 for b in bary)
          and abs(sum(bary) - 1.0) <= 1e-6 and tuple(float(b) for b in bary) == marker.barycentric,
          "placement barycentric must be a finite simplex point equal to the marker barycentric")
    depth, residual = pl["depth_m"], pl["projection_self_consistency_residual_px"]
    _need(_finite(depth) and depth > 0, "placement depth must be a finite non-boolean > 0")
    _need(_finite(residual) and 0 <= residual <= MAX_PROJECTION_RESIDUAL_PX,
          "placement residual must be a finite non-boolean in [0, 1e-3] px")
    point = pl["point_object_m"]
    _need(isinstance(point, list) and len(point) == 3 and all(_finite(v) for v in point), "placement object point malformed")
    cam = [sum(pose[i][j] * point[j] for j in range(3)) + pose[i][3] for i in range(3)]
    _need(cam[2] > 0 and abs(cam[2] - depth) <= 1e-6, "seed pose does not reproduce the recorded placement depth")
    u = raw_k[0] * cam[0] / cam[2] + raw_k[2]
    v = raw_k[4] * cam[1] / cam[2] + raw_k[5]
    recomputed = math.hypot(u - fixed_xy[0], v - fixed_xy[1])
    _need(recomputed <= MAX_PROJECTION_RESIDUAL_PX and abs(recomputed - residual) <= 1e-6,
          "frozen-point consistency: seed pose + raw K do not reproduce the frozen pixel / recorded residual")


# --------------------------------------------------------------------------
# output construction (deterministic from authenticated inputs)


def build_outputs(reg: Mapping, facts: Mapping, *, label: str) -> dict:
    inputs = reg["inputs"]
    c = EXPECTED_CONSTRAINTS
    bundle = {"schema": BUNDLE_SCHEMA, "family": output_family(label), "glb_path": inputs["glb"]["path"],
              "glb_sha256": inputs["glb"]["sha256"], "obj_id": OBJECT_ID, "principal_point_origin": c["origin"],
              "seed_pose_m": facts["seed_pose"], "seed_label": facts["seed_label"], "seed_frame_id": SEED_FRAME_ID,
              "metric_provenance": "dataset_metric_cad"}
    clip = {"schema": wc.CLIP_SCHEMA, "clip_id": f"hb-val-primesense-{SCENE}-obj{OBJECT_ID:06d}",
            "title": f"HB val_primesense {SCENE} CAD {OBJECT_ID} (lossy mpeg4 derivative of 120 RGB PNGs; prepared metric control)",
            "source_kind": "video_file", "clock_kind": "frame_index", "clock_source": None, "width": WIDTH,
            "height": HEIGHT,
            "media": {"root_id": "workbench", "path": facts["video_workspace_relative"], "sha256": inputs["video"]["sha256"],
                      "bytes": inputs["video"]["bytes"]},
            "frames": [{"frame_id": f"s{i:06d}", "source_frame_index": i, "timestamp_s": None, "media_time_s": None}
                       for i in range(FRAME_COUNT)],
            "intrinsics": {"status": "dataset_reference", "K": facts["raw_k"],
                           "source": (f"HB dataset reference K (camera-k sha256 {inputs['camera_k']['sha256'][:16]}); raw K "
                                      "unchanged; integer_centers is a reviewed assumption; not a phone calibration")},
            "onboarding": {"keyframe_ids": [f"s{i:06d}" for i in OPENING],
                           "selection": {"frame_id": SEED_FRAME_ID, "positive": [list(c["fixed_point_xy"])],
                                         "negative": [], "reviewed": False, "reviewed_by": None}},
            "scoring_frame_ids": [f"s{i:06d}" for i in SCORING],
            "evidence_role": "dataset_prepared_metric_control_development", "evaluator_inputs_read": False,
            "excluded_source_fields": [],
            "provenance": {
                "label": label, "family": output_family(label), "scene": SCENE, "object_id": OBJECT_ID,
                "trust_level": "historical_local_integrity", "upstream_authenticated": False,
                "registry_sha256": None,                                   # filled by assemble (see below)
                "clock": {"kind": "frame_index", "physical_fps": None, "physical_timestamps": None,
                          "presentation_fps": c["presentation_fps"], "presentation_fps_kind": "synthetic presentation rate",
                          "ten_second_prefix_attested": False,
                          "note": "the frame-index clock cannot attest a physical ten-second onboarding prefix"},
                "lossy_derivative": True,
                "principal_point": {"origin": c["origin"], "evidence": c["origin_evidence"],
                                    "verified_dataset_convention": False, "physical_calibration_proof": False,
                                    "raw_K_unchanged": True, "origin_shift": [0, 0], "crop": None, "rescale": None},
                "frozen_roi_xyxy": c["fixed_roi_xyxy"],
                "selection": {"reviewed": False, "observed_sam_tap_gate": "pending",
                              "cad_is_observed_foreground": False},
                "controlled_ground_truth_initialization": {"seed_label": facts["seed_label"],
                                                           "applies_once_to_frame_id": FIRST_SCORED_FRAME_ID,
                                                           "disclosure": GT_DISCLOSURE, "evaluator_free": False},
                "association": {"sha256": inputs["association"]["sha256"], "original_instance_ordinal": None,
                                "derived_instance_ordinal": facts["ordinal"],
                                "derived_ordinal_source": "accepted seed binding (source-order unique match)"},
                "registered": False, "admitted": False}}
    sidecar = {
        "schema": SCHEMA_SIDECAR, "tool_version": TOOL_VERSION, "label": label, "scene": SCENE, "object_id": OBJECT_ID,
        "artwork": {**ARTWORK, "artwork_sha256": facts["artwork_sha256"],
                    "frozen_record_sha256": facts["artwork_record_sha256"],
                    "opening_placement_sha256": inputs["opening_placement"]["sha256"], "ray_search_performed": False},
        "roles": {role: {"anchor_record_sha256": facts["artwork_record_sha256"], "seed_label": facts["seed_label"],
                         "seed_applies_once_to_frame_id": FIRST_SCORED_FRAME_ID, "independent_initialization": True,
                         "rearm_seed_on_loss": False} for role in ("primary_neural", "frozen_baseline")},
        "shared_anchor": True,
        "placement": {**facts["placement"], "point_changed_or_searched": False, "observed_mask_checked": False,
                      "selection_reviewed": False, "evidence": "frozen CAD ray hit only; CAD is not observed foreground"},
        "gates": {"observed_sam_review": "pending", "selection_reviewed": False},
        "origin": {"principal_point_origin": c["origin"], "evidence": c["origin_evidence"],
                   "verified_dataset_convention": False, "physical_calibration_proof": False, "raw_K": facts["raw_k"],
                   "origin_shift": [0, 0], "crop": None, "rescale": None, "K_operation": None},
        "fixed_point_xy": c["fixed_point_xy"], "frozen_roi_xyxy": c["fixed_roi_xyxy"],
        "mesh": {"glb_sha256": inputs["glb"]["sha256"], "glb_bytes": inputs["glb"]["bytes"],
                 "triangles": c["mesh_triangles"], "portable_triangle_target": PORTABLE_TRIANGLE_TARGET,
                 "portable_triangle_target_passed": False, "texture_edge_px": ARTWORK["edge_px"]},
        "association": {"original_instance_ordinal": None, "derived_instance_ordinal": facts["ordinal"]},
        "runtime_executed_or_tested_by_assembly": False, "family": output_family(label), "flags": output_flags(label)}
    return {BUNDLE_NAME: bundle, CLIP_NAME: clip, SIDECAR_NAME: sidecar}


def check_outputs(outputs: Mapping, *, root: Path) -> dict:
    """Core validators on the in-memory outputs (no files): exact bundle keys/values, ClipSpec + onboarding."""
    bundle, clip = outputs[BUNDLE_NAME], outputs[CLIP_NAME]
    _need(set(bundle) == {"schema", "family", "glb_path", "glb_sha256", "obj_id", "principal_point_origin", "seed_pose_m",
                          "seed_label", "seed_frame_id", "metric_provenance"}, "metric bundle keys differ")
    _need(str(bundle["seed_label"]).startswith(wn.SEED_PREFIXES), "seed label prefix")
    _rel_path(bundle["glb_path"], "bundle glb_path")
    spec = wc.ClipSpec.from_dict(clip)
    attestation = wc.check_onboarding(spec, spec.onboarding, spec.scoring_frame_ids)
    _need(attestation["prefix_bound_attested"] is False and len(spec.frames) == FRAME_COUNT
          and spec.scoring_frame_ids == tuple(f"s{i:06d}" for i in SCORING), "clip onboarding/scoring")
    return attestation


# --------------------------------------------------------------------------
# authentication of everything (guard active)


def _safe_dir(ctx: AssemblyContext, directory: Path) -> Path:
    """The output directory and EVERY ancestor from the repository root: inside the repository, link-free, and
    the directory itself a real directory. Checked before every publication and before any cleanup."""
    root = Path(os.path.abspath(ctx.root))
    d = Path(os.path.abspath(directory))
    _need(root in d.parents, "output directory lies outside the repository")
    mi._no_links(root, d)
    st = os.lstat(d)
    _need(stat.S_ISDIR(st.st_mode) and not mi._is_link(d), "output directory is not a real directory")
    return d


def _bind_video(ctx: AssemblyContext, workspace_relative: Any, registered_video: Path) -> None:
    """The accepted video receipt's ``workspace_relative`` must resolve (contained, traversal/link-free) to EXACTLY
    the sealed registry video; the clip's media path is that same string."""
    rel = _rel_path(workspace_relative, "video workspace_relative")
    root, ws = Path(os.path.abspath(ctx.root)), Path(os.path.abspath(ctx.workspace))
    _need(root in ws.parents, "workspace lies outside the repository")
    mi._no_links(root, ws)
    target = Path(os.path.abspath(ws.joinpath(*PurePosixPath(rel).parts)))
    _need(target == Path(os.path.abspath(registered_video)),
          "video workspace_relative does not resolve to the sealed registry video")
    mi._no_links(root, target)


def _authenticate(ctx: AssemblyContext, pin: Mapping, counter: Any) -> dict:
    reg_path = _contained(ctx, pin["path"], "assembly registry")
    counter.watch(reg_path, "registry")
    reg_raw, reg = _read_metadata(reg_path, expected_sha=pin["sha256"], expected_bytes=pin["bytes"], cap=REGISTRY_CAP,
                                  what="assembly registry")
    validate_registry(reg, label=pin["label"])
    current = {rel: mi.sha256_file(ROOT / rel) for rel in SOURCE_FILES}
    _same(current, reg["accepted_source_versions"], "current helper/source versions")
    scope = Path(os.path.abspath(ctx.root / reg["preparation_scope"]))
    mctx = mi.Context(root=ctx.root, workspace=ctx.workspace, preparation=scope)
    mi.preparation_scope(mctx)
    paths, docs, files = {}, {}, {"registry": (pin["path"], pin["bytes"])}
    for key, item in reg["inputs"].items():
        path = _contained(ctx, item["path"], key)
        (mi.inside_scope if key in IN_SCOPE_KEYS else mi.outside_scope_input)(mctx, path, key)
        counter.watch(path, key)
        paths[key] = path
        if item["kind"] == "metadata":
            _, docs[key] = _read_metadata(path, expected_sha=item["sha256"], expected_bytes=item["bytes"],
                                          cap=item["cap_bytes"], what=key)
        else:
            _seal_derivative(path, expected_sha=item["sha256"], expected_bytes=item["bytes"], cap=item["cap_bytes"],
                             what=key)
        files[key] = (item["path"], item["bytes"])
    try:
        facts = _check_relations(reg, docs, label=pin["label"], root=ctx.root)
    except (KeyError, TypeError, IndexError) as exc:
        raise ExecutionControlError(f"accepted record is missing an expected field ({type(exc).__name__})") from None
    _bind_video(ctx, facts["video_workspace_relative"], paths["video"])
    outputs = build_outputs(reg, facts, label=pin["label"])
    outputs[CLIP_NAME]["provenance"]["registry_sha256"] = pin["sha256"]
    attestation = check_outputs(outputs, root=ctx.root)
    new_bytes = len(reg_raw) + sum(reg["inputs"][k]["bytes"] for k in REVIEW_KEYS)
    seed_total = docs["seed_plan"]["input_budget"]["total"]
    _need(type(seed_total) is int, "seed plan input accounting malformed")
    inputs_budget = mi.input_budget(seed_total, new_bytes, 0)
    pinned = {"registry": pin["sha256"], **{k: v["sha256"] for k, v in reg["inputs"].items()}}
    return {"reg": reg, "mctx": mctx, "paths": paths, "files": files, "pinned": dict(sorted(pinned.items())),
            "facts": facts, "outputs": outputs, "attestation": attestation, "input_budget": inputs_budget,
            "assembly_root": Path(os.path.abspath(ctx.root / reg["output_directory"]))}


def _closure(ctx: AssemblyContext, files: Mapping[str, tuple[str, int]]) -> dict:
    """For EVERY registry input: re-check containment, regularity and the WHOLE ancestor chain (link refusal from the
    repository root) BEFORE the bounded re-seal at its registered size; then the source closure and runtime.
    Refusals are reported as values (a mismatch), never raised."""
    hashes = {}
    for key, (rel, size) in sorted(files.items()):
        try:
            path = _contained(ctx, rel, key)
            st = os.lstat(path)
            hashes[key] = (media.sealed_sha256(path, size) if stat.S_ISREG(st.st_mode) and st.st_size == size
                           else f"refused: size/type changed ({st.st_size})")
        except (mi.InputsError, OSError) as exc:
            hashes[key] = f"refused: {type(exc).__name__}"
    return {"files": hashes, "sources": _sources(), "runtime": mi.runtime_identity()}


def expected_receipt_reads(reg: Mapping) -> dict:
    """The audited opens of the FIXED assemble path at the receipt snapshot (structurally required, exact):
    every metadata input and the registry: authentication read + pre-publication closure = 2; the video: seal +
    closure = 2; the GLB: seal + closure + the core ``MetricBundle`` loader = 3; the three published outputs:
    3 exclusive staging writes, 2 verification reads each + 1 core loader read of the bundle = 7."""
    reads = {f"{key}:read": 2 for key, item in reg["inputs"].items() if item["kind"] == "metadata"}
    reads.update({"registry:read": 2, "video:read": 2, "glb:read": 3, "assembly_output:read": 7,
                  "assembly_output:write": 3})
    return dict(sorted(reads.items()))


# --------------------------------------------------------------------------
# ledger + failure (assembly-v1 subtree <= 96 KiB)


class AssemblyLedger:
    """Duck-compatible with ``media.publish``/``media._commit`` and the reviewed ``sd._quarantine``. Before every
    write: the WHOLE assembly-v1 subtree (all attempts, failures, quarantined outputs, staging aliases) <= 96 KiB,
    the preparation scope within 12 MiB / 512 KiB metadata, the free floor and a fresh global derived check."""

    def __init__(self, mctx: mi.Context, out_dir: Path, assembly_root: Path):
        self.ctx, self.dir, self.scope, self.subtree = mctx, out_dir, mctx.scope, assembly_root
        self.stage = "assemble"
        self.files: dict[str, int] = {}
        self.owned: dict[str, tuple[int, int]] = {}

    def check_sequence(self, steps: Sequence[tuple[int, bool]], *, failure: bool = False) -> dict:
        if not steps or any(type(n) is not int or n < 0 for n, _ in steps):
            raise mi.InputsError("invalid write size")
        factor = media.TRANSIENT_FACTOR
        subtree, scope = media._usage(self.subtree), media._usage(self.scope)
        reserve = 0 if failure else factor * FAILURE_CAP
        done = peak_max = 0
        for nbytes, _ in steps:
            peak = factor * nbytes
            peak_max = max(peak_max, done + peak)
            if subtree["total"] + done + peak > SUBTREE_CAP - reserve:
                raise mi.Blocked(f"assembly subtree would peak at {subtree['total'] + done + peak} bytes, above "
                                 f"{SUBTREE_CAP - reserve} (retained attempts, failures and staging included)")
            if scope["total"] + done + peak > mi.FINAL_OUTPUTS_CAP - reserve:
                raise mi.Blocked(f"preparation scope would exceed {mi.FINAL_OUTPUTS_CAP - reserve}")
            if scope["metadata"] + done + peak > mi.METADATA_AGGREGATE_CAP - reserve:
                raise mi.Blocked(f"scope metadata would exceed {mi.METADATA_AGGREGATE_CAP - reserve}")
            done += nbytes
        if mi.shutil.disk_usage(self.ctx.workspace).free - peak_max < mi.FREE_RESERVE:
            raise mi.Blocked("the 1 GiB free-disk reserve would be violated")
        mi.workspace_budget(self.ctx, peak_max)
        return {"subtree": subtree, "scope": scope}

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        return self.check_sequence([(nbytes, True)], failure=failure)

    def owned_file(self, name: str) -> bool:
        return sd._presence(self, name) == "present_owned"

    def summary(self) -> dict:
        return {"files": dict(self.files), "owned": sorted(self.owned), "subtree_usage_now": media._usage(self.subtree),
                "scope_usage_now": media._usage(self.scope),
                "caps": {"assembly_subtree": SUBTREE_CAP, "scope_total": mi.FINAL_OUTPUTS_CAP,
                         "scope_metadata": mi.METADATA_AGGREGATE_CAP, "failure_record": FAILURE_CAP}}


def _publish(ctx: AssemblyContext, ledger: AssemblyLedger, name: str, data: bytes, **kwargs: Any) -> dict:
    _safe_dir(ctx, ledger.dir)                                   # ancestry re-checked before EVERY publication
    return media.publish(ledger, name, data, **kwargs)


def _failure(ctx: AssemblyContext, ledger: AssemblyLedger, exc: BaseException) -> None:
    """Independently quarantine every OWNED output with the reviewed seed mechanics (unique exclusive names,
    identity re-check, foreign names untouched), inventory remaining/uncertain names explicitly, publish a bounded
    failure record, and NEVER raise (the original error always propagates). If the output ancestry is unsafe,
    NOTHING is moved or written through it: the uncertain state goes to stderr only."""
    try:
        _safe_dir(ctx, ledger.dir)
    except Exception as unsafe:  # noqa: BLE001
        try:
            sys.stderr.write("failure.json NOT published; output ancestry unsafe, nothing moved or written: "
                             + json.dumps({"schema": SCHEMA_FAILURE, "status": "failed_output_ancestry_unsafe",
                                           "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                                           "ancestry_error": type(unsafe).__name__,
                                           "owned_outputs_state": "uncertain", "owned": sorted(ledger.owned)},
                                          sort_keys=True) + "\n")
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        records = []
        for name in sorted(ledger.owned):
            if name == FAILURE_NAME:
                continue
            _safe_dir(ctx, ledger.dir)       # before EVERY cleanup move; unsafe -> uncertain, nothing further moved
            try:
                records.append(sd._quarantine(ledger, name))
            except Exception as cleanup_exc:  # noqa: BLE001
                records.append({"name": name, "error": type(cleanup_exc).__name__})
        states = {n: sd._presence(ledger, n) for n in sorted(ledger.owned) if n != FAILURE_NAME}
        remaining = sorted(n for n, s in states.items() if s == "present_owned")
        uncertain = sorted(n for n, s in states.items() if s == "unknown")
        body = {"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "status": "failed_authoritative_outputs_remain"
                if remaining or uncertain else "failed", "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                "completed_receipt_published": False, "authoritative_files_remaining": remaining,
                "authoritative_files_uncertain": uncertain, "authoritative_states": states,
                "quarantined_outputs": [{k: r.get(k) for k in ("name", "retained_as", "error", "skipped")} for r in records]}
        data = mi.dump_json(body)
        if len(data) > FAILURE_CAP:
            body.update(reason=body["reason"][:160], quarantined_outputs=len(records))
            data = mi.dump_json(body)
    except Exception as inventory_exc:  # noqa: BLE001
        data = mi.dump_json({"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION,
                             "status": "failed_authoritative_outputs_uncertain",
                             "reason": f"{type(exc).__name__}: {str(exc)[:300]}", "completed_receipt_published": False,
                             "inventory_error": type(inventory_exc).__name__})
    try:
        if len(data) > FAILURE_CAP:
            raise mi.Blocked("failure record exceeds its cap")
        _publish(ctx, ledger, FAILURE_NAME, data, failure=True)
    except Exception as publish_exc:  # noqa: BLE001
        try:
            sys.stderr.write(f"failure.json NOT published ({type(publish_exc).__name__}); record on ONE line:\n"
                             f"{json.dumps(json.loads(data), sort_keys=True, separators=(',', ':'))}\n")
        except Exception:  # noqa: BLE001
            pass


def _out_dir(mctx: mi.Context, out: Any, assembly_root: Path) -> Path:
    """A NEW directory: exactly the registry's assembly-v1 directory, or a direct child attempt inside it."""
    out_dir = mi.check_output_dir(mctx, out)
    _need(out_dir == assembly_root or out_dir.parent == assembly_root,
          f"--out must be {assembly_root} or a new attempt directly inside it")
    if out_dir != assembly_root:
        mi._no_links(Path(os.path.abspath(mctx.root)), assembly_root)
    return out_dir


# --------------------------------------------------------------------------
# public API


def inspect_assembly(ctx: AssemblyContext) -> dict:
    """Authenticate everything, build the outputs in memory, and finish a full input/source/runtime closure
    before reporting readiness. No writes."""
    pin = _pin(ctx)
    base = mi.Context(root=ctx.root, workspace=ctx.workspace)
    with mi.guarded_stage(base) as opened, media._counting() as counter:
        code = {"sources": _sources(), "runtime": mi.runtime_identity()}
        auth = _authenticate(ctx, pin, counter)
        _same(_closure(ctx, auth["files"]), {"files": auth["pinned"], **code}, "inspection closure (inputs/sources/runtime)")
        _need(not opened and not counter.reads("annotation") and not counter.reads("camera"),
              "a payload/annotation/camera file was opened")
        reads = counter.snapshot()
    return {"status": "ready_to_assemble", "label": pin["label"], "family": output_family(pin["label"]),
            "registry_sha256": pin["sha256"],
            "planned_outputs": {name: mi.sha256_bytes(mi.dump_json(doc)) for name, doc in auth["outputs"].items()},
            "seed_label": auth["facts"]["seed_label"], "derived_instance_ordinal": auth["facts"]["ordinal"],
            "prefix_attestation": auth["attestation"], "input_budget": auth["input_budget"], "audited_opens": reads,
            "exit": 0}


def assemble(ctx: AssemblyContext, *, out: Any) -> dict:
    pin = _pin(ctx)
    base = mi.Context(root=ctx.root, workspace=ctx.workspace)
    ledger: AssemblyLedger | None = None
    state = {"dir": False}
    try:
        with mi.guarded_stage(base) as opened, media._counting() as counter:
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            auth = _authenticate(ctx, pin, counter)
            mctx = auth["mctx"]
            out_dir = _out_dir(mctx, out, auth["assembly_root"])
            ledger = AssemblyLedger(mctx, out_dir, auth["assembly_root"])
            counter.watch_prefix(auth["assembly_root"], "assembly_output")
            budget = mi.workspace_budget(mctx, media.TRANSIENT_FACTOR * sum(OUTPUT_CAPS.values()) + mi.DIR_ENTRY_RESERVE)
            payloads = {name: mi.dump_json(doc) for name, doc in auth["outputs"].items()}
            for name, data in payloads.items():
                _need(len(data) <= OUTPUT_CAPS[name], f"{name} exceeds its cap")
            # prospective, BEFORE any write: the exact sizes of the three built outputs + the receipt cap, with
            # transient aliases, the failure reserve, retained attempts/failures, scope, free floor and global budget
            ledger.check_sequence([(len(payloads[n]), True) for n in (BUNDLE_NAME, CLIP_NAME, SIDECAR_NAME)]
                                  + [(OUTPUT_CAPS[RECEIPT_NAME], True)])
            before = {"files": auth["pinned"], **code}
            _same(_closure(ctx, auth["files"]), before, "pre-publication closure (inputs/sources/runtime)")
            _need(not opened, "payload bytes were opened")
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            published = {}
            for name in (BUNDLE_NAME, CLIP_NAME, SIDECAR_NAME):
                published[name] = _publish(ctx, ledger, name, payloads[name])
                if name == BUNDLE_NAME:                                # core loader on the published bundle
                    glb = _contained(ctx, auth["outputs"][BUNDLE_NAME]["glb_path"], "bundle GLB")
                    _need(glb == auth["paths"]["glb"], "bundle GLB path does not resolve to the registered GLB")
                    parsed = wn.MetricBundle.from_file(out_dir / BUNDLE_NAME, root=ctx.root)
                    _same([parsed.family, parsed.glb_sha256, parsed.obj_id, parsed.principal_point_origin,
                           parsed.seed_label, parsed.seed_frame_id, parsed.metric_provenance,
                           parsed.seed_pose_m.tolist()],
                          [output_family(pin["label"]), auth["reg"]["inputs"]["glb"]["sha256"], OBJECT_ID, "integer_centers",
                           auth["facts"]["seed_label"], SEED_FRAME_ID, "dataset_metric_cad", auth["facts"]["seed_pose"]],
                          "core MetricBundle parse")
            receipt = _receipt(auth, pin, out_dir, published, code, budget, counter.snapshot())
            _same(receipt["reads"]["audited_opens"], expected_receipt_reads(auth["reg"]), "audited opens at the receipt")
            receipt_data = mi.dump_json(receipt)
            _need(len(receipt_data) <= OUTPUT_CAPS[RECEIPT_NAME], "assembly receipt exceeds its cap")
            published[RECEIPT_NAME] = _publish(ctx, ledger, RECEIPT_NAME, receipt_data)       # receipt LAST
            _same(_closure(ctx, auth["files"]), before, "post-publication closure (inputs/sources/runtime)")
            _safe_dir(ctx, out_dir)
            _same(sorted(e.name for e in out_dir.iterdir()), sorted(OUTPUT_NAMES), "post-publication output names")
            for name, meta in published.items():
                _same(media.sealed_sha256(out_dir / name, meta["bytes"]), meta["sha256"], f"post-publication {name} bytes")
            _need(not opened and not counter.reads("annotation") and not counter.reads("camera"),
                  "a payload/annotation/camera file was opened")
            total = counter.snapshot()
    except BaseException as exc:
        if state["dir"] and ledger is not None:
            _failure(ctx, ledger, exc)
        raise
    return {"status": "assembled_unadmitted_unregistered", "label": pin["label"], "family": output_family(pin["label"]),
            "out": mi.repo_rel(mctx, out_dir), "receipt_sha256": published[RECEIPT_NAME]["sha256"],
            "outputs": {n: published[n]["sha256"] for n in OUTPUT_NAMES}, "audited_opens_total": total, "exit": 0}


def _receipt(auth: Mapping, pin: Mapping, out_dir: Path, published: Mapping, code: Mapping, budget: Any,
             reads: Mapping) -> dict:
    """Receipt for the three content outputs (it never hashes itself; its own hash is returned by ``assemble``
    only after the post-publication closure)."""
    reg, mctx = auth["reg"], auth["mctx"]
    return {"schema": SCHEMA_RECEIPT, "tool_version": TOOL_VERSION, "label": pin["label"],
            "status": "assembled_unadmitted_unregistered", "scene": SCENE, "object_id": OBJECT_ID,
            "family": output_family(pin["label"]),
            "registry": {"path": pin["path"], "sha256": pin["sha256"], "bytes": pin["bytes"]},
            "inputs": {k: {"path": v["path"], "sha256": v["sha256"], "bytes": v["bytes"]} for k, v in reg["inputs"].items()},
            "outputs": {n: {"path": mi.repo_rel(mctx, out_dir / n), "sha256": published[n]["sha256"],
                            "bytes": published[n]["bytes"]} for n in (BUNDLE_NAME, CLIP_NAME, SIDECAR_NAME)},
            "constraints": dict(EXPECTED_CONSTRAINTS), "flags": output_flags(pin["label"]),
            "labels": {"seed": "controlled ground-truth initialisation (not automatic, not evaluator-free)",
                       "accuracy": "no accuracy claim", "phone": "no phone/mobile/performance claim",
                       "clock": "frame index; physical FPS unknown; synthetic 30 fps presentation only"},
            "disclosure": GT_DISCLOSURE, "derived_instance_ordinal": auth["facts"]["ordinal"],
            "seed_label": auth["facts"]["seed_label"], "prefix_attestation": auth["attestation"],
            "glb": {"triangles": EXPECTED_CONSTRAINTS["mesh_triangles"], "portable_triangle_target": PORTABLE_TRIANGLE_TARGET,
                    "portable": "FAIL", "bytes": reg["inputs"]["glb"]["bytes"], "copied_or_linked": False,
                    "texture_edge_px": ARTWORK["edge_px"]},
            "unresolved": list(UNRESOLVED), "input_budget": auth["input_budget"], "budget": budget,
            "reads": {"snapshot": "audited opens before the receipt was written", "audited_opens": dict(reads),
                      "annotation_opens": 0, "raw_camera_opens": 0},
            "sources": code["sources"], "runtime": code["runtime"],
            "closure": {"before_files": auth["pinned"], "pre_publication_matches": True,
                        "post_publication": "re-verified after the receipt; the stage returns the receipt hash only then"}}


def validate_output(ctx: AssemblyContext, *, out: Any) -> dict:
    """Read-only. Re-authenticate everything, rebuild the expected outputs from the frozen registry, require exact
    bytes for the three content outputs and every deterministic receipt field, the EXACT expected audit counters,
    historical budget fields within their fixed method/caps/bounds, current read-only usage within the subtree /
    scope / metadata caps (no free-floor requirement: nothing is written), then finish a full input/source/runtime
    closure AND an output reseal before reporting ``valid``."""
    pin = _pin(ctx)
    base = mi.Context(root=ctx.root, workspace=ctx.workspace)
    with mi.guarded_stage(base) as opened, media._counting() as counter:
        code = {"sources": _sources(), "runtime": mi.runtime_identity()}
        auth = _authenticate(ctx, pin, counter)
        mctx = auth["mctx"]
        out_path = Path(os.fspath(out))
        out_dir = Path(os.path.abspath(out_path if out_path.is_absolute() else ctx.root / out_path))
        _need(out_dir == auth["assembly_root"] or out_dir.parent == auth["assembly_root"], "--out is not an assembly output")
        _safe_dir(ctx, out_dir)
        _same(sorted(e.name for e in out_dir.iterdir()), sorted(OUTPUT_NAMES), "published output names")
        raws = {}
        for name in OUTPUT_NAMES:
            path = out_dir / name
            counter.watch(path, "assembly_output")
            st = os.lstat(path)
            _need(stat.S_ISREG(st.st_mode) and st.st_size <= OUTPUT_CAPS[name], f"{name}: not a regular capped file")
            raws[name] = mi.read_bounded(path, OUTPUT_CAPS[name], name)
        for name in (BUNDLE_NAME, CLIP_NAME, SIDECAR_NAME):
            _need(raws[name] == mi.dump_json(auth["outputs"][name]), f"{name} differs from the frozen expected output")
        receipt = mi.strict_json(raws[RECEIPT_NAME], RECEIPT_NAME)
        published = {n: {"sha256": mi.sha256_bytes(raws[n]), "bytes": len(raws[n])} for n in (BUNDLE_NAME, CLIP_NAME, SIDECAR_NAME)}
        expected = _receipt(auth, pin, out_dir, published, code, None, expected_receipt_reads(auth["reg"]))
        _need(isinstance(receipt, dict) and set(receipt) == set(expected), "assembly receipt fields differ")
        for key in sorted(set(expected) - {"budget"}):
            _same(receipt[key], expected[key], f"assembly receipt {key}")     # includes the EXACT audit counters
        # historical observation: fixed keys/method/caps and valid bounds only (not authenticated by bounds)
        media._check_budget(receipt["budget"], "assembly receipt",
                            planned_bytes=media.TRANSIENT_FACTOR * sum(OUTPUT_CAPS.values()) + mi.DIR_ENTRY_RESERVE)
        # current read-only usage, retained partials/siblings included (no free-floor/reserve: nothing is written)
        subtree, scope = media._usage(auth["assembly_root"]), media._usage(mctx.scope)
        _need(subtree["total"] <= SUBTREE_CAP and scope["total"] <= mi.FINAL_OUTPUTS_CAP
              and scope["metadata"] <= mi.METADATA_AGGREGATE_CAP,
              "current assembly subtree / scope / metadata usage exceeds its cap")
        # fresh final closure: inputs/sources/runtime AND an output reseal (never trusting the receipt's own claims)
        _same(_closure(ctx, auth["files"]), {"files": auth["pinned"], **code}, "validation closure (inputs/sources/runtime)")
        _safe_dir(ctx, out_dir)
        _same(sorted(e.name for e in out_dir.iterdir()), sorted(OUTPUT_NAMES), "published output names (reseal)")
        for name in OUTPUT_NAMES:
            _same(media.sealed_sha256(out_dir / name, len(raws[name])), mi.sha256_bytes(raws[name]),
                  f"{name} bytes (reseal)")
        _need(not opened and not counter.reads("annotation") and not counter.reads("camera"), "forbidden opens")
        total = counter.snapshot()
    return {"status": "valid", "label": pin["label"], "family": output_family(pin["label"]),
            "receipt_sha256": mi.sha256_bytes(raws[RECEIPT_NAME]),
            "outputs": {n: mi.sha256_bytes(raws[n]) for n in OUTPUT_NAMES}, "audited_opens_total": total, "exit": 0}


# --------------------------------------------------------------------------
# CLI (production pin only; no hash/object/frame/ordinal/K/origin/seed/force/overwrite flags)


def main(argv: Sequence[str] | None = None, ctx: AssemblyContext | None = None) -> int:
    base = ctx or AssemblyContext()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("inspect")
    for stage in ("assemble", "validate"):
        sub.add_parser(stage).add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "inspect":
            result = inspect_assembly(base)
        elif args.stage == "assemble":
            result = assemble(base, out=args.out)
        else:
            result = validate_output(base, out=args.out)
    except mi.InputsError as exc:
        code, status = ((2, "usage") if isinstance(exc, mi.UsageError) else (3, "blocked") if isinstance(exc, mi.Blocked)
                        else (1, "error"))
        print(json.dumps({"stage": args.stage, "status": status, "exit": code,
                          "reason": f"{type(exc).__name__}: {str(exc)[:600]}"}, sort_keys=True), file=sys.stderr)
        return code
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("exit", 0))


if __name__ == "__main__":
    if not sys.dont_write_bytecode:
        raise SystemExit("Use python -B (for example: python -B -S tools/prototype_bootstrap.py "
                         "tools.workbench_hb_scene3_execution_control ...)")
    raise SystemExit(main())
