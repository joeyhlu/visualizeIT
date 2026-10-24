"""Scene-000003 UNIQUE-INSTANCE controlled-init ground-truth seed (isolated; CPU; frame 9 only).

This is a CONTROLLED ground-truth initialisation from the historical BOP annotation. It is NOT automatic RGB
geometry, NOT segmentation, NOT neural execution and NOT an independent accuracy measurement. It never
assembles, registers or admits anything, never chooses a principal-point origin and never touches K.

``plan-seed``           METADATA ONLY (zero annotation opens). Authenticates, hash-before-parse, the pinned
                        prerequisites receipt and the frozen parent controlled-seed policy, re-authenticates
                        the accepted media allowlist through the accepted media helpers (profile / K /
                        association / scout / three historical anchors; the allowlist core is re-derived), and
                        derives the ``val_primesense/000003/scene_gt.json`` descriptor ONLY from the
                        authenticated historical index (original storage metadata, expanded SHA-256 / size /
                        CRC-32, annotation role preserved). The annotation file is ``lstat``-ed only.
``extract-unique-seed`` With the guard active, re-derives the plan core under the pins, then (budgets first)
                        grants ONE exact read-only exception for the annotation file and makes four bounded
                        passes: full pre-hash; lexical scan (``TopLevelFrameExtractor('9')``) while re-hashing
                        the whole file; pre-publication closure; final closure after BOTH outputs. Only frame
                        9 is captured; numbers are opaque tokens; exactly ONE ``obj_id == 4`` record must exist;
                        its source-order position is the derived instance ordinal; ONLY that record's R/t are
                        converted (``select_pose``: proper SO(3), mm * 0.001 exactly once).

The opening association stays unchanged (ordinal null); the derived ordinal lives only in the NEW binding.
Trust: historical local integrity only (``upstream_authenticated: false``). Exit codes: 0 completed;
1 failure; 2 usage; 3 blocked.
"""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (checked helpers; never mutated)
from tools import workbench_hb_camera_k as ck  # noqa: E402  (pure reuse-path canonicalisation)
from tools import workbench_hb_scene3_profile as pf  # noqa: E402
from tools import workbench_hb_scene3_media as media  # noqa: E402  (accepted pins/auth/publisher; stages never called)
from tools import workbench_hb_opening_seed as opening  # noqa: E402  (ONLY TopLevelFrameExtractor + select_pose)

TOOL_VERSION = "1"
SCHEMA_PLAN = "workbench_hb_scene3_seed_plan_v1"
SCHEMA_SEED = "workbench_hb_scene3_controlled_seed_v1"
SCHEMA_BINDING = "workbench_hb_scene3_seed_binding_v1"
SCHEMA_FAILURE = "workbench_hb_scene3_seed_failure_v1"
SCHEMA_POLICY = "workbench_hb_scene3_controlled_seed_policy_v1"
SCENE = "000003"
OBJECT_ID = 4
FRAME_INDEX = 9
FRAME_ID = "s000009"
APPLIES_ONCE_TO = "s000010"
GT_MEMBER = f"val_primesense/{SCENE}/scene_gt.json"
# Accepted ORIGINAL cached roles for this one exact member. The historical pinned index records "metadata";
# manufactured fixtures use "evaluation"/"annotation". The original value is preserved verbatim; it never
# grants generic metadata or automatic RGB access. Every other role (rgb, model, ...) is refused.
ANNOTATION_ROLES = ("evaluation", "annotation", "metadata")
SEMANTIC_ROLE = "controlled_ground_truth_annotation"
ROLE_DISCLOSURE = ("the original cached role is preserved verbatim; for this exact canonical scene_gt.json member it is "
                   "treated semantically as controlled ground-truth annotation, and it does NOT grant generic metadata "
                   "access or automatic RGB/geometry access")
PLAN_NAME, SEED_NAME, BINDING_NAME = "seed-plan.json", "controlled-seed.json", "seed-binding.json"
FAILURE_NAME, INVALIDATED_PREFIX = "failure.json", "invalidated-"
SEED_SUBTREE = "controlled-seed-v1"

# Production pins (no CLI override exists).
PREREQUISITES_PIN = {"path": ".cache/video-workbench/derived/hb-metric-control-000003-v1/prerequisites-v1/"
                             "prerequisites-receipt.json",
                     "sha256": "f11b9a3c41da9dc499c8e2d5fff766e018facc73af1ce3620ca9e8842b108681"}
POLICY_PIN = {"path": ".cache/video-workbench/derived/validation/hb-scene3-controlled-seed-policy-v1.json",
              "sha256": "063d4f59cc2293bfd5dae264e6957125e3c283dac8c294a5311228f70255ff72"}
STAGE_RECEIPT_PINS = {"allowlist": "e2dfa6f7ec4987d3d6c18eceb96fe739a76d05e69b4b1f44ff847078466a83d3",
                      "model_receipt": "106048f73191b2f1cb4feb8019534f64bde90233ae8dd9bf4f6dd5433e4b1ec7",
                      "video_receipt": "67ce007ebde5ea02e02841f8da39f41cdd10f02fe8c3ce421788e40989b842b9"}

ANNOTATION_CAP = 8 << 20                  # per pass, independently of the extractor's own 64 MiB cap
PREREQ_CAP = media.PREREQ_CAP
POLICY_CAP = 16 << 10
STAGE_RECEIPT_CAPS = {"allowlist": media.ALLOWLIST_CAP, "model_receipt": media.MODEL_RECEIPT_CAP,
                      "video_receipt": media.VIDEO_RECEIPT_CAP}
SUBTREE_CAP = 64 << 10                    # plan + seed + binding + failures + staging + partials, all stages
PLAN_CAP, SEED_CAP, BINDING_CAP, FAILURE_CAP = 24 << 10, 8 << 10, 16 << 10, 4 << 10
READ_CHUNK = 64 << 10
MAX_LEXEME = 64                           # characters of a number that is actually converted
MAX_OBJ_ID_DIGITS = 6

POLICY_KEYS = frozenset({"schema", "timestamp_utc", "reviewer", "scene", "object_id", "frame_id", "frame_index", "policy",
                         "frozen_before_annotation_access", "annotation_values_read", "association_sha256",
                         "prerequisites_sha256", "controlled_ground_truth_seed_authorized", "applies_once_to_frame_id",
                         "reuse_on_recovery", "scored_pose_values_materialized", "automatic_geometry_annotation_access",
                         "principal_point_origin", "principal_point_origin_review_required_for_assembly",
                         "translation", "maximum_annotation_bytes", "seed_subtree_total_cap_bytes", "quality_claim",
                         "phone_performance_claim"})
PREREQ_KEYS = frozenset({"schema", "tool_version", "label", "pins_kind", "status", "flags", "trust", "scene", "inputs",
                         "outputs", "verified", "association", "camera_k", "principal_point", "unresolved",
                         "not_published_here", "reads", "budget", "sources", "runtime", "preparation_scope", "closure"})
TRUST = {"trust_level": "historical_local_integrity", "upstream_authenticated": False,
         "basis": ("pinned prerequisites receipt and parent policy, the accepted media allowlist re-derived from the "
                   "profile/K/association/scout/historical anchors; the annotation descriptor comes only from the "
                   "authenticated historical index")}
SEED_FLAGS = {"controlled_init": True, "ground_truth_seed": True, "initial_ground_truth_read": True,
              "automatic_geometry_annotation_access": False, "scored_pose_values_materialized": False,
              "neural_executed": False, "gpu_used": False, "segmentation": False, "assembled": False,
              "registered": False, "admitted": False, "clipspec_published": False, "surface_package_published": False,
              "quality_claim": False, "independent_accuracy_claim": False, "mobile_verified": False,
              "phone_performance_claim": False, "family": mi.FAMILY, "head_sha256": mi.FLAGS["head_sha256"]}
DISCLOSURE = ("controlled-init seed: ALL bytes of the historical BOP scene_gt.json annotation file are READ and HASHED "
              "in four bounded passes (pre-hash, lexical scan, pre-publication closure, final closure); ONLY the "
              "frame-9 value is captured, and ONLY the selected obj_id 4 record's pose tokens are interpreted, to "
              "initialise one tracker pose. Not automatic geometry, not an evaluator-free run, not independent accuracy")
ANNOTATION_ACCESS = {"whole_file_bytes_read_and_hashed": True, "bounded_passes": 4,
                     "frame_values_captured": [str(FRAME_INDEX)], "records_interpreted": "selected obj_id 4 record only",
                     "other_frames_values_materialized": False, "other_pose_values_converted": 0,
                     "semantic_role": SEMANTIC_ROLE}
UNRESOLVED = ["assembly", "assembly_selection", "principal_point_origin", "artwork_placement", "segmentation",
              "registration", "admission"]
PLAN_CORE_KEYS = frozenset({"schema", "tool_version", "label", "pins_kind", "scene", "object_id", "frame_index",
                            "frame_id", "policy", "inputs", "bound_outputs", "anchors", "annotation_source", "trust",
                            "flags", "input_budget", "seed_subtree", "unresolved", "preparation_scope"})
PLAN_KEYS = PLAN_CORE_KEYS | {"sources", "runtime", "budget", "reads", "closure"}
PLAN_FLAGS = {**{k: v for k, v in SEED_FLAGS.items() if k != "initial_ground_truth_read"}, "annotation_opens": 0,
              "initial_ground_truth_read": False}


def _assert_helper_assumptions() -> None:
    media._assert_helper_assumptions()
    if (opening.SEED_FRAME_KEY != str(FRAME_INDEX) or opening.MAX_INSTANCES != 64 or opening.MAX_DEPTH != 64
            or opening.MAX_CAPTURE_BYTES != 1 << 20 or mi.MM_TO_M != 0.001 or media.SCENE != SCENE):
        raise mi.InputsError("accepted opening-seed/media helper assumptions changed")


# --------------------------------------------------------------------------
# context and pins


@dataclass(frozen=True)
class SyntheticSeedPins:
    """TEST-ONLY pins for manufactured fixtures (never settable from the CLI). Outputs are then labelled
    ``TEST_ONLY`` / ``synthetic_test_only`` and production validation refuses them."""
    media: media.SyntheticMediaPins
    prerequisites_sha256: str
    policy_review_sha256: str


@dataclass(frozen=True)
class SeedContext:
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"
    preparation: Path = ROOT / media.PREPARATION_PIN
    test_only_synthetic_pins: SyntheticSeedPins | None = None

    def media_context(self) -> media.MediaContext:
        s = self.test_only_synthetic_pins
        return media.MediaContext(self.root, self.workspace, self.preparation, s.media if s is not None else None)

    def mi(self) -> mi.Context:
        return self.media_context().mi()


def pins(ctx: SeedContext) -> dict:
    s = ctx.test_only_synthetic_pins
    if s is None:
        return {"kind": "historical", "label": "PRODUCTION_PINNED", "media": media.pins(ctx.media_context()),
                "prerequisites_sha256": PREREQUISITES_PIN["sha256"], "policy_sha256": POLICY_PIN["sha256"],
                "stage_receipts": dict(STAGE_RECEIPT_PINS)}
    if not isinstance(s, SyntheticSeedPins) or not mi._is_sha(s.prerequisites_sha256) or not mi._is_sha(s.policy_review_sha256):
        raise mi.UsageError("test-only synthetic seed pins are malformed")
    return {"kind": "synthetic_test_only", "label": "TEST_ONLY", "media": media.pins(ctx.media_context()),
            "prerequisites_sha256": s.prerequisites_sha256, "policy_sha256": s.policy_review_sha256,
            "stage_receipts": None}                     # bound only through the pinned prerequisites receipt


# --------------------------------------------------------------------------
# ledger (whole seed subtree <= 64 KiB) and failures


class SeedLedger:
    """Duck-compatible with ``media.publish``/``media._commit``. Before every write: the WHOLE seed subtree
    (every stage leaf, retained failures, staging partials and aliases) within 64 KiB, the preparation scope
    within the existing 12 MiB / 512 KiB aggregates (everything here is JSON metadata), the free floor and a
    fresh global derived check. Normal writes keep room for a failure record."""

    def __init__(self, mctx: mi.Context, out_dir: Path, stage: str):
        self.ctx, self.dir, self.stage, self.scope = mctx, out_dir, stage, mctx.scope
        self.subtree = out_dir.parent
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
                raise mi.Blocked(f"{self.stage}: seed subtree would peak at {subtree['total'] + done + peak} bytes, "
                                 f"above {SUBTREE_CAP - reserve} (retained failures and staging included)")
            if scope["total"] + done + peak > mi.FINAL_OUTPUTS_CAP - reserve:
                raise mi.Blocked(f"{self.stage}: preparation scope would exceed {mi.FINAL_OUTPUTS_CAP - reserve}")
            if scope["metadata"] + done + peak > mi.METADATA_AGGREGATE_CAP - reserve:
                raise mi.Blocked(f"{self.stage}: scope metadata would exceed {mi.METADATA_AGGREGATE_CAP - reserve}")
            done += nbytes
        if mi.shutil.disk_usage(self.ctx.workspace).free - peak_max < mi.FREE_RESERVE:
            raise mi.Blocked("the 1 GiB free-disk reserve would be violated")
        mi.workspace_budget(self.ctx, peak_max)
        return {"subtree": subtree, "scope": scope}

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        return self.check_sequence([(nbytes, True)], failure=failure)

    def owned_file(self, name: str) -> bool:
        if name not in self.owned:
            return False
        try:
            st = os.lstat(self.dir / name)
        except OSError:
            return False
        return stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == self.owned[name]

    def summary(self) -> dict:
        return {"stage": self.stage, "files": dict(self.files), "owned": sorted(self.owned),
                "subtree_usage_now": media._usage(self.subtree), "scope_usage_now": media._usage(self.scope),
                "caps": {"seed_subtree": SUBTREE_CAP, "scope_total": mi.FINAL_OUTPUTS_CAP,
                         "scope_metadata": mi.METADATA_AGGREGATE_CAP, "failure_record": FAILURE_CAP},
                "publication": "exclusive rename (Windows)" if media.RENAME_EXCLUSIVE else "link+unlink (2x counted)"}


QUARANTINE_ATTEMPTS = 8


def _presence(ledger: SeedLedger, name: str) -> str:
    """Explicit state of an owned authoritative name: ``present_owned`` | ``absent`` | ``replaced`` |
    ``unknown``. A lookup error is ``unknown`` and NEVER treated as proof of disappearance."""
    try:
        st = os.lstat(ledger.dir / name)
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unknown"
    if stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == ledger.owned.get(name):
        return "present_owned"
    return "replaced"


def _quarantine(ledger: SeedLedger, name: str) -> dict:
    """Move ONE owned output (identity re-checked) to a UNIQUE ``invalidated-...`` name exclusively. An existing
    (possibly foreign) name is never overwritten, renamed or deleted; bytes are kept. Windows ``os.rename``
    refuses an existing destination; POSIX uses link (refuses an existing name) + unlink, with the transient
    alias checked against the budget first. Bounded attempts. NEVER raises: every error (budget refusal,
    denied move, lookup failure) is recorded and the caller continues with the next output. The post-move
    identity inspection is best-effort; an unknown identity is reported as unknown."""
    record: dict = {"name": name}
    try:
        state = _presence(ledger, name)
        if state != "present_owned":
            record["skipped"] = f"authoritative name is {state}; left untouched"
            return record
        identity = ledger.owned[name]
        src = ledger.dir / name
        for attempt in range(QUARANTINE_ATTEMPTS):
            dst = ledger.dir / (f"{INVALIDATED_PREFIX}{name}" if attempt == 0
                                else f"{INVALIDATED_PREFIX}{secrets.token_hex(4)}-{name}")
            if os.path.lexists(dst):
                continue                                   # foreign or earlier quarantine: never touched
            if _presence(ledger, name) != "present_owned":  # identity re-check immediately before the move
                record["skipped"] = "owned file changed identity or became unreadable before quarantine"
                return record
            try:
                if media.RENAME_EXCLUSIVE:
                    os.rename(src, dst)
                else:
                    ledger.check_sequence([(os.lstat(src).st_size, True)], failure=True)
                    os.link(src, dst)
                    os.unlink(src)
            except FileExistsError:
                continue
            record["retained_as"] = dst.name
            try:                                           # best-effort post-move inspection
                st = os.lstat(dst)
                record["identity_preserved"] = (st.st_dev, st.st_ino) == identity
            except OSError as exc:
                record["identity_preserved"] = None
                record["post_move_lookup_error"] = type(exc).__name__
            return record
        record["error"] = f"no free quarantine name after {QUARANTINE_ATTEMPTS} attempts"
    except Exception as exc:  # noqa: BLE001 - cleanup must continue; reported, never hidden
        record["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return record


def _failure(ledger: SeedLedger, stage: str, exc: BaseException) -> None:
    """Quarantine EVERY owned authoritative output independently (bytes kept, foreign names untouched), then
    publish a bounded valid-JSON failure record (exception text only, truncated; never annotation data).
    Remaining or uncertain authoritative outputs are inventoried explicitly. This function NEVER raises, so
    the original stage exception always propagates; any cleanup/publication problem goes to stderr."""
    try:
        records = []
        for name in sorted(ledger.owned):
            if name == FAILURE_NAME:
                continue
            try:
                records.append(_quarantine(ledger, name))
            except Exception as cleanup_exc:  # noqa: BLE001 - e.g. a replaced/mocked quarantine raising
                records.append({"name": name, "error": f"{type(cleanup_exc).__name__}: {str(cleanup_exc)[:160]}"})
        states = {name: _presence(ledger, name) for name in sorted(ledger.owned) if name != FAILURE_NAME}
        remaining = sorted(n for n, s in states.items() if s == "present_owned")
        uncertain = sorted(n for n, s in states.items() if s == "unknown")
        retained: dict = {}
        try:
            for entry in sorted(ledger.dir.iterdir()):
                try:
                    retained[entry.name] = os.lstat(entry).st_size
                except OSError:
                    retained[entry.name] = None             # present but size unknown
        except OSError:
            retained = {"inventory_error": True}
        body = {"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage,
                "status": "failed_authoritative_outputs_remain" if remaining or uncertain else "failed",
                "reason": f"{type(exc).__name__}: {str(exc)[:600]}", "completed_receipt_published": False,
                "authoritative_files_remaining": remaining, "authoritative_files_uncertain": uncertain,
                "authoritative_states": states, "quarantined_outputs": records, "retained_files": retained}
        data = mi.dump_json(body)
        if len(data) > FAILURE_CAP:
            body.update(reason=body["reason"][:200], retained_files={"count": len(retained)},
                        quarantined_outputs=[{k: r.get(k) for k in ("name", "retained_as", "error")} for r in records])
            data = mi.dump_json(body)
    except Exception as inventory_exc:  # noqa: BLE001 - minimal record; never mask the original error
        body = {"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage,
                "status": "failed_authoritative_outputs_uncertain",
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}", "completed_receipt_published": False,
                "inventory_error": type(inventory_exc).__name__}
        data = mi.dump_json(body)
    try:
        if len(data) > FAILURE_CAP:
            raise mi.Blocked("failure record exceeds its cap")
        media.publish(ledger, FAILURE_NAME, data, failure=True)       # subtree/scope/free/global checks inside
    except Exception as publish_exc:  # noqa: BLE001 - never mask the original error
        try:
            line = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            sys.stderr.write(f"failure.json NOT published ({type(publish_exc).__name__}); record on ONE line:\n{line}\n")
        except Exception:  # noqa: BLE001
            pass


def _sources() -> dict:
    return dict(sorted({**media._sources(),
                        **{Path(m.__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(m.__file__))
                           for m in (opening,)},
                        Path(__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(__file__))}.items()))


def _out_dir(mctx: mi.Context, out: Any) -> Path:
    out_dir = mi.check_output_dir(mctx, out)
    if out_dir.parent != Path(os.path.abspath(mctx.scope / SEED_SUBTREE)):
        raise mi.UsageError(f"--out must be a NEW directory directly inside {SEED_SUBTREE}/ of the preparation scope")
    return out_dir


# --------------------------------------------------------------------------
# pure validation


def validate_policy(doc: Any, p: Mapping) -> dict:
    """Exact frozen parent policy: CAD 4, frame 9, exactly-one rule, bound association/prerequisites, caps."""
    mp = p["media"]
    if not isinstance(doc, dict) or set(doc) != POLICY_KEYS:
        raise mi.InputsError("seed policy has missing or unknown fields")
    fixed = {"schema": SCHEMA_POLICY, "scene": SCENE, "frame_id": FRAME_ID, "applies_once_to_frame_id": APPLIES_ONCE_TO,
             "association_sha256": mp["association_sha256"], "prerequisites_sha256": p["prerequisites_sha256"]}
    for key, value in fixed.items():
        if doc[key] != value:
            raise mi.InputsError(f"seed policy {key!r} differs from the pinned contract")
    trues = ("frozen_before_annotation_access", "controlled_ground_truth_seed_authorized",
             "principal_point_origin_review_required_for_assembly")
    falses = ("annotation_values_read", "reuse_on_recovery", "scored_pose_values_materialized",
              "automatic_geometry_annotation_access", "quality_claim", "phone_performance_claim")
    if any(doc[k] is not True for k in trues) or any(doc[k] is not False for k in falses) \
            or doc["principal_point_origin"] is not None:
        raise mi.InputsError("seed policy flags differ from the frozen controlled-seed contract")
    ints = {"object_id": mp["association_object_id"], "frame_index": FRAME_INDEX,
            "maximum_annotation_bytes": ANNOTATION_CAP, "seed_subtree_total_cap_bytes": SUBTREE_CAP}
    if any(type(doc[k]) is not int or doc[k] != v for k, v in ints.items()) or mp["association_object_id"] != OBJECT_ID:
        raise mi.InputsError("seed policy object/frame/caps differ (booleans are not integers)")
    for key in ("timestamp_utc", "reviewer", "policy", "translation"):
        if not isinstance(doc[key], str) or not doc[key].strip():
            raise mi.InputsError(f"seed policy {key!r} must be non-empty text")
    return doc


def validate_prerequisites(doc: Any, p: Mapping, scope_rel: str) -> dict:
    mp = p["media"]
    if not isinstance(doc, dict) or set(doc) != PREREQ_KEYS:
        raise mi.InputsError("prerequisites receipt has missing or unknown fields")
    fixed = {"schema": media.SCHEMA_PREREQ, "tool_version": media.TOOL_VERSION, "label": mp["label"],
             "pins_kind": mp["kind"], "status": "prepared_media_only", "flags": media.MEDIA_FLAGS, "trust": media.TRUST,
             "scene": SCENE, "principal_point": media.PRINCIPAL_POINT, "unresolved": media.UNRESOLVED,
             "preparation_scope": scope_rel}
    for key, value in fixed.items():
        if doc[key] != value:
            raise mi.InputsError(f"prerequisites receipt {key!r} differs (a TEST_ONLY receipt cannot pass as production)")
    inputs, outputs, verified = doc["inputs"], doc["outputs"], doc["verified"]
    try:
        bound = {k: inputs[k]["sha256"] for k in ("allowlist", "model_receipt", "video_receipt", "profile", "camera_k",
                                                    "association")}
        paths = {k: inputs[k]["path"] for k in ("allowlist", "model_receipt", "video_receipt")}
        outs = {k: outputs[k]["sha256"] for k in ("glb", "video")}
    except (KeyError, TypeError):
        raise mi.InputsError("prerequisites inputs/outputs malformed") from None
    if (bound["profile"], bound["camera_k"], bound["association"]) != (mp["profile_sha256"], mp["camera_k_sha256"],
                                                                       mp["association_sha256"]):
        raise mi.InputsError("prerequisites receipt does not bind the pinned profile/K/association")
    if p["stage_receipts"] is not None and {k: bound[k] for k in STAGE_RECEIPT_PINS} != p["stage_receipts"]:
        raise mi.InputsError("prerequisites receipt does not bind the accepted allowlist/model/video receipts")
    if not all(mi._is_sha(v) for v in (*bound.values(), *outs.values())) or not all(isinstance(v, str) for v in paths.values()):
        raise mi.InputsError("prerequisites hashes/paths malformed")
    if (not isinstance(verified, Mapping) or verified.get("allowlist_core_rederived_from_pinned_inputs") is not True
            or verified.get("glb_rebuilt_byte_identical") is not True
            or verified.get("model_receipt_declarations_reconstructed") is not True
            or verified.get("video_receipt_declarations_reconstructed") is not True):
        raise mi.InputsError("prerequisites receipt does not record its reconstructions")
    assoc = doc["association"]
    if not isinstance(assoc, Mapping) or assoc.get("object_id") != OBJECT_ID or assoc.get("instance_ordinal") is not None:
        raise mi.InputsError("prerequisites association must be CAD 4 with ordinal null")
    return {"bound": bound, "paths": paths, "outputs": outs}


def validate_annotation_row(row: Any, *, root: Path, canonical: Path, max_bytes: int = ANNOTATION_CAP) -> dict:
    """The exact index-bound ``scene_gt.json`` REUSE row (no file access). The annotation role is preserved,
    never relabelled as RGB/metadata."""
    member = GT_MEMBER
    if not isinstance(row, Mapping):
        raise mi.InputsError(f"{member}: index row missing")
    ints = ("sourceMethod", "sourceFlags", "sourceCompressedBytes", "sourceOffset", "expandedBytes", "crc32", "method",
            "compressedBytes")
    if any(type(row.get(k)) is not int or row.get(k) < 0 for k in ints):
        raise mi.InputsError(f"{member}: ZIP metadata must be non-negative integers (booleans are invalid)")
    roles, location = row.get("allowedRoles"), row.get("location")
    if (row.get("datasetRevision") != mi.DATASET_REVISION or row.get("archive") != mi.RGB_ARCHIVE
            or row.get("sourceMember") != member or not isinstance(location, Mapping) or location.get("kind") != "reuse"
            or not isinstance(roles, list) or len(roles) != 1 or roles[0] not in ANNOTATION_ROLES):
        raise mi.InputsError(f"{member}: revision/archive/member/reuse location/annotation role differ")
    size = row["expandedBytes"]
    if (row["method"] != 0 or row["compressedBytes"] != size or not 0 < size <= max_bytes or row["crc32"] > 0xFFFFFFFF
            or row["sourceMethod"] not in (0, 8) or row["sourceFlags"] & ~2048 or row["sourceCompressedBytes"] == 0
            or row["sourceCompressedBytes"] > size + (1 << 20)
            or (row["sourceMethod"] == 0 and row["sourceCompressedBytes"] != size)):
        raise mi.InputsError(f"{member}: stored/original ZIP method, flags or sizes are inconsistent or above the cap")
    if not (mi._is_sha(row.get("compressedSHA256")) and row.get("compressedSHA256") == row.get("uncompressedSHA256")):
        raise mi.InputsError(f"{member}: stored SHA-256 must equal the expanded SHA-256")
    form = ck._reuse_form(root, location.get("path"), canonical)
    return {"member": member, "archive": mi.RGB_ARCHIVE, "dataset_revision": mi.DATASET_REVISION, "role": roles[0],
            "semantic_role": SEMANTIC_ROLE, "role_disclosure": ROLE_DISCLOSURE,
            "location_kind": "reuse", "location_path_form": form,
            "stored": {"method": 0, "bytes": size, "sha256": row["compressedSHA256"]},
            "expanded_bytes": size, "expanded_sha256": row["uncompressedSHA256"], "crc32": row["crc32"],
            "original_zip": {"method": row["sourceMethod"], "compressed_bytes": row["sourceCompressedBytes"],
                             "flags": row["sourceFlags"], "offset": row["sourceOffset"]}}


class _Lexeme:
    """An opaque JSON number token: kept as text, never converted unless explicitly selected."""
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def decode_frame_value(captured: bytes) -> list:
    """Strict decode of the captured frame-9 value with OPAQUE numbers (no float/int conversion of any value,
    including unselected 1e400 or huge integers); duplicate keys and NaN/Infinity constants refused."""
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise mi.InputsError("frame 9 annotation repeats a key")
            out[key] = value
        return out

    def constant(token):
        raise mi.InputsError("frame 9 annotation holds a non-finite constant")
    try:
        value = json.loads(captured.decode("utf-8"), object_pairs_hook=pairs, parse_float=_Lexeme, parse_int=_Lexeme,
                           parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise mi.InputsError(f"frame 9 annotation is not strict JSON ({type(exc).__name__})") from None
    if not isinstance(value, list) or not 1 <= len(value) <= opening.MAX_INSTANCES:
        raise mi.InputsError("frame 9 annotation must be a non-empty bounded list")
    for entry in value:
        if (not isinstance(entry, dict) or set(entry) != {"cam_R_m2c", "cam_t_m2c", "obj_id"}
                or not isinstance(entry["obj_id"], _Lexeme)
                or not re.fullmatch(rf"[0-9]{{1,{MAX_OBJ_ID_DIGITS}}}", entry["obj_id"].text)
                or not isinstance(entry["cam_R_m2c"], list) or len(entry["cam_R_m2c"]) != 9
                or not isinstance(entry["cam_t_m2c"], list) or len(entry["cam_t_m2c"]) != 3
                or not all(isinstance(v, _Lexeme) for v in entry["cam_R_m2c"] + entry["cam_t_m2c"])):
            raise mi.InputsError("frame 9 entries must be {cam_R_m2c[9], cam_t_m2c[3], obj_id} with integer obj_id "
                                 "lexemes and numeric pose tokens (booleans/strings refused)")
    return value


def _convert(lexeme: _Lexeme) -> float:
    if len(lexeme.text) > MAX_LEXEME:
        raise mi.InputsError("selected pose number token is too long")
    value = float(lexeme.text)
    if not math.isfinite(value):
        raise mi.InputsError("selected pose number overflows or is non-finite")
    return value


def select_unique_seed(frame: list) -> dict:
    """Exactly ONE obj_id == 4 record (by exact integer lexeme); its source-order position is the ordinal.
    ONLY that record's R/t are converted; every other pose token stays opaque."""
    matches = [i for i, entry in enumerate(frame) if entry["obj_id"].text == str(OBJECT_ID)]
    if len(matches) != 1:
        raise mi.Blocked(f"frame 9 holds {len(matches)} obj_id == {OBJECT_ID} records; exactly one is required")
    ordinal = matches[0]
    selected = frame[ordinal]
    converted = {"cam_R_m2c": [_convert(v) for v in selected["cam_R_m2c"]],
                 "cam_t_m2c": [_convert(v) for v in selected["cam_t_m2c"]], "obj_id": OBJECT_ID}
    entries = [converted if i == ordinal else {"cam_R_m2c": e["cam_R_m2c"], "cam_t_m2c": e["cam_t_m2c"],
                                               "obj_id": int(e["obj_id"].text)}
               for i, e in enumerate(frame)]
    try:
        pose = opening.select_pose(entries, OBJECT_ID, ordinal)
    except (mi.InputsError, mi.Blocked):
        raise mi.InputsError("the selected record is not a finite proper rigid transform") from None
    return {"pose": pose, "ordinal": ordinal, "records": len(frame), "matches": 1,
            "selected_values_converted": 12, "other_pose_values_converted": 0,
            "obj_id_lexemes_converted_to_int": len(frame)}


# --------------------------------------------------------------------------
# bounded annotation passes (8 MiB per pass, independent of the extractor's own cap)


def annotation_pass(path: Path, descriptor: Mapping, *, scanner: Any = None) -> dict:
    """One bounded pass: ``lstat`` size first (a changed size is refused BEFORE reading), then the whole file
    in chunks (never beyond the expected size or 8 MiB) with SHA-256, CRC-32 and size compared at the end.
    With a scanner, every chunk is also fed to it while hashing."""
    expected = descriptor["expanded_bytes"]
    if type(expected) is not int or not 0 < expected <= ANNOTATION_CAP:
        raise mi.InputsError("annotation size is outside the 8 MiB cap")
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_size != expected:
        raise mi.InputsError("annotation file is not a regular file of the index-bound size")
    h, crc, count = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(min(READ_CHUNK, expected + 1 - count))
            if not block:
                break
            count += len(block)
            if count > expected or count > ANNOTATION_CAP:
                raise mi.InputsError("annotation file grew beyond its index-bound size")
            h.update(block)
            crc = binascii.crc32(block, crc)
            if scanner is not None:
                scanner.feed(block)
    result = {"bytes": count, "sha256": h.hexdigest(), "crc32": crc & 0xFFFFFFFF}
    if (result["bytes"], result["sha256"], result["crc32"]) != (expected, descriptor["expanded_sha256"], descriptor["crc32"]):
        raise mi.InputsError("annotation bytes differ from the index-bound size/SHA-256/CRC-32")
    return result


# --------------------------------------------------------------------------
# shared metadata authentication (guard active; no annotation exception)


def _locate(mctx: mi.Context, value: Any, name: str | None, what: str, *, inside: bool) -> Path:
    return media._locate(mctx, value, name, what, inside=inside)


def _authenticate(mctx: mi.Context, p: Mapping, counter: Any, prereq_path: Path, policy_path: Path) -> dict:
    """Pinned-SHA-before-parse metadata authentication and EXACT derivation of the plan core inputs."""
    mp = p["media"]
    scope_rel = mi.repo_rel(mctx, mctx.scope)
    counter.watch(prereq_path, "prerequisites")
    counter.watch(policy_path, "policy")
    prereq_doc = mi.strict_json(media._read_sealed(prereq_path, PREREQ_CAP, p["prerequisites_sha256"],
                                                   "prerequisites receipt"), "prerequisites receipt")
    pre = validate_prerequisites(prereq_doc, p, scope_rel)
    policy_raw = media._read_sealed(policy_path, POLICY_CAP, p["policy_sha256"], "seed policy")
    policy = validate_policy(mi.strict_json(policy_raw, "seed policy"), p)
    allow_doc, allow_path, payload_root = media._load_allowlist(mctx, mp, counter, pre["paths"]["allowlist"],
                                                                pre["bound"]["allowlist"])
    a, profile_path, k_path, assoc_path = media._reauthenticate(mctx, mp, counter, allow_doc)
    receipt_paths = {}
    for key in ("model_receipt", "video_receipt"):
        name = media.MODEL_RECEIPT_NAME if key == "model_receipt" else media.VIDEO_RECEIPT_NAME
        path = _locate(mctx, pre["paths"][key], name, key, inside=True)
        counter.watch(path, key)
        if media.sealed_sha256(path, STAGE_RECEIPT_CAPS[key]) != pre["bound"][key]:
            raise mi.InputsError(f"{key} bytes differ from the prerequisites binding")
        receipt_paths[key] = path
    index_path = a["auth"]["anchor_paths"]["resources_index"]
    pp = mp["profile_pins"]
    index = mi.strict_json(media._read_sealed(index_path, mi.MAX_ANCHOR_BYTES["resources_index"],
                                              pp["anchors"]["resources_index"], "resources index"), "resources index")
    gt_path = ck.canonical_reuse_path(mctx.root, pp["reuse_root"], GT_MEMBER)
    descriptor = validate_annotation_row(index["resources"].get(GT_MEMBER), root=mctx.root, canonical=gt_path)
    del index
    counter.watch(gt_path, "annotation")
    mi.contained_input(mctx, gt_path, "annotation source", allow_denied=True)      # lstat only; never opened here
    mi.outside_scope_input(mctx, gt_path, "annotation source")
    st = os.lstat(gt_path)
    if not stat.S_ISREG(st.st_mode) or st.st_size != descriptor["expanded_bytes"] or st.st_size > ANNOTATION_CAP:
        raise mi.Blocked("annotation source is missing, not regular, not of the index size or above 8 MiB")
    if counter.reads("annotation") or counter.reads("camera"):
        raise mi.GuardError("annotation/camera bytes were opened during metadata authentication")
    allow_total = allow_doc["input_budget"].get("total")
    if type(allow_total) is not int:
        raise mi.InputsError("allowlist input accounting malformed")
    files = {"prerequisites": (prereq_path, PREREQ_CAP), "policy": (policy_path, POLICY_CAP),
             "allowlist": (allow_path, media.ALLOWLIST_CAP),
             **{k: (v, STAGE_RECEIPT_CAPS[k]) for k, v in receipt_paths.items()},
             **media._input_files(mctx, a, profile_path, k_path, assoc_path)}
    pinned = {"prerequisites": p["prerequisites_sha256"], "policy": p["policy_sha256"],
              "allowlist": pre["bound"]["allowlist"], "model_receipt": pre["bound"]["model_receipt"],
              "video_receipt": pre["bound"]["video_receipt"], **media._pinned_inputs(mp)}
    core = {
        "schema": SCHEMA_PLAN, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"], "scene": SCENE,
        "object_id": OBJECT_ID, "frame_index": FRAME_INDEX, "frame_id": FRAME_ID,
        "policy": {"path": mi.repo_rel(mctx, policy_path), "sha256": p["policy_sha256"], "bytes": len(policy_raw),
                   "rule": "exactly one obj_id == 4 record in frame 9; source-order ordinal derived after uniqueness",
                   "applies_once_to_frame_id": APPLIES_ONCE_TO, "reuse_on_recovery": False},
        "inputs": {"prerequisites": {"path": mi.repo_rel(mctx, prereq_path), "sha256": p["prerequisites_sha256"]},
                   **{k: {"path": mi.repo_rel(mctx, files[k][0]), "sha256": pinned[k]}
                      for k in ("allowlist", "model_receipt", "video_receipt")},
                   **{k: dict(allow_doc["inputs"][k]) for k in ("profile", "camera_k", "association")}},
        "bound_outputs": {k: {"sha256": v, "reopened": False} for k, v in pre["outputs"].items()},
        "anchors": allow_doc["anchors"],
        "annotation_source": {**descriptor, "canonical_path": mi.repo_rel(mctx, gt_path), "stat_bytes": st.st_size,
                              "planned_passes": ["full_prehash", "lexical_scan_with_full_hash",
                                                 "pre_publication_closure", "final_closure_after_both_outputs"],
                              "cap_bytes_per_pass": ANNOTATION_CAP},
        "trust": dict(TRUST), "flags": dict(PLAN_FLAGS),
        "input_budget": {**mi.input_budget(allow_total, len(policy_raw) + descriptor["expanded_bytes"], 0),
                         "note": ("accepted allowlist total (base + association) + the new policy bytes + the actual "
                                  "annotation bytes; cached payloads are not double-counted")},
        "seed_subtree": {"path": mi.repo_rel(mctx, mctx.scope / SEED_SUBTREE), "cap_bytes": SUBTREE_CAP},
        "unresolved": list(UNRESOLVED), "preparation_scope": scope_rel}
    return {"core": core, "files": files, "pinned": dict(sorted(pinned.items())), "gt_path": gt_path,
            "descriptor": descriptor, "gt_identity": [st.st_size, st.st_mtime_ns, st.st_ino]}


def _state(files: Mapping) -> dict:
    """Bounded input hashes (refusals reported, not raised) + THIS module's source closure + runtime."""
    current = media._state(files)
    current["sources"] = _sources()
    return current


def _check(before: Mapping, current: Mapping, what: str) -> None:
    if current != before:
        changed = sorted(k for k in set(before) | set(current) if before.get(k) != current.get(k))
        raise mi.InputsError(f"{what}: closure changed {changed}")


# --------------------------------------------------------------------------
# stages


def plan_seed(ctx: SeedContext, *, prerequisites: Any, expected_prerequisites_sha256: str, policy_review: Any,
              expected_policy_review_sha256: str, out: Any) -> dict:
    _assert_helper_assumptions()
    p = pins(ctx)
    for given, pinned, what in ((expected_prerequisites_sha256, p["prerequisites_sha256"], "prerequisites"),
                                (expected_policy_review_sha256, p["policy_sha256"], "policy review")):
        if mi.check_sha(given, f"expected {what} SHA-256") != pinned:
            raise mi.Blocked(f"expected {what} SHA-256 is not the accepted pin; no trust can be derived")
    mctx = ctx.mi()
    out_dir = _out_dir(mctx, out)
    prereq_path = _locate(mctx, prerequisites, media.PREREQ_NAME, "prerequisites receipt", inside=True)
    policy_path = _locate(mctx, policy_review, None, "seed policy", inside=False)
    ledger = SeedLedger(mctx, out_dir, "plan-seed")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened, media._counting() as counter:
            counter.watch_prefix(out_dir, "stage_output")
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(PLAN_CAP, True)])                          # prospective, before any read
            budget = mi.workspace_budget(mctx, media.TRANSIENT_FACTOR * PLAN_CAP + mi.DIR_ENTRY_RESERVE)
            mi.input_budget(p["media"]["base_input_bytes"] + media.ASSOCIATION_CAP, POLICY_CAP + ANNOTATION_CAP, 0)
            auth = _authenticate(mctx, p, counter, prereq_path, policy_path)
            if opened:
                raise mi.GuardError("payload bytes were opened by the metadata plan")
            before = {"files": auth["pinned"], **code, "annotation_identity": auth["gt_identity"]}
            def current() -> dict:
                st = os.lstat(auth["gt_path"])
                return {**_state(auth["files"]), "annotation_identity": [st.st_size, st.st_mtime_ns, st.st_ino]}
            _check(before, current(), "pre-publication")
            reads = {"snapshot": PLAN_SNAPSHOT_NOTE, "audited_opens": counter.snapshot(),
                     "annotation_opens": 0, "camera_opens": 0, "payload_opens": 0,
                     "post_publication_planned": {"input_hashes": len(auth["files"]), "output_hashes": 1}}
            doc = {**auth["core"], "sources": code["sources"], "runtime": code["runtime"], "budget": budget, "reads": reads,
                   "closure": {"before_files": auth["pinned"], "annotation": PLAN_ANNOTATION_CLOSURE,
                               "pre_publication_matches": True}}
            validate_plan_evidence(doc, pinned_files=auth["pinned"], code=code)   # self-check: same contract
            data = mi.dump_json(doc)
            if len(data) > PLAN_CAP:
                raise mi.Blocked(f"seed plan is {len(data)} bytes, above {PLAN_CAP}")
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            published = media.publish(ledger, PLAN_NAME, data)
            _check(before, current(), "post-publication")
            if media.sealed_sha256(out_dir / PLAN_NAME, PLAN_CAP) != published["sha256"] \
                    or sorted(e.name for e in out_dir.iterdir()) != [PLAN_NAME]:
                raise mi.InputsError("post-publication: plan output changed")
            if counter.reads("annotation") or counter.reads("camera") or opened:
                raise mi.GuardError("annotation/camera/payload bytes were opened by the metadata plan")
            total = counter.snapshot()
    except BaseException as exc:
        if state["dir"]:
            _failure(ledger, "plan-seed", exc)
        raise
    return {"plan": mi.repo_rel(mctx, out_dir / PLAN_NAME), "plan_sha256": published["sha256"], "label": p["label"],
            "annotation_opens": 0, "audited_opens_total": total, "exit": 0}


PLAN_SNAPSHOT_NOTE = "audited opens after the pre-publication closure pass"
PLAN_ANNOTATION_CLOSURE = "lstat identity only (never opened here)"
PLAN_INPUT_LABELS = ("prerequisites", "policy", "allowlist", "model_receipt", "video_receipt", "profile", "camera_k",
                     "association", "scout_plan", "scout_receipt", "anchor:acquisition_receipt", "anchor:integrity_record",
                     "anchor:resources_index")
# authentication read + pre-publication closure read for every input; the resources index is additionally read
# by the accepted profile authentication, the media row derivation and this module's descriptor derivation.
EXPECTED_PLAN_READS = {f"{label}:read": (4 if label == "anchor:resources_index" else 2) for label in PLAN_INPUT_LABELS}


def _same(value: Any, expected: Any, what: str) -> None:
    """Exact equality with JSON types (true vs 1, 1 vs 1.0 differ)."""
    if mi.dump_json(value) != mi.dump_json(expected):
        raise mi.InputsError(f"seed plan {what} contradicts the fixed contract / authenticated evidence")


def validate_plan_evidence(doc: Mapping, *, pinned_files: Mapping, code: Mapping) -> None:
    """The plan's NON-core claims, checked strictly BEFORE any annotation grant: exact audited read counts
    (no camera/annotation/payload/stage-output labels), zero opens, the fixed planned closure counts, the
    fixed budget allowance within valid bounds, the exact closure record against the authenticated pinned
    files, and type-strict source closure/runtime equality."""
    reads = doc["reads"]
    if not isinstance(reads, Mapping) or set(reads) != {"snapshot", "audited_opens", "annotation_opens", "camera_opens",
                                                         "payload_opens", "post_publication_planned"}:
        raise mi.InputsError("seed plan reads record has missing or unknown fields")
    _same(reads["snapshot"], PLAN_SNAPSHOT_NOTE, "reads.snapshot")
    for key in ("annotation_opens", "camera_opens", "payload_opens"):
        _same(reads[key], 0, f"reads.{key}")
    _same(reads["audited_opens"], EXPECTED_PLAN_READS, "reads.audited_opens")
    _same(reads["post_publication_planned"], {"input_hashes": len(PLAN_INPUT_LABELS), "output_hashes": 1},
          "reads.post_publication_planned")
    media._check_budget(doc["budget"], "seed plan",
                        planned_bytes=media.TRANSIENT_FACTOR * PLAN_CAP + mi.DIR_ENTRY_RESERVE)
    closure = doc["closure"]
    if not isinstance(closure, Mapping) or set(closure) != {"before_files", "annotation", "pre_publication_matches"}:
        raise mi.InputsError("seed plan closure record has missing or unknown fields")
    _same(closure["before_files"], pinned_files, "closure.before_files")
    _same(closure["annotation"], PLAN_ANNOTATION_CLOSURE, "closure.annotation")
    _same(closure["pre_publication_matches"], True, "closure.pre_publication_matches")
    _same(doc["sources"], code["sources"], "source closure")
    _same(doc["runtime"], code["runtime"], "runtime")


def _validate_plan(doc: Any, p: Mapping) -> dict:
    if not isinstance(doc, dict) or set(doc) != PLAN_KEYS:
        raise mi.InputsError("seed plan has missing or unknown fields")
    if doc["schema"] != SCHEMA_PLAN or doc["label"] != p["label"] or doc["pins_kind"] != p["kind"]:
        raise mi.InputsError("seed plan schema/label differ (a TEST_ONLY plan cannot pass as production)")
    inputs = doc["inputs"]
    try:
        paths = {"prerequisites": inputs["prerequisites"]["path"], "policy": doc["policy"]["path"]}
    except (KeyError, TypeError):
        raise mi.InputsError("seed plan inputs malformed") from None
    return paths


def extract_unique_seed(ctx: SeedContext, *, plan: Any, expected_plan_sha256: str, out: Any) -> dict:
    _assert_helper_assumptions()
    p = pins(ctx)
    expected = mi.check_sha(expected_plan_sha256, "--expected-plan-sha256")
    mctx = ctx.mi()
    out_dir = _out_dir(mctx, out)
    ledger = SeedLedger(mctx, out_dir, "extract-unique-seed")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened, media._counting() as counter:     # guard BEFORE metadata auth
            counter.watch_prefix(out_dir, "stage_output")
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(SEED_CAP, True), (BINDING_CAP, True)])      # prospective, before any read
            budget = mi.workspace_budget(mctx, media.TRANSIENT_FACTOR * (SEED_CAP + BINDING_CAP) + mi.DIR_ENTRY_RESERVE)
            plan_path = _locate(mctx, plan, PLAN_NAME, "seed plan", inside=True)
            if plan_path.parent.parent != Path(os.path.abspath(mctx.scope / SEED_SUBTREE)):
                raise mi.InputsError("seed plan does not lie in the seed subtree")
            counter.watch(plan_path, "plan")
            doc = mi.strict_json(media._read_sealed(plan_path, PLAN_CAP, expected, "seed plan"), "seed plan")
            paths = _validate_plan(doc, p)
            _same(doc["sources"], code["sources"], "source closure")          # type-strict, before any metadata auth
            _same(doc["runtime"], code["runtime"], "runtime")
            prereq_path = _locate(mctx, paths["prerequisites"], media.PREREQ_NAME, "prerequisites receipt", inside=True)
            policy_path = _locate(mctx, paths["policy"], None, "seed policy", inside=False)
            auth = _authenticate(mctx, p, counter, prereq_path, policy_path)          # full re-derivation under pins
            if mi.dump_json(auth["core"]) != mi.dump_json({k: doc[k] for k in PLAN_CORE_KEYS}):
                raise mi.InputsError("accepted seed plan differs from a fresh re-derivation of its authenticated inputs")
            validate_plan_evidence(doc, pinned_files=auth["pinned"], code=code)  # non-core claims, BEFORE the grant
            planned = auth["core"]["input_budget"]
            inputs = mi.input_budget(planned["existing_input_bytes_parent_declared"], planned["candidate_payload_bytes"], 0)
            if opened:
                raise mi.GuardError("payload bytes were opened before the annotation grant")
            gt_path, descriptor = auth["gt_path"], auth["descriptor"]
            mi.add_read_exception(gt_path)                     # exactly ONE read-only file, after all prerequisites
            passes = {"full_prehash": annotation_pass(gt_path, descriptor)}
            scanner = opening.TopLevelFrameExtractor(str(FRAME_INDEX))
            passes["lexical_scan_with_full_hash"] = annotation_pass(gt_path, descriptor, scanner=scanner)
            frame = decode_frame_value(scanner.finish())
            del scanner
            selected = select_unique_seed(frame)
            del frame
            ordinal = selected["ordinal"]
            seed_label = (f"controlled_init:ground_truth:HB:{mi.DATASET_REVISION}:scene{SCENE}:obj{OBJECT_ID:06d}:"
                          f"instance{ordinal}:{FRAME_ID}")
            before = {"files": dict(sorted({**auth["pinned"], "plan": expected}.items())), **code,
                      "annotation": {k: descriptor[k] for k in ("expanded_bytes", "expanded_sha256", "crc32")}}
            files = {**auth["files"], "plan": (plan_path, PLAN_CAP)}

            def current() -> dict:
                try:
                    sealed = annotation_pass(gt_path, descriptor)
                    annotation = {"expanded_bytes": sealed["bytes"], "expanded_sha256": sealed["sha256"],
                                  "crc32": sealed["crc32"]}
                except (mi.InputsError, OSError) as exc:
                    annotation = {"refused": type(exc).__name__}
                return {**_state(files), "annotation": annotation}

            _check(before, current(), "pre-publication")                        # annotation pass 3
            seed = {"schema": SCHEMA_SEED, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
                    "seed_label": seed_label, "scene": SCENE, "object_id": OBJECT_ID, "frame_id": FRAME_ID,
                    "frame_index": FRAME_INDEX, "instance_ordinal": ordinal,
                    "seed_pose_m": selected["pose"],
                    "pose_convention": "BOP model-to-camera (cam_R_m2c, cam_t_m2c); not inverted or recentered",
                    "translation_units": "metres (BOP millimetres * 0.001 exactly once; rotation unchanged)",
                    "applies_once_to_frame_id": APPLIES_ONCE_TO, "reuse_on_recovery": False,
                    "other_pose_values_converted": 0, "flags": dict(SEED_FLAGS), "disclosure": DISCLOSURE,
                    "annotation_access": dict(ANNOTATION_ACCESS),
                    "principal_point_origin": None, "K_operation": None, "trust": dict(TRUST),
                    "plan": {"path": mi.repo_rel(mctx, plan_path), "sha256": expected}, "unresolved": list(UNRESOLVED)}
            seed_data = mi.dump_json(seed)
            if len(seed_data) > SEED_CAP:
                raise mi.Blocked("controlled seed exceeds its cap")
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            seed_meta = media.publish(ledger, SEED_NAME, seed_data)
            binding = {
                "schema": SCHEMA_BINDING, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
                "status": "controlled_seed_extracted", "seed_label": seed_label,
                "seed": {"path": mi.repo_rel(mctx, out_dir / SEED_NAME), "sha256": seed_meta["sha256"],
                         "bytes": seed_meta["bytes"]},
                "plan": {"path": mi.repo_rel(mctx, plan_path), "sha256": expected},
                "association": {**doc["inputs"]["association"], "instance_ordinal": None, "unchanged": True},
                "derived_instance_ordinal": ordinal,
                "ordinal_derivation": ("source-order position of the UNIQUE obj_id == 4 record in frame 9 (derived "
                                       "after uniqueness; not a match-count index)"),
                "frame9": {"records": selected["records"], "obj_id_4_matches": 1,
                           "selected_values_converted": selected["selected_values_converted"],
                           "other_pose_values_converted": 0,
                           "obj_id_lexemes_converted_to_int": selected["obj_id_lexemes_converted_to_int"],
                           "other_frames_values_materialized": False},
                "annotation_source": doc["annotation_source"],
                "annotation_passes": {"completed": {**passes, "pre_publication_closure": "matched"},
                                      "planned_after_this_record": ["final_closure_after_both_outputs"]},
                "flags": dict(SEED_FLAGS), "disclosure": DISCLOSURE, "annotation_access": dict(ANNOTATION_ACCESS),
                "trust": dict(TRUST),
                "reads": {"snapshot": "audited opens after the pre-publication closure, before this record",
                          "audited_opens": counter.snapshot(), "native_gpu_used": False},
                "budget": budget, "input_budget": inputs, "sources": code["sources"], "runtime": code["runtime"],
                "preparation_scope": mi.repo_rel(mctx, mctx.scope), "unresolved": list(UNRESOLVED),
                "closure": {"before_files": before["files"], "annotation": before["annotation"],
                            "pre_publication_matches": True}}
            binding_data = mi.dump_json(binding)
            if len(binding_data) > BINDING_CAP:
                raise mi.Blocked("seed binding exceeds its cap")
            binding_meta = media.publish(ledger, BINDING_NAME, binding_data)    # receipt last
            _check(before, current(), "final closure after both outputs")       # annotation pass 4
            for name, meta, cap in ((SEED_NAME, seed_meta, SEED_CAP), (BINDING_NAME, binding_meta, BINDING_CAP)):
                if media.sealed_sha256(out_dir / name, cap) != meta["sha256"]:
                    raise mi.InputsError(f"final closure: {name} bytes changed")
            if sorted(e.name for e in out_dir.iterdir()) != sorted([SEED_NAME, BINDING_NAME]):
                raise mi.InputsError("final closure: output names changed")
            if counter.reads("annotation") != 4 or counter.reads("camera") or opened:
                raise mi.GuardError("annotation/camera/payload read accounting differs from the four planned passes")
            total = counter.snapshot()
    except BaseException as exc:
        if state["dir"]:
            _failure(ledger, "extract-unique-seed", exc)
        raise
    return {"seed": mi.repo_rel(mctx, out_dir / SEED_NAME), "seed_sha256": seed_meta["sha256"],
            "binding": mi.repo_rel(mctx, out_dir / BINDING_NAME), "binding_sha256": binding_meta["sha256"],
            "seed_label": seed_label, "instance_ordinal": ordinal, "label": p["label"],
            "annotation_opens": 4, "audited_opens_total": total, "exit": 0}


# --------------------------------------------------------------------------
# CLI (production pins only; explicit non-zero exit records)


def main(argv: Sequence[str] | None = None, ctx: SeedContext | None = None) -> int:
    base = ctx or SeedContext()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("plan-seed")
    for flag in ("--prerequisites", "--expected-prerequisites-sha256", "--policy-review",
                 "--expected-policy-review-sha256", "--out"):
        s.add_argument(flag, required=True)
    s = sub.add_parser("extract-unique-seed")
    for flag in ("--plan", "--expected-plan-sha256", "--out"):
        s.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "plan-seed":
            result = plan_seed(base, prerequisites=args.prerequisites,
                               expected_prerequisites_sha256=args.expected_prerequisites_sha256,
                               policy_review=args.policy_review,
                               expected_policy_review_sha256=args.expected_policy_review_sha256, out=args.out)
        else:
            result = extract_unique_seed(base, plan=args.plan, expected_plan_sha256=args.expected_plan_sha256, out=args.out)
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
                         "tools.workbench_hb_scene3_seed ...)")
    raise SystemExit(main())
