"""Immutable HB scene-000003 metric PROFILE and its K-only derivative (metadata only, CPU only).

This module only builds a metadata profile and a fixed-K derivative. It does NOT prepare a model or
video, extract a seed, assemble, register or admit anything, run a network, the GPU or inference,
read GT/poses/masks/RGB pixels, or associate an object. Earlier scene-000002 receipts, contracts and
tools stay unchanged. It reuses pure helpers of ``workbench_hb_metric_inputs`` and
``workbench_hb_prefix_scout``, never mutates their constants and never calls their stages.

``build-profile``  Binds the parent-accepted scout plan + preview receipt (pinned SHA-256, verified
                   BEFORE parsing), re-verifies the three historical anchors BEFORE parsing, rebuilds
                   the scout metadata core and requires byte equality, and freezes: revision, scene
                   000003, all 120 exact RGB identities, 640x480, opening 0..9, seed 9, scoring
                   10..119, FRAME_INDEX clock (physical FPS/timestamps unknown), object id and
                   instance ordinal UNSET, CAD 5 merely a candidate (no association), principal-point
                   origin UNVERIFIED, and the index-bound camera REUSE row (stat only; never opened).
``extract-k``      Re-authenticates the accepted profile (expected SHA, structure, full rebuild from the
                   scout outputs and anchors), then hashes the whole bounded camera file BEFORE
                   parsing and streams it through a structural scanner that captures ONLY ``cam_K``
                   for source indices 0..119. Raw K is kept unchanged (no crop/rescale/origin shift).

Trust: historical local integrity only, ``upstream_authenticated: false``. Exit codes: 0 completed;
1 failure; 2 usage; 3 blocked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (pure helpers only; never mutated)
from tools import workbench_hb_prefix_scout as sc  # noqa: E402  (pure validators only; stages never called)
from tools import workbench_hb_camera_k as ck  # noqa: E402

TOOL_VERSION = "1"
SCHEMA_PROFILE = "workbench_hb_scene3_metric_profile_v1"
SCHEMA_K = "workbench_hb_scene3_camera_k_v1"
SCHEMA_FAILURE = "workbench_hb_scene3_profile_failure_v1"
SCENE = sc.SCENE
FRAME_COUNT = sc.FRAME_COUNT
OPENING, SEED_INDEX, SCORING = tuple(range(0, 10)), 9, tuple(range(10, 120))
IMAGE_SIZE = (640, 480)
CAMERA_MEMBER = sc.CALIBRATION_MEMBER
PROFILE_NAME, K_NAME, FAILURE_NAME = "scene3-profile.json", "camera-k.json", "failure.json"
INVALIDATED_PREFIX = "invalidated-"
AUTHORITATIVE_NAMES = (PROFILE_NAME, K_NAME)

# Production pins (parent-accepted scout outputs; historical camera metadata; accepted input accounting).
ACCEPTED_SCOUT = {"plan": "d76aaec4dabd33543aad06d330f520e95bfad394db84c147b47e56616e044201",
                  "preview_receipt": "a78e0c79e986df6a4ba8890550ef45ec91fac62d1c43f3f51fb0f11468b0b346"}
CAMERA_EXPANDED_SHA256 = "09f292d9305b3003a6356410adbce688548ed3c10c5b19d27c242d0ead7c809d"
INPUT_ACCOUNTED = 186_411_119              # existing + scene-3 candidate + scout cap + 12 MiB future-prep reserve
REUSE_ROOT = ".cache/prototype-sprint/hb-scene-metadata-v1"

STAGE_CAP = 128 << 10                      # per substage output directory (profile / K), temp aliases included
FAILURE_RESERVE = 16 << 10
PROFILE_CAP = 48 << 10                     # 2 x 48 KiB transient + 2 x 16 KiB failure reserve = 128 KiB
K_CAP = 32 << 10
CAMERA_CAP = ck.CAMERA_CAP
PROFILE_ALLOWANCE = 2 * PROFILE_CAP + mi.DIR_ENTRY_RESERVE
K_ALLOWANCE = 2 * K_CAP + mi.DIR_ENTRY_RESERVE

FLAGS = {"inputs_only": True, "metadata_only": True, "model_prepared": False, "video_prepared": False,
         "seed_extracted": False, "assembled": False, "registered": False, "admitted": False,
         "neural_executed": False, "gpu_used": False, "gt_read": False, "rgb_pixels_read": False,
         "camera_metadata_read": False, "poses": None, "masks": None, "quality_claim": False,
         "mobile_verified": False, "phone_performance_claim": False, "family": mi.FAMILY,
         "head_sha256": mi.FLAGS["head_sha256"]}
K_FLAGS = {**FLAGS, "camera_metadata_read": True, "fixed_k_metadata_control": True}
TRUST = {"trust_level": "historical_local_integrity", "upstream_authenticated": False,
         "basis": ("parent-accepted scout plan/preview receipt and the three historical HB anchors, each pinned by "
                   "SHA-256 and verified before parsing")}
IMAGE = {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1],
         "verification": ("HB primesense native size; PNG headers of frames 0 and 9 were verified by the accepted "
                          "scout preview; the other frames are verified only by a later preparation stage")}
FROZEN_SPLIT = {"opening": list(OPENING), "seed_index": SEED_INDEX, "scoring": list(SCORING),
                "fixed_before_viewing": True}
CLOCK = {"kind": "FRAME_INDEX", "physical_fps": None, "physical_timestamps": None,
         "note": "physical time and FPS are unknown; no ten-real-seconds claim"}
PRINCIPAL_POINT = {"origin_convention": None, "status": "UNVERIFIED_separate_review_required",
                   "note": "K is kept raw; no half-pixel/origin, crop or rescale adjustment is applied"}
CAMERA_STATUS = "PENDING_K_extraction"
CAMERA_NOTE = "scene-specific K is extracted only by extract-k after re-authenticating this profile"
ADJUSTMENTS = {"crop": None, "rescale": None, "origin_shift": None, "note": "raw cam_K preserved unchanged"}
CONTROL_NOTE = "metadata-only fixed-K control; no tracking-quality, phone or performance claim"
FRAME_COLUMNS = ["index", "member", "uncompressed_sha256", "compressed_sha256", "crc32", "expanded_bytes",
                 "compressed_bytes"]
CORE_KEYS = frozenset({"schema", "tool_version", "scene", "flags", "trust", "pins_kind", "scout", "anchors", "source",
                       "image", "frozen_split", "clock", "object", "principal_point", "frames", "rgb_totals", "camera",
                       "input_accounting"})
PROFILE_KEYS = CORE_KEYS | {"sources", "runtime", "budget", "preparation_scope", "closure"}
POST_PUBLICATION_NOTE = ("re-verified AFTER publication (inputs, sources, runtime and the current names/bytes of this "
                         "stage's outputs); a mismatch renames the owned output to 'invalidated-<name>', writes "
                         "failure.json and fails the stage")


def _object() -> dict:
    return {"object_id": None, "instance_ordinal": None,
            "cad_candidate": {"object_id": 5, "status": "candidate_only", "association": None,
                              "note": "CAD 5 is merely a candidate; no visual instance association has been made"}}


@dataclass(frozen=True)
class SyntheticPins:
    """TEST-ONLY replacement for the production pins (manufactured fixtures). The CLI never sets it and
    every output records ``pins_kind = "synthetic_test_only"``. The scout plan/receipt under test must
    have been produced with the scout's own synthetic pins of the same anchors."""
    anchors: Mapping[str, str]
    integrity_members: int
    rgb_encoded_bytes: int
    rgb_expanded_bytes: int | None
    scout_plan_sha256: str
    scout_receipt_sha256: str
    camera_expanded_sha256: str
    input_accounted: int
    reuse_root: str


@dataclass(frozen=True)
class ProfileContext:
    """Injectable roots. ``preparation`` is the fresh, exclusive scene-000003 preparation scope."""
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"
    preparation: Path | None = None
    synthetic: SyntheticPins | None = None

    def mi(self) -> mi.Context:
        if self.preparation is None:
            raise mi.UsageError("--preparation-root is required (a fresh, exclusive scene-000003 scope inside derived)")
        return mi.Context(root=self.root, workspace=self.workspace, preparation=Path(os.path.abspath(self.preparation)))


def pins(ctx: ProfileContext) -> dict:
    if ctx.synthetic is None:
        scout = sc.pins(sc.ScoutContext())
        return {"kind": "historical", "anchors": dict(mi.ANCHORS), "scout": scout,
                "scout_plan_sha256": ACCEPTED_SCOUT["plan"], "scout_receipt_sha256": ACCEPTED_SCOUT["preview_receipt"],
                "camera_expanded_sha256": CAMERA_EXPANDED_SHA256, "input_accounted": INPUT_ACCOUNTED,
                "reuse_root": REUSE_ROOT}
    s = ctx.synthetic
    shas = (s.scout_plan_sha256, s.scout_receipt_sha256, s.camera_expanded_sha256, *dict(s.anchors).values())
    if (set(s.anchors) != set(mi.ANCHORS) or not all(mi._is_sha(v) for v in shas)
            or type(s.integrity_members) is not int or type(s.rgb_encoded_bytes) is not int
            or type(s.input_accounted) is not int or s.input_accounted < 0 or not isinstance(s.reuse_root, str)):
        raise mi.UsageError("synthetic test pins are malformed")
    scout = {"kind": "synthetic_test_only", "anchors": dict(s.anchors), "integrity_members": s.integrity_members,
             "rgb_encoded_bytes": s.rgb_encoded_bytes, "rgb_expanded_bytes": s.rgb_expanded_bytes}
    return {"kind": "synthetic_test_only", "anchors": dict(s.anchors), "scout": scout,
            "scout_plan_sha256": s.scout_plan_sha256, "scout_receipt_sha256": s.scout_receipt_sha256,
            "camera_expanded_sha256": s.camera_expanded_sha256, "input_accounted": s.input_accounted,
            "reuse_root": s.reuse_root}


# --------------------------------------------------------------------------
# ledger, publication, failure (owned outputs only)


class PrepLedger:
    """Before every encoded write: the substage directory <= 128 KiB, the whole preparation scope
    <= the existing 12 MiB / 512 KiB metadata aggregates (retained failed siblings included), and the
    free-disk reserve. ``mi.publish_bytes`` keeps an owned temp AND its hard-linked final name until the
    temp is removed and both are counted, so every check uses ``2 x bytes``; normal writes keep
    ``2 x FAILURE_RESERVE`` for a failure record. ``record`` (called by the publisher right after ITS
    link) establishes ownership with the created file identity."""

    TRANSIENT_FACTOR = 2

    def __init__(self, mctx: mi.Context, out_dir: Path, stage: str):
        self.ctx, self.dir, self.stage, self.scope = mctx, out_dir, stage, mctx.scope
        self.files: dict[str, int] = {}
        self.owned: dict[str, tuple[int, int]] = {}

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        return self.check_sequence([(nbytes, metadata)], failure=failure)

    def check_sequence(self, steps: Sequence[tuple[int, bool]], *, failure: bool = False) -> dict:
        if not steps or any(type(n) is not int or n < 0 for n, _ in steps):
            raise mi.InputsError("invalid write size")
        scope, stage = mi.scope_usage(self.scope), mi.scope_usage(self.dir)
        reserve = 0 if failure else self.TRANSIENT_FACTOR * FAILURE_RESERVE
        done_total = done_meta = 0
        for nbytes, metadata in steps:
            peak = self.TRANSIENT_FACTOR * nbytes
            if stage["total"] + done_total + peak > STAGE_CAP - reserve:
                raise mi.Blocked(f"{self.stage}: substage would peak at {stage['total'] + done_total + peak} bytes "
                                 f"(temp + published name), above {STAGE_CAP - reserve}")
            if scope["total"] + done_total + peak > mi.FINAL_OUTPUTS_CAP - reserve:
                raise mi.Blocked(f"{self.stage}: preparation scope would peak above {mi.FINAL_OUTPUTS_CAP - reserve}")
            if metadata:
                if scope["metadata"] + done_meta + peak > mi.METADATA_AGGREGATE_CAP - reserve:
                    raise mi.Blocked(f"{self.stage}: scope metadata would peak above "
                                     f"{mi.METADATA_AGGREGATE_CAP - reserve}")
                done_meta += nbytes
            done_total += nbytes
        largest = max(n for n, _ in steps)
        if mi.shutil.disk_usage(self.ctx.workspace).free - self.TRANSIENT_FACTOR * largest - done_total < mi.FREE_RESERVE:
            raise mi.Blocked("the 1 GiB free-disk reserve would be violated")
        return {"scope": scope, "stage": stage}

    def record(self, name: str, nbytes: int, *, metadata: bool) -> None:
        self.files[name] = self.files.get(name, 0) + nbytes
        try:
            st = os.lstat(self.dir / name)
            self.owned[name] = (st.st_dev, st.st_ino)
        except OSError:
            self.owned[name] = (-1, -1)

    def owned_file(self, name: str) -> bool:
        if name not in self.owned:
            return False
        try:
            st = os.lstat(self.dir / name)
        except OSError:
            return False
        dev, ino = self.owned[name]
        return mi.stat.S_ISREG(st.st_mode) and ((st.st_dev, st.st_ino) == (dev, ino) or (dev, ino) == (-1, -1))

    def summary(self) -> dict:
        return {"stage": self.stage, "files": dict(self.files), "scope_usage_now": mi.scope_usage(self.scope),
                "caps": {"substage": STAGE_CAP, "scope_total": mi.FINAL_OUTPUTS_CAP,
                         "scope_metadata": mi.METADATA_AGGREGATE_CAP, "failure_reserve": FAILURE_RESERVE}}


def dump_compact(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n").encode("ascii")


def _failure_record(ledger: PrepLedger, stage: str, exc: BaseException, extra: Mapping | None = None) -> None:
    """Bounded valid-JSON failure record (<= FAILURE_RESERVE), after a FRESH global derived/free check with
    2x transient growth; if it cannot be published nothing is written and stderr gets it on one line."""
    retained = {}
    if ledger.dir.is_dir():
        for entry in sorted(ledger.dir.iterdir()):
            try:
                retained[entry.name] = os.lstat(entry).st_size
            except OSError:
                pass
    try:
        summary = ledger.summary()
    except Exception as summary_exc:  # noqa: BLE001 - never mask the original failure
        summary = {"error": f"{type(summary_exc).__name__}: {str(summary_exc)[:200]}"}
    body = {"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage, "status": "failed",
            "reason": f"{type(exc).__name__}: {str(exc)[:1500]}", "completed_receipt_published": False,
            "retained_files": retained, "ledger": summary, **(dict(extra) if extra else {})}
    data = mi.dump_json(body)
    if len(data) > FAILURE_RESERVE:
        body.update(reason=body["reason"][:300], ledger={"omitted": "failure record size bound"},
                    retained_files={"count": len(retained), "bytes": sum(retained.values()),
                                    "names": sorted(retained)[:16]})
        data = mi.dump_json(body)
    if len(data) > FAILURE_RESERVE:
        data = mi.dump_json({"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage,
                             "status": "failed", "reason": body["reason"][:300], "completed_receipt_published": False})
    try:
        if len(data) > FAILURE_RESERVE:
            raise mi.Blocked("failure record exceeds its reserve")
        mi.workspace_budget(ledger.ctx, PrepLedger.TRANSIENT_FACTOR * len(data))
        mi.publish_bytes(ledger, FAILURE_NAME, data, failure=True)
    except Exception as publish_exc:  # noqa: BLE001 - keep a terminal record, never mask the original error
        line = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        sys.stderr.write(f"failure.json NOT published ({type(publish_exc).__name__}: {str(publish_exc)[:300]}); "
                         f"the record follows on ONE stderr line as JSON:\n{line}\n")


def _fail_stage(ledger: PrepLedger, stage: str, exc: BaseException) -> None:
    """Rename an owned, still-identical authoritative output to ``invalidated-<name>`` (bytes kept); remove
    it only if the rename fails. Foreign files are never renamed or deleted."""
    records, foreign = [], []
    for name in AUTHORITATIVE_NAMES:
        src = ledger.dir / name
        if not os.path.lexists(src):
            continue
        if not ledger.owned_file(name):
            foreign.append(name)
            continue
        record: dict = {"name": name}
        try:
            record["sha256"] = mi.sha256_file(src)
        except Exception as hash_exc:  # noqa: BLE001 - invalidation must still happen
            record["sha256_error"] = f"{type(hash_exc).__name__}: {str(hash_exc)[:200]}"
        dst = ledger.dir / f"{INVALIDATED_PREFIX}{name}"
        try:
            if os.path.lexists(dst):
                raise FileExistsError(str(dst))
            os.rename(src, dst)
            record["retained_as"] = dst.name
        except OSError as rename_exc:
            record["rename_error"] = f"{type(rename_exc).__name__}: {str(rename_exc)[:200]}"
            try:
                os.unlink(src)
                record["removed"] = True
            except OSError as unlink_exc:
                record["still_present_under_authoritative_name"] = True
                record["unlink_error"] = f"{type(unlink_exc).__name__}: {str(unlink_exc)[:200]}"
        records.append(record)
    _failure_record(ledger, stage, exc, {"post_publication_failure": bool(records), "invalidated_outputs": records,
                                         "foreign_names_untouched": foreign})


def _post_publication_closure(before: Mapping, current: Mapping, out_dir: Path, outputs: Mapping[str, str]) -> None:
    if current != before:
        changed = sorted(k for k in set(before) | set(current) if before.get(k) != current.get(k))
        raise mi.InputsError(f"closure changed after publication: {changed}")
    names = sorted(entry.name for entry in out_dir.iterdir())
    if names != sorted(outputs):
        raise mi.InputsError(f"closure changed after publication: output names {names}")
    for name, sha in outputs.items():
        if mi.sha256_file(out_dir / name) != sha:
            raise mi.InputsError(f"closure changed after publication: {name} bytes")


def _sources() -> dict:
    extra = {Path(m.__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(m.__file__))
             for m in (sc, ck)}
    return dict(sorted({**mi.source_closure(), **extra,
                        Path(__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(__file__))}.items()))


def _closure(files: Mapping[str, Path], camera: tuple[Path, Mapping] | None = None, reads: list | None = None) -> dict:
    """Inputs/sources/runtime. The camera is NEVER hashed with the generic helper: it gets the bounded
    expected-size / 1 MiB-cap SHA-256 + CRC-32 pass (``ck.hash_bounded`` refuses a size change from
    ``lstat`` before reading), and a refusal is reported as a closure mismatch."""
    state = {"files": {k: mi.sha256_file(v) for k, v in sorted(files.items())}, "sources": _sources(),
             "runtime": mi.runtime_identity()}
    if camera is not None:
        path, row = camera
        if reads is not None:
            reads.append("closure")
        try:
            sealed = ck.hash_bounded(path, expected_bytes=row["expanded_bytes"], cap=CAMERA_CAP)
            state["camera"] = {"bytes": sealed["bytes"], "sha256": sealed["sha256"], "crc32": sealed["crc32"]}
        except (ck.CameraError, OSError) as exc:
            state["camera"] = {"refused": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return state


def _camera_seal(row: Mapping) -> dict:
    return {"bytes": row["expanded_bytes"], "sha256": row["expanded_sha256"], "crc32": row["crc32"]}


# --------------------------------------------------------------------------
# input authentication (pure checks + guarded reads)


def _locate(mctx: mi.Context, value: Any, name: str, what: str, *, inside: bool) -> Path:
    """Location only (lstat; never opened): regular non-link file with the exact producer name."""
    path = mi.contained_input(mctx, value, what)
    if path.name != name:
        raise mi.InputsError(f"{what} must be a {name}; refused {path.name!r} unread")
    if inside:
        mi.inside_scope(mctx, path, what)
    else:
        mi.outside_scope_input(mctx, path, what)
    return path


def _exclusive_scope(mctx: mi.Context, scout_scope: Path) -> None:
    scope, derived = mctx.scope, Path(os.path.abspath(mctx.derived))
    if scope == derived or derived not in scope.parents:
        raise mi.UsageError("the preparation scope must be its own directory strictly inside derived")
    if scope == scout_scope or scout_scope in scope.parents or scope in scout_scope.parents:
        raise mi.UsageError("the scene-000003 preparation scope must be separate from the scout scope")


def validate_scout_receipt(doc: Any, plan: Mapping, plan_rel: str, p: Mapping) -> dict:
    if not isinstance(doc, dict):
        raise mi.InputsError("scout preview receipt must be a JSON object")
    expect = {"schema": sc.SCHEMA_PREVIEW, "tool_version": sc.TOOL_VERSION, "status": "preview_published_candidate_only",
              "scene": SCENE, "flags": sc.FLAGS, "trust": sc.TRUST, "pins": p["kind"],
              "plan": {"path": plan_rel, "sha256": p["scout_plan_sha256"]},
              "authorized_indices": list(sc.PREVIEW_INDICES), "payload_opens": len(sc.PREVIEW_INDICES),
              "calibration_status": sc.CALIBRATION_STATUS, "clock": sc.CLOCK, "scout_scope": plan["scout_scope"]}
    for key, value in expect.items():
        if doc.get(key) != value:
            raise mi.InputsError(f"scout preview receipt field {key!r} differs from the accepted plan/pins")
    frames = doc.get("frames")
    if not isinstance(frames, list) or len(frames) != len(sc.PREVIEW_INDICES):
        raise mi.InputsError("scout preview receipt must record exactly frames 0 and 9")
    for record, index in zip(frames, sc.PREVIEW_INDICES):
        row = plan["rgb"][index]
        if (not isinstance(record, Mapping) or record.get("index") != index or record.get("frame_id") != row["frame_id"]
                or record.get("member") != row["member"] or record.get("png_sha256") != row["uncompressed_sha256"]
                or record.get("compressed_sha256") != row["compressed_sha256"] or record.get("crc32") != row["crc32"]
                or record.get("native_size") != list(IMAGE_SIZE)):
            raise mi.InputsError(f"scout preview frame {index} differs from the accepted plan row")
    return doc


def authenticate_inputs(mctx: mi.Context, p: Mapping, plan_path: Path, receipt_path: Path) -> dict:
    """Inside an ACTIVE guard with no read exceptions: verify + parse the scout outputs, then add the three
    anchors as exact READ-ONLY exceptions, verify them BEFORE parsing, rebuild the scout core and validate
    the camera reuse row (stat only)."""
    sp = p["scout"]
    raw = mi.read_bounded(plan_path, sc.PLAN_CAP, "scout plan")
    if mi.sha256_bytes(raw) != p["scout_plan_sha256"]:
        raise mi.InputsError("scout plan: SHA-256 differs from the accepted pin")
    plan = sc.validate_plan_doc(mi.strict_json(raw, "scout plan"), sp)
    del raw
    plan_rel = mi.repo_rel(mctx, plan_path)
    scout_scope = Path(os.path.abspath(mctx.root / plan["scout_scope"]))
    if scout_scope not in plan_path.parents or scout_scope not in receipt_path.parents:
        raise mi.InputsError("scout plan/receipt do not lie in the plan's scout scope")
    _exclusive_scope(mctx, scout_scope)
    budget = plan["input_budget"]
    if not isinstance(budget, Mapping) or type(budget.get("total")) is not int or budget["total"] != p["input_accounted"]:
        raise mi.InputsError("scout plan input accounting differs from the accepted total")
    raw = mi.read_bounded(receipt_path, sc.RECEIPT_CAP, "scout preview receipt")
    if mi.sha256_bytes(raw) != p["scout_receipt_sha256"]:
        raise mi.InputsError("scout preview receipt: SHA-256 differs from the accepted pin")
    receipt = validate_scout_receipt(mi.strict_json(raw, "scout preview receipt"), plan, plan_rel, p)
    del raw
    paths = {k: mi.contained_input(mctx, a["path"], k, allow_denied=True) for k, a in plan["anchors"].items()}
    for key, path in paths.items():
        mi.outside_scope_input(mctx, path, key)
        mi.add_read_exception(path)                                   # exact READ-ONLY anchor
    mi.allow_payloads(paths["resources_index"].parent / "payloads", ())  # EMPTY payload allowlist
    raws = sc._read_anchors(paths, sp)                                # SHA-256 BEFORE parsing
    core = sc._core_from_anchors(mctx, paths, raws, sp)
    if mi.dump_json(core) != mi.dump_json({k: plan[k] for k in sc.CORE_KEYS}):
        raise mi.InputsError("accepted scout plan differs from a fresh rebuild of the historical anchors")
    index = mi.strict_json(raws["resources_index"], "resources index")
    del raws
    camera_path = ck.canonical_reuse_path(mctx.root, p["reuse_root"], CAMERA_MEMBER)
    row = ck.validate_camera_row(index["resources"].get(CAMERA_MEMBER), member=CAMERA_MEMBER, archive=mi.RGB_ARCHIVE,
                                 revision=mi.DATASET_REVISION, expected_sha256=p["camera_expanded_sha256"],
                                 max_bytes=CAMERA_CAP, root=mctx.root, canonical=camera_path)
    del index
    listed = plan["calibration"].get("uncompressed_sha256")
    if listed is not None and listed != row["expanded_sha256"]:
        raise mi.InputsError("camera row SHA-256 differs from the scout plan's calibration listing")
    mi.outside_scope_input(mctx, camera_path, "camera metadata")
    mi._no_links(Path(os.path.abspath(mctx.root)), camera_path)
    try:
        st = os.lstat(camera_path)                                    # stat only; never opened here
    except FileNotFoundError:
        raise mi.Blocked(f"{CAMERA_MEMBER}: reused camera metadata is missing") from None
    if not mi.stat.S_ISREG(st.st_mode) or st.st_size != row["expanded_bytes"]:
        raise mi.InputsError(f"{CAMERA_MEMBER}: reused camera metadata is not a regular file of the index size")
    return {"plan": plan, "plan_path": plan_path, "plan_rel": plan_rel, "receipt": receipt,
            "receipt_path": receipt_path, "anchor_paths": paths, "camera_path": camera_path, "camera_row": row,
            "camera_stat_bytes": st.st_size}


def build_profile_core(mctx: mi.Context, p: Mapping, auth: Mapping) -> dict:
    plan = auth["plan"]
    rows = [[r["index"], r["member"], r["uncompressed_sha256"], r["compressed_sha256"], r["crc32"], r["expanded_bytes"],
             r["compressed_bytes"]] for r in plan["rgb"]]
    return {"schema": SCHEMA_PROFILE, "tool_version": TOOL_VERSION, "scene": SCENE, "flags": dict(FLAGS),
            "trust": dict(TRUST), "pins_kind": p["kind"],
            "scout": {"plan": {"path": auth["plan_rel"], "sha256": p["scout_plan_sha256"]},
                      "preview_receipt": {"path": mi.repo_rel(mctx, auth["receipt_path"]),
                                          "sha256": p["scout_receipt_sha256"]},
                      "scout_scope": plan["scout_scope"], "scout_plan_sources": plan["sources"],
                      "preview_frames": [f["frame_id"] for f in auth["receipt"]["frames"]]},
            "anchors": plan["anchors"],
            "source": {"corpus": mi.CORPUS, "dataset_revision": mi.DATASET_REVISION, "scene": SCENE,
                       "rgb_archive": mi.RGB_ARCHIVE, "selection_digest": plan["source"]["selection_digest"],
                       "payload_root": plan["source"]["payload_root"]},
            "image": dict(IMAGE), "frozen_split": dict(FROZEN_SPLIT), "clock": dict(CLOCK), "object": _object(),
            "principal_point": dict(PRINCIPAL_POINT), "frames": {"columns": list(FRAME_COLUMNS), "rows": rows},
            "rgb_totals": plan["rgb_totals"],
            "camera": {"member": CAMERA_MEMBER, "row": auth["camera_row"],
                       "reuse_path": mi.repo_rel(mctx, auth["camera_path"]), "stat_bytes": auth["camera_stat_bytes"],
                       "status": CAMERA_STATUS, "K": None, "note": CAMERA_NOTE},
            "input_accounting": {**mi.input_budget(p["input_accounted"], CAMERA_CAP, 0),
                                 "note": ("accepted scout total (already including the 12 MiB future preparation "
                                          "reserve) + a conservative 1 MiB camera input")}}


def validate_profile_doc(doc: Any, p: Mapping) -> dict:
    """Structural re-validation of an accepted profile (extract-k additionally rebuilds the core)."""
    if not isinstance(doc, dict) or set(doc) != PROFILE_KEYS:
        raise mi.InputsError("profile has missing or unknown fields")
    fixed = {"schema": SCHEMA_PROFILE, "tool_version": TOOL_VERSION, "scene": SCENE, "flags": FLAGS, "trust": TRUST,
             "image": IMAGE, "frozen_split": FROZEN_SPLIT, "clock": CLOCK, "object": _object(),
             "principal_point": PRINCIPAL_POINT, "pins_kind": p["kind"]}
    for key, value in fixed.items():
        if doc[key] != value:
            raise mi.InputsError(f"profile field {key!r} differs from the fixed contract")
    scout = doc["scout"]
    if (not isinstance(scout, Mapping) or not isinstance(scout.get("plan"), Mapping)
            or not isinstance(scout.get("preview_receipt"), Mapping)
            or scout["plan"].get("sha256") != p["scout_plan_sha256"]
            or scout["preview_receipt"].get("sha256") != p["scout_receipt_sha256"]):
        raise mi.InputsError("profile does not bind the accepted scout outputs")
    camera = doc["camera"]
    if (not isinstance(camera, Mapping) or camera.get("member") != CAMERA_MEMBER or camera.get("status") != CAMERA_STATUS
            or camera.get("K") is not None or not isinstance(camera.get("row"), Mapping)
            or camera["row"].get("expanded_sha256") != p["camera_expanded_sha256"]):
        raise mi.InputsError("profile camera block must stay PENDING and bind the pinned camera metadata")
    frames = doc["frames"]
    if not isinstance(frames, Mapping) or frames.get("columns") != FRAME_COLUMNS or not isinstance(frames.get("rows"), list) \
            or len(frames["rows"]) != FRAME_COUNT:
        raise mi.InputsError("profile must hold exactly 120 RGB identities")
    for i, row in enumerate(frames["rows"]):
        if (not isinstance(row, list) or len(row) != len(FRAME_COLUMNS) or type(row[0]) is not int or row[0] != i
                or row[1] != f"{sc.RGB_PREFIX}{i:06d}.png" or not mi._is_sha(row[2]) or not mi._is_sha(row[3])
                or not all(mi._is_count(v) for v in row[4:])):
            raise mi.InputsError(f"profile RGB identity {i} is malformed")
    accounting = doc["input_accounting"]
    if not isinstance(accounting, Mapping) or accounting.get("total") != p["input_accounted"] + CAMERA_CAP:
        raise mi.InputsError("profile input accounting differs from the accepted total")
    if not isinstance(doc["preparation_scope"], str):
        raise mi.InputsError("profile preparation scope malformed")
    return doc


def _expect_hashes(given: Any, pinned: Mapping[str, str], what: str) -> None:
    if not isinstance(given, Mapping) or set(given) != set(pinned):
        raise mi.UsageError(f"{what} hashes must name exactly {sorted(pinned)}")
    for key, value in given.items():
        mi.check_sha(value, f"{what} {key} SHA-256")
    if dict(given) != dict(pinned):
        raise mi.Blocked(f"{what} hashes are not the accepted pins; no trust can be derived")


# --------------------------------------------------------------------------
# stages


def build_profile(ctx: ProfileContext, *, scout_plan: str | os.PathLike, scout_receipt: str | os.PathLike,
                  accepted_scout_hashes: Mapping[str, str], historical_anchor_hashes: Mapping[str, str],
                  out: str | os.PathLike) -> dict:
    p = pins(ctx)
    _expect_hashes(accepted_scout_hashes, {"plan": p["scout_plan_sha256"], "preview_receipt": p["scout_receipt_sha256"]},
                   "accepted scout")
    _expect_hashes(historical_anchor_hashes, p["anchors"], "historical anchor")
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    plan_path = _locate(mctx, scout_plan, sc.PLAN_NAME, "scout plan", inside=False)
    receipt_path = _locate(mctx, scout_receipt, sc.RECEIPT_NAME, "scout preview receipt", inside=False)
    ledger = PrepLedger(mctx, out_dir, "build-profile")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened:                        # EMPTY allowlist, no read exceptions yet
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(PROFILE_CAP, True)])              # prospective, before any source read
            budget = mi.workspace_budget(mctx, PROFILE_ALLOWANCE)
            mi.input_budget(p["input_accounted"], CAMERA_CAP, 0)
            plan_path = _locate(mctx, scout_plan, sc.PLAN_NAME, "scout plan", inside=False)
            auth = authenticate_inputs(mctx, p, plan_path, receipt_path)
            core = build_profile_core(mctx, p, auth)
            if opened:
                raise mi.GuardError("the profile stage opened payload bytes")
            files = {"scout_plan": plan_path, "scout_receipt": receipt_path,
                     **{f"anchor:{k}": v for k, v in auth["anchor_paths"].items()}}
            before = {"files": dict(sorted({"scout_plan": p["scout_plan_sha256"], "scout_receipt": p["scout_receipt_sha256"],
                                            **{f"anchor:{k}": v for k, v in p["anchors"].items()}}.items())), **code}
            doc = {**core, "sources": code["sources"], "runtime": code["runtime"], "budget": budget,
                   "preparation_scope": mi.repo_rel(mctx, mctx.scope),
                   "closure": {"before": "scout outputs + anchors + sources + runtime at stage start",
                               "pre_publication_matches": True, "post_publication": POST_PUBLICATION_NOTE}}
            validate_profile_doc(doc, p)
            data = dump_compact(doc)
            if len(data) > PROFILE_CAP:
                raise mi.Blocked(f"profile is {len(data)} bytes, above {PROFILE_CAP}")
            if _closure(files) != before:
                raise mi.InputsError("scout outputs/anchors/sources/runtime changed before publication")
            ledger.check(len(data), metadata=True)
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            published = mi.publish_bytes(ledger, PROFILE_NAME, data)    # ownership recorded by ledger.record
            _post_publication_closure(before, _closure(files), out_dir, {PROFILE_NAME: published["sha256"]})
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "build-profile", exc)
        raise
    return {"profile": mi.repo_rel(mctx, out_dir / PROFILE_NAME), "profile_sha256": published["sha256"],
            "frames": FRAME_COUNT, "camera_status": CAMERA_STATUS, "pins": p["kind"], "exit": 0}


def extract_camera_k(ctx: ProfileContext, *, profile: str | os.PathLike, expected_profile_sha256: str,
                     out: str | os.PathLike) -> dict:
    p = pins(ctx)
    expected = mi.check_sha(expected_profile_sha256, "--expected-profile-sha256")
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    _locate(mctx, profile, PROFILE_NAME, "profile", inside=True)          # location only; nothing read yet
    requested = range(FRAME_COUNT)
    ledger = PrepLedger(mctx, out_dir, "extract-k")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened:                        # EMPTY allowlist, no read exceptions yet
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(K_CAP, True)])                    # prospective, before any source read
            budget = mi.workspace_budget(mctx, K_ALLOWANCE)
            inputs = mi.input_budget(p["input_accounted"], CAMERA_CAP, 0)
            profile_path = _locate(mctx, profile, PROFILE_NAME, "profile", inside=True)
            raw = mi.read_bounded(profile_path, PROFILE_CAP, "profile")
            if mi.sha256_bytes(raw) != expected:
                raise mi.InputsError("profile: SHA-256 differs from the parent-accepted value")
            doc = validate_profile_doc(mi.strict_json(raw, "profile"), p)
            del raw
            if doc["preparation_scope"] != mi.repo_rel(mctx, mctx.scope):
                raise mi.InputsError("profile was produced for a different preparation scope")
            if doc["sources"] != code["sources"] or doc["runtime"] != code["runtime"]:
                raise mi.InputsError("source closure/runtime differ from the accepted profile")
            plan_path = _locate(mctx, doc["scout"]["plan"]["path"], sc.PLAN_NAME, "scout plan", inside=False)
            receipt_path = _locate(mctx, doc["scout"]["preview_receipt"]["path"], sc.RECEIPT_NAME,
                                   "scout preview receipt", inside=False)
            auth = authenticate_inputs(mctx, p, plan_path, receipt_path)
            if dump_compact(build_profile_core(mctx, p, auth)) != dump_compact({k: doc[k] for k in CORE_KEYS}):
                raise mi.InputsError("accepted profile differs from a fresh re-authentication of its inputs")
            if opened:
                raise mi.GuardError("payload bytes were opened before the camera grant")
            camera_path, row = auth["camera_path"], auth["camera_row"]
            mi.add_read_exception(camera_path)                        # exact READ-ONLY camera member
            camera_reads: list[str] = []
            camera_reads.append("hash_pass")
            first = ck.hash_bounded(camera_path, expected_bytes=row["expanded_bytes"], cap=CAMERA_CAP)
            if first["sha256"] != row["expanded_sha256"] or first["crc32"] != row["crc32"]:
                raise mi.InputsError(f"{CAMERA_MEMBER}: SHA-256/CRC-32 differs from the index-bound row")
            camera_reads.append("scan_pass")
            scan = ck.scan_camera_file(camera_path, expected_bytes=row["expanded_bytes"],
                                       expected_sha256=row["expanded_sha256"], requested=requested)
            k = ck.validate_k_window(scan["captured"], requested, IMAGE_SIZE)
            del scan["captured"]
            if opened:
                raise mi.GuardError("the K stage opened payload bytes")
            files = {"profile": profile_path, "scout_plan": plan_path, "scout_receipt": receipt_path,
                     **{f"anchor:{key}": v for key, v in auth["anchor_paths"].items()}}
            camera = (camera_path, row)
            before = {"files": dict(sorted({"profile": expected, "scout_plan": p["scout_plan_sha256"],
                                            "scout_receipt": p["scout_receipt_sha256"],
                                            **{f"anchor:{key}": v for key, v in p["anchors"].items()}}.items())),
                      **code, "camera": _camera_seal(row)}
            if _closure(files, camera, camera_reads) != before:
                raise mi.InputsError("profile/scout/anchors/camera/sources/runtime changed before publication")
            reads = {"extraction_passes": {"hash_pass": camera_reads.count("hash_pass"),
                                           "scan_pass": camera_reads.count("scan_pass")},
                     "closure_passes": {"pre_publication_completed": camera_reads.count("closure"),
                                        "post_publication_planned": 1},
                     "total_camera_opens_on_success": len(camera_reads) + 1,
                     "bounded_by": {"expected_bytes": row["expanded_bytes"], "cap": CAMERA_CAP},
                     "note": ("every pass is bounded by the index size and the 1 MiB cap; the post-publication "
                              "closure pass runs AFTER this record is published, and a mismatch invalidates it")}
            result = {"schema": SCHEMA_K, "tool_version": TOOL_VERSION, "scene": SCENE, "flags": dict(K_FLAGS),
                      "trust": dict(TRUST), "pins_kind": p["kind"],
                      "profile": {"path": mi.repo_rel(mctx, profile_path), "sha256": expected},
                      "camera_source": {**row, "reuse_path": mi.repo_rel(mctx, camera_path), "reads": reads},
                      "K": k, "image": dict(IMAGE), "adjustments": dict(ADJUSTMENTS),
                      "principal_point": dict(PRINCIPAL_POINT), "clock": dict(CLOCK), "object": _object(),
                      "scanner": {"stats": scan["stats"], "requested_source_indices": {"first": 0, "last": FRAME_COUNT - 1},
                                  "captured_only": "cam_K of requested frames; all other values validated and skipped "
                                                   "without numeric conversion"},
                      "control": CONTROL_NOTE, "sources": code["sources"], "runtime": code["runtime"],
                      "budget": budget, "input_budget": inputs, "preparation_scope": mi.repo_rel(mctx, mctx.scope),
                      "closure": {"before": "profile + scout outputs + anchors + camera + sources + runtime",
                                  "pre_publication_matches": True, "post_publication": POST_PUBLICATION_NOTE}}
            data = mi.dump_json(result)
            if len(data) > K_CAP:
                raise mi.Blocked(f"camera-K record is {len(data)} bytes, above {K_CAP}")
            ledger.check(len(data), metadata=True)
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            published = mi.publish_bytes(ledger, K_NAME, data)          # ownership recorded by ledger.record
            _post_publication_closure(before, _closure(files, camera, camera_reads), out_dir,
                                      {K_NAME: published["sha256"]})
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "extract-k", exc)
        raise
    return {"camera_k": mi.repo_rel(mctx, out_dir / K_NAME), "camera_k_sha256": published["sha256"],
            "camera_opens": {"extraction_passes": 2, "closure_passes": camera_reads.count("closure"),
                             "total": len(camera_reads)},
            "K": k["values"], "frames": FRAME_COUNT, "pins": p["kind"], "exit": 0}


# --------------------------------------------------------------------------
# CLI (never sets synthetic pins)


def main(argv: Sequence[str] | None = None, ctx: ProfileContext | None = None) -> int:
    base = ctx or ProfileContext()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("build-profile")
    s.add_argument("--scout-plan", required=True)
    s.add_argument("--expected-scout-plan-sha256", required=True)
    s.add_argument("--scout-receipt", required=True)
    s.add_argument("--expected-scout-receipt-sha256", required=True)
    s.add_argument("--expected-receipt-sha256", required=True)
    s.add_argument("--expected-index-sha256", required=True)
    s.add_argument("--expected-integrity-sha256", required=True)
    s.add_argument("--preparation-root", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("extract-k")
    s.add_argument("--profile", required=True)
    s.add_argument("--expected-profile-sha256", required=True)
    s.add_argument("--preparation-root", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(args.preparation_root)
        root = Path(os.path.abspath(root if root.is_absolute() else base.root / root))
        run = ProfileContext(root=base.root, workspace=base.workspace, preparation=root, synthetic=base.synthetic)
        if args.stage == "build-profile":
            result = build_profile(run, scout_plan=args.scout_plan, scout_receipt=args.scout_receipt,
                                   accepted_scout_hashes={"plan": args.expected_scout_plan_sha256,
                                                          "preview_receipt": args.expected_scout_receipt_sha256},
                                   historical_anchor_hashes={"acquisition_receipt": args.expected_receipt_sha256,
                                                             "resources_index": args.expected_index_sha256,
                                                             "integrity_record": args.expected_integrity_sha256},
                                   out=args.out)
        else:
            result = extract_camera_k(run, profile=args.profile, expected_profile_sha256=args.expected_profile_sha256,
                                      out=args.out)
    except mi.UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except mi.Blocked as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 3
    except mi.InputsError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("exit", 0))


if __name__ == "__main__":
    if not sys.dont_write_bytecode:
        raise SystemExit("Use python -B (for example: python -B -S tools/prototype_bootstrap.py "
                         "tools.workbench_hb_scene3_profile ...)")
    raise SystemExit(main())
