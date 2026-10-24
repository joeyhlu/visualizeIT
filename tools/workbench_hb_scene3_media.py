"""Scene-000003 CPU MEDIA preparation (prepared media only; CPU; no GPU, inference, GT or admission).

Consumes the immutable, parent-accepted scene-000003 profile and K derivative plus a SEPARATELY
pinned parent opening CAD association review, and prepares exactly:

``media-allowlist``     METADATA ONLY (zero RGB/CAD payload opens; the raw camera file is never
                        reopened). Authenticates profile / K / association / scout / three historical
                        anchors by pinned SHA-256 BEFORE parsing, rebuilds the profile core, then
                        derives the complete RGB 0..119 rows and ONLY the reviewed CAD row from the
                        pinned historical index and compares every identity with the accepted profile,
                        scout plan and review.
``prepare-model``       Streams exactly the one reviewed CAD payload through the accepted strict PLY
                        parser (the header refuses > 200 000 ORIGINAL faces before any body
                        allocation), writes a metre GLB with the accepted writer (mm * 0.001 exactly
                        once; indices/winding/vertex colours preserved; normals policy unchanged) and
                        verifies it with the accepted reader + producer decoder.
``prepare-video``       Pipes the 120 native 640x480 RGB frames one by one through the pinned FFmpeg
                        encoder (exact single-use process grant), reopens/decodes 120 frames + EOF and
                        the exact s000010 seek; source PNG/pixel hashes recorded separately from the
                        lossy derivative pixels. FRAME_INDEX clock; the 30 fps presentation rate is
                        synthetic and attests nothing physical.
``check-prerequisites`` Re-authenticates every input, rebuilds the allowlist core, re-streams the CAD and
                        rebuilds the GLB byte-for-byte, re-verifies the 120 RGB payloads and their pixel
                        hashes, re-decodes the video, and states ``prepared_media_only`` with the
                        unresolved items. No ClipSpec / SurfacePackage, registration or admission.

The old scene-000002 stages (``load_allowlist``/``prepare_model``/``prepare_video``) are never called and
no helper global is mutated; reused low-level helpers have their documented assumptions asserted.
The profile's CAD 5 *candidate hint* is unbound and is never used to select or exclude anything.
Trust: historical local integrity only (``upstream_authenticated: false``). The association is a
QUALITATIVE parent review, not an independent attachment-accuracy annotation.
Exit codes: 0 completed; 1 failure; 2 usage; 3 blocked.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (checked low-level helpers; never mutated)
from tools import workbench_hb_prefix_scout as sc  # noqa: E402  (pure validators only)
from tools import workbench_hb_scene3_profile as pf  # noqa: E402  (pure validators/authentication only)
from tools import workbench_hb_camera_k as ck  # noqa: E402  (pure K validation only; camera never reopened)

TOOL_VERSION = "1"
SCHEMA_ALLOWLIST = "workbench_hb_scene3_media_allowlist_v1"
SCHEMA_MODEL = "workbench_hb_scene3_model_receipt_v1"
SCHEMA_VIDEO = "workbench_hb_scene3_video_receipt_v1"
SCHEMA_PREREQ = "workbench_hb_scene3_media_prerequisites_v1"
SCHEMA_FAILURE = "workbench_hb_scene3_media_failure_v1"
SCHEMA_ASSOCIATION = "workbench_hb_scene3_opening_association_v1"
SCENE = "000003"
FRAME_COUNT = 120
OPENING, SEED_INDEX, SCORING = tuple(range(0, 10)), 9, tuple(range(10, 120))
IMAGE_SIZE = (640, 480)
ALLOWLIST_NAME = "media-allowlist.json"
MODEL_RECEIPT_NAME, VIDEO_NAME, VIDEO_RECEIPT_NAME = "model-receipt.json", "video.mp4", "video-receipt.json"
PREREQ_NAME, FAILURE_NAME, INVALIDATED_PREFIX = "prerequisites-receipt.json", "failure.json", "invalidated-"

# Production pins (accepted; no CLI override exists).
PROFILE_PIN = {"path": ".cache/video-workbench/derived/hb-metric-control-000003-v1/profile/scene3-profile.json",
               "sha256": "de57a486f9014b97d8c00427a7fd97101d02c8ee8e5bce67282f9ade51072fb8"}
CAMERA_K_PIN = {"path": ".cache/video-workbench/derived/hb-metric-control-000003-v1/camera-k/camera-k.json",
                "sha256": "5b542672f25baa23e703132c4f4825ff190cb380df4b56a7d750de020ae1ba78"}
ASSOCIATION_PIN = {"path": ".cache/video-workbench/derived/validation/hb-scene3-opening-cad4-association-v1.json",
                   "sha256": "1e719262a94ef8b9ee9f0f441c056fb9b0a66d749acb7162c3e4050699dc8986", "object_id": 4}
PREPARATION_PIN = ".cache/video-workbench/derived/hb-metric-control-000003-v1"
BASE_INPUT_BYTES = 187_459_695            # accepted total; already charges all 10 CADs and the 1 MiB camera

ASSOCIATION_CAP = 64 << 10
ALLOWLIST_CAP = 128 << 10                 # allowlist share 192 KiB - failure reserve
MODEL_RECEIPT_CAP = 16 << 10              # model share = GLB_CAP + 32 KiB: GLB + receipt + failure reserve
VIDEO_RECEIPT_CAP = 128 << 10             # video share = VIDEO_CAP + 160 KiB
PREREQ_CAP = 64 << 10                     # prerequisites share 96 KiB
FAILURE_RESERVE = 16 << 10
STAGE_CAP_KEYS = {"media-allowlist": "filter-allowlist", "prepare-model": "prepare-model",
                  "prepare-video": "prepare-video", "check-prerequisites": "assemble"}
RENAME_EXCLUSIVE = os.name == "nt"        # Windows rename refuses an existing destination (never replaces)
TRANSIENT_FACTOR = 1 if RENAME_EXCLUSIVE else 2   # POSIX falls back to link+unlink (both names counted)

MEDIA_FLAGS = {"inputs_only": True, "prepared_media_only": True, "neural_executed": False, "gpu_used": False,
               "gt_read": False, "poses": None, "masks": None, "seed_pose": None, "instance_ordinal": None,
               "camera_raw_reopened": False, "assembled": False, "registered": False, "admitted": False,
               "clipspec_published": False, "surface_package_published": False, "quality_claim": False,
               "mobile_verified": False, "phone_performance_claim": False, "family": mi.FAMILY,
               "head_sha256": mi.FLAGS["head_sha256"]}
TRUST = {"trust_level": "historical_local_integrity", "upstream_authenticated": False,
         "basis": ("accepted scene-000003 profile and K derivative, the separately pinned parent association "
                   "review, the accepted scout outputs and the three historical anchors, each pinned by SHA-256 "
                   "and verified before parsing")}
FROZEN_SPLIT = {"opening": list(OPENING), "seed_index": SEED_INDEX, "scoring": list(SCORING),
                "fixed_before_viewing": True}
CLOCK = {"kind": "FRAME_INDEX", "physical_fps": None, "physical_timestamps": None,
         "presentation_fps": 30, "presentation_fps_kind": "synthetic presentation rate; NOT a physical capture rate",
         "prefix_attestation": "neither container duration nor PTS attests a physical 10-second prefix"}
IMAGE = {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1], "crop": None, "rescale": None, "rotation": None}
UNRESOLVED = ["instance_ordinal", "seed_pose", "principal_point_origin", "artwork_placement", "assembly",
              "segmentation", "admission"]
ASSOCIATION_NOTE = ("qualitative parent opening review (separately pinned); NOT an independent attachment-accuracy "
                    "annotation; the profile's CAD 5 candidate hint is unbound and was not used to select or exclude")
PRINCIPAL_POINT = {"origin_convention": None, "status": "UNVERIFIED_separate_review_required",
                   "origin_shift": None, "note": "raw K unchanged; no origin shift is guessed"}
ASSOCIATION_KEYS = frozenset({"schema", "timestamp_utc", "reviewer", "scene", "verdict", "profile", "selection", "cad",
                              "instance_ordinal", "evidence", "boundaries", "limits", "quality_claim"})
BOUNDARY_KEYS = frozenset({"scored_images_used", "gt_used", "poses_used", "evaluator_masks_or_depth_used",
                           "neural_inference_used"})
CORE_KEYS = frozenset({"schema", "tool_version", "label", "pins_kind", "scene", "flags", "trust", "inputs", "anchors",
                       "source", "frozen_split", "clock", "image", "rgb", "rgb_totals", "model", "association",
                       "camera_k", "principal_point", "unresolved", "input_budget", "payload_opens",
                       "payload_stats", "preparation_scope"})
ALLOWLIST_KEYS = CORE_KEYS | {"sources", "runtime", "budget", "reads", "closure"}


def _assert_helper_assumptions() -> None:
    """The reused accepted helpers are scene-agnostic only under these documented values."""
    expected = {"FRAME_COUNT": FRAME_COUNT, "SEED_INDEX": SEED_INDEX, "OPENING": OPENING, "SCORING": SCORING,
                "IMAGE_SIZE": IMAGE_SIZE, "MM_TO_M": 0.001, "FACE_CAP": 200_000, "PORTABLE_TRIANGLES": 20_000,
                "RGB_ARCHIVE": "hb_val_primesense.zip", "MODEL_ARCHIVE": "hb_models.zip", "PRESENTATION_FPS": 30,
                "DATASET_REVISION": "985ebaef7b1bd4bffb2039f18dd81faa647fd060"}
    for name, value in expected.items():
        if getattr(mi, name) != value:
            raise mi.InputsError(f"accepted helper assumption changed: workbench_hb_metric_inputs.{name}")
    if sc.SCENE != SCENE or pf.SCENE != SCENE or pf.IMAGE_SIZE != IMAGE_SIZE or pf.FRAME_COUNT != FRAME_COUNT:
        raise mi.InputsError("accepted scout/profile scene constants changed")


# --------------------------------------------------------------------------
# context and pins


@dataclass(frozen=True)
class SyntheticMediaPins:
    """TEST-ONLY pins for manufactured fixtures. Never settable from the CLI; every output then records
    ``label = "TEST_ONLY"`` / ``pins_kind = "synthetic_test_only"`` and production validation refuses it."""
    profile: pf.SyntheticPins
    profile_sha256: str
    camera_k_sha256: str
    association_sha256: str
    association_object_id: int


@dataclass(frozen=True)
class MediaContext:
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"
    preparation: Path = ROOT / PREPARATION_PIN
    test_only_synthetic_pins: SyntheticMediaPins | None = None

    def mi(self) -> mi.Context:
        return mi.Context(root=self.root, workspace=self.workspace, preparation=Path(os.path.abspath(self.preparation)))


def pins(ctx: MediaContext) -> dict:
    s = ctx.test_only_synthetic_pins
    if s is None:
        pp = pf.pins(pf.ProfileContext())
        return {"kind": "historical", "label": "PRODUCTION_PINNED", "profile_pins": pp,
                "profile_sha256": PROFILE_PIN["sha256"], "camera_k_sha256": CAMERA_K_PIN["sha256"],
                "association_sha256": ASSOCIATION_PIN["sha256"], "association_object_id": ASSOCIATION_PIN["object_id"],
                "base_input_bytes": BASE_INPUT_BYTES}
    if (not isinstance(s.profile, pf.SyntheticPins) or not all(mi._is_sha(v) for v in (
            s.profile_sha256, s.camera_k_sha256, s.association_sha256))
            or type(s.association_object_id) is not int or s.association_object_id not in mi.MODEL_IDS):
        raise mi.UsageError("test-only synthetic media pins are malformed")
    pp = pf.pins(pf.ProfileContext(synthetic=s.profile))
    return {"kind": "synthetic_test_only", "label": "TEST_ONLY", "profile_pins": pp,
            "profile_sha256": s.profile_sha256, "camera_k_sha256": s.camera_k_sha256,
            "association_sha256": s.association_sha256, "association_object_id": s.association_object_id,
            "base_input_bytes": pp["input_accounted"] + pf.CAMERA_CAP}


# --------------------------------------------------------------------------
# honest read accounting (audit hook; inert unless a stage is counting)

_COUNT: dict[str, Any] = {"active": False, "files": {}, "prefixes": (), "counts": collections.Counter()}
_COUNT_HOOK = False


def _count_hook(event: str, args: tuple) -> None:
    c = _COUNT
    if not c["active"] or event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = mi._norm(args[0])
    label = c["files"].get(path)
    if label is None:
        for prefix, plabel in c["prefixes"]:
            if path.startswith(prefix):
                label = plabel
                break
    if label is not None:
        c["counts"][f"{label}:{'write' if mi._write_mode(args) else 'read'}"] += 1


@contextmanager
def _counting() -> Iterator["_Counter"]:
    global _COUNT_HOOK
    if _COUNT["active"]:
        raise mi.GuardError("nested read counters are refused")
    if not _COUNT_HOOK:
        sys.addaudithook(_count_hook)
        _COUNT_HOOK = True
    _COUNT.update(active=True, files={}, prefixes=(), counts=collections.Counter())
    try:
        yield _Counter()
    finally:
        _COUNT.update(active=False, files={}, prefixes=())


class _Counter:
    def watch(self, path: Path, label: str) -> None:
        _COUNT["files"] = {**_COUNT["files"], mi._norm(path): label}

    def watch_prefix(self, path: Path, label: str) -> None:
        _COUNT["prefixes"] = (*_COUNT["prefixes"], (mi._norm(path) + os.sep, label))

    def snapshot(self) -> dict:
        return dict(sorted(_COUNT["counts"].items()))

    def reads(self, label: str) -> int:
        return _COUNT["counts"].get(f"{label}:read", 0)


# --------------------------------------------------------------------------
# bounded hashing, ledger, exclusive publication, failures


def sealed_sha256(path: Path, cap: int) -> str:
    """Bounded hash: regular non-link file of at most ``cap`` bytes; a file that grows is refused at the
    cap without reading the remainder."""
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
        raise mi.InputsError(f"{Path(path).name}: not a regular file within its {cap} byte cap")
    h, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(min(mi.CHUNK, cap + 1 - count))
            if not block:
                break
            count += len(block)
            if count > cap:
                raise mi.InputsError(f"{Path(path).name}: grew beyond its {cap} byte cap")
            h.update(block)
    return h.hexdigest()


def _read_sealed(path: Path, cap: int, expected: str, what: str) -> bytes:
    raw = mi.read_bounded(path, cap, what)
    if mi.sha256_bytes(raw) != expected:
        raise mi.InputsError(f"{what}: SHA-256 differs from the pinned/expected value")   # BEFORE parsing
    return raw


JSON_NAME = re.compile(r"\.json(?:$|\.)", re.IGNORECASE)   # x.json, .x.json.staging-*, invalidated-x.json
USAGE_ENTRY_LIMIT = 200_000


def logical_usage(path: Path) -> dict:
    """Bounded, link-refusing logical accounting owned by THIS module: every regular file under ``path``
    counts toward the total, and every name containing ``.json`` (published, ``.x.json.staging-*``
    partials, ``invalidated-*.json``, failure records) counts as METADATA. The accepted
    ``mi.scope_usage`` classifies only a final ``.json`` suffix, so both are taken and the larger wins."""
    total = metadata = files = seen = 0
    if not os.path.lexists(path):
        return {"total": 0, "metadata": 0, "files": 0}
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        seen += 1
        if seen > USAGE_ENTRY_LIMIT:
            raise mi.Blocked(f"more than {USAGE_ENTRY_LIMIT} entries under {path}; refusing unbounded accounting")
        if mi._is_link(current):
            raise mi.Blocked(f"link inside the accounted scope: {current}")
        st = os.lstat(current)
        if stat.S_ISDIR(st.st_mode):
            with os.scandir(current) as entries:
                stack.extend(Path(e.path) for e in entries)
        elif stat.S_ISREG(st.st_mode):
            total += st.st_size
            files += 1
            if JSON_NAME.search(current.name):
                metadata += st.st_size
        else:
            raise mi.Blocked(f"non-regular entry inside the accounted scope: {current}")
    return {"total": total, "metadata": metadata, "files": files}


def _usage(path: Path) -> dict:
    accepted, own = mi.scope_usage(path), logical_usage(path)
    return {k: max(accepted[k], own[k]) for k in ("total", "metadata", "files")}


class MediaLedger:
    """Before every encoded write: the substage directory within the accepted per-stage share
    (``mi.STAGE_CAPS``), the WHOLE preparation scope (accepted profile/K outputs, retained failures,
    staging partials and aliases, JSON staging names counted as metadata) within the 12 MiB / 512 KiB
    aggregates, a FRESH global 1 GiB derived check and the 1 GiB free floor. Normal writes keep room for a
    failure record. Ownership is recorded only when THIS module's commit created the final name, from the
    owned staging identity captured BEFORE the exclusive commit (never by the encoder's early ``record``)."""

    def __init__(self, mctx: mi.Context, out_dir: Path, stage: str):
        self.ctx, self.dir, self.stage, self.scope = mctx, out_dir, stage, mctx.scope
        self.stage_cap = mi.STAGE_CAPS[STAGE_CAP_KEYS[stage]]["total"]
        self.files: dict[str, int] = {}
        self.owned: dict[str, tuple[int, int]] = {}
        self.staging: dict[str, int] = {}

    def check_sequence(self, steps: Sequence[tuple[int, bool]], *, failure: bool = False, factor: int = TRANSIENT_FACTOR,
                       global_check: bool = True) -> dict:
        if not steps or any(type(n) is not int or n < 0 for n, _ in steps):
            raise mi.InputsError("invalid write size")
        scope, stage = _usage(self.scope), _usage(self.dir)
        reserve = 0 if failure else TRANSIENT_FACTOR * FAILURE_RESERVE
        done_total = done_meta = peak_max = 0
        for nbytes, metadata in steps:
            peak = factor * nbytes
            peak_max = max(peak_max, done_total + peak)
            if stage["total"] + done_total + peak > self.stage_cap - reserve:
                raise mi.Blocked(f"{self.stage}: substage would peak at {stage['total'] + done_total + peak} bytes, "
                                 f"above {self.stage_cap - reserve}")
            if scope["total"] + done_total + peak > mi.FINAL_OUTPUTS_CAP - reserve:
                raise mi.Blocked(f"{self.stage}: preparation scope would peak at {scope['total'] + done_total + peak} "
                                 f"bytes, above {mi.FINAL_OUTPUTS_CAP - reserve}")
            if metadata:
                if scope["metadata"] + done_meta + peak > mi.METADATA_AGGREGATE_CAP - reserve:
                    raise mi.Blocked(f"{self.stage}: scope metadata would exceed {mi.METADATA_AGGREGATE_CAP - reserve}")
                done_meta += nbytes
            done_total += nbytes
        if mi.shutil.disk_usage(self.ctx.workspace).free - peak_max < mi.FREE_RESERVE:
            raise mi.Blocked("the 1 GiB free-disk reserve would be violated")
        if global_check:
            mi.workspace_budget(self.ctx, peak_max)
        return {"scope": scope, "stage": stage}

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        return self.check_sequence([(nbytes, metadata)], failure=failure)

    def own(self, name: str) -> None:
        st = os.lstat(self.dir / name)
        self.owned[name] = (st.st_dev, st.st_ino)

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
                "staging_reported": dict(self.staging), "scope_usage_now": _usage(self.scope),
                "caps": {"substage": self.stage_cap, "scope_total": mi.FINAL_OUTPUTS_CAP,
                         "scope_metadata": mi.METADATA_AGGREGATE_CAP, "failure_reserve": FAILURE_RESERVE},
                "publication": "exclusive rename (Windows)" if RENAME_EXCLUSIVE else "link+unlink (2x counted)"}


class _EncoderLedger:
    """Adapter for the accepted ``FfmpegPipeEncoder``: each stdout chunk APPENDS to the single owned
    staging file (no alias), so growth is 1x. EVERY append is checked against the substage, the whole
    scope, the free floor AND a fresh global derived check. Its early ``record('video.mp4')`` from
    ``__exit__`` is kept as staging accounting only and never confers final ownership."""

    def __init__(self, ledger: MediaLedger):
        self.ledger = ledger

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        return self.ledger.check_sequence([(nbytes, False)], factor=1, global_check=True)

    def record(self, name: str, nbytes: int, *, metadata: bool) -> None:
        self.ledger.staging[f"encoder-reported:{name}"] = nbytes


def _commit(ledger: MediaLedger, staging: Path, name: str) -> None:
    """Exclusive, no-overwrite publication of an owned staging file; never ``os.replace``, never deletes a
    foreign name. The owned staging identity (st_dev, st_ino) is captured BEFORE the commit and becomes
    the final name's ownership the moment the exclusive commit succeeds, with no post-commit lookup that
    could fail. Windows ``os.rename`` refuses an existing destination and keeps the file identity; POSIX
    uses link (refuses an existing name) + unlink of the owned staging alias."""
    final = ledger.dir / name
    st = os.lstat(staging)
    if not stat.S_ISREG(st.st_mode):
        raise mi.InputsError(f"{staging.name}: owned staging is not a regular file")
    identity = (st.st_dev, st.st_ino)
    if os.path.lexists(final):
        raise mi.InputsError(f"{final} already exists; a foreign output is never replaced")
    if RENAME_EXCLUSIVE:
        os.rename(staging, final)
        ledger.owned[name] = identity
    else:
        os.link(staging, final)
        ledger.owned[name] = identity
        os.unlink(staging)


def publish(ledger: MediaLedger, name: str, data: bytes, *, metadata: bool = True, failure: bool = False) -> dict:
    if "/" in name or "\\" in name or name.startswith("."):
        raise mi.InputsError(f"invalid output name {name!r}")
    ledger.check(len(data), metadata=metadata, failure=failure)
    if os.path.lexists(ledger.dir / name):
        raise mi.InputsError(f"{ledger.dir / name} already exists; a foreign output is never replaced")
    staging = ledger.dir / f".{name}.staging-{secrets.token_hex(6)}"
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if type(n) is not int or n <= 0 or n > len(view):
                raise mi.InputsError(f"{name}: staging write reported invalid progress {n!r}")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = mi.sha256_bytes(data)
    if sealed_sha256(staging, len(data)) != digest:
        raise mi.InputsError(f"{name}: staged bytes do not verify (owned staging retained)")
    _commit(ledger, staging, name)
    ledger.files[name] = len(data)
    if sealed_sha256(ledger.dir / name, len(data)) != digest:
        raise mi.InputsError(f"{name}: published bytes do not verify")
    return {"name": name, "sha256": digest, "bytes": len(data)}


def _failure_record(ledger: MediaLedger, stage: str, exc: BaseException, extra: Mapping | None = None) -> None:
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
                    retained_files={"count": len(retained), "bytes": sum(retained.values()), "names": sorted(retained)[:16]})
        data = mi.dump_json(body)
    if len(data) > FAILURE_RESERVE:
        data = mi.dump_json({"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage, "status": "failed",
                             "reason": body["reason"][:300], "completed_receipt_published": False})
    try:
        if len(data) > FAILURE_RESERVE:
            raise mi.Blocked("failure record exceeds its reserve")
        publish(ledger, FAILURE_NAME, data, failure=True)     # fresh scope/substage/global/free checks inside
    except Exception as publish_exc:  # noqa: BLE001 - terminal record, never mask the original error
        line = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        sys.stderr.write(f"failure.json NOT published ({type(publish_exc).__name__}: {str(publish_exc)[:300]}); "
                         f"the record follows on ONE stderr line as JSON:\n{line}\n")


def _fail_stage(ledger: MediaLedger, stage: str, exc: BaseException, extra: Mapping | None = None) -> None:
    """Every final name THIS stage created and that is still the very file it created is renamed to
    ``invalidated-<name>`` (bytes kept); foreign names, staging partials and earlier stages' outputs
    (including the accepted profile/K) are never renamed or deleted."""
    records = []
    for name in sorted(ledger.owned):
        if name == FAILURE_NAME or not ledger.owned_file(name):
            continue
        src, dst = ledger.dir / name, ledger.dir / f"{INVALIDATED_PREFIX}{name}"
        record: dict = {"name": name}
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
    _failure_record(ledger, stage, exc, {"invalidated_outputs": records, **(dict(extra) if extra else {})})


def _sources() -> dict:
    return dict(sorted({**pf._sources(),
                        Path(__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(__file__))}.items()))


def _state(files: Mapping[str, tuple[Path, int]], extra: Mapping | None = None) -> dict:
    """Closure state: bounded hashes of metadata files (refusals reported, not raised), sources, runtime."""
    hashes = {}
    for key, (path, cap) in sorted(files.items()):
        try:
            hashes[key] = sealed_sha256(path, cap)
        except (mi.InputsError, OSError) as exc:
            hashes[key] = f"refused: {type(exc).__name__}: {str(exc)[:160]}"
    return {"files": hashes, "sources": _sources(), "runtime": mi.runtime_identity(), **(dict(extra) if extra else {})}


def _payload_identity(paths: Sequence[Path]) -> list:
    out = []
    for path in paths:
        try:
            st = os.lstat(path)
            out.append([st.st_size, st.st_mtime_ns, st.st_ino, stat.S_ISREG(st.st_mode)])
        except OSError as exc:
            out.append(f"refused: {type(exc).__name__}")
    return out


def _post_publication(before: Mapping, current: Mapping, out_dir: Path, outputs: Mapping[str, tuple[str, int]]) -> None:
    if current != before:
        changed = sorted(k for k in set(before) | set(current) if before.get(k) != current.get(k))
        raise mi.InputsError(f"closure changed after publication: {changed}")
    names = sorted(e.name for e in out_dir.iterdir())
    if names != sorted(outputs):
        raise mi.InputsError(f"closure changed after publication: output names {names}")
    for name, (sha, cap) in outputs.items():
        if sealed_sha256(out_dir / name, cap) != sha:
            raise mi.InputsError(f"closure changed after publication: {name} bytes")


# --------------------------------------------------------------------------
# pure validation


def _split(i: int) -> str:
    return "opening_seed" if i == SEED_INDEX else "opening" if i in OPENING else "scoring"


def _ints(value: Any, n: int) -> bool:
    return isinstance(value, list) and len(value) == n and all(type(v) is int for v in value)


def validate_association(doc: Any, *, profile_doc: Mapping, profile_rel: str, p: Mapping) -> dict:
    """Exact schema of the parent's qualitative opening association. The CAD id comes ONLY from this
    pinned review (never from the profile's unbound candidate hint)."""
    if not isinstance(doc, dict) or set(doc) != ASSOCIATION_KEYS:
        raise mi.InputsError("association review has missing or unknown fields")
    if (doc["schema"] != SCHEMA_ASSOCIATION or doc["scene"] != SCENE or doc["verdict"] != "positive_visual_match"
            or doc["instance_ordinal"] is not None or doc["quality_claim"] is not False
            or not isinstance(doc["reviewer"], str) or not doc["reviewer"].strip()
            or not isinstance(doc["limits"], str) or not isinstance(doc["timestamp_utc"], str)):
        raise mi.InputsError("association review schema/scene/verdict/ordinal/quality fields differ")
    if doc["profile"] != {"path": profile_rel, "sha256": p["profile_sha256"]}:
        raise mi.InputsError("association review does not bind the accepted profile")
    boundaries = doc["boundaries"]
    if not isinstance(boundaries, Mapping) or set(boundaries) != BOUNDARY_KEYS or any(v is not False for v in boundaries.values()):
        raise mi.InputsError("association review boundaries must all be false (no scored images/GT/poses/masks/NN)")
    sel = doc["selection"]
    if not isinstance(sel, Mapping) or set(sel) != {"frame_id", "source_frame_index", "source_png_sha256",
                                                     "native_image_size", "positive_point_xy", "roi_xyxy"}:
        raise mi.InputsError("association selection has missing or unknown fields")
    index = sel["source_frame_index"]
    if type(index) is not int or index not in OPENING or sel["frame_id"] != f"s{index:06d}":
        raise mi.InputsError("association selection must be an opening frame (0..9); scored frames are refused")
    frame_rows = profile_doc["frames"]["rows"]
    if sel["source_png_sha256"] != frame_rows[index][2] or sel["native_image_size"] != list(IMAGE_SIZE):
        raise mi.InputsError("association source PNG/size differ from the accepted profile")
    point, roi = sel["positive_point_xy"], sel["roi_xyxy"]
    if not (_ints(point, 2) and _ints(roi, 4) and 0 <= roi[0] < roi[2] <= IMAGE_SIZE[0]
            and 0 <= roi[1] < roi[3] <= IMAGE_SIZE[1] and roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]):
        raise mi.InputsError("association point/ROI must be integer native pixels with the point inside the ROI")
    cad = doc["cad"]
    if not isinstance(cad, Mapping) or set(cad) != {"object_id", "member", "compressed_sha256", "uncompressed_sha256"}:
        raise mi.InputsError("association CAD block has missing or unknown fields")
    oid = cad["object_id"]
    if type(oid) is not int or oid not in mi.MODEL_IDS or oid != p["association_object_id"]:
        raise mi.InputsError("association CAD id is not the explicitly pinned reviewed id")
    if cad["member"] != f"models/obj_{oid:06d}.ply" or not mi._is_sha(cad["compressed_sha256"]) \
            or not mi._is_sha(cad["uncompressed_sha256"]):
        raise mi.InputsError("association CAD member/seals malformed")
    ev = doc["evidence"]
    inspected = ev.get("opening_source_indices_inspected") if isinstance(ev, Mapping) else None
    preview = ev.get("opening_preview") if isinstance(ev, Mapping) else None
    if (not isinstance(inspected, list) or not inspected or any(type(i) is not int or i not in OPENING for i in inspected)
            or not isinstance(preview, Mapping) or preview.get("sha256") != p["profile_pins"]["scout_receipt_sha256"]):
        raise mi.InputsError("association evidence must stay within the opening and bind the accepted preview")
    return doc


def validate_k_doc(doc: Any, *, profile_doc: Mapping, profile_rel: str, p: Mapping) -> dict:
    """The accepted K derivative; raw K re-validated from its lexemes (the camera is NOT reopened)."""
    pp = p["profile_pins"]
    if not isinstance(doc, dict):
        raise mi.InputsError("camera-K record must be a JSON object")
    fixed = {"schema": pf.SCHEMA_K, "tool_version": pf.TOOL_VERSION, "scene": SCENE, "flags": pf.K_FLAGS,
             "trust": pf.TRUST, "pins_kind": pp["kind"], "adjustments": pf.ADJUSTMENTS,
             "principal_point": pf.PRINCIPAL_POINT, "profile": {"path": profile_rel, "sha256": p["profile_sha256"]}}
    for key, value in fixed.items():
        if doc.get(key) != value:
            raise mi.InputsError(f"camera-K field {key!r} differs from the accepted contract")
    k = doc.get("K")
    if not isinstance(k, Mapping) or not isinstance(k.get("lexemes"), list):
        raise mi.InputsError("camera-K record lacks raw lexemes")
    checked = ck.validate_k_window({0: list(k["lexemes"])}, [0], IMAGE_SIZE)
    if any(k.get(f) != checked[f] for f in ("values", "fx", "fy", "cx", "cy", "matrix")) \
            or k.get("window") != {"first": 0, "last": FRAME_COUNT - 1, "count": FRAME_COUNT} \
            or k.get("constant_across_window") is not True:
        raise mi.InputsError("camera-K values/window differ from its raw lexemes")
    row = profile_doc["camera"]["row"]
    source = doc.get("camera_source")
    if not isinstance(source, Mapping) or any(source.get(key) != value for key, value in row.items()):
        raise mi.InputsError("camera-K source row differs from the accepted profile camera row")
    return {"values": checked["values"], "lexemes": list(k["lexemes"]), "fx": checked["fx"], "fy": checked["fy"],
            "cx": checked["cx"], "cy": checked["cy"], "raw_unchanged": True, "origin_shift": None,
            "camera_member_sha256": row["expanded_sha256"]}


def derive_rows(index: Any, *, profile_doc: Mapping, plan: Mapping, association: Mapping) -> tuple[list, dict]:
    """Complete RGB 0..119 + ONLY the reviewed CAD row from the pinned historical index; every identity is
    compared with the accepted profile, the scout plan and the review."""
    resources = index.get("resources") if isinstance(index, Mapping) else None
    if not isinstance(resources, Mapping):
        raise mi.InputsError("resources index lacks resources")
    rgb = []
    for i in range(FRAME_COUNT):
        member = f"{sc.RGB_PREFIX}{i:06d}.png"
        row = mi.validate_index_row(resources.get(member), member, mi.RGB_ARCHIVE, "rgb", mi.MAX_RGB_EXPANDED)
        compact = [i, member, row["uncompressed_sha256"], row["compressed_sha256"], row["crc32"], row["expanded_bytes"],
                   row["compressed_bytes"]]
        if profile_doc["frames"]["rows"][i] != compact:
            raise mi.InputsError(f"{member}: index row differs from the accepted profile identity")
        scout_row = plan["rgb"][i]
        if any(scout_row.get(k) != v for k, v in row.items()):
            raise mi.InputsError(f"{member}: index row differs from the accepted scout plan row")
        rgb.append({"index": i, "frame_id": f"s{i:06d}", "split": _split(i), **row})
    oid = association["cad"]["object_id"]
    member = f"models/obj_{oid:06d}.ply"
    row = mi.validate_index_row(resources.get(member), member, mi.MODEL_ARCHIVE, "model", mi.MAX_MODEL_EXPANDED)
    bindings = [b for b in index.get("selectedMeshBindings") or [] if isinstance(b, Mapping) and b.get("objectId") == oid]
    if (len(bindings) != 1 or bindings[0].get("meshMember") != member
            or bindings[0].get("meshSHA256") != row["uncompressed_sha256"]
            or bindings[0].get("meshExpandedBytes") != row["expanded_bytes"]
            or bindings[0].get("meshCRC32") != f"{row['crc32']:08x}"):
        raise mi.InputsError(f"object {oid}: index mesh binding disagrees with its member row")
    if (row["compressed_sha256"] != association["cad"]["compressed_sha256"]
            or row["uncompressed_sha256"] != association["cad"]["uncompressed_sha256"]):
        raise mi.InputsError(f"object {oid}: index seals differ from the association review")
    return rgb, {"object_id": oid, **row, "identity_note": mi.IDENTITY_NOTE}


def validate_allowlist_doc(doc: Any, p: Mapping) -> dict:
    """Full pure re-validation of a media allowlist (before any payload permission)."""
    pp = p["profile_pins"]
    if not isinstance(doc, dict) or set(doc) != ALLOWLIST_KEYS:
        raise mi.InputsError("media allowlist has missing or unknown fields")
    fixed = {"schema": SCHEMA_ALLOWLIST, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
             "scene": SCENE, "flags": MEDIA_FLAGS, "trust": TRUST, "frozen_split": FROZEN_SPLIT, "clock": CLOCK,
             "image": IMAGE, "principal_point": PRINCIPAL_POINT, "unresolved": UNRESOLVED}
    for key, value in fixed.items():
        if doc[key] != value:
            raise mi.InputsError(f"media allowlist field {key!r} differs (a TEST_ONLY allowlist cannot pass as production)")
    if type(doc["payload_opens"]) is not int or doc["payload_opens"] != 0:
        raise mi.InputsError("media allowlist must record zero payload opens")
    inputs = doc["inputs"]
    try:
        bound = {"profile": inputs["profile"]["sha256"], "camera_k": inputs["camera_k"]["sha256"],
                 "association": inputs["association"]["sha256"], "scout_plan": inputs["scout_plan"]["sha256"],
                 "scout_receipt": inputs["scout_receipt"]["sha256"]}
    except (KeyError, TypeError):
        raise mi.InputsError("media allowlist inputs block malformed") from None
    if bound != {"profile": p["profile_sha256"], "camera_k": p["camera_k_sha256"], "association": p["association_sha256"],
                 "scout_plan": pp["scout_plan_sha256"], "scout_receipt": pp["scout_receipt_sha256"]}:
        raise mi.InputsError("media allowlist does not bind the pinned inputs")
    anchors = doc["anchors"]
    if not isinstance(anchors, Mapping) or set(anchors) != set(mi.ANCHORS) or any(
            not isinstance(a, Mapping) or a.get("sha256") != pp["anchors"][k] for k, a in anchors.items()):
        raise mi.InputsError("media allowlist anchors differ from the pins")
    src = doc["source"]
    if (not isinstance(src, Mapping) or src.get("dataset_revision") != mi.DATASET_REVISION or src.get("scene") != SCENE
            or src.get("rgb_archive") != mi.RGB_ARCHIVE or src.get("model_archive") != mi.MODEL_ARCHIVE
            or not isinstance(src.get("payload_root"), str)
            or PurePosixPath(src["payload_root"]) != PurePosixPath(anchors["resources_index"]["path"]).parent / "payloads"):
        raise mi.InputsError("media allowlist source identity/payload root differs")
    rgb = doc["rgb"]
    if not isinstance(rgb, list) or len(rgb) != FRAME_COUNT:
        raise mi.InputsError("media allowlist must hold exactly 120 RGB rows")
    for i, row in enumerate(rgb):
        mi._revalidate_row(row, f"{sc.RGB_PREFIX}{i:06d}.png", mi.RGB_ARCHIVE, "rgb", mi.MAX_RGB_EXPANDED, mi.RGB_ROW_KEYS)
        if type(row["index"]) is not int or row["index"] != i or row["frame_id"] != f"s{i:06d}" or row["split"] != _split(i):
            raise mi.InputsError(f"media allowlist RGB row {i} index/frame/split differ")
    totals = {"encoded_bytes": sum(r["compressed_bytes"] for r in rgb), "expanded_bytes": sum(r["expanded_bytes"] for r in rgb)}
    if doc["rgb_totals"] != totals or totals["encoded_bytes"] != pp["scout"]["rgb_encoded_bytes"]:
        raise mi.InputsError("media allowlist RGB totals differ")
    model = doc["model"]
    if not isinstance(model, Mapping) or model.get("object_id") != p["association_object_id"] \
            or type(model.get("object_id")) is not int or model.get("identity_note") != mi.IDENTITY_NOTE:
        raise mi.InputsError("media allowlist model is not exactly the reviewed CAD")
    mi._revalidate_row(model, f"models/obj_{model['object_id']:06d}.ply", mi.MODEL_ARCHIVE, "model",
                       mi.MAX_MODEL_EXPANDED, mi.MODEL_ROW_KEYS)
    assoc = doc["association"]
    if (not isinstance(assoc, Mapping) or assoc.get("object_id") != model["object_id"] or assoc.get("instance_ordinal") is not None
            or assoc.get("independent_attachment_accuracy") is not False):
        raise mi.InputsError("media allowlist association block differs")
    kblock = doc["camera_k"]
    if not isinstance(kblock, Mapping) or ck.validate_k_window({0: list(kblock.get("lexemes") or [])}, [0], IMAGE_SIZE)["values"] \
            != kblock.get("values") or kblock.get("raw_unchanged") is not True or kblock.get("origin_shift") is not None:
        raise mi.InputsError("media allowlist K block differs from its raw lexemes")
    stats = doc["payload_stats"]
    expected_paths = [r["payload_path"] for r in rgb] + [model["payload_path"]]
    if not isinstance(stats, Mapping) or sorted(stats) != sorted(expected_paths) or any(
            stats[r["payload_path"]] != r["compressed_bytes"] for r in [*rgb, model]):
        raise mi.InputsError("media allowlist payload stats differ from its rows")
    accounting = doc["input_budget"]
    if not isinstance(accounting, Mapping) or accounting.get("existing_input_bytes_parent_declared") != p["base_input_bytes"]:
        raise mi.InputsError("media allowlist input accounting differs from the accepted base")
    if not isinstance(doc["preparation_scope"], str):
        raise mi.InputsError("media allowlist preparation scope malformed")
    return doc


# --------------------------------------------------------------------------
# shared authentication (guard active; exact read-only metadata exceptions only)


def _locate(mctx: mi.Context, value: Any, name: str | None, what: str, *, inside: bool) -> Path:
    path = mi.contained_input(mctx, value, what)
    if name is not None and path.name != name:
        raise mi.InputsError(f"{what} must be a {name}; refused {path.name!r} unread")
    if name is None and path.suffix.lower() != ".json":
        raise mi.InputsError(f"{what} must be a .json review")
    (mi.inside_scope if inside else mi.outside_scope_input)(mctx, path, what)
    return path


def _authenticate(mctx: mi.Context, p: Mapping, counter: _Counter, profile_path: Path, k_path: Path,
                  assoc_path: Path) -> dict:
    """Pinned-SHA-before-parse reads of the profile, K and association; re-authentication of the profile by
    rebuilding its core from the scout outputs + anchors; one more pinned index read for row derivation.
    The camera is stat'ed by the accepted profile authentication but NEVER opened."""
    pp = p["profile_pins"]
    for path, label in ((profile_path, "profile"), (k_path, "camera_k"), (assoc_path, "association")):
        counter.watch(path, label)
    profile_doc = pf.validate_profile_doc(mi.strict_json(_read_sealed(profile_path, pf.PROFILE_CAP, p["profile_sha256"],
                                                                      "profile"), "profile"), pp)
    if profile_doc["preparation_scope"] != mi.repo_rel(mctx, mctx.scope):
        raise mi.InputsError("profile was produced for a different preparation scope")
    if profile_doc["input_accounting"].get("total") != p["base_input_bytes"]:
        raise mi.InputsError("profile input accounting differs from the accepted base total")
    profile_rel = mi.repo_rel(mctx, profile_path)
    kdoc = mi.strict_json(_read_sealed(k_path, pf.K_CAP, p["camera_k_sha256"], "camera-K record"), "camera-K record")
    kblock = validate_k_doc(kdoc, profile_doc=profile_doc, profile_rel=profile_rel, p=p)
    assoc_raw = _read_sealed(assoc_path, ASSOCIATION_CAP, p["association_sha256"], "association review")
    association = validate_association(mi.strict_json(assoc_raw, "association review"), profile_doc=profile_doc,
                                       profile_rel=profile_rel, p=p)
    plan_path = _locate(mctx, profile_doc["scout"]["plan"]["path"], sc.PLAN_NAME, "scout plan", inside=False)
    receipt_path = _locate(mctx, profile_doc["scout"]["preview_receipt"]["path"], sc.RECEIPT_NAME,
                           "scout preview receipt", inside=False)
    counter.watch(plan_path, "scout_plan")
    counter.watch(receipt_path, "scout_receipt")
    camera_path = ck.canonical_reuse_path(mctx.root, pp["reuse_root"], pf.CAMERA_MEMBER)
    counter.watch(camera_path, "camera")
    for key, anchor in profile_doc["anchors"].items():                    # watched BEFORE they are read
        counter.watch(mi.contained_input(mctx, anchor["path"], key, allow_denied=True), f"anchor:{key}")
    auth = pf.authenticate_inputs(mctx, pp, plan_path, receipt_path)       # adds exact read-only anchor exceptions
    for key, path in auth["anchor_paths"].items():
        if _COUNT["files"].get(mi._norm(path)) != f"anchor:{key}":
            raise mi.InputsError(f"anchor {key} path differs between the profile and the scout plan")
    if pf.dump_compact(pf.build_profile_core(mctx, pp, auth)) != pf.dump_compact({k: profile_doc[k] for k in pf.CORE_KEYS}):
        raise mi.InputsError("accepted profile differs from a fresh re-authentication of its inputs")
    index_path = auth["anchor_paths"]["resources_index"]
    index = mi.strict_json(_read_sealed(index_path, mi.MAX_ANCHOR_BYTES["resources_index"], pp["anchors"]["resources_index"],
                                        "resources index"), "resources index")
    rgb, model = derive_rows(index, profile_doc=profile_doc, plan=auth["plan"], association=association)
    del index
    if counter.reads("camera"):
        raise mi.GuardError("the raw camera file was opened during metadata authentication")
    payload_root = index_path.parent / "payloads"
    stats = {}
    for row in [*rgb, model]:
        path = mi.safe_member_path(payload_root.parent, row["payload_path"], row["member"])
        st = os.lstat(path) if os.path.lexists(path) else None                  # stat only; never opened
        if st is None or not stat.S_ISREG(st.st_mode) or st.st_size != row["compressed_bytes"]:
            raise mi.Blocked(f"{row['member']}: cached payload missing or not of the indexed size")
        stats[row["payload_path"]] = st.st_size
    return {"profile_doc": profile_doc, "profile_rel": profile_rel, "kblock": kblock, "association": association,
            "association_bytes": len(assoc_raw), "auth": auth, "plan_path": plan_path, "receipt_path": receipt_path,
            "rgb": rgb, "model": model, "payload_root": payload_root, "payload_stats": stats}


def _core(mctx: mi.Context, p: Mapping, a: Mapping, k_path: Path, assoc_path: Path) -> dict:
    plan, assoc = a["auth"]["plan"], a["association"]
    pp = p["profile_pins"]
    return {"schema": SCHEMA_ALLOWLIST, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
            "scene": SCENE, "flags": dict(MEDIA_FLAGS), "trust": dict(TRUST),
            "inputs": {"profile": {"path": a["profile_rel"], "sha256": p["profile_sha256"]},
                       "camera_k": {"path": mi.repo_rel(mctx, k_path), "sha256": p["camera_k_sha256"]},
                       "association": {"path": mi.repo_rel(mctx, assoc_path), "sha256": p["association_sha256"],
                                       "bytes": a["association_bytes"]},
                       "scout_plan": {"path": mi.repo_rel(mctx, a["plan_path"]), "sha256": pp["scout_plan_sha256"]},
                       "scout_receipt": {"path": mi.repo_rel(mctx, a["receipt_path"]), "sha256": pp["scout_receipt_sha256"]}},
            "anchors": plan["anchors"],
            "source": {"corpus": mi.CORPUS, "dataset_revision": mi.DATASET_REVISION, "scene": SCENE,
                       "rgb_archive": mi.RGB_ARCHIVE, "model_archive": mi.MODEL_ARCHIVE,
                       "selection_digest": plan["source"]["selection_digest"], "payload_root": plan["source"]["payload_root"]},
            "frozen_split": dict(FROZEN_SPLIT), "clock": dict(CLOCK), "image": dict(IMAGE), "rgb": a["rgb"],
            "rgb_totals": {"encoded_bytes": sum(r["compressed_bytes"] for r in a["rgb"]),
                           "expanded_bytes": sum(r["expanded_bytes"] for r in a["rgb"])},
            "model": a["model"],
            "association": {"object_id": assoc["cad"]["object_id"], "member": assoc["cad"]["member"],
                            "frame_id": assoc["selection"]["frame_id"],
                            "source_frame_index": assoc["selection"]["source_frame_index"],
                            "positive_point_xy": assoc["selection"]["positive_point_xy"],
                            "roi_xyxy": assoc["selection"]["roi_xyxy"], "verdict": assoc["verdict"],
                            "reviewer": assoc["reviewer"], "instance_ordinal": None,
                            "independent_attachment_accuracy": False, "note": ASSOCIATION_NOTE},
            "camera_k": a["kblock"], "principal_point": dict(PRINCIPAL_POINT), "unresolved": list(UNRESOLVED),
            "input_budget": {**mi.input_budget(p["base_input_bytes"], a["association_bytes"], 0),
                             "note": ("accepted base (already charging all 10 CADs and the 1 MiB camera) + the new "
                                      "association review bytes; cached payloads are not double-counted")},
            "payload_opens": 0, "payload_stats": a["payload_stats"], "preparation_scope": mi.repo_rel(mctx, mctx.scope)}


def _input_files(mctx: mi.Context, a: Mapping, profile_path: Path, k_path: Path, assoc_path: Path) -> dict:
    files = {"profile": (profile_path, pf.PROFILE_CAP), "camera_k": (k_path, pf.K_CAP),
             "association": (assoc_path, ASSOCIATION_CAP), "scout_plan": (a["plan_path"], sc.PLAN_CAP),
             "scout_receipt": (a["receipt_path"], sc.RECEIPT_CAP)}
    for key, path in a["auth"]["anchor_paths"].items():
        files[f"anchor:{key}"] = (path, mi.MAX_ANCHOR_BYTES[key])
    return files


def _pinned_inputs(p: Mapping) -> dict:
    pp = p["profile_pins"]
    return dict(sorted({"profile": p["profile_sha256"], "camera_k": p["camera_k_sha256"],
                        "association": p["association_sha256"], "scout_plan": pp["scout_plan_sha256"],
                        "scout_receipt": pp["scout_receipt_sha256"],
                        **{f"anchor:{k}": v for k, v in pp["anchors"].items()}}.items()))


def _check_pins(given: Mapping[str, Any], pinned: Mapping[str, str]) -> None:
    for key, value in given.items():
        mi.check_sha(value, f"expected {key} SHA-256")
        if value != pinned[key]:
            raise mi.Blocked(f"expected {key} SHA-256 is not the accepted pin; no trust can be derived")


def _load_allowlist(mctx: mi.Context, p: Mapping, counter: _Counter, allowlist: Any, expected: str) -> tuple[dict, Path, Path]:
    path = _locate(mctx, allowlist, ALLOWLIST_NAME, "media allowlist", inside=True)
    counter.watch(path, "allowlist")
    doc = validate_allowlist_doc(mi.strict_json(_read_sealed(path, ALLOWLIST_CAP, expected, "media allowlist"),
                                                "media allowlist"), p)
    if doc["preparation_scope"] != mi.repo_rel(mctx, mctx.scope):
        raise mi.InputsError("media allowlist was produced for a different preparation scope")
    payload_root = Path(os.path.abspath(mctx.root / doc["source"]["payload_root"]))
    if payload_root.name != "payloads" or Path(os.path.abspath(mctx.root)) not in payload_root.parents:
        raise mi.InputsError("media allowlist payload root is not a contained payloads directory")
    mi._no_links(Path(os.path.abspath(mctx.root)), payload_root)
    mi.outside_scope_input(mctx, payload_root, "payload root")
    counter.watch_prefix(payload_root, "payload")
    return doc, path, payload_root


def _payload_paths(payload_root: Path, rows: Sequence[Mapping]) -> list[Path]:
    paths = []
    for row in rows:
        path = mi._payload_path(payload_root, row)
        st = os.lstat(path) if os.path.lexists(path) else None
        if st is None or not stat.S_ISREG(st.st_mode) or st.st_size != row["compressed_bytes"]:
            raise mi.Blocked(f"{row['member']}: payload missing or not of the allowlisted size")
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# stages


MODEL_UNITS = {"source": "millimetres", "source_reference": mi.BOP_FORMAT_URL, "output": "metres",
               "factor": mi.MM_TO_M, "applied": "exactly once"}
CLEANUP_OK = {"reaped": True, "returncode": 0, "threads_alive": [], "staging_fd_closed": True, "pipes_closed": True}
DECODE_NOTE = "lossy derivative: decoded pixels are distinct from the source PNG pixels"
SNAPSHOT_NOTE = "audited opens after the pre-publication closure pass, before this receipt was written"
VIDEO_PAYLOAD_CLOSURE = ("pre-publication: lstat identity of every RGB payload; post-publication: bounded re-seal "
                         "(size, SHA-256, CRC-32) of every RGB payload")
NATIVE_DECODER_NOTE = "OpenCV VideoCapture opens are not audit-visible; counted by construction"
MODEL_RECEIPT_KEYS = frozenset({"schema", "tool_version", "label", "pins_kind", "status", "flags", "trust", "scene",
                                "object_id", "association_note", "allowlist", "source", "units", "glb", "geometry",
                                "portable_targets", "reads", "budget", "input_budget", "sources", "runtime",
                                "preparation_scope", "closure", "ledger"})
VIDEO_RECEIPT_KEYS = frozenset({"schema", "tool_version", "label", "pins_kind", "status", "flags", "trust", "scene",
                                "allowlist", "video", "codec", "process", "encoder_cleanup", "clock", "image", "split",
                                "decode", "frames", "reads", "budget", "input_budget", "sources", "runtime",
                                "mobile_verified", "preparation_scope", "closure", "ledger"})
FRAME_KEYS = frozenset({"index", "frame_id", "split", "member", "png_sha256", "source_pixel_sha256",
                        "derivative_decoded_pixel_sha256", "decoded_identical_to_source"})


def _ffmpeg_candidates(mctx: mi.Context) -> list[Path]:
    """The two paths ``mi.ffmpeg_executable`` may hash (context root, then repository root)."""
    return [Path(os.path.abspath(mctx.root / mi.FFMPEG["path"])), Path(os.path.abspath(ROOT / mi.FFMPEG["path"]))]


def _reauthenticate(mctx: mi.Context, p: Mapping, counter: _Counter, doc: Mapping) -> tuple[dict, Path, Path, Path]:
    """Before ANY payload grant: re-read the pinned profile/K/association/scout/anchors from the paths the
    allowlist names, re-derive the allowlist core from the historical index and require byte equality.
    Payload rows used afterwards come from this re-derivation, never from caller-rehashed allowlist rows."""
    profile_path = _locate(mctx, doc["inputs"]["profile"]["path"], pf.PROFILE_NAME, "profile", inside=True)
    k_path = _locate(mctx, doc["inputs"]["camera_k"]["path"], pf.K_NAME, "camera-K record", inside=True)
    assoc_path = _locate(mctx, doc["inputs"]["association"]["path"], None, "association review", inside=False)
    a = _authenticate(mctx, p, counter, profile_path, k_path, assoc_path)
    if pf.dump_compact(_core(mctx, p, a, k_path, assoc_path)) != pf.dump_compact({k: doc[k] for k in CORE_KEYS}):
        raise mi.InputsError("media allowlist differs from a fresh re-derivation from its authenticated inputs")
    return a, profile_path, k_path, assoc_path


def _before(p: Mapping, code: Mapping, extra_files: Mapping[str, str]) -> dict:
    return {"files": dict(sorted({**_pinned_inputs(p), **extra_files}.items())), **code}


def _check_pre(files: Mapping, before: Mapping, extra: Mapping | None = None, expected_extra: Mapping | None = None) -> None:
    if {**_state(files), **(dict(extra) if extra else {})} != {**before, **(dict(expected_extra) if expected_extra else {})}:
        raise mi.InputsError("inputs/sources/runtime/outputs changed before publication")


def _reseal(paths: Sequence[Path], rows: Sequence[Mapping]) -> list:
    """Bounded actual-byte re-seal (size from lstat FIRST, then SHA-256/CRC-32 within the expected size)."""
    out = []
    for path, row in zip(paths, rows):
        try:
            mi.verified_payload_stream(path, row)
            out.append(True)
        except (mi.InputsError, OSError, zlib.error, ValueError) as exc:     # corrupt DEFLATE raises zlib.error
            out.append(f"refused: {type(exc).__name__}")
    return out


def build_allowlist(ctx: MediaContext, *, profile: Any, expected_profile_sha256: str, camera_k: Any,
                    expected_camera_k_sha256: str, association_review: Any, expected_association_review_sha256: str,
                    out: Any) -> dict:
    _assert_helper_assumptions()
    p = pins(ctx)
    _check_pins({"profile": expected_profile_sha256, "camera_k": expected_camera_k_sha256,
                 "association": expected_association_review_sha256},
                {"profile": p["profile_sha256"], "camera_k": p["camera_k_sha256"], "association": p["association_sha256"]})
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    profile_path = _locate(mctx, profile, pf.PROFILE_NAME, "profile", inside=True)
    k_path = _locate(mctx, camera_k, pf.K_NAME, "camera-K record", inside=True)
    assoc_path = _locate(mctx, association_review, None, "association review", inside=False)
    if os.lstat(assoc_path).st_size > ASSOCIATION_CAP:
        raise mi.InputsError(f"association review exceeds {ASSOCIATION_CAP} bytes")
    ledger = MediaLedger(mctx, out_dir, "media-allowlist")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened, _counting() as counter:
            counter.watch_prefix(out_dir, "stage_output")
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(ALLOWLIST_CAP, True)])                      # prospective, before any read
            budget = mi.workspace_budget(mctx, TRANSIENT_FACTOR * ALLOWLIST_CAP + mi.DIR_ENTRY_RESERVE)
            mi.input_budget(p["base_input_bytes"], ASSOCIATION_CAP, 0)
            a = _authenticate(mctx, p, counter, profile_path, k_path, assoc_path)
            mi.allow_payloads(a["payload_root"], ())                              # EMPTY payload allowlist
            core = _core(mctx, p, a, k_path, assoc_path)
            if opened:
                raise mi.GuardError("the metadata allowlist stage opened payload bytes")
            files = _input_files(mctx, a, profile_path, k_path, assoc_path)
            before = _before(p, code, {})
            _check_pre(files, before)                                             # pre-publication closure pass
            reads = {"snapshot": SNAPSHOT_NOTE, "audited_opens": counter.snapshot(), "payload_opens": 0,
                     "camera_opens": 0, "post_publication_planned": {"input_hashes": len(files), "output_hashes": 1}}
            doc = {**core, "sources": code["sources"], "runtime": code["runtime"], "budget": budget, "reads": reads,
                   "closure": {"before_files": before["files"], "pre_publication_matches": True,
                               "post_publication": "re-verified after publication; mismatch invalidates this output"}}
            validate_allowlist_doc(doc, p)
            data = pf.dump_compact(doc)
            if len(data) > ALLOWLIST_CAP:
                raise mi.Blocked(f"media allowlist is {len(data)} bytes, above {ALLOWLIST_CAP}")
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            published = publish(ledger, ALLOWLIST_NAME, data)
            _post_publication(before, _state(files), out_dir, {ALLOWLIST_NAME: (published["sha256"], ALLOWLIST_CAP)})
            if counter.reads("camera") or opened:
                raise mi.GuardError("camera or payload bytes were opened by the metadata stage")
            total_reads = counter.snapshot()
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "media-allowlist", exc)
        raise
    return {"allowlist": mi.repo_rel(mctx, out_dir / ALLOWLIST_NAME), "allowlist_sha256": published["sha256"],
            "object_id": core["model"]["object_id"], "label": p["label"], "payload_opens": 0,
            "audited_opens_total": total_reads, "exit": 0}


def prepare_model(ctx: MediaContext, *, allowlist: Any, expected_allowlist_sha256: str, out: Any) -> dict:
    _assert_helper_assumptions()
    p = pins(ctx)
    expected = mi.check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256")
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    ledger = MediaLedger(mctx, out_dir, "prepare-model")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened, _counting() as counter:
            counter.watch_prefix(out_dir, "stage_output")
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(mi.GLB_CAP, False), (MODEL_RECEIPT_CAP, True)])  # prospective, before any read
            budget = mi.workspace_budget(mctx, TRANSIENT_FACTOR * (mi.GLB_CAP + MODEL_RECEIPT_CAP) + mi.DIR_ENTRY_RESERVE)
            doc, allow_path, payload_root = _load_allowlist(mctx, p, counter, allowlist, expected)
            a, profile_path, k_path, assoc_path = _reauthenticate(mctx, p, counter, doc)   # grants still EMPTY
            if mi._norm(a["payload_root"]) != mi._norm(payload_root):
                raise mi.InputsError("allowlist payload root differs from the authenticated index location")
            inputs = mi.input_budget(p["base_input_bytes"], a["association_bytes"], 0)
            files = {"allowlist": (allow_path, ALLOWLIST_CAP), **_input_files(mctx, a, profile_path, k_path, assoc_path)}
            before = _before(p, code, {"allowlist": expected})
            if opened:
                raise mi.GuardError("payload bytes were opened before the CAD grant")
            row = a["model"]                                                         # re-derived, not caller-supplied
            (path,) = _payload_paths(payload_root, [row])
            mi.allow_payloads(payload_root, [path])                                   # exactly the reviewed CAD
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            parser = mi.AsciiPlyStream()                     # header refuses > 200 000 faces BEFORE body allocation
            verified = mi.verified_payload_stream(path, row, parser.feed)
            mesh = parser.finish()
            name = f"hb_obj_{row['object_id']:06d}.glb"
            glb, info = mi.build_glb(mesh, name=name[:-4], source_sha256=row["uncompressed_sha256"])
            if len(glb) > mi.GLB_CAP:
                raise mi.Blocked(f"GLB is {len(glb)} bytes, above the {mi.GLB_CAP} byte cap; nothing decimated")
            glb_meta = publish(ledger, name, glb, metadata=False)
            roundtrip = mi.verify_glb(out_dir / name, glb, mesh, info.pop("normals_out"))
            bounds_m = _glb_bounds(glb, mesh)
            faces = mesh["header"]["face_count"]
            del mesh
            stream_reads = counter.reads("payload")
            if stream_reads != 1 or set(opened) != {mi._norm(path)}:
                raise mi.GuardError("the selected CAD must be streamed exactly once and nothing else opened")
            _check_pre(files, before, {"glb": sealed_sha256(out_dir / name, mi.GLB_CAP)}, {"glb": glb_meta["sha256"]})
            receipt = {
                "schema": SCHEMA_MODEL, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
                "status": "prepared", "flags": dict(MEDIA_FLAGS), "trust": dict(TRUST), "scene": SCENE,
                "object_id": row["object_id"], "association_note": ASSOCIATION_NOTE,
                "allowlist": {"path": mi.repo_rel(mctx, allow_path), "sha256": expected},
                "source": _model_source(row, verified, parser.header), "units": dict(MODEL_UNITS),
                "glb": {"path": mi.repo_rel(mctx, out_dir / name), "sha256": glb_meta["sha256"],
                        "bytes": glb_meta["bytes"], "cap_bytes": mi.GLB_CAP, **info},
                "geometry": _model_geometry(parser.header, bounds_m, roundtrip),
                "portable_targets": _portable(faces),
                "reads": {"snapshot": SNAPSHOT_NOTE, "audited_opens": counter.snapshot(), "cad_payload_stream_opens": 1,
                          "post_publication_planned": {"cad_payload_reseal": 1, "input_hashes": len(files),
                                                       "output_hashes": 2}},
                "budget": budget, "input_budget": inputs, "sources": code["sources"], "runtime": code["runtime"],
                "preparation_scope": mi.repo_rel(mctx, mctx.scope),
                "closure": {"before_files": before["files"], "pre_publication_matches": True}}
            receipt["ledger"] = ledger.summary()
            data = mi.dump_json(receipt)
            if len(data) > MODEL_RECEIPT_CAP:
                raise mi.Blocked("model receipt exceeds its cap")
            published = publish(ledger, MODEL_RECEIPT_NAME, data)
            _post_publication({**before, "cad": [True]}, {**_state(files), "cad": _reseal([path], [row])}, out_dir,
                              {name: (glb_meta["sha256"], mi.GLB_CAP), MODEL_RECEIPT_NAME: (published["sha256"], MODEL_RECEIPT_CAP)})
            if counter.reads("camera"):
                raise mi.GuardError("the raw camera file was opened")
            total_reads = counter.snapshot()
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "prepare-model", exc)
        raise
    return {"model_receipt": mi.repo_rel(mctx, out_dir / MODEL_RECEIPT_NAME), "model_receipt_sha256": published["sha256"],
            "glb_sha256": glb_meta["sha256"], "triangles": faces, "label": p["label"],
            "audited_opens_total": total_reads, "exit": 0}


def _glb_bounds(glb: bytes, mesh: Mapping) -> dict:
    gltf, _ = mi.read_glb_document(glb)
    position = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    expected = mi.metre_positions(mesh["positions_mm"])
    bounds = {"min": [float(v) for v in expected.min(axis=0)], "max": [float(v) for v in expected.max(axis=0)]}
    if {"min": position.get("min"), "max": position.get("max")} != bounds:
        raise mi.InputsError("GLB position bounds differ from the converted source")
    return bounds


def _model_source(row: Mapping, verified: Mapping, header: Mapping) -> dict:
    return {"member": row["member"], "archive": row["archive"], "dataset_revision": mi.DATASET_REVISION,
            **{k: verified[k] for k in ("compressed_sha256", "uncompressed_sha256", "crc32", "compressed_bytes",
                                        "expanded_bytes")}, "header": dict(header)}


def _model_geometry(header: Mapping, bounds_m: Mapping, roundtrip: Mapping) -> dict:
    return {"vertices": header["vertex_count"], "triangles": header["face_count"], "execution_cap_triangles": mi.FACE_CAP,
            "bounds_m": bounds_m, "roundtrip": roundtrip, "decimated": False, "uv": "none fabricated", "seed_pose": None}


def _portable(faces: int) -> dict:
    return {"triangles_target": mi.PORTABLE_TRIANGLES, "triangles": "PASS" if faces <= mi.PORTABLE_TRIANGLES else "FAIL",
            "bytes_target": mi.PORTABLE_BYTES,
            "note": "reported separately from the execution cap; nothing is decimated"}


def prepare_video(ctx: MediaContext, *, allowlist: Any, expected_allowlist_sha256: str, out: Any) -> dict:
    from bench import workbench_geometry as wg
    _assert_helper_assumptions()
    p = pins(ctx)
    expected = mi.check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256")
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    settings = mi.codec_settings()
    ledger = MediaLedger(mctx, out_dir, "prepare-video")
    state: dict = {"dir": False, "staging": None, "encoder": None}
    try:
        with mi.guarded_stage(mctx) as opened, _counting() as counter:
            counter.watch_prefix(out_dir, "stage_output")
            for candidate in _ffmpeg_candidates(mctx):
                counter.watch(candidate, "ffmpeg_executable")                     # BEFORE the first hash
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(mi.VIDEO_CAP, False), (VIDEO_RECEIPT_CAP, True)], factor=1)   # rename: no alias
            budget = mi.workspace_budget(mctx, mi.VIDEO_CAP + TRANSIENT_FACTOR * VIDEO_RECEIPT_CAP + mi.DIR_ENTRY_RESERVE)
            doc, allow_path, payload_root = _load_allowlist(mctx, p, counter, allowlist, expected)
            a, profile_path, k_path, assoc_path = _reauthenticate(mctx, p, counter, doc)   # grants still EMPTY
            if mi._norm(a["payload_root"]) != mi._norm(payload_root):
                raise mi.InputsError("allowlist payload root differs from the authenticated index location")
            rows = a["rgb"]                                                          # re-derived, not caller-supplied
            if doc["frozen_split"] != FROZEN_SPLIT or len(rows) != FRAME_COUNT \
                    or [r["split"] for r in rows] != [_split(i) for i in range(FRAME_COUNT)]:
                raise mi.InputsError("frozen split / 120 frames differ")
            inputs = mi.input_budget(p["base_input_bytes"], a["association_bytes"], 0)
            files = {"allowlist": (allow_path, ALLOWLIST_CAP), **_input_files(mctx, a, profile_path, k_path, assoc_path)}
            before = _before(p, code, {"allowlist": expected})
            if opened:
                raise mi.GuardError("payload bytes were opened before the RGB grant")
            paths = _payload_paths(payload_root, rows)
            mi.ffmpeg_executable(mctx)                             # pinned bytes verified before any payload
            identity_before = _payload_identity(paths)
            mi.allow_payloads(payload_root, paths)
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            staging = out_dir / f".staging-{secrets.token_hex(6)}.mp4"
            state["staging"] = staging
            frames = []
            with mi.FfmpegPipeEncoder(mctx, _EncoderLedger(ledger), staging) as encoder:
                state["encoder"] = encoder
                for row, path in zip(rows, paths):
                    chunks: list[bytes] = []
                    verified = mi.verified_payload_stream(path, row, chunks.append)
                    rgb = mi.decode_png_rgb(b"".join(chunks))           # native 640x480; no crop/rescale/rotate
                    encoder.write_frame(rgb)
                    frames.append({"index": row["index"], "frame_id": row["frame_id"], "split": row["split"],
                                   "member": row["member"], "png_sha256": verified["uncompressed_sha256"],
                                   "source_pixel_sha256": wg.array_sha256(rgb)})
                    del chunks, rgb
                process = encoder.finish()
            if encoder.cleanup != CLEANUP_OK:
                raise mi.InputsError(f"FFmpeg cleanup incomplete: {encoder.cleanup}")
            size = os.lstat(staging).st_size
            if size > mi.VIDEO_CAP or size != process["stdout_bytes"] or process["exit_code"] != 0:
                raise mi.InputsError(f"derivative video size {size} differs from the drained stream or exceeds the cap")
            video_sha = sealed_sha256(staging, mi.VIDEO_CAP)
            decoded = mi.reopen_and_decode(staging)                # 120 sequential + EOF, exact s000010 seek
            for frame, digest in zip(frames, decoded["pixel_sha256"]):
                frame["derivative_decoded_pixel_sha256"] = digest
                frame["decoded_identical_to_source"] = digest == frame["source_pixel_sha256"]
            if not RENAME_EXCLUSIVE:
                ledger.check_sequence([(size, False)], factor=1)     # link alias of the staging file
            _commit(ledger, staging, VIDEO_NAME)
            state["staging"] = None
            payload_streams = counter.reads("payload")
            if payload_streams != FRAME_COUNT or set(opened) != {mi._norm(x) for x in paths}:
                raise mi.GuardError("each RGB payload must be streamed exactly once and nothing else opened")
            _check_pre(files, before, {"video": sealed_sha256(out_dir / VIDEO_NAME, mi.VIDEO_CAP),
                                       "payload_identity": _payload_identity(paths)},
                       {"video": video_sha, "payload_identity": identity_before})
            receipt = {
                "schema": SCHEMA_VIDEO, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
                "status": "prepared", "flags": dict(MEDIA_FLAGS), "trust": dict(TRUST), "scene": SCENE,
                "allowlist": {"path": mi.repo_rel(mctx, allow_path), "sha256": expected},
                "video": _video_block(mctx, out_dir / VIDEO_NAME, video_sha, size),
                "codec": settings, "process": process, "encoder_cleanup": encoder.cleanup,
                "clock": dict(CLOCK), "image": dict(IMAGE), "split": dict(FROZEN_SPLIT),
                "decode": _decode_block(decoded), "frames": frames,
                "reads": {"snapshot": SNAPSHOT_NOTE, "audited_opens": counter.snapshot(),
                          "rgb_payload_stream_opens": FRAME_COUNT, "native_decoder_opens": 2,
                          "native_decoder_note": NATIVE_DECODER_NOTE,
                          "post_publication_planned": {"rgb_payload_reseal": FRAME_COUNT, "input_hashes": len(files),
                                                       "output_hashes": 2}},
                "budget": budget, "input_budget": inputs, "sources": code["sources"],
                "runtime": {**code["runtime"], "ffmpeg": dict(mi.FFMPEG)}, "mobile_verified": False,
                "preparation_scope": mi.repo_rel(mctx, mctx.scope),
                "closure": {"before_files": before["files"], "pre_publication_matches": True,
                            "payload_closure": VIDEO_PAYLOAD_CLOSURE}}
            receipt["ledger"] = ledger.summary()
            data = mi.dump_json(receipt)
            if len(data) > VIDEO_RECEIPT_CAP:
                raise mi.Blocked("video receipt exceeds its cap")
            published = publish(ledger, VIDEO_RECEIPT_NAME, data)
            _post_publication({**before, "payloads": [True] * FRAME_COUNT},
                              {**_state(files), "payloads": _reseal(paths, rows)}, out_dir,
                              {VIDEO_NAME: (video_sha, mi.VIDEO_CAP), VIDEO_RECEIPT_NAME: (published["sha256"], VIDEO_RECEIPT_CAP)})
            if counter.reads("camera"):
                raise mi.GuardError("the raw camera file was opened")
            total_reads = counter.snapshot()
    except BaseException as exc:
        if state["dir"]:
            staging = state.get("staging")
            _fail_stage(ledger, "prepare-video", exc, {
                "staging_retained_bytes": os.lstat(staging).st_size if staging is not None and os.path.lexists(staging) else 0,
                "encoder_cleanup": getattr(state.get("encoder"), "cleanup", None)})
        raise
    return {"video_receipt": mi.repo_rel(mctx, out_dir / VIDEO_RECEIPT_NAME), "video_receipt_sha256": published["sha256"],
            "video_sha256": video_sha, "video_bytes": size, "label": p["label"], "audited_opens_total": total_reads,
            "exit": 0}


def _video_block(mctx: mi.Context, path: Path, sha: str, size: int) -> dict:
    return {"path": mi.repo_rel(mctx, path),
            "workspace_relative": Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(mctx.workspace))).as_posix(),
            "sha256": sha, "bytes": size, "cap_bytes": mi.VIDEO_CAP}


def _decode_block(decoded: Mapping) -> dict:
    return {"sequential_frames": decoded["frames"], "end_of_video_after": FRAME_COUNT,
            "dimensions": decoded["dimensions"], "backend": decoded["backend"],
            "frame_reader_seek_check": decoded["seek_check"], "pixel_identity_claimed": False, "note": DECODE_NOTE}


# --------------------------------------------------------------------------
# strict receipt reconstruction (check-prerequisites)


def _same(value: Any, expected: Any, what: str) -> None:
    """Exact equality including JSON types (1 vs 1.0 vs true are different)."""
    if pf.dump_compact(value) != pf.dump_compact(expected):
        raise mi.InputsError(f"{what} contradicts the actual reconstruction / fixed contract")


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _check_budget(value: Any, what: str, *, planned_bytes: int) -> None:
    keys = {"derived_bytes", "planned_bytes", "free_bytes", "derived_limit", "free_reserve", "method"}
    if (not isinstance(value, Mapping) or set(value) != keys
            or not all(_count(value[k]) for k in ("derived_bytes", "planned_bytes", "free_bytes"))
            or value["derived_limit"] != mi.DERIVED_LIMIT or value["free_reserve"] != mi.FREE_RESERVE
            or type(value["derived_limit"]) is not int or type(value["free_reserve"]) is not int
            or value["derived_bytes"] + value["planned_bytes"] > mi.DERIVED_LIMIT
            or value["free_bytes"] - value["planned_bytes"] < mi.FREE_RESERVE
            or value["planned_bytes"] != planned_bytes
            or value["method"] != "lstat sizes; links not followed; logical bytes (no hard-link deduction)"):
        raise mi.InputsError(f"{what}: budget record is malformed or above its limits")


def _check_ledger(value: Any, stage: str, *, files: Mapping, owned: list, staging: Mapping,
                  minimum_metadata_bytes: int, output_bytes: int) -> None:
    caps = {"substage": mi.STAGE_CAPS[STAGE_CAP_KEYS[stage]]["total"], "scope_total": mi.FINAL_OUTPUTS_CAP,
            "scope_metadata": mi.METADATA_AGGREGATE_CAP, "failure_reserve": FAILURE_RESERVE}
    usage = value.get("scope_usage_now") if isinstance(value, Mapping) else None
    if (not isinstance(value, Mapping) or set(value) != {"stage", "files", "owned", "staging_reported", "scope_usage_now",
                                                         "caps", "publication"}
            or value["stage"] != stage or value["caps"] != caps
            or value["publication"] != ("exclusive rename (Windows)" if RENAME_EXCLUSIVE else "link+unlink (2x counted)")
            or not isinstance(usage, Mapping) or set(usage) != {"total", "metadata", "files"}
            or not all(_count(usage[k]) for k in usage) or usage["total"] > mi.FINAL_OUTPUTS_CAP
            or usage["metadata"] > mi.METADATA_AGGREGATE_CAP
            or usage["metadata"] > usage["total"]
            or usage["metadata"] < minimum_metadata_bytes
            or usage["total"] < minimum_metadata_bytes + output_bytes
            or usage["files"] < 3 + len(owned)):
        raise mi.InputsError(f"{stage}: ledger record is malformed")
    _same(value["caps"], caps, f"{stage} ledger caps")
    _same(value["files"], files, f"{stage} ledger files")
    _same(value["owned"], owned, f"{stage} ledger ownership")
    _same(value["staging_reported"], staging, f"{stage} ledger staging accounting")


def _check_reads(value: Any, stage: str, expected_fixed: Mapping) -> None:
    opens = value.get("audited_opens") if isinstance(value, Mapping) else None
    if (not isinstance(value, Mapping) or set(value) != {"snapshot", "audited_opens", *expected_fixed}
            or value["snapshot"] != SNAPSHOT_NOTE or not isinstance(opens, Mapping)
            or not all(isinstance(k, str) and _count(v) for k, v in opens.items())
            or any(k.startswith("camera:") for k in opens)):
        raise mi.InputsError(f"{stage}: reads record is malformed or reports camera/GT opens")
    for key, expected in expected_fixed.items():
        _same(value[key], expected, f"{stage} reads.{key}")
    # These are counters at the successful producer's named snapshot, not
    # reconstructed historical free-space observations. A closed set also
    # refuses unreported camera/foreign reads and float/bool counter values.
    expected_opens = {f"{label}:read": 2 for label in (
        "allowlist", "profile", "camera_k", "association", "scout_plan", "scout_receipt",
        "anchor:acquisition_receipt", "anchor:integrity_record")}
    expected_opens.update({"anchor:resources_index:read": 3, "stage_output:write": 1})
    if stage == "prepare-model":
        expected_opens.update({"payload:read": 1, "stage_output:read": 4})
    elif stage == "prepare-video":
        expected_opens.update({"payload:read": FRAME_COUNT, "stage_output:read": 2,
                               "ffmpeg_executable:read": 2})
    else:
        raise mi.InputsError(f"unsupported receipt snapshot stage: {stage}")
    _same(opens, expected_opens, f"{stage} audited read/write snapshot")


def validate_model_receipt(doc: Any, *, p: Mapping, expected: Mapping) -> dict:
    """Every declaration must equal the actual reconstruction (``expected``) or the fixed contract."""
    if not isinstance(doc, dict) or set(doc) != MODEL_RECEIPT_KEYS:
        raise mi.InputsError("model receipt has missing or unknown fields")
    for key in ("schema", "tool_version", "label", "pins_kind", "status", "flags", "trust", "scene", "object_id",
                "association_note", "allowlist", "source", "units", "glb", "geometry", "portable_targets",
                "input_budget", "sources", "runtime", "preparation_scope", "closure"):
        _same(doc[key], expected[key], f"model receipt {key}")
    _check_reads(doc["reads"], "prepare-model", {"cad_payload_stream_opens": 1,
                                                  "post_publication_planned": expected["planned"]})
    _check_budget(doc["budget"], "model receipt", planned_bytes=
                  TRANSIENT_FACTOR * (mi.GLB_CAP + MODEL_RECEIPT_CAP) + mi.DIR_ENTRY_RESERVE)
    _check_ledger(doc["ledger"], "prepare-model", files={expected["glb_name"]: expected["glb"]["bytes"]},
                  owned=[expected["glb_name"]], staging={},
                  minimum_metadata_bytes=expected["minimum_metadata_bytes"], output_bytes=expected["glb"]["bytes"])
    return doc


def validate_video_receipt(doc: Any, *, p: Mapping, expected: Mapping) -> dict:
    if not isinstance(doc, dict) or set(doc) != VIDEO_RECEIPT_KEYS:
        raise mi.InputsError("video receipt has missing or unknown fields")
    for key in ("schema", "tool_version", "label", "pins_kind", "status", "flags", "trust", "scene", "allowlist", "video",
                "codec", "encoder_cleanup", "clock", "image", "split", "decode", "frames", "input_budget", "sources",
                "runtime", "mobile_verified", "preparation_scope", "closure"):
        _same(doc[key], expected[key], f"video receipt {key}")
    process = doc["process"]
    if (not isinstance(process, Mapping) or set(process) != {"exit_code", "stdout_bytes", "stderr_bytes", "stderr_tail"}
            or type(process["exit_code"]) is not int or process["exit_code"] != 0
            or type(process["stdout_bytes"]) is not int or process["stdout_bytes"] != expected["video"]["bytes"]
            or not _count(process["stderr_bytes"]) or not isinstance(process["stderr_tail"], str)
            or len(process["stderr_tail"]) > mi.STDERR_TAIL):
        raise mi.InputsError("video receipt process record contradicts a successful bounded run of the actual file")
    _check_reads(doc["reads"], "prepare-video", {"rgb_payload_stream_opens": FRAME_COUNT, "native_decoder_opens": 2,
                                                  "native_decoder_note": NATIVE_DECODER_NOTE,
                                                  "post_publication_planned": expected["planned"]})
    _check_budget(doc["budget"], "video receipt", planned_bytes=
                  mi.VIDEO_CAP + TRANSIENT_FACTOR * VIDEO_RECEIPT_CAP + mi.DIR_ENTRY_RESERVE)
    _check_ledger(doc["ledger"], "prepare-video", files={}, owned=[VIDEO_NAME],
                  staging={f"encoder-reported:{VIDEO_NAME}": expected["video"]["bytes"]},
                  minimum_metadata_bytes=expected["minimum_metadata_bytes"], output_bytes=expected["video"]["bytes"])
    return doc


def check_prerequisites(ctx: MediaContext, *, allowlist: Any, expected_allowlist_sha256: str, model_receipt: Any,
                        expected_model_receipt_sha256: str, video_receipt: Any, expected_video_receipt_sha256: str,
                        out: Any) -> dict:
    from bench import workbench_geometry as wg
    _assert_helper_assumptions()
    p = pins(ctx)
    expected = {"allowlist": mi.check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256"),
                "model_receipt": mi.check_sha(expected_model_receipt_sha256, "--expected-model-receipt-sha256"),
                "video_receipt": mi.check_sha(expected_video_receipt_sha256, "--expected-video-receipt-sha256")}
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    model_path = _locate(mctx, model_receipt, MODEL_RECEIPT_NAME, "model receipt", inside=True)
    video_receipt_path = _locate(mctx, video_receipt, VIDEO_RECEIPT_NAME, "video receipt", inside=True)
    ledger = MediaLedger(mctx, out_dir, "check-prerequisites")
    state = {"dir": False}
    try:
        with mi.guarded_stage(mctx) as opened, _counting() as counter:
            counter.watch_prefix(out_dir, "stage_output")
            code = {"sources": _sources(), "runtime": mi.runtime_identity()}
            ledger.check_sequence([(PREREQ_CAP, True)])
            budget = mi.workspace_budget(mctx, TRANSIENT_FACTOR * PREREQ_CAP + mi.DIR_ENTRY_RESERVE)
            doc, allow_path, payload_root = _load_allowlist(mctx, p, counter, allowlist, expected["allowlist"])
            allow_rel = mi.repo_rel(mctx, allow_path)
            a, profile_path, k_path, assoc_path = _reauthenticate(mctx, p, counter, doc)
            if mi._norm(a["payload_root"]) != mi._norm(payload_root) or opened:
                raise mi.InputsError("payload root differs or payload bytes opened during re-authentication")
            counter.watch(model_path, "model_receipt")
            counter.watch(video_receipt_path, "video_receipt")
            model = mi.strict_json(_read_sealed(model_path, MODEL_RECEIPT_CAP, expected["model_receipt"], "model receipt"),
                                   "model receipt")
            video = mi.strict_json(_read_sealed(video_receipt_path, VIDEO_RECEIPT_CAP, expected["video_receipt"],
                                                "video receipt"), "video receipt")
            oid = a["model"]["object_id"]
            glb_name = f"hb_obj_{oid:06d}.glb"
            glb_path = model_path.parent / glb_name
            video_path = video_receipt_path.parent / VIDEO_NAME
            counter.watch(glb_path, "glb_output")
            counter.watch(video_path, "video_output")
            glb_bytes = mi.read_bounded(glb_path, mi.GLB_CAP, "GLB")
            video_size = os.lstat(video_path).st_size
            video_sha = sealed_sha256(video_path, mi.VIDEO_CAP)
            # actual reconstruction: CAD -> identical GLB; RGB -> identical PNG/pixel hashes; video re-decoded
            mi.allow_payloads(payload_root, ())
            (cad_path,) = _payload_paths(payload_root, [a["model"]])
            rgb_paths = _payload_paths(payload_root, a["rgb"])
            mi.allow_payloads(payload_root, [cad_path, *rgb_paths])
            parser = mi.AsciiPlyStream()
            verified = mi.verified_payload_stream(cad_path, a["model"], parser.feed)
            mesh = parser.finish()
            rebuilt, info = mi.build_glb(mesh, name=glb_name[:-4], source_sha256=a["model"]["uncompressed_sha256"])
            if rebuilt != glb_bytes:
                raise mi.InputsError("GLB is not byte-identical to a rebuild from the reviewed CAD")
            roundtrip = mi.verify_glb(glb_path, glb_bytes, mesh, info.pop("normals_out"))
            bounds = _glb_bounds(glb_bytes, mesh)
            del mesh, rebuilt
            frames = []
            for row, path in zip(a["rgb"], rgb_paths):
                chunks: list[bytes] = []
                rgb_verified = mi.verified_payload_stream(path, row, chunks.append)
                rgb = mi.decode_png_rgb(b"".join(chunks))
                frames.append({"index": row["index"], "frame_id": row["frame_id"], "split": row["split"],
                               "member": row["member"], "png_sha256": rgb_verified["uncompressed_sha256"],
                               "source_pixel_sha256": wg.array_sha256(rgb)})
                del chunks, rgb
            decoded = mi.reopen_and_decode(video_path)
            for frame, digest in zip(frames, decoded["pixel_sha256"]):
                frame["derivative_decoded_pixel_sha256"] = digest
                frame["decoded_identical_to_source"] = digest == frame["source_pixel_sha256"]
            input_files = _input_files(mctx, a, profile_path, k_path, assoc_path)
            before_files = _before(p, code, {"allowlist": expected["allowlist"]})["files"]
            common = {"tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"], "status": "prepared",
                      "flags": MEDIA_FLAGS, "trust": TRUST, "scene": SCENE,
                      "allowlist": {"path": allow_rel, "sha256": expected["allowlist"]},
                      "input_budget": mi.input_budget(p["base_input_bytes"], a["association_bytes"], 0),
                      "sources": code["sources"], "preparation_scope": mi.repo_rel(mctx, mctx.scope),
                      # These three immutable metadata files necessarily existed
                      # before either producer snapshot. Failed siblings may add
                      # bytes/files; their exact historical totals are not rebuilt.
                      "minimum_metadata_bytes": sum(os.lstat(path).st_size for path in
                                                    (profile_path, k_path, allow_path))}
            validate_model_receipt(model, p=p, expected={
                **common, "schema": SCHEMA_MODEL, "object_id": oid, "association_note": ASSOCIATION_NOTE,
                "source": _model_source(a["model"], verified, parser.header), "units": MODEL_UNITS,
                "glb": {"path": mi.repo_rel(mctx, glb_path), "sha256": mi.sha256_bytes(glb_bytes), "bytes": len(glb_bytes),
                        "cap_bytes": mi.GLB_CAP, **info},
                "geometry": _model_geometry(parser.header, bounds, roundtrip),
                "portable_targets": _portable(parser.header["face_count"]), "runtime": code["runtime"],
                "closure": {"before_files": before_files, "pre_publication_matches": True}, "glb_name": glb_name,
                "planned": {"cad_payload_reseal": 1, "input_hashes": len(input_files) + 1, "output_hashes": 2}})
            validate_video_receipt(video, p=p, expected={
                **common, "schema": SCHEMA_VIDEO, "video": _video_block(mctx, video_path, video_sha, video_size),
                "codec": mi.codec_settings(), "encoder_cleanup": CLEANUP_OK, "clock": CLOCK, "image": IMAGE,
                "split": FROZEN_SPLIT, "decode": _decode_block(decoded), "frames": frames,
                "runtime": {**code["runtime"], "ffmpeg": dict(mi.FFMPEG)}, "mobile_verified": False,
                "closure": {"before_files": before_files, "pre_publication_matches": True,
                            "payload_closure": VIDEO_PAYLOAD_CLOSURE},
                "planned": {"rgb_payload_reseal": FRAME_COUNT, "input_hashes": len(input_files) + 1, "output_hashes": 2}})
            del glb_bytes
            if counter.reads("payload") != 1 + FRAME_COUNT or counter.reads("camera"):
                raise mi.GuardError("payload/camera reads differ from the planned reconstruction")
            files = {"allowlist": (allow_path, ALLOWLIST_CAP), "model_receipt": (model_path, MODEL_RECEIPT_CAP),
                     "video_receipt": (video_receipt_path, VIDEO_RECEIPT_CAP), **input_files}
            before = {"files": dict(sorted({**_pinned_inputs(p), **expected}.items())), **code}
            outputs = {"glb": model["glb"]["sha256"], "video": video_sha}

            def current_outputs() -> dict:
                result = {}
                for key, path, cap in (("glb", glb_path, mi.GLB_CAP), ("video", video_path, mi.VIDEO_CAP)):
                    try:
                        result[key] = sealed_sha256(path, cap)
                    except (mi.InputsError, OSError) as exc:
                        result[key] = f"refused: {type(exc).__name__}"
                return result

            _check_pre(files, before, {"outputs": current_outputs()}, {"outputs": outputs})
            receipt = {
                "schema": SCHEMA_PREREQ, "tool_version": TOOL_VERSION, "label": p["label"], "pins_kind": p["kind"],
                "status": "prepared_media_only", "flags": dict(MEDIA_FLAGS), "trust": dict(TRUST), "scene": SCENE,
                "inputs": {"allowlist": {"path": allow_rel, "sha256": expected["allowlist"]},
                           "model_receipt": {"path": mi.repo_rel(mctx, model_path), "sha256": expected["model_receipt"]},
                           "video_receipt": {"path": mi.repo_rel(mctx, video_receipt_path),
                                             "sha256": expected["video_receipt"]}, **doc["inputs"]},
                "outputs": {"glb": {"path": model["glb"]["path"], "sha256": model["glb"]["sha256"]},
                            "video": {"path": video["video"]["path"], "sha256": video_sha}},
                "verified": {"allowlist_core_rederived_from_pinned_inputs": True, "glb_rebuilt_byte_identical": True,
                             "model_receipt_declarations_reconstructed": True,
                             "video_receipt_declarations_reconstructed": True,
                             "rgb_png_and_source_pixel_hashes_rechecked": FRAME_COUNT,
                             "derivative_decode_rechecked": decoded["frames"], "seek_check": decoded["seek_check"]},
                "association": doc["association"], "camera_k": doc["camera_k"], "principal_point": dict(PRINCIPAL_POINT),
                "unresolved": list(UNRESOLVED),
                "not_published_here": ["ClipSpec", "SurfacePackage", "registration", "neural_admission"],
                "reads": {"snapshot": SNAPSHOT_NOTE, "audited_opens": counter.snapshot(), "native_decoder_opens": 2,
                          "native_decoder_note": NATIVE_DECODER_NOTE,
                          "post_publication_planned": {"input_and_receipt_hashes": len(files), "output_hashes": 2,
                                                       "cad_payload_reseal": 1, "rgb_payload_reseal": FRAME_COUNT}},
                "budget": budget, "sources": code["sources"], "runtime": code["runtime"],
                "preparation_scope": mi.repo_rel(mctx, mctx.scope),
                "closure": {"before_files": before["files"], "before_outputs": outputs, "pre_publication_matches": True}}
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            data = mi.dump_json(receipt)
            if len(data) > PREREQ_CAP:
                raise mi.Blocked("prerequisites receipt exceeds its cap")
            published = publish(ledger, PREREQ_NAME, data)
            # A final success receipt closes the original bytes too, not only
            # the rebuilt GLB/video and their metadata. Size/mtime are insufficient.
            resealed = _reseal([cad_path, *rgb_paths], [a["model"], *a["rgb"]])
            _post_publication({**before, "outputs": outputs, "payloads": [True] * (1 + FRAME_COUNT)},
                              {**_state(files), "outputs": current_outputs(), "payloads": resealed}, out_dir,
                              {PREREQ_NAME: (published["sha256"], PREREQ_CAP)})
            if counter.reads("payload") != 2 * (1 + FRAME_COUNT) or counter.reads("camera"):
                raise mi.GuardError("final prerequisite source/camera read accounting differs")
            total_reads = counter.snapshot()
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "check-prerequisites", exc)
        raise
    return {"prerequisites_receipt": mi.repo_rel(mctx, out_dir / PREREQ_NAME),
            "prerequisites_receipt_sha256": published["sha256"], "status": "prepared_media_only",
            "unresolved": list(UNRESOLVED), "label": p["label"], "audited_opens_total": total_reads, "exit": 0}


# --------------------------------------------------------------------------
# CLI (production pins only; no override)


def main(argv: Sequence[str] | None = None, ctx: MediaContext | None = None) -> int:
    base = ctx or MediaContext()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("build-allowlist")
    for flag in ("--profile", "--expected-profile-sha256", "--camera-k", "--expected-camera-k-sha256",
                 "--association-review", "--expected-association-review-sha256", "--out"):
        s.add_argument(flag, required=True)
    for stage in ("prepare-model", "prepare-video"):
        s = sub.add_parser(stage)
        for flag in ("--allowlist", "--expected-allowlist-sha256", "--out"):
            s.add_argument(flag, required=True)
    s = sub.add_parser("check-prerequisites")
    for flag in ("--allowlist", "--expected-allowlist-sha256", "--model-receipt", "--expected-model-receipt-sha256",
                 "--video-receipt", "--expected-video-receipt-sha256", "--out"):
        s.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "build-allowlist":
            result = build_allowlist(base, profile=args.profile, expected_profile_sha256=args.expected_profile_sha256,
                                     camera_k=args.camera_k, expected_camera_k_sha256=args.expected_camera_k_sha256,
                                     association_review=args.association_review,
                                     expected_association_review_sha256=args.expected_association_review_sha256,
                                     out=args.out)
        elif args.stage == "prepare-model":
            result = prepare_model(base, allowlist=args.allowlist, expected_allowlist_sha256=args.expected_allowlist_sha256,
                                   out=args.out)
        elif args.stage == "prepare-video":
            result = prepare_video(base, allowlist=args.allowlist, expected_allowlist_sha256=args.expected_allowlist_sha256,
                                   out=args.out)
        else:
            result = check_prerequisites(base, allowlist=args.allowlist,
                                         expected_allowlist_sha256=args.expected_allowlist_sha256,
                                         model_receipt=args.model_receipt,
                                         expected_model_receipt_sha256=args.expected_model_receipt_sha256,
                                         video_receipt=args.video_receipt,
                                         expected_video_receipt_sha256=args.expected_video_receipt_sha256, out=args.out)
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
                         "tools.workbench_hb_scene3_media ...)")
    raise SystemExit(main())
