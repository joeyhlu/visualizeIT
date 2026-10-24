"""Bounded cached-prefix SCOUT for HB val_primesense scene 000003 (inputs only, CPU only).

Why: scene 000002 opening frames 0/9 show two hand tools and an iron; the converted model 1 is a
bear, so a visual instance binding was REFUSED there. Previewing the six allowlisted CAD
candidates (1/3/4/5/6/9) found no match, and 2/7/8/10 exceed the 200 k execution cap and stay
unpreviewed. Nothing here guesses object ids, relaxes caps, decimates, extracts poses or judges
model quality. The frozen six-case suite and head 27e06a51... stay unchanged.

Scene 000003 was chosen PROSPECTIVELY as the first unreviewed scene in ascending order and the
smallest cached encoded footprint (120 RGB frames, 35 632 589 bytes). This is a new source scope,
not a retry of the closed scene-000002 metric tools.

Stages (each wholly inside the shared audit guard, which starts with an EMPTY payload allowlist;
the three historical anchors are exact READ-ONLY exceptions):

``plan``            Metadata only. Verifies the anchor bytes against their pinned SHA-256 BEFORE
                    parsing, validates exactly 120 RGB rows ``val_primesense/000003/rgb/NNNNNN.png``
                    in ``hb_val_primesense.zip`` (encoded total 35 632 589 bytes), stats the
                    payloads (ZERO opens) and freezes opening 0..9 / seed 9 / future scoring
                    10..119. Camera K stays PENDING. No masks, poses or object id.
``preview-opening`` Only after a parent-accepted plan (expected SHA) that is fully rebuilt and
                    compared. ONLY frame indices 0 and 9 are allowed; anything else (above all 10)
                    is refused before any payload is opened. Writes ONE labelled contact sheet
                    (<= 1024 px wide) plus a receipt.

Trust is historical local integrity, NOT upstream authentication. Exit codes: 0 completed;
1 failure; 2 usage; 3 blocked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (pure helpers only; never mutated)

TOOL_VERSION = "1"
TOOL_ID = "visualizeit.workbench_hb_prefix_scout"
SCHEMA_PLAN = "workbench_hb_prefix_scout_plan_v1"
SCHEMA_PREVIEW = "workbench_hb_prefix_scout_preview_v1"
SCHEMA_FAILURE = "workbench_hb_prefix_scout_failure_v1"
INVALIDATED_PREFIX = "invalidated-"
SCENE = "000003"
FRAME_COUNT = 120
OPENING = tuple(range(0, 10))
SEED_INDEX = 9
SCORING = tuple(range(10, 120))
PREVIEW_INDICES = (0, 9)
SCENE_ENCODED_BYTES = 35_632_589          # prospective selection pin (cached encoded footprint)
RGB_PREFIX = f"val_primesense/{SCENE}/rgb/"
CALIBRATION_MEMBER = f"val_primesense/{SCENE}/scene_camera.json"
PLAN_NAME, SHEET_NAME, RECEIPT_NAME = "scout-plan.json", "opening-contact-sheet.png", "preview-receipt.json"

SCOUT_TOTAL_CAP = 2 << 20                 # plan + preview + retained failures/siblings in the scout scope
SCOUT_METADATA_CAP = 512 << 10
FAILURE_RESERVE = 16 << 10
PLAN_CAP = 192 << 10
RECEIPT_CAP = 128 << 10
SHEET_CAP = 768 << 10                     # so the 2x transient peak of the sheet fits the 2 MiB scope
FUTURE_PREPARATION_RESERVE = 12 << 20     # counted in the new-input arithmetic, never consumed here
PLAN_ALLOWANCE = 2 * PLAN_CAP + mi.DIR_ENTRY_RESERVE
PREVIEW_ALLOWANCE = 2 * (SHEET_CAP + RECEIPT_CAP) + mi.DIR_ENTRY_RESERVE
SHEET_PNG_COMPRESSION = 9                 # lossless; maximum zlib effort

SHEET_MAX_WIDTH = 1024
THUMB_SCALE = 0.75                        # 640x480 -> 480x360 exactly
THUMB = (480, 360)
MARGIN = 16
LABEL_H = 30

FROZEN_SPLIT = {"opening": list(OPENING), "seed_index": SEED_INDEX, "scoring_future": list(SCORING),
                "fixed_before_viewing": True, "preview_authorized_indices": list(PREVIEW_INDICES)}
CLOCK = {"kind": "FRAME_INDEX", "physical_fps": None, "physical_timestamps": None,
         "note": "physical time and FPS are unknown; no ten-real-seconds claim"}
FLAGS = {"inputs_only": True, "admitted": False, "registered": False, "neural_executed": False,
         "gpu_used": False, "gt_read": False, "masks": None, "poses": None, "object_id": None,
         "object_binding": "none", "quality_claim": False, "mobile_verified": False,
         "candidate_only": True, "head_sha256": mi.FLAGS["head_sha256"]}
TRUST = {"trust_level": "historical_local_integrity", "upstream_authenticated": False,
         "basis": "the three historical HB anchors pinned by SHA-256 and verified before parsing"}
CALIBRATION_STATUS = "PENDING_scene_specific_K_only_verification"
CALIBRATION_NOTE = ("not opened; an object-labelled calibration registry row is not proof of this scene's K")
SELECTION_NOTE = ("scene 000003 chosen prospectively: first unreviewed scene in ascending order and the "
                  "smallest cached encoded footprint (120 RGB frames, 35632589 bytes)")
EXCLUDED_NOTE = "GT, depth, masks, models and every other scene/member are excluded and never opened or copied"


@dataclass(frozen=True)
class SyntheticPins:
    """TEST-ONLY replacement for the historical pins (manufactured fixtures). The CLI never sets
    it, and every output records ``pins.kind = "synthetic_test_only"``."""
    anchors: Mapping[str, str]
    integrity_members: int
    rgb_encoded_bytes: int
    rgb_expanded_bytes: int | None = None


@dataclass(frozen=True)
class ScoutContext:
    """Injectable roots. ``scout_root`` is the scout's own fresh scope inside ``derived``."""
    root: Path = ROOT
    workspace: Path = ROOT / ".cache/video-workbench"
    scout_root: Path | None = None
    synthetic: SyntheticPins | None = None

    def mi(self) -> mi.Context:
        scope = self.scout_root if self.scout_root is not None else self.workspace / "derived" / f"hb-prefix-scout-{SCENE}"
        return mi.Context(root=self.root, workspace=self.workspace, preparation=Path(os.path.abspath(scope)))


def pins(ctx: ScoutContext) -> dict:
    if ctx.synthetic is None:
        return {"kind": "historical", "anchors": dict(mi.ANCHORS), "integrity_members": mi.INTEGRITY_MEMBERS,
                "rgb_encoded_bytes": SCENE_ENCODED_BYTES, "rgb_expanded_bytes": None}
    s = ctx.synthetic
    if set(s.anchors) != set(mi.ANCHORS) or not all(mi._is_sha(v) for v in s.anchors.values()) \
            or type(s.integrity_members) is not int or type(s.rgb_encoded_bytes) is not int:
        raise mi.UsageError("synthetic test pins are malformed")
    return {"kind": "synthetic_test_only", "anchors": dict(s.anchors), "integrity_members": s.integrity_members,
            "rgb_encoded_bytes": s.rgb_encoded_bytes, "rgb_expanded_bytes": s.rgb_expanded_bytes}


class ScoutLedger:
    """Aggregate accounting of the WHOLE scout scope (retained failures/siblings included) before every
    encoded disk write: <= 2 MiB total and <= 512 KiB metadata; plus the free-disk reserve.
    Duck-compatible with ``mi.publish_bytes``.

    TRANSIENT peak: ``mi.publish_bytes`` holds an owned temp AND its hard-linked published name until
    the temp is removed, and ``scope_usage`` counts both names. Every check therefore requires
    ``usage + 2 * nbytes`` to fit (normal and failure publication alike), so the cap holds at the peak,
    not only after the temp is gone. Normal writes also keep ``2 * FAILURE_RESERVE`` free, which is the
    transient peak of the largest failure record ``_scout_failure`` will ever publish."""

    TRANSIENT_FACTOR = 2

    def __init__(self, mctx: mi.Context, out_dir: Path, stage: str):
        self.ctx, self.dir, self.stage, self.scope = mctx, out_dir, stage, mctx.scope
        self.files: dict[str, int] = {}
        self.owned: dict[str, tuple[int, int]] = {}

    def check(self, nbytes: int, *, metadata: bool, failure: bool = False) -> dict:
        return self.check_sequence([(nbytes, metadata)], failure=failure)

    def check_sequence(self, steps: Sequence[tuple[int, bool]], *, failure: bool = False) -> dict:
        """Prospective check of sequential publications: at step k the scope holds the current usage
        plus every earlier final output plus 2x the bytes being published."""
        if not steps or any(type(n) is not int or n < 0 for n, _ in steps):
            raise mi.InputsError("invalid write size")
        usage = mi.scope_usage(self.scope)
        reserve = 0 if failure else self.TRANSIENT_FACTOR * FAILURE_RESERVE
        done_total = done_meta = 0
        for nbytes, metadata in steps:
            peak_total = usage["total"] + done_total + self.TRANSIENT_FACTOR * nbytes
            if peak_total > SCOUT_TOTAL_CAP - reserve:
                raise mi.Blocked(f"scout scope holds {usage['total']} bytes; publishing {nbytes} more would peak at "
                                 f"{peak_total} (temp + published name), above {SCOUT_TOTAL_CAP - reserve}")
            if metadata:
                peak_meta = usage["metadata"] + done_meta + self.TRANSIENT_FACTOR * nbytes
                if peak_meta > SCOUT_METADATA_CAP - reserve:
                    raise mi.Blocked(f"scout metadata would peak at {peak_meta}, above {SCOUT_METADATA_CAP - reserve}")
                done_meta += nbytes
            done_total += nbytes
        largest = max(n for n, _ in steps)
        if mi.shutil.disk_usage(self.ctx.workspace).free - self.TRANSIENT_FACTOR * largest - done_total < mi.FREE_RESERVE:
            raise mi.Blocked("the 1 GiB free-disk reserve would be violated")
        return usage

    def record(self, name: str, nbytes: int, *, metadata: bool) -> None:
        """Called by ``mi.publish_bytes`` right after ITS ``os.link`` created the published name and
        before the final hash check: the name now exists because of this stage, so ownership (with the
        file identity at creation) is established here. A name that already existed (a foreign output)
        makes the publisher refuse before linking, so it is never recorded or touched."""
        self.files[name] = self.files.get(name, 0) + nbytes
        try:
            st = os.lstat(self.dir / name)
            self.owned[name] = (st.st_dev, st.st_ino)
        except OSError:
            self.owned[name] = (-1, -1)          # created by us but not identifiable; see owned_file()

    def owned_file(self, name: str) -> bool:
        """True only for a name this stage created that still is the very file it created."""
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
                "caps": {"total": SCOUT_TOTAL_CAP, "metadata": SCOUT_METADATA_CAP, "failure_reserve": FAILURE_RESERVE}}


# --------------------------------------------------------------------------
# pure checks


def validate_anchor_documents(receipt: Any, index: Any, integrity: Any, p: Mapping) -> None:
    """Receipt / index / integrity record must bind each other and the pins (mirrors the filter rules)."""
    if not all(isinstance(x, Mapping) for x in (receipt, index, integrity)):
        raise mi.InputsError("anchors must be JSON objects")
    selection = receipt.get("selectionDigest")
    if (receipt.get("status") != "complete" or receipt.get("datasetRevision") != mi.DATASET_REVISION
            or receipt.get("indexSHA256") != p["anchors"]["resources_index"] or not mi._is_sha(selection)):
        raise mi.InputsError("acquisition receipt is not the complete pinned acquisition of this revision/index")
    if (integrity.get("status") != "verified" or integrity.get("datasetRevision") != mi.DATASET_REVISION
            or integrity.get("indexSHA256") != p["anchors"]["resources_index"]
            or integrity.get("selectionDigest") != selection or integrity.get("verifiedMembers") != p["integrity_members"]):
        raise mi.InputsError("integrity record does not bind the pinned index/revision/selection")
    if (index.get("schemaVersion") != 1 or index.get("corpus") != mi.CORPUS or index.get("datasetRevision") != mi.DATASET_REVISION
            or index.get("storageKind") != "original_zip_members" or index.get("selectionDigest") != selection
            or not isinstance(index.get("resources"), Mapping) or len(index["resources"]) != p["integrity_members"]):
        raise mi.InputsError("resources index identity differs from the receipt/integrity bindings")
    if index.get("acquisitionSourceHashes") != receipt.get("sourceHashes"):
        raise mi.InputsError("acquisition source seals differ between the index and the receipt")


def _split(i: int) -> str:
    return "opening_seed" if i == SEED_INDEX else "opening" if i in OPENING else "scoring_future"


def build_plan_core(receipt: Any, index: Any, integrity: Any, p: Mapping, *, anchors: Mapping,
                    payload_root: str) -> dict:
    """Deterministic metadata core (no file access). Raises on wrong scene/member/row/flags/totals."""
    validate_anchor_documents(receipt, index, integrity, p)
    resources = index["resources"]
    expected = [f"{RGB_PREFIX}{i:06d}.png" for i in range(FRAME_COUNT)]
    present = sorted(m for m in resources if isinstance(m, str) and m.startswith(RGB_PREFIX))
    if present != expected:
        raise mi.InputsError(f"scene {SCENE} RGB members are not exactly {expected[0]}..{expected[-1]}")
    rgb = []
    for i, member in enumerate(expected):
        row = mi.validate_index_row(resources[member], member, mi.RGB_ARCHIVE, "rgb", mi.MAX_RGB_EXPANDED)
        rgb.append({"index": i, "frame_id": f"s{i:06d}", "split": _split(i), **row})
    totals = {"encoded_bytes": sum(r["compressed_bytes"] for r in rgb),
              "expanded_bytes": sum(r["expanded_bytes"] for r in rgb)}
    if totals["encoded_bytes"] != p["rgb_encoded_bytes"] or (
            p["rgb_expanded_bytes"] is not None and totals["expanded_bytes"] != p["rgb_expanded_bytes"]):
        raise mi.InputsError(f"scene {SCENE} RGB totals {totals} differ from the pinned footprint")
    cal_row = resources.get(CALIBRATION_MEMBER)
    cal_sha = cal_row.get("uncompressedSHA256") if isinstance(cal_row, Mapping) else None
    calibration = {"member": CALIBRATION_MEMBER, "listed_in_index": isinstance(cal_row, Mapping),
                   "uncompressed_sha256": cal_sha if mi._is_sha(cal_sha) else None,
                   "status": CALIBRATION_STATUS, "note": CALIBRATION_NOTE}
    return {"schema": SCHEMA_PLAN, "tool_version": TOOL_VERSION, "scene": SCENE, "flags": dict(FLAGS),
            "trust": dict(TRUST), "pins": {"kind": p["kind"], "rgb_encoded_bytes": p["rgb_encoded_bytes"],
                                            "integrity_members": p["integrity_members"]},
            "anchors": {k: dict(v) for k, v in anchors.items()},
            "source": {"corpus": mi.CORPUS, "dataset_revision": mi.DATASET_REVISION, "scene": SCENE,
                       "rgb_archive": mi.RGB_ARCHIVE, "selection_digest": receipt["selectionDigest"],
                       "payload_root": payload_root},
            "source_seals": {"acquisition_source_hashes": receipt.get("sourceHashes")},
            "selection": SELECTION_NOTE, "frozen_split": dict(FROZEN_SPLIT), "clock": dict(CLOCK),
            "calibration": calibration, "rgb": rgb, "rgb_totals": totals, "excluded": EXCLUDED_NOTE}


CORE_KEYS = frozenset({"schema", "tool_version", "scene", "flags", "trust", "pins", "anchors", "source", "source_seals",
                       "selection", "frozen_split", "clock", "calibration", "rgb", "rgb_totals", "excluded"})
PLAN_KEYS = CORE_KEYS | {"sources", "runtime", "existing_input_bytes", "input_budget", "budget", "payload_stats",
                         "payload_opens", "scout_scope", "closure"}


def validate_plan_doc(doc: Any, p: Mapping) -> dict:
    """Pure structural re-validation of an accepted plan (before any payload permission)."""
    if not isinstance(doc, dict) or set(doc) != PLAN_KEYS:
        raise mi.InputsError("plan has missing or unknown fields")
    if (doc["schema"] != SCHEMA_PLAN or doc["tool_version"] != TOOL_VERSION or doc["scene"] != SCENE
            or doc["flags"] != FLAGS or doc["trust"] != TRUST or doc["frozen_split"] != FROZEN_SPLIT
            or doc["clock"] != CLOCK or doc["excluded"] != EXCLUDED_NOTE or doc["selection"] != SELECTION_NOTE
            or doc["payload_opens"] != 0 or type(doc["payload_opens"]) is not int):
        raise mi.InputsError("plan schema/flags/split/clock/notes differ")
    if doc["pins"] != {"kind": p["kind"], "rgb_encoded_bytes": p["rgb_encoded_bytes"],
                       "integrity_members": p["integrity_members"]}:
        raise mi.InputsError("plan was produced under different pins")
    if not isinstance(doc["anchors"], Mapping) or set(doc["anchors"]) != set(mi.ANCHORS) or any(
            not isinstance(a, Mapping) or a.get("sha256") != p["anchors"][k] or a.get("verified_before_parsing") is not True
            for k, a in doc["anchors"].items()):
        raise mi.InputsError("plan anchors are not the pinned, verified-before-parsing records")
    cal = doc["calibration"]
    if not isinstance(cal, Mapping) or cal.get("status") != CALIBRATION_STATUS or cal.get("member") != CALIBRATION_MEMBER:
        raise mi.InputsError("plan calibration status must stay PENDING")
    rgb = doc["rgb"]
    if not isinstance(rgb, list) or len(rgb) != FRAME_COUNT:
        raise mi.InputsError("plan must hold exactly 120 RGB rows")
    for i, row in enumerate(rgb):
        member = f"{RGB_PREFIX}{i:06d}.png"
        if not isinstance(row, Mapping) or row.get("index") != i or type(row.get("index")) is not int \
                or row.get("frame_id") != f"s{i:06d}" or row.get("split") != _split(i) or row.get("member") != member:
            raise mi.InputsError(f"plan RGB row {i} identity differs")
        stripped = {k: v for k, v in row.items() if k not in ("index", "frame_id", "split")}
        index_form = {"datasetRevision": stripped.get("dataset_revision"), "archive": stripped.get("archive"),
                      "sourceMember": member, "sourceMethod": stripped.get("method"), "sourceFlags": stripped.get("flags"),
                      "sourceOffset": stripped.get("source_offset"), "sourceCompressedBytes": stripped.get("compressed_bytes"),
                      "expandedBytes": stripped.get("expanded_bytes"), "crc32": stripped.get("crc32"),
                      "method": stripped.get("method"), "compressedBytes": stripped.get("compressed_bytes"),
                      "allowedRoles": stripped.get("roles"),
                      "location": {"kind": stripped.get("location_kind"), "path": stripped.get("payload_path")},
                      "compressedSHA256": stripped.get("compressed_sha256"),
                      "uncompressedSHA256": stripped.get("uncompressed_sha256")}
        rebuilt = mi.validate_index_row(index_form, member, mi.RGB_ARCHIVE, "rgb", mi.MAX_RGB_EXPANDED)
        if set(stripped) != set(rebuilt) or any(type(rebuilt[k]) is not type(stripped[k]) or rebuilt[k] != stripped[k]
                                                for k in rebuilt):
            raise mi.InputsError(f"plan RGB row {i} conflicts with its revalidation")
    totals = {"encoded_bytes": sum(r["compressed_bytes"] for r in rgb), "expanded_bytes": sum(r["expanded_bytes"] for r in rgb)}
    if doc["rgb_totals"] != totals or totals["encoded_bytes"] != p["rgb_encoded_bytes"]:
        raise mi.InputsError("plan RGB totals differ from its rows or the pinned footprint")
    if type(doc["existing_input_bytes"]) is not int or doc["existing_input_bytes"] < 0:
        raise mi.InputsError("plan existing_input_bytes malformed")
    return doc


def check_preview_frames(frames: Any) -> tuple[int, ...]:
    """EXACTLY the prospectively authorized opening pair (0, 9), in any order, each once. A subset,
    a duplicate, a non-integer and above all the first scored frame 10 are refused before any path
    is touched or payload opened."""
    if not isinstance(frames, (list, tuple)) or any(type(f) is not int for f in frames):
        raise mi.UsageError("preview frames must be a list of integer indices")
    if len(frames) != len(PREVIEW_INDICES) or sorted(frames) != list(PREVIEW_INDICES):
        raise mi.UsageError(f"preview is authorized ONLY for exactly the opening indices {list(PREVIEW_INDICES)}; "
                            f"refused {list(frames)}")
    return PREVIEW_INDICES


def check_existing_input_bytes(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise mi.UsageError("existing_input_bytes must be a non-negative integer (booleans are invalid)")
    return value


def scout_input_budget(existing: int, p: Mapping) -> dict:
    return mi.input_budget(existing, p["rgb_encoded_bytes"], SCOUT_TOTAL_CAP + FUTURE_PREPARATION_RESERVE)


# --------------------------------------------------------------------------
# contact sheet (pure rendering)


def wrap_text(text: str, width_px: int, font, scale: float, thickness: int) -> list[str]:
    from bench.vision import cv2
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if cv2.getTextSize(trial, font, scale, thickness)[0][0] <= width_px:
            current = trial
            continue
        if not current:
            raise mi.InputsError(f"banner word {word!r} cannot fit the sheet width")
        lines.append(current)
        current = word
        if cv2.getTextSize(current, font, scale, thickness)[0][0] > width_px:
            raise mi.InputsError(f"banner word {word!r} cannot fit the sheet width")
    if current:
        lines.append(current)
    return lines


BANNER = (f"HB val_primesense scene {SCENE} - opening-prefix scout, CANDIDATE ONLY.",
          "Frames shown: s000000 (index 0) and s000009 (index 9, planned seed); opening 0..9 only; no future frames.",
          "Clock FRAME_INDEX; physical time and FPS unknown; no ten-real-seconds claim.",
          "Camera K: PENDING scene-specific verification.",
          "No object binding, no admission, no quality claim. Native 640x480 shown at 0.75 scale.")


def build_contact_sheet(frames: Sequence[tuple[int, np.ndarray]]) -> tuple[np.ndarray, dict]:
    """One labelled sheet <= 1024 px wide (RGB) for EXACTLY frames 0 and 9 in that order (the banner
    names both, so any other set is refused). Every banner qualifier is wrapped to fit (never clipped)."""
    from bench.vision import cv2
    if [index for index, _ in frames] != list(PREVIEW_INDICES):
        raise mi.InputsError(f"the contact sheet shows exactly frames {list(PREVIEW_INDICES)}; got "
                             f"{[index for index, _ in frames]}")
    font, scale, thick, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 22
    width = len(PREVIEW_INDICES) * THUMB[0] + (len(PREVIEW_INDICES) + 1) * MARGIN
    if width > SHEET_MAX_WIDTH:
        raise mi.InputsError("contact sheet would exceed 1024 px")
    lines = [line for text in BANNER for line in wrap_text(text, width - 2 * MARGIN, font, scale, thick)]
    banner_h = MARGIN + len(lines) * line_h + MARGIN // 2
    height = banner_h + THUMB[1] + LABEL_H + MARGIN
    sheet = np.full((height, width, 3), 32, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(sheet, line, (MARGIN, MARGIN + (k + 1) * line_h - 6), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
    mapping = []
    for slot, (index, rgb) in enumerate(frames):
        if rgb.shape != (mi.IMAGE_SIZE[1], mi.IMAGE_SIZE[0], 3) or rgb.dtype != np.uint8:
            raise mi.InputsError("preview frame is not native 640x480 RGB8")
        thumb = cv2.resize(rgb, THUMB, interpolation=cv2.INTER_AREA)
        x0, y0 = MARGIN + slot * (THUMB[0] + MARGIN), banner_h
        sheet[y0:y0 + THUMB[1], x0:x0 + THUMB[0]] = thumb
        label = f"s{index:06d} | index {index} | {'opening, planned seed' if index == SEED_INDEX else 'opening'}"
        if cv2.getTextSize(label, font, scale, thick)[0][0] > THUMB[0]:
            raise mi.InputsError("thumbnail label does not fit")
        cv2.putText(sheet, label, (x0, y0 + THUMB[1] + LABEL_H - 9), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
        mapping.append({"index": index, "frame_id": f"s{index:06d}", "native_size": list(mi.IMAGE_SIZE),
                        "thumbnail_size": list(THUMB), "scale": THUMB_SCALE, "offset_xy": [x0, y0],
                        "pixel_centre_mapping": "sheet = offset + (native + 0.5) * scale - 0.5 (per axis)",
                        "resample": "cv2.INTER_AREA"})
    return sheet, {"width": width, "height": height, "banner_lines": lines, "mapping": mapping}


# --------------------------------------------------------------------------
# stages


def _anchor_paths(ctx: ScoutContext, mctx: mi.Context, receipt_path: str, index_path: str, integrity_path: str) -> dict:
    paths = {"acquisition_receipt": mi.contained_input(mctx, receipt_path, "acquisition receipt", allow_denied=True),
             "resources_index": mi.contained_input(mctx, index_path, "resources index", allow_denied=True),
             "integrity_record": mi.contained_input(mctx, integrity_path, "integrity record", allow_denied=True)}
    for key, path in paths.items():
        mi.outside_scope_input(mctx, path, key)
    return paths


def _read_anchors(paths: Mapping[str, Path], p: Mapping) -> dict:
    raws = {}
    for key, path in paths.items():
        raw = mi.read_bounded(path, mi.MAX_ANCHOR_BYTES[key], key)
        if mi.sha256_bytes(raw) != p["anchors"][key]:                    # BEFORE any parsing
            raise mi.InputsError(f"{key}: SHA-256 differs from the pinned historical anchor")
        raws[key] = raw
    return raws


def _core_from_anchors(mctx: mi.Context, paths: Mapping[str, Path], raws: Mapping[str, bytes], p: Mapping) -> dict:
    anchors = {k: {"path": mi.repo_rel(mctx, paths[k]), "sha256": p["anchors"][k], "bytes": len(raws[k]),
                   "verified_before_parsing": True} for k in paths}
    payload_root = mi.repo_rel(mctx, paths["resources_index"].parent / "payloads")
    return build_plan_core(mi.strict_json(raws["acquisition_receipt"], "acquisition receipt"),
                           mi.strict_json(raws["resources_index"], "resources index"),
                           mi.strict_json(raws["integrity_record"], "integrity record"), p,
                           anchors=anchors, payload_root=payload_root)


def _payload_stats(mctx: mi.Context, paths: Mapping[str, Path], rows: Sequence[Mapping]) -> dict:
    base = paths["resources_index"].parent
    stats = {}
    for row in rows:
        path = mi.safe_member_path(base, row["payload_path"], row["member"])
        st = os.lstat(path) if os.path.lexists(path) else None                 # stat only; never opened
        if st is None or not mi.stat.S_ISREG(st.st_mode) or st.st_size != row["compressed_bytes"]:
            raise mi.Blocked(f"{row['member']}: cached payload missing or not of the planned size")
        stats[row["payload_path"]] = st.st_size
    return stats


def _sources() -> dict:
    return {**mi.source_closure(), Path(__file__).resolve().relative_to(ROOT).as_posix(): mi.sha256_file(Path(__file__))}


POST_PUBLICATION_NOTE = ("re-verified AFTER publication (anchors, sources, runtime, plan and the current bytes and "
                         "names of this stage's outputs); a mismatch renames the authoritative output to "
                         f"'{INVALIDATED_PREFIX}<name>', writes failure.json and fails the stage")


def _scout_failure(ledger: ScoutLedger, stage: str, exc: BaseException, extra: Mapping | None = None) -> None:
    """Bounded failure record (never truncated mid-JSON, never above FAILURE_RESERVE, published with
    the 2x transient check and no reserve). If it cannot be published it goes to stderr explicitly."""
    retained = {}
    if ledger.dir.is_dir():
        for entry in sorted(ledger.dir.iterdir()):
            try:
                retained[entry.name] = os.lstat(entry).st_size
            except OSError:
                pass
    try:
        ledger_summary = ledger.summary()
    except Exception as summary_exc:  # noqa: BLE001 - never mask the original failure
        ledger_summary = {"error": f"{type(summary_exc).__name__}: {str(summary_exc)[:200]}"}
    body = {"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage, "status": "failed",
            "reason": f"{type(exc).__name__}: {str(exc)[:1500]}", "completed_receipt_published": False,
            "retained_files": retained, "ledger": ledger_summary, **(dict(extra) if extra else {})}
    data = mi.dump_json(body)
    if len(data) > FAILURE_RESERVE:
        body.update(reason=body["reason"][:300], ledger={"omitted": "failure record size bound"},
                    retained_files={"count": len(retained), "bytes": sum(retained.values()),
                                    "names": sorted(retained)[:16]})
        data = mi.dump_json(body)
    if len(data) > FAILURE_RESERVE:
        data = mi.dump_json({"schema": SCHEMA_FAILURE, "tool_version": TOOL_VERSION, "stage": stage,
                             "status": "failed", "reason": body["reason"][:300], "completed_receipt_published": False,
                             "retained_files": {"count": len(retained), "bytes": sum(retained.values())}})
    try:
        if len(data) > FAILURE_RESERVE:
            raise mi.Blocked("failure record exceeds its reserve")
        # FRESH global check (derived+scratch+runs+masks+suite and free disk) with the 2x transient growth
        # of temp + published name; mi.publish_bytes(failure=True) itself only checks the scope ledger.
        mi.workspace_budget(ledger.ctx, ScoutLedger.TRANSIENT_FACTOR * len(data))
        mi.publish_bytes(ledger, "failure.json", data, failure=True)
    except Exception as publish_exc:  # noqa: BLE001 - keep a terminal record, never mask the original error
        line = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        sys.stderr.write(f"failure.json NOT published ({type(publish_exc).__name__}: {str(publish_exc)[:300]}); "
                         f"the record follows on ONE stderr line as JSON:\n{line}\n")


AUTHORITATIVE_NAMES = (PLAN_NAME, RECEIPT_NAME)


def _fail_stage(ledger: ScoutLedger, stage: str, exc: BaseException) -> None:
    """Retain failure evidence. An AUTHORITATIVE output (plan / receipt) that THIS stage created (ownership
    recorded by ``ScoutLedger.record`` right after the publisher's link, so it also covers errors raised
    later inside the publisher, e.g. its final hash read) and that is still the very file it created is
    renamed to ``invalidated-<name>`` (bytes kept, no longer the completed artifact); only if the rename
    fails is that owned output removed. Foreign files under those names are never renamed or deleted.
    Non-authoritative artifacts (the sheet) are kept as is."""
    records, foreign = [], []
    for name in AUTHORITATIVE_NAMES:
        src = ledger.dir / name
        if not os.path.lexists(src):
            continue
        if not ledger.owned_file(name):
            foreign.append(name)                       # not created by this stage: left exactly as found
            continue
        record = {"name": name}
        try:
            record["sha256"] = mi.sha256_file(src)
        except Exception as hash_exc:  # noqa: BLE001 - the invalidation must still happen
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
    _scout_failure(ledger, stage, exc, {"post_publication_failure": bool(records), "invalidated_outputs": records,
                                        "foreign_names_untouched": foreign})


def _post_publication_closure(before: Mapping, current: Mapping, out_dir: Path, outputs: Mapping[str, str]) -> None:
    """Closure interval ends AFTER the final publication: inputs, code, runtime AND the current names and
    bytes of this stage's own outputs must still match."""
    if current != before:
        changed = sorted(k for k in set(before) | set(current) if before.get(k) != current.get(k))
        raise mi.InputsError(f"closure changed after publication: {changed}")
    names = sorted(entry.name for entry in out_dir.iterdir())
    if names != sorted(outputs):
        raise mi.InputsError(f"closure changed after publication: output names {names}")
    for name, sha in outputs.items():
        if mi.sha256_file(out_dir / name) != sha:
            raise mi.InputsError(f"closure changed after publication: {name} bytes")


def _locate_plan(mctx: mi.Context, plan: str | os.PathLike) -> Path:
    """Plan location only (lstat, never opened): an existing regular non-link ``scout-plan.json`` inside
    the scout scope. A payload, an anchor or any other file is refused before a byte is read."""
    path = mi.contained_input(mctx, plan, "plan")
    if path.name != PLAN_NAME:
        raise mi.InputsError(f"plan must be a {PLAN_NAME} produced by this scout; refused {path.name!r} unread")
    mi.inside_scope(mctx, path, "plan")
    return path


def plan_stage(ctx: ScoutContext, *, receipt_path: str, expected_receipt_sha256: str, index_path: str,
               expected_index_sha256: str, integrity_path: str, expected_integrity_sha256: str,
               existing_input_bytes: int, out: str) -> dict:
    existing = check_existing_input_bytes(existing_input_bytes)
    p = pins(ctx)
    expected = {"acquisition_receipt": mi.check_sha(expected_receipt_sha256, "--expected-receipt-sha256"),
                "resources_index": mi.check_sha(expected_index_sha256, "--expected-index-sha256"),
                "integrity_record": mi.check_sha(expected_integrity_sha256, "--expected-integrity-sha256")}
    if expected != p["anchors"]:
        raise mi.Blocked("expected anchor SHA-256 values are not the pinned historical anchors; no trust can be derived")
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    paths = _anchor_paths(ctx, mctx, receipt_path, index_path, integrity_path)
    ledger = ScoutLedger(mctx, out_dir, "plan")
    state: dict = {"dir": False}
    try:
        with mi.guarded_stage(mctx, read_exceptions=paths.values()) as opened:
            before = {"anchors": dict(p["anchors"]), "sources": _sources(), "runtime": mi.runtime_identity()}
            mi.allow_payloads(paths["resources_index"].parent / "payloads", ())      # EMPTY payload allowlist
            ledger.check_sequence([(PLAN_CAP, True)])                                 # prospective, before reads
            budget = mi.workspace_budget(mctx, PLAN_ALLOWANCE)
            inputs = scout_input_budget(existing, p)
            raws = _read_anchors(paths, p)
            core = _core_from_anchors(mctx, paths, raws, p)
            stats = _payload_stats(mctx, paths, core["rgb"])
            if opened:
                raise mi.GuardError("the plan stage opened payload bytes")
            doc = {**core, "sources": before["sources"], "runtime": before["runtime"], "existing_input_bytes": existing,
                   "input_budget": inputs, "budget": budget, "payload_stats": stats, "payload_opens": 0,
                   "scout_scope": mi.repo_rel(mctx, mctx.scope),
                   "closure": {"before": "anchors + sources + runtime at stage start", "pre_publication_matches": True,
                               "post_publication": POST_PUBLICATION_NOTE}}
            validate_plan_doc(doc, p)
            data = mi.dump_json(doc)
            if len(data) > PLAN_CAP:
                raise mi.Blocked(f"plan is {len(data)} bytes, above {PLAN_CAP}")
            if {"anchors": {k: mi.sha256_file(v) for k, v in paths.items()}, "sources": _sources(),
                    "runtime": mi.runtime_identity()} != before:
                raise mi.InputsError("anchors/sources/runtime changed during planning")
            ledger.check(len(data), metadata=True)
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            published = mi.publish_bytes(ledger, PLAN_NAME, data)      # ownership recorded by ledger.record
            _post_publication_closure(before, {"anchors": {k: mi.sha256_file(v) for k, v in paths.items()},
                                               "sources": _sources(), "runtime": mi.runtime_identity()},
                                      out_dir, {PLAN_NAME: published["sha256"]})
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "plan", exc)
        raise
    return {"plan": mi.repo_rel(mctx, out_dir / PLAN_NAME), "plan_sha256": published["sha256"],
            "rgb_frames": FRAME_COUNT, "encoded_bytes": core["rgb_totals"]["encoded_bytes"], "payload_opens": 0,
            "pins": p["kind"], "exit": 0}


def preview_stage(ctx: ScoutContext, *, plan: str, expected_plan_sha256: str, frames: Sequence[int] = PREVIEW_INDICES,
                  out: str) -> dict:
    from bench.vision import cv2
    from bench import workbench_geometry as wg
    indices = check_preview_frames(list(frames))                       # BEFORE any path or payload
    p = pins(ctx)
    mctx = ctx.mi()
    out_dir = mi.check_output_dir(mctx, out)
    expected_plan = mi.check_sha(expected_plan_sha256, "--expected-plan-sha256")
    _locate_plan(mctx, plan)                                           # location only; nothing read yet
    ledger = ScoutLedger(mctx, out_dir, "preview")
    state: dict = {"dir": False}
    try:
        # Whole-stage guard: EMPTY payload allowlist and NO read exceptions until the plan is validated.
        with mi.guarded_stage(mctx) as opened:
            plan_path = _locate_plan(mctx, plan)                       # re-checked immediately before reading
            raw_plan = mi.read_bounded(plan_path, PLAN_CAP, "plan")
            if mi.sha256_bytes(raw_plan) != expected_plan:
                raise mi.InputsError("plan: SHA-256 differs from the parent-accepted value")
            doc = validate_plan_doc(mi.strict_json(raw_plan, "plan"), p)
            del raw_plan
            if doc["scout_scope"] != mi.repo_rel(mctx, mctx.scope):
                raise mi.InputsError("plan was produced for a different scout scope")
            paths = {k: mi.contained_input(mctx, a["path"], k, allow_denied=True) for k, a in doc["anchors"].items()}
            for key, path in paths.items():
                mi.outside_scope_input(mctx, path, key)
            payload_root = paths["resources_index"].parent / "payloads"
            mi.allow_payloads(payload_root, ())                                    # EMPTY payload allowlist
            for path in paths.values():
                mi.add_read_exception(path)                                        # exact READ-ONLY anchors
            before = {"plan": expected_plan, "anchors": dict(p["anchors"]), "sources": _sources(),
                      "runtime": mi.runtime_identity()}
            if before["sources"] != doc["sources"] or before["runtime"] != doc["runtime"]:
                raise mi.InputsError("source closure/runtime differ from the accepted plan")
            raws = _read_anchors(paths, p)
            core = _core_from_anchors(mctx, paths, raws, p)
            if mi.dump_json(core) != mi.dump_json({k: doc[k] for k in CORE_KEYS}):
                raise mi.InputsError("accepted plan differs from a fresh metadata rebuild of the anchors")
            rows = [doc["rgb"][i] for i in indices]
            if _payload_stats(mctx, paths, doc["rgb"]) != doc["payload_stats"]:
                raise mi.InputsError("payload stats changed since the plan")
            payload_paths = [mi.safe_member_path(payload_root.parent, r["payload_path"], r["member"]) for r in rows]
            ledger.check_sequence([(SHEET_CAP, False), (RECEIPT_CAP, True)])     # prospective, before payloads
            budget = mi.workspace_budget(mctx, PREVIEW_ALLOWANCE)
            inputs = scout_input_budget(doc["existing_input_bytes"], p)
            if opened:
                raise mi.GuardError("payload bytes were opened before the preview grant")
            mi.allow_payloads(payload_root, payload_paths)                         # exactly the authorized frames
            decoded, frame_records = [], []
            for row, path in zip(rows, payload_paths):
                chunks: list[bytes] = []
                verified = mi.verified_payload_stream(path, row, chunks.append)
                raw = b"".join(chunks)
                del chunks
                mi.png_rgb8_header(raw)
                rgb = mi.decode_png_rgb(raw)
                del raw
                decoded.append((row["index"], rgb))
                frame_records.append({"index": row["index"], "frame_id": row["frame_id"], "split": row["split"],
                                      "member": row["member"], "png_sha256": verified["uncompressed_sha256"],
                                      "compressed_sha256": verified["compressed_sha256"], "crc32": verified["crc32"],
                                      "native_size": list(mi.IMAGE_SIZE),
                                      "decoded_rgb_semantic_sha256": wg.array_sha256(rgb)})
            if sorted(opened) != sorted(mi._norm(x) for x in payload_paths) or len(opened) != len(payload_paths):
                raise mi.GuardError("payload opens differ from the exact authorized frames")
            sheet, layout = build_contact_sheet(decoded)
            del decoded
            ok, encoded = cv2.imencode(".png", np.ascontiguousarray(sheet[:, :, ::-1]),
                                       [cv2.IMWRITE_PNG_COMPRESSION, SHEET_PNG_COMPRESSION])
            if not ok:
                raise mi.InputsError("contact sheet encoding failed")
            sheet_bytes = encoded.tobytes()
            if len(sheet_bytes) > SHEET_CAP:
                raise mi.Blocked(f"contact sheet is {len(sheet_bytes)} bytes, above {SHEET_CAP}")
            ledger.check_sequence([(len(sheet_bytes), False), (RECEIPT_CAP, True)])
            mi._make_fresh_dir(out_dir)
            state["dir"] = True
            sheet_meta = mi.publish_bytes(ledger, SHEET_NAME, sheet_bytes, metadata=False)
            check = cv2.imdecode(np.frombuffer((out_dir / SHEET_NAME).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if check is None or check.shape != (layout["height"], layout["width"], 3) or check.shape[1] > SHEET_MAX_WIDTH:
                raise mi.InputsError("published contact sheet does not decode to its layout")
            receipt = {"schema": SCHEMA_PREVIEW, "tool_version": TOOL_VERSION, "status": "preview_published_candidate_only",
                       "scene": SCENE, "flags": dict(FLAGS), "trust": dict(TRUST), "pins": p["kind"],
                       "plan": {"path": mi.repo_rel(mctx, plan_path), "sha256": expected_plan},
                       "frames": frame_records, "authorized_indices": list(indices),
                       "contact_sheet": {**sheet_meta, "width": layout["width"], "height": layout["height"],
                                         "semantic_sha256": wg.array_sha256(sheet),
                                         "png_compression": SHEET_PNG_COMPRESSION},
                       "mapping": layout["mapping"], "banner_lines": layout["banner_lines"],
                       "clock": dict(CLOCK), "calibration_status": CALIBRATION_STATUS,
                       "payload_opens": len(opened), "budget": budget, "input_budget": inputs,
                       "scout_scope": mi.repo_rel(mctx, mctx.scope),
                       "closure": {"before": before, "pre_publication_matches": True,
                                   "post_publication": POST_PUBLICATION_NOTE},
                       "ledger": ledger.summary()}

            def current() -> dict:
                return {"plan": mi.sha256_file(plan_path), "anchors": {k: mi.sha256_file(v) for k, v in paths.items()},
                        "sources": _sources(), "runtime": mi.runtime_identity()}

            if current() != before or mi.sha256_file(out_dir / SHEET_NAME) != sheet_meta["sha256"]:
                raise mi.InputsError("plan/anchors/sources/runtime or the sheet changed during the preview")
            data = mi.dump_json(receipt)
            if len(data) > RECEIPT_CAP:
                raise mi.Blocked("preview receipt exceeds its cap")
            published = mi.publish_bytes(ledger, RECEIPT_NAME, data)   # ownership recorded by ledger.record
            _post_publication_closure(before, current(), out_dir,
                                      {SHEET_NAME: sheet_meta["sha256"], RECEIPT_NAME: published["sha256"]})
    except BaseException as exc:
        if state["dir"]:
            _fail_stage(ledger, "preview-opening", exc)
        raise
    return {"receipt": mi.repo_rel(mctx, out_dir / RECEIPT_NAME), "receipt_sha256": published["sha256"],
            "contact_sheet_sha256": sheet_meta["sha256"], "frames": list(indices), "payload_opens": len(indices),
            "exit": 0}




# --------------------------------------------------------------------------
# CLI (never sets synthetic pins)


def _int_arg(text: str) -> int:
    if not re.fullmatch(r"[0-9]{1,15}", text):
        raise argparse.ArgumentTypeError("must be a non-negative decimal integer")
    return int(text)


def main(argv: Sequence[str] | None = None, ctx: ScoutContext | None = None) -> int:
    base = ctx or ScoutContext()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--acquisition-receipt", required=True)
    s.add_argument("--expected-receipt-sha256", required=True)
    s.add_argument("--resources-index", required=True)
    s.add_argument("--expected-index-sha256", required=True)
    s.add_argument("--integrity-record", required=True)
    s.add_argument("--expected-integrity-sha256", required=True)
    s.add_argument("--existing-input-bytes", type=_int_arg, required=True)
    s.add_argument("--scout-root", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("preview-opening")
    s.add_argument("--plan", required=True)
    s.add_argument("--expected-plan-sha256", required=True)
    s.add_argument("--frame", type=_int_arg, action="append")
    s.add_argument("--scout-root", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(args.scout_root)
        root = Path(os.path.abspath(root if root.is_absolute() else base.root / root))
        run_ctx = ScoutContext(root=base.root, workspace=base.workspace, scout_root=root, synthetic=base.synthetic)
        if args.stage == "plan":
            result = plan_stage(run_ctx, receipt_path=args.acquisition_receipt,
                                expected_receipt_sha256=args.expected_receipt_sha256,
                                index_path=args.resources_index, expected_index_sha256=args.expected_index_sha256,
                                integrity_path=args.integrity_record,
                                expected_integrity_sha256=args.expected_integrity_sha256,
                                existing_input_bytes=args.existing_input_bytes, out=args.out)
        else:
            result = preview_stage(run_ctx, plan=args.plan, expected_plan_sha256=args.expected_plan_sha256,
                                   frames=args.frame if args.frame else list(PREVIEW_INDICES), out=args.out)
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
                         "tools.workbench_hb_prefix_scout ...)")
    raise SystemExit(main())
