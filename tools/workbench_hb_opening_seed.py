"""Isolated CONTROLLED_INIT opening-seed extractor for HB val_primesense scene 000002 (frame 9 only).

This is the ONLY place that reads raw scene annotation bytes. It is a narrow, explicit
exception for a controlled-init opening seed. The input preparer, automatic geometry and
inference never import it, and automatic geometry continues to forbid all GT.

Before any annotation byte is read, the caller must supply all of:

* the expected SHA-256 of the raw ``scene_gt.json``;
* opening frame 9 (nothing else is accepted);
* an explicit object id and instance index;
* the frozen opening RGB selection review (expected SHA), which binds frame 9, its allowlisted
  RGB source SHA, the object id, the instance and a qualitative opening-only parent attestation;
* the SHA-only allowlist (expected SHA).

The annotation file is first hashed in full (bounded) and compared. It is then streamed through
a LEXICAL top-level scanner that captures ONLY the value of key ``"9"``. Every other frame's
value is skipped by structure: no numeric value is converted, no dict or list is built, and
duplicate top-level keys are rejected. No full-file ``json.load``/``json.loads`` is used. Only the
selected instance's pose is output: R (proper SO(3)) and t converted mm -> metres exactly once.
The output names no other frame value or candidate pose.

The seed applies ONCE to s000010 and is never reused on recovery (the workbench runtime enforces
single use).

    extract --scene-gt P --expected-scene-gt-sha256 H --frame-index 9 --object-id N --instance-index K
            --selection-review P --expected-selection-review-sha256 H
            --allowlist P --expected-allowlist-sha256 H --out NEW_DIR

Exit codes: 0 completed; 1 failure; 2 usage; 3 blocked (missing or ambiguous binding).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (one-way: the preparer never imports this)

TOOL_VERSION = "1"
SEED_FRAME_KEY = "9"
MAX_GT_BYTES = 64 << 20
MAX_CAPTURE_BYTES = 1 << 20
MAX_DEPTH = 64
MAX_INSTANCES = 64
_WS = frozenset(b" \t\r\n")
_LITERAL_CHARS = frozenset(b"0123456789+-.eEtruefalsn")
_LITERAL_RE = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null")
_KEY_RE = re.compile(r"[0-9]{1,6}")


class TopLevelFrameExtractor:
    """Byte-level lexical scanner over a top-level JSON object. Captures the raw bytes of exactly
    one key's value; every other value is skipped structurally (brackets, strings and literal
    token shapes only; no numbers are converted and nothing is materialised)."""

    def __init__(self, target_key: str, max_capture: int = MAX_CAPTURE_BYTES):
        if not _KEY_RE.fullmatch(target_key):
            raise mi.UsageError("target key must be a decimal frame id")
        self.target, self.max_capture = target_key, max_capture
        self.state = "start"
        self.keys: set[str] = set()
        self.key = bytearray()
        self.capture: bytearray | None = None
        self.captured: bytes | None = None
        self.stack: list[int] = []
        self.in_string = False
        self.escape = False
        self.literal = bytearray()
        self.total = 0

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        if self.total > MAX_GT_BYTES:
            raise mi.InputsError("annotation stream exceeds its cap")
        for b in data:
            self._byte(b)

    def _fail(self, why: str) -> None:
        raise mi.InputsError(f"annotation JSON: {why}")

    def _byte(self, b: int) -> None:
        s = self.state
        if s in ("value_start", "value"):
            self._value_byte(b)
        elif s == "start":
            if b == ord("{"):
                self.state = "key_or_close"
            elif b not in _WS:
                self._fail("top level must be an object")
        elif s in ("key_or_close", "key_expect"):
            if b == ord('"'):
                self.state, self.key = "key", bytearray()
            elif b == ord("}") and s == "key_or_close":
                self.state = "done"
            elif b not in _WS:
                self._fail("expected a key")
        elif s == "key":
            if b == ord('"'):
                try:
                    key = self.key.decode("ascii")
                except UnicodeDecodeError:
                    self._fail("non-ASCII top-level key")
                if not _KEY_RE.fullmatch(key):
                    self._fail("top-level keys must be plain decimal frame ids (escapes unsupported)")
                if key in self.keys:
                    self._fail(f"duplicate top-level key {key!r}")
                self.keys.add(key)
                self.capture = bytearray() if key == self.target else None
                self.state = "colon"
            elif b == ord("\\") or b < 0x20:
                self._fail("escaped or control characters in a top-level key")
            else:
                self.key.append(b)
                if len(self.key) > 16:
                    self._fail("top-level key too long")
        elif s == "colon":
            if b == ord(":"):
                self.state = "value_start"
            elif b not in _WS:
                self._fail("expected ':'")
        elif s == "after_value":
            if b == ord(","):
                self.state = "key_expect"
            elif b == ord("}"):
                self.state = "done"
            elif b not in _WS:
                self._fail("expected ',' or '}'")
        elif s == "done":
            if b not in _WS:
                self._fail("data after the top-level object")

    def _complete(self) -> None:
        if self.capture is not None:
            self.captured = bytes(self.capture)
            self.capture = None
        self.state = "after_value"

    def _value_byte(self, b: int) -> None:
        if self.state == "value_start":
            if b in _WS:
                return
            self.state = "value"
        if self.literal:
            if b in _LITERAL_CHARS:
                self.literal.append(b)
                self._keep(b)
                return
            if not _LITERAL_RE.fullmatch(bytes(self.literal)):          # lexical shape only; never converted
                self._fail("invalid literal token")
            self.literal = bytearray()
            if not self.stack:
                self._complete()
                self._byte(b)                                          # the delimiter belongs to the top level
                return
        self._keep(b)
        if self.in_string:
            if self.escape:
                self.escape = False
            elif b == ord("\\"):
                self.escape = True
            elif b == ord('"'):
                self.in_string = False
                if not self.stack:
                    self._complete()
            elif b < 0x20:
                self._fail("control character in a string")
            return
        if b == ord('"'):
            self.in_string = True
        elif b in (ord("{"), ord("[")):
            self.stack.append(b)
            if len(self.stack) > MAX_DEPTH:
                self._fail("nesting too deep")
        elif b in (ord("}"), ord("]")):
            if not self.stack or self.stack.pop() != (ord("{") if b == ord("}") else ord("[")):
                self._fail("unbalanced brackets")
            if not self.stack:
                self._complete()
        elif b in (ord(","), ord(":")):
            if not self.stack:
                self._fail("delimiter outside a container")
        elif b in _WS:
            pass
        elif b in _LITERAL_CHARS:
            self.literal.append(b)
        else:
            self._fail(f"unexpected byte {b!r}")

    def _keep(self, b: int) -> None:
        if self.capture is not None:
            self.capture.append(b)
            if len(self.capture) > self.max_capture:
                self._fail("selected frame value exceeds its cap")

    def finish(self) -> bytes:
        if self.state != "done" or self.literal or self.stack or self.in_string:
            self._fail("truncated top-level object")
        if self.captured is None:
            raise mi.Blocked(f"frame key {self.target!r} is absent from the annotation file")
        return self.captured


def extract_frame_value(chunks: Iterable[bytes], target_key: str = SEED_FRAME_KEY) -> bytes:
    scanner = TopLevelFrameExtractor(target_key)
    for chunk in chunks:
        scanner.feed(chunk)
    return scanner.finish()


def select_pose(frame_value: Any, object_id: int, instance_index: int) -> list[list[float]]:
    """The explicitly bound instance only; mm translation -> metres exactly once."""
    from bench import workbench_contract as wc
    if not isinstance(frame_value, list) or not 1 <= len(frame_value) <= MAX_INSTANCES:
        raise mi.InputsError("frame 9 annotation must be a non-empty bounded list")
    for entry in frame_value:
        if not isinstance(entry, dict) or set(entry) != {"cam_R_m2c", "cam_t_m2c", "obj_id"} \
                or type(entry["obj_id"]) is not int:
            raise mi.InputsError("frame 9 annotation entries must be {cam_R_m2c, cam_t_m2c, obj_id}")
    if not 0 <= instance_index < len(frame_value):
        raise mi.Blocked("explicit instance index is not present in frame 9")
    entry = frame_value[instance_index]
    if entry["obj_id"] != object_id:
        raise mi.Blocked("the explicitly bound instance is not the bound object id; ambiguous binding")
    R, t = entry["cam_R_m2c"], entry["cam_t_m2c"]
    if not (isinstance(R, list) and len(R) == 9 and isinstance(t, list) and len(t) == 3) or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in R + t):
        raise mi.InputsError("selected R/t must be 9 + 3 finite numbers")
    pose = [[float(R[0]), float(R[1]), float(R[2]), float(t[0]) * mi.MM_TO_M],
            [float(R[3]), float(R[4]), float(R[5]), float(t[1]) * mi.MM_TO_M],
            [float(R[6]), float(R[7]), float(R[8]), float(t[2]) * mi.MM_TO_M],
            [0.0, 0.0, 0.0, 1.0]]
    try:
        wc._pose(pose, "seed_pose_m")                      # proper SO(3) + homogeneous row
    except (wc.ContractError, TypeError, ValueError) as exc:
        raise mi.InputsError(f"selected pose is not a proper rigid transform: {exc}") from None
    return pose


def _hash_stream(path: Path) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(mi.CHUNK), b""):
            size += len(block)
            if size > MAX_GT_BYTES:
                raise mi.InputsError("annotation file exceeds its cap")
            h.update(block)
    return h.hexdigest(), size


def extract_seed(ctx: mi.Context, *, scene_gt: str, expected_scene_gt_sha256: str, frame_index: int,
                 object_id: int, instance_index: int, selection_review: str, expected_selection_review_sha256: str,
                 allowlist: str, expected_allowlist_sha256: str, out: str) -> dict:
    if type(frame_index) is not int or frame_index != mi.SEED_INDEX:
        raise mi.UsageError("only opening frame 9 can supply the controlled-init seed")
    if type(object_id) is not int or object_id not in mi.MODEL_IDS or type(instance_index) is not int or instance_index < 0:
        raise mi.UsageError("explicit object id (1..10) and instance index (>= 0) are required")
    expected_gt = mi.check_sha(expected_scene_gt_sha256, "--expected-scene-gt-sha256")
    expected_allow = mi.check_sha(expected_allowlist_sha256, "--expected-allowlist-sha256")
    expected_review = mi.check_sha(expected_selection_review_sha256, "--expected-selection-review-sha256")
    out_dir = mi.check_output_dir(ctx, out)
    gt_path = mi.contained_input(ctx, scene_gt, "scene annotation", allow_denied=True)
    if gt_path.name != "scene_gt.json":
        raise mi.UsageError("the seed source must be an explicit scene_gt.json metadata file")
    mi.outside_scope_input(ctx, gt_path, "scene annotation")
    own = Path(__file__).resolve()
    with mi.guarded_stage(ctx) as opened:                                     # GT not yet readable
        before = {"seed_module": mi.sha256_file(own), "sources": mi.source_closure(), "runtime": mi.runtime_identity(),
                  "allowlist": expected_allow, "selection_review": expected_review, "scene_gt": expected_gt}
        allow, allow_path, payload_root = mi.load_allowlist(ctx, allowlist, expected_allow)
        mi._set_payload_root(payload_root)                                   # zero payload opens
        review_raw, review_path = mi.read_pinned(ctx, selection_review, expected_review, mi.MAX_REVIEW_BYTES,
                                                 "selection review")
        mi.outside_scope_input(ctx, review_path, "selection review")
        review = mi.validate_selection_review(mi.strict_json(review_raw, "selection review"), allow)
        if review["object_id"] != object_id or review["instance_index"] != instance_index:
            raise mi.Blocked("object id / instance index differ from the reviewed opening binding")
        if not any(m["object_id"] == object_id for m in allow["models"]):
            raise mi.Blocked("the bound object id has no allowlisted CAD model")
        budget = mi.workspace_budget(ctx, mi.STAGE_ALLOWANCE["extract-seed"])
        mi.add_read_exception(gt_path)                       # only now: all prerequisites are bound (read-only)
        digest, size = _hash_stream(gt_path)                 # full pre-hash, verified BEFORE any parsing
        if digest != expected_gt:
            raise mi.InputsError("scene annotation SHA-256 differs from the caller's expected value")
        check = hashlib.sha256()

        def chunks():
            with open(gt_path, "rb") as stream:
                for block in iter(lambda: stream.read(64 << 10), b""):
                    check.update(block)
                    yield block
        raw_value = extract_frame_value(chunks(), SEED_FRAME_KEY)
        scan_digest = check.hexdigest()
        if scan_digest != expected_gt:
            raise mi.InputsError("scene annotation changed between pre-hashing and the scan")
        pose = select_pose(mi.strict_json(raw_value, "frame 9 annotation"), object_id, instance_index)
        label = (f"controlled_init:dataset_onboarding:HB:{mi.DATASET_REVISION[:12]}:val_primesense/{mi.SCENE}:"
                 f"obj_{object_id:06d}:instance{instance_index}:s{mi.SEED_INDEX:06d}")
        seed = {
            "schema": mi.SCHEMA_SEED, "tool_version": TOOL_VERSION, "corpus": mi.CORPUS,
            "dataset_revision": mi.DATASET_REVISION, "scene": mi.SCENE, "frame_index": mi.SEED_INDEX,
            "frame_id": f"s{mi.SEED_INDEX:06d}", "object_id": object_id, "instance_index": instance_index,
            "seed_pose_m": pose, "units": "metres",
            "translation_conversion": "BOP millimetres * 0.001 applied exactly once; rotation unchanged",
            "units_reference": mi.BOP_FORMAT_URL, "seed_label": label,
            "label_kind": "controlled_init dataset onboarding pose of opening frame 9; never scored, never evaluator",
            "selection_review_sha256": expected_review, "rgb_source_sha256": review["rgb_source_sha256"],
            "allowlist_sha256": expected_allow,
            "scene_gt": {"path": mi.repo_rel(ctx, gt_path), "sha256": digest, "scan_sha256": scan_digest, "bytes": size,
                         "expected_sha256_matched_before_parsing": True},
            "applies_once_to_frame_id": f"s{mi.SEED_INDEX + 1:06d}", "reuse_on_recovery": False,
            "other_frames_materialized": False,
            "extraction": "lexical top-level scan; only key \"9\" captured; other values skipped without numeric "
                          "conversion or materialisation (a lexical scanner, not a full JSON grammar validator)",
            "trust": dict(mi.TRUST),
        }
        if opened:
            raise mi.GuardError("seed extraction opened payload bytes")

        def closure_now() -> dict:
            return {"seed_module": mi.sha256_file(own), "sources": mi.source_closure(),
                    "runtime": mi.runtime_identity(), "allowlist": mi.sha256_file(allow_path),
                    "selection_review": mi.sha256_file(review_path), "scene_gt": _hash_stream(gt_path)[0]}
        if closure_now() != before:
            raise mi.InputsError("seed module/sources/runtime/allowlist/review/annotation changed during extraction")
        ledger = mi.OutputLedger(ctx, out_dir, "extract-seed")
        seed_bytes = mi.dump_json(seed)
        ledger.check(len(seed_bytes), metadata=True)
        mi._make_fresh_dir(out_dir)
        try:
            seed_meta = mi.publish_bytes(ledger, "seed.json", seed_bytes)
            # Re-check the complete closure AFTER the intermediate output and immediately before success.
            if closure_now() != before or mi.sha256_file(out_dir / "seed.json") != seed_meta["sha256"]:
                raise mi.InputsError("seed inputs/sources/runtime or the published seed changed during publication; "
                                     "no success receipt")
            receipt = {"schema": mi.SCHEMA_SEED_RECEIPT, "tool_version": TOOL_VERSION, "status": "extracted",
                       "seed": seed_meta, "budget": budget, "ledger": ledger.summary(),
                       "closure": {"before": before, "after_matches": True, "after_publication_matches": True},
                       "preparation_scope": mi.repo_rel(ctx, ctx.scope),
                       "admitted": False, "registered": False}
            published = mi.publish_bytes(ledger, "seed-receipt.json", mi.dump_json(receipt))
        except BaseException as exc:
            mi._failure(ledger, "extract-seed", exc)
            raise
    return {"seed": mi.repo_rel(ctx, out_dir / "seed.json"), "seed_sha256": seed_meta["sha256"],
            "receipt_sha256": published["sha256"], "exit": 0}


def _int(text: str) -> int:
    if not re.fullmatch(r"[0-9]{1,6}", text):
        raise argparse.ArgumentTypeError("must be a non-negative decimal integer")
    return int(text)


def main(argv: Sequence[str] | None = None, ctx: mi.Context | None = None) -> int:
    ctx = ctx or mi.Context()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("extract")
    s.add_argument("--scene-gt", required=True)
    s.add_argument("--expected-scene-gt-sha256", required=True)
    s.add_argument("--frame-index", type=_int, required=True)
    s.add_argument("--object-id", type=_int, required=True)
    s.add_argument("--instance-index", type=_int, required=True)
    s.add_argument("--selection-review", required=True)
    s.add_argument("--expected-selection-review-sha256", required=True)
    s.add_argument("--allowlist", required=True)
    s.add_argument("--expected-allowlist-sha256", required=True)
    s.add_argument("--preparation-root")
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        ctx = mi.apply_preparation_root(ctx, args.preparation_root)
        result = extract_seed(ctx, scene_gt=args.scene_gt, expected_scene_gt_sha256=args.expected_scene_gt_sha256,
                              frame_index=args.frame_index, object_id=args.object_id,
                              instance_index=args.instance_index, selection_review=args.selection_review,
                              expected_selection_review_sha256=args.expected_selection_review_sha256,
                              allowlist=args.allowlist, expected_allowlist_sha256=args.expected_allowlist_sha256,
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
    return 0


if __name__ == "__main__":
    if not sys.dont_write_bytecode:
        raise SystemExit("Use python -B (for example: python -B -S tools/prototype_bootstrap.py "
                         "tools.workbench_hb_opening_seed ...)")
    raise SystemExit(main())
