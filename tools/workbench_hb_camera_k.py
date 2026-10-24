"""Pure, bounded helpers for the HB scene-000003 camera-K derivative (no stage, no CLI, no I/O policy).

The stages live in ``tools/workbench_hb_scene3_profile.py``; this module only supplies:

* ``validate_camera_row``  - the exact index-bound ``scene_camera.json`` REUSE row (stored method 0,
  stored size/SHA == expanded size/SHA, canonical ``hb-scene-metadata-v1`` path, role ``metadata``)
  with the original ZIP method/compressed size kept as separate provenance. The ordinary RGB
  validator is NOT used (it only admits ``payload`` rows).
* ``hash_bounded``         - pass 1: size/SHA-256/CRC-32 of the whole bounded file BEFORE any parsing.
* ``StructuralScanner``    - pass 2: a streaming RFC 8259 grammar validator (UTF-8, duplicate keys at
  every level, numeric lexical grammar across chunk boundaries, depth bound) that captures ONLY the
  ``cam_K`` arrays of the requested top-level frame keys. Every other value (R, t, depth_scale,
  out-of-window frames, strings, literals) is validated lexically and skipped: no numeric conversion,
  no array/object construction and no materialised camera document.
* ``validate_k_window``    - exactly 9 finite numbers per requested frame, zero-skew pinhole form,
  positive fx/fy, principal point inside the native image, identical across the window. Raw K is
  preserved: no crop, rescale or origin adjustment.

Trust: historical local integrity only (``upstream_authenticated: false``).
"""
from __future__ import annotations

import binascii
import codecs
import hashlib
import math
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import workbench_hb_metric_inputs as mi  # noqa: E402  (pure helpers only; never mutated)

CAMERA_ROLE = "metadata"
CAMERA_CAP = 1 << 20                     # camera input bound (bytes)
READ_CHUNK = 64 << 10
MAX_DEPTH = 64
MAX_KEY_CHARS = 256
MAX_CAPTURED_NUMBER_CHARS = 64
K_LEN = 9
TOP_KEY = re.compile(r"(0|[1-9][0-9]{0,8})\Z")
WHITESPACE = frozenset(" \t\n\r")
DIGITS = frozenset("0123456789")
HEX = frozenset("0123456789abcdefABCDEF")
ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
LITERALS = {"t": "true", "f": "false", "n": "null"}
NUMBER_ACCEPT = frozenset({"zero", "int", "frac", "exp"})


class CameraError(mi.InputsError):
    """Camera metadata is missing, malformed, inconsistent or incompatible (fail closed)."""


def check_requested(requested: Any) -> list[int]:
    """Validate EVERY original element before any sort/set normalisation: each must be a plain
    non-negative ``int`` (booleans, floats, strings refused), non-empty and without repeats, so a
    hidden ``False``/``True`` or a bad element can never be absorbed by de-duplication."""
    if isinstance(requested, (str, bytes, bytearray, Mapping)):
        raise mi.UsageError("requested frame indices must be an iterable of integers")
    try:
        items = list(requested)
    except TypeError:
        raise mi.UsageError("requested frame indices must be an iterable of integers") from None
    if not items:
        raise mi.UsageError("no requested frame indices")
    for item in items:
        if type(item) is not int or item < 0:
            raise mi.UsageError(f"requested frame index {item!r} is not a non-negative integer (booleans are invalid)")
    if len(set(items)) != len(items):
        raise mi.UsageError("requested frame indices repeat")
    return sorted(items)


# --------------------------------------------------------------------------
# index row (pure)


def canonical_reuse_path(root: Path, reuse_root: str, member: str) -> Path:
    if not isinstance(member, str) or any(p in ("", ".", "..") for p in member.split("/")) or "\\" in member:
        raise CameraError(f"unsafe member {member!r}")
    if not isinstance(reuse_root, str) or ".." in PurePosixPath(reuse_root).parts or PurePosixPath(reuse_root).is_absolute():
        raise CameraError("unsafe reuse root")
    return Path(os.path.abspath(Path(root).joinpath(*PurePosixPath(reuse_root).parts, *PurePosixPath(member).parts)))


def _reuse_form(root: Path, text: Any, canonical: Path) -> str:
    if not isinstance(text, str) or not text or "\x00" in text or "://" in text or text.lower().startswith("file:"):
        raise CameraError("reuse location path missing or not a local path")
    candidate = Path(text)
    form = "absolute" if candidate.is_absolute() else "repository_relative"
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    if ".." in candidate.parts:
        raise CameraError("reuse location path uses traversal")
    if os.path.normcase(os.path.abspath(candidate)) != os.path.normcase(str(canonical)):
        raise CameraError("reuse location is not the canonical hb-scene-metadata-v1 member path")
    return form


def validate_camera_row(row: Any, *, member: str, archive: str, revision: str, expected_sha256: str,
                        max_bytes: int, root: Path, canonical: Path) -> dict:
    """The exact index-bound camera REUSE row (no file access)."""
    if not isinstance(row, Mapping):
        raise CameraError(f"{member}: index row missing")
    ints = ("sourceMethod", "sourceFlags", "sourceCompressedBytes", "sourceOffset", "expandedBytes", "crc32", "method",
            "compressedBytes")
    if any(type(row.get(k)) is not int or row.get(k) < 0 for k in ints):
        raise CameraError(f"{member}: ZIP metadata must be non-negative integers (booleans are invalid)")
    location = row.get("location")
    if (row.get("datasetRevision") != revision or row.get("archive") != archive or row.get("sourceMember") != member
            or not isinstance(location, Mapping) or location.get("kind") != "reuse"
            or row.get("allowedRoles") != [CAMERA_ROLE]):
        raise CameraError(f"{member}: revision/archive/member/location kind/role differ from the expected reuse row")
    size = row["expandedBytes"]
    if (row["method"] != 0 or row["compressedBytes"] != size or not 0 < size <= max_bytes
            or row["crc32"] > 0xFFFFFFFF or row["sourceMethod"] not in (0, 8) or row["sourceFlags"] & ~2048
            or row["sourceCompressedBytes"] == 0 or row["sourceCompressedBytes"] > size + (1 << 20)
            or (row["sourceMethod"] == 0 and row["sourceCompressedBytes"] != size)):
        raise CameraError(f"{member}: stored/original ZIP method, flags or sizes are inconsistent")
    if not (mi._is_sha(row.get("compressedSHA256")) and row.get("compressedSHA256") == row.get("uncompressedSHA256")
            == expected_sha256):
        raise CameraError(f"{member}: stored SHA-256 must equal the expanded SHA-256 and the pinned expectation")
    form = _reuse_form(root, location.get("path"), canonical)
    return {"member": member, "archive": archive, "dataset_revision": revision, "role": CAMERA_ROLE,
            "location_kind": "reuse", "location_path_form": form,
            "stored": {"method": 0, "bytes": size, "sha256": row["compressedSHA256"]},
            "expanded_bytes": size, "expanded_sha256": row["uncompressedSHA256"], "crc32": row["crc32"],
            "original_zip": {"method": row["sourceMethod"], "compressed_bytes": row["sourceCompressedBytes"],
                             "flags": row["sourceFlags"], "offset": row["sourceOffset"]}}


# --------------------------------------------------------------------------
# pass 1: whole-file hash before parsing


def hash_bounded(path: Path, *, expected_bytes: int, cap: int = CAMERA_CAP) -> dict:
    if type(expected_bytes) is not int or not 0 < expected_bytes <= cap:
        raise CameraError("camera size is outside the input cap")
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_size != expected_bytes:
        raise CameraError("camera file is not a regular file of the index-bound size")
    h, crc, count = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(READ_CHUNK)
            if not block:
                break
            count += len(block)
            if count > expected_bytes:
                raise CameraError("camera file grew beyond its index-bound size")
            h.update(block)
            crc = binascii.crc32(block, crc)
    if count != expected_bytes:
        raise CameraError("camera file is shorter than its index-bound size")
    return {"bytes": count, "sha256": h.hexdigest(), "crc32": crc & 0xFFFFFFFF}


# --------------------------------------------------------------------------
# pass 2: streaming structural scanner


class StructuralScanner:
    """Incremental JSON validator capturing ONLY ``cam_K`` of the requested top-level frame keys.

    Feed arbitrary byte chunks (boundaries may split UTF-8 sequences, escapes, numbers or literals).
    Top-level keys must be canonical decimal frame ids (``0``, ``1``, ... no leading zeros); a
    requested frame must be an object holding exactly one ``cam_K`` array of exactly 9 numbers.
    Skipped numbers are validated by the lexical grammar only and never converted."""

    def __init__(self, requested: Iterable[int]):
        req = check_requested(requested)
        self.requested = {str(i): i for i in req}
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self._stack: list[dict] = []
        self._top = "value"                          # "value" | "end"
        self._lex: str | None = None                 # None | "str" | "num" | "lit"
        self._str_key = False
        self._str_buf: list[str] = []
        self._str_len = 0
        self._esc = False
        self._uni = 0
        self._uni_buf = ""
        self._high: int | None = None
        self._num_state = ""
        self._num_buf: list[str] | None = None
        self._lit = ""
        self._lit_pos = 0
        self._first_char = True
        self.captured: dict[int, list[str]] = {}
        self.stats = {"chars": 0, "top_level_keys": 0, "requested_frames_found": 0,
                      "numbers_captured": 0, "numbers_skipped_unconverted": 0, "max_depth": 0}

    # -- public
    def feed(self, data: bytes) -> None:
        try:
            text = self._utf8.decode(data)
        except UnicodeDecodeError as exc:
            raise CameraError(f"camera JSON is not valid UTF-8 ({exc.reason})") from None
        for ch in text:
            self._char(ch)

    def finish(self) -> dict:
        try:
            text = self._utf8.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise CameraError(f"camera JSON ends inside a UTF-8 sequence ({exc.reason})") from None
        for ch in text:
            self._char(ch)
        if self._lex == "num":
            self._finish_number()
        if self._lex is not None or self._stack or self._top != "end":
            raise CameraError("camera JSON is truncated or incomplete")
        missing = [i for i in self.requested.values() if i not in self.captured]
        if missing:
            raise CameraError(f"cam_K missing for requested frame(s) {missing[:8]}{'...' if len(missing) > 8 else ''}")
        return {"captured": {i: list(self.captured[i]) for i in sorted(self.captured)}, "stats": dict(self.stats)}

    # -- dispatch
    def _char(self, ch: str) -> None:
        self.stats["chars"] += 1
        if self._first_char:
            self._first_char = False
            if ch == "\ufeff":
                raise CameraError("camera JSON must not start with a byte-order mark")
        if self._lex == "str":
            self._string_char(ch)
            return
        if self._lex == "num":
            if self._number_char(ch):
                return
            self._finish_number()
        elif self._lex == "lit":
            self._literal_char(ch)
            return
        if ch in WHITESPACE:
            return
        if not self._stack and self._top == "end":
            raise CameraError("camera JSON has trailing data after the top-level value")
        frame = self._stack[-1] if self._stack else None
        if ch == "}":
            if frame is None or frame["kind"] != "obj" or frame["state"] not in ("key_or_end", "comma_or_end"):
                raise CameraError("unexpected '}'")
            self._stack.pop()
            self._end_value()
            return
        if ch == "]":
            if frame is None or frame["kind"] != "arr" or frame["state"] not in ("value_or_end", "comma_or_end"):
                raise CameraError("unexpected ']'")
            self._stack.pop()
            if frame["role"] == "camk":
                if len(frame["values"]) != K_LEN:
                    raise CameraError(f"cam_K of frame {frame['frame']} has {len(frame['values'])} entries, not 9")
                self.captured[frame["frame"]] = frame["values"]
            self._end_value()
            return
        if ch == ",":
            if frame is None or frame["state"] != "comma_or_end":
                raise CameraError("unexpected ','")
            frame["state"] = "key" if frame["kind"] == "obj" else "value"
            return
        if ch == ":":
            if frame is None or frame["kind"] != "obj" or frame["state"] != "colon":
                raise CameraError("unexpected ':'")
            frame["state"] = "value"
            return
        if ch == '"' and frame is not None and frame["kind"] == "obj" and frame["state"] in ("key_or_end", "key"):
            self._start_string(key=True)
            return
        expecting_value = (frame is None and self._top == "value") or (frame is not None and (
            (frame["kind"] == "obj" and frame["state"] == "value")
            or (frame["kind"] == "arr" and frame["state"] in ("value_or_end", "value"))))
        if not expecting_value:
            raise CameraError(f"unexpected character {ch!r}")
        self._start_value(ch)

    # -- values
    def _role(self) -> tuple[str, int | None]:
        if not self._stack:
            return "top", None
        frame = self._stack[-1]
        if frame["kind"] == "obj":
            if frame["role"] == "top":
                index = self.requested.get(frame["key"])
                return ("frame", index) if index is not None else ("skip", None)
            if frame["role"] == "frame":
                return ("camk", frame["frame"]) if frame["key"] == "cam_K" else ("skip", None)
            return "skip", None
        return ("camk_elem", frame["frame"]) if frame["role"] == "camk" else ("skip", None)

    def _push(self, kind: str, role: str, index: int | None) -> None:
        if len(self._stack) >= MAX_DEPTH:
            raise CameraError(f"camera JSON nests deeper than {MAX_DEPTH}")
        frame = {"kind": kind, "role": role, "frame": index,
                 "state": "key_or_end" if kind == "obj" else "value_or_end"}
        if kind == "obj":
            frame.update(keys=set(), key=None)
        if role == "camk":
            frame["values"] = []
        self._stack.append(frame)
        self.stats["max_depth"] = max(self.stats["max_depth"], len(self._stack))

    def _start_value(self, ch: str) -> None:
        role, index = self._role()
        if ch == "{":
            if role in ("camk", "camk_elem"):
                raise CameraError("cam_K must be a flat array of numbers")
            self._push("obj", role, index)
        elif ch == "[":
            if role == "top":
                raise CameraError("camera JSON top level must be an object keyed by frame id")
            if role == "frame":
                raise CameraError(f"camera entry for frame {index} must be an object")
            if role == "camk_elem":
                raise CameraError("cam_K must be a flat array of numbers")
            self._push("arr", role, index)
        elif ch == '"':
            self._reject_scalar(role, index, "a string")
            self._start_string(key=False)
        elif ch == "-" or ch in DIGITS:
            if role in ("top", "frame", "camk"):
                self._reject_scalar(role, index, "a number")
            self._lex = "num"
            self._num_state = "sign" if ch == "-" else "zero" if ch == "0" else "int"
            self._num_buf = [ch] if role == "camk_elem" else None
        elif ch in LITERALS:
            self._reject_scalar(role, index, "a literal (true/false/null)")
            self._lex, self._lit, self._lit_pos = "lit", LITERALS[ch], 1
        else:
            raise CameraError(f"unexpected character {ch!r}")

    def _reject_scalar(self, role: str, index: int | None, what: str) -> None:
        if role == "top":
            raise CameraError("camera JSON top level must be an object keyed by frame id")
        if role == "frame":
            raise CameraError(f"camera entry for frame {index} must be an object, not {what}")
        if role == "camk":
            raise CameraError(f"cam_K of frame {index} must be an array, not {what}")
        if role == "camk_elem":
            raise CameraError(f"cam_K of frame {index} holds {what}; only finite numbers are allowed")

    def _end_value(self) -> None:
        if not self._stack:
            self._top = "end"
            return
        self._stack[-1]["state"] = "comma_or_end"

    # -- strings
    def _start_string(self, *, key: bool) -> None:
        self._lex, self._str_key, self._str_buf, self._str_len = "str", key, [], 0
        self._esc, self._uni, self._uni_buf, self._high = False, 0, "", None

    def _string_char(self, ch: str) -> None:
        """Escapes decode to the SAME code points as literal UTF-8, so duplicate keys are compared
        decoded: a valid escaped surrogate pair (``\\ud83d\\ude00``) combines into one non-BMP code
        point. A lone or reversed surrogate escape is unsupported and fails closed (keys and values)."""
        if self._uni:
            if ch not in HEX:
                raise CameraError("malformed \\u escape")
            self._uni_buf += ch
            self._uni -= 1
            if not self._uni:
                self._code_unit(int(self._uni_buf, 16))
            return
        if self._esc:
            self._esc = False
            if ch == "u":
                self._uni, self._uni_buf = 4, ""
            elif self._high is not None:
                raise CameraError("unsupported lone high-surrogate escape")
            elif ch in ESCAPES:
                self._key_append(ESCAPES[ch])
            else:
                raise CameraError(f"invalid string escape \\{ch}")
            return
        if ch == "\\":
            self._esc = True
            return
        if self._high is not None:
            raise CameraError("unsupported lone high-surrogate escape")
        if ch == '"':
            self._lex = None
            if self._str_key:
                self._key("".join(self._str_buf))
            else:
                self._end_value()
            return
        if ord(ch) < 0x20:
            raise CameraError("unescaped control character in a string")
        self._key_append(ch)

    def _code_unit(self, unit: int) -> None:
        if self._high is not None:
            if not 0xDC00 <= unit <= 0xDFFF:
                raise CameraError("unsupported lone high-surrogate escape")
            self._key_append(chr(0x10000 + ((self._high - 0xD800) << 10) + (unit - 0xDC00)))
            self._high = None
        elif 0xD800 <= unit <= 0xDBFF:
            self._high = unit
        elif 0xDC00 <= unit <= 0xDFFF:
            raise CameraError("unsupported lone low-surrogate escape")
        else:
            self._key_append(chr(unit))

    def _key_append(self, ch: str) -> None:
        if self._str_key:
            self._str_len += 1
            if self._str_len > MAX_KEY_CHARS:
                raise CameraError("object key is too long")
            self._str_buf.append(ch)

    def _key(self, key: str) -> None:
        frame = self._stack[-1]
        if key in frame["keys"]:
            raise CameraError(f"duplicate JSON key {key[:40]!r}")
        frame["keys"].add(key)
        if frame["role"] == "top":
            if not TOP_KEY.match(key):
                raise CameraError(f"top-level key {key[:40]!r} is not a canonical decimal frame id")
            self.stats["top_level_keys"] += 1
            if key in self.requested:
                self.stats["requested_frames_found"] += 1
        frame["key"], frame["state"] = key, "colon"

    # -- numbers (lexical grammar only; conversion happens later and ONLY for captured cam_K)
    def _number_char(self, ch: str) -> bool:
        s, digit = self._num_state, ch in DIGITS
        if s == "sign":
            nxt = "zero" if ch == "0" else "int" if digit else None
        elif s == "zero":
            nxt = "frac0" if ch == "." else "exp0" if ch in "eE" else None
        elif s == "int":
            nxt = "int" if digit else "frac0" if ch == "." else "exp0" if ch in "eE" else None
        elif s == "frac0":
            nxt = "frac" if digit else None
        elif s == "frac":
            nxt = "frac" if digit else "exp0" if ch in "eE" else None
        elif s == "exp0":
            nxt = "expsign" if ch in "+-" else "exp" if digit else None
        elif s == "expsign":
            nxt = "exp" if digit else None
        else:                                         # exp
            nxt = "exp" if digit else None
        if nxt is None:
            return False                              # caller finishes the number, then re-dispatches ch
        self._num_state = nxt
        if self._num_buf is not None:
            self._num_buf.append(ch)
            if len(self._num_buf) > MAX_CAPTURED_NUMBER_CHARS:
                raise CameraError("cam_K number literal is too long")
        return True

    def _finish_number(self) -> None:
        if self._num_state not in NUMBER_ACCEPT:
            raise CameraError("malformed JSON number")
        self._lex = None
        if self._num_buf is not None:
            frame = self._stack[-1]
            if len(frame["values"]) >= K_LEN:
                raise CameraError(f"cam_K of frame {frame['frame']} has more than 9 entries")
            frame["values"].append("".join(self._num_buf))
            self.stats["numbers_captured"] += 1
        else:
            self.stats["numbers_skipped_unconverted"] += 1
        self._num_buf = None
        self._end_value()

    # -- literals
    def _literal_char(self, ch: str) -> None:
        if ch != self._lit[self._lit_pos]:
            raise CameraError("malformed JSON literal")
        self._lit_pos += 1
        if self._lit_pos == len(self._lit):
            self._lex = None
            self._end_value()


def scan_camera_json(chunks: Iterable[bytes], requested: Iterable[int]) -> dict:
    """Test/inspection API: scan an iterable of byte chunks (any boundaries)."""
    scanner = StructuralScanner(requested)
    for chunk in chunks:
        scanner.feed(chunk)
    return scanner.finish()


def scan_camera_file(path: Path, *, expected_bytes: int, expected_sha256: str, requested: Iterable[int],
                     chunk: int = READ_CHUNK, cap: int = CAMERA_CAP) -> dict:
    """Pass 2: stream the file through the scanner while re-hashing it; any change since pass 1 fails."""
    if type(chunk) is not int or chunk <= 0:
        raise mi.UsageError("chunk must be a positive integer")
    if type(expected_bytes) is not int or not 0 < expected_bytes <= cap:
        raise CameraError("camera size is outside the input cap")
    scanner, h, count = StructuralScanner(requested), hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            count += len(block)
            if count > expected_bytes:
                raise CameraError("camera file grew between passes")
            h.update(block)
            scanner.feed(block)
    if count != expected_bytes or h.hexdigest() != expected_sha256:
        raise CameraError("camera bytes changed between the hash pass and the scan pass")
    result = scanner.finish()
    result["sha256"], result["bytes"] = h.hexdigest(), count
    return result


# --------------------------------------------------------------------------
# K window (pure)


def validate_k_window(captured: Mapping[int, Any], requested: Iterable[int], image_size: tuple[int, int]) -> dict:
    """Exactly 9 finite numbers per requested frame, zero-skew pinhole form, positive fx/fy, principal point
    strictly inside the native image, identical across the whole window. Values are kept raw."""
    indices = check_requested(requested)
    width, height = image_size
    first: tuple[int, tuple[float, ...], list[str]] | None = None
    raw_identical = True
    for index in indices:
        lexemes = captured.get(index)
        if not isinstance(lexemes, list) or len(lexemes) != K_LEN or not all(isinstance(t, str) for t in lexemes):
            raise CameraError(f"cam_K of frame {index} is missing or not 9 numbers")
        values = []
        for text in lexemes:
            value = float(text)
            if not math.isfinite(value):
                raise CameraError(f"cam_K of frame {index} holds a non-finite or overflowing number {text[:24]}")
            values.append(value)
        k = tuple(values)
        if first is None:
            first = (index, k, list(lexemes))
        elif k != first[1]:
            raise CameraError(f"cam_K varies across the window (frame {index} differs from frame {first[0]})")
        elif list(lexemes) != first[2]:
            raw_identical = False
    _, k, lexemes = first
    fx, skew, cx, r1, fy, cy, r2a, r2b, one = k
    if skew != 0 or r1 != 0 or r2a != 0 or r2b != 0 or one != 1:
        raise CameraError("cam_K is not in zero-skew pinhole form [[fx,0,cx],[0,fy,cy],[0,0,1]]")
    if not (fx > 0 and fy > 0):
        raise CameraError("cam_K focal lengths must be positive")
    if not (0 < cx < width and 0 < cy < height):
        raise CameraError("cam_K principal point lies outside the native image")
    return {"values": list(k), "lexemes": list(lexemes), "matrix": [list(k[0:3]), list(k[3:6]), list(k[6:9])],
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "window": {"first": indices[0], "last": indices[-1], "count": len(indices)},
            "constant_across_window": True, "raw_lexemes_identical_across_window": raw_identical}
