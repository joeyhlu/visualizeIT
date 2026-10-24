"""Bounded, immutable, content-addressed stage cache for the video workbench.

Layout under the cache root (all paths in records are POSIX, relative to root)::

    entries/<key>/record.json      canonical JSON record (written last)
    entries/<key>/files/<name>     immutable copies of the stored files
    tmp/<key>.<pid>.<token>/       writer-owned staging directory
    writer.lock/owner-<pid>-<token>  exclusive writer lock (directory + zero-byte marker)

Entries are published by an atomic directory rename while the writer lock is
held. Lookups never write and revalidate every digest, length and path. The
byte cap covers every non-directory object under the root, including staged
copies and the record; the lock adds zero bytes, so cooperating writers never
exceed the cap. There is no automatic eviction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterator, Mapping

__all__ = [
    "CacheBudgetError",
    "CacheCollisionError",
    "CacheError",
    "CacheIntegrityError",
    "CacheLockTimeout",
    "CacheSourceChangedError",
    "CacheValueError",
    "LOCK_BYTES",
    "RECORD_FORMAT",
    "StageCache",
    "canonical_json",
    "hash_file",
    "stage_key",
]

RECORD_FORMAT = "visualizeit.workbench_cache.v1"
KEY_SCHEMA = "visualizeit.workbench_stage_key.v1"
DEFAULT_MAX_BYTES = 1 << 30
DEFAULT_RESERVE_BYTES = 1 << 30
DEFAULT_LOCK_TIMEOUT = 30.0
CHUNK_BYTES = 1 << 20
MAX_RECORD_BYTES = 16 << 20
MAX_NAME_COMPONENT = 128
MAX_NAME_LENGTH = 512

ENTRIES_DIR = "entries"
TEMP_DIR = "tmp"
FILES_DIR = "files"
RECORD_NAME = "record.json"
LOCK_NAME = "writer.lock"

_KEY_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")
_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_RECORD_FIELDS = frozenset(["files", "format", "key", "payload", "payload_sha256", "record_sha256"])
_FILE_FIELDS = frozenset(["path", "sha256", "size"])

_STATE_LOCK = threading.Lock()
_ISSUED_LOCK_TOKENS: set[str] = set()
_ACTIVE_LOCK_TOKENS: set[str] = set()
_CREATED_TEMPS: set[str] = set()
_ACTIVE_TEMPS: set[str] = set()


class CacheError(Exception):
    """Base class for stage-cache failures."""


class CacheValueError(CacheError, ValueError):
    """An argument is invalid, ambiguous or unsafe."""


class CacheIntegrityError(CacheError):
    """A stored entry is corrupt, partial, tampered or unsafe; never a hit."""


class CacheCollisionError(CacheError):
    """A valid entry exists for the key with different payload or file bytes."""


class CacheBudgetError(CacheError):
    """The byte cap or the free-space reserve would be violated."""


class CacheLockTimeout(CacheError):
    """The writer lock could not be acquired within the bounded time."""


class CacheSourceChangedError(CacheError):
    """A source file changed between hashing and copying."""


# --------------------------------------------------------------------------
# Canonical JSON


def _check_json(value: Any, where: str, active: set[int]) -> None:
    kind = type(value)
    if value is None or kind is bool or kind is int:
        return
    if kind is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CacheValueError(f"{where}: string is not valid UTF-8") from exc
        return
    if kind is float:
        if not math.isfinite(value):
            raise CacheValueError(f"{where}: non-finite number")
        return
    if kind is list or kind is dict:
        ident = id(value)
        if ident in active:
            raise CacheValueError(f"{where}: circular reference")
        active.add(ident)
        try:
            if kind is list:
                for index, item in enumerate(value):
                    _check_json(item, f"{where}[{index}]", active)
            else:
                for name, item in value.items():
                    if type(name) is not str:
                        raise CacheValueError(f"{where}: non-string key {name!r} is ambiguous")
                    _check_json(name, f"{where}: key", active)
                    _check_json(item, f"{where}.{name}", active)
        finally:
            active.discard(ident)
        return
    raise CacheValueError(f"{where}: unsupported type {kind.__name__}")


def canonical_json(value: Any) -> bytes:
    """Return canonical UTF-8 JSON (sorted keys, no whitespace, finite only)."""
    _check_json(value, "value", set())
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_canonical(data: bytes, what: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in pairs:
            if name in result:
                raise CacheIntegrityError(f"{what}: duplicate JSON key {name!r}")
            result[name] = item
        return result

    def constant(name: str) -> Any:
        raise CacheIntegrityError(f"{what}: non-finite JSON constant {name}")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant)
        canonical = canonical_json(value)
    except CacheIntegrityError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise CacheIntegrityError(f"{what}: invalid JSON ({exc})") from exc
    if canonical != data:
        raise CacheIntegrityError(f"{what}: JSON is not canonical")
    return value


# --------------------------------------------------------------------------
# Path safety


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _REPARSE_POINT)


def _absolute(path: Any, what: str) -> Path:
    if isinstance(path, (bytes, bytearray)) or not isinstance(path, (str, os.PathLike)):
        raise CacheValueError(f"{what}: expected a filesystem path")
    raw = Path(path)
    if not str(raw) or ".." in raw.parts:
        raise CacheValueError(f"{what}: empty path or parent traversal: {path!s}")
    return Path(os.path.abspath(raw))


def _check_no_links(path: Path, what: str, *, allow_missing: bool) -> bool:
    """Reject symlink/reparse components; return whether the full path exists."""
    parts = path.parts
    for index in range(1, len(parts) + 1):
        current = Path(*parts[:index])
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return False
            raise
        if _is_link(st):
            raise CacheValueError(f"{what}: symlink or reparse point in path: {current}")
    return True


def _source_path(path: Any, what: str) -> Path:
    absolute = _absolute(path, what)
    _check_no_links(absolute, what, allow_missing=False)
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise CacheValueError(f"{what}: not a regular file: {absolute}")
    return absolute


def _open_regular(path: Path, what: str, error: type[CacheError] = CacheValueError):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise error(f"{what}: not a regular file: {path}")
        return os.fdopen(fd, "rb", buffering=0)
    except BaseException:
        os.close(fd)
        raise


def _hash_stream(handle, limit: int | None, what: str, error: type[CacheError]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = handle.read(CHUNK_BYTES)
        if not chunk:
            return digest.hexdigest(), size
        size += len(chunk)
        if limit is not None and size > limit:
            raise error(f"{what}: longer than {limit} bytes")
        digest.update(chunk)


def hash_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 of a regular, symlink-free file."""
    source = _source_path(path, "hash_file")
    with _open_regular(source, "hash_file") as handle:
        return _hash_stream(handle, None, "hash_file", CacheValueError)[0]


def _validate_name(name: Any) -> tuple[str, ...]:
    if type(name) is not str or not name:
        raise CacheValueError(f"file name must be a non-empty string: {name!r}")
    if len(name) > MAX_NAME_LENGTH:
        raise CacheValueError(f"file name too long: {name[:40]!r}...")
    if "\\" in name or ":" in name or "\x00" in name or name.startswith("/"):
        raise CacheValueError(f"file name must be a portable relative POSIX path: {name!r}")
    parts = tuple(name.split("/"))
    for part in parts:
        if part in ("", ".", ".."):
            raise CacheValueError(f"file name has an empty or traversal component: {name!r}")
        if len(part) > MAX_NAME_COMPONENT or not _COMPONENT_RE.match(part) or part.endswith("."):
            raise CacheValueError(f"file name component is not portable: {name!r}")
        if part.split(".", 1)[0].upper() in _RESERVED_NAMES:
            raise CacheValueError(f"file name uses a reserved device name: {name!r}")
    return parts


def _validate_names(names: list[str]) -> dict[str, tuple[str, ...]]:
    parsed = {name: _validate_name(name) for name in names}
    folded_files: dict[str, str] = {}
    folded_dirs: set[str] = set()
    for name, parts in parsed.items():
        folded = name.casefold()
        if folded in folded_files:
            raise CacheValueError(f"file names differ only by case: {folded_files[folded]!r}, {name!r}")
        folded_files[folded] = name
        for index in range(1, len(parts)):
            folded_dirs.add("/".join(parts[:index]).casefold())
    clash = sorted(set(folded_files) & folded_dirs)
    if clash:
        raise CacheValueError(f"file name is also used as a directory: {folded_files[clash[0]]!r}")
    return parsed


def _validate_key(key: Any) -> str:
    if type(key) is not str or not _KEY_RE.match(key):
        raise CacheValueError("cache key must be 64 lowercase hexadecimal characters")
    return key


def stage_key(
    stage: str,
    inputs: Mapping[str, Any],
    configuration: Mapping[str, Any],
    versions: Mapping[str, Any],
) -> str:
    """Return the SHA-256 of the canonical JSON description of a stage run."""
    if type(stage) is not str or not stage or stage.strip() != stage:
        raise CacheValueError("stage must be a non-empty string without surrounding whitespace")
    for name, value in (("inputs", inputs), ("configuration", configuration), ("versions", versions)):
        if type(value) is not dict:
            raise CacheValueError(f"{name} must be a dict")
    document = {
        "configuration": configuration,
        "inputs": inputs,
        "schema": KEY_SCHEMA,
        "stage": stage,
        "versions": versions,
    }
    return _sha256(canonical_json(document))


# --------------------------------------------------------------------------
# Records


@dataclass(frozen=True)
class _Source:
    name: str
    parts: tuple[str, ...]
    sha256: str
    size: int
    data: bytes | None = None
    path: Path | None = None


def _collect_sources(files: Any) -> list[_Source]:
    if files is None:
        return []
    if type(files) is not dict:
        raise CacheValueError("files must be a dict of relative name -> Path or bytes")
    parsed = _validate_names(list(files))
    sources = []
    for name in sorted(files):
        value = files[name]
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            sources.append(_Source(name, parsed[name], _sha256(data), len(data), data=data))
        elif isinstance(value, PurePath):
            path = _source_path(value, f"files[{name!r}]")
            with _open_regular(path, f"files[{name!r}]") as handle:
                digest, size = _hash_stream(handle, None, f"files[{name!r}]", CacheValueError)
            sources.append(_Source(name, parsed[name], digest, size, path=path))
        else:
            raise CacheValueError(f"files[{name!r}] must be a pathlib.Path or bytes, not {type(value).__name__}")
    return sources


def _relative_file_path(key: str, name: str) -> str:
    return f"{ENTRIES_DIR}/{key}/{FILES_DIR}/{name}"


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    record = dict(body)
    record["record_sha256"] = _sha256(canonical_json(body))
    return record


def _build_record(key: str, payload: dict[str, Any], sources: list[_Source]) -> dict[str, Any]:
    files = {
        source.name: {"path": _relative_file_path(key, source.name), "sha256": source.sha256, "size": source.size}
        for source in sources
    }
    return _seal(
        {
            "files": files,
            "format": RECORD_FORMAT,
            "key": key,
            "payload": payload,
            "payload_sha256": _sha256(canonical_json(payload)),
        }
    )


def _validate_record(record: Any, key: str) -> dict[str, tuple[str, ...]]:
    what = f"entry {key}"
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        raise CacheIntegrityError(f"{what}: record fields are not {sorted(_RECORD_FIELDS)}")
    if record["format"] != RECORD_FORMAT:
        raise CacheIntegrityError(f"{what}: unsupported or legacy record format")
    if record["key"] != key:
        raise CacheIntegrityError(f"{what}: record key does not match its directory")
    payload = record["payload"]
    if type(payload) is not dict or record["payload_sha256"] != _sha256(canonical_json(payload)):
        raise CacheIntegrityError(f"{what}: payload digest mismatch")
    files = record["files"]
    if type(files) is not dict:
        raise CacheIntegrityError(f"{what}: files manifest is not an object")
    try:
        parsed = _validate_names(list(files))
    except CacheValueError as exc:
        raise CacheIntegrityError(f"{what}: unsafe manifest name ({exc})") from exc
    for name, meta in files.items():
        if type(meta) is not dict or set(meta) != _FILE_FIELDS:
            raise CacheIntegrityError(f"{what}: malformed manifest entry {name!r}")
        if type(meta["sha256"]) is not str or not _KEY_RE.match(meta["sha256"]):
            raise CacheIntegrityError(f"{what}: malformed digest for {name!r}")
        if type(meta["size"]) is not int or meta["size"] < 0:
            raise CacheIntegrityError(f"{what}: malformed size for {name!r}")
        if meta["path"] != _relative_file_path(key, name):
            raise CacheIntegrityError(f"{what}: stored path for {name!r} is not the canonical entry path")
    body = {name: value for name, value in record.items() if name != "record_sha256"}
    if record["record_sha256"] != _sha256(canonical_json(body)):
        raise CacheIntegrityError(f"{what}: record digest mismatch")
    return parsed


# --------------------------------------------------------------------------
# Filesystem helpers


def _tree_bytes(path: Path) -> int:
    """Sum the sizes of every non-directory entry below ``path`` without following links."""
    total = 0
    try:
        scanner = _retry_transient(os.scandir, path)
    except FileNotFoundError:
        return 0
    with scanner:
        for entry in scanner:
            try:
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(st.st_mode) and not _is_link(st):
                total += _tree_bytes(Path(entry.path))
            else:
                total += st.st_size
    return total


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _within(path: Path, parent: Path) -> bool:
    child = os.path.normcase(os.path.realpath(path))
    base = os.path.normcase(os.path.realpath(parent))
    return child.startswith(base.rstrip(os.sep) + os.sep)


def _owner_name(pid: int, token: str) -> str:
    return f"owner-{pid:020d}-{token}"


# The lock is a directory holding one zero-byte owner marker: it adds no bytes.
LOCK_BYTES = 0
TRANSIENT_RETRY_SECONDS = 5.0


def _retry_transient(action, *args):
    """Retry Windows sharing violations / pending-delete denials for a bounded time."""
    deadline = time.monotonic() + TRANSIENT_RETRY_SECONDS
    delay = 0.002
    while True:
        try:
            return action(*args)
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.05)


# --------------------------------------------------------------------------
# Cache


class StageCache:
    """Bounded immutable content-addressed cache rooted at ``root``."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        reserve_bytes: int = DEFAULT_RESERVE_BYTES,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> None:
        for name, value in (("max_bytes", max_bytes), ("reserve_bytes", reserve_bytes)):
            if type(value) is not int or value < 0:
                raise CacheValueError(f"{name} must be a non-negative int")
        if type(lock_timeout) not in (int, float) or not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise CacheValueError("lock_timeout must be a finite non-negative number")
        self._root = _absolute(root, "cache root")
        if _check_no_links(self._root, "cache root", allow_missing=True) and not self._root.is_dir():
            raise CacheValueError(f"cache root is not a directory: {self._root}")
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes
        self.lock_timeout = float(lock_timeout)

    @property
    def root(self) -> Path:
        return self._root

    def usage_bytes(self) -> int:
        """Return the bytes of every file currently under the cache root."""
        return _tree_bytes(self._root)

    # ---------------------------------------------------------------- lookup

    def lookup(self, key: str) -> dict[str, Any] | None:
        """Return the revalidated record for ``key`` or ``None`` if absent.

        Never writes. Corrupt, partial or unsafe entries raise
        :class:`CacheIntegrityError`.
        """
        return self._load_entry(_validate_key(key))

    def _check_root(self) -> bool:
        try:
            exists = _check_no_links(self._root, "cache root", allow_missing=True)
        except CacheValueError as exc:
            raise CacheIntegrityError(str(exc)) from exc
        if exists and not stat.S_ISDIR(os.lstat(self._root).st_mode):
            raise CacheIntegrityError(f"cache root is not a directory: {self._root}")
        return exists

    def _safe_dir(self, path: Path, what: str) -> bool:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return False
        if _is_link(st) or not stat.S_ISDIR(st.st_mode):
            raise CacheIntegrityError(f"{what} is not a plain directory: {path}")
        return True

    def _load_entry(self, key: str) -> dict[str, Any] | None:
        if not self._check_root():
            return None
        entries = self._root / ENTRIES_DIR
        if not self._safe_dir(entries, "entries directory"):
            return None
        entry = entries / key
        if not self._safe_dir(entry, f"entry {key}"):
            return None
        what = f"entry {key}"
        if not _within(entry, self._root):
            raise CacheIntegrityError(f"{what}: resolves outside the cache root")
        children = set(os.listdir(entry))
        if children != {RECORD_NAME, FILES_DIR}:
            raise CacheIntegrityError(f"{what}: partial or unexpected contents {sorted(children)}")
        record_path = entry / RECORD_NAME
        record_stat = os.lstat(record_path)
        if _is_link(record_stat) or not stat.S_ISREG(record_stat.st_mode):
            raise CacheIntegrityError(f"{what}: record is not a regular file")
        with _open_regular(record_path, what, CacheIntegrityError) as handle:
            data = handle.read(MAX_RECORD_BYTES + 1)
        if len(data) > MAX_RECORD_BYTES:
            raise CacheIntegrityError(f"{what}: record exceeds {MAX_RECORD_BYTES} bytes")
        record = _parse_canonical(data, what)
        names = _validate_record(record, key)

        files_dir = entry / FILES_DIR
        if not self._safe_dir(files_dir, f"{what} files directory"):
            raise CacheIntegrityError(f"{what}: files directory missing")
        found, dirs = self._scan_files(files_dir, (), what)
        if found != set(names):
            missing = sorted(set(names) - found)
            extra = sorted(found - set(names))
            raise CacheIntegrityError(f"{what}: manifest mismatch (missing={missing}, extra={extra})")
        allowed_dirs = {"/".join(parts[:i]) for parts in names.values() for i in range(1, len(parts))}
        if dirs - allowed_dirs:
            raise CacheIntegrityError(f"{what}: unexpected directories {sorted(dirs - allowed_dirs)}")
        for name, parts in names.items():
            meta = record["files"][name]
            path = files_dir.joinpath(*parts)
            if not _within(path, files_dir):
                raise CacheIntegrityError(f"{what}: file {name!r} resolves outside the entry")
            with _open_regular(path, f"{what} file {name!r}", CacheIntegrityError) as handle:
                digest, size = _hash_stream(handle, meta["size"], f"{what} file {name!r}", CacheIntegrityError)
            if size != meta["size"] or digest != meta["sha256"]:
                raise CacheIntegrityError(f"{what}: file {name!r} digest or length mismatch")
        return record

    def _scan_files(self, directory: Path, prefix: tuple[str, ...], what: str) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        dirs: set[str] = set()
        with os.scandir(directory) as scanner:
            for item in scanner:
                st = item.stat(follow_symlinks=False)
                rel = prefix + (item.name,)
                if _is_link(st):
                    raise CacheIntegrityError(f"{what}: link inside entry: {'/'.join(rel)}")
                if stat.S_ISDIR(st.st_mode):
                    dirs.add("/".join(rel))
                    sub_files, sub_dirs = self._scan_files(Path(item.path), rel, what)
                    files |= sub_files
                    dirs |= sub_dirs
                elif stat.S_ISREG(st.st_mode):
                    files.add("/".join(rel))
                else:
                    raise CacheIntegrityError(f"{what}: special file inside entry: {'/'.join(rel)}")
        return files, dirs

    # ----------------------------------------------------------------- store

    def store(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        files: dict[str, Path | bytes] | None = None,
    ) -> dict[str, Any]:
        """Atomically publish an immutable entry and return its record.

        If a valid entry already exists it is returned only when payload and
        file bytes are identical; otherwise :class:`CacheCollisionError`.
        """
        key = _validate_key(key)
        if type(payload) is not dict:
            raise CacheValueError("payload must be a dict")
        canonical_json(payload)
        sources = _collect_sources(files)
        record = _build_record(key, payload, sources)
        record_bytes = canonical_json(record)
        if len(record_bytes) > MAX_RECORD_BYTES:
            raise CacheValueError(f"record exceeds {MAX_RECORD_BYTES} bytes")

        self._prepare_root()
        with self._writer_lock():
            self._cleanup_own_stale_temps()
            existing = self._load_entry(key)
            if existing is not None:
                if canonical_json(existing) == record_bytes:
                    return existing
                raise CacheCollisionError(f"entry {key} exists with different payload or file bytes")
            planned = sum(source.size for source in sources) + len(record_bytes)
            self._check_budget(planned)
            temp = self._make_temp(key)
            try:
                self._stage(temp, sources, record_bytes, planned)
                usage = self.usage_bytes()
                if usage > self.max_bytes:
                    raise CacheBudgetError(f"cache usage {usage} exceeds max_bytes {self.max_bytes}")
                self._publish(temp, self._root / ENTRIES_DIR / key)
            except BaseException:
                self._remove_own_temp(temp)
                raise
            with _STATE_LOCK:
                _ACTIVE_TEMPS.discard(str(temp))
                _CREATED_TEMPS.discard(str(temp))
            stored = self._load_entry(key)
        if stored is None or canonical_json(stored) != record_bytes:
            raise CacheIntegrityError(f"entry {key}: published record does not verify")
        return stored

    def _prepare_root(self) -> None:
        _check_no_links(self._root, "cache root", allow_missing=True)
        self._root.mkdir(parents=True, exist_ok=True)
        _check_no_links(self._root, "cache root", allow_missing=False)
        if not stat.S_ISDIR(os.lstat(self._root).st_mode):
            raise CacheValueError(f"cache root is not a directory: {self._root}")
        for name in (ENTRIES_DIR, TEMP_DIR):
            path = self._root / name
            path.mkdir(exist_ok=True)
            if not self._safe_dir(path, name):
                raise CacheIntegrityError(f"{name} directory vanished")

    def _check_budget(self, planned: int) -> None:
        usage = self.usage_bytes()
        if usage + planned > self.max_bytes:
            raise CacheBudgetError(
                f"cache would hold {usage + planned} bytes (usage {usage} + new {planned}) > max_bytes {self.max_bytes}"
            )
        self._check_reserve(planned)

    def _check_reserve(self, remaining: int) -> None:
        free = shutil.disk_usage(self._root).free
        if free - remaining < self.reserve_bytes:
            raise CacheBudgetError(
                f"writing {remaining} more bytes would leave {free - remaining} free < reserve {self.reserve_bytes}"
            )

    def _make_temp(self, key: str) -> Path:
        temp = self._root / TEMP_DIR / f"{key}.{os.getpid()}.{secrets.token_hex(8)}"
        temp.mkdir()
        with _STATE_LOCK:
            _CREATED_TEMPS.add(str(temp))
            _ACTIVE_TEMPS.add(str(temp))
        return temp

    def _stage(self, temp: Path, sources: list[_Source], record_bytes: bytes, planned: int) -> None:
        written = 0
        files_dir = temp / FILES_DIR
        files_dir.mkdir()
        for source in sources:
            target = files_dir.joinpath(*source.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not _within(target, temp):
                raise CacheValueError(f"file {source.name!r} escapes the staging directory")
            digest = hashlib.sha256()
            size = 0
            with self._create(target) as out:
                if source.data is not None:
                    chunks: Iterator[bytes] = (
                        source.data[i : i + CHUNK_BYTES] for i in range(0, len(source.data), CHUNK_BYTES)
                    )
                    written, size = self._write_chunks(out, chunks, digest, written, planned, source.size, source.name)
                else:
                    assert source.path is not None
                    _check_no_links(source.path, f"files[{source.name!r}]", allow_missing=False)
                    with _open_regular(source.path, f"files[{source.name!r}]") as handle:
                        chunks = iter(lambda: handle.read(CHUNK_BYTES), b"")
                        written, size = self._write_chunks(
                            out, chunks, digest, written, planned, source.size, source.name
                        )
                out.flush()
                os.fsync(out.fileno())
            if size != source.size or digest.hexdigest() != source.sha256:
                raise CacheSourceChangedError(f"source for {source.name!r} changed while it was being stored")
        with self._create(temp / RECORD_NAME) as out:
            chunks = (record_bytes[i : i + CHUNK_BYTES] for i in range(0, len(record_bytes), CHUNK_BYTES))
            written, _ = self._write_chunks(
                out, chunks, hashlib.sha256(), written, planned, len(record_bytes), RECORD_NAME
            )
            out.flush()
            os.fsync(out.fileno())
        _fsync_dir(temp)

    def _write_chunks(self, out, chunks, digest, written: int, planned: int, limit: int, name: str) -> tuple[int, int]:
        size = 0
        for chunk in chunks:
            size += len(chunk)
            if size > limit:
                raise CacheSourceChangedError(f"source for {name!r} grew while it was being stored")
            self._check_reserve(planned - written)
            out.write(chunk)
            digest.update(chunk)
            written += len(chunk)
        return written, size

    @staticmethod
    def _create(path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.fdopen(os.open(path, flags, 0o644), "wb")

    def _publish(self, temp: Path, entry: Path) -> None:
        if os.path.lexists(entry):
            raise CacheCollisionError(f"entry appeared during store: {entry.name}")
        _retry_transient(os.rename, temp, entry)
        _fsync_dir(entry.parent)

    def _remove_own_temp(self, temp: Path) -> None:
        """Remove a staging/released directory this process created and currently claims."""
        key = str(temp)
        with _STATE_LOCK:
            if key not in _CREATED_TEMPS:
                return
        try:
            _retry_transient(shutil.rmtree, temp)
            removed = True
        except FileNotFoundError:
            removed = True
        except OSError:
            removed = False  # Kept in _CREATED_TEMPS so a later writer can retry.
        with _STATE_LOCK:
            _ACTIVE_TEMPS.discard(key)
            if removed:
                _CREATED_TEMPS.discard(key)

    def _cleanup_own_stale_temps(self) -> None:
        temp_root = os.path.normcase(str(self._root / TEMP_DIR))
        with _STATE_LOCK:
            stale = [
                path
                for path in _CREATED_TEMPS
                if path not in _ACTIVE_TEMPS and os.path.normcase(os.path.dirname(path)) == temp_root
            ]
            _ACTIVE_TEMPS.update(stale)  # Claim so no other thread removes them concurrently.
        for path in stale:
            self._remove_own_temp(Path(path))

    # ------------------------------------------------------------------ lock
    #
    # The writer lock is a directory created with an exclusive ``mkdir``; its
    # only content is a zero-byte owner marker. It therefore adds no bytes to
    # the cap, and waiting writers never open it (an open handle without
    # FILE_SHARE_DELETE made the holder's unlink fail with WinError 32).
    # Release renames the whole directory away atomically, then deletes it.

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        lock = self._root / LOCK_NAME
        token = secrets.token_hex(16)
        with _STATE_LOCK:
            _ISSUED_LOCK_TOKENS.add(token)
            _ACTIVE_LOCK_TOKENS.add(token)
        try:
            self._check_reserve(0)
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    os.mkdir(lock)
                    break
                except (FileExistsError, PermissionError) as exc:
                    # PermissionError: Windows reports a lock directory that is
                    # still pending deletion as access denied.
                    if self._reclaim_own_stale_lock(lock):
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CacheLockTimeout(f"writer lock busy after {self.lock_timeout}s: {lock}") from exc
                    time.sleep(min(0.01, remaining))
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
                os.close(os.open(lock / _owner_name(os.getpid(), token), flags, 0o644))
            except BaseException:
                _retry_transient(os.rmdir, lock)  # Created by this call's exclusive mkdir and still empty.
                raise
        except BaseException:
            with _STATE_LOCK:
                _ISSUED_LOCK_TOKENS.discard(token)
                _ACTIVE_LOCK_TOKENS.discard(token)
            raise
        try:
            yield
        finally:
            try:
                self._release_lock(lock, token)
            finally:
                with _STATE_LOCK:
                    _ACTIVE_LOCK_TOKENS.discard(token)

    def _release_lock(self, lock: Path, token: str) -> bool:
        """Release the lock owned by ``token`` of this process; warn if impossible."""
        pid = os.getpid()
        if not os.path.lexists(lock / _owner_name(pid, token)):
            with _STATE_LOCK:
                _ISSUED_LOCK_TOKENS.discard(token)
            warnings.warn(f"cache lock {lock} no longer holds this writer's owner marker", RuntimeWarning, stacklevel=3)
            return False
        released = self._root / TEMP_DIR / f"released-lock.{pid}.{token}"
        with _STATE_LOCK:
            _CREATED_TEMPS.add(str(released))
            _ACTIVE_TEMPS.add(str(released))
        try:
            _retry_transient(os.rename, lock, released)
        except OSError as exc:
            with _STATE_LOCK:
                _ACTIVE_TEMPS.discard(str(released))
                _CREATED_TEMPS.discard(str(released))
            # The token stays issued so this process can reclaim the lock later.
            warnings.warn(f"could not release cache lock {lock}: {exc}", RuntimeWarning, stacklevel=3)
            return False
        with _STATE_LOCK:
            _ISSUED_LOCK_TOKENS.discard(token)
        self._remove_own_temp(released)
        return True

    def _reclaim_own_stale_lock(self, lock: Path) -> bool:
        """Release a lock left by this process with an inactive token; never guess others."""
        pid = os.getpid()
        with _STATE_LOCK:
            stale = [token for token in _ISSUED_LOCK_TOKENS if token not in _ACTIVE_LOCK_TOKENS]
            _ACTIVE_LOCK_TOKENS.update(stale)  # Claim so only one thread acts on each token.
        try:
            for token in stale:
                if os.path.lexists(lock / _owner_name(pid, token)):
                    return self._release_lock(lock, token)
            return False
        finally:
            with _STATE_LOCK:
                _ACTIVE_LOCK_TOKENS.difference_update(stale)
