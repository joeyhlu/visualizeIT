"""Checked RGBA PNG evidence writer.

An overlay is registered only after its PNG has been encoded, written to an
exclusive owned temp file, read back, decoded pixel-exactly equal to the input,
and published at a path that did not exist before. Any failure raises
``ImageWriteError``; callers then record the failure instead of a path.

Channel convention: callers pass RGBA uint8; OpenCV encodes BGRA, so the
writer swaps to BGRA before encoding and the reader swaps back to RGBA.
No model runtime is imported here (OpenCV is imported lazily).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import uuid

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ImageWriteError(RuntimeError):
    """An RGBA PNG could not be encoded, written, verified or published."""


def check_rgba(rgba) -> np.ndarray:
    """Return a contiguous HxWx4 uint8 array or raise ``ImageWriteError``."""
    if not isinstance(rgba, np.ndarray):
        raise ImageWriteError("overlay must be a numpy array")
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[0] < 1 or rgba.shape[1] < 1:
        raise ImageWriteError(f"overlay must be HxWx4 uint8 RGBA (got {rgba.dtype} {rgba.shape})")
    return np.ascontiguousarray(rgba)


def encode_png_rgba(rgba) -> bytes:
    from .vision import cv2
    arr = check_rgba(rgba)
    try:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))
    except Exception as exc:  # noqa: BLE001 - cv2.error and friends become one typed failure
        raise ImageWriteError(f"PNG encode raised {type(exc).__name__}: {exc}") from None
    if not ok or buf is None:
        raise ImageWriteError("PNG encode returned failure")
    data = buf.tobytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ImageWriteError("PNG encode produced no PNG data")
    return data


def decode_png_rgba(data: bytes, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Strict decode: PNG signature, 8-bit, exactly four channels; returns RGBA."""
    from .vision import cv2
    if not isinstance(data, (bytes, bytearray)) or not bytes(data[:8]) == PNG_SIGNATURE:
        raise ImageWriteError("not PNG data")
    try:
        img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # noqa: BLE001
        raise ImageWriteError(f"PNG decode raised {type(exc).__name__}: {exc}") from None
    if img is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 4:
        raise ImageWriteError("PNG is not 8-bit four-channel RGBA")
    if expected_shape is not None and tuple(img.shape[:2]) != tuple(expected_shape):
        raise ImageWriteError(f"PNG size {img.shape[1]}x{img.shape[0]} differs from the expected frame size")
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def inspect_png_rgba(data: bytes, *, expected_shape: tuple[int, int] | None = None) -> dict:
    """Strict RGBA decode plus the alpha summary used to cross-check render stats."""
    rgba = decode_png_rgba(data, expected_shape=expected_shape)
    alpha = rgba[:, :, 3]
    return {"width": int(rgba.shape[1]), "height": int(rgba.shape[0]),
            "alpha_binary": bool(np.isin(alpha, (0, 255)).all()), "alpha_on_pixels": int(np.count_nonzero(alpha == 255)),
            "sha256": hashlib.sha256(bytes(data)).hexdigest(), "bytes": len(data)}


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise ImageWriteError("short write")
        view = view[n:]


def _same_owned_file(target: Path, tmp: Path) -> bool:
    """True only if ``target`` is still a regular file with the owned temp's identity (our hard link)."""
    try:
        t, o = os.lstat(target), os.lstat(tmp)
    except OSError:
        return False
    return stat.S_ISREG(t.st_mode) and t.st_ino != 0 and os.path.samestat(t, o)


def _unpublish(target: Path, tmp: Path) -> str:
    """Remove ONLY our own final link; an externally replaced or pre-existing file is never touched."""
    if not _same_owned_file(target, tmp):
        return "final path left untouched (not provably this writer's link)"
    try:
        os.unlink(target)
    except OSError as exc:
        return f"own final link could not be removed ({type(exc).__name__}: {exc})"
    return "own final link removed"


def publish_new_bytes(path: Path, data: bytes, *, check=None) -> dict:
    """Publish ``data`` at a path that must not exist; never overwrites.

    The bytes go to an exclusive owned temp file in the same directory (checked
    writes, flush, fsync), are read back (and passed to ``check`` if given),
    then hard-linked to ``path``, which fails if ``path`` appeared meanwhile.
    The owned temp is kept until the published path has been re-read, compared
    and confirmed to be the same file. If that post-publication check fails,
    only our own final link is removed (same file identity as the owned temp),
    and ``ImageWriteError`` is raised. The owned temp is always removed last.
    Returns ``{"path", "sha256", "bytes"}`` only after full verification.
    """
    data = bytes(data)
    target = Path(path)
    if os.path.lexists(target):
        raise ImageWriteError(f"{target.name} already exists; evidence is never overwritten")
    if not target.parent.is_dir():
        raise ImageWriteError(f"parent directory of {target.name} does not exist")
    tmp = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    owned = False
    try:
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        except OSError as exc:
            raise ImageWriteError(f"temp file could not be created: {type(exc).__name__}: {exc}") from None
        owned = True
        try:
            _write_all(fd, data)
            os.fsync(fd)
        except OSError as exc:
            raise ImageWriteError(f"temp write failed: {type(exc).__name__}: {exc}") from None
        finally:
            os.close(fd)
        try:
            back = tmp.read_bytes()
        except OSError as exc:
            raise ImageWriteError(f"temp read-back failed: {type(exc).__name__}: {exc}") from None
        if back != data:
            raise ImageWriteError("temp read-back differs from the bytes written")
        if check is not None:
            check(back)
        try:
            os.link(tmp, target)
        except FileExistsError:
            raise ImageWriteError(f"{target.name} appeared concurrently; not overwritten") from None
        except OSError as exc:
            raise ImageWriteError(f"publication failed: {type(exc).__name__}: {exc}") from None
        # Published. Verify while the owned temp still pins the file identity.
        try:
            if not _same_owned_file(target, tmp):
                raise ImageWriteError("published path is not this writer's file")
            try:
                final = target.read_bytes()
            except OSError as exc:
                raise ImageWriteError(f"published file unreadable: {type(exc).__name__}: {exc}") from None
            if final != data:
                raise ImageWriteError("published file differs from the bytes written")
        except BaseException as exc:
            outcome = _unpublish(target, tmp)
            if isinstance(exc, ImageWriteError):
                raise ImageWriteError(f"{exc}; {outcome}") from None
            if isinstance(exc, Exception):
                raise ImageWriteError(f"post-publication check raised {type(exc).__name__}: {exc}; {outcome}") from None
            raise
    finally:
        if owned:
            try:
                tmp.unlink()
            except OSError:
                pass  # a leftover hidden temp is never registered as evidence
    return {"path": str(target), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_png_rgba(path: Path, rgba, *, max_bytes: int | None = None) -> dict:
    """Publish ``rgba`` as a new PNG at ``path``; never overwrites.

    The encoding must decode pixel-exactly to ``rgba`` before and after the
    temp write. Returns ``{"path", "sha256", "bytes"}`` only after publication.
    """
    arr = check_rgba(rgba)
    data = encode_png_rgba(arr)
    if max_bytes is not None and len(data) > int(max_bytes):
        raise ImageWriteError(f"encoded PNG is {len(data)} bytes, above the {int(max_bytes)} byte cap")

    def exact(raw: bytes) -> None:
        if not np.array_equal(decode_png_rgba(raw, expected_shape=arr.shape[:2]), arr):
            raise ImageWriteError("PNG does not decode to the input pixels")

    exact(data)
    return publish_new_bytes(path, data, check=exact)
