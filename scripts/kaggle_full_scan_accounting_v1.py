"""Faster full-scan accounting for the authenticated Kaggle facade.

Every call still walks the complete retained tree. POSIX uses ``scandir``
with directory-relative, no-follow descriptors; other platforms delegate to
the pinned immutable adapter implementation.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import stat

from scripts.kaggle_supervision_v1 import Facade as _PinnedFacade


_POSIX_NOFOLLOW = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.scandir in os.supports_fd
)


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory(path: Path) -> int:
    """Open every component from / without following a replaced symlink."""
    absolute = Path(os.path.abspath(path))
    fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, _directory_flags(), dir_fd=fd)
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise FileNotFoundError(errno.ENOENT, "path changed during full scan", str(path)) from error
                raise
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_child_directory(parent_fd: int, name: str, path: Path, expected) -> int:
    try:
        fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise FileNotFoundError(errno.ENOENT, "directory entry changed during full scan", str(path)) from error
        raise
    try:
        observed = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise
    if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
        os.close(fd)
        raise FileNotFoundError(errno.ENOENT, "directory entry changed during full scan", str(path))
    return fd


def _inventory_directory(fd: int, directory: Path, observed: set[Path]) -> None:
    """Collect every symlink without following it; caller owns ``fd``."""
    try:
        with os.scandir(fd) as entries:
            for entry in entries:
                path = directory / entry.name
                if entry.is_symlink():
                    observed.add(path)
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                info = entry.stat(follow_symlinks=False)
                child_fd = _open_child_directory(fd, entry.name, path, info)
                _inventory_directory(child_fd, path, observed)
    finally:
        os.close(fd)


def _runtime_link_policy_once(adapter, receipt: dict, working: Path) -> dict[Path, Path]:
    root = adapter.contained(Path(receipt["runtime_root"]), working)
    if root == working or not root.is_dir():
        raise adapter.ContractError("dedicated authenticated runtime directory required")
    entries = receipt.get("runtime_links")
    if not isinstance(entries, list):
        raise adapter.ContractError("bootstrap runtime link inventory required")
    policy: dict[Path, Path] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "target"}:
            raise adapter.ContractError("runtime link requires path and exact target")
        adapter.safe_member(item["path"])
        path = root / item["path"]
        adapter.no_symlinks(path.parent)
        if path in policy or not path.is_symlink() or os.readlink(path) != item["target"]:
            raise adapter.ContractError(f"runtime link inventory changed: {path}")
        lexical = Path(os.path.abspath(path.parent / item["target"]))
        if not lexical.is_relative_to(root):
            raise adapter.ContractError(f"runtime link escapes runtime root: {path}")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise adapter.ContractError(f"broken/cyclic runtime link: {path}") from error
        if not resolved.is_relative_to(root) or (resolved.is_dir() and path.parent.is_relative_to(resolved)):
            raise adapter.ContractError(f"escaping/cyclic runtime directory alias: {path}")
        policy[path] = resolved

    observed: set[Path] = set()
    _inventory_directory(_open_directory(root), root, observed)
    if observed != set(policy):
        raise adapter.ContractError("unapproved runtime links appeared or authenticated links disappeared")
    return policy


def _scan_tree_bytes(adapter, roots: tuple[Path, ...], approved_links: dict[Path, Path] | None) -> int:
    seen_files: set[tuple[int, int]] = set()
    seen_directories: set[tuple[int, int]] = set()
    approved = approved_links or {}
    total = 0

    def scan_open_directory(fd: int, logical_path: Path) -> None:
        nonlocal total
        try:
            directory_info = os.fstat(fd)
            identity = directory_info.st_dev, directory_info.st_ino
            if identity in seen_directories:
                return
            seen_directories.add(identity)
            with os.scandir(fd) as entries:
                for entry in entries:
                    visit_entry(fd, entry, logical_path / entry.name)
        finally:
            os.close(fd)

    def account_info(path: Path, parent_fd: int, name: str, info) -> None:
        nonlocal total
        if stat.S_ISDIR(info.st_mode):
            child_fd = _open_child_directory(parent_fd, name, path, info)
            scan_open_directory(child_fd, path)
        elif stat.S_ISREG(info.st_mode):
            identity = info.st_dev, info.st_ino
            if identity not in seen_files:
                seen_files.add(identity)
                total += info.st_size
        else:
            raise adapter.ContractError(f"nonregular resident path: {path}")

    def visit_path(path: Path) -> None:
        absolute = Path(os.path.abspath(path))
        if absolute == Path(os.sep):
            root_fd = _open_directory(absolute)
            root_info = os.fstat(root_fd)
            scan_open_directory(root_fd, path)
            return
        parent_fd = _open_directory(absolute.parent)
        try:
            info = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                visit_link(path)
            else:
                account_info(path, parent_fd, absolute.name, info)
        finally:
            os.close(parent_fd)

    def visit_link(path: Path) -> None:
        if path not in approved or path.resolve(strict=True) != approved[path]:
            raise adapter.ContractError(f"unapproved accounting symlink: {path}")
        visit_path(approved[path])

    def visit_entry(parent_fd: int, entry, path: Path) -> None:
        if entry.is_symlink():
            visit_link(path)
            return
        info = entry.stat(follow_symlinks=False)
        # lstat catches a replacement that happened after DirEntry's cached type check.
        if stat.S_ISLNK(info.st_mode):
            visit_link(path)
        else:
            account_info(path, parent_fd, entry.name, info)

    for root in roots:
        adapter.no_symlinks(root)
        if root.exists():
            visit_path(root)
    return total


class FullScanFacade(_PinnedFacade):
    """Drop-in supervision facade with faster, still-complete accounting scans."""

    def runtime_link_policy(self, receipt: dict, working: Path) -> dict[Path, Path]:
        if not _POSIX_NOFOLLOW:
            return self.adapter.runtime_link_policy(receipt, working)
        for attempt in range(3):
            try:
                return _runtime_link_policy_once(self.adapter, receipt, working)
            except FileNotFoundError:
                if attempt == 2:
                    raise

    def unique_tree_bytes(self, roots, *, approved_links=None) -> int:
        roots = tuple(roots)
        if not _POSIX_NOFOLLOW:
            return super().unique_tree_bytes(roots, approved_links=approved_links)
        for attempt in range(3):
            try:
                return _scan_tree_bytes(self.adapter, roots, approved_links)
            except FileNotFoundError:
                if attempt == 2:
                    raise
