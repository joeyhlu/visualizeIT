#!/usr/bin/env python3
"""Kaggle source installer and bounded ROOT18 adaptation CLI (stdlib only).

Import, --help, install-source and status never import a model or acquire assets.
Evidence is supplied by an operator: hashes preserve provenance, not truth.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import types
from typing import Any
import uuid
import zipfile


ARCHIVE_SHA256 = "cbc1e698fce71f1be1561ac1033a1a7bbbe82fa1a8ddbd6050036ba56b6f8a7a"
ARCHIVE_BYTES = 5_700_194
ROOT18 = "visualizeit-cloud-v18"
ROOTS = {
    ROOT18: (287, "1d45840df6f3f031a031ac133d4adccf50db8a17c9ce9d206380c5a63ae54850"),
    "visualizeit-cloud-v29": (337, "0af8915205cc29e4ceb11e8b7956eb34f354a77760f330f244286d36f7a850a9"),
    "visualizeit-cloud-v30": (344, "19c242dd8abe9448c8a7210a41dd32be817cacbb055d59cba190f384ac407015"),
}
PYTHON_VERSION = "3.11.13"
UV_VERSION = "0.12.22"
GOTRACK_CHECKPOINT_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
GIB = 1024**3
MAX_JSON = 16 * 1024**2
SCALE_SOURCE_PINS = {
    "quality_adaptation_scale": "614c12856f91f13b9c5c9cf4b88be3958293df855d11183521189386d953f141",
    "quality_kaggle_preflight": "74167feb34929151e8fd0fd6c37c6000c7ee961b71bf8cb81ee2974e148184d8",
    "quality_ranker_data": "47ebe96f0c083ff549042273ee0405d2df6159d64bc28b37bfcbbe8f4e44d4d4",
}
STATE_IDENTITY_KEYS = ("id", "physical_object_id", "scene_id", "sequence_id", "frame_id",
    "source_frame_id", "refinement_state", "mesh_sha256", "background_id", "background_kind",
    "provenance_kind", "aliases")
STAGES = ("extract", "assemble", "preflight", "train", "resume", "export")


class ContractError(ValueError):
    """An identity, admission, containment or resource contract was not met."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def no_symlinks(path: Path) -> Path:
    """Reject symlinks at every existing path component, including parents."""
    absolute = Path(os.path.abspath(path))
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ContractError(f"symlink path component: {part}")
    return absolute


def contained(path: Path, root: Path) -> Path:
    path, root = no_symlinks(path), no_symlinks(root)
    if not path.is_relative_to(root):
        raise ContractError(f"path escapes explicit root {root}: {path}")
    return path


def parse_json_bytes(data: bytes, path: Path) -> Any:
    if len(data) > MAX_JSON:
        raise ContractError(f"JSON exceeds {MAX_JSON} bytes: {path}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ContractError(value)))


def read_json(path: Path) -> Any:
    with path.open("rb") as stream:
        return parse_json_bytes(stream.read(MAX_JSON + 1), path)


def atomic_json(path: Path, value: Any) -> None:
    no_symlinks(path)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def positive(value: Any, name: str, ceiling: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or value > ceiling):
        raise ContractError(f"{name} must be positive, finite and <= {ceiling}")
    return value


def integer(value: Any, name: str, ceiling: int) -> int:
    positive(value, name, ceiling)
    if type(value) is not int:
        raise ContractError(f"{name} must be an integer")
    return value


def verify_descriptor(value: dict, roots: tuple[Path, ...]) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ContractError("file descriptor requires exactly path, bytes, sha256")
    path = no_symlinks(Path(value["path"]))
    if not Path(value["path"]).is_absolute() or not any(path.is_relative_to(root) for root in roots):
        raise ContractError(f"descriptor outside explicit roots: {path}")
    if (type(value["bytes"]) is not int or value["bytes"] < 0 or not path.is_file()
            or path.stat().st_size != value["bytes"] or file_sha256(path) != value["sha256"]):
        raise ContractError(f"missing or changed file identity: {path}")
    return path


def safe_member(name: str) -> None:
    p = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or p.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or str(p) != name):
        raise ContractError(f"unsafe archive member: {name!r}")


@dataclass(frozen=True)
class ArchiveSpec:
    sha256: str = ARCHIVE_SHA256
    size: int = ARCHIVE_BYTES
    # Tests supply an explicit tiny specification; the CLI never accepts a replacement.
    roots: Any = None


def authenticated_archive(archive: Path, spec: ArchiveSpec = ArchiveSpec()) -> dict[str, bytes]:
    archive = no_symlinks(archive)
    if archive.stat().st_size != spec.size:
        raise ContractError("source archive byte count mismatch")
    raw = archive.read_bytes()
    if digest(raw) != spec.sha256:
        raise ContractError("source archive SHA-256 mismatch")
    roots = ROOTS if spec.roots is None else spec.roots
    payload: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            safe_member(info.filename)
            mode = info.external_attr >> 16
            if (info.filename in payload or info.is_dir() or stat.S_ISLNK(mode)
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or info.flag_bits & 1 or info.file_size > 16 * 1024**2):
                raise ContractError(f"duplicate/nonregular/oversize archive member: {info.filename}")
            payload[info.filename] = z.read(info)  # CRC checked before any writes.
            if sum(len(value) for value in payload.values()) > 128 * 1024**2:
                raise ContractError("uncompressed source exceeds 128 MiB")
    validate_payload(payload, roots)
    return payload


def validate_payload(payload: dict[str, bytes], roots: dict) -> None:
    expected = set()
    for root, (count, manifest_hash) in roots.items():
        name = f"{root}/cloud-source-manifest.json"
        if name not in payload or digest(payload[name]) != manifest_hash:
            raise ContractError(f"root manifest mismatch: {root}")
        manifest = json.loads(payload[name])
        files = manifest["files"]
        if not isinstance(files, dict) or len(files) != count:
            raise ContractError(f"root file count mismatch: {root}")
        expected.add(name)
        for relative, checksum in files.items():
            safe_member(relative)
            member = f"{root}/{relative}"
            expected.add(member)
            if member not in payload or digest(payload[member]) != checksum:
                raise ContractError(f"source member identity mismatch: {member}")
    if set(payload) != expected:
        raise ContractError("archive membership differs from exact three root manifests")


def authenticated_directory(source: Path) -> dict[str, bytes]:
    source = no_symlinks(source)
    if not source.is_dir():
        raise ContractError("expanded source must be an existing directory")
    payload = {}
    seen = set()
    observed_directories = set()
    total = 0
    for parent, directories, files in os.walk(source, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            no_symlinks(path)
            if path.is_dir():
                observed_directories.add(path.relative_to(source).as_posix())
                continue
            info = path.stat()
            physical = info.st_dev, info.st_ino
            if not stat.S_ISREG(info.st_mode) or physical in seen:
                raise ContractError(f"nonregular or aliased expanded source file: {path}")
            seen.add(physical)
            relative = path.relative_to(source).as_posix()
            safe_member(relative)
            if info.st_size > 16 * 1024**2:
                raise ContractError("expanded source member exceeds 16 MiB")
            total += info.st_size
            if total > 128 * 1024**2:
                raise ContractError("expanded source exceeds 128 MiB")
            payload[relative] = path.read_bytes()
    validate_payload(payload, ROOTS)
    expected_directories = {parent.as_posix() for name in payload for parent in PurePosixPath(name).parents if str(parent) != "."}
    if observed_directories != expected_directories:
        raise ContractError("expanded source has unexpected/missing directories")
    return payload


def source_receipt(kind: str, *, spec: ArchiveSpec = ArchiveSpec()) -> dict:
    base = {"schema": "kaggle_source_install_v1", "kind": kind,
            "files": sum(count + 1 for count, _ in (ROOTS if spec.roots is None else spec.roots).values()),
            "completed": True}
    if kind == "archive":
        base.update(archive_sha256=spec.sha256, archive_bytes=spec.size)
    else:
        base["root_manifests"] = {root: checksum for root, (_, checksum) in ROOTS.items()}
    return base


def install_source(archive: Path | None, destination: Path, *, spec: ArchiveSpec = ArchiveSpec(),
                   source_directory: Path | None = None) -> dict:
    source = archive if archive is not None else source_directory
    if source is None or (archive is not None and source_directory is not None):
        raise ContractError("provide exactly one archive or expanded source directory")
    if not source.is_absolute() or not destination.is_absolute():
        raise ContractError("source archive and destination must be explicit absolute paths")
    destination = no_symlinks(destination)
    if destination.exists() or not destination.parent.is_dir():
        raise ContractError("source installation requires a fresh destination and existing parent")
    if source_directory is not None and (destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ContractError("expanded source and destination must be separate")
    payload = authenticated_archive(archive, spec) if archive is not None else authenticated_directory(source_directory)
    # No mkdir, receipt or extraction occurs until every byte passes validation.
    destination.mkdir()
    for member, data in payload.items():
        target = contained(destination / member, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    receipt = source_receipt("archive" if archive is not None else "expanded_directory", spec=spec)
    atomic_json(destination / "source-install.json", receipt)
    return receipt


def verify_source(destination: Path) -> Path:
    destination = no_symlinks(destination)
    receipt = read_json(destination / "source-install.json")
    if receipt.get("kind") not in ("archive", "expanded_directory") or receipt != source_receipt(receipt["kind"]):
        raise ContractError("source installation receipt mismatch")
    for root, (count, checksum) in ROOTS.items():
        folder = contained(destination / root, destination)
        manifest = folder / "cloud-source-manifest.json"
        if file_sha256(manifest) != checksum:
            raise ContractError(f"installed manifest changed: {root}")
        files = read_json(manifest)["files"]
        if len(files) != count:
            raise ContractError("installed root count changed")
        for relative, expected in files.items():
            path = contained(folder / relative, folder)
            if not path.is_file() or file_sha256(path) != expected:
                raise ContractError(f"installed source changed: {path}")
        # Model/source acquisition has a separate explicit pinned API. It may add
        # assets only in its established cache; extra importable project code fails.
        for path in folder.rglob("*"):
            no_symlinks(path)
            if path.is_file():
                relative = path.relative_to(folder).as_posix()
                if (relative not in files and relative != "cloud-source-manifest.json"
                        and not relative.startswith(".cache/model-quality/")):
                    raise ContractError(f"unexpected installed source member: {path}")
    return destination / ROOT18


def clean_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "PIP_", "UV_"))
           and key not in {"VIRTUAL_ENV", "CONDA_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH"}}
    env.update(VISUALIZEIT_CLOUD="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               XFORMERS_DISABLED="1", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


RUNTIME_PROBE = r'''
import hashlib, importlib.metadata as m, json, pathlib, platform, sys
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576), b''):h.update(b)
 return h.hexdigest()
print(json.dumps({'python':platform.python_version(),'executable':str(pathlib.Path(sys.executable).absolute()),
 'executable_sha256':sha(sys.executable),'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))
'''


def runtime_probe(python: Path) -> dict:
    if not python.is_absolute() or not python.is_file():
        raise ContractError("explicit existing absolute interpreter required")
    result = subprocess.run([str(python), "-I", "-B", "-c", RUNTIME_PROBE], check=True,
                            capture_output=True, text=True, timeout=30, env=clean_environment())
    return json.loads(result.stdout)


def verify_runtime(python: Path, receipt: dict, root18: Path,
                   allowed_roots: tuple[Path, ...]) -> dict:
    if receipt.get("runtime", {}).get("executable_sha256") != file_sha256(python):
        raise ContractError("interpreter executable identity changed before runtime probe")
    observed = runtime_probe(python)
    if (receipt.get("schema") != "kaggle_isolated_runtime_v1"
            or receipt.get("uv_version") != UV_VERSION
            or receipt.get("runtime") != observed or observed["python"] != PYTHON_VERSION
            or observed["prefix"] == observed["base_prefix"]):
        raise ContractError("isolated runtime identity/version changed or bootstrap receipt missing")
    for name, expected in {"torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
                           "numpy": "1.26.4", "onnxruntime": "1.20.1", "pyrender": "0.1.45"}.items():
        if observed["packages"].get(name) != expected:
            raise ContractError(f"runtime requires {name}=={expected}")
    for relative in ("bench/runtime/requirements-ranker.txt", "bench/runtime/requirements-quality.txt"):
        if receipt.get("requirements", {}).get(relative) != file_sha256(root18 / relative):
            raise ContractError(f"runtime requirement provenance changed: {relative}")
    if receipt.get("python_request") != PYTHON_VERSION:
        raise ContractError("exact managed interpreter request provenance missing")
    runtime_root = no_symlinks(Path(receipt["runtime_root"]))
    if (not python.is_relative_to(runtime_root) or not python.resolve(strict=True).is_relative_to(runtime_root)
            or not Path(observed["prefix"]).is_relative_to(runtime_root)):
        raise ContractError("interpreter/prefix escaped authenticated runtime directory")
    for key in ("uv_wheel", "resolved_requirements"):
        verify_descriptor(receipt[key], allowed_roots)
    runtime_link_policy(receipt, Path(receipt["runtime_root"]).parent)
    return observed


def runtime_link_policy(receipt: dict, working: Path) -> dict[Path, Path]:
    """Authenticate only the bootstrap's exact internal runtime link layout."""
    root = contained(Path(receipt["runtime_root"]), working)
    if root == working or not root.is_dir():
        raise ContractError("dedicated authenticated runtime directory required")
    entries = receipt.get("runtime_links")
    if not isinstance(entries, list):
        raise ContractError("bootstrap runtime link inventory required")
    policy = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "target"}:
            raise ContractError("runtime link requires path and exact target")
        safe_member(item["path"])
        path = root / item["path"]
        no_symlinks(path.parent)
        if path in policy or not path.is_symlink() or os.readlink(path) != item["target"]:
            raise ContractError(f"runtime link inventory changed: {path}")
        lexical = Path(os.path.abspath(path.parent / item["target"]))
        if not lexical.is_relative_to(root):
            raise ContractError(f"runtime link escapes runtime root: {path}")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ContractError(f"broken/cyclic runtime link: {path}") from error
        if not resolved.is_relative_to(root) or (resolved.is_dir() and path.parent.is_relative_to(resolved)):
            raise ContractError(f"escaping/cyclic runtime directory alias: {path}")
        policy[path] = resolved
    observed = set()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink():
                observed.add(path)
    if observed != set(policy):
        raise ContractError("unapproved runtime links appeared or authenticated links disappeared")
    return policy


def unique_tree_bytes(roots: tuple[Path, ...], *, approved_links: dict[Path, Path] | None = None) -> int:
    """Union of resident files, including nested roots/hard links only once."""
    seen: set[tuple[int, int]] = set()
    directories_seen: set[tuple[int, int]] = set()
    approved = approved_links or {}
    total = 0
    def visit(path: Path) -> None:
        nonlocal total
        if path.is_symlink():
            if path not in approved or path.resolve(strict=True) != approved[path]:
                raise ContractError(f"unapproved accounting symlink: {path}")
            path = approved[path]
        info = path.stat()
        identity = info.st_dev, info.st_ino
        if stat.S_ISDIR(info.st_mode):
            if identity not in directories_seen:
                directories_seen.add(identity)
                for child in path.iterdir():
                    visit(child)
        elif stat.S_ISREG(info.st_mode):
            if identity not in seen:
                seen.add(identity)
                total += info.st_size
        else:
            raise ContractError(f"nonregular resident path: {path}")
    for root in roots:
        no_symlinks(root)
        if root.exists():
            visit(root)
    return total


def free_by_filesystem(roots: tuple[Path, ...]) -> dict[int, int]:
    result = {}
    for root in roots:
        path = root
        while not path.exists():
            path = path.parent
        device = path.stat().st_dev
        free = shutil.disk_usage(path).free
        result[device] = min(result.get(device, free), free)
    return result


def validate_limits(limits: dict) -> dict:
    expected = {"max_seconds", "max_rss_bytes", "max_output_bytes", "max_log_bytes", "reserve_bytes"}
    if set(limits) != expected:
        raise ContractError(f"limits require {sorted(expected)}")
    positive(limits["max_seconds"], "max_seconds", 86400)
    integer(limits["max_rss_bytes"], "max_rss_bytes", 64 * GIB)
    integer(limits["max_output_bytes"], "max_output_bytes", 8 * GIB)
    integer(limits["max_log_bytes"], "max_log_bytes", 64 * 1024**2)
    integer(limits["reserve_bytes"], "reserve_bytes", 64 * GIB)
    if limits["reserve_bytes"] < GIB:
        raise ContractError("at least 1 GiB free-space reserve required")
    return limits


def verify_admission(evidence: dict, stage: str, request: dict, identity: str,
                     limits: dict, working: Path, *, verified_metadata: dict | None = None) -> None:
    if platform.system() != "Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ContractError("execution requires the actual Kaggle Linux session")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise ContractError("Kaggle mount evidence is missing")
    if (evidence.get("schema") != "kaggle_execution_evidence_v1"
            or evidence.get("provider") != "kaggle" or evidence.get("stage") != stage
            or evidence.get("identity_sha256") != identity
            or not evidence.get("session_id") or not evidence.get("evidence_source")
            or not re.fullmatch(r"https://www\.kaggle\.com/code/[^/]+/[^/?#]+", evidence.get("notebook_url", ""))):
        raise ContractError("actual session/stage/identity provenance evidence required")
    now = datetime.now(timezone.utc).timestamp()
    observed = datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00")).timestamp()
    expires = datetime.fromisoformat(evidence["expires_at"].replace("Z", "+00:00")).timestamp()
    if not now - 3600 <= observed <= now or not now < expires <= observed + 3600:
        raise ContractError("session/quota evidence expired, future-dated or older than one hour")
    quota = evidence["quota"]
    if (quota.get("paid") is not False or quota.get("gpu_enabled") is not True
            or min(positive(quota.get("gpu_remaining_seconds"), "GPU quota", 30 * 3600),
                   positive(quota.get("session_remaining_seconds"), "session quota", 86400)) < limits["max_seconds"]):
        raise ContractError("observed free GPU/session quota does not cover this invocation")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=15, env=clean_environment())
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if devices.count(f"{evidence['gpu']['uuid']}, {evidence['gpu']['name']}") != 1:
        raise ContractError("the explicitly selected observed GPU identity changed")
    if evidence.get("parity_passed") is not True or evidence.get("authentic_inputs_reviewed") is not True:
        raise ContractError("crop/token parity and authentic input review must pass")
    if evidence.get("test_arrays_opened") is not False:
        raise ContractError("sealed test arrays must remain unopened")
    # Full detailed parity and input reports are hashed in request.inputs. These
    # booleans cannot independently establish correctness of operator evidence.
    roles = {item["role"] for item in request["inputs"]}
    if not {"parity_report", "input_review"}.issubset(roles):
        raise ContractError("hashed detailed parity/input-review reports required")
    if verified_metadata is None:
        verified_metadata = run_metadata_check(request, stage, Path(request["source_install"]) / ROOT18)
    m = evidence["measurements"]
    positive(m.get("measured_units"), "measured_units", 10**9)
    seconds = positive(m.get("measured_seconds"), "measured_seconds", 86400)
    units = positive(m.get("planned_units"), "planned_units", 10**9)
    cfg = read_json(Path(request["config"]["path"]))
    if stage in ("train", "resume") and units < cfg.get("max_steps", 2000):
        raise ContractError("measurement projection must cover the complete frozen optimizer-step cap")
    if stage == "extract":
        rows = read_json(Path(request["crop_manifest"]["path"]))["rows"]
        if units < len(rows):
            raise ContractError("measurement projection must cover all requested crop rows")
    if seconds / m["measured_units"] * units > limits["max_seconds"]:
        raise ContractError("representative measured time does not fit deadline")
    runtime_receipt = read_json(Path(request["runtime"]["path"]))
    runtime_root = contained(Path(runtime_receipt["runtime_root"]), working)
    for forbidden in (Path(request["source_install"]), Path(request["output_root"])):
        if runtime_root.is_relative_to(forbidden) or forbidden.is_relative_to(runtime_root):
            raise ContractError("runtime directory must be separate from source and output")
    resident = unique_tree_bytes((working,), approved_links=runtime_link_policy(runtime_receipt, working))
    if (type(m.get("resident_bytes")) is not int or m["resident_bytes"] < resident
            or type(m.get("additional_peak_bytes")) is not int or m["additional_peak_bytes"] < 0
            or m.get("storage_baseline") != "current_resident_before_stage"):
        raise ContractError("measured resident union/additional peak/storage baseline required")
    storage_limit = integer(m.get("storage_limit_bytes"), "storage_limit_bytes", 128 * GIB)
    extra = m["additional_peak_bytes"]
    if (extra + limits["reserve_bytes"] > min(free_by_filesystem((working,)).values())
            or m["resident_bytes"] + extra + limits["reserve_bytes"] > storage_limit):
        raise ContractError("additional storage/reserve or total storage quota does not fit")
    if limits["max_output_bytes"] > storage_limit or extra < limits["max_log_bytes"]:
        raise ContractError("output cap exceeds quota or log headroom omitted")
    if stage in ("train", "resume", "export"):
        scale = request.get("scale_admission")
        if not scale:
            raise ContractError("existing reviewed scale admission report required")
        report = read_json(Path(scale["path"]))
        if (report.get("schema") != "adaptation_scale_admission_v1"
                or report.get("admitted") is not True or report.get("blockers") != []):
            raise ContractError("prepared inventory scale admission has unresolved blockers")
        plan = read_json(Path(request["global_inventory"]["path"]))
        if report.get("plan_sha256") != plan.get("plan_sha256") or not plan.get("plan_sha256"):
            raise ContractError("scale admission must bind the frozen global inventory plan")


def command(stage: str, request: dict) -> list[str]:
    base = [request["python"], "-E", "-s", "-B", "-m"]
    paths, limits = request["paths"], request["limits"]
    if stage == "extract":
        extraction = request["extraction"]
        return base + ["bench.quality_adaptation_prepare", "--manifest", request["crop_manifest"]["path"],
                       "--output", paths["extraction_dir"], "--device", "cuda", "--max-seconds",
                       str(extraction["max_seconds"]), "--max-output-bytes", str(extraction["max_output_bytes"]),
                       "--max-rows", str(extraction["max_rows"]), "--cloud-execution"]
    if stage == "assemble":
        return base + ["bench.quality_adaptation_manifest", "--extraction", paths["extraction_dir"],
                       "--labels", request["labels"]["path"], "--output", paths["manifest_output"]]
    if stage == "export":
        return base + ["bench.quality_adaptation_train", "export", "--run", paths["train_dir"],
                       "--output", paths["export_output"], "--cloud-execution"]
    result = base + ["bench.quality_adaptation_train", stage, "--manifest", request["dataset"]["path"],
                     "--config", request["config"]["path"]]
    if stage != "preflight":
        result += ["--output", paths["train_dir"], "--cloud-execution"]
    return result


def identity_document(request: dict, runtime: dict) -> dict:
    receipt = Path(request["source_install"]) / "source-install.json"
    return {"source_archive_reference": ARCHIVE_SHA256, "root18_manifest": ROOTS[ROOT18][1],
            "source_install_receipt": read_json(no_symlinks(receipt)) if receipt.is_file() else None,
            "mode": request.get("mode", "extract"),
            "adapter_sha256": file_sha256(Path(__file__)),
            "roots": {key: request[key] for key in
                      ("source_install", "input_root", "working_root", "output_root")},
            "python": request["python"], "runtime": runtime,
            "runtime_receipt": request["runtime"], "config": request["config"],
            "global_inventory": request["global_inventory"], "inputs": request["inputs"],
            "paths": request["paths"], "extraction": request.get("extraction"),
            "crop_manifest": request.get("crop_manifest"), "labels": request.get("labels"),
            "scale_sources": request.get("scale_sources"), "identity_bridge": request.get("identity_bridge"),
            "preparation_identity": {key: request.get("preparation", {}).get(key)
                                     for key in ("producer", "config", "output_root")}}


def load_scale_validator(request: dict, root18: Path):
    """Execute authenticated source bytes, independently of bytecode caches."""
    allowed = (Path(request["input_root"]), Path(request["working_root"]))
    paths = {}
    descriptors = {}
    for name in ("quality_adaptation_scale", "quality_kaggle_preflight"):
        descriptor = request["scale_sources"][name]
        if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}
                or descriptor.get("sha256") != SCALE_SOURCE_PINS[name]
                or Path(descriptor["path"]).suffix != ".py"
                or type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0):
            raise ContractError(f"reviewed metadata validator descriptor changed: {name}")
        path = no_symlinks(Path(descriptor["path"]))
        if not Path(descriptor["path"]).is_absolute() or not any(path.is_relative_to(root) for root in allowed):
            raise ContractError(f"metadata validator outside explicit roots: {path}")
        paths[name] = path
        descriptors[name] = descriptor
    paths["quality_ranker_data"] = contained(root18 / "bench/quality_ranker_data.py", root18)
    sources = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            source = stream.read(MAX_JSON + 1)
        descriptor = descriptors.get(name)
        if (len(source) > MAX_JSON or digest(source) != SCALE_SOURCE_PINS[name]
                or (descriptor is not None and len(source) != descriptor["bytes"])):
            raise ContractError(f"reviewed metadata validator changed: {name}")
        sources[name] = source
    namespace = "_visualizeit_verified_scale"
    package = types.ModuleType(namespace)
    package.__path__ = []  # Only explicitly authenticated modules below can resolve.
    sys.modules[namespace] = package
    for name in ("quality_kaggle_preflight", "quality_ranker_data", "quality_adaptation_scale"):
        fullname = namespace + "." + name
        module = types.ModuleType(fullname)
        module.__package__ = namespace
        module.__file__ = str(paths[name])
        sys.modules[fullname] = module
        setattr(package, name, module)
        exec(compile(sources[name], str(paths[name]), "exec", dont_inherit=True), module.__dict__)
    return sys.modules[namespace + ".quality_adaptation_scale"]


def metadata_document(descriptor: dict, allowed: tuple[Path, ...]) -> dict:
    path = no_symlinks(Path(descriptor["path"]))
    if path.suffix != ".json":
        raise ContractError(f"metadata must be explicit JSON, never an array archive: {path}")
    return read_json(verify_descriptor(descriptor, allowed))


def indexed_rows(rows: Any, name: str) -> dict[str, dict]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
        raise ContractError(f"{name} requires row objects with stable ids")
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ContractError(f"duplicate {name} row id")
    return result


def payload_path(descriptor: dict, base: Path, allowed: tuple[Path, ...]) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
        raise ContractError("payload descriptor requires path, bytes and sha256")
    if (type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0
            or not re.fullmatch("[0-9a-f]{64}", descriptor.get("sha256", ""))):
        raise ContractError("invalid payload byte/hash identity")
    path = Path(descriptor["path"])
    if not path.is_absolute():
        safe_member(descriptor["path"])
        path = base / path
    path = no_symlinks(path)
    if not any(path.is_relative_to(root) for root in allowed):
        raise ContractError(f"payload descriptor escapes allowed roots: {path}")
    return path


def same_payload(left: dict, left_root: Path, right: dict, right_root: Path,
                 allowed: tuple[Path, ...]) -> bool:
    return (payload_path(left, left_root, allowed) == payload_path(right, right_root, allowed)
            and left["sha256"] == right["sha256"] and left["bytes"] == right["bytes"])


def reject_sealed_payload_aliases(active_paths: set[str], sealed_paths: set[str],
                                  requested_paths: set[str]) -> None:
    """Use path/stat metadata only; never hash or open a held-out payload."""
    active = {no_symlinks(Path(path)) for path in active_paths | requested_paths}
    sealed = {no_symlinks(Path(path)) for path in sealed_paths}
    if active & sealed:
        raise ContractError("active/request input overlaps a sealed payload path")
    sealed_files = {}
    for path in sealed:
        try:
            info = path.stat()
        except FileNotFoundError:
            continue  # Sealed metadata does not require acquiring its payload.
        if not stat.S_ISREG(info.st_mode):
            raise ContractError(f"nonregular sealed payload path: {path}")
        sealed_files[(info.st_dev, info.st_ino)] = path
    for path in active:
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if (info.st_dev, info.st_ino) in sealed_files:
            raise ContractError("active/request input physically aliases a sealed payload")


def metadata_check(request: dict, stage: str, root18: Path) -> dict:
    """Join all frozen identities before hashing or opening any array payload.

    An input-provenance JSON's *file bytes* must be the inputs_sha256 frozen by
    the existing planner. A freshly invented bridge cannot reassign that link.
    This contract requires authentic producer provenance; legacy opaque digests
    without their original identity bundle are deliberately unsupported.
    """
    if request.get("mode", "extract") == "prepared" and stage in ("extract", "assemble"):
        raise ContractError("prepared mode does not extract or assemble")
    allowed = (Path(request["input_root"]), Path(request["working_root"]))
    scale = load_scale_validator(request, root18)
    plan = metadata_document(request["global_inventory"], allowed)
    scale.validate_plan(plan)  # Reconstruct existing grouping/splits/seals, not a copy.
    if not plan["feasible"] or plan["blockers"]:
        raise ContractError("frozen scale plan is infeasible")
    states = indexed_rows(plan["global_manifest"]["records"], "global plan")
    bridge = metadata_document(request["identity_bridge"], allowed)
    if (bridge.get("schema") != "kaggle_correspondence_identity_bridge_v1"
            or bridge.get("plan_sha256") != plan["plan_sha256"]
            or bridge.get("manifest_sha256") != plan["global_manifest"]["manifest_sha256"]
            or bridge.get("bridge_sha256") != canonical({key: value for key, value in bridge.items() if key != "bridge_sha256"})):
        raise ContractError("identity bridge seal/global plan mismatch")
    bridge_rows = indexed_rows(bridge.get("rows"), "identity bridge")
    if set(bridge_rows) != set(states):
        raise ContractError("bridge must retain every global state exactly once, including sealed/development metadata")
    crops_doc = metadata_document(request["crop_manifest"], allowed)
    labels_doc = metadata_document(request["labels"], allowed)
    if (crops_doc.get("schema") != "correspondence_input_crops_v1"
            or labels_doc.get("schema") != "correspondence_label_inventory_v1"):
        raise ContractError("actual crop and label metadata schemas required")
    crops = indexed_rows(crops_doc.get("rows"), "crop manifest")
    labels = indexed_rows(labels_doc.get("rows"), "label inventory")
    expected = {ident for ident, row in states.items() if row["split"] != "dev"}
    if set(labels) != expected:
        raise ContractError("label inventory must retain the exact global TRAIN/validation/TEST ids and failures")
    bundles = {}
    sealed_paths = set()
    permitted_payloads = set()
    expected_crops = set()
    ownership = {}
    crop_root = Path(request["crop_manifest"]["path"]).parent
    extraction_root = Path(request["paths"]["extraction_dir"])
    # First pass reads provenance JSON and compares metadata only. It never
    # verifies an image/mesh/NPZ file before all frozen split joins have passed.
    for ident, state in states.items():
        ref = bridge_rows[ident].get("input_provenance")
        if not isinstance(ref, dict) or ref.get("sha256") != state["inputs_sha256"]:
            raise ContractError(f"input provenance is not the original frozen inputs_sha256: {ident}")
        bundle = metadata_document(ref, allowed)
        bundles[ident] = bundle
        if (bundle.get("schema") != "kaggle_input_identity_v1"
                or bundle.get("identity") != {key: state[key] for key in STATE_IDENTITY_KEYS}
                or bundle.get("capture_group") != state["scene_id"]):
            raise ContractError(f"authoritative source/object/sequence/scene/mesh/background identity changed: {ident}")
        split = {"validation": "val"}.get(state["split"], state["split"])
        active = split in ("train", "val")
        status = bundle.get("status")
        if status not in ("available", "unavailable", "sealed") or (active and status == "sealed"):
            raise ContractError(f"invalid provenance availability: {ident}")
        if status == "unavailable" and not bundle.get("failure_reason"):
            raise ContractError("unavailable provenance must retain its failure reason")
        for key in ("source_frame", "mesh", "crop", "labels"):
            descriptor = bundle.get(key)
            if descriptor is None:
                if active and status == "available":
                    raise ContractError(f"authentic available state lacks {key}: {ident}")
                continue
            path = payload_path(descriptor, Path(ref["path"]).parent, allowed)
            (permitted_payloads if active else sealed_paths).add(str(path))
            if key == "mesh" and descriptor["sha256"] != state["mesh_sha256"]:
                raise ContractError(f"metric mesh descriptor differs from frozen identity: {ident}")
            if key in ("source_frame", "crop"):
                owner = ownership.setdefault((key, descriptor["sha256"]), split)
                if owner != split:
                    raise ContractError(f"source/crop content leaks between frozen splits: {ident}")
        rgb = bundle.get("query_rgb_sha256")
        if rgb is not None:
            if not re.fullmatch("[0-9a-f]{64}", rgb):
                raise ContractError("invalid authoritative query RGB hash")
            owner = ownership.setdefault(("query_rgb", rgb), split)
            if owner != split:
                raise ContractError("query RGB content crosses frozen split boundary")
        if ident not in expected:
            continue
        row = labels[ident]
        if (row.get("split") != split or row.get("object_id") != state["physical_object_id"]
                or row.get("sequence_id") != state["sequence_id"] or row.get("capture_group") != state["scene_id"]
                or row.get("query_rgb_sha256") != rgb):
            raise ContractError(f"label id/split/object/sequence/source RGB differs from frozen state: {ident}")
        if active and status == "available":
            expected_crops.add(ident)
            crop = crops.get(ident)
            if crop is None or not same_payload(crop["crops"], crop_root, bundle["crop"], Path(ref["path"]).parent, allowed):
                raise ContractError(f"actual crop descriptor differs from frozen provenance: {ident}")
        if row.get("status") == "available":
            label = bundle.get("labels")
            label_output = bundle.get("label_output")
            if label_output is not None:
                safe_member(label_output)
            target = payload_path(row["labels"], extraction_root, allowed)
            if (label is None or label_output is None or target != extraction_root / label_output
                    or row["labels"]["sha256"] != label["sha256"] or row["labels"]["bytes"] != label["bytes"]):
                raise ContractError(f"actual label descriptor differs from frozen provenance: {ident}")
            (permitted_payloads if active else sealed_paths).add(str(target))
        elif row.get("status") != "unavailable" or not row.get("failure_reason"):
            raise ContractError("failed/unavailable label rows must remain explicit")
        if not active:
            for key in ("features", "labels"):
                if key in row:
                    sealed_paths.add(str(payload_path(row[key], extraction_root, allowed)))
    if set(crops) != expected_crops:
        raise ContractError("crop manifest must contain exactly the available frozen TRAIN/validation rows; no TEST/development arrays")
    index = recipe = None
    if stage != "extract":
        metadata_root = Path(request["input_root"]) if request.get("mode", "extract") == "prepared" else Path(request["output_root"])
        recipe_path = contained(extraction_root / "recipe.json", metadata_root)
        index_path = contained(extraction_root / "index.json", metadata_root)
        recipe, index_doc = read_json(recipe_path), read_json(index_path)
        if (recipe.get("schema") != "frozen_gotrack_decoder_recipe_v1"
                or recipe.get("crop_manifest_sha256") != canonical(crops_doc)
                or recipe.get("producer_sha256") != file_sha256(root18 / "bench/quality_adaptation_prepare.py")
                or recipe.get("annotation_inputs") != [] or recipe.get("device") != "cuda"
                or index_doc.get("recipe_sha256") != canonical(recipe)):
            raise ContractError("actual extraction recipe/index does not bind the frozen crops/producer")
        index = indexed_rows(index_doc.get("rows"), "extraction index")
        if set(index) != set(crops):
            raise ContractError("extraction must retain every requested row/failure exactly once")
        for ident, feature in index.items():
            if feature.get("status") == "available":
                if (feature.get("crops") != crops[ident]["crops"]
                        or feature.get("query_rgb_sha256") != bundles[ident].get("query_rgb_sha256")):
                    raise ContractError("extraction source/crop/RGB identities changed")
                permitted_payloads.add(str(payload_path(feature["features"], extraction_root, allowed)))
                split = states[ident]["split"]
                owner = ownership.setdefault(("features", feature["features"]["sha256"]), split)
                if owner != split:
                    raise ContractError("extracted feature content crosses frozen split boundary")
            elif feature.get("status") != "unavailable" or not feature.get("failure_reason"):
                raise ContractError("extraction failures must remain observable")
        if "dataset" in request:
            dataset = metadata_document(request["dataset"], allowed)
            dataset_rows = indexed_rows(dataset.get("rows"), "dataset")
            if dataset.get("schema") != "correspondence_head_dataset_v1" or set(dataset_rows) != expected:
                raise ContractError("assembled dataset must retain exact frozen ids including TEST/failures")
            if dataset.get("provenance", {}).get("feature_recipe_sha256") != canonical(recipe):
                raise ContractError("dataset feature recipe differs from actual frozen extraction")
            for ident, row in dataset_rows.items():
                wanted = dict(labels[ident])
                if wanted["split"] != "test" and wanted["status"] == "available":
                    feature = index.get(ident, {})
                    if feature.get("status") != "available":
                        raise ContractError("available dataset/label row hides failed extraction")
                    wanted["features"] = feature["features"]
                if row != wanted:
                    raise ContractError(f"dataset row/split/features/labels differs from frozen identity bridge: {ident}")
    reject_sealed_payload_aliases(permitted_payloads, sealed_paths,
        {item["file"]["path"] for item in request["inputs"]})
    # Arbitrarily labeling a sealed payload as a 'training_input' cannot cause
    # its hash to be read before the metadata reconciliation above.
    for item in request["inputs"]:
        path = no_symlinks(Path(item["file"]["path"]))
        if str(path) in sealed_paths:
            raise ContractError("sealed/development payload declared as an execution input")
        if path.suffix != ".json" and str(path) not in permitted_payloads:
            checkpoint = root18 / ".cache/model-quality/checkpoints/gotrack_checkpoint.pt"
            source_files = read_json(root18 / "bench/runtime/adaptation-sources.json")["gotrack"]["files"]
            pinned_source = {str(root18 / relative): checksum for relative, checksum in source_files.items()}
            if not ((item["role"] == "checkpoint" and path == checkpoint and item["file"]["sha256"] == GOTRACK_CHECKPOINT_SHA256)
                    or (item["role"] == "source_asset" and pinned_source.get(str(path)) == item["file"]["sha256"])):
                raise ContractError("payload is not bound by frozen non-test provenance or the existing asset lock")
    # At this point every id/split/descriptor relationship has been reconciled.
    # Hash only explicitly non-test source/crop/label files, never sealed payloads.
    def verify_active_payloads():
        for ident, bundle in bundles.items():
            if states[ident]["split"] not in ("train", "validation") or bundle["status"] != "available":
                continue
            if request.get("mode", "extract") != "prepared":
                keys = ("source_frame", "mesh", "crop") if stage == "extract" else ("source_frame", "mesh", "crop", "labels")
                for key in keys:
                    descriptor = bundle[key]
                    path = payload_path(descriptor, Path(bridge_rows[ident]["input_provenance"]["path"]).parent, allowed)
                    verify_descriptor({**descriptor, "path": str(path)}, allowed)
            if stage != "extract" and labels[ident]["status"] == "available":
                descriptor = labels[ident]["labels"]
                verify_descriptor({**descriptor, "path": str(payload_path(descriptor, extraction_root, allowed))}, allowed)
    if stage == "extract":
        verify_active_payloads()
        return {"schema": "kaggle_metadata_join_v1", "rows": len(states), "plan_sha256": plan["plan_sha256"], "sealed_payloads_opened": False}
    preparation = request["preparation"]
    if Path(preparation["output_root"]) != extraction_root:
        raise ContractError("original preparation receipts must refer to this global extraction root")
    if Path(preparation["producer"]["path"]) != root18 / "bench/quality_adaptation_prepare.py":
        raise ContractError("preparation producer must be the authenticated ROOT18 extractor")
    producer = verify_descriptor(preparation["producer"], allowed)
    config_doc = metadata_document(preparation["config"], allowed)
    if (producer != root18 / "bench/quality_adaptation_prepare.py"
            or config_doc != {**request["extraction"], "device": "cuda"}):
        raise ContractError("preparation producer/config differs from actual extraction contract")
    receipts = [metadata_document(ref, allowed) for ref in preparation["receipts"]]
    expected_shards = {shard["shard_sha256"]: shard for shard in plan["shards"]}
    if len(receipts) != len(expected_shards) or {r.get("shard_sha256") for r in receipts} != set(expected_shards):
        raise ContractError("all original unique preparation shard receipts required")
    # Reconcile ALL receipt rows before existing validators may hash outputs.
    for receipt in receipts:
        shard = expected_shards[receipt["shard_sha256"]]
        rows = indexed_rows(receipt.get("rows"), "preparation receipt")
        if set(rows) != {row["id"] for row in shard["rows"]}:
            raise ContractError("preparation receipt changed frozen shard membership")
        for ident, row in rows.items():
            feature = (index or {}).get(ident, {})
            if row.get("inputs_sha256") != states[ident]["inputs_sha256"]:
                raise ContractError("original preparation input identity changed")
            if row.get("status") == "available":
                if feature.get("status") != "available" or row.get("output") != feature.get("features"):
                    raise ContractError("prepared output receipt is unrelated to actual extracted features")
            elif row.get("status") != "failed" or feature.get("status") == "available" or not row.get("failure_reason"):
                raise ContractError("preparation receipts must preserve actual failure rows")
    verify_active_payloads()
    for receipt in receipts:
        scale.validate_preparation_receipt(plan, receipt,
            producer_sha256=preparation["producer"]["sha256"], config_sha256=preparation["config"]["sha256"],
            output_root=extraction_root)
    admission = None
    if stage in ("train", "resume", "export"):
        measurements = metadata_document(preparation["measurements"], allowed)
        admission = scale.assess_admission(plan, receipts, measurements=measurements,
            producer_sha256=preparation["producer"]["sha256"], config_sha256=preparation["config"]["sha256"], output_root=extraction_root)
        if not admission["admitted"] or metadata_document(request["scale_admission"], allowed) != admission:
            raise ContractError("scale admission differs from recomputed original receipts/measurements or is blocked")
    return {"schema": "kaggle_metadata_join_v1", "rows": len(states),
            "plan_sha256": plan["plan_sha256"], "sealed_payloads_opened": False, "prepared_admission": admission}


def run_metadata_check(request: dict, stage: str, root18: Path) -> dict:
    result = subprocess.run([request["python"], "-I", "-B", str(Path(__file__).absolute()),
                             "_metadata-check", "--stage", stage, "--root18", str(root18)],
                            input=json.dumps(request, allow_nan=False), check=False,
                            capture_output=True, text=True, timeout=120, env=clean_environment())
    if result.returncode != 0:
        raise ContractError("metadata admission blocked: " + result.stderr[-2000:])
    return json.loads(result.stdout)


def verify_neural_assets(root18: Path, inputs: list[dict]) -> None:
    """Authenticate the existing pinned source/weights before a neural child."""
    lock = read_json(root18 / "bench/runtime/adaptation-sources.json")
    for relative, expected in lock["gotrack"]["files"].items():
        safe_member(relative)
        source = contained(root18 / relative, root18)
        if not source.is_file() or file_sha256(source) != expected:
            raise ContractError(f"pre-existing pinned GoTrack source missing/changed: {source}")
    checkpoint = contained(root18 / ".cache/model-quality/checkpoints/gotrack_checkpoint.pt", root18)
    declared = [item["file"] for item in inputs
                if item["role"] == "checkpoint" and item["file"]["path"] == str(checkpoint)]
    if (len(declared) != 1 or declared[0]["sha256"] != GOTRACK_CHECKPOINT_SHA256
            or not checkpoint.is_file() or file_sha256(checkpoint) != GOTRACK_CHECKPOINT_SHA256):
        raise ContractError("authentic pinned GoTrack checkpoint must exist and be declared before neural execution")
    cache = root18 / ".cache/model-quality"
    if unique_tree_bytes((cache / "sources", cache / "checkpoints")) > 8 * GIB:
        raise ContractError("existing model source/checkpoint cache exceeds 8 GiB")


def validate_output_layout(paths: dict) -> None:
    if Path(paths["export_output"]).parent != Path(paths["train_dir"]):
        raise ContractError("export must be a fresh direct-child file inside train_dir")


def validate_stage_paths(request: dict, roots: dict[str, Path], mode: str) -> None:
    paths = request["paths"]
    for key in ("extraction_dir", "manifest_output", "train_dir", "export_output"):
        path = Path(paths[key])
        boundary = roots["input_root"] if mode == "prepared" and key in ("extraction_dir", "manifest_output") else roots["output_root"]
        if not path.is_absolute() or contained(path, boundary) == boundary:
            raise ContractError(f"{key} must be inside the explicit {boundary}")
    if Path(paths["manifest_output"]).parent != Path(paths["extraction_dir"]):
        raise ContractError("assembled global manifest must be in extraction directory")
    if "dataset" in request and request["dataset"]["path"] != paths["manifest_output"]:
        raise ContractError("trainer must use the one assembled global dataset")
    validate_output_layout(paths)
    for left, right in (("extraction_dir", "train_dir"), ("extraction_dir", "export_output")):
        a, b = Path(paths[left]), Path(paths[right])
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ContractError(f"stage outputs overlap: {left}/{right}")


def trainer_progress(output: Path, state: dict, *, now: float | None = None,
                     observe_invocation: bool = False) -> dict:
    """Read last confirmed optimizer metadata, never a checkpoint or payload."""
    previous = state.get("optimizer_progress") if isinstance(state.get("optimizer_progress"), dict) else {}
    result = {**previous, "last_confirmed_steps": previous.get("last_confirmed_steps"),
              "completion_confirmed": False, "current_progress_known": False,
              "freshness": "missing", "source": previous.get("source")}
    attempts = [row for row in state.get("stages", []) if row.get("stage") in ("train", "resume")]
    if not attempts:
        result["freshness"] = "not_started"
        return result
    attempt = attempts[-1]
    if state.get("identity", {}).get("roots", {}).get("output_root") != str(output):
        raise ContractError("status run identity does not match requested output root")
    train = contained(Path(state["identity"]["paths"]["train_dir"]), output)
    if train == output:
        raise ContractError("trainer status requires a dedicated child run directory")
    path = contained(train / "status.json", train)
    if not path.is_file():
        return result
    try:
        # Fields, hash and mtime must come from the same opened inode when the
        # actual trainer atomically replaces its running status concurrently.
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            raw = stream.read(MAX_JSON + 1)
        value = parse_json_bytes(raw, path)
    except (ValueError, OSError):
        result["freshness"] = "invalid"
        return result
    if (not isinstance(value, dict) or value.get("status") not in ("running", "completed", "incomplete", "failed")
            or type(value.get("steps")) is not int or value["steps"] < 0
            or type(value.get("epoch")) is not int or value["epoch"] < 0
            or isinstance(value.get("elapsed_seconds"), bool)
            or not isinstance(value.get("elapsed_seconds"), (float, int))
            or not math.isfinite(value["elapsed_seconds"]) or value["elapsed_seconds"] < 0
            or value.get("quality_claim", False) is not False or value.get("test_arrays_opened", False) is not False):
        result["freshness"] = "invalid"
        return result
    observed = info.st_mtime
    current = time.time() if now is None else now
    started = datetime.fromisoformat(attempt["started_at"]).timestamp()
    freshness = "fresh"
    if observed < started - 1:
        freshness = "older_than_latest_invocation"
    elif current - observed > 120 or observed > current + 1:
        freshness = "stale"
    elif attempt.get("status") == "failed":
        freshness = "failed_invocation"
    elif attempt.get("status") == "running":
        # A persisted running marker is insufficient proof the adapter survived.
        pid = attempt.get("adapter_pid")
        if sys.platform != "linux" or type(pid) is not int or not Path(f"/proc/{pid}").is_dir():
            freshness = "adapter_liveness_unconfirmed"
    source = {"path": str(path), "sha256": digest(raw), "bytes": len(raw),
              "as_of": datetime.fromtimestamp(observed, timezone.utc).isoformat(),
              "trainer_source_sha256": attempt.get("trainer_source_sha256")}
    if (not observe_invocation and attempt.get("status") != "running" and previous and previous.get("source")
            and previous["source"].get("sha256") != source["sha256"]):
        return {**previous, "freshness": "changed_after_invocation", "completion_confirmed": False,
                "current_progress_known": False}
    result.update(last_confirmed_steps=value["steps"], epoch=value["epoch"],
                  trainer_status=value["status"], elapsed_seconds=value["elapsed_seconds"],
                  source=source, freshness=freshness)
    result["completion_confirmed"] = (freshness == "fresh" and attempt.get("status") == "completed"
                                      and value["status"] == "completed")
    # The count is a timestamped last observation, never a live extrapolation.
    return result


def read_status(output: Path) -> dict:
    output = no_symlinks(output)
    path = contained(output / "adapter-state.json", output)
    if not path.is_file():
        return {"status": "not_started"}
    state = read_json(path)
    progress = trainer_progress(output, state)
    adapter_status = state.get("status", "unknown")
    effective = adapter_status
    if progress["freshness"] != "not_started" and not progress["completion_confirmed"]:
        effective = "failed" if adapter_status == "failed" else "incomplete"
    return {**state, "status": effective, "adapter_status": adapter_status, "optimizer_progress": progress}


def validate_request(request: dict, stage: str) -> tuple[Path, dict, str]:
    if request.get("schema") != "kaggle_adaptation_request_v1":
        raise ContractError("request schema must be kaggle_adaptation_request_v1")
    mode = request.get("mode", "extract")
    if mode not in ("extract", "prepared") or (mode == "prepared" and stage in ("extract", "assemble")):
        raise ContractError("prepared mode accepts preflight/train/resume/export only")
    roots = {}
    for key in ("source_install", "input_root", "working_root", "output_root"):
        p = Path(request[key])
        if not p.is_absolute():
            raise ContractError(f"{key} must be absolute")
        roots[key] = no_symlinks(p)
    working = contained(roots["working_root"], Path("/kaggle/working"))
    contained(roots["input_root"], Path("/kaggle/input"))
    contained(roots["output_root"], working)
    contained(roots["source_install"], working)
    if roots["output_root"] == working or roots["output_root"].is_relative_to(roots["source_install"]):
        raise ContractError("dedicated output directory outside installed source required")
    root18 = verify_source(roots["source_install"])
    allowed = roots["input_root"], working
    for key in ("runtime", "config", "global_inventory", "evidence", "identity_bridge"):
        metadata_document(request[key], allowed)
    if not isinstance(request["inputs"], list) or not request["inputs"]:
        raise ContractError("explicit complete relevant input inventory required")
    names = set()
    for item in request["inputs"]:
        if set(item) != {"role", "file"} or not item["role"] or item["role"] == "test":
            raise ContractError("inputs require a non-test role and file descriptor")
        path = no_symlinks(Path(item["file"]["path"]))
        if str(path) in names:
            raise ContractError("duplicate relevant input descriptor")
        names.add(str(path))
    for key in ("crop_manifest", "labels", "dataset", "scale_admission"):
        if key in request:
            metadata_document(request[key], allowed)
    required = {"extract": ("crop_manifest",), "assemble": ("labels",),
                "preflight": ("dataset",), "train": ("dataset",), "resume": ("dataset",),
                "export": ("dataset",)}[stage]
    if any(key not in request for key in required):
        raise ContractError(f"{stage} requires {required}")
    paths = request["paths"]
    validate_stage_paths(request, roots, mode)
    cfg = read_json(Path(request["config"]["path"]))
    if cfg.get("device") != "cuda":
        raise ContractError("this reviewed Kaggle route requires explicit CUDA, never CPU fallback")
    limits = validate_limits(request["limits"])
    if stage in ("train", "resume"):
        positive(cfg.get("max_seconds", 900.0), "trainer max_seconds", limits["max_seconds"])
        integer(cfg.get("max_output_bytes", 1536 * 1024**2), "trainer max_output_bytes", limits["max_output_bytes"])
    if stage == "extract":
        extraction = request["extraction"]
        integer(extraction["max_rows"], "extraction.max_rows", 10000)
        integer(extraction["max_seconds"], "extraction.max_seconds", min(86400, limits["max_seconds"]))
        integer(extraction["max_output_bytes"], "extraction.max_output_bytes", limits["max_output_bytes"])
    python = Path(request["python"])
    no_symlinks(python.parent)
    if not python.is_absolute() or not python.resolve().is_relative_to(working):
        raise ContractError("training interpreter and its managed target must remain in explicit working root")
    runtime = verify_runtime(python, read_json(Path(request["runtime"]["path"])), root18, allowed)
    metadata = run_metadata_check(request, stage, root18)
    for item in request["inputs"]:
        verify_descriptor(item["file"], allowed)
    if stage in ("extract", "train", "resume", "export"):
        verify_neural_assets(root18, request["inputs"])
    identity = canonical(identity_document(request, runtime))
    verify_admission(read_json(Path(request["evidence"]["path"])), stage, request, identity, limits, working, verified_metadata=metadata)
    return root18, runtime, identity


def group_rss(pid: int) -> int:
    """Linux process-group RSS sampled from procfs (conservative shared pages)."""
    if sys.platform != "linux":
        return 0  # Runner admission rejects non-Linux; local cleanup tests only.
    total = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pid:  # state, ppid, pgrp
                total += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue  # Process exited between directory scan and read.
    return total


def terminate_group(child: subprocess.Popen, *, grace_seconds: float = 5) -> None:
    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(child.pid, sig)
            elif child.poll() is None:
                child.terminate() if sig == signal.SIGTERM else child.kill()
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # Kill surviving descendants even if the group leader exited promptly.
    send(signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    child.wait(timeout=5)


def supervise(argv: list[str], *, cwd: Path, output_root: Path, working: Path,
              log_path: Path, limits: dict, env: dict) -> dict:
    started = time.monotonic()
    child = None
    reader = None
    overflow = threading.Event()
    reader_error: list[str] = []
    peak_rss = 0
    reason = None
    status = "failed"
    returncode = None
    old_handlers = {}
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, interrupted)
    try:
        with log_path.open("xb") as log:
            child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=os.name == "posix")
            def copy_log():
                written = 0
                try:
                    while True:
                        chunk = child.stdout.read(65536)
                        if not chunk:
                            break
                        remaining = limits["max_log_bytes"] - written
                        log.write(chunk[:max(0, remaining)])
                        written += min(len(chunk), max(0, remaining))
                        log.flush()
                        if len(chunk) > remaining:
                            overflow.set()
                except (OSError, ValueError) as error:
                    reader_error.append(str(error))
                    overflow.set()
            reader = threading.Thread(target=copy_log, daemon=True)
            reader.start()
            while child.poll() is None:
                peak_rss = max(peak_rss, group_rss(child.pid))
                if time.monotonic() - started >= limits["max_seconds"]:
                    reason = "deadline_exceeded"
                elif peak_rss > limits["max_rss_bytes"]:
                    reason = "process_group_rss_exceeded"
                elif overflow.is_set():
                    reason = "log_limit_or_log_io_error"
                elif unique_tree_bytes((output_root,)) > limits["max_output_bytes"]:
                    reason = "output_storage_exceeded"
                elif min(free_by_filesystem((working,)).values()) < limits["reserve_bytes"]:
                    reason = "filesystem_reserve_exhausted"
                if reason:
                    break
                time.sleep(0.1)
            terminate_group(child)
            reader.join(timeout=5)
            returncode = child.returncode
            if reader.is_alive():
                reason = reason or "log_pipe_not_closed"
            if overflow.is_set():
                reason = reason or "log_limit_or_log_io_error"
            if unique_tree_bytes((output_root,)) > limits["max_output_bytes"]:
                reason = reason or "output_storage_exceeded"
            if reason or returncode == 3:
                status = "incomplete"
                reason = reason or "child_reported_incomplete"
            elif returncode == 0:
                status = "completed"
            else:
                reason = "child_failed"
    except KeyboardInterrupt:
        status, reason = "incomplete", "interrupted"
    except Exception as error:
        status, reason = "failed", f"{type(error).__name__}: {error}"
    finally:
        if child is not None:
            terminate_group(child)
            returncode = child.returncode
            if child.stdout:
                child.stdout.close()
        if reader is not None:
            reader.join(timeout=5)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return {"status": status, "reason": reason, "returncode": returncode,
            "elapsed_seconds": time.monotonic() - started, "peak_group_rss_bytes": peak_rss,
            "log": str(log_path), "log_errors": reader_error, "outputs_preserved": True}


@contextmanager
def run_lock(output: Path):
    lock = output / ".adapter-lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def execute(stage: str, request_path: Path, expected_sha256: str) -> dict:
    request_path = no_symlinks(request_path)
    if file_sha256(request_path) != expected_sha256:
        raise ContractError("run request SHA-256 mismatch")
    request = read_json(request_path)
    root18, runtime, identity = validate_request(request, stage)
    output = Path(request["output_root"])
    output.mkdir(exist_ok=True)
    with run_lock(output):
        state_path = output / "adapter-state.json"
        if state_path.exists():
            state = read_json(state_path)
            if state.get("identity_sha256") != identity:
                raise ContractError("run identity changed: source/runtime/config/global inventory/inputs/paths")
        else:
            state = {"schema": "kaggle_adaptation_state_v1", "identity_sha256": identity,
                     "identity": identity_document(request, runtime), "stages": [], "quality_claim": False,
                     "test_arrays_opened": False}
        paths = request["paths"]
        dataset = request.get("dataset")
        if stage in ("preflight", "train", "resume", "export"):
            if state.get("dataset") not in (None, dataset):
                raise ContractError("global dataset changed; per-shard resumes are forbidden")
            state["dataset"] = dataset
        if stage != "extract" and "preparation" in request:
            if state.get("preparation") not in (None, request["preparation"]):
                raise ContractError("original preparation receipts/config/measurements changed")
            state["preparation"] = request["preparation"]
        completed = {r["stage"] for r in state["stages"] if r["status"] == "completed"}
        if stage in ("train", "resume", "export") and "preflight" not in completed:
            raise ContractError("successful preflight required before training/resume/export")
        if stage == "resume" and (not state.get("training_started") or not (Path(paths["train_dir"]) / "last.pt").is_file()):
            raise ContractError("resume requires this adapter's prior run and trainer last.pt")
        if stage in ("resume", "export"):
            name = "last.pt" if stage == "resume" else "best.pt"
            saved = state.get("checkpoint_files", {}).get(name)
            if not saved:
                raise ContractError(f"adapter checkpoint identity missing: {name}")
            verify_descriptor(saved, (output,))
        if stage == "export" and not any(r["stage"] in ("train", "resume") and r["status"] == "completed" for r in state["stages"]):
            raise ContractError("export requires completed training")
        target = {"extract": "extraction_dir", "assemble": "manifest_output",
                  "train": "train_dir", "export": "export_output"}.get(stage)
        if target and Path(paths[target]).exists():
            raise ContractError(f"{stage} cannot overwrite existing {paths[target]}")
        row = {"stage": stage, "status": "running", "request_sha256": expected_sha256,
               "started_at": datetime.now(timezone.utc).isoformat(), "adapter_pid": os.getpid(),
               "trainer_source_sha256": file_sha256(root18 / "bench/quality_adaptation_train.py") if stage in ("train", "resume") else None,
               "argv": command(stage, request)}
        if stage == "train":
            state["training_started"] = True
        state["stages"].append(row)
        state["status"] = "running"
        atomic_json(state_path, state)
        env = clean_environment()
        env["CUDA_VISIBLE_DEVICES"] = read_json(Path(request["evidence"]["path"]))["gpu"]["uuid"]
        result = supervise(row["argv"], cwd=root18, output_root=output,
                           working=Path(request["working_root"]),
                           log_path=output / f"stage-{len(state['stages']):04d}-{stage}.log",
                           limits=request["limits"], env=env)
        row.update(result)
        if stage in ("train", "resume"):
            state["optimizer_progress"] = trainer_progress(output, state, observe_invocation=True)
            if result["status"] == "completed" and not state["optimizer_progress"]["completion_confirmed"]:
                row.update(status="incomplete", reason="trainer_completion_status_unconfirmed")
                result = row
            state["checkpoint_files"] = {}
            for name in ("last.pt", "best.pt"):
                checkpoint = contained(Path(paths["train_dir"]) / name, output)
                if checkpoint.is_file():
                    state["checkpoint_files"][name] = {"path": str(checkpoint),
                        "bytes": checkpoint.stat().st_size, "sha256": file_sha256(checkpoint)}
        state["status"] = result["status"]
        atomic_json(state_path, state)
        return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("install-source", help="authenticate complete archive before fresh extraction")
    source = install.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--source-directory", type=Path)
    install.add_argument("--destination", required=True, type=Path)
    status = sub.add_parser("status", help="read persisted receipts; never starts work")
    status.add_argument("--output-root", required=True, type=Path)
    verify = sub.add_parser("verify-source", help="recheck installed code and three manifests")
    verify.add_argument("--destination", required=True, type=Path)
    probe = sub.add_parser("runtime-probe", help="metadata only; never imports Torch")
    probe.add_argument("--python", required=True, type=Path)
    identity_parser = sub.add_parser("request-identity", help="compute evidence binding; does not admit execution")
    identity_parser.add_argument("--request", required=True, type=Path)
    metadata_parser = sub.add_parser("_metadata-check", help=argparse.SUPPRESS)
    metadata_parser.add_argument("--stage", choices=STAGES, required=True)
    metadata_parser.add_argument("--root18", type=Path, required=True)
    for stage in STAGES:
        p = sub.add_parser(stage)
        p.add_argument("--request", required=True, type=Path)
        p.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "install-source":
            result = install_source(args.archive, args.destination, source_directory=args.source_directory)
        elif args.command == "verify-source":
            result = {"root18": str(verify_source(args.destination)), "status": "completed"}
        elif args.command == "runtime-probe":
            result = runtime_probe(args.python)
        elif args.command == "request-identity":
            request = read_json(no_symlinks(args.request))
            receipt = read_json(no_symlinks(Path(request["runtime"]["path"])))
            result = {"identity_sha256": canonical(identity_document(request, receipt["runtime"])),
                      "execution_admitted": False}
        elif args.command == "_metadata-check":
            raw = sys.stdin.read(MAX_JSON + 1)
            if len(raw) > MAX_JSON:
                raise ContractError("metadata worker request exceeds limit")
            result = metadata_check(json.loads(raw), args.stage, args.root18)
        elif args.command == "status":
            result = read_status(no_symlinks(args.output_root))
        else:
            result = execute(args.command, args.request, args.request_sha256)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result.get("status", "completed") == "completed" else 3
    except (ContractError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError,
            zipfile.BadZipFile) as error:
        print(json.dumps({"status": "blocked", "error": f"{type(error).__name__}: {error}",
                          "quality_claim": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
