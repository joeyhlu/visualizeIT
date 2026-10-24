"""Read-only Kaggle evidence probe; never starts a process or compute session."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil


REQUIREMENTS = {"torch": "2.5.1", "torchvision": "0.20.1",
                "onnx": "1.17.0", "onnxruntime": "1.20.1"}
MAX_JSON = 8 * 1024**2


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON:
        raise ValueError("metadata JSON exceeds 8 MiB")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    def finite_tree(item, depth=0):
        if depth > 32:
            raise ValueError("metadata JSON nesting exceeds 32")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite_tree(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                finite_tree(child, depth + 1)
    finite_tree(value)
    return value


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def require_digest(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("expected lowercase SHA256")
    return value


def write_report(path, report):
    """Exclusive output creation prevents clobbering an input or earlier report."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def inspect_environment(request):
    """Inspect installed distributions and explicit paths, without importing ML."""
    blockers = []
    provenance = request.get("provenance", {})
    if (not isinstance(provenance, dict) or provenance.get("provider") != "kaggle" or
            not all(isinstance(provenance.get(key), str) and provenance[key].strip()
                    for key in ("notebook_url", "observed_at", "evidence_source"))):
        blockers.append("explicit_kaggle_provenance_missing")
    python = platform.python_version()
    system = platform.system()
    packages = {}
    for name, expected in REQUIREMENTS.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        packages[name] = {"installed": installed, "required": expected}
        if installed is None or installed.split("+", 1)[0] != expected:
            blockers.append(f"unsupported_package:{name}")
    if tuple(platform.python_version_tuple()[:2]) not in (("3", "10"), ("3", "11")):
        blockers.append("unsupported_python_requires_3.10_or_3.11")
    if system != "Linux":
        blockers.append("unsupported_system_requires_Linux")
    roots = {}
    proposed = request.get("proposed_roots", {})
    if not isinstance(proposed, dict) or not proposed:
        blockers.append("proposed_roots_missing")
        proposed = {}
    for name, value in sorted(proposed.items()):
        path = Path(value)
        try:
            exists = path.is_dir()
            free = shutil.disk_usage(path).free if exists else None
        except OSError:
            exists, free = False, None
        roots[name] = {"path": str(path), "exists": exists, "actual_free_bytes": free}
        if not exists:
            blockers.append(f"root_unavailable:{name}")
    archive = request.get("source_archive")
    archive_report = {"verified": False}
    if isinstance(archive, dict):
        expected = require_digest(archive.get("sha256"))
        path = Path(archive["path"])
        try:
            actual = digest(path)
            size = path.stat().st_size
            verified = actual == expected and size == archive.get("bytes")
            archive_report = {"path": str(path), "sha256": actual, "bytes": size,
                              "expected_sha256": expected, "verified": verified}
        except OSError:
            pass
    if not archive_report["verified"]:
        blockers.append("source_archive_missing_or_changed")
    assets = []
    for asset in request.get("assets", []):
        path = Path(asset["path"])
        require_digest(asset.get("sha256"))
        try:
            present = path.is_file() and path.stat().st_size == asset.get("bytes")
        except OSError:
            present = False
        assets.append({"path": str(path), "present_with_expected_bytes": present,
                       "expected_sha256": asset["sha256"], "content_verified": False})
    if not assets or any(not row["present_with_expected_bytes"] for row in assets):
        blockers.append("required_assets_missing")
    # Distribution metadata and source ZIP integrity cannot prove CUDA, native
    # rendering, checkpoint compatibility, or train/serve crop/token parity.
    blockers.extend(["asset_content_verification_required", "hardware_probe_required",
                     "provider_adapter_execution_required", "train_serve_parity_required",
                     "measured_preparation_training_storage_evidence_required"])
    return {"schema": "kaggle_preflight_v1", "provenance": provenance,
            "python": python, "system": system, "packages": packages,
            "requirements": {"python": ["3.10", "3.11"], "system": "Linux",
                             "packages": REQUIREMENTS,
                             "additional_frozen_requirements": "bench/runtime/requirements-quality.txt"},
            "roots": roots, "source_archive": archive_report, "assets": assets,
            "hardware": "unknown_not_probed", "blockers": blockers,
            "ready_for_training": False, "quality_claim": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect_environment(read_json(args.request))
    write_report(args.output, report)
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
