"""Neural backends for the workbench: primary adapted GoTrack head, frozen
baseline comparator, and the ``workbench_integer_metric_v1`` serving profile.

Primary model (no fallback): flow-terminal export
``27e06a51...b9311`` (80,165,526 bytes) installed onto the strict frozen base
network through the unchanged atomic installer
``quality_gotrack_inference.install_adapted_head`` with a head-only context
from ``quality_adaptation_export_parity.context_for_export``. The frozen base
is a *separate* same-frame comparator and never replaces the primary model.

Full-video neural execution requires a serving admission earned by
measurement, never by a flag:

1. ``prepare_workbench_contract`` authenticates model/source/input bindings.
2. ``validate_metric_geometry`` measures manufactured plane/cube geometry
   (crop/native mapping, PnP invariance, unit boundaries, renderer depth) and
   requires deliberately wrong controls to fail.
3. ``run_fullimage_head_parity`` runs true RGB->backbone->decoder->head twice
   (reference ``best.pt['head']`` vs installed export) on frozen unlabelled
   cases and stores every compared array.
4. ``verify_serving_admission`` recomputes bindings and comparisons from the
   stored evidence. Admission grants experimental execution compatibility only,
   never model quality. Head-only parity alone cannot admit.

Nothing here downloads weights or runs a network on import. Running the gates
requires local weights and is scheduled by the parent operator only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import workbench_contract as wc

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "workbench_integer_metric_v1"
RECORD_RELATIVE = "bench/runtime/gotrack-flow-terminal-20261021.json"
RECORD_SHA256 = "05fe3012a18c2ed2669bcf2690e0b48650d9472fc1bffdf0980b1a253e6fc770"
SMALL_RECORDS_RELATIVE = ".cache/cloud-prepare/flow-terminal-small-records-20261021.json"
SMALL_RECORDS_SHA256 = "09226dda6f08c739b05c42540156784b4783be5bf3d65b34cf0549b9959104aa"
BACKUP_RECEIPT_RELATIVE = ".cache/cloud-prepare/flow-terminal-verified-private-backup-20261021.json"
BACKUP_RECEIPT_SHA256 = "ebce3be1d6cebbed217a7e7d6990eab47d46f644debb6ae8fad7f3f478408975"
LOADER_PARITY_RELATIVE = ".cache/cloud-prepare/flow-terminal-actual-loader-parity-20261021.json"
LOADER_PARITY_SHA256 = "5c30e71407ba12033c515bf23e9caf25d1822d491ba05dd2726e3c02a31101ae"
PRIVATE_BACKUP_HANDLE = "achitaan/visualizeit-flow-terminal-20261021-27e06a51"

PRIMARY = {
    "artifact_id": "flow-terminal-20261021-final285",
    "export_sha256": "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311",
    "export_bytes": 80_165_526,
    "base_checkpoint_sha256": "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086",
    "base_checkpoint_bytes": 1_608_850_339,
    "reference_checkpoint_sha256": "0c211531ffdee325e6504411f9dee07c83cc242cd8d048c115548aed6de2fa3b",
    "reference_checkpoint_bytes": 80_203_390,
    "training_config_sha256": "d11504b85172c4c404debc7defa948243520fbbfabf7b995f0031d2184c5269f",
    "training_lock_sha256": "44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33",
    "upstream_revision": "68f76055755f2a4a8967e13ece834f975f008bdf",
    "installer_sha256": "d1cd703ffe389ea72947d86a37ddbd42a065323af35c9e73547362027d3486b3",
}
HISTORICAL_OPTIONAL = {
    "existing-cached-pilot-20261020-285-updates": "520ba2ebe2357098d878dea6ba60e7b74fc9e96ce2010d21c2d73592d5004cd1",
    "head-lr1e6-20261020": "c795b5168c5598c28514614ccc8a36016b3e8dfd63e64e9f048a28ac67872baa",
}
DEFAULT_MODEL_DIR = ".cache/video-workbench/models/flow-terminal-20261021"
BASE_CHECKPOINT_RELATIVE = ".cache/model-quality/checkpoints/gotrack_checkpoint.pt"
UPSTREAM_RELATIVE = ".cache/model-quality/sources/gotrack"

PARITY_TOLERANCE = {"float_atol": 1e-6, "float_rtol": 0.0, "discrete": "exact",
                    "declared_before_execution": True}
PARITY_SCHEMA = "workbench_fullimage_head_parity_v1"
GEOMETRY_SCHEMA = "workbench_metric_geometry_validation_v1"
CONTRACT_SCHEMA = "workbench_integer_metric_contract_v1"
ADMISSION_SCHEMA = "workbench_serving_admission_v1"
RECORDED_ARTIFACT_SCHEMA = "workbench_fresh_recorded_artifact_v1"
RECORDED_EXECUTION_SCOPE = "finite_recorded_only"
RECORDED_LINEAGE = "fresh_local_pinned_artifact_verification_not_historical_receipts"
CONFIG_METADATA_VERIFICATION = "identity_crossbound_in_authenticated_export_and_reference_metadata"
CONFIG_DICTIONARY_VERIFICATION = "canonical_dictionary_and_authenticated_export_and_reference_metadata"
CONFIDENCE_THRESHOLD = 0.3
SAMPLING_MAX_ERROR_PX = 1e-3

# Production files reused (not modified) by this profile; bound by current hash.
REUSED_SOURCES = (
    "bench/quality_gotrack.py", "bench/quality_gotrack_inference.py",
    "bench/quality_adaptation_export_parity.py", "bench/quality_adaptation_data.py",
    "bench/quality_capture.py", "bench/quality_assets.py", "bench/quality_contract.py",
    "bench/vision.py",
)
NEW_SOURCES = (
    "bench/workbench_neural.py", "bench/workbench_contract.py", "bench/workbench_geometry.py",
    "bench/workbench_cli.py", "bench/workbench_telemetry.py", "bench/workbench_render.py", RECORD_RELATIVE,
    "bench/workbench_reconstruction.py",  # CLI failure records call its bounded-text helpers.
    "bench/workbench_frame_boundary.py",  # the CLI executes every neural frame through this boundary.
    "bench/workbench_live_pose.py",  # explicitly rejects fresh recorded-only evidence.
)

# Explicit, narrow review map for THIS profile only. The historical training
# lock and installer pins are untouched; these two files were already modified
# before the workbench existed. Each entry pins both the historical training
# hash and the exact reviewed current hash; any other changed historical file,
# or any further change to these two, is unreviewed and blocks admission.
REVIEWED_REPLACEMENTS = MappingProxyType({
    "bench/quality_gotrack.py": MappingProxyType({
        "training_sha256": "76df6cb3c1368d1d723acd5e00cde83a62bbd153ef032e54e2845998ff6d09ae",
        "current_sha256": "0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29",
        "scope": "workbench_integer_metric_v1 execution compatibility only",
        "reason": "WorkbenchIntegerRefiner executes the unchanged load_network, TexturedRenderer (legacy_v1, "
                  "integer_centers_v1) and _integer_profile_correspondences paths; the changed GoTrackRefiner, "
                  "ranker and capture paths are not executed by this profile",
    }),
    "bench/quality_runner.py": MappingProxyType({
        "training_sha256": "2266cadf050a884c36a38a3cc39ef591176a190e9dbbe830ac3141aae761b70c",
        "current_sha256": "bca306d1342dce2566c786fed94a7564ae090e11bf42243da2ea7bc9ce841ad1",
        "scope": "workbench_integer_metric_v1 execution compatibility only",
        "reason": "not imported or executed by the workbench integer-metric profile",
    }),
})
REVIEW_EVIDENCE = MappingProxyType({
    "path": ".cache/workbench-historical-source-replacement.diff",
    "sha256": "30520de45084ce1f948929e38331eb3b4691be9a731d15381d487c54ed262398",
    "reviewer": "parent Codex source trace of the actual WorkbenchIntegerRefiner path",
    "quality_claim": False,
})


def review_map() -> dict:
    return {"replacements": {k: dict(v) for k, v in REVIEWED_REPLACEMENTS.items()},
            "evidence": dict(REVIEW_EVIDENCE), "profile": PROFILE}


class NeuralUnavailable(RuntimeError):
    """A real prerequisite for neural execution is missing; nothing was run."""


class AdmissionError(ValueError):
    """Evidence does not support serving admission."""


# --------------------------------------------------------------------------
# hashing helpers

def sha256_file(path: Path, *, limit: int | None = None) -> str:
    h = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            total += len(block)
            if limit is not None and total > limit:
                raise AdmissionError(f"{path} exceeds its byte bound")
            h.update(block)
    return h.hexdigest()


def array_digest(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _regular(path: Path, name: str) -> Path:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise NeuralUnavailable(f"{name} is not a regular local file: {p}")
    return p.resolve(strict=True)


def _authenticated_bytes(path: Path, size: int, digest: str, name: str) -> bytes:
    p = _regular(path, name)
    if p.stat().st_size != size:
        raise AdmissionError(f"{name} byte size differs from the pinned artifact")
    raw = p.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise AdmissionError(f"{name} SHA-256 differs from the pinned artifact")
    return raw


# --------------------------------------------------------------------------
# primary record

def _verify_runtime_record(root: Path) -> dict:
    """Authenticate the already-pinned public record without making a history claim."""
    from . import quality_adaptation_export_parity as parity
    record, manifest_sha = parity.read_manifest(root / RECORD_RELATIVE, RECORD_SHA256)
    for key in ("export_sha256", "export_bytes", "base_checkpoint_sha256", "reference_checkpoint_sha256",
                "reference_checkpoint_bytes", "training_config_sha256", "training_lock_sha256",
                "upstream_revision", "installer_sha256", "artifact_id"):
        if record[key] != PRIMARY[key]:
            raise AdmissionError(f"runtime record {key} differs from the pinned primary identity")
    return {"record": record, "record_sha256": manifest_sha}


def _pinned_file_binding(path: Path, name: str, digest: str, size: int | None = None) -> dict:
    p = _regular(path, name)
    actual_size = p.stat().st_size
    if size is not None and actual_size != size:
        raise AdmissionError(f"{name} byte size differs from the pinned artifact")
    if sha256_file(p, limit=size) != digest:
        raise AdmissionError(f"{name} SHA-256 differs from the pinned artifact")
    return {"path": str(p), "bytes": actual_size, "sha256": digest}


def _checkpoint_training_metadata(path: Path, size: int, digest: str, *, torch_module=None) -> dict:
    """Read only after exact byte authentication, always with safe weights-only load.

    This imports Torch but constructs/runs no model. No pickle fallback or extra
    safe-global allowlist is permitted. Tensor payloads are discarded immediately.
    """
    raw = _authenticated_bytes(path, size, digest, "training-identity checkpoint")
    if torch_module is None:
        import torch as torch_module
    loaded = torch_module.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("head"), Mapping):
        raise AdmissionError("authenticated checkpoint has no head metadata mapping")
    fields = ("schema", "config_sha256", "source_lock_sha256", "dataset_sha256",
              "base_checkpoint_sha256", "deployment", "quality_claim")
    return {key: loaded[key] for key in fields if key in loaded}


def create_recorded_artifact_proof(*, root: Path = ROOT, head_path: Path, reference_path: Path,
                                   training_config_path: Path | None = None,
                                   training_config_metadata_only: bool = False) -> dict:
    """Earn NEW recorded-only artifact evidence; not historical training certification.

    The caller explicitly chooses an authenticated config dictionary OR identity
    crossbinding in exact runtime/export/reference metadata when the dictionary
    is unavailable. This never reconstructs historical small/backup/loader records,
    grants no admission, and is never accepted for live use. No model is run.
    """
    if type(training_config_metadata_only) is not bool or ((training_config_path is not None)
                                                         == training_config_metadata_only):
        raise AdmissionError("choose exactly one: training config dictionary or explicit metadata-only identity")
    root = Path(root).resolve(strict=True)
    verified = _verify_runtime_record(root)
    record = verified["record"]
    files = {
        "head": _pinned_file_binding(head_path, "primary head", PRIMARY["export_sha256"], PRIMARY["export_bytes"]),
        "reference": _pinned_file_binding(reference_path, "reference best.pt", PRIMARY["reference_checkpoint_sha256"],
                                           PRIMARY["reference_checkpoint_bytes"]),
        "base": _pinned_file_binding(root / BASE_CHECKPOINT_RELATIVE, "frozen base", PRIMARY["base_checkpoint_sha256"],
                                      PRIMARY["base_checkpoint_bytes"]),
        "installer": _pinned_file_binding(root / "bench/quality_gotrack_inference.py", "atomic installer",
                                           PRIMARY["installer_sha256"]),
        "training_lock": _pinned_file_binding(root / "bench/runtime/adaptation-sources.json", "training source lock",
                                               PRIMARY["training_lock_sha256"]),
        "runtime_record": _pinned_file_binding(root / RECORD_RELATIVE, "runtime record", RECORD_SHA256),
    }
    headers = {}
    for role, schema in (("head", "correspondence_head_export_v1"),
                         ("reference", "correspondence_head_checkpoint_v1")):
        binding = files[role]
        header = _checkpoint_training_metadata(Path(binding["path"]), binding["bytes"], binding["sha256"])
        for key, expected in (("schema", schema), ("config_sha256", record["training_config_sha256"]),
                              ("source_lock_sha256", record["training_lock_sha256"]),
                              ("dataset_sha256", record["dataset_sha256"])):
            if header.get(key) != expected:
                raise AdmissionError(f"{role} checkpoint {key} differs from pinned runtime identity")
        if role == "head" and (header.get("base_checkpoint_sha256") != record["base_checkpoint_sha256"]
                or header.get("deployment") != record["export_deployment"] or header.get("quality_claim") is not False):
            raise AdmissionError("export checkpoint base/deployment/quality metadata differs")
        headers[role] = header
    config_identity = {"identity_sha256": record["training_config_sha256"],
                       "verification": CONFIG_METADATA_VERIFICATION, "raw_config_available": False,
                       "historical_training_reproduction_claim": False,
                       "checkpoint_deserialization": "torch.load(weights_only=True,map_location=cpu); no model execution"}
    if training_config_path is not None:
        config_path = _regular(training_config_path, "training configuration")
        if config_path.stat().st_size > 1 << 20:
            raise AdmissionError("training configuration exceeds 1 MiB")
        config_raw = config_path.read_bytes()
        config = wc.loads_strict(config_raw)
        # Historical config identity is canonical JSON, NOT presentation bytes.
        config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()
        if not isinstance(config, dict) or config_sha != PRIMARY["training_config_sha256"]:
            raise AdmissionError("training configuration canonical identity differs from the pinned artifact")
        files["training_config"] = {"path": str(config_path), "bytes": len(config_raw),
                                    "sha256": hashlib.sha256(config_raw).hexdigest(), "canonical_sha256": config_sha}
        config_identity.update(verification=CONFIG_DICTIONARY_VERIFICATION, raw_config_available=True)
    closure = source_closure(root, include_upstream=True)
    if (closure["historical_missing"] or closure["unreviewed_replacements"]
            or not closure["review_evidence_present"] or any(v is None for v in closure["current_files"].values())):
        raise AdmissionError("complete pinned/reviewed source closure is required for fresh recorded proof")
    body = {"schema": RECORDED_ARTIFACT_SCHEMA, "execution_scope": RECORDED_EXECUTION_SCOPE,
            "lineage": RECORDED_LINEAGE, "repository_root": str(root), "record_sha256": verified["record_sha256"],
            "primary": dict(PRIMARY), "files": files, "training_configuration": config_identity,
            "authenticated_checkpoint_metadata": headers, "source_closure": closure, "runtime_packages": runtime_packages(),
            "historical_small_records_verified": False, "historical_receipts_verified": False,
            "live_eligible": False, "execution_admission_granted": False, "quality_claim": False}
    return {**body, "proof_sha256": wc.canonical_sha256(body)}


def verify_recorded_artifact_proof(proof: Mapping[str, Any], *, root: Path = ROOT) -> dict:
    """Recompute every binding. A self-hash or stored positive verdict is not evidence."""
    if not isinstance(proof, Mapping):
        raise AdmissionError("recorded artifact proof must be a mapping")
    body = dict(proof)
    stored = body.pop("proof_sha256", None)
    if body.get("schema") != RECORDED_ARTIFACT_SCHEMA or wc.canonical_sha256(body) != stored:
        raise AdmissionError("recorded artifact proof schema or self-hash differs")
    if body.get("repository_root") != str(Path(root).resolve(strict=True)):
        raise AdmissionError("recorded artifact proof belongs to another repository location; earn a fresh proof")
    try:
        files = body["files"]
        config = body["training_configuration"]
        if type(config["raw_config_available"]) is not bool:
            raise AdmissionError("raw config availability must be a boolean")
        raw_available = config["raw_config_available"]
        config_path = Path(files["training_config"]["path"]) if raw_available else None
        fresh = create_recorded_artifact_proof(root=root, head_path=Path(files["head"]["path"]),
            reference_path=Path(files["reference"]["path"]), training_config_path=config_path,
            training_config_metadata_only=not raw_available)
    except (KeyError, TypeError) as exc:
        raise AdmissionError("recorded artifact proof file bindings are invalid") from exc
    if dict(proof) != fresh:
        raise AdmissionError("recorded artifact proof drift: scope, identity, files, source or runtime differs")
    return fresh


def load_recorded_artifact_proof(path: Path, *, root: Path = ROOT) -> dict:
    p = _regular(path, "recorded artifact proof")
    if p.stat().st_size > 1 << 20:
        raise AdmissionError("recorded artifact proof exceeds 1 MiB")
    return verify_recorded_artifact_proof(wc.loads_strict(p.read_bytes()), root=root)


def _recorded_lineage(proof: Mapping[str, Any]) -> dict:
    return {"mode": RECORDED_ARTIFACT_SCHEMA, "proof_sha256": proof["proof_sha256"],
            "lineage": RECORDED_LINEAGE, "historical_small_records_verified": False,
            "historical_receipts_verified": False,
            "training_config_verification": proof["training_configuration"]["verification"],
            "raw_training_config_available": proof["training_configuration"]["raw_config_available"]}


def _contract_recorded_proof(contract: WorkbenchContract, proof: Mapping[str, Any] | None,
                             *, root: Path = ROOT) -> None:
    expected = contract.body.get("recorded_artifact_proof")
    if expected is None and proof is None:
        return  # historical install/verification remains unchanged
    if expected is None or proof is None or dict(proof) != expected:
        raise AdmissionError("explicit recorded artifact opt-in must match the contract; modes cannot be mixed")
    verify_recorded_artifact_proof(proof, root=root)
    if (contract.body.get("execution_scope") != RECORDED_EXECUTION_SCOPE
            or contract.body.get("live_eligible") is not False
            or contract.body.get("artifact_lineage") != _recorded_lineage(proof)):
        raise AdmissionError("recorded-only contract scope/lineage differs")


def verify_primary_record(root: Path = ROOT, *, recorded_artifact: Mapping[str, Any] | None = None) -> dict:
    """Default: exact historical small-records. Opt-in: NEW finite-recorded-only proof."""
    if recorded_artifact is not None:
        proof = verify_recorded_artifact_proof(recorded_artifact, root=root)
        return {**_verify_runtime_record(root), "artifact_lineage": _recorded_lineage(proof)}
    from . import quality_adaptation_export_parity as parity
    record_path = root / RECORD_RELATIVE
    record, manifest_sha = parity.read_manifest(record_path, RECORD_SHA256)
    small = root / SMALL_RECORDS_RELATIVE
    if not small.is_file():
        raise NeuralUnavailable(f"source small-records file is missing: {SMALL_RECORDS_RELATIVE}")
    if sha256_file(small) != SMALL_RECORDS_SHA256:
        raise AdmissionError("small-records SHA-256 differs from the pinned value")
    source = wc.loads_strict(small.read_bytes())["inference-artifact.json"]
    if source != record:
        raise AdmissionError("runtime record differs from the exact inference-artifact.json")
    for key in ("export_sha256", "export_bytes", "base_checkpoint_sha256", "reference_checkpoint_sha256",
                "reference_checkpoint_bytes", "training_config_sha256", "training_lock_sha256",
                "upstream_revision", "installer_sha256", "artifact_id"):
        if record[key] != PRIMARY[key]:
            raise AdmissionError(f"runtime record {key} differs from the pinned primary identity")
    return {"record": record, "record_sha256": manifest_sha}


def primary_identity() -> dict:
    return {"role": "primary_neural", "artifact_id": PRIMARY["artifact_id"],
            "export_sha256": PRIMARY["export_sha256"], "export_bytes": PRIMARY["export_bytes"],
            "base_checkpoint_sha256": PRIMARY["base_checkpoint_sha256"], "profile": PROFILE,
            "installed_head_sha256": None, "installed_head_state_sha256": None, "quality_claim": False}


def baseline_identity() -> dict:
    return {"role": "frozen_baseline", "base_checkpoint_sha256": PRIMARY["base_checkpoint_sha256"],
            "head": "frozen_base_pose_head", "profile": PROFILE, "installed_head_sha256": None,
            "installed_head_state_sha256": None, "quality_claim": False}


def module_state_sha256(module) -> str:
    """Digest of the actual installed tensors (sorted keys, dtype, shape, bytes)."""
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
        h.update(array.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# preflight

def _check(name: str, ok: bool, detail: str, action: str | None = None) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "action": None if ok else action}


def neural_preflight(*, root: Path = ROOT, head_path: Path | None = None, reference_path: Path | None = None,
                     verify_base_hash: bool = False, admission_path: Path | None = None,
                     recorded_artifact: Mapping[str, Any] | None = None) -> dict:
    """Diagnose every real prerequisite without importing torch or running a network."""
    import importlib.util
    head_path = Path(head_path) if head_path else root / DEFAULT_MODEL_DIR / "head.pt"
    reference_path = Path(reference_path) if reference_path else root / DEFAULT_MODEL_DIR / "best.pt"
    checks = []
    try:
        verify_primary_record(root, recorded_artifact=recorded_artifact)
        detail = (f"fresh local recorded-only proof {recorded_artifact['proof_sha256']}; historical records NOT verified"
                  if recorded_artifact is not None else f"{RECORD_RELATIVE} matches {SMALL_RECORDS_RELATIVE}")
        checks.append(_check("primary_record", True, detail))
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        checks.append(_check("primary_record", False, f"{type(exc).__name__}: {exc}",
                             "restore the exact runtime record and small-records file"))
    for name, rel, digest in (("backup_receipt", BACKUP_RECEIPT_RELATIVE, BACKUP_RECEIPT_SHA256),
                              ("loader_parity_receipt", LOADER_PARITY_RELATIVE, LOADER_PARITY_SHA256)):
        p = root / rel
        ok = p.is_file() and sha256_file(p) == digest
        checks.append(_check(name, ok, rel, f"restore {rel} (SHA {digest[:12]})"))
    try:
        installer = root / "bench/quality_gotrack_inference.py"
        ok = sha256_file(installer) == PRIMARY["installer_sha256"]
        checks.append(_check("atomic_installer", ok, "quality_gotrack_inference.py hash",
                             "installer source changed; restore the reviewed d1cd703f bytes"))
    except OSError as exc:
        checks.append(_check("atomic_installer", False, str(exc), "restore installer source"))
    if head_path.is_file():
        size = head_path.stat().st_size
        ok = size == PRIMARY["export_bytes"] and sha256_file(head_path) == PRIMARY["export_sha256"]
        checks.append(_check("primary_head_weights", ok, f"{head_path} ({size} bytes)",
                             "head.pt bytes/hash differ from 27e06a51; replace with the exact export"))
    else:
        checks.append(_check("primary_head_weights", False, f"missing {head_path}",
                             f"operator copies run/head.pt from private backup {PRIVATE_BACKUP_HANDLE} v1 "
                             f"to {head_path} (no automatic download)"))
    if reference_path.is_file():
        size = reference_path.stat().st_size
        ok = size == PRIMARY["reference_checkpoint_bytes"] and sha256_file(reference_path) == PRIMARY["reference_checkpoint_sha256"]
        checks.append(_check("reference_best_checkpoint", ok, f"{reference_path} ({size} bytes)",
                             "best.pt differs from 0c211531"))
    else:
        checks.append(_check("reference_best_checkpoint", False, f"missing {reference_path}",
                             "needed only for full-image parity: copy run/best.pt from the same private backup"))
    base = root / BASE_CHECKPOINT_RELATIVE
    if base.is_file():
        size = base.stat().st_size
        ok = size == PRIMARY["base_checkpoint_bytes"]
        detail = f"{size} bytes"
        if ok and verify_base_hash:
            ok = sha256_file(base) == PRIMARY["base_checkpoint_sha256"]
            detail += ", SHA verified" if ok else ", SHA mismatch"
        elif ok:
            detail += " (SHA verified at load by quality_gotrack.load_network)"
        checks.append(_check("base_checkpoint", ok, detail, "restore the pinned f7d127ab base checkpoint"))
    else:
        checks.append(_check("base_checkpoint", False, f"missing {BASE_CHECKPOINT_RELATIVE}",
                             "restore the pinned base checkpoint (no automatic download here)"))
    checks.append(_check("upstream_sources", (root / UPSTREAM_RELATIVE).is_dir(), UPSTREAM_RELATIVE,
                         "restore pinned upstream GoTrack sources"))
    torch_ok = importlib.util.find_spec("torch") is not None
    checks.append(_check("torch_runtime", torch_ok, "torch importable" if torch_ok else "torch missing",
                         "use the reviewed GoTrack runtime environment"))
    admission = None
    if admission_path is not None and Path(admission_path).is_file():
        admission = str(admission_path)
    checks.append(_check("serving_admission", admission is not None,
                         admission or "no verified workbench_integer_metric_v1 admission",
                         "earn admission: prepare_workbench_contract -> validate_metric_geometry (with measured "
                         "renderer/sampler) -> run_fullimage_head_parity -> verify_serving_admission"))
    ready_to_install = all(c["ok"] for c in checks if c["name"] in (
        "primary_record", "atomic_installer", "primary_head_weights", "base_checkpoint", "upstream_sources",
        "torch_runtime"))
    return {
        "schema": "workbench_neural_preflight_v1", "profile": PROFILE, "primary": primary_identity(),
        "checks": checks, "ready_to_install_head": ready_to_install,
        "geometry_admission": False if admission is None else "unverified_path_supplied",
        "status": "ready_for_admission_verification" if ready_to_install and admission else "blocked",
        "blocked_reasons": [c["name"] for c in checks if not c["ok"]],
        "note": "Head-only cached-token parity is not full-image admission; no network was run.",
    }


# --------------------------------------------------------------------------
# head installation (real APIs)

def install_primary_head(network, head_path: Path, *, device: str = "cpu", torch_module=None,
                         recorded_artifact: Mapping[str, Any] | None = None):
    """Install the exact flow-terminal export through the unchanged atomic installer."""
    from . import quality_adaptation_export_parity as parity
    from . import quality_gotrack_inference as installer
    verified = verify_primary_record(recorded_artifact=recorded_artifact)
    head_path = _regular(head_path, "primary head export")
    if head_path.stat().st_size != PRIMARY["export_bytes"]:
        raise AdmissionError("primary head byte size differs")
    context = parity.context_for_export(verified["record"], verified["record_sha256"], head_path)
    network, provenance = installer.install_adapted_head(network, context, device=device, torch_module=torch_module)
    provenance = dict(provenance)
    provenance.update({"profile": PROFILE, "role": "primary_neural",
                       "installer_sha256": PRIMARY["installer_sha256"],
                       "installed_head_state_sha256": module_state_sha256(network.pose_head)})
    if recorded_artifact is not None:
        provenance.update(artifact_lineage=verified["artifact_lineage"], execution_scope=RECORDED_EXECUTION_SCOPE,
                          live_eligible=False)
    provenance["install_digest"] = wc.canonical_sha256(provenance)
    return network, provenance


def load_reference_head(network, reference_path: Path, *, device: str = "cpu", torch_module=None):
    """Atomically load ``best.pt['head']`` (a head state, not a full network)."""
    if torch_module is None:
        import torch as torch_module
    raw = _authenticated_bytes(reference_path, PRIMARY["reference_checkpoint_bytes"],
                               PRIMARY["reference_checkpoint_sha256"], "reference best.pt")
    loaded = torch_module.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("head"), Mapping):
        raise AdmissionError("reference checkpoint has no ['head'] state dictionary")
    live = getattr(network, "pose_head", None)
    if live is None:
        raise AdmissionError("network has no pose_head")
    expected = live.state_dict()
    state = loaded["head"]
    if set(state) != set(expected):
        raise AdmissionError("reference head keys differ from the frozen pose head")
    for key, value in state.items():
        if tuple(value.shape) != tuple(expected[key].shape) or value.dtype != expected[key].dtype:
            raise AdmissionError(f"reference head tensor {key} shape/dtype differs")
    candidate = copy.deepcopy(live)
    candidate.load_state_dict(state, strict=True)
    candidate.to(device)
    candidate.eval()
    candidate.requires_grad_(False)
    network.pose_head = candidate
    provenance = {"role": "reference_best_head", "reference_checkpoint_sha256": PRIMARY["reference_checkpoint_sha256"],
                  "profile": PROFILE, "installed_head_state_sha256": module_state_sha256(network.pose_head)}
    provenance["install_digest"] = wc.canonical_sha256(provenance)
    return network, provenance


def load_base_network(device: str = "cpu"):
    """Strict frozen base network via the existing loader (verifies f7d127ab)."""
    from . import quality_gotrack
    return quality_gotrack.load_network(device)


def load_primary_network(head_path: Path, device: str = "cpu", *,
                         recorded_artifact: Mapping[str, Any] | None = None):
    if recorded_artifact is not None:
        verify_recorded_artifact_proof(recorded_artifact)
    network = load_base_network(device)
    return install_primary_head(network, head_path, device=device, recorded_artifact=recorded_artifact)


def load_baseline_network(device: str = "cpu"):
    """Frozen base network; its actual pose head is identified by a tensor digest."""
    network = load_base_network(device)
    provenance = dict(baseline_identity())
    provenance["installed_head_state_sha256"] = module_state_sha256(network.pose_head)
    provenance["install_digest"] = wc.canonical_sha256(provenance)
    return network, provenance


# --------------------------------------------------------------------------
# source closure

def historical_source_map(root: Path = ROOT) -> dict:
    from . import quality_gotrack_inference as installer
    lock_path = root / installer.TRAINING_LOCK_RELATIVE_PATH
    if sha256_file(lock_path) != PRIMARY["training_lock_sha256"]:
        raise AdmissionError("historical training source lock changed")
    lock = wc.loads_strict(lock_path.read_bytes())
    files = lock["gotrack"]["files"]
    if lock["gotrack"].get("revision") != PRIMARY["upstream_revision"]:
        raise AdmissionError("historical upstream revision differs")
    return dict(files)


def source_closure(root: Path = ROOT, *, include_upstream: bool = True) -> dict:
    """Bind current code independently of the historical aggregate hash.

    The historical training map is retained verbatim. Every listed file whose
    current hash differs is classified: ``reviewed`` only when it exactly
    matches an entry of ``REVIEWED_REPLACEMENTS`` (path, historical training
    hash and current hash); everything else is ``unreviewed`` and blocks
    admission. New/reused workbench files and the review map itself are hashed
    individually. Historical hashes do not cover new code.
    """
    historical = historical_source_map(root)
    current_hist, replacements, missing = {}, {}, []
    reviewed, unreviewed = {}, {}
    if include_upstream:
        for rel, expected in sorted(historical.items()):
            p = root / rel
            if not p.is_file():
                missing.append(rel)
                continue
            actual = sha256_file(p)
            current_hist[rel] = actual
            if actual != expected:
                row = {"training_sha256": expected, "current_sha256": actual}
                replacements[rel] = row
                entry = REVIEWED_REPLACEMENTS.get(rel)
                if (entry is not None and entry["training_sha256"] == expected
                        and entry["current_sha256"] == actual):
                    reviewed[rel] = {**row, "scope": entry["scope"], "reason": entry["reason"]}
                else:
                    why = ("not in the reviewed replacement map" if entry is None else
                           "historical pin differs from the review map" if entry["training_sha256"] != expected else
                           "current hash differs from the reviewed hash")
                    unreviewed[rel] = {**row, "reason": why}
    evidence = root / REVIEW_EVIDENCE["path"]
    evidence_ok = evidence.is_file() and sha256_file(evidence) == REVIEW_EVIDENCE["sha256"]
    current = {}
    for rel in REUSED_SOURCES + NEW_SOURCES:
        p = root / rel
        current[rel] = sha256_file(p) if p.is_file() else None
    review = review_map()
    body = {"historical_map_sha256": wc.canonical_sha256(historical), "historical_file_count": len(historical),
            "historical_missing": missing, "replacements": replacements,
            "reviewed_replacements": reviewed, "unreviewed_replacements": unreviewed,
            "review_map_sha256": wc.canonical_sha256(review), "review_evidence_present": evidence_ok,
            "current_files": current, "upstream_checked": include_upstream}
    body["closure_sha256"] = wc.canonical_sha256({"current": current, "replacements": replacements,
                                                   "historical_current": current_hist,
                                                   "review_map_sha256": body["review_map_sha256"]})
    return body


def _render_factory(*, require_pins: bool = False) -> dict:
    """Zero-storage renderer factory identity (policy + pinned sources + adapter hash)."""
    from . import workbench_render
    return workbench_render.factory_identity(require_pins=require_pins)


def runtime_packages() -> dict:
    out = {"python": ".".join(map(str, sys.version_info[:3])), "platform": platform.platform()}
    for name in ("torch", "numpy", "opencv-python", "opencv-python-headless", "pyrender", "trimesh"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


# --------------------------------------------------------------------------
# metric mesh (GLB) authentication without third-party loaders

def _reject_external_resources(gltf: Mapping[str, Any]) -> None:
    """Every GLB resource must be inside the hashed bytes: buffers/images may use
    the embedded BIN chunk, bufferViews or ``data:`` URIs only. External URIs
    (files or network) would let a loader read content the hash never covered."""
    for kind in ("buffers", "images"):
        for item in gltf.get(kind, []) or []:
            uri = item.get("uri") if isinstance(item, Mapping) else None
            if uri is not None and not (isinstance(uri, str) and uri.startswith("data:")):
                raise AdmissionError(f"GLB {kind} reference external URI content; only embedded resources are allowed")


def glb_inventory(path: Path, *, max_bytes: int = 200_000_000) -> dict:
    """Parse the GLB container: node transforms, meshes, textures, buffer hash."""
    p = _regular(path, "metric mesh GLB")
    raw = p.read_bytes()
    if len(raw) > max_bytes or len(raw) < 20:
        raise AdmissionError("GLB size outside bounds")
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise AdmissionError("not a glTF 2.0 binary container")
    offset, gltf, bin_chunk = 12, None, b""
    while offset < len(raw):
        clen, ctype = struct.unpack_from("<I4s", raw, offset)
        chunk = raw[offset + 8: offset + 8 + clen]
        if ctype == b"JSON":
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            bin_chunk = chunk
        offset += 8 + clen
    if not isinstance(gltf, dict):
        raise AdmissionError("GLB has no JSON chunk")
    _reject_external_resources(gltf)
    nodes = gltf.get("nodes", [])
    transforms = []
    for node in nodes:
        transforms.append({k: node.get(k) for k in ("matrix", "translation", "rotation", "scale", "mesh", "children")
                           if k in node})
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "node_count": len(nodes), "node_transforms_sha256": wc.canonical_sha256(transforms),
            "mesh_count": len(gltf.get("meshes", [])), "texture_count": len(gltf.get("textures", [])),
            "image_count": len(gltf.get("images", [])), "binary_chunk_sha256": hashlib.sha256(bin_chunk).hexdigest(),
            "units_declared": "metres (glTF 2.0 specification)"}


def native_K(K: Sequence[float] | np.ndarray, principal_point_origin: str) -> np.ndarray:
    """Internal integer-centre K. Only a documented corner origin subtracts 0.5."""
    M = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    if principal_point_origin == "integer_centers":
        return M
    if principal_point_origin == "corner":
        M[0, 2] -= 0.5
        M[1, 2] -= 0.5
        return M
    raise AdmissionError("principal point origin must be documented as integer_centers or corner")


# --------------------------------------------------------------------------
# 1. contract

@dataclass(frozen=True)
class WorkbenchContract:
    body: dict
    sha256: str

    def to_dict(self) -> dict:
        return {**self.body, "contract_sha256": self.sha256}


def prepare_workbench_contract(cases: Sequence[Mapping[str, Any]], *, root: Path = ROOT,
                               head_path: Path, reference_path: Path, verify_base: bool = True,
                               include_upstream: bool = True,
                               recorded_artifact: Mapping[str, Any] | None = None) -> WorkbenchContract:
    """Authenticate model, source and per-case input bindings for the new profile.

    Each case: {case_id, family, rgb_path, rgb_sha256, mask_path, mask_sha256,
    segmentation: {producer, manifest_sha256}, K (9 floats),
    principal_point_origin, initial_pose_m (4x4), initialization_source,
    glb_path, glb_sha256, obj_id, metric_provenance}. ``initialization_source``
    must come from inference/onboarding; evaluator-derived values are rejected.
    Relative auto-reconstructed (object_unit) geometry is excluded.
    """
    if recorded_artifact is not None and (not verify_base or not include_upstream):
        raise AdmissionError("fresh recorded artifact proof requires complete base/source verification")
    verified = verify_primary_record(root, recorded_artifact=recorded_artifact)
    _authenticated_bytes(head_path, PRIMARY["export_bytes"], PRIMARY["export_sha256"], "primary export")
    _authenticated_bytes(reference_path, PRIMARY["reference_checkpoint_bytes"],
                         PRIMARY["reference_checkpoint_sha256"], "reference best.pt")
    base = root / BASE_CHECKPOINT_RELATIVE
    if not base.is_file() or base.stat().st_size != PRIMARY["base_checkpoint_bytes"]:
        raise NeuralUnavailable("base checkpoint missing or wrong size")
    if verify_base and sha256_file(base) != PRIMARY["base_checkpoint_sha256"]:
        raise AdmissionError("base checkpoint SHA-256 differs")
    if not cases:
        raise AdmissionError("contract needs at least one frozen case")
    families = set()
    rows = []
    for case in cases:
        case = dict(case)
        cid = wc._ident(case["case_id"], "case_id")
        family = wc._str(case["family"], "family", max_len=64)
        families.add(family)
        if case.get("metric_provenance") not in ("measured_metric_cad", "dataset_metric_cad"):
            raise AdmissionError(f"{cid}: only metric CAD geometry is admissible; relative reconstructions are excluded")
        init_src = str(case.get("initialization_source", ""))
        if not init_src.startswith(("inference:", "onboarding:")):
            raise AdmissionError(f"{cid}: initialization must come from inference/onboarding, not evaluator data")
        rgb_sha = sha256_file(_regular(Path(case["rgb_path"]), "case RGB"))
        if rgb_sha != case["rgb_sha256"]:
            raise AdmissionError(f"{cid}: RGB hash differs")
        mask_sha = sha256_file(_regular(Path(case["mask_path"]), "case mask"))
        if mask_sha != case["mask_sha256"]:
            raise AdmissionError(f"{cid}: mask hash differs")
        seg = case.get("segmentation")
        if not isinstance(seg, Mapping) or not seg.get("producer") or not wc.SHA_RE.fullmatch(str(seg.get("manifest_sha256", ""))):
            raise AdmissionError(f"{cid}: segmentation producer/manifest binding is required")
        glb = glb_inventory(Path(case["glb_path"]))
        if glb["sha256"] != case["glb_sha256"]:
            raise AdmissionError(f"{cid}: GLB hash differs")
        glb["path"] = str(Path(case["glb_path"]).resolve())
        K = native_K(case["K"], case["principal_point_origin"])
        pose = wc._pose(case["initial_pose_m"], f"{cid}.initial_pose_m")
        rows.append({"case_id": cid, "family": family, "rgb_sha256": rgb_sha, "mask_sha256": mask_sha,
                     "rgb_path": str(Path(case["rgb_path"]).resolve()), "mask_path": str(Path(case["mask_path"]).resolve()),
                     "segmentation": {"producer": str(seg["producer"]), "manifest_sha256": seg["manifest_sha256"]},
                     "native_K": K.tolist(), "principal_point_origin": case["principal_point_origin"],
                     "initial_pose_m": [list(r) for r in pose], "initialization_source": init_src,
                     "glb": glb, "obj_id": int(case["obj_id"]), "metric_provenance": case["metric_provenance"]})
    body = {"schema": CONTRACT_SCHEMA, "profile": PROFILE, "primary": primary_identity(),
            "record_sha256": verified["record_sha256"], "reference_checkpoint_sha256": PRIMARY["reference_checkpoint_sha256"],
            "base_checkpoint_sha256": PRIMARY["base_checkpoint_sha256"], "cases": rows,
            "families": sorted(families), "runtime_packages": runtime_packages(),
            "source_closure": source_closure(root, include_upstream=include_upstream),
            "renderer_factory": _render_factory(require_pins=True),
            "evaluator_inputs_read": False, "quality_claim": False}
    if recorded_artifact is not None:
        body.update(recorded_artifact_proof=dict(recorded_artifact), artifact_lineage=verified["artifact_lineage"],
                    execution_scope=RECORDED_EXECUTION_SCOPE, live_eligible=False)
    return WorkbenchContract(body, wc.canonical_sha256(body))


# --------------------------------------------------------------------------
# 2. metric geometry validation

def crop_rotation(Kn: np.ndarray, center_uv: Sequence[float]) -> np.ndarray:
    """Proper rotation R_cn (crop-from-native) aiming the crop axis at a native pixel."""
    ray = np.linalg.solve(Kn, np.array([center_uv[0], center_uv[1], 1.0]))
    z = ray / np.linalg.norm(ray)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows: crop axes in native coordinates


def native_from_crop_uv(q: np.ndarray, Kc: np.ndarray, Rcn: np.ndarray, Kn: np.ndarray) -> np.ndarray:
    """native ray = Kn * Rcn^T * inv(Kc) * q (integer-centre pixels both sides)."""
    h = np.column_stack([np.asarray(q, dtype=np.float64), np.ones(len(q))])
    rays = (Kn @ Rcn.T @ np.linalg.inv(Kc) @ h.T).T
    return rays[:, :2] / rays[:, 2:3]


def crop_sampling_map(Kn, Kc, Rcn, size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    uv = native_from_crop_uv(np.column_stack([xx.ravel(), yy.ravel()]), Kc, Rcn, Kn)
    return uv[:, 0].reshape(size, size), uv[:, 1].reshape(size, size)


def bilinear_sample(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Exact float64 bilinear sampling at integer-centre coordinates (reference sampler)."""
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    x0 = np.floor(map_x).astype(int)
    y0 = np.floor(map_y).astype(int)
    fx, fy = map_x - x0, map_y - y0
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return ((1 - fx) * (1 - fy) * img[y0c, x0c] + fx * (1 - fy) * img[y0c, x1c]
            + (1 - fx) * fy * img[y1c, x0c] + fx * fy * img[y1c, x1c])


def reference_crop_sampler(native_image, Kn, Kc, Rcn, size):
    mx, my = crop_sampling_map(Kn, Kc, Rcn, size)
    return bilinear_sample(native_image, mx, my)


def measure_crop_sampler(sampler: Callable, Kn, Kc, Rcn, size: int, native_hw: tuple[int, int]) -> dict:
    """Feed a linear coordinate ramp through ``sampler`` and compare with analytic UV."""
    h, w = native_hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = np.stack([xx, yy, np.zeros_like(xx)], axis=2)
    crop = np.asarray(sampler(ramp, Kn, Kc, Rcn, size), dtype=np.float64)
    mx, my = crop_sampling_map(Kn, Kc, Rcn, size)
    interior = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    if crop.shape[:2] != (size, size) or not interior.any():
        return {"max_error_px": None, "interior_samples": int(interior.sum()), "passed": False}
    err = np.maximum(np.abs(crop[..., 0] - mx), np.abs(crop[..., 1] - my))[interior]
    max_err = float(err.max())
    return {"max_error_px": max_err, "interior_samples": int(interior.sum()),
            "threshold_px": SAMPLING_MAX_ERROR_PX, "passed": max_err < SAMPLING_MAX_ERROR_PX,
            "crop_sha256": array_digest(crop.astype(np.float32))}


def _cube_points(size_mm: float) -> np.ndarray:
    s = size_mm / 2
    g = np.linspace(-s, s, 4)
    pts = np.array([[x, y, z] for x in g for y in g for z in g])
    return pts[np.max(np.abs(pts), axis=1) >= s - 1e-9]


def _project(K, pose, pts):
    cam = pts @ pose[:3, :3].T + pose[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]


def _rt(rx, ry, rz, t):
    from .vision import cv2
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P


def opengl_projection(K: np.ndarray, width: int, height: int, near: float, far: float) -> np.ndarray:
    """OpenGL projection for integer-centre K: the GL boundary adds +0.5."""
    cx, cy = K[0, 2] + 0.5, K[1, 2] + 0.5
    P = np.zeros((4, 4))
    P[0, 0] = 2 * K[0, 0] / width
    P[1, 1] = 2 * K[1, 1] / height
    P[0, 2] = 1 - 2 * cx / width
    P[1, 2] = 2 * cy / height - 1
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1
    return P


def _gl_window(Pgl, cam_cv, width, height):
    """Window coordinates (origin top-left corner) of OpenCV camera points."""
    cam_gl = cam_cv * np.array([1.0, -1.0, -1.0])
    clip = (Pgl @ np.column_stack([cam_gl, np.ones(len(cam_gl))]).T).T
    ndc = clip[:, :3] / clip[:, 3:4]
    wx = (ndc[:, 0] + 1) / 2 * width
    wy = (1 - (ndc[:, 1] + 1) / 2) * height
    return np.column_stack([wx, wy])


def validate_metric_geometry(*, renderer_probe: Callable | None = None, crop_sampler: Callable | None = None,
                             evidence_dir: Path | None = None, production: Mapping[str, Any] | None = None) -> dict:
    """Measured geometry checks on manufactured plane/cube scenes.

    ``renderer_probe(K, pose_mm, width, height) -> {depth_mm, allocation, render_policy}``
    must be the actual production renderer; ``crop_sampler(native, Kn, Kc, Rcn, size)``
    the actual production crop sampler, or pass ``production`` from
    ``measure_production_crop`` (actual upstream crop + template renderer).
    Without both measurements the result is ``analytic_only`` and
    admission-ineligible.
    """
    from .vision import cv2
    checks: dict[str, dict] = {}
    W, H, S = 1024, 1280, 280
    Kn = np.array([[910.0, 0, 498.25], [0, 905.0, 655.75], [0, 0, 1]])  # off-centre
    pose_mm = _rt(0.35, -0.6, 0.2, [35.0, -20.0, 650.0])
    cube = _cube_points(120.0)
    uv_n, z_n = _project(Kn, pose_mm, cube)
    centre = uv_n.mean(axis=0)
    Rcn = crop_rotation(Kn, centre)
    Kc = np.array([[1400.0, 0, (S - 1) / 2], [0, 1400.0, (S - 1) / 2], [0, 0, 1]])
    C = np.eye(4)
    C[:3, :3] = Rcn
    checks["crop_transform_proper"] = {"det": float(np.linalg.det(Rcn)),
                                       "orthonormal_error": float(np.abs(Rcn @ Rcn.T - np.eye(3)).max()),
                                       "translation": [0.0, 0.0, 0.0]}
    checks["crop_transform_proper"]["passed"] = (abs(checks["crop_transform_proper"]["det"] - 1) < 1e-12
                                                 and checks["crop_transform_proper"]["orthonormal_error"] < 1e-12)
    # crop-space projection -> native mapping
    crop_pose = C @ pose_mm
    uv_c, _ = _project(Kc, crop_pose, cube)
    back = native_from_crop_uv(uv_c, Kc, Rcn, Kn)
    err = float(np.abs(back - uv_n).max())
    checks["crop_to_native_projection"] = {"max_error_px": err, "passed": err < 1e-9}
    # unprojection: Xc = z inv(Kc)[u,v,1]; Xo = Rco^T (Xc - tco)
    cam_c = cube @ crop_pose[:3, :3].T + crop_pose[:3, 3]
    xc = cam_c[:, 2:3] * (np.column_stack([uv_c, np.ones(len(uv_c))]) @ np.linalg.inv(Kc).T)
    xo = (xc - crop_pose[:3, 3]) @ crop_pose[:3, :3]
    err = float(np.abs(xo - cube).max())
    checks["unprojection_roundtrip"] = {"max_error_mm": err, "passed": err < 1e-9}
    # flow target -> native ray, with a known synthetic flow
    flow = np.column_stack([np.sin(np.arange(len(uv_c))) * 3.0, np.cos(np.arange(len(uv_c))) * 2.0])
    q = uv_c + flow
    native_q = native_from_crop_uv(q, Kc, Rcn, Kn)
    Hnc = Kn @ Rcn.T @ np.linalg.inv(Kc)
    hq = np.column_stack([q, np.ones(len(q))]) @ Hnc.T
    err = float(np.abs(hq[:, :2] / hq[:, 2:3] - native_q).max())
    checks["flow_target_native_ray"] = {"max_error_px": err, "passed": err < 1e-9}
    # noiseless PnP in crop, transform by inv(C) to native, check invariance
    ok, rvec, tvec = cv2.solvePnP(cube, uv_c, Kc, None, flags=cv2.SOLVEPNP_SQPNP)
    rvec, tvec = cv2.solvePnPRefineLM(cube, uv_c, Kc, None, rvec, tvec)
    solved = np.eye(4)
    solved[:3, :3] = cv2.Rodrigues(rvec)[0]
    solved[:3, 3] = tvec.ravel()
    native_pose = np.linalg.inv(C) @ solved
    uv_solved, _ = _project(Kn, native_pose, cube)
    perr = float(np.abs(uv_solved - uv_n).max())
    terr = float(np.abs(native_pose[:3, 3] - pose_mm[:3, 3]).max())
    checks["noiseless_pnp_invariance"] = {"solver_ok": bool(ok), "native_reprojection_max_px": perr,
                                          "translation_max_error_mm": terr,
                                          "passed": bool(ok) and perr < 1e-6 and terr < 1e-4}
    # deliberately wrong controls must fail
    Kwrong = Kc.copy()
    Kwrong[0, 2] += 0.5
    Kwrong[1, 2] += 0.5
    ok_w, rv_w, tv_w = cv2.solvePnP(cube, uv_c, Kwrong, None, flags=cv2.SOLVEPNP_SQPNP)
    wrong = np.eye(4)
    wrong[:3, :3] = cv2.Rodrigues(rv_w)[0]
    wrong[:3, 3] = tv_w.ravel()
    uv_w, _ = _project(Kn, np.linalg.inv(C) @ wrong, cube)
    half_err = float(np.abs(uv_w - uv_n).max())
    checks["control_wrong_half_pixel"] = {"native_reprojection_max_px": half_err,
                                          "detected": half_err > 0.1, "passed": half_err > 0.1}
    ok_u, rv_u, tv_u = cv2.solvePnP(cube / 1000.0, uv_c, Kc, None, flags=cv2.SOLVEPNP_SQPNP)
    unit_t = tv_u.ravel()  # metres-as-millimetres misuse
    unit_err = float(np.abs(unit_t - crop_pose[:3, 3]).max())
    checks["control_1000_unit_factor"] = {"translation_error_mm": unit_err, "detected": unit_err > 1.0,
                                          "passed": unit_err > 1.0}
    # unit boundary: GLB metres -> render/PnP mm -> public metres
    pose_m = pose_mm.copy()
    pose_m[:3, 3] /= 1000.0
    roundtrip = pose_m.copy()
    roundtrip[:3, 3] *= 1000.0
    roundtrip[:3, 3] /= 1000.0
    checks["unit_boundary_roundtrip"] = {"max_error_m": float(np.abs(roundtrip - pose_m).max()),
                                         "passed": float(np.abs(roundtrip - pose_m).max()) < 1e-15,
                                         "chain": "glb_metres->render_pnp_millimetres->public_metres"}
    # OpenGL boundary: integer-centre pixel u lands at window x = u + 0.5
    Pgl = opengl_projection(Kn, W, H, 10.0, 5000.0)
    cam = cube @ pose_mm[:3, :3].T + pose_mm[:3, 3]
    win = _gl_window(Pgl, cam, W, H)
    gl_err = float(np.abs(win - (uv_n + 0.5)).max())
    checks["opengl_half_pixel_boundary"] = {"max_error_px": gl_err, "passed": gl_err < 1e-6}
    # crop sampling: reference always; production sampler only if supplied
    checks["reference_crop_sampling"] = measure_crop_sampler(reference_crop_sampler, Kn, Kc, Rcn, S, (H, W))
    measured = {}
    if crop_sampler is not None:
        measured["crop_sampler"] = measure_crop_sampler(crop_sampler, Kn, Kc, Rcn, S, (H, W))
    if renderer_probe is not None:
        measured["renderer"] = _measure_renderer(renderer_probe, Kn, W, H, evidence_dir)
    if production is not None:
        if "crop_sampler" in production:
            measured["crop_sampler"] = dict(production["crop_sampler"])
        if "template_renderer" in production:
            measured["template_renderer"] = dict(production["template_renderer"])
    analytic_ok = all(c.get("passed") for c in checks.values())
    measured_ok = (bool(measured) and all(k in measured for k in ("crop_sampler", "renderer", "template_renderer"))
                   and all(m.get("passed") for m in measured.values()))
    if not analytic_ok or any(not m.get("passed") for m in measured.values()):
        status = "failed"
    elif measured_ok:
        status = "measured"
    else:
        status = "analytic_only"  # includes partial measurement (renderer or sampler absent)
    body = {"schema": GEOMETRY_SCHEMA, "profile": PROFILE, "renderer_factory": _render_factory(),
            "checks": checks, "measured": measured,
            "analytic_passed": analytic_ok, "status": status,
            "admission_eligible": status == "measured",
            "note": "Manufactured scenes only; tests may inject fake probes, but admission requires the actual "
                    "production renderer and crop sampler measurements."}
    body["receipt_sha256"] = wc.canonical_sha256(body)
    return body


PLANE_HALF_MM = 150.0
DEPTH_MAX_RELATIVE_ERROR = 1e-4
COVERAGE_MIN_IOU = 0.995


def plane_depth_metrics(depth: np.ndarray, K: np.ndarray, pose_mm: np.ndarray, width: int, height: int) -> dict:
    """Compare a rendered metric-plane depth map with the independent rasterizer."""
    from . import workbench_geometry as wg
    half = PLANE_HALF_MM
    verts = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (height, width):
        return {"depth_max_relative_error": None, "coverage_iou": None, "boundary_mismatch_pixels": None,
                "reason": "depth shape differs"}
    ref = wg.rasterize(verts, faces, np.asarray(K, float), np.asarray(pose_mm, float), width, height).depth
    ref_cov = np.isfinite(ref)
    got_cov = np.isfinite(depth) & (depth > 0)
    both = ref_cov & got_cov
    union = ref_cov | got_cov
    rel = float(np.max(np.abs(depth[both] - ref[both]) / ref[both])) if both.any() else None
    return {"depth_max_relative_error": rel, "coverage_iou": float(both.sum() / max(union.sum(), 1)),
            "boundary_mismatch_pixels": int((union & ~both).sum())}


def depth_row_passes(row: Mapping[str, Any]) -> bool:
    rel, iou = row.get("depth_max_relative_error"), row.get("coverage_iou")
    return (isinstance(rel, (int, float)) and not isinstance(rel, bool) and math.isfinite(rel)
            and 0.0 <= rel < DEPTH_MAX_RELATIVE_ERROR and isinstance(iou, (int, float)) and not isinstance(iou, bool)
            and math.isfinite(iou) and iou > COVERAGE_MIN_IOU)


def save_depth_evidence(evidence_dir: Path | None, name: str, depth: np.ndarray, K: np.ndarray,
                        pose_mm: np.ndarray, width: int, height: int) -> dict | None:
    """Store the exact float64 depth plus the parameters needed to recompute the gate."""
    if evidence_dir is None:
        return None
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, np.asarray(depth, dtype=np.float64), allow_pickle=False)
    return {"file": path.name, "sha256": sha256_file(path), "K": np.asarray(K, float).tolist(),
            "pose_mm": np.asarray(pose_mm, float).tolist(), "width": int(width), "height": int(height)}


def _measure_renderer(probe: Callable, K: np.ndarray, W: int, H: int, evidence_dir: Path | None) -> dict:
    """Render a fronto-tilted metric plane; compare depth with an independent rasterizer."""
    pose = _rt(0.25, -0.15, 0.05, [10.0, 5.0, 700.0])
    out = probe(K, pose, W, H)
    depth = np.asarray(out.get("depth_mm"), dtype=np.float64)
    if depth.shape != (H, W):
        return {"passed": False, "reason": "renderer depth shape differs"}
    metrics = plane_depth_metrics(depth, K, pose, W, H)
    allocation = out.get("allocation")
    passed = (depth_row_passes(metrics) and isinstance(allocation, Mapping)
              and bool(allocation) and isinstance(out.get("render_policy"), str))
    result = {**metrics, "allocation": dict(allocation) if isinstance(allocation, Mapping) else None,
              "render_policy": out.get("render_policy"), "depth_sha256": array_digest(depth.astype(np.float32)),
              "evidence": save_depth_evidence(evidence_dir, "native_plane_depth_mm", depth, K, pose, W, H),
              "passed": bool(passed)}
    return result


# --------------------------------------------------------------------------
# 3. full-image head parity

def deterministic_torch(torch) -> dict:
    torch.manual_seed(0)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {"seed": 0, "tf32": False, "amp": False, "cudnn_deterministic": True}


class ForwardCounter:
    """Observe real module execution (not self-reported) via forward hooks."""

    MODULES = ("backbone", "decoder", "pose_head")

    def __init__(self, network):
        self.counts = {m: 0 for m in self.MODULES}
        self.captured: dict[str, Any] = {}
        self._handles = []
        for name in self.MODULES:
            module = getattr(network, name, None)
            if module is None or not hasattr(module, "register_forward_hook"):
                raise AdmissionError(f"network has no hookable {name}")
            self._handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(_module, _inputs, output):
            self.counts[name] += 1
            if name == "decoder":
                value = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else output
                if hasattr(value, "detach"):
                    self.captured["decoder_reference"] = value.detach().float().cpu().numpy().copy()
        return hook

    def take(self) -> dict:
        out, self.captured = self.captured, {}
        return out

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class WorkbenchIntegerRefiner:
    """Separate dispatch for ``workbench_integer_metric_v1`` (old registry untouched).

    One iteration: perspective crop around the initial pose, legacy_v1 unlit
    integer-centre template rendering, network forward, integer-grid
    correspondences, deterministic PnP; returns every intermediate as arrays.
    """

    def __init__(self, network, device: str = "cpu", *, renderer_factory: Callable | None = None):
        self.network = network
        self.device = device
        self.renderer_factory = renderer_factory
        self._renderers: dict[str, Any] = {}

    def _renderer(self, case):
        key = case["glb_sha256"]
        if key not in self._renderers:
            if self.renderer_factory is not None:
                self._renderers[key] = self.renderer_factory(case)
            else:
                from . import workbench_render as wr
                self._renderers[key] = wr.WorkbenchZeroStorageRenderer(case["glb_path"], int(case["obj_id"]))
        return self._renderers[key]

    def render_proof(self, case) -> dict | None:
        renderer = self._renderers.get(case["glb_sha256"])
        return getattr(renderer, "metadata", None)

    def trace(self, case: Mapping[str, Any]) -> dict:
        import torch
        from .vision import cv2
        from . import quality_gotrack as qg
        qg.upstream_path()
        from utils import data_util, misc, structs  # type: ignore
        rgb = case["rgb"]
        mask = case["mask"]
        K = np.asarray(case["native_K"], dtype=np.float64)
        pose_m = np.asarray(case["initial_pose_m"], dtype=np.float64)
        pose_mm = pose_m.copy()
        pose_mm[:3, 3] *= 1000.0
        renderer = self._renderer(case)
        h, w = rgb.shape[:2]
        camera = structs.PinholePlaneCameraModel(width=w, height=h, f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]),
                                                 T_world_from_eye=np.eye(4))
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None] / 255
        input_mask = torch.from_numpy((np.asarray(mask) > 0).copy()).float()[None]
        with torch.inference_mode():
            data, cameras, crop_from_orig = data_util.compute_gotrack_inputs_from_init_poses(
                input_rgbs=image, input_cameras=[camera],
                init_poses_cam_from_model=torch.from_numpy(pose_mm.astype(np.float32))[None],
                renderer=renderer, obj_ids=[int(case["obj_id"])], object_vertices=[renderer.vertices_m * 1000],
                crop_size=(280, 280), crop_rel_pad=.1, cropping_type="perspective_2d_box", ssaa_factor=1.,
                background_type="gray", input_masks=input_mask)
            template = data["templates"]
            crop_k = misc.get_intrinsic_matrix(cameras[0])
            flow_t, conf_t = self.network(data["crop_rgbs"].to(self.device), template.rgbs.to(self.device),
                                          template.masks.to(self.device))
        flow = flow_t[0].permute(1, 2, 0).float().cpu().numpy()
        conf = conf_t[0].float().cpu().numpy()
        (crop_pose, xyz, target, native_uv, confidence, valid, integer_pixels,
         _xyz_cam) = qg._integer_profile_correspondences(template.depths, crop_k, crop_from_orig[0], pose_mm, flow,
                                                          conf, template.masks[0], data["crop_masks"][0], K)
        retained = np.flatnonzero(valid.ravel()).astype(np.int64)
        obj = xyz[valid].astype(np.float64)
        tgt = target[valid].astype(np.float64)
        result = {"solver_state": np.array(0, np.int8), "ransac_inliers": np.zeros(0, np.int64),
                  "solved_native_pose_m": np.full((4, 4), np.nan)}
        if len(obj) >= 24:
            ids = np.linspace(0, len(obj) - 1, min(len(obj), 10000), dtype=int)
            cv2.setRNGSeed(int(case.get("rng_seed", 0)))
            rvec0 = cv2.Rodrigues(crop_pose[:3, :3])[0]
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj[ids], tgt[ids], crop_k, None, rvec0,
                                                     crop_pose[:3, 3].copy(), useExtrinsicGuess=True,
                                                     iterationsCount=100, reprojectionError=3.0,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= 6:
                chosen = ids[inl.ravel()]
                rvec, tvec = cv2.solvePnPRefineLM(obj[chosen], tgt[chosen], crop_k, None, rvec, tvec)
                refined = np.eye(4)
                refined[:3, :3] = cv2.Rodrigues(rvec)[0]
                refined[:3, 3] = tvec.ravel()
                native = np.linalg.inv(np.asarray(crop_from_orig[0], dtype=np.float64)) @ refined
                native[:3, 3] /= 1000.0
                result = {"solver_state": np.array(1, np.int8), "ransac_inliers": np.sort(chosen).astype(np.int64),
                          "solved_native_pose_m": native}
            else:
                result["solver_state"] = np.array(-1, np.int8)
        return {
            "crop_rgbs": data["crop_rgbs"].float().cpu().numpy(),
            "crop_masks": np.asarray(data["crop_masks"].float().cpu().numpy()),
            "template_rgbs": template.rgbs.float().cpu().numpy(),
            "template_masks": np.asarray(template.masks.cpu().numpy()).astype(bool),
            "template_depths_mm": template.depths.float().cpu().numpy(),
            "crop_k": np.asarray(crop_k, dtype=np.float64),
            "crop_from_orig": np.asarray(crop_from_orig[0], dtype=np.float64),
            "flow": flow.astype(np.float32), "confidence": conf.astype(np.float32),
            "confidence_mask": conf > CONFIDENCE_THRESHOLD,
            "retained_indices": retained, "native_uv": native_uv.astype(np.float64),
            **result,
        }

    def close(self):
        errors = []
        try:
            for renderer in list(self._renderers.values()):
                close = getattr(renderer, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001 - close all, then report
                        errors.append(exc)
        finally:
            self._renderers.clear()
            self.network = None  # drop the network reference held by this runner
            self.renderer_factory = None
        if errors:
            raise errors[0]


DISCRETE_KEYS = ("confidence_mask", "retained_indices", "ransac_inliers", "solver_state", "template_masks")
# NaN is a legitimate encoded value only here (unsolved pose, invalid native UV).
NAN_ALLOWED_KEYS = ("solved_native_pose_m", "native_uv")


def compare_traces(reference: Mapping[str, Any], candidate: Mapping[str, Any], tolerance=PARITY_TOLERANCE) -> dict:
    if set(reference) != set(candidate):
        return {"passed": False, "reason": "trace keys differ", "fields": {}}
    fields, passed = {}, True
    for key in sorted(reference):
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        row = {"shape": list(a.shape), "dtype": a.dtype.str}
        if a.shape != b.shape or a.dtype != b.dtype:
            row.update(passed=False, reason="shape/dtype differ")
        elif key in DISCRETE_KEYS or a.dtype.kind in "biu":
            row.update(passed=bool(np.array_equal(a, b)), comparison="exact")
        else:
            nan_ok = key in NAN_ALLOWED_KEYS
            both_nan = (np.isnan(a) & np.isnan(b)) if (nan_ok and a.dtype.kind == "f") else np.zeros(a.shape, bool)
            finite = np.isfinite(a) & np.isfinite(b)
            if not np.all(finite | both_nan):
                row.update(passed=False, reason="non-finite values" if not nan_ok else "non-finite mismatch")
            else:
                diff = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite]))) if finite.any() else 0.0
                limit = tolerance["float_atol"] + tolerance["float_rtol"] * (
                    float(np.max(np.abs(a[finite]))) if finite.any() else 0.0)
                row.update(max_abs_diff=diff, limit=limit, passed=diff <= limit, comparison="float")
        row["reference_sha256"] = array_digest(a)
        row["candidate_sha256"] = array_digest(b)
        passed &= row["passed"]
        fields[key] = row
    return {"passed": bool(passed), "fields": fields}


def decode_case_bytes(rgb_raw: bytes, mask_raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Single decoding rule for parity/serving cases: RGB uint8 HxWx3, mask = gray > 127."""
    from .vision import cv2
    bgr = cv2.imdecode(np.frombuffer(rgb_raw, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.imdecode(np.frombuffer(mask_raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if bgr is None or gray is None:
        raise AdmissionError("case image or mask is not decodable")
    if gray.shape != bgr.shape[:2]:
        raise AdmissionError("case mask dimensions differ from the RGB image")
    return np.ascontiguousarray(bgr[:, :, ::-1]), gray > 127


def run_fullimage_head_parity(cases: Sequence[Mapping[str, Any]], *, contract: WorkbenchContract,
                              head_path: Path, reference_path: Path, output_dir: Path,
                              device: str = "cpu", base_loader: Callable | None = None,
                              runner_factory: Callable | None = None, torch_module=None,
                              tolerance: Mapping[str, Any] = PARITY_TOLERANCE,
                              recorded_artifact: Mapping[str, Any] | None = None) -> dict:
    """Run the reference head and the installed export sequentially on the same
    frozen RGB cases and store every compared array.

    ``cases`` items add decoded ``rgb``/``mask`` arrays to their contract rows;
    the arrays are re-hashed here against the contract before execution.
    Insufficient correspondences in both variants is an unavailable pose with
    numeric parity intact, not a parity failure.
    """
    _contract_recorded_proof(contract, recorded_artifact)
    if dict(tolerance) != PARITY_TOLERANCE:
        raise AdmissionError("parity tolerance must be the pre-declared constant")
    if runtime_packages() != contract.body.get("runtime_packages"):
        raise AdmissionError("runtime packages differ from the authenticated contract")
    if torch_module is None:
        import torch as torch_module
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    by_id = {row["case_id"]: row for row in contract.body["cases"]}
    families = {by_id[c["case_id"]]["family"] for c in cases if c["case_id"] in by_id}
    if set(contract.body["families"]) - families:
        raise AdmissionError("parity must include one frozen case from every intended family")
    prepared = []
    for case in cases:
        row = by_id.get(case["case_id"])
        if row is None:
            raise AdmissionError("parity case is not in the authenticated contract")
        rgb_raw = _regular(Path(case["rgb_path"]), "case RGB").read_bytes()
        mask_raw = _regular(Path(case["mask_path"]), "case mask").read_bytes()
        if hashlib.sha256(rgb_raw).hexdigest() != row["rgb_sha256"] or hashlib.sha256(mask_raw).hexdigest() != row["mask_sha256"]:
            raise AdmissionError(f"{case['case_id']}: input bytes changed after contract")
        # Execute only arrays decoded from the exact authenticated bytes; caller
        # arrays (if any) must be identical, so constant/wrong-frame arrays cannot pass.
        rgb, mask = decode_case_bytes(rgb_raw, mask_raw)
        for name, supplied, decoded in (("rgb", case.get("rgb"), rgb), ("mask", case.get("mask"), mask)):
            if supplied is not None:
                supplied = np.asarray(supplied)
                if supplied.shape != decoded.shape or not np.array_equal(supplied.astype(decoded.dtype), decoded):
                    raise AdmissionError(f"{case['case_id']}: supplied {name} array differs from the decoded file")
        # The executed GLB must be the authenticated one: same resolved path (when
        # recorded) and the same bytes, checked before any network is loaded.
        glb_path = Path(case.get("glb_path") or row["glb"].get("path", ""))
        recorded = row["glb"].get("path")
        if recorded is not None and glb_path.resolve() != Path(recorded):
            raise AdmissionError(f"{case['case_id']}: executed GLB path differs from the contract")
        if glb_inventory(glb_path)["sha256"] != row["glb"]["sha256"]:
            raise AdmissionError(f"{case['case_id']}: executed GLB bytes differ from the contract")
        prepared.append({**row, "rgb": rgb, "mask": mask,
                         "glb_path": str(glb_path), "glb_sha256": row["glb"]["sha256"],
                         "rgb_array_sha256": array_digest(rgb), "mask_array_sha256": array_digest(mask),
                         "rng_seed": int(case.get("rng_seed", 0))})
    base_loader = base_loader or load_base_network
    runner_factory = runner_factory or (lambda net, dev: WorkbenchIntegerRefiner(net, dev))
    events = output_dir / "phase-events.jsonl"
    _phase_event(events, "parity_start", cases=[c["case_id"] for c in prepared], device=device)
    variants = {}
    for variant in ("reference", "export"):
        # Each variant runs in its own function frame; when it returns, no local
        # (network, runner, counter, bound methods) can keep the previous
        # network alive while the next base checkpoint is loaded.
        variants[variant] = _parity_variant(variant, prepared=prepared, output_dir=output_dir, device=device,
                                            base_loader=base_loader, runner_factory=runner_factory,
                                            torch_module=torch_module, head_path=head_path,
                                            reference_path=reference_path, events=events, recorded_artifact=recorded_artifact)
        try:
            release = _release_device_memory(torch_module)
        except BaseException as exc:
            _phase_event(events, f"{variant}_release_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        _phase_event(events, f"{variant}_released", **release)
    comparisons = {}
    for case in prepared:
        cid = case["case_id"]
        reference = _load_trace(output_dir, variants["reference"]["files"][cid])
        export = _load_trace(output_dir, variants["export"]["files"][cid])
        cmp = compare_traces(reference, export, tolerance)
        del reference, export
        executed = all(variants[v]["forward_counts"][cid][m] >= 1 for v in variants for m in ForwardCounter.MODULES)
        comparisons[cid] = {"passed": cmp["passed"] and executed, "full_network_executed": executed,
                            "fields": cmp["fields"]}
    _phase_event(events, "compared", all_passed=all(c["passed"] for c in comparisons.values()))
    receipt = {
        "schema": PARITY_SCHEMA, "profile": PROFILE, "contract_sha256": contract.sha256,
        "renderer_factory": _render_factory(),
        "tolerance": dict(PARITY_TOLERANCE), "device": device, "runtime_packages": runtime_packages(),
        "source_closure_sha256": contract.body["source_closure"]["closure_sha256"],
        "inputs": {c["case_id"]: {"rgb_sha256": c["rgb_sha256"], "mask_sha256": c["mask_sha256"],
                                  "rgb_array_sha256": c["rgb_array_sha256"], "mask_array_sha256": c["mask_array_sha256"],
                                  "glb_sha256": c["glb_sha256"], "rng_seed": c["rng_seed"]} for c in prepared},
        "variants": {v: {"provenance": d["provenance"], "settings": d["settings"],
                         "forward_counts": d["forward_counts"], "render_proofs": d["render_proofs"],
                         "files": d["files"]}
                     for v, d in variants.items()},
        "comparisons": comparisons,
        "all_passed": all(c["passed"] for c in comparisons.values()),
        "scope": "experimental execution compatibility; not model quality",
    }
    if recorded_artifact is not None:
        receipt.update(artifact_lineage=_recorded_lineage(recorded_artifact),
                       execution_scope=RECORDED_EXECUTION_SCOPE, live_eligible=False)
    receipt["receipt_sha256"] = wc.canonical_sha256(receipt)
    (output_dir / "parity-receipt.json").write_bytes(wc.canonical_json(receipt))
    _phase_event(events, "receipt_written", receipt_sha256=receipt["receipt_sha256"])
    return receipt


def _phase_event(path: Path, phase: str, **detail) -> None:
    """Durable per-phase resource event (fsynced JSON line) so a native crash
    still identifies the last completed phase. Sampling never masks failures."""
    try:
        from . import workbench_telemetry as wt
        row = wt.resource_sample(phase, disk_path=Path(path).parent, extra=detail)
    except Exception as exc:  # noqa: BLE001
        row = {"label": phase, "time": time.time(), "sample_error": f"{type(exc).__name__}: {exc}"[:300], **detail}
    try:
        with Path(path).open("ab") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        pass


def _release_device_memory(torch_module) -> dict:
    """Collect garbage, then synchronize and release cached CUDA blocks when CUDA
    is in use. CUDA errors (availability/initialization queries, synchronize,
    empty_cache) propagate: a release is never reported after a CUDA fault."""
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available() or not cuda.is_initialized():
        return {"cuda_in_use": False, "released": False}
    cuda.synchronize()
    cuda.empty_cache()
    return {"cuda_in_use": True, "released": True}


def _load_trace(output_dir: Path, ref: Mapping[str, str]) -> dict:
    path = Path(output_dir) / ref["file"]
    if sha256_file(path) != ref["sha256"]:
        raise AdmissionError(f"stored parity trace {ref['file']} changed before comparison")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _parity_variant(variant: str, *, prepared, output_dir: Path, device: str, base_loader: Callable,
                    runner_factory: Callable, torch_module, head_path: Path, reference_path: Path,
                    events: Path, recorded_artifact: Mapping[str, Any] | None = None) -> dict:
    """Load, install, trace and fully release one parity variant.

    Returns only plain data (provenance, settings, forward counts, render
    proofs, saved-trace file references). Every reference to the network,
    runner, counter hooks and renderers is dropped in ``finally`` (also on
    failure), so nothing survives into the next variant's checkpoint load.
    """
    network = runner = counter = None
    completed = False
    files, counts, proofs = {}, {}, {}
    try:
        settings = deterministic_torch(torch_module)
        _phase_event(events, f"{variant}_load_start")
        network = base_loader(device)
        _phase_event(events, f"{variant}_base_loaded")
        if variant == "reference":
            network, provenance = load_reference_head(network, reference_path, device=device, torch_module=torch_module)
        else:
            kwargs = {"recorded_artifact": recorded_artifact} if recorded_artifact is not None else {}
            network, provenance = install_primary_head(network, head_path, device=device, torch_module=torch_module, **kwargs)
        _phase_event(events, f"{variant}_head_installed")
        counter = ForwardCounter(network)
        runner = runner_factory(network, device)
        for case in prepared:
            cid = case["case_id"]
            _phase_event(events, f"{variant}_case_start", case_id=cid)
            before = dict(counter.counts)
            trace = dict(runner.trace(case))
            trace.update({f"decoder_{k}": v for k, v in counter.take().items()})
            counts[cid] = {m: counter.counts[m] - before[m] for m in counter.MODULES}
            path = Path(output_dir) / f"{variant}-{cid}.npz"
            np.savez(path, **{k: np.asarray(v) for k, v in trace.items()})
            del trace
            files[cid] = {"file": path.name, "sha256": sha256_file(path)}
            proof = runner.render_proof(case) if hasattr(runner, "render_proof") else None
            proofs[cid] = copy.deepcopy(proof) if proof is not None else None
            _phase_event(events, f"{variant}_case_saved", case_id=cid)
        result = {"provenance": provenance, "settings": settings, "files": files, "forward_counts": counts,
                  "render_proofs": proofs}
        completed = True
        return result
    finally:
        _phase_event(events, f"{variant}_cleanup_start")
        errors = []
        if counter is not None:
            try:
                counter.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if runner is not None and hasattr(runner, "close"):
            try:
                runner.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        network = runner = counter = None
        gc.collect()
        _phase_event(events, f"{variant}_cleanup_done", cleanup_errors=[f"{type(e).__name__}: {e}"[:200] for e in errors])
        if errors and completed:
            # Never report a variant as complete when its GPU/GL resources were not released.
            raise AdmissionError(f"{variant} parity cleanup failed: {errors[0]}") from errors[0]


# --------------------------------------------------------------------------
# 4. admission verification (recomputes, never trusts booleans)

@dataclass(frozen=True)
class ServingAdmission:
    admitted: bool
    reasons: tuple[str, ...]
    body: dict

    def to_dict(self) -> dict:
        return {**self.body, "admitted": self.admitted, "reasons": list(self.reasons)}


def parity_proof_reasons(receipt: Mapping[str, Any], contract_ids: set[str]) -> list[str]:
    """Render proofs must cover exactly the contract case IDs for both variants."""
    from . import workbench_render as wr
    reasons: list[str] = []
    variants_seen = receipt.get("variants") or {}
    if set(variants_seen) != {"reference", "export"}:
        reasons.append("parity receipt must contain exactly the reference and export variants")
    for variant in ("reference", "export"):
        info = variants_seen.get(variant)
        if not isinstance(info, Mapping):
            reasons.append(f"parity receipt lacks the {variant} variant")
            continue
        proofs = info.get("render_proofs") or {}
        if not isinstance(proofs, Mapping) or set(proofs) != set(contract_ids):
            have = set(proofs) if isinstance(proofs, Mapping) else set()
            reasons.append(f"{variant}: render proofs must cover exactly the contract cases "
                           f"(missing {sorted(set(contract_ids) - have)}, unexpected {sorted(have - set(contract_ids))})")
            continue
        for cid, proof in proofs.items():
            if (not isinstance(proof, Mapping) or proof.get("policy") != wr.POLICY
                    or not proof.get("framebuffer_proof")):
                reasons.append(f"{cid}/{variant}: parity templates lack a verified {wr.POLICY} render proof")
    return reasons


def _recompute_depth_evidence(evidence: Any, evidence_dir: Path | None) -> dict | None:
    """Reload stored float64 depth (hash-checked) and recompute the plane gate."""
    if not isinstance(evidence, Mapping) or evidence_dir is None:
        return None
    try:
        name = str(evidence["file"])
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            return None
        path = Path(evidence_dir) / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != evidence["sha256"]:
            return None
        depth = np.load(path, allow_pickle=False)
        if depth.dtype != np.float64:
            return None
        return plane_depth_metrics(depth, np.asarray(evidence["K"], float), np.asarray(evidence["pose_mm"], float),
                                   int(evidence["width"]), int(evidence["height"]))
    except (KeyError, TypeError, ValueError, OSError):
        return None


def verify_serving_admission(parity_dir: Path, *, contract: WorkbenchContract, geometry: Mapping[str, Any],
                             root: Path = ROOT, include_upstream: bool = True,
                             geometry_evidence_dir: Path | None = None,
                             recorded_artifact: Mapping[str, Any] | None = None,
                             execution_context: str = "recorded") -> ServingAdmission:
    reasons: list[str] = []
    if execution_context not in ("recorded", "live"):
        raise AdmissionError("unknown execution context")
    try:
        fresh_contract = (contract.body.get("recorded_artifact_proof") is not None
                          or contract.body.get("execution_scope") == RECORDED_EXECUTION_SCOPE)
        if (fresh_contract or recorded_artifact is not None) and execution_context != "recorded":
            raise AdmissionError("fresh artifact proof is finite recorded-only; LIVE use is forbidden")
        _contract_recorded_proof(contract, recorded_artifact, root=root)
        if recorded_artifact is not None and not include_upstream:
            raise AdmissionError("fresh artifact proof requires complete source verification")
    except (AdmissionError, NeuralUnavailable, OSError, ValueError) as exc:
        return ServingAdmission(False, (str(exc),), {"schema": ADMISSION_SCHEMA})
    parity_dir = Path(parity_dir)
    receipt_path = parity_dir / "parity-receipt.json"
    try:
        receipt = wc.loads_strict(receipt_path.read_bytes())
    except (OSError, ValueError) as exc:
        return ServingAdmission(False, (f"parity receipt unreadable: {exc}",), {"schema": ADMISSION_SCHEMA})
    stored_sha = receipt.pop("receipt_sha256", None)
    if wc.canonical_sha256(receipt) != stored_sha:
        reasons.append("parity receipt self-hash mismatch")
    if receipt.get("schema") != PARITY_SCHEMA or receipt.get("profile") != PROFILE:
        reasons.append("parity receipt schema/profile differs")
    if recorded_artifact is not None:
        if (receipt.get("artifact_lineage") != _recorded_lineage(recorded_artifact)
                or receipt.get("execution_scope") != RECORDED_EXECUTION_SCOPE or receipt.get("live_eligible") is not False):
            reasons.append("parity receipt recorded-only lineage/scope differs")
    elif receipt.get("artifact_lineage") is not None or receipt.get("execution_scope") == RECORDED_EXECUTION_SCOPE:
        reasons.append("fresh recorded-only parity cannot admit a historical contract")
    if receipt.get("contract_sha256") != contract.sha256:
        reasons.append("parity receipt is bound to another contract")
    if receipt.get("tolerance") != PARITY_TOLERANCE:
        reasons.append("parity tolerance differs from the pre-declared constant")
    current = source_closure(root, include_upstream=include_upstream)
    if current["closure_sha256"] != contract.body["source_closure"]["closure_sha256"]:
        reasons.append("current source closure differs from the contract")
    if current["unreviewed_replacements"]:
        for rel, row in sorted(current["unreviewed_replacements"].items()):
            reasons.append(f"unreviewed historical replacement {rel}: {row['reason']} "
                           f"(training {row['training_sha256'][:12]}, current {row['current_sha256'][:12]})")
    if current["reviewed_replacements"] and not current["review_evidence_present"]:
        reasons.append("reviewed replacements require the pinned review evidence file")
    if current["historical_missing"]:
        reasons.append(f"{len(current['historical_missing'])} historical source files are missing")
    if current["review_map_sha256"] != contract.body["source_closure"].get("review_map_sha256"):
        reasons.append("review map differs from the contract")
    packages_now = runtime_packages()
    if packages_now != contract.body.get("runtime_packages"):
        changed = sorted(k for k in set(packages_now) | set(contract.body.get("runtime_packages", {}))
                         if packages_now.get(k) != contract.body.get("runtime_packages", {}).get(k))
        reasons.append(f"runtime packages differ from the contract: {changed}")
    if receipt.get("runtime_packages") != contract.body.get("runtime_packages"):
        reasons.append("parity ran under runtime packages that differ from the contract")
    try:
        verify_primary_record(root, recorded_artifact=recorded_artifact)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"primary record: {exc}")
    for case in contract.body["cases"]:
        cid = case["case_id"]
        inp = receipt.get("inputs", {}).get(cid)
        if inp is None:
            reasons.append(f"{cid}: no parity evidence")
            continue
        if inp.get("rgb_sha256") != case["rgb_sha256"] or inp.get("mask_sha256") != case["mask_sha256"]:
            reasons.append(f"{cid}: parity inputs differ from the contract")
        try:
            rgb_raw = _regular(Path(case["rgb_path"]), "case RGB").read_bytes()
            mask_raw = _regular(Path(case["mask_path"]), "case mask").read_bytes()
            if (hashlib.sha256(rgb_raw).hexdigest() != case["rgb_sha256"]
                    or hashlib.sha256(mask_raw).hexdigest() != case["mask_sha256"]):
                reasons.append(f"{cid}: case files changed after the contract")
            else:
                rgb, mask = decode_case_bytes(rgb_raw, mask_raw)
                if (array_digest(rgb) != inp.get("rgb_array_sha256")
                        or array_digest(mask) != inp.get("mask_array_sha256")):
                    reasons.append(f"{cid}: executed arrays do not match the decoded case files")
        except (KeyError, NeuralUnavailable, AdmissionError, OSError) as exc:
            reasons.append(f"{cid}: case files cannot be re-verified ({exc})")
        try:
            glb_path = case["glb"].get("path")
            if not glb_path or glb_inventory(Path(glb_path))["sha256"] != case["glb"]["sha256"]:
                reasons.append(f"{cid}: executed GLB is missing or changed since the contract")
            elif inp.get("glb_sha256") != case["glb"]["sha256"]:
                reasons.append(f"{cid}: parity executed a different GLB")
        except (NeuralUnavailable, AdmissionError, OSError) as exc:
            reasons.append(f"{cid}: GLB cannot be re-verified ({exc})")
        traces = {}
        for variant, prov_key in (("reference", "reference_checkpoint_sha256"), ("export", None)):
            info = receipt.get("variants", {}).get(variant, {})
            prov = info.get("provenance", {})
            if variant == "reference" and prov.get(prov_key) != PRIMARY["reference_checkpoint_sha256"]:
                reasons.append("reference variant provenance differs")
            if variant == "export" and prov.get("head", {}).get("installed_sha256") != PRIMARY["export_sha256"]:
                reasons.append("export variant was not installed from the pinned export")
            if variant == "export" and recorded_artifact is not None:
                if (prov.get("artifact_lineage") != _recorded_lineage(recorded_artifact)
                        or prov.get("execution_scope") != RECORDED_EXECUTION_SCOPE or prov.get("live_eligible") is not False):
                    reasons.append("export installation recorded-only lineage/scope differs")
            counts = info.get("forward_counts", {}).get(cid, {})
            if any(int(counts.get(m, 0)) < 1 for m in ForwardCounter.MODULES):
                reasons.append(f"{cid}/{variant}: observed full-network execution missing")
            ref = info.get("files", {}).get(cid, {})
            path = parity_dir / str(ref.get("file", ""))
            if not path.is_file() or sha256_file(path) != ref.get("sha256"):
                reasons.append(f"{cid}/{variant}: stored trace missing or altered")
                continue
            with np.load(path, allow_pickle=False) as data:
                traces[variant] = {k: data[k] for k in data.files}
        if len(traces) == 2:
            recomputed = compare_traces(traces["reference"], traces["export"])
            if not recomputed["passed"]:
                failed = [k for k, v in recomputed["fields"].items() if not v["passed"]]
                reasons.append(f"{cid}: recomputed parity failed for {failed[:6]}")
            for variant, trace in traces.items():
                if not np.isfinite(trace.get("flow", np.array([np.nan]))).all():
                    reasons.append(f"{cid}/{variant}: non-finite flow")
    geometry_body = dict(geometry)
    geometry_sha = geometry_body.pop("receipt_sha256", None)
    from . import workbench_render as wr
    factory_now = _render_factory()
    if not factory_now.get("pins_match"):
        reasons.append("zero-storage render sources differ from their pins")
    for label, value in (("contract", contract.body.get("renderer_factory")),
                         ("geometry", geometry_body.get("renderer_factory")),
                         ("parity", receipt.get("renderer_factory"))):
        if value != factory_now:
            reasons.append(f"renderer factory identity in {label} differs from the current {wr.POLICY} adapter")
    for row_name in ("renderer", "template_renderer"):
        row = (geometry_body.get("measured") or {}).get(row_name) or {}
        if row.get("render_policy") != wr.POLICY or not (row.get("allocation") or {}).get("proof"):
            reasons.append(f"geometry {row_name} measurement lacks a verified {wr.POLICY} allocation proof")
    reasons.extend(parity_proof_reasons(receipt, {row["case_id"] for row in contract.body["cases"]}))
    if wc.canonical_sha256(geometry_body) != geometry_sha:
        reasons.append("geometry receipt self-hash mismatch")
    if geometry_body.get("schema") != GEOMETRY_SCHEMA:
        reasons.append("geometry receipt schema differs")
    measured = geometry_body.get("measured", {})
    sampler = measured.get("crop_sampler", {})
    # Both the native and the template depth gates are enforced independently and
    # recomputed from stored evidence; recorded booleans/numbers are not trusted.
    for row_name in ("renderer", "template_renderer"):
        row = measured.get(row_name) or {}
        if not depth_row_passes(row):
            reasons.append(f"geometry {row_name} depth/IoU outside the fixed gates "
                           f"(rel < {DEPTH_MAX_RELATIVE_ERROR}, IoU > {COVERAGE_MIN_IOU})")
        recomputed = _recompute_depth_evidence(row.get("evidence"), geometry_evidence_dir)
        if recomputed is None:
            reasons.append(f"geometry {row_name} depth evidence is missing or altered; cannot recompute")
        else:
            if not depth_row_passes(recomputed):
                reasons.append(f"geometry {row_name} recomputed depth/IoU fails the fixed gates")
            if (recomputed["depth_max_relative_error"] != row.get("depth_max_relative_error")
                    or recomputed["coverage_iou"] != row.get("coverage_iou")):
                reasons.append(f"geometry {row_name} recorded metrics differ from recomputation")
    if geometry_body.get("status") != "measured" or geometry_body.get("admission_eligible") is not True:
        reasons.append("geometry receipt is not a measured, admission-eligible validation")
    if not (isinstance(sampler.get("max_error_px"), (int, float)) and sampler["max_error_px"] < SAMPLING_MAX_ERROR_PX):
        reasons.append("measured production crop-sampling evidence is missing or exceeds 0.001 px")
    if (sampler.get("method") != CROP_MEASUREMENT_METHOD or sampler.get("failures")
            or not (sampler.get("rgb_quantization_check") or {}).get("passed")
            or not (sampler.get("negative_control_wrong_half_pixel") or {}).get("detected")):
        reasons.append("crop evidence is not an observed-upstream-remap measurement with RGB check and negative control")
    analytic = geometry_body.get("checks", {})
    for name in ("crop_to_native_projection", "noiseless_pnp_invariance", "control_wrong_half_pixel",
                 "control_1000_unit_factor", "opengl_half_pixel_boundary", "unprojection_roundtrip"):
        if not analytic.get(name, {}).get("passed"):
            reasons.append(f"geometry check {name} not satisfied")
    # Re-run the analytic part now; a stale or edited receipt cannot admit.
    fresh = validate_metric_geometry()
    if not fresh["analytic_passed"]:
        reasons.append("fresh analytic geometry validation failed")
    body = {"schema": ADMISSION_SCHEMA, "profile": PROFILE, "contract_sha256": contract.sha256,
            "parity_receipt_sha256": stored_sha, "geometry_receipt_sha256": geometry_sha,
            "scope": "experimental execution compatibility only; never model quality",
            "families": contract.body["families"], "quality_claim": False,
            "relative_reconstructions_admitted": False,
            "reviewed_replacements": current["reviewed_replacements"],
            "unreviewed_replacements": current["unreviewed_replacements"],
            "review_map_sha256": current["review_map_sha256"]}
    if recorded_artifact is not None:
        body.update(artifact_lineage=_recorded_lineage(recorded_artifact), execution_scope=RECORDED_EXECUTION_SCOPE,
                    live_eligible=False)
    return ServingAdmission(not reasons, tuple(dict.fromkeys(reasons)), body)


# --------------------------------------------------------------------------
# session lifecycle

class NeuralBackend:
    """Protocol for an admitted full-image backend (concrete per family)."""

    def install_provenance(self) -> dict:  # pragma: no cover - protocol
        raise NotImplementedError

    def estimate(self, rgb: np.ndarray, observed_mask: np.ndarray, K: np.ndarray, state: dict) -> dict:
        """Return {pose_m: 4x4|None, status: tracking|recovering|lost, reason: str|None, metrics: dict}."""
        raise NotImplementedError  # pragma: no cover

    def reset(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


class NeuralSession:
    """Validated lifecycle around a real admitted backend.

    Every ``process`` call checks the frame/image/mask identities, monotonic
    source index and that the install provenance is unchanged since init.
    Without a verified admission covering the clip family, construction fails;
    callers then emit blocked rows for every scheduled frame.
    """

    def __init__(self, role: str, backend: NeuralBackend, admission: ServingAdmission, *, run_id: str,
                 clip: wc.ClipSpec, family: str, renderer: Callable | None = None):
        if role not in ("primary_neural", "frozen_baseline"):
            raise ValueError("neural session role must be primary_neural or frozen_baseline")
        if not isinstance(admission, ServingAdmission) or not admission.admitted:
            raise NeuralUnavailable("no verified serving admission")
        if family not in admission.body.get("families", []):
            raise NeuralUnavailable(f"serving admission does not cover clip family {family}")
        self.role, self.backend, self.admission = role, backend, admission
        self.run_id, self.clip = run_id, clip
        self._provenance = dict(backend.install_provenance())
        self._digest = wc.canonical_sha256(self._provenance)
        if role == "primary_neural" and self._provenance.get("head", {}).get("installed_sha256") != PRIMARY["export_sha256"]:
            raise NeuralUnavailable("primary session must run the installed flow-terminal export")
        if role == "primary_neural" and RECORDED_EXECUTION_SCOPE in (
                admission.body.get("execution_scope"), self._provenance.get("execution_scope")):
            if (self._provenance.get("artifact_lineage") != admission.body.get("artifact_lineage")
                    or self._provenance.get("execution_scope") != admission.body.get("execution_scope")):
                raise NeuralUnavailable("primary installed proof differs from recorded-only admission")
        self._closed = False
        self._last_index = None
        self._state: dict = {}
        # renderer(pose_m, observed_mask, K) -> {"rgba", "stats"}: the frozen,
        # geometry-bound artwork shared by both neural roles (NeuralArtwork.render).
        self.renderer = renderer
        self.last_overlay: np.ndarray | None = None

    def identity(self) -> dict:
        """Identity from the validated installed provenance, not requested constants."""
        base = primary_identity() if self.role == "primary_neural" else baseline_identity()
        head = self._provenance.get("head") if isinstance(self._provenance.get("head"), Mapping) else {}
        installed = head.get("installed_sha256") if self.role == "primary_neural" else None
        extras_fn = getattr(self.backend, "identity_extras", None)
        extras = dict(extras_fn(self.clip.intrinsics.K)) if callable(extras_fn) else {}
        if self.admission.body.get("execution_scope") == RECORDED_EXECUTION_SCOPE:
            extras.update(artifact_lineage=self.admission.body["artifact_lineage"],
                          execution_scope=RECORDED_EXECUTION_SCOPE, live_eligible=False)
        return {**base, **extras, "installed_head_sha256": installed,
                "installed_head_state_sha256": self._provenance.get("installed_head_state_sha256"),
                "install_digest": self._digest, "admission_contract_sha256": self.admission.body["contract_sha256"]}

    def reset(self) -> None:
        self._require_open()
        self.backend.reset()
        self._state = {}
        self._last_index = None

    def close(self) -> None:
        if not self._closed:
            self.backend.close()
        self._closed = True

    def _require_open(self):
        if self._closed:
            raise RuntimeError("neural session is closed")

    def process(self, frame: wc.FrameSpec, rgb: np.ndarray, observed_mask: np.ndarray | None,
                identity: Mapping[str, Any], K: np.ndarray) -> wc.FrameResult:
        from . import workbench_geometry as wg
        self._require_open()
        self.last_overlay = None
        if wc.canonical_sha256(dict(self.backend.install_provenance())) != self._digest:
            raise AdmissionError("backend install provenance changed during the session")
        ident = self.identity()

        def row(mask_state, pose_state, reason, pose=None, metrics=None, image_sha=None, mask_sha=None,
                rendered=False, evidence=None):
            # "rendered" only when an actual artwork render produced pixels for this pose.
            render = "rendered" if (pose is not None and rendered) else (
                "suppressed" if mask_state == "observed" and pose is None else "unavailable")
            return wc.FrameResult(self.run_id, self.clip.clip_id, frame.frame_id, frame.source_frame_index,
                                  frame.timestamp_s, frame.media_time_s, self.role, ident, mask_state, pose_state,
                                  render, reason, pose, "metric" if pose is not None else None, image_sha,
                                  frame.frame_id if mask_sha else None, mask_sha, dict(metrics or {}),
                                  dict(evidence or {}))
        if identity.get("frame_id") != frame.frame_id:
            return row("rejected", "unavailable", "frame_identity_mismatch")
        if self._last_index is not None and frame.source_frame_index <= self._last_index:
            return row("rejected", "unavailable", "stale_image: source_frame_index did not advance")
        image_sha = wg.array_sha256(np.asarray(rgb))
        if identity.get("image_sha256") != image_sha:
            return row("rejected", "unavailable", "image_identity_mismatch")
        self._last_index = frame.source_frame_index
        if observed_mask is None:
            self._state = {}
            return row("unavailable", "unavailable", "observed_mask_unavailable", image_sha=image_sha)
        msha = wg.mask_sha256(observed_mask)
        if np.asarray(observed_mask).shape != np.asarray(rgb).shape[:2]:
            self._state = {}
            return row("rejected", "unavailable", "mask_dimension_mismatch", image_sha=image_sha)
        if identity.get("mask_frame_id") != frame.frame_id or identity.get("mask_sha256") != msha:
            self._state = {}
            return row("rejected", "unavailable", "stale_mask: mask identity does not match this frame", image_sha=image_sha)
        if not (np.asarray(observed_mask) > 0).any():
            self._state = {}
            return row("observed", "lost", "observed_mask_empty", image_sha=image_sha, mask_sha=msha)
        self._state["_frame_id"] = frame.frame_id
        out = self.backend.estimate(np.asarray(rgb), np.asarray(observed_mask) > 0, np.asarray(K, float), self._state)
        status, pose = out.get("status"), out.get("pose_m")
        metrics = {k: (None if v is None else float(v)) for k, v in dict(out.get("metrics", {})).items()}
        extra = dict(out.get("evidence") or {})
        if status == "tracking" and pose is not None:
            wc._pose(np.asarray(pose).tolist(), "neural pose")
            stats = out.get("render_stats") if isinstance(out.get("render_stats"), Mapping) else {}
            rendered = int(stats.get("rendered_pixels", 0) or 0) > 0
            if self.renderer is not None:
                # Render only for this validated current pose with this frame's observed mask.
                try:
                    drawn = self.renderer(np.asarray(pose, float), np.asarray(observed_mask) > 0, np.asarray(K, float))
                    stats = dict(drawn["stats"])
                    rendered = int(stats.get("rendered_pixels", 0)) > 0
                    self.last_overlay = drawn["rgba"] if rendered else None
                except Exception as exc:  # noqa: BLE001 - never mark rendered without pixels
                    stats, rendered = {"reason": f"render_failed: {type(exc).__name__}: {exc}"}, False
            elif not stats:
                extra["render"] = "no artwork renderer bound to this session"
            if stats:
                stats = dict(stats)
                if stats.get("reason") is None and "rendered_pixels" in stats and stats["rendered_pixels"] == 0:
                    stats["reason"] = "no visible supported artwork pixels"
                extra["render_stats"] = stats
            return row("observed", "tracking", None, [list(map(float, r)) for r in np.asarray(pose)], metrics,
                       image_sha, msha, rendered=rendered, evidence=extra)
        if status not in ("recovering", "lost"):
            status = "lost"
        return row("observed", status, out.get("reason") or f"{status}: no validated pose", None, metrics,
                   image_sha, msha, evidence=extra)


def blocked_results(run_id: str, clip: wc.ClipSpec, scheduled: Sequence[str], role: str, reason: str) -> list[wc.FrameResult]:
    """Every scheduled frame stays counted and unavailable; no pose is invented."""
    identity = primary_identity() if role == "primary_neural" else baseline_identity()
    position = {f.frame_id: f for f in clip.frames}
    return [wc.blocked_result(run_id, clip, position[fid], role, identity, reason) for fid in scheduled]


def neural_stage_status(preflight: Mapping[str, Any]) -> str:
    missing = ", ".join(preflight.get("blocked_reasons", [])) or "serving_admission"
    return f"neural_blocked: {missing}"


# --------------------------------------------------------------------------
# persisted gate artifacts

def save_contract(contract: WorkbenchContract, path: Path) -> None:
    with Path(path).open("xb") as stream:
        stream.write(wc.canonical_json(contract.to_dict()))


def load_contract(path: Path) -> WorkbenchContract:
    body = wc.loads_strict(Path(path).read_bytes())
    stored = body.pop("contract_sha256", None)
    if body.get("schema") != CONTRACT_SCHEMA or wc.canonical_sha256(body) != stored:
        raise AdmissionError("contract file schema or self-hash differs")
    return WorkbenchContract(body, stored)


def load_admission(directory: Path, *, root: Path = ROOT, include_upstream: bool = True,
                   recorded_artifact: Mapping[str, Any] | None = None,
                   execution_context: str = "recorded") -> ServingAdmission:
    """Re-verify an admission directory {contract.json, geometry.json, geometry-evidence/, parity/};
    never trusts stored verdicts."""
    directory = Path(directory)
    contract = load_contract(directory / "contract.json")
    geometry = wc.loads_strict((directory / "geometry.json").read_bytes())
    return verify_serving_admission(directory / "parity", contract=contract, geometry=geometry, root=root,
                                    include_upstream=include_upstream,
                                    geometry_evidence_dir=directory / "geometry-evidence",
                                    recorded_artifact=recorded_artifact, execution_context=execution_context)


# --------------------------------------------------------------------------
# production measurements (actual renderer + crop pipeline; parent runs them)

def write_plane_glb(path: Path, half_m: float = 0.15) -> str:
    """Minimal double-sided metric plane GLB (z = 0, metres) for renderer probes."""
    positions = np.array([[-half_m, -half_m, 0], [half_m, -half_m, 0], [half_m, half_m, 0], [-half_m, half_m, 0]],
                         np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], np.uint16)
    pos_bytes, idx_bytes = positions.tobytes(), indices.tobytes()
    pad = (4 - len(idx_bytes) % 4) % 4
    binary = pos_bytes + idx_bytes + b"\x00" * pad
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
            "materials": [{"doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.6, 0.3, 1.0]}}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
                            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(idx_bytes), "target": 34963}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                           "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
                          {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    raw = struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def production_renderer_probe(scratch_dir: Path) -> Callable:
    """Probe for ``validate_metric_geometry`` using the zero-storage adapter around
    the actual TexturedRenderer (legacy_v1, integer_centers_v1, unlit)."""
    from . import quality_gotrack as qg
    from . import workbench_render as wr
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    glb = scratch_dir / "probe-plane.glb"
    glb_sha = write_plane_glb(glb)

    def probe(K, pose_mm, width, height):
        qg.upstream_path()
        from utils import structs  # type: ignore
        renderer = wr.WorkbenchZeroStorageRenderer(str(glb), 1)
        try:
            camera = structs.PinholePlaneCameraModel(width=int(width), height=int(height), f=(K[0, 0], K[1, 1]),
                                                     c=(K[0, 2], K[1, 2]),
                                                     T_world_from_eye=np.linalg.inv(np.asarray(pose_mm, float)))
            out = renderer.render_object_model(1, camera)
            depth = None
            for key, value in out.items():
                if getattr(key, "name", str(key)).upper().endswith("DEPTH"):
                    depth = np.asarray(value, dtype=np.float64)
            if depth is None:
                raise AdmissionError("renderer returned no depth")
            depth = np.where(depth > 0, depth, np.inf)
            proof = renderer.metadata
            if not proof or proof.get("policy") != wr.POLICY:
                raise AdmissionError("zero-storage renderer produced no verified allocation proof")
            allocation = {"viewport": proof["framebuffer_proof"]["viewport"], "render_policy": wr.POLICY,
                          "inner_render_policy": renderer.render_policy, "coordinate_mode": renderer.coordinate_mode,
                          "unlit": renderer.unlit, "disable_multisampling": renderer.disable_multisampling,
                          "asset_sha256": glb_sha, "proof": proof}
            return {"depth_mm": depth, "allocation": allocation, "render_policy": wr.POLICY}
        finally:
            renderer.close()
    return probe


UPSTREAM_CROP_SOURCES = ("utils/im_util.py", "utils/crop_generation.py", "utils/data_util.py")
CROP_MEASUREMENT_METHOD = "observed_upstream_remap_maps_v1"


def upstream_crop_sources(root: Path = ROOT) -> dict:
    """Hash the upstream crop sources and require their historical training pins."""
    historical = historical_source_map(root)
    out = {}
    for rel in UPSTREAM_CROP_SOURCES:
        key = f"{UPSTREAM_RELATIVE}/{rel}"
        actual = sha256_file(root / key)
        if historical.get(key) != actual:
            raise AdmissionError(f"upstream crop source {rel} differs from its historical pin")
        out[key] = actual
    return out


def observed_crop_measurement(captured: Sequence[Mapping[str, Any]], ramp_chw: np.ndarray, crop_chw: np.ndarray,
                              Kn: np.ndarray, Kc: np.ndarray, C: np.ndarray, native_hw: tuple[int, int],
                              sources: Mapping[str, str], remap: Callable) -> dict:
    """Judge the actual upstream crop from its observed ``cv2.remap`` calls.

    Requires exactly one RGB (uint8 HxWx3) and one mask (int16 HxW) remap call
    with identical maps. The observed maps must match the analytic
    native-from-crop map within 0.001 px. The output RGB must equal the
    declared upstream 8-bit quantization ``np.uint8(rgb * 255)`` remapped with
    the same maps and interpolation, divided by 255. A deliberately wrong
    half-pixel analytic map must be detected (> 0.001 px).
    """
    H, W = native_hw
    reasons = []
    rgb_calls = [c for c in captured if c["src"].dtype == np.uint8 and c["src"].shape == (H, W, 3)]
    mask_calls = [c for c in captured if c["src"].dtype == np.int16 and c["src"].shape == (H, W)]
    if len(captured) != 2 or len(rgb_calls) != 1 or len(mask_calls) != 1:
        reasons.append(f"expected exactly one RGB and one mask remap call, observed {len(captured)}")
    map_err = None
    control = None
    rgb_diff = None
    if rgb_calls:
        rc = rgb_calls[0]
        mx, my = crop_sampling_map(Kn, Kc, C[:3, :3], rc["map_x"].shape[1])
        if rc["map_x"].shape != mx.shape or rc["map_y"].shape != my.shape or rc["extra_args"]:
            reasons.append("observed RGB remap maps have unexpected shape or arguments")
        else:
            map_err = float(max(np.abs(rc["map_x"].astype(np.float64) - mx).max(),
                                np.abs(rc["map_y"].astype(np.float64) - my).max()))
            wrong = Kn.copy()
            wrong[0, 2] += 0.5
            wrong[1, 2] += 0.5
            wx, wy = crop_sampling_map(wrong, Kc, C[:3, :3], rc["map_x"].shape[1])
            control_err = float(max(np.abs(rc["map_x"] - wx).max(), np.abs(rc["map_y"] - wy).max()))
            control = {"max_error_px": control_err, "detected": control_err > SAMPLING_MAX_ERROR_PX}
            if mask_calls:
                mc = mask_calls[0]
                if not (np.array_equal(mc["map_x"], rc["map_x"]) and np.array_equal(mc["map_y"], rc["map_y"])
                        and mc["interpolation"] == rc["interpolation"]):
                    reasons.append("mask remap maps differ from the RGB maps")
            quantized = np.uint8(np.transpose(ramp_chw, (1, 2, 0)) * 255)
            if not np.array_equal(rc["src"], quantized):
                reasons.append("upstream RGB source differs from the declared np.uint8(rgb * 255) quantization")
            expected = remap(quantized, rc["map_x"], rc["map_y"], rc["interpolation"])
            if not np.array_equal(expected, rc["out"]):
                reasons.append("observed remap output differs from an independent identical remap")
            expected_crop = (np.transpose(expected, (2, 0, 1)) / 255.0).astype(np.float32)
            if expected_crop.shape != crop_chw.shape:
                reasons.append("returned crop shape differs from the remapped RGB")
            else:
                rgb_diff = float(np.abs(expected_crop - np.asarray(crop_chw, np.float32)).max())
                if rgb_diff != 0.0:
                    reasons.append("returned crop RGB differs from the declared quantization + interpolation")
    if map_err is None or map_err >= SAMPLING_MAX_ERROR_PX:
        reasons.append("observed remap maps differ from the analytic crop mapping by >= 0.001 px")
    if control is None or not control["detected"]:
        reasons.append("wrong half-pixel negative control was not detected")
    return {"method": CROP_MEASUREMENT_METHOD, "max_error_px": map_err, "threshold_px": SAMPLING_MAX_ERROR_PX,
            "observed_remap_calls": [{"dtype": c["src"].dtype.str, "shape": list(c["src"].shape),
                                      "interpolation": c["interpolation"], "map_x_sha256": array_digest(c["map_x"]),
                                      "map_y_sha256": array_digest(c["map_y"])} for c in captured],
            "rgb_quantization_check": {"declared": "np.uint8(rgb * 255) then cv2.remap, / 255",
                                       "max_abs_diff": rgb_diff, "passed": rgb_diff == 0.0},
            "negative_control_wrong_half_pixel": control, "upstream_sources": dict(sources),
            "crop_sha256": array_digest(np.asarray(crop_chw, np.float32)),
            "failures": reasons, "passed": not reasons,
            "note": "maps measured directly; an 8-bit coordinate ramp cannot measure sub-pixel accuracy"}


def measure_production_crop(scratch_dir: Path, evidence_dir: Path | None = None) -> dict:
    """Run the actual upstream perspective crop (unchanged helpers) and observe
    its ``cv2.remap`` maps through a scoped, restored hook inside the global
    GPU stage. Maps are compared with the analytic crop mapping (< 0.001 px);
    the RGB output is checked against the declared 8-bit quantization. The
    template-depth measurement is unchanged."""
    import torch
    from . import quality_gotrack as qg
    qg.upstream_path()
    from utils import data_util, misc, structs  # type: ignore
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    glb = scratch_dir / "crop-plane.glb"
    write_plane_glb(glb)
    W, H = 1024, 1280
    Kn = np.array([[910.0, 0, 498.25], [0, 905.0, 655.75], [0, 0, 1]])
    pose_mm = _rt(0.25, -0.15, 0.05, [10.0, 5.0, 700.0])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ramp = np.stack([xx / W, yy / H, np.zeros_like(xx)], axis=0)[None]
    from . import workbench_telemetry as wt
    from . import workbench_render as wr
    if not wt.gpu_stage_active():
        raise AdmissionError("production crop measurement must run inside the global gpu_stage lock")
    renderer = wr.WorkbenchZeroStorageRenderer(str(glb), 1)
    try:
        camera = structs.PinholePlaneCameraModel(width=W, height=H, f=(Kn[0, 0], Kn[1, 1]), c=(Kn[0, 2], Kn[1, 2]),
                                                 T_world_from_eye=np.eye(4))
        sources = upstream_crop_sources()
        from utils import im_util  # type: ignore
        cv2_module = getattr(im_util, "cv2", None)
        if cv2_module is None or cv2_module is not sys.modules.get("cv2"):
            raise AdmissionError("upstream im_util does not use the shared cv2 module; remap cannot be observed")
        original_remap = cv2_module.remap
        captured: list[dict] = []

        def observed_remap(src, map_x, map_y, interpolation, *args, **kwargs):
            out = original_remap(src, map_x, map_y, interpolation, *args, **kwargs)
            captured.append({"src": np.array(src, copy=True), "map_x": np.array(map_x, copy=True),
                             "map_y": np.array(map_y, copy=True), "interpolation": int(interpolation),
                             "extra_args": len(args) + len(kwargs), "out": np.array(out, copy=True)})
            return out
        cv2_module.remap = observed_remap  # scoped hook; restored below
        try:
            with torch.inference_mode():
                data, cameras, crop_from_orig = data_util.compute_gotrack_inputs_from_init_poses(
                    input_rgbs=torch.from_numpy(ramp), input_cameras=[camera],
                    init_poses_cam_from_model=torch.from_numpy(pose_mm.astype(np.float32))[None],
                    renderer=renderer, obj_ids=[1], object_vertices=[renderer.vertices_m * 1000], crop_size=(280, 280),
                    crop_rel_pad=.1, cropping_type="perspective_2d_box", ssaa_factor=1., background_type="gray",
                    input_masks=torch.ones((1, H, W)))
        finally:
            cv2_module.remap = original_remap
        crop = data["crop_rgbs"][0].cpu().numpy()
        Kc = np.asarray(misc.get_intrinsic_matrix(cameras[0]), dtype=np.float64)
        C = np.asarray(crop_from_orig[0], dtype=np.float64)
        sampler = observed_crop_measurement(captured, ramp[0], crop, Kn, Kc, C, (H, W), sources, original_remap)
        template_depth = data["templates"].depths[0].double().cpu().numpy()
        while template_depth.ndim > 2:
            template_depth = template_depth[0]
        template_pose = C @ pose_mm
        metrics = plane_depth_metrics(template_depth, Kc, template_pose, 280, 280)
        proof = renderer.metadata
        if not proof or proof.get("policy") != wr.POLICY:
            raise AdmissionError("zero-storage template renderer produced no verified allocation proof")
        renderer_row = {**metrics,
                        "allocation": {"viewport": proof["framebuffer_proof"]["viewport"], "render_policy": wr.POLICY,
                                       "inner_render_policy": renderer.render_policy,
                                       "coordinate_mode": renderer.coordinate_mode, "path": "template crop",
                                       "proof": proof},
                        "render_policy": wr.POLICY,
                        "depth_sha256": array_digest(template_depth.astype(np.float32)),
                        "evidence": save_depth_evidence(evidence_dir, "template_plane_depth_mm", template_depth, Kc,
                                                        template_pose, 280, 280),
                        "passed": depth_row_passes(metrics)}
        return {"crop_sampler": sampler, "template_renderer": renderer_row}
    finally:
        renderer.close()


# --------------------------------------------------------------------------
# concrete admitted metric backend (GoTrack new head / frozen base)

SEED_PREFIXES = ("controlled_init:", "onboarding:")


@dataclass(frozen=True)
class MetricBundle:
    family: str
    glb_path: Path
    glb_sha256: str
    obj_id: int
    principal_point_origin: str
    seed_pose_m: np.ndarray | None
    seed_label: str | None
    seed_frame_id: str | None
    metric_provenance: str

    @classmethod
    def from_file(cls, path: Path, *, root: Path = ROOT) -> "MetricBundle":
        d = wc._keys(wc.loads_strict(Path(path).read_bytes()),
                     ("schema", "family", "glb_path", "glb_sha256", "obj_id", "principal_point_origin",
                      "seed_pose_m", "seed_label", "seed_frame_id", "metric_provenance"), (), "metric bundle")
        if d["schema"] != "workbench_metric_bundle_v1":
            raise AdmissionError("metric bundle schema differs")
        glb = Path(d["glb_path"])
        glb = glb if glb.is_absolute() else root / glb
        if glb_inventory(glb)["sha256"] != d["glb_sha256"]:
            raise AdmissionError("metric bundle GLB hash differs")
        if d["principal_point_origin"] not in ("integer_centers", "corner"):
            raise AdmissionError("metric bundle must document the principal point origin")
        if d["metric_provenance"] not in ("measured_metric_cad", "dataset_metric_cad"):
            raise AdmissionError("relative reconstructions are not metric bundles")
        seed = None
        if d["seed_pose_m"] is not None:
            if not str(d["seed_label"] or "").startswith(SEED_PREFIXES):
                raise AdmissionError("seed must be labelled controlled_init:/onboarding:, never evaluator data")
            seed = np.array(wc._pose(d["seed_pose_m"], "seed_pose_m"), dtype=np.float64)
        return cls(wc._str(d["family"], "family", max_len=64), glb, d["glb_sha256"], int(d["obj_id"]),
                   d["principal_point_origin"], seed, d["seed_label"], d["seed_frame_id"], d["metric_provenance"])


def historical_metric_bundle(name: str, *, principal_point_origin: str, root: Path = ROOT) -> dict:
    """Bundle for a historical case: pinned GLB + explicitly permitted controlled
    onboarding pose (labelled ``controlled_init``; never a scored evaluator pose)."""
    base = root / ".cache/model-quality/inputs" / name
    data = wc.loads_strict((base / "input.json").read_bytes())
    glb = base / data["asset"]
    sha = sha256_file(glb)
    if sha != data["source_hashes"][data["asset"]]:
        raise AdmissionError("historical GLB hash differs from declared source_hashes")
    if principal_point_origin not in ("integer_centers", "corner"):
        raise AdmissionError("principal point origin must be declared explicitly by the operator")
    return {"schema": "workbench_metric_bundle_v1", "family": "historical_1024x1280",
            "glb_path": glb.relative_to(root).as_posix(), "glb_sha256": sha, "obj_id": int(data["object_id"]),
            "principal_point_origin": principal_point_origin,
            "seed_pose_m": data["controlled_initial_pose"],
            "seed_label": f"controlled_init:{name}:input.json controlled_initial_pose (final onboarding pose)",
            "seed_frame_id": f"s{int(data['setup_frame_id']):06d}", "metric_provenance": "dataset_metric_cad"}


class GoTrackMetricBackend(NeuralBackend):
    """Full-image GoTrack refinement through ``WorkbenchIntegerRefiner``.

    The first pose comes only from the bundle's labelled single-use seed; later
    frames use the previous validated pose as the initial guess. After any
    loss the seed is not reused: recovery is reported unavailable (automatic
    reacquisition, e.g. a FoundPose bank, is not integrated here). Public
    tracking requires two consecutive consistent solutions.
    """

    def __init__(self, role: str, bundle: MetricBundle, *, head_path: Path | None = None, device: str = "cpu",
                 iterations: int = 5, min_inliers: int = 24, network_loader: Callable | None = None,
                 refiner_factory: Callable | None = None, consistency_deg: float = 10.0,
                 consistency_translation: float = 0.1, recorded_artifact: Mapping[str, Any] | None = None):
        if role == "primary_neural":
            if head_path is None:
                raise NeuralUnavailable("primary backend requires the pinned head.pt path")
            network, provenance = (network_loader or (lambda d: load_primary_network(Path(head_path), d, recorded_artifact=recorded_artifact)))(device)
        elif role == "frozen_baseline":
            network, provenance = (network_loader or load_baseline_network)(device)
        else:
            raise ValueError("role must be primary_neural or frozen_baseline")
        self.role, self.bundle, self.device = role, bundle, device
        self.network, self._provenance = network, dict(provenance)
        self.refiner = (refiner_factory or (lambda n, d: WorkbenchIntegerRefiner(n, d)))(network, device)
        self.iterations, self.min_inliers = iterations, min_inliers
        self.consistency = (consistency_deg, consistency_translation)
        self._seed_used = False
        # Frame on which the labelled seed may be used: the frame immediately
        # after ``bundle.seed_frame_id``. Set by the pipeline; None blocks seeding.
        self.seed_target_frame_id: str | None = None

    def identity_extras(self, clip_K) -> dict:
        """Calibration/initialization facts the evaluator needs (no labels)."""
        extras = {"principal_point_origin": self.bundle.principal_point_origin,
                  "initialization": self.bundle.seed_label, "seed_frame_id": self.bundle.seed_frame_id,
                  "seed_target_frame_id": self.seed_target_frame_id, "glb_sha256": self.bundle.glb_sha256,
                  "metric_units": "metres"}
        if clip_K is not None:
            extras["effective_K"] = native_K(np.asarray(clip_K, float).ravel(), self.bundle.principal_point_origin).ravel().tolist()
        return extras

    def install_provenance(self) -> dict:
        return dict(self._provenance)

    def limits(self) -> dict:
        return {"initialization": self.bundle.seed_label, "seed_single_use": True,
                "recovery": "unavailable_after_loss (no automatic reacquisition integrated)",
                "iterations": self.iterations, "min_inliers": self.min_inliers,
                "public_after_consecutive_consistent": 2}

    def reset(self) -> None:
        self._seed_used = False

    def close(self) -> None:
        close = getattr(self.refiner, "close", None)
        if callable(close):
            close()
        self.network = self.refiner = None
        gc.collect()

    def estimate(self, rgb, observed_mask, K, state) -> dict:
        Kn = native_K(np.asarray(K, float).ravel(), self.bundle.principal_point_origin)
        init, label = state.get("prior_pose"), "prior_validated_pose"
        if init is None:
            frame_id = state.get("_frame_id")
            if self._seed_used or self.bundle.seed_pose_m is None:
                state.clear()
                return {"status": "lost", "pose_m": None,
                        "reason": "recovery_unavailable: no validated prior pose and the labelled seed is single-use",
                        "evidence": {"limits": self.limits()}}
            if self.seed_target_frame_id is None or frame_id != self.seed_target_frame_id:
                state.clear()
                return {"status": "lost", "pose_m": None,
                        "reason": "seed_not_applicable: the onboarding seed applies only to the frame immediately "
                                  f"after {self.bundle.seed_frame_id} (got {frame_id})",
                        "evidence": {"limits": self.limits()}}
            init, label = self.bundle.seed_pose_m, self.bundle.seed_label
            self._seed_used = True
        pose = np.asarray(init, dtype=np.float64)
        inliers = 0
        for iteration in range(self.iterations):
            trace = self.refiner.trace({"rgb": rgb, "mask": observed_mask, "native_K": Kn, "initial_pose_m": pose,
                                        "glb_path": str(self.bundle.glb_path), "glb_sha256": self.bundle.glb_sha256,
                                        "obj_id": self.bundle.obj_id, "rng_seed": iteration})
            inliers = int(np.asarray(trace["ransac_inliers"]).size)
            if int(np.asarray(trace["solver_state"])) != 1 or inliers < self.min_inliers:
                state.clear()
                return {"status": "lost", "pose_m": None,
                        "reason": f"solver_unavailable at iteration {iteration} ({inliers} inliers)",
                        "metrics": {"inliers": inliers, "iterations": iteration + 1},
                        "evidence": {"initialization": label}}
            pose = np.asarray(trace["solved_native_pose_m"], dtype=np.float64)
        previous = state.get("prior_pose")
        consecutive = int(state.get("consecutive", 0))
        state["prior_pose"] = pose
        metrics = {"inliers": inliers, "iterations": self.iterations}
        evidence = {"initialization": label}
        if consecutive >= 1 and previous is not None:
            from . import workbench_geometry as wg
            ang = wg.rotation_angle_deg(previous[:3, :3], pose[:3, :3])
            dt = float(np.linalg.norm(previous[:3, 3] - pose[:3, 3]) / max(np.linalg.norm(previous[:3, 3]), 1e-9))
            # Diagnostics only: record the measured deltas next to the unchanged thresholds.
            ang_ok, dt_ok = bool(ang <= self.consistency[0]), bool(dt <= self.consistency[1])
            for key, value in (("temporal_rotation_delta_deg", ang), ("temporal_relative_translation", dt),
                               ("temporal_rotation_threshold_deg", self.consistency[0]),
                               ("temporal_relative_translation_threshold", self.consistency[1])):
                if math.isfinite(float(value)):
                    metrics[key] = float(value)
            evidence["temporal_consistency"] = {
                "checked": True, "compared_with": "previous checked candidate pose",
                "rotation_within_threshold": ang_ok, "relative_translation_within_threshold": dt_ok}
            if ang <= self.consistency[0] and dt <= self.consistency[1]:
                state["consecutive"] = consecutive + 1
                return {"status": "tracking", "pose_m": pose, "metrics": metrics, "evidence": evidence}
            state["consecutive"] = 1
            return {"status": "recovering", "pose_m": None, "metrics": metrics, "evidence": evidence,
                    "reason": ("recovering: current candidate failed temporal consistency "
                               f"(rotation delta {float(ang):.3f} deg vs threshold {float(self.consistency[0]):.3f} deg; "
                               f"relative translation {dt:.4f} vs threshold {float(self.consistency[1]):.4f}); "
                               "artwork withheld")}
        evidence["temporal_consistency"] = {
            "checked": False, "pending": "no previous checked pose since initialization/loss"}
        state["consecutive"] = 1
        return {"status": "recovering", "pose_m": None, "metrics": metrics, "evidence": evidence,
                "reason": ("recovering: initial temporal check pending; the first checked pose after "
                           "initialization/loss has no previous checked pose to compare and is withheld")}


# --------------------------------------------------------------------------
# prompted segmentation (SAM 2.1 video predictor; observed masks only)

SAM2_CHECKPOINT_RELATIVE = ".cache/model-quality/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_SOURCES_RELATIVE = ".cache/model-quality/sources/sam2"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def segmentation_preflight(*, root: Path = ROOT, verify_hash: bool = False) -> dict:
    import importlib.util
    from .quality_assets import MODELS
    ckpt = root / SAM2_CHECKPOINT_RELATIVE
    expected = MODELS["sam2.1_hiera_base_plus.pt"]["sha256"].lower()
    checks = [_check("sam2_checkpoint", ckpt.is_file(), str(ckpt), "restore the pinned SAM 2.1 checkpoint")]
    if ckpt.is_file() and verify_hash:
        ok = sha256_file(ckpt) == expected
        checks.append(_check("sam2_checkpoint_sha256", ok, expected[:12], "checkpoint hash differs from the pin"))
    checks.append(_check("sam2_sources", (root / SAM2_SOURCES_RELATIVE).is_dir(), SAM2_SOURCES_RELATIVE,
                         "restore pinned SAM 2 sources"))
    checks.append(_check("torch_runtime", importlib.util.find_spec("torch") is not None, "torch",
                         "use the reviewed runtime"))
    ready = all(c["ok"] for c in checks)
    return {"ready": ready, "checks": checks, "checkpoint_sha256": expected, "config": SAM2_CONFIG,
            "blocked_reasons": [c["name"] for c in checks if not c["ok"]]}


class _CausalFrames:
    """Lazy SAM frame list; reads only frames already released to the predictor."""

    def __init__(self, read_rgb: Callable[[int], np.ndarray], count: int, image_size: int, released: int,
                 height: int, width: int):
        self.read_rgb, self.count, self.image_size = read_rgb, count, image_size
        self.released, self.height, self.width = released, height, width
        self.cache: dict[int, Any] = {}
        self.reads: list[int] = []

    def __len__(self):
        return self.count

    def release(self, index: int):
        if index != self.released + 1 and index > self.released:
            raise AdmissionError("segmentation frames must be released one at a time")
        self.released = max(self.released, index)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        if not isinstance(index, int) or not 0 <= index < self.count or index > self.released:
            raise AdmissionError(f"segmentation attempted to read unreleased (future) frame {index}")
        if index in self.cache:
            return self.cache[index]
        rgb = self.read_rgb(index)
        if rgb.shape[:2] != (self.height, self.width):
            raise AdmissionError("segmentation frame dimensions differ")
        self.reads.append(index)
        resized = np.array(Image.fromarray(rgb).resize((self.image_size, self.image_size)))
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255
        t = (t - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        self.cache[index] = t
        while len(self.cache) > 4:
            self.cache.pop(min(self.cache))
        return t


def _sam_state_entries(state: Any) -> int | None:
    """Count cached per-frame entries in a SAM 2 inference state (best effort)."""
    try:
        total = 0
        for key in ("output_dict", "consolidated_frame_inds"):
            value = state.get(key) if isinstance(state, Mapping) else None
            if isinstance(value, Mapping):
                total += sum(len(v) for v in value.values() if hasattr(v, "__len__"))
        per_obj = state.get("output_dict_per_obj") if isinstance(state, Mapping) else None
        if isinstance(per_obj, Mapping):
            for obj in per_obj.values():
                if isinstance(obj, Mapping):
                    total += sum(len(v) for v in obj.values() if hasattr(v, "__len__"))
        cached = state.get("cached_features") if isinstance(state, Mapping) else None
        if isinstance(cached, Mapping):
            total += len(cached)
        return total
    except Exception:  # noqa: BLE001
        return None


def segment_prompted(read_rgb: Callable[[int], np.ndarray], count: int, prompt_index: int,
                     selection: wc.Selection, *, height: int, width: int, write_mask: Callable[[int, np.ndarray], None],
                     device: str = "cpu", cancel=None, root: Path = ROOT,
                     resource_samples: list | None = None, disk_path: Path | None = None,
                     sample_every: int = 16) -> dict:
    """Prompt SAM 2.1 with reviewed positive/negative source-pixel points at
    ``prompt_index`` and emit observed masks causally.

    Frames ``0..prompt_index`` are the onboarding prefix (available at prompt
    time; masks before the prompt use reverse propagation within the prefix).
    Later frames are released strictly one at a time during forward
    propagation, so no future frame is read before its mask is produced.

    ``resource_samples`` (caller-owned, so it survives failures) receives a
    measured sample at initialization, every ``sample_every`` emitted frames
    and at the end: RSS, CUDA allocated/reserved/free bytes (null plus error
    when CUDA is broken), SAM state entry count and free disk bytes.
    """
    pre = segmentation_preflight(root=root, verify_hash=True)
    if not pre["ready"]:
        raise NeuralUnavailable(f"segmentation prerequisites missing: {pre['blocked_reasons']}")
    import torch
    from . import workbench_telemetry as wt
    init_started = time.perf_counter()
    sys.path.insert(0, str(root / SAM2_SOURCES_RELATIVE))
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    from sam2.modeling.position_encoding import PositionEmbeddingRandom  # type: ignore
    import sam2.sam2_video_predictor as predictor_module  # type: ignore
    from .quality_sam2 import deterministic_position_grid
    PositionEmbeddingRandom.forward = deterministic_position_grid
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(root / SAM2_CHECKPOINT_RELATIVE), device=device,
                                           apply_postprocessing=False, vos_optimized=False)
    frames = _CausalFrames(read_rgb, count, predictor.image_size, prompt_index, height, width)
    original = predictor_module.load_video_frames
    predictor_module.load_video_frames = lambda *a, **k: (frames, height, width)
    try:
        with torch.inference_mode():
            state = predictor.init_state("workbench-causal-source", offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
    finally:
        predictor_module.load_video_frames = original
    wt.gpu_synchronize()
    initialization_ms = (time.perf_counter() - init_started) * 1000.0
    samples = resource_samples if resource_samples is not None else []

    def sample(label: str) -> None:
        try:
            samples.append(wt.resource_sample(label, disk_path=disk_path,
                                              extra={"sam_state_entries": _sam_state_entries(state),
                                                     "emitted_frames": len(emitted)}))
        except Exception as exc:  # noqa: BLE001 - sampling must never mask the real failure
            samples.append({"label": label, "sample_error": f"{type(exc).__name__}: {exc}"[:300]})
    frame_ms: dict[int, float] = {}
    # integer pixel centres -> SAM continuous corner-origin coordinates
    points = np.array([[x + 0.5, y + 0.5] for x, y in selection.positive + selection.negative], np.float32)
    labels = np.array([1] * len(selection.positive) + [0] * len(selection.negative), np.int32)
    emitted = []
    clock = {"t": time.perf_counter()}
    sample("initialized")

    def emit(frame_index, logits):
        wt.gpu_synchronize()
        now = time.perf_counter()
        frame_ms[int(frame_index)] = (now - clock["t"]) * 1000.0
        mask = (logits[0, 0].detach().cpu().numpy() > 0)
        if mask.shape != (height, width):
            raise AdmissionError("SAM mask dimensions differ from the source frame")
        write_mask(frame_index, mask)
        emitted.append(int(frame_index))
        if len(emitted) % max(1, int(sample_every)) == 0:
            sample(f"after_{len(emitted)}_frames")
        clock["t"] = time.perf_counter()
    with torch.inference_mode():
        predictor.add_new_points_or_box(state, frame_idx=prompt_index, obj_id=1, points=points, labels=labels)
        if prompt_index > 0:
            for frame_index, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=prompt_index,
                                                                             reverse=True):
                if cancel is not None and cancel.is_set():
                    break
                if list(obj_ids) != [1] or frame_index > prompt_index:
                    raise AdmissionError("unexpected reverse propagation output")
                if frame_index != prompt_index:
                    emit(frame_index, logits)
        iterator = predictor.propagate_in_video(state, start_frame_idx=prompt_index, reverse=False)
        while True:
            if cancel is not None and cancel.is_set():
                break
            try:
                frame_index, obj_ids, logits = next(iterator)
            except StopIteration:
                break
            if list(obj_ids) != [1]:
                raise AdmissionError("unexpected object ids from SAM")
            emit(frame_index, logits)
            if frame_index + 1 < count:
                frames.release(frame_index + 1)
    sample("finished")
    return {"emitted_frames": sorted(emitted), "frames_read": frames.reads,
            "telemetry": {"initialization_ms": initialization_ms,
                          "segmentation_ms_per_frame": {str(k): v for k, v in sorted(frame_ms.items())},
                          "note": "per-frame time from previous emission (includes decode of that frame); "
                                  "GPU synchronized when CUDA is used",
                          "process_peak_rss_bytes": wt.process_peak_rss_bytes(), **wt.gpu_peak_bytes(),
                          "resource_samples": list(samples)},
            "causal_guard": "unreleased-frame reads raise; forward frames released one at a time",
            "checkpoint_sha256": pre["checkpoint_sha256"], "config": SAM2_CONFIG, "device": device,
            "prompt_index": prompt_index, "cancelled": bool(cancel is not None and cancel.is_set()),
            "point_convention": "integer pixel centres + 0.5 -> SAM corner-origin coordinates"}


# --------------------------------------------------------------------------
# frozen geometry-bound artwork for both neural roles

_GL_COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GL_WIDTH = {"SCALAR": 1, "VEC3": 3}


def _node_matrix(node: Mapping[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF is column-major
    M = np.eye(4)
    if "rotation" in node:
        x, y, z, w = (float(v) for v in node["rotation"])
        M[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                              [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                              [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    if "scale" in node:
        M[:3, :3] = M[:3, :3] @ np.diag(np.asarray(node["scale"], dtype=np.float64))
    if "translation" in node:
        M[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    return M


def glb_mesh(path: Path, *, max_faces: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Triangle geometry of a GLB in its metric object frame with every node
    transform applied (same frame as the GoTrack renderer). Unsupported glTF
    features (external buffers, sparse accessors, non-triangle modes) are rejected."""
    raw = _regular(Path(path), "metric mesh GLB").read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise AdmissionError("not a glTF 2.0 binary container")
    offset, gltf, binary = 12, None, b""
    while offset < len(raw):
        clen, ctype = struct.unpack_from("<I4s", raw, offset)
        chunk = raw[offset + 8: offset + 8 + clen]
        if ctype == b"JSON":
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            binary = chunk
        offset += 8 + clen
    if not isinstance(gltf, dict):
        raise AdmissionError("GLB has no JSON chunk")
    _reject_external_resources(gltf)
    if any("uri" in b for b in gltf.get("buffers", [])):
        raise AdmissionError("only the embedded GLB binary buffer is supported for geometry")

    def accessor(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        if "sparse" in acc or acc.get("type") not in _GL_WIDTH or acc.get("componentType") not in _GL_COMPONENTS:
            raise AdmissionError("unsupported GLB accessor")
        view = gltf["bufferViews"][acc["bufferView"]]
        dtype = np.dtype(_GL_COMPONENTS[acc["componentType"]]).newbyteorder("<")
        width = _GL_WIDTH[acc["type"]]
        count = int(acc["count"])
        start = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        stride = int(view.get("byteStride", 0)) or dtype.itemsize * width
        if start + stride * (count - 1) + dtype.itemsize * width > len(binary):
            raise AdmissionError("GLB accessor exceeds its buffer")
        rows = np.lib.stride_tricks.as_strided(np.frombuffer(binary, np.uint8, offset=start),
                                               shape=(count, dtype.itemsize * width), strides=(stride, 1))
        return np.ascontiguousarray(rows).view(dtype).reshape(count, width)

    verts, faces, base = [], [], 0
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    stack = [(n, np.eye(4)) for n in scene.get("nodes", range(len(gltf.get("nodes", []))))]
    while stack:
        index, parent = stack.pop()
        node = gltf["nodes"][index]
        world = parent @ _node_matrix(node)
        stack.extend((c, world) for c in node.get("children", []))
        if "mesh" not in node:
            continue
        for prim in gltf["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                raise AdmissionError("only triangle primitives are supported")
            pos = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            idx = (accessor(prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(pos)))
            tri = idx.astype(np.int64).reshape(-1, 3)
            verts.append(pos @ world[:3, :3].T + world[:3, 3])
            faces.append(tri + base)
            base += len(pos)
    if not verts:
        raise AdmissionError("GLB contains no triangle geometry")
    v = np.concatenate(verts)
    f = np.concatenate(faces)
    if len(f) > max_faces or not np.isfinite(v).all():
        raise AdmissionError("GLB geometry is too large or non-finite")
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return v, f[keep]


class NeuralArtwork:
    """Frozen artwork anchored once on the pinned metric mesh and shared by both
    neural roles (identical anchor). Anchor: the first reviewed positive point
    on the seed (selection) frame, ray-cast with the labelled seed pose. Units
    are metres. Face support comes from CAD geometry (not observation); folded,
    steep, back-facing, overlapped or mask-outside pixels are still suppressed,
    and nothing renders without a validated current pose and observed mask."""

    def __init__(self, bundle: MetricBundle, selection: wc.Selection, artwork_kind: str, *, clip_K,
                 width: int, height: int, selection_mask: np.ndarray | None, size_fraction: float = 0.25,
                 max_faces: int = 200_000):
        from . import workbench_geometry as wg
        if bundle.seed_pose_m is None:
            raise NeuralUnavailable("artwork anchor needs the labelled seed pose")
        if selection.frame_id != bundle.seed_frame_id:
            raise NeuralUnavailable("artwork anchor needs the selection on the seed frame "
                                    f"({selection.frame_id} != {bundle.seed_frame_id})")
        if artwork_kind not in ("checkerboard", "directional"):
            raise NeuralUnavailable("only fixed built-in artwork is bound for neural roles")
        v, f = glb_mesh(bundle.glb_path, max_faces=max_faces)
        self.vertices, self.faces = v, f.astype(np.int64)
        self.K = native_K(np.asarray(clip_K, float).ravel(), bundle.principal_point_origin)
        self.width, self.height = width, height
        u, vv = selection.positive[0]
        hit = wg.anchor_from_pixel(v, f, self.K, bundle.seed_pose_m, u, vv, width=width, height=height,
                                   observed_mask=selection_mask)
        size = size_fraction * float(np.max(np.ptp(v, axis=0)))
        self.art = wg.builtin_artwork(artwork_kind, 256)
        art_sha = wg.array_sha256(self.art)
        self.marker = wg.make_marker(v, f, hit, bundle.seed_pose_m, size, art_sha)
        self.support = np.full(len(f), 3, dtype=np.int32)
        self.patch = wg.marker_patch(v, f, self.support, self.marker)
        self.record = {"schema": "workbench_neural_artwork_v1", "units": "metres", "glb_sha256": bundle.glb_sha256,
                       "marker": self.marker, "anchor_pixel": [u, vv], "anchor_frame_id": selection.frame_id,
                       "seed_label": bundle.seed_label, "artwork": {"kind": artwork_kind, "edge_px": 256,
                                                                     "sha256": art_sha, "generation": "fixed_builtin_v1"},
                       "support_source": "metric CAD geometry (not observation-derived)",
                       "principal_point_origin": bundle.principal_point_origin, "effective_K": self.K.ravel().tolist(),
                       "patch_faces": int(len(self.patch["faces"])), "mesh_faces": int(len(f)),
                       "anchor_mask_checked": selection_mask is not None}
        self.sha256 = wc.canonical_sha256(self.record)

    def render(self, pose_m: np.ndarray, observed_mask: np.ndarray, _clip_K=None) -> dict:
        from . import workbench_geometry as wg
        return wg.render_marker(self.vertices, self.faces, self.support, self.marker, self.art, self.K,
                                np.asarray(pose_m, float), self.width, self.height, observed_mask, patch=self.patch)
