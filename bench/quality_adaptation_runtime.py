"""Pinned upstream adapters and bounded persistence; no imports trigger inference."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

from .quality_adaptation_data import read_json, sha256, canonical_hash

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "bench/runtime/adaptation-sources.json"
GOTRACK_CHECKPOINT = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
SAM_CHECKPOINT = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
MODEL_BUDGET = 8 * 1024**3


def cloud_guard(explicit: bool) -> None:
    """Operator assertion, not a claim that Linux itself proves cloud placement."""
    if not explicit or os.environ.get("VISUALIZEIT_CLOUD") != "1" or platform.system() != "Linux":
        raise RuntimeError("requires approved Linux cloud host, VISUALIZEIT_CLOUD=1 and --cloud-execution")


def tree_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"symlinks are not permitted in bounded output/cache tree: {item}")
        if item.is_file():
            total += item.stat().st_size
    return total


def verify_sources(kind: str = "gotrack") -> dict:
    lock = read_json(LOCK)
    if kind not in ("gotrack", "sam2"):
        raise ValueError("unknown upstream source")
    for relative, expected in lock[kind]["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"pinned upstream/source adapter mismatch: {relative}")
    return {"source_lock_sha256": sha256(LOCK), "upstream_revision": lock[kind]["revision"],
            "source_files_sha256": canonical_hash(lock[kind]["files"])}


def verify_checkpoint(kind="gotrack") -> Path:
    names = {"gotrack": ("gotrack_checkpoint.pt", GOTRACK_CHECKPOINT),
             "sam2": ("sam2.1_hiera_base_plus.pt", SAM_CHECKPOINT)}
    name, expected = names[kind]
    path = ROOT / ".cache/model-quality/checkpoints" / name
    if not path.is_file() or path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"missing or mismatched pre-existing checkpoint: {path}; downloads are never automatic")
    cache = ROOT / ".cache/model-quality"
    if sum(tree_bytes(cache / child) for child in ("checkpoints", "sources")) > MODEL_BUDGET:
        raise ValueError("source and checkpoint cache exceeds 8 GiB")
    return path


def runtime_versions(*, sam=False) -> dict:
    expected = {"torch": "2.5.1", "torchvision": "0.20.1", "numpy": "1.26.4",
                "einops": "0.8.0"}
    if sam:
        expected.update({"hydra-core": "1.3.2", "omegaconf": "2.3.0", "submitit": "1.5.2",
                         "tensordict": "0.6.2", "tensorboard": "2.17.1", "pycocotools": "2.0.8"})
    actual = {}
    for package, version in expected.items():
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as e:
            raise RuntimeError(f"missing optional cloud dependency {package}=={version}") from e
        if actual[package].split("+", 1)[0] != version:
            raise RuntimeError(f"expected {package}=={version}, found {actual[package]}")
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise RuntimeError("adaptation runtime requires Python 3.10 or 3.11")
    return actual


def _source_imports(kind):
    source = ROOT / ".cache/model-quality/sources" / kind
    prefixes = ("model", "utils", "dinov2") if kind == "gotrack" else ("training", "sam2")
    for name, module in list(sys.modules.items()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(source.resolve()):
                raise RuntimeError(f"conflicting upstream import {name}; use a fresh process")
    paths = [source]
    if kind == "gotrack":
        paths += [source / "external/dinov2", source / "external/bop_toolkit"]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    return source


def load_head(*, initialize: bool, device: str):
    """Load the actual pinned DPTHead, never a substitute architecture.

    Only pose_head parameters are trainable. Backbone/decoder are absent from
    this process: their frozen outputs are authenticated feature archives.
    """
    verify_sources()
    _source_imports("gotrack")
    import torch
    from model.heads.dpt.model import DPTHead
    from model.heads.dpt.config import DPTHeadOpts
    head = DPTHead(DPTHeadOpts())
    if initialize:
        checkpoint = verify_checkpoint()
        # Official hash is checked first. No fallback to arbitrary pickle loading.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = "models.1.pose_head."
        selected = {k[len(prefix):]: v for k, v in state["model_state_dict"].items() if k.startswith(prefix)}
        if not selected:
            raise ValueError("official checkpoint has no models.1.pose_head namespace")
        head.load_state_dict(selected, strict=True)
    return head.to(device)


def load_frozen_network(device: str):
    verify_sources()
    verify_checkpoint()
    _source_imports("gotrack")
    # Existing loader authenticates the official checkpoint and prevents its
    # upstream constructors from downloading DINO/hub weights.
    from .quality_gotrack import load_network
    network = load_network(device)
    network.requires_grad_(False)
    return network.eval()


def _check_destination(root: Path, target: Path) -> None:
    if root.is_symlink() or not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("output path must remain inside its nonsymlink run directory")


@contextmanager
def atomic_stream(root: Path, target: Path, *, max_file_bytes: int, max_output_bytes: int,
                  reserve_bytes: int = 1024**3):
    """Reserve transient space before writing; old checkpoint survives failures."""
    _check_destination(root, target)
    current = tree_bytes(root)
    if current + max_file_bytes > max_output_bytes:
        raise ValueError("insufficient output budget for atomic temporary artifact")
    if shutil.disk_usage(root).free < max_file_bytes + reserve_bytes:
        raise ValueError("insufficient free storage for artifact and reserve")
    temp = root / (".adaptation-" + uuid.uuid4().hex + ".tmp")
    class CappedWriter:
        def __init__(self, raw):
            self.raw = raw
        def write(self, data):
            if self.raw.tell() + len(data) > max_file_bytes:
                raise ValueError("artifact exceeded reserved file size")
            return self.raw.write(data)
        def __getattr__(self, name):
            return getattr(self.raw, name)
    try:
        with temp.open("xb") as raw:
            yield CappedWriter(raw)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def write_json(root: Path, name: str, value, *, max_output_bytes=1536 * 1024**2):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with atomic_stream(root, root / name, max_file_bytes=len(encoded),
                       max_output_bytes=max_output_bytes) as f:
        f.write(encoded)


def save_checkpoint(root: Path, name: str, state: dict, config, *, torch_module=None):
    if name not in ("last.pt", "best.pt"):
        raise ValueError("retention permits only last.pt and best.pt")
    if torch_module is None:
        import torch as torch_module
    with atomic_stream(root, root / name, max_file_bytes=config.max_checkpoint_bytes,
                       max_output_bytes=config.max_output_bytes) as f:
        torch_module.save(state, f)


def check_resume(state: dict, *, identity: str, config_hash: str, source_hash: str) -> None:
    if (state.get("schema") != "correspondence_head_checkpoint_v1" or state.get("dataset_sha256") != identity
            or state.get("config_sha256") != config_hash or state.get("source_lock_sha256") != source_hash):
        raise ValueError("resume checkpoint schema/dataset/config/source identity mismatch")
    for key in ("epoch", "batch_cursor", "steps", "bad_epochs"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f"invalid checkpoint {key}")
    elapsed = state.get("elapsed_seconds")
    import math
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid checkpoint elapsed_seconds")


def training_state_digest(value) -> str:
    """Hash nested optimizer tensors/scalars independently of serializer bytes."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            h.update(b"dict\0")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(f"sequence:{len(item)}\0".encode())
            for child in item:
                visit(child)
        elif hasattr(item, "detach"):
            a = item.detach().cpu().contiguous().numpy()
            h.update(b"tensor\0" + a.dtype.str.encode() + b"\0" + str(a.shape).encode() + b"\0" + a.tobytes())
        else:
            h.update((type(item).__name__ + ":" + json.dumps(item, allow_nan=False) + "\0").encode())
    visit(value)
    return h.hexdigest()


def load_export_into_network(network, export_path: Path, *, expected_sha256: str,
                             cloud_execution: bool = False):
    """Explicit research-only deployment hook; existing frozen default is unchanged.

    Caller must first construct the pinned frozen GoTrack network. Strict head
    keys match its ``pose_head``; no ground-truth arrays enter this interface.
    This hook is intentionally cloud-only until independent replay validates it.
    """
    cloud_guard(cloud_execution)
    verify_sources()
    if sha256(export_path) != expected_sha256:
        raise ValueError("adapted head export hash mismatch")
    import torch
    value = torch.load(export_path, map_location="cpu", weights_only=True)
    if value.get("schema") != "correspondence_head_export_v1" or value.get("source_lock_sha256") != sha256(LOCK):
        raise ValueError("incompatible adapted head export")
    if value.get("base_checkpoint_sha256") != GOTRACK_CHECKPOINT:
        raise ValueError("export does not match frozen GoTrack checkpoint")
    network.pose_head.load_state_dict(value["head"], strict=True)
    network.eval()
    return network
