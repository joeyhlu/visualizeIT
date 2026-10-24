"""Bounded, provider-independent cloud training for frozen pose features.

No optimization is permitted on this workstation. The train/resume/smoke CLI
requires Linux and --cloud-execution. Only train/validation labels enter this
module; sealed test evaluation belongs to quality_ranker_evaluate.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time
from typing import Any

import numpy as np


class _TrainingPurpose(str, Enum):
    SUBSTANTIVE = "substantive_v1"
    INTEGRATION_SMOKE = "integration_smoke_v1"
    CONFIDENCE_REJECTION_RESEARCH = "confidence_rejection_research_v1"
    EXPERIMENTAL_ORDERING = "experimental_ordering_v1"


_INTEGRATION_CALIBRATION_UNAVAILABLE_REASON = "integration_smoke_does_not_fit_calibration"
_INTEGRATION_SMOKE_DEADLINE_SECONDS = 300.0


class IncompleteResearchRun(RuntimeError):
    """A resumable research checkpoint exists, but the run is not complete."""
    def __init__(self, reason: str, checkpoint: Path):
        super().__init__(f"Confidence research stopped incomplete ({reason}); resume {checkpoint}")
        self.reason = reason
        self.checkpoint = checkpoint


@dataclass(frozen=True)
class TrainConfig:
    architecture: str = "linear"
    seed: int = 42
    epochs: int = 40
    batch_size: int = 8  # Frames stay intact for within-frame ordering.
    accumulation: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    pairwise_weight: float = 0.5
    patience: int = 6
    min_delta: float = 0.00001
    max_seconds: float = 1800.0
    max_output_bytes: int = 64 * 1024 * 1024
    max_checkpoint_bytes: int = 16 * 1024 * 1024
    max_candidates_per_group: int = 256
    max_feature_rows: int = 100000
    amp: bool = False
    device: str = "cpu"

    def validate(self) -> None:
        if self.architecture not in ("linear", "mlp64_32"):
            raise ValueError("architecture must be linear or mlp64_32")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("epochs", "batch_size", "accumulation", "patience",
                     "max_output_bytes", "max_checkpoint_bytes", "max_candidates_per_group", "max_feature_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "pairwise_weight", "min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.device not in ("cpu", "cuda") or (self.amp and self.device != "cuda"):
            raise ValueError("device must be cpu/cuda; AMP requires CUDA")


def cloud_guard(explicit: bool) -> None:
    """An explicit cloud assertion plus Linux prevents accidental local runs.

    This is an operator boundary, not a claim that Linux implies cloud hosting.
    The caller must provision the approved cloud host before using this flag.
    """
    if not explicit or platform.system() != "Linux":
        raise RuntimeError("Optimization requires an approved Linux cloud host and --cloud-execution")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(purpose: _TrainingPurpose | None = None) -> dict[str, str]:
    names = (
        "quality_ranker_train.py", "quality_ranker_features.py", "quality_pose_ranker.py",
        "quality_ranker_data.py", "quality_contract.py", "quality_gotrack.py",
    )
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        names += (
            "quality_confidence_rejection.py", "quality_confidence_evaluate.py",
            "quality_pipeline.py", "quality_runner.py", "quality_offline.py",
        )
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        names += ("quality_ordering_research.py",)
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def _validate_research_config(config: TrainConfig, *, smoke: bool = False) -> None:
    expected = TrainConfig(
        architecture=config.architecture, seed=42, epochs=2 if smoke else 40,
        batch_size=8, accumulation=4, learning_rate=.001, weight_decay=.0001,
        pairwise_weight=.5, patience=6, min_delta=.00001, max_seconds=300.,
        max_output_bytes=64 * 1024 * 1024, max_checkpoint_bytes=16 * 1024 * 1024,
        max_candidates_per_group=256, max_feature_rows=100000,
        amp=False, device="cpu",
    )
    if config != expected:
        kind = "smoke" if smoke else "training"
        raise ValueError(f"Confidence research {kind} config differs from the frozen CPU contract")


def fit_normalization(training_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(training_features, dtype=np.float64)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("Normalization needs nonempty finite training features")
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def ordering_pairs(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Only strictly different quality labels yield order constraints.

    BCE retains every eligible candidate, including frames with all bad labels.
    All-bad groups must never invent a preferred winner.
    """
    target = np.asarray(targets, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Quality targets must be a finite vector in [0,1]")
    high, low = np.where(target[:, None] > target[None, :])
    return high.astype(np.int64), low.astype(np.int64)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def bce(logits: np.ndarray, targets: np.ndarray) -> float:
    z, y = np.asarray(logits, dtype=float), np.asarray(targets, dtype=float)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def interior_thresholds(values):
    """Validation decision cutoffs away from observed score ties where possible."""
    values = np.unique(np.asarray(values, dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Calibration cutoffs require finite probabilities in [0,1]")
    edges = np.unique(np.concatenate(([0.], values, [1.])))
    return sorted(set([0., 1.] + ((edges[:-1] + edges[1:]) * .5).tolist()))


def calibration(logits: np.ndarray, targets: np.ndarray, *,
                strict_threshold: bool = False) -> dict[str, float]:
    """Fixed validation-only grid; no optimizer or test access.

    BCE is a proper score for candidate quality probabilities. The calibrated
    acceptance threshold maximizes balanced accuracy (equal class importance).
    Threshold ties choose the more conservative threshold deterministically.
    """
    z, y = np.asarray(logits, dtype=float), np.asarray(targets, dtype=float)
    if z.shape != y.shape or z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Calibration needs matched finite validation vectors")
    ordering_pairs(y)
    candidates = [(bce(z / temperature + bias, y), temperature, bias)
                  for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
                  for bias in (-2., -1., -0.5, 0., 0.5, 1., 2.)]
    loss, temperature, bias = min(candidates)
    p, positive = sigmoid(z / temperature + bias), y >= 0.5
    # A one-class validation set cannot establish an acceptance threshold.
    if not positive.any() or positive.all():
        raise ValueError("Validation calibration requires both good and bad candidates")
    choices = []
    thresholds = interior_thresholds(p)
    if strict_threshold:
        thresholds = [threshold for threshold in thresholds if 0. < threshold < 1.]
        if not thresholds:
            raise ValueError("Strict calibration found no interior probability threshold")
    for threshold in thresholds:
        accepted = p >= threshold
        balanced = 0.5 * (accepted[positive].mean() + (~accepted[~positive]).mean())
        choices.append((float(balanced), float(threshold)))
    balanced, threshold = max(choices)
    return {"temperature": float(temperature), "bias": float(bias),
            "acceptance_probability": threshold, "validation_bce": loss,
            "validation_balanced_accuracy": balanced}


def describe(features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    x = np.asarray(features, dtype=float)
    quantiles = np.quantile(x, [0., .01, .25, .5, .75, .99, 1.], axis=0)
    centered = x - x.mean(axis=0)
    denom = np.sqrt(np.sum(centered ** 2, axis=0))
    corr = (centered.T @ centered) / np.maximum(denom[:, None] * denom[None, :], 1e-12)
    return {"rows": len(x), "feature_quantiles": quantiles.tolist(),
            "feature_missingness": np.mean(~np.isfinite(x), axis=0).tolist(),
            "feature_zero_fraction": np.mean(x == 0, axis=0).tolist(),
            "encoded_missing_fraction": np.mean(x[:, 1::2] == 0, axis=0).tolist(),
            "feature_correlations": corr.tolist(), "target_mean": float(np.mean(targets)),
            "target_good_fraction": float(np.mean(np.asarray(targets) >= .5))}


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def storage_guard(output: Path, config: TrainConfig, *, reserve: int = 0) -> None:
    used = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
    if used + reserve > config.max_output_bytes:
        raise RuntimeError("Training output storage budget exhausted")
    if shutil.disk_usage(output).free < reserve + 64 * 1024 * 1024:
        raise RuntimeError("Cloud training needs a 64 MiB free-space reserve")


def memory_metrics(torch: Any, device: str) -> dict[str, int]:
    import resource  # Cloud Linux only; ru_maxrss is KiB on Linux.
    return {"host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0}


def build_model(torch: Any, width: int, architecture: str) -> Any:
    if architecture == "linear":
        return torch.nn.Sequential(torch.nn.Linear(width, 1))
    if architecture == "mlp64_32":
        return torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    raise ValueError("Unknown architecture")


def batch_loss(torch: Any, logits: Any, targets: Any, lengths: list[int], weight: float) -> Any:
    # Equal frame weighting avoids frames with many alternatives dominating BCE.
    losses, offset = [], 0
    for length in lengths:
        z, y = logits[offset:offset + length], targets[offset:offset + length]
        quality = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        high, low = ordering_pairs(y.detach().cpu().numpy())
        pair = torch.nn.functional.softplus(-(z[high] - z[low])).mean() if len(high) else z.sum() * 0
        losses.append(quality + weight * pair)
        offset += length
    return torch.stack(losses).mean()


def _flatten(groups: list[dict], names: list[str], config: TrainConfig) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rows, labels, indices = [], [], []
    for group in groups:
        eligible = [c for c in group["candidates"] if c.get("eligible", True)]
        if not eligible:
            continue
        if len(eligible) > config.max_candidates_per_group:
            raise ValueError("Candidate count exceeds bounded training group limit")
        start = len(rows)
        for candidate in eligible:
            vector = candidate["features"]
            rows.append([vector[name] for name in names] if isinstance(vector, dict) else vector)
            labels.append(candidate["quality_target"])
            if len(rows) > config.max_feature_rows:
                raise ValueError("Feature rows exceed bounded training memory limit")
        indices.append(np.arange(start, len(rows), dtype=np.int64))
    x, y = np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all() or not len(indices):
        raise ValueError("Split needs eligible finite feature rows in the frozen feature schema")
    ordering_pairs(y)
    return x, y, indices


def _rng_state(torch: Any) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(torch: Any, state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(torch: Any, output: Path, name: str, state: dict, config: TrainConfig) -> Path:
    # At most last.pt, best.pt and one temporary checkpoint; no unbounded epochs.
    storage_guard(output, config, reserve=config.max_checkpoint_bytes)
    path, temp = output / name, output / (name + ".tmp")
    try:
        with temp.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if temp.stat().st_size > config.max_checkpoint_bytes:
            raise RuntimeError("Checkpoint exceeds configured size limit")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def _read_checkpoint(torch: Any, path: Path) -> dict:
    # Resume files contain Python/NumPy RNG tuples: load only the operator's own
    # trusted run checkpoints. torch.load is never used for public model JSON.
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Resume checkpoint exceeds 16 MiB limit")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Resume checkpoint must contain a state object")
    return state


def _load_dataset(manifest: Path, *, purpose: _TrainingPurpose = _TrainingPurpose.SUBSTANTIVE) -> dict:
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        from .quality_ordering_research import load_research_dataset
        return load_research_dataset(manifest)
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        from .quality_confidence_rejection import load_research_dataset
        return load_research_dataset(manifest)
    from .quality_ranker_data import audit_cache, load_training_split
    from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
    if not isinstance(purpose, _TrainingPurpose):
        raise ValueError("Training dataset purpose must be explicit")
    if manifest.name != "cache_manifest.json":
        raise ValueError("--manifest must point to the frozen cache_manifest.json")
    cache_dir = manifest.parent
    audit = audit_cache(cache_dir, training_only=True)
    if audit.get("approved") is not True:
        raise ValueError("Frozen cache audit must be approved before training")
    if (audit["identity"].get("feature_names") != list(FEATURE_NAMES) or
            audit["identity"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION):
        raise ValueError("Frozen cache feature schema differs from the inference ranker")
    if (purpose is _TrainingPurpose.SUBSTANTIVE and
            audit.get("ranking_signal_available") is not True):
        raise ValueError("Candidate-error audit found no usable ranking alternatives in train and validation; "
                         "ranking alone cannot address this cache. Inspect candidate_error_audit before training.")
    if type(audit.get("ranking_signal_available")) is not bool:
        raise ValueError("Frozen cache audit has a malformed ranking-signal result")
    train_groups = load_training_split(cache_dir, "train")
    validation_groups = load_training_split(cache_dir, "validation")
    return {"train": train_groups,
            "validation": validation_groups,
            "identity": audit["identity"], "feature_names": list(FEATURE_NAMES),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "ranking_signal_available": audit["ranking_signal_available"]}


def _check_deadline(deadline: float | None, phase: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"Integration smoke exceeded its 300-second deadline during {phase}")


def _integration_supervision_summary(groups: list[dict], names: list[str], config: TrainConfig) -> dict[str, int | bool]:
    """Validate real-cache labels and count the supervision actually available."""
    for group in groups:
        context = group.get("context")
        if not isinstance(context, dict):
            raise ValueError("Integration smoke group context must be an object")
        provenance_kind = context.get("provenance_kind")
        if provenance_kind == "fixture":
            raise ValueError("Integration smoke rejects fixture-provenance training data")
        if provenance_kind not in ("real", "synthetic"):
            raise ValueError("Integration smoke requires explicit real or synthetic cache provenance")
        candidates = group.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Integration smoke candidates must be an array")
        for candidate in candidates:
            if not isinstance(candidate, dict) or type(candidate.get("eligible")) is not bool:
                raise ValueError("Integration smoke candidate eligibility is malformed")
            if candidate["eligible"]:
                target = candidate.get("quality_target")
                if type(target) is not int or target not in (0, 1):
                    raise ValueError("Integration smoke requires finite binary labels for eligible candidates")

    features, targets, indices = _flatten(groups, names, config)
    positives = int(np.count_nonzero(targets == 1))
    negatives = int(np.count_nonzero(targets == 0))
    mixed_groups = 0
    pair_count = 0
    for index in indices:
        labels = targets[index]
        group_positives = int(np.count_nonzero(labels == 1))
        group_negatives = int(np.count_nonzero(labels == 0))
        mixed_groups += int(group_positives > 0 and group_negatives > 0)
        pair_count += len(ordering_pairs(labels)[0])
    return {
        "eligible_groups": len(indices),
        "eligible_candidates": int(len(features)),
        "good_label_count": positives,
        "bad_label_count": negatives,
        "mixed_group_count": mixed_groups,
        "ordering_pair_count": pair_count,
        "pair_supervision_available": pair_count > 0,
    }


def _torch_state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_resume_identity(loaded: dict, identity: dict, config: TrainConfig,
                              purpose: _TrainingPurpose) -> None:
    if not isinstance(loaded, dict):
        raise ValueError("Resume checkpoint must contain a state object")
    checkpoint_identity = loaded.get("identity")
    if (loaded.get("purpose") != purpose.value or
            not isinstance(checkpoint_identity, dict) or
            checkpoint_identity.get("purpose") != purpose.value):
        raise ValueError("Resume purpose differs from the checkpoint")
    if checkpoint_identity != identity or loaded.get("config") != asdict(config):
        raise ValueError("Resume config/source/data/split identities differ")


def _require_checkpoint_purpose(state: dict, purpose: _TrainingPurpose, *, operation: str) -> None:
    if not isinstance(state, dict):
        raise ValueError("Training checkpoint must contain a state object")
    identity = state.get("identity")
    if (state.get("purpose") != purpose.value or not isinstance(identity, dict) or
            identity.get("purpose") != purpose.value):
        if state.get("purpose") == _TrainingPurpose.INTEGRATION_SMOKE.value:
            raise ValueError("Normal export explicitly rejects integration-smoke checkpoints")
        raise ValueError(f"Checkpoint purpose does not match the requested {operation} purpose")
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint {operation} config is malformed")
    try:
        expected_config_sha256 = canonical_hash({"purpose": purpose.value, "config": config})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint {operation} config is malformed") from exc
    if identity.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"Checkpoint {operation} purpose/config identity differs")


def _write_research_completion(torch: Any, output: Path, selected: Path,
                               final: Path, config: TrainConfig,
                               termination_reason: str) -> dict:
    selected_state = _read_checkpoint(torch, selected)
    final_state = _read_checkpoint(torch, final)
    final_progress = final_state.get("progress")
    if (selected.name != "best.pt" or final.name != "last.pt" or
            selected_state.get("purpose") != _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH.value or
            final_state.get("purpose") != _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH.value or
            selected_state.get("identity") != final_state.get("identity") or
            selected_state.get("config") != asdict(config) or
            final_state.get("config") != asdict(config) or
            not isinstance(final_progress, dict)):
        raise AssertionError("Confidence completion did not retain the validation-selected best checkpoint")
    selected_loss = selected_state.get("progress", {}).get("best_loss")
    if (not isinstance(selected_loss, (int, float)) or isinstance(selected_loss, bool) or
            not math.isfinite(float(selected_loss)) or
            not isinstance(final_progress.get("best_loss"), (int, float)) or
            isinstance(final_progress.get("best_loss"), bool) or
            float(selected_loss) != float(final_progress["best_loss"])):
        raise AssertionError("Confidence completion best loss differs from the selected checkpoint")
    identity = final_state["identity"]
    receipt = {
        "schema_version": 1, "artifact_purpose": _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH.value,
        "policy": "legacy_veto_v1", "complete": True,
        "termination_reason": termination_reason,
        "checkpoint_selection": "legacy_selected_validation_uncalibrated_bce",
        "identity_sha256": canonical_hash(identity),
        "config_sha256": identity["config_sha256"],
        "selected_checkpoint": "best.pt", "selected_checkpoint_sha256": file_hash(selected),
        "selected_model_sha256": _torch_state_sha256(selected_state["model"]),
        "selected_validation_bce": float(selected_loss),
        "final_checkpoint": "last.pt", "final_checkpoint_sha256": file_hash(final),
        "epochs_completed": int(final_progress["epoch"]),
        "optimizer_steps": int(final_progress["step"]),
        "test_read": False,
    }
    atomic_json(output / "completion.json", receipt)
    return receipt


def _validate_research_completion(torch: Any, checkpoint: Path,
                                  selected_state: dict) -> dict:
    checkpoint = Path(checkpoint)
    if checkpoint.name != "best.pt":
        raise ValueError("Confidence export requires the validation-selected best.pt")
    completion_path = checkpoint.parent / "completion.json"
    if not completion_path.is_file():
        raise ValueError("Confidence export requires a completed research run")
    try:
        receipt = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Confidence completion receipt is unreadable") from exc
    fields = {
        "schema_version", "artifact_purpose", "policy", "complete",
        "termination_reason", "checkpoint_selection", "identity_sha256",
        "config_sha256", "selected_checkpoint", "selected_checkpoint_sha256",
        "selected_model_sha256", "selected_validation_bce", "final_checkpoint",
        "final_checkpoint_sha256", "epochs_completed", "optimizer_steps", "test_read",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise ValueError("Confidence completion receipt fields are malformed")
    if (type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1 or
            receipt["artifact_purpose"] != _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH.value or
            receipt["policy"] != "legacy_veto_v1" or receipt["complete"] is not True or
            receipt["termination_reason"] not in ("epochs_complete", "early_stopping") or
            receipt["checkpoint_selection"] != "legacy_selected_validation_uncalibrated_bce" or
            receipt["selected_checkpoint"] != "best.pt" or receipt["final_checkpoint"] != "last.pt" or
            receipt["test_read"] is not False):
        raise ValueError("Confidence completion receipt contradicts the research contract")
    identity = selected_state.get("identity")
    config = selected_state.get("config")
    if not isinstance(identity, dict) or not isinstance(config, dict):
        raise ValueError("Confidence selected checkpoint differs from completion receipt")
    try:
        research_config = TrainConfig(**config)
        _validate_research_config(research_config, smoke=research_config.epochs == 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence selected checkpoint config is malformed") from exc
    if (type(receipt["epochs_completed"]) is not int or
            type(receipt["optimizer_steps"]) is not int or
            receipt["epochs_completed"] <= 0 or receipt["optimizer_steps"] <= 0 or
            receipt["epochs_completed"] > research_config.epochs or
            (receipt["termination_reason"] == "epochs_complete" and
             receipt["epochs_completed"] != research_config.epochs) or
            (receipt["termination_reason"] == "early_stopping" and
             receipt["epochs_completed"] >= research_config.epochs)):
        raise ValueError("Confidence completion receipt progress is malformed")
    receipt_loss = receipt["selected_validation_bce"]
    if (type(receipt_loss) not in (int, float) or not math.isfinite(float(receipt_loss))):
        raise ValueError("Confidence completion receipt validation BCE is malformed")
    if (
            receipt["identity_sha256"] != canonical_hash(identity) or
            receipt["config_sha256"] != identity.get("config_sha256") or
            receipt["selected_checkpoint_sha256"] != file_hash(checkpoint) or
            receipt["selected_model_sha256"] != _torch_state_sha256(selected_state["model"])):
        raise ValueError("Confidence selected checkpoint differs from completion receipt")
    loss = selected_state.get("progress", {}).get("best_loss")
    if (not isinstance(loss, (int, float)) or isinstance(loss, bool) or
            not math.isfinite(float(loss)) or float(loss) != float(receipt_loss)):
        raise ValueError("Confidence selected checkpoint lacks validation-selected BCE proof")
    final_path = checkpoint.parent / "last.pt"
    if not final_path.is_file() or file_hash(final_path) != receipt["final_checkpoint_sha256"]:
        raise ValueError("Confidence final checkpoint differs from completion receipt")
    final_state = _read_checkpoint(torch, final_path)
    final_progress = final_state.get("progress")
    if (not isinstance(final_progress, dict) or
            type(final_progress.get("epoch")) is not int or
            type(final_progress.get("step")) is not int or
            type(final_progress.get("bad_epochs")) is not int or
            not isinstance(final_progress.get("best_loss"), (int, float)) or
            isinstance(final_progress.get("best_loss"), bool) or
            not math.isfinite(float(final_progress["best_loss"]))):
        raise ValueError("Confidence final state progress is malformed")
    if (final_state.get("purpose") != _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH.value or
            final_state.get("identity") != identity or final_state.get("config") != config or
            final_progress["epoch"] != receipt["epochs_completed"] or
            final_progress["step"] != receipt["optimizer_steps"] or
            float(final_progress["best_loss"]) != float(loss) or
            (receipt["termination_reason"] == "early_stopping" and
             final_progress["bad_epochs"] < research_config.patience)):
        raise ValueError("Confidence final state does not bind the completed selected checkpoint")
    return receipt


def calibrate_margin(groups: list[dict], probabilities: np.ndarray, acceptance: float) -> float:
    """Validation-only balanced accuracy over accepted top candidates.

    Distinctness uses the same pose geometry and fixed tracker thresholds as
    inference. Equal-score ties follow the candidate ID, just as learned rank.
    """
    from types import SimpleNamespace
    from .quality_contract import distinct_pose
    frames, offset = [], 0
    for group in groups:
        candidates = [c for c in group["candidates"] if c.get("eligible", True)]
        scores = probabilities[offset:offset + len(candidates)]
        offset += len(candidates)
        if not candidates:
            continue
        context = group.get("context", {})
        settings = SimpleNamespace(recovery_rotation_degrees=float(context.get("recovery_rotation_degrees", 20.)),
                                   recovery_translation_m=float(context.get("recovery_translation_m", .03)))
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["candidate_id"]))
        top = order[0]
        pose = np.asarray(candidates[top]["pose"], dtype=float)
        competing = [float(scores[i]) for i in order[1:]
                     if distinct_pose(pose, np.asarray(candidates[i]["pose"], dtype=float), settings)]
        gap = (float(scores[top]) - max(competing)
               if competing and context.get("recovery", False) else 1.0)
        frames.append((float(scores[top]), gap, candidates[top]["quality_target"] >= .5))
    if offset != len(probabilities) or not frames:
        raise ValueError("Margin calibration probabilities do not match validation groups")
    good = np.array([f[2] for f in frames], dtype=bool)
    if good.all() or not good.any():
        # Without both correct and incorrect top choices, prefer the maximum
        # rejection margin; validation cannot justify permissive ambiguity.
        return 1.0
    trials = []
    for margin in interior_thresholds([f[1] for f in frames]):
        accepted = np.array([prob >= acceptance and gap >= margin for prob, gap, _ in frames])
        balanced = .5 * (accepted[good].mean() + (~accepted[~good]).mean())
        trials.append((float(balanced), float(margin)))
    return max(trials)[1]


def decision_signatures(groups, probabilities, acceptance, margin):
    """Use the production decision rule to check cross-runtime export behavior."""
    from types import SimpleNamespace
    from .quality_pose_ranker import LearnedPoseRanker, RankCandidate
    scorer = SimpleNamespace(probability=lambda row: float(row[0]),
                             acceptance_probability=float(acceptance),
                             distinct_pose_margin_probability=float(margin),
                             selection_enabled=True)
    scorer.rank_with_scores = lambda context, candidates, scores: LearnedPoseRanker.rank_with_scores(
        scorer, context, candidates, scores)
    decisions, offset = [], 0
    for group in groups:
        candidates = [c for c in group["candidates"] if c.get("eligible", True)]
        rows = []
        for index, candidate in enumerate(candidates):
            rows.append(RankCandidate(candidate["candidate_id"], index,
                                      SimpleNamespace(pose=np.asarray(candidate["pose"], dtype=float)),
                                      {}, [probabilities[offset + index]]))
        offset += len(rows)
        result = LearnedPoseRanker.rank(scorer, group.get("context", {}), rows)
        decisions.append((result.ordered_ids, result.selected_id, result.rejection_reason))
    if offset != len(probabilities):
        raise ValueError("Decision parity probabilities differ from validation candidate count")
    return decisions


def _integration_decision_signatures(groups: list[dict], probabilities: np.ndarray, ranker: Any) -> list[tuple]:
    """Exercise the loaded ranker's real disabled decision path for each score stream."""
    from types import SimpleNamespace
    from .quality_pose_ranker import RankCandidate
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    decisions, offset = [], 0
    for group in groups:
        candidates = [candidate for candidate in group["candidates"] if candidate.get("eligible", True)]
        rows = [RankCandidate(candidate["candidate_id"], index,
                               SimpleNamespace(pose=np.asarray(candidate["pose"], dtype=float)),
                               {}, candidate["features"])
                for index, candidate in enumerate(candidates)]
        scores = values[offset:offset + len(rows)]
        if len(scores) != len(rows):
            raise ValueError("Integration decision scores do not match eligible validation candidates")
        offset += len(rows)
        result = ranker.rank_with_scores(group.get("context", {}), rows, scores)
        if result.selected_id is not None or result.rejection_reason != "integration_only_selection_disabled":
            raise AssertionError("Integration artifact did not reject model selection explicitly")
        decisions.append((result.ordered_ids, result.selected_id, result.rejection_reason))
    if offset != len(values):
        raise ValueError("Integration decision scores exceed eligible validation candidates")
    return decisions


def _train_core(manifest: Path, output: Path, config: TrainConfig, *, cloud_execution: bool,
                purpose: _TrainingPurpose, resume: Path | None = None,
                stop_after_step: int | None = None, deadline: float | None = None) -> Path:
    cloud_guard(cloud_execution)
    if not isinstance(purpose, _TrainingPurpose):
        raise ValueError("Training purpose must be explicit")
    if purpose is _TrainingPurpose.INTEGRATION_SMOKE:
        expected = TrainConfig(architecture="linear", seed=42, epochs=2, batch_size=1,
                               accumulation=1, patience=2, max_seconds=300.0,
                               amp=False, device="cpu")
        if config != expected:
            raise ValueError("Integration smoke training config is fixed to linear CPU, seed 42, two epochs, and 300 seconds")
    elif purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        _validate_research_config(config, smoke=config.epochs == 2)
    elif purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        if config.pairwise_weight <= 0.:
            raise ValueError("Ordering research requires positive pairwise_weight")
    _check_deadline(deadline, "training setup")
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not str(torch.__version__).startswith("2.5.1"):
        raise RuntimeError("Cloud trainer requires pinned PyTorch 2.5.1")
    if (purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH and
            np.__version__ != "1.26.4"):
        raise RuntimeError("Confidence research requires pinned NumPy 1.26.4")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    dataset = _load_dataset(manifest, purpose=purpose)
    names, version = list(dataset["feature_names"]), dataset["feature_schema_version"]
    if purpose is _TrainingPurpose.INTEGRATION_SMOKE:
        _integration_supervision_summary(dataset["train"], names, config)
        _integration_supervision_summary(dataset["validation"], names, config)
    train_x, train_y, train_groups = _flatten(dataset["train"], names, config)
    val_x, val_y, val_groups = _flatten(dataset["validation"], names, config)
    # Split isolation is the data module's responsibility. Its audit must bind
    # cache, label and split hashes before train/validation are exposed here.
    identity = {"manifest_sha256": file_hash(manifest), "data": dataset["identity"],
                "source": source_identity(purpose),
                "config_sha256": canonical_hash({"purpose": purpose.value, "config": asdict(config)}),
                "purpose": purpose.value,
                "feature_names": names, "feature_schema_version": version,
                "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                            "python": platform.python_version(), "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(),
                            "device": torch.cuda.get_device_name() if config.device == "cuda" else "cpu"}}
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        from .quality_confidence_rejection import POLICY
        identity.update(policy=POLICY, config=asdict(config),
                        label_policy_sha256=dataset["label_policy_sha256"])
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        identity.update(config=asdict(config), ordering_evidence=dataset["ordering_evidence"],
                        ordering_evidence_sha256=canonical_hash(dataset["ordering_evidence"]))
    mean, scale = fit_normalization(train_x)
    x = torch.tensor((train_x - mean) / scale, dtype=torch.float32, device=config.device)
    y = torch.tensor(train_y, dtype=torch.float32, device=config.device)
    vx = torch.tensor((val_x - mean) / scale, dtype=torch.float32, device=config.device)
    vy = torch.tensor(val_y, dtype=torch.float32, device=config.device)
    selected_validation_x = selected_validation_y = svx = svy = None
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        selected_validation_x = np.asarray(
            [row["features"] for row in dataset["selected_validation"]], dtype=np.float64)
        selected_validation_y = np.asarray(
            [row["quality_target"] for row in dataset["selected_validation"]], dtype=np.float64)
        if (selected_validation_x.ndim != 2 or selected_validation_x.shape[1] != len(names) or
                not np.isfinite(selected_validation_x).all()):
            raise ValueError("Confidence selected validation features are malformed")
        svx = torch.tensor((selected_validation_x - mean) / scale,
                           dtype=torch.float32, device=config.device)
        svy = torch.tensor(selected_validation_y, dtype=torch.float32, device=config.device)
        # The research EDA is persisted before model initialization. Its
        # validation outliers are defined against train-only IQR bounds.
        from .quality_confidence_rejection import research_eda
        if resume is not None and resume.resolve().parent != output.resolve():
            raise ValueError("Resume must use the original run output directory (retains best.pt)")
        output.mkdir(parents=True, exist_ok=True)
        if resume is not None and (output / "completion.json").exists():
            raise ValueError("A completed confidence run cannot be resumed")
        if resume is None and any((output / name).exists()
                                  for name in ("last.pt", "best.pt", "events.jsonl",
                                               "completion.json")):
            raise ValueError("Fresh training requires an unused output directory")
        atomic_json(output / "eda.json", {
            "train": research_eda(dataset["train"], names),
            "validation": research_eda(dataset["validation"], names,
                                       train_reference=train_x),
            "identity": identity, "normalization_fit_split": "train_eligible_labeled_only",
            "development_fitting": True, "test_read": False,
        })
    model = build_model(torch, len(names), config.architecture).to(config.device)
    initial_model_sha256 = _torch_state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    output.mkdir(parents=True, exist_ok=True)
    state = {"epoch": 0, "cursor": 0, "order": [], "step": 0, "best_loss": math.inf,
             "bad_epochs": 0, "elapsed_seconds": 0.0}
    if resume:
        if resume.resolve().parent != output.resolve():
            raise ValueError("Resume must use the original run output directory (retains best.pt)")
        loaded = _read_checkpoint(torch, resume)
        _validate_resume_identity(loaded, identity, config, purpose)
        if loaded.get("initial_model_sha256") != initial_model_sha256:
            raise ValueError("Resume initial model identity differs")
        if not np.array_equal(mean, loaded["mean"]) or not np.array_equal(scale, loaded["scale"]):
            raise ValueError("Resume training normalization differs")
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        scaler.load_state_dict(loaded["scaler"])
        state.update(loaded["progress"])
        if state["epoch"] > 0 and not (output / "best.pt").is_file():
            raise ValueError("Resume is missing the retained validation-selected best.pt")
        _restore_rng(torch, loaded["rng"])
    elif any((output / name).exists() for name in ("last.pt", "best.pt", "events.jsonl",
                                                   "completion.json")):
        raise ValueError("Fresh training requires an unused output directory")
    storage_guard(output, config)
    if purpose is not _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        atomic_json(output / "eda.json", {"train": describe(train_x, train_y), "validation": describe(val_x, val_y),
                    "identity": identity, "normalization_fit_split": "train", "test_read": False})
    started, previous_elapsed = time.monotonic(), state["elapsed_seconds"]

    def save(name: str) -> Path:
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        return _save(torch, output, name, {"schema_version": 1, "purpose": purpose.value,
                "identity": identity, "initial_model_sha256": initial_model_sha256,
                "config": asdict(config), "mean": mean, "scale": scale,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": _rng_state(torch), "progress": dict(state)}, config)

    def event(value: dict) -> None:
        storage_guard(output, config, reserve=8192)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    try:
        while state["epoch"] < config.epochs and state["bad_epochs"] < config.patience:
            _check_deadline(deadline, "epoch start")
            if not state["order"]:
                state["order"] = np.random.permutation(len(train_groups)).tolist()
            model.train()
            while state["cursor"] < len(train_groups):
                if deadline is not None and time.monotonic() >= deadline:
                    partial = save("last.pt")
                    event({"kind": "stopped", "reason": "outer_deadline", "step": state["step"]})
                    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
                        raise IncompleteResearchRun("outer_deadline", partial)
                    return output / "last.pt"
                if previous_elapsed + time.monotonic() - started >= config.max_seconds:
                    partial = save("last.pt")
                    event({"kind": "stopped", "reason": "runtime_limit", "step": state["step"]})
                    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
                        raise IncompleteResearchRun("runtime_limit", partial)
                    return output / "last.pt"
                optimizer.zero_grad(set_to_none=True)
                # Each optimizer step is an indivisible resume boundary. A
                # partial final accumulation uses its actual frame count.
                stop = min(state["cursor"] + config.batch_size * config.accumulation, len(train_groups))
                total_groups = stop - state["cursor"]
                step_loss = 0.0
                for cursor in range(state["cursor"], stop, config.batch_size):
                    _check_deadline(deadline, "training batch")
                    selected = [train_groups[i] for i in state["order"][cursor:min(cursor + config.batch_size, stop)]]
                    row_ids = np.concatenate(selected)
                    with torch.autocast(device_type=config.device, enabled=config.amp):
                        logits = model(x[row_ids]).squeeze(-1)
                        loss = batch_loss(torch, logits, y[row_ids], [len(g) for g in selected], config.pairwise_weight)
                        if not bool(torch.isfinite(loss).all()):
                            raise FloatingPointError("Training loss became nonfinite")
                        fraction = len(selected) / total_groups
                    scaler.scale(loss * fraction).backward()
                    step_loss += float(loss.detach().cpu()) * fraction
                scaler.step(optimizer)
                scaler.update()
                state["cursor"], state["step"] = stop, state["step"] + 1
                if not math.isfinite(step_loss):
                    raise FloatingPointError("Observed optimizer-step loss is nonfinite")
                save("last.pt")
                event({"kind": "step", "epoch": state["epoch"], "step": state["step"],
                       "loss": step_loss, "elapsed_seconds": state["elapsed_seconds"],
                       **memory_metrics(torch, config.device)})
                if deadline is not None and time.monotonic() >= deadline:
                    event({"kind": "stopped", "reason": "outer_deadline", "step": state["step"]})
                    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
                        raise IncompleteResearchRun("outer_deadline", output / "last.pt")
                    return output / "last.pt"
                if stop_after_step is not None and state["step"] >= stop_after_step:
                    return output / "last.pt"
            model.eval()
            with torch.no_grad():
                val_logits = model(vx).squeeze(-1)
                if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
                    selected_val_logits = model(svx).squeeze(-1)
                    validation_loss = bce(selected_val_logits.cpu().numpy(), selected_validation_y)
                    reported_validation_bce = validation_loss
                else:
                    validation_loss = float(batch_loss(torch, val_logits, vy, [len(g) for g in val_groups], config.pairwise_weight).cpu())
                    reported_validation_bce = bce(val_logits.cpu().numpy(), val_y)
                train_logits = model(x).squeeze(-1).cpu().numpy()
            improved = validation_loss < state["best_loss"] - config.min_delta
            if improved:
                state["best_loss"] = validation_loss
            state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
            event({"kind": "epoch", "epoch": state["epoch"], "validation_loss": validation_loss,
                   "train_bce": bce(train_logits, train_y), "validation_bce": reported_validation_bce,
                   "validation_scope": ("legacy_selected_labeled_eligible" if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH else "eligible_candidate_objective"),
                   "selected": improved, "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                   **memory_metrics(torch, config.device)})
            scheduler.step()
            state["epoch"], state["cursor"], state["order"] = state["epoch"] + 1, 0, []
            if improved:
                save("best.pt")
            # Commit best before advancing last: interruption cannot make last
            # refer to a selected checkpoint that was never persisted.
            save("last.pt")
        selected = output / "best.pt"
        if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
            _ordering_completion(torch, selected, config=config, write=True)
        if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
            if not selected.is_file():
                raise RuntimeError("Confidence research completed without a validation-selected best checkpoint")
            final = output / "last.pt"
            termination = ("epochs_complete" if state["epoch"] >= config.epochs
                           else "early_stopping")
            _write_research_completion(torch, output, selected, final, config, termination)
        return selected
    except KeyboardInterrupt:
        # Last checkpoint is the completed optimizer boundary; do not serialize
        # advanced RNG or partially updated gradients from an interrupted step.
        event({"kind": "interrupted", "resume": "last.pt"})
        raise


def train(manifest: Path, output: Path, config: TrainConfig, *, cloud_execution: bool,
          resume: Path | None = None, stop_after_step: int | None = None) -> Path:
    """Run substantive training with the normal audited ranking-signal gates."""
    return _train_core(manifest, output, config, cloud_execution=cloud_execution,
                       purpose=_TrainingPurpose.SUBSTANTIVE, resume=resume,
                       stop_after_step=stop_after_step)


def ordering_train(manifest: Path, output: Path, config: TrainConfig, *, cloud_execution: bool,
                   resume: Path | None = None, stop_after_step: int | None = None) -> Path:
    """Fit experimental ordering only; never establishes heldout ordering gains."""
    return _train_core(manifest, output, config, cloud_execution=cloud_execution,
                       purpose=_TrainingPurpose.EXPERIMENTAL_ORDERING, resume=resume,
                       stop_after_step=stop_after_step)


def _ordering_completion(torch: Any, checkpoint: Path, *, config: TrainConfig | None = None,
                         write: bool = False) -> dict:
    """Bind completion to the retained selected and final checkpoints, not a partial run."""
    from .quality_ordering_research import CHECKPOINT_SELECTION
    if checkpoint.name != "best.pt":
        raise ValueError("Ordering export requires validation-selected best.pt")
    selected = _read_checkpoint(torch, checkpoint)
    final_path = checkpoint.parent / "last.pt"
    final = _read_checkpoint(torch, final_path)
    for state in (selected, final):
        _require_checkpoint_purpose(state, _TrainingPurpose.EXPERIMENTAL_ORDERING, operation="completion")
    config = config or TrainConfig(**selected["config"])
    progress, selected_progress = final["progress"], selected["progress"]
    if (selected["identity"] != final["identity"] or selected["config"] != final["config"] or
            selected["config"] != asdict(config) or
            progress["best_loss"] != selected_progress["best_loss"] or
            not math.isfinite(float(progress["best_loss"])) or
            type(progress["epoch"]) is not int or type(progress["step"]) is not int or
            progress["epoch"] <= 0 or progress["epoch"] > config.epochs or progress["step"] <= 0 or
            progress["cursor"] != 0 or progress["order"] != [] or
            not (progress["epoch"] == config.epochs or progress["bad_epochs"] >= config.patience)):
        raise ValueError("Ordering export requires a completed run and retained validation-selected objective")
    if (type(selected_progress.get("epoch")) is not int or
            not 0 < selected_progress["epoch"] <= progress["epoch"] or
            type(selected_progress.get("step")) is not int or
            not 0 < selected_progress["step"] <= progress["step"] or
            selected_progress.get("cursor") != 0 or selected_progress.get("order") != [] or
            selected.get("initial_model_sha256") != final.get("initial_model_sha256")):
        raise ValueError("Ordering completion has no completed validation-selected epoch proof")
    receipt = {"artifact_purpose": _TrainingPurpose.EXPERIMENTAL_ORDERING.value,
               "complete": True, "checkpoint_selection": CHECKPOINT_SELECTION,
               "identity_sha256": canonical_hash(selected["identity"]),
               "selected_checkpoint_sha256": file_hash(checkpoint),
               "final_checkpoint_sha256": file_hash(final_path),
               "selected_model_sha256": _torch_state_sha256(selected["model"]),
               "selected_validation_objective": float(progress["best_loss"]),
               "epochs_completed": progress["epoch"], "optimizer_steps": progress["step"],
               "termination_reason": "epochs_complete" if progress["epoch"] == config.epochs else "early_stopping",
               "test_read": False, "dev_read": False}
    path = checkpoint.parent / "completion.json"
    if write:
        atomic_json(path, receipt)
    elif not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Ordering completion receipt mismatch or missing")
    return receipt


def confidence_train(manifest: Path, output: Path, config: TrainConfig, *,
                     cloud_execution: bool, resume: Path | None = None,
                     stop_after_step: int | None = None) -> Path:
    """Run the frozen research-only legacy-selection confidence experiment."""
    _validate_research_config(config)
    return _train_core(
        manifest, output, config, cloud_execution=cloud_execution,
        purpose=_TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH,
        resume=resume, stop_after_step=stop_after_step)


def _export_core(checkpoint: Path, manifest: Path, output: Path, *, cloud_execution: bool,
                 purpose: _TrainingPurpose, onnx: bool = True,
                 deadline: float | None = None) -> dict:
    cloud_guard(cloud_execution)
    if not isinstance(purpose, _TrainingPurpose):
        raise ValueError("Export purpose must be explicit")
    _check_deadline(deadline, "export setup")
    import torch
    state = _read_checkpoint(torch, checkpoint)
    _require_checkpoint_purpose(state, purpose, operation="export")
    ordering_completion = None
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        ordering_completion = _ordering_completion(torch, checkpoint)
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        if (not str(torch.__version__).startswith("2.5.1") or
                np.__version__ != "1.26.4"):
            raise RuntimeError("Confidence export requires pinned PyTorch 2.5.1 and NumPy 1.26.4")
        _validate_research_completion(torch, checkpoint, state)
    dataset = _load_dataset(manifest, purpose=purpose)
    identity = state["identity"]
    if (file_hash(manifest) != identity["manifest_sha256"] or
            dataset["identity"] != identity["data"] or
            source_identity(purpose) != identity["source"]):
        raise ValueError("Export source/data/split identities differ from checkpoint")
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        if (identity.get("ordering_evidence") != dataset["ordering_evidence"] or
                identity.get("ordering_evidence_sha256") != canonical_hash(dataset["ordering_evidence"]) or
                identity.get("config") != state["config"] or state["config"].get("pairwise_weight", 0) <= 0):
            raise ValueError("Ordering export evidence/config differs from checkpoint")
    config = TrainConfig(**state["config"])
    config.validate()
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        _validate_research_config(config, smoke=config.epochs == 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    storage_guard(output.parent, config, reserve=config.max_checkpoint_bytes)
    names = identity["feature_names"]
    model = build_model(torch, len(names), config.architecture)
    model.load_state_dict(state["model"])
    model.eval()
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        features = np.asarray([row["features"] for row in dataset["selected_validation"]],
                              dtype=np.float64)
        targets = np.asarray([row["quality_target"] for row in dataset["selected_validation"]],
                             dtype=np.float64)
    else:
        features, targets, _ = _flatten(dataset["validation"], names, config)
    normalized = (features - state["mean"]) / state["scale"]
    with torch.no_grad():
        logits = model(torch.tensor(normalized, dtype=torch.float32)).squeeze(-1).numpy()
    _check_deadline(deadline, "model scoring")
    if purpose is _TrainingPurpose.INTEGRATION_SMOKE:
        return _export_integration_smoke(state, checkpoint, dataset, output, config,
                                         model, features, logits, torch, onnx=onnx,
                                         deadline=deadline)
    if purpose is _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH:
        return _export_confidence_research(
            state, checkpoint, dataset, output, config, model, logits, targets,
            torch, onnx=onnx, deadline=deadline)
    calibrated = calibration(logits, targets)
    layers = []
    for module in model:
        if isinstance(module, torch.nn.Linear):
            layers.append({"weight": module.weight.detach().numpy().tolist(),
                           "bias": module.bias.detach().numpy().tolist(), "activation": "identity"})
        elif isinstance(module, torch.nn.ReLU):
            layers[-1]["activation"] = "relu"
    last = layers[-1]
    last["weight"] = (np.asarray(last["weight"]) / calibrated["temperature"]).tolist()
    last["bias"] = (np.asarray(last["bias"]) / calibrated["temperature"] + calibrated["bias"]).tolist()
    # Margin is supplied by the validation-only decision calibration module.
    margin = calibrate_margin(dataset["validation"], sigmoid(logits / calibrated["temperature"] + calibrated["bias"]),
                              calibrated["acceptance_probability"])
    # Training checkpoints bind their execution purpose. Keep that internal
    # field out of the unchanged production v1 artifact identity contract.
    artifact_identity = {key: value for key, value in identity.items() if key != "purpose"}
    payload = {"schema_version": 1, "feature_schema_version": identity["feature_schema_version"],
               "feature_names": names, "architecture": config.architecture,
               "mean": state["mean"].tolist(), "scale": state["scale"].tolist(), "layers": layers,
               "acceptance_probability": calibrated["acceptance_probability"],
               "distinct_pose_margin_probability": float(margin),
               "provenance": {"identity": artifact_identity, "checkpoint_sha256": file_hash(checkpoint),
                              "calibration": calibrated, "calibration_split": "validation", "test_read": False}}
    from .quality_pose_ranker import LearnedPoseRanker, ExperimentalOrderingRanker
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        from .quality_ordering_research import ARTIFACT_PURPOSE, POLICY
        payload.update(schema_version=4, artifact_purpose=ARTIFACT_PURPOSE,
                       policy=POLICY, research_only=True)
        payload["provenance"].update(identity=identity, completion=ordering_completion,
                                     completion_sha256=canonical_hash(ordering_completion))
        ranker = ExperimentalOrderingRanker.from_checkpoint(payload)
    else:
        ranker = LearnedPoseRanker.from_checkpoint(payload)
    expected = sigmoid(logits / calibrated["temperature"] + calibrated["bias"])
    actual = np.asarray([ranker.probability(row) for row in features])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    expected_decisions = decision_signatures(dataset["validation"], expected,
                                             payload["acceptance_probability"], margin)
    if decision_signatures(dataset["validation"], actual, payload["acceptance_probability"], margin) != expected_decisions:
        raise AssertionError("JSON export changes validation selection/rejection decisions")
    receipt = {"numpy_max_absolute_error": float(np.max(np.abs(actual - expected))), "rows": len(features),
               "numpy_decision_parity": True, "decision_groups": len(expected_decisions), "onnx_checked": False}
    if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
        receipt.update(artifact_purpose=purpose.value, research_only=True,
                       ordering_evidence_sha256=identity["ordering_evidence_sha256"],
                       completion_sha256=canonical_hash(ordering_completion),
                       validation_ordering_evidence=dataset["ordering_evidence"]["validation_ordering_evidence"],
                       test_read=False, dev_read=False)
    if onnx:
        import onnxruntime as ort
        class Portable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.register_buffer("mean", torch.tensor(state["mean"], dtype=torch.float32))
                self.register_buffer("scale", torch.tensor(state["scale"], dtype=torch.float32))
            def forward(self, rows):
                return torch.sigmoid(self.model((rows - self.mean) / self.scale).squeeze(-1) / calibrated["temperature"] + calibrated["bias"])
        portable = Portable().eval()
        onnx_path = output.with_suffix(".onnx")
        onnx_temp = output.with_suffix(".onnx.tmp")
        torch.onnx.export(portable, torch.tensor(features[:min(8, len(features))], dtype=torch.float32), str(onnx_temp),
                          input_names=["features"], output_names=["probability"],
                          dynamic_axes={"features": {0: "candidates"}, "probability": {0: "candidates"}}, opset_version=17)
        if purpose is _TrainingPurpose.EXPERIMENTAL_ORDERING:
            import onnx as onnx_format
            graph = onnx_format.load(str(onnx_temp))
            onnx_format.helper.set_model_props(graph, {
                "artifact_purpose": purpose.value, "research_only": "true",
                "ordering_evidence_sha256": identity["ordering_evidence_sha256"],
                "config_sha256": identity["config_sha256"], "checkpoint_sha256": file_hash(checkpoint),
                "completion_sha256": canonical_hash(ordering_completion),
            })
            onnx_format.save(graph, str(onnx_temp))
        session = ort.InferenceSession(str(onnx_temp), providers=["CPUExecutionProvider"])
        predicted = session.run(None, {"features": features.astype(np.float32)})[0]
        np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-6)
        if decision_signatures(dataset["validation"], predicted, payload["acceptance_probability"], margin) != expected_decisions:
            raise AssertionError("ONNX export changes validation selection/rejection decisions")
        if onnx_temp.stat().st_size > config.max_checkpoint_bytes:
            raise RuntimeError("ONNX checkpoint exceeds configured size limit")
        os.replace(onnx_temp, onnx_path)
        receipt.update(onnx_checked=True, onnx_decision_parity=True,
                       onnx_max_absolute_error=float(np.max(np.abs(predicted - expected))), onnx_sha256=file_hash(onnx_path))
    atomic_json(output, payload)
    receipt["model_sha256"] = file_hash(output)
    atomic_json(output.with_suffix(".export.json"), receipt)
    return receipt


def _confidence_parity_rows(selected_all: list[dict], probabilities: np.ndarray,
                            ranker: Any) -> list[dict]:
    """Full validation-frame veto signatures with probability kept separate."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    rows, offset = [], 0
    for item in selected_all:
        legacy = item["legacy_decision"]
        probability = None
        if legacy.selected_id is None:
            decision = ranker.rank_with_selected_probability(
                item["frame"].get("context", {}), item["candidates"], float("nan"))
        else:
            if offset >= len(values):
                raise ValueError("Confidence parity probability stream is too short")
            probability = float(values[offset]); offset += 1
            decision = ranker.rank_with_selected_probability(
                item["frame"].get("context", {}), item["candidates"], probability)
        rows.append({
            "object_id": item["frame"]["object_id"],
            "sequence_id": item["frame"]["sequence_id"],
            "frame_id": item["frame"]["frame_id"],
            "ordered_ids": list(decision.ordered_ids),
            "legacy_scores": dict(decision.scores),
            "selected_id": decision.selected_id,
            "rejection_reason": decision.rejection_reason,
            "selected_probability": probability,
            "label_present": item["label"] is not None,
        })
    if offset != len(values):
        raise ValueError("Confidence parity probability stream has unused values")
    return rows


def _confidence_decision_signatures(rows: list[dict]) -> list[dict]:
    """Compare decisions exactly; probability diagnostics have a separate tolerance.

    Export already checks probabilities with assert_allclose. Including their
    float32/float64 rounding differences in exact decision equality rejects an
    otherwise equivalent export. Every other field stays in the signature.
    """
    return [{key: value for key, value in row.items()
             if key != "selected_probability"} for row in rows]


def _export_confidence_research(state: dict, checkpoint: Path, dataset: dict,
                                output: Path, config: TrainConfig, model: Any,
                                labeled_logits: np.ndarray, targets: np.ndarray,
                                torch: Any, *, onnx: bool,
                                deadline: float | None) -> dict:
    from .quality_confidence_rejection import (
        ANGULAR_OBSERVABILITY_INTERPRETATION, ARTIFACT_PURPOSE,
        CALIBRATION_RULE, CALIBRATION_SCOPE, CHECKPOINT_SELECTION, POLICY,
        QUALITY_TARGET_DEFINITION, UNSUPPORTED_CLAIMS,
    )
    calibrated = calibration(labeled_logits, targets, strict_threshold=True)
    layers = []
    for module in model:
        if isinstance(module, torch.nn.Linear):
            layers.append({"weight": module.weight.detach().numpy().tolist(),
                           "bias": module.bias.detach().numpy().tolist(),
                           "activation": "identity"})
        elif isinstance(module, torch.nn.ReLU):
            layers[-1]["activation"] = "relu"
    # Fold temperature and bias exactly once into the final affine layer.
    layers[-1]["weight"] = (
        np.asarray(layers[-1]["weight"]) / calibrated["temperature"]).tolist()
    layers[-1]["bias"] = (
        np.asarray(layers[-1]["bias"]) / calibrated["temperature"] +
        calibrated["bias"]).tolist()
    payload = {
        "schema_version": 3, "artifact_purpose": ARTIFACT_PURPOSE,
        "policy": POLICY, "research_only": True,
        "feature_schema_version": state["identity"]["feature_schema_version"],
        "feature_names": state["identity"]["feature_names"],
        "architecture": config.architecture,
        "mean": state["mean"].tolist(), "scale": state["scale"].tolist(),
        "layers": layers,
        "acceptance_probability": calibrated["acceptance_probability"],
        "provenance": {
            "identity": state["identity"],
            "checkpoint_sha256": file_hash(checkpoint),
            "label_policy": {
                "sha256": dataset["label_policy_sha256"],
                "frozen": dataset["label_policy"],
                "quality_target_definition": QUALITY_TARGET_DEFINITION,
                "angular_observability_interpretation": ANGULAR_OBSERVABILITY_INTERPRETATION,
            },
            "supervision_counts": dataset["supervision_counts"],
            "calibration": calibrated, "calibration_split": "validation",
            "calibration_scope": CALIBRATION_SCOPE,
            "calibration_rule": CALIBRATION_RULE,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "development_fitting": True, "test_read": False,
            "unsupported_claims": list(UNSUPPORTED_CLAIMS),
        },
    }
    from .quality_pose_ranker import ConfidenceRejectionRanker
    ranker = ConfidenceRejectionRanker.from_checkpoint(payload)

    selected_all = dataset["selected_all_validation"]
    all_features = np.asarray([
        row["selected_candidate"].features for row in selected_all
        if row["legacy_decision"].selected_id is not None
    ], dtype=np.float64)
    normalized_all = (all_features - state["mean"]) / state["scale"]
    with torch.no_grad():
        raw_all = model(torch.tensor(normalized_all, dtype=torch.float32)).squeeze(-1).numpy()
    expected = sigmoid(raw_all / calibrated["temperature"] + calibrated["bias"])
    actual = np.asarray([ranker.probability(row) for row in all_features], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    expected_rows = _confidence_parity_rows(selected_all, expected, ranker)
    actual_rows = _confidence_parity_rows(selected_all, actual, ranker)
    if _confidence_decision_signatures(actual_rows) != _confidence_decision_signatures(expected_rows):
        raise AssertionError("Confidence JSON export changes complete legacy-veto decisions")
    receipt = {
        "artifact_purpose": ARTIFACT_PURPOSE, "policy": POLICY,
        "research_only": True, "checkpoint_sha256": file_hash(checkpoint),
        "rows": len(all_features), "validation_frames": len(selected_all),
        "torch_numpy_max_absolute_error": float(np.max(np.abs(actual - expected))),
        "torch_numpy_probability_parity": True,
        "torch_numpy_complete_veto_decision_parity": True,
        "parity_rows": actual_rows, "onnx_checked": False,
        "development_fitting": True, "test_read": False,
        "unsupported_claims": list(UNSUPPORTED_CLAIMS),
    }
    if onnx:
        import onnx
        import onnxruntime as ort
        if not str(onnx.__version__).startswith("1.17"):
            raise RuntimeError("Confidence research requires pinned ONNX 1.17")
        if not str(ort.__version__).startswith("1.20"):
            raise RuntimeError("Confidence research requires pinned ONNX Runtime 1.20")

        class Portable(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model = model
                self.register_buffer("mean", torch.tensor(state["mean"], dtype=torch.float32))
                self.register_buffer("scale", torch.tensor(state["scale"], dtype=torch.float32))

            def forward(self, rows):
                logits = self.model((rows - self.mean) / self.scale).squeeze(-1)
                return torch.sigmoid(logits / calibrated["temperature"] + calibrated["bias"])

        onnx_path, temporary = output.with_suffix(".onnx"), output.with_suffix(".onnx.tmp")
        try:
            torch.onnx.export(
                Portable().eval(), torch.tensor(all_features[:min(8, len(all_features))], dtype=torch.float32),
                str(temporary), input_names=["features"], output_names=["probability"],
                dynamic_axes={"features": {0: "legacy_selected"}, "probability": {0: "legacy_selected"}},
                opset_version=17)
            session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
            predicted = np.asarray(session.run(None, {"features": all_features.astype(np.float32)})[0],
                                   dtype=np.float64).reshape(-1)
            np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-6)
            if (_confidence_decision_signatures(_confidence_parity_rows(selected_all, predicted, ranker)) !=
                    _confidence_decision_signatures(expected_rows)):
                raise AssertionError("Confidence ONNX export changes complete legacy-veto decisions")
            if temporary.stat().st_size > config.max_checkpoint_bytes:
                raise RuntimeError("ONNX checkpoint exceeds configured size limit")
            os.replace(temporary, onnx_path)
        finally:
            temporary.unlink(missing_ok=True)
        receipt.update(
            onnx_checked=True, onnx_probability_parity=True,
            onnx_complete_veto_decision_parity=True,
            onnx_torch_max_absolute_error=float(np.max(np.abs(predicted - expected))),
            onnx_sha256=file_hash(onnx_path))
    atomic_json(output, payload)
    receipt["model_sha256"] = file_hash(output)
    atomic_json(output.with_suffix(".export.json"), receipt)
    _check_deadline(deadline, "confidence research export")
    return receipt


def _export_integration_smoke(state: dict, checkpoint: Path, dataset: dict, output: Path,
                              config: TrainConfig, model: Any, features: np.ndarray,
                              logits: np.ndarray, torch: Any, *, onnx: bool,
                              deadline: float | None) -> dict:
    from .quality_pose_ranker import LearnedPoseRanker
    names = state["identity"]["feature_names"]
    training_summary = _integration_supervision_summary(dataset["train"], names, config)
    validation_summary = _integration_supervision_summary(dataset["validation"], names, config)
    torch_probabilities = torch.sigmoid(torch.from_numpy(np.asarray(logits)))
    expected = np.asarray(torch_probabilities.cpu().numpy(), dtype=np.float64).reshape(-1)
    layers = []
    for module in model:
        if isinstance(module, torch.nn.Linear):
            layers.append({"weight": module.weight.detach().numpy().tolist(),
                           "bias": module.bias.detach().numpy().tolist(), "activation": "identity"})
        elif isinstance(module, torch.nn.ReLU):
            layers[-1]["activation"] = "relu"
    payload = {
        "schema_version": 2,
        "artifact_purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
        "selection_enabled": False,
        "calibration_unavailable_reason": _INTEGRATION_CALIBRATION_UNAVAILABLE_REASON,
        "feature_schema_version": state["identity"]["feature_schema_version"],
        "feature_names": names,
        "architecture": config.architecture,
        "mean": state["mean"].tolist(),
        "scale": state["scale"].tolist(),
        "layers": layers,
        "provenance": {
            "artifact_purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
            "identity": state["identity"],
            "checkpoint_sha256": file_hash(checkpoint),
            "test_read": False,
        },
    }
    ranker = LearnedPoseRanker.from_checkpoint(payload)
    actual = np.asarray([ranker.probability(row) for row in features], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    expected_decisions = _integration_decision_signatures(dataset["validation"], expected, ranker)
    actual_decisions = _integration_decision_signatures(dataset["validation"], actual, ranker)
    if actual_decisions != expected_decisions:
        raise AssertionError("NumPy export changes integration-only score order or rejection decisions")

    _check_deadline(deadline, "integration ONNX export")
    if not onnx:
        raise ValueError("Integration smoke requires the ONNX score-path parity check")
    import onnx
    import onnxruntime as ort

    model_bytes = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    checkpoint_sha256 = file_hash(checkpoint)
    identity_sha256 = canonical_hash(state["identity"])

    class Portable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor(state["mean"], dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(state["scale"], dtype=torch.float32))

        def forward(self, rows):
            return torch.sigmoid(self.model((rows - self.mean) / self.scale).squeeze(-1))

    portable = Portable().eval()
    onnx_path = output.with_suffix(".onnx")
    onnx_temp = output.with_suffix(".onnx.tmp")
    try:
        torch.onnx.export(
            portable, torch.tensor(features[:min(8, len(features))], dtype=torch.float32),
            str(onnx_temp), input_names=["features"], output_names=["probability"],
            dynamic_axes={"features": {0: "candidates"}, "probability": {0: "candidates"}},
            opset_version=17)
        graph = onnx.load(str(onnx_temp))
        graph_metadata = {item.key: item.value for item in graph.metadata_props}
        graph_metadata.update({
            "artifact_purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
            "selection_enabled": "false",
            "calibration_unavailable_reason": _INTEGRATION_CALIBRATION_UNAVAILABLE_REASON,
            "checkpoint_sha256": checkpoint_sha256,
            "model_sha256": model_sha256,
            "identity_sha256": identity_sha256,
        })
        del graph.metadata_props[:]
        for key, value in sorted(graph_metadata.items()):
            item = graph.metadata_props.add()
            item.key, item.value = key, value
        onnx.save(graph, str(onnx_temp))
        session = ort.InferenceSession(str(onnx_temp), providers=["CPUExecutionProvider"])
        predicted = np.asarray(session.run(None, {"features": features.astype(np.float32)})[0],
                               dtype=np.float64).reshape(-1)
        np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-6)
        onnx_decisions = _integration_decision_signatures(dataset["validation"], predicted, ranker)
        if onnx_decisions != expected_decisions:
            raise AssertionError("ONNX export changes integration-only score order or rejection decisions")
        graph = onnx.load(str(onnx_temp))
        actual_metadata = {item.key: item.value for item in graph.metadata_props}
        for key, value in graph_metadata.items():
            if actual_metadata.get(key) != value:
                raise AssertionError(f"ONNX integration metadata is missing or changed: {key}")
        if onnx_temp.stat().st_size > config.max_checkpoint_bytes:
            raise RuntimeError("ONNX checkpoint exceeds configured size limit")
        _check_deadline(deadline, "integration ONNX parity")
        os.replace(onnx_temp, onnx_path)
    finally:
        onnx_temp.unlink(missing_ok=True)

    atomic_json(output, payload)
    if file_hash(output) != model_sha256:
        raise AssertionError("Integration model JSON hash differs from its prepared export hash")
    receipt = {
        "artifact_purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
        "selection_enabled": False,
        "calibration_unavailable_reason": _INTEGRATION_CALIBRATION_UNAVAILABLE_REASON,
        "checkpoint_sha256": checkpoint_sha256,
        "identity_sha256": identity_sha256,
        "model_sha256": file_hash(output),
        "onnx_sha256": file_hash(onnx_path),
        "eligible_validation_rows": int(len(features)),
        "torch_numpy_max_absolute_error": float(np.max(np.abs(actual - expected))),
        "onnx_torch_max_absolute_error": float(np.max(np.abs(predicted - expected))),
        "torch_numpy_score_parity": True,
        "torch_numpy_order_and_disabled_decision_parity": True,
        "onnx_score_parity": True,
        "onnx_order_and_disabled_decision_parity": True,
        "decision_groups": len(expected_decisions),
        "onnx_checked": True,
        "onnx_metadata": graph_metadata,
        "supervision": {"train": training_summary, "validation": validation_summary},
    }
    atomic_json(output.with_suffix(".export.json"), receipt)
    _check_deadline(deadline, "integration export receipt")
    return receipt


def export(checkpoint: Path, manifest: Path, output: Path, *, cloud_execution: bool,
           onnx: bool = True) -> dict:
    """Export a substantive calibrated v1 artifact; integration checkpoints are refused."""
    return _export_core(checkpoint, manifest, output, cloud_execution=cloud_execution,
                        purpose=_TrainingPurpose.SUBSTANTIVE, onnx=onnx)


def ordering_export(checkpoint: Path, manifest: Path, output: Path, *, cloud_execution: bool,
                    onnx: bool = True) -> dict:
    return _export_core(checkpoint, manifest, output, cloud_execution=cloud_execution,
                        purpose=_TrainingPurpose.EXPERIMENTAL_ORDERING, onnx=onnx)


def confidence_export(checkpoint: Path, manifest: Path, output: Path, *,
                      cloud_execution: bool, onnx: bool = True) -> dict:
    """Export the strict schema-v3 research artifact; other purposes fail closed."""
    return _export_core(
        checkpoint, manifest, output, cloud_execution=cloud_execution,
        purpose=_TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH, onnx=onnx)


def state_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for smoke receipts, including arrays/tensors."""
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    # Avoid importing torch in analytic tests.
    if type(left).__module__.startswith("torch") and hasattr(left, "equal"):
        return type(left) is type(right) and bool(left.equal(right))
    return type(left) is type(right) and bool(left == right)


def _optimizer_evidence(run_directory: Path, checkpoint: Path, *,
                        expected_steps: int | None = None) -> dict[str, Any]:
    torch = __import__("torch")
    selected_state = _read_checkpoint(torch, checkpoint)
    last_checkpoint = run_directory / "last.pt"
    final_state = _read_checkpoint(torch, last_checkpoint if last_checkpoint.is_file() else checkpoint)
    events_path = run_directory / "events.jsonl"
    steps = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("kind") == "step":
                steps.append(event)
    losses = [event.get("loss") for event in steps]
    finite_losses = [value for value in losses
                     if type(value) in (int, float) and math.isfinite(float(value))]
    observed = len(steps)
    if observed <= 0 or observed != final_state["progress"]["step"]:
        raise AssertionError("Run receipt could not verify every completed optimizer step")
    if expected_steps is not None and observed != expected_steps:
        raise AssertionError("Run receipt observed a different optimizer-step count than audited training groups require")
    if len(finite_losses) != observed:
        raise AssertionError("Run receipt found a missing or nonfinite optimizer-step loss")
    initial_sha256 = selected_state.get("initial_model_sha256")
    current_sha256 = _torch_state_sha256(selected_state["model"])
    changed = isinstance(initial_sha256, str) and current_sha256 != initial_sha256
    if not changed:
        raise AssertionError("Observed optimization did not change model parameters")
    selected_parameters_finite = all(bool(torch.isfinite(value).all())
                                     for value in selected_state["model"].values())
    final_parameters_finite = all(bool(torch.isfinite(value).all())
                                  for value in final_state["model"].values())
    if not selected_parameters_finite or not final_parameters_finite:
        raise AssertionError("Run receipt found nonfinite selected or final model parameters")
    evidence = {
        "observed_optimizer_steps": observed,
        "finite_step_loss_count": len(finite_losses),
        "step_loss_min": float(min(finite_losses)),
        "step_loss_max": float(max(finite_losses)),
        "parameters_changed_from_initial": True,
        "initial_model_sha256": initial_sha256,
        "selected_model_sha256": current_sha256,
        "final_model_sha256": _torch_state_sha256(final_state["model"]),
    }
    if expected_steps is not None:
        evidence.update({
            "expected_optimizer_steps": expected_steps,
            "selected_model_parameters_finite": selected_parameters_finite,
            "final_model_parameters_finite": final_parameters_finite,
        })
    return evidence


def smoke(manifest: Path, output: Path, *, cloud_execution: bool, architecture: str = "linear",
          device: str = "cpu", amp: bool = False,
          _purpose: _TrainingPurpose = _TrainingPurpose.SUBSTANTIVE) -> dict:
    """Actual interrupted/resumed optimizer parity, only on the cloud host."""
    cloud_guard(cloud_execution)
    config = TrainConfig(architecture=architecture, epochs=2, batch_size=1, accumulation=1,
                         patience=2, max_seconds=300, device=device, amp=amp)
    whole = _train_core(manifest, output / "uninterrupted", config, cloud_execution=True, purpose=_purpose)
    partial = _train_core(manifest, output / "resumed", config, cloud_execution=True,
                          purpose=_purpose, stop_after_step=1)
    import torch
    partial_state = _read_checkpoint(torch, partial)
    partial_evidence = {"checkpoint_sha256": file_hash(partial),
                        "progress": _confidence_partial_progress(partial_state["progress"])}
    resumed = _train_core(manifest, output / "resumed", config, cloud_execution=True,
                          purpose=_purpose, resume=partial)
    a, b = _read_checkpoint(torch, whole), _read_checkpoint(torch, resumed)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for key in ("optimizer", "scheduler", "scaler", "rng", "mean", "scale", "purpose", "identity", "config"):
        if not state_equal(a[key], b[key]):
            raise AssertionError(f"Interrupted/resumed selected {key} differs")
    final_a = _read_checkpoint(torch, output / "uninterrupted" / "last.pt")
    final_b = _read_checkpoint(torch, output / "resumed" / "last.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng"):
        if not state_equal(final_a[key], final_b[key]):
            raise AssertionError(f"Interrupted/resumed {key} does not match uninterrupted state")
    for key in ("epoch", "cursor", "order", "step", "best_loss", "bad_epochs"):
        if not state_equal(final_a["progress"][key], final_b["progress"][key]):
            raise AssertionError(f"Interrupted/resumed progress {key} differs")
    whole_evidence = _optimizer_evidence(output / "uninterrupted", whole)
    resumed_evidence = _optimizer_evidence(output / "resumed", resumed)
    receipt = {"interrupted_at_step": 1, "resume_exact_model_parity": True,
               "resume_exact_optimizer_scheduler_scaler_rng_parity": True,
               "interruption_kind": "controlled_stop_after_completed_optimizer_step",
               "architecture": config.architecture, "device": device, "amp": amp,
               "optimizer_evidence": {"uninterrupted": whole_evidence, "resumed": resumed_evidence},
               "uninterrupted_checkpoint_sha256": file_hash(whole),
               "resumed_checkpoint_sha256": file_hash(resumed)}
    receipt.update(artifact_purpose=_purpose.value, partial_checkpoint=partial_evidence, test_read=False, dev_read=False)
    receipt["export"] = _export_core(resumed, manifest, output / "smoke-model.json",
                                      cloud_execution=True, purpose=_purpose)
    atomic_json(output / "smoke-receipt.json", receipt)
    return receipt


def ordering_smoke(manifest: Path, output: Path, *, cloud_execution: bool,
                   architecture: str = "linear") -> dict:
    """Two actual cloud epochs, exact interrupted resume and JSON/ONNX parity."""
    cloud_guard(cloud_execution)
    if output.exists():
        raise FileExistsError("Ordering smoke requires a fresh output directory")
    return smoke(manifest, output, cloud_execution=True, architecture=architecture,
                 _purpose=_TrainingPurpose.EXPERIMENTAL_ORDERING)


def _resume_with_partial_snapshot(
        partial_checkpoint: Path, *,
        snapshot_partial: Callable[[Path], dict[str, Any]],
        resume_run: Callable[[Path], Path],
        finish: Callable[[Path, dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Retain partial-checkpoint evidence before resume may overwrite its path."""
    partial_snapshot = snapshot_partial(partial_checkpoint)
    resumed_checkpoint = resume_run(partial_checkpoint)
    return finish(resumed_checkpoint, partial_snapshot)


def _confidence_partial_progress(progress: dict) -> dict:
    """JSON receipt summary for interruption before the first validation epoch.

    The resumable binary checkpoint retains all progress, including the initial
    best_loss=inf sentinel. It is unavailable evidence, not a measured loss, and
    must not enter the strict JSON smoke receipt.
    """
    result = {name: progress[name] for name in ("epoch", "cursor", "step")}
    if (any(type(value) is not int for value in result.values()) or
            result["epoch"] < 0 or result["cursor"] <= 0 or result["step"] != 1):
        raise AssertionError("Confidence partial checkpoint progress is malformed")
    return result


def integration_smoke(manifest: Path, output: Path, *, cloud_execution: bool) -> dict:
    """Run a purpose-limited linear score-path smoke on approved cloud CPU only."""
    cloud_guard(cloud_execution)
    if output.exists():
        raise FileExistsError("Integration smoke requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + _INTEGRATION_SMOKE_DEADLINE_SECONDS
    config = TrainConfig(architecture="linear", seed=42, epochs=2, batch_size=1,
                         accumulation=1, patience=2,
                         max_seconds=_INTEGRATION_SMOKE_DEADLINE_SECONDS,
                         amp=False, device="cpu")
    audited = _load_dataset(manifest, purpose=_TrainingPurpose.INTEGRATION_SMOKE)
    train_summary = _integration_supervision_summary(audited["train"],
                                                    audited["feature_names"], config)
    _integration_supervision_summary(audited["validation"], audited["feature_names"], config)
    expected_steps = config.epochs * train_summary["eligible_groups"]
    if expected_steps <= 0:
        raise ValueError("Integration smoke requires at least one audited eligible training group")
    _check_deadline(deadline, "audited group count")
    whole = _train_core(manifest, output / "uninterrupted", config,
                        cloud_execution=True, purpose=_TrainingPurpose.INTEGRATION_SMOKE,
                        deadline=deadline)
    _check_deadline(deadline, "uninterrupted training")
    partial = _train_core(manifest, output / "resumed", config,
                          cloud_execution=True, purpose=_TrainingPurpose.INTEGRATION_SMOKE,
                          stop_after_step=1, deadline=deadline)
    _check_deadline(deadline, "controlled interruption")
    import torch

    def snapshot_partial(path: Path) -> dict[str, Any]:
        state = _read_checkpoint(torch, path)
        _require_checkpoint_purpose(state, _TrainingPurpose.INTEGRATION_SMOKE,
                                    operation="resume")
        progress = state.get("progress")
        if not isinstance(progress, dict) or progress.get("step") != 1 or progress.get("cursor") != 1:
            raise AssertionError("Controlled interruption did not stop after exactly one completed optimizer step")
        return {
            "checkpoint_sha256": file_hash(path),
            "purpose": state["purpose"],
            "identity_sha256": canonical_hash(state["identity"]),
            "progress": {name: progress[name] for name in ("epoch", "cursor", "step")},
        }

    def resume_run(path: Path) -> Path:
        result = _train_core(manifest, output / "resumed", config,
                             cloud_execution=True, purpose=_TrainingPurpose.INTEGRATION_SMOKE,
                             resume=path, deadline=deadline)
        _check_deadline(deadline, "resumed training")
        return result

    def finish(resumed: Path, partial_snapshot: dict[str, Any]) -> dict[str, Any]:
        selected_whole = _read_checkpoint(torch, whole)
        selected_resumed = _read_checkpoint(torch, resumed)
        for name in selected_whole["model"]:
            torch.testing.assert_close(selected_whole["model"][name], selected_resumed["model"][name], rtol=0, atol=0)
        final_whole = _read_checkpoint(torch, output / "uninterrupted" / "last.pt")
        final_resumed = _read_checkpoint(torch, output / "resumed" / "last.pt")
        for key in ("model", "optimizer", "scheduler", "scaler", "rng"):
            if not state_equal(final_whole[key], final_resumed[key]):
                raise AssertionError(f"Interrupted/resumed final {key} does not match uninterrupted state")
        for key in ("epoch", "cursor", "order", "step", "best_loss", "bad_epochs"):
            if not state_equal(final_whole["progress"][key], final_resumed["progress"][key]):
                raise AssertionError(f"Interrupted/resumed progress {key} differs")
        for key in ("purpose", "identity", "config", "initial_model_sha256"):
            if not state_equal(final_whole[key], final_resumed[key]):
                raise AssertionError(f"Interrupted/resumed checkpoint {key} differs")
        if (partial_snapshot["purpose"] != _TrainingPurpose.INTEGRATION_SMOKE.value or
                partial_snapshot["identity_sha256"] != canonical_hash(final_resumed["identity"])):
            raise AssertionError("Controlled interruption checkpoint identity changed across resume")
        for name, state in (("uninterrupted", final_whole), ("resumed", final_resumed)):
            if state["progress"]["epoch"] != config.epochs:
                raise AssertionError(f"{name} run did not complete both configured epochs")
            if state["progress"]["step"] != expected_steps:
                raise AssertionError(f"{name} run did not complete every audited training-group step")

        _check_deadline(deadline, "optimizer evidence collection")
        whole_evidence = _optimizer_evidence(output / "uninterrupted", whole,
                                             expected_steps=expected_steps)
        resumed_evidence = _optimizer_evidence(output / "resumed", resumed,
                                               expected_steps=expected_steps)
        if whole_evidence["observed_optimizer_steps"] != resumed_evidence["observed_optimizer_steps"]:
            raise AssertionError("Interrupted/resumed optimizer-step counts differ")
        if whole_evidence["step_loss_min"] != resumed_evidence["step_loss_min"] or whole_evidence["step_loss_max"] != resumed_evidence["step_loss_max"]:
            raise AssertionError("Interrupted/resumed observed step-loss summaries differ")

        export_receipt = _export_core(resumed, manifest, output / "integration-model.json",
                                      cloud_execution=True,
                                      purpose=_TrainingPurpose.INTEGRATION_SMOKE,
                                      onnx=True, deadline=deadline)
        _check_deadline(deadline, "integration export")
        receipt = {
            "artifact_purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
            "selection_enabled": False,
            "interrupted_at_step": 1,
            "partial_checkpoint_sha256": partial_snapshot["checkpoint_sha256"],
            "partial_checkpoint_progress": partial_snapshot["progress"],
            "resume_exact_selected_model_parity": True,
            "resume_exact_final_model_optimizer_scheduler_scaler_rng_progress_parity": True,
            "interruption_kind": "controlled_stop_after_completed_optimizer_step",
            "architecture": config.architecture,
            "device": config.device,
            "amp": config.amp,
            "seed": config.seed,
            "epochs": config.epochs,
            "outer_deadline_seconds": _INTEGRATION_SMOKE_DEADLINE_SECONDS,
            "audited_training_groups": train_summary["eligible_groups"],
            "expected_optimizer_steps": expected_steps,
            "optimizer_evidence": {"uninterrupted": whole_evidence, "resumed": resumed_evidence},
            "uninterrupted_checkpoint_sha256": file_hash(whole),
            "resumed_checkpoint_sha256": file_hash(resumed),
            "export": export_receipt,
            "test_read": False,
        }
        atomic_json(output / "integration-smoke-receipt.json", receipt)
        _check_deadline(deadline, "integration receipt")
        return receipt

    return _resume_with_partial_snapshot(partial, snapshot_partial=snapshot_partial,
                                         resume_run=resume_run, finish=finish)


def confidence_smoke(manifest: Path, output: Path, *, cloud_execution: bool,
                     architecture: str = "linear") -> dict:
    """Two-epoch research-purpose CPU resume/export parity smoke."""
    cloud_guard(cloud_execution)
    if output.exists():
        raise FileExistsError("Confidence smoke requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=False)
    config = TrainConfig(
        architecture=architecture, seed=42, epochs=2, batch_size=8,
        accumulation=4, learning_rate=.001, weight_decay=.0001,
        pairwise_weight=.5, patience=6, min_delta=.00001, max_seconds=300.,
        max_output_bytes=64 * 1024 * 1024,
        max_checkpoint_bytes=16 * 1024 * 1024,
        amp=False, device="cpu")
    _validate_research_config(config, smoke=True)
    purpose = _TrainingPurpose.CONFIDENCE_REJECTION_RESEARCH
    audited = _load_dataset(manifest, purpose=purpose)
    eligible_groups = sum(bool([candidate for candidate in group["candidates"]
                                if candidate.get("eligible", True)])
                          for group in audited["train"])
    steps_per_epoch = math.ceil(eligible_groups /
                                (config.batch_size * config.accumulation))
    expected_steps = config.epochs * steps_per_epoch
    if expected_steps <= 0:
        raise ValueError("Confidence smoke requires eligible audited training groups")
    whole = _train_core(manifest, output / "uninterrupted", config,
                        cloud_execution=True, purpose=purpose)
    partial = _train_core(manifest, output / "resumed", config,
                          cloud_execution=True, purpose=purpose, stop_after_step=1)
    import torch

    def snapshot_partial(path: Path) -> dict[str, Any]:
        state = _read_checkpoint(torch, path)
        _require_checkpoint_purpose(state, purpose, operation="resume")
        if state["progress"]["step"] != 1:
            raise AssertionError("Confidence smoke did not stop after one optimizer step")
        return {"checkpoint_sha256": file_hash(path),
                "progress": _confidence_partial_progress(state["progress"]),
                "identity_sha256": canonical_hash(state["identity"])}

    def resume_run(path: Path) -> Path:
        return _train_core(manifest, output / "resumed", config,
                           cloud_execution=True, purpose=purpose, resume=path)

    def finish(resumed: Path, partial_snapshot: dict[str, Any]) -> dict[str, Any]:
        selected_a, selected_b = _read_checkpoint(torch, whole), _read_checkpoint(torch, resumed)
        for key in ("model", "optimizer", "scheduler", "scaler", "rng", "mean", "scale",
                    "purpose", "identity", "config", "initial_model_sha256"):
            if not state_equal(selected_a[key], selected_b[key]):
                raise AssertionError(f"Confidence interrupted/resumed selected {key} differs")
        for key in ("epoch", "cursor", "order", "step", "best_loss", "bad_epochs"):
            if not state_equal(selected_a["progress"][key], selected_b["progress"][key]):
                raise AssertionError(f"Confidence interrupted/resumed selected progress {key} differs")
        final_a = _read_checkpoint(torch, output / "uninterrupted" / "last.pt")
        final_b = _read_checkpoint(torch, output / "resumed" / "last.pt")
        for key in ("model", "optimizer", "scheduler", "scaler", "rng"):
            if not state_equal(final_a[key], final_b[key]):
                raise AssertionError(f"Confidence interrupted/resumed final {key} differs")
        for key in ("epoch", "cursor", "order", "step", "best_loss", "bad_epochs"):
            if not state_equal(final_a["progress"][key], final_b["progress"][key]):
                raise AssertionError(f"Confidence interrupted/resumed final progress {key} differs")
        for key in ("purpose", "identity", "config", "initial_model_sha256"):
            if not state_equal(final_a[key], final_b[key]):
                raise AssertionError(f"Confidence interrupted/resumed checkpoint {key} differs")
        if partial_snapshot["identity_sha256"] != canonical_hash(final_b["identity"]):
            raise AssertionError("Confidence partial identity changed during resume")
        for name, state in (("uninterrupted", final_a), ("resumed", final_b)):
            if state["progress"]["epoch"] != config.epochs or state["progress"]["step"] != expected_steps:
                raise AssertionError(f"Confidence {name} smoke did not complete every configured step")
        whole_evidence = _optimizer_evidence(output / "uninterrupted", whole,
                                             expected_steps=expected_steps)
        resumed_evidence = _optimizer_evidence(output / "resumed", resumed,
                                               expected_steps=expected_steps)
        whole_completion = _validate_research_completion(torch, whole, selected_a)
        resumed_completion = _validate_research_completion(torch, resumed, selected_b)
        if (whole_completion["checkpoint_selection"] != resumed_completion["checkpoint_selection"] or
                whole_completion["epochs_completed"] != config.epochs or
                resumed_completion["epochs_completed"] != config.epochs):
            raise AssertionError("Confidence smoke completion receipts differ")
        receipt = {
            "artifact_purpose": purpose.value, "policy": "legacy_veto_v1",
            "research_only": True, "smoke_config_sha256": canonical_hash(asdict(config)),
            "partial_checkpoint_sha256": partial_snapshot["checkpoint_sha256"],
            "partial_checkpoint_progress": partial_snapshot["progress"],
            "resume_exact_selected_model_parity": True,
            "resume_exact_selected_model_optimizer_scheduler_scaler_rng_progress_parity": True,
            "resume_exact_final_model_optimizer_scheduler_scaler_rng_progress_parity": True,
            "audited_training_groups": eligible_groups,
            "expected_optimizer_steps": expected_steps,
            "optimizer_evidence": {"uninterrupted": whole_evidence,
                                   "resumed": resumed_evidence},
            "completion": {"uninterrupted": whole_completion,
                           "resumed": resumed_completion},
            "uninterrupted_checkpoint_sha256": file_hash(whole),
            "resumed_checkpoint_sha256": file_hash(resumed),
            "export": confidence_export(
                resumed, manifest, output / "confidence-smoke-model.json",
                cloud_execution=True, onnx=True),
            "development_fitting": True, "test_read": False,
        }
        atomic_json(output / "confidence-smoke-receipt.json", receipt)
        return receipt

    return _resume_with_partial_snapshot(
        partial, snapshot_partial=snapshot_partial,
        resume_run=resume_run, finish=finish)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "resume", "export", "smoke", "integration-smoke",
                 "confidence-train", "confidence-resume", "confidence-export",
                 "confidence-smoke", "ordering-train", "ordering-resume", "ordering-export", "ordering-smoke"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--cloud-execution", action="store_true")
        if name in ("train", "resume", "confidence-train", "confidence-resume", "ordering-train", "ordering-resume"):
            command.add_argument("--config", type=Path)
            command.add_argument("--architecture", choices=("linear", "mlp64_32"), default="linear")
            if name in ("resume", "confidence-resume", "ordering-resume"):
                command.add_argument("--checkpoint", type=Path, required=True)
        if name in ("export", "confidence-export", "ordering-export"):
            command.add_argument("--checkpoint", type=Path, required=True)
        if name == "smoke":
            command.add_argument("--architecture", choices=("linear", "mlp64_32"), default="linear")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--amp", action="store_true")
        if name in ("confidence-smoke", "ordering-smoke"):
            command.add_argument("--architecture", choices=("linear", "mlp64_32"), default="linear")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = smoke(args.manifest, args.output, cloud_execution=args.cloud_execution,
                       architecture=args.architecture, device=args.device, amp=args.amp)
    elif args.command == "integration-smoke":
        result = integration_smoke(args.manifest, args.output,
                                   cloud_execution=args.cloud_execution)
    elif args.command == "confidence-smoke":
        result = confidence_smoke(args.manifest, args.output,
                                  cloud_execution=args.cloud_execution,
                                  architecture=args.architecture)
    elif args.command == "ordering-smoke":
        result = ordering_smoke(args.manifest, args.output,
                                cloud_execution=args.cloud_execution, architecture=args.architecture)
    elif args.command == "confidence-export":
        result = confidence_export(args.checkpoint, args.manifest, args.output,
                                   cloud_execution=args.cloud_execution)
    elif args.command == "ordering-export":
        result = ordering_export(args.checkpoint, args.manifest, args.output,
                                 cloud_execution=args.cloud_execution)
    elif args.command == "export":
        result = export(args.checkpoint, args.manifest, args.output, cloud_execution=args.cloud_execution)
    else:
        config = TrainConfig(**json.loads(args.config.read_text())) if args.config else TrainConfig(architecture=args.architecture)
        if args.command.startswith("ordering-"):
            result = str(ordering_train(
                args.manifest, args.output, config, cloud_execution=args.cloud_execution,
                resume=args.checkpoint if args.command == "ordering-resume" else None))
        elif args.command.startswith("confidence-"):
            result = str(confidence_train(
                args.manifest, args.output, config, cloud_execution=args.cloud_execution,
                resume=args.checkpoint if args.command == "confidence-resume" else None))
        else:
            result = str(train(args.manifest, args.output, config, cloud_execution=args.cloud_execution,
                               resume=args.checkpoint if args.command == "resume" else None))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
