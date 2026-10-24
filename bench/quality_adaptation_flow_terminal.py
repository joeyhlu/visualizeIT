"""Bounded DPT flow-row continuation with a frozen native confidence row.

Training consumes the audited frozen-token dataset and updates only rows 0:2 of
the pinned DPT output convolution. The terminal confidence row remains byte
identical to the official checkpoint and is checked against cached features
before a run can be marked complete or exported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np

from . import quality_adaptation_data as data_module
from . import quality_adaptation_feature_union as feature_union_module
from . import quality_adaptation_runtime as runtime_module
from . import quality_adaptation_train as train_module
from .quality_adaptation_data import (HOOKS, SIZE, canonical_hash, data_report,
                                      flow_metrics, load_dataset, object_summary,
                                      read_json, sha256, visibility_metrics)
from .quality_adaptation_runtime import (GOTRACK_CHECKPOINT, LOCK, atomic_stream,
                                         check_resume, cloud_guard, load_head,
                                         runtime_versions, save_checkpoint,
                                         training_state_digest, verify_sources,
                                         write_json)
from .quality_adaptation_train import (TrainConfig, adaptation_loss, epoch_batches,
                                       read_config, tensor_batch,
                                       validate_feature_receipt)


FLOW_LAYER_INDEX = 4
FLOW_CHANNELS = 256
OUTPUT_CHANNELS = 3
TRAINABLE_PARAMETER_COUNT = 2 * FLOW_CHANNELS + 2
CONFIDENCE_THRESHOLD = np.float32(0.3)
CONFIDENCE_TOLERANCE = 1e-6
CUDA_BOOTSTRAP_SHA256 = "285fa7749dfab8f95cd9bf41a3bce22ccdad3731e1bcb5b4553f1253015fbf8e"
CUDA_POLICY_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VISUALIZEIT_CUDA_BOOTSTRAP_SHA256": CUDA_BOOTSTRAP_SHA256,
}


def continuation_config(base: TrainConfig) -> TrainConfig:
    """Pin the experiment while retaining the caller's runtime and size caps."""
    fixed = replace(base, seed=521, epochs=3, batch_size=1,
                    learning_rate=1e-5, weight_decay=1e-4,
                    visibility_weight=0.0, gradient_clip=1.0,
                    checkpoint_every_steps=8, amp=False)
    fixed.validate()
    return fixed


def policy_document(ds, cfg: TrainConfig, versions: dict, *, feature_union=False) -> dict:
    """Hash the changed and reused source boundary plus fixed training policy."""
    files = {
        "flow_terminal": Path(__file__),
        "train": Path(train_module.__file__),
        "runtime": Path(runtime_module.__file__),
        "data": Path(data_module.__file__),
    }
    policy = {
        "schema": "correspondence_flow_terminal_policy_v1",
        "dataset_sha256": ds.identity,
        "config_sha256": canonical_hash(asdict(cfg)),
        "source_lock_sha256": sha256(LOCK),
        "base_checkpoint_sha256": GOTRACK_CHECKPOINT,
        "source_files_sha256": {name: sha256(path) for name, path in files.items()},
        "runtime_versions": dict(versions),
        "terminal": {
            "module_index": FLOW_LAYER_INDEX,
            "input_channels": FLOW_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
            "trainable_weight_rows": [0, 1],
            "frozen_confidence_row": 2,
            "trainable_parameters": TRAINABLE_PARAMETER_COUNT,
            "confidence_activation": "native DPT sigmoid",
        },
        "loss": "mean_per_frame_visible_smooth_l1_beta_1px; visibility_weight=0",
        "selection": "final_epoch_only; one post-training train+validation evaluation",
        "runtime_policy": {
            "requested_device": cfg.device,
            "cuda_bootstrap_sha256": CUDA_BOOTSTRAP_SHA256 if cfg.device == "cuda" else None,
            "required_environment": CUDA_POLICY_ENV if cfg.device == "cuda" else {},
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "deterministic_algorithms": True,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "autocast_amp": False,
            "automatic_device_upgrade": False,
        },
    }
    if feature_union:
        provenance = getattr(ds, "document", {}).get("provenance", {})
        recipe_sha = provenance.get("feature_recipe_sha256")
        if (not isinstance(recipe_sha, str) or len(recipe_sha) != 64
                or any(char not in "0123456789abcdef" for char in recipe_sha)):
            raise ValueError("feature-union policy requires its canonical recipe identity")
        files["feature_union"] = Path(feature_union_module.__file__)
        policy["source_files_sha256"] = {name: sha256(path) for name, path in files.items()}
        policy["feature_union"] = {
            "enabled": True,
            "recipe_sha256": recipe_sha,
            "module_sha256": sha256(Path(feature_union_module.__file__)),
        }
    return policy


def assert_runtime_environment(cfg: TrainConfig) -> None:
    """Require the reviewed bootstrap and deterministic environment for CUDA."""
    if cfg.device != "cuda":
        return
    mismatches = {key: (os.environ.get(key), wanted)
                  for key, wanted in CUDA_POLICY_ENV.items()
                  if os.environ.get(key) != wanted}
    if mismatches:
        raise RuntimeError(f"CUDA runtime environment differs from pinned policy: {mismatches}")


def configure_runtime_policy(cfg: TrainConfig, torch) -> dict:
    """Apply and attest actual numerical settings before loading a model."""
    assert_runtime_environment(cfg)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("Torch inter-op thread count already initialized outside pinned policy")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable; no automatic resource upgrade")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA flow-terminal runtime requires exactly one visible GPU")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
            "uuid": str(getattr(properties, "uuid", "")) or None,
        }
        cuda_version = torch.version.cuda
    else:
        gpu = None
        cuda_version = None
    actual = {
        "requested_device": cfg.device,
        "selected_device": "cuda:0" if cfg.device == "cuda" else "cpu",
        "environment": ({key: os.environ.get(key) for key in CUDA_POLICY_ENV}
                        if cfg.device == "cuda" else {}),
        "cuda_bootstrap_sha256": os.environ.get("VISUALIZEIT_CUDA_BOOTSTRAP_SHA256") if cfg.device == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cfg.device == "cuda" else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS") if cfg.device == "cuda" else None,
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS") if cfg.device == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_version,
        "gpu": gpu,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "autocast_amp": False,
    }
    expected = policy_document(SimpleNamespace(identity="0" * 64), cfg, {})["runtime_policy"]
    if (actual["requested_device"] != expected["requested_device"]
            or actual["selected_device"] != ("cuda:0" if cfg.device == "cuda" else "cpu")
            or actual["environment"] != expected["required_environment"]
            or actual["cuda_bootstrap_sha256"] != expected["cuda_bootstrap_sha256"]
            or actual["cublas_workspace_config"] != (":4096:8" if cfg.device == "cuda" else None)
            or actual["threads"] != expected["torch_num_threads"]
            or actual["interop_threads"] != expected["torch_num_interop_threads"]
            or actual["deterministic_algorithms"] is not True
            or actual["matmul_allow_tf32"] is not False
            or actual["cudnn_allow_tf32"] is not False
            or actual["cudnn_benchmark"] is not False
            or actual["cudnn_deterministic"] is not True):
        raise RuntimeError("actual Torch numerical settings differ from the pinned runtime policy")
    return actual


def preflight(manifest: Path, cfg: TrainConfig, *, feature_union=False) -> dict:
    """Validate provenance and report train-only EDA without opening validation arrays."""
    if type(feature_union) is not bool:
        raise ValueError("feature_union must be an explicit boolean")
    cfg = continuation_config(cfg)
    assert_runtime_environment(cfg)
    ds = load_dataset(manifest, inspect_arrays=False)
    union_receipt = None
    if feature_union:
        union_receipt = feature_union_module.validate_union_receipt(ds)
    else:
        # The ordinary receipt validator rejects a union recipe's distinct
        # schema before any payload is read, preserving its normal source checks.
        validate_feature_receipt(ds)
    source = verify_sources()
    # data_report reads only TRAIN payloads; held-out inventory counts are metadata.
    report = data_report(ds)
    policy = policy_document(ds, cfg, runtime_versions(), feature_union=feature_union)
    result = {
        "schema": "correspondence_flow_terminal_preflight_v1",
        "config": asdict(cfg),
        "config_sha256": canonical_hash(asdict(cfg)),
        **source,
        "policy_sha256": canonical_hash(policy),
        "policy": policy,
        "data": report,
        "trainable": "DPT output convolution weight/bias rows 0:2 only",
        "trainable_parameters": TRAINABLE_PARAMETER_COUNT,
        "loss": policy["loss"],
        "selection": policy["selection"],
        "train_arrays_opened_for_eda": True,
        "validation_arrays_opened": False,
        "test_payload_reads": 0,
        "cloud_training_validated": False,
    }
    if union_receipt is not None:
        result["feature_union_receipt"] = union_receipt
    return result


def _flow_terminal_class(torch):
    class FlowTerminal(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.flow_weight = torch.nn.Parameter(original.weight[:2].detach().clone())
            self.flow_bias = torch.nn.Parameter(original.bias[:2].detach().clone())
            self.register_buffer("confidence_weight", original.weight[2:3].detach().clone())
            self.register_buffer("confidence_bias", original.bias[2:3].detach().clone())
            self.stride = original.stride
            self.padding = original.padding
            self.dilation = original.dilation
            self.groups = original.groups
            self.padding_mode = original.padding_mode
            self.in_channels = original.in_channels
            self.out_channels = original.out_channels
            self.kernel_size = original.kernel_size
            self._flow_terminal_v1 = True

        def forward(self, value):
            # The original pinned layer uses zero padding. Keep a single fused
            # three-row convolution so confidence sees precisely the same path.
            if self.padding_mode != "zeros":
                raise ValueError("flow terminal requires the pinned zero-padding convolution")
            weight = torch.cat((self.flow_weight, self.confidence_weight), dim=0)
            bias = torch.cat((self.flow_bias, self.confidence_bias), dim=0)
            return torch.nn.functional.conv2d(
                value, weight, bias, self.stride, self.padding,
                self.dilation, self.groups)

    return FlowTerminal


def attach_flow_terminal(head, torch=None):
    """Replace the DPT output sequence's layer 4 with flow rows and frozen confidence."""
    if torch is None:
        import torch as torch_module
        torch = torch_module
    container = getattr(head, "head", None)
    if container is None:
        raise ValueError("DPTHead must expose its pinned output sequence as .head")
    original = container[FLOW_LAYER_INDEX]
    if (not isinstance(original, torch.nn.Conv2d)
            or original.in_channels != FLOW_CHANNELS
            or original.out_channels != OUTPUT_CHANNELS
            or tuple(original.kernel_size) != (1, 1)
            or original.bias is None):
        raise ValueError("pinned DPT head[4] must be biased Conv2d(256, 3, 1)")
    if original.padding_mode != "zeros":
        raise ValueError("pinned DPT head[4] must use zero padding")
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    terminal = _flow_terminal_class(torch)(original)
    container[FLOW_LAYER_INDEX] = terminal
    for parameter in terminal.parameters():
        parameter.requires_grad_(True)
    head.eval()
    trainable = [p for p in head.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("flow terminal trainable parameter count differs from 514")
    if len(trainable) != 2:
        raise ValueError("only the two flow-row parameters may be trainable")
    return terminal


def ordinary_head_state(head) -> dict:
    """Reconstruct strict upstream DPT state keys from the wrapped terminal."""
    container = getattr(head, "head", None)
    if container is None:
        raise ValueError("DPTHead must expose its pinned output sequence as .head")
    terminal = container[FLOW_LAYER_INDEX]
    if not getattr(terminal, "_flow_terminal_v1", False):
        raise ValueError("head does not have the flow terminal wrapper")
    prefix = f"head.{FLOW_LAYER_INDEX}."
    state = head.state_dict()
    ordinary = {key: value.detach().cpu().clone()
                for key, value in state.items() if not key.startswith(prefix)}
    expected_terminal_keys = {
        prefix + "flow_weight", prefix + "flow_bias",
        prefix + "confidence_weight", prefix + "confidence_bias",
    }
    if {key for key in state if key.startswith(prefix)} != expected_terminal_keys:
        raise ValueError("flow terminal state keys differ from the versioned wrapper")
    ordinary[prefix + "weight"] = torch_cat_cpu(terminal, "weight")
    ordinary[prefix + "bias"] = torch_cat_cpu(terminal, "bias")
    return ordinary


def torch_cat_cpu(terminal, name: str):
    import torch
    if name == "weight":
        return torch.cat((terminal.flow_weight, terminal.confidence_weight), dim=0).detach().cpu().clone()
    if name == "bias":
        return torch.cat((terminal.flow_bias, terminal.confidence_bias), dim=0).detach().cpu().clone()
    raise ValueError("unknown ordinary terminal tensor")


def _cpu_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _bitwise_equal(left, right) -> bool:
    a, b = _cpu_array(left), _cpu_array(right)
    return (a.shape == b.shape and a.dtype == b.dtype
            and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def validate_frozen_state(saved: dict, official: dict) -> None:
    """Reject a checkpoint that changed any frozen tensor or confidence row."""
    if not isinstance(saved, dict) or not isinstance(official, dict) or set(saved) != set(official):
        raise ValueError("checkpoint ordinary head keys differ from official DPT head")
    weight_key = f"head.{FLOW_LAYER_INDEX}.weight"
    bias_key = f"head.{FLOW_LAYER_INDEX}.bias"
    if weight_key not in official or bias_key not in official:
        raise ValueError("official DPT head lacks the pinned terminal tensors")
    for key in official:
        a, b = _cpu_array(saved[key]), _cpu_array(official[key])
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"checkpoint ordinary head shape/dtype mismatch: {key}")
        if key == weight_key:
            if not _bitwise_equal(a[2:], b[2:]):
                raise ValueError("checkpoint changed frozen confidence convolution weights")
        elif key == bias_key:
            if not _bitwise_equal(a[2:], b[2:]):
                raise ValueError("checkpoint changed frozen confidence convolution bias")
        elif not _bitwise_equal(a, b):
            raise ValueError(f"checkpoint changed frozen official head tensor: {key}")


def frozen_state(ordinary: dict) -> dict:
    """Select every frozen tensor, including only row 2 of terminal weight/bias."""
    weight_key = f"head.{FLOW_LAYER_INDEX}.weight"
    bias_key = f"head.{FLOW_LAYER_INDEX}.bias"
    if weight_key not in ordinary or bias_key not in ordinary:
        raise ValueError("ordinary DPT state lacks terminal weight/bias")
    result = {key: value for key, value in ordinary.items()
              if key not in (weight_key, bias_key)}
    result[weight_key] = ordinary[weight_key][2:]
    result[bias_key] = ordinary[bias_key][2:]
    return result


def restore_flow_state(head, saved: dict, official: dict) -> None:
    """Validate official frozen tensors, then restore only the first two rows."""
    validate_frozen_state(saved, official)
    terminal = head.head[FLOW_LAYER_INDEX]
    import torch
    with torch.no_grad():
        terminal.flow_weight.copy_(saved[f"head.{FLOW_LAYER_INDEX}.weight"][:2].to(
            device=terminal.flow_weight.device, dtype=terminal.flow_weight.dtype))
        terminal.flow_bias.copy_(saved[f"head.{FLOW_LAYER_INDEX}.bias"][:2].to(
            device=terminal.flow_bias.device, dtype=terminal.flow_bias.dtype))


def validate_frozen_identity(saved: dict, official: dict, claimed_digest: str) -> None:
    validate_frozen_state(saved, official)
    saved_digest = training_state_digest(frozen_state(saved))
    official_digest = training_state_digest(frozen_state(official))
    if saved_digest != official_digest or claimed_digest != official_digest:
        raise ValueError("checkpoint frozen-state digest differs from official head")


def confidence_parity(actual, baseline) -> dict:
    """Compare native float32 probabilities and the consumer's strict >0.3 mask."""
    actual = np.asarray(actual)
    baseline = np.asarray(baseline)
    reasons = []
    if actual.shape != (SIZE, SIZE) or baseline.shape != (SIZE, SIZE):
        reasons.append("shape_mismatch")
    if actual.dtype != np.dtype("float32") or baseline.dtype != np.dtype("float32"):
        reasons.append("native_float32_required")
    if not np.isfinite(actual).all() or not np.isfinite(baseline).all():
        reasons.append("nonfinite_confidence")
    if (actual.shape == baseline.shape and np.isfinite(actual).all()
            and np.isfinite(baseline).all()):
        maximum = float(np.max(np.abs(actual - baseline)))
        decision_changes = int(np.count_nonzero((actual > CONFIDENCE_THRESHOLD)
                                                != (baseline > CONFIDENCE_THRESHOLD)))
    else:
        maximum, decision_changes = None, None
    if maximum is not None and maximum > CONFIDENCE_TOLERANCE:
        reasons.append("confidence_difference_exceeds_tolerance")
    if decision_changes not in (None, 0):
        reasons.append("native_threshold_decisions_changed")
    if (np.isfinite(actual).all() and np.any((actual < 0) | (actual > 1))):
        reasons.append("confidence_outside_probability_range")
    return {
        "passed": not reasons,
        "max_abs_difference": maximum,
        "decision_changes": decision_changes,
        "pixels": int(actual.size) if actual.ndim == 2 else None,
        "threshold": float(CONFIDENCE_THRESHOLD),
        "tolerance": CONFIDENCE_TOLERANCE,
        "reasons": reasons,
    }


def validate_resume_cursor(state: dict, rows: list[dict], cfg: TrainConfig) -> None:
    """Bind the serialized epoch/cursor/step count to the deterministic order."""
    epoch, cursor, steps = state.get("epoch"), state.get("batch_cursor"), state.get("steps")
    if any(type(x) is not int or x < 0 for x in (epoch, cursor, steps)):
        raise ValueError("checkpoint epoch/cursor/steps are malformed")
    if epoch > cfg.epochs:
        raise ValueError("checkpoint epoch exceeds configured continuation")
    expected = 0
    for prior_epoch in range(epoch):
        expected += len(epoch_batches(rows, prior_epoch, cfg))
    if epoch < cfg.epochs:
        current_batches = epoch_batches(rows, epoch, cfg)
        if cursor > len(current_batches):
            raise ValueError("checkpoint cursor exceeds deterministic epoch")
        expected += cursor
    elif cursor != 0:
        raise ValueError("completed epoch cursor must be zero")
    if steps != expected or steps > cfg.max_steps:
        raise ValueError("checkpoint steps do not match deterministic cursor or configured cap")
    if state.get("bad_epochs") != 0:
        raise ValueError("flow-terminal policy does not use early stopping")


def validate_optimizer_state(value: dict, cfg: TrainConfig, steps: int) -> None:
    """Reject resumed AdamW settings/slots that differ from the fixed recipe."""
    if not isinstance(value, dict) or set(value) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state has malformed top-level keys")
    groups = value.get("param_groups")
    slots = value.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(slots, dict):
        raise ValueError("checkpoint optimizer must contain one AdamW group")
    group = groups[0]
    expected = {
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    if not isinstance(group, dict):
        raise ValueError("checkpoint optimizer parameter group is malformed")
    for key, wanted in expected.items():
        actual = group.get(key)
        if key in ("lr", "weight_decay", "eps"):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or float(actual) != float(wanted):
                raise ValueError(f"checkpoint AdamW {key} differs from fixed continuation config")
        elif key == "betas":
            if not isinstance(actual, (tuple, list)) or tuple(actual) != wanted:
                raise ValueError("checkpoint AdamW betas differ from fixed defaults")
        elif actual != wanted:
            raise ValueError(f"checkpoint AdamW {key} differs from fixed defaults")
    params = group.get("params")
    if params != [0, 1]:
        raise ValueError("checkpoint AdamW must reference exactly the two flow tensors")
    if steps == 0:
        if slots:
            raise ValueError("zero-step checkpoint unexpectedly has AdamW slots")
        return
    if set(slots) != set(params):
        raise ValueError("checkpoint AdamW slots differ from the two flow tensors")
    for parameter_id in params:
        slot = slots[parameter_id]
        if not isinstance(slot, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= set(slot):
            raise ValueError("checkpoint AdamW moment state is incomplete")
        step = slot["step"]
        if hasattr(step, "detach"):
            step = step.detach().cpu().item()
        if (isinstance(step, bool) or not isinstance(step, (int, float))
                or not math.isfinite(float(step)) or float(step) != float(steps)):
            raise ValueError("checkpoint AdamW moment step differs from global optimizer updates")


def checkpoint_integrity_digest(state: dict) -> str:
    """Detect accidental edits to head, optimizer, RNG, cursor, or receipts."""
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a mapping")
    return training_state_digest({key: value for key, value in state.items()
                                  if key != "integrity_sha256"})


def validate_checkpoint_integrity(state: dict) -> None:
    claimed = state.get("integrity_sha256")
    if not isinstance(claimed, str) or claimed != checkpoint_integrity_digest(state):
        raise ValueError("checkpoint integrity digest mismatch")


class BudgetStop(RuntimeError):
    pass


def _model_outputs(head, tokens, torch, batch_size: int):
    flow, confidence = head(tokens, (SIZE, SIZE))
    if (tuple(flow.shape) != (batch_size, 2, SIZE, SIZE)
            or tuple(confidence.shape) != (batch_size, SIZE, SIZE)):
        raise ValueError("DPT output shape differs from flow/confidence serving contract")
    if flow.dtype != torch.float32 or confidence.dtype != torch.float32:
        raise ValueError("flow-terminal path requires native float32 outputs")
    if not torch.isfinite(flow).all() or not torch.isfinite(confidence).all():
        raise ValueError("nonfinite DPT flow/confidence output")
    return flow, confidence


def _metric_rows(rows: list[dict], *, seed: int) -> dict:
    if not rows:
        raise ValueError("no rows available for final evaluation")
    return object_summary(rows, seed=seed)


def final_evaluation(head, ds, cfg: TrainConfig, torch, check_budget) -> dict:
    """One forward over all available TRAIN+VAL rows; never touch TEST payloads."""
    head.eval()
    evaluated = {"train": [], "val": []}
    parity_rows = []
    errors = []
    with torch.inference_mode():
        for split in ("train", "val"):
            for row in ds.rows(split):
                check_budget()
                features, labels = ds.arrays(row)
                tokens = [None] * 12
                for hook in HOOKS:
                    tokens[hook] = torch.from_numpy(features[f"tokens_{hook}"][None]).to(cfg.device)
                try:
                    flow, confidence = _model_outputs(head, tokens, torch, 1)
                    predicted_flow = flow[0].detach().cpu().numpy()
                    predicted_confidence = confidence[0].detach().cpu().numpy()
                    parity = confidence_parity(predicted_confidence, features["baseline_confidence"])
                    parity_rows.append({"id": row["id"], "split": split, **parity})
                    actual_flow = {"id": row["id"], "object_id": row["object_id"],
                                   **flow_metrics(predicted_flow, labels["flow"], labels["visible"]),
                                   "visibility": visibility_metrics(predicted_confidence, labels)}
                    baseline_flow = {"id": row["id"], "object_id": row["object_id"],
                                     **flow_metrics(features["baseline_flow"], labels["flow"], labels["visible"]),
                                     "visibility": visibility_metrics(features["baseline_confidence"], labels)}
                    evaluated[split].append({"adapted": actual_flow, "frozen": baseline_flow})
                    if not parity["passed"]:
                        errors.append({"id": row["id"], "split": split,
                                       "reason": "confidence_parity_failed",
                                       "details": parity["reasons"]})
                except Exception as exc:
                    errors.append({"id": row["id"], "split": split,
                                   "reason": f"{type(exc).__name__}: {exc}"})
                    if not parity_rows or parity_rows[-1].get("id") != row["id"]:
                        parity_rows.append({"id": row["id"], "split": split,
                                            "passed": False, "reasons": ["forward_or_metric_failure"]})

    summaries = {}
    for split in ("train", "val"):
        try:
            actual = [item["adapted"] for item in evaluated[split]]
            baseline = [item["frozen"] for item in evaluated[split]]
            summaries[split] = {
                "adapted": _metric_rows(actual, seed=cfg.seed),
                "frozen": _metric_rows(baseline, seed=cfg.seed),
                "frames": actual,
                "frozen_frames": baseline,
                "availability": {
                    "available": len(ds.rows(split)),
                    "total": len(ds.rows(split, available=False)),
                },
            }
        except Exception as exc:
            summaries[split] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append({"split": split, "reason": summaries[split]["error"]})
    inventory = {
        split: {"available": len(ds.rows(split)),
                "total": len(ds.rows(split, available=False)),
                "unavailable": len(ds.rows(split, available=False)) - len(ds.rows(split))}
        for split in ("train", "val", "test")
    }
    passed = not errors and all(row.get("passed") is True for row in parity_rows)
    return {
        "schema": "correspondence_flow_terminal_evaluation_v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "train": summaries.get("train"),
        "validation": {
            **summaries.get("val", {}),
            "label_origin": ds.document["provenance"]["label_origin"],
            "test_arrays_opened": False,
        },
        "confidence_parity": {
            "passed": all(row.get("passed") is True for row in parity_rows),
            "tolerance": CONFIDENCE_TOLERANCE,
            "strict_native_float32_threshold": float(CONFIDENCE_THRESHOLD),
            "rows": parity_rows,
            "rows_checked": len(parity_rows),
            "decision_changes": sum(row.get("decision_changes") or 0 for row in parity_rows),
            "max_abs_difference": max(
                (row["max_abs_difference"] for row in parity_rows
                 if row.get("max_abs_difference") is not None), default=None),
        },
        "inventory": inventory,
        "test_payload_reads": 0,
        "errors": errors,
    }


def _make_state(ds, cfg, cfg_hash, policy_hash, versions, runtime_policy,
                head, optimizer, torch, feature_union_identity=None):
    ordinary = ordinary_head_state(head)
    state = {
        "schema": "correspondence_head_checkpoint_v1",
        "dataset_sha256": ds.identity,
        "config_sha256": cfg_hash,
        "source_lock_sha256": sha256(LOCK),
        "policy_sha256": policy_hash,
        "source_files_sha256": policy_document(
            ds, cfg, versions, feature_union=feature_union_identity is not None
        )["source_files_sha256"],
        "frozen_state_sha256": training_state_digest(frozen_state(ordinary)),
        "epoch": 0,
        "batch_cursor": 0,
        "steps": 0,
        "bad_epochs": 0,
        "best_epe_px": None,
        "elapsed_seconds": 0.0,
        "epoch_loss_sum": 0.0,
        "epoch_loss_batches": 0,
        "history": [],
        "completed": False,
        "versions": dict(versions),
        "runtime_policy": dict(runtime_policy),
        "head": ordinary_head_state(head),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else [],
        "final_evaluation": None,
    }
    if feature_union_identity is not None:
        state["feature_union"] = copy.deepcopy(feature_union_identity)
    return state


def validate_feature_union_mode(state: dict, expected_identity) -> None:
    """Reject resume/export state whose feature-union opt-in identity changed."""
    if not isinstance(state, dict) or state.get("feature_union") != expected_identity:
        raise ValueError("resume feature-union mode/recipe/module identity mismatch")


def train(manifest: Path, output: Path, cfg: TrainConfig, *, cloud_execution=False,
          resume=False, stop_after_steps: int | None = None,
          feature_union=False) -> dict:
    if type(feature_union) is not bool:
        raise ValueError("feature_union must be an explicit boolean")
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps <= 0):
        raise ValueError("stop_after_steps must be a positive invocation step count")
    cloud_guard(cloud_execution)  # Before data or model access.
    cfg = continuation_config(cfg)
    report = preflight(manifest, cfg, feature_union=feature_union)
    versions = runtime_versions()
    ds = load_dataset(manifest, inspect_arrays=False)
    union_identity = None
    if feature_union:
        feature_union_module.validate_union_receipt(ds)
        union_identity = {
            "enabled": True,
            "recipe_sha256": ds.document["provenance"]["feature_recipe_sha256"],
            "module_sha256": sha256(Path(feature_union_module.__file__)),
        }
    elif read_json(ds.root / "recipe.json").get("schema") == feature_union_module.UNION_RECIPE_SCHEMA:
        raise ValueError("frozen-feature union datasets require explicit --feature-union")
    cfg_hash = canonical_hash(asdict(cfg))
    policy = policy_document(ds, cfg, versions, feature_union=feature_union)
    policy_hash = canonical_hash(policy)
    if resume:
        if not output.is_dir() or not (output / "last.pt").is_file():
            raise ValueError("resume requires an existing run with last.pt")
    else:
        output.mkdir(parents=True, exist_ok=False)
    import torch
    resume_state = None
    if resume:
        resume_state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        validate_checkpoint_integrity(resume_state)
        validate_feature_union_mode(resume_state, union_identity)
    runtime_attestation = configure_runtime_policy(cfg, torch)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    head = load_head(initialize=True, device=cfg.device)
    official_state = None
    terminal = attach_flow_terminal(head, torch)
    trainable = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    if sum(p.numel() for p in trainable) != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("optimizer parameter set is not exactly the 514 flow-row values")
    # The wrapped head starts from the official checkpoint on every invocation,
    # including resume, so frozen-state identity is checked against a fresh base.
    official_state = ordinary_head_state(head)
    state = _make_state(ds, cfg, cfg_hash, policy_hash, versions,
                        runtime_attestation, head, optimizer, torch,
                        feature_union_identity=union_identity)
    if resume:
        state = resume_state
        check_resume(state, identity=ds.identity, config_hash=cfg_hash,
                     source_hash=sha256(LOCK))
        validate_checkpoint_integrity(state)
        if state.get("policy_sha256") != policy_hash or state.get("source_files_sha256") != policy["source_files_sha256"]:
            raise ValueError("resume checkpoint flow-terminal policy/source identity mismatch")
        if state.get("versions") != versions:
            raise ValueError("runtime versions changed since checkpoint")
        if state.get("runtime_policy") != runtime_attestation:
            raise ValueError("actual runtime/CUDA numerical policy changed since checkpoint")
        if state.get("completed"):
            raise ValueError("run already completed; create a new run for a new experiment")
        if (state.get("final_evaluation") or {}).get("status") == "failed":
            raise ValueError("final confidence parity failed; this run cannot be resumed or exported")
        validate_resume_cursor(state, ds.rows("train"), cfg)
        validate_frozen_identity(state.get("head"), official_state,
                                 state.get("frozen_state_sha256"))
        restore_flow_state(head, state["head"], official_state)
        validate_optimizer_state(state.get("optimizer"), cfg, state["steps"])
        optimizer.load_state_dict(state["optimizer"])
        if cfg.device != "cuda" and state.get("cuda_rng"):
            raise ValueError("CPU continuation checkpoint unexpectedly contains CUDA RNG state")
        if cfg.device == "cuda" and not state.get("cuda_rng"):
            raise ValueError("CUDA continuation checkpoint lacks CUDA RNG state")
        torch.set_rng_state(state["cpu_rng"])
        if cfg.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    resumed_from = ({"steps": state["steps"], "epoch": state["epoch"],
                     "batch_cursor": state["batch_cursor"]} if resume else None)
    invocation_start_steps = state["steps"]
    write_json(output, "preflight.json", report, max_output_bytes=cfg.max_output_bytes)
    write_json(output, "config.json", asdict(cfg), max_output_bytes=cfg.max_output_bytes)
    started = time.monotonic()
    prior_elapsed = float(state["elapsed_seconds"])
    stop = {"signal": None}
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda signum, _frame: stop.update(signal=signum))

    def check_budget():
        elapsed = time.monotonic() - started
        if stop["signal"] is not None:
            raise BudgetStop("signal_requested")
        if prior_elapsed + elapsed >= cfg.max_total_seconds:
            raise BudgetStop("total_time_limit")
        if elapsed >= cfg.max_seconds:
            raise BudgetStop("invocation_time_limit")

    def snapshot():
        state.update(
            head=ordinary_head_state(head),
            optimizer=optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if cfg.device == "cuda" else [],
            elapsed_seconds=prior_elapsed + time.monotonic() - started,
        )
        state.pop("integrity_sha256", None)
        state["integrity_sha256"] = checkpoint_integrity_digest(state)

    def checkpoint(name="last.pt"):
        snapshot()
        save_checkpoint(output, name, state, cfg, torch_module=torch)

    def final_checkpoints():
        snapshot()
        save_checkpoint(output, "last.pt", state, cfg, torch_module=torch)
        save_checkpoint(output, "best.pt", state, cfg, torch_module=torch)

    status = "failed"
    reason = None
    try:
        checkpoint()  # Recoverable start before the first optimizer update.
        while state["epoch"] < cfg.epochs:
            batches = epoch_batches(ds.rows("train"), state["epoch"], cfg)
            if state["batch_cursor"] > len(batches):
                raise ValueError("checkpoint cursor exceeds deterministic epoch")
            # The entire DPT head remains in eval mode; only the wrapper's two
            # trainable tensors receive gradients.
            head.eval()
            for cursor in range(state["batch_cursor"], len(batches)):
                check_budget()
                if state["steps"] >= cfg.max_steps:
                    raise BudgetStop("step_limit")
                tokens, labels = tensor_batch(ds, batches[cursor], cfg.device, torch)
                for key in ("visible", "known"):
                    if not labels[key].any():
                        raise ValueError(f"empty {key} support in supervised training batch")
                optimizer.zero_grad(set_to_none=True)
                flow, confidence = _model_outputs(head, tokens, torch, len(batches[cursor]))
                loss = adaptation_loss(flow, confidence, labels, cfg, torch)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite flow-terminal loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.gradient_clip,
                                                      error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite flow-terminal gradients")
                optimizer.step()
                state["steps"] += 1
                state["batch_cursor"] = cursor + 1
                state["epoch_loss_sum"] += float(loss.detach().cpu())
                state["epoch_loss_batches"] += 1
                state["last_step"] = {
                    "row_ids": [row["id"] for row in batches[cursor]],
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(norm.detach().cpu()),
                }
                if state["steps"] % cfg.checkpoint_every_steps == 0:
                    checkpoint()
                    write_json(output, "status.json", {
                        "status": "running", "epoch": state["epoch"],
                        "steps": state["steps"], "elapsed_seconds": state["elapsed_seconds"],
                    }, max_output_bytes=cfg.max_output_bytes)
                if (stop_after_steps is not None
                        and state["steps"] - invocation_start_steps >= stop_after_steps):
                    raise BudgetStop("invocation_step_limit")
            if state["epoch_loss_batches"] <= 0:
                raise ValueError("epoch has no optimizer updates")
            state["history"].append({
                "epoch": state["epoch"], "steps": state["steps"],
                "train_loss": state["epoch_loss_sum"] / state["epoch_loss_batches"],
            })
            state.update(epoch=state["epoch"] + 1, batch_cursor=0,
                         epoch_loss_sum=0.0, epoch_loss_batches=0, bad_epochs=0)
            checkpoint()

        # Candidate state is persisted as incomplete before the single final
        # evaluation. A failed parity check can never be exported as completed.
        checkpoint()
        evaluation = final_evaluation(head, ds, cfg, torch, check_budget)
        evaluation["policy_sha256"] = policy_hash
        evaluation["runtime_versions"] = dict(versions)
        evaluation["runtime_policy"] = runtime_attestation
        evaluation["stage_elapsed_seconds"] = time.monotonic() - started
        evaluation["cuda_peak_allocated_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if cfg.device == "cuda" else None)
        evaluation["cuda_peak_reserved_bytes"] = (
            int(torch.cuda.max_memory_reserved()) if cfg.device == "cuda" else None)
        write_json(output, "final-evaluation.json", evaluation,
                   max_output_bytes=cfg.max_output_bytes)
        state["final_evaluation"] = evaluation
        if not evaluation["passed"]:
            checkpoint()
            raise ValueError("final confidence parity/evaluation failed; run remains incomplete")
        if len(state["history"]) != cfg.epochs or not state["history"]:
            raise ValueError("completed training history differs from fixed epoch count")
        state["history"][-1]["validation"] = evaluation["validation"]
        state["completed"] = True
        state["best_epe_px"] = evaluation["validation"]["adapted"]["object_macro_epe_px"]
        final_checkpoints()  # Both names carry the same final ordinary DPT state.
        write_json(output, "history.json", state["history"], max_output_bytes=cfg.max_output_bytes)
        status = "completed"
        return {"status": status, "steps": state["steps"], "epoch": state["epoch"],
                "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                "policy_sha256": policy_hash,
                "cuda_peak_allocated_bytes": evaluation["cuda_peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": evaluation["cuda_peak_reserved_bytes"]}
    except BudgetStop as exc:
        status, reason = "incomplete", str(exc)
        checkpoint()
        return {"status": status, "reason": reason, "steps": state["steps"],
                "epoch": state["epoch"], "policy_sha256": policy_hash}
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        write_json(output, "status.json", {
            "status": status, "reason": reason,
            "epoch": state["epoch"], "steps": state["steps"],
            "elapsed_seconds": prior_elapsed + time.monotonic() - started,
            "stage_elapsed_seconds": time.monotonic() - started,
            "resumed_from": resumed_from, "last_step": state.get("last_step"),
            "resume": "last.pt" if (output / "last.pt").exists() else None,
            "policy_sha256": policy_hash,
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if cfg.device == "cuda" else None),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if cfg.device == "cuda" else None),
            "quality_claim": False, "test_arrays_opened": False,
        }, max_output_bytes=cfg.max_output_bytes)


def _checkpoint_policy_check(state: dict, cfg: TrainConfig, versions: dict,
                             runtime_policy: dict) -> str:
    feature_union = state.get("feature_union")
    if feature_union is None:
        ds = SimpleNamespace(identity=state.get("dataset_sha256"))
        is_union = False
    else:
        if (not isinstance(feature_union, dict)
                or set(feature_union) != {"enabled", "recipe_sha256", "module_sha256"}
                or feature_union.get("enabled") is not True):
            raise ValueError("checkpoint feature-union identity is malformed")
        for key in ("recipe_sha256", "module_sha256"):
            value = feature_union.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("checkpoint feature-union identity hash is malformed")
        if feature_union["module_sha256"] != sha256(Path(feature_union_module.__file__)):
            raise ValueError("checkpoint feature-union module source changed")
        ds = SimpleNamespace(identity=state.get("dataset_sha256"), document={
            "provenance": {"feature_recipe_sha256": feature_union["recipe_sha256"]}})
        is_union = True
    cfg_hash = canonical_hash(asdict(cfg))
    policy_hash = canonical_hash(policy_document(ds, cfg, versions, feature_union=is_union))
    check_resume(state, identity=ds.identity, config_hash=cfg_hash,
                 source_hash=sha256(LOCK))
    validate_checkpoint_integrity(state)
    validate_optimizer_state(state.get("optimizer"), cfg, state.get("steps", -1))
    if (state.get("policy_sha256") != policy_hash
            or state.get("versions") != versions
            or state.get("runtime_policy") != runtime_policy
            or state.get("source_files_sha256") != policy_document(
                ds, cfg, versions, feature_union=is_union)["source_files_sha256"]):
        raise ValueError("checkpoint policy/runtime/source identity mismatch")
    return policy_hash


def require_exportable_state(last: dict, best: dict) -> None:
    if (not last.get("completed") or not best.get("completed")
            or not best.get("history") or not best["history"][-1].get("validation")
            or (best.get("final_evaluation") or {}).get("passed") is not True):
        raise ValueError("export requires successful final confidence parity and completed training")


def export(run: Path, destination: Path, *, cloud_execution=False) -> dict:
    cloud_guard(cloud_execution)
    cfg = continuation_config(read_config(run / "config.json"))
    versions = runtime_versions()
    verify_sources()
    import torch
    runtime_attestation = configure_runtime_policy(cfg, torch)
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    policy_hash = _checkpoint_policy_check(last, cfg, versions, runtime_attestation)
    if _checkpoint_policy_check(best, cfg, versions, runtime_attestation) != policy_hash:
        raise ValueError("best/last checkpoints differ in flow-terminal policy")
    require_exportable_state(last, best)
    if set(best["head"]) != set(last["head"]):
        raise ValueError("best/last ordinary head state keys differ")
    for key in best["head"]:
        if not torch.equal(best["head"][key], last["head"][key]):
            raise ValueError("best/last are not the same final ordinary DPT candidate")
    # Reinitialize the official head to recheck every frozen tensor, then
    # strict-load the complete ordinary state into a consumer-shaped instance.
    official_head = load_head(initialize=True, device="cpu")
    official_state = {key: value.detach().cpu().clone()
                      for key, value in official_head.state_dict().items()}
    validate_frozen_identity(best["head"], official_state,
                             best.get("frozen_state_sha256"))
    head = load_head(initialize=False, device="cpu")
    head.load_state_dict(best["head"], strict=True)
    if sum(p.numel() for p in head.parameters()) <= 0:
        raise ValueError("strictly loaded ordinary DPT head has no parameters")
    if destination.exists() or destination.resolve().parent != run.resolve():
        raise ValueError("export must be a new file inside the run directory")
    validation = dict(best["history"][-1]["validation"])
    validation["flow_terminal_policy_sha256"] = policy_hash
    validation["confidence_parity"] = best["final_evaluation"]["confidence_parity"]
    artifact = {
        "schema": "correspondence_head_export_v1",
        "head": best["head"],
        "source_lock_sha256": sha256(LOCK),
        "base_checkpoint_sha256": GOTRACK_CHECKPOINT,
        "dataset_sha256": best["dataset_sha256"],
        "config_sha256": canonical_hash(asdict(cfg)),
        "validation": validation,
        "quality_claim": False,
        "deployment": "explicit_cloud_research_only_pose_head_replacement",
    }
    # Keep this exact 9-key top-level schema for the existing installer.
    with atomic_stream(run, destination, max_file_bytes=cfg.max_checkpoint_bytes,
                       max_output_bytes=cfg.max_output_bytes) as stream:
        torch.save(artifact, stream)
    receipt = {
        "path": destination.name, "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "trainable_parameters": TRAINABLE_PARAMETER_COUNT,
        "policy_sha256": policy_hash,
        "loaded_strictly": True, "confidence_parity_passed": True,
        "inference_parity_executed": False, "mobile_ready": False,
    }
    if best.get("feature_union") is not None:
        receipt["feature_union"] = copy.deepcopy(best["feature_union"])
    write_json(run, destination.name + ".json", receipt,
               max_output_bytes=cfg.max_output_bytes)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--feature-union", action="store_true",
                             help="require and bind a frozen-feature cache union receipt")
        if name != "preflight":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--cloud-execution", action="store_true")
            command.add_argument("--stop-after-steps", type=int)
    command = sub.add_parser("export")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.manifest, read_config(args.config),
                           feature_union=args.feature_union)
    elif args.command == "export":
        result = export(args.run, args.output, cloud_execution=args.cloud_execution)
    else:
        result = train(args.manifest, args.output, read_config(args.config),
                       cloud_execution=args.cloud_execution,
                       resume=args.command == "resume",
                       stop_after_steps=args.stop_after_steps,
                       feature_union=args.feature_union)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 3 if result.get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
