"""Bounded cloud training of the released DPT correspondence head on frozen tokens.

This is a new head-adaptation recipe, not the unreleased original GoTrack trainer.
The network predicts template-to-query pixel displacement and visibility. No pose
or annotation is a model input; sealed test data is excluded from this module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np

from .quality_adaptation_data import (HOOKS, SIZE, canonical_hash, data_report, flow_metrics,
                                      load_dataset, object_summary, read_json, sha256, visibility_metrics)
from .quality_adaptation_runtime import (GOTRACK_CHECKPOINT, LOCK, atomic_stream, check_resume,
                                         cloud_guard, load_head, runtime_versions, save_checkpoint,
                                         verify_sources, write_json, training_state_digest)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 0.00001
    weight_decay: float = 0.0001
    visibility_weight: float = 1.0
    patience: int = 3
    min_delta_px: float = 0.001
    gradient_clip: float = 1.0
    checkpoint_every_steps: int = 10
    max_steps: int = 2000
    max_seconds: float = 900.0
    max_total_seconds: float = 3600.0
    max_checkpoint_bytes: int = 384 * 1024**2
    max_output_bytes: int = 1536 * 1024**2
    device: str = "cpu"
    amp: bool = False

    def validate(self):
        for key in ("epochs", "batch_size", "patience", "checkpoint_every_steps", "max_steps",
                    "max_checkpoint_bytes", "max_output_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be a uint32 integer")
        for key in ("learning_rate", "gradient_clip", "max_seconds", "max_total_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("weight_decay", "visibility_weight", "min_delta_px"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.device not in ("cpu", "cuda") or type(self.amp) is not bool or (self.amp and self.device != "cuda"):
            raise ValueError("device must be cpu/cuda; AMP requires CUDA")
        if (self.batch_size > 4 or self.epochs > 1000 or self.max_steps > 100000
                or self.max_total_seconds > 86400 or self.max_seconds > self.max_total_seconds
                or self.max_output_bytes > 8 * 1024**3 or self.max_checkpoint_bytes > 1024**3
                or self.max_output_bytes < 3 * self.max_checkpoint_bytes + 8 * 1024**2):
            raise ValueError("config exceeds hard bounds or cannot retain best+last+atomic temporary checkpoint")


def read_config(path: Path) -> TrainConfig:
    value = read_json(path)
    if not isinstance(value, dict) or set(value) - set(TrainConfig.__dataclass_fields__):
        raise ValueError("unknown training config fields")
    cfg = TrainConfig(**value)
    cfg.validate()
    return cfg


def validate_feature_receipt(ds) -> None:
    """Bind archives to actual extractor receipts, beyond manifest assertions."""
    recipe = read_json(ds.root / "recipe.json")
    index = read_json(ds.root / "index.json")
    identity = canonical_hash(recipe)
    from .quality_adaptation_smoke import validate_frozen_feature_source
    validate_frozen_feature_source(recipe)
    if (identity != ds.document["provenance"]["feature_recipe_sha256"]
            or index.get("recipe_sha256") != identity or recipe.get("annotation_inputs") != []
            or recipe.get("producer_sha256") != sha256(Path(__file__).with_name("quality_adaptation_prepare.py"))
            or ds.document["provenance"]["source_lock_sha256"] != sha256(LOCK)
            or recipe.get("hooks") != list(HOOKS) or recipe.get("schema") != "frozen_gotrack_decoder_recipe_v1"):
        raise ValueError("dataset feature recipe is incompatible or unauthenticated")
    extracted = {r["id"]: r for r in index["rows"]}
    if len(extracted) != len(index["rows"]):
        raise ValueError("duplicate extraction ids")
    if set(extracted) - {r["id"] for r in ds.document["rows"]}:
        raise ValueError("dataset omits extracted rows/failures")
    available = {key: r for key, r in extracted.items() if r["status"] == "available"}
    for split in ("train", "val"):
        for row in ds.rows(split):
            receipt = available.get(row["id"], {})
            if receipt.get("features") != row["features"]:
                raise ValueError("dataset feature descriptor differs from extractor index")
            if receipt.get("query_rgb_sha256") != row["query_rgb_sha256"]:
                raise ValueError("dataset query RGB hash differs from extractor index")


def preflight(manifest: Path, cfg: TrainConfig) -> dict:
    cfg.validate()
    ds = load_dataset(manifest)
    validate_feature_receipt(ds)
    source = verify_sources()
    return {"schema": "correspondence_head_preflight_v1", "config": asdict(cfg),
            "config_sha256": canonical_hash(asdict(cfg)), **source, "data": data_report(ds),
            "trainable": "all upstream DPTHead pose_head parameters; backbone and decoder frozen",
            "loss": "mean_per_frame_visible_smooth_l1_beta_1px_plus_known_visibility_BCE",
            "selection_metric": "validation_object_macro_endpoint_error_crop_pixels",
            "rationale": "pixel EPE measures correspondence displacement without rewarding background; object macro averaging avoids long sequences dominating",
            "test_arrays_opened": False, "cloud_training_validated": False}


def epoch_batches(rows: list[dict], epoch: int, cfg: TrainConfig) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda r: r["id"])
    random.Random(cfg.seed + epoch).shuffle(ordered)
    return [ordered[i:i + cfg.batch_size] for i in range(0, len(ordered), cfg.batch_size)]


def tensor_batch(ds, rows, device, torch):
    arrays = [ds.arrays(row) for row in rows]
    tokens = [None] * 12
    for hook in HOOKS:
        tokens[hook] = torch.from_numpy(np.stack([f[f"tokens_{hook}"] for f, _ in arrays])).to(device)
    labels = {key: torch.from_numpy(np.stack([y[key] for _, y in arrays])).to(device)
              for key in ("flow", "visible", "known")}
    return tokens, labels


def adaptation_loss(flow, confidence, labels, cfg, torch):
    # Per-frame averaging prevents an object with a large visible area from
    # dominating. One-pixel SmoothL1 balances small precision errors and outliers.
    # BCE uses known foreground support only; unknown visibility is never negative.
    flow = flow.float()
    confidence = confidence.float()
    if not torch.isfinite(flow).all() or not torch.isfinite(confidence).all():
        raise ValueError("nonfinite network output")
    visible, known = labels["visible"], labels["known"]
    delta = torch.nn.functional.smooth_l1_loss(flow, labels["flow"], reduction="none", beta=1.).mean(dim=1)
    flow_loss = ((delta * visible).flatten(1).sum(1) / visible.flatten(1).sum(1)).mean()
    # Upstream head returns sigmoid probabilities, so BCE rather than logits BCE.
    visibility = torch.nn.functional.binary_cross_entropy(confidence.clamp(1e-6, 1 - 1e-6),
                                                           visible.float(), reduction="none")
    visibility_loss = ((visibility * known).flatten(1).sum(1) / known.flatten(1).sum(1)).mean()
    return flow_loss + cfg.visibility_weight * visibility_loss


class BudgetStop(RuntimeError):
    pass


def evaluate_validation(head, ds, cfg, torch, check_budget) -> dict:
    head.eval()
    actual, baseline = [], []
    with torch.inference_mode():
        for row in ds.rows("val"):
            check_budget()
            features, label = ds.arrays(row)
            tokens = [None] * 12
            for hook in HOOKS:
                tokens[hook] = torch.from_numpy(features[f"tokens_{hook}"][None]).to(cfg.device)
            flow, confidence = head(tokens, (SIZE, SIZE))
            predicted = flow[0].float().cpu().numpy()
            actual.append({"id": row["id"], "object_id": row["object_id"],
                           **flow_metrics(predicted, label["flow"], label["visible"]),
                           "visibility": visibility_metrics(confidence[0].float().cpu().numpy(), label)})
            baseline.append({"id": row["id"], "object_id": row["object_id"],
                             **flow_metrics(features["baseline_flow"], label["flow"], label["visible"]),
                             "visibility": visibility_metrics(features["baseline_confidence"], label)})
    all_rows = ds.rows("val", available=False)
    return {"adapted": object_summary(actual, seed=cfg.seed), "frozen": object_summary(baseline, seed=cfg.seed),
            "frames": actual, "frozen_frames": baseline,
            "availability": {"available": len(actual), "total": len(all_rows)},
            "label_origin": ds.document["provenance"]["label_origin"],
            "test_arrays_opened": False}


def train(manifest: Path, output: Path, cfg: TrainConfig, *, cloud_execution=False, resume=False,
          stop_after_steps: int | None = None) -> dict:
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps <= 0):
        raise ValueError("stop_after_steps must be a positive invocation step count")
    cloud_guard(cloud_execution)  # Before importing Torch or loading any model.
    report = preflight(manifest, cfg)
    versions = runtime_versions()
    ds = load_dataset(manifest, inspect_arrays=False)
    cfg_hash = canonical_hash(asdict(cfg))
    if resume:
        if not output.is_dir() or not (output / "last.pt").is_file():
            raise ValueError("resume requires an existing run with last.pt")
    else:
        output.mkdir(parents=True, exist_ok=False)
    import torch
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no automatic resource upgrade")
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    head = load_head(initialize=not resume, device=cfg.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    state = {"schema": "correspondence_head_checkpoint_v1", "dataset_sha256": ds.identity,
             "config_sha256": cfg_hash, "source_lock_sha256": sha256(LOCK), "epoch": 0,
             "batch_cursor": 0, "steps": 0, "bad_epochs": 0, "best_epe_px": None,
             "elapsed_seconds": 0., "epoch_loss_sum": 0., "epoch_loss_batches": 0,
             "history": [], "completed": False, "versions": versions}
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        check_resume(state, identity=ds.identity, config_hash=cfg_hash, source_hash=sha256(LOCK))
        if state.get("versions") != versions:
            raise ValueError("runtime versions changed since checkpoint")
        if state.get("completed"):
            raise ValueError("run already completed; create a new run for a new experiment")
        head.load_state_dict(state["head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        if cfg.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    resumed_from = ({"steps": state["steps"], "epoch": state["epoch"],
                     "batch_cursor": state["batch_cursor"],
                     "optimizer_state_entries": len(optimizer.state),
                     "optimizer_sha256": training_state_digest(optimizer.state_dict())} if resume else None)
    invocation_start_steps = state["steps"]
    write_json(output, "preflight.json", report, max_output_bytes=cfg.max_output_bytes)
    write_json(output, "config.json", asdict(cfg), max_output_bytes=cfg.max_output_bytes)
    started = time.monotonic()
    prior_elapsed = state["elapsed_seconds"]
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

    def checkpoint(name="last.pt"):
        state.update(head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cfg.device == "cuda" else [],
                     elapsed_seconds=prior_elapsed + time.monotonic() - started)
        save_checkpoint(output, name, state, cfg, torch_module=torch)

    status = "failed"
    reason = None
    try:
        checkpoint()  # A recoverable starting point before the first optimizer step.
        while state["epoch"] < cfg.epochs:
            batches = epoch_batches(ds.rows("train"), state["epoch"], cfg)
            if state["batch_cursor"] > len(batches):
                raise ValueError("checkpoint cursor exceeds deterministic epoch")
            head.train()
            for cursor in range(state["batch_cursor"], len(batches)):
                check_budget()
                if state["steps"] >= cfg.max_steps:
                    raise BudgetStop("step_limit")
                tokens, labels = tensor_batch(ds, batches[cursor], cfg.device, torch)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.amp):
                    flow, confidence = head(tokens, (SIZE, SIZE))
                # Probability-space BCE is intentionally outside autocast; Torch
                # rejects BCE under autocast because its backward can underflow.
                loss = adaptation_loss(flow, confidence, labels, cfg, torch)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite adaptation loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite adaptation gradients")
                scaler.step(optimizer)
                scaler.update()
                state["steps"] += 1
                state["batch_cursor"] = cursor + 1
                state["epoch_loss_sum"] += float(loss.detach().cpu())
                state["epoch_loss_batches"] += 1
                state["last_step"] = {"row_ids": [r["id"] for r in batches[cursor]],
                                      "loss": float(loss.detach().cpu()),
                                      "gradient_norm": float(norm.detach().cpu())}
                if state["steps"] % cfg.checkpoint_every_steps == 0:
                    checkpoint()
                    write_json(output, "status.json", {"status": "running", "epoch": state["epoch"],
                               "steps": state["steps"], "elapsed_seconds": state["elapsed_seconds"]},
                               max_output_bytes=cfg.max_output_bytes)
                if stop_after_steps is not None and state["steps"] - invocation_start_steps >= stop_after_steps:
                    raise BudgetStop("invocation_step_limit")
            metrics = evaluate_validation(head, ds, cfg, torch, check_budget)
            score = metrics["adapted"]["object_macro_epe_px"]
            improved = state["best_epe_px"] is None or score < state["best_epe_px"] - cfg.min_delta_px
            state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
            if improved:
                state["best_epe_px"] = score
            state["history"].append({"epoch": state["epoch"], "steps": state["steps"],
                                     "train_loss": state["epoch_loss_sum"] / state["epoch_loss_batches"],
                                     "validation": metrics})
            state.update(epoch=state["epoch"] + 1, batch_cursor=0, epoch_loss_sum=0., epoch_loss_batches=0)
            state["completed"] = state["epoch"] >= cfg.epochs or state["bad_epochs"] >= cfg.patience
            if improved:
                checkpoint("best.pt")
            checkpoint()
            write_json(output, "history.json", state["history"], max_output_bytes=cfg.max_output_bytes)
            if state["completed"]:
                break
        status = "completed"
    except BudgetStop as e:
        status, reason = "incomplete", str(e)
        checkpoint()
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        write_json(output, "status.json", {"status": status, "reason": reason,
                   "epoch": state["epoch"], "steps": state["steps"],
                   "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                   "resumed_from": resumed_from, "last_step": state.get("last_step"),
                   "resume": "last.pt" if (output / "last.pt").exists() else None,
                   "quality_claim": False, "test_arrays_opened": False}, max_output_bytes=cfg.max_output_bytes)
    return {"status": status, "reason": reason}


def export(run: Path, destination: Path, *, cloud_execution=False) -> dict:
    cloud_guard(cloud_execution)
    cfg = read_config(run / "config.json")
    runtime_versions()
    verify_sources()
    import torch
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    check_resume(best, identity=last["dataset_sha256"], config_hash=canonical_hash(asdict(cfg)), source_hash=sha256(LOCK))
    if not last.get("completed") or not best.get("history"):
        raise ValueError("export requires completed training and a complete validation-selected checkpoint")
    head = load_head(initialize=False, device="cpu")
    head.load_state_dict(best["head"], strict=True)
    if destination.exists() or destination.resolve().parent != run.resolve():
        raise ValueError("export must be a new file inside the run directory")
    artifact = {"schema": "correspondence_head_export_v1", "head": best["head"],
                "source_lock_sha256": sha256(LOCK), "base_checkpoint_sha256": GOTRACK_CHECKPOINT,
                "dataset_sha256": best["dataset_sha256"], "config_sha256": best["config_sha256"],
                "validation": best["history"][-1]["validation"], "quality_claim": False,
                "deployment": "explicit_cloud_research_only_pose_head_replacement"}
    with atomic_stream(run, destination, max_file_bytes=cfg.max_checkpoint_bytes,
                       max_output_bytes=cfg.max_output_bytes) as f:
        torch.save(artifact, f)
    receipt = {"path": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size,
               "trainable_parameters": sum(p.numel() for p in head.parameters()),
               "loaded_strictly": True, "inference_parity_executed": False, "mobile_ready": False}
    write_json(run, destination.name + ".json", receipt, max_output_bytes=cfg.max_output_bytes)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        if name != "preflight":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--cloud-execution", action="store_true")
            p.add_argument("--stop-after-steps", type=int)
    p = sub.add_parser("export")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.manifest, read_config(args.config))
    elif args.command == "export":
        result = export(args.run, args.output, cloud_execution=args.cloud_execution)
    else:
        result = train(args.manifest, args.output, read_config(args.config),
                       cloud_execution=args.cloud_execution, resume=args.command == "resume",
                       stop_after_steps=args.stop_after_steps)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 3 if result.get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
