"""Optional SAM2 mask-decoder adaptation through the released upstream Trainer.

Requires reviewed segmentation-bottleneck evidence and separately annotated
object/sequence-disjoint data. It is never selected by the frozen inference path.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import time

import numpy as np

from .quality_adaptation_data import canonical_hash, descriptor, read_json, sha256
from .quality_adaptation_runtime import (LOCK, ROOT, SAM_CHECKPOINT, _source_imports, atomic_stream,
                                         cloud_guard, runtime_versions, tree_bytes, verify_checkpoint,
                                         verify_sources, write_json)

TEMPLATE = "sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml"


@dataclass(frozen=True)
class SamConfig:
    seed: int = 42
    epochs: int = 3
    patience: int = 2
    min_delta_iou: float = .001
    learning_rate: float = .00001
    train_frames: int = 2
    max_sequences: int = 64
    max_frames_per_sequence: int = 8
    max_seconds: float = 900.
    max_total_seconds: float = 3600.
    max_checkpoint_bytes: int = 768 * 1024**2
    max_output_bytes: int = 3 * 1024**3
    device: str = "cpu"
    amp: bool = False

    def validate(self):
        for key in ("seed", "epochs", "patience", "train_frames", "max_sequences", "max_frames_per_sequence",
                    "max_checkpoint_bytes", "max_output_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key == "seed" else 1):
                raise ValueError(f"invalid SAM {key}")
        for key in ("min_delta_iou", "learning_rate", "max_seconds", "max_total_seconds"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"invalid SAM {key}")
        if (self.epochs > 10 or not 2 <= self.train_frames <= self.max_frames_per_sequence <= 8
                or self.max_sequences > 256 or self.max_total_seconds > 86400
                or self.max_seconds > self.max_total_seconds or self.seed > 2**32 - 1
                or self.max_checkpoint_bytes > 1024**3 or self.max_output_bytes > 8 * 1024**3
                or self.max_output_bytes < 3 * self.max_checkpoint_bytes + 64 * 1024**2
                or self.device not in ("cpu", "cuda") or type(self.amp) is not bool
                or (self.amp and self.device != "cuda")):
            raise ValueError("SAM configuration exceeds supported bounds")


def validate_sam_dataset(manifest: Path, cfg: SamConfig, *, inspect_images=True) -> dict:
    cfg.validate()
    doc = read_json(manifest)
    if (set(doc) != {"schema", "label_origin", "segmentation_bottleneck", "sequences"}
            or doc["schema"] != "sam2_adaptation_dataset_v1" or doc["label_origin"] != "independent_annotations"):
        raise ValueError("SAM requires independent annotations and the explicit adaptation schema")
    evidence = read_json(descriptor(manifest.parent, doc["segmentation_bottleneck"], verify=True))
    if (set(evidence) != {"schema", "reviewed", "reason", "baseline_iou", "evaluation_protocol"}
            or evidence["schema"] != "segmentation_bottleneck_v1" or evidence["reviewed"] is not True
            or not isinstance(evidence["reason"], str) or not evidence["reason"].strip()
            or evidence["evaluation_protocol"] != "independent_masks_initial_prompt_only"
            or isinstance(evidence["baseline_iou"], bool) or not isinstance(evidence["baseline_iou"], (float, int))
            or not 0 <= evidence["baseline_iou"] <= 1):
        raise ValueError("SAM opt-in requires reviewed segmentation-bottleneck evidence")
    rows = doc["sequences"]
    if not isinstance(rows, list) or not 3 <= len(rows) <= cfg.max_sequences:
        raise ValueError("SAM sequence inventory is empty or exceeds its bound")
    owners = {k: {} for k in ("id", "object_id", "capture_group", "rgb_sha256")}
    ids = set()
    split_counts = {s: {"total": 0, "available": 0} for s in ("train", "val", "test")}
    train_fractions = []
    for row in rows:
        base = {"id", "object_id", "capture_group", "split", "status"}
        expected = base | ({"frames"} if row.get("status") == "available" else {"failure_reason"})
        if set(row) != expected or row["split"] not in split_counts or row["status"] not in ("available", "unavailable"):
            raise ValueError("invalid SAM sequence record")
        for key in ("id", "object_id", "capture_group"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row[key]):
                raise ValueError("SAM identity must be a safe folder/object identifier")
            previous = owners[key].setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError(f"SAM split leakage: {key}")
        if row["id"] in ids:
            raise ValueError("duplicate SAM sequence id")
        ids.add(row["id"])
        split_counts[row["split"]]["total"] += 1
        if row["status"] == "unavailable":
            if not isinstance(row["failure_reason"], str) or not row["failure_reason"]:
                raise ValueError("unavailable SAM sequence requires reason")
            continue
        frames = row["frames"]
        if not isinstance(frames, list) or not cfg.train_frames <= len(frames) <= cfg.max_frames_per_sequence:
            raise ValueError("SAM sequence frame count outside configured bounds")
        split_counts[row["split"]]["available"] += 1
        for frame_index, frame in enumerate(frames):
            if set(frame) != {"rgb", "mask"}:
                raise ValueError("SAM frames require RGB and independent mask descriptors")
            for key, folder, extension in (("rgb", "images", "jpg"), ("mask", "masks", "png")):
                expected_path = f"{folder}/{row['id']}/{frame_index:05d}.{extension}"
                if frame[key]["path"] != expected_path:
                    raise ValueError("SAM files must match contiguous DAVIS-style numbered folders")
                descriptor(manifest.parent, frame[key], verify=inspect_images and row["split"] != "test")
            previous = owners["rgb_sha256"].setdefault(frame["rgb"]["sha256"], row["split"])
            if previous != row["split"]:
                raise ValueError("SAM split leakage: duplicate RGB content")
            if inspect_images and row["split"] != "test":
                from PIL import Image
                with Image.open(manifest.parent / frame["rgb"]["path"]) as im:
                    if im.format != "JPEG" or im.mode != "RGB" or im.width * im.height > 4 * 1024**2:
                        raise ValueError("SAM input must be bounded RGB JPEG")
                    shape = im.size
                    im.load()
                with Image.open(manifest.parent / frame["mask"]["path"]) as im:
                    if im.format != "PNG" or im.mode not in ("P", "L") or im.size != shape:
                        raise ValueError("SAM annotation must be matching palette/grayscale PNG")
                    pixels = np.asarray(im)
                if not np.isin(pixels, [0, 1]).all() or (frame_index == 0 and not np.any(pixels == 1)):
                    raise ValueError("SAM single-object masks use 0 background/1 object; first frame needs foreground")
                if row["split"] == "train":
                    train_fractions.append(float((pixels == 1).mean()))
        if inspect_images and row["split"] != "test":
            for folder, extension in (("images", "jpg"), ("masks", "png")):
                actual = {p.name for p in (manifest.parent / folder / row["id"]).iterdir()}
                expected = {f"{i:05d}.{extension}" for i in range(len(frames))}
                if actual != expected:
                    raise ValueError("SAM upstream loader would discover unmanifested files")
    for split, count in split_counts.items():
        if not count["total"] or (split != "test" and not count["available"]):
            raise ValueError(f"SAM missing {split} inventory/available sequences")
    return {"document": doc, "dataset_sha256": canonical_hash(doc), "availability": split_counts,
            "train_foreground_fraction_quantiles": np.quantile(train_fractions, [0, .5, 1]).tolist() if train_fractions else None,
            "test_arrays_opened": False}


def build_sam_config(template: dict, manifest: Path, output: Path, cfg: SamConfig) -> dict:
    """Pure mutation of the actual pinned upstream Hydra template."""
    result = copy.deepcopy(template)
    result["scratch"].update(train_batch_size=1, num_train_workers=0, num_frames=cfg.train_frames,
                             max_num_objects=1, base_lr=cfg.learning_rate, vision_lr=cfg.learning_rate,
                             num_epochs=cfg.epochs, phases_per_epoch=1)
    result["dataset"].update(img_folder=str(manifest.parent / "images"), gt_folder=str(manifest.parent / "masks"),
                             file_list_txt=str(output / "train.txt"), multiplier=1)
    trainer = result["trainer"]
    trainer.update(mode="train", max_epochs=cfg.epochs, accelerator=cfg.device, seed_value=cfg.seed, val_epoch_freq=1)
    trainer["distributed"]["backend"] = "gloo" if cfg.device == "cpu" else "nccl"
    trainer["optim"]["amp"]["enabled"] = cfg.amp
    trainer["model"].update(num_init_cond_frames_for_train=1, rand_init_cond_frames_for_train=False,
                             num_frames_to_correct_for_train=1, rand_frames_to_correct_for_train=False,
                             num_init_cond_frames_for_eval=1, num_frames_to_correct_for_eval=1,
                             prob_to_use_pt_input_for_eval=0., prob_to_use_box_input_for_eval=0.)
    # Frozen module groups retain their upstream optimizer membership but never
    # receive gradients; only sam_mask_decoder is enabled by the trainer adapter.
    trainer["checkpoint"]["save_dir"] = str(output / "checkpoints")
    trainer["checkpoint"]["save_freq"] = 0
    trainer["checkpoint"]["model_weight_initializer"]["state_dict"]["checkpoint_path"] = str(
        ROOT / ".cache/model-quality/checkpoints/sam2.1_hiera_base_plus.pt")
    trainer["logging"]["log_dir"] = str(output / "logs")
    trainer["logging"]["tensorboard_writer"]["log_dir"] = str(output / "tensorboard")
    trainer["logging"]["log_freq"] = 10
    train = trainer["data"]["train"]
    train.update(num_workers=0, pin_memory=cfg.device == "cuda", drop_last=False)
    # In this upstream class the flag only controls exception retry/substitution.
    # Fail on a broken sample instead of silently training on a different video.
    train["datasets"][0]["dataset"]["datasets"][0]["training"] = False
    val = copy.deepcopy(train)
    val.update(shuffle=False)
    val["datasets"] = [{
        "_target_": "training.dataset.vos_dataset.VOSDataset", "training": False, "multiplier": 1,
        "video_dataset": {"_target_": "training.dataset.vos_raw_dataset.PNGRawDataset",
                          "img_folder": "${dataset.img_folder}", "gt_folder": "${dataset.gt_folder}",
                          "file_list_txt": str(output / "val.txt")},
        "sampler": {"_target_": "training.dataset.vos_sampler.EvalSampler"},
        "transforms": [{"_target_": "training.dataset.transforms.ComposeAPI", "transforms": [
            {"_target_": "training.dataset.transforms.RandomResizeAPI", "sizes": "${scratch.resolution}", "square": True,
             "consistent_transform": True},
            {"_target_": "training.dataset.transforms.ToTensorAPI"},
            {"_target_": "training.dataset.transforms.NormalizeAPI", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}]}],
    }]
    trainer["data"]["val"] = val
    result["launcher"].update(num_nodes=1, gpus_per_node=1, experiment_log_dir=str(output))
    result["submitit"]["use_cluster"] = False
    return result


def segmentation_scores(predicted: np.ndarray, target: np.ndarray) -> dict:
    if predicted.shape != target.shape or predicted.dtype != bool or target.dtype != bool:
        raise ValueError("segmentation metric expects equal-shaped bool masks")
    intersection = np.count_nonzero(predicted & target)
    union = np.count_nonzero(predicted | target)
    denominator = np.count_nonzero(predicted) + np.count_nonzero(target)
    return {"iou": intersection / union if union else 1.,
            "dice": 2 * intersection / denominator if denominator else 1.}


class SamStopped(RuntimeError):
    pass


def trainer_class(base, *, output, cfg, identity, validation_objects, prior_elapsed, torch):
    """Extend checkpoint/budget/metric hooks; optimization stays in upstream Trainer."""
    class BoundedSAMTrainer(base):
        def __init__(self, **kwargs):
            self.adaptation_started = time.monotonic()
            self.adaptation_stop = False
            self.adaptation_state = {"best_iou": None, "bad_epochs": 0, "history": [], "baseline": None}
            super().__init__(**kwargs)

        def _construct_optimizers(self):
            enabled = []
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith("sam_mask_decoder."))
                if parameter.requires_grad:
                    enabled.append(name)
            if not enabled:
                raise ValueError("pinned SAM model has no sam_mask_decoder parameters")
            super()._construct_optimizers()

        def _check_budget(self):
            elapsed = time.monotonic() - self.adaptation_started
            if (self.adaptation_stop or elapsed >= cfg.max_seconds or prior_elapsed + elapsed >= cfg.max_total_seconds):
                raise SamStopped("time_or_signal_limit")
            if tree_bytes(output) > cfg.max_output_bytes - cfg.max_checkpoint_bytes:
                raise SamStopped("output_budget_limit")

        def _step(self, batch, model, phase):
            self._check_budget()
            unwrapped = model.module if hasattr(model, "module") else model
            if phase == "train":
                for name, module in unwrapped.named_children():
                    module.train(name == "sam_mask_decoder")
            return super()._step(batch, model, phase)

        def _save_checkpoint(self, checkpoint, checkpoint_path):
            path = Path(checkpoint_path)
            if path.name not in ("checkpoint.pt", "best.pt"):
                raise ValueError("SAM retention permits only checkpoint.pt and best.pt")
            checkpoint["adaptation"] = {"identity": identity, "state": self.adaptation_state,
                                         "cpu_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                                         "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else []}
            np_state = np.random.get_state()
            checkpoint["adaptation"]["numpy_rng"] = [np_state[0], np_state[1].tolist(), *np_state[2:]]
            unwrapped = self.model.module if hasattr(self.model, "module") else self.model
            checkpoint["adaptation"]["prompt_rng"] = unwrapped.rng.bit_generator.state
            with atomic_stream(output, path, max_file_bytes=cfg.max_checkpoint_bytes,
                               max_output_bytes=cfg.max_output_bytes) as f:
                torch.save(checkpoint, f)

        def _load_resuming_checkpoint(self, path):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("adaptation", {}).get("identity") != identity:
                raise ValueError("SAM resume dataset/config/source identity mismatch")
            super()._load_resuming_checkpoint(path)
            meta = checkpoint["adaptation"]
            self.adaptation_state = meta["state"]
            torch.set_rng_state(meta["cpu_rng"])
            random.setstate(meta["python_rng"])
            n = meta["numpy_rng"]
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
            if cfg.device == "cuda":
                torch.cuda.set_rng_state_all(meta["cuda_rng"])
            self.model.rng.bit_generator.state = meta["prompt_rng"]

        def run_val(self):
            history = self.adaptation_state["history"]
            if history and not history[-1]["initial_frozen_baseline"] and history[-1]["epoch"] == self.epoch:
                return  # Upstream reruns the previous epoch's validation on resume.
            scored = []
            def capture(_module, inputs, outputs):
                batch = inputs[0]
                # Evaluation uses one initial annotated mask. Exclude that frame;
                # all later frames are propagated without annotation corrections.
                for t in range(1, len(outputs)):
                    pred = outputs[t]["pred_masks_high_res"][:, 0].detach().cpu().numpy() > 0
                    target = batch.masks[t].detach().cpu().numpy().astype(bool)
                    identifiers = batch.metadata.unique_objects_identifier[t].detach().cpu().numpy()
                    for i in range(len(pred)):
                        video_id = int(identifiers[i, 0])
                        scored.append({"object_id": validation_objects[video_id], "video_id": video_id,
                                       **segmentation_scores(pred[i], target[i])})
            hook = self.model.register_forward_hook(capture)
            try:
                super().run_val()
            finally:
                hook.remove()
            if not scored:
                raise ValueError("SAM validation produced no post-prompt frames")
            grouped = {}
            for row in scored:
                grouped.setdefault(row["object_id"], []).append(row)
            object_iou = np.array([np.mean([r["iou"] for r in grouped[k]]) for k in sorted(grouped)])
            score = float(object_iou.mean())
            metrics = {"object_macro_iou": score,
                       "object_macro_dice": float(np.mean([np.mean([r["dice"] for r in grouped[k]]) for k in sorted(grouped)])),
                       "objects": len(grouped), "scored_frames": len(scored), "bootstrap_95ci_iou": None,
                       "protocol": "initial_annotation_mask_then_unprompted_frames", "test_arrays_opened": False}
            if len(object_iou) > 1:
                rng = np.random.default_rng(cfg.seed)
                metrics["bootstrap_95ci_iou"] = np.quantile(rng.choice(object_iou, (2000, len(object_iou))).mean(1), [.025, .975]).tolist()
            initial = self.adaptation_state["baseline"] is None
            if initial:
                self.adaptation_state["baseline"] = metrics
            improved = self.adaptation_state["best_iou"] is None or score > self.adaptation_state["best_iou"] + cfg.min_delta_iou
            self.adaptation_state["bad_epochs"] = 0 if improved else self.adaptation_state["bad_epochs"] + 1
            if improved:
                self.adaptation_state["best_iou"] = score
            self.adaptation_state["history"].append({"epoch": self.epoch, "initial_frozen_baseline": initial, "metrics": metrics})
            self.save_checkpoint(self.epoch if initial else self.epoch + 1,
                                 checkpoint_names=["checkpoint", "best"] if improved else ["checkpoint"])
            write_json(output, "validation.json", self.adaptation_state, max_output_bytes=cfg.max_output_bytes)
            if self.adaptation_state["bad_epochs"] >= cfg.patience:
                raise SamStopped("early_stopping")
    return BoundedSAMTrainer


def run(manifest: Path, output: Path, cfg: SamConfig, *, cloud_execution=False, enable_sam=False, resume=False):
    cloud_guard(cloud_execution)
    if not enable_sam:
        raise ValueError("optional SAM training requires --enable-sam-adaptation and bottleneck evidence")
    report = validate_sam_dataset(manifest, cfg)
    source_identity = verify_sources("sam2")
    verify_checkpoint("sam2")
    versions = runtime_versions(sam=True)
    source = _source_imports("sam2")
    import torch
    from omegaconf import OmegaConf
    from training.trainer import Trainer
    import training.trainer as upstream_trainer
    from training.utils.train_utils import register_omegaconf_resolvers
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA is unavailable; no resource upgrade is automatic")
    identity = canonical_hash({"dataset": report["dataset_sha256"], "config": asdict(cfg),
                               "sources": source_identity, "versions": versions})
    prior_elapsed = 0.
    if resume:
        previous = read_json(output / "status.json")
        if previous["identity"] != identity or previous["status"] == "completed":
            raise ValueError("SAM resume identity differs or run already completed")
        prior_elapsed = previous["elapsed_seconds"]
        if not (output / "checkpoints/checkpoint.pt").is_file():
            raise ValueError("SAM has no completed-epoch checkpoint to resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        names = sorted(r["id"] for r in report["document"]["sequences"] if r["split"] == split and r["status"] == "available")
        text = ("\n".join(names) + "\n").encode()
        with atomic_stream(output, output / (split + ".txt"), max_file_bytes=len(text), max_output_bytes=cfg.max_output_bytes) as f:
            f.write(text)
    register_omegaconf_resolvers()
    template = OmegaConf.to_container(OmegaConf.load(source / TEMPLATE), resolve=False)
    generated = build_sam_config(template, manifest.resolve(), output.resolve(), cfg)
    write_json(output, "upstream-config.json", generated, max_output_bytes=cfg.max_output_bytes)
    write_json(output, "config.json", asdict(cfg), max_output_bytes=cfg.max_output_bytes)
    write_json(output, "preflight.json", {k: v for k, v in report.items() if k != "document"}, max_output_bytes=cfg.max_output_bytes)
    config = OmegaConf.create(generated)
    kwargs = OmegaConf.to_container(config.trainer, resolve=True)
    kwargs.pop("_target_")
    validation_rows = sorted((r for r in report["document"]["sequences"] if r["split"] == "val" and r["status"] == "available"), key=lambda r: r["id"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    bounded = trainer_class(Trainer, output=output, cfg=cfg, identity=identity,
                             validation_objects={i: r["object_id"] for i, r in enumerate(validation_rows)},
                             prior_elapsed=prior_elapsed, torch=torch)
    trainer = None
    started = time.monotonic()
    status, reason = "failed", None
    handlers = {}
    try:
        # Released Trainer logs every environment value, including provider keys.
        # Suppress that diagnostic in this isolated process without editing source.
        original_env_log = upstream_trainer.log_env_variables
        upstream_trainer.log_env_variables = lambda: None
        try:
            trainer = bounded(**kwargs)
        finally:
            upstream_trainer.log_env_variables = original_env_log
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: setattr(trainer, "adaptation_stop", True))
        if trainer.adaptation_state["baseline"] is None:
            trainer.run_val()
        trainer.run()
        status = "completed"
    except SamStopped as exc:
        reason = str(exc)
        status = "completed" if reason == "early_stopping" else "incomplete"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        write_json(output, "status.json", {"status": status, "reason": reason, "identity": identity,
                   "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                   "resume_semantics": "last_completed_epoch_replay_partial_epoch",
                   "trainable": "sam_mask_decoder_only", "quality_claim": False,
                   "availability": report["availability"]}, max_output_bytes=cfg.max_output_bytes)
    return {"status": status, "reason": reason}


def export(run: Path, destination: Path, *, cloud_execution=False):
    """Export upstream-compatible {'model': state_dict}; keep the export private."""
    cloud_guard(cloud_execution)
    verify_sources("sam2")
    runtime_versions(sam=True)
    status = read_json(run / "status.json")
    cfg = SamConfig(**read_json(run / "config.json"))
    cfg.validate()
    if status["status"] != "completed" or destination.exists() or destination.resolve().parent != run.resolve():
        raise ValueError("SAM export requires completed run and a new file inside its run directory")
    import torch
    best = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
    if best.get("adaptation", {}).get("identity") != status["identity"]:
        raise ValueError("SAM selected checkpoint identity mismatch")
    with atomic_stream(run, destination, max_file_bytes=cfg.max_checkpoint_bytes, max_output_bytes=cfg.max_output_bytes) as f:
        torch.save({"model": best["model"]}, f)
    result = {"path": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size,
              "source_lock_sha256": sha256(LOCK), "base_checkpoint_sha256": SAM_CHECKPOINT,
              "inference_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
              "selection": best["adaptation"]["state"], "quality_claim": False,
              "inference_parity_executed": False, "public_upload": False}
    write_json(run, destination.name + ".json", result, max_output_bytes=cfg.max_output_bytes)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "resume", "export"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cloud-execution", action="store_true")
    parser.add_argument("--enable-sam-adaptation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "export":
        if args.run is None or args.output is None:
            parser.error("export requires --run and --output")
        print(json.dumps(export(args.run, args.output, cloud_execution=args.cloud_execution), indent=2))
        return 0
    if args.config is None or args.manifest is None:
        parser.error("--config and --manifest are required")
    cfg = SamConfig(**read_json(args.config))
    if args.command == "preflight":
        result = validate_sam_dataset(args.manifest, cfg)
        result.pop("document")
    else:
        if args.output is None:
            parser.error("--output is required for train/resume")
        result = run(args.manifest, args.output, cfg, cloud_execution=args.cloud_execution,
                     enable_sam=args.enable_sam_adaptation, resume=args.command == "resume")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 3 if result.get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
