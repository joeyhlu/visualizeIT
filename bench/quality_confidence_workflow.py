"""Run the frozen confidence-capacity comparison on an approved cloud host.

The workflow is provider independent: callers supply the audited cache manifest
and a fresh output directory.  A local dry run only prints the exact subprocess
plan.  It does not audit data, import the trainer, create output, or launch a
stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

from .quality_assets import save_result
from .quality_confidence_rejection import (
    ARTIFACT_PURPOSE,
    POLICY,
    load_research_dataset,
)
from .quality_offline import run_stage


ROOT = Path(__file__).resolve().parents[1]
OVERALL_TIMEOUT_SECONDS = 880.0
OUTPUT_LIMIT_BYTES = 256 * 1024**2
AGGREGATE_NAME = "confidence-workflow.json"
SOURCE_PATHS = (
    "bench/quality_confidence_workflow.py",
    "bench/quality_ranker_train.py",
    "bench/quality_confidence_rejection.py",
    "bench/quality_confidence_evaluate.py",
    "bench/quality_pose_ranker.py",
    "bench/quality_ranker_data.py",
    "bench/quality_ranker_features.py",
    "bench/quality_contract.py",
    "bench/quality_gotrack.py",
    "bench/quality_pipeline.py",
    "bench/quality_runner.py",
    "bench/quality_offline.py",
)
_PARITY_FIELDS = (
    "torch_numpy_probability_parity",
    "torch_numpy_complete_veto_decision_parity",
    "onnx_checked",
    "onnx_probability_parity",
    "onnx_complete_veto_decision_parity",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _cloud_guard(explicit: bool) -> None:
    if not explicit or platform.system() != "Linux":
        raise RuntimeError(
            "Confidence optimization requires an approved Linux cloud host "
            "and --cloud-execution"
        )


def _validate_output_target(output: Path) -> None:
    output = Path(output)
    if output.is_symlink():
        raise FileExistsError(f"Output must not be a symlink: {output}")
    if output.exists():
        raise FileExistsError(f"Workflow requires a fresh output directory: {output}")


def _fresh_output_paths(output: Path) -> tuple[Path, Path]:
    """Check the caller path before following any link, then check its target."""
    requested = Path(output).absolute()
    _validate_output_target(requested)
    resolved = requested.resolve()
    if resolved != requested:
        _validate_output_target(resolved)
    return requested, resolved


def build_stage_plan(manifest: Path, output: Path, *, root: Path = ROOT) -> list[dict[str, Any]]:
    """Return the exact frozen ten-stage subprocess plan without reading data."""
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    root = Path(root).resolve()
    plans: list[dict[str, Any]] = []
    for architecture, alias, config_name in (
        ("linear", "linear", "ranker-confidence-linear-cpu.json"),
        ("mlp64_32", "mlp", "ranker-confidence-mlp-cpu.json"),
    ):
        smoke = output / f"smoke-{alias}"
        run = output / alias
        config = root / "bench/runtime" / config_name
        plans.extend([
            {
                "name": f"smoke-{alias}",
                "timeout_cap_seconds": 300.0,
                "command": [
                    sys.executable, "-B", "-m", "bench.quality_ranker_train",
                    "confidence-smoke", "--manifest", str(manifest),
                    "--output", str(smoke), "--architecture", architecture,
                    "--cloud-execution",
                ],
            },
            {
                "name": f"train-{alias}",
                "timeout_cap_seconds": 310.0,
                "command": [
                    sys.executable, "-B", "-m", "bench.quality_ranker_train",
                    "confidence-train", "--manifest", str(manifest),
                    "--output", str(run), "--config", str(config),
                    "--cloud-execution",
                ],
            },
            {
                "name": f"export-{alias}",
                "timeout_cap_seconds": 90.0,
                "command": [
                    sys.executable, "-B", "-m", "bench.quality_ranker_train",
                    "confidence-export", "--manifest", str(manifest),
                    "--checkpoint", str(run / "best.pt"),
                    "--output", str(run / "model.json"), "--cloud-execution",
                ],
            },
        ])
        for split in ("train", "validation"):
            plans.append({
                "name": f"evaluate-{alias}-{split}",
                "timeout_cap_seconds": 60.0,
                "command": [
                    sys.executable, "-B", "-m", "bench.quality_confidence_evaluate",
                    "--cache-dir", str(manifest.parent), "--split", split,
                    "--checkpoint", str(run / "model.json"),
                    "--output", str(run / f"{split}.json"),
                ],
            })
    return plans


def _load_frozen_configs(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Validate the repository configs with the trainer's existing contract."""
    # Kept inside the execution path so --dry-run does not import the trainer.
    from .quality_ranker_train import (
        TrainConfig,
        _validate_research_config,
        canonical_hash,
    )

    result: dict[str, dict[str, Any]] = {}
    for alias, architecture, relative in (
        ("linear", "linear", Path("bench/runtime/ranker-confidence-linear-cpu.json")),
        ("mlp", "mlp64_32", Path("bench/runtime/ranker-confidence-mlp-cpu.json")),
    ):
        path = Path(root) / relative
        document = _read_json_object(path, f"frozen {alias} config")
        try:
            config = TrainConfig(**document)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Frozen {alias} confidence config is malformed") from exc
        if config.architecture != architecture or config.device != "cpu" or config.amp is not False:
            raise ValueError(
                f"Frozen {alias} config must use {architecture} on CPU without AMP"
            )
        try:
            _validate_research_config(config)
        except ValueError as exc:
            raise ValueError(f"Frozen {alias} confidence config is malformed") from exc
        effective = asdict(config)
        result[alias] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "document": document,
            "effective": effective,
            "identity_config_sha256": canonical_hash({
                "purpose": ARTIFACT_PURPOSE,
                "config": effective,
            }),
        }
    return result


def _source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {name: _sha256_file(Path(root) / name) for name in SOURCE_PATHS}


def _evaluation_source_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    """Project the captured workflow closure to the evaluator's exact schema."""
    return {
        Path(name).name: digest
        for name, digest in source_hashes.items()
        if name != "bench/quality_confidence_workflow.py"
    }


def _require_research_contract(document: dict[str, Any], description: str) -> None:
    if (
        document.get("artifact_purpose") != ARTIFACT_PURPOSE
        or document.get("policy") != POLICY
        or document.get("research_only") is not True
        or document.get("test_read") is not False
    ):
        raise ValueError(f"{description} contradicts the frozen research-only contract")


def _require_export_parity(document: dict[str, Any], description: str) -> None:
    _require_research_contract(document, description)
    if any(document.get(field) is not True for field in _PARITY_FIELDS):
        raise ValueError(f"{description} does not prove JSON and ONNX veto parity")


def _validate_smoke(output: Path, alias: str) -> str:
    path = output / f"smoke-{alias}" / "confidence-smoke-receipt.json"
    receipt = _read_json_object(path, f"{alias} smoke receipt")
    _require_research_contract(receipt, f"{alias} smoke receipt")
    for field in (
        "resume_exact_selected_model_parity",
        "resume_exact_selected_model_optimizer_scheduler_scaler_rng_progress_parity",
        "resume_exact_final_model_optimizer_scheduler_scaler_rng_progress_parity",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"{alias} smoke receipt lacks exact resume parity: {field}")
    export = receipt.get("export")
    if not isinstance(export, dict):
        raise ValueError(f"{alias} smoke receipt lacks export evidence")
    _require_export_parity(export, f"{alias} smoke export evidence")
    return str(path)


def _validate_architecture_outputs(
        output: Path, alias: str, architecture: str,
        config: dict[str, Any], cache_identity: dict[str, Any],
        evaluation_source_hashes: dict[str, str]) -> list[str]:
    run = output / alias
    best, final = run / "best.pt", run / "last.pt"
    if not best.is_file() or not final.is_file():
        raise ValueError(f"{alias} run lacks best.pt or last.pt")

    completion_path = run / "completion.json"
    completion = _read_json_object(completion_path, f"{alias} completion receipt")
    if (
        completion.get("artifact_purpose") != ARTIFACT_PURPOSE
        or completion.get("policy") != POLICY
        or completion.get("complete") is not True
        or completion.get("test_read") is not False
        or completion.get("selected_checkpoint") != "best.pt"
        or completion.get("final_checkpoint") != "last.pt"
        or completion.get("selected_checkpoint_sha256") != _sha256_file(best)
        or completion.get("final_checkpoint_sha256") != _sha256_file(final)
        or completion.get("config_sha256") != config["identity_config_sha256"]
    ):
        raise ValueError(f"{alias} completion receipt is incomplete or unbound")

    model_path, onnx_path = run / "model.json", run / "model.onnx"
    export_path = run / "model.export.json"
    if not model_path.is_file() or not onnx_path.is_file():
        raise ValueError(f"{alias} export lacks model.json or model.onnx")
    model = _read_json_object(model_path, f"{alias} confidence model")
    # Reuse the runtime's strict schema-v3 provenance and layer-shape loader.
    # Construction validates the portable math contract without scoring data.
    from .quality_pose_ranker import ConfidenceRejectionRanker
    ConfidenceRejectionRanker.from_checkpoint(model)
    identity = model.get("provenance", {}).get("identity", {})
    if (
        model.get("architecture") != architecture
        or identity.get("data") != cache_identity
        or identity.get("config") != config["effective"]
        or identity.get("config_sha256") != config["identity_config_sha256"]
        or model.get("provenance", {}).get("checkpoint_sha256") != _sha256_file(best)
    ):
        raise ValueError(f"{alias} model provenance differs from this workflow")

    export = _read_json_object(export_path, f"{alias} export receipt")
    _require_export_parity(export, f"{alias} export receipt")
    if (
        export.get("checkpoint_sha256") != _sha256_file(best)
        or export.get("model_sha256") != _sha256_file(model_path)
        or export.get("onnx_sha256") != _sha256_file(onnx_path)
    ):
        raise ValueError(f"{alias} export receipt hashes do not bind its artifacts")

    artifacts = [completion_path, best, final, model_path, onnx_path, export_path]
    for split in ("train", "validation"):
        evaluation_path = run / f"{split}.json"
        evaluation = _read_json_object(evaluation_path, f"{alias} {split} evaluation")
        _require_research_contract(evaluation, f"{alias} {split} evaluation")
        source_hashes = evaluation.get("source_hashes")
        if (
            evaluation.get("kind") != "confidence-rejection-research-evaluation-v1"
            or evaluation.get("split") != split
            or evaluation.get("cache_identity") != cache_identity
            or evaluation.get("checkpoint_sha256") != _sha256_file(model_path)
            or source_hashes != evaluation_source_hashes
        ):
            raise ValueError(f"{alias} {split} evaluation is incomplete or unbound")
        artifacts.append(evaluation_path)
    return [str(path) for path in artifacts]


def _validate_required_outputs(
        output: Path, configs: dict[str, dict[str, Any]],
        cache_identity: dict[str, Any],
        source_hashes: dict[str, str]) -> list[str]:
    artifacts = []
    evaluation_sources = _evaluation_source_hashes(source_hashes)
    for alias, architecture in (("linear", "linear"), ("mlp", "mlp64_32")):
        artifacts.append(_validate_smoke(output, alias))
        artifacts.extend(_validate_architecture_outputs(
            output, alias, architecture, configs[alias], cache_identity,
            evaluation_sources))
    return artifacts


def _require_deadline(started: float, phase: str) -> float:
    remaining = OVERALL_TIMEOUT_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError(
            f"Workflow deadline reached {phase}; preserve artifacts and do not claim completion"
        )
    return remaining


def _aggregate_base(
        manifest: Path, output: Path, dataset: dict[str, Any],
        configs: dict[str, dict[str, Any]], plans: list[dict[str, Any]],
        source_hashes: dict[str, str]) -> dict[str, Any]:
    cache_identity = dataset.get("identity")
    if not isinstance(cache_identity, dict):
        raise ValueError("Audited confidence dataset identity is malformed")
    return {
        "schema_version": 1,
        "kind": "portable-confidence-capacity-workflow-v1",
        "purpose": ARTIFACT_PURPOSE,
        "policy": POLICY,
        "research_only": True,
        "test_read": False,
        "billing_stopped": None,
        "complete": False,
        "manifest": {"path": str(manifest), "sha256": _sha256_file(manifest)},
        "cache_identity": cache_identity,
        "configs": configs,
        "source_hashes": source_hashes,
        "runner_provenance": {
            "module": "bench.quality_confidence_workflow",
            "sha256": source_hashes["bench/quality_confidence_workflow.py"],
            "configs": {
                name: {
                    "path": value["path"],
                    "sha256": value["sha256"],
                    "identity_config_sha256": value["identity_config_sha256"],
                }
                for name, value in configs.items()
            },
            "data_identity": cache_identity,
        },
        "scoped_provenance": {
            "audit_scope": "train_validation_only",
            "development_fitting": True,
            "test_read": False,
            "operations": ["confidence-smoke", "confidence-train", "confidence-export",
                           "quality_confidence_evaluate:train,validation"],
            "excluded_operations": ["confidence-resume", "promotion", "replay",
                                    "dataset-preparation", "test-evaluation"],
            "stage_runner": "bench.quality_offline.run_stage",
            "dataset_loader": "bench.quality_confidence_rejection.load_research_dataset",
        },
        "planned_stages": plans,
        "exact_commands": [
            {"stage": plan["name"], "command": plan["command"]} for plan in plans
        ],
        "stages": [],
        "required_artifacts": [],
        "current_stage": None,
        "failure": None,
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    # This guard deliberately precedes cache loading and all output access.
    _cloud_guard(args.cloud_execution)
    requested_output, output = _fresh_output_paths(args.output)

    manifest = args.manifest.resolve(strict=True)
    configs = _load_frozen_configs()
    dataset = load_research_dataset(manifest)
    plans = build_stage_plan(manifest, output)
    source_hashes = _source_hashes()
    summary = _aggregate_base(manifest, output, dataset, configs, plans, source_hashes)

    # Cache/config/manifest validation is complete; only now may output exist.
    _validate_output_target(requested_output)
    if output != requested_output:
        _validate_output_target(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / AGGREGATE_NAME
    started = time.monotonic()
    save_result(receipt_path, summary)
    try:
        for plan in plans:
            summary["current_stage"] = plan["name"]
            remaining = _require_deadline(started, f"before {plan['name']}")
            receipt = run_stage(
                plan["name"], plan["command"], output,
                timeout_s=min(plan["timeout_cap_seconds"], remaining),
                storage_limit_bytes=OUTPUT_LIMIT_BYTES,
                storage_root=output,
                output_limit_bytes=OUTPUT_LIMIT_BYTES,
            )
            summary["stages"].append(receipt)
            _require_deadline(started, f"after {plan['name']}")
            save_result(receipt_path, summary)
        summary["current_stage"] = "artifact-validation"
        summary["required_artifacts"] = _validate_required_outputs(
            output, configs, dataset["identity"], source_hashes)
        _require_deadline(started, "after artifact validation")
        if (
            len(summary["stages"]) != 10
            or any(stage.get("status") != "complete" for stage in summary["stages"])
        ):
            raise RuntimeError("All ten subprocess stages must complete")
        summary["complete"] = True
        summary["current_stage"] = None
    except BaseException as exc:
        summary["complete"] = False
        summary["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "current_stage": summary["current_stage"],
        }
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        save_result(receipt_path, summary)
        print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-execution", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        _, output = _fresh_output_paths(args.output)
        plan = {
            "schema_version": 1,
            "kind": "portable-confidence-capacity-workflow-dry-run-v1",
            "executable": False,
            "reason": (
                "Dry run only: execute these commands through this workflow on an approved "
                "Linux cloud host with --cloud-execution after audited cache validation."
            ),
            "test_read": False,
            "output_created": False,
            "stages": build_stage_plan(args.manifest, output),
        }
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return plan
    return _execute(args)


if __name__ == "__main__":
    main()
