"""Freeze one HOT3D17 single-view initialization input from an existing GLB.

This command performs no download, rendering, inference, training or model
loading. It freezes only the train inventory, split, recipe and provenance
needed to render one new RGB frame for a five-candidate initialization trial.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


ASSET = (
    17,
    "train",
    5975664,
    "e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc",
)
SEQUENCE_ID = "ranker-single-view-hot3d17-v1-yaw75-pitchm15"
SCENE_ID = "synthetic-scene:hot3d17:single-view-v1:yaw75-pitchm15"
RECIPE_ID = "ranker-single-view-synthetic-v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="existing cloud folder containing obj_000017.glb")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="fresh destination for the frozen preparation")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or os.environ.get("VISUALIZEIT_CLOUD") != "1":
        raise RuntimeError("Preparation runs only in the approved Linux cloud workspace")

    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    from bench.quality_ranker_data import freeze_split_manifest, _write_json, _file_sha256
    from bench.quality_ranker_synthetic import seal_recipe, validate_single_view_recipe

    if args.assets.is_symlink():
        raise ValueError("An existing, nonsymlink cloud asset directory is required")
    assets = args.assets.resolve(strict=True)
    if not assets.is_dir():
        raise ValueError("An existing, nonsymlink cloud asset directory is required")
    number, split_name, expected_size, expected_digest = ASSET
    mesh_path = assets / f"obj_{number:06d}.glb"
    if (mesh_path.is_symlink() or not mesh_path.is_file() or
            mesh_path.stat().st_size != expected_size):
        raise ValueError(f"Missing or changed pinned metric asset: {mesh_path.name}")
    if _file_sha256(mesh_path) != expected_digest:
        raise ValueError(f"Metric asset hash differs from acquisition pin: {mesh_path.name}")

    physical_id = f"hot3d:obj_{number:06d}"
    common = {
        "physical_object_id": physical_id,
        "sequence_id": SEQUENCE_ID,
        "scene_id": SCENE_ID,
        "split": split_name,
        "mesh_sha256": expected_digest,
    }
    inventory = {"records": [{
        **common,
        "frame_id": 0,
        "source_frame_id": 0,
        "background_id": "procedural:gray",
        "background_kind": "procedural",
        "provenance_kind": "synthetic",
    }]}
    policy = {"max_rotation_error_deg": 10.0,
              "max_translation_error_m": 0.02}
    frozen = freeze_split_manifest(inventory, label_policy=policy)
    sequence = {
        **common,
        "yaw0_deg": 75,
        "pitch_deg": -15,
        "observationally_ambiguous": True,
    }

    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Single-view preparation requires a fresh, non-symlink destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    claim_path = output.with_name(output.name + ".claim")
    descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
        _write_json(stage / "inventory.json", inventory)
        _write_json(stage / "split.json", frozen)
        source_binding = {
            "generator_path": "bench/quality_ranker_synthetic.py",
            "generator_sha256": _file_sha256(source_root / "bench" / "quality_ranker_synthetic.py"),
            "preparation_path": "scripts/cloud_ranker_prepare_single_view.py",
            "preparation_sha256": _file_sha256(Path(__file__).resolve()),
        }
        recipe = seal_recipe({
            "schema_version": 1,
            "kind": "quality-ranker-synthetic-single-view-recipe-v1",
            "recipe_id": RECIPE_ID,
            "source_binding": source_binding,
            "split_binding": {
                "manifest_sha256": frozen["manifest_sha256"],
                "file_sha256": _file_sha256(stage / "split.json"),
            },
            "options": {
                "frames": 1,
                "width": 640,
                "height": 480,
                "fps": 10.0,
                "seed": 0,
                "mesh_units": "metres",
                "background_rgb": [127, 127, 127],
                "intrinsics": [[576, 0, 319.5], [0, 576, 239.5], [0, 0, 1]],
                "trajectory": {
                    "rotation_order": "RyRx",
                    "frame_index": 0,
                    "yaw0_deg": 75,
                    "pitch_deg": -15,
                    "yaw_sweep_deg": 20.0,
                    "roll_deg": 0.0,
                },
                "occluder": None,
                "limits": {
                    "max_output_bytes": 16 * 1024**2,
                    "max_video_bytes": 4 * 1024**2,
                },
                "label_policy": frozen["label_policy"],
            },
            "sequences": [sequence],
        })
        _write_json(stage / "recipe.json", recipe)
        validate_single_view_recipe(
            stage / "recipe.json", stage / "split.json", mesh_path, SEQUENCE_ID,
            frames=1, width=640, height=480, fps=10.0, mesh_units="metres",
            observationally_ambiguous=True)
        _write_json(stage / "preparation.json", {
            "schema_version": 1,
            "status": "frozen_before_rendering",
            "kind": "quality-ranker-single-view-preparation-v1",
            "asset": {
                "path": str(mesh_path),
                "bytes": expected_size,
                "sha256": expected_digest,
                "physical_object_id": physical_id,
                "split": split_name,
            },
            "frames": 1,
            "sequences": 1,
            "splits": ["train"],
            "split_sha256": frozen["manifest_sha256"],
            "recipe_sha256": recipe["recipe_sha256"],
            "source_binding": source_binding,
            "raw_labels_observationally_ambiguous": True,
            "raw_labels_in_inference_bundle": False,
            "planned_initialization_candidates": 5,
            "candidate_generation_started": False,
            "witness_labels_created": False,
            "training_started": False,
            "quality_claim_created": False,
            "gpu_processes_started": 0,
            "new_model_weights_bytes": 0,
            "neural_execution_started": False,
        })
        if output.exists() or output.is_symlink():
            raise FileExistsError("Single-view preparation destination appeared during staging")
        os.replace(stage, output)
        stage = None
    except Exception as error:
        if stage is not None and stage.exists():
            try:
                _write_json(stage / "incomplete.json", {
                    "schema_version": 1,
                    "status": "incomplete",
                    "reason": str(error)[:512],
                    "promoted": False,
                })
            except Exception:
                pass
        raise
    finally:
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass
    print(json.dumps({"status": "frozen", "output": str(output),
                      "recipe_sha256": recipe["recipe_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
