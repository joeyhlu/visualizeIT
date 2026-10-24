"""Freeze the bounded paired-view orientation study from existing cloud GLBs.

This command performs no download, rendering, inference, training or model
loading. A separate blind rendered-RGB review is required before candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


ASSETS = (
    (17, "train", 5975664,
     "e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc"),
    (27, "validation", 19341132,
     "b5354f0785c6a4b97dcadb3fe2622e9dc7e2c794bd69d648b72482dc7bdb8e0a"),
)

SCHEDULE = (
    (17, "train", 45, -15, "yaw45-pitchm15"),
    (27, "validation", 45, -15, "yaw45-pitchm15"),
    (17, "train", 225, 15, "yaw225-pitchp15"),
    (27, "validation", 225, 15, "yaw225-pitchp15"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="existing cloud folder containing the two pinned GLBs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or os.environ.get("VISUALIZEIT_CLOUD") != "1":
        raise RuntimeError("Preparation runs only in the approved Linux cloud workspace")

    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    from bench.quality_ranker_data import freeze_split_manifest, _write_json, _file_sha256
    from bench.quality_ranker_synthetic import seal_recipe, validate_orientation_recipe

    if args.assets.is_symlink():
        raise ValueError("An existing, nonsymlink cloud asset directory is required")
    assets = args.assets.resolve(strict=True)
    if not assets.is_dir():
        raise ValueError("An existing, nonsymlink cloud asset directory is required")
    asset_receipts = []
    asset_by_number = {}
    for number, split, size, digest in ASSETS:
        path = assets / f"obj_{number:06d}.glb"
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or changed pinned metric asset: {path.name}")
        if _file_sha256(path) != digest:
            raise ValueError(f"Metric asset hash differs from acquisition pin: {path.name}")
        physical_id = f"hot3d:obj_{number:06d}"
        asset_receipts.append({"path": str(path), "bytes": size, "sha256": digest,
                               "physical_object_id": physical_id, "split": split})
        asset_by_number[number] = {"path": path, "digest": digest, "split": split}

    records, sequences = [], []
    for number, split, yaw, pitch, suffix in SCHEDULE:
        asset = asset_by_number.get(number)
        if asset is None or asset["split"] != split:
            raise ValueError("Orientation schedule and pinned asset ownership differ")
        physical_id = f"hot3d:obj_{number:06d}"
        sequence_id = f"ranker-orientation-hot3d{number}-v1-{suffix}"
        scene_id = f"synthetic-scene:hot3d{number}:orientation-v1:{suffix}"
        common = {"physical_object_id": physical_id, "sequence_id": sequence_id,
                  "scene_id": scene_id, "split": split,
                  "mesh_sha256": asset["digest"]}
        sequences.append({**common, "yaw0_deg": yaw, "pitch_deg": pitch,
                          "observationally_ambiguous": True})
        for ordinal in range(12):
            records.append({**common, "frame_id": ordinal,
                            "source_frame_id": ordinal,
                            "background_id": "procedural:gray",
                            "background_kind": "procedural",
                            "provenance_kind": "synthetic"})

    policy = {"max_rotation_error_deg": 10.0,
              "max_translation_error_m": 0.02}
    inventory = {"records": records}
    frozen = freeze_split_manifest(inventory, label_policy=policy)

    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Orientation preparation requires a fresh, non-symlink destination")
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
            "preparation_path": "scripts/cloud_ranker_prepare_orientation.py",
            "preparation_sha256": _file_sha256(Path(__file__).resolve()),
        }
        recipe = seal_recipe({
            "schema_version": 1,
            "kind": "quality-ranker-synthetic-orientation-recipe-v1",
            "recipe_id": "ranker-orientation-synthetic-v1",
            "source_binding": source_binding,
            "split_binding": {"manifest_sha256": frozen["manifest_sha256"],
                              "file_sha256": _file_sha256(stage / "split.json")},
            "options": {
                "frames": 12, "width": 640, "height": 480, "fps": 10.0,
                "seed": 0, "mesh_units": "metres", "background_rgb": [127, 127, 127],
                "intrinsics": [[576, 0, 319.5], [0, 576, 239.5], [0, 0, 1]],
                "trajectory": {"rotation_order": "RyRx", "yaw_sweep_deg": 20.0,
                               "roll_deg": 0.0,
                               "paired_views": [
                                   {"yaw0_deg": 45, "pitch_deg": -15},
                                   {"yaw0_deg": 225, "pitch_deg": 15},
                               ]},
                "occluder": None,
                "limits": {"max_output_bytes": 32 * 1024**2,
                           "max_video_bytes": 8 * 1024**2},
                "label_policy": frozen["label_policy"],
            },
            "sequences": sequences,
        })
        _write_json(stage / "recipe.json", recipe)
        for sequence in sequences:
            number = 17 if sequence["physical_object_id"].endswith("000017") else 27
            validate_orientation_recipe(
                stage / "recipe.json", stage / "split.json",
                asset_by_number[number]["path"], sequence["sequence_id"],
                frames=12, width=640, height=480, fps=10.0,
                mesh_units="metres", observationally_ambiguous=True)
        _write_json(stage / "preparation.json", {
            "status": "frozen_before_rendering",
            "assets": asset_receipts,
            "frames": 48,
            "sequences": 4,
            "split_sha256": frozen["manifest_sha256"],
            "recipe_sha256": recipe["recipe_sha256"],
            "source_binding": source_binding,
            "external_budget_constraints": {
                "enforced_by": "parent",
                "wall_seconds": 45 * 60,
                "aggregate_new_data_results_bytes": 256 * 1024**2,
                "model_cache_bytes": 8 * 1024**3,
                "credit_floor": 3.71,
            },
            "gpu_processes_started": 0,
            "new_model_weights_bytes": 0,
            "neural_execution_started": False,
            "training_started": False,
            "orientation_review_required_before_candidate_generation": True,
            "orientation_review_created": False,
            "raw_labels_observationally_ambiguous": True,
        })
        if output.exists() or output.is_symlink():
            raise FileExistsError("Orientation preparation destination appeared during staging")
        os.replace(stage, output)
        stage = None
    except Exception as error:
        if stage is not None and stage.exists():
            try:
                _write_json(stage / "incomplete.json", {
                    "status": "incomplete", "reason": str(error)[:512],
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
