"""Freeze the broader synthetic experiment using already acquired cloud assets.

No download, rendering, model inference or optimizer runs in this command.
The first fixed collection is exploratory synthetic data, not real-video
accuracy evidence. Raw orientation labels remain conservative until a separate
rendered-RGB observability review is recorded before candidate generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


ASSETS = (
    (17, "train", 5975664,
     "e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc"),
    (27, "validation", 19341132,
     "b5354f0785c6a4b97dcadb3fe2622e9dc7e2c794bd69d648b72482dc7bdb8e0a"),
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
    from bench.quality_ranker_synthetic import seal_recipe, validate_recipe

    assets = args.assets.resolve(strict=True)
    if not assets.is_dir() or args.assets.is_symlink():
        raise ValueError("An existing, nonsymlink cloud asset directory is required")
    asset_receipts = []
    records, sequences = [], []
    for number, split, size, digest in ASSETS:
        path = assets / f"obj_{number:06d}.glb"
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or changed pinned metric asset: {path.name}")
        if _file_sha256(path) != digest:
            raise ValueError(f"Metric asset hash differs from acquisition pin: {path.name}")
        physical_id = f"hot3d:obj_{number:06d}"
        asset_receipts.append({"path": str(path), "bytes": size, "sha256": digest,
                               "physical_object_id": physical_id, "split": split})
        for yaw in (0, 90, 180, 270):
            for pitch in (-25, 25):
                view = f"yaw{yaw}-pitch{'m' if pitch < 0 else 'p'}25"
                sequence_id = f"ranker-multiview-hot3d{number}-v1-{view}"
                scene_id = f"synthetic-scene:hot3d{number}:multiview-v1:{view}"
                common = dict(physical_object_id=physical_id, sequence_id=sequence_id,
                              scene_id=scene_id, split=split, mesh_sha256=digest)
                sequences.append({**common, "yaw0_deg": yaw, "pitch_deg": pitch,
                                  "observationally_ambiguous": True})
                for ordinal in range(24):
                    records.append({**common, "frame_id": ordinal,
                                    "source_frame_id": ordinal,
                                    "background_id": "procedural:gray",
                                    "background_kind": "procedural",
                                    "provenance_kind": "synthetic"})
    # Does training indirectly contain test information? No reserved coffee-pot
    # mesh, renders, crops or backgrounds are read. Train and validation meshes,
    # physical identities, scenes and complete sequences have distinct owners.
    policy = {"max_rotation_error_deg": 10.0, "max_translation_error_m": 0.02}
    inventory = {"records": records}
    frozen = freeze_split_manifest(inventory, label_policy=policy)
    output = args.output_dir
    if output.is_symlink():
        raise ValueError("Preparation destination must not be a symlink")
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(output / "inventory.json", inventory)
        _write_json(output / "split.json", frozen)
        recipe = seal_recipe({
            "schema_version": 1, "kind": "quality-ranker-synthetic-recipe-v1",
            "recipe_id": "ranker-broader-synthetic-v1",
            "split_binding": {"manifest_sha256": frozen["manifest_sha256"],
                              "file_sha256": _file_sha256(output / "split.json")},
            "options": {
                "frames": 24, "width": 640, "height": 480, "fps": 10.0,
                "seed": 0, "mesh_units": "metres", "background_rgb": [127, 127, 127],
                "intrinsics": [[576, 0, 319.5], [0, 576, 239.5], [0, 0, 1]],
                "trajectory": {"rotation_order": "RyRx", "yaw_sweep_deg": 40.0,
                               "roll_deg": 0.0, "yaw_origins_deg": [0, 90, 180, 270],
                               "pitch_deg": [-25, 25]},
                "occluder": {"frames_inclusive": [6, 13], "width_fraction": 0.22,
                             "height_fraction": 0.70, "rgb": [32, 32, 32],
                             "rounding": "half_up", "placement_rounding": "floor_plus_half",
                             "center_x_fractions": [0.2, 0.8], "center_y_fraction": 0.5},
                "limits": {"max_output_bytes": 268435456, "max_video_bytes": 67108864},
                "label_policy": frozen["label_policy"],
            },
            "sequences": sequences,
        })
        _write_json(output / "recipe.json", recipe)
        asset_by_identity = {row["physical_object_id"]: Path(row["path"])
                             for row in asset_receipts}
        for sequence in sequences:
            validate_recipe(output / "recipe.json", output / "split.json",
                            asset_by_identity[sequence["physical_object_id"]],
                            sequence["sequence_id"], frames=24, width=640,
                            height=480, fps=10.0, mesh_units="metres",
                            observationally_ambiguous=True)
        _write_json(output / "preparation.json", {
            "status": "frozen_before_rendering", "assets": asset_receipts,
            "frames": 384, "sequences": 16,
            "split_sha256": frozen["manifest_sha256"],
            "recipe_sha256": recipe["recipe_sha256"],
            "operator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "resource_ceilings": {"gpu_seconds": 5400, "existing_credits": 1.0,
                                  "model_cache_bytes": 8 * 1024**3,
                                  "aggregate_new_data_results_bytes": 1024**3},
            "reserve": {"physical_object_id": "hot3d:obj_000007",
                        "downloaded": False, "rendered": False, "evaluated": False},
            "neural_execution_started": False, "training_started": False,
            "orientation_review_required_before_candidate_generation": True,
            "billing_shutdown_verified": None,
        })
    except Exception as error:
        _write_json(output / "incomplete.json", {"status": "incomplete",
                                                 "reason": str(error)})
        raise
    print(json.dumps({"status": "frozen", "output": str(output),
                      "recipe_sha256": recipe["recipe_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
