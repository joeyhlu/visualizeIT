"""Preparation software fixtures only: no renders, datasets, models or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .quality_ranker_data import read_split_manifest
from .quality_ranker_synthetic import validate_recipe


class CloudPreparationTests(unittest.TestCase):
    def _module(self):
        path = Path(__file__).resolve().parents[1] / "scripts/cloud_ranker_prepare_multiview.py"
        spec = importlib.util.spec_from_file_location("cloud_preparation_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_windows_guard_precedes_data_preparation(self):
        module = self._module()
        with patch.object(module.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "approved Linux cloud"):
                module.main(["--assets", "absent-fixture-assets", "--output-dir", "absent-fixture-output"])

    def test_fixed_collection_round_trips_strict_recipe_and_retains_test_reserve(self):
        module = self._module()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            # Tiny arbitrary bytes exercise hash binding, not a metric-mesh claim.
            specs = []
            for number, split in ((17, "train"), (27, "validation")):
                payload = f"software fixture {number}".encode()
                (assets / f"obj_{number:06d}.glb").write_bytes(payload)
                specs.append((number, split, len(payload), hashlib.sha256(payload).hexdigest()))
            output = root / "frozen"
            with patch.object(module, "ASSETS", tuple(specs)), \
                    patch.object(module.platform, "system", return_value="Linux"), \
                    patch.dict(module.os.environ, {"VISUALIZEIT_CLOUD": "1"}):
                module.main(["--assets", str(assets), "--output-dir", str(output)])
            split = read_split_manifest(output / "split.json")
            self.assertEqual(len(split["groups"]), 16)
            self.assertEqual(sum(len(g["frames"]) for g in split["groups"]), 384)
            self.assertEqual({g["physical_object_id"] for g in split["groups"]},
                             {"hot3d:obj_000017", "hot3d:obj_000027"})
            for group in split["groups"]:
                number = 17 if group["split"] == "train" else 27
                context = validate_recipe(
                    output / "recipe.json", output / "split.json",
                    assets / f"obj_{number:06d}.glb", group["sequence_id"],
                    frames=24, width=640, height=480, fps=10.0,
                    mesh_units="metres", observationally_ambiguous=True)
                self.assertEqual(context["group"], group)
            receipt = json.loads((output / "preparation.json").read_text())
            self.assertFalse(receipt["training_started"])
            self.assertFalse(receipt["neural_execution_started"])
            self.assertEqual(receipt["reserve"]["physical_object_id"], "hot3d:obj_000007")
            self.assertFalse(receipt["reserve"]["downloaded"])
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"inventory.json", "split.json", "recipe.json", "preparation.json"})
            with patch.object(module, "ASSETS", tuple(specs)), \
                    patch.object(module.platform, "system", return_value="Linux"), \
                    patch.dict(module.os.environ, {"VISUALIZEIT_CLOUD": "1"}):
                with self.assertRaises(FileExistsError):
                    module.main(["--assets", str(assets), "--output-dir", str(output)])


if __name__ == "__main__":
    unittest.main()
