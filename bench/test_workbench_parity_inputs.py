"""Admission must authenticate the bytes it executes, before any network loads."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from bench.vision import cv2
from bench import workbench_neural as neural


class ParityInputTests(unittest.TestCase):
    def attempt(self, directory, alteration):
        root = Path(directory)
        rgb = np.full((6, 8, 3), 30, np.uint8)
        mask = np.full((6, 8), 255, np.uint8)
        rp, mp, gp = root / "rgb.png", root / "mask.png", root / "mesh.glb"
        cv2.imwrite(str(rp), rgb)
        cv2.imwrite(str(mp), mask)
        gp.write_bytes(b"pinned geometry fixture")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        row = {"case_id": "fixture", "family": "fixture", "rgb_path": str(rp), "mask_path": str(mp),
               "rgb_sha256": digest(rp), "mask_sha256": digest(mp), "glb": {"sha256": digest(gp)}}
        case = {**row, "rgb": rgb.copy(), "mask": mask > 127, "glb_path": str(gp)}
        packages = neural.runtime_packages()
        if alteration == "rgb":
            case["rgb"][0, 0] = 0
        elif alteration == "mask":
            case["mask"][0, 0] = False
        elif alteration == "glb":
            gp.write_bytes(b"different geometry")
        elif alteration == "runtime":
            packages = {**packages, "torch": "changed-runtime"}
        contract = neural.WorkbenchContract({"cases": [row], "families": ["fixture"],
                                              "runtime_packages": packages}, "a" * 64)
        loader = Mock(side_effect=AssertionError("network must never load for invalid inputs"))
        with self.assertRaises(neural.AdmissionError):
            neural.run_fullimage_head_parity([case], contract=contract, head_path=root / "absent.pt",
                reference_path=root / "absent-best.pt", output_dir=root / "parity",
                torch_module=object(), base_loader=loader)
        loader.assert_not_called()

    def test_wrong_decoded_rgb_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.attempt(d, "rgb")

    def test_wrong_decoded_mask_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.attempt(d, "mask")

    def test_changed_executed_glb_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.attempt(d, "glb")

    def test_changed_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.attempt(d, "runtime")


if __name__ == "__main__":
    unittest.main()
