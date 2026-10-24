"""Model-free regression tests for the full-scan accounting facade."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import kaggle_adaptation as adapter
from scripts import kaggle_full_scan_accounting_v1 as accounting
from scripts import kaggle_supervision_v1 as supervision


ROOT = Path(__file__).resolve().parents[1]
# The facade authenticates this immutable scanner by file hash. Accounting
# tests never call it, so keep the suite independent of its numerical imports.
scanner = SimpleNamespace(__file__=str(ROOT / "bench/quality_adaptation_portable.py"))


class FullScanAccountingTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".cache/full-scan-accounting-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.facade = accounting.FullScanFacade(adapter, scanner)

    def symlink(self, link: Path, target: Path | str, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"filesystem symlinks unavailable: {error}")

    def test_facade_retains_original_authentication_and_supervision_composition(self):
        self.assertIsInstance(self.facade, supervision.Facade)
        self.assertIs(self.facade.adapter, adapter)
        self.assertIs(self.facade.scanner, scanner)

    @unittest.skipUnless(accounting._POSIX_NOFOLLOW, "POSIX no-follow descriptor traversal is unavailable")
    def test_posix_no_follow_fast_path_is_active(self):
        self.assertTrue(accounting._POSIX_NOFOLLOW, "Linux tests must exercise the accelerated descriptor path")

    @unittest.skipUnless(accounting._POSIX_NOFOLLOW, "POSIX no-follow descriptor traversal is unavailable")
    def test_child_directory_descriptor_closes_when_fstat_fails(self):
        with (
            patch.object(accounting.os, "open", return_value=12345),
            patch.object(accounting.os, "fstat", side_effect=OSError("injected fstat failure")),
            patch.object(accounting.os, "close") as close,
        ):
            with self.assertRaisesRegex(OSError, "injected fstat failure"):
                accounting._open_child_directory(7, "child", self.root / "child", expected=None)
        close.assert_called_once_with(12345)

    def test_full_union_matches_original_for_nested_repeated_hardlinked_and_missing_roots(self):
        nested = self.root / "nested"
        nested.mkdir()
        first = self.root / "first.bin"
        first.write_bytes(b"12345")
        second = nested / "second.bin"
        second.write_bytes(b"abc")
        distinct_same_size = self.root / "same-size.bin"
        distinct_same_size.write_bytes(b"67890")
        hardlink = self.root / "hardlink.bin"
        try:
            os.link(first, hardlink)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"hard links unavailable: {error}")

        roots = (self.root, nested, first, self.root, self.root / "missing")
        expected = adapter.unique_tree_bytes(roots)
        actual = self.facade.unique_tree_bytes(roots)
        self.assertEqual(actual, expected, f"new full scan returned {actual}; original returned {expected}")

    def test_each_call_observes_new_files_without_snapshot_caching(self):
        root = self.root / "data"
        root.mkdir()
        (root / "first").write_bytes(b"123")
        first = self.facade.unique_tree_bytes((root,))
        (root / "added").write_bytes(b"4567")
        second = self.facade.unique_tree_bytes((root,))
        self.assertEqual(first, 3, f"first full scan returned {first}; fixture contains 3 bytes")
        self.assertEqual(second, 7, f"second full scan returned {second}; fixture now contains 7 bytes")

    def test_runtime_policy_and_accounting_keep_internal_file_and_directory_aliases(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        target_file = runtime / "python-real"
        target_file.write_bytes(b"12345")
        self.symlink(runtime / "python", "python-real")
        package = runtime / "package"
        (package / "nested").mkdir(parents=True)
        (package / "nested" / "module.bin").write_bytes(b"abcdefgh")
        self.symlink(runtime / "package-alias", "package", directory=True)
        receipt = {
            "runtime_root": str(runtime),
            "runtime_links": [
                {"path": "python", "target": "python-real"},
                {"path": "package-alias", "target": "package"},
            ],
        }
        original_policy = adapter.runtime_link_policy(receipt, self.root)
        fast_policy = self.facade.runtime_link_policy(receipt, self.root)
        self.assertEqual(fast_policy, original_policy, f"runtime policy differs: {fast_policy!r} != {original_policy!r}")
        expected = adapter.unique_tree_bytes((runtime,), approved_links=original_policy)
        actual = self.facade.unique_tree_bytes((runtime,), approved_links=fast_policy)
        self.assertEqual(actual, expected, f"alias-aware total differs: {actual} != {expected}")
        self.assertEqual(actual, 13, f"aliases must count their file targets once; got {actual}")

    def test_runtime_inventory_rejects_added_unapproved_alias(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        (runtime / "target").write_bytes(b"x")
        receipt = {"runtime_root": str(runtime), "runtime_links": []}
        self.symlink(runtime / "extra", "target")
        with self.assertRaisesRegex(adapter.ContractError, "unapproved runtime links"):
            self.facade.runtime_link_policy(receipt, self.root)

    def test_runtime_policy_rejects_escaping_cycle_and_pinned_target_mismatch(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        escaping = runtime / "escape"
        self.symlink(escaping, "../outside")
        receipt = {"runtime_root": str(runtime), "runtime_links": [{"path": "escape", "target": "../outside"}]}
        with self.assertRaisesRegex(adapter.ContractError, "escapes runtime root"):
            self.facade.runtime_link_policy(receipt, self.root)
        escaping.unlink()

        cycle = runtime / "cycle"
        self.symlink(cycle, "cycle")
        cyclic_receipt = {"runtime_root": str(runtime), "runtime_links": [{"path": "cycle", "target": "cycle"}]}
        with self.assertRaisesRegex(adapter.ContractError, "broken/cyclic runtime link"):
            self.facade.runtime_link_policy(cyclic_receipt, self.root)
        cycle.unlink()

        target = runtime / "real"
        target.write_bytes(b"data")
        alias = runtime / "alias"
        self.symlink(alias, "real")
        wrong_pin = {"runtime_root": str(runtime), "runtime_links": [{"path": "alias", "target": "different"}]}
        with self.assertRaisesRegex(adapter.ContractError, "runtime link inventory changed"):
            self.facade.runtime_link_policy(wrong_pin, self.root)

    def test_authenticated_path_replacement_is_rejected_by_byte_scan(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        (runtime / "first").write_bytes(b"123")
        (runtime / "second").write_bytes(b"4567")
        alias = runtime / "alias"
        self.symlink(alias, "first")
        receipt = {"runtime_root": str(runtime), "runtime_links": [{"path": "alias", "target": "first"}]}
        approved = self.facade.runtime_link_policy(receipt, self.root)
        alias.unlink()
        alias.symlink_to("second")
        with self.assertRaisesRegex(adapter.ContractError, "unapproved accounting symlink"):
            self.facade.unique_tree_bytes((runtime,), approved_links=approved)

    def test_missing_approved_alias_target_is_rejected(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        alias = runtime / "alias"
        self.symlink(alias, "vanished")
        receipt = {"runtime_root": str(runtime), "runtime_links": [{"path": "alias", "target": "vanished"}]}
        with self.assertRaisesRegex(adapter.ContractError, "broken/cyclic runtime link"):
            self.facade.runtime_link_policy(receipt, self.root)

    def test_fifo_remains_a_nonregular_resident_path_error(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable on this platform")
        with tempfile.TemporaryDirectory(prefix="kaggle-accounting-fifo-") as temporary:
            root = Path(temporary)
            try:
                os.mkfifo(root / "pipe")
            except OSError as error:
                self.skipTest(f"temporary filesystem cannot create FIFO fixtures: {error}")
            with self.assertRaisesRegex(adapter.ContractError, "nonregular resident path"):
                self.facade.unique_tree_bytes((root,))

    def test_file_not_found_retries_the_entire_byte_snapshot(self):
        root = self.root / "data"
        root.mkdir()
        (root / "payload").write_bytes(b"123456")
        original_scandir = os.scandir
        calls = []

        def churn_once(path):
            calls.append(path)
            if len(calls) == 1:
                raise FileNotFoundError("simulated directory replacement")
            return original_scandir(path)

        with patch.object(accounting.os, "scandir", side_effect=churn_once):
            actual = self.facade.unique_tree_bytes((root,))
        self.assertEqual(actual, 6, f"retry scan returned {actual}; complete tree contains 6 bytes")
        self.assertEqual(len(calls), 2, f"expected one complete retry, observed {len(calls)} scandir calls")

    def test_runtime_policy_retries_whole_inventory_after_disappearing_directory(self):
        runtime = self.root / "runtime"
        (runtime / "nested").mkdir(parents=True)
        (runtime / "nested" / "file").write_bytes(b"x")
        receipt = {"runtime_root": str(runtime), "runtime_links": []}
        original_scandir = os.scandir
        calls = []

        def disappear_once(path):
            calls.append(path)
            if len(calls) == 1:
                raise FileNotFoundError("simulated disappearing runtime directory")
            return original_scandir(path)

        with patch.object(accounting.os, "scandir", side_effect=disappear_once):
            self.assertEqual(self.facade.runtime_link_policy(receipt, self.root), {})
        # One failed root attempt is followed by a complete root + nested-directory walk.
        self.assertEqual(len(calls), 3, f"expected one complete inventory retry, observed {len(calls)} scandir calls")

    def test_permission_errors_are_not_retried(self):
        root = self.root / "data"
        root.mkdir()
        calls = []

        def denied(_path):
            calls.append(True)
            raise PermissionError("denied")

        with patch.object(accounting.os, "scandir", side_effect=denied):
            with self.assertRaises(PermissionError):
                self.facade.unique_tree_bytes((root,))
        self.assertEqual(len(calls), 1, f"permission errors must fail on the first snapshot; calls={len(calls)}")

    def test_file_not_found_retry_is_bounded_at_three_complete_attempts(self):
        root = self.root / "data"
        root.mkdir()
        calls = []

        def always_missing(_path):
            calls.append(True)
            raise FileNotFoundError("persistent tree churn")

        with patch.object(accounting.os, "scandir", side_effect=always_missing):
            with self.assertRaises(FileNotFoundError):
                self.facade.unique_tree_bytes((root,))
        self.assertEqual(len(calls), 3, f"snapshot retry budget is 3; observed {len(calls)} attempts")

    def test_module_imports_without_loading_model_packages(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            "import scripts.kaggle_full_scan_accounting_v1; "
            "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, f"model-free import failed: {result.stderr or result.stdout}")

    def test_source_imports_no_model_modules(self):
        tree = ast.parse(Path(accounting.__file__).read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(imports & {"torch", "torchvision", "tensorflow", "jax"}, f"model imports found: {imports}")


if __name__ == "__main__":
    unittest.main()
