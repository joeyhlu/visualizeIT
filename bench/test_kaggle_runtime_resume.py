from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch
import zipfile

from scripts import kaggle_runtime_resume as resume


ROOT = Path(__file__).resolve().parents[1]


class LiveStorageTests(unittest.TestCase):
    def test_live_broken_link_is_counted_without_following_its_missing_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary) / "runtime"
            root.mkdir()
            wheel = Path(temporary) / "uv.whl"
            wheel.write_bytes(b"wheel")
            link = root / "python3-config"
            try:
                link.symlink_to("python3.11-config")
            except OSError as error:
                self.skipTest(f"host does not permit symlinks: {error}")
            # The actual Kaggle extraction stopped on this temporarily dangling link.
            self.assertEqual(resume.tree_bytes(root, wheel), 5 + link.lstat().st_size)
            with self.assertRaises(resume.ResumeError):
                resume.runtime_links(root)  # Final admission remains strict.

    def test_live_directory_link_does_not_scan_external_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary) / "runtime"
            root.mkdir()
            external = Path(temporary) / "external"
            external.mkdir()
            (external / "payload").write_bytes(b"x" * 100)
            wheel = Path(temporary) / "uv.whl"
            wheel.write_bytes(b"wheel")
            link = root / "outside"
            try:
                link.symlink_to(external, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"host does not permit symlinks: {error}")
            self.assertEqual(resume.tree_bytes(root, wheel), 5 + link.lstat().st_size)
            with self.assertRaises(resume.ResumeError):
                resume.runtime_links(root)

    def test_file_removed_between_listing_and_lstat_is_tolerated(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary) / "runtime"
            root.mkdir()
            wheel = Path(temporary) / "uv.whl"
            wheel.write_bytes(b"wheel")
            with patch.object(resume.os, "walk", return_value=iter([(str(root), [], ["renamed"])])):
                self.assertEqual(resume.tree_bytes(root, wheel), 5)

    def test_persistent_storage_inspection_failure_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary) / "runtime"
            root.mkdir()
            wheel = Path(temporary) / "uv.whl"
            wheel.write_bytes(b"wheel")
            original = Path.lstat
            def inaccessible(path, *args, **kwargs):
                if path == wheel:
                    raise PermissionError("fixture access denied")
                return original(path, *args, **kwargs)
            with patch.object(Path, "lstat", inaccessible):
                with self.assertRaises(resume.ResumeError):
                    resume.tree_bytes(root, wheel)

    def test_unreadable_tree_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary) / "runtime"
            root.mkdir()
            wheel = Path(temporary) / "uv.whl"
            wheel.write_bytes(b"wheel")
            def failed_walk(base, **kwargs):
                kwargs["onerror"](PermissionError(13, "fixture access denied", str(base)))
                return iter(())
            with patch.object(resume.os, "walk", side_effect=failed_walk):
                with self.assertRaises(resume.ResumeError):
                    resume.tree_bytes(root, wheel)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fake_uv_wheel(path: Path, binary: bytes = b"fake uv executable") -> dict[str, object]:
    dist = "uv-0.12.22.dist-info"
    metadata_name = f"{dist}/METADATA"
    record_name = f"{dist}/RECORD"
    script_name = "uv-0.12.22.data/scripts/uv"
    metadata = b"Metadata-Version: 2.1\nName: uv\nVersion: 0.12.22\n"
    record_io = io.StringIO(newline="")
    writer = csv.writer(record_io, lineterminator="\n")
    writer.writerow([script_name, resume._record_digest(binary), str(len(binary))])
    writer.writerow([metadata_name, resume._record_digest(metadata), str(len(metadata))])
    writer.writerow([record_name, "", ""])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(metadata_name, metadata)
        archive.writestr(script_name, binary)
        archive.writestr(record_name, record_io.getvalue())
    return {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
            "binary_bytes": len(binary), "binary_sha256": digest(binary),
            "script_member": script_name}


def load_unchanged_launcher():
    launcher_path = ROOT / "scripts" / "kaggle_adaptation.py"
    spec = importlib.util.spec_from_file_location("kaggle_adaptation_resume_test", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the existing launcher for compatibility test")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_are_rejected(self):
        with self.assertRaisesRegex(resume.ResumeError, "duplicate JSON key"):
            resume.strict_json_bytes(b'{"pid":1,"pid":2}', "fixture")
        with self.assertRaisesRegex(resume.ResumeError, "non-finite"):
            resume.strict_json_bytes(b'{"value":NaN}', "fixture")

    def test_parent_observed_failed_attempt_is_pinned(self):
        evidence = ROOT / ".cache" / "cloud-prepare" / "kaggle-runtime-first-attempt-actual-20261018.json"
        observed = resume.verify_failed_attempt(evidence)
        self.assertEqual(observed["pid"], 129)
        self.assertEqual(observed["exit_code"], 2)
        self.assertFalse(observed["training_started"])
        self.assertEqual(observed["new_optimizer_updates"], 0)


class UvWheelTests(unittest.TestCase):
    def test_real_record_hash_length_and_installed_binary_bytes_are_required(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            wheel = root / "uv.whl"
            expected = make_fake_uv_wheel(wheel)
            evidence = resume.authenticate_uv_wheel(
                wheel,
                expected_wheel_sha256=expected["sha256"],
                expected_wheel_bytes=expected["bytes"],
                expected_binary_sha256=expected["binary_sha256"],
                expected_binary_bytes=expected["binary_bytes"],
            )
            self.assertEqual(evidence["script_member"], expected["script_member"])
            self.assertEqual(evidence["script_sha256"], expected["binary_sha256"])
            self.assertEqual(evidence["record_sha256"], resume._record_digest(b"fake uv executable"))

            runtime = root / "runtime"
            binary = runtime / "uv-tool" / "bin" / "uv"
            binary.parent.mkdir(parents=True)
            binary_data = b"fake uv executable"
            binary.write_bytes(binary_data)
            metadata_path = runtime / "uv-tool" / "uv-0.12.22.dist-info" / "METADATA"
            record_path = metadata_path.with_name("RECORD")
            metadata_path.parent.mkdir(parents=True)
            metadata_data = b"Metadata-Version: 2.1\nName: uv\nVersion: 0.12.22\n"
            metadata_path.write_bytes(metadata_data)
            record_path.write_text(
                f"../bin/uv,{resume._record_digest(binary_data)},{len(binary_data)}\n",
                encoding="utf-8",
            )
            accepted = resume.inspect_partial_runtime(
                runtime, expected_binary_sha256=expected["binary_sha256"],
                expected_binary_bytes=expected["binary_bytes"],
            )
            self.assertEqual(accepted["uv_binary"]["sha256"], expected["binary_sha256"])
            self.assertEqual(
                resume.authenticate_installed_uv(
                    runtime, evidence,
                    expected_binary_sha256=expected["binary_sha256"],
                    expected_binary_bytes=expected["binary_bytes"],
                )["sha256"],
                expected["binary_sha256"],
            )

            with self.assertRaisesRegex(resume.ResumeError, "uv binary"):
                resume.authenticate_installed_uv(runtime, {
                    **evidence, "script_sha256": "0" * 64,
                })

    def test_version_string_cannot_replace_record_and_binary_authentication(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            wheel = root / "uv.whl"
            expected = make_fake_uv_wheel(wheel)
            with self.assertRaisesRegex(resume.ResumeError, "SHA-256"):
                resume.authenticate_uv_wheel(
                    wheel,
                    expected_wheel_sha256="0" * 64,
                    expected_wheel_bytes=expected["bytes"],
                    expected_binary_sha256=expected["binary_sha256"],
                    expected_binary_bytes=expected["binary_bytes"],
                )

    def test_corrupt_record_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            wheel = root / "uv.whl"
            expected = make_fake_uv_wheel(wheel)
            with zipfile.ZipFile(wheel) as source:
                members = {name: source.read(name) for name in source.namelist()}
            record_name = next(name for name in members if name.endswith(".dist-info/RECORD"))
            members[record_name] = b"not,a,valid,RECORD\n"
            with zipfile.ZipFile(wheel, "w") as target:
                for name, data in members.items():
                    target.writestr(name, data)
            with self.assertRaises(resume.ResumeError):
                resume.authenticate_uv_wheel(
                    wheel,
                    expected_wheel_sha256=digest(wheel.read_bytes()),
                    expected_wheel_bytes=wheel.stat().st_size,
                    expected_binary_sha256=expected["binary_sha256"],
                    expected_binary_bytes=expected["binary_bytes"],
                )


class PartialRuntimeTests(unittest.TestCase):
    def make_partial(self, root: Path) -> Path:
        binary = root / "uv-tool" / "bin" / "uv"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"pinned uv")
        return binary

    def test_only_uv_tool_partial_state_is_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            runtime = Path(temporary) / "runtime"
            binary = self.make_partial(runtime)
            accepted = resume.inspect_partial_runtime(
                runtime, expected_binary_sha256=digest(b"pinned uv"),
                expected_binary_bytes=len(b"pinned uv"),
            )
            self.assertEqual(accepted["uv_binary"]["path"], str(binary))

            (runtime / "venv").mkdir()
            with self.assertRaisesRegex(resume.ResumeError, "only the original uv-tool"):
                resume.inspect_partial_runtime(
                    runtime, expected_binary_sha256=digest(b"pinned uv"),
                    expected_binary_bytes=len(b"pinned uv"),
                )

    def test_symlinks_and_changed_uv_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            runtime = Path(temporary) / "runtime"
            self.make_partial(runtime)
            link = runtime / "uv-tool" / "extra-link"
            symlink_supported = True
            try:
                os.symlink(runtime / "uv-tool" / "bin" / "uv", link)
            except (OSError, NotImplementedError):
                symlink_supported = False
            if symlink_supported:
                with self.assertRaisesRegex(resume.ResumeError, "symlinks"):
                    resume.inspect_partial_runtime(
                        runtime, expected_binary_sha256=digest(b"pinned uv"),
                        expected_binary_bytes=len(b"pinned uv"),
                    )

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            runtime = Path(temporary) / "runtime"
            binary = self.make_partial(runtime)
            binary.write_bytes(b"changed uv")
            with self.assertRaisesRegex(resume.ResumeError, "hash differs"):
                resume.inspect_partial_runtime(
                    runtime, expected_binary_sha256=digest(b"pinned uv"),
                    expected_binary_bytes=len(b"changed uv"),
                )


class CommandContractTests(unittest.TestCase):
    def test_uv_version_accepts_actual_authenticated_binary_output(self):
        resume.verify_uv_version_output("uv 0.12.22 (x86_64-unknown-linux-gnu)\n")

    def test_uv_version_accepts_exact_plain_fixture_form(self):
        resume.verify_uv_version_output("uv 0.12.22\n")

    def test_uv_version_rejects_other_release_wrong_platform_and_suffix_garbage(self):
        invalid_outputs = (
            "uv 0.12.23 (x86_64-unknown-linux-gnu)\n",
            "uv 0.12.22 (aarch64-unknown-linux-gnu)\n",
            "uv 0.12.22 (x86_64-unknown-linux-gnu) extra\n",
            "uv 0.12.22 (x86_64-unknown-linux-gnu)\nextra\n",
        )
        for output in invalid_outputs:
            with self.subTest(output=output), self.assertRaisesRegex(
                    resume.ResumeError, "pinned 0.12.22 Linux version"):
                resume.verify_uv_version_output(output)

    def test_remaining_commands_preserve_exact_interpreter_package_and_source_pins(self):
        root = Path("/kaggle/working/visualizeit-pinned-runtime-v1")
        root18 = Path("/kaggle/working/visualizeit-reviewed-source-installer-smoke-20261018-v1/visualizeit-cloud-v18")
        uv = root / "uv-tool" / "bin" / "uv"
        managed = root / "python" / "cpython-3.11.13" / "bin" / "python3.11"
        training = root / "venv" / "bin" / "python"
        launcher = Path("/kaggle/working/visualizeit-prepared-source-v3-20261018/scripts/kaggle_adaptation.py")
        commands = resume.build_commands(uv, managed, training, root, root18, launcher)
        self.assertEqual([name for name, _, _ in commands], [
            "uv-version", "python-install", "python-find", "managed-python-check", "venv",
            "bootstrap-packages", "torch", "requirements", "pip-check", "pip-freeze", "runtime-probe",
        ])
        self.assertEqual(commands[1][1], [str(uv), "python", "install", "3.11.13"])
        self.assertIn("pip==24.3.1", commands[5][1])
        self.assertIn("torch==2.5.1+cu124", commands[6][1])
        self.assertIn("torchvision==0.20.1+cu124", commands[6][1])
        self.assertEqual(commands[7][1][-3:], [
            str(root18 / "bench/runtime/requirements-ranker.txt"), "--index-url", "https://pypi.org/simple",
        ])
        self.assertEqual(commands[9][1][-2:], ["freeze", "--all"])
        self.assertEqual(commands[10][1][-2:], ["--python", str(training)])

    def test_environment_strips_python_pip_uv_and_loader_injection(self):
        source = {
            "PATH": "/usr/bin", "PYTHONPATH": "/poison", "PYTHONHOME": "/poison",
            "PIP_INDEX_URL": "https://poison", "UV_EXTRA_INDEX_URL": "https://poison",
            "VIRTUAL_ENV": "/poison", "CONDA_PREFIX": "/poison", "LD_PRELOAD": "/poison",
            "LD_LIBRARY_PATH": "/poison", "KEEP_ME": "yes",
        }
        env = resume.clean_environment(source, Path("/kaggle/working/runtime"))
        self.assertEqual(env["KEEP_ME"], "yes")
        self.assertEqual(env["PATH"], "/usr/bin")
        for key in ("PYTHONPATH", "PYTHONHOME", "PIP_INDEX_URL", "UV_EXTRA_INDEX_URL",
                    "VIRTUAL_ENV", "CONDA_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PIP_NO_CACHE_DIR"], "1")
        self.assertEqual(env["UV_NO_TELEMETRY"], "1")

    def test_subprocess_spy_uses_argument_list_sanitized_env_and_no_shell(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            runtime = working / "runtime"
            runtime.mkdir()
            wheel = working / "uv.whl"
            wheel.write_bytes(b"wheel")
            output = runtime / "stdout.txt"
            log = runtime / "logs" / "cmd.log"
            captured: dict[str, object] = {}

            class CompletedProcess:
                pid = 900001
                returncode = 0

                def poll(self):
                    return 0

            def fake_popen(argv, **kwargs):
                captured.update(argv=argv, **kwargs)
                kwargs["stdout"].write(b"uv 0.12.22\n")
                return CompletedProcess()

            with patch.object(resume, "check_resources"):
                result = resume.run_command(
                    ["/runtime/bin/uv", "--version"], cwd=working,
                    env=resume.clean_environment({"PATH": "/bin", "PYTHONPATH": "bad"}, runtime),
                    log_path=log, stdout_path=output, deadline=time_monotonic() + 10,
                    runtime_root=runtime, wheel=wheel, working=working,
                    popen=fake_popen,
                )
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(captured["argv"], ["/runtime/bin/uv", "--version"])
            self.assertFalse(captured["shell"])
            self.assertNotIn("PYTHONPATH", captured["env"])
            self.assertEqual(output.read_text(encoding="utf-8"), "uv 0.12.22\n")

    def test_run_creates_authenticated_tmpdir_before_first_install_command(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            source_install = working / "source-install"
            root18 = source_install / "visualizeit-cloud-v18"
            (root18 / "bench" / "runtime").mkdir(parents=True)
            for relative in resume.REQUIREMENT_FILES:
                (root18 / relative).write_text("pinned fixture\n", encoding="utf-8")
            runtime = working / "runtime"
            (runtime / "uv-tool" / "bin").mkdir(parents=True)
            wheel = working / "uv.whl"
            wheel.write_bytes(b"wheel fixture")
            operator = resume.RuntimeResume.__new__(resume.RuntimeResume)
            operator.launcher = working / "kaggle_adaptation.py"
            operator.source_install = source_install
            operator.runtime_root = runtime
            operator.wheel = wheel
            operator.failed_attempt = working / "failed-attempt.json"
            operator.failed_attempt.write_text("{}", encoding="utf-8")
            operator.evidence = {
                "failure": {"sha256": "failed"}, "bootstrap": {"sha256": "bootstrap"},
                "launcher": {"sha256": "launcher"}, "wheel": {"sha256": "wheel"},
                "installed_uv": {"sha256": "uv"}, "partial": {"runtime_root": str(runtime)},
            }
            operator.args = SimpleNamespace()
            operator.clock = time_monotonic
            operator.working = working
            operator.root18 = root18
            operator.host_python = Path(sys.executable)
            operator.env = resume.clean_environment({"PATH": "/usr/bin"}, runtime)
            operator.uv = runtime / "uv-tool" / "bin" / "uv"
            operator.managed_python = runtime / "python" / "cpython-3.11.13" / "bin" / "python3.11"
            operator.training_python = runtime / "venv" / "bin" / "python"
            operator.history = []
            operator.requirement_hashes = {}
            operator.marker = runtime / "resume-started.json"
            operator.resume_receipt = runtime / "runtime-resume-receipt.json"
            operator.generic_receipt = runtime / "runtime-receipt.json"
            observed = {}

            def stop_after_first_command(argv, **kwargs):
                tmpdir = Path(kwargs["env"]["TMPDIR"])
                observed.update(argv=argv, tmpdir=tmpdir, exists=tmpdir.is_dir(), parent=tmpdir.parent)
                raise resume.ResumeError("intentional run-fixture stop")

            operator.command_runner = stop_after_first_command
            operator._source_verify = lambda: None
            with patch.object(resume, "check_resources"), \
                    patch.object(resume, "build_commands", return_value=[
                        ("uv-version", [str(operator.uv), "--version"], "uv-version.txt")
                    ]):
                with self.assertRaisesRegex(resume.ResumeError, "intentional run-fixture stop"):
                    operator.run()

            self.assertEqual(observed["argv"], [str(operator.uv), "--version"])
            self.assertTrue(observed["exists"])
            self.assertEqual(observed["tmpdir"], runtime / "tmp")
            self.assertEqual(observed["parent"], runtime)
            self.assertTrue(operator.resume_receipt.is_file())
            self.assertFalse(operator.generic_receipt.exists())


class ProcessGroupCleanupTests(unittest.TestCase):
    class Process:
        pid = 987654

        def __init__(self, returncode, polls=None):
            self.returncode = returncode
            self.wait_calls = []
            self.polls = list(polls or [])

        def poll(self):
            if self.polls:
                return self.polls.pop(0)
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            return self.returncode

    def test_exited_leader_does_not_hide_surviving_group_descendant(self):
        process = self.Process(0)
        group = {"alive": True}
        signals = []
        ticks = iter((0.0, 0.0, 0.06, 0.12, 0.13))

        def killpg(pgid, sig):
            self.assertEqual(pgid, process.pid)
            if sig == 0:
                if not group["alive"]:
                    raise ProcessLookupError
                return None
            signals.append(sig)
            if sig == resume.PROCESS_GROUP_SIGKILL:
                group["alive"] = False

        with patch.object(resume.os, "killpg", side_effect=killpg, create=True), \
                patch.object(resume.time, "monotonic", side_effect=lambda: next(ticks)), \
                patch.object(resume.time, "sleep", return_value=None):
            resume._kill_process_group(process, term_grace=0.1, kill_grace=0.1)

        self.assertEqual(signals, [resume.PROCESS_GROUP_SIGTERM, resume.PROCESS_GROUP_SIGKILL])
        self.assertFalse(group["alive"])
        self.assertTrue(process.wait_calls)

    def test_timeout_kills_surviving_group_after_leader_has_exited(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            runtime = working / "runtime"
            runtime.mkdir()
            wheel = working / "uv.whl"
            wheel.write_bytes(b"wheel")
            process = self.Process(None)
            group = {"alive": True}
            signals = []
            clock_state = {"values": [10.0, 11.0], "last": 11.0}

            def monotonic():
                if clock_state["values"]:
                    clock_state["last"] = clock_state["values"].pop(0)
                else:
                    clock_state["last"] += 0.06
                return clock_state["last"]

            def killpg(pgid, sig):
                self.assertEqual(pgid, process.pid)
                if sig == 0:
                    if not group["alive"]:
                        raise ProcessLookupError
                    return None
                signals.append(sig)
                if sig == resume.PROCESS_GROUP_SIGKILL:
                    group["alive"] = False
                    process.returncode = -9

            def popen(argv, **kwargs):
                return process

            with patch.object(resume, "check_resources"), \
                    patch.object(resume.os, "killpg", side_effect=killpg, create=True), \
                    patch.object(resume.time, "monotonic", side_effect=monotonic), \
                    patch.object(resume, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.1), \
                    patch.object(resume, "PROCESS_GROUP_KILL_GRACE_SECONDS", 0.1), \
                    patch.object(resume.time, "sleep", return_value=None):
                with self.assertRaisesRegex(resume.ResumeError, "wall limit exhausted"):
                    resume.run_command(
                        ["/runtime/uv", "python", "install", "3.11.13"], cwd=working,
                        env={"TMPDIR": str(runtime / "tmp")}, log_path=runtime / "logs" / "cmd.log",
                        stdout_path=None, deadline=11.0, runtime_root=runtime, wheel=wheel,
                        working=working, popen=popen,
                    )

            self.assertEqual(signals, [resume.PROCESS_GROUP_SIGTERM, resume.PROCESS_GROUP_SIGKILL])
            self.assertFalse(group["alive"])
            self.assertTrue(process.wait_calls)

    def test_nonzero_exited_leader_still_cleans_group_descendants(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            runtime = working / "runtime"
            runtime.mkdir()
            wheel = working / "uv.whl"
            wheel.write_bytes(b"wheel")
            process = self.Process(2)
            group = {"alive": True}
            signals = []
            clock_state = {"value": 0.0}

            def monotonic():
                clock_state["value"] += 0.06
                return clock_state["value"]

            def killpg(pgid, sig):
                self.assertEqual(pgid, process.pid)
                if sig == 0:
                    if not group["alive"]:
                        raise ProcessLookupError
                    return None
                signals.append(sig)
                if sig == resume.PROCESS_GROUP_SIGKILL:
                    group["alive"] = False

            with patch.object(resume, "check_resources"), \
                    patch.object(resume.os, "killpg", side_effect=killpg, create=True), \
                    patch.object(resume.time, "monotonic", side_effect=monotonic), \
                    patch.object(resume, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.1), \
                    patch.object(resume, "PROCESS_GROUP_KILL_GRACE_SECONDS", 0.1), \
                    patch.object(resume.time, "sleep", return_value=None):
                with self.assertRaisesRegex(resume.ResumeError, "command exited 2"):
                    resume.run_command(
                        ["/runtime/uv", "python", "install", "3.11.13"], cwd=working,
                        env={"TMPDIR": str(runtime / "tmp")}, log_path=runtime / "logs" / "cmd.log",
                        stdout_path=None, deadline=100.0, runtime_root=runtime, wheel=wheel,
                        working=working, popen=lambda argv, **kwargs: process,
                    )

            self.assertEqual(signals, [resume.PROCESS_GROUP_SIGTERM, resume.PROCESS_GROUP_SIGKILL])
            self.assertFalse(group["alive"])
            self.assertTrue(process.wait_calls)

    def test_cancellation_cleans_the_owned_process_group(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            runtime = working / "runtime"
            runtime.mkdir()
            wheel = working / "uv.whl"
            wheel.write_bytes(b"wheel")
            process = self.Process(None)
            group = {"alive": True}
            signals = []

            def killpg(pgid, sig):
                self.assertEqual(pgid, process.pid)
                if sig == 0:
                    if not group["alive"]:
                        raise ProcessLookupError
                    return None
                signals.append(sig)
                if sig == resume.PROCESS_GROUP_SIGTERM:
                    group["alive"] = False
                    process.returncode = -15

            with patch.object(resume, "check_resources"), \
                    patch.object(resume.os, "killpg", side_effect=killpg, create=True), \
                    patch.object(resume.time, "sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    resume.run_command(
                        ["/runtime/uv", "python", "install", "3.11.13"], cwd=working,
                        env={"TMPDIR": str(runtime / "tmp")}, log_path=runtime / "logs" / "cmd.log",
                        stdout_path=None, deadline=time_monotonic() + 10, runtime_root=runtime,
                        wheel=wheel, working=working,
                        popen=lambda argv, **kwargs: process,
                    )

            self.assertEqual(signals, [resume.PROCESS_GROUP_SIGTERM])
            self.assertFalse(group["alive"])
            self.assertTrue(process.wait_calls)


def time_monotonic() -> float:
    import time
    return time.monotonic()


class ReceiptTests(unittest.TestCase):
    def test_failure_receipt_is_terminal_and_has_no_generic_success_document(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            runtime = Path(temporary)
            operator = resume.RuntimeResume.__new__(resume.RuntimeResume)
            operator.history = [{"stage": "torch", "status": "started"}]
            operator.started_at = 1.0
            operator.started_at_unix = 100.0
            operator.clock = lambda: 2.0
            operator.evidence = {
                "failure": {"sha256": "failed"}, "bootstrap": {"sha256": "bootstrap"},
                "launcher": {"sha256": "launcher"}, "wheel": {"sha256": "wheel"},
                "installed_uv": {"sha256": "uv"}, "partial": {"runtime_root": str(runtime)},
            }
            operator.marker = runtime / "resume-started.json"
            operator.resume_receipt = runtime / "runtime-resume-receipt.json"
            operator.generic_receipt = runtime / "runtime-receipt.json"
            operator._marker_document = lambda status: {
                "schema": "kaggle_isolated_runtime_resume_attempt_v1", "status": status,
                "stages": operator.history,
            }
            operator.marker.write_text(json.dumps(operator._marker_document("running")))
            operator._write_failure(resume.ResumeError("torch command failed"))
            failure = json.loads(operator.resume_receipt.read_text(encoding="utf-8"))
            self.assertEqual(failure["schema"], resume.SCHEMA)
            self.assertEqual(failure["status"], "failed")
            self.assertFalse(failure["compatibility_receipt_written"])
            self.assertFalse(failure["training_executed"])
            self.assertFalse(operator.generic_receipt.exists())
            with self.assertRaises(FileExistsError):
                operator.resume_receipt.open("x")

    def test_runtime_probe_requires_successful_pinned_package_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            runtime = Path(temporary) / "runtime"
            python = runtime / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            (runtime / "python").mkdir(parents=True)
            python.write_bytes(b"fake python")
            observed = {
                "python": "3.11.13", "executable": str(python),
                "executable_sha256": digest(b"fake python"),
                "prefix": str(runtime / "venv"), "base_prefix": str(runtime / "python"),
                "packages": {"torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
                             "numpy": "1.26.4", "onnxruntime": "1.20.1", "pyrender": "0.1.45"},
            }
            self.assertEqual(resume.verify_runtime_probe(observed, python, runtime), observed)
            changed = {**observed, "packages": {**observed["packages"], "torch": "2.5.1+cpu"}}
            with self.assertRaisesRegex(resume.ResumeError, "pinned Python/package contract"):
                resume.verify_runtime_probe(changed, python, runtime)

    def test_compatibility_receipt_is_accepted_by_unchanged_runtime_verifier(self):
        launcher = load_unchanged_launcher()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            root18 = working / "visualizeit-cloud-v18"
            runtime_root = working / "runtime"
            python = runtime_root / "venv" / "bin" / "python"
            root18.mkdir(parents=True)
            python.parent.mkdir(parents=True)
            (root18 / "bench" / "runtime").mkdir(parents=True)
            (root18 / "bench" / "runtime" / "requirements-ranker.txt").write_bytes(b"ranker pins\n")
            (root18 / "bench" / "runtime" / "requirements-quality.txt").write_bytes(b"quality pins\n")
            python.write_bytes(b"fake isolated python executable")
            (runtime_root / "resolved-requirements.txt").write_bytes(b"torch==2.5.1+cu124\n")
            wheel = working / "uv.whl"
            wheel.write_bytes(b"verified wheel fixture")
            runtime = {
                "python": "3.11.13", "executable": str(python),
                "executable_sha256": digest(python.read_bytes()),
                "prefix": str(runtime_root / "venv"), "base_prefix": str(runtime_root / "python"),
                "packages": {"torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
                             "numpy": "1.26.4", "onnxruntime": "1.20.1", "pyrender": "0.1.45"},
            }
            uv_receipt = {
                "schema": "kaggle_isolated_runtime_v1", "uv_version": "0.12.22",
                "runtime_root": str(runtime_root), "runtime_links": [],
                "python_request": "3.11.13", "runtime": runtime,
                "uv_wheel": {"path": str(wheel), "bytes": wheel.stat().st_size,
                             "sha256": digest(wheel.read_bytes())},
                "resolved_requirements": {
                    "path": str(runtime_root / "resolved-requirements.txt"),
                    "bytes": (runtime_root / "resolved-requirements.txt").stat().st_size,
                    "sha256": digest((runtime_root / "resolved-requirements.txt").read_bytes()),
                },
                "requirements": {
                    relative: digest((root18 / relative).read_bytes())
                    for relative in resume.REQUIREMENT_FILES
                },
                "assets_acquired": False, "training_executed": False,
                "producer": {"name": "scripts/kaggle_runtime_resume.py",
                             "sha256": "producer-pin", "original_bootstrap_completed": False},
            }
            with patch.object(launcher, "runtime_probe", return_value=runtime):
                observed = launcher.verify_runtime(python, uv_receipt, root18, (root18, working))
            self.assertEqual(observed, runtime)


class ImportSafetyTests(unittest.TestCase):
    def test_module_import_and_help_do_not_execute_processes_or_load_models(self):
        models_before = {name for name in sys.modules if name.split(".", 1)[0] in {
            "torch", "sam2", "bop_toolkit", "foundpose", "quality_assets",
        }}
        with patch.object(subprocess, "Popen", side_effect=AssertionError("import/help launched a process")):
            stdout = StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                resume.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--cloud-execution", stdout.getvalue())
        models_after = {name for name in sys.modules if name.split(".", 1)[0] in {
            "torch", "sam2", "bop_toolkit", "foundpose", "quality_assets",
        }}
        self.assertEqual(models_after, models_before)


class ResourceGuardTests(unittest.TestCase):
    def test_low_free_space_and_tree_overage_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            working = Path(temporary)
            runtime = working / "runtime"
            runtime.mkdir()
            wheel = working / "uv.whl"
            wheel.write_bytes(b"fixture")
            with patch.object(resume.shutil, "disk_usage", return_value=SimpleNamespace(free=resume.MIN_WORKING_FREE_BYTES - 1)):
                with self.assertRaisesRegex(resume.ResumeError, "free reserve"):
                    resume.check_resources(runtime, wheel, working)
            with patch.object(resume.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * resume.GIB)), \
                    patch.object(resume, "tree_bytes", return_value=resume.MAX_RUNTIME_BYTES + 1):
                with self.assertRaisesRegex(resume.ResumeError, "18 GiB"):
                    resume.check_resources(runtime, wheel, working)


if __name__ == "__main__":
    unittest.main()
