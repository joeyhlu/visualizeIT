#!/usr/bin/env python3
"""Produce bounded TRAIN resource observations using the frozen Kaggle adapter.

Stdlib-only import/help. Only an explicitly admitted Kaggle GPU invocation may
import Torch, inside the isolated checkpoint-inspection child. No acquisition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import types
import uuid

ADAPTER_SHA256 = "1823fc93ef95057a3f2e4928c3d43123f130ce719683260fdce4e5f58212b388"
GIB = 1024**3
MAX_REQUEST_BYTES = 16 * 1024**2


class PilotError(ValueError):
    """Pilot identity, resource or measurement contract failed."""


class PilotDeadline(PilotError):
    """The shared post-admission execution allowance expired."""


@contextmanager
def execution_deadline(deadline: float):
    """Interrupt Linux hashing/I/O/subprocess revalidation at the shared deadline.

    subprocess.run kills/reaps its direct child on an interrupting exception.
    The frozen metadata child and nvidia-smi do not create neural descendants.
    Supervised neural children separately retain the adapter's group cleanup.
    """
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise PilotDeadline("shared pilot execution deadline exceeded")
    if (sys.platform != "linux" or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()):
        raise PilotError("interrupting execution deadline requires the admitted Linux main thread")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise PilotError("execution deadline timer is already in use")
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(_signal, _frame):
        raise PilotDeadline("shared pilot execution deadline exceeded")
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
        if time.monotonic() >= deadline:
            raise PilotDeadline("shared pilot execution deadline exceeded")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def seal(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_adapter(path: Path):
    """Authenticate and compile the SAME bytes; never consult cached bytecode."""
    if not path.is_absolute() or path.is_symlink() or path.stat().st_size > 1024**2:
        raise PilotError("explicit regular frozen adapter source required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ADAPTER_SHA256:
        raise PilotError("frozen adapter source SHA-256 mismatch")
    name = "_visualizeit_frozen_kaggle_adapter"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module  # dataclasses resolves its module during execution.
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.no_symlinks(path)
    return module


def read_request(path: Path, expected_sha256: str) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REQUEST_BYTES:
        raise PilotError("bounded regular pilot request required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PilotError("pilot request byte SHA-256 mismatch")
    # Reuse the frozen adapter's strict duplicate/nonfinite JSON parser after load.
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PilotError(f"duplicate request key: {key}")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique_pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(PilotError(f"nonfinite JSON: {value}")))
    if not isinstance(value, dict) or value.get("schema") != "kaggle_train_pilot_request_v1":
        raise PilotError("explicit kaggle_train_pilot_request_v1 request required")
    return value


def validate_caps(k, request: dict) -> dict:
    steps = k.integer(request.get("stop_after_steps"), "pilot invocation steps", 4)
    limits = k.validate_limits(request["limits"])
    k.positive(limits["max_seconds"], "pilot wall seconds", 900)
    k.integer(limits["max_rss_bytes"], "pilot RSS bytes", 8 * GIB)
    k.integer(limits["max_output_bytes"], "pilot output bytes", 2 * GIB)
    k.integer(request.get("storage_limit_bytes"), "observed working quota", 128 * GIB)
    if limits["max_log_bytes"] >= limits["max_output_bytes"]:
        raise PilotError("pilot output cap must include log and result headroom")
    if steps < 1:
        raise PilotError("at least one invocation optimizer step required")
    return limits


def pilot_identity(k, pilot: dict, runtime: dict) -> dict:
    r = pilot["training"]
    return {"schema": "kaggle_train_pilot_identity_v1", "pilot_id": pilot["pilot_id"],
            "full_run_id": pilot["full_run_id"], "adapter_sha256": ADAPTER_SHA256,
            "full_run_identity": k.identity_document(r, runtime),
            "dataset": r["dataset"], "stop_after_steps": pilot["stop_after_steps"],
            "output_root": pilot["pilot_output_root"], "limits": pilot["limits"],
            "storage_limit_bytes": pilot["storage_limit_bytes"],
            "train_states_requested": pilot["train_states_requested"]}


def validate_evidence(k, pilot: dict, identity: str, limits: dict) -> dict:
    r = pilot["training"]
    evidence = k.metadata_document(pilot["evidence"], (Path(r["input_root"]), Path(r["working_root"])))
    if platform.system() != "Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise PilotError("pilot execution requires an actual Kaggle Linux session")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise PilotError("actual Kaggle mounts missing")
    if (evidence.get("schema") != "kaggle_train_pilot_evidence_v1"
            or evidence.get("provider") != "kaggle" or evidence.get("stage") != "pilot"
            or evidence.get("identity_sha256") != identity
            or not evidence.get("session_id") or not evidence.get("evidence_source")
            or not re.fullmatch(r"https://www\.kaggle\.com/code/[^/]+/[^/?#]+", evidence.get("notebook_url", ""))):
        raise PilotError("fresh session/pilot identity provenance required")
    now = datetime.now(timezone.utc).timestamp()
    observed = datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00")).timestamp()
    expires = datetime.fromisoformat(evidence["expires_at"].replace("Z", "+00:00")).timestamp()
    if not now - 300 <= observed <= now < expires <= observed + 300:
        raise PilotError("pilot evidence must be observed within five minutes and unexpired")
    quota = evidence["quota"]
    if (quota.get("paid") is not False or quota.get("gpu_enabled") is not True
            or min(k.positive(quota.get("gpu_remaining_seconds"), "free GPU seconds", 30 * 3600),
                   k.positive(quota.get("session_remaining_seconds"), "session seconds", 86400)) < limits["max_seconds"]):
        raise PilotError("actual free GPU/session quota does not cover pilot wall cap")
    actual = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15, env=k.clean_environment())
    gpu = evidence["gpu"]
    devices = [[v.strip() for v in line.split(",")] for line in actual.stdout.splitlines()]
    selected = [d for d in devices if len(d) == 4 and d[:2] == [gpu["uuid"], gpu["name"]]]
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu["uuid"]) or len(selected) != 1:
        raise PilotError("observed selected GPU UUID/name changed")
    free_mib, total_mib = map(int, selected[0][2:])
    if not 0 < free_mib <= total_mib:
        raise PilotError("actual selected GPU free memory unavailable")
    evidence = {**evidence, "actual_gpu_free_bytes_at_admission": free_mib * 1024**2,
                "actual_gpu_total_bytes": total_mib * 1024**2}
    if (evidence.get("parity_passed") is not True or evidence.get("authentic_inputs_reviewed") is not True
            or evidence.get("test_arrays_opened") is not False):
        raise PilotError("authentic input review, detailed parity and sealed TEST prerequisites required")
    reports = {}
    for role in ("parity_report", "input_review"):
        selected = [i["file"] for i in r["inputs"] if i["role"] == role]
        if len(selected) != 1:
            raise PilotError(f"exactly one hashed detailed {role} required")
        report = k.metadata_document(selected[0], (Path(r["input_root"]), Path(r["working_root"])))
        if (report.get("passed") is not True or report.get("test_arrays_opened") is not False
                or report.get("dataset_sha256") != r["dataset"]["sha256"]
                or report.get("config_sha256") != r["config"]["sha256"]
                or report.get("plan_sha256") != evidence.get("plan_sha256")
                or not isinstance(report.get("checks"), list) or not report["checks"]
                or any(not isinstance(c, dict) or not c.get("name") or c.get("passed") is not True
                       or "observed" not in c or "expected" not in c for c in report["checks"])):
            raise PilotError(f"detailed {role} must bind dataset/config/plan and actual checks")
        reports[role] = selected[0]
    return evidence


def validate_pilot(k, pilot: dict) -> dict:
    """Admission for obtaining a FIRST observation; full admission stays untouched."""
    limits = validate_caps(k, pilot)
    for key in ("pilot_id", "full_run_id"):
        if str(uuid.UUID(pilot[key])) != pilot[key]:
            raise PilotError(f"{key} requires a canonical UUID")
    if pilot["pilot_id"] == pilot["full_run_id"]:
        raise PilotError("pilot and eventual full run need distinct identities")
    r = pilot["training"]
    if r.get("schema") != "kaggle_adaptation_request_v1" or r.get("mode", "extract") not in ("extract", "prepared"):
        raise PilotError("unchanged complete adapter training request required")
    roots = {key: k.no_symlinks(Path(r[key])) for key in
             ("source_install", "input_root", "working_root", "output_root")}
    if any(not p.is_absolute() for p in roots.values()):
        raise PilotError("explicit absolute application roots required")
    working = k.contained(roots["working_root"], Path("/kaggle/working"))
    k.contained(roots["input_root"], Path("/kaggle/input"))
    k.contained(roots["source_install"], working)
    full = k.contained(roots["output_root"], working)
    output = k.contained(Path(pilot["pilot_output_root"]), working)
    if not output.is_absolute() or output == working or output.exists():
        raise PilotError("fresh dedicated pilot output required; no overwrites or resumes")
    for other in (roots["source_install"], full):
        if output.is_relative_to(other) or other.is_relative_to(output) or other == working:
            raise PilotError("pilot/full/source outputs must be disjoint")
    if full == working or full.is_relative_to(roots["source_install"]):
        raise PilotError("dedicated eventual full output outside source required")
    k.validate_stage_paths(r, roots, r.get("mode", "extract"))
    if Path(r["paths"]["train_dir"]).exists():
        raise PilotError("eventual full trainer directory must be fresh")
    root18 = k.verify_source(roots["source_install"])
    k.validate_limits(r["limits"])
    allowed = roots["input_root"], working
    for key in ("runtime", "config", "global_inventory", "identity_bridge", "dataset", "crop_manifest", "labels"):
        k.metadata_document(r[key], allowed)
    if not isinstance(r["inputs"], list) or not r["inputs"]:
        raise PilotError("complete non-test input descriptor inventory required")
    names = set()
    for item in r["inputs"]:
        if set(item) != {"role", "file"} or not item["role"] or item["role"] == "test":
            raise PilotError("inputs require non-test role/file descriptors")
        p = k.no_symlinks(Path(item["file"]["path"]))
        if str(p) in names:
            raise PilotError("duplicate input descriptor")
        names.add(str(p))
    python = Path(r["python"])
    k.no_symlinks(python.parent)
    if not python.is_absolute() or not python.resolve(strict=True).is_relative_to(working):
        raise PilotError("isolated cloud interpreter must remain in working root")
    receipt = k.read_json(Path(r["runtime"]["path"]))
    runtime = k.verify_runtime(python, receipt, root18, allowed)
    runtime_root = k.contained(Path(receipt["runtime_root"]), working)
    for other in (output, full, roots["source_install"]):
        if runtime_root.is_relative_to(other) or other.is_relative_to(runtime_root):
            raise PilotError("runtime/source/full/pilot directories must be disjoint")
    # Exact frozen full global join including all receipts and active/sealed
    # aliases. No fabricated timings or replacement full-admission functions.
    metadata = k.run_metadata_check(r, "preflight", root18)
    for item in r["inputs"]:
        k.verify_descriptor(item["file"], allowed)
    k.verify_neural_assets(root18, r["inputs"])
    cfg = k.read_json(Path(r["config"]["path"]))
    if cfg.get("device") != "cuda":
        raise PilotError("explicit CUDA config required; no CPU fallback")
    if cfg.get("max_output_bytes", 1536 * 1024**2) > limits["max_output_bytes"]:
        raise PilotError("unchanged trainer output cap exceeds pilot output cap")
    plan = k.read_json(Path(r["global_inventory"]["path"]))
    if pilot["train_states_requested"] not in {x["train_states_requested"] for x in plan["target_requests"]}:
        raise PilotError("pilot target must select an existing frozen train-state request")
    identity_document = pilot_identity(k, pilot, runtime)
    identity = seal(identity_document)
    evidence = validate_evidence(k, pilot, identity, limits)
    if evidence.get("plan_sha256") != metadata["plan_sha256"]:
        raise PilotError("pilot evidence changed frozen plan identity")
    links = k.runtime_link_policy(receipt, working)
    resident = k.unique_tree_bytes((working,), approved_links=links)
    free = min(k.free_by_filesystem((working,)).values())
    extra = limits["max_output_bytes"]
    if extra + limits["reserve_bytes"] > free or resident + extra + limits["reserve_bytes"] > pilot["storage_limit_bytes"]:
        raise PilotError("actual working storage/free quota cannot fit pilot cap plus reserve")
    return {"root18": root18, "runtime": runtime, "identity": identity, "identity_document": identity_document,
            "evidence": evidence, "resident": resident, "free": free, "links": links, "metadata": metadata}


def trainer_argv(k, pilot: dict) -> list[str]:
    argv = k.command("train", pilot["training"])
    argv[argv.index("--output") + 1] = str(Path(pilot["pilot_output_root"]) / "train")
    return argv + ["--stop-after-steps", str(pilot["stop_after_steps"])]


# Executed ONLY in the verified isolated interpreter. -I excludes ambient paths;
# the one authenticated ROOT18 is explicitly inserted. The config defaults and
# dataset canonical identity come from the original trainer/data implementation.
INSPECT = r'''
import dataclasses, hashlib, json, pathlib, sys
root, request, output = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2]), pathlib.Path(sys.argv[3])
sys.path.insert(0, str(root))
from bench.quality_adaptation_train import read_config
from bench.quality_adaptation_data import load_dataset, canonical_hash
from bench.quality_adaptation_runtime import LOCK, sha256, runtime_versions
cfg = read_config(pathlib.Path(request['config']['path']))
ds = load_dataset(pathlib.Path(request['dataset']['path']), inspect_arrays=False)
result = {'config': dataclasses.asdict(cfg), 'config_sha256': canonical_hash(dataclasses.asdict(cfg)),
          'dataset_sha256': ds.identity, 'source_lock_sha256': sha256(LOCK), 'versions': runtime_versions()}
if output.is_dir():
 import torch
 path = output / 'last.pt'
 with path.open('rb') as f:
  raw_hash = hashlib.file_digest(f, 'sha256').hexdigest()
  f.seek(0)
  state = torch.load(f, map_location='cpu', weights_only=True)
 result['checkpoint_sha256'] = raw_hash
 result['checkpoint'] = {k: state.get(k) for k in ('schema','dataset_sha256','config_sha256','source_lock_sha256',
     'steps','epoch','batch_cursor','completed','elapsed_seconds','versions')}
 result['optimizer_steps'] = [v['step'].item() if hasattr(v.get('step'), 'item') else v.get('step')
                              for v in state.get('optimizer', {}).get('state', {}).values()]
 result['optimizer_state_entries'] = len(result['optimizer_steps'])
 result['resume_state_present'] = all(k in state for k in ('head','optimizer','scaler','cpu_rng','cuda_rng'))
print(json.dumps(result, allow_nan=False))
'''


def remaining_limits(pilot: dict, deadline: float) -> dict:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise PilotError("whole pilot wall deadline exceeded")
    return {**pilot["limits"], "max_seconds": min(seconds, pilot["limits"]["max_seconds"])}


def inspect_training(k, pilot: dict, root18: Path, output: Path, *, log_path: Path, deadline: float) -> dict:
    """Bound checkpoint loading as well as training; its stdout is a bounded file."""
    r = pilot["training"]
    with execution_deadline(deadline):
        result = k.supervise([r["python"], "-I", "-B", "-c", INSPECT, str(root18),
                              json.dumps(r, allow_nan=False), str(output)], cwd=root18,
                             output_root=Path(pilot["pilot_output_root"]), working=Path(r["working_root"]),
                             log_path=log_path, limits=remaining_limits(pilot, deadline), env=k.clean_environment())
        if result["status"] != "completed" or result["returncode"] != 0:
            raise PilotError(f"isolated trainer/checkpoint inspection failed: {result['reason']}")
        return k.read_json(log_path)


def reconcile(k, pilot: dict, result: dict, before: dict, after: dict, status: dict, progress: dict) -> int:
    if (result.get("status") != "incomplete" or result.get("returncode") != 3
            or result.get("reason") != "child_reported_incomplete"
            or status.get("status") != "incomplete" or status.get("reason") != "invocation_step_limit"
            or status.get("resumed_from") is not None or status.get("test_arrays_opened") is not False
            or status.get("quality_claim") is not False or progress.get("freshness") != "fresh"):
        raise PilotError("only a clean invocation_step_limit incomplete TRAIN can yield measurements")
    for key in ("config", "config_sha256", "dataset_sha256", "source_lock_sha256", "versions"):
        if before.get(key) != after.get(key):
            raise PilotError(f"trainer inspection identity changed: {key}")
    cfg = before["config"]
    if k.canonical(cfg) != before["config_sha256"]:
        raise PilotError("canonical original trainer defaults/config hash differs")
    checkpoint = after["checkpoint"]
    for key in ("config_sha256", "dataset_sha256", "source_lock_sha256", "versions"):
        if checkpoint.get(key) != before[key]:
            raise PilotError(f"checkpoint identity differs from original trainer: {key}")
    steps = checkpoint.get("steps")
    if (checkpoint.get("schema") != "correspondence_head_checkpoint_v1" or checkpoint.get("completed") is not False
            or type(steps) is not int or not 0 < steps <= pilot["stop_after_steps"]
            or steps != status.get("steps") or steps != progress.get("last_confirmed_steps")
            or checkpoint.get("epoch") != status.get("epoch")
            or checkpoint.get("epoch", -1) < 0 or type(checkpoint.get("batch_cursor")) is not int
            or checkpoint["batch_cursor"] < 0 or after.get("resume_state_present") is not True
            or not after.get("optimizer_steps") or any(isinstance(s, bool) or not isinstance(s, (int, float))
                or not math.isfinite(s) or s != steps for s in after["optimizer_steps"])):
        raise PilotError("checkpoint/status/actual optimizer step delta is invalid or exceeds cap")
    for value in (checkpoint.get("elapsed_seconds"), status.get("elapsed_seconds"), result.get("elapsed_seconds")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise PilotError("actual positive finite elapsed observations required")
    if cfg.get("device") != "cuda" or cfg.get("max_steps", 0) < steps:
        raise PilotError("checkpoint steps/config mismatch")
    return steps


def full_run_allowance(k, pilot: dict, context: dict, before: dict, *, deadline: float) -> dict:
    """Bound all full-scope export checks, including frozen fixed-timeout children."""
    with execution_deadline(deadline):
        return _full_run_allowance(k, pilot, context, before)


def _full_run_allowance(k, pilot: dict, context: dict, before: dict) -> dict:
    """Conservative export gate; a sampled prefix is never storage admission."""
    r = pilot["training"]
    working = k.no_symlinks(Path(r["working_root"]))
    allowed = Path(r["input_root"]), working
    if seal(pilot_identity(k, pilot, context["runtime"])) != context["identity"]:
        raise PilotError("full-run export changed pilot/full request identity")
    root18 = k.verify_source(Path(r["source_install"]))
    if k.file_sha256(root18 / "bench/runtime/adaptation-sources.json") != before["source_lock_sha256"]:
        raise PilotError("full-run export source lock differs from measured trainer")
    raw_cfg = k.metadata_document(r["config"], allowed)
    cfg = before["config"]
    if k.canonical(cfg) != before["config_sha256"] or any(cfg.get(key) != value for key, value in raw_cfg.items()):
        raise PilotError("full-run export config differs from original canonical trainer defaults")
    dataset = k.metadata_document(r["dataset"], allowed)
    if k.canonical(dataset) != before["dataset_sha256"]:
        raise PilotError("full-run export dataset differs from measured complete global dataset")
    plan = k.metadata_document(r["global_inventory"], allowed)
    metadata = k.run_metadata_check(r, "preflight", root18)
    if (metadata["plan_sha256"] != context["metadata"]["plan_sha256"]
            or plan["plan_sha256"] != metadata["plan_sha256"]
            or pilot["train_states_requested"] not in {t["train_states_requested"] for t in plan["target_requests"]}
            or plan["counts"]["train"]["refinement_states"] < pilot["train_states_requested"]):
        raise PilotError("full-run export requires the same complete frozen train-state target")
    # The old scale schema charges prepared bytes to its resident/quota scope.
    # A read-only input quota cannot be inferred from a working-only snapshot.
    # Preserve observations, but emit neither legacy record for that scope.
    extraction = k.no_symlinks(Path(r["paths"]["extraction_dir"]))
    if (not extraction.is_relative_to(working) or not extraction.is_dir()
            or extraction.stat().st_dev != working.stat().st_dev):
        raise PilotError("separate/read-only prepared-input quota scope is unmeasured; no full-run records")
    full_output = k.contained(Path(r["output_root"]), working)
    if full_output.exists() and full_output.stat().st_dev != working.stat().st_dev:
        raise PilotError("full-run output is outside the measured working filesystem quota")
    limits = k.validate_limits(r["limits"])
    k.integer(cfg["max_steps"], "full planned optimizer steps", 100000)
    k.positive(cfg["max_seconds"], "full trainer wall bound", limits["max_seconds"])
    checkpoint_cap = k.integer(cfg["max_checkpoint_bytes"], "full checkpoint cap", GIB)
    trainer_cap = k.integer(cfg["max_output_bytes"], "full trainer output cap", limits["max_output_bytes"])
    if trainer_cap < 3 * checkpoint_cap + 8 * 1024**2:
        raise PilotError("full trainer cap cannot retain last/best/atomic checkpoint and metadata")
    current_full = k.unique_tree_bytes((full_output,))
    required_envelope = current_full + trainer_cap + limits["max_log_bytes"] + 2 * k.MAX_JSON
    if required_envelope > limits["max_output_bytes"]:
        raise PilotError("full supervisor output cap cannot fit retained/atomic checkpoint, log and metadata envelope")
    evidence = validate_evidence(k, pilot, context["identity"], pilot["limits"])
    quota = evidence["quota"]
    if (min(quota["gpu_remaining_seconds"], quota["session_remaining_seconds"]) < limits["max_seconds"]
            or quota.get("storage_limit_bytes") != pilot["storage_limit_bytes"]
            or quota.get("storage_scope") != str(working)):
        raise PilotError("fresh full wall/storage quota and exact working storage scope required for export")
    links = k.runtime_link_policy(k.read_json(Path(r["runtime"]["path"])), working)
    resident = k.unique_tree_bytes((working,), approved_links=links)
    free = min(k.free_by_filesystem((working,)).values())
    # Use the ENTIRE full-run supervisor output cap as additional allowance,
    # without subtracting existing pilot/prepared files. It is an enforced bound,
    # not the sampled observed peak; the deliberate overcount is conservative.
    allowance = limits["max_output_bytes"]
    reserve = max(limits["reserve_bytes"], pilot["limits"]["reserve_bytes"])
    if (allowance + reserve > free
            or resident + allowance + reserve > quota["storage_limit_bytes"]):
        raise PilotError("full-run enforced output allowance plus reserve does not fit actual free/quota storage")
    return {"additional_output_allowance_bytes": allowance, "basis": "full_run_supervisor_max_output_bytes",
            "is_measured_peak": False, "required_output_envelope_bytes": required_envelope,
            "trainer_output_cap_bytes": trainer_cap, "checkpoint_cap_bytes": checkpoint_cap,
            "log_cap_bytes": limits["max_log_bytes"], "adapter_metadata_allowance_bytes": 2 * k.MAX_JSON,
            "resident_bytes": resident, "actual_free_bytes": free, "reserve_bytes": reserve,
            "storage_limit_bytes": quota["storage_limit_bytes"], "storage_scope": str(working),
            "dataset_sha256": before["dataset_sha256"], "config_sha256": before["config_sha256"],
            "source_lock_sha256": before["source_lock_sha256"], "planned_optimizer_steps": cfg["max_steps"],
            "plan_sha256": plan["plan_sha256"], "full_run_id": pilot["full_run_id"]}


def measurement_records(k, pilot: dict, context: dict, result: dict, before: dict,
                        steps: int, peak_storage: int, free_after: int, *, resident_after: int | None = None,
                        deadline: float | None = None) -> dict:
    """Always retain observations; emit legacy records only after full-scope fit."""
    r, limits = pilot["training"], pilot["limits"]
    # Standalone model-free callers receive a finite allowance too. execute
    # ALWAYS passes its existing absolute deadline; export never resets it.
    deadline = time.monotonic() + limits["max_seconds"] if deadline is None else deadline
    resident_after = peak_storage if resident_after is None else resident_after
    extra = max(0, peak_storage - context["resident"])
    common = {"sampling": "100ms filesystem snapshots; instantaneous peak may be higher",
              "observed_additional_peak_bytes_lower_bound": extra,
              "enforced_additional_output_cap_bytes": limits["max_output_bytes"],
              "peak_is_exact": False, "quality_claim": False, "full_training_admitted": False}
    records = {"forecast": {"training_seconds_linear": result["elapsed_seconds"] / steps * before["config"]["max_steps"],
                            "representative_full_run_fit_not_established": True},
               "observed": {**common, "trainer_wall_seconds": result["elapsed_seconds"],
                            "peak_sampled_group_rss_bytes": result["peak_group_rss_bytes"],
                            "working_storage_high_water_bytes": peak_storage, "resident_bytes_after": resident_after,
                            "free_bytes_after": free_after},
               "enforced_bounds": limits, "admission_records_emitted": False}
    try:
        bound = full_run_allowance(k, pilot, context, before, deadline=deadline)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        records["admission_records_withheld_reason"] = str(error)
        return records
    provenance = {**common, "legacy_additional_peak_field_basis": "enforced_full_run_upper_bound_not_measured_peak",
                  "full_run_allowance": bound}
    scale = {"plan_sha256": context["metadata"]["plan_sha256"],
             "train_states_requested": pilot["train_states_requested"],
             "measured_training_steps": steps, "training_seconds": result["elapsed_seconds"],
             "planned_optimizer_steps": before["config"]["max_steps"],
             "actual_free_bytes": bound["actual_free_bytes"], "storage_limit_bytes": bound["storage_limit_bytes"],
             "time_limit_seconds": r["limits"]["max_seconds"], "reserve_bytes": bound["reserve_bytes"],
             "storage_baseline": "after_preparation", "measured_resident_bytes": bound["resident_bytes"],
             "measured_additional_peak_bytes": bound["additional_output_allowance_bytes"], "pilot_provenance": provenance}
    stage = {"measured_units": steps, "measured_seconds": result["elapsed_seconds"],
             "planned_units": before["config"]["max_steps"], "resident_bytes": bound["resident_bytes"],
             "additional_peak_bytes": bound["additional_output_allowance_bytes"], "storage_baseline": "current_resident_before_stage",
             "storage_limit_bytes": bound["storage_limit_bytes"], "pilot_provenance": provenance}
    if time.monotonic() >= deadline:
        records["admission_records_withheld_reason"] = "shared pilot execution deadline exceeded before export"
        return records
    records.update(scale_measurements=scale, stage_measurements=stage, admission_records_emitted=True,
                   full_run_allowance=bound)
    return records


def execute(path: Path, expected_sha256: str) -> dict:
    pilot = read_request(path, expected_sha256)
    k = load_adapter(Path(pilot["adapter"]["path"]))
    k.no_symlinks(path)
    if k.file_sha256(path) != expected_sha256 or k.parse_json_bytes(path.read_bytes(), path) != pilot:
        raise PilotError("pilot request changed during authentication")
    allowed = Path(pilot["training"]["input_root"]), Path(pilot["training"]["working_root"])
    k.verify_descriptor(pilot["adapter"], allowed)
    context = validate_pilot(k, pilot)
    output = Path(pilot["pilot_output_root"])
    output.mkdir()  # deliberately no parents, exist_ok or prior-run adoption.
    record = {"schema": "kaggle_train_pilot_result_v1", "status": "running", "measurement_complete": False,
              "identity_sha256": context["identity"], "identity": context["identity_document"],
              "request_sha256": expected_sha256, "quality_claim": False, "test_arrays_opened": False,
              "admission_evidence": context["evidence"], "evidence_descriptor": pilot["evidence"],
              "metadata_join": context["metadata"],
              "trainer_status": "not_started", "full_training_admitted": False, "outputs_preserved": True}
    k.atomic_json(output / "pilot-result.json", record)
    started = time.monotonic()
    deadline = started + pilot["limits"]["max_seconds"]
    try:
        with k.run_lock(output):
            before = inspect_training(k, pilot, context["root18"], output / "train",
                                      log_path=output / "inspect-before.json", deadline=deadline)
            if before["config"]["device"] != "cuda":
                raise PilotError("canonical original trainer config must select CUDA")
            # Runtime/source/evidence checks are repeated immediately before the
            # neural launch; preparation/inspection time must not stale admission.
            with execution_deadline(deadline):
                context = validate_pilot_for_launch(k, pilot, context)
                record["admission_evidence"] = context["evidence"]
                argv = trainer_argv(k, pilot)
                row = {"stage": "train", "status": "running", "argv": argv,
                       "started_at": datetime.now(timezone.utc).isoformat(), "adapter_pid": os.getpid(),
                       "trainer_source_sha256": k.file_sha256(context["root18"] / "bench/quality_adaptation_train.py")}
            env = k.clean_environment()
            env["CUDA_VISIBLE_DEVICES"] = context["evidence"]["gpu"]["uuid"]
            stop = threading.Event()
            samples = [context["resident"]]
            errors = []
            def sample():
                while not stop.is_set():
                    try:
                        samples[0] = max(samples[0], k.unique_tree_bytes((Path(pilot["training"]["working_root"]),), approved_links=context["links"]))
                    except Exception as error:
                        errors.append(str(error))
                        return
                    stop.wait(.1)
            observer = threading.Thread(target=sample, daemon=True)
            observer.start()
            try:
                result = k.supervise(argv, cwd=context["root18"], output_root=output,
                    working=Path(pilot["training"]["working_root"]), log_path=output / "trainer.log",
                    limits=remaining_limits(pilot, deadline), env=env)
            finally:
                stop.set()
                observer.join(timeout=5)
            row.update(result)
            record.update(trainer_invocation=row, trainer_status=result["status"])
            if (result["status"] != "incomplete" or result["returncode"] != 3
                    or result["reason"] != "child_reported_incomplete"):
                raise PilotError(f"trainer control failed; no measurements: {result['reason']}")
            if observer.is_alive() or errors:
                raise PilotError(f"filesystem observation failed: {errors}")
            with execution_deadline(deadline):
                samples[0] = max(samples[0], k.unique_tree_bytes((Path(pilot["training"]["working_root"]),), approved_links=context["links"]))
                train = output / "train"
                status = k.read_json(k.contained(train / "status.json", output))
                state = {"identity": {"roots": {"output_root": str(output)}, "paths": {"train_dir": str(train)}},
                         "stages": [row]}
                progress = k.trainer_progress(output, state, observe_invocation=True)
                record.update(trainer_status=status.get("status"), optimizer_progress=progress)
                checkpoint = k.contained(train / "last.pt", output)
                desc = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": k.file_sha256(checkpoint)}
                if desc["bytes"] > before["config"]["max_checkpoint_bytes"]:
                    raise PilotError("checkpoint exceeds original config size bound")
            after = inspect_training(k, pilot, context["root18"], train, log_path=output / "inspect-after.json", deadline=deadline)
            with execution_deadline(deadline):
                if after["checkpoint_sha256"] != desc["sha256"] or k.file_sha256(checkpoint) != desc["sha256"]:
                    raise PilotError("checkpoint changed across authenticated weights_only inspection")
                steps = reconcile(k, pilot, result, before, after, status, progress)
                resident_after = k.unique_tree_bytes((Path(pilot["training"]["working_root"]),), approved_links=context["links"])
                samples[0] = max(samples[0], resident_after)
                free = min(k.free_by_filesystem((Path(pilot["training"]["working_root"]),)).values())
            record.update(measurement_records(k, pilot, context, result, before, steps, samples[0], free,
                                              resident_after=resident_after, deadline=deadline))
            record.update(status="measurement_complete", measurement_complete=True, checkpoint=desc,
                          checkpoint_metadata=after, optimizer_step_delta=steps,
                          checkpoint_promotion="standalone pilot; no automatic full-run promotion")
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        record["total_observed_seconds"] = time.monotonic() - started
        k.atomic_json(output / "pilot-result.json", record)
    return record


def validate_pilot_for_launch(k, pilot: dict, context: dict) -> dict:
    """Refresh actual source/runtime/free-space/session facts after inspection."""
    r = pilot["training"]
    allowed = Path(r["input_root"]), Path(r["working_root"])
    for key in ("runtime", "config", "global_inventory", "identity_bridge", "dataset", "crop_manifest", "labels"):
        k.metadata_document(r[key], allowed)
    root18 = k.verify_source(Path(r["source_install"]))
    runtime = k.verify_runtime(Path(r["python"]), k.read_json(Path(r["runtime"]["path"])), root18,
                               (Path(r["input_root"]), Path(r["working_root"])))
    if runtime != context["runtime"] or seal(pilot_identity(k, pilot, runtime)) != context["identity"]:
        raise PilotError("source/runtime/config identity changed before launch")
    evidence = validate_evidence(k, pilot, context["identity"], pilot["limits"])
    k.run_metadata_check(r, "preflight", root18)
    for item in r["inputs"]:
        k.verify_descriptor(item["file"], allowed)
    k.verify_neural_assets(root18, r["inputs"])
    free = min(k.free_by_filesystem((Path(r["working_root"]),)).values())
    if free < pilot["limits"]["max_output_bytes"] + pilot["limits"]["reserve_bytes"]:
        raise PilotError("observed free storage no longer fits cap/reserve")
    return {**context, "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.request, args.request_sha256)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error), "measurement_complete": False}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["measurement_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
