"""Run existing model stages sequentially in separate processes.

This command never trains, provisions resources, or downloads weights. Run cloud
bootstrap explicitly when weights are absent. Each stage has its own log and
resource receipt; a failed stage stops the pipeline and preserves its output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .quality_assets import CACHE, BUDGET, digest, save_result

STAGES = ('smoke', 'banks', 'cnos-bank', 'segment', 'pose')


def stage_commands(bundle, output, *, device='cuda', stages=STAGES, limit=None,
                   ranker='legacy', checkpoint=None, patch_features=False,
                   masks=None, gotrack_head=None, gotrack_head_sha256=None,
                   gotrack_inference_config=None,
                   gotrack_inference_config_sha256=None,
                   gotrack_runtime_compat_config=None,
                   gotrack_runtime_compat_config_sha256=None,
                   cloud_execution=False, unlit_templates=False,
                   disable_multisampling=False):
    from .quality_runner import read_input
    manifest = read_input(Path(bundle))
    output = Path(output)
    masks = Path(masks) if masks is not None else output / 'masks'
    selected = list(stages)
    if not selected or any(s not in STAGES for s in selected) or len(set(selected)) != len(selected):
        raise ValueError('Stages must be unique supported stage names')
    if selected != sorted(selected, key=STAGES.index):
        raise ValueError('Stages must retain smoke/banks/cnos-bank/segment/pose order')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('Frame limit must be positive')
    count = len(manifest.get('frame_ids', []))
    if count and limit is not None and limit > count:
        raise ValueError('Frame limit exceeds the bundle scored window')
    # Existing physical SAM requires omission (not --limit N) for a full run.
    effective_limit = None if limit == count else limit
    if (ranker in ('learned', 'confidence-research', 'ordering-research')) != (checkpoint is not None):
        raise ValueError('A checkpoint ranker requires a checkpoint')
    if checkpoint is not None:
        from .quality_pose_ranker import make_ranker
        make_ranker(ranker, Path(checkpoint))
    inference_context = None
    if (gotrack_head is not None or gotrack_head_sha256 is not None or
            gotrack_inference_config is not None or
            gotrack_inference_config_sha256 is not None or
            gotrack_runtime_compat_config is not None or
            gotrack_runtime_compat_config_sha256 is not None or cloud_execution):
        from .quality_gotrack_inference import options_from_values, prepare_inference
        options = options_from_values(
            head_path=gotrack_head, head_sha256=gotrack_head_sha256,
            config_path=gotrack_inference_config,
            config_sha256=gotrack_inference_config_sha256,
            compatibility_config_path=gotrack_runtime_compat_config,
            compatibility_config_sha256=gotrack_runtime_compat_config_sha256,
            cloud_execution=cloud_execution)
        if 'pose' not in selected:
            raise ValueError('GoTrack head/config options require the pose stage')
        inference_context = prepare_inference(options, bundle, manifest)
        from .quality_runner import _validate_inference_runner_options
        _validate_inference_runner_options(inference_context, mode='complete',
            forced_occlusion=None, model_memory=False,
            unlit_templates=unlit_templates, appearance_check=False,
            disable_multisampling=disable_multisampling,
            pnp_no_extrinsic_guess=False,
            chronological_mug_capture_root=None,
            chronological_mug_capture_branch=None)
    pose_stem = f'{ranker}'
    if inference_context is not None:
        head_identity = inference_context.requested_head_sha256 or 'frozen'
        pose_stem = f'{inference_context.profile}-{head_identity}-{ranker}'
    smoke_stem = f'{manifest["object"]}-{device}'
    if inference_context is not None:
        smoke_stem += '-unlit-no-msaa'
    targets = {'smoke': CACHE / 'smoke' / f'{smoke_stem}.json',
               'banks': CACHE / 'banks' / f'{manifest["object"]}-foundpose.pt',
               'cnos-bank': output / 'cnos-bank.json', 'segment': masks,
               'pose': output / f'{pose_stem}.json'}
    for stage in ('smoke', 'banks'):
        if not targets[stage].resolve().is_relative_to(CACHE.resolve()):
            raise ValueError('Object alias escapes model cache')
    commands = []
    for stage in selected:
        command = [sys.executable, '-B', '-m', 'bench.quality_runner', stage,
                   '--bundle', str(Path(bundle).resolve()), '--output', str(targets[stage].resolve()),
                   '--device', device]
        if stage in ('segment', 'pose') and effective_limit is not None:
            command += ['--limit', str(effective_limit)]
        if inference_context is not None and stage in ('smoke', 'pose'):
            command += ['--unlit-templates', '--disable-multisampling']
        if stage == 'pose':
            command += ['--mode', 'complete', '--masks', str(masks.resolve()),
                        '--ranker', ranker]
            trace_path = (output / f'{pose_stem}.candidates.jsonl'
                          if inference_context is not None else output / 'candidates.jsonl')
            command += ['--candidate-trace', str(trace_path.resolve())]
            if checkpoint is not None:
                command += ['--ranker-checkpoint', str(Path(checkpoint).resolve())]
            if patch_features:
                command += ['--ranker-patch-features']
            if inference_context is not None:
                command += ['--gotrack-inference-config', str(Path(gotrack_inference_config).resolve()),
                    '--gotrack-inference-config-sha256', gotrack_inference_config_sha256,
                    '--cloud-execution']
                if gotrack_runtime_compat_config is not None:
                    command += ['--gotrack-runtime-compat-config',
                        str(Path(gotrack_runtime_compat_config).resolve()),
                        '--gotrack-runtime-compat-config-sha256',
                        gotrack_runtime_compat_config_sha256]
                if gotrack_head is not None:
                    command += ['--gotrack-head', str(Path(gotrack_head).resolve()),
                        '--gotrack-head-sha256', gotrack_head_sha256]
        commands.append((stage, command))
    return commands


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def run_stage(stage, command, output, *, timeout_s, storage_limit_bytes, storage_root,
              output_limit_bytes=256 * 1024**2, inference_provenance=None,
              inference_output=None):
    import psutil
    started = time.monotonic()
    log = Path(output) / f'{stage}.log'
    peak = 0
    process = None
    status = 'failed'
    reason = None
    exit_code = None
    last_storage_check = -float('inf')
    used = tree_bytes(storage_root)
    output_used = tree_bytes(output)
    if used > storage_limit_bytes or output_used > output_limit_bytes:
        raise ValueError('Storage budget already exceeded before stage start')
    try:
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            while process.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True))
                    peak = max(peak, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if elapsed - last_storage_check >= 10:
                    used = tree_bytes(storage_root)
                    output_used = tree_bytes(output)
                    last_storage_check = elapsed
                if elapsed >= timeout_s or used > storage_limit_bytes or output_used > output_limit_bytes:
                    reason = 'runtime_limit' if elapsed >= timeout_s else 'storage_limit'
                    break
                time.sleep(.25)
            if reason is None:
                exit_code = process.wait()
                status = 'complete' if exit_code == 0 else 'failed'
                reason = None if exit_code == 0 else 'stage_exit_nonzero'
                # A short stage can grow past a budget between periodic samples.
                used, output_used = tree_bytes(storage_root), tree_bytes(output)
                if used > storage_limit_bytes or output_used > output_limit_bytes:
                    status, reason = 'failed', 'storage_limit'
    finally:
        if process is not None and process.poll() is None:
            # SAM can launch CNOS children. Terminate this stage's descendants too.
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            for child in descendants:
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            _, alive = psutil.wait_procs(descendants, timeout=2)
            for child in alive:
                try: child.kill()
                except psutil.NoSuchProcess: pass
            psutil.wait_procs(alive, timeout=2)
        receipt = dict(stage=stage, command=command, status=status, failure_reason=reason,
            exit_code=exit_code, elapsed_s=time.monotonic() - started,
            sampled_peak_process_tree_rss_bytes=peak or None, storage_bytes=tree_bytes(storage_root),
            log=str(log), billing_stopped=None)
        if stage == 'pose' and inference_provenance is not None:
            receipt['gotrack_inference_preflight'] = inference_provenance
            observed = None
            if inference_output is not None and Path(inference_output).is_file():
                try:
                    observed = _read_child_inference_identity(
                        inference_output, inference_provenance)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    receipt['inference_provenance_error'] = f'{type(exc).__name__}: {exc}'
                    if status == 'complete':
                        status = 'failed'
                        reason = 'pose_inference_provenance_invalid'
                        receipt['status'] = status
                        receipt['failure_reason'] = reason
            if observed is None:
                receipt['gotrack_inference'] = inference_provenance
                receipt['inference_identity_source'] = 'preflight_only'
                if status == 'complete':
                    status, reason = 'failed', 'pose_inference_output_missing'
                    receipt['status'] = status
                    receipt['failure_reason'] = reason
            else:
                receipt['gotrack_inference'] = observed
                receipt['inference_identity_source'] = 'pose_output'
        save_result(Path(output) / f'{stage}.receipt.json', receipt)
    if status != 'complete':
        raise RuntimeError(f'{stage} failed ({reason}); inspect {log}')
    return receipt


def _read_child_inference_identity(path, preflight):
    """Read the identity actually recorded by the pose process, including failures."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError('Pose result must be a JSON object')
    identity = document.get('provenance', {}).get('gotrack_inference')
    if not isinstance(identity, dict):
        raise ValueError('Pose result lacks its GoTrack inference identity')
    if any(identity.get(field) != preflight.get(field)
           for field in ('schema', 'profile', 'config_sha256')):
        raise ValueError('Pose result inference identity differs from preflight')
    expected_head = preflight.get('head', {})
    actual_head = identity.get('head', {})
    if (not isinstance(actual_head, dict) or
            actual_head.get('requested_sha256') != expected_head.get('requested_sha256') or
            actual_head.get('base_checkpoint_sha256') != expected_head.get('base_checkpoint_sha256')):
        raise ValueError('Pose result head identity differs from preflight')
    installation = identity.get('head_installation')
    if not isinstance(installation, dict) or set(installation) != {'requested', 'installed'}:
        raise ValueError('Pose result head installation receipt is malformed')
    requested_sha = expected_head.get('requested_sha256')
    if installation['requested'] is not (requested_sha is not None):
        raise ValueError('Pose result head installation request differs from preflight')
    if requested_sha is None:
        if installation['installed'] is not True or actual_head.get('installed_sha256') is not None:
            raise ValueError('Frozen-head pose result reports an impossible installation state')
    elif installation['installed'] is True:
        if actual_head.get('installed_sha256') != requested_sha:
            raise ValueError('Pose result installed head SHA differs from the requested export')
    elif installation['installed'] is not False or actual_head.get('installed_sha256') is not None:
        raise ValueError('Failed pose result reports an impossible adapted-head installation state')
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ranker', choices=['legacy', 'deterministic', 'learned', 'confidence-research', 'ordering-research'], default='legacy')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--patch-features', action='store_true')
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--gotrack-head', type=Path)
    parser.add_argument('--gotrack-head-sha256')
    parser.add_argument('--gotrack-inference-config', type=Path)
    parser.add_argument('--gotrack-inference-config-sha256')
    parser.add_argument('--gotrack-runtime-compat-config', type=Path)
    parser.add_argument('--gotrack-runtime-compat-config-sha256')
    parser.add_argument('--cloud-execution', action='store_true')
    parser.add_argument('--unlit-templates', action='store_true')
    parser.add_argument('--disable-multisampling', action='store_true')
    parser.add_argument('--stage-timeout-s', type=float, default=1800)
    parser.add_argument('--storage-limit-gib', type=float, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not (0 < args.stage_timeout_s <= 86400 and 0 < args.storage_limit_gib <= 8):
        parser.error('Stage timeout must be 0..86400 seconds; model storage budget 0..8 GiB')
    inference_identity = None
    if args.gotrack_inference_config is not None:
        from .quality_gotrack_inference import options_from_values, prepare_inference
        from .quality_runner import read_input
        options = options_from_values(
            head_path=args.gotrack_head, head_sha256=args.gotrack_head_sha256,
            config_path=args.gotrack_inference_config,
            config_sha256=args.gotrack_inference_config_sha256,
            compatibility_config_path=args.gotrack_runtime_compat_config,
            compatibility_config_sha256=args.gotrack_runtime_compat_config_sha256,
            cloud_execution=args.cloud_execution)
        context = prepare_inference(options, args.bundle, read_input(args.bundle))
        inference_identity = context.identity_dict()
    commands = stage_commands(args.bundle, args.output, device=args.device, stages=args.stages,
        limit=args.limit, ranker=args.ranker, checkpoint=args.checkpoint,
        patch_features=args.patch_features, masks=args.masks,
        gotrack_head=args.gotrack_head,
        gotrack_head_sha256=args.gotrack_head_sha256,
        gotrack_inference_config=args.gotrack_inference_config,
        gotrack_inference_config_sha256=args.gotrack_inference_config_sha256,
        gotrack_runtime_compat_config=args.gotrack_runtime_compat_config,
        gotrack_runtime_compat_config_sha256=args.gotrack_runtime_compat_config_sha256,
        cloud_execution=args.cloud_execution,
        unlit_templates=args.unlit_templates,
        disable_multisampling=args.disable_multisampling)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=1, bundle_sha256=digest(args.bundle / 'input.json'),
                   stages=[], complete=False, billing_stopped=None)
    if inference_identity is not None:
        summary['gotrack_inference'] = inference_identity
    try:
        for stage, command in commands:
            inference_output = None
            if stage == 'pose' and inference_identity is not None:
                inference_output = Path(command[command.index('--output') + 1])
            try:
                receipt = run_stage(stage, command, args.output, timeout_s=args.stage_timeout_s,
                    storage_limit_bytes=min(BUDGET, int(args.storage_limit_gib * 1024**3)), storage_root=CACHE,
                    inference_provenance=inference_identity if stage == 'pose' else None,
                    inference_output=inference_output)
            except RuntimeError:
                receipt_path = args.output / f'{stage}.receipt.json'
                if receipt_path.is_file():
                    failed_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                    summary['stages'].append(failed_receipt)
                    if stage == 'pose' and 'gotrack_inference' in failed_receipt:
                        summary['gotrack_inference'] = failed_receipt['gotrack_inference']
                    save_result(args.output / 'pipeline.json', summary)
                raise
            summary['stages'].append(receipt)
            if stage == 'pose' and inference_identity is not None:
                summary['gotrack_inference'] = receipt['gotrack_inference']
            save_result(args.output / 'pipeline.json', summary)
        summary['complete'] = True
    finally:
        save_result(args.output / 'pipeline.json', summary)


if __name__ == '__main__':
    main()
