"""Read-only cloud preflight and cost estimate. Never provisions resources."""
from __future__ import annotations

import argparse
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from .quality_assets import MODELS, SOURCES, SUBMODULES, CACHE, ROOT, digest, save_result


def require_cloud():
    if platform.system() != 'Linux' or os.environ.get('VISUALIZEIT_CLOUD') != '1':
        raise RuntimeError('Run this command inside your approved Linux cloud workspace with VISUALIZEIT_CLOUD=1')


def estimate_cost(*, gpu_hours, gpu_usd_per_hour, storage_gib, storage_usd_per_gib_month,
                  storage_days, credits_usd=0):
    values = (gpu_hours, gpu_usd_per_hour, storage_gib, storage_usd_per_gib_month, storage_days, credits_usd)
    import math
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Cost inputs must be finite and nonnegative')
    compute = gpu_hours * gpu_usd_per_hour
    storage = storage_gib * storage_usd_per_gib_month * storage_days / 30
    return dict(compute_usd=compute, storage_usd=storage,
                estimated_out_of_pocket_usd=max(0, compute + storage - credits_usd),
                excludes='Taxes, deposits, egress, provider rounding; verify in account console',
                paid_resource_approval=False)


def preflight(root: Path, *, require_models=False):
    require_cloud()
    root = root.resolve(strict=True)
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'opencv-python-headless', 'pyrender', 'onnx', 'onnxruntime'):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    checks = {}
    for name, spec in MODELS.items():
        path = CACHE / 'checkpoints' / name
        checks[name] = dict(present=path.is_file(), expected_sha256=spec['sha256'])
        if path.is_file():
            checks[name].update(bytes=path.stat().st_size, sha256=digest(path))
            checks[name]['verified'] = checks[name]['sha256'] == spec['sha256']
        else:
            checks[name]['verified'] = False
    report = dict(schema_version=1, system=platform.platform(), python=platform.python_version(),
        packages=packages, git_revision=revision,
        source_revisions=SOURCES, submodules=SUBMODULES,
        adapter_sha256={p.name: digest(p) for p in sorted((ROOT / 'bench').glob('quality_*.py'))},
        checkpoints=checks, disk_free_bytes=shutil.disk_usage(root).free,
        cloud_root=str(root), opengl_platform=os.environ.get('PYOPENGL_PLATFORM'),
        account_free_credits_verified=False, billing_stopped=None,
        required_user_check='Confirm free compute AND persistent storage quota in provider console; disable automatic top-ups.')
    if require_models and not all(c['verified'] for c in checks.values()):
        raise ValueError('Pinned model checks failed; run cloud bootstrap or repair hashes before inference')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-models', action='store_true')
    p = sub.add_parser('estimate')
    p.add_argument('--gpu-hours', type=float, required=True,
                   help='Include preparation, templates, extraction, optimization, export and evaluation')
    p.add_argument('--gpu-usd-per-hour', type=float, required=True)
    p.add_argument('--storage-gib', type=float, required=True)
    p.add_argument('--storage-usd-per-gib-month', type=float, required=True)
    p.add_argument('--storage-days', type=float, required=True)
    p.add_argument('--credits-usd', type=float, default=0)
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        report = preflight(args.root, require_models=args.require_models)
        save_result(args.output, report)
    else:
        report = estimate_cost(**{k: v for k, v in vars(args).items() if k != 'command'})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
