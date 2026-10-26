"""Bounded native recorded SAM/GoTrack CLI execution; no training or LIVE support.

Copies no weights, creates no admission, relaxes no gate, and changes no model
values. Invoke with `--output NEW_DIR -- <native workbench CLI arguments>`.
Native JSON receipts/results remain authoritative; this adds resource and
observational-forward evidence. Keep raw receipts/private weight paths local.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[1]
MAX_RSS = 7 * 1024**3
MAX_OUTPUT = 512 * 1024**2
MIN_FREE_DISK = 10 * 1024**3
SOURCE_FILES = (
    'bench/workbench_neural.py', 'bench/workbench_cli.py', 'bench/workbench_contract.py',
    'bench/quality_gotrack.py', 'bench/quality_gotrack_inference.py',
    'bench/workbench_geometry.py', 'bench/workbench_render.py',
    'bench/workbench_frame_boundary.py', 'bench/workbench_telemetry.py',
    'bench/runtime/requirements-tracking-sam-cpu.txt',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def safe_base_loader(neural, torch):
    """Force safe CPU deserialization ONLY inside wn.load_base_network.

    Native export/reference BytesIO loads are untouched. No unsafe fallback,
    custom safe globals, or changes to protected loader files are permitted.
    """
    original_base, original_load = neural.load_base_network, torch.load

    def strict_load(*args, **kwargs):
        kwargs.update(weights_only=True, map_location='cpu')
        return original_load(*args, **kwargs)

    def guarded_base(*args, **kwargs):
        torch.load = strict_load
        try:
            return original_base(*args, **kwargs)
        finally:
            torch.load = original_load

    neural.load_base_network = guarded_base
    try:
        yield
    finally:
        neural.load_base_network = original_base
        torch.load = original_load


@contextmanager
def observe_network_loads(gotrack, destination):
    """Count actual module forwards; hooks return None and never retain tensors."""
    original, counts, handles = gotrack.load_network, {}, []
    networks, own_hooks = [], set()

    def observed_loader(device):
        network = original(device)
        networks.append(weakref.ref(network))
        key = 'network_' + str(len(counts))
        counts[key] = dict(backbone=0, decoder=0, pose_head=0)

        def make_hook(component):
            def hook(_module, _inputs, _output):
                counts[key][component] += 1
                if component == 'pose_head':
                    write_json(destination, counts)
            return hook

        for component in counts[key]:
            hook = make_hook(component)
            own_hooks.add(hook)
            handles.append(getattr(network, component).register_forward_hook(hook))
        return network

    gotrack.load_network = observed_loader
    try:
        yield counts
    finally:
        gotrack.load_network = original
        for handle in handles:
            handle.remove()
        # Native installers deepcopy the head after registration. Functions keep
        # identity through deepcopy, whereas RemovableHandle points at the old
        # hook dictionary. Clean only our hooks on current modules as well.
        for reference in networks:
            network = reference()
            if network is None:
                continue
            for component in ('backbone', 'decoder', 'pose_head'):
                module = getattr(network, component, None)
                hooks = getattr(module, '_forward_hooks', {})
                for key, hook in list(hooks.items()):
                    if hook in own_hooks:
                        del hooks[key]
        write_json(destination, counts)


def output_size(paths):
    files = set()
    for base in map(Path, paths):
        if not base.exists():
            continue
        for path in (base.rglob('*') if base.is_dir() else [base]):
            if path.is_file():
                files.add(path.resolve())
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # Native PNG/JSON publication can atomically rename/unlink a temp
            # file between this sampled directory walk and its stat. Ignore
            # only disappeared files; all other accounting errors fail closed.
            continue
    return total


def guard_reason(*, rss, elapsed, free_disk, output_bytes, max_seconds, max_rss,
                 max_output=MAX_OUTPUT, min_free_disk=MIN_FREE_DISK):
    if rss > max_rss:
        return 'RSS limit exceeded'
    if elapsed > max_seconds:
        return 'wall-time limit exceeded'
    if free_disk < min_free_disk:
        return '10-GiB disk reserve violated'
    if output_bytes > max_output:
        return 'output growth limit exceeded'
    return None


def stage_budget(profile, command, requested=None, roles=None):
    """Predeclared measured wall ceilings; no numerical/resource gate changes."""
    if profile not in ('pilot', 'recorded300', 'recorded300pair'):
        raise ValueError('unknown bounded budget profile')
    ceiling = 900 if profile == 'pilot' else {'segment': 1800, 'run': 2700, 'gate': 900}.get(command, 0)
    if profile == 'recorded300pair' and command == 'run':
        declared = roles.split(',') if isinstance(roles, str) else list(roles or ())
        if declared != ['primary_neural', 'frozen_baseline']:
            raise ValueError('pair budget requires explicit primary_neural,frozen_baseline in native order')
        ceiling = 5400
    value = ceiling if requested is None else requested
    if not 0 < value <= ceiling:
        raise ValueError('requested wall time exceeds the declared stage ceiling')
    return value, ceiling


def changed_sources(root, frozen):
    changed = []
    for relative, expected in frozen.items():
        try:
            same = digest(Path(root) / relative) == expected
        except OSError:
            same = False
        if not same:
            changed.append(relative)
    return changed


def supervise(command, output, *, watch_output=(), max_seconds=900., max_rss=MAX_RSS,
              max_output=MAX_OUTPUT, min_free_disk=MIN_FREE_DISK, sample_seconds=.25,
              root=ROOT, frozen_sources=None, environment=None, metadata=None, wall_ceiling=900.):
    """Run one child process group; preserve terminal/partial evidence on failure.

    Lower injectable limits support tiny tests; the CLI never raises production
    ceilings. Output accounting measures new growth in watched native paths, not
    pre-existing datasets or checkpoints. RSS is sampled, not a continuous peak.
    """
    import psutil
    if not 0 < max_seconds <= wall_ceiling <= 5400 or not 0 < max_rss <= MAX_RSS:
        raise ValueError('wall/RSS limits exceed the declared bounded stage ceiling/7GiB')
    if not 0 < max_output <= MAX_OUTPUT or not 0 < sample_seconds <= .25:
        raise ValueError('output/sample limits exceed bounded protocol')
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(frozen_sources or {})
    watched = [output, *map(Path, watch_output)]
    baseline_output_bytes = output_size(watched)
    write_json(output / 'input-freeze.json', dict(
        nativeCLIArgs=(metadata or {}).get('nativeCLIArgs'), budgetProfile=(metadata or {}).get('budgetProfile'), sourceSHA256=frozen,
        runnerSHA256=digest(__file__), maxSeconds=max_seconds, maxRSSBytes=max_rss,
        maxOutputGrowthBytes=max_output, minimumFreeDiskBytes=min_free_disk,
        rssSamplingSeconds=sample_seconds, watchedPaths=list(map(str, watched)),
        baselineWatchedBytes=baseline_output_bytes, threads=dict(torch=8, interop=1, OpenBLAS=1, OpenCV=1),
        instrumentation='observational forward hooks only; no tensor/parameter/output changes',
        trainingSteps=0))
    start, peak, reason, code, child = time.monotonic(), 0, None, None, None
    owned_descendants = {}
    if shutil.disk_usage(root).free < min_free_disk + max_output:
        reason = 'insufficient disk reserve plus bounded output budget'
    try:
        if reason is None:
            with (output / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                print('STARTED', child.pid, flush=True)
                while child.poll() is None:
                    try:
                        process = psutil.Process(child.pid)
                        descendants = process.children(recursive=True)
                        owned_descendants.update({p.pid: p for p in descendants})
                        rss = sum(p.memory_info().rss for p in [process, *descendants] if p.is_running())
                    except psutil.NoSuchProcess:
                        break
                    peak = max(peak, rss)
                    reason = guard_reason(rss=rss, elapsed=time.monotonic() - start,
                        free_disk=shutil.disk_usage(root).free,
                        output_bytes=max(0, output_size(watched) - baseline_output_bytes),
                        max_seconds=max_seconds, max_rss=max_rss, max_output=max_output,
                        min_free_disk=min_free_disk)
                    if reason:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(sample_seconds)
                code = child.wait()
    except BaseException as exc:
        reason = f'supervisor failure: {type(exc).__name__}: {exc}'[:1000]
        if isinstance(exc, KeyboardInterrupt):
            reason = 'supervisor interrupted'
    finally:
        if child is not None:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait()
            # The root can exit before a video decoder/other descendant. Its
            # owned process group must not escape the resource/time budget.
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                pass
            else:
                reason = reason or 'worker exited leaving live descendants; owned process group terminated'
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Also stop observed descendants that established their own process
            # group (normal native decoders do not need to escape supervision).
            for descendant in owned_descendants.values():
                try:
                    if descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE:
                        reason = reason or 'worker exited leaving live descendants; owned process tree terminated'
                        descendant.kill()  # psutil checks PID reuse before signalling
                except psutil.NoSuchProcess:
                    pass
    elapsed = time.monotonic() - start
    if reason is None:
        reason = guard_reason(rss=peak, elapsed=elapsed, free_disk=shutil.disk_usage(root).free,
            output_bytes=max(0, output_size(watched) - baseline_output_bytes), max_seconds=max_seconds,
            max_rss=max_rss, max_output=max_output, min_free_disk=min_free_disk)
    drift = changed_sources(root, frozen)
    result = dict(exitCode=code, killedReason=reason, elapsedSeconds=elapsed,
                  peakSampledRSSBytes=peak, sourceDrift=drift, rssSamplingSeconds=sample_seconds,
                  maxSeconds=max_seconds, maxRSSBytes=max_rss, gpuUsed=False, trainingSteps=0)
    write_json(output / 'supervision.json', result)
    print(json.dumps(result), flush=True)
    return result


def native_plan(arguments):
    sys.path.insert(0, str(ROOT))
    from bench import workbench_cli
    native = workbench_cli.build_parser().parse_args(arguments)
    if native.command not in ('gate', 'run', 'segment'):
        raise ValueError('only native recorded gate/run/segment commands are supported')
    if getattr(native, 'device', 'cpu') != 'cpu':
        raise ValueError('this bounded protocol is CPU-only; non-CPU devices are refused')
    workspace = Path(native.workspace).resolve()
    paths = []
    if native.command in ('run', 'segment'):
        paths += [workspace / name for name in ('runs', 'masks', 'derived', 'scratch')]
    elif native.gate == 'geometry':
        paths += [workspace / 'scratch', Path(native.out), Path(native.out).parent / 'geometry-evidence']
    elif hasattr(native, 'out'):
        paths += [Path(native.out)]
    return native.command, paths, getattr(native, 'roles', None)


def worker(output, arguments):
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT))
    from bench.vision import cv2
    from bench import workbench_neural as neural, quality_gotrack, workbench_cli
    cv2.setNumThreads(1)
    settings = neural.deterministic_torch(torch)
    write_json(output / 'runtime.json', dict(torch=torch.__version__, torchThreads=torch.get_num_threads(),
        torchInteropThreads=torch.get_num_interop_threads(), openCVThreads=cv2.getNumThreads(),
        logicalCPUs=os.cpu_count(), platform=platform.platform(),
        OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'), OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
        CUDAAvailable=torch.cuda.is_available(), strictLoadScope='wn.load_base_network only; native head/reference loads untouched',
        deterministicSettings=settings, trainingSteps=0))
    with safe_base_loader(neural, torch), observe_network_loads(quality_gotrack, output / 'observed-full-forward-counts.json'):
        return workbench_cli.main(arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--budget-profile', choices=('pilot', 'recorded300', 'recorded300pair'), default='pilot')
    parser.add_argument('--max-rss', type=int, default=MAX_RSS)
    parser.add_argument('--watch-output', type=Path, action='append', default=[])
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('cli', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.cli[1:] if args.cli and args.cli[0] == '--' else args.cli
    if not arguments:
        parser.error('supply native CLI arguments after --')
    command_kind, watched, declared_roles = native_plan(arguments)
    max_seconds, ceiling = stage_budget(args.budget_profile, command_kind, args.max_seconds, declared_roles)
    if args._worker:
        if os.environ.get('VISUALIZEIT_BOUNDED_NATIVE_CHILD') != '1':
            raise ValueError('internal worker must be launched through the bounded supervisor')
        return worker(args.output.resolve(), arguments)
    frozen = {name: digest(ROOT / name) for name in SOURCE_FILES}
    frozen[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(__file__)
    environment = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', XFORMERS_DISABLED='1', SAM2_BUILD_CUDA='0', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', PYOPENGL_PLATFORM='egl', LIBGL_ALWAYS_SOFTWARE='1', MESA_SHADER_CACHE_DISABLE='true',
        VISUALIZEIT_BOUNDED_NATIVE_CHILD='1')
    command = [sys.executable, str(Path(__file__).resolve()), '--_worker', '--output', str(args.output.resolve()),
               '--budget-profile', args.budget_profile, '--', *arguments]
    result = supervise(command, args.output, watch_output=[*watched, *args.watch_output],
        max_seconds=max_seconds, wall_ceiling=ceiling, max_rss=args.max_rss, environment=environment,
        frozen_sources=frozen, metadata=dict(nativeCLIArgs=arguments, budgetProfile=args.budget_profile))
    return 0 if result['exitCode'] == 0 and result['killedReason'] is None and not result['sourceDrift'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
