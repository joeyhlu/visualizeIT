"""Versioned supervision facade; immutable adapter remains the validation service.

No models are imported. Output accounting uses the authenticated portable scanner;
runtime link accounting retries complete immutable-adapter traversals only.
"""
from __future__ import annotations

import signal
import subprocess
import threading
import time
import os
import select


VERSION = 'kaggle_supervision_v1'
ADAPTER_SHA256 = '1823fc93ef95057a3f2e4928c3d43123f130ce719683260fdce4e5f58212b388'
SCANNER_SHA256 = '48d91331c39a43602d5b9ec2d87f005dfeee1c65dca563109a15fb8f633aadc4'


def stable_tree_bytes(adapter, roots, *, approved_links=None):
    """Retry the whole approved-link traversal; never accept a partial sum."""
    roots = tuple(roots)
    for attempt in range(3):
        try:
            return adapter.unique_tree_bytes(roots, approved_links=approved_links)
        except FileNotFoundError:
            if attempt == 2:
                raise


class Facade:
    """Explicit dependency for old inspection/validation APIs; no module mutation."""
    def __init__(self, adapter, scanner):
        from pathlib import Path
        if adapter.file_sha256(Path(adapter.__file__)) != ADAPTER_SHA256:
            raise ValueError('immutable validation adapter changed')
        if adapter.file_sha256(Path(scanner.__file__)) != SCANNER_SHA256:
            raise ValueError('immutable no-follow scanner changed')
        self.adapter, self.scanner = adapter, scanner

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def unique_tree_bytes(self, roots, *, approved_links=None):
        return stable_tree_bytes(self.adapter, roots, approved_links=approved_links)

    def supervise(self, argv, *, cwd, output_root, working, log_path, limits, env):
        return supervise(self.adapter, self.scanner, argv, cwd=cwd, output_root=output_root,
            working=working, log_path=log_path, limits=limits, env=env)


def supervise(adapter, scanner, argv, *, cwd, output_root, working, log_path, limits, env):
    """Bound process group, storage, reserve and logs; preserve cancellation reason."""
    started = time.monotonic()
    child = reader = log = None
    overflow = threading.Event()
    reader_stop = threading.Event()
    reader_errors = []
    peak_rss = 0
    reason = None
    status = 'failed'
    returncode = None
    old_handlers = {}
    cleanup_errors = []

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    def finish_reader():
        if reader is not None:
            reader_stop.set()
            reader.join(timeout=5)
            if reader.is_alive():
                cleanup_errors.append('log_reader_did_not_stop')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        log = log_path.open('xb')
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')

        def copy_log():
            written = 0
            try:
                while True:
                    if os.name == 'posix':
                        ready, _, _ = select.select([child.stdout], [], [], .1)
                        if not ready:
                            if reader_stop.is_set():
                                # A writer escaped the managed process group.
                                reader_errors.append('log pipe remained open after process-group cleanup')
                                overflow.set()
                                break
                            continue
                        chunk = os.read(child.stdout.fileno(), 65536)
                    else:
                        chunk = child.stdout.read(65536)
                    if not chunk:
                        break
                    remaining = max(0, limits['max_log_bytes'] - written)
                    log.write(chunk[:remaining])
                    written += min(len(chunk), remaining)
                    log.flush()
                    if len(chunk) > remaining:
                        overflow.set()
            except (OSError, ValueError) as error:
                reader_errors.append(f'{type(error).__name__}: {error}')
                overflow.set()

        reader = threading.Thread(target=copy_log, daemon=True)
        reader.start()
        while child.poll() is None:
            peak_rss = max(peak_rss, adapter.group_rss(child.pid))
            if time.monotonic() - started >= limits['max_seconds']:
                reason = 'deadline_exceeded'
            elif peak_rss > limits['max_rss_bytes']:
                reason = 'process_group_rss_exceeded'
            elif overflow.is_set():
                reason = 'log_limit_or_log_io_error'
            elif scanner._usage((output_root,)) > limits['max_output_bytes']:
                reason = 'output_storage_exceeded'
            elif min(adapter.free_by_filesystem((working,)).values()) < limits['reserve_bytes']:
                reason = 'filesystem_reserve_exhausted'
            if reason:
                break
            time.sleep(.1)
        adapter.terminate_group(child)
        finish_reader()
        returncode = child.returncode
        if overflow.is_set():
            reason = reason or 'log_limit_or_log_io_error'
        if scanner._usage((output_root,)) > limits['max_output_bytes']:
            reason = reason or 'output_storage_exceeded'
        if cleanup_errors:
            reason = reason or 'log_pipe_not_closed'
        if reason or returncode == 3:
            status, reason = 'incomplete', reason or 'child_reported_incomplete'
        elif returncode == 0:
            status = 'completed'
        else:
            reason = 'child_failed'
    except KeyboardInterrupt:
        status, reason = 'incomplete', 'interrupted'
    except scanner.ResourceError as error:
        status, reason = 'incomplete', f'ResourceError: {error}'
    except Exception as error:
        status, reason = 'failed', f'{type(error).__name__}: {error}'
    finally:
        if child is not None:
            try:
                adapter.terminate_group(child)
                returncode = child.returncode
            except Exception as error:
                cleanup_errors.append(f'group_cleanup: {type(error).__name__}: {error}')
        try:
            finish_reader()
        except Exception as error:
            cleanup_errors.append(f'reader_cleanup: {type(error).__name__}: {error}')
        # Never close the log while the writer is still running.
        if log is not None and (reader is None or not reader.is_alive()):
            log.close()
        if child is not None and child.stdout and (reader is None or not reader.is_alive()):
            child.stdout.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if cleanup_errors:
        status, reason = 'failed', reason or 'supervision_cleanup_failed'
    return dict(status=status, reason=reason, returncode=returncode,
        elapsed_seconds=time.monotonic() - started, peak_group_rss_bytes=peak_rss,
        log=str(log_path), log_errors=reader_errors, cleanup_errors=cleanup_errors,
        outputs_preserved=True, supervision_version=VERSION)
