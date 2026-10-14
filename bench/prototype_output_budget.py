"""Actual compact/HB result writers; retained partials remain in every ledger."""
from __future__ import annotations
import io
import json
from pathlib import Path
import shutil

UNIT = 4096
RESULT_CAP = 128 * 1024 * 1024
SCRATCH_CAP = 32 * 1024 * 1024
CACHE_CAP = 416 * 1024 * 1024
TOTAL_CAP = 1_000_000_000
FREE_FLOOR = 1024 * 1024 * 1024
SPRINT = Path(__file__).resolve().parents[1] / '.cache/prototype-sprint'
TEMP_SUFFIXES = ('.tmp', '.part', '.pending')


class CompactBudgetExceeded(RuntimeError):
    """Abort a compact stage without an inference seal/completion marker."""


def allocated(size):
    return ((size + UNIT - 1) // UNIT) * UNIT


def ledger(root, *, scratch_only=False):
    root = Path(root)
    if not root.exists():
        return 0
    total = 0 if scratch_only else UNIT
    for path in root.rglob('*'):
        if path.is_symlink():
            raise CompactBudgetExceeded('Symlink in accounted result/cache/sprint tree')
        if path.is_dir():
            if not scratch_only:
                total += UNIT
        elif path.is_file() and (not scratch_only or path.name.endswith(TEMP_SUFFIXES)):
            total += max(UNIT, allocated(path.stat().st_size))
    return total


class CompactOutputBudget:
    def __init__(self, cache, result, *, sprint_root=None, free_probe=None):
        self.sprint = Path(sprint_root or SPRINT).resolve()
        self.cache, self.result = Path(cache).resolve(), Path(result).resolve()
        if (self.result == self.sprint or not self.result.is_relative_to(self.sprint)
                or self.cache == self.sprint or not self.cache.is_relative_to(self.sprint)
                or self.result.is_relative_to(self.cache) or self.cache.is_relative_to(self.result)):
            raise CompactBudgetExceeded('Compact result/cache must be separate accounted sprint descendants')
        self.free_probe = free_probe or (lambda: shutil.disk_usage(self.sprint).free)
        self.check()

    def check(self, growth=0, *, scratch_growth=0):
        if growth < 0 or scratch_growth < 0:
            raise CompactBudgetExceeded('Negative projected output growth')
        if ledger(self.result) + growth > RESULT_CAP:
            raise CompactBudgetExceeded('Compact results exceed 128 MiB')
        if ledger(self.sprint, scratch_only=True) + scratch_growth > SCRATCH_CAP:
            raise CompactBudgetExceeded('Retained scratch and temporary peak exceed 32 MiB')
        if ledger(self.cache) > CACHE_CAP:
            raise CompactBudgetExceeded('HB compact cache exceeds 416 MiB')
        if ledger(self.sprint) + growth > TOTAL_CAP:
            raise CompactBudgetExceeded('Retained sprint exceeds 1,000,000,000 bytes')
        if self.free_probe() - growth < FREE_FLOOR:
            raise CompactBudgetExceeded('Real free-after-growth is below 1 GiB')

    def path(self, path):
        path = Path(path).resolve()
        if path != self.result and not path.is_relative_to(self.result):
            raise CompactBudgetExceeded('Output is outside the accounted result root')
        return path

    def mkdir(self, path):
        path = self.path(path)
        missing = []
        cursor = path
        while not cursor.exists():
            if cursor == self.sprint:
                raise CompactBudgetExceeded('Accounted sprint root must already exist')
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            self.check(UNIT)
            directory.mkdir()

    def open_partial(self, destination):
        destination = self.path(destination)
        self.mkdir(destination.parent)
        temporary = destination.with_name(destination.name + '.part')
        if temporary.exists():
            raise CompactBudgetExceeded('Retained partial blocks retry; use a fresh result root')
        self.check(UNIT, scratch_growth=UNIT)
        source = temporary.open('x+b')
        return GuardedFile(source, self, temporary), destination

    def finish(self, writer, destination):
        writer.flush()
        writer.close()
        self.check()  # Old destination and full temporary are still counted together.
        writer.path.replace(self.path(destination))

    def write_bytes(self, destination, raw):
        writer, destination = self.open_partial(destination)
        try:
            writer.write(raw)
            self.finish(writer, destination)
        finally:
            writer.close()  # Failed partial remains accounted and is never removed.

    def write_json(self, destination, value):
        self.write_bytes(destination, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'))

    def copy(self, source, destination):
        writer, destination = self.open_partial(destination)
        try:
            with Path(source).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    writer.write(chunk)
            self.finish(writer, destination)
        finally:
            writer.close()

    def npz(self, destination, **arrays):
        import numpy as np
        writer, destination = self.open_partial(destination)
        try:
            np.savez_compressed(writer, **arrays)
            self.finish(writer, destination)
        finally:
            writer.close()


class GuardedFile(io.BufferedIOBase):
    """A seekable NPZ/ZIP sink checking every actual allocated extent increase."""
    def __init__(self, source, budget, path):
        super().__init__()
        self.source, self.budget, self.path = source, budget, path
        self.extent = source.seek(0, 2)

    def writable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        return self.source.seek(offset, whence)

    def tell(self):
        return self.source.tell()

    def write(self, raw):
        end = self.source.tell() + len(raw)
        # The ledger conservatively charges an empty partial one unit, but its
        # first real write can still allocate physical disk. Charge that first
        # extent again for this boundary so a fresh free-space drop cannot pass.
        increase = max(0, allocated(end) - allocated(self.extent))
        self.budget.check(increase, scratch_growth=increase)
        count = self.source.write(raw)
        self.source.flush()  # Next ledger includes this actual partial extent.
        self.extent = max(self.extent, self.source.tell())
        return count

    def flush(self):
        if not self.source.closed:
            self.source.flush()

    def close(self):
        if not self.closed:
            super().close()
            self.source.close()
