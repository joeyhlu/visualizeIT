"""Run prototype modules with explicit package paths; never execute site/.pth hooks."""
from pathlib import Path
import runpy
import sys


def main():
    if not sys.flags.no_site:
        raise SystemExit('Use python -B -S tools/prototype_bootstrap.py MODULE [arguments].')
    root = Path(__file__).resolve().parents[1]
    packages = Path(sys.executable).resolve().parent / 'Lib' / 'site-packages'
    sys.path[:0] = [str(root), str(root / '.cache' / 'vision'),
                    str(root / '.cache' / 'plotting'), str(packages)]
    if len(sys.argv) < 2:
        raise SystemExit('A module is required.')
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
