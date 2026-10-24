"""Explicit Kaggle RAW capture stage entry point; never downloads assets."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.quality_adaptation_portable import main

if __name__ == '__main__':
    raise SystemExit(main())
