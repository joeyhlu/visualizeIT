"""Use the isolated workspace wheel when present; ordinary venvs work unchanged."""
from pathlib import Path
import sys
import os

try:
    import cv2
except ModuleNotFoundError:
    local = Path(__file__).resolve().parents[1]/'.cache'/'vision'
    if os.name != 'nt' or not local.exists():
        raise
    sys.path.insert(0, str(local))
    import cv2
