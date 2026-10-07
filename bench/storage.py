"""Bounded atomic artifact output; leave the same disk reserve as acquisition."""
from io import BytesIO
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 256*1024*1024  # Native HD video derivatives need more room than GIF studies.
RESERVE = 512*1024*1024


def write_artifact(path, data):
    path = Path(path)
    if isinstance(data, str): data = data.encode('utf-8')
    used = sum(p.stat().st_size for directory in ('mapping-improvements','video-tracking','video-interruptions','video-hd','video-60','model-quality')
               for p in (ROOT/'artifacts'/directory).rglob('*') if p.is_file())
    prior = path.stat().st_size if path.exists() else 0
    if used-prior+len(data) > LIMIT or shutil.disk_usage(path.parent).free-len(data) < RESERVE:
        raise ValueError('Artifact budget/disk reserve reached; existing results retained.')
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        temporary.write_bytes(data); temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_image(image, path, **kwargs):
    from PIL import Image
    buffer = BytesIO()
    image.save(buffer, format=Image.registered_extensions()[Path(path).suffix.lower()], **kwargs)
    write_artifact(path, buffer.getvalue())


def save_arrays(path, **kwargs):
    import numpy as np
    buffer = BytesIO(); np.savez_compressed(buffer, **kwargs)
    write_artifact(path,buffer.getvalue())
