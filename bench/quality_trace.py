"""Opt-in inference diagnostics; never reads evaluation data."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def trace(stage, **arrays):
    target = os.environ.get('VISUALIZEIT_QUALITY_TRACE')
    if not target:
        return
    record = {'stage': stage, 'arrays': {}}
    for key, value in arrays.items():
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        record['arrays'][key] = {
            'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'mean': float(value.mean()) if value.size else None,
        }
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
