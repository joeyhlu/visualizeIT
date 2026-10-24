"""Small adapters around the existing offline stages; no model ownership here.

Frame/MaskPrediction/PoseCandidate retain the canonical OpenCV/metre contract.
Evaluation references deliberately have no place in these inference interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Protocol, Sequence

import numpy as np

from .quality_contract import Frame, MaskPrediction, PoseCandidate


class Segmenter(Protocol):
    def segment(self, frame: Frame) -> MaskPrediction: ...
    def close(self) -> None: ...


class CandidateGenerator(Protocol):
    def recover(self, frame: Frame, mask: np.ndarray, top_k: int = 5) -> Sequence[np.ndarray]: ...


class PoseRefiner(Protocol):
    def refine(self, frame: Frame, mask: np.ndarray, seed: np.ndarray) -> PoseCandidate | None: ...


class TemporalValidator(Protocol):
    def update(self, frame: Frame, segmentation: MaskPrediction) -> dict: ...


class SurfaceRenderer(Protocol):
    def render_object_model(self, obj_id, camera_model_c2w, **kwargs): ...
    def close(self) -> None: ...


def memory_snapshot() -> dict:
    """Measured process high-water RSS and CUDA allocator peaks, when available.

    No torch import or CUDA initialization is performed solely for diagnostics.
    CUDA allocator values exclude the driver's and other processes' allocations.
    """
    result = {'process_peak_rss_bytes': None, 'cuda_peak_allocated_bytes': None,
              'cuda_peak_reserved_bytes': None}
    try:
        if sys.platform == 'win32':
            import psutil
            result['process_peak_rss_bytes'] = int(psutil.Process().memory_info().peak_wset)
        else:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['process_peak_rss_bytes'] = int(rss * (1 if sys.platform == 'darwin' else 1024))
    except (ImportError, AttributeError, OSError):
        pass
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        result['cuda_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        result['cuda_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
    return result


class CandidateTraceError(RuntimeError):
    """A required audit trace could not be persisted; abort this run."""


class CandidateTraceSink:
    """Bounded append-only frame trace, separate from public tracking output.

    A new output is required; an interrupted run cannot silently mix trajectories.
    Object/sequence identities are join keys only, never ranker features.
    """
    def __init__(self, path: Path, *, object_id: str, sequence_id: str,
                 provenance: dict, context: dict | None = None,
                 source_frame_ids: dict | None = None, max_bytes: int = 64 * 1024**2):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if max_bytes <= 0:
            raise ValueError('Candidate trace max_bytes must be positive')
        self.max_bytes = max_bytes
        self.identity = dict(object_id=str(object_id), sequence_id=str(sequence_id),
                             provenance=provenance)
        self.context = dict(context or {})
        self.source_frame_ids = dict(source_frame_ids or {})
        self.path.open('x', encoding='utf-8').close()
        self.started = time.perf_counter()

    def __call__(self, record: dict) -> None:
        row = {**record, **self.identity, 'trace_schema_version': 1,
               'context': {**record.get('context', {}), **self.context},
               'source_frame_id': self.source_frame_ids.get(record.get('frame_id'), record.get('frame_id')),
               'resource_memory': memory_snapshot(),
               'elapsed_s': time.perf_counter() - self.started}
        payload = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        if self.path.stat().st_size + len(payload) > self.max_bytes:
            raise CandidateTraceError('Candidate trace byte budget exhausted')
        # Open/close each frame: even a failed model stage leaves a readable prefix.
        try:
            with self.path.open('ab') as stream:
                stream.write(payload)
                stream.flush()
        except OSError as exc:
            raise CandidateTraceError('Candidate trace write failed') from exc


@dataclass(frozen=True)
class RankingOptions:
    name: str = 'legacy'
    checkpoint: Path | None = None
    trace: Path | None = None
    patch_features: bool = False

    @property
    def active(self) -> bool:
        return self.name != 'legacy' or self.trace is not None or self.patch_features

    def validate(self):
        if self.name not in ('legacy', 'deterministic', 'learned', 'confidence-research', 'ordering-research'):
            raise ValueError('Unknown pose ranker')
        if (self.name in ('learned', 'confidence-research', 'ordering-research')) != (self.checkpoint is not None):
            raise ValueError('A checkpoint ranker requires --ranker-checkpoint')
        if self.checkpoint is not None:
            from .quality_pose_ranker import make_ranker
            make_ranker(self.name, self.checkpoint)  # fail before loading any large model
        if self.trace is not None and self.trace.exists():
            raise ValueError('Candidate trace already exists; use a new run directory')
        return self


def configure_ranking(options: RankingOptions, *, refiner, manifest: dict,
                      provenance: dict):
    """Configure only opt-in ranking; legacy defaults do no feature work."""
    from .quality_pose_ranker import make_ranker
    ranker = make_ranker(options.name, options.checkpoint) if options.active else None
    # Worker implements this opt-in attribute; no extra encoder/model is loaded.
    refiner.collect_ranker_features = bool(
        options.patch_features or options.name in ('learned', 'confidence-research', 'ordering-research'))
    sink = None
    if options.trace is not None:
        context = {key: manifest[key] for key in ('physical_object_id', 'sequence_id',
            'scene_id', 'background_id', 'background_kind', 'provenance_kind') if key in manifest}
        context['mesh_sha256'] = manifest.get('mesh_sha256', manifest['source_hashes'][manifest['asset']])
        source_ids = manifest.get('original_source_frame_ids', manifest.get('_timing', {}).get('source_frame_ids', {}))
        sink = CandidateTraceSink(options.trace,
            object_id=manifest.get('physical_object_id', manifest['object']),
            sequence_id=manifest.get('sequence_id', provenance['input_manifest_sha256']),
            context=context,
            source_frame_ids={int(key): value for key, value in source_ids.items()},
            provenance={**provenance, 'ranker': options.name,
                        'patch_features': refiner.collect_ranker_features})
        if options.name == 'confidence-research':
            raw_sink = sink
            def sink(record):
                enriched = dict(record)
                # The contract seeds rank_decision even on frames where ranking
                # never ran. Consume the rank-call diagnostic once so a later
                # unavailable frame cannot inherit an earlier probability.
                diagnostic = ranker.consume_research_diagnostic()
                enriched['research_diagnostic'] = (None if diagnostic is None
                                                    else dict(diagnostic))
                raw_sink(enriched)
    return ranker, sink
