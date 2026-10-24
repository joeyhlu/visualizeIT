"""Frozen, inference-only feature schema for pose-candidate ranking.

Feature extraction consumes only current-image validation evidence, causal
tracker state already measured by the quality contract, and optional features
computed from the actual GoTrack query/template pair. It never accepts labels.
"""
from collections.abc import Mapping
import math


FEATURE_SCHEMA_VERSION = 1

# Every scalar is followed by an explicit availability bit. Missing values are
# encoded as (0, 0), while present values are encoded as (value, 1).
_PAIRED_FIELDS = (
    ('correspondences_log1p', 'correspondences'),
    ('inlier_ratio', 'inlier_ratio'),
    ('mean_inlier_confidence', 'mean_inlier_confidence'),
    ('median_reprojection_720', 'median_reprojection_720'),
    ('p95_reprojection_720', 'p95_reprojection_720'),
    ('spatial_support', 'spatial_support'),
    ('validation_score', 'score'),
    ('rotation_from_last_valid_degrees', 'rotation_from_last_valid_degrees'),
    ('translation_from_last_valid_m', 'translation_from_last_valid_m'),
    ('seconds_since_last_valid', 'seconds_since_last_valid'),
    ('frames_since_last_valid', 'frames_since_last_valid'),
)
_APPEARANCE_FIELDS = (
    'appearance_patch_cosine_00',
    'appearance_patch_cosine_01',
    'appearance_patch_cosine_10',
    'appearance_patch_cosine_11',
    'appearance_valid_fraction',
)

FEATURE_NAMES = tuple(
    name
    for value_name, _ in _PAIRED_FIELDS
    for name in (value_name, f'has_{value_name}')
) + tuple(
    name
    for value_name in _APPEARANCE_FIELDS
    for name in (value_name, f'has_{value_name}')
)
FEATURE_VECTOR_SHAPE = (len(FEATURE_NAMES),)


def _finite_scalar(value):
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def extract_features(stats, extra=None):
    """Return a fixed-length finite float list with explicit missing bits.

    ``stats`` is the same-frame validation mapping. ``extra`` is an optional
    detached GoTrack appearance summary, with fields named exactly as the
    appearance entries in ``FEATURE_NAMES``. Values that are unavailable,
    malformed, or nonfinite are represented as zero with a zero availability
    bit. Correspondence counts are stored as ``log1p(count)``; inlier ratio is
    derived only from same-frame inlier/correspondence counts when absent.
    """
    stats = _mapping(stats)
    extra = _mapping(extra)
    values = []
    for value_name, stats_name in _PAIRED_FIELDS:
        raw = stats.get(stats_name)
        if value_name == 'correspondences_log1p':
            scalar = _finite_scalar(raw)
            if scalar is not None and scalar >= 0:
                scalar = math.log1p(scalar)
            else:
                scalar = None
        elif value_name == 'inlier_ratio':
            scalar = _finite_scalar(stats.get('inlier_ratio'))
            if scalar is None:
                count = _finite_scalar(stats.get('correspondences'))
                inliers = _finite_scalar(stats.get('inliers'))
                scalar = (inliers / count if count is not None and count > 0 and
                          inliers is not None and 0 <= inliers <= count else None)
            if scalar is not None and not 0 <= scalar <= 1:
                scalar = None
        else:
            scalar = _finite_scalar(raw)
        if scalar is None:
            values.extend((0.0, 0.0))
        else:
            values.extend((float(scalar), 1.0))

    for name in _APPEARANCE_FIELDS:
        scalar = _finite_scalar(extra.get(name))
        if name.startswith('appearance_patch_cosine_') and scalar is not None:
            # Cosine similarity is mathematically bounded; tolerate tiny
            # floating-point drift but reject malformed summaries.
            if scalar < -1.000001 or scalar > 1.000001:
                scalar = None
            else:
                scalar = min(1.0, max(-1.0, scalar))
        elif name == 'appearance_valid_fraction' and scalar is not None:
            if scalar < 0 or scalar > 1:
                scalar = None
        if scalar is None:
            values.extend((0.0, 0.0))
        else:
            values.extend((float(scalar), 1.0))

    if len(values) != FEATURE_VECTOR_SHAPE[0]:
        raise AssertionError('Frozen pose-ranker feature schema length changed')
    return values

