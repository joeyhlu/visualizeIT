"""Experimental, model-free display policy. Decisions are never FrameResult evidence.

Callers authenticate source/row/mask bytes before consume. FrameBoundary performs
that authentication for live session rows. No candidate generator or backend
reacquisition/checkpoint restore is implemented here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from . import workbench_contract as wc
from . import workbench_geometry as wg

POLICY_VERSION = 'experimental_display_policy_v1'


class PolicyRejected(ValueError):
    """Integrity or admission failure; refuse instead of reporting tracking loss."""


def _digest(value):
    if not isinstance(value, str) or wc.SHA_RE.fullmatch(value) is None:
        raise PolicyRejected('identity must be a lowercase SHA-256 digest')
    return value


@dataclass(frozen=True)
class DisplayDecision:
    state: str
    pose: tuple[float, ...] | None
    opacity: float
    reason: str
    age_frames: int | None
    age_ms: float | None
    epoch: int
    frame_id: str
    accepted_measured: bool
    current_mask_trusted: bool

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SurfaceAdmission:
    """Expected identities from a trusted manifest, compared with actual records.

    Chart checks below are admission of declared fixture bounds, not a geometric
    proof of arbitrary surface reconstruction or independently verified anchors.
    """
    units: str
    metric_provenance: str
    calibration_sha256: str
    geometry_sha256: str
    anchor_sha256: str
    chart_sha256: str
    coordinate_frame: str = 'object_metric'

    def verify(self, *, calibration_sha256, geometry_sha256, anchor_record, chart_record):
        if self.units != 'metric' or self.metric_provenance not in ('measured_metric_cad', 'dataset_metric_cad'):
            raise PolicyRejected('only authoritative metric CAD bundles are admitted')
        if self.coordinate_frame != 'object_metric':
            raise PolicyRejected('coordinate frame must be object_metric')
        for value in (self.calibration_sha256, self.geometry_sha256, self.anchor_sha256, self.chart_sha256):
            _digest(value)
        if (calibration_sha256, geometry_sha256) != (self.calibration_sha256, self.geometry_sha256):
            raise PolicyRejected('calibration or geometry identity differs')
        if wc.canonical_sha256(anchor_record) != self.anchor_sha256 or wc.canonical_sha256(chart_record) != self.chart_sha256:
            raise PolicyRejected('anchor or chart identity differs')
        if not isinstance(anchor_record, Mapping) or not isinstance(chart_record, Mapping):
            raise PolicyRejected('anchor and chart must be records')
        if (not anchor_record.get('anchor_id') or anchor_record.get('geometry_sha256') != self.geometry_sha256
                or anchor_record.get('chart_id') != chart_record.get('chart_id')):
            raise PolicyRejected('persistent anchor correspondence differs')
        if chart_record.get('domain') not in ('plane', 'cylinder', 'observed_partial_mesh') or not chart_record.get('chart_id'):
            raise PolicyRejected('unsupported chart domain')
        if chart_record.get('orientation_valid') is not True or chart_record.get('overlap_free') is not True:
            raise PolicyRejected('chart orientation/fold/overlap is invalid')
        distortion, limit = chart_record.get('distortion'), chart_record.get('max_distortion')
        if (isinstance(distortion, bool) or isinstance(limit, bool) or not isinstance(distortion, (int, float))
                or not isinstance(limit, (int, float)) or not math.isfinite(distortion) or not math.isfinite(limit)
                or not 0 <= distortion <= limit):
            raise PolicyRejected('chart distortion exceeds its frozen limit')


@dataclass(frozen=True)
class ReplayIdentity:
    source_sha256: str
    mask_sha256: str
    head_sha256: str
    config_sha256: str
    policy_sha256: str
    state_sha256: str
    epoch: int

    def validate(self):
        for field in dataclasses.fields(self):
            if field.name != 'epoch':
                _digest(getattr(self, field.name))
        if type(self.epoch) is not int or self.epoch < 0:
            raise PolicyRejected('invalid replay epoch')


def verify_checkpoint(payload: bytes, *, expected: ReplayIdentity, expected_prefix: tuple[str, ...], expected_state: bytes):
    """Verify against independently trusted expectations, never a self-claimed hash.

    This authenticates a saved evidence contract. It does not restore neural state.
    Expected state bytes and causal prefix must come from the trusted same-arm run.
    """
    expected.validate()
    if not isinstance(expected_state, bytes) or hashlib.sha256(expected_state).hexdigest() != expected.state_sha256:
        raise PolicyRejected('trusted state bytes differ from expected state identity')
    if not expected_prefix or len(set(expected_prefix)) != len(expected_prefix):
        raise PolicyRejected('expected full causal prefix required')
    try:
        body = wc.loads_strict(payload)
        if set(body) != {'schema', 'identity', 'prefix_frame_ids', 'state_hex'} or body['schema'] != 'tracking_checkpoint_v1':
            raise PolicyRejected('checkpoint schema differs')
        decoded = body['identity']
        if not isinstance(decoded, dict) or set(decoded) != {field.name for field in dataclasses.fields(ReplayIdentity)}:
            raise PolicyRejected('checkpoint identity keys differ')
        authenticated = ReplayIdentity(**decoded)
        authenticated.validate()
        if dataclasses.asdict(authenticated) != dataclasses.asdict(expected) or tuple(body['prefix_frame_ids']) != tuple(expected_prefix):
            raise PolicyRejected('checkpoint arm/epoch/prefix identity differs')
        if bytes.fromhex(body['state_hex']) != expected_state:
            raise PolicyRejected('checkpoint state bytes differ')
    except (TypeError, KeyError, ValueError) as exc:
        raise PolicyRejected(f'checkpoint refused: {exc}') from exc


def verify_full_prefix(context):
    """Require every causal frame following the single-use authorized seed."""
    binding = context.binding
    bundle = binding.get('bundle')
    if not bundle or context.role not in ('primary_neural', 'frozen_baseline'):
        raise PolicyRejected('experimental policy requires a bound metric seed bundle')
    order = tuple(frame.frame_id for frame in context.clip.frames)
    seed = bundle['seed_frame_id']
    start, end = order.index(seed) + 1, order.index(context.scheduled_frame_ids[-1]) + 1
    if context.permitted_order != order[start:end]:
        raise PolicyRejected('full causal prefix from the authorized seed required; skipped frames refused')


@dataclass(frozen=True)
class RecoveryObservation:
    """Explicit current user re-onboarding, authenticated by the caller.

    Merely constructing this record does not authorize backend reseeding. Each
    observation must be current, uniquely identified and bound to source/mask.
    """
    authorization_id: str
    observation_id: str
    frame_id: str
    epoch: int
    image_sha256: str
    mask_sha256: str
    seed_label: str
    fresh_seed: bool
    identity_valid: bool
    orientation_valid: bool
    geometry_valid: bool
    calibration_valid: bool
    seed_sha256: str | None = None


class TrackingPolicy:
    """Bounded hold-last-pose prediction; no motion/visibility extrapolation.

    Prediction is opt-in, at most two consecutive frames, .67/.33 opacity, and
    at most 100 ms when genuine source timestamps are present. Null clocks use
    ordinals only. Recovery candidates are withheld until two fresh consistent
    observations. Recovering rows never produce artwork.
    """
    def __init__(self, *, allow_prediction=False):
        if type(allow_prediction) is not bool:
            raise PolicyRejected('allow_prediction must be boolean')
        self._allow_prediction = allow_prediction
        self._consumed_seed_sha256 = None
        self._consumed_seed_hashes = set()
        self.reset(0)

    @property
    def allow_prediction(self):
        return self._allow_prediction

    def identity(self):
        return dict(version=POLICY_VERSION, allow_prediction=self.allow_prediction,
                    max_prediction_frames=2, max_prediction_age_ms=100., fade=[.67, .33],
                    consistency_deg=10., consistency_relative_translation=.1,
                    prediction_method='hold_last_measured_pose', recovery='explicit_user_two_observations')

    def bind_initial_seed(self, seed_sha256):
        _digest(seed_sha256)
        if self._consumed_seed_sha256 is not None and self._consumed_seed_sha256 != seed_sha256:
            raise PolicyRejected('consumed original seed identity cannot be replaced')
        self._consumed_seed_sha256 = seed_sha256
        self._consumed_seed_hashes.add(seed_sha256)

    def reset(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise PolicyRejected('invalid epoch')
        self._epoch = epoch
        self._ordinal = None
        self._timestamp = None
        self._clock_null = None
        self._prior = None
        self._prior_ordinal = self._prior_timestamp = None
        self._hard_loss = False
        self._row_identity = None
        self._frame_ids = set()
        self._source_index = None
        self._candidate = None
        self._candidate_ordinal = None
        self._authorization = None
        self._observations = set()
        self._authorized_observations = {}
        self._active_authorization = None

    def _retire_recovery_event(self):
        self._authorized_observations = {}
        self._active_authorization = None
        self._authorization = None
        self._candidate = None
        self._candidate_ordinal = None

    def authorize_reonboarding(self, *, authorization_id: str, epoch: int,
                              expected_observations: tuple[RecoveryObservation, ...],
                              consumed_seed_sha256: str):
        """Register an explicit trusted user event, never data from row evidence.

        The caller must obtain fresh observations independently; this function
        checks their identity bindings only. FrameBoundary has no backend re-arm
        integration and therefore does not expose this as backend recovery.
        """
        _digest(consumed_seed_sha256)
        if type(epoch) is not int or epoch != self._epoch or not authorization_id or not expected_observations:
            raise PolicyRejected('explicit current-epoch re-onboarding authorization required')
        checked = {}
        frame_ids = set()
        for observation in expected_observations:
            if (not isinstance(observation, RecoveryObservation) or observation.authorization_id != authorization_id
                    or observation.epoch != epoch or observation.fresh_seed is not True
                    or not observation.seed_label.startswith('onboarding:')):
                raise PolicyRejected('fresh user onboarding observations required')
            _digest(observation.seed_sha256)
            _digest(observation.image_sha256)
            _digest(observation.mask_sha256)
            if observation.seed_sha256 == consumed_seed_sha256 or observation.seed_sha256 in self._consumed_seed_hashes:
                raise PolicyRejected('consumed historical seed cannot authorize recovery')
            if not observation.observation_id or observation.observation_id in checked:
                raise PolicyRejected('recovery observations must be distinct')
            if observation.frame_id in frame_ids:
                raise PolicyRejected('recovery observations require distinct source frames')
            frame_ids.add(observation.frame_id)
            checked[observation.observation_id] = observation
        # Validate the complete event before binding the original seed or making
        # any other mutation, so a refused duplicate-frame event is atomic.
        self.bind_initial_seed(consumed_seed_sha256)
        self._authorized_observations = checked
        self._active_authorization = authorization_id
        self._authorization = None
        self._candidate = None
        self._candidate_ordinal = None

    def _recovery_valid(self, observation, row, epoch, trusted):
        if not isinstance(observation, RecoveryObservation) or not trusted:
            return False
        r = observation
        return (self._active_authorization == r.authorization_id
                and self._authorized_observations.get(r.observation_id) == r
                and bool(r.authorization_id) and bool(r.observation_id) and r.observation_id not in self._observations
                and r.frame_id == row.frame_id and r.epoch == epoch
                and r.image_sha256 == row.image_sha256 and r.mask_sha256 == row.mask_sha256
                and r.fresh_seed is True and r.seed_label.startswith('onboarding:')
                and all(v is True for v in (r.identity_valid, r.orientation_valid, r.geometry_valid, r.calibration_valid)))

    def consume(self, row: wc.FrameResult, *, ordinal: int, epoch: int, current_mask_trusted: bool,
                ambiguous_identity: bool = False, recovery: RecoveryObservation | None = None) -> DisplayDecision:
        if type(epoch) is not int or epoch < 0 or epoch != self._epoch:
            raise PolicyRejected('epoch differs; explicitly reset before consuming')
        if type(ordinal) is not int or ordinal < 0 or (self._ordinal is not None and ordinal != self._ordinal + 1):
            raise PolicyRejected('causal ordinals must be consecutive')
        if type(current_mask_trusted) is not bool or type(ambiguous_identity) is not bool:
            raise PolicyRejected('trust and ambiguity must be explicit booleans')
        row = wc.FrameResult.from_dict(row.to_dict())
        identity = wc.canonical_sha256([row.run_id, row.clip_id, row.backend_role, row.backend_identity])
        if self._row_identity is not None and self._row_identity != identity:
            raise PolicyRejected('source/run/backend identity changed within epoch')
        if row.pose is not None and row.pose_units != 'metric':
            raise PolicyRejected('relative or unknown-scale pose cannot enter metric display policy')
        ts = row.timestamp_s
        if self._clock_null is not None and (ts is None) != self._clock_null:
            raise PolicyRejected('clock kind changed within epoch')
        if ts is not None and (not math.isfinite(ts) or (self._timestamp is not None and ts <= self._timestamp)):
            raise PolicyRejected('genuine timestamps must increase')
        if row.frame_id in self._frame_ids:
            raise PolicyRejected('frame_id must be unique within the epoch')
        if self._source_index is not None and row.source_frame_index <= self._source_index:
            raise PolicyRejected('source_frame_index must strictly increase within the epoch')
        # All rejection checks precede state mutation. Caller ordinals alone
        # cannot turn repeated source evidence into fresh recovery observations.
        self._row_identity = identity
        self._frame_ids.add(row.frame_id)
        self._source_index = row.source_frame_index
        self._clock_null = ts is None
        self._ordinal, self._timestamp = ordinal, ts
        trusted = (current_mask_trusted and row.mask_state == 'observed' and row.mask_frame_id == row.frame_id
                   and row.mask_sha256 is not None and row.image_sha256 is not None and not ambiguous_identity)
        age = None if self._prior is None else ordinal - self._prior_ordinal
        ms = None if self._prior is None or ts is None else (ts - self._prior_timestamp) * 1000
        def decision(state, pose=None, opacity=0.0, reason='tracking unavailable', measured=False):
            return DisplayDecision(state, pose, opacity, reason, age, ms, epoch, row.frame_id, measured, trusted)
        if row.pose_state == 'recovering':
            if self._prior is not None:
                self._hard_loss = True
            self._candidate = None
            self._candidate_ordinal = None
            return decision('recovering', reason='recovering: artwork suppressed')
        if row.pose is not None and trusted:
            pose = tuple(float(v) for v in np.asarray(row.pose).reshape(-1))
            if self._hard_loss:
                if not self._recovery_valid(recovery, row, epoch, trusted):
                    self._candidate = None
                    self._candidate_ordinal = None
                    return decision('recovering', reason='fresh explicit user re-onboarding required')
                self._observations.add(recovery.observation_id)
                # A seed is consumed as soon as it actually supplies an admitted
                # candidate, including an inconsistent or pending candidate. Its
                # active event may finish a fresh pair with the same seed, but a
                # later authorization/epoch can never reuse it.
                self._consumed_seed_hashes.add(recovery.seed_sha256)
                consistent = False
                if (self._candidate is not None and self._candidate_ordinal == ordinal - 1
                        and self._authorization == recovery.authorization_id):
                    a, b = np.asarray(self._candidate).reshape(4, 4), np.asarray(pose).reshape(4, 4)
                    angle = wg.rotation_angle_deg(a[:3, :3], b[:3, :3])
                    relative = np.linalg.norm(a[:3, 3] - b[:3, 3]) / max(np.linalg.norm(a[:3, 3]), 1e-9)
                    consistent = angle <= 10 and relative <= .1
                self._candidate, self._candidate_ordinal = pose, ordinal
                self._authorization = recovery.authorization_id
                if not consistent:
                    return decision('recovering', reason='first fresh consistent observation withheld')
                self._hard_loss = False
                self._retire_recovery_event()
            self._prior, self._prior_ordinal, self._prior_timestamp = pose, ordinal, ts
            age, ms = 0, None if ts is None else 0.0
            return decision('measured', pose, 1.0, 'accepted current measurement', True)
        self._candidate = None
        self._candidate_ordinal = None
        # Once hard loss has occurred it cannot be bypassed by later good rows.
        if (not self._hard_loss and self.allow_prediction and trusted and self._prior is not None
                and row.pose_state in ('lost', 'unavailable') and age in (1, 2)
                and (ms is None or ms <= 100.0 + 1e-9)):
            return decision('predicted', self._prior, .67 if age == 1 else .33,
                            'experimental hold-last-pose; current observed mask only')
        self._hard_loss = True
        self._retire_recovery_event()
        return decision('lost', reason='ambiguous identity' if ambiguous_identity else 'prediction expired or visibility unavailable')
