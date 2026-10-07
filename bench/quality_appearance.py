"""Experimental fixed-texture appearance evidence, independent of network flow.

Visible pixels always come from the current segmentation. Model depth is
rendered geometry, not measured camera depth. Conservative thresholds are a
research configuration, not validated general-object acceptance rules.
"""
from dataclasses import dataclass
import numpy as np
from .vision import cv2
from .quality_contract import Frame, canonical_pose
from .quality_time import LEGACY, PHYSICAL, FrameKey, TimePolicy, LEGACY_POLICY


@dataclass(frozen=True)
class AppearanceSettings:
    grayscale_min: float = .7
    gradient_min: float = .25
    minimum_pixels: int = 100
    minimum_visible_overlap: float = .5


def appearance_metrics(rgb, color, visible):
    gray = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2GRAY)
    model = cv2.cvtColor(color.astype(np.float32), cv2.COLOR_RGB2GRAY)
    support = cv2.erode(visible.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if support.sum() < 100: return dict(state='insufficient_overlap')
    a, b = gray[support], model[support]
    def corr(x, y):
        if x.std() < .005 or y.std() < .005: return None
        return float(np.corrcoef(x, y)[0, 1])
    ga = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    gb = np.hypot(cv2.Sobel(model, cv2.CV_32F, 1, 0), cv2.Sobel(model, cv2.CV_32F, 0, 1))
    return dict(state='measured', pixels=int(support.sum()), grayscale_correlation=corr(a, b),
                gradient_magnitude_correlation=corr(ga[support], gb[support]))


def appearance_decision(stats, settings):
    gray, gradient = stats.get('grayscale_correlation'), stats.get('gradient_magnitude_correlation')
    if (stats.get('state') != 'measured' or stats.get('pixels', 0) < settings.minimum_pixels
            or stats.get('visible_overlap', 1) < settings.minimum_visible_overlap or gray is None or gradient is None):
        return True, None  # Uninformative appearance is never positive evidence.
    valid = not (gray < settings.grayscale_min and gradient < settings.gradient_min)
    return valid, None if valid else 'contradicts_rendered_texture'


class AppearanceCheckedBackend:
    """Optional evidence hook; GoTrack still supplies every accepted pose."""
    def __init__(self, backend, renderer, settings=None, *, time_policy=LEGACY_POLICY):
        if not isinstance(time_policy, TimePolicy):
            raise ValueError('time_policy must be a TimePolicy')
        self.backend = backend; self.settings = settings or AppearanceSettings()
        if renderer is backend.refiner.renderer:
            raise ValueError('Appearance needs a separate renderer to preserve GoTrack crop context')
        self.time_policy = time_policy
        self.clock_modes_supported = frozenset((time_policy.mode,))
        if time_policy.mode == PHYSICAL:
            if (getattr(backend, 'time_policy', None) != time_policy or
                    PHYSICAL not in getattr(backend, 'clock_modes_supported', ()) or
                    not callable(getattr(backend, 'clear_motion', None))):
                raise ValueError('Physical appearance backend requires a matching timed backend protocol')
        self.renderer = renderer
        self.frame = None; self.mask = None
        self._chronology_key = None
        self._cleared_key = None

    def refine(self, *args): return self.backend.refine(*args)
    def recover(self, *args, **kwargs): return self.backend.recover(*args, **kwargs)
    def commit(self, *args): return self.backend.commit(*args)
    def motion_seed(self, *args): return self.backend.motion_seed(*args)

    def clear_motion(self, key: FrameKey, *, reason: str):
        if self.time_policy.mode != PHYSICAL or not isinstance(key, FrameKey) or key.clock_mode != PHYSICAL:
            raise ValueError('Appearance motion cleanup requires a physical FrameKey')
        if not isinstance(reason, str) or not reason:
            raise ValueError('Appearance motion cleanup requires a reason')
        if self._chronology_key is not None:
            if (key.frame_id != self._chronology_key.frame_id + 1 or
                    key.timestamp_s <= self._chronology_key.timestamp_s):
                raise ValueError('Appearance motion cleanup requires the next physical key')
        try:
            self.backend.clear_motion(key, reason=reason)
        except Exception:
            self.frame = None
            self.mask = None
            self._chronology_key = key
            self._cleared_key = None
            raise
        self.frame = None
        self.mask = None
        self._chronology_key = key
        self._cleared_key = key

    def observe(self, frame, mask):
        if not isinstance(frame, Frame) or frame.clock_mode != self.time_policy.mode:
            raise ValueError('Appearance backend clock mode mismatch')
        key = frame.key
        same_cleared_key = self.time_policy.mode == PHYSICAL and self._cleared_key == key
        if self.time_policy.mode == PHYSICAL and self._chronology_key is not None and not same_cleared_key:
            if (key.frame_id != self._chronology_key.frame_id + 1 or
                    key.timestamp_s <= self._chronology_key.timestamp_s):
                raise ValueError('Appearance backend requires consecutive physical frames')
        self.backend.observe(frame, mask)
        self.frame = frame; self.mask = mask
        if self.time_policy.mode == PHYSICAL:
            self._chronology_key = key
            self._cleared_key = None

    def validate_motion(self, candidate, frame_or_legacy_id):
        if self.time_policy.mode == PHYSICAL:
            if not isinstance(frame_or_legacy_id, Frame):
                raise ValueError('Physical appearance validation requires the current Frame')
            requested_key = frame_or_legacy_id.key
            motion_arg = frame_or_legacy_id
            if self.frame is None or self.frame.key != requested_key:
                raise ValueError('Current appearance frame key required')
        else:
            if isinstance(frame_or_legacy_id, Frame):
                raise ValueError('Legacy appearance validation requires an integer frame ID')
            requested_key = None
            motion_arg = frame_or_legacy_id
        valid, reason, stats = self.backend.validate_motion(candidate, motion_arg)
        if not valid: return valid, reason, stats
        if self.frame is None:
            raise ValueError('Current appearance frame required')
        if self.time_policy.mode == PHYSICAL:
            if self.frame.key != requested_key:
                raise ValueError('Current appearance frame key required')
        elif self.frame.frame_id != frame_or_legacy_id:
            raise ValueError('Current appearance frame required')
        if self.mask is None: return valid, reason, {**stats, 'appearance': {'state': 'unavailable'}}
        from utils import structs, renderer_base
        frame = self.frame; h, w = frame.rgb.shape[:2]; height = min(640, h); width = round(w*height/h)
        k = np.array(frame.intrinsics, copy=True); k[0] *= width/w; k[1] *= height/h
        rgb = cv2.resize(frame.rgb, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize((self.mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        pose = np.linalg.inv(canonical_pose(candidate.pose)); pose[:3, 3] *= 1000
        camera = structs.PinholePlaneCameraModel(width=width, height=height,
            f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]), T_world_from_eye=pose)
        refiner = self.backend.refiner
        output = self.renderer.render_object_model(refiner.obj_id, camera)
        visible = mask & (output[renderer_base.RenderType.DEPTH] > 0)
        appearance = appearance_metrics(rgb, output[renderer_base.RenderType.COLOR], visible)
        appearance['visible_overlap'] = float(visible.sum()/max(1, mask.sum()))
        valid, reason = appearance_decision(appearance, self.settings)
        return valid, reason, {**stats, 'appearance': appearance}
