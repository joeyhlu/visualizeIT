"""Private, image-supported motion evidence for points on the fixed model.

This is a recovery seed and an independent correspondence check, never a
display pose. No inferred geometry, labels, future frames or mask silhouettes
are used as motion measurements. GoTrack remains the pose acceptance source.
"""
import numpy as np
from .quality_contract import Frame, checked_pose, project
from .quality_time import LEGACY, PHYSICAL, FrameKey, TimePolicy, LEGACY_POLICY
from .vision import cv2


class ModelPointMemory:
    clock_modes_supported = frozenset((LEGACY, PHYSICAL))

    def __init__(self, depth_renderer, max_age=30, *, time_policy=LEGACY_POLICY):
        if not isinstance(time_policy, TimePolicy):
            raise ValueError('time_policy must be a TimePolicy')
        if time_policy.mode == PHYSICAL and max_age != 30:
            raise ValueError('Physical model memory uses its seconds TTL; max_age must stay at its legacy default')
        self.depth_renderer = depth_renderer
        self.time_policy = time_policy
        self._max_age = max_age
        self.gray = None
        self.points = None
        self.xyz = None
        self.anchor_id = None
        self.anchor_time_s = None
        self.last_id = None
        self.last_time_s = None
        self.motion_pose = None
        self.evidence = None
        self._cleared_key = None
        self._observed_key = None

    @property
    def max_age(self):
        return self._max_age

    @max_age.setter
    def max_age(self, value):
        if self.time_policy.mode == PHYSICAL:
            raise ValueError('Physical model memory does not accept frame-count max_age changes')
        self._max_age = value

    @staticmethod
    def image(frame):
        h, w = frame.rgb.shape[:2]
        size = (max(1, w//2), max(1, h//2))
        return cv2.resize(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def camera(frame, gray):
        k = np.array(frame.intrinsics, copy=True)
        k[0] *= gray.shape[1]/frame.rgb.shape[1]
        k[1] *= gray.shape[0]/frame.rgb.shape[0]
        return k

    def _frame_key(self, frame):
        if not isinstance(frame, Frame):
            raise ValueError('Motion memory requires a Frame')
        key = frame.key
        if key.clock_mode != self.time_policy.mode:
            raise ValueError('Model memory clock mode mismatch')
        return key

    def _validate_next_key(self, key, *, allow_cleared_same=False):
        if not isinstance(key, FrameKey) or key.clock_mode != self.time_policy.mode:
            raise ValueError('Model memory clock mode mismatch')
        if self.last_id is None:
            if self.time_policy.mode == PHYSICAL and key.frame_id != 0:
                raise ValueError('Physical model memory chronology must start at ordinal zero')
            return False
        if allow_cleared_same and self._cleared_key == key:
            return True
        if key.frame_id != self.last_id + 1:
            raise ValueError('Model memory requires consecutive images')
        if self.time_policy.mode == PHYSICAL and key.timestamp_s <= self.last_time_s:
            raise ValueError('Model memory timestamps must be strictly increasing')
        return False

    def _expire_anchor(self):
        self.gray = None
        self.points = None
        self.xyz = None
        self.anchor_id = None
        self.anchor_time_s = None
        self.motion_pose = None
        self.evidence = None

    def clear_motion(self, key: FrameKey, *, reason: str):
        """Drop all image/model evidence at a known chronology boundary."""
        if not isinstance(reason, str) or not reason:
            raise ValueError('Motion cleanup requires a reason')
        same_key = self._validate_next_key(key)
        # All key and chronology validation precedes state mutation or rendering.
        self._expire_anchor()
        self.last_id = key.frame_id
        self.last_time_s = key.timestamp_s
        self._cleared_key = key if key.clock_mode == PHYSICAL else None
        self._observed_key = None

    def observe(self, frame, mask):
        key = self._frame_key(frame)
        same_cleared_key = self._validate_next_key(key, allow_cleared_same=True)
        current = self.image(frame)
        if self.time_policy.mode == PHYSICAL:
            self._observed_key = key
        self.evidence = None
        if self.time_policy.mode == PHYSICAL and self.anchor_time_s is not None:
            if key.timestamp_s - self.anchor_time_s > self.time_policy.private_pose_memory_s:
                self._expire_anchor()
        if not same_cleared_key:
            self.last_id = key.frame_id
            self.last_time_s = key.timestamp_s
        self._cleared_key = None
        expired_by_age = (self.time_policy.mode == LEGACY and self.anchor_id is not None and
                          frame.frame_id-self.anchor_id > self.max_age)
        if (self.points is None or len(self.points) < 24 or self.gray is None or self.anchor_id is None or
                expired_by_age or mask is None or not mask.any() or current.std() < 1):
            self.points = None
            self.gray = current
            return
        before = self.points.reshape(-1, 1, 2).astype(np.float32)
        after, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, current, before, None, winSize=(21, 21), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01))
        if after is None:
            self.points = None; self.gray = current; return
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, self.gray, after, None, winSize=(21, 21), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01))
        if backward is None:
            self.points = None; self.gray = current; return
        uv = after[:, 0]
        h, w = current.shape
        keep = (status[:, 0] > 0) & (back_status[:, 0] > 0) & np.isfinite(uv).all(1)
        keep &= np.linalg.norm(backward[:, 0]-before[:, 0], axis=1) < 1.
        keep &= (uv[:, 0] >= 1) & (uv[:, 0] < w-1) & (uv[:, 1] >= 1) & (uv[:, 1] < h-1)
        visible = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        visible = cv2.erode(visible, np.ones((3, 3), np.uint8))
        ids = np.flatnonzero(keep); xy = uv[ids].astype(int)
        keep[ids] &= visible[xy[:, 1], xy[:, 0]] > 0
        self.points = uv[keep]; self.xyz = self.xyz[keep]; self.gray = current
        if len(self.points) < 24:
            return
        k = self.camera(frame, current)
        cv2.setRNGSeed((frame.frame_id*43) & 0x7fffffff)
        pose = self.motion_pose
        ok, r, t, inliers = cv2.solvePnPRansac(self.xyz.astype(np.float64), self.points.astype(np.float64), k, None,
            rvec=cv2.Rodrigues(pose[:3, :3])[0], tvec=pose[:3, 3].copy(), useExtrinsicGuess=True,
            iterationsCount=1000, reprojectionError=3*h/720, confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 20 or len(inliers)/len(self.points) < .6:
            return
        ids = inliers[:, 0]
        r, t = cv2.solvePnPRefineLM(self.xyz[ids], self.points[ids], k, None, r, t)
        candidate = np.eye(4); candidate[:3, :3] = cv2.Rodrigues(r)[0]; candidate[:3, 3] = t.ravel()
        if candidate[2, 3] <= .01:
            return
        yy, xx = np.nonzero(visible)
        box_area = max(1, (xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))
        support = cv2.contourArea(cv2.convexHull(self.points[ids]))/box_area
        if support < .12:
            return
        self.motion_pose = checked_pose(candidate)
        self.evidence = dict(xyz=self.xyz[ids].copy(), pixels=self.points[ids].copy(),
                             intrinsics=k, scale_720=720/h, support=float(support), frame_id=frame.frame_id,
                             timestamp_s=frame.timestamp_s, clock_mode=frame.clock_mode, key=key)

    def commit(self, frame, mask, pose):
        """Refresh reference points by ray intersection with the fixed mesh."""
        key = self._frame_key(frame)
        if self.time_policy.mode == PHYSICAL:
            if (self.last_id != key.frame_id or self.last_time_s != key.timestamp_s or
                    self._observed_key != key):
                raise ValueError('Physical model-memory commit must match the current observed frame')
        gray = self.image(frame); k = self.camera(frame, gray)
        depth = self.depth_renderer(pose, k, gray.shape[1], gray.shape[0])
        if depth.shape != gray.shape:
            raise ValueError('Model depth and motion image must have identical dimensions')
        visible = cv2.resize((mask > 0).astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        visible &= (np.isfinite(depth) & (depth > .01)).astype(np.uint8)
        visible = cv2.erode(visible, np.ones((3, 3), np.uint8))
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=600, qualityLevel=.01, minDistance=6, mask=visible, blockSize=5)
        self.gray = gray; self.anchor_id = frame.frame_id; self.anchor_time_s = frame.timestamp_s
        self.last_id = frame.frame_id; self.last_time_s = frame.timestamp_s
        self.motion_pose = checked_pose(pose); self.evidence = None
        self._cleared_key = None
        if corners is None or len(corners) < 24:
            self.points = None; self.xyz = None; return
        self.points = corners[:, 0]
        # Rendered depth samples at pixel centers; 3D points remain on this
        # canonical model, never on an accumulated point cloud.
        ix = np.floor(self.points).astype(int)
        pixels = ix.astype(float)+.5
        camera_xyz = np.c_[pixels, np.ones(len(pixels))]@np.linalg.inv(k).T
        camera_xyz *= depth[ix[:, 1], ix[:, 0], None]
        self.xyz = (camera_xyz-pose[:3, 3])@pose[:3, :3]
        self.points = pixels.astype(np.float32)

    def _matches_evidence(self, frame_or_legacy_id):
        if self.time_policy.mode == PHYSICAL:
            if not isinstance(frame_or_legacy_id, Frame):
                raise ValueError('Physical motion lookup requires the current Frame')
            key = self._frame_key(frame_or_legacy_id)
            return self.evidence is not None and self.evidence['key'] == key
        if isinstance(frame_or_legacy_id, Frame):
            raise ValueError('Legacy motion lookup requires an integer frame ID')
        return self.evidence is not None and self.evidence['frame_id'] == frame_or_legacy_id

    def seed(self, frame_or_legacy_id):
        if not self._matches_evidence(frame_or_legacy_id):
            return None
        return self.motion_pose.copy()

    def validate(self, candidate, frame_or_legacy_id):
        if not self._matches_evidence(frame_or_legacy_id):
            return True, None, {'state': 'unavailable'}
        e = self.evidence
        pixels, z = project(e['xyz'], candidate.pose, e['intrinsics'])
        errors = np.linalg.norm(pixels-e['pixels'], axis=1)*e['scale_720']
        stats = dict(state='available', points=len(errors), spatial_support=e['support'],
                     median_error_720=float(np.median(errors)), p95_error_720=float(np.percentile(errors, 95)))
        valid = bool(np.all(z > .01) and stats['median_error_720'] <= 5. and stats['p95_error_720'] <= 10.)
        return valid, None if valid else 'contradicts_observed_model_motion', stats
