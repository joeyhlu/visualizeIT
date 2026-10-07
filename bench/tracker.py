"""Initialized RGB rigid tracker. No evaluation annotations are accepted here."""
from dataclasses import dataclass
import numpy as np
from .vision import cv2


@dataclass
class TrackingResult:
    frame_id: int
    state: str
    reason: str | None
    rotation: np.ndarray | None
    translation: np.ndarray | None
    statistics: dict

    def manifest(self):
        matrix = None
        if self.rotation is not None:
            matrix = np.eye(4); matrix[:3, :3] = self.rotation; matrix[:3, 3] = self.translation
        return dict(schemaVersion=1, frameId=self.frame_id, state=self.state, failureReason=self.reason,
                    cvCameraFromSourceObject=matrix.tolist() if matrix is not None else None,
                    units='metres', objectBasis='Original right-handed BOP source model axes', **self.statistics)


class RigidTracker:
    def __init__(self, reference_points, descriptors, model_positions, previous_gray=None,
                 previous_pixels=None, previous_points=None):
        self.reference = np.asarray(reference_points, dtype=np.float64)
        self.descriptors = np.asarray(descriptors, dtype=np.uint8)
        self.model = np.asarray(model_positions, dtype=np.float64)
        if self.reference.shape != (len(self.descriptors), 3) or self.descriptors.ndim != 2 or self.descriptors.shape[1] != 32:
            raise ValueError('ORB reference descriptors and 3D points must agree.')
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous_gray = previous_gray
        self.previous_pixels = previous_pixels
        self.previous_points = previous_points
        self.lost = previous_pixels is None
        self.iteration = 0

    def match(self, gray):
        features, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2 or len(self.descriptors) < 2:
            return np.empty((0, 2)), np.empty((0, 3))
        forward = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        reverse = self.matcher.match(descriptors, self.descriptors)
        mutual = {match.queryIdx: match.trainIdx for match in reverse}
        pairs = [a for group in forward if len(group) == 2 for a, b in [group]
                 if a.distance < .75*b.distance and mutual.get(a.trainIdx) == a.queryIdx]
        return np.array([features[p.trainIdx].pt for p in pairs]).reshape(-1, 2), self.reference[[p.queryIdx for p in pairs]]

    def flow(self, gray):
        if self.previous_gray is None or self.previous_pixels is None or not len(self.previous_pixels):
            return np.empty((0, 2)), np.empty((0, 3))
        original = np.asarray(self.previous_pixels, dtype=np.float32).reshape(-1, 1, 2)
        current, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, original, None, winSize=(21, 21), maxLevel=3)
        if current is None: return np.empty((0, 2)), np.empty((0, 3))
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, current, None, winSize=(21, 21), maxLevel=3)
        good = (status.ravel() == 1)&(reverse_status.ravel() == 1)&(np.linalg.norm(backward.reshape(-1, 2)-original.reshape(-1, 2), axis=1) <= 1)
        points = current.reshape(-1, 2)
        good &= (points[:, 0] >= 0)&(points[:, 1] >= 0)&(points[:, 0] < gray.shape[1])&(points[:, 1] < gray.shape[0])
        return points[good], np.asarray(self.previous_points)[good]

    def solve(self, pixels, points, k, frame_id):
        stats = dict(correspondences=len(points), inliers=0, inlierRatio=0, medianReprojectionErrorPixels=None,
                     featureCoverage=0, trackingConfidence=None)
        def failure(reason, state='lost'):
            return TrackingResult(frame_id, state, reason, None, None, stats)
        if len(points) < 12: return failure('insufficient_correspondences')
        spread = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
        if spread[1] < 1e-6 or spread[2]/max(spread[0], 1e-15) < .001:
            return failure('planar_or_underconstrained_correspondences', 'limited')
        cv2.setRNGSeed(17)
        try:
            success, rvec, translation, inliers = cv2.solvePnPRansac(points, pixels.astype(np.float64), k, None,
                iterationsCount=150, reprojectionError=3, confidence=.99, flags=cv2.SOLVEPNP_EPNP)
            if not success or inliers is None: return failure('pnp_failed')
            inliers = inliers.ravel()
            stats['inliers'] = len(inliers); stats['inlierRatio'] = len(inliers)/len(points)
            if len(inliers) < 12 or stats['inlierRatio'] < .5: return failure('insufficient_inliers')
            rvec, translation = cv2.solvePnPRefineLM(points[inliers], pixels[inliers].astype(np.float64), k, None, rvec, translation)
            rotation = cv2.Rodrigues(rvec)[0]
            translation = translation.ravel()
            projected = cv2.projectPoints(points[inliers], rvec, translation, k, None)[0].reshape(-1, 2)
            error = np.linalg.norm(projected-pixels[inliers], axis=1)
            stats['medianReprojectionErrorPixels'] = float(np.median(error))
            model_camera = self.model@rotation.T+translation
            if not np.isfinite(model_camera).all() or (model_camera[:, 2] <= .03).any(): return failure('invalid_or_behind_camera_pose')
            model_projected = cv2.projectPoints(self.model, rvec, translation, k, None)[0].reshape(-1, 2)
            area = np.prod(np.maximum(np.ptp(model_projected, axis=0), 1))
            stats['featureCoverage'] = float(np.prod(np.ptp(pixels[inliers], axis=0))/area)
            if stats['medianReprojectionErrorPixels'] > 3: return failure('reprojection_error')
            if stats['featureCoverage'] < .2: return failure('insufficient_feature_coverage', 'limited')
            stats['trackingConfidence'] = 'thresholds_met_not_a_probability'
            return TrackingResult(frame_id, 'tracking', None, rotation, translation, stats), pixels[inliers], points[inliers]
        except cv2.error:
            return failure('opencv_pose_error')

    def update(self, rgb, k, frame_id):
        gray = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pixels, points = self.flow(gray) if not self.lost else (np.empty((0, 2)), np.empty((0, 3)))
        if self.iteration%5 == 0 or self.lost or len(points) < 12:
            matched, reference = self.match(gray)
            pixels = np.concatenate([pixels, matched]); points = np.concatenate([points, reference])
        # One observed pixel per model location; persistent flow takes precedence.
        if len(points):
            _, unique = np.unique(np.round(points/.0005).astype(np.int64), axis=0, return_index=True)
            unique.sort(); points, pixels = points[unique], pixels[unique]
        result = self.solve(pixels, points, k, frame_id)
        if isinstance(result, tuple):
            result, self.previous_pixels, self.previous_points = result
            self.lost = False
        else:
            self.previous_pixels = None; self.previous_points = None; self.lost = True
        self.previous_gray = gray
        self.iteration += 1
        return result
