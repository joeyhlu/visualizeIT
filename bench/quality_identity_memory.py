"""Experimental image identity checks against accepted, fixed-mesh keyframes.

Supplemental evidence, not a replacement pose tracker. Unlike consecutive
flow, descriptors can reconnect to an earlier view after point loss. All 3D
points come from the fixed model at a previously accepted pose. No reference
dataset poses, annotations or future images enter this module.
"""
import numpy as np
from .quality_contract import checked_pose, project
from .quality_memory import ModelPointMemory
from .vision import cv2


class SurfaceIdentityMemory:
    def __init__(self, depth_renderer, capacity=8):
        self.depth_renderer = depth_renderer
        self.capacity = capacity
        self.views = []
        self.last_id = None
        self.orb = cv2.ORB_create(nfeatures=1500, edgeThreshold=16, patchSize=31)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, frame, mask):
        gray = ModelPointMemory.image(frame)
        if mask is None or not mask.any() or gray.std() < 1: return gray, [], None
        visible = cv2.resize((mask > 0).astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Avoid descriptors dominated by a hand or background at the boundary.
        visible = cv2.erode(visible, np.ones((9, 9), np.uint8))
        points, descriptors = self.orb.detectAndCompute(gray, visible*255)
        return gray, points, descriptors

    def commit(self, frame, mask, pose):
        pose = checked_pose(pose)
        if self.views:
            previous = self.views[-1]['pose']
            angle = np.degrees(np.arccos(np.clip((np.trace(previous[:3, :3].T@pose[:3, :3])-1)/2, -1, 1)))
            if angle < 15 and np.linalg.norm(previous[:3, 3]-pose[:3, 3]) < .03: return False
        gray, points, descriptors = self.features(frame, mask)
        if descriptors is None or len(points) < 48: return False
        k = ModelPointMemory.camera(frame, gray)
        depth = self.depth_renderer(pose, k, gray.shape[1], gray.shape[0])
        pixels = np.array([p.pt for p in points]); ix = pixels.astype(int)
        z = depth[ix[:, 1], ix[:, 0]]
        valid = np.isfinite(z) & (z > .01)
        if valid.sum() < 48: return False
        # Pair sampled depth with its raster pixel center.
        pixels = ix[valid].astype(float)+.5
        camera_xyz = np.c_[pixels, np.ones(len(pixels))]@np.linalg.inv(k).T*z[valid, None]
        xyz = (camera_xyz-pose[:3, 3])@pose[:3, :3]
        self.views.append(dict(frame_id=frame.frame_id, pose=pose.copy(), xyz=xyz,
                               descriptors=descriptors[valid].copy()))
        self.views = self.views[-self.capacity:]
        return True

    def check(self, frame, mask, candidate):
        if self.last_id is not None and frame.frame_id != self.last_id+1:
            raise ValueError('Identity checks require consecutive input images')
        self.last_id = frame.frame_id
        gray, points, descriptors = self.features(frame, mask)
        unavailable = dict(state='unavailable', keyframes=len(self.views))
        if descriptors is None or len(points) < 24 or not self.views: return True, None, unavailable
        k = ModelPointMemory.camera(frame, gray); best = None
        for view in self.views:
            forward = self.matcher.knnMatch(view['descriptors'], descriptors, k=2)
            backward = self.matcher.match(descriptors, view['descriptors'])
            reverse = {m.queryIdx: m.trainIdx for m in backward}
            matches = [pair[0] for pair in forward if len(pair) == 2 and pair[0].distance < .7*pair[1].distance
                       and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
            if len(matches) < 24: continue
            xyz = np.array([view['xyz'][m.queryIdx] for m in matches], np.float64)
            uv = np.array([points[m.trainIdx].pt for m in matches], np.float64)
            cv2.setRNGSeed((frame.frame_id*67+view['frame_id']) & 0x7fffffff)
            ok, r, t, inliers = cv2.solvePnPRansac(xyz, uv, k, None, iterationsCount=2000,
                reprojectionError=3*gray.shape[0]/720, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 20 or len(inliers)/len(matches) < .6: continue
            ids = inliers[:, 0]
            visible = cv2.resize((mask > 0).astype(np.uint8), (gray.shape[1], gray.shape[0]))
            yy, xx = np.nonzero(visible)
            support = cv2.contourArea(cv2.convexHull(uv[ids].astype(np.float32)))/max(1, (xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))
            if support < .12: continue
            # Geometric consensus is independent of the pose being inspected.
            reprojection, z = project(xyz[ids], candidate.pose, k)
            errors = np.linalg.norm(reprojection-uv[ids], axis=1)*720/gray.shape[0]
            stats = dict(state='available', keyframe=view['frame_id'], points=len(ids), spatial_support=float(support),
                median_error_720=float(np.median(errors)), p95_error_720=float(np.percentile(errors, 95)))
            valid = bool(np.all(z > .01) and stats['median_error_720'] <= 5 and stats['p95_error_720'] <= 10)
            evidence = (len(ids), valid, stats)
            if best is None or evidence[0] > best[0]: best = evidence
        if best is None: return True, None, unavailable
        return best[1], None if best[1] else 'contradicts_surface_identity', best[2]
