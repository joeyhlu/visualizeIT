import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans  # Imported but unused; safe to remove if needed
import matplotlib.pyplot as plt

class VisualOdometry:
    def __init__(self, calibration_matrix, label="VO"):
        self.label = label
        self.K = calibration_matrix
        self.poses = [np.eye(4)]
        self.orb = cv.ORB_create(nfeatures=3000)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        # Trajectory in x-z plane
        self.x_coords = [0]
        self.z_coords = [0]

        # Lower threshold if your motion is small
        self.min_translation_threshold = 0.001

    def detect_and_compute(self, gray_frame, mask=None):
        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, mask)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        if des1 is None or des2 is None or len(des2) < 2:
            return []
        matches = self.flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return good_matches

    def extract_matched_points(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def compute_pose(self, pts1, pts2):
        # Not enough points => skip
        if pts1 is None or pts2 is None or len(pts1) < 8:
            return None, None

        # Lower threshold to reject outliers more aggressively
        E, mask = cv.findEssentialMat(
            pts1, pts2, self.K,
            method=cv.RANSAC,
            prob=0.999,
            threshold=0.5  # lowered from 1.0
        )
        if E is None:
            return None, None

        retval, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Raise the inlier count requirement if your scene has more features
        if retval < 8:  # You can raise to 30 or 50 for more robust checking
            return None, None

        # Check if translation is large enough
        t_norm = np.linalg.norm(t)
        if t_norm < self.min_translation_threshold:
            return None, None

        return R, t

    def update_pose(self, R, t):
        if R is None or t is None:
            return
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()

        # We keep the same logic: appending the new pose as old_pose @ inv(T)
        # because recoverPose returns camera1->camera2, and we track camera->world
        self.poses.append(self.poses[-1] @ np.linalg.inv(T))

        self.x_coords.append(self.poses[-1][0, 3])
        self.z_coords.append(self.poses[-1][2, 3])

    def process_frame(self, frame, mask=None):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp, des = self.detect_and_compute(gray, mask)

        if self.prev_des is not None and des is not None:
            matches = self.match_features(self.prev_des, des)
            if matches:
                pts1, pts2 = self.extract_matched_points(self.prev_kp, kp, matches)
                R, t = self.compute_pose(pts1, pts2)
                self.update_pose(R, t)

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        # Return keypoints for visualization if needed
        return kp
