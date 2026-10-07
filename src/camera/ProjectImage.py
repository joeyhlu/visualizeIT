import cv2
import numpy as np
import os
from .VisualOdometry import VisualOdometry

class ProjectImage:
    def __init__(self, calibration_matrix):
        self.vo = VisualOdometry(calibration_matrix)
        
    def create_object_mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        return 255 - cv2.inRange(hsv, lower_green, upper_green)

    def erode_mask(self, mask, erode_px=10):
        kernel = np.ones((erode_px, erode_px), np.uint8)
        return cv2.erode(mask, kernel, iterations=1)

    def order_corners(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_and_blend(self, frame_bgr, design_bgra, mask_region):
        out = frame_bgr.copy()
        if np.sum(mask_region) == 0:
            return out

        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return out

        largest_ct = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_ct, True)
        approx = cv2.approxPolyDP(largest_ct, 0.02 * peri, True)

        # Determine if the contour is more circular or rectangular
        circularity = 4 * np.pi * cv2.contourArea(largest_ct) / (peri * peri)
        is_circular = circularity > 0.8  # Threshold for circularity

        if is_circular:
            # Handle circular projection
            (x, y), radius = cv2.minEnclosingCircle(largest_ct)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Create a circular mask
            circle_mask = np.zeros_like(mask_region)
            cv2.circle(circle_mask, center, radius, 255, -1)
            
            # Resize design to fit circle
            size = int(radius * 2)
            design_resized = cv2.resize(design_bgra, (size, size))
            
            # Create output mask
            mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1], 4), dtype=np.uint8)
            x1, y1 = int(x - radius), int(y - radius)
            x2, y2 = x1 + size, y1 + size
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2)
            y2 = min(frame_bgr.shape[0], y2)
            
            # Calculate the region to copy from the resized design
            dx1 = max(0, -int(x - radius))
            dy1 = max(0, -int(y - radius))
            dx2 = dx1 + (x2 - x1)
            dy2 = dy1 + (y2 - y1)
            
            # Copy the design to the mask
            mask[y1:y2, x1:x2] = design_resized[dy1:dy2, dx1:dx2]
            
            # Apply the circular mask
            mask[circle_mask == 0] = 0
            
        else:
            # Handle rectangular/polygon projection
            if len(approx) != 4:
                # If not exactly 4 corners, use bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_ct)
                approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            else:
                approx = approx.reshape(-1, 2).astype(np.float32)

            # Order corners consistently
            approx = self.order_corners(approx)

            # Get design dimensions
            h_des, w_des = design_bgra.shape[:2]
            src_corners = np.array([[0, 0], [w_des, 0], [w_des, h_des], [0, h_des]], dtype=np.float32)

            # Apply perspective transform if we have pose information
            kp = self.vo.process_frame(frame_bgr, mask_region)
            if kp is not None and len(self.vo.poses) > 1:
                pose_transform = self.vo.poses[-1]
                try:
                    approx = cv2.perspectiveTransform(approx[np.newaxis, :, :], pose_transform[:3]).squeeze()
                except cv2.error:
                    pass

            # Compute and apply perspective transform
            M = cv2.getPerspectiveTransform(src_corners, approx)
            mask = cv2.warpPerspective(design_bgra, M, (frame_bgr.shape[1], frame_bgr.shape[0]))

        # Blend the warped design with the original frame
        alpha = (mask[..., 3] / 255.0) * (mask_region / 255.0)
        for c in range(3):
            out[..., c] = mask[..., c] * alpha + out[..., c] * (1 - alpha)

        return out

    def process_video(self, input_video, output_video, design_bgra, erode_px=10):
        if not os.path.exists(input_video):
            return

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            mask = self.create_object_mask(frame_bgr)
            mask_eroded = self.erode_mask(mask, erode_px)
            final_frame = self.warp_and_blend(frame_bgr, design_bgra, mask_eroded)
            writer.write(final_frame)

        cap.release()
        writer.release()