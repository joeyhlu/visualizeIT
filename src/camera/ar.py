import os
import cv2
import numpy as np
from .Tracker import CombinedTracker
from .GenerateImage import ImageGenerator
from .ProjectImage import ProjectImage
from .ObjectSegmentation import ObjectSegmentation
from .data import landmark_names


def smooth_pose_transform(prev_transform, curr_transform, alpha=0.9):
    if prev_transform is None or curr_transform is None:
        return curr_transform
    return alpha * prev_transform + (1 - alpha) * curr_transform


def create_arm_mask(frame, result, side="right"):
    """
    Create a binary mask for the arm region based on pose landmarks.
    """
    if not result.pose_landmarks:
        print("[WARN] No pose landmarks detected.")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # Select arm landmark indices
    if side == "right":
        indices = [12, 14, 16]  # Right shoulder, elbow, wrist
    elif side == "left":
        indices = [11, 13, 15]  # Left shoulder, elbow, wrist
    else:
        raise ValueError("Invalid side specified. Use 'right' or 'left'.")

    # Get the landmark positions
    h, w = frame.shape[:2]
    arm_points = []
    for idx in indices:
        lm = result.pose_landmarks.landmark[idx]
        if lm.visibility > 0.5:  # Filter by visibility threshold
            arm_points.append((int(lm.x * w), int(lm.y * h)))

    if len(arm_points) < 3:
        print("[WARN] Insufficient landmarks for arm region.")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    arm_points = np.array(arm_points, dtype=np.int32)
    cv2.fillPoly(mask, [arm_points], 255)
    return mask


def run(cls):
    calibration_matrix = np.array([
        [815.1466689653845, 0, 638.4755231076894],
        [0, 814.8708730117189, 361.6966488002967],
        [0, 0, 1]
    ])

    # Initialize our classes
    tracker = CombinedTracker(
        calibration_matrix=calibration_matrix,
        capture_index=0,
        cls=cls,
        confidence_threshold=0.5
    )
    segmenter = ObjectSegmentation(plot=False)
    image_generator = ImageGenerator(device="cuda")
    project_image = ProjectImage(calibration_matrix)

    # Get design input from user
    print("\nPlease describe your design:")
    design_prompt = input("> ")
    
    # Generate the design using ImageGenerator class
    design_bgra = image_generator.get_design_bgra(
        answer_method="text",
        answer=design_prompt,
        is_tattoo=True
    )
    
    if design_bgra is None:
        print("[ERROR] No design was loaded or generated. Exiting.")
        return

    cap = cv2.VideoCapture(tracker.capture_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] Cannot open video/camera index:", tracker.capture_index)
        return

    print("[INFO] Starting real-time overlay...")
    prev_keypoints = None
    prev_descriptors = None
    pose_transform = None
    prev_pose_transform = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get masks using tracker
        person_mask, nonperson_mask = tracker.get_masks_fullframe(frame)
        object_mask = person_mask if cls == 'person' else nonperson_mask

        if np.sum(object_mask) == 0:
            print("[WARN] Object mask is empty. Proceeding to next frame.")

        if cls == 'person':
            # Use ObjectSegmentation for person tracking
            processed_frame, result = segmenter.body_segmentation(frame)

            if result.pose_world_landmarks:
                current_pose_3d = np.array([
                    (lm.x, lm.y, lm.z) for lm in result.pose_world_landmarks.landmark
                ], dtype=np.float32)

                if segmenter.initial_pose_3d is None:
                    segmenter.initial_pose_3d = current_pose_3d.copy()
                    print("[INFO] Stored initial 3D pose reference.")
                    pose_transform = None
                else:
                    R, t = segmenter.find_rigid_transform(segmenter.initial_pose_3d, current_pose_3d)
                    pose_transform = np.eye(4, dtype=np.float32)
                    pose_transform[:3, :3] = R
                    pose_transform[:3, 3] = t

                arm_mask = create_arm_mask(frame, result, side="right")
                if np.sum(arm_mask) > 0:
                    object_mask = arm_mask
            else:
                print("[WARN] No 3D pose landmarks detected.")
                pose_transform = None
                object_mask = None

        # Project and blend the design using ProjectImage class
        if object_mask is not None and np.sum(object_mask) > 0:
            overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
            cv2.imshow("Mask Overlay", overlay)

            # Use ProjectImage class for warping and blending
            augmented_frame = project_image.warp_and_blend(frame, design_bgra, object_mask)
            cv2.imshow("Augmented Frame", augmented_frame)
        else:            
            cv2.imshow("Augmented Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run('bottle')
