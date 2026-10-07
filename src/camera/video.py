import os
import cv2
import numpy as np
import streamlit as st
from Tracker import CombinedTracker
from GenerateAndMap import get_design_bgra, warp_and_blend, compute_pose_transform
from ObjectSegmentation import ObjectSegmentation

from data import COCO_CLASSES
# Streamlit App Title
st.title("Visualize It")

# Streamlit Sidebar Inputs
st.sidebar.header("Configuration")
calibration_matrix = np.array([
    [st.sidebar.number_input("Calibration Matrix [0,0]", 815.1467), 0, st.sidebar.number_input("Calibration Matrix [0,2]", 638.4755)],
    [0, st.sidebar.number_input("Calibration Matrix [1,1]", 814.8709), st.sidebar.number_input("Calibration Matrix [1,2]", 361.6966)],
    [0, 0, 1]
])

capture_index = st.sidebar.number_input("Camera Index", value=0, min_value=0, step=1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
object_class = st.sidebar.selectbox("Object Class", COCO_CLASSES, index=0)
device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)


# Answering 
ans_method = "upload"
upload_mode = st.sidebar.checkbox("Upload an Image")
prompt = None

uploaded_image = None
if upload_mode:
    uploaded_image = st.file_uploader("Choose an Image", type=["jpg"])
    if uploaded_image is not None:  # Check if a file has been uploaded
        # Convert the uploaded file to a numpy array
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Save the image to disk
        save_path = "logo.jpg"
        cv2.imwrite(save_path, image)

        # Provide feedback to the user
        st.success(f"Image saved as {save_path}")
        prompt = str(save_path)
    else:
        st.warning("Please upload an image file.")

# Ask whether to generate a prompt
generate_prompt = st.sidebar.checkbox("Generate a Prompt?", value=False)
if generate_prompt:
    ans_method = 'generate'
    prompt = st.sidebar.text_input("Enter your prompt", value="Default Prompt")

is_tattoo = st.sidebar.checkbox("Use No Background Design", value=True)

# Helper Function for Pose Smoothing
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

    if len(arm_points) < 3:  # Not enough points to form a valid region
        print("[WARN] Insufficient landmarks for arm region.")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # Create a mask for the arm region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    arm_points = np.array(arm_points, dtype=np.int32)
    cv2.fillPoly(mask, [arm_points], 255)
    return mask

# Run Function
def run():
    tracker = CombinedTracker(
        calibration_matrix=calibration_matrix,
        capture_index=capture_index,
        cls=object_class,
        confidence_threshold=confidence_threshold
    )

    segmenter = ObjectSegmentation(plot=False)

    design_bgra = get_design_bgra(device=device, is_tattoo=is_tattoo, answer_method=ans_method, answer=prompt)
    if design_bgra is None:
        st.error("No design was loaded or generated. Exiting.")
        return

    cap = cv2.VideoCapture(tracker.capture_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        st.error(f"Cannot open video/camera index: {tracker.capture_index}")
        return

    st.info("Starting real-time overlay...")
    prev_keypoints = None
    prev_descriptors = None
    pose_transform = None
    prev_pose_transform = None

    # Streamlit Display Area
    frame_placeholder = st.empty()
    #augmented_frame_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get masks
        person_mask, nonperson_mask = tracker.get_masks_fullframe(frame)
        object_mask = person_mask if object_class == 'person' else nonperson_mask

        if object_class != 'person':
            orb = cv2.ORB_create()
            curr_keypoints, curr_descriptors = orb.detectAndCompute(frame, object_mask)
            if prev_keypoints is not None and prev_descriptors is not None:
                pose_transform = compute_pose_transform(prev_keypoints, curr_keypoints, prev_descriptors, curr_descriptors)
                pose_transform = smooth_pose_transform(prev_pose_transform, pose_transform)
                prev_pose_transform = pose_transform
            prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors
        else:
            # Step 2: Use ObjectSegmentation to calculate pose_transform and mask
            processed_frame, result = segmenter.body_segmentation(frame)  # Unpack correctly

            if result.pose_world_landmarks:
                # Extract current pose 3D landmarks
                current_pose_3d = np.array([
                    (lm.x, lm.y, lm.z) for lm in result.pose_world_landmarks.landmark
                ], dtype=np.float32)

                if segmenter.initial_pose_3d is None:
                    # Store the initial pose reference for alignment
                    segmenter.initial_pose_3d = current_pose_3d.copy()
                    print("[INFO] Stored initial 3D pose reference.")
                    pose_transform = None
                else:
                    # Compute rigid transformation (R, t) from reference to current pose
                    R, t = segmenter.find_rigid_transform(segmenter.initial_pose_3d, current_pose_3d)

                    # Build the 4x4 transformation matrix
                    pose_transform = np.eye(4, dtype=np.float32)
                    pose_transform[:3, :3] = R
                    pose_transform[:3, 3] = t
                    print("[DEBUG] Pose Transform Matrix:\n", pose_transform)

                # Create a mask for the right arm (or left)
                arm_mask = create_arm_mask(frame, result, side="right")  # Specify "right" or "left"
                if np.sum(arm_mask) == 0:
                    print("[WARN] Arm mask is empty. Proceeding with next frame.")
                    #continue
                else:
                    object_mask = arm_mask  # Update the mask to use the arm mask
            else:
                print("[WARN] No 3D pose landmarks detected. Skipping pose transformation.")
                pose_transform = None
                object_mask = None


        # Step 3: Warp and blend the design
        if object_mask is not None and np.sum(object_mask) > 0:
            overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
            #cv2.imshow("asd", overlay)

            if pose_transform is None:
                print("[INFO] Using static placement due to missing pose transform.")
                augmented_frame = warp_and_blend(frame, design_bgra, object_mask)
            else:
                augmented_frame = warp_and_blend(frame, design_bgra, object_mask, pose_transform)
                print("asdkas?")
            #cv2.imshow("Augmented Frame", augmented_frame)
            frame_placeholder.image(augmented_frame)

        else:            
            #cv2.imshow("Augmented Frame", frame)
            frame_placeholder.image(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the App
if st.sidebar.button("Start"):
    run()
