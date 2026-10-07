import cv2
import numpy as np
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os


def generate_ai_design(
    prompt: str,
    negative_prompt: str = "blurry, anything unrelated to the prompt, people, texture",
    model_name="runwayml/stable-diffusion-v1-5",
    width=512,
    height=512,
    device="cuda",
    guidance_scale=30,
    steps=20,
):
    print(f"[INFO] Loading model: {model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    print("[INFO] Prompt:", prompt)
    print("[INFO] Negative Prompt:", negative_prompt)
    print("[INFO] Generating design...")

    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )

    pil_img = result.images[0].convert("RGBA")
    return pil_img

def pillow_to_bgra(pil_img: Image.Image):
    arr = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def download_image_bgra(url_or_path: str) -> np.ndarray:
    try:
        if url_or_path.strip().lower().startswith("http"):
            resp = requests.get(url_or_path, timeout=10)
            resp.raise_for_status()
            pil_img = Image.open(BytesIO(resp.content)).convert("RGBA")
        else:
            if not os.path.exists(url_or_path):
                raise FileNotFoundError(f"File not found: {url_or_path}")
            pil_img = Image.open(url_or_path).convert("RGBA")
    except Exception as e:
        print(f"[ERROR] Could not load image from '{url_or_path}': {e}")
        return None

    return pillow_to_bgra(pil_img)

def remove_background_white_from_bgra(bgra: np.ndarray, threshold=240) -> np.ndarray:

    if bgra.shape[2] != 4:
        print("[WARN] remove_background_white_from_bgra: image missing alpha channel?")
        return bgra

    white_mask = np.all(bgra[..., :3] >= threshold, axis=2)
    bgra[white_mask, 3] = 0
    return bgra

def lower_tattoo_opacity(bgra: np.ndarray, alpha_value=128) -> np.ndarray:
    if bgra.shape[2] != 4:
        print("[WARN] no alpha channel, can't lower opacity.")
        return bgra

    out = bgra.copy()
    mask_non_transparent = out[..., 3] > 0
    out[mask_non_transparent, 3] = alpha_value
    return out

def get_design_bgra(device: str, answer_method: str, answer: str, is_tattoo=False) -> np.ndarray:
    if answer_method == "upload":
        url_or_path = answer.strip()
        design_bgra = download_image_bgra(url_or_path)
        if design_bgra is None:
            return None
    else:
        prompt = answer.strip()
        neg_prompt = "blurry, anything unrelated to the prompt, people, texture"
        if is_tattoo:
            prompt += ", black and white ink style"
        pil_img = generate_ai_design(
            prompt=prompt,
            negative_prompt=neg_prompt,
            device=device
        )
        design_bgra = pillow_to_bgra(pil_img)

    if design_bgra is None:
        return None

    if is_tattoo:
        choice_bg = "y"
        if choice_bg == 'y':
            design_bgra = remove_background_white_from_bgra(design_bgra, threshold=240)

        design_bgra = lower_tattoo_opacity(design_bgra, alpha_value=128)

    return design_bgra

def create_object_mask(frame_bgr: np.ndarray):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    return 255 - green_mask  

def erode_mask(mask: np.ndarray, erode_px=10):
    kernel = np.ones((erode_px, erode_px), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)

def order_corners(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   
    rect[2] = pts[np.argmax(s)]     
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)]  
    return rect

def warp_and_blend(frame_bgr, design_bgra, mask_region, pose_transform=None):
    """
    Warps and blends the design only within the mask region.

    Args:
        frame_bgr (np.ndarray): The original frame in BGR format.
        design_bgra (np.ndarray): The design image in BGRA format.
        mask_region (np.ndarray): Binary mask defining the region to overlay the design.
        pose_transform (np.ndarray): Homography matrix to apply perspective transformation (optional).

    Returns:
        np.ndarray: The frame with the design blended within the mask region.
    """
    out = frame_bgr.copy()

    # Validate mask
    if np.sum(mask_region) == 0:
        print("[WARN] Mask is empty. Skipping frame.")
        return out

    # Find contours
    contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[WARN] No contours found. Skipping frame.")
        return out

    largest_ct = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_ct, True)
    approx = cv2.approxPolyDP(largest_ct, 0.02 * peri, True)

    # Ensure exactly 4 points for perspective transform
    if len(approx) != 4:
        print(f"[WARN] Invalid corner points: {len(approx)}. Approximating to rectangle.")
        x, y, w, h = cv2.boundingRect(largest_ct)
        approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    else:
        approx = approx.reshape(-1, 2).astype(np.float32)

    print("[DEBUG] Contour Points (approx):", approx)

    # Design dimensions
    h_des, w_des = design_bgra.shape[:2]
    src_corners = np.array([[0, 0], [w_des, 0], [w_des, h_des], [0, h_des]], dtype=np.float32)

    # Apply pose transform if provided
    if pose_transform is not None:
        try:
            approx = cv2.perspectiveTransform(approx[np.newaxis, :, :], pose_transform).squeeze()
        except cv2.error as e:
            print("[ERROR] Perspective transform failed:", e)
            return out

    # Compute homography for warping
    M = cv2.getPerspectiveTransform(src_corners, approx)
    warped = cv2.warpPerspective(design_bgra, M, (frame_bgr.shape[1], frame_bgr.shape[0]))

    # Blend the warped design only within the mask
    mask_f = mask_region.astype(float) / 255.0  # Normalize mask to [0, 1]
    alpha = (warped[..., 3] / 255.0) * mask_f  # Combine mask and design alpha
    design_rgb = warped[..., :3]

    for c in range(3):  # Blend each channel
        out[..., c] = design_rgb[..., c] * alpha + out[..., c] * (1 - alpha)

    return out


def fill_mask_with_two_designs(frame_bgr, design_bgra_cap, design_bgra_body, mask_eroded, cap_ratio=0.2):
    out = frame_bgr.copy()
    cts, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cts:
        return out
    largest = max(cts, key=cv2.contourArea)
    x,y,w,h= cv2.boundingRect(largest)
    if w<=0 or h<=0:
        return out

    cap_height= int(h*cap_ratio)
    if cap_height<1:
        cap_height=1
    cap_y=y
    body_y=y+cap_height
    body_height= h-cap_height

    sub_mask_cap= np.zeros_like(mask_eroded)
    sub_mask_body= np.zeros_like(mask_eroded)

    sub_mask_cap[cap_y:cap_y+cap_height, x:x+w] = mask_eroded[cap_y:cap_y+cap_height, x:x+w]
    sub_mask_body[body_y:body_y+body_height, x:x+w] = mask_eroded[body_y:body_y+body_height, x:x+w]

    out= warp_and_blend(out, design_bgra_cap,  sub_mask_cap)
    out= warp_and_blend(out, design_bgra_body, sub_mask_body)
    return out

def process_video_single_design(input_video: str, output_video: str, design_bgra: np.ndarray, erode_px=10):
    if not os.path.exists(input_video):
        print(f"[ERROR] Input video '{input_video}' doesn't exist.")
        return
    cap= cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] cannot open '{input_video}'")
        return

    fps= cap.get(cv2.CAP_PROP_FPS)
    frame_w= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Single design: {frame_w}x{frame_h} @ {fps} fps, total frames={total_frames}")

    out_dir= os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc= cv2.VideoWriter_fourcc(*'mp4v')
    writer= cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

    frame_count= 0
    while True:
        ret, frame_bgr= cap.read()
        if not ret:
            break

        mask= create_object_mask(frame_bgr)
        mask_eroded= erode_mask(mask, erode_px=erode_px)
        final_frame= warp_and_blend(frame_bgr, design_bgra, mask_eroded)

        writer.write(final_frame)
        frame_count+=1
        if frame_count%50==0:
            print(f"[INFO] Single design processed {frame_count}/{total_frames}...")

    cap.release()
    writer.release()
    print(f"[INFO] Output saved => {output_video}")

def process_video_bottle(input_video: str, output_video: str, design_bgra_cap: np.ndarray,
                         design_bgra_body: np.ndarray, erode_px=10, cap_ratio=0.2):
    if not os.path.exists(input_video):
        print(f"[ERROR] '{input_video}' doesn't exist.")
        return
    cap= cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] cannot open '{input_video}'")
        return

    fps= cap.get(cv2.CAP_PROP_FPS)
    frame_w= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Bottle: {frame_w}x{frame_h} @ {fps} fps, total frames={total_frames}")

    out_dir= os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc= cv2.VideoWriter_fourcc(*'mp4v')
    writer= cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

    frame_count=0
    while True:
        ret, frame_bgr= cap.read()
        if not ret:
            break

        mask= create_object_mask(frame_bgr)
        mask_eroded= erode_mask(mask, erode_px=erode_px)
        final_frame= fill_mask_with_two_designs(
            frame_bgr, design_bgra_cap, design_bgra_body, mask_eroded, cap_ratio=cap_ratio
        )

        writer.write(final_frame)
        frame_count+=1
        if frame_count%50==0:
            print(f"[INFO] Bottle processed {frame_count}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"[INFO] Output saved => {output_video}")

def compute_pose_transform(prev_keypoints, curr_keypoints, prev_descriptors, curr_descriptors):
    # Check if descriptors are valid
    if prev_descriptors is None or curr_descriptors is None:
        print("[INFO] Descriptors are None. Skipping pose computation.")
        return None

    if prev_descriptors.dtype != np.uint8 or curr_descriptors.dtype != np.uint8:
        raise ValueError("Descriptors must be of type np.uint8 for ORB.")

    if prev_descriptors.shape[1] != curr_descriptors.shape[1]:
        raise ValueError("Descriptor dimensions do not match.")

    # Match features using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(prev_descriptors, curr_descriptors)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:  # Minimum number of points needed to compute homography
        print("[INFO] Not enough matches to compute pose transform.")
        return None

    # Extract matching points
    src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    mode= input("Select mode [tattoo tester / bottle / other]: ").strip().lower()

    if mode=="tattoo tester":
        print("[TATTOO TESTER MODE] => lower alpha to appear more natural.")
        design_bgra_tattoo= get_design_bgra(device=device, is_tattoo=True)
        if design_bgra_tattoo is None:
            print("[ERROR] No tattoo design loaded.")
            return

        input_video= "data/my_video.mp4"
        output_video= "data/my_video_tattoo_output.mp4"

        process_video_single_design(
            input_video, output_video, design_bgra_tattoo, erode_px=10
        )

    elif mode=="bottle":
        print("[BOTTLE MODE]")
        design_bgra_cap= get_design_bgra(device=device, is_tattoo=False)
        if design_bgra_cap is None:
            return
        design_bgra_body= get_design_bgra(device=device, is_tattoo=False)
        if design_bgra_body is None:
            return

        input_video= "data/my_video.mp4"
        output_video= "data/my_video_bottle_output.mp4"

        process_video_bottle(
            input_video, output_video,
            design_bgra_cap, design_bgra_body, erode_px=10, cap_ratio=0.2
        )

    else:
        print("[OTHER MODE] => single design alpha blend.")
        design_bgra= get_design_bgra(device=device, is_tattoo=False)
        if design_bgra is None:
            return

        input_video= "data/my_video.mp4"
        output_video= "data/my_video_other_output.mp4"

        process_video_single_design(
            input_video, output_video, design_bgra, erode_px=10
        )



