import cv2
import numpy as np
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

class ImageGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.pipe = None

    def _load_model(self):
        print(f"[INFO] Loading model: {self.model_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

    def generate_ai_design(
        self,
        prompt: str,
        negative_prompt: str = "blurry, anything unrelated to the prompt, people, texture",
        width=512,
        height=512,
        guidance_scale=30,
        steps=20,
    ):
        if self.pipe is None:
            self._load_model()

        print("[INFO] Prompt:", prompt)
        print("[INFO] Negative Prompt:", negative_prompt)
        print("[INFO] Generating design...")

        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps
            )

        pil_img = result.images[0].convert("RGBA")
        return pil_img

    @staticmethod
    def pillow_to_bgra(pil_img: Image.Image):
        arr = np.array(pil_img, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

    @staticmethod
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

        return ImageGenerator.pillow_to_bgra(pil_img)

    @staticmethod
    def remove_background_white_from_bgra(bgra: np.ndarray, threshold=240) -> np.ndarray:
        if bgra.shape[2] != 4:
            print("[WARN] remove_background_white_from_bgra: image missing alpha channel?")
            return bgra

        white_mask = np.all(bgra[..., :3] >= threshold, axis=2)
        bgra[white_mask, 3] = 0
        return bgra

    @staticmethod
    def lower_tattoo_opacity(bgra: np.ndarray, alpha_value=128) -> np.ndarray:
        if bgra.shape[2] != 4:
            print("[WARN] no alpha channel, can't lower opacity.")
            return bgra

        out = bgra.copy()
        mask_non_transparent = out[..., 3] > 0
        out[mask_non_transparent, 3] = alpha_value
        return out

    def get_design_bgra(self, answer_method: str, answer: str, is_tattoo=False) -> np.ndarray:
        if answer_method == "upload":
            url_or_path = answer.strip()
            design_bgra = self.download_image_bgra(url_or_path)
            if design_bgra is None:
                return None
        else:
            prompt = answer.strip()
            neg_prompt = "blurry, anything unrelated to the prompt, people, texture"
            if is_tattoo:
                prompt += ", black and white ink style"
            pil_img = self.generate_ai_design(
                prompt=prompt,
                negative_prompt=neg_prompt
            )
            design_bgra = self.pillow_to_bgra(pil_img)

        if design_bgra is None:
            return None

        if is_tattoo:
            choice_bg = "y"
            if choice_bg == 'y':
                design_bgra = self.remove_background_white_from_bgra(design_bgra, threshold=240)
            design_bgra = self.lower_tattoo_opacity(design_bgra, alpha_value=128)

        return design_bgra
