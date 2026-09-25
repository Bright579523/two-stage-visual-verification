"""
engine.py — Core Two-Stage Visual Verification Engine
Shared verification backend for Safety Monitoring & Quality Inspection.

Frozen Operating Point (Phase 2A Validated):
  - Rule: AND Logic
  - Stage 1 (OWL-ViT) Threshold (t1): 0.05
  - Stage 2 (CLIP) Threshold (t2): 0.20
"""

import time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    CLIPProcessor, CLIPModel
)

# Frozen operating parameters
V1_THRESHOLD = 0.05
V2_THRESHOLD = 0.20
DECISION_LOGIC = "AND"
CROP_PADDING_RATIO = 0.10


class TwoStageVerificationEngine:
    """Unified two-stage visual verification engine."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Singleton accessor to ensure models are loaded only once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VerificationEngine] Initializing on device: {self.device}...")

        print("[VerificationEngine] Loading OWL-ViT (Stage 1)...")
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        ).to(self.device)
        self.owl_model.eval()

        print("[VerificationEngine] Loading CLIP (Stage 2)...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_model.eval()

        print("[VerificationEngine] Both models loaded and ready!")

    def verify(self, image: Image.Image, prompt: str):
        """
        Executes sequential two-stage verification on input image and prompt.
        Returns visual artifacts, scores, and binary decision.
        """
        if image is None:
            raise ValueError("Input image is required.")
        if not prompt or not prompt.strip():
            raise ValueError("Target query prompt is required.")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        w, h = image.size

        # ---------------------------------------------------------
        # Stage 1: Candidate Proposal via OWL-ViT
        # ---------------------------------------------------------
        start_v1 = time.time()
        texts = [[prompt.strip()]]
        owl_inputs = self.owl_processor(text=texts, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            owl_outputs = self.owl_model(**owl_inputs)

        target_sizes = torch.tensor([[h, w]]).to(self.device)
        post_results = self.owl_processor.image_processor.post_process_object_detection(
            outputs=owl_outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

        latency_v1_ms = (time.time() - start_v1) * 1000.0

        if len(post_results["scores"]) == 0:
            # Fallback if no proposal generated
            return {
                "annotated_image": image,
                "crop_image": None,
                "v1_score": 0.0,
                "v1_pass": False,
                "v2_score": 0.0,
                "v2_pass": False,
                "decision": "Not Confirmed (Filtered)",
                "decision_color": "gray",
                "box": [0, 0, 0, 0],
                "box_dimensions": "0x0px",
                "resolution_warning": False,
                "latency_v1_ms": latency_v1_ms,
                "latency_v2_ms": 0.0
            }

        # Select highest-scoring candidate proposal
        best_idx = torch.argmax(post_results["scores"]).item()
        best_score = float(post_results["scores"][best_idx].item())
        best_box = [int(round(coord)) for coord in post_results["boxes"][best_idx].tolist()]

        xmin = max(0, min(best_box[0], w - 1))
        ymin = max(0, min(best_box[1], h - 1))
        xmax = max(xmin + 1, min(best_box[2], w))
        ymax = max(ymin + 1, min(best_box[3], h))

        box_w = xmax - xmin
        box_h = ymax - ymin

        v1_pass = best_score >= V1_THRESHOLD

        # ---------------------------------------------------------
        # Crop Extraction with 10% Spatial Padding
        # ---------------------------------------------------------
        pad_x = int(box_w * CROP_PADDING_RATIO)
        pad_y = int(box_h * CROP_PADDING_RATIO)
        crop_xmin = max(0, xmin - pad_x)
        crop_ymin = max(0, ymin - pad_y)
        crop_xmax = min(w, xmax + pad_x)
        crop_ymax = min(h, ymax + pad_y)

        cropped_roi = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        # Check resolution bottleneck (<100px)
        resolution_warning = (box_w < 100 or box_h < 100)

        # ---------------------------------------------------------
        # Stage 2: Visual Semantic Verification via CLIP
        # ---------------------------------------------------------
        start_v2 = time.time()
        clip_inputs = self.clip_processor(
            text=[prompt.strip()],
            images=cropped_roi,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            clip_outputs = self.clip_model(**clip_inputs)
            img_embed = clip_outputs.image_embeds / clip_outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_embed = clip_outputs.text_embeds / clip_outputs.text_embeds.norm(dim=-1, keepdim=True)
            clip_similarity = float((img_embed @ txt_embed.T).item())

        latency_v2_ms = (time.time() - start_v2) * 1000.0
        v2_pass = clip_similarity >= V2_THRESHOLD

        # ---------------------------------------------------------
        # AI Decision Gate: Frozen AND Rule
        # ---------------------------------------------------------
        is_confirmed = (v1_pass and v2_pass)
        decision = "Alert Confirmed" if is_confirmed else "Not Confirmed (Filtered)"
        decision_color = "#dc2626" if is_confirmed else "#16a34a"  # Red for Alert, Green for Filtered/Normal

        # ---------------------------------------------------------
        # Render Annotated Bounding Box
        # ---------------------------------------------------------
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        line_width = max(2, int(min(w, h) / 200))
        box_color = decision_color

        for i in range(line_width):
            draw.rectangle(
                [xmin - i, ymin - i, xmax + i, ymax + i],
                outline=box_color
            )

        label_text = f"{decision} | V1: {best_score:.3f} | V2: {clip_similarity:.3f}"
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        tb = draw.textbbox((xmin, max(0, ymin - 22)), label_text, font=font)
        draw.rectangle([tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3], fill=box_color)
        draw.text((xmin, max(0, ymin - 22)), label_text, fill="#ffffff", font=font)

        return {
            "annotated_image": annotated_image,
            "crop_image": cropped_roi,
            "v1_score": best_score,
            "v1_threshold": V1_THRESHOLD,
            "v1_pass": v1_pass,
            "v2_score": clip_similarity,
            "v2_threshold": V2_THRESHOLD,
            "v2_pass": v2_pass,
            "decision": decision,
            "is_confirmed": is_confirmed,
            "decision_color": decision_color,
            "box": [xmin, ymin, xmax, ymax],
            "box_dimensions": f"{box_w}x{box_h}px",
            "resolution_warning": resolution_warning,
            "latency_v1_ms": latency_v1_ms,
            "latency_v2_ms": latency_v2_ms
        }
