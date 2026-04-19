"""Video (visual) deepfake detector.

Per segment:
  1. Take N evenly-spaced frames.
  2. Run MTCNN face detection; crop the largest face (or skip).
  3. Run a HuggingFace image-classification model on each face crop.
  4. Aggregate per-frame "fake" probabilities into one segment score
     (mean / max / median).

This keeps the pipeline robust to model swaps: any HuggingFace image
classifier with real/fake-style labels will work — `normalize_fake_prob`
figures out which class is the "fake" class by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .utils import get_device, logger, normalize_fake_prob


@dataclass
class VideoDetectorConfig:
    model_id: str = "prithivMLmods/Deep-Fake-Detector-Model"
    face_min_size: int = 60
    face_margin: float = 0.2
    face_select: str = "largest"     # {largest, all}
    aggregate: str = "mean"          # {mean, max, median}
    no_face_strategy: str = "skip"   # {skip, use_full_frame, mark_unknown}
    batch_size: int = 8


class VideoDeepfakeDetector:
    def __init__(self, cfg: VideoDetectorConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or get_device()
        self._hf = None           # (image_processor, model)
        self._mtcnn = None

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if self._hf is not None:
            return self._hf
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        logger.info("Loading video deepfake model: %s", self.cfg.model_id)
        processor = AutoImageProcessor.from_pretrained(self.cfg.model_id)
        model = AutoModelForImageClassification.from_pretrained(self.cfg.model_id)
        model.eval().to(self.device)
        self._hf = (processor, model, torch)
        return self._hf

    def _load_face_detector(self):
        if self._mtcnn is not None:
            return self._mtcnn
        from facenet_pytorch import MTCNN

        logger.info("Loading MTCNN face detector")
        self._mtcnn = MTCNN(
            keep_all=(self.cfg.face_select == "all"),
            device=self.device,
            min_face_size=self.cfg.face_min_size,
            post_process=False,
        )
        return self._mtcnn

    # ------------------------------------------------------------------ #
    # Face cropping
    # ------------------------------------------------------------------ #
    def _crop_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Return face crops (RGB uint8 HxWx3). May be empty."""
        mtcnn = self._load_face_detector()
        from PIL import Image

        pil = Image.fromarray(frame)
        boxes, probs = mtcnn.detect(pil)
        if boxes is None or len(boxes) == 0:
            if self.cfg.no_face_strategy == "use_full_frame":
                return [frame]
            return []

        # Sort by area descending
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        order = np.argsort(areas)[::-1]
        if self.cfg.face_select == "largest":
            order = order[:1]

        crops: List[np.ndarray] = []
        H, W, _ = frame.shape
        for idx in order:
            x1, y1, x2, y2 = boxes[idx]
            w, h = x2 - x1, y2 - y1
            m = self.cfg.face_margin
            x1 = max(0, int(x1 - m * w))
            y1 = max(0, int(y1 - m * h))
            x2 = min(W, int(x2 + m * w))
            y2 = min(H, int(y2 + m * h))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crops.append(frame[y1:y2, x1:x2])
        return crops

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score_crops(self, crops: List[np.ndarray]) -> List[float]:
        if not crops:
            return []
        processor, model, torch = self._load_model()
        from PIL import Image

        pil_imgs = [Image.fromarray(c) for c in crops]
        probs_out: List[float] = []
        for i in range(0, len(pil_imgs), self.cfg.batch_size):
            batch = pil_imgs[i : i + self.cfg.batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            id2label = model.config.id2label
            for row in probs:
                probs_out.append(normalize_fake_prob(row, id2label))
        return probs_out

    @staticmethod
    def _aggregate(values: List[float], how: str) -> float:
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=np.float64)
        if how == "mean":
            return float(arr.mean())
        if how == "max":
            return float(arr.max())
        if how == "median":
            return float(np.median(arr))
        raise ValueError(f"Unknown aggregation '{how}'")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score_segment(
        self, frames: List[np.ndarray]
    ) -> Tuple[Optional[float], int, str, Optional[np.ndarray]]:
        """
        Returns (fake_prob, n_frames_scored, note, best_crop).

        fake_prob is None if no valid face was found and strategy=skip.
        best_crop is the RGB uint8 face crop from the single highest-scoring
        face in the segment (or None if no face was scored). Useful for
        attaching visual evidence to flagged segments in a UI.
        """
        if not frames:
            return None, 0, "no_frames", None

        per_frame_scores: List[float] = []
        best_score = -1.0
        best_crop: Optional[np.ndarray] = None
        n_scored = 0
        no_face_count = 0
        for frame in frames:
            crops = self._crop_faces(frame)
            if not crops:
                no_face_count += 1
                continue
            crop_scores = self._score_crops(crops)
            if not crop_scores:
                continue
            # Any face fake -> frame fake. Track the single most-fake crop.
            arr_scores = np.asarray(crop_scores, dtype=np.float64)
            max_idx = int(arr_scores.argmax())
            frame_score = float(arr_scores[max_idx])
            per_frame_scores.append(frame_score)
            if frame_score > best_score:
                best_score = frame_score
                best_crop = crops[max_idx]
            n_scored += 1

        if not per_frame_scores:
            if self.cfg.no_face_strategy == "mark_unknown":
                return None, 0, "no_face_in_segment", None
            if self.cfg.no_face_strategy == "use_full_frame":
                # Already handled inside _crop_faces — if we got here, scoring failed.
                return None, 0, "no_face_fallback_failed", None
            return None, 0, "no_face", None

        score = self._aggregate(per_frame_scores, self.cfg.aggregate)
        note = f"no_face_in_{no_face_count}/{len(frames)}" if no_face_count else ""
        return score, n_scored, note, best_crop
