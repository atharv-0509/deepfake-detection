"""Shared utilities: ffmpeg I/O, device selection, logging, label normalization."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger("deepfake_detector")


# --------------------------------------------------------------------------- #
# Device / logging
# --------------------------------------------------------------------------- #
def get_device() -> str:
    """Pick the best available torch device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch not installed yet
        pass
    return "cpu"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #
def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg / ffprobe not found on PATH. Install with "
            "`apt-get install ffmpeg` or `brew install ffmpeg`."
        )


def probe_duration(path: str | Path) -> float:
    """Return duration (seconds) of a media file via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def extract_audio_wav(
    video_path: str | Path,
    out_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """Extract mono PCM wav at `sample_rate` from video."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


# --------------------------------------------------------------------------- #
# Label normalization
# --------------------------------------------------------------------------- #
FAKE_KEYWORDS = {"fake", "spoof", "deepfake", "manipulated", "synthetic", "tampered", "ai"}
REAL_KEYWORDS = {"real", "bonafide", "genuine", "original", "authentic", "human"}


def normalize_fake_prob(logits_or_probs: np.ndarray, id2label: dict[int, str]) -> float:
    """Given a 1-D vector of class probabilities and an id->label map,
    return the probability mass on the "fake" class(es).

    Robust to models that use {FAKE, REAL}, {spoof, bonafide}, {0: 'Fake', 1: 'Real'}, etc.
    """
    probs = np.asarray(logits_or_probs).astype(np.float64).ravel()
    if probs.sum() <= 0 or np.isnan(probs).any():
        return 0.0

    # softmax if values look like logits
    if probs.min() < 0 or probs.max() > 1.0 + 1e-3:
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

    fake_mass = 0.0
    real_mass = 0.0
    for idx, p in enumerate(probs):
        label = str(id2label.get(idx, idx)).lower()
        if any(k in label for k in FAKE_KEYWORDS):
            fake_mass += p
        elif any(k in label for k in REAL_KEYWORDS):
            real_mass += p

    if fake_mass + real_mass == 0:
        # Unknown label scheme — assume index 1 is fake as a convention
        return float(probs[-1])
    # If only one side present, return its inverse when appropriate
    if fake_mass == 0 and real_mass > 0:
        return float(1.0 - real_mass)
    return float(fake_mass / max(fake_mass + real_mass, 1e-9))


# --------------------------------------------------------------------------- #
# Small data containers
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    index: int
    t_start: float
    t_end: float
    video_fake_prob: Optional[float] = None
    audio_fake_prob: Optional[float] = None
    fused_fake_prob: Optional[float] = None
    tampered: Optional[bool] = None
    video_n_frames_scored: int = 0
    audio_available: bool = False
    notes: str = ""
    face_crop_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "video_fake_prob": _r(self.video_fake_prob),
            "audio_fake_prob": _r(self.audio_fake_prob),
            "fused_fake_prob": _r(self.fused_fake_prob),
            "tampered": self.tampered,
            "video_n_frames_scored": self.video_n_frames_scored,
            "audio_available": self.audio_available,
            "notes": self.notes,
            "face_crop_path": self.face_crop_path,
        }


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def temp_workdir() -> Path:
    p = Path(tempfile.mkdtemp(prefix="dfdet_"))
    return p
