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


def probe_rotation_degrees(video_path: str | Path) -> int:
    """Return the display rotation (degrees clockwise) for a video.

    Handles both legacy `stream.tags.rotate` and modern side-data
    `rotation` entries (iPhone 14+, WhatsApp, newer Android). Returns 0
    if ffprobe is missing or no rotation is recorded.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate:side_data=rotation",
            "-of", "default=nw=1",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, text=True, timeout=30)
    except Exception:
        return 0

    # Parse lines like: "TAG:rotate=90" and/or "rotation=-180"
    deg = 0
    for line in out.splitlines():
        line = line.strip().lower()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key in ("tag:rotate", "rotate", "rotation"):
            try:
                deg = int(float(val))
                break  # first match wins; all frames share the same rotation
            except ValueError:
                continue
    # Normalize to 0..359 clockwise
    return ((deg % 360) + 360) % 360


def _transpose_filter_for_rotation(deg: int) -> Optional[str]:
    """Translate a clockwise rotation into an ffmpeg -vf transpose chain."""
    deg = ((deg % 360) + 360) % 360
    if deg == 0:
        return None
    if deg == 90:
        return "transpose=1"               # 90 clockwise
    if deg == 180:
        return "transpose=2,transpose=2"   # flip 180 via two CCW
    if deg == 270:
        return "transpose=2"               # 90 counter-clockwise
    return None  # non-orthogonal; leave to ffmpeg's autorotate


def normalize_video_for_pipeline(video_path: str | Path) -> str:
    """Transcode any video to H.264 + AAC with rotation baked into pixels.

    This guards against three common upload failure modes:
      - Rotation metadata (iPhone 14+ `display_matrix`, WhatsApp, older
        iPhones). We probe the rotation ourselves with ffprobe (covering
        both stream tags AND side-data), apply an explicit transpose
        filter, then strip the rotation tag so downstream decoders don't
        double-rotate.
      - VP9 (YouTube downloads) and HEVC (modern iPhone) — some decoders
        and the Gradio player reject these.
      - Weird containers / variable framerate / missing moov atoms.

    If ffmpeg isn't on PATH or the transcode fails, returns the original
    path so downstream code at least attempts to proceed.
    """
    import uuid

    if not video_path:
        return str(video_path)

    src = Path(video_path)
    rotation = probe_rotation_degrees(src)
    logger.info("probe_rotation_degrees(%s) = %d", src.name, rotation)

    out_path = Path(tempfile.gettempdir()) / f"dfdet_norm_{uuid.uuid4().hex[:8]}.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(src)]

    vf = _transpose_filter_for_rotation(rotation)
    if vf:
        cmd += ["-vf", vf]

    cmd += [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        # After baking rotation into pixels, clear the rotation metadata
        # so decoders don't re-apply it on top.
        "-metadata:s:v:0", "rotate=0",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        logger.info("Normalized upload -> %s (rotation baked: %d deg)", out_path, rotation)
        return str(out_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH — skipping normalization.")
        return str(video_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr_tail = ""
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            stderr_tail = e.stderr.decode("utf-8", "ignore")[-400:]
        logger.warning(
            "Normalization transcode failed (%s) — falling back to original.\n%s",
            type(e).__name__, stderr_tail,
        )
        return str(video_path)


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
