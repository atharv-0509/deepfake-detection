"""Audio tampering / spoof detector.

Per segment:
  1. Receive a 1s (or config-width) mono float32 waveform @ target SR.
  2. If voiced-ratio is below the floor, mark the segment unknown.
  3. Run a HuggingFace audio-classification model and return the
     probability mass on the "fake/spoof" class(es).

Any `AutoModelForAudioClassification` with real/fake-style labels works;
label names are resolved by `normalize_fake_prob`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utils import get_device, logger, normalize_fake_prob


@dataclass
class AudioDetectorConfig:
    model_id: str = "MelodyMachine/Deepfake-audio-detection"
    sample_rate: int = 16000
    min_voiced_ratio: float = 0.05


def _voiced_ratio(wav: np.ndarray, sr: int) -> float:
    """Rough VAD via short-time energy. Returns the fraction of frames whose
    RMS is above a floor. Avoids a heavy VAD dependency — good enough to skip
    silence/music-only segments."""
    if wav.size == 0:
        return 0.0
    frame = int(0.02 * sr)  # 20ms frames
    if frame <= 0:
        return 0.0
    n = (len(wav) // frame) * frame
    if n == 0:
        return 0.0
    x = wav[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(x**2, axis=1) + 1e-12)
    # Adaptive floor: 3x global median RMS or absolute floor 1e-3.
    floor = max(3 * float(np.median(rms)), 1e-3)
    return float((rms > floor).mean())


class AudioDeepfakeDetector:
    def __init__(self, cfg: AudioDetectorConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or get_device()
        self._hf = None  # (feature_extractor, model, torch)

    def _load(self):
        if self._hf is not None:
            return self._hf
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        logger.info("Loading audio deepfake model: %s", self.cfg.model_id)
        extractor = AutoFeatureExtractor.from_pretrained(self.cfg.model_id)
        model = AutoModelForAudioClassification.from_pretrained(self.cfg.model_id)
        model.eval().to(self.device)
        self._hf = (extractor, model, torch)
        return self._hf

    def score_segment(
        self, wav: Optional[np.ndarray], sr: int
    ) -> Tuple[Optional[float], bool, str]:
        """
        Returns (fake_prob, audio_available, note).
        fake_prob is None when audio is missing / too short / unvoiced.
        """
        if wav is None or wav.size < int(0.1 * sr):
            return None, False, "no_audio"

        if sr != self.cfg.sample_rate:
            import librosa

            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=self.cfg.sample_rate)
            sr = self.cfg.sample_rate

        vr = _voiced_ratio(wav, sr)
        if vr < self.cfg.min_voiced_ratio:
            return None, True, f"unvoiced(vr={vr:.2f})"

        extractor, model, torch = self._load()
        # Normalize and run
        inputs = extractor(
            wav.astype(np.float32),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        id2label = model.config.id2label
        p_fake = normalize_fake_prob(probs, id2label)
        return float(p_fake), True, ""
