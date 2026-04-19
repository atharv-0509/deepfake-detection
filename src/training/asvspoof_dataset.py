"""PyTorch Dataset for ASVspoof 2019 LA.

Loads FLAC clips listed in a manifest CSV (produced by
scripts/prepare_asvspoof.py), resamples to 16 kHz mono, and crops/pads
to a fixed 1-second window so training matches how the inference
pipeline scores audio (one prediction per 1s segment).

Returns dicts compatible with HuggingFace Trainer:
    {"input_values": FloatTensor[T], "labels": LongTensor[]}
The AutoFeatureExtractor is applied inside __getitem__ to avoid
memory blowups from pre-featurizing the whole dataset.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ASVspoofExample:
    file_path: str
    label: int
    spoof_type: str
    speaker_id: str


def _load_manifest(path: str | Path) -> list[ASVspoofExample]:
    rows: list[ASVspoofExample] = []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ASVspoofExample(
                    file_path=r["file_path"],
                    label=int(r["label"]),
                    spoof_type=r["spoof_type"],
                    speaker_id=r["speaker_id"],
                )
            )
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio as mono float32 at target_sr using soundfile + (optionally) librosa."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # librosa lazy-imported so inference-only installs don't need it
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def _crop_or_pad(audio: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    """Random-crop (train) or center-crop (eval) to target_len samples. Pad with zeros if shorter."""
    n = len(audio)
    if n == target_len:
        return audio
    if n > target_len:
        start = rng.randint(0, n - target_len)
        return audio[start : start + target_len]
    # shorter -> pad with zeros on the right
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = audio
    return out


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA as fixed 1-second 16 kHz clips.

    Parameters
    ----------
    manifest_csv : str
        Path to a CSV produced by scripts/prepare_asvspoof.py.
    feature_extractor : callable
        HuggingFace AutoFeatureExtractor instance.
    segment_seconds : float, default 1.0
        Clip length in seconds (must match segment_seconds in configs/default.yaml).
    sample_rate : int, default 16000
    training : bool, default True
        If True, random-crop and (optionally) add light noise.
        If False, center-crop, deterministic.
    noise_prob : float, default 0.0
        Probability of adding Gaussian noise at std=0.005 during training.
    seed : int, default 0
        RNG seed for reproducibility. Only affects training cropping/noise.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        feature_extractor: Callable,
        segment_seconds: float = 1.0,
        sample_rate: int = 16000,
        training: bool = True,
        noise_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.examples = _load_manifest(manifest_csv)
        self.feature_extractor = feature_extractor
        self.segment_seconds = segment_seconds
        self.sample_rate = sample_rate
        self.target_len = int(round(segment_seconds * sample_rate))
        self.training = training
        self.noise_prob = noise_prob
        self._seed = seed

    def __len__(self) -> int:
        return len(self.examples)

    def _rng_for(self, idx: int) -> random.Random:
        # Deterministic in eval, epoch-varying in train via global torch RNG state.
        if self.training:
            return random.Random()  # nondeterministic by design
        return random.Random(self._seed + idx)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        audio = _load_audio(ex.file_path, target_sr=self.sample_rate)
        rng = self._rng_for(idx)

        if not self.training:
            # deterministic center-crop
            n = len(audio)
            if n > self.target_len:
                start = (n - self.target_len) // 2
                audio = audio[start : start + self.target_len]
            elif n < self.target_len:
                pad = np.zeros(self.target_len, dtype=np.float32)
                pad[:n] = audio
                audio = pad
        else:
            audio = _crop_or_pad(audio, self.target_len, rng)
            if self.noise_prob > 0 and rng.random() < self.noise_prob:
                audio = audio + rng.gauss(0.0, 0.005) * np.random.randn(
                    self.target_len
                ).astype(np.float32)

        features = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        # feature_extractor returns [1, T]; squeeze the batch dim
        input_values = features["input_values"].squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(ex.label, dtype=torch.long),
        }


def compute_class_weights(manifest_csv: str | Path) -> torch.Tensor:
    """Return a 2-element tensor of inverse-frequency class weights.

    ASVspoof LA is heavily imbalanced (~10x more spoof than bonafide).
    Pass the returned weights as the `weight` arg of CrossEntropyLoss
    to counteract this without resampling.
    """
    examples = _load_manifest(manifest_csv)
    counts = np.array([0, 0], dtype=np.int64)
    for ex in examples:
        counts[ex.label] += 1
    counts = np.maximum(counts, 1)
    inv = counts.sum() / (2.0 * counts)
    return torch.tensor(inv, dtype=torch.float32)


def pos_weight_for_bce(manifest_csv: str | Path) -> float:
    """For BCE on logit[spoof], return n_bonafide / n_spoof."""
    examples = _load_manifest(manifest_csv)
    n_spoof = sum(1 for e in examples if e.label == 1)
    n_bona = sum(1 for e in examples if e.label == 0)
    return n_bona / max(n_spoof, 1)
