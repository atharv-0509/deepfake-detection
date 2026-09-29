"""Segment-level audio dataset for *partial* spoof detection.

The whole-file datasets (`ASVspoofDataset`) attach ONE label to every file,
so a model trained on them learns "is this entire utterance synthetic?" and
tends to flag a partially-manipulated clip as fake end-to-end. To detect
tampering at the segment level the model has to see files where real and
fake audio coexist, with per-segment ground truth.

This module reads a *segment-level* manifest — one row per audio file plus
the intervals (in seconds) that are fake — and enumerates fixed-width
windows across every file. Each window is labelled fake when a large enough
fraction of it overlaps a fake interval, so a single spliced clip produces a
mixture of bonafide and spoof training windows.

Manifest schema (CSV, produced by scripts/make_partialspoof.py):

    file_path,duration,fake_segments,speaker_id,source

where `fake_segments` is a ";"-separated list of "start-end" pairs in
seconds (empty = fully bonafide; a single "0-<duration>" pair = fully
spoof). Example rows::

    /data/real/LJ001.wav,3.20,,LJSpeech,bonafide
    /data/spoof/A06_1.wav,2.75,0-2.75,LA_0079,spoof
    /data/partial/mix_00001.wav,4.00,1.10-2.30;3.05-3.60,mixed,partial

Returns dicts compatible with HuggingFace Trainer::

    {"input_values": FloatTensor[T], "labels": LongTensor[]}
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .segment_labeling import parse_fake_segments, window_label


# --------------------------------------------------------------------------- #
# Manifest rows
# --------------------------------------------------------------------------- #
@dataclass
class SegmentExample:
    file_path: str
    duration: float
    fake_segments: List[Tuple[float, float]]
    speaker_id: str = ""
    source: str = ""


@dataclass
class _Window:
    example_idx: int
    t_start: float
    t_end: float
    label: int


def _load_segment_manifest(path: str | Path) -> List[SegmentExample]:
    rows: List[SegmentExample] = []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        required = {"file_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path}: manifest must have at least a 'file_path' column; "
                f"got {reader.fieldnames}"
            )
        for r in reader:
            dur = r.get("duration", "")
            rows.append(
                SegmentExample(
                    file_path=r["file_path"],
                    duration=float(dur) if dur not in (None, "") else 0.0,
                    fake_segments=parse_fake_segments(r.get("fake_segments")),
                    speaker_id=r.get("speaker_id", "") or "",
                    source=r.get("source", "") or "",
                )
            )
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio as mono float32 at target_sr."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


class SegmentLevelAudioDataset(Dataset):
    """Fixed-width windows over segment-labelled audio files.

    One dataset item is a single ``segment_seconds`` window, labelled by its
    overlap with the file's fake intervals. Windows are enumerated once at
    construction so ``len()`` is the total number of windows across all files
    (this is what makes it *segment*-level rather than *file*-level).

    Parameters
    ----------
    manifest_csv : path to a segment-level manifest (see module docstring).
    feature_extractor : HuggingFace AutoFeatureExtractor instance.
    segment_seconds : window width in seconds (match configs/default.yaml).
    segment_stride_seconds : hop between windows; defaults to no overlap.
    sample_rate : target sample rate.
    positive_overlap : fraction of a window that must be fake to label it 1.
    training : if True, apply light augmentation (jitter + optional noise).
    noise_prob : probability of adding Gaussian noise during training.
    max_windows_per_file : cap windows taken from any one file (0 = no cap).
    seed : RNG seed (eval is deterministic; train jitters within the window).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        feature_extractor: Callable,
        segment_seconds: float = 1.0,
        segment_stride_seconds: Optional[float] = None,
        sample_rate: int = 16000,
        positive_overlap: float = 0.5,
        training: bool = True,
        noise_prob: float = 0.0,
        max_windows_per_file: int = 0,
        seed: int = 0,
    ) -> None:
        self.examples = _load_segment_manifest(manifest_csv)
        self.feature_extractor = feature_extractor
        self.segment_seconds = float(segment_seconds)
        self.segment_stride_seconds = float(
            segment_stride_seconds or segment_seconds
        )
        self.sample_rate = int(sample_rate)
        self.target_len = int(round(self.segment_seconds * self.sample_rate))
        self.positive_overlap = float(positive_overlap)
        self.training = training
        self.noise_prob = noise_prob
        self.max_windows_per_file = int(max_windows_per_file)
        self._seed = seed
        # Cache the most-recently decoded file so adjacent windows of one file
        # (common in eval / no-shuffle) don't re-decode it.
        self._cache_path: Optional[str] = None
        self._cache_audio: Optional[np.ndarray] = None

        self.windows: List[_Window] = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"No windows produced from {manifest_csv}. Are durations set?"
            )

    # ------------------------------------------------------------------ #
    def _build_windows(self) -> List[_Window]:
        windows: List[_Window] = []
        for ei, ex in enumerate(self.examples):
            dur = ex.duration
            if dur <= 0:
                # Unknown duration — treat the whole file as one window.
                windows.append(
                    _Window(ei, 0.0, self.segment_seconds,
                            1 if ex.fake_segments else 0)
                )
                continue
            n_win = 0
            t = 0.0
            while t + 0.1 < dur:  # skip tiny tail (< 0.1s)
                t_end = min(t + self.segment_seconds, dur)
                lab = window_label(t, t_end, ex.fake_segments, self.positive_overlap)
                windows.append(_Window(ei, t, t_end, lab))
                n_win += 1
                if self.max_windows_per_file and n_win >= self.max_windows_per_file:
                    break
                t += self.segment_stride_seconds
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    # ------------------------------------------------------------------ #
    def labels(self) -> np.ndarray:
        """All window labels (handy for class weighting / stats)."""
        return np.asarray([w.label for w in self.windows], dtype=np.int64)

    def _get_audio(self, path: str) -> np.ndarray:
        if self._cache_path == path and self._cache_audio is not None:
            return self._cache_audio
        audio = _load_audio(path, target_sr=self.sample_rate)
        self._cache_path = path
        self._cache_audio = audio
        return audio

    def _slice_window(
        self, audio: np.ndarray, w: _Window, rng: random.Random
    ) -> np.ndarray:
        i0 = int(round(w.t_start * self.sample_rate))
        # Training: jitter the start a little so the model doesn't overfit to
        # the exact splice boundaries, but keep it inside the same window's
        # label region by clamping to +/- 10% of the window.
        if self.training:
            jitter = int(rng.uniform(-0.1, 0.1) * self.target_len)
            i0 = max(0, i0 + jitter)
        seg = audio[i0 : i0 + self.target_len]
        if len(seg) < self.target_len:
            pad = np.zeros(self.target_len, dtype=np.float32)
            pad[: len(seg)] = seg
            seg = pad
        return seg

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        ex = self.examples[w.example_idx]
        rng = random.Random() if self.training else random.Random(self._seed + idx)

        audio = self._get_audio(ex.file_path)
        seg = self._slice_window(audio, w, rng)

        if self.training and self.noise_prob > 0 and rng.random() < self.noise_prob:
            seg = seg + rng.gauss(0.0, 0.005) * np.random.randn(
                self.target_len
            ).astype(np.float32)

        features = self.feature_extractor(
            seg,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = features["input_values"].squeeze(0)
        return {
            "input_values": input_values,
            "labels": torch.tensor(w.label, dtype=torch.long),
        }


def compute_window_class_weights(dataset: SegmentLevelAudioDataset) -> torch.Tensor:
    """Inverse-frequency class weights over the dataset's *windows*.

    Partial-spoof data is usually bonafide-heavy at the window level (most of
    a clip is real), so weighting counteracts the imbalance without resampling.
    """
    labels = dataset.labels()
    counts = np.array(
        [int((labels == 0).sum()), int((labels == 1).sum())], dtype=np.int64
    )
    counts = np.maximum(counts, 1)
    inv = counts.sum() / (2.0 * counts)
    return torch.tensor(inv, dtype=torch.float32)
