"""Smoke tests for the pipeline. These do not download models; they test
that modules import and that the non-ML plumbing (segmenting, fusion,
label normalization) is correct.

Run with:
    python -m pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.fusion import FusionConfig, fuse
from src.utils import Segment, normalize_fake_prob


def test_fuse_both_modalities():
    cfg = FusionConfig(video_weight=0.6, audio_weight=0.4, threshold=0.5)
    p, flag = fuse(0.8, 0.2, cfg)
    # (0.6*0.8 + 0.4*0.2) / 1.0 = 0.56
    assert abs(p - 0.56) < 1e-6
    assert flag is True


def test_fuse_missing_video():
    cfg = FusionConfig(video_weight=0.6, audio_weight=0.4, threshold=0.5)
    p, flag = fuse(None, 0.9, cfg)
    assert p == 0.9
    assert flag is True


def test_fuse_both_missing():
    cfg = FusionConfig()
    p, flag = fuse(None, None, cfg)
    assert p is None
    assert flag is None


def test_fuse_max_strategy():
    cfg = FusionConfig(strategy="max", threshold=0.5)
    p, flag = fuse(0.3, 0.7, cfg)
    assert p == 0.7
    assert flag is True


def test_normalize_fake_prob_simple():
    probs = np.array([0.1, 0.9])
    id2label = {0: "REAL", 1: "FAKE"}
    assert abs(normalize_fake_prob(probs, id2label) - 0.9) < 1e-6


def test_normalize_fake_prob_alt_labels():
    probs = np.array([0.8, 0.2])
    id2label = {0: "bonafide", 1: "spoof"}
    assert abs(normalize_fake_prob(probs, id2label) - 0.2) < 1e-6


def test_normalize_fake_prob_case_insensitive():
    probs = np.array([0.3, 0.7])
    id2label = {0: "Real Image", 1: "Deepfake Image"}
    assert abs(normalize_fake_prob(probs, id2label) - 0.7) < 1e-6


def test_segment_roundtrip():
    s = Segment(index=0, t_start=0.0, t_end=1.0,
                video_fake_prob=0.1, audio_fake_prob=0.2,
                fused_fake_prob=0.14, tampered=False, audio_available=True)
    d = s.to_dict()
    assert d["t_start"] == 0.0
    assert d["tampered"] is False
    assert d["fused_fake_prob"] == 0.14
