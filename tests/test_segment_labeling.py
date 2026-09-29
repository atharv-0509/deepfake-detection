"""Tests for segment-level labelling helpers (torch-free).

These cover the interval math that turns per-file fake intervals into
per-window labels — the core of making detection segment-wise.

Run with:
    python -m pytest tests/test_segment_labeling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.training.segment_labeling import (  # noqa: E402
    fake_overlap_seconds,
    fmt_segments,
    merge_intervals,
    parse_fake_segments,
    window_label,
)


# --------------------------------------------------------------------------- #
# parse_fake_segments
# --------------------------------------------------------------------------- #
def test_parse_empty_is_bonafide():
    assert parse_fake_segments("") == []
    assert parse_fake_segments(None) == []


def test_parse_single_pair():
    assert parse_fake_segments("1.0-2.5") == [(1.0, 2.5)]


def test_parse_multiple_and_sorted():
    assert parse_fake_segments("3.0-4.0;1.0-2.0") == [(1.0, 2.0), (3.0, 4.0)]


def test_parse_drops_reversed_and_garbage():
    # reversed pair dropped, non-numeric dropped, comma treated like ';'
    assert parse_fake_segments("2.0-1.0,1.0-1.5,x-y") == [(1.0, 1.5)]


# --------------------------------------------------------------------------- #
# overlap + labelling
# --------------------------------------------------------------------------- #
def test_overlap_seconds():
    fakes = [(1.0, 2.0), (3.0, 3.5)]
    # window [0,2): overlaps (1,2) -> 1.0s
    assert abs(fake_overlap_seconds(0.0, 2.0, fakes) - 1.0) < 1e-9
    # window [3,4): overlaps (3,3.5) -> 0.5s
    assert abs(fake_overlap_seconds(3.0, 4.0, fakes) - 0.5) < 1e-9


def test_window_label_majority_fake():
    fakes = [(0.0, 0.8)]
    # 0.8 of a 1s window is fake -> >= 0.5 -> spoof
    assert window_label(0.0, 1.0, fakes, positive_overlap=0.5) == 1


def test_window_label_minority_fake_is_real():
    fakes = [(0.0, 0.3)]
    # only 0.3 of the window is fake -> below 0.5 -> bonafide
    assert window_label(0.0, 1.0, fakes, positive_overlap=0.5) == 0


def test_window_label_fully_real():
    assert window_label(0.0, 1.0, [], positive_overlap=0.5) == 0


def test_window_label_threshold_boundary_inclusive():
    fakes = [(0.0, 0.5)]
    # exactly 0.5 overlap fraction with default threshold -> spoof (>=)
    assert window_label(0.0, 1.0, fakes, positive_overlap=0.5) == 1


def test_window_label_respects_custom_threshold():
    fakes = [(0.0, 0.4)]
    assert window_label(0.0, 1.0, fakes, positive_overlap=0.3) == 1
    assert window_label(0.0, 1.0, fakes, positive_overlap=0.5) == 0


# --------------------------------------------------------------------------- #
# merge + format round-trip
# --------------------------------------------------------------------------- #
def test_merge_intervals_overlapping():
    assert merge_intervals([(0.0, 1.0), (0.9, 2.0)]) == [(0.0, 2.0)]


def test_merge_intervals_disjoint_kept():
    assert merge_intervals([(0.0, 1.0), (3.0, 4.0)]) == [(0.0, 1.0), (3.0, 4.0)]


def test_fmt_and_parse_roundtrip():
    fakes = [(1.0, 2.0), (3.5, 4.25)]
    assert parse_fake_segments(fmt_segments(fakes)) == fakes
