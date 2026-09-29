"""Pure (dependency-light) helpers for segment-level labelling.

Kept free of torch / soundfile so the interval math can be imported and
unit-tested cheaply, and shared by both the dataset class and the
partial-spoof synthesis script.

A "fake_segments" manifest cell is a ";"-separated list of "start-end"
timestamps in seconds (empty = fully bonafide). These helpers parse that
spec and decide, for a given fixed-width window, whether it counts as fake.
"""
from __future__ import annotations

from typing import List, Tuple


def parse_fake_segments(spec: str | None) -> List[Tuple[float, float]]:
    """Parse a ``fake_segments`` cell into a list of (start, end) seconds.

    Accepts ``None``/empty (fully bonafide), a single ``"a-b"`` pair, or a
    ";"-separated list ``"a1-b1;a2-b2"``. Commas are treated like ";".
    Reversed or zero-length pairs are dropped. Result is sorted by start.
    """
    if not spec:
        return []
    out: List[Tuple[float, float]] = []
    for chunk in str(spec).replace(",", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        a_str, _, b_str = chunk.partition("-")
        try:
            a, b = float(a_str), float(b_str)
        except ValueError:
            continue
        if b > a:
            out.append((a, b))
    out.sort()
    return out


def fake_overlap_seconds(
    win_start: float, win_end: float, fakes: List[Tuple[float, float]]
) -> float:
    """Total seconds of [win_start, win_end) covered by any fake interval."""
    total = 0.0
    for a, b in fakes:
        lo = max(win_start, a)
        hi = min(win_end, b)
        if hi > lo:
            total += hi - lo
    return total


def window_label(
    win_start: float,
    win_end: float,
    fakes: List[Tuple[float, float]],
    positive_overlap: float = 0.5,
) -> int:
    """Label a window fake (1) when the fraction of it covered by fake
    intervals is >= ``positive_overlap``, else bonafide (0)."""
    dur = win_end - win_start
    if dur <= 0:
        return 0
    frac = fake_overlap_seconds(win_start, win_end, fakes) / dur
    return 1 if frac >= positive_overlap else 0


def merge_intervals(
    intervals: List[Tuple[float, float]], gap: float = 0.01
) -> List[Tuple[float, float]]:
    """Merge overlapping / near-adjacent (within ``gap`` s) intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + gap:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def fmt_segments(fakes: List[Tuple[float, float]]) -> str:
    """Render intervals back into a ``fake_segments`` manifest cell."""
    return ";".join(f"{a:.3f}-{b:.3f}" for a, b in fakes)
