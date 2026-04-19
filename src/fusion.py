"""Late-fusion of the per-segment video and audio scores.

Supports two strategies:
- weighted_average: renormalize weights if one modality is missing.
- max: take the higher score (more conservative, flags any-modality tampering).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FusionConfig:
    strategy: str = "weighted_average"   # {weighted_average, max}
    video_weight: float = 0.6
    audio_weight: float = 0.4
    threshold: float = 0.5


def fuse(
    video_fake_prob: Optional[float],
    audio_fake_prob: Optional[float],
    cfg: FusionConfig,
) -> Tuple[Optional[float], Optional[bool]]:
    """Returns (fused_prob, tampered_flag).

    Missing modalities are dropped and the remaining modality is used directly.
    If both are missing, returns (None, None).
    """
    v = video_fake_prob
    a = audio_fake_prob
    if v is None and a is None:
        return None, None

    if cfg.strategy == "max":
        vals = [x for x in (v, a) if x is not None]
        fused = max(vals)
    elif cfg.strategy == "weighted_average":
        wv, wa = cfg.video_weight, cfg.audio_weight
        if v is None:
            fused = a
        elif a is None:
            fused = v
        else:
            total = wv + wa
            fused = (wv * v + wa * a) / total if total > 0 else 0.5 * (v + a)
    else:
        raise ValueError(f"Unknown fusion strategy: {cfg.strategy}")

    flag = bool(fused >= cfg.threshold) if fused is not None else None
    return fused, flag
