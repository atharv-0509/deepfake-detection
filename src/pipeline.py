"""End-to-end pipeline: video path in -> segment timeline out."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # tqdm is optional — fall back to a no-op wrapper
    def tqdm(iterable=None, **_kwargs):  # type: ignore
        return iterable if iterable is not None else iter(())

from .audio_detector import AudioDetectorConfig, AudioDeepfakeDetector
from .fusion import FusionConfig, fuse
from .segmenter import VideoAudioSegmenter
from .utils import Segment, check_ffmpeg, get_device, logger, setup_logging
from .video_detector import VideoDeepfakeDetector, VideoDetectorConfig


@dataclass
class PipelineConfig:
    segment_seconds: float = 1.0
    segment_stride_seconds: float = 1.0
    video: VideoDetectorConfig = field(default_factory=VideoDetectorConfig)
    audio: AudioDetectorConfig = field(default_factory=AudioDetectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    video_enabled: bool = True
    audio_enabled: bool = True
    frames_per_segment: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        vid = raw.get("video", {}) or {}
        aud = raw.get("audio", {}) or {}
        fus = raw.get("fusion", {}) or {}
        return cls(
            segment_seconds=float(raw.get("segment_seconds", 1.0)),
            segment_stride_seconds=float(raw.get("segment_stride_seconds", raw.get("segment_seconds", 1.0))),
            video_enabled=bool(vid.get("enabled", True)),
            audio_enabled=bool(aud.get("enabled", True)),
            frames_per_segment=int(vid.get("frames_per_segment", 5)),
            video=VideoDetectorConfig(
                model_id=vid.get("model_id", VideoDetectorConfig.model_id),
                face_min_size=int(vid.get("face_min_size", 60)),
                face_margin=float(vid.get("face_margin", 0.2)),
                face_select=vid.get("face_select", "largest"),
                aggregate=vid.get("aggregate", "mean"),
                no_face_strategy=vid.get("no_face_strategy", "skip"),
            ),
            audio=AudioDetectorConfig(
                model_id=aud.get("model_id", AudioDetectorConfig.model_id),
                sample_rate=int(aud.get("sample_rate", 16000)),
                min_voiced_ratio=float(aud.get("min_voiced_ratio", 0.05)),
            ),
            fusion=FusionConfig(
                strategy=fus.get("strategy", "weighted_average"),
                video_weight=float(fus.get("video_weight", 0.6)),
                audio_weight=float(fus.get("audio_weight", 0.4)),
                threshold=float(fus.get("threshold", 0.5)),
            ),
        )


class DeepfakePipeline:
    def __init__(self, cfg: PipelineConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or get_device()
        self.video_det = VideoDeepfakeDetector(cfg.video, self.device) if cfg.video_enabled else None
        self.audio_det = AudioDeepfakeDetector(cfg.audio, self.device) if cfg.audio_enabled else None

    # ------------------------------------------------------------------ #
    def run(
        self,
        video_path: str | Path,
        progress: bool = True,
        face_crops_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline.

        If `face_crops_dir` is provided, the best-scoring face crop from
        each segment where a face was detected is written to
        `{face_crops_dir}/seg_{index:05d}.jpg`, and the path is stored on
        the segment as `face_crop_path`. Useful for displaying visual
        evidence in a UI.
        """
        check_ffmpeg()
        segmenter = VideoAudioSegmenter(
            video_path=video_path,
            segment_seconds=self.cfg.segment_seconds,
            segment_stride_seconds=self.cfg.segment_stride_seconds,
            frames_per_segment=self.cfg.frames_per_segment,
            audio_sr=self.cfg.audio.sample_rate,
        )

        crops_dir: Optional[Path] = None
        if face_crops_dir is not None:
            crops_dir = Path(face_crops_dir)
            crops_dir.mkdir(parents=True, exist_ok=True)

        results: List[Segment] = []
        iterator = iter(segmenter)
        if progress:
            iterator = tqdm(iterator, total=len(segmenter), desc="segments")

        for payload in iterator:
            seg = Segment(
                index=payload.index,
                t_start=payload.t_start,
                t_end=payload.t_end,
            )

            if self.video_det is not None:
                v_prob, n_scored, note, best_crop = self.video_det.score_segment(payload.frames)
                seg.video_fake_prob = v_prob
                seg.video_n_frames_scored = n_scored
                if note:
                    seg.notes = (seg.notes + ";" + note).strip(";")
                if crops_dir is not None and best_crop is not None:
                    try:
                        from PIL import Image

                        out_p = crops_dir / f"seg_{payload.index:05d}.jpg"
                        Image.fromarray(best_crop).save(out_p, format="JPEG", quality=85)
                        seg.face_crop_path = str(out_p)
                    except Exception as e:
                        logger.warning("Failed to save face crop for seg %d: %s", payload.index, e)

            if self.audio_det is not None:
                a_prob, available, note = self.audio_det.score_segment(payload.audio, payload.audio_sr)
                seg.audio_fake_prob = a_prob
                seg.audio_available = available
                if note:
                    seg.notes = (seg.notes + ";" + note).strip(";")

            fused, flag = fuse(seg.video_fake_prob, seg.audio_fake_prob, self.cfg.fusion)
            seg.fused_fake_prob = fused
            seg.tampered = flag
            results.append(seg)

        return self._summarize(str(video_path), results)

    # ------------------------------------------------------------------ #
    def _summarize(self, video_path: str, segs: List[Segment]) -> Dict[str, Any]:
        flagged = [s for s in segs if s.tampered]
        scored = [s for s in segs if s.fused_fake_prob is not None]
        video_level_prob = (
            sum(s.fused_fake_prob for s in scored) / len(scored) if scored else None
        )
        tampered_fraction = len(flagged) / len(segs) if segs else 0.0
        logger.info(
            "Done. %d/%d segments flagged (%.1f%%). Video-level fake prob ≈ %s",
            len(flagged),
            len(segs),
            100 * tampered_fraction,
            f"{video_level_prob:.3f}" if video_level_prob is not None else "N/A",
        )
        return {
            "video": video_path,
            "segment_seconds": self.cfg.segment_seconds,
            "segment_stride_seconds": self.cfg.segment_stride_seconds,
            "threshold": self.cfg.fusion.threshold,
            "device": self.device,
            "segments": [s.to_dict() for s in segs],
            "video_level": {
                "fake_prob": None if video_level_prob is None else round(video_level_prob, 4),
                "tampered_fraction": round(tampered_fraction, 4),
                "n_segments": len(segs),
                "n_flagged": len(flagged),
            },
        }
