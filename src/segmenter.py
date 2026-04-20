"""Split an input video into fixed-size segments and iterate over
(frames, audio) pairs for each segment.

- Video frames: decoded lazily with PyAV; down-sampled to `frames_per_segment`
  per window so scoring cost stays bounded regardless of source fps.
- Audio: loaded once as a numpy array (mono, configured sample rate) and
  sliced per segment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .utils import (
    extract_audio_wav,
    logger,
    probe_duration,
    probe_rotation_degrees,
    temp_workdir,
)


@dataclass
class SegmentPayload:
    index: int
    t_start: float
    t_end: float
    frames: List[np.ndarray]       # list of HxWx3 uint8 RGB arrays (may be empty)
    audio: Optional[np.ndarray]    # 1-D float32 PCM or None
    audio_sr: int


def _uniform_indices(n_available: int, n_wanted: int) -> List[int]:
    if n_available <= 0 or n_wanted <= 0:
        return []
    if n_available <= n_wanted:
        return list(range(n_available))
    step = n_available / n_wanted
    return [int(i * step) for i in range(n_wanted)]


class VideoAudioSegmenter:
    """Yields `SegmentPayload`s covering the whole input video.

    Parameters
    ----------
    video_path : path to the input video.
    segment_seconds : window size in seconds.
    segment_stride_seconds : hop in seconds (defaults to segment_seconds).
    frames_per_segment : how many evenly-spaced frames to keep per window
        for the visual detector.
    audio_sr : target sample rate for the mono audio waveform.
    """

    def __init__(
        self,
        video_path: str | Path,
        segment_seconds: float = 1.0,
        segment_stride_seconds: Optional[float] = None,
        frames_per_segment: int = 5,
        audio_sr: int = 16000,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self.segment_seconds = float(segment_seconds)
        self.segment_stride_seconds = float(segment_stride_seconds or segment_seconds)
        self.frames_per_segment = int(frames_per_segment)
        self.audio_sr = int(audio_sr)

        self.duration = probe_duration(self.video_path)
        self._workdir = temp_workdir()
        self._audio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Audio loading
    # ------------------------------------------------------------------ #
    def _load_audio(self) -> Optional[np.ndarray]:
        if self._audio is not None:
            return self._audio
        try:
            wav_path = self._workdir / "audio.wav"
            extract_audio_wav(self.video_path, wav_path, sample_rate=self.audio_sr)
            import soundfile as sf

            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if sr != self.audio_sr:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=self.audio_sr)
            self._audio = np.asarray(data, dtype=np.float32)
            return self._audio
        except Exception as e:
            logger.warning("Audio extraction failed: %s (continuing video-only).", e)
            self._audio = None
            return None

    def _audio_slice(self, t_start: float, t_end: float) -> Optional[np.ndarray]:
        audio = self._load_audio()
        if audio is None:
            return None
        i0 = int(round(t_start * self.audio_sr))
        i1 = int(round(t_end * self.audio_sr))
        i1 = min(i1, len(audio))
        i0 = min(i0, i1)
        return audio[i0:i1] if i1 > i0 else None

    # ------------------------------------------------------------------ #
    # Frame iteration using PyAV
    # ------------------------------------------------------------------ #
    def _iter_frames_in_window(
        self, t_start: float, t_end: float, want: int
    ) -> List[np.ndarray]:
        """Seek into the container and return up to `want` evenly-spaced frames."""
        try:
            import av  # PyAV
        except ImportError as e:
            raise RuntimeError(
                "PyAV is required for video decoding. Install with `pip install av`."
            ) from e

        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base) if stream.time_base else 1 / 25

            # Phone/WhatsApp videos store orientation as metadata rather than
            # baking rotation into the pixels. PyAV does NOT auto-apply this,
            # so we read it once and rotate each frame post-decode. Rotation
            # may live in legacy stream tags (older iPhones/Android) OR in
            # side-data `display_matrix` (iPhone 14+, newer WhatsApp/Android).
            # We delegate to ffprobe via probe_rotation_degrees() so both
            # locations are handled. Without this, portrait phone clips decode
            # upside down and face classifiers flag every frame as "fake".
            rotate_k = 0
            try:
                rot_str = stream.metadata.get("rotate")
                deg = int(rot_str) % 360 if rot_str is not None else 0
            except (ValueError, TypeError):
                deg = 0
            if deg == 0:
                # Fall back to ffprobe, which also reads side-data.
                try:
                    deg = probe_rotation_degrees(self.video_path)
                except Exception:
                    deg = 0
            if deg:
                # Clockwise degrees; np.rot90 is CCW, so negate.
                rotate_k = (-deg // 90) % 4
                logger.info("Detected rotation %d deg -> np.rot90 k=%d", deg, rotate_k)
            # Seek to slightly before t_start to get the keyframe.
            seek_pts = int(max(0.0, t_start - 0.2) / time_base)
            try:
                container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
            except av.AVError:
                container.seek(0)

            # Target timestamps, evenly spaced.
            if want <= 0:
                return []
            targets = [
                t_start + (i + 0.5) * (t_end - t_start) / want for i in range(want)
            ]
            target_idx = 0
            frames: List[np.ndarray] = []

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * time_base)
                if t < t_start:
                    continue
                if t >= t_end:
                    break
                if target_idx < len(targets) and t >= targets[target_idx]:
                    arr = frame.to_ndarray(format="rgb24")
                    if rotate_k:
                        arr = np.rot90(arr, k=rotate_k).copy()
                    frames.append(arr)
                    target_idx += 1
                if len(frames) >= want:
                    break

            # Fallback: if nothing collected (short segment), grab anything.
            if not frames:
                container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * time_base)
                    if t >= t_end:
                        break
                    if t >= t_start:
                        arr = frame.to_ndarray(format="rgb24")
                        if rotate_k:
                            arr = np.rot90(arr, k=rotate_k).copy()
                        frames.append(arr)
                        if len(frames) >= want:
                            break
            return frames

    # ------------------------------------------------------------------ #
    # Public iterator
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[SegmentPayload]:
        n_segments = max(
            1,
            int(math.floor((self.duration - 1e-6) / self.segment_stride_seconds)) + 1,
        )
        for i in range(n_segments):
            t_start = i * self.segment_stride_seconds
            t_end = min(t_start + self.segment_seconds, self.duration)
            if t_end - t_start < 0.1:  # skip tiny tail
                continue
            frames = self._iter_frames_in_window(t_start, t_end, self.frames_per_segment)
            audio = self._audio_slice(t_start, t_end)
            yield SegmentPayload(
                index=i,
                t_start=t_start,
                t_end=t_end,
                frames=frames,
                audio=audio,
                audio_sr=self.audio_sr,
            )

    def __len__(self) -> int:
        return max(
            1,
            int(math.floor((self.duration - 1e-6) / self.segment_stride_seconds)) + 1,
        )
