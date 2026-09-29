#!/usr/bin/env python
"""Synthesize *partially* spoofed audio + a segment-level manifest.

The whole-file datasets (ASVspoof, WaveFake) label each clip as entirely
bonafide or entirely spoof, so a model trained on them only learns an
utterance-level decision and flags a tampered clip end-to-end. This script
turns an existing whole-file manifest into *partial-spoof* training data:
it splices bonafide and spoof audio into single clips and records exactly
which time intervals are fake, so downstream training can learn segment-level
boundaries.

Input: a manifest CSV with at least `file_path` and `label` columns
(0 = bonafide/real, 1 = spoof/fake) — i.e. the output of
scripts/prepare_asvspoof.py or scripts/prepare_wavefake.py.

Output:
  <out-dir>/audio/mix_*.wav        synthesized partial-spoof clips
  <out-dir>/<name>.csv             segment-level manifest with columns:
        file_path,duration,fake_segments,speaker_id,source

`fake_segments` is a ";"-separated list of "start-end" seconds (empty for a
fully-bonafide row). By default the manifest ALSO includes pass-through rows
for a sample of the original bonafide and spoof files (referenced in place, no
copy) so the model still sees clean fully-real and fully-fake examples
alongside the spliced ones.

Typical Kaggle usage (after prepare_asvspoof.py / prepare_wavefake.py):

    python scripts/make_partialspoof.py \
        --manifest data/asvspoof/asvspoof_train.csv \
        --out-dir data/partialspoof/train \
        --name partial_train \
        --n-partial 6000 \
        --seed 42

    python scripts/make_partialspoof.py \
        --manifest data/asvspoof/asvspoof_dev.csv \
        --out-dir data/partialspoof/dev \
        --name partial_dev \
        --n-partial 1000 \
        --seed 7
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.training.segment_labeling import fmt_segments, merge_intervals  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("make_partialspoof")


# --------------------------------------------------------------------------- #
# Manifest IO
# --------------------------------------------------------------------------- #
def load_pool(manifest: Path) -> Tuple[List[dict], List[dict]]:
    """Split a whole-file manifest into (bonafide_rows, spoof_rows)."""
    bona: List[dict] = []
    spoof: List[dict] = []
    with manifest.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "file_path" not in reader.fieldnames \
                or "label" not in reader.fieldnames:
            raise ValueError(
                f"{manifest}: expected columns 'file_path' and 'label'; "
                f"got {reader.fieldnames}"
            )
        for r in reader:
            try:
                lab = int(r["label"])
            except (ValueError, TypeError):
                continue
            (bona if lab == 0 else spoof).append(r)
    return bona, spoof


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def load_audio(path: str, sr: int) -> Optional[np.ndarray]:
    import soundfile as sf

    try:
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:  # unreadable / corrupt clip — skip it
        log.debug("skip unreadable %s: %s", path, e)
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio.astype(np.float32)


def take_chunk(
    audio: np.ndarray, sr: int, seconds: float, rng: random.Random
) -> np.ndarray:
    """Random-crop (or pad) a chunk of `seconds` from `audio`."""
    want = max(1, int(round(seconds * sr)))
    n = len(audio)
    if n <= want:
        out = np.zeros(want, dtype=np.float32)
        out[:n] = audio
        return out
    start = rng.randint(0, n - want)
    return audio[start : start + want]


def crossfade_concat(chunks: List[np.ndarray], fade_samples: int) -> np.ndarray:
    """Concatenate chunks with a short linear crossfade to avoid clicks."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    out = chunks[0].astype(np.float32).copy()
    for nxt in chunks[1:]:
        nxt = nxt.astype(np.float32)
        f = min(fade_samples, len(out), len(nxt))
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            out[-f:] = out[-f:] * (1.0 - ramp) + nxt[:f] * ramp
            out = np.concatenate([out, nxt[f:]])
        else:
            out = np.concatenate([out, nxt])
    return out


# --------------------------------------------------------------------------- #
# Splice one partial-spoof clip
# --------------------------------------------------------------------------- #
def build_partial_clip(
    bona: List[dict],
    spoof: List[dict],
    sr: int,
    rng: random.Random,
    min_chunk: float,
    max_chunk: float,
    min_segments: int,
    max_segments: int,
    fade_ms: float,
    force_mixed: bool = True,
) -> Optional[Tuple[np.ndarray, List[Tuple[float, float]]]]:
    """Return (waveform, fake_intervals_seconds) or None if audio unreadable.

    Builds a timeline of alternating bonafide/spoof chunks. Boundaries are
    recorded *after* accounting for crossfade shortening so the fake intervals
    line up with the rendered waveform.
    """
    n_seg = rng.randint(min_segments, max_segments)
    fade_samples = int(round(fade_ms / 1000.0 * sr))

    # Decide a fake/real pattern. Guarantee at least one of each when
    # force_mixed so every synthesized clip is genuinely partial.
    is_fake = [rng.random() < 0.5 for _ in range(n_seg)]
    if force_mixed:
        if not any(is_fake):
            is_fake[rng.randrange(n_seg)] = True
        if all(is_fake):
            is_fake[rng.randrange(n_seg)] = False

    chunks: List[np.ndarray] = []
    chunk_is_fake: List[bool] = []
    for want_fake in is_fake:
        pool = spoof if want_fake else bona
        if not pool:
            pool = bona if want_fake else spoof  # fall back if one side empty
            want_fake = not want_fake
        audio = None
        for _ in range(5):  # a few retries against unreadable files
            row = rng.choice(pool)
            audio = load_audio(row["file_path"], sr)
            if audio is not None and len(audio) >= int(0.1 * sr):
                break
            audio = None
        if audio is None:
            continue
        seconds = rng.uniform(min_chunk, max_chunk)
        chunks.append(take_chunk(audio, sr, seconds, rng))
        chunk_is_fake.append(want_fake)

    if len(chunks) < 2:
        return None

    wave = crossfade_concat(chunks, fade_samples)

    # Recompute interval boundaries on the *rendered* timeline. Each crossfade
    # overlaps `f` samples, shortening the total by that much per join.
    fakes: List[Tuple[float, float]] = []
    cursor = 0
    for i, ch in enumerate(chunks):
        f = min(fade_samples if i > 0 else 0, len(ch))
        start = cursor
        if chunk_is_fake[i]:
            fakes.append((start / sr, (start + len(ch)) / sr))
        cursor = start + len(ch) - f

    total = len(wave)
    # Clamp intervals to the rendered length and merge adjacent fakes.
    fakes = [(max(0.0, a), min(total / sr, b)) for a, b in fakes if b > a]
    fakes = merge_intervals(fakes)
    return wave, fakes


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, type=Path,
                    help="Whole-file manifest CSV (file_path,label,...).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory (audio/ + manifest CSV go here).")
    ap.add_argument("--name", default=None,
                    help="Manifest basename (default: derived from --manifest).")
    ap.add_argument("--n-partial", type=int, default=4000,
                    help="Number of spliced partial-spoof clips to synthesize.")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    ap.add_argument("--min-chunk", type=float, default=0.6,
                    help="Min seconds per spliced chunk.")
    ap.add_argument("--max-chunk", type=float, default=2.0,
                    help="Max seconds per spliced chunk.")
    ap.add_argument("--min-segments", type=int, default=2,
                    help="Min chunks per synthesized clip.")
    ap.add_argument("--max-segments", type=int, default=4,
                    help="Max chunks per synthesized clip.")
    ap.add_argument("--fade-ms", type=float, default=10.0,
                    help="Crossfade length between chunks (milliseconds).")
    ap.add_argument("--passthrough-real", type=int, default=2000,
                    help="How many original bonafide files to add as fully-real "
                         "rows (0 to disable). Referenced in place, not copied.")
    ap.add_argument("--passthrough-fake", type=int, default=2000,
                    help="How many original spoof files to add as fully-fake "
                         "rows (0 to disable). Referenced in place, not copied.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.manifest.exists():
        log.error("manifest not found: %s", args.manifest)
        return 2

    bona, spoof = load_pool(args.manifest)
    log.info("pool: %d bonafide, %d spoof", len(bona), len(spoof))
    if not bona or not spoof:
        log.error("Need both bonafide and spoof rows to splice partial clips.")
        return 2

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or (args.manifest.stem + "_partial")
    out_csv = args.out_dir / f"{name}.csv"

    import soundfile as sf

    rows: List[dict] = []

    # -- 1. Synthesize partial-spoof clips --------------------------------- #
    made = 0
    attempts = 0
    max_attempts = args.n_partial * 3 + 50
    while made < args.n_partial and attempts < max_attempts:
        attempts += 1
        built = build_partial_clip(
            bona, spoof, args.sr, rng,
            min_chunk=args.min_chunk, max_chunk=args.max_chunk,
            min_segments=args.min_segments, max_segments=args.max_segments,
            fade_ms=args.fade_ms,
        )
        if built is None:
            continue
        wave, fakes = built
        out_path = audio_dir / f"mix_{made:06d}.wav"
        sf.write(str(out_path), wave, args.sr, subtype="PCM_16")
        rows.append({
            "file_path": str(out_path.resolve()),
            "duration": f"{len(wave) / args.sr:.3f}",
            "fake_segments": fmt_segments(fakes),
            "speaker_id": "mixed",
            "source": "partial",
        })
        made += 1
        if made % 500 == 0:
            log.info("  synthesized %d/%d partial clips", made, args.n_partial)
    log.info("synthesized %d partial clips (%d attempts)", made, attempts)

    # -- 2. Pass-through originals (fully real / fully fake) --------------- #
    def add_passthrough(pool: List[dict], k: int, fake: bool) -> int:
        if k <= 0 or not pool:
            return 0
        pick = pool if len(pool) <= k else rng.sample(pool, k)
        added = 0
        for r in pick:
            audio = load_audio(r["file_path"], args.sr)
            if audio is None or len(audio) < int(0.1 * args.sr):
                continue
            dur = len(audio) / args.sr
            rows.append({
                "file_path": r["file_path"],
                "duration": f"{dur:.3f}",
                "fake_segments": f"0.000-{dur:.3f}" if fake else "",
                "speaker_id": r.get("speaker_id", "") or "",
                "source": "spoof" if fake else "bonafide",
            })
            added += 1
        return added

    n_real = add_passthrough(bona, args.passthrough_real, fake=False)
    n_fake = add_passthrough(spoof, args.passthrough_fake, fake=True)
    log.info("added pass-through: %d real, %d fake", n_real, n_fake)

    # -- 3. Write manifest ------------------------------------------------- #
    rng.shuffle(rows)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_path", "duration", "fake_segments",
                        "speaker_id", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info("wrote %s (%d rows: %d partial, %d real, %d fake)",
             out_csv, len(rows), made, n_real, n_fake)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
