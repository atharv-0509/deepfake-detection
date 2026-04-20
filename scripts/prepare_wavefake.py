#!/usr/bin/env python
"""Build train/dev/test manifest CSVs from a WaveFake + LJSpeech tree.

WaveFake (Frank & Schönherr, 2021) contains ~100k synthesized utterances
from 7+ modern neural vocoders (MelGAN family, Parallel WaveGAN, HiFi-GAN,
WaveGlow, ...) based on LJSpeech / JSUT base speech. The real audio is NOT
in the WaveFake release itself — you grab LJSpeech-1.1 separately.

Expected layout (adapt --wavefake-root / --ljspeech-root as needed):

    <wavefake_root>/
    ├── ljspeech_melgan/*.wav
    ├── ljspeech_hifiGAN/*.wav
    ├── ljspeech_parallel_wavegan/*.wav
    ├── ljspeech_full_band_melgan/*.wav
    ├── ljspeech_multi_band_melgan/*.wav
    ├── ljspeech_melgan_large/*.wav
    ├── ljspeech_waveglow/*.wav
    └── (optionally) jsut_*                 # Japanese, off by default

    <ljspeech_root>/
    ├── wavs/LJ###-####.wav                 # 13,100 real clips
    └── metadata.csv

Output CSVs (same schema as prepare_asvspoof.py so train_audio.py / the
ASVspoofDataset class Just Works):

    file_path,label,spoof_type,speaker_id,split

Usage:
    python scripts/prepare_wavefake.py \
        --wavefake-root /kaggle/input/wavefake \
        --ljspeech-root /kaggle/input/ljspeech-dataset/LJSpeech-1.1 \
        --out-dir data/wavefake \
        --per-vocoder-cap 2000        # optional — subsample for speed

Split strategy: 80/10/10 stratified by (label, spoof_type).
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_wavefake")


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def find_audio_files(dirpath: Path) -> list[Path]:
    """Recursively list audio files under dirpath."""
    if not dirpath.is_dir():
        return []
    out: list[Path] = []
    for p in dirpath.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out


def discover_vocoder_dirs(wavefake_root: Path, include_jsut: bool) -> dict[str, Path]:
    """Find subdirectories that look like WaveFake vocoder splits.

    Handles flat layouts (<root>/ljspeech_hifiGAN/*.wav), common
    Kaggle-mirror nesting (<root>/generated_audio/ljspeech_hifiGAN/*.wav),
    and non-standard names like
    `common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech/` (a
    newer Conformer+FastSpeech2+PWG attack present in some mirrors).

    Matches any subdirectory whose name contains 'ljspeech' (always) or
    'jsut' (if include_jsut=True).
    """
    candidates: dict[str, Path] = {}
    for parent in [wavefake_root] + [p for p in wavefake_root.iterdir() if p.is_dir()]:
        if not parent.is_dir():
            continue
        for sub in parent.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if "ljspeech" in name or (include_jsut and "jsut" in name):
                candidates.setdefault(sub.name, sub)
    return candidates


def stratified_split(
    rows: list[dict],
    strat_key: str,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split rows into train/dev/test, stratified by rows[i][strat_key]."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r[strat_key]].append(r)

    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * fractions[0])
        n_dev = int(n * fractions[1])
        out["train"].extend(items[:n_train])
        out["dev"].extend(items[n_train : n_train + n_dev])
        out["test"].extend(items[n_train + n_dev :])
    # Re-shuffle final splits so downstream batching isn't biased by vocoder.
    for split in out.values():
        rng.shuffle(split)
    return out


def build_fake_rows(
    vocoder_dirs: dict[str, Path],
    per_vocoder_cap: int | None,
    seed: int,
) -> list[dict]:
    """Return one row per fake audio file, stratified-capped if requested."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for voc_name, voc_dir in sorted(vocoder_dirs.items()):
        files = find_audio_files(voc_dir)
        if not files:
            log.warning("no audio found under %s — skipping", voc_dir)
            continue
        if per_vocoder_cap is not None and len(files) > per_vocoder_cap:
            rng.shuffle(files)
            files = files[:per_vocoder_cap]
        log.info("  %-35s  %d files", voc_name, len(files))
        for f in files:
            rows.append(
                {
                    "file_path": str(f.resolve()),
                    "label": 1,
                    "spoof_type": voc_name,
                    # Parse speaker from filename if LJSpeech-style (LJ001-0001_gen.wav)
                    "speaker_id": "LJSpeech" if "ljspeech" in voc_name.lower()
                    else "JSUT" if "jsut" in voc_name.lower()
                    else voc_name,
                }
            )
    return rows


def build_real_rows(
    ljspeech_root: Path,
    cap: int | None,
    seed: int,
) -> list[dict]:
    """Return one row per real LJSpeech file."""
    rng = random.Random(seed)
    # Canonical LJSpeech-1.1 has wavs/ as the audio directory.
    wavs_dir = ljspeech_root / "wavs"
    if not wavs_dir.is_dir():
        # Some mirrors drop files at root.
        wavs_dir = ljspeech_root
    files = find_audio_files(wavs_dir)
    if not files:
        raise FileNotFoundError(
            f"No audio found under {wavs_dir}. "
            "Did you point --ljspeech-root at the LJSpeech-1.1 directory?"
        )
    if cap is not None and len(files) > cap:
        rng.shuffle(files)
        files = files[:cap]
    log.info("  LJSpeech real                        %d files", len(files))
    return [
        {
            "file_path": str(f.resolve()),
            "label": 0,
            "spoof_type": "bonafide",
            "speaker_id": "LJSpeech",
        }
        for f in files
    ]


def write_split_csv(rows: Iterable[dict], split_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wavefake_{split_name}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_path", "label", "spoof_type", "speaker_id", "split"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "split": split_name})
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--wavefake-root",
        required=True,
        type=Path,
        help="Path to the WaveFake generated_audio root (contains "
        "ljspeech_hifiGAN/, ljspeech_melgan/, ...).",
    )
    ap.add_argument(
        "--ljspeech-root",
        required=True,
        type=Path,
        help="Path to LJSpeech-1.1 (contains wavs/ and metadata.csv).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/wavefake"),
    )
    ap.add_argument(
        "--per-vocoder-cap",
        type=int,
        default=None,
        help="Max fake clips per vocoder (for faster training). Default: use all.",
    )
    ap.add_argument(
        "--real-cap",
        type=int,
        default=None,
        help="Max real clips. Default: use all of LJSpeech.",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="Cap real clips to match total fake count (1:1 balance).",
    )
    ap.add_argument(
        "--include-jsut",
        action="store_true",
        help="Also include JSUT (Japanese) vocoder subdirs. Off by default.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Scan the layout and print counts without writing CSVs.",
    )
    args = ap.parse_args()

    wavefake_root = args.wavefake_root.expanduser().resolve()
    ljspeech_root = args.ljspeech_root.expanduser().resolve()

    if not wavefake_root.is_dir():
        log.error("wavefake root not found: %s", wavefake_root)
        return 2
    if not ljspeech_root.is_dir():
        log.error("ljspeech root not found: %s", ljspeech_root)
        return 2

    log.info("scanning WaveFake root: %s", wavefake_root)
    vocoder_dirs = discover_vocoder_dirs(wavefake_root, include_jsut=args.include_jsut)
    if not vocoder_dirs:
        log.error(
            "Found no ljspeech_*/ vocoder subdirs under %s. "
            "Check the layout — the Kaggle mirror sometimes nests under "
            "generated_audio/.",
            wavefake_root,
        )
        return 2

    log.info("found %d vocoder splits:", len(vocoder_dirs))
    fake_rows = build_fake_rows(vocoder_dirs, args.per_vocoder_cap, args.seed)
    log.info("total fake rows: %d", len(fake_rows))

    real_cap = args.real_cap
    if args.balance:
        real_cap = len(fake_rows) if real_cap is None else min(real_cap, len(fake_rows))
    log.info("scanning LJSpeech root: %s", ljspeech_root)
    real_rows = build_real_rows(ljspeech_root, real_cap, args.seed)
    log.info("total real rows: %d", len(real_rows))

    all_rows = fake_rows + real_rows
    log.info("total rows: %d (fake=%d, real=%d)",
             len(all_rows), len(fake_rows), len(real_rows))

    # Stratify on spoof_type so every vocoder appears in every split.
    splits = stratified_split(all_rows, strat_key="spoof_type", seed=args.seed)
    for name, rows in splits.items():
        n_fake = sum(1 for r in rows if r["label"] == 1)
        n_real = len(rows) - n_fake
        log.info("  %-5s  total=%d  fake=%d  real=%d", name, len(rows), n_fake, n_real)

    if args.check_only:
        log.info("check-only: no CSVs written.")
        return 0

    for name in ("train", "dev", "test"):
        out_path = write_split_csv(splits[name], name, args.out_dir)
        log.info("  → wrote %s", out_path)

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
