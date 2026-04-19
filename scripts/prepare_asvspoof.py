#!/usr/bin/env python
"""Build train/dev/eval manifest CSVs from an ASVspoof 2019 LA dataset tree.

Expected layout (the canonical one shipped by Datashare):

    <root>/
    ├── ASVspoof2019_LA_train/flac/LA_T_*.flac
    ├── ASVspoof2019_LA_dev/flac/LA_D_*.flac
    ├── ASVspoof2019_LA_eval/flac/LA_E_*.flac
    └── ASVspoof2019_LA_cm_protocols/
        ├── ASVspoof2019.LA.cm.train.trn.txt
        ├── ASVspoof2019.LA.cm.dev.trl.txt
        └── ASVspoof2019.LA.cm.eval.trl.txt

Each protocol line has 5 whitespace-separated columns:

    SPEAKER_ID  FILE_NAME  -  SYSTEM_ID  KEY

where column 3 is always "-" (a padding/environment column, unused in LA)
and SYSTEM_ID is "-" for bonafide and A01..A19 for spoof. Examples:

    LA_0079 LA_T_1138215 - - bonafide
    LA_0079 LA_T_1235126 - A06 spoof

This script writes CSVs with one row per utterance:

    file_path,label,spoof_type,speaker_id,split

`label` is 0 (bonafide) or 1 (spoof) so downstream code doesn't have to
remember the convention.

Usage:
    python scripts/prepare_asvspoof.py \
        --root /data/ASVspoof2019_LA \
        --out-dir data/asvspoof

Run `--check-only` to validate your layout without writing CSVs.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_asvspoof")


SPLIT_CONFIG = {
    "train": {
        "audio_dir": "ASVspoof2019_LA_train/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    },
    "dev": {
        "audio_dir": "ASVspoof2019_LA_dev/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    },
    "eval": {
        "audio_dir": "ASVspoof2019_LA_eval/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    },
}


def parse_protocol(path: Path) -> Iterable[dict]:
    """Yield one dict per line of an ASVspoof 2019 CM protocol file."""
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"{path}:{lineno}: expected 5 whitespace-separated fields, "
                    f"got {len(parts)}: {raw!r}"
                )
            # ASVspoof 2019 LA protocol columns:
            #   0: speaker_id    (LA_####)
            #   1: file_id       (LA_{T,D,E}_#######)
            #   2: "-"           (padding; environment col, unused in LA)
            #   3: system_id     ("-" for bonafide, A01..A19 for spoof)
            #   4: label         ("bonafide" or "spoof")
            speaker_id, file_id, _pad, system_id, label = parts
            if label not in ("bonafide", "spoof"):
                raise ValueError(
                    f"{path}:{lineno}: unexpected label {label!r}; "
                    "expected 'bonafide' or 'spoof'"
                )
            # Sanity-check: for spoof rows system_id should look like A##.
            if label == "spoof" and not (
                len(system_id) == 3 and system_id.startswith("A") and system_id[1:].isdigit()
            ):
                raise ValueError(
                    f"{path}:{lineno}: spoof row has unexpected system_id "
                    f"{system_id!r}; expected 'A01'..'A19'. Full line: {raw!r}"
                )
            yield {
                "file_id": file_id,
                "speaker_id": speaker_id,
                "spoof_type": system_id if label == "spoof" else "bonafide",
                "label_str": label,
                "label": 1 if label == "spoof" else 0,
            }


def build_split(root: Path, split: str, out_dir: Path, check_only: bool) -> dict:
    cfg = SPLIT_CONFIG[split]
    audio_dir = root / cfg["audio_dir"]
    protocol = root / cfg["protocol"]

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Missing audio dir: {audio_dir}")
    if not protocol.is_file():
        raise FileNotFoundError(f"Missing protocol file: {protocol}")

    rows = []
    missing = 0
    n_bonafide = 0
    n_spoof = 0

    for entry in parse_protocol(protocol):
        flac = audio_dir / f"{entry['file_id']}.flac"
        if not flac.is_file():
            missing += 1
            if missing <= 5:
                log.warning("missing audio: %s", flac)
            continue
        rows.append(
            {
                "file_path": str(flac.resolve()),
                "label": entry["label"],
                "spoof_type": entry["spoof_type"],
                "speaker_id": entry["speaker_id"],
                "split": split,
            }
        )
        if entry["label"] == 0:
            n_bonafide += 1
        else:
            n_spoof += 1

    if missing > 5:
        log.warning("... and %d more missing files in %s", missing - 5, split)

    stats = {
        "split": split,
        "total": len(rows),
        "bonafide": n_bonafide,
        "spoof": n_spoof,
        "missing": missing,
    }
    log.info(
        "%s: %d utterances (%d bonafide, %d spoof) — %d missing",
        split,
        stats["total"],
        stats["bonafide"],
        stats["spoof"],
        stats["missing"],
    )

    if not check_only and rows:
        out_path = out_dir / f"asvspoof_{split}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file_path", "label", "spoof_type", "speaker_id", "split"],
            )
            writer.writeheader()
            writer.writerows(rows)
        log.info("  → wrote %s", out_path)

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the ASVspoof2019_LA directory (the one containing "
        "ASVspoof2019_LA_train, ..._dev, ..._eval, ..._cm_protocols).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/asvspoof"),
        help="Where to write the CSV manifests (default: data/asvspoof/).",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "eval"],
        choices=["train", "dev", "eval"],
        help="Which splits to build (default: all three).",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the layout and print counts, but don't write CSVs.",
    )
    args = ap.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        log.error("root directory does not exist: %s", root)
        return 2

    log.info("scanning %s", root)

    all_stats = []
    for split in args.splits:
        try:
            all_stats.append(build_split(root, split, args.out_dir, args.check_only))
        except FileNotFoundError as e:
            log.error("%s", e)
            log.error(
                "Your ASVspoof tree doesn't match the expected layout. "
                "See the docstring at the top of this script."
            )
            return 2

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
