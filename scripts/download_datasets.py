"""Helpers for the standard public benchmarks used to train / evaluate
segment-level deepfake detectors.

These datasets all require EULA acceptance and are NOT downloadable
anonymously, so this script prints the official instructions and verifies
local paths rather than scraping restricted content.

Supported:
  - ff        : FaceForensics++ (face-only visual deepfakes)
  - dfdc      : Deepfake Detection Challenge (video + audio, ~470 GB)
  - asvspoof  : ASVspoof 2019 / 2021 (audio spoofing)
  - lavdf     : LAV-DF / AV-Deepfake1M (segment-level audio-visual, recommended)

Usage:
    python scripts/download_datasets.py ff
    python scripts/download_datasets.py verify --root /data/dfdc --name dfdc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


INSTRUCTIONS = {
    "ff": """
FaceForensics++  (https://github.com/ondyari/FaceForensics)

  1. Submit the access form:
       https://github.com/ondyari/FaceForensics#access
     (non-commercial research only).
  2. The maintainers email you a download script (`faceforensics_download_v4.py`).
  3. Download subsets:
       python faceforensics_download_v4.py <OUT_DIR> -d all -c c23 -t videos
     (c23 = web-quality compression; smallest of the useful settings).
  4. Frame-level labels: FaceForensics++ is video-level only; treat the
     whole manipulated video as fake for training. For SEGMENT-level
     localization, use LAV-DF or AV-Deepfake1M instead.
""".strip(),

    "dfdc": """
DFDC — Deepfake Detection Challenge  (https://ai.meta.com/datasets/dfdc/)

  1. Register with a Facebook AI / Meta AI account:
       https://ai.meta.com/datasets/dfdc/
  2. Accept the license; download links are emailed.
  3. The full training set is ~470 GB across 50 parts. The preview
     (~5 GB) is enough to smoke-test this pipeline:
       https://dfdc.ai
  4. Per-video labels are in `metadata.json`. Audio is included.

Expected layout:
  DFDC_ROOT/
    dfdc_train_part_00/
      *.mp4
      metadata.json
    ...
""".strip(),

    "asvspoof": """
ASVspoof 2019 / 2021  (https://www.asvspoof.org/)

  - 2019 LA: https://datashare.ed.ac.uk/handle/10283/3336  (logical access, TTS/VC)
  - 2019 PA: https://datashare.ed.ac.uk/handle/10283/3055  (physical access, replay)
  - 2021   : https://www.asvspoof.org/index2021.html

Each partition ships a protocols file with bonafide/spoof labels.
Use for the AUDIO branch only; these are audio-only recordings.
""".strip(),

    "lavdf": """
LAV-DF  (https://github.com/ControlNet/LAV-DF)
AV-Deepfake1M  (https://github.com/ControlNet/AV-Deepfake1M)

  These are the datasets to use for SEGMENT-LEVEL localization:
  each video ships with frame-level / second-level fake spans.
  Request access via the repo README, then:

    git clone https://github.com/ControlNet/LAV-DF
    cd LAV-DF && python download.py --root /data/lavdf

  metadata JSON gives:
    {"file": "...mp4", "n_fakes": 2,
     "fake_periods": [[start_sec, end_sec], ...]}

  Use `scripts/evaluate.py --dataset lavdf` to score against these spans.
""".strip(),
}


def print_instructions(name: str) -> int:
    if name not in INSTRUCTIONS:
        print(f"Unknown dataset: {name}. Known: {sorted(INSTRUCTIONS)}", file=sys.stderr)
        return 2
    print(INSTRUCTIONS[name])
    return 0


def verify_root(root: Path, name: str) -> int:
    if not root.exists():
        print(f"{root} does not exist.", file=sys.stderr)
        return 1
    expected = {
        "ff": ["original_sequences", "manipulated_sequences"],
        "dfdc": ["metadata.json"],       # per-part
        "asvspoof": ["protocols"],
        "lavdf": ["metadata.json"],
    }.get(name, [])
    missing = [e for e in expected if not any(root.rglob(e))]
    if missing:
        print(f"Missing expected entries under {root}: {missing}", file=sys.stderr)
        return 1
    print(f"OK: {root} looks like a {name} dataset root.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    for key in INSTRUCTIONS:
        sp = sub.add_parser(key, help=f"print instructions for '{key}'")
        sp.set_defaults(cmd=key)

    verify = sub.add_parser("verify", help="verify a local dataset root")
    verify.add_argument("--root", required=True)
    verify.add_argument("--name", required=True, choices=list(INSTRUCTIONS))

    args = p.parse_args()
    if args.cmd == "verify":
        return verify_root(Path(args.root), args.name)
    if args.cmd in INSTRUCTIONS:
        return print_instructions(args.cmd)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
