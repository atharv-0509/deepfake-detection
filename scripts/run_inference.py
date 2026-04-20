"""CLI: run the segment-level deepfake detector on a video.

Usage:
    python scripts/run_inference.py --video sample.mp4 \
        --config configs/default.yaml --out out/sample.timeline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline import DeepfakePipeline, PipelineConfig  # noqa: E402
from src.utils import (  # noqa: E402
    download_media_from_url,
    is_url,
    normalize_video_for_pipeline,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment-level deepfake detection.")
    p.add_argument(
        "--video", required=True,
        help="Local video file path OR a URL "
             "(YouTube / Instagram / TikTok / Twitter / direct mp4 / mp3).",
    )
    p.add_argument("--config", default="configs/default.yaml", help="YAML config.")
    p.add_argument("--out", default=None, help="Output JSON path.")
    p.add_argument("--device", default=None, help="Override torch device (cpu, cuda, mps).")
    p.add_argument("--crops-dir", default=None,
                   help="If set, dump the best face crop per segment as JPEGs here.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip the ffmpeg pre-normalization step. Only use if "
                        "you know your input is already H.264 with pixels upright.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    # If --video is a URL, fetch it locally first (YouTube / Instagram / TikTok /
    # Twitter / direct mp4 / mp3 all go through yt-dlp).
    local_source = args.video
    if is_url(args.video):
        print(f"Fetching URL: {args.video}")
        local_source = download_media_from_url(args.video)
        print(f"Downloaded -> {local_source}")

    # Normalize codec + rotation up front so the pipeline never sees a
    # weird iPhone / WhatsApp / YouTube file. Skip only when the user opts out.
    video_in = local_source if args.no_normalize else normalize_video_for_pipeline(local_source)

    cfg = PipelineConfig.from_yaml(args.config)
    pipe = DeepfakePipeline(cfg, device=args.device)
    result = pipe.run(
        video_in,
        progress=not args.no_progress,
        face_crops_dir=args.crops_dir,
    )

    # Resolve output path. For URLs, default to out/<id>.timeline.json next to the download.
    if args.out:
        out_path = Path(args.out)
    elif is_url(args.video):
        out_path = Path(local_source).with_suffix(".timeline.json")
    else:
        out_path = Path(args.video).with_suffix(".timeline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote timeline -> {out_path}")
    print(json.dumps(result["video_level"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
