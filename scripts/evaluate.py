"""Evaluate the pipeline against a dataset with segment-level or video-level labels.

Currently supports two dataset adaptors:
  - lavdf : segment-level (fake_periods) — reports segment F1 and AP@IoU.
  - dfdc  : video-level label ('FAKE'/'REAL') — reports video-level ROC-AUC.

Usage:
    python scripts/evaluate.py --dataset lavdf \
        --dataset-root /data/lavdf --config configs/default.yaml --out out/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline import DeepfakePipeline, PipelineConfig  # noqa: E402
from src.utils import setup_logging, logger  # noqa: E402


# --------------------------------------------------------------------------- #
# Dataset adaptors
# --------------------------------------------------------------------------- #
def _load_lavdf_index(root: Path) -> List[Dict]:
    """Return list of {file, fake_periods} dicts from LAV-DF metadata."""
    meta = root / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError(meta)
    with open(meta) as f:
        rows = json.load(f)
    out = []
    for r in rows:
        out.append(
            {
                "file": str(root / r.get("file", r.get("filename"))),
                "fake_periods": r.get("fake_periods", []),
            }
        )
    return out


def _load_dfdc_index(root: Path) -> List[Dict]:
    """Scan all dfdc_train_part_*/metadata.json files."""
    rows = []
    for meta in root.rglob("metadata.json"):
        with open(meta) as f:
            d = json.load(f)
        for fname, info in d.items():
            rows.append(
                {
                    "file": str(meta.parent / fname),
                    "label": info.get("label", "").upper(),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Segment-level labeling
# --------------------------------------------------------------------------- #
def _seg_labels_from_periods(
    segments: List[Dict], fake_periods: List[List[float]]
) -> List[int]:
    labels = []
    for s in segments:
        t0, t1 = s["t_start"], s["t_end"]
        overlap = 0.0
        for a, b in fake_periods:
            overlap += max(0.0, min(t1, b) - max(t0, a))
        labels.append(1 if overlap / max(1e-9, t1 - t0) >= 0.5 else 0)
    return labels


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _segment_metrics(y_true: List[int], y_score: List[float], threshold: float) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return {
        "n": len(y_true),
        "pos_rate": sum(y_true) / max(1, len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "f1_at_threshold": float(f1_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
        "threshold": threshold,
    }


def _video_level_metrics(rows: List[Dict]) -> Dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = [r["label"] for r in rows]
    p = [r["score"] for r in rows]
    return {
        "n": len(y),
        "pos_rate": sum(y) / max(1, len(y)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
    }


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["lavdf", "dfdc"])
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="out/eval.json")
    p.add_argument("--limit", type=int, default=None, help="Cap #videos for quick runs.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    cfg = PipelineConfig.from_yaml(args.config)
    pipe = DeepfakePipeline(cfg, device=args.device)

    root = Path(args.dataset_root)
    if args.dataset == "lavdf":
        index = _load_lavdf_index(root)
    else:
        index = _load_dfdc_index(root)
    if args.limit:
        index = index[: args.limit]

    all_true, all_score = [], []
    video_rows = []

    for row in index:
        try:
            result = pipe.run(row["file"], progress=False)
        except Exception as e:
            logger.warning("skip %s (%s)", row["file"], e)
            continue

        segs = result["segments"]
        scores = [s["fused_fake_prob"] for s in segs if s["fused_fake_prob"] is not None]
        valid = [s for s in segs if s["fused_fake_prob"] is not None]

        if args.dataset == "lavdf":
            y = _seg_labels_from_periods(valid, row["fake_periods"])
            all_true.extend(y)
            all_score.extend([s["fused_fake_prob"] for s in valid])
        else:  # dfdc video-level
            if not scores:
                continue
            video_rows.append(
                {
                    "file": row["file"],
                    "label": 1 if row["label"] == "FAKE" else 0,
                    "score": sum(scores) / len(scores),
                }
            )

    out = {"dataset": args.dataset, "root": str(root), "n_videos": len(index)}
    if args.dataset == "lavdf":
        out["segment_metrics"] = _segment_metrics(all_true, all_score, cfg.fusion.threshold)
    else:
        out["video_metrics"] = _video_level_metrics(video_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
