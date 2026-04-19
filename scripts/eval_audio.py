#!/usr/bin/env python
"""Evaluate a fine-tuned audio classifier on the ASVspoof 2019 LA eval set.

Reports: EER, ROC-AUC, PR-AUC, accuracy @ EER threshold, and per-attack EER
(A07..A19 for LA eval). Writes a JSON report.

Usage:
    python scripts/eval_audio.py \
        --model-path checkpoints/audio-asvspoof-lora/merged \
        --manifest data/asvspoof/asvspoof_eval.csv \
        --out out/audio_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_audio")


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Local directory with model.safetensors + config.json + "
        "preprocessor_config.json (the 'merged' dir from train_audio.py), "
        "or a HF hub model id.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/asvspoof/asvspoof_eval.csv"),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--segment-seconds", type=float, default=1.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("out/audio_eval.json"),
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Force cuda / cpu / mps. Default: auto-detect.",
    )
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.training.asvspoof_dataset import ASVspoofDataset  # noqa: E402

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("device=%s", device)

    log.info("loading model from %s", args.model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(args.model_path))
    model = AutoModelForAudioClassification.from_pretrained(str(args.model_path))
    model.to(device).eval()

    ds = ASVspoofDataset(
        args.manifest,
        feature_extractor,
        segment_seconds=args.segment_seconds,
        training=False,
    )
    log.info("eval set: %d utterances", len(ds))

    def collate(batch):
        input_values = torch.stack([b["input_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"input_values": input_values, "labels": labels}

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["input_values"].to(device, non_blocking=True)
            logits = model(input_values=inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(spoof)
            all_scores.extend(probs.float().cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
            if (i + 1) % 20 == 0:
                log.info("  batch %d / %d", i + 1, len(loader))

    scores = np.asarray(all_scores, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.int64)

    from sklearn.metrics import roc_auc_score, average_precision_score

    eer, eer_thr = compute_eer(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    preds_at_thr = (scores >= eer_thr).astype(np.int64)
    acc_at_thr = float((preds_at_thr == labels).mean())

    # Per-attack EER: group eval spoof examples by spoof_type.
    per_attack: dict = {}
    try:
        import csv

        with args.manifest.open() as f:
            rows = list(csv.DictReader(f))
        attack_scores: dict[str, list[float]] = defaultdict(list)
        bonafide_scores: list[float] = []
        for r, s in zip(rows, scores):
            if int(r["label"]) == 0:
                bonafide_scores.append(s)
            else:
                attack_scores[r["spoof_type"]].append(s)

        bona = np.asarray(bonafide_scores)
        for attack, attack_s in sorted(attack_scores.items()):
            attack_s = np.asarray(attack_s)
            y = np.concatenate(
                [np.zeros_like(bona, dtype=np.int64), np.ones_like(attack_s, dtype=np.int64)]
            )
            sc = np.concatenate([bona, attack_s])
            try:
                a_eer, _ = compute_eer(y, sc)
            except Exception:
                a_eer = float("nan")
            per_attack[attack] = {"eer": a_eer, "n": int(len(attack_s))}
    except Exception as e:
        log.warning("per-attack breakdown failed: %s", e)

    report = {
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "n": int(len(labels)),
        "metrics": {
            "eer": eer,
            "eer_threshold": eer_thr,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy_at_eer_threshold": acc_at_thr,
        },
        "per_attack": per_attack,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(report, f, indent=2)

    log.info("EER     = %.4f  (threshold %.3f)", eer, eer_thr)
    log.info("ROC-AUC = %.4f", roc_auc)
    log.info("PR-AUC  = %.4f", pr_auc)
    log.info("Acc@EER = %.4f", acc_at_thr)
    log.info("report saved -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
