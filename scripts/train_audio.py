#!/usr/bin/env python
"""Fine-tune a HuggingFace audio-classification model on ASVspoof 2019 LA.

Defaults to LoRA on Wav2Vec2 attention projections so it runs on a single
consumer GPU. Pass --full-finetune to update all weights (needs ~16 GB VRAM
for a base Wav2Vec2; bf16 or fp16 recommended).

Typical run on a cloud A100:

    python scripts/prepare_asvspoof.py --root /data/ASVspoof2019_LA
    python scripts/train_audio.py \
        --manifest-dir data/asvspoof \
        --output-dir checkpoints/audio-asvspoof-lora \
        --epochs 6 \
        --batch-size 32 \
        --lr 3e-4 \
        --lora-r 16

At the end you'll have:
    checkpoints/audio-asvspoof-lora/
    ├── adapter_model.safetensors    # LoRA weights
    ├── adapter_config.json
    ├── merged/                      # base + LoRA merged, ready for inference
    │   ├── model.safetensors
    │   ├── config.json
    │   └── preprocessor_config.json
    └── training_args.json

Plug the merged dir into configs/default.yaml:

    audio:
      model_id: /abs/path/to/checkpoints/audio-asvspoof-lora/merged
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_audio")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model-id",
        default="MelodyMachine/Deepfake-audio-detection",
        help="Base HuggingFace model (default matches inference default).",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/asvspoof"),
        help="Directory containing asvspoof_{train,dev,eval}.csv.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/audio-asvspoof-lora"),
    )
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--segment-seconds",
        type=float,
        default=1.0,
        help="Must match configs/default.yaml for the inference pipeline.",
    )
    ap.add_argument("--noise-prob", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA vs full
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        help="Update all weights instead of LoRA (slower, more VRAM).",
    )
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument(
        "--fp16",
        action="store_true",
        help="Mixed-precision fp16. Prefer bf16 on Ampere+ (see --bf16).",
    )
    ap.add_argument("--bf16", action="store_true", help="Mixed-precision bf16.")
    ap.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting (default: enabled).",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from a checkpoint directory.",
    )
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate and the threshold at which FAR == FRR.

    scores are P(spoof) (the positive-class probability). label 1 = spoof.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # The EER is where fpr and fnr cross.
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def build_metrics_fn():
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        # softmax, take prob of class 1 (spoof)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        spoof_prob = probs[:, 1]
        preds = (spoof_prob >= 0.5).astype(np.int64)

        acc = float((preds == labels).mean())
        try:
            roc_auc = float(roc_auc_score(labels, spoof_prob))
        except ValueError:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(labels, spoof_prob))
        except ValueError:
            pr_auc = float("nan")
        try:
            eer, eer_thr = compute_eer(labels, spoof_prob)
        except Exception:
            eer, eer_thr = float("nan"), float("nan")

        return {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "eer": eer,
            "eer_threshold": eer_thr,
        }

    return compute_metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    # Imports deferred so `--help` works without the heavy deps installed.
    import torch
    import transformers
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    # Our dataset class
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.training.asvspoof_dataset import (  # noqa: E402
        ASVspoofDataset,
        compute_class_weights,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------------- #
    # Load base model + feature extractor
    # ----------------------------------------------------------------------- #
    log.info("loading base model: %s", args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)

    label2id = {"bonafide": 0, "spoof": 1}
    id2label = {0: "bonafide", 1: "spoof"}
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # ----------------------------------------------------------------------- #
    # LoRA
    # ----------------------------------------------------------------------- #
    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model, TaskType

        # Targets that exist on Wav2Vec2 / HuBERT encoders:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            modules_to_save=["classifier", "projector"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        log.info("full fine-tune: all parameters trainable.")

    # ----------------------------------------------------------------------- #
    # Datasets
    # ----------------------------------------------------------------------- #
    train_csv = args.manifest_dir / "asvspoof_train.csv"
    dev_csv = args.manifest_dir / "asvspoof_dev.csv"
    if not train_csv.exists() or not dev_csv.exists():
        log.error(
            "Manifest CSVs not found under %s. "
            "Run scripts/prepare_asvspoof.py first.",
            args.manifest_dir,
        )
        return 2

    train_ds = ASVspoofDataset(
        train_csv,
        feature_extractor,
        segment_seconds=args.segment_seconds,
        training=True,
        noise_prob=args.noise_prob,
        seed=args.seed,
    )
    dev_ds = ASVspoofDataset(
        dev_csv,
        feature_extractor,
        segment_seconds=args.segment_seconds,
        training=False,
    )
    log.info("train=%d  dev=%d", len(train_ds), len(dev_ds))

    # Class-weighted loss (Wav2Vec2ForSequenceClassification uses CE internally)
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_csv)
        log.info("class weights (bonafide, spoof): %s", class_weights.tolist())
    else:
        class_weights = None

    # ----------------------------------------------------------------------- #
    # Custom Trainer to inject class-weighted loss
    # ----------------------------------------------------------------------- #
    class WeightedTrainer(Trainer):
        def __init__(self, *a, class_weights=None, **kw):
            super().__init__(*a, **kw)
            self._class_weights = class_weights

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = (
                self._class_weights.to(logits.device)
                if self._class_weights is not None
                else None
            )
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    # ----------------------------------------------------------------------- #
    # Data collator — pad input_values to the longest in the batch
    # ----------------------------------------------------------------------- #
    def collate(batch):
        input_values = [b["input_values"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch])
        # All are fixed length (segment_seconds * sample_rate), so stacking works.
        # But torch.stack requires identical shapes — which we have by construction.
        input_values = torch.stack(input_values)
        return {"input_values": input_values, "labels": labels}

    # ----------------------------------------------------------------------- #
    # TrainingArguments
    # ----------------------------------------------------------------------- #
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eer",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate,
        compute_metrics=build_metrics_fn(),
        class_weights=class_weights,
    )

    # ----------------------------------------------------------------------- #
    # Train
    # ----------------------------------------------------------------------- #
    log.info("starting training (transformers %s)", transformers.__version__)
    if args.resume_from is not None:
        trainer.train(resume_from_checkpoint=str(args.resume_from))
    else:
        trainer.train()

    # Final dev metrics for the log
    metrics = trainer.evaluate()
    log.info("final dev metrics: %s", metrics)

    # ----------------------------------------------------------------------- #
    # Save
    # ----------------------------------------------------------------------- #
    trainer.save_model(str(args.output_dir))
    feature_extractor.save_pretrained(str(args.output_dir))

    # Save merged-for-inference copy (LoRA only)
    if not args.full_finetune:
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        log.info("merging LoRA weights into base model -> %s", merged_dir)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        feature_extractor.save_pretrained(str(merged_dir))
    else:
        merged_dir = args.output_dir

    # Training config
    with (args.output_dir / "training_args.json").open("w") as f:
        json.dump(
            {**vars(args), "final_dev_metrics": metrics},
            f,
            indent=2,
            default=str,
        )

    log.info("done. To plug into inference, edit configs/default.yaml:")
    log.info("    audio.model_id: %s", merged_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
