#!/usr/bin/env python
"""Fine-tune the audio spoof model for *segment-level* (partial) detection.

Unlike scripts/train_audio.py — which labels every 1s crop with its file's
single label and therefore learns an utterance-level decision — this script
trains on *segment-level* manifests where real and fake audio coexist inside
one file. Each fixed-width window carries its own label (see
src/training/segment_dataset.py), so the model learns to localize tampering
instead of flagging the whole clip.

Defaults to LoRA on Wav2Vec2 attention projections (same recipe as
train_audio.py) so it runs on a single Kaggle GPU, and it can **refine your
existing model**: point --model-id at the merged/ directory produced by a
previous train_audio.py run and it stacks a fresh segment-level adapter on
top. You can also pass several segment manifests to combine partial-spoof
data built from different source datasets (ASVspoof, WaveFake, ...).

Build the manifests first with scripts/make_partialspoof.py, then:

    python scripts/train_audio_segments.py \
        --model-id checkpoints/audio-asvspoof-lora/merged \
        --train-csv data/partialspoof/train/partial_train.csv \
                    data/partialspoof/wavefake/partial_train.csv \
        --dev-csv   data/partialspoof/dev/partial_dev.csv \
        --output-dir checkpoints/audio-partial-lora \
        --epochs 5 --batch-size 32 --lr 3e-4 --lora-r 16

Output mirrors train_audio.py: adapter weights + a merged/ dir ready to drop
into configs/default.yaml as audio.model_id.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_audio_segments")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model-id",
        default="MelodyMachine/Deepfake-audio-detection",
        help="Base model OR a local merged/ dir from a prior run to refine.",
    )
    ap.add_argument(
        "--train-csv", type=Path, nargs="+", required=True,
        help="One or more segment-level manifests (make_partialspoof.py).",
    )
    ap.add_argument(
        "--dev-csv", type=Path, nargs="+", required=True,
        help="One or more segment-level dev manifests.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/audio-partial-lora"),
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--segment-seconds", type=float, default=1.0,
        help="Window width. Must match configs/default.yaml segment_seconds.",
    )
    ap.add_argument(
        "--segment-stride-seconds", type=float, default=None,
        help="Window hop (default = segment-seconds, i.e. no overlap).",
    )
    ap.add_argument(
        "--positive-overlap", type=float, default=0.5,
        help="Fraction of a window that must be fake to label it spoof.",
    )
    ap.add_argument(
        "--max-windows-per-file", type=int, default=0,
        help="Cap windows drawn from any one file (0 = no cap).",
    )
    ap.add_argument("--noise-prob", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA vs full
    ap.add_argument("--full-finetune", action="store_true",
                    help="Update all weights instead of LoRA.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable window-level inverse-frequency weighting.")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="Resume training from a checkpoint directory.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Metrics (segment/window level)
# --------------------------------------------------------------------------- #
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def build_metrics_fn():
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)
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
            seg_f1 = float(f1_score(labels, preds, zero_division=0))
        except ValueError:
            seg_f1 = float("nan")
        try:
            eer, eer_thr = compute_eer(labels, spoof_prob)
        except Exception:
            eer, eer_thr = float("nan"), float("nan")

        return {
            "accuracy": acc,
            "segment_f1": seg_f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "eer": eer,
            "eer_threshold": eer_thr,
        }

    return compute_metrics


# --------------------------------------------------------------------------- #
# Dataset assembly (support multiple manifests via ConcatDataset)
# --------------------------------------------------------------------------- #
def build_dataset(csvs: List[Path], feature_extractor, args, training: bool):
    from torch.utils.data import ConcatDataset

    from src.training.segment_dataset import SegmentLevelAudioDataset

    parts = []
    for c in csvs:
        if not c.exists():
            raise FileNotFoundError(f"manifest not found: {c}")
        parts.append(
            SegmentLevelAudioDataset(
                c,
                feature_extractor,
                segment_seconds=args.segment_seconds,
                segment_stride_seconds=args.segment_stride_seconds,
                positive_overlap=args.positive_overlap,
                training=training,
                noise_prob=args.noise_prob if training else 0.0,
                max_windows_per_file=args.max_windows_per_file,
                seed=args.seed,
            )
        )
    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)


def _window_label_counts(dataset) -> np.ndarray:
    """Total (bonafide, spoof) window counts across a (possibly Concat) dataset."""
    from torch.utils.data import ConcatDataset

    from src.training.segment_dataset import SegmentLevelAudioDataset

    counts = np.zeros(2, dtype=np.int64)
    subsets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for ds in subsets:
        if isinstance(ds, SegmentLevelAudioDataset):
            labs = ds.labels()
            counts[0] += int((labs == 0).sum())
            counts[1] += int((labs == 1).sum())
    return counts


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    import torch
    import transformers
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------------------------------------------------- #
    # Base model + feature extractor
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
    log.info("building train windows from %d manifest(s)", len(args.train_csv))
    train_ds = build_dataset(args.train_csv, feature_extractor, args, training=True)
    log.info("building dev windows from %d manifest(s)", len(args.dev_csv))
    dev_ds = build_dataset(args.dev_csv, feature_extractor, args, training=False)

    train_counts = _window_label_counts(train_ds)
    log.info("train windows=%d  (bonafide=%d, spoof=%d)",
             len(train_ds), int(train_counts[0]), int(train_counts[1]))
    log.info("dev   windows=%d", len(dev_ds))

    # Window-level class-weighted loss.
    if not args.no_class_weights:
        safe = np.maximum(train_counts, 1)
        inv = safe.sum() / (2.0 * safe)
        class_weights = torch.tensor(inv, dtype=torch.float32)
        log.info("window class weights (bonafide, spoof): %s",
                 class_weights.tolist())
    else:
        class_weights = None

    # ----------------------------------------------------------------------- #
    # Weighted trainer + collator
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

    def collate(batch):
        input_values = torch.stack([b["input_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
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
    log.info("starting segment-level training (transformers %s)",
             transformers.__version__)
    if args.resume_from is not None:
        trainer.train(resume_from_checkpoint=str(args.resume_from))
    else:
        trainer.train()

    metrics = trainer.evaluate()
    log.info("final dev metrics: %s", metrics)

    # ----------------------------------------------------------------------- #
    # Save
    # ----------------------------------------------------------------------- #
    trainer.save_model(str(args.output_dir))
    feature_extractor.save_pretrained(str(args.output_dir))

    if not args.full_finetune:
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        log.info("merging LoRA weights into base model -> %s", merged_dir)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        feature_extractor.save_pretrained(str(merged_dir))
    else:
        merged_dir = args.output_dir

    with (args.output_dir / "training_args.json").open("w") as f:
        json.dump(
            {**vars(args), "final_dev_metrics": metrics},
            f, indent=2, default=str,
        )

    log.info("done. Plug into inference via configs/default.yaml:")
    log.info("    audio.model_id: %s", merged_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
