# Segment-Level Deepfake Video & Audio Tampering Detection

A practical, runnable Python pipeline that ingests a video file and produces
**per-segment (1-second window)** predictions for both:

1. **Visual tampering / deepfake faces** — face extraction + pretrained image
   deepfake classifier (Xception / EfficientNet via HuggingFace).
2. **Audio tampering / spoof** — pretrained Wav2Vec2-based spoof detector
   operating on 1-second clips.

The two streams are fused into per-segment probabilities and a final timeline
JSON, so you get localization (which 1-second windows are tampered), not just
a single video-level label.

---

## Why segment-level?

Video-level "real vs fake" classifiers mask partial manipulation
(face-swap in the middle, audio overdub over the last few seconds, etc.).
A segment-level output lets you:

- Visualize a tamper timeline.
- Flag only the suspicious windows for human review.
- Evaluate localization metrics (AP@IoU, segment-F1) against datasets that
  provide frame-level labels (e.g., DFDC, AV-Deepfake1M, LAV-DF).

---

## Architecture

```
                         ┌──────────────────────────┐
 input video (.mp4) ─────▶   FFmpeg demux            │
                         │  - decode video frames   │
                         │  - extract 16 kHz mono wav
                         └──────────┬───────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
     ┌────────▼────────┐                          ┌───────▼────────┐
     │ Video segmenter │                          │ Audio segmenter│
     │ 1s windows      │                          │ 1s windows     │
     │ @ target fps    │                          │ @ 16 kHz       │
     └────────┬────────┘                          └───────┬────────┘
              │                                           │
     ┌────────▼────────┐                          ┌───────▼────────┐
     │ Face crop       │                          │ Wav2Vec2-based │
     │ (MTCNN)         │                          │ spoof detector │
     │                 │                          │                │
     │ Xception / EffN │                          │                │
     │ deepfake model  │                          │                │
     └────────┬────────┘                          └───────┬────────┘
              │ per-frame fake prob                       │ per-segment spoof prob
              │ → aggregate per 1s segment                │
              └─────────────────┬─────────────────────────┘
                                │
                      ┌─────────▼─────────┐
                      │ Late-fusion head  │  (weighted avg or LR)
                      └─────────┬─────────┘
                                │
                     per-segment JSON timeline
```

---

## Project layout

```
Deepfake detection/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # model ids, thresholds, fusion weights
├── src/
│   ├── __init__.py
│   ├── segmenter.py          # 1s-window video+audio segmenter
│   ├── video_detector.py     # face extraction + deepfake classifier
│   ├── audio_detector.py     # Wav2Vec2 spoof detector
│   ├── fusion.py             # late-fusion of the two streams
│   ├── pipeline.py           # orchestrator
│   └── utils.py              # ffmpeg I/O, logging, device mgmt
├── scripts/
│   ├── run_inference.py      # CLI: video in -> timeline JSON out
│   ├── download_datasets.py  # FaceForensics++, DFDC, ASVspoof helpers
│   └── evaluate.py           # segment-level AP / F1 on a dataset
└── tests/
    └── test_pipeline.py      # smoke tests with a synthetic clip
```

---

## Quickstart

```bash
# 1. Install system deps
#    Ubuntu:
sudo apt-get install -y ffmpeg

# 2. Python deps (use a venv)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run on a video
python scripts/run_inference.py \
    --video path/to/sample.mp4 \
    --config configs/default.yaml \
    --out out/sample.timeline.json
```

The output JSON looks like:

```json
{
  "video": "sample.mp4",
  "segment_seconds": 1.0,
  "segments": [
    {"t_start": 0.0, "t_end": 1.0,
     "video_fake_prob": 0.08, "audio_fake_prob": 0.11,
     "fused_fake_prob": 0.09, "tampered": false},
    {"t_start": 1.0, "t_end": 2.0,
     "video_fake_prob": 0.91, "audio_fake_prob": 0.34,
     "fused_fake_prob": 0.77, "tampered": true},
    ...
  ],
  "video_level": {"fake_prob": 0.42, "tampered_fraction": 0.18}
}
```

---

## Web UI (Gradio)

For interactive use there is a full web front-end at `app.py`:

```bash
# install the UI extras (once):
pip install -r requirements.txt       # now includes gradio + plotly

# local-only:
python app.py
# -> http://127.0.0.1:7860

# shareable public URL (Gradio tunnel):
python app.py --share
```

The UI has:

- drag-and-drop video upload,
- an Analyze button with a progress bar,
- an interactive **Plotly timeline** showing per-second video / audio / fused
  fake probabilities, with flagged windows shaded in red,
- a **threshold slider** that re-flags segments live (no re-inference),
- a **flagged-segments table** with precise timestamps,
- a **face-crop gallery** showing the most-suspicious face detected in each
  flagged window,
- a download link for the JSON report.

Face crops are written to a temp dir while the app is running and cleaned up
on the next analysis.

---

## Web UI (Verity — FastAPI + static HTML)

A second, more polished frontend lives at `static/index.html` with a
custom dark dashboard ("Verity"). It's served by a thin FastAPI backend
(`server.py`) that reuses the exact same `DeepfakePipeline` as the
CLI and Gradio app — no duplicate inference code.

```bash
# one-time install of the new extras:
pip install -r requirements.txt       # now also installs fastapi + uvicorn + python-multipart

# local dev server (auto-reload on file changes):
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# -> open http://127.0.0.1:8000
```

The UI exposes the same capabilities as the Gradio app:

- drag-and-drop or click-to-pick file upload,
- URL paste field (YouTube / Instagram / TikTok / Twitter / direct mp4
  or mp3 — fetched via yt-dlp),
- detection-threshold and segment-window sliders, plus an
  "advanced options" panel (audio spectral toggle, face-crop tracking,
  model-backbone select),
- a custom-drawn per-segment timeline with clean / flagged /
  uncertain bands and hover-readout,
- a flagged-segments table with severity chips and probability bars,
- a face-crop grid populated from `/api/face/<id>` (real JPEGs saved
  during inference),
- a JSON-export button that dumps the full per-segment report.

### HTTP endpoints

| Method | Path                | Purpose                                                              |
| ------ | ------------------- | -------------------------------------------------------------------- |
| GET    | `/`                 | Serves `static/index.html`                                           |
| GET    | `/api/health`       | `{status, device, pipeline_loaded}` — polled on page load             |
| POST   | `/api/analyze`      | Multipart: `file` or `url`, plus `threshold`, `window`, weights      |
| GET    | `/api/face/{id}`    | Serves a face-crop JPEG by its generated ID                          |

The `/api/analyze` response is shaped for the frontend (see `server._shape_response`
for the exact schema) and contains `verdict`, `fake_prob`, `flagged`, `total`,
per-segment probabilities, timeline bands, and face-crop IDs.

---

## Models used (pretrained, open-source)

| Stream | Model                                                 | Notes                                                                 |
|--------|-------------------------------------------------------|-----------------------------------------------------------------------|
| Video  | `prithivMLmods/Deep-Fake-Detector-Model` (ViT/Xception) | Image-level deepfake classifier; applied per face crop per frame.    |
| Audio  | `MelodyMachine/Deepfake-audio-detection`              | Wav2Vec2-based spoof detector; applied per 1s clip.                   |
| Face detection | MTCNN via `facenet-pytorch`                     | Fast multi-face detector for face crops.                              |

These are swappable in `configs/default.yaml`. You can drop in any
HuggingFace `image-classification` or `audio-classification` model with
`fake`/`real` labels.

For best accuracy, fine-tune on:
- **FaceForensics++** (face manipulation) — `scripts/download_datasets.py ff`
- **DFDC** (Deepfake Detection Challenge) — `scripts/download_datasets.py dfdc`
- **ASVspoof 2019/2021** (audio spoof) — `scripts/download_datasets.py asvspoof`
- **LAV-DF / AV-Deepfake1M** (segment-level audio-visual) — recommended for
  segment-level evaluation.

---

## Evaluation

```bash
python scripts/evaluate.py \
    --dataset dfdc \
    --dataset-root /data/dfdc \
    --config configs/default.yaml \
    --split val
```

Reports:
- Segment-level ROC-AUC / PR-AUC.
- Segment F1 at the tuned threshold.
- Localization AP@IoU=0.5 (when frame-level labels are available).
- Video-level AUC (for comparison with standard benchmarks).

---

## Fine-tuning on ASVspoof 2019 LA (audio branch)

The pretrained audio model (`MelodyMachine/Deepfake-audio-detection`)
is a general-purpose spoof detector. On ASVspoof 2019 LA eval you can
typically cut EER by 3-10x by fine-tuning on the matched distribution.

### 1. Expected dataset layout

After you accept the EULA at
<https://datashare.ed.ac.uk/handle/10283/3336> and extract the tarballs,
point the prepare script at the root directory containing:

```
ASVspoof2019_LA/
├── ASVspoof2019_LA_train/flac/LA_T_*.flac
├── ASVspoof2019_LA_dev/flac/LA_D_*.flac
├── ASVspoof2019_LA_eval/flac/LA_E_*.flac
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### 2. Moving data from your laptop to a cloud GPU

If the data lives on your local disk but you're training on a rented GPU:

```bash
# Lambda Labs / Vast / most Linux cloud boxes:
rsync -avz --progress /mnt/c/path/to/ASVspoof2019_LA/ \
    user@<cloud-ip>:/data/ASVspoof2019_LA/

# Google Cloud:
gsutil -m cp -r /mnt/c/path/to/ASVspoof2019_LA gs://<bucket>/
gcloud compute scp --recurse /local/path <instance>:/data/

# Or tar + scp for smaller transfer overhead:
tar -C /path/to -cf - ASVspoof2019_LA | \
    ssh user@<cloud-ip> "tar -C /data -xf -"
```

The full LA set is ~30 GB — expect 30-90 min on a reasonable home connection.

### 3. Build manifests

```bash
python scripts/prepare_asvspoof.py \
    --root /data/ASVspoof2019_LA \
    --out-dir data/asvspoof
```

This writes `data/asvspoof/asvspoof_{train,dev,eval}.csv` with
`(file_path, label, spoof_type, speaker_id, split)` rows. Use
`--check-only` to validate the layout without writing anything.

### 4. Train (LoRA fine-tune)

```bash
python scripts/train_audio.py \
    --manifest-dir data/asvspoof \
    --output-dir checkpoints/audio-asvspoof-lora \
    --epochs 6 \
    --batch-size 32 \
    --lr 3e-4 \
    --bf16            # or --fp16 on pre-Ampere GPUs
```

Trainable params with defaults: ~0.5% of the base model. Typical
A100 wall-clock: ~2–3 hours for 6 epochs. Best checkpoint (by dev EER)
is restored at the end.

Pass `--full-finetune` to update all weights instead (needs ~16 GB VRAM
at batch size 16, ~1.5x slower, usually 1–2 pp better EER).

### 5. Evaluate on the eval split

```bash
python scripts/eval_audio.py \
    --model-path checkpoints/audio-asvspoof-lora/merged \
    --manifest data/asvspoof/asvspoof_eval.csv \
    --out out/audio_eval.json
```

Reports EER, ROC-AUC, PR-AUC, and per-attack (A07–A19) EER breakdown.

### 6. Plug into the inference pipeline

Edit `configs/default.yaml`:

```yaml
audio:
  model_id: /abs/path/to/checkpoints/audio-asvspoof-lora/merged
```

Then run `scripts/run_inference.py` or `python app.py` as normal — the
pipeline loads your fine-tuned weights with no code changes.

### Expected numbers (rough guide)

Published Wav2Vec2-based ASVspoof 2019 LA EER figures land around:

| Setup | Eval EER |
|---|---|
| Off-the-shelf general spoof detector (no FT) | 15-30% |
| Our LoRA fine-tune, 6 epochs, base Wav2Vec2 | ~3-6% |
| Full fine-tune + AAM-softmax + aug | ~1-3% |

Your numbers will vary with base model, batch size, and augmentation.
Don't chase SOTA on the first run — get the loop working end-to-end, then
iterate.

---

## Limitations & honest notes

- The off-the-shelf HF models are **general-purpose** deepfake detectors.
  They generalize imperfectly to unseen manipulation methods — expect a
  drop on out-of-distribution attacks. Fine-tuning is recommended.
- Face-agnostic visual tampering (splicing, object edits) is **not**
  handled by a face-crop pipeline. For those, swap in a frame-level
  forensic model (e.g., `MantraNet`, `TruFor`).
- Audio detection assumes clean-ish speech. Very noisy clips, music, or
  non-speech audio reduce reliability.
- All downloads for the public benchmarks require EULA acceptance; the
  `download_datasets.py` script prints the correct URL and instructions
  rather than scraping restricted data.
