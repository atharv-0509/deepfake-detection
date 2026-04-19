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
