"""Gradio frontend for the segment-level deepfake detector.

Launches a local web UI with:
  - Drag-and-drop video upload
  - Advanced settings (threshold, modality weights, frames per segment)
  - Animated progress bar during inference
  - Plotly timeline chart of per-segment fake probabilities
  - Flagged segments table with click-to-jump timestamps
  - Synced video player
  - Face crop gallery for flagged segments
  - Live-updating threshold slider (no re-inference)
  - Downloadable JSON report

Run:
    python app.py              # localhost only
    python app.py --share      # shareable public gradio.live URL
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline import DeepfakePipeline, PipelineConfig  # noqa: E402
from src.utils import (  # noqa: E402
    download_media_from_url,
    get_device,
    is_url,
    normalize_video_for_pipeline,
    setup_logging,
)

setup_logging("INFO")


# --------------------------------------------------------------------------- #
# In-memory session cache so the threshold slider can re-flag without
# re-running inference. One active analysis per UI instance.
# --------------------------------------------------------------------------- #
class SessionState:
    def __init__(self):
        self.video_path: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.crops_dir: Optional[Path] = None
        self.pipeline: Optional[DeepfakePipeline] = None   # cached to skip reloads

    def clear_crops(self) -> None:
        if self.crops_dir and self.crops_dir.exists():
            shutil.rmtree(self.crops_dir, ignore_errors=True)
        self.crops_dir = None


STATE = SessionState()


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _summary_markdown(result: Dict[str, Any], threshold: float) -> str:
    segs = result["segments"]
    scored = [s for s in segs if s.get("fused_fake_prob") is not None]
    flagged = [s for s in scored if s["fused_fake_prob"] >= threshold]
    n = len(segs)
    frac = len(flagged) / max(1, n)

    # Verdict heuristic: flag if > 20% of valid segments cross threshold
    # OR the mean fused probability over the whole clip is above threshold.
    mean_prob = (sum(s["fused_fake_prob"] for s in scored) / len(scored)) if scored else 0.0
    if frac >= 0.2 or mean_prob >= threshold:
        verdict = "### ⚠️ Likely **TAMPERED**"
        color = "#b91c1c"
    else:
        verdict = "### ✅ Likely **AUTHENTIC**"
        color = "#047857"

    return f"""
<div style="padding:12px;border-left:4px solid {color};background:#f9fafb;border-radius:4px">
{verdict}

**Video-level fake probability:** `{mean_prob:.3f}`
**Flagged segments:** `{len(flagged)} / {n}` ({100*frac:.1f}%)
**Threshold:** `{threshold:.2f}` &nbsp;·&nbsp; **Device:** `{result.get("device","?")}` &nbsp;·&nbsp; **Window:** `{result.get("segment_seconds", 1.0)}s`

</div>
"""


def _timeline_figure(result: Dict[str, Any], threshold: float) -> go.Figure:
    segs = result["segments"]
    t_mid = [(s["t_start"] + s["t_end"]) / 2 for s in segs]
    video = [s.get("video_fake_prob") for s in segs]
    audio = [s.get("audio_fake_prob") for s in segs]
    fused = [s.get("fused_fake_prob") for s in segs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_mid, y=fused, name="Fused",
            mode="lines+markers",
            line=dict(color="#dc2626", width=3),
            marker=dict(size=6),
            hovertemplate="t=%{x:.1f}s<br>fused=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_mid, y=video, name="Video",
            mode="lines", line=dict(color="#2563eb", width=1.5, dash="solid"),
            hovertemplate="t=%{x:.1f}s<br>video=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_mid, y=audio, name="Audio",
            mode="lines", line=dict(color="#059669", width=1.5, dash="dot"),
            hovertemplate="t=%{x:.1f}s<br>audio=%{y:.3f}<extra></extra>",
        )
    )

    # Threshold line + shaded flagged regions
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#6b7280",
        annotation_text=f"threshold={threshold:.2f}", annotation_position="top right",
    )
    for s in segs:
        fp = s.get("fused_fake_prob")
        if fp is not None and fp >= threshold:
            fig.add_vrect(
                x0=s["t_start"], x1=s["t_end"],
                fillcolor="rgba(220,38,38,0.12)", line_width=0,
            )

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Fake probability",
        yaxis=dict(range=[-0.02, 1.02]),
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def _flagged_table(result: Dict[str, Any], threshold: float) -> List[List[Any]]:
    rows = []
    for s in result["segments"]:
        fp = s.get("fused_fake_prob")
        if fp is None or fp < threshold:
            continue
        rows.append(
            [
                f"{s['t_start']:.1f}–{s['t_end']:.1f}s",
                f"{fp:.3f}",
                f"{s['video_fake_prob']:.3f}" if s.get("video_fake_prob") is not None else "—",
                f"{s['audio_fake_prob']:.3f}" if s.get("audio_fake_prob") is not None else "—",
                s.get("notes") or "",
            ]
        )
    return rows


def _face_gallery(result: Dict[str, Any], threshold: float) -> List[Tuple[str, str]]:
    gallery: List[Tuple[str, str]] = []
    for s in result["segments"]:
        fp = s.get("fused_fake_prob")
        path = s.get("face_crop_path")
        if fp is None or fp < threshold or not path:
            continue
        if Path(path).exists():
            gallery.append((path, f"t={s['t_start']:.1f}s  ·  p={fp:.2f}"))
    return gallery


def _write_json_report(result: Dict[str, Any]) -> str:
    out = _ROOT / "out" / "gradio_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    return str(out)


def _render_all(result: Dict[str, Any], threshold: float):
    return (
        _summary_markdown(result, threshold),
        _timeline_figure(result, threshold),
        _flagged_table(result, threshold),
        _face_gallery(result, threshold),
        _write_json_report(result),
    )


def _empty_outputs(message: str):
    empty_fig = go.Figure()
    empty_fig.update_layout(height=380, xaxis_title="Time (s)", yaxis_title="Fake probability")
    return message, empty_fig, [], [], None


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def fetch_from_url(url: str, progress=gr.Progress(track_tqdm=False)):
    """Download a URL via yt-dlp and return the local path for the Video input."""
    url = (url or "").strip()
    if not url:
        return None, "_Paste a YouTube / Instagram / TikTok / direct mp4 link above._"
    if not is_url(url):
        return None, f"_Doesn't look like a URL: `{url}`_"
    try:
        progress(0.05, desc="Downloading via yt-dlp…")
        local = download_media_from_url(url)
        progress(1.0, desc="Done.")
        return local, f"_Fetched: `{Path(local).name}` — click **Analyze** to run detection._"
    except Exception as e:
        return None, f"_Download failed: {type(e).__name__}: {e}_"


def analyze(
    video_path: Optional[str],
    url: str,
    threshold: float,
    video_weight: float,
    audio_weight: float,
    frames_per_segment: int,
    progress=gr.Progress(track_tqdm=True),
):
    # Accept either an uploaded file OR a pasted URL. A URL in the textbox
    # overrides the uploader so the user doesn't have to hit "Fetch" first.
    url = (url or "").strip()
    if not video_path and is_url(url):
        progress(0.0, desc="Downloading from URL…")
        try:
            video_path = download_media_from_url(url)
        except Exception as e:
            yield _empty_outputs(f"_Download failed: {type(e).__name__}: {e}_")
            return

    if not video_path:
        yield _empty_outputs("_Upload a video or paste a URL to begin._")
        return

    progress(0.01, desc="Normalizing video (rotation + codec)…")
    video_path = normalize_video_for_pipeline(video_path)

    progress(0.02, desc="Loading config…")
    cfg = PipelineConfig.from_yaml(str(_ROOT / "configs" / "default.yaml"))
    cfg.fusion.threshold = float(threshold)
    cfg.fusion.video_weight = float(video_weight)
    cfg.fusion.audio_weight = float(audio_weight)
    cfg.frames_per_segment = int(frames_per_segment)
    cfg.video.frames_per_segment = cfg.frames_per_segment  # keep in sync

    STATE.clear_crops()
    STATE.crops_dir = Path(tempfile.mkdtemp(prefix="dfdet_crops_"))

    progress(0.05, desc=f"Loading models on {get_device()}…")
    if STATE.pipeline is None:
        STATE.pipeline = DeepfakePipeline(cfg)
    else:
        # Keep the cached detector instances (avoids re-downloading weights)
        # but refresh fusion / frames-per-segment knobs.
        STATE.pipeline.cfg = cfg

    progress(0.1, desc="Running detection (tqdm bar will advance)…")
    result = STATE.pipeline.run(video_path, progress=True, face_crops_dir=STATE.crops_dir)

    STATE.video_path = video_path
    STATE.result = result

    yield _render_all(result, float(threshold))


def rethreshold(threshold: float):
    """Re-flag the cached result without re-running inference."""
    if STATE.result is None:
        return _empty_outputs("_Run an analysis first._")
    return _render_all(STATE.result, float(threshold))


def echo_video(path):
    return path


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
#dfdet-container {max-width: 1400px; margin: auto;}
.gradio-container h1 {font-weight: 700;}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Segment-Level Deepfake Detector", css=CUSTOM_CSS, fill_width=True) as demo:
        gr.Markdown(
            """
            # Segment-Level Deepfake Detector
            Upload a video. Each **1-second window** is analysed independently for visual
            (face-crop) and audio (spoof) tampering. The two streams are fused into a
            per-segment probability so you get **localization**, not just a single verdict.
            """,
            elem_id="dfdet-container",
        )

        with gr.Row():
            # Left column: input & settings
            with gr.Column(scale=1, min_width=320):
                video_in = gr.Video(label="Input video", sources=["upload"], height=280)
                with gr.Group():
                    url_in = gr.Textbox(
                        label="…or paste a URL",
                        placeholder="https://www.youtube.com/watch?v=…  ·  instagram.com/reel/…  ·  direct .mp4 / .mp3",
                        lines=1,
                    )
                    fetch_btn = gr.Button("⬇️ Fetch from URL", variant="secondary", size="sm")
                    url_status = gr.Markdown(
                        "_YouTube / Instagram / TikTok / Twitter / direct mp4 or mp3 all work._"
                    )
                run_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                threshold = gr.Slider(
                    0.0, 1.0, value=0.30, step=0.01,
                    label="Decision threshold",
                    info="Drag after analysis to re-flag without re-running.",
                )
                with gr.Accordion("Advanced settings", open=False):
                    video_w = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Video weight")
                    audio_w = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Audio weight")
                    fps_seg = gr.Slider(
                        1, 15, value=5, step=1,
                        label="Frames scored per segment",
                        info="More frames = more stable video score, slower inference.",
                    )
                gr.Markdown(
                    f"<sub>Device: **{get_device()}**</sub>",
                )

            # Right column: results
            with gr.Column(scale=2):
                summary_md = gr.Markdown("_Upload a video and click Analyze._")
                chart = gr.Plot(label="Per-segment fake probabilities")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Playback")
                video_out = gr.Video(label="Video preview", height=320, interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("### 🚩 Flagged segments")
                flagged_df = gr.Dataframe(
                    headers=["Window", "Fused", "Video", "Audio", "Notes"],
                    datatype=["str", "str", "str", "str", "str"],
                    label=None,
                    wrap=True,
                    interactive=False,
                    row_count=(0, "dynamic"),
                )

        gr.Markdown("### 🧑 Face crops from flagged windows")
        face_gallery = gr.Gallery(
            columns=8, height=180, object_fit="cover",
            label=None, show_label=False, allow_preview=True,
        )

        json_file = gr.File(label="📄 Download JSON report", interactive=False)

        # Wire events
        video_in.change(echo_video, inputs=video_in, outputs=video_out)

        # Fetch URL -> populate the uploader + preview player, then user clicks Analyze.
        fetch_btn.click(
            fetch_from_url,
            inputs=[url_in],
            outputs=[video_in, url_status],
        )
        # Pressing Enter in the URL box also triggers the fetch.
        url_in.submit(
            fetch_from_url,
            inputs=[url_in],
            outputs=[video_in, url_status],
        )

        run_btn.click(
            analyze,
            inputs=[video_in, url_in, threshold, video_w, audio_w, fps_seg],
            outputs=[summary_md, chart, flagged_df, face_gallery, json_file],
        )

        # Live re-flag on threshold release (not during drag, to avoid spam)
        threshold.release(
            rethreshold,
            inputs=[threshold],
            outputs=[summary_md, chart, flagged_df, face_gallery, json_file],
        )

    return demo


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="Create a public gradio.live tunnel.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
