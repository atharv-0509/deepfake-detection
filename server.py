"""FastAPI backend for the Verity frontend (static/index.html).

This is a thin HTTP wrapper around the existing `DeepfakePipeline`. It
reuses every bit of inference code from `src/`; nothing about the model
or fusion changes. The frontend POSTs a file (or URL) + threshold +
window, and we return a JSON payload shaped for the Verity UI:

    {
      "verdict": "tampered" | "authentic",
      "fake_prob": 0.78,
      "threshold": 0.30,
      "flagged": 3, "total": 12,
      "audio_integrity": 0.61,
      "face_confidence": 0.92,
      "segments": [ {start, end, fused, video, audio, notes, sev, flagged, face_id}, ... ],
      "timeline":  [ {s, e, type}, ... ],
      "faces":     [ {id, label, flagged}, ... ],
      "device": "cpu",
      "window_s": 1.0
    }

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
logger = logging.getLogger("verity.server")

STATIC_DIR = _ROOT / "static"
CROPS_ROOT = _ROOT / "out" / "web_crops"
CROPS_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Pipeline cache — load once, reuse across requests.
# --------------------------------------------------------------------------- #
_PIPE: Optional[DeepfakePipeline] = None
# face_id -> absolute path on disk so /api/face/{id} can serve it
_FACE_INDEX: Dict[str, str] = {}


def _get_pipeline(threshold: float, video_w: float, audio_w: float, frames_per_seg: int) -> DeepfakePipeline:
    global _PIPE
    cfg = PipelineConfig.from_yaml(str(_ROOT / "configs" / "default.yaml"))
    cfg.fusion.threshold = float(threshold)
    cfg.fusion.video_weight = float(video_w)
    cfg.fusion.audio_weight = float(audio_w)
    cfg.frames_per_segment = int(frames_per_seg)
    cfg.video.frames_per_segment = cfg.frames_per_segment
    if _PIPE is None:
        logger.info("Loading DeepfakePipeline on %s …", get_device())
        _PIPE = DeepfakePipeline(cfg)
    else:
        _PIPE.cfg = cfg
    return _PIPE


# --------------------------------------------------------------------------- #
# Result -> Verity-shaped JSON
# --------------------------------------------------------------------------- #
def _fmt_ts(s: float) -> str:
    m, sec = divmod(int(round(s)), 60)
    return f"{m:02d}:{sec:02d}"


def _severity(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.50:
        return "medium"
    return "low"


def _timeline_type(p: Optional[float], thresh: float) -> str:
    if p is None:
        return "uncertain"
    if p >= thresh:
        return "flag"
    if p >= max(0.0, thresh - 0.12):
        return "uncertain"
    return "clean"


def _infer_notes(seg: Dict[str, Any]) -> str:
    """Fabricate a short human-readable tamper descriptor from per-modality scores."""
    v = seg.get("video_fake_prob")
    a = seg.get("audio_fake_prob")
    if seg.get("notes"):
        return str(seg["notes"])
    if v is not None and a is not None:
        if v >= 0.6 and a >= 0.6:
            return "Audio–visual joint tampering"
        if v >= 0.6 and a < 0.4:
            return "Visual forgery · face artifact"
        if a >= 0.6 and v < 0.4:
            return "Audio spoof · spectral mismatch"
        if v >= 0.5 and a >= 0.5:
            return "Audio–visual desync"
    if v is not None and v >= 0.5:
        return "Visual tamper · face region"
    if a is not None and a >= 0.5:
        return "Audio tamper · spectral"
    return "Flagged by fusion head"


def _shape_response(result: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Translate the pipeline output dict into the JSON shape the HTML expects."""
    segs_raw = result.get("segments", [])
    scored = [s for s in segs_raw if s.get("fused_fake_prob") is not None]
    flagged_segs = [s for s in scored if s["fused_fake_prob"] >= threshold]
    mean_prob = (sum(s["fused_fake_prob"] for s in scored) / len(scored)) if scored else 0.0
    frac = len(flagged_segs) / max(1, len(segs_raw))

    # Verdict: either >=20% of windows flagged, or mean probability >= threshold
    verdict = "tampered" if (frac >= 0.2 or mean_prob >= threshold) else "authentic"

    # Per-segment block for the timeline + table
    out_segments: List[Dict[str, Any]] = []
    for s in segs_raw:
        fp = s.get("fused_fake_prob")
        vp = s.get("video_fake_prob")
        ap = s.get("audio_fake_prob")
        face_id = None
        crop_path = s.get("face_crop_path")
        if crop_path and Path(crop_path).exists():
            face_id = str(uuid.uuid4().hex[:12])
            _FACE_INDEX[face_id] = crop_path
        out_segments.append(
            {
                "start":     float(s["t_start"]),
                "end":       float(s["t_end"]),
                "start_ts":  _fmt_ts(s["t_start"]),
                "end_ts":    _fmt_ts(s["t_end"]),
                "fused":     None if fp is None else round(fp, 4),
                "video":     None if vp is None else round(vp, 4),
                "audio":     None if ap is None else round(ap, 4),
                "flagged":   bool(fp is not None and fp >= threshold),
                "severity":  None if fp is None else _severity(fp),
                "type":      _infer_notes(s),
                "face_id":   face_id,
            }
        )

    # Timeline bands (collapsed runs of clean/flag/uncertain — frontend draws them)
    timeline: List[Dict[str, Any]] = []
    for s in segs_raw:
        fp = s.get("fused_fake_prob")
        t = _timeline_type(fp, threshold)
        if timeline and timeline[-1]["type"] == t and abs(timeline[-1]["e"] - s["t_start"]) < 1e-6:
            timeline[-1]["e"] = float(s["t_end"])
        else:
            timeline.append({"s": float(s["t_start"]), "e": float(s["t_end"]), "type": t})

    # Face gallery: pick up to 6 entries, flagged ones first
    face_entries: List[Dict[str, Any]] = []
    for s in out_segments:
        if s.get("face_id"):
            face_entries.append(
                {
                    "id": s["face_id"],
                    "label": s["start_ts"] + (" ⚠" if s["flagged"] else ""),
                    "flagged": s["flagged"],
                }
            )
    face_entries.sort(key=lambda x: (not x["flagged"],))
    face_entries = face_entries[:6]

    # Modality-level summaries for the metric cards
    v_scores = [s["video_fake_prob"] for s in segs_raw if s.get("video_fake_prob") is not None]
    a_scores = [s["audio_fake_prob"] for s in segs_raw if s.get("audio_fake_prob") is not None]
    audio_integrity = round(1.0 - (sum(a_scores) / len(a_scores)), 3) if a_scores else None
    # "face confidence" = 1 - mean visual fake prob (UI wants high=authentic looking)
    face_confidence = round(1.0 - (sum(v_scores) / len(v_scores)), 3) if v_scores else None

    duration = float(segs_raw[-1]["t_end"]) if segs_raw else 0.0

    return {
        "verdict":          verdict,
        "fake_prob":        round(mean_prob, 4),
        "threshold":        round(float(threshold), 3),
        "flagged":          len(flagged_segs),
        "total":            len(segs_raw),
        "audio_integrity":  audio_integrity,
        "face_confidence":  face_confidence,
        "duration":         duration,
        "duration_ts":      _fmt_ts(duration),
        "device":           result.get("device", "cpu"),
        "window_s":         result.get("segment_seconds", 1.0),
        "segments":         out_segments,
        "timeline":         timeline,
        "faces":            face_entries,
        "video":            Path(result.get("video", "")).name,
        "model":            "ViT-forgery + wav2vec2-LoRA (late fusion)",
        "backend":          "PyTorch · " + str(result.get("device", "cpu")),
        "datasets":         "FaceForensics++ · ASVspoof 2019 LA",
    }


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Verity · Segment-Level Deepfake Detection")

# Allow the static frontend served on the same origin; also permissive for dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=500, detail=f"Missing {idx}. Copy index.html into static/")
    return FileResponse(idx)


@app.get("/api/health")
def health():
    return {"status": "ok", "device": get_device(), "pipeline_loaded": _PIPE is not None}


@app.get("/api/face/{face_id}")
def face_crop(face_id: str):
    path = _FACE_INDEX.get(face_id)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="face not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    url: str = Form(""),
    threshold: float = Form(0.30),
    window: float = Form(1.0),
    video_weight: float = Form(0.8),
    audio_weight: float = Form(0.2),
    frames_per_segment: int = Form(5),
):
    url = (url or "").strip()
    tmp_local: Optional[Path] = None

    # ---- resolve input ---------------------------------------------------- #
    try:
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix or ".mp4"
            fd, path = tempfile.mkstemp(prefix="verity_up_", suffix=suffix)
            tmp_local = Path(path)
            with open(fd, "wb") as out:
                shutil.copyfileobj(file.file, out)
            source = str(tmp_local)
        elif url:
            if not is_url(url):
                raise HTTPException(status_code=400, detail=f"Not a URL: {url}")
            logger.info("Downloading via yt-dlp: %s", url)
            source = download_media_from_url(url)
        else:
            raise HTTPException(status_code=400, detail="No file or url provided")

        # ---- normalize ---------------------------------------------------- #
        source = normalize_video_for_pipeline(source)

        # ---- run pipeline ------------------------------------------------- #
        crops_dir = Path(tempfile.mkdtemp(prefix="verity_crops_", dir=str(CROPS_ROOT)))
        pipe = _get_pipeline(threshold, video_weight, audio_weight, frames_per_segment)
        # Override window size if caller asked for one
        pipe.cfg.segment_seconds = float(window)
        pipe.cfg.segment_stride_seconds = float(window)

        result = pipe.run(source, progress=False, face_crops_dir=crops_dir)
        payload = _shape_response(result, threshold)
        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {e}"},
        )
    finally:
        # We keep the uploaded file around so face crops in /api/face/* stay valid.
        # Periodic cleanup can be added later if needed.
        pass


# Serve any extra static assets (screenshots, favicon, etc.) under /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
