"""
FastAPI back-end for the modern selective sound-removal web app.

Serves the single-page front-end from ``./static`` and exposes a small JSON
API: model discovery, file upload, class detection, and per-source
processing. All heavy lifting lives in :mod:`audio_engine`; this module only
handles HTTP, temporary-file bookkeeping, and (for video uploads) muxing the
cleaned audio back over the original video track with ffmpeg.

Run directly for local development::

    uvicorn server:app --reload

or via ``launch.py`` for a public share link.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_engine as engine  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Per-process scratch space. Uploads survive between the detect and process
# calls; results are served back by token. The OS clears the temp dir between
# reboots, which is the lifetime we want for a stateless demo.
WORK_DIR = Path(tempfile.gettempdir()) / "snc_modern_web"
UPLOAD_DIR = WORK_DIR / "uploads"
RESULT_DIR = WORK_DIR / "results"
for _d in (UPLOAD_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Selective Sound Removal")
# Permissive CORS so the front-end can also be hosted on a different origin
# (e.g. a static host pointing at a remote API) without extra wiring.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class DetectReq(BaseModel):
    file_id: str
    model: str


class SoundSel(BaseModel):
    name: str
    strength: float = 1.0


class ProcessReq(BaseModel):
    file_id: str
    model: str
    sounds: list[SoundSel]


# ---------------------------------------------------------------------------
# File bookkeeping helpers
# ---------------------------------------------------------------------------
def _upload_path(file_id: str) -> Path:
    """Resolve a stored upload by id, rejecting traversal and misses."""
    safe = Path(file_id).name
    matches = list(UPLOAD_DIR.glob(f"{safe}.*")) + list(UPLOAD_DIR.glob(safe))
    if not matches:
        raise HTTPException(404, "Upload not found or expired — re-upload it.")
    return matches[0]


def _duration(path: Path):
    """Media duration in seconds via ffprobe, or ``None`` if it cannot be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True)
        return round(float(out.stdout.strip()), 2)
    except Exception:  # noqa: BLE001
        return None


def _mux_video(src: Path, clean_wav: Path, token: str):
    """Mux cleaned audio over the source's video track; ``None`` if no video."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True)
    if probe.stdout.strip() != "video":
        return None
    out = RESULT_DIR / f"{token}_clean.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-i", str(clean_wav), "-c:v", "copy", "-map", "0:v:0", "-map",
         "1:a:0", "-shortest", str(out)], check=True)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def list_models():
    models = engine.available_models()
    if not models:
        raise HTTPException(
            404, "No trained models found under saved_models/separation_models/.")
    return {"models": models}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    file_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{file_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {
        "file_id": file_id,
        "filename": file.filename,
        "is_video": suffix in engine.VIDEO_EXT,
        "duration": _duration(dest),
    }


@app.post("/api/detect")
def detect(req: DetectReq):
    path = _upload_path(req.file_id)
    try:
        return engine.detect(str(path), req.model)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/process")
def process(req: ProcessReq):
    if not req.sounds:
        raise HTTPException(400, "Select at least one sound to reduce or remove.")
    path = _upload_path(req.file_id)
    sounds = [{"name": s.name, "strength": s.strength} for s in req.sounds]
    try:
        original, cleaned, stems = engine.process(str(path), req.model, sounds)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

    token = uuid.uuid4().hex
    original_path = RESULT_DIR / f"{token}_original.wav"
    clean_path = RESULT_DIR / f"{token}_clean.wav"
    sf.write(original_path, original, engine.SAMPLE_RATE)
    sf.write(clean_path, cleaned, engine.SAMPLE_RATE)

    stem_items = []
    for i, (name, wave) in enumerate(stems):
        stem_path = RESULT_DIR / f"{token}_stem{i}.wav"
        sf.write(stem_path, wave, engine.SAMPLE_RATE)
        stem_items.append({"name": name, "url": f"/api/file/{stem_path.name}"})

    video_url = None
    if path.suffix.lower() in engine.VIDEO_EXT:
        muxed = _mux_video(path, clean_path, token)
        if muxed is not None:
            video_url = f"/api/file/{muxed.name}"

    return {
        "original_url": f"/api/file/{original_path.name}",
        "clean_url": f"/api/file/{clean_path.name}",
        "video_url": video_url,
        "stems": stem_items,
    }


@app.get("/api/file/{name}")
def get_file(name: str):
    path = RESULT_DIR / Path(name).name  # .name strips any traversal
    if not path.exists():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(path)


# Mount static assets last so it never shadows the API routes above.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
