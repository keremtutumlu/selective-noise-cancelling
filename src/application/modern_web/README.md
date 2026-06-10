# Modern Web App

A custom single-page front-end (FastAPI + vanilla JS) for the query-conditioned
separator. It replaces the Gradio interface with a faster, fully styled UI and
**per-source control**: every detected sound gets its own *Keep / Reduce /
Remove* choice instead of one global removal strength.

The classic Gradio app (`src/application/webapp.py`) is untouched and still
works — this lives alongside it.

## What it does

1. **Choose a model** — auto-discovers every `separator_unet_film_multi_*.h5`
   in `saved_models/separation_models/` and lists it in a dropdown.
2. **Upload audio or video** — drag-and-drop, with a live preview. Video keeps
   its picture; only the audio track is cleaned and re-muxed.
3. **Detected sounds** — each source shows a confidence bar and a *Keep /
   Reduce / Remove* control. *Reduce* exposes a strength slider (10–90 %).
   *Remove* is full attenuation. Quick "set all" chips are provided.
4. **Result** — before/after players side by side, the cleaned video (if any),
   and download buttons.
5. **Separated sources** — at the bottom, each source the model isolated is
   shown with its own player plus the raw detection scores. This panel is a
   preview surface intended to move behind an admin account later.

## Run locally

```bash
source ../../../venv/bin/activate   # the project venv (TensorFlow, librosa, …)
pip install fastapi "uvicorn[standard]" python-multipart
python launch.py                    # http://localhost:8000
```

`ffmpeg`/`ffprobe` must be on `PATH`, and at least one trained checkpoint must
exist under `saved_models/separation_models/`.

## Publish a public link (the quick path)

```bash
python launch.py --share
```

This opens a **Cloudflare Quick Tunnel** and prints a public
`https://<random>.trycloudflare.com` URL — no account or token needed. It is
the recommended way to demo from Colab, where the checkpoints already sit on
the mounted Drive. The Colab notebook `notebooks/colab_modern_webapp.ipynb`
wraps this in three cells.

Prefer ngrok? Set `NGROK_AUTHTOKEN` and `pip install pyngrok`; `launch.py`
will use it automatically.

## Publish with Docker (longer-lived)

Build from the repository root so the model code and checkpoints are in
context:

```bash
docker build -f src/application/modern_web/Dockerfile -t snc-web .
docker run --rm -p 8000:8000 snc-web
```

The same image runs on container hosts that expose a port (e.g. a Hugging Face
Space using the Docker SDK on port 8000). Checkpoints must be present in the
build context because they are gitignored.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/api/models`  | — | available checkpoints |
| POST | `/api/upload`  | multipart `file` | `file_id`, `is_video`, `duration` |
| POST | `/api/detect`  | `{file_id, model}` | detected + ranked sources |
| POST | `/api/process` | `{file_id, model, sounds:[{name, strength}]}` | cleaned audio/video + per-source stems |
| GET  | `/api/file/{name}` | — | a rendered result file |

`strength` is `0.0`–`1.0` per source: `0` keeps it, `1` removes it, values in
between reduce it.

## Files

| File | Role |
|---|---|
| `server.py` | FastAPI routes, file bookkeeping, video muxing |
| `audio_engine.py` | model loading, detection, per-source masking DSP |
| `launch.py` | uvicorn launcher + cloudflared/ngrok tunnel |
| `static/index.html` | page structure |
| `static/styles.css` | theme |
| `static/app.js` | front-end controller |
| `Dockerfile` | container build (from repo root) |
