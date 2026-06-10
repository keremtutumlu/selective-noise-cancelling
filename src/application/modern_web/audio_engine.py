"""
Audio/separation engine for the modern web front-end.

A thin, framework-agnostic layer over the trained FiLM-conditioned separator:
model discovery, lazy loading/caching, class detection, and per-source
processing where every detected source carries its *own* strength
(0.0 = keep untouched, 1.0 = full removal). The HTTP layer talks only to this
module and never to Keras directly, so the API stays free of TensorFlow
specifics.

The DSP mirrors the proven pipeline used by the classic app — a single
full-file STFT, 75 %-overlap masking, and a phase-preserving ISTFT — extended
so each selected source is attenuated independently instead of sharing one
global strength. That is what lets the UI offer a separate "keep / reduce /
remove" choice for every sound.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import tensorflow as tf

# Repo root: this file lives at src/application/modern_web/audio_engine.py
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
import model_config as cfg  # noqa: E402
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, SAMPLE_RATE, TIME_FRAMES,
)

MODELS_DIR = BASE_DIR / "saved_models" / "separation_models"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# 5-frame (~40 ms) smoother to suppress musical noise on the mask.
_SMOOTH_KERNEL = np.ones(5, dtype=np.float32) / 5.0

# Lazy model cache. Loading a Keras .h5 costs several seconds, so each
# checkpoint is loaded once on first use and shared across requests. The lock
# stops two concurrent requests from loading the same model twice.
_models: dict = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model discovery and loading
# ---------------------------------------------------------------------------
def _model_pairs() -> list[Path]:
    """Every checkpoint that has a matching ``_classes.json`` sidecar."""
    return [h5 for h5 in sorted(MODELS_DIR.glob("separator_unet_film_multi_*.h5"))
            if h5.with_name(h5.stem + "_classes.json").exists()]


def _default_model_name() -> Optional[str]:
    """Most recently modified checkpoint — the sensible default selection."""
    pairs = _model_pairs()
    if not pairs:
        return None
    return max(pairs, key=lambda p: p.stat().st_mtime).stem


def available_models() -> list[dict]:
    """List selectable checkpoints with their class count, default flagged.

    ``has_detection_head`` is left out here on purpose: determining it means
    loading the model, which is wasteful just to populate a dropdown. It is
    reported by :func:`detect`, where the model is loaded anyway.
    """
    default = _default_model_name()
    out = []
    for h5 in _model_pairs():
        try:
            classes = json.load(h5.with_name(h5.stem + "_classes.json").open())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "name": h5.stem,
            "num_classes": len(classes),
            "is_default": h5.stem == default,
        })
    return out


def _load_allowlist(name: str) -> Optional[list]:
    """Detection allow-list next to checkpoint ``name``, or ``None``.

    When present, detection only ranks/surfaces these classes; removal can
    still target any class in the model vocabulary. Restricting the candidate
    pool also cuts detection latency, since a forward pass runs per candidate.
    """
    path = MODELS_DIR / f"{name}_detect_allowlist.json"
    if not path.exists():
        return None
    try:
        names = json.load(path.open())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(names, list) or not names:
        return None
    return [str(n) for n in names]


def get_model(name: str) -> dict:
    """Return the cached entry for ``name``, loading it on first use."""
    with _lock:
        if name in _models:
            return _models[name]
        h5 = MODELS_DIR / f"{name}.h5"
        classes_path = MODELS_DIR / f"{name}_classes.json"
        if not h5.exists() or not classes_path.exists():
            raise FileNotFoundError(
                f"Model '{name}' not found under {MODELS_DIR}.")
        model = tf.keras.models.load_model(h5, compile=False)
        entry = {
            "model": model,
            "class_names": json.load(classes_path.open()),
            "allowed": _load_allowlist(name),
            "has_detection_head": len(model.outputs) > 1,
        }
        _models[name] = entry
        return entry


# ---------------------------------------------------------------------------
# Signal helpers (shared contract with the rest of the pipeline)
# ---------------------------------------------------------------------------
def _load_audio(path: str) -> np.ndarray:
    """Decode any audio/video file to a 16 kHz mono waveform via ffmpeg."""
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), tmp], check=True)
    audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
    return audio


def _to_windows(audio: np.ndarray) -> list:
    """Split into consecutive 1-second windows, zero-padding the tail."""
    n = max(1, int(np.ceil(len(audio) / SAMPLE_RATE)))
    padded = np.pad(audio, (0, n * SAMPLE_RATE - len(audio)))
    return [padded[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE] for i in range(n)]


def _spectrogram(window: np.ndarray):
    """Half-band linear magnitude ``(FREQ_BINS, TIME_FRAMES)`` for one window."""
    stft = librosa.stft(window, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)[:FREQ_BINS]
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return mag


def _query(class_names: list, name: str) -> np.ndarray:
    q = np.zeros(len(class_names), np.float32)
    q[class_names.index(name)] = 1.0
    return q


def _istft_with_phase(mag, phase_stft, length, original_peak):
    """Inverse-STFT a half-band magnitude using a reference complex STFT."""
    full = np.vstack([mag, np.zeros((1, mag.shape[1]), dtype=np.float32)])
    wave = librosa.istft(full * np.exp(1j * np.angle(phase_stft)),
                         hop_length=HOP_LENGTH, n_fft=N_FFT, length=length)
    wave = wave * original_peak
    peak = np.max(np.abs(wave))
    if peak > 1.0:
        wave = wave / peak
    return wave.astype(np.float32)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect(file_path: str, model_name: str, *, rel_cap: float = 0.80,
           top_k: int = 5, max_seconds: int = 8) -> dict:
    """Score each candidate class and return the ones that look present.

    For models with a detection head the per-class sigmoid probability is the
    primary score. For older mask-energy models the score is
    ``energy_ratio × (1 + specificity²)``. In both cases the threshold is
    ``max(0.05, rel_cap × top_score)`` — relative cap matches the eval script.

    Returns ``detected`` (cards the UI offers for removal) plus a longer
    ``ranked`` list (the bottom "separated sources" panel) and the
    ``has_detection_head`` flag so the UI can label the scoring mode.
    """
    entry = get_model(model_name)
    model = entry["model"]
    class_names = entry["class_names"]
    has_det = entry["has_detection_head"]
    allowed = entry["allowed"]
    allowed_set = None if allowed is None else set(allowed)
    candidates = [c for c in class_names
                  if allowed_set is None or c in allowed_set]

    audio = _load_audio(file_path)
    windows = _to_windows(audio)[:max_seconds]
    logs, lins = [], []
    for w in windows:
        # Match training: SeparationMixer peak-normalises every 1-second
        # mixture to |peak|=1.0, so the model only ever sees magnitudes in
        # that scale. Raw user audio is often much quieter.
        peak = float(np.max(np.abs(w)))
        if peak > 1e-6:
            w = w / peak
        lin = _spectrogram(w)[..., None]
        lins.append(lin)
        logs.append(np.log1p(lin))
    logs, lins = np.array(logs), np.array(lins)
    mix_energy = float(np.mean(lins ** 2)) + 1e-8

    scores: dict = {}
    energy_scores: dict = {}
    for name in candidates:
        q = np.tile(_query(class_names, name), (len(windows), 1))
        preds = model.predict([logs, q, lins], verbose=0)
        if has_det:
            stem, presence = preds[0], preds[1]
            scores[name] = float(np.mean(presence))
            energy_ratio = float(np.mean(stem ** 2) / mix_energy)
            specificity = float(np.std(stem) / (np.mean(stem) + 1e-8))
            energy_scores[name] = energy_ratio * (1.0 + specificity ** 2)
        else:
            est = preds
            energy_ratio = float(np.mean(est ** 2) / mix_energy)
            specificity = float(np.std(est) / (np.mean(est) + 1e-8))
            scores[name] = energy_ratio * (1.0 + specificity ** 2)

    # If the detection head produces a degenerate (near-uniform) score
    # distribution the head is undertrained — fall back to mask-energy scoring.
    # The v2.7 model is a known case: focal-loss weight collapse during training
    # left the head outputting probabilities ~0.5 regardless of audio content.
    if has_det and energy_scores:
        score_vals = np.array(list(scores.values()), dtype=np.float32)
        if float(score_vals.max() - score_vals.min()) < 0.15:
            scores = energy_scores
            has_det = False

    ranked = sorted(scores, key=scores.get, reverse=True)
    if not ranked:
        return {"has_detection_head": has_det, "detected": [], "ranked": []}

    cutoff = max(0.05, rel_cap * scores[ranked[0]])
    detected = [n for n in ranked if scores[n] >= cutoff][:top_k]

    def pack(names):
        return [{"name": n, "score": round(float(scores[n]), 4)} for n in names]

    return {
        "has_detection_head": has_det,
        "detected": pack(detected),
        "ranked": pack(ranked[:15]),
    }


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------
def process(file_path: str, model_name: str, sounds: list) -> tuple:
    """Attenuate each source by its own strength; return audio + per-source stems.

    ``sounds`` is a list of ``{"name": str, "strength": float}`` where strength
    is clamped to ``[0, 1]`` — 0 leaves the source untouched, 1 removes it
    fully, values in between reduce it. Every source also yields an extracted
    stem (the model's isolated estimate) for the verification panel.

    Returns ``(original_waveform, cleaned_waveform, [(name, stem_waveform), ...])``
    all at :data:`SAMPLE_RATE`.
    """
    entry = get_model(model_name)
    model = entry["model"]
    class_names = entry["class_names"]
    names = [s["name"] for s in sounds]
    strengths = np.clip(
        np.array([float(s.get("strength", 1.0)) for s in sounds], np.float32),
        0.0, 1.0)

    audio = _load_audio(file_path)
    # Time-domain peak normalisation matches the training input scale; the same
    # peak is reapplied after the ISTFT so output loudness tracks the input.
    audio_peak = float(np.max(np.abs(audio))) + 1e-8
    audio_norm = audio / audio_peak
    queries = np.stack([_query(class_names, n) for n in names])

    full_stft = librosa.stft(audio_norm, n_fft=N_FFT, hop_length=HOP_LENGTH)
    full_mag = np.abs(full_stft)[:FREQ_BINS].astype(np.float32)
    T = full_mag.shape[1]

    out_acc = np.zeros((FREQ_BINS, T), dtype=np.float32)
    weight_acc = np.zeros(T, dtype=np.float32)
    extract_acc = [np.zeros((FREQ_BINS, T), dtype=np.float32) for _ in names]
    hann = np.hanning(TIME_FRAMES).astype(np.float32)
    step = TIME_FRAMES // 4  # 32 frames ≈ 0.25 s → 75 % overlap

    for start in range(0, T, step):
        end = min(start + TIME_FRAMES, T)
        actual = end - start
        chunk = full_mag[:, start:end]
        if actual < TIME_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TIME_FRAMES - actual)))

        lin = chunk[..., None]
        log = np.log1p(lin)
        preds = model.predict(
            [np.repeat(log[None], len(names), 0), queries,
             np.repeat(lin[None], len(names), 0)], verbose=0)
        est = preds[0] if isinstance(preds, list) else preds

        combined_mask = np.ones((FREQ_BINS, TIME_FRAMES), dtype=np.float32)
        for k in range(len(names)):
            est_mag = est[k, :, :, 0]
            # Amplitude ratio is more responsive than power ratio when the
            # model outputs conservative (small) magnitude estimates.
            amplitude_ratio = np.clip(est_mag / (chunk + 1e-8), 0.0, 1.0)
            combined_mask *= np.clip(1.0 - strengths[k] * amplitude_ratio,
                                     0.0, 1.0)
            extract_acc[k][:, start:end] += hann[:actual] * est_mag[:, :actual]

        for i in range(FREQ_BINS):
            combined_mask[i] = np.convolve(combined_mask[i], _SMOOTH_KERNEL,
                                           mode="same")
        combined_mask = np.clip(combined_mask, 0.0, 1.0)

        processed = chunk * combined_mask
        out_acc[:, start:end] += hann[:actual] * processed[:, :actual]
        weight_acc[start:end] += hann[:actual]

    out_mag = out_acc / np.maximum(weight_acc, 1e-8)
    cleaned = _istft_with_phase(out_mag, full_stft, len(audio_norm), audio_peak)

    stems = []
    for k in range(len(names)):
        ext_mag = extract_acc[k] / np.maximum(weight_acc, 1e-8)
        stems.append((names[k],
                      _istft_with_phase(ext_mag, full_stft, len(audio_norm),
                                        audio_peak)))
    return audio, cleaned, stems
