"""
Gradio web app for conditioned selective sound removal.

Upload an audio or video file; the app detects which sound classes are
present, lets you pick which to remove (and how strongly), and shows
the before/after result as playable audio (and video, for video input).

Launch (e.g. on Colab — prints a public share link):
    python src/application/webapp.py

Prerequisites:
    * saved_models/separation_models/separator_unet_film_multi_v2.0.h5
    * saved_models/separation_models/separator_unet_film_multi_v2.0_classes.json
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, SAMPLE_RATE, TIME_FRAMES,
)

_MODELS = BASE_DIR / "saved_models" / "separation_models"
_model = tf.keras.models.load_model(
    _MODELS / "separator_unet_film_multi_v2.0.h5", compile=False)
_class_names = json.load(
    (_MODELS / "separator_unet_film_multi_v2.0_classes.json").open())
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _load_audio(path: str) -> np.ndarray:
    """Decode any audio/video file to a 16 kHz mono waveform via ffmpeg."""
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(SAMPLE_RATE), tmp], check=True)
    audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
    return audio


def _to_windows(audio: np.ndarray):
    n = int(np.ceil(len(audio) / SAMPLE_RATE))
    padded = np.pad(audio, (0, n * SAMPLE_RATE - len(audio)))
    return [padded[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE] for i in range(n)]


def _spectrograms(window: np.ndarray):
    stft = librosa.stft(window, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)[:FREQ_BINS]
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return stft, mag


def _query(name: str) -> np.ndarray:
    q = np.zeros(len(_class_names), np.float32)
    q[_class_names.index(name)] = 1.0
    return q


def detect_sounds(file_path):
    """Detect present classes using U-Net stem energy + mask specificity score."""
    if not file_path:
        return gr.update(choices=[], value=[]), None
    audio = _load_audio(file_path)
    windows = _to_windows(audio)[:8]  # sample up to 8 s for speed
    logs, lins = [], []
    for w in windows:
        _, mag = _spectrograms(w)
        lin = mag[..., None]
        lins.append(lin)
        logs.append(np.log1p(lin))
    logs, lins = np.array(logs), np.array(lins)

    mix_energy = float(np.mean(lins ** 2)) + 1e-8
    scores = {}
    for name in _class_names:
        q = np.tile(_query(name), (len(windows), 1))
        est = _model.predict([logs, q, lins], verbose=0)
        energy_ratio = float(np.mean(est ** 2) / mix_energy)
        # Coefficient of variation: real sounds produce concentrated masks
        # (high std); broadband noise produces diffuse, uniform masks (low std).
        specificity = float(np.std(est) / (np.mean(est) + 1e-8))
        scores[name] = energy_ratio * (1.0 + specificity)

    ranked = sorted(scores, key=scores.get, reverse=True)
    # Relative cap (0.40) keeps the gap tight; floor is kept low so a
    # conservative model (high negative_prob) still surfaces real classes.
    cutoff = max(0.05, 0.40 * scores[ranked[0]])
    detected = [n for n in ranked if scores[n] >= cutoff][:10]
    return gr.update(choices=detected, value=[]), (SAMPLE_RATE, audio)


_SMOOTH_KERNEL = np.ones(5, dtype=np.float32) / 5.0  # 5-frame (~40 ms) mask smoother


def remove_sounds(file_path, selected, strength):
    """Attenuate selected classes with power-ratio masking and spectrogram OLA."""
    if not file_path or not selected:
        return None, None, None
    audio = _load_audio(file_path)
    queries = np.stack([_query(n) for n in selected])

    # One STFT for the whole file — phase stays globally consistent.
    full_stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    full_mag = np.abs(full_stft)[:FREQ_BINS].astype(np.float32)  # (FREQ_BINS, T)
    T = full_mag.shape[1]

    # Spectrogram-level 50 % overlap-add accumulators.
    out_acc = np.zeros((FREQ_BINS, T), dtype=np.float32)
    weight_acc = np.zeros(T, dtype=np.float32)
    hann = np.hanning(TIME_FRAMES).astype(np.float32)
    step = TIME_FRAMES // 2  # 64 frames ≈ 0.5 s

    for start in range(0, T, step):
        end = min(start + TIME_FRAMES, T)
        actual = end - start
        chunk = full_mag[:, start:end]
        if actual < TIME_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TIME_FRAMES - actual)))

        lin = chunk[..., None]
        log = np.log1p(lin)
        est = _model.predict(
            [np.repeat(log[None], len(selected), 0),
             queries,
             np.repeat(lin[None], len(selected), 0)], verbose=0)

        # Multiplicative amplitude-ratio masks across all selected classes.
        # Amplitude ratio (est/mix) is more responsive than power ratio (est²/mix²)
        # when the model outputs conservative (small) magnitude estimates.
        combined_mask = np.ones((FREQ_BINS, TIME_FRAMES), dtype=np.float32)
        for k in range(len(selected)):
            est_mag = est[k, :, :, 0]
            amplitude_ratio = np.clip(est_mag / (chunk + 1e-8), 0.0, 1.0)
            combined_mask *= np.clip(1.0 - strength * amplitude_ratio, 0.0, 1.0)

        # Temporal smoothing along the frame axis to suppress musical noise.
        for i in range(FREQ_BINS):
            combined_mask[i] = np.convolve(combined_mask[i], _SMOOTH_KERNEL,
                                           mode="same")
        combined_mask = np.clip(combined_mask, 0.0, 1.0)

        processed = chunk * combined_mask  # (FREQ_BINS, TIME_FRAMES)

        # Hanning-weighted overlap-add (time axis only).
        out_acc[:, start:end] += hann[:actual] * processed[:, :actual]
        weight_acc[start:end] += hann[:actual]

    # Normalise, restore Nyquist row, reconstruct with original full-file phase.
    out_mag = out_acc / np.maximum(weight_acc, 1e-8)
    out_full = np.vstack([out_mag, np.zeros((1, T), dtype=np.float32)])
    out = librosa.istft(out_full * np.exp(1j * np.angle(full_stft)),
                        hop_length=HOP_LENGTH, n_fft=N_FFT, length=len(audio))

    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak

    out_video = None
    if Path(file_path).suffix.lower() in _VIDEO_EXT:
        cleaned = tempfile.mktemp(suffix=".wav")
        sf.write(cleaned, out, SAMPLE_RATE)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", file_path],
            capture_output=True, text=True,
        )
        has_video = probe.stdout.strip() == "video"
        if has_video:
            out_video = tempfile.mktemp(suffix=".mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-i", file_path, "-i", cleaned,
                            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                            "-shortest", out_video], check=True)

    return (SAMPLE_RATE, audio), (SAMPLE_RATE, out), out_video


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Selective Sound Removal") as demo:
        gr.Markdown(
            "# Selective Sound Removal\n"
            "Upload an audio or video file, detect the sounds in it, and "
            "remove the ones you choose."
        )
        file_in = gr.File(label="Upload audio or video", type="filepath")
        analyze_btn = gr.Button("1. Analyze sounds", variant="primary")

        detected = gr.CheckboxGroup(
            label="2. Detected sounds — tick the ones to remove")
        strength = gr.Slider(0.0, 1.0, value=1.0, step=0.1,
                             label="Removal strength (1.0 = full removal, "
                                   "0.5 = quieter)")
        process_btn = gr.Button("3. Remove selected sounds", variant="primary")

        with gr.Row():
            before = gr.Audio(label="Before")
            after = gr.Audio(label="After")
        after_video = gr.Video(label="After (video)")

        analyze_btn.click(detect_sounds, inputs=file_in,
                          outputs=[detected, before])
        process_btn.click(remove_sounds,
                          inputs=[file_in, detected, strength],
                          outputs=[before, after, after_video])
    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
