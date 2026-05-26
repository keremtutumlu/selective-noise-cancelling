"""
Gradio web app for conditioned selective sound removal.

Upload an audio or video file; the app detects which sound classes are
present, lets you pick which to remove (and how strongly), and shows
the before/after result as playable audio (and video, for video input).

Each selected class also produces an *extracted stem* — the model's
estimate of what it identified as that class — so you can audibly verify
the detection before trusting the cleaned output.

Launch (e.g. on Colab — prints a public share link):
    python src/application/webapp.py

The app auto-discovers every trained checkpoint in
``saved_models/separation_models/`` (any ``separator_unet_film_multi_*.h5``
with a matching ``*_classes.json``) and exposes them in a dropdown so the
automation sweep's outputs can be compared by ear without editing code.
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

_MODELS_DIR = BASE_DIR / "saved_models" / "separation_models"
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# In-process model cache. Loading a Keras .h5 is slow (~5-10 s on T4) so we
# cache each model the first time the user selects it from the dropdown.
_loaded: dict = {}
_current: dict = {"name": None, "model": None, "class_names": None}


def _available_models() -> list[tuple[str, Path]]:
    """Every .h5 in the model dir that has a matching _classes.json."""
    pairs = []
    for h5 in sorted(_MODELS_DIR.glob("separator_unet_film_multi_*.h5")):
        names_path = h5.with_name(h5.stem + "_classes.json")
        if names_path.exists():
            pairs.append((h5.stem, h5))
    return pairs


def _load_model(name: str) -> None:
    """Load ``name`` into the cache and mark it current."""
    if name in _loaded:
        _current.update(name=name, **_loaded[name])
        return
    h5 = _MODELS_DIR / f"{name}.h5"
    classes = _MODELS_DIR / f"{name}_classes.json"
    model = tf.keras.models.load_model(h5, compile=False)
    class_names = json.load(classes.open())
    _loaded[name] = {"model": model, "class_names": class_names}
    _current.update(name=name, model=model, class_names=class_names)
    print(f"Loaded model '{name}' ({len(class_names)} classes).")


# Pre-load the most recently modified model so first detection is fast.
_initial = _available_models()
if not _initial:
    raise FileNotFoundError(
        f"No trained models found under {_MODELS_DIR}. Expected at least one "
        f"'separator_unet_film_multi_*.h5' with a matching '*_classes.json'.")
_default_name = max(_initial, key=lambda p: p[1].stat().st_mtime)[0]
_load_model(_default_name)

# Detection surfaces at most 10 classes, so 10 extracted-stem slots is the
# tightest bound that still covers any selection the user can make.
MAX_EXTRACTED_SLOTS = 10


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
    class_names = _current["class_names"]
    q = np.zeros(len(class_names), np.float32)
    q[class_names.index(name)] = 1.0
    return q


def _empty_extracted_slots():
    """Reset every extracted-stem slot to hidden and empty."""
    return [gr.update(value=None, visible=False)
            for _ in range(MAX_EXTRACTED_SLOTS)]


def detect_sounds(file_path):
    """Detect present classes using U-Net stem energy + mask specificity score."""
    cleared = _empty_extracted_slots()
    if not file_path:
        return [gr.update(choices=[], value=[]), None, None, None] + cleared
    audio = _load_audio(file_path)
    windows = _to_windows(audio)[:8]  # sample up to 8 s for speed
    logs, lins = [], []
    for w in windows:
        # Match training: SeparationMixer peak-normalises each 1-sec mixture
        # to time-domain |peak|=1.0, so the model only ever sees STFT mags
        # in that scale. Raw user audio is often much quieter, which throws
        # the conditioned U-Net into an untrained input regime.
        peak = float(np.max(np.abs(w)))
        if peak > 1e-6:
            w = w / peak
        _, mag = _spectrograms(w)
        lin = mag[..., None]
        lins.append(lin)
        logs.append(np.log1p(lin))
    logs, lins = np.array(logs), np.array(lins)

    mix_energy = float(np.mean(lins ** 2)) + 1e-8
    scores = {}
    model = _current["model"]
    for name in _current["class_names"]:
        q = np.tile(_query(name), (len(windows), 1))
        est = model.predict([logs, q, lins], verbose=0)
        energy_ratio = float(np.mean(est ** 2) / mix_energy)
        # CoV²: real sounds produce concentrated masks (high CoV); broadband
        # noise produces diffuse uniform masks (low CoV). Squaring sharpens
        # the gap between specific and diffuse detections.
        specificity = float(np.std(est) / (np.mean(est) + 1e-8))
        scores[name] = energy_ratio * (1.0 + specificity ** 2)

    ranked = sorted(scores, key=scores.get, reverse=True)
    # Relative cap at 0.65 of the winner (tighter than the old 0.40) to
    # reduce false positives; absolute floor keeps very quiet but real
    # classes reachable.
    cutoff = max(0.05, 0.65 * scores[ranked[0]])
    detected = [n for n in ranked if scores[n] >= cutoff][:10]
    return ([gr.update(choices=detected, value=[]),
             (SAMPLE_RATE, audio), None, None] + cleared)


_SMOOTH_KERNEL = np.ones(5, dtype=np.float32) / 5.0  # 5-frame (~40 ms) mask smoother


def _istft_with_phase(mag, phase_stft, length, original_peak):
    """Inverse-STFT a half-band magnitude using a reference complex STFT."""
    full = np.vstack([mag, np.zeros((1, mag.shape[1]), dtype=np.float32)])
    wave = librosa.istft(full * np.exp(1j * np.angle(phase_stft)),
                         hop_length=HOP_LENGTH, n_fft=N_FFT, length=length)
    wave = wave * original_peak
    peak = np.max(np.abs(wave))
    if peak > 1.0:
        wave = wave / peak
    return wave


def remove_sounds(file_path, selected, strength):
    """Attenuate selected classes and emit per-class extracted stems for
    audible verification of what each class was detected as."""
    cleared = _empty_extracted_slots()
    if not file_path or not selected:
        return [None, None, None] + cleared
    audio = _load_audio(file_path)

    # Time-domain peak normalisation: SeparationMixer normalises every 1-sec
    # mixture to |peak|=1.0 before computing the STFT.  Applying the same
    # normalisation once on the whole file is the closest match we can achieve
    # without re-doing per-window STFT (which would break the global OLA).
    audio_peak = float(np.max(np.abs(audio))) + 1e-8
    audio_norm = audio / audio_peak

    queries = np.stack([_query(n) for n in selected])

    # One STFT for the whole file — phase stays globally consistent.
    full_stft = librosa.stft(audio_norm, n_fft=N_FFT, hop_length=HOP_LENGTH)
    full_mag = np.abs(full_stft)[:FREQ_BINS].astype(np.float32)  # (FREQ_BINS, T)
    T = full_mag.shape[1]

    # Spectrogram-level 50 % overlap-add accumulators.
    out_acc = np.zeros((FREQ_BINS, T), dtype=np.float32)
    weight_acc = np.zeros(T, dtype=np.float32)
    # One accumulator per selected class for the extracted-stem display.
    extract_acc = [np.zeros((FREQ_BINS, T), dtype=np.float32) for _ in selected]
    hann = np.hanning(TIME_FRAMES).astype(np.float32)
    step = TIME_FRAMES // 4  # 32 frames ≈ 0.25 s — 75% overlap reduces boundary artifacts

    for start in range(0, T, step):
        end = min(start + TIME_FRAMES, T)
        actual = end - start
        chunk = full_mag[:, start:end]
        if actual < TIME_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TIME_FRAMES - actual)))

        lin = chunk[..., None]
        log = np.log1p(lin)
        est = _current["model"].predict(
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
            # Bank the model's per-class stem estimate for playback.
            extract_acc[k][:, start:end] += hann[:actual] * est_mag[:, :actual]

        # Temporal smoothing along the frame axis to suppress musical noise.
        for i in range(FREQ_BINS):
            combined_mask[i] = np.convolve(combined_mask[i], _SMOOTH_KERNEL,
                                           mode="same")
        combined_mask = np.clip(combined_mask, 0.0, 1.0)

        processed = chunk * combined_mask  # (FREQ_BINS, TIME_FRAMES)

        # Hanning-weighted overlap-add (time axis only).
        out_acc[:, start:end] += hann[:actual] * processed[:, :actual]
        weight_acc[start:end] += hann[:actual]

    # Reconstruct cleaned audio.
    out_mag = out_acc / np.maximum(weight_acc, 1e-8)
    out = _istft_with_phase(out_mag, full_stft, len(audio_norm), audio_peak)

    # Reconstruct one extracted stem per selected class.
    extracted_waves = []
    for k in range(len(selected)):
        ext_mag = extract_acc[k] / np.maximum(weight_acc, 1e-8)
        extracted_waves.append(
            _istft_with_phase(ext_mag, full_stft, len(audio_norm), audio_peak))

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

    # Pack per-class extracts into the fixed UI slots.
    extracted_updates = []
    for i in range(MAX_EXTRACTED_SLOTS):
        if i < len(selected):
            extracted_updates.append(gr.update(
                value=(SAMPLE_RATE, extracted_waves[i]),
                label=f"Extracted as: {selected[i]}",
                visible=True))
        else:
            extracted_updates.append(gr.update(value=None, visible=False))

    return ([(SAMPLE_RATE, audio), (SAMPLE_RATE, out), out_video]
            + extracted_updates)


def _on_model_change(name: str) -> str:
    """Dropdown callback: load the selected checkpoint and report its class count."""
    _load_model(name)
    return f"Active model: **{name}** — {len(_current['class_names'])} classes."


def build_app() -> gr.Blocks:
    available = [name for name, _ in _available_models()]
    with gr.Blocks(title="Selective Sound Removal") as demo:
        gr.Markdown(
            "# Selective Sound Removal\n"
            "Upload an audio or video file, detect the sounds in it, and "
            "remove the ones you choose."
        )
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=available, value=_current["name"],
                label="Model", scale=3,
                info="Trained checkpoints in saved_models/separation_models/. "
                     "Switch to compare automation-sweep outputs.")
            model_status = gr.Markdown(
                f"Active model: **{_current['name']}** — "
                f"{len(_current['class_names'])} classes.")
        model_dd.change(_on_model_change, inputs=model_dd, outputs=model_status)

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

        gr.Markdown(
            "### Extracted stems — verify what was detected as what\n"
            "One player per selected class, showing the model's estimate of "
            "what it heard as that class in the mixture. If the wrong thing "
            "comes out of a stem, the cleaned audio will reflect that mistake."
        )
        extracted_slots = [
            gr.Audio(label=f"Slot {i + 1}", visible=False)
            for i in range(MAX_EXTRACTED_SLOTS)
        ]

        analyze_btn.click(
            detect_sounds, inputs=file_in,
            outputs=[detected, before, after, after_video] + extracted_slots,
        )
        process_btn.click(
            remove_sounds, inputs=[file_in, detected, strength],
            outputs=[before, after, after_video] + extracted_slots,
        )
    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
