"""
Test the conditioned separator on a real audio (or video) file from disk.

A non-interactive companion to webapp.py: prints the full ranked detection
scoreboard, optionally separates one or more named classes, and writes the
cleaned WAV next to the input file. Useful for quick checks on user audio
without launching Gradio.

Run:
    python src/model_training/test_realworld.py path/to/audio.wav
    python src/model_training/test_realworld.py path/to/audio.wav siren dog
    python src/model_training/test_realworld.py path/to/audio.wav --strength 0.7 siren
"""
import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, SAMPLE_RATE, TIME_FRAMES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ABSOLUTE_FLOOR = 0.05
RELATIVE_CAP = 0.40
SMOOTH_KERNEL = np.ones(5, dtype=np.float32) / 5.0


def _load_audio(path: Path) -> np.ndarray:
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), tmp], check=True)
    audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
    return audio


def _query(class_names, name):
    q = np.zeros(len(class_names), np.float32)
    q[class_names.index(name)] = 1.0
    return q


def _spectrogram(waveform):
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)[:FREQ_BINS]
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return mag


def detect(model, class_names, audio):
    """Score every class on the first 8 sec of audio, print ranked table."""
    n = int(np.ceil(len(audio) / SAMPLE_RATE))
    padded = np.pad(audio, (0, n * SAMPLE_RATE - len(audio)))
    windows = [padded[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE] for i in range(n)][:8]

    logs, lins = [], []
    for w in windows:
        peak = np.max(np.abs(w))
        if peak > 1e-6:
            w = w / peak
        mag = _spectrogram(w)
        lin = mag[..., None]
        lins.append(lin)
        logs.append(np.log1p(lin))
    logs, lins = np.array(logs), np.array(lins)
    mix_energy = float(np.mean(lins ** 2)) + 1e-8

    scores = {}
    for name in class_names:
        q = np.tile(_query(class_names, name), (len(windows), 1))
        est = model.predict([logs, q, lins], verbose=0)
        energy_ratio = float(np.mean(est ** 2) / mix_energy)
        specificity = float(np.std(est) / (np.mean(est) + 1e-8))
        scores[name] = energy_ratio * (1.0 + specificity)

    ranked = sorted(scores, key=scores.get, reverse=True)
    cutoff = max(ABSOLUTE_FLOOR, RELATIVE_CAP * scores[ranked[0]])
    detected = [n for n in ranked if scores[n] >= cutoff][:10]

    print("\n  Detection scoreboard (top 15)")
    print(f"  {'Rank':<6}{'Class':<22}{'Score':>10}  {'Surfaced?':>10}")
    print("  " + "-" * 50)
    for i, name in enumerate(ranked[:15], start=1):
        mark = "yes" if name in detected else "no"
        print(f"  {i:<6}{name:<22}{scores[name]:>10.4f}  {mark:>10}")
    print(f"\n  Cutoff used: {cutoff:.4f}\n")
    return detected


def separate(model, class_names, audio, selected, strength: float):
    """Run the webapp removal pipeline. Returns the cleaned waveform."""
    queries = np.stack([_query(class_names, n) for n in selected])
    audio_peak = float(np.max(np.abs(audio))) + 1e-8
    audio_norm = audio / audio_peak

    full_stft = librosa.stft(audio_norm, n_fft=N_FFT, hop_length=HOP_LENGTH)
    full_mag = np.abs(full_stft)[:FREQ_BINS].astype(np.float32)
    T = full_mag.shape[1]

    out_acc = np.zeros((FREQ_BINS, T), dtype=np.float32)
    weight_acc = np.zeros(T, dtype=np.float32)
    hann = np.hanning(TIME_FRAMES).astype(np.float32)
    step = TIME_FRAMES // 2

    for start in range(0, T, step):
        end = min(start + TIME_FRAMES, T)
        actual = end - start
        chunk = full_mag[:, start:end]
        if actual < TIME_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TIME_FRAMES - actual)))
        lin = chunk[..., None]
        log = np.log1p(lin)
        est = model.predict(
            [np.repeat(log[None], len(selected), 0), queries,
             np.repeat(lin[None], len(selected), 0)], verbose=0)
        combined_mask = np.ones((FREQ_BINS, TIME_FRAMES), dtype=np.float32)
        for k in range(len(selected)):
            est_mag = est[k, :, :, 0]
            amplitude_ratio = np.clip(est_mag / (chunk + 1e-8), 0.0, 1.0)
            combined_mask *= np.clip(1.0 - strength * amplitude_ratio, 0.0, 1.0)
        for i in range(FREQ_BINS):
            combined_mask[i] = np.convolve(combined_mask[i], SMOOTH_KERNEL,
                                           mode="same")
        combined_mask = np.clip(combined_mask, 0.0, 1.0)
        processed = chunk * combined_mask
        out_acc[:, start:end] += hann[:actual] * processed[:, :actual]
        weight_acc[start:end] += hann[:actual]

    out_mag = out_acc / np.maximum(weight_acc, 1e-8)
    out_full = np.vstack([out_mag, np.zeros((1, T), dtype=np.float32)])
    out = librosa.istft(out_full * np.exp(1j * np.angle(full_stft)),
                        hop_length=HOP_LENGTH, n_fft=N_FFT, length=len(audio_norm))
    out = out * audio_peak
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_path", type=Path, help="Path to audio/video file")
    parser.add_argument("classes", nargs="*",
                        help="Class names to remove. If omitted, only runs detection.")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Removal strength in [0, 1] (default: 1.0)")
    parser.add_argument("--model", type=Path,
                        default=BASE_DIR / "saved_models" / "separation_models"
                                / "separator_unet_film_multi_v2.1.h5")
    args = parser.parse_args()

    classes_path = args.model.with_name(args.model.stem + "_classes.json")
    class_names = json.load(classes_path.open())
    print(f"\nModel: {args.model}")
    print(f"Audio: {args.audio_path}")
    print(f"Classes available: {len(class_names)}")

    model = tf.keras.models.load_model(args.model, compile=False)
    audio = _load_audio(args.audio_path)
    print(f"Loaded {len(audio) / SAMPLE_RATE:.1f} seconds of audio")

    detect(model, class_names, audio)

    if args.classes:
        unknown = [c for c in args.classes if c not in class_names]
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}. "
                             f"Available: {sorted(class_names)}")
        print(f"  Separating: {args.classes} at strength {args.strength}\n")
        out = separate(model, class_names, audio, args.classes, args.strength)
        out_path = args.audio_path.with_name(
            args.audio_path.stem + "_cleaned.wav")
        sf.write(out_path, out, SAMPLE_RATE)
        print(f"  Wrote: {out_path}\n")
    else:
        print("  No classes specified — separation skipped.")
        print("  To remove sounds, add class names as args, e.g.:")
        print(f"  python {sys.argv[0]} {args.audio_path} siren dog\n")


if __name__ == "__main__":
    main()
