"""
Selective sound removal application using the trained separation U-Net.

This is the user-facing end of the source-separation track. Given an audio
file and a per-class gain setting, it:

1. splits the audio into 1-second windows,
2. runs the U-Net on each window to estimate the 8 per-class stem
   magnitudes,
3. recombines the stems with the requested per-class gains
   (``0.0`` removes a sound, ``1.0`` keeps it, ``0.5`` halves it),
4. inverse-STFTs back to a waveform and writes the result.

Because the gain is applied per class, the same call can *remove* sounds,
*duck* them, or leave them untouched — this is the "selective" in
Selective Noise Cancellation.

Run from the project root:
    python src/application/selective_separation.py

Prerequisites:
    * saved_models/separation_models/best_separator_unet.h5
    * data/processed/separation_pipeline/class_names.json
    * an input WAV (defaults to data/test_samples/test_sound.wav)
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).parent.parent.parent
# The STFT contract lives with the model definition — reuse it verbatim.
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from separator_unet import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, NUM_CLASSES, SAMPLE_RATE, TIME_FRAMES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SelectiveSeparator:
    """Removes or attenuates chosen sound classes from an audio file."""

    def __init__(self, model_path: Path, class_names_path: Path):
        import tensorflow as tf

        if not model_path.exists():
            raise FileNotFoundError(f"Trained separator not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        with class_names_path.open() as f:
            self.class_names = json.load(f)
        logging.info(f"Loaded separator for classes: {self.class_names}")

    def _process_window(self, window: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """Separate one 1-second window and recombine it with per-class gains."""
        stft = librosa.stft(window, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.angle(stft)
        n_frames = stft.shape[1]

        mag = np.abs(stft).astype(np.float32)[:FREQ_BINS, :]   # drop Nyquist bin
        if mag.shape[1] < TIME_FRAMES:
            mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
        else:
            mag = mag[:, :TIME_FRAMES]

        lin = mag[np.newaxis, ..., np.newaxis]
        log = np.log1p(lin)
        est_mags = self.model.predict([log, lin], verbose=0)[0]  # (FREQ, TIME, 8)

        # Recombine: a gain-weighted sum of the estimated stem magnitudes.
        combined = np.zeros((FREQ_BINS, TIME_FRAMES), dtype=np.float32)
        for c in range(NUM_CLASSES):
            combined += gains[c] * est_mags[:, :, c]

        combined = combined[:, :n_frames]
        nyquist = np.zeros((1, n_frames), dtype=combined.dtype)
        combined_full = np.concatenate([combined, nyquist], axis=0)
        out_stft = combined_full * np.exp(1j * phase)
        return librosa.istft(out_stft, hop_length=HOP_LENGTH, n_fft=N_FFT,
                             length=len(window))

    def process(self, input_path: Path, output_path: Path,
                gains: Dict[str, float]) -> None:
        """
        Process ``input_path`` and write the result to ``output_path``.

        Args:
            gains: maps class name -> gain. Classes not listed default to
                1.0 (kept unchanged). Use 0.0 to remove a class.
        """
        gain_vec = np.ones(NUM_CLASSES, dtype=np.float32)
        for name, g in gains.items():
            if name not in self.class_names:
                raise ValueError(f"Unknown class '{name}'. Known: {self.class_names}")
            gain_vec[self.class_names.index(name)] = g

        removed = [n for n, g in gains.items() if g == 0.0]
        ducked = [f"{n}={g}" for n, g in gains.items() if 0.0 < g < 1.0]
        logging.info(f"Removing: {removed or 'none'} | Attenuating: {ducked or 'none'}")

        audio, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        n_windows = int(np.ceil(len(audio) / SAMPLE_RATE))
        padded = np.pad(audio, (0, n_windows * SAMPLE_RATE - len(audio)))

        out = np.concatenate([
            self._process_window(padded[w * SAMPLE_RATE:(w + 1) * SAMPLE_RATE], gain_vec)
            for w in range(n_windows)
        ])[:len(audio)]

        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak

        sf.write(output_path, out, SAMPLE_RATE)
        logging.info(f"Wrote {len(out) / SAMPLE_RATE:.1f}s of audio to {output_path}")


if __name__ == "__main__":
    separator = SelectiveSeparator(
        model_path=BASE_DIR / "saved_models" / "separation_models" / "best_separator_unet.h5",
        class_names_path=BASE_DIR / "data" / "processed" / "separation_pipeline" / "class_names.json",
    )
    # Example: remove engine and wind, keep everything else.
    separator.process(
        input_path=BASE_DIR / "data" / "test_samples" / "test_sound.wav",
        output_path=Path("separated_output.wav"),
        gains={"engine": 0.0, "wind": 0.0},
    )
