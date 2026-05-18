"""
Selective sound removal using the query-conditioned separator.

Given an audio file and a per-class gain setting, this:

1. splits the audio into 1-second windows,
2. for each class the user wants to attenuate, queries the conditioned
   U-Net for that class's estimated stem magnitude,
3. subtracts the attenuated portion of each estimated stem from the
   mixture magnitude (``gain 0.0`` removes a sound fully, ``1.0`` keeps
   it untouched, ``0.5`` halves it),
4. inverse-STFTs back to a waveform using the mixture phase.

Only classes with ``gain < 1.0`` are queried — everything else passes
through untouched, which is what makes the cancellation *selective*.

Run from the project root:
    python src/application/conditioned_selective_separation.py

Prerequisites:
    * saved_models/separation_models/best_conditioned_separator.h5
    * saved_models/separation_models/conditioned_class_names.json
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
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, SAMPLE_RATE, TIME_FRAMES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConditionedSelectiveSeparator:
    """Removes or attenuates chosen sound classes via the conditioned model."""

    def __init__(self, model_path: Path, class_names_path: Path):
        import tensorflow as tf

        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        with class_names_path.open() as f:
            self.class_names = json.load(f)
        logging.info(f"Loaded conditioned separator with {len(self.class_names)} classes.")

    def _query(self, name: str) -> np.ndarray:
        """One-hot query vector for class ``name``."""
        if name not in self.class_names:
            raise ValueError(f"Unknown class '{name}'.")
        q = np.zeros(len(self.class_names), dtype=np.float32)
        q[self.class_names.index(name)] = 1.0
        return q

    def _process_window(self, window: np.ndarray,
                        targets: Dict[str, float]) -> np.ndarray:
        """Attenuate the requested classes in one 1-second window."""
        stft = librosa.stft(window, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.angle(stft)
        n_frames = stft.shape[1]

        mag = np.abs(stft).astype(np.float32)[:FREQ_BINS, :]
        if mag.shape[1] < TIME_FRAMES:
            mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
        else:
            mag = mag[:, :TIME_FRAMES]

        lin = mag[..., np.newaxis]
        log = np.log1p(lin)

        # Query the model once per class to attenuate, batched together.
        names = list(targets.keys())
        queries = np.stack([self._query(n) for n in names])
        log_batch = np.repeat(log[np.newaxis], len(names), axis=0)
        lin_batch = np.repeat(lin[np.newaxis], len(names), axis=0)
        est_stems = self.model.predict([log_batch, queries, lin_batch], verbose=0)

        # Subtract the attenuated fraction of each estimated stem.
        out_mag = mag.copy()
        for k, name in enumerate(names):
            out_mag -= (1.0 - targets[name]) * est_stems[k, :, :, 0]
        out_mag = np.clip(out_mag, 0.0, None)

        out_mag = out_mag[:, :n_frames]
        nyquist = np.zeros((1, n_frames), dtype=out_mag.dtype)
        out_full = np.concatenate([out_mag, nyquist], axis=0)
        return librosa.istft(out_full * np.exp(1j * phase),
                             hop_length=HOP_LENGTH, n_fft=N_FFT, length=len(window))

    def process(self, input_path: Path, output_path: Path,
                gains: Dict[str, float]) -> None:
        """
        Process ``input_path`` and write the result to ``output_path``.

        Args:
            gains: maps class name -> gain. Only classes with gain < 1.0
                are touched; 0.0 removes a class, 0.5 halves it.
        """
        targets = {n: g for n, g in gains.items() if g < 1.0}
        if not targets:
            raise ValueError("No class has gain < 1.0 — nothing to remove.")
        logging.info(f"Attenuating: {targets}")

        audio, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        n_windows = int(np.ceil(len(audio) / SAMPLE_RATE))
        padded = np.pad(audio, (0, n_windows * SAMPLE_RATE - len(audio)))

        out = np.concatenate([
            self._process_window(padded[w * SAMPLE_RATE:(w + 1) * SAMPLE_RATE], targets)
            for w in range(n_windows)
        ])[:len(audio)]

        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak

        sf.write(output_path, out, SAMPLE_RATE)
        logging.info(f"Wrote {len(out) / SAMPLE_RATE:.1f}s of audio to {output_path}")


if __name__ == "__main__":
    models = BASE_DIR / "saved_models" / "separation_models"
    separator = ConditionedSelectiveSeparator(
        model_path=models / "best_conditioned_separator.h5",
        class_names_path=models / "conditioned_class_names.json",
    )
    # Example: remove engine and wind, keep everything else.
    separator.process(
        input_path=BASE_DIR / "data" / "test_samples" / "test_sound.wav",
        output_path=Path("conditioned_separated_output.wav"),
        gains={"engine": 0.0, "wind": 0.0},
    )
