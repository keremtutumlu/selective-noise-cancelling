"""
Evaluation script for the source-separation U-Net.

Measures separation quality on the held-out test split using **SI-SDR**
(scale-invariant signal-to-distortion ratio) — the standard metric in the
source-separation literature, reported in decibels (higher is better).

Unlike training (which works on magnitude spectrograms), evaluation works
on reconstructed *waveforms*: the model's estimated stem magnitudes are
combined with the mixture's phase and inverse-STFT'd back to audio, then
compared sample-by-sample against the ground-truth stems.

Two numbers are reported per class:

* **SI-SDR (mixture)** — quality of the unprocessed mixture against the
  reference stem. This is the "do nothing" baseline.
* **SI-SDR (model)**   — quality of the model's separated stem.

Their difference, **SI-SDRi** (improvement), is the headline result: how
many dB of separation the model adds over doing nothing.

Run from the project root:
    python src/model_training/evaluate_separator.py

Prerequisites:
    * saved_models/separation_models/best_separator_unet.h5
    * data/processed/separation_pipeline/{mixtures,stems}.npy + class_names.json
"""
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

try:  # works when run as a script (src/model_training on sys.path)
    from separator_unet import (
        FREQ_BINS, HOP_LENGTH, N_FFT, NUM_CLASSES, SAMPLE_RATE, TIME_FRAMES,
    )
except ImportError:  # works when imported as a package
    from model_training.separator_unet import (
        FREQ_BINS, HOP_LENGTH, N_FFT, NUM_CLASSES, SAMPLE_RATE, TIME_FRAMES,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-invariant SDR (dB) of ``estimate`` against ``reference``.

    The reference is projected out of the estimate; the ratio of projected
    energy to residual energy, in dB, is scale-invariant — it does not
    change if the estimate is uniformly louder or quieter.
    """
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) /
                                 (np.dot(noise, noise) + eps)))


class SeparatorEvaluator:
    """Reconstructs waveforms from the model and scores them with SI-SDR."""

    def __init__(self, data_dir: Path, model_path: Path, seed: int = 42):
        self.data_dir = data_dir
        self.model_path = model_path
        self.seed = seed

    def _mixture_spectrogram(self, waveform: np.ndarray):
        """
        Full complex STFT of a mixture plus the cropped magnitude inputs.

        Returns ``(phase, n_frames, lin_mag, log_mag)`` where ``phase`` and
        ``n_frames`` describe the full STFT (needed to invert it) and the
        two magnitudes are cropped to the U-Net's fixed input shape.
        """
        stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.angle(stft)
        n_frames = stft.shape[1]

        mag = np.abs(stft).astype(np.float32)[:FREQ_BINS, :]   # drop Nyquist bin
        if mag.shape[1] < TIME_FRAMES:
            mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
        else:
            mag = mag[:, :TIME_FRAMES]
        lin_mag = mag[..., np.newaxis]
        log_mag = np.log1p(lin_mag)
        return phase, n_frames, lin_mag, log_mag

    def _reconstruct(self, est_mag: np.ndarray, phase: np.ndarray,
                     n_frames: int) -> np.ndarray:
        """Inverse-STFT one estimated stem magnitude using the mixture phase."""
        mag = est_mag[:, :n_frames]                            # undo time padding
        nyquist = np.zeros((1, n_frames), dtype=mag.dtype)     # restore dropped bin
        mag_full = np.concatenate([mag, nyquist], axis=0)      # (1 + N_FFT/2, frames)
        stft = mag_full * np.exp(1j * phase)
        return librosa.istft(stft, hop_length=HOP_LENGTH, n_fft=N_FFT,
                             length=SAMPLE_RATE)

    def evaluate(self) -> None:
        print("\nSeparation U-Net — Evaluation Report")
        print(f"Model : {self.model_path}")
        print(f"Data  : {self.data_dir}")

        mixtures = np.load(self.data_dir / "mixtures.npy")
        stems = np.load(self.data_dir / "stems.npy")
        with (self.data_dir / "class_names.json").open() as f:
            class_names = json.load(f)

        # Reproduce the trainer's test split exactly (seed 42, last 10%).
        idx = np.arange(mixtures.shape[0])
        idx_tmp, idx_test = train_test_split(idx, test_size=0.10, random_state=self.seed)
        print(f"Test set: {len(idx_test)} mixtures\n")

        model = tf.keras.models.load_model(self.model_path)

        # Build the model inputs for every test mixture in one batch.
        log_batch = np.empty((len(idx_test), FREQ_BINS, TIME_FRAMES, 1), dtype=np.float32)
        lin_batch = np.empty_like(log_batch)
        phases, frame_counts = [], []
        for k, i in enumerate(idx_test):
            phase, n_frames, lin_mag, log_mag = self._mixture_spectrogram(mixtures[i])
            lin_batch[k] = lin_mag
            log_batch[k] = log_mag
            phases.append(phase)
            frame_counts.append(n_frames)

        print("Running separation on the test set...", end=" ", flush=True)
        est_mags = model.predict([log_batch, lin_batch], batch_size=16, verbose=0)
        print("done.\n")

        # Accumulate SI-SDR per class, only over samples where the class is present.
        sdr_model = {c: [] for c in class_names}
        sdr_mix = {c: [] for c in class_names}
        for k, i in enumerate(idx_test):
            for c, name in enumerate(class_names):
                ref = stems[i, c]
                if np.max(np.abs(ref)) < 1e-6:
                    continue  # class absent — SI-SDR undefined against silence
                est = self._reconstruct(est_mags[k, :, :, c], phases[k], frame_counts[k])
                sdr_model[name].append(si_sdr(est, ref))
                sdr_mix[name].append(si_sdr(mixtures[i], ref))

        print(f"  {'Class':<18}{'N':>6}{'SI-SDR mix':>14}{'SI-SDR model':>16}{'SI-SDRi':>12}")
        print(f"  {'-' * 18}{'-' * 6}{'-' * 14}{'-' * 16}{'-' * 12}")
        all_model, all_mix = [], []
        for name in class_names:
            if not sdr_model[name]:
                print(f"  {name:<18}{'0':>6}{'(absent in test set)':>42}")
                continue
            m_model = float(np.mean(sdr_model[name]))
            m_mix = float(np.mean(sdr_mix[name]))
            all_model.extend(sdr_model[name])
            all_mix.extend(sdr_mix[name])
            print(f"  {name:<18}{len(sdr_model[name]):>6}{m_mix:>12.2f} dB"
                  f"{m_model:>14.2f} dB{m_model - m_mix:>9.2f} dB")

        print(f"  {'-' * 18}{'-' * 6}{'-' * 14}{'-' * 16}{'-' * 12}")
        mean_model = float(np.mean(all_model))
        mean_mix = float(np.mean(all_mix))
        print(f"  {'AVERAGE':<18}{len(all_model):>6}{mean_mix:>12.2f} dB"
              f"{mean_model:>14.2f} dB{mean_model - mean_mix:>9.2f} dB")
        print(f"\n  SI-SDRi (improvement over the unprocessed mixture): "
              f"{mean_model - mean_mix:+.2f} dB")
        print("=" * 64 + "\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    SeparatorEvaluator(
        data_dir=BASE_DIR / "data" / "processed" / "separation_pipeline",
        model_path=BASE_DIR / "saved_models" / "separation_models" / "best_separator_unet.h5",
    ).evaluate()
