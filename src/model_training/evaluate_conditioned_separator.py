"""
Evaluation script for the query-conditioned separation U-Net.

Measures separation quality with **SI-SDR** (scale-invariant
signal-to-distortion ratio, in dB — higher is better) on a fixed test
set of on-the-fly mixtures.

For every test mixture the model is queried for the class that is
actually present; its estimated stem magnitude is combined with the
mixture phase, inverse-STFT'd to a waveform, and scored against the
ground-truth stem. Only positive examples are used (the test mixer is
built with ``negative_prob=0``), since SI-SDR is undefined against
silence.

Two numbers per class:

* **SI-SDR (mixture)** — the unprocessed mixture vs. the reference stem
  (the "do nothing" baseline).
* **SI-SDR (model)**   — the model's separated stem.

Their difference, **SI-SDRi**, is the headline result.

Run from the project root:
    python src/model_training/evaluate_conditioned_separator.py
"""
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "data_preparation"))
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, SAMPLE_RATE, TIME_FRAMES,
)
from separation_mixer import SeparationMixer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR (dB) of ``estimate`` against ``reference``."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) /
                                 (np.dot(noise, noise) + eps)))


class ConditionedSeparatorEvaluator:
    """Reconstructs waveforms from the conditioned model and scores SI-SDR."""

    def __init__(self, data_root: Path, model_path: Path,
                 n_test: int = 800, seed: int = 4242):
        self.data_root = data_root
        self.model_path = model_path
        self.n_test = n_test
        self.seed = seed

    def _mixture_spectrogram(self, waveform: np.ndarray):
        """Full STFT phase plus the cropped magnitude inputs for the U-Net."""
        stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.angle(stft)
        n_frames = stft.shape[1]

        mag = np.abs(stft).astype(np.float32)[:FREQ_BINS, :]
        if mag.shape[1] < TIME_FRAMES:
            mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
        else:
            mag = mag[:, :TIME_FRAMES]
        lin_mag = mag[..., np.newaxis]
        return phase, n_frames, lin_mag, np.log1p(lin_mag)

    def _reconstruct(self, est_mag: np.ndarray, phase: np.ndarray,
                     n_frames: int) -> np.ndarray:
        """Inverse-STFT an estimated stem magnitude using the mixture phase."""
        mag = est_mag[:, :n_frames]
        nyquist = np.zeros((1, n_frames), dtype=mag.dtype)
        mag_full = np.concatenate([mag, nyquist], axis=0)
        stft = mag_full * np.exp(1j * phase)
        return librosa.istft(stft, hop_length=HOP_LENGTH, n_fft=N_FFT,
                             length=SAMPLE_RATE)

    def evaluate(self) -> None:
        print("\nConditioned Separator — Evaluation Report")
        print(f"Model : {self.model_path}\n")

        # All test examples are positive: the query class is always present.
        mixer = SeparationMixer(self.data_root, negative_prob=0.0, seed=self.seed)
        class_names = mixer.class_names
        model = tf.keras.models.load_model(self.model_path, compile=False)

        # Draw a fixed test set of (mixture, query, target stem) waveforms.
        examples = [mixer._make_example() for _ in range(self.n_test)]

        log_b = np.empty((self.n_test, FREQ_BINS, TIME_FRAMES, 1), dtype=np.float32)
        lin_b = np.empty_like(log_b)
        query_b = np.zeros((self.n_test, len(class_names)), dtype=np.float32)
        phases, frames = [], []
        for k, (mixture, target_cls, _) in enumerate(examples):
            phase, n_frames, lin_mag, log_mag = self._mixture_spectrogram(mixture)
            lin_b[k] = lin_mag
            log_b[k] = log_mag
            query_b[k, class_names.index(target_cls)] = 1.0
            phases.append(phase)
            frames.append(n_frames)

        print(f"Running separation on {self.n_test} test mixtures...", end=" ", flush=True)
        est_mags = model.predict([log_b, query_b, lin_b], batch_size=16, verbose=0)
        print("done.\n")

        sdr_model = {c: [] for c in class_names}
        sdr_mix = {c: [] for c in class_names}
        for k, (mixture, target_cls, target_stem) in enumerate(examples):
            if np.max(np.abs(target_stem)) < 1e-6:
                continue
            est = self._reconstruct(est_mags[k, :, :, 0], phases[k], frames[k])
            sdr_model[target_cls].append(si_sdr(est, target_stem))
            sdr_mix[target_cls].append(si_sdr(mixture, target_stem))

        print(f"  {'Class':<20}{'N':>5}{'SI-SDR mix':>14}{'SI-SDR model':>16}{'SI-SDRi':>11}")
        print(f"  {'-' * 20}{'-' * 5}{'-' * 14}{'-' * 16}{'-' * 11}")
        all_model, all_mix = [], []
        for name in class_names:
            if not sdr_model[name]:
                continue
            m_model = float(np.mean(sdr_model[name]))
            m_mix = float(np.mean(sdr_mix[name]))
            all_model.extend(sdr_model[name])
            all_mix.extend(sdr_mix[name])
            print(f"  {name:<20}{len(sdr_model[name]):>5}{m_mix:>11.2f} dB"
                  f"{m_model:>13.2f} dB{m_model - m_mix:>8.2f} dB")

        print(f"  {'-' * 20}{'-' * 5}{'-' * 14}{'-' * 16}{'-' * 11}")
        mean_model = float(np.mean(all_model))
        mean_mix = float(np.mean(all_mix))
        print(f"  {'AVERAGE':<20}{len(all_model):>5}{mean_mix:>11.2f} dB"
              f"{mean_model:>13.2f} dB{mean_model - mean_mix:>8.2f} dB")
        print(f"\n  SI-SDRi (improvement over the unprocessed mixture): "
              f"{mean_model - mean_mix:+.2f} dB")
        print("=" * 68 + "\n")


if __name__ == "__main__":
    ConditionedSeparatorEvaluator(
        data_root=BASE_DIR / "data" / "raw",
        model_path=BASE_DIR / "saved_models" / "separation_models" / "separator_unet_film_multi_v2.2.h5",
    ).evaluate()
