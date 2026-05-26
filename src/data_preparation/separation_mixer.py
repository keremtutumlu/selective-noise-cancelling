"""
On-the-fly mixture generator for the query-conditioned separator.

Every training step draws a fresh random mixture from an in-memory clip
cache — there is no dataset file, so the class count is unbounded.

The clip cache is built by ``dataset_sources.load_all_datasets``, which
merges every dataset it finds under the data root (ESC-50, UrbanSound8K,
and any added later). The mixer itself is dataset-agnostic.

Each generated example is a single (mixture, query, target stem) triple:

* a 1-second mixture of 1-``max_mix`` random clips,
* a one-hot ``query`` selecting one class, and
* that class's isolated stem magnitude — or **silence** when the query
  asks for a class that is absent from the mixture (negative examples).

The generator yields data already shaped for the conditioned U-Net:

    ((log_magnitude, query, linear_magnitude), target_stem_magnitude)
"""
import logging
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import librosa
import numpy as np

try:  # works when run as a script (src/data_preparation on sys.path)
    from dataset_sources import load_all_datasets
    from conditioned_separator import FREQ_BINS, HOP_LENGTH, N_FFT, TIME_FRAMES
except ImportError:  # works when imported as a package
    from data_preparation.dataset_sources import load_all_datasets
    from model_training.conditioned_separator import (
        FREQ_BINS, HOP_LENGTH, N_FFT, TIME_FRAMES,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def waveform_to_magnitude(waveform: np.ndarray) -> np.ndarray:
    """STFT magnitude of a 1-second waveform, shaped ``(FREQ_BINS, TIME_FRAMES)``."""
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)[:FREQ_BINS, :]   # drop the Nyquist bin
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return mag


class SeparationMixer:
    """Generates conditioned-separation training examples on the fly."""

    def __init__(
        self,
        data_root: Path,
        target_sr: int = 16000,
        max_mix: int = 4,
        amp_range: Tuple[float, float] = (0.4, 1.0),
        negative_prob: float = 0.15,
        bg_noise_prob: float = 0.10,
        bg_snr_db_range: Tuple[float, float] = (15.0, 30.0),
        seed: int = 42,
    ):
        self.target_sr = target_sr
        self.window_length = target_sr  # 1 second
        self.max_mix = max_mix
        self.amp_range = amp_range
        self.negative_prob = negative_prob
        self.bg_noise_prob = bg_noise_prob
        self.bg_snr_db_range = bg_snr_db_range

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Merge every dataset found under data_root into one clip cache.
        self._waveform_cache: Dict[str, List[np.ndarray]] = load_all_datasets(
            data_root, target_sr)
        self.class_names: List[str] = sorted(self._waveform_cache)
        self.num_classes = len(self.class_names)
        logging.info(f"SeparationMixer ready: {self.num_classes} classes.")

    def _random_window(self, waveform: np.ndarray) -> np.ndarray:
        """Crop or zero-pad ``waveform`` to exactly ``window_length`` samples."""
        length = len(waveform)
        if length > self.window_length:
            start = self._rng.randint(0, length - self.window_length)
            return waveform[start:start + self.window_length]
        if length < self.window_length:
            return np.pad(waveform, (0, self.window_length - length))
        return waveform

    def _maybe_add_background_noise(self, mixture: np.ndarray) -> np.ndarray:
        """
        With probability ``bg_noise_prob``, mix broadband noise into the
        mixture at a random SNR drawn from ``bg_snr_db_range``. The target
        stem is **not** corrupted, so the model learns to ignore the noise.

        Half of the calls produce white noise, the other half pink noise
        (1/sqrt(f) FFT shaping) — covering both flat and 1/f-tilted
        real-world ambiances (HVAC, crowd murmur, traffic rumble).
        """
        if self._rng.random() >= self.bg_noise_prob:
            return mixture

        noise = self._np_rng.standard_normal(self.window_length).astype(np.float32)
        if self._rng.random() < 0.5:
            fft = np.fft.rfft(noise)
            scale = 1.0 / np.sqrt(np.arange(1, len(fft) + 1, dtype=np.float32))
            noise = np.fft.irfft(fft * scale, n=self.window_length).astype(np.float32)

        mix_rms = float(np.sqrt(np.mean(mixture ** 2))) + 1e-8
        noise_rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-8
        snr_db = self._rng.uniform(*self.bg_snr_db_range)
        target_noise_rms = mix_rms / (10.0 ** (snr_db / 20.0))
        return mixture + noise * (target_noise_rms / noise_rms)

    def _make_example(self):
        """Build one (mixture, query class, target stem) triple."""
        k = self._rng.randint(1, self.max_mix)
        present = self._rng.sample(self.class_names, k)

        windows: Dict[str, np.ndarray] = {}
        mixture = np.zeros(self.window_length, dtype=np.float32)
        for cls in present:
            clip = self._rng.choice(self._waveform_cache[cls])
            window = self._random_window(clip) * self._rng.uniform(*self.amp_range)
            windows[cls] = window
            mixture += window

        # Corrupt the mixture (but never the target stem) with broadband
        # noise so the model learns to discriminate target vs. ambiance.
        mixture = self._maybe_add_background_noise(mixture)

        peak = np.max(np.abs(mixture))
        if peak > 0:
            mixture /= peak
            for cls in windows:
                windows[cls] = windows[cls] / peak

        # Negative example: query an absent class -> the target is silence.
        absent = [c for c in self.class_names if c not in present]
        if absent and self._rng.random() < self.negative_prob:
            target_cls = self._rng.choice(absent)
            target_stem = np.zeros(self.window_length, dtype=np.float32)
        else:
            target_cls = self._rng.choice(present)
            target_stem = windows[target_cls]

        return mixture, target_cls, target_stem

    def generate(self) -> Iterator:
        """
        Infinite generator of conditioned-separation training examples.

        Yields:
            ``((log_mag, query, lin_mag), target_stem_mag)`` where each
            spectrogram is ``(FREQ_BINS, TIME_FRAMES, 1)`` float32 and
            ``query`` is a ``(num_classes,)`` one-hot float32 vector.
        """
        while True:
            mixture, target_cls, target_stem = self._make_example()

            lin_mag = waveform_to_magnitude(mixture)[..., np.newaxis]
            log_mag = np.log1p(lin_mag)
            target_mag = waveform_to_magnitude(target_stem)[..., np.newaxis]

            query = np.zeros(self.num_classes, dtype=np.float32)
            query[self.class_names.index(target_cls)] = 1.0

            yield (log_mag, query, lin_mag), target_mag


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    mixer = SeparationMixer(data_root=BASE_DIR / "data" / "raw")
    print(f"Classes ({mixer.num_classes}): {mixer.class_names}")
    (log_mag, query, lin_mag), target = next(mixer.generate())
    print(f"log_mag {log_mag.shape}, query {query.shape}, target {target.shape}")
