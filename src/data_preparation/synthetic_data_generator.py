"""
Multi-label synthetic dataset builder for the Selective Noise Cancellation
project.

The training objective is multi-label classification — at any given moment the
microphone may pick up several overlapping environmental sounds (e.g. a siren
*and* wind *and* an engine). This module synthesises that condition by mixing
1–3 ESC-50 clips at random amplitude ratios and emitting a multi-hot label.

Compared to the previous generator, this implementation:

* Loads each source WAV from disk **once** and caches the resampled waveform
  in memory. Generating N samples then costs O(N) random-window draws and
  log-mel transforms instead of O(N) disk reads.
* Emits a `class_names.json` alongside the .npy artefacts so downstream
  inference code does not have to hard-code the (alphabetical) class order.
* Treats the alphabetical class ordering as a public contract — see
  ``CANONICAL_CLASSES``.

Inputs:
    * ``data/raw/archive/esc50.csv`` — ESC-50 metadata.
    * ``data/raw/archive/audio/audio/`` — ESC-50 WAV files at 16 kHz.

Outputs (``data/processed/training_pipeline/``):
    * ``X_multi_features.npy`` — float32 ``(N, 64, 101, 3)`` log-mel tensors.
    * ``y_multi_labels.npy``   — float32 ``(N, 8)`` multi-hot vectors.
    * ``class_names.json``     — alphabetical list of class names.
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Public contract: class indices are positions in this alphabetical list.
# `ANCPredictor` and `quantize_for_edge.py` both rely on this ordering.
CANONICAL_CLASSES: Tuple[str, ...] = (
    "car_horn", "crying_baby", "dog", "engine",
    "keyboard_typing", "rain", "siren", "wind",
)


class MultiLabelDatasetBuilder:
    """
    Builds a multi-label log-mel dataset by mixing ESC-50 clips in memory.

    Usage:
        builder = MultiLabelDatasetBuilder(csv_path, audio_dir)
        builder.build(num_samples=15_000, output_dir=processed_dir)
    """

    def __init__(
        self,
        csv_path: Path,
        audio_dir: Path,
        target_classes: Tuple[str, ...] = CANONICAL_CLASSES,
        target_sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        max_mix: int = 3,
        snr_range: Tuple[float, float] = (0.4, 1.0),
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.target_classes = tuple(sorted(target_classes))
        self.num_classes = len(self.target_classes)
        self.target_sr = target_sr
        self.window_length = target_sr  # 1 second
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_mix = max_mix
        self.snr_range = snr_range

        random.seed(seed)
        np.random.seed(seed)

        self._waveform_cache: Dict[str, List[np.ndarray]] = self._load_all_waveforms()

    def _load_all_waveforms(self) -> Dict[str, List[np.ndarray]]:
        """Load and resample every target WAV into RAM exactly once."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        df = pd.read_csv(self.csv_path)
        df = df[df["category"].isin(self.target_classes)]

        cache: Dict[str, List[np.ndarray]] = {cls: [] for cls in self.target_classes}
        missing = 0
        for _, row in df.iterrows():
            path = self.audio_dir / row["filename"]
            if not path.exists():
                missing += 1
                continue
            waveform, _ = librosa.load(path, sr=self.target_sr)
            cache[row["category"]].append(waveform.astype(np.float32))

        if missing:
            logging.warning(f"Skipped {missing} missing files referenced in CSV.")
        for cls in self.target_classes:
            if not cache[cls]:
                raise RuntimeError(f"No usable WAVs found for class '{cls}'.")
            logging.info(f"Cached {len(cache[cls]):>4d} clips for '{cls}'.")
        return cache

    def _random_window(self, waveform: np.ndarray) -> np.ndarray:
        """Crop or zero-pad ``waveform`` to exactly ``self.window_length`` samples."""
        length = len(waveform)
        if length > self.window_length:
            start = random.randint(0, length - self.window_length)
            return waveform[start:start + self.window_length]
        if length < self.window_length:
            return np.pad(waveform, (0, self.window_length - length), mode="constant")
        return waveform

    def _superpose(self, tracks: List[np.ndarray]) -> np.ndarray:
        """Mix tracks at random per-track amplitudes, then peak-normalise to [-1, 1]."""
        mixed = np.zeros(self.window_length, dtype=np.float32)
        for track in tracks:
            mixed += track * random.uniform(*self.snr_range)
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed /= peak
        return mixed

    def _log_mel_rgb(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert a 1-second waveform into a Z-score-normalised log-mel spectrogram
        triplicated across 3 channels to satisfy MobileNetV2's RGB input layer.
        """
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=self.target_sr,
            n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return np.repeat(log_mel[..., np.newaxis], 3, axis=-1).astype(np.float32)

    def _generate_one(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single (features, multi_hot_label) pair."""
        k = random.randint(1, self.max_mix)
        chosen = random.sample(self.target_classes, k)

        label = np.zeros(self.num_classes, dtype=np.float32)
        windows: List[np.ndarray] = []
        for cls in chosen:
            label[self.target_classes.index(cls)] = 1.0
            waveform = random.choice(self._waveform_cache[cls])
            windows.append(self._random_window(waveform))

        return self._log_mel_rgb(self._superpose(windows)), label

    def build(self, num_samples: int, output_dir: Path) -> None:
        """Generate ``num_samples`` examples and serialise them to ``output_dir``."""
        logging.info(f"Generating {num_samples} synthetic multi-label samples...")
        output_dir.mkdir(parents=True, exist_ok=True)

        features = np.empty((num_samples, self.n_mels, 101, 3), dtype=np.float32)
        labels = np.empty((num_samples, self.num_classes), dtype=np.float32)

        log_every = max(1, num_samples // 20)
        for i in range(num_samples):
            features[i], labels[i] = self._generate_one()
            if (i + 1) % log_every == 0:
                logging.info(f"  {i + 1:>6d} / {num_samples}")

        np.save(output_dir / "X_multi_features.npy", features)
        np.save(output_dir / "y_multi_labels.npy", labels)
        with (output_dir / "class_names.json").open("w") as f:
            json.dump(list(self.target_classes), f, indent=2)

        logging.info(f"Features: {features.shape} {features.dtype}")
        logging.info(f"Labels:   {labels.shape} {labels.dtype}")
        logging.info(f"Per-class positives: "
                     f"{dict(zip(self.target_classes, labels.sum(axis=0).astype(int)))}")
        logging.info(f"Saved artefacts to {output_dir}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    builder = MultiLabelDatasetBuilder(
        csv_path=BASE_DIR / "data" / "raw" / "archive" / "esc50.csv",
        audio_dir=BASE_DIR / "data" / "raw" / "archive" / "audio" / "audio",
    )
    builder.build(
        num_samples=15_000,
        output_dir=BASE_DIR / "data" / "processed" / "training_pipeline",
    )
