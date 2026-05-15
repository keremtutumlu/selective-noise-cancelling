"""
Synthetic dataset builder for source-separation training.

Where ``synthetic_data_generator.py`` emits log-mel features and multi-hot
labels for the *classification* model, this module emits the raw waveforms
needed to train a *separation* model:

* the **mixture** waveform (1 s, 16 kHz mono), and
* the **8 isolated per-class stem waveforms** that sum to that mixture.

A stem is all-zeros when its class is absent from the mixture — those zero
targets are useful training signal: they teach the separator to output
silence for classes that are not present.

The mixture and its stems share a single peak-normalisation factor, so the
invariant ``mixture == stems.sum(axis=0)`` holds exactly. Mask-based
separation relies on this: a predicted mask multiplied by the mixture must
be able to reconstruct each stem.

Inputs:
    * ``data/raw/archive/esc50.csv`` — ESC-50 metadata.
    * ``data/raw/archive/audio/audio/`` — ESC-50 WAV files.

Outputs (``data/processed/separation_pipeline/``):
    * ``mixtures.npy``    — float32 ``(N, 16000)`` mixed waveforms.
    * ``stems.npy``       — float32 ``(N, 8, 16000)`` isolated per-class stems.
    * ``class_names.json`` — alphabetical list of the 8 class names.

Note on size: ``stems.npy`` is ``N x 8 x 16000 x 4`` bytes. The default
``N = 6000`` produces a ~3.1 GB stem file — generate this on Colab or a
machine with enough RAM, not a laptop.
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd

try:  # works when run as a script (src/data_preparation on sys.path)
    from synthetic_data_generator import CANONICAL_CLASSES
except ImportError:  # works when imported as a package
    from data_preparation.synthetic_data_generator import CANONICAL_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SeparationDatasetBuilder:
    """
    Builds a synthetic mixture/stems dataset for source-separation training.

    Usage:
        builder = SeparationDatasetBuilder(csv_path, audio_dir)
        builder.build(num_samples=6000, output_dir=separation_dir)
    """

    def __init__(
        self,
        csv_path: Path,
        audio_dir: Path,
        target_classes: Tuple[str, ...] = CANONICAL_CLASSES,
        target_sr: int = 16000,
        max_mix: int = 3,
        amp_range: Tuple[float, float] = (0.4, 1.0),
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.target_classes = tuple(sorted(target_classes))
        self.num_classes = len(self.target_classes)
        self.target_sr = target_sr
        self.window_length = target_sr  # 1 second
        self.max_mix = max_mix
        self.amp_range = amp_range

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

    def _generate_one(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single (mixture, stems) pair.

        Returns:
            mixture: float32 ``(window_length,)``
            stems:   float32 ``(num_classes, window_length)`` — row ``i`` is the
                     isolated waveform of class ``i``, all-zeros if absent.
        """
        k = random.randint(1, self.max_mix)
        chosen = random.sample(self.target_classes, k)

        stems = np.zeros((self.num_classes, self.window_length), dtype=np.float32)
        for cls in chosen:
            idx = self.target_classes.index(cls)
            waveform = random.choice(self._waveform_cache[cls])
            window = self._random_window(waveform)
            stems[idx] = window * random.uniform(*self.amp_range)

        mixture = stems.sum(axis=0)

        # One shared normalisation factor keeps mixture == stems.sum(0).
        peak = np.max(np.abs(mixture))
        if peak > 0:
            mixture = mixture / peak
            stems = stems / peak

        return mixture.astype(np.float32), stems.astype(np.float32)

    def build(self, num_samples: int, output_dir: Path) -> None:
        """Generate ``num_samples`` mixture/stems pairs and serialise them."""
        logging.info(f"Generating {num_samples} synthetic mixture/stems pairs...")
        output_dir.mkdir(parents=True, exist_ok=True)

        mixtures = np.empty((num_samples, self.window_length), dtype=np.float32)
        stems = np.empty((num_samples, self.num_classes, self.window_length), dtype=np.float32)

        log_every = max(1, num_samples // 20)
        for i in range(num_samples):
            mixtures[i], stems[i] = self._generate_one()
            if (i + 1) % log_every == 0:
                logging.info(f"  {i + 1:>6d} / {num_samples}")

        np.save(output_dir / "mixtures.npy", mixtures)
        np.save(output_dir / "stems.npy", stems)
        with (output_dir / "class_names.json").open("w") as f:
            json.dump(list(self.target_classes), f, indent=2)

        # A class is "active" in a sample if its stem has any non-zero energy.
        active_rate = (np.abs(stems).max(axis=2) > 0).mean(axis=0)
        logging.info(f"Mixtures: {mixtures.shape} {mixtures.dtype}")
        logging.info(f"Stems:    {stems.shape} {stems.dtype}")
        logging.info(f"Per-class active rate: "
                     f"{dict(zip(self.target_classes, active_rate.round(3).tolist()))}")
        logging.info(f"Saved artefacts to {output_dir}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    builder = SeparationDatasetBuilder(
        csv_path=BASE_DIR / "data" / "raw" / "archive" / "esc50.csv",
        audio_dir=BASE_DIR / "data" / "raw" / "archive" / "audio" / "audio",
    )
    builder.build(
        num_samples=6000,
        output_dir=BASE_DIR / "data" / "processed" / "separation_pipeline",
    )
