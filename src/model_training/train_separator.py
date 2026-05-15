"""
Training pipeline for the source-separation U-Net.

Loads the mixture/stems waveforms produced by ``separation_dataset.py``,
converts them to magnitude spectrograms, and trains the U-Net from
``separator_unet.py`` to predict one soft mask per class.

Training model
--------------
The bare U-Net maps a (log-compressed) magnitude spectrogram to 8 masks.
For training we wrap it so a standard ``model.fit`` works end to end:

    log magnitude  --\\
                      U-Net --> 8 masks --\\
    linear magnitude -----------------------> multiply --> 8 estimated
                                                            stem magnitudes

The masks are applied to the *linear* mixture magnitude; the U-Net sees
the *log-compressed* magnitude (better conditioned). Both are precomputed
as plain array inputs — there are no Lambda layers, so the saved ``.h5``
reloads without custom objects.

Loss
----
L1 (mean absolute error) between the estimated and the true per-class
stem magnitudes. L1 is the standard, robust choice for mask-based
magnitude separation.

Separation quality (SI-SDR, measured on reconstructed waveforms) is
reported by the dedicated evaluation script, not here — it needs the
inverse STFT and the mixture phase.

Output:
    saved_models/separation_models/best_separator_unet.h5
"""
import json
import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Multiply

try:  # works when run as a script (src/model_training on sys.path)
    from separator_unet import (
        FREQ_BINS, HOP_LENGTH, N_FFT, NUM_CLASSES, TIME_FRAMES,
        build_separator_unet,
    )
except ImportError:  # works when imported as a package
    from model_training.separator_unet import (
        FREQ_BINS, HOP_LENGTH, N_FFT, NUM_CLASSES, TIME_FRAMES,
        build_separator_unet,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def waveform_to_magnitude(waveform: np.ndarray) -> np.ndarray:
    """
    STFT magnitude of a 1-second waveform, shaped for the U-Net.

    Returns a ``(FREQ_BINS, TIME_FRAMES)`` float32 array: the Nyquist
    frequency bin is dropped and the time axis is padded/cropped to a
    fixed length so every spectrogram has identical shape.
    """
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)        # (1 + N_FFT/2, frames)
    mag = mag[:FREQ_BINS, :]                     # drop the Nyquist bin
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return mag


class SeparatorTrainer:
    """Trains the separation U-Net on synthetic mixture/stems waveforms."""

    def __init__(
        self,
        data_dir: Path,
        model_save_dir: Path,
        base_filters: int = 32,
        batch_size: int = 16,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.base_filters = base_filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.model_save_dir / "best_separator_unet.h5"

        tf.keras.utils.set_random_seed(seed)

    def _load_and_transform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load waveforms and convert them to spectrograms.

        Returns:
            log_mag:   float32 ``(N, FREQ_BINS, TIME_FRAMES, 1)`` — U-Net input.
            lin_mag:   float32 ``(N, FREQ_BINS, TIME_FRAMES, 1)`` — masking input.
            stem_mags: float16 ``(N, FREQ_BINS, TIME_FRAMES, NUM_CLASSES)`` — targets.
        """
        mixtures = np.load(self.data_dir / "mixtures.npy")        # (N, 16000)
        stems = np.load(self.data_dir / "stems.npy")              # (N, 8, 16000)
        with (self.data_dir / "class_names.json").open() as f:
            class_names = json.load(f)
        n = mixtures.shape[0]
        logging.info(f"Loaded {n} mixtures, {len(class_names)} classes: {class_names}")

        lin_mag = np.empty((n, FREQ_BINS, TIME_FRAMES, 1), dtype=np.float32)
        # float16 keeps the 8-channel target array roughly half the size in RAM.
        stem_mags = np.empty((n, FREQ_BINS, TIME_FRAMES, NUM_CLASSES), dtype=np.float16)

        log_every = max(1, n // 20)
        for i in range(n):
            lin_mag[i, :, :, 0] = waveform_to_magnitude(mixtures[i])
            for c in range(NUM_CLASSES):
                stem_mags[i, :, :, c] = waveform_to_magnitude(stems[i, c])
            if (i + 1) % log_every == 0:
                logging.info(f"  spectrograms {i + 1:>6d} / {n}")

        log_mag = np.log1p(lin_mag).astype(np.float32)
        logging.info(f"Spectrograms ready: input {lin_mag.shape}, targets {stem_mags.shape}")
        return log_mag, lin_mag, stem_mags

    def _build_training_model(self) -> Tuple[Model, Model]:
        """
        Build the trainable wrapper around the U-Net.

        Returns ``(training_model, unet)`` so the bare U-Net can be reused
        on its own. The training model has two inputs (log magnitude for
        the U-Net, linear magnitude for masking) and outputs the 8
        estimated stem magnitudes.
        """
        unet = build_separator_unet(
            input_shape=(FREQ_BINS, TIME_FRAMES, 1),
            num_classes=NUM_CLASSES,
            base_filters=self.base_filters,
        )
        log_in = Input(shape=(FREQ_BINS, TIME_FRAMES, 1), name="log_magnitude")
        lin_in = Input(shape=(FREQ_BINS, TIME_FRAMES, 1), name="linear_magnitude")

        masks = unet(log_in)                                  # (F, T, 8)
        est_stems = Multiply(name="apply_masks")([masks, lin_in])  # broadcast (F,T,1)

        training_model = Model([log_in, lin_in], est_stems, name="separator_training_model")
        return training_model, unet

    def _callbacks(self) -> list:
        return [
            EarlyStopping(monitor="val_loss", patience=self.patience,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(self.checkpoint_path), monitor="val_loss",
                            mode="min", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                              verbose=1, min_lr=1e-6),
        ]

    def train(self) -> Model:
        log_mag, lin_mag, stem_mags = self._load_and_transform()

        # 80 / 10 / 10 train/val/test split — same scheme and seed as the
        # classification trainer so results are comparable.
        idx = np.arange(log_mag.shape[0])
        idx_tmp, idx_test = train_test_split(idx, test_size=0.10, random_state=self.seed)
        idx_tr, idx_val = train_test_split(idx_tmp, test_size=1 / 9, random_state=self.seed)
        logging.info(f"Split: {len(idx_tr)} train / {len(idx_val)} val / {len(idx_test)} test")

        model, unet = self._build_training_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mae",
        )
        unet.summary(print_fn=logging.info)
        logging.info(f"Training model parameters: {model.count_params():,}")

        model.fit(
            x=[log_mag[idx_tr], lin_mag[idx_tr]],
            y=stem_mags[idx_tr].astype(np.float32),
            validation_data=(
                [log_mag[idx_val], lin_mag[idx_val]],
                stem_mags[idx_val].astype(np.float32),
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._callbacks(),
            verbose=2,
        )

        logging.info("Loading best checkpoint and evaluating on the test set...")
        best = tf.keras.models.load_model(self.checkpoint_path)
        test_loss = best.evaluate(
            [log_mag[idx_test], lin_mag[idx_test]],
            stem_mags[idx_test].astype(np.float32),
            verbose=0,
        )
        logging.info(f"Test L1 (magnitude) loss: {test_loss:.5f}")
        logging.info(f"Best model saved to {self.checkpoint_path}")
        return best


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    SeparatorTrainer(
        data_dir=BASE_DIR / "data" / "processed" / "separation_pipeline",
        model_save_dir=BASE_DIR / "saved_models" / "separation_models",
    ).train()
