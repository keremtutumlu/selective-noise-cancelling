"""
Training pipeline for the query-conditioned separation U-Net.

Mixtures are produced on the fly by ``SeparationMixer`` — there is no
dataset file to load. The training set is an infinite ``tf.data``
stream; a fixed validation set is materialised once so ``val_loss`` is
stable for early stopping and checkpointing.

Training model
--------------
The conditioned U-Net maps ``[log magnitude, class query]`` to one mask.
For training it is wrapped so a standard ``model.fit`` works:

    log magnitude  --\\
    class query    ---- conditioned U-Net --> mask --\\
    linear magnitude --------------------------------> multiply --> estimated
                                                                    stem magnitude

The mask is applied to the linear magnitude with a plain Multiply layer
(no Lambda — the saved ``.h5`` reloads without custom objects). Loss is
a multi-resolution L1 (full + half + quarter resolution) between the
estimated and the true target-stem magnitude.

Output:
    saved_models/separation_models/separator_unet_film_multi_v2.2.h5
    saved_models/separation_models/separator_unet_film_multi_v2.2_classes.json
"""
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Multiply


def _multi_res_l1(y_true, y_pred):
    """L1 at full + half + quarter resolution of the magnitude spectrogram.

    Coarser scales help the model converge on the overall spectral shape
    before fine-tuning individual bin values, improving mask quality.
    """
    loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    for i in range(1, 3):
        y_true = tf.nn.avg_pool2d(y_true, ksize=2, strides=2, padding="SAME")
        y_pred = tf.nn.avg_pool2d(y_pred, ksize=2, strides=2, padding="SAME")
        loss += (0.5 ** i) * tf.reduce_mean(tf.abs(y_true - y_pred))
    return loss

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "data_preparation"))
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, TIME_FRAMES, build_conditioned_separator,
)
from separation_mixer import SeparationMixer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConditionedSeparatorTrainer:
    """Trains the query-conditioned separator on on-the-fly mixtures."""

    def __init__(
        self,
        data_root: Path,
        model_save_dir: Path,
        base_filters: int = 32,
        batch_size: int = 16,
        epochs: int = 60,
        steps_per_epoch: int = 500,
        n_val: int = 800,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.model_save_dir = model_save_dir
        self.base_filters = base_filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.n_val = n_val
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.model_save_dir / "separator_unet_film_multi_v2.2.h5"

        tf.keras.utils.set_random_seed(seed)

    def _datasets(self) -> Tuple[tf.data.Dataset, tuple, list]:
        """Build the streaming train set and a fixed validation set."""
        train_mixer = SeparationMixer(self.data_root, seed=self.seed)
        val_mixer = SeparationMixer(self.data_root, seed=self.seed + 1)
        num_classes = train_mixer.num_classes

        spec_shape = (FREQ_BINS, TIME_FRAMES, 1)
        output_signature = (
            (tf.TensorSpec(spec_shape, tf.float32),
             tf.TensorSpec((num_classes,), tf.float32),
             tf.TensorSpec(spec_shape, tf.float32)),
            tf.TensorSpec(spec_shape, tf.float32),
        )
        train_ds = (
            tf.data.Dataset.from_generator(train_mixer.generate,
                                           output_signature=output_signature)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Materialise a fixed validation set so val_loss does not drift.
        logging.info(f"Building fixed validation set ({self.n_val} examples)...")
        gen = val_mixer.generate()
        examples = [next(gen) for _ in range(self.n_val)]
        val_log = np.stack([e[0][0] for e in examples])
        val_query = np.stack([e[0][1] for e in examples])
        val_lin = np.stack([e[0][2] for e in examples])
        val_target = np.stack([e[1] for e in examples])
        validation_data = ([val_log, val_query, val_lin], val_target)

        return train_ds, validation_data, train_mixer.class_names

    def _build_training_model(self, num_classes: int) -> Tuple[Model, Model]:
        """Wrap the conditioned U-Net so masking happens inside the model."""
        cond_unet = build_conditioned_separator(
            input_shape=(FREQ_BINS, TIME_FRAMES, 1),
            num_classes=num_classes,
            base_filters=self.base_filters,
        )
        log_in = Input(shape=(FREQ_BINS, TIME_FRAMES, 1), name="log_magnitude")
        query_in = Input(shape=(num_classes,), name="class_query")
        lin_in = Input(shape=(FREQ_BINS, TIME_FRAMES, 1), name="linear_magnitude")

        mask = cond_unet([log_in, query_in])
        est_stem = Multiply(name="apply_mask")([mask, lin_in])

        training_model = Model([log_in, query_in, lin_in], est_stem,
                               name="conditioned_separator_training_model")
        return training_model, cond_unet

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
        train_ds, validation_data, class_names = self._datasets()
        logging.info(f"Training on {len(class_names)} classes: {class_names}")

        # Persist the class list so the evaluator and application can build
        # the one-hot query without re-reading the dataset.
        names_path = self.model_save_dir / "separator_unet_film_multi_v2.2_classes.json"
        with names_path.open("w") as f:
            json.dump(class_names, f, indent=2)
        logging.info(f"Saved class names to {names_path}")

        model, cond_unet = self._build_training_model(len(class_names))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=_multi_res_l1,
        )
        cond_unet.summary(print_fn=logging.info)
        logging.info(f"Training model parameters: {model.count_params():,}")

        model.fit(
            train_ds,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=validation_data,
            epochs=self.epochs,
            callbacks=self._callbacks(),
            verbose=2,
        )
        logging.info(f"Best model saved to {self.checkpoint_path}")
        return model


if __name__ == "__main__":
    ConditionedSeparatorTrainer(
        data_root=BASE_DIR / "data" / "raw",
        model_save_dir=BASE_DIR / "saved_models" / "separation_models",
    ).train()
