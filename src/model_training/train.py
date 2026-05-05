"""
Multi-label MobileNetV2 trainer for the Selective Noise Cancellation project.

Loads ``X_multi_features.npy`` / ``y_multi_labels.npy`` produced by
``synthetic_data_generator.py`` and trains a slim MobileNetV2 (α=0.35) with
sigmoid output and binary cross-entropy. The α=0.35 width multiplier keeps the
INT8-quantised model under ~500 KB so it fits comfortably on a budget MCU
(e.g. ESP32-S3) without external PSRAM.

Training proceeds in two stages:

1. **Feature extraction.** The ImageNet-pretrained backbone is frozen and only
   the new classification head is trained. Stops the random-init head from
   destroying useful pretrained filters in the first few batches.
2. **Fine-tuning.** The full backbone is unfrozen and trained at a much lower
   learning rate to adapt low-level filters to log-mel spectrograms.

Output:
    * ``saved_models/base_models/best_mobilenetv2_multilabel.h5`` — best
      checkpoint by validation loss across both stages.
"""
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiLabelTrainer:
    """Two-stage transfer-learning trainer for multi-label log-mel classification."""

    def __init__(
        self,
        data_dir: Path,
        model_save_dir: Path,
        alpha: float = 0.35,
        batch_size: int = 64,
        head_epochs: int = 15,
        finetune_epochs: int = 35,
        head_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.alpha = alpha
        self.batch_size = batch_size
        self.head_epochs = head_epochs
        self.finetune_epochs = finetune_epochs
        self.head_lr = head_lr
        self.finetune_lr = finetune_lr
        self.seed = seed

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.model_save_dir / "best_mobilenetv2_multilabel.h5"

        tf.keras.utils.set_random_seed(seed)

    def _load_splits(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int,
    ]:
        """Load multi-label tensors and produce 80/10/10 train/val/test splits."""
        x = np.load(self.data_dir / "X_multi_features.npy")
        y = np.load(self.data_dir / "y_multi_labels.npy")
        with (self.data_dir / "class_names.json").open() as f:
            class_names = json.load(f)
        num_classes = len(class_names)

        logging.info(f"Loaded {x.shape[0]} samples, {num_classes} classes: {class_names}")
        logging.info(f"Per-class positive rate: {y.mean(axis=0).round(3).tolist()}")

        # Plain random split — multi-label stratification skipped per project decision.
        x_tmp, x_test, y_tmp, y_test = train_test_split(
            x, y, test_size=0.10, random_state=self.seed,
        )
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_tmp, y_tmp, test_size=1 / 9, random_state=self.seed,  # 1/9 of 90% ≈ 10%
        )
        return x_tr, y_tr, x_val, y_val, x_test, y_test, num_classes

    def _build_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> Tuple[Model, Model]:
        """Build (full_model, base_model) so the caller can toggle ``base.trainable``."""
        inputs = Input(shape=input_shape)
        base = MobileNetV2(
            input_tensor=inputs, include_top=False, weights="imagenet", alpha=self.alpha,
        )
        base.trainable = False  # Stage 1 starts frozen.

        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation="sigmoid")(x)

        model = Model(inputs, outputs, name=f"mobilenetv2_a{self.alpha}_multilabel")
        return model, base

    def _compile(self, model: Model, learning_rate: float) -> None:
        """Compile with BCE loss and multi-label-friendly metrics."""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
                tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                tf.keras.metrics.Recall(name="recall", thresholds=0.5),
                tf.keras.metrics.AUC(name="auc", multi_label=True),
            ],
        )

    def _make_checkpoint(self, best_so_far: float = np.inf) -> ModelCheckpoint:
        """
        Build a ModelCheckpoint that only saves when val_loss beats ``best_so_far``.

        Passing the stage 1 best val_loss into stage 2's checkpoint prevents
        stage 2 epoch 1 (which spikes due to BN layer re-initialisation) from
        overwriting the better stage 1 weights.
        """
        ckpt = ModelCheckpoint(
            filepath=str(self.checkpoint_path),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1,
        )
        ckpt.best = best_so_far
        return ckpt

    def _callbacks(self, patience: int, best_val_loss: float = np.inf) -> list:
        return [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
            self._make_checkpoint(best_so_far=best_val_loss),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        ]

    def train(self) -> Model:
        x_tr, y_tr, x_val, y_val, x_test, y_test, num_classes = self._load_splits()
        model, base = self._build_model(input_shape=x_tr.shape[1:], num_classes=num_classes)

        # Stage 1 — train the head only.
        logging.info("Stage 1/2: training classifier head with frozen backbone.")
        self._compile(model, self.head_lr)
        model.summary(print_fn=logging.info)
        stage1 = model.fit(
            x_tr, y_tr,
            validation_data=(x_val, y_val),
            epochs=self.head_epochs,
            batch_size=self.batch_size,
            callbacks=self._callbacks(patience=5),
            verbose=2,
        )
        stage1_best_val_loss = min(stage1.history["val_loss"])
        logging.info(f"Stage 1 best val_loss: {stage1_best_val_loss:.4f}")

        # Stage 2 — unfreeze backbone and fine-tune end-to-end.
        # The checkpoint threshold is seeded with stage 1's best so that the
        # temporary spike at epoch 1 (BN layers re-warming) never overwrites
        # the already-good stage 1 checkpoint.
        logging.info("Stage 2/2: fine-tuning full network at reduced learning rate.")
        base.trainable = True
        self._compile(model, self.finetune_lr)
        model.fit(
            x_tr, y_tr,
            validation_data=(x_val, y_val),
            epochs=self.finetune_epochs,
            batch_size=self.batch_size,
            callbacks=self._callbacks(patience=7, best_val_loss=stage1_best_val_loss),
            verbose=2,
        )

        # Evaluate the best checkpoint on the held-out test set.
        logging.info("Loading best checkpoint and evaluating on test set...")
        best = tf.keras.models.load_model(self.checkpoint_path)
        results = best.evaluate(x_test, y_test, verbose=0, return_dict=True)
        logging.info(f"Test metrics: { {k: round(v, 4) for k, v in results.items()} }")
        return best


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    MultiLabelTrainer(
        data_dir=BASE_DIR / "data" / "processed" / "training_pipeline",
        model_save_dir=BASE_DIR / "saved_models" / "base_models",
    ).train()
