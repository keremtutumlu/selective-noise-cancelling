"""
Training pipeline for the query-conditioned separation U-Net.

Mixtures are produced on the fly by ``SeparationMixer`` — there is no
dataset file to load. The training set is an infinite ``tf.data``
stream; a fixed validation set is materialised once so ``val_loss`` is
stable for early stopping and checkpointing.

Training model
--------------
The conditioned U-Net maps ``[log magnitude, class query]`` to one mask
(two outputs when ``with_detection_head=True``). For training it is wrapped
so a standard ``model.fit`` works:

    log magnitude  --\\
    class query    ---- conditioned U-Net --> mask --\\
    linear magnitude --------------------------------> multiply --> est stem magnitude
                                         \\-> class_presence (detection head only)

The mask is applied to the linear magnitude with a plain Multiply layer
(no Lambda — the saved ``.h5`` reloads without custom objects).

Losses:
- Separation : multi-resolution L1 (full + ½ + ¼ resolution), weight 1.0
- Detection  : binary cross-entropy on P(class present), weight 0.3
               (only when ``with_detection_head=True``)

The presence label is derived inside the training pipeline from the target
stem magnitude: presence = 1.0 if max|target_stem| > 1e-6 else 0.0.

Output (version comes from model_config — SNC_MODEL_VERSION env var or default):
    saved_models/separation_models/separator_unet_film_multi_<version>.h5
    saved_models/separation_models/separator_unet_film_multi_<version>_classes.json
"""
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Multiply

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "data_preparation"))
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
import model_config as cfg  # noqa: E402
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, TIME_FRAMES, build_conditioned_separator,
)
from separation_mixer import SeparationMixer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def _focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Binary focal loss for the detection head.

    Standard BCE treats every example equally, so an abundance of easy
    negatives drowns out the gradient signal from the few hard positives the
    head still gets wrong. Focal loss multiplies the per-example BCE by
    ``alpha_t * (1 - p_t)^gamma`` so well-classified examples contribute
    little and hard cases dominate the update. ``alpha=0.25, gamma=2`` are
    the values from the original RetinaNet paper; they work well as a
    drop-in BCE replacement when one class (here: "absent") dominates.

    Returns a closure compatible with ``model.compile(loss=...)``.
    """
    def loss_fn(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    loss_fn.__name__ = f"focal_loss_a{alpha}_g{gamma}"
    return loss_fn


class ConditionedSeparatorTrainer:
    """Trains the query-conditioned separator on on-the-fly mixtures."""

    def __init__(
        self,
        data_root: Path,
        model_save_dir: Path,
        model_version: Optional[str] = None,
        base_filters: int = 32,
        batch_size: int = 16,
        epochs: int = 60,
        steps_per_epoch: int = 500,
        n_val: int = 800,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seed: int = 42,
        # Mixer hyperparameters — automation sweeps usually vary these.
        negative_prob: float = 0.15,
        bg_noise_prob: float = 0.10,
        bg_snr_db_range: Tuple[float, float] = (15.0, 30.0),
        max_mix: int = 4,
        min_clips_per_class: int = 40,
        over_firing_classes: Optional[List[str]] = None,
        over_firing_weight: float = 3.0,
        with_detection_head: bool = False,
        detection_head_dim: int = 128,
        detection_head_dropout: float = 0.0,
        detection_loss_weight: float = 0.3,
        detection_use_focal: bool = False,
        jit_compile: bool = False,
    ):
        self.data_root = data_root
        self.model_save_dir = model_save_dir
        self.model_version = model_version or cfg.model_version()
        self.base_filters = base_filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.n_val = n_val
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.negative_prob = negative_prob
        self.bg_noise_prob = bg_noise_prob
        self.bg_snr_db_range = bg_snr_db_range
        self.max_mix = max_mix
        self.min_clips_per_class = min_clips_per_class
        self.over_firing_classes = over_firing_classes
        self.over_firing_weight = over_firing_weight
        self.with_detection_head = with_detection_head
        self.detection_head_dim = detection_head_dim
        self.detection_head_dropout = detection_head_dropout
        self.detection_loss_weight = detection_loss_weight
        self.detection_use_focal = detection_use_focal
        self.jit_compile = jit_compile

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        stem = cfg.model_stem(self.model_version)
        self.checkpoint_path = self.model_save_dir / f"{stem}.h5"
        self.classes_path = self.model_save_dir / f"{stem}_classes.json"

        tf.keras.utils.set_random_seed(seed)

    def _mixer(self, seed_offset: int = 0) -> SeparationMixer:
        return SeparationMixer(
            self.data_root,
            max_mix=self.max_mix,
            negative_prob=self.negative_prob,
            bg_noise_prob=self.bg_noise_prob,
            bg_snr_db_range=self.bg_snr_db_range,
            min_clips_per_class=self.min_clips_per_class,
            over_firing_classes=self.over_firing_classes,
            over_firing_weight=self.over_firing_weight,
            seed=self.seed + seed_offset,
        )

    def _datasets(self) -> Tuple[tf.data.Dataset, tuple, list]:
        """Build the streaming train set and a fixed validation set."""
        train_mixer = self._mixer(0)
        val_mixer = self._mixer(1)
        num_classes = train_mixer.num_classes

        spec_shape = (FREQ_BINS, TIME_FRAMES, 1)
        output_signature = (
            (tf.TensorSpec(spec_shape, tf.float32),
             tf.TensorSpec((num_classes,), tf.float32),
             tf.TensorSpec(spec_shape, tf.float32)),
            tf.TensorSpec(spec_shape, tf.float32),
        )
        base_ds = tf.data.Dataset.from_generator(
            train_mixer.generate, output_signature=output_signature)

        if self.with_detection_head:
            # Derive a presence label from the target stem: 1.0 if the
            # queried class is present, 0.0 for negative examples.
            # IMPORTANT: return a tuple (not a list) so tf.data treats the
            # two tensors as a nested structure instead of packing them into
            # a single tensor (which would fail because their shapes differ).
            def _add_presence(inputs, target_mag):
                presence = tf.expand_dims(
                    tf.cast(tf.reduce_max(tf.abs(target_mag)) > 1e-6,
                            tf.float32), 0)
                return inputs, (target_mag, presence)
            train_ds = (base_ds
                        .map(_add_presence, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(self.batch_size)
                        .prefetch(tf.data.AUTOTUNE))
        else:
            train_ds = (base_ds
                        .batch(self.batch_size)
                        .prefetch(tf.data.AUTOTUNE))

        # Materialise a fixed validation set so val_loss does not drift.
        logging.info(f"Building fixed validation set ({self.n_val} examples)...")
        gen = val_mixer.generate()
        examples = [next(gen) for _ in range(self.n_val)]
        val_log = np.stack([e[0][0] for e in examples])
        val_query = np.stack([e[0][1] for e in examples])
        val_lin = np.stack([e[0][2] for e in examples])
        val_target = np.stack([e[1] for e in examples])

        if self.with_detection_head:
            val_presence = (
                np.max(np.abs(val_target), axis=(1, 2, 3)) > 1e-6
            ).astype(np.float32)[:, np.newaxis]
            # Tuple, not list — matches the generator's (target_mag, presence) structure.
            validation_data = (
                [val_log, val_query, val_lin],
                (val_target, val_presence),
            )
        else:
            validation_data = ([val_log, val_query, val_lin], val_target)

        return train_ds, validation_data, train_mixer.class_names

    def _build_training_model(self, num_classes: int) -> Tuple[Model, Model]:
        """Wrap the conditioned U-Net so masking happens inside the model."""
        cond_unet = build_conditioned_separator(
            input_shape=(FREQ_BINS, TIME_FRAMES, 1),
            num_classes=num_classes,
            base_filters=self.base_filters,
            with_detection_head=self.with_detection_head,
            detection_head_dim=self.detection_head_dim,
            detection_head_dropout=self.detection_head_dropout,
        )
        log_in = Input(shape=(FREQ_BINS, TIME_FRAMES, 1), name="log_magnitude")
        query_in = Input(shape=(num_classes,), name="class_query")
        lin_in = Input(shape=(FREQ_BINS, TIME_FRAMES, 1), name="linear_magnitude")

        # dtype="float32" on the mask-apply multiply keeps the estimated stem
        # (and therefore the L1 loss) in full precision under mixed_float16.
        if self.with_detection_head:
            mask, class_presence = cond_unet([log_in, query_in])
            est_stem = Multiply(name="apply_mask", dtype="float32")([mask, lin_in])
            training_model = Model(
                [log_in, query_in, lin_in],
                [est_stem, class_presence],
                name="conditioned_separator_training_model",
            )
        else:
            mask = cond_unet([log_in, query_in])
            est_stem = Multiply(name="apply_mask", dtype="float32")([mask, lin_in])
            training_model = Model(
                [log_in, query_in, lin_in], est_stem,
                name="conditioned_separator_training_model",
            )
        return training_model, cond_unet

    def _make_optimizer(self):
        """Adam, wrapped in LossScaleOptimizer under mixed precision.

        float16 gradients can underflow to zero; LossScaleOptimizer scales the
        loss up before backprop and unscales the gradients, preventing it. The
        wrap is guarded so any failure (or a float32 policy) falls back to plain
        Adam — training still proceeds, just without loss scaling.
        """
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        try:
            if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
                opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
                logging.info("Mixed precision active: wrapped Adam in "
                             "LossScaleOptimizer.")
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"LossScaleOptimizer not applied ({exc}); "
                            f"using plain Adam.")
        return opt

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
        # Tee stdout to a buffer for Drive logging (captures model.fit progress).
        _log_buf = io.StringIO()
        _orig_stdout = sys.stdout

        class _Tee:
            def write(self, s):
                _orig_stdout.write(s)
                _log_buf.write(s)
            def flush(self):
                _orig_stdout.flush()
            def fileno(self):
                return _orig_stdout.fileno()
            def isatty(self):
                return _orig_stdout.isatty()

        sys.stdout = _Tee()
        _error: Optional[Exception] = None
        history = None
        class_names: list = []
        try:
            train_ds, validation_data, class_names = self._datasets()
            logging.info(f"Training on {len(class_names)} classes: {class_names}")

            # Persist the class list so the evaluator and application can build
            # the one-hot query without re-reading the dataset.
            with self.classes_path.open("w") as f:
                json.dump(class_names, f, indent=2)
            logging.info(f"Saved class names to {self.classes_path}")

            model, cond_unet = self._build_training_model(len(class_names))
            optimizer = self._make_optimizer()
            if self.with_detection_head:
                det_loss = (_focal_loss(0.25, 2.0)
                            if self.detection_use_focal
                            else "binary_crossentropy")
                model.compile(
                    optimizer=optimizer,
                    loss=[_multi_res_l1, det_loss],
                    loss_weights=[1.0, self.detection_loss_weight],
                    jit_compile=self.jit_compile,
                )
                logging.info(
                    f"Detection loss: "
                    f"{'focal(0.25, 2.0)' if self.detection_use_focal else 'BCE'} "
                    f"@ weight {self.detection_loss_weight}, head dim "
                    f"{self.detection_head_dim}, dropout "
                    f"{self.detection_head_dropout}.")
            else:
                model.compile(
                    optimizer=optimizer,
                    loss=_multi_res_l1,
                    jit_compile=self.jit_compile,
                )
            cond_unet.summary(print_fn=logging.info)
            logging.info(f"Training model parameters: {model.count_params():,}")
            logging.info(f"Detection head: {self.with_detection_head}")
            logging.info(f"Mixed precision: "
                         f"{tf.keras.mixed_precision.global_policy().name}")
            logging.info(f"XLA (jit_compile): {self.jit_compile}")
            logging.info(f"Batch size: {self.batch_size}")

            history = model.fit(
                train_ds,
                steps_per_epoch=self.steps_per_epoch,
                validation_data=validation_data,
                epochs=self.epochs,
                callbacks=self._callbacks(),
                verbose=2,
            )
            logging.info(f"Best model saved to {self.checkpoint_path}")
            return model

        except Exception as exc:
            _error = exc
            raise

        finally:
            sys.stdout = _orig_stdout
            # Write a Drive log entry regardless of success or failure.
            if _error is not None:
                cfg.log_drive_run(
                    script="train_conditioned_separator",
                    version=self.model_version,
                    metrics={
                        "error_type": type(_error).__name__,
                        "error": str(_error)[:500],
                        "num_classes": len(class_names),
                        "with_detection_head": self.with_detection_head,
                    },
                    output=_log_buf.getvalue(),
                    notes=f"FAILED: {type(_error).__name__}",
                )
            elif history is not None:
                hist = history.history
                best_epoch = int(np.argmin(hist.get("val_loss", [0])))
                cfg.log_drive_run(
                    script="train_conditioned_separator",
                    version=self.model_version,
                    metrics={
                        "val_loss_best": float(min(hist.get("val_loss", [0]))),
                        "best_epoch": best_epoch + 1,
                        "epochs_trained": len(hist.get("val_loss", [])),
                        "num_classes": len(class_names),
                        "negative_prob": self.negative_prob,
                        "with_detection_head": self.with_detection_head,
                        "detection_head_dim": self.detection_head_dim,
                        "detection_head_dropout": self.detection_head_dropout,
                        "detection_loss_weight": self.detection_loss_weight,
                        "detection_use_focal": self.detection_use_focal,
                        "batch_size": self.batch_size,
                        "jit_compile": self.jit_compile,
                        "mixed_precision":
                            tf.keras.mixed_precision.global_policy().name,
                    },
                    output=_log_buf.getvalue(),
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the conditioned separator.")
    parser.add_argument(
        "--version", default=None,
        help=f"Model version tag, e.g. v2.6. Overrides the {cfg.ENV_VAR} env var. "
             f"Default: {cfg.model_version()}.")
    args = parser.parse_args()

    version = args.version or cfg.model_version()

    # Per-version training configuration.  Any version not listed here falls
    # back to the original v2.4 defaults (safe, no special augmentation).
    _V25_OVER_FIRING = ["siren", "thunderstorm", "clock_alarm"]
    # v2.6: expanded FP list (top-10 offenders from v2.5 threshold sweep)
    # + detection head to replace the mask-energy scoring heuristic.
    _V26_OVER_FIRING = [
        "thunderstorm", "wind", "clock_alarm", "siren",
        "door_wood_knock", "street_music", "rooster", "train",
        "airplane", "cat",
    ]
    # v2.7: refreshed FP list from the v2.6 detection sweep (see
    # docs/model_training_log.md). v2.7 doubled negative_prob and swapped BCE
    # for focal loss — but focal loss at weight 1.0 produced values ~0.07
    # (vs BCE ~0.46), effective gradient 10x too small → head barely trained.
    _V27_OVER_FIRING = [
        "pig", "drilling", "sea_waves", "helicopter", "laughing",
        "air_conditioner", "vacuum_cleaner", "frog", "crickets", "clapping",
    ]
    # v2.8: reverts detection loss to BCE (no focal-loss scaling problem),
    # raises loss weight to 0.5, keeps negative_prob=0.50 and v2.7's FP list.
    # Detection head stays at dim=128/dropout=0 (v2.6 values worked fine).
    _V28_OVER_FIRING = _V27_OVER_FIRING

    is_v25 = version == "v2.5"
    is_v26 = version == "v2.6"
    is_v27 = version == "v2.7"
    is_v28 = version == "v2.8"

    # --- Speed knobs (safe defaults, override via env without code edits) ---
    # Mixed precision (float16): 1.5-2x faster on Tensor Core GPUs (T4/A100/L4)
    # and halves activation memory. Output layers + loss are pinned to float32
    # (see conditioned_separator.py / apply_mask) so accuracy is unaffected.
    # Disable with SNC_MIXED_PRECISION=0 if you ever see NaN/inf losses.
    if os.environ.get("SNC_MIXED_PRECISION", "1") == "1":
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logging.info("Mixed precision enabled: mixed_float16 "
                         "(set SNC_MIXED_PRECISION=0 to disable).")
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"Mixed precision unavailable, using float32: {exc}")

    # Larger batch keeps a fast GPU busy. 32 is safe on T4 (16 GB) with mixed
    # precision; push SNC_BATCH_SIZE=64 on an A100/L4 (40+ GB).
    batch_size = int(os.environ.get("SNC_BATCH_SIZE", "32"))

    # XLA fuses the training step. Big win on A100/L4, helps on T4 too. If it
    # ever fails to compile an op, set SNC_XLA=0 (fails fast at step 1).
    use_xla = os.environ.get("SNC_XLA", "1") == "1"

    ConditionedSeparatorTrainer(
        data_root=cfg.DATA_ROOT,
        model_save_dir=cfg.MODELS_DIR,
        model_version=version,
        batch_size=batch_size,
        negative_prob=(0.50 if (is_v27 or is_v28) else
                       0.25 if (is_v25 or is_v26) else 0.15),
        over_firing_classes=(
            _V28_OVER_FIRING if is_v28 else
            _V27_OVER_FIRING if is_v27 else
            _V26_OVER_FIRING if is_v26 else
            _V25_OVER_FIRING if is_v25 else None
        ),
        over_firing_weight=3.0,
        with_detection_head=(is_v26 or is_v27 or is_v28),
        detection_head_dim=256 if is_v27 else 128,
        detection_head_dropout=0.3 if is_v27 else 0.0,
        detection_loss_weight=(1.0 if is_v27 else
                               0.5 if is_v28 else 0.3),
        detection_use_focal=is_v27,
        jit_compile=use_xla,
    ).train()
