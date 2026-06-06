"""
Query-conditioned separation U-Net for the Selective Noise Cancellation
project.

This is the *scalable* separation model. Where ``separator_unet.py``
emits one mask per class for a fixed 8-class set, this network is told
**which** class to extract and emits a single mask for it. The number of
classes only changes the size of the conditioning input — the
convolutional body is unchanged — so the same architecture handles 8, 50
or hundreds of classes, including classes from additional datasets.

Architecture
------------
Two inputs:

* ``mixture_log_magnitude`` ``(256, 128, 1)`` — log-compressed STFT
  magnitude of a 1-second mixture.
* ``class_query`` ``(num_classes,)`` — a one-hot vector selecting the
  class to extract.

The query is embedded once and turned into **FiLM** parameters (per-channel
scale ``gamma`` and shift ``beta``) applied at *every encoder level* and
the bottleneck. Conditioning all levels forces the encoder itself — not
only the decoder — to build class-specific features, improving mask
precision. The output is a single soft mask ``(256, 128, 1)`` in ``[0, 1]``;
multiplied by the mixture magnitude it estimates that class's isolated stem.

To remove several sounds, query the model once per target class.

Spectrogram contract
--------------------
Identical to ``separator_unet.py`` so the audio front-end is shared:
16 kHz mono, 1-second windows, ``n_fft=512``, ``hop_length=128``; the
Nyquist bin is dropped (256 bins) and the time axis padded to 128.
"""
from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Multiply,
    Reshape,
)

# --- Spectrogram contract (shared with the audio front-end) ----------------
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
FREQ_BINS = 256    # U-Net input height: 257 STFT bins minus the Nyquist bin.
TIME_FRAMES = 128  # U-Net input width: 126 frames for 1 s, padded to 128.

# Default class count. ESC-50 has 50 classes; override per dataset.
DEFAULT_NUM_CLASSES = 50


def _conv_block(x, filters: int):
    """Two 3x3 conv + BN + ReLU layers — the standard U-Net double conv."""
    for _ in range(2):
        x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def build_conditioned_separator(
    input_shape: Tuple[int, int, int] = (FREQ_BINS, TIME_FRAMES, 1),
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_filters: int = 32,
    embed_dim: int = 128,
    with_detection_head: bool = False,
) -> Model:
    """
    Build the query-conditioned separation U-Net.

    Args:
        input_shape: magnitude-spectrogram input, ``(freq, time, 1)``.
        num_classes: size of the one-hot ``class_query`` input.
        base_filters: channels of the first encoder block; deeper levels
            double it. ``32`` gives a ~8.3M-parameter model.
        embed_dim: dimension of the shared query embedding used by all
            FiLM layers.
        with_detection_head: when ``True``, add a lightweight classifier head
            on top of the FiLM-conditioned bottleneck that outputs a single
            sigmoid probability ``P(queried class present | mixture)``.  The
            model then returns ``[mask, class_presence]`` instead of just
            ``mask``.  The bottleneck is already conditioned on the class
            query via FiLM, so the detector implicitly knows *which* class it
            is assessing.

    Returns:
        A Keras ``Model`` mapping ``[log_magnitude, class_query]`` to:
        - ``mask``: soft mask ``(freq, time, 1)`` in ``[0, 1]``.
        - ``class_presence`` (only when ``with_detection_head=True``):
          scalar ``(1,)`` probability that the queried class is present.
    """
    log_mag = Input(shape=input_shape, name="mixture_log_magnitude")
    query = Input(shape=(num_classes,), name="class_query")

    # Shared query embedding — computed once, projected separately per level.
    embedding = Dense(embed_dim, activation="relu", name="query_embedding")(query)

    def _film(x, channels: int, level: str):
        """Apply FiLM: x <- gamma * x + beta, with per-level projections."""
        gamma = Reshape((1, 1, channels))(
            Dense(channels, name=f"film_gamma_{level}")(embedding))
        beta = Reshape((1, 1, channels))(
            Dense(channels, name=f"film_beta_{level}")(embedding))
        return Add(name=f"film_{level}")([Multiply()([x, gamma]), beta])

    # --- Encoder with FiLM at every level ---------------------------------
    e1 = _film(_conv_block(log_mag, base_filters), base_filters, "e1")          # (256, 128)
    e2 = _film(_conv_block(MaxPooling2D(2)(e1), base_filters * 2), base_filters * 2, "e2")   # (128, 64)
    e3 = _film(_conv_block(MaxPooling2D(2)(e2), base_filters * 4), base_filters * 4, "e3")   # (64, 32)
    e4 = _film(_conv_block(MaxPooling2D(2)(e3), base_filters * 8), base_filters * 8, "e4")   # (32, 16)
    bottleneck = _film(
        _conv_block(MaxPooling2D(2)(e4), base_filters * 16),
        base_filters * 16, "bottleneck",
    )  # (16, 8)

    # --- Decoder (skip connections already carry class-specific features) -
    d4 = Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bottleneck)
    d4 = _conv_block(Concatenate()([d4, e4]), base_filters * 8)       # (32, 16)

    d3 = Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(d4)
    d3 = _conv_block(Concatenate()([d3, e3]), base_filters * 4)       # (64, 32)

    d2 = Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(d3)
    d2 = _conv_block(Concatenate()([d2, e2]), base_filters * 2)       # (128, 64)

    d1 = Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d2)
    d1 = _conv_block(Concatenate()([d1, e1]), base_filters)           # (256, 128)

    # dtype="float32" pins the output to full precision even under a global
    # mixed_float16 policy: the sigmoid mask and the downstream loss must stay
    # in float32 for numerical stability (float16 saturates/over-rounds here).
    mask = Conv2D(1, 1, padding="same", activation="sigmoid",
                  name="target_mask", dtype="float32")(d1)

    if with_detection_head:
        # Detection head: bottleneck is already FiLM-conditioned on the class
        # query, so GlobalAveragePooling2D collapses it to a class-aware vector.
        # Dense(1, sigmoid) then predicts P(queried class present | mixture).
        det = GlobalAveragePooling2D(name="det_pool")(bottleneck)
        det = Dense(128, activation="relu", name="det_dense")(det)
        # float32 output for stable binary cross-entropy under mixed precision.
        class_presence = Dense(1, activation="sigmoid",
                               name="class_presence", dtype="float32")(det)
        return Model([log_mag, query], [mask, class_presence],
                     name=f"conditioned_separator_f{base_filters}_det")

    return Model([log_mag, query], mask,
                 name=f"conditioned_separator_f{base_filters}")


if __name__ == "__main__":
    model = build_conditioned_separator()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
