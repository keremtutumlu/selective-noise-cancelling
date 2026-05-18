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

The query is embedded and turned into **FiLM** parameters (a per-channel
scale ``gamma`` and shift ``beta``) that modulate the U-Net bottleneck.
This conditions the whole decoder on the requested class. The output is a
single soft mask ``(256, 128, 1)`` in ``[0, 1]``; multiplied by the
mixture magnitude it estimates that class's isolated stem.

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
) -> Model:
    """
    Build the query-conditioned separation U-Net.

    Args:
        input_shape: magnitude-spectrogram input, ``(freq, time, 1)``.
        num_classes: size of the one-hot ``class_query`` input.
        base_filters: channels of the first encoder block; deeper levels
            double it. ``32`` gives a ~8M-parameter model.

    Returns:
        A Keras ``Model`` mapping ``[log_magnitude, class_query]`` to a
        single soft mask ``(freq, time, 1)`` for the queried class.
    """
    log_mag = Input(shape=input_shape, name="mixture_log_magnitude")
    query = Input(shape=(num_classes,), name="class_query")

    # --- Encoder ----------------------------------------------------------
    e1 = _conv_block(log_mag, base_filters)                  # (256, 128)
    e2 = _conv_block(MaxPooling2D(2)(e1), base_filters * 2)  # (128, 64)
    e3 = _conv_block(MaxPooling2D(2)(e2), base_filters * 4)  # (64, 32)
    e4 = _conv_block(MaxPooling2D(2)(e3), base_filters * 8)  # (32, 16)
    bottleneck = _conv_block(MaxPooling2D(2)(e4), base_filters * 16)  # (16, 8)

    # --- FiLM conditioning ------------------------------------------------
    # The class query becomes a per-channel scale (gamma) and shift (beta)
    # that modulate the bottleneck feature map.
    channels = base_filters * 16
    embedding = Dense(128, activation="relu", name="query_embedding")(query)
    gamma = Reshape((1, 1, channels))(Dense(channels, name="film_gamma")(embedding))
    beta = Reshape((1, 1, channels))(Dense(channels, name="film_beta")(embedding))
    conditioned = Add(name="film_apply")([Multiply()([bottleneck, gamma]), beta])

    # --- Decoder (skip connections from the matching encoder level) -------
    d4 = Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(conditioned)
    d4 = _conv_block(Concatenate()([d4, e4]), base_filters * 8)       # (32, 16)

    d3 = Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(d4)
    d3 = _conv_block(Concatenate()([d3, e3]), base_filters * 4)       # (64, 32)

    d2 = Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(d3)
    d2 = _conv_block(Concatenate()([d2, e2]), base_filters * 2)       # (128, 64)

    d1 = Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d2)
    d1 = _conv_block(Concatenate()([d1, e1]), base_filters)           # (256, 128)

    mask = Conv2D(1, 1, padding="same", activation="sigmoid",
                  name="target_mask")(d1)

    return Model([log_mag, query], mask,
                 name=f"conditioned_separator_f{base_filters}")


if __name__ == "__main__":
    model = build_conditioned_separator()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
