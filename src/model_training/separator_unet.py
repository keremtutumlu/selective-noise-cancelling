"""
Multi-output separation U-Net for the Selective Noise Cancellation project.

This is the *separation* model — distinct from the classification
MobileNetV2 in ``train.py``. Instead of predicting which sounds are
present, it pulls them apart.

Architecture
------------
A 2-D U-Net (encoder / decoder with skip connections) that operates on the
magnitude spectrogram of a 1-second mixture:

    mixture magnitude  (256, 128, 1)
        -> U-Net
    8 masks            (256, 128, 8)   one soft mask per class, in [0, 1]

Each mask is multiplied by the mixture magnitude to estimate that class's
isolated stem. Summing a chosen subset of stems (optionally with per-class
gains) reconstructs the audio with selected sounds removed or attenuated.

Spectrogram contract
--------------------
The U-Net input/output sizes are fixed and derived from these STFT
parameters. ``train_separator.py`` imports these constants so the data
pipeline and the model agree.

* 16 kHz mono, 1-second windows.
* ``n_fft = 512`` -> 257 frequency bins; the Nyquist bin is dropped so the
  U-Net sees a power-of-two-friendly 256 bins, then padded back on output.
* ``hop_length = 128`` -> 126 time frames for 1 s of audio, padded to 128.

Masks are applied to the *linear* magnitude; the network itself is fed a
log-compressed magnitude (better conditioned). Both steps live in
``train_separator.py`` — this module only defines the network.
"""
from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
)

# --- Spectrogram contract (imported by the training pipeline) --------------
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
FREQ_BINS = 256    # U-Net input height: 257 STFT bins minus the Nyquist bin.
TIME_FRAMES = 128  # U-Net input width: 126 frames for 1 s, padded to 128.
NUM_CLASSES = 8


def _conv_block(x, filters: int):
    """Two 3x3 conv + BN + ReLU layers — the standard U-Net double conv."""
    for _ in range(2):
        x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def build_separator_unet(
    input_shape: Tuple[int, int, int] = (FREQ_BINS, TIME_FRAMES, 1),
    num_classes: int = NUM_CLASSES,
    base_filters: int = 32,
) -> Model:
    """
    Build the separation U-Net.

    Args:
        input_shape: magnitude-spectrogram input, ``(freq, time, 1)``.
        num_classes: number of output masks (one per separable class).
        base_filters: channel count of the first encoder block. Each
            deeper level doubles it; the bottleneck has ``base_filters*16``.
            ``32`` gives a ~8M-parameter model.

    Returns:
        A Keras ``Model`` mapping a magnitude spectrogram to ``num_classes``
        soft masks in ``[0, 1]`` of shape ``(freq, time, num_classes)``.
    """
    inputs = Input(shape=input_shape, name="mixture_magnitude")

    # --- Encoder ----------------------------------------------------------
    e1 = _conv_block(inputs, base_filters)             # (256, 128)
    e2 = _conv_block(MaxPooling2D(2)(e1), base_filters * 2)   # (128, 64)
    e3 = _conv_block(MaxPooling2D(2)(e2), base_filters * 4)   # (64, 32)
    e4 = _conv_block(MaxPooling2D(2)(e3), base_filters * 8)   # (32, 16)

    # --- Bottleneck -------------------------------------------------------
    bottleneck = _conv_block(MaxPooling2D(2)(e4), base_filters * 16)  # (16, 8)

    # --- Decoder (skip connections from the matching encoder level) -------
    d4 = Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bottleneck)
    d4 = _conv_block(Concatenate()([d4, e4]), base_filters * 8)       # (32, 16)

    d3 = Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(d4)
    d3 = _conv_block(Concatenate()([d3, e3]), base_filters * 4)       # (64, 32)

    d2 = Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(d3)
    d2 = _conv_block(Concatenate()([d2, e2]), base_filters * 2)       # (128, 64)

    d1 = Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d2)
    d1 = _conv_block(Concatenate()([d1, e1]), base_filters)           # (256, 128)

    masks = Conv2D(num_classes, 1, padding="same", activation="sigmoid",
                   name="class_masks")(d1)

    return Model(inputs, masks, name=f"separator_unet_f{base_filters}")


if __name__ == "__main__":
    model = build_separator_unet()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
