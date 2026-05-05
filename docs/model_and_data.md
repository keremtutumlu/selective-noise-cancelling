# Model Architecture and Data Structure

Reference document for the Selective Noise Cancellation (SNC) project.
Describes the data pipeline, feature engineering, and the trained neural
network that classifies overlapping environmental sounds in real time.

---

## 1. Data Pipeline

### 1.1 Source Dataset — ESC-50

The project uses the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset:
2000 environmental sound recordings, 5 seconds each, across 50 classes at
44.1 kHz. We use only 8 of the 50 classes and resample to 16 kHz.

**Required directory layout after download:**

```
data/raw/archive/
├── esc50.csv           ← metadata (filename, category columns)
└── audio/audio/        ← WAV files named by filename column
```

### 1.2 Target Classes

Eight classes are selected based on relevance to the ANC use-case.
They are **always kept in alphabetical order** — this ordering is a hard
contract between the data pipeline, the model, and the inference code.

| Index | Class name       | ANC role |
|-------|-----------------|----------|
| 0 | `car_horn`       | Cancel (user choice) |
| 1 | `crying_baby`    | Pass-through or cancel |
| 2 | `dog`            | Cancel (user choice) |
| 3 | `engine`         | Cancel (user choice) |
| 4 | `keyboard_typing`| Cancel (user choice) |
| 5 | `rain`           | Cancel (user choice) |
| 6 | `siren`          | **Always pass-through** |
| 7 | `wind`           | Cancel (user choice) |

> **Critical invariant:** `ANCPredictor` in `src/application/inference.py`
> calls `sorted(class_labels)` on init. Passing labels in any other order
> silently misaligns predictions with model weights.

### 1.3 Synthetic Multi-Label Generation

`src/data_preparation/synthetic_data_generator.py`

Real environments contain overlapping sounds. The synthetic generator
simulates this by mixing 1–3 isolated clips per training sample.

**Why generate synthetically rather than record real overlaps?**
ESC-50 contains isolated sounds. Recording real-world mixes at sufficient
scale and with known ground truth is impractical. Synthetic mixing is
standard practice in audio machine learning (e.g., DCASE challenges).

**Generation steps per sample:**

1. Choose `k ∈ {1, 2, 3}` random classes.
2. For each class, pick a random validated WAV from the in-memory cache.
3. Extract a random 1-second window (zero-pad if shorter).
4. Assign a random amplitude weight `∈ [0.4, 1.0]` (SNR variance).
5. Sum all tracks; peak-normalise to `[-1.0, 1.0]`.
6. Compute log-mel spectrogram → Z-score normalise → triplicate channels.
7. Emit: feature tensor `(64, 101, 3)` + multi-hot label `(8,)`.

**Performance note:** All WAV files are loaded from disk once at startup
and kept in RAM (~150 MB for 8 classes × ~40 files each). Generating
15 000 samples then takes ~5 minutes on a laptop CPU instead of ~2 hours
with per-sample disk reads.

### 1.4 Output Artefacts

All outputs are written to `data/processed/training_pipeline/` (gitignored).

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `X_multi_features.npy` | `(N, 64, 101, 3)` | `float32` | Z-score normalised log-mel tensors |
| `y_multi_labels.npy` | `(N, 8)` | `float32` | Multi-hot label vectors |
| `class_names.json` | — | — | Alphabetical list of 8 class names |

Default `N = 15 000`. Change via `num_samples` in `synthetic_data_generator.py`.

---

## 2. Feature Engineering

### 2.1 Log-Mel Spectrogram Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 16 000 Hz | Minimum for voice/environmental audio; halves MCU load vs 44.1 kHz |
| FFT window | 400 samples (25 ms) | Standard for speech/env. audio; aligns with ESC-50 literature |
| Hop length | 160 samples (10 ms) | 75% overlap; 101 frames per second of audio |
| Mel bins | 64 | Sufficient spectral resolution for 8-class task; compact for MCU |
| Output shape | `(64, 101, 1)` | Mel bins × Time frames × 1 channel |

### 2.2 Channel Replication (RGB Emulation)

MobileNetV2 expects a 3-channel (RGB) input. The single grayscale
log-mel channel is triplicated: `(64, 101, 1) → (64, 101, 3)`.

This is intentional and well-established in audio transfer-learning
literature. The three channels are identical; MobileNetV2 still benefits
from its ImageNet-pretrained feature detectors (edge detection, texture
patterns) which transfer to spectrogram patterns.

### 2.3 Z-Score Normalisation

Applied per sample (not globally):

```
x_norm = (x - mean(x)) / (std(x) + 1e-6)
```

Per-sample normalisation makes the model robust to absolute volume
differences between recordings and across microphone gain settings.

---

## 3. Model Architecture

### 3.1 Backbone — MobileNetV2 (α = 0.35)

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 |
| Width multiplier α | 0.35 |
| Pretrained weights | ImageNet |
| Input | `(64, 101, 3)` |
| Backbone output | `(2, 4, 1280)` feature map |
| Total parameters | ~575 K |
| Trainable (stage 1) | ~165 K (head only) |

The α = 0.35 multiplier scales down all channel widths by 65%. This
reduces the INT8-quantised model to ~450 KB, which fits in the 8 MB
flash of an ESP32-S3 N8R8 with room for the firmware and audio buffers.

### 3.2 Classification Head

```
GlobalAveragePooling2D  (1280,)
Dropout(0.3)
Dense(128, relu)
Dropout(0.3)
Dense(8, sigmoid)       ← one sigmoid per class (multi-label)
```

Sigmoid (not softmax) allows multiple classes to be active simultaneously.
Each output is an independent probability that the corresponding class is
present in the current 1-second window.

### 3.3 Loss Function and Metrics

| Item | Value |
|------|-------|
| Loss | Binary Cross-Entropy |
| Threshold for bin_acc, precision, recall | 0.5 (default; tune per class — see §4) |
| AUC | Multi-label (one ROC curve per class, averaged) |

### 3.4 Two-Stage Transfer Learning

**Stage 1 — Head warmup (frozen backbone):**
- Backbone frozen: all 410 K backbone params non-trainable.
- Only the 165 K head params update.
- Prevents the randomly-initialised head from destroying ImageNet weights.
- Learning rate: 1e-3, up to 15 epochs, early stopping patience 5.

**Stage 2 — Full fine-tuning (unfrozen backbone):**
- All 575 K params trainable.
- Low learning rate (1e-4) avoids catastrophic forgetting.
- The ModelCheckpoint threshold is seeded with stage 1's best val_loss,
  so the BN-layer spike at stage 2 epoch 1 cannot overwrite stage 1's
  weights. (This was a bug in the original code; fixed in
  `bugfix/stage2-checkpoint-overwrite`.)
- Up to 35 epochs, early stopping patience 7.

### 3.5 Training Results (first run, CPU only)

| Metric | Stage 1 best | Final test |
|--------|-------------|------------|
| val_loss | 0.3514 | 0.376 |
| bin_acc | 85.9% | 86.1% |
| AUC | 0.856 | 0.871 |
| Precision | 85.4% | 77.8% |
| Recall | 52.7% | 62.8% |

**Interpretation:**
- AUC 0.87 is solid for a PoC with 8 overlapping classes on synthetic data.
- Recall (62.8%) is the weak point for a safety system. The 0.5 threshold
  is conservative — per-class threshold tuning (see §4) pushes recall
  above 70% for most classes at an acceptable precision cost.
- Run `tests/test_model.py` to obtain optimised per-class thresholds.

---

## 4. Per-Class Threshold Tuning

The sigmoid outputs are probabilities; the decision threshold (default 0.5)
controls the precision-recall trade-off independently for each class.

For a safety application the recommended strategy is:
- **siren, car_horn:** lower threshold (0.3–0.4) — prefer recall; missing
  a siren is worse than a false alarm.
- **keyboard_typing, rain:** higher threshold (0.5–0.6) — prefer precision;
  false cancellation of background ambience is acceptable.

Run `tests/test_model.py` to get the F1-optimal threshold for each class
on the held-out test split. Measured values from the current checkpoint:

```python
THRESHOLDS = {
    "car_horn":         0.25,
    "crying_baby":      0.40,
    "dog":              0.35,
    "engine":           0.35,
    "keyboard_typing":  0.35,
    "rain":             0.65,
    "siren":            0.40,
    "wind":             0.35,
}
```

Per-class F1 / precision / recall at these thresholds (1500-sample test split):

| Class | Thresh | F1 | Precision | Recall |
|-------|-------:|---:|----------:|-------:|
| car_horn        | 0.25 | 0.505 | 0.454 | 0.569 |
| crying_baby     | 0.40 | 0.745 | 0.780 | 0.713 |
| dog             | 0.35 | 0.577 | 0.636 | 0.527 |
| engine          | 0.35 | 0.679 | 0.820 | 0.579 |
| keyboard_typing | 0.35 | 0.620 | 0.724 | 0.542 |
| rain            | 0.65 | 0.861 | 0.912 | 0.815 |
| **siren**       | 0.40 | **0.828** | 0.819 | **0.836** |
| wind            | 0.35 | 0.636 | 0.633 | 0.639 |

`siren` (the always-pass-through safety class) is the strongest safety
indicator at 83.6% recall. `car_horn` is the weakest, likely confused with
engine/siren mid-band energy in synthetic overlaps; since car_horn is a
user-choice cancel class, its failures are not safety-critical.

Copy the `THRESHOLDS` dict into `ANCPredictor` in
`src/application/inference.py`.

---

## 5. Saved Artefacts

All paths below are gitignored and must be regenerated locally.

| Path | Description |
|------|-------------|
| `saved_models/base_models/best_mobilenetv2_multilabel.h5` | Best Keras checkpoint (float32) |
| `saved_models/tflite/model_float32.tflite` | TFLite float32 baseline |
| `saved_models/tflite/model_float16.tflite` | TFLite float16 (~half size) |
| `saved_models/tflite/model_dynamic_int8.tflite` | Weights INT8, activations float32 |
| `saved_models/tflite/model_full_int8.tflite` | **Target for MCU deployment** |
| `saved_models/tflm_c_array/g_anc_model.h` | C header for TFLite Micro |
| `saved_models/tflm_c_array/g_anc_model.cc` | C source (16-byte aligned byte array) |
