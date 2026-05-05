# Pipeline Guide — How to Run the Project

Step-by-step instructions for every stage of the SNC pipeline, from raw
dataset download through embedded C array export and application simulation.

---

## Prerequisites

### Hardware
- A Linux or macOS machine with at least 8 GB RAM (16 GB recommended for
  training with full fine-tuning enabled).
- No GPU required; training runs on CPU in ~20 minutes (15k samples).

### Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install tensorflow==2.21.0 \
            librosa==0.11.0 \
            soundfile \
            scikit-learn \
            numpy \
            pandas
```

> All scripts must be run with this venv active. The venv is gitignored.

### ESC-50 Dataset

1. Download from [GitHub](https://github.com/karolpiczak/ESC-50) or
   [Kaggle](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50).
2. Extract so the layout matches:

```
data/raw/archive/
├── esc50.csv
└── audio/
    └── audio/         ← .wav files go here
```

> `data/` is gitignored. You must create these directories manually.

---

## Stage 1 — Synthetic Data Generation

Loads all target WAVs into RAM once, then generates 15 000 synthetic
multi-label samples by mixing 1–3 clips at random amplitudes.

```bash
python src/data_preparation/synthetic_data_generator.py
```

**Runtime:** ~5–10 minutes on a laptop CPU.

**Outputs** (`data/processed/training_pipeline/`):

| File | Shape | Notes |
|------|-------|-------|
| `X_multi_features.npy` | `(15000, 64, 101, 3)` | Float32, Z-score normalised log-mel |
| `y_multi_labels.npy` | `(15000, 8)` | Float32 multi-hot vectors |
| `class_names.json` | — | Alphabetical class list; read by training and inference |

**Verify output:**
```bash
python -c "
import numpy as np, json
x = np.load('data/processed/training_pipeline/X_multi_features.npy')
y = np.load('data/processed/training_pipeline/y_multi_labels.npy')
names = json.load(open('data/processed/training_pipeline/class_names.json'))
print('X:', x.shape, x.dtype)
print('y:', y.shape, y.dtype)
print('classes:', names)
print('per-class rate:', y.mean(axis=0).round(3))
"
```

Expected per-class positive rate: approximately **0.25 for each class**
(each class appears in ~25% of samples by design).

---

## Stage 2 — Model Training

Two-stage transfer learning on the synthetic dataset.

```bash
python src/model_training/train.py
```

**Runtime:** ~20 minutes on a laptop CPU (stage 1: ~2 min, stage 2: ~18 min).

**What happens:**
1. Loads and splits the data (80 / 10 / 10 train/val/test, seed=42).
2. Stage 1: trains only the classification head (frozen MobileNetV2 backbone),
   15 epochs max, early stopping patience 5.
3. Stage 2: unfreezes the full network, fine-tunes at 1e-4 LR, 35 epochs
   max, early stopping patience 7. The checkpoint threshold is seeded with
   stage 1's best val_loss so no regression can overwrite a good checkpoint.
4. Evaluates the best overall checkpoint on the test set and prints metrics.

**Output:** `saved_models/base_models/best_mobilenetv2_multilabel.h5`

**Target metrics (approximate, CPU training):**

| Metric | Expected |
|--------|----------|
| val_loss | 0.35 – 0.38 |
| Binary accuracy | 85 – 88% |
| AUC (multi-label) | 0.85 – 0.90 |
| Recall @ 0.5 thresh | 60 – 70% |

> If recall is below 60%, run `tests/test_model.py` to find per-class
> thresholds — the 0.5 default is conservative.

---

## Stage 3 — Model Validation (Software-Only)

Validates the trained model without hardware. Run before ordering any
components.

```bash
python tests/test_model.py
```

**What it checks:**
- Model file loads correctly, correct input/output shapes and sigmoid activation.
- `class_names.json` matches `CANONICAL_CLASSES` in the data generator.
- Batch inference produces valid probabilities (no NaN, range [0,1]).
- Single-sample inference latency on this machine (for reference).
- Per-class F1-optimal threshold sweep on the held-out test set.
- Optional: WAV file inference (place a test file at
  `data/test_samples/test_sound.wav`).

**Most important output — optimal thresholds:**
```
THRESHOLDS = {
    "car_horn": 0.35,
    "siren": 0.30,
    ...
}
```
Copy these into `ANCPredictor` in `src/application/inference.py` to
replace the hard-coded 0.5 threshold before deploying.

---

## Stage 4 — Edge Optimisation (TFLite Conversion)

Converts the Keras model into four TFLite variants and benchmarks each.
Requires the trained `.h5` from Stage 2 and the `.npy` features from Stage 1
for INT8 calibration.

```bash
python src/model_optimization/quantize_for_edge.py
```

**Runtime:** ~2–3 minutes.

**Outputs** (`saved_models/tflite/`):

| File | Typical size | Use |
|------|-------------|-----|
| `model_float32.tflite` | ~2.3 MB | Sanity / accuracy baseline |
| `model_float16.tflite` | ~1.2 MB | GPU-capable edge devices |
| `model_dynamic_int8.tflite` | ~600 KB | Weights INT8, activations float32 |
| `model_full_int8.tflite` | ~450 KB | **MCU target** (TFLM-compatible) |

The script prints a comparison table:

```
Variant         Size (KB)     MAE vs Keras      Latency (ms)
-------       ----------     ------------      ------------
float32            2300.0         0.000000              4.20
float16            1150.0         0.000031              4.18
dynamic_int8        620.0         0.003100              3.90
full_int8           450.0         0.006200              3.50
```

`MAE vs Keras` is the mean absolute error between the TFLite and Keras
predictions on 50 calibration samples. Values below **0.01** are excellent;
above 0.05 means the calibration dataset is too small or unrepresentative —
increase `num_samples` in `_load_calibration_samples`.

---

## Stage 5 — C Array Export (TFLite Micro)

Packages `model_full_int8.tflite` as a 16-byte-aligned C array for
inclusion in MCU firmware (ESP-IDF, Arduino_TensorFlowLite, etc.).

```bash
python src/model_optimization/export_c_header.py
```

**Outputs** (`saved_models/tflm_c_array/`):

```
g_anc_model.h   ← include this in your MCU firmware
g_anc_model.cc  ← add to your build system sources
```

**Usage in ESP-IDF firmware (example):**

```cpp
#include "g_anc_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Construct interpreter using g_anc_model[] and g_anc_model_len
tflite::MicroInterpreter interpreter(
    tflite::GetModel(g_anc_model),
    resolver, tensor_arena, kTensorArenaSize
);
```

---

## Stage 6 — Application Scripts

> **Must be run from within `src/application/`** — the scripts use
> relative imports.

```bash
cd src/application
```

### 6a. End-to-end simulation

```bash
python simulate_anc.py
```

Loads `best_mobilenetv2_multilabel.h5`, classifies a test WAV, and if the
top class is in the user's blocklist applies phase inversion
(`signal × -1.0`) with a 3 ms hardware-latency shift.

Outputs: `anti_phase_test_sound.wav` in the current directory.

Requires: `data/test_samples/test_sound.wav` (any WAV, ≥1 second).

### 6b. Acoustic cancellation verification

```bash
python verify_cancellation.py
```

Simulates the loudspeaker delay (shifts anti-noise signal right by
`latency_samples`), sums original and anti-noise, and reports RMS
energy reduction in dB.

Requires: `anti_phase_test_sound.wav` produced by `simulate_anc.py`.

### 6c. Standalone inference

```bash
python inference.py
```

Classifies a single test WAV and prints class probabilities. Useful for
spot-checking the model on arbitrary audio.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: esc50.csv` | Dataset not downloaded or wrong path | Check `data/raw/archive/` layout |
| `No usable WAVs found for class 'X'` | WAV files missing from audio dir | Verify archive extraction |
| Stage 2 epoch 1 val_loss spikes to ~0.8 | Expected; BN layers re-warming | Normal — bug was fixed; checkpoint is protected |
| `model.predict` returns all values ~0.125 | Wrong checkpoint loaded (single-label softmax) | Verify `best_mobilenetv2_multilabel.h5`, not the old `_anc.h5` |
| TFLite full-INT8 conversion fails | TF op not supported in INT8 | Upgrade TF or use `dynamic_int8` variant |
| High MAE (>0.05) after INT8 conversion | Calibration set too small / random noise used | Run Stage 1 first, then Stage 4; check calibration path |

---

## Branch Overview

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `feature/multi-label-training` | Multi-label data prep and training |
| `bugfix/stage2-checkpoint-overwrite` | Fix for checkpoint regression in two-stage training |
| `feature/model-testing-docs` | Validation test suite and this documentation |
| `claude/train-embedded-model-Wisdb` | (merged) Initial TFLite / C-header export |
