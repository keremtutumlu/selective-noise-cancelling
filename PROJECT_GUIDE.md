# Project Guide

This file documents the project's architecture and conventions for anyone working in this repository.

## Project Overview

Selective Active Noise Cancellation (SNC) using a MobileNetV2 CNN deployed on Edge AI hardware. The system classifies environmental audio in real-time and applies phase inversion **only** to user-defined noise categories, allowing critical sounds (e.g., sirens) to pass through.

## Environment Setup

A virtualenv is at `venv/`. Key dependencies: TensorFlow 2.21.0, librosa 0.11.0, soundfile, scikit-learn, numpy, pandas. There is no `requirements.txt` — install from the venv or manually.

```bash
source venv/bin/activate
```

## Running the Pipeline

The pipeline has five sequential stages:

**1. Multi-label synthetic data generation** (mixes 1–3 ESC-50 clips per sample, in-memory cache):
```bash
python src/data_preparation/synthetic_data_generator.py
```
Outputs: `data/processed/training_pipeline/{X_multi_features.npy, y_multi_labels.npy, class_names.json}`

**2. Model training** (multi-label MobileNetV2 α=0.35, two-stage transfer learning):
```bash
python src/model_training/train.py
```
Outputs: `saved_models/base_models/best_mobilenetv2_multilabel.h5`

**3. Edge optimization** (TFLite conversion + INT8 quantization + benchmark):
```bash
python src/model_optimization/quantize_for_edge.py
```
Outputs: `saved_models/tflite/{model_float32, model_float16, model_dynamic_int8, model_full_int8}.tflite`

**4. C array export for TFLite Micro**:
```bash
python src/model_optimization/export_c_header.py
```
Outputs: `saved_models/tflm_c_array/{g_anc_model.h, g_anc_model.cc}`

**5. Application scripts** — must be run from within `src/application/` due to local relative imports:
```bash
cd src/application
python simulate_anc.py        # End-to-end: classify → generate anti-noise
python verify_cancellation.py # Acoustic simulation: measure dB noise reduction
python inference.py           # Standalone inference on a test audio file
```

Test audio files go in `data/test_samples/test_sound.wav`. `simulate_anc.py` must run before `verify_cancellation.py` (it produces `anti_phase_test_sound.wav`).

> `data/` and `saved_models/` are gitignored and must be generated locally.

## Architecture

### Data Flow
```
ESC-50 raw WAVs → MultiLabelDatasetBuilder → X_multi_features.npy / y_multi_labels.npy / class_names.json
                → MultiLabelTrainer (MobileNetV2 α=0.35) → best_mobilenetv2_multilabel.h5
                → EdgeModelOptimizer → model_full_int8.tflite
                → tflite_to_c_array → g_anc_model.{h,cc}
                → ANCPredictor → SystemOrchestrator → ActiveNoiseCanceller
```

### Feature Representation
All audio is resampled to **16 kHz**, then converted to a **Log-Mel Spectrogram** with a 25ms FFT window (`n_fft=400`), 10ms hop (`hop_length=160`), and 64 Mel bins. The single-channel spectrogram is **triplicated into 3 channels** (shape `(64, 101, 3)`) to satisfy MobileNetV2's RGB input requirement. Features are Z-score normalized per sample.

### Training Mode
Multi-label only: **sigmoid** output, **binary cross-entropy** loss, multi-hot labels. Backbone is MobileNetV2 with width multiplier **α=0.35** so the INT8-quantised model fits a budget MCU (e.g. ESP32-S3) without external PSRAM. Training is two-stage: (1) head-only with frozen ImageNet backbone, (2) full-network fine-tune at 1e-4. Class indices are positions in the alphabetically-sorted `CANONICAL_CLASSES` tuple; `class_names.json` is emitted alongside the .npy files so downstream code does not hard-code the order.

### Critical Invariant: Class Ordering
Classes are **always sorted alphabetically** in both training and inference. The canonical 8-class order is:
`car_horn, crying_baby, dog, engine, keyboard_typing, rain, siren, wind`

`ANCPredictor` enforces `sorted(class_labels)` on init — passing unsorted labels will silently misalign predictions with model weights.

### Application Layer
`SystemOrchestrator` (`simulate_anc.py`) wires together:
1. `ANCPredictor` — loads `.h5` model, extracts features, returns `{class: probability}` dict sorted by confidence
2. `ActiveNoiseCanceller` — if top class is in the user's blocklist, applies `signal × -1.0` (phase inversion) and shifts left by `latency_samples` (default: 48 samples at 3ms hardware delay)
3. `ANCAcousticSimulator` (`verify_cancellation.py`) — simulates the speaker delay by rolling the anti-noise signal back right, adds original + anti-noise, computes dB RMS reduction

### Edge AI Constraints
- 1-second audio windows to meet <50ms MCU latency
- MobileNetV2 chosen over 1D-CNN for computational efficiency
- INT8 quantization via `src/model_optimization/quantize_for_edge.py` produces a TFLite Micro-compatible model; `export_c_header.py` packages it as an `alignas(16)` C array for direct inclusion in ESP-IDF / STM32CubeIDE projects.
