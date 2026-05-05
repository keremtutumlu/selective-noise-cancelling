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

The pipeline has four sequential stages:

**1. Single-label data preparation** (ESC-50 → `.npy` arrays):
```bash
python src/data_preparation/data_prep.py
```
Outputs: `data/processed/X_features.npy`, `data/processed/y_labels.npy`

**2. Multi-label synthetic data generation** (mixes 1–3 sounds per sample):
```bash
python src/data_preparation/synthetic_data_generator.py
```
Outputs: `data/processed/training_pipeline/X_multi_features.npy`, `data/processed/training_pipeline/y_multi_labels.npy`

**3. Model training**:
```bash
python src/model_training/train.py
```
Outputs: `saved_models/base_models/best_mobilenetv2_anc.h5`

**4. Application scripts** — must be run from within `src/application/` due to local relative imports:
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
ESC-50 raw WAVs → AudioPreprocessor → X_features.npy / y_labels.npy   (single-label)
                → SyntheticDatasetBuilder → X_multi_features.npy / y_multi_labels.npy  (multi-label)
                → ModelTrainer (MobileNetV2) → best_mobilenetv2_anc.h5
                → ANCPredictor → SystemOrchestrator → ActiveNoiseCanceller
```

### Feature Representation
All audio is resampled to **16 kHz**, then converted to a **Log-Mel Spectrogram** with a 25ms FFT window (`n_fft=400`), 10ms hop (`hop_length=160`), and 64 Mel bins. The single-channel spectrogram is **triplicated into 3 channels** (shape `(64, 101, 3)`) to satisfy MobileNetV2's RGB input requirement. Features are Z-score normalized per sample.

### Two Training Modes
- **Single-label** (`data_prep.py` + `train.py`): softmax output, categorical cross-entropy. Uses LabelEncoder (alphabetical ordering).
- **Multi-label** (`synthetic_data_generator.py`): sigmoid output, binary cross-entropy, multi-hot encoded labels. Classes are explicitly sorted alphabetically to maintain consistent index mapping.

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
- Planned: INT8 quantization + structural pruning via TensorFlow Lite for Microcontrollers (TFLM)
