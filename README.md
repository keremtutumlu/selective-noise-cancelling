# Selective Active Noise Cancellation (SNC) with Edge AI

## Project Overview

Traditional Active Noise Cancellation (ANC) systems apply blanket cancellation to all ambient noise, which creates safety hazards — blocking ambulance sirens or car horns, for example. This project builds a **Selective Noise Cancelling (SNC)** system using a lightweight CNN deployed on a budget MCU. The model classifies environmental audio in real time and applies destructive interference (phase inversion) **only** to user-defined noise categories, allowing critical sounds to pass through transparently.

---

## Current Status

Multi-label classification pipeline complete and validated on CPU.

| Metric | Value |
|--------|-------|
| Binary accuracy | 85.9% |
| AUC (multi-label) | 0.863 |
| Siren recall @ opt. threshold | **83.6%** |
| Host CPU inference latency | 39 ms (mean) |
| INT8 model size (target) | ~450 KB |

The model correctly identifies overlapping sound classes (e.g. siren + engine + rain simultaneously) and is quantised to TFLite INT8 for deployment on an ESP32-S3.

---

## Pipeline

Run stages in order. Activate the virtualenv first:

```bash
source venv/bin/activate
```

| Stage | Command | Output |
|-------|---------|--------|
| 1. Synthetic data generation | `python src/data_preparation/synthetic_data_generator.py` | `data/processed/training_pipeline/*.npy` |
| 2. Model training | `python src/model_training/train.py` | `saved_models/base_models/best_mobilenetv2_multilabel.h5` |
| 3. Model validation | `python tests/test_model.py` | Per-class F1 thresholds printed to stdout |
| 4. Edge optimisation | `python src/model_optimization/quantize_for_edge.py` | `saved_models/tflite/*.tflite` |
| 5. C array export | `python src/model_optimization/export_c_header.py` | `saved_models/tflm_c_array/g_anc_model.{h,cc}` |
| 6. Application sim | `cd src/application && python simulate_anc.py` | `anti_phase_test_sound.wav` |

See `docs/pipeline_guide.md` for prerequisites, expected runtimes, and troubleshooting.

---

## Architecture

### Audio Feature Representation

Raw audio → 16 kHz resample → Log-Mel spectrogram → Z-score normalise → 3-channel replicate

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 000 Hz |
| FFT window | 25 ms (`n_fft=400`) |
| Hop length | 10 ms (`hop_length=160`) |
| Mel bins | 64 |
| Output tensor | `(64, 101, 3)` |

### Model

MobileNetV2 α=0.35 backbone (ImageNet pretrained) + custom multi-label head:

```
GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Dropout(0.3) → Dense(8, sigmoid)
```

- **Loss:** Binary Cross-Entropy
- **Training:** Two-stage transfer learning — head-only warmup (frozen backbone), then full fine-tune at 1e-4
- **INT8 model:** ~450 KB — fits ESP32-S3 N8R8 flash without external PSRAM

### Target Classes

`car_horn`, `crying_baby`, `dog`, `engine`, `keyboard_typing`, `rain`, `siren` (always pass-through), `wind`

Classes are always kept in alphabetical order — this is a hard contract between the data pipeline, model, and inference code.

### Application Layer

1. **`ANCPredictor`** — extracts log-mel features from a 1-second audio window, returns per-class probabilities
2. **`ActiveNoiseCanceller`** — if top class is in the user's blocklist, inverts phase and applies 3 ms hardware-latency compensation
3. **`ANCAcousticSimulator`** — simulates speaker delay, measures RMS noise reduction in dB

---

## Repository Structure

```
selective-noise-cancelling/
├── src/
│   ├── data_preparation/       # Synthetic multi-label dataset generator
│   ├── model_training/         # Two-stage MobileNetV2 transfer learning
│   ├── model_optimization/     # TFLite conversion, INT8 quantisation, C array export
│   └── application/            # Inference, phase inversion, acoustic simulation
├── tests/
│   └── test_model.py           # Software-only validation suite (no hardware needed)
├── docs/
│   ├── model_and_data.md       # Architecture, feature engineering, training results
│   └── pipeline_guide.md       # Step-by-step run instructions, troubleshooting
├── data/                       # ESC-50 raw + processed features (gitignored)
└── saved_models/               # Keras .h5 and TFLite models (gitignored)
```

---

## Hardware Target

**ESP32-S3 DevKit** + **INMP441 MEMS microphone** (~$15–20 total)

The INT8 TFLite model is exported as an `alignas(16)` C array via `export_c_header.py` for direct inclusion in ESP-IDF or Arduino_TensorFlowLite firmware.
