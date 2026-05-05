# Selective Active Noise Cancellation (SNC) with Edge AI

## 📖 Project Abstract
Traditional Active Noise Cancellation (ANC) systems rely on the FxLMS algorithm, which applies a blanket cancellation to all ambient noise. This creates safety hazards (e.g., blocking ambulance sirens or car horns). This project aims to develop a **Selective Noise Cancelling (SNC)** system using Deep Learning. By utilizing an optimized Convolutional Neural Network (CNN) deployed on an Edge AI embedded system (MCU), the system classifies environmental audio in real-time and applies destructive interference (phase inversion) **only** to user-defined noise categories, allowing critical sounds to pass through transparently.

---

## 🚀 Current Status: Proof of Concept (PoC) Completed
As of March 2026, the initial PoC phase for single-label sound isolation has been successfully validated:
- **Accuracy:** Achieved **84.38%** test accuracy on unseen data using Transfer Learning.
- **Acoustic Validation:** Simulated phase inversion ($S_{anti} = -1.0 \times S_{in}$) yielded a **75.10 dB Noise Reduction** via destructive interference.
- **Architecture Pivot:** Transitioned from a computationally expensive 1D-CNN (Waveform) to a highly optimized **2D-CNN (MobileNetV2)** using 3-channel Log-Mel Spectrograms to meet the strict <50ms latency constraint of Edge devices.

---

## 🏗️ System Architecture & Workflow

The repository is modularized strictly by lifecycle stages to ensure Clean Code and Separation of Concerns.

### 1. Data Preparation (`src/data_preparation/`)
* **Objective:** Prepare raw acoustic data (ESC-50) for neural network ingestion.
* **Pipeline:** Resamples audio to 16kHz to relieve MCU overhead. Generates 1-second continuous random windows to meet real-time processing limits.
* **Multi-Label Augmentation (Current Phase):** Synthetically mixes 1 to 3 isolated sounds (e.g., Wind + Siren) at varying Signal-to-Noise Ratios (SNR) to simulate real-world overlapping audio. Outputs Multi-Hot encoded vectors.

### 2. Model Training (`src/model_training/`)
* **Current Backbone:** `MobileNetV2` initialized with ImageNet weights.
* **Topology:**
  - *Input:* (64, 101, 3) representing (Mel_Bins, Time_Steps, Channels).
  - *Activation:* Transitioning from `Softmax` (Single-label) to **`Sigmoid`** (Multi-label).
  - *Loss Function:* Transitioning from `Categorical Crossentropy` to **`Binary Crossentropy`**.
* **Edge Optimization (Upcoming):** Post-training 8-bit Quantization (INT8) and Structural Pruning via TensorFlow Lite for Microcontrollers (TFLM).

### 3. Application & Cancellation Engine (`src/application/`)
* **Predictor:** Continuously monitors the 1-second sliding audio buffer.
* **Canceller (Predictive ANC):** If a banned class is detected, the DSP module shifts the audio signal forward in time to compensate for hardware latency (e.g., 3ms) and inverts its phase by 180 degrees.
* **Verification:** Built-in digital acoustic simulator to calculate RMS energy reduction.

---

## 📂 Repository Structure

```text
selective-noise-cancelling/
├── data/                       # Contains raw ESC-50 and processed .npy arrays (Ignored in Git)
├── docs/                       # Academic reports, literature, and architectural decisions
│   └── Bitirme Ara Rapor Mart.pdf
├── src/                        # Source Code
│   ├── data_preparation/       # Audio resampling, feature extraction, and synthetic mixing
│   ├── model_training/         # Model architectures, training loops, and callbacks
│   └── application/            # Real-time inference logic and DSP cancellation module
└── README.md                   # Project documentation