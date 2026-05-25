# Model Training Log

One entry per trained checkpoint. Update this file after every training run.

---

## v1.0 — First working baseline

**File:** `separator_unet_film_multi_v1.0.h5`  
**Trained:** ~early May 2025 (approximate)  
**Status:** Working — detection and removal functional

### Hyperparameters
| Parameter | Value |
|---|---|
| `base_filters` | 32 |
| `batch_size` | 16 |
| `epochs` | 60 |
| `steps_per_epoch` | 500 |
| `n_val` | 800 |
| `learning_rate` | 1e-3 |
| `patience` | 10 |
| `negative_prob` | 0.25 |
| `bg_noise_prob` | 0.0 (no augmentation) |
| `bg_snr_db_range` | — |

### Dataset
- ESC-50 only (50 classes)

### Notes
- Established the FiLM-conditioned U-Net architecture and on-the-fly mixing pipeline.
- Detection and removal worked at this stage.
- Replaced by v1.x with UrbanSound8K added.

---

## v2.0 — Broken: aggressive augmentation caused training collapse

**File:** `separator_unet_film_multi_v2.0.h5`  
**Trained:** May 2025  
**Status:** BROKEN — do not use

### Hyperparameters
| Parameter | Value |
|---|---|
| `base_filters` | 32 |
| `batch_size` | 16 |
| `epochs` | 60 |
| `steps_per_epoch` | 500 |
| `n_val` | 800 |
| `learning_rate` | 1e-3 |
| `patience` | 10 |
| `negative_prob` | **0.45** |
| `bg_noise_prob` | **0.50** |
| `bg_snr_db_range` | **(5.0, 20.0) dB** |

### Dataset
- ESC-50 + UrbanSound8K (~56 classes)

### Failure analysis
Three compounding issues caused the model to fail completely:

1. **`negative_prob=0.45` is too high.** At 45 % negative examples the
   L1 loss strongly rewards outputting near-zero for everything, pushing
   the model into a "safe" silent-output equilibrium that satisfies
   negative examples at the cost of never learning positive examples well.

2. **`bg_snr_db_range=(5, 20) dB` is too aggressive.** At 5 dB SNR the
   noise amplitude is 56 % of the target signal. The model cannot
   reliably learn to separate a class when the supervision signal is
   buried under noise this heavy.

3. **Input scale mismatch at inference.** Training always peak-normalises
   each 1-second time-domain window to amplitude 1.0 before computing
   STFT. The webapp fed raw (unnormalised) audio → STFT magnitudes 3–10×
   smaller than the training distribution → model activations in an
   untrained regime.

### Symptoms
- Detection returned 0–2 irrelevant classes even for clear sounds (e.g.
  siren detected as something else).
- Removal had no audible effect on any class at any strength.

---

## v2.1 — Fixed augmentation, corrected inference normalisation

**File:** `separator_unet_film_multi_v2.1.h5`  
**Trained:** May 2026  
**Status:** Trained — partially working, superseded by v2.2

### Hyperparameters
| Parameter | Value | Change from v2.0 |
|---|---|---|
| `base_filters` | 32 | — |
| `batch_size` | 16 | — |
| `epochs` | 60 | — |
| `steps_per_epoch` | 500 | — |
| `n_val` | 800 | — |
| `learning_rate` | 1e-3 | — |
| `patience` | 10 | — |
| `negative_prob` | **0.30** | ↓ from 0.45 |
| `bg_noise_prob` | **0.10** | ↓ from 0.50 |
| `bg_snr_db_range` | **(15.0, 30.0) dB** | ↑ from (5, 20) |

### Dataset
- ESC-50 + UrbanSound8K (~56 classes)

### Changes vs v2.0
- `negative_prob` dialled back to 0.30: enough to suppress absent-class
  false positives without starving the positive-example gradient.
- `bg_noise_prob` reduced to 0.10: mild noise augmentation is kept for
  domain-gap robustness but no longer dominates the training distribution.
- `bg_snr_db_range` raised to (15, 30) dB: noise is 6–18 % of signal
  amplitude instead of 56 %, matching realistic real-world background levels.
- Inference normalisation fixed: `remove_sounds` now peak-normalises the
  full audio to time-domain |peak|=1.0 before STFT, matching training.
  `detect_sounds` already normalised per-window (no change needed).

### How to train
```bash
# Locally
python src/model_training/train_conditioned_separator.py

# On Colab
# Open notebooks/colab_train_conditioned_separator.ipynb, T4 GPU
```

Output is written to `saved_models/separation_models/separator_unet_film_multi_v2.1.h5`.

### Evaluation results (v2.1)

| Metric | Value |
|---|---|
| Diagnose overall | **PASS** (HEALTHY) |
| SI-SDRi average | **-22.18 dB** |
| Detection mean F1 | **0.21** |
| Detection FP : TP | **5.5 : 1** |

**Diagnose detail:** FiLM conditioning works (correct query > wrong query on most classes). Two failing classes: `cat` (−1.00× advantage) and `chirping_birds` (−0.52×).

**SI-SDR detail:** All classes show negative SI-SDRi, meaning the model's reconstructed stem is worse than just using the mixture as an estimate of the stem. Partly a mixture-phase-reuse limitation (spectrogram U-Net), partly evidence that the mask is not tight enough. Best classes: helicopter (−3.4 dB), frog (−10.0 dB), airplane (−6.9 dB).

**Detection detail:** Precision very low (0.10–0.25 for most classes), recall moderate (0.3–0.6). Cutoff too permissive, causing 5.5× more FP than TP. Several classes (chirping_birds, crackling_fire, fireworks, glass_breaking, keyboard_typing, water_drops) had F1 = 0.

**Listening tests (real audio):** Audio quality improved vs v2.0. Removal audible on selected class (e.g., air_conditioner). Two confirmed issues:
1. Regular pulsing artifact at ~0.25 s intervals (OLA boundary, 50% overlap → fixed in v2.2).
2. Too many irrelevant classes detected (FP rate → fixed by tighter cutoff in v2.2).

---

## v2.2 — Full-encoder FiLM + multi-resolution loss + tighter detection

**File:** `separator_unet_film_multi_v2.2.h5`  
**Trained:** — (pending, retrain from scratch)  
**Status:** Not yet trained

### Hyperparameters
| Parameter | Value | Change from v2.1 |
|---|---|---|
| `base_filters` | 32 | — |
| `batch_size` | 16 | — |
| `epochs` | 60 | — |
| `steps_per_epoch` | 500 | — |
| `n_val` | 800 | — |
| `learning_rate` | 1e-3 | — |
| `patience` | 10 | — |
| `negative_prob` | 0.30 | — |
| `bg_noise_prob` | 0.10 | — |
| `bg_snr_db_range` | (15.0, 30.0) dB | — |

### Dataset
- ESC-50 + UrbanSound8K (~56 classes)
- FSD50K support added (auto-detected; adds ~200 classes when present)

### Architecture changes vs v2.1
- **FiLM at every encoder level** (e1, e2, e3, e4, bottleneck) instead of bottleneck only. Each level gets its own gamma/beta projection from the shared 128-dim query embedding. Skip connections entering the decoder already carry class-specific activations, improving mask precision.

### Training changes vs v2.1
- **Multi-resolution L1 loss** (full + ½-res + ¼-res spatial pooling, weights 1.0 + 0.5 + 0.25). Coarser scales stabilise the overall spectral shape before fine detail is optimised.

### Inference changes vs v2.1
- **OLA step**: `TIME_FRAMES // 4` (32 frames ≈ 0.25 s, 75% overlap) instead of `TIME_FRAMES // 2` (50% overlap). Confirmed to reduce audible pulsing artifact at chunk boundaries.
- **Detection scoring**: `energy_ratio × (1 + CoV²)` instead of `(1 + CoV)`. Squared CoV amplifies the gap between specific and diffuse detections.
- **Detection cutoff**: 0.65 × winner (was 0.40). Tighter relative threshold reduces FP rate.

### How to train
```bash
python src/model_training/train_conditioned_separator.py
```

---

## How to add a new entry

After each training run, append a new `## v<X.Y>` section with:
- File name and training date
- Full hyperparameter table
- Dataset used
- Any architecture changes
- SI-SDRi results from `evaluate_conditioned_separator.py` (once available)
- Notes on what was changed and why
