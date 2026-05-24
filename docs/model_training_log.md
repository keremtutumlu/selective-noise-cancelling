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
**Trained:** — (pending, retrain from scratch)  
**Status:** Not yet trained

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

---

## How to add a new entry

After each training run, append a new `## v<X.Y>` section with:
- File name and training date
- Full hyperparameter table
- Dataset used
- Any architecture changes
- SI-SDRi results from `evaluate_conditioned_separator.py` (once available)
- Notes on what was changed and why
