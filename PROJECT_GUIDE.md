# Project Guide

This file documents the project's architecture and conventions for anyone working in this repository.

## Project Overview

Query-conditioned source separation for selective sound removal. A FiLM-conditioned U-Net is told *which* sound class to extract from a 1-second audio window and emits a soft spectrogram mask for it. A Gradio web app drives the pipeline end-to-end: upload audio/video → detect present classes → tick which ones to remove → render the cleaned output (audio, and video if the upload had a video track).

> Project conventions (branch names, commits, model file naming, no AI footprint) are defined in `RULES.md`. Read it before any new work — it overrides defaults.

## Environment Setup

```bash
source venv/bin/activate
```

Key deps: TensorFlow 2.21.0, librosa 0.11.0, soundfile, gradio, numpy, pandas. `ffmpeg` and `ffprobe` must be on PATH (the webapp shells out to them for video I/O).

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Train | `python src/model_training/train_conditioned_separator.py` | `saved_models/separation_models/best_conditioned_separator.h5` + `conditioned_class_names.json` |
| 2. Evaluate (SI-SDR) | `python src/model_training/evaluate_conditioned_separator.py` | Per-class SI-SDRi table on stdout |
| 3. Webapp | `python src/application/webapp.py` | Gradio share URL |

The Colab notebooks under `notebooks/` mirror stages 1 and 3 for users without a local GPU.

> `data/` and `saved_models/` are gitignored and must be generated/downloaded locally.

## Architecture

### Data flow
```
ESC-50 + UrbanSound8K (raw WAVs)
    → dataset_sources.load_all_datasets        # in-memory {class: [waveform]} cache
    → SeparationMixer                          # on-the-fly mixtures, queries, target stems
    → ConditionedSeparatorTrainer              # FiLM U-Net, L1 on magnitude
    → best_conditioned_separator.h5
    → webapp.py                                # detect → mask → ISTFT → render
```

### Spectrogram contract
All audio is resampled to **16 kHz mono**. STFT: `n_fft=512`, `hop_length=128`, the Nyquist bin is dropped so the U-Net sees `(FREQ_BINS=256, TIME_FRAMES=128, 1)` log-magnitude inputs. One U-Net call covers ~1 second.

### Model
FiLM-conditioned 2-D U-Net (`src/model_training/conditioned_separator.py`):
- Inputs: log-magnitude `(256, 128, 1)`, one-hot `class_query` `(num_classes,)`
- The query is embedded and turned into per-channel FiLM scale/shift that modulate the bottleneck features
- Output: sigmoid soft mask `(256, 128, 1)` in `[0, 1]`
- Training wrapper multiplies the mask by the linear magnitude inside the graph so the saved `.h5` has three inputs (`[log_mag, class_query, linear_mag]`) and outputs an estimated stem magnitude — no Lambda layers, reloads without custom objects.
- Loss: L1 between estimated and true stem magnitude.

### Datasets
`src/data_preparation/dataset_sources.py` merges every dataset it finds under `data/raw/`:
- ESC-50 at `data/raw/archive/` (50 environmental classes)
- UrbanSound8K at `data/raw/urbansound8k/` (10 urban; 4 alias into ESC-50 via `CLASS_ALIASES`, 6 are new)

With both present the mixer exposes ~56 classes. Adding a dataset = writing a new `load_<name>()` and calling it from `load_all_datasets`; the mixer, model query size, and training adapt automatically.

### Mixer
`SeparationMixer` (`src/data_preparation/separation_mixer.py`) draws an infinite stream of `(mixture, query, target_stem)` triples. Each example mixes 1–`max_mix` random clips at amplitudes drawn from `amp_range`. With probability `negative_prob` the query asks for an absent class and the target is silence — these negatives teach the model to output near-zero masks for missing classes.

### Web app
`src/application/webapp.py` (Gradio):
- `detect_sounds` — runs the U-Net per class on a few seconds of input and scores `(energy_ratio) × (1 + mask_specificity)`; classes above an absolute floor and a relative gap are surfaced as checkboxes.
- `remove_sounds` — computes one full-file STFT; for each `TIME_FRAMES`-frame chunk (50% overlap) runs the U-Net for every selected class, combines multiplicative power-ratio masks (`mask = clip(1 − strength·est²/mix², 0, 1)`), smooths the mask over time, applies Hanning-weighted overlap-add on the magnitude, then a single ISTFT with the original phase. For video uploads it muxes the cleaned audio back over the original video track via `ffmpeg`.

## Naming and version control

Per `RULES.md`:
- **Branches:** `feature/<desc>`, `bugfix/<desc>`, `experiment/<desc>`. No AI references.
- **Commits:** Capitalised past-tense verb, atomic — e.g. *"Added multi-resolution STFT loss"*, *"Fixed audio click at chunk boundaries"*.
- **Saved models:** `<task>_<architecture>_<dataset-or-keyfeature>_v<major>.<minor>.<ext>` (e.g. `separator_unet_film_multi_v1.0.h5`). Matching metadata files share the prefix.
