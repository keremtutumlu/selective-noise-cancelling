# Selective Sound Removal

Query-conditioned source separation for selectively removing chosen sound classes (sirens, dog barks, keyboard typing, etc.) from any audio or video file, while leaving everything else untouched. A Gradio web app drives the full pipeline: upload → detect present sounds → tick which to remove → download the cleaned result.

## How it works

A FiLM-conditioned 2-D U-Net takes a 1-second log-magnitude spectrogram plus a one-hot **class query** and returns a soft spectrogram mask for that class. Removal is performed by combining `1 − strength × est²/mix²` ratio masks per selected class, applying spectrogram-level Hanning overlap-add across the file, and inverse-STFT-ing back to audio with the original mixture phase. For video uploads, the cleaned audio is muxed back over the original video track with `ffmpeg`.

## Pipeline

```bash
source venv/bin/activate
```

| Stage | Command | Output |
|---|---|---|
| 1. Train | `python src/model_training/train_conditioned_separator.py` | Trained `.h5` + class list under `saved_models/separation_models/` |
| 2. Evaluate | `python src/model_training/evaluate_conditioned_separator.py` | Per-class SI-SDR / SI-SDRi table |
| 3. Web app | `python src/application/webapp.py` | Gradio share URL |

Run stages 1 and 3 from the project root. The Colab notebooks under `notebooks/` mirror them for users without a local GPU.

`data/raw/` and `saved_models/` are gitignored — see `docs/conditioned_separation_guide.md` for dataset layout and download links.

## Spectrogram contract

| Parameter | Value |
|---|---|
| Sample rate | 16 000 Hz mono |
| FFT window | `n_fft=512` (32 ms) |
| Hop length | `hop_length=128` (8 ms) |
| Bins fed to model | 256 (Nyquist dropped) |
| Frames per window | 128 (≈1 s) |
| U-Net input | `(256, 128, 1)` log-magnitude + `(num_classes,)` one-hot query |
| U-Net output | `(256, 128, 1)` soft mask in `[0, 1]` |

## Datasets

The clip cache is assembled by `src/data_preparation/dataset_sources.py` from whatever it finds under `data/raw/`:

| Dataset | Location | Classes |
|---|---|---|
| ESC-50 | `data/raw/archive/` | 50 environmental |
| UrbanSound8K | `data/raw/urbansound8k/` | 10 urban (4 alias into ESC-50, 6 new) |

With both present the model trains on ~56 classes. Add another dataset by writing a `load_<name>()` returning `{class: [waveform]}` and calling it from `load_all_datasets`; the mixer, model query size, and training adapt automatically.

## Repository layout

```
selective-noise-cancelling/
├── src/
│   ├── data_preparation/           # dataset loaders + on-the-fly mixer
│   ├── model_training/             # FiLM U-Net + trainer + SI-SDR evaluator
│   └── application/                # Gradio web app
├── notebooks/                      # Colab counterparts (training + webapp)
├── docs/                           # conditioned_separation_guide.md
├── data/                           # raw + processed audio (gitignored)
└── saved_models/                   # Keras checkpoints (gitignored)
```

## Conventions

Branching, commit style, model file naming, and authorship rules live in `RULES.md`. Read it before opening a PR.
