# Source-Separation Pipeline — Run Guide

How to run the **source-separation** track of the Selective Noise
Cancellation (SNC) project: train a U-Net that splits a mixture of
sounds into 8 per-class stems, then remove or attenuate any chosen
sound while the rest pass through.

This track lives on the `feature/source-separation` branch and is
separate from the classification pipeline documented in
`pipeline_guide.md`. The two share the ESC-50 dataset and the same
8-class contract, but use different models and produce different
artefacts.

---

## 1. What this pipeline does

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `src/data_preparation/separation_dataset.py` | Generate synthetic mixtures + isolated per-class stems |
| 2 | `src/model_training/train_separator.py` | Train the separation U-Net |
| 3 | `src/model_training/evaluate_separator.py` | Score separation quality with SI-SDR |
| 4 | `src/application/selective_separation.py` | Remove/attenuate sounds in an audio file |

The model itself is defined in `src/model_training/separator_unet.py`.

```
ESC-50 WAVs
    -> SeparationDatasetBuilder  -> mixtures.npy, stems.npy
    -> SeparatorTrainer (U-Net)  -> best_separator_unet.h5
    -> SeparatorEvaluator        -> SI-SDR results
    -> SelectiveSeparator        -> selected sounds removed
```

---

## 2. Recommended: run on Google Colab

Training the U-Net is GPU-heavy. The bundled notebook runs the whole
pipeline on a Colab GPU and persists everything to Google Drive.

1. Open `notebooks/colab_train_separator.ipynb` in Colab:
   `colab.research.google.com` → *File → Open notebook → GitHub* →
   `keremtutumlu/selective-noise-cancelling` → branch
   `feature/source-separation` → `notebooks/colab_train_separator.ipynb`.
2. **Runtime → Change runtime type → GPU** (pick **High-RAM** if your
   plan offers it — Stage 2 holds several GB of spectrograms in memory).
3. If the repository is private, add a GitHub token as a Colab Secret
   named `GITHUB_TOKEN` (key icon in the sidebar, scope `repo`).
4. Run the cells top to bottom. ESC-50 downloads automatically the
   first time; data and checkpoints are written to
   `MyDrive/snc/` so a runtime disconnect never loses progress.

The notebook ends with an interactive stage: upload any audio file,
choose which classes to remove, and download the result.

---

## 3. Alternative: run locally

Only practical if you have a reasonably fast machine and enough RAM
(~8 GB free for Stage 2). Activate the project virtualenv first:

```bash
source venv/bin/activate
```

ESC-50 must be present at `data/raw/archive/` with this layout:

```
data/raw/archive/
├── esc50.csv
└── audio/audio/*.wav
```

### Stage 1 — Generate the separation dataset

```bash
python src/data_preparation/separation_dataset.py
```

**Outputs** (`data/processed/separation_pipeline/`):

| File | Shape | Notes |
|------|-------|-------|
| `mixtures.npy` | `(N, 16000)` | float32 mixed waveforms (1 s, 16 kHz) |
| `stems.npy` | `(N, 8, 16000)` | float32 isolated per-class stems |
| `class_names.json` | — | Alphabetical 8-class list |

Default `N = 6000` → `stems.npy` is ~3.1 GB. Lower `num_samples` in the
script's `__main__` block if disk or RAM is tight.

### Stage 2 — Train the separation U-Net

```bash
python src/model_training/train_separator.py
```

Converts the waveforms to magnitude spectrograms, trains the U-Net,
and saves the best checkpoint by validation loss.

**Output:** `saved_models/separation_models/best_separator_unet.h5`

**Runtime:** ~30–45 min on a Colab T4 GPU; several hours on a laptop CPU.

To preview the model architecture without training:

```bash
python src/model_training/separator_unet.py
```

### Stage 3 — Evaluate (SI-SDR)

```bash
python src/model_training/evaluate_separator.py
```

Reconstructs waveforms from the model and scores them against the
ground-truth stems on the held-out test split. Prints a per-class
table:

```
Class                 N    SI-SDR mix    SI-SDR model     SI-SDRi
car_horn            210      -0.40 dB        6.80 dB     +7.20 dB
...
AVERAGE             ...
```

* **SI-SDR mix** — quality of the untouched mixture (the "do nothing"
  baseline).
* **SI-SDR model** — quality of the model's separated stem.
* **SI-SDRi** — improvement of the model over doing nothing. This is
  the headline number to report in the thesis.

### Stage 4 — Remove sounds from an audio file

```bash
python src/application/selective_separation.py
```

By default this removes `engine` and `wind` from
`data/test_samples/test_sound.wav` and writes `separated_output.wav`.
Edit the `gains` dict in the script's `__main__` block to choose
different classes — `0.0` removes a sound, `1.0` keeps it, `0.5`
halves it.

---

## 4. The model in brief

* **Architecture:** 2-D U-Net (encoder/decoder with skip connections),
  `separator_unet.py`. ~8M parameters at the default size.
* **Input:** log-compressed magnitude spectrogram of a 1-second mixture,
  `256 x 128` (frequency x time).
* **Output:** 8 soft masks in `[0, 1]`, one per class. Each mask
  multiplied by the mixture magnitude estimates that class's stem.
* **Loss:** L1 (mean absolute error) between estimated and true stem
  magnitudes.
* **STFT contract:** 16 kHz, `n_fft=512`, `hop_length=128`. These
  constants are defined once in `separator_unet.py` and imported by the
  training, evaluation, and application code.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: esc50.csv` | Dataset not in place | Check `data/raw/archive/` layout |
| Stage 2 runs out of memory | `stems.npy` + spectrograms too large for RAM | Use a Colab High-RAM runtime, or lower `num_samples` in Stage 1 |
| Stage 2 extremely slow | Training on CPU | Use a Colab GPU runtime |
| `Trained separator not found` in Stage 4 | Stage 2 not finished | Run Stage 2 first; check `saved_models/separation_models/` |
| Output audio has faint clicks at 1-second marks | Window-boundary artefacts | Expected for the PoC — windows are processed independently |
| SI-SDRi is near 0 dB or negative | Model under-trained | Train longer / generate more data in Stage 1 |

---

## 6. How this fits the thesis

The separation U-Net is an original model you build, train, and
evaluate — that is the core contribution. AudioSep
(`notebooks/colab_source_separation.ipynb`) is a much larger
pre-trained model used only as a **reference baseline**: a thesis
chapter can compare your compact, ESC-50-specialised separator against
a general-purpose state-of-the-art system, discussing the quality vs.
size vs. specialisation trade-offs.

Suggested evaluation to report:

* Per-class and average SI-SDR / SI-SDRi from Stage 3.
* Model size and training time vs. AudioSep.
* Qualitative listening examples from Stage 4.
* Discussion of failure cases (overlapping sounds, the window-boundary
  artefact, classes with few ESC-50 examples).
