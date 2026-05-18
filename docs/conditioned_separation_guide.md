# Conditioned Separation Pipeline — Run Guide

How to run the **query-conditioned** source-separation track: train a
U-Net that is told *which* sound class to extract and removes any chosen
sounds from an audio file while the rest pass through.

This track lives on the `feature/conditioned-separation` branch. It
supersedes the fixed 8-class separator (`feature/source-separation`):
it scales to all 50 ESC-50 classes — and to additional datasets — with
a single architecture.

---

## 1. Why this track exists

The fixed separator emitted one mask per class and pre-generated a
`stems.npy` file sized `N x num_classes x 16000`. Neither scales: at 50
classes the file is ~19 GB and 47 of every 50 mask channels just
predict silence.

This track fixes both:

* **On-the-fly mixing** — mixtures are synthesised during training from
  an in-memory clip cache. No dataset file; the class count is
  unbounded.
* **Query conditioning** — the U-Net takes a one-hot *class query* and
  emits a single mask for that class. Class count only changes the
  query input size, never the convolutional body.

| Component | File |
|-----------|------|
| Model | `src/model_training/conditioned_separator.py` |
| Mixer | `src/data_preparation/separation_mixer.py` |
| Training | `src/model_training/train_conditioned_separator.py` |
| Evaluation | `src/model_training/evaluate_conditioned_separator.py` |
| Application | `src/application/conditioned_selective_separation.py` |
| Colab notebook | `notebooks/colab_train_conditioned_separator.ipynb` |

---

## 2. GPU note — use a T4

Newer Colab GPUs (A100, L4, Blackwell) can fail with
`CUDA_ERROR_INVALID_PTX`: the installed TensorFlow has no compiled
kernels for them. **Runtime → Change runtime type → GPU → T4** avoids
this. T4 is fully supported and trains this model fine.

---

## 3. Recommended: run on Google Colab

1. Open `notebooks/colab_train_conditioned_separator.ipynb` in Colab:
   `colab.research.google.com` → *File → Open notebook → GitHub* →
   `keremtutumlu/selective-noise-cancelling` → branch
   `feature/conditioned-separation`.
2. **Runtime → Change runtime type → GPU (T4)**.
3. Private repo? Add a GitHub token as a Colab Secret named
   `GITHUB_TOKEN` (key icon, scope `repo`).
4. Run the cells top to bottom. ESC-50 downloads automatically the
   first time; the model checkpoint is written to `MyDrive/snc/`.

The notebook ends with an interactive stage: upload any audio file,
choose which classes to remove, and download the result.

---

## 4. Alternative: run locally

Activate the project virtualenv first:

```bash
source venv/bin/activate
```

ESC-50 must be at `data/raw/archive/` (`esc50.csv` +
`audio/audio/*.wav`).

### Stage 1 — Train

```bash
python src/model_training/train_conditioned_separator.py
```

Streams on-the-fly mixtures through the conditioned U-Net.

**Outputs** (`saved_models/separation_models/`):

* `best_conditioned_separator.h5` — best checkpoint by validation loss.
* `conditioned_class_names.json` — class list for the query vector.

**Runtime:** ~45–90 min on a Colab T4; much longer on CPU.

To preview the architecture without training:

```bash
python src/model_training/conditioned_separator.py
```

### Stage 2 — Evaluate (SI-SDR)

```bash
python src/model_training/evaluate_conditioned_separator.py
```

Builds a fixed positive-only test set, queries the model for the
present class in each mixture, reconstructs waveforms, and prints a
per-class table:

```
Class                  N    SI-SDR mix    SI-SDR model    SI-SDRi
dog                   18      -1.10 dB        5.90 dB     +7.00 dB
...
AVERAGE              ...
```

* **SI-SDR mix** — the unprocessed mixture (baseline).
* **SI-SDR model** — the model's separated stem.
* **SI-SDRi** — improvement of the model over doing nothing. The
  headline number for the thesis.

### Stage 3 — Remove sounds from an audio file

```bash
python src/application/conditioned_selective_separation.py
```

By default this removes `engine` and `wind` from
`data/test_samples/test_sound.wav` and writes
`conditioned_separated_output.wav`. Edit the `gains` dict in the
script's `__main__` block — `0.0` removes a sound, `1.0` keeps it,
`0.5` halves it. Classes not listed are left untouched.

---

## 5. The model in brief

* **Architecture:** conditioned 2-D U-Net (`conditioned_separator.py`).
  Encoder/decoder with skip connections; the class query is embedded
  into **FiLM** parameters (per-channel scale and shift) that modulate
  the bottleneck.
* **Inputs:** log-magnitude spectrogram `(256, 128, 1)` + one-hot
  `class_query` `(num_classes,)`.
* **Output:** one soft mask `(256, 128, 1)` for the queried class.
* **Loss:** L1 between the estimated and true target-stem magnitude.
* **STFT contract:** 16 kHz, `n_fft=512`, `hop_length=128`, 1-second
  windows — constants in `conditioned_separator.py`.

---

## 6. Datasets and adding more

The clip cache is built by `src/data_preparation/dataset_sources.py`,
which merges every dataset found under `data/raw/`:

| Dataset | Location | Classes |
|---------|----------|---------|
| ESC-50 | `data/raw/archive/` | 50 environmental |
| UrbanSound8K | `data/raw/urbansound8k/` | 10 urban (4 merge into ESC-50, 6 new) |

Overlapping classes are merged via `CLASS_ALIASES` in
`dataset_sources.py` (e.g. UrbanSound8K `dog_bark` pools into ESC-50
`dog`). With both datasets present the model trains on ~56 classes.

The Colab notebook downloads both datasets automatically. To download
UrbanSound8K manually, fetch
`https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz` (~6 GB),
extract it, and place the `UrbanSound8K` folder at
`data/raw/urbansound8k/` (so `metadata/UrbanSound8K.csv` and
`audio/fold1/...` exist).

**To add another dataset** (e.g. FSD50K): write a `load_<name>`
function in `dataset_sources.py` returning a `{class_name: [waveform]}`
dict, and call it from `load_all_datasets`. Nothing else changes — the
mixer, model query size, and training all adapt to the new class count
automatically. Note: very large datasets that exceed RAM would need the
cache changed from load-all-into-memory to lazy/disk-backed loading.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA_ERROR_INVALID_PTX` | GPU too new for the TF build | Use a T4 runtime |
| `FileNotFoundError: esc50.csv` | Dataset not in place | Check `data/raw/archive/` layout |
| Training very slow | Running on CPU | Use a Colab GPU runtime |
| GPU under-used during training | Generator is the bottleneck | Acceptable for a PoC; raise `batch_size` if RAM allows |
| `Trained model not found` in Stage 3 | Stage 1 not finished | Run Stage 1 first |
| Faint clicks at 1-second marks in output | Window-boundary artefacts | Expected for the PoC |
| SI-SDRi near 0 dB | Model under-trained | Raise `epochs` / `steps_per_epoch` in the trainer |

---

## 8. How this fits the thesis

The conditioned separator is an original model you design, train, and
evaluate — the core contribution. Good things to report:

* Per-class and average SI-SDR / SI-SDRi from Stage 2.
* The architecture choice: why query conditioning scales where a
  fixed multi-output model does not (the class-imbalance argument).
* A comparison against AudioSep
  (`notebooks/colab_source_separation.ipynb`) as a large pre-trained
  baseline — compact specialised model vs. general-purpose system.
* Qualitative listening examples from Stage 3.
* Limitations: window-boundary artefacts, classes with few ESC-50
  examples, overlapping sounds of similar timbre.
