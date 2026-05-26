# Automation Guide

How to run unattended training-evaluation sweeps on Colab Pro+ and compare
the resulting models in the webapp without editing code.

This guide describes the **target architecture** for the automation. The
pieces are implemented in:

* `notebooks/colab_automated_training.ipynb` — orchestration notebook
* `experiments/experiments.json` — hyperparameter variants to try
* `experiments/results.csv` — per-run metric log (written to Drive)
* `src/application/webapp.py` — model dropdown UI

---

## Goal

Run a multi-experiment sweep unattended:

1. Open the automation notebook on Colab Pro+ → "Run all" → close browser.
2. Background execution trains every variant in `experiments.json`, one
   after another. Each variant is fully evaluated and a row is appended
   to `results.csv` on Drive.
3. Come back later, launch the webapp, and pick any model from the
   dropdown to compare them by ear.

No code edits between experiments. No babysitting.

---

## Architecture

### 1. Experiment config

`experiments/experiments.json` defines what to try:

```json
{
  "experiments": [
    {
      "id": "v2.3_baseline",
      "negative_prob": 0.15,
      "bg_noise_prob": 0.10,
      "epochs": 60
    },
    {
      "id": "v2.3_neg010",
      "negative_prob": 0.10
    },
    {
      "id": "v2.3_neg020",
      "negative_prob": 0.20
    }
  ]
}
```

Every field except `id` is optional and overrides the default in
`ConditionedSeparatorTrainer.__init__` / `SeparationMixer.__init__`.
The `id` becomes the model's filename suffix:
`separator_unet_film_multi_<id>.h5`.

### 2. Orchestration notebook

`notebooks/colab_automated_training.ipynb` runs the sweep:

```
Stage 0 — Mount Drive, clone repo, install deps
Stage 1 — Download datasets (ESC-50 + UrbanSound8K + FSD50K, only if missing)
Stage 2 — Load experiments.json
Stage 3 — For each experiment:
            a. Build trainer with overridden hyperparameters
            b. Train → save separator_unet_film_multi_<id>.h5 to Drive
            c. Run diagnose → record PASS/FAIL + diagnose metrics
            d. Run evaluate_conditioned_separator → record SI-SDRi
            e. Run evaluate_detection → record mean F1
            f. Append row to results.csv
            g. On crash: log error, skip to next experiment
Stage 4 — Print final leaderboard table
```

Crashes don't abort the sweep — every experiment is wrapped in
`try/except`, the error is logged to `results.csv`, and the loop
continues to the next one. This is critical for unattended runs.

### 3. Results log

`experiments/results.csv` columns:

| Column | Description |
|---|---|
| `id` | Experiment id (matches model filename) |
| `timestamp` | When the run finished |
| `negative_prob`, `bg_noise_prob`, `epochs`, ... | Hyperparameters used |
| `diagnose_pass` | `True` / `False` |
| `diagnose_mean_ratio` | Test 1 mean out/in ratio |
| `si_sdri` | Mean SI-SDRi across all test classes (dB) |
| `f1` | Mean detection F1 |
| `status` | `ok` / `crashed: <reason>` |
| `notes` | Free-text (empty by default) |

The notebook's last cell prints this sorted by `si_sdri` descending so
you can see the leaderboard at a glance.

### 4. Webapp model selection

A new dropdown in `webapp.py` lists every `.h5` in
`saved_models/separation_models/` that has a matching `_classes.json`.
Selecting one triggers a model reload via Gradio state. Detection and
removal then use the selected model.

This means: you don't edit `webapp.py` between experiments. Drive →
symlink → dropdown → pick → test.

---

## How to run a sweep

### One-time setup

1. Get Colab Pro+ (background execution requires it).
2. Create `experiments/` folder on Drive at the project root.
3. Drop an `experiments.json` in it (see above for format).

### Each sweep

1. Open `notebooks/colab_automated_training.ipynb` on Colab Pro+.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.
4. Once the first cell mounts Drive and the second one starts training,
   you can close the tab. Background execution keeps it running.
5. Come back when the notebook finishes (Drive shows the new `.h5`
   files; an email notification fires when the notebook ends).

### Checking progress mid-run

* Drive auto-syncs `results.csv` after every experiment. Open it from a
  phone or another machine to see which experiments have completed.
* The notebook's logs are visible in the Colab "Recent" tab even with
  the browser closed.

### Picking which model to keep

1. Open `colab_webapp.ipynb` on Colab.
2. Wait for the Gradio share URL.
3. Use the **Model** dropdown at the top to switch between any of the
   trained variants.
4. Pick the same audio file for each model so the comparison is fair.
5. Update `docs/model_training_log.md` with the winner and a one-line
   justification.

---

## What's automatable vs. what's not

| Concern | Automated? | Notes |
|---|---|---|
| Training | Yes | Trainer accepts all hyperparameters via init |
| Diagnose health check | Yes | Returns PASS/FAIL, logged in CSV |
| SI-SDRi evaluation | Yes | `evaluate_conditioned_separator.py` returns numbers |
| Detection F1 evaluation | Yes | `evaluate_detection.py` returns numbers |
| Crash recovery | Yes | Each experiment in `try/except`, log + skip |
| **Listening tests** | **No** | Perceptual quality (pulsing, removal aggressiveness) is not in any metric. v2.2's pulsing fix and v2.2's "removal too gentle" issue were both invisible to SI-SDRi and F1. |
| **Hyperparameter strategy** | **Partial** | The notebook runs whatever `experiments.json` says. You still write the JSON by hand. (Could later swap in Optuna for adaptive search.) |
| **Architectural changes** | **No** | Changing FiLM placement or loss function is a code edit, not a config change. Keep these as separate branches/PRs. |

---

## Recommended workflow

1. **Plan a sweep.** Based on the last model's listening test, decide
   the axis you want to explore (e.g. "removal too gentle → try
   negative_prob ∈ {0.10, 0.15, 0.20}").
2. **Write `experiments.json`.** 3-6 variants is the sweet spot for a
   single sweep (12-24 hours of T4 training).
3. **Launch on Colab Pro+** and close the tab.
4. **Come back, dinleyerek seç.** Use the webapp dropdown to compare
   the top 2-3 metric performers by ear.
5. **Log the winner** in `model_training_log.md` with the listening
   notes. The losers stay on Drive — they cost nothing to keep.
6. **Repeat** with the next axis.

---

## Files to be created (next implementation step)

* `experiments/experiments.json` — sample 3-variant config
* `notebooks/colab_automated_training.ipynb` — orchestration notebook
* Webapp dropdown — patch `src/application/webapp.py` to enumerate
  available models on launch and add a `gr.Dropdown` bound to a model-
  reload callback

The training entry point already accepts every hyperparameter through
`ConditionedSeparatorTrainer.__init__`; only `negative_prob` and
`bg_noise_prob` need to be passed through to the mixer (one-line wiring
change). No deep refactor required.
