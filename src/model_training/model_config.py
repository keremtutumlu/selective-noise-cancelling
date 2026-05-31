"""
Central configuration for model versioning and shared paths.

One place decides which checkpoint the whole pipeline — training, SI-SDR
evaluation, detection-F1 evaluation, diagnosis, real-world testing — reads and
writes. A version bump no longer means editing several scripts and committing
the change.

Override the active version *without touching any source file* by setting the
``SNC_MODEL_VERSION`` environment variable. This is the intended Colab workflow:
set it once near the top of the notebook and every later cell follows.

    # one cell, near the top of the notebook
    import os
    os.environ["SNC_MODEL_VERSION"] = "v2.5"   # change this to switch versions

    # every later cell now targets v2.5 automatically
    !python src/model_training/train_conditioned_separator.py
    !python src/model_training/evaluate_conditioned_separator.py
    !python src/model_training/evaluate_detection.py

``os.environ`` set in the kernel is inherited by the ``!python`` subprocesses, so
this redirects the whole pipeline with no code edits. From a notebook cell that
imports the code directly the same applies:

    import model_config as cfg
    cfg.model_path()        # -> .../separator_unet_film_multi_v2.5.h5

Every helper reads the version lazily, so setting the env var after import still
takes effect. A ``--version`` CLI flag on each script overrides the env var for
one-off runs (e.g. comparing v2.3 against v2.4 without re-exporting).
"""
import os
from pathlib import Path
from typing import Optional

# Repo root: this file lives at src/model_training/model_config.py
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "saved_models" / "separation_models"

# Default checkpoint version, used when SNC_MODEL_VERSION is unset and no
# --version flag is passed. Bump this only to move the committed default;
# day-to-day version switching should go through the env var or the flag.
DEFAULT_MODEL_VERSION = "v2.4"

# Saved-model naming scheme from RULES.md:
#   <task>_<architecture>_<dataset-or-keyfeature>_v<major>.<minor>
MODEL_STEM_PREFIX = "separator_unet_film_multi"

# Environment variable that overrides the default version pipeline-wide.
ENV_VAR = "SNC_MODEL_VERSION"


def model_version() -> str:
    """Active model version: ``SNC_MODEL_VERSION`` env var, else the default."""
    return os.environ.get(ENV_VAR, DEFAULT_MODEL_VERSION).strip() or DEFAULT_MODEL_VERSION


def model_stem(version: Optional[str] = None) -> str:
    """File stem for a version, e.g. ``separator_unet_film_multi_v2.4``."""
    return f"{MODEL_STEM_PREFIX}_{version or model_version()}"


def model_path(version: Optional[str] = None) -> Path:
    """Path to the ``.h5`` checkpoint for the active (or given) version."""
    return MODELS_DIR / f"{model_stem(version)}.h5"


def classes_path(version: Optional[str] = None) -> Path:
    """Path to the sibling ``_classes.json`` for the active (or given) version."""
    return MODELS_DIR / f"{model_stem(version)}_classes.json"
