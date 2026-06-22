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
    os.environ["SNC_MODEL_VERSION"] = "v2.6"   # change this to switch versions

    # every later cell now targets v2.6 automatically
    !python src/model_training/train_conditioned_separator.py
    !python src/model_training/evaluate_conditioned_separator.py
    !python src/model_training/evaluate_detection.py

``os.environ`` set in the kernel is inherited by the ``!python`` subprocesses, so
this redirects the whole pipeline with no code edits. From a notebook cell that
imports the code directly the same applies:

    import model_config as cfg
    cfg.model_path()        # -> .../separator_unet_film_multi_v2.6.h5

Every helper reads the version lazily, so setting the env var after import still
takes effect. A ``--version`` CLI flag on each script overrides the env var for
one-off runs (e.g. comparing v2.3 against v2.4 without re-exporting).

Drive audit log
---------------
Set ``SNC_DRIVE_LOG_DIR`` to a directory on mounted Drive before running any
script.  Every script calls ``log_drive_run()`` at the end to append a
timestamped JSONL entry with version, key metrics, and the last 4 kB of stdout.
If the env var is unset or the path does not exist the call is a silent no-op,
so the local workflow is unaffected.

    import os
    os.environ["SNC_DRIVE_LOG_DIR"] = "/content/drive/MyDrive/snc/run_logs"
"""
import datetime
import json
import os
import socket
from pathlib import Path
from typing import List, Optional

# Repo root: this file lives at src/model_training/model_config.py
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "saved_models" / "separation_models"

# Default checkpoint version, used when SNC_MODEL_VERSION is unset and no
# --version flag is passed. Bump this only to move the committed default;
# day-to-day version switching should go through the env var or the flag.
DEFAULT_MODEL_VERSION = "v3.0"

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


def detect_allowlist_path(version: Optional[str] = None) -> Path:
    """Path to the optional detection allow-list for the active (or given) version.

    Sits next to the checkpoint as ``..._detect_allowlist.json`` and, when
    present, holds a JSON list of class names that detection is permitted to
    surface. The model still *knows* its full vocabulary (so any class can be
    queried for removal), but detection ranking/cutoff is computed over the
    allow-list only — keeping classes the local datasets cannot validate
    (e.g. FSD50K-only ``ringtone``, ``telephone``) out of the candidate pool
    so they can never become false positives.
    """
    return MODELS_DIR / f"{model_stem(version)}_detect_allowlist.json"


def load_detect_allowlist(version: Optional[str] = None) -> Optional[List[str]]:
    """Detection allow-list for a version, or ``None`` if no file exists.

    ``None`` means "no restriction" — detection falls back to the model's full
    class vocabulary, the pre-allow-list behaviour. An empty or malformed file
    is treated the same as missing, so a bad file never silently hides every
    class.
    """
    path = detect_allowlist_path(version)
    if not path.exists():
        return None
    try:
        names = json.load(path.open())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(names, list) or not names:
        return None
    return [str(n) for n in names]


# ---------------------------------------------------------------------------
# Drive audit log
# ---------------------------------------------------------------------------
DRIVE_LOG_ENV = "SNC_DRIVE_LOG_DIR"


def log_drive_run(
    script: str,
    version: str,
    metrics: dict,
    output: str = "",
    notes: str = "",
) -> None:
    """Append a timestamped run-log entry to ``run_log.json`` on Drive.

    The file is a single pretty-printed JSON array of entries — readable at
    a glance, not the line-delimited JSONL format that was hard to scan.
    Writes are atomic (write to .tmp, then rename) so a crash mid-write
    never corrupts the log.

    Reads ``SNC_DRIVE_LOG_DIR`` from the environment. If the variable is
    unset or the directory does not exist the call is a silent no-op, so
    nothing breaks when running locally without Drive mounted.

    Colab one-liner (put near the top of the notebook):
        os.environ["SNC_DRIVE_LOG_DIR"] = "/content/drive/MyDrive/snc/run_logs"
    """
    log_dir_str = os.environ.get(DRIVE_LOG_ENV, "").strip()
    if not log_dir_str:
        return
    log_dir = Path(log_dir_str)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "run_log.json"
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "script": script,
            "version": version,
            "metrics": metrics,
            "output": output[-4000:] if output else "",
            "notes": notes,
            "host": socket.gethostname(),
        }
        # Load existing entries; tolerate missing or corrupt files by starting
        # over instead of crashing — the audit log must never block a run.
        entries: list = []
        if log_path.exists():
            try:
                loaded = json.loads(log_path.read_text())
                if isinstance(loaded, list):
                    entries = loaded
            except (json.JSONDecodeError, OSError):
                pass

        # One-time migration: if the legacy run_log.jsonl exists, fold its
        # entries into the new array so the history is not lost.
        legacy = log_dir / "run_log.jsonl"
        if not entries and legacy.exists():
            try:
                with legacy.open() as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                pass

        entries.append(entry)
        # Atomic write: never leave a half-written log file behind.
        tmp_path = log_path.with_suffix(".json.tmp")
        with tmp_path.open("w") as f:
            json.dump(entries, f, indent=2)
        tmp_path.replace(log_path)
        print(f"[drive_log] Appended to {log_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[drive_log] WARNING: could not write to {log_dir}: {exc}")
