"""
Audio dataset loaders for the conditioned separator's on-the-fly mixer.

Each loader returns a ``{class_name: [waveform, ...]}`` dict of isolated
single-source clips resampled to a common sample rate. ``load_all_datasets``
merges every dataset it finds under a data root into one dict, so adding a
dataset only means writing a new loader and a detection check here — the
mixer, model, and training code adapt to the new class count automatically.

Supported datasets:
    * ESC-50         — 50 environmental classes (one zip from GitHub).
    * UrbanSound8K   — 10 urban classes; 4 of them merge into ESC-50
                       classes via ``CLASS_ALIASES``, 6 are new.

To add another dataset (e.g. FSD50K), write a ``load_<name>`` function
that returns the same dict shape and call it from ``load_all_datasets``.
"""
import logging
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cross-dataset duplicate classes are merged onto a single canonical name
# so their clips pool together instead of becoming separate labels.
CLASS_ALIASES = {
    "dog_bark": "dog",          # UrbanSound8K -> ESC-50
    "engine_idling": "engine",  # UrbanSound8K -> ESC-50
}


def _canonical(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


def load_esc50(archive_dir: Path, target_sr: int = 16000) -> Dict[str, List[np.ndarray]]:
    """Load ESC-50 from ``archive_dir`` (expects ``esc50.csv`` + ``audio/audio/``)."""
    df = pd.read_csv(archive_dir / "esc50.csv")
    audio_dir = archive_dir / "audio" / "audio"
    cache: Dict[str, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        path = audio_dir / row["filename"]
        if not path.exists():
            continue
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
        cache.setdefault(_canonical(row["category"]), []).append(wav.astype(np.float32))
    logging.info(f"ESC-50: {sum(len(v) for v in cache.values())} clips, {len(cache)} classes.")
    return cache


def load_urbansound8k(root_dir: Path, target_sr: int = 16000) -> Dict[str, List[np.ndarray]]:
    """Load UrbanSound8K from ``root_dir`` (expects ``metadata/`` + ``audio/foldN/``)."""
    df = pd.read_csv(root_dir / "metadata" / "UrbanSound8K.csv")
    audio_dir = root_dir / "audio"
    cache: Dict[str, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        path = audio_dir / f"fold{row['fold']}" / row["slice_file_name"]
        if not path.exists():
            continue
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
        if wav.size == 0:
            continue
        cache.setdefault(_canonical(row["class"]), []).append(wav.astype(np.float32))
    logging.info(f"UrbanSound8K: {sum(len(v) for v in cache.values())} clips, {len(cache)} classes.")
    return cache


def load_all_datasets(data_root: Path, target_sr: int = 16000) -> Dict[str, List[np.ndarray]]:
    """
    Merge every dataset found under ``data_root`` into one class->clips dict.

    Looks for:
        ``data_root/archive/esc50.csv``                -> ESC-50
        ``data_root/urbansound8k/metadata/UrbanSound8K.csv`` -> UrbanSound8K
    """
    merged: Dict[str, List[np.ndarray]] = {}

    esc50_dir = data_root / "archive"
    if (esc50_dir / "esc50.csv").exists():
        for cls, clips in load_esc50(esc50_dir, target_sr).items():
            merged.setdefault(cls, []).extend(clips)

    us8k_dir = data_root / "urbansound8k"
    if (us8k_dir / "metadata" / "UrbanSound8K.csv").exists():
        for cls, clips in load_urbansound8k(us8k_dir, target_sr).items():
            merged.setdefault(cls, []).extend(clips)

    if not merged:
        raise FileNotFoundError(
            f"No datasets found under {data_root}. Expected ESC-50 at 'archive/' "
            f"and/or UrbanSound8K at 'urbansound8k/'."
        )
    logging.info(f"Merged: {len(merged)} classes, "
                 f"{sum(len(v) for v in merged.values())} clips total.")
    return merged
