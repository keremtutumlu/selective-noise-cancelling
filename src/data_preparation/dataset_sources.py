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
    * FSD50K         — ~200 AudioSet-labelled classes; duplicates merge
                       into ESC-50 via ``CLASS_ALIASES``. Under-supported
                       leaf classes are pruned by ``min_clips_per_class``
                       in ``load_all_datasets``.

To add another dataset, write a ``load_<name>`` function that returns the
same dict shape and call it from ``load_all_datasets``.
"""
import logging
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cross-dataset duplicate classes are merged onto a single canonical name
# so their clips pool together instead of becoming separate labels.
CLASS_ALIASES = {
    # UrbanSound8K → ESC-50
    "dog_bark": "dog",
    "engine_idling": "engine",
    # FSD50K (AudioSet) → ESC-50 canonical names. Keys are the
    # *post-normalisation* labels produced by ``_to_snake`` (commas and
    # spaces collapse to a single underscore), not the raw FSD50K strings.
    "bark": "dog",
    "meow": "cat",
    "laughter": "laughing",
    "cough": "coughing",
    "sneeze": "sneezing",
    "baby_cry_infant_cry": "crying_baby",
    "alarm": "clock_alarm",
    "rain_on_surface": "rain",
    "waves_surf": "sea_waves",
}


def _canonical(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


def _to_snake(name: str) -> str:
    """'Electric guitar' → 'electric_guitar'; 'Waves, surf' → 'waves_surf'."""
    return re.sub(r"[\s\-/,]+", "_", name.strip().lower()).strip("_")


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


def load_fsd50k(root_dir: Path, target_sr: int = 16000,
                max_duration: float = 4.0) -> Dict[str, List[np.ndarray]]:
    """Load FSD50K from ``root_dir``.

    Expects the standard FSD50K layout:
        ``root_dir/FSD50K.ground_truth/dev.csv``
        ``root_dir/FSD50K.ground_truth/eval.csv``
        ``root_dir/FSD50K.dev_audio/<fname>.wav``
        ``root_dir/FSD50K.eval_audio/<fname>.wav``

    FSD50K labels are AudioSet-hierarchical, so almost every clip has
    multiple comma-separated labels (e.g. ``"Bark,Dog,Animal"``). The
    first label is taken as the canonical class: it's the leaf/most
    specific entry per FSD50K convention. Clips are trimmed to
    ``max_duration`` seconds on load to keep total RAM usage bounded
    (the mixer only ever takes random 1-second windows anyway).
    """
    cache: Dict[str, List[np.ndarray]] = {}
    total = 0
    missing_audio = 0
    for csv_name, audio_dir_name in [
        ("FSD50K.ground_truth/dev.csv", "FSD50K.dev_audio"),
        ("FSD50K.ground_truth/eval.csv", "FSD50K.eval_audio"),
    ]:
        csv_path = root_dir / csv_name
        audio_dir = root_dir / audio_dir_name
        if not csv_path.exists():
            continue
        if not audio_dir.exists():
            logging.warning(f"FSD50K: {audio_dir} not found, skipping {csv_name}.")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            raw_labels = str(row["labels"]).split(",")
            cls = _canonical(_to_snake(raw_labels[0]))
            path = audio_dir / f"{row['fname']}.wav"
            if not path.exists():
                missing_audio += 1
                continue
            wav, _ = librosa.load(path, sr=target_sr, mono=True,
                                  duration=max_duration)
            if wav.size == 0:
                continue
            cache.setdefault(cls, []).append(wav.astype(np.float32))
            total += 1
    if missing_audio:
        logging.warning(f"FSD50K: {missing_audio} CSV rows had no matching audio file.")
    logging.info(f"FSD50K: {total} clips, {len(cache)} classes.")
    return cache


# Bumping this invalidates every on-disk cache (use when the decode/merge
# logic changes in a way that alters the cached waveforms).
_CACHE_VERSION = 1


def _dataset_tag(data_root: Path) -> str:
    """Short id for which datasets exist under ``data_root`` (e.g. ``esc-us8k``).

    The decoded cache's *contents* depend on which datasets are present, but
    the per-config filename only captured sample rate and the clip floor. So a
    56-class ESC-50+UrbanSound8K cache and a 227-class run that also has FSD50K
    shared one filename — fetching either from Drive could load the wrong one.
    Encoding the present datasets keeps the two caches distinct.
    """
    tags = []
    if (data_root / "archive" / "esc50.csv").exists():
        tags.append("esc")
    if (data_root / "urbansound8k" / "metadata" / "UrbanSound8K.csv").exists():
        tags.append("us8k")
    if (data_root / "fsd50k" / "FSD50K.ground_truth" / "dev.csv").exists():
        tags.append("fsd")
    return "-".join(tags) if tags else "none"


def _cache_file(data_root: Path, target_sr: int, min_clips_per_class: int) -> Path:
    """Per-config cache path. Any parameter that changes the merged result is
    encoded in the filename so a different config never reads a stale cache.

    Defaults to ``data_root`` but honours ``SNC_DATA_CACHE_DIR`` — on Colab the
    data root is a Drive symlink, and writing/reading the cache there is exactly
    the slow tiny-file I/O we are trying to avoid. Point this at local SSD
    (e.g. ``/content``) so the cache itself is fast.
    """
    cache_dir = Path(os.environ.get("SNC_DATA_CACHE_DIR", str(data_root)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (cache_dir / f"_decoded_cache_v{_CACHE_VERSION}"
            f"_sr{target_sr}_min{min_clips_per_class}"
            f"_{_dataset_tag(data_root)}.pkl")


def _maybe_fetch_cache_from_drive(cache_path: Path) -> None:
    """Pull a Drive-mirrored copy of ``cache_path`` to local SSD if present.

    A fresh Colab container has an empty local SSD, so the decoded-dataset
    cache is gone even though a copy sits on Drive. Fetching that one large
    blob (a few GB at sustained throughput) takes under a minute, versus the
    multi-hour decode of ~50k tiny WAVs it replaces. This makes training and
    evaluation boot fast on their own — running ``scripts/prep_data_cache.py``
    first becomes optional rather than required.

    No-op unless ``SNC_DRIVE_CACHE_DIR`` is set and holds a matching file.
    """
    drive_dir = os.environ.get("SNC_DRIVE_CACHE_DIR", "").strip()
    if not drive_dir:
        return
    drive_file = Path(drive_dir) / cache_path.name
    if not drive_file.exists():
        return
    try:
        logging.info(f"Fetching decoded-dataset cache from Drive "
                     f"({drive_file}, {drive_file.stat().st_size / 1e9:.2f} GB)...")
        tmp = cache_path.with_suffix(cache_path.suffix + ".part")
        shutil.copyfile(drive_file, tmp)
        os.replace(tmp, cache_path)
        logging.info(f"Cache fetched to {cache_path} — skipped the decode.")
    except OSError as exc:
        logging.warning(f"Could not fetch cache from Drive ({exc}); "
                        f"falling back to decode.")


def _maybe_mirror_cache_to_drive(cache_path: Path) -> None:
    """Copy a freshly built local cache back to Drive for future sessions.

    Pairs with :func:`_maybe_fetch_cache_from_drive`: the first session to
    decode mirrors the result to Drive, every later session fetches it. No-op
    unless ``SNC_DRIVE_CACHE_DIR`` is set; the copy is best-effort so a Drive
    write failure never aborts a successful decode.
    """
    drive_dir = os.environ.get("SNC_DRIVE_CACHE_DIR", "").strip()
    if not drive_dir:
        return
    drive_file = Path(drive_dir) / cache_path.name
    if drive_file.exists() and drive_file.stat().st_size == cache_path.stat().st_size:
        return
    try:
        Path(drive_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Mirroring decoded-dataset cache to Drive ({drive_file})...")
        tmp = drive_file.with_suffix(drive_file.suffix + ".part")
        shutil.copyfile(cache_path, tmp)
        os.replace(tmp, drive_file)
        logging.info("Cache mirrored — future sessions skip the decode.")
    except OSError as exc:
        logging.warning(f"Could not mirror cache to Drive ({exc}).")


def load_all_datasets(data_root: Path, target_sr: int = 16000,
                      min_clips_per_class: int = 40,
                      cache: bool = True,
                      cache_path: Optional[Path] = None,
                      keep_classes: Optional[List[str]] = None,
                      ) -> Dict[str, List[np.ndarray]]:
    """
    Merge every dataset found under ``data_root`` into one class->clips dict.

    Looks for:
        ``data_root/archive/esc50.csv``                         -> ESC-50
        ``data_root/urbansound8k/metadata/UrbanSound8K.csv``    -> UrbanSound8K
        ``data_root/fsd50k/FSD50K.ground_truth/dev.csv``        -> FSD50K

    ``min_clips_per_class`` drops any class left with fewer than that many
    clips **after merging**. FSD50K's AudioSet leaf labels have a long tail
    of classes backed by only a handful of clips; a class learned from so
    few examples produces a diffuse, non-discriminative mask that fires on
    unrelated audio (the v2.3 false-positive problem — e.g. ``purr``,
    ``boom``, ``bass_guitar``). Pruning happens post-merge so cross-dataset
    aliases pool first (FSD50K ``bark`` clips count toward ESC-50 ``dog``)
    and the well-supported ESC-50/UrbanSound8K classes — all far above the
    floor — are never affected.

    ``cache`` (default True): decoding ~10k+ small WAVs with ``librosa.load``
    is the slowest step by far, and on a Drive-backed data root reading
    thousands of tiny files cold can take *over an hour* — repeated once per
    evaluation script. With caching on, the fully decoded+merged result is
    pickled once to ``cache_path`` (default: a per-config file under
    ``data_root``); every later call reads that single file in seconds.
    Set ``SNC_DISABLE_DATA_CACHE=1`` or ``cache=False`` to force a fresh
    decode. Delete the ``_decoded_cache_*.pkl`` file to rebuild it.

    ``keep_classes`` (default None = no restriction): when given, the returned
    dict is restricted to just those class names. This is the v3.0
    curated-vocabulary path — training on a small, well-detected subset rather
    than the full ~56-class set. The filter is applied to the *returned* dict
    only; the on-disk cache always holds the full decoded vocabulary, so a
    curated-subset run and a full-vocabulary run share one cache file without
    one ever poisoning the other. Requested names absent from the data are
    logged and skipped; if none match a ``ValueError`` is raised.
    """
    if os.environ.get("SNC_DISABLE_DATA_CACHE") == "1":
        cache = False
    cache_path = cache_path or _cache_file(data_root, target_sr, min_clips_per_class)

    def _select(full: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """Restrict to ``keep_classes`` (if set). Applied to the return value
        only, never to what is cached on disk."""
        if not keep_classes:
            return full
        keep = set(keep_classes)
        selected = {c: full[c] for c in full if c in keep}
        missing = sorted(keep - set(full))
        msg = (f"keep_classes: restricted to {len(selected)}/{len(full)} "
               f"classes ({sum(len(v) for v in selected.values())} clips).")
        if missing:
            msg += f" {len(missing)} requested class(es) not in data: {missing}"
        logging.info(msg)
        if not selected:
            raise ValueError(
                f"keep_classes matched none of the {len(full)} available "
                f"classes. Requested: {sorted(keep)}. "
                f"Available: {sorted(full)}.")
        return selected

    # Cold local SSD (fresh Colab container) but a Drive copy exists: pull it
    # rather than re-decoding ~50k WAVs. Makes the prep step optional.
    if cache and not cache_path.exists():
        _maybe_fetch_cache_from_drive(cache_path)

    if cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                merged = pickle.load(f)
            logging.info(
                f"Loaded decoded dataset from cache: {cache_path.name} "
                f"({len(merged)} classes, "
                f"{sum(len(v) for v in merged.values())} clips).")
            return _select(merged)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            logging.warning(f"Cache {cache_path.name} unreadable ({exc}); "
                            f"re-decoding from source audio.")

    merged: Dict[str, List[np.ndarray]] = {}

    esc50_dir = data_root / "archive"
    if (esc50_dir / "esc50.csv").exists():
        for cls, clips in load_esc50(esc50_dir, target_sr).items():
            merged.setdefault(cls, []).extend(clips)

    us8k_dir = data_root / "urbansound8k"
    if (us8k_dir / "metadata" / "UrbanSound8K.csv").exists():
        for cls, clips in load_urbansound8k(us8k_dir, target_sr).items():
            merged.setdefault(cls, []).extend(clips)

    fsd50k_dir = data_root / "fsd50k"
    if (fsd50k_dir / "FSD50K.ground_truth" / "dev.csv").exists():
        for cls, clips in load_fsd50k(fsd50k_dir, target_sr).items():
            merged.setdefault(cls, []).extend(clips)

    if not merged:
        raise FileNotFoundError(
            f"No datasets found under {data_root}. Expected ESC-50 at 'archive/' "
            f"and/or UrbanSound8K at 'urbansound8k/'."
        )

    if min_clips_per_class > 1:
        dropped = sorted(c for c, v in merged.items()
                         if len(v) < min_clips_per_class)
        for cls in dropped:
            del merged[cls]
        if dropped:
            logging.info(f"Pruned {len(dropped)} class(es) under "
                         f"{min_clips_per_class} clips: {dropped}")
        if not merged:
            raise ValueError(
                f"min_clips_per_class={min_clips_per_class} pruned every class. "
                f"Lower the floor or add more data."
            )

    logging.info(f"Merged: {len(merged)} classes, "
                 f"{sum(len(v) for v in merged.values())} clips total.")

    if cache:
        # Write atomically (temp file + replace) so an interrupted write never
        # leaves a half-written cache that later loads as garbage.
        try:
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp.open("wb") as f:
                pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            logging.info(f"Cached decoded dataset to {cache_path.name} "
                         f"(delete to rebuild).")
            # Mirror the fresh cache to Drive so the next session fetches it.
            _maybe_mirror_cache_to_drive(cache_path)
        except OSError as exc:
            logging.warning(f"Could not write dataset cache: {exc}")

    return _select(merged)
