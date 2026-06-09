"""
Drive-mirrored cache for the decoded dataset — fast session boot.

``dataset_sources.load_all_datasets`` already caches its merged dictionary as
a single pickle so that the slow "decode 50 000 tiny WAVs" step only happens
once. That cache, however, lives on the local SSD (deliberately — reading
many small files off mounted Drive is what we are escaping in the first
place), and Colab wipes the SSD at runtime end. So every session decodes
from scratch: roughly **two to four hours** when FSD50K is on Drive.

This script round-trips that cache through Drive. The cache file is one
large blob (a few GB), which Drive serves at sustained throughput — copying
it back to SSD takes 30–90 seconds. Once the file is on SSD,
``load_all_datasets`` finds it and the dataset "loads" in seconds.

Modes
-----
Default (no flag)::

    python scripts/prep_data_cache.py

    If a matching cache exists in ``$SNC_DRIVE_CACHE_DIR``, copy it to
    ``$SNC_DATA_CACHE_DIR`` (defaults to ``/content``) and stop. Otherwise
    decode every dataset under ``data/raw/``, write the cache to local SSD,
    then mirror it back to Drive for next time.

``--force-rebuild``::

    Re-decode even when a Drive copy already exists; re-mirror the result.
    Use after CLASS_ALIASES changes or a dataset addition.

``--info``::

    List what's on Drive and on SSD, no I/O. Handy when a notebook cell
    seems suspicious about the cache state.

Environment
-----------
``SNC_DRIVE_CACHE_DIR``
    Directory on mounted Drive where the cache is mirrored, e.g.
    ``/content/drive/MyDrive/snc/cache``. Required for both auto-fetch and
    the auto-mirror after a build.

``SNC_DATA_CACHE_DIR``
    Local SSD directory the underlying ``dataset_sources`` cache lives in
    (default: ``/content``). Subsequent ``!python ...`` cells inherit this
    via ``os.environ``, so set it once in the notebook and forget about it.

Usage in a notebook (one cell)::

    os.environ['SNC_DATA_CACHE_DIR']  = '/content'
    os.environ['SNC_DRIVE_CACHE_DIR'] = f'{DRIVE_ROOT}/cache'
    !python scripts/prep_data_cache.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Repo root: this file lives at scripts/prep_data_cache.py
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "data" / "raw"

# Must match the format defined in dataset_sources._cache_file. We do not
# import that helper because building the cache imports tensorflow/librosa
# transitively, and ``--info`` should stay zero-dep so it runs even when the
# Python env is half-set-up.
DEFAULT_CACHE_VERSION = 1
DEFAULT_TARGET_SR = 16000
DEFAULT_MIN_CLIPS = 40


def _dataset_tag(data_root: Path) -> str:
    """Which datasets exist under ``data_root`` (e.g. ``esc-us8k``).

    Mirror of ``dataset_sources._dataset_tag`` (kept inline so ``--info`` stays
    zero-dep). The cache contents depend on which datasets are present, so the
    tag is part of the filename — an ESC-50+UrbanSound8K cache never gets
    confused with one that also has FSD50K.
    """
    tags = []
    if (data_root / "archive" / "esc50.csv").exists():
        tags.append("esc")
    if (data_root / "urbansound8k" / "metadata" / "UrbanSound8K.csv").exists():
        tags.append("us8k")
    if (data_root / "fsd50k" / "FSD50K.ground_truth" / "dev.csv").exists():
        tags.append("fsd")
    return "-".join(tags) if tags else "none"


def _cache_name(version: int = DEFAULT_CACHE_VERSION,
                target_sr: int = DEFAULT_TARGET_SR,
                min_clips: int = DEFAULT_MIN_CLIPS,
                tag: str = "esc-us8k-fsd") -> str:
    return f"_decoded_cache_v{version}_sr{target_sr}_min{min_clips}_{tag}.pkl"


def _human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _copy_with_progress(src: Path, dst: Path) -> None:
    """``shutil.copy2`` with a one-line periodic progress print."""
    size = src.stat().st_size
    chunk = 8 * 1024 * 1024  # 8 MB
    copied = 0
    started = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_print = 0.0
    with src.open("rb") as r, tmp.open("wb") as w:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            w.write(buf)
            copied += len(buf)
            now = time.time()
            if now - last_print > 1.0:
                pct = 100.0 * copied / size if size else 100.0
                rate = copied / max(now - started, 1e-6) / (1024 * 1024)
                print(f"  [prep] {pct:5.1f}%  {_human(copied)} / "
                      f"{_human(size)}  ({rate:6.1f} MB/s)", flush=True)
                last_print = now
    os.replace(tmp, dst)
    shutil.copystat(src, dst)
    elapsed = time.time() - started
    rate = size / max(elapsed, 1e-6) / (1024 * 1024)
    print(f"  [prep] done in {elapsed:.1f}s  ({rate:.1f} MB/s avg)", flush=True)


def _resolve_dirs() -> tuple[Path, Path]:
    """Validate the two cache dirs from the environment."""
    drive_raw = os.environ.get("SNC_DRIVE_CACHE_DIR", "").strip()
    if not drive_raw:
        raise SystemExit(
            "SNC_DRIVE_CACHE_DIR is not set. Set it to a Drive directory, "
            "e.g. /content/drive/MyDrive/snc/cache, then re-run.")
    local_raw = os.environ.get("SNC_DATA_CACHE_DIR", "/content").strip()
    return Path(drive_raw), Path(local_raw)


def _build(local_cache: Path, target_sr: int, min_clips: int) -> Path:
    """Decode every dataset, returning the path of the produced cache file."""
    # Match what training/eval expect.
    os.environ["SNC_DATA_CACHE_DIR"] = str(local_cache)
    local_cache.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(BASE_DIR / "src" / "data_preparation"))
    from dataset_sources import load_all_datasets  # noqa: E402

    print(f"[prep] decoding datasets from {DATA_ROOT} "
          f"(target_sr={target_sr}, min_clips={min_clips})", flush=True)
    print(f"[prep] this is slow on first run — expect 30-60 min for "
          f"ESC-50 + UrbanSound8K + FSD50K", flush=True)
    started = time.time()
    load_all_datasets(DATA_ROOT, target_sr=target_sr,
                      min_clips_per_class=min_clips)
    elapsed = time.time() - started
    print(f"[prep] decode finished in {elapsed/60:.1f} min", flush=True)

    cache_file = local_cache / _cache_name(DEFAULT_CACHE_VERSION,
                                           target_sr, min_clips,
                                           _dataset_tag(DATA_ROOT))
    if not cache_file.exists():
        raise RuntimeError(
            f"Decode completed but expected cache file {cache_file} is "
            f"missing — has _CACHE_VERSION or the filename format changed?")
    return cache_file


def _info(drive_cache: Path, local_cache: Path) -> None:
    """Print which cache files exist on Drive and SSD."""
    print(f"[prep] Drive cache dir: {drive_cache}")
    if drive_cache.exists():
        for p in sorted(drive_cache.glob("_decoded_cache_*.pkl")):
            print(f"  drive  {_human(p.stat().st_size):>10}  {p.name}")
    else:
        print("  drive  (directory missing)")
    print(f"[prep] Local cache dir: {local_cache}")
    if local_cache.exists():
        for p in sorted(local_cache.glob("_decoded_cache_*.pkl")):
            print(f"  local  {_human(p.stat().st_size):>10}  {p.name}")
    else:
        print("  local  (directory missing)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror the decoded-dataset cache between Drive and SSD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Decode from raw even if Drive copy exists.")
    parser.add_argument("--info", action="store_true",
                        help="List cache files on Drive/SSD and exit.")
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR,
                        help="Sample rate the cache was built for (16000).")
    parser.add_argument("--min-clips", type=int, default=DEFAULT_MIN_CLIPS,
                        help="min_clips_per_class the cache was built for (40).")
    args = parser.parse_args()

    drive_cache, local_cache = _resolve_dirs()
    if args.info:
        _info(drive_cache, local_cache)
        return

    cache_name = _cache_name(DEFAULT_CACHE_VERSION, args.target_sr,
                             args.min_clips, _dataset_tag(DATA_ROOT))
    drive_file = drive_cache / cache_name
    local_file = local_cache / cache_name

    # Fast path: Drive already has a copy, just bring it to SSD.
    if drive_file.exists() and not args.force_rebuild:
        if (local_file.exists()
                and local_file.stat().st_size == drive_file.stat().st_size):
            print(f"[prep] cache already on SSD ({local_file}) — nothing to do.")
            return
        print(f"[prep] copying cache from Drive → SSD "
              f"({_human(drive_file.stat().st_size)})", flush=True)
        _copy_with_progress(drive_file, local_file)
        print(f"[prep] cache ready: {local_file}")
        print(f"[prep] later cells inherit SNC_DATA_CACHE_DIR={local_cache} "
              f"automatically.")
        return

    # Build path: decode from raw, then mirror to Drive.
    built = _build(local_cache, args.target_sr, args.min_clips)
    print(f"[prep] mirroring to Drive: {drive_file}", flush=True)
    _copy_with_progress(built, drive_file)
    print(f"[prep] cache mirrored — future sessions will skip the decode.")


if __name__ == "__main__":
    main()
