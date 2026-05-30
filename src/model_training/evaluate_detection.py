"""
Detection-quality evaluation for the webapp.detect_sounds logic.

Builds synthetic mixtures whose present-class lists are known, runs the
detection scoring exactly as the webapp does, and measures how often the
surfaced classes match the ground truth. Reports per-class precision /
recall / F1 plus a confusion summary.

Run from project root:
    python src/model_training/evaluate_detection.py
"""
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "data_preparation"))
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, HOP_LENGTH, N_FFT, TIME_FRAMES,
)
from separation_mixer import SeparationMixer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Detection thresholds — kept in sync with webapp.detect_sounds.
ABSOLUTE_FLOOR = 0.05
RELATIVE_CAP = 0.65


def _load_model_classes(model_path: Path) -> list:
    """Class list the model was trained with, from the sibling ``_classes.json``.

    Defines the query-vector size and class -> index mapping. The model may
    know more classes (e.g. FSD50K) than the locally-mounted datasets expose;
    querying off the local mixer instead crashes with a shape mismatch
    (``expected (None, 235), found (1, 56)``).
    """
    classes_path = model_path.with_name(model_path.stem + "_classes.json")
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Model class list not found: {classes_path}. It is saved next to "
            f"the .h5 at training time and is required to build the query vector."
        )
    return json.load(classes_path.open())


def _spectrogram(waveform):
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)[:FREQ_BINS]
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return mag


def detect_classes(model, class_names, mixture):
    """Webapp-equivalent detection: peak-normalise, score every class, threshold.

    Every class query for one mixture is run in a single batched predict (the
    one-hot rows of an identity matrix) rather than one predict per class, so
    scoring a full 200+-class vocabulary stays fast.
    """
    peak = np.max(np.abs(mixture))
    if peak > 1e-6:
        mixture = mixture / peak
    mag = _spectrogram(mixture)
    lin = mag[..., None]
    log = np.log1p(lin)
    mix_energy = float(np.mean(lin ** 2)) + 1e-8

    n = len(class_names)
    log_b = np.repeat(log[None], n, axis=0)
    lin_b = np.repeat(lin[None], n, axis=0)
    query_b = np.eye(n, dtype=np.float32)  # row i = one-hot query for class i
    est = model.predict([log_b, query_b, lin_b], batch_size=64, verbose=0)

    scores = {}
    for i, name in enumerate(class_names):
        e = est[i]
        energy_ratio = float(np.mean(e ** 2) / mix_energy)
        specificity = float(np.std(e) / (np.mean(e) + 1e-8))
        scores[name] = energy_ratio * (1.0 + specificity ** 2)

    ranked = sorted(scores, key=scores.get, reverse=True)
    cutoff = max(ABSOLUTE_FLOOR, RELATIVE_CAP * scores[ranked[0]])
    detected = [n for n in ranked if scores[n] >= cutoff][:10]
    return set(detected), scores


def evaluate(model_path: Path, data_root: Path, n_test: int = 200,
             seed: int = 7777) -> dict:
    """Run the detection sweep and return a metric dict alongside the printout."""
    print(f"\nDetection Evaluation — {model_path.name}")
    print(f"Test mixtures: {n_test}\n")

    model = tf.keras.models.load_model(model_path, compile=False)
    model_classes = _load_model_classes(model_path)
    # Always have at least one class present so detection is meaningful.
    mixer = SeparationMixer(data_root, negative_prob=0.0, seed=seed)
    # Mixtures are built only from classes we have local audio for; the model
    # is queried over its FULL training vocabulary, so classes with no local
    # audio (e.g. FSD50K) still count as false positives when they fire — the
    # "bass_guitar shows up everywhere" failure mode shows up here.
    model_set = set(model_classes)
    available = [c for c in mixer.class_names if c in model_set]
    if not available:
        raise RuntimeError(
            "No local classes overlap the model vocabulary — nothing to test.")
    print(f"Model vocabulary: {len(model_classes)} classes; "
          f"{len(available)} have local audio for mixtures.\n")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    n_present = defaultdict(int)

    for i in range(n_test):
        # Build a mixture and remember which classes went into it.
        k = mixer._rng.randint(1, min(mixer.max_mix, len(available)))
        present = set(mixer._rng.sample(available, k))

        mixture = np.zeros(mixer.window_length, dtype=np.float32)
        for cls in present:
            clip = mixer._rng.choice(mixer._waveform_cache[cls])
            window = mixer._random_window(clip) * mixer._rng.uniform(*mixer.amp_range)
            mixture += window
        mixture = mixer._maybe_add_background_noise(mixture)

        detected, _ = detect_classes(model, model_classes, mixture)

        for cls in present:
            n_present[cls] += 1
            if cls in detected:
                tp[cls] += 1
            else:
                fn[cls] += 1
        for cls in detected:
            if cls not in present:
                fp[cls] += 1

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_test} mixtures processed")

    print("\n  {:<22}{:>6}{:>8}{:>8}{:>8}".format(
        "Class", "N", "Prec", "Rec", "F1"))
    print("  " + "-" * 52)
    f1s = []
    for cls in sorted(available):
        if n_present[cls] == 0:
            continue
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        print(f"  {cls:<22}{n_present[cls]:>6}{prec:>8.2f}{rec:>8.2f}{f1:>8.2f}")
    print("  " + "-" * 52)
    print(f"  {'MEAN F1':<22}{'':<6}{'':<8}{'':<8}{np.mean(f1s):>8.2f}")
    print(f"\n  Total true positives:  {sum(tp.values())}")
    print(f"  Total false positives: {sum(fp.values())}")
    print(f"  Total false negatives: {sum(fn.values())}")

    # Worst false-positive offenders across the WHOLE vocabulary, including
    # classes with no local audio (these can only ever be false positives).
    # Surfaces the "one class fires on everything" failure mode directly.
    top_fp = sorted(fp.items(), key=lambda kv: kv[1], reverse=True)[:10]
    if top_fp:
        available_set = set(available)
        print("\n  Top false-positive classes (fired but not present):")
        for cls, count in top_fp:
            tag = "" if cls in available_set else "  [no local audio]"
            print(f"    {cls:<22}{count:>5} FPs{tag}")
    print("=" * 60 + "\n")
    return {
        "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "tp_total": int(sum(tp.values())),
        "fp_total": int(sum(fp.values())),
        "fn_total": int(sum(fn.values())),
        "n_test": n_test,
    }


if __name__ == "__main__":
    evaluate(
        model_path=BASE_DIR / "saved_models" / "separation_models"
                            / "separator_unet_film_multi_v2.4.h5",
        data_root=BASE_DIR / "data" / "raw",
    )
