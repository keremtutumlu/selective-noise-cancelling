"""
Detection-quality evaluation for the webapp.detect_sounds logic.

Builds synthetic mixtures whose present-class lists are known, runs the
detection scoring exactly as the webapp does, and measures how often the
surfaced classes match the ground truth. Reports per-class precision /
recall / F1 plus a confusion summary.

Run from project root:
    python src/model_training/evaluate_detection.py
"""
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


def _query(class_names, name):
    q = np.zeros(len(class_names), np.float32)
    q[class_names.index(name)] = 1.0
    return q


def _spectrogram(waveform):
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft).astype(np.float32)[:FREQ_BINS]
    if mag.shape[1] < TIME_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, TIME_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :TIME_FRAMES]
    return mag


def detect_classes(model, class_names, mixture):
    """Webapp-equivalent detection: peak-normalise, score, threshold."""
    peak = np.max(np.abs(mixture))
    if peak > 1e-6:
        mixture = mixture / peak
    mag = _spectrogram(mixture)
    lin = mag[..., None]
    log = np.log1p(lin)
    mix_energy = float(np.mean(lin ** 2)) + 1e-8

    scores = {}
    for name in class_names:
        q = _query(class_names, name)
        est = model.predict([log[None], q[None], lin[None]], verbose=0)
        energy_ratio = float(np.mean(est ** 2) / mix_energy)
        specificity = float(np.std(est) / (np.mean(est) + 1e-8))
        scores[name] = energy_ratio * (1.0 + specificity ** 2)

    ranked = sorted(scores, key=scores.get, reverse=True)
    cutoff = max(ABSOLUTE_FLOOR, RELATIVE_CAP * scores[ranked[0]])
    detected = [n for n in ranked if scores[n] >= cutoff][:10]
    return set(detected), scores


def evaluate(model_path: Path, data_root: Path, n_test: int = 200,
             seed: int = 7777):
    print(f"\nDetection Evaluation — {model_path.name}")
    print(f"Test mixtures: {n_test}\n")

    model = tf.keras.models.load_model(model_path, compile=False)
    # Always have at least one class present so detection is meaningful.
    mixer = SeparationMixer(data_root, negative_prob=0.0, seed=seed)
    class_names = mixer.class_names

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    n_present = defaultdict(int)

    for i in range(n_test):
        # Build a mixture and remember which classes went into it.
        k = mixer._rng.randint(1, mixer.max_mix)
        present = set(mixer._rng.sample(class_names, k))

        mixture = np.zeros(mixer.window_length, dtype=np.float32)
        for cls in present:
            clip = mixer._rng.choice(mixer._waveform_cache[cls])
            window = mixer._random_window(clip) * mixer._rng.uniform(*mixer.amp_range)
            mixture += window
        mixture = mixer._maybe_add_background_noise(mixture)

        detected, _ = detect_classes(model, class_names, mixture)

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
    for cls in sorted(class_names):
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
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate(
        model_path=BASE_DIR / "saved_models" / "separation_models"
                            / "separator_unet_film_multi_v2.2.h5",
        data_root=BASE_DIR / "data" / "raw",
    )
