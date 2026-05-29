"""
Quick health check for the conditioned separator model.

Three tests catch the failure modes seen in v2.0:

1. **Output non-zero** — does the model produce meaningful magnitude for a
   known present class? A collapsed model outputs near-zero everywhere.
2. **FiLM discrimination** — does the same input produce different outputs
   for different class queries? A model that ignores the query produces
   identical output regardless of class.
3. **Correct-vs-wrong advantage** — across many classes, does the correct
   query consistently produce more output energy than wrong queries?

A healthy model passes all three; the broken v2.0 model fails on 1 and 3.

Run from project root:
    python src/model_training/diagnose_model.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "data_preparation"))
sys.path.insert(0, str(BASE_DIR / "src" / "model_training"))
from conditioned_separator import (  # noqa: E402
    FREQ_BINS, TIME_FRAMES,
)
from separation_mixer import SeparationMixer, waveform_to_magnitude  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_model_classes(model_path: Path) -> list:
    """Class list the model was trained with, from the sibling ``_classes.json``.

    This defines the query-vector size and the class -> index mapping, which
    is **not** the same as the classes the locally-mounted datasets expose:
    the model may know more classes (e.g. FSD50K) than are present at eval
    time. Building the query off the local mixer instead crashes with a
    shape mismatch (``expected (None, 235), found (1, 56)``).
    """
    classes_path = model_path.with_name(model_path.stem + "_classes.json")
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Model class list not found: {classes_path}. It is saved next to "
            f"the .h5 at training time and is required to build the query vector."
        )
    return json.load(classes_path.open())


def _query(class_names, name):
    q = np.zeros(len(class_names), np.float32)
    q[class_names.index(name)] = 1.0
    return q


def _clean_clip(mixer, class_name):
    """1-second clip of one class, peak-normalised to |peak|=1.0 (training-style)."""
    clips = mixer._waveform_cache[class_name]
    clip = clips[len(clips) // 2]
    if len(clip) >= mixer.window_length:
        start = (len(clip) - mixer.window_length) // 2
        clip = clip[start:start + mixer.window_length]
    else:
        clip = np.pad(clip, (0, mixer.window_length - len(clip)))
    peak = np.max(np.abs(clip))
    if peak > 1e-6:
        clip = clip / peak
    return clip.astype(np.float32)


def _model_input(waveform):
    mag = waveform_to_magnitude(waveform)
    lin = mag[..., None]
    log = np.log1p(lin)
    return lin, log


def diagnose(model_path: Path, data_root: Path, n_classes: int = 12,
             seed: int = 0) -> dict:
    """Run all three health tests and return a metric dict."""
    print(f"\nModel diagnostic — {model_path.name}")
    print(f"Data root    — {data_root}\n")

    model = tf.keras.models.load_model(model_path, compile=False)
    model_classes = _load_model_classes(model_path)
    mixer = SeparationMixer(data_root, seed=seed)
    # Query vocabulary = the classes the model was trained on (may exceed the
    # locally-mounted datasets). Test only the classes we have local audio for.
    model_set = set(model_classes)
    available = [c for c in mixer.class_names if c in model_set]
    if not available:
        raise RuntimeError(
            "No local classes overlap the model vocabulary — nothing to test.")
    test_classes = available[:n_classes]
    print(f"Model vocabulary: {len(model_classes)} classes; "
          f"{len(available)} have local audio for testing.\n")

    # ---------- Test 1: output magnitude for a present class ----------
    print("Test 1 — Output magnitude on a known present class")
    print(f"  {'Class':<22}{'Input max':>11}{'Output max':>13}{'Out/In':>9}")
    print("  " + "-" * 55)
    ratios = []
    for cls in test_classes:
        clip = _clean_clip(mixer, cls)
        lin, log = _model_input(clip)
        q = _query(model_classes, cls)
        est = model.predict([log[None], q[None], lin[None]], verbose=0)
        inp_max, out_max = float(np.max(lin)), float(np.max(est))
        ratio = out_max / (inp_max + 1e-8)
        ratios.append(ratio)
        mark = "ok" if ratio > 0.10 else "LOW"
        print(f"  {cls:<22}{inp_max:>11.3f}{out_max:>13.3f}{ratio:>9.2f}  {mark}")
    mean_ratio = float(np.mean(ratios))
    test1_pass = mean_ratio > 0.10
    print(f"  → mean out/in ratio: {mean_ratio:.3f} "
          f"({'PASS' if test1_pass else 'FAIL — model may be collapsed'})\n")

    # ---------- Test 2: FiLM discrimination on a single input ----------
    print("Test 2 — FiLM discrimination (same input, different queries)")
    test_cls = test_classes[0]
    clip = _clean_clip(mixer, test_cls)
    lin, log = _model_input(clip)
    energies = {}
    for cls in test_classes[:6]:
        q = _query(model_classes, cls)
        est = model.predict([log[None], q[None], lin[None]], verbose=0)
        energies[cls] = float(np.mean(est ** 2))
    correct_e = energies[test_cls]
    other_e = float(np.mean([v for k, v in energies.items() if k != test_cls]))
    print(f"  Input clip is: {test_cls}")
    for cls, e in energies.items():
        mark = " <- correct query" if cls == test_cls else ""
        print(f"  {cls:<22} energy = {e:.6f}{mark}")
    advantage = (correct_e - other_e) / (other_e + 1e-9)
    test2_pass = advantage > 0.5
    print(f"  → correct vs. wrong advantage: {advantage:+.2f}x "
          f"({'PASS' if test2_pass else 'FAIL — FiLM not discriminating'})\n")

    # ---------- Test 3: per-class correct/wrong advantage ----------
    print("Test 3 — Per-class correct-vs-wrong query advantage")
    print(f"  {'Class':<22}{'Correct':>10}{'Wrong avg':>11}{'Advantage':>11}")
    print("  " + "-" * 54)
    advantages = []
    for cls in test_classes:
        clip = _clean_clip(mixer, cls)
        lin, log = _model_input(clip)
        q = _query(model_classes, cls)
        est = model.predict([log[None], q[None], lin[None]], verbose=0)
        correct_e = float(np.mean(est ** 2))

        wrong_classes = [c for c in test_classes if c != cls][:3]
        wrong_es = []
        for wc in wrong_classes:
            qw = _query(model_classes, wc)
            estw = model.predict([log[None], qw[None], lin[None]], verbose=0)
            wrong_es.append(float(np.mean(estw ** 2)))
        wrong_avg = float(np.mean(wrong_es))
        adv = (correct_e - wrong_avg) / (wrong_avg + 1e-9)
        advantages.append(adv)
        mark = "ok" if adv > 0.3 else "LOW"
        print(f"  {cls:<22}{correct_e:>10.4f}{wrong_avg:>11.4f}"
              f"{adv:>10.2f}x  {mark}")
    mean_adv = float(np.mean(advantages))
    test3_pass = mean_adv > 0.5
    print(f"  → mean advantage: {mean_adv:+.2f}x "
          f"({'PASS' if test3_pass else 'FAIL — model not class-specific'})\n")

    # ---------- Overall verdict ----------
    overall = test1_pass and test2_pass and test3_pass
    print("=" * 60)
    print(f"  OVERALL: {'HEALTHY — model produces class-specific output' if overall else 'BROKEN — model has collapsed or failed to learn class conditioning'}")
    print("=" * 60 + "\n")
    return {
        "pass": overall,
        "mean_out_in_ratio": mean_ratio,
        "film_advantage": advantage,
        "mean_class_advantage": mean_adv,
    }


if __name__ == "__main__":
    diagnose(
        model_path=BASE_DIR / "saved_models" / "separation_models"
                            / "separator_unet_film_multi_v2.3.h5",
        data_root=BASE_DIR / "data" / "raw",
    )
