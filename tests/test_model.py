"""
Software-only model validation suite for the Selective Noise Cancellation project.

Runs entirely without hardware. Uses the trained .h5 model and the processed
.npy dataset to verify correctness, measure inference latency on this host
machine, and find per-class classification thresholds that maximise F1 score.

Run from the project root:
    python tests/test_model.py

All tests print a human-readable report. The optimal thresholds printed at the
end of the run should be copied into the ANCPredictor (inference.py) to replace
the hard-coded 0.5 threshold.

Prerequisites:
    * saved_models/base_models/best_mobilenetv2_multilabel.h5
    * data/processed/training_pipeline/{X_multi_features.npy,
                                        y_multi_labels.npy,
                                        class_names.json}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _result(label: str, status: str, detail: str = "") -> None:
    icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]"}[status]
    line = f"  {icon}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


# ---------------------------------------------------------------------------
# Test 1: Model file and loading
# ---------------------------------------------------------------------------

def test_model_loading(model_path: Path):
    _header("Test 1: Model loading")
    import tensorflow as tf

    if not model_path.exists():
        _result("Model file exists", FAIL, str(model_path))
        return None

    _result("Model file exists", PASS, f"{model_path.stat().st_size / 1e6:.1f} MB")

    try:
        model = tf.keras.models.load_model(model_path)
        _result("Model loads without error", PASS)
    except Exception as exc:
        _result("Model loads without error", FAIL, str(exc))
        return None

    inp = model.input_shape[1:]
    out = model.output_shape[1]
    _result("Input shape", PASS if inp == (64, 101, 3) else FAIL, str(inp))
    _result("Output units", PASS if out == 8 else WARN, str(out))

    last_activation = model.layers[-1].get_config().get("activation", "unknown")
    _result(
        "Output activation is sigmoid",
        PASS if last_activation == "sigmoid" else FAIL,
        last_activation,
    )
    return model


# ---------------------------------------------------------------------------
# Test 2: Class names contract
# ---------------------------------------------------------------------------

def test_class_names(data_dir: Path):
    _header("Test 2: Class-name contract")
    from data_preparation.synthetic_data_generator import CANONICAL_CLASSES

    json_path = data_dir / "class_names.json"
    if not json_path.exists():
        _result("class_names.json exists", FAIL)
        return None

    with json_path.open() as f:
        stored = json.load(f)

    _result("class_names.json exists", PASS)
    _result(
        "Matches CANONICAL_CLASSES",
        PASS if tuple(stored) == CANONICAL_CLASSES else FAIL,
        str(stored),
    )
    _result("Sorted alphabetically", PASS if stored == sorted(stored) else FAIL)
    return stored


# ---------------------------------------------------------------------------
# Test 3: Shape / dtype sanity on a batch
# ---------------------------------------------------------------------------

def test_batch_inference(model, data_dir: Path):
    _header("Test 3: Batch inference — shape and dtype")
    import tensorflow as tf

    x_path = data_dir / "X_multi_features.npy"
    if not x_path.exists():
        _result("Feature file exists", FAIL)
        return

    x = np.load(x_path, mmap_mode="r")
    _result("Feature file exists", PASS, f"shape={x.shape} dtype={x.dtype}")

    batch = x[:8].astype(np.float32)
    try:
        preds = model.predict(batch, verbose=0)
        _result("Prediction succeeds on 8-sample batch", PASS)
    except Exception as exc:
        _result("Prediction succeeds on 8-sample batch", FAIL, str(exc))
        return

    _result("Output shape is (8, 8)", PASS if preds.shape == (8, 8) else FAIL, str(preds.shape))
    in_range = bool(np.all(preds >= 0) and np.all(preds <= 1))
    _result("All outputs in [0, 1]", PASS if in_range else FAIL)
    _result("No NaN in outputs", PASS if not np.any(np.isnan(preds)) else FAIL)


# ---------------------------------------------------------------------------
# Test 4: Inference latency
# ---------------------------------------------------------------------------

def test_latency(model, n_runs: int = 100):
    _header(f"Test 4: Single-sample inference latency ({n_runs} runs)")
    import tensorflow as tf

    dummy = np.random.randn(1, 64, 101, 3).astype(np.float32)
    # Warm-up
    for _ in range(5):
        model.predict(dummy, verbose=0)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    print(f"  Mean latency : {mean_ms:.1f} ms")
    print(f"  P95  latency : {p95_ms:.1f} ms")
    # Target: <50 ms on MCU; laptop float32 is much slower, so we just report.
    status = PASS if mean_ms < 500 else WARN  # generous threshold for CPU laptop
    _result("Latency within CI bounds (<500 ms on host CPU)", status, f"{mean_ms:.1f} ms mean")
    return mean_ms


# ---------------------------------------------------------------------------
# Test 5: Per-class threshold sweep → optimal F1
# ---------------------------------------------------------------------------

def test_threshold_sweep(model, data_dir: Path, class_names: list):
    _header("Test 5: Per-class threshold sweep (F1 optimisation)")
    from sklearn.model_selection import train_test_split

    x = np.load(data_dir / "X_multi_features.npy").astype(np.float32)
    y = np.load(data_dir / "y_multi_labels.npy").astype(np.float32)

    # Reproduce the exact same test split used during training (seed=42, size=10%).
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
    print(f"  Test set: {x_test.shape[0]} samples")

    print("  Running predictions on test set...", end=" ", flush=True)
    preds = model.predict(x_test, batch_size=64, verbose=0)
    print("done.")

    thresholds = np.arange(0.05, 0.96, 0.05)
    optimal_thresholds = {}
    print()
    print(f"  {'Class':<18} {'Opt.Thresh':>10} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for i, cls in enumerate(class_names):
        best_f1, best_thresh, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0
        y_true = y_test[:, i]
        y_prob = preds[:, i]
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(float)
            tp = float(np.sum((y_pred == 1) & (y_true == 1)))
            fp = float(np.sum((y_pred == 1) & (y_true == 0)))
            fn = float(np.sum((y_pred == 0) & (y_true == 1)))
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_thresh, best_prec, best_rec = f1, thr, prec, rec
        optimal_thresholds[cls] = round(float(best_thresh), 2)
        print(f"  {cls:<18} {best_thresh:>10.2f} {best_f1:>8.3f} {best_prec:>8.3f} {best_rec:>8.3f}")

    print()
    print("  Copy these thresholds into ANCPredictor (inference.py):")
    print(f"  THRESHOLDS = {json.dumps(optimal_thresholds, indent=4)}")
    return optimal_thresholds


# ---------------------------------------------------------------------------
# Test 6: WAV file inference (optional)
# ---------------------------------------------------------------------------

def test_wav_inference(model, class_names: list):
    _header("Test 6: WAV file inference (optional)")
    import librosa

    wav_path = BASE_DIR / "data" / "test_samples" / "test_sound.wav"
    if not wav_path.exists():
        _result(
            "test_sound.wav",
            WARN,
            f"Not found at {wav_path}. Place a WAV there to enable this test.",
        )
        return

    TARGET_SR = 16000
    audio, _ = librosa.load(wav_path, sr=TARGET_SR)
    # Take the first 1-second window
    window = audio[:TARGET_SR] if len(audio) >= TARGET_SR else np.pad(audio, (0, TARGET_SR - len(audio)))

    mel = librosa.feature.melspectrogram(y=window, sr=TARGET_SR, n_fft=400, hop_length=160, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    tensor = np.repeat(log_mel[..., np.newaxis], 3, axis=-1)[np.newaxis].astype(np.float32)

    probs = model.predict(tensor, verbose=0)[0]
    _result("WAV loaded and preprocessed", PASS, f"{len(audio)/TARGET_SR:.2f}s audio → (64, 101, 3) tensor")
    print("\n  Predictions:")
    for cls, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {cls:<18} {prob:.3f}  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = BASE_DIR / "saved_models" / "base_models" / "best_mobilenetv2_multilabel.h5"
    DATA_DIR   = BASE_DIR / "data" / "processed" / "training_pipeline"

    print("\nSelective Noise Cancellation — Model Validation Report")
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_DIR}")

    model       = test_model_loading(MODEL_PATH)
    class_names = test_class_names(DATA_DIR)

    if model is None:
        print("\nCritical: model failed to load. Remaining tests skipped.")
        sys.exit(1)

    if class_names is None:
        class_names = ["car_horn", "crying_baby", "dog", "engine",
                       "keyboard_typing", "rain", "siren", "wind"]
        print("  (Using hard-coded class names as fallback)")

    test_batch_inference(model, DATA_DIR)
    test_latency(model)

    if (DATA_DIR / "X_multi_features.npy").exists():
        thresholds = test_threshold_sweep(model, DATA_DIR, class_names)
    else:
        print("\n[WARN] Dataset not found — skipping threshold sweep.")

    test_wav_inference(model, class_names)

    print("\n" + "=" * 60)
    print("  Validation complete.")
    print("=" * 60 + "\n")
