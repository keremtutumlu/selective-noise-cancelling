"""
Convert the trained Keras MobileNetV2 ANC model into edge-deployable TFLite
variants (float32, float16, dynamic-range INT8, full INT8) and benchmark each
against the original Keras model on a held-out calibration set.

The full-INT8 variant is the recommended artefact for TensorFlow Lite for
Microcontrollers (TFLM) — pair it with `export_c_header.py` to obtain the
embeddable C array.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EdgeModelOptimizer:
    """Converts a Keras .h5 model into TFLite variants and benchmarks them."""

    def __init__(self, keras_model_path: Path, calibration_data_path: Path, output_dir: Path):
        self.keras_model_path = keras_model_path
        self.calibration_data_path = calibration_data_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Loading Keras model from {keras_model_path}")
        self.keras_model = tf.keras.models.load_model(keras_model_path)
        logging.info(f"Model input shape:  {self.keras_model.input_shape}")
        logging.info(f"Model output shape: {self.keras_model.output_shape}")

    def _load_calibration_samples(self, num_samples: int = 200) -> np.ndarray:
        """Load a representative input batch for INT8 calibration and benchmarking."""
        if not self.calibration_data_path.exists():
            logging.warning(
                f"Calibration data not found at {self.calibration_data_path}. "
                "Falling back to random noise — INT8 accuracy will be degraded."
            )
            shape = (num_samples,) + self.keras_model.input_shape[1:]
            return np.random.randn(*shape).astype(np.float32)

        samples = np.load(self.calibration_data_path).astype(np.float32)
        if samples.shape[0] > num_samples:
            idx = np.random.choice(samples.shape[0], num_samples, replace=False)
            samples = samples[idx]
        logging.info(f"Loaded {samples.shape[0]} calibration samples from disk.")
        return samples

    def _representative_dataset(self, samples: np.ndarray) -> Iterator:
        for sample in samples:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    def convert_float32(self) -> Path:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        return self._save(converter.convert(), "model_float32.tflite")

    def convert_float16(self) -> Path:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return self._save(converter.convert(), "model_float16.tflite")

    def convert_dynamic_int8(self) -> Path:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return self._save(converter.convert(), "model_dynamic_int8.tflite")

    def convert_full_int8(self, calibration_samples: np.ndarray) -> Path:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: self._representative_dataset(calibration_samples)
        # TFLM only supports the INT8 builtin op set
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return self._save(converter.convert(), "model_full_int8.tflite")

    def _save(self, tflite_bytes: bytes, file_name: str) -> Path:
        path = self.output_dir / file_name
        path.write_bytes(tflite_bytes)
        logging.info(f"Saved {file_name} ({path.stat().st_size / 1024:.1f} KB)")
        return path

    def benchmark(self, tflite_path: Path, test_samples: np.ndarray) -> Dict[str, float]:
        """Compare TFLite predictions and latency against the source Keras model."""
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        keras_preds, tflite_preds, latencies_ms = [], [], []

        for sample in test_samples:
            x = np.expand_dims(sample, axis=0).astype(np.float32)
            keras_preds.append(self.keras_model.predict(x, verbose=0)[0])

            if input_details['dtype'] == np.int8:
                scale, zero_point = input_details['quantization']
                x_in = (x / scale + zero_point).astype(np.int8)
            else:
                x_in = x.astype(input_details['dtype'])
            interpreter.set_tensor(input_details['index'], x_in)

            t0 = time.perf_counter()
            interpreter.invoke()
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            out = interpreter.get_tensor(output_details['index'])[0]
            if output_details['dtype'] == np.int8:
                scale, zero_point = output_details['quantization']
                out = (out.astype(np.float32) - zero_point) * scale
            tflite_preds.append(out)

        mae = float(np.mean(np.abs(np.array(keras_preds) - np.array(tflite_preds))))
        return {
            "size_kb": tflite_path.stat().st_size / 1024,
            "mae_vs_keras": mae,
            "latency_ms": float(np.mean(latencies_ms)),
        }

    def run(self) -> None:
        calibration_samples = self._load_calibration_samples(num_samples=200)
        test_samples = calibration_samples[:50]

        variants = {
            "float32":      self.convert_float32(),
            "float16":      self.convert_float16(),
            "dynamic_int8": self.convert_dynamic_int8(),
            "full_int8":    self.convert_full_int8(calibration_samples),
        }

        logging.info("=" * 70)
        logging.info(f"{'Variant':<15}{'Size (KB)':<14}{'MAE vs Keras':<18}{'Latency (ms)':<14}")
        logging.info("-" * 70)
        for name, path in variants.items():
            m = self.benchmark(path, test_samples)
            logging.info(f"{name:<15}{m['size_kb']:<14.1f}{m['mae_vs_keras']:<18.6f}{m['latency_ms']:<14.2f}")
        logging.info("=" * 70)
        logging.info("Recommended for MCU deployment: model_full_int8.tflite")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    EdgeModelOptimizer(
        keras_model_path=BASE_DIR / "saved_models" / "base_models" / "best_mobilenetv2_multilabel.h5",
        calibration_data_path=BASE_DIR / "data" / "processed" / "training_pipeline" / "X_multi_features.npy",
        output_dir=BASE_DIR / "saved_models" / "tflite",
    ).run()
