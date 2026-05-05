"""
Export a .tflite model as a TensorFlow Lite for Microcontrollers (TFLM) compatible
C array (.h + .cc). Output is byte-identical to `xxd -i model.tflite` but written
in pure Python, with 16-byte alignment (required by TFLM on Cortex-M / Xtensa).

Drop the resulting files into your TFLM project (e.g. ESP-IDF + esp-tflite-micro,
STM32CubeIDE + X-CUBE-AI, Arduino_TensorFlowLite) and reference `g_anc_model`
when constructing `tflite::MicroInterpreter`.
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_HEADER_TEMPLATE = """// Auto-generated from {source_name} -- DO NOT EDIT.
// Selective Noise Cancelling MobileNetV2 INT8 model for TFLite Micro.

#ifndef {guard}
#define {guard}

extern const unsigned char {var_name}[];
extern const unsigned int {var_name}_len;

#endif  // {guard}
"""

_SOURCE_TEMPLATE = """// Auto-generated from {source_name} -- DO NOT EDIT.
#include "{header_name}"

alignas(16) const unsigned char {var_name}[] = {{
{byte_array}
}};
const unsigned int {var_name}_len = {byte_count};
"""


def tflite_to_c_array(tflite_path: Path, output_dir: Path, var_name: str = "g_anc_model") -> None:
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found at {tflite_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data = tflite_path.read_bytes()

    bytes_per_line = 12
    lines = [
        "  " + ", ".join(f"0x{b:02x}" for b in data[i:i + bytes_per_line]) + ","
        for i in range(0, len(data), bytes_per_line)
    ]

    header_name = f"{var_name}.h"
    source_name = f"{var_name}.cc"
    guard = f"{var_name.upper()}_H_"

    (output_dir / header_name).write_text(_HEADER_TEMPLATE.format(
        source_name=tflite_path.name, guard=guard, var_name=var_name,
    ))
    (output_dir / source_name).write_text(_SOURCE_TEMPLATE.format(
        source_name=tflite_path.name, header_name=header_name, var_name=var_name,
        byte_array="\n".join(lines), byte_count=len(data),
    ))

    logging.info(
        f"Wrote {output_dir / header_name} and {output_dir / source_name} "
        f"({len(data)} bytes / {len(data) / 1024:.1f} KB)."
    )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    tflite_to_c_array(
        tflite_path=BASE_DIR / "saved_models" / "tflite" / "model_full_int8.tflite",
        output_dir=BASE_DIR / "saved_models" / "tflm_c_array",
    )
