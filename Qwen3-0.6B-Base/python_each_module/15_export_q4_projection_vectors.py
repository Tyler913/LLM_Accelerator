import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

MATRIX_NAME = "q_proj"
START_ROW = 0
OUT_ROWS = 16
HIDDEN_SIZE = 1024
Q4_GROUP_SIZE = 64
Q4_VALUES_PER_BYTE = 2
ACT_WIDTH = 16
SCALE_WIDTH = 16
ROW_ACC_WIDTH = 48


def twos_complement_hex(value: int, width_bits: int) -> str:
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    return f"{value & mask:0{hex_digits}x}"


def unpack_signed_int4(packed: np.ndarray, values_per_row: int) -> np.ndarray:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((packed.shape[0], values_per_row), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    q = nibbles.astype(np.int16)
    q[q >= 8] -= 16
    return q.astype(np.int8)


def write_hex_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data_path = Q4_VECTOR_DIR / "qkv_layer0_last_token_q4.npz"
    manifest_path = Q4_VECTOR_DIR / "manifest.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing Q4 vector file: {data_path}")

    data = np.load(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    activation_q4_12 = data["input_norm_q4_12"].astype(np.int16)
    packed_weight = data[f"{MATRIX_NAME[0]}_weight_q4_packed"][START_ROW : START_ROW + OUT_ROWS]
    scale_q2_14 = data[f"{MATRIX_NAME[0]}_scale_q2_14"][START_ROW : START_ROW + OUT_ROWS]

    weight_q4 = unpack_signed_int4(packed_weight, HIDDEN_SIZE)
    group_count = HIDDEN_SIZE // Q4_GROUP_SIZE
    x_grouped = activation_q4_12.astype(np.int64).reshape(group_count, Q4_GROUP_SIZE)
    w_grouped = weight_q4.astype(np.int64).reshape(OUT_ROWS, group_count, Q4_GROUP_SIZE)
    partial_sums = np.sum(w_grouped * x_grouped[None, :, :], axis=2, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14.astype(np.int64)
    output_q26 = np.sum(scaled_sums, axis=1, dtype=np.int64)

    actual_q4 = data[f"actual_{MATRIX_NAME[0]}_q4"][START_ROW : START_ROW + OUT_ROWS]
    recomputed_float = output_q26.astype(np.float64) / float(1 << 26)
    max_recompute_diff = float(
        np.max(np.abs(recomputed_float.astype(np.float32) - actual_q4.astype(np.float32)))
    )
    if max_recompute_diff != 0.0:
        raise RuntimeError(
            f"Internal Q26 recompute mismatch against artifact: {max_recompute_diff}"
        )

    prefix = "q4_gemv_projection_1024_real"
    activation_path = SIM_VECTOR_DIR / f"{prefix}_activation.hex"
    weight_path = SIM_VECTOR_DIR / f"{prefix}_weight.hex"
    scale_path = SIM_VECTOR_DIR / f"{prefix}_scale.hex"
    expected_path = SIM_VECTOR_DIR / f"{prefix}_expected.hex"

    write_hex_lines(
        activation_path,
        [twos_complement_hex(int(value), ACT_WIDTH) for value in activation_q4_12],
    )
    write_hex_lines(
        weight_path,
        [twos_complement_hex(int(value), 4) for value in weight_q4.reshape(-1)],
    )
    write_hex_lines(
        scale_path,
        [twos_complement_hex(int(value), SCALE_WIDTH) for value in scale_q2_14.reshape(-1)],
    )
    write_hex_lines(
        expected_path,
        [twos_complement_hex(int(value), ROW_ACC_WIDTH) for value in output_q26],
    )

    print("Exported Q4 projection RTL test vectors")
    print("=" * 80)
    print(f"Source: {data_path}")
    print(f"Format: {manifest['weight_format']['name']}")
    print(f"Matrix: {MATRIX_NAME}")
    print(f"Rows: {START_ROW}..{START_ROW + OUT_ROWS - 1}")
    print(f"Activation: {activation_path}")
    print(f"Weight:     {weight_path}")
    print(f"Scale:      {scale_path}")
    print(f"Expected:   {expected_path}")
    print(f"Expected Q26 outputs: {[int(value) for value in output_q26]}")


if __name__ == "__main__":
    main()
