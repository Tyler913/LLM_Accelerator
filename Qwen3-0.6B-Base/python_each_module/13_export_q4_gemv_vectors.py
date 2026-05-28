import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FP32_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_fp32_v0"
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"

HIDDEN_SIZE = 1024
Q4_GROUP_SIZE = 64
Q4_VALUES_PER_BYTE = 2
Q4_MIN = -8
Q4_MAX = 7
ACT_WIDTH = 16
ACT_FRAC = 12
SCALE_WIDTH = 16
SCALE_FRAC = 14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source_vector() -> dict[str, Any]:
    manifest_path = FP32_VECTOR_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    qkv_entry = next(
        (entry for entry in manifest["vectors"] if entry["file"] == "qkv_layer0_last_token.npz"),
        None,
    )
    if qkv_entry is None:
        raise RuntimeError("Source manifest does not describe qkv_layer0_last_token.npz")

    qkv_path = FP32_VECTOR_DIR / qkv_entry["file"]
    actual = sha256_file(qkv_path)
    expected = qkv_entry["sha256"]
    if actual != expected:
        raise RuntimeError(
            f"Source checksum mismatch for {qkv_path.name}: expected {expected}, got {actual}"
        )
    return manifest


def quantize_activation_q4_12(x: np.ndarray) -> np.ndarray:
    scale = float(1 << ACT_FRAC)
    lower = -(1 << (ACT_WIDTH - 1))
    upper = (1 << (ACT_WIDTH - 1)) - 1
    quantized = np.rint(x.astype(np.float32) * scale)
    return np.clip(quantized, lower, upper).astype(np.int16)


def pack_signed_int4(q: np.ndarray) -> np.ndarray:
    if q.ndim != 2:
        raise ValueError(f"Expected a 2D int4 matrix, got shape {q.shape}")
    if q.shape[1] % Q4_VALUES_PER_BYTE != 0:
        raise ValueError("Input feature count must be even for int4 packing")

    q_i16 = q.astype(np.int16)
    if np.any(q_i16 < Q4_MIN) or np.any(q_i16 > Q4_MAX):
        raise ValueError("int4 values must be in [-8, 7]")

    nibbles = q_i16 & 0xF
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    return (low | (high << 4)).astype(np.uint8)


def unpack_signed_int4(packed: np.ndarray, values_per_row: int) -> np.ndarray:
    if values_per_row % Q4_VALUES_PER_BYTE != 0:
        raise ValueError("values_per_row must be even")
    if packed.shape[1] != values_per_row // Q4_VALUES_PER_BYTE:
        raise ValueError(
            f"Packed width {packed.shape[1]} does not match {values_per_row} int4 values"
        )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((packed.shape[0], values_per_row), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    q = nibbles.astype(np.int16)
    q[q >= 8] -= 16
    return q.astype(np.int8)


def quantize_weight_q4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = np.ascontiguousarray(weight.astype(np.float32))
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {weight.shape}")

    out_features, in_features = weight.shape
    if in_features != HIDDEN_SIZE:
        raise ValueError(f"Expected in_features={HIDDEN_SIZE}, got {in_features}")
    if in_features % Q4_GROUP_SIZE != 0:
        raise ValueError("Input feature count must be divisible by Q4_GROUP_SIZE")

    groups_per_row = in_features // Q4_GROUP_SIZE
    grouped = weight.reshape(out_features, groups_per_row, Q4_GROUP_SIZE)
    absmax = np.max(np.abs(grouped), axis=2)
    raw_scale = absmax / float(Q4_MAX)

    scale_factor = float(1 << SCALE_FRAC)
    scale_q64 = np.rint(raw_scale * scale_factor).astype(np.int64)
    scale_q64 = np.where((absmax > 0.0) & (scale_q64 == 0), 1, scale_q64)
    if np.any(scale_q64 < 0) or np.any(scale_q64 > np.iinfo(np.uint16).max):
        raise ValueError("Q4 scale does not fit unsigned 16-bit Q2.14 storage")

    scale_q = scale_q64.astype(np.uint16)
    scale_used = scale_q.astype(np.float32) / scale_factor
    safe_scale = np.where(scale_used > 0.0, scale_used, 1.0).astype(np.float32)
    q_grouped = np.rint(grouped / safe_scale[:, :, None])
    q_grouped = np.clip(q_grouped, Q4_MIN, Q4_MAX).astype(np.int8)
    q_int4 = q_grouped.reshape(out_features, in_features)
    q_packed = pack_signed_int4(q_int4)

    unpacked = unpack_signed_int4(q_packed, in_features)
    if not np.array_equal(unpacked, q_int4):
        raise RuntimeError("Internal int4 pack/unpack self-check failed")

    return q_packed, scale_q, q_int4


def q4_gemv_from_ints(
    x_q4_12: np.ndarray,
    weight_q4: np.ndarray,
    scale_q2_14: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_q4_12.shape != (HIDDEN_SIZE,):
        raise ValueError(f"Expected activation shape [{HIDDEN_SIZE}], got {x_q4_12.shape}")

    out_features, in_features = weight_q4.shape
    groups_per_row = in_features // Q4_GROUP_SIZE
    x_grouped = x_q4_12.astype(np.int64).reshape(groups_per_row, Q4_GROUP_SIZE)
    w_grouped = weight_q4.astype(np.int64).reshape(out_features, groups_per_row, Q4_GROUP_SIZE)
    partial_sums = np.sum(w_grouped * x_grouped[None, :, :], axis=2, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14.astype(np.int64)
    output_q26 = np.sum(scaled_sums, axis=1, dtype=np.int64)
    output = output_q26.astype(np.float64) / float(1 << (ACT_FRAC + SCALE_FRAC))
    return output.astype(np.float32), partial_sums, scaled_sums


def compare_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main() -> None:
    source_manifest = verify_source_vector()
    qkv_data = np.load(FP32_VECTOR_DIR / "qkv_layer0_last_token.npz")

    input_norm = np.ascontiguousarray(qkv_data["input_norm"].astype(np.float32))
    input_norm_q4_12 = quantize_activation_q4_12(input_norm)

    q_packed, q_scale, q_int4 = quantize_weight_q4(qkv_data["q_weight"])
    k_packed, k_scale, k_int4 = quantize_weight_q4(qkv_data["k_weight"])
    v_packed, v_scale, v_int4 = quantize_weight_q4(qkv_data["v_weight"])

    q_actual, q_partial, q_scaled = q4_gemv_from_ints(input_norm_q4_12, q_int4, q_scale)
    k_actual, _k_partial, _k_scaled = q4_gemv_from_ints(input_norm_q4_12, k_int4, k_scale)
    v_actual, _v_partial, _v_scaled = q4_gemv_from_ints(input_norm_q4_12, v_int4, v_scale)

    metrics = {
        "q_proj": compare_metrics(q_actual, qkv_data["expected_q"]),
        "k_proj": compare_metrics(k_actual, qkv_data["expected_k"]),
        "v_proj": compare_metrics(v_actual, qkv_data["expected_v"]),
    }

    qkv_q4_path = Q4_VECTOR_DIR / "qkv_layer0_last_token_q4.npz"
    dot64_path = Q4_VECTOR_DIR / "q_proj_row0_group0_dot64.npz"

    write_npz(
        qkv_q4_path,
        input_norm_fp32=input_norm,
        input_norm_q4_12=input_norm_q4_12,
        q_weight_q4_packed=q_packed,
        q_scale_q2_14=q_scale,
        k_weight_q4_packed=k_packed,
        k_scale_q2_14=k_scale,
        v_weight_q4_packed=v_packed,
        v_scale_q2_14=v_scale,
        expected_q_fp32=qkv_data["expected_q"].astype(np.float32),
        expected_k_fp32=qkv_data["expected_k"].astype(np.float32),
        expected_v_fp32=qkv_data["expected_v"].astype(np.float32),
        actual_q_q4=q_actual,
        actual_k_q4=k_actual,
        actual_v_q4=v_actual,
        prompt_position=qkv_data["prompt_position"],
        token_id=qkv_data["token_id"],
    )

    first_group = slice(0, Q4_GROUP_SIZE)
    write_npz(
        dot64_path,
        activation_q4_12=input_norm_q4_12[first_group],
        weight_q4_packed=q_packed[0, : Q4_GROUP_SIZE // Q4_VALUES_PER_BYTE],
        weight_q4_unpacked=q_int4[0, first_group],
        scale_q2_14=np.array(q_scale[0, 0], dtype=np.uint16),
        partial_sum_int64=np.array(q_partial[0, 0], dtype=np.int64),
        scaled_sum_q26_int64=np.array(q_scaled[0, 0], dtype=np.int64),
        expected_float32=np.array(
            q_scaled[0, 0] / float(1 << (ACT_FRAC + SCALE_FRAC)),
            dtype=np.float32,
        ),
        matrix_name=np.array("q_proj"),
        row_index=np.array(0, dtype=np.int32),
        group_index=np.array(0, dtype=np.int32),
    )

    manifest = {
        "format_version": 1,
        "name": "qwen3_0p6b_q4_v0",
        "purpose": "Verilog-facing custom Q4 vectors for first Layer 0 Q/K/V GEMV bring-up.",
        "source_vector_set": source_manifest["name"],
        "source_vector_file": "qkv_layer0_last_token.npz",
        "scope": {
            "model": "Qwen/Qwen3-0.6B-Base",
            "layer": 0,
            "matrices": ["q_proj", "k_proj", "v_proj"],
            "reason": (
                "This scope exercises the first post-RMSNorm GEMV path and the "
                "main Q4 packing/scale contract without freezing the entire model format."
            ),
        },
        "activation_format": {
            "name": "signed_q4_12",
            "width_bits": ACT_WIDTH,
            "fraction_bits": ACT_FRAC,
            "scale": f"2^-{ACT_FRAC}",
            "rounding": "numpy_rint_round_to_nearest_even",
            "stored_dtype": "int16",
        },
        "weight_format": {
            "name": "custom_groupwise_symmetric_q4_v0",
            "stored_dtype": "packed_uint8",
            "q_min": Q4_MIN,
            "q_max": Q4_MAX,
            "zero_point": None,
            "group_size": Q4_GROUP_SIZE,
            "groups_per_row": HIDDEN_SIZE // Q4_GROUP_SIZE,
            "scale_dtype": "uint16_q2_14",
            "scale_fraction_bits": SCALE_FRAC,
            "scale_formula": "round((max(abs(weight_group)) / 7) * 2^14)",
            "nonzero_scale_rule": "if absmax > 0 and the rounded scale is 0, store scale_q2_14 = 1",
            "quant_formula": "round(weight / (scale_q2_14 / 2^14)), clamped to [-8, 7]",
            "packing": "two signed two's-complement int4 values per byte; even input index in low nibble, odd input index in high nibble",
            "layout": "row-major by output row; groups are contiguous 64-column spans inside each row",
        },
        "integer_gemv_formula": {
            "partial_sum": "partial[row, group] = sum_j activation_q4_12[col] * weight_q4[row, col]",
            "scaled_sum": "scaled[row, group] = partial[row, group] * scale_q2_14[row, group]",
            "output": "sum_group(scaled[row, group]) / 2^(12 + 14)",
        },
        "arrays": {
            qkv_q4_path.name: {
                "input_norm_fp32": [HIDDEN_SIZE],
                "input_norm_q4_12": [HIDDEN_SIZE],
                "q_weight_q4_packed": [2048, HIDDEN_SIZE // Q4_VALUES_PER_BYTE],
                "q_scale_q2_14": [2048, HIDDEN_SIZE // Q4_GROUP_SIZE],
                "k_weight_q4_packed": [1024, HIDDEN_SIZE // Q4_VALUES_PER_BYTE],
                "k_scale_q2_14": [1024, HIDDEN_SIZE // Q4_GROUP_SIZE],
                "v_weight_q4_packed": [1024, HIDDEN_SIZE // Q4_VALUES_PER_BYTE],
                "v_scale_q2_14": [1024, HIDDEN_SIZE // Q4_GROUP_SIZE],
                "actual_q_q4": [2048],
                "actual_k_q4": [1024],
                "actual_v_q4": [1024],
            },
            dot64_path.name: {
                "activation_q4_12": [Q4_GROUP_SIZE],
                "weight_q4_packed": [Q4_GROUP_SIZE // Q4_VALUES_PER_BYTE],
                "weight_q4_unpacked": [Q4_GROUP_SIZE],
                "scale_q2_14": [],
                "partial_sum_int64": [],
                "scaled_sum_q26_int64": [],
                "expected_float32": [],
            },
        },
        "metrics_against_fp32_expected": metrics,
        "files": [
            {"file": qkv_q4_path.name, "sha256": sha256_file(qkv_q4_path)},
            {"file": dot64_path.name, "sha256": sha256_file(dot64_path)},
        ],
    }

    manifest_path = Q4_VECTOR_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported Q4 GEMV test vectors")
    print("=" * 80)
    print(f"Vector directory: {Q4_VECTOR_DIR}")
    print("Scope: layer0 q_proj/k_proj/v_proj")
    print(f"Activation format: signed int16 Q4.{ACT_FRAC}")
    print(f"Weight format: signed int4, group size {Q4_GROUP_SIZE}, uint16 Q2.{SCALE_FRAC} scales")
    for name, values in metrics.items():
        print(
            f"{name}: max_abs_error={values['max_abs_error']:.8f} "
            f"mean_abs_error={values['mean_abs_error']:.8f} rmse={values['rmse']:.8f}"
        )
    print(f"Wrote: {qkv_q4_path.name}")
    print(f"Wrote: {dot64_path.name}")
    print(f"Wrote: {manifest_path.name}")


if __name__ == "__main__":
    main()
