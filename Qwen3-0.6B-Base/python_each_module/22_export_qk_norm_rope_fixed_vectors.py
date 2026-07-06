from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    PROMPT,
    encode_prompt,
    load_model,
    manual_rope_cos_sin,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

PREFIX = "qk_norm_rope_stage_128_real"
MODULE_NAME = "layer0.self_attn.qk_norm_rope_stage"

HIDDEN_SIZE = 1024
NUM_Q_HEADS = 16
NUM_K_HEADS = 8
HEAD_DIM = 128
Q_ROWS = NUM_Q_HEADS * HEAD_DIM
K_ROWS = NUM_K_HEADS * HEAD_DIM
Q4_GROUP_SIZE = 64
GROUP_COUNT = HIDDEN_SIZE // Q4_GROUP_SIZE
Q4_VALUES_PER_BYTE = 2

IN_WIDTH = 24
IN_FRAC = 12
GAMMA_WIDTH = 16
GAMMA_FRAC = 7
INV_RMS_WIDTH = 24
INV_RMS_FRAC = 16
OUT_WIDTH = 24
OUT_FRAC = 12
SUM_WIDTH = 64
TRIG_WIDTH = 16
TRIG_FRAC = 15
ACT_FRAC = 12
SCALE_FRAC = 14
Q26_TO_Q12_SHIFT = SCALE_FRAC
EPS_Q24 = 17
RMS_OUTPUT_SHIFT = IN_FRAC + INV_RMS_FRAC + GAMMA_FRAC - OUT_FRAC
ROPE_OUTPUT_SHIFT = OUT_FRAC + TRIG_FRAC - OUT_FRAC

Q12_12_MIN = -(1 << (IN_WIDTH - 1))
Q12_12_MAX = (1 << (IN_WIDTH - 1)) - 1


def twos_complement_hex(value: int, width_bits: int) -> str:
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    return f"{value & mask:0{hex_digits}x}"


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def unsigned_limits(width_bits: int) -> tuple[int, int]:
    return 0, (1 << width_bits) - 1


def quantize_signed(
    values: np.ndarray,
    width_bits: int,
    frac_bits: int,
) -> tuple[np.ndarray, int]:
    low, high = signed_limits(width_bits)
    scaled = np.rint(values.astype(np.float64) * float(1 << frac_bits))
    saturation_count = int(np.count_nonzero((scaled < low) | (scaled > high)))
    return np.clip(scaled, low, high).astype(np.int64), saturation_count


def quantize_unsigned(
    values: np.ndarray,
    width_bits: int,
    frac_bits: int,
) -> tuple[np.ndarray, int]:
    low, high = unsigned_limits(width_bits)
    scaled = np.rint(values.astype(np.float64) * float(1 << frac_bits))
    saturation_count = int(np.count_nonzero((scaled < low) | (scaled > high)))
    return np.clip(scaled, low, high).astype(np.int64), saturation_count


def saturate_signed(value: int, width_bits: int) -> tuple[int, bool]:
    low, high = signed_limits(width_bits)
    if value > high:
        return high, True
    if value < low:
        return low, True
    return value, False


def unpack_signed_int4_matrix(packed: np.ndarray, values_per_row: int) -> np.ndarray:
    if packed.shape[1] != values_per_row // Q4_VALUES_PER_BYTE:
        raise ValueError(
            f"Packed row width mismatch: expected {values_per_row // Q4_VALUES_PER_BYTE}, "
            f"got {packed.shape[1]}"
        )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((packed.shape[0], values_per_row), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    q = nibbles.astype(np.int16)
    q[q >= 8] -= 16
    return q.astype(np.int8)


def q26_to_q12_12_words(row_sum_q26: np.ndarray) -> np.ndarray:
    shifted = row_sum_q26.astype(np.int64) >> Q26_TO_Q12_SHIFT
    clipped = np.clip(shifted, Q12_12_MIN, Q12_12_MAX)
    return clipped.astype(np.int64)


def compute_projection_q12_12(
    *,
    activation_q12_12: np.ndarray,
    packed_weight: np.ndarray,
    scale_q2_14: np.ndarray,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    weight_q4 = unpack_signed_int4_matrix(packed_weight[:row_count], HIDDEN_SIZE)
    activation_grouped = activation_q12_12.astype(np.int64).reshape(
        GROUP_COUNT, Q4_GROUP_SIZE
    )
    weight_grouped = weight_q4.astype(np.int64).reshape(
        row_count, GROUP_COUNT, Q4_GROUP_SIZE
    )
    partial_sums = np.sum(
        weight_grouped * activation_grouped[None, :, :],
        axis=2,
        dtype=np.int64,
    )
    scaled_sums = partial_sums * scale_q2_14[:row_count].astype(np.int64)
    row_sum_q26 = np.sum(scaled_sums, axis=1, dtype=np.int64)
    return q26_to_q12_12_words(row_sum_q26), row_sum_q26


def fixed_rmsnorm_vector(
    input_q12_12: np.ndarray,
    gamma_uq8_8: np.ndarray,
) -> dict[str, Any]:
    input_int = [int(value) for value in input_q12_12.tolist()]
    gamma_int = [int(value) for value in gamma_uq8_8.tolist()]

    sum_squares = sum(value * value for value in input_int)
    mean_square = sum_squares >> int(math.log2(HEAD_DIM))
    sqrt_radicand = mean_square + EPS_Q24
    rms_q12_12 = math.isqrt(sqrt_radicand)

    if rms_q12_12 == 0:
        inv_rms_uq8_16 = (1 << INV_RMS_WIDTH) - 1
    else:
        numerator = 1 << (IN_FRAC + INV_RMS_FRAC)
        inv_rms_uq8_16 = min(numerator // rms_q12_12, (1 << INV_RMS_WIDTH) - 1)

    output: list[int] = []
    saturation = False
    for x_value, gamma_value in zip(input_int, gamma_int, strict=True):
        product = x_value * inv_rms_uq8_16 * gamma_value
        shifted = product >> RMS_OUTPUT_SHIFT
        output_value, did_saturate = saturate_signed(shifted, OUT_WIDTH)
        output.append(output_value)
        saturation = saturation or did_saturate

    return {
        "sum_squares": sum_squares,
        "mean_square": mean_square,
        "sqrt_radicand": sqrt_radicand,
        "rms_q12_12": rms_q12_12,
        "inv_rms_uq8_16": inv_rms_uq8_16,
        "output_q12_12": np.array(output, dtype=np.int64),
        "saturation": saturation,
    }


def fixed_rmsnorm_heads(
    input_heads_q12_12: np.ndarray,
    gamma_uq8_8: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    outputs = np.zeros_like(input_heads_q12_12, dtype=np.int64)
    debug: list[dict[str, Any]] = []
    saturation = False

    for head_index in range(input_heads_q12_12.shape[0]):
        result = fixed_rmsnorm_vector(input_heads_q12_12[head_index], gamma_uq8_8)
        outputs[head_index] = result["output_q12_12"]
        debug.append(result)
        saturation = saturation or bool(result["saturation"])

    return outputs, debug, saturation


def fixed_rope_reference(
    x_q12_12: np.ndarray,
    cos_q1_15: np.ndarray,
    sin_q1_15: np.ndarray,
) -> tuple[np.ndarray, bool]:
    output = np.zeros_like(x_q12_12, dtype=np.int64)
    saturation = False
    half_dim = HEAD_DIM // 2

    for head in range(x_q12_12.shape[0]):
        for dim in range(HEAD_DIM):
            if dim < half_dim:
                rotate_dim = dim + half_dim
                rotated_value = -int(x_q12_12[head, rotate_dim])
            else:
                rotate_dim = dim - half_dim
                rotated_value = int(x_q12_12[head, rotate_dim])

            rope_sum = (
                int(x_q12_12[head, dim]) * int(cos_q1_15[dim])
                + rotated_value * int(sin_q1_15[dim])
            )
            shifted = rope_sum >> ROPE_OUTPUT_SHIFT
            output_value, did_saturate = saturate_signed(shifted, OUT_WIDTH)
            output[head, dim] = output_value
            saturation = saturation or did_saturate

    return output, saturation


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [twos_complement_hex(int(value), width_bits) for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scalar(path: Path, value: int, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(twos_complement_hex(value, width_bits) + "\n", encoding="utf-8")


def read_hex_lines(path: Path, width_bits: int, *, signed: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    sign_bit = 1 << (width_bits - 1)
    full = 1 << width_bits
    values: list[int] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        raw = int(text.replace("_", ""), 16)
        if raw < 0 or raw >= full:
            raise ValueError(f"{path}:{lineno}: value out of {width_bits}-bit range: {text}")
        if signed and (raw & sign_bit):
            raw -= full
        values.append(raw)
    return np.asarray(values, dtype=np.int64)


def as_float_array(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor_to_float32(x).numpy(), dtype=np.float32)


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed-point q/k norm + RoPE RTL test vectors."
    )
    parser.add_argument("--layer-id", type=int, default=0, help="Decoder layer index")
    parser.add_argument("--input", type=Path, default=None, help="Input per-layer Q4 QKV NPZ")
    parser.add_argument("--prefix", type=str, default=None, help="Output filename prefix")
    parser.add_argument(
        "--qkv-expected-hex",
        type=Path,
        default=None,
        help="Optional Q/K/V I32_Q12_12 expected hex to use instead of recomputing projections",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default=None,
        help="Optional human-readable source label for generated metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer_id < 0:
        raise ValueError(f"layer-id must be non-negative, got {args.layer_id}")

    prefix = args.prefix if args.prefix is not None else (
        PREFIX if args.layer_id == 0 else f"layer{args.layer_id}_qk_norm_rope_stage_128_real"
    )
    module_name = f"layer{args.layer_id}.self_attn.qk_norm_rope_stage"
    q4_path = args.input if args.input is not None else (
        Q4_VECTOR_DIR / f"qkv_layer{args.layer_id}_last_token_q4.npz"
    )
    q4_path = q4_path.resolve()
    if not q4_path.is_file():
        raise FileNotFoundError(f"Missing {q4_path}. Export the requested layer Q4 vectors first.")

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    data = np.load(q4_path)
    selected_position = int(np.asarray(data["prompt_position"]).item())
    selected_token_id = int(np.asarray(data["token_id"]).item())
    if selected_position != prompt_len - 1:
        raise RuntimeError(
            f"Q4 artifact position {selected_position} does not match prompt length {prompt_len}"
        )
    if selected_token_id != int(input_ids[0, selected_position].item()):
        raise RuntimeError("Q4 artifact token id does not match tokenizer output")

    if args.layer_id >= len(backbone.layers):
        raise ValueError(f"layer-id {args.layer_id} is outside model layer range")

    layer = backbone.layers[args.layer_id]
    attn = layer.self_attn

    if args.qkv_expected_hex is None:
        activation_q12_12 = data["input_norm_q4_12"].astype(np.int64)
        q_input_flat, q_row_sum_q26 = compute_projection_q12_12(
            activation_q12_12=activation_q12_12,
            packed_weight=data["q_weight_q4_packed"],
            scale_q2_14=data["q_scale_q2_14"],
            row_count=Q_ROWS,
        )
        k_input_flat, k_row_sum_q26 = compute_projection_q12_12(
            activation_q12_12=activation_q12_12,
            packed_weight=data["k_weight_q4_packed"],
            scale_q2_14=data["k_scale_q2_14"],
            row_count=K_ROWS,
        )
        q_recompute_float = (
            q_row_sum_q26.astype(np.float64) / float(1 << (ACT_FRAC + SCALE_FRAC))
        ).astype(np.float32)
        k_recompute_float = (
            k_row_sum_q26.astype(np.float64) / float(1 << (ACT_FRAC + SCALE_FRAC))
        ).astype(np.float32)
        q_projection_diff = float(
            np.max(np.abs(q_recompute_float - data["actual_q_q4"].astype(np.float32)))
        )
        k_projection_diff = float(
            np.max(np.abs(k_recompute_float - data["actual_k_q4"].astype(np.float32)))
        )
        q_input_source = "integer QMAP/Q4 q_proj recompute, then arithmetic >> 14 to I32_Q12_12"
        k_input_source = "integer QMAP/Q4 k_proj recompute, then arithmetic >> 14 to I32_Q12_12"
    else:
        qkv_words = read_hex_lines(args.qkv_expected_hex, 32, signed=True)
        expected_words = Q_ROWS + K_ROWS + K_ROWS
        if qkv_words.shape != (expected_words,):
            raise RuntimeError(
                f"qkv expected shape mismatch: {qkv_words.shape}, expected {(expected_words,)}"
            )
        q_low, q_high = signed_limits(IN_WIDTH)
        if np.any((qkv_words < q_low) | (qkv_words > q_high)):
            raise RuntimeError(f"qkv expected values exceed signed {IN_WIDTH}-bit range")
        q_input_flat = qkv_words[:Q_ROWS]
        k_input_flat = qkv_words[Q_ROWS : Q_ROWS + K_ROWS]
        q_projection_diff = None
        k_projection_diff = None
        q_input_source = f"Q output slice from {args.qkv_expected_hex}"
        k_input_source = f"K output slice from {args.qkv_expected_hex}"

    q_input_heads = q_input_flat.reshape(NUM_Q_HEADS, HEAD_DIM)
    k_input_heads = k_input_flat.reshape(NUM_K_HEADS, HEAD_DIM)

    q_gamma_float = as_float_array(attn.q_norm.weight)
    k_gamma_float = as_float_array(attn.k_norm.weight)
    if q_gamma_float.shape != (HEAD_DIM,) or k_gamma_float.shape != (HEAD_DIM,):
        raise RuntimeError(
            f"Unexpected gamma shapes q={q_gamma_float.shape}, k={k_gamma_float.shape}"
        )
    q_gamma_q8_7, q_gamma_saturation_count = quantize_signed(
        q_gamma_float, GAMMA_WIDTH, GAMMA_FRAC
    )
    k_gamma_q8_7, k_gamma_saturation_count = quantize_signed(
        k_gamma_float, GAMMA_WIDTH, GAMMA_FRAC
    )

    q_norm_q12_12, q_norm_debug, q_norm_saturation = fixed_rmsnorm_heads(
        q_input_heads, q_gamma_q8_7
    )
    k_norm_q12_12, k_norm_debug, k_norm_saturation = fixed_rmsnorm_heads(
        k_input_heads, k_gamma_q8_7
    )
    norm_saturation = q_norm_saturation or k_norm_saturation

    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros((1, prompt_len, HIDDEN_SIZE), dtype=torch.float32)
    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, dummy_x, position_ids)
    cos_float = as_float_array(cos[0, selected_position, :])
    sin_float = as_float_array(sin[0, selected_position, :])
    cos_q1_15, cos_saturation_count = quantize_signed(cos_float, TRIG_WIDTH, TRIG_FRAC)
    sin_q1_15, sin_saturation_count = quantize_signed(sin_float, TRIG_WIDTH, TRIG_FRAC)

    q_rope_q12_12, q_rope_saturation = fixed_rope_reference(
        q_norm_q12_12, cos_q1_15, sin_q1_15
    )
    k_rope_q12_12, k_rope_saturation = fixed_rope_reference(
        k_norm_q12_12, cos_q1_15, sin_q1_15
    )
    rope_saturation = q_rope_saturation or k_rope_saturation

    files = {
        "q_input": SIM_VECTOR_DIR / f"{prefix}_q_input.hex",
        "k_input": SIM_VECTOR_DIR / f"{prefix}_k_input.hex",
        "q_gamma": SIM_VECTOR_DIR / f"{prefix}_q_gamma.hex",
        "k_gamma": SIM_VECTOR_DIR / f"{prefix}_k_gamma.hex",
        "cos": SIM_VECTOR_DIR / f"{prefix}_cos.hex",
        "sin": SIM_VECTOR_DIR / f"{prefix}_sin.hex",
        "q_norm_expected": SIM_VECTOR_DIR / f"{prefix}_q_norm_expected.hex",
        "k_norm_expected": SIM_VECTOR_DIR / f"{prefix}_k_norm_expected.hex",
        "q_rope_expected": SIM_VECTOR_DIR / f"{prefix}_q_rope_expected.hex",
        "k_rope_expected": SIM_VECTOR_DIR / f"{prefix}_k_rope_expected.hex",
        "norm_saturation": SIM_VECTOR_DIR / f"{prefix}_norm_saturation.hex",
        "rope_saturation": SIM_VECTOR_DIR / f"{prefix}_rope_saturation.hex",
        "saturation": SIM_VECTOR_DIR / f"{prefix}_saturation.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }

    write_hex_lines(files["q_input"], q_input_heads, IN_WIDTH)
    write_hex_lines(files["k_input"], k_input_heads, IN_WIDTH)
    write_hex_lines(files["q_gamma"], q_gamma_q8_7, GAMMA_WIDTH)
    write_hex_lines(files["k_gamma"], k_gamma_q8_7, GAMMA_WIDTH)
    write_hex_lines(files["cos"], cos_q1_15, TRIG_WIDTH)
    write_hex_lines(files["sin"], sin_q1_15, TRIG_WIDTH)
    write_hex_lines(files["q_norm_expected"], q_norm_q12_12, OUT_WIDTH)
    write_hex_lines(files["k_norm_expected"], k_norm_q12_12, OUT_WIDTH)
    write_hex_lines(files["q_rope_expected"], q_rope_q12_12, OUT_WIDTH)
    write_hex_lines(files["k_rope_expected"], k_rope_q12_12, OUT_WIDTH)
    write_scalar(files["norm_saturation"], 1 if norm_saturation else 0, 1)
    write_scalar(files["rope_saturation"], 1 if rope_saturation else 0, 1)
    write_scalar(files["saturation"], 1 if (norm_saturation or rope_saturation) else 0, 1)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": prefix,
        "module": module_name,
        "layer_id": args.layer_id,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "source": {
            "q4_npz": q4_path.relative_to(REPO_ROOT).as_posix(),
            "qkv_expected_hex": None
            if args.qkv_expected_hex is None
            else args.qkv_expected_hex.resolve().relative_to(REPO_ROOT).as_posix(),
            "source_label": args.source_label,
            "q_input_source": q_input_source,
            "k_input_source": k_input_source,
        },
        "formats": {
            "q_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "k_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "q_gamma": f"signed {GAMMA_WIDTH}-bit Q8.{GAMMA_FRAC}",
            "k_gamma": f"signed {GAMMA_WIDTH}-bit Q8.{GAMMA_FRAC}",
            "cos_sin": f"signed {TRIG_WIDTH}-bit Q1.{TRIG_FRAC}",
            "norm_output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
            "rope_output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
            "inv_rms": f"unsigned {INV_RMS_WIDTH}-bit UQ8.{INV_RMS_FRAC}",
        },
        "fixed_point": {
            "eps_q24": EPS_Q24,
            "rms_output_shift": RMS_OUTPUT_SHIFT,
            "rope_output_shift": ROPE_OUTPUT_SHIFT,
        },
        "saturation": {
            "q_gamma_quantize_count": q_gamma_saturation_count,
            "k_gamma_quantize_count": k_gamma_saturation_count,
            "cos_quantize_count": cos_saturation_count,
            "sin_quantize_count": sin_saturation_count,
            "norm_output": bool(norm_saturation),
            "rope_output": bool(rope_saturation),
        },
        "debug": {
            "q_projection_recompute_vs_artifact_max_abs_diff": q_projection_diff,
            "k_projection_recompute_vs_artifact_max_abs_diff": k_projection_diff,
            "q_input_q12_12_max_abs": int(np.max(np.abs(q_input_flat))),
            "k_input_q12_12_max_abs": int(np.max(np.abs(k_input_flat))),
            "q_norm_q12_12_max_abs": int(np.max(np.abs(q_norm_q12_12))),
            "k_norm_q12_12_max_abs": int(np.max(np.abs(k_norm_q12_12))),
            "q_rope_q12_12_max_abs": int(np.max(np.abs(q_rope_q12_12))),
            "k_rope_q12_12_max_abs": int(np.max(np.abs(k_rope_q12_12))),
            "q_norm_head0": {
                "sum_squares": int(q_norm_debug[0]["sum_squares"]),
                "mean_square": int(q_norm_debug[0]["mean_square"]),
                "rms_q12_12": int(q_norm_debug[0]["rms_q12_12"]),
                "inv_rms_uq8_16": int(q_norm_debug[0]["inv_rms_uq8_16"]),
            },
            "k_norm_head0": {
                "sum_squares": int(k_norm_debug[0]["sum_squares"]),
                "mean_square": int(k_norm_debug[0]["mean_square"]),
                "rms_q12_12": int(k_norm_debug[0]["rms_q12_12"]),
                "inv_rms_uq8_16": int(k_norm_debug[0]["inv_rms_uq8_16"]),
            },
            "q_rope_float_max_abs": float(
                np.max(np.abs(fixed_to_float(q_rope_q12_12, OUT_FRAC)))
            ),
            "k_rope_float_max_abs": float(
                np.max(np.abs(fixed_to_float(k_rope_q12_12, OUT_FRAC)))
            ),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point q/k norm + RoPE RTL test vectors")
    print("=" * 80)
    print(f"Module: {module_name}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Q input:          {files['q_input']}")
    print(f"K input:          {files['k_input']}")
    print(f"Q norm expected:  {files['q_norm_expected']}")
    print(f"K norm expected:  {files['k_norm_expected']}")
    print(f"Q RoPE expected:  {files['q_rope_expected']}")
    print(f"K RoPE expected:  {files['k_rope_expected']}")
    print(f"Debug meta:       {files['meta']}")
    print(f"Norm saturation:  {int(norm_saturation)}")
    print(f"RoPE saturation:  {int(rope_saturation)}")
    if q_projection_diff is None or k_projection_diff is None:
        print("Projection recompute diff: skipped for external QKV expected input")
    else:
        print(
            "Projection recompute diff: "
            f"q={q_projection_diff:.8g} "
            f"k={k_projection_diff:.8g}"
        )


if __name__ == "__main__":
    main()
