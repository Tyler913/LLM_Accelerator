from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    PROMPT,
    capture_module_io,
    encode_prompt,
    get_rms_norm_eps,
    load_model,
    manual_apply_rope,
    manual_linear_from_weight,
    manual_rms_norm,
    manual_rope_cos_sin,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

PREFIX = "rope_qk_layer_128_real"
MODULE_NAME = "layer0.self_attn.rope_qk"

NUM_Q_HEADS = 16
NUM_K_HEADS = 8
HEAD_DIM = 128

IN_WIDTH = 24
IN_FRAC = 12
TRIG_WIDTH = 16
TRIG_FRAC = 15
OUT_WIDTH = 24
OUT_FRAC = 12
OUTPUT_SHIFT = IN_FRAC + TRIG_FRAC - OUT_FRAC


def twos_complement_hex(value: int, width_bits: int) -> str:
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    return f"{value & mask:0{hex_digits}x}"


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def quantize_signed(
    values: np.ndarray,
    width_bits: int,
    frac_bits: int,
) -> tuple[np.ndarray, int]:
    low, high = signed_limits(width_bits)
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


def fixed_rope_reference(
    x_q12_12: np.ndarray,
    cos_q1_15: np.ndarray,
    sin_q1_15: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if x_q12_12.ndim != 2 or x_q12_12.shape[1] != HEAD_DIM:
        raise RuntimeError(f"Expected [heads, {HEAD_DIM}] input, got {x_q12_12.shape}")

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
            shifted = rope_sum >> OUTPUT_SHIFT
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


def as_float_array(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor_to_float32(x).numpy(), dtype=np.float32)


def max_abs_diff_fixed_vs_float(fixed: np.ndarray, frac_bits: int, reference: np.ndarray) -> float:
    fixed_float = fixed.astype(np.float64) / float(1 << frac_bits)
    return float(np.max(np.abs(fixed_float - reference.astype(np.float64))))


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)

    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn
    hidden_size = int(model.config.hidden_size)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))

    if (num_q_heads, num_kv_heads, head_dim) != (NUM_Q_HEADS, NUM_K_HEADS, HEAD_DIM):
        raise RuntimeError(
            "Unexpected attention shape: "
            f"q_heads={num_q_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = layer0.input_layernorm.register_forward_hook(
        capture_module_io(records, "layer0_input_norm")
    )
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=True)
    finally:
        hook.remove()

    input_norm = require_output(records, "layer0_input_norm")
    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)

    q_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.q_proj.weight))
    k_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.k_proj.weight))
    q_view = q_flat.view(batch_size, prompt_len, NUM_Q_HEADS, HEAD_DIM)
    k_view = k_flat.view(batch_size, prompt_len, NUM_K_HEADS, HEAD_DIM)

    q_norm = manual_rms_norm(q_view, tensor_to_float32(attn0.q_norm.weight), rms_eps)
    k_norm = manual_rms_norm(k_view, tensor_to_float32(attn0.k_norm.weight), rms_eps)
    q_states = q_norm.transpose(1, 2)
    k_states = k_norm.transpose(1, 2)

    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, input_norm, position_ids)
    q_rope_float, k_rope_float = manual_apply_rope(q_states, k_states, cos, sin)

    q_input_float = as_float_array(q_states[0, :, selected_position, :])
    k_input_float = as_float_array(k_states[0, :, selected_position, :])
    cos_float = as_float_array(cos[0, selected_position, :])
    sin_float = as_float_array(sin[0, selected_position, :])
    q_expected_float = as_float_array(q_rope_float[0, :, selected_position, :])
    k_expected_float = as_float_array(k_rope_float[0, :, selected_position, :])

    q_input_q12_12, q_input_saturation_count = quantize_signed(
        q_input_float, IN_WIDTH, IN_FRAC
    )
    k_input_q12_12, k_input_saturation_count = quantize_signed(
        k_input_float, IN_WIDTH, IN_FRAC
    )
    cos_q1_15, cos_saturation_count = quantize_signed(cos_float, TRIG_WIDTH, TRIG_FRAC)
    sin_q1_15, sin_saturation_count = quantize_signed(sin_float, TRIG_WIDTH, TRIG_FRAC)

    q_output_q12_12, q_output_saturation = fixed_rope_reference(
        q_input_q12_12, cos_q1_15, sin_q1_15
    )
    k_output_q12_12, k_output_saturation = fixed_rope_reference(
        k_input_q12_12, cos_q1_15, sin_q1_15
    )
    output_saturation = q_output_saturation or k_output_saturation

    files = {
        "q_input": SIM_VECTOR_DIR / f"{PREFIX}_q_input.hex",
        "k_input": SIM_VECTOR_DIR / f"{PREFIX}_k_input.hex",
        "cos": SIM_VECTOR_DIR / f"{PREFIX}_cos.hex",
        "sin": SIM_VECTOR_DIR / f"{PREFIX}_sin.hex",
        "q_expected": SIM_VECTOR_DIR / f"{PREFIX}_q_expected.hex",
        "k_expected": SIM_VECTOR_DIR / f"{PREFIX}_k_expected.hex",
        "saturation": SIM_VECTOR_DIR / f"{PREFIX}_saturation.hex",
        "meta": SIM_VECTOR_DIR / f"{PREFIX}_meta.json",
    }

    write_hex_lines(files["q_input"], q_input_q12_12, IN_WIDTH)
    write_hex_lines(files["k_input"], k_input_q12_12, IN_WIDTH)
    write_hex_lines(files["cos"], cos_q1_15, TRIG_WIDTH)
    write_hex_lines(files["sin"], sin_q1_15, TRIG_WIDTH)
    write_hex_lines(files["q_expected"], q_output_q12_12, OUT_WIDTH)
    write_hex_lines(files["k_expected"], k_output_q12_12, OUT_WIDTH)
    write_scalar(files["saturation"], 1 if output_saturation else 0, 1)

    q_fixed_float_max_abs_diff = max_abs_diff_fixed_vs_float(
        q_output_q12_12, OUT_FRAC, q_expected_float
    )
    k_fixed_float_max_abs_diff = max_abs_diff_fixed_vs_float(
        k_output_q12_12, OUT_FRAC, k_expected_float
    )

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": PREFIX,
        "module": MODULE_NAME,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "formats": {
            "q_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "k_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "cos_sin": f"signed {TRIG_WIDTH}-bit Q1.{TRIG_FRAC}",
            "expected_output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
        },
        "saturation": {
            "q_input_quantize_count": q_input_saturation_count,
            "k_input_quantize_count": k_input_saturation_count,
            "cos_quantize_count": cos_saturation_count,
            "sin_quantize_count": sin_saturation_count,
            "output": bool(output_saturation),
        },
        "debug": {
            "q_fixed_vs_fp32_rope_max_abs_diff": q_fixed_float_max_abs_diff,
            "k_fixed_vs_fp32_rope_max_abs_diff": k_fixed_float_max_abs_diff,
            "q_input_float_max_abs": float(np.max(np.abs(q_input_float))),
            "k_input_float_max_abs": float(np.max(np.abs(k_input_float))),
            "cos_float_min": float(np.min(cos_float)),
            "cos_float_max": float(np.max(cos_float)),
            "sin_float_min": float(np.min(sin_float)),
            "sin_float_max": float(np.max(sin_float)),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point RoPE RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Q input:     {files['q_input']}")
    print(f"K input:     {files['k_input']}")
    print(f"Cos:         {files['cos']}")
    print(f"Sin:         {files['sin']}")
    print(f"Q expected:  {files['q_expected']}")
    print(f"K expected:  {files['k_expected']}")
    print(f"Debug meta:  {files['meta']}")
    print(f"Output saturation: {int(output_saturation)}")
    print(f"Q fixed output vs FP32 RoPE max abs diff: {q_fixed_float_max_abs_diff:.8g}")
    print(f"K fixed output vs FP32 RoPE max abs diff: {k_fixed_float_max_abs_diff:.8g}")


if __name__ == "__main__":
    main()
