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
    load_model,
    require_input0,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
POST_NORM_PREFIX = "post_attention_residual_norm_stage_real"

PREFIX = "mlp_gate_up_proj_stage_real"
MODULE_NAME = "layer0.mlp.gate_up_proj_stage"

INPUT_SIZE = 1024
OUT_FEATURES = 3072
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
Q4_MIN = -8
Q4_MAX = 7

ACT_WIDTH = 24
ACT_FRAC = 12
WEIGHT_WIDTH = 4
SCALE_WIDTH = 16
SCALE_FRAC = 14
OUT_WIDTH = 24
OUT_FRAC = 12
ROW_ACC_WIDTH = 56
Q26_TO_Q12_SHIFT = SCALE_FRAC


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    lines = [f"{int(value) & mask:0{hex_digits}x}" for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_signed_hex_lines(path: Path, width_bits: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    sign_bit = 1 << (width_bits - 1)
    full = 1 << width_bits
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw = int(stripped, 16)
        values.append(raw - full if (raw & sign_bit) else raw)
    return np.array(values, dtype=np.int64)


def pack_words(values: np.ndarray, width_bits: int, word_bits: int) -> np.ndarray:
    if width_bits <= 0 or width_bits > word_bits or (word_bits % width_bits) != 0:
        raise ValueError(f"width_bits must divide word_bits, got {width_bits}, {word_bits}")

    flat = values.reshape(-1).astype(np.int64)
    values_per_word = word_bits // width_bits
    padded_count = ((flat.size + values_per_word - 1) // values_per_word) * values_per_word
    mask = (1 << width_bits) - 1
    padded = np.zeros(padded_count, dtype=np.uint64)
    padded[: flat.size] = np.bitwise_and(flat, mask).astype(np.uint64)
    grouped = padded.reshape(-1, values_per_word)
    words: list[int] = []
    for row in grouped:
        word = 0
        for index, value in enumerate(row.tolist()):
            word |= int(value) << (index * width_bits)
        words.append(word)
    return np.array(words, dtype=object)


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def saturate_signed_array(values: np.ndarray, width_bits: int) -> tuple[np.ndarray, int]:
    low, high = signed_limits(width_bits)
    saturation_count = int(np.count_nonzero((values < low) | (values > high)))
    return np.clip(values, low, high).astype(np.int64), saturation_count


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def quantize_weight_q4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.ascontiguousarray(weight.astype(np.float32))
    if weight.shape != (OUT_FEATURES, INPUT_SIZE):
        raise ValueError(f"Expected weight shape {(OUT_FEATURES, INPUT_SIZE)}, got {weight.shape}")

    grouped = weight.reshape(OUT_FEATURES, GROUP_COUNT, GROUP_SIZE)
    absmax = np.max(np.abs(grouped), axis=2)
    raw_scale = absmax / float(Q4_MAX)
    scale_q64 = np.rint(raw_scale * float(1 << SCALE_FRAC)).astype(np.int64)
    scale_q64 = np.where((absmax > 0.0) & (scale_q64 == 0), 1, scale_q64)
    if np.any(scale_q64 < 0) or np.any(scale_q64 > np.iinfo(np.uint16).max):
        raise ValueError("Q4 scale does not fit unsigned Q2.14")

    scale_q2_14 = scale_q64.astype(np.uint16)
    scale_used = scale_q2_14.astype(np.float32) / float(1 << SCALE_FRAC)
    safe_scale = np.where(scale_used > 0.0, scale_used, 1.0).astype(np.float32)
    q_grouped = np.rint(grouped / safe_scale[:, :, None])
    q = np.clip(q_grouped, Q4_MIN, Q4_MAX).astype(np.int8)
    return q.reshape(OUT_FEATURES, INPUT_SIZE), scale_q2_14


def compute_q4_gemv_q26(
    activation_q12_12: np.ndarray,
    weight_q4: np.ndarray,
    scale_q2_14: np.ndarray,
) -> np.ndarray:
    x_grouped = activation_q12_12.astype(np.int64).reshape(GROUP_COUNT, GROUP_SIZE)
    w_grouped = weight_q4.astype(np.int64).reshape(OUT_FEATURES, GROUP_COUNT, GROUP_SIZE)
    partial_sums = np.sum(w_grouped * x_grouped[None, :, :], axis=2, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14.astype(np.int64)
    return np.sum(scaled_sums, axis=1, dtype=np.int64)


def q26_to_q12_12(row_sum_q26: np.ndarray) -> tuple[np.ndarray, int]:
    shifted = row_sum_q26.astype(np.int64) >> Q26_TO_Q12_SHIFT
    return saturate_signed_array(shifted, OUT_WIDTH)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    layer0 = backbone.layers[0]
    mlp0 = layer0.mlp

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.post_attention_layernorm.register_forward_hook(capture_module_io(records, "post_norm")),
        mlp0.gate_proj.register_forward_hook(capture_module_io(records, "gate_proj")),
        mlp0.up_proj.register_forward_hook(capture_module_io(records, "up_proj")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    post_norm_path = SIM_VECTOR_DIR / f"{POST_NORM_PREFIX}_expected_norm.hex"
    post_norm_meta_path = SIM_VECTOR_DIR / f"{POST_NORM_PREFIX}_meta.json"
    activation_q12_12 = read_signed_hex_lines(post_norm_path, ACT_WIDTH)
    if activation_q12_12.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected post-norm vector shape {(INPUT_SIZE,)}, got {activation_q12_12.shape}")
    if post_norm_meta_path.is_file():
        post_norm_meta = json.loads(post_norm_meta_path.read_text(encoding="utf-8"))
        if int(post_norm_meta["selected_position"]) != selected_position:
            raise RuntimeError("post-norm vector selected position does not match prompt")
        if int(post_norm_meta["selected_token_id"]) != selected_token_id:
            raise RuntimeError("post-norm vector selected token does not match prompt")

    gate_weight_fp32 = np.ascontiguousarray(tensor_to_float32(mlp0.gate_proj.weight).numpy(), dtype=np.float32)
    up_weight_fp32 = np.ascontiguousarray(tensor_to_float32(mlp0.up_proj.weight).numpy(), dtype=np.float32)
    gate_weight_q4, gate_scale_q2_14 = quantize_weight_q4(gate_weight_fp32)
    up_weight_q4, up_scale_q2_14 = quantize_weight_q4(up_weight_fp32)

    gate_q26 = compute_q4_gemv_q26(activation_q12_12, gate_weight_q4, gate_scale_q2_14)
    up_q26 = compute_q4_gemv_q26(activation_q12_12, up_weight_q4, up_scale_q2_14)
    gate_q12_12, gate_sat_count = q26_to_q12_12(gate_q26)
    up_q12_12, up_sat_count = q26_to_q12_12(up_q26)

    hf_post_norm = require_input0(records, "gate_proj")[0, selected_position, :].numpy()
    hf_gate = require_output(records, "gate_proj")[0, selected_position, :].numpy()
    hf_up = require_output(records, "up_proj")[0, selected_position, :].numpy()
    fixed_input_diff = fixed_to_float(activation_q12_12, ACT_FRAC) - hf_post_norm.astype(np.float64)
    gate_diff = fixed_to_float(gate_q12_12, OUT_FRAC) - hf_gate.astype(np.float64)
    up_diff = fixed_to_float(up_q12_12, OUT_FRAC) - hf_up.astype(np.float64)

    files = {
        "activation": SIM_VECTOR_DIR / f"{PREFIX}_activation.hex",
        "gate_weight_words32": SIM_VECTOR_DIR / f"{PREFIX}_gate_weight_words32.hex",
        "gate_scale_words32": SIM_VECTOR_DIR / f"{PREFIX}_gate_scale_words32.hex",
        "gate_weight_chunks4096": SIM_VECTOR_DIR / f"{PREFIX}_gate_weight_chunks4096.hex",
        "gate_scale_chunks4096": SIM_VECTOR_DIR / f"{PREFIX}_gate_scale_chunks4096.hex",
        "up_weight_words32": SIM_VECTOR_DIR / f"{PREFIX}_up_weight_words32.hex",
        "up_scale_words32": SIM_VECTOR_DIR / f"{PREFIX}_up_scale_words32.hex",
        "up_weight_chunks4096": SIM_VECTOR_DIR / f"{PREFIX}_up_weight_chunks4096.hex",
        "up_scale_chunks4096": SIM_VECTOR_DIR / f"{PREFIX}_up_scale_chunks4096.hex",
        "gate_expected_q26": SIM_VECTOR_DIR / f"{PREFIX}_gate_expected_q26.hex",
        "gate_expected_q12_12": SIM_VECTOR_DIR / f"{PREFIX}_gate_expected_q12_12.hex",
        "up_expected_q26": SIM_VECTOR_DIR / f"{PREFIX}_up_expected_q26.hex",
        "up_expected_q12_12": SIM_VECTOR_DIR / f"{PREFIX}_up_expected_q12_12.hex",
        "meta": SIM_VECTOR_DIR / f"{PREFIX}_meta.json",
    }

    write_hex_lines(files["activation"], activation_q12_12, ACT_WIDTH)
    write_hex_lines(files["gate_weight_words32"], pack_words(gate_weight_q4, WEIGHT_WIDTH, 32), 32)
    write_hex_lines(files["gate_scale_words32"], pack_words(gate_scale_q2_14, SCALE_WIDTH, 32), 32)
    write_hex_lines(files["gate_weight_chunks4096"], pack_words(gate_weight_q4, WEIGHT_WIDTH, 4096), 4096)
    write_hex_lines(files["gate_scale_chunks4096"], pack_words(gate_scale_q2_14, SCALE_WIDTH, 4096), 4096)
    write_hex_lines(files["up_weight_words32"], pack_words(up_weight_q4, WEIGHT_WIDTH, 32), 32)
    write_hex_lines(files["up_scale_words32"], pack_words(up_scale_q2_14, SCALE_WIDTH, 32), 32)
    write_hex_lines(files["up_weight_chunks4096"], pack_words(up_weight_q4, WEIGHT_WIDTH, 4096), 4096)
    write_hex_lines(files["up_scale_chunks4096"], pack_words(up_scale_q2_14, SCALE_WIDTH, 4096), 4096)
    write_hex_lines(files["gate_expected_q26"], gate_q26, ROW_ACC_WIDTH)
    write_hex_lines(files["gate_expected_q12_12"], gate_q12_12, OUT_WIDTH)
    write_hex_lines(files["up_expected_q26"], up_q26, ROW_ACC_WIDTH)
    write_hex_lines(files["up_expected_q12_12"], up_q12_12, OUT_WIDTH)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": PREFIX,
        "module": MODULE_NAME,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "source_post_norm_prefix": POST_NORM_PREFIX,
        "shape": {
            "input_size": INPUT_SIZE,
            "out_features": OUT_FEATURES,
            "group_size": GROUP_SIZE,
            "group_count": GROUP_COUNT,
        },
        "formats": {
            "activation": f"signed {ACT_WIDTH}-bit Q12.{ACT_FRAC}",
            "weight_words32": "signed int4 packed into 32-bit words, 8 weights per word, lowest index in low nibble",
            "scale_words32": f"unsigned {SCALE_WIDTH}-bit Q2.{SCALE_FRAC}, 2 scales per 32-bit word",
            "chunks4096": "simulation-optimized 4096-bit chunks, lowest flattened index in low bits",
            "row_sum": f"signed {ROW_ACC_WIDTH}-bit Q26",
            "output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
        },
        "saturation": {
            "gate_output_count": int(gate_sat_count),
            "up_output_count": int(up_sat_count),
        },
        "debug": {
            "post_norm_q12_12_max_abs": int(np.max(np.abs(activation_q12_12))),
            "fixed_input_vs_hf_post_norm_max_abs_diff": float(np.max(np.abs(fixed_input_diff))),
            "gate_q26_min": int(np.min(gate_q26)),
            "gate_q26_max": int(np.max(gate_q26)),
            "up_q26_min": int(np.min(up_q26)),
            "up_q26_max": int(np.max(up_q26)),
            "gate_q12_12_max_abs": int(np.max(np.abs(gate_q12_12))),
            "up_q12_12_max_abs": int(np.max(np.abs(up_q12_12))),
            "fixed_gate_vs_hf_max_abs_diff": float(np.max(np.abs(gate_diff))),
            "fixed_gate_vs_hf_mean_abs_diff": float(np.mean(np.abs(gate_diff))),
            "fixed_up_vs_hf_max_abs_diff": float(np.max(np.abs(up_diff))),
            "fixed_up_vs_hf_mean_abs_diff": float(np.mean(np.abs(up_diff))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point MLP gate/up Q4 projection RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Activation: {files['activation']}")
    print(f"Gate expected: {files['gate_expected_q12_12']}")
    print(f"Up expected:   {files['up_expected_q12_12']}")
    print(f"Debug meta:    {files['meta']}")
    print(
        "Fixed gate/up vs HF: "
        f"gate_max={meta['debug']['fixed_gate_vs_hf_max_abs_diff']:.8g} "
        f"gate_mean={meta['debug']['fixed_gate_vs_hf_mean_abs_diff']:.8g} "
        f"up_max={meta['debug']['fixed_up_vs_hf_max_abs_diff']:.8g} "
        f"up_mean={meta['debug']['fixed_up_vs_hf_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
