from __future__ import annotations

import argparse
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
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

DEFAULT_MLP_HIDDEN_PREFIX = "mlp_silu_mul_stage_real"
DEFAULT_PREFIX = "mlp_down_proj_stage_real"

INPUT_SIZE = 3072
OUT_FEATURES = 1024
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
ROW_ACC_WIDTH = ACT_WIDTH + WEIGHT_WIDTH + 6 + SCALE_WIDTH + 6 + 2
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point MLP down projection RTL vectors.")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--hidden-prefix", default=DEFAULT_MLP_HIDDEN_PREFIX)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_id = int(args.layer_id)
    hidden_prefix = str(args.hidden_prefix)
    prefix = str(args.prefix)
    module_name = f"layer{layer_id}.mlp.down_proj_stage"

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")
    if layer_id < 0 or layer_id >= len(backbone.layers):
        raise ValueError(f"layer_id {layer_id} is outside model layer range 0..{len(backbone.layers) - 1}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    layer = backbone.layers[layer_id]
    mlp = layer.mlp

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = mlp.down_proj.register_forward_hook(capture_module_io(records, "down_proj"))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()

    hidden_path = SIM_VECTOR_DIR / f"{hidden_prefix}_expected_hidden_q12_12.hex"
    hidden_meta_path = SIM_VECTOR_DIR / f"{hidden_prefix}_meta.json"
    activation_q12_12 = read_signed_hex_lines(hidden_path, ACT_WIDTH)
    if activation_q12_12.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected mlp_hidden vector shape {(INPUT_SIZE,)}, got {activation_q12_12.shape}")
    if hidden_meta_path.is_file():
        hidden_meta = json.loads(hidden_meta_path.read_text(encoding="utf-8"))
        if int(hidden_meta["selected_position"]) != selected_position:
            raise RuntimeError("mlp_hidden vector selected position does not match prompt")
        if int(hidden_meta["selected_token_id"]) != selected_token_id:
            raise RuntimeError("mlp_hidden vector selected token does not match prompt")
        if int(hidden_meta.get("layer_id", layer_id)) != layer_id:
            raise RuntimeError("mlp_hidden vector layer_id does not match requested layer")

    down_weight_fp32 = np.ascontiguousarray(tensor_to_float32(mlp.down_proj.weight).numpy(), dtype=np.float32)
    down_weight_q4, down_scale_q2_14 = quantize_weight_q4(down_weight_fp32)
    down_q26 = compute_q4_gemv_q26(activation_q12_12, down_weight_q4, down_scale_q2_14)
    down_q12_12, down_sat_count = q26_to_q12_12(down_q26)

    hf_down = require_output(records, "down_proj")[0, selected_position, :].numpy().astype(np.float64)
    fixed_down_float = fixed_to_float(down_q12_12, OUT_FRAC)
    down_diff = fixed_down_float - hf_down

    files = {
        "activation": SIM_VECTOR_DIR / f"{prefix}_activation.hex",
        "weight_words32": SIM_VECTOR_DIR / f"{prefix}_weight_words32.hex",
        "scale_words32": SIM_VECTOR_DIR / f"{prefix}_scale_words32.hex",
        "weight_chunks4096": SIM_VECTOR_DIR / f"{prefix}_weight_chunks4096.hex",
        "scale_chunks4096": SIM_VECTOR_DIR / f"{prefix}_scale_chunks4096.hex",
        "expected_q26": SIM_VECTOR_DIR / f"{prefix}_expected_q26.hex",
        "expected_q12_12": SIM_VECTOR_DIR / f"{prefix}_expected_q12_12.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }

    write_hex_lines(files["activation"], activation_q12_12, ACT_WIDTH)
    write_hex_lines(files["weight_words32"], pack_words(down_weight_q4, WEIGHT_WIDTH, 32), 32)
    write_hex_lines(files["scale_words32"], pack_words(down_scale_q2_14, SCALE_WIDTH, 32), 32)
    write_hex_lines(files["weight_chunks4096"], pack_words(down_weight_q4, WEIGHT_WIDTH, 4096), 4096)
    write_hex_lines(files["scale_chunks4096"], pack_words(down_scale_q2_14, SCALE_WIDTH, 4096), 4096)
    write_hex_lines(files["expected_q26"], down_q26, ROW_ACC_WIDTH)
    write_hex_lines(files["expected_q12_12"], down_q12_12, OUT_WIDTH)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": prefix,
        "module": module_name,
        "layer_id": layer_id,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "source_mlp_hidden_prefix": hidden_prefix,
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
            "down_output_count": int(down_sat_count),
        },
        "debug": {
            "activation_q12_12_min": int(np.min(activation_q12_12)),
            "activation_q12_12_max": int(np.max(activation_q12_12)),
            "activation_q12_12_max_abs": int(np.max(np.abs(activation_q12_12))),
            "down_q26_min": int(np.min(down_q26)),
            "down_q26_max": int(np.max(down_q26)),
            "down_q12_12_min": int(np.min(down_q12_12)),
            "down_q12_12_max": int(np.max(down_q12_12)),
            "down_q12_12_max_abs": int(np.max(np.abs(down_q12_12))),
            "fixed_down_vs_hf_max_abs_diff": float(np.max(np.abs(down_diff))),
            "fixed_down_vs_hf_mean_abs_diff": float(np.mean(np.abs(down_diff))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point MLP down projection RTL test vectors")
    print("=" * 80)
    print(f"Module: {module_name}")
    print(f"Layer: {layer_id}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Activation: {files['activation']}")
    print(f"Expected:   {files['expected_q12_12']}")
    print(f"Debug meta: {files['meta']}")
    print(
        "Fixed down vs HF: "
        f"max={meta['debug']['fixed_down_vs_hf_max_abs_diff']:.8g} "
        f"mean={meta['debug']['fixed_down_vs_hf_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
