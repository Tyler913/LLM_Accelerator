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
    capture_module_io,
    encode_prompt,
    load_model,
    require_input0,
)
from vector_workspace import resolve_sim_vector_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = resolve_sim_vector_dir(REPO_ROOT)
DEFAULT_GATE_UP_PREFIX = "mlp_gate_up_proj_stage_real"
DEFAULT_PREFIX = "mlp_silu_mul_stage_real"

FEATURES = 3072
IN_WIDTH = 24
IN_FRAC = 12
SIGMOID_WIDTH = 16
SIGMOID_FRAC = 16
SIGMOID_LUT_INDEX_FRAC = 6
SIGMOID_LUT_MIN_REAL = -8.0
SIGMOID_LUT_MAX_REAL = 8.0
SIGMOID_LUT_SIZE = int((SIGMOID_LUT_MAX_REAL - SIGMOID_LUT_MIN_REAL) * (1 << SIGMOID_LUT_INDEX_FRAC)) + 1
OUT_WIDTH = 24
OUT_FRAC = 12


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


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def saturate_signed_array(values: np.ndarray, width_bits: int) -> tuple[np.ndarray, int]:
    low, high = signed_limits(width_bits)
    saturation_count = int(np.count_nonzero((values < low) | (values > high)))
    return np.clip(values, low, high).astype(np.int64), saturation_count


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def build_sigmoid_lut() -> np.ndarray:
    values: list[int] = []
    for index in range(SIGMOID_LUT_SIZE):
        x = SIGMOID_LUT_MIN_REAL + float(index) / float(1 << SIGMOID_LUT_INDEX_FRAC)
        y = 1.0 / (1.0 + math.exp(-x))
        values.append(int(np.clip(round(y * float(1 << SIGMOID_FRAC)), 0, (1 << SIGMOID_WIDTH) - 1)))
    return np.array(values, dtype=np.int64)


def gate_to_lut_index(gate_q12_12: np.ndarray) -> np.ndarray:
    min_q = int(round(SIGMOID_LUT_MIN_REAL * float(1 << IN_FRAC)))
    max_q = int(round(SIGMOID_LUT_MAX_REAL * float(1 << IN_FRAC)))
    shift = IN_FRAC - SIGMOID_LUT_INDEX_FRAC
    half_step = 1 << (shift - 1)
    clipped = np.clip(gate_q12_12.astype(np.int64), min_q, max_q)
    rounded = (clipped - min_q + half_step) >> shift
    return np.clip(rounded, 0, SIGMOID_LUT_SIZE - 1).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point MLP SiLU/multiply RTL vectors.")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--gate-up-prefix", default=DEFAULT_GATE_UP_PREFIX)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_id = int(args.layer_id)
    gate_up_prefix = str(args.gate_up_prefix)
    prefix = str(args.prefix)
    module_name = f"layer{layer_id}.mlp.silu_mul_stage"

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    if layer_id < 0 or layer_id >= len(backbone.layers):
        raise ValueError(f"layer_id {layer_id} is outside model layer range 0..{len(backbone.layers) - 1}")

    layer = backbone.layers[layer_id]
    mlp = layer.mlp

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        mlp.down_proj.register_forward_hook(capture_module_io(records, "down_proj")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    gate_path = SIM_VECTOR_DIR / f"{gate_up_prefix}_gate_expected_q12_12.hex"
    up_path = SIM_VECTOR_DIR / f"{gate_up_prefix}_up_expected_q12_12.hex"
    gate_up_meta_path = SIM_VECTOR_DIR / f"{gate_up_prefix}_meta.json"
    gate_q12_12 = read_signed_hex_lines(gate_path, IN_WIDTH)
    up_q12_12 = read_signed_hex_lines(up_path, IN_WIDTH)
    if gate_q12_12.shape != (FEATURES,) or up_q12_12.shape != (FEATURES,):
        raise RuntimeError(
            f"Expected gate/up shape {(FEATURES,)}, got {gate_q12_12.shape} and {up_q12_12.shape}"
        )
    if gate_up_meta_path.is_file():
        gate_up_meta = json.loads(gate_up_meta_path.read_text(encoding="utf-8"))
        if int(gate_up_meta["selected_position"]) != selected_position:
            raise RuntimeError("gate/up vector selected position does not match prompt")
        if int(gate_up_meta["selected_token_id"]) != selected_token_id:
            raise RuntimeError("gate/up vector selected token does not match prompt")
        if int(gate_up_meta.get("layer_id", layer_id)) != layer_id:
            raise RuntimeError("gate/up vector layer_id does not match requested layer")

    sigmoid_lut = build_sigmoid_lut()
    lut_index = gate_to_lut_index(gate_q12_12)
    sigmoid_q0_16 = sigmoid_lut[lut_index]

    silu_product = gate_q12_12.astype(np.int64) * sigmoid_q0_16.astype(np.int64)
    silu_gate_q12_12, silu_sat_count = saturate_signed_array(silu_product >> SIGMOID_FRAC, OUT_WIDTH)

    hidden_product = silu_gate_q12_12.astype(np.int64) * up_q12_12.astype(np.int64)
    hidden_q12_12, hidden_sat_count = saturate_signed_array(hidden_product >> IN_FRAC, OUT_WIDTH)

    hf_hidden = require_input0(records, "down_proj")[0, selected_position, :].numpy().astype(np.float64)
    fixed_hidden_float = fixed_to_float(hidden_q12_12, OUT_FRAC)
    ideal_fixed_hidden = (
        (fixed_to_float(gate_q12_12, IN_FRAC) / (1.0 + np.exp(-fixed_to_float(gate_q12_12, IN_FRAC))))
        * fixed_to_float(up_q12_12, IN_FRAC)
    )
    hidden_vs_hf = fixed_hidden_float - hf_hidden
    hidden_vs_ideal = fixed_hidden_float - ideal_fixed_hidden

    files = {
        "gate_input": SIM_VECTOR_DIR / f"{prefix}_gate_input.hex",
        "up_input": SIM_VECTOR_DIR / f"{prefix}_up_input.hex",
        "sigmoid_lut": SIM_VECTOR_DIR / f"{prefix}_sigmoid_lut.hex",
        "expected_lut_index": SIM_VECTOR_DIR / f"{prefix}_expected_lut_index.hex",
        "expected_sigmoid_q0_16": SIM_VECTOR_DIR / f"{prefix}_expected_sigmoid_q0_16.hex",
        "expected_silu_gate_q12_12": SIM_VECTOR_DIR / f"{prefix}_expected_silu_gate_q12_12.hex",
        "expected_hidden_q12_12": SIM_VECTOR_DIR / f"{prefix}_expected_hidden_q12_12.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }

    write_hex_lines(files["gate_input"], gate_q12_12, IN_WIDTH)
    write_hex_lines(files["up_input"], up_q12_12, IN_WIDTH)
    write_hex_lines(files["sigmoid_lut"], sigmoid_lut, SIGMOID_WIDTH)
    write_hex_lines(files["expected_lut_index"], lut_index, 16)
    write_hex_lines(files["expected_sigmoid_q0_16"], sigmoid_q0_16, SIGMOID_WIDTH)
    write_hex_lines(files["expected_silu_gate_q12_12"], silu_gate_q12_12, OUT_WIDTH)
    write_hex_lines(files["expected_hidden_q12_12"], hidden_q12_12, OUT_WIDTH)

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
        "source_gate_up_prefix": gate_up_prefix,
        "shape": {
            "features": FEATURES,
        },
        "formats": {
            "gate_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "up_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "sigmoid_lut": f"unsigned {SIGMOID_WIDTH}-bit UQ0.{SIGMOID_FRAC}",
            "sigmoid_lut_range": f"[{SIGMOID_LUT_MIN_REAL}, {SIGMOID_LUT_MAX_REAL}]",
            "sigmoid_lut_step": f"1/{1 << SIGMOID_LUT_INDEX_FRAC}",
            "sigmoid_lut_size": SIGMOID_LUT_SIZE,
            "silu_gate": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
            "hidden": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
        },
        "saturation": {
            "silu_gate_count": int(silu_sat_count),
            "hidden_count": int(hidden_sat_count),
        },
        "debug": {
            "gate_q12_12_min": int(np.min(gate_q12_12)),
            "gate_q12_12_max": int(np.max(gate_q12_12)),
            "up_q12_12_min": int(np.min(up_q12_12)),
            "up_q12_12_max": int(np.max(up_q12_12)),
            "lut_index_min": int(np.min(lut_index)),
            "lut_index_max": int(np.max(lut_index)),
            "sigmoid_q0_16_min": int(np.min(sigmoid_q0_16)),
            "sigmoid_q0_16_max": int(np.max(sigmoid_q0_16)),
            "silu_gate_q12_12_min": int(np.min(silu_gate_q12_12)),
            "silu_gate_q12_12_max": int(np.max(silu_gate_q12_12)),
            "hidden_q12_12_min": int(np.min(hidden_q12_12)),
            "hidden_q12_12_max": int(np.max(hidden_q12_12)),
            "fixed_hidden_vs_hf_max_abs_diff": float(np.max(np.abs(hidden_vs_hf))),
            "fixed_hidden_vs_hf_mean_abs_diff": float(np.mean(np.abs(hidden_vs_hf))),
            "fixed_hidden_vs_ideal_fixed_silu_max_abs_diff": float(np.max(np.abs(hidden_vs_ideal))),
            "fixed_hidden_vs_ideal_fixed_silu_mean_abs_diff": float(np.mean(np.abs(hidden_vs_ideal))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point MLP SiLU/multiply RTL test vectors")
    print("=" * 80)
    print(f"Module: {module_name}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Gate input:      {files['gate_input']}")
    print(f"Up input:        {files['up_input']}")
    print(f"Sigmoid LUT:     {files['sigmoid_lut']}")
    print(f"Hidden expected: {files['expected_hidden_q12_12']}")
    print(f"Debug meta:      {files['meta']}")
    print(
        "Fixed hidden diff: "
        f"hf_max={meta['debug']['fixed_hidden_vs_hf_max_abs_diff']:.8g} "
        f"hf_mean={meta['debug']['fixed_hidden_vs_hf_mean_abs_diff']:.8g} "
        f"ideal_fixed_max={meta['debug']['fixed_hidden_vs_ideal_fixed_silu_max_abs_diff']:.8g} "
        f"ideal_fixed_mean={meta['debug']['fixed_hidden_vs_ideal_fixed_silu_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
