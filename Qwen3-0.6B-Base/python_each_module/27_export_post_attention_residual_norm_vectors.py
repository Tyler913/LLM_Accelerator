from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from common import (
    PROMPT,
    capture_module_io,
    encode_prompt,
    get_rms_norm_eps,
    load_model,
    require_input0,
    require_output,
    tensor_to_float32,
)
from vector_workspace import resolve_sim_vector_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = resolve_sim_vector_dir(REPO_ROOT)
RMSNORM_EXPORTER_PATH = Path(__file__).with_name("17_export_rmsnorm_fixed_vectors.py")

DEFAULT_O_PROJ_PREFIX = "o_proj_stage_real"
DEFAULT_PREFIX = "post_attention_residual_norm_stage_real"

INPUT_SIZE = 1024
RESIDUAL_WIDTH = 24
RESIDUAL_FRAC = 10
O_PROJ_WIDTH = 24
O_PROJ_FRAC = 12
GAMMA_WIDTH = 16
GAMMA_FRAC = 7
NORM_OUT_WIDTH = 24
NORM_OUT_FRAC = 12
SUM_WIDTH = 64
INV_RMS_WIDTH = 24
INV_RMS_FRAC = 16
MEAN_SHIFT = 10
EPS_Q20 = 1
DIV_NUM_SHIFT = RESIDUAL_FRAC + INV_RMS_FRAC
NORM_OUTPUT_SHIFT = RESIDUAL_FRAC + INV_RMS_FRAC + GAMMA_FRAC - NORM_OUT_FRAC


def load_rmsnorm_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rmsnorm_fixed_exporter",
        RMSNORM_EXPORTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {RMSNORM_EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    lines = [f"{int(value) & mask:0{hex_digits}x}" for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scalar(path: Path, value: int, width_bits: int) -> None:
    write_hex_lines(path, np.array([value], dtype=np.int64), width_bits)


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


def saturate_signed(value: int, width_bits: int) -> tuple[int, bool]:
    low, high = signed_limits(width_bits)
    if value > high:
        return high, True
    if value < low:
        return low, True
    return value, False


def fixed_rmsnorm_reference(
    input_q14_10: np.ndarray,
    gamma_q8_7: np.ndarray,
) -> dict[str, Any]:
    input_int = [int(value) for value in input_q14_10.tolist()]
    gamma_int = [int(value) for value in gamma_q8_7.tolist()]

    sum_squares = sum(value * value for value in input_int)
    mean_square = sum_squares >> MEAN_SHIFT
    sqrt_radicand = mean_square + EPS_Q20
    rms_q14_10 = math.isqrt(sqrt_radicand)

    if rms_q14_10 == 0:
        inv_rms_uq8_16 = (1 << INV_RMS_WIDTH) - 1
        div_remainder = 0
    else:
        numerator = 1 << DIV_NUM_SHIFT
        raw_quotient = numerator // rms_q14_10
        inv_rms_uq8_16 = min(raw_quotient, (1 << INV_RMS_WIDTH) - 1)
        div_remainder = numerator - (inv_rms_uq8_16 * rms_q14_10)

    output_q12_12: list[int] = []
    saturation = False
    for x_value, gamma_value in zip(input_int, gamma_int, strict=True):
        product = x_value * inv_rms_uq8_16 * gamma_value
        shifted = product >> NORM_OUTPUT_SHIFT
        saturated, did_saturate = saturate_signed(shifted, NORM_OUT_WIDTH)
        output_q12_12.append(saturated)
        saturation = saturation or did_saturate

    return {
        "sum_squares": sum_squares,
        "mean_square": mean_square,
        "sqrt_radicand": sqrt_radicand,
        "rms_q14_10": rms_q14_10,
        "inv_rms_uq8_16": inv_rms_uq8_16,
        "div_remainder": div_remainder,
        "output_q12_12": np.array(output_q12_12, dtype=np.int64),
        "saturation": saturation,
    }


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point post-attention residual/RMSNorm vectors.")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--o-proj-prefix", default=DEFAULT_O_PROJ_PREFIX)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--residual-hex", type=Path, default=None, help="Optional I32/Q14.10 residual input override")
    parser.add_argument("--residual-source-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_id = int(args.layer_id)
    o_proj_prefix = str(args.o_proj_prefix)
    prefix = str(args.prefix)
    module_name = f"layer{layer_id}.post_attention_residual_norm_stage"

    rms = load_rmsnorm_exporter()

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

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer.register_forward_hook(capture_module_io(records, "layer")),
        layer.post_attention_layernorm.register_forward_hook(capture_module_io(records, "post_attention_norm")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    residual_fp32 = np.ascontiguousarray(
        require_input0(records, "layer")[0, selected_position, :].to(torch.float32).numpy(),
        dtype=np.float32,
    )
    hf_post_attention_hidden = np.ascontiguousarray(
        require_input0(records, "post_attention_norm")[0, selected_position, :].to(torch.float32).numpy(),
        dtype=np.float32,
    )
    hf_post_norm = np.ascontiguousarray(
        require_output(records, "post_attention_norm")[0, selected_position, :].to(torch.float32).numpy(),
        dtype=np.float32,
    )
    gamma = np.ascontiguousarray(
        tensor_to_float32(layer.post_attention_layernorm.weight).numpy(),
        dtype=np.float32,
    )
    eps = float(get_rms_norm_eps(layer.post_attention_layernorm, model.config))

    if residual_fp32.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected residual shape {(INPUT_SIZE,)}, got {residual_fp32.shape}")
    if gamma.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected gamma shape {(INPUT_SIZE,)}, got {gamma.shape}")
    o_proj_path = SIM_VECTOR_DIR / f"{o_proj_prefix}_expected_q12_12.hex"
    o_proj_meta_path = SIM_VECTOR_DIR / f"{o_proj_prefix}_meta.json"
    o_proj_q12_12 = read_signed_hex_lines(o_proj_path, O_PROJ_WIDTH)
    if o_proj_q12_12.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected o_proj vector shape {(INPUT_SIZE,)}, got {o_proj_q12_12.shape}")
    if o_proj_meta_path.is_file():
        o_proj_meta = json.loads(o_proj_meta_path.read_text(encoding="utf-8"))
        if int(o_proj_meta["selected_position"]) != selected_position:
            raise RuntimeError("o_proj vector selected position does not match prompt")
        if int(o_proj_meta["selected_token_id"]) != selected_token_id:
            raise RuntimeError("o_proj vector selected token does not match prompt")
        if int(o_proj_meta.get("layer_id", layer_id)) != layer_id:
            raise RuntimeError("o_proj vector layer_id does not match requested layer")

    residual_q14_10 = rms.quantize_signed(residual_fp32, RESIDUAL_WIDTH, RESIDUAL_FRAC)
    residual_source = {
        "mode": "hf_layer_input_quantized",
        "source_name": None,
        "source_hex": None,
        "override_max_abs_diff_vs_hf_quantized": None,
    }
    if args.residual_hex is not None:
        residual_override = read_signed_hex_lines(args.residual_hex, 32)
        if residual_override.shape != (INPUT_SIZE,):
            raise RuntimeError(
                f"residual override shape mismatch: expected {(INPUT_SIZE,)}, got {residual_override.shape}"
            )
        low, high = signed_limits(RESIDUAL_WIDTH)
        if np.any((residual_override < low) | (residual_override > high)):
            raise RuntimeError(f"residual override contains values outside signed {RESIDUAL_WIDTH}-bit range")
        residual_source = {
            "mode": "hex_override",
            "source_name": args.residual_source_name,
            "source_hex": str(args.residual_hex),
            "override_max_abs_diff_vs_hf_quantized": int(
                np.max(np.abs(residual_override - residual_q14_10))
            ),
        }
        residual_q14_10 = residual_override.astype(np.int64)
    o_proj_q14_10 = o_proj_q12_12 >> (O_PROJ_FRAC - RESIDUAL_FRAC)
    post_attention_hidden_q14_10, residual_saturation_count = saturate_signed_array(
        residual_q14_10 + o_proj_q14_10,
        RESIDUAL_WIDTH,
    )
    gamma_q8_7 = rms.quantize_signed(gamma, GAMMA_WIDTH, GAMMA_FRAC)
    fixed_norm = fixed_rmsnorm_reference(post_attention_hidden_q14_10, gamma_q8_7)
    post_norm_q12_12 = fixed_norm["output_q12_12"]

    post_hidden_float = fixed_to_float(post_attention_hidden_q14_10, RESIDUAL_FRAC)
    post_norm_float = fixed_to_float(post_norm_q12_12, NORM_OUT_FRAC)
    residual_float = fixed_to_float(residual_q14_10, RESIDUAL_FRAC)

    files = {
        "residual_input": SIM_VECTOR_DIR / f"{prefix}_residual_input.hex",
        "o_proj_input": SIM_VECTOR_DIR / f"{prefix}_o_proj_input.hex",
        "expected_residual": SIM_VECTOR_DIR / f"{prefix}_expected_residual.hex",
        "gamma": SIM_VECTOR_DIR / f"{prefix}_gamma.hex",
        "expected_norm": SIM_VECTOR_DIR / f"{prefix}_expected_norm.hex",
        "sum_squares": SIM_VECTOR_DIR / f"{prefix}_sum_squares.hex",
        "mean_square": SIM_VECTOR_DIR / f"{prefix}_mean_square.hex",
        "sqrt_radicand": SIM_VECTOR_DIR / f"{prefix}_sqrt_radicand.hex",
        "rms": SIM_VECTOR_DIR / f"{prefix}_rms.hex",
        "inv_rms": SIM_VECTOR_DIR / f"{prefix}_inv_rms.hex",
        "residual_saturation": SIM_VECTOR_DIR / f"{prefix}_residual_saturation.hex",
        "norm_saturation": SIM_VECTOR_DIR / f"{prefix}_norm_saturation.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }

    write_hex_lines(files["residual_input"], residual_q14_10, RESIDUAL_WIDTH)
    write_hex_lines(files["o_proj_input"], o_proj_q12_12, O_PROJ_WIDTH)
    write_hex_lines(files["expected_residual"], post_attention_hidden_q14_10, RESIDUAL_WIDTH)
    write_hex_lines(files["gamma"], gamma_q8_7, GAMMA_WIDTH)
    write_hex_lines(files["expected_norm"], post_norm_q12_12, NORM_OUT_WIDTH)
    write_scalar(files["sum_squares"], int(fixed_norm["sum_squares"]), SUM_WIDTH)
    write_scalar(files["mean_square"], int(fixed_norm["mean_square"]), SUM_WIDTH)
    write_scalar(files["sqrt_radicand"], int(fixed_norm["sqrt_radicand"]), SUM_WIDTH)
    write_scalar(files["rms"], int(fixed_norm["rms_q14_10"]), RESIDUAL_WIDTH)
    write_scalar(files["inv_rms"], int(fixed_norm["inv_rms_uq8_16"]), INV_RMS_WIDTH)
    write_scalar(files["residual_saturation"], 1 if residual_saturation_count else 0, 1)
    write_scalar(files["norm_saturation"], 1 if fixed_norm["saturation"] else 0, 1)

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
        "eps_float": eps,
        "eps_q20": int(EPS_Q20),
        "residual_source": residual_source,
        "source_o_proj_prefix": o_proj_prefix,
        "formats": {
            "residual_input": f"signed {RESIDUAL_WIDTH}-bit Q14.{RESIDUAL_FRAC}",
            "o_proj_input": f"signed {O_PROJ_WIDTH}-bit Q12.{O_PROJ_FRAC}",
            "expected_residual": f"signed {RESIDUAL_WIDTH}-bit Q14.{RESIDUAL_FRAC}",
            "gamma": f"signed {GAMMA_WIDTH}-bit Q8.{GAMMA_FRAC}",
            "expected_norm": f"signed {NORM_OUT_WIDTH}-bit Q12.{NORM_OUT_FRAC}",
        },
        "saturation": {
            "residual_count": int(residual_saturation_count),
            "norm_output": bool(fixed_norm["saturation"]),
        },
        "debug": {
            "residual_input_q14_10_max_abs": int(np.max(np.abs(residual_q14_10))),
            "o_proj_q12_12_max_abs": int(np.max(np.abs(o_proj_q12_12))),
            "post_attention_hidden_q14_10_max_abs": int(np.max(np.abs(post_attention_hidden_q14_10))),
            "sum_squares": int(fixed_norm["sum_squares"]),
            "mean_square": int(fixed_norm["mean_square"]),
            "sqrt_radicand": int(fixed_norm["sqrt_radicand"]),
            "rms_q14_10": int(fixed_norm["rms_q14_10"]),
            "inv_rms_uq8_16": int(fixed_norm["inv_rms_uq8_16"]),
            "div_remainder": int(fixed_norm["div_remainder"]),
            "fixed_residual_input_vs_hf_layer_input_max_abs_diff": float(
                np.max(np.abs(residual_float - residual_fp32.astype(np.float64)))
            ),
            "fixed_post_attention_hidden_vs_hf_max_abs_diff": float(
                np.max(np.abs(post_hidden_float - hf_post_attention_hidden.astype(np.float64)))
            ),
            "fixed_post_attention_hidden_vs_hf_mean_abs_diff": float(
                np.mean(np.abs(post_hidden_float - hf_post_attention_hidden.astype(np.float64)))
            ),
            "fixed_post_norm_vs_hf_max_abs_diff": float(
                np.max(np.abs(post_norm_float - hf_post_norm.astype(np.float64)))
            ),
            "fixed_post_norm_vs_hf_mean_abs_diff": float(
                np.mean(np.abs(post_norm_float - hf_post_norm.astype(np.float64)))
            ),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point post-attention residual/RMSNorm RTL test vectors")
    print("=" * 80)
    print(f"Module: {module_name}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Residual input:   {files['residual_input']}")
    print(f"O projection:     {files['o_proj_input']}")
    print(f"Expected residual:{files['expected_residual']}")
    print(f"Gamma:            {files['gamma']}")
    print(f"Expected norm:    {files['expected_norm']}")
    print(f"Debug meta:       {files['meta']}")
    print(
        "Fixed post-norm vs HF post_attention_layernorm: "
        f"max_abs={meta['debug']['fixed_post_norm_vs_hf_max_abs_diff']:.8g} "
        f"mean_abs={meta['debug']['fixed_post_norm_vs_hf_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
