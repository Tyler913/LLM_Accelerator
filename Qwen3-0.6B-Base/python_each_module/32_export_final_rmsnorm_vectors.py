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
    get_rms_norm_eps,
    load_model,
    require_input0,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

PREFIX = "final_rmsnorm_stage_real"
MODULE_NAME = "model.final_norm"

INPUT_SIZE = 1024
IN_WIDTH = 24
IN_FRAC = 10
GAMMA_WIDTH = 16
GAMMA_FRAC = 7
INV_RMS_WIDTH = 24
INV_RMS_FRAC = 16
OUT_WIDTH = 24
OUT_FRAC = 12
SUM_WIDTH = 64
SUM_FRAC = 2 * IN_FRAC
MEAN_SHIFT = 10
RMS_WIDTH = IN_WIDTH
RMS_FRAC = IN_FRAC
DIV_NUM_SHIFT = RMS_FRAC + INV_RMS_FRAC
EPS_Q20 = 1
OUTPUT_SHIFT = IN_FRAC + INV_RMS_FRAC + GAMMA_FRAC - OUT_FRAC


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


def quantize_signed_with_saturation(
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
    saturation_count = 0
    for x_value, gamma_value in zip(input_int, gamma_int, strict=True):
        product = x_value * inv_rms_uq8_16 * gamma_value
        shifted = product >> OUTPUT_SHIFT
        saturated, did_saturate = saturate_signed(shifted, OUT_WIDTH)
        output_q12_12.append(saturated)
        saturation_count += 1 if did_saturate else 0

    return {
        "sum_squares": sum_squares,
        "mean_square": mean_square,
        "sqrt_radicand": sqrt_radicand,
        "rms_q14_10": rms_q14_10,
        "inv_rms_uq8_16": inv_rms_uq8_16,
        "div_remainder": div_remainder,
        "output_q12_12": np.array(output_q12_12, dtype=np.int64),
        "output_saturation_count": saturation_count,
    }


def as_float_array(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor_to_float32(x).numpy(), dtype=np.float32)


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point final RMSNorm RTL vectors.")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument(
        "--input-hex",
        type=Path,
        default=None,
        help="Optional signed fixed-point Q14.10 input vector hex. Defaults to the HF final_norm input.",
    )
    parser.add_argument(
        "--input-hex-width",
        type=int,
        default=32,
        help="Bit width used to decode --input-hex. QMAP word files should use 32.",
    )
    parser.add_argument("--input-source-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = str(args.prefix)

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())

    gamma = as_float_array(backbone.norm.weight)
    eps = float(get_rms_norm_eps(backbone.norm, model.config))

    if gamma.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected gamma shape {(INPUT_SIZE,)}, got {gamma.shape}")

    expected_float: np.ndarray | None = None
    input_source = "hf_final_norm_input"
    if args.input_hex is None:
        records: dict[str, dict[str, torch.Tensor]] = {}
        hook = backbone.norm.register_forward_hook(capture_module_io(records, "final_norm"))
        try:
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
        finally:
            hook.remove()

        input_hidden = as_float_array(require_input0(records, "final_norm")[0, selected_position, :])
        expected_float = as_float_array(require_output(records, "final_norm")[0, selected_position, :])
        if input_hidden.shape != (INPUT_SIZE,):
            raise RuntimeError(f"Expected input shape {(INPUT_SIZE,)}, got {input_hidden.shape}")
        input_q14_10, input_sat_count = quantize_signed_with_saturation(input_hidden, IN_WIDTH, IN_FRAC)
    else:
        input_source = args.input_source_name or args.input_hex.as_posix()
        input_q14_10 = read_signed_hex_lines(args.input_hex, int(args.input_hex_width))
        if input_q14_10.shape != (INPUT_SIZE,):
            raise RuntimeError(f"Expected input shape {(INPUT_SIZE,)}, got {input_q14_10.shape}")
        low, high = signed_limits(IN_WIDTH)
        if np.any(input_q14_10 < low) or np.any(input_q14_10 > high):
            raise RuntimeError(f"--input-hex contains values outside signed {IN_WIDTH}-bit range")
        input_sat_count = 0

    gamma_q8_7, gamma_sat_count = quantize_signed_with_saturation(gamma, GAMMA_WIDTH, GAMMA_FRAC)
    fixed_ref = fixed_rmsnorm_reference(input_q14_10, gamma_q8_7)
    output_q12_12 = fixed_ref["output_q12_12"]

    fixed_output_float = fixed_to_float(output_q12_12, OUT_FRAC)
    if expected_float is not None:
        diff = fixed_output_float - expected_float.astype(np.float64)
        max_abs_diff = float(np.max(np.abs(diff)))
        mean_abs_diff = float(np.mean(np.abs(diff)))
    else:
        max_abs_diff = None
        mean_abs_diff = None

    files = {
        "input": SIM_VECTOR_DIR / f"{prefix}_input.hex",
        "gamma": SIM_VECTOR_DIR / f"{prefix}_gamma.hex",
        "expected": SIM_VECTOR_DIR / f"{prefix}_expected.hex",
        "sum_squares": SIM_VECTOR_DIR / f"{prefix}_sum_squares.hex",
        "mean_square": SIM_VECTOR_DIR / f"{prefix}_mean_square.hex",
        "sqrt_radicand": SIM_VECTOR_DIR / f"{prefix}_sqrt_radicand.hex",
        "rms": SIM_VECTOR_DIR / f"{prefix}_rms.hex",
        "inv_rms": SIM_VECTOR_DIR / f"{prefix}_inv_rms.hex",
        "saturation": SIM_VECTOR_DIR / f"{prefix}_saturation.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }

    write_hex_lines(files["input"], input_q14_10, IN_WIDTH)
    write_hex_lines(files["gamma"], gamma_q8_7, GAMMA_WIDTH)
    write_hex_lines(files["expected"], output_q12_12, OUT_WIDTH)
    write_scalar(files["sum_squares"], int(fixed_ref["sum_squares"]), SUM_WIDTH)
    write_scalar(files["mean_square"], int(fixed_ref["mean_square"]), SUM_WIDTH)
    write_scalar(files["sqrt_radicand"], int(fixed_ref["sqrt_radicand"]), SUM_WIDTH)
    write_scalar(files["rms"], int(fixed_ref["rms_q14_10"]), RMS_WIDTH)
    write_scalar(files["inv_rms"], int(fixed_ref["inv_rms_uq8_16"]), INV_RMS_WIDTH)
    write_scalar(files["saturation"], 1 if fixed_ref["output_saturation_count"] else 0, 1)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": prefix,
        "module": MODULE_NAME,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "input_source": input_source,
        "eps_float": eps,
        "eps_q20": int(EPS_Q20),
        "formats": {
            "input": f"signed {IN_WIDTH}-bit Q14.{IN_FRAC}",
            "gamma": f"signed {GAMMA_WIDTH}-bit Q8.{GAMMA_FRAC}",
            "expected_output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
            "inv_rms": f"unsigned {INV_RMS_WIDTH}-bit UQ8.{INV_RMS_FRAC}",
        },
        "saturation": {
            "input_quantization_count": int(input_sat_count),
            "gamma_quantization_count": int(gamma_sat_count),
            "output_count": int(fixed_ref["output_saturation_count"]),
        },
        "debug": {
            "input_q14_10_min": int(np.min(input_q14_10)),
            "input_q14_10_max": int(np.max(input_q14_10)),
            "input_q14_10_max_abs": int(np.max(np.abs(input_q14_10))),
            "gamma_q8_7_min": int(np.min(gamma_q8_7)),
            "gamma_q8_7_max": int(np.max(gamma_q8_7)),
            "gamma_negative_count": int(np.count_nonzero(gamma_q8_7 < 0)),
            "sum_squares": int(fixed_ref["sum_squares"]),
            "mean_square": int(fixed_ref["mean_square"]),
            "sqrt_radicand": int(fixed_ref["sqrt_radicand"]),
            "rms_q14_10": int(fixed_ref["rms_q14_10"]),
            "inv_rms_uq8_16": int(fixed_ref["inv_rms_uq8_16"]),
            "div_remainder": int(fixed_ref["div_remainder"]),
            "output_q12_12_min": int(np.min(output_q12_12)),
            "output_q12_12_max": int(np.max(output_q12_12)),
            "output_q12_12_max_abs": int(np.max(np.abs(output_q12_12))),
            "fixed_output_vs_hf_max_abs_diff": max_abs_diff,
            "fixed_output_vs_hf_mean_abs_diff": mean_abs_diff,
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point final RMSNorm RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prefix: {prefix}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Input source: {input_source}")
    print(f"Input:       {files['input']}")
    print(f"Gamma:       {files['gamma']}")
    print(f"Expected:    {files['expected']}")
    print(f"Debug meta:  {files['meta']}")
    print(f"sum_squares: {fixed_ref['sum_squares']}")
    print(f"mean_square: {fixed_ref['mean_square']}")
    print(f"rms_q14_10:  {fixed_ref['rms_q14_10']}")
    print(f"inv_rms:     {fixed_ref['inv_rms_uq8_16']}")
    print(f"saturation:  {int(fixed_ref['output_saturation_count'] != 0)}")
    if max_abs_diff is not None and mean_abs_diff is not None:
        print(
            "Fixed final RMSNorm vs HF: "
            f"max={max_abs_diff:.8g} "
            f"mean={mean_abs_diff:.8g}"
        )


if __name__ == "__main__":
    main()
