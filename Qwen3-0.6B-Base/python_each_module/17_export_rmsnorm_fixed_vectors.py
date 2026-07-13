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

PREFIX = "rmsnorm_1024_real"
MODULE_NAME = "layer0.input_layernorm"
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


def twos_complement_hex(value: int, width_bits: int) -> str:
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    return f"{value & mask:0{hex_digits}x}"


def unsigned_hex(value: int, width_bits: int) -> str:
    if value < 0:
        raise ValueError(f"Expected unsigned value, got {value}")
    return twos_complement_hex(value, width_bits)


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def unsigned_limits(width_bits: int) -> tuple[int, int]:
    return 0, (1 << width_bits) - 1


def read_signed_hex_lines(path: Path, width_bits: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    sign_bit = 1 << (width_bits - 1)
    full = 1 << width_bits
    values: list[int] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw = int(stripped, 16)
        if raw < 0 or raw >= full:
            raise ValueError(f"{path}:{lineno}: value 0x{raw:x} does not fit {width_bits} bits")
        values.append(raw - full if (raw & sign_bit) else raw)
    return np.array(values, dtype=np.int64)


def quantize_signed(values: np.ndarray, width_bits: int, frac_bits: int) -> np.ndarray:
    low, high = signed_limits(width_bits)
    scaled = np.rint(values.astype(np.float64) * float(1 << frac_bits))
    return np.clip(scaled, low, high).astype(np.int64)


def quantize_unsigned(values: np.ndarray, width_bits: int, frac_bits: int) -> np.ndarray:
    low, high = unsigned_limits(width_bits)
    scaled = np.rint(values.astype(np.float64) * float(1 << frac_bits))
    return np.clip(scaled, low, high).astype(np.int64)


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
        shifted = product >> OUTPUT_SHIFT
        saturated, did_saturate = saturate_signed(shifted, OUT_WIDTH)
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


def write_hex_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scalar(path: Path, value: int, width_bits: int, signed: bool = False) -> None:
    if signed:
        line = twos_complement_hex(value, width_bits)
    else:
        line = unsigned_hex(value, width_bits)
    write_hex_lines(path, [line])


def as_float_array(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor_to_float32(x).numpy(), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point input RMSNorm RTL vectors.")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SIM_VECTOR_DIR,
        help="Directory for all generated vector and metadata files.",
    )
    parser.add_argument("--layer-id", type=int, default=0, help="Decoder layer input RMSNorm to export")
    parser.add_argument(
        "--input-hex",
        type=Path,
        default=None,
        help="Optional signed fixed-point Q14.10 hidden input hex. Defaults to the HF layer input.",
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
    layer_id = int(args.layer_id)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    layer = backbone.layers[layer_id]
    module_name = f"layer{layer_id}.input_layernorm"

    gamma = as_float_array(layer.input_layernorm.weight)
    eps = float(get_rms_norm_eps(layer.input_layernorm, model.config))

    if gamma.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected gamma shape {(INPUT_SIZE,)}, got {gamma.shape}")

    expected_float: np.ndarray | None = None
    input_source = f"hf_{module_name}_input"
    if args.input_hex is None:
        records: dict[str, dict[str, torch.Tensor]] = {}
        hook = layer.input_layernorm.register_forward_hook(
            capture_module_io(records, "input_norm")
        )
        try:
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
        finally:
            hook.remove()

        input_hidden = as_float_array(
            require_input0(records, "input_norm")[0, selected_position, :]
        )
        expected_float = as_float_array(
            require_output(records, "input_norm")[0, selected_position, :]
        )

        if input_hidden.shape != (INPUT_SIZE,):
            raise RuntimeError(f"Expected input shape {(INPUT_SIZE,)}, got {input_hidden.shape}")
        input_q14_10 = quantize_signed(input_hidden, IN_WIDTH, IN_FRAC)
    else:
        input_source = args.input_source_name or args.input_hex.as_posix()
        input_q14_10 = read_signed_hex_lines(args.input_hex, int(args.input_hex_width))
        if input_q14_10.shape != (INPUT_SIZE,):
            raise RuntimeError(f"Expected input shape {(INPUT_SIZE,)}, got {input_q14_10.shape}")
        low, high = signed_limits(IN_WIDTH)
        if np.any(input_q14_10 < low) or np.any(input_q14_10 > high):
            raise RuntimeError(f"--input-hex contains values outside signed {IN_WIDTH}-bit range")

    gamma_q8_7 = quantize_signed(gamma, GAMMA_WIDTH, GAMMA_FRAC)
    fixed_ref = fixed_rmsnorm_reference(input_q14_10, gamma_q8_7)
    output_q12_12 = fixed_ref["output_q12_12"]

    fixed_output_float = output_q12_12.astype(np.float64) / float(1 << OUT_FRAC)
    if expected_float is None:
        float_max_abs_diff = None
    else:
        float_max_abs_diff = float(
            np.max(np.abs(fixed_output_float - expected_float.astype(np.float64)))
        )

    input_path = output_dir / f"{prefix}_input.hex"
    gamma_path = output_dir / f"{prefix}_gamma.hex"
    expected_path = output_dir / f"{prefix}_expected.hex"
    sum_path = output_dir / f"{prefix}_sum_squares.hex"
    mean_path = output_dir / f"{prefix}_mean_square.hex"
    radicand_path = output_dir / f"{prefix}_sqrt_radicand.hex"
    rms_path = output_dir / f"{prefix}_rms.hex"
    inv_rms_path = output_dir / f"{prefix}_inv_rms.hex"
    saturation_path = output_dir / f"{prefix}_saturation.hex"
    meta_path = output_dir / f"{prefix}_meta.json"

    write_hex_lines(
        input_path,
        [twos_complement_hex(int(value), IN_WIDTH) for value in input_q14_10],
    )
    write_hex_lines(
        gamma_path,
        [twos_complement_hex(int(value), GAMMA_WIDTH) for value in gamma_q8_7],
    )
    write_hex_lines(
        expected_path,
        [twos_complement_hex(int(value), OUT_WIDTH) for value in output_q12_12],
    )
    write_scalar(sum_path, int(fixed_ref["sum_squares"]), SUM_WIDTH)
    write_scalar(mean_path, int(fixed_ref["mean_square"]), SUM_WIDTH)
    write_scalar(radicand_path, int(fixed_ref["sqrt_radicand"]), SUM_WIDTH)
    write_scalar(rms_path, int(fixed_ref["rms_q14_10"]), RMS_WIDTH)
    write_scalar(inv_rms_path, int(fixed_ref["inv_rms_uq8_16"]), INV_RMS_WIDTH)
    write_scalar(saturation_path, 1 if fixed_ref["saturation"] else 0, 1)

    meta = {
        "format_version": 1,
        "name": prefix,
        "module": module_name,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "input_source": input_source,
        "eps_float": eps,
        "eps_q20": EPS_Q20,
        "formats": {
            "input": f"signed {IN_WIDTH}-bit Q14.{IN_FRAC}",
            "gamma": f"signed {GAMMA_WIDTH}-bit Q8.{GAMMA_FRAC}",
            "expected_output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
            "inv_rms": f"unsigned {INV_RMS_WIDTH}-bit UQ8.{INV_RMS_FRAC}",
        },
        "debug": {
            "sum_squares": int(fixed_ref["sum_squares"]),
            "mean_square": int(fixed_ref["mean_square"]),
            "sqrt_radicand": int(fixed_ref["sqrt_radicand"]),
            "rms_q14_10": int(fixed_ref["rms_q14_10"]),
            "inv_rms_uq8_16": int(fixed_ref["inv_rms_uq8_16"]),
            "div_remainder": int(fixed_ref["div_remainder"]),
            "saturation": bool(fixed_ref["saturation"]),
            "fixed_vs_fp32_output_max_abs_diff": float_max_abs_diff,
        },
        "files": {
            "input": input_path.name,
            "gamma": gamma_path.name,
            "expected": expected_path.name,
            "sum_squares": sum_path.name,
            "mean_square": mean_path.name,
            "sqrt_radicand": radicand_path.name,
            "rms": rms_path.name,
            "inv_rms": inv_rms_path.name,
            "saturation": saturation_path.name,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point RMSNorm RTL test vectors")
    print("=" * 80)
    print(f"Module: {module_name}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Input source: {input_source}")
    print(f"Input:       {input_path}")
    print(f"Gamma:       {gamma_path}")
    print(f"Expected:    {expected_path}")
    print(f"Debug meta:  {meta_path}")
    print(f"sum_squares: {fixed_ref['sum_squares']}")
    print(f"mean_square: {fixed_ref['mean_square']}")
    print(f"rms_q14_10:  {fixed_ref['rms_q14_10']}")
    print(f"inv_rms:     {fixed_ref['inv_rms_uq8_16']}")
    print(f"saturation:  {int(fixed_ref['saturation'])}")
    if float_max_abs_diff is not None:
        print(f"Fixed output vs FP32 model max abs diff: {float_max_abs_diff:.8g}")


if __name__ == "__main__":
    main()
