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
    require_output,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
POST_ATTENTION_PREFIX = "post_attention_residual_norm_stage_real"
DOWN_PREFIX = "mlp_down_proj_stage_real"

PREFIX = "mlp_residual_add_stage_real"
MODULE_NAME = "layer0.mlp.residual_add_stage"

INPUT_SIZE = 1024
POST_ATTENTION_WIDTH = 24
POST_ATTENTION_FRAC = 10
DOWN_WIDTH = 24
DOWN_FRAC = 12
OUT_WIDTH = 24
OUT_FRAC = 10
DOWN_TO_OUT_SHIFT = DOWN_FRAC - OUT_FRAC


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


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def load_and_check_meta(path: Path, selected_position: int, selected_token_id: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if int(meta["selected_position"]) != selected_position:
        raise RuntimeError(f"{path.name} selected position does not match prompt")
    if int(meta["selected_token_id"]) != selected_token_id:
        raise RuntimeError(f"{path.name} selected token does not match prompt")
    return meta


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    layer0 = backbone.layers[0]

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = layer0.register_forward_hook(capture_module_io(records, "layer0"))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()

    post_attn_path = SIM_VECTOR_DIR / f"{POST_ATTENTION_PREFIX}_expected_residual.hex"
    post_attn_meta_path = SIM_VECTOR_DIR / f"{POST_ATTENTION_PREFIX}_meta.json"
    down_path = SIM_VECTOR_DIR / f"{DOWN_PREFIX}_expected_q12_12.hex"
    down_meta_path = SIM_VECTOR_DIR / f"{DOWN_PREFIX}_meta.json"

    post_attn_q14_10 = read_signed_hex_lines(post_attn_path, POST_ATTENTION_WIDTH)
    down_q12_12 = read_signed_hex_lines(down_path, DOWN_WIDTH)
    if post_attn_q14_10.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected post-attention shape {(INPUT_SIZE,)}, got {post_attn_q14_10.shape}")
    if down_q12_12.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected down projection shape {(INPUT_SIZE,)}, got {down_q12_12.shape}")

    load_and_check_meta(post_attn_meta_path, selected_position, selected_token_id)
    load_and_check_meta(down_meta_path, selected_position, selected_token_id)

    down_q14_10 = down_q12_12 >> DOWN_TO_OUT_SHIFT
    layer_out_q14_10, saturation_count = saturate_signed_array(
        post_attn_q14_10 + down_q14_10,
        OUT_WIDTH,
    )

    hf_layer_out = require_output(records, "layer0")[0, selected_position, :].numpy().astype(np.float64)
    layer_out_float = fixed_to_float(layer_out_q14_10, OUT_FRAC)
    diff = layer_out_float - hf_layer_out

    files = {
        "post_attn_hidden": SIM_VECTOR_DIR / f"{PREFIX}_post_attn_hidden.hex",
        "down_out": SIM_VECTOR_DIR / f"{PREFIX}_down_out.hex",
        "expected_layer_out": SIM_VECTOR_DIR / f"{PREFIX}_expected_layer_out.hex",
        "residual_saturation": SIM_VECTOR_DIR / f"{PREFIX}_residual_saturation.hex",
        "meta": SIM_VECTOR_DIR / f"{PREFIX}_meta.json",
    }

    write_hex_lines(files["post_attn_hidden"], post_attn_q14_10, POST_ATTENTION_WIDTH)
    write_hex_lines(files["down_out"], down_q12_12, DOWN_WIDTH)
    write_hex_lines(files["expected_layer_out"], layer_out_q14_10, OUT_WIDTH)
    write_scalar(files["residual_saturation"], 1 if saturation_count else 0, 1)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": PREFIX,
        "module": MODULE_NAME,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "source_post_attention_prefix": POST_ATTENTION_PREFIX,
        "source_down_prefix": DOWN_PREFIX,
        "shape": {
            "input_size": INPUT_SIZE,
        },
        "formats": {
            "post_attn_hidden": f"signed {POST_ATTENTION_WIDTH}-bit Q14.{POST_ATTENTION_FRAC}",
            "down_out": f"signed {DOWN_WIDTH}-bit Q12.{DOWN_FRAC}",
            "expected_layer_out": f"signed {OUT_WIDTH}-bit Q14.{OUT_FRAC}",
        },
        "saturation": {
            "residual_count": int(saturation_count),
        },
        "debug": {
            "post_attn_hidden_q14_10_min": int(np.min(post_attn_q14_10)),
            "post_attn_hidden_q14_10_max": int(np.max(post_attn_q14_10)),
            "post_attn_hidden_q14_10_max_abs": int(np.max(np.abs(post_attn_q14_10))),
            "down_q12_12_min": int(np.min(down_q12_12)),
            "down_q12_12_max": int(np.max(down_q12_12)),
            "down_q12_12_max_abs": int(np.max(np.abs(down_q12_12))),
            "layer_out_q14_10_min": int(np.min(layer_out_q14_10)),
            "layer_out_q14_10_max": int(np.max(layer_out_q14_10)),
            "layer_out_q14_10_max_abs": int(np.max(np.abs(layer_out_q14_10))),
            "fixed_layer_out_vs_hf_max_abs_diff": float(np.max(np.abs(diff))),
            "fixed_layer_out_vs_hf_mean_abs_diff": float(np.mean(np.abs(diff))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point MLP residual-add RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Post-attn hidden: {files['post_attn_hidden']}")
    print(f"Down output:      {files['down_out']}")
    print(f"Expected layer:   {files['expected_layer_out']}")
    print(f"Debug meta:       {files['meta']}")
    print(
        "Fixed Layer 0 output vs HF: "
        f"max={meta['debug']['fixed_layer_out_vs_hf_max_abs_diff']:.8g} "
        f"mean={meta['debug']['fixed_layer_out_vs_hf_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
