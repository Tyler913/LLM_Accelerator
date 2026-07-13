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
)
from vector_workspace import resolve_sim_vector_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = resolve_sim_vector_dir(REPO_ROOT)
DEFAULT_POST_ATTENTION_PREFIX = "post_attention_residual_norm_stage_real"
DEFAULT_DOWN_PREFIX = "mlp_down_proj_stage_real"
DEFAULT_PREFIX = "mlp_residual_add_stage_real"

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


def load_and_check_meta(path: Path, selected_position: int, selected_token_id: int, layer_id: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if int(meta["selected_position"]) != selected_position:
        raise RuntimeError(f"{path.name} selected position does not match prompt")
    if int(meta["selected_token_id"]) != selected_token_id:
        raise RuntimeError(f"{path.name} selected token does not match prompt")
    if int(meta.get("layer_id", layer_id)) != layer_id:
        raise RuntimeError(f"{path.name} layer_id does not match requested layer")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point MLP residual-add RTL vectors.")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--post-attention-prefix", default=DEFAULT_POST_ATTENTION_PREFIX)
    parser.add_argument("--down-prefix", default=DEFAULT_DOWN_PREFIX)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_id = int(args.layer_id)
    post_attention_prefix = str(args.post_attention_prefix)
    down_prefix = str(args.down_prefix)
    prefix = str(args.prefix)
    module_name = f"layer{layer_id}.mlp.residual_add_stage"

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
    hook_name = f"layer{layer_id}"

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = layer.register_forward_hook(capture_module_io(records, hook_name))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()

    post_attn_path = SIM_VECTOR_DIR / f"{post_attention_prefix}_expected_residual.hex"
    post_attn_meta_path = SIM_VECTOR_DIR / f"{post_attention_prefix}_meta.json"
    down_path = SIM_VECTOR_DIR / f"{down_prefix}_expected_q12_12.hex"
    down_meta_path = SIM_VECTOR_DIR / f"{down_prefix}_meta.json"

    post_attn_q14_10 = read_signed_hex_lines(post_attn_path, POST_ATTENTION_WIDTH)
    down_q12_12 = read_signed_hex_lines(down_path, DOWN_WIDTH)
    if post_attn_q14_10.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected post-attention shape {(INPUT_SIZE,)}, got {post_attn_q14_10.shape}")
    if down_q12_12.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected down projection shape {(INPUT_SIZE,)}, got {down_q12_12.shape}")

    load_and_check_meta(post_attn_meta_path, selected_position, selected_token_id, layer_id)
    load_and_check_meta(down_meta_path, selected_position, selected_token_id, layer_id)

    down_q14_10 = down_q12_12 >> DOWN_TO_OUT_SHIFT
    layer_out_q14_10, saturation_count = saturate_signed_array(
        post_attn_q14_10 + down_q14_10,
        OUT_WIDTH,
    )

    hf_layer_out = require_output(records, hook_name)[0, selected_position, :].numpy().astype(np.float64)
    layer_out_float = fixed_to_float(layer_out_q14_10, OUT_FRAC)
    diff = layer_out_float - hf_layer_out

    files = {
        "post_attn_hidden": SIM_VECTOR_DIR / f"{prefix}_post_attn_hidden.hex",
        "down_out": SIM_VECTOR_DIR / f"{prefix}_down_out.hex",
        "expected_layer_out": SIM_VECTOR_DIR / f"{prefix}_expected_layer_out.hex",
        "residual_saturation": SIM_VECTOR_DIR / f"{prefix}_residual_saturation.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }

    write_hex_lines(files["post_attn_hidden"], post_attn_q14_10, POST_ATTENTION_WIDTH)
    write_hex_lines(files["down_out"], down_q12_12, DOWN_WIDTH)
    write_hex_lines(files["expected_layer_out"], layer_out_q14_10, OUT_WIDTH)
    write_scalar(files["residual_saturation"], 1 if saturation_count else 0, 1)

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
        "source_post_attention_prefix": post_attention_prefix,
        "source_down_prefix": down_prefix,
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
    print(f"Module: {module_name}")
    print(f"Layer: {layer_id}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Post-attn hidden: {files['post_attn_hidden']}")
    print(f"Down output:      {files['down_out']}")
    print(f"Expected layer:   {files['expected_layer_out']}")
    print(f"Debug meta:       {files['meta']}")
    print(
        f"Fixed Layer {layer_id} output vs HF: "
        f"max={meta['debug']['fixed_layer_out_vs_hf_max_abs_diff']:.8g} "
        f"mean={meta['debug']['fixed_layer_out_vs_hf_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
