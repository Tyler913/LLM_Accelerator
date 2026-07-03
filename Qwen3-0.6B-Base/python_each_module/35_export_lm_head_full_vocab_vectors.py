from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

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
FINAL_NORM_PREFIX = "final_rmsnorm_stage_real"
DEFAULT_PREFIX = "lm_head_argmax_full_vocab_real"

INPUT_SIZE = 1024
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
Q4_MIN = -8
Q4_MAX = 7

ACT_WIDTH = 24
ACT_FRAC = 12
WEIGHT_WIDTH = 4
SCALE_WIDTH = 16
SCALE_FRAC = 14
PARTIAL_WIDTH = ACT_WIDTH + WEIGHT_WIDTH + 6
SCALED_WIDTH = PARTIAL_WIDTH + SCALE_WIDTH
ROW_ACC_WIDTH = SCALED_WIDTH + 4 + 2

LM_HEAD_WEIGHT_BASE_ADDR = 0x4_0010_0000
LM_HEAD_SCALE_BASE_ADDR = LM_HEAD_WEIGHT_BASE_ADDR + 0x04A3_0000


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


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    lines = [f"{int(value) & mask:0{hex_digits}x}" for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scalar(path: Path, value: int, width_bits: int) -> None:
    write_hex_lines(path, np.array([value], dtype=np.int64), width_bits)


def append_hex_lines(file: TextIO, values: np.ndarray, width_bits: int) -> None:
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    file.write("\n".join(f"{int(value) & mask:0{hex_digits}x}" for value in values.reshape(-1)))
    file.write("\n")


def pack_q4_to_words32(q4: np.ndarray) -> np.ndarray:
    flat = np.bitwise_and(q4.reshape(-1).astype(np.int16), 0x0F).astype(np.uint32)
    if (flat.size % 8) != 0:
        raise ValueError("Q4 flat size must be divisible by 8")
    grouped = flat.reshape(-1, 8)
    words = np.zeros(grouped.shape[0], dtype=np.uint32)
    for index in range(8):
        words |= grouped[:, index] << (index * 4)
    return words


def pack_scale_to_words32(scale: np.ndarray) -> np.ndarray:
    flat = scale.reshape(-1).astype(np.uint32)
    if (flat.size % 2) != 0:
        raise ValueError("Scale flat size must be divisible by 2")
    grouped = flat.reshape(-1, 2)
    return grouped[:, 0] | (grouped[:, 1] << 16)


def quantize_weight_q4_chunk(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weight.shape[1] != INPUT_SIZE:
        raise ValueError(f"Expected weight second dimension {INPUT_SIZE}, got {weight.shape}")

    row_count = weight.shape[0]
    grouped = np.ascontiguousarray(weight.astype(np.float32)).reshape(row_count, GROUP_COUNT, GROUP_SIZE)
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
    return q.reshape(row_count, INPUT_SIZE), scale_q2_14


def compute_q4_logits_q26(
    activation_q12_12: np.ndarray,
    weight_q4: np.ndarray,
    scale_q2_14: np.ndarray,
) -> np.ndarray:
    row_count = weight_q4.shape[0]
    x_grouped = activation_q12_12.astype(np.int64).reshape(GROUP_COUNT, GROUP_SIZE)
    w_grouped = weight_q4.astype(np.int64).reshape(row_count, GROUP_COUNT, GROUP_SIZE)
    partial_sums = np.sum(w_grouped * x_grouped[None, :, :], axis=2, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14.astype(np.int64)
    return np.sum(scaled_sums, axis=1, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-vocabulary Q4 LM-head vectors for RTL simulation.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = str(args.prefix)
    tile_rows = int(args.tile_rows)
    chunk_rows = int(args.chunk_rows)
    if tile_rows <= 0:
        raise ValueError("--tile-rows must be positive")
    if chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    tokenizer, model, _backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = model.lm_head.register_forward_hook(capture_module_io(records, "lm_head"))
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()

    activation_path = SIM_VECTOR_DIR / f"{FINAL_NORM_PREFIX}_expected.hex"
    activation_meta_path = SIM_VECTOR_DIR / f"{FINAL_NORM_PREFIX}_meta.json"
    activation_q12_12 = read_signed_hex_lines(activation_path, ACT_WIDTH)
    if activation_q12_12.shape != (INPUT_SIZE,):
        raise RuntimeError(f"Expected final_norm shape {(INPUT_SIZE,)}, got {activation_q12_12.shape}")
    if activation_meta_path.is_file():
        final_meta = json.loads(activation_meta_path.read_text(encoding="utf-8"))
        if int(final_meta["selected_position"]) != selected_position:
            raise RuntimeError("final RMSNorm vector selected position does not match prompt")
        if int(final_meta["selected_token_id"]) != selected_token_id:
            raise RuntimeError("final RMSNorm vector selected token does not match prompt")

    lm_weight = model.lm_head.weight.detach().to(dtype=torch.float32, device="cpu")
    vocab_size = int(lm_weight.shape[0])
    if int(lm_weight.shape[1]) != INPUT_SIZE:
        raise RuntimeError(f"Expected lm_head weight width {INPUT_SIZE}, got {lm_weight.shape}")
    if (vocab_size % tile_rows) != 0:
        raise RuntimeError(f"vocab_size {vocab_size} must be divisible by tile_rows {tile_rows}")

    files = {
        "activation": SIM_VECTOR_DIR / f"{prefix}_activation.hex",
        "weight_words32": SIM_VECTOR_DIR / f"{prefix}_weight_words32.hex",
        "scale_words32": SIM_VECTOR_DIR / f"{prefix}_scale_words32.hex",
        "expected_scan_logits_q26": SIM_VECTOR_DIR / f"{prefix}_expected_scan_logits_q26.hex",
        "expected_best_token": SIM_VECTOR_DIR / f"{prefix}_expected_best_token.hex",
        "expected_best_score_q26": SIM_VECTOR_DIR / f"{prefix}_expected_best_score_q26.hex",
        "scan_base_token": SIM_VECTOR_DIR / f"{prefix}_scan_base_token.hex",
        "weight_base_addr": SIM_VECTOR_DIR / f"{prefix}_weight_base_addr.hex",
        "scale_base_addr": SIM_VECTOR_DIR / f"{prefix}_scale_base_addr.hex",
        "meta": SIM_VECTOR_DIR / f"{prefix}_meta.json",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_hex_lines(files["activation"], activation_q12_12, ACT_WIDTH)
    write_scalar(files["scan_base_token"], 0, 32)
    write_scalar(files["weight_base_addr"], LM_HEAD_WEIGHT_BASE_ADDR, 64)
    write_scalar(files["scale_base_addr"], LM_HEAD_SCALE_BASE_ADDR, 64)

    best_token = -1
    best_score = -(1 << 62)
    top_tokens: list[tuple[int, int]] = []

    with (
        files["weight_words32"].open("w", encoding="utf-8") as weight_file,
        files["scale_words32"].open("w", encoding="utf-8") as scale_file,
        files["expected_scan_logits_q26"].open("w", encoding="utf-8") as logits_file,
    ):
        for row_start in range(0, vocab_size, chunk_rows):
            row_end = min(row_start + chunk_rows, vocab_size)
            weight_chunk = np.ascontiguousarray(lm_weight[row_start:row_end].numpy(), dtype=np.float32)
            q_chunk, scale_chunk = quantize_weight_q4_chunk(weight_chunk)
            logits_q26 = compute_q4_logits_q26(activation_q12_12, q_chunk, scale_chunk)

            append_hex_lines(weight_file, pack_q4_to_words32(q_chunk), 32)
            append_hex_lines(scale_file, pack_scale_to_words32(scale_chunk), 32)
            append_hex_lines(logits_file, logits_q26, ROW_ACC_WIDTH)

            chunk_best_index = int(np.argmax(logits_q26))
            chunk_best_score = int(logits_q26[chunk_best_index])
            if chunk_best_score > best_score:
                best_score = chunk_best_score
                best_token = row_start + chunk_best_index
            top_tokens.extend((row_start + int(i), int(logits_q26[int(i)])) for i in np.argsort(logits_q26)[-16:])
            top_tokens = sorted(top_tokens, key=lambda item: (-item[1], item[0]))[:16]
            print(
                f"chunk [{row_start}, {row_end}) best={row_start + chunk_best_index} "
                f"score_q26={chunk_best_score}"
            )

    if best_token < 0:
        raise RuntimeError("Failed to find Q4 argmax")

    hf_logits = require_output(records, "lm_head")[0, selected_position, :].numpy().astype(np.float64)
    hf_argmax = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
    fp32_argmax = int(np.argmax(hf_logits))
    best_token_text = tokenizer.decode([best_token])
    hf_token_text = tokenizer.decode([hf_argmax])
    fp32_token_text = tokenizer.decode([fp32_argmax])

    write_scalar(files["expected_best_token"], best_token, 32)
    write_scalar(files["expected_best_score_q26"], best_score, ROW_ACC_WIDTH)

    meta = {
        "format_version": 1,
        "name": prefix,
        "module": "model.lm_head_argmax_full_vocab",
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "source_final_norm_prefix": FINAL_NORM_PREFIX,
        "shape": {
            "vocab_size": vocab_size,
            "input_size": INPUT_SIZE,
            "group_size": GROUP_SIZE,
            "group_count": GROUP_COUNT,
            "scan_base": 0,
            "scan_rows": vocab_size,
            "tile_rows": tile_rows,
            "tile_count": vocab_size // tile_rows,
        },
        "memory_layout": {
            "weight_base_addr": int(LM_HEAD_WEIGHT_BASE_ADDR),
            "scale_base_addr": int(LM_HEAD_SCALE_BASE_ADDR),
            "weight_row_bytes": int(INPUT_SIZE * WEIGHT_WIDTH // 8),
            "scale_row_bytes": int(GROUP_COUNT * SCALE_WIDTH // 8),
            "scan_weight_bytes": int(vocab_size * INPUT_SIZE * WEIGHT_WIDTH // 8),
            "scan_scale_bytes": int(vocab_size * GROUP_COUNT * SCALE_WIDTH // 8),
            "full_vocab_weight_bytes": int(vocab_size * INPUT_SIZE * WEIGHT_WIDTH // 8),
            "full_vocab_scale_bytes": int(vocab_size * GROUP_COUNT * SCALE_WIDTH // 8),
        },
        "formats": {
            "activation": f"signed {ACT_WIDTH}-bit Q12.{ACT_FRAC}",
            "weight_words32": "full-vocabulary packed Q4 row-major data, eight Q4 values per 32-bit word",
            "scale_words32": "full-vocabulary scale data, two Q2.14 values per 32-bit word",
            "logit": f"signed {ROW_ACC_WIDTH}-bit Q26",
        },
        "argmax": {
            "full_q4_token": int(best_token),
            "full_q4_token_text": best_token_text,
            "full_q4_score_q26": int(best_score),
            "hf_argmax_token": int(hf_argmax),
            "hf_argmax_token_text": hf_token_text,
            "fp32_lm_head_argmax_token": int(fp32_argmax),
            "fp32_lm_head_argmax_token_text": fp32_token_text,
        },
        "debug": {
            "activation_q12_12_min": int(np.min(activation_q12_12)),
            "activation_q12_12_max": int(np.max(activation_q12_12)),
            "activation_q12_12_max_abs": int(np.max(np.abs(activation_q12_12))),
            "top_full_q4_tokens": [
                {"token": int(token), "score_q26": int(score), "text": tokenizer.decode([int(token)])}
                for token, score in top_tokens[:8]
            ],
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported full-vocabulary Q4 LM-head vectors")
    print("=" * 80)
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Full Q4 argmax: {best_token} / {best_token_text!r}, score_q26={best_score}")
    print(f"HF argmax:      {hf_argmax} / {hf_token_text!r}")
    print(f"FP32 argmax:    {fp32_argmax} / {fp32_token_text!r}")
    print(f"Scan window:    [0, {vocab_size}) rows={vocab_size}, tiles={vocab_size // tile_rows}")
    print(f"Weight words32: {files['weight_words32']}")
    print(f"Scale words32:  {files['scale_words32']}")
    print(f"Logits:         {files['expected_scan_logits_q26']}")
    print(f"Debug meta:     {files['meta']}")


if __name__ == "__main__":
    main()
