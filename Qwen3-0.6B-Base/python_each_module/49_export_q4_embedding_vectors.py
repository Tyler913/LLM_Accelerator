from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import PROMPT, encode_prompt, load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Temp" / "q4_embedding_vectors"

INPUT_SIZE = 1024
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
Q4_MIN = -8
Q4_MAX = 7
SCALE_FRAC = 14
OUTPUT_FRAC = 10
DEQUANT_SHIFT = SCALE_FRAC - OUTPUT_FRAC
WEIGHT_BASE_ADDR = 0x4_0010_0000
SCALE_BASE_ADDR = WEIGHT_BASE_ADDR + 0x04A3_0000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one tied Q4 embedding row and exact RTL Q14.10 output."
    )
    parser.add_argument("--token-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def quantize_row(weight_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped = np.ascontiguousarray(weight_row, dtype=np.float32).reshape(
        GROUP_COUNT, GROUP_SIZE
    )
    absmax = np.max(np.abs(grouped), axis=1)
    raw_scale = absmax / float(Q4_MAX)
    scale_q64 = np.rint(raw_scale * float(1 << SCALE_FRAC)).astype(np.int64)
    scale_q64 = np.where((absmax > 0.0) & (scale_q64 == 0), 1, scale_q64)
    if np.any(scale_q64 < 0) or np.any(scale_q64 > np.iinfo(np.uint16).max):
        raise ValueError("embedding Q4 scale does not fit unsigned Q2.14")

    scale_q2_14 = scale_q64.astype(np.uint16)
    scale_used = scale_q2_14.astype(np.float32) / float(1 << SCALE_FRAC)
    safe_scale = np.where(scale_used > 0.0, scale_used, 1.0).astype(np.float32)
    q = np.rint(grouped / safe_scale[:, None])
    q = np.clip(q, Q4_MIN, Q4_MAX).astype(np.int8)
    return q.reshape(INPUT_SIZE), scale_q2_14


def pack_q4_words32(q4: np.ndarray) -> np.ndarray:
    nibbles = np.bitwise_and(q4.astype(np.int16), 0x0F).astype(np.uint32)
    grouped = nibbles.reshape(-1, 8)
    words = np.zeros(grouped.shape[0], dtype=np.uint32)
    for index in range(8):
        words |= grouped[:, index] << (index * 4)
    return words


def pack_scale_words32(scales: np.ndarray) -> np.ndarray:
    grouped = scales.astype(np.uint32).reshape(-1, 2)
    return grouped[:, 0] | (grouped[:, 1] << 16)


def write_hex(path: Path, values: np.ndarray, width_bits: int) -> None:
    mask = (1 << width_bits) - 1
    digits = (width_bits + 3) // 4
    lines = [f"{int(value) & mask:0{digits}x}" for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    token_id = int(input_ids[0, -1].item()) if args.token_id is None else int(args.token_id)

    embedding_weight = backbone.embed_tokens.weight.detach().to(
        dtype=model.lm_head.weight.dtype, device="cpu"
    )
    lm_head_weight = model.lm_head.weight.detach().to(device="cpu")
    vocab_size, input_size = map(int, embedding_weight.shape)
    if input_size != INPUT_SIZE:
        raise RuntimeError(f"expected embedding width {INPUT_SIZE}, got {input_size}")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"token id {token_id} is outside 0..{vocab_size - 1}")
    if not np.array_equal(embedding_weight[token_id].numpy(), lm_head_weight[token_id].numpy()):
        raise RuntimeError("embed_tokens and lm_head rows are not tied")

    fp32_row = embedding_weight[token_id].to(dtype=model.dtype).float().numpy()
    weight_q4, scale_q2_14 = quantize_row(fp32_row)
    group_scale = np.repeat(scale_q2_14.astype(np.int64), GROUP_SIZE)
    product_q14 = weight_q4.astype(np.int64) * group_scale
    output_q14_10 = np.right_shift(product_q14, DEQUANT_SHIFT).astype(np.int64)
    output_float = output_q14_10.astype(np.float64) / float(1 << OUTPUT_FRAC)
    max_abs_error = float(np.max(np.abs(output_float - fp32_row.astype(np.float64))))

    files = {
        "weight": output_dir / "embedding_weight_words32.hex",
        "scale": output_dir / "embedding_scale_words32.hex",
        "expected": output_dir / "embedding_expected_q14_10.hex",
        "token": output_dir / "embedding_token_id.hex",
        "meta": output_dir / "embedding_meta.json",
    }
    write_hex(files["weight"], pack_q4_words32(weight_q4), 32)
    write_hex(files["scale"], pack_scale_words32(scale_q2_14), 32)
    write_hex(files["expected"], output_q14_10, 32)
    write_hex(files["token"], np.array([token_id], dtype=np.int64), 32)

    meta = {
        "format_version": 1,
        "module": "q4_embedding_lookup",
        "prompt": PROMPT,
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id]),
        "vocab_size": vocab_size,
        "input_size": INPUT_SIZE,
        "group_size": GROUP_SIZE,
        "group_count": GROUP_COUNT,
        "weight_base_addr": WEIGHT_BASE_ADDR,
        "scale_base_addr": SCALE_BASE_ADDR,
        "weight_row_bytes": INPUT_SIZE // 2,
        "scale_row_bytes": GROUP_COUNT * 2,
        "output_format": "signed 32-bit Q14.10",
        "dequant_rule": "signed(q4 * scale_q2_14) arithmetic-right-shift 4",
        "max_abs_error_vs_fp32": max_abs_error,
        "output_min": int(np.min(output_q14_10)),
        "output_max": int(np.max(output_q14_10)),
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")

    print("Exported tied Q4 embedding lookup vectors")
    print(f"token={token_id} text={tokenizer.decode([token_id])!r}")
    print(f"output_dir={output_dir}")
    print(f"Q14.10 range={meta['output_min']}..{meta['output_max']}")
    print(f"max_abs_error_vs_fp32={max_abs_error:.9g}")


if __name__ == "__main__":
    main()
