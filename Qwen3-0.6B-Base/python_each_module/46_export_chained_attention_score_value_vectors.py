from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
Q_COUNT = NUM_Q_HEADS * HEAD_DIM
KV_COUNT = NUM_KV_HEADS * HEAD_DIM

IN_WIDTH = 24
IN_FRAC = 12
SCORE_WIDTH = 64
SCORE_FRAC = 2 * IN_FRAC
SCALE_WIDTH = 32
SCALE_FRAC = 31
EXP_WIDTH = 24
EXP_FRAC = 20
EXP_LUT_STEP_FRAC = 4
EXP_LUT_SIZE = (16 << EXP_LUT_STEP_FRAC) + 1
PROB_WIDTH = 24
PROB_FRAC = 16
OUT_WIDTH = 24
OUT_FRAC = 12


def read_hex_lines(path: Path, width_bits: int, *, signed: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    sign_bit = 1 << (width_bits - 1)
    full = 1 << width_bits
    values: list[int] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        raw = int(text.replace("_", ""), 16)
        if raw < 0 or raw >= full:
            raise ValueError(f"{path}:{lineno}: value out of {width_bits}-bit range: {text}")
        if signed and (raw & sign_bit):
            raw -= full
        values.append(raw)
    return np.asarray(values, dtype=np.int64)


def read_scalar(path: Path, width_bits: int, *, signed: bool) -> int:
    values = read_hex_lines(path, width_bits, signed=signed)
    if values.shape != (1,):
        raise RuntimeError(f"Expected one scalar in {path}, got {values.shape}")
    return int(values[0])


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    digits = (width_bits + 3) // 4
    lines = [f"{int(value) & mask:0{digits}x}" for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scalar(path: Path, value: int, width_bits: int) -> None:
    write_hex_lines(path, np.asarray([value], dtype=np.int64), width_bits)


def arithmetic_shift_right(value: int, shift: int) -> int:
    return int(value) >> shift


def saturate_signed(value: int, width_bits: int) -> tuple[int, bool]:
    low = -(1 << (width_bits - 1))
    high = (1 << (width_bits - 1)) - 1
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def build_exp_lut() -> np.ndarray:
    values = []
    for index in range(EXP_LUT_SIZE):
        x = -float(index) / float(1 << EXP_LUT_STEP_FRAC)
        values.append(int(round(math.exp(x) * float(1 << EXP_FRAC))))
    return np.asarray(values, dtype=np.int64)


def score_diff_to_lut_index(max_score: int, score: int) -> int:
    neg_diff = max_score - score
    if neg_diff <= 0:
        return 0
    shift = SCORE_FRAC - EXP_LUT_STEP_FRAC
    rounded = (neg_diff + (1 << (shift - 1))) >> shift
    return int(min(rounded, EXP_LUT_SIZE - 1))


def fixed_softmax_probs(scores_q24_24: np.ndarray, exp_lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.zeros_like(scores_q24_24, dtype=np.int64)
    lut_indices = np.zeros_like(scores_q24_24, dtype=np.int64)
    for head in range(scores_q24_24.shape[0]):
        head_scores = scores_q24_24[head]
        max_score = int(np.max(head_scores))
        exp_values = []
        for position, score in enumerate(head_scores):
            index = score_diff_to_lut_index(max_score, int(score))
            lut_indices[head, position] = index
            exp_values.append(int(exp_lut[index]))
        exp_sum = sum(exp_values)
        if exp_sum <= 0:
            raise RuntimeError("softmax exp_sum underflowed to zero")
        for position, exp_value in enumerate(exp_values):
            probs[head, position] = (exp_value << PROB_FRAC) // exp_sum
    return probs, lut_indices


def fixed_value_accum(probs_q0_16: np.ndarray, v_cache_q12_12: np.ndarray) -> tuple[np.ndarray, bool]:
    output = np.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=np.int64)
    saturation = False
    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // KV_REPEAT
        for dim in range(HEAD_DIM):
            acc = 0
            for position in range(probs_q0_16.shape[1]):
                acc += int(probs_q0_16[q_head, position]) * int(v_cache_q12_12[position, kv_head, dim])
            shifted = arithmetic_shift_right(acc, PROB_FRAC)
            output_value, did_saturate = saturate_signed(shifted, OUT_WIDTH)
            output[q_head, dim] = output_value
            saturation = saturation or did_saturate
    return output, saturation


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chained Layer1 attention score/value vectors.")
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--score-prefix", type=str, default="layer1_chained_attention_score_stage_real")
    parser.add_argument("--value-prefix", type=str, default="layer1_chained_attention_softmax_value_stage_real")
    parser.add_argument(
        "--q-rope-hex",
        type=Path,
        default=SIM_VECTOR_DIR / "qmap_layer1_chained_attention_frontend_q_rope_expected_words32.hex",
    )
    parser.add_argument(
        "--base-k-cache-hex",
        type=Path,
        default=SIM_VECTOR_DIR / "layer1_attention_score_stage_real_k_cache.hex",
    )
    parser.add_argument(
        "--base-v-cache-hex",
        type=Path,
        default=SIM_VECTOR_DIR / "layer1_attention_softmax_value_stage_real_v_cache.hex",
    )
    parser.add_argument(
        "--current-k-hex",
        type=Path,
        default=SIM_VECTOR_DIR / "layer1_chained_kv_cache_append_real_k_input.hex",
    )
    parser.add_argument(
        "--current-v-hex",
        type=Path,
        default=SIM_VECTOR_DIR / "layer1_chained_kv_cache_append_real_v_input.hex",
    )
    parser.add_argument(
        "--score-scale-hex",
        type=Path,
        default=SIM_VECTOR_DIR / "layer1_attention_score_stage_real_score_scale_q0_31.hex",
    )
    parser.add_argument(
        "--source-meta",
        type=Path,
        default=SIM_VECTOR_DIR / "layer1_chained_qk_norm_rope_stage_128_real_meta.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_meta = json.loads(args.source_meta.read_text(encoding="utf-8"))
    selected_position = int(source_meta["selected_position"])
    selected_token_id = int(source_meta["selected_token_id"])
    selected_token_text = str(source_meta["selected_token_text"])
    token_ids = [int(value) for value in source_meta["token_ids"]]

    q_current = read_hex_lines(args.q_rope_hex, 32, signed=True)
    if q_current.shape != (Q_COUNT,):
        raise RuntimeError(f"q_rope shape mismatch: {q_current.shape}, expected {(Q_COUNT,)}")
    q_current = q_current.reshape(NUM_Q_HEADS, HEAD_DIM)

    cache_length = selected_position + 1
    k_cache = read_hex_lines(args.base_k_cache_hex, IN_WIDTH, signed=True)
    v_cache = read_hex_lines(args.base_v_cache_hex, IN_WIDTH, signed=True)
    expected_cache_words = cache_length * KV_COUNT
    if k_cache.shape != (expected_cache_words,):
        raise RuntimeError(f"k_cache shape mismatch: {k_cache.shape}, expected {(expected_cache_words,)}")
    if v_cache.shape != (expected_cache_words,):
        raise RuntimeError(f"v_cache shape mismatch: {v_cache.shape}, expected {(expected_cache_words,)}")
    k_cache = k_cache.reshape(cache_length, NUM_KV_HEADS, HEAD_DIM)
    v_cache = v_cache.reshape(cache_length, NUM_KV_HEADS, HEAD_DIM)

    current_k = read_hex_lines(args.current_k_hex, IN_WIDTH, signed=True).reshape(NUM_KV_HEADS, HEAD_DIM)
    current_v = read_hex_lines(args.current_v_hex, IN_WIDTH, signed=True).reshape(NUM_KV_HEADS, HEAD_DIM)
    k_cache[selected_position] = current_k
    v_cache[selected_position] = current_v

    score_scale_q0_31 = read_scalar(args.score_scale_hex, SCALE_WIDTH, signed=True)
    expected_raw = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.int64)
    expected_scaled = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.int64)
    expected_q_head = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.uint8)
    expected_kv_head = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.uint8)
    expected_position = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.uint8)

    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // KV_REPEAT
        for position in range(cache_length):
            dot = int(
                np.sum(
                    q_current[q_head].astype(np.int64) * k_cache[position, kv_head].astype(np.int64),
                    dtype=np.int64,
                )
            )
            expected_raw[q_head, position] = dot
            expected_scaled[q_head, position] = arithmetic_shift_right(dot * score_scale_q0_31, SCALE_FRAC)
            expected_q_head[q_head, position] = q_head
            expected_kv_head[q_head, position] = kv_head
            expected_position[q_head, position] = position

    exp_lut = build_exp_lut()
    probs_q0_16, lut_indices = fixed_softmax_probs(expected_scaled, exp_lut)
    attn_out_q12_12, value_saturation = fixed_value_accum(probs_q0_16, v_cache)

    score_files = {
        "q_input": SIM_VECTOR_DIR / f"{args.score_prefix}_q_input.hex",
        "k_cache": SIM_VECTOR_DIR / f"{args.score_prefix}_k_cache.hex",
        "score_scale": SIM_VECTOR_DIR / f"{args.score_prefix}_score_scale_q0_31.hex",
        "cache_length": SIM_VECTOR_DIR / f"{args.score_prefix}_cache_length.hex",
        "expected_raw": SIM_VECTOR_DIR / f"{args.score_prefix}_expected_raw.hex",
        "expected_scaled": SIM_VECTOR_DIR / f"{args.score_prefix}_expected_scaled.hex",
        "expected_q_head": SIM_VECTOR_DIR / f"{args.score_prefix}_expected_q_head.hex",
        "expected_kv_head": SIM_VECTOR_DIR / f"{args.score_prefix}_expected_kv_head.hex",
        "expected_position": SIM_VECTOR_DIR / f"{args.score_prefix}_expected_position.hex",
        "meta": SIM_VECTOR_DIR / f"{args.score_prefix}_meta.json",
    }
    value_files = {
        "score_input": SIM_VECTOR_DIR / f"{args.value_prefix}_score_input.hex",
        "score_q_head": SIM_VECTOR_DIR / f"{args.value_prefix}_score_q_head.hex",
        "score_kv_head": SIM_VECTOR_DIR / f"{args.value_prefix}_score_kv_head.hex",
        "score_position": SIM_VECTOR_DIR / f"{args.value_prefix}_score_position.hex",
        "v_cache": SIM_VECTOR_DIR / f"{args.value_prefix}_v_cache.hex",
        "exp_lut": SIM_VECTOR_DIR / f"{args.value_prefix}_exp_lut.hex",
        "cache_length": SIM_VECTOR_DIR / f"{args.value_prefix}_cache_length.hex",
        "expected_prob": SIM_VECTOR_DIR / f"{args.value_prefix}_expected_prob.hex",
        "expected_lut_index": SIM_VECTOR_DIR / f"{args.value_prefix}_expected_lut_index.hex",
        "expected_out": SIM_VECTOR_DIR / f"{args.value_prefix}_expected_out.hex",
        "meta": SIM_VECTOR_DIR / f"{args.value_prefix}_meta.json",
    }

    write_hex_lines(score_files["q_input"], q_current, IN_WIDTH)
    write_hex_lines(score_files["k_cache"], k_cache, IN_WIDTH)
    write_scalar(score_files["score_scale"], score_scale_q0_31, SCALE_WIDTH)
    write_scalar(score_files["cache_length"], cache_length, 16)
    write_hex_lines(score_files["expected_raw"], expected_raw, SCORE_WIDTH)
    write_hex_lines(score_files["expected_scaled"], expected_scaled, SCORE_WIDTH)
    write_hex_lines(score_files["expected_q_head"], expected_q_head, 8)
    write_hex_lines(score_files["expected_kv_head"], expected_kv_head, 8)
    write_hex_lines(score_files["expected_position"], expected_position, 8)

    write_hex_lines(value_files["score_input"], expected_scaled, SCORE_WIDTH)
    write_hex_lines(value_files["score_q_head"], expected_q_head, 8)
    write_hex_lines(value_files["score_kv_head"], expected_kv_head, 8)
    write_hex_lines(value_files["score_position"], expected_position, 8)
    write_hex_lines(value_files["v_cache"], v_cache, IN_WIDTH)
    write_hex_lines(value_files["exp_lut"], exp_lut, EXP_WIDTH)
    write_scalar(value_files["cache_length"], cache_length, 16)
    write_hex_lines(value_files["expected_prob"], probs_q0_16, PROB_WIDTH)
    write_hex_lines(value_files["expected_lut_index"], lut_indices, 16)
    write_hex_lines(value_files["expected_out"], attn_out_q12_12, OUT_WIDTH)

    common_meta: dict[str, Any] = {
        "format_version": 1,
        "layer_id": args.layer_id,
        "prompt": source_meta.get("prompt"),
        "token_ids": token_ids,
        "selected_position": selected_position,
        "selected_token_id": selected_token_id,
        "selected_token_text": selected_token_text,
        "source": {
            "q_rope_hex": relpath(args.q_rope_hex),
            "base_k_cache_hex": relpath(args.base_k_cache_hex),
            "base_v_cache_hex": relpath(args.base_v_cache_hex),
            "current_k_hex": relpath(args.current_k_hex),
            "current_v_hex": relpath(args.current_v_hex),
            "score_scale_hex": relpath(args.score_scale_hex),
        },
        "shape": {
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "kv_repeat": KV_REPEAT,
            "cache_length": cache_length,
            "score_count": int(expected_scaled.size),
            "output_count": int(attn_out_q12_12.size),
        },
        "fixed_point": {
            "attention_scale_q0_31": score_scale_q0_31,
            "score_frac": SCORE_FRAC,
            "prob_frac": PROB_FRAC,
            "out_frac": OUT_FRAC,
        },
    }
    score_meta: dict[str, Any] = {
        **common_meta,
        "name": args.score_prefix,
        "module": f"layer{args.layer_id}.self_attn.attention_score_stage",
        "files": {name: relpath(path) for name, path in score_files.items() if name != "meta"},
        "debug": {
            "q_current_q12_12_max_abs": int(np.max(np.abs(q_current))),
            "k_cache_q12_12_max_abs": int(np.max(np.abs(k_cache))),
            "raw_score_min": int(np.min(expected_raw)),
            "raw_score_max": int(np.max(expected_raw)),
            "scaled_score_min": int(np.min(expected_scaled)),
            "scaled_score_max": int(np.max(expected_scaled)),
        },
    }
    value_meta: dict[str, Any] = {
        **common_meta,
        "name": args.value_prefix,
        "module": f"layer{args.layer_id}.self_attn.attention_softmax_value_stage",
        "files": {name: relpath(path) for name, path in value_files.items() if name != "meta"},
        "debug": {
            "value_saturation": bool(value_saturation),
            "prob_min": int(np.min(probs_q0_16)),
            "prob_max": int(np.max(probs_q0_16)),
            "attn_out_q12_12_max_abs": int(np.max(np.abs(attn_out_q12_12))),
        },
    }
    score_files["meta"].write_text(json.dumps(score_meta, indent=2) + "\n", encoding="utf-8")
    value_files["meta"].write_text(json.dumps(value_meta, indent=2) + "\n", encoding="utf-8")

    print("Exported chained attention score/value vectors")
    print("=" * 80)
    print(f"Layer/position: layer={args.layer_id}, position={selected_position}")
    print(f"Cache length:   {cache_length}")
    print(f"Score prefix:   {args.score_prefix}")
    print(f"Value prefix:   {args.value_prefix}")
    print(f"Attn out max:   {int(np.max(np.abs(attn_out_q12_12)))}")
    print(f"Value sat:      {int(value_saturation)}")


if __name__ == "__main__":
    main()
