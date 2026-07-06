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
    load_model,
    manual_apply_rope,
    manual_cached_attention,
    manual_linear_from_weight,
    manual_repeat_kv,
    manual_rms_norm,
    manual_rope_cos_sin,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
QK_EXPORTER_PATH = Path(__file__).with_name("22_export_qk_norm_rope_fixed_vectors.py")

DEFAULT_PREFIX = "attention_softmax_value_stage_real"

HIDDEN_SIZE = 1024
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_ROWS = NUM_Q_HEADS * HEAD_DIM
K_ROWS = NUM_KV_HEADS * HEAD_DIM
V_ROWS = NUM_KV_HEADS * HEAD_DIM
KV_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS

IN_WIDTH = 24
IN_FRAC = 12
ACT_QUANT_WIDTH = 16
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


def load_qk_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qk_norm_rope_exporter", QK_EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {QK_EXPORTER_PATH}")
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


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def quantize_activation_q4_12(values: np.ndarray) -> np.ndarray:
    low = -(1 << (ACT_QUANT_WIDTH - 1))
    high = (1 << (ACT_QUANT_WIDTH - 1)) - 1
    scaled = np.rint(values.astype(np.float64) * float(1 << IN_FRAC))
    return np.clip(scaled, low, high).astype(np.int64)


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
    return np.array(values, dtype=np.int64)


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


def fixed_value_accum(
    probs_q0_16: np.ndarray,
    v_cache_q12_12: np.ndarray,
) -> tuple[np.ndarray, bool]:
    output = np.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=np.int64)
    saturation = False

    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // KV_REPEAT
        for dim in range(HEAD_DIM):
            acc = 0
            for position in range(probs_q0_16.shape[1]):
                acc += int(probs_q0_16[q_head, position]) * int(
                    v_cache_q12_12[position, kv_head, dim]
                )
            shifted = arithmetic_shift_right(acc, PROB_FRAC)
            output_value, did_saturate = saturate_signed(shifted, OUT_WIDTH)
            output[q_head, dim] = output_value
            saturation = saturation or did_saturate

    return output, saturation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-point attention softmax/value RTL test vectors.")
    parser.add_argument("--layer-id", type=int, default=0, help="Decoder layer index to export")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Q4 QKV artifact npz. Defaults to qkv_layer{layer_id}_last_token_q4.npz.",
    )
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help="Output vector file prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qk = load_qk_exporter()

    q4_path = args.input
    if q4_path is None:
        q4_path = Q4_VECTOR_DIR / f"qkv_layer{args.layer_id}_last_token_q4.npz"
    if not q4_path.is_file():
        raise FileNotFoundError(f"Missing {q4_path}. Run the matching QKV Q4 vector exporter first.")

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    data = np.load(q4_path)
    selected_position = int(np.asarray(data["prompt_position"]).item())
    selected_token_id = int(np.asarray(data["token_id"]).item())
    if selected_position != prompt_len - 1:
        raise RuntimeError(
            f"Q4 artifact position {selected_position} does not match prompt length {prompt_len}"
        )
    if selected_token_id != int(input_ids[0, selected_position].item()):
        raise RuntimeError("Q4 artifact token id does not match tokenizer output")

    if args.layer_id < 0 or args.layer_id >= len(backbone.layers):
        raise ValueError(f"layer_id out of range: {args.layer_id}")
    layer = backbone.layers[args.layer_id]
    attn = layer.self_attn
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = layer.input_layernorm.register_forward_hook(capture_module_io(records, "input_norm"))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()

    input_norm_fp32 = np.ascontiguousarray(
        require_output(records, "input_norm")[0].to(torch.float32).numpy(),
        dtype=np.float32,
    )
    input_norm_q12_12 = quantize_activation_q4_12(input_norm_fp32)
    if not np.array_equal(input_norm_q12_12[selected_position], data["input_norm_q4_12"]):
        raise RuntimeError("Selected-position input_norm quantization drifted from Q4 artifact")

    q_gamma_float = qk.as_float_array(attn.q_norm.weight)
    k_gamma_float = qk.as_float_array(attn.k_norm.weight)
    q_gamma_q8_7, q_gamma_saturation_count = qk.quantize_signed(
        q_gamma_float,
        qk.GAMMA_WIDTH,
        qk.GAMMA_FRAC,
    )
    k_gamma_q8_7, k_gamma_saturation_count = qk.quantize_signed(
        k_gamma_float,
        qk.GAMMA_WIDTH,
        qk.GAMMA_FRAC,
    )

    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros((1, prompt_len, HIDDEN_SIZE), dtype=torch.float32)
    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, dummy_x, position_ids)

    q_rope_by_pos = np.zeros((prompt_len, NUM_Q_HEADS, HEAD_DIM), dtype=np.int64)
    k_rope_by_pos = np.zeros((prompt_len, NUM_KV_HEADS, HEAD_DIM), dtype=np.int64)
    v_cache_q12_12 = np.zeros((prompt_len, NUM_KV_HEADS, HEAD_DIM), dtype=np.int64)
    norm_saturation = False
    rope_saturation = False
    cos_saturation_total = 0
    sin_saturation_total = 0

    for position in range(prompt_len):
        q_input_flat, _q_row_sum_q26 = qk.compute_projection_q12_12(
            activation_q12_12=input_norm_q12_12[position],
            packed_weight=data["q_weight_q4_packed"],
            scale_q2_14=data["q_scale_q2_14"],
            row_count=Q_ROWS,
        )
        k_input_flat, _k_row_sum_q26 = qk.compute_projection_q12_12(
            activation_q12_12=input_norm_q12_12[position],
            packed_weight=data["k_weight_q4_packed"],
            scale_q2_14=data["k_scale_q2_14"],
            row_count=K_ROWS,
        )
        v_input_flat, _v_row_sum_q26 = qk.compute_projection_q12_12(
            activation_q12_12=input_norm_q12_12[position],
            packed_weight=data["v_weight_q4_packed"],
            scale_q2_14=data["v_scale_q2_14"],
            row_count=V_ROWS,
        )

        q_input_heads = q_input_flat.reshape(NUM_Q_HEADS, HEAD_DIM)
        k_input_heads = k_input_flat.reshape(NUM_KV_HEADS, HEAD_DIM)
        v_cache_q12_12[position] = v_input_flat.reshape(NUM_KV_HEADS, HEAD_DIM)

        q_norm_q12_12, _q_norm_debug, q_norm_saturation = qk.fixed_rmsnorm_heads(
            q_input_heads,
            q_gamma_q8_7,
        )
        k_norm_q12_12, _k_norm_debug, k_norm_saturation = qk.fixed_rmsnorm_heads(
            k_input_heads,
            k_gamma_q8_7,
        )
        norm_saturation = norm_saturation or q_norm_saturation or k_norm_saturation

        cos_float = qk.as_float_array(cos[0, position, :])
        sin_float = qk.as_float_array(sin[0, position, :])
        cos_q1_15, cos_saturation_count = qk.quantize_signed(
            cos_float,
            qk.TRIG_WIDTH,
            qk.TRIG_FRAC,
        )
        sin_q1_15, sin_saturation_count = qk.quantize_signed(
            sin_float,
            qk.TRIG_WIDTH,
            qk.TRIG_FRAC,
        )
        cos_saturation_total += int(cos_saturation_count)
        sin_saturation_total += int(sin_saturation_count)

        q_rope_q12_12, q_rope_saturation = qk.fixed_rope_reference(
            q_norm_q12_12,
            cos_q1_15,
            sin_q1_15,
        )
        k_rope_q12_12, k_rope_saturation = qk.fixed_rope_reference(
            k_norm_q12_12,
            cos_q1_15,
            sin_q1_15,
        )
        rope_saturation = rope_saturation or q_rope_saturation or k_rope_saturation
        q_rope_by_pos[position] = q_rope_q12_12
        k_rope_by_pos[position] = k_rope_q12_12

    q_current = q_rope_by_pos[selected_position]
    k_cache = k_rope_by_pos[: selected_position + 1]
    v_cache = v_cache_q12_12[: selected_position + 1]
    cache_length = k_cache.shape[0]

    score_scale_q0_31 = int(round(float(attn.scaling) * float(1 << SCALE_FRAC)))
    if score_scale_q0_31 <= 0 or score_scale_q0_31 >= (1 << (SCALE_WIDTH - 1)):
        raise RuntimeError(f"Unexpected attention scale quantization: {score_scale_q0_31}")

    scaled_scores = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.int64)
    score_q_head = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.uint8)
    score_kv_head = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.uint8)
    score_position = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.uint8)

    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // KV_REPEAT
        for position in range(cache_length):
            dot = int(
                np.sum(
                    q_current[q_head].astype(np.int64)
                    * k_cache[position, kv_head].astype(np.int64),
                    dtype=np.int64,
                )
            )
            scaled_scores[q_head, position] = arithmetic_shift_right(
                dot * score_scale_q0_31,
                SCALE_FRAC,
            )
            score_q_head[q_head, position] = q_head
            score_kv_head[q_head, position] = kv_head
            score_position[q_head, position] = position

    exp_lut = build_exp_lut()
    probs_q0_16, lut_indices = fixed_softmax_probs(scaled_scores, exp_lut)
    attn_out_q12_12, value_saturation = fixed_value_accum(probs_q0_16, v_cache)

    with torch.no_grad():
        input_norm_t = torch.from_numpy(input_norm_fp32).unsqueeze(0)
        q_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn.q_proj.weight))
        k_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn.k_proj.weight))
        v_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn.v_proj.weight))
        q_view = q_flat_fp32.view(batch_size, prompt_len, NUM_Q_HEADS, HEAD_DIM)
        k_view = k_flat_fp32.view(batch_size, prompt_len, NUM_KV_HEADS, HEAD_DIM)
        v_view = v_flat_fp32.view(batch_size, prompt_len, NUM_KV_HEADS, HEAD_DIM)
        q_norm = manual_rms_norm(q_view, tensor_to_float32(attn.q_norm.weight), rms_eps)
        k_norm = manual_rms_norm(k_view, tensor_to_float32(attn.k_norm.weight), rms_eps)
        q_states = q_norm.transpose(1, 2)
        k_states = k_norm.transpose(1, 2)
        v_states = v_view.transpose(1, 2)
        cos_ref, sin_ref = manual_rope_cos_sin(backbone.rotary_emb, input_norm_t, position_ids)
        q_rope_fp32, k_rope_fp32 = manual_apply_rope(q_states, k_states, cos_ref, sin_ref)
        k_repeated_fp32 = manual_repeat_kv(k_rope_fp32, KV_REPEAT)
        v_repeated_fp32 = manual_repeat_kv(v_states, KV_REPEAT)
        attn_fp32, prob_fp32 = manual_cached_attention(
            q_rope_fp32[:, :, selected_position : selected_position + 1, :],
            k_repeated_fp32[:, :, :cache_length, :],
            v_repeated_fp32[:, :, :cache_length, :],
            float(attn.scaling),
        )
        attn_fp32_np = np.ascontiguousarray(attn_fp32[0, :, 0, :].numpy(), dtype=np.float32)
        prob_fp32_np = np.ascontiguousarray(prob_fp32[0, :, 0, :].numpy(), dtype=np.float32)

    attn_fixed_float = fixed_to_float(attn_out_q12_12, OUT_FRAC)
    attn_diff = attn_fixed_float.astype(np.float64) - attn_fp32_np.astype(np.float64)
    prob_fixed_float = fixed_to_float(probs_q0_16, PROB_FRAC)
    prob_diff = prob_fixed_float.astype(np.float64) - prob_fp32_np.astype(np.float64)

    files = {
        "score_input": SIM_VECTOR_DIR / f"{args.prefix}_score_input.hex",
        "score_q_head": SIM_VECTOR_DIR / f"{args.prefix}_score_q_head.hex",
        "score_kv_head": SIM_VECTOR_DIR / f"{args.prefix}_score_kv_head.hex",
        "score_position": SIM_VECTOR_DIR / f"{args.prefix}_score_position.hex",
        "v_cache": SIM_VECTOR_DIR / f"{args.prefix}_v_cache.hex",
        "exp_lut": SIM_VECTOR_DIR / f"{args.prefix}_exp_lut.hex",
        "cache_length": SIM_VECTOR_DIR / f"{args.prefix}_cache_length.hex",
        "expected_prob": SIM_VECTOR_DIR / f"{args.prefix}_expected_prob.hex",
        "expected_lut_index": SIM_VECTOR_DIR / f"{args.prefix}_expected_lut_index.hex",
        "expected_out": SIM_VECTOR_DIR / f"{args.prefix}_expected_out.hex",
        "meta": SIM_VECTOR_DIR / f"{args.prefix}_meta.json",
    }

    write_hex_lines(files["score_input"], scaled_scores, SCORE_WIDTH)
    write_hex_lines(files["score_q_head"], score_q_head, 8)
    write_hex_lines(files["score_kv_head"], score_kv_head, 8)
    write_hex_lines(files["score_position"], score_position, 8)
    write_hex_lines(files["v_cache"], v_cache, IN_WIDTH)
    write_hex_lines(files["exp_lut"], exp_lut, EXP_WIDTH)
    write_scalar(files["cache_length"], cache_length, 16)
    write_hex_lines(files["expected_prob"], probs_q0_16, PROB_WIDTH)
    write_hex_lines(files["expected_lut_index"], lut_indices, 16)
    write_hex_lines(files["expected_out"], attn_out_q12_12, OUT_WIDTH)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": args.prefix,
        "module": f"layer{args.layer_id}.self_attn.attention_softmax_value_stage",
        "layer_id": args.layer_id,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": selected_position,
        "selected_token_id": selected_token_id,
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "shape": {
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "kv_repeat": KV_REPEAT,
            "cache_length": cache_length,
            "score_count": int(scaled_scores.size),
            "v_request_count": int(NUM_Q_HEADS * HEAD_DIM * cache_length),
            "output_count": int(attn_out_q12_12.size),
        },
        "formats": {
            "score_input": f"signed {SCORE_WIDTH}-bit Q24.{SCORE_FRAC}",
            "v_cache": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC}",
            "exp_lut": f"unsigned {EXP_WIDTH}-bit UQ0.{EXP_FRAC}",
            "prob": f"unsigned {PROB_WIDTH}-bit UQ0.{PROB_FRAC}",
            "output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
        },
        "fixed_point": {
            "exp_lut_step": f"1/{1 << EXP_LUT_STEP_FRAC}",
            "exp_lut_max_negative": 16.0,
            "exp_lut_size": EXP_LUT_SIZE,
            "prob_frac": PROB_FRAC,
            "attention_scale_float": float(attn.scaling),
            "attention_scale_q0_31": score_scale_q0_31,
        },
        "saturation": {
            "q_gamma_quantize_count": int(q_gamma_saturation_count),
            "k_gamma_quantize_count": int(k_gamma_saturation_count),
            "cos_quantize_count_total": int(cos_saturation_total),
            "sin_quantize_count_total": int(sin_saturation_total),
            "norm_output": bool(norm_saturation),
            "rope_output": bool(rope_saturation),
            "value_output": bool(value_saturation),
        },
        "debug": {
            "scaled_score_float_min": float(np.min(fixed_to_float(scaled_scores, SCORE_FRAC))),
            "scaled_score_float_max": float(np.max(fixed_to_float(scaled_scores, SCORE_FRAC))),
            "prob_row_sum_min": int(np.min(np.sum(probs_q0_16, axis=1))),
            "prob_row_sum_max": int(np.max(np.sum(probs_q0_16, axis=1))),
            "prob_nonzero_count": int(np.count_nonzero(probs_q0_16)),
            "lut_index_min": int(np.min(lut_indices)),
            "lut_index_max": int(np.max(lut_indices)),
            "v_cache_q12_12_max_abs": int(np.max(np.abs(v_cache))),
            "attn_out_q12_12_max_abs": int(np.max(np.abs(attn_out_q12_12))),
            "fixed_prob_vs_fp32_max_abs_diff": float(np.max(np.abs(prob_diff))),
            "fixed_prob_vs_fp32_mean_abs_diff": float(np.mean(np.abs(prob_diff))),
            "fixed_attn_out_vs_fp32_max_abs_diff": float(np.max(np.abs(attn_diff))),
            "fixed_attn_out_vs_fp32_mean_abs_diff": float(np.mean(np.abs(attn_diff))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point attention softmax/value RTL test vectors")
    print("=" * 80)
    print(f"Module: layer{args.layer_id}.self_attn.attention_softmax_value_stage")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Cache length: {cache_length}")
    print(f"Score count: {scaled_scores.size}")
    print(f"V requests: {NUM_Q_HEADS * HEAD_DIM * cache_length}")
    print(f"Output count: {attn_out_q12_12.size}")
    print(f"Score input:      {files['score_input']}")
    print(f"V cache:          {files['v_cache']}")
    print(f"Expected prob:    {files['expected_prob']}")
    print(f"Expected output:  {files['expected_out']}")
    print(f"Debug meta:       {files['meta']}")
    print(
        "Fixed attn output vs FP32 reference: "
        f"max_abs={meta['debug']['fixed_attn_out_vs_fp32_max_abs_diff']:.8g} "
        f"mean_abs={meta['debug']['fixed_attn_out_vs_fp32_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
