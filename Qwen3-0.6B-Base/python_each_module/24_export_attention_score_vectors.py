from __future__ import annotations

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

PREFIX = "attention_score_stage_real"
MODULE_NAME = "layer0.self_attn.attention_score_stage"

HIDDEN_SIZE = 1024
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_ROWS = NUM_Q_HEADS * HEAD_DIM
K_ROWS = NUM_KV_HEADS * HEAD_DIM
KV_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS

IN_WIDTH = 24
IN_FRAC = 12
ACT_QUANT_WIDTH = 16
SCORE_WIDTH = 64
SCORE_FRAC = 2 * IN_FRAC
SCALE_WIDTH = 32
SCALE_FRAC = 31


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


def main() -> None:
    qk = load_qk_exporter()

    q4_path = Q4_VECTOR_DIR / "qkv_layer0_last_token_q4.npz"
    if not q4_path.is_file():
        raise FileNotFoundError(f"Missing {q4_path}. Run 13_export_q4_gemv_vectors.py first.")

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
    if NUM_Q_HEADS % NUM_KV_HEADS != 0:
        raise RuntimeError("NUM_Q_HEADS must be an integer multiple of NUM_KV_HEADS")

    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = layer0.input_layernorm.register_forward_hook(capture_module_io(records, "input_norm"))
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

    q_gamma_float = qk.as_float_array(attn0.q_norm.weight)
    k_gamma_float = qk.as_float_array(attn0.k_norm.weight)
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

        q_input_heads = q_input_flat.reshape(NUM_Q_HEADS, HEAD_DIM)
        k_input_heads = k_input_flat.reshape(NUM_KV_HEADS, HEAD_DIM)
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
    cache_length = k_cache.shape[0]

    score_scale_q0_31 = int(round(float(attn0.scaling) * float(1 << SCALE_FRAC)))
    if score_scale_q0_31 <= 0 or score_scale_q0_31 >= (1 << (SCALE_WIDTH - 1)):
        raise RuntimeError(f"Unexpected attention scale quantization: {score_scale_q0_31}")

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
                    q_current[q_head].astype(np.int64)
                    * k_cache[position, kv_head].astype(np.int64),
                    dtype=np.int64,
                )
            )
            scaled = arithmetic_shift_right(dot * score_scale_q0_31, SCALE_FRAC)
            expected_raw[q_head, position] = dot
            expected_scaled[q_head, position] = scaled
            expected_q_head[q_head, position] = q_head
            expected_kv_head[q_head, position] = kv_head
            expected_position[q_head, position] = position

    with torch.no_grad():
        input_norm_t = torch.from_numpy(input_norm_fp32).unsqueeze(0)
        q_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn0.q_proj.weight))
        k_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn0.k_proj.weight))
        q_view = q_flat_fp32.view(batch_size, prompt_len, NUM_Q_HEADS, HEAD_DIM)
        k_view = k_flat_fp32.view(batch_size, prompt_len, NUM_KV_HEADS, HEAD_DIM)
        q_norm = manual_rms_norm(q_view, tensor_to_float32(attn0.q_norm.weight), rms_eps)
        k_norm = manual_rms_norm(k_view, tensor_to_float32(attn0.k_norm.weight), rms_eps)
        q_states = q_norm.transpose(1, 2)
        k_states = k_norm.transpose(1, 2)
        cos_ref, sin_ref = manual_rope_cos_sin(backbone.rotary_emb, input_norm_t, position_ids)
        q_rope_fp32, k_rope_fp32 = manual_apply_rope(q_states, k_states, cos_ref, sin_ref)
        k_repeated_fp32 = manual_repeat_kv(k_rope_fp32, KV_REPEAT)
        fp32_scores = (
            torch.matmul(
                q_rope_fp32[:, :, selected_position : selected_position + 1, :],
                k_repeated_fp32[:, :, :cache_length, :].transpose(2, 3),
            )
            * float(attn0.scaling)
        )
        fp32_scores_np = np.ascontiguousarray(fp32_scores[0, :, 0, :].numpy(), dtype=np.float32)

    fixed_scaled_float = fixed_to_float(expected_scaled, SCORE_FRAC)
    score_diff = fixed_scaled_float.astype(np.float64) - fp32_scores_np.astype(np.float64)

    files = {
        "q_input": SIM_VECTOR_DIR / f"{PREFIX}_q_input.hex",
        "k_cache": SIM_VECTOR_DIR / f"{PREFIX}_k_cache.hex",
        "score_scale": SIM_VECTOR_DIR / f"{PREFIX}_score_scale_q0_31.hex",
        "cache_length": SIM_VECTOR_DIR / f"{PREFIX}_cache_length.hex",
        "expected_raw": SIM_VECTOR_DIR / f"{PREFIX}_expected_raw.hex",
        "expected_scaled": SIM_VECTOR_DIR / f"{PREFIX}_expected_scaled.hex",
        "expected_q_head": SIM_VECTOR_DIR / f"{PREFIX}_expected_q_head.hex",
        "expected_kv_head": SIM_VECTOR_DIR / f"{PREFIX}_expected_kv_head.hex",
        "expected_position": SIM_VECTOR_DIR / f"{PREFIX}_expected_position.hex",
        "meta": SIM_VECTOR_DIR / f"{PREFIX}_meta.json",
    }

    write_hex_lines(files["q_input"], q_current, IN_WIDTH)
    write_hex_lines(files["k_cache"], k_cache, IN_WIDTH)
    write_scalar(files["score_scale"], score_scale_q0_31, SCALE_WIDTH)
    write_scalar(files["cache_length"], cache_length, 16)
    write_hex_lines(files["expected_raw"], expected_raw, SCORE_WIDTH)
    write_hex_lines(files["expected_scaled"], expected_scaled, SCORE_WIDTH)
    write_hex_lines(files["expected_q_head"], expected_q_head, 8)
    write_hex_lines(files["expected_kv_head"], expected_kv_head, 8)
    write_hex_lines(files["expected_position"], expected_position, 8)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": PREFIX,
        "module": MODULE_NAME,
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
            "score_count": int(expected_raw.size),
            "k_request_count": int(expected_raw.size * HEAD_DIM),
        },
        "formats": {
            "q_input": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC} current-token Q after q_norm and RoPE",
            "k_cache": f"signed {IN_WIDTH}-bit Q12.{IN_FRAC} cached K after k_norm and RoPE",
            "score_raw": f"signed {SCORE_WIDTH}-bit Q24.{SCORE_FRAC} dot-product sum",
            "score_scaled": (
                f"signed {SCORE_WIDTH}-bit Q24.{SCORE_FRAC}; "
                f"score_raw * Q0.{SCALE_FRAC} attention scale >> {SCALE_FRAC}"
            ),
        },
        "fixed_point": {
            "attention_scale_float": float(attn0.scaling),
            "attention_scale_q0_31": score_scale_q0_31,
            "score_frac": SCORE_FRAC,
            "scale_frac": SCALE_FRAC,
        },
        "saturation": {
            "q_gamma_quantize_count": int(q_gamma_saturation_count),
            "k_gamma_quantize_count": int(k_gamma_saturation_count),
            "cos_quantize_count_total": int(cos_saturation_total),
            "sin_quantize_count_total": int(sin_saturation_total),
            "norm_output": bool(norm_saturation),
            "rope_output": bool(rope_saturation),
        },
        "debug": {
            "q_current_q12_12_max_abs": int(np.max(np.abs(q_current))),
            "k_cache_q12_12_max_abs": int(np.max(np.abs(k_cache))),
            "raw_score_min": int(np.min(expected_raw)),
            "raw_score_max": int(np.max(expected_raw)),
            "scaled_score_min": int(np.min(expected_scaled)),
            "scaled_score_max": int(np.max(expected_scaled)),
            "scaled_score_float_min": float(np.min(fixed_scaled_float)),
            "scaled_score_float_max": float(np.max(fixed_scaled_float)),
            "fixed_scaled_vs_fp32_score_max_abs_diff": float(np.max(np.abs(score_diff))),
            "fixed_scaled_vs_fp32_score_mean_abs_diff": float(np.mean(np.abs(score_diff))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point attention-score RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Cache length: {cache_length}")
    print(f"Score count: {expected_raw.size}")
    print(f"K requests: {expected_raw.size * HEAD_DIM}")
    print(f"Attention scale: float={float(attn0.scaling):.10f} q0.31={score_scale_q0_31}")
    print(f"Q input:          {files['q_input']}")
    print(f"K cache:          {files['k_cache']}")
    print(f"Expected raw:     {files['expected_raw']}")
    print(f"Expected scaled:  {files['expected_scaled']}")
    print(f"Debug meta:       {files['meta']}")
    print(
        "Fixed scaled score vs FP32 reference: "
        f"max_abs={meta['debug']['fixed_scaled_vs_fp32_score_max_abs_diff']:.8g} "
        f"mean_abs={meta['debug']['fixed_scaled_vs_fp32_score_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
