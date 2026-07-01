from __future__ import annotations

import importlib.util
import json
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
    require_input0,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
SOFTMAX_EXPORTER_PATH = Path(__file__).with_name("25_export_attention_softmax_value_vectors.py")

PREFIX = "o_proj_stage_real"
MODULE_NAME = "layer0.self_attn.o_proj_stage"

INPUT_SIZE = 2048
OUT_FEATURES = 1024
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
Q4_VALUES_PER_BYTE = 2
Q4_MIN = -8
Q4_MAX = 7

ACT_WIDTH = 24
ACT_FRAC = 12
WEIGHT_WIDTH = 4
SCALE_WIDTH = 16
SCALE_FRAC = 14
OUT_WIDTH = 24
OUT_FRAC = 12
ROW_ACC_WIDTH = 64
Q26_TO_Q12_SHIFT = SCALE_FRAC


def load_softmax_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "attention_softmax_value_exporter",
        SOFTMAX_EXPORTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SOFTMAX_EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    hex_digits = (width_bits + 3) // 4
    lines = [f"{int(value) & mask:0{hex_digits}x}" for value in values.reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pack_words32(values: np.ndarray, width_bits: int) -> np.ndarray:
    if width_bits <= 0 or width_bits > 32 or (32 % width_bits) != 0:
        raise ValueError(f"width_bits must divide 32, got {width_bits}")

    flat = values.reshape(-1).astype(np.int64)
    values_per_word = 32 // width_bits
    padded_count = ((flat.size + values_per_word - 1) // values_per_word) * values_per_word
    mask = (1 << width_bits) - 1

    padded = np.zeros(padded_count, dtype=np.uint64)
    padded[: flat.size] = np.bitwise_and(flat, mask).astype(np.uint64)
    grouped = padded.reshape(-1, values_per_word)
    shifts = (np.arange(values_per_word, dtype=np.uint64) * np.uint64(width_bits)).reshape(1, -1)
    return np.sum(grouped << shifts, axis=1, dtype=np.uint64).astype(np.uint32)


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def saturate_signed_array(values: np.ndarray, width_bits: int) -> tuple[np.ndarray, int]:
    low, high = signed_limits(width_bits)
    saturation_count = int(np.count_nonzero((values < low) | (values > high)))
    return np.clip(values, low, high).astype(np.int64), saturation_count


def fixed_to_float(values: np.ndarray, frac_bits: int) -> np.ndarray:
    return values.astype(np.float64) / float(1 << frac_bits)


def quantize_weight_q4_general(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.ascontiguousarray(weight.astype(np.float32))
    if weight.shape != (OUT_FEATURES, INPUT_SIZE):
        raise ValueError(f"Expected o_proj weight shape {(OUT_FEATURES, INPUT_SIZE)}, got {weight.shape}")

    grouped = weight.reshape(OUT_FEATURES, GROUP_COUNT, GROUP_SIZE)
    absmax = np.max(np.abs(grouped), axis=2)
    raw_scale = absmax / float(Q4_MAX)
    scale_q64 = np.rint(raw_scale * float(1 << SCALE_FRAC)).astype(np.int64)
    scale_q64 = np.where((absmax > 0.0) & (scale_q64 == 0), 1, scale_q64)
    if np.any(scale_q64 < 0) or np.any(scale_q64 > np.iinfo(np.uint16).max):
        raise ValueError("o_proj Q4 scale does not fit unsigned Q2.14")

    scale_q2_14 = scale_q64.astype(np.uint16)
    scale_used = scale_q2_14.astype(np.float32) / float(1 << SCALE_FRAC)
    safe_scale = np.where(scale_used > 0.0, scale_used, 1.0).astype(np.float32)
    q_grouped = np.rint(grouped / safe_scale[:, :, None])
    q = np.clip(q_grouped, Q4_MIN, Q4_MAX).astype(np.int8)
    return q.reshape(OUT_FEATURES, INPUT_SIZE), scale_q2_14


def compute_q4_gemv_q26(
    activation_q12_12: np.ndarray,
    weight_q4: np.ndarray,
    scale_q2_14: np.ndarray,
) -> np.ndarray:
    x_grouped = activation_q12_12.astype(np.int64).reshape(GROUP_COUNT, GROUP_SIZE)
    w_grouped = weight_q4.astype(np.int64).reshape(OUT_FEATURES, GROUP_COUNT, GROUP_SIZE)
    partial_sums = np.sum(w_grouped * x_grouped[None, :, :], axis=2, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14.astype(np.int64)
    return np.sum(scaled_sums, axis=1, dtype=np.int64)


def q26_to_q12_12(row_sum_q26: np.ndarray) -> tuple[np.ndarray, int]:
    shifted = row_sum_q26.astype(np.int64) >> Q26_TO_Q12_SHIFT
    return saturate_signed_array(shifted, OUT_WIDTH)


def main() -> None:
    softmax_exporter = load_softmax_exporter()
    qk = softmax_exporter.load_qk_exporter()

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.input_layernorm.register_forward_hook(capture_module_io(records, "input_norm")),
        attn0.o_proj.register_forward_hook(capture_module_io(records, "o_proj")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    input_norm_fp32 = np.ascontiguousarray(
        require_output(records, "input_norm")[0].to(torch.float32).numpy(),
        dtype=np.float32,
    )
    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())

    q4_path = Q4_VECTOR_DIR / "qkv_layer0_last_token_q4.npz"
    if q4_path.is_file():
        data = np.load(q4_path)
        artifact_position = int(np.asarray(data["prompt_position"]).item())
        artifact_token = int(np.asarray(data["token_id"]).item())
        if artifact_position != selected_position or artifact_token != selected_token_id:
            raise RuntimeError("Q4 artifact selected token does not match prompt")

    input_norm_q12_12 = softmax_exporter.quantize_activation_q4_12(input_norm_fp32)
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
    dummy_x = torch.zeros((1, prompt_len, softmax_exporter.HIDDEN_SIZE), dtype=torch.float32)
    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, dummy_x, position_ids)

    q_rope_by_pos = np.zeros((prompt_len, softmax_exporter.NUM_Q_HEADS, softmax_exporter.HEAD_DIM), dtype=np.int64)
    k_rope_by_pos = np.zeros((prompt_len, softmax_exporter.NUM_KV_HEADS, softmax_exporter.HEAD_DIM), dtype=np.int64)
    v_cache_q12_12 = np.zeros((prompt_len, softmax_exporter.NUM_KV_HEADS, softmax_exporter.HEAD_DIM), dtype=np.int64)
    norm_saturation = False
    rope_saturation = False

    if not q4_path.is_file():
        raise FileNotFoundError(f"Missing {q4_path}. Run 13_export_q4_gemv_vectors.py first.")
    q4_data = np.load(q4_path)

    for position in range(prompt_len):
        q_input_flat, _ = qk.compute_projection_q12_12(
            activation_q12_12=input_norm_q12_12[position],
            packed_weight=q4_data["q_weight_q4_packed"],
            scale_q2_14=q4_data["q_scale_q2_14"],
            row_count=softmax_exporter.Q_ROWS,
        )
        k_input_flat, _ = qk.compute_projection_q12_12(
            activation_q12_12=input_norm_q12_12[position],
            packed_weight=q4_data["k_weight_q4_packed"],
            scale_q2_14=q4_data["k_scale_q2_14"],
            row_count=softmax_exporter.K_ROWS,
        )
        v_input_flat, _ = qk.compute_projection_q12_12(
            activation_q12_12=input_norm_q12_12[position],
            packed_weight=q4_data["v_weight_q4_packed"],
            scale_q2_14=q4_data["v_scale_q2_14"],
            row_count=softmax_exporter.V_ROWS,
        )

        q_input_heads = q_input_flat.reshape(softmax_exporter.NUM_Q_HEADS, softmax_exporter.HEAD_DIM)
        k_input_heads = k_input_flat.reshape(softmax_exporter.NUM_KV_HEADS, softmax_exporter.HEAD_DIM)
        v_cache_q12_12[position] = v_input_flat.reshape(softmax_exporter.NUM_KV_HEADS, softmax_exporter.HEAD_DIM)

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
        cos_q1_15, _cos_sat = qk.quantize_signed(
            cos_float,
            qk.TRIG_WIDTH,
            qk.TRIG_FRAC,
        )
        sin_q1_15, _sin_sat = qk.quantize_signed(
            sin_float,
            qk.TRIG_WIDTH,
            qk.TRIG_FRAC,
        )
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

    score_scale_q0_31 = int(round(float(attn0.scaling) * float(1 << softmax_exporter.SCALE_FRAC)))
    scaled_scores = np.zeros((softmax_exporter.NUM_Q_HEADS, cache_length), dtype=np.int64)
    for q_head in range(softmax_exporter.NUM_Q_HEADS):
        kv_head = q_head // softmax_exporter.KV_REPEAT
        for position in range(cache_length):
            dot = int(
                np.sum(
                    q_current[q_head].astype(np.int64)
                    * k_cache[position, kv_head].astype(np.int64),
                    dtype=np.int64,
                )
            )
            scaled_scores[q_head, position] = softmax_exporter.arithmetic_shift_right(
                dot * score_scale_q0_31,
                softmax_exporter.SCALE_FRAC,
            )

    exp_lut = softmax_exporter.build_exp_lut()
    probs_q0_16, _lut_indices = softmax_exporter.fixed_softmax_probs(scaled_scores, exp_lut)
    attn_out_q12_12, value_saturation = softmax_exporter.fixed_value_accum(probs_q0_16, v_cache)
    activation_q12_12 = attn_out_q12_12.reshape(INPUT_SIZE).astype(np.int64)

    o_weight_fp32 = np.ascontiguousarray(tensor_to_float32(attn0.o_proj.weight).numpy(), dtype=np.float32)
    weight_q4, scale_q2_14 = quantize_weight_q4_general(o_weight_fp32)
    row_sum_q26 = compute_q4_gemv_q26(activation_q12_12, weight_q4, scale_q2_14)
    expected_q12_12, output_saturation_count = q26_to_q12_12(row_sum_q26)

    with torch.no_grad():
        input_norm_t = torch.from_numpy(input_norm_fp32).unsqueeze(0)
        q_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn0.q_proj.weight))
        k_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn0.k_proj.weight))
        v_flat_fp32 = manual_linear_from_weight(input_norm_t, tensor_to_float32(attn0.v_proj.weight))
        q_view = q_flat_fp32.view(batch_size, prompt_len, softmax_exporter.NUM_Q_HEADS, softmax_exporter.HEAD_DIM)
        k_view = k_flat_fp32.view(batch_size, prompt_len, softmax_exporter.NUM_KV_HEADS, softmax_exporter.HEAD_DIM)
        v_view = v_flat_fp32.view(batch_size, prompt_len, softmax_exporter.NUM_KV_HEADS, softmax_exporter.HEAD_DIM)
        q_norm = manual_rms_norm(q_view, tensor_to_float32(attn0.q_norm.weight), rms_eps)
        k_norm = manual_rms_norm(k_view, tensor_to_float32(attn0.k_norm.weight), rms_eps)
        q_states = q_norm.transpose(1, 2)
        k_states = k_norm.transpose(1, 2)
        v_states = v_view.transpose(1, 2)
        cos_ref, sin_ref = manual_rope_cos_sin(backbone.rotary_emb, input_norm_t, position_ids)
        q_rope_fp32, k_rope_fp32 = manual_apply_rope(q_states, k_states, cos_ref, sin_ref)
        k_repeated_fp32 = manual_repeat_kv(k_rope_fp32, softmax_exporter.KV_REPEAT)
        v_repeated_fp32 = manual_repeat_kv(v_states, softmax_exporter.KV_REPEAT)
        attn_fp32, _prob_fp32 = manual_cached_attention(
            q_rope_fp32[:, :, selected_position : selected_position + 1, :],
            k_repeated_fp32[:, :, :cache_length, :],
            v_repeated_fp32[:, :, :cache_length, :],
            float(attn0.scaling),
        )
        attn_concat_fp32 = attn_fp32.transpose(1, 2).contiguous().reshape(batch_size, 1, INPUT_SIZE)
        o_proj_fp32_from_fp32_attn = manual_linear_from_weight(
            attn_concat_fp32,
            tensor_to_float32(attn0.o_proj.weight),
        )[0, 0].numpy()
        o_proj_input_reference = require_input0(records, "o_proj")[0, selected_position].numpy()
        o_proj_output_reference = require_output(records, "o_proj")[0, selected_position].numpy()

    fixed_o_proj_float = fixed_to_float(expected_q12_12, OUT_FRAC)
    q4_vs_fp32_diff = fixed_o_proj_float - o_proj_fp32_from_fp32_attn.astype(np.float64)
    q4_input_diff = fixed_to_float(activation_q12_12, ACT_FRAC) - o_proj_input_reference.astype(np.float64)
    output_ref_diff = fixed_o_proj_float - o_proj_output_reference.astype(np.float64)

    files = {
        "activation": SIM_VECTOR_DIR / f"{PREFIX}_activation.hex",
        "weight": SIM_VECTOR_DIR / f"{PREFIX}_weight.hex",
        "weight_words32": SIM_VECTOR_DIR / f"{PREFIX}_weight_words32.hex",
        "scale": SIM_VECTOR_DIR / f"{PREFIX}_scale.hex",
        "scale_words32": SIM_VECTOR_DIR / f"{PREFIX}_scale_words32.hex",
        "expected_q26": SIM_VECTOR_DIR / f"{PREFIX}_expected_q26.hex",
        "expected_q12_12": SIM_VECTOR_DIR / f"{PREFIX}_expected_q12_12.hex",
        "meta": SIM_VECTOR_DIR / f"{PREFIX}_meta.json",
    }
    write_hex_lines(files["activation"], activation_q12_12, ACT_WIDTH)
    write_hex_lines(files["weight"], weight_q4.reshape(-1), WEIGHT_WIDTH)
    write_hex_lines(files["weight_words32"], pack_words32(weight_q4, WEIGHT_WIDTH), 32)
    write_hex_lines(files["scale"], scale_q2_14.reshape(-1), SCALE_WIDTH)
    write_hex_lines(files["scale_words32"], pack_words32(scale_q2_14, SCALE_WIDTH), 32)
    write_hex_lines(files["expected_q26"], row_sum_q26, ROW_ACC_WIDTH)
    write_hex_lines(files["expected_q12_12"], expected_q12_12, OUT_WIDTH)

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
            "input_size": INPUT_SIZE,
            "out_features": OUT_FEATURES,
            "group_size": GROUP_SIZE,
            "group_count": GROUP_COUNT,
        },
        "formats": {
            "activation": f"signed {ACT_WIDTH}-bit Q12.{ACT_FRAC} fixed attention output",
            "weight": "signed int4 unpacked one weight per hex line",
            "weight_words32": "signed int4 packed into 32-bit words, 8 weights per word, lowest index in low nibble",
            "scale": f"unsigned {SCALE_WIDTH}-bit Q2.{SCALE_FRAC}",
            "scale_words32": "unsigned Q2.14 packed into 32-bit words, 2 scales per word, lowest index in low halfword",
            "row_sum": f"signed {ROW_ACC_WIDTH}-bit Q26",
            "output": f"signed {OUT_WIDTH}-bit Q12.{OUT_FRAC}",
        },
        "saturation": {
            "q_gamma_quantize_count": int(q_gamma_saturation_count),
            "k_gamma_quantize_count": int(k_gamma_saturation_count),
            "norm_output": bool(norm_saturation),
            "rope_output": bool(rope_saturation),
            "value_output": bool(value_saturation),
            "o_proj_output_count": int(output_saturation_count),
        },
        "debug": {
            "attn_input_q12_12_max_abs": int(np.max(np.abs(activation_q12_12))),
            "row_sum_q26_min": int(np.min(row_sum_q26)),
            "row_sum_q26_max": int(np.max(row_sum_q26)),
            "expected_q12_12_max_abs": int(np.max(np.abs(expected_q12_12))),
            "fixed_attn_input_vs_fp32_o_proj_input_max_abs_diff": float(np.max(np.abs(q4_input_diff))),
            "fixed_o_proj_vs_fp32_from_fp32_attn_max_abs_diff": float(np.max(np.abs(q4_vs_fp32_diff))),
            "fixed_o_proj_vs_fp32_from_fp32_attn_mean_abs_diff": float(np.mean(np.abs(q4_vs_fp32_diff))),
            "fixed_o_proj_vs_hf_o_proj_output_max_abs_diff": float(np.max(np.abs(output_ref_diff))),
            "fixed_o_proj_vs_hf_o_proj_output_mean_abs_diff": float(np.mean(np.abs(output_ref_diff))),
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point o_proj RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Activation: {files['activation']}")
    print(f"Weight:     {files['weight']}")
    print(f"Weight32:   {files['weight_words32']}")
    print(f"Scale:      {files['scale']}")
    print(f"Scale32:    {files['scale_words32']}")
    print(f"Expected:   {files['expected_q12_12']}")
    print(f"Debug meta: {files['meta']}")
    print(
        "Fixed o_proj vs FP32 from FP32 attention: "
        f"max_abs={meta['debug']['fixed_o_proj_vs_fp32_from_fp32_attn_max_abs_diff']:.8g} "
        f"mean_abs={meta['debug']['fixed_o_proj_vs_fp32_from_fp32_attn_mean_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
