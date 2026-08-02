from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from common import load_model, manual_rope_cos_sin, tensor_to_float32


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Temp" / "persistent_multitoken_layer0_golden"
DEFAULT_ALL_LAYERS_OUTPUT_DIR = REPO_ROOT / "Temp" / "persistent_multitoken_full28_golden"

MODEL_LAYER_COUNT = 28
HIDDEN_SIZE = 1024
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_ROWS = NUM_Q_HEADS * HEAD_DIM
KV_ROWS = NUM_KV_HEADS * HEAD_DIM
KV_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS
MAX_CONTEXT = 256
KV_CACHE_BASE_ADDR = 0x0000_0004_1410_0000


def load_helper(filename: str, module_name: str) -> ModuleType:
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_int_auto(text: str) -> int:
    return int(text.replace("_", ""), 0)


def parse_layer_count(text: str) -> int:
    try:
        value = int(text.replace("_", ""), 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--layer-count must be an integer in 1..{MODEL_LAYER_COUNT}"
        ) from error
    if value < 1 or value > MODEL_LAYER_COUNT:
        raise argparse.ArgumentTypeError(
            f"--layer-count must be in 1..{MODEL_LAYER_COUNT}, got {value}"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an exact fixed-point persistent multi-token golden for one layer, "
            "the first N layers, or all Qwen3 decoder layers. Positions always start "
            "at zero and share retained per-layer K/V caches."
        )
    )
    parser.add_argument(
        "--token-ids",
        type=int,
        nargs="+",
        default=[374, 537],
        help="Token-id sequence; token 0 is evaluated at position 0 (default: 374 537).",
    )
    layer_mode = parser.add_mutually_exclusive_group()
    layer_mode.add_argument(
        "--layer-id",
        type=int,
        default=0,
        help="Run one decoder layer only (default: 0).",
    )
    layer_mode.add_argument(
        "--layer-count",
        type=parse_layer_count,
        default=None,
        metavar="N",
        help=(
            f"Run Layers 0 through N-1 (1..{MODEL_LAYER_COUNT}), chain their outputs, "
            "then run exact final RMSNorm and a full-vocabulary tied-Q4 LM-head scan. "
            f"N={MODEL_LAYER_COUNT} is classified as a complete full-model export."
        ),
    )
    layer_mode.add_argument(
        "--all-layers",
        action="store_true",
        help=(
            "Run every decoder layer in one process, chaining each layer's Q14.10 outputs, "
            "then run exact final RMSNorm and a full-vocabulary tied-Q4 LM-head scan."
        ),
    )
    parser.add_argument(
        "--input-hidden-hex",
        type=Path,
        default=None,
        help=(
            "Optional signed word32 Q14.10 matrix with token_count*1024 values. "
            "Required for layer-id > 0; chains beginning at Layer 0 default to exact "
            "tied-Q4 embeddings."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--kv-cache-base",
        type=parse_int_auto,
        default=KV_CACHE_BASE_ADDR,
    )
    parser.add_argument(
        "--lm-head-chunk-rows",
        type=int,
        default=1024,
        help="Rows quantized per full-vocabulary LM-head chunk in chained-layer modes.",
    )
    return parser.parse_args()


def as_float32(values: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor_to_float32(values).numpy(), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_limits(width_bits: int) -> tuple[int, int]:
    return -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1


def require_signed(values: np.ndarray, width_bits: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    low, high = signed_limits(width_bits)
    if np.any((result < low) | (result > high)):
        raise RuntimeError(f"{name} contains values outside signed {width_bits}-bit range")
    return result


def emit_hex(
    *,
    helper: ModuleType,
    output_dir: Path,
    files: dict[str, dict[str, Any]],
    relative_name: str,
    values: np.ndarray | list[int] | int,
    width_bits: int,
    signed: bool,
    logical_format: str,
) -> Path:
    path = output_dir / relative_name
    array = np.asarray([values] if np.isscalar(values) else values, dtype=np.int64)
    helper.write_hex_lines(path, array, width_bits)
    readback = helper.read_hex_lines(path, width_bits, signed=signed)
    expected = array.reshape(-1)
    if not np.array_equal(readback, expected):
        mismatch = np.flatnonzero(readback != expected)
        first = int(mismatch[0]) if mismatch.size else -1
        raise RuntimeError(
            f"Hex round-trip mismatch in {relative_name} at index {first}: "
            f"expected={expected[first] if first >= 0 else 'n/a'} "
            f"actual={readback[first] if first >= 0 else 'n/a'}"
        )
    files[relative_name] = {
        "count": int(expected.size),
        "storage_width_bits": width_bits,
        "signed_readback": signed,
        "logical_format": logical_format,
        "sha256": sha256_file(path),
    }
    return path


def q4_embedding_hidden(
    *,
    token_ids: list[int],
    model: Any,
    backbone: Any,
    embedding_helper: ModuleType,
) -> tuple[np.ndarray, list[float]]:
    embedding_weight = backbone.embed_tokens.weight.detach().to(
        dtype=model.lm_head.weight.dtype,
        device="cpu",
    )
    lm_head_weight = model.lm_head.weight.detach().to(device="cpu")
    vocab_size, hidden_size = map(int, embedding_weight.shape)
    if hidden_size != HIDDEN_SIZE:
        raise RuntimeError(f"Unexpected embedding width: {hidden_size}")

    outputs: list[np.ndarray] = []
    max_errors: list[float] = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"token id {token_id} is outside 0..{vocab_size - 1}")
        embed_row = embedding_weight[token_id]
        lm_head_row = lm_head_weight[token_id]
        if not np.array_equal(embed_row.numpy(), lm_head_row.numpy()):
            raise RuntimeError(f"Embedding/LM-head tie check failed for token {token_id}")
        fp32_row = embed_row.to(dtype=model.dtype).float().numpy()
        weight_q4, scale_q2_14 = embedding_helper.quantize_row(fp32_row)
        group_scale = np.repeat(scale_q2_14.astype(np.int64), embedding_helper.GROUP_SIZE)
        hidden_q14_10 = np.right_shift(
            weight_q4.astype(np.int64) * group_scale,
            embedding_helper.DEQUANT_SHIFT,
        )
        require_signed(hidden_q14_10, 24, f"embedding token {token_id}")
        outputs.append(hidden_q14_10)
        fixed_float = hidden_q14_10.astype(np.float64) / float(1 << 10)
        max_errors.append(float(np.max(np.abs(fixed_float - fp32_row.astype(np.float64)))))
    return np.stack(outputs), max_errors


def load_hidden_override(path: Path, token_count: int, qmap_helper: ModuleType) -> np.ndarray:
    words = qmap_helper.read_i32_hex(path.resolve()).astype(np.int64)
    expected = token_count * HIDDEN_SIZE
    if words.shape != (expected,):
        raise RuntimeError(
            f"input-hidden-hex has {words.size} values; expected {expected} "
            f"({token_count} tokens * {HIDDEN_SIZE})"
        )
    require_signed(words, 24, "input-hidden-hex")
    return words.reshape(token_count, HIDDEN_SIZE)


def quantize_layer_parameters(
    *,
    layer: Any,
    q4_helper: ModuleType,
    qk_helper: ModuleType,
    rms_helper: ModuleType,
    o_proj_helper: ModuleType,
    gate_up_helper: ModuleType,
    down_helper: ModuleType,
) -> dict[str, np.ndarray]:
    attn = layer.self_attn
    mlp = layer.mlp

    q_packed, q_scale, _ = q4_helper.quantize_weight_q4(as_float32(attn.q_proj.weight))
    k_packed, k_scale, _ = q4_helper.quantize_weight_q4(as_float32(attn.k_proj.weight))
    v_packed, v_scale, _ = q4_helper.quantize_weight_q4(as_float32(attn.v_proj.weight))
    o_weight, o_scale = o_proj_helper.quantize_weight_q4_general(as_float32(attn.o_proj.weight))
    gate_weight, gate_scale = gate_up_helper.quantize_weight_q4(as_float32(mlp.gate_proj.weight))
    up_weight, up_scale = gate_up_helper.quantize_weight_q4(as_float32(mlp.up_proj.weight))
    down_weight, down_scale = down_helper.quantize_weight_q4(as_float32(mlp.down_proj.weight))

    q_gamma, _ = qk_helper.quantize_signed(
        as_float32(attn.q_norm.weight), qk_helper.GAMMA_WIDTH, qk_helper.GAMMA_FRAC
    )
    k_gamma, _ = qk_helper.quantize_signed(
        as_float32(attn.k_norm.weight), qk_helper.GAMMA_WIDTH, qk_helper.GAMMA_FRAC
    )
    input_gamma = rms_helper.quantize_signed(
        as_float32(layer.input_layernorm.weight),
        rms_helper.GAMMA_WIDTH,
        rms_helper.GAMMA_FRAC,
    )
    post_gamma = rms_helper.quantize_signed(
        as_float32(layer.post_attention_layernorm.weight),
        rms_helper.GAMMA_WIDTH,
        rms_helper.GAMMA_FRAC,
    )
    return {
        "q_packed": q_packed,
        "q_scale": q_scale,
        "k_packed": k_packed,
        "k_scale": k_scale,
        "v_packed": v_packed,
        "v_scale": v_scale,
        "o_weight": o_weight,
        "o_scale": o_scale,
        "gate_weight": gate_weight,
        "gate_scale": gate_scale,
        "up_weight": up_weight,
        "up_scale": up_scale,
        "down_weight": down_weight,
        "down_scale": down_scale,
        "q_gamma": q_gamma,
        "k_gamma": k_gamma,
        "input_gamma": input_gamma,
        "post_gamma": post_gamma,
    }


def projection(
    qmap_helper: ModuleType,
    activation: np.ndarray,
    packed_weight: np.ndarray,
    scale: np.ndarray,
    row_count: int,
    name: str,
) -> np.ndarray:
    result = qmap_helper.compute_matrix_outputs(
        activation_q12_12=activation,
        packed_weight=packed_weight,
        scale_q2_14=scale,
        actual_q4_float=None,
        matrix_name=name,
        row_count=row_count,
    )
    return require_signed(np.asarray(result["q12_12"], dtype=np.int64), 24, name)


def build_kv_write_trace(
    *,
    kv_helper: ModuleType,
    kv_cache_base: int,
    layer_id: int,
    position: int,
    k_current: np.ndarray,
    v_current: np.ndarray,
) -> dict[str, np.ndarray]:
    addresses: list[int] = []
    data: list[int] = []
    kinds: list[int] = []
    heads: list[int] = []
    dims: list[int] = []
    for kind, source in ((0, k_current), (1, v_current)):
        for head in range(NUM_KV_HEADS):
            for dim in range(HEAD_DIM):
                addresses.append(
                    kv_helper.kv_addr(
                        base_addr=kv_cache_base,
                        layer_id=layer_id,
                        kv_kind=kind,
                        head=head,
                        position=position,
                        dim=dim,
                    )
                )
                data.append(kv_helper.sign_extend_to_word(int(source[head, dim])))
                kinds.append(kind)
                heads.append(head)
                dims.append(dim)
    return {
        "addr": np.asarray(addresses, dtype=np.int64),
        "data": np.asarray(data, dtype=np.int64),
        "kind": np.asarray(kinds, dtype=np.int64),
        "head": np.asarray(heads, dtype=np.int64),
        "dim": np.asarray(dims, dtype=np.int64),
    }


def attention_score_value(
    *,
    attention_helper: ModuleType,
    q_current: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    score_scale_q0_31: int,
    exp_lut: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    cache_length = int(k_cache.shape[0])
    raw = np.zeros((NUM_Q_HEADS, cache_length), dtype=np.int64)
    scaled = np.zeros_like(raw)
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
            raw[q_head, position] = dot
            scaled[q_head, position] = attention_helper.arithmetic_shift_right(
                dot * score_scale_q0_31,
                attention_helper.SCALE_FRAC,
            )
    probs, lut_indices = attention_helper.fixed_softmax_probs(scaled, exp_lut)
    output, saturated = attention_helper.fixed_value_accum(probs, v_cache)
    return {
        "raw": raw,
        "scaled": scaled,
        "probs": probs,
        "lut_indices": lut_indices,
        "output": require_signed(output, 24, "attention output"),
        "saturated": bool(saturated),
    }


def scoped_name(scope: str, name: str) -> str:
    return f"{scope}/{name}" if scope else name


def run_fixed_layer(
    *,
    layer_id: int,
    layer: Any,
    token_ids: list[int],
    hidden_inputs: np.ndarray,
    cos_q1_15: np.ndarray,
    sin_q1_15: np.ndarray,
    kv_cache_base: int,
    tokenizer: Any,
    helpers: dict[str, ModuleType],
    output_dir: Path,
    files: dict[str, dict[str, Any]],
    scope: str,
    exp_lut: np.ndarray,
    sigmoid_lut: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    q4 = helpers["q4"]
    rms = helpers["rms"]
    qmap = helpers["qmap"]
    qk = helpers["qk"]
    kv = helpers["kv"]
    chained = helpers["chained"]
    o_proj = helpers["o_proj"]
    post = helpers["post"]
    gate_up = helpers["gate_up"]
    silu = helpers["silu"]
    down = helpers["down"]
    residual = helpers["residual"]

    params = quantize_layer_parameters(
        layer=layer,
        q4_helper=q4,
        qk_helper=qk,
        rms_helper=rms,
        o_proj_helper=o_proj,
        gate_up_helper=gate_up,
        down_helper=down,
    )
    emit_hex(
        helper=chained,
        output_dir=output_dir,
        files=files,
        relative_name=scoped_name(scope, "input_hidden_q14_10_words32.hex"),
        values=hidden_inputs,
        width_bits=32,
        signed=True,
        logical_format="signed Q14.10 [position][hidden]",
    )

    k_cache = np.zeros((len(token_ids), NUM_KV_HEADS, HEAD_DIM), dtype=np.int64)
    v_cache = np.zeros_like(k_cache)
    score_scale_q0_31 = int(
        round(float(layer.self_attn.scaling) * float(1 << chained.SCALE_FRAC))
    )
    if score_scale_q0_31 <= 0 or score_scale_q0_31 >= (1 << 31):
        raise RuntimeError(f"Invalid Layer {layer_id} attention scale Q0.31: {score_scale_q0_31}")

    position_records: list[dict[str, Any]] = []
    traces: list[dict[str, np.ndarray]] = []
    layer_outputs: list[np.ndarray] = []
    all_addresses: list[int] = []
    for position, token_id in enumerate(token_ids):
        position_scope = scoped_name(scope, f"position_{position:03d}")
        hidden_q14_10 = hidden_inputs[position]
        input_norm_result = rms.fixed_rmsnorm_reference(hidden_q14_10, params["input_gamma"])
        input_norm_q12_12 = require_signed(
            input_norm_result["output_q12_12"],
            24,
            f"layer {layer_id} position {position} input norm",
        )

        q_flat = projection(
            qmap, input_norm_q12_12, params["q_packed"], params["q_scale"], Q_ROWS, "q_proj"
        )
        k_flat = projection(
            qmap, input_norm_q12_12, params["k_packed"], params["k_scale"], KV_ROWS, "k_proj"
        )
        v_flat = projection(
            qmap, input_norm_q12_12, params["v_packed"], params["v_scale"], KV_ROWS, "v_proj"
        )
        q_heads = q_flat.reshape(NUM_Q_HEADS, HEAD_DIM)
        k_heads = k_flat.reshape(NUM_KV_HEADS, HEAD_DIM)
        v_heads = v_flat.reshape(NUM_KV_HEADS, HEAD_DIM)
        q_norm, _, q_norm_sat = qk.fixed_rmsnorm_heads(q_heads, params["q_gamma"])
        k_norm, _, k_norm_sat = qk.fixed_rmsnorm_heads(k_heads, params["k_gamma"])
        q_rope, q_rope_sat = qk.fixed_rope_reference(
            q_norm, cos_q1_15[position], sin_q1_15[position]
        )
        k_rope, k_rope_sat = qk.fixed_rope_reference(
            k_norm, cos_q1_15[position], sin_q1_15[position]
        )
        q_rope = require_signed(q_rope, 24, f"layer {layer_id} position {position} q_rope")
        k_rope = require_signed(k_rope, 24, f"layer {layer_id} position {position} k_rope")

        retained_k = k_cache[:position].copy()
        retained_v = v_cache[:position].copy()
        k_cache[position] = k_rope
        v_cache[position] = v_heads
        if not np.array_equal(k_cache[:position], retained_k):
            raise RuntimeError(f"Layer {layer_id} K cache retention failed at position {position}")
        if not np.array_equal(v_cache[:position], retained_v):
            raise RuntimeError(f"Layer {layer_id} V cache retention failed at position {position}")

        trace = build_kv_write_trace(
            kv_helper=kv,
            kv_cache_base=kv_cache_base,
            layer_id=layer_id,
            position=position,
            k_current=k_rope,
            v_current=v_heads,
        )
        if np.unique(trace["addr"]).size != 2 * KV_ROWS:
            raise RuntimeError(f"Layer {layer_id} duplicate K/V address at position {position}")
        traces.append(trace)
        all_addresses.extend(int(value) for value in trace["addr"])

        attention = attention_score_value(
            attention_helper=chained,
            q_current=q_rope,
            k_cache=k_cache[: position + 1],
            v_cache=v_cache[: position + 1],
            score_scale_q0_31=score_scale_q0_31,
            exp_lut=exp_lut,
        )
        attn_out = np.asarray(attention["output"], dtype=np.int64).reshape(-1)
        o_q26 = o_proj.compute_q4_gemv_q26(attn_out, params["o_weight"], params["o_scale"])
        o_q12_12, o_sat_count = o_proj.q26_to_q12_12(o_q26)
        post_hidden, post_residual_sat = post.saturate_signed_array(
            hidden_q14_10 + (o_q12_12 >> (post.O_PROJ_FRAC - post.RESIDUAL_FRAC)),
            post.RESIDUAL_WIDTH,
        )
        post_norm_result = post.fixed_rmsnorm_reference(post_hidden, params["post_gamma"])
        post_norm = require_signed(
            post_norm_result["output_q12_12"], 24, f"layer {layer_id} post-attention norm"
        )

        gate_q26 = gate_up.compute_q4_gemv_q26(
            post_norm, params["gate_weight"], params["gate_scale"]
        )
        up_q26 = gate_up.compute_q4_gemv_q26(post_norm, params["up_weight"], params["up_scale"])
        gate_q12_12, gate_sat_count = gate_up.q26_to_q12_12(gate_q26)
        up_q12_12, up_sat_count = gate_up.q26_to_q12_12(up_q26)
        sigmoid_index = silu.gate_to_lut_index(gate_q12_12)
        sigmoid_q0_16 = sigmoid_lut[sigmoid_index]
        silu_gate, silu_sat_count = silu.saturate_signed_array(
            (gate_q12_12.astype(np.int64) * sigmoid_q0_16.astype(np.int64))
            >> silu.SIGMOID_FRAC,
            silu.OUT_WIDTH,
        )
        mlp_hidden, mlp_hidden_sat_count = silu.saturate_signed_array(
            (silu_gate.astype(np.int64) * up_q12_12.astype(np.int64)) >> silu.IN_FRAC,
            silu.OUT_WIDTH,
        )
        down_q26 = down.compute_q4_gemv_q26(
            mlp_hidden, params["down_weight"], params["down_scale"]
        )
        down_q12_12, down_sat_count = down.q26_to_q12_12(down_q26)
        layer_output, layer_residual_sat = residual.saturate_signed_array(
            post_hidden + (down_q12_12 >> residual.DOWN_TO_OUT_SHIFT),
            residual.OUT_WIDTH,
        )
        layer_outputs.append(layer_output)

        per_position_outputs = {
            "token_id.hex": (token_id, 32, False, "unsigned token id"),
            "cache_length.hex": (position + 1, 16, False, "unsigned retained-cache length"),
            "input_hidden_q14_10_words32.hex": (hidden_q14_10, 32, True, "signed Q14.10"),
            "input_norm_q12_12_words32.hex": (input_norm_q12_12, 32, True, "signed Q12.12"),
            "q_proj_q12_12_words32.hex": (q_flat, 32, True, "signed Q12.12"),
            "k_proj_q12_12_words32.hex": (k_flat, 32, True, "signed Q12.12"),
            "v_proj_q12_12_words32.hex": (v_flat, 32, True, "signed Q12.12"),
            "q_rope_q12_12_words32.hex": (q_rope, 32, True, "signed Q12.12 [q_head][dim]"),
            "k_rope_q12_12_words32.hex": (k_rope, 32, True, "signed Q12.12 [kv_head][dim]"),
            "kv_write_addr.hex": (trace["addr"], 64, False, "64-bit byte address, K then V"),
            "kv_write_data_words32.hex": (
                trace["data"], 32, False, "signed Q12.12 sign-extended word32"
            ),
            "kv_write_kind.hex": (trace["kind"], 4, False, "0=K, 1=V"),
            "kv_write_head.hex": (trace["head"], 8, False, "KV head index"),
            "kv_write_dim.hex": (trace["dim"], 8, False, "head dimension index"),
            "k_cache_q12_12_words32.hex": (
                k_cache[: position + 1], 32, True, "signed Q12.12 [position][kv_head][dim]"
            ),
            "v_cache_q12_12_words32.hex": (
                v_cache[: position + 1], 32, True, "signed Q12.12 [position][kv_head][dim]"
            ),
            "score_raw_q24_24.hex": (
                attention["raw"], 64, True, "signed Q24.24 [q_head][position]"
            ),
            "score_scaled_q24_24.hex": (
                attention["scaled"], 64, True, "signed Q24.24 [q_head][position]"
            ),
            "softmax_prob_q0_16_words32.hex": (
                attention["probs"], 32, False, "unsigned Q0.16 [q_head][position]"
            ),
            "softmax_lut_index.hex": (
                attention["lut_indices"], 16, False, "exp LUT index [q_head][position]"
            ),
            "attention_output_q12_12_words32.hex": (
                attn_out, 32, True, "signed Q12.12 [q_head][dim]"
            ),
            "o_proj_q12_12_words32.hex": (o_q12_12, 32, True, "signed Q12.12"),
            "post_attention_hidden_q14_10_words32.hex": (
                post_hidden, 32, True, "signed Q14.10"
            ),
            "post_attention_norm_q12_12_words32.hex": (
                post_norm, 32, True, "signed Q12.12"
            ),
            "mlp_gate_q12_12_words32.hex": (gate_q12_12, 32, True, "signed Q12.12"),
            "mlp_up_q12_12_words32.hex": (up_q12_12, 32, True, "signed Q12.12"),
            "mlp_hidden_q12_12_words32.hex": (mlp_hidden, 32, True, "signed Q12.12"),
            "mlp_down_q12_12_words32.hex": (down_q12_12, 32, True, "signed Q12.12"),
            "layer_output_q14_10_words32.hex": (layer_output, 32, True, "signed Q14.10"),
        }
        for filename, (values, width, is_signed, logical_format) in per_position_outputs.items():
            emit_hex(
                helper=chained,
                output_dir=output_dir,
                files=files,
                relative_name=scoped_name(position_scope, filename),
                values=values,
                width_bits=width,
                signed=is_signed,
                logical_format=logical_format,
            )

        position_records.append(
            {
                "position": position,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "cache_length": position + 1,
                "retained_prior_positions": position,
                "kv_write_count": int(trace["addr"].size),
                "first_k_addr": f"0x{int(trace['addr'][0]):016X}",
                "last_k_addr": f"0x{int(trace['addr'][KV_ROWS - 1]):016X}",
                "first_v_addr": f"0x{int(trace['addr'][KV_ROWS]):016X}",
                "last_v_addr": f"0x{int(trace['addr'][-1]):016X}",
                "max_abs": {
                    "q_rope_q12_12": int(np.max(np.abs(q_rope))),
                    "k_rope_q12_12": int(np.max(np.abs(k_rope))),
                    "v_q12_12": int(np.max(np.abs(v_heads))),
                    "attention_output_q12_12": int(np.max(np.abs(attn_out))),
                    "layer_output_q14_10": int(np.max(np.abs(layer_output))),
                },
                "saturation": {
                    "input_norm": bool(input_norm_result["saturation"]),
                    "q_norm": bool(q_norm_sat),
                    "k_norm": bool(k_norm_sat),
                    "q_rope": bool(q_rope_sat),
                    "k_rope": bool(k_rope_sat),
                    "attention_value": bool(attention["saturated"]),
                    "o_proj_count": int(o_sat_count),
                    "post_residual_count": int(post_residual_sat),
                    "post_norm": bool(post_norm_result["saturation"]),
                    "gate_count": int(gate_sat_count),
                    "up_count": int(up_sat_count),
                    "silu_count": int(silu_sat_count),
                    "mlp_hidden_count": int(mlp_hidden_sat_count),
                    "down_count": int(down_sat_count),
                    "layer_residual_count": int(layer_residual_sat),
                },
            }
        )

    layer_output_matrix = np.stack(layer_outputs)
    emit_hex(
        helper=chained,
        output_dir=output_dir,
        files=files,
        relative_name=scoped_name(scope, "layer_output_q14_10_words32.hex"),
        values=layer_output_matrix,
        width_bits=32,
        signed=True,
        logical_format="signed Q14.10 [position][hidden], directly chainable into layer-id+1",
    )
    if np.unique(np.asarray(all_addresses, dtype=np.int64)).size != len(all_addresses):
        raise RuntimeError(f"Layer {layer_id} K/V addresses overlap across positions")
    if len(traces) > 1:
        expected_stride = HEAD_DIM * kv.ELEMENT_BYTES
        for position in range(1, len(traces)):
            if not np.all(traces[position]["addr"] - traces[position - 1]["addr"] == expected_stride):
                raise RuntimeError(f"Layer {layer_id} K/V position stride failed at {position}")

    record = {
        "layer_id": layer_id,
        "score_scale_q0_31": score_scale_q0_31,
        "input_hidden_max_abs": [int(np.max(np.abs(row))) for row in hidden_inputs],
        "output_hidden_max_abs": [int(np.max(np.abs(row))) for row in layer_output_matrix],
        "position_results": position_records,
        "cache_retention": True,
        "kv_address_stride": True,
        "kv_address_non_overlap": True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return layer_output_matrix, record, np.asarray(all_addresses, dtype=np.int64)


def run_final_tail(
    *,
    final_hidden: np.ndarray,
    token_ids: list[int],
    tokenizer: Any,
    model: Any,
    backbone: Any,
    final_norm: ModuleType,
    lm_head: ModuleType,
    chained: ModuleType,
    output_dir: Path,
    files: dict[str, dict[str, Any]],
    chunk_rows: int,
    full_model: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    if chunk_rows <= 0:
        raise ValueError("--lm-head-chunk-rows must be positive")
    gamma_q8_7, gamma_sat = final_norm.quantize_signed_with_saturation(
        as_float32(backbone.norm.weight), final_norm.GAMMA_WIDTH, final_norm.GAMMA_FRAC
    )
    norm_outputs: list[np.ndarray] = []
    norm_debug: list[dict[str, Any]] = []
    for position in range(len(token_ids)):
        result = final_norm.fixed_rmsnorm_reference(final_hidden[position], gamma_q8_7)
        norm_outputs.append(
            require_signed(result["output_q12_12"], 24, f"final norm position {position}")
        )
        norm_debug.append(result)
    final_norm_matrix = np.stack(norm_outputs)
    emit_hex(
        helper=chained,
        output_dir=output_dir,
        files=files,
        relative_name="final/final_hidden_q14_10_words32.hex",
        values=final_hidden,
        width_bits=32,
        signed=True,
        logical_format=(
            "signed Q14.10 [position][hidden] after all decoder layers"
            if full_model
            else "signed Q14.10 [position][hidden] after selected decoder-layer chain"
        ),
    )
    emit_hex(
        helper=chained,
        output_dir=output_dir,
        files=files,
        relative_name="final/final_norm_q12_12_words32.hex",
        values=final_norm_matrix,
        width_bits=32,
        signed=True,
        logical_format="signed Q12.12 [position][hidden]",
    )

    lm_weight = model.lm_head.weight.detach().to(dtype=torch.float32, device="cpu")
    vocab_size, input_size = map(int, lm_weight.shape)
    if input_size != HIDDEN_SIZE:
        raise RuntimeError(f"LM-head width {input_size} does not match {HIDDEN_SIZE}")
    full_logits = np.zeros((len(token_ids), vocab_size), dtype=np.int64)
    best_tokens = [-1 for _ in token_ids]
    best_scores = [-(1 << 62) for _ in token_ids]
    top_tokens: list[list[tuple[int, int]]] = [[] for _ in token_ids]
    chunk_count = (vocab_size + chunk_rows - 1) // chunk_rows
    for chunk_index, row_start in enumerate(range(0, vocab_size, chunk_rows)):
        row_end = min(row_start + chunk_rows, vocab_size)
        weight_chunk = np.ascontiguousarray(lm_weight[row_start:row_end].numpy(), dtype=np.float32)
        q_chunk, scale_chunk = lm_head.quantize_weight_q4_chunk(weight_chunk)
        for position in range(len(token_ids)):
            logits = lm_head.compute_q4_logits_q26(
                final_norm_matrix[position], q_chunk, scale_chunk
            )
            full_logits[position, row_start:row_end] = logits
            local_index = int(np.argmax(logits))
            local_score = int(logits[local_index])
            if local_score > best_scores[position]:
                best_scores[position] = local_score
                best_tokens[position] = row_start + local_index
            top_tokens[position].extend(
                (row_start + int(index), int(logits[int(index)]))
                for index in np.argsort(logits)[-16:]
            )
            top_tokens[position] = sorted(
                top_tokens[position], key=lambda item: (-item[1], item[0])
            )[:16]
        if (chunk_index + 1) % 16 == 0 or row_end == vocab_size:
            print(f"LM head chunks: {chunk_index + 1}/{chunk_count} rows={row_end}/{vocab_size}")

    require_signed(full_logits, lm_head.ROW_ACC_WIDTH, "full-vocabulary Q4 logits")
    for position in range(len(token_ids)):
        exact_argmax = int(np.argmax(full_logits[position]))
        if exact_argmax != best_tokens[position]:
            raise RuntimeError(
                f"Position {position} streamed argmax {best_tokens[position]} != full {exact_argmax}"
            )
        if int(full_logits[position, exact_argmax]) != best_scores[position]:
            raise RuntimeError(f"Position {position} streamed LM-head score mismatch")

    emit_hex(
        helper=chained,
        output_dir=output_dir,
        files=files,
        relative_name="final/lm_head_full_vocab_logits_q26.hex",
        values=full_logits,
        width_bits=lm_head.ROW_ACC_WIDTH,
        signed=True,
        logical_format="signed Q26 [position][vocab_row], exact full-vocabulary scan order",
    )
    position_results: list[dict[str, Any]] = []
    for position, source_token in enumerate(token_ids):
        emit_hex(
            helper=chained,
            output_dir=output_dir,
            files=files,
            relative_name=f"final/position_{position:03d}/argmax_token.hex",
            values=best_tokens[position],
            width_bits=32,
            signed=False,
            logical_format="unsigned full-vocabulary Q4 argmax token",
        )
        emit_hex(
            helper=chained,
            output_dir=output_dir,
            files=files,
            relative_name=f"final/position_{position:03d}/argmax_score_q26.hex",
            values=best_scores[position],
            width_bits=lm_head.ROW_ACC_WIDTH,
            signed=True,
            logical_format="signed Q26 winning LM-head score",
        )
        position_results.append(
            {
                "position": position,
                "source_token_id": source_token,
                "source_token_text": tokenizer.decode([source_token]),
                "final_hidden_max_abs_q14_10": int(np.max(np.abs(final_hidden[position]))),
                "final_norm_max_abs_q12_12": int(np.max(np.abs(final_norm_matrix[position]))),
                "final_norm_output_saturation_count": int(
                    norm_debug[position]["output_saturation_count"]
                ),
                "argmax_token": best_tokens[position],
                "argmax_token_text": tokenizer.decode([best_tokens[position]]),
                "argmax_score_q26": best_scores[position],
                "top_q4_tokens": [
                    {
                        "token": int(token),
                        "score_q26": int(score),
                        "text": tokenizer.decode([int(token)]),
                    }
                    for token, score in top_tokens[position][:8]
                ],
            }
        )
    return {
        "final_norm_gamma_quantize_saturation_count": int(gamma_sat),
        "vocab_size": vocab_size,
        "lm_head_chunk_rows": chunk_rows,
        "lm_head_chunk_count": chunk_count,
        "argmax_tie_rule": "strict greater-than update in ascending row order; lowest row wins ties",
        "position_results": position_results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    layer_prefix_mode = args.layer_count is not None
    chained_tail_mode = args.all_layers or layer_prefix_mode
    token_ids = [int(value) for value in args.token_ids]
    if not token_ids:
        raise ValueError("--token-ids must contain at least one token")
    if len(token_ids) > MAX_CONTEXT:
        raise ValueError(f"token count {len(token_ids)} exceeds MAX_CONTEXT={MAX_CONTEXT}")
    if args.layer_id < 0:
        raise ValueError("--layer-id must be non-negative")
    if args.kv_cache_base < 0 or (args.kv_cache_base & 0x3) != 0:
        raise ValueError("--kv-cache-base must be a non-negative, 4-byte-aligned address")

    helpers = {
        "q4": load_helper("13_export_q4_gemv_vectors.py", "persistent_q4_helper"),
        "rms": load_helper("17_export_rmsnorm_fixed_vectors.py", "persistent_rms_helper"),
        "qmap": load_helper("21_export_qmap_qkv_projection_image.py", "persistent_qmap_helper"),
        "qk": load_helper("22_export_qk_norm_rope_fixed_vectors.py", "persistent_qk_helper"),
        "kv": load_helper("23_export_kv_cache_append_vectors.py", "persistent_kv_helper"),
        "chained": load_helper(
            "46_export_chained_attention_score_value_vectors.py", "persistent_attention_helper"
        ),
        "o_proj": load_helper("26_export_o_proj_vectors.py", "persistent_o_proj_helper"),
        "post": load_helper("27_export_post_attention_residual_norm_vectors.py", "persistent_post_helper"),
        "gate_up": load_helper("28_export_mlp_gate_up_vectors.py", "persistent_gate_up_helper"),
        "silu": load_helper("29_export_mlp_silu_mul_vectors.py", "persistent_silu_helper"),
        "down": load_helper("30_export_mlp_down_vectors.py", "persistent_down_helper"),
        "residual": load_helper("31_export_mlp_residual_add_vectors.py", "persistent_residual_helper"),
        "embedding": load_helper("49_export_q4_embedding_vectors.py", "persistent_embedding_helper"),
    }
    final_norm = None
    lm_head = None
    if chained_tail_mode:
        final_norm = load_helper("32_export_final_rmsnorm_vectors.py", "persistent_final_norm_helper")
        lm_head = load_helper("35_export_lm_head_full_vocab_vectors.py", "persistent_lm_head_helper")

    # This is the only checkpoint load in either mode. Layer parameters and LM-head
    # chunks are quantized from this one resident model object.
    tokenizer, model, backbone = load_model()
    layer_count = len(backbone.layers)
    if args.all_layers and layer_count != MODEL_LAYER_COUNT:
        raise RuntimeError(
            f"Expected Qwen3-0.6B to have {MODEL_LAYER_COUNT} layers, found {layer_count}"
        )
    if layer_prefix_mode and args.layer_count > layer_count:
        raise ValueError(
            f"layer-count {args.layer_count} exceeds checkpoint layer count {layer_count}"
        )
    if not chained_tail_mode and args.layer_id >= layer_count:
        raise ValueError(f"layer-id {args.layer_id} is outside 0..{layer_count - 1}")
    if args.all_layers:
        layer_ids = list(range(layer_count))
    elif layer_prefix_mode:
        layer_ids = list(range(args.layer_count))
    else:
        layer_ids = [args.layer_id]
    layer_prefix = layer_ids == list(range(len(layer_ids)))
    model_complete = chained_tail_mode and layer_ids == list(range(layer_count))

    embedding_errors: list[float] | None = None
    if args.input_hidden_hex is not None:
        hidden_inputs = load_hidden_override(args.input_hidden_hex, len(token_ids), helpers["qmap"])
        hidden_source = "explicit signed word32 Q14.10 input-hidden matrix"
    else:
        if not chained_tail_mode and args.layer_id != 0:
            raise ValueError("--input-hidden-hex is required when --layer-id is not zero")
        hidden_inputs, embedding_errors = q4_embedding_hidden(
            token_ids=token_ids,
            model=model,
            backbone=backbone,
            embedding_helper=helpers["embedding"],
        )
        hidden_source = "tied embedding row quantized by 49_export_q4_embedding_vectors.py"

    position_ids = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros((1, len(token_ids), HIDDEN_SIZE), dtype=torch.float32)
    with torch.no_grad():
        cos, sin = manual_rope_cos_sin(backbone.rotary_emb, dummy_x, position_ids)
    cos_q1_15 = np.zeros((len(token_ids), HEAD_DIM), dtype=np.int64)
    sin_q1_15 = np.zeros_like(cos_q1_15)
    trig_saturation = 0
    for position in range(len(token_ids)):
        cos_q1_15[position], cos_sat = helpers["qk"].quantize_signed(
            as_float32(cos[0, position]), helpers["qk"].TRIG_WIDTH, helpers["qk"].TRIG_FRAC
        )
        sin_q1_15[position], sin_sat = helpers["qk"].quantize_signed(
            as_float32(sin[0, position]), helpers["qk"].TRIG_WIDTH, helpers["qk"].TRIG_FRAC
        )
        trig_saturation += int(cos_sat) + int(sin_sat)

    output_dir_arg = args.output_dir
    if model_complete and output_dir_arg == DEFAULT_OUTPUT_DIR:
        output_dir_arg = DEFAULT_ALL_LAYERS_OUTPUT_DIR
    elif layer_prefix_mode and output_dir_arg == DEFAULT_OUTPUT_DIR:
        output_dir_arg = (
            REPO_ROOT / "Temp" / f"persistent_multitoken_first{args.layer_count}_golden"
        )
    output_dir = output_dir_arg.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    emit_hex(
        helper=helpers["chained"],
        output_dir=output_dir,
        files=files,
        relative_name="rope_cos_q1_15_words32.hex",
        values=cos_q1_15,
        width_bits=32,
        signed=True,
        logical_format="signed Q1.15 widened to word32, [position][dim]",
    )
    emit_hex(
        helper=helpers["chained"],
        output_dir=output_dir,
        files=files,
        relative_name="rope_sin_q1_15_words32.hex",
        values=sin_q1_15,
        width_bits=32,
        signed=True,
        logical_format="signed Q1.15 widened to word32, [position][dim]",
    )
    emit_hex(
        helper=helpers["chained"],
        output_dir=output_dir,
        files=files,
        relative_name=(
            "initial_hidden_q14_10_words32.hex"
            if chained_tail_mode
            else "input_hidden_q14_10_words32.hex"
        ),
        values=hidden_inputs,
        width_bits=32,
        signed=True,
        logical_format="signed Q14.10 [position][hidden]",
    )

    exp_lut = helpers["chained"].build_exp_lut()
    sigmoid_lut = helpers["silu"].build_sigmoid_lut()
    layer_results: list[dict[str, Any]] = []
    current_hidden = hidden_inputs
    global_addresses: set[int] = set()
    for ordinal, layer_id in enumerate(layer_ids):
        print(f"Layer {layer_id:02d}/{layer_ids[-1]:02d}: quantize and run {len(token_ids)} positions")
        scope = f"layer_{layer_id:02d}" if chained_tail_mode else ""
        current_hidden, layer_record, addresses = run_fixed_layer(
            layer_id=layer_id,
            layer=backbone.layers[layer_id],
            token_ids=token_ids,
            hidden_inputs=current_hidden,
            cos_q1_15=cos_q1_15,
            sin_q1_15=sin_q1_15,
            kv_cache_base=args.kv_cache_base,
            tokenizer=tokenizer,
            helpers=helpers,
            output_dir=output_dir,
            files=files,
            scope=scope,
            exp_lut=exp_lut,
            sigmoid_lut=sigmoid_lut,
        )
        for address in addresses.tolist():
            if int(address) in global_addresses:
                raise RuntimeError(f"K/V address 0x{int(address):016X} overlaps another layer")
            global_addresses.add(int(address))
        layer_results.append(layer_record)
        print(
            f"Layer {layer_id:02d}: PASS elapsed={layer_record['elapsed_seconds']:.3f}s "
            f"out_max={layer_record['output_hidden_max_abs']}"
        )
        if ordinal + 1 < len(layer_ids):
            gc.collect()

    final_tail: dict[str, Any] | None = None
    if chained_tail_mode:
        assert final_norm is not None and lm_head is not None
        print("Final RMSNorm + full-vocabulary tied-Q4 LM head")
        final_tail = run_final_tail(
            final_hidden=current_hidden,
            token_ids=token_ids,
            tokenizer=tokenizer,
            model=model,
            backbone=backbone,
            final_norm=final_norm,
            lm_head=lm_head,
            chained=helpers["chained"],
            output_dir=output_dir,
            files=files,
            chunk_rows=args.lm_head_chunk_rows,
            full_model=model_complete,
        )

    helper_reuse = [
        "13_export_q4_gemv_vectors.py: weight quantization",
        "17_export_rmsnorm_fixed_vectors.py: input RMSNorm",
        "21_export_qmap_qkv_projection_image.py: Q/K/V Q4 projection",
        "22_export_qk_norm_rope_fixed_vectors.py: q/k RMSNorm and RoPE",
        "23_export_kv_cache_append_vectors.py: physical K/V addresses and sign extension",
        "46_export_chained_attention_score_value_vectors.py: score, softmax, value accumulation",
        "26..31: exact attention output, residual, and MLP stages",
        "49_export_q4_embedding_vectors.py: tied Q4 embedding",
    ]
    if chained_tail_mode:
        helper_reuse.extend(
            [
                "32_export_final_rmsnorm_vectors.py: exact final RMSNorm",
                "35_export_lm_head_full_vocab_vectors.py: chunked Q4 full-vocabulary logits and argmax",
            ]
        )
    if model_complete:
        manifest_name = "persistent_multitoken_full_model_manifest.json"
        manifest_id = "qwen3_0p6b_full28_persistent_multitoken_fixed_golden"
        manifest_purpose = "Exact RTL-arithmetic persistent multi-token full-model golden"
    elif layer_prefix_mode:
        manifest_name = "persistent_multitoken_layer_prefix_manifest.json"
        manifest_id = (
            f"qwen3_0p6b_first{args.layer_count}_persistent_multitoken_fixed_golden"
        )
        manifest_purpose = (
            "Exact RTL-arithmetic persistent multi-token truncated-prefix diagnostic "
            "with final-tail scan"
        )
    else:
        manifest_name = "persistent_multitoken_manifest.json"
        manifest_id = (
            f"qwen3_0p6b_layer{args.layer_id}_persistent_multitoken_fixed_golden"
        )
        manifest_purpose = "Exact RTL-arithmetic persistent multi-token one-layer seam golden"
    manifest_path = output_dir / manifest_name
    manifest: dict[str, Any] = {
        "format_version": 2 if chained_tail_mode else 1,
        "name": manifest_id,
        "purpose": manifest_purpose,
        "selection_mode": (
            "all_layers"
            if args.all_layers
            else ("layer_prefix" if layer_prefix_mode else "single_layer")
        ),
        "all_layers": model_complete,
        "layer_ids": layer_ids,
        "layer_count": len(layer_ids),
        "layer_prefix": layer_prefix,
        "model_complete": model_complete,
        "tail_semantics": (
            "full_model_decode"
            if model_complete
            else (
                "truncated_prefix_diagnostic"
                if chained_tail_mode
                else "not_present"
            )
        ),
        "valid_as_full_model_decode": bool(model_complete and final_tail is not None),
        "token_ids": token_ids,
        "positions": list(range(len(token_ids))),
        "hidden_source": hidden_source,
        "input_hidden_hex": None if args.input_hidden_hex is None else str(args.input_hidden_hex.resolve()),
        "embedding_max_abs_error_vs_fp32": embedding_errors,
        "cache": {
            "base_addr": f"0x{args.kv_cache_base:016X}",
            "max_context": MAX_CONTEXT,
            "layout": "[layer][kind K/V][kv_head][position][dim], word32 elements",
            "write_order_per_position": "all K heads/dims, then all V heads/dims",
            "position_stride_bytes": HEAD_DIM * helpers["kv"].ELEMENT_BYTES,
            "writes_per_layer_position": 2 * KV_ROWS,
        },
        "arithmetic_contract": {
            "hidden_and_layer_output": "signed 24-bit Q14.10 stored in word32",
            "norm_projection_rope_attention_mlp": "signed 24-bit Q12.12 stored in word32",
            "q4_weights": "groupwise symmetric signed int4, group 64, unsigned Q2.14 scales",
            "rope": "signed Q1.15 widened to word32",
            "attention_scores": "signed 64-bit Q24.24",
            "softmax_probability": "unsigned 24-bit Q0.16",
            "lm_head_scores": "signed Q26 at helper ROW_ACC_WIDTH",
        },
        "helper_reuse": helper_reuse,
        "layer_results": layer_results,
        "final_tail": final_tail,
        "self_check": {
            "status": "PASS",
            "model_checkpoint_loads": 1,
            "layer_weight_quantizations": len(layer_ids),
            "hex_roundtrip_files": len(files),
            "cache_retention_every_layer": True,
            "kv_address_stride_every_layer": True,
            "kv_global_non_overlap": True,
            "trig_saturation_count": trig_saturation,
            "full_vocab_argmax_checked": bool(chained_tail_mode),
            "full_model_argmax_checked": bool(model_complete),
        },
        "scope": {
            "exact": (
                "Every integer result uses the same quantizers, shifts, saturation, LUTs, address "
                "mapping, and accumulation helpers as the passing RTL vector exporters."
            ),
            "intentional_approximation": (
                "Only the project's required custom Q4 weights and declared fixed-point formats "
                "approximate the FP32 checkpoint; no extra Python-side approximation is introduced."
            ),
            "not_included": "This is a software golden/export self-check; RTL timing is verified separately.",
        },
        "elapsed_seconds": round(time.perf_counter() - total_started, 3),
        "files": files,
    }
    if not chained_tail_mode:
        manifest["layer_id"] = args.layer_id
        manifest["position_results"] = layer_results[0]["position_results"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Persistent multi-token fixed-point golden: PASS")
    print("=" * 80)
    print(f"Layers:            {layer_ids[0]}..{layer_ids[-1]} ({len(layer_ids)})")
    print(f"Token IDs:         {token_ids}")
    print(f"Positions:         0..{len(token_ids) - 1}")
    print(f"K/V base:          0x{args.kv_cache_base:016X}")
    if final_tail is not None:
        for item in final_tail["position_results"]:
            print(
                f"Position {item['position']} argmax: {item['argmax_token']} "
                f"{item['argmax_token_text']!r} score_q26={item['argmax_score_q26']}"
            )
    print(f"Files round-trip:  {len(files)}")
    print(f"Output directory:  {output_dir}")
    print(f"Manifest:          {manifest_path}")


if __name__ == "__main__":
    main()
