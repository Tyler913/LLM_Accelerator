from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from common import PROMPT, encode_prompt, load_model, manual_rope_cos_sin


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
QK_EXPORTER_PATH = Path(__file__).with_name("22_export_qk_norm_rope_fixed_vectors.py")

PREFIX = "kv_cache_append_real"
MODULE_NAME = "layer0.self_attn.kv_cache_append"

KV_CACHE_BASE_ADDR = 0x0000_0004_1410_0000
LAYER_ID = 0
MAX_CONTEXT = 256
ELEMENT_BYTES = 4

HIDDEN_SIZE = 1024
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_ROWS = NUM_KV_HEADS * HEAD_DIM
IN_WIDTH = 24
DATA_WIDTH = 32


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


def sign_extend_to_word(value: int, source_width: int = IN_WIDTH) -> int:
    sign_bit = 1 << (source_width - 1)
    mask = (1 << source_width) - 1
    narrowed = value & mask
    if narrowed & sign_bit:
        return narrowed | (((1 << (DATA_WIDTH - source_width)) - 1) << source_width)
    return narrowed


def kv_addr(
    *,
    base_addr: int,
    layer_id: int,
    kv_kind: int,
    head: int,
    position: int,
    dim: int,
) -> int:
    position_stride = HEAD_DIM * ELEMENT_BYTES
    head_stride = MAX_CONTEXT * position_stride
    kind_stride = NUM_KV_HEADS * head_stride
    layer_stride = 2 * kind_stride
    return (
        base_addr
        + layer_id * layer_stride
        + kv_kind * kind_stride
        + head * head_stride
        + position * position_stride
        + dim * ELEMENT_BYTES
    )


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
    if selected_position >= MAX_CONTEXT:
        raise RuntimeError(f"Selected position {selected_position} exceeds MAX_CONTEXT={MAX_CONTEXT}")

    attn0 = backbone.layers[0].self_attn
    activation_q12_12 = data["input_norm_q4_12"].astype(np.int64)

    k_input_flat, k_row_sum_q26 = qk.compute_projection_q12_12(
        activation_q12_12=activation_q12_12,
        packed_weight=data["k_weight_q4_packed"],
        scale_q2_14=data["k_scale_q2_14"],
        row_count=KV_ROWS,
    )
    v_input_flat, v_row_sum_q26 = qk.compute_projection_q12_12(
        activation_q12_12=activation_q12_12,
        packed_weight=data["v_weight_q4_packed"],
        scale_q2_14=data["v_scale_q2_14"],
        row_count=KV_ROWS,
    )

    k_input_heads = k_input_flat.reshape(NUM_KV_HEADS, HEAD_DIM)
    v_input_heads = v_input_flat.reshape(NUM_KV_HEADS, HEAD_DIM)

    k_gamma_float = qk.as_float_array(attn0.k_norm.weight)
    k_gamma_q8_7, k_gamma_saturation_count = qk.quantize_signed(
        k_gamma_float,
        qk.GAMMA_WIDTH,
        qk.GAMMA_FRAC,
    )
    k_norm_q12_12, k_norm_debug, k_norm_saturation = qk.fixed_rmsnorm_heads(
        k_input_heads,
        k_gamma_q8_7,
    )

    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros((1, prompt_len, HIDDEN_SIZE), dtype=torch.float32)
    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, dummy_x, position_ids)
    cos_float = qk.as_float_array(cos[0, selected_position, :])
    sin_float = qk.as_float_array(sin[0, selected_position, :])
    cos_q1_15, cos_saturation_count = qk.quantize_signed(cos_float, qk.TRIG_WIDTH, qk.TRIG_FRAC)
    sin_q1_15, sin_saturation_count = qk.quantize_signed(sin_float, qk.TRIG_WIDTH, qk.TRIG_FRAC)
    k_rope_q12_12, k_rope_saturation = qk.fixed_rope_reference(
        k_norm_q12_12,
        cos_q1_15,
        sin_q1_15,
    )

    expected_addr: list[int] = []
    expected_data: list[int] = []
    expected_kind: list[int] = []
    expected_head: list[int] = []
    expected_dim: list[int] = []

    for kv_kind, source in [(0, k_rope_q12_12), (1, v_input_heads)]:
        for head in range(NUM_KV_HEADS):
            for dim in range(HEAD_DIM):
                value = int(source[head, dim])
                expected_addr.append(
                    kv_addr(
                        base_addr=KV_CACHE_BASE_ADDR,
                        layer_id=LAYER_ID,
                        kv_kind=kv_kind,
                        head=head,
                        position=selected_position,
                        dim=dim,
                    )
                )
                expected_data.append(sign_extend_to_word(value))
                expected_kind.append(kv_kind)
                expected_head.append(head)
                expected_dim.append(dim)

    expected_addr_array = np.array(expected_addr, dtype=np.uint64)
    expected_data_array = np.array(expected_data, dtype=np.uint64)
    expected_kind_array = np.array(expected_kind, dtype=np.uint8)
    expected_head_array = np.array(expected_head, dtype=np.uint8)
    expected_dim_array = np.array(expected_dim, dtype=np.uint8)

    files = {
        "k_input": SIM_VECTOR_DIR / f"{PREFIX}_k_input.hex",
        "v_input": SIM_VECTOR_DIR / f"{PREFIX}_v_input.hex",
        "expected_addr": SIM_VECTOR_DIR / f"{PREFIX}_expected_addr.hex",
        "expected_data": SIM_VECTOR_DIR / f"{PREFIX}_expected_data.hex",
        "expected_kind": SIM_VECTOR_DIR / f"{PREFIX}_expected_kind.hex",
        "expected_head": SIM_VECTOR_DIR / f"{PREFIX}_expected_head.hex",
        "expected_dim": SIM_VECTOR_DIR / f"{PREFIX}_expected_dim.hex",
        "meta": SIM_VECTOR_DIR / f"{PREFIX}_meta.json",
    }

    write_hex_lines(files["k_input"], k_rope_q12_12, IN_WIDTH)
    write_hex_lines(files["v_input"], v_input_heads, IN_WIDTH)
    write_hex_lines(files["expected_addr"], expected_addr_array, 64)
    write_hex_lines(files["expected_data"], expected_data_array, DATA_WIDTH)
    write_hex_lines(files["expected_kind"], expected_kind_array, 4)
    write_hex_lines(files["expected_head"], expected_head_array, 8)
    write_hex_lines(files["expected_dim"], expected_dim_array, 8)

    k_recompute_float = (
        k_row_sum_q26.astype(np.float64) / float(1 << (qk.ACT_FRAC + qk.SCALE_FRAC))
    ).astype(np.float32)
    v_recompute_float = (
        v_row_sum_q26.astype(np.float64) / float(1 << (qk.ACT_FRAC + qk.SCALE_FRAC))
    ).astype(np.float32)

    meta: dict[str, Any] = {
        "format_version": 1,
        "name": PREFIX,
        "module": MODULE_NAME,
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": selected_position,
        "selected_token_id": selected_token_id,
        "selected_token_text": tokenizer.decode([selected_token_id]),
        "cache": {
            "base_addr": f"0x{KV_CACHE_BASE_ADDR:016X}",
            "layer_id": LAYER_ID,
            "max_context": MAX_CONTEXT,
            "element_bytes": ELEMENT_BYTES,
            "write_order": "all K heads/dims first, then all V heads/dims",
            "write_count": len(expected_addr),
            "first_k_addr": f"0x{expected_addr[0]:016X}",
            "last_k_addr": f"0x{expected_addr[KV_ROWS - 1]:016X}",
            "first_v_addr": f"0x{expected_addr[KV_ROWS]:016X}",
            "last_v_addr": f"0x{expected_addr[-1]:016X}",
        },
        "formats": {
            "k_input": f"signed {IN_WIDTH}-bit Q12.12 post-RoPE K",
            "v_input": f"signed {IN_WIDTH}-bit Q12.12 V projection output",
            "write_data": f"signed {IN_WIDTH}-bit Q12.12 sign-extended to {DATA_WIDTH}-bit word",
            "write_addr": "64-bit byte address",
        },
        "saturation": {
            "k_gamma_quantize_count": int(k_gamma_saturation_count),
            "cos_quantize_count": int(cos_saturation_count),
            "sin_quantize_count": int(sin_saturation_count),
            "k_norm_output": bool(k_norm_saturation),
            "k_rope_output": bool(k_rope_saturation),
        },
        "debug": {
            "k_projection_recompute_vs_artifact_max_abs_diff": float(
                np.max(np.abs(k_recompute_float - data["actual_k_q4"].astype(np.float32)))
            ),
            "v_projection_recompute_vs_artifact_max_abs_diff": float(
                np.max(np.abs(v_recompute_float - data["actual_v_q4"].astype(np.float32)))
            ),
            "k_rope_q12_12_max_abs": int(np.max(np.abs(k_rope_q12_12))),
            "v_input_q12_12_max_abs": int(np.max(np.abs(v_input_heads))),
            "k_norm_head0": {
                "sum_squares": int(k_norm_debug[0]["sum_squares"]),
                "mean_square": int(k_norm_debug[0]["mean_square"]),
                "rms_q12_12": int(k_norm_debug[0]["rms_q12_12"]),
                "inv_rms_uq8_16": int(k_norm_debug[0]["inv_rms_uq8_16"]),
            },
        },
        "files": {name: path.name for name, path in files.items() if name != "meta"},
    }
    files["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Exported fixed-point KV cache append RTL test vectors")
    print("=" * 80)
    print(f"Module: {MODULE_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"K input:       {files['k_input']}")
    print(f"V input:       {files['v_input']}")
    print(f"Expected addr: {files['expected_addr']}")
    print(f"Expected data: {files['expected_data']}")
    print(f"Debug meta:    {files['meta']}")
    print(f"Write count:   {len(expected_addr)}")
    print(f"First K addr:  0x{expected_addr[0]:016X}")
    print(f"Last K addr:   0x{expected_addr[KV_ROWS - 1]:016X}")
    print(f"First V addr:  0x{expected_addr[KV_ROWS]:016X}")
    print(f"Last V addr:   0x{expected_addr[-1]:016X}")


if __name__ == "__main__":
    main()
