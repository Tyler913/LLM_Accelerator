import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    PROMPT,
    capture_module_io,
    encode_prompt,
    get_rms_norm_eps,
    load_model,
    require_input0,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_fp32_v0"


def as_float32_array(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor_to_float32(x).numpy(), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    selected_position = prompt_len - 1
    selected_token_id = int(input_ids[0, selected_position].item())
    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.input_layernorm.register_forward_hook(
            capture_module_io(records, "layer0_input_norm")
        ),
        attn0.q_proj.register_forward_hook(capture_module_io(records, "q_proj")),
        attn0.k_proj.register_forward_hook(capture_module_io(records, "k_proj")),
        attn0.v_proj.register_forward_hook(capture_module_io(records, "v_proj")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    layer0_eps = np.array(
        get_rms_norm_eps(layer0.input_layernorm, model.config),
        dtype=np.float32,
    )

    input_hidden = require_input0(records, "layer0_input_norm")[0, selected_position, :]
    norm_output = require_output(records, "layer0_input_norm")[0, selected_position, :]
    input_norm = require_output(records, "layer0_input_norm")[0, selected_position, :]

    rmsnorm_path = VECTOR_DIR / "rmsnorm_layer0_last_token.npz"
    write_npz(
        rmsnorm_path,
        input_hidden=as_float32_array(input_hidden),
        norm_weight=as_float32_array(layer0.input_layernorm.weight),
        eps=layer0_eps,
        expected_output=as_float32_array(norm_output),
        prompt_position=np.array(selected_position, dtype=np.int32),
        token_id=np.array(selected_token_id, dtype=np.int32),
    )

    qkv_path = VECTOR_DIR / "qkv_layer0_last_token.npz"
    write_npz(
        qkv_path,
        input_norm=as_float32_array(input_norm),
        q_weight=as_float32_array(attn0.q_proj.weight),
        k_weight=as_float32_array(attn0.k_proj.weight),
        v_weight=as_float32_array(attn0.v_proj.weight),
        expected_q=as_float32_array(require_output(records, "q_proj")[0, selected_position, :]),
        expected_k=as_float32_array(require_output(records, "k_proj")[0, selected_position, :]),
        expected_v=as_float32_array(require_output(records, "v_proj")[0, selected_position, :]),
        prompt_position=np.array(selected_position, dtype=np.int32),
        token_id=np.array(selected_token_id, dtype=np.int32),
    )

    manifest = {
        "format_version": 1,
        "name": "qwen3_0p6b_fp32_v0",
        "purpose": "FPGA/RTL golden vectors for first RMSNorm and Q/K/V GEMV bring-up.",
        "model_dir": str(backbone.config.name_or_path)
        if hasattr(backbone, "config")
        else "Qwen3-0.6B-Base",
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": selected_position,
        "selected_token_id": selected_token_id,
        "dtype": "float32",
        "vectors": [
            {
                "file": rmsnorm_path.name,
                "sha256": sha256_file(rmsnorm_path),
                "arrays": {
                    "input_hidden": [int(model.config.hidden_size)],
                    "norm_weight": [int(model.config.hidden_size)],
                    "eps": [],
                    "expected_output": [int(model.config.hidden_size)],
                },
            },
            {
                "file": qkv_path.name,
                "sha256": sha256_file(qkv_path),
                "arrays": {
                    "input_norm": [int(model.config.hidden_size)],
                    "q_weight": [
                        int(model.config.num_attention_heads)
                        * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)),
                        int(model.config.hidden_size),
                    ],
                    "k_weight": [
                        int(model.config.num_key_value_heads)
                        * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)),
                        int(model.config.hidden_size),
                    ],
                    "v_weight": [
                        int(model.config.num_key_value_heads)
                        * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)),
                        int(model.config.hidden_size),
                    ],
                    "expected_q": [
                        int(model.config.num_attention_heads)
                        * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
                    ],
                    "expected_k": [
                        int(model.config.num_key_value_heads)
                        * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
                    ],
                    "expected_v": [
                        int(model.config.num_key_value_heads)
                        * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
                    ],
                },
            },
        ],
    }

    manifest_path = VECTOR_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported FPGA FP32 test vectors")
    print("=" * 80)
    print(f"Vector directory: {VECTOR_DIR}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Token ids: {manifest['token_ids']}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    print(f"Wrote: {rmsnorm_path.name}")
    print(f"Wrote: {qkv_path.name}")
    print(f"Wrote: {manifest_path.name}")


if __name__ == "__main__":
    main()
