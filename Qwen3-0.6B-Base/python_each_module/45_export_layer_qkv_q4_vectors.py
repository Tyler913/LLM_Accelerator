import argparse
import hashlib
import importlib.util
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
    require_input0,
    require_output,
    tensor_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
Q4_EXPORTER_PATH = Path(__file__).with_name("13_export_q4_gemv_vectors.py")


def load_q4_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("q4_gemv_vectors", Q4_EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Q4 helper script: {Q4_EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def default_output(layer_id: int) -> Path:
    return Q4_VECTOR_DIR / f"qkv_layer{layer_id}_last_token_q4.npz"


def default_manifest(layer_id: int) -> Path:
    return Q4_VECTOR_DIR / f"qkv_layer{layer_id}_last_token_q4_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-layer Q/K/V Q4 vectors for QMAP projection simulation."
    )
    parser.add_argument("--layer-id", type=int, default=1, help="Decoder layer index to export")
    parser.add_argument(
        "--position",
        type=int,
        default=-1,
        help="Prompt token position to export; -1 selects the last prompt token",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output Q4 NPZ")
    parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q4 = load_q4_helpers()

    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape
    if batch_size != 1:
        raise RuntimeError(f"Expected batch size 1, got {batch_size}")

    num_layers = int(model.config.num_hidden_layers)
    if args.layer_id < 0 or args.layer_id >= num_layers:
        raise ValueError(f"layer-id must be in range 0..{num_layers - 1}, got {args.layer_id}")

    selected_position = args.position if args.position >= 0 else prompt_len - 1
    if selected_position < 0 or selected_position >= prompt_len:
        raise ValueError(f"position must be in range 0..{prompt_len - 1}, got {selected_position}")

    selected_token_id = int(input_ids[0, selected_position].item())
    layer = backbone.layers[args.layer_id]
    attn = layer.self_attn
    output_path = args.output if args.output is not None else default_output(args.layer_id)
    manifest_path = args.manifest if args.manifest is not None else default_manifest(args.layer_id)

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer.input_layernorm.register_forward_hook(
            capture_module_io(records, "layer_input_norm")
        ),
        attn.q_proj.register_forward_hook(capture_module_io(records, "q_proj")),
        attn.k_proj.register_forward_hook(capture_module_io(records, "k_proj")),
        attn.v_proj.register_forward_hook(capture_module_io(records, "v_proj")),
    ]

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    input_hidden = require_input0(records, "layer_input_norm")[0, selected_position, :]
    input_norm = require_output(records, "layer_input_norm")[0, selected_position, :]
    expected_q = as_float32_array(require_output(records, "q_proj")[0, selected_position, :])
    expected_k = as_float32_array(require_output(records, "k_proj")[0, selected_position, :])
    expected_v = as_float32_array(require_output(records, "v_proj")[0, selected_position, :])

    input_norm_fp32 = as_float32_array(input_norm)
    input_norm_q4_12 = q4.quantize_activation_q4_12(input_norm_fp32)

    q_packed, q_scale, q_int4 = q4.quantize_weight_q4(as_float32_array(attn.q_proj.weight))
    k_packed, k_scale, k_int4 = q4.quantize_weight_q4(as_float32_array(attn.k_proj.weight))
    v_packed, v_scale, v_int4 = q4.quantize_weight_q4(as_float32_array(attn.v_proj.weight))

    q_actual, _q_partial, _q_scaled = q4.q4_gemv_from_ints(input_norm_q4_12, q_int4, q_scale)
    k_actual, _k_partial, _k_scaled = q4.q4_gemv_from_ints(input_norm_q4_12, k_int4, k_scale)
    v_actual, _v_partial, _v_scaled = q4.q4_gemv_from_ints(input_norm_q4_12, v_int4, v_scale)

    metrics = {
        "q_proj": q4.compare_metrics(q_actual, expected_q),
        "k_proj": q4.compare_metrics(k_actual, expected_k),
        "v_proj": q4.compare_metrics(v_actual, expected_v),
    }

    write_npz(
        output_path,
        input_hidden_fp32=as_float32_array(input_hidden),
        input_norm_fp32=input_norm_fp32,
        input_norm_q4_12=input_norm_q4_12,
        q_weight_q4_packed=q_packed,
        q_scale_q2_14=q_scale,
        k_weight_q4_packed=k_packed,
        k_scale_q2_14=k_scale,
        v_weight_q4_packed=v_packed,
        v_scale_q2_14=v_scale,
        expected_q_fp32=expected_q,
        expected_k_fp32=expected_k,
        expected_v_fp32=expected_v,
        actual_q_q4=q_actual,
        actual_k_q4=k_actual,
        actual_v_q4=v_actual,
        prompt_position=np.array(selected_position, dtype=np.int32),
        token_id=np.array(selected_token_id, dtype=np.int32),
        layer_id=np.array(args.layer_id, dtype=np.int32),
    )

    manifest = {
        "format_version": 1,
        "name": f"qwen3_0p6b_layer{args.layer_id}_qkv_q4_v0",
        "purpose": "Per-layer Verilog-facing custom Q4 Q/K/V projection vectors.",
        "model_dir": str(backbone.config.name_or_path)
        if hasattr(backbone, "config")
        else "Qwen3-0.6B-Base",
        "prompt": PROMPT,
        "token_ids": [int(value) for value in input_ids.flatten().tolist()],
        "selected_position": selected_position,
        "selected_token_id": selected_token_id,
        "scope": {
            "layer": args.layer_id,
            "matrices": ["q_proj", "k_proj", "v_proj"],
            "source": "full-prompt eager forward hooks on target layer input_layernorm/q_proj/k_proj/v_proj",
        },
        "activation_format": {
            "name": "signed_q4_12",
            "width_bits": q4.ACT_WIDTH,
            "fraction_bits": q4.ACT_FRAC,
            "stored_dtype": "int16",
        },
        "weight_format": {
            "name": "custom_groupwise_symmetric_q4_v0",
            "stored_dtype": "packed_uint8",
            "q_min": q4.Q4_MIN,
            "q_max": q4.Q4_MAX,
            "group_size": q4.Q4_GROUP_SIZE,
            "scale_dtype": "uint16_q2_14",
            "scale_fraction_bits": q4.SCALE_FRAC,
        },
        "arrays": {
            output_path.name: {
                "input_hidden_fp32": list(as_float32_array(input_hidden).shape),
                "input_norm_fp32": list(input_norm_fp32.shape),
                "input_norm_q4_12": list(input_norm_q4_12.shape),
                "q_weight_q4_packed": list(q_packed.shape),
                "q_scale_q2_14": list(q_scale.shape),
                "k_weight_q4_packed": list(k_packed.shape),
                "k_scale_q2_14": list(k_scale.shape),
                "v_weight_q4_packed": list(v_packed.shape),
                "v_scale_q2_14": list(v_scale.shape),
                "actual_q_q4": list(q_actual.shape),
                "actual_k_q4": list(k_actual.shape),
                "actual_v_q4": list(v_actual.shape),
            }
        },
        "metrics_against_fp32_expected": metrics,
        "files": [
            {
                "file": output_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
                "sha256": sha256_file(output_path),
            }
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported per-layer Q4 QKV vectors")
    print("=" * 80)
    print(f"Layer: {args.layer_id}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Selected position: {selected_position}")
    print(f"Selected token id: {selected_token_id} / {tokenizer.decode([selected_token_id])!r}")
    for name, values in metrics.items():
        print(
            f"{name}: max_abs_error={values['max_abs_error']:.8f} "
            f"mean_abs_error={values['mean_abs_error']:.8f} rmse={values['rmse']:.8f}"
        )
    print(f"Wrote: {output_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()}")
    print(f"Wrote: {manifest_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()}")


if __name__ == "__main__":
    main()
