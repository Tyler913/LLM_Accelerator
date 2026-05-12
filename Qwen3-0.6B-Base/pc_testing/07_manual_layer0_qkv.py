from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1]

PROMPT = "The future of FPGA is"
TOKEN_INDEX = 0
SAMPLE_COUNT = 8


MATH_NOTES = """
Manual Layer 0 RMSNorm + Q/K/V projection reference
====================================================

This script verifies the first small hardware-shaped slice of the model:

  token id
    -> embedding lookup
    -> layer 0 input RMSNorm
    -> layer 0 q_proj / k_proj / v_proj

It compares manual tensor math against Hugging Face module hook outputs.
The goal is to make the math explicit enough to port this block to FPGA DSPs.

Symbols
-------

  H       = hidden size = 1024
  QH      = number of query heads = 16
  KVH     = number of key/value heads = 8
  D       = head dimension = 128
  QO      = Q output width = QH * D = 2048
  KVO     = K/V output width = KVH * D = 1024
  token   = integer token id
  E       = embedding table, shape [vocab_size, H]
  x       = embedding vector for one token, shape [H]
  gamma   = RMSNorm learned scale, shape [H]
  eps     = RMSNorm epsilon, Qwen3 uses 1e-6 here
  Wq      = q_proj weight, shape [QO, H]
  Wk      = k_proj weight, shape [KVO, H]
  Wv      = v_proj weight, shape [KVO, H]

Embedding lookup
----------------

For j in [0, H):

  x[j] = E[token, j]

This is a table read, not a multiply. In hardware, token id selects one row
from the embedding table.

RMSNorm
-------

Qwen3 uses RMSNorm, not LayerNorm. RMSNorm does not subtract the mean.

For one hidden vector x[0..H-1]:

  sum_square = sum_{j=0}^{H-1}(x[j] * x[j])
  mean_square = sum_square / H
  inv_rms = 1 / sqrt(mean_square + eps)
  norm[j] = x[j] * inv_rms * gamma[j]

Hardware view:

  1. Square H input values.
  2. Reduce-add the H squares.
  3. Divide by H.
  4. Add eps.
  5. Compute reciprocal square root.
  6. Multiply every x[j] by inv_rms and by gamma[j].

The reciprocal square root is the non-GEMV scalar operation to plan for.

Linear projection / GEMV
------------------------

PyTorch Linear stores weight as [out_features, in_features].
For y = Linear(norm), each output element is:

  y[o] = bias[o] + sum_{i=0}^{H-1}(norm[i] * W[o, i])

Qwen3 projection layers usually have no bias, so bias[o] is often 0.

For Q:

  q[o] = sum_{i=0}^{H-1}(norm[i] * Wq[o, i]), o in [0, 2048)

For K:

  k[o] = sum_{i=0}^{H-1}(norm[i] * Wk[o, i]), o in [0, 1024)

For V:

  v[o] = sum_{i=0}^{H-1}(norm[i] * Wv[o, i]), o in [0, 1024)

Hardware view:

  Each q/k/v output element is one dot product.
  Each dot product consumes H multiply-accumulate operations.
  Q has 2048 dot products; K has 1024; V has 1024.

Head reshape after projection
-----------------------------

This script checks the flat projection outputs. The next attention steps
reshape them conceptually as:

  q_head[h, d] = q[h * D + d], h in [0, 16), d in [0, 128)
  k_head[h, d] = k[h * D + d], h in [0, 8),  d in [0, 128)
  v_head[h, d] = v[h * D + d], h in [0, 8),  d in [0, 128)

RoPE, q_norm, k_norm, attention score, softmax, weighted V accumulation, and
o_proj are intentionally left for later scripts.
"""


def tensor_to_reference(x: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and convert it to CPU float32 for stable comparison."""
    return x.detach().to(dtype=torch.float32, device="cpu").clone()


def capture_tensor_output(
    records: dict[str, torch.Tensor],
    name: str,
) -> Callable[[Any, Any, Any], None]:
    def hook(module: Any, inputs: Any, output: Any) -> None:
        if not torch.is_tensor(output):
            raise TypeError(f"{name} hook expected a tensor output, got {type(output)}")
        records[name] = tensor_to_reference(cast(torch.Tensor, output))

    return hook


def get_backbone(model: Any) -> Any:
    # Qwen3ForCausalLM stores the transformer backbone under model.model.
    return model.model if hasattr(model, "model") else model


def get_rms_norm_eps(norm_module: Any, model_config: Any) -> float:
    if hasattr(norm_module, "variance_epsilon"):
        return float(norm_module.variance_epsilon)
    if hasattr(norm_module, "eps"):
        return float(norm_module.eps)
    return float(model_config.rms_norm_eps)


def manual_embedding_lookup(embedding_module: Any, token_id: int) -> torch.Tensor:
    """Manual embedding formula: x[j] = E[token_id, j]."""
    embedding_table = tensor_to_reference(embedding_module.weight)
    return embedding_table[token_id].view(1, 1, -1)


def manual_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Manual RMSNorm:

      sum_square = sum(x[j] * x[j])
      mean_square = sum_square / H
      inv_rms = 1 / sqrt(mean_square + eps)
      norm[j] = x[j] * inv_rms * weight[j]

    The reduction dimension is the final hidden dimension.
    """
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    mean_square = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x_fp32 * inv_rms * weight_fp32


def manual_linear(x: torch.Tensor, linear_module: Any) -> torch.Tensor:
    """
    Manual PyTorch Linear/GEMV:

      y[o] = bias[o] + sum_{i=0}^{in_features-1}(x[i] * W[o, i])

    PyTorch stores W as [out_features, in_features], so the tensor expression
    is x @ W.T.
    """
    weight = tensor_to_reference(linear_module.weight)
    bias = linear_module.bias
    y = x.to(torch.float32) @ weight.t()
    if bias is not None:
        y = y + tensor_to_reference(bias)
    return y


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


def format_values(x: torch.Tensor, count: int = SAMPLE_COUNT) -> str:
    flat = x.flatten()[:count].tolist()
    return "[" + ", ".join(f"{value:.8f}" for value in flat) + "]"


def compare_tensor(
    name: str,
    manual: torch.Tensor,
    hf: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    manual = manual.to(torch.float32)
    hf = hf.to(torch.float32)
    diff = manual - hf

    print("-" * 80)
    print(name)
    print(f"  manual shape: {list(manual.shape)}")
    print(f"  HF hook shape: {list(hf.shape)}")
    print(f"  max abs error:  {max_abs_error(manual, hf):.10e}")
    print(f"  mean abs error: {mean_abs_error(manual, hf):.10e}")
    print(f"  allclose(atol={atol}, rtol={rtol}): {torch.allclose(manual, hf, atol=atol, rtol=rtol)}")
    print(f"  manual first {SAMPLE_COUNT}: {format_values(manual)}")
    print(f"  HF first {SAMPLE_COUNT}:     {format_values(hf)}")
    print(f"  diff first {SAMPLE_COUNT}:   {format_values(diff)}")


def print_model_dimensions(config: Any) -> None:
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))

    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim

    print("Model dimensions used by this script")
    print("=" * 80)
    print(f"H hidden size: {hidden_size}")
    print(f"Q heads: {num_attention_heads}")
    print(f"KV heads: {num_key_value_heads}")
    print(f"Head dim: {head_dim}")
    print(f"Q flat width: {q_width}")
    print(f"K/V flat width: {kv_width}")
    print(f"RMSNorm eps: {config.rms_norm_eps}")
    print()


def print_projection_shapes(attn0: Any) -> None:
    print("Layer 0 projection weight shapes")
    print("=" * 80)
    print(f"q_proj.weight: {list(attn0.q_proj.weight.shape)} = [Q output, H input]")
    print(f"k_proj.weight: {list(attn0.k_proj.weight.shape)} = [K output, H input]")
    print(f"v_proj.weight: {list(attn0.v_proj.weight.shape)} = [V output, H input]")
    print(f"q_proj.bias present: {attn0.q_proj.bias is not None}")
    print(f"k_proj.bias present: {attn0.k_proj.bias is not None}")
    print(f"v_proj.bias present: {attn0.v_proj.bias is not None}")
    print()


def main() -> None:
    print(MATH_NOTES)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float32

    print("Runtime")
    print("=" * 80)
    print(f"Model dir: {MODEL_DIR}")
    print(f"Device: {device_name}")
    print(f"Load dtype: {dtype}")
    print()

    # AutoModel builds a dynamic Qwen3 module at runtime. The Any annotations
    # keep IDE type checkers from applying overly generic Hugging Face stubs.
    tokenizer: Any = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    loaded_model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        local_files_only=True,
    )
    model: Any = loaded_model.to(device)

    model.eval()
    backbone: Any = get_backbone(model)
    layer0: Any = backbone.layers[0]
    attn0: Any = layer0.self_attn

    print_model_dimensions(model.config)
    print_projection_shapes(attn0)

    encoded: Any = tokenizer(PROMPT, return_tensors="pt")
    input_ids = cast(torch.Tensor, encoded.input_ids).to(device)
    token = input_ids[:, TOKEN_INDEX:TOKEN_INDEX + 1]
    token_id = int(token.item())

    print("Prompt")
    print("=" * 80)
    print(f"Prompt text: {PROMPT!r}")
    print(f"All token ids: {input_ids[0].tolist()}")
    print(f"Selected token index: {TOKEN_INDEX}")
    print(f"Selected token id: {token_id}")
    print(f"Selected token text: {tokenizer.decode([token_id])!r}")
    print()

    records: dict[str, torch.Tensor] = {}
    hooks = [
        backbone.embed_tokens.register_forward_hook(
            capture_tensor_output(records, "embed_tokens")
        ),
        layer0.input_layernorm.register_forward_hook(
            capture_tensor_output(records, "layer0.input_layernorm")
        ),
        attn0.q_proj.register_forward_hook(
            capture_tensor_output(records, "layer0.self_attn.q_proj")
        ),
        attn0.k_proj.register_forward_hook(
            capture_tensor_output(records, "layer0.self_attn.k_proj")
        ),
        attn0.v_proj.register_forward_hook(
            capture_tensor_output(records, "layer0.self_attn.v_proj")
        ),
    ]

    try:
        with torch.no_grad():
            model(
                input_ids=token,
                use_cache=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    required_records = [
        "embed_tokens",
        "layer0.input_layernorm",
        "layer0.self_attn.q_proj",
        "layer0.self_attn.k_proj",
        "layer0.self_attn.v_proj",
    ]
    missing = [name for name in required_records if name not in records]
    if missing:
        raise RuntimeError(f"Missing hook outputs: {missing}")

    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)

    manual_embed = manual_embedding_lookup(backbone.embed_tokens, token_id)
    norm_weight = tensor_to_reference(layer0.input_layernorm.weight)
    manual_norm = manual_rms_norm(manual_embed, norm_weight, rms_eps)
    manual_q = manual_linear(manual_norm, attn0.q_proj)
    manual_k = manual_linear(manual_norm, attn0.k_proj)
    manual_v = manual_linear(manual_norm, attn0.v_proj)

    print("Manual vs Hugging Face hook comparison")
    print("=" * 80)
    compare_tensor("embedding lookup: x[j] = E[token, j]", manual_embed, records["embed_tokens"])
    compare_tensor(
        "layer0 input RMSNorm: norm[j] = x[j] * rsqrt(mean(x*x) + eps) * gamma[j]",
        manual_norm,
        records["layer0.input_layernorm"],
    )
    compare_tensor(
        "layer0 q_proj GEMV: q[o] = sum_i(norm[i] * Wq[o, i])",
        manual_q,
        records["layer0.self_attn.q_proj"],
    )
    compare_tensor(
        "layer0 k_proj GEMV: k[o] = sum_i(norm[i] * Wk[o, i])",
        manual_k,
        records["layer0.self_attn.k_proj"],
    )
    compare_tensor(
        "layer0 v_proj GEMV: v[o] = sum_i(norm[i] * Wv[o, i])",
        manual_v,
        records["layer0.self_attn.v_proj"],
    )

    print()
    print("Head reshape indices for the next script")
    print("=" * 80)
    print("q flat [2048] -> q heads [16, 128]: q_head[h, d] = q[h * 128 + d]")
    print("k flat [1024] -> k heads [8, 128]:  k_head[h, d] = k[h * 128 + d]")
    print("v flat [1024] -> v heads [8, 128]:  v_head[h, d] = v[h * 128 + d]")
    print()
    print("Validation complete.")


if __name__ == "__main__":
    main()
