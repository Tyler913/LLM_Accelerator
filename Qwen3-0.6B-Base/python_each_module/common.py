from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1]
PROMPT = "The future of FPGA is"
SAMPLE_COUNT = 6


def load_model() -> tuple[Any, Any, Any]:
    """Load tokenizer, causal LM model, and the transformer backbone."""
    tokenizer: Any = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    loaded_model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model: Any = loaded_model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model, get_backbone(model)


def encode_prompt(tokenizer: Any) -> torch.Tensor:
    encoded: Any = tokenizer(PROMPT, return_tensors="pt")
    return cast(torch.Tensor, encoded.input_ids).to(torch.device("cpu"))


def tensor_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Return a detached CPU float32 tensor."""
    return x.detach().to(dtype=torch.float32, device="cpu")


def clone_reference(x: torch.Tensor) -> torch.Tensor:
    """Save a detached copy for stable comparisons."""
    return tensor_to_float32(x).clone()


def get_backbone(model: Any) -> Any:
    return model.model if hasattr(model, "model") else model


def get_rms_norm_eps(norm_module: Any, model_config: Any) -> float:
    if hasattr(norm_module, "variance_epsilon"):
        return float(norm_module.variance_epsilon)
    if hasattr(norm_module, "eps"):
        return float(norm_module.eps)
    return float(model_config.rms_norm_eps)


def capture_module_io(
    records: dict[str, dict[str, torch.Tensor]],
    name: str,
) -> Callable[[Any, Any, Any], None]:
    """Create a forward hook that stores the first input and tensor output."""

    def hook(module: Any, inputs: Any, output: Any) -> None:
        item: dict[str, torch.Tensor] = {}

        if isinstance(inputs, tuple) and inputs and torch.is_tensor(inputs[0]):
            item["input0"] = clone_reference(cast(torch.Tensor, inputs[0]))

        if torch.is_tensor(output):
            item["output"] = clone_reference(cast(torch.Tensor, output))
        elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            item["output"] = clone_reference(cast(torch.Tensor, output[0]))
        else:
            raise TypeError(f"{name} hook expected tensor output, got {type(output)}")

        records[name] = item

    return hook


def require_output(records: dict[str, dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if name not in records or "output" not in records[name]:
        raise RuntimeError(f"Missing output record for {name}")
    return records[name]["output"]


def require_input0(records: dict[str, dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if name not in records or "input0" not in records[name]:
        raise RuntimeError(f"Missing input0 record for {name}")
    return records[name]["input0"]


def extract_kv_pairs(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Support multiple Transformers cache layouts."""
    if isinstance(past_key_values, (tuple, list)):
        return [(layer_kv[0], layer_kv[1]) for layer_kv in past_key_values]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))

    if hasattr(past_key_values, "layers"):
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer in past_key_values.layers:
            k = None
            v = None

            for k_name in ["keys", "key_cache", "k_cache"]:
                if hasattr(layer, k_name):
                    k = getattr(layer, k_name)
                    break

            for v_name in ["values", "value_cache", "v_cache"]:
                if hasattr(layer, v_name):
                    v = getattr(layer, v_name)
                    break

            if k is not None and v is not None:
                pairs.append((k, v))

        if pairs:
            return pairs

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def manual_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    mean_square = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x_fp32 * inv_rms * weight_fp32


def manual_linear_from_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    y = x.to(torch.float32) @ weight.to(torch.float32).t()
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y


def manual_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    return torch.cat((-x_second, x_first), dim=-1)


def manual_rope_cos_sin(
    rotary_emb: Any,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = tensor_to_float32(rotary_emb.inv_freq)
    attention_scaling = float(rotary_emb.attention_scaling)

    batch = position_ids.shape[0]
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float().cpu()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def manual_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rope = (q * cos) + (manual_rotate_half(q) * sin)
    k_rope = (k * cos) + (manual_rotate_half(k) * sin)
    return q_rope, k_rope


def append_cache(old_cache: torch.Tensor | None, current_value: torch.Tensor) -> torch.Tensor:
    if old_cache is None:
        return current_value.clone()
    return torch.cat((old_cache, current_value), dim=2)


def manual_repeat_kv(x: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if repeat_count == 1:
        return x

    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, repeat_count, seq_len, head_dim)
    return x.reshape(batch, kv_heads * repeat_count, seq_len, head_dim)


def manual_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    seq_len = scores.shape[-1]
    future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(
        future_mask.view(1, 1, seq_len, seq_len),
        torch.finfo(scores.dtype).min,
    )
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(weights, v)
    return output, weights


def manual_cached_attention(
    q_current: torch.Tensor,
    k_cache_repeated: torch.Tensor,
    v_cache_repeated: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q_current, k_cache_repeated.transpose(2, 3)) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_current.dtype)
    output = torch.matmul(weights, v_cache_repeated)
    return output, weights


def manual_decoder_layer_cached(
    hidden: torch.Tensor,
    layer: Any,
    rotary_emb: Any,
    position_ids: torch.Tensor,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_repeat: int,
    model_config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = layer.self_attn
    mlp = layer.mlp
    rms_eps = get_rms_norm_eps(layer.input_layernorm, model_config)

    residual = hidden
    input_norm = manual_rms_norm(hidden, tensor_to_float32(layer.input_layernorm.weight), rms_eps)

    q_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn.q_proj.weight))
    k_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn.k_proj.weight))
    v_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn.v_proj.weight))

    q_view = q_flat.view(batch_size, 1, num_q_heads, head_dim)
    k_view = k_flat.view(batch_size, 1, num_kv_heads, head_dim)
    v_view = v_flat.view(batch_size, 1, num_kv_heads, head_dim)

    q_norm = manual_rms_norm(q_view, tensor_to_float32(attn.q_norm.weight), rms_eps)
    k_norm = manual_rms_norm(k_view, tensor_to_float32(attn.k_norm.weight), rms_eps)

    q_states = q_norm.transpose(1, 2)
    k_states = k_norm.transpose(1, 2)
    v_states = v_view.transpose(1, 2)

    cos, sin = manual_rope_cos_sin(rotary_emb, hidden, position_ids)
    q_rope, current_k_rope = manual_apply_rope(q_states, k_states, cos, sin)

    k_cache = append_cache(k_cache, current_k_rope)
    v_cache = append_cache(v_cache, v_states)

    k_cache_repeated = manual_repeat_kv(k_cache, kv_repeat)
    v_cache_repeated = manual_repeat_kv(v_cache, kv_repeat)
    attn_output_heads, _attn_weights = manual_cached_attention(
        q_rope,
        k_cache_repeated,
        v_cache_repeated,
        float(attn.scaling),
    )
    attn_concat = attn_output_heads.transpose(1, 2).contiguous().reshape(batch_size, 1, -1)

    o_proj = manual_linear_from_weight(attn_concat, tensor_to_float32(attn.o_proj.weight))
    after_attention = residual + o_proj

    mlp_residual = after_attention
    post_norm = manual_rms_norm(
        after_attention,
        tensor_to_float32(layer.post_attention_layernorm.weight),
        rms_eps,
    )

    gate = manual_linear_from_weight(post_norm, tensor_to_float32(mlp.gate_proj.weight))
    up = manual_linear_from_weight(post_norm, tensor_to_float32(mlp.up_proj.weight))
    mlp_hidden = F.silu(gate) * up
    mlp_down = manual_linear_from_weight(mlp_hidden, tensor_to_float32(mlp.down_proj.weight))
    layer_output = mlp_residual + mlp_down
    return layer_output, k_cache, v_cache


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()


def format_values(x: torch.Tensor, count: int = SAMPLE_COUNT) -> str:
    values = x.to(torch.float32).flatten()[:count].tolist()
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def compare_tensor(
    name: str,
    manual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    show_values: bool = True,
) -> bool:
    manual = manual.to(torch.float32)
    reference = reference.to(torch.float32)
    ok = torch.allclose(manual, reference, atol=atol, rtol=rtol)

    print("-" * 80)
    print(name)
    print(f"  manual shape: {list(manual.shape)}")
    print(f"  reference shape: {list(reference.shape)}")
    print(f"  max abs error: {max_abs_error(manual, reference):.10e}")
    print(f"  mean abs error: {mean_abs_error(manual, reference):.10e}")
    print(f"  allclose: {ok}")
    if show_values:
        print(f"  manual first {SAMPLE_COUNT}: {format_values(manual)}")
        print(f"  reference first {SAMPLE_COUNT}: {format_values(reference)}")
    return bool(ok)


def finish(ok: bool, module_name: str) -> None:
    if not ok:
        raise RuntimeError(f"{module_name} validation failed.")
    print()
    print(f"PASS: {module_name} validation matched Hugging Face.")
