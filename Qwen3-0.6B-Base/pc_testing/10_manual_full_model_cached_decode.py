from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1]

PROMPT = "The future of FPGA is"
SAMPLE_COUNT = 6
SELECTED_LAYERS = [0, 1, 27]


EXPLANATION = """
手写完整模型：单 token cached decode 参考实现
============================================

这个脚本是 09_manual_layer0_cached_decode.py 的下一步。

09 做了什么：

  只手写 Layer 0。
  每次输入一个 token。
  手动维护 Layer 0 的 K/V cache。
  验证 Layer 0 的输出和 Hugging Face past_key_values 对齐。

10 做什么：

  手写完整 Qwen3-0.6B-Base 的单 token cached decode。
  每次仍然只输入一个 token。
  但是这次会从 Layer 0 一直算到 Layer 27。
  每一层都有自己独立的 K/V cache。
  最后再做 final RMSNorm、LM head、argmax，得到 next token。

也就是说，本脚本在验证 FPGA 以后真正想实现的概念函数：

  next_token = run_one_token(input_token, position)

完整数据流：

  input token id
    -> embedding lookup
    -> Layer 0 cached decode
    -> Layer 1 cached decode
    -> ...
    -> Layer 27 cached decode
    -> final RMSNorm
    -> tied LM head
    -> argmax
    -> next token id

每一层的输入/输出：

  每一层输入 hidden:

    hidden: [B, 1, H] = [1, 1, 1024]

  每一层输出 hidden:

    hidden: [B, 1, H] = [1, 1, 1024]

  Layer i 的输出会成为 Layer i+1 的输入。

每一层的 KV Cache：

  Qwen3-0.6B-Base 有 28 层，所以真实 decode 时有 28 份 K/V cache。
  第 0 层的 cache 和第 1 层的 cache 不是同一个东西。
  因为每一层的输入 hidden 不同，所以每一层算出来的 K/V 也不同。

  对每一层 layer_id：

    K_cache[layer_id]: [B, KVH, T, D] = [1, 8, T, 128]
    V_cache[layer_id]: [B, KVH, T, D] = [1, 8, T, 128]

  T 是目前已经处理过多少个 token。
  prompt 有 5 个 token 时，prefill 完成后每一层都是：

    K_cache[layer_id]: [1, 8, 5, 128]
    V_cache[layer_id]: [1, 8, 5, 128]

  如果再把生成出来的第一个 token 喂回去，T 会变成 6。

每一层 cached attention 的数学：

  当前层输入 hidden 先经过 input RMSNorm：

    norm = hidden * rsqrt(mean(hidden^2) + eps) * gamma

  然后算当前 token 的 Q/K/V：

    q = norm @ Wq.T
    k = norm @ Wk.T
    v = norm @ Wv.T

  reshape:

    q: [1, 1, 2048] -> [1, 1, 16, 128] -> [1, 16, 1, 128]
    k: [1, 1, 1024] -> [1, 1, 8, 128]  -> [1, 8, 1, 128]
    v: [1, 1, 1024] -> [1, 1, 8, 128]  -> [1, 8, 1, 128]

  Q/K 做 q_norm/k_norm，V 不做。
  Q/K 做 RoPE，V 不做。

  K cache 写入的是 RoPE 之后的 K：

    K_cache[layer_id][:, :, position, :] = k_rope[:, :, 0, :]

  V cache 写入的是 v_proj reshape/transpose 之后的 V：

    V_cache[layer_id][:, :, position, :] = v[:, :, 0, :]

  当前 token 的 Q 不写 cache，因为 Q 只在当前 token 的 attention 中用一次。

  Attention 读取当前层自己的 cache：

    score[k_pos] = dot(q_current, K_cache[layer_id][k_pos]) / sqrt(128)
    prob = softmax(score)
    attn_out = sum_k_pos(prob[k_pos] * V_cache[layer_id][k_pos])

  cached decode 不需要额外 causal mask。
  因为 cache 里只有过去 token 和当前 token，没有未来 token。

Final RMSNorm 和 LM head：

  28 层全部算完后，hidden 还是 [1, 1, 1024]。
  final RMSNorm:

    final_hidden = RMSNorm(hidden)

  LM head:

    logits[vocab_id] = dot(final_hidden, embedding_table[vocab_id])

  因为 Qwen3-0.6B-Base tie_word_embeddings = true，
  所以 LM head 和 embedding table 使用同一套词表权重。

  logits shape:

    [B, 1, vocab_size] = [1, 1, 151936]

  greedy argmax:

    next_token = argmax(logits)
"""


def tensor_to_float32(x: torch.Tensor) -> torch.Tensor:
    """把 Tensor 放在 CPU float32 上；如果已经是 CPU float32，一般不会额外复制。"""
    return x.detach().to(dtype=torch.float32, device="cpu")


def clone_reference(x: torch.Tensor) -> torch.Tensor:
    """复制一份 CPU float32 Tensor，用来保存 Hugging Face 的对照结果。"""
    return tensor_to_float32(x).clone()


def get_backbone(model: Any) -> Any:
    """Qwen3ForCausalLM 的 transformer 主体通常存放在 model.model。"""
    return model.model if hasattr(model, "model") else model


def get_rms_norm_eps(norm_module: Any, model_config: Any) -> float:
    """读取 RMSNorm epsilon。"""
    if hasattr(norm_module, "variance_epsilon"):
        return float(norm_module.variance_epsilon)
    if hasattr(norm_module, "eps"):
        return float(norm_module.eps)
    return float(model_config.rms_norm_eps)


def extract_kv_pairs(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """兼容不同 Transformers 版本的 KV cache 返回格式。"""
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
    """
    手写 RMSNorm。

    数学:
      mean_square = mean(x * x)
      y = x * rsqrt(mean_square + eps) * weight
    """
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
    """
    手写 Linear / GEMV。

    PyTorch Linear 权重 shape 是 [out_features, in_features]。

    数学:
      y[o] = sum_i(x[i] * W[o, i]) + bias[o]
    """
    y = x.to(torch.float32) @ weight.to(torch.float32).t()
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y


def manual_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE 里的 rotate_half。

    如果 D = 128:

      x = [x0, ..., x63, x64, ..., x127]

    那么:

      rotate_half(x) = [-x64, ..., -x127, x0, ..., x63]
    """
    half = x.shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    return torch.cat((-x_second, x_first), dim=-1)


def manual_rope_cos_sin(rotary_emb: Any, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    手写 Qwen3 RoPE cos/sin。

    position_ids 是当前 token 的绝对位置，例如 [[0]]、[[1]]、[[5]]。
    """
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
    """对当前 token 的 Q/K 应用 RoPE。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rope = (q * cos) + (manual_rotate_half(q) * sin)
    k_rope = (k * cos) + (manual_rotate_half(k) * sin)
    return q_rope, k_rope


def append_cache(old_cache: torch.Tensor | None, current_value: torch.Tensor) -> torch.Tensor:
    """
    沿 token 时间维度追加 K 或 V。

    old_cache:     None 或 [B, KVH, T_old, D]
    current_value: [B, KVH, 1,     D]
    new_cache:     [B, KVH, T_old + 1, D]
    """
    if old_cache is None:
        return current_value.clone()
    return torch.cat((old_cache, current_value), dim=2)


def manual_repeat_kv(x: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """GQA: 把 KV heads 从 KVH 复制到 QH。"""
    if repeat_count == 1:
        return x

    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, repeat_count, seq_len, head_dim)
    return x.reshape(batch, kv_heads * repeat_count, seq_len, head_dim)


def manual_cached_attention(
    q_current: torch.Tensor,
    k_cache_repeated: torch.Tensor,
    v_cache_repeated: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    手写 cached attention。

    q_current:        [B, QH, 1, D]
    k_cache_repeated: [B, QH, T, D]
    v_cache_repeated: [B, QH, T, D]

    score = q @ k.T / sqrt(D)
    prob = softmax(score)
    output = prob @ v
    """
    scores = torch.matmul(q_current, k_cache_repeated.transpose(2, 3)) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_current.dtype)
    output = torch.matmul(weights, v_cache_repeated)
    return output, weights


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()


def format_values(x: torch.Tensor, count: int = SAMPLE_COUNT) -> str:
    values = x.to(torch.float32).flatten()[:count].tolist()
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def compare_tensor_summary(
    name: str,
    manual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    show_values: bool = True,
) -> bool:
    """打印两个 Tensor 的概要误差。"""
    manual = manual.to(torch.float32)
    reference = reference.to(torch.float32)
    ok = torch.allclose(manual, reference, atol=atol, rtol=rtol)

    print("-" * 80)
    print(name)
    print(f"  manual shape: {list(manual.shape)}")
    print(f"  HF shape:     {list(reference.shape)}")
    print(f"  max abs error:  {max_abs_error(manual, reference):.10e}")
    print(f"  mean abs error: {mean_abs_error(manual, reference):.10e}")
    print(f"  torch.allclose(atol={atol}, rtol={rtol}): {ok}")
    if show_values:
        print(f"  manual first {SAMPLE_COUNT}: {format_values(manual)}")
        print(f"  HF first {SAMPLE_COUNT}:     {format_values(reference)}")
    return bool(ok)


def layer_cache_error(
    manual_k_cache: torch.Tensor,
    manual_v_cache: torch.Tensor,
    hf_k_cache: torch.Tensor,
    hf_v_cache: torch.Tensor,
) -> tuple[float, float]:
    """返回某一层 K cache 和 V cache 的最大绝对误差。"""
    k_error = max_abs_error(manual_k_cache, hf_k_cache)
    v_error = max_abs_error(manual_v_cache, hf_v_cache)
    return k_error, v_error


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
    """
    手写单个 decoder layer 的 cached decode。

    输入 hidden shape:
      [B, 1, H]

    输出 hidden shape:
      [B, 1, H]

    同时返回更新后的本层 K/V cache。
    """
    attn = layer.self_attn
    mlp = layer.mlp

    rms_eps = get_rms_norm_eps(layer.input_layernorm, model_config)

    # 1. Attention 前的 RMSNorm
    residual = hidden
    input_norm = manual_rms_norm(
        hidden,
        tensor_to_float32(layer.input_layernorm.weight),
        rms_eps,
    )

    # 2. 当前 token 的 q_proj/k_proj/v_proj
    q_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn.q_proj.weight))
    k_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn.k_proj.weight))
    v_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn.v_proj.weight))

    # 3. reshape 到 head 形状
    q_view = q_flat.view(batch_size, 1, num_q_heads, head_dim)
    k_view = k_flat.view(batch_size, 1, num_kv_heads, head_dim)
    v_view = v_flat.view(batch_size, 1, num_kv_heads, head_dim)

    # 4. q_norm/k_norm，V 不做这一步
    q_norm = manual_rms_norm(q_view, tensor_to_float32(attn.q_norm.weight), rms_eps)
    k_norm = manual_rms_norm(k_view, tensor_to_float32(attn.k_norm.weight), rms_eps)

    q_states = q_norm.transpose(1, 2)
    k_states = k_norm.transpose(1, 2)
    v_states = v_view.transpose(1, 2)

    # 5. RoPE。K 做完 RoPE 后写入 cache，V 原样写入 cache。
    cos, sin = manual_rope_cos_sin(rotary_emb, hidden, position_ids)
    q_rope, current_k_rope = manual_apply_rope(q_states, k_states, cos, sin)

    k_cache = append_cache(k_cache, current_k_rope)
    v_cache = append_cache(v_cache, v_states)

    # 6. 读取本层 cache，做 GQA repeat 和 cached attention。
    k_cache_repeated = manual_repeat_kv(k_cache, kv_repeat)
    v_cache_repeated = manual_repeat_kv(v_cache, kv_repeat)
    attn_output_heads, _attn_weights = manual_cached_attention(
        q_rope,
        k_cache_repeated,
        v_cache_repeated,
        float(attn.scaling),
    )
    attn_concat = attn_output_heads.transpose(1, 2).contiguous().reshape(batch_size, 1, -1)

    # 7. o_proj + residual
    o_proj = manual_linear_from_weight(attn_concat, tensor_to_float32(attn.o_proj.weight))
    after_attention = residual + o_proj

    # 8. MLP 前 RMSNorm
    mlp_residual = after_attention
    post_norm = manual_rms_norm(
        after_attention,
        tensor_to_float32(layer.post_attention_layernorm.weight),
        rms_eps,
    )

    # 9. Qwen gated MLP: down(silu(gate) * up)
    gate = manual_linear_from_weight(post_norm, tensor_to_float32(mlp.gate_proj.weight))
    up = manual_linear_from_weight(post_norm, tensor_to_float32(mlp.up_proj.weight))
    mlp_hidden = F.silu(gate) * up
    mlp_down = manual_linear_from_weight(mlp_hidden, tensor_to_float32(mlp.down_proj.weight))

    layer_output = mlp_residual + mlp_down
    return layer_output, k_cache, v_cache


def summarize_all_layer_caches(
    manual_k_caches: list[torch.Tensor | None],
    manual_v_caches: list[torch.Tensor | None],
    hf_kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    selected_layers: list[int],
) -> bool:
    """对比 28 层 cache，并打印几个代表层的 shape 和误差。"""
    all_ok = True
    worst_k_error = 0.0
    worst_v_error = 0.0
    worst_k_layer = -1
    worst_v_layer = -1

    for layer_id, (hf_k, hf_v) in enumerate(hf_kv_pairs):
        manual_k = manual_k_caches[layer_id]
        manual_v = manual_v_caches[layer_id]
        if manual_k is None or manual_v is None:
            raise RuntimeError(f"Missing manual cache for layer {layer_id}")

        hf_k_ref = clone_reference(hf_k)
        hf_v_ref = clone_reference(hf_v)
        k_error, v_error = layer_cache_error(manual_k, manual_v, hf_k_ref, hf_v_ref)

        if k_error > worst_k_error:
            worst_k_error = k_error
            worst_k_layer = layer_id
        if v_error > worst_v_error:
            worst_v_error = v_error
            worst_v_layer = layer_id

        layer_ok = torch.allclose(manual_k, hf_k_ref, atol=1e-5, rtol=1e-5) and torch.allclose(
            manual_v,
            hf_v_ref,
            atol=1e-5,
            rtol=1e-5,
        )
        all_ok = all_ok and bool(layer_ok)

        if layer_id in selected_layers:
            print(
                f"  Layer {layer_id:02d} cache: "
                f"K {list(manual_k.shape)} max_err={k_error:.3e}, "
                f"V {list(manual_v.shape)} max_err={v_error:.3e}"
            )

    print(
        "  Worst cache error: "
        f"K layer {worst_k_layer} max_err={worst_k_error:.3e}, "
        f"V layer {worst_v_layer} max_err={worst_v_error:.3e}"
    )
    return all_ok


def run_manual_one_token(
    current_token_ids: torch.Tensor,
    position: int,
    embedding_table: torch.Tensor,
    backbone: Any,
    model: Any,
    manual_k_caches: list[torch.Tensor | None],
    manual_v_caches: list[torch.Tensor | None],
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    手写完整 run_one_token。

    返回:
      logits: [B, 1, vocab_size]
      final_hidden: [B, 1, H]
      next_token: [B, 1]
      hidden_before_final_norm: [B, 1, H]
    """
    position_ids = torch.tensor([[position]], dtype=torch.long)

    hidden = embedding_table[current_token_ids.cpu()]

    for layer_id, layer in enumerate(backbone.layers):
        hidden, new_k_cache, new_v_cache = manual_decoder_layer_cached(
            hidden=hidden,
            layer=layer,
            rotary_emb=backbone.rotary_emb,
            position_ids=position_ids,
            k_cache=manual_k_caches[layer_id],
            v_cache=manual_v_caches[layer_id],
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_repeat=kv_repeat,
            model_config=model.config,
        )
        manual_k_caches[layer_id] = new_k_cache
        manual_v_caches[layer_id] = new_v_cache

    hidden_before_final_norm = hidden
    final_norm_eps = get_rms_norm_eps(backbone.norm, model.config)
    final_hidden = manual_rms_norm(
        hidden_before_final_norm,
        tensor_to_float32(backbone.norm.weight),
        final_norm_eps,
    )

    logits = manual_linear_from_weight(final_hidden, tensor_to_float32(model.lm_head.weight))
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return logits, final_hidden, next_token, hidden_before_final_norm


def print_step_header(position: int, token_id: int, token_text: str, is_generated_token: bool) -> None:
    print()
    print("=" * 80)
    print(f"Token position = {position}")
    print(f"Token id = {token_id}")
    print(f"Token text = {token_text!r}")
    print(f"Token source = {'generated token fed back into model' if is_generated_token else 'prompt token'}")


def main() -> None:
    print(EXPLANATION)

    device = torch.device("cpu")
    dtype = torch.float32

    print("Run settings")
    print("=" * 80)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Device: {device}")
    print(f"Load dtype: {dtype}")
    print()

    tokenizer: Any = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    loaded_model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    )
    model: Any = loaded_model.to(device)
    model.eval()

    backbone: Any = get_backbone(model)

    encoded: Any = tokenizer(PROMPT, return_tensors="pt")
    input_ids = cast(torch.Tensor, encoded.input_ids).to(device)
    batch_size, prompt_len = input_ids.shape

    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    intermediate_size = int(model.config.intermediate_size)
    vocab_size = int(model.config.vocab_size)
    kv_repeat = num_q_heads // num_kv_heads

    print("Input prompt")
    print("=" * 80)
    print(f"Prompt: {PROMPT!r}")
    print(f"Prompt token ids: {input_ids[0].tolist()}")
    print(f"Prompt token text: {[tokenizer.decode([int(token_id)]) for token_id in input_ids[0].tolist()]}")
    print()

    print("Model dimensions")
    print("=" * 80)
    print(f"B batch size: {batch_size}")
    print("S_new current input tokens per step: 1")
    print(f"H hidden size: {hidden_size}")
    print(f"Layers: {num_layers}")
    print(f"QH query heads: {num_q_heads}")
    print(f"KVH key/value heads: {num_kv_heads}")
    print(f"D head dim: {head_dim}")
    print(f"I MLP intermediate size: {intermediate_size}")
    print(f"Vocab size: {vocab_size}")
    print(f"KV repeat count: {kv_repeat}")
    print()

    embedding_table = tensor_to_float32(backbone.embed_tokens.weight)

    manual_k_caches: list[torch.Tensor | None] = [None for _ in range(num_layers)]
    manual_v_caches: list[torch.Tensor | None] = [None for _ in range(num_layers)]
    hf_past_key_values: Any = None

    all_checks_ok = True
    last_manual_next_token: torch.Tensor | None = None
    last_hf_next_token: torch.Tensor | None = None

    with torch.no_grad():
        # 先 serial prefill prompt。
        for position in range(prompt_len):
            current_token_ids = input_ids[:, position : position + 1]
            token_id = int(current_token_ids[0, 0].item())
            token_text = tokenizer.decode([token_id])
            print_step_header(position, token_id, token_text, is_generated_token=False)

            hf_outputs: Any = model(
                input_ids=current_token_ids,
                past_key_values=hf_past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            hf_past_key_values = hf_outputs.past_key_values
            hf_kv_pairs = extract_kv_pairs(hf_past_key_values)

            manual_logits, manual_final_hidden, manual_next_token, _manual_raw_hidden = run_manual_one_token(
                current_token_ids=current_token_ids,
                position=position,
                embedding_table=embedding_table,
                backbone=backbone,
                model=model,
                manual_k_caches=manual_k_caches,
                manual_v_caches=manual_v_caches,
                batch_size=batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_repeat=kv_repeat,
            )

            hf_logits = clone_reference(hf_outputs.logits)
            hf_final_hidden = clone_reference(hf_outputs.hidden_states[-1])
            hf_next_token = torch.argmax(hf_logits[:, -1, :], dim=-1, keepdim=True)

            print("Cache comparison for selected layers")
            cache_ok = summarize_all_layer_caches(
                manual_k_caches,
                manual_v_caches,
                hf_kv_pairs,
                SELECTED_LAYERS,
            )

            final_hidden_ok = compare_tensor_summary(
                f"position {position}: final RMSNorm output",
                manual_final_hidden,
                hf_final_hidden,
                show_values=False,
            )
            logits_ok = compare_tensor_summary(
                f"position {position}: logits",
                manual_logits,
                hf_logits,
                show_values=position == prompt_len - 1,
            )

            manual_token_id = int(manual_next_token.item())
            hf_token_id = int(hf_next_token.item())
            print("-" * 80)
            print("Greedy argmax")
            print(f"  manual next token id/text: {manual_token_id} / {tokenizer.decode([manual_token_id])!r}")
            print(f"  HF next token id/text:     {hf_token_id} / {tokenizer.decode([hf_token_id])!r}")
            print(f"  match: {manual_token_id == hf_token_id}")

            all_checks_ok = (
                all_checks_ok
                and cache_ok
                and final_hidden_ok
                and logits_ok
                and (manual_token_id == hf_token_id)
            )
            last_manual_next_token = manual_next_token
            last_hf_next_token = hf_next_token

        if last_manual_next_token is None or last_hf_next_token is None:
            raise RuntimeError("No prompt tokens were processed.")

        # 再把 prompt 之后预测出的第一个 token 喂回模型，验证真正 decode 这一步。
        generated_position = prompt_len
        generated_token_ids = last_manual_next_token.to(device)
        generated_token_id = int(generated_token_ids.item())
        generated_token_text = tokenizer.decode([generated_token_id])
        print_step_header(
            generated_position,
            generated_token_id,
            generated_token_text,
            is_generated_token=True,
        )

        if int(last_manual_next_token.item()) != int(last_hf_next_token.item()):
            raise RuntimeError("Manual and HF generated token differ before feedback decode.")

        hf_outputs = model(
            input_ids=generated_token_ids,
            past_key_values=hf_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        hf_past_key_values = hf_outputs.past_key_values
        hf_kv_pairs = extract_kv_pairs(hf_past_key_values)

        manual_logits, manual_final_hidden, manual_next_token, _manual_raw_hidden = run_manual_one_token(
            current_token_ids=generated_token_ids,
            position=generated_position,
            embedding_table=embedding_table,
            backbone=backbone,
            model=model,
            manual_k_caches=manual_k_caches,
            manual_v_caches=manual_v_caches,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_repeat=kv_repeat,
        )

        hf_logits = clone_reference(hf_outputs.logits)
        hf_final_hidden = clone_reference(hf_outputs.hidden_states[-1])
        hf_next_token = torch.argmax(hf_logits[:, -1, :], dim=-1, keepdim=True)

        print("Cache comparison for selected layers after generated-token feedback")
        cache_ok = summarize_all_layer_caches(
            manual_k_caches,
            manual_v_caches,
            hf_kv_pairs,
            SELECTED_LAYERS,
        )
        final_hidden_ok = compare_tensor_summary(
            f"position {generated_position}: final RMSNorm output",
            manual_final_hidden,
            hf_final_hidden,
            show_values=False,
        )
        logits_ok = compare_tensor_summary(
            f"position {generated_position}: logits",
            manual_logits,
            hf_logits,
            show_values=True,
        )

        manual_token_id = int(manual_next_token.item())
        hf_token_id = int(hf_next_token.item())
        print("-" * 80)
        print("Greedy argmax after generated-token feedback")
        print(f"  manual next token id/text: {manual_token_id} / {tokenizer.decode([manual_token_id])!r}")
        print(f"  HF next token id/text:     {hf_token_id} / {tokenizer.decode([hf_token_id])!r}")
        print(f"  match: {manual_token_id == hf_token_id}")

        all_checks_ok = (
            all_checks_ok
            and cache_ok
            and final_hidden_ok
            and logits_ok
            and (manual_token_id == hf_token_id)
        )

    print()
    print("=" * 80)
    print("KV Cache memory math")
    values_per_token_per_layer = 2 * num_kv_heads * head_dim
    values_per_token_all_layers = num_layers * values_per_token_per_layer
    print(f"Per token per layer KV values = 2 * KVH * D = 2 * {num_kv_heads} * {head_dim} = {values_per_token_per_layer}")
    print(f"Per token all layers KV values = {num_layers} * {values_per_token_per_layer} = {values_per_token_all_layers}")
    for context_len in [128, 256, 512, 1024]:
        fp16_bytes = context_len * values_per_token_all_layers * 2
        int8_bytes = context_len * values_per_token_all_layers
        print(
            f"context_len={context_len}: "
            f"FP16/BF16 KV ~= {fp16_bytes / 1024 / 1024:.2f} MiB, "
            f"INT8 KV ~= {int8_bytes / 1024 / 1024:.2f} MiB"
        )

    if not all_checks_ok:
        raise RuntimeError("At least one full-model cached decode check failed.")

    print()
    print("Validation complete: manual full-model cached decode matches Hugging Face.")


if __name__ == "__main__":
    main()
