from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1]

PROMPT = "The future of FPGA is"
SAMPLE_COUNT = 6


EXPLANATION = """
手写 Layer 0：单 token cached decode 参考实现
=============================================

这个脚本和 08_manual_layer0_full.py 的关系：

  08:
    一次性把整个 prompt 的 5 个 token 放进 Layer 0。
    输入 shape 是 [B, S, H] = [1, 5, 1024]。
    Attention 里面一次性得到 5 个 token 的 Q/K/V，然后用 causal mask 防止前面的
    token 看到未来 token。

  09:
    模拟 FPGA 更容易实现的 run_one_token(input_token, position) 路径。
    每次只处理 1 个 token。
    当前 token 的输入 shape 是 [B, 1, H] = [1, 1, 1024]。
    当前 token 只新算自己的 Q/K/V。
    K/V 会被写入 KV Cache。
    Attention 读取“从 token 0 到当前 token”的全部 K/V cache。

为什么要有 KV Cache：

  Transformer decode 是自回归的。
  当我们正在处理第 t 个 token 时，它需要看见 token 0..t 的 K/V。

  如果没有 KV Cache：
    处理 token t 时，需要重新计算 token 0..t 的 K/V。
    下一个 token t+1 又要重新计算 token 0..t+1 的 K/V。
    前面 token 的 K/V 会被反复重算，非常浪费。

  如果有 KV Cache：
    token 0 的 K/V 算完后存起来。
    token 1 只需要新算 token 1 的 K/V，再把它追加到 cache。
    token 2 只需要新算 token 2 的 K/V，再把它追加到 cache。
    以后 attention 直接读 cache，不重算历史 token 的 K/V。

Layer 0 的 KV Cache 里到底存什么：

  对 Qwen3-0.6B-Base：

    B   = batch size = 1
    KVH = key/value heads = 8
    D   = head dim = 128
    T   = 当前 cache 已经存了多少个 token

  K cache shape:

    [B, KVH, T, D] = [1, 8, T, 128]

  V cache shape:

    [B, KVH, T, D] = [1, 8, T, 128]

  注意：

    K cache 存的是“已经做完 RoPE 之后的 K”。
    V cache 存的是“v_proj 之后 reshape/transpose 的 V”。
    V 不做 RoPE，也不做 q_norm/k_norm。
    Q 不存进 cache，因为 Q 只给当前 token 马上用一次。

KV Cache 的写入流程：

  假设当前处理的位置是 position = t。

  1. 当前 token 经过 input RMSNorm。
  2. 当前 token 经过 q_proj/k_proj/v_proj。
  3. Q reshape 成 [B, 1, QH, D]，K/V reshape 成 [B, 1, KVH, D]。
  4. Q/K 做各自的 RMSNorm，V 不做。
  5. Q/K 转成 attention 用的形状：

       Q: [B, QH, 1, D]
       K: [B, KVH, 1, D]
       V: [B, KVH, 1, D]

  6. 根据 position = t 生成 RoPE 的 cos/sin。
  7. 对当前 Q/K 做 RoPE，得到 q_rope 和 k_rope。
  8. 把当前 k_rope 和当前 V 追加到 cache：

       old_k_cache: [B, KVH, T_old, D]
       current_k:   [B, KVH, 1,     D]
       new_k_cache: [B, KVH, T_old + 1, D]

       old_v_cache: [B, KVH, T_old, D]
       current_v:   [B, KVH, 1,     D]
       new_v_cache: [B, KVH, T_old + 1, D]

  数学上，这个 append 不是加法，而是沿着 token 时间方向拼接：

       new_k_cache[:, :, 0:T_old, :] = old_k_cache
       new_k_cache[:, :, T_old,   :] = current_k[:, :, 0, :]

       new_v_cache[:, :, 0:T_old, :] = old_v_cache
       new_v_cache[:, :, T_old,   :] = current_v[:, :, 0, :]

KV Cache 的读取和使用流程：

  当前 token 的 Q 只有一个位置：

       q_current: [B, QH, 1, D]

  Cache 里有从 0 到 t 的 K/V：

       k_cache: [B, KVH, T, D]
       v_cache: [B, KVH, T, D]

  因为 QH = 16，KVH = 8，所以这是 GQA。
  每一个 K/V head 服务 2 个 Q head。
  读出来的 K/V 会先 repeat 成：

       repeated_k_cache: [B, QH, T, D]
       repeated_v_cache: [B, QH, T, D]

  然后当前 token 对历史所有 token 打分：

       score[b, h, 0, k_pos]
         = dot(q_current[b, h, 0, :], repeated_k_cache[b, h, k_pos, :])
           / sqrt(D)

  这里 k_pos = 0..T-1。
  冒号 ":" 的意思是“取这一整段 128 维 head 向量”。
  dot 是点积：

       dot(q, k) = sum_{d=0}^{127} q[d] * k[d]

  然后 softmax：

       prob[k_pos] = exp(score[k_pos]) / sum_j(exp(score[j]))

  最后用概率加权求和 V：

       attn_out[b, h, 0, d]
         = sum_{k_pos=0}^{T-1} prob[b, h, 0, k_pos]
           * repeated_v_cache[b, h, k_pos, d]

cached decode 为什么不需要 causal mask：

  08 里一次性输入了 5 个 token。
  token 0、1、2、3、4 同时存在，所以必须用 causal mask 遮住未来 token。

  09 里当前只处理一个 token。
  Cache 里只有已经处理过的历史 token 加当前 token。
  未来 token 还没有算出来，也没有写进 cache。
  所以 attention 根本读不到未来 token，不需要额外 mask。

FPGA 视角：

  对每一层、每一个 token，都要做两类 cache 操作：

    写：
      把当前 token 的 K/V 写入 cache RAM。
      概念地址可以看成:
        K_cache[layer_id][kv_head][position][dim]
        V_cache[layer_id][kv_head][position][dim]

    读：
      当前 token 做 attention 时，读取:
        K_cache[layer_id][kv_head][0..position][dim]
        V_cache[layer_id][kv_head][0..position][dim]

  本脚本只手写 Layer 0。
  真实模型有 28 层，所以真实硬件需要 28 份独立的 K/V cache。
"""


def to_reference(x: torch.Tensor) -> torch.Tensor:
    """把 Tensor 复制到 CPU float32，方便稳定比较。"""
    return x.detach().to(dtype=torch.float32, device="cpu").clone()


def get_backbone(model: Any) -> Any:
    """Qwen3ForCausalLM 的 transformer 主体通常存放在 model.model。"""
    return model.model if hasattr(model, "model") else model


def get_rms_norm_eps(norm_module: Any, model_config: Any) -> float:
    """读取 RMSNorm 的 epsilon。不同模块命名可能略有不同。"""
    if hasattr(norm_module, "variance_epsilon"):
        return float(norm_module.variance_epsilon)
    if hasattr(norm_module, "eps"):
        return float(norm_module.eps)
    return float(model_config.rms_norm_eps)


def capture_module_io(
    records: dict[str, dict[str, torch.Tensor]],
    name: str,
) -> Callable[[Any, Any, Any], None]:
    """注册 forward hook：保存模块的第一个输入和输出。"""

    def hook(module: Any, inputs: Any, output: Any) -> None:
        item: dict[str, torch.Tensor] = {}

        if isinstance(inputs, tuple) and inputs and torch.is_tensor(inputs[0]):
            item["input0"] = to_reference(cast(torch.Tensor, inputs[0]))

        if torch.is_tensor(output):
            item["output"] = to_reference(cast(torch.Tensor, output))
        elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            item["output"] = to_reference(cast(torch.Tensor, output[0]))
        else:
            raise TypeError(f"{name} hook expected tensor output, got {type(output)}")

        records[name] = item

    return hook


def extract_kv_pairs(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    读取 Hugging Face 返回的 KV cache。

    Transformers 的 cache 格式会随版本变化，所以这里兼容几种常见结构：

      1. 旧版本: tuple/list，每层一个 (k, v)
      2. DynamicCache: key_cache/value_cache
      3. 新结构: layers，每层里面有 keys/values 等字段
    """
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

    输入:
      x: [..., N]
      weight: [N]

    输出:
      y: [..., N]

    数学:
      mean_square = mean(x * x)
      y = x * rsqrt(mean_square + eps) * weight
    """
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    mean_square = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x_fp32 * inv_rms * weight_fp32


def manual_linear(x: torch.Tensor, linear_module: Any) -> torch.Tensor:
    """
    手写 Linear / GEMV。

    输入:
      x: [..., in_features]
      weight: [out_features, in_features]

    输出:
      y: [..., out_features]

    数学:
      y[o] = sum_i(x[i] * W[o, i]) + bias[o]
    """
    weight = to_reference(linear_module.weight)
    y = x.to(torch.float32) @ weight.t()
    if linear_module.bias is not None:
        y = y + to_reference(linear_module.bias)
    return y


def manual_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE 里的 rotate_half。

    假设最后一维 D = 128，half = 64。

      x = [x0, x1, ..., x63, x64, x65, ..., x127]

    拆成:

      x_first  = [x0,  x1,  ..., x63]
      x_second = [x64, x65, ..., x127]

    然后:

      rotate_half(x) = concat(-x_second, x_first)

    展开:

      rotate_half(x)
        = [-x64, -x65, ..., -x127, x0, x1, ..., x63]
    """
    half = x.shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    return torch.cat((-x_second, x_first), dim=-1)


def manual_rope_cos_sin(rotary_emb: Any, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    手写 Qwen3 RoPE 的 cos/sin 生成。

    输入:
      x: [B, 1, H]，这里只用它的 dtype/device 概念
      position_ids: [B, 1]，当前 token 的位置，例如 [[3]]

    输出:
      cos: [B, 1, D]
      sin: [B, 1, D]

    数学，令 D = 128，half = 64:

      inv_freq[i] = 1 / rope_theta^(2i / D),  i = 0..63
      angle[b, 0, i] = position_ids[b, 0] * inv_freq[i]

    这一步得到 64 个角度，然后复制成 128 维：

      emb = [angle0, ..., angle63, angle0, ..., angle63]
    """
    inv_freq = to_reference(rotary_emb.inv_freq)
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
    """
    手写 RoPE。

    输入:
      q: [B, QH, 1, D]
      k: [B, KVH, 1, D]
      cos/sin: [B, 1, D]

    输出:
      q_rope: [B, QH, 1, D]
      k_rope: [B, KVH, 1, D]

    数学:
      q_rope = q * cos + rotate_half(q) * sin
      k_rope = k * cos + rotate_half(k) * sin
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rope = (q * cos) + (manual_rotate_half(q) * sin)
    k_rope = (k * cos) + (manual_rotate_half(k) * sin)
    return q_rope, k_rope


def append_cache(old_cache: torch.Tensor | None, current_value: torch.Tensor) -> torch.Tensor:
    """
    沿着 token 时间维度追加 cache。

    输入:
      old_cache:     None 或 [B, KVH, T_old, D]
      current_value: [B, KVH, 1,     D]

    输出:
      new_cache: [B, KVH, T_old + 1, D]

    数学:
      new_cache[:, :, 0:T_old, :] = old_cache
      new_cache[:, :, T_old,   :] = current_value[:, :, 0, :]
    """
    if old_cache is None:
        return current_value.clone()
    return torch.cat((old_cache, current_value), dim=2)


def manual_repeat_kv(x: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """
    GQA 的 K/V head 复制。

    输入:
      x: [B, KVH, T, D]

    输出:
      repeated: [B, QH, T, D]

    当前模型:
      QH = 16
      KVH = 8
      repeat_count = 2
    """
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    手写 cached self-attention。

    输入:
      q_current:        [B, QH, 1, D]
      k_cache_repeated: [B, QH, T, D]
      v_cache_repeated: [B, QH, T, D]

    输出:
      attn_output: [B, QH, 1, D]
      attn_weights: [B, QH, 1, T]
      scores: [B, QH, 1, T]

    数学:
      score[t] = dot(q_current, k_cache[t]) / sqrt(D)
      prob = softmax(score)
      attn_output = sum_t(prob[t] * v_cache[t])

    这里不加 causal mask，因为 cache 里没有未来 token。
    """
    scores = torch.matmul(q_current, k_cache_repeated.transpose(2, 3)) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_current.dtype)
    output = torch.matmul(weights, v_cache_repeated)
    return output, weights, scores


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


def format_values(x: torch.Tensor, count: int = SAMPLE_COUNT) -> str:
    values = x.flatten()[:count].tolist()
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def compare_tensor(
    name: str,
    manual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """打印手写 Tensor 和 Hugging Face Tensor 的误差，并返回是否 allclose。"""
    manual = manual.to(torch.float32)
    reference = reference.to(torch.float32)
    diff = manual - reference
    ok = torch.allclose(manual, reference, atol=atol, rtol=rtol)

    print("-" * 80)
    print(name)
    print(f"  手写 shape: {list(manual.shape)}")
    print(f"  HF shape:   {list(reference.shape)}")
    print(f"  最大绝对误差 max abs error:  {max_abs_error(manual, reference):.10e}")
    print(f"  平均绝对误差 mean abs error: {mean_abs_error(manual, reference):.10e}")
    print(f"  torch.allclose(atol={atol}, rtol={rtol}): {ok}")
    print(f"  手写前 {SAMPLE_COUNT} 个值: {format_values(manual)}")
    print(f"  HF 前 {SAMPLE_COUNT} 个值:   {format_values(reference)}")
    print(f"  差值前 {SAMPLE_COUNT} 个值: {format_values(diff)}")
    return bool(ok)


def require_output(records: dict[str, dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if name not in records or "output" not in records[name]:
        raise RuntimeError(f"Missing output record for {name}")
    return records[name]["output"]


def require_input0(records: dict[str, dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if name not in records or "input0" not in records[name]:
        raise RuntimeError(f"Missing input0 record for {name}")
    return records[name]["input0"]


def print_token_stage(title: str, tensor: torch.Tensor, math: str) -> None:
    print()
    print(title)
    print(f"  shape: {list(tensor.shape)}")
    print(f"  数学/含义: {math}")


def main() -> None:
    print(EXPLANATION)

    # 这里强制使用 CPU + float32，目标是做稳定的 golden reference。
    device = torch.device("cpu")
    dtype = torch.float32

    print("运行设置")
    print("=" * 80)
    print(f"模型目录: {MODEL_DIR}")
    print(f"设备: {device}")
    print(f"加载 dtype: {dtype}")
    print()

    tokenizer: Any = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    # 强制 eager attention，这样 Hugging Face 内部也使用普通 matmul + softmax，
    # 便于和我们手写的数学公式逐项对齐。
    loaded_model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    )
    model: Any = loaded_model.to(device)
    model.eval()

    backbone: Any = get_backbone(model)
    layer0: Any = backbone.layers[0]
    attn0: Any = layer0.self_attn
    mlp0: Any = layer0.mlp

    encoded: Any = tokenizer(PROMPT, return_tensors="pt")
    input_ids = cast(torch.Tensor, encoded.input_ids).to(device)
    batch_size, total_tokens = input_ids.shape

    hidden_size = int(model.config.hidden_size)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    intermediate_size = int(model.config.intermediate_size)
    num_layers = int(model.config.num_hidden_layers)
    kv_repeat = num_q_heads // num_kv_heads

    print("输入 prompt")
    print("=" * 80)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token ids: {input_ids[0].tolist()}")
    print(f"Token text: {[tokenizer.decode([int(token_id)]) for token_id in input_ids[0].tolist()]}")
    print(f"Total tokens: {total_tokens}")
    print()

    print("模型维度")
    print("=" * 80)
    print(f"B batch size: {batch_size}")
    print("S_new 当前每次输入的 token 数: 1")
    print(f"H hidden size: {hidden_size}")
    print(f"QH query heads: {num_q_heads}")
    print(f"KVH key/value heads: {num_kv_heads}")
    print(f"D head dim: {head_dim}")
    print(f"I MLP intermediate size: {intermediate_size}")
    print(f"Layers: {num_layers}")
    print(f"KV repeat count: {kv_repeat}")
    print(f"attention scaling = 1 / sqrt(D): {attn0.scaling}")
    print()

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        backbone.embed_tokens.register_forward_hook(capture_module_io(records, "embed_tokens")),
        layer0.register_forward_hook(capture_module_io(records, "layer0")),
        layer0.input_layernorm.register_forward_hook(capture_module_io(records, "layer0.input_layernorm")),
        attn0.q_proj.register_forward_hook(capture_module_io(records, "layer0.self_attn.q_proj")),
        attn0.k_proj.register_forward_hook(capture_module_io(records, "layer0.self_attn.k_proj")),
        attn0.v_proj.register_forward_hook(capture_module_io(records, "layer0.self_attn.v_proj")),
        attn0.q_norm.register_forward_hook(capture_module_io(records, "layer0.self_attn.q_norm")),
        attn0.k_norm.register_forward_hook(capture_module_io(records, "layer0.self_attn.k_norm")),
        attn0.o_proj.register_forward_hook(capture_module_io(records, "layer0.self_attn.o_proj")),
        layer0.post_attention_layernorm.register_forward_hook(
            capture_module_io(records, "layer0.post_attention_layernorm")
        ),
        mlp0.gate_proj.register_forward_hook(capture_module_io(records, "layer0.mlp.gate_proj")),
        mlp0.up_proj.register_forward_hook(capture_module_io(records, "layer0.mlp.up_proj")),
        mlp0.down_proj.register_forward_hook(capture_module_io(records, "layer0.mlp.down_proj")),
    ]

    embedding_table = to_reference(backbone.embed_tokens.weight)
    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)
    input_norm_weight = to_reference(layer0.input_layernorm.weight)
    q_norm_weight = to_reference(attn0.q_norm.weight)
    k_norm_weight = to_reference(attn0.k_norm.weight)
    post_norm_weight = to_reference(layer0.post_attention_layernorm.weight)

    manual_k_cache: torch.Tensor | None = None
    manual_v_cache: torch.Tensor | None = None
    hf_past_key_values: Any = None
    all_checks_ok = True

    try:
        with torch.no_grad():
            for position in range(total_tokens):
                current_token_ids = input_ids[:, position : position + 1]
                current_token_id = int(current_token_ids[0, 0].item())
                current_token_text = tokenizer.decode([current_token_id])
                current_position_ids = torch.tensor([[position]], dtype=torch.long)

                print()
                print("=" * 80)
                print(f"处理 token position = {position}")
                print(f"token id = {current_token_id}")
                print(f"token text = {current_token_text!r}")
                print(f"当前 position_ids = {current_position_ids[0].tolist()}")

                # 先跑 Hugging Face 的 cached decode 作为对照。
                # records.clear() 后，本轮 hook 只保存当前 token 的模块输入输出。
                records.clear()
                hf_outputs: Any = model(
                    input_ids=current_token_ids,
                    past_key_values=hf_past_key_values,
                    use_cache=True,
                )
                hf_past_key_values = hf_outputs.past_key_values
                hf_kv_pairs = extract_kv_pairs(hf_past_key_values)
                hf_layer0_k_cache = to_reference(hf_kv_pairs[0][0])
                hf_layer0_v_cache = to_reference(hf_kv_pairs[0][1])

                old_cache_len = 0 if manual_k_cache is None else manual_k_cache.shape[2]

                # -----------------------------------------------------------------
                # 第 1 段：当前 token 的 embedding lookup
                # -----------------------------------------------------------------
                x = embedding_table[current_token_ids.cpu()]
                print_token_stage(
                    "第 1 段：Embedding lookup",
                    x,
                    "x[b, 0, j] = E[token_id[b, 0], j]；每次只查当前一个 token。",
                )

                # -----------------------------------------------------------------
                # 第 2 段：Layer 0 input RMSNorm
                # -----------------------------------------------------------------
                input_norm = manual_rms_norm(x, input_norm_weight, rms_eps)
                print_token_stage(
                    "第 2 段：input RMSNorm",
                    input_norm,
                    "y[j] = x[j] * rsqrt(mean(x*x) + eps) * gamma[j]，shape 仍是 [1, 1, 1024]。",
                )

                # -----------------------------------------------------------------
                # 第 3 段：当前 token 的 q_proj / k_proj / v_proj
                # -----------------------------------------------------------------
                q_flat = manual_linear(input_norm, attn0.q_proj)
                k_flat = manual_linear(input_norm, attn0.k_proj)
                v_flat = manual_linear(input_norm, attn0.v_proj)
                print_token_stage(
                    "第 3 段：当前 token 的 q_proj / k_proj / v_proj",
                    q_flat,
                    "当前 token 只新算自己的 Q/K/V；历史 token 的 K/V 不重算，后面从 cache 读取。",
                )

                # -----------------------------------------------------------------
                # 第 4 段：reshape 到多头形状，并做 q_norm / k_norm
                # -----------------------------------------------------------------
                q_view = q_flat.view(batch_size, 1, num_q_heads, head_dim)
                k_view = k_flat.view(batch_size, 1, num_kv_heads, head_dim)
                v_view = v_flat.view(batch_size, 1, num_kv_heads, head_dim)

                q_norm = manual_rms_norm(q_view, q_norm_weight, rms_eps)
                k_norm = manual_rms_norm(k_view, k_norm_weight, rms_eps)
                print_token_stage(
                    "第 4 段：q_norm / k_norm",
                    q_norm,
                    "Q/K 每个 head 内部做 128 维 RMSNorm；V 不做这一步。",
                )

                q_states = q_norm.transpose(1, 2)
                k_states = k_norm.transpose(1, 2)
                v_states = v_view.transpose(1, 2)

                # -----------------------------------------------------------------
                # 第 5 段：当前 token 的 RoPE
                # -----------------------------------------------------------------
                cos, sin = manual_rope_cos_sin(backbone.rotary_emb, x, current_position_ids)
                q_rope, current_k_rope = manual_apply_rope(q_states, k_states, cos, sin)
                print_token_stage(
                    "第 5 段：RoPE",
                    q_rope,
                    "只对当前 token 的 Q/K 做 RoPE；K 做完 RoPE 后才写入 cache，V 不做 RoPE。",
                )

                # -----------------------------------------------------------------
                # 第 6 段：写入 Layer 0 的 K/V cache
                # -----------------------------------------------------------------
                manual_k_cache = append_cache(manual_k_cache, current_k_rope)
                manual_v_cache = append_cache(manual_v_cache, v_states)
                new_cache_len = manual_k_cache.shape[2]

                print()
                print("第 6 段：写入 KV Cache")
                print(f"  写入前 T_old = {old_cache_len}")
                print("  当前 K shape:", list(current_k_rope.shape))
                print("  当前 V shape:", list(v_states.shape))
                print("  写入后 K cache shape:", list(manual_k_cache.shape))
                print("  写入后 V cache shape:", list(manual_v_cache.shape))
                print("  数学/含义:")
                print("    new_cache[:, :, 0:T_old, :] 保留历史 token")
                print("    new_cache[:, :, T_old,   :] 写入当前 token")
                print("    T 每处理一个 token 就增长 1")

                # -----------------------------------------------------------------
                # 第 7 段：读取 cache，做 GQA repeat 和 cached attention
                # -----------------------------------------------------------------
                k_cache_repeated = manual_repeat_kv(manual_k_cache, kv_repeat)
                v_cache_repeated = manual_repeat_kv(manual_v_cache, kv_repeat)
                attn_output_heads, attn_weights, attn_scores = manual_cached_attention(
                    q_rope,
                    k_cache_repeated,
                    v_cache_repeated,
                    float(attn0.scaling),
                )
                attn_concat = attn_output_heads.transpose(1, 2).contiguous().reshape(batch_size, 1, -1)

                print()
                print("第 7 段：读取 KV Cache 并计算 cached attention")
                print("  q_current shape:", list(q_rope.shape))
                print("  repeated K cache shape:", list(k_cache_repeated.shape))
                print("  repeated V cache shape:", list(v_cache_repeated.shape))
                print("  attention score shape:", list(attn_scores.shape))
                print("  attention prob shape:", list(attn_weights.shape))
                print("  attention concat shape:", list(attn_concat.shape))
                print("  数学/含义:")
                print("    score[k_pos] = dot(q_current, k_cache[k_pos]) / sqrt(128)")
                print("    prob = softmax(score over k_pos = 0..T-1)")
                print("    attn_out[d] = sum_k_pos(prob[k_pos] * v_cache[k_pos, d])")
                print("    cache 里没有未来 token，所以这里不需要 causal mask")

                # -----------------------------------------------------------------
                # 第 8 段：o_proj + attention residual
                # -----------------------------------------------------------------
                o_proj = manual_linear(attn_concat, attn0.o_proj)
                after_attention = x + o_proj
                print_token_stage(
                    "第 8 段：o_proj + residual add",
                    after_attention,
                    "attention_residual = 当前 Layer 输入 x + o_proj(attention_output)。",
                )

                # -----------------------------------------------------------------
                # 第 9 段：post-attention RMSNorm
                # -----------------------------------------------------------------
                post_norm = manual_rms_norm(after_attention, post_norm_weight, rms_eps)
                print_token_stage(
                    "第 9 段：post-attention RMSNorm",
                    post_norm,
                    "进入 MLP 前再做一次 1024 维 RMSNorm。",
                )

                # -----------------------------------------------------------------
                # 第 10 段：MLP + final residual add
                # -----------------------------------------------------------------
                gate = manual_linear(post_norm, mlp0.gate_proj)
                up = manual_linear(post_norm, mlp0.up_proj)
                mlp_hidden = F.silu(gate) * up
                mlp_down = manual_linear(mlp_hidden, mlp0.down_proj)
                layer0_output = after_attention + mlp_down
                print_token_stage(
                    "第 10 段：MLP + final residual add",
                    layer0_output,
                    "Layer0 输出 = attention_residual + W_down(silu(W_gate*x) * W_up*x)。",
                )

                print()
                print("本 token 手写结果 vs Hugging Face cached decode 结果")
                checks = [
                    compare_tensor(
                        f"token {position}: q_norm output before transpose",
                        q_norm,
                        require_output(records, "layer0.self_attn.q_norm"),
                    ),
                    compare_tensor(
                        f"token {position}: current K written into cache",
                        current_k_rope,
                        hf_layer0_k_cache[:, :, -1:, :],
                    ),
                    compare_tensor(
                        f"token {position}: current V written into cache",
                        v_states,
                        hf_layer0_v_cache[:, :, -1:, :],
                    ),
                    compare_tensor(
                        f"token {position}: full Layer0 K cache after append",
                        manual_k_cache,
                        hf_layer0_k_cache,
                    ),
                    compare_tensor(
                        f"token {position}: full Layer0 V cache after append",
                        manual_v_cache,
                        hf_layer0_v_cache,
                    ),
                    compare_tensor(
                        f"token {position}: attention concat before o_proj",
                        attn_concat,
                        require_input0(records, "layer0.self_attn.o_proj"),
                    ),
                    compare_tensor(
                        f"token {position}: Layer 0 final output",
                        layer0_output,
                        require_output(records, "layer0"),
                    ),
                ]
                all_checks_ok = all_checks_ok and all(checks)

                if hf_layer0_k_cache.shape[2] != new_cache_len:
                    raise RuntimeError(
                        f"HF cache length {hf_layer0_k_cache.shape[2]} != manual cache length {new_cache_len}"
                    )

    finally:
        for hook in hooks:
            hook.remove()

    print()
    print("=" * 80)
    print("KV Cache 内存计算")
    values_per_token_per_layer = 2 * num_kv_heads * head_dim
    values_per_token_all_layers = num_layers * values_per_token_per_layer
    print(f"每层每 token 的 KV 数值个数 = 2 * KVH * D = 2 * {num_kv_heads} * {head_dim} = {values_per_token_per_layer}")
    print(
        "全模型每 token 的 KV 数值个数 = "
        f"layers * 2 * KVH * D = {num_layers} * {values_per_token_per_layer} = {values_per_token_all_layers}"
    )
    for context_len in [128, 256, 512, 1024]:
        fp16_bytes = context_len * values_per_token_all_layers * 2
        int8_bytes = context_len * values_per_token_all_layers * 1
        print(f"context_len={context_len}: FP16/BF16 KV ≈ {fp16_bytes / 1024 / 1024:.2f} MiB, INT8 KV ≈ {int8_bytes / 1024 / 1024:.2f} MiB")

    print()
    if not all_checks_ok:
        raise RuntimeError("有至少一个手写 Tensor 没有通过 torch.allclose 检查。")

    print("验证完成：手写 Layer 0 cached decode 和 Hugging Face past_key_values 对齐。")


if __name__ == "__main__":
    main()
