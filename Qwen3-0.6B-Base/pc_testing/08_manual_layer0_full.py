from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1]

PROMPT = "The future of FPGA is"
SAMPLE_COUNT = 8


EXPLANATION = """
完整手写 Layer 0 参考实现
========================

这个脚本的目标：

  1. 用 Hugging Face 加载真实 Qwen3-0.6B-Base 权重。
  2. 手写完整的第 0 层 Decoder Layer 计算。
  3. 把手写结果和 Hugging Face 第 0 层的中间结果逐段对比。

这里说的 "完整 Layer 0" 包括：

  token ids
    -> embedding lookup
    -> input RMSNorm
    -> q_proj / k_proj / v_proj
    -> q_norm / k_norm
    -> RoPE
    -> causal self-attention
    -> o_proj
    -> residual add
    -> post-attention RMSNorm
    -> MLP gate_proj / up_proj / silu / down_proj
    -> residual add
    -> Layer 0 output

为了让 RoPE 和 attention 真的被验证，本脚本使用整个 prompt：

  "The future of FPGA is"

这个 prompt 当前会被分成 5 个 token，所以 seq_len = 5。
这样 position id 是 [0, 1, 2, 3, 4]，RoPE 不再只是 position 0 的恒等变换，
attention 也会真正执行 causal mask 和 softmax。

重要形状：

  B = batch size = 1
  S = sequence length = 5
  H = hidden size = 1024
  QH = query heads = 16
  KVH = key/value heads = 8
  D = head dim = 128
  I = MLP intermediate size = 3072

  input_ids:               [B, S]
  embedding output x:      [B, S, H]
  input RMSNorm output:    [B, S, H]
  q_proj flat output:      [B, S, QH * D]  = [1, 5, 2048]
  k_proj flat output:      [B, S, KVH * D] = [1, 5, 1024]
  v_proj flat output:      [B, S, KVH * D] = [1, 5, 1024]
  q after reshape/norm:    [B, S, QH, D]   = [1, 5, 16, 128]
  k after reshape/norm:    [B, S, KVH, D]  = [1, 5, 8, 128]
  q for attention:         [B, QH, S, D]   = [1, 16, 5, 128]
  k for attention:         [B, KVH, S, D]  = [1, 8, 5, 128]
  v for attention:         [B, KVH, S, D]  = [1, 8, 5, 128]
  repeated k/v:            [B, QH, S, D]   = [1, 16, 5, 128]
  attention scores:        [B, QH, S, S]   = [1, 16, 5, 5]
  attention output heads:  [B, QH, S, D]   = [1, 16, 5, 128]
  attention concat:        [B, S, QH * D]  = [1, 5, 2048]
  o_proj output:           [B, S, H]       = [1, 5, 1024]
  MLP gate/up output:      [B, S, I]       = [1, 5, 3072]
  MLP down output:         [B, S, H]       = [1, 5, 1024]
  Layer 0 output:          [B, S, H]       = [1, 5, 1024]

核心数学：

1. Embedding lookup

   x[b, s, j] = E[token_id[b, s], j]

   这一步是查表，不是乘法。

2. RMSNorm

   对最后一维做归一化：

   mean_square = sum_j(x[j] * x[j]) / N
   inv_rms = 1 / sqrt(mean_square + eps)
   y[j] = x[j] * inv_rms * gamma[j]

   input_layernorm 的 N = 1024。
   q_norm/k_norm 的 N = 128，因为它们只对每个 head 内部做 RMSNorm。

3. Linear / GEMV

   PyTorch Linear 的权重存储为 [out_features, in_features]。

   y[o] = bias[o] + sum_i(x[i] * W[o, i])

   Qwen3 这里的 q/k/v/o/gate/up/down projection 都没有 bias。

4. RoPE

   RoPE 只作用在 Q 和 K 上，不作用在 V 上。

   假设某一个 head 里的向量 x 长度是 D = 128。
   先把 x 拆成前后两半，每一半长度是 64：

     x = [x0, x1, ..., x63, x64, x65, ..., x127]

     x_first  = [x0,  x1,  ..., x63]
     x_second = [x64, x65, ..., x127]

   concat(a, b) 的意思是“拼接两个向量”，不是加法，也不是乘法。
   如果：

     a = [a0, a1, a2]
     b = [b0, b1, b2]

   那么：

     concat(a, b) = [a0, a1, a2, b0, b1, b2]

   所以：

     rotate_half(x) = concat(-x_second, x_first)

   展开后就是：

     rotate_half(x)
       = [-x64, -x65, ..., -x127, x0, x1, ..., x63]

   写成下标公式，令 half = 64：

     rotate_half(x)[i]        = -x[i + half],  i = 0..63
     rotate_half(x)[i + half] =  x[i],         i = 0..63

   RoPE 的 cos/sin 来自 token 的 position。
   对每个 position p 和每个半维度下标 i = 0..63：

     inv_freq[i] = 1 / rope_theta^(2i / D)
     angle[p, i] = p * inv_freq[i]
     c[p, i] = cos(angle[p, i])
     s[p, i] = sin(angle[p, i])

   代码里 cos/sin 会被复制成 128 维：

     cos_full[p] = [c[p,0], ..., c[p,63], c[p,0], ..., c[p,63]]
     sin_full[p] = [s[p,0], ..., s[p,63], s[p,0], ..., s[p,63]]

   最后：

     q_rope = q * cos_full + rotate_half(q) * sin_full
     k_rope = k * cos_full + rotate_half(k) * sin_full

   对某一对维度 (i, i+64)，它等价于二维旋转：

     q_rope[i]      = q[i]      * c[i] - q[i+64] * s[i]
     q_rope[i + 64] = q[i + 64] * c[i] + q[i]    * s[i]

   K 的公式完全一样，只是把 q 换成 k。

5. GQA 的 KV repeat

   Q 有 16 个 head，K/V 只有 8 个 head。
   每个 K/V head 会服务 2 个 Q head。

   kv_head_id = q_head_id // 2

6. Causal self-attention

   score[b, h, q_pos, k_pos] =
       dot(q[b, h, q_pos, :], k[b, h, k_pos, :]) / sqrt(D)

   如果 k_pos > q_pos，说明未来 token 不允许被当前 token 看到：

       score = -inf

   prob = softmax(score, dim=-1)

   attn_out[b, h, q_pos, d] =
       sum_k_pos(prob[b, h, q_pos, k_pos] * v[b, h, k_pos, d])

7. MLP

   gate = x @ W_gate.T
   up = x @ W_up.T
   hidden = silu(gate) * up
   down = hidden @ W_down.T

8. Residual add

   attention_residual = layer_input + attention_output
   layer_output = attention_residual + mlp_output
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
        else:
            raise TypeError(f"{name} hook expected tensor output, got {type(output)}")

        records[name] = item

    return hook


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

    这不是一个神秘函数，它只是在最后一维上做“拆半 + 换位置 + 变号”。

    假设最后一维 D = 128，half = 64。
    对一个 head 向量:

      x = [x0, x1, ..., x63, x64, x65, ..., x127]

    先拆成两半:

      x_first  = [x0,  x1,  ..., x63]
      x_second = [x64, x65, ..., x127]

    concat(a, b) 表示把两个向量首尾拼起来:

      concat([a0, a1], [b0, b1]) = [a0, a1, b0, b1]

    所以:

      rotate_half(x) = concat(-x_second, x_first)

    展开后:

      rotate_half(x)
        = [-x64, -x65, ..., -x127, x0, x1, ..., x63]

    下标公式:

      rotate_half(x)[i]        = -x[i + half],  i = 0..63
      rotate_half(x)[i + half] =  x[i],         i = 0..63

    在代码里 torch.cat((-x_second, x_first), dim=-1) 就是这个 concat。
    """
    half = x.shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    return torch.cat((-x_second, x_first), dim=-1)


def manual_rope_cos_sin(rotary_emb: Any, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    手写 Qwen3 RoPE 的 cos/sin 生成。

    输入:
      x: [B, S, H]，这里只用它的 dtype/device 概念
      position_ids: [B, S]

    输出:
      cos: [B, S, D]
      sin: [B, S, D]

    数学，令 D = 128，half = 64:

      inv_freq[i] = 1 / rope_theta^(2i / D),  i = 0..63
      angle[b, s, i] = position_ids[b, s] * inv_freq[i]

    这一步得到的是 64 个角度。
    之后把这 64 个角度复制一遍，变成 128 维:

      emb[b, s] =
        [angle0, angle1, ..., angle63, angle0, angle1, ..., angle63]

    所以:

      cos[b, s, d] = cos(emb[b, s, d])
      sin[b, s, d] = sin(emb[b, s, d])

    这里的 concat(freqs, freqs) 也是向量拼接:

      concat([f0, f1], [f0, f1]) = [f0, f1, f0, f1]
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
      q: [B, QH, S, D]
      k: [B, KVH, S, D]
      cos/sin: [B, S, D]

    输出:
      q_rope: [B, QH, S, D]
      k_rope: [B, KVH, S, D]

    数学:
      q_rope = q * cos + rotate_half(q) * sin
      k_rope = k * cos + rotate_half(k) * sin

    展开到一对维度上看，令 half = 64。
    对每个 token position、每个 head、每个 i = 0..63:

      c = cos[position, i]
      s = sin[position, i]

      q_rope[i]        = q[i]        * c - q[i + half] * s
      q_rope[i + half] = q[i + half] * c + q[i]        * s

      k_rope[i]        = k[i]        * c - k[i + half] * s
      k_rope[i + half] = k[i + half] * c + k[i]        * s

    也就是说，每一对 (i, i+64) 都是在做一个二维旋转:

      [new_first ]   [ cos  -sin ] [old_first ]
      [new_second] = [ sin   cos ] [old_second]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rope = (q * cos) + (manual_rotate_half(q) * sin)
    k_rope = (k * cos) + (manual_rotate_half(k) * sin)
    return q_rope, k_rope


def manual_repeat_kv(x: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """
    GQA 的 K/V head 复制。

    输入:
      x: [B, KVH, S, D]

    输出:
      repeated: [B, QH, S, D]

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


def manual_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    手写 causal self-attention。

    输入:
      q: [B, QH, S, D]
      k: [B, QH, S, D]
      v: [B, QH, S, D]

    输出:
      attn_output: [B, QH, S, D]
      attn_weights: [B, QH, S, S]

    数学:
      score = q @ k.T / sqrt(D)
      score[q_pos, k_pos > q_pos] = -inf
      prob = softmax(score)
      attn_output = prob @ v
    """
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    seq_len = scores.shape[-1]
    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(future_mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(weights, v)
    return output, weights


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
) -> None:
    """打印手写 Tensor 和 Hugging Face Tensor 的误差。"""
    manual = manual.to(torch.float32)
    reference = reference.to(torch.float32)
    diff = manual - reference

    print("-" * 80)
    print(name)
    print(f"  手写 shape: {list(manual.shape)}")
    print(f"  HF shape:   {list(reference.shape)}")
    print(f"  最大绝对误差 max abs error:  {max_abs_error(manual, reference):.10e}")
    print(f"  平均绝对误差 mean abs error: {mean_abs_error(manual, reference):.10e}")
    print(f"  torch.allclose(atol={atol}, rtol={rtol}): {torch.allclose(manual, reference, atol=atol, rtol=rtol)}")
    print(f"  手写前 {SAMPLE_COUNT} 个值: {format_values(manual)}")
    print(f"  HF 前 {SAMPLE_COUNT} 个值:   {format_values(reference)}")
    print(f"  差值前 {SAMPLE_COUNT} 个值: {format_values(diff)}")


def require_output(records: dict[str, dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if name not in records or "output" not in records[name]:
        raise RuntimeError(f"Missing output record for {name}")
    return records[name]["output"]


def require_input0(records: dict[str, dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if name not in records or "input0" not in records[name]:
        raise RuntimeError(f"Missing input0 record for {name}")
    return records[name]["input0"]


def print_step(title: str, shape: torch.Tensor, math: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print(f"输出 shape: {list(shape.shape)}")
    print(math)


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
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    hidden_size = int(model.config.hidden_size)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    intermediate_size = int(model.config.intermediate_size)
    kv_repeat = num_q_heads // num_kv_heads

    print("输入 prompt")
    print("=" * 80)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token ids: {input_ids[0].tolist()}")
    print(f"Token text: {[tokenizer.decode([int(token_id)]) for token_id in input_ids[0].tolist()]}")
    print(f"position_ids: {position_ids[0].tolist()}")
    print()

    print("模型维度")
    print("=" * 80)
    print(f"B batch size: {batch_size}")
    print(f"S sequence length: {seq_len}")
    print(f"H hidden size: {hidden_size}")
    print(f"QH query heads: {num_q_heads}")
    print(f"KVH key/value heads: {num_kv_heads}")
    print(f"D head dim: {head_dim}")
    print(f"I MLP intermediate size: {intermediate_size}")
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

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                use_cache=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)

    # -------------------------------------------------------------------------
    # 第 1 段：Embedding lookup
    # -------------------------------------------------------------------------
    embedding_table = to_reference(backbone.embed_tokens.weight)
    x = embedding_table[input_ids.cpu()]
    print_step(
        "第 1 段：Embedding lookup",
        x,
        "数学: x[b, s, j] = E[token_id[b, s], j]，这是查表，不是乘法。",
    )

    # -------------------------------------------------------------------------
    # 第 2 段：Layer 0 input RMSNorm
    # -------------------------------------------------------------------------
    input_norm = manual_rms_norm(x, to_reference(layer0.input_layernorm.weight), rms_eps)
    print_step(
        "第 2 段：input RMSNorm",
        input_norm,
        "数学: y[j] = x[j] * rsqrt(mean(x*x) + eps) * gamma[j]，这里 N=1024。",
    )

    # -------------------------------------------------------------------------
    # 第 3 段：q_proj / k_proj / v_proj
    # -------------------------------------------------------------------------
    q_flat = manual_linear(input_norm, attn0.q_proj)
    k_flat = manual_linear(input_norm, attn0.k_proj)
    v_flat = manual_linear(input_norm, attn0.v_proj)
    print_step(
        "第 3 段：q_proj / k_proj / v_proj",
        q_flat,
        "数学: q[o] = sum_i(input_norm[i] * Wq[o, i])；k/v 同理。",
    )

    # -------------------------------------------------------------------------
    # 第 4 段：reshape 到多头形状，并做 q_norm / k_norm
    # -------------------------------------------------------------------------
    q_view = q_flat.view(batch_size, seq_len, num_q_heads, head_dim)
    k_view = k_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    v_view = v_flat.view(batch_size, seq_len, num_kv_heads, head_dim)

    q_norm = manual_rms_norm(q_view, to_reference(attn0.q_norm.weight), rms_eps)
    k_norm = manual_rms_norm(k_view, to_reference(attn0.k_norm.weight), rms_eps)
    print_step(
        "第 4 段：q_norm / k_norm",
        q_norm,
        "数学: 对每个 head 的 128 维向量做 RMSNorm；V 不做 q_norm/k_norm。",
    )

    # Hugging Face attention 内部使用 [B, heads, S, D]。
    q_states = q_norm.transpose(1, 2)
    k_states = k_norm.transpose(1, 2)
    v_states = v_view.transpose(1, 2)

    # -------------------------------------------------------------------------
    # 第 5 段：RoPE 旋转位置编码
    # -------------------------------------------------------------------------
    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, x, position_ids.cpu())
    q_rope, k_rope = manual_apply_rope(q_states, k_states, cos, sin)
    print_step(
        "第 5 段：RoPE",
        q_rope,
        "数学: q_rope = q*cos + rotate_half(q)*sin；k_rope 同理；V 不做 RoPE。",
    )

    # -------------------------------------------------------------------------
    # 第 6 段：GQA repeat + causal self-attention
    # -------------------------------------------------------------------------
    k_repeated = manual_repeat_kv(k_rope, kv_repeat)
    v_repeated = manual_repeat_kv(v_states, kv_repeat)
    attn_output_heads, attn_weights = manual_causal_attention(
        q_rope,
        k_repeated,
        v_repeated,
        float(attn0.scaling),
    )
    attn_concat = attn_output_heads.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
    print_step(
        "第 6 段：causal self-attention",
        attn_concat,
        "数学: scores = q @ k.T / sqrt(128)，加 causal mask，softmax 后再乘 V。",
    )
    print(f"attention weights shape: {list(attn_weights.shape)}")

    # -------------------------------------------------------------------------
    # 第 7 段：o_proj + attention residual
    # -------------------------------------------------------------------------
    o_proj = manual_linear(attn_concat, attn0.o_proj)
    after_attention = x + o_proj
    print_step(
        "第 7 段：o_proj + residual add",
        after_attention,
        "数学: attention_residual = layer_input + o_proj(attention_output)。",
    )

    # -------------------------------------------------------------------------
    # 第 8 段：post-attention RMSNorm
    # -------------------------------------------------------------------------
    post_norm = manual_rms_norm(after_attention, to_reference(layer0.post_attention_layernorm.weight), rms_eps)
    print_step(
        "第 8 段：post-attention RMSNorm",
        post_norm,
        "数学: y[j] = x[j] * rsqrt(mean(x*x) + eps) * gamma[j]，这里 N=1024。",
    )

    # -------------------------------------------------------------------------
    # 第 9 段：MLP
    # -------------------------------------------------------------------------
    gate = manual_linear(post_norm, mlp0.gate_proj)
    up = manual_linear(post_norm, mlp0.up_proj)
    mlp_hidden = F.silu(gate) * up
    mlp_down = manual_linear(mlp_hidden, mlp0.down_proj)
    layer0_output = after_attention + mlp_down
    print_step(
        "第 9 段：MLP + final residual add",
        layer0_output,
        "数学: down = W_down * (silu(W_gate*x) * (W_up*x))；Layer0 输出 = residual + down。",
    )

    print()
    print("=" * 80)
    print("手写结果 vs Hugging Face hook 结果")

    compare_tensor("embedding lookup", x, require_output(records, "embed_tokens"))
    compare_tensor("input RMSNorm", input_norm, require_output(records, "layer0.input_layernorm"))
    compare_tensor("q_proj flat", q_flat, require_output(records, "layer0.self_attn.q_proj"))
    compare_tensor("k_proj flat", k_flat, require_output(records, "layer0.self_attn.k_proj"))
    compare_tensor("v_proj flat", v_flat, require_output(records, "layer0.self_attn.v_proj"))
    compare_tensor("q_norm output before transpose", q_norm, require_output(records, "layer0.self_attn.q_norm"))
    compare_tensor("k_norm output before transpose", k_norm, require_output(records, "layer0.self_attn.k_norm"))
    compare_tensor("attention concat before o_proj", attn_concat, require_input0(records, "layer0.self_attn.o_proj"))
    compare_tensor("o_proj output", o_proj, require_output(records, "layer0.self_attn.o_proj"))
    compare_tensor(
        "attention residual before post-attention RMSNorm",
        after_attention,
        require_input0(records, "layer0.post_attention_layernorm"),
    )
    compare_tensor("post-attention RMSNorm", post_norm, require_output(records, "layer0.post_attention_layernorm"))
    compare_tensor("MLP gate_proj", gate, require_output(records, "layer0.mlp.gate_proj"))
    compare_tensor("MLP up_proj", up, require_output(records, "layer0.mlp.up_proj"))
    compare_tensor("MLP hidden before down_proj = silu(gate) * up", mlp_hidden, require_input0(records, "layer0.mlp.down_proj"))
    compare_tensor("MLP down_proj", mlp_down, require_output(records, "layer0.mlp.down_proj"))
    compare_tensor("Layer 0 final output", layer0_output, require_output(records, "layer0"))

    print()
    print("验证完成。")


if __name__ == "__main__":
    main()
