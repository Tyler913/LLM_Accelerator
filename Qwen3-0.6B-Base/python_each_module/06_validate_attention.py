import torch

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    finish,
    get_rms_norm_eps,
    load_model,
    manual_apply_rope,
    manual_causal_attention,
    manual_linear_from_weight,
    manual_repeat_kv,
    manual_rms_norm,
    manual_rope_cos_sin,
    require_input0,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, seq_len = input_ids.shape
    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn

    hidden_size = int(model.config.hidden_size)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    kv_repeat = num_q_heads // num_kv_heads
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

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

    input_norm = require_output(records, "input_norm")
    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)

    q_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.q_proj.weight))
    k_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.k_proj.weight))
    v_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.v_proj.weight))

    q_view = q_flat.view(batch_size, seq_len, num_q_heads, head_dim)
    k_view = k_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    v_view = v_flat.view(batch_size, seq_len, num_kv_heads, head_dim)

    q_norm = manual_rms_norm(q_view, tensor_to_float32(attn0.q_norm.weight), rms_eps)
    k_norm = manual_rms_norm(k_view, tensor_to_float32(attn0.k_norm.weight), rms_eps)

    q_states = q_norm.transpose(1, 2)
    k_states = k_norm.transpose(1, 2)
    v_states = v_view.transpose(1, 2)

    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, input_norm, position_ids)
    q_rope, k_rope = manual_apply_rope(q_states, k_states, cos, sin)
    k_repeated = manual_repeat_kv(k_rope, kv_repeat)
    v_repeated = manual_repeat_kv(v_states, kv_repeat)

    attn_heads, attn_weights = manual_causal_attention(q_rope, k_repeated, v_repeated, float(attn0.scaling))
    attn_concat = attn_heads.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)

    print("Causal attention validation")
    print("=" * 80)
    print("Formula: score = q @ k.T / sqrt(D), causal mask, softmax, output = prob @ v")
    print(f"Attention weights shape: {list(attn_weights.shape)}")

    ok = compare_tensor("attention concat before o_proj", attn_concat, require_input0(records, "o_proj"))
    finish(ok, "causal attention")


if __name__ == "__main__":
    main()
