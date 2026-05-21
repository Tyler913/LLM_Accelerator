import torch

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    extract_kv_pairs,
    finish,
    get_rms_norm_eps,
    load_model,
    manual_apply_rope,
    manual_linear_from_weight,
    manual_rms_norm,
    manual_rope_cos_sin,
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
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.input_layernorm.register_forward_hook(capture_module_io(records, "input_norm")),
        attn0.q_norm.register_forward_hook(capture_module_io(records, "q_norm")),
        attn0.k_norm.register_forward_hook(capture_module_io(records, "k_norm")),
    ]
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()

    input_norm = require_output(records, "input_norm")
    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)

    q_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.q_proj.weight))
    k_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.k_proj.weight))
    q_view = q_flat.view(batch_size, seq_len, num_q_heads, head_dim)
    k_view = k_flat.view(batch_size, seq_len, num_kv_heads, head_dim)

    manual_q_norm = manual_rms_norm(q_view, tensor_to_float32(attn0.q_norm.weight), rms_eps)
    manual_k_norm = manual_rms_norm(k_view, tensor_to_float32(attn0.k_norm.weight), rms_eps)

    q_states = manual_q_norm.transpose(1, 2)
    k_states = manual_k_norm.transpose(1, 2)
    cos, sin = manual_rope_cos_sin(backbone.rotary_emb, input_norm, position_ids)
    manual_q_rope, manual_k_rope = manual_apply_rope(q_states, k_states, cos, sin)

    hf_layer0_k_cache = tensor_to_float32(extract_kv_pairs(outputs.past_key_values)[0][0])

    print("Q/K norm and RoPE validation")
    print("=" * 80)
    print("Q/K norm formula: y = x * rsqrt(mean(x*x) + eps) * gamma")
    print("RoPE formula: rope(x) = x*cos(position) + rotate_half(x)*sin(position)")
    print(f"Manual q_rope shape, used by attention but not directly exposed by HF: {list(manual_q_rope.shape)}")

    checks = [
        compare_tensor("q_norm", manual_q_norm, require_output(records, "q_norm")),
        compare_tensor("k_norm", manual_k_norm, require_output(records, "k_norm")),
        compare_tensor("post-RoPE K equals HF layer0 K cache", manual_k_rope, hf_layer0_k_cache),
    ]
    finish(all(checks), "Q/K norm and RoPE")


if __name__ == "__main__":
    main()
