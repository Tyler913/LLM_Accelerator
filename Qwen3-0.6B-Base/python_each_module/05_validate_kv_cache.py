import torch

from common import (
    append_cache,
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
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, total_tokens = input_ids.shape
    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn

    hidden_size = int(model.config.hidden_size)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    rms_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)
    embedding_table = tensor_to_float32(backbone.embed_tokens.weight)

    manual_k_cache: torch.Tensor | None = None
    manual_v_cache: torch.Tensor | None = None
    hf_past_key_values = None
    all_ok = True

    print("Layer0 KV cache validation")
    print("=" * 80)
    print("K cache stores post-RoPE K. V cache stores reshaped v_proj output.")

    with torch.no_grad():
        for position in range(total_tokens):
            current_token_ids = input_ids[:, position : position + 1]
            hf_outputs = model(
                input_ids=current_token_ids,
                past_key_values=hf_past_key_values,
                use_cache=True,
            )
            hf_past_key_values = hf_outputs.past_key_values
            hf_layer0_k, hf_layer0_v = extract_kv_pairs(hf_past_key_values)[0]

            x = embedding_table[current_token_ids.cpu()]
            input_norm = manual_rms_norm(x, tensor_to_float32(layer0.input_layernorm.weight), rms_eps)

            q_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.q_proj.weight))
            k_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.k_proj.weight))
            v_flat = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.v_proj.weight))

            q_view = q_flat.view(batch_size, 1, num_q_heads, head_dim)
            k_view = k_flat.view(batch_size, 1, num_kv_heads, head_dim)
            v_view = v_flat.view(batch_size, 1, num_kv_heads, head_dim)

            q_norm = manual_rms_norm(q_view, tensor_to_float32(attn0.q_norm.weight), rms_eps)
            k_norm = manual_rms_norm(k_view, tensor_to_float32(attn0.k_norm.weight), rms_eps)
            q_states = q_norm.transpose(1, 2)
            k_states = k_norm.transpose(1, 2)
            v_states = v_view.transpose(1, 2)

            position_ids = torch.tensor([[position]], dtype=torch.long)
            cos, sin = manual_rope_cos_sin(backbone.rotary_emb, x, position_ids)
            _q_rope, current_k_rope = manual_apply_rope(q_states, k_states, cos, sin)

            manual_k_cache = append_cache(manual_k_cache, current_k_rope)
            manual_v_cache = append_cache(manual_v_cache, v_states)

            print()
            print(f"After token position {position}, cache length T = {position + 1}")
            checks = [
                compare_tensor("full K cache", manual_k_cache, tensor_to_float32(hf_layer0_k), show_values=False),
                compare_tensor("full V cache", manual_v_cache, tensor_to_float32(hf_layer0_v), show_values=False),
            ]
            all_ok = all_ok and all(checks)

    finish(all_ok, "Layer0 KV cache")


if __name__ == "__main__":
    main()
