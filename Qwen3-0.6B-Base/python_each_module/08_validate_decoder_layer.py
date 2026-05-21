import torch

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    extract_kv_pairs,
    finish,
    load_model,
    manual_decoder_layer_cached,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, total_tokens = input_ids.shape
    layer0 = backbone.layers[0]
    embedding_table = tensor_to_float32(backbone.embed_tokens.weight)

    hidden_size = int(model.config.hidden_size)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    kv_repeat = num_q_heads // num_kv_heads

    records: dict[str, dict[str, torch.Tensor]] = {}
    hook = layer0.register_forward_hook(capture_module_io(records, "layer0"))
    manual_k_cache: torch.Tensor | None = None
    manual_v_cache: torch.Tensor | None = None
    hf_past_key_values = None
    all_ok = True

    print("Complete Layer0 cached decoder-layer validation")
    print("=" * 80)

    try:
        with torch.no_grad():
            for position in range(total_tokens):
                current_token_ids = input_ids[:, position : position + 1]
                records.clear()
                hf_outputs = model(
                    input_ids=current_token_ids,
                    past_key_values=hf_past_key_values,
                    use_cache=True,
                )
                hf_past_key_values = hf_outputs.past_key_values
                hf_layer0_k, hf_layer0_v = extract_kv_pairs(hf_past_key_values)[0]

                hidden = embedding_table[current_token_ids.cpu()]
                position_ids = torch.tensor([[position]], dtype=torch.long)
                layer_output, manual_k_cache, manual_v_cache = manual_decoder_layer_cached(
                    hidden=hidden,
                    layer=layer0,
                    rotary_emb=backbone.rotary_emb,
                    position_ids=position_ids,
                    k_cache=manual_k_cache,
                    v_cache=manual_v_cache,
                    batch_size=batch_size,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    kv_repeat=kv_repeat,
                    model_config=model.config,
                )

                print()
                print(f"Token position {position}")
                checks = [
                    compare_tensor("Layer0 output", layer_output, require_output(records, "layer0"), show_values=False),
                    compare_tensor("Layer0 K cache", manual_k_cache, tensor_to_float32(hf_layer0_k), show_values=False),
                    compare_tensor("Layer0 V cache", manual_v_cache, tensor_to_float32(hf_layer0_v), show_values=False),
                ]
                all_ok = all_ok and all(checks)
    finally:
        hook.remove()

    finish(all_ok, "complete Layer0 cached decoder layer")


if __name__ == "__main__":
    main()
