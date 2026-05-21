import torch

from common import (
    compare_tensor,
    encode_prompt,
    extract_kv_pairs,
    finish,
    get_rms_norm_eps,
    load_model,
    manual_decoder_layer_cached,
    manual_linear_from_weight,
    manual_rms_norm,
    tensor_to_float32,
)


def run_manual_one_token(
    current_token_ids: torch.Tensor,
    position: int,
    embedding_table: torch.Tensor,
    backbone,
    model,
    manual_k_caches: list[torch.Tensor | None],
    manual_v_caches: list[torch.Tensor | None],
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = embedding_table[current_token_ids.cpu()]
    position_ids = torch.tensor([[position]], dtype=torch.long)

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

    final_eps = get_rms_norm_eps(backbone.norm, model.config)
    final_hidden = manual_rms_norm(hidden, tensor_to_float32(backbone.norm.weight), final_eps)
    logits = manual_linear_from_weight(final_hidden, tensor_to_float32(model.lm_head.weight))
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return logits, next_token


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    batch_size, prompt_len = input_ids.shape

    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))
    kv_repeat = num_q_heads // num_kv_heads
    embedding_table = tensor_to_float32(backbone.embed_tokens.weight)

    manual_k_caches: list[torch.Tensor | None] = [None for _ in range(num_layers)]
    manual_v_caches: list[torch.Tensor | None] = [None for _ in range(num_layers)]
    hf_past_key_values = None
    all_ok = True
    last_manual_next: torch.Tensor | None = None

    print("Full run_one_token validation")
    print("=" * 80)

    with torch.no_grad():
        for position in range(prompt_len):
            current_token_ids = input_ids[:, position : position + 1]
            hf_outputs = model(
                input_ids=current_token_ids,
                past_key_values=hf_past_key_values,
                use_cache=True,
            )
            hf_past_key_values = hf_outputs.past_key_values

            manual_logits, manual_next = run_manual_one_token(
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

            hf_logits = tensor_to_float32(hf_outputs.logits)
            hf_next = torch.argmax(hf_logits[:, -1, :], dim=-1, keepdim=True)
            manual_id = int(manual_next.item())
            hf_id = int(hf_next.item())

            print()
            print(f"Token position {position}")
            checks = [
                compare_tensor("logits", manual_logits, hf_logits, show_values=position == prompt_len - 1),
                manual_id == hf_id,
            ]
            print(f"Manual next token: {manual_id} / {tokenizer.decode([manual_id])!r}")
            print(f"HF next token:     {hf_id} / {tokenizer.decode([hf_id])!r}")

            hf_kv_pairs = extract_kv_pairs(hf_past_key_values)
            for layer_id in [0, 1, 27]:
                manual_k = manual_k_caches[layer_id]
                manual_v = manual_v_caches[layer_id]
                if manual_k is None or manual_v is None:
                    raise RuntimeError(f"Missing manual cache for layer {layer_id}")
                checks.append(torch.allclose(manual_k, tensor_to_float32(hf_kv_pairs[layer_id][0]), atol=1e-5, rtol=1e-5))
                checks.append(torch.allclose(manual_v, tensor_to_float32(hf_kv_pairs[layer_id][1]), atol=1e-5, rtol=1e-5))
                print(f"Layer {layer_id:02d} cache shape K={list(manual_k.shape)} V={list(manual_v.shape)}")

            all_ok = all_ok and all(checks)
            last_manual_next = manual_next

        if last_manual_next is None:
            raise RuntimeError("No token was processed.")

        generated_position = prompt_len
        generated_token_ids = last_manual_next
        hf_outputs = model(
            input_ids=generated_token_ids,
            past_key_values=hf_past_key_values,
            use_cache=True,
        )
        manual_logits, manual_next = run_manual_one_token(
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
        hf_logits = tensor_to_float32(hf_outputs.logits)
        hf_next = torch.argmax(hf_logits[:, -1, :], dim=-1, keepdim=True)

        manual_id = int(manual_next.item())
        hf_id = int(hf_next.item())
        generated_id = int(generated_token_ids.item())
        print()
        print(f"Generated-token feedback position {generated_position}")
        print(f"Fed-back token: {generated_id} / {tokenizer.decode([generated_id])!r}")
        checks = [
            compare_tensor("feedback logits", manual_logits, hf_logits, show_values=True),
            manual_id == hf_id,
        ]
        print(f"Manual next token after feedback: {manual_id} / {tokenizer.decode([manual_id])!r}")
        print(f"HF next token after feedback:     {hf_id} / {tokenizer.decode([hf_id])!r}")
        all_ok = all_ok and all(checks)

    finish(all_ok, "full run_one_token")


if __name__ == "__main__":
    main()
