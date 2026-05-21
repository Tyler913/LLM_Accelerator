import torch

from common import (
    PROMPT,
    capture_module_io,
    compare_tensor,
    encode_prompt,
    finish,
    load_model,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    records: dict[str, dict[str, torch.Tensor]] = {}

    hook = backbone.embed_tokens.register_forward_hook(capture_module_io(records, "embed_tokens"))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        hook.remove()

    embedding_table = tensor_to_float32(backbone.embed_tokens.weight)
    manual_embedding = embedding_table[input_ids.cpu()]

    print("Embedding validation")
    print("=" * 80)
    print(f"Prompt: {PROMPT!r}")
    print(f"Token ids: {input_ids[0].tolist()}")
    print(f"Token text: {[tokenizer.decode([int(token_id)]) for token_id in input_ids[0].tolist()]}")
    print("Formula: output[b, s, h] = embedding_table[token_id[b, s], h]")

    ok = compare_tensor("embedding lookup", manual_embedding, require_output(records, "embed_tokens"))
    finish(ok, "embedding")


if __name__ == "__main__":
    main()
