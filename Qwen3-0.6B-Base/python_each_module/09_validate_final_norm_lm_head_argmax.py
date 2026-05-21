import torch

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    finish,
    get_rms_norm_eps,
    load_model,
    manual_linear_from_weight,
    manual_rms_norm,
    require_input0,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        backbone.norm.register_forward_hook(capture_module_io(records, "final_norm")),
        model.lm_head.register_forward_hook(capture_module_io(records, "lm_head")),
    ]
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    final_eps = get_rms_norm_eps(backbone.norm, model.config)
    manual_final = manual_rms_norm(
        require_input0(records, "final_norm"),
        tensor_to_float32(backbone.norm.weight),
        final_eps,
    )
    manual_logits = manual_linear_from_weight(manual_final, tensor_to_float32(model.lm_head.weight))

    manual_next = torch.argmax(manual_logits[:, -1, :], dim=-1, keepdim=True)
    hf_next = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    print("Final RMSNorm, LM head, and argmax validation")
    print("=" * 80)
    print("LM head formula: logits[vocab_id] = dot(final_hidden, embedding_table[vocab_id])")

    checks = [
        compare_tensor("final RMSNorm", manual_final, require_output(records, "final_norm"), show_values=False),
        compare_tensor("LM head logits", manual_logits, require_output(records, "lm_head"), show_values=True),
    ]

    manual_id = int(manual_next.item())
    hf_id = int(hf_next.item())
    print("-" * 80)
    print(f"Manual argmax token: {manual_id} / {tokenizer.decode([manual_id])!r}")
    print(f"HF argmax token:     {hf_id} / {tokenizer.decode([hf_id])!r}")
    checks.append(manual_id == hf_id)
    finish(all(checks), "final norm, LM head, and argmax")


if __name__ == "__main__":
    main()
