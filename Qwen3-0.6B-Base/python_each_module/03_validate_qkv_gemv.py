import torch

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    finish,
    load_model,
    manual_linear_from_weight,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    layer0 = backbone.layers[0]
    attn0 = layer0.self_attn

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.input_layernorm.register_forward_hook(capture_module_io(records, "input_norm")),
        attn0.q_proj.register_forward_hook(capture_module_io(records, "q_proj")),
        attn0.k_proj.register_forward_hook(capture_module_io(records, "k_proj")),
        attn0.v_proj.register_forward_hook(capture_module_io(records, "v_proj")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    input_norm = require_output(records, "input_norm")
    manual_q = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.q_proj.weight))
    manual_k = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.k_proj.weight))
    manual_v = manual_linear_from_weight(input_norm, tensor_to_float32(attn0.v_proj.weight))

    print("Q/K/V GEMV validation")
    print("=" * 80)
    print("Formula: y[o] = sum_i(x[i] * W[o, i]) + bias[o]")

    checks = [
        compare_tensor("q_proj", manual_q, require_output(records, "q_proj")),
        compare_tensor("k_proj", manual_k, require_output(records, "k_proj")),
        compare_tensor("v_proj", manual_v, require_output(records, "v_proj")),
    ]
    finish(all(checks), "Q/K/V GEMV")


if __name__ == "__main__":
    main()
