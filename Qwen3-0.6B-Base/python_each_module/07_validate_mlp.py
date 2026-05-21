import torch
import torch.nn.functional as F

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    finish,
    load_model,
    manual_linear_from_weight,
    require_input0,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    layer0 = backbone.layers[0]
    mlp0 = layer0.mlp

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.register_forward_hook(capture_module_io(records, "layer0")),
        layer0.post_attention_layernorm.register_forward_hook(capture_module_io(records, "post_norm")),
        mlp0.gate_proj.register_forward_hook(capture_module_io(records, "gate_proj")),
        mlp0.up_proj.register_forward_hook(capture_module_io(records, "up_proj")),
        mlp0.down_proj.register_forward_hook(capture_module_io(records, "down_proj")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    post_norm = require_output(records, "post_norm")
    after_attention = require_input0(records, "post_norm")

    gate = manual_linear_from_weight(post_norm, tensor_to_float32(mlp0.gate_proj.weight))
    up = manual_linear_from_weight(post_norm, tensor_to_float32(mlp0.up_proj.weight))
    mlp_hidden = F.silu(gate) * up
    down = manual_linear_from_weight(mlp_hidden, tensor_to_float32(mlp0.down_proj.weight))
    layer_output = after_attention + down

    print("MLP validation")
    print("=" * 80)
    print("Formula: down = W_down * (silu(W_gate*x) * W_up*x), output = residual + down")

    checks = [
        compare_tensor("gate_proj", gate, require_output(records, "gate_proj")),
        compare_tensor("up_proj", up, require_output(records, "up_proj")),
        compare_tensor("MLP hidden before down_proj", mlp_hidden, require_input0(records, "down_proj")),
        compare_tensor("down_proj", down, require_output(records, "down_proj")),
        compare_tensor("Layer0 output after MLP residual", layer_output, require_output(records, "layer0")),
    ]
    finish(all(checks), "MLP")


if __name__ == "__main__":
    main()
