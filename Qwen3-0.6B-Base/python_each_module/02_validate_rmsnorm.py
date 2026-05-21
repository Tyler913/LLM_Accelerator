import torch

from common import (
    capture_module_io,
    compare_tensor,
    encode_prompt,
    finish,
    get_rms_norm_eps,
    load_model,
    manual_rms_norm,
    require_input0,
    require_output,
    tensor_to_float32,
)


def main() -> None:
    tokenizer, model, backbone = load_model()
    input_ids = encode_prompt(tokenizer)
    layer0 = backbone.layers[0]

    records: dict[str, dict[str, torch.Tensor]] = {}
    hooks = [
        layer0.input_layernorm.register_forward_hook(capture_module_io(records, "layer0_input_norm")),
        backbone.norm.register_forward_hook(capture_module_io(records, "final_norm")),
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    layer0_eps = get_rms_norm_eps(layer0.input_layernorm, model.config)
    final_eps = get_rms_norm_eps(backbone.norm, model.config)

    manual_layer0_norm = manual_rms_norm(
        require_input0(records, "layer0_input_norm"),
        tensor_to_float32(layer0.input_layernorm.weight),
        layer0_eps,
    )
    manual_final_norm = manual_rms_norm(
        require_input0(records, "final_norm"),
        tensor_to_float32(backbone.norm.weight),
        final_eps,
    )

    print("RMSNorm validation")
    print("=" * 80)
    print("Formula: y = x * rsqrt(mean(x*x) + eps) * gamma")

    checks = [
        compare_tensor("layer0.input_layernorm", manual_layer0_norm, require_output(records, "layer0_input_norm")),
        compare_tensor("backbone.final_norm", manual_final_norm, require_output(records, "final_norm")),
    ]
    finish(all(checks), "RMSNorm")


if __name__ == "__main__":
    main()
