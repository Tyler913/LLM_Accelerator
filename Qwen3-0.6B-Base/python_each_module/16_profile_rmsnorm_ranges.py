from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from common import (
    get_rms_norm_eps,
    load_model,
    tensor_to_float32,
)


PROFILE_FEEDBACK_TOKENS = 1
PROMPT_SUITE = [
    ("one_char", "A"),
    ("short_word", "Hello"),
    ("baseline", "The future of FPGA is"),
    ("short_question", "Why do transformers use RMSNorm?"),
    (
        "numbers_and_code",
        "Given x = [3, -1, 4], compute a dot product in fixed point and explain overflow.",
    ),
    (
        "hardware_request",
        "Design a small FPGA module that streams quantized weights from DDR4, multiplies "
        "them by an activation vector, accumulates partial sums, and reports the final token.",
    ),
    (
        "medium_explanation",
        "An FPGA LLM accelerator has to balance memory bandwidth, arithmetic precision, "
        "and control complexity. The processing system should load compact Q4 artifacts, "
        "program base addresses, start the fabric, and read status registers, while the "
        "programmable logic performs RMSNorm, rotary position embedding, attention, MLP, "
        "and greedy argmax for each generated token.",
    ),
    (
        "long_technical_paragraph",
        "This profiling prompt is intentionally longer than the earlier bring-up sentence. "
        "It discusses quantized transformer inference on an edge FPGA, including packed "
        "signed int4 weights, per-group scales, fixed-point activations, residual streams, "
        "RMS normalization, key-value cache layout, memory-mapped control registers, and "
        "incremental validation with Python golden vectors. The goal is not to produce a "
        "beautiful answer, but to exercise many token positions and create a broader range "
        "of hidden states before choosing conservative hardware widths for the first RTL "
        "implementation. If the prompt is too narrow, a later layer may overflow even though "
        "the initial smoke test looked perfect.",
    ),
]
SIGNED_16_FORMATS = [
    ("Q2.14", 2, 14, -2.0, 2.0 - 2.0**-14),
    ("Q3.13", 3, 13, -4.0, 4.0 - 2.0**-13),
    ("Q4.12", 4, 12, -8.0, 8.0 - 2.0**-12),
    ("Q5.11", 5, 11, -16.0, 16.0 - 2.0**-11),
    ("Q6.10", 6, 10, -32.0, 32.0 - 2.0**-10),
    ("Q7.9", 7, 9, -64.0, 64.0 - 2.0**-9),
    ("Q8.8", 8, 8, -128.0, 128.0 - 2.0**-8),
    ("Q9.7", 9, 7, -256.0, 256.0 - 2.0**-7),
    ("Q10.6", 10, 6, -512.0, 512.0 - 2.0**-6),
    ("Q11.5", 11, 5, -1024.0, 1024.0 - 2.0**-5),
    ("Q12.4", 12, 4, -2048.0, 2048.0 - 2.0**-4),
    ("Q13.3", 13, 3, -4096.0, 4096.0 - 2.0**-3),
    ("Q14.2", 14, 2, -8192.0, 8192.0 - 2.0**-2),
    ("Q15.1", 15, 1, -16384.0, 16384.0 - 2.0**-1),
]
UNSIGNED_16_FORMATS = [
    ("UQ2.14", 2, 14, 0.0, 4.0 - 2.0**-14),
    ("UQ4.12", 4, 12, 0.0, 16.0 - 2.0**-12),
    ("UQ6.10", 6, 10, 0.0, 64.0 - 2.0**-10),
    ("UQ8.8", 8, 8, 0.0, 256.0 - 2.0**-8),
    ("UQ10.6", 10, 6, 0.0, 1024.0 - 2.0**-6),
    ("UQ12.4", 12, 4, 0.0, 4096.0 - 2.0**-4),
    ("UQ14.2", 14, 2, 0.0, 16384.0 - 2.0**-2),
]
UNSIGNED_24_FORMATS = [
    ("UQ8.16", 8, 16, 0.0, 256.0 - 2.0**-16),
    ("UQ10.14", 10, 14, 0.0, 1024.0 - 2.0**-14),
    ("UQ12.12", 12, 12, 0.0, 4096.0 - 2.0**-12),
]


@dataclass
class RangeStats:
    count: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    max_abs: float = 0.0
    abs_sum: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        x = tensor_to_float32(x)
        if x.numel() == 0:
            return
        self.count += int(x.numel())
        self.min_value = min(self.min_value, float(torch.min(x).item()))
        self.max_value = max(self.max_value, float(torch.max(x).item()))
        self.max_abs = max(self.max_abs, float(torch.max(torch.abs(x)).item()))
        self.abs_sum += float(torch.sum(torch.abs(x)).item())

    @property
    def mean_abs(self) -> float:
        return self.abs_sum / self.count if self.count else 0.0


@dataclass
class ModuleStats:
    name: str
    category: str
    call_count: int = 0
    input_stats: RangeStats = field(default_factory=RangeStats)
    output_stats: RangeStats = field(default_factory=RangeStats)
    inv_rms_stats: RangeStats = field(default_factory=RangeStats)
    sum_squares_stats: RangeStats = field(default_factory=RangeStats)
    gamma_stats: RangeStats = field(default_factory=RangeStats)


@dataclass
class PromptRun:
    name: str
    char_count: int
    token_count: int
    feedback_token_ids: list[int] = field(default_factory=list)


def choose_signed_format(max_abs: float, formats: list[tuple[str, int, int, float, float]]) -> str:
    for name, _int_bits, frac_bits, low, high in formats:
        if -max_abs >= low and max_abs <= high:
            return f"{name} ({frac_bits} frac bits)"
    return "needs wider format"


def choose_unsigned_format(max_value: float, formats: list[tuple[str, int, int, float, float]]) -> str:
    for name, _int_bits, frac_bits, low, high in formats:
        if max_value >= low and max_value <= high:
            return f"{name} ({frac_bits} frac bits)"
    return "needs wider format"


def category_for_name(name: str) -> str:
    if name.endswith(".self_attn.q_norm"):
        return "q_norm"
    if name.endswith(".self_attn.k_norm"):
        return "k_norm"
    if name.endswith(".input_layernorm"):
        return "input_layernorm"
    if name.endswith(".post_attention_layernorm"):
        return "post_attention_layernorm"
    if name in {"norm", "model.norm"} or name.endswith(".model.norm"):
        return "final_norm"
    return "other"


def register_rmsnorm_hooks(model: Any, model_config: Any) -> tuple[list[Any], dict[str, ModuleStats]]:
    stats: dict[str, ModuleStats] = {}
    hooks: list[Any] = []

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        category = category_for_name(name)
        if category == "other":
            continue

        module_stats = ModuleStats(name=name, category=category)
        module_stats.gamma_stats.update(module.weight)
        stats[name] = module_stats
        eps = get_rms_norm_eps(module, model_config)

        def make_hook(item: ModuleStats, norm_eps: float):
            def hook(_module: Any, inputs: Any, output: Any) -> None:
                if not isinstance(inputs, tuple) or not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(f"{item.name} hook expected tensor input0")
                if not torch.is_tensor(output):
                    raise TypeError(f"{item.name} hook expected tensor output")

                x = tensor_to_float32(inputs[0])
                y = tensor_to_float32(output)
                mean_square = torch.mean(x * x, dim=-1)
                sum_squares = torch.sum(x * x, dim=-1)
                inv_rms = torch.rsqrt(mean_square + float(norm_eps))

                item.call_count += 1
                item.input_stats.update(x)
                item.output_stats.update(y)
                item.sum_squares_stats.update(sum_squares)
                item.inv_rms_stats.update(inv_rms)

            return hook

        hooks.append(module.register_forward_hook(make_hook(module_stats, eps)))

    return hooks, stats


def aggregate_by_category(stats: dict[str, ModuleStats]) -> dict[str, ModuleStats]:
    aggregate: dict[str, ModuleStats] = {}

    for item in stats.values():
        category = item.category
        if category not in aggregate:
            aggregate[category] = ModuleStats(name=category, category=category)
        target = aggregate[category]
        target.call_count += item.call_count
        target.input_stats.update_stats(item.input_stats)
        target.output_stats.update_stats(item.output_stats)
        target.inv_rms_stats.update_stats(item.inv_rms_stats)
        target.sum_squares_stats.update_stats(item.sum_squares_stats)
        target.gamma_stats.update_stats(item.gamma_stats)

    return aggregate


def update_stats_from_stats(self: RangeStats, other: RangeStats) -> None:
    if other.count == 0:
        return
    self.count += other.count
    self.min_value = min(self.min_value, other.min_value)
    self.max_value = max(self.max_value, other.max_value)
    self.max_abs = max(self.max_abs, other.max_abs)
    self.abs_sum += other.abs_sum


RangeStats.update_stats = update_stats_from_stats  # type: ignore[attr-defined]


def print_stats_table(title: str, rows: list[ModuleStats]) -> None:
    print()
    print(title)
    print("-" * 112)
    print(
        f"{'name':40s} {'calls':>5s} "
        f"{'in_max_abs':>12s} {'out_max_abs':>12s} {'gamma_max':>10s} "
        f"{'inv_max':>10s} {'sum_sq_max':>12s}"
    )
    for item in rows:
        print(
            f"{item.name[:40]:40s} {item.call_count:5d} "
            f"{item.input_stats.max_abs:12.6g} {item.output_stats.max_abs:12.6g} "
            f"{item.gamma_stats.max_abs:10.6g} {item.inv_rms_stats.max_abs:10.6g} "
            f"{item.sum_squares_stats.max_abs:12.6g}"
        )


def print_format_recommendations(rows: list[ModuleStats]) -> None:
    global_input_max = max((item.input_stats.max_abs for item in rows), default=0.0)
    global_output_max = max((item.output_stats.max_abs for item in rows), default=0.0)
    global_gamma_max = max((item.gamma_stats.max_abs for item in rows), default=0.0)
    global_inv_max = max((item.inv_rms_stats.max_abs for item in rows), default=0.0)
    global_sum_sq_max = max((item.sum_squares_stats.max_abs for item in rows), default=0.0)

    print()
    print("Suggested fixed-point starting formats")
    print("-" * 112)
    print(f"Observed global input max abs:  {global_input_max:.10g}")
    print(f"Observed global output max abs: {global_output_max:.10g}")
    print(f"Observed global gamma max abs:  {global_gamma_max:.10g}")
    print(f"Observed global inv_rms max:    {global_inv_max:.10g}")
    print(f"Observed global sum_sq max:     {global_sum_sq_max:.10g}")
    print()
    print(f"input/output signed 16-bit minimum fit: {choose_signed_format(max(global_input_max, global_output_max), SIGNED_16_FORMATS)}")
    print(f"gamma signed 16-bit minimum fit:        {choose_signed_format(global_gamma_max, SIGNED_16_FORMATS)}")
    print(f"gamma unsigned 16-bit minimum fit:      {choose_unsigned_format(global_gamma_max, UNSIGNED_16_FORMATS)}")
    print(f"inv_rms unsigned 24-bit minimum fit:    {choose_unsigned_format(global_inv_max, UNSIGNED_24_FORMATS)}")
    print()
    print("Important read:")
    print("  The full-model observed residual stream does not fit signed int16 Q4.12.")
    print("  Layer 0 Q4.12 remains useful as a bring-up vector, but full-model RMSNorm")
    print("  needs either wider activations, per-buffer scaling, or a deliberate clamp policy.")


def encode_text(tokenizer: Any, text: str) -> torch.Tensor:
    encoded: Any = tokenizer(text, return_tensors="pt")
    return encoded.input_ids.to(torch.device("cpu"))


def run_prompt(
    tokenizer: Any,
    model: Any,
    name: str,
    text: str,
) -> PromptRun:
    input_ids = encode_text(tokenizer, text)
    feedback_token_ids: list[int] = []

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        current_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(PROFILE_FEEDBACK_TOKENS):
            outputs = model(
                input_ids=current_token_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            feedback_token_ids.append(int(current_token_ids.item()))
            current_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    return PromptRun(
        name=name,
        char_count=len(text),
        token_count=int(input_ids.shape[1]),
        feedback_token_ids=feedback_token_ids,
    )


def print_prompt_table(prompt_runs: list[PromptRun]) -> None:
    print()
    print("Prompt suite")
    print("-" * 112)
    print(f"{'name':28s} {'chars':>8s} {'tokens':>8s} {'feedback_token_ids':>24s}")
    for item in prompt_runs:
        print(
            f"{item.name[:28]:28s} {item.char_count:8d} {item.token_count:8d} "
            f"{str(item.feedback_token_ids):>24s}"
        )


def main() -> None:
    tokenizer, model, backbone = load_model()
    hooks, stats = register_rmsnorm_hooks(model, model.config)
    prompt_runs: list[PromptRun] = []

    try:
        for name, text in PROMPT_SUITE:
            prompt_runs.append(run_prompt(tokenizer, model, name, text))
    finally:
        for hook in hooks:
            hook.remove()

    aggregate = aggregate_by_category(stats)
    category_rows = [aggregate[name] for name in sorted(aggregate)]
    module_rows = sorted(stats.values(), key=lambda item: item.input_stats.max_abs, reverse=True)

    print("RMSNorm range profile")
    print("=" * 112)
    print(f"Prompts profiled: {len(prompt_runs)}")
    print(f"Total prompt tokens: {sum(item.token_count for item in prompt_runs)}")
    print(f"Feedback tokens per prompt: {PROFILE_FEEDBACK_TOKENS}")
    print(f"Layers: {len(backbone.layers)}")
    print(f"RMSNorm modules hooked: {len(stats)}")

    print_prompt_table(prompt_runs)
    print_stats_table("By category", category_rows)
    print_stats_table("Top modules by input max_abs", module_rows[:12])
    print_format_recommendations(list(stats.values()))


if __name__ == "__main__":
    main()
