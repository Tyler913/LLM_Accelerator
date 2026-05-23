# Current State

Last updated: 2026-05-23

This file is the short working-state handoff. For stable project context, read
`PROJECT_CONTEXT.md` first. For workflow rules, read `AGENTS.md`.

## Current Goal

Build a first FPGA demo that can take a prompt and generate text with a small
Qwen3 dense decoder model.

Current target:

- Model: `Qwen/Qwen3-0.6B-Base`
- Local model directory: `Qwen3-0.6B-Base/`
- Inference style: text continuation
- Decode policy: greedy argmax
- Prefill plan: serial prefill using the same one-token path as decode
- First context length target: 128 or 256 tokens
- First quantization direction: simple custom Q4 weight-only format later

## Cloud Sync Status

Cross-platform sync is split between GitHub and Hugging Face:

- GitHub remote: `https://github.com/Tyler913/LLM_Accelerator.git`
- Hugging Face model mirror:
  `Tyler01/qwen3-0p6b-base-llm-accelerator`
- Hugging Face model revision:
  `d297782df3b18206f4b1caea202cf6272bae3aa9`
- Mirror contents: model/tokenizer assets only, including
  `model.safetensors`; project validation scripts remain in GitHub.

Fresh machines should clone GitHub, create the `llm_fpga` conda environment,
log in to Hugging Face if needed, run `init/download_model_assets.py`, then run
`init/verify_assets.py`.

The conceptual hardware function remains:

```text
next_token = run_one_token(input_token, position)
```

During prompt prefill, PS calls this once per prompt token. During decode, PS
calls it once per generated token.

## Python Validation Status

The Python validation phase is complete for the current FP32 reference.

Validated coverage:

- Model loading and generation
- Serial prefill/decode with Hugging Face `past_key_values`
- Model config, weight shapes, and KV cache shapes
- Manual Layer 0 Q/K/V and full non-cached Layer 0
- Manual Layer 0 cached decode
- Manual full 28-layer single-token cached decode
- Module-by-module checks for embedding, RMSNorm, GEMV, RoPE, KV cache,
  attention, MLP, decoder layer, final norm, LM head, argmax, and full
  `run_one_token`

Detailed validation logic now lives in the Python files instead of this
document:

- Full-model and exploratory references:
  `Qwen3-0.6B-Base/pc_testing/`
- Per-module FPGA bring-up references:
  `Qwen3-0.6B-Base/python_each_module/`
- Per-module validation index:
  `Qwen3-0.6B-Base/python_each_module/README.md`

Useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
```

Known validated prompt:

```text
The future of FPGA is
```

Observed greedy continuation checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## Key Runtime Facts

Model facts needed for first FPGA planning:

- Layers: 28
- Hidden size: 1024
- MLP intermediate size: 3072
- Query heads: 16
- KV heads: 8
- Head dim: 128
- GQA ratio: 2 query heads per KV head
- Vocabulary size: 151,936
- RMSNorm epsilon: `1e-6`
- RoPE theta: 1,000,000
- Tied embedding/LM-head table shape: `[151936, 1024]`

Single-token layer flow:

```text
[1024]
  -> input RMSNorm
  -> q_proj [2048] -> [16, 128]
  -> k_proj [1024] -> [8, 128]
  -> v_proj [1024] -> [8, 128]
  -> q_norm/k_norm
  -> RoPE on Q/K
  -> attention over per-layer KV cache
  -> attention concat [2048]
  -> o_proj [1024]
  -> residual
  -> post-attention RMSNorm
  -> gate/up [3072]
  -> silu(gate) * up
  -> down [1024]
  -> residual
```

KV cache shape per layer:

```text
K: [1, 8, T, 128]
V: [1, 8, T, 128]
```

Cache contents:

- K cache stores K after q/k RMSNorm and RoPE.
- V cache stores reshaped/transposed `v_proj` output without RoPE.
- Q is not cached.

KV cache memory estimate across all 28 layers:

- Per token per layer: `2 * 8 * 128 = 2048` values
- Per token all layers: `28 * 2048 = 57344` values
- Context 128: FP16/BF16 about 14 MiB, INT8 about 7 MiB
- Context 256: FP16/BF16 about 28 MiB, INT8 about 14 MiB
- Context 512: FP16/BF16 about 56 MiB, INT8 about 28 MiB
- Context 1024: FP16/BF16 about 112 MiB, INT8 about 56 MiB

## Immediate Next Step

Move from Python validation into FPGA preparation:

1. Export compact FPGA test vectors from the validated Python references.
2. Draft the first memory map for weights, per-layer KV cache, activation
   buffers, RoPE tables, logits, and argmax output.
3. Start small FPGA/HLS kernel bring-up against those vectors, beginning with
   RMSNorm and GEMV before assembling a complete decoder layer.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- `CURRENT_STATE.md` should stay concise. Do not add long validation logs here;
  put reusable validation code in Python files and summarize only durable
  results.
