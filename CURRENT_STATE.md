# Current State

Last updated: 2026-05-12

This file tracks the working state of the FPGA LLM accelerator project. It is
meant to be read after `PROJECT_CONTEXT.md` before continuing implementation.

## Progress Tracking Policy

Update this file automatically after meaningful project progress, preferably in
the same turn as the work. Meaningful progress includes:

- adding or changing a PC-side validation script
- running a validation and learning a useful result
- finding an important tensor shape, memory estimate, layout detail, or model
  behavior
- choosing or changing a model, quantization, PS/PL split, memory layout, or
  hardware direction
- hitting a blocker or documenting a known failure mode
- changing the immediate next step

Keep updates concise and reusable. Record the finding, evidence, command or
script when useful, result, and next action. Do not paste long chat transcripts
or raw logs unless the exact output is necessary.

Use the docs this way:

- `CURRENT_STATE.md`: living progress tracker and immediate next steps
- `PROJECT_CONTEXT.md`: durable project facts and stable design direction
- `AGENTS.md`: repository workflow rules for future agents

## Current Goal

Build a first FPGA demo that can take a prompt and generate text. The first
version does not need chat behavior, high quality sampling, high throughput, or
paper-level optimization. It only needs the full inference chain to run.

Current target:

- Model: `Qwen/Qwen3-0.6B-Base`
- Local directory: `Qwen3-0.6B-Base/`
- Task style: text continuation
- Decode policy: greedy argmax
- Prefill plan: serial prefill using the same one-token path as decode
- First context length target: 128 or 256 tokens
- Later weight target: custom simple Q4, not GGUF/GPTQ/AWQ hardware parsing

## System Split

Current intended first-version split:

- PS:
  - tokenizer
  - detokenizer
  - PL control
  - send prompt/generated token ids to PL
  - receive generated token ids from PL
- PL:
  - embedding lookup
  - complete single-token forward through all 28 layers
  - KV cache read/write
  - final RMSNorm
  - tied LM head
  - greedy argmax

The PL should eventually expose a simple conceptual function:

```text
next_token = run_one_token(input_token, position)
```

During prompt prefill, PS calls this function for each prompt token. During
decode, PS calls it for each generated token.

## Model Facts

Model configuration observed locally:

- Architecture: `Qwen3ForCausalLM`
- Model type: dense decoder-only transformer
- Total parameters: 596,049,920
- Weight dtype in `model.safetensors`: `bfloat16`
- Vocabulary size: 151,936
- Hidden size: 1,024
- Intermediate/MLP size: 3,072
- Layers: 28
- Attention heads: 16
- KV heads: 8
- Head dim: 128
- GQA ratio: 2 Q heads per K/V head
- Max position embeddings: 32,768
- RoPE theta: 1,000,000, stored under `rope_parameters.rope_theta`
- RMSNorm epsilon: `1e-6`
- Activation: `silu`
- `tie_word_embeddings`: true
- BOS/EOS token id: 151,643

The tied embedding/LM-head table has shape `[151936, 1024]`.

## Mental Model

For one token, the model flow is:

```text
token id
  -> embedding table
  -> Layer 0
  -> Layer 1
  -> ...
  -> Layer 27
  -> final RMSNorm
  -> tied LM head
  -> argmax
  -> next token id
```

Each decoder layer takes one `[1024]` hidden vector and returns one `[1024]`
hidden vector.

Layer shape flow:

```text
x: [1024]
  -> input RMSNorm: [1024]
  -> q_proj: [1024] -> [2048] -> [16, 128]
  -> k_proj: [1024] -> [1024] -> [8, 128]
  -> v_proj: [1024] -> [1024] -> [8, 128]
  -> q_norm/k_norm
  -> RoPE on q/k
  -> attention over K/V cache
  -> attention concat: [2048]
  -> o_proj: [2048] -> [1024]
  -> residual add
  -> post-attention RMSNorm: [1024]
  -> gate_proj: [1024] -> [3072]
  -> up_proj: [1024] -> [3072]
  -> silu(gate) * up: [3072]
  -> down_proj: [3072] -> [1024]
  -> residual add
```

Important terminology:

- `Wq/Wk/Wv` are fixed learned weight matrices stored in `model.safetensors`.
- `q/k/v` are runtime intermediate vectors computed from the current hidden
  state using those fixed matrices.
- A head is a slice of the Q/K/V vector. Here, Q has 16 heads of 128 values,
  while K and V each have 8 heads of 128 values.
- In GQA, `kv_head_id = q_head_id // 2`.

## PC Testing Progress

Scripts live under `Qwen3-0.6B-Base/pc_testing/`.

Completed checks:

- `01_test_generate.py`
  - Runs standard Hugging Face `model.generate`.
  - Confirms the downloaded model and tokenizer can generate text.
  - Prompt `The future of FPGA is` generates the expected prefix:
    `The future of FPGA is a fascinating and rapidly evolving field...`
- `02_serial_decode.py`
  - Manually feeds prompt tokens one at a time with `past_key_values`.
  - Then decodes one token at a time using greedy argmax.
  - Produces the same visible prefix as `01_test_generate.py`.
  - This is the closest current software reference for the planned PL
    `run_one_token` behavior.
- `03_inspect_config.py`
  - Prints model structure and confirms the important config values listed
    above.
- `04_inspect_weights.py`
  - Enumerates safetensors tensor names, shapes, dtypes, parameter counts, and
    rough memory estimates.
- `05_inspect_kv_cache.py`
  - Confirms K/V cache shape growth during serial prefill and decode.
  - Fixed to resolve `MODEL_DIR` relative to the script path.
- `06_trace_layer0_tensors.py`
  - Hooks selected modules and prints layer 0 tensor shape flow.
  - Fixed to resolve `MODEL_DIR` relative to the script path.
- `07_manual_layer0_qkv.py`
  - Adds a hardware-oriented manual reference for layer 0 embedding lookup,
    input RMSNorm, and `q_proj`/`k_proj`/`v_proj` GEMV.
  - Documents the math formulas used for embedding lookup, RMSNorm, PyTorch
    Linear/GEMV, and Q/K/V head reshaping inside the script output and source.
  - Uses explicit `Any` casts and tensor annotations around Hugging Face
    dynamic model objects so IDE/Pylance static analysis does not mistake
    Qwen3 modules for plain tensors or generic modules.
  - Validated with:
    `conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/07_manual_layer0_qkv.py`
  - Result: manual FP32 tensors matched Hugging Face hook outputs for the
    first token of `The future of FPGA is` with zero observed max absolute
    error for embedding, RMSNorm, q_proj, k_proj, and v_proj.
- English cleanup
  - Converted Chinese comments/docstrings in `02_serial_decode.py`,
    `05_inspect_kv_cache.py`, and `06_trace_layer0_tensors.py` to English.
  - Added the repository rule that persisted code comments, docstrings, script
    output labels, and Markdown documentation should stay in English unless the
    user explicitly requests otherwise.
  - Verified no Chinese characters remain in self-authored project files after
    excluding model tokenizer/merge/vocabulary artifacts.
- Cross-platform initialization flow
  - Added `init/CROSS_PLATFORM_SYNC.md` as the runbook for recreating local
    assets on macOS, Linux, or Windows.
  - Added `init/HUGGINGFACE_WORKFLOW.md` to document Hugging Face Hub login,
    large artifact upload/download, private artifact repository creation, and
    revision-pinning practice.
  - Verified the local `llm_fpga` environment exposes the modern `hf` CLI with
    `auth`, `download`, `upload`, `upload-large-folder`, and `repos create`
    commands.
  - Added `init/model_assets.json`, `init/download_model_assets.py`, and
    `init/verify_assets.py` for model asset download and validation.
  - Added `init/requirements.txt` for the current PC-side Python package set.
  - Updated `.gitignore` so model weights, generated weight packs, caches, and
    FPGA build artifacts stay out of normal Git history.
  - Updated the root `README.md` to point to the context and init runbooks.
- Repository hygiene
  - Diagnosed a `.git` size spike caused by 20.33 GiB of orphaned
    `.git/objects/pack/tmp_pack_*` files and an unreachable
    `model.safetensors` blob that had previously entered the Git object store
    before being committed.
  - Cleaned the orphaned Git objects and Finder-created `.DS_Store` files
    inside `.git`; confirmed `Qwen3-0.6B-Base/model.safetensors` is ignored by
    Git and `.git` is back to a small local size.
  - Added a tracked Git safety hook in `init/git-hooks/pre-commit` plus
    `init/install_git_safety_hooks.sh`; the hook blocks accidental commits of
    common model/artifact extensions and staged files larger than 50 MiB.
  - Installed the safety hook into the current clone's `.git/hooks/pre-commit`.

Use this command style for Python:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/<script>.py
```

## KV Cache Observations

Prompt:

```text
The future of FPGA is
```

Token ids:

```text
[785, 3853, 315, 89462, 374]
```

Token text:

```text
785   -> 'The'
3853  -> ' future'
315   -> ' of'
89462 -> ' FPGA'
374   -> ' is'
```

Each layer exposes K and V as:

```text
[batch, num_key_value_heads, seq_len, head_dim]
```

Observed shape:

```text
K shape = [1, 8, T, 128]
V shape = [1, 8, T, 128]
```

As each token is processed, `T` grows:

- After token 0: `T = 1`
- After token 1: `T = 2`
- After token 2: `T = 3`
- After token 3: `T = 4`
- After token 4: `T = 5`
- After generated token `' a'`: `T = 6`

Memory math:

- Per layer per token: `2 * 8 * 128 = 2048` values
- Across 28 layers per token: `28 * 2048 = 57344` values
- FP16/BF16 KV cache:
  - context 128: about 14 MiB
  - context 256: about 28 MiB
  - context 512: about 56 MiB
  - context 1024: about 112 MiB
- INT8 KV cache:
  - context 128: about 7 MiB
  - context 256: about 14 MiB
  - context 512: about 28 MiB
  - context 1024: about 56 MiB

For first hardware bring-up, context 128 or 256 is enough.

## Layer 0 Trace Observations

`06_trace_layer0_tensors.py` confirms these runtime shapes for one-token
forward:

- `embed_tokens`: input `[1, 1]`, output `[1, 1, 1024]`
- `layer0.input_layernorm`: `[1, 1, 1024] -> [1, 1, 1024]`
- `layer0.self_attn.q_proj`: `[1, 1, 1024] -> [1, 1, 2048]`
- `layer0.self_attn.k_proj`: `[1, 1, 1024] -> [1, 1, 1024]`
- `layer0.self_attn.v_proj`: `[1, 1, 1024] -> [1, 1, 1024]`
- `layer0.self_attn.q_norm`: `[1, 1, 16, 128] -> [1, 1, 16, 128]`
- `layer0.self_attn.k_norm`: `[1, 1, 8, 128] -> [1, 1, 8, 128]`
- `layer0.self_attn.o_proj`: `[1, 1, 2048] -> [1, 1, 1024]`
- `layer0.post_attention_layernorm`: `[1, 1, 1024] -> [1, 1, 1024]`
- `layer0.mlp.gate_proj`: `[1, 1, 1024] -> [1, 1, 3072]`
- `layer0.mlp.up_proj`: `[1, 1, 1024] -> [1, 1, 3072]`
- `layer0.mlp.down_proj`: `[1, 1, 3072] -> [1, 1, 1024]`
- `final_norm`: `[1, 1, 1024] -> [1, 1, 1024]`
- `lm_head`: `[1, 1, 1024] -> [1, 1, 151936]`

The hook output does not show RoPE, QK dot products, softmax, or weighted V
accumulation because those are implemented inside the attention forward path as
tensor operations, not separate modules registered in the script.

## Quantization Direction

Do not start by implementing GGUF, GPTQ, AWQ, FP8, or Q4_K in hardware.

The planned first custom Q4 format is:

- signed int4 weights
- group-wise symmetric quantization
- group size 64 initially
- one scale per group
- likely FP16 scale for the first version
- no zero point

Formula:

```text
real_weight = scale[group] * int4_weight
```

Quantization sketch:

```text
scale = max(abs(weight_group)) / 7
q = round(weight / scale)
q = clamp(q, -8, 7)
```

Keep activations, accumulators, and KV cache simpler at first. Q4 here means
weight-only 4-bit quantization, not that every intermediate value is int4.

## Next Steps

Completed immediate script:

```text
07_manual_layer0_qkv.py
```

Goal:

- Use the real loaded PyTorch model weights.
- Take the first token embedding.
- Manually implement layer 0 RMSNorm.
- Manually implement `q_proj`, `k_proj`, and `v_proj` as matrix-vector
  multiplies.
- Compare manual outputs against Hugging Face hook outputs.
- Print max absolute error and a few sample values.

Immediate next validation:

- Reproduce `q_norm` and `k_norm`.
- Reproduce RoPE for position 0 and position 1.

After that:

- Reproduce one layer of attention using the observed KV cache layout.
- Build a full FP32 Python reference for single-token inference.
- Add custom Q4 quantization.
- Export a simple FPGA binary weight format and memory map.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep repository code, comments, docstrings, script output labels, and
  Markdown documentation in English unless the user explicitly requests
  otherwise.
- Large model weights and generated FPGA weight packs should be restored from
  `init/` scripts or external artifact storage, not committed to normal Git.
- Prefer `dtype=...` with newer Transformers, though older scripts may still
  use `torch_dtype=...`.
- For scripts under `Qwen3-0.6B-Base/pc_testing/`, prefer:

```python
MODEL_DIR = Path(__file__).resolve().parents[1]
```

This avoids failures when running scripts by absolute path from the repository
root.
