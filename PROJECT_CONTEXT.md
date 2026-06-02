# Project Context

This repository is for an FPGA-based LLM accelerator project inspired by the
Hummingbird+ paper:

- `paper/3748173.3779189.pdf`
- Title: `Hummingbird+: Advancing FPGA-based LLM Deployment from Research
  Prototype to Edge Product`

For the current working state and immediate next step, read
`CURRENT_STATE.md` after this file.

## Long-Term Goal

Build toward an FPGA inference stack that can run a small dense decoder-only
LLM end to end, then use that working chain as the base for quantization,
memory layout, kernel optimization, and hardware bring-up.

The user's primary purpose for this project is to become stronger at Verilog/RTL
and FPGA PL development. The PS side should stay as small as practical and serve
as bring-up, control, loading, and validation support rather than becoming the
center of the project.

The first milestone prioritizes correctness and observability over speed:

- serial prompt prefill
- single-token cached decode
- greedy argmax
- required custom Q4 weight-only format for large PL-stored weights
- no chat-specific behavior required
- hand-written Verilog/SystemVerilog RTL for PL compute blocks by default
- no High-Level Synthesis (HLS) unless explicitly requested later

## Baseline Model

Local baseline:

- Hugging Face repo: `Qwen/Qwen3-0.6B-Base`
- Private project mirror: `Tyler01/qwen3-0p6b-base-llm-accelerator`
- Mirror revision: `d297782df3b18206f4b1caea202cf6272bae3aa9`
- Local directory: `Qwen3-0.6B-Base/`
- Model type: dense decoder-only transformer
- Architecture: `Qwen3ForCausalLM`
- Weight file: `Qwen3-0.6B-Base/model.safetensors`
- Stored weight dtype: `bfloat16`

The private mirror is for cross-platform restore of the model/tokenizer assets.
The source model remains `Qwen/Qwen3-0.6B-Base`; project validation scripts in
`Qwen3-0.6B-Base/pc_testing/` and `Qwen3-0.6B-Base/python_each_module/` are
tracked in GitHub, not in the Hugging Face model mirror.

Important model facts:

- Parameters: 596,049,920
- Vocabulary size: 151,936
- Hidden size: 1,024
- Intermediate/MLP size: 3,072
- Layers: 28
- Attention heads: 16
- KV heads: 8
- Head dim: 128
- GQA ratio: 2 query heads per K/V head
- Max position embeddings: 32,768
- RoPE theta: 1,000,000
- RMSNorm epsilon: `1e-6`
- Activation: `silu`
- `tie_word_embeddings`: true
- BOS/EOS token id: 151,643

Important shapes:

- Embedding/LM-head table: `[151936, 1024]`
- Q projection: `[1024] -> [2048] -> [16, 128]`
- K projection: `[1024] -> [1024] -> [8, 128]`
- V projection: `[1024] -> [1024] -> [8, 128]`
- Q/K head RMSNorm gamma vectors: `[128]` each
- Attention concat before `o_proj`: `[2048]`
- `o_proj`: `[2048] -> [1024]`
- MLP gate/up: `[1024] -> [3072]`
- MLP down: `[3072] -> [1024]`
- Layer input, post-attention, and final RMSNorm gamma vectors: `[1024]`

## Target System Split

Current intended first-version split:

- PS:
  - board, runtime, storage, and peripheral initialization
  - tokenizer and detokenizer
  - Q4 artifact loading and transfer into FPGA-visible memory
  - PL control register writes and status polling
  - send prompt/generated token ids, position, and sequence metadata to PL
  - receive generated token ids from PL
  - validation, debug printout, and optional readback of intermediate buffers
- PL:
  - embedding lookup
  - all model-side prefill computation
  - all model-side decode computation
  - complete single-token forward through all 28 layers
  - Q4 GEMV for large model weights
  - RMSNorm, RoPE, attention, MLP, final norm, and LM-head scan
  - KV cache read/write
  - greedy argmax
  - implemented with hand-written RTL as the default development path

The PS must not become the model compute path. It may schedule one token at a
time, load data, configure addresses, wait for completion, and print debug
evidence, but the first-version goal is that both prompt prefill and generated
token decode execute their model math in PL.

The PL should eventually expose:

```text
next_token = run_one_token(input_token, position)
```

Prompt prefill should reuse this same one-token path. This keeps the first FPGA
target aligned with the validated software reference.

Practical first-version call flow:

```text
PS:
  prompt_text -> tokenizer -> prompt token ids
  load custom Q4 artifacts and metadata into FPGA-visible memory
  initialize control registers and base addresses

prefill loop:
  for each prompt token:
    PS writes input_token and position
    PS starts PL run_one_token
    PL executes the full one-token model path and appends K/V cache
    PS waits for done/status

decode loop:
  previous_token = last prefill/decode output token
  while generation continues:
    PS writes previous_token and position
    PS starts PL run_one_token
    PL executes the full one-token model path, reads/appends K/V cache, and
      computes greedy argmax
    PS reads output_token and detokenizes it
```

## Hardware Target

Current user-reported FPGA target:

- Xilinx Zynq UltraScale+ MPSoC
- Device: `XCZU2EG`
- Vivado project part: `xczu2eg-sfvc784-2-i`
- Vivado version: 2025.1
- PS DDR: 2 GB
- PL DDR4: 0.5 GB
- First KV cache context target for memory-map planning: 256 tokens

The first memory-map planning document is `FPGA_MEMORY_MAP.md`. The dual-memory
target means the first hardware architecture should distinguish PS-owned
runtime/staging memory from PL-side accelerator storage. The PL DDR4 is too
small for the complete BF16 baseline weights, so the first deployable PL
weight-storage path must be the project custom Q4 weight-only format plus KV
cache and activation buffers, subject to the detailed capacity budget in the
memory-map document.

## KV Cache Facts

Each decoder layer owns separate K and V caches:

```text
K: [batch, num_key_value_heads, seq_len, head_dim] = [1, 8, T, 128]
V: [batch, num_key_value_heads, seq_len, head_dim] = [1, 8, T, 128]
```

The validated Python references confirm:

- K cache stores K after q/k RMSNorm and RoPE.
- V cache stores reshaped/transposed `v_proj` output without RoPE.
- Q is not cached.
- Cached decode does not need an additional causal mask because the cache only
  contains past tokens plus the current token, never future tokens.

Memory estimate across all 28 layers:

- Per token per layer: `2 * 8 * 128 = 2048` values
- Per token all layers: `28 * 2048 = 57344` values
- Context 128: FP16/BF16 about 14 MiB, INT8 about 7 MiB
- Context 256: FP16/BF16 about 28 MiB, INT8 about 14 MiB

For first hardware bring-up, context 128 or 256 is enough.

## Validation References

Python validation is complete for the current FP32 reference. The detailed
validation history is intentionally kept in executable scripts rather than in
this document.

Use these locations:

- `Qwen3-0.6B-Base/pc_testing/`
  - exploratory and full-reference scripts
  - includes full 28-layer cached decode reference
- `Qwen3-0.6B-Base/python_each_module/`
  - focused module-by-module checks for FPGA bring-up
  - see `Qwen3-0.6B-Base/python_each_module/README.md`

Most useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
```

The module directory covers:

- embedding
- RMSNorm
- Q/K/V GEMV
- q_norm/k_norm and RoPE
- KV cache append/read
- attention
- MLP
- complete Layer 0 cached decoder layer
- final RMSNorm, LM head, argmax
- complete 28-layer cached `run_one_token`

Known validated prompt:

```text
The future of FPGA is
```

Known validated greedy checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## Quantization Direction

Q4 weight-only quantization is a durable project constraint for the first
deployable PL implementation. Do not build a first PL DDR4 storage plan that
requires full BF16/FP32 model weights.

The Verilog-facing Q4 v0 format and current Layer 0 Q/K/V artifact are
documented in `Q4_FORMAT.md`.

Do not start with GGUF, GPTQ, AWQ, FP8, or Q4_K hardware parsing.

The required first custom Q4 format:

- signed int4 weights
- group-wise symmetric quantization
- group size 64 initially
- one scale per group
- current Verilog bring-up scale format: unsigned 16-bit fixed-point `Q2.14`
- no zero point

Formula:

```text
real_weight = scale[group] * int4_weight
```

Sketch:

```text
scale = max(abs(weight_group)) / 7
q = round(weight / scale)
q = clamp(q, -8, 7)
```

Q4 here means weight-only quantization. It applies to the large model weights
stored in PL DDR4, including embedding/LM-head and GEMV matrices. Small
non-matrix parameters such as RMSNorm gamma, Q4 scales, metadata, activations,
accumulators, and KV cache may use simpler fixed-point or FP16-style formats as
explicitly chosen, but they do not relax the Q4 requirement for large weights.

## RMSNorm Range Calibration

The current RMSNorm range profiling script is:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/16_profile_rmsnorm_ranges.py
```

It hooks all 113 RMSNorm modules in `Qwen3-0.6B-Base`:

- 28 input layer norms
- 28 post-attention layer norms
- 28 q_norm modules
- 28 k_norm modules
- 1 final norm

The 2026-05-31 prompt-suite run covered 8 prompts from a one-character input
through a long technical paragraph, for 257 prompt tokens total, plus one
generated feedback token per prompt. The observed maxima were:

| Quantity | Observed Max |
| --- | ---: |
| RMSNorm input max abs | about `6691.77` |
| RMSNorm output max abs | about `588.13` |
| RMSNorm gamma max abs | `192` |
| inv_rms max | about `46.08` |
| sum_squares max | about `4.55e7` |

This is a calibration suite, not a proof over all possible prompts. Still, it
is much broader than the Layer 0 single-token bring-up vector and shows that a
single global signed int16 `Q4.12` activation format is not safe for full-model
RMSNorm/residual streams.

Current conservative starting formats for RMSNorm RTL planning:

- residual/RMSNorm input: signed 24-bit `Q14.10`
- RMSNorm output: signed 24-bit `Q12.12`
- RMSNorm gamma: unsigned 16-bit `UQ8.8`
- inv_rms: unsigned 24-bit `UQ8.16`
- sum_squares accumulator: unsigned 64-bit

Layer 0 `Q4.12` remains useful as a small bring-up vector and for the existing
Q4 GEMV tests, but it should not be treated as the durable full-model
activation format without per-buffer scaling, a wider format, or an explicit
clamp policy.

The current Q4 GEMV RTL defaults to `ACT_WIDTH=24` for the planned signed
`Q12.12` RMSNorm output path. Its internal widths are derived from
`ACT_WIDTH`, so the original 16-bit `Q4.12` bring-up tests remain supported by
explicitly instantiating GEMV with `ACT_WIDTH=16`. With the current group size
and Q2.14 scales, the default 24-bit path derives:

- product width: 28 bits
- 64-lane partial width: 34 bits
- scaled group width: 50 bits
- 16-group row accumulator width: 56 bits

## Immediate Technical Direction

The current phase is FPGA bring-up:

1. Run the PS-side AXI BRAM write/read smoke test at `0x8000_0000`.
2. Add and map PL DDR4 only after the PS-to-PL AXI path is proven.
3. Bring up small hand-written RTL compute blocks against the exported FP32
   vectors, starting with RMSNorm and GEMV.

Keep the exact next action in `CURRENT_STATE.md`; keep detailed address
planning in `FPGA_MEMORY_MAP.md`.

## Python Environment Rule

Every Python command in this repository must use:

```bash
conda run -n llm_fpga python ...
```

Do not use system Python, base conda Python, or another environment unless the
user explicitly changes the rule.

## Repository Hygiene

- Preserve model artifacts and paper files unless the user explicitly asks to
  remove them.
- Keep large model weights and generated FPGA/model artifacts out of normal Git
  history.
- Use `init/CROSS_PLATFORM_SYNC.md` and `init/` scripts to restore assets on
  new machines.
- Keep the local Git safety hook installed with
  `init/install_git_safety_hooks.sh` when possible.
- Persisted repository docs should stay concise. Put detailed validation logic
  in Python scripts, not in project handoff docs.
