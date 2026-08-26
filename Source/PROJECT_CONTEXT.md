# Project Context

This document contains durable architecture, hardware, and workflow
constraints. It deliberately does not record incremental simulations, dated
board evidence, temporary directories, or artifact hashes.

For the authoritative release state and next gates, read
`Source/CURRENT_STATE.md`. For the concise release-facing validation summary,
read `Source/RELEASE_VALIDATION.md`. Detailed historical evidence is not part
of the release tree.

The project is inspired by the Hummingbird+ paper retained at
`Source/3748173.3779189.pdf`.

## Goal and First-Version Boundary

The long-term goal is an FPGA inference stack for a small dense decoder-only
LLM. The project prioritizes learning hand-written Verilog/SystemVerilog RTL
and FPGA PL design; PS software is supporting infrastructure for loading,
control, tokenization, detokenization, and validation.

The completed first-version contract is correctness-first:

- serial prompt prefill;
- single-token cached decode;
- greedy argmax;
- PL-owned model math for prefill and decode;
- custom Q4 weight-only storage for large PL-DDR weights;
- a 256-token KV-cache target; and
- hand-written Verilog/SystemVerilog compute blocks by default.

Do not substitute BF16/FP32 full-weight storage, full-precision GEMV, or an
HLS compute flow for this first deployable path. BF16/FP32 data is reference
and bring-up data only.

## Baseline Model

The software baseline is `Qwen/Qwen3-0.6B-Base`, locally restored under
`Qwen3-0.6B-Base/`. Its weights are stored as `model.safetensors` and are not
committed. `init/` contains the asset restore and verification flow.

Important architecture facts:

| Property | Value |
| --- | ---: |
| Model type | `Qwen3ForCausalLM` |
| Parameters | 596,049,920 |
| Vocabulary | 151,936 |
| Hidden size | 1,024 |
| Intermediate size | 3,072 |
| Layers | 28 |
| Attention heads / KV heads | 16 / 8 |
| Head dimension | 128 |
| Maximum position embeddings | 32,768 |
| RoPE theta | 1,000,000 |
| RMSNorm epsilon | `1e-6` |
| BOS/EOS token ID | 151,643 |

The core tensor shapes are:

```text
embedding / tied LM head: [151936, 1024]
Q projection:             [1024] -> [2048] -> [16, 128]
K, V projections:         [1024] -> [1024] -> [8, 128]
attention output:         [2048] -> [1024]
MLP gate, up:             [1024] -> [3072]
MLP down:                 [3072] -> [1024]
```

The Q/K head-normalization gamma vectors are `[128]`; layer and final RMSNorm
gamma vectors are `[1024]`. RMSNorm gamma can contain negative values, so its
deployable representation must be signed.

## System Boundary

The intended model boundary is:

```text
PS: tokenizer -> token IDs -> configure/load/start/poll PL -> token IDs -> detokenizer
PL: embedding -> 28 decoder layers -> KV cache -> final RMSNorm -> LM head -> argmax
```

The reusable hardware operation is conceptually:

```text
next_token = run_one_token(input_token, position)
```

Prompt prefill reuses this one-token operation for every prompt token. During
decode, PS feeds the observed output token back as the next input token. PS may
load runtime data, program register tables, poll status, and report debug
information; it must not become the model-compute path.

## Hardware Target

The target is an ALIENTEK/ATK MPSoC-P4 board using a Xilinx Zynq UltraScale+
MPSoC (`XCZU2EG`, Vivado part `xczu2eg-sfvc784-2-i`).

- PS DDR target: 2 GiB.
- PL DDR4 target: 512 MiB on a x16 DDR4 interface.
- PL weights must use the custom Q4 path because the complete BF16 baseline
  does not fit in PL DDR4.
- The first user-facing Ethernet path is the `PS_ETH` connector through PS
  GEM3 and the YT8521 PHY. `PL_ETH` is outside the first-version standalone
  networking path.

Current aperture, register, reset, and runtime-layout details belong in
`Source/FPGA_MEMORY_MAP.md`; PS Ethernet wiring and bring-up belong in
`Source/PS_NETWORK_BRINGUP.md`.

## Quantization and Numeric Direction

The first deployable PL runtime uses custom Q4 weight-only quantization for all
large model matrices stored in PL DDR4:

- signed int4 weights, range `[-8, 7]`;
- symmetric group-wise quantization with no zero point;
- initial group size `64`;
- one scale per group, stored as unsigned 16-bit `Q2.14`; and
- row-major grouping over contiguous input columns.

QMAP is the descriptor-based bridge between generated runtime artifacts, PS
loaders, and PL readers. The normative documents are:

- `Source/Q4_FORMAT.md` for Q4 packing and fixed-point interpretation;
- `Source/QMAP_FORMAT.md` for descriptor and packet semantics; and
- `Source/FPGA_MEMORY_MAP.md` for PL-DDR placement.

Current fixed-point planning uses signed 24-bit `Q14.10` for residual/RMSNorm
inputs, signed 24-bit `Q12.12` for normalized and compute streams, signed
16-bit `Q8.7` for RMSNorm gamma, and unsigned 24-bit `UQ8.16` for inverse RMS.
These are project contracts, not permission to treat all intermediates as one
unbounded global format: new buffers require range analysis and an explicit
saturation policy.

## KV Cache

Each decoder layer owns separate K and V caches:

```text
K, V: [batch, num_key_value_heads, sequence_length, head_dim]
      [1, 8, T, 128]
```

K is stored after Q/K RMSNorm and RoPE; V is stored after reshaping the V
projection without RoPE. Q is not cached. Cached decode needs no additional
causal mask because the cache contains only past tokens and the current token.

For all 28 layers, each token requires `2 * 8 * 128 = 2048` cache values per
layer, or 57,344 values across the model. A 256-token context is about 28 MiB
at FP16/BF16 or 14 MiB at INT8 before implementation-specific padding.

## Repository and Validation Policy

`FPGA_Project/` is the only canonical RTL, Vivado, software, and simulation
source tree. `lmdeploy/qwen3-0p6b-q4-qweb-demo/` is the portable board release
candidate; it carries manifests, launch support, selected pinned board
artifacts, and no second editable source tree.

Keep model weights, Q4 runtime segments, generated FPGA outputs, temporary
workspaces, and simulator traces out of normal Git history. The tracked
simulation vectors are currently versioned golden fixtures because existing
testbenches consume them directly; do not remove them until a pinned
fetch-and-verify workflow replaces that dependency.

Every Python command must use the project environment:

```bash
conda run -n llm_fpga python ...
```

Useful source-level checks are:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

Use `init/CROSS_PLATFORM_SYNC.md` for asset restoration, and use the portable
demo README for board-launch prerequisites. Do not use broad staging commands
such as `git add .` or `git add -A`.

## Authoritative Documents

| Question | Source of truth |
| --- | --- |
| Release status, limitations, and next gate | `Source/CURRENT_STATE.md` |
| Release-facing validation summary | `Source/RELEASE_VALIDATION.md` |
| Q4 numerical contract | `Source/Q4_FORMAT.md` |
| QMAP packet contract | `Source/QMAP_FORMAT.md` |
| Current PL-DDR map | `Source/FPGA_MEMORY_MAP.md` |
| PS Ethernet and Web bring-up | `Source/PS_NETWORK_BRINGUP.md` |
