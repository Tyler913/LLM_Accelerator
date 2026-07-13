# Project Context

This repository is for an FPGA-based LLM accelerator project inspired by the
Hummingbird+ paper:

- `paper/3748173.3779189.pdf`
- Title: `Hummingbird+: Advancing FPGA-based LLM Deployment from Research
  Prototype to Edge Product`

For the current working state and immediate next step, read
`Source/CURRENT_STATE.md` after this file.

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
- Current Layer 0 q/k norm gamma fixed-point format: signed 16-bit `Q8.7`
  in the local RTL stage. Layer 0 `q_norm.weight` includes negative values,
  so unsigned gamma is not valid for q/k norm.
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
- Board/material: ALIENTEK/ATK MPSoC-P4 V1.4 reference material, schematic
  title `ATK_DFZU2EG_ZU4EV_1V4`
- PS DDR: 2 GB
- PL DDR4: 0.5 GB / nominal 512 MiB, confirmed as real PL-side DDR4 wiring on
  FPGA Bank 64
- PL DDR4 physical interface: x16 DDR4, `VCCO_64` at `DDR_1V2`,
  `INTERNAL_VREF 0.6`, two DQS pairs, two DM/DBI pins, 100 MHz differential
  PL reference clock on AE5/AF5
- PL DDR4 physical board marking reported on 2026-06-03: `SEC 325`,
  `K4A4G16`, `BCTD`, `6WC0150SC`, matching the Samsung `K4A4G165WF-BCTD`
  4Gb x16 DDR4 datasheet in the board material directory. The schematic
  page-18 `MT40A256M16GE-083E` label is treated as an alternate/library
  placeholder for the populated board unless later BOM evidence says otherwise.
- PS-to-PL PL DDR4 access checkpoint passed on 2026-06-07. The current
  reset-fix design maps AXI BRAM at `0xA000_0000` through `0xA000_1FFF`, DDR4
  status GPIO at `0xA001_0000`, and PL DDR4 at `0x4_0000_0000` through
  `0x4_1FFF_FFFF`. The standalone smoke app reported DDR4 status `0x5`
  (`calib_complete=1`, `ui_reset=0`, `axi_resetn=1`) and exact write/readback
  matches at the PL DDR4 base, near-base, middle, and final aligned word.
- QMAP dot64 PS load/readback passed on 2026-06-07. The app wrote the
  1536-byte first QMAP image to `0x4_1B10_0000`, read it back exactly, and
  checked the QMAP header, four descriptors, and selected payload values.
- QMAP dot64 PL AXI-master compute passed on hardware on 2026-06-11. The PL
  read the QMAP image from real PL DDR4 through AXI and returned status `0xA`,
  partial sum `0x60AF`, and scaled Q26 result `0x2E1366`.
- QMAP row1024 PL AXI-master compute passed on hardware on 2026-06-23. The PS
  loaded the 4096-byte row1024 image at `0x4_1B20_0000`, read it back exactly,
  validated the header, started the PL row compute path, and read back status
  `0xA` plus `row_sum_q26_low32=0xFFCA_DDC7`, representing `-3482169`.
- PS DDR capacity check: the schematic uses four x16 DDR4 devices across
  `PS_DDR4_DQ[63:0]`; with the same 4Gb-class devices this gives 16Gb total,
  matching the 2 GB PS DDR target.
- First KV cache context target for memory-map planning: 256 tokens

The memory-map planning document is `Source/FPGA_MEMORY_MAP.md`. The dual-memory
target distinguishes PS-owned runtime/staging memory from PL-side accelerator
storage. The base PS-to-PL PL DDR4 aperture is now hardware-proven, but the
accelerator layout inside that aperture is still draft. The PL DDR4 is too
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
documented in `Source/Q4_FORMAT.md`.

Descriptor-based PL DDR4 tensor staging is documented in
`Source/QMAP_FORMAT.md`.
QMAP v1 is the intended bridge between generated Q4 artifacts, PS loaders, and
future PL readers. It starts with the real Layer 0 `q_proj` row 0 group 0
dot64 vector and has already scaled to a complete row1024 GEMV image. The next
QMAP direction is a formal inference contract, not another smoke-test packet:
keep a persistent model manifest for loaded PL DDR4 tensors, then use runtime
work packets to describe which tensors a PL kernel reads and which buffers it
writes.

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
- layer input RMSNorm gamma: signed 16-bit `Q8.7`; Layer 1
  `input_layernorm.weight` includes negative values, so unsigned `UQ8.8` is
  not a deployable per-layer format
- post-attention RMSNorm gamma: signed 16-bit `Q8.7` for the current Layer 0
  post_attention_layernorm RTL vector; that tensor includes negative values
- final RMSNorm gamma: signed-capable format required; the current model tensor
  also includes a few negative values, so unsigned gamma is not a safe general
  rule
- q_norm/k_norm gamma: signed 16-bit `Q8.7` for the current Layer 0
  q/k norm + RoPE stage
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

The PS-to-PL memory bring-up phase and the first PL AXI-master QMAP compute
bring-up phase have both reached hardware checkpoints:

1. AXI BRAM, DDR4 status GPIO, and PL DDR4 are reachable from PS.
2. The dot64 QMAP image can be loaded by PS, read by PL through AXI, and
   computed by the Q4 dot-product RTL.
3. The row1024 QMAP image can be loaded by PS, read by PL through AXI, and
   computed by the full-row Q4 GEMV RTL.

The current durable hardware-facing direction is Layer 0 local one-token
datapath integration:

```text
input_norm[1024]
  -> q_proj/k_proj/v_proj Q4 GEMV
  -> Q/K/V output buffers in PL DDR4
  -> q_norm/k_norm, RoPE, KV-cache append, attention, and o_proj
  -> post-attention residual/RMSNorm/MLP
  -> final RMSNorm and LM-head scan/argmax
```

The important new capability is PL write-back. The existing row1024 hardware
checkpoint proves the PL AXI read path and Q4 row GEMV datapath. The local
QKV projection step now has a QMAP exporter, a descriptor-driven projection
compute path, an AXI write adapter, and a Vivado-facing AXI read/write top that
passes local Icarus memory-model simulation through the full `q_proj/k_proj/v_proj`
row counts `2048/1024/1024`. The local attention-front-end now also passes
through K/V cache append: `qk_norm_rope_stage_128.sv` consumes Q4/QMAP fixed Q/K
projection words, applies signed q/k gamma over all 24 heads, and matches
q_norm/k_norm plus RoPE golden outputs exactly; `kv_cache_append.sv` and
`qk_norm_rope_kv_cache_stage.sv` then write exact K/V cache address/data streams
under backpressure. `attention_score_stage.sv` now adds the next local step:
current-token `q_rope` reads a 5-position K cache through a request/response
interface, applies the 16-to-8-head GQA mapping, and emits exact raw/scaled
attention scores under request stalls, response latency, and score-stream
backpressure. `attention_softmax_value_stage.sv` then consumes those scores,
uses a fixed exp-LUT softmax to produce UQ0.16 probabilities, reads cached V
values through a request/response interface, and emits exact Q12.12
`attn_out[16,128]` words under input gaps, V-response latency, and output
backpressure. `o_proj_stage.sv` now consumes that fixed `attn_out[2048]`,
reuses the Q4 GEMV projection controller with a 2048-wide input, and emits exact
Q12.12 `o_proj_out[1024]` words under output backpressure. The post-attention
residual/RMSNorm stage now consumes the fixed `o_proj_out[1024]`, adds it to
the signed Q14.10 residual stream, applies signed Q8.7
`post_attention_layernorm` gamma, and emits exact Q12.12 post-norm words. The
MLP gate/up projection stage now consumes those post-norm words, runs Layer 0
`gate_proj[3072]` and `up_proj[3072]` through the same Q4 weight-only GEMV
contract, and emits exact Q12.12 gate/up pairs under output backpressure. The
MLP SiLU/multiply stage now consumes those gate/up pairs, uses a fixed UQ0.16
sigmoid LUT, and emits exact Q12.12 `mlp_hidden[3072]` words under input and
output backpressure. The MLP down projection stage now consumes that fixed
`mlp_hidden[3072]`, runs Layer 0 `down_proj[1024]` through the same Q4
weight-only GEMV contract extended to a 3072-wide input, and emits exact Q12.12
`down_out[1024]` words under output backpressure. The final MLP residual stage
now consumes `post_attn_hidden[1024]` and `down_out[1024]`, emits exact Q14.10
`layer_out[1024]` words, and covers both busy-period spurious start and
post-done restart behavior in simulation. The full-model current-token final
RMSNorm stage now also passes local simulation from the Python reference's
28-layer final hidden state, with signed Q8.7 final gamma and exact Q12.12
output words. The first tiled Q4 LM-head scan plus greedy argmax stage also
passes locally for a 1024-row scan window that contains the Python-proven
full-Q4/HF argmax token, and the same window now passes through a
memory-backed tile reader/wrapper using exported LM-head weight/scale base
addresses. A runtime LM-head tile scheduler now reuses that memory-backed
engine across multiple tile windows and passes local scans for `64` tiles and
`23` tiles. A QMAP descriptor-backed LM-head wrapper now reads final-norm,
weight, scale, scan-range, output, and debug descriptors, runs the scheduler,
and writes the output token/score descriptor locally. The same QMAP wrapper has
also passed a full `151936`-row / `9496`-tile vocabulary scan in Vivado xsim,
returning the Python-proven token `264` and score `1365150750` with exact Q26
logit checks. The first QMAP final-token tail wrapper now composes
descriptor-provided `final_hidden[1024]` and signed final RMSNorm gamma,
writes `final_norm[1024]` back to a QMAP activation descriptor, then invokes
the full-vocabulary QMAP LM-head wrapper and writes output token/score. This
tail wrapper passes compact Icarus and full-vocabulary Vivado xsim locally.
The first QMAP attention front-end wrapper now also passes locally: it consumes
Q/K/V projection output buffers, q/k gamma, and RoPE cos/sin descriptors, then
writes exact K/V cache entries and exact Q RoPE output. The next QMAP
attention score/value wrapper now also passes locally: it consumes Q RoPE,
reads K/V cache through descriptor-derived addresses, streams scores into
softmax/value, and writes exact `attn_out[2048]`. The QMAP `o_proj` wrapper
now also passes locally: it consumes `attn_out[2048]`, reads persistent Q4
`o_proj` weight/scale rows, and writes exact `o_proj_out[1024]` for the Layer 0
and true Layer 1 packet instances. Hardware confirmation is still pending. The
QMAP post-attention residual/RMSNorm
wrapper now also passes locally: it consumes descriptor-visible residual
input, `o_proj_out[1024]`, and signed post-attention gamma, then writes exact
post-attention hidden and post-norm buffers for the Layer 0 and true Layer 1
packet instances. The QMAP MLP gate/up wrapper now also passes locally: it
consumes descriptor-visible `post_norm[1024]`, reads
persistent Q4 gate/up weight/scale rows, and writes exact gate/up `[3072]`
buffers for the Layer 0 and true Layer 1 packet instances. The QMAP MLP
SiLU/multiply wrapper now also passes locally:
it consumes descriptor-visible gate/up `[3072]` plus a fixed UQ0.16 sigmoid
LUT and writes exact `mlp_hidden[3072]` for the Layer 0 and true Layer 1 packet
instances. The QMAP MLP down wrapper now also passes locally: it consumes
descriptor-visible `mlp_hidden[3072]`, reads persistent per-layer down-proj Q4
weight/scale rows, and writes exact `down_out[1024]` for Layer 0 and true
Layer 1 packet instances. The QMAP final MLP
residual wrapper now also passes locally:
it consumes descriptor-visible `post_attn_hidden[1024]` and `down_out[1024]`,
then writes exact `layer_out[1024]` while catching descriptor/protocol error
paths with no writes. The first local Layer 0 body scheduler now also passes:
it chains the QMAP post-attention residual/RMSNorm, MLP gate/up,
SiLU/multiply, down, and final residual wrappers behind one memory port, with
later descriptors patched to read the previous wrapper's actual write-back
buffers. The wider local Layer 0 scheduler now also passes: it chains the QMAP
attention front-end, attention score/value, `o_proj`, and the body scheduler
behind one memory port with exact K/V cache, Q RoPE, attention, `o_proj`, MLP,
and final `layer_out[1024]` write-back. The first local Layer 0 compute
scheduler now also passes: it runs full QKV projection first, patches the
attention front-end Q/K/V descriptors to those QKV output buffers, and then
runs the wider Layer 0 scheduler through final `layer_out[1024]`. This is a
local RTL simulation checkpoint, not a new board checkpoint. The first local
one-token/layer-loop boundary now also passes for a true chained two-layer
path: it exposes layer index/count, token position, hidden-buffer bases,
KV-cache base, per-layer QMAP packet base tables, done/error masks, and shared
memory ownership, then runs Layer 0 and true Layer 1 in sequence with exact
write-back through Layer 1 `layer_out[1024]`. It also rejects missing-table or
out-of-range layer-loop requests with no memory traffic. The QKV
exporter/AXI wrapper path is now parameterized for true Layer
1 data: `45_export_layer_qkv_q4_vectors.py` exports Layer 1 last-token Q/K/V
Q4 vectors, `21_export_qmap_qkv_projection_image.py --layer-id 1` emits Layer
1 QMAP packets, and the full `2048/1024/1024` Layer 1 QKV packet at
`0x4_1008_0000` passes local AXI write-back comparison. The first downstream
Layer 1 attention front-end packet now also passes at `0x4_1502_0000`, using
parameterized q/k norm+RoPE, KV-cache append, and QMAP packet exporters to
match exact K/V cache writes plus exact Q RoPE write-back. The next Layer 1
attention score/value packet now also passes at `0x4_1503_0000`, using
parameterized score/value golden exporters and a QMAP packet exporter to match
exact K/V cache reads plus exact `attn_out[2048]` write-back. The true Layer 1
`o_proj` packet now also passes at `0x4_1504_0000`, reading persistent Layer 1
`o_proj` Q4 weight/scale bases at `0x4_0700_0000` / `0x4_0710_0000` and
writing exact `o_proj_out[1024]`. The true Layer 1 post-attention
residual/RMSNorm packet now also passes at `0x4_1505_0000`, writing exact
`post_attention_hidden[1024]` and `post_norm[1024]`. The true Layer 1 MLP
gate/up packet now also passes at `0x4_1506_0000`, reading persistent Layer 1
gate/up Q4 weight/scale bases at `0x4_0720_0000` / `0x4_0738_0000` and
`0x4_0740_0000` / `0x4_0758_0000`, then writing exact gate/up `[3072]`. The
true Layer 1 MLP SiLU/multiply packet now also passes at `0x4_1507_0000`,
consuming those gate/up outputs plus the fixed sigmoid LUT and writing exact
`mlp_hidden[3072]`. The true Layer 1 MLP down packet now also passes at
`0x4_1508_0000`, reading persistent Layer 1 down-proj Q4 weight/scale bases at
`0x4_0760_0000` / `0x4_0778_0000` and writing exact `down_out[1024]`. The true
Layer 1 final MLP residual packet now also passes at `0x4_1509_0000`,
consuming Layer 1 post-attention hidden plus Layer 1 `down_out[1024]` and
writing exact `layer_out[1024]`. The focused true Layer 0 -> Layer 1 loop now
passes locally with exact write-back, layer done mask `0x3`, and zero
mismatches. The chained per-layer artifact flow is now reusable through
`47_export_chained_layer_qmap_artifacts.py`, and Layer 2 artifacts generated
from the Layer 1 output pass focused Layer2-only scheduler validation with
active layer 2, layer done mask `0x4`, and zero mismatches.

The final-token tail path has also been parameterized for chained layer
outputs. Using the input-RMSNorm-enabled Layer 2 chained `layer_out[1024]` as
final RMSNorm input, the regenerated full-vocabulary Q4 LM-head tail passes
Vivado xsim with exact token `537`, exact score `850086863`, all `9496` tiles
checked, and zero logit diff. The local scheduler-to-tail handoff now passes in
shared-memory Vivado xsim runs, and the reusable `qmap_one_token_top.sv`
boundary automatically selects the scheduler-reported final layer output as the
tail hidden source while keeping the final-tail packet descriptor stable.

The per-layer input RMSNorm QMAP wrapper before QKV now passes both as a
standalone descriptor-backed RTL/xsim stage and as an optional pre-stage inside
`qmap_layer0_compute_scheduler.sv`: the focused local precheck proves
`hidden[1024] + input_layernorm.gamma -> input_norm[1024] -> Q/K/V` with QKV
golden words regenerated from the RTL RMSNorm output. Real Layer 1 and Layer 2
chained artifacts include their own input RMSNorm runtime packets, and their
QKV activation descriptors point to the corresponding RMSNorm output buffers.
The stronger top-level local baseline now runs through a software-like AXI-Lite
control boundary productized as `qmap_one_token_mmio_top.sv` and
`qmap_one_token_axil_top.sv`; the generic `axi4lite_to_mmio_regs.sv` adapter has
focused stall/error/order coverage, and the wrapper preserves the no-memory
validation behavior. The older mixed bounded-three-layer baseline was first
superseded by one continuous token-driven three-layer lineage and is now
superseded again by the complete model-depth local path:
`53_export_embedding_full28_final_chain.py` propagates tied-Q4 embedding through
all 28 input-RMSNorm-enabled transformer layers, then regenerates final RMSNorm
and the full-vocabulary tied LM head from Layer 27. The continuous AXI-Lite XSim
session
`Temp/embedding_full28_axil_tail_regression/20260713_164001` passes with layer
mask `0x0fffffff`, exact token/score `537/1155032971`, exact scheduler and top
counters, all `281` QMAP packets observed, and every producer write response
before its consumer read. Its `508` physical address intervals are collision-free
inside the PL DDR aperture, and all simulation products remain under `Temp/`.
The first PS-side runtime skeleton now lives under
`FPGA_Project/software/qmap_one_token_runtime/`, mirroring the RTL register map
and providing configure/start/poll/result helpers plus a no-memory AXI-Lite
validation smoke. The first BD-facing shell `qmap_one_token_axi_top.sv` now
adapts that control seam plus the simple memory request/write port into one
AXI4-Lite slave and one AXI4 memory master, with integration notes in
`FPGA_Project/Vivado_Project/ONE_TOKEN_AXI_TOP_BD_PLAN.md`. Its focused Icarus
smoke test now proves the BD-facing shell can accept the no-memory validation
request through `S_AXI` without issuing any `M_AXI` transaction. A safe-by-default
Tcl scaffold, `FPGA_Project/Vivado_Project/scripts/one_token_axi_top_bd_scaffold.tcl`,
now dry-runs the planned BD edits and requires `--apply` before modifying the
current row1024 BD. The next model-facing direction is persistent multi-token
decode: retain per-layer K/V cache across invocations, increment position, feed
the selected token back into embedding, and then add prompt prefill. Vivado
block-design/Vitis integration remains deferred unless a board-specific issue
requires it.

Keep the exact next action in `Source/CURRENT_STATE.md`; keep detailed address
planning in `Source/FPGA_MEMORY_MAP.md`.

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
- Core project handoff Markdown files live under `Source/`; the root
  `README.md` remains the entry point.
- Keep large model weights and generated FPGA/model artifacts out of normal Git
  history.
- Use `init/CROSS_PLATFORM_SYNC.md` and `init/` scripts to restore assets on
  new machines.
- Keep the local Git safety hook installed with
  `init/install_git_safety_hooks.sh` when possible.
- Persisted repository docs should stay concise. Put detailed validation logic
  in Python scripts, not in project handoff docs.
