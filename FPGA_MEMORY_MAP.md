# FPGA Memory Map

Status: working draft, updated through successful AXI BRAM smoke-test hardware run

This document defines the first FPGA-visible memory layout for the Qwen3
0.6B accelerator bring-up. It distinguishes confirmed addresses from planning
allocations that still depend on adding the PL DDR4 memory controller.

For the current project state, read `PROJECT_CONTEXT.md` and
`CURRENT_STATE.md` first.

## Hardware Target

- Device family: Xilinx Zynq UltraScale+ MPSoC
- Device: `XCZU2EG`
- Vivado project part: `xczu2eg-sfvc784-2-i`
- Board name: custom / board part currently unset in the Vivado project
- PS DDR: 2 GiB, confirmed in the exported handoff as DDR low memory
  `0x0000_0000` through `0x7FFF_FFFF`
- PS DDR type/config: DDR4, 64-bit controller bus, x16 DRAM device width,
  nominal DDR controller frequency about 600 MHz in the current PS config
- PL DDR4: 0.5 GB / nominal 512 MiB, user-reported, not yet instantiated in
  the current Vivado block design
- Vivado/Vitis version: Vivado 2025.1.1 in the current project
- First software model: `Qwen/Qwen3-0.6B-Base`
- First accelerator goal: single-token `run_one_token(input_token, position)`

TODO:

- Replace custom board placeholder if the vendor board/product name becomes
  known.
- PL DDR4 physical base/range: base TODO, range expected to be about 512 MiB.
- PL DDR4 memory controller IP: TODO.
- Decide whether PS reaches PL DDR4 directly through an AXI memory-mapped path,
  through DMA, or through a staged debug path.
- Decide the PL kernel access path to PL DDR4 after the memory controller is
  added.

## Confirmed Current Address Map

This section is the source of truth for what exists in the current exported
AXI BRAM smoke-test hardware:

```text
FPGA_Project/Vivado_Project/llm_system_axi_bram_smoke.xsa
```

| Space / IP | Interface | Base | High | Size | Status |
| --- | --- | ---: | ---: | ---: | --- |
| PS DDR low memory | PS DDR | `0x0000_0000` | `0x7FFF_FFFF` | 2 GiB | confirmed in XSA/HWH |
| AXI BRAM smoke-test memory | `M_AXI_HPM0_LPD` -> AXI SmartConnect -> AXI BRAM Controller `S_AXI` | `0x8000_0000` | `0x8000_1FFF` | 8 KiB | confirmed in Address Editor, XSA/HWH, and hardware run |
| PL DDR4 | TODO | TODO | TODO | about 512 MiB expected | not in current block design |

Current PS-to-PL smoke-test fabric:

```text
M_AXI_HPM0_LPD
  -> AXI SmartConnect
  -> AXI BRAM Controller
  -> Block Memory Generator
```

Confirmed interface facts from the current handoff:

- `M_AXI_HPM0_LPD` is the active PS master used for the BRAM smoke test.
- `M_AXI_HPM0_LPD` data width: 32 bits.
- AXI BRAM controller `S_AXI` data width: 32 bits.
- AXI BRAM controller address width: 13 bits.
- AXI BRAM controller uses one BRAM interface in the passing hardware
  configuration (`C_SINGLE_PORT_BRAM=1` / `SINGLE_PORT_BRAM=1`).
- BRAM depth: 2048 32-bit words = 8192 bytes.
- `pl_clk0` frequency in the exported handoff: about 96.97 MHz.
- `M_AXI_HPM0_FPD` remains enabled but intentionally unconnected for now.
  It is the only known incomplete address path in the current block design and
  is a candidate for a later high-bandwidth path, including possible PL DDR4
  bring-up work.

## Design Intent

The first version prioritizes correctness, observability, and incremental
bring-up over peak throughput.

High-level split:

- PS:
  - tokenizer and detokenizer
  - prompt and generated token buffers
  - model/artifact loading
  - accelerator control
  - debug and validation orchestration
- PL:
  - one-token forward datapath kernels
  - FPGA-visible activation buffers
  - KV cache read/write
  - future custom Q4 weight reads
  - final LM-head scan and greedy argmax

## Addressing Rules

- All offsets in this document are byte offsets relative to the selected memory
  space unless marked as physical addresses.
- Physical addresses are written as byte addresses.
- Current AXI BRAM smoke-test access should use 32-bit word accesses aligned to
  4-byte boundaries.
- Endianness: little-endian for PS-side scalar accesses.
- Scalar format: unsigned 32-bit words for the first BRAM read/write test.
- Vector format: TODO for accelerator data; keep exported Python vectors as the
  reference until a hardware transfer format is fixed.
- Matrix layout: TODO; row-major remains the working candidate for first GEMV
  bring-up.
- Cache coherency rule between PS and PL: TODO. For the BRAM smoke test, use
  direct volatile memory-mapped accesses or cache-disabled/flush-invalidate
  handling in the standalone application.
- Required flush/invalidate operations: TODO after the Vitis domain/cache
  settings are chosen.
- The current standalone AXI BRAM smoke test has validated direct 32-bit
  `Xil_Out32` / `Xil_In32` accesses to `0x8000_0000` through `0x8000_1FFF`.
  Do not assume the same cache/coherency behavior for future PL DDR4 or DMA
  paths until those paths are tested.

## Memory Spaces

### PS DDR

Purpose: host/runtime memory owned primarily by the processing system.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| PS DDR low memory | `0x0000_0000` | 2 GiB | PS | Main PS memory aperture, `0x0000_0000` through `0x7FFF_FFFF` | confirmed |
| Runtime workspace | linker/domain selected | TODO | PS | bare-metal text/data, stack, heap, or later OS runtime | planned |
| Token buffers | PS DDR offset TODO | TODO | PS | prompt token ids, generated token ids | planned |
| Loader staging | PS DDR offset TODO | TODO | PS | model or artifact staging before PL DDR copy | planned |
| Debug capture | PS DDR offset TODO | TODO | PS | kernel outputs copied back for validation | planned |
| Reserved | PS DDR offset TODO | TODO | PS | future use | planned |

Notes:

- The current handoff exposes DDR low memory as `0x0000_0000` through
  `0x7FFF_FFFF`.
- Do not hard-code PS DDR suballocations until the Vitis standalone app and
  linker script are created.

### PL BRAM Smoke Test

Purpose: minimal PS-to-PL memory-mapped access test before adding PL DDR4.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| AXI BRAM smoke-test memory | `0x8000_0000` | 8 KiB | PS write/read through `M_AXI_HPM0_LPD` | simple pattern write/read validation | passed in hardware |

Validated smoke-test pattern:

```text
write/read 32-bit words:
  0x8000_0000 <- 0xA5A50000
  0x8000_0004 <- 0x5A5A0001
  0x8000_0008 <- 0x12345678
  0x8000_0400 <- 0xDEADBEEF
  0x8000_1FFC <- 0xC001D00D
```

Do not place accelerator control registers in this BRAM range long-term. It is
only a connectivity smoke test unless deliberately repurposed later.

### PL DDR4

Purpose: accelerator-side storage for weights, KV cache, and working buffers.

Current status: user-reported hardware capacity is 0.5 GB / nominal 512 MiB,
but no PL DDR4 controller or address aperture exists in the current exported
Vivado design. The table below uses offsets relative to a future PL DDR4 base.
Replace `PL_DDR4_BASE + offset` with physical addresses only after the
controller is instantiated and Address Editor assigns the range.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| Header / memory-map metadata | `PL_DDR4_BASE + 0x0000_0000` | 1 MiB | PS/PL | magic, version, checksums, layout table | draft |
| Weight region | `PL_DDR4_BASE + 0x0010_0000` | 320 MiB | PS load, PL read | future Q4 weights and scales | draft |
| KV cache region | `PL_DDR4_BASE + 0x1410_0000` | 32 MiB | PL read/write | per-layer K/V cache, context 256 first | draft |
| Activation buffers | `PL_DDR4_BASE + 0x1610_0000` | 64 MiB | PL read/write | hidden, normed, q/k/v, attention, MLP scratch, debug snapshots | draft |
| RoPE table | `PL_DDR4_BASE + 0x1A10_0000` | 8 MiB | PS load or PL read | cos/sin table if precomputed | draft |
| Logits / argmax scratch | `PL_DDR4_BASE + 0x1A90_0000` | 8 MiB | PL write, PS read | LM-head tile output and final token id | draft |
| Test-vector staging | `PL_DDR4_BASE + 0x1B10_0000` | 32 MiB | PS load, PL read/write | optional HLS/RTL bring-up data | draft |
| Reserved | `PL_DDR4_BASE + 0x1D10_0000` | 47 MiB | PS/PL | expansion room within nominal 512 MiB | draft |

Draft relative coverage:

```text
PL_DDR4_BASE + 0x0000_0000
  through
PL_DDR4_BASE + 0x1FFF_FFFF
```

This assumes a 512 MiB usable aperture. Recompute the table if the actual PL
DDR4 controller exposes less than 512 MiB or if alignment constraints reserve
part of the address space.

## Capacity Budget Worksheet

Use this section to prove that the planned PL DDR4 layout fits before building
binary artifacts.

Known model facts:

- Parameters: 596,049,920
- Stored baseline dtype: `bfloat16`
- Baseline full weights: about 1.19 GB, too large for 0.5 GB PL DDR4
- Planned first Q4: signed int4, group-wise symmetric, group size 64, no zero
  point

| Item | Formula | Estimated Size | Decision |
| --- | --- | ---: | --- |
| Nominal PL DDR4 aperture | user-reported 0.5 GB | 512 MiB assumed for draft | must be confirmed in Vivado |
| Q4 packed weights | parameters * 4 bits | about 298.0 MB / 284.2 MiB | fits inside 320 MiB draft weight region |
| Q4 scales | `(parameters / 64) * scale_bytes` | FP16 scales about 18.6 MB / 17.8 MiB | fits with Q4 weights in 320 MiB region |
| Q4 metadata/alignment | format header, per-tensor metadata, padding | about 18 MiB margin if FP16 scales | draft margin |
| KV cache context 128 | 28 * 2 * 8 * 128 * 128 * bytes | FP16/BF16 about 14 MiB | optional |
| KV cache context 256 | 28 * 2 * 8 * 256 * 128 * bytes | FP16/BF16 about 28 MiB | first target, allocate 32 MiB |
| KV cache context 512 | 28 * 2 * 8 * 512 * 128 * bytes | FP16/BF16 about 56 MiB | does not fit current 32 MiB KV allocation |
| Activation buffers | one-token buffers plus debug room | allocate 64 MiB | draft |
| RoPE table | depends on max context and dtype | allocate 8 MiB | draft; enough for small-context precomputed tables |
| Logits scratch | vocab_size * bytes or tiled | full FP32 logits about 0.58 MiB | allocate 8 MiB for tiled/debug use |
| Test-vector staging | exported bring-up vectors and scratch | allocate 32 MiB | draft |
| Reserved margin | remaining nominal 512 MiB | 47 MiB | draft |
| Total draft PL DDR4 layout | sum above | 512 MiB | must match final PL DDR4 aperture |

Current capacity read:

- Q4 weights plus FP16 scales should fit in 0.5 GB PL DDR4 with context 256 KV
  cache, assuming modest activation/debug buffers and no full BF16 weights.
- Exact fit must still be checked after the PL DDR4 usable address range,
  alignment rules, and Q4 artifact format are fixed.
- If FP32 scales are used instead of FP16 scales, the scale table grows to
  about 35.5 MiB. That still appears plausible, but it reduces weight-region
  metadata/alignment margin and should be re-budgeted before generating a full
  artifact.

## Weight Layout

TODO: Define this before generating full-model Q4 artifacts.

### Region Header

| Field | Type | Description | Status |
| --- | --- | --- | --- |
| magic | TODO | identifies this artifact format | TODO |
| format_version | TODO | memory-map/weight-format version | TODO |
| source_model | TODO | model source and revision | TODO |
| quant_format | TODO | e.g. custom_groupwise_symmetric_q4 | TODO |
| group_size | TODO | expected initial value: 64 | TODO |
| scale_dtype | TODO | expected first version: FP16 or FP32 | TODO |
| checksum | TODO | artifact integrity check | TODO |

### Per-Layer Weight Order

Fill in exact offsets and sizes after choosing packing and alignment.

| Layer Item | Shape | Format | Offset | Size | Notes |
| --- | --- | --- | ---: | ---: | --- |
| embed_tokens / lm_head | `[151936, 1024]` | TODO | TODO | TODO | tied weights |
| layer N input RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |
| layer N q_proj | `[2048, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N k_proj | `[1024, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N v_proj | `[1024, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N q_norm weight | `[128]` or config-derived | TODO | TODO | TODO | confirm shape |
| layer N k_norm weight | `[128]` or config-derived | TODO | TODO | TODO | confirm shape |
| layer N o_proj | `[1024, 2048]` | TODO | TODO | TODO | row-major candidate |
| layer N post-attn RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |
| layer N gate_proj | `[3072, 1024]` | TODO | TODO | TODO | MLP |
| layer N up_proj | `[3072, 1024]` | TODO | TODO | TODO | MLP |
| layer N down_proj | `[1024, 3072]` | TODO | TODO | TODO | MLP |
| final RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |

Open decisions:

- Store embedding/LM head as Q4 immediately, or keep a higher precision first?
- Use row-major, column-major, or tiled layout for GEMV?
- Pack two signed int4 values per byte in low/high nibble order:
- Scale placement: separate scale array or interleaved by block:
- Alignment per matrix:
- Checksum granularity:

## KV Cache Layout

Logical shape per layer:

```text
K: [1, 8, T, 128]
V: [1, 8, T, 128]
```

Current validated facts:

- K cache stores K after q/k RMSNorm and RoPE.
- V cache stores reshaped/transposed `v_proj` output without RoPE.
- Q is not cached.

TODO: Choose a physical layout.

Candidate layout:

```text
kv_cache[layer][kv_kind][head][position][head_dim]
```

Where:

- `kv_kind = 0` for K
- `kv_kind = 1` for V

| Field | Value | Status |
| --- | --- | --- |
| Max context | 256 for first target | draft |
| Cache dtype | FP16/BF16 first planning assumption | draft |
| Layer stride | `2 * 8 * max_context * 128 * bytes_per_value` | draft |
| K/V stride | `8 * max_context * 128 * bytes_per_value` | draft |
| Head stride | `max_context * 128 * bytes_per_value` | draft |
| Position stride | `128 * bytes_per_value` | draft |
| Head-dim stride | `bytes_per_value` | draft |
| Append address formula | TODO | TODO |
| Read address formula | TODO | TODO |

## Activation Buffers

Logical first-version one-token buffers:

| Buffer | Shape | Format | Location | Reuse Rule | Status |
| --- | --- | --- | --- | --- | --- |
| hidden_in | `[1024]` | TODO | PL DDR4 or on-chip | input to layer | TODO |
| input_norm | `[1024]` | TODO | TODO | RMSNorm output, GEMV input | TODO |
| q_flat | `[2048]` | TODO | TODO | Q projection output | TODO |
| k_flat | `[1024]` | TODO | TODO | K projection output | TODO |
| v_flat | `[1024]` | TODO | TODO | V projection output | TODO |
| q_rope | `[16, 128]` | TODO | TODO | attention input | TODO |
| k_rope | `[8, 128]` | TODO | TODO | cache append | TODO |
| v_state | `[8, 128]` | TODO | TODO | cache append | TODO |
| attn_out | `[2048]` | TODO | TODO | o_proj input | TODO |
| layer_hidden | `[1024]` | TODO | TODO | residual result | TODO |
| post_norm | `[1024]` | TODO | TODO | MLP input | TODO |
| gate | `[3072]` | TODO | TODO | MLP intermediate | TODO |
| up | `[3072]` | TODO | TODO | MLP intermediate | TODO |
| mlp_hidden | `[3072]` | TODO | TODO | silu(gate) * up | TODO |
| logits_tile | TODO | TODO | TODO | LM-head tile scan | TODO |

Open decisions:

- Which buffers must stay on-chip for performance?
- Which buffers can live in PL DDR4 for first correctness bring-up?
- Use ping-pong hidden buffers between layers?
- Debug capture points:

## RoPE Table

TODO: Decide whether RoPE cos/sin values are precomputed or generated in PL.

| Option | Pros | Cons | Decision |
| --- | --- | --- | --- |
| PS precomputes table in PL DDR4 | simpler PL logic | consumes PL DDR4 and bandwidth | TODO |
| PL computes cos/sin | less table memory | more math/control complexity | TODO |
| Hybrid small table/tile | possible compromise | more bookkeeping | TODO |

Fields to define:

- Max context:
- Dtype:
- Shape:
- Layout:
- Address formula:

## Logits and Argmax

The vocabulary size is 151,936. A full logits vector can be large, so the first
hardware plan may scan the tied LM head in tiles and keep only the running
argmax.

TODO:

- Full logits stored or tiled scan only:
- Logit dtype:
- Tile size:
- Reduction method:
- Argmax output address:
- Tie-breaking rule:

## Control Register Map

TODO: Fill this after choosing AXI4-Lite or another control path.

Current AXI BRAM smoke-test range is not the final control register map. It is
only used to verify that PS can access a PL memory-mapped slave:

| Smoke-Test Item | Address | Width | Access | Description |
| --- | ---: | ---: | --- | --- |
| BRAM word 0 | `0x8000_0000` | 32 | R/W | first pattern-test word |
| BRAM word N | `0x8000_0000 + 4*N` | 32 | R/W | valid for `0 <= N < 2048` |

Future accelerator control should probably use AXI4-Lite registers separate
from this temporary BRAM aperture.

| Register | Offset | Width | Access | Description |
| --- | ---: | ---: | --- | --- |
| control | TODO | TODO | R/W | start, done, idle, error |
| input_token | TODO | TODO | W | current token id |
| position | TODO | TODO | W | current sequence position |
| sequence_length | TODO | TODO | W | cache length after append |
| output_token | TODO | TODO | R | greedy argmax token id |
| status | TODO | TODO | R | debug/error status |
| weight_base | TODO | TODO | W | PL DDR4 weight base |
| kv_cache_base | TODO | TODO | W | PL DDR4 KV cache base |
| activation_base | TODO | TODO | W | PL DDR4 activation base |
| rope_base | TODO | TODO | W | PL DDR4 RoPE table base |
| debug_base | TODO | TODO | W | optional debug output base |

## Data Movement Sequence

TODO: Confirm the actual boot/runtime flow.

Candidate first-version flow:

1. PS boots and initializes runtime.
2. PS loads model/artifact metadata from storage into PS DDR.
3. PS copies required FPGA artifacts into PL DDR4.
4. PS writes memory-map/control registers.
5. PS sends prompt tokens one at a time.
6. PL runs `run_one_token(input_token, position)`.
7. PL appends K/V cache and writes greedy next token.
8. PS reads the generated token and repeats decode.
9. Optional debug mode copies intermediate buffers back to PS DDR.

Open questions:

- Does PS access future PL DDR4 directly, or through DMA?
- Is future PL DDR4 coherent with PS caches?
- How are large artifacts loaded into PL DDR4 on Windows/Linux during bring-up?
- What is the fastest reliable path for test-vector staging?
- Does `M_AXI_HPM0_FPD` become the preferred high-bandwidth path, or is it
  disabled until a concrete PL DDR4 controller path is ready?

## Validation Plan

Use the exported FP32 test vectors first:

```text
artifacts/test_vectors/qwen3_0p6b_fp32_v0/
```

Bring-up order:

1. Run the AXI BRAM smoke test from PS:
   - write/read 32-bit patterns at `0x8000_0000`
   - cover at least the first few words, a middle address, and the final
     aligned word at `0x8000_1FFC`
   - status: passed in hardware with exact read-back matches
2. RMSNorm kernel reads `input_hidden`, `norm_weight`, and `eps`; compare with
   `expected_output`.
3. GEMV kernel reads `input_norm` and one projection matrix; compare with
   `expected_q`, `expected_k`, or `expected_v`.
4. Extend vectors and memory regions for RoPE, KV cache, attention, MLP, and
   complete Layer 0.
5. Only after FP32/BF16-style flow is understood, introduce Q4 GEMV vectors and
   the final packed weight layout.

Pass/fail fields to record:

| Kernel | Vector | Max Abs Error | Mean Abs Error | Status |
| --- | --- | ---: | ---: | --- |
| PS-to-PL AXI BRAM | direct pattern test at `0x8000_0000` | exact match | exact match | passed |
| RMSNorm | `rmsnorm_layer0_last_token.npz` | TODO | TODO | TODO |
| Q GEMV | `qkv_layer0_last_token.npz` | TODO | TODO | TODO |
| K GEMV | `qkv_layer0_last_token.npz` | TODO | TODO | TODO |
| V GEMV | `qkv_layer0_last_token.npz` | TODO | TODO | TODO |

## Revision Log

| Date | Change | Notes |
| --- | --- | --- |
| 2026-05-24 | Initial skeleton | Adds PS DDR / PL DDR4 split and TODO sections |
| 2026-05-26 | Add confirmed AXI BRAM smoke-test map and DDR planning draft | Records PS DDR low range, BRAM `0x8000_0000`-`0x8000_1FFF`, and a relative 512 MiB PL DDR4 layout draft |
| 2026-05-27 | Record passing AXI BRAM hardware run | Confirms repeated 32-bit PS write/read access through `M_AXI_HPM0_LPD` at offsets `0x0`, `0x4`, `0x8`, `0x400`, and `0x1FFC` |
