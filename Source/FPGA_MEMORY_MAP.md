# FPGA Memory Map

Status: working draft, updated through the QMAP row1024 PL AXI master
bitstream, clean short-path Vitis workspace recreation, and hardware smoke pass
on 2026-06-23

This document defines the first FPGA-visible memory layout for the Qwen3
0.6B accelerator bring-up. It distinguishes hardware-proven base apertures
from draft accelerator subregions that still need real RTL data-movement and
kernel integration.

The descriptor-based tensor staging format for PL DDR4 is
`Source/QMAP_FORMAT.md`. Q4 quantization and packing semantics remain in
`Source/Q4_FORMAT.md`.

For the current project state, read `Source/PROJECT_CONTEXT.md` and
`Source/CURRENT_STATE.md` first.

## Hardware Target

- Device family: Xilinx Zynq UltraScale+ MPSoC
- Device: `XCZU2EG`
- Vivado project part: `xczu2eg-sfvc784-2-i`
- Board/material: ALIENTEK/ATK MPSoC-P4 V1.4 reference material, schematic
  title `ATK_DFZU2EG_ZU4EV_1V4`; Vivado board part currently unset
- PS DDR: 2 GiB physical board target. The current standalone BSP exports the
  low-memory region as `0x0000_0000` through `0x7FEF_FFFF`; keep software
  allocations linker/domain driven rather than hard-coding the top of PS DDR.
- PS DDR type/config: DDR4, 64-bit controller bus, x16 DRAM device width,
  nominal DDR controller frequency about 600 MHz in the current PS config
- PL DDR4: 0.5 GB / nominal 512 MiB, confirmed in the board material as
  PL-side DDR4 wiring and now instantiated in the Vivado block design as a
  DDR4 SDRAM MIG IP with AXI4 enabled
- PL DDR4 board interface: x16 DDR4 on FPGA Bank 64, `VCCO_64` tied to
  `DDR_1V2`, `INTERNAL_VREF 0.6`, two DQS pairs, two DM/DBI pins, and
  `c0_ddr4_reset_n` / `PL_DDR4_RST` on AG9
- PL DDR4 board clock source: 100 MHz differential PL clock on AE5/AF5, named
  `sys_clk_p/sys_clk_n` in the board XDC and `PL_CLK0_P/PL_CLK0_N` in the
  schematic
- PL DDR4 device evidence: physical chip marking reported on 2026-06-03 is
  `SEC 325`, `K4A4G16`, `BCTD`, `6WC0150SC`, matching the Samsung
  `K4A4G165WF-BCTD` 4Gb x16 DDR4 datasheet in the board material directory.
  The schematic page-18 `MT40A256M16GE-083E` label is treated as an
  alternate/library placeholder for the populated board unless later BOM
  evidence says otherwise.
- Capacity cross-check: one 4Gb x16 PL DDR4 device gives 512 MiB / 0.5 GB.
  Schematic pages 14-17 show four x16 PS DDR4 devices feeding
  `PS_DDR4_DQ[63:0]`; four 4Gb-class devices give 16Gb total / 2 GB.
- Vivado/Vitis version: Vivado 2025.1.1 in the current project
- First software model: `Qwen/Qwen3-0.6B-Base`
- First accelerator goal: single-token `run_one_token(input_token, position)`

Current PL DDR4 controller configuration:

- MIG IP cell: `ddr4_0`
- Vivado equivalent preset: `MT40A256M16GE-075E`, used as a 4Gb x16 DDR4
  component equivalent for the populated Samsung `K4A4G165WF-BCTD`
- Memory device interface speed: `833 ps` / DDR4-2400-class operation
- Reference input clock: about `100.040 MHz`, differential, from the board PL
  clock pins AE5/AF5
- PHY/controller ratio: `4:1`
- DDR data width: x16
- AXI data width: 128 bits
- AXI address width: 29 bits, matching a 512 MiB memory aperture
- AXI narrow burst: enabled for 32-bit PS smoke-test accesses through a wider
  AXI slave
- Data mask/DBI: `DM NO DBI`
- Memory address map: `ROW COLUMN BANK`

Completed hardware checkpoint:

- The current reset-fix hardware handoff is
  `FPGA_Project/Vivado_Project/llm_system_pl_ddr4_aux_reset_fix.xsa`.
- The current Vitis platform is `llm_pl_ddr4_aux_reset_fix_platform`.
- The standalone smoke app proved PS read/write access to the current AXI BRAM
  aperture, DDR4 status GPIO, and PL DDR4 aperture on hardware.
- The QMAP load/readback smoke app proved that PS can place the first 1536-byte
  QMAP v1 dot64 image at `0x4_1B10_0000` in PL DDR4 and read it back exactly.
- The current PL AXI master handoff is
  `FPGA_Project/Vivado_Project/llm_system_qmap_row1024_pl_master.xsa`.
- The current clean short-path Vitis workspace is `F:\vws`, with platform
  `p_r1024` and application `a_r1024` for the passing row1024 board run.
- The QMAP dot64 PL master smoke app proved that PL can read the QMAP image
  from real PL DDR4 through its AXI master and produce the expected Q4 dot64
  result.
- The QMAP row1024 PL master smoke app proved that PL can read the 4096-byte
  row1024 QMAP image from real PL DDR4 through its AXI master and produce the
  expected full-row Q4 GEMV result `-3482169`.

## Current Address Map

This section is the source of truth for the current Vivado block design and
successful board smoke test.

Current QMAP row1024 PL master hardware handoff:

```text
FPGA_Project/Vivado_Project/llm_system_qmap_row1024_pl_master.xsa
```

Previous QMAP dot64 PL master hardware handoff, kept as the first PL AXI
master smoke checkpoint:

```text
FPGA_Project/Vivado_Project/llm_system_qmap_dot64_pl_master.xsa
```

Previous reset-fix hardware handoff, kept as the PS-to-PL DDR4 proven
checkpoint:

```text
FPGA_Project/Vivado_Project/llm_system_pl_ddr4_aux_reset_fix.xsa
```

Previous BRAM-only checkpoint, kept only as historical evidence:

```text
FPGA_Project/Vivado_Project/llm_system_axi_bram_smoke.xsa
```

| Space / IP | Interface | Base | High | Size | Status |
| --- | --- | ---: | ---: | ---: | --- |
| PS DDR low memory | PS DDR | `0x0000_0000` | `0x7FEF_FFFF` | about 2 GiB minus reserved top window | exported in current standalone BSP |
| AXI BRAM memory | `M_AXI_HPM0_FPD` -> AXI SmartConnect `M00_AXI` -> AXI BRAM Controller `S_AXI` | `0xA000_0000` | `0xA000_1FFF` | 8 KiB | passed current hardware smoke |
| DDR4 status AXI GPIO | `M_AXI_HPM0_FPD` -> AXI SmartConnect `M02_AXI` -> AXI GPIO `S_AXI` | `0xA001_0000` | `0xA001_FFFF` | 64 KiB | passed current hardware smoke |
| QMAP smoke control/status AXI GPIO | PS `M_AXI_HPM0_FPD` -> AXI SmartConnect -> AXI GPIO `S_AXI` | `0xA002_0000` | `0xA002_FFFF` | 64 KiB | passed dot64 and row1024 PL master hardware smoke |
| QMAP smoke result AXI GPIO | PS `M_AXI_HPM0_FPD` -> AXI SmartConnect -> AXI GPIO `S_AXI` | `0xA003_0000` | `0xA003_FFFF` | 64 KiB | passed dot64 and row1024 PL master hardware smoke |
| PL DDR4 | `M_AXI_HPM0_FPD` -> AXI SmartConnect `M01_AXI` -> AXI Clock Converter -> `ddr4_0/C0_DDR4_S_AXI` | `0x4_0000_0000` | `0x4_1FFF_FFFF` | 512 MiB | passed current hardware smoke |

Current PS-to-PL fabric:

```text
M_AXI_HPM0_FPD
  -> AXI SmartConnect
      M00_AXI -> AXI BRAM Controller -> Block Memory Generator
      M01_AXI -> AXI Clock Converter -> DDR4 MIG C0_DDR4_S_AXI
      M02_AXI -> AXI GPIO DDR4 status register
      additional AXI GPIO slaves for QMAP smoke control/status/result

qmap_row1024_axi_smoke_0/M_AXI
  -> AXI SmartConnect S01_AXI
      -> AXI Clock Converter -> DDR4 MIG C0_DDR4_S_AXI
```

Confirmed block-design and BSP facts:

- `M_AXI_HPM0_FPD` is the active PS master for the current PL memory fabric.
- `M_AXI_HPM0_LPD` is disabled in the current block design to remove the
  incomplete address path used during early bring-up.
- AXI BRAM controller `S_AXI` data width: 32 bits.
- AXI BRAM controller address width: 13 bits.
- AXI BRAM controller uses one BRAM interface in the passing hardware
  configuration (`C_SINGLE_PORT_BRAM=1` / `SINGLE_PORT_BRAM=1`).
- BRAM depth: 2048 32-bit words = 8192 bytes.
- `pl_clk0` frequency in the exported handoff: about 96.97 MHz.
- The DDR4 AXI Clock Converter uses the PS/SmartConnect clock on its `S_AXI`
  side and `ddr4_0/c0_ddr4_ui_clk` on its `M_AXI` side.
- `ddr4_0/c0_ddr4_ui_clk_sync_rst` feeds the DDR UI-domain
  `proc_sys_reset_0/ext_reset_in`.
- `proc_sys_reset_0/aux_reset_in` is tied high because that input is
  active-low in the current IP configuration.
- `proc_sys_reset_0/peripheral_aresetn` drives both `ddr4_0/c0_ddr4_aresetn`
  and the clock converter `m_axi_aresetn`.
- The current BSP defines `XPAR_AXI_GPIO_0_BASEADDR` and
  `XPAR_XGPIO_0_BASEADDR` as `0xA001_0000`, with GPIO width `0x3`.
- The current BSP defines `XPAR_DDR4_0_BASEADDRESS` as `0x400000000` and
  `XPAR_DDR4_0_HIGHADDRESS` as `0x41fffffff`.
- PL DDR4 is above the 32-bit address range. Bare-metal software must use a
  64-bit-capable address type such as `UINTPTR` for `0x4_0000_0000`.

## Design Intent

The first version prioritizes correctness, observability, and incremental
bring-up over peak throughput.

The project focus is PL design with hand-written Verilog/SystemVerilog RTL.
PS-side bare-metal code is used for control, loading, and validation, but it is
not the main learning target. Do not assume an HLS flow unless the project
direction is explicitly changed later.

High-level split:

- PS:
  - tokenizer and detokenizer
  - prompt and generated token buffers
  - model/artifact loading
  - accelerator control
  - debug and validation orchestration
- PL:
  - one-token forward datapath RTL blocks
  - FPGA-visible activation buffers
  - KV cache read/write
  - custom Q4 weight reads as the default PL weight path
  - final LM-head scan and greedy argmax

## Addressing Rules

- All offsets in this document are byte offsets relative to the selected memory
  space unless marked as physical addresses.
- Physical addresses are written as byte addresses.
- Current AXI BRAM and PL DDR4 smoke-test accesses use 32-bit word accesses
  aligned to 4-byte boundaries.
- Endianness: little-endian for PS-side scalar accesses.
- Scalar format: unsigned 32-bit words for the current BRAM and PL DDR4
  smoke-test reads/writes.
- Vector format: QMAP v1 for structured PL DDR4 tensor staging; keep exported
  Python vectors as golden references.
- Matrix layout: QMAP descriptors record logical shape and physical byte
  strides. Row-major contiguous Q4 groups remain the first GEMV bring-up
  layout from `Source/Q4_FORMAT.md`.
- Cache coherency rule between PS and PL: TODO. For the current standalone
  smoke tests, use direct memory-mapped accesses plus explicit cache
  flush/invalidate handling where the app touches cached regions.
- Required flush/invalidate operations: TODO after the Vitis domain/cache
  settings are chosen.
- The current standalone smoke app has validated direct 32-bit `Xil_Out32` /
  `Xil_In32` accesses to AXI BRAM at `0xA000_0000` through `0xA000_1FFF` and
  PL DDR4 at selected addresses in `0x4_0000_0000` through `0x4_1FFF_FFFF`.
- PL DDR4 addresses are 64-bit physical addresses. C code must use `UINTPTR`
  or another 64-bit-capable type and should print high/low 32-bit halves for
  debug output.
- Do not assume future PL masters, DMA engines, or cached PS buffer paths have
  the same coherency behavior as the current direct standalone MMIO smoke app.

## Memory Spaces

### PS DDR

Purpose: host/runtime memory owned primarily by the processing system.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| PS DDR low memory | `0x0000_0000` | about 2 GiB minus reserved top window | PS | Main standalone BSP aperture, `0x0000_0000` through `0x7FEF_FFFF` | confirmed |
| Runtime workspace | linker/domain selected | TODO | PS | bare-metal text/data, stack, heap, or later OS runtime | planned |
| Token buffers | PS DDR offset TODO | TODO | PS | prompt token ids, generated token ids | planned |
| Loader staging | PS DDR offset TODO | TODO | PS | model or artifact staging before PL DDR copy | planned |
| Debug capture | PS DDR offset TODO | TODO | PS | kernel outputs copied back for validation | planned |
| Reserved | PS DDR offset TODO | TODO | PS | future use | planned |

Notes:

- The current standalone BSP exposes DDR low memory as `0x0000_0000` through
  `0x7FEF_FFFF`. The physical board target remains 2 GiB PS DDR.
- Do not hard-code PS DDR suballocations until the Vitis standalone app and
  linker script are created.

### PL BRAM Smoke Test

Purpose: minimal PS-to-PL memory-mapped access test kept in the current block
design as a quick sanity check before touching PL DDR4.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| AXI BRAM smoke-test memory | `0xA000_0000` | 8 KiB | PS write/read through `M_AXI_HPM0_FPD` | simple pattern write/read validation | passed in current hardware |

Validated smoke-test pattern:

```text
write/read 32-bit words:
  0xA000_0000 <- 0xA5A50000
  0xA000_0004 <- 0x5A5A0001
  0xA000_0008 <- 0x12345678
  0xA000_0400 <- 0xDEADBEEF
  0xA000_1FFC <- 0xC001D00D
```

Do not place accelerator control registers in this BRAM range long-term. It is
only a connectivity smoke test unless deliberately repurposed later.

The earlier BRAM-only checkpoint used `0x8000_0000` through `0x8000_1FFF` via
`M_AXI_HPM0_LPD`. The integrated PL DDR4 design moved the active PS-to-PL
fabric to `M_AXI_HPM0_FPD` and remapped this smoke BRAM to `0xA000_0000`.

### DDR4 Status GPIO

Purpose: PS-readable debug/status register for the PL DDR4 controller and DDR
AXI reset-release path.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| DDR4 status AXI GPIO | `0xA001_0000` | 64 KiB | PL status, PS read | DDR4 calibration/reset status bits | passed in current hardware |

Bit layout:

| Bit | Signal | Good Value | Meaning |
| ---: | --- | ---: | --- |
| 0 | `ddr4_0/c0_init_calib_complete` | `1` | DDR4 MIG calibration complete |
| 1 | `ddr4_0/c0_ddr4_ui_clk_sync_rst` | `0` | DDR UI-domain reset is deasserted |
| 2 | `proc_sys_reset_0/peripheral_aresetn` | `1` | DDR AXI-side active-low reset is released |

The expected good status word is `0x5`.

### PL DDR4

Purpose: accelerator-side storage for weights, KV cache, and working buffers.

Current status: the PL DDR4 controller is instantiated in the Vivado block
design, Address Editor assigns a 512 MiB aperture, block-design validation and
bitstream generation pass, and PS standalone write/readback smoke testing
passes on hardware.

Current physical base:

```text
PL_DDR4_BASE = 0x4_0000_0000
PL_DDR4_HIGH = 0x4_1FFF_FFFF
```

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| Header / memory-map metadata | `0x4_0000_0000` | 1 MiB | PS/PL | QMAP headers, descriptor tables, version/checksum metadata | draft |
| Weight region | `0x4_0010_0000` | 320 MiB | PS load, PL read | required Q4 weights and scales | draft |
| KV cache region | `0x4_1410_0000` | 32 MiB | PL read/write | per-layer K/V cache, context 256 first | draft |
| Activation buffers | `0x4_1610_0000` | 64 MiB | PL read/write | hidden, normed, q/k/v, attention, MLP scratch, debug snapshots | draft |
| RoPE table | `0x4_1A10_0000` | 8 MiB | PS load or PL read | cos/sin table if precomputed | draft |
| Logits / argmax scratch | `0x4_1A90_0000` | 8 MiB | PL write, PS read | LM-head tile output and final token id | draft |
| Test-vector staging | `0x4_1B10_0000` | 32 MiB | PS load, PL read/write | QMAP v1 test images and optional RTL bring-up data | dot64 and row1024 QMAP images passed PS load/readback and PL master read/compute |
| Reserved | `0x4_1D10_0000` | 47 MiB | PS/PL | expansion room within nominal 512 MiB | draft |

Draft relative coverage:

```text
0x4_0000_0000
  through
0x4_1FFF_FFFF
```

This assumes a 512 MiB usable aperture. Recompute the table if the actual PL
DDR4 controller exposes less than 512 MiB or if alignment constraints reserve
part of the address space.

Validated current smoke-test points:

```text
write/read 32-bit words:
  0x4_00000000 <- 0xD4D40000
  0x4_00000004 <- 0x4D4D0001
  0x4_00001000 <- 0x13572468
  0x4_10000000 <- 0x24681357
  0x4_1FFFFFFC <- 0xDD44AA55
```

First descriptor-based staging contract:

- Format: QMAP v1, documented in `Source/QMAP_FORMAT.md`.
- First QMAP image base: `0x4_1B10_0000`.
- First QMAP image: Layer 0 `q_proj` row 0 group 0 dot64 vector.
- Header: `0x4_1B10_0000`.
- Descriptor table: `0x4_1B10_0100`.
- Payload base: `0x4_1B10_0500`.
- Initial purpose: PS writes and reads back a structured tensor image, then the
  PL QMAP dot64 read/compute smoke path consumes the same descriptor contract
  through its AXI master.
- Hardware result: `qmap_load_smoke_app` passed on 2026-06-07. It wrote the
  1536-byte image with SHA256
  `b56319cf576fe8486e3586ee49a4194323cdfe4f4e6208c8b6057e741e5978d4`,
  read it back byte-for-byte, and checked the QMAP header, four descriptors,
  and selected payload values.
- PL master hardware result: `qmap_pl_compute_smoke_app` passed on
  2026-06-11. It loads the same image, starts `qmap_dot64_axi_smoke_0`,
  observes PL status `0xA`, and reads back `partial_sum_low32=0x60AF` plus
  `scaled_sum_q26_low32=0x2E1366`.

Second descriptor-based staging image:

- Second QMAP image base: `0x4_1B20_0000`.
- Second QMAP image: Layer 0 `q_proj` row 0 full row1024 vector.
- Header: `0x4_1B20_0000`.
- Descriptor table: `0x4_1B20_0100`.
- Payload base: `0x4_1B20_0500`.
- Image size: `0x1000` / 4096 bytes.
- Payloads:
  - activation `[1024]` at `0x4_1B20_0500`, 2048 bytes
  - packed Q4 weight `[1,1024]` at `0x4_1B20_0D00`, 512 bytes
  - Q2.14 scales `[1,16]` at `0x4_1B20_0F00`, 32 bytes
  - expected row Q26 sum at `0x4_1B20_0F40`, 8 bytes
- Local RTL result: `qmap_row1024_compute_path.sv` and
  `qmap_row1024_axi_smoke_top.sv` pass simulation. The AXI smoke top uses 18
  read bursts and reports status `0xA`, with row sum `-3482169`.
- PL master hardware result: `qmap_row1024_pl_compute_smoke_app` passed on
  2026-06-23 in the clean short-path Vitis workspace `F:\vws`. It loads the
  same image, confirms 4096-byte readback, validates the QMAP header, starts
  the row1024 PL compute path, observes PL status `0xA`, and reads back
  `row_sum_q26_low32=0xFFCA_DDC7` plus
  `expected_row_sum_q26_low32=0xFFCA_DDC7`, both representing `-3482169`.

## Capacity Budget Worksheet

Use this section to prove that the planned PL DDR4 layout fits before building
binary artifacts.

Known model facts:

- Parameters: 596,049,920
- Stored baseline dtype: `bfloat16`
- Baseline full weights: about 1.19 GB, too large for 0.5 GB PL DDR4
- Required first Q4: signed int4, group-wise symmetric, group size 64, no zero
  point
- Full BF16/FP32 weights are reference data only and must not be the first PL
  DDR4 storage target.

| Item | Formula | Estimated Size | Decision |
| --- | --- | ---: | --- |
| Nominal PL DDR4 aperture | Address Editor assignment | 512 MiB | assigned in Vivado and passed PS write/readback smoke |
| Q4 packed weights | parameters * 4 bits | about 298.0 MB / 284.2 MiB | fits inside 320 MiB draft weight region |
| Q4 scales | `(parameters / 64) * scale_bytes` | 16-bit scales about 18.6 MB / 17.8 MiB | fits with Q4 weights in 320 MiB region |
| Q4 metadata/alignment | format header, per-tensor metadata, padding | about 18 MiB margin with 16-bit scales | draft margin |
| KV cache context 128 | 28 * 2 * 8 * 128 * 128 * bytes | FP16/BF16 about 14 MiB; fixed 32-bit padded about 28 MiB | optional |
| KV cache context 256 | 28 * 2 * 8 * 256 * 128 * bytes | FP16/BF16 about 28 MiB; fixed 32-bit padded about 56 MiB | first fixed-point target needs more than current 32 MiB draft allocation |
| KV cache context 512 | 28 * 2 * 8 * 512 * 128 * bytes | FP16/BF16 about 56 MiB; fixed 32-bit padded about 112 MiB | does not fit current 32 MiB KV allocation |
| Activation buffers | one-token buffers plus debug room | allocate 64 MiB | draft |
| RoPE table | depends on max context and dtype | allocate 8 MiB | draft; enough for small-context precomputed tables |
| Logits scratch | vocab_size * bytes or tiled | full FP32 logits about 0.58 MiB | allocate 8 MiB for tiled/debug use |
| Test-vector staging | exported bring-up vectors and scratch | allocate 32 MiB | draft |
| Reserved margin | remaining nominal 512 MiB | 47 MiB | draft |
| Total draft PL DDR4 layout | sum above | 512 MiB | must match final PL DDR4 aperture |

Current capacity read:

- Q4 weights plus 16-bit scales should fit in 0.5 GB PL DDR4 with context 256 KV
  cache, assuming modest activation/debug buffers and no full BF16 weights.
- The base 512 MiB PL DDR4 aperture is now hardware-proven, but exact full-model
  fit must still be checked after alignment rules and the final Q4 artifact
  format are fixed.
- If FP32 scales are used instead of FP16 scales, the scale table grows to
  about 35.5 MiB. That still appears plausible, but it reduces weight-region
  metadata/alignment margin and should be re-budgeted before generating a full
  artifact.

## Weight Layout

Draft: use QMAP descriptors for both small bring-up images and future
full-model Q4 artifacts. Do not invent a one-off fixed packet layout for the
dot64 test; it should be the first small instance of the same descriptor
scheme.

Large model weights stored in PL DDR4 must use the project custom Q4
weight-only format. This includes embedding/LM-head and all projection/MLP
matrices. RMSNorm gamma vectors, Q4 scales, metadata, activations,
accumulators, and KV cache may use separate explicitly chosen formats, but they
do not permit a BF16/FP32 full-weight PL DDR4 layout.

The current Verilog-facing Q4 v0 format is documented in
`Source/Q4_FORMAT.md`. The DDR descriptor/image contract is documented in
`Source/QMAP_FORMAT.md`.
Current bring-up artifact scope:

```text
Layer 0 q_proj/k_proj/v_proj only
weights: signed int4, group size 64, two weights packed per byte
scales: unsigned 16-bit fixed-point Q2.14
activation test input: signed int16 fixed-point Q4.12
layout: row-major, per-output-row contiguous 64-column groups
```

### Region Header

| Field | Type | Description | Status |
| --- | --- | --- | --- |
| magic | `u32` | QMAP magic `0x50414D51` / ASCII `QMAP` | draft v1 |
| format_version | `u32` | QMAP version, current value `1` | draft v1 |
| descriptor_table_addr | `u64` | physical address of fixed-size tensor descriptors | draft v1 |
| payload_base_addr | `u64` | first payload byte address | draft v1 |
| image_bytes | `u64` | total occupied image bytes | draft v1 |
| checksum32 | `u32` | optional descriptor/payload checksum; `0` if unused | open |

### Per-Layer Weight Order

Fill in exact tensor ids, offsets, and sizes after extending QMAP from the
first dot64 image to row, tile, projection, layer, and full-model images.

| Layer Item | Shape | Format | Offset | Size | Notes |
| --- | --- | --- | ---: | ---: | --- |
| embed_tokens / lm_head | `[151936, 1024]` | TODO | TODO | TODO | tied weights |
| layer N input RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |
| layer N q_proj | `[2048, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N k_proj | `[1024, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N v_proj | `[1024, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N q_norm weight | `[128]` | TODO | TODO | TODO | shape confirmed; choose fixed-point gamma format |
| layer N k_norm weight | `[128]` | TODO | TODO | TODO | shape confirmed; choose fixed-point gamma format |
| layer N o_proj | `[1024, 2048]` | TODO | TODO | TODO | row-major candidate |
| layer N post-attn RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |
| layer N gate_proj | `[3072, 1024]` | TODO | TODO | TODO | MLP |
| layer N up_proj | `[3072, 1024]` | TODO | TODO | TODO | MLP |
| layer N down_proj | `[1024, 3072]` | TODO | TODO | TODO | MLP |
| final RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |

Open decisions:

- Embedding/LM head are part of the required Q4 weight path; decide only the
  exact packing, scale placement, and validation tolerance.
- Use row-major, column-major, or tiled layout for GEMV?
- Pack two signed int4 values per byte in low/high nibble order as defined in
  `Source/Q4_FORMAT.md` for the first QMAP images.
- Scale placement: current Q4 v0 artifact stores separate scale arrays; QMAP
  should initially use separate scale descriptors, with interleaving left as a
  deliberate later optimization.
- Alignment per matrix:
- Checksum granularity and checksum algorithm:

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

Current first physical layout:

```text
kv_cache[layer][kv_kind][head][position][head_dim]
```

Where:

- `kv_kind = 0` for K
- `kv_kind = 1` for V

| Field | Value | Status |
| --- | --- | --- |
| Max context | 256 first target; 512 supported by address generator | draft |
| Cache dtype | signed 24-bit `Q12.12` padded to 32-bit DDR words for the current fixed-point RTL path | draft |
| Layer stride | `2 * 8 * max_context * 128 * bytes_per_value` | draft |
| K/V stride | `8 * max_context * 128 * bytes_per_value` | draft |
| Head stride | `max_context * 128 * bytes_per_value` | draft |
| Position stride | `128 * bytes_per_value` | draft |
| Head-dim stride | `bytes_per_value`; current fixed-point default is 4 bytes | draft |
| Append address formula | `base + layer*layer_stride + kv_kind*kv_stride + head*head_stride + position*position_stride + dim*bytes_per_value` | initial RTL implemented |
| Read address formula | same formula, sweeping `position` from 0 through current cache length minus 1 | initial RTL implemented |

The first RTL address generator is:

```text
FPGA_Project/rtl/kv_cache_addr_gen.sv
```

It currently generates byte addresses and range-valid flags only. It does not
perform DDR4 reads/writes, AXI handshakes, burst planning, or data packing.
Those functions belong in later append/read controllers. The default
`ELEMENT_BYTES=4` follows the current fixed-point RTL widths: RoPE outputs
signed 24-bit `Q12.12` K values, and V values should be converted into the same
cache element format before storage.

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

TODO: Fill the final accelerator register map after choosing AXI4-Lite or
another long-term control path.

Current AXI BRAM and DDR4 status GPIO ranges are not the final accelerator
control register map. The QMAP GPIOs are also temporary smoke-test registers
for PL AXI master read/compute paths:

| Smoke-Test Item | Address | Width | Access | Description |
| --- | ---: | ---: | --- | --- |
| BRAM word 0 | `0xA000_0000` | 32 | R/W | first pattern-test word |
| BRAM word N | `0xA000_0000 + 4*N` | 32 | R/W | valid for `0 <= N < 2048` |
| DDR4 status | `0xA001_0000` | 3 useful bits | R | `bit0=calib_complete`, `bit1=ui_reset`, `bit2=axi_resetn`; good value is `0x5` |
| QMAP smoke control | `0xA002_0000` | 2 useful bits | W | AXI GPIO channel 1: `bit0=i_start`, `bit1=i_clear`; pulse the bit high then return to zero |
| QMAP smoke status | `0xA002_0008` | 4 useful bits | R | AXI GPIO channel 2: `bit0=busy`, `bit1=done_sticky`, `bit2=error_sticky`, `bit3=compare_match_sticky`; expected pass value is `0xA` |
| QMAP dot64 partial result | `0xA003_0000` | 32 | R | AXI GPIO channel 1 in the dot64 smoke top: `partial_sum_low32`; expected smoke value is `0x0000_60AF` / `24751` |
| QMAP dot64 scaled result | `0xA003_0008` | 32 | R | AXI GPIO channel 2 in the dot64 smoke top: `scaled_sum_q26_low32`; expected smoke value is `0x002E_1366` / `3019622` |
| QMAP row1024 row result | `0xA003_0000` | 32 | R | AXI GPIO channel 1 in the row1024 smoke top: `row_sum_q26_low32`; expected smoke value is `0xFFCA_DDC7` / `-3482169` |
| QMAP row1024 expected result | `0xA003_0008` | 32 | R | AXI GPIO channel 2 in the row1024 smoke top: `expected_row_sum_q26_low32`; expected smoke value is `0xFFCA_DDC7` / `-3482169` |

Future accelerator control should probably use AXI4-Lite registers separate
from these temporary smoke apertures.

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

Current state and open questions:

- Direct PS memory-mapped access to PL DDR4 through `M_AXI_HPM0_FPD` is proven
  for standalone 32-bit smoke-test writes/readbacks.
- The first QMAP v1 dot64 PL DDR4 staging image has passed PS load/readback.
- The first PL-side QMAP descriptor reader, payload fetcher, Q4 dot64 datapath,
  AXI4 read master adapter, and Vivado-facing smoke top have passed simulation.
- `qmap_pl_compute_smoke_app` passed on the board, proving the PL AXI master
  can read the QMAP image from real PL DDR4 and return the expected dot64
  results through the temporary GPIOs.
- The second QMAP v1 row1024 PL DDR4 staging image has passed PS load/readback,
  QMAP header validation, row compute, and result readback on the board through
  `qmap_row1024_pl_compute_smoke_app`.
- Decide later whether DMA is needed for faster bulk copies.
- Future cache coherency policy is still open for PS buffers, PL masters, DMA,
  or cached runtime paths.
- Large-artifact loading flow on Windows/Linux remains open until the Q4
  full-model artifact format and transfer path are fixed.

## Validation Plan

Use the exported FP32 test vectors first:

```text
artifacts/test_vectors/qwen3_0p6b_fp32_v0/
```

Bring-up order:

1. Run the AXI BRAM smoke test from PS:
   - write/read 32-bit patterns at `0xA000_0000`
   - cover at least the first few words, a middle address, and the final
     aligned word at `0xA000_1FFC`
   - status: passed in current hardware with exact read-back matches
2. Run the PL DDR4 smoke test from PS:
   - use the reset-fix Vitis platform built from
     `FPGA_Project/Vivado_Project/llm_system_pl_ddr4_aux_reset_fix.xsa`
   - read the DDR4 status GPIO at `0xA001_0000`
   - require `calib_complete=1`, `ui_reset=0`, and `axi_resetn=1`
   - write/read 32-bit patterns at the first, middle, and final aligned words
     of `0x4_0000_0000` through `0x4_1FFF_FFFF`
   - use `UINTPTR` or another 64-bit-capable address type in C
   - status: passed in current hardware with exact read-back matches
3. Export and verify a QMAP v1 dot64 staging image from
   `q_proj_row0_group0_dot64.npz`; status: passed in Python exporter.
4. PS writes and reads back that QMAP image at `0x4_1B10_0000`.
   - status: passed on hardware with exact 1536-byte readback, expected header,
     four descriptors, and payload spot checks.
5. Build and run `qmap_pl_compute_smoke_app` so the PL AXI master consumes the
   QMAP descriptors from real PL DDR4 and streams payloads into the existing Q4
   dot64 datapath.
   - status: passed on hardware with PL status `0xA` and expected result GPIO
     values.
6. Build and run `qmap_row1024_pl_compute_smoke_app` so the PL AXI master
   consumes the row1024 QMAP descriptors from real PL DDR4 and streams payloads
   into the full-row Q4 GEMV path.
   - status: passed on hardware with PL status `0xA` and result GPIO values
     `0xFFCA_DDC7` / `0xFFCA_DDC7`, representing `-3482169`.
7. Extend row1024 to a small multi-row tile, then validate simulation and board
   result words before scaling toward a larger `q_proj` block.
8. RMSNorm RTL block reads `input_hidden`, `norm_weight`, and `eps`; compare
   with `expected_output`.
9. Q4 GEMV RTL block starts from the 64-value dot-product smoke vector in
   `qwen3_0p6b_q4_v0/q_proj_row0_group0_dot64.npz`, then expands to full
   Layer 0 Q/K/V using `qkv_layer0_last_token_q4.npz`.
10. Extend vectors and memory regions for RoPE, KV cache, attention, MLP, and
   complete Layer 0.
11. Use FP32 vectors as golden references, but design GEMV/weight-storage RTL
   around the required Q4 path from the start. The current Q4 v0 artifact is
   the first packed-weight contract; extend it before scaling beyond Layer 0
   Q/K/V.

Pass/fail fields to record:

| Kernel | Vector | Max Abs Error | Mean Abs Error | Status |
| --- | --- | ---: | ---: | --- |
| PS-to-PL AXI BRAM | direct pattern test at `0xA000_0000` | exact match | exact match | passed |
| DDR4 status GPIO | direct read at `0xA001_0000` | exact expected status `0x5` | exact expected status `0x5` | passed |
| PS-to-PL PL DDR4 | direct pattern test at `0x4_0000_0000`, `0x4_0000_0004`, `0x4_0000_1000`, `0x4_1000_0000`, and `0x4_1FFF_FFFC` | exact match | exact match | passed |
| QMAP dot64 load/readback | `q_proj_row0_group0_dot64.qmap.bin` at `0x4_1B10_0000` | exact 1536-byte readback | exact header/descriptor/payload spot checks | passed |
| QMAP dot64 PL master compute | `q_proj_row0_group0_dot64.qmap.bin` at `0x4_1B10_0000` | exact PL status `0xA` | exact GPIO results `0x60AF` / `0x2E1366` | passed |
| QMAP row1024 PL master compute | `q_proj_row0_row1024.qmap.bin` at `0x4_1B20_0000` | exact PL status `0xA` | exact GPIO results `0xFFCA_DDC7` / `0xFFCA_DDC7` | passed |
| RMSNorm | `rmsnorm_layer0_last_token.npz` | TODO | TODO | TODO |
| Q4 Q GEMV | `qwen3_0p6b_q4_v0/qkv_layer0_last_token_q4.npz` | 0.22418976 | 0.01856172 | passed in Python verifier |
| Q4 K GEMV | `qwen3_0p6b_q4_v0/qkv_layer0_last_token_q4.npz` | 0.12317824 | 0.01752916 | passed in Python verifier |
| Q4 V GEMV | `qwen3_0p6b_q4_v0/qkv_layer0_last_token_q4.npz` | 0.07140590 | 0.01565306 | passed in Python verifier |

## Revision Log

| Date | Change | Notes |
| --- | --- | --- |
| 2026-05-24 | Initial skeleton | Adds PS DDR / PL DDR4 split and TODO sections |
| 2026-05-26 | Add confirmed AXI BRAM smoke-test map and DDR planning draft | Records PS DDR low range, BRAM `0x8000_0000`-`0x8000_1FFF`, and a relative 512 MiB PL DDR4 layout draft |
| 2026-05-27 | Record passing AXI BRAM hardware run | Confirms repeated 32-bit PS write/read access through `M_AXI_HPM0_LPD` at offsets `0x0`, `0x4`, `0x8`, `0x400`, and `0x1FFC` |
| 2026-05-28 | Strengthen Q4 as required PL weight path | Records that large PL-stored weights must use the project custom Q4 weight-only format; BF16/FP32 weights are reference data only |
| 2026-05-28 | Add Layer 0 Q/K/V Q4 v0 artifact contract | Uses signed int4 weights, group size 64, Q2.14 scales, Q4.12 activation test input, and Python-verified Q/K/V GEMV metrics |
| 2026-06-02 | Confirm q_norm/k_norm tensor shapes | Layer 0 `q_norm.weight` and `k_norm.weight` are both `[128]`; fixed-point gamma format remains open |
| 2026-06-02 | Add KV cache address formula RTL | Implements `kv_cache_addr_gen.sv` for `cache[layer][kv_kind][head][position][dim]`, with context-256 and context-512 smoke tests |
| 2026-06-02 | Align KV cache element stride with current fixed-point RTL | Changes the default KV cache address element stride to 4 bytes for signed 24-bit `Q12.12` K/V values padded to 32-bit DDR words |
| 2026-06-03 | Record MPSoC-P4 PL DDR4 board facts | Confirms real Bank 64 x16 PL DDR4 wiring and 512 MiB-class device evidence, while keeping physical base/range TODO until Vivado instantiation and hardware readback |
| 2026-06-03 | Confirm populated DDR4 marking | Physical marking `SEC 325` / `K4A4G16` / `BCTD` matches Samsung `K4A4G165WF-BCTD`; one x16 4Gb PL chip gives 0.5 GB and four PS x16 chips give 2 GB |
| 2026-06-04 | Assign PL DDR4 in Vivado block design | Adds DDR4 MIG AXI path through `M_AXI_HPM0_FPD`; BRAM moves to `0xA000_0000`, PL DDR4 maps to `0x4_0000_0000`-`0x4_1FFF_FFFF`; block-design validation passes, hardware readback pending |
| 2026-06-04 | Generate PL DDR4 bitstream and XSA | Synthesis, implementation, and bitstream generation pass for `llm_system_wrapper.bit`; routed timing meets constraints with WNS `0.572 ns`, route status reports `0` routing errors, and routed DRC has warnings but no errors; exported XSA is `FPGA_Project/Vivado_Project/llm_system_pl_ddr4_smoke.xsa`; Vitis platform/application and board readback remained pending at that time |
| 2026-06-07 | Add DDR4 readiness status GPIO | Adds PS-readable `0xA001_0000` AXI GPIO with `calib_complete`, `ui_reset`, and `axi_resetn` status bits; expected good status is `0x5` |
| 2026-06-07 | Prove PS-to-PL DDR4 hardware path | Final reset-fix XSA `llm_system_pl_ddr4_aux_reset_fix.xsa` and platform `llm_pl_ddr4_aux_reset_fix_platform` pass board smoke: BRAM at `0xA000_0000`, status GPIO at `0xA001_0000`, and PL DDR4 at `0x4_0000_0000` through `0x4_1FFF_FFFF` all read/write as expected |
| 2026-06-07 | Define QMAP v1 staging contract | Adds descriptor-based PL DDR4 tensor staging for the first dot64 Q4 image and future scalable model artifacts |
| 2026-06-07 | Prove QMAP dot64 PS load/readback | `qmap_load_smoke_app` writes the 1536-byte QMAP image to `0x4_1B10_0000`, reads it back exactly, and checks header, descriptor, and payload fields |
| 2026-06-11 | Prepare QMAP dot64 PL AXI master board smoke | Adds temporary control/status GPIO at `0xA002_0000`, result GPIO at `0xA003_0000`, generated/exported `llm_system_qmap_dot64_pl_master.xsa`, and prepared `qmap_pl_compute_smoke_app` to load QMAP, start PL compute, and check status/result |
| 2026-06-11 | Prove QMAP dot64 PL AXI master hardware path | `qmap_pl_compute_smoke_app` passes on board: DDR4 status `0x5`, QMAP readback passes for 1536 bytes, PL status `0xA`, partial sum `0x60AF`, scaled sum `0x2E1366` |
| 2026-06-23 | Prove QMAP row1024 PL AXI master hardware path | `qmap_row1024_pl_compute_smoke_app` passes on board from short Vitis workspace `F:\vws`: DDR4 status `0x5`, QMAP readback passes for 4096 bytes, header checks pass, PL status `0xA`, and both row result words are `0xFFCA_DDC7` / `-3482169` |
