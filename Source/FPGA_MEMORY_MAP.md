# FPGA Memory Map

Status: current physical apertures and full28 runtime layout, updated
2026-08-01. The latest hardware PASS remains the 2026-06-23 row1024 PL-master
smoke. The checked-in BD has since been replaced by the complete
resource-reduced Qwen accelerator and has generated a routed bitstream/fixed
XSA, but it has not yet received a full28 hardware PASS.

This document defines the FPGA-visible memory layout for Qwen3-0.6B bring-up.
It distinguishes three kinds of evidence: physical apertures already proven by
the row1024 board checkpoint, the address map in the current routed full28 BD,
and current runtime subregions that are locally/package audited but still await
their first physical full28 run.

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

Hardware and release checkpoints:

- Hardware-proven: PS access to DDR4 status/PL DDR4, QMAP dot64 PL reads and
  compute, and the QMAP row1024 PL read/full-row GEMV result `-3482169`.
- Current routed full28 candidate:
  `Temp/boardready_board_build_20260726/full_board_impl_v8_current/`.
- Current fixed XSA:
  `llm_system_qwen3_one_token_boardready.xsa` in that candidate directory.
- Current short-path Vitis components: `F:\vws\p_qot`,
  `F:\vws\a_qctl`, and `F:\vws\a_qmdl`.
- Current release state is `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED`: the
  persistent full28 XSim audit and self-contained package verification passed;
  the first physical full28 run remains pending.

## Current Address Map

This section is the source of truth for the current routed full28 Vivado block
design. Hardware status is stated separately so a routed aperture is not
mistaken for a physical-board PASS.

| Space / IP | Interface | Base | High | Size | Status |
| --- | --- | ---: | ---: | ---: | --- |
| PS DDR low memory | PS DDR | `0x0000_0000` | `0x7FEF_FFFF` | about 2 GiB minus reserved top window | current standalone BSP |
| DDR4 status AXI GPIO | PS `M_AXI_HPM0_FPD` -> SmartConnect -> AXI GPIO | `0xA001_0000` | `0xA001_FFFF` | 64 KiB | address current; status `0x5` hardware-proven |
| Qwen one-token control | PS `M_AXI_HPM0_FPD` -> SmartConnect -> `qmap_one_token_axi_bd/S_AXI` | `0xA004_0000` | `0xA004_FFFF` | 64 KiB | routed and exported; first hardware run pending |
| PL DDR4 | PS and accelerator masters -> SmartConnect -> AXI Clock Converter -> `ddr4_0/C0_DDR4_S_AXI` | `0x4_0000_0000` | `0x4_1FFF_FFFF` | 512 MiB | aperture hardware-proven; full28 contents/run pending |

Current fabric:

```text
PS M_AXI_HPM0_FPD
  -> AXI SmartConnect
      -> AXI GPIO DDR4 status register
      -> qmap_one_token_axi_bd/S_AXI
      -> AXI Clock Converter -> DDR4 MIG C0_DDR4_S_AXI

qmap_one_token_axi_bd/M_AXI
  -> AXI SmartConnect
      -> AXI Clock Converter -> DDR4 MIG C0_DDR4_S_AXI
```

Confirmed block-design and BSP facts:

- `M_AXI_HPM0_FPD` is the active PS master for the current PL memory fabric.
- The old AXI BRAM and row1024 GPIO apertures at `0xA000_0000`,
  `0xA002_0000`, and `0xA003_0000` are not present in the current BD.
- `pl_clk0` frequency in the exported handoff: about 96.97 MHz.
- The DDR4 AXI Clock Converter uses the PS/SmartConnect clock on its `S_AXI`
  side and `ddr4_0/c0_ddr4_ui_clk` on its `M_AXI` side.
- `ddr4_0/c0_ddr4_ui_clk_sync_rst` feeds the DDR UI-domain
  `proc_sys_reset_0/ext_reset_in`.
- `proc_sys_reset_0/aux_reset_in` is tied high because that input is
  active-low in the current IP configuration.
- `proc_sys_reset_0/peripheral_aresetn` drives both `ddr4_0/c0_ddr4_aresetn`
  and the clock converter `m_axi_aresetn`.
- The current BSP exports DDR4 status at `0xA001_0000`, accelerator control at
  `0xA004_0000`, and PL DDR4 at
  `0x4_0000_0000..0x4_1FFF_FFFF`.
- PL DDR4 is above the 32-bit address range. Bare-metal software must use a
  64-bit-capable address type such as `UINTPTR` for `0x4_0000_0000`.

Historical handoffs retained as evidence:

- `llm_system_qmap_row1024_pl_master.xsa`;
- `llm_system_qmap_dot64_pl_master.xsa`;
- `llm_system_pl_ddr4_aux_reset_fix.xsa`;
- `llm_system_axi_bram_smoke.xsa`.

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
- The current accelerator AXI datapath uses 32-bit data words aligned to
  4-byte boundaries and 64-bit physical addresses. The old BRAM smoke also used
  32-bit words.
- Endianness: little-endian for PS-side scalar accesses.
- Scalar format: unsigned 32-bit words unless a QMAP dtype specifies a signed
  fixed-point interpretation.
- Vector format: QMAP v1 for structured PL DDR4 tensor staging; keep exported
  Python vectors as golden references.
- Matrix layout: QMAP descriptors record logical shape and physical byte
  strides. Row-major contiguous Q4 groups remain the first GEMV bring-up
  layout from `Source/Q4_FORMAT.md`.
- The first full28 board release loads PL DDR through XSDB before the model ELF
  runs and uses volatile MMIO only for sentinels/control; it does not rely on a
  PS-cached PL-DDR staging buffer. A later in-application loader or DMA path must
  define explicit cache flush/invalidate ownership before use.
- The historical standalone smoke app validated direct 32-bit `Xil_Out32` /
  `Xil_In32` accesses to AXI BRAM at `0xA000_0000` through `0xA000_1FFF` and
  PL DDR4 at selected addresses in `0x4_0000_0000` through `0x4_1FFF_FFFF`.
- PL DDR4 addresses are 64-bit physical addresses. C code must use `UINTPTR`
  or another 64-bit-capable type and should print high/low 32-bit halves for
  debug output.
- Do not assume a future DMA or cached PS buffer path has the same coherency
  behavior as the current XSDB preload/volatile-control release path.

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

### Historical PL BRAM Smoke Test

Purpose: preserve the minimal PS-to-PL memory-mapped access evidence from the
row1024-era design. This BRAM is no longer present in the current full28 BD.

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

Current status: the 512 MiB physical aperture and PS/PL access have passed
hardware smoke tests. The current full28 BD validates, routes, generates a
bitstream/fixed XSA, and exports the same aperture to Vitis. Loading and
executing the complete full28 runtime on physical hardware is the next board
test, not an already-passed checkpoint.

Current physical base:

```text
PL_DDR4_BASE = 0x4_0000_0000
PL_DDR4_HIGH = 0x4_1FFF_FFFF
```

Current board-runtime facts:

| Item | Value | Evidence state |
| --- | ---: | --- |
| binary segments | `61` | package-audited |
| total segment bytes | `394,547,200` | package-audited |
| first loaded address | `0x4_0010_0000` | package-audited |
| last end-exclusive address | `0x4_1A14_0000` | package-audited |
| tied embedding / LM-head Q4 start | `0x4_0010_0000` | manifest and runtime image |
| per-layer persistent/QKV block start | `0x4_0500_0000` | 28 blocks, stride `0x0084_0000` |
| KV cache start | `0x4_1410_0000` | 28 layers, `0x0020_0000` bytes/layer |
| body-QMAP block start | `0x4_1790_0000` | 28 blocks, stride `0x0010_0000` |
| final-tail QMAP | `0x4_1950_0000` | runtime manifest |
| hidden ping-pong A/B | `0x4_1960_0000` / `0x4_1960_1000` | runtime manifest |
| runtime RoPE table start | `0x4_1A10_0000` | runtime manifest |
| QMAP packet count | `281` | 28 x 10 layer packets + final tail |
| zero-initialized mutable regions | `397` | release audit |

The 397 zero regions cover 14 output families per layer, 28 K/V-cache slices,
both hidden buffers, and final-tail norm/output scratch. The board candidate
must compute these values; it cannot pass by reading a preloaded golden output.

The table below is the earlier capacity-budget partition. Use the manifest facts
above for the current board image; keep this older partition only as a planning
cross-check.

| Region | Base | Size | Owner | Contents | Status |
| --- | ---: | ---: | --- | --- | --- |
| Header / memory-map metadata | `0x4_0000_0000` | 1 MiB | PS/PL | QMAP model manifest, runtime work packets, descriptor tables, version/checksum metadata | draft |
| Weight region | `0x4_0010_0000` | 320 MiB | PS load, PL read | required Q4 weights and scales | draft; LM-head plus Layer 0/Layer 1 `o_proj`, gate/up, and down local memory-reader layouts exercised from this region |
| KV cache region | `0x4_1410_0000` | 32 MiB | PL read/write | per-layer K/V cache, context 256 first | draft |
| Activation buffers | `0x4_1610_0000` | 64 MiB | PL read/write | one-token hidden, normed, Q/K/V, attention, MLP scratch, debug snapshots | draft |
| RoPE table | `0x4_1A10_0000` | 8 MiB | PS load or PL read | cos/sin table if precomputed | draft |
| Logits / argmax scratch | `0x4_1A90_0000` | 8 MiB | PL write, PS read | LM-head tile output and final token id | draft |
| Test-vector staging | `0x4_1B10_0000` | 32 MiB | PS load, PL read/write | QMAP smoke images, optional exported vectors, and debug staging | dot64 and row1024 QMAP images passed PS load/readback and PL master read/compute |
| Reserved | `0x4_1D10_0000` | 47 MiB | PS/PL | expansion room within nominal 512 MiB | draft |

The complete self-contained QKV regression packet is `0x0022_B000` bytes.
When based at the legacy default `0x4_0008_0000`, it extends through
`0x4_002A_AFFF` and therefore overlaps the persistent weight region beginning
at `0x4_0010_0000`. That layout is valid only for isolated QKV regression where
the packet owns the overlapping bytes. It must not coexist with the full tied
embedding/LM-head weights. The integrated token-driven frontend regression
places the self-contained packet at `0x4_1B40_0000` inside test-vector staging.
Deployment work packets must instead use non-overlapping descriptor targets in
the persistent weight and activation regions.

Current LM-head local memory-layout checkpoint:

```text
lm_head_weight_base       = 0x4_0010_0000
lm_head_weight_row_bytes  = 512
lm_head_weight_full_bytes = 0x04A3_0000
lm_head_weight_high       = 0x4_04B2_FFFF

lm_head_scale_base        = 0x4_04B3_0000
lm_head_scale_row_bytes   = 32
lm_head_scale_full_bytes  = 0x004A_3000
lm_head_scale_high        = 0x4_04FD_2FFF

o_proj_weight_base        = 0x4_0600_0000
o_proj_weight_row_bytes   = 1024
o_proj_weight_full_bytes  = 0x0010_0000
o_proj_weight_high        = 0x4_060F_FFFF

o_proj_scale_base         = 0x4_0610_0000
o_proj_scale_row_bytes    = 64
o_proj_scale_full_bytes   = 0x0001_0000
o_proj_scale_high         = 0x4_0610_FFFF

layer1_o_proj_weight_base = 0x4_0700_0000
layer1_o_proj_weight_high = 0x4_070F_FFFF
layer1_o_proj_scale_base  = 0x4_0710_0000
layer1_o_proj_scale_high  = 0x4_0710_FFFF

layer1_mlp_gate_weight_base = 0x4_0720_0000
layer1_mlp_gate_weight_high = 0x4_0737_FFFF
layer1_mlp_gate_scale_base  = 0x4_0738_0000
layer1_mlp_gate_scale_high  = 0x4_0739_7FFF
layer1_mlp_up_weight_base   = 0x4_0740_0000
layer1_mlp_up_weight_high   = 0x4_0757_FFFF
layer1_mlp_up_scale_base    = 0x4_0758_0000
layer1_mlp_up_scale_high    = 0x4_0759_7FFF
layer1_mlp_down_weight_base = 0x4_0760_0000
layer1_mlp_down_weight_high = 0x4_0777_FFFF
layer1_mlp_down_scale_base  = 0x4_0778_0000
layer1_mlp_down_scale_high  = 0x4_0779_7FFF

mlp_gate_weight_base      = 0x4_0620_0000
mlp_gate_weight_row_bytes = 512
mlp_gate_weight_full_bytes = 0x0018_0000
mlp_gate_weight_high      = 0x4_0637_FFFF

mlp_gate_scale_base       = 0x4_0638_0000
mlp_gate_scale_row_bytes  = 32
mlp_gate_scale_full_bytes = 0x0001_8000
mlp_gate_scale_high       = 0x4_0639_7FFF

mlp_up_weight_base        = 0x4_0640_0000
mlp_up_weight_row_bytes   = 512
mlp_up_weight_full_bytes  = 0x0018_0000
mlp_up_weight_high        = 0x4_0657_FFFF

mlp_up_scale_base         = 0x4_0658_0000
mlp_up_scale_row_bytes    = 32
mlp_up_scale_full_bytes   = 0x0001_8000
mlp_up_scale_high         = 0x4_0659_7FFF

mlp_down_weight_base      = 0x4_0660_0000
mlp_down_weight_row_bytes = 1536
mlp_down_weight_full_bytes = 0x0018_0000
mlp_down_weight_high      = 0x4_0677_FFFF

mlp_down_scale_base       = 0x4_0678_0000
mlp_down_scale_row_bytes  = 96
mlp_down_scale_full_bytes = 0x0001_8000
mlp_down_scale_high       = 0x4_0679_7FFF
```

The compact local memory-model tests load the `[0,1024)` scan window from
exported hex files. The full-vocabulary xsim runs load the complete
151,936-row table through the same base addresses, so the reader path already
uses the intended full-vocabulary physical layout.

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

Formal QMAP inference layout draft:

- Persistent model manifest base: `0x4_0000_0000`
- Legacy isolated QKV regression packet base: `0x4_0008_0000`
- Integrated token-driven frontend QKV staging base: `0x4_1B40_0000`
- QMAP LM-head runtime packet base: `0x4_0500_0000`
- QMAP final-token tail runtime packet base: `0x4_0501_0000`
- QMAP attention front-end runtime packet base: `0x4_0502_0000`
- QMAP Layer 1 attention front-end runtime packet base: `0x4_1502_0000`
- QMAP attention score/value runtime packet base: `0x4_0503_0000`
- QMAP Layer 1 attention score/value runtime packet base: `0x4_1503_0000`
- QMAP `o_proj` runtime packet base: `0x4_0504_0000`
- QMAP Layer 1 `o_proj` runtime packet base: `0x4_1504_0000`
- QMAP post-attention residual/RMSNorm runtime packet base: `0x4_0505_0000`
- QMAP Layer 1 post-attention residual/RMSNorm runtime packet base:
  `0x4_1505_0000`
- QMAP MLP gate/up runtime packet base: `0x4_0506_0000`
- QMAP Layer 1 MLP gate/up runtime packet base: `0x4_1506_0000`
- QMAP MLP SiLU/multiply runtime packet base: `0x4_0507_0000`
- QMAP Layer 1 MLP SiLU/multiply runtime packet base: `0x4_1507_0000`
- QMAP MLP down runtime packet base: `0x4_0508_0000`
- QMAP Layer 1 MLP down runtime packet base: `0x4_1508_0000`
- QMAP final MLP residual runtime packet base: `0x4_0509_0000`
- QMAP Layer 1 final MLP residual runtime packet base: `0x4_1509_0000`
- QMAP input RMSNorm runtime packet base: `0x4_050A_0000`
- QMAP Layer 1 input RMSNorm runtime packet base: `0x4_150A_0000`
- QMAP Layer 2 input RMSNorm runtime packet base: `0x4_250A_0000`
- Current Layer 2 final layer-output buffer automatically selected by the
  top final-tail handoff: `0x4_2509_2540`
- Bring-up smoke images remain in the test-vector staging region.
- The manifest describes tensors loaded once into weight, KV-cache,
  activation, RoPE, and logits regions.
- Runtime work packets describe the specific tensors consumed and produced by
  one PL kernel step. The pre-QKV input RMSNorm packet now produces
  descriptor-visible `input_norm[1024]`; the first large Q4 GEMV packet remains
  Layer 0 QKV projection, and the same QKV packet contract has now also been
  proven with true Layer 1 data.
- Work packets should point to persistent tensor regions by descriptor
  `base_addr`; they should not duplicate large Q4 weights inside each packet.
- The current `21_export_qmap_qkv_projection_image.py` exporter can emit a
  self-contained packet for simulation and bring-up. The default full generated
  packet uses the runtime work-packet base `0x4_0008_0000`, 12 active
  descriptors, 32 descriptor slots, and covers full Layer 0 Q/K/V row counts.
  Its `0x0022_B000`-byte extent overlaps the persistent weight region, so this
  default is a legacy isolated-regression layout, not a concurrent deployment
  layout. The continuous embedding-to-QKV regression overrides the runtime
  packet base to non-overlapping staging address `0x4_1B40_0000`; the RTL QMAP
  descriptor contract is unchanged.
  With `--layer-id 1` and `--qmap-base 0x4_1008_0000`, the same packet
  contract now covers true Layer 1 Q/K/V row counts and passes local AXI
  write-back comparison.
- The current `48_export_qmap_input_rmsnorm_image.py` exporter emits the
  pre-QKV input RMSNorm packet at `0x4_050A_0000`: `hidden[1024]`,
  signed `I16_Q8_7` input-layernorm gamma, `input_norm[1024]` output scratch,
  and expected debug data. The local wrapper proves exact `input_norm[1024]`
  write-back and a bad-gamma-dtype no-write error path in the current
  signed-gamma Icarus recheck. Layer 1 and Layer 2 chained input RMSNorm packets are generated at
  `0x4_150A_0000` and `0x4_250A_0000`; their QKV packets read
  `0x4_150A_2540` and `0x4_250A_2540` through generated activation descriptors.
  The Layer 0 compute scheduler can now optionally run this packet before QKV;
  its focused precheck uses QKV expected words regenerated from the RTL RMSNorm
  output. The same nonzero input-RMSNorm base table also reaches the reusable
  one-token top wrapper in local simulation.
- `51_export_embedding_layer0_full_chain.py` now regenerates the complete
  token-374 Layer 0 lineage under `Temp`: tied embedding at
  `0x4_0509_2540`, input RMSNorm output/QKV activation at `0x4_050A_2540`,
  staged self-contained QKV at `0x4_1B40_0000`, and the existing Layer 0 body
  packets at `0x4_0502_0000` through `0x4_0509_0000`. The chained packet
  descriptors are patched in the shared-memory testbench so every consumer
  reads the preceding producer's write-back buffer. The final residual writes
  `layer_out[1024]` back to `0x4_0509_2540`, safely reusing the input-hidden
  buffer only after all residual consumers have finished.
- The current `37_export_qmap_attention_frontend_image.py` exporter emits the
  next per-layer body packet at `0x4_0502_0000`: Q/K/V projection outputs,
  q/k gamma, RoPE cos/sin, a KV-cache descriptor, and Q RoPE output scratch.
  The local wrapper proves exact K/V cache writes and exact Q RoPE write-back.
  With Layer 1 q/k+RoPE and KV-cache vector prefixes, the same packet contract
  now also passes at `0x4_1502_0000`.
- The current `38_export_qmap_attention_score_value_image.py` exporter emits
  the next attention-body packet at `0x4_0503_0000`: Q RoPE input, K/V cache,
  softmax exp LUT, and `attn_out[2048]` output scratch. The local wrapper
  proves exact K/V cache reads and exact attention-output write-back. With
  Layer 1 score/value vector prefixes, the same packet contract now also passes
  at `0x4_1503_0000`.
- The current `39_export_qmap_o_proj_image.py` exporter emits the next
  per-layer body packet at `0x4_0504_0000`: `attn_out[2048]`, persistent
  `o_proj` Q4 weight/scale references, and `o_proj_out[1024]` output scratch.
  The local wrapper proves exact persistent weight/scale row reads and exact
  output write-back. With Layer 1 `o_proj` vector prefixes plus
  `--qmap-base 0x4_1504_0000`, `--weight-base 0x4_0700_0000`, and
  `--scale-base 0x4_0710_0000`, the same packet contract now also passes with
  true Layer 1 data.
- The current `40_export_qmap_post_attention_residual_norm_image.py` exporter
  emits the next per-layer body packet at `0x4_0505_0000`: residual input,
  `o_proj_out[1024]`, signed post-attention RMSNorm gamma,
  post-attention hidden output scratch, and post-norm output scratch. The
  local wrapper proves exact hidden and post-norm write-back. With Layer 1
  vector prefixes and `--qmap-base 0x4_1505_0000`, the same packet contract now
  also passes with true Layer 1 residual input, Layer 1 `o_proj_out[1024]`, and
  Layer 1 post-attention gamma.
- The current `41_export_qmap_mlp_gate_up_image.py` exporter emits the next
  per-layer body packet at `0x4_0506_0000`: `post_norm[1024]`, persistent
  Layer 0 gate/up Q4 weight/scale references, and gate/up output scratch. The
  local wrapper proves exact persistent gate/up weight/scale row reads and
  exact gate/up write-back. With Layer 1 vector prefixes,
  `--qmap-base 0x4_1506_0000`, and persistent Layer 1 gate/up bases
  `0x4_0720_0000`, `0x4_0738_0000`, `0x4_0740_0000`, and `0x4_0758_0000`,
  the same packet contract now also passes with true Layer 1 post-norm input
  and true Layer 1 MLP gate/up weights.
- The current `42_export_qmap_mlp_silu_mul_image.py` exporter emits the next
  per-layer body packet at `0x4_0507_0000`: gate/up `[3072]`, a fixed UQ0.16
  sigmoid LUT, hidden output scratch, and expected hidden debug data. The local
  wrapper proves exact LUT/gate/up reads and exact `mlp_hidden[3072]`
  write-back. With `--layer-id 1`, `--qmap-base 0x4_1507_0000`, and
  `--gate-up-qmap-prefix qmap_layer1_mlp_gate_up`, the same packet contract now
  also passes with true Layer 1 gate/up inputs.
- The current `43_export_qmap_mlp_down_image.py` exporter emits the next
  per-layer body packet at `0x4_0508_0000`: `mlp_hidden[3072]`, persistent
  Layer 0 down-proj Q4 weight/scale references, down output scratch, and
  expected down debug data. The local wrapper proves exact persistent down
  weight/scale row reads and exact `down_out[1024]` write-back. With Layer 1
  vector prefixes, `--qmap-base 0x4_1508_0000`, and persistent Layer 1 down
  bases `0x4_0760_0000` and `0x4_0778_0000`, the same packet contract now also
  passes with true Layer 1 `mlp_hidden[3072]` input and true Layer 1 MLP down
  weights.
- The current `44_export_qmap_mlp_residual_add_image.py` exporter emits the
  next per-layer body packet at `0x4_0509_0000`:
  `post_attn_hidden[1024]`, `down_out[1024]`, layer-output scratch, and
  expected layer-output debug data. The local wrapper proves exact
  `layer_out[1024]` write-back and descriptor/protocol error paths with no
  writes. With Layer 1 vector prefixes, `--qmap-base 0x4_1509_0000`,
  `--post-attn-qmap-prefix qmap_layer1_post_attention_residual_norm`, and
  `--down-qmap-prefix qmap_layer1_mlp_down`, the same packet contract now also
  passes with true Layer 1 post-attention hidden and true Layer 1 down output.
- `qmap_layer0_body_scheduler.sv` is not a new runtime packet. It locally
  composes the existing post-attention residual/RMSNorm, MLP gate/up,
  MLP SiLU/multiply, MLP down, and final MLP residual packets behind one memory
  request/write interface. Its testbench patches downstream descriptor
  `base_addr` fields so the next stage reads the previous stage's actual
  write-back buffer.
- `qmap_layer0_full_scheduler.sv` is also not a new runtime packet. It locally
  composes the attention front-end, attention score/value, `o_proj`, and
  Layer 0 body scheduler behind one memory request/write interface. Its
  testbench patches chained descriptor `base_addr` fields so the score/value,
  `o_proj`, post-attention, MLP, and final residual consumers read the actual
  upstream write-back buffers.
- `qmap_layer0_compute_scheduler.sv` is also not a new runtime packet. It
  locally composes the full QKV projection packet with the Layer 0 full
  scheduler behind one memory request/write interface. Its testbench patches
  the attention front-end Q/K/V descriptor `base_addr` fields to the Q/K/V
  output buffers written by the QKV projection stage before starting the
  downstream Layer 0 scheduler.
- `qmap_one_token_layer_scheduler.sv` is the first local layer-loop boundary.
  It is not a new runtime packet either. It exposes layer index/count,
  hidden-buffer bases, KV-cache base, token position, per-layer QMAP packet
  base tables, done/error masks, and one shared memory interface. The first
  supported local compute case is `layer_start_index=0`, `layer_count=1`, and
  the first local loop-control expansion is a two-layer alias run where layer 1
  table entries intentionally point at the current Layer 0 packets. Missing
  selected/ranged table entries and out-of-range layer-loop requests exit with
  error and no memory traffic.

The first Layer 0 QKV projection packet should reserve 32 descriptor slots and
describe:

```text
input_norm[1024] read from activation buffers
q_proj weight/scale read from weight region
k_proj weight/scale read from weight region
v_proj weight/scale read from weight region
q_out[2048] written to activation buffers
k_out[1024] written to activation buffers
v_out[1024] written to activation buffers
optional debug spot-check tensor
```

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
| embed_tokens / lm_head | `[151936, 1024]` | packed signed Q4 weight + unsigned Q2.14 group64 scales | weights `0x4_0010_0000`, scales `0x4_04B3_0000` | weights `0x04A3_0000` B, scales `0x004A_3000` B | tied row-major matrix; one 512-byte weight row and one 32-byte scale row per token |
| layer N input RMSNorm weight | `[1024]` | signed `I16_Q8_7` for current QMAP runtime packets | TODO | 2048 B per layer if tightly packed, currently padded to 4096 B in debug packets | Layer 1 has negative input-layernorm gamma entries; unsigned gamma is not deployable |
| layer N q_proj | `[2048, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N k_proj | `[1024, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N v_proj | `[1024, 1024]` | TODO | TODO | TODO | row-major candidate |
| layer N q_norm weight | `[128]` | signed `I16_Q8_7` for current Layer 0 RTL | TODO | 256 B per layer | Layer 0 has negative q_norm gamma entries; full-model format remains subject to range review |
| layer N k_norm weight | `[128]` | signed `I16_Q8_7` for current Layer 0 RTL | TODO | 256 B per layer | Current Layer 0 max fits; full-model format remains subject to range review |
| layer N o_proj | `[1024, 2048]` | packed signed Q4 weight + Q2.14 scales | `0x4_0600_0000` / `0x4_0610_0000` for current Layer 0 sim window; `0x4_0700_0000` / `0x4_0710_0000` for current Layer 1 sim window | 0x0010_0000 weight + 0x0001_0000 scales | row-major, one 1024-byte weight row and one 64-byte scale row per output row |
| layer N post-attn RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |
| layer N gate_proj | `[3072, 1024]` | packed signed Q4 weight + Q2.14 scales | `0x4_0620_0000` / `0x4_0638_0000` for current Layer 0 sim window; `0x4_0720_0000` / `0x4_0738_0000` for current Layer 1 sim window | 0x0018_0000 weight + 0x0001_8000 scales | MLP gate, one 512-byte weight row and one 32-byte scale row per output row |
| layer N up_proj | `[3072, 1024]` | packed signed Q4 weight + Q2.14 scales | `0x4_0640_0000` / `0x4_0658_0000` for current Layer 0 sim window; `0x4_0740_0000` / `0x4_0758_0000` for current Layer 1 sim window | 0x0018_0000 weight + 0x0001_8000 scales | MLP up, one 512-byte weight row and one 32-byte scale row per output row |
| layer N down_proj | `[1024, 3072]` | packed signed Q4 weight + Q2.14 scales | `0x4_0660_0000` / `0x4_0678_0000` for current Layer 0 sim window; `0x4_0760_0000` / `0x4_0778_0000` for current Layer 1 sim window | 0x0018_0000 weight + 0x0001_8000 scales | MLP down, one 1536-byte weight row and one 96-byte scale row per output row |
| final RMSNorm weight | `[1024]` | TODO | TODO | TODO | gamma |

Open decisions:

- Embedding and LM head share the same row-major signed-Q4/group64/Q2.14
  weight/scale contract. The local exporter verifies exact tied-row equality;
  full-matrix artifact packaging and deployment tolerance remain open.
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
| Cache dtype | signed 24-bit `Q12.12` padded to 32-bit DDR words for the current fixed-point RTL path | local RTL append passed |
| Layer stride | `2 * 8 * max_context * 128 * bytes_per_value` | draft |
| K/V stride | `8 * max_context * 128 * bytes_per_value` | draft |
| Head stride | `max_context * 128 * bytes_per_value` | draft |
| Position stride | `128 * bytes_per_value` | draft |
| Head-dim stride | `bytes_per_value`; current fixed-point default is 4 bytes | draft |
| Append address formula | `base + layer*layer_stride + kv_kind*kv_stride + head*head_stride + position*position_stride + dim*bytes_per_value` | local RTL append passed |
| Read address formula | same formula, sweeping `position` from 0 through current cache length minus 1 | initial RTL implemented |

The first RTL address generator is:

```text
FPGA_Project/rtl/model/attention/kv_cache_addr_gen.sv
```

It generates byte addresses and range-valid flags. The first append controller
is:

```text
FPGA_Project/rtl/model/attention/kv_cache_append.sv
```

`kv_cache_append.sv` writes K first and V second for one token position, pads
signed 24-bit `Q12.12` elements into 32-bit DDR words, and keeps its write
stream stable under backpressure. AXI burst planning and the future read
controller still belong in later memory-facing wrappers.

## Activation Buffers

Logical first-version one-token buffers. These are the tensors that make the
model path reusable across RTL blocks. Exact byte offsets inside the activation
region can be fixed when the first QKV projection exporter is written.

| Buffer | Shape | Format | Location | Reuse Rule | Status |
| --- | --- | --- | --- | --- | --- |
| hidden_a | `[1024]` | `I32_Q14_10` or stage-specific `I32_Q12_12` | activation region | ping-pong layer input/output | draft |
| hidden_b | `[1024]` | `I32_Q14_10` or stage-specific `I32_Q12_12` | activation region | ping-pong layer input/output | draft |
| input_norm | `[1024]` | `I32_Q12_12` | activation region | RMSNorm output, Q/K/V GEMV input | next packet input |
| q_flat | `[2048]` | `I32_Q12_12` | activation region | Q projection output, q_norm input | next packet output |
| k_flat | `[1024]` | `I32_Q12_12` | activation region | K projection output, k_norm input | next packet output |
| v_flat | `[1024]` | `I32_Q12_12` | activation region | V projection output, V-cache append source | next packet output |
| q_normed | `[16, 128]` | `I32_Q12_12` | activation region or on-chip | RoPE input | local RTL stage passed |
| k_normed | `[8, 128]` | `I32_Q12_12` | activation region or on-chip | RoPE input | local RTL stage passed |
| q_rope | `[16, 128]` | `I32_Q12_12` | activation region or on-chip | attention input | local RTL stage passed |
| k_rope | `[8, 128]` | `I32_Q12_12` | activation region or direct cache write | K-cache append source | local RTL append passed |
| v_state | `[8, 128]` | `I32_Q12_12` | activation region or direct cache write | V-cache append source | local RTL append passed |
| attn_out | `[2048]` | `I32_Q12_12` | activation region | `o_proj` input | local RTL stage passed |
| o_proj_out | `[1024]` | `I32_Q12_12` | activation region | attention residual add input | local RTL stage passed |
| post_attn_hidden | `[1024]` | `I32_Q14_10` | activation region | post-attention residual result | local RTL stage passed |
| post_norm | `[1024]` | `I32_Q12_12` | activation region | MLP input | local RTL stage passed |
| gate | `[3072]` | `I32_Q12_12` | activation region or streamed | MLP intermediate | local RTL stage passed |
| up | `[3072]` | `I32_Q12_12` | activation region or streamed | MLP intermediate | local RTL stage passed |
| mlp_hidden | `[3072]` | `I32_Q12_12` | activation region or streamed | `silu(gate) * up`, down-projection input | local RTL stage passed |
| down_out | `[1024]` | `I32_Q12_12` | activation region or streamed | final MLP residual input | QMAP MLP down wrapper now writes exact output locally |
| layer_out | `[1024]` | `I32_Q14_10` | activation region | `post_attn_hidden + down_out`, next layer input | QMAP final MLP residual wrapper now writes exact output locally |
| final_norm | `[1024]` | `I32_Q12_12` | activation or logits region | LM-head input | final-token tail QMAP wrapper writes and rereads this buffer locally |
| logits_tile | tile-dependent | implementation-defined | logits/argmax region | LM-head tile scan | full-vocab local QMAP wrapper passed |
| output_token | `[1]` | `U32` | logits/argmax region or control register | greedy argmax result | final-token tail QMAP wrapper writes token/score locally |

Open decisions:

- Which buffers must stay on-chip for performance after correctness is proven?
- Which debug snapshots should be kept in PL DDR4 for board bring-up?
- Exact activation-region byte offsets and alignment:
- Exact conversion policy between residual formats and Q/K/V `I32_Q12_12`:

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

The vocabulary size is 151,936. A full logits vector can be large, so the
current hardware plan scans the tied LM head in 16-row tiles and keeps only the
running argmax. The descriptor-backed local full-vocab proof passes with
`9496` tiles, token `264`, score `1365150750`, and exact Q26 logit checks.
The first QMAP final-token tail wrapper now also proves the preceding
`final_hidden -> final_norm` write-back before that full-vocab scan.

TODO:

- Full logits stored or tiled scan only:
- Logit dtype:
- Tile size:
- Reduction method:
- Argmax output address:
- Tie-breaking rule:

## Control Register Map

The current local one-token control contract lives in
`qmap_one_token_control_regs.sv`. It is exposed directly by
`qmap_one_token_mmio_top.sv` through a tiny valid/ready register port and now
also through `qmap_one_token_axil_top.sv` via the generic
`axi4lite_to_mmio_regs.sv` AXI4-Lite slave adapter. The PS-side mirror of this
contract now lives in `FPGA_Project/software/qmap_one_token_runtime/`:
`qmap_one_token_regs.h` keeps the offsets/table ids/status masks,
`qmap_one_token_runtime.h` keeps configure/start/poll/result helpers, and
`main.c` builds as either the no-memory validation smoke or the full28
persistent two-token model smoke. The BD-facing RTL is
`qmap_one_token_axi_bd.v` over `qmap_one_token_axi_top.sv`; the checked-in BD
assigns and routes this register map at `0xA004_0000` / 64 KiB and exposes its
PL-DDR traffic through `M_AXI`.

Historical row1024-era smoke registers, no longer present in the current BD:

| Historical Smoke-Test Item | Address | Width | Access | Description |
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

Current one-token register offsets use byte addresses inside a 12-bit register
space. The physical PS address is `0xA004_0000 + offset`:

| Register | Offset | Width | Access | Description |
| --- | ---: | ---: | --- | --- |
| control | `0x000` | 32 | W | `bit0=start` pulse when not busy; `bit1=clear` done/error/command sticky bits |
| status | `0x004` | 32 | R | `bit0=busy`, `bit1=done_sticky`, `bit2=error_sticky`, `bit3=command_error_sticky`, `bit4=top_error`, `bit5=tail_error`, `bit6=tail_norm_saturation`, `bits15:8=top_state`, `bits23:16=top_phase` |
| layer_start | `0x008` | 32 | R/W | first scheduler layer index |
| layer_count | `0x00C` | 32 | R/W | number of layers to run; validation rejects zero/out-of-range requests before memory traffic |
| position | `0x010` | 32 | R/W | current sequence position |
| input_token | `0x014` | 32 | R/W | token id; when embedding is enabled it selects the tied Q4 row used to generate `input_hidden[1024]` |
| input_hidden_base | `0x020/0x024` | 64 | R/W | input hidden buffer base address |
| output_hidden_base | `0x028/0x02C` | 64 | R/W | layer output buffer base address |
| kv_cache_base | `0x030/0x034` | 64 | R/W | KV-cache base address |
| final_tail_qmap_base | `0x038/0x03C` | 64 | R/W | final-token tail QMAP packet base |
| final_hidden_override | `0x040/0x044`, `0x048` | 64 + 1 | R/W | optional debug override base and valid bit; normal top runs use scheduler-reported last-layer output |
| embedding_enable | `0x04C` | 1 useful bit | R/W | `bit0=1` runs tied-Q4 embedding before the layer scheduler; busy writes are rejected through command sticky status |
| table_select | `0x050` | 32 | R/W | `{layer[15:8], table_id[7:0]}` for per-layer QMAP base table updates |
| table_data | `0x054/0x058` | 64 | R/W | shadow QMAP base address for selected table/layer |
| table_commit | `0x05C` | 32 | W | commit selected table/layer when `bit0=1`; invalid table/layer or busy commit sets command/error sticky |
| output_token | `0x060` | 32 | R | greedy argmax token id from final tail |
| output_score | `0x064/0x068` | 64 | R | sign-extended final-tail best score Q26 |
| layers | `0x06C` | 32 | R | `{layers_completed[15:0], layers_started[15:0]}` |
| layer_done_mask | `0x070` | 32 | R | per-layer done mask |
| layer_error_mask | `0x074` | 32 | R | per-layer validation/error mask |
| last_output_base | `0x078/0x07C` | 64 | R | scheduler-reported last layer output base |
| tail_hidden_base | `0x080/0x084` | 64 | R | final-tail effective hidden input base |
| tail_tiles_started/completed | `0x088/0x08C` | 32 each | R | final LM-head tile counters |
| mem_read_reqs/words | `0x090/0x094` | 32 each | R | aggregate top memory read counters |
| mem_write_reqs/words | `0x098/0x09C` | 32 each | R | aggregate top memory write counters |
| embedding_weight_base | `0x0A0/0x0A4` | 64 | R/W | tied packed-Q4 matrix base; token row address is base plus `token_id*512` |
| embedding_scale_base | `0x0A8/0x0AC` | 64 | R/W | tied Q2.14 scale matrix base; token row address is base plus `token_id*32` |

Software source-of-truth notes:

- `FPGA_Project/software/qmap_one_token_runtime/qmap_one_token_regs.h` mirrors the
  table above and should be updated in lockstep with
  `qmap_one_token_control_regs.sv`.
- `qmap_one_token_runtime.h` writes 64-bit addresses as low word then high word,
  commits QMAP base tables through `table_select/table_data/table_commit`, and
  treats `input_norm` table entries as optional so the current Layer0 QKV-first
  plus Layer1/2 input-RMSNorm mixed contract is representable. It also validates
  and writes optional embedding enable/weight/scale fields; the no-memory smoke
  explicitly disables embedding.
- `qmap_one_token_runtime/main.c` is currently a no-memory AXI-Lite seam smoke:
  it starts `layer_count=0`, expects done+error and layer0 error, and requires all
  memory counters to remain zero before clearing sticky status.
- `qmap_one_token_axi_top.sv` is the first Vivado-facing shell around the same
  contract. Its `S_AXI` aperture should be mapped separately from the old smoke
  GPIOs, while its `M_AXI` master should target the existing PL-DDR aperture at
  `0x4_0000_0000` for first bring-up.

## Data Movement Sequence

TODO: Confirm the actual boot/runtime flow.

Candidate first-version flow:

1. PS boots and initializes runtime.
2. PS loads model/artifact metadata from storage into PS DDR.
3. PS copies required FPGA artifacts into PL DDR4.
4. PS writes memory-map/control registers.
5. PS sends prompt tokens one at a time.
6. PL reads the tied Q4 embedding row for `input_token`, writes Q14.10
   `input_hidden[1024]`, and runs `run_one_token(input_token, position)`.
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
- The first token-driven PL stage now passes local Icarus and Vivado XSim.
  `q4_embedding_lookup.sv` reads one 512-byte tied Q4 row plus one 32-byte scale
  row and writes exact Q14.10 `input_hidden[1024]`. The AXI-Lite and BD-facing
  tests prove software register launch, two read bursts, one 4096-byte local
  write, and five legal physical AXI write bursts. This is local RTL evidence,
  not a new board result.
- Decide later whether DMA is needed for faster bulk copies.
- Future cache coherency remains open only for a later in-application loader,
  DMA, or cached PS-buffer path. The current board release uses XSDB preload.
- The current Windows board flow has a fixed 61-segment loader and verified Q4
  full-model runtime. A portable Linux/in-application bulk loader is future
  work.

## Current Board Validation Plan

Closed pre-board gates: the full28 persistent two-token XSim, independent
timing/KV-retention audit, and 92-file package inventory/size/SHA256/readme
verification all passed. The physical plan is now:

1. Run the packaged AXI-Lite no-memory control smoke and require
   `PASS qot_run_no_memory_validation_smoke`.
2. Run the packaged full28 model smoke and require the exact position-0 and
   position-1 token/score UART lines recorded in
   `FPGA_Project/Vivado_Project/ONE_TOKEN_AXI_TOP_BD_PLAN.md`.
3. Preserve the UART log. If a gate fails, debug that exact interface/address/
   stage. Do not replace this plan with another small row test unless the
   failure specifically points there.

## Historical Validation Plan and Gap Closure

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
7. Define and export the first formal Layer 0 QKV projection QMAP work packet.
   It should reference persistent Q/K/V Q4 weights/scales, read
   `input_norm[1024]`, and write Q/K/V output buffers in PL DDR4.
   - status: passed locally in Python exporter. The full packet covers
     Q/K/V rows `2048/1024/1024`; the compact simulation packet covers
     `4/2/2` rows with the same descriptor contract.
8. Add PL AXI write-back support so Q/K/V projection results can become inputs
   to q_norm/k_norm, RoPE, KV-cache append, and attention.
   - status: local RTL passed. `qmap_qkv_projection_compute_path.sv` writes
     Q/K/V output buffers through the project-local write stream and matches
     Python I32_Q12.12 expected words; `axi4_write_master.sv` passes a focused
     AXI4 write-burst simulation.
9. Wrap the QKV projection path into a Vivado-facing AXI read/write top and
   prove compact QKV packet write-back from real PL DDR4 on hardware.
   - status: local RTL integration passed through full Q/K/V scale.
     `qmap_qkv_projection_axi_smoke_top.sv` connects the QKV compute path to
     AXI read/write masters and the parameterized Icarus memory-model test
     passes compact, medium, larger, and full packets. Hardware confirmation is
     still pending.
10. Add local q/k norm + RoPE integration after QKV projection.
    - status: local RTL passed. `qk_norm_rope_stage_128.sv` consumes Q/K
      projection outputs from the same Q4/QMAP fixed-point contract, applies
      signed `Q8.7` q_norm/k_norm gamma over all 24 heads, runs RoPE, and
      matches all Q norm, K norm, Q RoPE, and K RoPE golden words with
      `max_abs_diff=0`.
11. Add local K/V cache append after q/k norm and RoPE.
    - status: local RTL passed. `kv_cache_append.sv` writes exact K then V cache
      address/data streams from fixed golden vectors, and
      `qk_norm_rope_kv_cache_stage.sv` combines q/k norm, RoPE, and K/V cache
      append with exact Q/K RoPE matches plus `2048` accepted cache writes.
12. Add local attention-score generation from current Q and cached K.
    - status: local RTL passed. `attention_score_stage.sv` uses a K-cache
      request/response interface, applies 16-Q-head to 8-KV-head GQA mapping,
      and emits exact raw and scaled scores for the 5-position prompt cache.
      The test accepts `10240` K requests/responses and `80` score outputs under
      request stalls, variable response latency, and score-stream stalls.
13. Add local softmax/value accumulation from attention scores and cached V.
    - status: local RTL passed. `attention_softmax_value_stage.sv` consumes the
      `80` scaled score stream, uses a 257-entry exp LUT to produce UQ0.16
      probabilities, reads V cache through a request/response interface, and
      emits exact `attn_out[16,128]` Q12.12 words. The test accepts `10240`
      V requests/responses and `2048` outputs under input gaps, request stalls,
      response latency, and output stalls.
14. Add local attention output projection from `attn_out[2048]`.
    - status: local RTL passed. `o_proj_stage.sv` consumes the fixed
      softmax/value output vector, reuses the Q4 GEMV projection controller
      with `INPUT_SIZE=2048`, and emits exact Q12.12 `o_proj_out[1024]` words
      under output-stream backpressure.
15. Add local post-attention residual add and post-attention RMSNorm hookup.
    - status: local RTL passed. `post_attention_residual_norm_stage.sv`
      consumes fixed `o_proj_out[1024]`, adds it to the Layer 0 residual input
      in signed Q14.10, applies signed Q8.7 post-attention RMSNorm gamma, and
      emits exact Q12.12 post-norm words.
16. Add local MLP gate/up projections from post-attention RMSNorm output.
    - status: local RTL passed. `mlp_gate_up_proj_stage.sv` consumes fixed
      `post_norm[1024]`, runs Layer 0 `gate_proj[3072]` and `up_proj[3072]`
      through the same Q4 GEMV projection controller, and emits exact Q12.12
      gate/up pairs under output-stream backpressure.
17. Add local SiLU plus elementwise multiply from the passing gate/up outputs.
    - status: local RTL passed. `mlp_silu_mul_stage.sv` consumes the passing
      gate/up Q12.12 pairs, uses a UQ0.16 sigmoid LUT over `[-8, 8]` with
      `1/64` spacing, and emits exact Q12.12
      `mlp_hidden[3072] = silu(gate[3072]) * up[3072]` words under input and
      output backpressure.
18. Add local MLP down projection from `mlp_hidden[3072]`.
    - status: local RTL passed. `mlp_down_proj_stage.sv` consumes fixed
      `mlp_hidden[3072]`, runs Layer 0 `down_proj[1024]` through the same Q4
      GEMV projection controller with `INPUT_SIZE=3072`, and emits exact
      Q12.12 `down_out[1024]` words under output-stream backpressure.
19. Add local final MLP residual from `post_attn_hidden[1024]` and
    `down_out[1024]`.
    - status: local RTL passed. `mlp_residual_add_stage.sv` wraps the existing
      sequential residual add block and computes
      `post_attn_hidden[1024] + down_out[1024] -> layer_out[1024]` exactly
      against Q14.10 golden words.
    - QMAP status: local wrapper passed. `qmap_mlp_residual_add_compute_path.sv`
      consumes descriptor-visible `post_attn_hidden[1024]` and
      `down_out[1024]`, writes exact Q14.10 `layer_out[1024]`, and covers both
      invalid descriptor and malformed payload-`last` error paths with no
      writes.
20. Add local final RMSNorm.
    - status: local RTL passed. `final_rmsnorm_stage.sv` consumes the
      full-model Python reference's current-token final hidden vector, applies
      signed Q8.7 final norm gamma, and emits exact Q12.12 final-norm words
      across two runs with trace-audited control timing.
21. Add local tied LM-head scan and greedy argmax from `final_norm[1024]`.
    - status: local tiled RTL, local memory-backed wrapper, and runtime tile
      scheduler pass. `lm_head_argmax_stage.sv` consumes the passing final
      RMSNorm vector, requests 16-row Q4 weight/scale tiles, checks all
      scan-window logits exactly, and returns token `264` with score
      `1365150750`. `lm_head_tile_mem_reader.sv` and
      `lm_head_argmax_mem_stage.sv` fetch the same tiles through a
      project-local memory request/response interface using exported
      weight/scale base addresses. `lm_head_argmax_tile_scheduler.sv` then
      reuses the memory-backed stage across runtime scan counts, with default
      capacity for `9496` tiles / `151936` vocabulary rows. The QMAP wrapper
      `qmap_lm_head_argmax_compute_path.sv` now reads descriptor-provided
      final-norm, weight, scale, scan-range, output, and expected/debug tensors,
      runs the scheduler, and writes token/score output words. The Python
      exporter also proves token `264` is the full-Q4/HF argmax. The full
      `9496`-tile vocabulary simulation now passes in Vivado xsim with exact
      token `264`, score `1365150750`, and `151936` checked logits.
22. Compose the first QMAP final-token tail wrapper.
    - status: local RTL/xsim passed. `qmap_final_token_tail_compute_path.sv`
      reads descriptor-provided final hidden and signed final RMSNorm gamma,
      writes `final_norm[1024]` to the descriptor activation buffer, invokes
      the descriptor-backed LM-head wrapper, and writes final token/score
      words. Compact Icarus covers the invalid gamma descriptor path; full
      Vivado xsim covers `9496` tiles / `151936` logits with exact token `264`
      and score `1365150750`.
23. Compose the first QMAP attention front-end wrapper.
    - status: local RTL passed. `qmap_attention_frontend_compute_path.sv` reads
      descriptor-provided Q/K/V projection outputs, q/k gamma, and RoPE cos/sin
      tables, runs q/k norm + RoPE + KV-cache append, writes exact K/V cache
      address/data words, and writes exact `q_rope[2048]` to the descriptor
      output buffer. The testbench covers read/write backpressure, spurious
      start while busy, invalid RoPE-cos dtype error handling, and all wrapper
      states in the trace audit.
24. Compose the first QMAP attention score/value wrapper.
    - status: local RTL passed. `qmap_attention_score_value_compute_path.sv`
      reads descriptor-provided Q RoPE and exp LUT payloads, converts K/V cache
      requests into exact 4-byte memory reads, streams score into
      softmax/value, captures `attn_out[2048]`, and writes it to the descriptor
      output buffer. The testbench covers read/write backpressure, exact K/V
      read order, exact output write data, invalid exp-LUT dtype error
      handling, and all wrapper states in the trace audit.
25. Compose the first local QMAP full Layer 0 scheduler.
    - status: local RTL passed. `qmap_layer0_full_scheduler.sv` chains the
      existing attention front-end, attention score/value, `o_proj`, and
      Layer 0 body scheduler behind one memory interface.
      `tb_qmap_layer0_full_scheduler.sv` patches chained descriptor
      `base_addr` fields so producer write-back buffers feed downstream
      consumers, then checks exact K/V cache, Q RoPE, attention, `o_proj`,
      MLP, and final `layer_out[1024]` writes. The traced run covers
      `38055` read requests, `1579650` read-response words, `2058` write
      requests, `20480` write-data words, and an invalid first-stage
      descriptor path that exits with no writes.
26. Compose the first local QMAP Layer 0 compute scheduler.
    - status: local RTL passed. `qmap_layer0_compute_scheduler.sv` chains the
      full QKV projection packet into the passing full Layer 0 scheduler behind
      one memory interface. `tb_qmap_layer0_compute_scheduler.sv` patches the
      attention front-end Q/K/V input descriptors to the QKV-produced output
      buffers, then checks exact Q/K/V, K/V cache, Q RoPE, attention, `o_proj`,
      MLP, and final `layer_out[1024]` writes. The traced run covers `46264`
      read requests, `2138130` read-response words, `6154` write requests,
      `24576` write-data words, a QKV-stage no-write invalid descriptor path,
      and a later attention front-end invalid descriptor path after QKV has
      completed.
27. Compose the first local one-token/layer-loop scheduler boundary.
    - status: local RTL passed. `qmap_one_token_layer_scheduler.sv` wraps the
      passing Layer 0 compute scheduler with an explicit layer-loop contract:
      layer index/count, position, input/output hidden-buffer bases, KV-cache
      base, per-layer QMAP packet base tables, shared memory ownership, layer
      done/error masks, and aggregate counters. The first testbench-supported
      case is `layer_start_index=0`, `layer_count=1`; it preserves the exact
      QKV-through-`layer_out[1024]` write-back and the normal `46264`/`2138130`
      read plus `6154`/`24576` write counts. A two-layer alias case
      (`layer_count=2`, layer 1 table entries aliasing the Layer 0 packets)
      also passes with layer done mask `0x3` and aggregate `92528`/`4276260`
      reads plus `12308`/`49152` writes. A missing selected Layer 0 QKV
      base-table entry, a missing Layer 1 table entry, and out-of-range
      `layer_start_index=28` requests exit with error and no memory traffic.
28. Add true Layer 1 QKV vector/export/AXI coverage.
    - status: local Python export and RTL passed. `45_export_layer_qkv_q4_vectors.py`
      emits `qkv_layer1_last_token_q4.npz`; `21_export_qmap_qkv_projection_image.py`
      now accepts `--layer-id`, and the AXI QKV wrapper takes runtime
      `i_qmap_base_addr`. Compact and full Layer 1 QKV packets at
      `0x4_1008_0000` match Python expected I32_Q12.12 output words. The full
      run covers `2048/1024/1024` rows, `4096` writes, `8209` read bursts, and
      status `0xA`.
29. Q4 GEMV RTL block starts from the 64-value dot-product smoke vector in
    `qwen3_0p6b_q4_v0/q_proj_row0_group0_dot64.npz`, then expands to full
    Layer 0 Q/K/V using `qkv_layer0_last_token_q4.npz`.
30. Extend vectors and memory regions for RoPE, KV cache, attention, MLP, and
    complete Layer 0.
31. Use FP32 vectors as golden references, but design GEMV/weight-storage RTL
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
| QMAP Layer 0 QKV packet export | `layer0_qkv_projection.qmap.bin` at `0x4_0008_0000` | Q/K/V Q26 recompute diff `0.0` | full row coverage `2048/1024/1024` | passed locally |
| QMAP QKV projection AXI top write-back | full packet `q_rows=2048`, `k_rows=1024`, `v_rows=1024` | exact I32_Q12.12 output word match | 4096 AXI writes, 8209 AXI read bursts, status `0xA` | passed in Icarus |
| QMAP Layer 1 QKV packet export/write-back | `qkv_layer1_last_token_q4.npz`, compact/full QMAP packets at `0x4_1008_0000` | exact I32_Q12.12 output word match after Q4 recompute | full row coverage `2048/1024/1024`, 4096 AXI writes, 8209 AXI read bursts, status `0xA` | passed in Icarus |
| Q/K norm + RoPE stage | `qk_norm_rope_stage_128_real` from Q4/QMAP fixed Q/K projection words | exact Q norm, K norm, Q RoPE, K RoPE word match | 24 heads, 10612 cycles, no saturation | passed in Icarus |
| KV cache append | `kv_cache_append_real` from Q4/QMAP fixed K/V projection words | exact cache address/data/kind/head/dim word match | 2048 writes, 652 stall cycles, trace audit passed | passed in Icarus |
| Q/K norm + RoPE + KV cache append stage | combined `qk_norm_rope_stage_128_real` and `kv_cache_append_real` vectors | exact Q RoPE, K RoPE, and cache write-stream match | 2048 cache writes, 887 stall cycles, first cache valid cycle 10614 | passed in Icarus |
| Attention score stage | `attention_score_stage_real` from Q4/QMAP fixed Q and 5-position K cache | exact raw Q24.24 and scaled Q24.24 score match | 10240 K requests/responses, 80 scores, trace audit passed | passed in Icarus |
| Attention softmax/value stage | `attention_softmax_value_stage_real` from fixed scores and 5-position V cache | exact UQ0.16 probabilities and Q12.12 `attn_out` words | 10240 V requests/responses, 2048 outputs, trace audit passed | passed in Icarus |
| Attention o_proj stage | `o_proj_stage_real` from fixed `attn_out[2048]` and Layer 0 Q4 `o_proj.weight[1024,2048]` | exact Q12.12 `o_proj_out[1024]` word match | 1024 outputs, 618 output stall cycles, 35073 compute cycles | passed in Icarus |
| QMAP attention o_proj wrapper | `qmap_o_proj_runtime.qmap.bin` plus persistent Layer 0 `o_proj` Q4 weight/scale vectors | exact Q12.12 `o_proj_out[1024]` write-back | 1024 weight-row reads, 1024 scale-row reads, 2070 read requests including invalid descriptor run, one 4096-byte output write burst, trace audit passed | passed in Icarus |
| QMAP Layer 1 attention o_proj wrapper | `layer1_o_proj_runtime.qmap.bin` at `0x4_1504_0000` plus persistent Layer 1 `o_proj` Q4 weight/scale bases `0x4_0700_0000` / `0x4_0710_0000` | exact Q12.12 `o_proj_out[1024]` write-back | 1024 weight-row reads, 1024 scale-row reads, 2070 read requests including invalid descriptor run, one 4096-byte output write burst, trace audit passed | passed in Icarus |
| Post-attention residual + RMSNorm stage | `post_attention_residual_norm_stage_real` from fixed `o_proj_out[1024]` and Layer 0 residual input | exact Q14.10 residual and Q12.12 post-norm word match | 1024 residual words, 3132 stage cycles, trace audit passed | passed in Icarus |
| QMAP post-attention residual + RMSNorm wrapper | `post_attention_residual_norm_runtime.qmap.bin` with residual input, `o_proj_out[1024]`, and signed post-attention gamma descriptors | exact Q14.10 post-attention hidden and exact Q12.12 post-norm write-back | 30 read requests including invalid descriptor run, 3616 response words, two 4096-byte output write bursts, trace audit passed | passed in Icarus |
| QMAP Layer 1 post-attention residual + RMSNorm wrapper | `layer1_post_attention_residual_norm_runtime.qmap.bin` at `0x4_1505_0000` with Layer 1 residual input, `o_proj_out[1024]`, and signed post-attention gamma descriptors | exact Q14.10 post-attention hidden and exact Q12.12 post-norm write-back | 30 read requests including invalid descriptor run, 3616 response words, two 4096-byte output write bursts, `sum_squares=75359102`, `inv_rms=247634`, trace audit passed | passed in Icarus |
| QMAP MLP gate/up wrapper | `mlp_gate_up_runtime.qmap.bin` plus persistent Layer 0 gate/up Q4 weight/scale vectors | exact Q12.12 `gate[3072]` and `up[3072]` write-back | 3072 row completions, 12314 read requests including invalid descriptor run, 837280 response words, two 12288-byte output write bursts, trace audit passed | passed in Icarus |
| QMAP Layer 1 MLP gate/up wrapper | `layer1_mlp_gate_up_runtime.qmap.bin` at `0x4_1506_0000` plus persistent Layer 1 gate/up Q4 weight/scale bases `0x4_0720_0000`, `0x4_0738_0000`, `0x4_0740_0000`, and `0x4_0758_0000` | exact Q12.12 `gate[3072]` and `up[3072]` write-back | 3072 row completions, 12314 read requests including invalid descriptor run, 837280 response words, two 12288-byte output write bursts, zero unknown reads, invalid descriptor no-write path, trace audit passed | passed in Icarus |
| QMAP Layer 1 MLP SiLU/multiply wrapper | `layer1_mlp_silu_mul_runtime.qmap.bin` at `0x4_1507_0000` with true Layer 1 gate/up inputs and sigmoid LUT | exact Q12.12 `mlp_hidden[3072]` write-back | 3072 stage inputs, 3072 stage outputs, 36 normal read requests, 7377 normal response words, one 12288-byte output write burst at `0x4_1507_7980`, zero unknown reads, invalid descriptor no-write path, trace audit passed | passed in Icarus |
| QMAP Layer 1 MLP down wrapper | `layer1_mlp_down_runtime.qmap.bin` at `0x4_1508_0000` plus persistent Layer 1 down-proj Q4 weight/scale bases `0x4_0760_0000` and `0x4_0778_0000` | exact Q12.12 `down_out[1024]` write-back | 1024 row completions, 3098 read requests including invalid descriptor run, 421280 response words, one 4096-byte output write burst at `0x4_1508_3940`, zero unknown reads, invalid descriptor no-write path, trace audit passed | passed in Icarus |
| QMAP Layer 1 final MLP residual wrapper | `layer1_mlp_residual_add_runtime.qmap.bin` at `0x4_1509_0000` with true Layer 1 post-attention hidden and true Layer 1 `down_out[1024]` inputs | exact Q14.10 `layer_out[1024]` write-back | normal: 14 read requests, 2224 response words, one 4096-byte output write at `0x4_1509_2540`; full error test: 27 read requests, 2577 response words, bad descriptor and bad payload-last no-write paths, trace audit passed | passed in Icarus |
| MLP gate/up projection stage | `mlp_gate_up_proj_stage_real` from fixed `post_norm[1024]` and Layer 0 Q4 gate/up weights | exact Q12.12 gate and up word match | 3072 output pairs, 1394 output stall cycles, 52993 compute cycles, trace audit passed | passed in Icarus |
| MLP SiLU/multiply stage | `mlp_silu_mul_stage_real` from fixed gate/up Q12.12 outputs and sigmoid LUT | exact Q12.12 `mlp_hidden[3072]` word match | 3072 inputs, 3072 outputs, 747 input stall cycles, 961 output stall cycles, trace audit passed | passed in Icarus |
| QMAP MLP down wrapper | `mlp_down_runtime.qmap.bin` plus persistent Layer 0 down-proj Q4 weight/scale vectors | exact Q12.12 `down_out[1024]` write-back | 1024 row completions, 3098 read requests including invalid descriptor run, 421280 response words, one 4096-byte output write burst, trace audit passed | passed in Icarus |
| MLP down projection stage | `mlp_down_proj_stage_real` from fixed `mlp_hidden[3072]` and Layer 0 Q4 `down_proj.weight[1024,3072]` | exact Q12.12 `down_out[1024]` word match | 1024 outputs, 497 output stall cycles, 52481 compute cycles, trace audit passed | passed in Icarus |
| QMAP final MLP residual wrapper | `mlp_residual_add_runtime.qmap.bin` with `post_attn_hidden[1024]`, `down_out[1024]`, and layer-output scratch | exact Q14.10 `layer_out[1024]` write-back | valid write-back plus invalid descriptor and malformed payload-`last` no-write error paths, 27 read requests, 2577 response words, one 4096-byte output write burst, trace audit passed | passed in Icarus |
| QMAP input RMSNorm wrapper | `input_rmsnorm_runtime.qmap.bin` with `hidden[1024]`, signed `I16_Q8_7` input-layernorm gamma, and input-norm scratch | exact Q12.12 `input_norm[1024]` write-back | 14 read requests, 2224 response words, one 4096-byte output write, bad-gamma-dtype no-write error path, trace audit passed | passed in Icarus |
| QMAP Layer 0 input-RMSNorm-to-QKV precheck | `qmap_layer0_compute_scheduler.sv` with `input_rmsnorm_runtime.qmap.bin` at `0x4_050A_0000`, QKV activation patched to `0x4_050A_2540`, and QKV golden regenerated with `--activation-hex` | exact `input_norm[1024]` plus exact Q/K/V `2048/1024/1024` write-back | `+qkv_precheck +fastmem +notrace`: cycle `2288643`, `8234` read requests, `561040` response words, `4097` write requests, `5120` write-data words, frontend invalid descriptor stops downstream work, zero mismatches | passed in Icarus |
| QMAP one-token input-RMSNorm-to-QKV precheck | `qmap_one_token_layer_scheduler.sv` with nonzero layer-0 `i_input_norm_qmap_base_addr_table`, QKV activation patched to `0x4_050A_2540`, and chained QKV golden from RTL RMSNorm output | exact `input_norm[1024]` plus exact Q/K/V `2048/1024/1024` write-back through the reusable layer-loop contract | `+input_norm_qkv_only +fastmem`: cycle `2288647`, `8234` read requests, `561040` response words, `4097` write requests, `5120` write-data words; trace shows RMSNorm write-last at cycle `7623` and QKV activation reads at `8487/9000/9513/10026`, with zero reads before write completion | passed in Icarus |
| QMAP one-token top input-RMSNorm-to-QKV precheck | `qmap_one_token_top.sv` passes `i_input_norm_qmap_base_addr_table` into the scheduler; TB is compiled with `QMAP_ONE_TOKEN_TB_USE_TOP` and `QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL`, and `+input_norm_qkv_only` loads the generated QKV-from-input image whose activation descriptor points at `0x4_050A_2540` | exact `input_norm[1024]` plus exact Q/K/V `2048/1024/1024` write-back through the top wrapper; final tail does not launch in this focused frontend-boundary precheck | `+input_norm_qkv_only +fastmem`: cycle `2288651`, `8234` read requests, `561040` response words, `4097` write requests, `5120` write-data words; top trace shows RMSNorm write-last at cycle `7625` and QKV activation reads at `8489/9002/9515/10028`, with zero reads before write completion | passed in Icarus |
| QMAP one-token Layer1/Layer2 input-RMSNorm-to-QKV prechecks | `tb_qmap_one_token_layer_scheduler.sv` shared-memory model loads generated input RMSNorm packets at `0x4_150A_0000` and `0x4_250A_0000`; Layer 1/2 QKV activation descriptors point at `0x4_150A_2540` and `0x4_250A_2540` | exact per-layer `input_norm[1024]` plus exact Q/K/V `2048/1024/1024` write-back through the reusable layer-loop contract | `+l1_input_norm_only +fastmem +notrace` and `+l2_input_norm_only +fastmem +notrace`: cycle `2288647`, `8234` read requests, `561040` response words, `4097` write requests, `5120` write-data words; Layer 2 trace shows write-last at cycle `7623` and QKV activation reads at `8487/9000/9513/10026`, with zero reads before write completion | passed in Icarus |
| QMAP one-token Layer1/Layer2 input-RMSNorm full-layer scheduler | `+l1_only` and `+l2_only` now enable the per-layer input RMSNorm table before QKV and run the complete QKV -> attention -> MLP -> residual layer body through the reusable scheduler | exact `input_norm[1024]`, Q/K/V `2048/1024/1024`, all downstream stage write-backs, and final `layer_out[1024]` | both runs: cycle `6700589`, `46278` read requests, `2140354` response words, `6155` write requests, `25600` write-data words; masks `0x2` / `0x4`; Layer 2 full trace confirms zero early reads from input_norm and Q/K/V producer buffers | passed in Icarus |
| QMAP one-token top Layer2 input-RMSNorm-to-final-tail handoff | `qmap_one_token_top.sv` sequences the Layer 2 input-RMSNorm-enabled scheduler into `qmap_final_token_tail_compute_path.sv` through one shared memory port; by default the tail hidden input is selected from the scheduler-reported last layer output `0x4_2509_2540` while the final-tail packet remains at `0x4_0501_0000` | exact Layer 2 `layer_out[1024]` consumption, exact final RMSNorm write-back, and exact full-vocab LM-head output token `537` / score `850086863` | Vivado xsim `+l2_top_tail_only +fastmem +progress`: scheduler done cycle `6700589`, tail done `50800994`, top pulses `1/1/1/1`, top counters `131772/22807266` reads and `6157/26627` writes, tail read counts `4/4/4/75968/9496`, tail writes `1024/3`; trace has `307615` data rows and confirms hidden read `6701175`, norm write done `6708411`, norm read `6709001`, LM-head read `6711056`, output write request/data `50800983..50800986`, top done `50800996` | passed in xsim |
| QMAP one-token productized MMIO wrapper no-memory validation | `qmap_one_token_mmio_top.sv` instantiates `qmap_one_token_control_regs.sv` plus `qmap_one_token_top.sv` behind one register port and one memory port | register-launched top/scheduler validation exit, sticky status/layer-error/counter readback, and zero memory traffic | Icarus `tb_qmap_one_token_mmio_top.sv`: writes scalar registers, starts invalid `layer_count=0`, observes top error and layer0 error mask, reads zero memory counters, and clears done/command sticky status | passed in Icarus |
| AXI4-Lite to tiny-MMIO adapter | `axi4lite_to_mmio_regs.sv` serializes a single 32-bit AXI4-Lite slave transaction into the tiny register port used by `qmap_one_token_control_regs.sv` | AW-before-W, W-before-AW, B/R stalls, register ready stalls, downstream errors, unaligned/partial write rejects, and write-priority over simultaneous reads | Icarus `tb_axi4lite_to_mmio_regs.sv`: returns `OKAY` or `SLVERR` as expected, holds valid/data under stalls, and avoids tiny-MMIO side effects for local rejects | passed in Icarus |
| QMAP one-token AXI-Lite wrapper no-memory validation | `qmap_one_token_axil_top.sv` wraps `axi4lite_to_mmio_regs.sv` around `qmap_one_token_mmio_top.sv` | AXI-Lite-launched top/scheduler validation exit, sticky status/layer-error/counter readback, and zero memory traffic | Icarus `tb_qmap_one_token_axil_top.sv`: writes scalar registers through AXI-Lite, starts invalid `layer_count=0`, observes top error and layer0 error mask, reads zero memory counters, and preserves clear behavior; `tb_qmap_one_token_mmio_top.sv` was rerun afterward as a regression | passed in Icarus |
| QMAP one-token BD-facing AXI shell no-memory validation | `qmap_one_token_axi_top.sv` wraps `qmap_one_token_axil_top.sv` with `axi4_read_master.sv`/`axi4_write_master.sv` to expose Vivado-facing `S_AXI` and `M_AXI` | same register-launched validation exit through the BD-facing shell, with no AXI4 memory-master traffic | Icarus `tb_qmap_one_token_axi_top.sv`: starts invalid `layer_count=0` through `S_AXI`, observes done/error/layer0-error register readback and zero memory counters, and checks that no `M_AXI_ARVALID`, `M_AXI_AWVALID`, `M_AXI_WVALID`, `M_AXI_RREADY`, or `M_AXI_BREADY` traffic is issued | passed in Icarus |
| Tied-Q4 embedding lookup | `q4_embedding_lookup.sv` plus vectors exported directly from local `embed_tokens/lm_head` row equality | exact signed Q14.10 `input_hidden[1024]` from one 512-byte signed-Q4 row and one 32-byte Q2.14 scale row | Icarus `tb_q4_embedding_lookup.sv`: exact 1024-word output, counters `2/136` reads and `1/1024` writes, request/data stalls, invalid token, malformed read-last, write-error, and repeated-start coverage | passed in Icarus |
| AXI-Lite token-to-embedding composition | `qmap_one_token_axil_top.sv` configured through new embedding registers | AXI-Lite token/weight/scale/hidden configuration launches embedding before scheduler validation and preserves sticky status | exact 1024-word write-back, counters `2/136` reads and `1/1024` writes, embedding write completion before scheduler validation and top done | passed in Icarus |
| BD-facing token-to-embedding AXI path | `qmap_one_token_axi_top.sv` with real AXI4 read/write adapters | exact tied-Q4 output through physical AXI bursts, including a 4096-byte local hidden write larger than one AXI burst | two read bursts; five write bursts of `256/256/176/256/80` beats respecting 256-beat and 4 KiB limits; exact 1024 words; fresh Icarus and Vivado 2024.2 XSim compile/elaboration/run in `Temp/q4_embedding_regression/20260713_124011` | passed in Icarus and xsim |
| Continuous tied-Q4 embedding -> Layer 0 input RMSNorm -> full QKV | token `374`, persistent tied weight/scale bases `0x4_0010_0000` / `0x4_04B3_0000`, hidden output `0x4_0509_2540`, RMSNorm output/QKV activation `0x4_050A_2540`, and non-overlapping self-contained QKV staging base `0x4_1B40_0000` | exact embedding `1024`, input-norm `1024`, and Q/K/V `2048/1024/1024` words under read/write backpressure; producer responses precede every consumer read | fresh export, Vivado 2024.2 XSim, and independent CSV timing audit in `Temp/embedding_layer0_frontend_regression/20260713_131628`; scheduler reads `8234/561040`, writes `4097/5120`; aggregate reads `8236/561176`, writes `4098/6144`; final QKV write/scheduler/top done cycles `1857932/1858412/1858414` | passed in xsim |
| Continuous tied-Q4 embedding -> complete Layer 0 | Temp-contained token-374 artifacts from embedding/input RMSNorm/full QKV through q/k norm + RoPE, KV append, attention score/value, `o_proj`, post-attention residual/RMSNorm, gate/up, SiLU/multiply, down, and final residual | exact comparison of all 16 write kinds and every output word; exact cache write order; trace-audited producer response before each consumer read | `Temp/embedding_layer0_full_regression/20260713_135629`: scheduler reads `46278/2140354`, writes `6155/25600`; aggregate reads `46289/2140762`, writes `6156/26624`; final layer write/scheduler/tail-start/top cycles `5091425/5091436/5091437/5091822`. The tail gamma descriptor is intentionally invalid so the focused run stops after the successful layer with zero tail writes | passed in xsim |
| QMAP one-token AXI-Lite wrapper Layer1 -> Layer2 top-to-tail launch | `qmap_one_token_axil_top.sv` launches the same Layer1 -> Layer2 -> final-tail path through AXI-Lite register writes/reads into the productized wrapper | exact Layer 1 and Layer 2 scheduler write-back, automatic Layer 2 output selection for final tail, exact token `537` / score `850086863`, exact status/counter readback, and aggregate counters matching the MMIO baseline | Vivado xsim snapshot `qmap_one_token_axil_l1_l2_tail_xsim` with `+l1_l2_mmio_top_tail_only +fastmem +notrace +progress`: scheduler done cycle `13402176`, tail done `57502581`, first hidden/LM-head/output cycles `13402762`/`13412643`/`57502570`, layer mask `0x6`, scheduler counters `92556/4280708` reads and `12310/51200` writes, top counters `178050/24947620` reads and `12312/52227` writes, tail reads `4/4/4/75968/9496`, tail writes `1024/3`, `0` final write mismatches; log `FPGA_Project/sim/xsim_qmap_one_token_axil_l1_l2_tail_run.log` | passed in xsim |
| Historical QMAP one-token AXI-Lite wrapper mixed true3 top-to-tail launch | `qmap_one_token_axil_top.sv` launches Layer0(QKV-first) -> Layer1 -> Layer2 -> final-tail through AXI-Lite register writes/reads; Layer1/2 use input-RMSNorm-enabled artifacts while Layer0 remains on the existing QKV-first full-layer chain | exact three-layer scheduler write-back across the historical mixed artifact contract, automatic Layer 2 output selection for final tail, exact token `537` / score `850086863`, layer mask `0x7`, and exact status/counter readback | Vivado xsim snapshot `qmap_one_token_axil_true3_tail_xsim` with `+true3_mmio_top_tail_only +fastmem +notrace +progress`: scheduler done cycle `20095284`, tail done `64195689`, first hidden/LM-head/output cycles `20095870`/`20105751`/`64195678`, scheduler counters `138820/6418838` reads and `18464/75776` writes, top counters `224314/27085750` reads and `18466/76803` writes, tail reads `4/4/4/75968/9496`, tail writes `1024/3`, `0` final write mismatches; retained only as a historical baseline | passed in xsim |
| Continuous tied-Q4 embedding -> complete Layer 0 -> Layer 1 -> Layer 2 -> full-vocabulary tail through AXI-Lite | token `374` launches tied-Q4 embedding, then three complete input-RMSNorm-enabled transformer layers from newly chained outputs, followed by final RMSNorm and the full `151936`-row tied LM-head scan; Layer 0 uses non-overlapping QKV staging at `0x4_1B40_0000` | every embedding/layer/tail write word, address, burst count, response, producer-before-consumer handoff, final token `537`, and final Q26 score `838805253` match the independently generated true3 artifacts | `Temp/embedding_true3_axil_tail_regression/20260713_145056`: scheduler reads `138834/6421062`, writes `18465/76800`; aggregate reads `224330/27088110`, writes `18468/78851`; embedding response cycle `2540`, Layer 0/1/2 final responses `6703116/13403696/20104276`, scheduler/tail/top completion cycles `20104287/20104288/64204694`; independent CSV audit pairs all `224330` reads and `18468` writes with zero mismatches | passed in xsim |
| Continuous tied-Q4 embedding -> all 28 layers -> full-vocabulary tail through AXI-Lite | token `374`; collision-free 28-layer layout with persistent layer block base/stride `0x4_0500_0000`/`0x0084_0000`, body-QMAP base/stride `0x4_1790_0000`/`0x0010_0000`, KV-cache base/stride `0x4_1410_0000`/`0x0020_0000`, and final-tail QMAP `0x4_1950_0000` | exact comparison of every embedding/layer/tail write; all `281` QMAP packets and `508` non-overlapping physical intervals covered; exact token `537`, score `1155032971`, layer mask `0x0fffffff`, and response-before-consumer ordering | `Temp/embedding_full28_axil_tail_regression/20260713_164001`: scheduler reads `1295784/59929912`, writes `172340/716800`; aggregate reads `1381280/80596960`, writes `172343/718851`; embedding response `11290`, first/last layer responses `6711866/187627526`, scheduler/tail/first-tail-read/top cycles `187627537/187627538/187628123/231727944`; streamed event audit pairs every request/completion/response with zero mismatches | passed in xsim |
| QMAP one-token MMIO-style Layer1 -> Layer2 top-to-tail launch | `qmap_one_token_control_regs.sv` drives `qmap_one_token_top.sv` through software-like register writes/table commits in `tb_qmap_one_token_layer_scheduler.sv +l1_l2_mmio_top_tail_only`; this proves the local register contract beneath the AXI-Lite wrapper | exact Layer 1 and Layer 2 scheduler write-back, automatic Layer 2 output selection for final tail, exact token `537` / score `850086863`, exact status/counter readback | Vivado xsim `+l1_l2_mmio_top_tail_only +fastmem +notrace +progress`: scheduler done cycle `13401430`, tail done `57501835`, layer mask `0x6`, top counters `178050/24947620` reads and `12312/52227` writes, tail hidden/last output `0x4_2509_2540`, tail writes `1024/3`, `0` final write mismatches | passed in xsim |
| Final MLP residual add stage | `mlp_residual_add_stage_real` from fixed `post_attn_hidden[1024]` and `down_out[1024]` | exact Q14.10 `layer_out[1024]` word match | two full runs, 1024 outputs, 1026 stage cycles, spurious start covered, trace audit passed | passed in Icarus |
| QMAP Layer 0 body scheduler | Existing QMAP post-attention residual/RMSNorm, MLP gate/up, MLP SiLU/multiply, MLP down, and final MLP residual packets with patched chained buffer descriptors | exact intermediate write-backs through Q14.10 `layer_out[1024]` | normal: 15465 read requests, 1270961 response words, seven output write bursts, 13312 write-data words; invalid first-stage descriptor path exits with no writes; trace audit passed | passed in Icarus |
| QMAP Layer 0 full scheduler | Existing QMAP attention front-end, attention score/value, `o_proj`, and Layer 0 body scheduler packets with patched chained buffer descriptors | exact write-backs from K/V cache and Q RoPE through Q14.10 `layer_out[1024]` | normal: 38055 read requests, 1579650 response words, 2058 write requests, 20480 write-data words; invalid first-stage descriptor path exits with no writes; trace audit passed | passed in Icarus |
| QMAP Layer 0 compute scheduler | Full QKV projection packet plus QMAP Layer 0 full scheduler packets with patched frontend Q/K/V and downstream chained buffer descriptors | exact write-backs from Q/K/V projection through Q14.10 `layer_out[1024]` | normal: 46264 read requests, 2138130 response words, 6154 write requests, 24576 write-data words; QKV invalid descriptor path exits with no writes; frontend invalid descriptor path exits after QKV writes only; trace audit confirms a 731-cycle QKV producer-to-consumer gap | passed in Icarus |
| QMAP one-token layer scheduler | `qmap_layer0_compute_scheduler.sv` wrapped by an explicit layer-loop contract with layer index/count, hidden-buffer bases, KV-cache base, token position, per-layer QMAP packet base tables, and shared memory ownership | exact QKV-through-`layer_out[1024]` write-back for the supported Layer 0 loop case and exact repeated write-back for the two-layer alias loop | normal: one layer started/completed, layer done mask `0x1`, same 46264/2138130 read and 6154/24576 write counts; two-layer alias: two layers started/completed, layer done mask `0x3`, 92528/4276260 read and 12308/49152 write counts; QKV/frontend errors propagate through layer masks; missing base-table entries and out-of-range layer index requests exit with 0 read/write traffic; trace audit passed | passed in Icarus |
| Final RMSNorm stage | `final_rmsnorm_stage_real` from full-model current-token final hidden and signed Q8.7 final gamma | exact Q12.12 final-norm word match plus exact debug scalars | two full runs, 2106 stage cycles, sum/sqrt/div/apply coverage, trace audit passed | passed in Icarus |
| LM-head scan + greedy argmax stage | `lm_head_argmax_stage_real` from final RMSNorm output and tied Q4 LM-head rows `[0,1024)` | exact Q26 logit word match and best token `264` / score `1365150750` | two full runs, 128 tile requests, 128 tile responses, 2048 logits checked, trace audit passed | passed in Icarus |
| LM-head memory-backed scan-window wrapper | `lm_head_argmax_stage_real` plus exported LM-head weight/scale memory words for rows `[0,1024)` | exact Q26 logit word match and best token `264` / score `1365150750` | two full runs, 1152 read requests, 278528 response words, 2048 logits checked, trace audit passed | passed in Icarus |
| LM-head runtime tile scheduler | `lm_head_argmax_stage_real` plus exported LM-head weight/scale memory words and runtime tile counts | exact Q26 logit word match and best token `264` / score `1365150750` | 64-tile and 23-tile scans, 783 read requests, 189312 response words, 1392 logits checked, trace audit passed | passed in Icarus |
| QMAP LM-head descriptor wrapper | compact and full-vocab QMAP runtime packets plus persistent LM-head weight/scale vectors | exact output token `264`, score `1365150750`, and Q26 logit checks | compact: 64-tile and 23-tile descriptor scans plus invalid tile_count path; full: 9496 tiles, 151936 logits, 85475 read requests, 20664528 response words | compact passed in Icarus; full passed in xsim |
| QMAP final-token tail wrapper | final hidden, signed final RMSNorm gamma, final-norm scratch, and persistent LM-head descriptors | exact final-norm write-back plus output token `264`, score `1365150750`, and Q26 logit checks | compact: 64 tiles, 1024 logits, invalid gamma descriptor path; full: 9496 tiles, 151936 logits, 85494 read requests, 20666912 response words, 1024 norm writes | compact passed in Icarus; full passed in xsim |
| QMAP attention front-end wrapper | Q/K/V projection outputs, q/k gamma, RoPE cos/sin, KV-cache descriptor, and Q RoPE output descriptor | exact K/V cache address/data writes plus exact Q RoPE write-back | 31 read requests, 4944 read-response words, 2049 write requests, 4096 write-data words, invalid RoPE-cos dtype path, trace audit passed | passed in Icarus |
| QMAP attention score/value wrapper | Q RoPE input, K/V cache descriptor, exp LUT, and attention-output descriptor | exact K/V cache read order plus exact `attn_out[2048]` write-back | 20496 read requests, 22961 read-response words, 10240 K reads, 10240 V reads, 2048 output writes, invalid exp-LUT dtype path, trace audit passed | passed in Icarus |
| AXI4 write master | one aligned 4-beat write burst | exact AW/W/B behavior | no error | passed in Icarus |
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
| 2026-06-23 | Define formal QMAP inference-packet direction | Moves the next hardware-facing plan from extra GEMV smoke images to persistent model manifests, runtime work packets, Layer 0 QKV projection descriptors, and PL write-back into activation buffers |
| 2026-06-23 | Add local QMAP QKV projection write-back path | Adds `21_export_qmap_qkv_projection_image.py`, `qmap_qkv_projection_compute_path.sv`, and `axi4_write_master.sv`; Python export and Icarus simulations pass against compact Q/K/V expected output words |
| 2026-06-28 | Add full local QKV AXI top simulation pass | Adds `qmap_qkv_projection_axi_smoke_top.sv` and a parameterized AXI memory-model testbench; compact, medium, larger, and full Layer 0 Q/K/V packets pass local write-back comparison, with full scale `2048/1024/1024` rows and `4096` output writes |
| 2026-06-28 | Add local q/k norm + RoPE integration pass | Adds signed-gamma q/k norm support and `qk_norm_rope_stage_128.sv`; the real-vector test consumes Q4/QMAP fixed Q/K projection words, runs all 24 heads, and matches q_norm/k_norm plus RoPE golden outputs exactly |
| 2026-06-28 | Add local KV cache append integration pass | Adds `23_export_kv_cache_append_vectors.py`, `kv_cache_append.sv`, and `qk_norm_rope_kv_cache_stage.sv`; standalone and combined tests match exact K/V cache write addresses/data under backpressure |
| 2026-06-28 | Add local attention-score integration pass | Adds `24_export_attention_score_vectors.py`, `attention_score_stage.sv`, and `tb_attention_score_stage.sv`; the testbench and CSV trace audit prove exact GQA K-cache request order, response pairing, and raw/scaled score outputs |
| 2026-06-28 | Add local attention softmax/value pass | Adds `25_export_attention_softmax_value_vectors.py`, `attention_softmax_value_stage.sv`, and `tb_attention_softmax_value_stage.sv`; the testbench and CSV trace audit prove exact fixed softmax probabilities, V-cache response pairing, and attention-output words |
| 2026-06-28 | Add local attention o_proj pass | Adds `26_export_o_proj_vectors.py`, `o_proj_stage.sv`, and `tb_o_proj_stage.sv`; the testbench proves exact Q12.12 `o_proj_out[1024]` words under output backpressure |
| 2026-06-29 | Add local post-attention residual/RMSNorm pass | Adds `27_export_post_attention_residual_norm_vectors.py`, `residual_add_1024.sv`, `post_attention_residual_norm_stage.sv`, and `tb_post_attention_residual_norm_stage.sv`; the testbench proves exact residual and post-norm words plus trace-audited control timing |
| 2026-06-29 | Add local MLP gate/up projection pass | Adds `28_export_mlp_gate_up_vectors.py`, `mlp_gate_up_proj_stage.sv`, and `tb_mlp_gate_up_proj_stage.sv`; the testbench proves exact Q12.12 gate/up pairs for all 3072 rows under output backpressure |
| 2026-06-29 | Add local MLP SiLU/multiply pass | Adds `29_export_mlp_silu_mul_vectors.py`, `mlp_silu_mul_stage.sv`, and `tb_mlp_silu_mul_stage.sv`; the testbench proves exact Q12.12 `mlp_hidden[3072]` words under input/output backpressure with CSV trace audit |
| 2026-06-30 | Add local MLP down projection pass | Adds `30_export_mlp_down_vectors.py`, `mlp_down_proj_stage.sv`, and `tb_mlp_down_proj_stage.sv`; the testbench proves exact Q12.12 `down_out[1024]` words under output backpressure with CSV trace audit |
| 2026-06-30 | Add local final MLP residual add pass | Adds `31_export_mlp_residual_add_vectors.py`, `mlp_residual_add_stage.sv`, and `tb_mlp_residual_add_stage.sv`; the testbench proves exact Q14.10 `layer_out[1024]` words across two runs with CSV trace audit |
| 2026-06-30 | Add local final RMSNorm pass | Adds `32_export_final_rmsnorm_vectors.py`, `final_rmsnorm_stage.sv`, and `tb_final_rmsnorm_stage.sv`; the testbench proves exact Q12.12 final-norm words and scalar debug values across two runs with CSV trace audit |
| 2026-06-30 | Add local LM-head scan and greedy argmax pass | Adds `33_export_lm_head_argmax_vectors.py`, `lm_head_argmax_stage.sv`, and `tb_lm_head_argmax_stage.sv`; the testbench proves exact 1024-row Q4 LM-head scan-window logits and token `264` argmax across two runs with CSV trace audit |
| 2026-07-01 | Add local memory-backed LM-head scan-window pass | Adds LM-head weight/scale memory hex exports, `lm_head_tile_mem_reader.sv`, `lm_head_argmax_mem_stage.sv`, and `tb_lm_head_argmax_mem_stage.sv`; the testbench proves the same 1024-row LM-head window through memory requests with `1152` read bursts and exact token `264` |
| 2026-07-01 | Add local runtime LM-head tile scheduler pass | Adds `lm_head_argmax_tile_scheduler.sv` and `tb_lm_head_argmax_tile_scheduler.sv`; the testbench proves runtime scan counts of `64` and `23` tiles through the memory-backed LM-head stage, with exact token `264`, score `1365150750`, and `1392` checked logits |
| 2026-07-01 | Add local QMAP LM-head descriptor wrapper pass | Adds `34_export_qmap_lm_head_argmax_image.py`, `qmap_lm_head_argmax_compute_path.sv`, and `tb_qmap_lm_head_argmax_compute_path.sv`; the testbench proves descriptor-provided final-norm, weight/scale, scan-range, output, and debug tensors across `64`-tile and `23`-tile scans plus an invalid descriptor error path |
| 2026-07-02 | Add local full-vocab QMAP LM-head pass | Adds streaming full-vocab LM-head vector export and xsim full-vocab QMAP wrapper simulation; the run covers `9496` tiles / `151936` logits and returns exact token `264`, score `1365150750`, and `max_abs_logit_diff=0` |
| 2026-07-02 | Add first QMAP final-token tail wrapper pass | Adds `36_export_qmap_final_token_tail_image.py`, `qmap_final_token_tail_compute_path.sv`, and `tb_qmap_final_token_tail_compute_path.sv`; compact Icarus and full-vocab xsim prove descriptor-visible `final_hidden -> final_norm write-back -> LM-head argmax -> output token/score` |
| 2026-07-03 | Add first QMAP attention front-end wrapper pass | Adds `37_export_qmap_attention_frontend_image.py`, `qmap_attention_frontend_compute_path.sv`, and `tb_qmap_attention_frontend_compute_path.sv`; Icarus proves descriptor-visible `Q/K/V + q/k gamma + RoPE -> K/V cache append + Q RoPE write-back` |
| 2026-07-03 | Add first QMAP attention score/value wrapper pass | Adds `38_export_qmap_attention_score_value_image.py`, `qmap_attention_score_value_compute_path.sv`, and `tb_qmap_attention_score_value_compute_path.sv`; Icarus proves descriptor-visible `Q RoPE + K/V cache + exp LUT -> attn_out[2048]` |
| 2026-07-03 | Add first QMAP attention o_proj wrapper pass | Adds `39_export_qmap_o_proj_image.py`, `qmap_o_proj_compute_path.sv`, and `tb_qmap_o_proj_compute_path.sv`; Icarus proves descriptor-visible `attn_out[2048] + persistent o_proj Q4 weight/scale -> o_proj_out[1024]` |
| 2026-07-04 | Add first QMAP post-attention residual/RMSNorm wrapper pass | Adds `40_export_qmap_post_attention_residual_norm_image.py`, `qmap_post_attention_residual_norm_compute_path.sv`, and `tb_qmap_post_attention_residual_norm_compute_path.sv`; Icarus proves descriptor-visible `residual + o_proj_out + post-attention gamma -> post-attention hidden + post_norm` |
| 2026-07-04 | Add first QMAP MLP gate/up wrapper pass | Adds `41_export_qmap_mlp_gate_up_image.py`, `qmap_mlp_gate_up_compute_path.sv`, and `tb_qmap_mlp_gate_up_compute_path.sv`; Icarus proves descriptor-visible `post_norm[1024] + persistent gate/up Q4 weight/scale -> gate[3072] + up[3072]` |
| 2026-07-04 | Add first QMAP MLP SiLU/multiply wrapper pass | Adds `42_export_qmap_mlp_silu_mul_image.py`, `qmap_mlp_silu_mul_compute_path.sv`, and `tb_qmap_mlp_silu_mul_compute_path.sv`; Icarus proves descriptor-visible `gate[3072] + up[3072] + sigmoid LUT -> mlp_hidden[3072]` |
| 2026-07-04 | Add first QMAP MLP down wrapper pass | Adds `43_export_qmap_mlp_down_image.py`, `qmap_mlp_down_compute_path.sv`, and `tb_qmap_mlp_down_compute_path.sv`; Icarus proves descriptor-visible `mlp_hidden[3072] + persistent down-proj Q4 weight/scale -> down_out[1024]` |
| 2026-07-04 | Add first QMAP final MLP residual wrapper pass | Adds `44_export_qmap_mlp_residual_add_image.py`, `qmap_mlp_residual_add_compute_path.sv`, and `tb_qmap_mlp_residual_add_compute_path.sv`; Icarus proves descriptor-visible `post_attn_hidden[1024] + down_out[1024] -> layer_out[1024]` with descriptor/protocol no-write error paths |
| 2026-07-04 | Add first local QMAP Layer 0 body scheduler pass | Adds `qmap_layer0_body_scheduler.sv` and `tb_qmap_layer0_body_scheduler.sv`; Icarus chains post-attention residual/RMSNorm through final MLP residual behind one memory interface with exact write-back through `layer_out[1024]` |
| 2026-07-04 | Add first local QMAP full Layer 0 scheduler pass | Adds `qmap_layer0_full_scheduler.sv` and `tb_qmap_layer0_full_scheduler.sv`; Icarus chains attention front-end through final `layer_out[1024]` behind one memory interface with exact write-back and a no-write first-stage error path |
| 2026-07-04 | Add first local QMAP Layer 0 compute scheduler pass | Adds `qmap_layer0_compute_scheduler.sv` and `tb_qmap_layer0_compute_scheduler.sv`; Icarus chains full QKV projection through final `layer_out[1024]` behind one memory interface with exact write-back, QKV-stage no-write error handling, downstream error propagation, and trace-confirmed producer-before-consumer ordering |
| 2026-07-04 | Add first local one-token layer-loop boundary pass | Adds `qmap_one_token_layer_scheduler.sv`, `tb_qmap_one_token_layer_scheduler.sv`, and `tb_qmap_one_token_layer_scheduler_validation.sv`; Icarus runs Layer 0 compute through an explicit per-layer-base-table loop contract, proves a two-layer alias loop with aggregate counters and active-layer switching, and covers propagated child errors plus no-memory invalid-table/out-of-range exits |
| 2026-07-05 | Add true Layer 1 QKV packet/export coverage | Adds `45_export_layer_qkv_q4_vectors.py`, parameterizes `21_export_qmap_qkv_projection_image.py` with `--layer-id`, and makes `qmap_qkv_projection_axi_smoke_top.sv` consume a runtime QMAP base; compact and full Layer 1 QKV packets at `0x4_1008_0000` pass local AXI write-back comparison |
| 2026-07-05 | Add true Layer 1 attention front-end packet coverage | Parameterizes `22_export_qk_norm_rope_fixed_vectors.py`, `23_export_kv_cache_append_vectors.py`, `37_export_qmap_attention_frontend_image.py`, and `tb_qmap_attention_frontend_compute_path.sv` for Layer 1 vector prefixes and runtime QMAP base; the Layer 1 packet at `0x4_1502_0000` matches exact K/V cache and Q RoPE write-back with trace-confirmed cache-before-Q-RoPE ordering |
| 2026-07-05 | Add true Layer 1 attention score/value packet coverage | Parameterizes `24_export_attention_score_vectors.py`, `25_export_attention_softmax_value_vectors.py`, `38_export_qmap_attention_score_value_image.py`, and `tb_qmap_attention_score_value_compute_path.sv` for Layer 1 vector prefixes and runtime QMAP base; the Layer 1 packet at `0x4_1503_0000` matches exact K/V cache reads and `attn_out[2048]` write-back with trace-confirmed K-before-V-before-output ordering |
| 2026-07-05 | Add true Layer 1 attention o_proj packet coverage | Parameterizes `26_export_o_proj_vectors.py`, `39_export_qmap_o_proj_image.py`, and `tb_qmap_o_proj_compute_path.sv` for Layer 1 vector prefixes, runtime QMAP base, and persistent `o_proj` weight/scale bases; the Layer 1 packet at `0x4_1504_0000` matches exact persistent row reads and `o_proj_out[1024]` write-back with an invalid-descriptor no-write path |
| 2026-07-05 | Add true Layer 1 post-attention residual/RMSNorm packet coverage | Parameterizes `27_export_post_attention_residual_norm_vectors.py`, `40_export_qmap_post_attention_residual_norm_image.py`, `qmap_post_attention_residual_norm_compute_path.sv`, and `tb_qmap_post_attention_residual_norm_compute_path.sv` for Layer 1 metadata and runtime QMAP base; the Layer 1 packet at `0x4_1505_0000` matches exact post-attention hidden and post-norm write-back with an invalid-descriptor no-write path |
| 2026-07-05 | Add true Layer 1 MLP gate/up packet coverage | Parameterizes `28_export_mlp_gate_up_vectors.py`, `41_export_qmap_mlp_gate_up_image.py`, `qmap_mlp_gate_up_compute_path.sv`, and `tb_qmap_mlp_gate_up_compute_path.sv` for Layer 1 metadata, runtime QMAP base, and persistent gate/up weight/scale bases; the Layer 1 packet at `0x4_1506_0000` matches exact gate/up `[3072]` write-back with an invalid-descriptor no-write path |
| 2026-07-05 | Add true Layer 1 MLP SiLU/multiply packet coverage | Parameterizes `29_export_mlp_silu_mul_vectors.py`, `42_export_qmap_mlp_silu_mul_image.py`, and `tb_qmap_mlp_silu_mul_compute_path.sv` for Layer 1 metadata, runtime QMAP base, and true Layer 1 gate/up packet inputs; the Layer 1 packet at `0x4_1507_0000` matches exact `mlp_hidden[3072]` write-back with an invalid-descriptor no-write path |
| 2026-07-06 | Add true Layer 1 MLP down packet coverage | Parameterizes `30_export_mlp_down_vectors.py`, `43_export_qmap_mlp_down_image.py`, `qmap_mlp_down_compute_path.sv`, and `tb_qmap_mlp_down_compute_path.sv` for Layer 1 metadata, runtime QMAP base, and persistent down-proj weight/scale bases; the Layer 1 packet at `0x4_1508_0000` matches exact `down_out[1024]` write-back with exact persistent row reads and an invalid-descriptor no-write path |
| 2026-07-06 | Add true Layer 1 final MLP residual packet coverage | Parameterizes `31_export_mlp_residual_add_vectors.py`, `44_export_qmap_mlp_residual_add_image.py`, and `tb_qmap_mlp_residual_add_compute_path.sv` for Layer 1 metadata, runtime QMAP base, and true Layer 1 post-attention/down packet inputs; the Layer 1 packet at `0x4_1509_0000` matches exact `layer_out[1024]` write-back with bad descriptor and bad payload-last no-write paths |
| 2026-07-08 | Add QMAP input RMSNorm wrapper pass | Adds `48_export_qmap_input_rmsnorm_image.py`, `qmap_input_rmsnorm_compute_path.sv`, and `tb_qmap_input_rmsnorm_compute_path.sv`; local simulation proves standalone descriptor-visible `hidden[1024] + input_layernorm.gamma -> input_norm[1024]` write-back |
| 2026-07-08 | Integrate input RMSNorm before Layer 0 QKV scheduler | Adds optional input RMSNorm pre-stage to `qmap_layer0_compute_scheduler.sv`, per-layer base table plumbing through `qmap_one_token_layer_scheduler.sv`, chained QKV golden generated with `--activation-hex`, and focused `+qkv_precheck` Icarus pass |
| 2026-07-09 | Promote input RMSNorm -> QKV into one-token scheduler and top TBs | Adds `+input_norm_qkv_only` coverage to `tb_qmap_one_token_layer_scheduler.sv`, connects `i_input_norm_qmap_base_addr_table` through `qmap_one_token_top.sv`, proves exact input-norm and Q/K/V write-back in both scheduler and top builds, asserts that the focused top precheck does not launch final tail, and parses trace evidence that QKV reads the RMSNorm output only after the producer write completes |
| 2026-07-09 | Promote input RMSNorm gamma to signed per-layer contract | Changes input RMSNorm gamma packets and RTL validation to signed `I16_Q8_7` after Layer 1 artifact generation exposed negative `input_layernorm.weight`; regenerates Layer 0/1/2 input-normalized QKV artifacts with activation descriptors pointing at RMSNorm output buffers |
| 2026-07-10 | Add Layer1/Layer2 input RMSNorm scheduler prechecks | Extends `tb_qmap_one_token_layer_scheduler.sv` shared-memory image loading, write routing, per-layer input-norm expected data, and producer-before-consumer checks for Layer 1/2 input RMSNorm packets; focused Icarus prechecks prove exact input-norm and Q/K/V write-back for both layers |
| 2026-07-10 | Promote Layer1/Layer2 full scheduler paths to input RMSNorm | Updates focused `+l1_only` and `+l2_only` scheduler scenarios so their QKV activation source is the per-layer RMSNorm output; Icarus proves exact full-layer write-back and trace-audited producer-before-consumer ordering |
| 2026-07-10 | Promote Layer2 scheduler-to-final-tail top handoff to input RMSNorm | Regenerates Layer2-chained final RMSNorm, full-vocab LM-head, and final-tail QMAP artifacts from the current input-RMSNorm-enabled Layer 2 output; Vivado xsim proves `qmap_one_token_top.sv +l2_top_tail_only` automatically selects scheduler-written `0x4_2509_2540`, returns token `537` / score `850086863`, and trace-audits scheduler-to-tail ordering |
| 2026-07-10 | Add Layer1 -> Layer2 top-to-final-tail local xsim pass | Adds `+l1_l2_top_tail_only` coverage to `tb_qmap_one_token_layer_scheduler.sv`; Vivado xsim proves `qmap_one_token_top.sv` can run Layer 1 then Layer 2, select the scheduler-reported Layer 2 `layer_out[1024]` for final tail without prepatching the tail descriptor, return token `537` / score `850086863`, and trace-audit Layer1-to-Layer2 plus Layer2-to-tail producer-before-consumer ordering |
| 2026-07-11 | Add local MMIO-style one-token control proof | `qmap_one_token_control_regs.sv` plus `+l1_l2_mmio_top_tail_only` proves software-like register writes can launch Layer 1 -> Layer 2 -> final-tail through `qmap_one_token_top.sv`, returning token `537` / score `850086863` with exact status/counter readback; `qmap_one_token_mmio_top.sv` then productizes the same local register/top seam and passes a no-memory wrapper smoke test before AXI4-Lite work |
| 2026-07-11 | Add AXI4-Lite one-token control seam | `axi4lite_to_mmio_regs.sv` adds a focused, stall/error-tested AXI4-Lite-to-tiny-MMIO adapter; `qmap_one_token_axil_top.sv` wraps it around `qmap_one_token_mmio_top.sv` and passes the same invalid `layer_count=0` no-memory validation path through AXI-Lite writes/reads, with the tiny-MMIO wrapper regression still passing |
| 2026-07-11 | Promote AXI4-Lite wrapper to positive Layer1 -> Layer2 -> tail xsim | Extends the productized wrapper debug outputs and the reusable one-token memory-model TB so `qmap_one_token_axil_top.sv` can run the Layer1 -> Layer2 -> final-tail positive path through AXI-Lite register writes/reads; xsim returns token `537` / score `850086863`, layer mask `0x6`, top counters `178050/24947620` reads and `12312/52227` writes, and zero final write mismatches |
| 2026-07-11 | Promote AXI4-Lite wrapper to bounded mixed true3 top-to-tail xsim | Runs `qmap_one_token_axil_top.sv` through AXI-Lite register writes/reads for Layer0(QKV-first) -> Layer1 -> Layer2 -> final-tail; xsim returns token `537` / score `850086863`, layer mask `0x7`, top counters `224314/27085750` reads and `18466/76803` writes, and zero final write mismatches, establishing the current model-facing local regression before Vivado/Vitis integration planning |
| 2026-07-11 | Add PS-side one-token AXI-Lite runtime skeleton | Adds `FPGA_Project/software/qmap_one_token_runtime/` with `qmap_one_token_regs.h`, `qmap_one_token_runtime.h`, and a Vitis `main.c` no-memory validation smoke; the headers mirror the RTL register map/table ids, support table commits and run/result helpers, and syntax-check with host GCC pending a Vivado BD base-address assignment |
| 2026-07-11 | Add BD-facing one-token AXI shell/runbook | Adds `qmap_one_token_axi_top.sv`, wrapping `qmap_one_token_axil_top.sv` with the existing AXI4 read/write masters so Vivado can integrate one `S_AXI` control slave and one `M_AXI` PL-DDR master; adds `ONE_TOKEN_AXI_TOP_BD_PLAN.md` with current apertures, recommended `0xA004_0000` control base, wiring sequence, and pre-board regression gates |
| 2026-07-11 | Add BD-facing one-token AXI shell no-memory smoke | Adds `tb_qmap_one_token_axi_top.sv`, proving the Vivado-facing shell accepts the invalid `layer_count=0` no-memory validation request through `S_AXI`, reports done/error/layer0 error through register reads, keeps counters at zero, and issues no `M_AXI` read/write traffic |
| 2026-07-11 | Add safe Vivado BD integration scaffold | Adds `Vivado_Project/scripts/one_token_axi_top_bd_scaffold.tcl`, a dry-run-by-default Tcl scaffold that checks required RTL, plans `qmap_one_token_axi_top_0` insertion, expands `axi_smc`, assigns `S_AXI` at `0xA004_0000` and `M_AXI` to PL DDR, and requires `--apply` before modifying `llm_system.bd`; dry-run succeeds under Vivado 2024.2 |
| 2026-07-13 | Make large local writes legal AXI4 traffic | Extends `axi4_write_master.sv` to split one local write request at both 256-beat and 4 KiB boundaries; focused tests cover a 4096-byte layer/embedding write, stalls, BRESP failure, and malformed local requests |
| 2026-07-13 | Add the tied-Q4 token embedding stage | Adds `49_export_q4_embedding_vectors.py`, `q4_embedding_lookup.sv`, optional embedding sequencing in `qmap_one_token_top.sv`, AXI-Lite register/runtime fields, and standalone/AXI-Lite/BD-facing tests; exact model-derived Q14.10 output passes Icarus and Vivado XSim |
| 2026-07-13 | Contain simulation products under Temp | Adds timestamped local, XSim/timing-audit, and embedding regressions whose output roots are restricted to `Temp/`; generated vectors, compiler products, logs, and traces no longer need to be placed beside source files |
| 2026-07-13 | Detect and remove the integrated QKV/weight address collision | Records that the legacy full self-contained QKV packet at `0x4_0008_0000` extends into the persistent weight region at `0x4_0010_0000`; integrated frontend simulation now stages that packet at `0x4_1B40_0000` while persistent tied weights remain in place |
| 2026-07-13 | Add the continuous embedding -> input RMSNorm -> full QKV proof | Adds a Temp-contained chain exporter, shared-memory XSim testbench, reusable regression runner, and independent timing-trace checker; exact model-derived outputs and producer-before-consumer ordering pass under backpressure |
| 2026-07-13 | Extend the continuous token path through complete Layer 0 | Makes the existing chained attention/MLP exporters Temp-workspace aware, extends the orchestrator to Layer 0 embedding input, and adds `51_export_embedding_layer0_full_chain.py`, full shared-memory XSim coverage, a reusable runner, and independent exact timing/address audit |
| 2026-07-13 | Replace mixed true3 with a continuous token-driven true3 AXI-Lite proof | Adds `52_export_embedding_true3_final_chain.py`, extends the shared-memory AXI-Lite testbench from tied-Q4 embedding through three complete layers and the full-vocabulary final tail, and adds a reusable Temp-contained XSim runner plus an independent event/timing audit; exact token `537`, score `838805253`, counters, addresses, and producer-response-before-consumer ordering all pass |
| 2026-07-13 | Complete the continuous 28-layer single-token AXI-Lite proof | Adds `53_export_embedding_full28_final_chain.py`, a collision-free full-depth address manifest/audit, manifest-generated Temp-only testbench loads, generic 28-layer scoreboarding, and a streamed independent timing checker; fresh XSim reaches exact token `537` / score `1155032971`, layer mask `0x0fffffff`, all `281` QMAP packets, exact counters, and strict producer-response-before-consumer ordering with every artifact under `Temp/embedding_full28_axil_tail_regression/20260713_164001` |
| 2026-07-26 | Integrate and route the resource-reduced full28 board design | Replaces the row1024 BD with `qmap_one_token_axi_bd`, assigns DDR status at `0xA001_0000`, control at `0xA004_0000`, and PL DDR4 at `0x4_0000_0000..0x4_1FFF_FFFF`; the fixed bit/XSA closes timing at WNS `+0.208 ns`, fully routes `119383/119383` nets, and has no routing/DRC/methodology Error or Critical Warning |
| 2026-07-26 | Freeze the full28 PL-DDR board runtime | `57_pack_qmap_runtime_load_plan.py` emits 61 SHA256-verified binary segments totaling `394547200` bytes, preserves source plan/manifest provenance, and spans first address `0x4_0010_0000` through last end-exclusive `0x4_1A14_0000` |
| 2026-07-27 | Add board release and false-positive gates | Vitis builds control/model apps from the current XSA; XSDB automation checks DDR status and all 281 QMAP headers; packaging requires 397 mutable regions to be zero before launch and records state `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED` only after the full28 persistent two-token audit passes |
| 2026-08-01 | Close local full28 persistent release gates | The resource-reduced no-reset two-token XSim returns exact token/score `28458/1227344433` then `64/1015661901`; the independent event audit proves one reset release, exact reads/writes, all 281 QMAP packets per step, and retained position-0 K/V reads in every layer. `Temp/boardready_qwen3_full28_20260801` then passes its 92-file package verifier and becomes the board-test-ready, not-yet-hardware-validated release |
