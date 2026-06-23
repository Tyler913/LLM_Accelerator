# Current State

Last updated: 2026-06-23

This file is the concise working-state handoff. For durable project context,
read `Source/PROJECT_CONTEXT.md` first. For detailed address planning, read
`Source/FPGA_MEMORY_MAP.md`. For descriptor-based PL DDR4 tensor staging, read
`Source/QMAP_FORMAT.md`. For workflow rules, read `Source/AGENTS.md`.

## Current Goal

Build a first FPGA demo that can take a prompt and generate text with the local
`Qwen/Qwen3-0.6B-Base` model.

Current first-version target:

- Serial prefill and single-token cached decode
- Greedy argmax text continuation
- PS-side tokenizer, detokenizer, artifact loading, PL control, status polling,
  and debug output
- PL-side `next_token = run_one_token(input_token, position)` model path
- PL owns model-side prefill and decode math
- Initial context target: 128 or 256 tokens
- Required PL DDR4 weight path: custom Q4 weight-only format
- PL compute blocks use hand-written Verilog/SystemVerilog by default

## Software Baseline

The FP32 Python reference is complete for the first-version target. It covers
serial prefill/decode, full 28-layer cached single-token decode, and focused
module checks for the FPGA bring-up path.

Key locations:

- Full-model references: `Qwen3-0.6B-Base/pc_testing/`
- Module references: `Qwen3-0.6B-Base/python_each_module/`
- Module validation index:
  `Qwen3-0.6B-Base/python_each_module/README.md`
- FP32 test vectors:
  `artifacts/test_vectors/qwen3_0p6b_fp32_v0/` (ignored by Git)
- Q4 bring-up vectors:
  `artifacts/test_vectors/qwen3_0p6b_q4_v0/` (ignored by Git)

Useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/14_verify_q4_gemv_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/16_profile_rmsnorm_ranges.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/19_export_qmap_dot64_image.py --c-header FPGA_Project/software/qmap_load_smoke/qmap_dot64_image.h
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/20_export_qmap_row1024_image.py --c-header FPGA_Project/software/qmap_row1024_pl_compute_smoke/qmap_row1024_image.h
```

Stable reference prompt:

```text
The future of FPGA is
```

Known greedy checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## RTL Status

The current hand-written RTL bring-up stack includes:

- Q4 GEMV:
  - `q4_dot_product_64.sv`
  - `q4_gemv_row_1024.sv`
  - `q4_gemv_tile_1024.sv`
  - `q4_gemv_projection_1024.sv`
- RMSNorm:
  - `rmsnorm_sum_squares_1024.sv`
  - `fixed_sqrt_u64.sv`
  - `fixed_udiv.sv`
  - `rmsnorm_apply_1024.sv`
  - `rmsnorm_1024.sv`
- RoPE:
  - `rope_qk_layer_128.sv`
- KV cache address generation:
  - `kv_cache_addr_gen.sv`
- QMAP/DDR data movement bring-up:
  - `qmap_defs.svh`
  - `qmap_header_reader.sv`
  - `qmap_descriptor_reader.sv`
  - `qmap_dot64_reader.sv`
  - `qmap_dot64_payload_fetcher.sv`
  - `qmap_dot64_compute_path.sv`
  - `qmap_row1024_payload_fetcher.sv`
  - `qmap_row1024_compute_path.sv`
  - `axi4_read_master.sv`
  - `qmap_dot64_axi_smoke_top.sv`
  - `qmap_dot64_axi_smoke_bd.v`
  - `qmap_row1024_axi_smoke_top.sv`
  - `qmap_row1024_axi_smoke_bd.v`

Current fixed-point direction:

- Large weights: project custom Q4 weight-only format from
  `Source/Q4_FORMAT.md`
- Q4 scales: current bring-up uses unsigned 16-bit `Q2.14`
- RMSNorm/residual input: signed 24-bit `Q14.10`
- RMSNorm output and Q/K/RoPE path: signed 24-bit `Q12.12`
- RMSNorm gamma: unsigned 16-bit `UQ8.8`
- KV cache first RTL storage plan: signed 24-bit `Q12.12` padded to 32-bit
  DDR words

Latest local RTL/software validation state:

- Existing Icarus simulation set under `FPGA_Project/sim/` passed in the latest
  full local recheck.
- Q4 GEMV real-vector RTL checks pass for Layer 0 q_proj rows 0 through 15.
- RMSNorm real-vector RTL check passes for the Layer 0 input_layernorm
  last-token vector.
- RoPE real-vector RTL check passes for the Layer 0 last-token Q/K vector.
- `kv_cache_addr_gen.sv` passes context-256 and context-512 address smoke
  vectors.
- Python FP32 module vectors and Q4 GEMV vectors pass their current verifiers.
- `19_export_qmap_dot64_image.py` exports the first QMAP v1 PL DDR4 staging
  image from `q_proj_row0_group0_dot64.npz` and self-checks the header,
  descriptor table, and payload bytes.
- `qmap_load_smoke_app` passed on hardware. It writes the embedded 1536-byte
  QMAP dot64 image to PL DDR4 at `0x4_1B10_0000`, reads it back exactly, and
  checks the QMAP header, four descriptors, and payload spot values.
- `qmap_dot64_reader.sv` passes RTL simulation against the generated
  `qmap_dot64_image_words32.hex` memory image. It issues the project-local
  memory request/response reads, decodes the QMAP header, and captures the four
  dot64 descriptor slots for activation, packed Q4 weight, Q2.14 scale, and
  expected/debug result tensors.
- `qmap_dot64_payload_fetcher.sv` passes RTL simulation after
  `qmap_dot64_reader.sv`. It uses the descriptor base/nbytes fields to fetch
  the activation vector, packed Q4 weight group, Q2.14 scale, and expected
  partial/scaled debug values from the same generated QMAP memory image.
- `qmap_dot64_compute_path.sv` passes the first QMAP-backed dot64 compute
  wrapper simulation. This synthesizable wrapper owns the sequence:
  `qmap_dot64_reader.sv`, then `qmap_dot64_payload_fetcher.sv`, then
  `q4_dot_product_64.sv`. It compares the computed partial sum and scaled Q26
  result against the QMAP expected/debug payload. The matched values are
  `partial_sum=24751` and `scaled_sum_q26=3019622`.
- `axi4_read_master.sv` passes a focused RTL smoke simulation for one aligned
  4-beat AXI4 read burst. This is the first read-only adapter from the
  project-local memory request/response interface toward a future Vivado AXI
  master integration.
- `tb_qmap_dot64_compute_path_axi4.sv` passes the first AXI-backed QMAP dot64
  compute simulation. The path is `qmap_dot64_compute_path.sv ->
  axi4_read_master.sv -> AXI read memory model`, with the memory model loaded
  from `qmap_dot64_image_words32.hex`. The simulation observed 9 AXI read
  bursts and matched `partial_sum=24751`, `scaled_sum_q26=3019622`.
- `qmap_dot64_axi_smoke_top.sv` is the Vivado-facing top for the first PL-side
  QMAP dot64 hardware smoke path. It wraps `qmap_dot64_compute_path.sv` and
  `axi4_read_master.sv`, exposes a conventional AXI master port for Block
  Design integration, and provides sticky PS-visible status/result outputs.
  Its focused simulation passes with status `0xA`, 9 AXI read bursts,
  `partial_sum_low32=24751`, and `scaled_sum_q26_low32=3019622`.
- `qmap_dot64_axi_smoke_bd.v` is the Block Design friendly wrapper for that
  smoke path. It has fixed-width Verilog ports and no parameters/includes at
  the BD-facing boundary, then instantiates `qmap_dot64_axi_smoke_top.sv`
  internally. Its AXI/clock interface metadata currently uses the existing
  Vivado PS PL clock frequency, `96,968,727 Hz`, to match
  `zynq_ultra_ps_e_0/pl_clk0` and `axi_smc/S01_AXI`.
- `20_export_qmap_row1024_image.py` exports the second QMAP v1 image,
  `q_proj` Layer 0 row 0 row1024, at `0x4_1B20_0000`. The image is 4096 bytes,
  stores activation `[1024]`, packed Q4 weight `[1,1024]`, scale `[1,16]`,
  and expected row sum debug payload. The expected result is
  `row_sum_q26_int64=-3482169`.
- `qmap_row1024_compute_path.sv` passes RTL simulation against
  `qmap_row1024_image_words32.hex`. It reads QMAP descriptors, fetches the
  row payloads in 4-group batches, runs `q4_gemv_row_1024.sv`, and matches the
  expected row sum.
- `qmap_row1024_payload_fetcher.sv`, `qmap_row1024_compute_path.sv`,
  `qmap_row1024_axi_smoke_top.sv`, and `qmap_row1024_axi_smoke_bd.v` now form
  the Vivado-facing row1024 smoke path. The AXI simulation passes with 18 read
  bursts, status `0xA`, and row result `-3482169`.
- The row1024 PL master smoke path has passed on hardware. PS writes the
  4096-byte QMAP image to `0x4_1B20_0000`, readback and header checks pass,
  PL reports status `0xA`, and both result GPIO channels return
  `0xFFCA_DDC7` / `-3482169`.
- RTL source files now use explicit `input wire logic` ports. This keeps
  `default_nettype none` enabled while satisfying Vivado 2025.1 synthesis,
  which rejects plain `input logic` as an implicit net in this flow.

## Hardware Bring-Up Status

PS-to-PL DDR4 access is complete as a hardware checkpoint. The first
PL-initiated QMAP dot64 read/compute smoke design passed on hardware, and the
row1024 QMAP read/compute smoke design has now also been integrated into the
Vivado block design, synthesized, implemented, bitstream-generated, exported
for Vitis, and passed on hardware.

Current Vivado/Vitis artifacts:

- Vivado project: `FPGA_Project/Vivado_Project/LLM_FPGA.xpr`
- Current exported hardware handoff:
  `FPGA_Project/Vivado_Project/llm_system_qmap_row1024_pl_master.xsa`
- Current generated bitstream:
  `FPGA_Project/Vivado_Project/LLM_FPGA.runs/impl_1/llm_system_wrapper.bit`
- Current short-path Vitis workspace:
  `F:\vws`
- Current Vitis row1024 platform:
  `F:\vws\p_r1024\`
- Current Vitis row1024 PL compute smoke app:
  `F:\vws\a_r1024\`
- Current Vitis DDR4 smoke baseline in the same clean workspace:
  `F:\vws\p_ddr4\` and `F:\vws\a_ddr4\`
- Previous proven PS-to-PL DDR4 platform:
  `Vitis_Workspace/llm_pl_ddr4_aux_reset_fix_platform/`
- Durable standalone smoke-test source:
  `FPGA_Project/software/pl_ddr4_smoke/main.c`
- Durable QMAP load/readback smoke-test source:
  `FPGA_Project/software/qmap_load_smoke/main.c`
- Embedded QMAP dot64 C header for that smoke app:
  `FPGA_Project/software/qmap_load_smoke/qmap_dot64_image.h`
- Durable QMAP PL compute smoke-test source:
  `FPGA_Project/software/qmap_pl_compute_smoke/main.c`
- Durable QMAP PL compute embedded image header:
  `FPGA_Project/software/qmap_pl_compute_smoke/qmap_dot64_image.h`
- Durable QMAP row1024 PL compute smoke-test source:
  `FPGA_Project/software/qmap_row1024_pl_compute_smoke/main.c`
- Durable QMAP row1024 PL compute embedded image header:
  `FPGA_Project/software/qmap_row1024_pl_compute_smoke/qmap_row1024_image.h`
- Current Vitis DDR4 workspace app copy:
  `F:\vws\a_ddr4\src\main.c`
- Current QMAP row1024 Vitis workspace app copy:
  `F:\vws\a_r1024\src\main.c`

Current PS-to-PL memory fabric:

```text
M_AXI_HPM0_FPD
  -> AXI SmartConnect
      M00_AXI -> AXI BRAM Controller -> Block Memory Generator
      M01_AXI -> AXI Clock Converter -> ddr4_0/C0_DDR4_S_AXI
      M02_AXI -> AXI GPIO DDR4 status register
      additional AXI GPIO slaves for QMAP smoke control/status/result

qmap_row1024_axi_smoke_0/M_AXI
  -> AXI SmartConnect S01_AXI
      -> AXI Clock Converter -> ddr4_0/C0_DDR4_S_AXI
```

Current address map:

| Space / IP | Base | High | Size | Status |
| --- | ---: | ---: | ---: | --- |
| PS DDR low memory | `0x0000_0000` | `0x7FEF_FFFF` | about 2 GiB minus reserved top window | exported in current BSP |
| AXI BRAM smoke memory | `0xA000_0000` | `0xA000_1FFF` | 8 KiB | passed in current hardware smoke |
| DDR4 status AXI GPIO | `0xA001_0000` | `0xA001_FFFF` | 64 KiB | passed in current hardware smoke |
| QMAP smoke control/status AXI GPIO | `0xA002_0000` | `0xA002_FFFF` | 64 KiB | passed in dot64 and row1024 PL master hardware smoke |
| QMAP smoke result AXI GPIO | `0xA003_0000` | `0xA003_FFFF` | 64 KiB | passed in dot64 and row1024 PL master hardware smoke |
| PL DDR4 | `0x4_0000_0000` | `0x4_1FFF_FFFF` | 512 MiB | passed in current hardware smoke |

DDR4 status GPIO bit layout at `0xA001_0000`:

| Bit | Signal | Good Value |
| ---: | --- | ---: |
| 0 | `ddr4_0/c0_init_calib_complete` | `1` |
| 1 | `ddr4_0/c0_ddr4_ui_clk_sync_rst` | `0` |
| 2 | `proc_sys_reset_0/peripheral_aresetn` | `1` |

The good DDR4 status word is therefore `0x5`.

Current reset wiring:

- `ddr4_0/c0_ddr4_ui_clk_sync_rst` feeds the DDR UI-domain
  `proc_sys_reset_0/ext_reset_in`.
- `proc_sys_reset_0/aux_reset_in` is tied high because
  `proc_sys_reset_0` treats that input as active-low.
- `proc_sys_reset_0/mb_debug_sys_rst` is tied low.
- `proc_sys_reset_0/peripheral_aresetn` drives the DDR-side reset release path,
  including `ddr4_0/c0_ddr4_aresetn` and the AXI Clock Converter M_AXI-side
  reset.

Current board-run result:

- Boot mode: JTAG boot (`0000`)
- Serial path: CH340 USB-UART on COM110, 115200 baud
- Previous PS-to-PL DDR4 smoke launch style: `psu_init.tcl` enabled, FSBL
  initialization disabled
- Current QMAP PL master launch style: FSBL initialization enabled,
  `psu_init.tcl` disabled. This avoids a DAP transaction error while polling
  PS DDR PHY register `0xFD080030` in `psu_init.tcl`.
- On Windows, use a short Vitis workspace path such as `F:\vws` and short
  component names such as `p_ddr4`, `a_ddr4`, `p_r1024`, and `a_r1024`.
  Longer paths can make Vitis/CMake/Ninja fail BSP builds while opening
  generated `.obj.d` dependency files.
- AXI BRAM smoke passed at:
  - `0x0_A0000000`
  - `0x0_A0000004`
  - `0x0_A0000008`
  - `0x0_A0000400`
  - `0x0_A0001FFC`
- DDR4 status reported:
  `DDR4 status raw 0x5 calib_complete=1 ui_reset=0 axi_resetn=1`
- PL DDR4 smoke passed at:
  - `0x4_00000000`
  - `0x4_00000004`
  - `0x4_00001000`
  - `0x4_10000000`
  - `0x4_1FFFFFFC`
- QMAP load/readback smoke passed:
  - image SHA256:
    `b56319cf576fe8486e3586ee49a4194323cdfe4f4e6208c8b6057e741e5978d4`
  - image base: `0x4_1B10_0000`
  - image size: `0x600` / 1536 bytes
  - descriptor table: `0x4_1B10_0100`
  - payload base: `0x4_1B10_0500`
  - descriptor count/capacity: `4` / `8`
  - byte-for-byte PS readback compare: passed
- QMAP dot64 PL master smoke passed:
  - PS writes the same embedded QMAP image to `0x4_1B10_0000`
  - QMAP readback compare passes for 1536 bytes
  - PS clears and starts the PL compute path through the control GPIO at
    `0xA002_0000`
  - final PL status is `0xA`
  - result GPIO values at `0xA003_0000` are
    `partial_sum_low32=0x60AF` and `scaled_sum_q26_low32=0x2E1366`
- QMAP row1024 PL master smoke passed:
  - PS writes the embedded 4096-byte QMAP image to `0x4_1B20_0000`
  - QMAP readback compare passes for 4096 bytes
  - header checks pass for magic `0x50414D51`, version `1`, descriptor count
    `4`, descriptor capacity `8`, descriptor table `0x4_1B20_0100`, payload
    base `0x4_1B20_0500`, and image size `0x1000`
  - PS clears and starts the PL compute path through the control GPIO at
    `0xA002_0000`
  - final PL status is `0xA`
  - result GPIO values at `0xA003_0000` are
    `row_sum_q26_low32=0xFFCA_DDC7` and
    `expected_row_sum_q26_low32=0xFFCA_DDC7`, both representing `-3482169`

This proves the first PS-to-PL DDR4 path through
`M_AXI_HPM0_FPD -> axi_smc -> axi_clock_converter_0 -> ddr4_0/C0_DDR4_S_AXI`
is working for 32-bit standalone smoke-test accesses.

It also proves PL AXI master read paths from real PL DDR4 into the QMAP/Q4
dot64 and row1024 compute chains:
`qmap_*_axi_smoke_0/M_AXI -> axi_smc -> axi_clock_converter_0 ->
ddr4_0/C0_DDR4_S_AXI`.

Previous useful checkpoint:

- The earlier BRAM-only hardware checkpoint used `M_AXI_HPM0_LPD` and
  `0x8000_0000` through `0x8000_1FFF`.
- The current integrated PL DDR4 design uses `M_AXI_HPM0_FPD`; the BRAM smoke
  aperture moved to `0xA000_0000`.

## Open Gaps

- PL DDR4 is accessible from PS. The PL QMAP descriptor reader, payload
  fetcher, Q4 dot64 compute chain, row1024 compute chain, and read-only AXI4
  adapter are now wrapped for Vivado, integrated into block designs, and have
  passed board validation against real PL DDR4.
- QMAP v1 now defines the first descriptor-based PL DDR4 staging contract. The
  dot64 image exporter and standalone PS loader/readback app have passed on
  hardware. The first PL descriptor reader, payload fetcher, and Q4 dot64
  compute hookup are now wrapped in a synthesizable dot64 smoke controller.
  This wrapper is a bring-up micro-kernel, not the final full-model scheduler.
- The full-model memory-map layout for Q4 artifacts, KV cache, activation
  buffers, RoPE tables, and debug regions is still draft.
- RMSNorm, GEMV, RoPE, and KV address RTL blocks are validated in focused
  simulations, but they are not yet integrated into a streaming or memory-
  mapped one-token datapath.
- Attention score/softmax/value accumulation, residual add, SiLU, MLP
  elementwise multiply, final LM-head scan, and greedy argmax are still
  pending in RTL.
- Cache coherency, bulk transfer strategy, and any future DMA policy are still
  open. The current proven path is direct PS memory-mapped 32-bit access.
- A first Vivado bring-up attempt for row1024 showed LUT over-utilization with
  the earlier full-row payload buffering. The current row1024 RTL is
  resource-reduced for bring-up by fetching and computing 4 groups at a time,
  and that reduced path has now passed synthesis, implementation, bitstream
  generation, Vitis launch, and board validation.

## Immediate Next Step

Scale from one row1024 result to a small multi-row tile.

Recommended next slice:

1. Keep the passing row1024 hardware design as the board smoke baseline.
2. Export a small QMAP tile image for consecutive `q_proj` rows, for example
   4 or 8 rows, using the same activation vector and per-row Q4 weights/scales.
3. Extend the RTL path so it loops over row descriptors or row offsets and
   accumulates one `row_sum_q26` result per row.
4. Validate the tile in simulation first, then rebuild the Vivado smoke top and
   run a board smoke app that checks all tile result words.
5. Only after the small tile passes, scale toward a larger `q_proj` block and
   then a fuller attention/MLP datapath.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Treat BF16/FP32 model data as software reference and validation input only.
  The first PL DDR4 weight layout and GEMV datapaths must assume custom Q4
  weight-only storage for large weights.
- Core project handoff Markdown files live under `Source/`; the root
  `README.md` remains the entry point.
- Keep Q4 quantization semantics in `Source/Q4_FORMAT.md` and DDR tensor
  placement in `Source/QMAP_FORMAT.md`.
- Keep PS-side code as orchestration/support only. Do not move model math for
  prefill or decode back into PS unless the project direction explicitly
  changes.
- Treat `Vitis_Workspace/` and `F:\vws\` as local regenerated workspaces. Keep
  durable application source under `FPGA_Project/software/` and recreate Vitis
  platform/application/build outputs from the Vivado hardware export. On
  Windows, prefer the short `F:\vws` path for new Vitis workspaces to avoid BSP
  build failures from generated path lengths.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- Keep `Source/CURRENT_STATE.md` concise. Store detailed validation logic in
  scripts and summarize only durable results here.
