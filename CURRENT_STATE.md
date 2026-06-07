# Current State

Last updated: 2026-06-07

This file is the concise working-state handoff. For durable project context,
read `PROJECT_CONTEXT.md` first. For detailed address planning, read
`FPGA_MEMORY_MAP.md`. For descriptor-based PL DDR4 tensor staging, read
`QMAP_FORMAT.md`. For workflow rules, read `AGENTS.md`.

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

Current fixed-point direction:

- Large weights: project custom Q4 weight-only format from `Q4_FORMAT.md`
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

## Hardware Bring-Up Status

PS-to-PL DDR4 access is complete as a hardware checkpoint.

Current Vivado/Vitis artifacts:

- Vivado project: `FPGA_Project/Vivado_Project/LLM_FPGA.xpr`
- Current exported hardware handoff:
  `FPGA_Project/Vivado_Project/llm_system_pl_ddr4_aux_reset_fix.xsa`
- Current generated bitstream:
  `FPGA_Project/Vivado_Project/LLM_FPGA.runs/impl_1/llm_system_wrapper.bit`
- Current Vitis platform:
  `Vitis_Workspace/llm_pl_ddr4_aux_reset_fix_platform/`
- Durable standalone smoke-test source:
  `FPGA_Project/software/pl_ddr4_smoke/main.c`
- Durable QMAP load/readback smoke-test source:
  `FPGA_Project/software/qmap_load_smoke/main.c`
- Embedded QMAP dot64 C header for that smoke app:
  `FPGA_Project/software/qmap_load_smoke/qmap_dot64_image.h`
- Current Vitis workspace app copy:
  `Vitis_Workspace/pl_ddr4_smoke_app/src/main.c`
- Current QMAP Vitis workspace app copy:
  `Vitis_Workspace/qmap_load_smoke_app/main.c`

Current PS-to-PL memory fabric:

```text
M_AXI_HPM0_FPD
  -> AXI SmartConnect
      M00_AXI -> AXI BRAM Controller -> Block Memory Generator
      M01_AXI -> AXI Clock Converter -> ddr4_0/C0_DDR4_S_AXI
      M02_AXI -> AXI GPIO DDR4 status register
```

Current address map:

| Space / IP | Base | High | Size | Status |
| --- | ---: | ---: | ---: | --- |
| PS DDR low memory | `0x0000_0000` | `0x7FEF_FFFF` | about 2 GiB minus reserved top window | exported in current BSP |
| AXI BRAM smoke memory | `0xA000_0000` | `0xA000_1FFF` | 8 KiB | passed in current hardware smoke |
| DDR4 status AXI GPIO | `0xA001_0000` | `0xA001_FFFF` | 64 KiB | passed in current hardware smoke |
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
- Vitis launch style: `psu_init.tcl` enabled, FSBL initialization disabled
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

This proves the first PS-to-PL DDR4 path through
`M_AXI_HPM0_FPD -> axi_smc -> axi_clock_converter_0 -> ddr4_0/C0_DDR4_S_AXI`
is working for 32-bit standalone smoke-test accesses.

Previous useful checkpoint:

- The earlier BRAM-only hardware checkpoint used `M_AXI_HPM0_LPD` and
  `0x8000_0000` through `0x8000_1FFF`.
- The current integrated PL DDR4 design uses `M_AXI_HPM0_FPD`; the BRAM smoke
  aperture moved to `0xA000_0000`.

## Open Gaps

- PL DDR4 is accessible from PS, but there is not yet a real PL data mover,
  AXI master, DMA path, or compute kernel consuming PL DDR4 data.
- QMAP v1 now defines the first descriptor-based PL DDR4 staging contract. The
  dot64 image exporter and standalone PS loader/readback app have passed on
  hardware. A PL descriptor reader is not built yet.
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

## Immediate Next Step

Start the first QMAP-backed PL DDR4 data movement step for the accelerator
path.

Recommended next slice:

1. Design the first narrow PL-side QMAP descriptor reader.
2. Keep the first PL reader focused on fetching and decoding QMAP header fields
   and descriptor slots from `0x4_1B10_0000`.
3. Simulate that reader against the known QMAP header/descriptor contents
   before integrating any AXI master or compute datapath.
4. After descriptor parsing works, connect the reader to payload fetch for the
   activation, Q4 weight, and scale tensors, then to `q4_dot_product_64`.

Keep this step narrow. The goal is not full-model loading yet; it is to turn
the proven PL DDR4 aperture into a stable software/hardware tensor contract.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Treat BF16/FP32 model data as software reference and validation input only.
  The first PL DDR4 weight layout and GEMV datapaths must assume custom Q4
  weight-only storage for large weights.
- Keep Q4 quantization semantics in `Q4_FORMAT.md` and DDR tensor placement in
  `QMAP_FORMAT.md`.
- Keep PS-side code as orchestration/support only. Do not move model math for
  prefill or decode back into PS unless the project direction explicitly
  changes.
- Treat `Vitis_Workspace/` as a local regenerated workspace. Keep durable
  application source under `FPGA_Project/software/` and recreate Vitis
  platform/application/build outputs from the Vivado hardware export.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- Keep `CURRENT_STATE.md` concise. Store detailed validation logic in scripts
  and summarize only durable results here.
