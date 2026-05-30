# Current State

Last updated: 2026-05-30

This file is the concise working-state handoff. For stable project context,
read `PROJECT_CONTEXT.md` first. For workflow rules, read `AGENTS.md`.

## Current Goal

Build a first FPGA demo that can take a prompt and generate text with the local
`Qwen/Qwen3-0.6B-Base` model.

Current first-version target:

- Serial prefill and single-token cached decode
- Greedy argmax text continuation
- PS-side tokenizer, detokenizer, and PL control
- PL-side `next_token = run_one_token(input_token, position)` path
- PL owns all model-side prefill and decode math; PS schedules tokens, loads
  artifacts, writes control registers, polls status, and handles text I/O/debug
- Initial context target: 128 or 256 tokens
- Required PL DDR4 weight path: custom Q4 weight-only format

## Software Baseline

The FP32 Python reference is complete for the first-version target. It covers
serial prefill/decode, full 28-layer cached single-token decode, and focused
module checks for the FPGA bring-up path.

Key locations:

- Full-model references: `Qwen3-0.6B-Base/pc_testing/`
- Module references: `Qwen3-0.6B-Base/python_each_module/`
- Module validation index:
  `Qwen3-0.6B-Base/python_each_module/README.md`
- Test vectors:
  `artifacts/test_vectors/qwen3_0p6b_fp32_v0/` (ignored by Git)
- Q4 bring-up vectors:
  `artifacts/test_vectors/qwen3_0p6b_q4_v0/` (ignored by Git)

Useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/13_export_q4_gemv_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/14_verify_q4_gemv_vectors.py
```

Latest local validation:

- 2026-05-28: `conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py`
  passed. The existing FP32 RMSNorm and Q/K/V GEMV vectors are usable as the
  first RTL bring-up references.
- 2026-05-28: Layer 0 Q/K/V custom Q4 v0 export and verification passed with
  signed int4 weights, group size 64, `Q2.14` scales, and `Q4.12` activation
  test input. See `Q4_FORMAT.md`.
- 2026-05-29: `q4_dot_product_64` FSM skeleton compiles with
  `iverilog -g2012 -tnull FPGA_Project\rtl\q4_dot_product_64.sv`; datapath
  implementation is still pending.
- 2026-05-30: On macOS, `conda run -n llm_fpga python init/verify_assets.py`
  passed and the local Git safety hook matches `init/git-hooks/pre-commit`.
  The ignored FP32 and Q4 vector directories were regenerated locally with
  scripts `11_export_fpga_test_vectors.py` and `13_export_q4_gemv_vectors.py`.
- 2026-05-30: On macOS, `12_verify_fpga_test_vectors.py` and
  `14_verify_q4_gemv_vectors.py` both passed. The regenerated dot64 smoke
  vector still has `partial_sum_int64 = 24751` and
  `scaled_sum_q26_int64 = 3019622`.
- 2026-05-30: On macOS, `iverilog -g2012 -tnull
  FPGA_Project/rtl/q4_dot_product_64.sv` passes with the Homebrew `iverilog`;
  datapath implementation is still pending.
- 2026-05-30: User added the first `q4_dot_product_64` datapath draft
  (`current_activation`, `current_weight_q4`, `current_product`, RUN
  accumulation, and SCALE multiply). Syntax still passes with `iverilog`, but
  review found that SCALE must convert the unsigned `i_scale_q2_14` into a
  signed positive value before multiplying by signed `o_partial_sum`.
- 2026-05-30: After adding signed scale conversion, `q4_dot_product_64`
  passed a temporary `iverilog`/`vvp` smoke simulation generated from
  `q_proj_row0_group0_dot64.npz`: `o_partial_sum = 24751` and
  `o_scaled_sum_q26 = 3019622`.
- 2026-05-30: Added a reusable Icarus Verilog testbench at
  `FPGA_Project/sim/tb_q4_dot_product_64.sv`. Running it generates
  `FPGA_Project/wave/q4_dot_product_64.vcd` and passes the same dot64 smoke
  vector (`24751`, `3019622`). Waveform/build outputs are ignored by Git.
- 2026-05-30: Added the initial `q4_gemv_row_1024` RTL interface skeleton with
  1024-wide activation/weight inputs, 16 Q2.14 group scales, busy/done, and a
  signed 48-bit Q26 row accumulator output. The empty skeleton passes
  `iverilog -g2012 -tnull`.
- 2026-05-30: `q4_gemv_row_1024` now has the parallel-compute FSM draft,
  16 generated `q4_dot_product_64` instances, and a combinational 16-way
  scaled-sum adder draft. Elaboration with `q4_dot_product_64` passes
  `iverilog -g2012 -tnull`; output-register logic is still pending.
- 2026-05-30: `q4_gemv_row_1024` output logic now elaborates and passed a
  temporary `iverilog`/`vvp` smoke simulation against Q4 artifact data for
  `q_proj` row 0: `o_row_sum_q26 = -3482169`, matching the Python-computed
  exact Q26 row sum.
- 2026-05-30: Updated `q4_gemv_row_1024` comments to describe the current
  parallel 16-dot64 architecture and added a reusable testbench at
  `FPGA_Project/sim/tb_q4_gemv_row_1024.sv`. Running it generates
  `FPGA_Project/wave/q4_gemv_row_1024.vcd` and passes the `q_proj` row 0 smoke
  vector with `o_row_sum_q26 = -3482169`.
- 2026-05-30: Added the initial `q4_gemv_tile_1024` RTL interface skeleton.
  It parameterizes `OUT_ROWS` for 1/2/4-row parallel experiments, shares one
  1024-wide activation input, accepts `OUT_ROWS` packed Q4 weight rows and
  scale rows, and outputs flattened signed Q26 row results. The empty skeleton
  passes `iverilog -g2012 -tnull`.
- 2026-05-30: Implemented the first `q4_gemv_tile_1024` parallel row
  controller. It pulses `OUT_ROWS` generated `q4_gemv_row_1024` instances,
  waits for all row `done` bits, and packs the signed Q26 row outputs into
  `o_output_flat`. Elaboration passes for `OUT_ROWS=1`, `2`, and default `4`
  with Icarus Verilog.
- 2026-05-30: Added a reusable `q4_gemv_tile_1024` testbench at
  `FPGA_Project/sim/tb_q4_gemv_tile_1024.sv` plus generated hex vector files
  under `FPGA_Project/sim/vectors/`. Running it generates
  `FPGA_Project/wave/q4_gemv_tile_1024.vcd` and passes `q_proj` rows 0..3 with
  Q26 outputs `[-3482169, 7403300, 4069596, -6026990]`.

Stable reference prompt:

```text
The future of FPGA is
```

Known greedy checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## FPGA Bring-Up Status

The first PS-to-PL AXI communication checkpoint is now successful.

The development direction is now explicitly PL-first and RTL-first:

- The main project purpose is to learn and practice Verilog/RTL and FPGA PL
  development.
- PS-side bare-metal code is support infrastructure for control, loading,
  serial output, and validation.
- Do not use HLS as the default path. Future PL compute blocks should be
  hand-written Verilog/SystemVerilog unless the user explicitly changes this.

Vivado project:

```text
FPGA_Project/Vivado_Project/LLM_FPGA.xpr
```

Exported hardware:

```text
FPGA_Project/Vivado_Project/llm_system_axi_bram_smoke.xsa
```

Current Vivado design summary:

- Vivado 2025.1.1, part `xczu2eg-sfvc784-2-i`
- Zynq UltraScale+ MPSoC with PS DDR, QSPI, UART0, SD1, PL0 clock,
  fabric reset, `M_AXI_HPM0_LPD`, and `M_AXI_HPM0_FPD`
- AXI BRAM smoke-test path:
  `M_AXI_HPM0_LPD` -> AXI SmartConnect -> AXI BRAM Controller -> Block
  Memory Generator
- AXI BRAM address range: `0x8000_0000` through `0x8000_1FFF` (8 KiB)
- AXI BRAM Controller is now configured for one BRAM interface
  (`C_SINGLE_PORT_BRAM=1` / `SINGLE_PORT_BRAM=1`)
- `M_AXI_HPM0_FPD` remains enabled but intentionally unconnected
- Bitstream generation and hardware export completed successfully

Vitis workspace:

```text
Vitis_Workspace/
```

Current Vitis components:

- Platform: `Vitis_Workspace/llm_axi_bram_platform`
- Application: `Vitis_Workspace/axi_bram_smoke_app`
- App source: `FPGA_Project/software/axi_bram_smoke/main.c`
- App ELF:
  `Vitis_Workspace/axi_bram_smoke_app/build/axi_bram_smoke_app.elf`
- Launch config:
  `Vitis_Workspace/axi_bram_smoke_app/_ide/launch.json`

Confirmed hardware run:

- Boot mode switches: all `ON` (`0000`, JTAG boot)
- Serial: CH340 USB-UART on COM21, 115200 baud, UART0 on MIO42/MIO43
- Board initialization: TCL / `psu_init.tcl`, not FSBL
- Vitis launch programs the platform bitstream from
  `Vitis_Workspace/llm_axi_bram_platform/hw/sdt/llm_system_axi_bram_smoke.bit`
- The smoke-test app repeatedly passes 32-bit write/read checks at offsets
  `0x0`, `0x4`, `0x8`, `0x400`, and `0x1FFC`

## Issues And Fixes

Problems encountered during bring-up:

- Vitis BSP clean/build initially failed after moving directories because
  generated CMake/Vitis files contained stale absolute paths.
- The ARM DAP was not visible until the board, tools, and boot-mode/JTAG setup
  were aligned.
- The generated platform did not provide the expected `sw/boot/fsbl.elf`, so
  the default FSBL launch path was not usable.
- The first AXI BRAM hardware run reached `main()` but read back `0x00000008`
  after writing `0xA5A50000` at `0x8000_0000`.
- After the BRAM configuration was fixed, Vitis still failed once because the
  application launch configuration pointed to a stale app-local bitstream copy.

Solutions applied:

- Recreated `Vitis_Workspace/` under the shorter `F:/LLM_Accelerator` path.
- Used boot mode `0000` for JTAG boot and the CH340 `USB_UART (PS_PORT)` path
  for serial output.
- Switched Vitis board initialization from FSBL to TCL / `psu_init.tcl`.
- Changed the AXI BRAM Controller from two BRAM interfaces to one BRAM
  interface, matching the single connected Block Memory Generator port.
- Regenerated the bitstream, exported a new XSA with bitstream, rebuilt the
  Vitis platform/application, and updated `launch.json` to program the updated
  platform bitstream.

## Immediate Next Step

Use the working AXI path as the base for real PL-first RTL development:

1. Review the visible Vivado/Vitis/source changes and decide the minimal
   hardware checkpoint to commit.
2. Start the first hand-written Verilog/SystemVerilog PL compute block under
   `FPGA_Project/`, using the exported FP32 vectors as the reference.
3. Keep PS-side C minimal: only enough to load inputs, start/check hardware,
   read outputs, and print pass/fail evidence.
4. Continue refining `FPGA_MEMORY_MAP.md` when PL DDR4 is instantiated,
   especially PL DDR4 base/range, usable capacity, PS-to-PL transfer path, and
   Q4/KV/activation budgets.
5. Implement `q4_dot_product_64` against `q_proj_row0_group0_dot64.npz`:
   unpack 64 signed int4 weights, multiply by 64 signed `Q4.12` activations,
   accumulate the raw partial sum, then multiply by the unsigned `Q2.14` scale.
   RMSNorm remains the next activation-side block.
6. Extend the vector exporter as new hardware blocks are added, especially
   RoPE, KV cache append/read, attention, MLP, and complete Layer 0 cached
   decode.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Treat BF16/FP32 model data as software reference and validation input only.
  The first PL DDR4 weight layout and GEMV datapaths must assume custom Q4
  weight-only storage for large weights.
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
