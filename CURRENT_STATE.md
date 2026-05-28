# Current State

Last updated: 2026-05-28

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

Useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py
```

Latest local validation:

- 2026-05-28: `conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py`
  passed. The existing FP32 RMSNorm and Q/K/V GEMV vectors are usable as the
  first RTL bring-up references.

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
   `FPGA_Project/verilog/`, using the exported FP32 vectors as the reference.
3. Keep PS-side C minimal: only enough to load inputs, start/check hardware,
   read outputs, and print pass/fail evidence.
4. Continue refining `FPGA_MEMORY_MAP.md` when PL DDR4 is instantiated,
   especially PL DDR4 base/range, usable capacity, PS-to-PL transfer path, and
   Q4/KV/activation budgets.
5. Bring up RMSNorm first, then GEMV, as hand-written RTL blocks.
6. Extend the vector exporter as new hardware blocks are added, especially
   RoPE, KV cache append/read, attention, MLP, and complete Layer 0 cached
   decode.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Treat BF16/FP32 model data as software reference and validation input only.
  The first PL DDR4 weight layout and GEMV datapaths must assume custom Q4
  weight-only storage for large weights.
- Treat `Vitis_Workspace/` as a local regenerated workspace. Keep durable
  application source under `FPGA_Project/software/` and recreate Vitis
  platform/application/build outputs from the Vivado hardware export.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- Keep `CURRENT_STATE.md` concise. Store detailed validation logic in scripts
  and summarize only durable results here.
