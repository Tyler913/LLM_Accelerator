# One-token AXI-Lite / AXI4 Vivado integration plan

Status: local RTL seam is ready for BD planning, but the checked-in Vivado block
design still contains the row1024 smoke design. Do not replace the proven
row1024 bitstream until the one-token seam passes the staged gates below.

## Current BD baseline

Current checked-in BD: `LLM_FPGA.srcs/sources_1/bd/llm_system/llm_system.bd`.

Known PS-visible apertures in the current design:

| Purpose | Address | Range | Status |
| --- | ---: | ---: | --- |
| AXI BRAM smoke | `0xA000_0000` | `8 KiB` | current BD |
| DDR4 status GPIO | `0xA001_0000` | `64 KiB` | current BD |
| QMAP row1024 smoke control/status GPIO | `0xA002_0000` | `64 KiB` | current BD |
| QMAP row1024 smoke result GPIO | `0xA003_0000` | `64 KiB` | current BD |
| PL DDR4 memory map | `0x4_0000_0000` | `512 MiB` | current BD |

Recommended first one-token register aperture: `0xA004_0000`, range `64 KiB`.
The RTL register map only consumes the low 4 KiB (`AXI_ADDR_WIDTH=12`), but the
64 KiB BD aperture matches the surrounding smoke-test address granularity and
keeps `xparameters.h` simple.

## New RTL seam

`FPGA_Project/rtl/qmap_one_token_axi_top.sv` is the BD-facing shell. It exposes:

- `S_AXI`: 32-bit AXI4-Lite slave for `qmap_one_token_control_regs.sv`.
- `M_AXI`: 32-bit AXI4 master for PL-DDR reads/writes, using the existing
  `axi4_read_master.sv` and `axi4_write_master.sv` adapters.
- Debug/status pins (`o_busy`, `o_done`, `o_error`, `o_mem_error`, counters,
  token/score low word) that may be left internal initially or routed to debug
  GPIO/ILA later.

The wrapper is intentionally policy-free: register semantics stay in
`qmap_one_token_control_regs.sv`, AXI4-Lite protocol handling stays in
`axi4lite_to_mmio_regs.sv`, and compute sequencing stays in
`qmap_one_token_top.sv`.

## Scripted scaffold

A safe-by-default Tcl scaffold is available at
`scripts/one_token_axi_top_bd_scaffold.tcl`.

Dry-run only, no BD edits:

```tcl
vivado -mode batch -source FPGA_Project/Vivado_Project/scripts/one_token_axi_top_bd_scaffold.tcl
```

Apply and validate when ready:

```tcl
vivado -mode batch -source FPGA_Project/Vivado_Project/scripts/one_token_axi_top_bd_scaffold.tcl -tclargs --apply --validate
```

The dry-run path has been executed under Vivado 2024.2 and prints the planned
edits without opening/saving the current BD. The `--apply` path still needs a
human review of `llm_system.bd`, synthesis/resource utilization, and address map
before implementation or XSA export.

## Suggested BD wiring sequence

1. Add/sync RTL sources.
   - Include `FPGA_Project/rtl/qmap_one_token_axi_top.sv` plus the existing RTL
     dependencies already used by local xsim.
   - Set `qmap_one_token_axi_top` as a module-reference BD cell, or package it as
     custom IP only after the module-reference path validates.
2. Clock/reset.
   - Connect `aclk` to the same PL/DDR fabric clock domain used by the current
     row1024 PL master smoke path.
   - Connect `aresetn` to the corresponding active-low synchronized reset.
3. Control path.
   - Connect PS `M_AXI_HPM*_FPD` (through the existing/intermediate AXI
     interconnect or SmartConnect) to `qmap_one_token_axi_top/S_AXI`.
   - Assign `S_AXI` base `0xA004_0000`, range `64 KiB`.
4. Memory path.
   - Connect `qmap_one_token_axi_top/M_AXI` to the PL DDR4 slave path currently
     used by the row1024 PL master smoke.
   - Preserve PL DDR4 base `0x4_0000_0000`, range `512 MiB` for first bring-up.
5. Validation/build gates.
   - Validate BD.
   - Generate wrapper.
   - Synthesis first; do not launch implementation until synthesis/resource
     utilization is reviewed.
   - Export XSA only after synthesis and address-map inspection are sane.
6. Vitis no-memory smoke.
   - Build `FPGA_Project/software/qmap_one_token_runtime/main.c`.
   - Define `QOT_BASEADDR` from the generated `xparameters.h` symbol or compiler
     flags if the symbol name differs.
   - Run the no-memory validation smoke before loading model artifacts into PL
     DDR. A pass is done+error with layer0 error and all memory counters zero.

## Required local regressions before board-facing changes

- `tb_axi4lite_to_mmio_regs.sv` Icarus smoke.
- `tb_qmap_one_token_axil_top.sv` Icarus no-memory smoke.
- `qmap_one_token_axil_l1_l2_tail_xsim` positive Layer1 -> Layer2 -> tail xsim.
- `qmap_one_token_axil_true3_tail_mixed_xsim` bounded mixed true3 top-to-tail
  xsim.
- `tb_qmap_one_token_axi_top.sv` Icarus no-memory smoke for the BD-facing
  `S_AXI`/`M_AXI` shell.
- `one_token_axi_top_bd_scaffold.tcl` Vivado dry-run before any `--apply` run.
- Host syntax check for `FPGA_Project/software/qmap_one_token_runtime/main.c`.
- Header/RTL register-map consistency check for `qmap_one_token_regs.h`.

## Open items before real model runtime

- Full model artifact manifest and PS loader for all 28 layer QMAP packet bases.
- Cache coherency policy for PS writes into PL DDR and PL writes read back by PS.
- Whether to keep direct PS memory-mapped 32-bit artifact copies or add DMA.
- Layer 0 input-RMSNorm full-chain artifacts if the project wants to remove the
  current QKV-first Layer 0 baseline.
