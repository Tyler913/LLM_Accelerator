# QMAP one-token runtime skeleton

This directory is the first PS-side source for the `qmap_one_token_axil_top.sv`
control boundary. It is intentionally small and does not replace the existing
DDR/QMAP smoke apps.

Files:

- `qmap_one_token_regs.h` mirrors the byte offsets, status bits, control bits,
  and per-layer QMAP table ids from `FPGA_Project/rtl/top/one_token/qmap_one_token_control_regs.sv`.
- `qmap_one_token_runtime.h` provides header-only helpers for 32/64-bit AXI-Lite
  register access, table commits, run configuration, start/poll/result readback,
  optional tied-Q4 embedding configuration, and a no-memory validation smoke.
- `main.c` is a Vitis app skeleton for the no-memory validation smoke. Define
  `QOT_BASEADDR` from the Vivado/Vitis `xparameters.h` address assigned to the
  one-token AXI-Lite slave before running it on board.

The no-memory validation smoke intentionally starts `layer_count=0`. A passing
hardware run should reach done+error, report layer-0 validation error, and keep
all memory request/write counters at zero. This checks the AXI-Lite register
seam before any model artifacts are loaded into PL DDR.

A real model run will need a PS loader/manifest step to fill a 28-entry
`qot_layer_qmap_bases_t` table with physical PL-DDR QMAP packet addresses, then
call `qot_configure_run()`, `qot_start()`, `qot_poll_done()`, and
`qot_read_result()`. Set `embedding_enable`, `embedding_weight_base`, and
`embedding_scale_base` in `qot_run_config_t` to make `input_token_id` generate
the input hidden vector in PL before Layer 0. The no-memory smoke explicitly
disables embedding so stale configuration cannot issue memory traffic.
