# QMAP one-token and persistent-decode runtime

This directory is the durable PS-side source for
`qmap_one_token_axil_top.sv`. It now supports both the small AXI-Lite
no-memory validation smoke and the Qwen3-0.6B full28 persistent two-token
board smoke.

## Runtime sources

- `qmap_one_token_regs.h` mirrors the RTL byte offsets, status/control bits,
  and per-layer QMAP table identifiers.
- `qmap_one_token_runtime.h` provides header-only AXI-Lite configuration,
  table-commit, launch, polling, result-readback, and persistent decode helpers.
- `qmap_model_config_generated.h` is the 28-layer physical-address table
  generated from the current full-chain runtime manifest.
- `main.c` builds in two modes:
  - default: `layer_count=0` no-memory validation of the control seam;
  - `QOT_MODEL_BOARD_SMOKE=1`: token `374` at position 0, followed by feedback
    token `28458` at position 1 without clearing QMAP tables or KV cache.
- `test_runtime_host.c` exercises the software contract without Vitis hardware.

## Reproducible Vitis build

`create_vitis_workspace.py` is run by Vitis Python, not ordinary CPython. It
creates a short-path standalone A53 platform and two applications:

- `a_qctl`: no-memory AXI-Lite control smoke;
- `a_qmdl`: full28 persistent two-token model smoke.

Set `QOT_XSA` to the routed board XSA and optionally set
`QOT_VITIS_WORKSPACE` (default `F:\vws`). The generated launches use FSBL,
keep `runPsuInit=false`, and stop the model app at entry so the PL-DDR runtime
image can be loaded first.

## Board-test release

- `launch_qwen3_board.tcl` performs the deterministic XSDB sequence: system
  reset, bitstream programming, XSA memory-map load, FSBL, PL DDR4 calibration
  polling, 61-segment runtime load, all 281 QMAP-header checks, ELF download,
  and application start.
- `run_board_smoke.ps1` invokes that launcher in `control` or `model` mode.
- `package_board_release.py` refuses to create a release unless post-route
  timing/routing/DRC, the complete runtime image, and the full28 persistent
  two-token XSim audit all pass. It also proves that all 397 accelerator-writable
  packet/cache/hidden/tail regions are zero-initialized, so board output cannot
  be satisfied by preloaded golden scratch data. It additionally compares all
  280 per-layer QMAP bases and all eight global model-address constants compiled
  into the model app against the runtime's exact full-chain manifest. It emits
  a self-contained package with per-file SHA256 values and an explicit
  `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED` state.
- `verify_board_release.py` independently rehashes the complete packaged
  inventory before the package is used at the bench. It also rejects malformed
  board-readme control characters or missing verify/control/model commands.

## Current verified release

The 2026-08-01 release uses regression
`Temp/boardready_full28_persistent_regression_v6_20260801/20260801_181607` and
package `Temp/boardready_qwen3_full28_20260801`. XSim and the independent
persistent event audit both passed; the package verifier then checked 92 files,
61 copied runtime segments, every size/SHA256, and the board commands. Its state
is `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED`.

From the package directory, the next commands are:

```powershell
.\run_board_smoke.ps1 -Mode control
.\run_board_smoke.ps1 -Mode model
```

Run them in that order and preserve the UART log.

The model UART contract is:

```text
PASS token position=0 output=28458 score=1227344433
PASS token position=1 output=64 score=1015661901
PASS Qwen3-0.6B full28 persistent two-token board smoke
```

These lines are the hardware acceptance criterion; producing a bitstream or
package alone is not a board PASS.
