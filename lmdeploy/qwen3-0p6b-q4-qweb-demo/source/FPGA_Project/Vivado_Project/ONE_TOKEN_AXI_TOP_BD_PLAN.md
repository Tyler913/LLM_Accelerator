# Qwen3 one-token AXI-Lite / AXI4 board integration

Last updated: 2026-08-08

Status: the one-token accelerator is integrated into `llm_system.bd`, and the
current board design has completed synthesis, implementation, routing,
bitstream generation, fixed XSA export, Vitis application builds, the full28
persistent two-token XSim/audit, self-contained package verification, and the
physical full28 fixed two-token board smoke. The previous row1024 design
remains valuable hardware history, but it is no longer the latest hardware
milestone or the current block-design contents.

The current board design is intentionally resource-reduced, not
function-reduced:

- one reusable Q4 group lane for the projection datapaths;
- one reusable LM-head row lane;
- BRAM-backed QMAP base tables and large matrix/activation storage;
- the same 28-layer QMAP packet contract, fixed-point math, KV-cache layout,
  full `151936`-row LM-head scan, and token/score results as the local model
  reference.

## Current block design

Checked-in BD:
`FPGA_Project/Vivado_Project/LLM_FPGA.srcs/sources_1/bd/llm_system/llm_system.bd`.

Accelerator module-reference seam:
`FPGA_Project/rtl/top/one_token/qmap_one_token_axi_bd.v`.

The wrapper instantiates `qmap_one_token_axi_top.sv` and exposes:

- `S_AXI`: 32-bit AXI4-Lite slave for software configuration, launch, status,
  counters, and result readback;
- `M_AXI`: 64-bit-address / 32-bit-data AXI4 master for QMAP, Q4, activation,
  KV-cache, RoPE, and result traffic in PL DDR4.

Current address map:

| Purpose | Base | High | Size |
| --- | ---: | ---: | ---: |
| DDR4 status GPIO | `0xA001_0000` | `0xA001_FFFF` | 64 KiB |
| Qwen one-token control | `0xA004_0000` | `0xA004_FFFF` | 64 KiB |
| PL DDR4 | `0x4_0000_0000` | `0x4_1FFF_FFFF` | 512 MiB |

The old AXI BRAM aperture and row1024 control/result GPIO apertures at
`0xA000_0000`, `0xA002_0000`, and `0xA003_0000` are historical only; they are
not present in the current accelerator BD.

Fabric:

```text
PS M_AXI_HPM0_FPD
  -> AXI SmartConnect
      -> DDR4 status GPIO
      -> qmap_one_token_axi_bd/S_AXI
      -> AXI clock converter -> PL DDR4

qmap_one_token_axi_bd/M_AXI
  -> AXI SmartConnect
      -> AXI clock converter -> PL DDR4
```

The accelerator and PS-facing fabric run from the current `pl_clk0` clock
(about 96.97 MHz). The DDR4 status word remains:

```text
bit 0 = c0_init_calib_complete
bit 1 = c0_ddr4_ui_clk_sync_rst
bit 2 = peripheral_aresetn
good value = 0x5
```

## Reproducible Vivado build

Use:

```powershell
vivado -mode batch `
  -source FPGA_Project/Vivado_Project/scripts/build_one_token_board.tcl `
  -tclargs --jobs=4 --out-dir=<Temp output directory>
```

The build script:

1. validates and regenerates the BD;
2. disables stale incremental synthesis reuse;
3. resets the accelerator OOC child and top synthesis;
4. resets implementation even when a current synthesis is reused;
5. implements through `write_bitstream`;
6. emits post-route timing, routing, utilization, DRC, methodology, clock, and
   power reports;
7. copies the bitstream and exports a fixed XSA containing that exact bitstream.

Current routed candidate:
`Temp/boardready_board_build_20260726/full_board_impl_v8_current/`.

Release numbers:

| Gate | Result |
| --- | ---: |
| CLB LUT | `43,726 / 47,232` (`92.58%`) |
| CLB registers | `72,621 / 94,464` (`76.88%`) |
| Block RAM tiles | `94.5 / 150` (`63.00%`) |
| DSP | `68 / 240` (`28.33%`) |
| WNS | `+0.208 ns` |
| WHS | `+0.010 ns` |
| WPWS | `+0.039 ns` |
| routed nets | `119,383 / 119,383` |
| routing errors | `0` |
| DRC Error / Critical Warning | `0 / 0` |
| methodology Error / Critical Warning | `0 / 0` |
| register/latch pins without clock | `0` |
| unconstrained internal endpoints | `0` |
| accepted asynchronous output-delay exception | `C0_DDR4_0_reset_n` only |

Warning-only rules are retained in `board_readiness_audit.json` by name and
count. The release script permits only the reviewed DSP pipeline/reset,
debug-hub/MIG asynchronous-reset, inferred-BRAM write-enable, control-set, and
unused AXI-converter-load advisory categories present in this implementation;
any new warning rule blocks a future package until reviewed.

The LUT margin is small. The first full28 hardware run now passes, but do not
make speculative parallelism or debug-logic increases; any RTL change must
repeat the complete implementation and board-validation gates.

## Runtime image and software

The board runtime is generated as 61 non-overlapping binary segments:

- total loaded bytes: `394,547,200`;
- first loaded address: `0x4_0010_0000`;
- last end-exclusive address: `0x4_1A14_0000`;
- QMAP headers checked by the board launcher: `281`;
- accelerator-writable regions required to be zero before launch: `397`.

The generated XSDB loader must contain exactly one
`dow -data -bypass-cache-sync` command for every segment. PL-DDR physical
writes occur while Cortex-A53 #0 is halted, so the debugger must not perform
A53 cache maintenance for those writes. The packaged launcher's default FPGA
device filter is `name =~ "PL"`; `QOT_DEVICE_FILTER` remains available as an
explicit override for a multi-device JTAG chain.

The zero-region rule covers every stage output, both hidden buffers, all 28
layers' KV-cache storage, and the final-tail scratch/output. It prevents a
preloaded golden output from hiding a compute or write-back failure.

Durable PS-side source:
`FPGA_Project/software/qmap_one_token_runtime/`.

The reproducible short-path Vitis workspace is `F:\vws`:

| Component | Purpose |
| --- | --- |
| `p_qot` | standalone Cortex-A53 platform from the current fixed XSA |
| `a_qctl` | AXI-Lite no-memory validation smoke |
| `a_qmdl` | full28 persistent two-token model smoke |

Both launches use FSBL and `runPsuInit=false`. The model launch stops at entry
so the PL-DDR runtime can be loaded before the application runs.

## Board launch order

The release package is created by
`FPGA_Project/software/qmap_one_token_runtime/package_board_release.py`.
It refuses to package a candidate unless the implementation, XSA/bitstream,
runtime image, Vitis workspace, and full28 persistent regression all pass.

The packaged `launch_qwen3_board.tcl` performs:

1. system reset and bitstream programming;
2. XSA memory-map load;
3. FSBL execution through `XFsbl_Exit`;
4. PL DDR4 calibration polling for status `0x5`;
5. all 61 runtime segment downloads using `-bypass-cache-sync`;
6. all 281 QMAP header checks;
7. control or model ELF download and launch.

Always test in this order:

```powershell
.\run_board_smoke.ps1 -Mode control
.\run_board_smoke.ps1 -Mode model
```

Control acceptance:

```text
PASS qot_run_no_memory_validation_smoke
```

Model acceptance:

```text
PASS token position=0 output=28458 score=1227344433
PASS token position=1 output=64 score=1015661901
PASS Qwen3-0.6B full28 persistent two-token board smoke
```

Only those UART results establish a hardware PASS. Successful routing,
bitstream generation, packaging, or local simulation establishes readiness to
test, not board validation.

## 2026-08-08 Physical Board Closure

The physical model run completed the full acceptance contract:

- loader: `61/61` runtime segments downloaded successfully;
- QMAP preflight: `281/281` packet headers returned the expected magic;
- position 0: `374 -> 28458`, signed Q26 score `1227344433`;
- position 1: `28458 -> 64`, signed Q26 score `1015661901`;
- both steps: `done=1`, `err=0`, `28/28` layers started and completed,
  `done_mask=0x0fffffff`, `error_mask=0`, and no reported top/tail/command
  error;
- final UART result:
  `PASS Qwen3-0.6B full28 persistent two-token board smoke`.

This closes the fixed two-token board smoke for the current routed design. It
does not close an arbitrary user-text interface: the board application still
uses fixed token IDs and does not include tokenizer, detokenizer, EOS/stop,
sampling, or arbitrary-length prompt-to-text integration.

## Local release gates closed

The resource-reduced full28 no-reset two-token XSim passed in
`Temp/boardready_full28_persistent_regression_v6_20260801/20260801_181607`.
It produced token/score `28458/1227344433` then `64/1015661901`, with one reset
release. The independent event audit proves exact per-stage writes, all 281
QMAP packets per token, and that position 1 reads every layer's retained
position-0 K/V only after position-0 writes complete.

The self-contained release `Temp/boardready_qwen3_full28_20260801` then passed
its 92-file inventory, size, SHA256, and board-readme semantic verifier. Its
recorded state remains `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED` because
that is the historical pre-board readiness result. Physical validation was
subsequently completed on 2026-08-08 using control mode followed by model mode;
the preserved UART evidence, not a rewrite of the old readiness state, is the
board-PASS record.
