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
  - `QOT_MODEL_BOARD_SMOKE=1`: fixed token `374` at position 0, followed by
    fixed token `28458` at position 1 without clearing QMAP tables or KV cache.
    The second input equals the verified first output, but this smoke does not
    exercise an autonomous PL token-feedback loop.
- `test_runtime_host.c` exercises the software contract without Vitis hardware.

## Reproducible Vitis build

`create_vitis_workspace.py` is run by Vitis Python, not ordinary CPython. It
creates a short-path standalone A53 platform and three applications:

- `a_qctl`: no-memory AXI-Lite control smoke;
- `a_qmdl`: full28 persistent two-token model smoke.
- `a_qgen`: bounded interactive UTF-8/text or token-ID prompt/decode
  application. It embeds the deterministic PS-native Qwen tokenizer asset,
  uses durable sources from `../qmap_prompt_demo/`, and leaves both smoke apps
  intact.

Set `QOT_XSA` to the routed board XSA, set `QOT_TOKENIZER_ASSET` to the verified
`qwen3_tokenizer.qtk`, and optionally set `QOT_VITIS_WORKSPACE` (default
`F:\vws`). The generated launches use FSBL,
keep `runPsuInit=false`, and stop the model app at entry so the PL-DDR runtime
image can be loaded first.

## Board-test release

- `launch_qwen3_board.tcl` performs the deterministic XSDB sequence: system
  reset, bitstream programming, XSA memory-map load, FSBL, PL DDR4 calibration
  polling, 61-segment runtime load, all 281 QMAP-header checks, ELF download,
  and application start. Its FPGA-device filter defaults to `name =~ "PL"` and
  can be overridden with `QOT_DEVICE_FILTER` for a multi-device JTAG chain.
- Every PL-DDR segment is loaded with
  `dow -data -bypass-cache-sync <file> <physical-address>`. The bypass is a
  required contract, not an optional speed setting: PL-DDR writes must not
  trigger Cortex-A53 cache flush/invalidate operations.
- `run_board_smoke.ps1` invokes that launcher in `control`, `model`, or
  `generate` mode.
- `package_board_release.py` refuses to create a release unless post-route
  timing/routing/DRC, the complete runtime image, and the full28 persistent
  two-token XSim audit all pass. It also proves that all 397 accelerator-writable
  packet/cache/hidden/tail regions are zero-initialized, so board output cannot
  be satisfied by preloaded golden scratch data. It additionally compares all
  280 per-layer QMAP bases and all eight global model-address constants compiled
  into the model app against the runtime's exact full-chain manifest. It emits
  a self-contained format-v2 package with per-file SHA256 values and an explicit
  `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED` state. All four packaged ELFs
  must hash-identically match the audited Vitis workspace build outputs. Release
  assembly occurs in a same-parent staging directory and only replaces an
  existing recognized release after the new package is complete. The generation
  audit also requires the exact seven-source `a_qgen` compile list and the fixed
  Qwen3 tokenizer asset (`3,629,566` bytes, SHA256
  `c20242603ef4144e3f3f2ec4ba97c0e9c315aadd41f1bd2c5740e2a7ffa03a7d`).
- `verify_board_release.py` independently rehashes the complete packaged
  inventory before the package is used at the bench. It strictly validates
  manifest paths, counts, byte totals, and hashes. It accepts historical
  format-v1 control/model packages, while format v2 additionally requires the
  audited `a_qgen` ELF and generate command.

## Current verified board milestone

The preserved 2026-08-01 pre-board release uses regression
`Temp/boardready_full28_persistent_regression_v6_20260801/20260801_181607` and
package `Temp/boardready_qwen3_full28_20260801`. XSim and the independent
persistent event audit both passed; the package verifier then checked 92 files,
61 copied runtime segments, every size/SHA256, and the board commands. Its state
is `BOARD_TEST_READY_NOT_YET_HARDWARE_VALIDATED`. Keep that state in the
historical package: it records what was known before physical execution and is
not intended to be rewritten after the fact.

On 2026-08-08 the corresponding physical full28 flow passed. The launcher used
the cache-bypass download contract, loaded `61/61` runtime segments, verified
`281/281` QMAP headers, and then started the model application. The control
smoke had already passed before the model run.

From the package directory, the reproducible commands are:

```powershell
.\run_board_smoke.ps1 -Mode control
.\run_board_smoke.ps1 -Mode model
```

Run them in that order and preserve the UART log. With the default launcher no
extra device variable is needed when the FPGA target is uniquely selected by
`name =~ "PL"`; override `QOT_DEVICE_FILTER` only when required by the JTAG
chain.

The model UART contract is:

```text
PASS token position=0 output=28458 score=1227344433
PASS token position=1 output=64 score=1015661901
PASS Qwen3-0.6B full28 persistent two-token board smoke
```

These lines are the hardware acceptance criterion; producing a bitstream or
package alone is not a board PASS.

The 2026-08-08 UART also reported the following completion contract for both
positions:

| Position | Input | Output | Signed Q26 score | Status | Layers | Done mask | Error mask |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `0` | `374` | `28458` | `1227344433` | `done=1, err=0` | `28/28` | `0x0fffffff` | `0` |
| `1` | `28458` | `64` | `1015661901` | `done=1, err=0` | `28/28` | `0x0fffffff` | `0` |

This remains the fixed two-token hardware validation baseline. A later
2026-08-09 `a_qgen` build adds PS-side tokenization, detokenization, EOS/stop
handling, and bounded dynamic feedback, but those new interactive paths still
require their own UART board acceptance before they inherit hardware-PASS
status. Sampling and arbitrary-length generation remain outside the scope.
