# RTL Source Organization

This directory contains the durable hand-written Verilog/SystemVerilog source
for the PL accelerator. Module names and public ports are independent of the
directory layout.

## Directory Layers

```text
rtl/
  include/                 shared preprocessor definitions
  lib/
    bus/                   AXI4 and AXI4-Lite adapters
    math/                  reusable fixed-point arithmetic
    q4/                    Q4 unpack, embedding, dot-product, and GEMV cores
    vector/                generic fixed-size vector operations
  model/
    norm/                  RMSNorm primitives and final-norm stage
    attention/             Q/K norm, RoPE, KV cache, attention, and o_proj
    mlp/                   gate/up, SiLU, down, and residual stages
    lm_head/               tiled tied-LM-head scan and argmax
  qmap/
    protocol/              QMAP header/descriptor readers and payload fetchers
    compute/               descriptor-backed model compute paths
    scheduler/             per-layer and one-token layer schedulers
  top/
    one_token/             control registers and one-token integration tops
    smoke/                 bounded board bring-up wrappers
```

The dependency direction is intentionally one-way:

```text
include/lib -> model -> qmap/protocol+compute -> qmap/scheduler -> top
```

## Canonical Build Inputs

- `rtl_sources.list` is the only authoritative list of synthesizable `.sv` and
  `.v` files. Entries are relative to this directory and dependency ordered.
- `include_dirs.list` is the authoritative include-directory list.
- `FPGA_Project/sim/load_rtl_manifest.ps1` validates both lists, rejects
  duplicate or missing entries, and requires the source manifest to cover the
  complete RTL source tree exactly.

The formal PowerShell regressions and the Vivado BD scaffold consume these
manifests. Do not add a new RTL source only to an individual test command or
Vivado project snapshot.

When adding or moving RTL:

1. Put the file in the narrowest matching module directory.
2. Add or move its entry in `rtl_sources.list`.
3. Add any public header directory to `include_dirs.list`.
4. Run `FPGA_Project/sim/run_one_token_local_regression.ps1` first.
5. Run the integration regression whose datapath is affected.
6. Sync the Vivado source set through the tracked Tcl flow before synthesis.

Generated simulator/Vivado products are not source truth and must not be
copied back into this tree.
