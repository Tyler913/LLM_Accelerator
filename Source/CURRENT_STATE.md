# Current Release Status

Last updated: 2026-08-26

This is the authoritative current handoff for the repository.
`Source/PROJECT_CONTEXT.md` contains durable architecture and workflow
constraints; `Source/RELEASE_VALIDATION.md` contains the release-facing
validation summary. Detailed historical evidence is intentionally excluded from
the release tree.

## Release Summary

The correctness-first, JTAG-loaded Qwen3-0.6B FPGA demo is complete for its
defined first-version scope:

- Hardware: XCZU2EG MPSoC with 512 MiB PL DDR4 and PS Ethernet (GEM3).
- Model path: tied custom Q4 weights, serial prefill, cached single-token
  decode, greedy argmax, and a 256-token context.
- PL owns embedding, all 28 decoder layers, KV cache, final RMSNorm, and the
  vocabulary scan. PS performs loading, control, tokenization, detokenization,
  and network/UART orchestration.
- RTL is hand-written Verilog/SystemVerilog; BF16/FP32 model data is software
  reference data, not the deployed PL weight path.

The original physical validation closed three gates:

| Date | Gate | Result |
| --- | --- | --- |
| 2026-08-08 | Full28 persistent two-token board smoke | PASS |
| 2026-08-12 | UART arbitrary prompt-to-text | PASS |
| 2026-08-17 | Board-hosted QWEB prompt-to-text | PASS |

The portable package in `lmdeploy/qwen3-0p6b-q4-qweb-demo/` has passed its
local file-integrity audit. It contains hash-identical accepted inputs, but its
relative-path launcher has **not** yet received a new physical-board run. Call
it a release candidate until that one revalidation passes.

## Canonical Locations

- `FPGA_Project/` is the only canonical RTL, Vivado, software, and simulation
  source tree.
- `Qwen3-0.6B-Base/` contains the upstream configuration/tokenizer metadata
  and project Python reference/export tools. `model.safetensors` is restored
  locally and is never committed.
- `lmdeploy/qwen3-0p6b-q4-qweb-demo/` contains the portable board package,
  manifests, launch support, and pinned demo artifacts.
- `init/` restores and verifies external model assets.

## Reproducibility Commands

Every Python command must use the project Conda environment:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

On Windows, audit the portable board package before attempting a hardware run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 -AuditOnly
```

The model and Q4 runtime download sources must remain pinned to exact revisions
before a public reproducible release is claimed.

## Release Boundaries

- Never commit model weights, Q4 runtime segments, temporary exports, Vivado
  workspaces, implementation products, or simulator traces.
- The checked-in simulation vectors are versioned golden test fixtures. They
  remain tracked until an external artifact store plus fetch-and-verify command
  replaces their direct testbench dependencies.
- The selected bitstream, XSA, ELF files, and tokenizer asset are pinned demo
  deliverables. Moving them to GitHub Release assets requires a matching
  manifest/launcher update and a fresh physical-board validation.
- The paper PDF is retained locally for reference. Verify redistribution rights
  before publishing it in a public repository.

## Next Release Gates

1. Run the portable package once on the physical board and preserve the new
   evidence; only then promote it from release candidate to board-accepted.
2. Choose and add a root project license plus third-party notices, including
   Qwen attribution and the paper's redistribution status.
3. Decide whether public model restoration uses a public pinned upstream
   revision or remains an authenticated/private workflow.

For architecture and durable constraints, read `Source/PROJECT_CONTEXT.md`.
For the current Q4 and QMAP contracts, read `Source/Q4_FORMAT.md` and
`Source/QMAP_FORMAT.md`.
