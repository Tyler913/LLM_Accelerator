# LLM Accelerator

Hand-written RTL and board software for a Qwen3-0.6B FPGA inference demo on an
XCZU2EG MPSoC. The deployed path stores large model weights in PL DDR4 using
the project's custom Q4 weight-only format; the PL executes embedding, all 28
decoder layers, KV-cache operations, final RMSNorm, and greedy LM-head argmax.

## Project Status

The original bench accepted the complete JTAG-loaded demo:

- 2026-08-08: full28 persistent two-token board smoke;
- 2026-08-12: arbitrary prompt-to-text through UART; and
- 2026-08-17: board-hosted QWEB prompt-to-text.

The portable package is
[`lmdeploy/qwen3-0p6b-q4-qweb-demo/`](lmdeploy/qwen3-0p6b-q4-qweb-demo/). Its
contents have passed the local integrity audit, but its new relative-path
launcher still needs one physical-board rerun before it can be called
board-accepted. Details and limits are in
[Source/CURRENT_STATE.md](Source/CURRENT_STATE.md).

## Quick Start

Python commands must use the `llm_fpga` Conda environment.

```bash
git clone <repository-url> LLM_Accelerator
cd LLM_Accelerator
conda run -n llm_fpga python init/download_q4_runtime.py
conda run -n llm_fpga python init/verify_q4_runtime.py
```

On a Windows machine with AMD Vitis/XSDB and the target board, audit the
portable demo before a hardware launch:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\lmdeploy\qwen3-0p6b-q4-qweb-demo\run_demo.ps1 -AuditOnly
```

The full board prerequisites, UART capture, and Web UI procedure are in the
[portable demo README](lmdeploy/qwen3-0p6b-q4-qweb-demo/README.md).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `FPGA_Project/` | Canonical RTL, Vivado project, board software, simulators, and versioned golden test vectors. |
| `Qwen3-0.6B-Base/` | Qwen configuration/tokenizer metadata plus Python reference, export, and validation tools. The model weights are external. |
| `lmdeploy/qwen3-0p6b-q4-qweb-demo/` | Portable release candidate: manifests, launcher, selected board artifacts, and Q4 runtime contract. |
| `init/` | Download, pin, and verification utilities for external assets. |
| `Source/` | Current status, architecture, Q4/QMAP specifications, memory map, and validation records. |

`FPGA_Project/` is the only canonical implementation tree. Do not modify a
generated release archive or build directory as if it were canonical source.

## Development References

- [Current status](Source/CURRENT_STATE.md)
- [Project architecture and constraints](Source/PROJECT_CONTEXT.md)
- [Q4 numerical contract](Source/Q4_FORMAT.md)
- [QMAP descriptor contract](Source/QMAP_FORMAT.md)
- [PL DDR4 memory map](Source/FPGA_MEMORY_MAP.md)