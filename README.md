# LLM_Accelerator

FPGA-based LLM accelerator project inspired by Hummingbird+.

Latest hardware checkpoint: the QMAP row1024 PL AXI-master smoke path passed
on board on 2026-06-23. Start with
[Source/CURRENT_STATE.md](Source/CURRENT_STATE.md) for the current handoff and
next step.

Latest local RTL checkpoint: Layer 0 has passed through attention,
post-attention residual/RMSNorm, MLP gate/up projection, MLP SiLU/multiply,
MLP down projection, and final MLP residual add in Icarus. The full-model
current-token final RMSNorm stage, the tiled Q4 LM-head argmax core, the
memory-backed LM-head tile reader/wrapper for a 1024-row scan window, and a
runtime tile scheduler plus QMAP descriptor-backed wrapper for multi-window
LM-head scans also pass locally.

Start here for persistent project context:

- [Source/PROJECT_CONTEXT.md](Source/PROJECT_CONTEXT.md)
- [Source/CURRENT_STATE.md](Source/CURRENT_STATE.md)
- [Source/FPGA_MEMORY_MAP.md](Source/FPGA_MEMORY_MAP.md)
- [Source/Q4_FORMAT.md](Source/Q4_FORMAT.md)
- [Source/QMAP_FORMAT.md](Source/QMAP_FORMAT.md)
- [Source/AGENTS.md](Source/AGENTS.md)

For cross-platform setup and model asset restoration, start here:

- [init/CROSS_PLATFORM_SYNC.md](init/CROSS_PLATFORM_SYNC.md)

Large model weights are intentionally not stored in normal Git history. Use the
`init/` runbook and scripts to restore `Qwen3-0.6B-Base/model.safetensors` on a
fresh macOS, Linux, or Windows machine.
