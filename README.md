# LLM_Accelerator

FPGA-based LLM accelerator project inspired by Hummingbird+.

Start here for persistent project context:

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
- [CURRENT_STATE.md](CURRENT_STATE.md)
- [Q4_FORMAT.md](Q4_FORMAT.md)

For cross-platform setup and model asset restoration, start here:

- [init/CROSS_PLATFORM_SYNC.md](init/CROSS_PLATFORM_SYNC.md)

Large model weights are intentionally not stored in normal Git history. Use the
`init/` runbook and scripts to restore `Qwen3-0.6B-Base/model.safetensors` on a
fresh macOS, Linux, or Windows machine.
