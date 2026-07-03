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
LM-head scans also pass locally. The same QMAP LM-head wrapper now also passes
a full-vocabulary `151936`-row / `9496`-tile scan in Vivado xsim. The first
memory-mapped one-token tail wrapper now composes final RMSNorm write-back plus
full-vocabulary QMAP LM-head argmax locally. The first memory-mapped Layer 0
attention front-end wrapper now also passes locally: it reads Q/K/V projection
outputs plus q/k gamma and RoPE tables from a QMAP packet, writes exact K/V
cache entries, and writes exact Q RoPE output for the current token. The next
QMAP attention score/value wrapper also passes locally: it consumes Q RoPE plus
K/V cache reads, streams score into softmax/value, and writes exact
`attn_out[2048]`. The QMAP `o_proj` wrapper now also passes locally: it
consumes `attn_out[2048]`, reads persistent Layer 0 Q4 `o_proj`
weight/scale rows, and writes exact `o_proj_out[1024]`. The QMAP
post-attention residual/RMSNorm wrapper now also passes locally: it consumes
descriptor-visible residual input, `o_proj_out[1024]`, and signed
post-attention gamma, then writes exact post-attention hidden and post-norm
buffers. The QMAP MLP gate/up wrapper now also passes locally: it consumes
descriptor-visible `post_norm[1024]`, reads persistent Layer 0 Q4 gate/up
weight/scale rows, and writes exact gate/up `[3072]` buffers. The QMAP MLP
SiLU/multiply wrapper now also passes locally: it consumes descriptor-visible
gate/up `[3072]` plus a sigmoid LUT and writes exact `mlp_hidden[3072]`.
The QMAP MLP down wrapper now also passes locally: it consumes
descriptor-visible `mlp_hidden[3072]`, reads persistent Layer 0 Q4 down-proj
weight/scale rows, and writes exact `down_out[1024]`. The QMAP final MLP
residual wrapper now also passes locally: it consumes descriptor-visible
`post_attn_hidden[1024]` and `down_out[1024]`, writes exact
`layer_out[1024]`, and catches descriptor/protocol error paths with no writes.

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
