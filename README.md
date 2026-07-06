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
consumes `attn_out[2048]`, reads persistent Q4 `o_proj` weight/scale rows,
and writes exact `o_proj_out[1024]` for the Layer 0 and true Layer 1 packet
instances. The QMAP
post-attention residual/RMSNorm wrapper now also passes locally: it consumes
descriptor-visible residual input, `o_proj_out[1024]`, and signed
post-attention gamma, then writes exact post-attention hidden and post-norm
buffers for the Layer 0 and true Layer 1 packet instances. The QMAP MLP
gate/up wrapper now also passes locally: it consumes
descriptor-visible `post_norm[1024]`, reads persistent Q4 gate/up
weight/scale rows, and writes exact gate/up `[3072]` buffers for the Layer 0
and true Layer 1 packet instances. The QMAP MLP
SiLU/multiply wrapper now also passes locally: it consumes descriptor-visible
gate/up `[3072]` plus a sigmoid LUT and writes exact `mlp_hidden[3072]` for
the Layer 0 and true Layer 1 packet instances.
The QMAP MLP down wrapper now also passes locally: it consumes
descriptor-visible `mlp_hidden[3072]`, reads persistent per-layer Q4 down-proj
weight/scale rows, and writes exact `down_out[1024]` for the Layer 0 and true
Layer 1 packet instances. The QMAP final MLP
residual wrapper now also passes locally: it consumes descriptor-visible
`post_attn_hidden[1024]` and `down_out[1024]`, writes exact
`layer_out[1024]`, and catches descriptor/protocol error paths with no writes.
`47_export_chained_layer_qmap_artifacts.py` now regenerates chained per-layer
QMAP artifacts from the previous layer output. Layer 2 artifacts have been
generated from the Layer 1 `layer_out[1024]`; the Layer 2 attention score/value
and final MLP residual-add unit TBs pass locally, and the one-token scheduler
TB now recognizes Layer 2 packet/weight/cache windows. Its focused `+l2_only`
run passes with active layer 2, layer done mask `0x4`, and zero mismatches. A
`+true3_only` long-run entry is compiled for full Layer 0 -> Layer 1 -> Layer 2
regression when needed.
The first local Layer 0 body scheduler now chains the QMAP post-attention
residual/RMSNorm, MLP gate/up, MLP SiLU/multiply, MLP down, and final MLP
residual wrappers behind one memory request/write interface with exact
write-back through `layer_out[1024]`. The wider local Layer 0 scheduler now
also passes: it chains attention front-end, attention score/value, `o_proj`,
and that body scheduler behind one memory interface, proving exact write-back
from K/V cache and Q RoPE through final `layer_out[1024]`.
The first local Layer 0 compute scheduler now also passes: it runs the full
QKV projection packet first, patches the attention front-end Q/K/V inputs to
the actual QKV output buffers, then runs the wider Layer 0 scheduler behind
one memory interface with exact Q/K/V through final `layer_out[1024]`
write-back. The first local one-token/layer-loop boundary now also passes for
the implemented QMAP layer chain: it exposes layer index/count, hidden-buffer
bases, KV-cache base, token position, and per-layer QMAP packet base tables,
runs Layer 0 through that boundary, then runs a true Layer 0 -> Layer 1 chain
using Layer 0 write-back as Layer 1 input. It rejects missing-table or
out-of-range layer-loop requests before any memory traffic. The QKV
packet/export path has
now been parameterized for true Layer 1 data as well: Layer 1 Q/K/V Q4 vectors,
compact QMAP simulation hex, and a full `2048/1024/1024` AXI-backed QKV packet
at `0x4_1008_0000` pass local golden-word comparison. The remaining
multi-layer gap is narrower now: the Layer 1 attention front-end QMAP packet
also passes locally at `0x4_1502_0000`, with exact K/V cache writes and exact
Q RoPE write-back. The Layer 1 attention score/value QMAP packet also passes
locally at `0x4_1503_0000`, with exact K/V cache reads and exact
`attn_out[2048]` write-back. The Layer 1 `o_proj` QMAP packet now also passes
locally at `0x4_1504_0000`, with exact persistent weight/scale row reads and
exact `o_proj_out[1024]` write-back. The Layer 1 post-attention
residual/RMSNorm QMAP packet now also passes locally at `0x4_1505_0000`, with
exact `post_attention_hidden[1024]` and `post_norm[1024]` write-back. The
Layer 1 MLP gate/up QMAP packet now also passes locally at `0x4_1506_0000`,
with exact gate/up `[3072]` write-back. The Layer 1 MLP SiLU/multiply QMAP
packet now also passes locally at `0x4_1507_0000`, with exact
`mlp_hidden[3072]` write-back. The Layer 1 MLP down QMAP packet now also
passes locally at `0x4_1508_0000`, with exact persistent down-proj row reads
and exact `down_out[1024]` write-back. The Layer 1 final MLP residual QMAP
packet now also passes locally at `0x4_1509_0000`, with exact
`layer_out[1024]` write-back. The focused true two-layer scheduler run now
passes locally with exact write-back from Layer 0 QKV through Layer 1 final
`layer_out[1024]`, layer done mask `0x3`, and zero mismatches. The next real
gap is either running the compiled true three-layer long regression when useful
or composing the chained layer-loop output with the existing final-token tail
wrapper.

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
