# LLM_Accelerator

FPGA-based LLM accelerator project inspired by Hummingbird+.

Latest hardware PASS (2026-08-08): the complete Qwen3-0.6B tied-Q4 path passed
on the XCZU2EG board for the fixed two-position validation sequence. Both
positions completed all 28 layers with `done_mask=0x0fffffff`, `error_mask=0`,
and exact token/score results `28458/1227344433` then `64/1015661901`.

Current board-validated milestone (2026-08-08): the complete Qwen3-0.6B
`token -> tied-Q4 embedding -> 28 transformer layers -> final RMSNorm ->
151936-row tied LM head` path is now integrated into the Vivado block design.
The final Vivado export is under `Temp/final_vivado_export_20260802`. The routed
implementation uses `43,726 / 47,232` CLB LUTs (`92.58%`),
`72,621 / 94,464` registers (`76.88%`), `8,799 / 8,820` CLBs (`99.76%`),
`94.5 / 150` BRAM tiles, and `68 / 240` DSPs. WNS is `+0.208 ns`, WHS is
`+0.010 ns`, all `119,383` routable nets are fully routed, and there is no
routing/DRC/methodology Error or Critical Warning.

The complete 394,547,200-byte PL-DDR runtime is packed into 61 verified
segments. Its 281 QMAP packets, model-manifest provenance, non-overlap/aperture
rules, and 397 zero-initialized accelerator-writable regions are release-gated.
The latest short-path Vitis workspace `F:\vwi` builds the AXI-Lite control
smoke, the full28 persistent two-token model smoke, and the interactive
text/token application from the current XSA. The single
requested final full28 no-reset two-token XSim, including its independent
exact-write, producer-before-consumer, address, timing, and retained-KV audit,
passed in `Temp/final_qwen3_full28_20260802/20260802_162706`. Its exact outputs
are token/score `28458/1227344433` at position 0 and `64/1015661901` at
position 1.
The historical self-contained release
`Temp/boardready_qwen3_full28_20260801` had independently verified 92 files but
was no longer present at the bench. Its retained final manifest was repacked
without rerunning simulation into 61 verified PL-DDR segments. The AXI-Lite
control smoke, complete runtime load, all 281 QMAP-header checks, and the full28
persistent two-token model smoke then passed on physical hardware. See
[Source/BOARD_VALIDATION_20260808.md](Source/BOARD_VALIDATION_20260808.md) for
the artifact hashes, exact acceptance evidence, and remaining prompt-integration
boundary.

Latest PS/application milestone (2026-08-09): `qmap_prompt_demo` now implements
bounded arbitrary token-ID prefill, actual argmax feedback, EOS/IM_END/max/context
stops, and exact PS-native Qwen Unicode/ByteLevel-BPE tokenization plus raw-byte
detokenization. Its 79-case Hugging-Face differential suite and the host chain
`The future of FPGA is -> 785,3853,315,89462,374 -> 264,26291 ->
" a fascinating"` pass. The clean `a_qgen` ELF embeds the verified tokenizer
asset and is packaged in the verified non-release workbench
`F:\qot_boardtest_prompt_text_v9_20260809`. That format-v5 workbench contains
88 hash-audited files: the preserved 61-segment board runtime, `a_qgen`, the
PC Web Serial GUI, and a strict eight-transaction pyserial acceptance tool.
It also pins the trusted ELF, launcher, UI, and acceptance-tool provenance;
the tool's 21 host tests and packaged transcript replay pass. This new
interactive path has not yet been run on the board, so the latest hardware
claim remains the fixed 2026-08-08 two-position PASS.

The board-hosted Web path is now staged under
`FPGA_Project/software/qmap_web_demo/`: bounded fragmented HTTP/1.1, strict
prompt/token JSON, the exact tokenizer-backed generation job, and health,
submit, status, and raw-output routes pass host checks. Its production C
functions also pass a 1 KiB stack-frame gate, and a read-only XSA audit guards
both the future GEM3/MDIO/TTC0 configuration and the existing QMAP/PL-DDR
address contracts. The current XSA is expectedly rejected because networking
is still disabled. A guarded generator is ready to create a separate
`F:\vwn`/`p_net`/`a_net_echo` Vitis workspace only from a passing future XSA;
no lwIP or Ethernet board PASS is claimed yet.
Start with
[Source/CURRENT_STATE.md](Source/CURRENT_STATE.md) for the live handoff.

The long section below is retained as implementation history. It records how
the current full-depth datapath was assembled and validated; it is not the
current next-step list.

Detailed local RTL history: Layer 0 has passed through attention,
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
run passes with active layer 2, layer done mask `0x4`, and zero mismatches.
The AXI-Lite-controlled bounded true three-layer top-to-tail xsim now also
passes the current artifact contract: Layer 0 remains QKV-first while Layers 1
and 2 use the input-RMSNorm-enabled chain, then the final tail consumes Layer 2
output. The final-token tail path now also has a Layer2-chained artifact set:
final RMSNorm vectors are generated from the Layer 2
`layer_out[1024]`, full-vocabulary Q4 LM-head logits are regenerated from that
final_norm output, and the QMAP final-token tail wrapper passes Vivado xsim with
token `537`, exact score `850086863`, all `9496` tiles checked, and
`max_abs_logit_diff=0`.
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
`layer_out[1024]`, layer done mask `0x3`, and zero mismatches. The reusable
one-token top/control boundary now also passes a stronger multi-layer local
xsim run: `qmap_one_token_top.sv` starts Layer 1, then Layer 2, keeps the
final-tail packet descriptor stable, automatically selects the
scheduler-reported Layer 2 `layer_out[1024]` at `0x4_2509_2540` as the tail
hidden source, then runs final RMSNorm plus full-vocabulary LM-head through one
shared memory port. Vivado xsim `+l1_l2_top_tail_only +fastmem +progress`
reports scheduler done at cycle `13401169`, tail done at cycle `57501574`,
token `537`, score `850086863`, exact control pulses, exact aggregate counters,
and zero final output mismatches. The trace
`qmap_one_token_top_layer1_layer2_input_norm_to_final_tail_trace.csv` confirms
Layer 1 output write-back completes before Layer 2 reads it, Layer 2 output
write-back completes before final-tail hidden reads, final-norm output is read
only after write-back, and the output token/score write completes before top
`done`.
The same control boundary now also passes through the new MMIO-style register
contract: `qmap_one_token_control_regs.sv` plus the `+l1_l2_mmio_top_tail_only`
Vivado xsim scenario launches Layer 1 -> Layer 2 -> final-tail through
software-like register writes/table commits, reads back token `537`, score
`850086863`, layer mask `0x6`, exact top counters, and zero final write
mismatches.
A productized local wrapper, `qmap_one_token_mmio_top.sv`, now instantiates the
control register block and `qmap_one_token_top.sv` behind one register port and
one memory port. Its focused Icarus smoke test proves a register-launched
scheduler validation exit returns sticky status/layer-error/counter readback
with zero memory traffic. The first AXI4-Lite seam is also now local-RTL proven:
`axi4lite_to_mmio_regs.sv` serializes AXI4-Lite reads/writes into the same tiny
MMIO register port, and `qmap_one_token_axil_top.sv` wraps it around the one-token
MMIO top. The focused adapter test covers write/read ordering, stalls, and error
responses; the focused AXI-Lite top smoke test proves the same no-memory
scheduler validation exit through AXI-Lite writes/reads, with the tiny-MMIO
wrapper regression still passing. The AXI-Lite wrapper has now also passed the
positive Layer1 -> Layer2 -> final-tail memory-model xsim path through register
writes/reads: scheduler done cycle `13402176`, tail done cycle `57502581`, token
`537`, score `850086863`, layer mask `0x6`, top counters `178050/24947620`
reads and `12312/52227` writes, and zero final write mismatches. The same
AXI-Lite wrapper now also passes the bounded true three-layer top-to-tail xsim
using the current mixed artifact contract: Layer 0 QKV-first, Layers 1/2
input-RMSNorm-enabled, layer mask `0x7`, scheduler done cycle `20095284`, tail
done cycle `64195689`, token `537`, score `850086863`, top counters
`224314/27085750` reads and `18466/76803` writes, and zero final write
mismatches. The first PS-side runtime skeleton is now checked in under
`FPGA_Project/software/qmap_one_token_runtime/`: it mirrors the register offsets
and table ids, provides header-only configure/start/poll/result helpers, and
includes a Vitis no-memory validation smoke for the AXI-Lite seam before model
artifacts are loaded. A BD-facing shell, `qmap_one_token_axi_top.sv`, now wraps
that control seam with the existing lightweight AXI4 read/write masters so Vivado
can see one `S_AXI` register slave and one `M_AXI` PL-DDR master; its focused
Icarus smoke `tb_qmap_one_token_axi_top.sv` now proves the BD-facing shell can
run the same no-memory validation exit through `S_AXI` while issuing no `M_AXI`
read/write traffic. The integration runbook is
`FPGA_Project/Vivado_Project/ONE_TOKEN_AXI_TOP_BD_PLAN.md`, and the safe-by-default
Tcl scaffold `FPGA_Project/Vivado_Project/scripts/one_token_axi_top_bd_scaffold.tcl`
dry-runs under Vivado 2024.2 without modifying the current row1024 BD.
The per-layer input RMSNorm QMAP wrapper now
also passes locally: `qmap_input_rmsnorm_compute_path.sv` reads
descriptor-visible `hidden[1024]` and signed `I16_Q8_7` input-layernorm gamma
from a packet at `0x4_050A_0000`,
writes exact `input_norm[1024]`, covers request/write backpressure plus a bad
gamma dtype no-write error path, and passes the current signed-gamma Icarus
recheck with an independent CSV trace audit. That wrapper is now integrated as
an optional pre-stage in
`qmap_layer0_compute_scheduler.sv`: the focused `+qkv_precheck` Icarus run
patches QKV activation to the wrapper-written `input_norm[1024]`, regenerates
QKV golden words from that RTL RMSNorm output, and matches exact Q/K/V
write-back before stopping at the downstream frontend boundary. The same
pre-stage is now also plumbed through the reusable one-token scheduler and top
contracts: `tb_qmap_one_token_layer_scheduler.sv +input_norm_qkv_only
+fastmem`, including the `QMAP_ONE_TOKEN_TB_USE_TOP` build, writes exact
`input_norm[1024]`, writes exact Q/K/V `2048/1024/1024`, reports
`8234/561040` reads and `4097/5120` writes, and its top CSV trace shows all
four QKV activation reads from the RMSNorm output buffer occur after the
RMSNorm write-back burst completes. The QKV exporter now also emits external
activation descriptor bases for this path: Layer 0 reads `0x4_050A_2540`,
Layer 1 reads `0x4_150A_2540`, and Layer 2 reads `0x4_250A_2540`. The
one-token scheduler TB now also loads the generated Layer 1 and Layer 2 input
RMSNorm packets at `0x4_150A_0000` and `0x4_250A_0000`: focused
`+l1_input_norm_only` and `+l2_input_norm_only` Icarus prechecks write exact
`input_norm[1024]`, write exact Q/K/V `2048/1024/1024`, and stop at the
intentional frontend boundary with the same `8234/561040` read and
`4097/5120` write counters. The Layer 2 trace confirms the RMSNorm output
write completes before all four QKV activation reads. The promoted full-layer
focused paths now use that input-normalized activation contract too:
`+l1_only` and `+l2_only` enable Layer 1/2 input RMSNorm, run through the full
layer body, and pass exact write-back with cycle `6700589`,
`46278/2140354` reads, `6155/25600` writes, and layer done masks `0x2` and
`0x4`. The Layer 2 full trace also confirms no early reads from the RMSNorm or
Q/K/V producer buffers.

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
