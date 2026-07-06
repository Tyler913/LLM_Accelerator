# Current State

Last updated: 2026-07-06

This file is the concise working-state handoff. For durable project context,
read `Source/PROJECT_CONTEXT.md` first. For detailed address planning, read
`Source/FPGA_MEMORY_MAP.md`. For descriptor-based PL DDR4 tensor staging, read
`Source/QMAP_FORMAT.md`. For workflow rules, read `Source/AGENTS.md`.

## Current Goal

Build a first FPGA demo that can take a prompt and generate text with the local
`Qwen/Qwen3-0.6B-Base` model.

Current first-version target:

- Serial prefill and single-token cached decode
- Greedy argmax text continuation
- PS-side tokenizer, detokenizer, artifact loading, PL control, status polling,
  and debug output
- PL-side `next_token = run_one_token(input_token, position)` model path
- PL owns model-side prefill and decode math
- Initial context target: 128 or 256 tokens
- Required PL DDR4 weight path: custom Q4 weight-only format
- PL compute blocks use hand-written Verilog/SystemVerilog by default

## Software Baseline

The FP32 Python reference is complete for the first-version target. It covers
serial prefill/decode, full 28-layer cached single-token decode, and focused
module checks for the FPGA bring-up path.

Key locations:

- Full-model references: `Qwen3-0.6B-Base/pc_testing/`
- Module references: `Qwen3-0.6B-Base/python_each_module/`
- Module validation index:
  `Qwen3-0.6B-Base/python_each_module/README.md`
- FP32 test vectors:
  `artifacts/test_vectors/qwen3_0p6b_fp32_v0/` (ignored by Git)
- Q4 bring-up vectors:
  `artifacts/test_vectors/qwen3_0p6b_q4_v0/` (ignored by Git)

Useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/14_verify_q4_gemv_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/16_profile_rmsnorm_ranges.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/19_export_qmap_dot64_image.py --c-header FPGA_Project/software/qmap_load_smoke/qmap_dot64_image.h
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/20_export_qmap_row1024_image.py --c-header FPGA_Project/software/qmap_row1024_pl_compute_smoke/qmap_row1024_image.h
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/21_export_qmap_qkv_projection_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/26_export_o_proj_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/27_export_post_attention_residual_norm_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/28_export_mlp_gate_up_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/29_export_mlp_silu_mul_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/30_export_mlp_down_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/31_export_mlp_residual_add_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/32_export_final_rmsnorm_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/33_export_lm_head_argmax_vectors.py --scan-rows 1024 --tile-rows 16 --chunk-rows 2048
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/34_export_qmap_lm_head_argmax_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/35_export_lm_head_full_vocab_vectors.py --chunk-rows 1024
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/34_export_qmap_lm_head_argmax_image.py --lm-prefix lm_head_argmax_full_vocab_real --output artifacts/test_vectors/qwen3_0p6b_qmap_v1/lm_head_argmax_full_vocab_runtime.qmap.bin --manifest artifacts/test_vectors/qwen3_0p6b_qmap_v1/lm_head_argmax_full_vocab_runtime_manifest.json --sim-hex FPGA_Project/sim/vectors/qmap_lm_head_argmax_full_vocab_image_words32.hex --expected-hex FPGA_Project/sim/vectors/qmap_lm_head_argmax_full_vocab_expected_words32.hex
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/36_export_qmap_final_token_tail_image.py --lm-prefix lm_head_argmax_stage_real --output artifacts/test_vectors/qwen3_0p6b_qmap_v1/final_token_tail_compact_runtime.qmap.bin --manifest artifacts/test_vectors/qwen3_0p6b_qmap_v1/final_token_tail_compact_runtime_manifest.json --sim-hex FPGA_Project/sim/vectors/qmap_final_token_tail_compact_image_words32.hex --expected-hex FPGA_Project/sim/vectors/qmap_final_token_tail_compact_expected_words32.hex
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/36_export_qmap_final_token_tail_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/37_export_qmap_attention_frontend_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/38_export_qmap_attention_score_value_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/39_export_qmap_o_proj_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/40_export_qmap_post_attention_residual_norm_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/41_export_qmap_mlp_gate_up_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/42_export_qmap_mlp_silu_mul_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/43_export_qmap_mlp_down_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/44_export_qmap_mlp_residual_add_image.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/47_export_chained_layer_qmap_artifacts.py --layer-id 2
```

Stable reference prompt:

```text
The future of FPGA is
```

Known greedy checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## RTL Status

The current hand-written RTL bring-up stack includes:

- Q4 GEMV:
  - `q4_dot_product_64.sv`
  - `q4_gemv_row_1024.sv`
  - `q4_gemv_tile_1024.sv`
  - `q4_gemv_projection_1024.sv`
- RMSNorm:
  - `rmsnorm_sum_squares_1024.sv`
  - `fixed_sqrt_u64.sv`
  - `fixed_udiv.sv`
  - `rmsnorm_apply_1024.sv`
  - `rmsnorm_1024.sv`
  - `residual_add_1024.sv`
  - `post_attention_residual_norm_stage.sv`
  - `final_rmsnorm_stage.sv`
- RoPE:
  - `rope_qk_layer_128.sv`
- Attention front-end integration:
  - `qk_norm_128.sv`
  - `qk_norm_rope_stage_128.sv`
  - `attention_score_stage.sv`
  - `attention_softmax_value_stage.sv`
- Attention output projection:
  - `o_proj_stage.sv`
- MLP:
  - `mlp_gate_up_proj_stage.sv`
  - `mlp_silu_mul_stage.sv`
  - `mlp_down_proj_stage.sv`
  - `mlp_residual_add_stage.sv`
- Final output:
  - `lm_head_argmax_stage.sv`
  - `lm_head_tile_mem_reader.sv`
  - `lm_head_argmax_mem_stage.sv`
  - `lm_head_argmax_tile_scheduler.sv`
- KV cache append:
  - `kv_cache_addr_gen.sv`
  - `kv_cache_append.sv`
  - `qk_norm_rope_kv_cache_stage.sv`
- QMAP/DDR data movement bring-up:
  - `qmap_defs.svh`
  - `qmap_header_reader.sv`
  - `qmap_descriptor_reader.sv`
  - `qmap_dot64_reader.sv`
  - `qmap_dot64_payload_fetcher.sv`
  - `qmap_dot64_compute_path.sv`
  - `qmap_row1024_payload_fetcher.sv`
  - `qmap_row1024_compute_path.sv`
  - `axi4_read_master.sv`
  - `axi4_write_master.sv`
  - `qmap_qkv_projection_compute_path.sv`
  - `qmap_qkv_projection_axi_smoke_top.sv`
  - `qmap_attention_frontend_compute_path.sv`
  - `qmap_attention_score_value_compute_path.sv`
  - `qmap_o_proj_compute_path.sv`
  - `qmap_post_attention_residual_norm_compute_path.sv`
  - `qmap_mlp_gate_up_compute_path.sv`
  - `qmap_mlp_silu_mul_compute_path.sv`
  - `qmap_mlp_down_compute_path.sv`
  - `qmap_mlp_residual_add_compute_path.sv`
  - `qmap_layer0_body_scheduler.sv`
  - `qmap_layer0_full_scheduler.sv`
  - `qmap_layer0_compute_scheduler.sv`
  - `qmap_one_token_layer_scheduler.sv`
  - `qmap_lm_head_argmax_compute_path.sv`
  - `qmap_final_token_tail_compute_path.sv`
  - `qmap_dot64_axi_smoke_top.sv`
  - `qmap_dot64_axi_smoke_bd.v`
  - `qmap_row1024_axi_smoke_top.sv`
  - `qmap_row1024_axi_smoke_bd.v`

Current fixed-point direction:

- Large weights: project custom Q4 weight-only format from
  `Source/Q4_FORMAT.md`
- Q4 scales: current bring-up uses unsigned 16-bit `Q2.14`
- RMSNorm/residual input: signed 24-bit `Q14.10`
- RMSNorm output and Q/K/RoPE path: signed 24-bit `Q12.12`
- Layer 0 input RMSNorm gamma uses unsigned 16-bit `UQ8.8` in the current
  real-vector RTL check.
- Layer 0 post-attention RMSNorm gamma uses signed 16-bit `Q8.7`; the actual
  `post_attention_layernorm.weight` includes negative values. The final norm
  also contains some negative gamma values, so unsigned gamma is not a safe
  full-model default.
- Q/K head RMSNorm gamma: signed 16-bit `Q8.7` in the current Layer 0
  q/k norm + RoPE RTL stage. This was corrected after confirming Layer 0
  `q_norm.weight` contains negative values.
- KV cache first RTL storage plan: signed 24-bit `Q12.12` padded to 32-bit
  DDR words

Latest local RTL/software validation state:

- Existing Icarus simulation set under `FPGA_Project/sim/` passed in the latest
  full local recheck.
- Q4 GEMV real-vector RTL checks pass for Layer 0 q_proj rows 0 through 15.
- RMSNorm real-vector RTL check passes for the Layer 0 input_layernorm
  last-token vector.
- RoPE real-vector RTL check passes for the Layer 0 last-token Q/K vector.
- `kv_cache_addr_gen.sv` passes context-256 and context-512 address smoke
  vectors.
- Python FP32 module vectors and Q4 GEMV vectors pass their current verifiers.
- `19_export_qmap_dot64_image.py` exports the first QMAP v1 PL DDR4 staging
  image from `q_proj_row0_group0_dot64.npz` and self-checks the header,
  descriptor table, and payload bytes.
- `qmap_load_smoke_app` passed on hardware. It writes the embedded 1536-byte
  QMAP dot64 image to PL DDR4 at `0x4_1B10_0000`, reads it back exactly, and
  checks the QMAP header, four descriptors, and payload spot values.
- `qmap_dot64_reader.sv` passes RTL simulation against the generated
  `qmap_dot64_image_words32.hex` memory image. It issues the project-local
  memory request/response reads, decodes the QMAP header, and captures the four
  dot64 descriptor slots for activation, packed Q4 weight, Q2.14 scale, and
  expected/debug result tensors.
- `qmap_dot64_payload_fetcher.sv` passes RTL simulation after
  `qmap_dot64_reader.sv`. It uses the descriptor base/nbytes fields to fetch
  the activation vector, packed Q4 weight group, Q2.14 scale, and expected
  partial/scaled debug values from the same generated QMAP memory image.
- `qmap_dot64_compute_path.sv` passes the first QMAP-backed dot64 compute
  wrapper simulation. This synthesizable wrapper owns the sequence:
  `qmap_dot64_reader.sv`, then `qmap_dot64_payload_fetcher.sv`, then
  `q4_dot_product_64.sv`. It compares the computed partial sum and scaled Q26
  result against the QMAP expected/debug payload. The matched values are
  `partial_sum=24751` and `scaled_sum_q26=3019622`.
- `axi4_read_master.sv` passes a focused RTL smoke simulation for one aligned
  4-beat AXI4 read burst. This is the first read-only adapter from the
  project-local memory request/response interface toward a future Vivado AXI
  master integration.
- `tb_qmap_dot64_compute_path_axi4.sv` passes the first AXI-backed QMAP dot64
  compute simulation. The path is `qmap_dot64_compute_path.sv ->
  axi4_read_master.sv -> AXI read memory model`, with the memory model loaded
  from `qmap_dot64_image_words32.hex`. The simulation observed 9 AXI read
  bursts and matched `partial_sum=24751`, `scaled_sum_q26=3019622`.
- `qmap_dot64_axi_smoke_top.sv` is the Vivado-facing top for the first PL-side
  QMAP dot64 hardware smoke path. It wraps `qmap_dot64_compute_path.sv` and
  `axi4_read_master.sv`, exposes a conventional AXI master port for Block
  Design integration, and provides sticky PS-visible status/result outputs.
  Its focused simulation passes with status `0xA`, 9 AXI read bursts,
  `partial_sum_low32=24751`, and `scaled_sum_q26_low32=3019622`.
- `qmap_dot64_axi_smoke_bd.v` is the Block Design friendly wrapper for that
  smoke path. It has fixed-width Verilog ports and no parameters/includes at
  the BD-facing boundary, then instantiates `qmap_dot64_axi_smoke_top.sv`
  internally. Its AXI/clock interface metadata currently uses the existing
  Vivado PS PL clock frequency, `96,968,727 Hz`, to match
  `zynq_ultra_ps_e_0/pl_clk0` and `axi_smc/S01_AXI`.
- `20_export_qmap_row1024_image.py` exports the second QMAP v1 image,
  `q_proj` Layer 0 row 0 row1024, at `0x4_1B20_0000`. The image is 4096 bytes,
  stores activation `[1024]`, packed Q4 weight `[1,1024]`, scale `[1,16]`,
  and expected row sum debug payload. The expected result is
  `row_sum_q26_int64=-3482169`.
- `qmap_row1024_compute_path.sv` passes RTL simulation against
  `qmap_row1024_image_words32.hex`. It reads QMAP descriptors, fetches the
  row payloads in 4-group batches, runs `q4_gemv_row_1024.sv`, and matches the
  expected row sum.
- `qmap_row1024_payload_fetcher.sv`, `qmap_row1024_compute_path.sv`,
  `qmap_row1024_axi_smoke_top.sv`, and `qmap_row1024_axi_smoke_bd.v` now form
  the Vivado-facing row1024 smoke path. The AXI simulation passes with 18 read
  bursts, status `0xA`, and row result `-3482169`.
- The row1024 PL master smoke path has passed on hardware. PS writes the
  4096-byte QMAP image to `0x4_1B20_0000`, readback and header checks pass,
  PL reports status `0xA`, and both result GPIO channels return
  `0xFFCA_DDC7` / `-3482169`.
- `21_export_qmap_qkv_projection_image.py` exports QKV projection work-packet
  images with 12 active descriptors in a 32-slot QMAP table. The default Layer
  0 full packet covers `q_proj[2048]`, `k_proj[1024]`, and `v_proj[1024]` at
  `0x4_0008_0000`; Python recomputes Q/K/V Q26 from the packed Q4
  weights/scales and matches the Q4 artifact with zero recompute diff before
  converting outputs to I32_Q12.12 words. The exporter now also accepts
  `--layer-id` so metadata and descriptor `aux` fields identify the target
  layer instead of being hard-coded to Layer 0.
- `45_export_layer_qkv_q4_vectors.py` exports true per-layer Q/K/V Q4 vectors
  from full-prompt model hooks. The current Layer 1 last-token vector is
  `artifacts/test_vectors/qwen3_0p6b_q4_v0/qkv_layer1_last_token_q4.npz`;
  Q4-vs-FP32 max/mean abs errors are Q `0.07864405` / `0.01045488`, K
  `0.05217079` / `0.00988104`, and V `0.03997995` / `0.00910271`.
- The compact QKV simulation packet uses the same descriptor contract with
  `q_rows=4`, `k_rows=2`, and `v_rows=2` so Icarus can complete quickly while
  still exercising descriptor-driven Q/K/V reads and output-buffer writes.
- `qmap_qkv_projection_compute_path.sv` passes RTL simulation against
  `qmap_qkv_projection_image_words32.hex` and
  `qmap_qkv_projection_expected_words32.hex`. It reads the QMAP descriptors,
  loads the I32_Q12.12 activation vector, loops through Q/K/V rows using
  `q4_gemv_row_1024.sv`, converts Q26 sums to I32_Q12.12, writes Q/K/V output
  buffers, and matches all Python expected output words. The compact run wrote
  8 rows through the local write interface with no error.
- `axi4_write_master.sv` passes a focused RTL smoke simulation for one aligned
  4-beat AXI4 write burst and B-channel response. This is the first write-side
  adapter from the project-local write stream toward Vivado AXI integration.
- `qmap_qkv_projection_axi_smoke_top.sv` now wraps the QKV projection path with
  `axi4_read_master.sv` and `axi4_write_master.sv`, and takes a runtime
  `i_qmap_base_addr` so the same RTL wrapper can consume non-Layer0 packets.
  Its parameterized Icarus integration testbench passes compact, medium,
  larger, and full Layer 0 QKV packets. It also passes true Layer 1 QKV:
  compact `q_rows=4`, `k_rows=2`, `v_rows=2` and full `2048/1024/1024` packets
  at `0x4_1008_0000`. The full Layer 1 run completes `8209` AXI read bursts
  and `4096` AXI write bursts, reports status `0xA`, and matches every Python
  expected I32_Q12.12 output word.
- `qk_norm_128.sv` and `qk_norm_rope_stage_128.sv` add the first real
  downstream attention-front-end stage after QKV projection. The exporter
  `22_export_qk_norm_rope_fixed_vectors.py` recomputes Layer 0 Q/K projection
  words from the same Q4/QMAP integer contract, applies signed `Q8.7`
  q_norm/k_norm gamma, and emits fixed golden vectors for q_norm/k_norm and
  RoPE.
- `tb_qk_norm_rope_stage_128.sv` passes in Icarus. It runs all 24 heads
  (`16` Q heads and `8` K heads), completes in `10612` cycles, reports
  `norm_heads_done=24`, and matches Q norm, K norm, Q RoPE, and K RoPE output
  buffers with `max_abs_diff=0` and no saturation. The existing
  `tb_rmsnorm_1024_real.sv` and `tb_rope_qk_layer_128.sv` regression
  simulations still pass after adding signed-gamma support to the RMSNorm core.
- `23_export_kv_cache_append_vectors.py` exports the first fixed K/V cache
  append vector from the same Layer 0 Q4/QMAP integer contract. It stores
  RoPE-transformed K and raw reshaped V as signed 24-bit `Q12.12` values padded
  to 32-bit words and emits exact expected cache write addresses/data.
- `kv_cache_append.sv` passes in Icarus against the generated vector. The
  standalone test writes `2048` words, checks exact K then V address/data order,
  survives deterministic backpressure, and audits that write-stream signals stay
  stable while stalled.
- `qk_norm_rope_kv_cache_stage.sv` now combines q/k norm, RoPE, and K/V cache
  append for the Layer 0 last-token path. Its combined testbench matches Q RoPE
  and K RoPE with `max_abs_diff=0`, accepts `2048` cache writes, observes the
  first valid cache write after the norm/RoPE phase, and passes the same
  stall-stability trace audit.
- `24_export_attention_score_vectors.py` exports the first fixed attention
  score vector. It uses the existing Q4 weights/scales, quantizes prompt
  `input_norm` values into the same Q12.12 activation contract, builds a
  5-position K cache after k_norm/RoPE, and emits current-token Q plus exact
  raw and scaled score golden words.
- `attention_score_stage.sv` passes in Icarus against that vector. It requests
  K-cache elements through a ready/valid read interface, applies the Qwen3 GQA
  mapping from 16 Q heads to 8 KV heads, emits raw Q24.24 dot-product sums and
  Q0.31-scaled scores, and matches all `80` score outputs exactly. The
  testbench accepts `10240` K requests/responses, injects request backpressure
  plus variable response latency, stalls the score output stream, and its CSV
  trace audit confirms request order, response pairing, score order, and
  stall-stable signals.
- `25_export_attention_softmax_value_vectors.py` exports the first fixed
  softmax/value vector. It uses the same 5-position prompt cache, converts
  scaled Q24.24 scores into UQ0.16 probabilities through a 257-entry UQ0.20
  exp LUT with 1/16 score steps, and emits signed Q12.12 `attn_out[16,128]`
  golden words.
- `attention_softmax_value_stage.sv` passes in Icarus against that vector. It
  accepts the `80` score stream, computes per-head probabilities, reads V cache
  through a ready/valid request/response interface, and matches all `2048`
  Q12.12 attention-output words exactly. The testbench injects score input
  gaps, `10240` V requests/responses with backpressure and variable response
  latency, output-stream stalls, and a CSV trace audit confirms score order,
  V response pairing, probability values, output order, and stall-stable
  signals.
- `26_export_o_proj_vectors.py` exports attention output projection vectors.
  The default Layer 0 vector reuses the fixed softmax/value `attn_out[2048]`
  vector, quantizes `attn0.o_proj.weight[1024,2048]` into the same signed Q4
  plus unsigned Q2.14 scale contract, and emits both unpacked debug hex and
  32-bit packed simulation hex for the large weight/scale tensors. The exporter
  now also accepts `--layer-id`, `--input`, and `--prefix`; the current Layer 1
  vector is `layer1_o_proj_stage_real`, with fixed `o_proj` versus FP32
  max/mean abs diff `0.081983894` / `0.012546605`.
- `o_proj_stage.sv` passes in Icarus against that vector. It reuses the
  parameterized Q4 GEMV projection controller with `INPUT_SIZE=2048`, converts
  Q26 row sums to signed Q12.12 outputs, and matches all `1024` Python golden
  words exactly under output-stream backpressure. The full run accepted 1024
  outputs, observed 618 output stall cycles, completed 35073 compute cycles,
  and reported `max_abs_output_diff=0`.
- `27_export_post_attention_residual_norm_vectors.py` exports the first fixed
  post-attention residual/RMSNorm vector. It consumes the locally generated
  fixed `o_proj_stage_real_expected_q12_12.hex`, quantizes the Layer 0 residual
  input as signed Q14.10, uses signed Q8.7
  `post_attention_layernorm.weight`, and emits exact residual and post-norm
  golden vectors plus RMSNorm debug scalars.
- `residual_add_1024.sv` and `post_attention_residual_norm_stage.sv` pass in
  Icarus against that vector. The combined test matches all `1024`
  post-attention residual Q14.10 words and all `1024` post-norm Q12.12 words
  exactly, with `sum_squares=33222329`, `mean_square=32443`, `rms_q10=180`,
  `inv_rms=372827`, no saturation, and `max_abs_diff=0` for both output
  vectors. The CSV trace audit observed one busy-period spurious start pulse,
  one done cycle, zero error cycles, and the expected
  residual-then-RMSNorm state sequence.
- `28_export_mlp_gate_up_vectors.py` exports the first fixed Layer 0 MLP
  gate/up projection vector. It consumes the locally generated
  `post_attention_residual_norm_stage_real_expected_norm.hex` post-norm vector,
  quantizes `mlp.gate_proj.weight[3072,1024]` and
  `mlp.up_proj.weight[3072,1024]` with the project signed Q4 plus unsigned
  Q2.14 scale contract, and emits Q26 plus Q12.12 golden outputs.
- `mlp_gate_up_proj_stage.sv` passes in Icarus against that vector. It runs
  gate and up projections from the same `post_norm[1024]` activation, reuses
  the Q4 GEMV projection controller, converts Q26 row sums to signed Q12.12,
  and streams exact gate/up output pairs. The full run accepted all `3072`
  output pairs under deterministic output backpressure, observed `1394`
  output stall cycles, completed `52993` compute cycles, matched every Python
  golden gate/up word with `max_abs_diff=0`, covered one busy-period spurious
  start pulse, and passed the CSV trace audit.
- `29_export_mlp_silu_mul_vectors.py` exports fixed MLP SiLU/multiply vectors.
  It now accepts `--layer-id`, `--gate-up-prefix`, and `--prefix`, consumes the
  matching locally generated gate/up Q12.12 outputs, builds a UQ0.16 sigmoid LUT
  over `[-8, 8]` with `1/64` spacing, and emits LUT index, sigmoid,
  `silu(gate)`, and `mlp_hidden[3072]` golden words. The original Layer 0 vector
  has no saturation, ranges from `-3969` to `10573` in Q12.12, and is within
  about `0.00271` max absolute error versus ideal fixed-point SiLU. The current
  true Layer 1 vector is `layer1_mlp_silu_mul_stage_real`; it consumes
  `layer1_mlp_gate_up_proj_stage_real`, has no saturation, ranges from `-5564`
  to `3118`, and its ideal fixed-point max/mean abs errors are `0.0059265907` /
  `0.00024485353`.
- `mlp_silu_mul_stage.sv` passes in Icarus against that vector. It streams
  gate/up pairs through a ready/valid input, checks a flattened sigmoid LUT,
  computes `silu(gate) * up`, and emits exact Q12.12 `mlp_hidden` words. The
  full run accepted all `3072` inputs and `3072` outputs, injected `1031`
  input-gap cycles, `747` input-stall cycles, and `961` output-stall cycles,
  matched every Python golden word with `max_abs_hidden_diff=0`, covered one
  busy-period spurious start pulse, and passed the CSV trace audit with one
  done cycle, zero error cycles, zero saturation cycles, and monotonic
  input/output index order.
- `30_export_mlp_down_vectors.py` exports fixed MLP down projection vectors. It
  now accepts `--layer-id`, `--hidden-prefix`, and `--prefix`, consumes the
  selected layer's locally generated `mlp_hidden[3072]`, quantizes
  `mlp.down_proj.weight[1024,3072]` with the project signed Q4 plus unsigned
  Q2.14 scale contract, and emits Q26 plus Q12.12 golden outputs. The original
  Layer 0 vector has no output saturation; Q12.12 `down_out[1024]` ranges from
  `-2351` to `7843`, and fixed down projection versus the HF path has max error
  about `0.16191256` and mean error about `0.022316866`. The current true
  Layer 1 vector is `layer1_mlp_down_proj_stage_real`; it consumes
  `layer1_mlp_silu_mul_stage_real`, has no output saturation, ranges from
  `-2550` to `15159`, and its fixed down projection versus the HF path has
  max/mean abs diff `0.18561149` / `0.029139201`.
- `mlp_down_proj_stage.sv` passes in Icarus against that vector. It reuses the
  parameterized Q4 GEMV projection controller with `INPUT_SIZE=3072`, converts
  Q26 row sums to signed Q12.12, and streams exact `down_out[1024]` words.
  The full run accepted all `1024` outputs, observed `497` output-stall cycles,
  completed `52481` compute cycles, matched every Python golden word with
  `max_abs_output_diff=0`, covered one busy-period spurious start pulse, and
  passed the CSV trace audit with one done cycle, zero error cycles, zero
  saturation cycles, sequential output rows, correct final `out_last`, and all
  64 output tiles covered.
- `31_export_mlp_residual_add_vectors.py` exports the final fixed Layer 0 MLP
  residual vector. It consumes the locally generated
  `post_attn_hidden[1024]` Q14.10 and `down_out[1024]` Q12.12 vectors, emits
  exact Q14.10 `layer_out[1024]` golden words, and records no residual
  saturation. The generated `layer_out` ranges from `-1726` to `4593` in
  Q14.10; fixed Layer 0 output versus the HF path has max error about
  `0.15058082` and mean error about `0.024729879`.
- `mlp_residual_add_stage.sv` passes in Icarus against that vector. It wraps
  the existing sequential `residual_add_1024.sv` block for
  `post_attn_hidden[1024] + down_out[1024] -> layer_out[1024]`, matches all
  `1024` Q14.10 output words exactly on two consecutive runs, reports
  `max_abs_diff=0`, covers one busy-period spurious start pulse, observes two
  one-cycle done pulses with no adjacent done cycles, and passes an independent
  CSV trace audit with zero error and zero saturation cycles.
- `32_export_final_rmsnorm_vectors.py` exports the first fixed full-model
  current-token final RMSNorm vector. It captures `model.norm` input/output
  after the full 28-layer Python reference, quantizes final hidden as signed
  Q14.10 and final RMSNorm gamma as signed Q8.7, and emits exact Q12.12 golden
  output plus debug scalars. The vector has no input, gamma, or output
  saturation; final hidden max_abs is `203263` in Q14.10, final gamma includes
  `3` negative fixed-point entries, and fixed final RMSNorm versus HF has max
  error about `0.019233763` and mean error about `0.00043280968`.
- `final_rmsnorm_stage.sv` passes in Icarus against that vector. It wraps
  `rmsnorm_1024.sv` with signed gamma enabled, matches all `1024` Q12.12
  output words exactly on two consecutive runs, and matches
  `sum_squares=271992247168`, `mean_square=265617428`, `rms_q10=16297`, and
  `inv_rms=4117`. The independent CSV trace audit observed two done pulses,
  no adjacent done cycles, one busy-period spurious start pulse, full
  sum/sqrt/div/apply sub-state coverage, zero error cycles, and zero
  saturation cycles.
- `33_export_lm_head_argmax_vectors.py` exports the first fixed Q4 LM-head
  scan/argmax vector from the passing full-model final RMSNorm output. It
  quantizes the tied `lm_head.weight` path as signed Q4 plus unsigned Q2.14
  scales, proves the full Q4 argmax in Python is token `264` (`' a'`) with
  score `1365150750`, matching the HF/FP32 argmax, and emits a tiled
  1024-row scan window `[0,1024)` that contains that winner.
- `lm_head_argmax_stage.sv` passes in Icarus against that vector. It requests
  16-row Q4 weight/scale tiles through a ready/valid tile interface, reuses
  `q4_gemv_tile_1024.sv`, checks all `2048` logits across two full runs
  exactly, reports `max_abs_logit_diff=0`, returns best token `264` and score
  `1365150750`, covers one busy-period spurious start pulse, and passes an
  independent CSV trace audit with `128` tile requests, `128` tile responses,
  `128` tile updates, no error cycles, no adjacent done cycles, and exact
  `0..63` tile order on both runs. This also fixed a shared
  `q4_gemv_tile_1024.sv` output-hold bug by registering tile outputs when all
  row lanes complete; the older tile and projection smoke tests pass after
  updating their timeout budgets to the current row-kernel latency.
- `lm_head_tile_mem_reader.sv` and `lm_head_argmax_mem_stage.sv` turn that
  LM-head tile interface into a project-local memory-facing path for the same
  1024-row scan window. The reader fetches each 16-row tile as eight 1024-byte
  packed-Q4 weight bursts plus one 512-byte scale burst from exported
  LM-head base addresses, then feeds `lm_head_argmax_stage.sv`.
  `tb_lm_head_argmax_mem_stage.sv` passes in Icarus across two full runs:
  best token `264`, score `1365150750`, `2048` checked logits,
  `max_abs_logit_diff=0`, `1152` memory read requests, `278528` response
  words, `1152` response-last handshakes, no error cycles, no adjacent done
  cycles, full core/reader state coverage, and one busy-period spurious start
  pulse.
- `lm_head_argmax_tile_scheduler.sv` reuses `lm_head_argmax_mem_stage.sv` as a
  one-tile child engine, keeps the global best token/score across runtime tile
  windows, and defaults to `MAX_TILES=9496` for the full `151936`-row tied
  embedding/LM-head vocabulary. `tb_lm_head_argmax_tile_scheduler.sv` passes in
  Icarus across two runtime scan counts: `64` tiles for the existing 1024-row
  window and `23` tiles for a 368-row prefix. It reports final token `264`,
  score `1365150750`, `783` memory read requests, `189312` response words,
  `696` weight requests, `87` scale requests, `1392` checked logits,
  `max_abs_logit_diff=0`, no error cycles, no adjacent done cycles, full
  scheduler/core/reader state coverage, and tile coverage
  `max_tile_run1=63` / `max_tile_run2=22`. This proves the runtime scheduler
  control path locally.
- `34_export_qmap_lm_head_argmax_image.py` exports a QMAP LM-head runtime packet
  at `0x4_0500_0000`. The packet has six active descriptors for metadata,
  final RMSNorm activation, persistent LM-head Q4 weights, persistent Q2.14
  scales, output token/score, and expected/debug token/score. The weight and
  scale descriptors expose the scan range through `aux2=scan_base` and
  `aux3=tile_count`. The exporter now accepts `--lm-prefix`, so the same QMAP
  descriptor contract can wrap both compact and full-vocabulary LM-head vector
  sets.
- `35_export_lm_head_full_vocab_vectors.py` exports the full-vocabulary
  LM-head Q4 simulation memory without reformatting the RTL contract:
  activation `[1024]`, persistent weight words for all `151936` rows,
  persistent scale words, `151936` Q26 golden logits, and expected token/score.
  Python proves the full Q4 argmax is token `264` (`' a'`) with score
  `1365150750`, matching the HF/FP32 argmax.
- `qmap_lm_head_argmax_compute_path.sv` passes in Icarus against that packet.
  It reads the QMAP header/descriptors, validates the LM-head descriptor
  contract, reads `final_norm[1024]`, runs `lm_head_argmax_tile_scheduler.sv`,
  and writes `{token, score_low32, score_high32}` to the output descriptor.
  `tb_qmap_lm_head_argmax_compute_path.sv` covers two successful descriptor
  scan counts (`64` tiles and `23` tiles) plus an invalid `tile_count=0`
  descriptor. The passing run reports token `264`, score `1365150750`,
  `812` read requests, `191984` response words, `696` weight requests,
  `87` scale requests, `2` output write bursts, `6` output write words,
  `1392` checked logits, `max_abs_logit_diff=0`, no error rows in the two
  successful runs, no adjacent done cycles, and full wrapper/scheduler/core/
  reader state coverage. The invalid descriptor run raises error, writes no
  output, and clears the externally visible best result.
- The same QMAP LM-head wrapper passes the full-vocabulary run in Vivado xsim
  with `QMAP_LM_HEAD_TB_FULL_VOCAB`: `9496` tiles, `151936` checked logits,
  `85475` read requests, `20664528` response words, `75968` weight requests,
  `9496` scale requests, one output write burst, token `264`, score
  `1365150750`, `max_abs_logit_diff=0`, one done pulse, and no error.
- `36_export_qmap_final_token_tail_image.py` exports the first QMAP
  final-token tail runtime packet at `0x4_0501_0000`. It uses eight
  descriptors for metadata, final-norm scratch/LM-head activation,
  persistent LM-head Q4 weight and Q2.14 scale references, output token/score,
  expected/debug token/score, final hidden input, and signed final RMSNorm
  gamma. This packet keeps the already-proven LM-head descriptor contract and
  adds the preceding final RMSNorm write-back.
- `qmap_final_token_tail_compute_path.sv` is the first memory-mapped
  one-token wrapper boundary. It reads `final_hidden[1024]` and
  `final_norm.gamma[1024]` from QMAP descriptors, runs
  `final_rmsnorm_stage.sv`, writes `final_norm[1024]` back to the descriptor
  output buffer, then invokes `qmap_lm_head_argmax_compute_path.sv` and writes
  the final token/score descriptor. The compact Icarus run covers a 64-tile
  LM-head scan plus an invalid gamma descriptor path: token `264`, score
  `1365150750`, `1024` exact final-norm write words, `1024` checked logits,
  `max_abs_logit_diff=0`, no norm mismatches, stale best-result clearing on
  invalid descriptor, and no adjacent done pulses. The full-vocabulary Vivado
  xsim run covers `9496` tiles / `151936` logits with the same token and
  score, `75968` weight requests, `9496` scale requests, `1024` final-norm
  write words, `3` token/score output words, `max_abs_logit_diff=0`, one done
  pulse, and zero error/saturation rows in the trace audit.
- `37_export_qmap_attention_frontend_image.py` exports the first QMAP
  attention front-end runtime packet at `0x4_0502_0000`. It uses ten active
  descriptors in a 16-slot table for metadata, Q/K/V projection output buffers,
  signed q/k gamma, RoPE cos/sin tables, the KV-cache descriptor, and Q RoPE
  output scratch. It now also accepts dynamic q/k+RoPE vector prefixes,
  KV-cache vector prefixes, and a runtime `--qmap-base`, so the same packet
  contract can target non-Layer0 data.
- `qmap_attention_frontend_compute_path.sv` is the first memory-mapped
  per-layer body wrapper after QKV projection write-back. It reads Q/K/V flat
  buffers, q/k gamma, and RoPE cos/sin from QMAP descriptors, runs
  `qk_norm_rope_kv_cache_stage.sv`, writes exact K/V cache entries, then writes
  `q_rope[2048]` to the descriptor output buffer. The Icarus testbench matches
  `2048` K/V cache writes and `2048` Q RoPE write words exactly, accepts
  `31` read requests / `4944` read-response words and `2049` write requests /
  `4096` write-data words, injects read/write request and data backpressure,
  covers a busy-period spurious start pulse, and covers an invalid RoPE-cos
  dtype descriptor that raises error with no writes. The trace audit covers all
  wrapper states and confirms cache writes finish before Q RoPE write-back.
- The q/k norm+RoPE and KV-cache vector exporters are now parameterized for
  true Layer 1 inputs. The current Layer 1 attention-front-end packet uses
  Layer 1 QKV projection output vectors at packet base `0x4_1502_0000`,
  position `4`, and KV-cache base `0x4_1410_0000`. Its Icarus run matches
  `2048` K/V cache writes and `2048` Q RoPE write words exactly, accepts the
  same `31` read requests / `4944` read-response words and `2049` write
  requests / `4096` write-data words, reports max read latency `5`, covers the
  same invalid-descriptor no-write path, and trace-audits cache write-back from
  `0x4_1430_0800` through `0x4_144E_09FC` before Q RoPE writes from
  `0x4_1502_5140` through `0x4_1502_713C`.
- `38_export_qmap_attention_score_value_image.py` exports the next QMAP
  attention-body runtime packet at `0x4_0503_0000`. It uses five active
  descriptors for metadata, Q RoPE input, the K/V cache descriptor, the
  softmax exp LUT, and `attn_out[2048]` output scratch. It now also accepts
  dynamic score/value vector prefixes and a runtime `--qmap-base`, so the same
  packet format can target true Layer 1 data.
- `qmap_attention_score_value_compute_path.sv` composes
  `attention_score_stage.sv` and `attention_softmax_value_stage.sv` behind a
  memory-mapped QMAP boundary. It reads Q RoPE and the exp LUT from QMAP
  descriptors, uses `kv_cache_addr_gen.sv` to turn K/V requests into exact
  4-byte cache reads, streams scaled scores directly into softmax/value, and
  writes `attn_out[2048]` back to the descriptor output buffer. The Icarus
  testbench matches `10240` K-cache reads, `10240` V-cache reads, and `2048`
  `attn_out` write words exactly. The passing run accepts `20496` read
  requests / `22961` read-response words, one output write request /
  `2048` write-data words, injects `7826` read-request stall cycles and `999`
  write-data stall cycles with max read latency `8`, and covers an invalid
  exp-LUT dtype descriptor that raises error with no writes. The trace audit
  confirms K reads finish before V reads, V/capture finish before write-back,
  and all wrapper states are covered.
- The attention score and softmax/value golden-vector exporters are now
  parameterized for true Layer 1 inputs. The current Layer 1 score/value packet
  uses QMAP base `0x4_1503_0000`, position `4`, cache length `5`, and writes
  `attn_out[2048]` at `0x4_1503_2980`. Its Icarus run matches all `10240`
  K-cache reads, all `10240` V-cache reads, and all `2048` attention output
  write words exactly, accepts `20496` read requests / `22961` read-response
  words and one write request / `2048` write-data words, reports max read
  latency `8`, and covers the same invalid exp-LUT no-write path. The trace
  audit confirms K reads from `0x4_1410_0000` through `0x4_141E_09FC`, then V
  reads from `0x4_1420_0000` through `0x4_142E_09FC`, then the output write
  request at `0x4_1503_2980`.
- `39_export_qmap_o_proj_image.py` exports the next QMAP per-layer body packet
  at `0x4_0504_0000` by default and now accepts `--layer-id`, `--qmap-base`,
  `--weight-base`, and `--scale-base`. It uses six active descriptors for
  metadata, `attn_out[2048]`, persistent `o_proj` Q4 weight, persistent Q2.14
  scale, `o_proj_out[1024]` output scratch, and expected/debug output words.
  The default Layer 0 persistent simulation addresses are `0x4_0600_0000` for
  the packed Q4 weight tensor and `0x4_0610_0000` for the scale tensor. The
  current true Layer 1 packet is `layer1_o_proj_runtime.qmap.bin` at
  `0x4_1504_0000`, with persistent Layer 1 `o_proj` weight/scale bases
  `0x4_0700_0000` and `0x4_0710_0000`.
- `qmap_o_proj_compute_path.sv` consumes that packet behind the same
  project-local memory request/write interface. It validates descriptor IDs,
  roles, dtypes, ranks, dimensions, bytes, layer id, and matrix id; reads
  `attn_out[2048]` in eight 1024-byte bursts; then reads each of the 1024
  persistent Q4 weight rows and scale rows, runs `q4_gemv_row_1024.sv` with
  `INPUT_SIZE=2048`, stores each converted Q12.12 row word, and writes
  `o_proj_out[1024]` as one 4096-byte output burst. The Icarus testbench
  matches all `1024` output words exactly, accepts `2070` read requests /
  `280992` read-response words across the normal plus invalid-descriptor
  runs, accepts one output write request / `1024` write-data words, covers
  request/response/write-data backpressure and a busy-period spurious start,
  and covers an invalid Q4 weight group-size descriptor that raises error with
  no writes. Trace audit confirms `1024` row stores from row 0 to row 1023,
  one output write burst after the final row store, `wr_last=1` on the final
  write word, and non-adjacent done pulses.
- The same QMAP `o_proj` wrapper now passes with true Layer 1 data. The Layer 1
  run uses QMAP base `0x4_1504_0000`, reads `attn_out[2048]` from
  `0x4_1504_0540`, reads `1024` persistent Q4 weight rows from
  `0x4_0700_0000` through `0x4_070F_FC00`, reads `1024` persistent scale rows
  from `0x4_0710_0000` through `0x4_0710_FFC0`, and writes exact
  `o_proj_out[1024]` at `0x4_1504_2540`. The passing Icarus run matches all
  `1024` output words exactly, accepts `2070` read requests /
  `280992` read-response words across normal plus invalid runs, and confirms
  the invalid descriptor path issues only QMAP reads and no writes.
- The same QMAP post-attention residual/RMSNorm wrapper now passes with true
  Layer 1 data. The Layer 1 packet uses QMAP base `0x4_1505_0000`, reads
  residual input from `0x4_1505_0540`, reads `o_proj_out[1024]` from
  `0x4_1505_1540`, reads post-attention gamma from `0x4_1505_2540`, then
  writes exact `post_attention_hidden[1024]` at `0x4_1505_3540` and exact
  `post_norm[1024]` at `0x4_1505_4540`. The passing Icarus run matches both
  output buffers exactly, accepts `30` read requests / `3616` read-response
  words across normal plus invalid runs, accepts two write requests /
  `2048` write-data words, reports `sum_squares=75359102` and
  `inv_rms=247634`, and confirms the invalid descriptor path issues only QMAP
  reads and no writes.
- `40_export_qmap_post_attention_residual_norm_image.py` exports the next QMAP
  per-layer body packet at `0x4_0505_0000`. It uses eight active descriptors
  for metadata, residual input, `o_proj_out[1024]`, signed post-attention
  RMSNorm gamma, post-attention hidden output, post-norm output, expected
  hidden, and expected norm. The image is `0x8000` bytes and keeps all vectors
  as descriptor-visible 32-bit words, with 24-bit or 16-bit element contracts
  recorded in the descriptors.
- `qmap_post_attention_residual_norm_compute_path.sv` consumes that packet
  behind the project-local memory request/write interface. It validates
  descriptor IDs, roles, dtypes, ranks, flags, dimensions, byte counts, layer
  id, stage id, and `o_proj` matrix id; reads residual input, `o_proj_out`, and
  post-attention gamma as three groups of four 1024-byte bursts; runs
  `post_attention_residual_norm_stage.sv`; then writes exact
  `post_attention_hidden[1024]` and `post_norm[1024]` as two 4096-byte output
  bursts. The Icarus testbench matches both write-back buffers exactly,
  accepts `30` read requests / `3616` read-response words across the normal
  plus invalid-descriptor runs, accepts two output write requests / `2048`
  write-data words, covers request/response/write-data backpressure and a
  busy-period spurious start, and covers an invalid gamma dtype descriptor that
  raises error with no writes. The normal run reports `residual_count=1024`,
  `stage_cycle_count=3132`, `sum_squares=33222329`, `inv_rms=372827`, and no
  residual or norm saturation. Trace audit confirms the stage-complete snapshot
  and first hidden write request occur on cycle `10616`, write data runs from
  cycle `10617` through `13281`, the normal done pulse is cycle `13286`, and
  normal/bad done pulses are not adjacent.
- `41_export_qmap_mlp_gate_up_image.py` exports the next QMAP per-layer body
  packet at `0x4_0506_0000`. It uses ten active descriptors for metadata,
  `post_norm[1024]`, persistent Layer 0 gate/up Q4 weights, persistent gate/up
  Q2.14 scales, gate/up output buffers, and expected/debug gate/up outputs.
  The runtime image is `0xE000` bytes. Persistent simulation bases are
  `0x4_0620_0000` for gate weight, `0x4_0638_0000` for gate scale,
  `0x4_0640_0000` for up weight, and `0x4_0658_0000` for up scale. The
  exporter now also accepts `--layer-id`, `--qmap-base`, and explicit gate/up
  weight/scale bases. With `--layer-id 1`, `--qmap-base 0x4_1506_0000`,
  `--gate-weight-base 0x4_0720_0000`, `--gate-scale-base 0x4_0738_0000`,
  `--up-weight-base 0x4_0740_0000`, and `--up-scale-base 0x4_0758_0000`, the
  same packet contract now passes with true Layer 1 post-norm input and true
  Layer 1 MLP gate/up weights.
- `qmap_mlp_gate_up_compute_path.sv` consumes that packet behind the
  project-local memory request/write interface. It validates descriptor IDs,
  roles, dtypes, ranks, flags, dimensions, byte counts, stage id, layer id,
  and gate/up matrix ids; reads `post_norm[1024]` in four 1024-byte bursts;
  then for each of 3072 rows reads gate weight, gate scale, up weight, and up
  scale from persistent DDR regions, runs two `q4_gemv_row_1024.sv` cores in
  parallel, stores the converted Q12.12 gate/up row words, and writes
  `gate[3072]` plus `up[3072]` as two 12288-byte output bursts. The Icarus
  testbench matches both write-back buffers exactly, accepts `12314` read
  requests / `837280` read-response words across the normal plus
  invalid-descriptor runs, accepts two output write requests / `6144`
  write-data words, covers request/response/write-data backpressure and a
  busy-period spurious start, and covers an invalid up-scale dtype descriptor
  that raises error with no writes. Trace audit confirms 3072 row completions,
  gate/up write requests at `0x4_0506_1940` and `0x4_0506_4940`, both length
  12288 bytes, and non-adjacent normal/error done pulses. The true Layer 1
  run uses `qmap_mlp_gate_up_compute_path_layer1_trace.csv`; it has normal
  read counts of `qmap=11`, `activation=4`, `gate_weight=3072`,
  `gate_scale=3072`, `up_weight=3072`, and `up_scale=3072`, writes gate/up at
  `0x4_1506_1940` and `0x4_1506_4940`, has zero unknown reads, zero normal
  error rows, no invalid-path writes, and exact gate/up write-back.
- `42_export_qmap_mlp_silu_mul_image.py` exports the next QMAP per-layer body
  packet. It now accepts `--layer-id`, `--qmap-base`, and
  `--gate-up-qmap-prefix`, while preserving the same six-descriptor packet
  contract. The default Layer 0 packet remains at `0x4_0507_0000`; the current
  true Layer 1 packet is `layer1_mlp_silu_mul_runtime.qmap.bin` at
  `0x4_1507_0000`. Both images are `0xE000` bytes, consume descriptor-visible
  gate/up `[3072]` plus a 1025-word UQ0.16 sigmoid LUT, and write
  `mlp_hidden[3072]`.
- `qmap_mlp_silu_mul_compute_path.sv` consumes that packet behind the
  project-local memory request/write interface. It validates descriptor IDs,
  roles, dtypes, ranks, flags, dimensions, byte counts, stage id, and layer id;
  reads gate, up, and LUT chunks from the QMAP image; streams them through
  `mlp_silu_mul_stage.sv`; and writes a single 12288-byte hidden output burst.
  The Icarus testbench matches all hidden write-back words exactly for Layer 0
  and true Layer 1. The true Layer 1 run uses
  `qmap_mlp_silu_mul_compute_path_layer1_trace.csv`; it has normal read counts
  of `qmap=7`, `gate=12`, `up=12`, and `lut=5`, normal read words of `qmap=208`,
  `gate=3072`, `up=3072`, and `lut=1025`, writes hidden at `0x4_1507_7980`,
  has zero unknown reads, zero normal error rows, no invalid-path writes, and
  monotonic stage output progress through all `3072` hidden words.
- RTL source files now use explicit `input wire logic` ports. This keeps
  `default_nettype none` enabled while satisfying Vivado 2025.1 synthesis,
  which rejects plain `input logic` as an implicit net in this flow.

## Hardware Bring-Up Status

PS-to-PL DDR4 access is complete as a hardware checkpoint. The first
PL-initiated QMAP dot64 read/compute smoke design passed on hardware, and the
row1024 QMAP read/compute smoke design has now also been integrated into the
Vivado block design, synthesized, implemented, bitstream-generated, exported
for Vitis, and passed on hardware.

Current Vivado/Vitis artifacts:

- Vivado project: `FPGA_Project/Vivado_Project/LLM_FPGA.xpr`
- Current exported hardware handoff:
  `FPGA_Project/Vivado_Project/llm_system_qmap_row1024_pl_master.xsa`
- Current generated bitstream:
  `FPGA_Project/Vivado_Project/LLM_FPGA.runs/impl_1/llm_system_wrapper.bit`
- Current short-path Vitis workspace:
  `F:\vws`
- Current Vitis row1024 platform:
  `F:\vws\p_r1024\`
- Current Vitis row1024 PL compute smoke app:
  `F:\vws\a_r1024\`
- Current Vitis DDR4 smoke baseline in the same clean workspace:
  `F:\vws\p_ddr4\` and `F:\vws\a_ddr4\`
- Previous proven PS-to-PL DDR4 platform:
  `Vitis_Workspace/llm_pl_ddr4_aux_reset_fix_platform/`
- Durable standalone smoke-test source:
  `FPGA_Project/software/pl_ddr4_smoke/main.c`
- Durable QMAP load/readback smoke-test source:
  `FPGA_Project/software/qmap_load_smoke/main.c`
- Embedded QMAP dot64 C header for that smoke app:
  `FPGA_Project/software/qmap_load_smoke/qmap_dot64_image.h`
- Durable QMAP PL compute smoke-test source:
  `FPGA_Project/software/qmap_pl_compute_smoke/main.c`
- Durable QMAP PL compute embedded image header:
  `FPGA_Project/software/qmap_pl_compute_smoke/qmap_dot64_image.h`
- Durable QMAP row1024 PL compute smoke-test source:
  `FPGA_Project/software/qmap_row1024_pl_compute_smoke/main.c`
- Durable QMAP row1024 PL compute embedded image header:
  `FPGA_Project/software/qmap_row1024_pl_compute_smoke/qmap_row1024_image.h`
- Current Vitis DDR4 workspace app copy:
  `F:\vws\a_ddr4\src\main.c`
- Current QMAP row1024 Vitis workspace app copy:
  `F:\vws\a_r1024\src\main.c`

Current PS-to-PL memory fabric:

```text
M_AXI_HPM0_FPD
  -> AXI SmartConnect
      M00_AXI -> AXI BRAM Controller -> Block Memory Generator
      M01_AXI -> AXI Clock Converter -> ddr4_0/C0_DDR4_S_AXI
      M02_AXI -> AXI GPIO DDR4 status register
      additional AXI GPIO slaves for QMAP smoke control/status/result

qmap_row1024_axi_smoke_0/M_AXI
  -> AXI SmartConnect S01_AXI
      -> AXI Clock Converter -> ddr4_0/C0_DDR4_S_AXI
```

Current address map:

| Space / IP | Base | High | Size | Status |
| --- | ---: | ---: | ---: | --- |
| PS DDR low memory | `0x0000_0000` | `0x7FEF_FFFF` | about 2 GiB minus reserved top window | exported in current BSP |
| AXI BRAM smoke memory | `0xA000_0000` | `0xA000_1FFF` | 8 KiB | passed in current hardware smoke |
| DDR4 status AXI GPIO | `0xA001_0000` | `0xA001_FFFF` | 64 KiB | passed in current hardware smoke |
| QMAP smoke control/status AXI GPIO | `0xA002_0000` | `0xA002_FFFF` | 64 KiB | passed in dot64 and row1024 PL master hardware smoke |
| QMAP smoke result AXI GPIO | `0xA003_0000` | `0xA003_FFFF` | 64 KiB | passed in dot64 and row1024 PL master hardware smoke |
| PL DDR4 | `0x4_0000_0000` | `0x4_1FFF_FFFF` | 512 MiB | passed in current hardware smoke |

DDR4 status GPIO bit layout at `0xA001_0000`:

| Bit | Signal | Good Value |
| ---: | --- | ---: |
| 0 | `ddr4_0/c0_init_calib_complete` | `1` |
| 1 | `ddr4_0/c0_ddr4_ui_clk_sync_rst` | `0` |
| 2 | `proc_sys_reset_0/peripheral_aresetn` | `1` |

The good DDR4 status word is therefore `0x5`.

Current reset wiring:

- `ddr4_0/c0_ddr4_ui_clk_sync_rst` feeds the DDR UI-domain
  `proc_sys_reset_0/ext_reset_in`.
- `proc_sys_reset_0/aux_reset_in` is tied high because
  `proc_sys_reset_0` treats that input as active-low.
- `proc_sys_reset_0/mb_debug_sys_rst` is tied low.
- `proc_sys_reset_0/peripheral_aresetn` drives the DDR-side reset release path,
  including `ddr4_0/c0_ddr4_aresetn` and the AXI Clock Converter M_AXI-side
  reset.

Current board-run result:

- Boot mode: JTAG boot (`0000`)
- Serial path: CH340 USB-UART on COM110, 115200 baud
- Previous PS-to-PL DDR4 smoke launch style: `psu_init.tcl` enabled, FSBL
  initialization disabled
- Current QMAP PL master launch style: FSBL initialization enabled,
  `psu_init.tcl` disabled. This avoids a DAP transaction error while polling
  PS DDR PHY register `0xFD080030` in `psu_init.tcl`.
- On Windows, use a short Vitis workspace path such as `F:\vws` and short
  component names such as `p_ddr4`, `a_ddr4`, `p_r1024`, and `a_r1024`.
  Longer paths can make Vitis/CMake/Ninja fail BSP builds while opening
  generated `.obj.d` dependency files.
- AXI BRAM smoke passed at:
  - `0x0_A0000000`
  - `0x0_A0000004`
  - `0x0_A0000008`
  - `0x0_A0000400`
  - `0x0_A0001FFC`
- DDR4 status reported:
  `DDR4 status raw 0x5 calib_complete=1 ui_reset=0 axi_resetn=1`
- PL DDR4 smoke passed at:
  - `0x4_00000000`
  - `0x4_00000004`
  - `0x4_00001000`
  - `0x4_10000000`
  - `0x4_1FFFFFFC`
- QMAP load/readback smoke passed:
  - image SHA256:
    `b56319cf576fe8486e3586ee49a4194323cdfe4f4e6208c8b6057e741e5978d4`
  - image base: `0x4_1B10_0000`
  - image size: `0x600` / 1536 bytes
  - descriptor table: `0x4_1B10_0100`
  - payload base: `0x4_1B10_0500`
  - descriptor count/capacity: `4` / `8`
  - byte-for-byte PS readback compare: passed
- QMAP dot64 PL master smoke passed:
  - PS writes the same embedded QMAP image to `0x4_1B10_0000`
  - QMAP readback compare passes for 1536 bytes
  - PS clears and starts the PL compute path through the control GPIO at
    `0xA002_0000`
  - final PL status is `0xA`
  - result GPIO values at `0xA003_0000` are
    `partial_sum_low32=0x60AF` and `scaled_sum_q26_low32=0x2E1366`
- QMAP row1024 PL master smoke passed:
  - PS writes the embedded 4096-byte QMAP image to `0x4_1B20_0000`
  - QMAP readback compare passes for 4096 bytes
  - header checks pass for magic `0x50414D51`, version `1`, descriptor count
    `4`, descriptor capacity `8`, descriptor table `0x4_1B20_0100`, payload
    base `0x4_1B20_0500`, and image size `0x1000`
  - PS clears and starts the PL compute path through the control GPIO at
    `0xA002_0000`
  - final PL status is `0xA`
  - result GPIO values at `0xA003_0000` are
    `row_sum_q26_low32=0xFFCA_DDC7` and
    `expected_row_sum_q26_low32=0xFFCA_DDC7`, both representing `-3482169`

This proves the first PS-to-PL DDR4 path through
`M_AXI_HPM0_FPD -> axi_smc -> axi_clock_converter_0 -> ddr4_0/C0_DDR4_S_AXI`
is working for 32-bit standalone smoke-test accesses.

It also proves PL AXI master read paths from real PL DDR4 into the QMAP/Q4
dot64 and row1024 compute chains:
`qmap_*_axi_smoke_0/M_AXI -> axi_smc -> axi_clock_converter_0 ->
ddr4_0/C0_DDR4_S_AXI`.

Previous useful checkpoint:

- The earlier BRAM-only hardware checkpoint used `M_AXI_HPM0_LPD` and
  `0x8000_0000` through `0x8000_1FFF`.
- The current integrated PL DDR4 design uses `M_AXI_HPM0_FPD`; the BRAM smoke
  aperture moved to `0xA000_0000`.

## Open Gaps

- PL DDR4 is accessible from PS. The PL QMAP descriptor reader, payload
  fetcher, Q4 dot64 compute chain, row1024 compute chain, and read-only AXI4
  adapter are now wrapped for Vivado, integrated into block designs, and have
  passed board validation against real PL DDR4. The QKV projection compute path
  and AXI write adapter are now wrapped into a Vivado-facing AXI read/write top
  and pass full-size local AXI memory-model simulation, but they have not yet
  run on hardware.
- QMAP v1 now defines the first descriptor-based PL DDR4 staging contract. The
  dot64 image exporter and standalone PS loader/readback app have passed on
  hardware. The first PL descriptor reader, payload fetcher, and Q4 dot64
  compute hookup are now wrapped in a synthesizable dot64 smoke controller.
  This wrapper is a bring-up micro-kernel, not the final full-model scheduler.
- `Source/QMAP_FORMAT.md` now defines the next formal inference direction:
  persistent model manifests for tensors loaded once into PL DDR4, plus
  runtime work packets for each PL kernel step. The first formal work packet is
  Layer 0 Q/K/V projection with descriptor-provided input, weight/scale, and
  output-buffer addresses.
- The full-model memory-map layout for Q4 artifacts, KV cache, activation
  buffers, RoPE tables, and debug regions is still draft.
- RMSNorm, GEMV, RoPE, K/V cache append, attention score, softmax/value,
  `o_proj`, post-attention residual/RMSNorm, MLP gate/up projection, MLP
  SiLU/multiply, MLP down projection, final MLP residual add, full-model final
  RMSNorm, the tiled LM-head argmax core, the memory-backed 1024-row LM-head
  scan-window wrapper, the runtime LM-head tile scheduler, and the QMAP
  descriptor-backed LM-head wrapper are validated in focused simulations,
  including the full `151936`-row / `9496`-tile LM-head vocabulary sweep. The
  first memory-mapped one-token tail wrapper now also composes final RMSNorm
  write-back into that full-vocabulary QMAP LM-head path. The first
  memory-mapped per-layer body slice now wraps Q/K norm, RoPE, and K/V cache
  append behind QMAP descriptors, producing exact K/V cache writes and exact
  Q RoPE write-back locally. The next QMAP attention-body slice wraps
  attention score plus softmax/value behind descriptor-visible Q RoPE, K/V
  cache, exp LUT, and `attn_out` buffers. The following QMAP `o_proj` slice
  now consumes `attn_out[2048]`, persistent Q4 `o_proj` weight/scale
  descriptors, and writes exact `o_proj_out[1024]` locally for Layer 0 and
  true Layer 1 packet instances. The following QMAP post-attention
  residual/RMSNorm slice now consumes descriptor-visible residual input,
  `o_proj_out[1024]`, and signed post-attention gamma, then writes exact
  post-attention hidden and post-norm buffers locally for Layer 0 and true
  Layer 1 packet instances. The following QMAP MLP gate/up slice now consumes
  descriptor-visible `post_norm[1024]`, persistent Q4 gate/up weight/scale
  descriptors, and writes exact gate/up `[3072]` buffers locally for Layer 0
  and true Layer 1 packet instances.
  The following QMAP MLP SiLU/multiply slice now consumes descriptor-visible
  gate/up `[3072]` plus a fixed UQ0.16 sigmoid LUT, runs
  `mlp_silu_mul_stage.sv`, and writes exact `mlp_hidden[3072]` locally for
  Layer 0 and true Layer 1 packet instances.
- The following QMAP MLP down-projection slice now consumes descriptor-visible
  `mlp_hidden[3072]`, reads persistent per-layer Q4 down-proj weight/scale rows,
  splits each 1536-byte weight row into 1024-byte plus 512-byte reads, and
  writes exact `down_out[1024]` locally for Layer 0 and true Layer 1 packet
  instances. The default Layer 0 packet remains at `0x4_0508_0000` with
  persistent bases `0x4_0660_0000` and `0x4_0678_0000`. The true Layer 1 packet
  now passes at `0x4_1508_0000` with persistent bases `0x4_0760_0000` and
  `0x4_0778_0000`, output write-back at `0x4_1508_3940`, `1024` completed rows,
  `3091` normal read requests, `421072` normal read-response words, one
  4096-byte output write, zero write mismatches, and `max_abs_output=0`. Trace
  audit confirms normal read counts `qmap=7`, `activation=12`, `weight=2048`,
  `scale=1024`, zero unknown reads, monotonic `rows_done` through `1024`, and an
  invalid scale-dtype descriptor path that issues only QMAP reads and no writes.
- The following QMAP final MLP residual slice now consumes descriptor-visible
  `post_attn_hidden[1024]` and `down_out[1024]`, runs
  `mlp_residual_add_stage.sv`, and writes exact Q14.10 `layer_out[1024]`
  locally. The passing Icarus run covered a valid full write-back, a
  bad-down-dtype descriptor path with no writes, and a malformed payload
  `last` protocol path with no writes. The normal run used `14` read requests,
  `2224` read-response words, one 4096-byte output write request, and `1024`
  write-data words. The full three-run test covered `27` read requests,
  `2577` read-response words, request/response/write-data backpressure,
  a busy-period spurious start, and CSV trace audit. The same exporter and
  testbench now also pass true Layer 1 data at QMAP base `0x4_1509_0000`:
  output write-back is `0x4_1509_2540`, normal reads split into `qmap=6`,
  `post=4`, and `down=4` requests, normal counters are `14` read requests /
  `2224` read-response words plus one `4096`-byte output write, and the bad
  descriptor/protocol paths still issue no writes.
- The first local Layer 0 body scheduler now starts the already passing QMAP
  post-attention residual/RMSNorm, MLP gate/up, MLP SiLU/multiply, MLP down,
  and final MLP residual wrappers in order behind one memory request/write
  interface. Its testbench patches downstream descriptor `base_addr` fields so
  each later wrapper reads the previous wrapper's actual write-back buffer,
  not a standalone golden input copy. The passing Icarus run covered
  `15465` normal read requests, `1270961` normal read-response words, seven
  write bursts, `13312` write-data words, exact write-back for every
  intermediate buffer through `layer_out[1024]`, request/response/write-data
  backpressure, busy-period spurious start, and an invalid first-stage
  descriptor path with no writes.
- The wider local Layer 0 scheduler now starts QMAP attention front-end,
  attention score/value, `o_proj`, and the passing Layer 0 body scheduler in
  order behind one memory request/write interface. Its testbench patches
  downstream descriptor `base_addr` fields so Q RoPE feeds score/value,
  `attn_out[2048]` feeds `o_proj`, `o_proj_out[1024]` feeds post-attention
  residual/RMSNorm, and every later MLP slice reads the previous slice's actual
  write-back buffer. The passing Icarus run covered `38055` normal read
  requests, `1579650` normal read-response words, `2058` write requests,
  `20480` write-data words, exact K/V cache, Q RoPE, attention, `o_proj`, MLP,
  and final `layer_out[1024]` write-back, plus an invalid first-stage
  descriptor path with no writes. CSV trace audit confirmed four top-level
  stage transitions and the ten non-cache output write requests after the
  `2048` K/V cache single-word writes.
- The first local Layer 0 compute scheduler now starts the full QMAP QKV
  projection packet first, then starts the passing wider Layer 0 scheduler
  behind the same memory request/write interface. Its testbench patches the
  attention front-end Q/K/V input descriptor bases to the Q/K/V output buffers
  written by the QKV projection stage, then preserves the existing downstream
  descriptor patch chain through final `layer_out[1024]`. The passing Icarus
  run covered `46264` normal read requests, `2138130` normal read-response
  words, `6154` write requests, `24576` write-data words, exact Q/K/V,
  K/V cache, Q RoPE, attention, `o_proj`, MLP, and final `layer_out[1024]`
  write-back. Error coverage includes a QKV-stage invalid activation dtype
  path with `13` read requests, `400` read-response words, and no writes, plus
  a later attention front-end invalid descriptor path after QKV has completed.
  CSV trace audit confirmed QKV single-word writes of `2048/1024/1024` Q/K/V
  rows and a `731`-cycle gap between the last QKV V write request and the first
  downstream read from a QKV-produced buffer.
- The first local one-token/layer-loop scheduler boundary now wraps the QMAP
  layer compute chain and makes the reusable loop contract explicit:
  `layer_start_index`, `layer_count`, token `position`, input/output
  hidden-buffer bases, KV-cache base, flattened per-layer QMAP packet base
  tables, one shared memory port, layer done/error masks, and aggregate
  counters. The normal Layer 0 case (`layer_start_index=0`, `layer_count=1`)
  reuses the full QKV-through-Layer0 golden/write-back checks and observes the
  same totals: `46264` read requests, `2138130` read-response words, `6154`
  write requests, and `24576` write-data words. The focused true two-layer run
  now also passes locally: Layer 0 runs from QKV through final
  `layer_out[1024]`, Layer 1 consumes that Layer 0 write-back through chained
  QKV, attention, `o_proj`, post-attention residual/RMSNorm, MLP gate/up,
  SiLU/multiply, MLP down, and final MLP residual packets, and the loop exits
  with layer done mask `0x3`. The passing run completed at cycle `13385925`
  with aggregate totals `92528` read requests, `4276260` read-response words,
  `12308` write requests, and `49152` write-data words. Producer write totals
  across both layers were QKV `4096/2048/2048`, cache `4096`, Q RoPE `4096`,
  attention `4096`, `o_proj` `2048`, post hidden/norm `2048/2048`, gate/up
  `6144/6144`, SiLU `6144`, down `2048`, and final layer output `2048`, with
  `mismatch_count=0`, `write_mismatches=0`, and `max_abs=0`.
  The chained artifact flow has now been factored into
  `Qwen3-0.6B-Base/python_each_module/47_export_chained_layer_qmap_artifacts.py`.
  It emits per-layer QKV/body QMAP packets from a selected previous-layer
  output and records a chained-layer manifest. Layer 2 artifacts have been
  generated from `qmap_layer1_chained_mlp_residual_add_expected_words32.hex`;
  their QMAP bases are `0x4_2008_0000` for QKV and `0x4_2502_0000` through
  `0x4_2509_0000` for body stages, with persistent Layer 2 weight windows at
  `0x4_0800_0000` through `0x4_0878_0000`. Layer 2 attention score/value and
  final MLP residual-add unit TBs pass with exact golden write-back. The
  one-token scheduler TB memory/scoreboard model now recognizes Layer 2
  packet, cache, and persistent-weight windows; focused `+l2_only +fastmem
  +notrace +progress` passes at cycle `6692967` with `active_layer=2`,
  `layer_done_mask=0x4`, `mismatch_count=0`, `write_mismatches=0`, and
  `max_abs=0`. A compiled `+true3_only` entry is available for a full Layer 0
  -> Layer 1 -> Layer 2 long regression, but that longer Icarus run is not a
  routine smoke test.
  Error coverage still includes QKV-stage and later frontend error propagation,
  a missing selected Layer 0 QKV base-table entry, a missing Layer 1 table
  entry, and out-of-range `layer_start_index=28`, all with the expected masks
  and `0` read/write traffic for pre-start validation failures. A separate
  validation-only TB also covers zero `layer_count` and zero input-hidden base
  as no-memory exits.
- The LM-head QMAP wrapper now proves descriptor-visible scan control over the
  1024-row scan window, a shorter 368-row prefix, and the complete full-vocab
  path. Full-vocab simulation used Vivado xsim with the same scheduler and
  reported exact token `264`, score `1365150750`, and `max_abs_logit_diff=0`.
- The QMAP final-token tail wrapper proves the first descriptor-visible
  composed inference tail:
  `final_hidden + final_norm.gamma -> final_norm write-back -> full-vocab
  LM-head argmax -> output token/score`. This is local RTL/xsim evidence, not
  a new board checkpoint.
- Cache coherency, bulk transfer strategy, and any future DMA policy are still
  open. The current proven path is direct PS memory-mapped 32-bit access.
- The isolated Q4 GEMV smoke-test phase is considered complete enough after the
  QMAP dot64 and row1024 PL-master board passes. Do not spend the next project
  step on an extra row-loop or small multi-row tile smoke test unless a future
  integration issue makes that specific debug slice necessary.
- A first Vivado bring-up attempt for row1024 showed LUT over-utilization with
  the earlier full-row payload buffering. The current row1024 RTL is
  resource-reduced for bring-up by fetching and computing 4 groups at a time,
  and that reduced path has now passed synthesis, implementation, bitstream
  generation, Vitis launch, and board validation.

## Immediate Next Step

Use the full locally passing one-token Layer 0 loop-boundary scheduler,
full-model final RMSNorm stage, tiled LM-head argmax stage, memory-backed
LM-head scan-window wrapper, runtime LM-head tile scheduler, QMAP
descriptor-backed full-vocabulary LM-head wrapper, the QMAP final-token tail
wrapper, the true Layer 1 QKV AXI pass, the true Layer 1 QMAP attention
front-end pass, the true Layer 1 QMAP attention score/value pass, the true
Layer 1 QMAP `o_proj` pass, and the true Layer 1 QMAP post-attention
residual/RMSNorm pass, the true Layer 1 QMAP MLP gate/up pass, and the true
Layer 1 QMAP MLP SiLU/multiply pass, the true Layer 1 QMAP MLP down pass, the
true Layer 1 QMAP final MLP residual pass, the focused true Layer 0 -> Layer 1
loop pass, the reusable chained-layer artifact generator, and the focused
Layer 2-only scheduler pass as regression baselines. Continue locally by
composing the chained layer-loop output with the existing final-token tail
wrapper, or by running the compiled `+true3_only` long regression only when a
full three-layer scheduler proof is worth the wall time.

Recommended next slice:

1. Keep the passing row1024 board design, the full QKV AXI simulation, and
   `tb_qk_norm_rope_kv_cache_stage.sv`, `tb_attention_score_stage.sv`, and
   `tb_attention_softmax_value_stage.sv`, `tb_o_proj_stage.sv`, and
   `tb_qmap_o_proj_compute_path.sv`, plus
   `tb_qmap_post_attention_residual_norm_compute_path.sv`,
   `tb_qmap_mlp_gate_up_compute_path.sv`,
   `tb_qmap_mlp_silu_mul_compute_path.sv`,
   `tb_qmap_mlp_down_compute_path.sv`,
   `tb_qmap_mlp_residual_add_compute_path.sv`,
   `tb_qmap_layer0_body_scheduler.sv`,
   `tb_qmap_layer0_full_scheduler.sv`,
   `tb_qmap_layer0_compute_scheduler.sv`,
   `tb_qmap_one_token_layer_scheduler.sv`,
   `tb_post_attention_residual_norm_stage.sv` and
   `tb_mlp_gate_up_proj_stage.sv`, `tb_mlp_silu_mul_stage.sv`, and
   `tb_mlp_down_proj_stage.sv` and `tb_mlp_residual_add_stage.sv`, as
   regression baselines. Keep `tb_final_rmsnorm_stage.sv`,
   `tb_lm_head_argmax_stage.sv`, `tb_lm_head_argmax_mem_stage.sv`, and
   `tb_lm_head_argmax_tile_scheduler.sv` plus
   `tb_qmap_lm_head_argmax_compute_path.sv` as the current full-model
   final-output regressions.
2. Treat the passing focused true Layer 0 -> Layer 1 loop plus the focused
   Layer 2-only scheduler run as the current model-facing regressions. Together
   they prove the true chained producer/consumer descriptor bases through
   final Layer 1 `layer_out[1024]` and prove the same table-driven scheduler
   contract can consume Layer 2 packet/cache/weight windows.
3. The next useful target is to compose the chained layer-loop output with the
   existing final-token tail wrapper and verify final RMSNorm plus
   full-vocabulary LM-head argmax from the chained layer output. Keep the full
   `+true3_only` scheduler run as a deliberate long regression, not the default
   next action.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Treat BF16/FP32 model data as software reference and validation input only.
  The first PL DDR4 weight layout and GEMV datapaths must assume custom Q4
  weight-only storage for large weights.
- Core project handoff Markdown files live under `Source/`; the root
  `README.md` remains the entry point.
- Keep Q4 quantization semantics in `Source/Q4_FORMAT.md` and DDR tensor
  placement in `Source/QMAP_FORMAT.md`.
- Keep PS-side code as orchestration/support only. Do not move model math for
  prefill or decode back into PS unless the project direction explicitly
  changes.
- Treat `Vitis_Workspace/` and `F:\vws\` as local regenerated workspaces. Keep
  durable application source under `FPGA_Project/software/` and recreate Vitis
  platform/application/build outputs from the Vivado hardware export. On
  Windows, prefer the short `F:\vws` path for new Vitis workspaces to avoid BSP
  build failures from generated path lengths.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- Keep `Source/CURRENT_STATE.md` concise. Store detailed validation logic in
  scripts and summarize only durable results here.
