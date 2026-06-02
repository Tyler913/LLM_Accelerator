# Current State

Last updated: 2026-06-02

This file is the concise working-state handoff. For stable project context,
read `PROJECT_CONTEXT.md` first. For workflow rules, read `AGENTS.md`.

## Current Goal

Build a first FPGA demo that can take a prompt and generate text with the local
`Qwen/Qwen3-0.6B-Base` model.

Current first-version target:

- Serial prefill and single-token cached decode
- Greedy argmax text continuation
- PS-side tokenizer, detokenizer, and PL control
- PL-side `next_token = run_one_token(input_token, position)` path
- PL owns all model-side prefill and decode math; PS schedules tokens, loads
  artifacts, writes control registers, polls status, and handles text I/O/debug
- Initial context target: 128 or 256 tokens
- Required PL DDR4 weight path: custom Q4 weight-only format

## Software Baseline

The FP32 Python reference is complete for the first-version target. It covers
serial prefill/decode, full 28-layer cached single-token decode, and focused
module checks for the FPGA bring-up path.

Key locations:

- Full-model references: `Qwen3-0.6B-Base/pc_testing/`
- Module references: `Qwen3-0.6B-Base/python_each_module/`
- Module validation index:
  `Qwen3-0.6B-Base/python_each_module/README.md`
- Test vectors:
  `artifacts/test_vectors/qwen3_0p6b_fp32_v0/` (ignored by Git)
- Q4 bring-up vectors:
  `artifacts/test_vectors/qwen3_0p6b_q4_v0/` (ignored by Git)

Useful commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/13_export_q4_gemv_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/14_verify_q4_gemv_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/15_export_q4_projection_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/16_profile_rmsnorm_ranges.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/17_export_rmsnorm_fixed_vectors.py
```

## Current RTL Progress

The project now has reusable hand-written RTL compute blocks for Q4 GEMV,
RMSNorm, and the first RoPE attention-front-end stage.

Q4 GEMV stack:

- `q4_dot_product_64.sv`: one 64-element group dot product with signed int4
  weights and unsigned Q2.14 scale.
- `q4_gemv_row_1024.sv`: one 1024-wide output row using 16 parallel dot64
  engines.
- `q4_gemv_tile_1024.sv`: parallel row tile, default `OUT_ROWS=4`.
- `q4_gemv_projection_1024.sv`: projection controller that reuses one tile
  across serial output-row tiles.

Q4 GEMV fixed-point/interface state:

- Large weights use the project custom Q4 weight-only path from `Q4_FORMAT.md`.
- GEMV activation input now defaults to signed 24-bit `Q12.12`, matching the
  planned RMSNorm output path.
- The original 16-bit `Q4.12` bring-up vectors are still supported through
  explicit `ACT_WIDTH=16` testbench/module overrides.
- Product, partial, scaled, and row-accumulator widths are derived from
  `ACT_WIDTH`. Default 24-bit activation derives 28-bit products, 34-bit
  64-lane partial sums, 50-bit scaled group sums, and 56-bit row accumulators.

RMSNorm stack:

- `rmsnorm_sum_squares_1024.sv`: serial square-and-accumulate over one vector.
- `fixed_sqrt_u64.sv`: sequential trial-bit unsigned square root.
- `fixed_udiv.sv`: sequential unsigned shift-subtract divider.
- `rmsnorm_apply_1024.sv`: serial multiply-scale-saturate apply stage.
- `rmsnorm_1024.sv`: top-level controller across sum, mean/epsilon, sqrt,
  reciprocal, and apply.

RMSNorm fixed-point state:

- Input residual/RMSNorm stream: signed 24-bit `Q14.10`.
- Gamma: unsigned 16-bit `UQ8.8`.
- Sum-squares accumulator: unsigned 64-bit with `2*IN_FRAC` scaling.
- RMS/sqrt output: unsigned 24-bit `Q14.10`.
- Inverse RMS: unsigned 24-bit `UQ8.16`.
- RMSNorm output: signed 24-bit `Q12.12`, saturated to signed 24-bit range.
- Current epsilon uses `EPS_Q20=1`, representing about `9.54e-7`.

RoPE stack:

- `rope_qk_layer_128.sv`: first 128-d head-dimension RoPE stage for one
  token's Q/K heads. It accepts Q as `[16, 128]`, K as `[8, 128]`, shared
  `[128]` cos/sin vectors for the current position, and produces RoPE-applied
  Q/K outputs.

RoPE fixed-point/interface state:

- Q/K input: signed 24-bit `Q12.12`.
- Cos/sin input: signed 16-bit `Q1.15`.
- Q/K output: signed 24-bit `Q12.12`, saturated.
- Current implementation processes one scalar lane per cycle and keeps the
  flattened-vector bring-up style used by the existing RMSNorm/GEMV modules.

RMSNorm range profiling:

- `16_profile_rmsnorm_ranges.py` profiles all 113 RMSNorm modules in the local
  Qwen3 model.
- Latest prompt-suite profile covered 8 prompts and 257 prompt tokens plus one
  generated feedback token per prompt.
- Observed maxima: input abs about `6691.77`, output abs about `588.13`, gamma
  max `192`, inv_rms max about `46.08`, and sum_squares max about `4.55e7`.
- This profile justifies the current 24-bit RMSNorm formats for first RTL
  bring-up, but it is not a formal proof over all possible prompts.

## Latest Validation

Local RTL recheck on 2026-06-02:

- Re-ran every existing `FPGA_Project/sim/*.vvp` simulation locally; all passed.
- Key real-vector checkpoints remained unchanged:
  `tb_q4_gemv_projection_1024_real.sv` reported 284 waited cycles after start,
  and `tb_rmsnorm_1024_real.sv` reported 2104 waited cycles after start.
- `git status --short` was clean after the recheck; generated `.vcd`/`.vvp`
  files remain ignored.

Model tensor shape check on 2026-06-02:

- A `safetensors` inspection under `conda run -n llm_fpga ...` confirmed
  Layer 0 `q_norm.weight` and `k_norm.weight` are both shape `[128]`.
- The same check confirmed the expected Layer 0 projection and MLP matrix
  shapes already documented in `PROJECT_CONTEXT.md`.

RoPE RTL checks:

- `rope_qk_layer_128.sv` Icarus elaboration passed with:
  `iverilog -g2012 -tnull -s rope_qk_layer_128 FPGA_Project/rtl/rope_qk_layer_128.sv`
- A real-vector RoPE exporter and testbench are still pending.

Python/software checks:

- `12_verify_fpga_test_vectors.py` passed for the FP32 module vectors.
- `14_verify_q4_gemv_vectors.py` passed for the custom Q4 bring-up vectors.
- `15_export_q4_projection_vectors.py` exports real q_proj projection vectors.
- `16_profile_rmsnorm_ranges.py` produced the range profile summarized above.
- `17_export_rmsnorm_fixed_vectors.py` exports real Layer 0 input_layernorm
  fixed-point RTL vectors from the reference prompt path.

Q4 GEMV RTL checks:

- `tb_q4_dot_product_64.sv` passes with `partial_sum = 24751` and
  `scaled_sum_q26 = 3019622`.
- `tb_q4_gemv_row_1024.sv` passes q_proj row 0 with
  `row_sum_q26 = -3482169`.
- `tb_q4_gemv_tile_1024.sv` passes q_proj rows 0..3 with Q26 outputs
  `[-3482169, 7403300, 4069596, -6026990]`.
- `tb_q4_gemv_projection_1024.sv` passes two repeated 4-row projection tiles.
- `tb_q4_gemv_projection_1024_real.sv` passes real q_proj rows 0..15 with exact
  Q26 outputs and reported 284 waited cycles after start.
- Icarus elaboration passes for default-width q_proj-style
  `OUT_FEATURES=2048`, k/v-style `OUT_FEATURES=1024`, and explicit
  `ACT_WIDTH=16` compatibility.

RMSNorm RTL checks:

- `tb_rmsnorm_sum_squares_1024.sv` passes an 8-element smoke vector with
  `sum_squares = 31457454`.
- `tb_fixed_sqrt_u64.sv` passes integer and Q20-style square-root vectors with
  24-cycle normal latency.
- `tb_fixed_udiv.sv` passes integer, RMSNorm-style `2^26 / rms_q10`, and
  divide-by-zero vectors.
- `tb_rmsnorm_apply_1024.sv` passes nominal multiply-scale vectors and
  saturation vectors.
- `tb_rmsnorm_1024.sv` passes 8-element end-to-end +/-1.0 and +/-2.0 vectors
  with expected sum, mean, inv_rms, and Q12 outputs.
- `tb_rmsnorm_1024_real.sv` passes the real Layer 0 input_layernorm last-token
  vector exactly. The exported expected debug values are `sum_squares=589959`,
  `mean_square=576`, `sqrt_radicand=577`, `rms_q10=24`, `inv_rms=2796202`,
  `saturation=0`, and output `max_abs_diff=0`. The real-vector simulation
  reported 2104 waited cycles after start.
- VCD inspection confirmed expected handshakes for sqrt/divider/apply and the
  top-level `START_SUM -> START_SQRT -> START_DIV -> START_APPLY -> DONE`
  ordering.

## Open Gaps

- RMSNorm has real Layer 0 input_layernorm validation, but not yet broader
  coverage for `post_attention_layernorm`, `q_norm`, `k_norm`, or final norm.
- GEMV projection has real q_proj rows 0..15 validation, but not yet broad
  q/k/v/o/MLP coverage.
- RoPE now has an initial RTL module, but no real-vector testbench yet.
- Attention score/softmax/value accumulation, KV cache read/write, residual
  add, SiLU, MLP elementwise multiply, final LM-head scan, and greedy argmax
  are still pending.
- Current RTL uses flattened vectors for bring-up. Memory-mapped or streaming
  PS/PL integration is still a later step.

Stable reference prompt:

```text
The future of FPGA is
```

Known greedy checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## Hardware Bring-Up Snapshot

The first PS-to-PL AXI communication checkpoint is successful.

Development direction:

- Keep the project PL-first and RTL-first.
- Use hand-written Verilog/SystemVerilog for PL compute blocks by default.
- Keep PS-side bare-metal code as orchestration, loading, control, and
  validation support.

Current hardware checkpoint:

- Vivado project: `FPGA_Project/Vivado_Project/LLM_FPGA.xpr`
- Exported XSA: `FPGA_Project/Vivado_Project/llm_system_axi_bram_smoke.xsa`
- Vivado version/part: 2025.1.1, `xczu2eg-sfvc784-2-i`
- AXI smoke path:
  `M_AXI_HPM0_LPD -> SmartConnect -> AXI BRAM Controller -> Block Memory`
- AXI BRAM range: `0x8000_0000` through `0x8000_1FFF` (8 KiB)
- Vitis app source: `FPGA_Project/software/axi_bram_smoke/main.c`
- Confirmed run: 32-bit writes/reads pass at offsets `0x0`, `0x4`, `0x8`,
  `0x400`, and `0x1FFC`

Important bring-up notes:

- Boot mode used for the successful run: `0000` JTAG boot.
- Serial path: CH340 USB-UART, 115200 baud, UART0 on MIO42/MIO43.
- Board initialization used TCL / `psu_init.tcl`, not FSBL.
- AXI BRAM Controller must stay single-port for the current connected BRAM
  topology.
- Earlier Vitis path/bitstream issues were fixed by regenerating the platform,
  rebuilding the app, and pointing launch config at the platform bitstream.

## Immediate Next Step

Next phase: broaden real-vector coverage, then move into the attention
front-end.

1. Add a real-vector exporter for `rope_qk_layer_128.sv` using Layer 0
   q/k after q_norm/k_norm and the current-position RoPE cos/sin table.
2. Add `tb_rope_qk_layer_128.sv` and compare the RTL output against the
   exported fixed-point RoPE reference.
3. Optionally extend `17_export_rmsnorm_fixed_vectors.py` and
   `tb_rmsnorm_1024_real.sv` beyond Layer 0 `input_layernorm` to cover
   `post_attention_layernorm` and final norm cases.
4. Extend GEMV projection vectors beyond q_proj rows 0..15 when useful:
   q/k/v projection tiles first, then o_proj and MLP gate/up/down projections.
5. Continue the attention front-end after RoPE validation with KV cache
   append/read planning.
6. Keep PS-side C minimal until these PL compute blocks have stable standalone
   simulation evidence. The PS should remain orchestration/control, not model
   compute.
7. Continue refining `FPGA_MEMORY_MAP.md` when PL DDR4 is instantiated,
   especially PL DDR4 base/range, usable capacity, PS-to-PL transfer path, and
   Q4/KV/activation budgets.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Treat BF16/FP32 model data as software reference and validation input only.
  The first PL DDR4 weight layout and GEMV datapaths must assume custom Q4
  weight-only storage for large weights.
- Keep PS-side code as orchestration/support only. Do not move model math for
  prefill or decode back into PS unless the project direction explicitly
  changes.
- Treat `Vitis_Workspace/` as a local regenerated workspace. Keep durable
  application source under `FPGA_Project/software/` and recreate Vitis
  platform/application/build outputs from the Vivado hardware export.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- Keep `CURRENT_STATE.md` concise. Store detailed validation logic in scripts
  and summarize only durable results here.
