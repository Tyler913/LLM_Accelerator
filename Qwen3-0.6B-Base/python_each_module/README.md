# Module-by-module Python validation

This directory contains focused Python validation scripts for the modules that
will become FPGA kernels or FPGA-visible datapath blocks.

Run scripts from the repository root with:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/<script>.py
```

The scripts use `Qwen3-0.6B-Base/` as the local model directory and compare
manual FP32 math against Hugging Face eager execution.

Script map:

- `01_validate_embedding.py`: token id to embedding lookup
- `02_validate_rmsnorm.py`: RMSNorm, including layer input norm and final norm
- `03_validate_qkv_gemv.py`: q_proj/k_proj/v_proj GEMV
- `04_validate_qk_norm_rope.py`: q_norm/k_norm and RoPE K cache result
- `05_validate_kv_cache.py`: token-by-token K/V cache append behavior
- `06_validate_attention.py`: causal attention score, softmax, weighted V
- `07_validate_mlp.py`: gate/up/silu/down MLP block
- `08_validate_decoder_layer.py`: complete Layer 0 cached decoder layer
- `09_validate_final_norm_lm_head_argmax.py`: final norm, tied LM head, argmax
- `10_validate_full_run_one_token.py`: complete 28-layer cached run_one_token
- `11_export_fpga_test_vectors.py`: exports compact FP32 golden vectors for
  first FPGA/RTL bring-up
- `12_verify_fpga_test_vectors.py`: verifies exported RMSNorm and Q/K/V GEMV
  vectors without reloading the model
- `13_export_q4_gemv_vectors.py`: exports Verilog-facing custom Q4 Layer 0
  Q/K/V GEMV vectors from the FP32 vector artifact
- `14_verify_q4_gemv_vectors.py`: verifies Q4 packing, fixed-point scales,
  dot64 smoke data, and reconstructed Q/K/V GEMV outputs
- `15_export_q4_projection_vectors.py`: exports real q_proj rows 0..15 hex
  vectors for the projection-level RTL testbench
- `16_profile_rmsnorm_ranges.py`: profiles RMSNorm input/output/gamma/inv_rms
  ranges across all model RMSNorm modules for the reference prompt path
- `17_export_rmsnorm_fixed_vectors.py`: exports real Layer 0 RMSNorm
  fixed-point hex vectors for the `rmsnorm_1024` RTL testbench
- `18_export_rope_fixed_vectors.py`: exports real Layer 0 last-token Q/K,
  cos/sin, and fixed-point RoPE expected hex vectors for the
  `rope_qk_layer_128` RTL testbench
- `19_export_qmap_dot64_image.py`: exports the first QMAP v1 binary PL DDR4
  staging image and optional C header from the Q4 dot64 vector
- `20_export_qmap_row1024_image.py`: exports the QMAP v1 row1024 Layer 0
  `q_proj` row 0 image used by the row GEMV RTL and board smoke path
- `21_export_qmap_qkv_projection_image.py`: exports the first Layer 0 QKV
  projection QMAP work-packet image, plus Python Q12.12 expected words for
  RTL write-back comparison
- `22_export_qk_norm_rope_fixed_vectors.py`: exports fixed q/k norm and RoPE
  vectors from the Q4/QMAP Q/K projection contract
- `23_export_kv_cache_append_vectors.py`: exports fixed K/V cache append
  address/data vectors after q/k norm and RoPE
- `24_export_attention_score_vectors.py`: exports current-token Q, cached K,
  and exact raw/scaled attention score vectors
- `25_export_attention_softmax_value_vectors.py`: exports fixed softmax
  probabilities, V-cache reads, and attention-output vectors
- `26_export_o_proj_vectors.py`: exports Layer 0 attention output projection
  Q4/fixed vectors for `attn_out[2048] -> o_proj_out[1024]`
- `27_export_post_attention_residual_norm_vectors.py`: exports fixed
  post-attention residual add and post-attention RMSNorm vectors
- `28_export_mlp_gate_up_vectors.py`: exports Layer 0 MLP gate/up projection
  Q4/fixed vectors from the passing post-attention RMSNorm output
- `29_export_mlp_silu_mul_vectors.py`: exports fixed SiLU/multiply vectors for
  `gate[3072], up[3072] -> mlp_hidden[3072]`
- `30_export_mlp_down_vectors.py`: exports Layer 0 MLP down-projection
  Q4/fixed vectors for `mlp_hidden[3072] -> down_out[1024]`
- `31_export_mlp_residual_add_vectors.py`: exports final Layer 0 MLP residual
  vectors for `post_attn_hidden[1024] + down_out[1024] -> layer_out[1024]`
- `32_export_final_rmsnorm_vectors.py`: exports full-model current-token final
  RMSNorm vectors for `final_hidden[1024] -> final_norm[1024]`
- `33_export_lm_head_argmax_vectors.py`: exports tiled Q4 LM-head scan,
  memory-layout hex, and greedy argmax vectors from `final_norm[1024]`
- `34_export_qmap_lm_head_argmax_image.py`: wraps the LM-head activation,
  persistent weight/scale descriptors, scan range, and output token/score into
  a QMAP runtime packet for RTL simulation
- `35_export_lm_head_full_vocab_vectors.py`: streams full-vocabulary LM-head
  Q4 weight/scale/logit vectors for the descriptor-backed full-vocab RTL scan
- `36_export_qmap_final_token_tail_image.py`: wraps final hidden, final
  RMSNorm gamma, final-norm scratch, persistent LM-head weight/scale
  descriptors, and output token/score into the first QMAP final-token tail
  runtime packet
- `37_export_qmap_attention_frontend_image.py`: wraps Q/K/V projection outputs,
  q/k norm gamma, RoPE cos/sin tables, KV-cache placement, and Q RoPE scratch
  into the first QMAP attention front-end runtime packet
- `38_export_qmap_attention_score_value_image.py`: wraps Q RoPE input, K/V
  cache placement, softmax exp LUT, and `attn_out[2048]` scratch into the next
  QMAP attention-body runtime packet
- `39_export_qmap_o_proj_image.py`: wraps `attn_out[2048]`, persistent Layer 0
  `o_proj` Q4 weight/scale descriptors, and `o_proj_out[1024]` scratch into a
  QMAP runtime packet
- `40_export_qmap_post_attention_residual_norm_image.py`: wraps residual input,
  `o_proj_out[1024]`, signed post-attention RMSNorm gamma, post-attention
  hidden scratch, and post-norm scratch into the next QMAP per-layer body packet
- `41_export_qmap_mlp_gate_up_image.py`: wraps `post_norm[1024]`, persistent
  Layer 0 gate/up Q4 weight/scale descriptors, and gate/up output scratch into
  the next QMAP per-layer body packet
- `42_export_qmap_mlp_silu_mul_image.py`: wraps gate/up `[3072]`, the fixed
  sigmoid LUT, hidden output scratch, and expected hidden debug data into the
  next QMAP per-layer body packet
- `43_export_qmap_mlp_down_image.py`: wraps `mlp_hidden[3072]`, persistent
  Layer 0 down-proj Q4 weight/scale descriptors, down output scratch, and
  expected down debug data into the next QMAP per-layer body packet
- `44_export_qmap_mlp_residual_add_image.py`: wraps
  `post_attn_hidden[1024]`, `down_out[1024]`, layer-output scratch, and
  expected layer-output debug data into the next QMAP per-layer body packet
- `run_all_module_validations.py`: runs the core validation scripts `01`-`10`
  in order
