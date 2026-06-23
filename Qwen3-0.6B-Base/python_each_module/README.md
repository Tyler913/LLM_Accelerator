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
- `run_all_module_validations.py`: runs the core validation scripts `01`-`10`
  in order
