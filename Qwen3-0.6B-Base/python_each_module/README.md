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
- `run_all_module_validations.py`: runs every script above in order
