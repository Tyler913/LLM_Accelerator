# Current State

Last updated: 2026-05-26

This file is the short working-state handoff. For stable project context, read
`PROJECT_CONTEXT.md` first. For workflow rules, read `AGENTS.md`.

## Current Goal

Build a first FPGA demo that can take a prompt and generate text with a small
Qwen3 dense decoder model.

Current target:

- Model: `Qwen/Qwen3-0.6B-Base`
- Local model directory: `Qwen3-0.6B-Base/`
- Inference style: text continuation
- Decode policy: greedy argmax
- Prefill plan: serial prefill using the same one-token path as decode
- First context length target: 128 or 256 tokens
- First quantization direction: simple custom Q4 weight-only format later

## Cloud Sync Status

Cross-platform sync is split between GitHub and Hugging Face:

- GitHub remote: `https://github.com/Tyler913/LLM_Accelerator.git`
- Hugging Face model mirror:
  `Tyler01/qwen3-0p6b-base-llm-accelerator`
- Hugging Face model revision:
  `d297782df3b18206f4b1caea202cf6272bae3aa9`
- Mirror contents: model/tokenizer assets only, including
  `model.safetensors`; project validation scripts remain in GitHub.

Fresh machines should clone GitHub, create the `llm_fpga` conda environment,
log in to Hugging Face if needed, run `init/download_model_assets.py`, then run
`init/verify_assets.py`.

The conceptual hardware function remains:

```text
next_token = run_one_token(input_token, position)
```

During prompt prefill, PS calls this once per prompt token. During decode, PS
calls it once per generated token.

## Python Validation Status

The FP32 Python reference is complete for the current first-version target.
Validated coverage includes serial prefill/decode, full 28-layer cached
single-token decode, and module-by-module checks for the FPGA bring-up path.

Detailed validation logic lives in the Python files instead of this document:

- Full-model and exploratory references:
  `Qwen3-0.6B-Base/pc_testing/`
- Per-module FPGA bring-up references:
  `Qwen3-0.6B-Base/python_each_module/`
- Per-module validation index:
  `Qwen3-0.6B-Base/python_each_module/README.md`

Useful validation commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/pc_testing/10_manual_full_model_cached_decode.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/run_all_module_validations.py
```

Stable reference prompt:

```text
The future of FPGA is
```

Greedy continuation checkpoints:

- Prompt next token: token id `264`, text `' a'`
- After feeding `' a'` back into cached decode: token id `26291`, text
  `' fascinating'`

## FPGA Test Vector Status

Initial FP32 FPGA/HLS bring-up vectors are exported and verified for the first
RMSNorm and Q/K/V GEMV kernels.

Scripts:

- Export:
  `Qwen3-0.6B-Base/python_each_module/11_export_fpga_test_vectors.py`
- Verify:
  `Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py`

Generated local artifact directory, ignored by Git through `artifacts/`:

```text
artifacts/test_vectors/qwen3_0p6b_fp32_v0/
```

Current vector set:

- `manifest.json`
- `rmsnorm_layer0_last_token.npz`: input/weight/expected tensors for
  Layer 0 RMSNorm
- `qkv_layer0_last_token.npz`: normalized input, Q/K/V projection weights,
  and expected Q/K/V outputs

Export/verify commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/11_export_fpga_test_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/12_verify_fpga_test_vectors.py
```

Verified prompt/token selection: prompt `The future of FPGA is`, selected
position `4`, selected token id `374` (`' is'`).

Current max abs errors: RMSNorm `2.3841857910e-07`, `q_proj`
`1.9073486328e-06`, `k_proj` `1.6689300537e-06`, `v_proj`
`2.6822090149e-07`.

## FPGA Memory Map Status

Memory-map document: `FPGA_MEMORY_MAP.md`.

Current confirmed coverage:

- Confirmed PS DDR low range: `0x0000_0000` through `0x7FFF_FFFF` (2 GiB)
- Confirmed AXI BRAM smoke-test range:
  `0x8000_0000` through `0x8000_1FFF` (8 KiB)
- Confirmed smoke-test path:
  `M_AXI_HPM0_LPD` -> AXI SmartConnect -> AXI BRAM Controller -> Block Memory
  Generator
- Draft relative PL DDR4 layout for a nominal 512 MiB aperture, including
  Q4 weights/scales, KV cache, activation buffers, RoPE table,
  logits/argmax scratch, test-vector staging, and reserved space
- PL DDR4 controller base/range remains TODO because PL DDR4 is not yet
  instantiated in the current Vivado design

Durable hardware target facts are in `PROJECT_CONTEXT.md`; detailed address
planning lives only in `FPGA_MEMORY_MAP.md`.

## Vivado Project Status

Vivado project location:

```text
FPGA_Project/Vivado_Project/LLM_FPGA.xpr
```

Current project facts:

- Vivado 2025.1.1 project, part `xczu2eg-sfvc784-2-i`, board part unset.
- Zynq UltraScale+ MPSoC is configured with PS DDR, QSPI, UART0, SD1, PL0
  clock, fabric reset, `M_AXI_HPM0_LPD`, and `M_AXI_HPM0_FPD`.
- AXI BRAM smoke-test fabric is connected on `M_AXI_HPM0_LPD`:
  `M_AXI_HPM0_LPD` -> AXI SmartConnect -> AXI BRAM Controller -> Block Memory
  Generator.
- Address Editor assignment: base `0x8000_0000`, range `8K`, high
  `0x8000_1FFF`.
- `M_AXI_HPM0_FPD` remains enabled but intentionally unconnected; it is the
  only known incomplete address path.
- Block design validation, HDL wrapper generation, synthesis, implementation,
  bitstream generation, and hardware export all completed successfully with
  0 errors and 0 critical warnings.
- Exported XSA with bitstream:
  `FPGA_Project/Vivado_Project/llm_system_axi_bram_smoke.xsa`
- The XSA archive contains `llm_system_axi_bram_smoke.bit`, `llm_system.hwh`,
  `psu_init.*`, and `sysdef.xml`.
- Vivado Git hygiene was reviewed for the upcoming commit: keep source-bearing
  `.xpr`, `.bd`, `.xci`, HDL, XDC, and intentional input files visible; ignore
  regenerable Vivado outputs.
- No tracked `.v`, `.sv`, `.xci`, `.bd`, or `.xdc` sources have been committed
  yet.

Generated Vivado directories removed locally: `LLM_FPGA.cache/`,
`LLM_FPGA.hw/`, `LLM_FPGA.ip_user_files/`, and `LLM_FPGA.sim/`. `.gitignore`
now covers common Vivado generated directories and artifacts.

Stable model shapes, layer flow, and KV-cache facts are intentionally kept in
`PROJECT_CONTEXT.md`, not repeated here.

## Immediate Next Step

Continue FPGA preparation:

1. Use the exported hardware to run a minimal PS-side memory write/read test
   against BRAM base address `0x8000_0000`.
2. Fill in and review `FPGA_MEMORY_MAP.md`, especially PL DDR4 base/range,
   usable capacity, PS-to-PL transfer path, and Q4/KV/activation budgets.
3. Start small FPGA/HLS kernel bring-up against the exported vectors, beginning
   with RMSNorm and GEMV before assembling a complete decoder layer.
4. Extend the vector exporter as each new hardware block is added, especially
   RoPE, KV cache append/read, attention, MLP, and complete Layer 0 cached
   decode.

## Practical Notes

- Always use `conda run -n llm_fpga ...` for Python commands.
- Keep large model weights and generated artifacts out of Git history.
- Do not use broad staging commands such as `git add .` unless explicitly
  requested.
- `CURRENT_STATE.md` should stay concise. Do not add long validation logs here;
  put reusable validation code in Python files and summarize only durable
  results.
