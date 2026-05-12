# Project Context

This repository is for building an FPGA-based LLM accelerator project inspired
by the Hummingbird+ paper.

For the latest working state, validation progress, and next steps, also read
`CURRENT_STATE.md`.

## Long-Term Goal

The project goal is to study and implement an LLM accelerator that can run on
FPGA hardware, following the direction of the paper:

- `paper/3748173.3779189.pdf`
- Title: `Hummingbird+: Advancing FPGA-based LLM Deployment from Research Prototype to Edge Product`
- Main idea: move FPGA LLM acceleration from research prototypes toward a
  practical edge product, with attention to low-cost FPGA devices, memory
  bandwidth, quantized LLM inference, decode/prefill performance, and hardware
  resource efficiency.

The project should gradually turn this paper and the local model artifacts into
an implementable accelerator workflow, including model inspection, weight
formatting, quantization-aware data layout, FPGA-friendly kernels, simulation,
verification, and eventually hardware-oriented implementation.

## Current Model Baseline

The local baseline model is:

- Hugging Face repo: `Qwen/Qwen3-0.6B-Base`
- Local directory: `Qwen3-0.6B-Base/`
- Important files:
  - `config.json`
  - `generation_config.json`
  - `model.safetensors`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
  - `README.md`
  - `LICENSE`

This small Qwen3 0.6B model is the starting point for understanding model
structure, tokenizer behavior, tensor shapes, weight storage, and later
hardware mapping experiments.

The current project target is text continuation rather than chat alignment.
`Qwen3-0.6B-Base` is preferred because its Hugging Face safetensors format is
straightforward to inspect, manually reimplement in Python, quantize into a
custom hardware format, and eventually map to FPGA memory.

### Qwen3-0.6B-Base Structure Notes

Initial PC-side inspection results:

- Architecture: `Qwen3ForCausalLM`
- Model type: `qwen3`
- Model form: dense decoder-only transformer, not MoE
- Weight file: `Qwen3-0.6B-Base/model.safetensors`
- Stored weight dtype: `bfloat16`
- Number of tensors: 310
- Total parameters: 596,049,920
- Vocabulary size: 151,936
- Hidden size: 1,024
- Intermediate/MLP size: 3,072
- Layers: 28
- Attention heads: 16
- KV heads: 8
- Head dim: 128
- Max position embeddings: 32,768
- RoPE theta: 1,000,000, stored under `rope_parameters.rope_theta`
- RMSNorm epsilon: `1e-6`
- Activation: `silu`
- `tie_word_embeddings`: true
- BOS/EOS token id: 151,643

Parameter grouping from `model.safetensors`:

- Embedding: 155,582,464 params
- Attention blocks: 176,167,936 params
- MLP blocks: 264,241,152 params
- Norms: 58,368 params
- Total: 596,049,920 params

Rough memory estimates from `pc_testing/04_inspect_weights.py`:

- BF16/FP16 weights: about 1,136.88 MiB
- INT8 weights: about 568.44 MiB
- INT4 weights: about 284.22 MiB
- INT4 with FP16 scales, group size 64: about 301.98 MiB

Single-token layer shape summary:

- Embedding table: `[151936, 1024]`
- Layer input/output hidden vector: `[1024]`
- Q projection: `[1024] -> [2048]`, reshaped to `16 x 128`
- K projection: `[1024] -> [1024]`, reshaped to `8 x 128`
- V projection: `[1024] -> [1024]`, reshaped to `8 x 128`
- GQA ratio: `16 / 8 = 2`, so two Q heads share one K/V head
- Attention concat before output projection: `[2048]`
- Output projection: `[2048] -> [1024]`
- MLP gate/up projections: `[1024] -> [3072]`
- MLP down projection: `[3072] -> [1024]`
- Final tied LM head uses the embedding table as `[151936, 1024]`

KV cache shape observed from local serial decode:

- Each layer stores K and V independently.
- Per layer and processed token:
  - K: `[8, 128]`
  - V: `[8, 128]`
- Transformers exposes this as `[batch, num_key_value_heads, seq_len,
  head_dim]`, observed as `[1, 8, T, 128]`.
- Per token per layer: `2 * 8 * 128 = 2048` values.
- Per token across all 28 layers: `28 * 2048 = 57344` values.
- Approximate KV cache memory:
  - context 128: FP16/BF16 about 14 MiB, INT8 about 7 MiB
  - context 256: FP16/BF16 about 28 MiB, INT8 about 14 MiB
  - context 512: FP16/BF16 about 56 MiB, INT8 about 28 MiB
  - context 1024: FP16/BF16 about 112 MiB, INT8 about 56 MiB

## PC Testing Scripts

The directory `Qwen3-0.6B-Base/pc_testing/` contains initial PC-side validation
scripts:

- `01_test_generate.py`: loads tokenizer/model locally and runs standard
  `model.generate` on the prompt `The future of FPGA is`.
- `02_serial_decode.py`: manually performs token-by-token prefill and decode
  with `past_key_values`; useful as a software reference for future hardware
  decode verification.
- `03_inspect_config.py`: prints important model config fields and the full
  loaded Transformers config.
- `04_inspect_weights.py`: enumerates safetensors weights, shapes, dtypes,
  total params, and rough quantized memory estimates.
- `05_inspect_kv_cache.py`: serially feeds prompt tokens, prints K/V cache
  shape growth for layers 0, 1, and 27, and estimates KV cache memory.
- `06_trace_layer0_tensors.py`: registers hooks on embedding, layer 0
  attention/MLP modules, final norm, and LM head to show actual tensor shape
  flow for token position 0 and token position 1 with cache.
- `07_manual_layer0_qkv.py`: manually reproduces layer 0 embedding lookup,
  input RMSNorm, and `q_proj`/`k_proj`/`v_proj` GEMV using real model weights,
  prints FPGA-oriented formulas, and compares the manual tensors against
  Hugging Face hook outputs.

Observed validation result:

- `01_test_generate.py` and `02_serial_decode.py` both run successfully on CPU
  in the `llm_fpga` environment.
- For greedy decoding with the same prompt, both paths produce the same visible
  output prefix, confirming that the manual KV-cache decode path is aligned
  with standard generation at this stage.
- `05_inspect_kv_cache.py` confirms that all 28 layers maintain K/V caches
  with shape `[1, 8, T, 128]`, where `T` grows by one after each processed
  token.
- `06_trace_layer0_tensors.py` confirms the actual layer 0 flow:
  embedding `[1, 1, 1024]`, q_proj `[1, 1, 2048]`, k/v_proj `[1, 1, 1024]`,
  q_norm `[1, 1, 16, 128]`, k_norm `[1, 1, 8, 128]`, o_proj `[1, 1, 1024]`,
  MLP gate/up `[1, 1, 3072]`, down `[1, 1, 1024]`, and LM head
  `[1, 1, 151936]`.
- `07_manual_layer0_qkv.py` confirms that manual FP32 embedding lookup,
  RMSNorm, and layer 0 Q/K/V projections match Hugging Face hook outputs with
  zero observed max absolute error for the first prompt token.
- Some older scripts may still use `torch_dtype=...`; Transformers reports
  this as deprecated and recommends `dtype=...`.

## Current Implementation Direction

The first FPGA milestone prioritizes a complete, understandable inference
chain over throughput. The current intended flow is:

- PS side:
  - tokenizer and detokenizer
  - simple control of PL execution
  - prompt token dispatch and generated token collection
- PL side:
  - one `run_one_token(input_token, position)` path
  - embedding lookup
  - all 28 decoder layers
  - KV cache read/write
  - final RMSNorm
  - tied LM head
  - greedy argmax

For the first version, prompt prefill should be serial: send prompt tokens to
the same `run_one_token` path one at a time. This avoids implementing a
separate parallel prefill engine and keeps the hardware target close to the
validated `02_serial_decode.py` behavior.

Planned quantization direction:

- Do not start with GGUF, GPTQ, AWQ, FP8, or Q4_K hardware support.
- Define a simple custom Q4 format later:
  - signed int4 weights
  - group-wise symmetric quantization
  - group size 64 initially
  - one scale per group, likely FP16 at first
  - `real_weight = scale[group] * int4_weight`
- Keep Python reference and FPGA implementation aligned to the same custom
  format.

Recommended initial constraints:

- Greedy decoding only, no temperature/top-k/top-p.
- Context length 128 or 256 for first hardware bring-up.
- Optimize only after the full prompt-to-text chain runs.

Immediate next technical step:

- Extend the manual layer 0 reference past Q/K/V projection by reproducing
  `q_norm` and `k_norm`, then RoPE for token positions 0 and 1, and compare
  those tensors against Hugging Face hook outputs or an equivalent traced
  reference.

## Paper Notes

Initial extracted facts from the paper:

- The paper targets practical FPGA-based LLM edge deployment.
- It emphasizes that LLM inference is often memory-bound, making memory
  bandwidth and capacity central design constraints.
- It focuses on MoE LLM deployment, using Qwen3-30B-A3B as a representative
  larger model in the paper.
- Reported paper system target:
  - FPGA family: Zynq UltraScale XCZU2CG/3EG SoC
  - Memory: 24 GB total
  - Estimated BOM: under 150 USD in mass production
  - Quantization: GPTQ INT4 for Qwen3-30B-A3B
  - Reported speed: over 18 token/s decode and over 50 token/s prefill
- Relevant accelerator ideas include GEMV optimization, scalar engine
  optimization, quantized attention/KV cache handling, and memory-aware
  scheduling.

## Python Environment Rule

For this repository, every Python command must run inside the conda environment:

```bash
conda run -n llm_fpga python ...
```

or, for interactive shell work:

```bash
conda activate llm_fpga
```

Do not use the system Python, base conda environment, or another virtual
environment for this project unless the user explicitly changes the rule.

Currently installed project-related Python packages include:

- `huggingface_hub`
- `pypdf`

## Working Preferences

- Preserve the downloaded model files under `Qwen3-0.6B-Base/`.
- The `.cache` folder generated inside the model directory by Hugging Face is
  download metadata and can be regenerated; it is not required for model
  loading.
- Do not commit large model weights or generated FPGA weight artifacts to the
  main Git repository. Use `init/CROSS_PLATFORM_SYNC.md` and the scripts under
  `init/` to restore large assets on each machine.
- Keep project notes in this repository so future chats can recover context by
  reading files instead of depending on chat memory.
- After meaningful progress, update `CURRENT_STATE.md` in the same turn when
  feasible. Promote only durable decisions or stable technical facts into this
  file.
- Keep persisted project files in English, including code comments, docstrings,
  script output labels, and Markdown documentation, unless the user explicitly
  asks for another language.
- When adding scripts, prefer clear, reproducible commands and document how to
  run them with `llm_fpga`.
