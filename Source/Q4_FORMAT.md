# Q4 Format And Bring-Up Vectors

Status: Q4 v0 is defined for the first Verilog-facing Layer 0 Q/K/V GEMV
bring-up. It is not yet a full-model quantization artifact.

For overall project state, read `Source/PROJECT_CONTEXT.md` and
`Source/CURRENT_STATE.md`. For physical memory placement, read
`Source/FPGA_MEMORY_MAP.md`. For descriptor-based PL DDR4 tensor staging, read
`Source/QMAP_FORMAT.md`.

## Scope

The current Q4 artifact quantizes only these Layer 0 matrices:

- `model.layers.0.self_attn.q_proj.weight`, shape `[2048, 1024]`
- `model.layers.0.self_attn.k_proj.weight`, shape `[1024, 1024]`
- `model.layers.0.self_attn.v_proj.weight`, shape `[1024, 1024]`

Source FP32 vector:

```text
artifacts/test_vectors/qwen3_0p6b_fp32_v0/qkv_layer0_last_token.npz
```

Generated Q4 vectors:

```text
artifacts/test_vectors/qwen3_0p6b_q4_v0/manifest.json
artifacts/test_vectors/qwen3_0p6b_q4_v0/qkv_layer0_last_token_q4.npz
artifacts/test_vectors/qwen3_0p6b_q4_v0/q_proj_row0_group0_dot64.npz
```

These artifact files are generated and ignored by Git. The tracked scripts are:

```text
Qwen3-0.6B-Base/python_each_module/13_export_q4_gemv_vectors.py
Qwen3-0.6B-Base/python_each_module/14_verify_q4_gemv_vectors.py
```

## DDR Staging Contract

This document defines Q4 quantization, packing, and fixed-point math semantics.
It does not define the full PL DDR4 image layout by itself.

The first descriptor-based PL DDR4 staging contract is
`Source/QMAP_FORMAT.md`.
QMAP v1 describes tensors with a fixed header, a tensor descriptor table, and
payload data. The first QMAP image will stage the real Layer 0
`q_proj` row 0 group 0 dot64 vector from
`q_proj_row0_group0_dot64.npz`.

Keep this split clear:

- `Source/Q4_FORMAT.md`: how Q4 weights, scales, activations, and expected
  integer math are interpreted.
- `Source/QMAP_FORMAT.md`: where those tensors live in PL DDR4 and how software
  or PL readers discover them.

## Why This Scope

Layer 0 Q/K/V is the first large matrix path immediately after RMSNorm, so it
exercises the Q4 format that Verilog needs next without committing the whole
model to an unproven layout. It covers both output shapes used by attention:
`q_proj` produces 2048 values, while `k_proj` and `v_proj` produce 1024 values.

This scope is large enough to validate packing, per-group scales, accumulator
widths, and output error, but small enough to regenerate quickly while RTL is
still changing.

## Q4 Weight Format V0

The first deployable PL DDR4 weight path must use project custom Q4
weight-only quantization for large model weights. BF16/FP32 weights are
reference data only.

Format:

- Quantized value: signed int4, two's-complement range `[-8, 7]`
- Zero point: none
- Group size: 64 weights
- Groups per 1024-wide row: 16
- Grouping: per output row, contiguous input columns
- Layout: row-major by output row
- Scale storage: unsigned 16-bit fixed-point `Q2.14`
- Scale value: `scale = scale_q2_14 / 2^14`

Scale generation:

```text
raw_scale = max(abs(weight_group)) / 7
scale_q2_14 = round(raw_scale * 2^14)
if absmax > 0 and scale_q2_14 == 0: scale_q2_14 = 1
```

Quantization:

```text
q = round(weight / (scale_q2_14 / 2^14))
q = clamp(q, -8, 7)
```

The exporter uses NumPy `rint` round-to-nearest-even. Verilog does not need to
repeat this quantization step for the first bring-up; it consumes already
packed Q4 weights and fixed-point activations from the artifact.

## Packing

Two signed int4 values are packed into one byte.

For input column `i`:

- even `i`: stored in low nibble `[3:0]`
- odd `i`: stored in high nibble `[7:4]`
- negative values use 4-bit two's-complement encoding

Examples:

```text
 0 -> 0x0
 7 -> 0x7
-1 -> 0xF
-8 -> 0x8
```

For a `[out_features, 1024]` matrix, packed weight shape is:

```text
[out_features, 512] uint8
```

Scale shape is:

```text
[out_features, 16] uint16
```

## Activation Format For This Vector

The current vector stores `input_norm` as signed int16 fixed-point `Q4.12`:

```text
activation_q4_12 = round(input_norm_fp32 * 2^12)
activation_float = activation_q4_12 / 2^12
```

This is a Verilog bring-up activation format, not Q4 activation
quantization. The project Q4 requirement applies to large weights. Activation
width may be revisited after more layers and residual paths are calibrated.

## Integer GEMV Contract

For each output row and group:

```text
partial[row, group] =
  sum_j activation_q4_12[group*64 + j] * weight_q4[row, group*64 + j]

scaled[row, group] =
  partial[row, group] * scale_q2_14[row, group]

output[row] =
  sum_group(scaled[row, group]) / 2^(12 + 14)
```

Recommended safe internal widths for the first RTL:

| Signal | Format | Suggested Width |
| --- | --- | ---: |
| activation | signed `Q4.12` | 16 |
| Q4 weight | signed int4 | 4 |
| one product | signed integer | 20 |
| 64-lane partial sum | signed integer | 26 |
| scale | unsigned `Q2.14` | 16 |
| scaled group sum | signed integer, Q26 scale denominator | 42 |
| 16-group row accumulator | signed integer, Q26 scale denominator | 48 |

The first small Verilog target can use:

```text
q_proj_row0_group0_dot64.npz
```

That file contains one 64-value activation slice, one 64-weight Q4 packed
slice, one `Q2.14` scale, the integer partial sum, the scaled integer result,
and the expected floating result.

## Validation Results

Commands:

```bash
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/13_export_q4_gemv_vectors.py
conda run -n llm_fpga python Qwen3-0.6B-Base/python_each_module/14_verify_q4_gemv_vectors.py
```

Latest results against FP32 expected Q/K/V outputs:

| Matrix | Max Abs Error | Mean Abs Error | RMSE |
| --- | ---: | ---: | ---: |
| `q_proj` | 0.22418833 | 0.01856173 | 0.02495133 |
| `k_proj` | 0.12317657 | 0.01752916 | 0.02286997 |
| `v_proj` | 0.07140587 | 0.01565307 | 0.01975244 |

The verifier also confirms exact reconstruction from packed bytes and scales,
including the `q_proj` row 0 group 0 dot64 smoke vector.

Board-level Q4/QMAP bring-up has now passed both the dot64 smoke path and the
row1024 full-row smoke path. The row1024 board run reported PL status `0xA`
and `row_sum_q26_low32=0xFFCA_DDC7`, matching the expected `-3482169` row
result.

## Known Tradeoffs

- This is not a language-quality measurement. It validates the first hardware
  Q4 GEMV contract against one prompt/token vector.
- `Q2.14` scales are chosen because they are easier for hand-written RTL than
  FP16 scales and have enough range for this Q/K/V artifact. Full-model export
  must still check scale range across all matrices.
- `Q4.12` activations fit this Layer 0 normalized input vector. Later residual
  and MLP paths may need wider activation formats or per-buffer scaling.
- Row-major contiguous groups are simple and transparent for bring-up. Tiled
  storage may be introduced later for bandwidth, but it should preserve the
  same signed int4 and scale semantics unless a deliberate format revision is
  made.
