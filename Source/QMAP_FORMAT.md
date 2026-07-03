# QMAP Format

Status: draft v1 descriptor format for PL DDR4 tensor staging and future
full-model artifact layout. The dot64 image and the full row1024 image have
both passed PS load/readback and PL AXI-master read/compute hardware smoke
tests. The first Layer 0 QKV projection work packet, local PL write-back path,
AXI write adapter, downstream local attention through `o_proj`,
post-attention residual/RMSNorm hookup, MLP gate/up projection, MLP
SiLU/multiply, MLP down projection, and final MLP residual add now pass Icarus
simulation. The full-model current-token final RMSNorm stage also passes
Icarus simulation, and the first tiled Q4 LM-head scan plus greedy argmax stage
passes Icarus simulation for a 1024-row scan window. A memory-backed
LM-head tile reader/wrapper, runtime tile scheduler, and QMAP descriptor-backed
LM-head wrapper now also pass local multi-window scans for `64` and `23`
tiles plus the full `9496`-tile vocabulary sweep. The first QMAP final-token
tail wrapper now composes final RMSNorm write-back with full-vocabulary
LM-head argmax in local RTL/xsim. The first QMAP attention front-end wrapper
now also passes local RTL simulation, reading Q/K/V projection outputs plus
q/k gamma and RoPE tables through descriptors and writing exact K/V cache plus
Q RoPE outputs. The next QMAP attention score/value wrapper now also passes
local RTL simulation, reading Q RoPE plus K/V cache and writing exact
`attn_out[2048]`. The QMAP `o_proj` wrapper now consumes that attention
output, persistent Layer 0 Q4 `o_proj` weight/scale descriptors, and writes
exact `o_proj_out[1024]` locally. The QMAP post-attention residual/RMSNorm
wrapper now consumes residual input, that `o_proj_out[1024]`, and signed
post-attention gamma, then writes exact post-attention hidden and post-norm
buffers locally. The QMAP MLP gate/up wrapper now consumes descriptor-visible
`post_norm[1024]`, persistent Layer 0 Q4 gate/up weight/scale descriptors, and
writes exact gate/up `[3072]` buffers locally. The QMAP MLP SiLU/multiply
wrapper now consumes descriptor-visible gate/up `[3072]` plus a fixed UQ0.16
sigmoid LUT and writes exact `mlp_hidden[3072]` locally. The QMAP MLP down
wrapper now consumes descriptor-visible `mlp_hidden[3072]`, reads persistent
Layer 0 Q4 down-proj weight/scale rows, and writes exact `down_out[1024]`
locally. The QMAP final MLP residual wrapper now consumes descriptor-visible
`post_attn_hidden[1024]` and `down_out[1024]`, then writes exact
`layer_out[1024]` locally.

For Q4 quantization semantics, read `Source/Q4_FORMAT.md`. For physical PL
DDR4 placement, read `Source/FPGA_MEMORY_MAP.md`.

## Purpose

QMAP is the project's descriptor-based memory-map format for tensors stored in
PL DDR4. It is designed so the first tiny bring-up case and the later full
model use the same structure:

```text
QMAP header
tensor descriptor table
payload data
```

The first QMAP instances describe real Qwen3 Layer 0 `q_proj` data: first row
0 group 0 as a dot64 vector, then row 0 as a complete row1024 GEMV. The
long-term direction is not to keep making one-off test packets. QMAP should
become the stable descriptor contract between:

- persistent model artifacts stored once in PL DDR4
- runtime work packets that tell PL which tensors to read and which buffers to
  write for the current kernel or token step
- optional debug/golden tensors used only during bring-up

QMAP describes where data is and how to interpret it. It does not define the
math for Q4 itself; that contract stays in `Source/Q4_FORMAT.md`.

## Design Rules

- All multi-byte fields are little-endian.
- All addresses are byte addresses.
- PL DDR4 physical addresses must use 64-bit fields.
- Descriptor and payload addresses should be at least 4-byte aligned.
- Payloads used by future burst readers should be aligned to 64 bytes or
  better when practical.
- Descriptors should describe logical tensor shape separately from physical
  byte size. This matters for packed Q4 weights, where logical shape can be
  `[rows, cols]` while physical row width is `cols / 2` bytes.
- Debug/golden tensors are allowed in QMAP, but they must be marked with a
  debug flag so the future inference path can ignore them.

## Header

The QMAP header lives at the base of a QMAP image. The dot64 test-vector QMAP
image is placed at `0x4_1B10_0000`, the start of the draft PL DDR4 test-vector
staging region. The row1024 test-vector QMAP image is placed at
`0x4_1B20_0000`.

Header size: 256 bytes.

| Offset | Type | Field | Description |
| ---: | --- | --- | --- |
| `0x00` | `u32` | `magic` | ASCII `QMAP`, stored as `0x50414D51` |
| `0x04` | `u32` | `version` | Format version, current value `1` |
| `0x08` | `u32` | `header_bytes` | Current value `256` |
| `0x0C` | `u32` | `descriptor_bytes` | Current value `128` |
| `0x10` | `u32` | `descriptor_count` | Number of valid descriptors |
| `0x14` | `u32` | `descriptor_capacity` | Number of descriptor slots reserved in this image |
| `0x18` | `u64` | `descriptor_table_addr` | Physical address of descriptor table |
| `0x20` | `u64` | `payload_base_addr` | First payload byte address |
| `0x28` | `u64` | `image_base_addr` | Physical base address of the QMAP image |
| `0x30` | `u64` | `image_bytes` | Total bytes occupied by this QMAP image |
| `0x38` | `u32` | `flags` | Image-level flags, initially `0` |
| `0x3C` | `u32` | `checksum32` | Optional checksum over descriptors and payload; `0` if unused |
| `0x40` | `u32[48]` | `reserved` | Must be written as zero |

Bring-up images should reserve eight descriptor slots even if only a few are
valid. This keeps the payload base stable and leaves room for adding an output
or debug descriptor without changing the image shape.

Formal runtime work packets should reserve more descriptor slots from the
start. The first Layer 0 QKV projection packet should use at least 32
descriptor slots, with the payload base moved after the larger descriptor
table. A full persistent model manifest may use hundreds of descriptors.

Recommended first image base:

```text
qmap_base              = 0x4_1B10_0000
descriptor_table_addr  = qmap_base + 0x0100
descriptor_capacity    = 8
payload_base_addr      = qmap_base + 0x0500
```

Recommended second image base:

```text
qmap_base              = 0x4_1B20_0000
descriptor_table_addr  = qmap_base + 0x0100
descriptor_capacity    = 8
payload_base_addr      = qmap_base + 0x0500
```

## Tensor Descriptor

Each tensor descriptor is 128 bytes. Descriptors are fixed-size so PS software
and simple PL readers can index them directly:

```text
descriptor_addr = descriptor_table_addr + tensor_slot * 128
```

| Offset | Type | Field | Description |
| ---: | --- | --- | --- |
| `0x00` | `u32` | `tensor_id` | Stable id inside this QMAP image |
| `0x04` | `u32` | `role` | Tensor role enum |
| `0x08` | `u32` | `dtype` | Data type enum |
| `0x0C` | `u32` | `rank` | Number of valid dimensions in `dims` |
| `0x10` | `u32` | `flags` | Tensor flags |
| `0x14` | `u32` | `element_bits` | Logical element width in bits; use `4` for Q4 logical values |
| `0x18` | `u32` | `group_size` | Q4 group size or `0` if not grouped |
| `0x1C` | `u32` | `scale_tensor_id` | Matching scale tensor id, or `0xFFFF_FFFF` if none |
| `0x20` | `u64` | `base_addr` | Physical byte address of payload |
| `0x28` | `u64` | `nbytes` | Physical payload byte count |
| `0x30` | `u32[4]` | `dims` | Logical tensor dimensions |
| `0x40` | `u64[4]` | `strides_bytes` | Physical byte strides for each logical dimension |
| `0x60` | `u32` | `aux0` | Role-specific metadata, initially matrix id when useful |
| `0x64` | `u32` | `aux1` | Role-specific metadata, initially layer index when useful |
| `0x68` | `u32` | `aux2` | Role-specific metadata, initially row start when useful |
| `0x6C` | `u32` | `aux3` | Role-specific metadata, initially group/column start when useful |
| `0x70` | `u32` | `checksum32` | Optional tensor payload checksum; `0` if unused |
| `0x74` | `u32[3]` | `reserved` | Must be written as zero |

## Role Enum

| Value | Name | Meaning |
| ---: | --- | --- |
| `0` | `QMAP_ROLE_UNUSED` | Unused descriptor slot |
| `1` | `QMAP_ROLE_ACTIVATION` | Activation/input vector |
| `2` | `QMAP_ROLE_Q4_WEIGHT` | Packed Q4 weight tensor |
| `3` | `QMAP_ROLE_Q4_SCALE` | Per-group scale tensor |
| `4` | `QMAP_ROLE_OUTPUT` | Kernel output tensor |
| `5` | `QMAP_ROLE_EXPECTED` | Debug/golden expected tensor |
| `6` | `QMAP_ROLE_METADATA` | Extra metadata payload |
| `7` | `QMAP_ROLE_KV_CACHE` | KV cache tensor |
| `8` | `QMAP_ROLE_ROPE_TABLE` | RoPE cos/sin tensor |
| `9` | `QMAP_ROLE_PARAMETER` | Small read-only model parameter such as RMSNorm gamma or LUT |

## Dtype Enum

| Value | Name | Meaning |
| ---: | --- | --- |
| `0` | `QMAP_DTYPE_UNUSED` | Unused |
| `1` | `QMAP_DTYPE_U8` | Unsigned 8-bit integer |
| `2` | `QMAP_DTYPE_I8` | Signed 8-bit integer |
| `3` | `QMAP_DTYPE_U16` | Unsigned 16-bit integer |
| `4` | `QMAP_DTYPE_I16` | Signed 16-bit integer |
| `5` | `QMAP_DTYPE_U32` | Unsigned 32-bit integer |
| `6` | `QMAP_DTYPE_I32` | Signed 32-bit integer |
| `7` | `QMAP_DTYPE_U64` | Unsigned 64-bit integer |
| `8` | `QMAP_DTYPE_I64` | Signed 64-bit integer |
| `16` | `QMAP_DTYPE_PACKED_Q4_S4` | Two signed int4 values per byte |
| `17` | `QMAP_DTYPE_U16_Q2_14` | Unsigned 16-bit fixed-point `Q2.14` |
| `18` | `QMAP_DTYPE_I16_Q4_12` | Signed 16-bit fixed-point `Q4.12` |
| `19` | `QMAP_DTYPE_I32_Q12_12` | Signed `Q12.12` value padded to 32 bits |
| `20` | `QMAP_DTYPE_I32_Q14_10` | Signed `Q14.10` value padded to 32 bits |
| `21` | `QMAP_DTYPE_U16_Q8_8` | Unsigned `UQ8.8` value for RMSNorm gamma |
| `22` | `QMAP_DTYPE_I16_Q8_7` | Signed `Q8.7` value for q_norm/k_norm gamma |
| `23` | `QMAP_DTYPE_I16_Q1_15` | Signed `Q1.15` value for RoPE cos/sin |
| `24` | `QMAP_DTYPE_U32_Q0_20` | Unsigned `UQ0.20` value padded to 32 bits |
| `25` | `QMAP_DTYPE_U16_Q0_16` | Unsigned `UQ0.16` value padded to 32 bits |

## Tensor Flags

| Bit | Name | Meaning |
| ---: | --- | --- |
| `0` | `QMAP_TENSOR_F_ROW_MAJOR` | Logical tensor is row-major |
| `1` | `QMAP_TENSOR_F_PACKED_Q4_LOW_EVEN` | Even logical Q4 index is stored in low nibble |
| `2` | `QMAP_TENSOR_F_READ_ONLY` | PL should only read this tensor |
| `3` | `QMAP_TENSOR_F_WRITE_ONLY` | PL is expected to write this tensor |
| `4` | `QMAP_TENSOR_F_DEBUG_ONLY` | Debug/golden data, not needed for inference |

## Descriptor Aux Convention

The fixed descriptor fields should be enough for simple PL readers. The `aux`
fields provide a small amount of role-specific metadata without changing the
128-byte descriptor size:

| Field | Default Meaning |
| --- | --- |
| `aux0` | matrix id or tensor-family id |
| `aux1` | layer index |
| `aux2` | row start, head index, or buffer index depending on role |
| `aux3` | group start, column start, or position start depending on role |

Do not make PL depend on descriptor slot number as the semantic meaning. Slot
number is only a table index. Tensor identity comes from `tensor_id`, `role`,
`aux`, and shape metadata.

## Matrix Ids

Matrix ids are optional descriptor metadata used by early software and debug
flows. They are not a replacement for `tensor_id`.

| Value | Name |
| ---: | --- |
| `1` | `q_proj` |
| `2` | `k_proj` |
| `3` | `v_proj` |
| `4` | `o_proj` |
| `5` | `gate_proj` |
| `6` | `up_proj` |
| `7` | `down_proj` |
| `8` | `embed_lm_head` |

## First Instance: Dot64 Smoke

The first QMAP image should describe the real Qwen3 Layer 0
`q_proj` row 0 group 0 dot64 vector from:

```text
artifacts/test_vectors/qwen3_0p6b_q4_v0/q_proj_row0_group0_dot64.npz
```

This image is a minimal descriptor instance, not a special one-off packet.

Recommended placement:

```text
qmap_base      = 0x4_1B10_0000
payload_base   = 0x4_1B10_0500
image_bytes    = 0x0000_0600
```

Descriptor table:

| Tensor Id | Role | Dtype | Shape | Payload Address | Bytes | Notes |
| ---: | --- | --- | --- | ---: | ---: | --- |
| `1` | activation | `I16_Q4_12` | `[64]` | `0x4_1B10_0500` | 128 | activation slice for columns 0..63 |
| `2` | Q4 weight | `PACKED_Q4_S4` | `[1, 64]` logical | `0x4_1B10_0580` | 32 | `q_proj` row 0, group 0 |
| `3` | Q4 scale | `U16_Q2_14` | `[1, 1]` | `0x4_1B10_05A0` | 2 | scale for row 0 group 0 |
| `4` | expected | `I64` | `[2]` | `0x4_1B10_05C0` | 16 | debug-only partial sum and scaled Q26 sum |

For packed Q4 tensors, the innermost logical dimension is sub-byte. In QMAP
v1, set `strides_bytes[0]` to the physical row stride and set the packed
innermost byte stride to `0`; the `PACKED_Q4_S4` dtype plus
`QMAP_TENSOR_F_PACKED_Q4_LOW_EVEN` flag define how logical columns map to
nibbles.

Expected descriptor metadata:

```text
tensor 1:
  aux0 = 1      # q_proj
  aux1 = 0      # layer index
  aux2 = 0      # row start
  aux3 = 0      # group start

tensor 2:
  aux0 = 1      # q_proj
  aux1 = 0
  aux2 = 0
  aux3 = 0
  group_size = 64
  scale_tensor_id = 3

tensor 3:
  aux0 = 1
  aux1 = 0
  aux2 = 0
  aux3 = 0

tensor 4:
  flags includes QMAP_TENSOR_F_DEBUG_ONLY
```

Hardware validation:

- Apps: `qmap_load_smoke_app`, then `qmap_pl_compute_smoke_app`.
- Durable sources:
  `FPGA_Project/software/qmap_load_smoke/` and
  `FPGA_Project/software/qmap_pl_compute_smoke/`.
- Image base: `0x4_1B10_0000`.
- Image size: 1536 bytes / `0x600`.
- Image SHA256:
  `b56319cf576fe8486e3586ee49a4194323cdfe4f4e6208c8b6057e741e5978d4`.
- Result: PS wrote the image into PL DDR4, read it back byte-for-byte, and
  checked the header, four descriptor slots, and selected payload values.
- PL master result: PS loaded the same image, started
  `qmap_dot64_axi_smoke_0`, and PL read the QMAP image from real PL DDR4
  through AXI. The board run reported status `0xA`, partial sum `0x60AF`,
  and scaled Q26 sum `0x2E1366`.

## Second Instance: Row1024 Smoke

The second QMAP image extends the same descriptor structure from one dot64
group to one complete Layer 0 `q_proj` output row. It is still a smoke image,
but it is the first descriptor-driven shape that matches a real GEMV row:

```text
row_sum_q26 =
  sum_group0_to_15(
    sum_j activation[group*64+j] * q4_weight[row, group*64+j]
    * scale_q2_14[row, group]
  )
```

Recommended placement:

```text
qmap_base      = 0x4_1B20_0000
payload_base   = 0x4_1B20_0500
image_bytes    = 0x0000_1000
```

Descriptor table:

| Tensor Id | Role | Dtype | Shape | Payload Address | Bytes | Notes |
| ---: | --- | --- | --- | ---: | ---: | --- |
| `1` | activation | `I16_Q4_12` | `[1024]` | `0x4_1B20_0500` | 2048 | full normalized activation vector |
| `2` | Q4 weight | `PACKED_Q4_S4` | `[1, 1024]` logical | `0x4_1B20_0D00` | 512 | `q_proj` row 0, all 16 groups |
| `3` | Q4 scale | `U16_Q2_14` | `[1, 16]` | `0x4_1B20_0F00` | 32 | one scale per 64-column group |
| `4` | expected | `I64` | `[1]` | `0x4_1B20_0F40` | 8 | debug-only expected row Q26 sum |

Expected descriptor metadata:

```text
tensor 1:
  aux0 = 1      # q_proj
  aux1 = 0      # layer index
  aux2 = 0      # row start
  aux3 = 0      # group start

tensor 2:
  aux0 = 1      # q_proj
  aux1 = 0
  aux2 = 0
  aux3 = 0
  group_size = 64
  scale_tensor_id = 3
  strides_bytes[0] = 512
  strides_bytes[1] = 0

tensor 3:
  aux0 = 1
  aux1 = 0
  aux2 = 0
  aux3 = 0
  strides_bytes[0] = 32
  strides_bytes[1] = 2

tensor 4:
  flags includes QMAP_TENSOR_F_DEBUG_ONLY
```

The row1024 smoke reader must not assume that one tensor fits in one AXI
burst. With a 32-bit AXI data path, the bring-up RTL now fetches the payload in
4-group batches so that the PL does not need to buffer the full 1024-wide row at
once.

Local validation:

- Exporter: `Qwen3-0.6B-Base/python_each_module/20_export_qmap_row1024_image.py`.
- Binary image: `artifacts/test_vectors/qwen3_0p6b_qmap_v1/q_proj_row0_row1024.qmap.bin`.
- Simulation hex:
  `FPGA_Project/sim/vectors/qmap_row1024_image_words32.hex`.
- Image SHA256:
  `2065fb798848e2779847f2b2055bb6ce51b50fd23547e66de43aa758374a917d`.
- Expected row result:
  `row_sum_q26_int64 = -3482169`, low 32-bit word `0xFFCA_DDC7`.
- RTL simulation result:
  `qmap_row1024_compute_path.sv` passed the QMAP-backed row1024 chain.
- AXI simulation result:
  `qmap_row1024_axi_smoke_top.sv` passed with 18 AXI read bursts, status
  `0xA`, and row result `-3482169`.

Hardware validation:

- App: `qmap_row1024_pl_compute_smoke_app` in the clean short-path Vitis
  workspace, with durable source kept under
  `FPGA_Project/software/qmap_row1024_pl_compute_smoke/`.
- Vitis recreation path used for the passing run: short workspace `F:\vws`,
  platform `p_r1024`, application `a_r1024`.
- Image base: `0x4_1B20_0000`.
- Image size: 4096 bytes / `0x1000`.
- Image SHA256:
  `2065fb798848e2779847f2b2055bb6ce51b50fd23547e66de43aa758374a917d`.
- Result: PS wrote the image into PL DDR4, read it back byte-for-byte, and
  checked the QMAP header fields, descriptor table address, payload base, image
  base, and image size.
- PL master result: PS started the row1024 smoke top through the temporary
  control GPIO at `0xA002_0000`; PL read the QMAP image from real PL DDR4
  through AXI and reported status `0xA`. The result GPIO at `0xA003_0000`
  returned `row_sum_q26_low32=0xFFCA_DDC7` and
  `expected_row_sum_q26_low32=0xFFCA_DDC7`, both representing `-3482169`.

## From Smoke Images to Inference Packets

The dot64 and row1024 images proved the descriptor format and the PL AXI read
path. The next step is to use the same descriptor idea for real inference data
movement.

QMAP now has two intended uses:

1. Persistent model manifest
2. Runtime work packet

### Persistent Model Manifest

The persistent model manifest describes tensors that are loaded once into PL
DDR4 and reused across prompt prefill and decode:

- Q4 packed weights and Q2.14 scales for embedding/LM-head, attention
  projections, and MLP matrices
- RMSNorm gamma vectors and q/k norm gamma vectors. Current local RTL keeps
  signed-capable gamma handling where needed: q_norm/k_norm, post-attention
  RMSNorm, and final RMSNorm use signed `I16_Q8_7` in the current fixed
  vectors; the older Layer 0 input RMSNorm bring-up vector still uses
  unsigned `U16_Q8_8`.
- optional RoPE cos/sin tables
- fixed base addresses for KV cache, activation buffers, logits/argmax
  scratch, and debug regions

For the full Qwen3-0.6B model, the manifest will likely need hundreds of
descriptors. That is acceptable because descriptor tables are fixed-size and
small compared with model weights. The header `descriptor_capacity` should be
chosen for the artifact, not kept at the bring-up value of eight.

The persistent manifest should not be regenerated per token. PS loads it
during model setup, and PL uses it as the address/shape reference for the
runtime scheduler.

### Runtime Work Packet

A runtime work packet describes one PL task or a small sequence of PL tasks.
It should not duplicate large model weights. It should point at already-loaded
weight, scale, activation, KV-cache, and output-buffer regions using absolute
PL DDR4 addresses in each descriptor.

The first formal work packet is the Layer 0 QKV projection packet. It is the
bridge from the proven Q4 row1024 GEMV path into the real model datapath:

```text
input_norm[1024]
  -> q_proj Q4 GEMV -> q_out[2048]
  -> k_proj Q4 GEMV -> k_out[1024]
  -> v_proj Q4 GEMV -> v_out[1024]
```

The packet should use 32 descriptor slots initially:

```text
descriptor_capacity    = 32
descriptor_table_addr  = qmap_base + 0x0100
payload_base_addr      = qmap_base + 0x1100
```

Only small metadata or optional debug payloads need to live inside the packet
image. Large tensors should live in the normal PL DDR4 weight, activation, or
KV-cache regions and be referenced by descriptor `base_addr`.

### Layer 0 QKV Projection Packet

The first formal packet should describe these tensors:

| Tensor Id | Role | Dtype | Shape | Flags | Notes |
| ---: | --- | --- | --- | --- | --- |
| `1` | metadata | `U32` | implementation-defined | read-only | layer index, selected token position, debug controls |
| `2` | activation | `I32_Q12_12` | `[1024]` | read-only | Layer 0 normalized activation input |
| `3` | Q4 weight | `PACKED_Q4_S4` | `[2048, 1024]` logical | read-only, packed-low-even | Layer 0 `q_proj`; row stride 512 bytes |
| `4` | Q4 scale | `U16_Q2_14` | `[2048, 16]` | read-only | one scale per 64-column group for `q_proj` |
| `5` | Q4 weight | `PACKED_Q4_S4` | `[1024, 1024]` logical | read-only, packed-low-even | Layer 0 `k_proj`; row stride 512 bytes |
| `6` | Q4 scale | `U16_Q2_14` | `[1024, 16]` | read-only | one scale per 64-column group for `k_proj` |
| `7` | Q4 weight | `PACKED_Q4_S4` | `[1024, 1024]` logical | read-only, packed-low-even | Layer 0 `v_proj`; row stride 512 bytes |
| `8` | Q4 scale | `U16_Q2_14` | `[1024, 16]` | read-only | one scale per 64-column group for `v_proj` |
| `9` | output | `I32_Q12_12` | `[2048]` | write-only | Q projection output buffer |
| `10` | output | `I32_Q12_12` | `[1024]` | write-only | K projection output buffer |
| `11` | output | `I32_Q12_12` | `[1024]` | write-only | V projection output buffer |
| `12` | expected | implementation-defined | small spot-check tensor | read-only, debug-only | optional golden words or checksum for bring-up |

Descriptor links:

- tensor `3` uses `scale_tensor_id = 4`
- tensor `5` uses `scale_tensor_id = 6`
- tensor `7` uses `scale_tensor_id = 8`
- output tensors are write-only and must not be used as expected/golden data
- debug tensors must be marked `QMAP_TENSOR_F_DEBUG_ONLY`

The PL QKV projection engine should treat Q, K, and V as the same operation
over different descriptors. The matrix dimensions change, but the row GEMV
contract stays:

```text
for each output row:
  for each group of 64 input elements:
    dot64(input_norm[group], packed_q4_weight[row, group])
    scaled_sum += partial_sum * scale[row, group]
  output[row] = convert_Q26_to_Q12_12(scaled_sum)
  write output[row] to the descriptor-provided output buffer
```

This row loop is part of the real Q/K/V projection, not an additional smoke
test. The critical new hardware capability is write-back through a PL AXI
write path.

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/21_export_qmap_qkv_projection_image.py`.
- Full packet:
  `artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection.qmap.bin`.
- Full packet base: `0x4_0008_0000`.
- Full packet size: `0x0022_B000`.
- Full packet row coverage: `q_rows=2048`, `k_rows=1024`, `v_rows=1024`.
- Full packet SHA256:
  `97ce0445a9ba67879ce10093380f5287483550930e4fca2e8c2957ab9d3ea162`.
- Python proof: Q/K/V Q26 recomputation from packed Q4 weights/scales matches
  the source Q4 artifact exactly before Q26-to-I32_Q12.12 conversion.
- Maximum Q12.12 quantization difference versus the Q4 float artifact:
  Q `0.00024390220642089844`, K `0.00024396181106567383`,
  V `0.00024393200874328613`.
- Compact simulation packet:
  `artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection_sim.qmap.bin`.
- Compact simulation packet size: `0x0000_4000`.
- Compact simulation row coverage: `q_rows=4`, `k_rows=2`, `v_rows=2`.
- Compact simulation packet SHA256:
  `e50f4294bad3ff852ca18d52bf5013243ec74d5f5779093691bed16a59d09097`.
- Simulation hex:
  `FPGA_Project/sim/vectors/qmap_qkv_projection_image_words32.hex`.
- Expected output hex:
  `FPGA_Project/sim/vectors/qmap_qkv_projection_expected_words32.hex`.
- RTL simulation result:
  `qmap_qkv_projection_compute_path.sv` wrote all compact Q/K/V output words
  and matched Python I32_Q12.12 expected values. The compact run used 33 read
  requests, wrote 8 output rows, and reported no error.
- AXI write adapter result:
  `axi4_write_master.sv` passed a focused 4-beat aligned AXI4 write-burst
  simulation with a valid B-channel response.

The current exporter can emit self-contained images for simulation and local
bring-up. In the later persistent-manifest flow, the same descriptors should
point to already-loaded weight, scale, activation, and output-buffer regions
instead of copying large model tensors into every runtime packet.

### Attention Front-End Runtime Packet

The first per-layer body runtime packet starts immediately after QKV projection
write-back. It consumes descriptor-visible Q/K/V buffers, q/k norm gamma, and
RoPE tables, then writes K/V cache entries and a Q RoPE output buffer for the
current token.

Dataflow:

```text
q_out[2048] + k_out[1024] + v_out[1024]
  + q_norm.gamma[128] + k_norm.gamma[128]
  + rope_cos[128] + rope_sin[128]
  -> qk_norm_rope_kv_cache_stage
  -> append K/V cache for one layer/position
  -> write q_rope[2048]
```

The current packet base is `0x4_0502_0000`. It reserves 16 descriptor slots,
uses ten active descriptors, and places the payload after the table at
`qmap_base + 0x0900`.

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | layer index, position, and debug summary |
| `1` | Q projection output | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[2048]`, read-only |
| `2` | K projection output | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, read-only |
| `3` | V projection output | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, read-only |
| `4` | q_norm gamma | `QMAP_ROLE_PARAMETER` | `I16_Q8_7`, `[128]`, read-only, padded to 32-bit words in the current packet |
| `5` | k_norm gamma | `QMAP_ROLE_PARAMETER` | `I16_Q8_7`, `[128]`, read-only, padded to 32-bit words in the current packet |
| `6` | RoPE cos | `QMAP_ROLE_ROPE_TABLE` | `I16_Q1_15`, `[128]`, read-only, padded to 32-bit words in the current packet |
| `7` | RoPE sin | `QMAP_ROLE_ROPE_TABLE` | `I16_Q1_15`, `[128]`, read-only, padded to 32-bit words in the current packet |
| `8` | KV cache | `QMAP_ROLE_KV_CACHE` | `I32_Q12_12`, `[2,8,256,128]`, base `0x4_1410_0000` in the current local layout |
| `9` | Q RoPE output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[2048]`, write-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/37_export_qmap_attention_frontend_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_attention_frontend_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_attention_frontend_compute_path.sv`.
- Icarus result: exact `2048` K/V cache write words and exact `2048`
  Q RoPE write words, `31` accepted read requests, `4944` accepted
  read-response words, `2049` accepted write requests, `4096` accepted
  write-data words, deterministic read/write backpressure, busy-period
  spurious start coverage, and an invalid RoPE-cos dtype descriptor path that
  errors with no writes.

### Attention Score/Value Runtime Packet

The next per-layer body packet consumes the Q RoPE output buffer and K/V cache
from the attention front-end path, computes current-token attention scores,
feeds them directly into softmax/value, and writes the attention output before
`o_proj`.

Dataflow:

```text
q_rope[2048] + K cache + V cache + exp_lut[257]
  -> attention_score_stage
  -> attention_softmax_value_stage
  -> write attn_out[2048]
```

The current packet base is `0x4_0503_0000`. It reserves eight descriptor slots,
uses five active descriptors, and places payload data after the table at
`qmap_base + 0x0500`.

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `aux0=score_scale_q0_31`, `aux1=layer_id`, `aux2=cache_length`, `aux3=position` |
| `1` | Q RoPE input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[2048]`, read-only |
| `2` | K/V cache | `QMAP_ROLE_KV_CACHE` | `I32_Q12_12`, `[2,8,256,128]`, read-only for this packet |
| `3` | softmax exp LUT | `QMAP_ROLE_PARAMETER` | `U32_Q0_20`, `[257]`, read-only |
| `4` | attention output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[2048]`, write-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/38_export_qmap_attention_score_value_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_attention_score_value_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_attention_score_value_compute_path.sv`.
- Icarus result: exact `10240` K-cache reads, exact `10240` V-cache reads,
  exact `2048` `attn_out` write words, `20496` accepted read requests,
  `22961` accepted read-response words, one output write request, `2048`
  accepted write-data words, read/write backpressure coverage, and an invalid
  exp-LUT dtype descriptor path that errors with no writes.

### Attention Output Projection Runtime Packet

The next per-layer body packet consumes the `attn_out[2048]` buffer written by
the attention score/value wrapper and applies the Layer 0 `o_proj` matrix. The
large `o_proj` Q4 weight and Q2.14 scale tensors are persistent references, not
runtime-packet payload copies.

Dataflow:

```text
attn_out[2048] + persistent o_proj.weight[1024,2048]
  + persistent o_proj.scale[1024,32]
  -> q4_gemv_row_1024(INPUT_SIZE=2048)
  -> write o_proj_out[1024]
```

The current packet base is `0x4_0504_0000`. It reserves eight descriptor slots,
uses six active descriptors, and places payload data after the table at
`qmap_base + 0x0500`.

Persistent simulation layout:

```text
o_proj_weight_base       = 0x4_0600_0000
o_proj_weight_row_bytes  = 1024
o_proj_weight_full_bytes = 0x0010_0000
o_proj_scale_base        = 0x4_0610_0000
o_proj_scale_row_bytes   = 64
o_proj_scale_full_bytes  = 0x0001_0000
```

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `U32`, `aux0=o_proj`, `aux1=layer_id`, `aux2=1024`, `aux3=2048` |
| `1` | attention output input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[2048]`, read-only |
| `2` | `o_proj` Q4 weight | `QMAP_ROLE_Q4_WEIGHT` | `PACKED_Q4_S4`, `[1024,2048]`, persistent base `0x4_0600_0000` |
| `3` | `o_proj` Q2.14 scale | `QMAP_ROLE_Q4_SCALE` | `U16_Q2_14`, `[1024,32]`, persistent base `0x4_0610_0000` |
| `4` | `o_proj` output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[1024]`, write-only |
| `5` | expected output | `QMAP_ROLE_EXPECTED` | `I32_Q12_12`, `[1024]`, debug-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/39_export_qmap_o_proj_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_o_proj_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_o_proj_compute_path.sv`.
- Icarus result: exact `1024` Q4 weight-row reads, exact `1024` scale-row
  reads, exact `1024` `o_proj_out` write words, `2070` accepted read requests,
  `280992` accepted read-response words, one output write request, `1024`
  accepted write-data words, request/response/write-data backpressure coverage,
  a busy-period spurious start pulse, and an invalid Q4 weight group-size
  descriptor path that errors with no writes. The CSV trace audit confirms row
  stores run from row 0 through row 1023 before the single output write-back
  burst and that done pulses are not adjacent.

### Post-Attention Residual/RMSNorm Runtime Packet

The next per-layer body packet consumes the `o_proj_out[1024]` buffer and the
Layer 0 residual stream, applies the post-attention residual add, then applies
post-attention RMSNorm with signed `Q8.7` gamma.

Dataflow:

```text
residual_input[1024] + o_proj_out[1024] + post_attention_gamma[1024]
  -> residual_add_1024
  -> rmsnorm_1024(signed gamma)
  -> write post_attention_hidden[1024]
  -> write post_norm[1024]
```

The current packet base is `0x4_0505_0000`. It reserves and uses eight active
descriptor slots and places payload data after the table at
`qmap_base + 0x0500`. The generated local simulation image is `0x8000` bytes.

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `U32`, `aux0=post_attn_residual_norm`, `aux1=layer_id`, `aux2=1024` |
| `1` | residual input | `QMAP_ROLE_ACTIVATION` | `I32_Q14_10`, `[1024]`, read-only |
| `2` | `o_proj` output input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, read-only, `aux0=o_proj` |
| `3` | post-attention gamma | `QMAP_ROLE_PARAMETER` | `I16_Q8_7`, `[1024]`, read-only, stored as 32-bit words |
| `4` | post-attention hidden | `QMAP_ROLE_OUTPUT` | `I32_Q14_10`, `[1024]`, write-only |
| `5` | post-norm output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[1024]`, write-only |
| `6` | expected hidden | `QMAP_ROLE_EXPECTED` | `I32_Q14_10`, `[1024]`, debug-only |
| `7` | expected norm | `QMAP_ROLE_EXPECTED` | `I32_Q12_12`, `[1024]`, debug-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/40_export_qmap_post_attention_residual_norm_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_post_attention_residual_norm_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_post_attention_residual_norm_compute_path.sv`.
- Icarus result: exact `1024` post-attention hidden write words, exact `1024`
  post-norm write words, `30` accepted read requests, `3616` accepted
  read-response words, two output write requests, `2048` accepted write-data
  words, request/response/write-data backpressure coverage, a busy-period
  spurious start pulse, and an invalid gamma dtype descriptor path that errors
  with no writes. The CSV trace audit confirms the stage-complete snapshot and
  first hidden write request occur in the same cycle, all write data completes
  before done, and normal/bad done pulses are not adjacent.

### MLP Gate/Up Runtime Packet

The next per-layer body packet consumes the post-attention RMSNorm output and
computes Layer 0 MLP gate/up projections from persistent Q4 weight/scale
tensors.

Dataflow:

```text
post_norm[1024]
  + persistent gate_proj.weight[3072,1024] / gate_proj.scale[3072,16]
  + persistent up_proj.weight[3072,1024] / up_proj.scale[3072,16]
  -> two q4_gemv_row_1024 cores in parallel
  -> write gate[3072]
  -> write up[3072]
```

The current packet base is `0x4_0506_0000`. It uses ten active descriptors,
sets `descriptor_capacity=16`, and places payload data after the table at
`qmap_base + 0x0900`. The generated local simulation image is `0xE000` bytes.

Persistent simulation layout:

```text
gate_weight_base       = 0x4_0620_0000
gate_weight_row_bytes  = 512
gate_weight_full_bytes = 0x0018_0000
gate_scale_base        = 0x4_0638_0000
gate_scale_row_bytes   = 32
gate_scale_full_bytes  = 0x0001_8000
up_weight_base         = 0x4_0640_0000
up_weight_row_bytes    = 512
up_weight_full_bytes   = 0x0018_0000
up_scale_base          = 0x4_0658_0000
up_scale_row_bytes     = 32
up_scale_full_bytes    = 0x0001_8000
```

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `U32`, `aux0=mlp_gate_up`, `aux1=layer_id`, `aux2=3072`, `aux3=1024` |
| `1` | post-norm input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, read-only |
| `2` | gate Q4 weight | `QMAP_ROLE_Q4_WEIGHT` | `PACKED_Q4_S4`, `[3072,1024]`, persistent base `0x4_0620_0000` |
| `3` | gate Q2.14 scale | `QMAP_ROLE_Q4_SCALE` | `U16_Q2_14`, `[3072,16]`, persistent base `0x4_0638_0000` |
| `4` | up Q4 weight | `QMAP_ROLE_Q4_WEIGHT` | `PACKED_Q4_S4`, `[3072,1024]`, persistent base `0x4_0640_0000` |
| `5` | up Q2.14 scale | `QMAP_ROLE_Q4_SCALE` | `U16_Q2_14`, `[3072,16]`, persistent base `0x4_0658_0000` |
| `6` | gate output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[3072]`, write-only |
| `7` | up output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[3072]`, write-only |
| `8` | expected gate | `QMAP_ROLE_EXPECTED` | `I32_Q12_12`, `[3072]`, debug-only |
| `9` | expected up | `QMAP_ROLE_EXPECTED` | `I32_Q12_12`, `[3072]`, debug-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/41_export_qmap_mlp_gate_up_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_mlp_gate_up_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_mlp_gate_up_compute_path.sv`.
- Icarus result: exact `3072` gate write words, exact `3072` up write words,
  `12314` accepted read requests, `837280` accepted read-response words, two
  output write requests, `6144` accepted write-data words,
  request/response/write-data backpressure coverage, a busy-period spurious
  start pulse, and an invalid up-scale dtype descriptor path that errors with
  no writes. The CSV trace audit confirms 3072 row completions, gate/up write
  requests at `0x4_0506_1940` and `0x4_0506_4940`, both length 12288 bytes,
  all write data completes before done, and normal/bad done pulses are not
  adjacent.

### MLP SiLU/Multiply Runtime Packet

The next per-layer body packet consumes the MLP gate/up outputs and computes the
Layer 0 MLP activation intermediate.

Dataflow:

```text
gate[3072] + up[3072] + sigmoid_lut[1025]
  -> mlp_silu_mul_stage
  -> write mlp_hidden[3072]
```

The current packet base is `0x4_0507_0000`. It uses six active descriptors,
sets `descriptor_capacity=16`, and places payload data after the table at
`qmap_base + 0x0900`. The generated local simulation image is `0xE000` bytes.
The sigmoid LUT uses unsigned `UQ0.16` values over `[-8, 8]` with `1/64`
spacing; each LUT entry is stored in the low 16 bits of one 32-bit DDR word.

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `U32`, `aux0=mlp_silu_mul`, `aux1=layer_id`, `aux2=3072`, `aux3=1025` |
| `1` | gate input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[3072]`, read-only |
| `2` | up input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[3072]`, read-only |
| `3` | sigmoid LUT | `QMAP_ROLE_PARAMETER` | `U16_Q0_16`, `[1025]`, read-only, padded to 32-bit words |
| `4` | hidden output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[3072]`, write-only |
| `5` | expected hidden | `QMAP_ROLE_EXPECTED` | `I32_Q12_12`, `[3072]`, debug-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/42_export_qmap_mlp_silu_mul_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_mlp_silu_mul_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_mlp_silu_mul_compute_path.sv`.
- Icarus result: exact `3072` stage input handshakes, exact `3072` stage
  output handshakes, exact `3072` hidden write words, `43` accepted read
  requests across the normal plus invalid-descriptor runs, `7585` accepted
  read-response words, one output write request, `3072` accepted write-data
  words, request/response/write-data backpressure coverage, a busy-period
  spurious start pulse, and an invalid LUT dtype descriptor path that errors
  with no writes. The normal DUT counters are `36` read requests, `7377`
  read-response words, one write request, and `3072` write words. The CSV trace
  audit confirms the hidden write request at `0x4_0507_7980`, length `12288`
  bytes, all write data completes before done, and normal/bad done pulses are
  not adjacent.

### MLP Down Runtime Packet

The next per-layer body packet consumes the MLP SiLU/multiply hidden output and
computes the Layer 0 MLP down projection from persistent Q4 weight/scale
tensors.

Dataflow:

```text
mlp_hidden[3072]
  + persistent down_proj.weight[1024,3072] / down_proj.scale[1024,48]
  -> q4_gemv_row_1024 with INPUT_SIZE=3072
  -> write down_out[1024]
```

The current packet base is `0x4_0508_0000`. It uses six active descriptors,
sets `descriptor_capacity=16`, and places payload data after the table at
`qmap_base + 0x0900`. The generated local simulation image is `0x6000` bytes.

Persistent simulation layout:

```text
down_weight_base        = 0x4_0660_0000
down_weight_row_bytes   = 1536
down_weight_full_bytes  = 0x0018_0000
down_scale_base         = 0x4_0678_0000
down_scale_row_bytes    = 96
down_scale_full_bytes   = 0x0001_8000
```

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `U32`, `aux0=mlp_down`, `aux1=layer_id`, `aux2=1024`, `aux3=3072` |
| `1` | MLP hidden input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[3072]`, read-only |
| `2` | down Q4 weight | `QMAP_ROLE_Q4_WEIGHT` | `PACKED_Q4_S4`, `[1024,3072]`, persistent base `0x4_0660_0000`, row stride `1536` bytes |
| `3` | down Q2.14 scale | `QMAP_ROLE_Q4_SCALE` | `U16_Q2_14`, `[1024,48]`, persistent base `0x4_0678_0000`, row stride `96` bytes |
| `4` | down output | `QMAP_ROLE_OUTPUT` | `I32_Q12_12`, `[1024]`, write-only |
| `5` | expected down output | `QMAP_ROLE_EXPECTED` | `I32_Q12_12`, `[1024]`, debug-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/43_export_qmap_mlp_down_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_mlp_down_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_mlp_down_compute_path.sv`.
- Icarus result: exact `1024` row completions, exact `1024` output write
  words, `3098` accepted read requests across the normal plus invalid-descriptor
  runs, `421280` accepted read-response words, one output write request, `1024`
  accepted write-data words, request/response/write-data backpressure coverage,
  a busy-period spurious start pulse, and an invalid scale dtype descriptor path
  that errors with no writes. The normal DUT counters are `3091` read requests,
  `421072` read-response words, one write request, and `1024` write words. The
  CSV trace audit confirms the down output write request at `0x4_0508_3940`,
  length `4096` bytes, all write data completes before done, and normal/bad
  done pulses are not adjacent.

### Final MLP Residual Runtime Packet

The next per-layer body packet consumes the post-attention hidden vector and the
MLP down-projection output, then computes the Layer 0 final MLP residual add.

Dataflow:

```text
post_attn_hidden[1024] + down_out[1024]
  -> mlp_residual_add_stage
  -> write layer_out[1024]
```

The current packet base is `0x4_0509_0000`. It uses five active descriptors,
sets `descriptor_capacity=8`, and places payload data after the table at
`qmap_base + 0x0500`. The generated local simulation image is `0x5000` bytes.

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | `U32`, `aux0=mlp_residual_add`, `aux1=layer_id`, `aux2=1024` |
| `1` | post-attention hidden input | `QMAP_ROLE_ACTIVATION` | `I32_Q14_10`, `[1024]`, read-only |
| `2` | MLP down output input | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, read-only |
| `3` | layer output | `QMAP_ROLE_OUTPUT` | `I32_Q14_10`, `[1024]`, write-only |
| `4` | expected layer output | `QMAP_ROLE_EXPECTED` | `I32_Q14_10`, `[1024]`, debug-only |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/44_export_qmap_mlp_residual_add_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_mlp_residual_add_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_mlp_residual_add_compute_path.sv`.
- Icarus result: exact `1024` output words, one `4096`-byte output write
  burst, request/response/write-data backpressure coverage, a busy-period
  spurious start pulse, a bad down-input dtype descriptor path with no writes,
  and a malformed payload `last` protocol path with no writes. The normal DUT
  counters are `14` read requests, `2224` read-response words, one write
  request, and `1024` write words. The full three-run test covers `27` read
  requests and `2577` read-response words. The CSV trace audit confirms the
  layer output write request at `0x4_0509_2540`, length `4096` bytes, all
  write data completes before done, and all error paths complete without
  writes.

### LM-head Memory-Backed Scan Contract

The first LM-head memory-facing path uses the tied embedding/LM-head table as a
Q4 weight matrix with one Q2.14 scale per 64 columns. The current local vector
exporters are:

```text
Qwen3-0.6B-Base/python_each_module/33_export_lm_head_argmax_vectors.py
Qwen3-0.6B-Base/python_each_module/35_export_lm_head_full_vocab_vectors.py
```

The exported layout keeps full-vocabulary base addresses even when the local
testbench loads only the `[0,1024)` scan window:

```text
lm_head_weight_base       = 0x4_0010_0000
lm_head_weight_row_bytes  = 512
lm_head_weight_full_bytes = 0x04A3_0000
lm_head_scale_base        = 0x4_04B3_0000
lm_head_scale_row_bytes   = 32
lm_head_scale_full_bytes  = 0x004A_3000
```

`lm_head_tile_mem_reader.sv` reads one 16-row tile as eight 1024-byte packed-Q4
weight bursts plus one 512-byte scale burst. `lm_head_argmax_mem_stage.sv`
wraps that reader around `lm_head_argmax_stage.sv` without changing the argmax
core. The local memory-model testbench passes for two complete 1024-row scans:
`1152` read requests, `278528` response words, `2048` checked logits, exact
token `264`, exact score `1365150750`, and no error cycles.

`lm_head_argmax_tile_scheduler.sv` reuses that memory-backed stage as a
one-tile child engine and keeps the global best token/score across runtime tile
windows. Its default capacity is `MAX_TILES=9496`, matching the full
`151936`-row vocabulary at 16 rows per tile. The current local scheduler
testbench proves two runtime scan counts: `64` tiles for the existing
1024-row window and `23` tiles for a 368-row prefix. It matches exact token
`264`, exact score `1365150750`, and `1392` checked logits with
`max_abs_logit_diff=0`.

`34_export_qmap_lm_head_argmax_image.py` exports QMAP runtime packets for this
path. The packet base is `0x4_0500_0000`, uses six active descriptors, and
keeps the large LM-head weights/scales as persistent tensor references instead
of copying them into the runtime packet. The same descriptor contract is used
for the compact `[0,1024)` scan and the full `[0,151936)` vocabulary scan:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | scan metadata and debug summary |
| `1` | final RMSNorm output | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, read-only |
| `2` | LM-head Q4 weight | `QMAP_ROLE_Q4_WEIGHT` | `PACKED_Q4_S4`, `[151936,1024]`, `aux2=scan_base`, `aux3=tile_count` |
| `3` | LM-head Q2.14 scale | `QMAP_ROLE_Q4_SCALE` | `U16_Q2_14`, `[151936,16]`, same scan aux fields |
| `4` | output token/score | `QMAP_ROLE_OUTPUT` | three `U32` words: token, score low32, score high32 |
| `5` | expected token/score | `QMAP_ROLE_EXPECTED` | debug-only copy of the three output words |

`qmap_lm_head_argmax_compute_path.sv` validates those descriptors, reads the
final RMSNorm activation, runs the scheduler, and writes
`{token, score_low32, score_high32}` to the output descriptor. Its testbench
covers descriptor-provided `64`-tile and `23`-tile scans, plus an invalid
`tile_count=0` descriptor that must raise error and perform no output write.
The compact Icarus run reports token `264`, score `1365150750`, `812` read
requests, `191984` response words, `1392` checked logits, and
`max_abs_logit_diff=0`. The full-vocabulary Vivado xsim run uses
`QMAP_LM_HEAD_TB_FULL_VOCAB` and reports `9496` tiles, `151936` checked logits,
`85475` read requests, `20664528` response words, one output write burst,
token `264`, score `1365150750`, and `max_abs_logit_diff=0`.

### Final-Token Tail Runtime Packet

The first composed one-token tail packet adds final RMSNorm in front of the
descriptor-backed LM-head scan. It is intentionally still a runtime packet, not
a persistent manifest. The packet base is `0x4_0501_0000`, uses eight active
descriptors, and keeps the large LM-head weights/scales as persistent tensor
references.

Dataflow:

```text
final_hidden[1024] + final_norm.gamma[1024]
  -> final_rmsnorm_stage
  -> write final_norm[1024] to the QMAP activation descriptor
  -> qmap_lm_head_argmax_compute_path
  -> write output token/score descriptor
```

Descriptor table:

| Slot | Tensor | Role | Key fields |
| ---: | --- | --- | --- |
| `0` | metadata | `QMAP_ROLE_METADATA` | tail metadata and debug summary |
| `1` | final-norm scratch / LM-head activation | `QMAP_ROLE_ACTIVATION` | `I32_Q12_12`, `[1024]`, written by final RMSNorm then read by LM-head |
| `2` | LM-head Q4 weight | `QMAP_ROLE_Q4_WEIGHT` | `PACKED_Q4_S4`, `[151936,1024]`, `aux2=scan_base`, `aux3=tile_count` |
| `3` | LM-head Q2.14 scale | `QMAP_ROLE_Q4_SCALE` | `U16_Q2_14`, `[151936,16]`, same scan aux fields |
| `4` | output token/score | `QMAP_ROLE_OUTPUT` | three `U32` words: token, score low32, score high32 |
| `5` | expected token/score | `QMAP_ROLE_EXPECTED` | debug-only copy of the three output words |
| `6` | final hidden input | `QMAP_ROLE_ACTIVATION` | `I32_Q14_10`, `[1024]`, read-only |
| `7` | final RMSNorm gamma | `QMAP_ROLE_PARAMETER` | `I16_Q8_7`, `[1024]`, read-only signed gamma padded to 32-bit words in the current packet |

Local validation:

- Exporter:
  `Qwen3-0.6B-Base/python_each_module/36_export_qmap_final_token_tail_image.py`.
- RTL wrapper:
  `FPGA_Project/rtl/qmap_final_token_tail_compute_path.sv`.
- Testbench:
  `FPGA_Project/sim/tb_qmap_final_token_tail_compute_path.sv`.
- Compact Icarus result: `64` LM-head tiles, token `264`, score
  `1365150750`, `1024` exact final-norm write words, `1024` checked logits,
  `max_abs_logit_diff=0`, invalid gamma descriptor error path covered, and no
  stale best-result exposure.
- Full-vocabulary Vivado xsim result: `9496` tiles, `151936` checked logits,
  token `264`, score `1365150750`, `1024` final-norm write words, `3`
  token/score output words, `max_abs_logit_diff=0`, one done pulse, and zero
  error/saturation rows in the trace audit.

## Full-Model Runtime Write-Back Policy

For correctness-first inference, these runtime values should be writable in PL
DDR4, even if later optimized versions keep some of them on chip:

| Tensor Family | Shape | Suggested Dtype | Write Policy |
| --- | --- | --- | --- |
| hidden ping-pong buffers | `[1024]` | `I32_Q14_10` or `I32_Q12_12` by stage | write after embedding/layer output |
| RMSNorm output | `[1024]` | `I32_Q12_12` | write for projection input and debug |
| Q projection output | `[2048]` | `I32_Q12_12` | write for q_norm/RoPE input |
| K projection output | `[1024]` | `I32_Q12_12` | write for k_norm/RoPE input |
| V projection output | `[1024]` | `I32_Q12_12` | write for KV-cache append |
| post-RoPE K cache | `[28, 8, T, 128]` | `I32_Q12_12` | append one position per layer/token |
| V cache | `[28, 8, T, 128]` | `I32_Q12_12` | append one position per layer/token |
| attention output before `o_proj` | `[2048]` | `I32_Q12_12` | QMAP score/value wrapper now writes exact `attn_out[2048]` locally |
| `o_proj_out` | `[1024]` | `I32_Q12_12` | QMAP `o_proj` wrapper now writes exact output locally |
| post-attention hidden | `[1024]` | `I32_Q14_10` | QMAP post-attention wrapper writes and validates this residual result before post-attention RMSNorm |
| MLP gate/up outputs | `[3072]` each | `I32_Q12_12` | QMAP MLP gate/up wrapper now writes exact gate/up buffers locally |
| MLP SiLU/multiply intermediate | `[3072]` | `I32_Q12_12` | QMAP MLP SiLU/multiply wrapper now writes exact hidden locally; this is the MLP down-projection input |
| MLP down output | `[1024]` | `I32_Q12_12` | QMAP MLP down wrapper now writes exact down output locally; this is the final MLP residual input |
| layer output | `[1024]` | `I32_Q14_10` | QMAP final MLP residual wrapper now writes exact layer output locally; this is the next layer input |
| final RMSNorm output | `[1024]` | `I32_Q12_12` | local RTL final-norm stage passes; LM-head input |
| logits/argmax scratch | tiled | implementation-defined | local RTL tiled argmax core, memory-backed 1024-row wrapper, runtime tile scheduler, QMAP descriptor wrapper, and full-vocab QMAP LM-head scan pass; writes final token id and token score |

Do not require all of these tensors to be stored forever. The rule is that a
formal packet must make every consumed or produced tensor explicit. Later RTL
can replace DDR write-back with streaming or on-chip buffering only after the
descriptor-level contract is clear.

The PL reader and scheduler should therefore be designed around descriptors and
tensor ids, not around a hard-coded 256-byte packet.

## Open Decisions

- Exact checksum algorithm for `checksum32`.
- Whether full-model QMAP images store absolute physical addresses only, or
  store relative offsets plus a runtime base address.
- Final descriptor count/capacity for the persistent full-model manifest.
- Exact tensor ids for the persistent full-model manifest.
- Whether a future DMA loader should preserve the same physical QMAP image
  layout or translate it while copying into PL DDR4.
