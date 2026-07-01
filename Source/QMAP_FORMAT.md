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
tiles. The full `9496`-tile vocabulary proof remains pending.

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
| `9` | `QMAP_ROLE_PARAMETER` | Small read-only model parameter such as RMSNorm gamma |

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
- RMSNorm gamma vectors and q/k norm gamma vectors. Current local RTL uses
  unsigned `U16_Q8_8` for layer input/post/final RMSNorm gamma and signed
  `I16_Q8_7` for q_norm/k_norm gamma.
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

### LM-head Memory-Backed Scan Contract

The first LM-head memory-facing path uses the tied embedding/LM-head table as a
Q4 weight matrix with one Q2.14 scale per 64 columns. The current local vector
exporter is:

```text
Qwen3-0.6B-Base/python_each_module/33_export_lm_head_argmax_vectors.py
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

`34_export_qmap_lm_head_argmax_image.py` exports the first QMAP runtime packet
for this path. The packet base is `0x4_0500_0000`, uses six active descriptors,
and keeps the large LM-head weights/scales as persistent tensor references
instead of copying them into the runtime packet:

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
The full `151936`-row / `9496`-tile vocabulary scan is still the next proof.

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
| attention output before `o_proj` | `[2048]` | `I32_Q12_12` | local RTL supports streaming into `o_proj`; DDR write-back remains useful for debug |
| post-attention hidden | `[1024]` | `I32_Q14_10` | local RTL writes/validates this residual result before post-attention RMSNorm |
| MLP gate/up outputs | `[3072]` each | `I32_Q12_12` | local RTL projection stage passes; write for correctness bring-up if not streamed |
| MLP SiLU/multiply intermediate | `[3072]` | `I32_Q12_12` | local RTL stage passes; this is the MLP down-projection input |
| MLP down output | `[1024]` | `I32_Q12_12` | local RTL stage passes; this is the final MLP residual input |
| layer output | `[1024]` | `I32_Q14_10` or `I32_Q12_12` | local RTL final residual stage passes; write to next layer input buffer |
| final RMSNorm output | `[1024]` | `I32_Q12_12` | local RTL final-norm stage passes; LM-head input |
| logits/argmax scratch | tiled | implementation-defined | local RTL tiled argmax core, memory-backed 1024-row wrapper, runtime tile scheduler, and QMAP descriptor wrapper pass; future full-vocab path should write final token id and optional debug logits |

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
