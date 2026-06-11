# QMAP Format

Status: draft v1 descriptor format for PL DDR4 tensor staging and future
full-model artifact layout. The first dot64 image has passed both PS
load/readback and PL AXI-master read/compute hardware smoke tests.

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

The first QMAP instance will describe the real Qwen3 Layer 0
`q_proj` row 0 group 0 dot64 vector. Later instances can describe a full GEMV
row, a projection tile, all Layer 0 Q/K/V projections, or the full model by
adding descriptors and increasing tensor shapes.

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

The QMAP header lives at the base of a QMAP image. The first test-vector QMAP
image will be placed at `0x4_1B10_0000`, the start of the draft PL DDR4
test-vector staging region.

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

The first bring-up image should reserve eight descriptor slots even if only a
few are valid. This keeps the payload base stable and leaves room for adding an
output or debug descriptor without changing the image shape.

Recommended first image base:

```text
qmap_base              = 0x4_1B10_0000
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

## Tensor Flags

| Bit | Name | Meaning |
| ---: | --- | --- |
| `0` | `QMAP_TENSOR_F_ROW_MAJOR` | Logical tensor is row-major |
| `1` | `QMAP_TENSOR_F_PACKED_Q4_LOW_EVEN` | Even logical Q4 index is stored in low nibble |
| `2` | `QMAP_TENSOR_F_READ_ONLY` | PL should only read this tensor |
| `3` | `QMAP_TENSOR_F_WRITE_ONLY` | PL is expected to write this tensor |
| `4` | `QMAP_TENSOR_F_DEBUG_ONLY` | Debug/golden data, not needed for inference |

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
burst. With a 32-bit AXI data path, the 2048-byte activation payload requires
two 1024-byte reads when using the current one-burst `axi4_read_master`.

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
  `qmap_row1024_axi_smoke_top.sv` passed with 10 AXI read bursts, status
  `0xA`, and row result `-3482169`.

## Scale-Up Path

The dot64 instance scales without changing the QMAP structure:

```text
dot64:
  activation [64]
  weight logical [1, 64]
  scale [1, 1]

row1024:
  activation [1024]
  weight logical [1, 1024], physical [1, 512 bytes]
  scale [1, 16]

q_proj tile:
  activation [1024]
  weight logical [tile_rows, 1024]
  scale [tile_rows, 16]
  output [tile_rows]

full q_proj:
  activation [1024]
  weight logical [2048, 1024]
  scale [2048, 16]
  output [2048]

full layer or full model:
  add descriptors for q/k/v/o, MLP matrices, RMSNorm gamma, activations,
  KV cache, RoPE table, outputs, and optional debug tensors
```

The PL reader should therefore be designed around descriptors and tensor ids,
not around a hard-coded 256-byte packet.

## Open Decisions

- Exact checksum algorithm for `checksum32`.
- Whether full-model QMAP images store absolute physical addresses only, or
  store relative offsets plus a runtime base address.
- Final descriptor set for full Q4 model artifacts.
- Whether a future DMA loader should preserve the same physical QMAP image
  layout or translate it while copying into PL DDR4.
