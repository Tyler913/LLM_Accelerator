from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
QMAP_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_qmap_v1"

O_PROJ_PREFIX = "o_proj_stage_real"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "o_proj_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "o_proj_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_o_proj_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_o_proj_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0504_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 8
DESCRIPTOR_COUNT = 6
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0500
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

O_PROJ_WEIGHT_BASE_ADDR = 0x4_0600_0000
O_PROJ_SCALE_BASE_ADDR = 0x4_0610_0000

INPUT_SIZE = 2048
OUT_FEATURES = 1024
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
ACT_WIDTH = 24
WEIGHT_WIDTH = 4
SCALE_WIDTH = 16
OUT_WIDTH = 24
ROW_ACC_WIDTH = 64
LAYER_ID = 0

TENSOR_ID_METADATA = 45
TENSOR_ID_ACTIVATION = 46
TENSOR_ID_WEIGHT = 47
TENSOR_ID_SCALE = 48
TENSOR_ID_OUTPUT = 49
TENSOR_ID_EXPECTED = 50

ROLE_ACTIVATION = 1
ROLE_Q4_WEIGHT = 2
ROLE_Q4_SCALE = 3
ROLE_OUTPUT = 4
ROLE_EXPECTED = 5
ROLE_METADATA = 6

DTYPE_U32 = 5
DTYPE_PACKED_Q4_S4 = 16
DTYPE_U16_Q2_14 = 17
DTYPE_I32_Q12_12 = 19

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_PACKED_Q4_LOW_EVEN = 1 << 1
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3
TENSOR_F_DEBUG_ONLY = 1 << 4

MATRIX_ID_O_PROJ = 4
NO_TENSOR_ID = 0xFFFF_FFFF


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hex_lines(path: Path, width_bits: int, *, signed: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    sign_bit = 1 << (width_bits - 1)
    full = 1 << width_bits
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw = int(stripped, 16)
        if signed and (raw & sign_bit):
            raw -= full
        values.append(raw)
    return np.array(values, dtype=np.int64)


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    digits = (width_bits + 3) // 4
    flat = values.reshape(-1)
    path.write_text("\n".join(f"{int(value) & mask:0{digits}x}" for value in flat) + "\n", encoding="utf-8")


def write_scalar(path: Path, value: int, width_bits: int) -> None:
    write_hex_lines(path, np.array([value], dtype=np.uint64), width_bits)


def parse_int_auto(value: str) -> int:
    return int(value.replace("_", ""), 0)


def image_to_words32(image: bytes) -> np.ndarray:
    padded = image + bytes((-len(image)) % 4)
    return np.array(struct.unpack("<" + "I" * (len(padded) // 4), padded), dtype=np.uint32)


def as_le_bytes(array: np.ndarray, dtype: str) -> bytes:
    converted = np.ascontiguousarray(array.astype(np.dtype(dtype), copy=False))
    return converted.tobytes(order="C")


def pack_header(qmap_base: int, image_bytes: int) -> bytes:
    descriptor_table_addr = qmap_base + DESCRIPTOR_TABLE_OFFSET
    payload_base_addr = qmap_base + PAYLOAD_BASE_OFFSET
    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<IIIIIIQQQQII",
        header,
        0,
        QMAP_MAGIC,
        QMAP_VERSION,
        HEADER_BYTES,
        DESCRIPTOR_BYTES,
        DESCRIPTOR_COUNT,
        DESCRIPTOR_CAPACITY,
        descriptor_table_addr,
        payload_base_addr,
        qmap_base,
        image_bytes,
        0,
        0,
    )
    return bytes(header)


def pack_descriptor(
    *,
    tensor_id: int,
    role: int,
    dtype: int,
    rank: int,
    flags: int,
    element_bits: int,
    group_size: int,
    scale_tensor_id: int,
    base_addr: int,
    nbytes: int,
    dims: tuple[int, int, int, int],
    strides: tuple[int, int, int, int],
    aux: tuple[int, int, int, int],
) -> bytes:
    descriptor = struct.pack(
        "<8I2Q4I4Q8I",
        tensor_id,
        role,
        dtype,
        rank,
        flags,
        element_bits,
        group_size,
        scale_tensor_id,
        base_addr,
        nbytes,
        *dims,
        *strides,
        *aux,
        0,
        0,
        0,
        0,
    )
    if len(descriptor) != DESCRIPTOR_BYTES:
        raise RuntimeError(f"descriptor size mismatch: {len(descriptor)}")
    return descriptor


def add_payload(payloads: list[dict[str, Any]], *, name: str, cursor: int, payload: bytes) -> int:
    offset = align_up(cursor, PAYLOAD_ALIGNMENT)
    payloads.append({"name": name, "offset": offset, "payload": payload})
    return offset + len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a QMAP runtime packet for o_proj simulation.")
    parser.add_argument("--prefix", default=O_PROJ_PREFIX)
    parser.add_argument("--layer-id", type=int, default=LAYER_ID)
    parser.add_argument("--qmap-base", type=parse_int_auto, default=QMAP_BASE)
    parser.add_argument("--weight-base", type=parse_int_auto, default=O_PROJ_WEIGHT_BASE_ADDR)
    parser.add_argument("--scale-base", type=parse_int_auto, default=O_PROJ_SCALE_BASE_ADDR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sim-hex", type=Path, default=DEFAULT_SIM_HEX)
    parser.add_argument("--expected-hex", type=Path, default=DEFAULT_EXPECTED_HEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = str(args.prefix)
    layer_id = int(args.layer_id)
    qmap_base = int(args.qmap_base)
    weight_base_addr = int(args.weight_base)
    scale_base_addr = int(args.scale_base)

    activation = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_activation.hex", ACT_WIDTH, signed=True)
    expected = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_expected_q12_12.hex", OUT_WIDTH, signed=True)
    weight_words = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_weight_words32.hex", 32, signed=False)
    scale_words = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_scale_words32.hex", 32, signed=False)

    if activation.shape != (INPUT_SIZE,):
        raise RuntimeError(f"activation shape mismatch: expected {(INPUT_SIZE,)}, got {activation.shape}")
    if expected.shape != (OUT_FEATURES,):
        raise RuntimeError(f"expected shape mismatch: expected {(OUT_FEATURES,)}, got {expected.shape}")

    weight_row_bytes = INPUT_SIZE * WEIGHT_WIDTH // 8
    scale_row_bytes = GROUP_COUNT * SCALE_WIDTH // 8
    weight_bytes = OUT_FEATURES * weight_row_bytes
    scale_bytes = OUT_FEATURES * scale_row_bytes

    if weight_words.size * 4 != weight_bytes:
        raise RuntimeError(f"weight_words size mismatch: got {weight_words.size * 4}, expected {weight_bytes}")
    if scale_words.size * 4 != scale_bytes:
        raise RuntimeError(f"scale_words size mismatch: got {scale_words.size * 4}, expected {scale_bytes}")

    activation_i32 = activation.astype(np.int32)
    expected_i32 = expected.astype(np.int32)
    metadata_words = np.array(
        [
            layer_id,
            OUT_FEATURES,
            INPUT_SIZE,
            GROUP_SIZE,
            GROUP_COUNT,
            weight_base_addr & 0xFFFF_FFFF,
            scale_base_addr & 0xFFFF_FFFF,
            MATRIX_ID_O_PROJ,
        ],
        dtype=np.uint32,
    )

    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET
    cursor = add_payload(payloads, name="metadata", cursor=cursor, payload=as_le_bytes(metadata_words, "<u4"))
    cursor = add_payload(payloads, name="attn_out_q12_12", cursor=cursor, payload=as_le_bytes(activation_i32, "<i4"))
    cursor = add_payload(payloads, name="o_proj_out_q12_12", cursor=cursor, payload=bytes(OUT_FEATURES * 4))
    cursor = add_payload(payloads, name="expected_o_proj_out_q12_12", cursor=cursor, payload=as_le_bytes(expected_i32, "<i4"))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return qmap_base + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    packed_ro = ro | TENSOR_F_PACKED_Q4_LOW_EVEN
    expected_flags = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY

    descriptors = [
        pack_descriptor(
            tensor_id=TENSOR_ID_METADATA,
            role=ROLE_METADATA,
            dtype=DTYPE_U32,
            rank=1,
            flags=ro,
            element_bits=32,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("metadata"),
            nbytes=size("metadata"),
            dims=(int(metadata_words.size), 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_O_PROJ, layer_id, OUT_FEATURES, INPUT_SIZE),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_ACTIVATION,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=ACT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("attn_out_q12_12"),
            nbytes=INPUT_SIZE * 4,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_O_PROJ, layer_id, 0, 0),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_WEIGHT,
            role=ROLE_Q4_WEIGHT,
            dtype=DTYPE_PACKED_Q4_S4,
            rank=2,
            flags=packed_ro,
            element_bits=WEIGHT_WIDTH,
            group_size=GROUP_SIZE,
            scale_tensor_id=TENSOR_ID_SCALE,
            base_addr=weight_base_addr,
            nbytes=weight_bytes,
            dims=(OUT_FEATURES, INPUT_SIZE, 0, 0),
            strides=(weight_row_bytes, 0, 0, 0),
            aux=(MATRIX_ID_O_PROJ, layer_id, 0, OUT_FEATURES),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_SCALE,
            role=ROLE_Q4_SCALE,
            dtype=DTYPE_U16_Q2_14,
            rank=2,
            flags=ro,
            element_bits=SCALE_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=scale_base_addr,
            nbytes=scale_bytes,
            dims=(OUT_FEATURES, GROUP_COUNT, 0, 0),
            strides=(scale_row_bytes, 2, 0, 0),
            aux=(MATRIX_ID_O_PROJ, layer_id, 0, OUT_FEATURES),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_OUTPUT,
            role=ROLE_OUTPUT,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=wo,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("o_proj_out_q12_12"),
            nbytes=OUT_FEATURES * 4,
            dims=(OUT_FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_O_PROJ, layer_id, 0, OUT_FEATURES),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_EXPECTED,
            role=ROLE_EXPECTED,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=expected_flags,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("expected_o_proj_out_q12_12"),
            nbytes=OUT_FEATURES * 4,
            dims=(OUT_FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_O_PROJ, layer_id, 0, OUT_FEATURES),
        ),
    ]

    image = bytearray(image_bytes)
    image[0:HEADER_BYTES] = pack_header(qmap_base, image_bytes)
    for slot, descriptor in enumerate(descriptors):
        offset = DESCRIPTOR_TABLE_OFFSET + slot * DESCRIPTOR_BYTES
        image[offset : offset + DESCRIPTOR_BYTES] = descriptor
    for item in payloads:
        offset = int(item["offset"])
        payload = item["payload"]
        image[offset : offset + len(payload)] = payload

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(image))
    write_hex_lines(args.sim_hex, image_to_words32(bytes(image)), 32)
    write_hex_lines(args.expected_hex, expected_i32, 32)
    write_scalar(SIM_VECTOR_DIR / f"{prefix}_weight_base_addr.hex", weight_base_addr, 64)
    write_scalar(SIM_VECTOR_DIR / f"{prefix}_scale_base_addr.hex", scale_base_addr, 64)

    manifest = {
        "format_version": 1,
        "name": f"qmap_layer{layer_id}_o_proj_runtime",
        "prefix": prefix,
        "qmap_base": qmap_base,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "layer_id": layer_id,
            "input_size": INPUT_SIZE,
            "out_features": OUT_FEATURES,
            "group_size": GROUP_SIZE,
            "group_count": GROUP_COUNT,
        },
        "memory_layout": {
            "weight_base_addr": weight_base_addr,
            "scale_base_addr": scale_base_addr,
            "weight_row_bytes": weight_row_bytes,
            "scale_row_bytes": scale_row_bytes,
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "activation_addr": addr("attn_out_q12_12"),
            "output_addr": addr("o_proj_out_q12_12"),
            "expected_addr": addr("expected_o_proj_out_q12_12"),
        },
        "expected": {
            "output_words": int(expected_i32.size),
            "min": int(np.min(expected_i32)),
            "max": int(np.max(expected_i32)),
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "expected_hex": relpath(args.expected_hex),
            "weight_words32": relpath(SIM_VECTOR_DIR / f"{prefix}_weight_words32.hex"),
            "scale_words32": relpath(SIM_VECTOR_DIR / f"{prefix}_scale_words32.hex"),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP o_proj runtime packet")
    print("=" * 80)
    print(f"QMAP base:       0x{qmap_base:016X}")
    print(f"Layer:           {layer_id}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Activation words:{activation_i32.size}")
    print(f"Weight bytes:    0x{weight_bytes:X} @ 0x{weight_base_addr:016X}")
    print(f"Scale bytes:     0x{scale_bytes:X} @ 0x{scale_base_addr:016X}")
    print(f"Expected words:  {expected_i32.size}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
