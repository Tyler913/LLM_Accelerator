from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from vector_workspace import resolve_sim_vector_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_VECTOR_DIR = resolve_sim_vector_dir(REPO_ROOT)
QMAP_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_qmap_v1"

PREFIX = "mlp_down_proj_stage_real"
DEFAULT_SILU_QMAP_PREFIX = "qmap_mlp_silu_mul"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "mlp_down_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "mlp_down_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_mlp_down_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_mlp_down_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0508_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 16
DESCRIPTOR_COUNT = 6
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0900
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

MLP_DOWN_WEIGHT_BASE_ADDR = 0x4_0660_0000
MLP_DOWN_SCALE_BASE_ADDR = 0x4_0678_0000

INPUT_SIZE = 3072
OUT_FEATURES = 1024
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
ACT_WIDTH = 24
WEIGHT_WIDTH = 4
SCALE_WIDTH = 16
OUT_WIDTH = 24
LAYER_ID = 0

TENSOR_ID_METADATA = 75
TENSOR_ID_ACTIVATION = 76
TENSOR_ID_WEIGHT = 77
TENSOR_ID_SCALE = 78
TENSOR_ID_OUTPUT = 79
TENSOR_ID_EXPECTED = 80

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

MATRIX_ID_DOWN_PROJ = 7
STAGE_ID_MLP_DOWN = 4
NO_TENSOR_ID = 0xFFFF_FFFF


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def parse_int_auto(value: str) -> int:
    return int(value.replace("_", ""), 0)


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
    parser = argparse.ArgumentParser(description="Export a QMAP runtime packet for MLP down projection simulation.")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--layer-id", type=int, default=LAYER_ID)
    parser.add_argument("--qmap-base", type=parse_int_auto, default=QMAP_BASE)
    parser.add_argument("--weight-base", type=parse_int_auto, default=MLP_DOWN_WEIGHT_BASE_ADDR)
    parser.add_argument("--scale-base", type=parse_int_auto, default=MLP_DOWN_SCALE_BASE_ADDR)
    parser.add_argument("--silu-qmap-prefix", default=DEFAULT_SILU_QMAP_PREFIX)
    parser.add_argument("--activation-base-addr", type=parse_int_auto, default=None)
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
    weight_base = int(args.weight_base)
    scale_base = int(args.scale_base)
    silu_qmap_prefix = str(args.silu_qmap_prefix)

    activation = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_activation.hex", ACT_WIDTH, signed=True)
    expected = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_expected_q12_12.hex", OUT_WIDTH, signed=True)
    weight_words = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_weight_words32.hex", 32, signed=False)
    scale_words = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_scale_words32.hex", 32, signed=False)

    if activation.shape != (INPUT_SIZE,):
        raise RuntimeError(f"activation shape mismatch: expected {(INPUT_SIZE,)}, got {activation.shape}")
    if expected.shape != (OUT_FEATURES,):
        raise RuntimeError(f"expected shape mismatch: expected {(OUT_FEATURES,)}, got {expected.shape}")

    chained_hidden_path = SIM_VECTOR_DIR / f"{silu_qmap_prefix}_expected_hidden_words32.hex"
    if chained_hidden_path.is_file():
        chained_hidden = read_hex_lines(chained_hidden_path, 32, signed=True)
        if chained_hidden.shape != (INPUT_SIZE,):
            raise RuntimeError(f"chained hidden shape mismatch: expected {(INPUT_SIZE,)}, got {chained_hidden.shape}")
        if not np.array_equal(chained_hidden.astype(np.int64), activation.astype(np.int64)):
            mismatch = int(np.nonzero(chained_hidden.astype(np.int64) != activation.astype(np.int64))[0][0])
            raise RuntimeError(f"MLP down activation does not match QMAP SiLU/mul hidden at index {mismatch}")

    activation_bytes = INPUT_SIZE * 4
    output_bytes = OUT_FEATURES * 4
    weight_row_bytes = INPUT_SIZE * WEIGHT_WIDTH // 8
    scale_row_bytes = GROUP_COUNT * SCALE_WIDTH // 8
    weight_bytes = OUT_FEATURES * weight_row_bytes
    scale_bytes = OUT_FEATURES * scale_row_bytes

    if weight_words.size * 4 != weight_bytes:
        raise RuntimeError(f"weight size mismatch: got {weight_words.size * 4}, expected {weight_bytes}")
    if scale_words.size * 4 != scale_bytes:
        raise RuntimeError(f"scale size mismatch: got {scale_words.size * 4}, expected {scale_bytes}")

    activation_i32 = activation.astype(np.int32)
    expected_i32 = expected.astype(np.int32)
    metadata_words = np.array(
        [
            layer_id,
            OUT_FEATURES,
            INPUT_SIZE,
            GROUP_SIZE,
            GROUP_COUNT,
            weight_base & 0xFFFF_FFFF,
            scale_base & 0xFFFF_FFFF,
            MATRIX_ID_DOWN_PROJ,
            STAGE_ID_MLP_DOWN,
            weight_row_bytes,
            scale_row_bytes,
            0,
        ],
        dtype=np.uint32,
    )

    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET
    cursor = add_payload(payloads, name="metadata", cursor=cursor, payload=as_le_bytes(metadata_words, "<u4"))
    cursor = add_payload(payloads, name="mlp_hidden_q12_12", cursor=cursor, payload=as_le_bytes(activation_i32, "<i4"))
    cursor = add_payload(payloads, name="down_out_q12_12", cursor=cursor, payload=bytes(output_bytes))
    cursor = add_payload(payloads, name="expected_down_q12_12", cursor=cursor, payload=as_le_bytes(expected_i32, "<i4"))
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
    stage_aux = (STAGE_ID_MLP_DOWN, layer_id, OUT_FEATURES, INPUT_SIZE)
    matrix_aux = (MATRIX_ID_DOWN_PROJ, layer_id, 0, OUT_FEATURES)

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
            aux=stage_aux,
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
            base_addr=(
                args.activation_base_addr
                if args.activation_base_addr is not None
                else addr("mlp_hidden_q12_12")
            ),
            nbytes=activation_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=stage_aux,
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
            base_addr=weight_base,
            nbytes=weight_bytes,
            dims=(OUT_FEATURES, INPUT_SIZE, 0, 0),
            strides=(weight_row_bytes, 0, 0, 0),
            aux=matrix_aux,
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
            base_addr=scale_base,
            nbytes=scale_bytes,
            dims=(OUT_FEATURES, GROUP_COUNT, 0, 0),
            strides=(scale_row_bytes, 2, 0, 0),
            aux=matrix_aux,
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
            base_addr=addr("down_out_q12_12"),
            nbytes=output_bytes,
            dims=(OUT_FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=matrix_aux,
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
            base_addr=addr("expected_down_q12_12"),
            nbytes=output_bytes,
            dims=(OUT_FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=matrix_aux,
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
    write_scalar(SIM_VECTOR_DIR / f"{prefix}_weight_base_addr.hex", weight_base, 64)
    write_scalar(SIM_VECTOR_DIR / f"{prefix}_scale_base_addr.hex", scale_base, 64)

    focused_meta_path = SIM_VECTOR_DIR / f"{prefix}_meta.json"
    focused_meta: dict[str, Any] = {}
    if focused_meta_path.is_file():
        focused_meta = json.loads(focused_meta_path.read_text(encoding="utf-8"))
        if int(focused_meta.get("layer_id", layer_id)) != layer_id:
            raise RuntimeError("MLP down vector layer_id does not match requested layer")

    manifest = {
        "format_version": 1,
        "name": f"qmap_layer{layer_id}_mlp_down_runtime",
        "prefix": prefix,
        "layer_id": layer_id,
        "qmap_base": qmap_base,
        "source_silu_qmap_prefix": silu_qmap_prefix,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "input_size": INPUT_SIZE,
            "out_features": OUT_FEATURES,
            "group_size": GROUP_SIZE,
            "group_count": GROUP_COUNT,
        },
        "memory_layout": {
            "weight_base_addr": weight_base,
            "scale_base_addr": scale_base,
            "weight_row_bytes": weight_row_bytes,
            "scale_row_bytes": scale_row_bytes,
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "activation_addr": (
                args.activation_base_addr
                if args.activation_base_addr is not None
                else addr("mlp_hidden_q12_12")
            ),
            "output_addr": addr("down_out_q12_12"),
            "expected_addr": addr("expected_down_q12_12"),
        },
        "expected": {
            "words": int(expected_i32.size),
            "min": int(np.min(expected_i32)),
            "max": int(np.max(expected_i32)),
        },
        "focused_meta": {
            key: focused_meta.get(key)
            for key in ("layer_id", "selected_position", "selected_token_id", "selected_token_text", "source_mlp_hidden_prefix")
            if key in focused_meta
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

    print("Exported QMAP MLP down runtime packet")
    print("=" * 80)
    print(f"Layer:           {layer_id}")
    print(f"QMAP base:       0x{qmap_base:016X}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Activation words:{activation_i32.size}")
    print(f"Weight:          0x{weight_bytes:X} @ 0x{weight_base:016X}")
    print(f"Scale:           0x{scale_bytes:X} @ 0x{scale_base:016X}")
    print(f"Expected words:  {expected_i32.size}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
