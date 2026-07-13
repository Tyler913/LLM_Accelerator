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

PREFIX = "mlp_silu_mul_stage_real"
GATE_UP_QMAP_PREFIX = "qmap_mlp_gate_up"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "mlp_silu_mul_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "mlp_silu_mul_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_mlp_silu_mul_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_mlp_silu_mul_expected_hidden_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0507_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 16
DESCRIPTOR_COUNT = 6
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0900
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

FEATURES = 3072
IN_WIDTH = 24
IN_FRAC = 12
OUT_WIDTH = 24
OUT_FRAC = 12
SIGMOID_WIDTH = 16
SIGMOID_FRAC = 16
SIGMOID_LUT_INDEX_FRAC = 6
SIGMOID_LUT_MIN_INT = -8
SIGMOID_LUT_MAX_INT = 8
SIGMOID_LUT_SIZE = ((SIGMOID_LUT_MAX_INT - SIGMOID_LUT_MIN_INT) << SIGMOID_LUT_INDEX_FRAC) + 1
LAYER_ID = 0

TENSOR_ID_METADATA = 69
TENSOR_ID_GATE = 70
TENSOR_ID_UP = 71
TENSOR_ID_LUT = 72
TENSOR_ID_HIDDEN = 73
TENSOR_ID_EXPECTED = 74

ROLE_ACTIVATION = 1
ROLE_OUTPUT = 4
ROLE_EXPECTED = 5
ROLE_METADATA = 6
ROLE_PARAMETER = 9

DTYPE_U32 = 5
DTYPE_I32_Q12_12 = 19
DTYPE_U16_Q0_16 = 25

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3
TENSOR_F_DEBUG_ONLY = 1 << 4

MATRIX_ID_GATE_PROJ = 5
MATRIX_ID_UP_PROJ = 6
STAGE_ID_MLP_SILU_MUL = 3
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
    parser = argparse.ArgumentParser(description="Export a QMAP runtime packet for MLP SiLU/multiply simulation.")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--layer-id", type=int, default=LAYER_ID)
    parser.add_argument("--qmap-base", type=parse_int_auto, default=QMAP_BASE)
    parser.add_argument("--gate-up-qmap-prefix", default=GATE_UP_QMAP_PREFIX)
    parser.add_argument("--gate-base-addr", type=parse_int_auto, default=None)
    parser.add_argument("--up-base-addr", type=parse_int_auto, default=None)
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
    gate_up_qmap_prefix = str(args.gate_up_qmap_prefix)

    gate_input = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_gate_input.hex", IN_WIDTH, signed=True)
    up_input = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_up_input.hex", IN_WIDTH, signed=True)
    sigmoid_lut = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_sigmoid_lut.hex", SIGMOID_WIDTH, signed=False)
    expected_hidden = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_expected_hidden_q12_12.hex", OUT_WIDTH, signed=True)

    if gate_input.shape != (FEATURES,):
        raise RuntimeError(f"gate input shape mismatch: expected {(FEATURES,)}, got {gate_input.shape}")
    if up_input.shape != (FEATURES,):
        raise RuntimeError(f"up input shape mismatch: expected {(FEATURES,)}, got {up_input.shape}")
    if sigmoid_lut.shape != (SIGMOID_LUT_SIZE,):
        raise RuntimeError(f"sigmoid LUT shape mismatch: expected {(SIGMOID_LUT_SIZE,)}, got {sigmoid_lut.shape}")
    if expected_hidden.shape != (FEATURES,):
        raise RuntimeError(f"hidden expected shape mismatch: expected {(FEATURES,)}, got {expected_hidden.shape}")

    qmap_gate_path = SIM_VECTOR_DIR / f"{gate_up_qmap_prefix}_expected_gate_words32.hex"
    qmap_up_path = SIM_VECTOR_DIR / f"{gate_up_qmap_prefix}_expected_up_words32.hex"
    if qmap_gate_path.is_file() and qmap_up_path.is_file():
        qmap_gate = read_hex_lines(qmap_gate_path, 32, signed=True)
        qmap_up = read_hex_lines(qmap_up_path, 32, signed=True)
        if not np.array_equal(qmap_gate, gate_input):
            raise RuntimeError("SiLU gate input does not match QMAP gate/up gate write-back vector")
        if not np.array_equal(qmap_up, up_input):
            raise RuntimeError("SiLU up input does not match QMAP gate/up up write-back vector")

    vector_bytes = FEATURES * 4
    lut_bytes = SIGMOID_LUT_SIZE * 4

    gate_i32 = gate_input.astype(np.int32)
    up_i32 = up_input.astype(np.int32)
    expected_i32 = expected_hidden.astype(np.int32)
    lut_u32 = sigmoid_lut.astype(np.uint32)
    metadata_words = np.array(
        [
            layer_id,
            FEATURES,
            SIGMOID_LUT_SIZE,
            IN_WIDTH,
            IN_FRAC,
            OUT_WIDTH,
            OUT_FRAC,
            SIGMOID_WIDTH,
            SIGMOID_FRAC,
            SIGMOID_LUT_INDEX_FRAC,
            SIGMOID_LUT_MIN_INT & 0xFFFF_FFFF,
            SIGMOID_LUT_MAX_INT & 0xFFFF_FFFF,
            STAGE_ID_MLP_SILU_MUL,
        ],
        dtype=np.uint32,
    )

    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET
    cursor = add_payload(payloads, name="metadata", cursor=cursor, payload=as_le_bytes(metadata_words, "<u4"))
    cursor = add_payload(payloads, name="gate_q12_12", cursor=cursor, payload=as_le_bytes(gate_i32, "<i4"))
    cursor = add_payload(payloads, name="up_q12_12", cursor=cursor, payload=as_le_bytes(up_i32, "<i4"))
    cursor = add_payload(payloads, name="sigmoid_lut_uq0_16_words32", cursor=cursor, payload=as_le_bytes(lut_u32, "<u4"))
    cursor = add_payload(payloads, name="hidden_q12_12", cursor=cursor, payload=bytes(vector_bytes))
    cursor = add_payload(payloads, name="expected_hidden_q12_12", cursor=cursor, payload=as_le_bytes(expected_i32, "<i4"))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return qmap_base + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    expected_flags = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY
    stage_aux = (STAGE_ID_MLP_SILU_MUL, layer_id, FEATURES, SIGMOID_LUT_SIZE)

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
            tensor_id=TENSOR_ID_GATE,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=(
                args.gate_base_addr
                if args.gate_base_addr is not None
                else addr("gate_q12_12")
            ),
            nbytes=vector_bytes,
            dims=(FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_GATE_PROJ, layer_id, 0, FEATURES),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_UP,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=(
                args.up_base_addr
                if args.up_base_addr is not None
                else addr("up_q12_12")
            ),
            nbytes=vector_bytes,
            dims=(FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_UP_PROJ, layer_id, 0, FEATURES),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_LUT,
            role=ROLE_PARAMETER,
            dtype=DTYPE_U16_Q0_16,
            rank=1,
            flags=ro,
            element_bits=SIGMOID_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("sigmoid_lut_uq0_16_words32"),
            nbytes=lut_bytes,
            dims=(SIGMOID_LUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=stage_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_HIDDEN,
            role=ROLE_OUTPUT,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=wo,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("hidden_q12_12"),
            nbytes=vector_bytes,
            dims=(FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=stage_aux,
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
            base_addr=addr("expected_hidden_q12_12"),
            nbytes=vector_bytes,
            dims=(FEATURES, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=stage_aux,
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

    focused_meta_path = SIM_VECTOR_DIR / f"{prefix}_meta.json"
    focused_meta: dict[str, Any] = {}
    if focused_meta_path.is_file():
        focused_meta = json.loads(focused_meta_path.read_text(encoding="utf-8"))
        if int(focused_meta.get("layer_id", layer_id)) != layer_id:
            raise RuntimeError("MLP SiLU/multiply vector layer_id does not match requested layer")

    manifest = {
        "format_version": 1,
        "name": f"qmap_layer{layer_id}_mlp_silu_mul_runtime",
        "prefix": prefix,
        "layer_id": layer_id,
        "qmap_base": qmap_base,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "features": FEATURES,
            "sigmoid_lut_size": SIGMOID_LUT_SIZE,
        },
        "memory_layout": {
            "gate_addr": (
                args.gate_base_addr
                if args.gate_base_addr is not None
                else addr("gate_q12_12")
            ),
            "up_addr": (
                args.up_base_addr
                if args.up_base_addr is not None
                else addr("up_q12_12")
            ),
            "sigmoid_lut_addr": addr("sigmoid_lut_uq0_16_words32"),
            "hidden_addr": addr("hidden_q12_12"),
            "expected_hidden_addr": addr("expected_hidden_q12_12"),
            "vector_bytes": vector_bytes,
            "lut_bytes": lut_bytes,
        },
        "expected": {
            "hidden_words": int(expected_i32.size),
            "hidden_min": int(np.min(expected_i32)),
            "hidden_max": int(np.max(expected_i32)),
        },
        "focused_meta": {
            key: focused_meta.get(key)
            for key in ("layer_id", "selected_position", "selected_token_id", "selected_token_text", "source_gate_up_prefix")
            if key in focused_meta
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "expected_hidden_hex": relpath(args.expected_hex),
            "gate_input": relpath(SIM_VECTOR_DIR / f"{prefix}_gate_input.hex"),
            "up_input": relpath(SIM_VECTOR_DIR / f"{prefix}_up_input.hex"),
            "sigmoid_lut": relpath(SIM_VECTOR_DIR / f"{prefix}_sigmoid_lut.hex"),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP MLP SiLU/multiply runtime packet")
    print("=" * 80)
    print(f"Layer:           {layer_id}")
    print(f"QMAP base:       0x{qmap_base:016X}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Gate/up words:   {gate_i32.size} / {up_i32.size}")
    print(f"Sigmoid LUT:     {lut_u32.size} words ({lut_bytes} bytes)")
    print(f"Expected hidden: {expected_i32.size} words")
    print(f"Hidden addr:     0x{addr('hidden_q12_12'):016X}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
