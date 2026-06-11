import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
QMAP_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_qmap_v1"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

DEFAULT_INPUT = Q4_VECTOR_DIR / "qkv_layer0_last_token_q4.npz"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "q_proj_row0_row1024.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "q_proj_row0_row1024_manifest.json"
DEFAULT_C_HEADER = QMAP_VECTOR_DIR / "q_proj_row0_row1024_qmap.h"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_row1024_image_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_1B20_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 8
DESCRIPTOR_COUNT = 4
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0500
IMAGE_BYTES = 0x1000

ACTIVATION_OFFSET = 0x0500
WEIGHT_OFFSET = 0x0D00
SCALE_OFFSET = 0x0F00
EXPECTED_OFFSET = 0x0F40

HIDDEN_SIZE = 1024
Q4_GROUP_SIZE = 64
Q4_VALUES_PER_BYTE = 2
GROUP_COUNT = HIDDEN_SIZE // Q4_GROUP_SIZE
ACT_FRAC = 12
SCALE_FRAC = 14

TENSOR_ID_ACTIVATION = 1
TENSOR_ID_WEIGHT = 2
TENSOR_ID_SCALE = 3
TENSOR_ID_EXPECTED = 4

ROLE_ACTIVATION = 1
ROLE_Q4_WEIGHT = 2
ROLE_Q4_SCALE = 3
ROLE_EXPECTED = 5

DTYPE_I64 = 8
DTYPE_PACKED_Q4_S4 = 16
DTYPE_U16_Q2_14 = 17
DTYPE_I16_Q4_12 = 18

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_PACKED_Q4_LOW_EVEN = 1 << 1
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_DEBUG_ONLY = 1 << 4

MATRIX_ID_Q_PROJ = 1
NO_TENSOR_ID = 0xFFFF_FFFF


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_addr(value: int) -> str:
    text = f"{value:X}"
    groups = []
    while text:
        groups.append(text[-4:])
        text = text[:-4]
    return "0x" + "_".join(reversed(groups))


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def as_le_bytes(array: np.ndarray, dtype: str) -> bytes:
    converted = np.ascontiguousarray(array.astype(np.dtype(dtype), copy=False))
    return converted.tobytes(order="C")


def require_shape(data: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = data[name]
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {value.shape}")
    return value


def unpack_signed_int4_row(packed_row: np.ndarray, values_per_row: int) -> np.ndarray:
    if packed_row.shape != (values_per_row // Q4_VALUES_PER_BYTE,):
        raise ValueError(
            f"Packed row shape mismatch: expected {(values_per_row // Q4_VALUES_PER_BYTE,)}, "
            f"got {packed_row.shape}"
        )

    low = packed_row & 0x0F
    high = (packed_row >> 4) & 0x0F
    nibbles = np.empty((values_per_row,), dtype=np.uint8)
    nibbles[0::2] = low
    nibbles[1::2] = high
    q = nibbles.astype(np.int16)
    q[q >= 8] -= 16
    return q.astype(np.int8)


def compute_row_sum_q26(
    activation_q4_12: np.ndarray,
    weight_packed_row: np.ndarray,
    scale_q2_14_row: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    weight_q4 = unpack_signed_int4_row(weight_packed_row, HIDDEN_SIZE)
    activation_grouped = activation_q4_12.astype(np.int64).reshape(GROUP_COUNT, Q4_GROUP_SIZE)
    weight_grouped = weight_q4.astype(np.int64).reshape(GROUP_COUNT, Q4_GROUP_SIZE)
    partial_sums = np.sum(activation_grouped * weight_grouped, axis=1, dtype=np.int64)
    scaled_sums = partial_sums * scale_q2_14_row.astype(np.int64)
    row_sum_q26 = int(np.sum(scaled_sums, dtype=np.int64))
    return row_sum_q26, partial_sums, scaled_sums


def pack_header(qmap_base: int) -> bytes:
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
        IMAGE_BYTES,
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
        raise RuntimeError(f"Internal descriptor size error: {len(descriptor)}")
    return descriptor


def parse_header(image: bytes) -> dict[str, int]:
    fields = struct.unpack_from("<IIIIIIQQQQII", image, 0)
    names = [
        "magic",
        "version",
        "header_bytes",
        "descriptor_bytes",
        "descriptor_count",
        "descriptor_capacity",
        "descriptor_table_addr",
        "payload_base_addr",
        "image_base_addr",
        "image_bytes",
        "flags",
        "checksum32",
    ]
    return dict(zip(names, fields, strict=True))


def parse_descriptor(image: bytes, slot: int) -> dict[str, Any]:
    offset = DESCRIPTOR_TABLE_OFFSET + slot * DESCRIPTOR_BYTES
    fields = struct.unpack_from("<8I2Q4I4Q8I", image, offset)
    return {
        "tensor_id": fields[0],
        "role": fields[1],
        "dtype": fields[2],
        "rank": fields[3],
        "flags": fields[4],
        "element_bits": fields[5],
        "group_size": fields[6],
        "scale_tensor_id": fields[7],
        "base_addr": fields[8],
        "nbytes": fields[9],
        "dims": list(fields[10:14]),
        "strides_bytes": list(fields[14:18]),
        "aux": list(fields[18:22]),
        "checksum32": fields[22],
        "reserved": list(fields[23:26]),
    }


def build_qmap_image(source_npz: Path, qmap_base: int) -> tuple[bytes, dict[str, Any]]:
    if not source_npz.is_file():
        raise FileNotFoundError(
            f"Missing {source_npz}. Run 13_export_q4_gemv_vectors.py first."
        )

    data = np.load(source_npz)
    activation = require_shape(data, "input_norm_q4_12", (HIDDEN_SIZE,))
    weight_packed = require_shape(data, "q_weight_q4_packed", (2048, HIDDEN_SIZE // 2))[0]
    scale_q2_14 = require_shape(data, "q_scale_q2_14", (2048, GROUP_COUNT))[0]
    row_sum_q26, partial_sums, scaled_sums = compute_row_sum_q26(
        activation,
        weight_packed,
        scale_q2_14,
    )

    expected_float32 = np.float32(row_sum_q26 / float(1 << (ACT_FRAC + SCALE_FRAC)))
    actual_q = require_shape(data, "actual_q_q4", (2048,))
    if not np.isclose(actual_q[0], expected_float32, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            f"Internal row sum mismatch: computed {expected_float32}, "
            f"source actual_q_q4[0] {actual_q[0]}"
        )

    payloads = {
        "activation": (ACTIVATION_OFFSET, as_le_bytes(activation, "<i2")),
        "weight": (WEIGHT_OFFSET, as_le_bytes(weight_packed, "<u1")),
        "scale": (SCALE_OFFSET, as_le_bytes(scale_q2_14, "<u2")),
        "expected": (EXPECTED_OFFSET, struct.pack("<q", row_sum_q26)),
    }

    image = bytearray(IMAGE_BYTES)
    image[0:HEADER_BYTES] = pack_header(qmap_base)

    descriptor_specs = [
        {
            "name": "activation_q4_12",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_ACTIVATION,
                role=ROLE_ACTIVATION,
                dtype=DTYPE_I16_Q4_12,
                rank=1,
                flags=TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY,
                element_bits=16,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=qmap_base + ACTIVATION_OFFSET,
                nbytes=len(payloads["activation"][1]),
                dims=(HIDDEN_SIZE, 0, 0, 0),
                strides=(2, 0, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, 0, 0, 0),
            ),
        },
        {
            "name": "weight_q4_packed",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_WEIGHT,
                role=ROLE_Q4_WEIGHT,
                dtype=DTYPE_PACKED_Q4_S4,
                rank=2,
                flags=(
                    TENSOR_F_ROW_MAJOR
                    | TENSOR_F_PACKED_Q4_LOW_EVEN
                    | TENSOR_F_READ_ONLY
                ),
                element_bits=4,
                group_size=Q4_GROUP_SIZE,
                scale_tensor_id=TENSOR_ID_SCALE,
                base_addr=qmap_base + WEIGHT_OFFSET,
                nbytes=len(payloads["weight"][1]),
                dims=(1, HIDDEN_SIZE, 0, 0),
                strides=(HIDDEN_SIZE // 2, 0, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, 0, 0, 0),
            ),
        },
        {
            "name": "scale_q2_14",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_SCALE,
                role=ROLE_Q4_SCALE,
                dtype=DTYPE_U16_Q2_14,
                rank=2,
                flags=TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY,
                element_bits=16,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=qmap_base + SCALE_OFFSET,
                nbytes=len(payloads["scale"][1]),
                dims=(1, GROUP_COUNT, 0, 0),
                strides=(GROUP_COUNT * 2, 2, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, 0, 0, 0),
            ),
        },
        {
            "name": "expected_row_sum_q26",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_EXPECTED,
                role=ROLE_EXPECTED,
                dtype=DTYPE_I64,
                rank=1,
                flags=TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY,
                element_bits=64,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=qmap_base + EXPECTED_OFFSET,
                nbytes=len(payloads["expected"][1]),
                dims=(1, 0, 0, 0),
                strides=(8, 0, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, 0, 0, 0),
            ),
        },
    ]

    for slot, spec in enumerate(descriptor_specs):
        start = DESCRIPTOR_TABLE_OFFSET + slot * DESCRIPTOR_BYTES
        image[start : start + DESCRIPTOR_BYTES] = spec["descriptor"]

    for _name, (offset, payload) in payloads.items():
        image[offset : offset + len(payload)] = payload

    image_bytes = bytes(image)
    verify_image(image_bytes, payloads, qmap_base)

    descriptors = [parse_descriptor(image_bytes, slot) for slot in range(DESCRIPTOR_COUNT)]
    manifest = {
        "format": "qmap_v1",
        "name": "qwen3_0p6b_qmap_v1_row1024",
        "source_npz": relpath(source_npz),
        "image_base_addr": format_addr(qmap_base),
        "descriptor_table_addr": format_addr(qmap_base + DESCRIPTOR_TABLE_OFFSET),
        "payload_base_addr": format_addr(qmap_base + PAYLOAD_BASE_OFFSET),
        "image_bytes": IMAGE_BYTES,
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "row1024": {
            "matrix_name": "q_proj",
            "layer_index": 0,
            "row_index": 0,
            "group_count": GROUP_COUNT,
            "group_size": Q4_GROUP_SIZE,
            "row_sum_q26_int64": row_sum_q26,
            "expected_float32": float(expected_float32),
            "partial_sum_int64": partial_sums.astype(np.int64).tolist(),
            "scaled_sum_q26_int64": scaled_sums.astype(np.int64).tolist(),
        },
        "descriptors": descriptors,
        "payload_offsets": {
            name: {
                "offset": f"0x{offset:04X}",
                "address": format_addr(qmap_base + offset),
                "bytes": len(payload),
            }
            for name, (offset, payload) in payloads.items()
        },
    }
    return image_bytes, manifest


def verify_image(
    image: bytes,
    payloads: dict[str, tuple[int, bytes]],
    qmap_base: int,
) -> None:
    if len(image) != IMAGE_BYTES:
        raise RuntimeError(f"Image size mismatch: expected {IMAGE_BYTES}, got {len(image)}")

    header = parse_header(image)
    expected_header = {
        "magic": QMAP_MAGIC,
        "version": QMAP_VERSION,
        "header_bytes": HEADER_BYTES,
        "descriptor_bytes": DESCRIPTOR_BYTES,
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "descriptor_table_addr": qmap_base + DESCRIPTOR_TABLE_OFFSET,
        "payload_base_addr": qmap_base + PAYLOAD_BASE_OFFSET,
        "image_base_addr": qmap_base,
        "image_bytes": IMAGE_BYTES,
        "flags": 0,
        "checksum32": 0,
    }
    if header != expected_header:
        raise RuntimeError(f"Header verification failed: {header}")

    for name, (offset, payload) in payloads.items():
        actual = image[offset : offset + len(payload)]
        if actual != payload:
            raise RuntimeError(f"Payload verification failed for {name}")


def write_word32_hex(image: bytes, output: Path) -> None:
    if len(image) % 4 != 0:
        raise ValueError("Image length must be 32-bit aligned")
    output.parent.mkdir(parents=True, exist_ok=True)
    words = struct.unpack(f"<{len(image) // 4}I", image)
    output.write_text("".join(f"{word:08x}\n" for word in words), encoding="utf-8")


def write_outputs(
    image: bytes,
    manifest: dict[str, Any],
    output: Path,
    manifest_path: Path,
    sim_hex: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image)
    write_word32_hex(image, sim_hex)
    manifest["files"] = [
        {
            "file": relpath(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
        },
        {
            "file": relpath(sim_hex),
            "sha256": sha256_file(sim_hex),
            "words32": len(image) // 4,
        },
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def format_c_array(image: bytes, values_per_line: int = 12) -> str:
    lines = []
    for offset in range(0, len(image), values_per_line):
        chunk = image[offset : offset + values_per_line]
        values = ", ".join(f"0x{value:02X}" for value in chunk)
        lines.append(f"    {values},")
    return "\n".join(lines)


def write_c_header(image: bytes, output: Path, qmap_base: int, image_sha256: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""#ifndef QMAP_ROW1024_IMAGE_H
#define QMAP_ROW1024_IMAGE_H

#include "xil_types.h"

#define QMAP_ROW1024_BASE_ADDR 0x{qmap_base:016X}ULL
#define QMAP_ROW1024_IMAGE_SIZE {len(image)}U
#define QMAP_ROW1024_IMAGE_SHA256 "{image_sha256}"

static const u8 qmap_row1024_image[QMAP_ROW1024_IMAGE_SIZE] = {{
{format_c_array(image)}
}};

#endif
"""
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a QMAP v1 binary image for Layer 0 q_proj row 0 row1024."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Q4 NPZ")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output QMAP image")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Output manifest JSON",
    )
    parser.add_argument(
        "--c-header",
        type=Path,
        default=DEFAULT_C_HEADER,
        help="Output C header with the QMAP image embedded as a u8 array",
    )
    parser.add_argument(
        "--sim-hex",
        type=Path,
        default=DEFAULT_SIM_HEX,
        help="Output little-endian 32-bit word hex for RTL simulation",
    )
    parser.add_argument(
        "--qmap-base",
        type=lambda text: int(text.replace("_", ""), 0),
        default=QMAP_BASE,
        help="Physical QMAP base address, default 0x4_1B20_0000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, manifest = build_qmap_image(args.input, args.qmap_base)
    write_outputs(image, manifest, args.output, args.manifest, args.sim_hex)
    image_sha256 = sha256_file(args.output)
    write_c_header(image, args.c_header, args.qmap_base, image_sha256)

    print("Exported QMAP v1 row1024 image")
    print("=" * 80)
    print(f"Input: {relpath(args.input)}")
    print(f"Output: {relpath(args.output)}")
    print(f"Manifest: {relpath(args.manifest)}")
    print(f"C header: {relpath(args.c_header)}")
    print(f"Sim hex: {relpath(args.sim_hex)}")
    print(f"Image base: {format_addr(args.qmap_base)}")
    print(f"Image bytes: 0x{len(image):04X}")
    print(f"Descriptor count/capacity: {DESCRIPTOR_COUNT}/{DESCRIPTOR_CAPACITY}")
    print(f"row_sum_q26_int64: {manifest['row1024']['row_sum_q26_int64']}")
    print(f"expected_float32: {manifest['row1024']['expected_float32']}")
    print(f"SHA256: {image_sha256}")
    print()
    print("Payloads:")
    for name, entry in manifest["payload_offsets"].items():
        print(f"  {name}: {entry['address']} +{entry['offset']} bytes={entry['bytes']}")


if __name__ == "__main__":
    main()
