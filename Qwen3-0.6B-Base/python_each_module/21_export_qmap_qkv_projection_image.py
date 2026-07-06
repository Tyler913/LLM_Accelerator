import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
QMAP_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_qmap_v1"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"

DEFAULT_INPUT = Q4_VECTOR_DIR / "qkv_layer0_last_token_q4.npz"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "layer0_qkv_projection.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "layer0_qkv_projection_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_qkv_projection_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_qkv_projection_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0008_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 32
DESCRIPTOR_COUNT = 12
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x1100
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

HIDDEN_SIZE = 1024
Q_ROWS_FULL = 2048
KV_ROWS_FULL = 1024
Q4_GROUP_SIZE = 64
Q4_VALUES_PER_BYTE = 2
GROUP_COUNT = HIDDEN_SIZE // Q4_GROUP_SIZE
ACT_FRAC = 12
SCALE_FRAC = 14
Q26_TO_Q12_SHIFT = SCALE_FRAC
Q12_12_MIN = -(1 << 23)
Q12_12_MAX = (1 << 23) - 1

TENSOR_ID_METADATA = 1
TENSOR_ID_ACTIVATION = 2
TENSOR_ID_Q_WEIGHT = 3
TENSOR_ID_Q_SCALE = 4
TENSOR_ID_K_WEIGHT = 5
TENSOR_ID_K_SCALE = 6
TENSOR_ID_V_WEIGHT = 7
TENSOR_ID_V_SCALE = 8
TENSOR_ID_Q_OUT = 9
TENSOR_ID_K_OUT = 10
TENSOR_ID_V_OUT = 11
TENSOR_ID_EXPECTED = 12

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

MATRIX_ID_Q_PROJ = 1
MATRIX_ID_K_PROJ = 2
MATRIX_ID_V_PROJ = 3
MATRIX_ID_QKV_EXPECTED = 0x51564B
NO_TENSOR_ID = 0xFFFF_FFFF


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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


def read_i32_hex(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing activation override hex: {path}")

    values: list[int] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        raw = int(text.replace("_", ""), 16)
        if raw < 0 or raw > 0xFFFF_FFFF:
            raise ValueError(f"{path}:{lineno}: word32 hex out of range: {text}")
        if raw & 0x8000_0000:
            raw -= 0x1_0000_0000
        values.append(raw)

    return np.asarray(values, dtype=np.int32)


def unpack_signed_int4_matrix(packed: np.ndarray, values_per_row: int) -> np.ndarray:
    if packed.shape[1] != values_per_row // Q4_VALUES_PER_BYTE:
        raise ValueError(
            f"Packed row width mismatch: expected {values_per_row // Q4_VALUES_PER_BYTE}, "
            f"got {packed.shape[1]}"
        )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((packed.shape[0], values_per_row), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    q = nibbles.astype(np.int16)
    q[q >= 8] -= 16
    return q.astype(np.int8)


def q26_to_q12_12_words(row_sum_q26: np.ndarray) -> np.ndarray:
    shifted = row_sum_q26.astype(np.int64) >> Q26_TO_Q12_SHIFT
    clipped = np.clip(shifted, Q12_12_MIN, Q12_12_MAX)
    return clipped.astype(np.int32)


def compute_matrix_outputs(
    *,
    activation_q12_12: np.ndarray,
    packed_weight: np.ndarray,
    scale_q2_14: np.ndarray,
    actual_q4_float: Optional[np.ndarray],
    matrix_name: str,
    row_count: int,
) -> dict[str, Any]:
    if row_count < 1:
        raise ValueError(f"{matrix_name} row_count must be positive")
    if row_count > packed_weight.shape[0]:
        raise ValueError(
            f"{matrix_name} row_count {row_count} exceeds available rows {packed_weight.shape[0]}"
        )

    packed_rows = np.ascontiguousarray(packed_weight[:row_count])
    scale_rows = np.ascontiguousarray(scale_q2_14[:row_count])
    weight_q4 = unpack_signed_int4_matrix(packed_rows, HIDDEN_SIZE)
    activation_grouped = activation_q12_12.astype(np.int64).reshape(GROUP_COUNT, Q4_GROUP_SIZE)
    weight_grouped = weight_q4.astype(np.int64).reshape(row_count, GROUP_COUNT, Q4_GROUP_SIZE)
    partial_sums = np.sum(
        weight_grouped * activation_grouped[None, :, :],
        axis=2,
        dtype=np.int64,
    )
    scaled_sums = partial_sums * scale_rows.astype(np.int64)
    row_sum_q26 = np.sum(scaled_sums, axis=1, dtype=np.int64)
    q12_12 = q26_to_q12_12_words(row_sum_q26)

    recomputed_float32 = (row_sum_q26.astype(np.float64) / float(1 << (ACT_FRAC + SCALE_FRAC))).astype(
        np.float32
    )
    q12_float32 = (q12_12.astype(np.float64) / float(1 << ACT_FRAC)).astype(np.float32)
    if actual_q4_float is None:
        max_recompute_diff = None
        max_q12_diff = None
    else:
        source_float32 = actual_q4_float[:row_count].astype(np.float32)
        max_recompute_diff = float(np.max(np.abs(recomputed_float32 - source_float32)))
        if max_recompute_diff != 0.0:
            raise RuntimeError(
                f"{matrix_name} Q26 recompute mismatch against artifact: {max_recompute_diff}"
            )
        max_q12_diff = float(np.max(np.abs(q12_float32 - source_float32)))

    return {
        "packed_rows": packed_rows,
        "scale_rows": scale_rows,
        "row_sum_q26": row_sum_q26.astype(np.int64),
        "q12_12": q12_12,
        "max_recompute_diff": max_recompute_diff,
        "max_q12_12_diff": max_q12_diff,
        "first_row_sum_q26": int(row_sum_q26[0]),
        "first_q12_12_word": int(q12_12[0]),
        "first_q12_12_float": float(q12_float32[0]),
    }


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


def add_payload(
    payloads: list[dict[str, Any]],
    *,
    name: str,
    offset_cursor: int,
    payload: bytes,
) -> int:
    offset = align_up(offset_cursor, PAYLOAD_ALIGNMENT)
    payloads.append({"name": name, "offset": offset, "payload": payload})
    return offset + len(payload)


def build_payload_layout(
    *,
    metadata_words: np.ndarray,
    activation_q12_12: np.ndarray,
    q_outputs: dict[str, Any],
    k_outputs: dict[str, Any],
    v_outputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET

    cursor = add_payload(
        payloads,
        name="metadata",
        offset_cursor=cursor,
        payload=as_le_bytes(metadata_words, "<u4"),
    )
    cursor = add_payload(
        payloads,
        name="activation_q12_12",
        offset_cursor=cursor,
        payload=as_le_bytes(activation_q12_12, "<i4"),
    )
    cursor = add_payload(
        payloads,
        name="q_weight_q4_packed",
        offset_cursor=cursor,
        payload=as_le_bytes(q_outputs["packed_rows"], "<u1"),
    )
    cursor = add_payload(
        payloads,
        name="q_scale_q2_14",
        offset_cursor=cursor,
        payload=as_le_bytes(q_outputs["scale_rows"], "<u2"),
    )
    cursor = add_payload(
        payloads,
        name="k_weight_q4_packed",
        offset_cursor=cursor,
        payload=as_le_bytes(k_outputs["packed_rows"], "<u1"),
    )
    cursor = add_payload(
        payloads,
        name="k_scale_q2_14",
        offset_cursor=cursor,
        payload=as_le_bytes(k_outputs["scale_rows"], "<u2"),
    )
    cursor = add_payload(
        payloads,
        name="v_weight_q4_packed",
        offset_cursor=cursor,
        payload=as_le_bytes(v_outputs["packed_rows"], "<u1"),
    )
    cursor = add_payload(
        payloads,
        name="v_scale_q2_14",
        offset_cursor=cursor,
        payload=as_le_bytes(v_outputs["scale_rows"], "<u2"),
    )
    cursor = add_payload(
        payloads,
        name="q_out",
        offset_cursor=cursor,
        payload=bytes(q_outputs["q12_12"].shape[0] * 4),
    )
    cursor = add_payload(
        payloads,
        name="k_out",
        offset_cursor=cursor,
        payload=bytes(k_outputs["q12_12"].shape[0] * 4),
    )
    cursor = add_payload(
        payloads,
        name="v_out",
        offset_cursor=cursor,
        payload=bytes(v_outputs["q12_12"].shape[0] * 4),
    )
    expected_qkv = np.concatenate(
        [q_outputs["q12_12"], k_outputs["q12_12"], v_outputs["q12_12"]]
    ).astype(np.int32)
    cursor = add_payload(
        payloads,
        name="expected_qkv_q12_12",
        offset_cursor=cursor,
        payload=as_le_bytes(expected_qkv, "<i4"),
    )
    return payloads, align_up(cursor, IMAGE_ALIGNMENT)


def payload_map(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in payloads}


def descriptor_specs(
    *,
    qmap_base: int,
    layer_id: int,
    payloads: dict[str, dict[str, Any]],
    q_rows: int,
    k_rows: int,
    v_rows: int,
    metadata_words: int,
) -> list[dict[str, Any]]:
    def addr(name: str) -> int:
        return qmap_base + int(payloads[name]["offset"])

    def size(name: str) -> int:
        return len(payloads[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    packed_ro = ro | TENSOR_F_PACKED_Q4_LOW_EVEN
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    expected_flags = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY

    return [
        {
            "name": "metadata",
            "descriptor": pack_descriptor(
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
                dims=(metadata_words, 0, 0, 0),
                strides=(4, 0, 0, 0),
                aux=(MATRIX_ID_QKV_EXPECTED, layer_id, 0, 0),
            ),
        },
        {
            "name": "activation_q12_12",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_ACTIVATION,
                role=ROLE_ACTIVATION,
                dtype=DTYPE_I32_Q12_12,
                rank=1,
                flags=ro,
                element_bits=32,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("activation_q12_12"),
                nbytes=size("activation_q12_12"),
                dims=(HIDDEN_SIZE, 0, 0, 0),
                strides=(4, 0, 0, 0),
                aux=(MATRIX_ID_QKV_EXPECTED, layer_id, 0, 0),
            ),
        },
        {
            "name": "q_weight_q4_packed",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_Q_WEIGHT,
                role=ROLE_Q4_WEIGHT,
                dtype=DTYPE_PACKED_Q4_S4,
                rank=2,
                flags=packed_ro,
                element_bits=4,
                group_size=Q4_GROUP_SIZE,
                scale_tensor_id=TENSOR_ID_Q_SCALE,
                base_addr=addr("q_weight_q4_packed"),
                nbytes=size("q_weight_q4_packed"),
                dims=(q_rows, HIDDEN_SIZE, 0, 0),
                strides=(HIDDEN_SIZE // 2, 0, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "q_scale_q2_14",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_Q_SCALE,
                role=ROLE_Q4_SCALE,
                dtype=DTYPE_U16_Q2_14,
                rank=2,
                flags=ro,
                element_bits=16,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("q_scale_q2_14"),
                nbytes=size("q_scale_q2_14"),
                dims=(q_rows, GROUP_COUNT, 0, 0),
                strides=(GROUP_COUNT * 2, 2, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "k_weight_q4_packed",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_K_WEIGHT,
                role=ROLE_Q4_WEIGHT,
                dtype=DTYPE_PACKED_Q4_S4,
                rank=2,
                flags=packed_ro,
                element_bits=4,
                group_size=Q4_GROUP_SIZE,
                scale_tensor_id=TENSOR_ID_K_SCALE,
                base_addr=addr("k_weight_q4_packed"),
                nbytes=size("k_weight_q4_packed"),
                dims=(k_rows, HIDDEN_SIZE, 0, 0),
                strides=(HIDDEN_SIZE // 2, 0, 0, 0),
                aux=(MATRIX_ID_K_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "k_scale_q2_14",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_K_SCALE,
                role=ROLE_Q4_SCALE,
                dtype=DTYPE_U16_Q2_14,
                rank=2,
                flags=ro,
                element_bits=16,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("k_scale_q2_14"),
                nbytes=size("k_scale_q2_14"),
                dims=(k_rows, GROUP_COUNT, 0, 0),
                strides=(GROUP_COUNT * 2, 2, 0, 0),
                aux=(MATRIX_ID_K_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "v_weight_q4_packed",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_V_WEIGHT,
                role=ROLE_Q4_WEIGHT,
                dtype=DTYPE_PACKED_Q4_S4,
                rank=2,
                flags=packed_ro,
                element_bits=4,
                group_size=Q4_GROUP_SIZE,
                scale_tensor_id=TENSOR_ID_V_SCALE,
                base_addr=addr("v_weight_q4_packed"),
                nbytes=size("v_weight_q4_packed"),
                dims=(v_rows, HIDDEN_SIZE, 0, 0),
                strides=(HIDDEN_SIZE // 2, 0, 0, 0),
                aux=(MATRIX_ID_V_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "v_scale_q2_14",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_V_SCALE,
                role=ROLE_Q4_SCALE,
                dtype=DTYPE_U16_Q2_14,
                rank=2,
                flags=ro,
                element_bits=16,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("v_scale_q2_14"),
                nbytes=size("v_scale_q2_14"),
                dims=(v_rows, GROUP_COUNT, 0, 0),
                strides=(GROUP_COUNT * 2, 2, 0, 0),
                aux=(MATRIX_ID_V_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "q_out",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_Q_OUT,
                role=ROLE_OUTPUT,
                dtype=DTYPE_I32_Q12_12,
                rank=1,
                flags=wo,
                element_bits=32,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("q_out"),
                nbytes=size("q_out"),
                dims=(q_rows, 0, 0, 0),
                strides=(4, 0, 0, 0),
                aux=(MATRIX_ID_Q_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "k_out",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_K_OUT,
                role=ROLE_OUTPUT,
                dtype=DTYPE_I32_Q12_12,
                rank=1,
                flags=wo,
                element_bits=32,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("k_out"),
                nbytes=size("k_out"),
                dims=(k_rows, 0, 0, 0),
                strides=(4, 0, 0, 0),
                aux=(MATRIX_ID_K_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "v_out",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_V_OUT,
                role=ROLE_OUTPUT,
                dtype=DTYPE_I32_Q12_12,
                rank=1,
                flags=wo,
                element_bits=32,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("v_out"),
                nbytes=size("v_out"),
                dims=(v_rows, 0, 0, 0),
                strides=(4, 0, 0, 0),
                aux=(MATRIX_ID_V_PROJ, layer_id, 0, 0),
            ),
        },
        {
            "name": "expected_qkv_q12_12",
            "descriptor": pack_descriptor(
                tensor_id=TENSOR_ID_EXPECTED,
                role=ROLE_EXPECTED,
                dtype=DTYPE_I32_Q12_12,
                rank=1,
                flags=expected_flags,
                element_bits=32,
                group_size=0,
                scale_tensor_id=NO_TENSOR_ID,
                base_addr=addr("expected_qkv_q12_12"),
                nbytes=size("expected_qkv_q12_12"),
                dims=(q_rows + k_rows + v_rows, 0, 0, 0),
                strides=(4, 0, 0, 0),
                aux=(MATRIX_ID_QKV_EXPECTED, layer_id, 0, 0),
            ),
        },
    ]


def build_qmap_image(
    *,
    source_npz: Path,
    qmap_base: int,
    layer_id: int,
    q_rows: int,
    k_rows: int,
    v_rows: int,
    activation_hex: Optional[Path] = None,
    activation_source_name: Optional[str] = None,
) -> tuple[bytes, dict[str, Any], np.ndarray]:
    if not source_npz.is_file():
        raise FileNotFoundError(
            f"Missing {source_npz}. Run 13_export_q4_gemv_vectors.py first."
        )
    if layer_id < 0 or layer_id > 0xFFFF_FFFF:
        raise ValueError(f"layer_id must fit uint32, got {layer_id}")

    data = np.load(source_npz)
    source_activation_q4_12 = require_shape(data, "input_norm_q4_12", (HIDDEN_SIZE,)).astype(np.int32)
    if activation_hex is None:
        activation_q4_12 = source_activation_q4_12
        activation_mode = "source_npz"
        activation_hex_relpath = None
        activation_override_max_abs_diff_vs_npz = 0
    else:
        activation_q4_12 = read_i32_hex(activation_hex)
        if activation_q4_12.shape != (HIDDEN_SIZE,):
            raise ValueError(
                f"activation override shape mismatch: expected {(HIDDEN_SIZE,)}, "
                f"got {activation_q4_12.shape}"
            )
        activation_mode = "hex_override"
        activation_hex_relpath = relpath(activation_hex)
        activation_override_max_abs_diff_vs_npz = int(
            np.max(np.abs(activation_q4_12.astype(np.int64) - source_activation_q4_12.astype(np.int64)))
        )
    input_norm_fp32 = require_shape(data, "input_norm_fp32", (HIDDEN_SIZE,))
    q_weight = require_shape(data, "q_weight_q4_packed", (Q_ROWS_FULL, HIDDEN_SIZE // 2))
    q_scale = require_shape(data, "q_scale_q2_14", (Q_ROWS_FULL, GROUP_COUNT))
    k_weight = require_shape(data, "k_weight_q4_packed", (KV_ROWS_FULL, HIDDEN_SIZE // 2))
    k_scale = require_shape(data, "k_scale_q2_14", (KV_ROWS_FULL, GROUP_COUNT))
    v_weight = require_shape(data, "v_weight_q4_packed", (KV_ROWS_FULL, HIDDEN_SIZE // 2))
    v_scale = require_shape(data, "v_scale_q2_14", (KV_ROWS_FULL, GROUP_COUNT))

    q_outputs = compute_matrix_outputs(
        activation_q12_12=activation_q4_12,
        packed_weight=q_weight,
        scale_q2_14=q_scale,
        actual_q4_float=None
        if activation_hex is not None
        else require_shape(data, "actual_q_q4", (Q_ROWS_FULL,)),
        matrix_name="q_proj",
        row_count=q_rows,
    )
    k_outputs = compute_matrix_outputs(
        activation_q12_12=activation_q4_12,
        packed_weight=k_weight,
        scale_q2_14=k_scale,
        actual_q4_float=None
        if activation_hex is not None
        else require_shape(data, "actual_k_q4", (KV_ROWS_FULL,)),
        matrix_name="k_proj",
        row_count=k_rows,
    )
    v_outputs = compute_matrix_outputs(
        activation_q12_12=activation_q4_12,
        packed_weight=v_weight,
        scale_q2_14=v_scale,
        actual_q4_float=None
        if activation_hex is not None
        else require_shape(data, "actual_v_q4", (KV_ROWS_FULL,)),
        matrix_name="v_proj",
        row_count=v_rows,
    )

    activation_round = np.round(input_norm_fp32.astype(np.float64) * float(1 << ACT_FRAC)).astype(
        np.int32
    )
    input_requant_max_abs_diff = (
        None
        if activation_hex is not None
        else int(np.max(np.abs(activation_round - activation_q4_12)))
    )

    metadata_words = np.array(
        [
            layer_id,
            int(np.asarray(data["prompt_position"]).item()),
            int(np.asarray(data["token_id"]).item()),
            q_rows,
            k_rows,
            v_rows,
            HIDDEN_SIZE,
            Q4_GROUP_SIZE,
            GROUP_COUNT,
            ACT_FRAC,
            SCALE_FRAC,
            Q26_TO_Q12_SHIFT,
            TENSOR_ID_ACTIVATION,
            TENSOR_ID_Q_OUT,
            TENSOR_ID_K_OUT,
            TENSOR_ID_V_OUT,
        ],
        dtype=np.uint32,
    )

    payloads, image_bytes = build_payload_layout(
        metadata_words=metadata_words,
        activation_q12_12=activation_q4_12,
        q_outputs=q_outputs,
        k_outputs=k_outputs,
        v_outputs=v_outputs,
    )
    payloads_by_name = payload_map(payloads)
    descriptors = descriptor_specs(
        qmap_base=qmap_base,
        layer_id=layer_id,
        payloads=payloads_by_name,
        q_rows=q_rows,
        k_rows=k_rows,
        v_rows=v_rows,
        metadata_words=metadata_words.shape[0],
    )

    image = bytearray(image_bytes)
    image[0:HEADER_BYTES] = pack_header(qmap_base, image_bytes)
    for slot, spec in enumerate(descriptors):
        start = DESCRIPTOR_TABLE_OFFSET + slot * DESCRIPTOR_BYTES
        image[start : start + DESCRIPTOR_BYTES] = spec["descriptor"]
    for item in payloads:
        offset = int(item["offset"])
        payload = item["payload"]
        image[offset : offset + len(payload)] = payload

    image_bytes_blob = bytes(image)
    verify_image(
        image=image_bytes_blob,
        payloads=payloads,
        qmap_base=qmap_base,
        image_bytes=image_bytes,
    )

    expected_qkv = np.concatenate(
        [q_outputs["q12_12"], k_outputs["q12_12"], v_outputs["q12_12"]]
    ).astype(np.int32)
    parsed_descriptors = [parse_descriptor(image_bytes_blob, slot) for slot in range(DESCRIPTOR_COUNT)]
    manifest = {
        "format": "qmap_v1",
        "name": f"qwen3_0p6b_layer{layer_id}_qkv_projection_packet",
        "source_npz": relpath(source_npz),
        "activation_source": {
            "mode": activation_mode,
            "source_name": activation_source_name,
            "source_hex": activation_hex_relpath,
            "source_npz_tensor": "input_norm_q4_12",
            "override_max_abs_diff_vs_npz": activation_override_max_abs_diff_vs_npz,
        },
        "layer_id": layer_id,
        "image_base_addr": format_addr(qmap_base),
        "descriptor_table_addr": format_addr(qmap_base + DESCRIPTOR_TABLE_OFFSET),
        "payload_base_addr": format_addr(qmap_base + PAYLOAD_BASE_OFFSET),
        "image_bytes": image_bytes,
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "row_counts": {
            "q_rows": q_rows,
            "k_rows": k_rows,
            "v_rows": v_rows,
            "total_rows": q_rows + k_rows + v_rows,
        },
        "fixed_point": {
            "activation_dtype": "I32_Q12_12",
            "scale_dtype": "U16_Q2_14",
            "accumulator_dtype": "signed_Q26_integer",
            "output_dtype": "I32_Q12_12",
            "q26_to_q12_12_shift": Q26_TO_Q12_SHIFT,
            "q12_12_saturate_min": Q12_12_MIN,
            "q12_12_saturate_max": Q12_12_MAX,
            "input_requant_max_abs_diff": input_requant_max_abs_diff,
        },
        "proof": {
            "q_proj_max_recompute_diff_vs_artifact": q_outputs["max_recompute_diff"],
            "k_proj_max_recompute_diff_vs_artifact": k_outputs["max_recompute_diff"],
            "v_proj_max_recompute_diff_vs_artifact": v_outputs["max_recompute_diff"],
            "q_proj_max_q12_12_diff_vs_q4_float": q_outputs["max_q12_12_diff"],
            "k_proj_max_q12_12_diff_vs_q4_float": k_outputs["max_q12_12_diff"],
            "v_proj_max_q12_12_diff_vs_q4_float": v_outputs["max_q12_12_diff"],
        },
        "sample_outputs": {
            "q_proj": {
                "row0_q26": q_outputs["first_row_sum_q26"],
                "row0_q12_12_word": q_outputs["first_q12_12_word"],
                "row0_q12_12_float": q_outputs["first_q12_12_float"],
            },
            "k_proj": {
                "row0_q26": k_outputs["first_row_sum_q26"],
                "row0_q12_12_word": k_outputs["first_q12_12_word"],
                "row0_q12_12_float": k_outputs["first_q12_12_float"],
            },
            "v_proj": {
                "row0_q26": v_outputs["first_row_sum_q26"],
                "row0_q12_12_word": v_outputs["first_q12_12_word"],
                "row0_q12_12_float": v_outputs["first_q12_12_float"],
            },
        },
        "descriptors": parsed_descriptors,
        "payload_offsets": {
            item["name"]: {
                "offset": f"0x{int(item['offset']):08X}",
                "address": format_addr(qmap_base + int(item["offset"])),
                "bytes": len(item["payload"]),
            }
            for item in payloads
        },
    }
    return image_bytes_blob, manifest, expected_qkv


def verify_image(
    *,
    image: bytes,
    payloads: list[dict[str, Any]],
    qmap_base: int,
    image_bytes: int,
) -> None:
    if len(image) != image_bytes:
        raise RuntimeError(f"Image size mismatch: expected {image_bytes}, got {len(image)}")

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
        "image_bytes": image_bytes,
        "flags": 0,
        "checksum32": 0,
    }
    if header != expected_header:
        raise RuntimeError(f"Header verification failed: {header}")

    unused_start = DESCRIPTOR_TABLE_OFFSET + DESCRIPTOR_COUNT * DESCRIPTOR_BYTES
    unused_end = DESCRIPTOR_TABLE_OFFSET + DESCRIPTOR_CAPACITY * DESCRIPTOR_BYTES
    if any(image[unused_start:unused_end]):
        raise RuntimeError("Unused descriptor slots are not zero-filled")

    for item in payloads:
        offset = int(item["offset"])
        payload = item["payload"]
        actual = image[offset : offset + len(payload)]
        if actual != payload:
            raise RuntimeError(f"Payload verification failed for {item['name']}")


def write_word32_hex(image: bytes, output: Path) -> None:
    if len(image) % 4 != 0:
        raise ValueError("Image length must be 32-bit aligned")
    output.parent.mkdir(parents=True, exist_ok=True)
    words = struct.unpack(f"<{len(image) // 4}I", image)
    output.write_text("".join(f"{word:08x}\n" for word in words), encoding="utf-8")


def write_expected_hex(expected_qkv: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    words = [int(value) & 0xFFFF_FFFF for value in expected_qkv.astype(np.int32)]
    output.write_text("".join(f"{word:08x}\n" for word in words), encoding="utf-8")


def write_outputs(
    *,
    image: bytes,
    manifest: dict[str, Any],
    expected_qkv: np.ndarray,
    output: Path,
    manifest_path: Path,
    sim_hex: Path,
    expected_hex: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image)
    write_word32_hex(image, sim_hex)
    write_expected_hex(expected_qkv, expected_hex)
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
        {
            "file": relpath(expected_hex),
            "sha256": sha256_file(expected_hex),
            "words32": expected_qkv.shape[0],
        },
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a QMAP v1 QKV projection work-packet image."
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
        "--sim-hex",
        type=Path,
        default=DEFAULT_SIM_HEX,
        help="Output little-endian 32-bit word hex for RTL simulation",
    )
    parser.add_argument(
        "--expected-hex",
        type=Path,
        default=DEFAULT_EXPECTED_HEX,
        help="Output expected Q/K/V I32_Q12_12 words for RTL checks",
    )
    parser.add_argument(
        "--qmap-base",
        type=lambda text: int(text.replace("_", ""), 0),
        default=QMAP_BASE,
        help="Physical QMAP base address, default 0x4_0008_0000",
    )
    parser.add_argument("--layer-id", type=int, default=0, help="Decoder layer index")
    parser.add_argument("--q-rows", type=int, default=Q_ROWS_FULL, help="q_proj output rows")
    parser.add_argument("--k-rows", type=int, default=KV_ROWS_FULL, help="k_proj output rows")
    parser.add_argument("--v-rows", type=int, default=KV_ROWS_FULL, help="v_proj output rows")
    parser.add_argument(
        "--activation-hex",
        type=Path,
        default=None,
        help="Optional I32_Q12_12 activation word32 hex override",
    )
    parser.add_argument(
        "--activation-source-name",
        type=str,
        default=None,
        help="Optional human-readable activation source label for the manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, manifest, expected_qkv = build_qmap_image(
        source_npz=args.input,
        qmap_base=args.qmap_base,
        layer_id=args.layer_id,
        q_rows=args.q_rows,
        k_rows=args.k_rows,
        v_rows=args.v_rows,
        activation_hex=args.activation_hex,
        activation_source_name=args.activation_source_name,
    )
    write_outputs(
        image=image,
        manifest=manifest,
        expected_qkv=expected_qkv,
        output=args.output,
        manifest_path=args.manifest,
        sim_hex=args.sim_hex,
        expected_hex=args.expected_hex,
    )

    print(f"Exported QMAP v1 Layer {args.layer_id} QKV projection packet")
    print("=" * 80)
    print(f"Input: {relpath(args.input)}")
    print(f"Output: {relpath(args.output)}")
    print(f"Manifest: {relpath(args.manifest)}")
    print(f"Sim hex: {relpath(args.sim_hex)}")
    print(f"Expected hex: {relpath(args.expected_hex)}")
    print(f"Image base: {format_addr(args.qmap_base)}")
    print(f"Image bytes: 0x{len(image):08X}")
    print(f"Descriptor count/capacity: {DESCRIPTOR_COUNT}/{DESCRIPTOR_CAPACITY}")
    print(
        "Rows: "
        f"q={manifest['row_counts']['q_rows']} "
        f"k={manifest['row_counts']['k_rows']} "
        f"v={manifest['row_counts']['v_rows']}"
    )
    print("Proof:")
    for key, value in manifest["proof"].items():
        print(f"  {key}: {value}")
    print("Sample output words:")
    for name, sample in manifest["sample_outputs"].items():
        print(
            f"  {name}: row0_q26={sample['row0_q26']} "
            f"row0_q12_12_word={sample['row0_q12_12_word']} "
            f"row0_q12_12_float={sample['row0_q12_12_float']:.8f}"
        )
    print(f"SHA256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
