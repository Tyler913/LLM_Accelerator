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

FINAL_NORM_PREFIX = "final_rmsnorm_stage_real"
LM_HEAD_PREFIX = "lm_head_argmax_full_vocab_real"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "final_token_tail_full_vocab_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "final_token_tail_full_vocab_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_final_token_tail_full_vocab_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_final_token_tail_full_vocab_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0501_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 8
DESCRIPTOR_COUNT = 8
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0500
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

INPUT_SIZE = 1024
GROUP_SIZE = 64
GROUP_COUNT = INPUT_SIZE // GROUP_SIZE
TILE_ROWS = 16
HIDDEN_WIDTH = 24
NORM_WIDTH = 24
GAMMA_WIDTH = 16
WEIGHT_WIDTH = 4
SCALE_WIDTH = 16
ACT_WIDTH = 24
PARTIAL_WIDTH = ACT_WIDTH + WEIGHT_WIDTH + 6
SCALED_WIDTH = PARTIAL_WIDTH + SCALE_WIDTH
ROW_ACC_WIDTH = SCALED_WIDTH + 4 + 2
VOCAB_SIZE_DEFAULT = 151_936

TENSOR_ID_METADATA = 20
TENSOR_ID_NORM_OUTPUT = 21
TENSOR_ID_WEIGHT = 22
TENSOR_ID_SCALE = 23
TENSOR_ID_OUTPUT = 24
TENSOR_ID_EXPECTED = 25
TENSOR_ID_FINAL_HIDDEN = 26
TENSOR_ID_FINAL_GAMMA = 27

ROLE_ACTIVATION = 1
ROLE_Q4_WEIGHT = 2
ROLE_Q4_SCALE = 3
ROLE_OUTPUT = 4
ROLE_EXPECTED = 5
ROLE_METADATA = 6
ROLE_PARAMETER = 9

DTYPE_U32 = 5
DTYPE_PACKED_Q4_S4 = 16
DTYPE_U16_Q2_14 = 17
DTYPE_I32_Q12_12 = 19
DTYPE_I32_Q14_10 = 20
DTYPE_I16_Q8_7 = 22

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_PACKED_Q4_LOW_EVEN = 1 << 1
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3
TENSOR_F_DEBUG_ONLY = 1 << 4

MATRIX_ID_EMBED_LM_HEAD = 8
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
    parser = argparse.ArgumentParser(description="Export a QMAP final-token tail packet for RTL simulation.")
    parser.add_argument("--lm-prefix", default=LM_HEAD_PREFIX)
    parser.add_argument("--final-norm-prefix", default=FINAL_NORM_PREFIX)
    parser.add_argument("--qmap-base", type=lambda value: int(value.replace("_", ""), 0), default=QMAP_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sim-hex", type=Path, default=DEFAULT_SIM_HEX)
    parser.add_argument("--expected-hex", type=Path, default=DEFAULT_EXPECTED_HEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lm_prefix = str(args.lm_prefix)
    final_norm_prefix = str(args.final_norm_prefix)
    qmap_base = int(args.qmap_base)

    final_hidden = read_hex_lines(SIM_VECTOR_DIR / f"{final_norm_prefix}_input.hex", HIDDEN_WIDTH, signed=True)
    final_gamma = read_hex_lines(SIM_VECTOR_DIR / f"{final_norm_prefix}_gamma.hex", GAMMA_WIDTH, signed=True)
    final_norm_expected = read_hex_lines(SIM_VECTOR_DIR / f"{final_norm_prefix}_expected.hex", NORM_WIDTH, signed=True)

    scan_base = int(read_hex_lines(SIM_VECTOR_DIR / f"{lm_prefix}_scan_base_token.hex", 32, signed=False)[0])
    weight_base = int(read_hex_lines(SIM_VECTOR_DIR / f"{lm_prefix}_weight_base_addr.hex", 64, signed=False)[0])
    scale_base = int(read_hex_lines(SIM_VECTOR_DIR / f"{lm_prefix}_scale_base_addr.hex", 64, signed=False)[0])
    best_token = int(read_hex_lines(SIM_VECTOR_DIR / f"{lm_prefix}_expected_best_token.hex", 32, signed=False)[0])
    best_score = int(read_hex_lines(SIM_VECTOR_DIR / f"{lm_prefix}_expected_best_score_q26.hex", ROW_ACC_WIDTH, signed=True)[0])

    lm_meta = json.loads((SIM_VECTOR_DIR / f"{lm_prefix}_meta.json").read_text(encoding="utf-8"))
    shape = lm_meta["shape"]
    vocab_size = int(shape.get("vocab_size", VOCAB_SIZE_DEFAULT))
    scan_rows = int(shape["scan_rows"])
    tile_rows = int(shape["tile_rows"])
    tile_count = int(shape["tile_count"])
    if final_hidden.shape != (INPUT_SIZE,):
        raise RuntimeError(f"final hidden shape mismatch: {final_hidden.shape}")
    if final_gamma.shape != (INPUT_SIZE,):
        raise RuntimeError(f"final gamma shape mismatch: {final_gamma.shape}")
    if final_norm_expected.shape != (INPUT_SIZE,):
        raise RuntimeError(f"final norm expected shape mismatch: {final_norm_expected.shape}")
    if tile_rows != TILE_ROWS or scan_rows != tile_count * tile_rows:
        raise RuntimeError("LM-head tile metadata is inconsistent")

    score_u64 = best_score & ((1 << 64) - 1)
    expected_words = np.array(
        [
            best_token & 0xFFFF_FFFF,
            score_u64 & 0xFFFF_FFFF,
            (score_u64 >> 32) & 0xFFFF_FFFF,
        ],
        dtype=np.uint32,
    )
    metadata_words = np.array(
        [
            scan_base,
            scan_rows,
            tile_rows,
            tile_count,
            vocab_size,
            best_token,
            int(expected_words[1]),
            int(expected_words[2]),
            INPUT_SIZE,
            HIDDEN_WIDTH,
            GAMMA_WIDTH,
            NORM_WIDTH,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint32,
    )

    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET
    cursor = add_payload(payloads, name="metadata", cursor=cursor, payload=as_le_bytes(metadata_words, "<u4"))
    cursor = add_payload(payloads, name="final_hidden_q14_10", cursor=cursor, payload=as_le_bytes(final_hidden, "<i4"))
    cursor = add_payload(payloads, name="final_gamma_q8_7", cursor=cursor, payload=as_le_bytes(final_gamma, "<i4"))
    cursor = add_payload(payloads, name="final_norm_q12_12", cursor=cursor, payload=bytes(INPUT_SIZE * 4))
    cursor = add_payload(payloads, name="output_token_score", cursor=cursor, payload=bytes(12))
    cursor = add_payload(payloads, name="expected_token_score", cursor=cursor, payload=as_le_bytes(expected_words, "<u4"))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return qmap_base + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    rw = TENSOR_F_ROW_MAJOR
    packed_ro = ro | TENSOR_F_PACKED_Q4_LOW_EVEN
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    expected_flags = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY
    weight_row_bytes = INPUT_SIZE * WEIGHT_WIDTH // 8
    scale_row_bytes = GROUP_COUNT * SCALE_WIDTH // 8

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
            aux=(MATRIX_ID_EMBED_LM_HEAD, 0, scan_base, tile_count),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_NORM_OUTPUT,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=rw,
            element_bits=NORM_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("final_norm_q12_12"),
            nbytes=size("final_norm_q12_12"),
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_EMBED_LM_HEAD, 0, 0, 0),
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
            nbytes=vocab_size * weight_row_bytes,
            dims=(vocab_size, INPUT_SIZE, 0, 0),
            strides=(weight_row_bytes, 0, 0, 0),
            aux=(MATRIX_ID_EMBED_LM_HEAD, 0, scan_base, tile_count),
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
            nbytes=vocab_size * scale_row_bytes,
            dims=(vocab_size, GROUP_COUNT, 0, 0),
            strides=(scale_row_bytes, 2, 0, 0),
            aux=(MATRIX_ID_EMBED_LM_HEAD, 0, scan_base, tile_count),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_OUTPUT,
            role=ROLE_OUTPUT,
            dtype=DTYPE_U32,
            rank=1,
            flags=wo,
            element_bits=32,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("output_token_score"),
            nbytes=12,
            dims=(3, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_EMBED_LM_HEAD, 0, scan_base, tile_count),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_EXPECTED,
            role=ROLE_EXPECTED,
            dtype=DTYPE_U32,
            rank=1,
            flags=expected_flags,
            element_bits=32,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("expected_token_score"),
            nbytes=12,
            dims=(3, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(MATRIX_ID_EMBED_LM_HEAD, 0, scan_base, tile_count),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_FINAL_HIDDEN,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q14_10,
            rank=1,
            flags=ro,
            element_bits=HIDDEN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("final_hidden_q14_10"),
            nbytes=size("final_hidden_q14_10"),
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(0, 0, 0, 0),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_FINAL_GAMMA,
            role=ROLE_PARAMETER,
            dtype=DTYPE_I16_Q8_7,
            rank=1,
            flags=ro,
            element_bits=GAMMA_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("final_gamma_q8_7"),
            nbytes=size("final_gamma_q8_7"),
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=(0, 0, 0, 0),
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
    write_hex_lines(args.expected_hex, expected_words, 32)

    manifest = {
        "format_version": 1,
        "name": "qmap_final_token_tail_runtime",
        "lm_prefix": lm_prefix,
        "final_norm_prefix": final_norm_prefix,
        "qmap_base": qmap_base,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "vocab_size": vocab_size,
            "scan_base": scan_base,
            "scan_rows": scan_rows,
            "tile_rows": tile_rows,
            "tile_count": tile_count,
            "input_size": INPUT_SIZE,
            "group_count": GROUP_COUNT,
        },
        "memory_layout": {
            "final_hidden_addr": addr("final_hidden_q14_10"),
            "final_gamma_addr": addr("final_gamma_q8_7"),
            "final_norm_addr": addr("final_norm_q12_12"),
            "weight_base_addr": weight_base,
            "scale_base_addr": scale_base,
            "output_addr": addr("output_token_score"),
            "expected_addr": addr("expected_token_score"),
            "weight_row_bytes": weight_row_bytes,
            "scale_row_bytes": scale_row_bytes,
        },
        "expected": {
            "best_token": best_token,
            "best_score_q26": best_score,
            "output_words32": [int(value) for value in expected_words.tolist()],
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "expected_hex": relpath(args.expected_hex),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP final-token tail runtime packet")
    print("=" * 80)
    print(f"Final norm:    {final_norm_prefix}")
    print(f"LM prefix:     {lm_prefix}")
    print(f"QMAP base:     0x{qmap_base:016X}")
    print(f"Image bytes:   0x{image_bytes:X}")
    print(f"Scan window:   [{scan_base}, {scan_base + scan_rows}) rows={scan_rows}, tiles={tile_count}")
    print(f"Expected:      token={best_token}, score_q26={best_score}")
    print(f"Binary:        {args.output}")
    print(f"Simulation hex:{args.sim_hex}")
    print(f"Manifest:      {args.manifest}")


if __name__ == "__main__":
    main()
