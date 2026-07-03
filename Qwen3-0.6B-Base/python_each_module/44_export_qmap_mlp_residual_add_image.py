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

PREFIX = "mlp_residual_add_stage_real"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "mlp_residual_add_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "mlp_residual_add_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_mlp_residual_add_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_mlp_residual_add_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0509_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 8
DESCRIPTOR_COUNT = 5
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0500
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

INPUT_SIZE = 1024
POST_ATTENTION_WIDTH = 24
POST_ATTENTION_FRAC = 10
DOWN_WIDTH = 24
DOWN_FRAC = 12
OUT_WIDTH = 24
OUT_FRAC = 10
DOWN_TO_OUT_SHIFT = DOWN_FRAC - OUT_FRAC
LAYER_ID = 0

TENSOR_ID_METADATA = 81
TENSOR_ID_POST_ATTN = 82
TENSOR_ID_DOWN = 83
TENSOR_ID_OUTPUT = 84
TENSOR_ID_EXPECTED = 85

ROLE_ACTIVATION = 1
ROLE_OUTPUT = 4
ROLE_EXPECTED = 5
ROLE_METADATA = 6

DTYPE_U32 = 5
DTYPE_I32_Q12_12 = 19
DTYPE_I32_Q14_10 = 20

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3
TENSOR_F_DEBUG_ONLY = 1 << 4

MATRIX_ID_DOWN_PROJ = 7
STAGE_ID_POST_ATTN_RESIDUAL_NORM = 1
STAGE_ID_MLP_RESIDUAL_ADD = 5
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
    parser = argparse.ArgumentParser(
        description="Export a QMAP runtime packet for Layer 0 final MLP residual-add simulation."
    )
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sim-hex", type=Path, default=DEFAULT_SIM_HEX)
    parser.add_argument("--expected-hex", type=Path, default=DEFAULT_EXPECTED_HEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = str(args.prefix)

    post_attn_hidden = read_hex_lines(
        SIM_VECTOR_DIR / f"{prefix}_post_attn_hidden.hex",
        POST_ATTENTION_WIDTH,
        signed=True,
    )
    down_out = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_down_out.hex", DOWN_WIDTH, signed=True)
    expected = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_expected_layer_out.hex", OUT_WIDTH, signed=True)
    saturation = read_hex_lines(SIM_VECTOR_DIR / f"{prefix}_residual_saturation.hex", 1, signed=False)

    for name, values in (
        ("post_attn_hidden", post_attn_hidden),
        ("down_out", down_out),
        ("expected", expected),
    ):
        if values.shape != (INPUT_SIZE,):
            raise RuntimeError(f"{name} shape mismatch: expected {(INPUT_SIZE,)}, got {values.shape}")
    if saturation.shape != (1,):
        raise RuntimeError(f"saturation shape mismatch: expected {(1,)}, got {saturation.shape}")

    chained_post_path = SIM_VECTOR_DIR / "qmap_post_attention_residual_norm_expected_hidden_words32.hex"
    if chained_post_path.is_file():
        chained_post = read_hex_lines(chained_post_path, 32, signed=True)
        if chained_post.shape != (INPUT_SIZE,):
            raise RuntimeError(f"chained post-attn shape mismatch: expected {(INPUT_SIZE,)}, got {chained_post.shape}")
        if not np.array_equal(chained_post.astype(np.int64), post_attn_hidden.astype(np.int64)):
            mismatch = int(np.nonzero(chained_post.astype(np.int64) != post_attn_hidden.astype(np.int64))[0][0])
            raise RuntimeError(f"post_attn_hidden does not match QMAP post-attention hidden at index {mismatch}")

    chained_down_path = SIM_VECTOR_DIR / "qmap_mlp_down_expected_words32.hex"
    if chained_down_path.is_file():
        chained_down = read_hex_lines(chained_down_path, 32, signed=True)
        if chained_down.shape != (INPUT_SIZE,):
            raise RuntimeError(f"chained down shape mismatch: expected {(INPUT_SIZE,)}, got {chained_down.shape}")
        if not np.array_equal(chained_down.astype(np.int64), down_out.astype(np.int64)):
            mismatch = int(np.nonzero(chained_down.astype(np.int64) != down_out.astype(np.int64))[0][0])
            raise RuntimeError(f"down_out does not match QMAP MLP down output at index {mismatch}")

    post_i32 = post_attn_hidden.astype(np.int32)
    down_i32 = down_out.astype(np.int32)
    expected_i32 = expected.astype(np.int32)
    vector_bytes = INPUT_SIZE * 4
    metadata_words = np.array(
        [
            LAYER_ID,
            INPUT_SIZE,
            POST_ATTENTION_WIDTH,
            POST_ATTENTION_FRAC,
            DOWN_WIDTH,
            DOWN_FRAC,
            OUT_WIDTH,
            OUT_FRAC,
            DOWN_TO_OUT_SHIFT,
            STAGE_ID_MLP_RESIDUAL_ADD,
            int(saturation[0]),
        ],
        dtype=np.uint32,
    )

    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET
    cursor = add_payload(payloads, name="metadata", cursor=cursor, payload=as_le_bytes(metadata_words, "<u4"))
    cursor = add_payload(payloads, name="post_attn_hidden_q14_10", cursor=cursor, payload=as_le_bytes(post_i32, "<i4"))
    cursor = add_payload(payloads, name="down_out_q12_12", cursor=cursor, payload=as_le_bytes(down_i32, "<i4"))
    cursor = add_payload(payloads, name="layer_out_q14_10", cursor=cursor, payload=bytes(vector_bytes))
    cursor = add_payload(payloads, name="expected_layer_out_q14_10", cursor=cursor, payload=as_le_bytes(expected_i32, "<i4"))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return QMAP_BASE + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    expected_flags = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY
    stage_aux = (STAGE_ID_MLP_RESIDUAL_ADD, LAYER_ID, INPUT_SIZE, 0)
    post_aux = (STAGE_ID_POST_ATTN_RESIDUAL_NORM, LAYER_ID, INPUT_SIZE, 0)
    down_aux = (MATRIX_ID_DOWN_PROJ, LAYER_ID, INPUT_SIZE, 0)

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
            tensor_id=TENSOR_ID_POST_ATTN,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q14_10,
            rank=1,
            flags=ro,
            element_bits=POST_ATTENTION_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("post_attn_hidden_q14_10"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=post_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_DOWN,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=DOWN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("down_out_q12_12"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=down_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_OUTPUT,
            role=ROLE_OUTPUT,
            dtype=DTYPE_I32_Q14_10,
            rank=1,
            flags=wo,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("layer_out_q14_10"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=stage_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_EXPECTED,
            role=ROLE_EXPECTED,
            dtype=DTYPE_I32_Q14_10,
            rank=1,
            flags=expected_flags,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("expected_layer_out_q14_10"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=stage_aux,
        ),
    ]

    image = bytearray(image_bytes)
    image[0:HEADER_BYTES] = pack_header(QMAP_BASE, image_bytes)
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

    manifest = {
        "format_version": 1,
        "name": "qmap_mlp_residual_add_runtime",
        "prefix": prefix,
        "qmap_base": QMAP_BASE,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "input_size": INPUT_SIZE,
            "post_attention_width": POST_ATTENTION_WIDTH,
            "down_width": DOWN_WIDTH,
            "out_width": OUT_WIDTH,
            "down_to_out_shift": DOWN_TO_OUT_SHIFT,
        },
        "memory_layout": {
            "post_attn_hidden_addr": addr("post_attn_hidden_q14_10"),
            "down_out_addr": addr("down_out_q12_12"),
            "layer_out_addr": addr("layer_out_q14_10"),
            "expected_addr": addr("expected_layer_out_q14_10"),
            "vector_bytes": vector_bytes,
        },
        "expected": {
            "words": int(expected_i32.size),
            "min": int(np.min(expected_i32)),
            "max": int(np.max(expected_i32)),
            "saturation": int(saturation[0]),
        },
        "focused_meta": {
            key: focused_meta.get(key)
            for key in ("selected_position", "selected_token_id", "selected_token_text", "source_post_attention_prefix", "source_down_prefix")
            if key in focused_meta
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "expected_hex": relpath(args.expected_hex),
            "focused_meta": relpath(focused_meta_path) if focused_meta_path.is_file() else None,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP MLP residual-add runtime packet")
    print("=" * 80)
    print(f"QMAP base:       0x{QMAP_BASE:016X}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Input words:     {INPUT_SIZE}")
    print(f"Expected words:  {expected_i32.size}")
    print(f"Saturation:      {int(saturation[0])}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
