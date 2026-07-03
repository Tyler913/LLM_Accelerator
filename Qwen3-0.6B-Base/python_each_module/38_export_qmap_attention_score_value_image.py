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

SCORE_PREFIX = "attention_score_stage_real"
VALUE_PREFIX = "attention_softmax_value_stage_real"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "attention_score_value_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "attention_score_value_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_attention_score_value_image_words32.hex"
DEFAULT_ATTN_OUT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_attention_score_value_attn_out_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0503_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 8
DESCRIPTOR_COUNT = 5
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0500
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_CONTEXT = 256
CACHE_BASE_ADDR = 0x4_1410_0000
Q_COUNT = NUM_Q_HEADS * HEAD_DIM
KV_COUNT = NUM_KV_HEADS * HEAD_DIM
EXP_LUT_SIZE = 257
IN_WIDTH = 24
SCORE_SCALE_WIDTH = 32
EXP_WIDTH = 24
OUT_WIDTH = 24
DATA_WIDTH = 32

TENSOR_ID_METADATA = 40
TENSOR_ID_Q_ROPE = 41
TENSOR_ID_KV_CACHE = 42
TENSOR_ID_EXP_LUT = 43
TENSOR_ID_ATTN_OUT = 44

ROLE_ACTIVATION = 1
ROLE_OUTPUT = 4
ROLE_METADATA = 6
ROLE_KV_CACHE = 7
ROLE_PARAMETER = 9

DTYPE_U32 = 5
DTYPE_I32_Q12_12 = 19
DTYPE_U32_Q0_20 = 24

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3

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


def read_scalar(path: Path, width_bits: int, *, signed: bool) -> int:
    values = read_hex_lines(path, width_bits, signed=signed)
    if values.shape != (1,):
        raise RuntimeError(f"Expected one scalar in {path}, got {values.shape}")
    return int(values[0])


def write_hex_lines(path: Path, values: np.ndarray, width_bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << width_bits) - 1
    digits = (width_bits + 3) // 4
    flat = values.reshape(-1)
    path.write_text(
        "\n".join(f"{int(value) & mask:0{digits}x}" for value in flat) + "\n",
        encoding="utf-8",
    )


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
    parser = argparse.ArgumentParser(description="Export a QMAP attention score/value packet for RTL simulation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sim-hex", type=Path, default=DEFAULT_SIM_HEX)
    parser.add_argument("--attn-out-expected-hex", type=Path, default=DEFAULT_ATTN_OUT_EXPECTED_HEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    score_meta = json.loads((SIM_VECTOR_DIR / f"{SCORE_PREFIX}_meta.json").read_text(encoding="utf-8"))
    value_meta = json.loads((SIM_VECTOR_DIR / f"{VALUE_PREFIX}_meta.json").read_text(encoding="utf-8"))

    selected_position = int(score_meta["selected_position"])
    cache_length = int(score_meta["shape"]["cache_length"])
    score_scale_q0_31 = read_scalar(
        SIM_VECTOR_DIR / f"{SCORE_PREFIX}_score_scale_q0_31.hex",
        SCORE_SCALE_WIDTH,
        signed=True,
    )
    value_cache_length = int(value_meta["shape"]["cache_length"])
    if value_cache_length != cache_length:
        raise RuntimeError(f"cache_length mismatch: score={cache_length} value={value_cache_length}")
    if int(value_meta["fixed_point"]["attention_scale_q0_31"]) != score_scale_q0_31:
        raise RuntimeError("attention score scale drifted between score and value exporters")

    q_rope = read_hex_lines(SIM_VECTOR_DIR / f"{SCORE_PREFIX}_q_input.hex", IN_WIDTH, signed=True)
    k_cache = read_hex_lines(SIM_VECTOR_DIR / f"{SCORE_PREFIX}_k_cache.hex", IN_WIDTH, signed=True)
    v_cache = read_hex_lines(SIM_VECTOR_DIR / f"{VALUE_PREFIX}_v_cache.hex", IN_WIDTH, signed=True)
    exp_lut = read_hex_lines(SIM_VECTOR_DIR / f"{VALUE_PREFIX}_exp_lut.hex", EXP_WIDTH, signed=False)
    attn_out_expected = read_hex_lines(
        SIM_VECTOR_DIR / f"{VALUE_PREFIX}_expected_out.hex",
        OUT_WIDTH,
        signed=True,
    )

    expected_shapes = {
        "q_rope": (q_rope, Q_COUNT),
        "k_cache": (k_cache, cache_length * KV_COUNT),
        "v_cache": (v_cache, cache_length * KV_COUNT),
        "exp_lut": (exp_lut, EXP_LUT_SIZE),
        "attn_out_expected": (attn_out_expected, Q_COUNT),
    }
    for name, (array, size) in expected_shapes.items():
        if array.shape != (size,):
            raise RuntimeError(f"{name} shape mismatch: {array.shape}, expected {(size,)}")

    metadata_words = np.array(
        [
            0,  # layer_id
            selected_position,
            cache_length,
            score_scale_q0_31 & 0xFFFF_FFFF,
            CACHE_BASE_ADDR & 0xFFFF_FFFF,
            (CACHE_BASE_ADDR >> 32) & 0xFFFF_FFFF,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            MAX_CONTEXT,
            Q_COUNT,
            EXP_LUT_SIZE,
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
    cursor = add_payload(payloads, name="q_rope_q12_12", cursor=cursor, payload=as_le_bytes(q_rope, "<i4"))
    cursor = add_payload(payloads, name="exp_lut_uq0_20", cursor=cursor, payload=as_le_bytes(exp_lut, "<u4"))
    cursor = add_payload(payloads, name="attn_out_q12_12", cursor=cursor, payload=bytes(Q_COUNT * 4))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return QMAP_BASE + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    kv_ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    word_stride = 4
    kv_cache_bytes = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * word_stride

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
            aux=(score_scale_q0_31 & 0xFFFF_FFFF, 0, cache_length, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_Q_ROPE,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("q_rope_q12_12"),
            nbytes=size("q_rope_q12_12"),
            dims=(Q_COUNT, 0, 0, 0),
            strides=(word_stride, 0, 0, 0),
            aux=(0, 0, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_KV_CACHE,
            role=ROLE_KV_CACHE,
            dtype=DTYPE_I32_Q12_12,
            rank=4,
            flags=kv_ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=CACHE_BASE_ADDR,
            nbytes=kv_cache_bytes,
            dims=(2, NUM_KV_HEADS, MAX_CONTEXT, HEAD_DIM),
            strides=(
                NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * word_stride,
                MAX_CONTEXT * HEAD_DIM * word_stride,
                HEAD_DIM * word_stride,
                word_stride,
            ),
            aux=(0, 0, cache_length, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_EXP_LUT,
            role=ROLE_PARAMETER,
            dtype=DTYPE_U32_Q0_20,
            rank=1,
            flags=ro,
            element_bits=EXP_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("exp_lut_uq0_20"),
            nbytes=size("exp_lut_uq0_20"),
            dims=(EXP_LUT_SIZE, 0, 0, 0),
            strides=(word_stride, 0, 0, 0),
            aux=(0, 0, 0, 0),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_ATTN_OUT,
            role=ROLE_OUTPUT,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=wo,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("attn_out_q12_12"),
            nbytes=size("attn_out_q12_12"),
            dims=(Q_COUNT, 0, 0, 0),
            strides=(word_stride, 0, 0, 0),
            aux=(0, 0, 0, selected_position),
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
    write_hex_lines(args.attn_out_expected_hex, attn_out_expected.astype(np.int64), DATA_WIDTH)

    manifest = {
        "format_version": 1,
        "name": "qmap_attention_score_value_runtime",
        "qmap_base": QMAP_BASE,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "q_count": Q_COUNT,
            "kv_count": KV_COUNT,
            "max_context": MAX_CONTEXT,
            "cache_length": cache_length,
            "selected_position": selected_position,
            "score_count": int(NUM_Q_HEADS * cache_length),
            "k_request_count": int(NUM_Q_HEADS * cache_length * HEAD_DIM),
            "v_request_count": int(NUM_Q_HEADS * cache_length * HEAD_DIM),
            "output_count": Q_COUNT,
        },
        "fixed_point": {
            "attention_scale_q0_31": score_scale_q0_31,
            "exp_lut_size": EXP_LUT_SIZE,
            "exp_lut_dtype": "UQ0.20 padded to 32-bit words",
        },
        "memory_layout": {
            "q_rope_addr": addr("q_rope_q12_12"),
            "kv_cache_base_addr": CACHE_BASE_ADDR,
            "exp_lut_addr": addr("exp_lut_uq0_20"),
            "attn_out_addr": addr("attn_out_q12_12"),
        },
        "expected": {
            "attn_out_words": Q_COUNT,
            "k_cache_words_loaded_by_tb": int(k_cache.size),
            "v_cache_words_loaded_by_tb": int(v_cache.size),
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "attn_out_expected_hex": relpath(args.attn_out_expected_hex),
            "k_cache_source_hex": relpath(SIM_VECTOR_DIR / f"{SCORE_PREFIX}_k_cache.hex"),
            "v_cache_source_hex": relpath(SIM_VECTOR_DIR / f"{VALUE_PREFIX}_v_cache.hex"),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP attention score/value runtime packet")
    print("=" * 80)
    print(f"QMAP base:       0x{QMAP_BASE:016X}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Position/cache:  position={selected_position}, cache_length={cache_length}")
    print(f"Q RoPE words:    {Q_COUNT}")
    print(f"Exp LUT words:   {EXP_LUT_SIZE}")
    print(f"Attn out words:  {Q_COUNT}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
