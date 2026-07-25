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

QK_PREFIX = "qk_norm_rope_stage_128_real"
KV_PREFIX = "kv_cache_append_real"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "attention_frontend_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "attention_frontend_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_attention_frontend_image_words32.hex"
DEFAULT_Q_ROPE_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_attention_frontend_q_rope_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_0502_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 16
DESCRIPTOR_COUNT = 10
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0900
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_CONTEXT = 256
Q_COUNT = NUM_Q_HEADS * HEAD_DIM
KV_COUNT = NUM_KV_HEADS * HEAD_DIM
IN_WIDTH = 24
GAMMA_WIDTH = 16
TRIG_WIDTH = 16
OUT_WIDTH = 24
DATA_WIDTH = 32

TENSOR_ID_METADATA = 30
TENSOR_ID_Q_FLAT = 31
TENSOR_ID_K_FLAT = 32
TENSOR_ID_V_FLAT = 33
TENSOR_ID_Q_GAMMA = 34
TENSOR_ID_K_GAMMA = 35
TENSOR_ID_ROPE_COS = 36
TENSOR_ID_ROPE_SIN = 37
TENSOR_ID_KV_CACHE = 38
TENSOR_ID_Q_ROPE = 39

ROLE_ACTIVATION = 1
ROLE_OUTPUT = 4
ROLE_METADATA = 6
ROLE_KV_CACHE = 7
ROLE_ROPE_TABLE = 8
ROLE_PARAMETER = 9

DTYPE_U32 = 5
DTYPE_I32_Q12_12 = 19
DTYPE_I16_Q8_7 = 22
DTYPE_I16_Q1_15 = 23

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3

MATRIX_ID_Q_PROJ = 1
MATRIX_ID_K_PROJ = 2
MATRIX_ID_V_PROJ = 3
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
    parser = argparse.ArgumentParser(description="Export a QMAP attention front-end packet for RTL simulation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sim-hex", type=Path, default=DEFAULT_SIM_HEX)
    parser.add_argument("--q-rope-expected-hex", type=Path, default=DEFAULT_Q_ROPE_EXPECTED_HEX)
    parser.add_argument("--qk-prefix", type=str, default=QK_PREFIX, help="q/k norm + RoPE vector prefix")
    parser.add_argument("--kv-prefix", type=str, default=KV_PREFIX, help="KV cache append vector prefix")
    parser.add_argument(
        "--qmap-base",
        type=lambda text: int(text.replace("_", ""), 0),
        default=QMAP_BASE,
        help="Physical QMAP base address",
    )
    parser.add_argument("--q-base-addr", type=lambda text: int(text.replace("_", ""), 0), default=None)
    parser.add_argument("--k-base-addr", type=lambda text: int(text.replace("_", ""), 0), default=None)
    parser.add_argument("--v-base-addr", type=lambda text: int(text.replace("_", ""), 0), default=None)
    parser.add_argument(
        "--runtime-rope-cos-base-addr",
        type=lambda text: int(text.replace("_", ""), 0),
        default=None,
        help="External persistent [MAX_CONTEXT, HEAD_DIM] cos table base",
    )
    parser.add_argument(
        "--runtime-rope-sin-base-addr",
        type=lambda text: int(text.replace("_", ""), 0),
        default=None,
        help="External persistent [MAX_CONTEXT, HEAD_DIM] sin table base",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qmap_base = args.qmap_base
    runtime_rope_table = (args.runtime_rope_cos_base_addr is not None) or (
        args.runtime_rope_sin_base_addr is not None
    )
    if runtime_rope_table and (
        args.runtime_rope_cos_base_addr is None or args.runtime_rope_sin_base_addr is None
    ):
        raise ValueError("runtime RoPE mode requires both cos and sin base addresses")
    if runtime_rope_table and (
        (args.runtime_rope_cos_base_addr & 0x3) != 0
        or (args.runtime_rope_sin_base_addr & 0x3) != 0
    ):
        raise ValueError("runtime RoPE base addresses must be 4-byte aligned")

    q_meta = json.loads((SIM_VECTOR_DIR / f"{args.qk_prefix}_meta.json").read_text(encoding="utf-8"))
    kv_meta = json.loads((SIM_VECTOR_DIR / f"{args.kv_prefix}_meta.json").read_text(encoding="utf-8"))
    selected_position = int(q_meta["selected_position"])
    layer_id = int(kv_meta["cache"]["layer_id"])
    cache_base = int(str(kv_meta["cache"]["base_addr"]), 16)
    cache_write_count = int(kv_meta["cache"]["write_count"])

    q_input = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_q_input.hex", IN_WIDTH, signed=True)
    k_input = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_k_input.hex", IN_WIDTH, signed=True)
    v_input = read_hex_lines(SIM_VECTOR_DIR / f"{args.kv_prefix}_v_input.hex", IN_WIDTH, signed=True)
    q_gamma = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_q_gamma.hex", GAMMA_WIDTH, signed=True)
    k_gamma = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_k_gamma.hex", GAMMA_WIDTH, signed=True)
    cos = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_cos.hex", TRIG_WIDTH, signed=True)
    sin = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_sin.hex", TRIG_WIDTH, signed=True)
    q_rope_expected = read_hex_lines(SIM_VECTOR_DIR / f"{args.qk_prefix}_q_rope_expected.hex", OUT_WIDTH, signed=True)

    expected_shapes = {
        "q_input": (q_input, Q_COUNT),
        "k_input": (k_input, KV_COUNT),
        "v_input": (v_input, KV_COUNT),
        "q_gamma": (q_gamma, HEAD_DIM),
        "k_gamma": (k_gamma, HEAD_DIM),
        "cos": (cos, HEAD_DIM),
        "sin": (sin, HEAD_DIM),
        "q_rope_expected": (q_rope_expected, Q_COUNT),
    }
    for name, (array, size) in expected_shapes.items():
        if array.shape != (size,):
            raise RuntimeError(f"{name} shape mismatch: {array.shape}, expected {(size,)}")

    metadata_words = np.array(
        [
            layer_id,
            selected_position,
            MAX_CONTEXT,
            cache_base & 0xFFFF_FFFF,
            (cache_base >> 32) & 0xFFFF_FFFF,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            Q_COUNT,
            KV_COUNT,
            cache_write_count,
            0,
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
    cursor = add_payload(payloads, name="q_flat_q12_12", cursor=cursor, payload=as_le_bytes(q_input, "<i4"))
    cursor = add_payload(payloads, name="k_flat_q12_12", cursor=cursor, payload=as_le_bytes(k_input, "<i4"))
    cursor = add_payload(payloads, name="v_flat_q12_12", cursor=cursor, payload=as_le_bytes(v_input, "<i4"))
    cursor = add_payload(payloads, name="q_gamma_q8_7", cursor=cursor, payload=as_le_bytes(q_gamma, "<i4"))
    cursor = add_payload(payloads, name="k_gamma_q8_7", cursor=cursor, payload=as_le_bytes(k_gamma, "<i4"))
    if not runtime_rope_table:
        cursor = add_payload(payloads, name="rope_cos_q1_15", cursor=cursor, payload=as_le_bytes(cos, "<i4"))
        cursor = add_payload(payloads, name="rope_sin_q1_15", cursor=cursor, payload=as_le_bytes(sin, "<i4"))
    cursor = add_payload(payloads, name="q_rope_output_q12_12", cursor=cursor, payload=bytes(Q_COUNT * 4))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return qmap_base + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    rw = TENSOR_F_ROW_MAJOR
    q_stride = 4
    kv_cache_bytes = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * 4
    rope_table_bytes = MAX_CONTEXT * HEAD_DIM * 4
    rope_rank = 2 if runtime_rope_table else 1
    rope_dims = (MAX_CONTEXT, HEAD_DIM, 0, 0) if runtime_rope_table else (HEAD_DIM, 0, 0, 0)
    rope_strides = (HEAD_DIM * 4, 4, 0, 0) if runtime_rope_table else (q_stride, 0, 0, 0)
    rope_cos_base_addr = (
        args.runtime_rope_cos_base_addr if runtime_rope_table else addr("rope_cos_q1_15")
    )
    rope_sin_base_addr = (
        args.runtime_rope_sin_base_addr if runtime_rope_table else addr("rope_sin_q1_15")
    )
    rope_nbytes = rope_table_bytes if runtime_rope_table else HEAD_DIM * 4

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
            aux=(0, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_Q_FLAT,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=args.q_base_addr if args.q_base_addr is not None else addr("q_flat_q12_12"),
            nbytes=size("q_flat_q12_12"),
            dims=(Q_COUNT, 0, 0, 0),
            strides=(q_stride, 0, 0, 0),
            aux=(MATRIX_ID_Q_PROJ, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_K_FLAT,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=args.k_base_addr if args.k_base_addr is not None else addr("k_flat_q12_12"),
            nbytes=size("k_flat_q12_12"),
            dims=(KV_COUNT, 0, 0, 0),
            strides=(q_stride, 0, 0, 0),
            aux=(MATRIX_ID_K_PROJ, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_V_FLAT,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=ro,
            element_bits=IN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=args.v_base_addr if args.v_base_addr is not None else addr("v_flat_q12_12"),
            nbytes=size("v_flat_q12_12"),
            dims=(KV_COUNT, 0, 0, 0),
            strides=(q_stride, 0, 0, 0),
            aux=(MATRIX_ID_V_PROJ, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_Q_GAMMA,
            role=ROLE_PARAMETER,
            dtype=DTYPE_I16_Q8_7,
            rank=1,
            flags=ro,
            element_bits=GAMMA_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("q_gamma_q8_7"),
            nbytes=size("q_gamma_q8_7"),
            dims=(HEAD_DIM, 0, 0, 0),
            strides=(q_stride, 0, 0, 0),
            aux=(0, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_K_GAMMA,
            role=ROLE_PARAMETER,
            dtype=DTYPE_I16_Q8_7,
            rank=1,
            flags=ro,
            element_bits=GAMMA_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("k_gamma_q8_7"),
            nbytes=size("k_gamma_q8_7"),
            dims=(HEAD_DIM, 0, 0, 0),
            strides=(q_stride, 0, 0, 0),
            aux=(0, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_ROPE_COS,
            role=ROLE_ROPE_TABLE,
            dtype=DTYPE_I16_Q1_15,
            rank=rope_rank,
            flags=ro,
            element_bits=TRIG_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=rope_cos_base_addr,
            nbytes=rope_nbytes,
            dims=rope_dims,
            strides=rope_strides,
            aux=(0, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_ROPE_SIN,
            role=ROLE_ROPE_TABLE,
            dtype=DTYPE_I16_Q1_15,
            rank=rope_rank,
            flags=ro,
            element_bits=TRIG_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=rope_sin_base_addr,
            nbytes=rope_nbytes,
            dims=rope_dims,
            strides=rope_strides,
            aux=(0, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_KV_CACHE,
            role=ROLE_KV_CACHE,
            dtype=DTYPE_I32_Q12_12,
            rank=4,
            flags=rw,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=cache_base,
            nbytes=kv_cache_bytes,
            dims=(2, NUM_KV_HEADS, MAX_CONTEXT, HEAD_DIM),
            strides=(
                NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * 4,
                MAX_CONTEXT * HEAD_DIM * 4,
                HEAD_DIM * 4,
                4,
            ),
            aux=(0, layer_id, 0, selected_position),
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_Q_ROPE,
            role=ROLE_OUTPUT,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=wo,
            element_bits=OUT_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("q_rope_output_q12_12"),
            nbytes=size("q_rope_output_q12_12"),
            dims=(Q_COUNT, 0, 0, 0),
            strides=(q_stride, 0, 0, 0),
            aux=(0, layer_id, 0, selected_position),
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
    write_hex_lines(args.q_rope_expected_hex, q_rope_expected.astype(np.int64), DATA_WIDTH)

    manifest = {
        "format_version": 1,
        "name": f"qmap_layer{layer_id}_attention_frontend_runtime",
        "qmap_base": qmap_base,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "source_prefixes": {
            "qk_prefix": args.qk_prefix,
            "kv_prefix": args.kv_prefix,
        },
        "shape": {
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "q_count": Q_COUNT,
            "kv_count": KV_COUNT,
            "max_context": MAX_CONTEXT,
            "selected_position": selected_position,
            "layer_id": layer_id,
            "runtime_rope_table": runtime_rope_table,
        },
        "memory_layout": {
            "q_flat_addr": args.q_base_addr if args.q_base_addr is not None else addr("q_flat_q12_12"),
            "k_flat_addr": args.k_base_addr if args.k_base_addr is not None else addr("k_flat_q12_12"),
            "v_flat_addr": args.v_base_addr if args.v_base_addr is not None else addr("v_flat_q12_12"),
            "q_gamma_addr": addr("q_gamma_q8_7"),
            "k_gamma_addr": addr("k_gamma_q8_7"),
            "cos_addr": rope_cos_base_addr,
            "sin_addr": rope_sin_base_addr,
            "rope_table_shape": [MAX_CONTEXT, HEAD_DIM] if runtime_rope_table else [HEAD_DIM],
            "rope_row_stride_bytes": HEAD_DIM * 4,
            "kv_cache_base_addr": cache_base,
            "q_rope_output_addr": addr("q_rope_output_q12_12"),
        },
        "expected": {
            "q_rope_words": Q_COUNT,
            "cache_write_count": cache_write_count,
            "first_k_addr": kv_meta["cache"]["first_k_addr"],
            "last_k_addr": kv_meta["cache"]["last_k_addr"],
            "first_v_addr": kv_meta["cache"]["first_v_addr"],
            "last_v_addr": kv_meta["cache"]["last_v_addr"],
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "q_rope_expected_hex": relpath(args.q_rope_expected_hex),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP attention front-end runtime packet")
    print("=" * 80)
    print(f"QMAP base:       0x{qmap_base:016X}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Layer/position:  layer={layer_id}, position={selected_position}")
    print(f"Q/K/V words:     {Q_COUNT} / {KV_COUNT} / {KV_COUNT}")
    print(f"Cache writes:    {cache_write_count}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
