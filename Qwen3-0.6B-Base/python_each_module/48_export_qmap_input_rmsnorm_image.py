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

PREFIX = "rmsnorm_1024_real"
DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "input_rmsnorm_runtime.qmap.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "input_rmsnorm_runtime_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "qmap_input_rmsnorm_image_words32.hex"
DEFAULT_EXPECTED_HEX = SIM_VECTOR_DIR / "qmap_input_rmsnorm_expected_words32.hex"

QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_BASE = 0x4_050A_0000
HEADER_BYTES = 256
DESCRIPTOR_BYTES = 128
DESCRIPTOR_CAPACITY = 8
DESCRIPTOR_COUNT = 5
DESCRIPTOR_TABLE_OFFSET = 0x0100
PAYLOAD_BASE_OFFSET = 0x0500
PAYLOAD_ALIGNMENT = 64
IMAGE_ALIGNMENT = 4096

INPUT_SIZE = 1024
HIDDEN_WIDTH = 24
GAMMA_WIDTH = 16
NORM_WIDTH = 24
HIDDEN_FRAC = 10
GAMMA_FRAC = 7
NORM_FRAC = 12
LAYER_ID = 0

TENSOR_ID_METADATA = 86
TENSOR_ID_HIDDEN = 87
TENSOR_ID_GAMMA = 88
TENSOR_ID_NORM = 89
TENSOR_ID_EXPECTED = 90

ROLE_ACTIVATION = 1
ROLE_OUTPUT = 4
ROLE_EXPECTED = 5
ROLE_METADATA = 6
ROLE_PARAMETER = 9

DTYPE_U32 = 5
DTYPE_I32_Q12_12 = 19
DTYPE_I32_Q14_10 = 20
DTYPE_I16_Q8_7 = 22

TENSOR_F_ROW_MAJOR = 1 << 0
TENSOR_F_READ_ONLY = 1 << 2
TENSOR_F_WRITE_ONLY = 1 << 3
TENSOR_F_DEBUG_ONLY = 1 << 4

STAGE_ID_INPUT_RMSNORM = 6
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
    parser = argparse.ArgumentParser(
        description="Export a QMAP runtime packet for per-layer input RMSNorm simulation."
    )
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=SIM_VECTOR_DIR,
        help="Directory containing the fixed RMSNorm vectors named by --prefix.",
    )
    parser.add_argument("--layer-id", type=int, default=LAYER_ID)
    parser.add_argument("--qmap-base", type=parse_int_auto, default=QMAP_BASE)
    parser.add_argument(
        "--hidden-base-addr",
        type=parse_int_auto,
        default=None,
        help="Optional external hidden-buffer address used by the hidden descriptor.",
    )
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
    vector_dir = args.vector_dir.resolve()

    hidden = read_hex_lines(vector_dir / f"{prefix}_input.hex", HIDDEN_WIDTH, signed=True)
    gamma = read_hex_lines(vector_dir / f"{prefix}_gamma.hex", GAMMA_WIDTH, signed=True)
    expected_norm = read_hex_lines(vector_dir / f"{prefix}_expected.hex", NORM_WIDTH, signed=True)

    for name, values in (
        ("hidden", hidden),
        ("gamma", gamma),
        ("expected_norm", expected_norm),
    ):
        if values.shape != (INPUT_SIZE,):
            raise RuntimeError(f"{name} shape mismatch: expected {(INPUT_SIZE,)}, got {values.shape}")

    hidden_i32 = hidden.astype(np.int32)
    gamma_i32 = gamma.astype(np.int32)
    expected_i32 = expected_norm.astype(np.int32)
    vector_bytes = INPUT_SIZE * 4

    scalar_files = {
        "sum_squares": vector_dir / f"{prefix}_sum_squares.hex",
        "mean_square": vector_dir / f"{prefix}_mean_square.hex",
        "rms": vector_dir / f"{prefix}_rms.hex",
        "inv_rms": vector_dir / f"{prefix}_inv_rms.hex",
        "saturation": vector_dir / f"{prefix}_saturation.hex",
    }
    scalar_values = {
        name: int(read_hex_lines(path, 64 if name in ("sum_squares", "mean_square") else 32, signed=False)[0])
        for name, path in scalar_files.items()
    }
    metadata_words = np.array(
        [
            layer_id,
            INPUT_SIZE,
            HIDDEN_WIDTH,
            GAMMA_WIDTH,
            NORM_WIDTH,
            STAGE_ID_INPUT_RMSNORM,
            HIDDEN_FRAC,
            GAMMA_FRAC,
            NORM_FRAC,
            scalar_values["rms"],
            scalar_values["inv_rms"],
            scalar_values["saturation"],
            TENSOR_ID_HIDDEN,
            TENSOR_ID_NORM,
        ],
        dtype=np.uint32,
    )

    payloads: list[dict[str, Any]] = []
    cursor = PAYLOAD_BASE_OFFSET
    cursor = add_payload(payloads, name="metadata", cursor=cursor, payload=as_le_bytes(metadata_words, "<u4"))
    cursor = add_payload(payloads, name="hidden_q14_10", cursor=cursor, payload=as_le_bytes(hidden_i32, "<i4"))
    cursor = add_payload(payloads, name="input_gamma_q8_7", cursor=cursor, payload=as_le_bytes(gamma_i32, "<i4"))
    cursor = add_payload(payloads, name="input_norm_q12_12", cursor=cursor, payload=bytes(vector_bytes))
    cursor = add_payload(payloads, name="expected_input_norm_q12_12", cursor=cursor, payload=as_le_bytes(expected_i32, "<i4"))
    image_bytes = align_up(cursor, IMAGE_ALIGNMENT)

    payload_by_name = {item["name"]: item for item in payloads}

    def addr(name: str) -> int:
        return qmap_base + int(payload_by_name[name]["offset"])

    def size(name: str) -> int:
        return len(payload_by_name[name]["payload"])

    ro = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY
    wo = TENSOR_F_ROW_MAJOR | TENSOR_F_WRITE_ONLY
    expected_flags = TENSOR_F_ROW_MAJOR | TENSOR_F_READ_ONLY | TENSOR_F_DEBUG_ONLY
    common_aux = (STAGE_ID_INPUT_RMSNORM, layer_id, INPUT_SIZE, 0)

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
            aux=common_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_HIDDEN,
            role=ROLE_ACTIVATION,
            dtype=DTYPE_I32_Q14_10,
            rank=1,
            flags=ro,
            element_bits=HIDDEN_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=args.hidden_base_addr
            if args.hidden_base_addr is not None
            else addr("hidden_q14_10"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=common_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_GAMMA,
            role=ROLE_PARAMETER,
            dtype=DTYPE_I16_Q8_7,
            rank=1,
            flags=ro,
            element_bits=GAMMA_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("input_gamma_q8_7"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=common_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_NORM,
            role=ROLE_OUTPUT,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=wo,
            element_bits=NORM_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("input_norm_q12_12"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=common_aux,
        ),
        pack_descriptor(
            tensor_id=TENSOR_ID_EXPECTED,
            role=ROLE_EXPECTED,
            dtype=DTYPE_I32_Q12_12,
            rank=1,
            flags=expected_flags,
            element_bits=NORM_WIDTH,
            group_size=0,
            scale_tensor_id=NO_TENSOR_ID,
            base_addr=addr("expected_input_norm_q12_12"),
            nbytes=vector_bytes,
            dims=(INPUT_SIZE, 0, 0, 0),
            strides=(4, 0, 0, 0),
            aux=common_aux,
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

    meta_path = vector_dir / f"{prefix}_meta.json"
    focused_meta: dict[str, Any] = {}
    if meta_path.is_file():
        focused_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    manifest = {
        "format_version": 1,
        "name": f"qmap_layer{layer_id}_input_rmsnorm_runtime",
        "prefix": prefix,
        "qmap_base": qmap_base,
        "image_bytes": image_bytes,
        "sha256": sha256_file(args.output),
        "descriptor_count": DESCRIPTOR_COUNT,
        "descriptor_capacity": DESCRIPTOR_CAPACITY,
        "shape": {
            "layer_id": layer_id,
            "input_size": INPUT_SIZE,
            "hidden_width": HIDDEN_WIDTH,
            "gamma_width": GAMMA_WIDTH,
            "norm_width": NORM_WIDTH,
        },
        "memory_layout": {
            "hidden_addr": args.hidden_base_addr
            if args.hidden_base_addr is not None
            else addr("hidden_q14_10"),
            "hidden_is_external": args.hidden_base_addr is not None,
            "gamma_addr": addr("input_gamma_q8_7"),
            "input_norm_addr": addr("input_norm_q12_12"),
            "expected_norm_addr": addr("expected_input_norm_q12_12"),
            "vector_bytes": vector_bytes,
        },
        "expected": {
            "norm_words": int(expected_i32.size),
            "norm_min": int(np.min(expected_i32)),
            "norm_max": int(np.max(expected_i32)),
        },
        "debug_scalars": scalar_values,
        "focused_meta": {
            key: focused_meta.get(key)
            for key in ("selected_position", "selected_token_id", "module", "eps_float")
            if key in focused_meta
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
            "expected_hex": relpath(args.expected_hex),
            "focused_meta": relpath(meta_path) if meta_path.is_file() else None,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported QMAP input RMSNorm runtime packet")
    print("=" * 80)
    print(f"QMAP base:       0x{qmap_base:016X}")
    print(f"Layer:           {layer_id}")
    print(f"Image bytes:     0x{image_bytes:X}")
    print(f"Input words:     {INPUT_SIZE}")
    print(f"Output words:    {expected_i32.size}")
    print(f"Saturation:      {scalar_values['saturation']}")
    print(f"Binary:          {args.output}")
    print(f"Simulation hex:  {args.sim_hex}")
    print(f"Manifest:        {args.manifest}")


if __name__ == "__main__":
    main()
