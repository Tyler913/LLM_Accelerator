from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import load_model, manual_rope_cos_sin, tensor_to_float32
from vector_workspace import resolve_sim_vector_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
QMAP_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_qmap_v1"
SIM_VECTOR_DIR = resolve_sim_vector_dir(REPO_ROOT)

DEFAULT_OUTPUT = QMAP_VECTOR_DIR / "rope_context256_q1_15.bin"
DEFAULT_MANIFEST = QMAP_VECTOR_DIR / "rope_context256_q1_15_manifest.json"
DEFAULT_SIM_HEX = SIM_VECTOR_DIR / "rope_context256_q1_15_words32.hex"

DEFAULT_BASE_ADDR = 0x4_1A10_0000
DEFAULT_MAX_CONTEXT = 256
HIDDEN_SIZE = 1024
HEAD_DIM = 128
TRIG_WIDTH = 16
TRIG_FRAC = 15
WORD_BYTES = 4


def parse_int(text: str) -> int:
    return int(text.replace("_", ""), 0)


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_q1_15(values: np.ndarray) -> tuple[np.ndarray, int]:
    low = -(1 << (TRIG_WIDTH - 1))
    high = (1 << (TRIG_WIDTH - 1)) - 1
    scaled = np.rint(values.astype(np.float64) * float(1 << TRIG_FRAC))
    saturation_count = int(np.count_nonzero((scaled < low) | (scaled > high)))
    return np.clip(scaled, low, high).astype(np.int32), saturation_count


def write_words32_hex(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = values.reshape(-1)
    path.write_text(
        "\n".join(f"{int(value) & 0xFFFF_FFFF:08x}" for value in flat) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the persistent Qwen3 RoPE cos/sin table used by runtime-position RTL."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sim-hex", type=Path, default=DEFAULT_SIM_HEX)
    parser.add_argument("--base-addr", type=parse_int, default=DEFAULT_BASE_ADDR)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_context <= 0:
        raise ValueError("--max-context must be positive")
    if (args.base_addr & 0x3) != 0:
        raise ValueError("--base-addr must be 4-byte aligned")

    _, _, backbone = load_model()
    position_ids = torch.arange(args.max_context, dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros((1, args.max_context, HIDDEN_SIZE), dtype=torch.float32)
    with torch.no_grad():
        cos, sin = manual_rope_cos_sin(backbone.rotary_emb, dummy_x, position_ids)

    cos_float = tensor_to_float32(cos)[0].cpu().numpy()
    sin_float = tensor_to_float32(sin)[0].cpu().numpy()
    expected_shape = (args.max_context, HEAD_DIM)
    if cos_float.shape != expected_shape or sin_float.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected RoPE shapes cos={cos_float.shape}, sin={sin_float.shape}; "
            f"expected {expected_shape}"
        )

    cos_q1_15, cos_saturation_count = quantize_q1_15(cos_float)
    sin_q1_15, sin_saturation_count = quantize_q1_15(sin_float)
    cos_bytes = cos_q1_15.size * WORD_BYTES
    sin_bytes = sin_q1_15.size * WORD_BYTES
    cos_base_addr = args.base_addr
    sin_base_addr = cos_base_addr + cos_bytes

    payload = cos_q1_15.astype("<i4", copy=False).tobytes(order="C")
    payload += sin_q1_15.astype("<i4", copy=False).tobytes(order="C")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    write_words32_hex(args.sim_hex, np.concatenate((cos_q1_15.reshape(-1), sin_q1_15.reshape(-1))))

    if len(payload) != cos_bytes + sin_bytes:
        raise RuntimeError("RoPE payload size mismatch")
    if cos_q1_15[0, 0] != 0x7FFF or sin_q1_15[0, 0] != 0:
        raise RuntimeError("RoPE position-zero sanity check failed")

    manifest = {
        "format_version": 1,
        "name": f"qwen3_0p6b_rope_context{args.max_context}_q1_15",
        "base_addr": f"0x{args.base_addr:016X}",
        "total_bytes": len(payload),
        "sha256": sha256_file(args.output),
        "shape": [args.max_context, HEAD_DIM],
        "storage": "signed Q1.15 values widened to little-endian int32 words",
        "row_stride_bytes": HEAD_DIM * WORD_BYTES,
        "cos": {
            "base_addr": f"0x{cos_base_addr:016X}",
            "nbytes": cos_bytes,
            "saturation_count": cos_saturation_count,
        },
        "sin": {
            "base_addr": f"0x{sin_base_addr:016X}",
            "nbytes": sin_bytes,
            "saturation_count": sin_saturation_count,
        },
        "files": {
            "binary": relpath(args.output),
            "sim_hex": relpath(args.sim_hex),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Exported persistent Qwen3 RoPE runtime table")
    print("=" * 80)
    print(f"Shape:            [{args.max_context}, {HEAD_DIM}]")
    print(f"Cos base/bytes:   0x{cos_base_addr:016X} / 0x{cos_bytes:X}")
    print(f"Sin base/bytes:   0x{sin_base_addr:016X} / 0x{sin_bytes:X}")
    print(f"Total bytes:      0x{len(payload):X}")
    print(f"Binary:           {args.output}")
    print(f"Manifest:         {args.manifest}")


if __name__ == "__main__":
    main()
