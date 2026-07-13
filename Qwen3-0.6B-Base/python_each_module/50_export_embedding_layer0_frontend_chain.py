from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = REPO_ROOT / "Temp"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = TEMP_ROOT / "embedding_layer0_frontend_vectors"
QKV_SOURCE_NPZ = (
    REPO_ROOT
    / "artifacts"
    / "test_vectors"
    / "qwen3_0p6b_q4_v0"
    / "qkv_layer0_last_token_q4.npz"
)

EMBED_WEIGHT_BASE = 0x0000_0004_0010_0000
EMBED_SCALE_BASE = 0x0000_0004_04B3_0000
INPUT_HIDDEN_BASE = 0x0000_0004_0509_2540
INPUT_NORM_QMAP_BASE = 0x0000_0004_050A_0000
INPUT_NORM_OUTPUT_OFFSET = 0x2540
# The legacy self-contained QKV packet at 0x4_0008_0000 extends into the
# persistent tied-weight region. Keep this simulation packet in staging.
QKV_QMAP_BASE = 0x0000_0004_1B40_0000
RMS_PREFIX = "embedding_layer0_input_rmsnorm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Temp-contained tied-Q4 embedding -> Layer 0 input RMSNorm "
            "-> full QKV simulation chain."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-id", type=int, default=None)
    return parser.parse_args()


def require_temp_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(TEMP_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output directory must stay under {TEMP_ROOT.resolve()}") from exc
    return resolved


def run_script(name: str, *args: str | Path | int) -> list[str]:
    command = [sys.executable, str(SCRIPT_DIR / name), *(str(arg) for arg in args)]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def descriptor_by_tensor(manifest: dict[str, Any], tensor_id: int) -> dict[str, Any]:
    for descriptor in manifest["descriptors"]:
        if int(descriptor["tensor_id"]) == tensor_id:
            return descriptor
    raise RuntimeError(f"missing tensor descriptor {tensor_id}")


def main() -> None:
    args = parse_args()
    output_dir = require_temp_path(args.output_dir)
    embedding_dir = output_dir / "embedding"
    rms_dir = output_dir / "rmsnorm"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    rms_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    embedding_args: list[str | Path | int] = ["--output-dir", embedding_dir]
    if args.token_id is not None:
        embedding_args.extend(["--token-id", args.token_id])
    commands.append(run_script("49_export_q4_embedding_vectors.py", *embedding_args))

    embedding_expected = embedding_dir / "embedding_expected_q14_10.hex"
    embedding_meta_path = embedding_dir / "embedding_meta.json"
    embedding_meta = load_json(embedding_meta_path)
    token_id = int(embedding_meta["token_id"])

    commands.append(
        run_script(
            "17_export_rmsnorm_fixed_vectors.py",
            "--prefix",
            RMS_PREFIX,
            "--layer-id",
            0,
            "--input-hex",
            embedding_expected,
            "--input-hex-width",
            32,
            "--input-source-name",
            f"tied_q4_embedding_token_{token_id}",
            "--output-dir",
            rms_dir,
        )
    )

    input_norm_bin = output_dir / "layer0_input_rmsnorm.qmap.bin"
    input_norm_manifest_path = output_dir / "layer0_input_rmsnorm_manifest.json"
    input_norm_image_hex = output_dir / "layer0_input_rmsnorm_image_words32.hex"
    input_norm_expected_hex = output_dir / "layer0_input_rmsnorm_expected_words32.hex"
    commands.append(
        run_script(
            "48_export_qmap_input_rmsnorm_image.py",
            "--prefix",
            RMS_PREFIX,
            "--vector-dir",
            rms_dir,
            "--layer-id",
            0,
            "--qmap-base",
            hex(INPUT_NORM_QMAP_BASE),
            "--hidden-base-addr",
            hex(INPUT_HIDDEN_BASE),
            "--output",
            input_norm_bin,
            "--manifest",
            input_norm_manifest_path,
            "--sim-hex",
            input_norm_image_hex,
            "--expected-hex",
            input_norm_expected_hex,
        )
    )

    qkv_bin = output_dir / "layer0_qkv_from_embedding_rmsnorm.qmap.bin"
    qkv_manifest_path = output_dir / "layer0_qkv_from_embedding_rmsnorm_manifest.json"
    qkv_image_hex = output_dir / "layer0_qkv_from_embedding_rmsnorm_image_words32.hex"
    qkv_expected_hex = output_dir / "layer0_qkv_from_embedding_rmsnorm_expected_words32.hex"
    input_norm_output_base = INPUT_NORM_QMAP_BASE + INPUT_NORM_OUTPUT_OFFSET
    commands.append(
        run_script(
            "21_export_qmap_qkv_projection_image.py",
            "--input",
            QKV_SOURCE_NPZ,
            "--layer-id",
            0,
            "--qmap-base",
            hex(QKV_QMAP_BASE),
            "--activation-hex",
            input_norm_expected_hex,
            "--activation-source-name",
            RMS_PREFIX,
            "--activation-base-addr",
            hex(input_norm_output_base),
            "--output",
            qkv_bin,
            "--manifest",
            qkv_manifest_path,
            "--sim-hex",
            qkv_image_hex,
            "--expected-hex",
            qkv_expected_hex,
        )
    )

    input_norm_manifest = load_json(input_norm_manifest_path)
    qkv_manifest = load_json(qkv_manifest_path)
    qkv_activation = descriptor_by_tensor(qkv_manifest, 2)

    if int(embedding_meta["weight_base_addr"]) != EMBED_WEIGHT_BASE:
        raise RuntimeError("embedding weight base changed unexpectedly")
    if int(embedding_meta["scale_base_addr"]) != EMBED_SCALE_BASE:
        raise RuntimeError("embedding scale base changed unexpectedly")
    if int(input_norm_manifest["memory_layout"]["hidden_addr"]) != INPUT_HIDDEN_BASE:
        raise RuntimeError("input RMSNorm hidden descriptor does not consume embedding output")
    if int(input_norm_manifest["memory_layout"]["input_norm_addr"]) != input_norm_output_base:
        raise RuntimeError("input RMSNorm output address changed unexpectedly")
    if int(qkv_activation["base_addr"]) != input_norm_output_base:
        raise RuntimeError("QKV activation descriptor does not consume input RMSNorm output")

    files = {
        "embedding_weight_hex": embedding_dir / "embedding_weight_words32.hex",
        "embedding_scale_hex": embedding_dir / "embedding_scale_words32.hex",
        "embedding_expected_hex": embedding_expected,
        "embedding_token_hex": embedding_dir / "embedding_token_id.hex",
        "input_norm_image_hex": input_norm_image_hex,
        "input_norm_expected_hex": input_norm_expected_hex,
        "qkv_image_hex": qkv_image_hex,
        "qkv_expected_hex": qkv_expected_hex,
    }
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty output {name}: {path}")

    chain_manifest = {
        "format_version": 1,
        "name": "tied_q4_embedding_to_layer0_input_rmsnorm_to_qkv",
        "token_id": token_id,
        "token_text": embedding_meta["token_text"],
        "addresses": {
            "embedding_weight_base": EMBED_WEIGHT_BASE,
            "embedding_scale_base": EMBED_SCALE_BASE,
            "input_hidden_base": INPUT_HIDDEN_BASE,
            "input_norm_qmap_base": INPUT_NORM_QMAP_BASE,
            "input_norm_output_base": input_norm_output_base,
            "qkv_qmap_base": QKV_QMAP_BASE,
            "qkv_activation_base": int(qkv_activation["base_addr"]),
        },
        "counts": {
            "embedding_output_words": 1024,
            "input_norm_output_words": 1024,
            "q_rows": int(qkv_manifest["row_counts"]["q_rows"]),
            "k_rows": int(qkv_manifest["row_counts"]["k_rows"]),
            "v_rows": int(qkv_manifest["row_counts"]["v_rows"]),
        },
        "files": {
            name: {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
        "commands": commands,
    }
    chain_manifest_path = output_dir / "chain_manifest.json"
    chain_manifest_path.write_text(
        json.dumps(chain_manifest, indent=2) + "\n", encoding="ascii"
    )

    print("PASS: exported tied-Q4 embedding -> Layer 0 input RMSNorm -> full QKV chain")
    print(f"token={token_id} text={embedding_meta['token_text']!r}")
    print(f"output_dir={output_dir}")
    print(f"input_hidden=0x{INPUT_HIDDEN_BASE:016X}")
    print(f"input_norm_output=0x{input_norm_output_base:016X}")
    print(f"qkv_activation=0x{int(qkv_activation['base_addr']):016X}")


if __name__ == "__main__":
    main()
