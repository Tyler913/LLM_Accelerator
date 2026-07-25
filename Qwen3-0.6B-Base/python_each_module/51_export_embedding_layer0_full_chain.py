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

DEFAULT_OUTPUT_DIR = TEMP_ROOT / "embedding_layer0_full_chain_vectors"
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
QKV_STAGING_BASE = 0x0000_0004_1B40_0000
BODY_QMAP_BASE0 = 0x0000_0004_0500_0000
WEIGHT_WINDOW_BASE0 = 0x0000_0004_0600_0000
LAYER_QMAP_STRIDE = 0x0000_0000_1000_0000
WEIGHT_WINDOW_STRIDE = 0x0000_0000_0100_0000


def parse_int_auto(text: str) -> int:
    return int(text.replace("_", ""), 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Temp-contained tied-Q4 embedding through the complete "
            "Layer 0 QMAP chain."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-id", type=int, default=None)
    parser.add_argument("--input-hidden-base", type=parse_int_auto, default=INPUT_HIDDEN_BASE)
    parser.add_argument("--runtime-rope-cos-base-addr", type=parse_int_auto, default=None)
    parser.add_argument("--runtime-rope-sin-base-addr", type=parse_int_auto, default=None)
    parser.add_argument("--qkv-qmap-base0", type=parse_int_auto, default=QKV_STAGING_BASE)
    parser.add_argument("--body-qmap-base0", type=parse_int_auto, default=BODY_QMAP_BASE0)
    parser.add_argument("--qkv-qmap-stride", type=parse_int_auto, default=LAYER_QMAP_STRIDE)
    parser.add_argument("--body-qmap-stride", type=parse_int_auto, default=LAYER_QMAP_STRIDE)
    parser.add_argument("--weight-window-base0", type=parse_int_auto, default=WEIGHT_WINDOW_BASE0)
    parser.add_argument("--weight-window-stride", type=parse_int_auto, default=WEIGHT_WINDOW_STRIDE)
    parser.add_argument("--o-proj-weight-offset", type=parse_int_auto, default=0x0000_0000)
    parser.add_argument("--o-proj-scale-offset", type=parse_int_auto, default=0x0010_0000)
    parser.add_argument("--mlp-gate-weight-offset", type=parse_int_auto, default=0x0020_0000)
    parser.add_argument("--mlp-gate-scale-offset", type=parse_int_auto, default=0x0038_0000)
    parser.add_argument("--mlp-up-weight-offset", type=parse_int_auto, default=0x0040_0000)
    parser.add_argument("--mlp-up-scale-offset", type=parse_int_auto, default=0x0058_0000)
    parser.add_argument("--mlp-down-weight-offset", type=parse_int_auto, default=0x0060_0000)
    parser.add_argument("--mlp-down-scale-offset", type=parse_int_auto, default=0x0078_0000)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if (args.runtime_rope_cos_base_addr is None) != (args.runtime_rope_sin_base_addr is None):
        raise ValueError("runtime RoPE mode requires both cos and sin base addresses")
    output_dir = require_temp_path(args.output_dir)
    embedding_dir = output_dir / "embedding"
    layer_dir = output_dir / "layer0"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    layer_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    embedding_args: list[str | Path | int] = ["--output-dir", embedding_dir]
    if args.token_id is not None:
        embedding_args.extend(["--token-id", args.token_id])
    commands.append(run_script("49_export_q4_embedding_vectors.py", *embedding_args))

    embedding_meta_path = embedding_dir / "embedding_meta.json"
    embedding_expected = embedding_dir / "embedding_expected_q14_10.hex"
    embedding_meta = load_json(embedding_meta_path)
    token_id = int(embedding_meta["token_id"])
    layer_manifest_path = layer_dir / "qmap" / "layer0_chained_layer_manifest.json"

    layer_args: list[str | Path | int] = [
            "47_export_chained_layer_qmap_artifacts.py",
            "--layer-id",
            0,
            "--previous-layer-output-hex",
            embedding_expected,
            "--previous-layer-label",
            f"tied_q4_embedding_token_{token_id}",
            "--input-hidden-base",
            hex(args.input_hidden_base),
            "--qkv-npz",
            QKV_SOURCE_NPZ,
            "--qkv-qmap-base0",
            hex(args.qkv_qmap_base0),
            "--body-qmap-base0",
            hex(args.body_qmap_base0),
            "--qkv-qmap-stride",
            hex(args.qkv_qmap_stride),
            "--body-qmap-stride",
            hex(args.body_qmap_stride),
            "--weight-window-base0",
            hex(args.weight_window_base0),
            "--weight-window-stride",
            hex(args.weight_window_stride),
            "--o-proj-weight-offset",
            hex(args.o_proj_weight_offset),
            "--o-proj-scale-offset",
            hex(args.o_proj_scale_offset),
            "--mlp-gate-weight-offset",
            hex(args.mlp_gate_weight_offset),
            "--mlp-gate-scale-offset",
            hex(args.mlp_gate_scale_offset),
            "--mlp-up-weight-offset",
            hex(args.mlp_up_weight_offset),
            "--mlp-up-scale-offset",
            hex(args.mlp_up_scale_offset),
            "--mlp-down-weight-offset",
            hex(args.mlp_down_weight_offset),
            "--mlp-down-scale-offset",
            hex(args.mlp_down_scale_offset),
            "--output-root",
            layer_dir,
            "--manifest",
            layer_manifest_path,
    ]
    if args.runtime_rope_cos_base_addr is not None:
        layer_args.extend(
            [
                "--runtime-rope-cos-base-addr",
                hex(args.runtime_rope_cos_base_addr),
                "--runtime-rope-sin-base-addr",
                hex(args.runtime_rope_sin_base_addr),
            ]
        )
    commands.append(run_script(*layer_args))

    layer_manifest = load_json(layer_manifest_path)
    qmap_bases = layer_manifest["qmap_bases"]
    persistent_bases = layer_manifest["persistent_bases"]
    if int(embedding_meta["weight_base_addr"]) != EMBED_WEIGHT_BASE:
        raise RuntimeError("embedding weight base changed unexpectedly")
    if int(embedding_meta["scale_base_addr"]) != EMBED_SCALE_BASE:
        raise RuntimeError("embedding scale base changed unexpectedly")
    if int(str(qmap_bases["input_hidden"]).replace("_", ""), 0) != args.input_hidden_base:
        raise RuntimeError("Layer 0 RMSNorm does not consume the embedding output buffer")
    if int(str(qmap_bases["qkv"]).replace("_", ""), 0) != args.qkv_qmap_base0:
        raise RuntimeError("Layer 0 QKV packet base does not match the selected layout")

    stage_files: dict[str, Path] = {
        "embedding_expected": embedding_expected,
        "input_norm_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_input_rmsnorm_expected_words32.hex",
        "qkv_expected": layer_dir
        / "qmap"
        / "layer0_qkv_from_embedding_rmsnorm_full_expected_words32.hex",
        "q_rope_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_attention_frontend_q_rope_expected_words32.hex",
        "attention_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_attention_score_value_attn_out_expected_words32.hex",
        "o_proj_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_o_proj_expected_words32.hex",
        "post_hidden_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_post_attention_residual_norm_expected_hidden_words32.hex",
        "post_norm_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_post_attention_residual_norm_expected_norm_words32.hex",
        "gate_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_mlp_gate_up_expected_gate_words32.hex",
        "up_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_mlp_gate_up_expected_up_words32.hex",
        "mlp_hidden_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_mlp_silu_mul_expected_hidden_words32.hex",
        "down_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_mlp_down_expected_words32.hex",
        "layer_out_expected": layer_dir
        / "sim_vectors"
        / "qmap_layer0_chained_mlp_residual_add_expected_words32.hex",
    }
    for name, path in stage_files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty stage output {name}: {path}")

    manifest = {
        "format_version": 1,
        "name": "tied_q4_embedding_to_complete_layer0",
        "token_id": token_id,
        "token_text": embedding_meta["token_text"],
        "addresses": {
            "embedding_weight_base": EMBED_WEIGHT_BASE,
            "embedding_scale_base": EMBED_SCALE_BASE,
            "input_hidden_base": args.input_hidden_base,
            "qkv_staging_base": args.qkv_qmap_base0,
            "qmap_bases": qmap_bases,
            "persistent_bases": persistent_bases,
        },
        "counts": {
            "embedding": 1024,
            "input_norm": 1024,
            "q": 2048,
            "k": 1024,
            "v": 1024,
            "q_rope": 2048,
            "attention": 2048,
            "o_proj": 1024,
            "post_hidden": 1024,
            "post_norm": 1024,
            "gate": 3072,
            "up": 3072,
            "mlp_hidden": 3072,
            "down": 1024,
            "layer_out": 1024,
        },
        "layer_manifest": layer_manifest_path.relative_to(output_dir).as_posix(),
        "files": {
            name: {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in stage_files.items()
        },
        "commands": commands,
    }
    manifest_path = output_dir / "full_chain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")

    print("PASS: exported tied-Q4 embedding through the complete Layer 0 QMAP chain")
    print(f"token={token_id} text={embedding_meta['token_text']!r}")
    print(f"output_dir={output_dir}")
    print(f"qkv_staging=0x{args.qkv_qmap_base0:016X}")
    print(f"layer_out={stage_files['layer_out_expected']}")


if __name__ == "__main__":
    main()
