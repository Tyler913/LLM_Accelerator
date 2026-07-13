from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = REPO_ROOT / "Temp"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = TEMP_ROOT / "embedding_true3_final_chain_vectors"

LAYER0_OUTPUT_BASE = 0x0000_0004_0509_2540
LAYER1_OUTPUT_BASE = 0x0000_0004_1509_2540
LAYER2_OUTPUT_BASE = 0x0000_0004_2509_2540
FINAL_TAIL_QMAP_BASE = 0x0000_0004_0501_0000

FINAL_NORM_PREFIX = "final_rmsnorm_from_embedding_true3"
LM_HEAD_PREFIX = "lm_head_full_vocab_from_embedding_true3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one Temp-contained tied-Q4 embedding -> Layer0 -> Layer1 -> "
            "Layer2 -> final RMSNorm -> full-vocabulary LM-head chain."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-id", type=int, default=None)
    parser.add_argument("--lm-chunk-rows", type=int, default=1024)
    return parser.parse_args()


def require_temp_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(TEMP_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output directory must stay under {TEMP_ROOT.resolve()}") from exc
    return resolved


def run_script(
    name: str,
    *args: str | Path | int,
    env: dict[str, str] | None = None,
) -> list[str]:
    command = [sys.executable, str(SCRIPT_DIR / name), *(str(arg) for arg in args)]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return command


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_addr(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value.replace("_", ""), 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty generated file: {path}")
    return path


def layer_output_hex(layer_dir: Path, layer_id: int) -> Path:
    return (
        layer_dir
        / "sim_vectors"
        / f"qmap_layer{layer_id}_chained_mlp_residual_add_expected_words32.hex"
    )


def main() -> None:
    args = parse_args()
    output_dir = require_temp_path(args.output_dir)
    layer0_root = output_dir / "embedding_layer0"
    layer1_root = output_dir / "layer1"
    layer2_root = output_dir / "layer2"
    tail_root = output_dir / "final_tail"
    tail_vectors = tail_root / "sim_vectors"
    tail_qmap = tail_root / "qmap"
    for path in (layer0_root, layer1_root, layer2_root, tail_vectors, tail_qmap):
        path.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    layer0_args: list[str | Path | int] = ["--output-dir", layer0_root]
    if args.token_id is not None:
        layer0_args.extend(["--token-id", args.token_id])
    commands.append(run_script("51_export_embedding_layer0_full_chain.py", *layer0_args))

    layer0_output = require_file(layer_output_hex(layer0_root / "layer0", 0))
    layer1_manifest_path = layer1_root / "qmap" / "layer1_chained_layer_manifest.json"
    commands.append(
        run_script(
            "47_export_chained_layer_qmap_artifacts.py",
            "--layer-id",
            1,
            "--previous-layer-output-hex",
            layer0_output,
            "--previous-layer-label",
            "tied_embedding_complete_layer0",
            "--input-hidden-base",
            hex(LAYER0_OUTPUT_BASE),
            "--output-root",
            layer1_root,
            "--manifest",
            layer1_manifest_path,
        )
    )

    layer1_output = require_file(layer_output_hex(layer1_root, 1))
    layer2_manifest_path = layer2_root / "qmap" / "layer2_chained_layer_manifest.json"
    commands.append(
        run_script(
            "47_export_chained_layer_qmap_artifacts.py",
            "--layer-id",
            2,
            "--previous-layer-output-hex",
            layer1_output,
            "--previous-layer-label",
            "tied_embedding_complete_layer1",
            "--input-hidden-base",
            hex(LAYER1_OUTPUT_BASE),
            "--output-root",
            layer2_root,
            "--manifest",
            layer2_manifest_path,
        )
    )

    layer2_output = require_file(layer_output_hex(layer2_root, 2))
    tail_env = os.environ.copy()
    tail_env["QMAP_SIM_VECTOR_DIR"] = str(tail_vectors)

    commands.append(
        run_script(
            "32_export_final_rmsnorm_vectors.py",
            "--prefix",
            FINAL_NORM_PREFIX,
            "--input-hex",
            layer2_output,
            "--input-hex-width",
            32,
            "--input-source-name",
            "tied_embedding_complete_layer2",
            env=tail_env,
        )
    )
    final_norm_output = require_file(tail_vectors / f"{FINAL_NORM_PREFIX}_expected.hex")

    commands.append(
        run_script(
            "35_export_lm_head_full_vocab_vectors.py",
            "--prefix",
            LM_HEAD_PREFIX,
            "--chunk-rows",
            args.lm_chunk_rows,
            "--activation-hex",
            final_norm_output,
            "--activation-source-name",
            FINAL_NORM_PREFIX,
            env=tail_env,
        )
    )

    tail_binary = tail_qmap / "final_token_tail_embedding_true3.qmap.bin"
    tail_manifest_path = tail_qmap / "final_token_tail_embedding_true3_manifest.json"
    tail_image_hex = tail_vectors / "qmap_final_token_tail_embedding_true3_image_words32.hex"
    tail_expected_hex = tail_vectors / "qmap_final_token_tail_embedding_true3_expected_words32.hex"
    commands.append(
        run_script(
            "36_export_qmap_final_token_tail_image.py",
            "--lm-prefix",
            LM_HEAD_PREFIX,
            "--final-norm-prefix",
            FINAL_NORM_PREFIX,
            "--output",
            tail_binary,
            "--manifest",
            tail_manifest_path,
            "--sim-hex",
            tail_image_hex,
            "--expected-hex",
            tail_expected_hex,
            env=tail_env,
        )
    )

    layer0_manifest = load_json(layer0_root / "layer0" / "qmap" / "layer0_chained_layer_manifest.json")
    layer1_manifest = load_json(layer1_manifest_path)
    layer2_manifest = load_json(layer2_manifest_path)
    tail_manifest = load_json(tail_manifest_path)

    expected_contracts = (
        ("Layer0 output", layer0_manifest["qmap_bases"]["mlp_residual_add"], LAYER0_OUTPUT_BASE - 0x2540),
        ("Layer1 input", layer1_manifest["qmap_bases"]["input_hidden"], LAYER0_OUTPUT_BASE),
        ("Layer1 output", layer1_manifest["qmap_bases"]["mlp_residual_add"], LAYER1_OUTPUT_BASE - 0x2540),
        ("Layer2 input", layer2_manifest["qmap_bases"]["input_hidden"], LAYER1_OUTPUT_BASE),
        ("Layer2 output", layer2_manifest["qmap_bases"]["mlp_residual_add"], LAYER2_OUTPUT_BASE - 0x2540),
    )
    for label, actual, expected in expected_contracts:
        if parse_addr(actual) != expected:
            raise RuntimeError(f"{label} address mismatch: {actual} != 0x{expected:016X}")
    if int(tail_manifest["qmap_base"]) != FINAL_TAIL_QMAP_BASE:
        raise RuntimeError("final-tail QMAP base changed unexpectedly")

    required_files = {
        "embedding_layer0_manifest": layer0_root / "full_chain_manifest.json",
        "layer0_output": layer0_output,
        "layer1_manifest": layer1_manifest_path,
        "layer1_output": layer1_output,
        "layer2_manifest": layer2_manifest_path,
        "layer2_output": layer2_output,
        "final_norm_output": final_norm_output,
        "lm_head_weight": tail_vectors / f"{LM_HEAD_PREFIX}_weight_words32.hex",
        "lm_head_scale": tail_vectors / f"{LM_HEAD_PREFIX}_scale_words32.hex",
        "lm_head_logits": tail_vectors / f"{LM_HEAD_PREFIX}_expected_scan_logits_q26.hex",
        "tail_binary": tail_binary,
        "tail_image": tail_image_hex,
        "tail_expected": tail_expected_hex,
    }
    for path in required_files.values():
        require_file(path)

    manifest = {
        "format_version": 1,
        "name": "tied_q4_embedding_true3_full_vocab_final_chain",
        "token_id": load_json(layer0_root / "embedding" / "embedding_meta.json")["token_id"],
        "layer_count": 3,
        "addresses": {
            "layer0_output": LAYER0_OUTPUT_BASE,
            "layer1_output": LAYER1_OUTPUT_BASE,
            "layer2_output": LAYER2_OUTPUT_BASE,
            "final_tail_qmap": FINAL_TAIL_QMAP_BASE,
            "final_hidden_runtime_override": LAYER2_OUTPUT_BASE,
        },
        "final": {
            "norm_prefix": FINAL_NORM_PREFIX,
            "lm_head_prefix": LM_HEAD_PREFIX,
            "best_token": int(tail_manifest["expected"]["best_token"]),
            "best_score_q26": int(tail_manifest["expected"]["best_score_q26"]),
            "vocab_size": int(tail_manifest["shape"]["vocab_size"]),
        },
        "files": {
            name: {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in required_files.items()
        },
        "commands": commands,
    }
    manifest_path = output_dir / "true3_final_chain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")

    print("PASS: exported tied-Q4 embedding through three complete layers and final tail")
    print(f"output_dir={output_dir}")
    print(f"layer2_output=0x{LAYER2_OUTPUT_BASE:016X}")
    print(
        "final="
        f"token {manifest['final']['best_token']} "
        f"score_q26 {manifest['final']['best_score_q26']}"
    )


if __name__ == "__main__":
    main()
