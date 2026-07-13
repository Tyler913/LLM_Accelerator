from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_ROOT = REPO_ROOT / "Temp"
SIM_VECTOR_DIR = REPO_ROOT / "FPGA_Project" / "sim" / "vectors"
Q4_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
QMAP_VECTOR_DIR = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_qmap_v1"

QKV_QMAP_BASE0 = 0x0000_0004_0008_0000
BODY_QMAP_BASE0 = 0x0000_0004_0500_0000
LAYER_QMAP_STRIDE = 0x0000_0000_1000_0000
QKV_QMAP_STRIDE = LAYER_QMAP_STRIDE
BODY_QMAP_STRIDE = LAYER_QMAP_STRIDE
WEIGHT_WINDOW_BASE0 = 0x0000_0004_0600_0000
WEIGHT_WINDOW_STRIDE = 0x0000_0000_0100_0000
O_PROJ_WEIGHT_OFFSET = 0x0000_0000
O_PROJ_SCALE_OFFSET = 0x0010_0000
MLP_GATE_WEIGHT_OFFSET = 0x0020_0000
MLP_GATE_SCALE_OFFSET = 0x0038_0000
MLP_UP_WEIGHT_OFFSET = 0x0040_0000
MLP_UP_SCALE_OFFSET = 0x0058_0000
MLP_DOWN_WEIGHT_OFFSET = 0x0060_0000
MLP_DOWN_SCALE_OFFSET = 0x0078_0000
INPUT_RMSNORM_OUTPUT_OFFSET = 0x0000_2540
DESCRIPTOR_TABLE_WORD_OFFSET = 0x0100 // 4
DESCRIPTOR_WORDS = 32
DESC_BASE_LO_WORD = 8
DESC_BASE_HI_WORD = 9


def parse_int_auto(text: str) -> int:
    return int(text.replace("_", ""), 0)


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def format_addr(value: int) -> str:
    text = f"{value:016x}"
    return "0x" + "_".join(text[i : i + 4] for i in range(0, len(text), 4))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LayerBases:
    input_rmsnorm: int
    qkv: int
    attn_frontend: int
    attn_score_value: int
    o_proj: int
    post_attn_norm: int
    mlp_gate_up: int
    mlp_silu_mul: int
    mlp_down: int
    mlp_residual_add: int
    o_proj_weight: int
    o_proj_scale: int
    mlp_gate_weight: int
    mlp_gate_scale: int
    mlp_up_weight: int
    mlp_up_scale: int
    mlp_down_weight: int
    mlp_down_scale: int


@dataclass(frozen=True)
class LayerPrefixes:
    input_norm_prefix: str
    input_norm_qmap_prefix: str
    qkv_prefix: str
    qk_prefix: str
    kv_prefix: str
    base_score_prefix: str
    base_value_prefix: str
    score_prefix: str
    value_prefix: str
    frontend_qmap_prefix: str
    score_value_qmap_prefix: str
    o_proj_prefix: str
    o_proj_qmap_prefix: str
    post_prefix: str
    post_qmap_prefix: str
    gate_prefix: str
    gate_qmap_prefix: str
    silu_prefix: str
    silu_qmap_prefix: str
    down_prefix: str
    down_qmap_prefix: str
    residual_prefix: str
    residual_qmap_prefix: str


def layer_bases(layer_id: int) -> LayerBases:
    qkv_offset = layer_id * QKV_QMAP_STRIDE
    body_offset = layer_id * BODY_QMAP_STRIDE
    weight_base = WEIGHT_WINDOW_BASE0 + (layer_id * WEIGHT_WINDOW_STRIDE)
    body_base = BODY_QMAP_BASE0 + body_offset
    return LayerBases(
        input_rmsnorm=body_base + 0x000A_0000,
        qkv=QKV_QMAP_BASE0 + qkv_offset,
        attn_frontend=body_base + 0x0002_0000,
        attn_score_value=body_base + 0x0003_0000,
        o_proj=body_base + 0x0004_0000,
        post_attn_norm=body_base + 0x0005_0000,
        mlp_gate_up=body_base + 0x0006_0000,
        mlp_silu_mul=body_base + 0x0007_0000,
        mlp_down=body_base + 0x0008_0000,
        mlp_residual_add=body_base + 0x0009_0000,
        o_proj_weight=weight_base + O_PROJ_WEIGHT_OFFSET,
        o_proj_scale=weight_base + O_PROJ_SCALE_OFFSET,
        mlp_gate_weight=weight_base + MLP_GATE_WEIGHT_OFFSET,
        mlp_gate_scale=weight_base + MLP_GATE_SCALE_OFFSET,
        mlp_up_weight=weight_base + MLP_UP_WEIGHT_OFFSET,
        mlp_up_scale=weight_base + MLP_UP_SCALE_OFFSET,
        mlp_down_weight=weight_base + MLP_DOWN_WEIGHT_OFFSET,
        mlp_down_scale=weight_base + MLP_DOWN_SCALE_OFFSET,
    )


def layer_prefixes(layer_id: int) -> LayerPrefixes:
    qkv_prefix = (
        "layer0_qkv_from_embedding_rmsnorm_full"
        if layer_id == 0
        else f"layer{layer_id}_qkv_from_layer{layer_id - 1}_rtl_full"
    )
    return LayerPrefixes(
        input_norm_prefix=f"layer{layer_id}_chained_input_rmsnorm_stage_real",
        input_norm_qmap_prefix=f"qmap_layer{layer_id}_chained_input_rmsnorm",
        qkv_prefix=qkv_prefix,
        qk_prefix=f"layer{layer_id}_chained_qk_norm_rope_stage_128_real",
        kv_prefix=f"layer{layer_id}_chained_kv_cache_append_real",
        base_score_prefix=f"layer{layer_id}_attention_score_stage_real",
        base_value_prefix=f"layer{layer_id}_attention_softmax_value_stage_real",
        score_prefix=f"layer{layer_id}_chained_attention_score_stage_real",
        value_prefix=f"layer{layer_id}_chained_attention_softmax_value_stage_real",
        frontend_qmap_prefix=f"qmap_layer{layer_id}_chained_attention_frontend",
        score_value_qmap_prefix=f"qmap_layer{layer_id}_chained_attention_score_value",
        o_proj_prefix=f"layer{layer_id}_chained_o_proj_stage_real",
        o_proj_qmap_prefix=f"qmap_layer{layer_id}_chained_o_proj",
        post_prefix=f"layer{layer_id}_chained_post_attention_residual_norm_stage_real",
        post_qmap_prefix=f"qmap_layer{layer_id}_chained_post_attention_residual_norm",
        gate_prefix=f"layer{layer_id}_chained_mlp_gate_up_proj_stage_real",
        gate_qmap_prefix=f"qmap_layer{layer_id}_chained_mlp_gate_up",
        silu_prefix=f"layer{layer_id}_chained_mlp_silu_mul_stage_real",
        silu_qmap_prefix=f"qmap_layer{layer_id}_chained_mlp_silu_mul",
        down_prefix=f"layer{layer_id}_chained_mlp_down_proj_stage_real",
        down_qmap_prefix=f"qmap_layer{layer_id}_chained_mlp_down",
        residual_prefix=f"layer{layer_id}_chained_mlp_residual_add_stage_real",
        residual_qmap_prefix=f"qmap_layer{layer_id}_chained_mlp_residual_add",
    )


def default_previous_layer_hex(layer_id: int) -> Path:
    if layer_id <= 0:
        raise ValueError("chained export requires layer-id >= 1")
    if layer_id == 1:
        return SIM_VECTOR_DIR / "qmap_mlp_residual_add_expected_words32.hex"
    return SIM_VECTOR_DIR / f"qmap_layer{layer_id - 1}_chained_mlp_residual_add_expected_words32.hex"


def default_previous_layer_label(layer_id: int) -> str:
    if layer_id == 0:
        return "external_layer0_input_hidden"
    if layer_id == 1:
        return "layer0_qmap_mlp_residual_add"
    return f"layer{layer_id - 1}_chained_mlp_residual_add"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--layer-id", type=int, default=1, help="Target decoder layer to chain")
    parser.add_argument(
        "--previous-layer-output-hex",
        type=Path,
        default=None,
        help="I32_Q12.12 layer_out hex from the previous RTL/golden layer",
    )
    parser.add_argument(
        "--previous-layer-label",
        type=str,
        default=None,
        help="Human-readable source label stored in generated manifests",
    )
    parser.add_argument("--position", type=int, default=-1, help="Prompt position for Q4 vector export")
    parser.add_argument(
        "--force-q4",
        action="store_true",
        help="Regenerate qkv_layerN_last_token_q4.npz even when it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exporter commands and manifest plan without running them",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output chain manifest JSON",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional Temp-contained root for all generated sim/QMAP files",
    )
    parser.add_argument(
        "--input-hidden-base",
        type=parse_int_auto,
        default=None,
        help="Optional producer-written hidden-buffer address for input RMSNorm",
    )
    parser.add_argument(
        "--qkv-npz",
        type=Path,
        default=None,
        help="Per-layer QKV Q4 NPZ; defaults to qkv_layerN_last_token_q4.npz",
    )
    parser.add_argument("--qkv-qmap-base0", type=parse_int_auto, default=QKV_QMAP_BASE0)
    parser.add_argument("--body-qmap-base0", type=parse_int_auto, default=BODY_QMAP_BASE0)
    parser.add_argument("--layer-qmap-stride", type=parse_int_auto, default=LAYER_QMAP_STRIDE)
    parser.add_argument(
        "--qkv-qmap-stride",
        type=parse_int_auto,
        default=None,
        help="Optional QKV packet stride; defaults to --layer-qmap-stride",
    )
    parser.add_argument(
        "--body-qmap-stride",
        type=parse_int_auto,
        default=None,
        help="Optional body packet stride; defaults to --layer-qmap-stride",
    )
    parser.add_argument("--weight-window-base0", type=parse_int_auto, default=WEIGHT_WINDOW_BASE0)
    parser.add_argument("--weight-window-stride", type=parse_int_auto, default=WEIGHT_WINDOW_STRIDE)
    parser.add_argument("--o-proj-weight-offset", type=parse_int_auto, default=O_PROJ_WEIGHT_OFFSET)
    parser.add_argument("--o-proj-scale-offset", type=parse_int_auto, default=O_PROJ_SCALE_OFFSET)
    parser.add_argument("--mlp-gate-weight-offset", type=parse_int_auto, default=MLP_GATE_WEIGHT_OFFSET)
    parser.add_argument("--mlp-gate-scale-offset", type=parse_int_auto, default=MLP_GATE_SCALE_OFFSET)
    parser.add_argument("--mlp-up-weight-offset", type=parse_int_auto, default=MLP_UP_WEIGHT_OFFSET)
    parser.add_argument("--mlp-up-scale-offset", type=parse_int_auto, default=MLP_UP_SCALE_OFFSET)
    parser.add_argument("--mlp-down-weight-offset", type=parse_int_auto, default=MLP_DOWN_WEIGHT_OFFSET)
    parser.add_argument("--mlp-down-scale-offset", type=parse_int_auto, default=MLP_DOWN_SCALE_OFFSET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one chained per-layer QMAP artifact set from a previous RTL/golden layer_out."
    )
    add_common_args(parser)
    return parser.parse_args()


def run_command(cmd: list[str], *, dry_run: bool, commands: list[list[str]]) -> None:
    commands.append(cmd)
    printable = " ".join(cmd)
    if dry_run:
        print(f"DRY-RUN: {printable}")
        return
    print(f"RUN: {printable}", flush=True)
    env = os.environ.copy()
    env["QMAP_SIM_VECTOR_DIR"] = str(SIM_VECTOR_DIR.resolve())
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def script(name: str) -> str:
    return str(SCRIPT_DIR / name)


def py_cmd(name: str, *args: str | Path | int) -> list[str]:
    cmd = [sys.executable, script(name)]
    cmd.extend(str(arg) for arg in args)
    return cmd


def qmap_bin(name: str) -> Path:
    return QMAP_VECTOR_DIR / f"{name}.qmap.bin"


def qmap_manifest(name: str) -> Path:
    return QMAP_VECTOR_DIR / f"{name}_manifest.json"


def sim_hex(name: str) -> Path:
    return SIM_VECTOR_DIR / f"{name}_words32.hex"


def require_temp_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(TEMP_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output root must stay under {TEMP_ROOT.resolve()}") from exc
    return resolved


def descriptor_base_from_hex(image_hex: Path, slot: int) -> int:
    lo_index = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_BASE_LO_WORD
    hi_index = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_BASE_HI_WORD
    needed = hi_index + 1
    words: list[int] = []
    with image_hex.open("r", encoding="ascii") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                words.append(int(stripped, 16))
                if len(words) >= needed:
                    break
    if len(words) < needed:
        raise RuntimeError(f"descriptor table is truncated in {image_hex}")
    return words[lo_index] | (words[hi_index] << 32)


def build_plan(args: argparse.Namespace) -> tuple[LayerBases, LayerPrefixes, dict[str, Any]]:
    global QKV_QMAP_BASE0, BODY_QMAP_BASE0, LAYER_QMAP_STRIDE
    global QKV_QMAP_STRIDE, BODY_QMAP_STRIDE
    global WEIGHT_WINDOW_BASE0, WEIGHT_WINDOW_STRIDE, SIM_VECTOR_DIR, QMAP_VECTOR_DIR
    global O_PROJ_WEIGHT_OFFSET, O_PROJ_SCALE_OFFSET
    global MLP_GATE_WEIGHT_OFFSET, MLP_GATE_SCALE_OFFSET
    global MLP_UP_WEIGHT_OFFSET, MLP_UP_SCALE_OFFSET
    global MLP_DOWN_WEIGHT_OFFSET, MLP_DOWN_SCALE_OFFSET

    if args.layer_id < 0:
        raise ValueError("chained export requires --layer-id >= 0")
    if args.layer_id == 0 and args.previous_layer_output_hex is None:
        raise ValueError("Layer 0 chained export requires --previous-layer-output-hex")

    if args.output_root is not None:
        output_root = require_temp_path(args.output_root)
        SIM_VECTOR_DIR = output_root / "sim_vectors"
        QMAP_VECTOR_DIR = output_root / "qmap"
        if not args.dry_run:
            SIM_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
            QMAP_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    QKV_QMAP_BASE0 = int(args.qkv_qmap_base0)
    BODY_QMAP_BASE0 = int(args.body_qmap_base0)
    LAYER_QMAP_STRIDE = int(args.layer_qmap_stride)
    QKV_QMAP_STRIDE = int(args.qkv_qmap_stride or LAYER_QMAP_STRIDE)
    BODY_QMAP_STRIDE = int(args.body_qmap_stride or LAYER_QMAP_STRIDE)
    WEIGHT_WINDOW_BASE0 = int(args.weight_window_base0)
    WEIGHT_WINDOW_STRIDE = int(args.weight_window_stride)
    O_PROJ_WEIGHT_OFFSET = int(args.o_proj_weight_offset)
    O_PROJ_SCALE_OFFSET = int(args.o_proj_scale_offset)
    MLP_GATE_WEIGHT_OFFSET = int(args.mlp_gate_weight_offset)
    MLP_GATE_SCALE_OFFSET = int(args.mlp_gate_scale_offset)
    MLP_UP_WEIGHT_OFFSET = int(args.mlp_up_weight_offset)
    MLP_UP_SCALE_OFFSET = int(args.mlp_up_scale_offset)
    MLP_DOWN_WEIGHT_OFFSET = int(args.mlp_down_weight_offset)
    MLP_DOWN_SCALE_OFFSET = int(args.mlp_down_scale_offset)

    bases = layer_bases(args.layer_id)
    prefixes = layer_prefixes(args.layer_id)
    qkv_npz = args.qkv_npz or (Q4_VECTOR_DIR / f"qkv_layer{args.layer_id}_last_token_q4.npz")
    qkv_manifest = (
        qkv_npz.with_name(f"qkv_layer{args.layer_id}_last_token_q4_manifest.json")
        if args.qkv_npz is not None
        else Q4_VECTOR_DIR / f"qkv_layer{args.layer_id}_last_token_q4_manifest.json"
    )
    previous_hex = args.previous_layer_output_hex or default_previous_layer_hex(args.layer_id)
    previous_label = args.previous_layer_label or default_previous_layer_label(args.layer_id)
    manifest = args.manifest or (QMAP_VECTOR_DIR / f"layer{args.layer_id}_chained_layer_manifest.json")

    paths: dict[str, Any] = {
        "manifest": manifest,
        "qkv_npz": qkv_npz,
        "qkv_manifest": qkv_manifest,
        "previous_layer_output_hex": previous_hex,
        "previous_layer_label": previous_label,
        "output_root": require_temp_path(args.output_root) if args.output_root is not None else None,
        "input_norm": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_input_rmsnorm_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_input_rmsnorm_runtime"),
            "sim_hex": sim_hex(f"{prefixes.input_norm_qmap_prefix}_image"),
            "expected_hex": sim_hex(f"{prefixes.input_norm_qmap_prefix}_expected"),
        },
        "qkv": {
            "binary": qmap_bin(prefixes.qkv_prefix),
            "manifest": QMAP_VECTOR_DIR / f"{prefixes.qkv_prefix}_manifest.json",
            "sim_hex": QMAP_VECTOR_DIR / f"{prefixes.qkv_prefix}_image_words32.hex",
            "expected_hex": QMAP_VECTOR_DIR / f"{prefixes.qkv_prefix}_expected_words32.hex",
        },
        "qk_prefix": prefixes.qk_prefix,
        "kv_prefix": prefixes.kv_prefix,
        "base_score_prefix": prefixes.base_score_prefix,
        "base_value_prefix": prefixes.base_value_prefix,
        "frontend": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_attention_frontend_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_attention_frontend_runtime"),
            "sim_hex": sim_hex(f"{prefixes.frontend_qmap_prefix}_image"),
            "q_rope_expected_hex": sim_hex(f"{prefixes.frontend_qmap_prefix}_q_rope_expected"),
        },
        "score_value": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_attention_score_value_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_attention_score_value_runtime"),
            "sim_hex": sim_hex(f"{prefixes.score_value_qmap_prefix}_image"),
            "attn_out_expected_hex": sim_hex(f"{prefixes.score_value_qmap_prefix}_attn_out_expected"),
        },
        "o_proj": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_o_proj_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_o_proj_runtime"),
            "sim_hex": sim_hex(f"{prefixes.o_proj_qmap_prefix}_image"),
            "expected_hex": sim_hex(f"{prefixes.o_proj_qmap_prefix}_expected"),
        },
        "post": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_post_attention_residual_norm_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_post_attention_residual_norm_runtime"),
            "sim_hex": sim_hex(f"{prefixes.post_qmap_prefix}_image"),
            "expected_hidden_hex": sim_hex(f"{prefixes.post_qmap_prefix}_expected_hidden"),
            "expected_norm_hex": sim_hex(f"{prefixes.post_qmap_prefix}_expected_norm"),
        },
        "gate": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_mlp_gate_up_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_mlp_gate_up_runtime"),
            "sim_hex": sim_hex(f"{prefixes.gate_qmap_prefix}_image"),
            "expected_gate_hex": sim_hex(f"{prefixes.gate_qmap_prefix}_expected_gate"),
            "expected_up_hex": sim_hex(f"{prefixes.gate_qmap_prefix}_expected_up"),
        },
        "silu": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_mlp_silu_mul_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_mlp_silu_mul_runtime"),
            "sim_hex": sim_hex(f"{prefixes.silu_qmap_prefix}_image"),
            "expected_hex": sim_hex(f"{prefixes.silu_qmap_prefix}_expected_hidden"),
        },
        "down": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_mlp_down_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_mlp_down_runtime"),
            "sim_hex": sim_hex(f"{prefixes.down_qmap_prefix}_image"),
            "expected_hex": sim_hex(f"{prefixes.down_qmap_prefix}_expected"),
        },
        "residual": {
            "binary": qmap_bin(f"layer{args.layer_id}_chained_mlp_residual_add_runtime"),
            "manifest": qmap_manifest(f"layer{args.layer_id}_chained_mlp_residual_add_runtime"),
            "sim_hex": sim_hex(f"{prefixes.residual_qmap_prefix}_image"),
            "expected_hex": sim_hex(f"{prefixes.residual_qmap_prefix}_expected"),
        },
    }
    return bases, prefixes, paths


def emit_commands(
    args: argparse.Namespace,
    bases: LayerBases,
    prefixes: LayerPrefixes,
    paths: dict[str, Any],
) -> list[list[str]]:
    commands: list[list[str]] = []
    qkv_npz = Path(paths["qkv_npz"])
    previous_hex = Path(paths["previous_layer_output_hex"])
    previous_label = str(paths["previous_layer_label"])

    if not previous_hex.is_file() and not args.dry_run:
        raise FileNotFoundError(previous_hex)

    if args.force_q4 or not qkv_npz.is_file():
        cmd = py_cmd(
            "45_export_layer_qkv_q4_vectors.py",
            "--layer-id",
            args.layer_id,
            "--position",
            args.position,
            "--output",
            qkv_npz,
            "--manifest",
            paths["qkv_manifest"],
        )
        run_command(cmd, dry_run=args.dry_run, commands=commands)
    else:
        print(f"SKIP: existing {relpath(qkv_npz)}", flush=True)

    input_norm = paths["input_norm"]
    run_command(
        py_cmd(
            "17_export_rmsnorm_fixed_vectors.py",
            "--layer-id",
            args.layer_id,
            "--prefix",
            prefixes.input_norm_prefix,
            "--input-hex",
            previous_hex,
            "--input-hex-width",
            32,
            "--input-source-name",
            previous_label,
            "--output-dir",
            SIM_VECTOR_DIR,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    input_norm_cmd = py_cmd(
        "48_export_qmap_input_rmsnorm_image.py",
        "--prefix",
        prefixes.input_norm_prefix,
        "--vector-dir",
        SIM_VECTOR_DIR,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.input_rmsnorm),
        "--output",
        input_norm["binary"],
        "--manifest",
        input_norm["manifest"],
        "--sim-hex",
        input_norm["sim_hex"],
        "--expected-hex",
        input_norm["expected_hex"],
    )
    if args.input_hidden_base is not None:
        input_norm_cmd.extend(["--hidden-base-addr", hex(args.input_hidden_base)])
    run_command(
        input_norm_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    qkv = paths["qkv"]
    run_command(
        py_cmd(
            "21_export_qmap_qkv_projection_image.py",
            "--input",
            qkv_npz,
            "--layer-id",
            args.layer_id,
            "--qmap-base",
            hex(bases.qkv),
            "--activation-hex",
            input_norm["expected_hex"],
            "--activation-source-name",
            prefixes.input_norm_prefix,
            "--activation-base-addr",
            hex(bases.input_rmsnorm + INPUT_RMSNORM_OUTPUT_OFFSET),
            "--output",
            qkv["binary"],
            "--manifest",
            qkv["manifest"],
            "--sim-hex",
            qkv["sim_hex"],
            "--expected-hex",
            qkv["expected_hex"],
        ),
        dry_run=args.dry_run,
        commands=commands,
    )

    run_command(
        py_cmd(
            "22_export_qk_norm_rope_fixed_vectors.py",
            "--layer-id",
            args.layer_id,
            "--input",
            qkv_npz,
            "--prefix",
            prefixes.qk_prefix,
            "--qkv-expected-hex",
            qkv["expected_hex"],
            "--source-label",
            prefixes.qkv_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    run_command(
        py_cmd(
            "23_export_kv_cache_append_vectors.py",
            "--layer-id",
            args.layer_id,
            "--input",
            qkv_npz,
            "--prefix",
            prefixes.kv_prefix,
            "--qkv-expected-hex",
            qkv["expected_hex"],
            "--source-label",
            prefixes.qkv_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    frontend = paths["frontend"]
    frontend_cmd = py_cmd(
        "37_export_qmap_attention_frontend_image.py",
        "--qk-prefix",
        prefixes.qk_prefix,
        "--kv-prefix",
        prefixes.kv_prefix,
        "--qmap-base",
        hex(bases.attn_frontend),
        "--output",
        frontend["binary"],
        "--manifest",
        frontend["manifest"],
        "--sim-hex",
        frontend["sim_hex"],
        "--q-rope-expected-hex",
        frontend["q_rope_expected_hex"],
    )
    if not args.dry_run:
        frontend_cmd.extend(
            [
                "--q-base-addr",
                hex(descriptor_base_from_hex(qkv["sim_hex"], 8)),
                "--k-base-addr",
                hex(descriptor_base_from_hex(qkv["sim_hex"], 9)),
                "--v-base-addr",
                hex(descriptor_base_from_hex(qkv["sim_hex"], 10)),
            ]
        )
    run_command(
        frontend_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    run_command(
        py_cmd(
            "24_export_attention_score_vectors.py",
            "--layer-id",
            args.layer_id,
            "--input",
            qkv_npz,
            "--prefix",
            prefixes.base_score_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    run_command(
        py_cmd(
            "25_export_attention_softmax_value_vectors.py",
            "--layer-id",
            args.layer_id,
            "--input",
            qkv_npz,
            "--prefix",
            prefixes.base_value_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    run_command(
        py_cmd(
            "46_export_chained_attention_score_value_vectors.py",
            "--layer-id",
            args.layer_id,
            "--score-prefix",
            prefixes.score_prefix,
            "--value-prefix",
            prefixes.value_prefix,
            "--q-rope-hex",
            frontend["q_rope_expected_hex"],
            "--base-k-cache-hex",
            SIM_VECTOR_DIR / f"{prefixes.base_score_prefix}_k_cache.hex",
            "--base-v-cache-hex",
            SIM_VECTOR_DIR / f"{prefixes.base_value_prefix}_v_cache.hex",
            "--current-k-hex",
            SIM_VECTOR_DIR / f"{prefixes.kv_prefix}_k_input.hex",
            "--current-v-hex",
            SIM_VECTOR_DIR / f"{prefixes.kv_prefix}_v_input.hex",
            "--score-scale-hex",
            SIM_VECTOR_DIR / f"{prefixes.base_score_prefix}_score_scale_q0_31.hex",
            "--source-meta",
            SIM_VECTOR_DIR / f"{prefixes.qk_prefix}_meta.json",
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    score_value = paths["score_value"]
    score_value_cmd = py_cmd(
        "38_export_qmap_attention_score_value_image.py",
        "--score-prefix",
        prefixes.score_prefix,
        "--value-prefix",
        prefixes.value_prefix,
        "--qmap-base",
        hex(bases.attn_score_value),
        "--output",
        score_value["binary"],
        "--manifest",
        score_value["manifest"],
        "--sim-hex",
        score_value["sim_hex"],
        "--attn-out-expected-hex",
        score_value["attn_out_expected_hex"],
    )
    if not args.dry_run:
        score_value_cmd.extend(
            [
                "--q-rope-base-addr",
                hex(descriptor_base_from_hex(frontend["sim_hex"], 9)),
            ]
        )
    run_command(
        score_value_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    o_proj = paths["o_proj"]
    run_command(
        py_cmd(
            "26_export_o_proj_vectors.py",
            "--layer-id",
            args.layer_id,
            "--input",
            qkv_npz,
            "--prefix",
            prefixes.o_proj_prefix,
            "--activation-hex",
            score_value["attn_out_expected_hex"],
            "--activation-source-name",
            prefixes.score_value_qmap_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    o_proj_cmd = py_cmd(
        "39_export_qmap_o_proj_image.py",
        "--prefix",
        prefixes.o_proj_prefix,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.o_proj),
        "--weight-base",
        hex(bases.o_proj_weight),
        "--scale-base",
        hex(bases.o_proj_scale),
        "--output",
        o_proj["binary"],
        "--manifest",
        o_proj["manifest"],
        "--sim-hex",
        o_proj["sim_hex"],
        "--expected-hex",
        o_proj["expected_hex"],
    )
    if not args.dry_run:
        o_proj_cmd.extend(
            [
                "--activation-base-addr",
                hex(descriptor_base_from_hex(score_value["sim_hex"], 4)),
            ]
        )
    run_command(
        o_proj_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    post = paths["post"]
    run_command(
        py_cmd(
            "27_export_post_attention_residual_norm_vectors.py",
            "--layer-id",
            args.layer_id,
            "--o-proj-prefix",
            prefixes.o_proj_prefix,
            "--prefix",
            prefixes.post_prefix,
            "--residual-hex",
            previous_hex,
            "--residual-source-name",
            previous_label,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    post_cmd = py_cmd(
        "40_export_qmap_post_attention_residual_norm_image.py",
        "--prefix",
        prefixes.post_prefix,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.post_attn_norm),
        "--output",
        post["binary"],
        "--manifest",
        post["manifest"],
        "--sim-hex",
        post["sim_hex"],
        "--expected-hidden-hex",
        post["expected_hidden_hex"],
        "--expected-norm-hex",
        post["expected_norm_hex"],
    )
    if not args.dry_run:
        post_cmd.extend(
            [
                "--o-proj-base-addr",
                hex(descriptor_base_from_hex(o_proj["sim_hex"], 4)),
            ]
        )
        if args.input_hidden_base is not None:
            post_cmd.extend(["--residual-base-addr", hex(args.input_hidden_base)])
    run_command(
        post_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    gate = paths["gate"]
    run_command(
        py_cmd(
            "28_export_mlp_gate_up_vectors.py",
            "--layer-id",
            args.layer_id,
            "--post-norm-prefix",
            prefixes.post_prefix,
            "--prefix",
            prefixes.gate_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    gate_cmd = py_cmd(
        "41_export_qmap_mlp_gate_up_image.py",
        "--prefix",
        prefixes.gate_prefix,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.mlp_gate_up),
        "--gate-weight-base",
        hex(bases.mlp_gate_weight),
        "--gate-scale-base",
        hex(bases.mlp_gate_scale),
        "--up-weight-base",
        hex(bases.mlp_up_weight),
        "--up-scale-base",
        hex(bases.mlp_up_scale),
        "--output",
        gate["binary"],
        "--manifest",
        gate["manifest"],
        "--sim-hex",
        gate["sim_hex"],
        "--expected-gate-hex",
        gate["expected_gate_hex"],
        "--expected-up-hex",
        gate["expected_up_hex"],
    )
    if not args.dry_run:
        gate_cmd.extend(
            [
                "--activation-base-addr",
                hex(descriptor_base_from_hex(post["sim_hex"], 5)),
            ]
        )
    run_command(
        gate_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    silu = paths["silu"]
    run_command(
        py_cmd(
            "29_export_mlp_silu_mul_vectors.py",
            "--layer-id",
            args.layer_id,
            "--gate-up-prefix",
            prefixes.gate_prefix,
            "--prefix",
            prefixes.silu_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    silu_cmd = py_cmd(
        "42_export_qmap_mlp_silu_mul_image.py",
        "--prefix",
        prefixes.silu_prefix,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.mlp_silu_mul),
        "--gate-up-qmap-prefix",
        prefixes.gate_qmap_prefix,
        "--output",
        silu["binary"],
        "--manifest",
        silu["manifest"],
        "--sim-hex",
        silu["sim_hex"],
        "--expected-hex",
        silu["expected_hex"],
    )
    if not args.dry_run:
        silu_cmd.extend(
            [
                "--gate-base-addr",
                hex(descriptor_base_from_hex(gate["sim_hex"], 6)),
                "--up-base-addr",
                hex(descriptor_base_from_hex(gate["sim_hex"], 7)),
            ]
        )
    run_command(
        silu_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    down = paths["down"]
    run_command(
        py_cmd(
            "30_export_mlp_down_vectors.py",
            "--layer-id",
            args.layer_id,
            "--hidden-prefix",
            prefixes.silu_prefix,
            "--prefix",
            prefixes.down_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    down_cmd = py_cmd(
        "43_export_qmap_mlp_down_image.py",
        "--prefix",
        prefixes.down_prefix,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.mlp_down),
        "--weight-base",
        hex(bases.mlp_down_weight),
        "--scale-base",
        hex(bases.mlp_down_scale),
        "--silu-qmap-prefix",
        prefixes.silu_qmap_prefix,
        "--output",
        down["binary"],
        "--manifest",
        down["manifest"],
        "--sim-hex",
        down["sim_hex"],
        "--expected-hex",
        down["expected_hex"],
    )
    if not args.dry_run:
        down_cmd.extend(
            [
                "--activation-base-addr",
                hex(descriptor_base_from_hex(silu["sim_hex"], 4)),
            ]
        )
    run_command(
        down_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )

    residual = paths["residual"]
    run_command(
        py_cmd(
            "31_export_mlp_residual_add_vectors.py",
            "--layer-id",
            args.layer_id,
            "--post-attention-prefix",
            prefixes.post_prefix,
            "--down-prefix",
            prefixes.down_prefix,
            "--prefix",
            prefixes.residual_prefix,
        ),
        dry_run=args.dry_run,
        commands=commands,
    )
    residual_cmd = py_cmd(
        "44_export_qmap_mlp_residual_add_image.py",
        "--prefix",
        prefixes.residual_prefix,
        "--layer-id",
        args.layer_id,
        "--qmap-base",
        hex(bases.mlp_residual_add),
        "--post-attn-qmap-prefix",
        prefixes.post_qmap_prefix,
        "--down-qmap-prefix",
        prefixes.down_qmap_prefix,
        "--output",
        residual["binary"],
        "--manifest",
        residual["manifest"],
        "--sim-hex",
        residual["sim_hex"],
        "--expected-hex",
        residual["expected_hex"],
    )
    if not args.dry_run:
        residual_cmd.extend(
            [
                "--post-attn-base-addr",
                hex(descriptor_base_from_hex(post["sim_hex"], 4)),
                "--down-base-addr",
                hex(descriptor_base_from_hex(down["sim_hex"], 4)),
            ]
        )
    run_command(
        residual_cmd,
        dry_run=args.dry_run,
        commands=commands,
    )
    return commands


def serialize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, dict):
        return {key: serialize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_paths(item) for item in value]
    return value


def display_arg(arg: str) -> str:
    path = Path(arg)
    if path.is_absolute() and path.exists():
        try:
            return relpath(path)
        except ValueError:
            return str(path)
    return arg


def write_chain_manifest(
    args: argparse.Namespace,
    bases: LayerBases,
    prefixes: LayerPrefixes,
    paths: dict[str, Any],
    commands: list[list[str]],
) -> None:
    manifest_path = Path(paths["manifest"])
    files = serialize_paths(paths)
    hashes: dict[str, str] = {}
    if not args.dry_run:
        for stage_name, stage_paths in paths.items():
            if isinstance(stage_paths, dict):
                for key, path in stage_paths.items():
                    if isinstance(path, Path) and path.is_file():
                        hashes[f"{stage_name}.{key}"] = sha256_file(path)
    manifest = {
        "format_version": 1,
        "name": f"layer{args.layer_id}_chained_layer_qmap_artifacts",
        "layer_id": args.layer_id,
        "previous_layer": args.layer_id - 1,
        "previous_layer_output_hex": relpath(Path(paths["previous_layer_output_hex"])),
        "previous_layer_label": paths["previous_layer_label"],
        "address_formula": {
            "qkv_qmap_base0": format_addr(QKV_QMAP_BASE0),
            "body_qmap_base0": format_addr(BODY_QMAP_BASE0),
            "layer_qmap_stride": format_addr(LAYER_QMAP_STRIDE),
            "qkv_qmap_stride": format_addr(QKV_QMAP_STRIDE),
            "body_qmap_stride": format_addr(BODY_QMAP_STRIDE),
            "weight_window_base0": format_addr(WEIGHT_WINDOW_BASE0),
            "weight_window_stride": format_addr(WEIGHT_WINDOW_STRIDE),
            "persistent_offsets": {
                "o_proj_weight": format_addr(O_PROJ_WEIGHT_OFFSET),
                "o_proj_scale": format_addr(O_PROJ_SCALE_OFFSET),
                "mlp_gate_weight": format_addr(MLP_GATE_WEIGHT_OFFSET),
                "mlp_gate_scale": format_addr(MLP_GATE_SCALE_OFFSET),
                "mlp_up_weight": format_addr(MLP_UP_WEIGHT_OFFSET),
                "mlp_up_scale": format_addr(MLP_UP_SCALE_OFFSET),
                "mlp_down_weight": format_addr(MLP_DOWN_WEIGHT_OFFSET),
                "mlp_down_scale": format_addr(MLP_DOWN_SCALE_OFFSET),
            },
        },
        "qmap_bases": {
            "input_hidden": (
                format_addr(args.input_hidden_base)
                if args.input_hidden_base is not None
                else None
            ),
            "input_rmsnorm": format_addr(bases.input_rmsnorm),
            "input_rmsnorm_output": format_addr(
                bases.input_rmsnorm + INPUT_RMSNORM_OUTPUT_OFFSET
            ),
            "qkv": format_addr(bases.qkv),
            "attn_frontend": format_addr(bases.attn_frontend),
            "attn_score_value": format_addr(bases.attn_score_value),
            "o_proj": format_addr(bases.o_proj),
            "post_attn_norm": format_addr(bases.post_attn_norm),
            "mlp_gate_up": format_addr(bases.mlp_gate_up),
            "mlp_silu_mul": format_addr(bases.mlp_silu_mul),
            "mlp_down": format_addr(bases.mlp_down),
            "mlp_residual_add": format_addr(bases.mlp_residual_add),
        },
        "persistent_bases": {
            "o_proj_weight": format_addr(bases.o_proj_weight),
            "o_proj_scale": format_addr(bases.o_proj_scale),
            "mlp_gate_weight": format_addr(bases.mlp_gate_weight),
            "mlp_gate_scale": format_addr(bases.mlp_gate_scale),
            "mlp_up_weight": format_addr(bases.mlp_up_weight),
            "mlp_up_scale": format_addr(bases.mlp_up_scale),
            "mlp_down_weight": format_addr(bases.mlp_down_weight),
            "mlp_down_scale": format_addr(bases.mlp_down_scale),
        },
        "prefixes": prefixes.__dict__,
        "files": files,
        "sha256": hashes,
        "commands": [
            {
                "argv": [display_arg(part) for part in cmd]
            }
            for cmd in commands
        ],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2) + "\n")
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported chained layer manifest: {relpath(manifest_path)}")


def main() -> None:
    args = parse_args()
    bases, prefixes, paths = build_plan(args)
    commands = emit_commands(args, bases, prefixes, paths)
    write_chain_manifest(args, bases, prefixes, paths, commands)


if __name__ == "__main__":
    main()
