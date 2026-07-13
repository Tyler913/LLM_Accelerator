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
TEMP_ROOT = REPO_ROOT / "Temp"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = TEMP_ROOT / "embedding_full28_final_chain_vectors"
MAX_LAYERS = 28

PL_DDR_BASE = 0x0000_0004_0000_0000
PL_DDR_HIGH = 0x0000_0004_1FFF_FFFF
TIED_WEIGHT_BASE = 0x0000_0004_0010_0000
TIED_SCALE_BASE = 0x0000_0004_04B3_0000

LAYER_BLOCK_BASE = 0x0000_0004_0500_0000
LAYER_BLOCK_STRIDE = 0x0084_0000
BODY_QMAP_BASE = 0x0000_0004_1790_0000
BODY_QMAP_STRIDE = 0x0010_0000
KV_CACHE_BASE = 0x0000_0004_1410_0000
FINAL_TAIL_QMAP_BASE = 0x0000_0004_1950_0000

PERSISTENT_OFFSETS = {
    "o_proj_weight": 0x0023_0000,
    "o_proj_scale": 0x0033_0000,
    "mlp_gate_weight": 0x0034_0000,
    "mlp_gate_scale": 0x004C_0000,
    "mlp_up_weight": 0x004E_0000,
    "mlp_up_scale": 0x0066_0000,
    "mlp_down_weight": 0x0068_0000,
    "mlp_down_scale": 0x0080_0000,
}

QKV_IMAGE_BYTES = 0x0022_B000
OPROJ_WEIGHT_BYTES = 1024 * 1024
OPROJ_SCALE_BYTES = 1024 * 64
GATE_WEIGHT_BYTES = 3072 * 512
GATE_SCALE_BYTES = 3072 * 32
DOWN_WEIGHT_BYTES = 1024 * 1536
DOWN_SCALE_BYTES = 1024 * 96
TIED_VOCAB_SIZE = 151936
TIED_WEIGHT_BYTES = TIED_VOCAB_SIZE * 512
TIED_SCALE_BYTES = TIED_VOCAB_SIZE * 32

BODY_STAGE_LAYOUT = {
    "input_norm": (0x000A_0000, 0x0000_5000),
    "frontend": (0x0002_0000, 0x0000_8000),
    "score_value": (0x0003_0000, 0x0000_5000),
    "o_proj": (0x0004_0000, 0x0000_5000),
    "post": (0x0005_0000, 0x0000_8000),
    "gate": (0x0006_0000, 0x0000_E000),
    "silu": (0x0007_0000, 0x0000_E000),
    "down": (0x0008_0000, 0x0000_6000),
    "residual": (0x0009_0000, 0x0000_5000),
}

NUM_KV_HEADS = 8
MAX_CONTEXT = 256
HEAD_DIM = 128
KV_CACHE_BYTES_PER_LAYER = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * 4

DESCRIPTOR_TABLE_WORD_OFFSET = 0x0100 // 4
DESCRIPTOR_WORDS = 32
DESC_BASE_LO_WORD = 8
DESC_BASE_HI_WORD = 9


@dataclass(frozen=True)
class Interval:
    name: str
    base: int
    size: int

    @property
    def high(self) -> int:
        return self.base + self.size - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Temp-contained tied-Q4 embedding through a compact, "
            "collision-free decoder-layer chain and full-vocabulary final tail."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-id", type=int, default=None)
    parser.add_argument("--layer-count", type=int, default=MAX_LAYERS)
    parser.add_argument("--lm-chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Write and validate only the physical address plan without loading the model",
    )
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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative_output_path(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def layer_output_base(layer_id: int) -> int:
    return BODY_QMAP_BASE + (layer_id * BODY_QMAP_STRIDE) + 0x0009_2540


def layout_args() -> list[str | int]:
    return [
        "--qkv-qmap-base0",
        hex(LAYER_BLOCK_BASE),
        "--body-qmap-base0",
        hex(BODY_QMAP_BASE),
        "--qkv-qmap-stride",
        hex(LAYER_BLOCK_STRIDE),
        "--body-qmap-stride",
        hex(BODY_QMAP_STRIDE),
        "--weight-window-base0",
        hex(LAYER_BLOCK_BASE),
        "--weight-window-stride",
        hex(LAYER_BLOCK_STRIDE),
        "--o-proj-weight-offset",
        hex(PERSISTENT_OFFSETS["o_proj_weight"]),
        "--o-proj-scale-offset",
        hex(PERSISTENT_OFFSETS["o_proj_scale"]),
        "--mlp-gate-weight-offset",
        hex(PERSISTENT_OFFSETS["mlp_gate_weight"]),
        "--mlp-gate-scale-offset",
        hex(PERSISTENT_OFFSETS["mlp_gate_scale"]),
        "--mlp-up-weight-offset",
        hex(PERSISTENT_OFFSETS["mlp_up_weight"]),
        "--mlp-up-scale-offset",
        hex(PERSISTENT_OFFSETS["mlp_up_scale"]),
        "--mlp-down-weight-offset",
        hex(PERSISTENT_OFFSETS["mlp_down_weight"]),
        "--mlp-down-scale-offset",
        hex(PERSISTENT_OFFSETS["mlp_down_scale"]),
    ]


def layer_output_hex(layer_root: Path, layer_id: int) -> Path:
    return (
        layer_root
        / "sim_vectors"
        / f"qmap_layer{layer_id}_chained_mlp_residual_add_expected_words32.hex"
    )


def descriptor_base(image_hex: Path, slot: int) -> int:
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


def manifest_file(manifest: dict[str, Any], stage: str, key: str) -> Path:
    return require_file(resolve_repo_path(manifest["files"][stage][key]))


def build_layer_entry(
    *,
    layer_id: int,
    layer_root: Path,
    manifest_path: Path,
    input_hidden_base: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[Interval]]:
    manifest = load_json(manifest_path)
    if int(manifest["layer_id"]) != layer_id:
        raise RuntimeError(f"layer manifest id mismatch in {manifest_path}")

    prefixes = manifest["prefixes"]
    sim_dir = layer_root / "sim_vectors"
    files = {
        "input_norm_image": manifest_file(manifest, "input_norm", "sim_hex"),
        "qkv_image": manifest_file(manifest, "qkv", "sim_hex"),
        "frontend_image": manifest_file(manifest, "frontend", "sim_hex"),
        "score_image": manifest_file(manifest, "score_value", "sim_hex"),
        "oproj_image": manifest_file(manifest, "o_proj", "sim_hex"),
        "post_image": manifest_file(manifest, "post", "sim_hex"),
        "gate_image": manifest_file(manifest, "gate", "sim_hex"),
        "silu_image": manifest_file(manifest, "silu", "sim_hex"),
        "down_image": manifest_file(manifest, "down", "sim_hex"),
        "residual_image": manifest_file(manifest, "residual", "sim_hex"),
        "expected_input_norm": manifest_file(manifest, "input_norm", "expected_hex"),
        "expected_qkv": manifest_file(manifest, "qkv", "expected_hex"),
        "expected_q_rope": manifest_file(manifest, "frontend", "q_rope_expected_hex"),
        "expected_attn_out": manifest_file(manifest, "score_value", "attn_out_expected_hex"),
        "expected_oproj": manifest_file(manifest, "o_proj", "expected_hex"),
        "expected_post_hidden": manifest_file(manifest, "post", "expected_hidden_hex"),
        "expected_post_norm": manifest_file(manifest, "post", "expected_norm_hex"),
        "expected_gate": manifest_file(manifest, "gate", "expected_gate_hex"),
        "expected_up": manifest_file(manifest, "gate", "expected_up_hex"),
        "expected_silu": manifest_file(manifest, "silu", "expected_hex"),
        "expected_down": manifest_file(manifest, "down", "expected_hex"),
        "expected_layer": manifest_file(manifest, "residual", "expected_hex"),
        "k_cache": require_file(sim_dir / f"{prefixes['score_prefix']}_k_cache.hex"),
        "v_cache": require_file(sim_dir / f"{prefixes['value_prefix']}_v_cache.hex"),
        "expected_cache_addr": require_file(
            sim_dir / f"{prefixes['kv_prefix']}_expected_addr.hex"
        ),
        "expected_cache_data": require_file(
            sim_dir / f"{prefixes['kv_prefix']}_expected_data.hex"
        ),
        "expected_cache_kind": require_file(
            sim_dir / f"{prefixes['kv_prefix']}_expected_kind.hex"
        ),
        "oproj_weight": require_file(
            sim_dir / f"{prefixes['o_proj_prefix']}_weight_words32.hex"
        ),
        "oproj_scale": require_file(
            sim_dir / f"{prefixes['o_proj_prefix']}_scale_words32.hex"
        ),
        "gate_weight": require_file(
            sim_dir / f"{prefixes['gate_prefix']}_gate_weight_words32.hex"
        ),
        "gate_scale": require_file(
            sim_dir / f"{prefixes['gate_prefix']}_gate_scale_words32.hex"
        ),
        "up_weight": require_file(
            sim_dir / f"{prefixes['gate_prefix']}_up_weight_words32.hex"
        ),
        "up_scale": require_file(
            sim_dir / f"{prefixes['gate_prefix']}_up_scale_words32.hex"
        ),
        "down_weight": require_file(
            sim_dir / f"{prefixes['down_prefix']}_weight_words32.hex"
        ),
        "down_scale": require_file(
            sim_dir / f"{prefixes['down_prefix']}_scale_words32.hex"
        ),
    }

    qmap_bases = {name: parse_addr(value) for name, value in manifest["qmap_bases"].items()}
    persistent_bases = {
        name: parse_addr(value) for name, value in manifest["persistent_bases"].items()
    }

    write_bases = {
        "input_norm": descriptor_base(files["input_norm_image"], 3),
        "q": descriptor_base(files["qkv_image"], 8),
        "k": descriptor_base(files["qkv_image"], 9),
        "v": descriptor_base(files["qkv_image"], 10),
        "q_rope": descriptor_base(files["frontend_image"], 9),
        "attn_out": descriptor_base(files["score_image"], 4),
        "o_proj": descriptor_base(files["oproj_image"], 4),
        "post_hidden": descriptor_base(files["post_image"], 4),
        "post_norm": descriptor_base(files["post_image"], 5),
        "gate": descriptor_base(files["gate_image"], 6),
        "up": descriptor_base(files["gate_image"], 7),
        "silu": descriptor_base(files["silu_image"], 4),
        "down": descriptor_base(files["down_image"], 4),
        "layer": descriptor_base(files["residual_image"], 3),
    }

    contracts = [
        ("input hidden", descriptor_base(files["input_norm_image"], 1), input_hidden_base),
        ("QKV activation", descriptor_base(files["qkv_image"], 1), write_bases["input_norm"]),
        ("frontend Q", descriptor_base(files["frontend_image"], 1), write_bases["q"]),
        ("frontend K", descriptor_base(files["frontend_image"], 2), write_bases["k"]),
        ("frontend V", descriptor_base(files["frontend_image"], 3), write_bases["v"]),
        ("score Q", descriptor_base(files["score_image"], 1), write_bases["q_rope"]),
        ("o_proj activation", descriptor_base(files["oproj_image"], 1), write_bases["attn_out"]),
        ("post residual", descriptor_base(files["post_image"], 1), input_hidden_base),
        ("post o_proj", descriptor_base(files["post_image"], 2), write_bases["o_proj"]),
        ("gate activation", descriptor_base(files["gate_image"], 1), write_bases["post_norm"]),
        ("silu gate", descriptor_base(files["silu_image"], 1), write_bases["gate"]),
        ("silu up", descriptor_base(files["silu_image"], 2), write_bases["up"]),
        ("down activation", descriptor_base(files["down_image"], 1), write_bases["silu"]),
        ("residual hidden", descriptor_base(files["residual_image"], 1), write_bases["post_hidden"]),
        ("residual down", descriptor_base(files["residual_image"], 2), write_bases["down"]),
        ("layer output", write_bases["layer"], layer_output_base(layer_id)),
    ]
    for label, actual, expected in contracts:
        if actual != expected:
            raise RuntimeError(
                f"Layer {layer_id} {label} contract mismatch: "
                f"0x{actual:016X} != 0x{expected:016X}"
            )

    cache_low = KV_CACHE_BASE + (layer_id * KV_CACHE_BYTES_PER_LAYER)
    cache_high = cache_low + KV_CACHE_BYTES_PER_LAYER
    cache_addresses: list[int] = []
    with files["expected_cache_addr"].open("r", encoding="ascii") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                cache_addresses.append(int(stripped, 16))
    if len(cache_addresses) != 2048:
        raise RuntimeError(f"Layer {layer_id} cache address count is {len(cache_addresses)}, expected 2048")
    if any(address < cache_low or address >= cache_high for address in cache_addresses):
        raise RuntimeError(f"Layer {layer_id} cache writes escape its physical cache slice")

    binary_files = {
        stage: manifest_file(manifest, stage, "binary")
        for stage in (
            "input_norm",
            "qkv",
            "frontend",
            "score_value",
            "o_proj",
            "post",
            "gate",
            "silu",
            "down",
            "residual",
        )
    }
    stage_base_keys = {
        "input_norm": "input_rmsnorm",
        "qkv": "qkv",
        "frontend": "attn_frontend",
        "score_value": "attn_score_value",
        "o_proj": "o_proj",
        "post": "post_attn_norm",
        "gate": "mlp_gate_up",
        "silu": "mlp_silu_mul",
        "down": "mlp_down",
        "residual": "mlp_residual_add",
    }
    intervals = [
        Interval(
            f"layer{layer_id}.{stage}",
            qmap_bases[stage_base_keys[stage]],
            binary_files[stage].stat().st_size,
        )
        for stage in binary_files
    ]
    intervals.extend(
        [
            Interval(f"layer{layer_id}.o_proj_weight", persistent_bases["o_proj_weight"], OPROJ_WEIGHT_BYTES),
            Interval(f"layer{layer_id}.o_proj_scale", persistent_bases["o_proj_scale"], OPROJ_SCALE_BYTES),
            Interval(f"layer{layer_id}.gate_weight", persistent_bases["mlp_gate_weight"], GATE_WEIGHT_BYTES),
            Interval(f"layer{layer_id}.gate_scale", persistent_bases["mlp_gate_scale"], GATE_SCALE_BYTES),
            Interval(f"layer{layer_id}.up_weight", persistent_bases["mlp_up_weight"], GATE_WEIGHT_BYTES),
            Interval(f"layer{layer_id}.up_scale", persistent_bases["mlp_up_scale"], GATE_SCALE_BYTES),
            Interval(f"layer{layer_id}.down_weight", persistent_bases["mlp_down_weight"], DOWN_WEIGHT_BYTES),
            Interval(f"layer{layer_id}.down_scale", persistent_bases["mlp_down_scale"], DOWN_SCALE_BYTES),
        ]
    )

    entry = {
        "layer_id": layer_id,
        "manifest": relative_output_path(manifest_path, output_dir),
        "input_hidden_base": input_hidden_base,
        "output_hidden_base": write_bases["layer"],
        "qmap_bases": qmap_bases,
        "persistent_bases": persistent_bases,
        "write_bases": write_bases,
        "cache_slice": {"base": cache_low, "bytes": KV_CACHE_BYTES_PER_LAYER},
        "files": {
            name: relative_output_path(path, output_dir) for name, path in files.items()
        },
    }
    return entry, intervals


def audit_intervals(intervals: list[Interval]) -> dict[str, Any]:
    ordered = sorted(intervals, key=lambda item: (item.base, item.high, item.name))
    for interval in ordered:
        if interval.size <= 0:
            raise RuntimeError(f"non-positive interval size: {interval.name}")
        if interval.base < PL_DDR_BASE or interval.high > PL_DDR_HIGH:
            raise RuntimeError(
                f"{interval.name} is outside PL DDR: "
                f"0x{interval.base:016X}..0x{interval.high:016X}"
            )
    for previous, current in zip(ordered, ordered[1:]):
        if current.base <= previous.high:
            raise RuntimeError(
                "physical address collision: "
                f"{previous.name} 0x{previous.base:016X}..0x{previous.high:016X} and "
                f"{current.name} 0x{current.base:016X}..0x{current.high:016X}"
            )
    return {
        "status": "PASS",
        "aperture": {"base": PL_DDR_BASE, "high": PL_DDR_HIGH},
        "interval_count": len(ordered),
        "lowest": ordered[0].base,
        "highest": ordered[-1].high,
        "intervals": [
            {"name": item.name, "base": item.base, "high": item.high, "bytes": item.size}
            for item in ordered
        ],
    }


def planned_intervals(layer_count: int) -> list[Interval]:
    intervals = [
        Interval("tied_lm_head_weight", TIED_WEIGHT_BASE, TIED_WEIGHT_BYTES),
        Interval("tied_lm_head_scale", TIED_SCALE_BASE, TIED_SCALE_BYTES),
        Interval("full_model_kv_cache", KV_CACHE_BASE, layer_count * KV_CACHE_BYTES_PER_LAYER),
        Interval("final_tail_qmap", FINAL_TAIL_QMAP_BASE, 0x0000_4000),
    ]
    persistent_sizes = {
        "o_proj_weight": OPROJ_WEIGHT_BYTES,
        "o_proj_scale": OPROJ_SCALE_BYTES,
        "mlp_gate_weight": GATE_WEIGHT_BYTES,
        "mlp_gate_scale": GATE_SCALE_BYTES,
        "mlp_up_weight": GATE_WEIGHT_BYTES,
        "mlp_up_scale": GATE_SCALE_BYTES,
        "mlp_down_weight": DOWN_WEIGHT_BYTES,
        "mlp_down_scale": DOWN_SCALE_BYTES,
    }
    for layer_id in range(layer_count):
        layer_block = LAYER_BLOCK_BASE + (layer_id * LAYER_BLOCK_STRIDE)
        body_base = BODY_QMAP_BASE + (layer_id * BODY_QMAP_STRIDE)
        intervals.append(Interval(f"layer{layer_id}.qkv", layer_block, QKV_IMAGE_BYTES))
        for name, size in persistent_sizes.items():
            intervals.append(
                Interval(
                    f"layer{layer_id}.{name}",
                    layer_block + PERSISTENT_OFFSETS[name],
                    size,
                )
            )
        for name, (offset, size) in BODY_STAGE_LAYOUT.items():
            intervals.append(Interval(f"layer{layer_id}.{name}", body_base + offset, size))
    return intervals


def main() -> None:
    args = parse_args()
    if args.layer_count < 1 or args.layer_count > MAX_LAYERS:
        raise ValueError(f"layer-count must be in range 1..{MAX_LAYERS}")

    output_dir = require_temp_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned_audit = audit_intervals(planned_intervals(args.layer_count))
    planned_audit_path = output_dir / "physical_address_plan.json"
    planned_audit_path.write_text(
        json.dumps(planned_audit, indent=2) + "\n", encoding="ascii"
    )
    if args.layout_only:
        print(
            f"PASS: compact physical address plan fits {args.layer_count} layers "
            "without overlap"
        )
        print(f"address_plan={planned_audit_path}")
        return

    layer0_bundle = output_dir / "embedding_layer0"
    tail_root = output_dir / "final_tail"
    tail_vectors = tail_root / "sim_vectors"
    tail_qmap = tail_root / "qmap"
    for path in (layer0_bundle, tail_vectors, tail_qmap):
        path.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    layer0_args: list[str | Path | int] = [
        "--output-dir",
        layer0_bundle,
        "--input-hidden-base",
        hex(layer_output_base(0)),
        *layout_args(),
    ]
    if args.token_id is not None:
        layer0_args.extend(["--token-id", args.token_id])
    commands.append(run_script("51_export_embedding_layer0_full_chain.py", *layer0_args))

    layer_roots: list[Path] = [layer0_bundle / "layer0"]
    layer_manifests: list[Path] = [
        layer_roots[0] / "qmap" / "layer0_chained_layer_manifest.json"
    ]
    previous_output = require_file(layer_output_hex(layer_roots[0], 0))

    source_q4_root = REPO_ROOT / "artifacts" / "test_vectors" / "qwen3_0p6b_q4_v0"
    for layer_id in range(1, args.layer_count):
        layer_root = output_dir / f"layer{layer_id:02d}"
        manifest_path = layer_root / "qmap" / f"layer{layer_id}_chained_layer_manifest.json"
        source_q4 = source_q4_root / f"qkv_layer{layer_id}_last_token_q4.npz"
        qkv_npz = source_q4 if source_q4.is_file() else (
            layer_root / "q4" / f"qkv_layer{layer_id}_last_token_q4.npz"
        )
        commands.append(
            run_script(
                "47_export_chained_layer_qmap_artifacts.py",
                "--layer-id",
                layer_id,
                "--previous-layer-output-hex",
                previous_output,
                "--previous-layer-label",
                f"tied_embedding_complete_layer{layer_id - 1}",
                "--input-hidden-base",
                hex(layer_output_base(layer_id - 1)),
                "--qkv-npz",
                qkv_npz,
                *layout_args(),
                "--output-root",
                layer_root,
                "--manifest",
                manifest_path,
            )
        )
        layer_roots.append(layer_root)
        layer_manifests.append(manifest_path)
        previous_output = require_file(layer_output_hex(layer_root, layer_id))

    final_norm_prefix = f"final_rmsnorm_from_embedding_full{args.layer_count}"
    lm_head_prefix = f"lm_head_full_vocab_from_embedding_full{args.layer_count}"
    tail_env = os.environ.copy()
    tail_env["QMAP_SIM_VECTOR_DIR"] = str(tail_vectors)

    commands.append(
        run_script(
            "32_export_final_rmsnorm_vectors.py",
            "--prefix",
            final_norm_prefix,
            "--input-hex",
            previous_output,
            "--input-hex-width",
            32,
            "--input-source-name",
            f"tied_embedding_complete_layer{args.layer_count - 1}",
            env=tail_env,
        )
    )
    final_norm_output = require_file(tail_vectors / f"{final_norm_prefix}_expected.hex")

    commands.append(
        run_script(
            "35_export_lm_head_full_vocab_vectors.py",
            "--prefix",
            lm_head_prefix,
            "--chunk-rows",
            args.lm_chunk_rows,
            "--activation-hex",
            final_norm_output,
            "--activation-source-name",
            final_norm_prefix,
            env=tail_env,
        )
    )

    tail_binary = tail_qmap / "final_token_tail_embedding_full_chain.qmap.bin"
    tail_manifest_path = tail_qmap / "final_token_tail_embedding_full_chain_manifest.json"
    tail_image_hex = tail_vectors / "qmap_final_token_tail_embedding_full_chain_image_words32.hex"
    tail_expected_hex = tail_vectors / "qmap_final_token_tail_embedding_full_chain_expected_words32.hex"
    commands.append(
        run_script(
            "36_export_qmap_final_token_tail_image.py",
            "--lm-prefix",
            lm_head_prefix,
            "--final-norm-prefix",
            final_norm_prefix,
            "--qmap-base",
            hex(FINAL_TAIL_QMAP_BASE),
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

    layer_entries: list[dict[str, Any]] = []
    intervals: list[Interval] = []
    for layer_id, (layer_root, manifest_path) in enumerate(zip(layer_roots, layer_manifests)):
        input_hidden = layer_output_base(layer_id - 1) if layer_id > 0 else layer_output_base(0)
        entry, layer_intervals = build_layer_entry(
            layer_id=layer_id,
            layer_root=layer_root,
            manifest_path=manifest_path,
            input_hidden_base=input_hidden,
            output_dir=output_dir,
        )
        layer_entries.append(entry)
        intervals.extend(layer_intervals)

    tail_manifest = load_json(tail_manifest_path)
    if int(tail_manifest["qmap_base"]) != FINAL_TAIL_QMAP_BASE:
        raise RuntimeError("final-tail QMAP base changed unexpectedly")

    vocab_size = int(tail_manifest["shape"]["vocab_size"])
    weight_row_bytes = int(tail_manifest["memory_layout"]["weight_row_bytes"])
    scale_row_bytes = int(tail_manifest["memory_layout"]["scale_row_bytes"])
    intervals.extend(
        [
            Interval("tied_lm_head_weight", TIED_WEIGHT_BASE, vocab_size * weight_row_bytes),
            Interval("tied_lm_head_scale", TIED_SCALE_BASE, vocab_size * scale_row_bytes),
            Interval("full_model_kv_cache", KV_CACHE_BASE, args.layer_count * KV_CACHE_BYTES_PER_LAYER),
            Interval("final_tail_qmap", FINAL_TAIL_QMAP_BASE, tail_binary.stat().st_size),
        ]
    )
    address_audit = audit_intervals(intervals)
    address_audit_path = output_dir / "physical_address_audit.json"
    address_audit_path.write_text(json.dumps(address_audit, indent=2) + "\n", encoding="ascii")

    embedding_meta_path = layer0_bundle / "embedding" / "embedding_meta.json"
    embedding_meta = load_json(embedding_meta_path)
    tail_files = {
        "qmap_image": tail_image_hex,
        "expected_output": tail_expected_hex,
        "final_norm_expected": final_norm_output,
        "lm_head_weight": tail_vectors / f"{lm_head_prefix}_weight_words32.hex",
        "lm_head_scale": tail_vectors / f"{lm_head_prefix}_scale_words32.hex",
        "expected_logits": tail_vectors / f"{lm_head_prefix}_expected_scan_logits_q26.hex",
        "scan_base_token": tail_vectors / f"{lm_head_prefix}_scan_base_token.hex",
        "weight_base_addr": tail_vectors / f"{lm_head_prefix}_weight_base_addr.hex",
        "scale_base_addr": tail_vectors / f"{lm_head_prefix}_scale_base_addr.hex",
    }
    for path in tail_files.values():
        require_file(path)

    manifest = {
        "format_version": 2,
        "name": "tied_q4_embedding_full_decoder_chain",
        "token_id": int(embedding_meta["token_id"]),
        "layer_count": args.layer_count,
        "address_profile": "pl_ddr_512m_compact_v1",
        "address_formula": {
            "pl_ddr_base": PL_DDR_BASE,
            "pl_ddr_high": PL_DDR_HIGH,
            "layer_block_base": LAYER_BLOCK_BASE,
            "layer_block_stride": LAYER_BLOCK_STRIDE,
            "body_qmap_base": BODY_QMAP_BASE,
            "body_qmap_stride": BODY_QMAP_STRIDE,
            "persistent_offsets": PERSISTENT_OFFSETS,
            "kv_cache_base": KV_CACHE_BASE,
            "kv_cache_bytes_per_layer": KV_CACHE_BYTES_PER_LAYER,
            "final_tail_qmap": FINAL_TAIL_QMAP_BASE,
        },
        "embedding": {
            "weight_base": int(embedding_meta["weight_base_addr"]),
            "scale_base": int(embedding_meta["scale_base_addr"]),
            "output_base": layer_output_base(0),
            "files": {
                "weight": relative_output_path(
                    layer0_bundle / "embedding" / "embedding_weight_words32.hex", output_dir
                ),
                "scale": relative_output_path(
                    layer0_bundle / "embedding" / "embedding_scale_words32.hex", output_dir
                ),
                "expected": relative_output_path(
                    layer0_bundle / "embedding" / "embedding_expected_q14_10.hex", output_dir
                ),
                "token": relative_output_path(
                    layer0_bundle / "embedding" / "embedding_token_id.hex", output_dir
                ),
            },
        },
        "layers": layer_entries,
        "final_tail": {
            "qmap_base": FINAL_TAIL_QMAP_BASE,
            "runtime_hidden_override": layer_output_base(args.layer_count - 1),
            "norm_prefix": final_norm_prefix,
            "lm_head_prefix": lm_head_prefix,
            "best_token": int(tail_manifest["expected"]["best_token"]),
            "best_score_q26": int(tail_manifest["expected"]["best_score_q26"]),
            "vocab_size": vocab_size,
            "files": {
                name: relative_output_path(path, output_dir) for name, path in tail_files.items()
            },
        },
        "address_audit": {
            "status": address_audit["status"],
            "path": relative_output_path(address_audit_path, output_dir),
            "interval_count": address_audit["interval_count"],
            "lowest": address_audit["lowest"],
            "highest": address_audit["highest"],
        },
        "evidence": {
            "layer_manifests": [
                {
                    "path": relative_output_path(path, output_dir),
                    "sha256": sha256_file(path),
                }
                for path in layer_manifests
            ],
            "final_layer_output": {
                "path": relative_output_path(previous_output, output_dir),
                "sha256": sha256_file(previous_output),
            },
            "tail_binary": {
                "path": relative_output_path(tail_binary, output_dir),
                "sha256": sha256_file(tail_binary),
            },
        },
        "commands": commands,
    }
    manifest_path = output_dir / "full_chain_manifest.json"
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    manifest_path.write_text(manifest_text, encoding="ascii")
    if args.layer_count == MAX_LAYERS:
        (output_dir / "full28_final_chain_manifest.json").write_text(
            manifest_text, encoding="ascii"
        )

    print(
        f"PASS: exported tied-Q4 embedding through {args.layer_count} complete layers "
        "and the full-vocabulary final tail"
    )
    print(f"output_dir={output_dir}")
    print(f"address_audit={address_audit_path}")
    print(f"last_layer_output=0x{layer_output_base(args.layer_count - 1):016X}")
    print(
        "final="
        f"token {manifest['final_tail']['best_token']} "
        f"score_q26 {manifest['final_tail']['best_score_q26']}"
    )


if __name__ == "__main__":
    main()
