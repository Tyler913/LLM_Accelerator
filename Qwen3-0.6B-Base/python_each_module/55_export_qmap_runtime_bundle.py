from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "Temp" / "embedding_full28_final_chain_vectors" / "full_chain_manifest.json"
)
DEFAULT_HEADER = (
    REPO_ROOT
    / "FPGA_Project"
    / "software"
    / "qmap_one_token_runtime"
    / "qmap_model_config_generated.h"
)

LAYER_QMAP_FILES = {
    "input_norm_image": "input_rmsnorm",
    "qkv_image": "qkv",
    "frontend_image": "attn_frontend",
    "score_image": "attn_score_value",
    "oproj_image": "o_proj",
    "post_image": "post_attn_norm",
    "gate_image": "mlp_gate_up",
    "silu_image": "mlp_silu_mul",
    "down_image": "mlp_down",
    "residual_image": "mlp_residual_add",
}

LAYER_PERSISTENT_FILES = {
    "oproj_weight": "o_proj_weight",
    "oproj_scale": "o_proj_scale",
    "gate_weight": "mlp_gate_weight",
    "gate_scale": "mlp_gate_scale",
    "up_weight": "mlp_up_weight",
    "up_scale": "mlp_up_scale",
    "down_weight": "mlp_down_weight",
    "down_scale": "mlp_down_scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a runtime-enabled full-chain manifest into a C address table "
            "and an ordered PL-DDR load plan."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--load-plan", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_asset(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else manifest_dir / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def inspect_words32_hex(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    words = 0
    with path.open("rb") as file:
        for raw_line in file:
            digest.update(raw_line)
            text = raw_line.strip()
            if not text:
                continue
            if len(text) != 8:
                raise ValueError(f"{path}: expected one 8-digit word per line")
            try:
                int(text, 16)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid hexadecimal word {text!r}") from exc
            words += 1
    if words == 0:
        raise ValueError(f"{path}: empty words32 hex file")
    return words * 4, digest.hexdigest()


def relative_to_manifest(path: Path, manifest_dir: Path) -> str:
    try:
        return path.relative_to(manifest_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def u64(value: int) -> str:
    return f"UINT64_C(0x{value:016X})"


def validate_runtime_contract(manifest: dict[str, Any]) -> None:
    layers = manifest["layers"]
    runtime = manifest["runtime_context"]
    layer_count = int(manifest["layer_count"])
    if layer_count != len(layers) or layer_count < 1 or layer_count > 28:
        raise ValueError("full-chain layer count is inconsistent")
    if not runtime.get("enabled"):
        raise ValueError("runtime bundle export requires runtime_context.enabled=true")

    hidden_a = int(runtime["hidden_a_base"])
    hidden_b = int(runtime["hidden_b_base"])
    if hidden_a == 0 or hidden_b == 0 or hidden_a == hidden_b:
        raise ValueError("runtime hidden A/B bases must be nonzero and distinct")
    if (hidden_a & 3) != 0 or (hidden_b & 3) != 0:
        raise ValueError("runtime hidden A/B bases must be 4-byte aligned")
    if int(manifest["embedding"]["output_base"]) != hidden_a:
        raise ValueError("runtime embedding output must target hidden buffer A")

    for ordinal, layer in enumerate(layers):
        if int(layer["layer_id"]) != ordinal:
            raise ValueError(f"layer entry {ordinal} has mismatched layer_id")
        expected_input = hidden_a if (ordinal & 1) == 0 else hidden_b
        expected_output = hidden_b if (ordinal & 1) == 0 else hidden_a
        if int(layer.get("runtime_input_hidden_base", -1)) != expected_input:
            raise ValueError(f"layer {ordinal} runtime input hidden ping-pong mismatch")
        if int(layer.get("runtime_output_hidden_base", -1)) != expected_output:
            raise ValueError(f"layer {ordinal} runtime output hidden ping-pong mismatch")

    expected_last = hidden_b if (layer_count & 1) else hidden_a
    if int(runtime.get("last_layer_hidden_base", -1)) != expected_last:
        raise ValueError("runtime last-layer hidden base does not match layer-count parity")
    if int(manifest["final_tail"].get("runtime_hidden_override", -1)) != expected_last:
        raise ValueError("final-tail runtime hidden base does not match layer-count parity")


def make_header(manifest: dict[str, Any], manifest_sha256: str) -> str:
    layers = manifest["layers"]
    runtime = manifest["runtime_context"]
    embedding = manifest["embedding"]
    final_tail = manifest["final_tail"]
    layer_count = int(manifest["layer_count"])

    if layer_count != len(layers) or layer_count < 1 or layer_count > 28:
        raise ValueError("full-chain layer count is inconsistent")
    if not runtime.get("enabled"):
        raise ValueError("runtime bundle export requires runtime_context.enabled=true")

    lines = [
        "#ifndef QMAP_MODEL_CONFIG_GENERATED_H",
        "#define QMAP_MODEL_CONFIG_GENERATED_H",
        "",
        "/* Generated by 55_export_qmap_runtime_bundle.py. */",
        f"/* Source manifest SHA256: {manifest_sha256} */",
        "",
        "#include <stdint.h>",
        "#include \"qmap_one_token_runtime.h\"",
        "",
        f"#define QOT_MODEL_LAYER_COUNT {layer_count}u",
        f"#define QOT_MODEL_EMBED_WEIGHT_BASE {u64(int(embedding['weight_base']))}",
        f"#define QOT_MODEL_EMBED_SCALE_BASE {u64(int(embedding['scale_base']))}",
        f"#define QOT_MODEL_HIDDEN_A_BASE {u64(int(runtime['hidden_a_base']))}",
        f"#define QOT_MODEL_HIDDEN_B_BASE {u64(int(runtime['hidden_b_base']))}",
        f"#define QOT_MODEL_KV_CACHE_BASE {u64(int(manifest['address_formula']['kv_cache_base']))}",
        f"#define QOT_MODEL_FINAL_TAIL_QMAP_BASE {u64(int(final_tail['qmap_base']))}",
        f"#define QOT_MODEL_ROPE_COS_BASE {u64(int(runtime['rope_cos_base']))}",
        f"#define QOT_MODEL_ROPE_SIN_BASE {u64(int(runtime['rope_sin_base']))}",
        "",
        "static const qot_layer_qmap_bases_t",
        "qot_model_layer_qmap_bases[QOT_MODEL_LAYER_COUNT] = {",
    ]
    for expected_layer, layer in enumerate(layers):
        if int(layer["layer_id"]) != expected_layer:
            raise ValueError(f"layer entry {expected_layer} has mismatched layer_id")
        bases = layer["qmap_bases"]
        lines.extend(
            [
                "    {",
                f"        {u64(int(bases['qkv']))},",
                f"        {u64(int(bases['input_rmsnorm']))},",
                f"        {u64(int(bases['attn_frontend']))},",
                f"        {u64(int(bases['attn_score_value']))},",
                f"        {u64(int(bases['o_proj']))},",
                f"        {u64(int(bases['post_attn_norm']))},",
                f"        {u64(int(bases['mlp_gate_up']))},",
                f"        {u64(int(bases['mlp_silu_mul']))},",
                f"        {u64(int(bases['mlp_down']))},",
                f"        {u64(int(bases['mlp_residual_add']))}",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "static inline qot_run_config_t qot_model_default_run_config(void)",
            "{",
            "    qot_run_config_t cfg = {0};",
            "    cfg.layer_start = 0u;",
            "    cfg.layer_count = QOT_MODEL_LAYER_COUNT;",
            "    cfg.position = 0u;",
            "    cfg.runtime_context_enable = 1u;",
            "    cfg.embedding_enable = 1u;",
            "    cfg.embedding_weight_base = QOT_MODEL_EMBED_WEIGHT_BASE;",
            "    cfg.embedding_scale_base = QOT_MODEL_EMBED_SCALE_BASE;",
            "    cfg.input_hidden_base = QOT_MODEL_HIDDEN_A_BASE;",
            "    cfg.output_hidden_base = QOT_MODEL_HIDDEN_B_BASE;",
            "    cfg.kv_cache_base = QOT_MODEL_KV_CACHE_BASE;",
            "    cfg.final_tail_qmap_base = QOT_MODEL_FINAL_TAIL_QMAP_BASE;",
            "    cfg.final_hidden_override_valid = 0u;",
            "    cfg.final_hidden_override_base = UINT64_C(0);",
            "    cfg.layer_qmap_bases = qot_model_layer_qmap_bases;",
            "    cfg.layer_qmap_base_count = QOT_MODEL_LAYER_COUNT;",
            "    return cfg;",
            "}",
            "",
            "#endif /* QMAP_MODEL_CONFIG_GENERATED_H */",
            "",
        ]
    )
    return "\n".join(lines)


def build_load_plan(
    manifest: dict[str, Any], manifest_path: Path, manifest_sha256: str
) -> dict[str, Any]:
    manifest_dir = manifest_path.parent.resolve()
    runtime = manifest["runtime_context"]
    entries: list[dict[str, Any]] = []
    occupied: dict[int, dict[str, Any]] = {}

    def add_file(name: str, address: int, relative_path: str, encoding: str) -> None:
        path = resolve_asset(manifest_dir, relative_path)
        if encoding == "words32_hex_le":
            nbytes, sha256 = inspect_words32_hex(path)
        elif encoding == "binary":
            nbytes = path.stat().st_size
            if nbytes == 0:
                raise ValueError(f"{path}: empty binary file")
            sha256 = sha256_file(path)
        else:
            raise ValueError(f"unsupported load encoding {encoding}")
        if address in occupied:
            prior = occupied[address]
            if prior["nbytes"] != nbytes:
                raise ValueError(
                    f"load alias size mismatch at 0x{address:016X}: "
                    f"{prior['name']} vs {name}"
                )
            prior.setdefault("aliases", []).append(name)
            return
        entry = {
            "operation": "file",
            "name": name,
            "address": address,
            "nbytes": nbytes,
            "encoding": encoding,
            "path": relative_to_manifest(path, manifest_dir),
            "sha256": sha256,
        }
        occupied[address] = entry
        entries.append(entry)

    def add_zero(name: str, address: int, nbytes: int) -> None:
        entries.append(
            {
                "operation": "zero",
                "name": name,
                "address": address,
                "nbytes": nbytes,
            }
        )

    embedding = manifest["embedding"]
    add_file("tied_embedding_weight", int(embedding["weight_base"]), embedding["files"]["weight"], "words32_hex_le")
    add_file("tied_embedding_scale", int(embedding["scale_base"]), embedding["files"]["scale"], "words32_hex_le")

    for layer in manifest["layers"]:
        layer_id = int(layer["layer_id"])
        files = layer["files"]
        for file_key, base_key in LAYER_QMAP_FILES.items():
            add_file(
                f"layer{layer_id}.{base_key}_qmap",
                int(layer["qmap_bases"][base_key]),
                files[file_key],
                "words32_hex_le",
            )
        for file_key, base_key in LAYER_PERSISTENT_FILES.items():
            add_file(
                f"layer{layer_id}.{base_key}",
                int(layer["persistent_bases"][base_key]),
                files[file_key],
                "words32_hex_le",
            )

    final_tail = manifest["final_tail"]
    add_file("final_tail_qmap", int(final_tail["qmap_base"]), final_tail["files"]["qmap_image"], "words32_hex_le")
    add_file("tied_lm_head_weight_alias", int(embedding["weight_base"]), final_tail["files"]["lm_head_weight"], "words32_hex_le")
    add_file("tied_lm_head_scale_alias", int(embedding["scale_base"]), final_tail["files"]["lm_head_scale"], "words32_hex_le")

    rope_files = runtime.get("files")
    if not rope_files or not rope_files.get("binary"):
        raise ValueError("runtime manifest does not contain the persistent RoPE binary")
    add_file("runtime_rope_table", int(runtime["rope_cos_base"]), rope_files["binary"], "binary")

    layer_count = int(manifest["layer_count"])
    kv_bytes_per_layer = int(manifest["address_formula"]["kv_cache_bytes_per_layer"])
    add_zero("kv_cache", int(manifest["address_formula"]["kv_cache_base"]), layer_count * kv_bytes_per_layer)
    add_zero("runtime_hidden_a", int(runtime["hidden_a_base"]), 1024 * 4)
    add_zero("runtime_hidden_b", int(runtime["hidden_b_base"]), 1024 * 4)

    entries.sort(key=lambda item: (int(item["address"]), item["operation"] != "zero"))
    for prior, current in zip(entries, entries[1:]):
        prior_high = int(prior["address"]) + int(prior["nbytes"])
        if int(current["address"]) < prior_high:
            raise ValueError(
                f"load intervals overlap: {prior['name']} and {current['name']}"
            )
    pl_ddr_base = manifest.get("address_formula", {}).get("pl_ddr_base")
    pl_ddr_high = manifest.get("address_formula", {}).get("pl_ddr_high")
    if pl_ddr_base is not None and pl_ddr_high is not None:
        for item in entries:
            low = int(item["address"])
            high = low + int(item["nbytes"]) - 1
            if low < int(pl_ddr_base) or high > int(pl_ddr_high):
                raise ValueError(f"{item['name']} escapes the declared PL-DDR aperture")
    return {
        "format_version": 1,
        "name": "qwen3_0p6b_pl_ddr_runtime_load_plan",
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": manifest_sha256,
        "layer_count": layer_count,
        "runtime_context_required": True,
        "entries": entries,
        "entry_count": len(entries),
        "total_file_bytes": sum(
            int(item["nbytes"]) for item in entries if item["operation"] == "file"
        ),
        "total_zero_bytes": sum(
            int(item["nbytes"]) for item in entries if item["operation"] == "zero"
        ),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    validate_runtime_contract(manifest)

    header_text = make_header(manifest, manifest_sha256)
    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_header.write_text(header_text, encoding="ascii")

    load_plan_path = args.load_plan or manifest_path.with_name("pl_ddr_runtime_load_plan.json")
    load_plan = build_load_plan(manifest, manifest_path, manifest_sha256)
    load_plan_path.parent.mkdir(parents=True, exist_ok=True)
    load_plan_path.write_text(json.dumps(load_plan, indent=2) + "\n", encoding="ascii")

    print("PASS: exported QMAP runtime C configuration and PL-DDR load plan")
    print(f"header={args.output_header}")
    print(f"load_plan={load_plan_path}")
    print(f"entries={load_plan['entry_count']}")
    print(f"file_bytes={load_plan['total_file_bytes']}")
    print(f"zero_bytes={load_plan['total_zero_bytes']}")


if __name__ == "__main__":
    main()
