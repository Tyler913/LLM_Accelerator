from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
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

TIED_HIDDEN_SIZE = 1024
TIED_Q4_GROUP_SIZE = 64
TIED_WEIGHT_ROW_BYTES = TIED_HIDDEN_SIZE // 2
TIED_SCALE_ROW_BYTES = (TIED_HIDDEN_SIZE // TIED_Q4_GROUP_SIZE) * 2
QMAP_MAGIC = 0x50414D51
QMAP_VERSION = 1
QMAP_HEADER_BYTES = 256
QMAP_DESCRIPTOR_BYTES = 128
QMAP_DESCRIPTOR_TABLE_WORD_OFFSET = QMAP_HEADER_BYTES // 4
QMAP_DESCRIPTOR_WORDS = QMAP_DESCRIPTOR_BYTES // 4
QMAP_DESC_BASE_LO_WORD = 8
QMAP_DESC_BASE_HI_WORD = 9
FINAL_TAIL_WEIGHT_DESCRIPTOR_SLOT = 2
FINAL_TAIL_SCALE_DESCRIPTOR_SLOT = 3
FINAL_TAIL_DESCRIPTOR_PREFIX_WORDS = (
    QMAP_DESCRIPTOR_TABLE_WORD_OFFSET
    + (FINAL_TAIL_SCALE_DESCRIPTOR_SLOT + 1) * QMAP_DESCRIPTOR_WORDS
)


@dataclass(frozen=True)
class Words32HexInspection:
    nbytes: int
    sha256: str
    captured_words: tuple[int, ...] = ()


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


def inspect_words32_hex(
    path: Path,
    *,
    capture_start_word: int | None = None,
    capture_word_count: int = 0,
) -> Words32HexInspection:
    if capture_start_word is None:
        if capture_word_count != 0:
            raise ValueError("capture_word_count requires capture_start_word")
    elif capture_start_word < 0 or capture_word_count < 1:
        raise ValueError("captured words32 range must be nonempty and nonnegative")

    digest = hashlib.sha256()
    words = 0
    captured_words: list[int] = []
    capture_end_word = (
        capture_start_word + capture_word_count
        if capture_start_word is not None
        else None
    )
    with path.open("rb") as file:
        for raw_line in file:
            digest.update(raw_line)
            text = raw_line.strip()
            if not text:
                continue
            if len(text) != 8:
                raise ValueError(f"{path}: expected one 8-digit word per line")
            try:
                value = int(text, 16)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid hexadecimal word {text!r}") from exc
            if (
                capture_start_word is not None
                and capture_start_word <= words < capture_end_word
            ):
                captured_words.append(value)
            words += 1
    if words == 0:
        raise ValueError(f"{path}: empty words32 hex file")
    if (
        capture_start_word is not None
        and len(captured_words) != capture_word_count
    ):
        raise ValueError(
            f"{path}: requested words32 slice "
            f"[{capture_start_word}, {capture_end_word}) exceeds {words} words"
        )
    return Words32HexInspection(
        nbytes=words * 4,
        sha256=digest.hexdigest(),
        captured_words=tuple(captured_words),
    )


def inspect_tied_matrix_row(
    *,
    canonical_path: Path,
    row_path: Path,
    token_id: int,
    vocab_size: int,
    row_bytes: int,
    label: str,
) -> tuple[Words32HexInspection, Words32HexInspection]:
    if vocab_size < 1:
        raise ValueError("final-tail vocab_size must be positive")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"manifest token_id {token_id} is outside tied matrix rows 0..{vocab_size - 1}"
        )
    if row_bytes < 1 or (row_bytes & 3) != 0:
        raise ValueError(f"{label}: tied row size must be positive and word aligned")

    row_words = row_bytes // 4
    row_inspection = inspect_words32_hex(
        row_path,
        capture_start_word=0,
        capture_word_count=row_words,
    )
    if row_inspection.nbytes != row_bytes:
        raise ValueError(
            f"{row_path}: expected one {label} row ({row_bytes} bytes), "
            f"got {row_inspection.nbytes} bytes"
        )

    canonical_inspection = inspect_words32_hex(
        canonical_path,
        capture_start_word=token_id * row_words,
        capture_word_count=row_words,
    )
    expected_matrix_bytes = vocab_size * row_bytes
    if canonical_inspection.nbytes != expected_matrix_bytes:
        raise ValueError(
            f"{canonical_path}: expected complete {label} matrix "
            f"({vocab_size} rows, {expected_matrix_bytes} bytes), "
            f"got {canonical_inspection.nbytes} bytes"
        )
    if canonical_inspection.captured_words != row_inspection.captured_words:
        raise ValueError(
            f"{row_path}: does not match token {token_id} in canonical {label} matrix"
        )
    return canonical_inspection, row_inspection


def parse_final_tail_qmap_descriptor_bases(
    *,
    path: Path,
    inspection: Words32HexInspection,
    expected_qmap_base: int,
    expected_weight_base: int,
    expected_scale_base: int,
) -> tuple[int, int]:
    words = inspection.captured_words
    if len(words) != FINAL_TAIL_DESCRIPTOR_PREFIX_WORDS:
        raise ValueError(
            f"{path}: final-tail QMAP prefix must contain exactly "
            f"{FINAL_TAIL_DESCRIPTOR_PREFIX_WORDS} captured words"
        )

    def word64(index: int) -> int:
        return words[index] | (words[index + 1] << 32)

    if words[0] != QMAP_MAGIC or words[1] != QMAP_VERSION:
        raise ValueError(f"{path}: invalid QMAP magic/version")
    if words[2] != QMAP_HEADER_BYTES or words[3] != QMAP_DESCRIPTOR_BYTES:
        raise ValueError(f"{path}: unsupported QMAP header/descriptor size")

    descriptor_count = words[4]
    descriptor_capacity = words[5]
    if (
        descriptor_count <= FINAL_TAIL_SCALE_DESCRIPTOR_SLOT
        or descriptor_capacity < descriptor_count
    ):
        raise ValueError(f"{path}: final-tail QMAP descriptor table is incomplete")

    descriptor_table_addr = word64(6)
    qmap_base = word64(10)
    image_bytes = word64(12)
    if qmap_base != expected_qmap_base:
        raise ValueError(
            f"{path}: QMAP header base 0x{qmap_base:016X} does not match "
            f"manifest base 0x{expected_qmap_base:016X}"
        )
    if descriptor_table_addr != expected_qmap_base + QMAP_HEADER_BYTES:
        raise ValueError(f"{path}: QMAP descriptor-table address is inconsistent")
    if image_bytes != inspection.nbytes:
        raise ValueError(
            f"{path}: QMAP header image size {image_bytes} does not match "
            f"decoded words32 size {inspection.nbytes}"
        )
    if (
        QMAP_HEADER_BYTES + descriptor_count * QMAP_DESCRIPTOR_BYTES
        > inspection.nbytes
    ):
        raise ValueError(f"{path}: QMAP descriptor table escapes the image")

    def descriptor_base(slot: int) -> int:
        start = QMAP_DESCRIPTOR_TABLE_WORD_OFFSET + slot * QMAP_DESCRIPTOR_WORDS
        return word64(start + QMAP_DESC_BASE_LO_WORD)

    weight_base = descriptor_base(FINAL_TAIL_WEIGHT_DESCRIPTOR_SLOT)
    scale_base = descriptor_base(FINAL_TAIL_SCALE_DESCRIPTOR_SLOT)
    if weight_base != expected_weight_base:
        raise ValueError(
            f"{path}: QMAP weight descriptor base 0x{weight_base:016X} does not "
            f"match tied base 0x{expected_weight_base:016X}"
        )
    if scale_base != expected_scale_base:
        raise ValueError(
            f"{path}: QMAP scale descriptor base 0x{scale_base:016X} does not "
            f"match tied base 0x{expected_scale_base:016X}"
        )
    return weight_base, scale_base


def read_hex_scalar(path: Path, *, label: str) -> int:
    values = [
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(values) != 1:
        raise ValueError(f"{path}: expected exactly one {label} hexadecimal value")
    try:
        value = int(values[0], 16)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid {label} hexadecimal value") from exc
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{path}: {label} does not fit in 64 bits")
    return value


def relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_text_atomic(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(text)
            temporary_path = Path(file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    load_plan_path: Path,
) -> dict[str, Any]:
    manifest_dir = manifest_path.parent.resolve()
    load_plan_dir = load_plan_path.parent.resolve()
    runtime = manifest["runtime_context"]
    entries: list[dict[str, Any]] = []
    occupied: dict[int, dict[str, Any]] = {}

    def add_file(
        name: str,
        address: int,
        relative_path: str,
        encoding: str,
        inspection: Words32HexInspection | None = None,
    ) -> dict[str, Any]:
        path = resolve_asset(manifest_dir, relative_path)
        if encoding == "words32_hex_le":
            file_inspection = inspection or inspect_words32_hex(path)
            nbytes = file_inspection.nbytes
            sha256 = file_inspection.sha256
        elif encoding == "binary":
            if inspection is not None:
                raise ValueError("preinspection is only supported for words32 hex files")
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
            if prior["encoding"] != encoding or prior["sha256"] != sha256:
                raise ValueError(
                    f"load alias content mismatch at 0x{address:016X}: "
                    f"{prior['name']} vs {name}"
                )
            prior.setdefault("aliases", []).append(name)
            return prior
        entry = {
            "operation": "file",
            "name": name,
            "address": address,
            "nbytes": nbytes,
            "encoding": encoding,
            "path": relative_to_output(path, load_plan_dir),
            "sha256": sha256,
        }
        occupied[address] = entry
        entries.append(entry)
        return entry

    def add_verified_slice(
        entry: dict[str, Any],
        *,
        name: str,
        source_path: Path,
        source_inspection: Words32HexInspection,
        token_id: int,
        row_bytes: int,
    ) -> None:
        entry.setdefault("verified_slices", []).append(
            {
                "name": name,
                "relationship": "verified_exact_slice",
                "token_id": token_id,
                "source_path": relative_to_output(source_path, load_plan_dir),
                "source_nbytes": source_inspection.nbytes,
                "source_sha256": source_inspection.sha256,
                "canonical_byte_offset": token_id * row_bytes,
                "canonical_nbytes": row_bytes,
                "validation": "exact_words32_match",
            }
        )

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
    final_tail = manifest["final_tail"]
    token_id = int(manifest["token_id"])
    vocab_size = int(final_tail["vocab_size"])

    embedding_weight_path = resolve_asset(manifest_dir, embedding["files"]["weight"])
    embedding_scale_path = resolve_asset(manifest_dir, embedding["files"]["scale"])
    lm_head_weight_path = resolve_asset(
        manifest_dir, final_tail["files"]["lm_head_weight"]
    )
    lm_head_scale_path = resolve_asset(
        manifest_dir, final_tail["files"]["lm_head_scale"]
    )
    lm_head_weight_base_path = resolve_asset(
        manifest_dir, final_tail["files"]["weight_base_addr"]
    )
    lm_head_scale_base_path = resolve_asset(
        manifest_dir, final_tail["files"]["scale_base_addr"]
    )
    final_tail_qmap_path = resolve_asset(
        manifest_dir, final_tail["files"]["qmap_image"]
    )
    final_tail_qmap_inspection = inspect_words32_hex(
        final_tail_qmap_path,
        capture_start_word=0,
        capture_word_count=FINAL_TAIL_DESCRIPTOR_PREFIX_WORDS,
    )
    tied_weight_base = int(embedding["weight_base"])
    tied_scale_base = int(embedding["scale_base"])
    qmap_weight_base, qmap_scale_base = parse_final_tail_qmap_descriptor_bases(
        path=final_tail_qmap_path,
        inspection=final_tail_qmap_inspection,
        expected_qmap_base=int(final_tail["qmap_base"]),
        expected_weight_base=tied_weight_base,
        expected_scale_base=tied_scale_base,
    )
    sidecar_weight_base = read_hex_scalar(
        lm_head_weight_base_path, label="LM-head weight base"
    )
    sidecar_scale_base = read_hex_scalar(
        lm_head_scale_base_path, label="LM-head scale base"
    )
    if sidecar_weight_base != tied_weight_base:
        raise ValueError(
            "final-tail LM-head weight descriptor does not match tied embedding base"
        )
    if sidecar_scale_base != tied_scale_base:
        raise ValueError(
            "final-tail LM-head scale descriptor does not match tied embedding base"
        )
    if qmap_weight_base != tied_weight_base or qmap_weight_base != sidecar_weight_base:
        raise ValueError(
            "final-tail QMAP weight descriptor does not match tied embedding/sidecar base"
        )
    if qmap_scale_base != tied_scale_base or qmap_scale_base != sidecar_scale_base:
        raise ValueError(
            "final-tail QMAP scale descriptor does not match tied embedding/sidecar base"
        )
    lm_head_weight_inspection, embedding_weight_inspection = inspect_tied_matrix_row(
        canonical_path=lm_head_weight_path,
        row_path=embedding_weight_path,
        token_id=token_id,
        vocab_size=vocab_size,
        row_bytes=TIED_WEIGHT_ROW_BYTES,
        label="tied-Q4 weight",
    )
    lm_head_scale_inspection, embedding_scale_inspection = inspect_tied_matrix_row(
        canonical_path=lm_head_scale_path,
        row_path=embedding_scale_path,
        token_id=token_id,
        vocab_size=vocab_size,
        row_bytes=TIED_SCALE_ROW_BYTES,
        label="tied-Q4 scale",
    )

    tied_weight_entry = add_file(
        "tied_embedding_weight",
        int(embedding["weight_base"]),
        final_tail["files"]["lm_head_weight"],
        "words32_hex_le",
        lm_head_weight_inspection,
    )
    tied_weight_entry.setdefault("aliases", []).append("tied_lm_head_weight_alias")
    add_verified_slice(
        tied_weight_entry,
        name="tied_embedding_weight_row",
        source_path=embedding_weight_path,
        source_inspection=embedding_weight_inspection,
        token_id=token_id,
        row_bytes=TIED_WEIGHT_ROW_BYTES,
    )
    tied_scale_entry = add_file(
        "tied_embedding_scale",
        int(embedding["scale_base"]),
        final_tail["files"]["lm_head_scale"],
        "words32_hex_le",
        lm_head_scale_inspection,
    )
    tied_scale_entry.setdefault("aliases", []).append("tied_lm_head_scale_alias")
    add_verified_slice(
        tied_scale_entry,
        name="tied_embedding_scale_row",
        source_path=embedding_scale_path,
        source_inspection=embedding_scale_inspection,
        token_id=token_id,
        row_bytes=TIED_SCALE_ROW_BYTES,
    )

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

    add_file(
        "final_tail_qmap",
        int(final_tail["qmap_base"]),
        final_tail["files"]["qmap_image"],
        "words32_hex_le",
        final_tail_qmap_inspection,
    )

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
        "path_base": "load_plan_directory",
        "source_manifest": relative_to_output(manifest_path, load_plan_dir),
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

    output_header_path = args.output_header.resolve()
    load_plan_path = (
        args.load_plan.resolve()
        if args.load_plan is not None
        else manifest_path.with_name("pl_ddr_runtime_load_plan.json")
    )
    header_text = make_header(manifest, manifest_sha256)
    load_plan = build_load_plan(
        manifest, manifest_path, manifest_sha256, load_plan_path
    )
    load_plan_text = json.dumps(load_plan, indent=2) + "\n"

    write_text_atomic(output_header_path, header_text, encoding="ascii")
    write_text_atomic(load_plan_path, load_plan_text, encoding="ascii")

    print("PASS: exported QMAP runtime C configuration and PL-DDR load plan")
    print(f"header={output_header_path}")
    print(f"load_plan={load_plan_path}")
    print(f"entries={load_plan['entry_count']}")
    print(f"file_bytes={load_plan['total_file_bytes']}")
    print(f"zero_bytes={load_plan['total_zero_bytes']}")


if __name__ == "__main__":
    main()
