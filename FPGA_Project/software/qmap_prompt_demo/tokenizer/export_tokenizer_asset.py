"""Export the local Qwen tokenizer into a deterministic compact binary asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import Any

from tokenizer_asset_format import (
    ADDED_PROP_LSTRIP,
    ADDED_PROP_NORMALIZED,
    ADDED_PROP_RSTRIP,
    ADDED_PROP_SINGLE_WORD,
    ADDED_PROP_SPECIAL,
    ALIGNMENT,
    INVALID_TOKEN_ID,
    PREFIX_SIZE,
    SECTION_BYTE_TO_TOKEN,
    SECTION_ADDED_PROPS,
    SECTION_MERGE_LOOKUP,
    SECTION_METADATA_JSON,
    SECTION_STRUCT,
    SECTION_TOKEN_BYTES,
    SECTION_TOKEN_FLAGS,
    SECTION_TOKEN_OFFSETS,
    TOKEN_FLAG_ADDED,
    TOKEN_FLAG_SPECIAL,
    TOKEN_FLAG_VALID,
    SectionDescriptor,
    align_up,
    pack_prefix,
    pack_section_descriptor,
    sha256_bytes,
    sha256_file,
)
from unicode_table_builder import build_unicode_sections


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "Qwen3-0.6B-Base"
DEFAULT_OUTPUT = REPO_ROOT / "Temp" / "qmap_prompt_demo_tokenizer" / "qwen3_tokenizer.qtk"
DEFAULT_UNICODE_SOURCES = REPO_ROOT / "Temp" / "qmap_prompt_demo_unicode_sources"

SOURCE_FILENAMES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)


class ExportError(ValueError):
    """Raised when source tokenizer assets violate the expected Qwen contract."""


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize JSON deterministically with UTF-8 and a final newline."""

    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return (text + "\n").encode("utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"Cannot load JSON source {path}: {error}") from error


def gpt2_byte_encoder() -> dict[int, str]:
    """Return the exact reversible byte-to-Unicode map used by GPT-2/Qwen."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = list(byte_values)
    extra_index = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(256 + extra_index)
            extra_index += 1
    return dict(zip(byte_values, map(chr, code_points), strict=True))


def read_merges_txt(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "#version: 0.2":
        raise ExportError(f"{path} does not start with '#version: 0.2'")
    return [line for line in lines[1:] if line]


def normalize_added_record(token_id: int, record: dict[str, Any]) -> dict[str, Any]:
    required = {
        "content",
        "lstrip",
        "normalized",
        "rstrip",
        "single_word",
        "special",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ExportError(f"Added token {token_id} lacks fields: {', '.join(missing)}")
    return {
        "id": token_id,
        "content": str(record["content"]),
        "lstrip": bool(record["lstrip"]),
        "normalized": bool(record["normalized"]),
        "rstrip": bool(record["rstrip"]),
        "single_word": bool(record["single_word"]),
        "special": bool(record["special"]),
    }


def collect_added_tokens(
    tokenizer_json: dict[str, Any], tokenizer_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Merge tokenizer.json and tokenizer_config added-token records exactly."""

    by_id: dict[int, dict[str, Any]] = {}
    for raw_record in tokenizer_json.get("added_tokens", []):
        token_id = int(raw_record["id"])
        by_id[token_id] = normalize_added_record(token_id, raw_record)

    config_records = tokenizer_config.get("added_tokens_decoder", {})
    for token_id_text, raw_record in config_records.items():
        token_id = int(token_id_text)
        normalized = normalize_added_record(token_id, raw_record)
        previous = by_id.get(token_id)
        if previous is not None and previous != normalized:
            raise ExportError(
                f"Added token {token_id} differs between tokenizer.json and tokenizer_config.json"
            )
        by_id[token_id] = normalized

    return [by_id[token_id] for token_id in sorted(by_id)]


def source_record(path: Path) -> dict[str, Any]:
    try:
        relative_path = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        relative_path = str(path.resolve())
    return {
        "path": relative_path,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_sections(
    model_dir: Path,
    unicode_sources: Path,
) -> tuple[list[tuple[str, bytes, int, int]], dict[str, Any]]:
    source_paths = {name: model_dir / name for name in SOURCE_FILENAMES}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise ExportError("Missing tokenizer source files: " + ", ".join(missing))

    tokenizer_json = load_json(source_paths["tokenizer.json"])
    vocab_json = load_json(source_paths["vocab.json"])
    tokenizer_config = load_json(source_paths["tokenizer_config.json"])
    model_config = load_json(source_paths["config.json"])
    generation_config = load_json(source_paths["generation_config.json"])

    model = tokenizer_json.get("model", {})
    if model.get("type") != "BPE":
        raise ExportError(f"Expected BPE model, got {model.get('type')!r}")
    for key, expected in {
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "byte_fallback": False,
    }.items():
        if model.get(key) != expected:
            raise ExportError(f"Unexpected BPE setting {key}={model.get(key)!r}")
    if tokenizer_json.get("normalizer") != {"type": "NFC"}:
        raise ExportError("Only the exact NFC-normalized Qwen tokenizer is supported")

    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ExportError("tokenizer.json lacks BPE vocab/merges")
    if vocab != vocab_json:
        raise ExportError("vocab.json differs from tokenizer.json model.vocab")
    merges_txt = read_merges_txt(source_paths["merges.txt"])
    if merges != merges_txt:
        raise ExportError("merges.txt differs from tokenizer.json model.merges")

    base_vocab_count = len(vocab)
    ids = sorted(int(token_id) for token_id in vocab.values())
    if ids != list(range(base_vocab_count)):
        raise ExportError("Base vocabulary IDs are not dense from zero")
    tokens_by_id = [""] * base_vocab_count
    for token, raw_token_id in vocab.items():
        token_id = int(raw_token_id)
        if tokens_by_id[token_id]:
            raise ExportError(f"Duplicate base vocabulary ID {token_id}")
        tokens_by_id[token_id] = token

    added_tokens = collect_added_tokens(tokenizer_json, tokenizer_config)
    unsupported_added = [
        record["id"]
        for record in added_tokens
        if record["single_word"] or record["lstrip"] or
           record["rstrip"] or record["normalized"]
    ]
    if unsupported_added:
        raise ExportError(
            "PS-native literal longest-match supports only added tokens with "
            "single_word/lstrip/rstrip/normalized all false; unsupported IDs: "
            + ", ".join(str(token_id) for token_id in unsupported_added)
        )
    contents = [record["content"] for record in added_tokens]
    if len(contents) != len(set(contents)):
        raise ExportError("Duplicate added-token content makes longest-match ambiguous")
    if any(record["id"] < base_vocab_count for record in added_tokens):
        raise ExportError("An added token overlaps the base vocabulary")
    if added_tokens:
        expected_added_ids = list(
            range(base_vocab_count, max(record["id"] for record in added_tokens) + 1)
        )
        actual_added_ids = [record["id"] for record in added_tokens]
        if actual_added_ids != expected_added_ids:
            raise ExportError("Added token IDs are not contiguous after the base vocabulary")
        token_count = actual_added_ids[-1] + 1
    else:
        token_count = base_vocab_count

    model_vocab_size = int(model_config["vocab_size"])
    if token_count > model_vocab_size:
        raise ExportError("Tokenizer has more IDs than the model output vocabulary")
    eos_token_id = int(model_config["eos_token_id"])
    pad_token_id = int(tokenizer_config.get("pad_token_id", eos_token_id))
    eos_content = tokenizer_config.get("eos_token")
    pad_content = tokenizer_config.get("pad_token")
    added_by_id = {record["id"]: record for record in added_tokens}
    if added_by_id.get(eos_token_id, {}).get("content") != eos_content:
        raise ExportError("EOS token ID/content does not match added-token metadata")
    if pad_content != eos_content:
        raise ExportError("This exporter expects Qwen PAD to equal EOS")

    byte_encoder = gpt2_byte_encoder()
    byte_decoder = {symbol: byte_value for byte_value, symbol in byte_encoder.items()}
    byte_to_token: list[int] = []
    for byte_value in range(256):
        symbol = byte_encoder[byte_value]
        if symbol not in vocab:
            raise ExportError(f"Base vocabulary lacks ByteLevel symbol for byte {byte_value}")
        byte_to_token.append(int(vocab[symbol]))
    byte_to_token_blob = struct.pack("<256I", *byte_to_token)

    token_bytes: list[bytes] = []
    token_flags = bytearray(token_count)
    for token_id, token in enumerate(tokens_by_id):
        try:
            decoded = bytes(byte_decoder[character] for character in token)
        except KeyError as error:
            raise ExportError(
                f"Base token {token_id} contains a non-ByteLevel character {error.args[0]!r}"
            ) from error
        token_bytes.append(decoded)
        token_flags[token_id] = TOKEN_FLAG_VALID
    for record in added_tokens:
        token_id = int(record["id"])
        token_bytes.append(record["content"].encode("utf-8"))
        flags = TOKEN_FLAG_VALID | TOKEN_FLAG_ADDED
        if record["special"]:
            flags |= TOKEN_FLAG_SPECIAL
        token_flags[token_id] = flags
    if len(token_bytes) != token_count:
        raise AssertionError("Token byte table length does not match token_count")

    offsets = [0]
    token_byte_blob = bytearray()
    for decoded in token_bytes:
        token_byte_blob.extend(decoded)
        offsets.append(len(token_byte_blob))
    token_offsets_blob = struct.pack(f"<{len(offsets)}I", *offsets)

    added_props = bytearray()
    for record in added_tokens:
        properties = 0
        if record["special"]:
            properties |= ADDED_PROP_SPECIAL
        if record["single_word"]:
            properties |= ADDED_PROP_SINGLE_WORD
        if record["lstrip"]:
            properties |= ADDED_PROP_LSTRIP
        if record["rstrip"]:
            properties |= ADDED_PROP_RSTRIP
        if record["normalized"]:
            properties |= ADDED_PROP_NORMALIZED
        added_props.append(properties)

    merge_records: list[tuple[int, int, int]] = []
    for rank, merge in enumerate(merges):
        fields = merge.split(" ")
        if len(fields) != 2:
            raise ExportError(f"Merge rank {rank} is not a two-symbol pair: {merge!r}")
        left_token, right_token = fields
        try:
            left_id = int(vocab[left_token])
            right_id = int(vocab[right_token])
            output_id = int(vocab[left_token + right_token])
        except KeyError as error:
            raise ExportError(
                f"Merge rank {rank} references a missing vocabulary token"
            ) from error
        expected_output_id = 256 + rank
        if output_id != expected_output_id:
            raise ExportError(
                f"Merge rank {rank} output ID {output_id} != {expected_output_id}"
            )
        pair_key = (left_id << 32) | right_id
        merge_records.append((pair_key, rank, output_id))
    merge_records.sort(key=lambda item: item[0])
    if any(
        merge_records[index - 1][0] == merge_records[index][0]
        for index in range(1, len(merge_records))
    ):
        raise ExportError("Duplicate merge pair detected")
    merge_lookup_blob = bytearray()
    for pair_key, rank, _output_id in merge_records:
        merge_lookup_blob.extend(struct.pack("<QI", pair_key, rank))

    unicode_sections, unicode_metadata = build_unicode_sections(unicode_sources)
    source_records = {
        name: source_record(path) for name, path in sorted(source_paths.items())
    }
    metadata = {
        "schema_version": 1,
        "asset_kind": "qwen_bytelevel_bpe",
        "source_files": source_records,
        "normalizer": tokenizer_json.get("normalizer"),
        "unicode_tables": unicode_metadata,
        "pre_tokenizer": tokenizer_json.get("pre_tokenizer"),
        "post_processor": tokenizer_json.get("post_processor"),
        "decoder": tokenizer_json.get("decoder"),
        "bpe_model": {
            key: value for key, value in model.items() if key not in {"vocab", "merges"}
        },
        "added_tokens": added_tokens,
        "tokenizer_config": {
            key: tokenizer_config.get(key)
            for key in (
                "add_bos_token",
                "add_prefix_space",
                "bos_token",
                "chat_template",
                "clean_up_tokenization_spaces",
                "eos_token",
                "errors",
                "model_max_length",
                "pad_token",
                "split_special_tokens",
                "tokenizer_class",
                "unk_token",
            )
        },
        "model_config": {
            "model_type": model_config.get("model_type"),
            "vocab_size": model_vocab_size,
            "bos_token_id": model_config.get("bos_token_id"),
            "eos_token_id": eos_token_id,
        },
        "generation_config": generation_config,
        "invariants": {
            "base_vocab_ids_dense_from_zero": True,
            "added_token_ids_contiguous": True,
            "merge_output_id": "256 + merge_rank",
            "merge_lookup_order": "ascending uint64(left_id << 32 | right_id)",
            "token_bytes": "post-ByteLevel raw bytes; added tokens are literal UTF-8",
            "added_token_match": "literal leftmost-longest before NFC; all strip/single_word/normalized flags false",
            "normalization_unicode_version": "9.0.0 (tokenizers dependency)",
            "regex_unicode_version": "16.0.0 (regex-syntax dependency)",
        },
    }
    metadata_blob = canonical_json_bytes(metadata)

    sections = [
        (SECTION_BYTE_TO_TOKEN, byte_to_token_blob, 256, 4),
        (SECTION_MERGE_LOOKUP, bytes(merge_lookup_blob), len(merges), 12),
        (SECTION_TOKEN_OFFSETS, token_offsets_blob, len(offsets), 4),
        (SECTION_TOKEN_BYTES, bytes(token_byte_blob), token_count, 0),
        (SECTION_TOKEN_FLAGS, bytes(token_flags), token_count, 1),
        (SECTION_ADDED_PROPS, bytes(added_props), len(added_tokens), 1),
        *unicode_sections,
        (SECTION_METADATA_JSON, metadata_blob, 1, 0),
    ]
    summary = {
        "base_vocab_count": base_vocab_count,
        "added_token_count": len(added_tokens),
        "token_count": token_count,
        "model_vocab_size": model_vocab_size,
        "merge_count": len(merges),
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "unk_token_id": INVALID_TOKEN_ID,
        "source_files": source_records,
        "metadata": metadata,
        "unicode_sources": unicode_metadata["sources"],
        "unicode_table_generation": unicode_metadata["generator"],
    }
    return sections, summary


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def export_asset(
    model_dir: Path,
    unicode_sources: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    sections, summary = build_sections(model_dir, unicode_sources)
    header_size = align_up(PREFIX_SIZE + len(sections) * SECTION_STRUCT.size)

    descriptors: list[SectionDescriptor] = []
    cursor = header_size
    for name, section_data, count, record_size in sections:
        cursor = align_up(cursor, ALIGNMENT)
        descriptors.append(
            SectionDescriptor(
                name=name,
                offset=cursor,
                size=len(section_data),
                count=count,
                record_size=record_size,
                sha256_hex=sha256_bytes(section_data),
            )
        )
        cursor += len(section_data)
    file_size = cursor

    asset = bytearray(header_size)
    for descriptor, (_name, section_data, _count, _record_size) in zip(
        descriptors, sections, strict=True
    ):
        if len(asset) < descriptor.offset:
            asset.extend(b"\0" * (descriptor.offset - len(asset)))
        asset.extend(section_data)
    if len(asset) != file_size:
        raise AssertionError("Asset layout size mismatch")

    payload = bytes(asset[header_size:])
    payload_sha256 = sha256_bytes(payload)
    prefix = pack_prefix(
        header_size=header_size,
        section_count=len(descriptors),
        base_vocab_count=summary["base_vocab_count"],
        added_token_count=summary["added_token_count"],
        token_count=summary["token_count"],
        model_vocab_size=summary["model_vocab_size"],
        merge_count=summary["merge_count"],
        eos_token_id=summary["eos_token_id"],
        pad_token_id=summary["pad_token_id"],
        unk_token_id=summary["unk_token_id"],
        file_size=file_size,
        payload_size=len(payload),
        payload_sha256_hex=payload_sha256,
    )
    asset[0:PREFIX_SIZE] = prefix
    for index, descriptor in enumerate(descriptors):
        entry_start = PREFIX_SIZE + index * SECTION_STRUCT.size
        asset[entry_start : entry_start + SECTION_STRUCT.size] = pack_section_descriptor(
            descriptor
        )

    asset_bytes = bytes(asset)
    write_atomic(output_path, asset_bytes)

    manifest = {
        "schema_version": 1,
        "format": "QTKBPE1",
        "format_version": 1,
        "asset": {
            "path": output_path.name,
            "size": len(asset_bytes),
            "sha256": sha256_bytes(asset_bytes),
            "header_size": header_size,
            "payload_size": len(payload),
            "payload_sha256": payload_sha256,
        },
        "counts": {
            "base_vocab": summary["base_vocab_count"],
            "added_tokens": summary["added_token_count"],
            "tokenizer_ids": summary["token_count"],
            "model_vocab": summary["model_vocab_size"],
            "merges": summary["merge_count"],
        },
        "token_ids": {
            "eos": summary["eos_token_id"],
            "pad": summary["pad_token_id"],
            "unk": None,
            "first_undecodable_model_id": summary["token_count"],
            "last_model_id": summary["model_vocab_size"] - 1,
        },
        "sections": [
            {
                "name": descriptor.name,
                "offset": descriptor.offset,
                "size": descriptor.size,
                "count": descriptor.count,
                "record_size": descriptor.record_size,
                "sha256": descriptor.sha256_hex,
            }
            for descriptor in descriptors
        ],
        "source_files": summary["source_files"],
        "unicode_sources": summary["unicode_sources"],
        "unicode_table_generation": summary["unicode_table_generation"],
        "invariants": summary["metadata"]["invariants"],
        "exporter": {
            "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest, pretty=True)
    write_atomic(manifest_path, manifest_bytes)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--unicode-sources",
        type=Path,
        default=DEFAULT_UNICODE_SOURCES,
        help="Directory populated by fetch_unicode_sources.py",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to <output>.manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_path = args.output.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    try:
        manifest = export_asset(
            args.model_dir.resolve(),
            args.unicode_sources.resolve(),
            output_path,
            manifest_path,
        )
    except (ExportError, OSError, struct.error, ValueError) as error:
        print(f"FAIL tokenizer asset export: {error}", file=sys.stderr)
        return 1

    print("PASS tokenizer asset export")
    print(f"asset={output_path}")
    print(f"asset_size={manifest['asset']['size']}")
    print(f"asset_sha256={manifest['asset']['sha256']}")
    print(f"payload_sha256={manifest['asset']['payload_sha256']}")
    print(f"manifest={manifest_path}")
    for section in manifest["sections"]:
        print(
            f"section={section['name']} offset={section['offset']} "
            f"size={section['size']} sha256={section['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
