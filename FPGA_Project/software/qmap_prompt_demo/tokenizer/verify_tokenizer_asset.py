"""Verify a QTKBPE1 asset, its manifest, and HF tokenizer golden corpus."""

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
    FORMAT_VERSION,
    INVALID_TOKEN_ID,
    SECTION_ADDED_PROPS,
    SECTION_BYTE_TO_TOKEN,
    SECTION_MERGE_LOOKUP,
    SECTION_METADATA_JSON,
    SECTION_TOKEN_BYTES,
    SECTION_TOKEN_FLAGS,
    SECTION_TOKEN_OFFSETS,
    TOKEN_FLAG_ADDED,
    TOKEN_FLAG_SPECIAL,
    TOKEN_FLAG_VALID,
    AssetFormatError,
    ParsedAsset,
    parse_asset,
    sha256_file,
)
from unicode_table_builder import build_unicode_sections


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "Qwen3-0.6B-Base"
DEFAULT_ASSET = REPO_ROOT / "Temp" / "qmap_prompt_demo_tokenizer" / "qwen3_tokenizer.qtk"
DEFAULT_GOLDEN = Path(__file__).resolve().with_name("golden_corpus.json")
DEFAULT_UNICODE_SOURCES = REPO_ROOT / "Temp" / "qmap_prompt_demo_unicode_sources"

SOURCE_FILENAMES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)


class VerificationError(ValueError):
    """Raised when source, binary, manifest, or golden evidence disagrees."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot load JSON {path}: {error}") from error


def gpt2_byte_encoder() -> dict[int, str]:
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
    require(bool(lines) and lines[0].strip() == "#version: 0.2", "Bad merges header")
    return [line for line in lines[1:] if line]


def collect_added_tokens(
    tokenizer_json: dict[str, Any], tokenizer_config: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for raw in tokenizer_json.get("added_tokens", []):
        by_id[int(raw["id"])] = raw
    for token_id_text, raw in tokenizer_config.get("added_tokens_decoder", {}).items():
        token_id = int(token_id_text)
        previous = by_id.get(token_id)
        if previous is not None:
            for key in (
                "content",
                "lstrip",
                "normalized",
                "rstrip",
                "single_word",
                "special",
            ):
                require(
                    previous[key] == raw[key],
                    f"Added token {token_id} field {key} differs across sources",
                )
        by_id[token_id] = {"id": token_id, **raw}
    return by_id


def unpack_u32_array(data: bytes) -> list[int]:
    require(len(data) % 4 == 0, "u32 section size is not divisible by four")
    return [entry[0] for entry in struct.iter_unpack("<I", data)]


def validate_manifest(
    asset_path: Path,
    asset: ParsedAsset,
    manifest_path: Path,
    model_dir: Path,
    unicode_source_records: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == 1, "Manifest schema_version is not 1")
    require(manifest.get("format") == "QTKBPE1", "Manifest format is not QTKBPE1")
    require(
        manifest.get("format_version") == FORMAT_VERSION,
        "Manifest format_version mismatch",
    )
    manifest_asset = manifest.get("asset", {})
    require(manifest_asset.get("path") == asset_path.name, "Manifest asset filename mismatch")
    require(manifest_asset.get("size") == len(asset.data), "Manifest asset size mismatch")
    require(
        manifest_asset.get("sha256") == sha256_file(asset_path),
        "Manifest whole-file SHA-256 mismatch",
    )
    require(
        manifest_asset.get("header_size") == asset.header["header_size"],
        "Manifest header size mismatch",
    )
    require(
        manifest_asset.get("payload_size") == asset.header["payload_size"],
        "Manifest payload size mismatch",
    )
    require(
        manifest_asset.get("payload_sha256") == asset.header["payload_sha256"],
        "Manifest payload SHA-256 mismatch",
    )

    manifest_sections = {entry["name"]: entry for entry in manifest.get("sections", [])}
    require(set(manifest_sections) == set(asset.sections), "Manifest section names mismatch")
    for name, descriptor in asset.sections.items():
        entry = manifest_sections[name]
        for field in ("offset", "size", "count", "record_size"):
            require(
                int(entry[field]) == getattr(descriptor, field),
                f"Manifest section {name} {field} mismatch",
            )
        require(
            entry["sha256"] == descriptor.sha256_hex,
            f"Manifest section {name} SHA-256 mismatch",
        )

    source_records = manifest.get("source_files", {})
    require(set(source_records) == set(SOURCE_FILENAMES), "Manifest source-file set mismatch")
    for filename in SOURCE_FILENAMES:
        path = model_dir / filename
        record = source_records[filename]
        require(path.stat().st_size == record["size"], f"Source size mismatch for {filename}")
        require(sha256_file(path) == record["sha256"], f"Source hash mismatch for {filename}")
    require(
        manifest.get("unicode_sources") == unicode_source_records,
        "Manifest Unicode-source provenance mismatch",
    )
    return manifest


def validate_unicode_and_added_tables(
    asset: ParsedAsset,
    model_dir: Path,
    unicode_sources: Path,
) -> dict[str, Any]:
    expected_sections, unicode_metadata = build_unicode_sections(unicode_sources)
    for name, expected_bytes, expected_count, expected_record_size in expected_sections:
        descriptor = asset.sections[name]
        require(asset.section_bytes(name) == expected_bytes, f"Unicode section {name} mismatch")
        require(descriptor.count == expected_count, f"Unicode section {name} count mismatch")
        require(
            descriptor.record_size == expected_record_size,
            f"Unicode section {name} record size mismatch",
        )

    tokenizer_json = load_json(model_dir / "tokenizer.json")
    tokenizer_config = load_json(model_dir / "tokenizer_config.json")
    added_by_id = collect_added_tokens(tokenizer_json, tokenizer_config)
    props = asset.section_bytes(SECTION_ADDED_PROPS)
    require(len(props) == len(added_by_id), "ADDED_PROPS length mismatch")
    for index, token_id in enumerate(sorted(added_by_id)):
        record = added_by_id[token_id]
        expected = 0
        if record["special"]:
            expected |= ADDED_PROP_SPECIAL
        if record["single_word"]:
            expected |= ADDED_PROP_SINGLE_WORD
        if record["lstrip"]:
            expected |= ADDED_PROP_LSTRIP
        if record["rstrip"]:
            expected |= ADDED_PROP_RSTRIP
        if record["normalized"]:
            expected |= ADDED_PROP_NORMALIZED
        require(props[index] == expected, f"ADDED_PROPS mismatch for token {token_id}")
    return unicode_metadata


def validate_binary_tables(asset: ParsedAsset, model_dir: Path) -> dict[str, Any]:
    tokenizer_json = load_json(model_dir / "tokenizer.json")
    vocab_json = load_json(model_dir / "vocab.json")
    tokenizer_config = load_json(model_dir / "tokenizer_config.json")
    model_config = load_json(model_dir / "config.json")
    generation_config = load_json(model_dir / "generation_config.json")
    merges_txt = read_merges_txt(model_dir / "merges.txt")

    model = tokenizer_json["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    require(vocab == vocab_json, "vocab.json differs from tokenizer.json")
    require(merges == merges_txt, "merges.txt differs from tokenizer.json")
    base_vocab_count = len(vocab)
    added_by_id = collect_added_tokens(tokenizer_json, tokenizer_config)
    token_count = max([base_vocab_count - 1, *added_by_id]) + 1

    require(asset.header["base_vocab_count"] == base_vocab_count, "Base vocab count mismatch")
    require(asset.header["added_token_count"] == len(added_by_id), "Added count mismatch")
    require(asset.header["token_count"] == token_count, "Tokenizer ID count mismatch")
    require(asset.header["model_vocab_size"] == model_config["vocab_size"], "Model vocab mismatch")
    require(asset.header["merge_count"] == len(merges), "Merge count mismatch")
    require(asset.header["eos_token_id"] == model_config["eos_token_id"], "EOS mismatch")
    require(asset.header["pad_token_id"] == model_config["eos_token_id"], "PAD mismatch")
    require(asset.header["unk_token_id"] == INVALID_TOKEN_ID, "UNK sentinel mismatch")

    byte_encoder = gpt2_byte_encoder()
    byte_decoder = {symbol: byte_value for byte_value, symbol in byte_encoder.items()}
    byte_to_token = unpack_u32_array(asset.section_bytes(SECTION_BYTE_TO_TOKEN))
    require(len(byte_to_token) == 256, "BYTE_TO_TOKEN count mismatch")
    expected_byte_to_token = [int(vocab[byte_encoder[value]]) for value in range(256)]
    require(byte_to_token == expected_byte_to_token, "BYTE_TO_TOKEN data mismatch")

    offsets = unpack_u32_array(asset.section_bytes(SECTION_TOKEN_OFFSETS))
    token_blob = asset.section_bytes(SECTION_TOKEN_BYTES)
    token_flags = asset.section_bytes(SECTION_TOKEN_FLAGS)
    require(len(offsets) == token_count + 1, "TOKEN_OFFSETS count mismatch")
    require(offsets[0] == 0 and offsets[-1] == len(token_blob), "Bad token offset bounds")
    require(all(left <= right for left, right in zip(offsets, offsets[1:])), "Offsets regress")
    require(len(token_flags) == token_count, "TOKEN_FLAGS count mismatch")

    base_by_id = [""] * base_vocab_count
    for token, token_id in vocab.items():
        base_by_id[int(token_id)] = token
    for token_id in range(token_count):
        actual_bytes = token_blob[offsets[token_id] : offsets[token_id + 1]]
        if token_id < base_vocab_count:
            expected_bytes = bytes(byte_decoder[character] for character in base_by_id[token_id])
            expected_flags = TOKEN_FLAG_VALID
        else:
            record = added_by_id[token_id]
            expected_bytes = record["content"].encode("utf-8")
            expected_flags = TOKEN_FLAG_VALID | TOKEN_FLAG_ADDED
            if record["special"]:
                expected_flags |= TOKEN_FLAG_SPECIAL
        require(actual_bytes == expected_bytes, f"Token bytes mismatch at ID {token_id}")
        require(token_flags[token_id] == expected_flags, f"Token flags mismatch at ID {token_id}")

    expected_merge_records: list[tuple[int, int]] = []
    for rank, merge in enumerate(merges):
        left_token, right_token = merge.split(" ")
        left_id = int(vocab[left_token])
        right_id = int(vocab[right_token])
        require(
            int(vocab[left_token + right_token]) == 256 + rank,
            f"Merge output invariant fails at rank {rank}",
        )
        expected_merge_records.append(((left_id << 32) | right_id, rank))
    expected_merge_records.sort()
    merge_blob = asset.section_bytes(SECTION_MERGE_LOOKUP)
    require(len(merge_blob) == len(merges) * 12, "MERGE_LOOKUP byte size mismatch")
    actual_merge_records = list(struct.iter_unpack("<QI", merge_blob))
    require(actual_merge_records == expected_merge_records, "MERGE_LOOKUP data mismatch")

    metadata = json.loads(asset.section_bytes(SECTION_METADATA_JSON).decode("utf-8"))
    require(metadata["normalizer"] == {"type": "NFC"}, "Metadata normalizer mismatch")
    require(metadata["pre_tokenizer"] == tokenizer_json["pre_tokenizer"], "Pretokenizer mismatch")
    require(metadata["decoder"] == tokenizer_json["decoder"], "Decoder mismatch")
    require(metadata["model_config"]["vocab_size"] == model_config["vocab_size"], "Metadata vocab mismatch")
    require(metadata["generation_config"] == generation_config, "Generation config mismatch")
    for filename, record in metadata["source_files"].items():
        path = model_dir / filename
        require(path.stat().st_size == record["size"], f"Metadata source size mismatch {filename}")
        require(sha256_file(path) == record["sha256"], f"Metadata source hash mismatch {filename}")

    return {
        "offsets": offsets,
        "token_blob": token_blob,
        "token_flags": token_flags,
        "metadata": metadata,
    }


def asset_decode(
    token_ids: list[int],
    *,
    offsets: list[int],
    token_blob: bytes,
    token_flags: bytes,
    skip_special_tokens: bool,
) -> str:
    decoded = bytearray()
    for token_id in token_ids:
        require(0 <= token_id < len(token_flags), f"Undecodable token ID {token_id}")
        flags = token_flags[token_id]
        require(bool(flags & TOKEN_FLAG_VALID), f"Invalid token ID {token_id}")
        if skip_special_tokens and flags & TOKEN_FLAG_SPECIAL:
            continue
        decoded.extend(token_blob[offsets[token_id] : offsets[token_id + 1]])
    return decoded.decode("utf-8", errors="replace")


def validate_golden(
    golden_path: Path,
    model_dir: Path,
    asset_tables: dict[str, Any],
) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise VerificationError(
            "transformers is required; run this verifier with conda environment llm_fpga"
        ) from error

    golden = load_json(golden_path)
    require(golden.get("schema_version") == 1, "Golden schema_version mismatch")
    require(golden.get("kind") == "qwen_tokenizer_golden_corpus", "Golden kind mismatch")
    for filename, record in golden.get("source_files", {}).items():
        path = model_dir / filename
        require(path.stat().st_size == record["size"], f"Golden source size mismatch {filename}")
        require(sha256_file(path) == record["sha256"], f"Golden source hash mismatch {filename}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    require(tokenizer.vocab_size == golden["base_vocab_size"], "Golden base vocab mismatch")
    require(len(tokenizer) == golden["tokenizer_id_count"], "Golden tokenizer length mismatch")
    require(tokenizer.eos_token_id == golden["eos_token_id"], "Golden EOS mismatch")
    require(tokenizer.pad_token_id == golden["pad_token_id"], "Golden PAD mismatch")

    cases = golden.get("cases", [])
    require(bool(cases), "Golden corpus has no cases")
    seen_names: set[str] = set()
    for case in cases:
        name = case["name"]
        require(name not in seen_names, f"Duplicate golden case {name}")
        seen_names.add(name)
        encoded = tokenizer(
            case["text"],
            add_special_tokens=bool(case["add_special_tokens"]),
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = [int(token_id) for token_id in encoded.input_ids]
        require(input_ids == case["input_ids"], f"Golden input_ids mismatch for {name}")
        require(
            tokenizer.convert_ids_to_tokens(input_ids) == case["tokens"],
            f"Golden token strings mismatch for {name}",
        )
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_skip = tokenizer.decode(input_ids, skip_special_tokens=True)
        require(decoded == case["decoded"], f"HF decoded mismatch for {name}")
        require(
            decoded_skip == case["decoded_skip_special_tokens"],
            f"HF skip-special decoded mismatch for {name}",
        )
        require(
            asset_decode(
                input_ids,
                offsets=asset_tables["offsets"],
                token_blob=asset_tables["token_blob"],
                token_flags=asset_tables["token_flags"],
                skip_special_tokens=False,
            )
            == case["decoded"],
            f"Asset decoded mismatch for {name}",
        )
        require(
            asset_decode(
                input_ids,
                offsets=asset_tables["offsets"],
                token_blob=asset_tables["token_blob"],
                token_flags=asset_tables["token_flags"],
                skip_special_tokens=True,
            )
            == case["decoded_skip_special_tokens"],
            f"Asset skip-special decoded mismatch for {name}",
        )
    return len(cases)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--manifest", type=Path, help="Defaults to <asset>.manifest.json")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--unicode-sources", type=Path, default=DEFAULT_UNICODE_SOURCES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    asset_path = args.asset.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else asset_path.with_suffix(asset_path.suffix + ".manifest.json")
    )
    try:
        asset = parse_asset(asset_path)
        unicode_metadata = validate_unicode_and_added_tables(
            asset, args.model_dir.resolve(), args.unicode_sources.resolve()
        )
        manifest = validate_manifest(
            asset_path,
            asset,
            manifest_path,
            args.model_dir.resolve(),
            unicode_metadata["sources"],
        )
        require(
            manifest.get("unicode_table_generation") == unicode_metadata["generator"],
            "Manifest Unicode-table generator provenance mismatch",
        )
        tables = validate_binary_tables(asset, args.model_dir.resolve())
        require(
            tables["metadata"].get("unicode_tables") == unicode_metadata,
            "Metadata Unicode-table provenance mismatch",
        )
        case_count = validate_golden(args.golden.resolve(), args.model_dir.resolve(), tables)
    except (AssetFormatError, VerificationError, OSError, ValueError, struct.error) as error:
        print(f"FAIL tokenizer asset verification: {error}", file=sys.stderr)
        return 1

    print("PASS tokenizer asset verification")
    print(f"asset={asset_path}")
    print(f"asset_size={len(asset.data)}")
    print(f"asset_sha256={sha256_file(asset_path)}")
    print(f"sections={len(asset.sections)}")
    print(f"golden_cases={case_count}")
    print(
        "counts="
        f"base_vocab:{asset.header['base_vocab_count']} "
        f"added:{asset.header['added_token_count']} "
        f"tokenizer_ids:{asset.header['token_count']} "
        f"model_vocab:{asset.header['model_vocab_size']} "
        f"merges:{asset.header['merge_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
