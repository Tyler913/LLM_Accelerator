"""Build compact Unicode tables from the exact Rust crates used by tokenizers."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
import struct
from typing import Any

from tokenizer_asset_format import sha256_file


LOCK_PATH = Path(__file__).resolve().with_name("unicode_sources.lock.json")
TARGET_FOLD_LETTERS = "strevml d".replace(" ", "")

SECTION_U_NFC_CCC = "U_NFC_CCC"
SECTION_U_NFC_DECOMP = "U_NFC_DECOMP"
SECTION_U_NFC_SEQ = "U_NFC_SEQ"
SECTION_U_NFC_COMPOSE = "U_NFC_COMPOSE"
SECTION_U_LETTER = "U_LETTER"
SECTION_U_NUMBER = "U_NUMBER"
SECTION_U_SPACE = "U_SPACE"
SECTION_U_FOLD_ASCII = "U_FOLD_ASCII"

RUST_CHAR_PATTERN = r"'(?:\\u\{[0-9A-Fa-f]+\}|\\.|[^'\\])'"


class UnicodeTableError(ValueError):
    pass


def parse_rust_char(literal: str) -> int:
    inner = literal[1:-1]
    if inner.startswith(r"\u{") and inner.endswith("}"):
        return int(inner[3:-1], 16)
    escapes = {
        r"\0": 0,
        r"\t": 9,
        r"\n": 10,
        r"\r": 13,
        r"\'": 39,
        r'\"': 34,
        r"\\": 92,
    }
    if inner in escapes:
        return escapes[inner]
    if len(inner) != 1:
        raise UnicodeTableError(f"Unsupported Rust char literal {literal!r}")
    return ord(inner)


def constant_body(text: str, name: str) -> str:
    match = re.search(
        rf"(?:pub(?:\(crate\))?\s+)?const\s+{name}\s*:[^=]*=\s*&\[",
        text,
    )
    if match is None:
        raise UnicodeTableError(f"Cannot find Rust table {name}")
    end = text.find("];", match.end())
    if end < 0:
        raise UnicodeTableError(f"Cannot find end of Rust table {name}")
    return text[match.end() : end]


def parse_range_table(text: str, name: str) -> list[tuple[int, int]]:
    body = constant_body(text, name)
    pairs = re.findall(
        rf"\(\s*({RUST_CHAR_PATTERN})\s*,\s*({RUST_CHAR_PATTERN})\s*\)",
        body,
    )
    ranges = [(parse_rust_char(left), parse_rust_char(right)) for left, right in pairs]
    if not ranges or any(left > right for left, right in ranges):
        raise UnicodeTableError(f"Malformed or empty range table {name}")
    if any(ranges[index - 1][1] >= ranges[index][0] for index in range(1, len(ranges))):
        raise UnicodeTableError(f"Overlapping/unsorted range table {name}")
    return ranges


def parse_ccc(text: str) -> list[tuple[int, int]]:
    body = constant_body(text, "CANONICAL_COMBINING_CLASS_KV")
    packed_values = [int(value, 16) for value in re.findall(r"0x[0-9A-Fa-f]+", body)]
    records = sorted({(value >> 8, value & 0xFF) for value in packed_values if value != 0})
    if any(codepoint == 0 or ccc == 0 or codepoint > 0x10FFFF for codepoint, ccc in records):
        raise UnicodeTableError("Invalid canonical combining class record")
    return records


def parse_decompositions(text: str) -> tuple[list[tuple[int, int, int]], list[int]]:
    body = constant_body(text, "CANONICAL_DECOMPOSED_KV")
    entry_pattern = re.compile(rf"\(\s*(0x[0-9A-Fa-f]+)\s*,\s*&\[(.*?)\]\s*\)", re.S)
    mappings: dict[int, tuple[int, ...]] = {}
    for key_text, sequence_text in entry_pattern.findall(body):
        codepoint = int(key_text, 16)
        sequence = tuple(
            parse_rust_char(literal)
            for literal in re.findall(RUST_CHAR_PATTERN, sequence_text)
        )
        if codepoint == 0 and not sequence:
            continue
        if codepoint == 0 or not sequence or codepoint in mappings:
            raise UnicodeTableError("Invalid canonical decomposition entry")
        mappings[codepoint] = sequence
    if not mappings:
        raise UnicodeTableError("No canonical decomposition mappings parsed")

    records: list[tuple[int, int, int]] = []
    flattened: list[int] = []
    for codepoint, sequence in sorted(mappings.items()):
        records.append((codepoint, len(flattened), len(sequence)))
        flattened.extend(sequence)
    return records, flattened


def parse_compositions(text: str) -> list[tuple[int, int]]:
    body = constant_body(text, "COMPOSITION_TABLE_KV")
    entry_pattern = re.compile(
        rf"\(\s*(0x[0-9A-Fa-f]+)\s*,\s*({RUST_CHAR_PATTERN})\s*\)"
    )
    records: dict[int, int] = {}
    for key_text, result_literal in entry_pattern.findall(body):
        packed = int(key_text, 16)
        result = parse_rust_char(result_literal)
        if packed == 0 and result == 0:
            continue
        left = packed >> 16
        right = packed & 0xFFFF
        key = (left << 32) | right
        if left == 0 or right == 0 or key in records:
            raise UnicodeTableError("Invalid BMP composition entry")
        records[key] = result

    astral_start = text.find("fn composition_table_astral")
    astral_end = text.find("\n}\n", astral_start)
    if astral_start < 0 or astral_end < 0:
        raise UnicodeTableError("Cannot find astral composition function")
    astral_body = text[astral_start:astral_end]
    astral_pattern = re.compile(
        rf"\(\s*({RUST_CHAR_PATTERN})\s*,\s*({RUST_CHAR_PATTERN})\s*\)\s*"
        rf"=>\s*Some\(\s*({RUST_CHAR_PATTERN})\s*\)"
    )
    for left_literal, right_literal, result_literal in astral_pattern.findall(astral_body):
        left = parse_rust_char(left_literal)
        right = parse_rust_char(right_literal)
        key = (left << 32) | right
        if key in records:
            raise UnicodeTableError("Duplicate astral composition entry")
        records[key] = parse_rust_char(result_literal)
    if not records:
        raise UnicodeTableError("No composition mappings parsed")
    return sorted(records.items())


def parse_simple_folds(text: str) -> list[tuple[int, int]]:
    body = constant_body(text, "CASE_FOLDING_SIMPLE")
    entry_pattern = re.compile(
        rf"\(\s*({RUST_CHAR_PATTERN})\s*,\s*&\[(.*?)\]\s*\)", re.S
    )
    graph: dict[int, set[int]] = {}
    for source_literal, targets_text in entry_pattern.findall(body):
        source = parse_rust_char(source_literal)
        targets = [
            parse_rust_char(literal)
            for literal in re.findall(RUST_CHAR_PATTERN, targets_text)
        ]
        graph.setdefault(source, set()).update(targets)
        for target in targets:
            graph.setdefault(target, set()).add(source)

    result: dict[int, int] = {}
    for ascii_letter in TARGET_FOLD_LETTERS:
        target = ord(ascii_letter)
        queue = deque([target])
        equivalents = {target}
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, ()):
                if neighbor not in equivalents:
                    equivalents.add(neighbor)
                    queue.append(neighbor)
        for codepoint in equivalents:
            previous = result.get(codepoint)
            if previous is not None and previous != target:
                raise UnicodeTableError("Simple-fold class maps to two contraction letters")
            result[codepoint] = target
    return sorted(result.items())


def pack_records(format_string: str, records: list[tuple[int, ...]]) -> bytes:
    return b"".join(struct.pack(format_string, *record) for record in records)


def verify_sources(source_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    provenance: dict[str, Any] = {}
    for package_name, package_record in sorted(lock["packages"].items()):
        version = package_record["version"]
        package_root = source_root / f"{package_name}-{version}"
        if not package_root.is_dir():
            raise UnicodeTableError(
                f"Missing {package_root}; run fetch_unicode_sources.py first"
            )
        file_records: dict[str, Any] = {}
        for relative_name, expected_hash in package_record["files"].items():
            path = package_root / relative_name
            if not path.is_file() or sha256_file(path) != expected_hash.upper():
                raise UnicodeTableError(f"Pinned source mismatch: {path}")
            paths[f"{package_name}:{relative_name}"] = path
            file_records[relative_name] = {
                "size": path.stat().st_size,
                "sha256": expected_hash.upper(),
            }
        provenance[package_name] = {
            "version": version,
            "archive_sha256": package_record["archive_sha256"].upper(),
            "url": package_record["url"],
            "normalization_unicode_version": package_record["normalization_unicode_version"],
            "regex_unicode_version": package_record["regex_unicode_version"],
            "files": file_records,
        }
    return paths, provenance


def build_unicode_sections(
    source_root: Path,
) -> tuple[list[tuple[str, bytes, int, int]], dict[str, Any]]:
    paths, provenance = verify_sources(source_root)
    normalization_text = paths[
        "unicode-normalization-alignments:src/tables.rs"
    ].read_text(encoding="utf-8")
    category_text = paths[
        "regex-syntax:src/unicode_tables/general_category.rs"
    ].read_text(encoding="utf-8")
    space_text = paths[
        "regex-syntax:src/unicode_tables/perl_space.rs"
    ].read_text(encoding="utf-8")
    fold_text = paths[
        "regex-syntax:src/unicode_tables/case_folding_simple.rs"
    ].read_text(encoding="utf-8")

    if "UNICODE_VERSION: (u64, u64, u64) = (9, 0, 0)" not in normalization_text:
        raise UnicodeTableError("Normalization source is not Unicode 9.0.0")
    if "Unicode version: 16.0.0." not in category_text or \
       "Unicode version: 16.0.0." not in space_text or \
       "Unicode version: 16.0.0." not in fold_text:
        raise UnicodeTableError("Regex source tables are not Unicode 16.0.0")

    ccc = parse_ccc(normalization_text)
    decompositions, decomposition_sequence = parse_decompositions(normalization_text)
    compositions = parse_compositions(normalization_text)
    letters = parse_range_table(category_text, "LETTER")
    numbers = parse_range_table(category_text, "NUMBER")
    spaces = parse_range_table(space_text, "WHITE_SPACE")
    folds = parse_simple_folds(fold_text)

    sections = [
        (SECTION_U_NFC_CCC, pack_records("<IB3x", ccc), len(ccc), 8),
        (
            SECTION_U_NFC_DECOMP,
            pack_records("<III", decompositions),
            len(decompositions),
            12,
        ),
        (
            SECTION_U_NFC_SEQ,
            struct.pack(f"<{len(decomposition_sequence)}I", *decomposition_sequence),
            len(decomposition_sequence),
            4,
        ),
        (
            SECTION_U_NFC_COMPOSE,
            b"".join(struct.pack("<QI", key, value) for key, value in compositions),
            len(compositions),
            12,
        ),
        (SECTION_U_LETTER, pack_records("<II", letters), len(letters), 8),
        (SECTION_U_NUMBER, pack_records("<II", numbers), len(numbers), 8),
        (SECTION_U_SPACE, pack_records("<II", spaces), len(spaces), 8),
        (SECTION_U_FOLD_ASCII, pack_records("<II", folds), len(folds), 8),
    ]
    metadata = {
        "generator": {
            "unicode_sources.lock.json": {
                "size": LOCK_PATH.stat().st_size,
                "sha256": sha256_file(LOCK_PATH),
            },
            "unicode_table_builder.py": {
                "size": Path(__file__).resolve().stat().st_size,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "sources": provenance,
        "normalization": {
            "form": "NFC",
            "unicode_version": "9.0.0",
            "canonical_combining_class_records": len(ccc),
            "fully_decomposed_records": len(decompositions),
            "decomposition_codepoints": len(decomposition_sequence),
            "composition_records": len(compositions),
            "hangul": "algorithmic Unicode NFC decomposition/composition",
        },
        "regex": {
            "unicode_version": "16.0.0",
            "letter_ranges": len(letters),
            "number_ranges": len(numbers),
            "perl_space_ranges": len(spaces),
            "contraction_simple_fold_records": len(folds),
        },
    }
    return sections, metadata
