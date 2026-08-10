"""Shared definitions for the deterministic Qwen ByteLevel-BPE asset format."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from typing import Final


MAGIC: Final = b"QTKBPE1\0"
FORMAT_VERSION: Final = 1
ENDIAN_TAG: Final = 0x01020304
ALIGNMENT: Final = 64

PREFIX_SIZE: Final = 128
PREFIX_STRUCT: Final = struct.Struct("<8s14IQQ32s")
SECTION_STRUCT: Final = struct.Struct("<16sQQII32s")

FLAG_MERGE_OUTPUT_ID_IS_256_PLUS_RANK: Final = 1 << 0
FLAG_TOKEN_BYTES_ARE_DECODED: Final = 1 << 1
REQUIRED_FORMAT_FLAGS: Final = (
    FLAG_MERGE_OUTPUT_ID_IS_256_PLUS_RANK
    | FLAG_TOKEN_BYTES_ARE_DECODED
)

NORMALIZER_NFC: Final = 1
INVALID_TOKEN_ID: Final = 0xFFFFFFFF

TOKEN_FLAG_VALID: Final = 1 << 0
TOKEN_FLAG_ADDED: Final = 1 << 1
TOKEN_FLAG_SPECIAL: Final = 1 << 2

ADDED_PROP_SPECIAL: Final = 1 << 0
ADDED_PROP_SINGLE_WORD: Final = 1 << 1
ADDED_PROP_LSTRIP: Final = 1 << 2
ADDED_PROP_RSTRIP: Final = 1 << 3
ADDED_PROP_NORMALIZED: Final = 1 << 4

SECTION_BYTE_TO_TOKEN: Final = "BYTE_TO_TOKEN"
SECTION_MERGE_LOOKUP: Final = "MERGE_LOOKUP"
SECTION_TOKEN_OFFSETS: Final = "TOKEN_OFFSETS"
SECTION_TOKEN_BYTES: Final = "TOKEN_BYTES"
SECTION_TOKEN_FLAGS: Final = "TOKEN_FLAGS"
SECTION_ADDED_PROPS: Final = "ADDED_PROPS"
SECTION_U_NFC_CCC: Final = "U_NFC_CCC"
SECTION_U_NFC_DECOMP: Final = "U_NFC_DECOMP"
SECTION_U_NFC_SEQ: Final = "U_NFC_SEQ"
SECTION_U_NFC_COMPOSE: Final = "U_NFC_COMPOSE"
SECTION_U_LETTER: Final = "U_LETTER"
SECTION_U_NUMBER: Final = "U_NUMBER"
SECTION_U_SPACE: Final = "U_SPACE"
SECTION_U_FOLD_ASCII: Final = "U_FOLD_ASCII"
SECTION_METADATA_JSON: Final = "METADATA_JSON"

REQUIRED_SECTIONS: Final = (
    SECTION_BYTE_TO_TOKEN,
    SECTION_MERGE_LOOKUP,
    SECTION_TOKEN_OFFSETS,
    SECTION_TOKEN_BYTES,
    SECTION_TOKEN_FLAGS,
    SECTION_ADDED_PROPS,
    SECTION_U_NFC_CCC,
    SECTION_U_NFC_DECOMP,
    SECTION_U_NFC_SEQ,
    SECTION_U_NFC_COMPOSE,
    SECTION_U_LETTER,
    SECTION_U_NUMBER,
    SECTION_U_SPACE,
    SECTION_U_FOLD_ASCII,
    SECTION_METADATA_JSON,
)


class AssetFormatError(ValueError):
    """Raised when a tokenizer asset violates its binary contract."""


@dataclass(frozen=True)
class SectionDescriptor:
    """One section-table entry from a tokenizer asset."""

    name: str
    offset: int
    size: int
    count: int
    record_size: int
    sha256_hex: str


@dataclass(frozen=True)
class ParsedAsset:
    """Validated top-level asset fields plus its immutable bytes."""

    data: bytes
    header: dict[str, int | str]
    sections: dict[str, SectionDescriptor]

    def section_bytes(self, name: str) -> bytes:
        descriptor = self.sections.get(name)
        if descriptor is None:
            raise AssetFormatError(f"Missing section {name}")
        return self.data[descriptor.offset : descriptor.offset + descriptor.size]


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    """Round value upward to an alignment boundary."""

    if value < 0 or alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("align_up requires a non-negative value and power-of-two alignment")
    return (value + alignment - 1) & ~(alignment - 1)


def sha256_bytes(data: bytes) -> str:
    """Return an uppercase SHA-256 digest for bytes."""

    return sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest for a file without loading it twice."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode_section_name(name: str) -> bytes:
    """Encode a section name into the fixed 16-byte ASCII field."""

    raw = name.encode("ascii")
    if not raw or len(raw) > 15 or b"\0" in raw:
        raise AssetFormatError(f"Invalid section name {name!r}")
    return raw.ljust(16, b"\0")


def pack_prefix(
    *,
    header_size: int,
    section_count: int,
    base_vocab_count: int,
    added_token_count: int,
    token_count: int,
    model_vocab_size: int,
    merge_count: int,
    eos_token_id: int,
    pad_token_id: int,
    unk_token_id: int,
    file_size: int,
    payload_size: int,
    payload_sha256_hex: str,
) -> bytes:
    """Pack the fixed prefix and its reserved zero padding."""

    payload_digest = bytes.fromhex(payload_sha256_hex)
    if len(payload_digest) != 32:
        raise AssetFormatError("payload SHA-256 must contain 32 bytes")
    packed = PREFIX_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        header_size,
        ENDIAN_TAG,
        REQUIRED_FORMAT_FLAGS,
        section_count,
        base_vocab_count,
        added_token_count,
        token_count,
        model_vocab_size,
        merge_count,
        eos_token_id,
        pad_token_id,
        unk_token_id,
        NORMALIZER_NFC,
        file_size,
        payload_size,
        payload_digest,
    )
    if len(packed) > PREFIX_SIZE:
        raise AssertionError("prefix struct exceeded its fixed region")
    return packed.ljust(PREFIX_SIZE, b"\0")


def pack_section_descriptor(descriptor: SectionDescriptor) -> bytes:
    """Pack one section descriptor."""

    digest = bytes.fromhex(descriptor.sha256_hex)
    if len(digest) != 32:
        raise AssetFormatError(f"Section {descriptor.name} has an invalid SHA-256")
    return SECTION_STRUCT.pack(
        encode_section_name(descriptor.name),
        descriptor.offset,
        descriptor.size,
        descriptor.count,
        descriptor.record_size,
        digest,
    )


def parse_asset(path: Path) -> ParsedAsset:
    """Load and fully validate an asset's generic binary container."""

    data = path.read_bytes()
    if len(data) < PREFIX_SIZE:
        raise AssetFormatError(f"Asset is smaller than the {PREFIX_SIZE}-byte prefix")

    unpacked = PREFIX_STRUCT.unpack_from(data, 0)
    (
        magic,
        version,
        header_size,
        endian_tag,
        format_flags,
        section_count,
        base_vocab_count,
        added_token_count,
        token_count,
        model_vocab_size,
        merge_count,
        eos_token_id,
        pad_token_id,
        unk_token_id,
        normalizer_kind,
        file_size,
        payload_size,
        payload_digest,
    ) = unpacked

    if magic != MAGIC:
        raise AssetFormatError(f"Bad magic: {magic!r}")
    if version != FORMAT_VERSION:
        raise AssetFormatError(f"Unsupported format version {version}")
    if endian_tag != ENDIAN_TAG:
        raise AssetFormatError(f"Bad endian tag 0x{endian_tag:08X}")
    if format_flags != REQUIRED_FORMAT_FLAGS:
        raise AssetFormatError(f"Unexpected format flags 0x{format_flags:08X}")
    if normalizer_kind != NORMALIZER_NFC:
        raise AssetFormatError(f"Unsupported normalizer kind {normalizer_kind}")
    if file_size != len(data):
        raise AssetFormatError(f"Header file_size={file_size}, actual={len(data)}")
    if header_size < PREFIX_SIZE or header_size % ALIGNMENT != 0:
        raise AssetFormatError(f"Invalid aligned header size {header_size}")
    if payload_size != len(data) - header_size:
        raise AssetFormatError(
            f"Header payload_size={payload_size}, actual={len(data) - header_size}"
        )
    table_end = PREFIX_SIZE + section_count * SECTION_STRUCT.size
    if table_end > header_size or header_size > len(data):
        raise AssetFormatError("Section table does not fit in the declared header")
    if any(data[PREFIX_STRUCT.size:PREFIX_SIZE]):
        raise AssetFormatError("Non-zero bytes in reserved prefix padding")
    if any(data[table_end:header_size]):
        raise AssetFormatError("Non-zero bytes in reserved header padding")

    actual_payload_digest = sha256(data[header_size:]).digest()
    if actual_payload_digest != payload_digest:
        raise AssetFormatError(
            "Payload SHA-256 mismatch: "
            f"header={payload_digest.hex().upper()} "
            f"actual={actual_payload_digest.hex().upper()}"
        )

    sections: dict[str, SectionDescriptor] = {}
    ordered_sections: list[SectionDescriptor] = []
    for index in range(section_count):
        entry_offset = PREFIX_SIZE + index * SECTION_STRUCT.size
        name_raw, offset, size, count, record_size, digest = SECTION_STRUCT.unpack_from(
            data, entry_offset
        )
        try:
            name = name_raw.split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise AssetFormatError(f"Section {index} name is not ASCII") from error
        if not name or name in sections:
            raise AssetFormatError(f"Empty or duplicate section name {name!r}")
        if offset < header_size or offset % ALIGNMENT != 0:
            raise AssetFormatError(f"Section {name} has invalid offset {offset}")
        if offset + size > len(data):
            raise AssetFormatError(f"Section {name} extends beyond the asset")
        if record_size != 0 and count * record_size != size:
            raise AssetFormatError(
                f"Section {name} count*record_size does not equal its size"
            )
        actual_digest = sha256(data[offset : offset + size]).digest()
        if actual_digest != digest:
            raise AssetFormatError(f"Section {name} SHA-256 mismatch")
        descriptor = SectionDescriptor(
            name=name,
            offset=offset,
            size=size,
            count=count,
            record_size=record_size,
            sha256_hex=digest.hex().upper(),
        )
        sections[name] = descriptor
        ordered_sections.append(descriptor)

    previous_end = header_size
    for descriptor in sorted(ordered_sections, key=lambda item: item.offset):
        if descriptor.offset < previous_end:
            raise AssetFormatError(f"Section {descriptor.name} overlaps its predecessor")
        if any(data[previous_end:descriptor.offset]):
            raise AssetFormatError(
                f"Non-zero alignment padding before section {descriptor.name}"
            )
        previous_end = descriptor.offset + descriptor.size
    if any(data[previous_end:]):
        raise AssetFormatError("Non-zero bytes after the final section")

    missing = [name for name in REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise AssetFormatError(f"Missing required sections: {', '.join(missing)}")

    header: dict[str, int | str] = {
        "version": version,
        "header_size": header_size,
        "endian_tag": endian_tag,
        "format_flags": format_flags,
        "section_count": section_count,
        "base_vocab_count": base_vocab_count,
        "added_token_count": added_token_count,
        "token_count": token_count,
        "model_vocab_size": model_vocab_size,
        "merge_count": merge_count,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "unk_token_id": unk_token_id,
        "normalizer_kind": normalizer_kind,
        "file_size": file_size,
        "payload_size": payload_size,
        "payload_sha256": payload_digest.hex().upper(),
    }
    return ParsedAsset(data=data, header=header, sections=sections)
