"""Generate an ignored Python/Hugging Face reference fixture for C host tests."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import struct
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "Qwen3-0.6B-Base"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Temp"
    / "qmap_prompt_demo_tokenizer"
    / "qtk_host_reference.bin"
)

MAGIC = b"QTKREF1\0"
VERSION = 1
TOKEN_FLAG_VALID = 1 << 0
TOKEN_FLAG_ADDED = 1 << 1
TOKEN_FLAG_SPECIAL = 1 << 2

RAW_PIECES = (
    b"",
    b"Hello",
    b" Hello",
    b"token,token!",
    "你好".encode("utf-8"),
    "🙂".encode("utf-8"),
    b" \t\r\n",
    bytes(range(256)),
    b"a" * 257,
)


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


def build_fixture(model_dir: Path) -> tuple[bytes, int, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; use conda run -n llm_fpga"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    byte_encoder = gpt2_byte_encoder()
    byte_decoder = {symbol: value for value, symbol in byte_encoder.items()}
    base_vocab_count = int(tokenizer.vocab_size)
    token_count = len(tokenizer)
    added_decoder = tokenizer.added_tokens_decoder
    if set(added_decoder) != set(range(base_vocab_count, token_count)):
        raise RuntimeError("Added-token IDs are not contiguous after the base vocabulary")

    pieces: list[tuple[bytes, list[int]]] = []
    model = tokenizer.backend_tokenizer.model
    for raw_piece in RAW_PIECES:
        bytelevel_text = "".join(byte_encoder[value] for value in raw_piece)
        encoded = [int(token.id) for token in model.tokenize(bytelevel_text)]
        pieces.append((raw_piece, encoded))

    output = bytearray(
        struct.pack(
            "<8sIIII",
            MAGIC,
            VERSION,
            token_count,
            len(pieces),
            max(len(raw) for raw, _ in pieces),
        )
    )
    for token_id in range(token_count):
        if token_id < base_vocab_count:
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            if not isinstance(token_text, str):
                raise RuntimeError(f"No Python token string for base ID {token_id}")
            try:
                raw_bytes = bytes(byte_decoder[character] for character in token_text)
            except KeyError as error:
                raise RuntimeError(
                    f"Base token {token_id} contains a non-ByteLevel character"
                ) from error
            flags = TOKEN_FLAG_VALID
        else:
            added = added_decoder[token_id]
            raw_bytes = str(added).encode("utf-8")
            flags = TOKEN_FLAG_VALID | TOKEN_FLAG_ADDED
            if added.special:
                flags |= TOKEN_FLAG_SPECIAL
        output.extend(struct.pack("<II", len(raw_bytes), flags))
        output.extend(raw_bytes)

    for raw_piece, expected_ids in pieces:
        output.extend(struct.pack("<II", len(raw_piece), len(expected_ids)))
        output.extend(raw_piece)
        output.extend(struct.pack(f"<{len(expected_ids)}I", *expected_ids))

    return bytes(output), token_count, len(pieces)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        fixture, token_count, piece_count = build_fixture(args.model_dir.resolve())
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_bytes(fixture)
        temporary_path.replace(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL host reference generation: {error}", file=sys.stderr)
        return 1

    print("PASS host reference generation")
    print(f"fixture={output_path}")
    print(f"fixture_size={len(fixture)}")
    print(f"fixture_sha256={sha256(fixture).hexdigest().upper()}")
    print(f"token_slices={token_count}")
    print(f"raw_pieces={piece_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
