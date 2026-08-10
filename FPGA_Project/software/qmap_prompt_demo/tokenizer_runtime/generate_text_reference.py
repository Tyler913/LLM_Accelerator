"""Generate ignored HF references for the complete C text-tokenizer host test."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "Qwen3-0.6B-Base"
DEFAULT_GOLDEN = (
    REPO_ROOT
    / "FPGA_Project"
    / "software"
    / "qmap_prompt_demo"
    / "tokenizer"
    / "golden_corpus.json"
)
DEFAULT_DIFFERENTIAL = Path(__file__).resolve().with_name(
    "text_differential_corpus.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Temp"
    / "qmap_prompt_demo_tokenizer"
    / "qtk_text_reference.bin"
)

MAGIC = b"QTXTRF1\0"
VERSION = 1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_text(case: dict) -> str:
    has_text = "text" in case
    has_codepoints = "codepoints" in case
    if has_text == has_codepoints:
        raise ValueError(f"Case {case.get('name')} must have text xor codepoints")
    if has_text:
        return str(case["text"])
    return "".join(chr(int(value, 16)) for value in case["codepoints"])


def build_fixture(
    model_dir: Path,
    golden_path: Path,
    differential_path: Path,
) -> tuple[bytes, int, int, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required; use conda run -n llm_fpga") from error

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    golden = load_json(golden_path)
    differential = load_json(differential_path)
    if golden.get("schema_version") != 1 or differential.get("schema_version") != 1:
        raise ValueError("Unsupported corpus schema")

    records: list[tuple[str, str, list[int], str, str]] = []
    names: set[str] = set()
    for case in golden["cases"]:
        name = "golden/" + case["name"]
        text = case["text"]
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        ids = [int(token_id) for token_id in encoded]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_skip = tokenizer.decode(ids, skip_special_tokens=True)
        if ids != case["input_ids"] or decoded != case["decoded"] or \
           decoded_skip != case["decoded_skip_special_tokens"]:
            raise ValueError(f"Tracked golden drift for {case['name']}")
        records.append((name, text, ids, decoded, decoded_skip))
        names.add(name)

    for case in differential["cases"]:
        name = "differential/" + str(case["name"])
        if name in names:
            raise ValueError(f"Duplicate case name {name}")
        names.add(name)
        text = materialize_text(case)
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        ids = [int(token_id) for token_id in encoded]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_skip = tokenizer.decode(ids, skip_special_tokens=True)
        records.append((name, text, ids, decoded, decoded_skip))

    output = bytearray(
        struct.pack(
            "<8sIIII",
            MAGIC,
            VERSION,
            len(records),
            len(golden["cases"]),
            max(len(text.encode("utf-8")) for _, text, _, _, _ in records),
        )
    )
    for name, text, ids, decoded, decoded_skip in records:
        name_bytes = name.encode("utf-8")
        input_bytes = text.encode("utf-8")
        decoded_bytes = decoded.encode("utf-8")
        decoded_skip_bytes = decoded_skip.encode("utf-8")
        output.extend(
            struct.pack(
                "<IIIII",
                len(name_bytes),
                len(input_bytes),
                len(ids),
                len(decoded_bytes),
                len(decoded_skip_bytes),
            )
        )
        output.extend(name_bytes)
        output.extend(input_bytes)
        output.extend(struct.pack(f"<{len(ids)}I", *ids))
        output.extend(decoded_bytes)
        output.extend(decoded_skip_bytes)
    return bytes(output), len(records), len(golden["cases"]), len(differential["cases"])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--differential", type=Path, default=DEFAULT_DIFFERENTIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        fixture, case_count, golden_count, differential_count = build_fixture(
            args.model_dir.resolve(), args.golden.resolve(), args.differential.resolve()
        )
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_bytes(fixture)
        temporary.replace(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL text reference generation: {error}", file=sys.stderr)
        return 1

    print("PASS text reference generation")
    print(f"fixture={output_path}")
    print(f"fixture_size={len(fixture)}")
    print(f"fixture_sha256={sha256(fixture).hexdigest().upper()}")
    print(f"cases={case_count} golden={golden_count} differential={differential_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
