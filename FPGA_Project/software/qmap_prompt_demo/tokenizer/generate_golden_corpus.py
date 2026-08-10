"""Generate deterministic Qwen tokenizer goldens from the local HF assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

from tokenizer_asset_format import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "Qwen3-0.6B-Base"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("golden_corpus.json")

SOURCE_FILENAMES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "english",
        "category": "English words",
        "text": "The future of FPGA is",
        "add_special_tokens": False,
    },
    {
        "name": "chinese",
        "category": "Chinese and ASCII",
        "text": "你好，FPGA加速器。",
        "add_special_tokens": False,
    },
    {
        "name": "whitespace",
        "category": "spaces tabs LF and CRLF",
        "text": "  FPGA\tAI\n\nnext\r\nline  ",
        "add_special_tokens": False,
    },
    {
        "name": "punctuation",
        "category": "ASCII and Unicode punctuation",
        "text": "Hello, FPGA! (Q4): 1+1=2; ready?—yes.",
        "add_special_tokens": False,
    },
    {
        "name": "emoji",
        "category": "multi-byte emoji",
        "text": "FPGA 🚀🙂 + AI 🤖",
        "add_special_tokens": False,
    },
    {
        "name": "nfc_decomposed",
        "category": "NFC normalization",
        "text": "Cafe\u0301 nai\u0308ve",
        "add_special_tokens": False,
    },
    {
        "name": "special_tokens",
        "category": "Qwen chat control tokens",
        "text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "add_special_tokens": False,
    },
    {
        "name": "eos_literal",
        "category": "EOS literal",
        "text": "done<|endoftext|>",
        "add_special_tokens": False,
    },
)


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


def build_corpus(model_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer, __version__ as transformers_version
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; run this script with conda environment llm_fpga"
        ) from error

    source_paths = {name: model_dir / name for name in SOURCE_FILENAMES}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing tokenizer source files: " + ", ".join(missing))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    generated_cases: list[dict[str, Any]] = []
    for specification in CASES:
        encoded = tokenizer(
            specification["text"],
            add_special_tokens=specification["add_special_tokens"],
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = [int(token_id) for token_id in encoded.input_ids]
        generated_cases.append(
            {
                **specification,
                "input_ids": input_ids,
                "tokens": tokenizer.convert_ids_to_tokens(input_ids),
                "decoded": tokenizer.decode(input_ids, skip_special_tokens=False),
                "decoded_skip_special_tokens": tokenizer.decode(
                    input_ids, skip_special_tokens=True
                ),
            }
        )

    try:
        tokenizers_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        tokenizers_version = None

    return {
        "schema_version": 1,
        "kind": "qwen_tokenizer_golden_corpus",
        "model": model_dir.name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "base_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_id_count": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "transformers_version": transformers_version,
        "tokenizers_version": tokenizers_version,
        "source_files": {
            name: source_record(path) for name, path in sorted(source_paths.items())
        },
        "cases": generated_cases,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        corpus = build_corpus(args.model_dir.resolve())
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_bytes(serialized)
        temporary.replace(output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL golden corpus generation: {error}", file=sys.stderr)
        return 1

    print("PASS golden corpus generation")
    print(f"output={output}")
    print(f"cases={len(corpus['cases'])}")
    print(f"sha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
