# Qwen Tokenizer Asset Layer

This directory makes the tokenizer handoff reproducible without checking a
multi-megabyte generated binary into Git.

## Generate

Run from the repository root with the required project environment:

```powershell
conda run -n llm_fpga python FPGA_Project/software/qmap_prompt_demo/tokenizer/fetch_unicode_sources.py
```

This one-time/cached step downloads the two Cargo crates pinned by
`unicode_sources.lock.json`, verifies their archive and source/license hashes,
and extracts them only under ignored `Temp/`.

```powershell
conda run -n llm_fpga python FPGA_Project/software/qmap_prompt_demo/tokenizer/export_tokenizer_asset.py `
  --output Temp/qmap_prompt_demo_tokenizer/qwen3_tokenizer.qtk
```

The exporter cross-checks `tokenizer.json` against both `vocab.json` and
`merges.txt`, merges all added-token records from `tokenizer_config.json`, and
writes:

- `qwen3_tokenizer.qtk`
- `qwen3_tokenizer.qtk.manifest.json`

The default output is already the same path under `Temp/`.

## Rebuild the Golden Corpus

`golden_corpus.json` is intentionally small and tracked. Rebuild it only when
the tokenizer sources intentionally change:

```powershell
conda run -n llm_fpga python FPGA_Project/software/qmap_prompt_demo/tokenizer/generate_golden_corpus.py `
  --output Temp/qmap_prompt_demo_tokenizer/golden_corpus.json
```

Review the generated diff before replacing the tracked corpus. The cases cover
English, Chinese, whitespace, punctuation, emoji, decomposed Unicode requiring
NFC, Qwen chat-control tokens, and literal EOS.

## Verify

```powershell
conda run -n llm_fpga python FPGA_Project/software/qmap_prompt_demo/tokenizer/verify_tokenizer_asset.py `
  --asset Temp/qmap_prompt_demo_tokenizer/qwen3_tokenizer.qtk `
  --golden FPGA_Project/software/qmap_prompt_demo/tokenizer/golden_corpus.json
```

Verification covers the binary container, payload and section hashes, manifest,
source hashes, all byte/token, merge, added-token, and Unicode tables, live
Hugging Face results for every golden, and asset-backed detokenization.

See [FORMAT.md](FORMAT.md) for the complete binary contract.

See [UNICODE_SOURCES.md](UNICODE_SOURCES.md) for dependency provenance and
license locations. The complete allocation-free C11 consumer and its
Python-reference tests are in
[`../tokenizer_runtime`](../tokenizer_runtime/README.md).
