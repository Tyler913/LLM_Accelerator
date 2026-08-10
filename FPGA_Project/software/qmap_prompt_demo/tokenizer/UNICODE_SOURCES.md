# Unicode Table Provenance

The PS-native tokenizer does not substitute the host Python Unicode database
for the Rust tables used by Hugging Face Tokenizers. The generated QTK asset is
built from these exact upstream packages:

| Purpose | Package | Version | Unicode Version | Source |
| --- | --- | --- | --- | --- |
| NFC | `unicode-normalization-alignments` | 0.1.12 | 9.0.0 | <https://crates.io/crates/unicode-normalization-alignments/0.1.12> |
| Regex properties and simple folding | `regex-syntax` | 0.8.8 | 16.0.0 | <https://crates.io/crates/regex-syntax/0.8.8> |

These versions are the entries in the Python binding `Cargo.lock` for
`huggingface/tokenizers v0.22.2`. `unicode_sources.lock.json` records the Cargo
archive SHA-256 plus every table and license file SHA-256 consumed by the
exporter. `fetch_unicode_sources.py` rejects any mismatch before extraction.

The crates are not copied into Git. Their verified source and license files are
cached under `Temp/qmap_prompt_demo_unicode_sources/`; the derived compact
tables and their provenance are embedded in the generated QTK asset. Relevant
upstream license files are:

- `unicode-normalization-alignments-0.1.12/LICENSE-MIT`
- `unicode-normalization-alignments-0.1.12/LICENSE-APACHE`
- `unicode-normalization-alignments-0.1.12/COPYRIGHT`
- `regex-syntax-0.8.8/LICENSE-MIT`
- `regex-syntax-0.8.8/LICENSE-APACHE`
- `regex-syntax-0.8.8/src/unicode_tables/LICENSE-UNICODE`

This is a generated-table design, not a fork or partial transcription of either
library. The C runtime consumes only the compact audited records needed for
Qwen's fixed NFC and split semantics.
