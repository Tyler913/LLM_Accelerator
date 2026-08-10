# QTKBPE1 Tokenizer Asset Format

## Scope

`QTKBPE1` is the deterministic asset boundary between the checked-in Qwen
tokenizer sources and the PS-native tokenizer/detokenizer. It preserves
the exact Qwen ByteLevel-BPE vocabulary, merge ranks, added-token metadata,
normalizer/pretokenizer contract, and token-to-byte decoding data.

The Unicode tables are generated from the exact Rust dependencies used by the
installed `tokenizers 0.22.2`: Unicode-9 NFC data from
`unicode-normalization-alignments 0.1.12` and Unicode-16 regex data from
`regex-syntax 0.8.8`. Their archives and relevant source/license files are
cryptographically pinned by `unicode_sources.lock.json`.

## Byte Order and Alignment

- All integers are little-endian.
- The fixed prefix occupies 128 bytes.
- The section table immediately follows the prefix.
- The complete header and every section start are aligned to 64 bytes.
- Alignment padding is zero and is covered by the payload SHA-256.
- The final section ends at the declared file size; there is no required tail
  padding.

## Fixed Prefix

The prefix is encoded by `struct` format `<8s14IQQ32s` and then zero-padded to
128 bytes.

| Field | Type | Required Value / Meaning |
| --- | --- | --- |
| `magic` | `char[8]` | `QTKBPE1\0` |
| `version` | `u32` | `1` |
| `header_size` | `u32` | aligned end of prefix and section table |
| `endian_tag` | `u32` | `0x01020304` |
| `format_flags` | `u32` | bit 0: output ID is `256 + rank`; bit 1: token bytes are decoded bytes |
| `section_count` | `u32` | number of section descriptors |
| `base_vocab_count` | `u32` | BPE base vocabulary count |
| `added_token_count` | `u32` | added-token count after the base vocabulary |
| `token_count` | `u32` | count of IDs with a decode representation |
| `model_vocab_size` | `u32` | LM-head output vocabulary count |
| `merge_count` | `u32` | number of BPE merge ranks |
| `eos_token_id` | `u32` | EOS ID |
| `pad_token_id` | `u32` | PAD ID |
| `unk_token_id` | `u32` | `0xFFFFFFFF` when no UNK token exists |
| `normalizer_kind` | `u32` | `1` for NFC |
| `file_size` | `u64` | complete binary size |
| `payload_size` | `u64` | `file_size - header_size` |
| `payload_sha256` | `u8[32]` | SHA-256 of every byte from `header_size` to EOF |

## Section Descriptor

Each descriptor uses `struct` format `<16sQQII32s` (72 bytes).

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `char[16]` | NUL-terminated ASCII name |
| `offset` | `u64` | aligned file offset |
| `size` | `u64` | section bytes |
| `count` | `u32` | logical records/items |
| `record_size` | `u32` | fixed record size, or zero for variable data |
| `sha256` | `u8[32]` | SHA-256 over this section only |

## Required Sections

### `BYTE_TO_TOKEN`

Exactly 256 little-endian `u32` values. Entry `b` is the initial base token ID
for input byte `b` after the GPT-2 reversible byte-to-Unicode mapping.

### `MERGE_LOOKUP`

One 12-byte `<QI` record per merge:

- `u64 pair_key = (left_token_id << 32) | right_token_id`
- `u32 rank`

Records are sorted by ascending `pair_key` and contain no duplicates. The
current Qwen asset proves for every record that the merged output token ID is
`256 + rank`. A small PS runtime can therefore binary-search this section
without storing a second output-ID field.

### `TOKEN_OFFSETS`

`token_count + 1` little-endian `u32` offsets into `TOKEN_BYTES`. Token `i`
occupies `[offset[i], offset[i + 1])`.

### `TOKEN_BYTES`

Variable token data:

- Base BPE entries contain raw bytes after reversing ByteLevel encoding.
- Added-token entries contain their literal UTF-8 bytes.

Concatenating these slices and performing one streaming UTF-8 decode with
replacement-on-error implements the decoder data path. A generated token may
end in the middle of a UTF-8 sequence, so a UI must retain incomplete bytes
between token events rather than decode each token independently.

### `TOKEN_FLAGS`

One byte per tokenizer ID:

- bit 0: valid decode entry
- bit 1: added token
- bit 2: special token (removed by `skip_special_tokens`)

### `ADDED_PROPS`

One byte per added token, ordered from `base_vocab_count` upward. Bits encode
`special`, `single_word`, `lstrip`, `rstrip`, and `normalized`. The current
Qwen asset has only the `special` bit set where appropriate; the exporter and
C runtime reject nonzero behavioral flags because this runtime implements the
model's exact literal leftmost-longest added-token contract.

### Unicode NFC Sections

- `U_NFC_CCC`: sorted 8-byte `<IB3x>` codepoint/combining-class records;
- `U_NFC_DECOMP`: sorted 12-byte `<III>` records containing codepoint,
  sequence offset, and sequence length;
- `U_NFC_SEQ`: little-endian `u32` fully decomposed codepoints;
- `U_NFC_COMPOSE`: sorted 12-byte `<QI>` pair-key/composed-codepoint records.

Hangul decomposition and composition use the standard algorithm and are not
stored in the asset.

### Unicode Split Sections

- `U_LETTER`: sorted inclusive `<II>` ranges for regex `\p{L}`;
- `U_NUMBER`: sorted inclusive `<II>` ranges for regex `\p{N}`;
- `U_SPACE`: sorted inclusive `<II>` ranges for Unicode regex `\s`;
- `U_FOLD_ASCII`: sorted `<II>` codepoint/lowercase-ASCII records for the
  simple-case-fold classes used by Qwen's contraction alternative.

### `METADATA_JSON`

Canonical UTF-8 JSON with a final newline. It records source-file hashes,
normalizer, Qwen Unicode split expression, ByteLevel configuration, added-token
properties, chat template, model/generation token IDs, and proven invariants.

## Integrity and Provenance

The binary contains a payload SHA-256 and a SHA-256 for every section. The
generated `<asset>.manifest.json` additionally records:

- whole-file SHA-256 and byte size;
- all offsets, sizes, record counts, and section hashes;
- hashes and sizes of all source tokenizer/config files;
- exporter source hash;
- the tokenizer/model vocabulary boundary.

`verify_tokenizer_asset.py` rejects mismatched source files, malformed layout,
non-zero padding, overlaps, bad hashes, broken BPE invariants, table differences,
or golden-corpus drift.

## Model-only IDs

The tokenizer decodes IDs `0..151668`; the model LM head has IDs through
`151935`. IDs `151669..151935` have no tokenizer representation. The C runtime
returns `QTK_ERR_MODEL_ONLY_TOKEN` for this range. Generation must stop and
report the ID; it must not modulo-map, silently drop, or replace it with EOS.
