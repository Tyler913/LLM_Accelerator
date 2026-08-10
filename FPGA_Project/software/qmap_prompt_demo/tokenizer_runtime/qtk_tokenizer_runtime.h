#ifndef QTK_TOKENIZER_RUNTIME_H
#define QTK_TOKENIZER_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QTK_FORMAT_VERSION 1u
#define QTK_TOKEN_FLAG_VALID 0x01u
#define QTK_TOKEN_FLAG_ADDED 0x02u
#define QTK_TOKEN_FLAG_SPECIAL 0x04u
#define QTK_ADDED_PROP_SPECIAL 0x01u
#define QTK_ADDED_PROP_SINGLE_WORD 0x02u
#define QTK_ADDED_PROP_LSTRIP 0x04u
#define QTK_ADDED_PROP_RSTRIP 0x08u
#define QTK_ADDED_PROP_NORMALIZED 0x10u

typedef enum {
    QTK_OK = 0,
    QTK_ERR_NULL = -1,
    QTK_ERR_TOO_SMALL = -2,
    QTK_ERR_MAGIC = -3,
    QTK_ERR_VERSION = -4,
    QTK_ERR_ENDIAN = -5,
    QTK_ERR_FORMAT = -6,
    QTK_ERR_RANGE = -7,
    QTK_ERR_HASH = -8,
    QTK_ERR_SECTION = -9,
    QTK_ERR_TOKEN_ID = -10,
    QTK_ERR_WORKSPACE = -11,
    QTK_ERR_EMIT = -12,
    QTK_ERR_NOT_FOUND = -13,
    QTK_ERR_UTF8 = -14,
    QTK_ERR_UNICODE_WORKSPACE = -15,
    QTK_ERR_PIECE_WORKSPACE = -16,
    QTK_ERR_OUTPUT = -17,
    QTK_ERR_MODEL_ONLY_TOKEN = -18
} qtk_status_t;

typedef struct {
    const uint8_t *data;
    size_t size;
    uint32_t count;
    uint32_t record_size;
} qtk_section_view_t;

typedef struct {
    const uint8_t *data;
    size_t data_size;
    uint32_t base_vocab_count;
    uint32_t added_token_count;
    uint32_t token_count;
    uint32_t model_vocab_size;
    uint32_t merge_count;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    uint32_t unk_token_id;
    uint32_t normalizer_kind;
    qtk_section_view_t byte_to_token;
    qtk_section_view_t merge_lookup;
    qtk_section_view_t token_offsets;
    qtk_section_view_t token_bytes;
    qtk_section_view_t token_flags;
    qtk_section_view_t added_props;
    qtk_section_view_t unicode_nfc_ccc;
    qtk_section_view_t unicode_nfc_decomp;
    qtk_section_view_t unicode_nfc_sequence;
    qtk_section_view_t unicode_nfc_compose;
    qtk_section_view_t unicode_letter_ranges;
    qtk_section_view_t unicode_number_ranges;
    qtk_section_view_t unicode_space_ranges;
    qtk_section_view_t unicode_fold_ascii;
    qtk_section_view_t metadata_json;
} qtk_asset_t;

typedef struct {
    const uint8_t *bytes;
    size_t length;
    uint8_t flags;
} qtk_token_slice_t;

typedef int (*qtk_emit_bytes_fn)(
    void *context,
    const uint8_t *bytes,
    size_t length);

const char *qtk_status_name(qtk_status_t status);

/*
 * Parse and validate an in-memory QTKBPE1 asset without allocating memory.
 * The caller owns data and must keep it alive and unchanged while asset is used.
 * Container layout, all payload/section SHA-256 values, and table invariants are
 * checked before the view is published.
 */
qtk_status_t qtk_asset_init(
    qtk_asset_t *asset,
    const void *data,
    size_t data_size);

qtk_status_t qtk_token_slice(
    const qtk_asset_t *asset,
    uint32_t token_id,
    qtk_token_slice_t *slice);

/*
 * Emit raw token bytes in token order. UTF-8 decoding is intentionally left to
 * the consumer so a multibyte sequence split across token boundaries remains
 * intact. The callback must consume/copy bytes before returning.
 */
qtk_status_t qtk_detokenize_ids(
    const qtk_asset_t *asset,
    const uint32_t *token_ids,
    size_t token_id_count,
    int skip_special_tokens,
    qtk_emit_bytes_fn emit,
    void *emit_context);

qtk_status_t qtk_find_merge_rank(
    const qtk_asset_t *asset,
    uint32_t left_token_id,
    uint32_t right_token_id,
    uint32_t *rank);

/*
 * Encode exactly one already-normalized, already-pretokenized raw-byte piece.
 * ByteLevel mapping and BPE are performed in place in caller-owned workspace.
 * workspace_capacity must be at least raw_length uint32_t entries. On success,
 * the first *output_count entries are token IDs.
 *
 * This is not a complete Unicode tokenizer: NFC normalization, Qwen regex
 * pretokenization, and added/special-token matching are outside this API.
 */
qtk_status_t qtk_encode_raw_piece(
    const qtk_asset_t *asset,
    const uint8_t *raw_bytes,
    size_t raw_length,
    uint32_t *workspace,
    size_t workspace_capacity,
    size_t *output_count);

#ifdef __cplusplus
}
#endif

#endif
