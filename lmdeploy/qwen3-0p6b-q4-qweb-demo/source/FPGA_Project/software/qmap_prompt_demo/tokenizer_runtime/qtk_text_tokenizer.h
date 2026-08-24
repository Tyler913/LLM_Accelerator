#ifndef QTK_TEXT_TOKENIZER_H
#define QTK_TEXT_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "qtk_tokenizer_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t unicode_codepoints;
    size_t piece_token_ids;
    size_t output_token_ids;
} qtk_tokenizer_requirements_t;

/*
 * Tokenize valid UTF-8 with the exact fixed Qwen pipeline represented by the
 * asset: literal leftmost-longest added-token isolation, Unicode-9 NFC,
 * Unicode-16 regex splitting, ByteLevel mapping, and BPE.
 *
 * All storage belongs to the caller. Capacities are counts of array elements,
 * not bytes. On a workspace/output error, requirements reports the first known
 * exact requirement; output_count remains zero and output_ids is unspecified.
 * Calling again with the reported capacity advances to the next sizing stage.
 */
qtk_status_t qtk_tokenize_utf8(
    const qtk_asset_t *asset,
    const uint8_t *input_utf8,
    size_t input_length,
    uint32_t *unicode_workspace,
    size_t unicode_capacity,
    uint32_t *piece_workspace,
    size_t piece_capacity,
    uint32_t *output_ids,
    size_t output_capacity,
    size_t *output_count,
    qtk_tokenizer_requirements_t *requirements);

#ifdef __cplusplus
}
#endif

#endif
