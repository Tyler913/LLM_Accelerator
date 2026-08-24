#include "qtk_tokenizer_runtime.h"

#include <limits.h>
#include <string.h>

#define QTK_PREFIX_SIZE 128u
#define QTK_PREFIX_USED_SIZE 112u
#define QTK_SECTION_RECORD_SIZE 72u
#define QTK_ALIGNMENT 64u
#define QTK_ENDIAN_TAG UINT32_C(0x01020304)
#define QTK_REQUIRED_FLAGS UINT32_C(0x00000003)
#define QTK_NORMALIZER_NFC 1u
#define QTK_INVALID_TOKEN_ID UINT32_MAX
#define QTK_SHA256_SIZE 32u

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    uint8_t block[64];
    size_t block_size;
} qtk_sha256_t;

static uint32_t qtk_read_le32(const uint8_t *data)
{
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
}

static uint64_t qtk_read_le64(const uint8_t *data)
{
    return ((uint64_t)qtk_read_le32(data)) |
           ((uint64_t)qtk_read_le32(data + 4) << 32);
}

static uint32_t qtk_rotr32(uint32_t value, uint32_t shift)
{
    return (value >> shift) | (value << (32u - shift));
}

static uint32_t qtk_read_be32(const uint8_t *data)
{
    return ((uint32_t)data[0] << 24) |
           ((uint32_t)data[1] << 16) |
           ((uint32_t)data[2] << 8) |
           ((uint32_t)data[3]);
}

static void qtk_write_be64(uint8_t *data, uint64_t value)
{
    size_t index;

    for (index = 0u; index < 8u; ++index) {
        uint32_t shift = (uint32_t)((7u - index) * 8u);
        data[index] = (uint8_t)(value >> shift);
    }
}

static void qtk_sha256_transform(qtk_sha256_t *context, const uint8_t *block)
{
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0u; index < 16u; ++index) {
        words[index] = qtk_read_be32(block + index * 4u);
    }
    for (index = 16u; index < 64u; ++index) {
        uint32_t value15 = words[index - 15u];
        uint32_t value2 = words[index - 2u];
        uint32_t sigma0 = qtk_rotr32(value15, 7u) ^
                          qtk_rotr32(value15, 18u) ^
                          (value15 >> 3);
        uint32_t sigma1 = qtk_rotr32(value2, 17u) ^
                          qtk_rotr32(value2, 19u) ^
                          (value2 >> 10);
        words[index] = words[index - 16u] + sigma0 +
                       words[index - 7u] + sigma1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0u; index < 64u; ++index) {
        uint32_t sum1 = qtk_rotr32(e, 6u) ^ qtk_rotr32(e, 11u) ^ qtk_rotr32(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 = h + sum1 + choice + constants[index] + words[index];
        uint32_t sum0 = qtk_rotr32(a, 2u) ^ qtk_rotr32(a, 13u) ^ qtk_rotr32(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void qtk_sha256_init(qtk_sha256_t *context)
{
    static const uint32_t initial_state[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };

    memcpy(context->state, initial_state, sizeof(initial_state));
    context->total_bytes = 0u;
    context->block_size = 0u;
}

static void qtk_sha256_update(
    qtk_sha256_t *context,
    const uint8_t *data,
    size_t length)
{
    while (length != 0u) {
        size_t available = sizeof(context->block) - context->block_size;
        size_t take = length < available ? length : available;

        memcpy(context->block + context->block_size, data, take);
        context->block_size += take;
        context->total_bytes += (uint64_t)take;
        data += take;
        length -= take;
        if (context->block_size == sizeof(context->block)) {
            qtk_sha256_transform(context, context->block);
            context->block_size = 0u;
        }
    }
}

static void qtk_sha256_final(qtk_sha256_t *context, uint8_t digest[QTK_SHA256_SIZE])
{
    uint64_t total_bits = context->total_bytes * UINT64_C(8);
    size_t index;

    context->block[context->block_size++] = 0x80u;
    if (context->block_size > 56u) {
        memset(context->block + context->block_size, 0, 64u - context->block_size);
        qtk_sha256_transform(context, context->block);
        context->block_size = 0u;
    }
    memset(context->block + context->block_size, 0, 56u - context->block_size);
    qtk_write_be64(context->block + 56u, total_bits);
    qtk_sha256_transform(context, context->block);

    for (index = 0u; index < 8u; ++index) {
        uint32_t value = context->state[index];
        digest[index * 4u] = (uint8_t)(value >> 24);
        digest[index * 4u + 1u] = (uint8_t)(value >> 16);
        digest[index * 4u + 2u] = (uint8_t)(value >> 8);
        digest[index * 4u + 3u] = (uint8_t)value;
    }
}

static int qtk_sha256_matches(
    const uint8_t *data,
    size_t length,
    const uint8_t expected[QTK_SHA256_SIZE])
{
    qtk_sha256_t context;
    uint8_t actual[QTK_SHA256_SIZE];

    qtk_sha256_init(&context);
    qtk_sha256_update(&context, data, length);
    qtk_sha256_final(&context, actual);
    return memcmp(actual, expected, sizeof(actual)) == 0;
}

static int qtk_all_zero(const uint8_t *data, size_t length)
{
    size_t index;

    for (index = 0u; index < length; ++index) {
        if (data[index] != 0u) return 0;
    }
    return 1;
}

static int qtk_name_valid(const uint8_t *name)
{
    size_t index;
    int found_nul = 0;

    for (index = 0u; index < 16u; ++index) {
        uint8_t value = name[index];
        if (found_nul != 0) {
            if (value != 0u) return 0;
        } else if (value == 0u) {
            if (index == 0u) return 0;
            found_nul = 1;
        } else if (value < 0x20u || value > 0x7eu) {
            return 0;
        }
    }
    return found_nul;
}

static int qtk_name_equal(const uint8_t *name, const char *expected)
{
    size_t index = 0u;

    while (index < 16u && expected[index] != '\0') {
        if (name[index] != (uint8_t)expected[index]) return 0;
        ++index;
    }
    return index < 16u && name[index] == 0u && expected[index] == '\0';
}

static const uint8_t *qtk_descriptor(
    const uint8_t *data,
    uint32_t descriptor_index)
{
    return data + QTK_PREFIX_SIZE +
           (size_t)descriptor_index * QTK_SECTION_RECORD_SIZE;
}

static qtk_status_t qtk_read_section(
    const uint8_t *data,
    size_t data_size,
    size_t header_size,
    const uint8_t *descriptor,
    qtk_section_view_t *section)
{
    uint64_t offset64 = qtk_read_le64(descriptor + 16u);
    uint64_t size64 = qtk_read_le64(descriptor + 24u);
    uint32_t count = qtk_read_le32(descriptor + 32u);
    uint32_t record_size = qtk_read_le32(descriptor + 36u);
    size_t offset;
    size_t size;

    if (offset64 > (uint64_t)SIZE_MAX || size64 > (uint64_t)SIZE_MAX) {
        return QTK_ERR_RANGE;
    }
    offset = (size_t)offset64;
    size = (size_t)size64;
    if (size == 0u || offset < header_size ||
        (offset & (QTK_ALIGNMENT - 1u)) != 0u ||
        offset > data_size || size > data_size - offset) {
        return QTK_ERR_RANGE;
    }
    if (record_size != 0u &&
        ((size_t)count > SIZE_MAX / (size_t)record_size ||
         (size_t)count * (size_t)record_size != size)) {
        return QTK_ERR_FORMAT;
    }
    if (!qtk_sha256_matches(data + offset, size, descriptor + 40u)) {
        return QTK_ERR_HASH;
    }

    section->data = data + offset;
    section->size = size;
    section->count = count;
    section->record_size = record_size;
    return QTK_OK;
}

static qtk_status_t qtk_validate_offsets_and_flags(const qtk_asset_t *asset)
{
    uint32_t token_id;
    uint32_t previous = 0u;

    for (token_id = 0u; token_id <= asset->token_count; ++token_id) {
        uint32_t offset = qtk_read_le32(
            asset->token_offsets.data + (size_t)token_id * 4u);
        if (offset < previous || (size_t)offset > asset->token_bytes.size) {
            return QTK_ERR_FORMAT;
        }
        previous = offset;
    }
    if ((size_t)previous != asset->token_bytes.size) return QTK_ERR_FORMAT;

    for (token_id = 0u; token_id < asset->token_count; ++token_id) {
        uint8_t flags = asset->token_flags.data[token_id];
        if ((flags & (uint8_t)~(QTK_TOKEN_FLAG_VALID |
                                QTK_TOKEN_FLAG_ADDED |
                                QTK_TOKEN_FLAG_SPECIAL)) != 0u ||
            (flags & QTK_TOKEN_FLAG_VALID) == 0u) {
            return QTK_ERR_FORMAT;
        }
        if (token_id < asset->base_vocab_count) {
            if (flags != QTK_TOKEN_FLAG_VALID) return QTK_ERR_FORMAT;
        } else if ((flags & QTK_TOKEN_FLAG_ADDED) == 0u) {
            return QTK_ERR_FORMAT;
        }
    }
    for (token_id = 0u; token_id < asset->added_token_count; ++token_id) {
        uint8_t properties = asset->added_props.data[token_id];
        uint8_t flags = asset->token_flags.data[asset->base_vocab_count + token_id];
        if ((properties & (uint8_t)~(QTK_ADDED_PROP_SPECIAL |
                                     QTK_ADDED_PROP_SINGLE_WORD |
                                     QTK_ADDED_PROP_LSTRIP |
                                     QTK_ADDED_PROP_RSTRIP |
                                     QTK_ADDED_PROP_NORMALIZED)) != 0u ||
            (properties & (QTK_ADDED_PROP_SINGLE_WORD |
                           QTK_ADDED_PROP_LSTRIP |
                           QTK_ADDED_PROP_RSTRIP |
                           QTK_ADDED_PROP_NORMALIZED)) != 0u ||
            (((properties & QTK_ADDED_PROP_SPECIAL) != 0u) !=
             ((flags & QTK_TOKEN_FLAG_SPECIAL) != 0u))) {
            return QTK_ERR_FORMAT;
        }
    }
    return QTK_OK;
}

static int qtk_scalar_valid(uint32_t codepoint)
{
    return codepoint <= UINT32_C(0x10ffff) &&
           !(codepoint >= UINT32_C(0xd800) && codepoint <= UINT32_C(0xdfff));
}

static qtk_status_t qtk_validate_range_section(const qtk_section_view_t *section)
{
    uint32_t index;
    uint32_t previous_end = 0u;

    for (index = 0u; index < section->count; ++index) {
        const uint8_t *record = section->data + (size_t)index * 8u;
        uint32_t first = qtk_read_le32(record);
        uint32_t last = qtk_read_le32(record + 4u);
        if (!qtk_scalar_valid(first) || !qtk_scalar_valid(last) || first > last ||
            (index != 0u && first <= previous_end)) {
            return QTK_ERR_FORMAT;
        }
        previous_end = last;
    }
    return QTK_OK;
}

static qtk_status_t qtk_validate_unicode_tables(const qtk_asset_t *asset)
{
    uint32_t index;
    uint32_t previous_codepoint = 0u;
    uint64_t previous_key = 0u;
    qtk_status_t status;

    for (index = 0u; index < asset->unicode_nfc_ccc.count; ++index) {
        const uint8_t *record = asset->unicode_nfc_ccc.data + (size_t)index * 8u;
        uint32_t codepoint = qtk_read_le32(record);
        if (!qtk_scalar_valid(codepoint) || (index != 0u && codepoint <= previous_codepoint) ||
            record[4] == 0u || !qtk_all_zero(record + 5u, 3u)) {
            return QTK_ERR_FORMAT;
        }
        previous_codepoint = codepoint;
    }

    previous_codepoint = 0u;
    for (index = 0u; index < asset->unicode_nfc_decomp.count; ++index) {
        const uint8_t *record = asset->unicode_nfc_decomp.data + (size_t)index * 12u;
        uint32_t codepoint = qtk_read_le32(record);
        uint32_t offset = qtk_read_le32(record + 4u);
        uint32_t length = qtk_read_le32(record + 8u);
        if (!qtk_scalar_valid(codepoint) || (index != 0u && codepoint <= previous_codepoint) ||
            length == 0u || offset > asset->unicode_nfc_sequence.count ||
            length > asset->unicode_nfc_sequence.count - offset) {
            return QTK_ERR_FORMAT;
        }
        previous_codepoint = codepoint;
    }
    for (index = 0u; index < asset->unicode_nfc_sequence.count; ++index) {
        if (!qtk_scalar_valid(qtk_read_le32(
                asset->unicode_nfc_sequence.data + (size_t)index * 4u))) {
            return QTK_ERR_FORMAT;
        }
    }

    for (index = 0u; index < asset->unicode_nfc_compose.count; ++index) {
        const uint8_t *record = asset->unicode_nfc_compose.data + (size_t)index * 12u;
        uint64_t key = qtk_read_le64(record);
        uint32_t left = (uint32_t)(key >> 32);
        uint32_t right = (uint32_t)key;
        uint32_t result = qtk_read_le32(record + 8u);
        if ((index != 0u && key <= previous_key) || !qtk_scalar_valid(left) ||
            !qtk_scalar_valid(right) || !qtk_scalar_valid(result)) {
            return QTK_ERR_FORMAT;
        }
        previous_key = key;
    }

    status = qtk_validate_range_section(&asset->unicode_letter_ranges);
    if (status == QTK_OK) status = qtk_validate_range_section(&asset->unicode_number_ranges);
    if (status == QTK_OK) status = qtk_validate_range_section(&asset->unicode_space_ranges);
    if (status != QTK_OK) return status;

    previous_codepoint = 0u;
    for (index = 0u; index < asset->unicode_fold_ascii.count; ++index) {
        const uint8_t *record = asset->unicode_fold_ascii.data + (size_t)index * 8u;
        uint32_t codepoint = qtk_read_le32(record);
        uint32_t folded = qtk_read_le32(record + 4u);
        if (!qtk_scalar_valid(codepoint) || (index != 0u && codepoint <= previous_codepoint) ||
            (folded != (uint32_t)'s' && folded != (uint32_t)'t' &&
             folded != (uint32_t)'r' && folded != (uint32_t)'e' &&
             folded != (uint32_t)'v' && folded != (uint32_t)'m' &&
             folded != (uint32_t)'l' && folded != (uint32_t)'d')) {
            return QTK_ERR_FORMAT;
        }
        previous_codepoint = codepoint;
    }
    return QTK_OK;
}

static qtk_status_t qtk_validate_byte_and_merge_tables(const qtk_asset_t *asset)
{
    uint32_t index;
    uint64_t previous_key = 0u;

    for (index = 0u; index < 256u; ++index) {
        uint32_t token_id = qtk_read_le32(
            asset->byte_to_token.data + (size_t)index * 4u);
        if (token_id >= asset->base_vocab_count) return QTK_ERR_FORMAT;
    }

    for (index = 0u; index < asset->merge_count; ++index) {
        const uint8_t *record = asset->merge_lookup.data + (size_t)index * 12u;
        uint64_t key = qtk_read_le64(record);
        uint32_t rank = qtk_read_le32(record + 8u);
        uint32_t left = (uint32_t)(key >> 32);
        uint32_t right = (uint32_t)key;

        if ((index != 0u && key <= previous_key) ||
            left >= asset->base_vocab_count ||
            right >= asset->base_vocab_count ||
            rank >= asset->merge_count ||
            rank > UINT32_MAX - 256u ||
            rank + 256u >= asset->base_vocab_count) {
            return QTK_ERR_FORMAT;
        }
        previous_key = key;
    }
    return QTK_OK;
}

const char *qtk_status_name(qtk_status_t status)
{
    switch (status) {
    case QTK_OK: return "ok";
    case QTK_ERR_NULL: return "null";
    case QTK_ERR_TOO_SMALL: return "too_small";
    case QTK_ERR_MAGIC: return "magic";
    case QTK_ERR_VERSION: return "version";
    case QTK_ERR_ENDIAN: return "endian";
    case QTK_ERR_FORMAT: return "format";
    case QTK_ERR_RANGE: return "range";
    case QTK_ERR_HASH: return "hash";
    case QTK_ERR_SECTION: return "section";
    case QTK_ERR_TOKEN_ID: return "token_id";
    case QTK_ERR_WORKSPACE: return "workspace";
    case QTK_ERR_EMIT: return "emit";
    case QTK_ERR_NOT_FOUND: return "not_found";
    case QTK_ERR_UTF8: return "utf8";
    case QTK_ERR_UNICODE_WORKSPACE: return "unicode_workspace";
    case QTK_ERR_PIECE_WORKSPACE: return "piece_workspace";
    case QTK_ERR_OUTPUT: return "output";
    case QTK_ERR_MODEL_ONLY_TOKEN: return "model_only_token";
    default: return "unknown";
    }
}

qtk_status_t qtk_asset_init(
    qtk_asset_t *asset,
    const void *data_pointer,
    size_t data_size)
{
    static const uint8_t magic[8] = {'Q', 'T', 'K', 'B', 'P', 'E', '1', 0};
    const uint8_t *data = (const uint8_t *)data_pointer;
    uint32_t section_count;
    uint32_t descriptor_index;
    uint32_t other_index;
    size_t header_size;
    size_t table_size;
    uint64_t file_size64;
    uint64_t payload_size64;
    size_t cursor;
    qtk_status_t status;

    if (asset == NULL) return QTK_ERR_NULL;
    memset(asset, 0, sizeof(*asset));
    if (data == NULL) return QTK_ERR_NULL;
    if (data_size < QTK_PREFIX_SIZE) return QTK_ERR_TOO_SMALL;
    if (memcmp(data, magic, sizeof(magic)) != 0) return QTK_ERR_MAGIC;
    if (qtk_read_le32(data + 8u) != QTK_FORMAT_VERSION) return QTK_ERR_VERSION;
    if (qtk_read_le32(data + 16u) != QTK_ENDIAN_TAG) return QTK_ERR_ENDIAN;
    if (qtk_read_le32(data + 20u) != QTK_REQUIRED_FLAGS) return QTK_ERR_FORMAT;

    header_size = (size_t)qtk_read_le32(data + 12u);
    section_count = qtk_read_le32(data + 24u);
    if (header_size < QTK_PREFIX_SIZE || header_size > data_size ||
        (header_size & (QTK_ALIGNMENT - 1u)) != 0u ||
        section_count < 15u ||
        (uint64_t)section_count * UINT64_C(72) >
            (uint64_t)SIZE_MAX - UINT64_C(128)) {
        return QTK_ERR_FORMAT;
    }
    table_size = (size_t)section_count * QTK_SECTION_RECORD_SIZE;
    if (table_size > header_size - QTK_PREFIX_SIZE) return QTK_ERR_FORMAT;
    if (!qtk_all_zero(data + QTK_PREFIX_USED_SIZE,
                      QTK_PREFIX_SIZE - QTK_PREFIX_USED_SIZE) ||
        !qtk_all_zero(data + QTK_PREFIX_SIZE + table_size,
                      header_size - QTK_PREFIX_SIZE - table_size)) {
        return QTK_ERR_FORMAT;
    }

    file_size64 = qtk_read_le64(data + 64u);
    payload_size64 = qtk_read_le64(data + 72u);
    if (file_size64 != (uint64_t)data_size ||
        payload_size64 != (uint64_t)(data_size - header_size)) {
        return QTK_ERR_RANGE;
    }
    if (!qtk_sha256_matches(data + header_size,
                            data_size - header_size,
                            data + 80u)) {
        return QTK_ERR_HASH;
    }

    asset->data = data;
    asset->data_size = data_size;
    asset->base_vocab_count = qtk_read_le32(data + 28u);
    asset->added_token_count = qtk_read_le32(data + 32u);
    asset->token_count = qtk_read_le32(data + 36u);
    asset->model_vocab_size = qtk_read_le32(data + 40u);
    asset->merge_count = qtk_read_le32(data + 44u);
    asset->eos_token_id = qtk_read_le32(data + 48u);
    asset->pad_token_id = qtk_read_le32(data + 52u);
    asset->unk_token_id = qtk_read_le32(data + 56u);
    asset->normalizer_kind = qtk_read_le32(data + 60u);

    if (asset->base_vocab_count < 256u ||
        asset->added_token_count > UINT32_MAX - asset->base_vocab_count ||
        asset->token_count != asset->base_vocab_count + asset->added_token_count ||
        asset->token_count == UINT32_MAX ||
        asset->model_vocab_size < asset->token_count ||
        asset->merge_count > UINT32_MAX - 256u ||
        asset->base_vocab_count != asset->merge_count + 256u ||
        asset->normalizer_kind != QTK_NORMALIZER_NFC ||
        asset->eos_token_id >= asset->token_count ||
        asset->pad_token_id >= asset->token_count ||
        (asset->unk_token_id != QTK_INVALID_TOKEN_ID &&
         asset->unk_token_id >= asset->token_count)) {
        memset(asset, 0, sizeof(*asset));
        return QTK_ERR_FORMAT;
    }

    for (descriptor_index = 0u; descriptor_index < section_count; ++descriptor_index) {
        const uint8_t *descriptor = qtk_descriptor(data, descriptor_index);
        qtk_section_view_t section;
        qtk_section_view_t *destination = NULL;

        if (!qtk_name_valid(descriptor)) {
            memset(asset, 0, sizeof(*asset));
            return QTK_ERR_SECTION;
        }
        for (other_index = 0u; other_index < descriptor_index; ++other_index) {
            const uint8_t *other = qtk_descriptor(data, other_index);
            if (memcmp(descriptor, other, 16u) == 0) {
                memset(asset, 0, sizeof(*asset));
                return QTK_ERR_SECTION;
            }
        }
        status = qtk_read_section(data, data_size, header_size, descriptor, &section);
        if (status != QTK_OK) {
            memset(asset, 0, sizeof(*asset));
            return status;
        }

        if (qtk_name_equal(descriptor, "BYTE_TO_TOKEN")) {
            destination = &asset->byte_to_token;
        } else if (qtk_name_equal(descriptor, "MERGE_LOOKUP")) {
            destination = &asset->merge_lookup;
        } else if (qtk_name_equal(descriptor, "TOKEN_OFFSETS")) {
            destination = &asset->token_offsets;
        } else if (qtk_name_equal(descriptor, "TOKEN_BYTES")) {
            destination = &asset->token_bytes;
        } else if (qtk_name_equal(descriptor, "TOKEN_FLAGS")) {
            destination = &asset->token_flags;
        } else if (qtk_name_equal(descriptor, "ADDED_PROPS")) {
            destination = &asset->added_props;
        } else if (qtk_name_equal(descriptor, "U_NFC_CCC")) {
            destination = &asset->unicode_nfc_ccc;
        } else if (qtk_name_equal(descriptor, "U_NFC_DECOMP")) {
            destination = &asset->unicode_nfc_decomp;
        } else if (qtk_name_equal(descriptor, "U_NFC_SEQ")) {
            destination = &asset->unicode_nfc_sequence;
        } else if (qtk_name_equal(descriptor, "U_NFC_COMPOSE")) {
            destination = &asset->unicode_nfc_compose;
        } else if (qtk_name_equal(descriptor, "U_LETTER")) {
            destination = &asset->unicode_letter_ranges;
        } else if (qtk_name_equal(descriptor, "U_NUMBER")) {
            destination = &asset->unicode_number_ranges;
        } else if (qtk_name_equal(descriptor, "U_SPACE")) {
            destination = &asset->unicode_space_ranges;
        } else if (qtk_name_equal(descriptor, "U_FOLD_ASCII")) {
            destination = &asset->unicode_fold_ascii;
        } else if (qtk_name_equal(descriptor, "METADATA_JSON")) {
            destination = &asset->metadata_json;
        }
        if (destination != NULL) *destination = section;
    }

    if (asset->byte_to_token.data == NULL ||
        asset->merge_lookup.data == NULL ||
        asset->token_offsets.data == NULL ||
        asset->token_bytes.data == NULL ||
        asset->token_flags.data == NULL ||
        asset->added_props.data == NULL ||
        asset->unicode_nfc_ccc.data == NULL ||
        asset->unicode_nfc_decomp.data == NULL ||
        asset->unicode_nfc_sequence.data == NULL ||
        asset->unicode_nfc_compose.data == NULL ||
        asset->unicode_letter_ranges.data == NULL ||
        asset->unicode_number_ranges.data == NULL ||
        asset->unicode_space_ranges.data == NULL ||
        asset->unicode_fold_ascii.data == NULL ||
        asset->metadata_json.data == NULL) {
        memset(asset, 0, sizeof(*asset));
        return QTK_ERR_SECTION;
    }

    for (descriptor_index = 0u; descriptor_index < section_count; ++descriptor_index) {
        const uint8_t *left_descriptor = qtk_descriptor(data, descriptor_index);
        size_t left_offset = (size_t)qtk_read_le64(left_descriptor + 16u);
        size_t left_size = (size_t)qtk_read_le64(left_descriptor + 24u);

        for (other_index = descriptor_index + 1u;
             other_index < section_count;
             ++other_index) {
            const uint8_t *right_descriptor = qtk_descriptor(data, other_index);
            size_t right_offset = (size_t)qtk_read_le64(right_descriptor + 16u);
            size_t right_size = (size_t)qtk_read_le64(right_descriptor + 24u);
            if (left_offset < right_offset + right_size &&
                right_offset < left_offset + left_size) {
                memset(asset, 0, sizeof(*asset));
                return QTK_ERR_SECTION;
            }
        }
    }

    cursor = header_size;
    for (descriptor_index = 0u; descriptor_index < section_count; ++descriptor_index) {
        size_t next_offset = data_size;
        size_t next_size = 0u;
        int found = 0;

        for (other_index = 0u; other_index < section_count; ++other_index) {
            const uint8_t *descriptor = qtk_descriptor(data, other_index);
            size_t offset = (size_t)qtk_read_le64(descriptor + 16u);
            if (offset >= cursor && (found == 0 || offset < next_offset)) {
                next_offset = offset;
                next_size = (size_t)qtk_read_le64(descriptor + 24u);
                found = 1;
            }
        }
        if (found == 0 || !qtk_all_zero(data + cursor, next_offset - cursor)) {
            memset(asset, 0, sizeof(*asset));
            return QTK_ERR_SECTION;
        }
        cursor = next_offset + next_size;
    }
    if (cursor != data_size) {
        memset(asset, 0, sizeof(*asset));
        return QTK_ERR_SECTION;
    }

    if (asset->byte_to_token.count != 256u ||
        asset->byte_to_token.record_size != 4u ||
        asset->byte_to_token.size != 256u * 4u ||
        asset->merge_lookup.count != asset->merge_count ||
        asset->merge_lookup.record_size != 12u ||
        asset->token_offsets.count != asset->token_count + 1u ||
        asset->token_offsets.record_size != 4u ||
        asset->token_bytes.count != asset->token_count ||
        asset->token_bytes.record_size != 0u ||
        asset->token_flags.count != asset->token_count ||
        asset->token_flags.record_size != 1u ||
        asset->added_props.count != asset->added_token_count ||
        asset->added_props.record_size != 1u ||
        asset->unicode_nfc_ccc.record_size != 8u ||
        asset->unicode_nfc_decomp.record_size != 12u ||
        asset->unicode_nfc_sequence.record_size != 4u ||
        asset->unicode_nfc_compose.record_size != 12u ||
        asset->unicode_letter_ranges.record_size != 8u ||
        asset->unicode_number_ranges.record_size != 8u ||
        asset->unicode_space_ranges.record_size != 8u ||
        asset->unicode_fold_ascii.record_size != 8u ||
        asset->metadata_json.count != 1u ||
        asset->metadata_json.record_size != 0u) {
        memset(asset, 0, sizeof(*asset));
        return QTK_ERR_FORMAT;
    }

    status = qtk_validate_offsets_and_flags(asset);
    if (status == QTK_OK) status = qtk_validate_byte_and_merge_tables(asset);
    if (status == QTK_OK) status = qtk_validate_unicode_tables(asset);
    if (status != QTK_OK) memset(asset, 0, sizeof(*asset));
    return status;
}

qtk_status_t qtk_token_slice(
    const qtk_asset_t *asset,
    uint32_t token_id,
    qtk_token_slice_t *slice)
{
    uint32_t begin;
    uint32_t end;

    if (slice == NULL) return QTK_ERR_NULL;
    slice->bytes = NULL;
    slice->length = 0u;
    slice->flags = 0u;
    if (asset == NULL || asset->data == NULL) return QTK_ERR_NULL;
    if (token_id >= asset->token_count) {
        if (token_id < asset->model_vocab_size) return QTK_ERR_MODEL_ONLY_TOKEN;
        return QTK_ERR_TOKEN_ID;
    }

    begin = qtk_read_le32(asset->token_offsets.data + (size_t)token_id * 4u);
    end = qtk_read_le32(asset->token_offsets.data + ((size_t)token_id + 1u) * 4u);
    slice->bytes = asset->token_bytes.data + begin;
    slice->length = (size_t)(end - begin);
    slice->flags = asset->token_flags.data[token_id];
    return QTK_OK;
}

qtk_status_t qtk_detokenize_ids(
    const qtk_asset_t *asset,
    const uint32_t *token_ids,
    size_t token_id_count,
    int skip_special_tokens,
    qtk_emit_bytes_fn emit,
    void *emit_context)
{
    size_t index;

    if (asset == NULL || asset->data == NULL) return QTK_ERR_NULL;
    if (token_id_count != 0u && (token_ids == NULL || emit == NULL)) {
        return QTK_ERR_NULL;
    }
    for (index = 0u; index < token_id_count; ++index) {
        qtk_token_slice_t slice;
        qtk_status_t status = qtk_token_slice(asset, token_ids[index], &slice);
        if (status != QTK_OK) return status;
        if (skip_special_tokens != 0 &&
            (slice.flags & QTK_TOKEN_FLAG_SPECIAL) != 0u) {
            continue;
        }
        if (slice.length != 0u &&
            emit(emit_context, slice.bytes, slice.length) != 0) {
            return QTK_ERR_EMIT;
        }
    }
    return QTK_OK;
}

qtk_status_t qtk_find_merge_rank(
    const qtk_asset_t *asset,
    uint32_t left_token_id,
    uint32_t right_token_id,
    uint32_t *rank)
{
    uint64_t wanted;
    size_t low;
    size_t high;

    if (rank == NULL) return QTK_ERR_NULL;
    *rank = 0u;
    if (asset == NULL || asset->data == NULL) return QTK_ERR_NULL;
    wanted = ((uint64_t)left_token_id << 32) | (uint64_t)right_token_id;
    low = 0u;
    high = (size_t)asset->merge_lookup.count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        const uint8_t *record = asset->merge_lookup.data + middle * 12u;
        uint64_t key = qtk_read_le64(record);
        if (key < wanted) {
            low = middle + 1u;
        } else if (key > wanted) {
            high = middle;
        } else {
            *rank = qtk_read_le32(record + 8u);
            return QTK_OK;
        }
    }
    return QTK_ERR_NOT_FOUND;
}

qtk_status_t qtk_encode_raw_piece(
    const qtk_asset_t *asset,
    const uint8_t *raw_bytes,
    size_t raw_length,
    uint32_t *workspace,
    size_t workspace_capacity,
    size_t *output_count)
{
    size_t count;
    size_t index;

    if (output_count == NULL) return QTK_ERR_NULL;
    *output_count = 0u;
    if (asset == NULL || asset->data == NULL) return QTK_ERR_NULL;
    if (raw_length == 0u) return QTK_OK;
    if (raw_bytes == NULL || workspace == NULL) return QTK_ERR_NULL;
    if (workspace_capacity < raw_length) return QTK_ERR_WORKSPACE;

    for (index = 0u; index < raw_length; ++index) {
        workspace[index] = qtk_read_le32(
            asset->byte_to_token.data + (size_t)raw_bytes[index] * 4u);
    }
    count = raw_length;

    while (count > 1u) {
        uint32_t best_rank = 0u;
        uint32_t best_left = 0u;
        uint32_t best_right = 0u;
        int found = 0;
        size_t read_index;
        size_t write_index;

        for (index = 0u; index + 1u < count; ++index) {
            uint32_t rank;
            qtk_status_t status = qtk_find_merge_rank(
                asset, workspace[index], workspace[index + 1u], &rank);
            if (status == QTK_OK && (found == 0 || rank < best_rank)) {
                best_rank = rank;
                best_left = workspace[index];
                best_right = workspace[index + 1u];
                found = 1;
            } else if (status != QTK_OK && status != QTK_ERR_NOT_FOUND) {
                return status;
            }
        }
        if (found == 0) break;

        read_index = 0u;
        write_index = 0u;
        while (read_index < count) {
            if (read_index + 1u < count &&
                workspace[read_index] == best_left &&
                workspace[read_index + 1u] == best_right) {
                workspace[write_index++] = best_rank + 256u;
                read_index += 2u;
            } else {
                workspace[write_index++] = workspace[read_index++];
            }
        }
        count = write_index;
    }

    *output_count = count;
    return QTK_OK;
}
