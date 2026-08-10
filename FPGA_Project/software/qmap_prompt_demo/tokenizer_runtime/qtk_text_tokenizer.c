#include "qtk_text_tokenizer.h"

#include <limits.h>
#include <string.h>

#define QTK_HANGUL_SBASE UINT32_C(0xac00)
#define QTK_HANGUL_LBASE UINT32_C(0x1100)
#define QTK_HANGUL_VBASE UINT32_C(0x1161)
#define QTK_HANGUL_TBASE UINT32_C(0x11a7)
#define QTK_HANGUL_LCOUNT 19u
#define QTK_HANGUL_VCOUNT 21u
#define QTK_HANGUL_TCOUNT 28u
#define QTK_HANGUL_NCOUNT (QTK_HANGUL_VCOUNT * QTK_HANGUL_TCOUNT)
#define QTK_HANGUL_SCOUNT (QTK_HANGUL_LCOUNT * QTK_HANGUL_NCOUNT)

typedef struct {
    size_t offset;
    size_t length;
    uint32_t added_token_id;
    int is_added;
} qtk_input_segment_t;

static uint32_t qtk_text_read_le32(const uint8_t *data)
{
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
}

static uint64_t qtk_text_read_le64(const uint8_t *data)
{
    return ((uint64_t)qtk_text_read_le32(data)) |
           ((uint64_t)qtk_text_read_le32(data + 4) << 32);
}

static int qtk_text_scalar_valid(uint32_t codepoint)
{
    return codepoint <= UINT32_C(0x10ffff) &&
           !(codepoint >= UINT32_C(0xd800) && codepoint <= UINT32_C(0xdfff));
}

static qtk_status_t qtk_utf8_decode_one(
    const uint8_t *data,
    size_t length,
    size_t offset,
    uint32_t *codepoint,
    size_t *next_offset)
{
    uint8_t first;

    if (offset >= length) return QTK_ERR_UTF8;
    first = data[offset];
    if (first <= 0x7fu) {
        *codepoint = first;
        *next_offset = offset + 1u;
        return QTK_OK;
    }
    if (first >= 0xc2u && first <= 0xdfu) {
        uint8_t second;
        if (length - offset < 2u) return QTK_ERR_UTF8;
        second = data[offset + 1u];
        if ((second & 0xc0u) != 0x80u) return QTK_ERR_UTF8;
        *codepoint = ((uint32_t)(first & 0x1fu) << 6) |
                     (uint32_t)(second & 0x3fu);
        *next_offset = offset + 2u;
        return QTK_OK;
    }
    if (first >= 0xe0u && first <= 0xefu) {
        uint8_t second;
        uint8_t third;
        if (length - offset < 3u) return QTK_ERR_UTF8;
        second = data[offset + 1u];
        third = data[offset + 2u];
        if ((third & 0xc0u) != 0x80u ||
            (first == 0xe0u && (second < 0xa0u || second > 0xbfu)) ||
            (first == 0xedu && (second < 0x80u || second > 0x9fu)) ||
            (first != 0xe0u && first != 0xedu && (second & 0xc0u) != 0x80u)) {
            return QTK_ERR_UTF8;
        }
        *codepoint = ((uint32_t)(first & 0x0fu) << 12) |
                     ((uint32_t)(second & 0x3fu) << 6) |
                     (uint32_t)(third & 0x3fu);
        *next_offset = offset + 3u;
        return QTK_OK;
    }
    if (first >= 0xf0u && first <= 0xf4u) {
        uint8_t second;
        uint8_t third;
        uint8_t fourth;
        if (length - offset < 4u) return QTK_ERR_UTF8;
        second = data[offset + 1u];
        third = data[offset + 2u];
        fourth = data[offset + 3u];
        if ((third & 0xc0u) != 0x80u || (fourth & 0xc0u) != 0x80u ||
            (first == 0xf0u && (second < 0x90u || second > 0xbfu)) ||
            (first == 0xf4u && (second < 0x80u || second > 0x8fu)) ||
            (first != 0xf0u && first != 0xf4u && (second & 0xc0u) != 0x80u)) {
            return QTK_ERR_UTF8;
        }
        *codepoint = ((uint32_t)(first & 0x07u) << 18) |
                     ((uint32_t)(second & 0x3fu) << 12) |
                     ((uint32_t)(third & 0x3fu) << 6) |
                     (uint32_t)(fourth & 0x3fu);
        *next_offset = offset + 4u;
        return QTK_OK;
    }
    return QTK_ERR_UTF8;
}

static qtk_status_t qtk_utf8_validate(const uint8_t *data, size_t length)
{
    size_t offset = 0u;

    while (offset < length) {
        uint32_t codepoint;
        size_t next_offset;
        qtk_status_t status = qtk_utf8_decode_one(
            data, length, offset, &codepoint, &next_offset);
        if (status != QTK_OK || !qtk_text_scalar_valid(codepoint)) return QTK_ERR_UTF8;
        offset = next_offset;
    }
    return QTK_OK;
}

static int qtk_codepoint_in_ranges(
    const qtk_section_view_t *section,
    uint32_t codepoint)
{
    size_t low = 0u;
    size_t high = (size_t)section->count;

    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        const uint8_t *record = section->data + middle * 8u;
        uint32_t first = qtk_text_read_le32(record);
        uint32_t last = qtk_text_read_le32(record + 4u);
        if (codepoint < first) {
            high = middle;
        } else if (codepoint > last) {
            low = middle + 1u;
        } else {
            return 1;
        }
    }
    return 0;
}

static uint8_t qtk_nfc_ccc(const qtk_asset_t *asset, uint32_t codepoint)
{
    size_t low = 0u;
    size_t high = (size_t)asset->unicode_nfc_ccc.count;

    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        const uint8_t *record = asset->unicode_nfc_ccc.data + middle * 8u;
        uint32_t key = qtk_text_read_le32(record);
        if (key < codepoint) {
            low = middle + 1u;
        } else if (key > codepoint) {
            high = middle;
        } else {
            return record[4];
        }
    }
    return 0u;
}

static int qtk_nfc_decomposition(
    const qtk_asset_t *asset,
    uint32_t codepoint,
    uint32_t *sequence_offset,
    uint32_t *sequence_length)
{
    size_t low = 0u;
    size_t high = (size_t)asset->unicode_nfc_decomp.count;

    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        const uint8_t *record = asset->unicode_nfc_decomp.data + middle * 12u;
        uint32_t key = qtk_text_read_le32(record);
        if (key < codepoint) {
            low = middle + 1u;
        } else if (key > codepoint) {
            high = middle;
        } else {
            *sequence_offset = qtk_text_read_le32(record + 4u);
            *sequence_length = qtk_text_read_le32(record + 8u);
            return 1;
        }
    }
    return 0;
}

static size_t qtk_hangul_decomposition(
    uint32_t codepoint,
    uint32_t output[3])
{
    uint32_t s_index;
    uint32_t l_index;
    uint32_t v_index;
    uint32_t t_index;

    if (codepoint < QTK_HANGUL_SBASE ||
        codepoint >= QTK_HANGUL_SBASE + QTK_HANGUL_SCOUNT) {
        return 0u;
    }
    s_index = codepoint - QTK_HANGUL_SBASE;
    l_index = s_index / QTK_HANGUL_NCOUNT;
    v_index = (s_index % QTK_HANGUL_NCOUNT) / QTK_HANGUL_TCOUNT;
    t_index = s_index % QTK_HANGUL_TCOUNT;
    output[0] = QTK_HANGUL_LBASE + l_index;
    output[1] = QTK_HANGUL_VBASE + v_index;
    if (t_index != 0u) {
        output[2] = QTK_HANGUL_TBASE + t_index;
        return 3u;
    }
    return 2u;
}

static qtk_status_t qtk_nfc_required(
    const qtk_asset_t *asset,
    const uint8_t *data,
    size_t length,
    size_t *required)
{
    size_t offset = 0u;
    size_t count = 0u;

    while (offset < length) {
        uint32_t codepoint;
        size_t next_offset;
        uint32_t sequence_offset;
        uint32_t sequence_length;
        uint32_t hangul[3];
        size_t add_count;
        qtk_status_t status = qtk_utf8_decode_one(
            data, length, offset, &codepoint, &next_offset);
        if (status != QTK_OK) return status;
        add_count = qtk_hangul_decomposition(codepoint, hangul);
        if (add_count == 0u && qtk_nfc_decomposition(
                asset, codepoint, &sequence_offset, &sequence_length)) {
            add_count = (size_t)sequence_length;
        }
        if (add_count == 0u) add_count = 1u;
        if (add_count > SIZE_MAX - count) return QTK_ERR_RANGE;
        count += add_count;
        offset = next_offset;
    }
    *required = count;
    return QTK_OK;
}

static void qtk_nfc_append_ordered(
    const qtk_asset_t *asset,
    uint32_t *workspace,
    size_t *count,
    uint32_t codepoint)
{
    uint8_t ccc = qtk_nfc_ccc(asset, codepoint);
    size_t insert = *count;

    if (ccc != 0u) {
        while (insert != 0u) {
            uint8_t previous_ccc = qtk_nfc_ccc(asset, workspace[insert - 1u]);
            if (previous_ccc == 0u || previous_ccc <= ccc) break;
            workspace[insert] = workspace[insert - 1u];
            --insert;
        }
    }
    workspace[insert] = codepoint;
    ++(*count);
}

static int qtk_nfc_compose_pair(
    const qtk_asset_t *asset,
    uint32_t left,
    uint32_t right,
    uint32_t *result)
{
    uint32_t l_index;
    uint32_t v_index;
    uint32_t s_index;
    uint32_t t_index;
    uint64_t wanted;
    size_t low;
    size_t high;

    l_index = left - QTK_HANGUL_LBASE;
    if (l_index < QTK_HANGUL_LCOUNT) {
        v_index = right - QTK_HANGUL_VBASE;
        if (v_index < QTK_HANGUL_VCOUNT) {
            *result = QTK_HANGUL_SBASE +
                      (l_index * QTK_HANGUL_VCOUNT + v_index) * QTK_HANGUL_TCOUNT;
            return 1;
        }
    }
    s_index = left - QTK_HANGUL_SBASE;
    if (s_index < QTK_HANGUL_SCOUNT && s_index % QTK_HANGUL_TCOUNT == 0u) {
        t_index = right - QTK_HANGUL_TBASE;
        if (t_index > 0u && t_index < QTK_HANGUL_TCOUNT) {
            *result = left + t_index;
            return 1;
        }
    }

    wanted = ((uint64_t)left << 32) | (uint64_t)right;
    low = 0u;
    high = (size_t)asset->unicode_nfc_compose.count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        const uint8_t *record = asset->unicode_nfc_compose.data + middle * 12u;
        uint64_t key = qtk_text_read_le64(record);
        if (key < wanted) {
            low = middle + 1u;
        } else if (key > wanted) {
            high = middle;
        } else {
            *result = qtk_text_read_le32(record + 8u);
            return 1;
        }
    }
    return 0;
}

static size_t qtk_utf8_encode(uint32_t codepoint, uint8_t output[4])
{
    if (codepoint <= 0x7fu) {
        output[0] = (uint8_t)codepoint;
        return 1u;
    }
    if (codepoint <= 0x7ffu) {
        output[0] = (uint8_t)(0xc0u | (codepoint >> 6));
        output[1] = (uint8_t)(0x80u | (codepoint & 0x3fu));
        return 2u;
    }
    if (codepoint <= 0xffffu) {
        output[0] = (uint8_t)(0xe0u | (codepoint >> 12));
        output[1] = (uint8_t)(0x80u | ((codepoint >> 6) & 0x3fu));
        output[2] = (uint8_t)(0x80u | (codepoint & 0x3fu));
        return 3u;
    }
    output[0] = (uint8_t)(0xf0u | (codepoint >> 18));
    output[1] = (uint8_t)(0x80u | ((codepoint >> 12) & 0x3fu));
    output[2] = (uint8_t)(0x80u | ((codepoint >> 6) & 0x3fu));
    output[3] = (uint8_t)(0x80u | (codepoint & 0x3fu));
    return 4u;
}

static qtk_status_t qtk_nfc_normalize(
    const qtk_asset_t *asset,
    const uint8_t *data,
    size_t length,
    uint32_t *workspace,
    size_t workspace_capacity,
    const uint8_t **normalized_bytes,
    size_t *normalized_length)
{
    size_t required;
    size_t offset = 0u;
    size_t count = 0u;
    size_t read_index;
    size_t write_index;
    size_t byte_count = 0u;
    qtk_status_t status = qtk_nfc_required(asset, data, length, &required);

    if (status != QTK_OK) return status;
    if (required > workspace_capacity || (required != 0u && workspace == NULL)) {
        return QTK_ERR_UNICODE_WORKSPACE;
    }
    while (offset < length) {
        uint32_t codepoint;
        size_t next_offset;
        uint32_t sequence_offset;
        uint32_t sequence_length;
        uint32_t hangul[3];
        size_t hangul_count;
        size_t index;

        status = qtk_utf8_decode_one(data, length, offset, &codepoint, &next_offset);
        if (status != QTK_OK) return status;
        hangul_count = qtk_hangul_decomposition(codepoint, hangul);
        if (hangul_count != 0u) {
            for (index = 0u; index < hangul_count; ++index) {
                qtk_nfc_append_ordered(asset, workspace, &count, hangul[index]);
            }
        } else if (qtk_nfc_decomposition(
                       asset, codepoint, &sequence_offset, &sequence_length)) {
            for (index = 0u; index < (size_t)sequence_length; ++index) {
                uint32_t decomposed = qtk_text_read_le32(
                    asset->unicode_nfc_sequence.data +
                    ((size_t)sequence_offset + index) * 4u);
                qtk_nfc_append_ordered(asset, workspace, &count, decomposed);
            }
        } else {
            qtk_nfc_append_ordered(asset, workspace, &count, codepoint);
        }
        offset = next_offset;
    }

    write_index = 1u;
    if (count != 0u) {
        size_t starter_index = 0u;
        uint32_t starter = workspace[0];
        uint8_t last_ccc = qtk_nfc_ccc(asset, starter);

        for (read_index = 1u; read_index < count; ++read_index) {
            uint32_t codepoint = workspace[read_index];
            uint8_t ccc = qtk_nfc_ccc(asset, codepoint);
            uint32_t composite;
            if ((last_ccc == 0u || last_ccc < ccc) &&
                qtk_nfc_compose_pair(asset, starter, codepoint, &composite)) {
                workspace[starter_index] = composite;
                starter = composite;
                continue;
            }
            if (ccc == 0u) {
                starter_index = write_index;
                starter = codepoint;
            }
            last_ccc = ccc;
            workspace[write_index++] = codepoint;
        }
    } else {
        write_index = 0u;
    }

    for (read_index = 0u; read_index < write_index; ++read_index) {
        uint8_t encoded[4];
        size_t encoded_length = qtk_utf8_encode(workspace[read_index], encoded);
        memcpy((uint8_t *)workspace + byte_count, encoded, encoded_length);
        byte_count += encoded_length;
    }
    *normalized_bytes = (const uint8_t *)workspace;
    *normalized_length = byte_count;
    return QTK_OK;
}

static int qtk_fold_matches(
    const qtk_asset_t *asset,
    uint32_t codepoint,
    uint32_t ascii_lowercase)
{
    size_t low = 0u;
    size_t high = (size_t)asset->unicode_fold_ascii.count;

    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        const uint8_t *record = asset->unicode_fold_ascii.data + middle * 8u;
        uint32_t key = qtk_text_read_le32(record);
        if (key < codepoint) {
            low = middle + 1u;
        } else if (key > codepoint) {
            high = middle;
        } else {
            return qtk_text_read_le32(record + 4u) == ascii_lowercase;
        }
    }
    return 0;
}

static size_t qtk_match_contraction(
    const qtk_asset_t *asset,
    const uint8_t *data,
    size_t length,
    size_t offset)
{
    static const char *const suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    size_t suffix_index;

    if (data[offset] != (uint8_t)'\'') return offset;
    for (suffix_index = 0u; suffix_index < sizeof(suffixes) / sizeof(suffixes[0]);
         ++suffix_index) {
        const char *suffix = suffixes[suffix_index];
        size_t current = offset + 1u;
        size_t letter_index = 0u;
        int matches = 1;
        while (suffix[letter_index] != '\0') {
            uint32_t codepoint;
            size_t next;
            if (current >= length ||
                qtk_utf8_decode_one(data, length, current, &codepoint, &next) != QTK_OK ||
                !qtk_fold_matches(asset, codepoint, (uint32_t)(uint8_t)suffix[letter_index])) {
                matches = 0;
                break;
            }
            current = next;
            ++letter_index;
        }
        if (matches != 0) return current;
    }
    return offset;
}

static qtk_status_t qtk_next_piece(
    const qtk_asset_t *asset,
    const uint8_t *data,
    size_t length,
    size_t offset,
    size_t *piece_end)
{
    uint32_t first;
    size_t first_end;
    size_t contraction_end;
    qtk_status_t status = qtk_utf8_decode_one(
        data, length, offset, &first, &first_end);
    if (status != QTK_OK) return status;

    contraction_end = qtk_match_contraction(asset, data, length, offset);
    if (contraction_end != offset) {
        *piece_end = contraction_end;
        return QTK_OK;
    }

    if (qtk_codepoint_in_ranges(&asset->unicode_letter_ranges, first)) {
        size_t current = first_end;
        while (current < length) {
            uint32_t codepoint;
            size_t next;
            status = qtk_utf8_decode_one(data, length, current, &codepoint, &next);
            if (status != QTK_OK) return status;
            if (!qtk_codepoint_in_ranges(&asset->unicode_letter_ranges, codepoint)) break;
            current = next;
        }
        *piece_end = current;
        return QTK_OK;
    }
    if (first != (uint32_t)'\r' && first != (uint32_t)'\n' &&
        !qtk_codepoint_in_ranges(&asset->unicode_number_ranges, first) &&
        first_end < length) {
        uint32_t second;
        size_t current;
        status = qtk_utf8_decode_one(data, length, first_end, &second, &current);
        if (status != QTK_OK) return status;
        if (qtk_codepoint_in_ranges(&asset->unicode_letter_ranges, second)) {
            while (current < length) {
                uint32_t codepoint;
                size_t next;
                status = qtk_utf8_decode_one(data, length, current, &codepoint, &next);
                if (status != QTK_OK) return status;
                if (!qtk_codepoint_in_ranges(&asset->unicode_letter_ranges, codepoint)) break;
                current = next;
            }
            *piece_end = current;
            return QTK_OK;
        }
    }

    if (qtk_codepoint_in_ranges(&asset->unicode_number_ranges, first)) {
        *piece_end = first_end;
        return QTK_OK;
    }

    {
        size_t current = offset;
        uint32_t codepoint = first;
        size_t next = first_end;
        if (first == (uint32_t)' ') {
            current = first_end;
            if (current >= length) goto whitespace_alternatives;
            status = qtk_utf8_decode_one(data, length, current, &codepoint, &next);
            if (status != QTK_OK) return status;
        }
        if (!qtk_codepoint_in_ranges(&asset->unicode_space_ranges, codepoint) &&
            !qtk_codepoint_in_ranges(&asset->unicode_letter_ranges, codepoint) &&
            !qtk_codepoint_in_ranges(&asset->unicode_number_ranges, codepoint)) {
            current = next;
            while (current < length) {
                status = qtk_utf8_decode_one(data, length, current, &codepoint, &next);
                if (status != QTK_OK) return status;
                if (qtk_codepoint_in_ranges(&asset->unicode_space_ranges, codepoint) ||
                    qtk_codepoint_in_ranges(&asset->unicode_letter_ranges, codepoint) ||
                    qtk_codepoint_in_ranges(&asset->unicode_number_ranges, codepoint)) {
                    break;
                }
                current = next;
            }
            while (current < length) {
                status = qtk_utf8_decode_one(data, length, current, &codepoint, &next);
                if (status != QTK_OK) return status;
                if (codepoint != (uint32_t)'\r' && codepoint != (uint32_t)'\n') break;
                current = next;
            }
            *piece_end = current;
            return QTK_OK;
        }
    }

whitespace_alternatives:
    if (qtk_codepoint_in_ranges(&asset->unicode_space_ranges, first)) {
        size_t current = offset;
        size_t last_codepoint_start = offset;
        size_t last_newline_end = offset;
        size_t whitespace_count = 0u;
        while (current < length) {
            uint32_t codepoint;
            size_t next;
            status = qtk_utf8_decode_one(data, length, current, &codepoint, &next);
            if (status != QTK_OK) return status;
            if (!qtk_codepoint_in_ranges(&asset->unicode_space_ranges, codepoint)) break;
            last_codepoint_start = current;
            ++whitespace_count;
            current = next;
            if (codepoint == (uint32_t)'\r' || codepoint == (uint32_t)'\n') {
                last_newline_end = current;
            }
        }
        if (last_newline_end != offset) {
            *piece_end = last_newline_end;
        } else if (current == length || whitespace_count == 1u) {
            *piece_end = current;
        } else {
            *piece_end = last_codepoint_start;
        }
        return QTK_OK;
    }
    return QTK_ERR_FORMAT;
}

static qtk_status_t qtk_added_match_at(
    const qtk_asset_t *asset,
    const uint8_t *input,
    size_t input_length,
    size_t offset,
    uint32_t *token_id,
    size_t *match_length)
{
    uint32_t added_index;
    size_t best_length = 0u;
    uint32_t best_id = 0u;

    for (added_index = 0u; added_index < asset->added_token_count; ++added_index) {
        uint32_t candidate_id = asset->base_vocab_count + added_index;
        qtk_token_slice_t slice;
        qtk_status_t status = qtk_token_slice(asset, candidate_id, &slice);
        if (status != QTK_OK) return status;
        if (slice.length > best_length && slice.length <= input_length - offset &&
            memcmp(input + offset, slice.bytes, slice.length) == 0) {
            best_length = slice.length;
            best_id = candidate_id;
        }
    }
    if (best_length == 0u) return QTK_ERR_NOT_FOUND;
    *token_id = best_id;
    *match_length = best_length;
    return QTK_OK;
}

static qtk_status_t qtk_next_input_segment(
    const qtk_asset_t *asset,
    const uint8_t *input,
    size_t input_length,
    size_t offset,
    qtk_input_segment_t *segment)
{
    uint32_t token_id;
    size_t match_length;
    qtk_status_t status = qtk_added_match_at(
        asset, input, input_length, offset, &token_id, &match_length);

    if (status == QTK_OK) {
        segment->offset = offset;
        segment->length = match_length;
        segment->added_token_id = token_id;
        segment->is_added = 1;
        return QTK_OK;
    }
    if (status != QTK_ERR_NOT_FOUND) return status;

    segment->offset = offset;
    segment->is_added = 0;
    while (offset < input_length) {
        uint32_t codepoint;
        size_t next;
        status = qtk_utf8_decode_one(input, input_length, offset, &codepoint, &next);
        if (status != QTK_OK) return status;
        offset = next;
        if (offset < input_length && qtk_added_match_at(
                asset, input, input_length, offset, &token_id, &match_length) == QTK_OK) {
            break;
        }
    }
    segment->length = offset - segment->offset;
    segment->added_token_id = 0u;
    return QTK_OK;
}

static qtk_status_t qtk_find_unicode_requirement(
    const qtk_asset_t *asset,
    const uint8_t *input,
    size_t input_length,
    size_t *maximum)
{
    size_t offset = 0u;
    *maximum = 0u;
    while (offset < input_length) {
        qtk_input_segment_t segment;
        size_t required;
        qtk_status_t status = qtk_next_input_segment(
            asset, input, input_length, offset, &segment);
        if (status != QTK_OK) return status;
        if (segment.is_added == 0) {
            status = qtk_nfc_required(
                asset, input + segment.offset, segment.length, &required);
            if (status != QTK_OK) return status;
            if (required > *maximum) *maximum = required;
        }
        offset += segment.length;
    }
    return QTK_OK;
}

static qtk_status_t qtk_find_piece_requirement(
    const qtk_asset_t *asset,
    const uint8_t *input,
    size_t input_length,
    uint32_t *unicode_workspace,
    size_t unicode_capacity,
    size_t *maximum)
{
    size_t offset = 0u;
    *maximum = 0u;
    while (offset < input_length) {
        qtk_input_segment_t segment;
        qtk_status_t status = qtk_next_input_segment(
            asset, input, input_length, offset, &segment);
        if (status != QTK_OK) return status;
        if (segment.is_added == 0) {
            const uint8_t *normalized;
            size_t normalized_length;
            size_t piece_start = 0u;
            status = qtk_nfc_normalize(
                asset, input + segment.offset, segment.length,
                unicode_workspace, unicode_capacity,
                &normalized, &normalized_length);
            if (status != QTK_OK) return status;
            while (piece_start < normalized_length) {
                size_t piece_end;
                status = qtk_next_piece(
                    asset, normalized, normalized_length, piece_start, &piece_end);
                if (status != QTK_OK || piece_end <= piece_start) {
                    return status == QTK_OK ? QTK_ERR_FORMAT : status;
                }
                if (piece_end - piece_start > *maximum) {
                    *maximum = piece_end - piece_start;
                }
                piece_start = piece_end;
            }
        }
        offset += segment.length;
    }
    return QTK_OK;
}

static qtk_status_t qtk_process_tokens(
    const qtk_asset_t *asset,
    const uint8_t *input,
    size_t input_length,
    uint32_t *unicode_workspace,
    size_t unicode_capacity,
    uint32_t *piece_workspace,
    size_t piece_capacity,
    uint32_t *output_ids,
    size_t *token_count)
{
    size_t offset = 0u;
    size_t total = 0u;

    while (offset < input_length) {
        qtk_input_segment_t segment;
        qtk_status_t status = qtk_next_input_segment(
            asset, input, input_length, offset, &segment);
        if (status != QTK_OK) return status;
        if (segment.is_added != 0) {
            if (output_ids != NULL) output_ids[total] = segment.added_token_id;
            if (total == SIZE_MAX) return QTK_ERR_RANGE;
            ++total;
        } else {
            const uint8_t *normalized;
            size_t normalized_length;
            size_t piece_start = 0u;
            status = qtk_nfc_normalize(
                asset, input + segment.offset, segment.length,
                unicode_workspace, unicode_capacity,
                &normalized, &normalized_length);
            if (status != QTK_OK) return status;
            while (piece_start < normalized_length) {
                size_t piece_end;
                size_t piece_token_count;
                size_t index;
                status = qtk_next_piece(
                    asset, normalized, normalized_length, piece_start, &piece_end);
                if (status != QTK_OK || piece_end <= piece_start) {
                    return status == QTK_OK ? QTK_ERR_FORMAT : status;
                }
                status = qtk_encode_raw_piece(
                    asset, normalized + piece_start, piece_end - piece_start,
                    piece_workspace, piece_capacity, &piece_token_count);
                if (status != QTK_OK) return status;
                if (piece_token_count > SIZE_MAX - total) return QTK_ERR_RANGE;
                if (output_ids != NULL) {
                    for (index = 0u; index < piece_token_count; ++index) {
                        output_ids[total + index] = piece_workspace[index];
                    }
                }
                total += piece_token_count;
                piece_start = piece_end;
            }
        }
        offset += segment.length;
    }
    *token_count = total;
    return QTK_OK;
}

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
    qtk_tokenizer_requirements_t *requirements)
{
    size_t required_count;
    qtk_status_t status;

    if (output_count == NULL || requirements == NULL) return QTK_ERR_NULL;
    *output_count = 0u;
    memset(requirements, 0, sizeof(*requirements));
    if (asset == NULL || asset->data == NULL) return QTK_ERR_NULL;
    if (input_length != 0u && input_utf8 == NULL) return QTK_ERR_NULL;
    status = qtk_utf8_validate(input_utf8, input_length);
    if (status != QTK_OK) return status;
    if (input_length == 0u) return QTK_OK;

    status = qtk_find_unicode_requirement(
        asset, input_utf8, input_length, &requirements->unicode_codepoints);
    if (status != QTK_OK) return status;
    if (requirements->unicode_codepoints > unicode_capacity ||
        (requirements->unicode_codepoints != 0u && unicode_workspace == NULL)) {
        return QTK_ERR_UNICODE_WORKSPACE;
    }

    status = qtk_find_piece_requirement(
        asset, input_utf8, input_length,
        unicode_workspace, unicode_capacity,
        &requirements->piece_token_ids);
    if (status != QTK_OK) return status;
    if (requirements->piece_token_ids > piece_capacity ||
        (requirements->piece_token_ids != 0u && piece_workspace == NULL)) {
        return QTK_ERR_PIECE_WORKSPACE;
    }

    status = qtk_process_tokens(
        asset, input_utf8, input_length,
        unicode_workspace, unicode_capacity,
        piece_workspace, piece_capacity,
        NULL, &required_count);
    if (status != QTK_OK) return status;
    requirements->output_token_ids = required_count;
    if (required_count > output_capacity || (required_count != 0u && output_ids == NULL)) {
        return QTK_ERR_OUTPUT;
    }

    status = qtk_process_tokens(
        asset, input_utf8, input_length,
        unicode_workspace, unicode_capacity,
        piece_workspace, piece_capacity,
        output_ids, output_count);
    if (status != QTK_OK) {
        *output_count = 0u;
        return status;
    }
    return QTK_OK;
}
