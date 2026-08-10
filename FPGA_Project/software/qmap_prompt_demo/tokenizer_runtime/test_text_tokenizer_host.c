#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qtk_text_tokenizer.h"

#define TEXT_REFERENCE_PREFIX_SIZE 24u

typedef struct {
    const uint8_t *cursor;
    const uint8_t *end;
} text_reader_t;

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t length;
} text_collector_t;

static uint32_t text_read_le32(const uint8_t *data)
{
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
}

static int text_read_file(const char *path, uint8_t **data, size_t *size)
{
    FILE *stream;
    long file_length;
    uint8_t *buffer;

    *data = NULL;
    *size = 0u;
    stream = fopen(path, "rb");
    if (stream == NULL) return 1;
    if (fseek(stream, 0L, SEEK_END) != 0) {
        (void)fclose(stream);
        return 1;
    }
    file_length = ftell(stream);
    if (file_length < 0L || fseek(stream, 0L, SEEK_SET) != 0) {
        (void)fclose(stream);
        return 1;
    }
    buffer = (uint8_t *)malloc(file_length == 0L ? 1u : (size_t)file_length);
    if (buffer == NULL) {
        (void)fclose(stream);
        return 1;
    }
    if (file_length != 0L &&
        fread(buffer, 1u, (size_t)file_length, stream) != (size_t)file_length) {
        free(buffer);
        (void)fclose(stream);
        return 1;
    }
    if (fclose(stream) != 0) {
        free(buffer);
        return 1;
    }
    *data = buffer;
    *size = (size_t)file_length;
    return 0;
}

static const uint8_t *text_take(text_reader_t *reader, size_t length)
{
    const uint8_t *result;
    size_t remaining = (size_t)(reader->end - reader->cursor);
    if (length > remaining) return NULL;
    result = reader->cursor;
    reader->cursor += length;
    return result;
}

static int text_u32(text_reader_t *reader, uint32_t *value)
{
    const uint8_t *bytes = text_take(reader, 4u);
    if (bytes == NULL) return 1;
    *value = text_read_le32(bytes);
    return 0;
}

static int text_collect(void *context, const uint8_t *bytes, size_t length)
{
    text_collector_t *collector = (text_collector_t *)context;
    if (length > collector->capacity - collector->length) return 1;
    memcpy(collector->data + collector->length, bytes, length);
    collector->length += length;
    return 0;
}

static int print_case_failure(
    const uint8_t *name,
    uint32_t name_length,
    const char *message)
{
    int printable_length = name_length > (uint32_t)INT_MAX ? INT_MAX : (int)name_length;
    printf("FAIL: %.*s: %s\n", printable_length, (const char *)name, message);
    return 1;
}

static int test_one_case(
    const qtk_asset_t *asset,
    const uint8_t *name,
    uint32_t name_length,
    const uint8_t *input,
    uint32_t input_length,
    const uint8_t *expected_id_bytes,
    uint32_t expected_count,
    const uint8_t *expected_decoded,
    uint32_t expected_decoded_length,
    const uint8_t *expected_skip,
    uint32_t expected_skip_length,
    uint32_t *unicode_workspace,
    size_t unicode_capacity,
    uint32_t *piece_workspace,
    size_t piece_capacity,
    uint32_t *output_ids,
    size_t output_capacity)
{
    qtk_tokenizer_requirements_t requirements;
    size_t output_count = SIZE_MAX;
    qtk_status_t status;
    size_t index;
    uint8_t *decoded_buffer;
    text_collector_t collector;

    status = qtk_tokenize_utf8(
        asset, input_length == 0u ? NULL : input, (size_t)input_length,
        unicode_workspace, unicode_capacity,
        piece_workspace, piece_capacity,
        output_ids, output_capacity,
        &output_count, &requirements);
    if (status != QTK_OK) {
        char message[160];
        (void)snprintf(
            message, sizeof(message),
            "tokenizer status=%s unicode=%zu piece=%zu output=%zu",
            qtk_status_name(status), requirements.unicode_codepoints,
            requirements.piece_token_ids, requirements.output_token_ids);
        return print_case_failure(name, name_length, message);
    }
    if (output_count != (size_t)expected_count ||
        requirements.output_token_ids != output_count) {
        return print_case_failure(name, name_length, "token count mismatch");
    }
    for (index = 0u; index < output_count; ++index) {
        uint32_t expected = text_read_le32(expected_id_bytes + index * 4u);
        if (output_ids[index] != expected) {
            char message[128];
            (void)snprintf(
                message, sizeof(message),
                "token[%zu] actual=%u expected=%u", index,
                (unsigned)output_ids[index], (unsigned)expected);
            return print_case_failure(name, name_length, message);
        }
    }

    decoded_buffer = (uint8_t *)malloc(
        expected_decoded_length > expected_skip_length ?
        (size_t)expected_decoded_length + 1u : (size_t)expected_skip_length + 1u);
    if (decoded_buffer == NULL) {
        return print_case_failure(name, name_length, "decoded buffer allocation failed");
    }
    collector.data = decoded_buffer;
    collector.capacity = (size_t)expected_decoded_length;
    collector.length = 0u;
    status = qtk_detokenize_ids(
        asset, output_count == 0u ? NULL : output_ids, output_count,
        0, output_count == 0u ? NULL : text_collect, &collector);
    if (status != QTK_OK || collector.length != (size_t)expected_decoded_length ||
        memcmp(decoded_buffer, expected_decoded, collector.length) != 0) {
        free(decoded_buffer);
        return print_case_failure(name, name_length, "decoded UTF-8 bytes mismatch");
    }

    collector.capacity = (size_t)expected_skip_length;
    collector.length = 0u;
    status = qtk_detokenize_ids(
        asset, output_count == 0u ? NULL : output_ids, output_count,
        1, output_count == 0u ? NULL : text_collect, &collector);
    if (status != QTK_OK || collector.length != (size_t)expected_skip_length ||
        memcmp(decoded_buffer, expected_skip, collector.length) != 0) {
        free(decoded_buffer);
        return print_case_failure(name, name_length, "skip-special decoded bytes mismatch");
    }
    free(decoded_buffer);

    if (output_count != 0u) {
        size_t rejected_count = 123u;
        qtk_tokenizer_requirements_t rejected_requirements;
        status = qtk_tokenize_utf8(
            asset, input, (size_t)input_length,
            unicode_workspace, unicode_capacity,
            piece_workspace, piece_capacity,
            output_ids, output_count - 1u,
            &rejected_count, &rejected_requirements);
        if (status != QTK_ERR_OUTPUT || rejected_count != 0u ||
            rejected_requirements.output_token_ids != output_count) {
            return print_case_failure(name, name_length, "explicit output bound failed");
        }
    }
    if (requirements.unicode_codepoints != 0u) {
        size_t rejected_count = 123u;
        qtk_tokenizer_requirements_t rejected_requirements;
        status = qtk_tokenize_utf8(
            asset, input, (size_t)input_length,
            unicode_workspace, requirements.unicode_codepoints - 1u,
            piece_workspace, piece_capacity,
            output_ids, output_capacity,
            &rejected_count, &rejected_requirements);
        if (status != QTK_ERR_UNICODE_WORKSPACE || rejected_count != 0u ||
            rejected_requirements.unicode_codepoints !=
                requirements.unicode_codepoints) {
            return print_case_failure(name, name_length, "Unicode workspace bound failed");
        }
    }
    if (requirements.piece_token_ids != 0u) {
        size_t rejected_count = 123u;
        qtk_tokenizer_requirements_t rejected_requirements;
        status = qtk_tokenize_utf8(
            asset, input, (size_t)input_length,
            unicode_workspace, unicode_capacity,
            piece_workspace, requirements.piece_token_ids - 1u,
            output_ids, output_capacity,
            &rejected_count, &rejected_requirements);
        if (status != QTK_ERR_PIECE_WORKSPACE || rejected_count != 0u ||
            rejected_requirements.piece_token_ids != requirements.piece_token_ids) {
            return print_case_failure(name, name_length, "piece workspace bound failed");
        }
    }
    return 0;
}

static int test_invalid_utf8(
    const qtk_asset_t *asset,
    uint32_t *unicode_workspace,
    uint32_t *piece_workspace,
    uint32_t *output_ids)
{
    static const uint8_t invalid_sequences[][4] = {
        {0x80u, 0u, 0u, 0u},
        {0xc0u, 0x80u, 0u, 0u},
        {0xe0u, 0x80u, 0x80u, 0u},
        {0xedu, 0xa0u, 0x80u, 0u},
        {0xf0u, 0x80u, 0x80u, 0x80u},
        {0xf4u, 0x90u, 0x80u, 0x80u},
        {0xe2u, 0x82u, 0u, 0u}
    };
    static const size_t invalid_lengths[] = {1u, 2u, 3u, 3u, 4u, 4u, 2u};
    size_t index;

    for (index = 0u; index < sizeof(invalid_lengths) / sizeof(invalid_lengths[0]); ++index) {
        size_t output_count = 99u;
        qtk_tokenizer_requirements_t requirements;
        qtk_status_t status = qtk_tokenize_utf8(
            asset, invalid_sequences[index], invalid_lengths[index],
            unicode_workspace, 32u, piece_workspace, 32u,
            output_ids, 32u, &output_count, &requirements);
        if (status != QTK_ERR_UTF8 || output_count != 0u) {
            printf("FAIL: invalid UTF-8 case %zu was accepted\n", index);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const uint8_t reference_magic[8] = {
        'Q', 'T', 'X', 'T', 'R', 'F', '1', 0
    };
    uint8_t *asset_data = NULL;
    size_t asset_size = 0u;
    uint8_t *reference_data = NULL;
    size_t reference_size = 0u;
    qtk_asset_t asset;
    text_reader_t reader;
    uint32_t version;
    uint32_t case_count;
    uint32_t golden_count;
    uint32_t max_input_length;
    size_t workspace_capacity;
    uint32_t *unicode_workspace = NULL;
    uint32_t *piece_workspace = NULL;
    uint32_t *output_ids = NULL;
    uint32_t case_index;
    int result = 1;

    if (argc != 3) {
        printf("usage: %s <asset.qtk> <text-reference.bin>\n", argv[0]);
        return 2;
    }
    if (text_read_file(argv[1], &asset_data, &asset_size) != 0 ||
        text_read_file(argv[2], &reference_data, &reference_size) != 0) {
        printf("FAIL: cannot read asset or text reference\n");
        goto cleanup;
    }
    if (reference_size < TEXT_REFERENCE_PREFIX_SIZE ||
        memcmp(reference_data, reference_magic, sizeof(reference_magic)) != 0) {
        printf("FAIL: bad text reference header\n");
        goto cleanup;
    }
    version = text_read_le32(reference_data + 8u);
    case_count = text_read_le32(reference_data + 12u);
    golden_count = text_read_le32(reference_data + 16u);
    max_input_length = text_read_le32(reference_data + 20u);
    if (version != 1u || case_count < golden_count) {
        printf("FAIL: unsupported text reference version/count\n");
        goto cleanup;
    }
    if (qtk_asset_init(&asset, asset_data, asset_size) != QTK_OK) {
        printf("FAIL: cannot parse Unicode tokenizer asset\n");
        goto cleanup;
    }

    workspace_capacity = (size_t)max_input_length * 4u + 32u;
    unicode_workspace = (uint32_t *)malloc(workspace_capacity * sizeof(uint32_t));
    piece_workspace = (uint32_t *)malloc(workspace_capacity * sizeof(uint32_t));
    output_ids = (uint32_t *)malloc(workspace_capacity * sizeof(uint32_t));
    if (unicode_workspace == NULL || piece_workspace == NULL || output_ids == NULL) {
        printf("FAIL: cannot allocate host-test caller workspaces\n");
        goto cleanup;
    }

    reader.cursor = reference_data + TEXT_REFERENCE_PREFIX_SIZE;
    reader.end = reference_data + reference_size;
    for (case_index = 0u; case_index < case_count; ++case_index) {
        uint32_t name_length;
        uint32_t input_length;
        uint32_t expected_count;
        uint32_t decoded_length;
        uint32_t skip_length;
        const uint8_t *name;
        const uint8_t *input;
        const uint8_t *expected_ids;
        const uint8_t *decoded;
        const uint8_t *decoded_skip;

        if (text_u32(&reader, &name_length) != 0 ||
            text_u32(&reader, &input_length) != 0 ||
            text_u32(&reader, &expected_count) != 0 ||
            text_u32(&reader, &decoded_length) != 0 ||
            text_u32(&reader, &skip_length) != 0 ||
            (size_t)expected_count > (size_t)(reader.end - reader.cursor) / 4u) {
            printf("FAIL: truncated text reference record %u\n", (unsigned)case_index);
            goto cleanup;
        }
        name = text_take(&reader, (size_t)name_length);
        input = text_take(&reader, (size_t)input_length);
        expected_ids = text_take(&reader, (size_t)expected_count * 4u);
        decoded = text_take(&reader, (size_t)decoded_length);
        decoded_skip = text_take(&reader, (size_t)skip_length);
        if (name == NULL || input == NULL || expected_ids == NULL ||
            decoded == NULL || decoded_skip == NULL) {
            printf("FAIL: truncated text reference payload %u\n", (unsigned)case_index);
            goto cleanup;
        }
        if (test_one_case(
                &asset, name, name_length, input, input_length,
                expected_ids, expected_count, decoded, decoded_length,
                decoded_skip, skip_length,
                unicode_workspace, workspace_capacity,
                piece_workspace, workspace_capacity,
                output_ids, workspace_capacity) != 0) {
            goto cleanup;
        }
    }
    if (reader.cursor != reader.end) {
        printf("FAIL: trailing bytes in text reference\n");
        goto cleanup;
    }
    if (test_invalid_utf8(&asset, unicode_workspace, piece_workspace, output_ids) != 0) {
        goto cleanup;
    }

    printf("PASS: PS-native Qwen text tokenizer host differential\n");
    printf("  tracked_golden_cases=%u\n", (unsigned)golden_count);
    printf("  differential_cases=%u\n", (unsigned)(case_count - golden_count));
    printf("  total_cases=%u\n", (unsigned)case_count);
    printf("  invalid_utf8_cases=7\n");
    result = 0;

cleanup:
    free(output_ids);
    free(piece_workspace);
    free(unicode_workspace);
    free(reference_data);
    free(asset_data);
    return result;
}
