#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qtk_tokenizer_runtime.h"

#define REFERENCE_PREFIX_SIZE 24u
#define TEST_BUFFER_SIZE 4096u

typedef struct {
    const uint8_t *cursor;
    const uint8_t *end;
} fixture_reader_t;

typedef struct {
    uint8_t data[TEST_BUFFER_SIZE];
    size_t length;
    int fail;
} byte_collector_t;

static uint32_t read_le32(const uint8_t *data)
{
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
}

static int read_file(const char *path, uint8_t **data, size_t *size)
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

static const uint8_t *fixture_take(fixture_reader_t *reader, size_t length)
{
    const uint8_t *result;
    size_t remaining = (size_t)(reader->end - reader->cursor);

    if (length > remaining) return NULL;
    result = reader->cursor;
    reader->cursor += length;
    return result;
}

static int fixture_u32(fixture_reader_t *reader, uint32_t *value)
{
    const uint8_t *bytes = fixture_take(reader, 4u);
    if (bytes == NULL) return 1;
    *value = read_le32(bytes);
    return 0;
}

static int collect_bytes(void *context, const uint8_t *bytes, size_t length)
{
    byte_collector_t *collector = (byte_collector_t *)context;

    if (collector->fail != 0) return 1;
    if (length > sizeof(collector->data) - collector->length) return 1;
    memcpy(collector->data + collector->length, bytes, length);
    collector->length += length;
    return 0;
}

static int fail_emitter(void *context, const uint8_t *bytes, size_t length)
{
    (void)context;
    (void)bytes;
    (void)length;
    return 1;
}

static int test_parser_errors(uint8_t *asset_data, size_t asset_size)
{
    qtk_asset_t parsed;
    uint8_t saved;

    if (qtk_asset_init(NULL, asset_data, asset_size) != QTK_ERR_NULL) {
        printf("FAIL: NULL asset destination was accepted\n");
        return 1;
    }
    if (qtk_asset_init(&parsed, asset_data, 127u) != QTK_ERR_TOO_SMALL) {
        printf("FAIL: truncated asset was accepted\n");
        return 1;
    }

    saved = asset_data[0];
    asset_data[0] ^= 1u;
    if (qtk_asset_init(&parsed, asset_data, asset_size) != QTK_ERR_MAGIC) {
        printf("FAIL: corrupt magic was accepted\n");
        asset_data[0] = saved;
        return 1;
    }
    asset_data[0] = saved;

    saved = asset_data[16];
    asset_data[16] ^= 1u;
    if (qtk_asset_init(&parsed, asset_data, asset_size) != QTK_ERR_ENDIAN) {
        printf("FAIL: corrupt endian tag was accepted\n");
        asset_data[16] = saved;
        return 1;
    }
    asset_data[16] = saved;

    saved = asset_data[asset_size - 1u];
    asset_data[asset_size - 1u] ^= 1u;
    if (qtk_asset_init(&parsed, asset_data, asset_size) != QTK_ERR_HASH) {
        printf("FAIL: corrupt payload was accepted\n");
        asset_data[asset_size - 1u] = saved;
        return 1;
    }
    asset_data[asset_size - 1u] = saved;
    return 0;
}

static int test_all_token_slices(
    const qtk_asset_t *asset,
    fixture_reader_t *reader,
    uint32_t reference_token_count)
{
    uint32_t token_id;

    if (reference_token_count != asset->token_count) {
        printf("FAIL: reference token count does not match asset\n");
        return 1;
    }
    for (token_id = 0u; token_id < reference_token_count; ++token_id) {
        uint32_t expected_length;
        uint32_t expected_flags;
        const uint8_t *expected_bytes;
        qtk_token_slice_t actual;
        qtk_status_t status;

        if (fixture_u32(reader, &expected_length) != 0 ||
            fixture_u32(reader, &expected_flags) != 0) {
            printf("FAIL: truncated token reference at ID %u\n", (unsigned)token_id);
            return 1;
        }
        expected_bytes = fixture_take(reader, (size_t)expected_length);
        if (expected_bytes == NULL) {
            printf("FAIL: truncated token bytes at ID %u\n", (unsigned)token_id);
            return 1;
        }
        status = qtk_token_slice(asset, token_id, &actual);
        if (status != QTK_OK || actual.length != (size_t)expected_length ||
            actual.flags != (uint8_t)expected_flags ||
            memcmp(actual.bytes, expected_bytes, actual.length) != 0) {
            printf("FAIL: Python token slice mismatch at ID %u status=%s\n",
                   (unsigned)token_id, qtk_status_name(status));
            return 1;
        }
    }
    return 0;
}

static int test_piece_encoder(
    const qtk_asset_t *asset,
    fixture_reader_t *reader,
    uint32_t piece_count,
    uint32_t max_piece_length)
{
    uint32_t *workspace;
    uint32_t piece_index;

    workspace = (uint32_t *)malloc(
        max_piece_length == 0u ? sizeof(*workspace) :
        (size_t)max_piece_length * sizeof(*workspace));
    if (workspace == NULL) {
        printf("FAIL: cannot allocate host-test workspace\n");
        return 1;
    }

    for (piece_index = 0u; piece_index < piece_count; ++piece_index) {
        uint32_t raw_length;
        uint32_t expected_count;
        const uint8_t *raw_bytes;
        const uint8_t *expected_bytes;
        size_t actual_count = SIZE_MAX;
        size_t token_index;
        qtk_status_t status;

        if (fixture_u32(reader, &raw_length) != 0 ||
            fixture_u32(reader, &expected_count) != 0 ||
            raw_length > max_piece_length || expected_count > raw_length) {
            printf("FAIL: invalid piece reference %u\n", (unsigned)piece_index);
            free(workspace);
            return 1;
        }
        raw_bytes = fixture_take(reader, (size_t)raw_length);
        expected_bytes = fixture_take(reader, (size_t)expected_count * 4u);
        if (raw_bytes == NULL || expected_bytes == NULL) {
            printf("FAIL: truncated piece reference %u\n", (unsigned)piece_index);
            free(workspace);
            return 1;
        }
        status = qtk_encode_raw_piece(
            asset,
            raw_length == 0u ? NULL : raw_bytes,
            (size_t)raw_length,
            raw_length == 0u ? NULL : workspace,
            (size_t)max_piece_length,
            &actual_count);
        if (status != QTK_OK || actual_count != (size_t)expected_count) {
            printf("FAIL: piece %u count mismatch status=%s actual=%zu expected=%u\n",
                   (unsigned)piece_index, qtk_status_name(status), actual_count,
                   (unsigned)expected_count);
            free(workspace);
            return 1;
        }
        for (token_index = 0u; token_index < actual_count; ++token_index) {
            uint32_t expected = read_le32(expected_bytes + token_index * 4u);
            if (workspace[token_index] != expected) {
                printf("FAIL: piece %u token %zu mismatch actual=%u expected=%u\n",
                       (unsigned)piece_index, token_index,
                       (unsigned)workspace[token_index], (unsigned)expected);
                free(workspace);
                return 1;
            }
        }
    }
    free(workspace);
    return 0;
}

static int test_runtime_edges(const qtk_asset_t *asset)
{
    static const uint8_t one_byte[1] = {'A'};
    static const uint32_t sequence[3] = {9707u, 151643u, 9707u};
    static const uint32_t one_token[1] = {9707u};
    uint32_t workspace[1];
    size_t output_count = 99u;
    qtk_token_slice_t hello;
    byte_collector_t collector;
    qtk_status_t status;

    status = qtk_encode_raw_piece(asset, one_byte, 1u, workspace, 0u, &output_count);
    if (status != QTK_ERR_WORKSPACE || output_count != 0u) {
        printf("FAIL: undersized workspace was accepted\n");
        return 1;
    }
    if (qtk_token_slice(asset, asset->token_count, &hello) !=
        QTK_ERR_MODEL_ONLY_TOKEN) {
        printf("FAIL: model-only token ID policy was not enforced\n");
        return 1;
    }
    if (qtk_token_slice(asset, asset->model_vocab_size, &hello) != QTK_ERR_TOKEN_ID) {
        printf("FAIL: out-of-model token ID was accepted\n");
        return 1;
    }
    if (qtk_token_slice(asset, 9707u, &hello) != QTK_OK) {
        printf("FAIL: expected Hello token is unavailable\n");
        return 1;
    }

    memset(&collector, 0, sizeof(collector));
    status = qtk_detokenize_ids(asset, sequence, 3u, 1, collect_bytes, &collector);
    if (status != QTK_OK || collector.length != hello.length * 2u ||
        memcmp(collector.data, hello.bytes, hello.length) != 0 ||
        memcmp(collector.data + hello.length, hello.bytes, hello.length) != 0) {
        printf("FAIL: streaming detokenizer skip-special mismatch\n");
        return 1;
    }
    if (qtk_detokenize_ids(asset, one_token, 1u, 0, fail_emitter, NULL) !=
        QTK_ERR_EMIT) {
        printf("FAIL: emitter error was not propagated\n");
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const uint8_t reference_magic[8] = {
        'Q', 'T', 'K', 'R', 'E', 'F', '1', 0
    };
    uint8_t *asset_data = NULL;
    size_t asset_size = 0u;
    uint8_t *fixture_data = NULL;
    size_t fixture_size = 0u;
    fixture_reader_t reader;
    qtk_asset_t asset;
    qtk_status_t status;
    uint32_t version;
    uint32_t token_count;
    uint32_t piece_count;
    uint32_t max_piece_length;
    int result = 1;

    if (argc != 3) {
        printf("usage: %s <asset.qtk> <python-reference.bin>\n", argv[0]);
        return 2;
    }
    if (read_file(argv[1], &asset_data, &asset_size) != 0 ||
        read_file(argv[2], &fixture_data, &fixture_size) != 0) {
        printf("FAIL: cannot read asset or Python reference fixture\n");
        goto cleanup;
    }
    if (fixture_size < REFERENCE_PREFIX_SIZE ||
        memcmp(fixture_data, reference_magic, sizeof(reference_magic)) != 0) {
        printf("FAIL: bad Python reference fixture header\n");
        goto cleanup;
    }
    version = read_le32(fixture_data + 8u);
    token_count = read_le32(fixture_data + 12u);
    piece_count = read_le32(fixture_data + 16u);
    max_piece_length = read_le32(fixture_data + 20u);
    if (version != 1u) {
        printf("FAIL: unsupported Python reference version\n");
        goto cleanup;
    }

    status = qtk_asset_init(&asset, asset_data, asset_size);
    if (status != QTK_OK) {
        printf("FAIL: asset parser returned %s\n", qtk_status_name(status));
        goto cleanup;
    }
    if (test_parser_errors(asset_data, asset_size) != 0) goto cleanup;

    reader.cursor = fixture_data + REFERENCE_PREFIX_SIZE;
    reader.end = fixture_data + fixture_size;
    if (test_all_token_slices(&asset, &reader, token_count) != 0) goto cleanup;
    if (test_piece_encoder(&asset, &reader, piece_count, max_piece_length) != 0) {
        goto cleanup;
    }
    if (reader.cursor != reader.end) {
        printf("FAIL: trailing bytes in Python reference fixture\n");
        goto cleanup;
    }
    if (test_runtime_edges(&asset) != 0) goto cleanup;

    printf("PASS: QTKBPE1 C runtime\n");
    printf("  token_slices_checked=%u\n", (unsigned)token_count);
    printf("  python_piece_cases=%u\n", (unsigned)piece_count);
    printf("  runtime_dynamic_allocations=0\n");
    printf("  scope=raw_byte_piece_only_no_NFC_or_Unicode_pretokenizer\n");
    result = 0;

cleanup:
    free(fixture_data);
    free(asset_data);
    return result;
}
