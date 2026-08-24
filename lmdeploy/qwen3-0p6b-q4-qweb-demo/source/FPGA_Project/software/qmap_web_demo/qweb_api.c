#include "qweb_api.h"

#include <limits.h>
#include <string.h>

typedef struct qweb_json_cursor {
    const uint8_t *bytes;
    size_t length;
    size_t offset;
} qweb_json_cursor_t;

static qweb_api_parse_result_t qweb_api_result(
    qweb_api_parse_status_t status,
    size_t error_offset)
{
    qweb_api_parse_result_t result;

    result.status = status;
    result.error_offset = error_offset;
    return result;
}

static int qweb_json_is_space(uint8_t value)
{
    return value == (uint8_t)' ' || value == (uint8_t)'\t' ||
           value == (uint8_t)'\r' || value == (uint8_t)'\n';
}

static void qweb_json_skip_space(qweb_json_cursor_t *cursor)
{
    while (cursor->offset < cursor->length &&
           qweb_json_is_space(cursor->bytes[cursor->offset])) {
        ++cursor->offset;
    }
}

static int qweb_utf8_scalar(
    const uint8_t *bytes,
    size_t length,
    uint32_t *codepoint,
    size_t *scalar_length)
{
    uint8_t first;
    uint32_t value;
    size_t needed;
    size_t index;

    if (bytes == NULL || codepoint == NULL || scalar_length == NULL ||
        length == 0u) {
        return 0;
    }
    first = bytes[0];
    if (first <= 0x7fu) {
        *codepoint = first;
        *scalar_length = 1u;
        return 1;
    }
    if (first >= 0xc2u && first <= 0xdfu) {
        needed = 2u;
        value = (uint32_t)(first & 0x1fu);
    } else if (first >= 0xe0u && first <= 0xefu) {
        needed = 3u;
        value = (uint32_t)(first & 0x0fu);
    } else if (first >= 0xf0u && first <= 0xf4u) {
        needed = 4u;
        value = (uint32_t)(first & 0x07u);
    } else {
        return 0;
    }
    if (length < needed) return 0;
    for (index = 1u; index < needed; ++index) {
        if ((bytes[index] & 0xc0u) != 0x80u) return 0;
        value = (value << 6u) | (uint32_t)(bytes[index] & 0x3fu);
    }
    if ((needed == 3u && first == 0xe0u && bytes[1] < 0xa0u) ||
        (needed == 3u && first == 0xedu && bytes[1] > 0x9fu) ||
        (needed == 4u && first == 0xf0u && bytes[1] < 0x90u) ||
        (needed == 4u && first == 0xf4u && bytes[1] > 0x8fu) ||
        value > UINT32_C(0x10ffff) ||
        (value >= UINT32_C(0xd800) && value <= UINT32_C(0xdfff))) {
        return 0;
    }
    *codepoint = value;
    *scalar_length = needed;
    return 1;
}

static size_t qweb_utf8_encode(uint32_t codepoint, uint8_t encoded[4])
{
    if (codepoint <= UINT32_C(0x7f)) {
        encoded[0] = (uint8_t)codepoint;
        return 1u;
    }
    if (codepoint <= UINT32_C(0x7ff)) {
        encoded[0] = (uint8_t)(0xc0u | (codepoint >> 6u));
        encoded[1] = (uint8_t)(0x80u | (codepoint & UINT32_C(0x3f)));
        return 2u;
    }
    if (codepoint <= UINT32_C(0xffff)) {
        encoded[0] = (uint8_t)(0xe0u | (codepoint >> 12u));
        encoded[1] = (uint8_t)(0x80u |
                               ((codepoint >> 6u) & UINT32_C(0x3f)));
        encoded[2] = (uint8_t)(0x80u | (codepoint & UINT32_C(0x3f)));
        return 3u;
    }
    encoded[0] = (uint8_t)(0xf0u | (codepoint >> 18u));
    encoded[1] = (uint8_t)(0x80u | ((codepoint >> 12u) & UINT32_C(0x3f)));
    encoded[2] = (uint8_t)(0x80u | ((codepoint >> 6u) & UINT32_C(0x3f)));
    encoded[3] = (uint8_t)(0x80u | (codepoint & UINT32_C(0x3f)));
    return 4u;
}

static int qweb_json_hex_value(uint8_t value, uint32_t *digit)
{
    if (value >= (uint8_t)'0' && value <= (uint8_t)'9') {
        *digit = (uint32_t)(value - (uint8_t)'0');
        return 1;
    }
    if (value >= (uint8_t)'a' && value <= (uint8_t)'f') {
        *digit = (uint32_t)(value - (uint8_t)'a') + 10u;
        return 1;
    }
    if (value >= (uint8_t)'A' && value <= (uint8_t)'F') {
        *digit = (uint32_t)(value - (uint8_t)'A') + 10u;
        return 1;
    }
    return 0;
}

static qweb_api_parse_result_t qweb_json_parse_hex4(
    qweb_json_cursor_t *cursor,
    uint32_t *value)
{
    uint32_t parsed = 0u;
    size_t start = cursor->offset;
    size_t index;

    if (cursor->length - cursor->offset < 4u) {
        return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE, start);
    }
    for (index = 0u; index < 4u; ++index) {
        uint32_t digit;

        if (!qweb_json_hex_value(cursor->bytes[cursor->offset], &digit)) {
            return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE, cursor->offset);
        }
        parsed = (parsed << 4u) | digit;
        ++cursor->offset;
    }
    *value = parsed;
    return qweb_api_result(QWEB_API_PARSE_OK, 0u);
}

static qweb_api_parse_result_t qweb_json_append_decoded(
    uint8_t *output,
    size_t output_capacity,
    size_t *output_length,
    const uint8_t *bytes,
    size_t length,
    size_t error_offset)
{
    if (*output_length > output_capacity ||
        length > output_capacity - *output_length) {
        return qweb_api_result(QWEB_API_PARSE_ERR_CAPACITY, error_offset);
    }
    if (length != 0u) {
        memcpy(output + *output_length, bytes, length);
        *output_length += length;
    }
    return qweb_api_result(QWEB_API_PARSE_OK, 0u);
}

static qweb_api_parse_result_t qweb_json_parse_string(
    qweb_json_cursor_t *cursor,
    uint8_t *output,
    size_t output_capacity,
    size_t *output_length)
{
    size_t used = 0u;

    if (cursor->offset >= cursor->length ||
        cursor->bytes[cursor->offset] != (uint8_t)'"') {
        return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor->offset);
    }
    ++cursor->offset;
    while (cursor->offset < cursor->length) {
        size_t source_offset = cursor->offset;
        uint8_t current = cursor->bytes[cursor->offset++];

        if (current == (uint8_t)'"') {
            *output_length = used;
            return qweb_api_result(QWEB_API_PARSE_OK, 0u);
        }
        if (current < 0x20u) {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, source_offset);
        }
        if (current == (uint8_t)'\\') {
            uint8_t escaped;
            uint8_t decoded[4];
            size_t decoded_length = 1u;
            qweb_api_parse_result_t append_result;

            if (cursor->offset >= cursor->length) {
                return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE,
                                       cursor->offset);
            }
            escaped = cursor->bytes[cursor->offset++];
            switch (escaped) {
            case (uint8_t)'"': decoded[0] = (uint8_t)'"'; break;
            case (uint8_t)'\\': decoded[0] = (uint8_t)'\\'; break;
            case (uint8_t)'/': decoded[0] = (uint8_t)'/'; break;
            case (uint8_t)'b': decoded[0] = 0x08u; break;
            case (uint8_t)'f': decoded[0] = 0x0cu; break;
            case (uint8_t)'n': decoded[0] = (uint8_t)'\n'; break;
            case (uint8_t)'r': decoded[0] = (uint8_t)'\r'; break;
            case (uint8_t)'t': decoded[0] = (uint8_t)'\t'; break;
            case (uint8_t)'u': {
                uint32_t codepoint;
                qweb_api_parse_result_t hex_result =
                    qweb_json_parse_hex4(cursor, &codepoint);

                if (hex_result.status != QWEB_API_PARSE_OK) return hex_result;
                if (codepoint >= UINT32_C(0xd800) &&
                    codepoint <= UINT32_C(0xdbff)) {
                    uint32_t low;

                    if (cursor->length - cursor->offset < 6u ||
                        cursor->bytes[cursor->offset] != (uint8_t)'\\' ||
                        cursor->bytes[cursor->offset + 1u] != (uint8_t)'u') {
                        return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE,
                                               cursor->offset);
                    }
                    cursor->offset += 2u;
                    hex_result = qweb_json_parse_hex4(cursor, &low);
                    if (hex_result.status != QWEB_API_PARSE_OK) return hex_result;
                    if (low < UINT32_C(0xdc00) || low > UINT32_C(0xdfff)) {
                        return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE,
                                               cursor->offset - 4u);
                    }
                    codepoint = UINT32_C(0x10000) +
                                ((codepoint - UINT32_C(0xd800)) << 10u) +
                                (low - UINT32_C(0xdc00));
                } else if (codepoint >= UINT32_C(0xdc00) &&
                           codepoint <= UINT32_C(0xdfff)) {
                    return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE,
                                           cursor->offset - 4u);
                }
                decoded_length = qweb_utf8_encode(codepoint, decoded);
                break;
            }
            default:
                return qweb_api_result(QWEB_API_PARSE_ERR_ESCAPE,
                                       cursor->offset - 1u);
            }
            append_result = qweb_json_append_decoded(output,
                                                     output_capacity,
                                                     &used,
                                                     decoded,
                                                     decoded_length,
                                                     source_offset);
            if (append_result.status != QWEB_API_PARSE_OK) return append_result;
            continue;
        }
        if (current < 0x80u) {
            qweb_api_parse_result_t append_result =
                qweb_json_append_decoded(output,
                                         output_capacity,
                                         &used,
                                         &current,
                                         1u,
                                         source_offset);

            if (append_result.status != QWEB_API_PARSE_OK) return append_result;
        } else {
            uint32_t codepoint;
            size_t scalar_length;
            qweb_api_parse_result_t append_result;

            --cursor->offset;
            if (!qweb_utf8_scalar(cursor->bytes + cursor->offset,
                                  cursor->length - cursor->offset,
                                  &codepoint,
                                  &scalar_length)) {
                return qweb_api_result(QWEB_API_PARSE_ERR_UTF8,
                                       cursor->offset);
            }
            (void)codepoint;
            append_result = qweb_json_append_decoded(
                output,
                output_capacity,
                &used,
                cursor->bytes + cursor->offset,
                scalar_length,
                cursor->offset);
            if (append_result.status != QWEB_API_PARSE_OK) return append_result;
            cursor->offset += scalar_length;
        }
    }
    return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor->offset);
}

static qweb_api_parse_result_t qweb_json_parse_u32(
    qweb_json_cursor_t *cursor,
    uint32_t *value)
{
    uint64_t parsed = 0u;
    size_t start = cursor->offset;

    if (cursor->offset >= cursor->length ||
        cursor->bytes[cursor->offset] < (uint8_t)'0' ||
        cursor->bytes[cursor->offset] > (uint8_t)'9') {
        return qweb_api_result(QWEB_API_PARSE_ERR_NUMBER, cursor->offset);
    }
    if (cursor->bytes[cursor->offset] == (uint8_t)'0' &&
        cursor->offset + 1u < cursor->length &&
        cursor->bytes[cursor->offset + 1u] >= (uint8_t)'0' &&
        cursor->bytes[cursor->offset + 1u] <= (uint8_t)'9') {
        return qweb_api_result(QWEB_API_PARSE_ERR_NUMBER,
                               cursor->offset + 1u);
    }
    while (cursor->offset < cursor->length &&
           cursor->bytes[cursor->offset] >= (uint8_t)'0' &&
           cursor->bytes[cursor->offset] <= (uint8_t)'9') {
        uint32_t digit = (uint32_t)(cursor->bytes[cursor->offset] -
                                    (uint8_t)'0');

        parsed = parsed * UINT64_C(10) + (uint64_t)digit;
        if (parsed > UINT32_MAX) {
            return qweb_api_result(QWEB_API_PARSE_ERR_NUMBER, start);
        }
        ++cursor->offset;
    }
    if (cursor->offset < cursor->length &&
        (cursor->bytes[cursor->offset] == (uint8_t)'.' ||
         cursor->bytes[cursor->offset] == (uint8_t)'e' ||
         cursor->bytes[cursor->offset] == (uint8_t)'E' ||
         cursor->bytes[cursor->offset] == (uint8_t)'+' ||
         cursor->bytes[cursor->offset] == (uint8_t)'-')) {
        return qweb_api_result(QWEB_API_PARSE_ERR_NUMBER, cursor->offset);
    }
    *value = (uint32_t)parsed;
    return qweb_api_result(QWEB_API_PARSE_OK, 0u);
}

static qweb_api_parse_result_t qweb_json_parse_tokens(
    qweb_json_cursor_t *cursor,
    qweb_generate_request_t *request)
{
    if (cursor->offset >= cursor->length ||
        cursor->bytes[cursor->offset] != (uint8_t)'[') {
        return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor->offset);
    }
    ++cursor->offset;
    qweb_json_skip_space(cursor);
    if (cursor->offset < cursor->length &&
        cursor->bytes[cursor->offset] == (uint8_t)']') {
        return qweb_api_result(QWEB_API_PARSE_ERR_RANGE, cursor->offset);
    }

    for (;;) {
        uint32_t token_id;
        qweb_api_parse_result_t number_result;
        size_t token_offset;

        if (request->token_count >= QWEB_API_MAX_TOKENS) {
            return qweb_api_result(QWEB_API_PARSE_ERR_CAPACITY,
                                   cursor->offset);
        }
        token_offset = cursor->offset;
        number_result = qweb_json_parse_u32(cursor, &token_id);
        if (number_result.status != QWEB_API_PARSE_OK) return number_result;
        if (token_id >= QWEB_API_MODEL_VOCAB_SIZE) {
            return qweb_api_result(QWEB_API_PARSE_ERR_RANGE, token_offset);
        }
        request->token_ids[request->token_count++] = token_id;
        qweb_json_skip_space(cursor);
        if (cursor->offset >= cursor->length) {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor->offset);
        }
        if (cursor->bytes[cursor->offset] == (uint8_t)']') {
            ++cursor->offset;
            return qweb_api_result(QWEB_API_PARSE_OK, 0u);
        }
        if (cursor->bytes[cursor->offset] != (uint8_t)',') {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor->offset);
        }
        ++cursor->offset;
        qweb_json_skip_space(cursor);
        if (cursor->offset < cursor->length &&
            cursor->bytes[cursor->offset] == (uint8_t)']') {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor->offset);
        }
    }
}

qweb_api_parse_result_t qweb_api_parse_generate_request(
    const uint8_t *json,
    size_t length,
    qweb_generate_request_t *request)
{
    qweb_json_cursor_t cursor;
    int saw_prompt = 0;
    int saw_tokens = 0;
    int saw_max_new_tokens = 0;

    if (request == NULL) {
        return qweb_api_result(QWEB_API_PARSE_ERR_NULL, 0u);
    }
    memset(request, 0, sizeof(*request));
    if (json == NULL) {
        return qweb_api_result(QWEB_API_PARSE_ERR_NULL, 0u);
    }
    cursor.bytes = json;
    cursor.length = length;
    cursor.offset = 0u;
    qweb_json_skip_space(&cursor);
    if (cursor.offset == cursor.length) {
        return qweb_api_result(QWEB_API_PARSE_ERR_EMPTY, cursor.offset);
    }
    if (cursor.bytes[cursor.offset] != (uint8_t)'{') {
        return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor.offset);
    }
    ++cursor.offset;
    qweb_json_skip_space(&cursor);
    if (cursor.offset < cursor.length &&
        cursor.bytes[cursor.offset] == (uint8_t)'}') {
        return qweb_api_result(QWEB_API_PARSE_ERR_MISSING_MEMBER,
                               cursor.offset);
    }

    for (;;) {
        uint8_t key[32];
        size_t key_length = 0u;
        size_t key_offset = cursor.offset;
        qweb_api_parse_result_t value_result;

        value_result = qweb_json_parse_string(&cursor,
                                              key,
                                              sizeof(key),
                                              &key_length);
        if (value_result.status != QWEB_API_PARSE_OK) return value_result;
        qweb_json_skip_space(&cursor);
        if (cursor.offset >= cursor.length ||
            cursor.bytes[cursor.offset] != (uint8_t)':') {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor.offset);
        }
        ++cursor.offset;
        qweb_json_skip_space(&cursor);

        if (key_length == 6u && memcmp(key, "prompt", 6u) == 0) {
            if (saw_prompt) {
                return qweb_api_result(QWEB_API_PARSE_ERR_DUPLICATE_MEMBER,
                                       key_offset);
            }
            saw_prompt = 1;
            value_result = qweb_json_parse_string(&cursor,
                                                  request->prompt,
                                                  sizeof(request->prompt),
                                                  &request->prompt_length);
            if (value_result.status != QWEB_API_PARSE_OK) return value_result;
            if (request->prompt_length == 0u) {
                return qweb_api_result(QWEB_API_PARSE_ERR_RANGE, key_offset);
            }
        } else if (key_length == 6u && memcmp(key, "tokens", 6u) == 0) {
            if (saw_tokens) {
                return qweb_api_result(QWEB_API_PARSE_ERR_DUPLICATE_MEMBER,
                                       key_offset);
            }
            saw_tokens = 1;
            value_result = qweb_json_parse_tokens(&cursor, request);
            if (value_result.status != QWEB_API_PARSE_OK) return value_result;
        } else if (key_length == 14u &&
                   memcmp(key, "max_new_tokens", 14u) == 0) {
            size_t number_offset = cursor.offset;

            if (saw_max_new_tokens) {
                return qweb_api_result(QWEB_API_PARSE_ERR_DUPLICATE_MEMBER,
                                       key_offset);
            }
            saw_max_new_tokens = 1;
            value_result = qweb_json_parse_u32(&cursor,
                                               &request->max_new_tokens);
            if (value_result.status != QWEB_API_PARSE_OK) return value_result;
            if (request->max_new_tokens == 0u ||
                request->max_new_tokens > QWEB_API_MAX_TOKENS) {
                return qweb_api_result(QWEB_API_PARSE_ERR_RANGE, number_offset);
            }
        } else {
            return qweb_api_result(QWEB_API_PARSE_ERR_UNKNOWN_MEMBER,
                                   key_offset);
        }

        qweb_json_skip_space(&cursor);
        if (cursor.offset >= cursor.length) {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor.offset);
        }
        if (cursor.bytes[cursor.offset] == (uint8_t)'}') {
            ++cursor.offset;
            break;
        }
        if (cursor.bytes[cursor.offset] != (uint8_t)',') {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor.offset);
        }
        ++cursor.offset;
        qweb_json_skip_space(&cursor);
        if (cursor.offset < cursor.length &&
            cursor.bytes[cursor.offset] == (uint8_t)'}') {
            return qweb_api_result(QWEB_API_PARSE_ERR_SYNTAX, cursor.offset);
        }
    }

    qweb_json_skip_space(&cursor);
    if (cursor.offset != cursor.length) {
        return qweb_api_result(QWEB_API_PARSE_ERR_TRAILING, cursor.offset);
    }
    if (!saw_max_new_tokens) {
        return qweb_api_result(QWEB_API_PARSE_ERR_MISSING_MEMBER, cursor.offset);
    }
    if ((saw_prompt ? 1 : 0) + (saw_tokens ? 1 : 0) != 1) {
        return qweb_api_result(QWEB_API_PARSE_ERR_INPUT_CHOICE, cursor.offset);
    }
    request->kind = saw_prompt ? QWEB_GENERATE_INPUT_PROMPT
                               : QWEB_GENERATE_INPUT_TOKENS;
    return qweb_api_result(QWEB_API_PARSE_OK, 0u);
}

const char *qweb_api_parse_status_name(qweb_api_parse_status_t status)
{
    switch (status) {
    case QWEB_API_PARSE_OK: return "OK";
    case QWEB_API_PARSE_ERR_NULL: return "NULL";
    case QWEB_API_PARSE_ERR_EMPTY: return "EMPTY";
    case QWEB_API_PARSE_ERR_SYNTAX: return "SYNTAX";
    case QWEB_API_PARSE_ERR_UTF8: return "UTF8";
    case QWEB_API_PARSE_ERR_ESCAPE: return "ESCAPE";
    case QWEB_API_PARSE_ERR_NUMBER: return "NUMBER";
    case QWEB_API_PARSE_ERR_UNKNOWN_MEMBER: return "UNKNOWN_MEMBER";
    case QWEB_API_PARSE_ERR_DUPLICATE_MEMBER: return "DUPLICATE_MEMBER";
    case QWEB_API_PARSE_ERR_MISSING_MEMBER: return "MISSING_MEMBER";
    case QWEB_API_PARSE_ERR_INPUT_CHOICE: return "INPUT_CHOICE";
    case QWEB_API_PARSE_ERR_RANGE: return "RANGE";
    case QWEB_API_PARSE_ERR_CAPACITY: return "CAPACITY";
    case QWEB_API_PARSE_ERR_TRAILING: return "TRAILING";
    default: return "UNKNOWN";
    }
}

void qweb_json_writer_init(
    qweb_json_writer_t *writer,
    uint8_t *buffer,
    size_t capacity)
{
    if (writer == NULL) return;
    writer->buffer = buffer;
    writer->capacity = capacity;
    writer->length = 0u;
    writer->status = buffer == NULL ? QWEB_JSON_WRITE_ERR_NULL
                                    : QWEB_JSON_WRITE_OK;
}

qweb_json_write_status_t qweb_json_writer_append_bytes(
    qweb_json_writer_t *writer,
    const uint8_t *bytes,
    size_t length)
{
    if (writer == NULL) return QWEB_JSON_WRITE_ERR_NULL;
    if (writer->status != QWEB_JSON_WRITE_OK) return writer->status;
    if (bytes == NULL && length != 0u) {
        writer->status = QWEB_JSON_WRITE_ERR_NULL;
        return writer->status;
    }
    if (writer->length > writer->capacity ||
        length > writer->capacity - writer->length) {
        writer->status = QWEB_JSON_WRITE_ERR_CAPACITY;
        return writer->status;
    }
    if (length != 0u) {
        memcpy(writer->buffer + writer->length, bytes, length);
        writer->length += length;
    }
    return writer->status;
}

qweb_json_write_status_t qweb_json_writer_append_cstr(
    qweb_json_writer_t *writer,
    const char *text)
{
    if (text == NULL) {
        if (writer != NULL) writer->status = QWEB_JSON_WRITE_ERR_NULL;
        return QWEB_JSON_WRITE_ERR_NULL;
    }
    return qweb_json_writer_append_bytes(writer,
                                         (const uint8_t *)text,
                                         strlen(text));
}

static qweb_json_write_status_t qweb_json_writer_append_u64(
    qweb_json_writer_t *writer,
    uint64_t value)
{
    uint8_t digits[20];
    size_t used = 0u;
    size_t index;

    do {
        digits[used++] = (uint8_t)('0' + (value % UINT64_C(10)));
        value /= UINT64_C(10);
    } while (value != 0u);
    for (index = 0u; index < used / 2u; ++index) {
        uint8_t temporary = digits[index];
        digits[index] = digits[used - index - 1u];
        digits[used - index - 1u] = temporary;
    }
    return qweb_json_writer_append_bytes(writer, digits, used);
}

qweb_json_write_status_t qweb_json_writer_append_u32(
    qweb_json_writer_t *writer,
    uint32_t value)
{
    return qweb_json_writer_append_u64(writer, (uint64_t)value);
}

qweb_json_write_status_t qweb_json_writer_append_i64(
    qweb_json_writer_t *writer,
    int64_t value)
{
    uint64_t magnitude;

    if (value < 0) {
        qweb_json_write_status_t status = qweb_json_writer_append_cstr(writer,
                                                                       "-");
        if (status != QWEB_JSON_WRITE_OK) return status;
        magnitude = (uint64_t)(-(value + INT64_C(1))) + UINT64_C(1);
    } else {
        magnitude = (uint64_t)value;
    }
    return qweb_json_writer_append_u64(writer, magnitude);
}

qweb_json_write_status_t qweb_json_writer_append_string(
    qweb_json_writer_t *writer,
    const uint8_t *utf8,
    size_t length)
{
    static const uint8_t hex[] = "0123456789abcdef";
    size_t offset = 0u;
    qweb_json_write_status_t status;

    if (writer == NULL || (utf8 == NULL && length != 0u)) {
        if (writer != NULL) writer->status = QWEB_JSON_WRITE_ERR_NULL;
        return QWEB_JSON_WRITE_ERR_NULL;
    }
    status = qweb_json_writer_append_cstr(writer, "\"");
    if (status != QWEB_JSON_WRITE_OK) return status;
    while (offset < length) {
        uint8_t current = utf8[offset];

        if (current == (uint8_t)'"' || current == (uint8_t)'\\') {
            uint8_t escaped[2] = {(uint8_t)'\\', current};

            status = qweb_json_writer_append_bytes(writer, escaped, 2u);
            ++offset;
        } else if (current < 0x20u) {
            uint8_t escaped[6] = {
                (uint8_t)'\\', (uint8_t)'u', (uint8_t)'0', (uint8_t)'0',
                hex[current >> 4u], hex[current & 0x0fu]
            };

            status = qweb_json_writer_append_bytes(writer, escaped, 6u);
            ++offset;
        } else if (current < 0x80u) {
            status = qweb_json_writer_append_bytes(writer, &current, 1u);
            ++offset;
        } else {
            uint32_t codepoint;
            size_t scalar_length;

            if (!qweb_utf8_scalar(utf8 + offset,
                                  length - offset,
                                  &codepoint,
                                  &scalar_length)) {
                writer->status = QWEB_JSON_WRITE_ERR_UTF8;
                return writer->status;
            }
            (void)codepoint;
            status = qweb_json_writer_append_bytes(writer,
                                                   utf8 + offset,
                                                   scalar_length);
            offset += scalar_length;
        }
        if (status != QWEB_JSON_WRITE_OK) return status;
    }
    return qweb_json_writer_append_cstr(writer, "\"");
}

qweb_json_write_status_t qweb_api_format_error_json(
    uint8_t *output,
    size_t output_capacity,
    const char *code,
    const char *message,
    size_t *output_length)
{
    qweb_json_writer_t writer;

    if (output_length == NULL || code == NULL || message == NULL) {
        return QWEB_JSON_WRITE_ERR_NULL;
    }
    *output_length = 0u;
    qweb_json_writer_init(&writer, output, output_capacity);
    (void)qweb_json_writer_append_cstr(&writer, "{\"error\":{\"code\":");
    (void)qweb_json_writer_append_string(&writer,
                                         (const uint8_t *)code,
                                         strlen(code));
    (void)qweb_json_writer_append_cstr(&writer, ",\"message\":");
    (void)qweb_json_writer_append_string(&writer,
                                         (const uint8_t *)message,
                                         strlen(message));
    (void)qweb_json_writer_append_cstr(&writer, "}}");
    if (writer.status == QWEB_JSON_WRITE_OK) *output_length = writer.length;
    return writer.status;
}
