#include "qweb_http.h"

#include <stdio.h>
#include <string.h>

typedef struct qweb_http_header_parse_result {
    qweb_http_parse_status_t status;
    size_t error_offset;
} qweb_http_header_parse_result_t;

static qweb_http_feed_result_t qweb_http_feed_result(
    qweb_http_parse_status_t status,
    size_t consumed,
    size_t error_offset)
{
    qweb_http_feed_result_t result;

    result.status = status;
    result.consumed = consumed;
    result.error_offset = error_offset;
    return result;
}

static qweb_http_header_parse_result_t qweb_http_header_result(
    qweb_http_parse_status_t status,
    size_t error_offset)
{
    qweb_http_header_parse_result_t result;

    result.status = status;
    result.error_offset = error_offset;
    return result;
}

static int qweb_http_is_tchar(unsigned char value)
{
    if ((value >= (unsigned char)'0' && value <= (unsigned char)'9') ||
        (value >= (unsigned char)'A' && value <= (unsigned char)'Z') ||
        (value >= (unsigned char)'a' && value <= (unsigned char)'z')) {
        return 1;
    }
    switch (value) {
    case (unsigned char)'!':
    case (unsigned char)'#':
    case (unsigned char)'$':
    case (unsigned char)'%':
    case (unsigned char)'&':
    case (unsigned char)'\'':
    case (unsigned char)'*':
    case (unsigned char)'+':
    case (unsigned char)'-':
    case (unsigned char)'.':
    case (unsigned char)'^':
    case (unsigned char)'_':
    case (unsigned char)'`':
    case (unsigned char)'|':
    case (unsigned char)'~':
        return 1;
    default:
        return 0;
    }
}

static unsigned char qweb_http_ascii_lower(unsigned char value)
{
    if (value >= (unsigned char)'A' && value <= (unsigned char)'Z') {
        return (unsigned char)(value + ((unsigned char)'a' - (unsigned char)'A'));
    }
    return value;
}

static int qweb_http_case_equal(
    const char *left,
    size_t left_length,
    const char *right)
{
    size_t index;
    size_t right_length = strlen(right);

    if (left_length != right_length) return 0;
    for (index = 0u; index < left_length; ++index) {
        if (qweb_http_ascii_lower((unsigned char)left[index]) !=
            qweb_http_ascii_lower((unsigned char)right[index])) {
            return 0;
        }
    }
    return 1;
}

static size_t qweb_http_find_crlf(
    const char *bytes,
    size_t start,
    size_t length)
{
    size_t index;

    if (length < 2u || start >= length - 1u) return SIZE_MAX;
    for (index = start; index + 1u < length; ++index) {
        if (bytes[index] == '\r' && bytes[index + 1u] == '\n') {
            return index;
        }
    }
    return SIZE_MAX;
}

static int qweb_http_parse_decimal_size(
    const char *bytes,
    size_t length,
    size_t *value)
{
    size_t parsed = 0u;
    size_t index;

    if (bytes == NULL || value == NULL || length == 0u) return 0;
    for (index = 0u; index < length; ++index) {
        unsigned char current = (unsigned char)bytes[index];
        size_t digit;

        if (current < (unsigned char)'0' || current > (unsigned char)'9') {
            return 0;
        }
        digit = (size_t)(current - (unsigned char)'0');
        if (parsed > (SIZE_MAX - digit) / 10u) return 0;
        parsed = parsed * 10u + digit;
    }
    *value = parsed;
    return 1;
}

static int qweb_http_value_has_token(
    const char *bytes,
    size_t length,
    const char *wanted)
{
    size_t offset = 0u;

    while (offset < length) {
        size_t start;
        size_t end;

        while (offset < length &&
               (bytes[offset] == ' ' || bytes[offset] == '\t' ||
                bytes[offset] == ',')) {
            ++offset;
        }
        start = offset;
        while (offset < length && bytes[offset] != ',') ++offset;
        end = offset;
        while (end > start &&
               (bytes[end - 1u] == ' ' || bytes[end - 1u] == '\t')) {
            --end;
        }
        if (qweb_http_case_equal(bytes + start, end - start, wanted)) {
            return 1;
        }
    }
    return 0;
}

static qweb_http_header_parse_result_t qweb_http_parse_request_line(
    qweb_http_request_t *request,
    const char *header,
    size_t line_end)
{
    size_t first_space = SIZE_MAX;
    size_t second_space = SIZE_MAX;
    size_t index;
    size_t method_length;
    size_t target_length;
    size_t version_length;

    for (index = 0u; index < line_end; ++index) {
        if (header[index] == ' ') {
            if (first_space == SIZE_MAX) first_space = index;
            else if (second_space == SIZE_MAX) second_space = index;
            else return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX,
                                                index);
        }
    }
    if (first_space == SIZE_MAX || second_space == SIZE_MAX ||
        first_space == 0u || second_space == first_space + 1u ||
        second_space + 1u >= line_end) {
        return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, 0u);
    }

    method_length = first_space;
    target_length = second_space - first_space - 1u;
    version_length = line_end - second_space - 1u;
    if (method_length > QWEB_HTTP_MAX_METHOD_BYTES ||
        target_length > QWEB_HTTP_MAX_TARGET_BYTES) {
        return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE,
                                       first_space);
    }
    for (index = 0u; index < method_length; ++index) {
        if (!qweb_http_is_tchar((unsigned char)header[index])) {
            return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, index);
        }
    }
    if (header[first_space + 1u] != '/') {
        return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX,
                                       first_space + 1u);
    }
    for (index = first_space + 1u; index < second_space; ++index) {
        unsigned char value = (unsigned char)header[index];

        if (value < 0x21u || value > 0x7eu || value == (unsigned char)'#') {
            return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, index);
        }
    }
    if (version_length != 8u ||
        memcmp(header + second_space + 1u, "HTTP/1.1", 8u) != 0) {
        return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_VERSION,
                                       second_space + 1u);
    }

    memcpy(request->method, header, method_length);
    request->method[method_length] = '\0';
    request->method_length = method_length;
    memcpy(request->target, header + first_space + 1u, target_length);
    request->target[target_length] = '\0';
    request->target_length = target_length;
    return qweb_http_header_result(QWEB_HTTP_PARSE_IN_PROGRESS, 0u);
}

static qweb_http_header_parse_result_t qweb_http_parse_headers(
    qweb_http_request_t *request,
    const char *header,
    size_t header_length)
{
    size_t line_end;
    size_t offset;
    int saw_host = 0;
    int saw_content_length = 0;
    int saw_content_type = 0;
    qweb_http_header_parse_result_t result;

    line_end = qweb_http_find_crlf(header, 0u, header_length);
    if (line_end == SIZE_MAX) {
        return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, 0u);
    }
    result = qweb_http_parse_request_line(request, header, line_end);
    if (result.status < 0) return result;
    offset = line_end + 2u;

    while (offset + 1u < header_length) {
        size_t colon = SIZE_MAX;
        size_t name_start = offset;
        size_t name_length;
        size_t value_start;
        size_t value_end;
        size_t index;

        line_end = qweb_http_find_crlf(header, offset, header_length);
        if (line_end == SIZE_MAX) {
            return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, offset);
        }
        if (line_end == offset) {
            if (line_end + 2u != header_length) {
                return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX,
                                               line_end + 2u);
            }
            break;
        }
        if (header[offset] == ' ' || header[offset] == '\t') {
            return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, offset);
        }
        for (index = offset; index < line_end; ++index) {
            if (header[index] == ':') {
                colon = index;
                break;
            }
            if (!qweb_http_is_tchar((unsigned char)header[index])) {
                return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX,
                                               index);
            }
        }
        if (colon == SIZE_MAX || colon == name_start) {
            return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX, offset);
        }

        name_length = colon - name_start;
        value_start = colon + 1u;
        while (value_start < line_end &&
               (header[value_start] == ' ' || header[value_start] == '\t')) {
            ++value_start;
        }
        value_end = line_end;
        while (value_end > value_start &&
               (header[value_end - 1u] == ' ' ||
                header[value_end - 1u] == '\t')) {
            --value_end;
        }

        if (qweb_http_case_equal(header + name_start, name_length, "host")) {
            if (saw_host || value_start == value_end) {
                return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_HOST,
                                               name_start);
            }
            saw_host = 1;
        } else if (qweb_http_case_equal(header + name_start, name_length,
                                        "content-length")) {
            size_t content_length;

            if (saw_content_length ||
                !qweb_http_parse_decimal_size(header + value_start,
                                              value_end - value_start,
                                              &content_length)) {
                return qweb_http_header_result(
                    QWEB_HTTP_PARSE_ERR_CONTENT_LENGTH, name_start);
            }
            if (content_length > QWEB_HTTP_MAX_BODY_BYTES) {
                return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_BODY_TOO_LARGE,
                                               value_start);
            }
            saw_content_length = 1;
            request->content_length = content_length;
        } else if (qweb_http_case_equal(header + name_start, name_length,
                                        "transfer-encoding")) {
            return qweb_http_header_result(
                QWEB_HTTP_PARSE_ERR_TRANSFER_ENCODING, name_start);
        } else if (qweb_http_case_equal(header + name_start, name_length,
                                        "content-type")) {
            size_t content_type_length = value_end - value_start;

            if (saw_content_type) {
                return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_SYNTAX,
                                               name_start);
            }
            if (content_type_length > QWEB_HTTP_MAX_CONTENT_TYPE_BYTES) {
                return qweb_http_header_result(
                    QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE, value_start);
            }
            memcpy(request->content_type,
                   header + value_start,
                   content_type_length);
            request->content_type[content_type_length] = '\0';
            request->content_type_length = content_type_length;
            request->has_content_type = 1u;
            saw_content_type = 1;
        } else if (qweb_http_case_equal(header + name_start, name_length,
                                        "connection")) {
            if (qweb_http_value_has_token(header + value_start,
                                          value_end - value_start,
                                          "close")) {
                request->connection_close_requested = 1u;
            }
        }
        offset = line_end + 2u;
    }

    if (!saw_host) {
        return qweb_http_header_result(QWEB_HTTP_PARSE_ERR_HOST, header_length);
    }
    return qweb_http_header_result(QWEB_HTTP_PARSE_IN_PROGRESS, 0u);
}

void qweb_http_parser_init(qweb_http_parser_t *parser)
{
    if (parser == NULL) return;
    memset(parser, 0, sizeof(*parser));
    parser->status = QWEB_HTTP_PARSE_IN_PROGRESS;
}

qweb_http_feed_result_t qweb_http_parser_feed(
    qweb_http_parser_t *parser,
    const uint8_t *bytes,
    size_t length)
{
    size_t consumed = 0u;

    if (parser == NULL || (bytes == NULL && length != 0u)) {
        return qweb_http_feed_result(QWEB_HTTP_PARSE_ERR_NULL, 0u, 0u);
    }
    if (parser->status == QWEB_HTTP_PARSE_COMPLETE) {
        return qweb_http_feed_result(parser->status, 0u, parser->error_offset);
    }
    if (parser->status < 0) {
        return qweb_http_feed_result(QWEB_HTTP_PARSE_ERR_STATE,
                                     0u,
                                     parser->error_offset);
    }

    while (consumed < length) {
        if (parser->header_length < 4u ||
            memcmp(parser->header_bytes + parser->header_length - 4u,
                   "\r\n\r\n",
                   4u) != 0) {
            unsigned char current = bytes[consumed];
            unsigned char previous = parser->header_length == 0u
                                         ? 0u
                                         : (unsigned char)parser->header_bytes[
                                               parser->header_length - 1u];

            if (parser->header_length >= QWEB_HTTP_MAX_HEADER_BYTES) {
                parser->status = QWEB_HTTP_PARSE_ERR_HEADERS_TOO_LARGE;
                parser->error_offset = parser->total_received;
                return qweb_http_feed_result(parser->status,
                                             consumed,
                                             parser->error_offset);
            }
            if ((current == (unsigned char)'\n' &&
                 previous != (unsigned char)'\r') ||
                (previous == (unsigned char)'\r' &&
                 current != (unsigned char)'\n') ||
                (current != (unsigned char)'\r' &&
                 current != (unsigned char)'\n' &&
                 current != (unsigned char)'\t' &&
                 (current < 0x20u || current > 0x7eu))) {
                parser->status = QWEB_HTTP_PARSE_ERR_SYNTAX;
                parser->error_offset = parser->total_received;
                return qweb_http_feed_result(parser->status,
                                             consumed + 1u,
                                             parser->error_offset);
            }
            parser->header_bytes[parser->header_length++] = (char)current;
            parser->header_bytes[parser->header_length] = '\0';
            ++consumed;
            ++parser->total_received;

            if (parser->header_length >= 4u &&
                memcmp(parser->header_bytes + parser->header_length - 4u,
                       "\r\n\r\n",
                       4u) == 0) {
                qweb_http_header_parse_result_t header_result =
                    qweb_http_parse_headers(&parser->request,
                                            parser->header_bytes,
                                            parser->header_length);

                if (header_result.status < 0) {
                    parser->status = header_result.status;
                    parser->error_offset = header_result.error_offset;
                    return qweb_http_feed_result(parser->status,
                                                 consumed,
                                                 parser->error_offset);
                }
                if (parser->request.content_length == 0u) {
                    parser->status = QWEB_HTTP_PARSE_COMPLETE;
                    return qweb_http_feed_result(parser->status, consumed, 0u);
                }
            }
            continue;
        }

        {
            size_t remaining = parser->request.content_length -
                               parser->request.body_length;
            size_t available = length - consumed;
            size_t copied = remaining < available ? remaining : available;

            memcpy(parser->request.body + parser->request.body_length,
                   bytes + consumed,
                   copied);
            parser->request.body_length += copied;
            parser->total_received += copied;
            consumed += copied;
            if (parser->request.body_length == parser->request.content_length) {
                parser->status = QWEB_HTTP_PARSE_COMPLETE;
                return qweb_http_feed_result(parser->status, consumed, 0u);
            }
        }
    }

    return qweb_http_feed_result(QWEB_HTTP_PARSE_IN_PROGRESS, consumed, 0u);
}

const char *qweb_http_parse_status_name(qweb_http_parse_status_t status)
{
    switch (status) {
    case QWEB_HTTP_PARSE_IN_PROGRESS: return "IN_PROGRESS";
    case QWEB_HTTP_PARSE_COMPLETE: return "COMPLETE";
    case QWEB_HTTP_PARSE_ERR_NULL: return "NULL";
    case QWEB_HTTP_PARSE_ERR_STATE: return "STATE";
    case QWEB_HTTP_PARSE_ERR_SYNTAX: return "SYNTAX";
    case QWEB_HTTP_PARSE_ERR_VERSION: return "VERSION";
    case QWEB_HTTP_PARSE_ERR_HOST: return "HOST";
    case QWEB_HTTP_PARSE_ERR_CONTENT_LENGTH: return "CONTENT_LENGTH";
    case QWEB_HTTP_PARSE_ERR_TRANSFER_ENCODING: return "TRANSFER_ENCODING";
    case QWEB_HTTP_PARSE_ERR_HEADERS_TOO_LARGE: return "HEADERS_TOO_LARGE";
    case QWEB_HTTP_PARSE_ERR_BODY_TOO_LARGE: return "BODY_TOO_LARGE";
    case QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE: return "FIELD_TOO_LARGE";
    default: return "UNKNOWN";
    }
}

const char *qweb_http_reason_phrase(uint16_t status_code)
{
    switch (status_code) {
    case 200u: return "OK";
    case 202u: return "Accepted";
    case 400u: return "Bad Request";
    case 404u: return "Not Found";
    case 405u: return "Method Not Allowed";
    case 408u: return "Request Timeout";
    case 409u: return "Conflict";
    case 411u: return "Length Required";
    case 413u: return "Content Too Large";
    case 415u: return "Unsupported Media Type";
    case 422u: return "Unprocessable Content";
    case 431u: return "Request Header Fields Too Large";
    case 500u: return "Internal Server Error";
    case 503u: return "Service Unavailable";
    default: return NULL;
    }
}

static const char *qweb_http_content_type_name(
    qweb_http_content_type_t content_type)
{
    switch (content_type) {
    case QWEB_HTTP_CONTENT_JSON: return "application/json; charset=utf-8";
    case QWEB_HTTP_CONTENT_HTML: return "text/html; charset=utf-8";
    case QWEB_HTTP_CONTENT_JAVASCRIPT:
        return "text/javascript; charset=utf-8";
    case QWEB_HTTP_CONTENT_CSS: return "text/css; charset=utf-8";
    case QWEB_HTTP_CONTENT_TEXT: return "text/plain; charset=utf-8";
    case QWEB_HTTP_CONTENT_OCTETS: return "application/octet-stream";
    default: return NULL;
    }
}

qweb_http_format_status_t qweb_http_format_response(
    uint16_t status_code,
    qweb_http_content_type_t content_type,
    const uint8_t *body,
    size_t body_length,
    uint8_t *output,
    size_t output_capacity,
    size_t *output_length)
{
    char header[256];
    const char *reason;
    const char *type_name;
    int header_length;
    size_t total_length;

    if (output == NULL || output_length == NULL ||
        (body == NULL && body_length != 0u)) {
        return QWEB_HTTP_FORMAT_ERR_NULL;
    }
    *output_length = 0u;
    reason = qweb_http_reason_phrase(status_code);
    type_name = qweb_http_content_type_name(content_type);
    if (reason == NULL || type_name == NULL) {
        return QWEB_HTTP_FORMAT_ERR_STATUS;
    }

    header_length = snprintf(
        header,
        sizeof(header),
        "HTTP/1.1 %u %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n",
        (unsigned)status_code,
        reason,
        type_name,
        body_length);
    if (header_length < 0 || (size_t)header_length >= sizeof(header)) {
        return QWEB_HTTP_FORMAT_ERR_STATUS;
    }
    if ((size_t)header_length > SIZE_MAX - body_length) {
        return QWEB_HTTP_FORMAT_ERR_CAPACITY;
    }
    total_length = (size_t)header_length + body_length;
    if (total_length > output_capacity) {
        return QWEB_HTTP_FORMAT_ERR_CAPACITY;
    }

    memcpy(output, header, (size_t)header_length);
    if (body_length != 0u) {
        memcpy(output + (size_t)header_length, body, body_length);
    }
    *output_length = total_length;
    return QWEB_HTTP_FORMAT_OK;
}
