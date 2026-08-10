#ifndef QWEB_HTTP_H
#define QWEB_HTTP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deliberately small, fixed limits for a standalone/raw-lwIP application.
 * The parser never allocates and stops consuming input at one complete request.
 * qweb_http_parser_t is larger than the default BSP stack; allocate one
 * static/global instance per permitted connection and keep connection count
 * explicitly bounded.
 */
#define QWEB_HTTP_MAX_HEADER_BYTES       2048u
#define QWEB_HTTP_MAX_BODY_BYTES         8192u
#define QWEB_HTTP_MAX_METHOD_BYTES       8u
#define QWEB_HTTP_MAX_TARGET_BYTES       192u
#define QWEB_HTTP_MAX_CONTENT_TYPE_BYTES 63u

typedef enum qweb_http_parse_status {
    QWEB_HTTP_PARSE_IN_PROGRESS = 0,
    QWEB_HTTP_PARSE_COMPLETE = 1,
    QWEB_HTTP_PARSE_ERR_NULL = -1,
    QWEB_HTTP_PARSE_ERR_STATE = -2,
    QWEB_HTTP_PARSE_ERR_SYNTAX = -3,
    QWEB_HTTP_PARSE_ERR_VERSION = -4,
    QWEB_HTTP_PARSE_ERR_HOST = -5,
    QWEB_HTTP_PARSE_ERR_CONTENT_LENGTH = -6,
    QWEB_HTTP_PARSE_ERR_TRANSFER_ENCODING = -7,
    QWEB_HTTP_PARSE_ERR_HEADERS_TOO_LARGE = -8,
    QWEB_HTTP_PARSE_ERR_BODY_TOO_LARGE = -9,
    QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE = -10
} qweb_http_parse_status_t;

typedef struct qweb_http_request {
    char method[QWEB_HTTP_MAX_METHOD_BYTES + 1u];
    size_t method_length;
    char target[QWEB_HTTP_MAX_TARGET_BYTES + 1u];
    size_t target_length;
    char content_type[QWEB_HTTP_MAX_CONTENT_TYPE_BYTES + 1u];
    size_t content_type_length;
    uint8_t has_content_type;
    uint8_t connection_close_requested;
    size_t content_length;
    uint8_t body[QWEB_HTTP_MAX_BODY_BYTES];
    size_t body_length;
} qweb_http_request_t;

typedef struct qweb_http_parser {
    qweb_http_request_t request;
    char header_bytes[QWEB_HTTP_MAX_HEADER_BYTES + 1u];
    size_t header_length;
    size_t total_received;
    size_t error_offset;
    qweb_http_parse_status_t status;
} qweb_http_parser_t;

typedef struct qweb_http_feed_result {
    qweb_http_parse_status_t status;
    size_t consumed;
    size_t error_offset;
} qweb_http_feed_result_t;

void qweb_http_parser_init(qweb_http_parser_t *parser);

/*
 * Feed an arbitrary fragment. On completion, consumed names only the bytes
 * belonging to this request; any pipelined bytes remain with the caller.
 */
qweb_http_feed_result_t qweb_http_parser_feed(
    qweb_http_parser_t *parser,
    const uint8_t *bytes,
    size_t length);

const char *qweb_http_parse_status_name(qweb_http_parse_status_t status);

typedef enum qweb_http_content_type {
    QWEB_HTTP_CONTENT_JSON = 0,
    QWEB_HTTP_CONTENT_HTML,
    QWEB_HTTP_CONTENT_JAVASCRIPT,
    QWEB_HTTP_CONTENT_CSS,
    QWEB_HTTP_CONTENT_TEXT,
    QWEB_HTTP_CONTENT_OCTETS
} qweb_http_content_type_t;

typedef enum qweb_http_format_status {
    QWEB_HTTP_FORMAT_OK = 0,
    QWEB_HTTP_FORMAT_ERR_NULL = -1,
    QWEB_HTTP_FORMAT_ERR_STATUS = -2,
    QWEB_HTTP_FORMAT_ERR_CAPACITY = -3
} qweb_http_format_status_t;

const char *qweb_http_reason_phrase(uint16_t status_code);

/*
 * Format a complete close-delimited HTTP/1.1 response into caller storage.
 * body may contain arbitrary bytes and does not need to be NUL terminated.
 */
qweb_http_format_status_t qweb_http_format_response(
    uint16_t status_code,
    qweb_http_content_type_t content_type,
    const uint8_t *body,
    size_t body_length,
    uint8_t *output,
    size_t output_capacity,
    size_t *output_length);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_HTTP_H */
