#ifndef QWEB_API_H
#define QWEB_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These limits mirror the current full28 model/runtime boundary. */
#define QWEB_API_MAX_PROMPT_BYTES 4096u
#define QWEB_API_MAX_TOKENS 256u
#define QWEB_API_MODEL_VOCAB_SIZE 151936u

typedef enum qweb_generate_input_kind {
    QWEB_GENERATE_INPUT_INVALID = 0,
    QWEB_GENERATE_INPUT_PROMPT,
    QWEB_GENERATE_INPUT_TOKENS
} qweb_generate_input_kind_t;

typedef struct qweb_generate_request {
    qweb_generate_input_kind_t kind;
    uint32_t max_new_tokens;
    size_t prompt_length;
    uint8_t prompt[QWEB_API_MAX_PROMPT_BYTES];
    uint32_t token_count;
    uint32_t token_ids[QWEB_API_MAX_TOKENS];
} qweb_generate_request_t;

typedef enum qweb_api_parse_status {
    QWEB_API_PARSE_OK = 0,
    QWEB_API_PARSE_ERR_NULL = -1,
    QWEB_API_PARSE_ERR_EMPTY = -2,
    QWEB_API_PARSE_ERR_SYNTAX = -3,
    QWEB_API_PARSE_ERR_UTF8 = -4,
    QWEB_API_PARSE_ERR_ESCAPE = -5,
    QWEB_API_PARSE_ERR_NUMBER = -6,
    QWEB_API_PARSE_ERR_UNKNOWN_MEMBER = -7,
    QWEB_API_PARSE_ERR_DUPLICATE_MEMBER = -8,
    QWEB_API_PARSE_ERR_MISSING_MEMBER = -9,
    QWEB_API_PARSE_ERR_INPUT_CHOICE = -10,
    QWEB_API_PARSE_ERR_RANGE = -11,
    QWEB_API_PARSE_ERR_CAPACITY = -12,
    QWEB_API_PARSE_ERR_TRAILING = -13
} qweb_api_parse_status_t;

typedef struct qweb_api_parse_result {
    qweb_api_parse_status_t status;
    size_t error_offset;
} qweb_api_parse_result_t;

/*
 * Parse exactly one JSON object with max_new_tokens and exactly one of prompt
 * or tokens. JSON strings are decoded to strict UTF-8. The prompt is explicitly
 * length-delimited and may contain decoded NUL/control bytes. Parsing writes
 * directly into caller-owned storage to avoid a multi-kilobyte bare-metal
 * stack temporary. On failure, kind remains QWEB_GENERATE_INPUT_INVALID and
 * all other partially decoded fields must be ignored.
 */
qweb_api_parse_result_t qweb_api_parse_generate_request(
    const uint8_t *json,
    size_t length,
    qweb_generate_request_t *request);

const char *qweb_api_parse_status_name(qweb_api_parse_status_t status);

typedef enum qweb_json_write_status {
    QWEB_JSON_WRITE_OK = 0,
    QWEB_JSON_WRITE_ERR_NULL = -1,
    QWEB_JSON_WRITE_ERR_CAPACITY = -2,
    QWEB_JSON_WRITE_ERR_UTF8 = -3
} qweb_json_write_status_t;

typedef struct qweb_json_writer {
    uint8_t *buffer;
    size_t capacity;
    size_t length;
    qweb_json_write_status_t status;
} qweb_json_writer_t;

void qweb_json_writer_init(
    qweb_json_writer_t *writer,
    uint8_t *buffer,
    size_t capacity);
qweb_json_write_status_t qweb_json_writer_append_bytes(
    qweb_json_writer_t *writer,
    const uint8_t *bytes,
    size_t length);
qweb_json_write_status_t qweb_json_writer_append_cstr(
    qweb_json_writer_t *writer,
    const char *text);
qweb_json_write_status_t qweb_json_writer_append_u32(
    qweb_json_writer_t *writer,
    uint32_t value);
qweb_json_write_status_t qweb_json_writer_append_i64(
    qweb_json_writer_t *writer,
    int64_t value);
qweb_json_write_status_t qweb_json_writer_append_string(
    qweb_json_writer_t *writer,
    const uint8_t *utf8,
    size_t length);

/* Convenience body used by all bounded HTTP/API failures. */
qweb_json_write_status_t qweb_api_format_error_json(
    uint8_t *output,
    size_t output_capacity,
    const char *code,
    const char *message,
    size_t *output_length);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_API_H */
