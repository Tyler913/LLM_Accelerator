#ifndef QOT_PROTOCOL_H
#define QOT_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "qmap_one_token_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QOT_PROTOCOL_MAX_LINE_LENGTH 4096u
#define QOT_PROTOCOL_MAX_PROMPT_TOKENS QOT_MAX_CONTEXT

typedef enum qot_protocol_command_kind {
    QOT_PROTOCOL_COMMAND_INVALID = 0,
    QOT_PROTOCOL_COMMAND_HELP,
    QOT_PROTOCOL_COMMAND_PING,
    QOT_PROTOCOL_COMMAND_TOKENS,
    QOT_PROTOCOL_COMMAND_PROMPT
} qot_protocol_command_kind_t;

typedef enum qot_protocol_parse_status {
    QOT_PROTOCOL_PARSE_OK = 0,
    QOT_PROTOCOL_PARSE_ERR_NULL = -1,
    QOT_PROTOCOL_PARSE_ERR_EMPTY = -2,
    QOT_PROTOCOL_PARSE_ERR_TOO_LONG = -3,
    QOT_PROTOCOL_PARSE_ERR_UNKNOWN_COMMAND = -4,
    QOT_PROTOCOL_PARSE_ERR_SYNTAX = -5,
    QOT_PROTOCOL_PARSE_ERR_NUMBER = -6,
    QOT_PROTOCOL_PARSE_ERR_RANGE = -7,
    QOT_PROTOCOL_PARSE_ERR_COUNT = -8,
    QOT_PROTOCOL_PARSE_ERR_TRAILING = -9
} qot_protocol_parse_status_t;

typedef struct qot_protocol_command {
    qot_protocol_command_kind_t kind;
    uint32_t max_new_tokens;
    uint32_t token_count;
    uint32_t token_ids[QOT_PROTOCOL_MAX_PROMPT_TOKENS];
    size_t prompt_offset;
    size_t prompt_length;
} qot_protocol_command_t;

typedef struct qot_protocol_parse_result {
    qot_protocol_parse_status_t status;
    size_t error_offset;
} qot_protocol_parse_result_t;

qot_protocol_parse_result_t qot_protocol_parse(
    const char *line,
    size_t line_length,
    qot_protocol_command_t *command);

qot_protocol_parse_result_t qot_protocol_parse_cstr(
    const char *line,
    qot_protocol_command_t *command);

const char *qot_protocol_parse_status_name(qot_protocol_parse_status_t status);
const char *qot_protocol_help_text(void);

#ifdef __cplusplus
}
#endif

#endif /* QOT_PROTOCOL_H */
