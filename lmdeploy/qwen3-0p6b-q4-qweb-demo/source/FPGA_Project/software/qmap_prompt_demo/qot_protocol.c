#include "qot_protocol.h"

#include <limits.h>
#include <string.h>

typedef struct qot_protocol_cursor {
    const char *line;
    size_t length;
    size_t offset;
} qot_protocol_cursor_t;

static int qot_protocol_is_space(char value)
{
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

static void qot_protocol_skip_space(qot_protocol_cursor_t *cursor)
{
    while (cursor->offset < cursor->length &&
           qot_protocol_is_space(cursor->line[cursor->offset])) {
        ++cursor->offset;
    }
}

static int qot_protocol_word_equals(
    const qot_protocol_cursor_t *cursor,
    size_t start,
    size_t end,
    const char *expected)
{
    size_t expected_length = strlen(expected);
    size_t word_length = end - start;

    return word_length == expected_length &&
           memcmp(cursor->line + start, expected, expected_length) == 0;
}

static int qot_protocol_read_word(
    qot_protocol_cursor_t *cursor,
    size_t *start,
    size_t *end)
{
    qot_protocol_skip_space(cursor);
    if (cursor->offset >= cursor->length) return 0;
    *start = cursor->offset;
    while (cursor->offset < cursor->length &&
           !qot_protocol_is_space(cursor->line[cursor->offset])) {
        ++cursor->offset;
    }
    *end = cursor->offset;
    return 1;
}

static qot_protocol_parse_status_t qot_protocol_parse_u32_word(
    const qot_protocol_cursor_t *cursor,
    size_t start,
    size_t end,
    uint32_t *value)
{
    uint32_t parsed = 0u;
    size_t offset;

    if (start == end) return QOT_PROTOCOL_PARSE_ERR_NUMBER;
    for (offset = start; offset < end; ++offset) {
        uint32_t digit;
        char current = cursor->line[offset];

        if (current < '0' || current > '9') {
            return QOT_PROTOCOL_PARSE_ERR_NUMBER;
        }
        digit = (uint32_t)(current - '0');
        if (parsed > (UINT32_MAX - digit) / 10u) {
            return QOT_PROTOCOL_PARSE_ERR_NUMBER;
        }
        parsed = parsed * 10u + digit;
    }
    *value = parsed;
    return QOT_PROTOCOL_PARSE_OK;
}

static qot_protocol_parse_result_t qot_protocol_result(
    qot_protocol_parse_status_t status,
    size_t error_offset)
{
    qot_protocol_parse_result_t result;
    result.status = status;
    result.error_offset = error_offset;
    return result;
}

qot_protocol_parse_result_t qot_protocol_parse(
    const char *line,
    size_t line_length,
    qot_protocol_command_t *command)
{
    qot_protocol_command_t parsed_command;
    qot_protocol_cursor_t cursor;
    qot_protocol_parse_status_t number_status;
    size_t start;
    size_t end;
    size_t offset;
    uint32_t token_index;

    if (command != NULL) memset(command, 0, sizeof(*command));
    if (line == NULL || command == NULL) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_NULL, 0u);
    }
    if (line_length > QOT_PROTOCOL_MAX_LINE_LENGTH) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_TOO_LONG,
                                   QOT_PROTOCOL_MAX_LINE_LENGTH);
    }
    for (offset = 0u; offset < line_length; ++offset) {
        unsigned char current = (unsigned char)line[offset];
        if (current == 0u ||
            (current < 0x20u && !qot_protocol_is_space((char)current)) ||
            current == 0x7Fu) {
            return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_SYNTAX, offset);
        }
    }

    memset(&parsed_command, 0, sizeof(parsed_command));
    cursor.line = line;
    cursor.length = line_length;
    cursor.offset = 0u;

    if (!qot_protocol_read_word(&cursor, &start, &end)) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_EMPTY, cursor.offset);
    }

    if (qot_protocol_word_equals(&cursor, start, end, "HELP")) {
        parsed_command.kind = QOT_PROTOCOL_COMMAND_HELP;
    } else if (qot_protocol_word_equals(&cursor, start, end, "PING")) {
        parsed_command.kind = QOT_PROTOCOL_COMMAND_PING;
    } else if (qot_protocol_word_equals(&cursor, start, end, "TOKENS")) {
        parsed_command.kind = QOT_PROTOCOL_COMMAND_TOKENS;
    } else if (qot_protocol_word_equals(&cursor, start, end, "PROMPT")) {
        parsed_command.kind = QOT_PROTOCOL_COMMAND_PROMPT;
    } else {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_UNKNOWN_COMMAND,
                                   start);
    }

    if (parsed_command.kind == QOT_PROTOCOL_COMMAND_HELP ||
        parsed_command.kind == QOT_PROTOCOL_COMMAND_PING) {
        qot_protocol_skip_space(&cursor);
        if (cursor.offset != cursor.length) {
            return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_TRAILING,
                                       cursor.offset);
        }
        *command = parsed_command;
        return qot_protocol_result(QOT_PROTOCOL_PARSE_OK, cursor.offset);
    }

    if (!qot_protocol_read_word(&cursor, &start, &end)) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                                   cursor.offset);
    }
    number_status = qot_protocol_parse_u32_word(
        &cursor, start, end, &parsed_command.max_new_tokens);
    if (number_status != QOT_PROTOCOL_PARSE_OK) {
        return qot_protocol_result(number_status, start);
    }
    if (parsed_command.max_new_tokens == 0u ||
        parsed_command.max_new_tokens > QOT_MAX_CONTEXT) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_RANGE, start);
    }

    if (parsed_command.kind == QOT_PROTOCOL_COMMAND_PROMPT) {
        size_t prompt_end = cursor.length;

        while (cursor.offset < cursor.length &&
               (cursor.line[cursor.offset] == ' ' ||
                cursor.line[cursor.offset] == '\t')) {
            ++cursor.offset;
        }
        while (prompt_end > cursor.offset &&
               (cursor.line[prompt_end - 1u] == '\r' ||
                cursor.line[prompt_end - 1u] == '\n')) {
            --prompt_end;
        }
        if (cursor.offset >= prompt_end) {
            return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                                       cursor.offset);
        }
        for (offset = cursor.offset; offset < prompt_end; ++offset) {
            if (cursor.line[offset] == '\r' || cursor.line[offset] == '\n') {
                return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                                           offset);
            }
        }
        parsed_command.prompt_offset = cursor.offset;
        parsed_command.prompt_length = prompt_end - cursor.offset;
        *command = parsed_command;
        return qot_protocol_result(QOT_PROTOCOL_PARSE_OK, prompt_end);
    }

    if (!qot_protocol_read_word(&cursor, &start, &end)) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                                   cursor.offset);
    }
    number_status = qot_protocol_parse_u32_word(
        &cursor, start, end, &parsed_command.token_count);
    if (number_status != QOT_PROTOCOL_PARSE_OK) {
        return qot_protocol_result(number_status, start);
    }
    if (parsed_command.token_count == 0u ||
        parsed_command.token_count > QOT_PROTOCOL_MAX_PROMPT_TOKENS) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_RANGE, start);
    }

    for (token_index = 0u;
         token_index < parsed_command.token_count;
         ++token_index) {
        if (!qot_protocol_read_word(&cursor, &start, &end)) {
            return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_COUNT,
                                       cursor.offset);
        }
        number_status = qot_protocol_parse_u32_word(
            &cursor, start, end, &parsed_command.token_ids[token_index]);
        if (number_status != QOT_PROTOCOL_PARSE_OK) {
            return qot_protocol_result(number_status, start);
        }
        if (parsed_command.token_ids[token_index] >= QOT_VOCAB_SIZE) {
            return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_RANGE, start);
        }
    }

    qot_protocol_skip_space(&cursor);
    if (cursor.offset != cursor.length) {
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_COUNT,
                                   cursor.offset);
    }

    *command = parsed_command;
    return qot_protocol_result(QOT_PROTOCOL_PARSE_OK, cursor.offset);
}

qot_protocol_parse_result_t qot_protocol_parse_cstr(
    const char *line,
    qot_protocol_command_t *command)
{
    if (line == NULL) {
        if (command != NULL) memset(command, 0, sizeof(*command));
        return qot_protocol_result(QOT_PROTOCOL_PARSE_ERR_NULL, 0u);
    }
    return qot_protocol_parse(line, strlen(line), command);
}

const char *qot_protocol_parse_status_name(qot_protocol_parse_status_t status)
{
    switch (status) {
    case QOT_PROTOCOL_PARSE_OK: return "OK";
    case QOT_PROTOCOL_PARSE_ERR_NULL: return "NULL";
    case QOT_PROTOCOL_PARSE_ERR_EMPTY: return "EMPTY";
    case QOT_PROTOCOL_PARSE_ERR_TOO_LONG: return "TOO_LONG";
    case QOT_PROTOCOL_PARSE_ERR_UNKNOWN_COMMAND: return "UNKNOWN_COMMAND";
    case QOT_PROTOCOL_PARSE_ERR_SYNTAX: return "SYNTAX";
    case QOT_PROTOCOL_PARSE_ERR_NUMBER: return "NUMBER";
    case QOT_PROTOCOL_PARSE_ERR_RANGE: return "RANGE";
    case QOT_PROTOCOL_PARSE_ERR_COUNT: return "COUNT";
    case QOT_PROTOCOL_PARSE_ERR_TRAILING: return "TRAILING";
    default: return "UNKNOWN";
    }
}

const char *qot_protocol_help_text(void)
{
    return
        "HELP\n"
        "PING\n"
        "TOKENS <max_new> <count> <token_id_0> ... <token_id_n>\n"
        "PROMPT <max_new> <UTF-8 text>\n"
        "Limits: max_new=1..256, count=1..256, token_id=0..151935\n";
}
