#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qot_protocol.h"

static int expect(int condition, const char *message)
{
    if (condition) return 0;
    printf("FAIL: %s\n", message);
    return 1;
}

static int expect_status(
    const char *line,
    qot_protocol_parse_status_t expected,
    const char *message)
{
    qot_protocol_command_t command;
    qot_protocol_parse_result_t result =
        qot_protocol_parse_cstr(line, &command);

    if (result.status == expected) return 0;
    printf("FAIL: %s expected=%s actual=%s offset=%u\n",
           message,
           qot_protocol_parse_status_name(expected),
           qot_protocol_parse_status_name(result.status),
           (unsigned)result.error_offset);
    return 1;
}

static int test_valid_commands(void)
{
    qot_protocol_command_t command;
    qot_protocol_parse_result_t result;

    result = qot_protocol_parse_cstr("PING", &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_OK &&
               command.kind == QOT_PROTOCOL_COMMAND_PING,
               "PING command")) return 1;

    result = qot_protocol_parse_cstr(" \tHELP\r\n", &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_OK &&
               command.kind == QOT_PROTOCOL_COMMAND_HELP,
               "HELP command with surrounding whitespace")) return 1;

    result = qot_protocol_parse_cstr("TOKENS 3 2 374 28458", &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_OK &&
               command.kind == QOT_PROTOCOL_COMMAND_TOKENS &&
               command.max_new_tokens == 3u &&
               command.token_count == 2u &&
               command.token_ids[0] == 374u &&
               command.token_ids[1] == 28458u,
               "TOKENS command fields")) return 1;

    result = qot_protocol_parse_cstr("PROMPT 4 The future of FPGA is  ",
                                     &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_OK &&
               command.kind == QOT_PROTOCOL_COMMAND_PROMPT &&
               command.max_new_tokens == 4u &&
               command.prompt_length == strlen("The future of FPGA is  ") &&
               strncmp("PROMPT 4 The future of FPGA is  " +
                           command.prompt_offset,
                       "The future of FPGA is  ",
                       command.prompt_length) == 0,
               "PROMPT command preserves UTF-8 payload bytes")) return 1;

    result = qot_protocol_parse_cstr(
        "\tTOKENS\t256\t3\t0\t151643\t151935  ", &command);
    return expect(result.status == QOT_PROTOCOL_PARSE_OK &&
                  command.max_new_tokens == QOT_MAX_CONTEXT &&
                  command.token_count == 3u &&
                  command.token_ids[2] == QOT_VOCAB_SIZE - 1u,
                  "TOKENS accepts inclusive upper limits");
}

static int test_command_errors(void)
{
    if (expect_status("", QOT_PROTOCOL_PARSE_ERR_EMPTY,
                      "empty command")) return 1;
    if (expect_status("ping", QOT_PROTOCOL_PARSE_ERR_UNKNOWN_COMMAND,
                      "commands are case-sensitive")) return 1;
    if (expect_status("PING extra", QOT_PROTOCOL_PARSE_ERR_TRAILING,
                      "PING rejects arguments")) return 1;
    if (expect_status("HELP 1", QOT_PROTOCOL_PARSE_ERR_TRAILING,
                      "HELP rejects arguments")) return 1;
    if (expect_status("TOKENS", QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                      "TOKENS requires max_new")) return 1;
    if (expect_status("PROMPT", QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                      "PROMPT requires max_new")) return 1;
    if (expect_status("PROMPT 0 hello", QOT_PROTOCOL_PARSE_ERR_RANGE,
                      "PROMPT rejects zero max_new")) return 1;
    if (expect_status("PROMPT 1   ", QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                      "PROMPT rejects empty text")) return 1;
    if (expect_status("PROMPT 1 hello\nworld",
                      QOT_PROTOCOL_PARSE_ERR_SYNTAX,
                      "PROMPT rejects embedded line breaks")) return 1;
    if (expect_status("TOKENS -1 1 0", QOT_PROTOCOL_PARSE_ERR_NUMBER,
                      "negative max_new")) return 1;
    if (expect_status("TOKENS 4294967296 1 0",
                      QOT_PROTOCOL_PARSE_ERR_NUMBER,
                      "uint32 overflow")) return 1;
    if (expect_status("TOKENS 0 1 0", QOT_PROTOCOL_PARSE_ERR_RANGE,
                      "zero max_new")) return 1;
    if (expect_status("TOKENS 257 1 0", QOT_PROTOCOL_PARSE_ERR_RANGE,
                      "max_new above context")) return 1;
    if (expect_status("TOKENS 1 0", QOT_PROTOCOL_PARSE_ERR_RANGE,
                      "zero prompt count")) return 1;
    if (expect_status("TOKENS 1 257 0", QOT_PROTOCOL_PARSE_ERR_RANGE,
                      "prompt count above context")) return 1;
    if (expect_status("TOKENS 1 2 10", QOT_PROTOCOL_PARSE_ERR_COUNT,
                      "missing prompt token")) return 1;
    if (expect_status("TOKENS 1 1 10 11", QOT_PROTOCOL_PARSE_ERR_COUNT,
                      "extra prompt token")) return 1;
    if (expect_status("TOKENS 1 1 151936", QOT_PROTOCOL_PARSE_ERR_RANGE,
                      "out-of-vocabulary prompt token")) return 1;
    if (expect_status("TOKENS 1 1 1x", QOT_PROTOCOL_PARSE_ERR_NUMBER,
                      "numeric suffix")) return 1;
    return 0;
}

static int test_explicit_bounds(void)
{
    static const char embedded_nul[] = {'P', 'I', '\0', 'G'};
    qot_protocol_command_t command;
    qot_protocol_parse_result_t result;

    memset(&command, 0xA5, sizeof(command));
    result = qot_protocol_parse(NULL, 0u, &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_ERR_NULL &&
               command.kind == QOT_PROTOCOL_COMMAND_INVALID,
               "NULL input clears command")) return 1;

    result = qot_protocol_parse(embedded_nul, sizeof(embedded_nul), &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_ERR_SYNTAX &&
               result.error_offset == 2u,
               "embedded NUL is rejected")) return 1;

    result = qot_protocol_parse("X",
                                QOT_PROTOCOL_MAX_LINE_LENGTH + 1u,
                                &command);
    if (expect(result.status == QOT_PROTOCOL_PARSE_ERR_TOO_LONG,
               "length bound is checked before input access")) return 1;

    result = qot_protocol_parse_cstr("TOKENS 1 2 1", &command);
    return expect(result.status == QOT_PROTOCOL_PARSE_ERR_COUNT &&
                  command.kind == QOT_PROTOCOL_COMMAND_INVALID,
                  "failed parse never publishes a partial command");
}

int main(void)
{
    if (test_valid_commands() != 0) return 1;
    if (test_command_errors() != 0) return 1;
    if (test_explicit_bounds() != 0) return 1;
    printf("PASS: qot_protocol host tests\n");
    return 0;
}
