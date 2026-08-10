#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qot_uart.h"

typedef struct mock_uart_io {
    const unsigned char *input;
    size_t input_size;
    size_t input_offset;
    unsigned char output[256];
    size_t output_size;
    size_t fail_output_at;
} mock_uart_io_t;

static int mock_getc(void *context)
{
    mock_uart_io_t *io = (mock_uart_io_t *)context;
    if (io->input_offset >= io->input_size) return -1;
    return (int)io->input[io->input_offset++];
}

static int mock_putc(void *context, unsigned char value)
{
    mock_uart_io_t *io = (mock_uart_io_t *)context;
    if (io->output_size == io->fail_output_at ||
        io->output_size >= sizeof(io->output)) {
        return -1;
    }
    io->output[io->output_size++] = value;
    return 0;
}

static int expect(int condition, const char *message)
{
    if (condition) return 0;
    printf("FAIL: %s\n", message);
    return 1;
}

static void init_io(mock_uart_io_t *io, const unsigned char *input,
                    size_t input_size)
{
    memset(io, 0, sizeof(*io));
    io->input = input;
    io->input_size = input_size;
    io->fail_output_at = (size_t)-1;
}

static int test_crlf_and_consecutive_lines(void)
{
    static const unsigned char input[] = "abc\r\nnext\n";
    mock_uart_io_t io;
    qot_uart_t uart;
    char line[16];
    size_t length;
    qot_uart_read_status_t status;

    init_io(&io, input, sizeof(input) - 1u);
    qot_uart_init(&uart, mock_getc, mock_putc, &io, 1u);
    status = qot_uart_read_line(&uart, line, sizeof(line), &length);
    if (expect(status == QOT_UART_READ_OK && length == 3u &&
               strcmp(line, "abc") == 0,
               "CR terminates first line")) return 1;
    status = qot_uart_read_line(&uart, line, sizeof(line), &length);
    if (expect(status == QOT_UART_READ_OK && length == 4u &&
               strcmp(line, "next") == 0,
               "LF after CR is swallowed exactly once")) return 1;
    return expect(io.output_size == 11u,
                  "echo contains characters and normalized CRLF endings");
}

static int test_editing_and_partial_eof(void)
{
    static const unsigned char input[] = {'a', 'b', 0x08u, 'c', 'd'};
    mock_uart_io_t io;
    qot_uart_t uart;
    char line[16];
    size_t length;
    qot_uart_read_status_t status;

    init_io(&io, input, sizeof(input));
    qot_uart_init(&uart, mock_getc, mock_putc, &io, 0u);
    status = qot_uart_read_line(&uart, line, sizeof(line), &length);
    if (expect(status == QOT_UART_READ_OK && length == 3u &&
               strcmp(line, "acd") == 0,
               "backspace edits before partial EOF")) return 1;
    status = qot_uart_read_line(&uart, line, sizeof(line), &length);
    return expect(status == QOT_UART_READ_EOF && length == 0u,
                  "clean EOF after partial line");
}

static int test_overflow_drains_complete_line(void)
{
    static const unsigned char input[] = "abcdef\nok\n";
    mock_uart_io_t io;
    qot_uart_t uart;
    char line[4];
    size_t length;
    qot_uart_read_status_t status;

    init_io(&io, input, sizeof(input) - 1u);
    qot_uart_init(&uart, mock_getc, mock_putc, &io, 0u);
    status = qot_uart_read_line(&uart, line, sizeof(line), &length);
    if (expect(status == QOT_UART_READ_OVERFLOW && length == 3u &&
               strcmp(line, "abc") == 0,
               "overflow reports the bounded prefix")) return 1;
    status = qot_uart_read_line(&uart, line, sizeof(line), &length);
    return expect(status == QOT_UART_READ_OK && length == 2u &&
                  strcmp(line, "ok") == 0,
                  "overflow drains through newline before next command");
}

static int test_output_error(void)
{
    static const unsigned char input[] = "x\n";
    mock_uart_io_t io;
    qot_uart_t uart;
    char line[4];
    size_t length;

    init_io(&io, input, sizeof(input) - 1u);
    io.fail_output_at = 0u;
    qot_uart_init(&uart, mock_getc, mock_putc, &io, 1u);
    return expect(qot_uart_read_line(&uart, line, sizeof(line), &length) ==
                  QOT_UART_READ_IO_ERROR,
                  "echo output failure is propagated");
}

int main(void)
{
    if (test_crlf_and_consecutive_lines() != 0) return 1;
    if (test_editing_and_partial_eof() != 0) return 1;
    if (test_overflow_drains_complete_line() != 0) return 1;
    if (test_output_error() != 0) return 1;
    printf("PASS: qot_uart host tests\n");
    return 0;
}
