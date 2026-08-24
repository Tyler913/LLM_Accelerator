#include "qot_uart.h"

#include <stdio.h>
#include <string.h>

#if defined(__has_include)
#  if __has_include("xil_printf.h")
#    include "xil_printf.h"
#    define QOT_UART_HAS_XIL_BYTE_IO 1
#  endif
#endif

static int qot_uart_stdio_getc(void *context)
{
    (void)context;
#if defined(QOT_UART_HAS_XIL_BYTE_IO)
    return (int)(unsigned char)inbyte();
#else
    return getchar();
#endif
}

static int qot_uart_stdio_putc(void *context, unsigned char value)
{
    (void)context;
#if defined(QOT_UART_HAS_XIL_BYTE_IO)
    outbyte((char)value);
    return 0;
#else
    return putchar((int)value) == EOF ? -1 : 0;
#endif
}

static int qot_uart_echo_byte(qot_uart_t *uart, unsigned char value)
{
    if (uart->echo == 0u || uart->putc_fn == NULL) return 0;
    return uart->putc_fn(uart->io_context, value);
}

void qot_uart_init(
    qot_uart_t *uart,
    qot_uart_getc_fn getc_fn,
    qot_uart_putc_fn putc_fn,
    void *io_context,
    uint32_t echo)
{
    if (uart == NULL) return;
    uart->getc_fn = getc_fn;
    uart->putc_fn = putc_fn;
    uart->io_context = io_context;
    uart->echo = echo != 0u ? 1u : 0u;
    uart->swallow_next_lf = 0u;
}

void qot_uart_init_stdio(qot_uart_t *uart, uint32_t echo)
{
    qot_uart_init(uart, qot_uart_stdio_getc, qot_uart_stdio_putc, NULL, echo);
}

qot_uart_read_status_t qot_uart_read_line(
    qot_uart_t *uart,
    char *buffer,
    size_t capacity,
    size_t *line_length)
{
    size_t used = 0u;
    int overflow = 0;

    if (line_length != NULL) *line_length = 0u;
    if (uart == NULL || buffer == NULL || capacity == 0u ||
        line_length == NULL || uart->getc_fn == NULL) {
        return QOT_UART_READ_BAD_ARGUMENT;
    }
    buffer[0] = '\0';

    for (;;) {
        int input = uart->getc_fn(uart->io_context);
        unsigned char value;

        if (input < 0) {
            if (used == 0u) return QOT_UART_READ_EOF;
            break;
        }
        value = (unsigned char)input;

        if (uart->swallow_next_lf != 0u) {
            uart->swallow_next_lf = 0u;
            if (value == (unsigned char)'\n') continue;
        }

        if (value == (unsigned char)'\r' || value == (unsigned char)'\n') {
            if (value == (unsigned char)'\r') uart->swallow_next_lf = 1u;
            if (qot_uart_echo_byte(uart, (unsigned char)'\r') != 0 ||
                qot_uart_echo_byte(uart, (unsigned char)'\n') != 0) {
                return QOT_UART_READ_IO_ERROR;
            }
            break;
        }

        if (value == 0x08u || value == 0x7Fu) {
            if (!overflow && used != 0u) {
                --used;
                if (uart->echo != 0u &&
                    (qot_uart_echo_byte(uart, 0x08u) != 0 ||
                     qot_uart_echo_byte(uart, (unsigned char)' ') != 0 ||
                     qot_uart_echo_byte(uart, 0x08u) != 0)) {
                    return QOT_UART_READ_IO_ERROR;
                }
            }
            continue;
        }

        if (value < 0x20u && value != (unsigned char)'\t') continue;
        if (overflow) continue;
        if (used + 1u >= capacity) {
            overflow = 1;
            if (qot_uart_echo_byte(uart, 0x07u) != 0) {
                return QOT_UART_READ_IO_ERROR;
            }
            continue;
        }

        buffer[used++] = (char)value;
        if (qot_uart_echo_byte(uart, value) != 0) {
            return QOT_UART_READ_IO_ERROR;
        }
    }

    buffer[used] = '\0';
    *line_length = used;
    return overflow ? QOT_UART_READ_OVERFLOW : QOT_UART_READ_OK;
}

int qot_uart_write(qot_uart_t *uart, const char *bytes, size_t length)
{
    size_t offset;

    if (uart == NULL || bytes == NULL || uart->putc_fn == NULL) return -1;
    for (offset = 0u; offset < length; ++offset) {
        if (uart->putc_fn(uart->io_context,
                          (unsigned char)bytes[offset]) != 0) {
            return -1;
        }
    }
    return 0;
}

int qot_uart_puts(qot_uart_t *uart, const char *text)
{
    if (text == NULL) return -1;
    return qot_uart_write(uart, text, strlen(text));
}
