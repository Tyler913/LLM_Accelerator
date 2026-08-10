#ifndef QOT_UART_H
#define QOT_UART_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*qot_uart_getc_fn)(void *context);
typedef int (*qot_uart_putc_fn)(void *context, unsigned char value);

typedef enum qot_uart_read_status {
    QOT_UART_READ_OK = 0,
    QOT_UART_READ_EOF = 1,
    QOT_UART_READ_OVERFLOW = -1,
    QOT_UART_READ_IO_ERROR = -2,
    QOT_UART_READ_BAD_ARGUMENT = -3
} qot_uart_read_status_t;

typedef struct qot_uart {
    qot_uart_getc_fn getc_fn;
    qot_uart_putc_fn putc_fn;
    void *io_context;
    uint32_t echo;
    uint32_t swallow_next_lf;
} qot_uart_t;

void qot_uart_init(
    qot_uart_t *uart,
    qot_uart_getc_fn getc_fn,
    qot_uart_putc_fn putc_fn,
    void *io_context,
    uint32_t echo);

void qot_uart_init_stdio(qot_uart_t *uart, uint32_t echo);

/* capacity includes the trailing NUL byte. An overflowing line is discarded. */
qot_uart_read_status_t qot_uart_read_line(
    qot_uart_t *uart,
    char *buffer,
    size_t capacity,
    size_t *line_length);

int qot_uart_write(qot_uart_t *uart, const char *bytes, size_t length);
int qot_uart_puts(qot_uart_t *uart, const char *text);

#ifdef __cplusplus
}
#endif

#endif /* QOT_UART_H */
