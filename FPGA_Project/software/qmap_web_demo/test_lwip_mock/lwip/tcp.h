#ifndef TEST_LWIP_TCP_H
#define TEST_LWIP_TCP_H

#include "lwip/err.h"
#include "lwip/pbuf.h"

struct tcp_pcb;

typedef err_t (*tcp_accept_fn)(void *, struct tcp_pcb *, err_t);
typedef err_t (*tcp_recv_fn)(void *, struct tcp_pcb *, struct pbuf *, err_t);
typedef err_t (*tcp_sent_fn)(void *, struct tcp_pcb *, u16_t);
typedef err_t (*tcp_poll_fn)(void *, struct tcp_pcb *);
typedef void (*tcp_err_fn)(void *, err_t);

struct tcp_pcb {
    void *callback_argument;
    tcp_accept_fn accept_callback;
    tcp_recv_fn recv_callback;
    tcp_sent_fn sent_callback;
    tcp_poll_fn poll_callback;
    tcp_err_fn error_callback;
    u16_t send_buffer;
    u16_t bound_port;
    u8_t poll_interval;
    u8_t closed;
    u8_t aborted;
    u8_t listening;
    u8_t prio;
};

#define IPADDR_TYPE_ANY 0u
#define IP_ANY_TYPE ((const void *)0)
#define TCP_WRITE_FLAG_COPY 1u
#define TCP_PRIO_MIN 1u
#define TCP_PRIO_NORMAL 64u

struct tcp_pcb *tcp_new_ip_type(u8_t type);
void tcp_arg(struct tcp_pcb *pcb, void *argument);
void tcp_recv(struct tcp_pcb *pcb, tcp_recv_fn callback);
void tcp_sent(struct tcp_pcb *pcb, tcp_sent_fn callback);
void tcp_err(struct tcp_pcb *pcb, tcp_err_fn callback);
void tcp_accept(struct tcp_pcb *pcb, tcp_accept_fn callback);
void tcp_poll(struct tcp_pcb *pcb, tcp_poll_fn callback, u8_t interval);
void tcp_setprio(struct tcp_pcb *pcb, u8_t priority);
u16_t test_tcp_sndbuf(struct tcp_pcb *pcb);
#define tcp_sndbuf(pcb) test_tcp_sndbuf(pcb)
void tcp_recved(struct tcp_pcb *pcb, u16_t length);
err_t tcp_bind(struct tcp_pcb *pcb, const void *address, u16_t port);
struct tcp_pcb *tcp_listen_with_backlog_and_err(
    struct tcp_pcb *pcb,
    u8_t backlog,
    err_t *error);
void tcp_abort(struct tcp_pcb *pcb);
err_t tcp_close(struct tcp_pcb *pcb);
err_t tcp_write(
    struct tcp_pcb *pcb,
    const void *bytes,
    u16_t length,
    u8_t flags);
err_t tcp_output(struct tcp_pcb *pcb);

#endif /* TEST_LWIP_TCP_H */
