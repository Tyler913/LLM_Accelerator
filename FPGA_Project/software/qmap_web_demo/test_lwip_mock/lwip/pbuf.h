#ifndef TEST_LWIP_PBUF_H
#define TEST_LWIP_PBUF_H

#include "lwip/err.h"

struct pbuf {
    struct pbuf *next;
    void *payload;
    u16_t tot_len;
    u16_t len;
};

u8_t pbuf_free(struct pbuf *packet);

#endif /* TEST_LWIP_PBUF_H */
