#ifndef TEST_LWIP_NETIF_H
#define TEST_LWIP_NETIF_H

#include "lwip/ip4_addr.h"

struct netif {
    ip4_addr_t ip_addr;
};

#define netif_ip4_addr(network_interface) (&((network_interface)->ip_addr))

#endif /* TEST_LWIP_NETIF_H */
