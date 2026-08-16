#ifndef TEST_LWIP_IP4_ADDR_H
#define TEST_LWIP_IP4_ADDR_H

#include <stdint.h>

typedef struct ip4_addr {
    uint32_t addr;
} ip4_addr_t;

#define IP4ADDR_STRLEN_MAX 16
#define ip4_addr_get_byte(ipaddr, index) \
    (((const uint8_t *)&((ipaddr)->addr))[index])
#define ip4_addr1(ipaddr) ip4_addr_get_byte((ipaddr), 0)
#define ip4_addr2(ipaddr) ip4_addr_get_byte((ipaddr), 1)
#define ip4_addr3(ipaddr) ip4_addr_get_byte((ipaddr), 2)
#define ip4_addr4(ipaddr) ip4_addr_get_byte((ipaddr), 3)

#endif /* TEST_LWIP_IP4_ADDR_H */
